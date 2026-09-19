#include "models/qwen3_5/load.h"
#include "models/qwen3_5/execution/gdn.h"
#include "models/qwen3_5/execution/text.h"
#include "models/qwen3_5/frontend/frontend.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"
#include "core/linear_attention_state.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/scalar.h"
#include "ninfer/engine.h"
#include "serve/openai_chat.h"
#include "serve/translate.h"
#include "models/qwen3_5/program/prefix_identity.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace ninfer;
namespace family   = ninfer::models::qwen3_5;
namespace schedule = family::execution;

std::vector<float> read_tensor(const Tensor& tensor) {
    std::vector<float> values(tensor.numel());
    if (tensor.dtype == DType::BF16) {
        std::vector<std::uint16_t> words(tensor.numel());
        CUDA_CHECK(cudaMemcpy(words.data(), tensor.data, tensor.bytes(), cudaMemcpyDeviceToHost));
        for (std::size_t i = 0; i < values.size(); ++i) {
            values[i] = std::bit_cast<float>(static_cast<std::uint32_t>(words[i]) << 16);
        }
    } else {
        CUDA_CHECK(cudaMemcpy(values.data(), tensor.data, tensor.bytes(), cudaMemcpyDeviceToHost));
    }
    return values;
}

std::size_t compare(const char* label, const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) { throw std::runtime_error("comparison shape mismatch"); }
    double error = 0, norm = 0, maximum = 0;
    std::size_t different = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) {
            throw std::runtime_error("non-finite prefill output");
        }
        const double delta = static_cast<double>(a[i]) - b[i];
        error += delta * delta;
        norm += static_cast<double>(a[i]) * a[i];
        maximum = std::max(maximum, std::abs(delta));
        different += std::bit_cast<std::uint32_t>(a[i]) != std::bit_cast<std::uint32_t>(b[i]);
    }
    const double relative = std::sqrt(error / std::max(norm, 1e-30));
    std::cout << label << " different=" << different << '/' << a.size() << " rel_l2=" << relative
              << " max_abs=" << maximum << '\n';
    return different;
}

PromptInput input_prompt() {
    PromptInput input;
    input.options.enable_thinking = true;
    ChatMessage message;
    message.role = ChatRole::User;
    message.parts.push_back(
        {.kind = MessagePartKind::Text,
         .text = "\nSolve the following math problem step by step. Put your answer inside "
                 "\\boxed{}.\n\n"
                 "Six points $A, B, C, D, E,$ and $F$ lie in a straight line in that order. "
                 "Suppose that $G$ is a point not on the line and that $AC=26, BD=22, CE=31, "
                 "DF=33, AF=73, CG=40,$ and $DG=30.$ Find the area of $\\triangle BGE.$\n\n"
                 "Remember to put your answer inside \\boxed{}."});
    input.messages.push_back(std::move(message));
    input.context_cache.markers.push_back(
        {.after_message_count      = 1,
         .location                 = PromptCacheMarkerLocation::MessagePartBoundary,
         .after_message_part_count = 1});
    return input;
}

PromptInput tool_prompt(bool next_turn) {
    auto body = serve::RequestJson::parse(R"({
        "model":"qwen3.8-27b", "enable_thinking":true,
        "messages":[{"role":"system","content":"You are a coding assistant."},
                    {"role":"user","content":"Inspect README.md using read_file."}],
        "tools":[{"type":"function","function":{"name":"read_file",
                  "parameters":{"type":"object","properties":{"path":{"type":"string"}}}}}]
    })");
    if (next_turn) {
        body["messages"].push_back(serve::RequestJson::parse(R"({
            "role":"assistant", "content":"", "reasoning_content":"I will inspect the file.",
            "tool_calls":[{"id":"call_1","type":"function","function":
                          {"name":"read_file","arguments":"{\"path\":\"README.md\"}"}}]
        })"));
        body["messages"].push_back(serve::RequestJson::parse(R"({
            "role":"tool", "tool_call_id":"call_1",
            "content":"The repository contains a CUDA inference engine. Summarize its purpose in one sentence."
        })"));
    }
    const auto request = serve::parse_chat_completion_request(body, {});
    return serve::to_prompt_input(request.generation,
                                  {.enable_thinking = true, .preserve_thinking = true}, {});
}

int compare_prefill(const char* path) {
    DeviceContext device;
    auto loaded = family::load_model(path, {.vision = false}, device);
    schedule::Parameters parameters(*loaded);
    const auto& config         = loaded->config().text;
    const auto hidden_size     = schedule::dimension(config.hidden_size);
    const auto value_heads     = schedule::dimension(config.gdn->linear_num_value_heads);
    const auto convolution_dim = schedule::dimension(config.gdn->conv_channels());
    const auto value_dim =
        schedule::dimension(config.gdn->linear_num_value_heads * config.gdn->linear_value_head_dim);
    auto frontend =
        family::make_frontend(loaded->resources(), {.vision_enabled = false, .max_context = 512});
    auto prepared      = frontend.prepare(input_prompt());
    const auto& prompt = family::PreparedPromptAccess::view(prepared);
    const auto& ids    = prompt.token_ids;
    std::cout << "prompt tokens=" << ids.size() << '\n';
    std::cout << "rewrite frontiers:";
    for (auto f : prompt.identity.rewrite_execution_frontiers) { std::cout << ' ' << f; }
    std::cout << " checkpoint="
              << (prompt.identity.rewrite_checkpoint ? prompt.identity.rewrite_checkpoint->frontier
                                                     : 0)
              << " opportunities:";
    for (auto o : prompt.context_cache.opportunities) { std::cout << ' ' << o.frontier; }
    std::cout << '\n';
    const int T = static_cast<int>(ids.size());
    // Preserve the cold execution profile for a single-user prompt: user content, message close,
    // assistant opener, then thinking opener. Splitting the synthetic preamble can change
    // numerical results even when cached and cold execution still agree with each other.
    std::vector<int> baseline_frontiers;
    for (const char* suffix : {"<|im_end|>\n<|im_start|>assistant\n<think>\n",
                               "<|im_start|>assistant\n<think>\n", "<think>\n"}) {
        const auto tokens = frontend.tokenize_text(suffix);
        if (tokens.size() >= ids.size() ||
            !std::equal(tokens.rbegin(), tokens.rend(), ids.rbegin())) {
            throw std::runtime_error("prompt does not match the reference chat framing");
        }
        baseline_frontiers.push_back(T - static_cast<int>(tokens.size()));
    }
    baseline_frontiers.push_back(T);
    std::vector<int> ordinary_frontiers(prompt.identity.rewrite_execution_frontiers.begin(),
                                        prompt.identity.rewrite_execution_frontiers.end());
    ordinary_frontiers.push_back(T);
    std::vector<int> cached_frontiers = ordinary_frontiers;
    if (prompt.identity.rewrite_checkpoint) {
        cached_frontiers.push_back(prompt.identity.rewrite_checkpoint->frontier);
    }
    for (auto opportunity : prompt.context_cache.opportunities) {
        cached_frontiers.push_back(opportunity.frontier);
    }
    for (auto* frontiers : {&ordinary_frontiers, &cached_frontiers}) {
        std::sort(frontiers->begin(), frontiers->end());
        frontiers->erase(std::unique(frontiers->begin(), frontiers->end()), frontiers->end());
    }
    WorkspaceArena workspace(1ULL << 30);
    DeviceArena tensors(64ULL << 20);
    auto token_ids = tensors.alloc(DType::I32, {T});
    CUDA_CHECK(cudaMemcpy(token_ids.data, ids.data(), token_ids.bytes(), cudaMemcpyHostToDevice));
    auto embedding = tensors.alloc(DType::BF16, {hidden_size, T});
    ops::embedding(token_ids, parameters.text.token_embedding, embedding, device.stream);
    const auto& layer = parameters.text.layers[0];
    const auto& gdn   = std::get<schedule::GdnParameters>(layer.mixer);
    std::vector<std::vector<float>> reference;
    std::size_t differences = 0;
    for (bool split : {false, true}) {
        auto hidden = tensors.alloc(DType::BF16, {hidden_size, T});
        auto g      = tensors.alloc(DType::FP32, {value_heads, T});
        auto beta   = tensors.alloc(DType::FP32, {value_heads, T});
        auto qkv    = tensors.alloc(DType::BF16, {convolution_dim, T});
        auto gate   = tensors.alloc(DType::BF16, {value_dim, T});
        int start   = 0;
        for (int end : split ? cached_frontiers : ordinary_frontiers) {
            auto x  = embedding.slice(1, start, end - start);
            auto h  = hidden.slice(1, start, end - start);
            auto gs = g.slice(1, start, end - start);
            auto bs = beta.slice(1, start, end - start);
            auto qs = qkv.slice(1, start, end - start);
            auto zs = gate.slice(1, start, end - start);
            schedule::gdn_norm_control(x, layer.input_norm, config.rms_norm_eps, gdn, h, gs, bs,
                                       workspace, device.execution_view());
            schedule::gdn_projection(h, gdn, qs, zs, workspace, device.stream);
            start = end;
        }
        device.synchronize();
        std::vector<std::vector<float>> current{read_tensor(hidden), read_tensor(g),
                                                read_tensor(beta), read_tensor(qkv),
                                                read_tensor(gate)};
        if (!split) {
            reference = current;
        } else {
            const char* labels[] = {"L0 normalized", "L0 g", "L0 beta", "L0 qkv", "L0 gate"};
            for (int i = 0; i < 5; ++i) {
                differences += compare(labels[i], reference[i], current[i]);
            }
        }
    }

    LayoutBuilder layout;
    auto decoder_layout =
        family::plan_decoder_state(layout, {.full_attention_layers = 16,
                                            .mtp_layers            = 1,
                                            .capacity              = 512,
                                            .kv_heads              = 4,
                                            .attention_head_dim    = 256,
                                            .kv_storage = KvCacheStorage::KvarnK4V2Group128,
                                            .text_physical_page_groups = 4});
    auto state_layout = plan_linear_attention_state_pool(layout, {.layers         = 48,
                                                                  .conv_channels  = convolution_dim,
                                                                  .conv_width     = 3,
                                                                  .value_heads    = 48,
                                                                  .value_head_dim = 128,
                                                                  .key_head_dim   = 128,
                                                                  .slot_count     = 2});
    auto round_layout = family::begin_round_state_layout(
        layout, {.hidden = hidden_size, .output_rows = schedule::dimension(config.vocab_size)});
    family::complete_round_state_layout(layout, round_layout);
    DeviceBuffer storage(layout.finish(256));
    storage.fill();
    DeviceSpan backing{storage.p, storage.bytes};
    family::DecoderState decoder(backing, decoder_layout);
    LinearAttentionStatePool state(backing, state_layout);
    family::RoundState io(backing, round_layout);
    auto row         = decoder.text_kv.execution_tables().acquire(0);
    auto reservation = decoder.text_kv.page_pool().reserve(4);
    std::vector<DeviceKVPageLease> pages;
    pages.reserve(4);
    decoder.text_kv.page_pool().materialize(*reservation, 4, pages);
    decoder.text_kv.execution_tables().publish(row.handle(), 0, pages, device.stream);
    ops::set_i32_scalar(io.text_kv_table_row, 0, device.stream);
    auto prefill_hidden = tensors.alloc(DType::BF16, {hidden_size, 2048});
    std::vector<float> logits_reference, state_reference;
    for (const auto* frontiers : {&baseline_frontiers, &ordinary_frontiers, &cached_frontiers}) {
        state.zero_all(device.stream);
        decoder.text_kv.reset_kvarn_tail_row(0, device.stream);
        int start = 0;
        int slot  = 0;
        for (int end : *frontiers) {
            schedule::TextContext card(device, parameters, workspace,
                                       decoder.text_kv.execution_view(row), state, io,
                                       prefill_hidden, 2048, start, {}, &decoder.text_kv);
            const int destination = frontiers == &cached_frontiers ? 1 - slot : slot;
            card.set_linear_state_slots(slot, destination);
            (void)card.prefill_chunk(ids, start, end - start, end == T, 0);
            slot  = destination;
            start = end;
        }
        device.synchronize();
        auto logits = read_tensor(io.logits);
        std::vector<float> recurrent;
        for (unsigned layer = 0; layer < state.layer_count(); ++layer) {
            for (auto tensor : {state.conv_slot(layer, slot), state.recurrent_slot(layer, slot)}) {
                auto values = read_tensor(tensor);
                recurrent.insert(recurrent.end(), values.begin(), values.end());
            }
        }
        if (frontiers == &baseline_frontiers) {
            logits_reference = logits;
            state_reference  = recurrent;
        } else {
            std::cout << (frontiers == &ordinary_frontiers ? "cold vs historical profile\n"
                                                           : "cached vs historical profile\n");
            differences += compare("all convolution/recurrent state", state_reference, recurrent);
            differences += compare("final logits", logits_reference, logits);
        }
    }
    // OpenAI moves its automatic write marker to the newest message. Reusing the old prompt
    // must keep the same arithmetic as a cold prefill of the extended history.
    auto before        = family::PreparedPromptAccess::take(frontend.prepare(tool_prompt(false)));
    auto after         = family::PreparedPromptAccess::take(frontend.prepare(tool_prompt(true)));
    const int frontier = before.identity.rewrite_checkpoint->frontier;
    family::detail::ResidentPrefixIdentity resident;
    resident.assign(before);
    family::detail::PrefixShortlistDigests before_digest, after_digest;
    before_digest.assign(before);
    after_digest.assign(after);
    if (!family::detail::prefix_matches(after, before.token_ids, resident, frontier) ||
        before_digest.at(frontier) != after_digest.at(frontier)) {
        throw std::runtime_error("moving the automatic cache marker invalidated the tool prefix");
    }
    for (bool warm : {false, true}) {
        state.zero_all(device.stream);
        decoder.text_kv.reset_kvarn_tail_row(0, device.stream);
        int cursor = 0, slot = 0;
        const auto execute = [&](const family::PreparedPromptData& input, int stop) {
            while (cursor < stop) {
                int end = std::min(cursor + 2048, stop);
                const auto split =
                    std::upper_bound(input.identity.rewrite_execution_frontiers.begin(),
                                     input.identity.rewrite_execution_frontiers.end(), cursor);
                if (split != input.identity.rewrite_execution_frontiers.end()) {
                    end = std::min(end, static_cast<int>(*split));
                }
                schedule::TextContext card(device, parameters, workspace,
                                           decoder.text_kv.execution_view(row), state, io,
                                           prefill_hidden, 2048, cursor, {}, &decoder.text_kv);
                const int destination = warm ? 1 - slot : slot;
                card.set_linear_state_slots(slot, destination);
                (void)card.prefill_chunk(input.token_ids, cursor, end - cursor,
                                         end == static_cast<int>(after.token_ids.size()), 0);
                slot   = destination;
                cursor = end;
            }
        };
        if (warm) { execute(before, frontier); }
        execute(after, static_cast<int>(after.token_ids.size()));
        device.synchronize();
        auto logits = read_tensor(io.logits);
        std::vector<float> recurrent;
        for (unsigned layer = 0; layer < state.layer_count(); ++layer) {
            for (auto tensor : {state.conv_slot(layer, slot), state.recurrent_slot(layer, slot)}) {
                auto values = read_tensor(tensor);
                recurrent.insert(recurrent.end(), values.begin(), values.end());
            }
        }
        if (!warm) {
            logits_reference = std::move(logits);
            state_reference  = std::move(recurrent);
        } else {
            differences +=
                compare("tool-prefix convolution/recurrent state", state_reference, recurrent);
            differences += compare("tool-prefix logits", logits_reference, logits);
        }
    }
    return differences == 0 ? 0 : 1;
}

int main() {
    std::cout << std::unitbuf;
    const char* path = std::getenv("NINFER_TEST_ARTIFACT");
    if (!path) {
        std::cout << "SKIP: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }
    try {
        int failures = compare_prefill(path);
        std::vector<TokenId> expected;
        for (bool caching : {false, true}) {
            EngineOptions options;
            options.artifact_path         = path;
            options.max_context           = 512;
            options.kv_capacity           = KvCapacityPolicy::explicit_capacity(1024);
            options.max_concurrency       = 2;
            options.prefill_chunk         = 2048;
            options.kv_cache              = KvCacheStorage::KvarnK4V2Group128;
            options.context_cache.enabled = caching;
            options.enable_vision         = false;
            options.speculative           = {SpeculativeBackend::Mtp, 3, ProposalHead::Optimized};
            Engine engine(options);
            RequestOptions request;
            request.execution.requested_output_tokens = 128;
            request.execution.sampling.temperature    = 1;
            request.execution.sampling.seed           = 42;
            request.execution.sampling.top_p          = .95F;
            request.execution.sampling.top_k          = 20;
            request.stop.include_model_defaults       = false;
            auto result = engine.generate(engine.prepare(input_prompt()), request);
            if (result.reused_prompt_tokens != 0 || result.generated_token_ids.size() != 128) {
                throw std::runtime_error(
                    "cold Engine fixture did not execute its complete prompt/output");
            }
            if (!caching) {
                expected = result.generated_token_ids;
            } else if (expected != result.generated_token_ids) {
                ++failures;
            }
            std::cout << "Engine cache=" << caching
                      << " outputs=" << result.generated_token_ids.size()
                      << " exact=" << (expected == result.generated_token_ids) << '\n';
        }
        EngineOptions options;
        options.artifact_path                    = path;
        options.max_context                      = 1024;
        options.kv_capacity                      = KvCapacityPolicy::explicit_capacity(1024);
        options.prefill_chunk                    = 2048;
        options.kv_cache                         = KvCacheStorage::KvarnK4V2Group128;
        options.enable_vision                    = false;
        options.context_cache.device_state_slots = 4;
        options.speculative = {SpeculativeBackend::Mtp, 3, ProposalHead::Optimized};
        Engine engine(options);
        RequestOptions request;
        request.execution.sampling.temperature    = 0;
        request.execution.requested_output_tokens = 1;
        request.stop.include_model_defaults       = false;
        auto initial_prompt                       = engine.prepare(tool_prompt(false));
        const auto source_prompt_tokens           = initial_prompt.summary().prompt_tokens;
        const auto source = engine.generate(std::move(initial_prompt), request);
        request.execution.requested_output_tokens = 128;
        const auto warm = engine.generate(engine.prepare(tool_prompt(true)), request);
        request.execution.allow_prefix_reuse = false;
        const auto cold = engine.generate(engine.prepare(tool_prompt(true)), request);
        if (source.generated_token_ids.size() != 1 ||
            warm.reused_prompt_tokens != source_prompt_tokens || cold.reused_prompt_tokens != 0 ||
            warm.generated_token_ids.size() != 128 ||
            warm.generated_token_ids != cold.generated_token_ids) {
            throw std::runtime_error("cached and cold Engine tool continuations differ");
        }
        std::cout << "Engine tool-prefix reused=" << warm.reused_prompt_tokens
                  << " outputs=128 exact=1\n";
        return failures ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << "prefill precision test failed: " << error.what() << '\n';
        return 1;
    }
}
