#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "targets/qwen3_6_27b/impl/variant.h"
#include <ninfer/targets/qwen3_6_27b/package.h>
#include <ninfer/targets/qwen3_6/prepared_prompt.h>
#include "core/linear_attention_state.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/scalar.h"
#include "ninfer/engine.h"

#define NINFER_QWEN36_VARIANT    ::ninfer::targets::qwen3_6_27b::detail::Variant
#define NINFER_QWEN36_RUNTIME_NS qwen3_6_27b_runtime
#include "targets/qwen3_6/impl/runtime/text_context.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace ninfer;
namespace family   = ninfer::targets::qwen3_6;
namespace target   = ninfer::targets::qwen3_6_27b;
namespace schedule = family::detail::qwen3_6_27b_runtime::schedule;
using Config       = target::detail::TextConfig;

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

int compare_prefill(const char* path) {
    DeviceContext device;
    artifact::Reader reader(path);
    artifact::Binder binder(reader);
    auto load =
        target::detail::bind_artifact(binder, target::Package::resolve_weights(reader),
                                      {.vision = false, .speculative = SpeculativeBackend::None});
    auto materialized = artifact::materialize(reader, load.materialization, device);
    target::detail::LoadedModelData loaded(std::move(load.bindings), std::move(materialized));
    auto frontend =
        family::make_frontend(loaded.frontend, {.vision_enabled = false, .max_context = 512});
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
    auto embedding = tensors.alloc(DType::BF16, {Config::hidden, T});
    ops::embedding(token_ids, loaded.runtime.token_embedding, embedding, device.stream);
    const auto& layer = loaded.runtime.gdn_layers[0];
    std::vector<std::vector<float>> reference;
    std::size_t differences = 0;
    for (bool split : {false, true}) {
        auto hidden = tensors.alloc(DType::BF16, {Config::hidden, T});
        auto g      = tensors.alloc(DType::FP32, {Config::gdn_value_heads, T});
        auto beta   = tensors.alloc(DType::FP32, {Config::gdn_value_heads, T});
        auto qkv    = tensors.alloc(DType::BF16, {Config::convolution_dim, T});
        auto gate   = tensors.alloc(DType::BF16, {Config::value_dim, T});
        int start   = 0;
        for (int end : split ? cached_frontiers : ordinary_frontiers) {
            auto x  = embedding.slice(1, start, end - start);
            auto h  = hidden.slice(1, start, end - start);
            auto gs = g.slice(1, start, end - start);
            auto bs = beta.slice(1, start, end - start);
            auto qs = qkv.slice(1, start, end - start);
            auto zs = gate.slice(1, start, end - start);
            target::detail::Variant::gdn_norm_control_projection(
                x, layer.input_norm, Config::rms_epsilon, layer.projection, h, gs, bs, workspace,
                device.execution_view());
            target::detail::Variant::gdn_input_projection(
                h, layer.projection, qs, zs, family::TextPhase::Prefill, workspace, device.stream);
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
    auto state_layout =
        plan_linear_attention_state_pool(layout, {.layers         = 48,
                                                  .conv_channels  = Config::convolution_dim,
                                                  .conv_width     = 3,
                                                  .value_heads    = 48,
                                                  .value_head_dim = 128,
                                                  .key_head_dim   = 128,
                                                  .slot_count     = 2});
    auto round_layout = family::begin_round_state_layout(
        layout, {.hidden = Config::hidden, .output_rows = Config::output_rows});
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
    auto prefill_hidden = tensors.alloc(DType::BF16, {Config::hidden, 2048});
    std::vector<float> logits_reference, state_reference;
    for (bool split : {false, true}) {
        state.zero_all(device.stream);
        decoder.text_kv.reset_kvarn_tail_row(0, device.stream);
        int start = 0;
        int slot  = 0;
        for (int end : split ? cached_frontiers : ordinary_frontiers) {
            schedule::TextContext card(device, loaded.runtime, workspace,
                                       decoder.text_kv.execution_view(row), state, io,
                                       prefill_hidden, 2048, start, {}, &decoder.text_kv);
            const int destination = split ? 1 - slot : slot;
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
        if (!split) {
            logits_reference = logits;
            state_reference  = recurrent;
        } else {
            differences += compare("all convolution/recurrent state", state_reference, recurrent);
            differences += compare("final logits", logits_reference, logits);
        }
    }
    return differences == 0 ? 0 : 1;
}

int main() {
    std::cout << std::unitbuf;
    const char* path = std::getenv("NINFER_TEST_WEIGHTS");
    if (!path) {
        std::cout << "SKIP: NINFER_TEST_WEIGHTS is not set\n";
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
        return failures ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << "prefill precision test failed: " << error.what() << '\n';
        return 1;
    }
}
