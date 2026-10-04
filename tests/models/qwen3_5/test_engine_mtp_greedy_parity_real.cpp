#include "ninfer/engine.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr std::uint32_t kOutputTokens       = 512;
constexpr std::uint32_t kMaximumConcurrency = 8;

struct KvProfile {
    std::string_view name;
    ninfer::KvCacheStorage storage;
};

constexpr std::array kKvProfiles{
    KvProfile{"bf16", ninfer::KvCacheStorage::BFloat16},
    KvProfile{"int8", ninfer::KvCacheStorage::Int8Group64},
    KvProfile{"fp8", ninfer::KvCacheStorage::Fp8E4M3Row256},
    KvProfile{"nvfp4", ninfer::KvCacheStorage::Nvfp4Group16},
    KvProfile{"k8v4", ninfer::KvCacheStorage::Fp8KeyNvfp4Value},
    KvProfile{"kvarn", ninfer::KvCacheStorage::KvarnK4V2Group128},
};

ninfer::EngineOptions engine_options(const char* artifact, ninfer::KvCacheStorage kv_storage,
                                     std::uint32_t mtp_draft_tokens,
                                     std::uint32_t max_concurrency = 1, bool adaptive = false) {
    const bool mtp = mtp_draft_tokens != 0;
    ninfer::EngineOptions options;
    options.artifact_path   = artifact;
    options.max_context     = 512;
    options.kv_capacity     = ninfer::KvCapacityPolicy::explicit_capacity(512 * max_concurrency);
    options.max_concurrency = max_concurrency;
    options.prefill_chunk   = 128;
    options.kv_cache        = kv_storage;
    options.use_cuda_graph  = false;
    options.speculative.backend =
        mtp ? ninfer::SpeculativeBackend::Mtp : ninfer::SpeculativeBackend::None;
    options.speculative.draft_tokens = mtp_draft_tokens;
    options.speculative.mtp_draft_policy =
        adaptive && mtp ? ninfer::MtpDraftPolicy::Adaptive : ninfer::MtpDraftPolicy::Fixed;
    options.speculative.proposal_head =
        mtp ? ninfer::ProposalHead::Optimized : ninfer::ProposalHead::Full;
    return options;
}

ninfer::RequestOptions greedy_request(std::uint32_t output_tokens = kOutputTokens) {
    ninfer::RequestOptions request;
    request.execution.requested_output_tokens = output_tokens;
    request.execution.sampling.temperature    = 0.0F;
    request.execution.sampling.seed           = 424242;
    request.execution.allow_prefix_reuse      = false;
    request.stop.include_model_defaults       = false;
    return request;
}

ninfer::PromptInput prompt() {
    ninfer::ChatMessage message;
    message.role = ninfer::ChatRole::User;
    message.parts.push_back(ninfer::MessagePart{
        .kind  = ninfer::MessagePartKind::Text,
        .text  = "Write snake in Python, but in a code block. Do not use tools.",
        .media = {},
    });
    ninfer::PromptInput input;
    input.messages.push_back(std::move(message));
    input.options.enable_thinking = true;
    return input;
}

void verify_result(std::string_view label, const ninfer::GenerationResult& result,
                   const std::vector<ninfer::TokenId>& expected) {
    const auto& actual = result.generated_token_ids;
    if (actual.size() != expected.size() ||
        result.finish_reason != ninfer::FinishReason::OutputLimit) {
        throw std::runtime_error(std::string(label) + " did not reach its fixed output limit");
    }
    const auto [expected_mismatch, actual_mismatch] =
        std::mismatch(expected.begin(), expected.end(), actual.begin());
    if (expected_mismatch != expected.end()) {
        const std::size_t index = static_cast<std::size_t>(expected_mismatch - expected.begin());
        throw std::runtime_error(std::string(label) + " mismatch at token " +
                                 std::to_string(index) +
                                 ": expected=" + std::to_string(*expected_mismatch) +
                                 " actual=" + std::to_string(*actual_mismatch));
    }
}

struct ParityCases {
    ninfer::SpeculativeBackend backend = ninfer::SpeculativeBackend::Mtp;
    std::uint32_t output_tokens        = kOutputTokens;
    std::uint32_t prefill_chunk        = 1024;
    std::uint32_t concurrency          = 1;
    int sample                         = 0;
    int depth                          = -1;
    bool graphs                        = true;
    bool prefix_reuse                  = false;
    bool disable_context_cache         = false;
    bool full_proposal_head            = false;
    bool adaptive                      = false;
    bool compressed_scales             = false;
    bool tool_loop                     = false;
    std::vector<ninfer::TokenId> corpus;
};

void verify_tool_loop(const char* artifact, KvProfile profile, bool compressed_scales) {
    std::vector<ninfer::PromptInput> prompts;
    std::vector<ninfer::GenerationResult> expected;
    for (const bool adaptive : {false, true}) {
        auto options = engine_options(artifact, profile.storage, adaptive ? 15U : 0U, 2, adaptive);
        options.enable_nvfp4_scale_compression = compressed_scales;
        options.max_context                    = 65536;
        options.kv_capacity                    = ninfer::KvCapacityPolicy::explicit_capacity(65536);
        options.prefill_chunk                  = 2048;
        options.use_cuda_graph                 = true;
        options.context_cache.device_state_slots     = 2;
        options.context_cache.host_state_slots       = 2;
        options.context_cache.host_kv_capacity_bytes = 512ULL << 20;
        ninfer::Engine engine(options);
        auto request                         = greedy_request(2048);
        request.execution.allow_prefix_reuse = true;
        request.execution.thinking.budget    = 256;
        request.stop.include_model_defaults  = true;

        ninfer::PromptInput input;
        input.options.enable_thinking   = true;
        input.options.preserve_thinking = true;
        input.options.tool_jsons.push_back(
            R"({"type":"function","function":{"name":"read_chunk","description":"Read the next diagnostic chunk.","parameters":{"type":"object","properties":{"chunk":{"type":"integer"}},"required":["chunk"]}}})");
        for (int tool = 1; tool < 27; ++tool) {
            input.options.tool_jsons.push_back(
                std::string(R"({"type":"function","function":{"name":"unused_)") +
                std::to_string(tool) +
                R"(","description":"Unused diagnostic tool.","parameters":{"type":"object","properties":{"value":{"type":"string"}}}}})");
        }
        const auto message = [](ninfer::ChatRole role, std::string text) {
            ninfer::ChatMessage value;
            value.role = role;
            value.parts.push_back({.kind = ninfer::MessagePartKind::Text, .text = std::move(text)});
            return value;
        };
        input.messages.push_back(message(
            ninfer::ChatRole::System,
            "On every turn call read_chunk exactly once with the next integer chunk number. "
            "Do not finish or answer in prose."));
        input.messages.push_back(message(ninfer::ChatRole::User, "Read chunk 1."));

        for (unsigned turn = 0; turn < 3; ++turn) {
            if (!adaptive) { prompts.push_back(input); }
            const auto result       = engine.generate(engine.prepare(prompts[turn]), request);
            const std::string label = std::string(profile.name) + " tool-loop " +
                                      (adaptive ? "adaptive" : "off") +
                                      " turn=" + std::to_string(turn);
            if (result.tool_calls.size() != 1 || result.tool_calls.front().name != "read_chunk") {
                throw std::runtime_error(label + " failed to produce the expected tool call");
            }
            if (turn != 0 && result.reused_prompt_tokens == 0) {
                throw std::runtime_error(label + " did not exercise continuation reuse");
            }
            if (adaptive) {
                const auto& oracle = expected[turn];
                if (result.generated_token_ids != oracle.generated_token_ids ||
                    result.finish_reason != oracle.finish_reason ||
                    result.reasoning != oracle.reasoning || result.content != oracle.content ||
                    result.tool_calls.front().arguments_json !=
                        oracle.tool_calls.front().arguments_json) {
                    const auto mismatch = std::mismatch(
                        result.generated_token_ids.begin(), result.generated_token_ids.end(),
                        oracle.generated_token_ids.begin(), oracle.generated_token_ids.end());
                    throw std::runtime_error(
                        label + " differs from MTP-off at token " +
                        std::to_string(mismatch.first - result.generated_token_ids.begin()));
                }
            } else {
                expected.push_back(result);
                ninfer::ChatMessage assistant;
                assistant.role              = ninfer::ChatRole::Assistant;
                assistant.reasoning_content = result.reasoning;
                // Exercise the response-replay path when a client rewrites stored reasoning.
                if (turn == 1) { assistant.reasoning_content += "\nClient-normalized history."; }
                if (!result.content.empty()) {
                    assistant.parts.push_back(
                        {.kind = ninfer::MessagePartKind::Text, .text = result.content});
                }
                const std::string id = "call_" + std::to_string(turn);
                assistant.tool_calls.push_back(
                    {.id             = id,
                     .name           = "read_chunk",
                     .arguments_json = result.tool_calls.front().arguments_json});
                input.messages.push_back(std::move(assistant));
                std::string diagnostic;
                for (int line = 0; line < 500; ++line) {
                    diagnostic += "chunk=" + std::to_string(turn) +
                                  " line=" + std::to_string(line) +
                                  " key=value abcdefghijklmnopqrstuvwxyz0123456789 "
                                  "ABCDEFGHIJKLMNOPQRSTUVWXYZ9876543210\n";
                }
                auto tool_result         = message(ninfer::ChatRole::Tool, std::move(diagnostic));
                tool_result.tool_call_id = id;
                input.messages.push_back(std::move(tool_result));
            }
            std::cout << label << " tokens=" << result.generated_token_ids.size()
                      << " reused=" << result.reused_prompt_tokens
                      << " path=" << static_cast<int>(result.prefix_reuse_path) << " matched"
                      << std::endl;
        }
    }
}

void verify_parity(const char* artifact, KvProfile profile, const ParityCases& cases) {
    constexpr std::array<std::uint32_t, 5> long_contexts{8190, 32799, 122879, 196607, 245743};
    const int samples = cases.corpus.empty() ? 3 : 9;
    if (cases.sample >= samples) { throw std::invalid_argument("corpus samples require --corpus"); }
    std::array<std::array<std::vector<ninfer::TokenId>, kMaximumConcurrency>, 9> expected;
    const bool dflash2 = cases.backend == ninfer::SpeculativeBackend::DFlash2;
    const std::vector<std::uint32_t> depths =
        cases.depth > 0  ? std::vector<std::uint32_t>{0, static_cast<std::uint32_t>(cases.depth)}
        : cases.adaptive ? std::vector<std::uint32_t>{0, 15}
        : dflash2        ? std::vector<std::uint32_t>{0, 1, 3, 7, 15}
                         : std::vector<std::uint32_t>{0, 3, 1, 2, 4, 5, 15};
    for (std::uint32_t depth : depths) {
        // Every MTP width and fresh repeat uses ordinary greedy as oracle. DFlash retains its
        // same-width repeatability check; its wider target arithmetic has a separate contract.
        for (int repeat = 0; repeat < 2; ++repeat) {
            auto options =
                engine_options(artifact, profile.storage, depth, cases.concurrency, cases.adaptive);
            options.speculative.backend =
                depth == 0 ? ninfer::SpeculativeBackend::None : cases.backend;
            const auto prompt_capacity = cases.sample == 8   ? 231U
                                         : cases.sample >= 3 ? long_contexts[cases.sample - 3]
                                                             : (cases.sample == 0   ? 128U
                                                                : cases.sample == 1 ? 1024U
                                                                                    : 4096U);
            options.max_context        = cases.output_tokens + prompt_capacity + 16;
            if (cases.sample == 8) { options.max_context = std::max(16384U, options.max_context); }
            // Wide KVarN verification changes graph node count across the 1K route boundary.
            // Capture both profiles even when the test prompt itself is short.
            if (depth > 5) { options.max_context = std::max(options.max_context, 2048U); }
            // Reserve independently rounded rows. Rounding only the total under-reserves a page
            // for ragged long prompts and can serialize this concurrency test (e.g. sample 4).
            const auto kv_per_request = ((options.max_context + 127U) / 128U) * 128U;
            // A smaller logical ceiling can make the rounded capacity exceed C logical
            // address spaces for KV formats whose pages are smaller than 128 tokens.
            options.max_context = kv_per_request;
            options.kv_capacity =
                ninfer::KvCapacityPolicy::explicit_capacity(cases.concurrency * kv_per_request);
            options.prefill_chunk                  = cases.prefill_chunk;
            options.use_cuda_graph                 = cases.graphs;
            options.enable_nvfp4_scale_compression = cases.compressed_scales;
            options.context_cache.enabled          = !cases.disable_context_cache;
            if (cases.full_proposal_head) {
                options.speculative.proposal_head = ninfer::ProposalHead::Full;
            }
            if (cases.prefix_reuse) {
                options.context_cache.device_state_slots        = cases.concurrency;
                options.context_cache.max_private_continuations = cases.concurrency;
            }
            ninfer::Engine engine(options);
            auto request                         = greedy_request(cases.output_tokens);
            request.execution.allow_prefix_reuse = cases.prefix_reuse;
            for (int sample = 0; sample < samples; ++sample) {
                if (cases.sample >= 0 && sample != cases.sample) { continue; }
                const auto prepare = [&](std::uint32_t row) {
                    if (sample == 8) {
                        // Short, distinct corpus slices exercise low-acceptance compact batches.
                        // Equal output budgets preserve the full C8 frontier through the run.
                        const std::size_t begin = 128 * row;
                        if (cases.corpus.size() < begin + 231) {
                            throw std::invalid_argument(
                                "corpus is shorter than the selected prompt");
                        }
                        return engine.prepare_tokens(std::vector<ninfer::TokenId>(
                            cases.corpus.begin() + begin, cases.corpus.begin() + begin + 231));
                    }
                    const auto raw_prompt = [&](std::vector<ninfer::TokenId> tokens) {
                        // Distinct branches keep every prewarmed endpoint resident. Nested raw
                        // prefixes otherwise extend one continuation and replace its endpoint.
                        if (row > 0) {
                            const auto label =
                                engine.tokenize_text("Request " + std::to_string(row) + ":\n");
                            std::copy(label.begin(), label.end(), tokens.begin());
                        }
                        return engine.prepare_tokens(std::move(tokens));
                    };
                    if (sample >= 3) {
                        const auto length = long_contexts[sample - 3] - 7 * row;
                        if (cases.corpus.size() < length) {
                            throw std::invalid_argument(
                                "corpus is shorter than the selected prompt");
                        }
                        return raw_prompt(std::vector<ninfer::TokenId>(
                            cases.corpus.begin(), cases.corpus.begin() + length));
                    }
                    if (sample == 2) {
                        return raw_prompt(std::vector<ninfer::TokenId>(2110 + 31 * row, 198));
                    }
                    auto input = prompt();
                    if (sample == 1) {
                        std::string text;
                        for (std::uint32_t index = 0; index < 190 + 11 * row; ++index) {
                            text += "x ";
                        }
                        text +=
                            "\nWrite a numbered list of 200 distinct fictional identifiers. "
                            "Do not explain the task and do not stop before the list is complete.";
                        input.messages[0].parts[0].text = std::move(text);
                        input.options.enable_thinking   = false;
                    } else if (row > 0) {
                        input.messages[0].parts[0].text +=
                            "\nNumber the comments starting from " + std::to_string(row * 10) + ".";
                    }
                    return engine.prepare(std::move(input));
                };
                // Establish real reusable state, not a second measured generation of the same case.
                std::vector<ninfer::TokenId> warm_tokens;
                if (cases.prefix_reuse) {
                    auto warm_request                              = request;
                    warm_request.execution.requested_output_tokens = 1;
                    for (std::uint32_t row = 0; row < cases.concurrency; ++row) {
                        const auto warm = engine.generate(prepare(row), warm_request);
                        if (warm.generated_token_ids.size() != 1) {
                            throw std::runtime_error("prefix prewarm failed");
                        }
                        warm_tokens.push_back(warm.generated_token_ids.front());
                    }
                }
                const auto before = engine.runtime_stats();
                std::vector<std::uint32_t> prompt_tokens;
                std::vector<ninfer::GenerationHandle> handles;
                std::cout << "starting " << profile.name
                          << " spec=" << (dflash2 ? "dflash2" : "mtp") << " k=" << depth
                          << " sample=" << sample << " C=" << cases.concurrency
                          << " output=" << cases.output_tokens << " prefill=" << cases.prefill_chunk
                          << " graphs=" << cases.graphs << " prefix=" << cases.prefix_reuse
                          << " repeat=" << repeat << std::endl;
                for (std::uint32_t row = 0; row < cases.concurrency; ++row) {
                    auto prepared = prepare(row);
                    prompt_tokens.push_back(prepared.summary().prompt_tokens);
                    auto row_request = request;
                    if (sample != 8) { row_request.execution.requested_output_tokens -= row; }
                    handles.push_back(engine.submit(std::move(prepared), row_request));
                }
                for (std::uint32_t row = 0; row < cases.concurrency; ++row) {
                    const auto result = handles[row].wait();
                    const std::string label =
                        std::string(profile.name) + " k=" + std::to_string(depth) +
                        " sample=" + std::to_string(sample) + " row=" + std::to_string(row) +
                        " prompt=" + std::to_string(prompt_tokens[row]);
                    if (result.generated_token_ids.size() !=
                        cases.output_tokens - (sample == 8 ? 0 : row)) {
                        throw std::runtime_error(label +
                                                 " did not reach the requested decode length");
                    }
                    if (depth != 0 && (result.speculative.backend != cases.backend ||
                                       result.speculative.rounds == 0)) {
                        throw std::runtime_error(label + " did not execute the selected backend");
                    }
                    if (!dflash2 && depth != 0) {
                        const auto policy = cases.adaptive ? ninfer::MtpDraftPolicy::Adaptive
                                                           : ninfer::MtpDraftPolicy::Fixed;
                        if (result.speculative.mtp_draft_policy != policy ||
                            result.speculative.window_stats.size() !=
                                result.speculative.draft_window ||
                            result.speculative.drafted_per_position.size() !=
                                result.speculative.draft_window) {
                            throw std::runtime_error(label + " did not publish adaptive MTP stats");
                        }
                        std::uint64_t rounds              = 0;
                        std::uint64_t fallbacks           = 0;
                        std::uint64_t drafted             = 0;
                        std::uint64_t accepted            = 0;
                        std::uint64_t committed           = 0;
                        std::uint64_t drafted_by_position = 0;
                        double decode_seconds             = 0;
                        for (const auto& window : result.speculative.window_stats) {
                            const auto selected_width = static_cast<std::uint32_t>(
                                &window - result.speculative.window_stats.data() + 1);
                            if (cases.adaptive && window.rounds != 0 &&
                                selected_width != std::min(3U, depth) &&
                                selected_width != std::min(7U, depth) && selected_width != depth) {
                                throw std::runtime_error(
                                    label + " executed an intermediate adaptive width");
                            }
                            rounds += window.rounds;
                            fallbacks += window.fallback_steps;
                            drafted += window.drafted_tokens;
                            accepted += window.accepted_tokens;
                            committed += window.committed_tokens;
                            decode_seconds += window.decode_seconds;
                        }
                        for (const std::uint64_t count : result.speculative.drafted_per_position) {
                            drafted_by_position += count;
                        }
                        if (rounds !=
                                result.speculative.rounds + result.speculative.fallback_steps ||
                            fallbacks != result.speculative.fallback_steps ||
                            drafted != result.speculative.drafted_tokens ||
                            drafted_by_position != result.speculative.drafted_tokens ||
                            accepted != result.speculative.accepted_tokens ||
                            committed + 1U != result.generated_token_ids.size() ||
                            (!cases.adaptive && result.speculative.window_transitions != 0) ||
                            std::abs(decode_seconds - result.timings.decode_seconds) > 1e-6) {
                            throw std::runtime_error(label +
                                                     " adaptive MTP stats are not conserved");
                        }
                    }
                    if (cases.prefix_reuse &&
                        (result.reused_prompt_tokens != prompt_tokens[row] ||
                         result.generated_token_ids.front() != warm_tokens[row])) {
                        throw std::runtime_error(
                            label + " did not restore the prewarmed frontier: reused=" +
                            std::to_string(result.reused_prompt_tokens));
                    }
                    if ((depth == 0 || dflash2) && repeat == 0) {
                        expected[sample][row] = result.generated_token_ids;
                    }
                    verify_result(label, result, expected[sample][row]);
                    std::cout << label << " matched " << expected[sample][row].size()
                              << " tokens reused=" << result.reused_prompt_tokens;
                    if (cases.adaptive && depth != 0) {
                        std::cout << " windows=";
                        for (std::size_t k = 0; k < result.speculative.window_stats.size(); ++k) {
                            const auto rounds = result.speculative.window_stats[k].rounds;
                            if (rounds != 0) { std::cout << "K" << k + 1 << ":" << rounds << " "; }
                        }
                    }
                    std::cout << std::endl;
                }
                const auto after = engine.runtime_stats();
                if (cases.concurrency > 1 && after.decode_row_rounds - before.decode_row_rounds <=
                                                 after.decode_rounds - before.decode_rounds) {
                    throw std::runtime_error("concurrent case did not execute a compact batch");
                }
            }
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "skip: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }

    try {
        ParityCases cases;
        std::string_view selected_kv;
        for (int index = 1; index < argc; ++index) {
            const std::string_view argument(argv[index]);
            if (argument == "--spec" && index + 1 < argc) {
                const std::string_view backend(argv[++index]);
                if (backend == "dflash2") {
                    cases.backend = ninfer::SpeculativeBackend::DFlash2;
                } else if (backend == "mtp") {
                    cases.backend = ninfer::SpeculativeBackend::Mtp;
                } else {
                    throw std::invalid_argument("--spec requires mtp or dflash2");
                }
            } else if ((argument == "--output-tokens" || argument == "--sample" ||
                        argument == "--draft-tokens" || argument == "--prefill-chunk" ||
                        argument == "--concurrency") &&
                       index + 1 < argc) {
                const std::string_view value(argv[++index]);
                std::uint32_t number = 0;
                const auto parsed =
                    std::from_chars(value.data(), value.data() + value.size(), number);
                if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size()) {
                    throw std::invalid_argument("invalid integer for " + std::string(argument));
                }
                if (argument == "--output-tokens" && number >= 128 && number <= 16384) {
                    cases.output_tokens = number;
                } else if (argument == "--sample" && number < 9) {
                    cases.sample = static_cast<int>(number);
                } else if (argument == "--draft-tokens" && number >= 1 && number <= 15) {
                    cases.depth = static_cast<int>(number);
                } else if (argument == "--prefill-chunk" && number >= 1 && number <= 4096) {
                    cases.prefill_chunk = number;
                } else if (argument == "--concurrency" && number >= 1 &&
                           number <= kMaximumConcurrency) {
                    cases.concurrency = number;
                } else {
                    throw std::invalid_argument("out of range: " + std::string(argument));
                }
            } else if (argument == "--kv-dtype" && index + 1 < argc) {
                selected_kv = argv[++index];
                if (std::none_of(kKvProfiles.begin(), kKvProfiles.end(),
                                 [&](KvProfile profile) { return profile.name == selected_kv; })) {
                    throw std::invalid_argument(
                        "--kv-dtype requires bf16, int8, fp8, nvfp4, k8v4 or kvarn");
                }
            } else if (argument == "--no-cuda-graph") {
                cases.graphs = false;
            } else if (argument == "--prefix-reuse") {
                cases.prefix_reuse = true;
            } else if (argument == "--no-context-cache") {
                cases.disable_context_cache = true;
            } else if (argument == "--full-proposal-head") {
                cases.full_proposal_head = true;
            } else if (argument == "--adaptive") {
                cases.adaptive = true;
            } else if (argument == "--nvfp4-scale-compression") {
                cases.compressed_scales = true;
            } else if (argument == "--tool-loop") {
                cases.tool_loop = true;
            } else if (argument == "--corpus" && index + 1 < argc) {
                std::ifstream input(argv[++index]);
                ninfer::TokenId token;
                while (input >> token) { cases.corpus.push_back(token); }
                if (!input.eof() || cases.corpus.empty()) {
                    throw std::invalid_argument("cannot read token corpus");
                }
            } else {
                throw std::invalid_argument(
                    "usage: mtp_greedy_parity_real_test "
                    "[--output-tokens 128..16384] [--sample 0..8] "
                    "[--spec mtp|dflash2] [--draft-tokens K] [--adaptive] "
                    "[--prefill-chunk 1..4096] "
                    "[--concurrency 1..8] [--full-proposal-head] "
                    "[--kv-dtype bf16|int8|fp8|nvfp4|k8v4|kvarn] "
                    "[--no-cuda-graph] [--prefix-reuse] [--no-context-cache] "
                    "[--corpus PATH] [--tool-loop] "
                    "[--nvfp4-scale-compression]");
            }
        }
        if (cases.adaptive && cases.backend != ninfer::SpeculativeBackend::Mtp) {
            throw std::invalid_argument("--adaptive requires --spec mtp");
        }
        if (cases.disable_context_cache && (cases.prefix_reuse || cases.tool_loop)) {
            throw std::invalid_argument(
                "--no-context-cache cannot be combined with prefix/tool reuse");
        }
        if (cases.backend == ninfer::SpeculativeBackend::DFlash2) {
            if (cases.depth >= 0 && cases.depth != 1 && cases.depth != 3 && cases.depth != 7 &&
                cases.depth != 15) {
                throw std::invalid_argument("DFlash2 requires K=1,3,7,15");
            }
        } else if (cases.depth > 15) {
            throw std::invalid_argument("MTP requires K=1..15");
        }
        for (const KvProfile profile : kKvProfiles) {
            if (!selected_kv.empty() && profile.name != selected_kv) { continue; }
            if (cases.tool_loop) {
                verify_tool_loop(artifact, profile, cases.compressed_scales);
            } else {
                verify_parity(artifact, profile, cases);
            }
        }
    } catch (const std::exception& error) {
        std::cerr << "greedy MTP parity test failed: " << error.what() << '\n';
        return 1;
    }

    std::cout << "ok\n";
    return 0;
}
