#include "ninfer/engine.h"

#include <algorithm>
#include <array>
#include <charconv>
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
                                     std::uint32_t max_concurrency = 1) {
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
    bool full_proposal_head            = false;
    std::vector<ninfer::TokenId> corpus;
};

void verify_parity(const char* artifact, KvProfile profile, const ParityCases& cases) {
    constexpr std::array<std::uint32_t, 5> long_contexts{8190, 32799, 122879, 196607, 245743};
    const int samples = cases.corpus.empty() ? 3 : 8;
    if (cases.sample >= samples) { throw std::invalid_argument("long samples require --corpus"); }
    std::array<std::array<std::vector<ninfer::TokenId>, kMaximumConcurrency>, 8> expected;
    const bool dflash2                      = cases.backend == ninfer::SpeculativeBackend::DFlash2;
    const std::vector<std::uint32_t> depths = dflash2
                                                  ? std::vector<std::uint32_t>{0, 1, 3, 7, 15}
                                                  : std::vector<std::uint32_t>{0, 3, 1, 2, 4, 5};
    for (std::uint32_t depth : depths) {
        if (depth != 0 && cases.depth >= 0 && depth != cases.depth) { continue; }
        // Every MTP width and fresh repeat uses ordinary greedy as oracle. DFlash retains its
        // same-width repeatability check; its wider target arithmetic has a separate contract.
        for (int repeat = 0; repeat < 2; ++repeat) {
            auto options = engine_options(artifact, profile.storage, depth, cases.concurrency);
            options.speculative.backend =
                depth == 0 ? ninfer::SpeculativeBackend::None : cases.backend;
            const auto prompt_capacity = cases.sample >= 3 ? long_contexts[cases.sample - 3]
                                                           : (cases.sample == 0   ? 128U
                                                              : cases.sample == 1 ? 1024U
                                                                                  : 4096U);
            options.max_context        = cases.output_tokens + prompt_capacity + 16;
            options.kv_capacity    = ninfer::KvCapacityPolicy::explicit_capacity(cases.concurrency *
                                                                                 options.max_context);
            options.prefill_chunk  = cases.prefill_chunk;
            options.use_cuda_graph = cases.graphs;
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
                    row_request.execution.requested_output_tokens -= row;
                    handles.push_back(engine.submit(std::move(prepared), row_request));
                }
                for (std::uint32_t row = 0; row < cases.concurrency; ++row) {
                    const auto result = handles[row].wait();
                    const std::string label =
                        std::string(profile.name) + " k=" + std::to_string(depth) +
                        " sample=" + std::to_string(sample) + " row=" + std::to_string(row) +
                        " prompt=" + std::to_string(prompt_tokens[row]);
                    if (result.generated_token_ids.size() != cases.output_tokens - row) {
                        throw std::runtime_error(label +
                                                 " did not reach the requested decode length");
                    }
                    if (depth != 0 && (result.speculative.backend != cases.backend ||
                                       result.speculative.rounds == 0)) {
                        throw std::runtime_error(label + " did not execute the selected backend");
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
                              << " tokens reused=" << result.reused_prompt_tokens << std::endl;
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
                } else if (argument == "--sample" && number < 8) {
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
            } else if (argument == "--full-proposal-head") {
                cases.full_proposal_head = true;
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
                    "[--output-tokens 128..16384] [--sample 0..7] "
                    "[--spec mtp|dflash2] [--draft-tokens K] [--prefill-chunk 1..4096] "
                    "[--concurrency 1..8] [--full-proposal-head] "
                    "[--kv-dtype bf16|int8|fp8|nvfp4|k8v4|kvarn] "
                    "[--no-cuda-graph] [--prefix-reuse] [--corpus PATH]");
            }
        }
        if (cases.backend == ninfer::SpeculativeBackend::DFlash2) {
            if (cases.depth >= 0 && cases.depth != 1 && cases.depth != 3 && cases.depth != 7 &&
                cases.depth != 15) {
                throw std::invalid_argument("DFlash2 requires K=1,3,7,15");
            }
        } else if (cases.depth > 5) {
            throw std::invalid_argument("MTP requires K=1..5");
        }
        for (const KvProfile profile : kKvProfiles) {
            if (!selected_kv.empty() && profile.name != selected_kv) { continue; }
            verify_parity(artifact, profile, cases);
        }
    } catch (const std::exception& error) {
        std::cerr << "greedy MTP parity test failed: " << error.what() << '\n';
        return 1;
    }

    std::cout << "ok\n";
    return 0;
}
