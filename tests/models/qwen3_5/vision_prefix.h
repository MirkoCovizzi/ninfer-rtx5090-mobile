#pragma once

#include "ninfer/engine.h"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::test {

inline std::vector<TokenId> vision_prefix_reuse(Engine& engine, SpeculativeBackend backend) {
    std::vector<TokenId> continuation_tokens;
    for (const auto kind : {MediaKind::Image, MediaKind::Video}) {
        const std::string header = "P6\n64 64\n255\n";
        MessagePart media;
        media.kind              = MessagePartKind::Media;
        media.media.kind        = kind;
        media.media.media_type  = "image/x-portable-pixmap";
        media.media.source_name = "pattern.ppm";
        media.media.bytes.assign(header.begin(), header.end());
        for (int i = 0; i < 64 * 64; ++i) {
            media.media.bytes.push_back(i & 255);
            media.media.bytes.push_back((i * 3) & 255);
            media.media.bytes.push_back((i * 7) & 255);
        }
        ChatMessage user;
        user.role = ChatRole::User;
        user.parts.push_back(std::move(media));
        user.parts.push_back(
            {.kind = MessagePartKind::Text, .text = "Describe the pattern briefly.", .media = {}});
        PromptInput input;
        input.messages.push_back(std::move(user));
        input.options.enable_thinking = false;
        RequestOptions options;
        options.execution.requested_output_tokens = 8;
        options.execution.sampling.temperature    = 0.0F;
        options.execution.allow_prefix_reuse      = true;
        options.stop.include_model_defaults       = false;

        const auto first  = engine.generate(engine.prepare(input), options);
        const auto reused = engine.generate(engine.prepare(input), options);
        if (!first.prompt.has_media || first.timings.vision_seconds <= 0 ||
            first.generated_token_ids.size() != 8 ||
            first.finish_reason != FinishReason::OutputLimit ||
            first.speculative.backend != backend ||
            first.speculative.rounds + first.speculative.fallback_steps == 0 ||
            reused.reused_prompt_tokens == 0 || reused.timings.vision_seconds != 0 ||
            reused.finish_reason != FinishReason::OutputLimit ||
            first.generated_token_ids != reused.generated_token_ids) {
            throw std::runtime_error("Vision text-suffix prefix reuse changed the result");
        }

        ChatMessage assistant;
        assistant.role              = ChatRole::Assistant;
        assistant.reasoning_content = first.reasoning;
        assistant.parts.push_back(
            {.kind = MessagePartKind::Text, .text = first.content, .media = {}});
        auto next_media = input.messages.front();
        next_media.parts.front().media.bytes.back() ^= 0x5aU;
        next_media.parts.back().text = "Compare the two patterns briefly.";
        input.messages.push_back(std::move(assistant));
        input.messages.push_back(std::move(next_media));
        const auto appended                  = engine.generate(engine.prepare(input), options);
        options.execution.allow_prefix_reuse = false;
        const auto fresh                     = engine.generate(engine.prepare(input), options);
        if (appended.reused_prompt_tokens == 0 || appended.timings.vision_seconds <= 0 ||
            appended.generated_token_ids.size() != 8 ||
            appended.finish_reason != FinishReason::OutputLimit ||
            fresh.generated_token_ids.size() != 8 ||
            fresh.finish_reason != FinishReason::OutputLimit) {
            throw std::runtime_error(
                std::string("New-media continuation failed: kind=") +
                (kind == MediaKind::Image ? "image" : "video") +
                " reused=" + std::to_string(appended.reused_prompt_tokens) +
                " vision=" + std::to_string(appended.timings.vision_seconds) +
                " outputs=" + std::to_string(appended.generated_token_ids.size()));
        }
        // Full prefill and suffix reuse differ numerically. Callers compare these same-schedule
        // continuation tokens across Host and Device checkpoint placement instead.
        continuation_tokens.insert(continuation_tokens.end(), appended.generated_token_ids.begin(),
                                   appended.generated_token_ids.end());
    }
    return continuation_tokens;
}

} // namespace ninfer::test
