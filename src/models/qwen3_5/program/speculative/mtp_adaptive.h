#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <span>

namespace ninfer::models::qwen3_5::detail {

// Request-local complete-prefix evidence, independent of text formatting.
class MtpAdaptiveSignal final {
public:
    void reset() noexcept { successes_ = {}; }

    void observe(std::uint32_t proposed, std::uint32_t accepted) noexcept {
        // Budget and KVarN publication boundaries censor the observation; they are not failures.
        for (std::size_t i = 0; i < prefixes_.size(); ++i) {
            if (proposed < prefixes_[i]) { continue; }
            successes_[i] = accepted >= prefixes_[i] ? std::min(successes_[i] + 1U, 3U) : 0U;
        }
    }

    [[nodiscard]] bool ready_for_probe(std::uint32_t width) const noexcept {
        return successes_[width == 7 ? 1 : 0] == 3;
    }

private:
    static constexpr std::array<std::uint32_t, 2> prefixes_{3, 7};
    std::array<std::uint32_t, 2> successes_{};
};

// One decision for the exact-B compact batch: min(3,Kmax), min(7,Kmax), or Kmax. A trial gets two
// complete verification observations; the initial short-chain round is never trial evidence.
class MtpAdaptiveBatchController final {
public:
    void reset(std::uint32_t maximum_window) noexcept {
        *this            = MtpAdaptiveBatchController{};
        maximum_         = maximum_window;
        narrow_          = std::min(3U, maximum_);
        middle_          = std::min(7U, maximum_);
        selected_        = narrow_;
        preferred_probe_ = maximum_;
    }

    [[nodiscard]] std::uint32_t select(std::span<const MtpAdaptiveSignal* const> signals,
                                       std::span<const std::uint32_t> room,
                                       std::uint64_t cohort) noexcept {
        if (cohort != cohort_) {
            reset(maximum_);
            cohort_ = cohort;
        }
        if (maximum_ == narrow_) { return narrow_; }

        bool continuing = false;
        bool confident  = true;
        for (std::size_t row = 0; row < signals.size(); ++row) {
            // A short row must not prevent another continuing row from using the wide batch.
            if (room[row] <= maximum_ + 1U) { continue; }
            continuing = true;
            confident  = confident && signals[row]->ready_for_probe(selected_);
        }
        if (!continuing) {
            selected_ = narrow_;
            samples_  = 0;
            trial_    = false;
        } else if (selected_ < maximum_ && !trial_ && confident && cooldown_ == 0 &&
                   narrow_samples_ >= 3) {
            begin_trial(selected_ == narrow_ ? preferred_probe_ : maximum_);
        }
        return selected_;
    }

    void observe_execution(std::uint32_t window, std::span<const std::int32_t> proposed,
                           std::span<const std::int32_t> accepted, double seconds) noexcept {
        // Only settled, full-width rows are comparable. In particular, do not charge the
        // K3->Kmax chain-building round, clipped cache groups, or output-budget tails as losses.
        if (seconds <= 0 || proposed.empty()) { return; }
        for (const auto extent : proposed) {
            if (extent != static_cast<std::int32_t>(window)) { return; }
        }
        if (window == narrow_) {
            trial_          = false;
            narrow_seconds_ = narrow_samples_ == 0
                                  ? seconds
                                  : narrow_seconds_ + 0.25 * (seconds - narrow_seconds_);
            narrow_samples_ = std::min(narrow_samples_ + 1U, 3U);
            if (cooldown_ != 0) { --cooldown_; }
            return;
        }
        if (window == middle_ && middle_ < maximum_) {
            middle_seconds_ = middle_seconds_ == 0
                                  ? seconds
                                  : middle_seconds_ + 0.25 * (seconds - middle_seconds_);
            if (cooldown_ != 0) { --cooldown_; }
        }

        Sample sample{.seconds = seconds};
        for (const auto count : accepted) {
            sample.tokens += 1U + static_cast<std::uint32_t>(count);
            sample.narrow_tokens += 1U + std::min(narrow_, static_cast<std::uint32_t>(count));
            sample.middle_tokens += 1U + std::min(middle_, static_cast<std::uint32_t>(count));
        }
        recent_[cursor_] = sample;
        cursor_          = (cursor_ + 1U) % recent_.size();
        samples_         = std::min(samples_ + 1U, 2U);
        if (samples_ < 2) { return; }

        const double wide_rate =
            (recent_[0].tokens + recent_[1].tokens) / (recent_[0].seconds + recent_[1].seconds);
        // The same verified prefixes tell us how many tokens K3 could have licensed. Its
        // physical cost comes from recent narrow executions of this cohort, once per batch.
        const double narrow_rate =
            (recent_[0].narrow_tokens + recent_[1].narrow_tokens) / (2.0 * narrow_seconds_);
        const double middle_rate =
            middle_seconds_ > 0 && middle_ < window
                ? (recent_[0].middle_tokens + recent_[1].middle_tokens) / (2.0 * middle_seconds_)
                : 0;
        if (wide_rate < std::max(narrow_rate, middle_rate) * (trial_ ? 1.05 : 1.0)) {
            // K7 is the cheaper stable compromise: require a second losing rolling window
            // before abandoning established residency, rather than churning on one dip.
            if (window == middle_ && middle_ < maximum_ && !trial_ && ++losing_windows_ < 2) {
                return;
            }
            const bool useful_middle = middle_ > narrow_ && middle_ < window &&
                                       recent_[0].middle_tokens + recent_[1].middle_tokens >
                                           recent_[0].narrow_tokens + recent_[1].narrow_tokens &&
                                       (middle_seconds_ == 0 || middle_rate > narrow_rate * 1.05);
            // Repeated false positives at depth three should not cause constant expensive probes.
            if (window == maximum_) {
                cooldown_      = 8U << failed_trials_;
                failed_trials_ = std::min(failed_trials_ + 1U, 2U);
            } else {
                cooldown_ = 4;
            }
            // After a failed maximum-width trial, try the cheaper middle width on recovery.
            preferred_probe_ = middle_;
            if (useful_middle) {
                begin_trial(middle_);
            } else {
                selected_ = narrow_;
                samples_  = 0;
                trial_    = false;
            }
        } else {
            trial_          = false;
            losing_windows_ = 0;
            if (window == maximum_) {
                failed_trials_   = 0;
                preferred_probe_ = maximum_;
            }
        }
    }

private:
    void begin_trial(std::uint32_t width) noexcept {
        selected_       = width;
        samples_        = 0;
        cursor_         = 0;
        trial_          = true;
        losing_windows_ = 0;
    }

    struct Sample {
        double seconds              = 0;
        std::uint32_t tokens        = 0;
        std::uint32_t narrow_tokens = 0;
        std::uint32_t middle_tokens = 0;
    };

    std::uint32_t maximum_         = 1;
    std::uint32_t narrow_          = 1;
    std::uint32_t middle_          = 1;
    std::uint32_t preferred_probe_ = 1;
    std::uint32_t selected_        = 1;
    std::uint64_t cohort_          = 0;
    double narrow_seconds_         = 0;
    double middle_seconds_         = 0;
    std::uint32_t narrow_samples_  = 0;
    std::uint32_t cooldown_        = 0;
    std::uint32_t failed_trials_   = 0;
    std::uint32_t samples_         = 0;
    std::uint32_t losing_windows_  = 0;
    std::size_t cursor_            = 0;
    bool trial_                    = false;
    std::array<Sample, 2> recent_{};
};

} // namespace ninfer::models::qwen3_5::detail
