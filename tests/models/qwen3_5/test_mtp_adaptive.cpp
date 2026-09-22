#include "models/qwen3_5/program/speculative/mtp_adaptive.h"

#include <array>
#include <iostream>

namespace {
using namespace ninfer::models::qwen3_5::detail;

struct Fixture {
    MtpAdaptiveBatchController controller;
    MtpAdaptiveSignal signal;
    std::uint32_t available = 3;
    std::uint32_t room      = 4096;
    std::uint64_t cohort    = 1;

    explicit Fixture(std::uint32_t maximum = 15) {
        available = std::min(3U, maximum);
        controller.reset(maximum);
    }

    std::uint32_t width() {
        const std::array<const MtpAdaptiveSignal*, 1> signals{&signal};
        return controller.select(signals, {&room, 1}, cohort);
    }

    std::uint32_t step(std::uint32_t accepts, double seconds, std::uint32_t extent = 15) {
        const auto selected = width();
        const auto proposed =
            static_cast<std::int32_t>(std::min({selected, available, extent, room - 1}));
        const auto accepted =
            static_cast<std::int32_t>(std::min(accepts, static_cast<std::uint32_t>(proposed)));
        signal.observe(proposed, accepted);
        controller.observe_execution(selected, {&proposed, 1}, {&accepted, 1}, seconds);
        available = selected;
        return selected;
    }

    void promote() {
        for (int round = 0; round < 3; ++round) { step(3, 0.03); }
    }
};

int require(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}
} // namespace

int main() {
    int failures = 0;
    {
        Fixture f;
        for (int round = 0; round < 100; ++round) {
            failures += require(f.step(1, 0.03) == 3, "weak prefixes left K3");
        }
        f.promote();
        failures += require(f.width() == 15, "three full prefixes did not jump directly to K15");
        f.step(3, 1.0); // First wide round has only the preceding three-draft chain.
        failures += require(f.width() == 15, "short-chain transition was judged as a failed trial");
        f.step(15, 0.05);
        failures += require(f.width() == 15, "wide trial ended before two full samples");
        f.step(15, 0.05);
        for (int round = 0; round < 32; ++round) {
            failures += require(f.step(15, 0.05) == 15, "profitable K15 lost residency");
        }
        f.step(0, 0.05);
        f.step(0, 0.05);
        failures += require(f.width() == 3, "two collapsed rounds did not return to K3");
    }
    {
        Fixture f;
        f.promote();
        f.step(3, 0.1);
        f.step(3, 0.06);
        f.step(3, 0.06);
        failures += require(f.width() == 3, "unsuccessful wide trial did not contract");
        for (int round = 0; round < 8; ++round) {
            failures += require(f.step(3, 0.03) == 3, "failed trial was immediately repeated");
        }
        failures += require(f.width() == 7, "recovery did not try the cheaper middle width");
        f.step(3, 0.1);
        f.step(3, 0.06);
        f.step(3, 0.06);
        for (int round = 0; round < 4; ++round) {
            failures += require(f.step(3, 0.03) == 3, "middle failure did not cool down");
        }
        failures += require(f.width() == 7, "cheap middle retry inherited the maximum cooldown");
    }
    {
        Fixture f;
        f.promote();
        f.step(3, 0.1);
        f.step(15, 0.2);
        f.step(15, 0.2);
        failures += require(f.width() == 7, "expensive K15 did not try K7");
        f.step(7, 0.2);
        f.step(7, 0.2);
        failures += require(f.width() == 3, "perfect acceptance ignored excessive middle cost");
    }
    {
        Fixture f;
        f.promote();
        f.step(3, 0.1);
        f.step(15, 0.05);
        for (int round = 0; round < 4; ++round) {
            f.step(0, 0.2, 2); // Censored by a cache-publication boundary, not a full K15 failure.
            failures += require(f.width() == 15, "clipped verification ended the trial");
        }
        f.step(15, 0.05);
        failures += require(f.width() == 15, "full evidence after clipping was lost");
        f.room = 1;
        failures += require(f.width() == 3, "output tail retained K15");
        f.room = 4096;
        ++f.cohort;
        failures += require(f.width() == 3, "new cohort inherited wide residency or old costs");
    }
    {
        Fixture f;
        f.promote();
        f.step(3, 0.1);
        f.step(7, 0.09);
        f.step(7, 0.09);
        failures += require(f.width() == 7, "useful medium prefixes fell straight back to K3");
        for (int round = 0; round < 12; ++round) {
            failures += require(f.step(6, 0.04) == 7, "profitable K7 lost residency");
        }
        for (int round = 0; round < 3; ++round) { f.step(7, 0.04); }
        failures += require(f.width() == 15, "strong K7 prefixes did not probe K15");
        f.step(7, 1.0);
        failures += require(f.width() == 15, "K7->K15 chain-building round counted as a failure");
        f.step(15, 0.05);
        f.step(15, 0.05);
        failures += require(f.width() == 15, "profitable maximum did not regain residency");
        f.step(7, 0.05);
        f.step(7, 0.05);
        failures += require(f.width() == 7, "K15 ignored a faster measured K7 alternative");
        f.step(0, 0.04);
        f.step(0, 0.04);
        failures += require(f.width() == 3, "collapsed K7 did not return to reasoning width");
    }
    {
        Fixture f;
        f.promote();
        f.step(3, 0.1);
        f.step(7, 0.09);
        f.step(7, 0.09);
        f.step(6, 0.04);
        f.step(6, 0.04);
        f.step(4, 0.04);
        f.step(4, 0.04);
        failures += require(f.width() == 7, "one losing K7 window caused unnecessary churn");
        f.step(6, 0.04);
        failures += require(f.width() == 7, "recovered K7 failed to keep residency");
        f.step(0, 0.04);
        f.step(0, 0.04);
        f.step(0, 0.04);
        failures += require(f.width() == 3, "persistent K7 collapse did not contract");
    }
    for (const unsigned maximum : {1U, 2U, 3U, 5U, 7U, 10U, 15U}) {
        Fixture f(maximum);
        for (int round = 0; round < 80; ++round) {
            const auto width = f.step(maximum, 0.03);
            failures += require(width == std::min(3U, maximum) || width == std::min(7U, maximum) ||
                                    width == maximum,
                                "controller selected an intermediate or out-of-cap width");
        }
    }
    {
        MtpAdaptiveBatchController controller;
        controller.reset(15);
        std::array<MtpAdaptiveSignal, 2> rows;
        const std::array<const MtpAdaptiveSignal*, 2> signals{&rows[0], &rows[1]};
        const std::array<std::uint32_t, 2> room{4096, 4096};
        const std::array<std::int32_t, 2> narrow{3, 3}, wide{15, 15};
        for (int round = 0; round < 3; ++round) {
            (void)controller.select(signals, room, 1);
            for (auto& row : rows) { row.observe(3, 3); }
            controller.observe_execution(3, narrow, narrow, 0.03);
        }
        failures += require(controller.select(signals, room, 1) == 15,
                            "confident compact batch did not promote");
        controller.observe_execution(15, narrow, narrow, 0.1);
        controller.observe_execution(15, wide, wide, 0.06);
        controller.observe_execution(15, wide, wide, 0.06);
        failures += require(controller.select(signals, room, 1) == 15,
                            "shared execution time was charged per row");
    }
    if (failures != 0) { return 1; }
    std::cout << "mtp_adaptive: PASS\n";
}
