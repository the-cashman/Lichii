// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>

#include <catch2/catch_test_macros.hpp>

#include "video_core/nx2_pace.h"

using Tegra::Nx2Pace::Clock;
using Tegra::Nx2Pace::PresentGrid;
using Tegra::Nx2Pace::RateController;

namespace {
constexpr Clock::time_point Start{std::chrono::seconds{1}};
constexpr s64 Ms = 1'000'000;
Clock::time_point At(s64 ns) {
    return Start + std::chrono::nanoseconds{ns};
}
s64 Since(Clock::time_point t) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(t - Start).count();
}
} // Anonymous namespace

TEST_CASE("The present grid releases jittered frames evenly", "[nx2_pace]") {
    PresentGrid grid;
    constexpr s64 interval = 20 * Ms;
    // Frames ready on a 20 ms grid with up to 3 ms of delay: once the grid has seen the late
    // phase, releases are 20 ms apart minus the decay.
    const s64 jitter[] = {0, 3 * Ms, 1 * Ms, 2 * Ms, 0, 3 * Ms, 1 * Ms, 0};
    s64 last = -1;
    for (int i = 0; i < 40; ++i) {
        const auto slot = grid.Next(At(i * interval + jitter[i % 8]), interval, 0, 8 * Ms);
        if (i > 1 && last >= 0) {
            REQUIRE(Since(slot.release) - last == interval);
        }
        last = Since(slot.release);
    }
}

TEST_CASE("A late frame moves the grid and the decay drains it", "[nx2_pace]") {
    PresentGrid grid;
    constexpr s64 interval = 20 * Ms;
    REQUIRE(Since(grid.Next(At(0), interval, 100'000, 8 * Ms).release) == 0);
    const auto late = grid.Next(At(30 * Ms), interval, 100'000, 8 * Ms);
    REQUIRE(late.late);
    REQUIRE(Since(late.release) == 30 * Ms);
    // Back on the old phase (40, 60, ...): held 10 ms less 0.1 ms a frame, at most 8 ms.
    const auto held = grid.Next(At(40 * Ms), interval, 100'000, 8 * Ms);
    REQUIRE(!held.late);
    REQUIRE(Since(held.release) == 48 * Ms);
    s64 hold = 8 * Ms;
    for (int i = 3; i < 200; ++i) {
        const auto slot = grid.Next(At(i * interval), interval, 100'000, 8 * Ms);
        const s64 now_hold = Since(slot.release) - i * interval;
        REQUIRE(now_hold <= hold);
        hold = now_hold;
    }
    REQUIRE(hold == 0);
}

TEST_CASE("The margin keeps the grid behind the frames", "[nx2_pace]") {
    PresentGrid grid;
    constexpr s64 interval = 20 * Ms;
    // Frames on time: the hold grows to the 4 ms margin by 0.1 ms a frame and stays.
    for (int i = 0; i < 100; ++i) {
        grid.Next(At(i * interval), interval, 100'000, 8 * Ms, 4 * Ms);
    }
    const auto slot = grid.Next(At(100 * interval), interval, 100'000, 8 * Ms, 4 * Ms);
    REQUIRE(Since(slot.release) - 100 * interval == 4 * Ms);
    // A frame 3 ms late still goes on the grid.
    const auto late = grid.Next(At(101 * interval + 3 * Ms), interval, 100'000, 8 * Ms, 4 * Ms);
    REQUIRE(!late.late);
    REQUIRE(Since(late.release) - Since(slot.release) <= interval + 100'000);
}

TEST_CASE("A long gap restarts the grid", "[nx2_pace]") {
    PresentGrid grid;
    grid.Next(At(0), 20 * Ms, 0, 8 * Ms);
    const auto slot = grid.Next(At(500 * Ms), 20 * Ms, 0, 8 * Ms);
    REQUIRE(!slot.late);
    REQUIRE(Since(slot.release) == 500 * Ms);
}

TEST_CASE("The rate controller backs off below where it saturated and probes slowly",
          "[nx2_pace]") {
    RateController controller;
    s64 interval = 1'000'000'000 / 60;
    s64 t = 0;
    // The pipeline holds 55 Hz: faster rates make every present late.
    constexpr double capacity = 55.0;
    double max_seen_after_saturation = 0.0;
    int saturations = 0;
    for (int i = 0; i < 60 * 120; ++i, t += interval) {
        const double hz = 1e9 / static_cast<double>(interval);
        const s64 slack = hz > capacity ? -Ms : 10 * Ms;
        if (const auto d = controller.Add(At(t), slack, interval, 50.0, 120.0); d) {
            if (d->late_share > 0.03) {
                ++saturations;
                REQUIRE(d->interval_ns > interval);
            }
            interval = d->interval_ns;
            if (saturations > 0 && i > 60 * 20) {
                max_seen_after_saturation = std::max(max_seen_after_saturation, 1e9 / interval);
            }
        }
    }
    // Starts at 60 (late), drops below 55 and stays near it: few re-saturations in 2 minutes.
    REQUIRE(saturations >= 1);
    REQUIRE(saturations <= 5);
    REQUIRE(max_seen_after_saturation < 55.5);
    REQUIRE(1e9 / interval > 52.0);
}

TEST_CASE("The rate controller stays within its range", "[nx2_pace]") {
    RateController controller;
    s64 interval = 1'000'000'000 / 50;
    s64 t = 0;
    for (int i = 0; i < 50 * 10; ++i, t += interval) {
        if (const auto d = controller.Add(At(t), -Ms, interval, 50.0, 120.0)) {
            interval = d->interval_ns;
        }
    }
    REQUIRE(interval == 1'000'000'000 / 50);
}
