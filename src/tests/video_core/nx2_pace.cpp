// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

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

TEST_CASE("A long gap restarts the grid", "[nx2_pace]") {
    PresentGrid grid;
    grid.Next(At(0), 20 * Ms, 0, 8 * Ms);
    const auto slot = grid.Next(At(500 * Ms), 20 * Ms, 0, 8 * Ms);
    REQUIRE(!slot.late);
    REQUIRE(Since(slot.release) == 500 * Ms);
}

TEST_CASE("The rate controller slows down when late and speeds up with headroom", "[nx2_pace]") {
    RateController controller;
    s64 interval = 1'000'000'000 / 60;
    s64 t = 0;
    // 10 % late presents: 4 % slower after the first second.
    std::optional<RateController::Decision> decision;
    for (int i = 0; !decision; ++i, t += interval) {
        decision = controller.Add(At(t), i % 10 == 0 ? -Ms : Ms, interval, 50.0, 120.0);
    }
    REQUIRE(decision->changed);
    REQUIRE(decision->interval_ns > interval);
    interval = decision->interval_ns;
    // Then lots of slack: no speed-up for three windows, then 2 % faster each window.
    int changes = 0;
    for (int i = 0; i < 60 * 6; ++i, t += interval) {
        if (const auto d = controller.Add(At(t), interval / 2, interval, 50.0, 120.0);
            d && d->changed) {
            REQUIRE(d->interval_ns < interval);
            interval = d->interval_ns;
            ++changes;
        }
    }
    REQUIRE(changes >= 2);
    REQUIRE(changes <= 5);
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
