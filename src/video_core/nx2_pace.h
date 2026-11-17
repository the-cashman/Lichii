// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// Frame pacing for the Switch 2 path (EDEN_NX2_PACE): the game's presents are capped at a rate
// it can hold (fixed, or adapted from how much time the game has left before each deadline), and
// the presenter releases each frame on an even grid at that rate instead of whenever its
// composite happens to finish. On a VRR display the flips follow the presents, so an even
// release is an even picture; the cost is the time a frame is held (logged as hold).
//
//   EDEN_NX2_PACE=<hz>          fixed rate, e.g. 50
//   EDEN_NX2_PACE=auto[:<hz>]   adaptive, starting at <hz> (default 60), between
//                               EDEN_NX2_PACE_MIN (default 50) and EDEN_NX2_PACE_MAX (default 120)
//   EDEN_NX2_PACE_GUEST=0       don't cap the game's presents (presenter grid only)
//   EDEN_NX2_PACE_HOLD=0        don't hold presents (guest cap only)
//   EDEN_NX2_PACE_DECAY_US=<n>  how fast the presenter's grid moves back toward the frames
//                               after a late one (default 50 us a frame)
//   EDEN_NX2_PACE_MAXHOLD_MS=<n> never hold a frame longer (default 8)

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "common/common_types.h"
#include "common/logging.h"

namespace Tegra::Nx2Pace {

using Clock = std::chrono::steady_clock;

struct Config {
    bool enabled{};
    bool adaptive{};
    double start_hz{60.0};
    double min_hz{50.0};
    double max_hz{120.0};
    bool guest{true};
    bool hold{true};
    s64 decay_ns{50'000};
    s64 max_hold_ns{8'000'000};
};

namespace Detail {
inline double EnvDouble(const char* name, double fallback) {
    const char* text = std::getenv(name);
    return text != nullptr && *text != '\0' ? std::strtod(text, nullptr) : fallback;
}
inline bool EnvFlag(const char* name, bool fallback) {
    const char* text = std::getenv(name);
    return text != nullptr && *text != '\0' ? text[0] != '0' : fallback;
}
inline Config Parse() {
    Config config;
    const char* text = std::getenv("EDEN_NX2_PACE");
    if (text == nullptr || *text == '\0' || std::strcmp(text, "0") == 0 ||
        std::strcmp(text, "off") == 0) {
        return config;
    }
    config.min_hz = std::max(EnvDouble("EDEN_NX2_PACE_MIN", 50.0), 1.0);
    config.max_hz = std::max(EnvDouble("EDEN_NX2_PACE_MAX", 120.0), config.min_hz);
    if (std::strncmp(text, "auto", 4) == 0) {
        config.adaptive = true;
        config.start_hz = text[4] == ':' ? std::strtod(text + 5, nullptr) : 60.0;
        config.start_hz = std::clamp(config.start_hz, config.min_hz, config.max_hz);
    } else {
        config.start_hz = std::strtod(text, nullptr);
        if (config.start_hz < 1.0) {
            return config;
        }
    }
    config.enabled = true;
    config.guest = EnvFlag("EDEN_NX2_PACE_GUEST", true);
    config.hold = EnvFlag("EDEN_NX2_PACE_HOLD", true);
    config.decay_ns = static_cast<s64>(EnvDouble("EDEN_NX2_PACE_DECAY_US", 50.0) * 1000.0);
    config.max_hold_ns = static_cast<s64>(EnvDouble("EDEN_NX2_PACE_MAXHOLD_MS", 8.0) * 1e6);
    LOG_INFO(Render_Vulkan,
             "EDEN_NX2_PACE: {} {:.1f} Hz (range {:.0f}-{:.0f}), guest cap {}, presenter hold {}, "
             "decay {} us/frame, max hold {:.1f} ms",
             config.adaptive ? "adaptive from" : "fixed", config.start_hz, config.min_hz,
             config.max_hz, config.guest ? "on" : "off", config.hold ? "on" : "off",
             config.decay_ns / 1000, static_cast<double>(config.max_hold_ns) / 1e6);
    return config;
}
inline s64 HzToNs(double hz) {
    return static_cast<s64>(1e9 / hz);
}
} // namespace Detail

inline const Config& Get() {
    static const Config config = Detail::Parse();
    return config;
}

/// The current pace interval in ns, 0 while pacing is off.
inline std::atomic<s64>& IntervalStorage() {
    static std::atomic<s64> interval{Get().enabled ? Detail::HzToNs(Get().start_hz) : 0};
    return interval;
}
inline s64 IntervalNs() {
    return IntervalStorage().load(std::memory_order_relaxed);
}
/// The interval the game's presents are capped to (0: not capped by this).
inline s64 GuestIntervalNs() {
    return Get().guest ? IntervalNs() : 0;
}
/// Whether the presenter holds presents to its grid (needs the present thread).
inline bool HoldsPresents() {
    return Get().enabled && Get().hold;
}

/**
 * Chooses the pace rate from the game's slack at its present deadline: the time it had left
 * before the deadline (negative when it arrived late). Once a second: more than 3 % late ->
 * 4 % slower; under 0.5 % late, the 10th percentile of the slack above 10 % of the interval and
 * no slow-down for 3 s -> 2 % faster. Pure; the caller supplies the clock.
 */
class RateController {
public:
    struct Decision {
        s64 interval_ns;
        bool changed;
        double late_share;
        double p10_slack_ms;
        u32 samples;
    };

    std::optional<Decision> Add(Clock::time_point now, s64 slack_ns, s64 interval_ns,
                                double min_hz, double max_hz) {
        if (window_start == Clock::time_point{}) {
            window_start = now;
        }
        slacks.push_back(slack_ns);
        if (now - window_start < std::chrono::seconds(1) || slacks.size() < 20) {
            return std::nullopt;
        }
        const u32 samples = static_cast<u32>(slacks.size());
        const auto late = std::count_if(slacks.begin(), slacks.end(),
                                        [](s64 slack) { return slack < 0; });
        const double late_share = static_cast<double>(late) / samples;
        auto p10 = slacks.begin() + samples / 10;
        std::nth_element(slacks.begin(), p10, slacks.end());
        const s64 p10_slack = *p10;
        slacks.clear();
        window_start = now;
        ++windows_since_down;

        s64 next = interval_ns;
        if (late_share > 0.03) {
            next = static_cast<s64>(interval_ns * 1.04);
            windows_since_down = 0;
        } else if (late_share < 0.005 && windows_since_down >= 3 &&
                   p10_slack > interval_ns / 10) {
            next = static_cast<s64>(interval_ns * 0.98);
        }
        next = std::clamp(next, Detail::HzToNs(max_hz), Detail::HzToNs(min_hz));
        return Decision{next, next != interval_ns, late_share,
                        static_cast<double>(p10_slack) / 1e6, samples};
    }

private:
    Clock::time_point window_start{};
    std::vector<s64> slacks;
    u32 windows_since_down{3};
};

/// The game's present reached the pacer `slack_ns` before its deadline (negative: late).
/// Adapts the rate with EDEN_NX2_PACE=auto. Called from the present service thread only.
inline void OnGuestPresent(Clock::time_point now, s64 slack_ns) {
    const Config& config = Get();
    if (!config.enabled || !config.adaptive) {
        return;
    }
    static RateController controller;
    auto& interval = IntervalStorage();
    const auto decision = controller.Add(now, slack_ns, interval.load(std::memory_order_relaxed),
                                         config.min_hz, config.max_hz);
    if (decision && decision->changed) {
        interval.store(decision->interval_ns, std::memory_order_relaxed);
        LOG_INFO(Render_Vulkan,
                 "EDEN_NX2_PACE: {:.2f} Hz (late {:.1f} %, p10 slack {:.2f} ms over {} presents)",
                 1e9 / static_cast<double>(decision->interval_ns), decision->late_share * 100.0,
                 decision->p10_slack_ms, decision->samples);
    }
}

/**
 * The presenter's release grid. Each frame goes one interval after the previous release, or at
 * once when it is ready later than that (a late frame moves the grid). While frames are early the
 * grid creeps back toward them by `decay_ns` a frame, so after a late frame the added latency
 * drains away instead of staying. A gap of more than two intervals (loading, a pause) restarts
 * the grid. Pure; the caller supplies the times.
 */
class PresentGrid {
public:
    struct Slot {
        Clock::time_point release;
        bool late;
    };

    Slot Next(Clock::time_point ready, s64 interval_ns, s64 decay_ns, s64 max_hold_ns) {
        const std::chrono::nanoseconds interval{interval_ns};
        if (interval_ns <= 0 || !last || ready > *last + 3 * interval) {
            last = ready;
            return {ready, false};
        }
        const auto slot = *last + interval;
        if (ready >= slot) {
            last = ready;
            return {ready, ready > slot};
        }
        s64 hold = std::chrono::duration_cast<std::chrono::nanoseconds>(slot - ready).count();
        hold -= std::min(hold, decay_ns);
        hold = std::min(hold, max_hold_ns);
        last = ready + std::chrono::nanoseconds{hold};
        return {*last, false};
    }

private:
    std::optional<Clock::time_point> last;
};

} // namespace Tegra::Nx2Pace
