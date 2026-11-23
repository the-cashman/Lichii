// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <optional>

#include "common/common_types.h"

namespace Service {

// The clock a video decoder session (nx2:vdecsvc) drops late frames by: the pts the movie's
// audio has reached. Times are host microseconds; pause_end is when emulation last resumed.
class Nx2MovieClock {
public:
    enum class ReleaseTiming { FollowClient, AlignAtStart };

    // A longer gap between acquires is a pause, a longer lag a jump in the stream; both restart
    // the clock.
    static constexpr s64 PauseGap = 1'000'000;
    static constexpr s64 MaxCatchUp = 3'000'000;

    void Reset() {
        offset.reset();
    }

    // Minimum number of client presents before a held frame can align the playback clock.
    static constexpr u64 PacedPresents = 2;

    // The client acquired the frame at pts, at present count presents. The clock starts from the
    // first acquire and runs in real time from there. A frame acquired ahead of the clock doesn't
    // move it: Kirby Air Riders takes a frame before it shows it and holds it until its own clock
    // reaches it, and following those acquires ran the clock ahead of the picture, so 563 of 588
    // frames came out late.
    void Acquired(s64 now, u64 pts, s64 pause_end, u64 presents) {
        const s64 acquired_offset = now - static_cast<s64>(pts);
        if (!offset || now - last_acquire > PauseGap || pause_end > last_acquire ||
            pts < last_acquired_pts || acquired_offset - *offset > MaxCatchUp) {
            offset = acquired_offset;
            first_release = true;
        }
        last_acquire = now;
        last_acquired_pts = pts;
        last_acquire_presents = presents;
    }

    // Kirby can hold its first frames before starting its own clock. Ocarina can also give an
    // older image back several presents late during uninterrupted playback. Use its first
    // handoff to align the start, then keep elapsed time through those later stalls.
    void Released(s64 now, u64 presents, ReleaseTiming timing = ReleaseTiming::FollowClient) {
        if (offset && (first_release || timing == ReleaseTiming::FollowClient) &&
            presents - last_acquire_presents >= PacedPresents) {
            offset = now - static_cast<s64>(last_acquired_pts);
        }
        first_release = false;
    }

    // The pts the movie's audio has reached, unless the client hasn't acquired a frame yet or for
    // a while (paused; the next acquire restarts the clock).
    std::optional<u64> Due(s64 now, s64 pause_end) const {
        if (!offset) {
            return std::nullopt;
        }
        const s64 due = now - *offset;
        if (due < 0 || now - last_acquire > PauseGap || pause_end > last_acquire) {
            return std::nullopt;
        }
        return static_cast<u64>(due);
    }

    // The pts of the frame the client acquired last.
    u64 LastAcquiredPts() const {
        return last_acquired_pts;
    }

private:
    std::optional<s64> offset; ///< Host time minus pts of the frame the clock started from
    s64 last_acquire{};
    u64 last_acquired_pts{};
    u64 last_acquire_presents{};
    bool first_release{};
};

} // namespace Service
