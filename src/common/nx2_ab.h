// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <string>

#include "common/common_types.h"

// In-session A/B arms: one live run alternates between k configurations at frame boundaries, so
// a single boot gives many interleaved windows of each arm.
//   EDEN_NX2_AB_PERIOD=<frames>|<ms>ms  phase length (frames = composites on GPUBack); off unless set
//   EDEN_NX2_AB_ARMS=<k>                 arms (default 2); arm 0 = every ab switch off
//   EDEN_NX2_AB_START=<frames>|gate      first phase after that many frames, or once the
//                                        EDEN_NX2_PROBE_GATE file exists (gpuprobe.py creates it
//                                        when the route's measured segment starts; its removal
//                                        ends the phases). Default: 0.
//   EDEN_NX2_AB_SETTLE=<frames>          also log a "mark" line that many frames into each phase,
//                                        so the analysis can drop the frames right after a flip
// Phase order per cycle of k phases: 0..k-1 on even cycles, k-1..0 on odd ones (linear drift
// cancels in the per-cycle comparison). Phases flip on GPUBack right after a composite: draws and
// dispatches after it belong to the next frame. Each flip logs one line ("nx2 AB phase ...") with
// the phase, arm, cycle, frames, CLOCK_MONOTONIC ns and the CPU ns of GPUBack, GPU (front),
// VulkanWorker, the guest core threads and the process (oot-bench/abphase.py reads them).
// A switch opts in with Parse(): its env value "ab" is on in arm 1 only, "ab:<arm>" in that arm,
// "ab:<arm>:<value>" gives <value> in that arm (Value() returns "" elsewhere). Only switches that
// are safe to change between two frames on the back may opt in.
namespace Common::Nx2Ab {

extern std::atomic<u32> g_arm;
extern std::atomic<u32> g_phase;

/// Counts flips: a switch keeping state that is only maintained while it is on (a cache) can
/// compare it with the value it saw last and drop that state when it changed.
inline u32 PhaseSerial() {
    return g_phase.load(std::memory_order_relaxed);
}

/// The current arm (0 = base). A single relaxed load.
inline u32 Arm() {
    return g_arm.load(std::memory_order_relaxed);
}
/// Any arm but the base.
inline bool On() {
    return Arm() != 0;
}

struct Switch {
    enum class Kind : u8 { Off, On, Ab };
    Kind kind{Kind::Off};
    u32 arm{};
    std::string value; ///< the env value (On) or what follows "ab:<arm>:" (Ab)

    /// Whether the switch is on now.
    bool Active() const {
        return kind == Kind::On || (kind == Kind::Ab && Arm() == arm);
    }
    /// Its value now ("" when off).
    const std::string& Value() const {
        static const std::string empty;
        return Active() ? value : empty;
    }
};

/// Parses a switch's env value: null/empty = Off, "ab[:arm[:value]]" = Ab, else On.
Switch Parse(const char* env_value);

bool Enabled();
/// GPUBack, after each composite.
void OnFrame();

} // namespace Common::Nx2Ab
