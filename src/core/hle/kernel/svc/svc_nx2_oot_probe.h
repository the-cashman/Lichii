// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <string>
#include <vector>

#include "common/common_types.h"

namespace Core {
class System;
}

namespace Kernel::Svc {

/// Ocarina of Time debug probes (EDEN_NX2_OOT_PROBE, off when unset). patch_manager.cpp replaces
/// each listed main instruction with a branch to a 16-byte cave: svc #0xf4, the instruction, a
/// branch back and a tag word. This SVC logs the registers and memory the probe asks for (rate
/// limited per probe) and a hit count per probe every second. It changes no register.
///
/// EDEN_NX2_OOT_PROBE=<main offset hex>[/item...][#lines per second], comma separated. Items:
/// x0..x30, w0..w30, s0..s31, lr, sp; w:REG+OFF (u32), q:REG+OFF (u64), f:REG+OFF (f32),
/// b:REG+OFF (u8); gw:/gq:/gf:OFF read main+OFF. #0 counts only. Default items: x0 x1 x2 x19 lr.
bool CallNx2OotProbe(Core::System& system, u32 imm);

/// The main offsets EDEN_NX2_OOT_PROBE lists, in order (probe i's cave: Nx2OotProbeCave(i)).
std::vector<u64> Nx2OotProbeOffsets();

// Two runs of the text's zero padding that no built-in uses: 48 + 48 caves of 16 bytes.
constexpr u64 kNx2OotProbeCave = 0x3C46C00;
constexpr u64 kNx2OotProbeCaveEnd = 0x3C46F00;
constexpr u64 kNx2OotProbeCave2 = 0x3C46500; // after FPS Unlock's solver-steps hooks (0x3C46400..0x3C464FC)
constexpr u64 kNx2OotProbeCave2End = 0x3C46800;
constexpr size_t kNx2OotProbeMax = 96;
constexpr u32 kNx2OotProbeSvc = 0xF4;
constexpr u32 kNx2OotProbeTag = 0x0F4E0000;

constexpr u64 Nx2OotProbeCave(size_t i) {
    constexpr size_t first = (kNx2OotProbeCaveEnd - kNx2OotProbeCave) / 16;
    return i < first ? kNx2OotProbeCave + 16 * i : kNx2OotProbeCave2 + 16 * (i - first);
}

} // namespace Kernel::Svc
