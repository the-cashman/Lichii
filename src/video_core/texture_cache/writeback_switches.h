// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>

#include "common/common_types.h"
#include "common/settings.h"

namespace VideoCommon::WritebackSwitches {

inline constexpr u64 DuskbloodsNetworkTestProgramId = 0x0400D1D02399A000ULL;
inline constexpr u64 FireEmblemFortunesWeaveProgramId = 0x04007EF01F8AE000ULL;

/// Titles whose dropped GPU-written overlaps are written back, and whose images made over pages
/// such an image left stale load guest memory.
inline constexpr std::array WritebackTitles{
    DuskbloodsNetworkTestProgramId,
    FireEmblemFortunesWeaveProgramId, // sky and bloom noise after menus (#1951)
};

[[nodiscard]] constexpr bool WritebackTitle(u64 program_id) {
    return std::ranges::find(WritebackTitles, program_id) != WritebackTitles.end();
}

[[nodiscard]] inline bool WritebackTitle() {
    return WritebackTitle(Settings::GetCurrentProgramID());
}

[[nodiscard]] inline bool AnyWritebackOn() {
    return WritebackTitle();
}

[[nodiscard]] inline bool UploadStale() {
    return WritebackTitle();
}

} // namespace VideoCommon::WritebackSwitches
