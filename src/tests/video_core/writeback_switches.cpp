// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "video_core/texture_cache/writeback_switches.h"

TEST_CASE("Overlap write-backs are on for the tested titles only",
          "[video_core][texture_cache][writeback]") {
    using namespace VideoCommon::WritebackSwitches;
    REQUIRE(WritebackTitle(DuskbloodsNetworkTestProgramId));
    REQUIRE(WritebackTitle(FireEmblemFortunesWeaveProgramId));
    REQUIRE_FALSE(WritebackTitle(0));
    // Ocarina of Time: write-backs regress Castle Town (#2038), and #818's stale-upload skip is
    // its frame rate.
    REQUIRE_FALSE(WritebackTitle(0x0400486024DEC000ULL));
}
