// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Ocarina of Time HDR output (prototype, EDEN_NX2_OOT_HDR, default off).
//
// The game tonemaps its linear scene with a 3x4 colour matrix, bloom^p, exposure and the
// Narkowicz ACES fit, per channel, clamped to [0,1]. With HDR on, that fragment shader is
// replaced by one that keeps the same maths up to a knee and then extends the curve to the
// display's peak (in paper-white units), so the scene behind the HUD keeps its highlights.
// FSR 4 then runs in its HDR mode on linear colour, the 4K composite keeps the HUD at UI
// brightness, and the swapchain is HDR10 (A2B10G10R10 + ST.2084), PQ-encoded at presentation.
//
//   EDEN_NX2_OOT_HDR          0 off (default), 1 on when the surface offers HDR10 (ST.2084
//                             with a 10-bit format); without it the session stays SDR (the
//                             game's tonemap, FSR's SDR mode), so it can be left on
//   EDEN_NX2_OOT_HDR_PEAK     display peak in nits (1500)
//   EDEN_NX2_OOT_HDR_PAPER    game brightness: nits of the game's SDR white (203)
//   EDEN_NX2_OOT_HDR_UI       UI brightness: nits of HUD/menu white (203)
//   EDEN_NX2_OOT_HDR_GAMMA    2.2 (default): decode SDR as a 2.2 display would; 0: sRGB curve
//   EDEN_NX2_OOT_HDR_KNEE     SDR output level where the HDR curve leaves the game's (0.5)
//   EDEN_NX2_OOT_HDR_STATS    1: log the output's luminance percentiles every 600 frames
//   EDEN_NX2_OOT_HDR_DUMP     <file>: write one 4K composite (RGBA16F, encoded) there

#pragma once

#include <span>
#include <string>
#include <vector>

#include "common/common_types.h"

namespace Vulkan::OcarinaHdr {

struct Params {
    bool requested{};  ///< EDEN_NX2_OOT_HDR set and not 0
    float peak_nits{1500.0f};
    float paper_nits{203.0f};
    float ui_nits{203.0f};
    float gamma{2.2f}; ///< 0 = sRGB piecewise decode
    float knee{0.5f};
    bool stats{};
    const char* dump_path{};
};

const Params& Get();

// The mode (HDR or SDR) is chosen once per process, at the first use in the game (the tonemap's
// pipeline, the upscaler), from the latest swapchain: HDR when it is HDR10. Not at boot: the game's
// pipelines load before the window has its real surface (a small startup window that offers SDR
// formats only, even with HDR on), so a tonemap pipeline is not built at boot while undecided.

/// Each swapchain creation: whether the surface offered HDR10 and the swapchain took it, and a
/// description for the log.
void NoteSwapchain(bool hdr10, bool offered, std::string description);
/// Ocarina HDR presents in the latest swapchain: requested, the mode isn't SDR, and the swapchain
/// is HDR10. The presenter then uses that swapchain's format and this prototype's present pass
/// (vk_blit_screen.cpp), not Lichii's native HDR output (#1326).
[[nodiscard]] bool SwapchainHdr();
/// A swapchain may take HDR10 (requested, and the mode isn't SDR already).
[[nodiscard]] bool AllowHdr10Swapchain();

/// HDR tonemap, FSR 4 HDR mode and the HDR composite are in use: requested and the mode is HDR.
/// The first call chooses the mode (logged once).
[[nodiscard]] bool Active();

/// At boot: leave a pipeline with this fragment shader for its first draw, because it is a tonemap
/// the mode (still undecided) changes.
[[nodiscard]] bool DeferBootPipeline(u64 fragment_hash);

/// Fills the spec constants (SpecId 200..) of this prototype's own shaders with the settings.
[[nodiscard]] std::vector<u32> Specialize(std::span<const u32> code);

/// Replaces the game's tonemap fragment shader (by unique hash) with the HDR one, keeping its
/// descriptor bindings. Leaves `code` alone when inactive, for other shaders, or when the
/// shader's interface is not the expected one.
void PatchFragment(u64 hash, size_t stage_index, std::vector<u32>& code);

/// Composite value (what the composite stores) -> nits, as the HDR10 present pass decodes it.
[[nodiscard]] float DecodeToNits(float encoded);

} // namespace Vulkan::OcarinaHdr
