// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstdlib>
#include <cstring>
#include <iterator>
#include <span>
#include <string_view>
#include <vector>

#include "common/common_types.h"
#include "common/nx2_ab.h"

/// Draw-record cache (EDEN_NX2_DRAW_RECORD): a draw's full resolve (pipeline, image views and
/// samplers, buffer bindings, descriptor payload, render targets) keyed by everything the resolve
/// reads that isn't data, so a later draw with the same key can take the stored result instead of
/// resolving again. The backend side is renderer_vulkan/vk_draw_record.h.
///
/// EDEN_NX2_DRAW_RECORD:
///   unset or 0  off: no work, nothing allocated (one predictable branch per draw and frame)
///   dry         compute the key next to the full resolve and compare the stored outcome with the
///               one the resolve gave; counters only, rendering is unchanged
///   verify, 1   not implemented yet: run as dry (logged once)
/// EDEN_NX2_DRAW_RECORD_REPORT=<frames>  report window (default 60 frames)
/// EDEN_NX2_DRAW_RECORD_FRAMES=1         one line per frame as well
/// EDEN_NX2_DRAW_RECORD_HOLES=<n>        key hole samples logged (default 24)
namespace VideoCommon::Nx2DrawRecord {

enum class Mode : u8 { Off, Dry, Verify, On };

[[nodiscard]] inline Mode ParseMode(const char* value) {
    if (value == nullptr || value[0] == '\0' || std::string_view{value} == "0") {
        return Mode::Off;
    }
    const std::string_view text{value};
    if (text == "dry") {
        return Mode::Dry;
    }
    if (text == "verify") {
        return Mode::Verify;
    }
    return Mode::On;
}

/// EDEN_NX2_DRAW_RECORD as an in-session A/B switch (common/nx2_ab.h): "ab[:arm[:mode]]" runs
/// the mode (default 1) in that arm and is off in the others.
[[nodiscard]] inline const Common::Nx2Ab::Switch& ConfiguredSwitch() {
    // Off unless set (EDEN_NX2_DRAW_RECORD=1 turns the record path on). In-session A/B on the
    // Ocarina village (launcher env, no PGO) it cost 0.56-0.62 fps (GPUBack +0.27-0.34 ms a
    // frame) on this branch and on main-rumble; neutral in Mario Kart World.
    static const Common::Nx2Ab::Switch sw = [] {
        const char* const value = std::getenv("EDEN_NX2_DRAW_RECORD");
        return Common::Nx2Ab::Parse(value != nullptr ? value : "0");
    }();
    return sw;
}

/// The configured mode (what an A/B switch runs in its arm). Read once.
[[nodiscard]] inline Mode GetMode() {
    static const Mode mode = [] {
        const Common::Nx2Ab::Switch& sw = ConfiguredSwitch();
        if (sw.kind == Common::Nx2Ab::Switch::Kind::Off) {
            return Mode::Off;
        }
        return ParseMode(sw.value.empty() ? "1" : sw.value.c_str());
    }();
    return mode;
}

/// The mode now: the configured one, or Off in the arms an A/B switch is off in.
[[nodiscard]] inline Mode CurrentMode() {
    const Common::Nx2Ab::Switch& sw = ConfiguredSwitch();
    return sw.kind != Common::Nx2Ab::Switch::Kind::Ab || sw.Active() ? GetMode() : Mode::Off;
}

[[nodiscard]] inline bool Enabled() {
    static const bool enabled = GetMode() != Mode::Off;
    return enabled;
}

[[nodiscard]] inline u64 EnvNumber(const char* name, u64 fallback);

/// Whether the dry run's measurements run (dry, verify, or EDEN_NX2_DRAW_RECORD_STATS=1 with
/// =1): outcome hashes, payload tracking, miss attribution, binding captures.
[[nodiscard]] inline bool Stats() {
    static const bool stats =
        GetMode() == Mode::Dry || GetMode() == Mode::Verify ||
        (GetMode() == Mode::On && EnvNumber("EDEN_NX2_DRAW_RECORD_STATS", 0) != 0);
    return stats;
}

[[nodiscard]] inline u64 EnvNumber(const char* name, u64 fallback) {
    const char* const value = std::getenv(name);
    return value != nullptr && value[0] != '\0' ? std::strtoull(value, nullptr, 10) : fallback;
}

/// One 64-bit lane of the key and outcome hashes: a multiply-xorshift step per value.
[[nodiscard]] constexpr u64 Mix(u64 hash, u64 value) noexcept {
    hash ^= value + 0x9E3779B97F4A7C15ULL + (hash << 6) + (hash >> 2);
    hash *= 0xBF58476D1CE4E5B9ULL;
    return hash ^ (hash >> 31);
}

[[nodiscard]] inline u64 MixBytes(u64 hash, const void* data, size_t bytes) noexcept {
    const auto* const ptr = static_cast<const u8*>(data);
    size_t offset = 0;
    // Four independent lanes for long ranges (register blocks), joined at the end.
    if (bytes >= 32) {
        u64 lanes[4]{hash, hash ^ 0x9E3779B97F4A7C15ULL, hash ^ 0xC2B2AE3D27D4EB4FULL,
                     hash ^ 0x165667B19E3779F9ULL};
        for (; offset + 32 <= bytes; offset += 32) {
            for (size_t lane = 0; lane < 4; ++lane) {
                u64 word;
                std::memcpy(&word, ptr + offset + lane * 8, sizeof(word));
                lanes[lane] = (lanes[lane] ^ word) * 0xBF58476D1CE4E5B9ULL;
            }
        }
        hash = Mix(Mix(Mix(Mix(hash, lanes[0]), lanes[1]), lanes[2]), lanes[3]);
    }
    for (; offset + sizeof(u64) <= bytes; offset += sizeof(u64)) {
        u64 word;
        std::memcpy(&word, ptr + offset, sizeof(word));
        hash = Mix(hash, word);
    }
    if (offset < bytes) {
        u64 word = 0;
        std::memcpy(&word, ptr + offset, bytes - offset);
        hash = Mix(hash, word);
    }
    return Mix(hash, bytes);
}

struct Key {
    u64 lo{};
    u64 hi{};
    constexpr bool operator==(const Key&) const noexcept = default;
};

struct KeyHash {
    size_t operator()(const Key& key) const noexcept {
        return static_cast<size_t>(key.lo ^ (key.hi * 0xC2B2AE3D27D4EB4FULL));
    }
};

/// The texture and image bindings a Configure used, for the outcome compare: the number of views,
/// per view (handle << 32 | view id), then each sampler id. Called by GraphicsPipeline while
/// records are on.
template <typename Views, typename Samplers>
void CaptureBindings(std::vector<u64>& out, const Views& views, const Samplers& samplers) {
    out.clear();
    out.push_back(static_cast<u64>(std::size(views)));
    for (const auto& view : views) {
        out.push_back((u64{view.index} << 32) | u64{view.id.index});
    }
    for (const auto& sampler : samplers) {
        out.push_back(u64{sampler.index});
    }
}

} // namespace VideoCommon::Nx2DrawRecord
