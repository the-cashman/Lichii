// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <span>
#include <utility>
#include <vector>

#include "common/common_types.h"
#include "common/logging.h"
#include "video_core/macro_turing.h"

// List indexed draws: indexed draws whose index count is GPU-written.
//
// The culling draws of Donkey Kong Bananza write the index count of later draws into five-word
// records (count, instances, first index, base vertex, base instance), and each record is a
// command list: the last five parameters of the indexed indirect draw macro (3D method 0xe50
// there). To parse those words the front of the split waited for the back to run the work that
// writes them, for the host GPU to finish everything submitted and for the pages to be copied
// back, about twice a frame. The five words have the layout of a host indexed indirect draw
// command: the front sends the address of the list instead, and the back draws indirectly from
// the buffer the host GPU wrote.
//
// The registers and the constant buffer words the macro writes from the record (first index,
// base vertex, base instance) get the words guest memory holds. A list is only taken while
// those are one instance from index 0, vertex 0 and instance 0, as the reset draw of the game
// leaves them; mode 2 checks that the GPU-written record still says the same.
namespace Tegra::Nx2ListIndexedDraw {

/// EDEN_NX2_LIST_INDEXED_DRAW, read once: 1 (the default) = indirect draw, 0 = off (the front
/// reads the words back), 2 = off, and every list that would be taken is checked after its
/// words are read back.
[[nodiscard]] inline int Mode() {
    static const int mode = [] {
        const char* const value = std::getenv("EDEN_NX2_LIST_INDEXED_DRAW");
        return value != nullptr ? std::atoi(value) : 1;
    }();
    return mode;
}

/// EDEN_NX2_LIST_INDEXED_DRAW_DIAG: the counters below are logged every 2 s.
[[nodiscard]] inline bool Diagnostics() {
    static const bool enabled = std::getenv("EDEN_NX2_LIST_INDEXED_DRAW_DIAG") != nullptr;
    return enabled || Mode() == 2;
}

// The 3D registers the macro writes (checked against Maxwell3D::Regs where it is used).
inline constexpr u32 VertexIdBase = 0x446;
inline constexpr u32 BaseVertex = 0x50d;
inline constexpr u32 BaseInstance = 0x50e;
inline constexpr u32 DrawEnd = 0x585;
inline constexpr u32 DrawBegin = 0x586;
inline constexpr u32 IndexFirst = 0x5f7;
inline constexpr u32 IndexCount = 0x5f8;
inline constexpr u32 ConstBufferOffset = 0x8e3;
inline constexpr u32 ConstBufferData = 0x8e4;

/// Whether the macro at `start` of `code` is the indexed indirect draw: with the parameters
/// topology, index count, instances, first index, base vertex and base instance it writes the
/// first index, the base vertex and instance (registers, and constant buffer words 0x640 and
/// 0x644), draws `instances` times in an instance loop and writes the bases back to zero.
[[nodiscard]] inline bool IsIndexedDrawMacro(std::span<const u32> code, u32 start) {
    const auto effects = TuringMacroInterpreter::AnalyzeEffects(code, start);
    constexpr std::array<u32, 10> written{
        VertexIdBase, BaseVertex, BaseInstance,      DrawEnd,         DrawBegin,
        IndexFirst,   IndexCount, ConstBufferOffset, ConstBufferData, ConstBufferData + 1};
    if (!effects.methods_known || effects.data_ram ||
        !std::ranges::equal(effects.methods, written)) {
        return false;
    }
    TuringMacroInterpreter probe;
    constexpr std::array<std::array<u32, 6>, 3> samples{{
        {4U, 0x12345678U, 1U, 0x9ABCDEF0U, 0x0F0F0F0FU, 0xF0F0F0F0U},
        {0xFFFFU, 0xFFFFFFFFU, 0U, 1U, 0xFFFFFFFFU, 0U},
        {1U, 0U, 3U, 0U, 0U, 0xFFFFFFFFU},
    }};
    for (const auto& sample : samples) {
        const u32 topology = sample[0], count = sample[1], instances = sample[2];
        const u32 first = sample[3], base_vertex = sample[4], base_instance = sample[5];
        std::vector<std::pair<u32, u32>> expected{
            {IndexFirst, first},         {VertexIdBase, base_vertex},
            {BaseVertex, base_vertex},   {BaseInstance, base_instance},
            {ConstBufferOffset, 0x640U}, {ConstBufferData, base_vertex},
            {ConstBufferData + 1, base_instance}};
        for (u32 instance = 0; instance < instances; ++instance) {
            // Instance id (bits 27:26): First, then Subsequent.
            expected.emplace_back(DrawBegin, instance == 0 ? topology : topology | (1U << 26));
            expected.emplace_back(IndexCount, count);
            expected.emplace_back(DrawEnd, 0U);
        }
        expected.emplace_back(VertexIdBase, 0U);
        expected.emplace_back(BaseVertex, 0U);
        expected.emplace_back(BaseInstance, 0U);
        expected.emplace_back(ConstBufferOffset, 0x640U);
        expected.emplace_back(ConstBufferData, 0U);
        expected.emplace_back(ConstBufferData + 1, 0U);
        std::vector<u32> parameters(sample.begin(), sample.end());
        std::vector<std::pair<u32, u32>> emits;
        const auto result = probe.Execute(
            code, start, parameters, [](u32) { return 0U; },
            [&emits](u32 address, u32 value) {
                emits.emplace_back(address, value);
                return emits.size() < 64;
            });
        if (!result.success || result.consumed != sample.size() || probe.read_registers ||
            emits != expected) {
            return false;
        }
    }
    return true;
}

struct Stats {
    std::atomic<u64> taken{};          ///< Front: lists sent as an address
    std::atomic<u64> refused_record{}; ///< Front: record not one instance from 0, 0, 0
    std::atomic<u64> refused{};        ///< Front: another macro or call shape
    std::atomic<u64> drawn{};          ///< Back: drawn indirectly
    std::atomic<u64> dropped{};        ///< Back: address left at a draw of another kind
    std::atomic<u64> checked{};        ///< Front, mode 2: records read back
    std::atomic<u64> mismatched{};     ///< Front, mode 2: of those, changed past the count
};
inline Stats g_stats;

inline void Count(std::atomic<u64>& counter) {
    counter.fetch_add(1, std::memory_order_relaxed);
}

/// Front thread.
inline void MaybeLog() {
    if (!Diagnostics()) {
        return;
    }
    static auto window = std::chrono::steady_clock::now();
    const auto now = std::chrono::steady_clock::now();
    if (now - window < std::chrono::seconds{2}) {
        return;
    }
    window = now;
    const auto take = [](std::atomic<u64>& counter) {
        return counter.exchange(0, std::memory_order_relaxed);
    };
    const u64 taken = take(g_stats.taken);
    const u64 refused_record = take(g_stats.refused_record);
    const u64 refused = take(g_stats.refused);
    const u64 drawn = take(g_stats.drawn);
    const u64 dropped = take(g_stats.dropped);
    const u64 checked = take(g_stats.checked);
    const u64 mismatched = take(g_stats.mismatched);
    LOG_INFO(HW_GPU,
             "list indexed draw 2s: {} lists taken, {} refused by their record, {} refused "
             "otherwise | back: {} drawn indirectly, {} dropped | check: {} read back, {} "
             "changed past the count",
             taken, refused_record, refused, drawn, dropped, checked, mismatched);
}

} // namespace Tegra::Nx2ListIndexedDraw
