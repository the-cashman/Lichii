// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <cstring>
#include <map>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "common/common_types.h"

namespace Loader::FpsTarget {

/// A game's target frame rate: the global pointer at global_offset points to an object that
/// holds an int32 frame rate at field_offset. Games step their logic by 60 / target and present
/// by display / target, so the field is both their speed and their present interval.
struct Target {
    u64 global_offset{};
    u32 field_offset{};
    u32 readers{};
    u32 stores_of_thirty{};
};

namespace detail {

inline u32 ReadWord(std::span<const u8> image, u64 offset) {
    u32 value{};
    std::memcpy(&value, image.data() + offset, sizeof(value));
    return value;
}

constexpr bool IsAdrp(u32 word) {
    return (word & 0x9F000000) == 0x90000000;
}

constexpr bool IsLdrX(u32 word) {
    return (word & 0xFFC00000) == 0xF9400000;
}

/// Whether the instruction may write the general register. Stores, branches and FP/SIMD data
/// processing do not; fmov to a general register is the exception.
constexpr bool WritesGeneral(u32 word, u32 reg) {
    if ((word & 0x0A000000) == 0x08000000) { // Loads and stores.
        if (((word >> 22) & 1) == 0 || ((word >> 26) & 1) != 0) {
            return false; // A store, or a load into an FP/SIMD register.
        }
        return (word & 31) == reg;
    }
    if ((word & 0x1C000000) == 0x14000000) { // Branches, system instructions.
        return false;
    }
    if ((word & 0x1F000000) == 0x1E000000 && ((word >> 29) & 1) == 0 &&
        (word & 0xFFFFFC00) != 0x1E260000 && (word & 0xFFFFFC00) != 0x9E660000) {
        return false;
    }
    return (word & 31) == reg;
}

inline u64 AdrpTarget(u64 pc, u32 word) {
    s64 pages = static_cast<s64>(((word >> 5) & 0x7FFFF) << 2 | ((word >> 29) & 3));
    if ((pages & (s64{1} << 20)) != 0) {
        pages -= s64{1} << 21;
    }
    return static_cast<u64>(static_cast<s64>(pc & ~u64{0xFFF}) + pages * 0x1000);
}

/// The page an ADRP within eight words before `pc` wrote to `reg`, with nothing else writing the
/// register in between.
inline std::optional<u64> PageOf(std::span<const u8> image, u64 pc, u32 reg, u64 text_offset) {
    for (u32 i = 1; i <= 8; ++i) {
        if (pc < text_offset + i * sizeof(u32)) {
            break;
        }
        const u64 at = pc - i * sizeof(u32);
        const u32 word = ReadWord(image, at);
        if (IsAdrp(word) && (word & 31) == reg) {
            return AdrpTarget(at, word);
        }
        if (WritesGeneral(word, reg)) {
            return std::nullopt;
        }
    }
    return std::nullopt;
}

/// The global whose value `base` holds, for `ldr xB, [xA]` and its ADRP/ADD or GOT form.
inline std::optional<u64> GlobalOf(std::span<const u8> image, u64 pc, u32 base, u64 text_offset,
                                   u64 data_offset) {
    for (u32 i = 1; i <= 10; ++i) {
        if (pc < text_offset + i * sizeof(u32)) {
            break;
        }
        const u64 at = pc - i * sizeof(u32);
        const u32 word = ReadWord(image, at);
        if ((word & 31) == base && IsLdrX(word)) {
            if (((word >> 10) & 0xFFF) != 0) {
                return std::nullopt;
            }
            const u32 address_reg = (word >> 5) & 31;
            for (u32 j = 1; j <= 8; ++j) {
                if (at < text_offset + j * sizeof(u32)) {
                    break;
                }
                const u64 maker = at - j * sizeof(u32);
                const u32 next = ReadWord(image, maker);
                if ((next & 0xFFC00000) == 0x91000000 && (next & 31) == address_reg &&
                    ((next >> 5) & 31) == address_reg) {
                    const auto page = PageOf(image, maker, address_reg, text_offset);
                    if (!page) {
                        return std::nullopt;
                    }
                    const u64 global = *page + ((next >> 10) & 0xFFF);
                    if (global >= data_offset && global + sizeof(u64) <= image.size()) {
                        return global;
                    }
                    return std::nullopt;
                }
                if (IsLdrX(next) && (next & 31) == address_reg &&
                    ((next >> 5) & 31) == address_reg) {
                    const auto page = PageOf(image, maker, address_reg, text_offset);
                    if (!page) {
                        return std::nullopt;
                    }
                    const u64 entry = *page + ((next >> 10) & 0xFFF) * 8;
                    if (entry + sizeof(u64) > image.size()) {
                        return std::nullopt;
                    }
                    u64 value{};
                    std::memcpy(&value, image.data() + entry, sizeof(value));
                    // Not relocated yet: a module-relative address is still in the slot.
                    if (value >= data_offset && value + sizeof(u64) <= image.size()) {
                        return value;
                    }
                    return std::nullopt;
                }
                if (WritesGeneral(next, address_reg)) {
                    return std::nullopt;
                }
            }
            return std::nullopt;
        }
        if (WritesGeneral(word, base)) {
            return std::nullopt;
        }
    }
    return std::nullopt;
}

} // namespace detail

/// Finds the target frame rate field. Its readers load it into a float register, convert it and
/// divide 60.0 by it (ldr sN, [xB, #F]; scvtf; fdiv), and the game sets it with an immediate 30
/// or 60. A field with at least eight readers and a store of 30 is the target; exactly one field
/// must qualify. The search names no build: it is the same pattern nxnative's fps_target.py uses,
/// which found this field in every main it checked and nothing in any other game.
inline std::optional<Target> Find(std::span<const u8> image, u64 text_offset, u64 text_size,
                                  u64 data_offset) {
    constexpr u32 MinReaders = 8;
    constexpr u32 SixtyBits = 0x52A84E00; // movz wK, #0x4270, lsl #16 (60.0f)

    if (text_size % sizeof(u32) != 0 || text_offset > image.size() ||
        text_size > image.size() - text_offset) {
        return std::nullopt;
    }
    const u64 text_end = text_offset + text_size;
    std::map<std::pair<u64, u32>, std::vector<u64>> readers;
    std::map<std::pair<u64, u32>, std::vector<u32>> stores;

    for (u64 pc = text_offset; pc + sizeof(u32) <= text_end; pc += sizeof(u32)) {
        const u32 word = detail::ReadWord(image, pc);

        // ldr sT, [xB, #F]
        if ((word & 0xFFC00000) == 0xBD400000) {
            const u32 target_reg = word & 31;
            const u32 base = (word >> 5) & 31;
            const u32 field = ((word >> 10) & 0xFFF) * sizeof(u32);

            // scvtf sC, sT
            std::optional<u32> converted;
            u64 conv = 0;
            for (u32 i = 1; i <= 6; ++i) {
                const u64 at = pc + i * sizeof(u32);
                if (at + sizeof(u32) > text_end) {
                    break;
                }
                const u32 next = detail::ReadWord(image, at);
                if ((next & 0xFFFFFFE0) == (0x5E21D800 | (target_reg << 5))) {
                    converted = next & 31;
                    conv = at;
                    break;
                }
            }
            if (!converted) {
                continue;
            }

            // fdiv sD, sM, sC (dividend sM = 60.0f, divisor sC = the converted target)
            u64 div = 0;
            for (u32 i = 1; i <= 5; ++i) {
                const u64 at = conv + i * sizeof(u32);
                if (at + sizeof(u32) > text_end) {
                    break;
                }
                const u32 next = detail::ReadWord(image, at);
                if ((next & 0xFFE0FC00) == 0x1E201800 && ((next >> 16) & 31) == *converted) {
                    div = at;
                    break;
                }
            }
            if (div == 0) {
                continue;
            }
            const u32 dividend = (detail::ReadWord(image, div) >> 5) & 31;

            // fmov sM, wK
            u64 fmov = 0;
            for (u32 i = 1; i <= 12; ++i) {
                if (div < text_offset + i * sizeof(u32)) {
                    break;
                }
                const u64 at = div - i * sizeof(u32);
                const u32 next = detail::ReadWord(image, at);
                if ((next & 0xFFFFFC1F) == (0x1E270000 | dividend)) {
                    fmov = at;
                    break;
                }
                if (detail::WritesGeneral(next, dividend)) {
                    break;
                }
            }
            if (fmov == 0) {
                continue;
            }

            // movz wK, #0x4270, lsl #16
            const u32 constant_reg = (detail::ReadWord(image, fmov) >> 5) & 31;
            bool is_sixty = false;
            for (u32 i = 1; i <= 8; ++i) {
                if (fmov < text_offset + i * sizeof(u32)) {
                    break;
                }
                const u64 at = fmov - i * sizeof(u32);
                const u32 next = detail::ReadWord(image, at);
                if ((next & 0xFFFFFFE0) == SixtyBits && (next & 31) == constant_reg) {
                    is_sixty = true;
                    break;
                }
                if (detail::WritesGeneral(next, constant_reg)) {
                    break;
                }
            }
            if (!is_sixty) {
                continue;
            }

            if (const auto global =
                    detail::GlobalOf(image, pc, base, text_offset, data_offset)) {
                readers[{*global, field}].push_back(pc);
            }
            continue;
        }

        // str wK, [xB, #F]
        if ((word & 0xFFC00000) == 0xB9000000) {
            const u32 value_reg = word & 31;
            const u32 base = (word >> 5) & 31;
            const u32 field = ((word >> 10) & 0xFFF) * sizeof(u32);

            // movz wK, #30 or #60
            std::optional<u32> value;
            for (u32 i = 1; i <= 8; ++i) {
                if (pc < text_offset + i * sizeof(u32)) {
                    break;
                }
                const u64 at = pc - i * sizeof(u32);
                const u32 next = detail::ReadWord(image, at);
                if ((next & 0xFF800000) == 0x52800000 && (next & 31) == value_reg &&
                    ((next >> 21) & 1) == 0) {
                    value = (next >> 5) & 0xFFFF;
                    break;
                }
                if (detail::WritesGeneral(next, value_reg)) {
                    break;
                }
            }
            if (!value || (*value != 30 && *value != 60)) {
                continue;
            }
            if (const auto global =
                    detail::GlobalOf(image, pc, base, text_offset, data_offset)) {
                stores[{*global, field}].push_back(*value);
            }
        }
    }

    std::optional<Target> found;
    int qualifying = 0;
    for (const auto& [key, pcs] : readers) {
        if (pcs.size() < MinReaders) {
            continue;
        }
        const auto store = stores.find(key);
        if (store == stores.end()) {
            continue;
        }
        if (std::ranges::find(store->second, 30U) == store->second.end()) {
            continue;
        }
        ++qualifying;
        found = Target{
            .global_offset = key.first,
            .field_offset = key.second,
            .readers = static_cast<u32>(pcs.size()),
            .stores_of_thirty = static_cast<u32>(std::ranges::count(store->second, 30U)),
        };
    }
    if (qualifying != 1) {
        return std::nullopt;
    }
    return found;
}

} // namespace Loader::FpsTarget
