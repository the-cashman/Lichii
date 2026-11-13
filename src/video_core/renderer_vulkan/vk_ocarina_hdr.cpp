// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <string>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <optional>
#include <set>
#include <string_view>
#include <unordered_map>

#include "common/logging.h"
#include "video_core/host_shaders/ocarina_hdr_tonemap_frag_spv.h"
#include "video_core/renderer_vulkan/vk_ocarina_hdr.h"

namespace Vulkan::OcarinaHdr {
namespace {

// The game's tonemap fragment shaders (docs/nx2/oot-network-bypass.md). Only those whose
// translated interface matches (see InterfaceOf) are replaced.
constexpr std::array TONEMAP_HASHES{
    0xbded5ce917183214ULL, // outdoors (verified: Kokiri Forest, noon)
    0x3ac011bd2d772994ULL, // outdoors, another time of day
    0x6cb06edf14be1d79ULL, // indoors
};

// SPIR-V opcodes and decorations used here.
constexpr u32 OP_NAME = 5;
constexpr u32 OP_DECORATE = 71;
constexpr u32 OP_CONSTANT = 43;
constexpr u32 OP_SPEC_CONSTANT = 50;
constexpr u32 OP_VARIABLE = 59;
constexpr u32 OP_ACCESS_CHAIN = 65;
constexpr u32 DECORATION_SPEC_ID = 1;
constexpr u32 DECORATION_BINDING = 33;
constexpr u32 DECORATION_DESCRIPTOR_SET = 34;
constexpr u32 STORAGE_UNIFORM_CONSTANT = 0;
constexpr u32 STORAGE_UNIFORM = 2;
constexpr u32 STORAGE_STORAGE_BUFFER = 12;

// The replacement shader's placeholder bindings (set 7, bindings 10..12).
constexpr u32 PLACEHOLDER_SET = 7;
constexpr std::array<u32, 3> PLACEHOLDER_BINDINGS{10, 11, 12};

enum class Mode : int { Pending, Hdr, Sdr };
std::atomic<Mode> mode{Mode::Pending};
std::atomic<bool> swapchain_hdr{};
std::mutex swapchain_mutex;
bool swapchain_created{};          ///< under swapchain_mutex
std::string swapchain_description; ///< under swapchain_mutex

float EnvFloat(const char* name, float fallback) {
    const char* value = std::getenv(name);
    if (!value || *value == '\0') {
        return fallback;
    }
    char* end{};
    const float parsed = std::strtof(value, &end);
    return end != value && std::isfinite(parsed) ? parsed : fallback;
}

float SrgbDecode(float x) {
    return x <= 0.04045f ? x / 12.92f : std::pow((x + 0.055f) / 1.055f, 2.4f);
}

/// x where the game's ACES fit reaches y (0 < y < 1.03).
float AcesInverse(float y) {
    const double a = 2.51 - 2.43 * y;
    const double b = 0.03 - 0.59 * y;
    const double c = -0.14 * y;
    return static_cast<float>((-b + std::sqrt(b * b - 4 * a * c)) / (2 * a));
}

struct Instruction {
    u32 opcode;
    std::span<const u32> words; ///< including the first word
};

template <typename F>
bool ForEachInstruction(std::span<const u32> code, F&& f) {
    if (code.size() < 5 || code[0] != 0x07230203) {
        return false;
    }
    for (size_t i = 5; i < code.size();) {
        const u32 count = code[i] >> 16;
        if (count == 0 || i + count > code.size()) {
            return false;
        }
        f(i, Instruction{code[i] & 0xffff, code.subspan(i, count)});
        i += count;
    }
    return true;
}

std::string_view LiteralString(std::span<const u32> words) {
    const char* text = reinterpret_cast<const char*>(words.data());
    return {text, strnlen(text, words.size() * sizeof(u32))};
}

struct Binding {
    std::optional<u32> set;
    std::optional<u32> binding;
};

/// The tonemap's resources (c3, fs_tex2_20, fs_tex2_28), or nothing when the shader has others,
/// reads constant buffer words past 15, or lacks the ACES fit's constants.
std::optional<std::array<Binding, 3>> InterfaceOf(std::span<const u32> code) {
    std::unordered_map<u32, std::string_view> names;
    std::unordered_map<u32, Binding> bindings;
    std::unordered_map<u32, u32> constants;
    std::set<u32> resources;
    std::vector<std::span<const u32>> chains;
    if (!ForEachInstruction(code, [&](size_t, const Instruction& inst) {
            const auto w = inst.words;
            switch (inst.opcode) {
            case OP_NAME:
                if (w.size() >= 3) names[w[1]] = LiteralString(w.subspan(2));
                break;
            case OP_DECORATE:
                if (w.size() >= 4 && w[2] == DECORATION_BINDING) bindings[w[1]].binding = w[3];
                if (w.size() >= 4 && w[2] == DECORATION_DESCRIPTOR_SET) bindings[w[1]].set = w[3];
                break;
            case OP_CONSTANT:
                if (w.size() == 4) constants[w[2]] = w[3];
                break;
            case OP_VARIABLE:
                if (w.size() >= 4 && (w[3] == STORAGE_UNIFORM || w[3] == STORAGE_UNIFORM_CONSTANT ||
                                      w[3] == STORAGE_STORAGE_BUFFER)) {
                    resources.insert(w[2]);
                }
                break;
            case OP_ACCESS_CHAIN:
                chains.push_back(w);
                break;
            default:
                break;
            }
        })) {
        return std::nullopt;
    }
    static constexpr std::array<std::string_view, 3> wanted{"c3", "fs_tex2_20", "fs_tex2_28"};
    std::array<u32, 3> ids{};
    for (size_t i = 0; i < wanted.size(); ++i) {
        const auto it = std::ranges::find_if(names, [&](const auto& n) { return n.second == wanted[i]; });
        if (it == names.end() || !resources.contains(it->first)) {
            return std::nullopt;
        }
        ids[i] = it->first;
    }
    if (resources.size() != wanted.size()) {
        return std::nullopt;
    }
    // c3[0][index]: every index a constant below 16.
    for (const auto chain : chains) {
        if (chain.size() >= 6 && chain[3] == ids[0]) {
            const auto index = constants.find(chain[5]);
            if (index == constants.end() || index->second >= 16) {
                return std::nullopt;
            }
        }
    }
    for (const float aces : {2.43000007f, 0.589999974f, 0.140000001f, 2.50999999f, 0.0299999993f}) {
        const u32 bits = std::bit_cast<u32>(aces);
        if (std::ranges::none_of(constants, [bits](const auto& c) { return c.second == bits; })) {
            return std::nullopt;
        }
    }
    std::array<Binding, 3> result{};
    for (size_t i = 0; i < ids.size(); ++i) {
        result[i] = bindings[ids[i]];
        if (!result[i].set || !result[i].binding) {
            return std::nullopt;
        }
    }
    return result;
}

} // Anonymous namespace

const Params& Get() {
    static const Params params = [] {
        Params p;
        const char* mode = std::getenv("EDEN_NX2_OOT_HDR");
        const std::string_view value = mode ? mode : "";
        p.requested = !value.empty() && value != "0";
        p.peak_nits = std::clamp(EnvFloat("EDEN_NX2_OOT_HDR_PEAK", 1500.0f), 100.0f, 10000.0f);
        p.paper_nits = std::clamp(EnvFloat("EDEN_NX2_OOT_HDR_PAPER", 203.0f), 40.0f, p.peak_nits);
        p.ui_nits = std::clamp(EnvFloat("EDEN_NX2_OOT_HDR_UI", 203.0f), 40.0f, p.peak_nits);
        p.gamma = EnvFloat("EDEN_NX2_OOT_HDR_GAMMA", 2.2f);
        if (p.gamma != 0.0f) {
            p.gamma = std::clamp(p.gamma, 1.8f, 2.6f);
        }
        p.knee = std::clamp(EnvFloat("EDEN_NX2_OOT_HDR_KNEE", 0.5f), 0.1f, 0.9f);
        p.stats = EnvFloat("EDEN_NX2_OOT_HDR_STATS", 0.0f) != 0.0f;
        p.dump_path = std::getenv("EDEN_NX2_OOT_HDR_DUMP");
        if (p.requested) {
            LOG_INFO(Render_Vulkan,
                     "Ocarina HDR requested: peak {} nits, game white {} nits, UI white {} nits, "
                     "gamma {}, knee {}",
                     p.peak_nits, p.paper_nits, p.ui_nits,
                     p.gamma == 0.0f ? "sRGB" : std::to_string(p.gamma), p.knee);
        }
        return p;
    }();
    return params;
}

void NoteSwapchain(bool hdr10, bool offered, std::string description) {
    std::scoped_lock lock{swapchain_mutex};
    swapchain_created = true;
    swapchain_hdr = hdr10;
    swapchain_description = std::move(description);
    if (Get().requested && mode.load() == Mode::Hdr && !hdr10 && !offered) {
        static bool warned = false;
        if (!std::exchange(warned, true)) {
            LOG_WARNING(Render_Vulkan,
                        "Ocarina HDR: the surface stopped offering HDR10 ({}); highlights clip "
                        "until it does again",
                        swapchain_description);
        }
    }
}

bool SwapchainHdr() {
    // Only while this prototype presents: an HDR10 swapchain of Lichii's native HDR output
    // (#1326, hdr_mode) without EDEN_NX2_OOT_HDR, or after the session chose SDR, is #1326's.
    return Get().requested && mode.load() != Mode::Sdr && swapchain_hdr.load();
}

bool AllowHdr10Swapchain() {
    return Get().requested && mode.load() != Mode::Sdr;
}

bool Active() {
    if (!Get().requested) {
        return false;
    }
    if (const Mode current = mode.load(); current != Mode::Pending) {
        return current == Mode::Hdr;
    }
    std::scoped_lock lock{swapchain_mutex};
    if (mode.load() == Mode::Pending) {
        // The game uses its tonemap or the upscaler: the window is the real one by now.
        const bool hdr = swapchain_created && swapchain_hdr.load();
        mode = hdr ? Mode::Hdr : Mode::Sdr;
        if (hdr) {
            LOG_INFO(Render_Vulkan, "Ocarina HDR: HDR mode (swapchain {})", swapchain_description);
        } else {
            LOG_INFO(Render_Vulkan,
                     "Ocarina HDR: SDR mode, the surface offers no HDR10 (swapchain {}): the "
                     "game's tonemap and FSR's SDR mode",
                     swapchain_created ? swapchain_description : std::string{"not created yet"});
        }
    }
    return mode.load() == Mode::Hdr;
}

bool DeferBootPipeline(u64 fragment_hash) {
    if (!Get().requested || mode.load() != Mode::Pending ||
        std::ranges::find(TONEMAP_HASHES, fragment_hash) == TONEMAP_HASHES.end()) {
        return false;
    }
    static std::mutex log_mutex;
    static std::set<u64> logged;
    std::scoped_lock lock{log_mutex};
    if (logged.insert(fragment_hash).second) {
        LOG_INFO(Render_Vulkan,
                 "Ocarina HDR: pipelines with tonemap {:016x} are built at their first draw, once "
                 "the mode is chosen",
                 fragment_hash);
    }
    return true;
}

std::vector<u32> Specialize(std::span<const u32> code) {
    const auto& p = Get();
    // Peak in the tonemap's (game-linear, paper-white = 1) units, so that after the gamma
    // correction at presentation the brightest value lands on the display's peak.
    const float ratio = p.peak_nits / p.paper_nits;
    const float peak_linear = p.gamma == 0.0f
                                  ? ratio
                                  : SrgbDecode(std::pow(ratio, 1.0f / p.gamma));
    const std::unordered_map<u32, float> values{
        {200, peak_linear},
        {201, AcesInverse(p.knee)},
        {202, p.paper_nits / p.ui_nits},
        {203, p.gamma},
        {204, p.ui_nits},
        {205, p.peak_nits},
    };
    std::vector<u32> result(code.begin(), code.end());
    std::unordered_map<u32, u32> spec_ids; // result id -> SpecId
    ForEachInstruction(result, [&](size_t, const Instruction& inst) {
        if (inst.opcode == OP_DECORATE && inst.words.size() >= 4 &&
            inst.words[2] == DECORATION_SPEC_ID) {
            spec_ids[inst.words[1]] = inst.words[3];
        }
    });
    std::vector<std::pair<size_t, u32>> writes;
    ForEachInstruction(result, [&](size_t offset, const Instruction& inst) {
        if (inst.opcode == OP_SPEC_CONSTANT && inst.words.size() == 4) {
            const auto id = spec_ids.find(inst.words[2]);
            if (id != spec_ids.end()) {
                const auto value = values.find(id->second);
                if (value != values.end()) {
                    writes.emplace_back(offset + 3, std::bit_cast<u32>(value->second));
                }
            }
        }
    });
    for (const auto& [offset, bits] : writes) {
        result[offset] = bits;
    }
    return result;
}

void PatchFragment(u64 hash, size_t stage_index, std::vector<u32>& code) {
    if (stage_index != 4 || std::ranges::find(TONEMAP_HASHES, hash) == TONEMAP_HASHES.end() ||
        !Active()) {
        return;
    }
    static std::mutex log_mutex;
    static std::set<u64> logged;
    const bool first = [hash] {
        std::scoped_lock lock{log_mutex};
        return logged.insert(hash).second;
    }();
    const auto interface = InterfaceOf(code);
    if (!interface) {
        if (first) {
            LOG_WARNING(Render_Vulkan, "Ocarina HDR: tonemap {:016x} has an unexpected interface; left as is", hash);
        }
        return;
    }
    std::vector<u32> replacement = Specialize(OCARINA_HDR_TONEMAP_FRAG_SPV);
    u32 patched = 0;
    std::unordered_map<u32, u32> placeholder_of; // variable id -> index into interface
    ForEachInstruction(replacement, [&](size_t, const Instruction& inst) {
        if (inst.opcode == OP_DECORATE && inst.words.size() >= 4 &&
            inst.words[2] == DECORATION_BINDING) {
            const auto it = std::ranges::find(PLACEHOLDER_BINDINGS, inst.words[3]);
            if (it != PLACEHOLDER_BINDINGS.end()) {
                placeholder_of[inst.words[1]] =
                    static_cast<u32>(it - PLACEHOLDER_BINDINGS.begin());
            }
        }
    });
    ForEachInstruction(replacement, [&](size_t offset, const Instruction& inst) {
        if (inst.opcode != OP_DECORATE || inst.words.size() < 4) {
            return;
        }
        const auto slot = placeholder_of.find(inst.words[1]);
        if (slot == placeholder_of.end()) {
            return;
        }
        const Binding& original = (*interface)[slot->second];
        if (inst.words[2] == DECORATION_BINDING) {
            replacement[offset + 3] = *original.binding;
            ++patched;
        } else if (inst.words[2] == DECORATION_DESCRIPTOR_SET &&
                   inst.words[3] == PLACEHOLDER_SET) {
            replacement[offset + 3] = *original.set;
            ++patched;
        }
    });
    if (patched != 6) {
        if (first) {
            LOG_ERROR(Render_Vulkan, "Ocarina HDR: replacement shader bindings not found ({}); tonemap {:016x} left",
                      patched, hash);
        }
        return;
    }
    if (first) {
        LOG_INFO(Render_Vulkan, "Ocarina HDR: tonemap {:016x} replaced (cbuf set {} binding {}, textures {} {})",
                 hash, *(*interface)[0].set, *(*interface)[0].binding, *(*interface)[1].binding,
                 *(*interface)[2].binding);
    }
    code = std::move(replacement);
}

float DecodeToNits(float encoded) {
    const auto& p = Get();
    const float c = std::max(encoded, 0.0f);
    const float linear = p.gamma == 0.0f ? SrgbDecode(c) : std::pow(c, p.gamma);
    return std::min(linear * p.ui_nits, p.peak_nits);
}

} // namespace Vulkan::OcarinaHdr
