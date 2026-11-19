// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include <fmt/format.h>

#include "common/logging.h"
#include "core/arm/arm_interface.h"
#include "core/core.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/k_thread.h"
#include "core/hle/kernel/kernel.h"
#include "core/hle/kernel/svc.h"
#include "core/hle/kernel/svc/svc_nx2_oot_probe.h"
#include "core/hle/kernel/svc_types.h"
#include "core/memory.h"

namespace Kernel::Svc {
namespace {

constexpr u64 kOotProgramId = 0x0400486024DEC000ULL;

enum class Kind : u8 { X, W, S, Lr, Sp, MemW, MemQ, MemF, MemB, GlobW, GlobQ, GlobF, Diff };

struct Item {
    Kind kind{};
    u32 reg{};  // register number (X/W/S, or the base register of Mem*)
    s64 off{};  // Mem*/Glob*/Diff offset
    u32 size{}; // Diff: bytes
    std::string text;
};

// d:REG+OFF:SIZE (hex): the u32 words of [REG+OFF, +SIZE) are compared with the previous hit;
// words that changed are logged as "+offset:old>new", except words that have already changed more
// than kDiffQuiet times (timers, counters), so state that changes rarely stands out.
constexpr u32 kDiffQuiet = 6;
struct DiffState {
    bool primed{};
    std::vector<u32> words;
    std::vector<u16> changes;
};

struct Probe {
    u64 offset{};
    std::vector<Item> items;
    u32 per_second{30};
    std::string name;
};

u32 ParseReg(std::string_view s, bool& ok) {
    ok = false;
    if (s == "lr") {
        ok = true;
        return 30;
    }
    if (s.size() < 2 || (s[0] != 'x' && s[0] != 'w')) {
        return 0;
    }
    const u32 n = static_cast<u32>(std::strtoul(std::string{s.substr(1)}.c_str(), nullptr, 10));
    ok = n <= 30;
    return n;
}

bool ParseItem(std::string_view s, Item& item) {
    item.text = std::string{s};
    const auto colon = s.find(':');
    if (colon == std::string_view::npos) {
        if (s == "lr") {
            item.kind = Kind::Lr;
            return true;
        }
        if (s == "sp") {
            item.kind = Kind::Sp;
            return true;
        }
        if (s.size() >= 2 && (s[0] == 'x' || s[0] == 'w' || s[0] == 's')) {
            item.kind = s[0] == 'x' ? Kind::X : s[0] == 'w' ? Kind::W : Kind::S;
            item.reg = static_cast<u32>(std::strtoul(std::string{s.substr(1)}.c_str(), nullptr, 10));
            return item.reg <= 31;
        }
        return false;
    }
    const std::string_view type = s.substr(0, colon);
    std::string_view rest = s.substr(colon + 1);
    if (type == "d") {
        const auto colon2 = rest.find(':');
        if (colon2 == std::string_view::npos) {
            return false;
        }
        item.kind = Kind::Diff;
        item.size = static_cast<u32>(std::strtoul(std::string{rest.substr(colon2 + 1)}.c_str(), nullptr, 16)) & ~3U;
        rest = rest.substr(0, colon2);
        const auto plus = rest.find_first_of("+-");
        bool ok = false;
        item.reg = ParseReg(rest.substr(0, plus), ok);
        if (plus != std::string_view::npos) {
            const s64 v = static_cast<s64>(std::strtoull(std::string{rest.substr(plus + 1)}.c_str(), nullptr, 16));
            item.off = rest[plus] == '-' ? -v : v;
        }
        return ok && item.size != 0 && item.size <= 0x10000;
    }
    if (type == "gw" || type == "gq" || type == "gf") {
        item.kind = type == "gw" ? Kind::GlobW : type == "gq" ? Kind::GlobQ : Kind::GlobF;
        item.off = static_cast<s64>(std::strtoull(std::string{rest}.c_str(), nullptr, 16));
        return true;
    }
    if (type != "w" && type != "q" && type != "f" && type != "b") {
        return false;
    }
    item.kind = type == "w" ? Kind::MemW : type == "q" ? Kind::MemQ : type == "f" ? Kind::MemF : Kind::MemB;
    const auto plus = rest.find_first_of("+-");
    bool ok = false;
    item.reg = ParseReg(rest.substr(0, plus), ok);
    if (!ok) {
        return false;
    }
    if (plus != std::string_view::npos) {
        const s64 v = static_cast<s64>(std::strtoull(std::string{rest.substr(plus + 1)}.c_str(), nullptr, 16));
        item.off = rest[plus] == '-' ? -v : v;
    }
    return true;
}

std::vector<Probe> ParseProbes() {
    std::vector<Probe> probes;
    const char* env = std::getenv("EDEN_NX2_OOT_PROBE");
    if (env == nullptr || env[0] == 0) {
        return probes;
    }
    std::string_view text{env};
    while (!text.empty()) {
        const auto comma = text.find(',');
        std::string_view one = text.substr(0, comma);
        text = comma == std::string_view::npos ? std::string_view{} : text.substr(comma + 1);
        if (one.empty()) {
            continue;
        }
        Probe probe;
        probe.name = std::string{one};
        const auto hash = one.find('#');
        if (hash != std::string_view::npos) {
            probe.per_second = static_cast<u32>(std::strtoul(std::string{one.substr(hash + 1)}.c_str(), nullptr, 10));
            one = one.substr(0, hash);
        }
        const auto slash = one.find('/');
        probe.offset = std::strtoull(std::string{one.substr(0, slash)}.c_str(), nullptr, 16);
        if (slash != std::string_view::npos) {
            std::string_view items = one.substr(slash + 1);
            while (!items.empty()) {
                const auto s2 = items.find('/');
                Item item;
                if (ParseItem(items.substr(0, s2), item)) {
                    probe.items.push_back(item);
                } else {
                    LOG_WARNING(Kernel_SVC, "EDEN_NX2_OOT_PROBE: bad item {}", items.substr(0, s2));
                }
                items = s2 == std::string_view::npos ? std::string_view{} : items.substr(s2 + 1);
            }
        } else {
            for (const char* d : {"x0", "x1", "x2", "x19", "lr"}) {
                Item item;
                ParseItem(d, item);
                probe.items.push_back(item);
            }
        }
        probes.push_back(std::move(probe));
        if (probes.size() >= kNx2OotProbeMax) {
            break;
        }
    }
    return probes;
}

const std::vector<Probe>& Probes() {
    static const std::vector<Probe> probes = ParseProbes();
    return probes;
}

s64 NowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

struct Counts {
    std::array<u64, kNx2OotProbeMax> hits{};
    std::array<u32, kNx2OotProbeMax> logged{};
};

std::mutex g_mutex;
Counts g_counts{};
s64 g_window_ns{};

std::string Format(Core::Memory::Memory& memory, const Probe& probe, const ThreadContext& ctx,
                   u64 main_base) {
    std::string out;
    const auto reg = [&](u32 n) { return n == 31 ? ctx.sp : n == 30 ? ctx.lr : n == 29 ? ctx.fp : ctx.r[n]; };
    for (const Item& item : probe.items) {
        out += ' ';
        out += item.text;
        out += '=';
        u64 address = 0;
        if (item.kind == Kind::Diff) {
            out.pop_back();
            out.resize(out.size() - item.text.size());
            out.pop_back();
            continue;
        }
        switch (item.kind) {
        case Kind::X:
            out += fmt::format("{:#x}", reg(item.reg));
            continue;
        case Kind::W:
            out += fmt::format("{:#x}", static_cast<u32>(reg(item.reg)));
            continue;
        case Kind::S: {
            const u32 bits = static_cast<u32>(ctx.v[item.reg][0]);
            out += fmt::format("{:.4f}", std::bit_cast<float>(bits));
            continue;
        }
        case Kind::Lr:
            out += fmt::format("main+{:#x}", ctx.lr - main_base);
            continue;
        case Kind::Sp:
            out += fmt::format("{:#x}", ctx.sp);
            continue;
        case Kind::GlobW:
        case Kind::GlobQ:
        case Kind::GlobF:
            address = main_base + static_cast<u64>(item.off);
            break;
        default:
            address = reg(item.reg) + static_cast<u64>(item.off);
            break;
        }
        if (!memory.IsValidVirtualAddressRange(address, 8)) {
            out += "?";
            continue;
        }
        switch (item.kind) {
        case Kind::MemW:
        case Kind::GlobW:
            out += fmt::format("{:#x}", memory.Read32(address));
            break;
        case Kind::MemQ:
        case Kind::GlobQ:
            out += fmt::format("{:#x}", memory.Read64(address));
            break;
        case Kind::MemB:
            out += fmt::format("{:#x}", memory.Read8(address));
            break;
        default:
            out += fmt::format("{:.4f}", std::bit_cast<float>(memory.Read32(address)));
            break;
        }
    }
    return out;
}

std::mutex g_diff_mutex;
std::array<std::vector<DiffState>, kNx2OotProbeMax> g_diffs{};

std::string Diffs(Core::Memory::Memory& memory, size_t index, const Probe& probe,
                  const ThreadContext& ctx) {
    std::string out;
    std::scoped_lock lock{g_diff_mutex};
    auto& states = g_diffs[index];
    states.resize(probe.items.size());
    for (size_t i = 0; i < probe.items.size(); ++i) {
        const Item& item = probe.items[i];
        if (item.kind != Kind::Diff) {
            continue;
        }
        const u64 reg = item.reg == 30 ? ctx.lr : item.reg == 29 ? ctx.fp : ctx.r[item.reg];
        const u64 address = reg + static_cast<u64>(item.off);
        if (!memory.IsValidVirtualAddressRange(address, item.size)) {
            continue;
        }
        std::vector<u32> now(item.size / 4);
        memory.ReadBlock(address, now.data(), item.size);
        DiffState& st = states[i];
        if (!st.primed || st.words.size() != now.size()) {
            st.words = now;
            st.changes.assign(now.size(), 0);
            st.primed = true;
            continue;
        }
        u32 shown = 0;
        for (size_t w = 0; w < now.size(); ++w) {
            if (now[w] == st.words[w]) {
                continue;
            }
            if (st.changes[w] < kDiffQuiet && shown < 48) {
                out += fmt::format(" {}+{:x}:{:x}>{:x}", item.text.substr(0, item.text.rfind(':')),
                                   static_cast<u64>(item.off) + w * 4, st.words[w], now[w]);
                ++shown;
            }
            if (st.changes[w] < 0xFFFF) {
                ++st.changes[w];
            }
            st.words[w] = now[w];
        }
    }
    return out;
}

} // Anonymous namespace

std::vector<u64> Nx2OotProbeOffsets() {
    std::vector<u64> offsets;
    for (const Probe& probe : Probes()) {
        offsets.push_back(probe.offset);
    }
    return offsets;
}

bool CallNx2OotProbe(Core::System& system, u32 imm) {
    if (imm != kNx2OotProbeSvc) {
        return false;
    }
    auto& kernel = system.Kernel();
    auto& process = GetCurrentProcess(kernel);
    if (!process.Is64Bit() || process.GetProgramId() != kOotProgramId) {
        return false;
    }
    auto* const arm = process.GetArmInterface(kernel.CurrentPhysicalCoreIndex());
    if (arm == nullptr) {
        return true;
    }
    ThreadContext ctx{};
    arm->GetContext(ctx);
    auto& memory = process.GetMemory();
    // pc is the instruction after the svc: the tag word is at pc + 8.
    if (!memory.IsValidVirtualAddressRange(ctx.pc + 8, 4)) {
        return true;
    }
    const u32 tag = memory.Read32(ctx.pc + 8);
    const auto& probes = Probes();
    const u32 index = tag & 0xFFFF;
    if ((tag & 0xFFFF0000) != kNx2OotProbeTag || index >= probes.size() || index >= kNx2OotProbeMax) {
        return true;
    }
    const u64 main_base = ctx.pc - 4 - Nx2OotProbeCave(index);
    const Probe& probe = probes[index];
    const s64 now = NowNs();
    bool log_line = false;
    std::string summary;
    {
        std::scoped_lock lock{g_mutex};
        if (g_window_ns == 0) {
            g_window_ns = now;
        }
        if (now - g_window_ns >= 1'000'000'000) {
            for (size_t i = 0; i < probes.size() && i < kNx2OotProbeMax; ++i) {
                if (g_counts.hits[i] != 0 || probes.size() <= 16) {
                    summary += fmt::format(" {:x}={}", probes[i].offset, g_counts.hits[i]);
                }
            }
            g_counts = Counts{};
            g_window_ns = now;
        }
        ++g_counts.hits[index];
        if (g_counts.logged[index] < probe.per_second) {
            ++g_counts.logged[index];
            log_line = true;
        }
    }
    if (!summary.empty()) {
        LOG_INFO(Kernel_SVC, "OoT probe hits/s:{}", summary);
    }
    if (std::ranges::any_of(probe.items, [](const Item& i) { return i.kind == Kind::Diff; })) {
        const std::string diffs = Diffs(memory, index, probe, ctx);
        if (!diffs.empty()) {
            LOG_INFO(Kernel_SVC, "OoT probe {:x} f{} diff:{}", probe.offset,
                     Core::g_nx2_present_count.load(std::memory_order_relaxed), diffs);
        }
    }
    if (log_line) {
        LOG_INFO(Kernel_SVC, "OoT probe {:x} f{} t{}:{}", probe.offset,
                 Core::g_nx2_present_count.load(std::memory_order_relaxed),
                 GetCurrentThread(kernel).GetThreadId(), Format(memory, probe, ctx, main_base));
    }
    return true;
}

} // namespace Kernel::Svc
