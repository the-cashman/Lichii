// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Several A64 Jits on one SharedCodeCache, each on its own thread, as ArmDynarmic64 runs the
// guest cores of one process. Runners compile the same and different blocks, link and return
// through them and take exclusives; a writer rewrites a function they call, invalidates it and
// asks for full clears meanwhile. The cache is small, so it also fills up and clears itself.

#include <array>
#include <atomic>
#include <memory>
#include <optional>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "common/common_types.h"

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#endif
#include "dynarmic/interface/A64/a64.h"
#include "dynarmic/interface/exclusive_monitor.h"
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

namespace {

using namespace Dynarmic;
using Vector = A64::Vector;

constexpr u64 CallerBase = 0x1000;
constexpr u64 MutableBase = 0x2000; // movz x5, #version; ret
constexpr u64 WriterBase = 0x3000;  // bl MutableBase; svc #0
constexpr u64 CounterAddress = 0x8000;
constexpr u64 FuncBase = 0x1000'0000; // func i at FuncBase + 16 * i: x0 += i & 0xFFFF; ret

constexpr size_t Runners = 4;
constexpr u64 Rounds = 4;
constexpr u64 Iterations = 20000;

constexpr u32 Movz(u32 d, u64 imm) { return 0xD2800000 | u32(imm & 0xFFFF) << 5 | d; }
constexpr u32 AddReg(u32 d, u32 n, u32 m) { return 0x8B000000 | m << 16 | n << 5 | d; }
constexpr u32 AddImm(u32 d, u32 n, u32 imm) { return 0x91000000 | imm << 10 | n << 5 | d; }
constexpr u32 SubsImm(u32 d, u32 n, u32 imm) { return 0xF1000000 | imm << 10 | n << 5 | d; }
constexpr u32 CmpReg(u32 n, u32 m) { return 0xEB000000 | m << 16 | n << 5 | 31; }
constexpr u32 Blr(u32 n) { return 0xD63F0000 | n << 5; }
constexpr u32 Bl(s64 offset) { return 0x94000000 | (u32(offset / 4) & 0x3FFFFFF); }
constexpr u32 BCond(s32 offset, u32 cond) { return 0x54000000 | (u32(offset / 4) & 0x7FFFF) << 5 | cond; }
constexpr u32 CbnzW(u32 t, s32 offset) { return 0x35000000 | (u32(offset / 4) & 0x7FFFF) << 5 | t; }
constexpr u32 Ldaxr(u32 t, u32 n) { return 0xC85FFC00 | n << 5 | t; }
constexpr u32 Stlxr(u32 s, u32 t, u32 n) { return 0xC800FC00 | s << 16 | n << 5 | t; }
constexpr u32 Svc(u32 imm) { return 0xD4000001 | imm << 5; }
constexpr u32 Ret = 0xD65F03C0;
constexpr u32 Nop = 0xD503201F;
constexpr u32 CondNe = 1;
constexpr u32 CondLo = 3;

/// Calls func x3, x3 + 16, ... x1 times. After each it calls the mutable function, fails when x5
/// is older than x6 (the version published before the run started) and adds one to the counter.
constexpr std::array<u32, 13> Caller{
    Blr(3),
    Bl(s64(MutableBase) - s64(CallerBase + 4)),
    CmpReg(5, 6),
    BCond(9 * 4, CondLo), // to svc #1
    Ldaxr(9, 8),
    AddImm(9, 9, 1),
    Stlxr(10, 9, 8),
    CbnzW(10, -3 * 4),
    AddImm(3, 3, 16),
    SubsImm(1, 1, 1),
    BCond(-10 * 4, CondNe),
    Svc(0),
    Svc(1),
};

/// Guest memory all Jits see: the mutable function's version and the counter.
struct Guest {
    std::atomic<u64> version{1};   ///< What the mutable function returns now
    std::atomic<u64> published{1}; ///< A version whose invalidation has finished
    std::atomic<u64> counter{0};
};

class Env final : public A64::UserCallbacks {
public:
    explicit Env(Guest& guest_) : guest{guest_} {}

    Guest& guest;
    A64::Jit* jit = nullptr;
    std::optional<u32> svc;
    A64::UserConfig::Counters counters;

    std::optional<std::uint32_t> MemoryReadCode(u64 vaddr) override {
        if (vaddr - CallerBase < Caller.size() * 4) {
            return Caller[(vaddr - CallerBase) / 4];
        }
        if (vaddr == MutableBase) {
            return Movz(5, guest.version.load());
        }
        if (vaddr == MutableBase + 4) {
            return Ret;
        }
        if (vaddr == WriterBase) {
            return Bl(s64(MutableBase) - s64(WriterBase));
        }
        if (vaddr == WriterBase + 4) {
            return Svc(0);
        }
        if (vaddr >= FuncBase) {
            const u64 i = (vaddr - FuncBase) / 16;
            const std::array<u32, 4> func{Movz(2, i), AddReg(0, 0, 2), Ret, Nop};
            return func[(vaddr - FuncBase) / 4 % 4];
        }
        return Svc(2);
    }
    u64 MemoryRead(u64 vaddr, size_t size) override {
        return vaddr == CounterAddress && size == 8 ? guest.counter.load() : 0;
    }
    Vector MemoryRead128(u64) override { return {}; }
    void MemoryWrite(u64, u64, size_t) override { svc = 3; }
    void MemoryWrite128(u64, Vector) override { svc = 3; }
    bool MemoryWriteExclusive64(u64 vaddr, u64 value, u64 expected) override {
        return vaddr == CounterAddress && guest.counter.compare_exchange_strong(expected, value);
    }
    void CallSVC(std::uint32_t imm) override {
        svc = imm;
        jit->HaltExecution();
    }
    void ExceptionRaised(u64, A64::Exception) override {
        svc = 4;
        jit->HaltExecution();
    }
    void AddTicks(u64) override {}
    u64 GetTicksRemaining() override { return 0; }
    u64 GetCNTPCT() override { return 0; }
};

A64::UserConfig Config(Env& env, ExclusiveMonitor& monitor, std::shared_ptr<A64::SharedCodeCache> cache, u8 id) {
    A64::UserConfig config{};
    config.callbacks = &env;
    config.global_monitor = &monitor;
    config.processor_id = id;
    config.counters = &env.counters;
    config.shared_code_cache = std::move(cache);
    config.code_cache_size = 16 * 1024 * 1024;
    // ArmDynarmic64's multicore options: the ones a shared cache needs and those that change code.
    config.enable_cycle_counting = false;
    config.spill_to_free_gpr = true;
    config.scratch_copy_to_free_reg = true;
    config.elide_redundant_zext = true;
    config.fold_address_offsets = true;
    config.omit_block_frame = true;
    config.fast_dispatch_table_bits = 16;
    config.fast_dispatch_set_associative = true;
    config.rsb_size = 32;
    config.compact_block_exits = true;
    config.far_code = true;
    config.reuse_host_flags = true;
    config.halt_check_on_back_edges = true;
    config.host_call_return = true;
    return config;
}

/// Runs until the guest's svc: halts for cache invalidations and clears return early.
void RunToSvc(A64::Jit& jit, Env& env) {
    env.svc.reset();
    while (!env.svc) {
        jit.Run();
    }
}

} // namespace

TEST_CASE("Jits on one code cache run while others compile, invalidate and clear", "[jit_shared_cache]") {
    Guest guest;
    ExclusiveMonitor monitor{Runners + 1};
    const auto cache = A64::MakeSharedCodeCache();

    std::vector<std::unique_ptr<Env>> envs;
    std::vector<std::unique_ptr<A64::Jit>> jits;
    for (size_t i = 0; i <= Runners; ++i) {
        envs.push_back(std::make_unique<Env>(guest));
        jits.push_back(std::make_unique<A64::Jit>(Config(*envs[i], monitor, cache, u8(i))));
        envs[i]->jit = jits[i].get();
    }

    std::atomic<size_t> finished{0};
    std::atomic<u64> bad_sums{0};
    std::atomic<u64> bad_svcs{0};
    std::vector<std::thread> threads;
    for (size_t r = 0; r < Runners; ++r) {
        threads.emplace_back([&, r] {
            A64::Jit& jit = *jits[r];
            Env& env = *envs[r];
            for (u64 round = 0; round < Rounds; ++round) {
                // Runner pairs share their functions; each round moves to new ones.
                const u64 start = (r % 2) * Iterations / 2 + round * Iterations;
                jit.SetRegister(0, 0);
                jit.SetRegister(1, Iterations);
                jit.SetRegister(3, FuncBase + 16 * start);
                jit.SetRegister(6, guest.published.load());
                jit.SetRegister(8, CounterAddress);
                jit.SetPC(CallerBase);
                RunToSvc(jit, env);
                u64 sum = 0;
                for (u64 i = start; i < start + Iterations; ++i) {
                    sum += i & 0xFFFF;
                }
                bad_sums += jit.GetRegister(0) != sum;
                bad_svcs += *env.svc != 0;
            }
            ++finished;
        });
    }

    u64 rewrites = 0;
    u64 stale = 0;
    threads.emplace_back([&] {
        A64::Jit& jit = *jits[Runners];
        Env& env = *envs[Runners];
        while (finished.load() < Runners && guest.version.load() < 0xFFFF) {
            const u64 version = guest.version.load() + 1;
            guest.version.store(version);
            jit.InvalidateCacheRange(MutableBase, 8);
            guest.published.store(version);
            if (++rewrites % 64 == 0) {
                jit.ClearCache();
            }
            jit.SetPC(WriterBase);
            RunToSvc(jit, env);
            stale += jit.GetRegister(5) != version;
            bad_svcs += *env.svc != 0;
        }
    });

    for (std::thread& thread : threads) {
        thread.join();
    }

    u64 clears = 0;
    u64 compiled = 0;
    for (const auto& env : envs) {
        clears += env->counters.cache_clears;
        compiled += env->counters.blocks_compiled;
    }
    INFO("rewrites " << rewrites << ", blocks compiled " << compiled << ", clears " << clears);
    CHECK(bad_sums.load() == 0);
    CHECK(bad_svcs.load() == 0);
    CHECK(stale == 0);
    CHECK(guest.counter.load() == Runners * Rounds * Iterations);
    CHECK(rewrites > 0);
    CHECK(clears > 0);
}
