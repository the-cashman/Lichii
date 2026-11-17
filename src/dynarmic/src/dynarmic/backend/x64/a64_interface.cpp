// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

/* This file is part of the dynarmic project.
 * Copyright (c) 2016 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include <algorithm>
#include <atomic>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <thread>
#include <vector>

#include <boost/icl/interval_set.hpp>
#include "common/assert.h"
#include "dynarmic/common/fp/fpcr.h"
#include "dynarmic/common/llvm_disassemble.h"
#include <bit>

#include "dynarmic/backend/x64/a64_emit_x64.h"
#include "dynarmic/backend/x64/a64_jitstate.h"
#include "dynarmic/backend/x64/block_of_code.h"
#include "dynarmic/backend/x64/devirtualize.h"
#include "dynarmic/backend/x64/exclusive_monitor_friend.h"
#include "dynarmic/backend/x64/jitstate_info.h"
#include "dynarmic/common/atomic.h"
#include "dynarmic/frontend/A64/translate/a64_translate.h"
#include "dynarmic/interface/A64/a64.h"
#include "dynarmic/ir/basic_block.h"
#include "dynarmic/ir/opt_passes.h"

namespace Dynarmic::A64 {

using namespace Backend::X64;

static RunCodeCallbacks GenRunCodeCallbacks(A64::UserCallbacks* cb, CodePtr (*LookupBlock)(void* lookup_block_arg), void* arg, const A64::UserConfig& conf) {
    return RunCodeCallbacks{
        ArgCallback(LookupBlock, reinterpret_cast<u64>(arg)).ArgFromJitState(offsetof(A64JitState, jit_impl), reinterpret_cast<u64>(arg)),
        ArgCallback(Devirtualize<&A64::UserCallbacks::AddTicks>(cb)),
        ArgCallback(Devirtualize<&A64::UserCallbacks::GetTicksRemaining>(cb)),
        conf.enable_cycle_counting,
        HostCallReturn(conf) ? std::optional<u32>{u32(offsetof(A64JitState, host_stack_base))} : std::nullopt,
    };
}

static std::function<void(BlockOfCode&)> GenRCP(const A64::UserConfig& conf) {
    return [conf](BlockOfCode& code) {
        if (conf.page_table) {
            code.mov(code.r14, std::bit_cast<u64>(conf.page_table));
        }
        if (conf.fastmem_pointer) {
            code.mov(code.r13, *conf.fastmem_pointer);
        }
        if (HostCallReturn(conf)) {
            // A RET compares its target with the frame's [rsp + ABI_SHADOW_SPACE]; with no guest
            // frame that is the first StackLayout word (the spills live in the JIT state), which
            // holds a value no location descriptor has (bits 57-58 are never set in one).
            code.mov(code.qword[code.rsp + ABI_SHADOW_SPACE], -1);
            code.mov(code.qword[code.ABI_JIT_PTR + offsetof(A64JitState, host_stack_base)], code.rsp);
            code.lea(code.rax, code.ptr[code.rsp - HostCallFrameSize * HostCallMaxDepth]);
            code.mov(code.qword[code.ABI_JIT_PTR + offsetof(A64JitState, host_stack_limit)], code.rax);
        }
    };
}

// Largest power of two <= conf.rsb_size, between 8 and A64JitState::RSB_SIZE, minus one.
static size_t RsbPtrMask(const A64::UserConfig& conf) {
    size_t size = A64JitState::RSB_SIZE;
    while (size > 8 && size > conf.rsb_size) {
        size >>= 1;
    }
    return size - 1;
}

static Optimization::PolyfillOptions GenPolyfillOptions(const BlockOfCode& code) {
    return Optimization::PolyfillOptions{
        .sha256 = !code.HasHostFeature(HostFeature::SHA),
        .vector_multiply_widen = true,
    };
}

/// The code, block map and links of one or more Jits (UserConfig::shared_code_cache). Code runs
/// while other Jits compile: lookups take `mutex` shared, emitting and invalidating take it
/// exclusively, and links change with one atomic store. A dropped block's code stays until a full
/// clear, which waits until no Jit runs code. Each Jit keeps its fast dispatch table and RSB and
/// empties them when `epoch` moves, and marks in its own `running` flag whether it runs code, so
/// entering and leaving touch no cache line other Jits write.
struct SharedCodeCache final {
    /// Builds the cache for the first Jit; later Jits must emulate the same process.
    void Attach(A64JitState& jit_state, const std::atomic<bool>& running, const UserConfig& jit_conf, CodePtr (*lookup_block)(void*), void* lookup_block_arg) {
        std::unique_lock lock{mutex};
        if (!block_of_code) {
            conf.emplace(jit_conf);
            conf->shared_code_cache.reset();  // the cache must not own itself
            block_of_code.emplace(GenRunCodeCallbacks(conf->callbacks, lookup_block, lookup_block_arg, *conf), JitStateInfo{jit_state, RsbPtrMask(*conf)}, conf->code_cache_size, GenRCP(*conf));
            emitter.emplace(*block_of_code, *conf, nullptr);
            polyfill_options = GenPolyfillOptions(*block_of_code);
            if (jit_conf.fastmem_pointer) {
                emitter->exception_handler.SetFastmemCallback([this](u64 rip) {
                    // Faults come from JIT code, which never runs while its thread holds `mutex`.
                    std::unique_lock fault_lock{mutex};
                    const size_t blocks = emitter->block_descriptors.size();
                    const auto result = emitter->FastmemCallback(rip);
                    if (emitter->block_descriptors.size() != blocks) {
                        DroppedBlocksLocked();
                    }
                    return result;
                });
            }
        } else {
            ASSERT(jit_conf.page_table == conf->page_table && jit_conf.fastmem_pointer == conf->fastmem_pointer);
            ASSERT(!conf->enable_cycle_counting && conf->compact_block_exits && HostCallReturn(*conf));
        }
        jit_states.push_back(&jit_state);
        running_flags.push_back(&running);
    }

    void Detach(A64JitState& jit_state, const std::atomic<bool>& running) {
        std::unique_lock lock{mutex};
        std::erase(jit_states, &jit_state);
        std::erase(running_flags, &running);
    }

    /// Marks a Jit as running code; waits out a requested full clear first. The flag is stored
    /// before pending_clear is read and TryClear reads the flags after pending_clear is set (all
    /// sequentially consistent), so either the Jit sees the clear or the clear sees the Jit.
    void Enter(std::atomic<bool>& running, UserConfig::Counters* counters) {
        while (true) {
            running.store(true);
            if (!pending_clear.load()) {
                return;
            }
            running.store(false);
            TryClear(counters);
            if (pending_clear.load()) {
                std::this_thread::yield();
            }
        }
    }

    void Leave(std::atomic<bool>& running, UserConfig::Counters* counters) {
        running.store(false);
        if (pending_clear.load()) {
            TryClear(counters);
        }
    }

    void RequestClearLocked() {
        pending_clear.store(true);
        HaltAllLocked();
    }

    /// Blocks are gone: Jits must drop their RSB and fast dispatch entries. `entering` is not
    /// halted: it is about to run and drops its entries before it does.
    void DroppedBlocksLocked(const A64JitState* entering = nullptr) {
        epoch.fetch_add(1);
        HaltAllLocked(entering);
    }

    /// Drops the blocks of invalidated guest code. Runs when the asking Jit is out of JIT code:
    /// with one Jit nothing runs the code; with more, links change atomically and the dropped
    /// code stays until a full clear, so a Jit still in it finishes as before the invalidation.
    void ApplyRangesLocked(const A64JitState* entering = nullptr) {
        if (pending_ranges.empty()) {
            return;
        }
        if (emitter->InvalidateCacheRanges(pending_ranges)) {
            DroppedBlocksLocked(entering);
        }
        pending_ranges.clear();
    }

    void HaltAllLocked(const A64JitState* except = nullptr) {
        for (A64JitState* const jit_state : jit_states) {
            if (jit_state == except) {
                continue;
            }
            Atomic::Or(&jit_state->halt_reason, static_cast<u32>(HaltReason::CacheInvalidation));
        }
    }

    std::shared_mutex mutex;
    std::optional<UserConfig> conf;
    std::optional<BlockOfCode> block_of_code;
    std::optional<A64EmitX64> emitter;
    Optimization::PolyfillOptions polyfill_options;
    std::vector<A64JitState*> jit_states;
    std::vector<const std::atomic<bool>*> running_flags;
    std::atomic<u64> epoch{0};
    /// Bumped by every range invalidation: a block translated across one is translated again.
    std::atomic<u64> invalidations{0};
    std::atomic<bool> pending_clear{false};
    boost::icl::interval_set<u64> pending_ranges;

private:
    void TryClear(UserConfig::Counters* counters) {
        std::unique_lock lock{mutex};
        if (pending_clear.load() && std::none_of(running_flags.begin(), running_flags.end(), [](const std::atomic<bool>* running) { return running->load(); })) {
            ClearLocked(counters);
        }
    }

public:
    void ClearLocked(UserConfig::Counters* counters) {
        if (counters) {
            ++counters->cache_clears;
        }
        // ponytail: the cleared code stays resident until it is overwritten; madvise the tail
        // past the write pointer if clears ever show up in long sessions.
        block_of_code->ClearCache();
        emitter->ClearCache();
        pending_ranges.clear();
        epoch.fetch_add(1);
        pending_clear.store(false);
    }
};

std::shared_ptr<SharedCodeCache> MakeSharedCodeCache() {
    return std::make_shared<SharedCodeCache>();
}

struct Jit::Impl final {
public:
    Impl(Jit*, UserConfig conf)
        : conf(conf)
        , cache(conf.shared_code_cache ? conf.shared_code_cache : MakeSharedCodeCache())
    {
        ASSERT(conf.page_table_address_space_bits >= 12 && conf.page_table_address_space_bits <= 64);
        cache->Attach(jit_state, running, conf, &GetCurrentBlockThunk, this);
        fast_dispatch_table.resize(cache->emitter->fast_dispatch_table_size);
        epoch = cache->epoch.load();
        InitJitStatePointers();
    }

    ~Impl() {
        cache->Detach(jit_state, running);
    }

    HaltReason Run() {
        ASSERT(!is_executing);
        Enter();
        is_executing = true;
        // TODO: Check code alignment
        const CodePtr current_code_ptr = [this] {
            // RSB optimization
            const u32 new_rsb_ptr = (jit_state.rsb_ptr - 1) & u32(RsbPtrMask(conf));
            if (jit_state.GetUniqueHash() == jit_state.rsb_location_descriptors[new_rsb_ptr]) {
                jit_state.rsb_ptr = new_rsb_ptr;
                return CodePtr(jit_state.rsb_codeptrs[new_rsb_ptr]);
            }
            // Entries after an SVC: this Jit's fast dispatch table, filled here as the dispatcher
            // fills it, so most need no lookup (and no lock) in the code cache.
            const auto lookup = cache->emitter->fast_dispatch_table_lookup;
            if (!lookup) {
                return CodePtr((uintptr_t(GetCurrentBlock()) + 15) & ~uintptr_t(15));
            }
            const u64 location = jit_state.GetUniqueHash();
            A64EmitX64::FastDispatchEntry* const set = lookup(location, fast_dispatch_table.data());
            const size_t ways = cache->emitter->fast_dispatch_set_size;
            for (size_t way = 0; way < ways; ++way) {
                if (set[way].location_descriptor == location) {
                    return CodePtr(set[way].code_ptr);
                }
            }
            const CodePtr code_ptr = GetCurrentBlock();
            std::copy_backward(set, set + ways - 1, set + ways);
            set[0] = {location, code_ptr};
            return CodePtr((uintptr_t(code_ptr) + 15) & ~uintptr_t(15));
        }();
        const HaltReason hr = cache->block_of_code->RunCode(&jit_state, current_code_ptr);
        is_executing = false;
        cache->Leave(running, conf.counters);
        return hr;
    }

    HaltReason Step() {
        ASSERT(!is_executing);
        Enter();
        is_executing = true;
        const HaltReason hr = cache->block_of_code->StepCode(&jit_state, GetCurrentSingleStep());
        is_executing = false;
        cache->Leave(running, conf.counters);
        return hr;
    }

    void ClearCache() {
        std::unique_lock lock{cache->mutex};
        cache->RequestClearLocked();
    }

    void InvalidateCacheRange(u64 start_address, size_t length) {
        std::unique_lock lock{cache->mutex};
        const auto end_address = static_cast<u64>(start_address + length - 1);
        cache->pending_ranges.add(boost::icl::discrete_interval<u64>::closed(start_address, end_address));
        cache->invalidations.fetch_add(1);
        cache->HaltAllLocked();
    }

    void Reset() {
        ASSERT(!is_executing);
        jit_state = {};
        InitJitStatePointers();
    }

    void HaltExecution(HaltReason hr) {
        Atomic::Or(&jit_state.halt_reason, static_cast<u32>(hr));
    }

    void ClearHalt(HaltReason hr) {
        Atomic::And(&jit_state.halt_reason, ~static_cast<u32>(hr));
    }

    u64 GetSP() const {
        return jit_state.sp;
    }

    void SetSP(u64 value) {
        jit_state.sp = value;
    }

    u64 GetPC() const {
        return jit_state.pc;
    }

    void SetPC(u64 value) {
        jit_state.pc = value;
    }

    u64 GetRegister(size_t index) const {
        if (index == 31)
            return GetSP();
        return jit_state.reg.at(index);
    }

    void SetRegister(size_t index, u64 value) {
        if (index == 31)
            return SetSP(value);
        jit_state.reg.at(index) = value;
    }

    std::array<u64, 31> GetRegisters() const {
        return jit_state.reg;
    }

    void SetRegisters(const std::array<u64, 31>& value) {
        jit_state.reg = value;
    }

    Vector GetVector(size_t index) const {
        return {jit_state.vec.at(index * 2), jit_state.vec.at(index * 2 + 1)};
    }

    void SetVector(size_t index, Vector value) {
        jit_state.vec.at(index * 2) = value[0];
        jit_state.vec.at(index * 2 + 1) = value[1];
    }

    std::array<Vector, 32> GetVectors() const {
        std::array<Vector, 32> ret;
        static_assert(sizeof(ret) == sizeof(jit_state.vec));
        std::memcpy(ret.data(), jit_state.vec.data(), sizeof(jit_state.vec));
        return ret;
    }

    void SetVectors(const std::array<Vector, 32>& value) {
        static_assert(sizeof(value) == sizeof(jit_state.vec));
        std::memcpy(jit_state.vec.data(), value.data(), sizeof(jit_state.vec));
    }

    u32 GetFpcr() const {
        return jit_state.GetFpcr();
    }

    void SetFpcr(u32 value) {
        jit_state.SetFpcr(value);
    }

    u32 GetFpsr() const {
        return jit_state.GetFpsr();
    }

    void SetFpsr(u32 value) {
        jit_state.SetFpsr(value);
    }

    u32 GetPstate() const {
        return jit_state.GetPstate();
    }

    void SetPstate(u32 value) {
        jit_state.SetPstate(value);
    }

    void ClearExclusiveState() {
        jit_state.exclusive_state = 0;
    }

    bool IsExecuting() const {
        return is_executing;
    }

    /// #1225 warm-up: compiles the block at `location` into the (shared) code cache unless it is
    /// there, never clearing the cache, leaving `keep_free` bytes. Not counted as an execution
    /// compile (conf.counters->blocks_compiled).
    PrecompileResult Precompile(std::uint64_t location, std::size_t keep_free) {
        ASSERT(!is_executing);
        static constexpr size_t MINIMUM_REMAINING_CODESIZE = 1 * 1024 * 1024;
        const IR::LocationDescriptor descriptor{location};
        if (A64::LocationDescriptor{descriptor}.SingleStepping()) {
            return PrecompileResult::Unreadable;
        }
        {
            std::shared_lock lock{cache->mutex};
            if (cache->pending_ranges.empty() && cache->emitter->GetBasicBlock(descriptor)) {
                return PrecompileResult::Present;
            }
        }
        const u64 invalidations = cache->invalidations.load();
        bool readable = true;
        const auto get_code = [this, &readable](u64 vaddr) {
            const std::optional<u32> instruction = conf.callbacks->MemoryReadCode(vaddr);
            readable = readable && instruction.has_value();
            return instruction;
        };
        auto const arch_descriptor = A64::LocationDescriptor{descriptor};
        ir_block.Reset(arch_descriptor);
        A64::Translate(ir_block, arch_descriptor, get_code, {conf.define_unpredictable_behaviour, conf.wall_clock_cntpct, true, conf.elide_ld_st_barriers, conf.native_lse_atomics});
        if (!readable) {
            return PrecompileResult::Unreadable;
        }
        Optimization::Optimize(ir_block, conf, cache->polyfill_options);
        std::unique_lock lock{cache->mutex};
        cache->ApplyRangesLocked();
        if (cache->emitter->GetBasicBlock(descriptor)) {
            return PrecompileResult::Present;
        }
        if (invalidations != cache->invalidations.load()) {
            return PrecompileResult::Unreadable;  // the code changed meanwhile: skip it
        }
        // With far code (cold paths in the last quarter of the cache), SpaceRemaining is the
        // smaller of the near and far room: keep a quarter free there.
        const size_t keep = cache->block_of_code->HasFarCode() ? keep_free / 4 : keep_free;
        if (cache->pending_clear.load() ||
            cache->block_of_code->SpaceRemaining() < MINIMUM_REMAINING_CODESIZE + keep) {
            return PrecompileResult::Full;
        }
        cache->block_of_code->EnsureMemoryCommitted(MINIMUM_REMAINING_CODESIZE);
        cache->emitter->Emit(ir_block);
        return PrecompileResult::Compiled;
    }

    size_t CodeCacheUsed() const {
        std::shared_lock lock{cache->mutex};
        return size_t(reinterpret_cast<const char*>(cache->block_of_code->getCurr()) - reinterpret_cast<const char*>(cache->block_of_code->GetCodeBegin()));
    }

    std::string Disassemble() const {
        std::shared_lock lock{cache->mutex};
        const size_t size = reinterpret_cast<const char*>(cache->block_of_code->getCurr()) - reinterpret_cast<const char*>(cache->block_of_code->GetCodeBegin());
        auto const* p = reinterpret_cast<const char*>(cache->block_of_code->GetCodeBegin());
        return Common::DisassembleX64(p, p + size);
    }

private:
    /// Waits out a full clear and drops the RSB and fast dispatch entries of dropped blocks.
    void Enter() {
        // Cleared first: a clear or invalidation requested after this halts the Jit again.
        ClearHalt(HaltReason::CacheInvalidation);
        cache->Enter(running, conf.counters);
        if (cache->invalidations.load() != applied_invalidations) {
            std::unique_lock lock{cache->mutex};
            cache->ApplyRangesLocked(&jit_state);
            applied_invalidations = cache->invalidations.load();
        }
        if (const u64 now = cache->epoch.load(); now != epoch) {
            epoch = now;
            jit_state.ResetRSB();
            std::fill(fast_dispatch_table.begin(), fast_dispatch_table.end(), A64EmitX64::FastDispatchEntry{});
        }
    }

    void InitJitStatePointers() {
        jit_state.callbacks = std::bit_cast<u64>(conf.callbacks);
        jit_state.jit_impl = std::bit_cast<u64>(this);
        jit_state.user_config = std::bit_cast<u64>(&conf);
        jit_state.tpidr_el0 = std::bit_cast<u64>(conf.tpidr_el0);
        jit_state.tpidrro_el0 = std::bit_cast<u64>(conf.tpidrro_el0);
        if (conf.global_monitor) {
            jit_state.exclusive_address = std::bit_cast<u64>(GetExclusiveMonitorAddressPointer(conf.global_monitor, conf.processor_id));
            jit_state.exclusive_value = std::bit_cast<u64>(GetExclusiveMonitorValuePointer(conf.global_monitor, conf.processor_id));
        }
        jit_state.fast_dispatch_table = std::bit_cast<u64>(fast_dispatch_table.data());
    }

    static CodePtr GetCurrentBlockThunk(void* thisptr) {
        Jit::Impl* this_ = static_cast<Jit::Impl*>(thisptr);
        return this_->GetCurrentBlock();
    }

    IR::LocationDescriptor GetCurrentLocation() const {
        return IR::LocationDescriptor{jit_state.GetUniqueHash()};
    }

    CodePtr GetCurrentBlock() {
        if (conf.counters) {
            ++conf.counters->block_lookups;
        }
        return GetBlock(GetCurrentLocation());
    }

    CodePtr GetCurrentSingleStep() {
        return GetBlock(A64::LocationDescriptor{GetCurrentLocation()}.SetSingleStepping(true));
    }

    CodePtr GetBlock(IR::LocationDescriptor descriptor) {
        {
            std::shared_lock lock{cache->mutex};
            if (cache->pending_ranges.empty()) {
                if (auto block = cache->emitter->GetBasicBlock(descriptor))
                    return block->entrypoint;
            }
        }

        while (true) {
            // Translated without the lock; an invalidation meanwhile may have changed the code.
            const u64 invalidations = cache->invalidations.load();
            const auto get_code = [this](u64 vaddr) { return conf.callbacks->MemoryReadCode(vaddr); };
            // LocationDescriptor ctor() does important ops (like tflags) do not skip
            auto const arch_descriptor = A64::LocationDescriptor{descriptor};
            ir_block.Reset(arch_descriptor);
            A64::Translate(ir_block, arch_descriptor, get_code, {conf.define_unpredictable_behaviour, conf.wall_clock_cntpct, true, conf.elide_ld_st_barriers, conf.native_lse_atomics});
            Optimization::Optimize(ir_block, conf, cache->polyfill_options);

            std::unique_lock lock{cache->mutex};
            cache->ApplyRangesLocked();
            if (auto block = cache->emitter->GetBasicBlock(descriptor))
                return block->entrypoint;
            if (invalidations != cache->invalidations.load())
                continue;

            constexpr size_t MINIMUM_REMAINING_CODESIZE = 1 * 1024 * 1024;
            if (cache->pending_clear.load() || cache->block_of_code->SpaceRemaining() < MINIMUM_REMAINING_CODESIZE) {
                if (cache->jit_states.size() > 1) {
                    // Other Jits may run the code: halt all, and clear once none runs.
                    cache->RequestClearLocked();
                    return cache->block_of_code->GetReturnFromRunCodeAddress();
                }
                // Immediately evacuate cache
                cache->ClearLocked(conf.counters);
                epoch = cache->epoch.load();
                jit_state.ResetRSB();
                std::fill(fast_dispatch_table.begin(), fast_dispatch_table.end(), A64EmitX64::FastDispatchEntry{});
            }
            cache->block_of_code->EnsureMemoryCommitted(MINIMUM_REMAINING_CODESIZE);

            if (conf.counters) {
                ++conf.counters->blocks_compiled;
            }
            const CodePtr entrypoint = cache->emitter->Emit(ir_block).entrypoint;
            lock.unlock();
            // #1225: the warm-up records the block (and, with per-core caches, queues it for the
            // other cores; with the shared cache their precompile finds it present).
            conf.callbacks->BlockCompiled(descriptor.Value());
            return entrypoint;
        }
    }

    IR::Block ir_block = {LocationDescriptor(0, FP::FPCR(0), false)};
    const UserConfig conf;
    A64JitState jit_state;
    std::shared_ptr<SharedCodeCache> cache;
    std::vector<A64EmitX64::FastDispatchEntry> fast_dispatch_table;
    /// The cache's epoch this Jit's RSB and fast dispatch table are valid for.
    u64 epoch = 0;
    /// The cache's invalidations this Jit has seen applied.
    u64 applied_invalidations = 0;
    bool is_executing = false;
    /// Whether this Jit runs code of the cache (SharedCodeCache::Enter).
    std::atomic<bool> running{false};
};

Jit::Jit(UserConfig conf)
        : impl(std::make_unique<Jit::Impl>(this, conf)) {}

Jit::~Jit() = default;

HaltReason Jit::Run() {
    return impl->Run();
}

HaltReason Jit::Step() {
    return impl->Step();
}

void Jit::ClearCache() {
    impl->ClearCache();
}

void Jit::InvalidateCacheRange(u64 start_address, size_t length) {
    impl->InvalidateCacheRange(start_address, length);
}

PrecompileResult Jit::Precompile(std::uint64_t location, std::size_t keep_free) {
    return impl->Precompile(location, keep_free);
}

std::size_t Jit::CodeCacheUsed() const {
    return impl->CodeCacheUsed();
}

void Jit::Reset() {
    impl->Reset();
}

void Jit::HaltExecution(HaltReason hr) {
    impl->HaltExecution(hr);
}

void Jit::ClearHalt(HaltReason hr) {
    impl->ClearHalt(hr);
}

u64 Jit::GetSP() const {
    return impl->GetSP();
}

void Jit::SetSP(u64 value) {
    impl->SetSP(value);
}

u64 Jit::GetPC() const {
    return impl->GetPC();
}

void Jit::SetPC(u64 value) {
    impl->SetPC(value);
}

u64 Jit::GetRegister(size_t index) const {
    return impl->GetRegister(index);
}

void Jit::SetRegister(size_t index, u64 value) {
    impl->SetRegister(index, value);
}

std::array<u64, 31> Jit::GetRegisters() const {
    return impl->GetRegisters();
}

void Jit::SetRegisters(const std::array<u64, 31>& value) {
    impl->SetRegisters(value);
}

Vector Jit::GetVector(size_t index) const {
    return impl->GetVector(index);
}

void Jit::SetVector(size_t index, Vector value) {
    impl->SetVector(index, value);
}

std::array<Vector, 32> Jit::GetVectors() const {
    return impl->GetVectors();
}

void Jit::SetVectors(const std::array<Vector, 32>& value) {
    impl->SetVectors(value);
}

u32 Jit::GetFpcr() const {
    return impl->GetFpcr();
}

void Jit::SetFpcr(u32 value) {
    impl->SetFpcr(value);
}

u32 Jit::GetFpsr() const {
    return impl->GetFpsr();
}

void Jit::SetFpsr(u32 value) {
    impl->SetFpsr(value);
}

u32 Jit::GetPstate() const {
    return impl->GetPstate();
}

void Jit::SetPstate(u32 value) {
    impl->SetPstate(value);
}

void Jit::ClearExclusiveState() {
    impl->ClearExclusiveState();
}

bool Jit::IsExecuting() const {
    return impl->IsExecuting();
}

std::string Jit::Disassemble() const {
    return impl->Disassemble();
}

}  // namespace Dynarmic::A64
