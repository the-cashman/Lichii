// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <thread>

#include "common/nx2_address_wait.h"

#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

namespace Common {

/// EDEN_NX2_GPU_PROFILE: contended SpinRecursiveMutex acquisitions, their wait and how many slept.
struct SpinMutexContention {
    std::uint64_t count{};
    std::uint64_t ns{};
    std::uint64_t sleeps{};
};
struct SharedSpinMutexContention {
    std::atomic<std::uint64_t> count{};
    std::atomic<std::uint64_t> ns{};
    std::atomic<std::uint64_t> sleeps{};
};
inline bool g_count_spin_contention{};
inline thread_local SpinMutexContention t_spin_contention{};
inline SharedSpinMutexContention g_spin_contention{};

/// A recursive mutex that spins briefly before sleeping. The GPU caches' locks are taken tens of
/// thousands of times a second by the GPU thread, the fencing thread and CPU write traps, and
/// held for microseconds: std::recursive_mutex put a waiting thread to sleep at once, and its
/// wake-up took longer than the critical section (title 1's GPU thread spent ~8% asleep there).
class SpinRecursiveMutex {
public:
    void lock() {
        const std::thread::id self = std::this_thread::get_id();
        if (owner.load(std::memory_order_relaxed) == self) {
            ++depth;
            return;
        }
        std::uint32_t expected = 0;
        if (!state.compare_exchange_strong(expected, 1, std::memory_order_acquire,
                                           std::memory_order_relaxed)) {
            LockContended();
        }
        owner.store(self, std::memory_order_relaxed);
        depth = 1;
        RunOwnerHook();
    }

    bool try_lock() {
        const std::thread::id self = std::this_thread::get_id();
        if (owner.load(std::memory_order_relaxed) == self) {
            ++depth;
            return true;
        }
        std::uint32_t expected = 0;
        if (state.load(std::memory_order_relaxed) != 0 ||
            !state.compare_exchange_strong(expected, 1, std::memory_order_acquire,
                                           std::memory_order_relaxed)) {
            return false;
        }
        owner.store(self, std::memory_order_relaxed);
        depth = 1;
        RunOwnerHook();
        return true;
    }

    void unlock() {
        if (depth == 1) {
            RunOwnerHook();
        }
        if (--depth != 0) {
            return;
        }
        owner.store(std::thread::id{}, std::memory_order_relaxed);
        if (state.exchange(0, std::memory_order_release) == 2) {
            if (AddressWaits()) {
                Nx2AddressWait::WakeOne(state);
            } else {
                state.notify_one();
            }
        }
    }

    /// fn(context) runs on the owner right after each outermost lock and right before each
    /// outermost unlock (EDEN_NX2_WRITE_HANDOFF applies the guest writes queued meanwhile). Set
    /// it before other threads use the mutex, clear it before the context goes away.
    void SetOwnerHook(void (*fn)(void*), void* context) {
        hook_context.store(context, std::memory_order_relaxed);
        hook.store(fn, std::memory_order_release);
    }

private:
    void RunOwnerHook() {
        if (const auto fn = hook.load(std::memory_order_acquire)) {
            fn(hook_context.load(std::memory_order_relaxed));
        }
    }

    // About 10 microseconds of spinning before sleeping.
    static constexpr int SpinCount = 1024;

    /// EDEN_NX2_CACHE_LOCK_WAIT: sleepers use Common::Nx2AddressWait (WaitOnAddress on Windows)
    /// instead of std::atomic::wait, whose winpthreads condition variable made the unlocking GPU
    /// thread block on a shared mutex and wake every sleeper of the slot.
    static bool AddressWaits() {
        static const bool enabled = [] {
            const char* value = std::getenv("EDEN_NX2_CACHE_LOCK_WAIT");
            return value == nullptr || value[0] != '0';
        }();
        return enabled;
    }

    static void Pause() {
#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
        _mm_pause();
#elif defined(__aarch64__)
        asm volatile("yield");
#endif
    }

    void LockContended() {
        if (g_count_spin_contention) {
            const auto start = std::chrono::steady_clock::now();
            const bool slept = Acquire();
            const auto ns = static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - start)
                    .count());
            ++t_spin_contention.count;
            t_spin_contention.ns += ns;
            t_spin_contention.sleeps += slept ? 1 : 0;
            g_spin_contention.count.fetch_add(1, std::memory_order_relaxed);
            g_spin_contention.ns.fetch_add(ns, std::memory_order_relaxed);
            g_spin_contention.sleeps.fetch_add(slept ? 1 : 0, std::memory_order_relaxed);
            return;
        }
        Acquire();
    }

    /// Returns whether it slept.
    bool Acquire() {
        for (int spin = 0; spin < SpinCount; ++spin) {
            Pause();
            if (state.load(std::memory_order_relaxed) == 0) {
                std::uint32_t expected = 0;
                if (state.compare_exchange_weak(expected, 1, std::memory_order_acquire,
                                                std::memory_order_relaxed)) {
                    return false;
                }
            }
        }
        // 2: held with (possible) sleepers; taken this way it stays marked, so its unlock wakes
        // the next sleeper.
        while (state.exchange(2, std::memory_order_acquire) != 0) {
            if (AddressWaits()) {
                Nx2AddressWait::Wait(state, 2);
            } else {
                state.wait(2, std::memory_order_relaxed);
            }
        }
        return true;
    }

    std::atomic<std::uint32_t> state{0}; ///< 0 free, 1 held, 2 held with sleepers
    std::atomic<std::thread::id> owner{};
    std::uint32_t depth{}; ///< Recursion depth, touched only by the owner
    std::atomic<void (*)(void*)> hook{};
    std::atomic<void*> hook_context{};
};

} // namespace Common
