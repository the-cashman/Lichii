// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include <span>
#include <thread>
#include <utility>
#include <vector>

#include "common/common_types.h"
#include "common/logging.h"
#include "common/thread.h"

namespace VideoCommon {

/// EDEN_NX2_ASYNC_UNTRACK=1: the GPU back thread hands the page untracking it asks for (guest
/// writes applied at fences and through the write hand-off: a page's cached count drops and, at
/// zero, write access comes back with an mprotect, ~5 us each, ~100 a frame in Ocarina's village)
/// to a helper thread instead of waiting for the system calls.
///
/// Why it is safe (the same argument as RasterizerVulkan::OnCPUWrite's deferral after the lock):
/// only untracking is deferred, and a page that stays protected a little longer only traps a guest
/// write that then finds it untracked. Tracking stays immediate on the caller. The cached counts
/// are changed under DeviceMemoryManager's range lock in any order the two threads reach it, and
/// each protection change follows the count it was decided from under that lock, so a page ends
/// protected exactly when its count is non-zero. Device unmaps wait until the helper has applied
/// everything queued before them (Begin/EndDeferredCachedUpdates, as the guest threads' deferral).
class Nx2AsyncUntrack {
public:
    using Apply = void (*)(void* tracker, std::span<const std::pair<VAddr, size_t>>, s32 delta);
    using End = void (*)(void* tracker);

    [[nodiscard]] static bool Enabled() noexcept {
        static const bool enabled = [] {
            const char* value = std::getenv("EDEN_NX2_ASYNC_UNTRACK");
            return value != nullptr && value[0] == '1';
        }();
        return enabled;
    }

    /// Set on the GPU back thread (GpuSplit::BackLoop) when enabled.
    static inline thread_local bool t_defer{};

    static Nx2AsyncUntrack& Get() {
        static Nx2AsyncUntrack instance;
        return instance;
    }

    /// Queues one batch of runs; the caller has called the tracker's BeginDeferredCachedUpdates.
    void Push(void* tracker, Apply apply, End end, std::span<const std::pair<VAddr, size_t>> runs,
              s32 delta) {
        bool wake;
        {
            std::scoped_lock lock{mutex};
            batches.push_back(Batch{tracker, apply, end, delta, collected.size(), runs.size()});
            collected.insert(collected.end(), runs.begin(), runs.end());
            wake = sleeping;
            sleeping = false;
        }
        if (wake) {
            generation.fetch_add(1, std::memory_order_release);
            generation.notify_one();
        }
    }

private:
    struct Batch {
        void* tracker;
        Apply apply;
        End end;
        s32 delta;
        size_t begin;
        size_t count;
    };

    Nx2AsyncUntrack() {
        thread = std::jthread([this](std::stop_token token) { Run(token); });
    }

    ~Nx2AsyncUntrack() {
        thread.request_stop();
        generation.fetch_add(1, std::memory_order_release);
        generation.notify_one();
    }

    void Run(std::stop_token token) {
        Common::SetCurrentThreadName("GPUUntrack");
        std::vector<Batch> work;
        std::vector<std::pair<VAddr, size_t>> runs;
        while (!token.stop_requested()) {
            u32 seen;
            {
                std::scoped_lock lock{mutex};
                work.swap(batches);
                runs.swap(collected);
                seen = generation.load(std::memory_order_acquire);
                sleeping = work.empty();
            }
            if (work.empty()) {
                generation.wait(seen, std::memory_order_acquire);
                continue;
            }
            // In the order queued (a page can be untracked, tracked and untracked again).
            for (const Batch& batch : work) {
                batch.apply(batch.tracker,
                            std::span<const std::pair<VAddr, size_t>>(runs).subspan(batch.begin,
                                                                                   batch.count),
                            batch.delta);
                batch.end(batch.tracker);
            }
            work.clear();
            runs.clear();
        }
    }

    std::mutex mutex;
    std::vector<Batch> batches;
    std::vector<std::pair<VAddr, size_t>> collected;
    bool sleeping{};
    std::atomic<u32> generation{};
    std::jthread thread;
};

/// EDEN_NX2_LAZY_UNTRACK (on unless =0): a page the GPU back thread untracks keeps its cached
/// count (and so its write protection) for up to two frames. When the buffer cache tracks it again in that time
/// (the next draw's sync uploads the guest's write and re-protects it: Ocarina's usual cycle), the
/// held count is used instead of a new one: no unprotect and no protect, two mprotects saved.
/// Pages not tracked again are released at the frame tick (TickFrame), and any range is released
/// before a device unmap or invalidation reaches the buffer cache (RasterizerVulkan).
///
/// Invariant: a page's cached count = the trackers that track it + 1 while it is held here. A held
/// page is untracked in the buffer cache, so only a track (which consumes the hold) or a release
/// changes it; both take the hold out under this class's lock, so exactly one of them does.
/// While held, a guest write to the page traps as before the untrack and finds it untracked
/// (the same state RasterizerVulkan::OnCPUWrite's deferral leaves for a moment).
class Nx2LazyUntrack {
public:
    using Apply = Nx2AsyncUntrack::Apply;
    using End = Nx2AsyncUntrack::End;
    static constexpr u64 PageBits = 12;
    static constexpr u64 HoldFrames = 2;

    [[nodiscard]] static int Level() noexcept {
        static const int level = [] {
            // On (1) by default: Ocarina village/Kokiri +5-6 % fps (GPUBack -1 ms/frame), pacing
            // and images unchanged. =0 turns it off, =2 also holds guest-trap untracks.
            const char* value = std::getenv("EDEN_NX2_LAZY_UNTRACK");
            if (value == nullptr || value[0] == '\0') {
                return 1;
            }
            return value[0] == '1' || value[0] == '2' ? value[0] - '0' : 0;
        }();
        return level;
    }

    [[nodiscard]] static bool Enabled() noexcept {
        return Level() != 0;
    }

    /// =2: also the untracks of guest write traps that take the buffer cache's lock themselves
    /// (RasterizerVulkan::OnCPUWrite's deferral after the lock): their pages stay protected
    /// until the back thread tracks them again, so further writes trap meanwhile.
    [[nodiscard]] static bool HoldGuest() noexcept {
        return Level() == 2;
    }

    /// Set on the GPU back thread (GpuSplit::BackLoop) when enabled.
    static inline thread_local bool t_hold{};

    static Nx2LazyUntrack& Get() {
        static Nx2LazyUntrack instance;
        return instance;
    }

    [[nodiscard]] bool Any() const noexcept {
        return held_count.load(std::memory_order_acquire) != 0;
    }

    /// Holds the pages of `runs` (untracked by the caller) instead of releasing their counts.
    void Hold(void* tracker_, Apply apply_, End end_, std::span<const std::pair<VAddr, size_t>> runs) {
        std::scoped_lock lock{mutex};
        tracker = tracker_;
        apply = apply_;
        end = end_;
        for (const auto& [addr, size] : runs) {
            for (u64 page = addr >> PageBits; page < (addr + size) >> PageBits; ++page) {
                held.insert_or_assign(page, frame.load(std::memory_order_relaxed));
                ++stats[0];
            }
        }
        held_count.store(held.size(), std::memory_order_release);
    }

    /// Takes the held pages out of `runs` (sorted, disjoint), which the caller is about to track:
    /// their count is already there. Rewrites `runs` with what is left to track.
    template <class Ranges>
    void Consume(Ranges& runs) {
        std::scoped_lock lock{mutex};
        if (held.empty()) {
            return;
        }
        Ranges left;
        left.reserve(runs.size());
        for (const auto& [addr, size] : runs) {
            u64 run_begin = addr;
            for (u64 page = addr >> PageBits; page < (addr + size) >> PageBits; ++page) {
                const auto it = held.find(page);
                if (it == held.end()) {
                    continue;
                }
                held.erase(it);
                ++stats[1];
                const u64 page_addr = page << PageBits;
                if (page_addr > run_begin) {
                    left.emplace_back(run_begin, page_addr - run_begin);
                }
                run_begin = page_addr + (u64{1} << PageBits);
            }
            if (addr + size > run_begin) {
                left.emplace_back(run_begin, addr + size - run_begin);
            }
        }
        held_count.store(held.size(), std::memory_order_release);
        runs.swap(left);
    }

    /// Releases the held pages in [addr, addr + size) now (device unmaps and invalidations).
    void Release(u64 addr, u64 size) {
        Flush([&](u64 page, u64) { return page >= (addr >> PageBits) &&
                                          page < ((addr + size + (u64{1} << PageBits) - 1) >> PageBits); },
              2);
    }

    /// Frame tick: releases pages held for HoldFrames frames, logs counters every 600 frames.
    void Tick() {
        const u64 now = frame.fetch_add(1, std::memory_order_relaxed) + 1;
        if (Any()) {
            Flush([now](u64, u64 held_frame) { return held_frame + HoldFrames <= now; }, 3);
        }
        if (now % 600 == 0) {
            std::scoped_lock lock{mutex};
            LOG_INFO(HW_GPU,
                     "EDEN_NX2_LAZY_UNTRACK last 600 frames: held {} pages, tracked again {}, "
                     "released by unmap/invalidate {}, by age {}; held now {}",
                     stats[0], stats[1], stats[2], stats[3], held.size());
            stats = {};
        }
    }

private:
    template <class Pred>
    void Flush(Pred&& pred, size_t stat) {
        std::vector<std::pair<VAddr, size_t>> runs;
        void* flush_tracker;
        Apply flush_apply;
        {
            std::scoped_lock lock{mutex};
            for (auto it = held.begin(); it != held.end();) {
                if (pred(it->first, it->second)) {
                    runs.emplace_back(it->first << PageBits, size_t{1} << PageBits);
                    it = held.erase(it);
                    ++stats[stat];
                } else {
                    ++it;
                }
            }
            held_count.store(held.size(), std::memory_order_release);
            flush_tracker = tracker;
            flush_apply = apply;
        }
        if (runs.empty() || flush_tracker == nullptr) {
            return;
        }
        std::sort(runs.begin(), runs.end());
        size_t last = 0;
        for (size_t i = 1; i < runs.size(); ++i) {
            if (runs[last].first + runs[last].second == runs[i].first) {
                runs[last].second += runs[i].second;
            } else {
                runs[++last] = runs[i];
            }
        }
        runs.resize(last + 1);
        flush_apply(flush_tracker, runs, -1);
    }

    std::mutex mutex;
    std::unordered_map<u64, u64> held; ///< page -> frame it was held on
    std::atomic<size_t> held_count{};
    void* tracker{};
    Apply apply{};
    End end{};
    std::atomic<u64> frame{};
    std::array<u64, 4> stats{};
};

} // namespace VideoCommon
