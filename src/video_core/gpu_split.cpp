// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <optional>
#include <thread>

#ifdef _MSC_VER
#include <intrin.h>
#endif
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#ifdef __GLIBC__
#include <execinfo.h>
#endif
#include <cstring>
#include <string>

#include "common/dkb_cpu_trace.h"
#include "common/logging.h"
#include "common/settings.h"
#include "common/thread.h"
#include "core/core.h"
#include "video_core/engines/engine_interface.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/gpu_split.h"
#include "video_core/memory_manager.h"
#include "video_core/nx2.h"
#include "video_core/nx2_back_idle_trace.h"
#include "video_core/nx2_fast_draw.h"
#include "video_core/nx2_profile.h"

namespace VideoCommon {

namespace {

void Pause() {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    _mm_pause();
#elif defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    asm volatile("yield");
#else
    std::this_thread::yield();
#endif
}

[[maybe_unused]] std::atomic<u32> g_back_thread_id{};

// EDEN_NX2_RING_PREFETCH=0: the back reads each ring line only when it gets to it.
const bool g_ring_prefetch = Nx2Switch("EDEN_NX2_RING_PREFETCH", true);
/// How far ahead of the entry it runs the back prefetches ring words (16 lines).
constexpr u64 RingPrefetchWords = 128;

// EDEN_NX2_DELTA_CHAIN=0: a StateDelta entry returns to BackLoop before the entry it was sent
// with runs, instead of running it straight after (one pass of the loop's bookkeeping fewer).
const bool g_delta_chain = Nx2Switch("EDEN_NX2_DELTA_CHAIN", true);

void PrefetchRingLine(const u64* address) {
#if defined(__clang__) || defined(__GNUC__)
    __builtin_prefetch(address, 0, 3);
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    _mm_prefetch(reinterpret_cast<const char*>(address), _MM_HINT_T0);
#else
    (void)address;
#endif
}

// Back thread only, profiling only: the engine of the method entries is a Maxwell3D (zone
// "method 3D"), and the Call entries by thunk (each sending site's lambda has its own).
bool g_back_engine_3d = false;
struct CallStats {
    u64 thunk;
    u64 ticks;
    u64 count;
};
std::array<CallStats, 24> g_call_stats{};
u64 g_call_window_ns = 0;
u64 g_call_window_ticks = 0;

void NoteCall(u64 thunk, u64 ticks) {
    for (auto& stats : g_call_stats) {
        if (stats.thunk == thunk || stats.thunk == 0) {
            stats.thunk = thunk;
            stats.ticks += ticks;
            ++stats.count;
            return;
        }
    }
    // More sites than slots: the rest add to the last one.
    g_call_stats.back().ticks += ticks;
    ++g_call_stats.back().count;
}

void ResetCalls() {
    g_call_stats = {};
    g_call_window_ns = Nx2Profile::Now();
    g_call_window_ticks = Nx2Profile::Ticks();
}

/// Logs the Call entries per second by thunk, as offsets in the executable (llvm-symbolizer
/// --relative-address names the sending site), then starts a new window.
void ReportCalls() {
    const u64 now = Nx2Profile::Now();
    const u64 now_ticks = Nx2Profile::Ticks();
    const double window = static_cast<double>((std::max)(now - g_call_window_ns, u64{1}));
    const double window_ticks =
        static_cast<double>((std::max)(now_ticks - g_call_window_ticks, u64{1}));
    u64 base = 0;
#ifdef _WIN32
    base = reinterpret_cast<u64>(GetModuleHandleW(nullptr));
#endif
    std::string text;
    for (const auto& stats : g_call_stats) {
        if (stats.count == 0) {
            continue;
        }
        text += fmt::format(" {:#x} {:.0f}/s {:.2f} ms,",
                            stats.thunk >= base ? stats.thunk - base : stats.thunk,
                            static_cast<double>(stats.count) * 1e9 / window,
                            static_cast<double>(stats.ticks) * 1000.0 / window_ticks);
    }
    if (!text.empty()) {
        text.pop_back();
        LOG_INFO(HW_GPU, "GPU back thread calls per second by thunk (offset, inclusive):{}",
                 text);
    }
    ResetCalls();
}

BackIdleTrace::FrontState WaitState(DrainSite site) {
    return site == DrainSite::ListWriters || site == DrainSite::ListSync ||
                   site == DrainSite::ListFlush
               ? BackIdleTrace::FrontState::Writers
               : BackIdleTrace::FrontState::Drain;
}

// A wait of the front for its back, counted by site while g_acquire_stats counts.
class CountedWait {
public:
    explicit CountedWait(DrainSite site_) : site{static_cast<size_t>(site_)} {
        if (g_acquire_counting) {
            start = std::chrono::steady_clock::now();
        }
    }
    ~CountedWait() {
        if (!start) {
            return;
        }
        Count(g_acquire_stats.site_drains[site]);
        Count(g_acquire_stats.site_drain_ns[site],
              static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                   std::chrono::steady_clock::now() - *start)
                                   .count()));
    }

private:
    size_t site;
    std::optional<std::chrono::steady_clock::time_point> start;
};

} // Anonymous namespace

void GpuSplit::CheckOwner(const char* where) {
#ifdef _WIN32
    // Also while the back is idle: its idle hook submits (Scheduler::FlushDeferred).
    if (!instance.back_running.load(std::memory_order_acquire)) {
        return;
    }
    const DWORD thread = GetCurrentThreadId();
    if (thread == g_back_thread_id.load(std::memory_order_relaxed)) {
        return;
    }
    static std::atomic<u32> reports{};
    if (reports.fetch_add(1) >= 12) {
        return;
    }
    std::string stack;
    void* frames[40];
    const USHORT count = RtlCaptureStackBackTrace(1, 40, frames, nullptr);
    for (USHORT i = 0; i < count; ++i) {
        HMODULE module{};
        char name[MAX_PATH] = "?";
        if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                   GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                               static_cast<LPCSTR>(frames[i]), &module)) {
            GetModuleFileNameA(module, name, sizeof(name));
        }
        const char* base_name = std::strrchr(name, '\\');
        stack += fmt::format(" {}+{:#x}", base_name ? base_name + 1 : name,
                             reinterpret_cast<uintptr_t>(frames[i]) -
                                 reinterpret_cast<uintptr_t>(module));
    }
    LOG_ERROR(HW_GPU, "GPU split owner check: {} on thread {} while the back runs; stack:{}",
              where, thread, stack);
#else
    // Also while the back is idle: its idle hook submits (Scheduler::FlushDeferred).
    if (on_back_thread || !instance.back_running.load(std::memory_order_acquire)) {
        return;
    }
    static std::atomic<u32> reports{};
    if (reports.fetch_add(1) >= 12) {
        return;
    }
    std::string stack;
#ifdef __GLIBC__
    void* frames[40];
    const int count = backtrace(frames, 40);
    if (char** const symbols = backtrace_symbols(frames, count)) {
        for (int i = 1; i < count; ++i) {
            stack += fmt::format("\n  {}", symbols[i]);
        }
        std::free(symbols);
    }
#endif
    LOG_ERROR(HW_GPU, "GPU split owner check: {} on a thread other than the back; stack:{}", where,
              stack);
#endif
}

GpuSplit GpuSplit::instance;

void GpuSplit::RequestTick() {
    u64 state = tick_state.load(std::memory_order_relaxed);
    const u64 generation = state & ~(TickAccepting | TickRequested);
    do {
        if ((state & TickAccepting) == 0 ||
            (state & ~(TickAccepting | TickRequested)) != generation) {
            return;
        }
    } while (!tick_state.compare_exchange_weak(
        state, state | TickRequested, std::memory_order_seq_cst, std::memory_order_relaxed));
    // Paired with the back's check after setting back_waiting: one side sees the other.
    if (back_waiting.load(std::memory_order_seq_cst)) {
        wake.fetch_add(1, std::memory_order_seq_cst);
        wake.notify_one();
    }
}

void GpuSplit::MaybeStart(Core::System& system_) {
    static const char mode = [] {
        const char* value = std::getenv("EDEN_NX2_SPLIT");
        return value != nullptr ? value[0] : '2';
    }();
    const bool wanted = mode == '1' || mode == '2';
    if (!wanted || !Tegra::g_nx2_ampere.load(std::memory_order_relaxed)) {
        return;
    }
    // The front reads command lists straight from guest memory; a read that flushes the caches
    // first (safe DMA, accurate GPU) would touch back-owned state. The settings can change
    // mid-game: hand the engines back to the front (and start again when they allow it).
    const bool safe_reads = Settings::IsDMALevelDefault() ? Settings::IsGPULevelHigh()
                                                          : Settings::IsDMALevelSafe();
    const bool allowed = !safe_reads && !Settings::IsGPULevelHigh() &&
                         Settings::values.renderer_backend.GetValue() == Settings::RendererBackend::Vulkan;
    if (active && !allowed) {
        Drain();
        Stop();
        LOG_INFO(HW_GPU, "GPU thread split: stopped, the accuracy settings need the caches on the front");
        return;
    }
    if (active || !allowed) {
        return;
    }
    system = &system_;
    check_owner = std::getenv("EDEN_NX2_SPLIT_CHECK") != nullptr;
    const char* const verify = std::getenv("EDEN_NX2_ACQUIRE_VERIFY");
    g_acquire_verify = verify != nullptr && verify[0] == '1';
    g_acquire_counting = g_acquire_verify || Nx2Profile::Enabled() ||
                         std::getenv("EDEN_NX2_ACQUIRE_STATS") != nullptr ||
                         std::getenv("EDEN_NX2_GUEST_THREAD_STATS") != nullptr;
    ring = std::make_unique<u64[]>(RingWords);
    back_thread = std::jthread([this](std::stop_token stop_token) { BackLoop(stop_token); });
    active = true;
    ++starts;
    const char* jit = std::getenv("EDEN_NX2_MME_JIT");
    front_macros = mode == '2' && (jit == nullptr || jit[0] != '0');
    // EDEN_NX2_NARROW_DRAIN=0: the front's reads drain the split fully, as before; 2: draws count
    // as unbounded writes too (see NarrowDrains).
    static const char narrow_mode = [] {
        const char* value = std::getenv("EDEN_NX2_NARROW_DRAIN");
        return value != nullptr ? value[0] : '1';
    }();
    narrow = front_macros && narrow_mode != '0';
    narrow_strict = narrow && narrow_mode == '2';
    narrow_compute = narrow && Nx2Switch("EDEN_NX2_NARROW_COMPUTE", true);
    narrow_render_enable = narrow && Nx2Switch("EDEN_NX2_NARROW_RENDER_ENABLE", true);
    narrow_dma_recorded = narrow && Nx2Switch("EDEN_NX2_NARROW_DMA_RECORDED", true);
    narrow_list_sync = narrow && Nx2Switch("EDEN_NX2_NARROW_LIST_SYNC", true);
    sem_predict = narrow && Nx2Switch("EDEN_NX2_SEM_PREDICT", true);
    sem_predict_landed = sem_predict && Nx2Switch("EDEN_NX2_SEM_PREDICT_LANDED", true);
    acquire_poll = narrow && Nx2Switch("EDEN_NX2_ACQUIRE_POLL", true);
    list_exact = narrow && Nx2Switch("EDEN_NX2_LIST_EXACT", true);
    sync_gpu_semaphores = narrow && Nx2Switch("EDEN_NX2_SYNC_GPU_SEMAPHORES", true);
    publish_idle = Nx2Switch("EDEN_NX2_PUBLISH_IDLE", true);
    static const bool qmd_predict_switch = Nx2Switch("EDEN_NX2_QMD_PREDICT", true);
    static const bool acquire_sync_at_switch = Nx2Switch("EDEN_NX2_ACQUIRE_SYNC_AT", true);
    static const bool publish_waits_switch = Nx2Switch("EDEN_NX2_PUBLISH_WAITS", true);
    qmd_predict = narrow_compute && sem_predict && qmd_predict_switch;
    acquire_sync_at = narrow && acquire_sync_at_switch;
    publish_waits = publish_waits_switch;
    static const bool acquire_skip_barriers_switch =
        Nx2Switch("EDEN_NX2_ACQUIRE_SKIP_BARRIERS", true);
    acquire_skip_barriers = acquire_poll && acquire_skip_barriers_switch;
    if (narrow) {
        page_slots = std::make_unique<PageSlot[]>(PageBuckets * 4);
        word_slots = std::make_unique<WordSlot[]>(WordBuckets * 4);
        queued_releases = std::make_unique<QueuedRelease[]>(ReleaseBuckets * 4);
    }
    ++session;
    LOG_INFO(HW_GPU, "GPU thread split: engine methods run on a second thread{}{}",
             front_macros ? ", macros on the first" : "",
             narrow ? (narrow_strict ? ", narrow drains (strict)" : ", narrow drains") : "");
    if (narrow) {
        LOG_INFO(HW_GPU,
                 "GPU thread split narrow switches: compute {} render_enable {} dma_recorded {} "
                 "list_sync {} sem_predict {} sem_predict_landed {} acquire_poll {} list_exact {} "
                 "sync_gpu_semaphores {} qmd_predict {} acquire_sync_at {} publish_waits {} "
                 "acquire_skip_barriers {} sem_mirror {}",
                 narrow_compute, narrow_render_enable, narrow_dma_recorded, narrow_list_sync,
                 sem_predict, sem_predict_landed, acquire_poll, list_exact, sync_gpu_semaphores,
                 qmd_predict, acquire_sync_at, publish_waits, acquire_skip_barriers,
                 Tegra::Nx2SemaphoreMirrorEnabled());
    }
}

void GpuSplit::Stop() {
    if (!back_thread.joinable()) {
        return;
    }
    back_thread.request_stop();
    wake.fetch_add(1, std::memory_order_seq_cst);
    wake.notify_one();
    back_thread.join();
    active = false;
    // The split is static and outlives the emulation session. The next session's back thread
    // starts reading at 0: with the old positions kept it skipped through the new, empty ring a
    // wrap at a time and never reached the old write position (the next game's first GPU work
    // never ran, it waited on its syncpoint forever), and the engines it held were freed.
    ring.reset();
    write_position = 0;
    last_writer_position = 0;
    pending_write_count = 0;
    async_writes.clear();
    report_pages.clear();
    last_sync_position = 0;
    published_local = 0;
    consumed_cache = 0;
    publish_every = PublishEvery;
    front_engine = nullptr;
    pending_flush = nullptr;
    pending_flush_argument = nullptr;
    back_engine = nullptr;
    front_macros = false;
    narrow = false;
    narrow_strict = false;
    narrow_compute = false;
    narrow_render_enable = false;
    narrow_dma_recorded = false;
    narrow_list_sync = false;
    sem_predict = false;
    sem_predict_landed = false;
    acquire_poll = false;
    list_exact = false;
    sync_gpu_semaphores = false;
    qmd_predict = false;
    acquire_sync_at = false;
    publish_waits = false;
    acquire_skip_barriers = false;
    for (auto& page : gpu_semaphore_pages) {
        page.store(0, std::memory_order_relaxed);
    }
    gpu_semaphore_count.store(0, std::memory_order_relaxed);
    gpu_semaphore_saturated.store(false, std::memory_order_relaxed);
    unbounded_position = 0;
    unbounded_writer_position = 0;
    acquire_position = 0;
    write_floor = 0;
    guest_write_position = 0;
    release_barrier = 0;
    queued_releases.reset();
    page_slots.reset();
    word_slots.reset();
    huge_writes = {};
    next_huge = 0;
    spaces = {};
    space_last = {};
    used_spaces = 0;
    last_record = 0;
    last_page_slot = nullptr;
    published.store(0, std::memory_order_relaxed);
    consumed.store(0, std::memory_order_relaxed);
    back_waiting.store(false, std::memory_order_relaxed);
    back_idle.store(false, std::memory_order_relaxed);
    front_waiting.store(false, std::memory_order_relaxed);
}

u64* GpuSplit::ReserveSlow(size_t words) {
    size_t index = write_position & RingMask;
    if (index + words > RingWords) {
        // Entries never wrap: skip to the start of the ring.
        const size_t rest = RingWords - index;
        WaitForSpace(rest);
        ring[index] = static_cast<u64>(Kind::Wrap);
        write_position += rest;
        index = 0;
    }
    WaitForSpace(words);
    return &ring[index];
}

void GpuSplit::WaitForSpace(size_t words) {
    if (write_position + words - consumed_cache <= RingWords) {
        return;
    }
    Publish();
    Nx2Profile::Scope scope{Nx2Profile::Zone::SplitDrain};
    const CountedWait counted{DrainSite::RingFull};
    const u64 start = Nx2Profile::Enabled() ? Nx2Profile::Ticks() : 0;
    for (u32 spins = 0;; ++spins) {
        consumed_cache = consumed.load(std::memory_order_acquire);
        if (write_position + words - consumed_cache <= RingWords) {
            Nx2Profile::AddDrain(DrainSite::RingFull, start);
            return;
        }
        if (spins < 4096) {
            Pause();
        } else {
            std::this_thread::yield();
        }
    }
}

void GpuSplit::Publish() {
    if (published_local == write_position) {
        return;
    }
    published_local = write_position;
    publish_every = std::min<size_t>(publish_every * 2, PublishEvery);
    published.store(write_position, std::memory_order_seq_cst);
    // Taken back here: the publishes until the back runs again don't wake it once more each.
    if (back_waiting.load(std::memory_order_seq_cst) &&
        back_waiting.exchange(false, std::memory_order_seq_cst)) {
        wake.fetch_add(1, std::memory_order_seq_cst);
        wake.notify_one();
    }
}

void GpuSplit::Drain(DrainSite site) {
    RunPendingFlush();
    consumed_cache = consumed.load(std::memory_order_acquire);
    if (consumed_cache == write_position && BackQuiet()) {
        return;
    }
    Publish();
    Nx2Profile::Scope scope{Nx2Profile::Zone::SplitDrain};
    const BackIdleTrace::FrontWait idle_trace{WaitState(site)};
    const CountedWait counted{site};
    const u64 start = Nx2Profile::Enabled() ? Nx2Profile::Ticks() : 0;
    // Everything consumed is not enough: the back submits deferred work (Scheduler::DeferFlush)
    // from its idle hook afterwards, and the caller may use the scheduler once this returns.
    bool hook_asked = false;
    for (u32 spins = 0;; ++spins) {
        consumed_cache = consumed.load(std::memory_order_acquire);
        if (consumed_cache == write_position) {
            if (BackQuiet()) {
                publish_every = 8;
                Nx2Profile::AddDrain(site, start);
                return;
            }
            if (!hook_asked) {
                RequestIdleHook();
                hook_asked = true;
            }
            // Backoff sleeps only while entries are unconsumed: spin until the hook is done.
            if (spins < 16384) {
                Pause();
            } else {
                std::this_thread::yield();
            }
            continue;
        }
        Backoff(spins, write_position);
    }
}

bool GpuSplit::WaitForWriters() {
    // Held-back work goes first: a writer in it is noted before the target is taken.
    RunPendingFlush();
    return WaitFor(std::min(last_writer_position, write_position), DrainSite::ListWriters);
}

bool GpuSplit::WaitForWritesTo(u64 begin, u64 size) {
    consumed_cache = consumed.load(std::memory_order_acquire);
    const u64 end = begin + size;
    const u64 remembered = (std::min<u64>)(pending_write_count, pending_writes.size());
    u64 target = 0;
    u64 scanned = 0;
    for (; scanned < remembered; ++scanned) {
        const PendingWrite& write =
            pending_writes[(pending_write_count - 1 - scanned) % pending_writes.size()];
        if (write.position <= consumed_cache) {
            break;
        }
        if (write.begin < end && begin < write.end) {
            target = (std::max)(target, write.position);
        }
    }
    if (scanned == pending_writes.size()) {
        return false;
    }
    if (target != 0) {
        WaitFor(target, DrainSite::MacroDmaRead);
    }
    return true;
}

bool GpuSplit::HasQueuedWritesTo(u64 begin, u64 size) {
    consumed_cache = consumed.load(std::memory_order_acquire);
    const u64 end = begin + size;
    const u64 remembered = (std::min<u64>)(pending_write_count, pending_writes.size());
    for (u64 scanned = 0; scanned < remembered; ++scanned) {
        const PendingWrite& write =
            pending_writes[(pending_write_count - 1 - scanned) % pending_writes.size()];
        if (write.position <= consumed_cache) {
            return false;
        }
        if (write.begin < end && begin < write.end) {
            return true;
        }
    }
    return remembered == pending_writes.size();
}

bool GpuSplit::OverlayQueuedWrites(u64 since, u64 begin, u32* out, u32 words) {
    if ((begin & 3) != 0) {
        return false;
    }
    const u64 end = begin + u64{words} * sizeof(u32);
    const u64 remembered = (std::min<u64>)(pending_write_count, pending_writes.size());
    // Newest first, back to what the back had consumed before guest memory was read.
    std::array<u32, 64> overlapping;
    u32 found = 0;
    u64 scanned = 0;
    for (; scanned < remembered; ++scanned) {
        const u64 index = (pending_write_count - 1 - scanned) % pending_writes.size();
        const PendingWrite& write = pending_writes[index];
        if (write.position <= since) {
            break;
        }
        if (write.begin < end && begin < write.end) {
            if (write.words == 0 || (write.begin & 3) != 0 || found == overlapping.size()) {
                return false;
            }
            overlapping[found++] = static_cast<u32>(index);
        }
    }
    if (scanned == pending_writes.size()) {
        return false;
    }
    for (u32 i = found; i-- > 0;) {
        const PendingWrite& write = pending_writes[overlapping[i]];
        for (u32 w = 0; w < write.words; ++w) {
            const u64 address = write.begin + u64{w} * sizeof(u32);
            if (address >= begin && address < end) {
                out[(address - begin) / sizeof(u32)] = write.data[w];
            }
        }
    }
    return true;
}

bool GpuSplit::WaitForSync() {
    RunPendingFlush();
    return WaitFor(std::min(last_sync_position, write_position), DrainSite::ListSync);
}

void GpuSplit::NoteSyncAt(u64 position, UnboundedCause cause) {
    if (!acquire_sync_at) {
        NoteSync(cause);
        return;
    }
    const u64 sync = (std::min)(position, write_position);
    if (sync >= last_sync_position) {
        last_sync_position = sync;
    }
}

void GpuSplit::NoteSyncDone(UnboundedCause cause) {
    if (!acquire_sync_at) {
        NoteSync(cause);
        return;
    }
    consumed_cache = consumed.load(std::memory_order_acquire);
    NoteSyncAt(consumed_cache, cause);
}

u64 GpuSplit::GuestReadTarget(const void* space, GPUVAddr address, u64 size) {
    consumed_cache = consumed.load(std::memory_order_acquire);
    return (std::min)(PendingWriteTarget(space, address, size, guest_write_position),
                      write_position);
}

bool GpuSplit::WaitForGuestRead(u64 target) {
    RunPendingFlush();
    return WaitFor((std::min)(target, write_position), DrainSite::QmdRead);
}

bool GpuSplit::NarrowVerify() {
    static const bool verify = [] {
        const char* value = std::getenv("EDEN_NX2_NARROW_VERIFY");
        return value != nullptr && value[0] != '0';
    }();
    return verify;
}

void GpuSplit::ReportNarrowCheck(DrainSite site, u32 mismatches, u32 macro, GPUVAddr address) {
    struct Counts {
        std::atomic<u64> checks;
        std::atomic<u64> mismatched;
    };
    static std::array<Counts, static_cast<size_t>(DrainSite::Count)> counts{};
    Counts& count = counts[static_cast<size_t>(site)];
    const u64 checks = count.checks.fetch_add(1, std::memory_order_relaxed) + 1;
    if (mismatches != 0) {
        const u64 mismatched = count.mismatched.fetch_add(1, std::memory_order_relaxed) + 1;
        if (mismatched <= 32 || (mismatched & 1023) == 0) {
            LOG_WARNING(HW_GPU,
                        "Narrow drain verify: {} at {:#x} (macro {:#x}) read {} words other than "
                        "a full drain; {} of {} checks differed so far",
                        drain_site_names[static_cast<size_t>(site)], address, macro, mismatches,
                        mismatched, checks);
        }
    }
    if ((checks & 0xffff) == 0) {
        LOG_INFO(HW_GPU, "Narrow drain verify: {}: {} checks, {} differed",
                 drain_site_names[static_cast<size_t>(site)], checks,
                 count.mismatched.load(std::memory_order_relaxed));
    }
}

namespace {

[[nodiscard]] size_t HashBucket(u64 key, size_t buckets) {
    return static_cast<size_t>((key * 0x9E3779B97F4A7C15ULL) >> 32) & (buckets - 1);
}

} // Anonymous namespace

void GpuSplit::NoteGpuSemaphore(GPUVAddr address) {
    if (gpu_semaphore_saturated.load(std::memory_order_relaxed)) {
        return;
    }
    const u64 key = (address >> PageBits) + 1;
    size_t index = HashBucket(key, GpuSemaphoreSlots);
    for (size_t probe = 0; probe < 16; ++probe, index = (index + 1) & (GpuSemaphoreSlots - 1)) {
        u64 current = gpu_semaphore_pages[index].load(std::memory_order_acquire);
        if (current == key) {
            return;
        }
        if (current == 0) {
            if (gpu_semaphore_pages[index].compare_exchange_strong(current, key,
                                                                 std::memory_order_acq_rel)) {
                if (gpu_semaphore_count.fetch_add(1, std::memory_order_relaxed) + 1 >=
                    GpuSemaphoreSlots * 3 / 4) {
                    gpu_semaphore_saturated.store(true, std::memory_order_release);
                }
                return;
            }
            if (current == key) {
                return;
            }
        }
    }
    gpu_semaphore_saturated.store(true, std::memory_order_release);
}

namespace {
// Whether NoteGpuSemaphore saw `page` (or can't tell).
bool GpuSemaphorePage(const std::atomic<u64>* pages, size_t slots, u64 page) {
    const u64 key = page + 1;
    size_t index = HashBucket(key, slots);
    for (size_t probe = 0; probe < 16; ++probe, index = (index + 1) & (slots - 1)) {
        const u64 current = pages[index].load(std::memory_order_acquire);
        if (current == key) {
            return true;
        }
        if (current == 0) {
            return false;
        }
    }
    return true;
}
} // Anonymous namespace

bool GpuSplit::AcquireSyncs(GPUVAddr address) const {
    if (!sync_gpu_semaphores || gpu_semaphore_saturated.load(std::memory_order_acquire)) {
        return true;
    }
    return GpuSemaphorePage(gpu_semaphore_pages.data(), GpuSemaphoreSlots, address >> PageBits);
}

bool GpuSplit::MayHoldReports(GPUVAddr address, u64 size) {
    if (size == 0) {
        return false;
    }
    if (gpu_semaphore_saturated.load(std::memory_order_acquire)) {
        return true;
    }
    for (GPUVAddr page = address >> PageBits; page <= (address + size - 1) >> PageBits; ++page) {
        if (GpuSemaphorePage(gpu_semaphore_pages.data(), GpuSemaphoreSlots, page)) {
            return true;
        }
    }
    return false;
}

u32 GpuSplit::SpaceId(const void* space) {
    for (u32 i = 0; i < used_spaces; ++i) {
        if (spaces[i] == space) {
            return i;
        }
    }
    if (used_spaces < MaxSpaces - 1) {
        spaces[used_spaces] = space;
        return used_spaces++;
    }
    // The rest share the last id: their reads wait for all of its writes.
    return MaxSpaces - 1;
}

void GpuSplit::InsertPage(u64 key, bool coarse) {
    PageSlot* slot = last_page_slot;
    if (slot == nullptr || slot->key != key) {
        PageSlot* const bucket = &page_slots[HashBucket(key, PageBuckets) * 4];
        slot = nullptr;
        for (size_t i = 0; i < 4; ++i) {
            if (bucket[i].key == key) {
                slot = &bucket[i];
                break;
            }
        }
        if (slot == nullptr) {
            // A done slot, else the oldest (the floor waits for it).
            PageSlot* victim = &bucket[0];
            for (size_t i = 0; i < 4; ++i) {
                if (bucket[i].any <= consumed_cache) {
                    victim = &bucket[i];
                    break;
                }
                if (bucket[i].any < victim->any) {
                    victim = &bucket[i];
                }
            }
            RaiseFloor(victim->any);
            *victim = {key, 0, 0};
            slot = victim;
        }
        last_page_slot = slot;
    }
    if (coarse) {
        slot->coarse = write_position;
    }
    slot->any = write_position;
}

void GpuSplit::InsertWord(u64 key) {
    WordSlot* const bucket = &word_slots[HashBucket(key, WordBuckets) * 4];
    WordSlot* victim = nullptr;
    for (size_t i = 0; i < 4; ++i) {
        if (bucket[i].key == key) {
            bucket[i].position = write_position;
            return;
        }
        if (victim == nullptr && bucket[i].position <= consumed_cache) {
            victim = &bucket[i];
        }
    }
    if (victim == nullptr) {
        victim = &bucket[0];
        for (size_t i = 1; i < 4; ++i) {
            if (bucket[i].position < victim->position) {
                victim = &bucket[i];
            }
        }
        RaiseFloor(victim->position);
    }
    *victim = {key, write_position};
}

const GpuSplit::PageSlot* GpuSplit::FindPage(u64 key) const {
    const PageSlot* const bucket = &page_slots[HashBucket(key, PageBuckets) * 4];
    for (size_t i = 0; i < 4; ++i) {
        if (bucket[i].key == key) {
            return &bucket[i];
        }
    }
    return nullptr;
}

const GpuSplit::WordSlot* GpuSplit::FindWord(u64 key) const {
    const WordSlot* const bucket = &word_slots[HashBucket(key, WordBuckets) * 4];
    for (size_t i = 0; i < 4; ++i) {
        if (bucket[i].key == key) {
            return &bucket[i];
        }
    }
    return nullptr;
}

void GpuSplit::RecordWrite(const void* space, GPUVAddr address, u64 size, bool fine) {
    const u64 sid = SpaceId(space);
    space_last[sid] = write_position;
    last_record = write_position;
    const GPUVAddr end = address + size;
    const GPUVAddr first_page = address >> PageBits;
    const GPUVAddr last_page = (end - 1) >> PageBits;
    if (fine) {
        NoteGpuSemaphore(address);
    }
    if (fine && size <= MaxFineBytes) {
        for (GPUVAddr word = address >> WordBits; word <= (end - 1) >> WordBits; ++word) {
            InsertWord((word << 2) | sid);
        }
        for (GPUVAddr page = first_page; page <= last_page; ++page) {
            InsertPage((page << 2) | sid, false);
        }
        return;
    }
    if (last_page - first_page >= 64) {
        // Round robin replaces the oldest.
        HugeWrite& slot = huge_writes[next_huge];
        RaiseFloor(slot.position);
        slot = {space, address, end, write_position};
        next_huge = (next_huge + 1) % MaxHugeWrites;
        return;
    }
    for (GPUVAddr page = first_page; page <= last_page; ++page) {
        InsertPage((page << 2) | sid, true);
    }
}

u64 GpuSplit::PendingWriteTarget(const void* space, GPUVAddr address, u64 size, u64 base) const {
    u64 target = (std::max)(base, write_floor);
    if (page_slots == nullptr || size == 0) {
        return target;
    }
    const GPUVAddr end = address + size;
    u64 sid = MaxSpaces;
    for (u32 i = 0; i < used_spaces; ++i) {
        if (spaces[i] == space) {
            sid = i;
            break;
        }
    }
    // Another address space may map the same memory.
    for (u64 i = 0; i < MaxSpaces; ++i) {
        if (i != sid || i == MaxSpaces - 1) {
            target = (std::max)(target, space_last[i]);
        }
    }
    for (const HugeWrite& write : huge_writes) {
        if (write.position > target &&
            (write.space != space || (write.begin < end && address < write.end))) {
            target = write.position;
        }
    }
    if (sid == MaxSpaces) {
        // Nothing recorded in this space itself.
        return target;
    }
    const GPUVAddr first_page = address >> PageBits;
    const GPUVAddr last_page = (end - 1) >> PageBits;
    if (last_page - first_page >= 64) {
        return (std::max)(target, last_record);
    }
    const bool small = size <= 256;
    for (GPUVAddr page = first_page; page <= last_page; ++page) {
        const PageSlot* const slot = FindPage((page << 2) | sid);
        if (slot == nullptr || slot->any <= (std::max)(target, consumed_cache)) {
            // No pending write in the page (a pending slot that gave way raised the floor).
            continue;
        }
        if (!small) {
            target = slot->any;
            continue;
        }
        target = (std::max)(target, slot->coarse);
        const GPUVAddr page_begin = (std::max)(address, page << PageBits);
        const GPUVAddr page_end = (std::min)(end, (page + 1) << PageBits);
        for (GPUVAddr word = page_begin >> WordBits; word <= (page_end - 1) >> WordBits; ++word) {
            if (const WordSlot* const word_slot = FindWord((word << 2) | sid)) {
                target = (std::max)(target, word_slot->position);
            }
        }
    }
    return target;
}

bool GpuSplit::WrittenSince(const void* space, GPUVAddr address, u64 position) const {
    if (page_slots == nullptr || write_floor > position) {
        return true;
    }
    u64 sid = MaxSpaces;
    for (u32 i = 0; i < used_spaces; ++i) {
        if (spaces[i] == space) {
            sid = i;
            break;
        }
    }
    if (sid == MaxSpaces) {
        return true;
    }
    for (u64 i = 0; i < MaxSpaces; ++i) {
        if ((i != sid || i == MaxSpaces - 1) && space_last[i] > position) {
            return true;
        }
    }
    for (const HugeWrite& write : huge_writes) {
        if (write.position > position && write.begin <= address && address < write.end) {
            return true;
        }
    }
    // A slot replaced since no longer tells: count it as written.
    const PageSlot* const page = FindPage(((address >> PageBits) << 2) | sid);
    if (page == nullptr || page->coarse > position) {
        return true;
    }
    const WordSlot* const word = FindWord(((address >> WordBits) << 2) | sid);
    return word == nullptr || word->position > position;
}

bool GpuSplit::WaitForWritesTo(const void* space, GPUVAddr address, u64 size, DrainSite site,
                               NarrowBase base_kind) {
    // Held-back writes (constant buffer runs) are sent and recorded first.
    RunPendingFlush();
    consumed_cache = consumed.load(std::memory_order_acquire);
    u64 base = unbounded_position;
    if (base_kind == NarrowBase::Acquire) {
        base = acquire_position;
    } else if (base_kind == NarrowBase::Recorded) {
        base = 0;
    }
    const u64 target = PendingWriteTarget(space, address, size, base);
    return WaitFor((std::min)(target, write_position), site);
}

bool GpuSplit::WaitForWritersTo(const void* space, GPUVAddr address, u64 size, bool sync_only) {
    RunPendingFlush(); // As in WaitForWritesTo
    consumed_cache = consumed.load(std::memory_order_acquire);
    const u64 base = sync_only ? last_sync_position : unbounded_writer_position;
    const u64 target = PendingWriteTarget(space, address, size, base);
    return WaitFor((std::min)(target, write_position), DrainSite::NarrowListWriters);
}

GpuSplit::QueuedRelease* GpuSplit::FindRelease(const void* space, GPUVAddr address) {
    QueuedRelease* const bucket =
        &queued_releases[HashBucket(address >> WordBits, ReleaseBuckets) * 4];
    for (size_t i = 0; i < 4; ++i) {
        if (bucket[i].used && bucket[i].address == address && bucket[i].space == space) {
            return &bucket[i];
        }
    }
    return nullptr;
}

void GpuSplit::NoteRelease(const void* space, GPUVAddr address, u32 value, bool at_least) {
    if (!sem_predict || queued_releases == nullptr) {
        return;
    }
    const auto* const memory = static_cast<const Tegra::MemoryManager*>(space);
    const auto device = memory->GpuToCpuAddress(address);
    // Unmapped (OoT releases to address 0 at boot): the read would return 0 anyway, and log an
    // error for each release.
    const u32 before = device ? memory->Read<u32>(address) : 0;
    QueuedRelease* release = FindRelease(space, address);
    if (release == nullptr) {
        // A free slot, else the oldest.
        QueuedRelease* const bucket =
            &queued_releases[HashBucket(address >> WordBits, ReleaseBuckets) * 4];
        release = &bucket[0];
        for (size_t i = 0; i < 4; ++i) {
            if (!bucket[i].used) {
                release = &bucket[i];
                break;
            }
            if (bucket[i].position < release->position) {
                release = &bucket[i];
            }
        }
        release->used = false;
    }
    const u32 slot = device ? static_cast<u32>(Tegra::Nx2LandedSlot(*device)) : 0;
    const u32 landed = Tegra::g_nx2_semaphore_landed[slot].load(std::memory_order_acquire);
    if (!release->used || release->landed_slot != slot ||
        landed - release->landed_base >= release->sent) {
        // The releases before it landed: count from here.
        release->landed_base = landed;
        release->sent = 0;
    }
    release->space = space;
    release->address = address;
    release->position = write_position;
    release->value = value;
    release->before = before;
    release->landed_slot = slot;
    release->sent += device ? 1 : 0;
    release->used = true;
    release->at_least = at_least;
}

std::optional<u32> GpuSplit::PredictedSemaphore(const void* space, GPUVAddr address) {
    const std::optional<SemaphorePrediction> prediction = PredictedRelease(space, address);
    if (!prediction || prediction->at_least) {
        return std::nullopt;
    }
    return prediction->value;
}

std::optional<SemaphorePrediction> GpuSplit::PredictedRelease(const void* space,
                                                              GPUVAddr address) {
    if (!sem_predict || queued_releases == nullptr) {
        return std::nullopt;
    }
    consumed_cache = consumed.load(std::memory_order_acquire);
    const QueuedRelease* const release = FindRelease(space, address);
    // A write sent after the release (a copy, a compute report) may change the word; so may
    // work the front lost track of (NoteReleaseBarrier).
    if (release == nullptr || release->position < release_barrier ||
        WrittenSince(space, address, release->position)) {
        return std::nullopt;
    }
    const SemaphorePrediction prediction{
        .value = release->value,
        .at_least = release->at_least,
        .position = release->position,
    };
    if (release->position > consumed_cache) {
        return prediction;
    }
    if (!sem_predict_landed || release->sent == 0) {
        return std::nullopt;
    }
    // The back ran it. A fence may still hold the write back (one-word releases land when the
    // fence after them signals): while it hasn't landed and guest memory still holds the word
    // from before it, the word is the release's.
    const u32 landed =
        Tegra::g_nx2_semaphore_landed[release->landed_slot].load(std::memory_order_acquire);
    if (landed - release->landed_base >= release->sent) {
        return std::nullopt;
    }
    const u32 now = static_cast<const Tegra::MemoryManager*>(space)->Read<u32>(address);
    if (now != release->before && now != release->value) {
        return std::nullopt;
    }
    return prediction;
}

void GpuSplit::ForgetRelease(const void* space, GPUVAddr address) {
    if (!sem_predict || queued_releases == nullptr) {
        return;
    }
    if (QueuedRelease* const release = FindRelease(space, address)) {
        release->used = false;
    }
}

bool GpuSplit::WaitFor(u64 target, DrainSite site) {
    consumed_cache = consumed.load(std::memory_order_acquire);
    if (consumed_cache >= target) {
        return false;
    }
    // EDEN_NX2_PUBLISH_WAITS: the back gets the held-back work (a register delta, a draw batch)
    // too, so it has that to run once it reaches the target instead of waiting for the front.
    if (publish_waits) {
        RunPendingFlush();
    }
    Nx2Profile::Scope scope{Nx2Profile::Zone::WriterWait};
    const BackIdleTrace::FrontWait idle_trace{WaitState(site)};
    const CountedWait counted{site};
    wait_target.store(target, std::memory_order_seq_cst);
    Publish();
    const u64 start = Nx2Profile::Enabled() ? Nx2Profile::Ticks() : 0;
    for (u32 spins = 0;; ++spins) {
        consumed_cache = consumed.load(std::memory_order_acquire);
        if (consumed_cache >= target) {
            wait_target.store(~u64{0}, std::memory_order_relaxed);
            Nx2Profile::AddDrain(site, start);
            return true;
        }
        Backoff(spins, target);
    }
}

void GpuSplit::Backoff(u32 spins, u64 target) {
    if (spins < 16384) {
        Pause();
        return;
    }
    if (!back_running.load(std::memory_order_acquire)) {
        std::this_thread::yield();
        return;
    }
    // A long wait sleeps until the back stores a new position (port of #161's front_wake): the
    // back checks front_waiting after each store, the front checks the position after setting it.
    front_waiting.store(true, std::memory_order_seq_cst);
    const u32 seen = front_wake.load(std::memory_order_seq_cst);
    if (consumed.load(std::memory_order_seq_cst) < target) {
        front_wake.wait(seen, std::memory_order_seq_cst);
    }
    front_waiting.store(false, std::memory_order_relaxed);
}

void GpuSplit::SwitchEngine(Tegra::Engines::EngineInterface* engine) {
    // Before front_engine changes: the held-back work selects its own engine.
    RunPendingFlush();
    front_engine = engine;
    u64* const entry = Reserve(2);
    entry[0] = static_cast<u64>(Kind::Engine);
    entry[1] = reinterpret_cast<u64>(engine);
    Commit(2);
}

void GpuSplit::MultiMethod(Tegra::Engines::EngineInterface* engine, u32 method, const u32* words,
                           u32 amount, u32 methods_pending, const GPUVAddr* segment) {
    SelectEngine(engine);
    // Long runs (constant buffer and inline uploads) go in pieces, as when a method's data
    // continues in the next GPFIFO entry.
    while (amount > 0) {
        const u32 piece = std::min(amount, MaxMultiWords);
        const size_t head = segment != nullptr ? 3 : 2;
        const size_t size = head + (piece + 1) / 2;
        u64* const entry = Reserve(size);
        entry[0] = static_cast<u64>(Kind::Multi) | (segment != nullptr ? HasSegment : 0) |
                   (u64{method & 0xffff} << 8) | (u64{piece} << 32);
        entry[1] = methods_pending;
        if (segment != nullptr) {
            entry[2] = *segment;
        }
        entry[size - 1] = 0;
        std::memcpy(entry + head, words, size_t{piece} * sizeof(u32));
        Commit(size);
        words += piece;
        amount -= piece;
        methods_pending -= piece;
    }
}

void GpuSplit::DeferFunction(std::function<void()>&& func) {
    auto* const heap = new std::function<void()>(std::move(func));
    Defer([heap] {
        (*heap)();
        delete heap;
    });
}

size_t GpuSplit::Execute(const u64* entry, size_t words) {
    const u64 head = entry[0];
    switch (static_cast<Kind>(head & 0xf)) {
    case Kind::Wrap:
        return 0;
    case Kind::Engine:
        back_engine = reinterpret_cast<Tegra::Engines::EngineInterface*>(entry[1]);
        if (Nx2Profile::Enabled()) [[unlikely]] {
            g_back_engine_3d = Tegra::Engines::Maxwell3D::IsMaxwell3D(back_engine);
        }
        return 2;
    case Kind::Method: {
        const u32 method = static_cast<u32>((head >> 8) & 0xffff);
        const u32 argument = static_cast<u32>(head >> 32);
        auto* const engine = back_engine;
        if (engine->fast_maxwell3d) {
            // Nx2Perf: as below without the profile zones (off with fast_maxwell3d) and the
            // virtual call.
            using Tegra::Engines::Maxwell3D;
            auto* const maxwell3d = static_cast<Maxwell3D*>(engine);
            constexpr u32 cb_data = MAXWELL3D_REG_INDEX(const_buffer.buffer);
            if (method - cb_data < 16 && (head & HasSegment) == 0) {
                // Single constant buffer words in a row (macros write them one by one) go up in
                // one write. They are executable: each entry would only consume the sink first.
                std::array<u32, 64> methods;
                std::array<u32, 64> arguments;
                methods[0] = method;
                arguments[0] = argument;
                u32 count = 1;
                while (count < methods.size() && count < words) {
                    const u64 next = entry[count];
                    const u32 next_method = static_cast<u32>((next >> 8) & 0xffff);
                    if (static_cast<Kind>(next & 0xf) != Kind::Method || (next & HasSegment) != 0 ||
                        next_method - cb_data >= 16) {
                        break;
                    }
                    methods[count] = next_method;
                    arguments[count] = static_cast<u32>(next >> 32);
                    ++count;
                }
                maxwell3d->ConsumeSink(*system);
                if (count == 1) {
                    maxwell3d->CallMethod(*system, method, argument, true);
                } else {
                    maxwell3d->CallConstBufferWords(*system, methods.data(), arguments.data(),
                                                    count);
                }
                return count;
            }
            if ((head & Direct) != 0) {
                maxwell3d->ConsumeSink(*system);
                maxwell3d->CallMethod(*system, method, argument, true);
            } else if (!maxwell3d->execution_mask[method]) {
                maxwell3d->method_sink.emplace_back(method, argument);
            } else {
                maxwell3d->ConsumeSink(*system);
                if ((head & HasSegment) != 0) {
                    maxwell3d->current_dma_segment = entry[1];
                }
                maxwell3d->CallMethod(*system, method, argument, (head & IsLast) != 0);
            }
            return (head & HasSegment) != 0 ? 2 : 1;
        }
        // Profiling (fast_maxwell3d off): Maxwell3D's methods and the other engines' apart.
        const Nx2Profile::Zone zone =
            g_back_engine_3d ? Nx2Profile::Zone::Method3D : Nx2Profile::Zone::Method;
        if (!engine->reports_fast_draw_breaks) {
            Nx2FastDraw::Break(Nx2FastDraw::Reason::OtherEngine);
        }
        if ((head & Direct) != 0) {
            Nx2Profile::Scope method_scope{zone};
            engine->ConsumeSink(*system);
            engine->CallMethod(*system, method, argument, true);
        } else if (!engine->execution_mask[method]) {
            engine->method_sink.emplace_back(method, argument);
        } else {
            Nx2Profile::Scope method_scope{zone};
            engine->ConsumeSink(*system);
            if ((head & HasSegment) != 0) {
                engine->current_dma_segment = entry[1];
            }
            engine->CallMethod(*system, method, argument, (head & IsLast) != 0);
        }
        return (head & HasSegment) != 0 ? 2 : 1;
    }
    case Kind::Multi: {
        const u32 method = static_cast<u32>((head >> 8) & 0xffff);
        const u32 amount = static_cast<u32>(head >> 32);
        const size_t data = (head & HasSegment) != 0 ? 3 : 2;
        auto* const engine = back_engine;
        if (!engine->reports_fast_draw_breaks) {
            Nx2FastDraw::Break(Nx2FastDraw::Reason::OtherEngine);
        }
        Nx2Profile::Scope method_scope{Nx2Profile::Zone::Method};
        engine->ConsumeSink(*system);
        if ((head & HasSegment) != 0) {
            engine->current_dma_segment = entry[2];
        }
        engine->CallMultiMethod(*system, method, reinterpret_cast<const u32*>(entry + data),
                                amount, static_cast<u32>(entry[1]));
        return data + (amount + 1) / 2;
    }
    case Kind::Call: {
        if ((head & FastDrawNeutral) == 0) {
            Nx2FastDraw::Break(Nx2FastDraw::Reason::Call);
        }
        const size_t payload = static_cast<size_t>((head >> 8) & 0xffffff);
        const auto thunk = reinterpret_cast<void (*)(const u64*)>(entry[1]);
        if (Nx2Profile::Enabled()) [[unlikely]] {
            // By thunk (one per sending site): inclusive time, logged with the zones.
            Nx2Profile::Scope call_scope{Nx2Profile::Zone::Call};
            const u64 start = Nx2Profile::Ticks();
            thunk(entry + 2);
            NoteCall(entry[1], Nx2Profile::Ticks() - start);
        } else {
            thunk(entry + 2);
        }
        return 2 + payload;
    }
    case Kind::StateDelta: {
        // Only Maxwell3D sends these (FrontState).
        const u32 count = static_cast<u32>(head >> 32);
        {
            Nx2Profile::Scope delta_scope{Nx2Profile::Zone::DeltaApply};
            static_cast<Tegra::Engines::Maxwell3D*>(back_engine)
                ->ApplyFrontDelta(*system, entry + 1, count);
        }
        const size_t size = 1 + size_t{count};
        // EDEN_NX2_DELTA_CHAIN: the front sends a delta right before the entry that made it go
        // (FrontFlushPending, from Reserve): run that entry now when it is published too, as
        // BackLoop would next. Only kinds that never return 0 (Wrap) or chain again.
        if (g_delta_chain && size < words) {
            const Kind next = static_cast<Kind>(entry[size] & 0xf);
            if (next == Kind::Method || next == Kind::CbWrite || next == Kind::CbBind ||
                next == Kind::DrawBatch) {
                if (Nx2Profile::Enabled()) [[unlikely]] {
                    Nx2Profile::Add(Nx2Profile::Counter::RingEntries, 1);
                }
                return size + Execute(entry + size, words - size);
            }
        }
        return size;
    }
    case Kind::DrawBatch: {
        const u32 count = static_cast<u32>((head >> 8) & 0xff);
        const u32 repeats = static_cast<u32>(head >> 32);
        Nx2Profile::Scope batch_scope{Nx2Profile::Zone::DrawBatch};
        static_cast<Tegra::Engines::Maxwell3D*>(back_engine)
            ->RunDrawBatch(*system, entry + 1, count, repeats);
        return 1 + size_t{count};
    }
    case Kind::CbWrite: {
        const u32 count = static_cast<u32>((head >> 8) & 7);
        Nx2Profile::Scope cb_scope{Nx2Profile::Zone::CbWrite};
        // The words after the first (at most 3, see GpuSplit::CbWrite) stay in the ring words
        // after the head.
        static_cast<Tegra::Engines::Maxwell3D*>(back_engine)
            ->BackCbWrite(*system, static_cast<u32>((head >> 12) & 0xffff),
                          static_cast<u32>(head >> 32), entry + 1, count);
        return 1 + size_t{count} / 2;
    }
    case Kind::CbBind: {
        Nx2Profile::Scope bind_scope{Nx2Profile::Zone::CbBind};
        static_cast<Tegra::Engines::Maxwell3D*>(back_engine)
            ->BackCbBind(*system, static_cast<u32>((head >> 8) & 0xff),
                         static_cast<u32>(head >> 32));
        return 1;
    }
    case Kind::Writes: {
        // Only KeplerCompute sends these (FrontQueueWrite).
        Nx2FastDraw::Break(Nx2FastDraw::Reason::OtherEngine);
        const u32 count = static_cast<u32>(head >> 32);
        const size_t data = (head & HasSegment) != 0 ? 2 : 1;
        Nx2Profile::Scope method_scope{Nx2Profile::Zone::Method};
        if ((head & HasSegment) != 0) {
            back_engine->current_dma_segment = entry[1];
        }
        if (Nx2Profile::Enabled()) [[unlikely]] {
            Nx2Profile::Add(Nx2Profile::Counter::FusedWrites, count);
        }
        static_cast<Tegra::Engines::KeplerCompute*>(back_engine)
            ->BackWrites(*system, entry + data, count);
        return data + size_t{count};
    }
    }
    return 1;
}

void GpuSplit::BackLoop(std::stop_token stop_token) {
    const Common::DkbCpuTrace::Registration cpu_trace{Common::DkbCpuTrace::Role::GpuBack};
    const StutterTrace::ThreadScope stutter_thread{StutterTrace::Thread::Back};
    Common::SetCurrentThreadName("GPUBack");
    on_back_thread = true;
    tick_state.store(tick_state.load(std::memory_order_relaxed) + TickGeneration + TickAccepting,
                     std::memory_order_release);
    back_running.store(true, std::memory_order_release);
    Common::SetCurrentThreadPriority(Common::ThreadPriority::Critical);
    Common::SetCurrentThreadToPerformanceCores();
    if (Settings::values.pin_gpu_threads.GetValue()) {
        Common::PinCurrentThread(Common::PinnedRole::GpuBack);
    } else {
        Common::PinHostThread(Common::HostThreadRole::GpuBack);
    }
    system->RegisterHostThread();
#ifdef _WIN32
    g_back_thread_id.store(GetCurrentThreadId(), std::memory_order_relaxed);
#endif
    u64 read_position = 0;
    const bool back_lean = Nx2Switch("EDEN_NX2_BACK_LEAN", true);
    const bool profiling = Nx2Profile::Enabled();
    g_back_engine_3d = false;
    if (profiling) {
        ResetCalls();
    }
    // Counts across batches, so a stream of short batches still checks hook requests.
    u32 entries = 0;
    const bool idle_trace = BackIdleTrace::Enabled();
    u64 idle_since = 0;
    while (!stop_token.stop_requested()) {
        if (Nx2Profile::Report("GPU back thread", true)) [[unlikely]] {
            ReportCalls();
        }
        MaybeTick();
        u64 available = published.load(std::memory_order_acquire);
        if (available == read_position) {
            Nx2Profile::Scope scope{Nx2Profile::Zone::SplitWait};
            if (idle_trace && idle_since == 0) {
                idle_since = BackIdleTrace::Now();
            }
            back_idle.store(true, std::memory_order_relaxed);
            const auto spin_start = g_acquire_counting ? std::chrono::steady_clock::now()
                                                       : std::chrono::steady_clock::time_point{};
            // The front usually sends more within microseconds: spin briefly before sleeping.
            for (u32 spins = 0; spins < 8192 && available == read_position &&
                                !hook_requested.load(std::memory_order_relaxed) &&
                                (tick_state.load(std::memory_order_relaxed) & TickRequested) == 0;
                 ++spins) {
                Pause();
                available = published.load(std::memory_order_acquire);
            }
            if (g_acquire_counting) {
                Count(g_acquire_stats.back_waits);
                Count(g_acquire_stats.back_spin_ns,
                      static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                           std::chrono::steady_clock::now() - spin_start)
                                           .count()));
            }
            if (available == read_position) {
                hook_requested.store(false, std::memory_order_seq_cst);
                // The front may already see everything consumed: Drain() waits for back_quiet
                // too, so the deferred submit can't overlap whatever the front does next.
                if (!back_quiet.load(std::memory_order_relaxed)) {
                    if (idle_hook != nullptr) {
                        Nx2Profile::Scope hook_scope{Nx2Profile::Zone::IdleHook};
                        idle_hook(idle_argument);
                    }
                    back_quiet.store(true, std::memory_order_release);
                }
                available = published.load(std::memory_order_acquire);
            }
            if (available == read_position) {
                // Publish bumps `wake` when it sees back_waiting: read it before the last look
                // at `published`, so a publish in between ends the wait at once.
                back_waiting.store(true, std::memory_order_seq_cst);
                const u32 seen = wake.load(std::memory_order_seq_cst);
                available = published.load(std::memory_order_seq_cst);
                // Idle-hook and tick requests made since the hooks ran must prevent sleeping.
                if (available == read_position && !stop_token.stop_requested() &&
                    !hook_requested.load(std::memory_order_seq_cst) &&
                    (tick_state.load(std::memory_order_seq_cst) & TickRequested) == 0) {
                    Count(g_acquire_stats.back_sleeps);
                    wake.wait(seen, std::memory_order_seq_cst);
                }
                back_waiting.store(false, std::memory_order_relaxed);
                continue;
            }
        }
        if (idle_since != 0) {
            BackIdleTrace::NoteBackIdle(idle_since, BackIdleTrace::Now());
            idle_since = 0;
        }
        if (back_idle.load(std::memory_order_relaxed)) {
            back_idle.store(false, std::memory_order_relaxed);
        }
        Nx2Profile::Scope ring_scope{Nx2Profile::Zone::RingDecode};
        u64 last_store = read_position;
        // Before any entry runs, so a front that sees these entries consumed also sees this.
        back_quiet.store(false, std::memory_order_seq_cst);
        // EDEN_NX2_RING_PREFETCH: the ring lines ahead of the entry, up to a line short of what
        // the front published (a line it may still be writing would only bounce between the two
        // cores). The front wrote them moments ago: they sit in its core's caches.
        u64 prefetched = read_position;
        while (read_position != available) {
            const size_t index = read_position & RingMask;
            if (g_ring_prefetch) {
                const u64 ahead = (std::min)(read_position + RingPrefetchWords,
                                             available > 8 ? available - 8 : u64{0});
                prefetched = (std::max)(prefetched, read_position);
                while (prefetched + 8 <= ahead) {
                    prefetched += 8;
                    PrefetchRingLine(&ring[prefetched & RingMask]);
                }
            }
            const size_t size =
                Execute(&ring[index], std::min<u64>(available - read_position, RingWords - index));
            const size_t advance = size != 0 ? size : RingWords - index;
            read_position += advance;
            if (profiling) [[unlikely]] {
                Nx2Profile::Add(Nx2Profile::Counter::RingEntries, size != 0 ? 1 : 0);
                Nx2Profile::Add(Nx2Profile::Counter::RingWords, advance);
            }
            // EDEN_NX2_BACK_LEAN: the hook request (another thread's cache line) every 4 entries.
            if ((!back_lean || (++entries & 3) == 0) &&
                hook_requested.load(std::memory_order_relaxed) &&
                idle_hook != nullptr) [[unlikely]] {
                hook_requested.store(false, std::memory_order_seq_cst);
                Nx2Profile::Scope hook_scope{Nx2Profile::Zone::IdleHook};
                idle_hook(idle_argument);
            }
            // Serve a guest read before the next entry, even with more work queued on the front.
            MaybeTick();
            // Hand ring space back now and then while working through a long batch, and at once
            // when the front waits for this position (WaitForWriters).
            if ((read_position - last_store >= 4096 ||
                 read_position >= wait_target.load(std::memory_order_relaxed)) &&
                read_position != available) {
                consumed.store(read_position, std::memory_order_seq_cst);
                last_store = read_position;
                NotifyFront();
            }
        }
        consumed.store(read_position, std::memory_order_seq_cst);
        NotifyFront();
    }
    // Stop accepting ticks before leaving. A reader still has its normal queued GPU tick:
    // it runs on the front after a settings stop, or #1058 releases it at final shutdown.
    tick_state.fetch_and(~(TickAccepting | TickRequested), std::memory_order_seq_cst);
    back_running.store(false, std::memory_order_seq_cst);
    front_wake.fetch_add(1, std::memory_order_seq_cst);
    front_wake.notify_all();
}

void GpuSplit::ReportAcquire() {
    static u32 calls{};
    static auto next = std::chrono::steady_clock::now() + std::chrono::seconds{10};
    if (++calls % 256 != 0) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    if (now < next) {
        return;
    }
    next = now + std::chrono::seconds{10};
    auto& stats = g_acquire_stats;
    const auto take = [](std::atomic<u64>& counter) {
        return counter.exchange(0, std::memory_order_relaxed);
    };
    const u64 passed = take(stats.passed);
    const u64 by_queued = take(stats.by_queued);
    const u64 latched = take(stats.latched);
    const u64 drained = take(stats.drained);
    const u64 by_poll = take(stats.by_poll);
    const u64 by_narrow = take(stats.by_narrow);
    const u64 valve_drains = take(stats.valve_drains);
    const u64 polls = take(stats.polls);
    const u64 acquire_polls = take(stats.acquire_polls);
    const u64 list_polls = take(stats.list_polls);
    const u64 front_polls = take(stats.front_polls);
    const u64 front_sleeps = take(stats.front_sleeps);
    const u64 front_woken = take(stats.front_woken);
    const u64 front_stored = take(stats.front_stored);
    const u64 front_found_polling = take(stats.front_found_polling);
    const u64 front_found_late = take(stats.front_found_late);
    const u64 early_publishes = take(stats.early_publishes);
    const u64 settles = take(stats.settles);
    const u64 back_waits = take(stats.back_waits);
    const u64 back_sleeps = take(stats.back_sleeps);
    const double spin_ms = static_cast<double>(take(stats.back_spin_ns)) / 1e6;
    LOG_INFO(HW_GPU,
             "Acquires: {} passed without a drain ({} by a queued release, {} by the polled word, "
             "{} by the narrowed word), {} resumed by a latch, {} drained ({} by the 1 ms valve), "
             "{} polls waited without a drain in 10 s; polls: {} of acquires, {} of GPU-written "
             "list copies; back: {} waits for work, {} slept, {:.1f} ms spinning; {} publishes "
             "to an idle back, {} idle hooks asked by polls",
             passed, by_queued, by_poll, by_narrow, latched, drained, valve_drains, polls,
             acquire_polls, list_polls, back_waits, back_sleeps, spin_ms, early_publishes,
             settles);
    LOG_INFO(HW_GPU,
             "GPU thread waits for suspended channels in 10 s: {} in hardware, {} in the OS, {} "
             "ended by a wake, {} by a store to a semaphore word; channels that moved without "
             "either: {} found while polling, {} found late in the OS wait",
             front_polls, front_sleeps, front_woken, front_stored, front_found_polling,
             front_found_late);
    const u64 verified = take(stats.verified);
    std::string sites;
    for (size_t site = 0; site < drain_site_names.size(); ++site) {
        const u64 count = take(stats.site_drains[site]);
        const u64 ns = take(stats.site_drain_ns[site]);
        if (count != 0) {
            sites += fmt::format("{}{} {} ({:.1f} ms)", sites.empty() ? "" : ", ",
                                 drain_site_names[site], count, static_cast<double>(ns) / 1e6);
        }
    }
    if (!sites.empty()) {
        LOG_INFO(HW_GPU, "Front waits for the back in 10 s by site: {}", sites);
    }
    const u64 mismatches = take(stats.mismatches);
    const u64 unlanded = take(stats.unlanded);
    const u64 late = take(stats.late);
    const u64 moved = take(stats.moved);
    if (g_acquire_verify || mismatches != 0 || unlanded != 0 || late != 0) {
        LOG_WARNING(HW_GPU,
                    "Acquire verify: {} decisions without a drain checked, {} passed but not met "
                    "after a drain, {} met by a queued release not in memory after it, {} equal "
                    "or mask acquires whose word moved on after, {} polls waited though a drain "
                    "meets them, in 10 s",
                    verified, mismatches, unlanded, moved, late);
    }
}

} // namespace VideoCommon
