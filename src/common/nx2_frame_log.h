// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>

#include "common/common_types.h"

// EDEN_NX2_FRAME_LOG=<file>: per guest frame, the times (CLOCK_MONOTONIC ns) at which it passes
// each stage of the frame chain, plus per-frame counters, written as text lines by a low-priority
// thread (every 250 ms, and at exit). Off unless set; every hook tests Enabled() first.
//   V,<tick>,<ns>                       nx2:vi2 vsync timer tick (the guest starts frames on it)
//   F,<frame>,<ticks so far>,<stage ns...>,<counters...>,<gpu tick>
//   T,<frame>,<ns>,<cntpct>,<thread id>,<priority>,<core>,<cpu ticks>,<presenter>
//   E,<kind>,<ns>,<arg>                 events (EventKind)
// Stages (0 = not reached): present (guest present call entered), fence (its wait for the
// previous frame's fence ended), return (after vblank pacing), front (the front published the
// ring entry during which the back reached this frame's composite), back (composite started on
// GPUBack), recorded (composite recorded and flushed on GPUBack), submitted (VulkanWorker
// submitted the frame's last command buffer), gpu (that submission's timeline tick reached),
// presented (vkQueuePresentKHR returned), reached (the present's last fence was reached: the
// composite became due, on whichever thread reached it). Counters are cumulative, sampled at the
// back's composite: images created, staging upload bytes, mprotect calls, JIT blocks compiled,
// the front's time (ns) by state (parse, drain, list writers, pending semaphore, pending list
// copy, queue empty: nx2_back_idle_trace.h's FrontState), the back's idle ns (nothing published),
// and guest command list submissions. T lines (every EDEN_NX2_FRAME_LOG_THREADS presents, default
// 60): each guest thread's kernel CPU time (idle thread excluded, so no idle spin).
namespace Common::Nx2FrameLog {

enum class Stage : u32 {
    Present,
    Fence,
    Return,
    Front,
    Back,
    Recorded,
    Submitted,
    Gpu,
    Presented,
    Reached,
    Count,
};

enum class Counter : u32 {
    Images,
    UploadBytes,
    Mprotect,
    JitBlocks,
    FrontParseNs,
    FrontDrainNs,
    FrontWritersNs,
    FrontSemaphoreNs,
    FrontCopyNs,
    FrontQueueEmptyNs,
    BackIdleNs,
    Submits,
    DrainRingFullNs,  ///< front waits for ring room
    DrainAcquireNs,   ///< front drains for semaphore acquires / fence releases
    DrainOtherNs,     ///< every other drain / WaitFor site
    BackCpuNs,        ///< /proc schedstat of the registered threads: run time ...
    BackDelayNs,      ///< ... and time runnable but waiting for a CPU
    FrontCpuNs,
    FrontDelayNs,
    WorkerCpuNs,
    WorkerDelayNs,
    BackCycles,       ///< GPUBack's own hardware counters (perf_event_open on the thread):
    BackInstructions, ///< cycles, instructions retired, cache misses (user + kernel)
    BackCacheMisses,
    FrontCycles,      ///< the same for the front (GPU thread)
    FrontInstructions,
    FrontCacheMisses,
    WorkerCycles,     ///< ...and the VulkanWorker
    WorkerInstructions,
    WorkerCacheMisses,
    BackCoreIdleUsA,  ///< cpuidle residency (us, all states) of the lower CPU of the back's core
    BackCoreIdleUsB,  ///< ...and of its SMT sibling (EDEN_NX2_ISOLATE pins GPUBack to both)
    BackCpu,          ///< the CPU GPUBack ran on at the composite
    BackCoreIdleEntriesA, ///< cpuidle entries (usage, all states) of those two CPUs: a CPU that
    BackCoreIdleEntriesB, ///< woke from idle ran something (residency counts at idle exit only)
    Count,
};

enum class ThreadRole : u32 { Back, Front, Worker, Count };
/// The calling thread's schedstat is sampled at each composite as that role.
void RegisterThread(ThreadRole role);

enum class EventKind : u32 {
    Submit = 1,      ///< guest command list pushed to the front's queue (arg: channel)
    FrontState = 2,  ///< the front's state changed (arg: FrontState)
    BackIdle = 3,    ///< the back went idle (arg 1) or got work (arg 0)
    Reached = 4,     ///< a present's fences reached (arg: 1 on the scheduler owner, 0 handed off)
    VsyncWait = 5,   ///< a vsync tick, per event: threads waiting | signalled before << 8 | event << 16
};

namespace detail {
bool ReadEnabled();
}
inline bool Enabled() {
    static const bool enabled = detail::ReadEnabled();
    return enabled;
}
/// CLOCK_MONOTONIC (steady clock) in ns.
u64 Now();

/// Any thread. Only call when Enabled().
void Count(Counter counter, u64 amount = 1);
/// A counter kept elsewhere (cumulative), read at each composite.
void SetCounterSource(Counter counter, const std::atomic<u64>* source);
/// Any thread. Only call when Enabled().
void Event(EventKind kind, u64 ns, u64 arg);

/// nx2:vi2 vsync timer callback.
void OnVsync();
/// Guest present entered: a new frame id (from 1), remembered for the calling thread.
u64 BeginPresent();
/// The frame the calling thread's last BeginPresent started (0 if none or off).
u64 CurrentPresent();
void Stamp(u64 frame, Stage stage);
void StampAt(u64 frame, Stage stage, u64 ns);

/// Back, before the renderer composites `frame`: stamps Back and Front (`front_ns`), samples the
/// counters, and makes `frame` the CompositeFrame() until EndComposite.
void BeginComposite(u64 frame, u64 front_ns);
void EndComposite();
u64 CompositeFrame();
/// Recorded stage plus the frame's GPU tick.
void Recorded(u64 frame, u64 gpu_tick);

/// Front, GpuSplit::Publish: the ring is visible up to `position` now.
void NotePublish(u64 position);
/// When the entry starting at ring position `position` was first published (0 if unknown).
u64 PublishTimeOf(u64 position);

/// Presents between thread samples (0 = none).
u32 ThreadSampleInterval();
void ThreadSample(u64 frame, u64 ns, u64 cntpct, u64 thread_id, s32 priority, s32 core,
                  s64 cpu_ticks, bool presenter);

} // namespace Common::Nx2FrameLog
