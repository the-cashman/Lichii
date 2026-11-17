// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <cstring>
#include <fcntl.h>
#include <cerrno>
#include <sched.h>
#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "common/logging.h"
#include "common/nx2_frame_log.h"
#include "common/thread.h"

namespace Common::Nx2FrameLog {
namespace {

constexpr size_t Stages = static_cast<size_t>(Stage::Count);
constexpr size_t Counters = static_cast<size_t>(Counter::Count);
constexpr u64 FrameSlots = 4096;
constexpr u64 TickSlots = 4096;
constexpr u64 PublishSlots = 16384;
constexpr u64 EventSlots = 1 << 17;
/// A frame is written once this many newer frames were presented (its GPU and present stamps
/// come a few frames after its present).
constexpr u64 WriteLag = 64;

struct Slot {
    std::atomic<u64> frame{};
    std::atomic<u64> ticks{};
    std::array<std::atomic<u64>, Stages> stamps{};
    std::array<std::atomic<u64>, Counters> counters{};
    std::atomic<u64> gpu_tick{};
};

struct PublishSlot {
    std::atomic<u64> position{};
    std::atomic<u64> ns{};
};

struct EventSlot {
    std::atomic<u64> seq{}; ///< index + 1 once written
    u64 kind, ns, arg;
};

struct ThreadRow {
    u64 frame, ns, cntpct, thread_id;
    s32 priority, core;
    s64 cpu_ticks;
    bool presenter;
};

struct State {
    std::FILE* file{};
    std::array<Slot, FrameSlots> slots{};
    std::array<std::atomic<u64>, TickSlots> tick_ns{};
    std::array<PublishSlot, PublishSlots> publishes{};
    std::array<EventSlot, EventSlots> events{};
    std::atomic<u64> event_head{};
    u64 written_events{};
    std::array<std::atomic<u64>, Counters> counters{};
    std::array<const std::atomic<u64>*, Counters> sources{};
    std::atomic<u64> ticks{};
    std::atomic<u64> frames{};
    std::atomic<u64> publish_count{};
    std::mutex mutex; ///< written_*, the file
    std::mutex rows_mutex;
    std::vector<ThreadRow> thread_rows;
    u64 written_frames{};
    u64 written_ticks{};

    explicit State(const char* path) {
        file = std::fopen(path, "w");
        if (file == nullptr) {
            LOG_ERROR(HW_GPU, "EDEN_NX2_FRAME_LOG: can't open {}", path);
            return;
        }
        std::fprintf(file, "# nx2 frame log v2, CLOCK_MONOTONIC ns\n"
                           "# V,tick,ns\n"
                           "# F,frame,ticks,present,fence,return,front,back,recorded,submitted,gpu,"
                           "presented,reached,images,upload_bytes,mprotect,jit_blocks,front_parse_ns,"
                           "front_drain_ns,front_writers_ns,front_semaphore_ns,front_copy_ns,"
                           "front_queue_empty_ns,back_idle_ns,submits,drain_ringfull_ns,drain_acquire_ns,"
                           "drain_other_ns,back_cpu_ns,back_delay_ns,front_cpu_ns,front_delay_ns,"
                           "worker_cpu_ns,worker_delay_ns,back_cycles,back_instructions,back_cache_misses,"
                           "front_cycles,front_instructions,front_cache_misses,worker_cycles,"
                           "worker_instructions,worker_cache_misses,core_idle_us_a,core_idle_us_b,back_cpu,"
                           "gpu_tick\n"
                           "# E,kind,ns,arg (1 submit, 2 front state, 3 back idle, 4 reached)\n"
                           "# T,frame,ns,cntpct,thread_id,priority,core,cpu_ticks,presenter\n");
        std::fflush(file);
        LOG_INFO(HW_GPU, "EDEN_NX2_FRAME_LOG: writing {}", path);
        std::thread([this] { Writer(); }).detach();
        std::atexit([] { Flush(true); });
    }

    void Write(bool all) {
        std::scoped_lock lock{mutex};
        if (file == nullptr) {
            return;
        }
        const u64 tick_end = ticks.load(std::memory_order_acquire);
        if (tick_end - written_ticks > TickSlots) {
            written_ticks = tick_end - TickSlots;
        }
        for (; written_ticks < tick_end; ++written_ticks) {
            const u64 tick = written_ticks + 1;
            std::fprintf(file, "V,%llu,%llu\n", static_cast<unsigned long long>(tick),
                         static_cast<unsigned long long>(
                             tick_ns[tick % TickSlots].load(std::memory_order_relaxed)));
        }
        const u64 frame_count = frames.load(std::memory_order_acquire);
        const u64 frame_end = all ? frame_count : (frame_count > WriteLag ? frame_count - WriteLag : 0);
        if (frame_count - written_frames > FrameSlots) {
            written_frames = frame_count - FrameSlots;
        }
        for (; written_frames < frame_end; ++written_frames) {
            const u64 frame = written_frames + 1;
            const Slot& slot = slots[frame % FrameSlots];
            if (slot.frame.load(std::memory_order_acquire) != frame) {
                continue;
            }
            std::fprintf(file, "F,%llu,%llu", static_cast<unsigned long long>(frame),
                         static_cast<unsigned long long>(slot.ticks.load(std::memory_order_relaxed)));
            for (const auto& stamp : slot.stamps) {
                std::fprintf(file, ",%llu",
                             static_cast<unsigned long long>(stamp.load(std::memory_order_relaxed)));
            }
            for (const auto& counter : slot.counters) {
                std::fprintf(file, ",%llu",
                             static_cast<unsigned long long>(counter.load(std::memory_order_relaxed)));
            }
            std::fprintf(file, ",%llu\n",
                         static_cast<unsigned long long>(slot.gpu_tick.load(std::memory_order_relaxed)));
        }
        {
            const u64 head = event_head.load(std::memory_order_acquire);
            if (head - written_events > EventSlots) {
                written_events = head - EventSlots;
            }
            for (; written_events < head; ++written_events) {
                const EventSlot& e = events[written_events % EventSlots];
                if (e.seq.load(std::memory_order_acquire) != written_events + 1) {
                    break; // reserved, not written yet: next time
                }
                std::fprintf(file, "E,%llu,%llu,%llu\n", static_cast<unsigned long long>(e.kind),
                             static_cast<unsigned long long>(e.ns),
                             static_cast<unsigned long long>(e.arg));
            }
        }
        std::vector<ThreadRow> rows;
        {
            std::scoped_lock rows_lock{rows_mutex};
            rows.swap(thread_rows);
        }
        for (const ThreadRow& row : rows) {
            std::fprintf(file, "T,%llu,%llu,%llu,%llu,%d,%d,%lld,%d\n",
                         static_cast<unsigned long long>(row.frame),
                         static_cast<unsigned long long>(row.ns),
                         static_cast<unsigned long long>(row.cntpct),
                         static_cast<unsigned long long>(row.thread_id), row.priority, row.core,
                         static_cast<long long>(row.cpu_ticks), row.presenter ? 1 : 0);
        }
        std::fflush(file);
    }

    void Writer() {
        Common::SetCurrentThreadName("Nx2FrameLog");
        Common::SetCurrentThreadPriority(Common::ThreadPriority::Low);
        for (;;) {
            std::this_thread::sleep_for(std::chrono::milliseconds{250});
            Write(false);
        }
    }

    static void Flush(bool all);
};

const char* Path() {
    static const char* const path = [] {
        const char* value = std::getenv("EDEN_NX2_FRAME_LOG");
        return value != nullptr && value[0] != '\0' ? value : nullptr;
    }();
    return path;
}

State& S() {
    // Never destroyed: the writer thread and the atexit flush may outlive static destructors.
    static State* const state = new State(Path());
    return *state;
}

void State::Flush(bool all) {
    S().Write(all);
}

std::array<std::atomic<int>, static_cast<size_t>(ThreadRole::Count)> g_thread_fds{};
/// GPUBack's core (when pinned to <= 2 CPUs): the cpuidle state time files of each CPU.
std::array<std::vector<int>, 2> g_core_idle_fds;

u64 ReadIdleUs(const std::vector<int>& fds) {
    u64 sum = 0;
    char buf[32];
    for (const int fd : fds) {
        const ssize_t n = pread(fd, buf, sizeof(buf) - 1, 0);
        if (n > 0) {
            buf[n] = 0;
            sum += std::strtoull(buf, nullptr, 10);
        }
    }
    return sum;
}

/// Per role: the thread's perf event group (leader cycles, then instructions, cache misses).
std::array<std::atomic<int>, static_cast<size_t>(ThreadRole::Count)> g_perf_fds{-1, -1, -1};

int PerfOpen(u64 config, int group) {
    perf_event_attr attr{};
    attr.type = PERF_TYPE_HARDWARE;
    attr.size = sizeof(attr);
    attr.config = config;
    attr.exclude_hv = 1;
    attr.read_format = PERF_FORMAT_GROUP;
    return static_cast<int>(syscall(SYS_perf_event_open, &attr, 0, -1, group, PERF_FLAG_FD_CLOEXEC));
}

/// run ns and wait ns of a schedstat file (fd kept open; pread from 0).
bool ReadSchedstat(int fd, u64& run, u64& wait) {
    char buf[96];
    const ssize_t n = pread(fd, buf, sizeof(buf) - 1, 0);
    if (n <= 0) {
        return false;
    }
    buf[n] = 0;
    char* end{};
    run = std::strtoull(buf, &end, 10);
    wait = std::strtoull(end, nullptr, 10);
    return true;
}

thread_local u64 t_present_frame{};
u64 g_composite_frame{}; ///< back thread only

Slot* SlotOf(u64 frame) {
    if (frame == 0) {
        return nullptr;
    }
    Slot& slot = S().slots[frame % FrameSlots];
    return slot.frame.load(std::memory_order_relaxed) == frame ? &slot : nullptr;
}

} // namespace

bool detail::ReadEnabled() {
    return Path() != nullptr;
}

u64 Now() {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count());
}

void Count(Counter counter, u64 amount) {
    S().counters[static_cast<size_t>(counter)].fetch_add(amount, std::memory_order_relaxed);
}

void SetCounterSource(Counter counter, const std::atomic<u64>* source) {
    if (Enabled()) {
        S().sources[static_cast<size_t>(counter)] = source;
    }
}

void Event(EventKind kind, u64 ns, u64 arg) {
    State& s = S();
    const u64 index = s.event_head.fetch_add(1, std::memory_order_relaxed);
    EventSlot& e = s.events[index % EventSlots];
    e.kind = static_cast<u64>(kind);
    e.ns = ns;
    e.arg = arg;
    e.seq.store(index + 1, std::memory_order_release);
}

void OnVsync() {
    State& s = S();
    const u64 tick = s.ticks.load(std::memory_order_relaxed) + 1;
    s.tick_ns[tick % TickSlots].store(Now(), std::memory_order_relaxed);
    s.ticks.store(tick, std::memory_order_release);
}

u64 BeginPresent() {
    State& s = S();
    const u64 now = Now();
    const u64 frame = s.frames.load(std::memory_order_relaxed) + 1;
    Slot& slot = s.slots[frame % FrameSlots];
    slot.frame.store(0, std::memory_order_relaxed);
    for (auto& stamp : slot.stamps) {
        stamp.store(0, std::memory_order_relaxed);
    }
    for (auto& counter : slot.counters) {
        counter.store(0, std::memory_order_relaxed);
    }
    slot.gpu_tick.store(0, std::memory_order_relaxed);
    slot.ticks.store(s.ticks.load(std::memory_order_acquire), std::memory_order_relaxed);
    slot.stamps[static_cast<size_t>(Stage::Present)].store(now, std::memory_order_relaxed);
    slot.frame.store(frame, std::memory_order_release);
    s.frames.store(frame, std::memory_order_release);
    t_present_frame = frame;
    return frame;
}

u64 CurrentPresent() {
    return Enabled() ? t_present_frame : 0;
}

void Stamp(u64 frame, Stage stage) {
    StampAt(frame, stage, Now());
}

void StampAt(u64 frame, Stage stage, u64 ns) {
    if (Slot* const slot = SlotOf(frame)) {
        slot->stamps[static_cast<size_t>(stage)].store(ns, std::memory_order_relaxed);
    }
}

void RegisterThread(ThreadRole role) {
    if (!Enabled()) {
        return;
    }
    const std::string path = "/proc/self/task/" + std::to_string(static_cast<long>(gettid())) +
                             "/schedstat";
    const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd >= 0) {
        g_thread_fds[static_cast<size_t>(role)].store(fd, std::memory_order_release);
    }
    if (role == ThreadRole::Back) {
        cpu_set_t set;
        CPU_ZERO(&set);
        if (sched_getaffinity(0, sizeof(set), &set) == 0 && CPU_COUNT(&set) <= 2) {
            size_t slot = 0;
            for (int c = 0; c < CPU_SETSIZE && slot < 2; ++c) {
                if (!CPU_ISSET(c, &set)) {
                    continue;
                }
                for (int state = 0; state < 16; ++state) {
                    const std::string file = "/sys/devices/system/cpu/cpu" + std::to_string(c) +
                                             "/cpuidle/state" + std::to_string(state) + "/time";
                    const int fd = open(file.c_str(), O_RDONLY | O_CLOEXEC);
                    if (fd < 0) {
                        break;
                    }
                    g_core_idle_fds[slot].push_back(fd);
                }
                ++slot;
            }
        }
    }
    {
        const int leader = PerfOpen(PERF_COUNT_HW_CPU_CYCLES, -1);
        if (leader >= 0) {
            PerfOpen(PERF_COUNT_HW_INSTRUCTIONS, leader);
            PerfOpen(PERF_COUNT_HW_CACHE_MISSES, leader);
            g_perf_fds[static_cast<size_t>(role)].store(leader, std::memory_order_release);
        } else {
            LOG_WARNING(HW_GPU, "EDEN_NX2_FRAME_LOG: no perf counters for thread role {} ({})",
                        static_cast<u32>(role), errno);
        }
    }
}

void BeginComposite(u64 frame, u64 front_ns) {
    g_composite_frame = frame;
    Slot* const slot = SlotOf(frame);
    if (slot == nullptr) {
        return;
    }
    {
        State& st = S();
        for (size_t role = 0; role < g_thread_fds.size(); ++role) {
            const int fd = g_thread_fds[role].load(std::memory_order_acquire);
            u64 run{}, wait{};
            if (fd > 0 && ReadSchedstat(fd, run, wait)) {
                const size_t base = static_cast<size_t>(Counter::BackCpuNs) + 2 * role;
                st.counters[base].store(run, std::memory_order_relaxed);
                st.counters[base + 1].store(wait, std::memory_order_relaxed);
            }
        }
        st.counters[static_cast<size_t>(Counter::BackCoreIdleUsA)].store(ReadIdleUs(g_core_idle_fds[0]));
        st.counters[static_cast<size_t>(Counter::BackCoreIdleUsB)].store(ReadIdleUs(g_core_idle_fds[1]));
        st.counters[static_cast<size_t>(Counter::BackCpu)].store(static_cast<u64>(sched_getcpu()));
        for (size_t role = 0; role < g_perf_fds.size(); ++role) {
            const int fd = g_perf_fds[role].load(std::memory_order_acquire);
            u64 values[1 + 3]{}; // nr, then the values
            if (fd >= 0 && read(fd, values, sizeof(values)) > 0) {
                for (u64 i = 0; i < values[0] && i < 3; ++i) {
                    st.counters[static_cast<size_t>(Counter::BackCycles) + 3 * role + i].store(
                        values[1 + i]);
                }
            }
        }
    }
    slot->stamps[static_cast<size_t>(Stage::Back)].store(Now(), std::memory_order_relaxed);
    slot->stamps[static_cast<size_t>(Stage::Front)].store(front_ns, std::memory_order_relaxed);
    State& s = S();
    for (size_t i = 0; i < Counters; ++i) {
        const std::atomic<u64>* const source = s.sources[i];
        slot->counters[i].store(source != nullptr ? source->load(std::memory_order_relaxed)
                                                  : s.counters[i].load(std::memory_order_relaxed),
                                std::memory_order_relaxed);
    }
}

void EndComposite() {
    g_composite_frame = 0;
}

u64 CompositeFrame() {
    return Enabled() ? g_composite_frame : 0;
}

void Recorded(u64 frame, u64 gpu_tick) {
    if (Slot* const slot = SlotOf(frame)) {
        slot->stamps[static_cast<size_t>(Stage::Recorded)].store(Now(), std::memory_order_relaxed);
        slot->gpu_tick.store(gpu_tick, std::memory_order_relaxed);
    }
}

void NotePublish(u64 position) {
    State& s = S();
    const u64 index = s.publish_count.load(std::memory_order_relaxed);
    PublishSlot& slot = s.publishes[index % PublishSlots];
    slot.position.store(~u64{0}, std::memory_order_relaxed);
    slot.ns.store(Now(), std::memory_order_relaxed);
    slot.position.store(position, std::memory_order_release);
    s.publish_count.store(index + 1, std::memory_order_release);
}

u64 PublishTimeOf(u64 position) {
    State& s = S();
    const u64 count = s.publish_count.load(std::memory_order_acquire);
    // Newest first: the answer is the oldest publish that reached past `position`.
    u64 found = 0;
    for (u64 i = 0; i < count && i < 4096; ++i) {
        const PublishSlot& slot = s.publishes[(count - 1 - i) % PublishSlots];
        const u64 published = slot.position.load(std::memory_order_acquire);
        if (published == ~u64{0}) {
            break;
        }
        if (published <= position) {
            break;
        }
        found = slot.ns.load(std::memory_order_relaxed);
    }
    return found;
}

u32 ThreadSampleInterval() {
    static const u32 interval = [] {
        if (!Enabled()) {
            return 0U;
        }
        const char* value = std::getenv("EDEN_NX2_FRAME_LOG_THREADS");
        return value != nullptr ? static_cast<u32>(std::strtoul(value, nullptr, 10)) : 60U;
    }();
    return interval;
}

void ThreadSample(u64 frame, u64 ns, u64 cntpct, u64 thread_id, s32 priority, s32 core,
                  s64 cpu_ticks, bool presenter) {
    State& s = S();
    std::scoped_lock lock{s.rows_mutex};
    s.thread_rows.push_back({frame, ns, cntpct, thread_id, priority, core, cpu_ticks, presenter});
}

} // namespace Common::Nx2FrameLog
