// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>

#include "common/common_types.h"
#include "common/polyfill_thread.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

class Device;

/**
 * EDEN_NX2_PRESENT_LOG=<file>: one CSV line per presented frame with the times (steady clock =
 * CLOCK_MONOTONIC, ns) of its hand-off by the composite, its release by the pacer, the
 * vkQueuePresentKHR call and return, and the GPU completion of its copy into the swapchain image
 * (the earliest the compositor can show it). With VK_GOOGLE_display_timing (gamescope's WSI layer)
 * also the compositor's own present times, as "T" lines. A thread waits for the copies and writes
 * the file, off the presenting thread.
 */
class Nx2PresentLog {
public:
    struct Record {
        u64 seq{};
        u32 frame_index{};
        s64 composite_ns{};
        s64 ready_ns{};   ///< copy done, when the presenter waited for it (pacing), else 0
        s64 release_ns{}; ///< pacer's release time, 0 without pacing
        s64 call_ns{};
        s64 return_ns{};
        s64 interval_ns{}; ///< pace interval at the time
        bool late{};
        u64 present_id{}; ///< #2028's present id (host vsync), 0 if none
        VkFence fence{}; ///< the copy's fence, waited for by the log thread when ready_ns is 0
    };

    struct Timing {
        u32 present_id;
        u64 desired;
        u64 actual;
        u64 earliest;
        u64 margin;
    };

    /// Null unless EDEN_NX2_PRESENT_LOG names a file that can be opened.
    static std::unique_ptr<Nx2PresentLog> Create(const Device& device);

    explicit Nx2PresentLog(const Device& device, std::FILE* file);
    ~Nx2PresentLog();

    static bool Wanted();

    void Push(const Record& record);
    void PushTiming(const Timing& timing);
    /// A present wait for `present_id` returned: the compositor showed it (#2028's wait thread).
    void PushShown(u64 present_id, s64 shown_ns);

    /// Waits until the log thread no longer waits on this frame's fence (before it is reset).
    void WaitFrame(u32 frame_index);

    [[nodiscard]] static s64 Now();

private:
    void Run(std::stop_token token);

    const Device& device;
    std::FILE* file;
    std::mutex mutex;
    std::condition_variable_any cv;
    std::deque<Record> records;
    std::deque<Timing> timings;
    std::deque<std::pair<u64, s64>> shown;
    std::array<std::atomic<bool>, 16> pending{};
    std::jthread thread;
};

} // namespace Vulkan
