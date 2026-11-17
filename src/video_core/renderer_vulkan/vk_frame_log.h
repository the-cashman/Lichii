// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <utility>

#include "common/common_types.h"
#include "common/polyfill_thread.h"

namespace Vulkan {

class MasterSemaphore;

/// EDEN_NX2_FRAME_LOG: stamps each frame's GPU stage when its last submission's timeline tick is
/// reached, from a low-priority thread blocked in vkWaitSemaphores (frames in submission order).
class FrameLogGpuWaiter {
public:
    explicit FrameLogGpuWaiter(MasterSemaphore& semaphore);
    ~FrameLogGpuWaiter();

    void Push(u64 frame, u64 tick);

private:
    void Run(std::stop_token stop_token);

    MasterSemaphore& semaphore;
    std::mutex mutex;
    std::condition_variable_any cv;
    std::deque<std::pair<u64, u64>> queue;
    std::jthread thread;
};

} // namespace Vulkan
