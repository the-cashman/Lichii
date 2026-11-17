// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "common/nx2_frame_log.h"
#include "common/thread.h"
#include "video_core/renderer_vulkan/vk_frame_log.h"
#include "video_core/renderer_vulkan/vk_master_semaphore.h"

namespace Vulkan {

FrameLogGpuWaiter::FrameLogGpuWaiter(MasterSemaphore& semaphore_) : semaphore{semaphore_} {
    thread = std::jthread([this](std::stop_token stop_token) { Run(stop_token); });
}

FrameLogGpuWaiter::~FrameLogGpuWaiter() {
    thread.request_stop();
    cv.notify_all();
    if (thread.joinable()) {
        thread.join();
    }
}

void FrameLogGpuWaiter::Push(u64 frame, u64 tick) {
    {
        std::scoped_lock lock{mutex};
        // A waiter far behind (a stuck GPU) drops old frames rather than grow.
        if (queue.size() >= 256) {
            queue.pop_front();
        }
        queue.emplace_back(frame, tick);
    }
    cv.notify_one();
}

void FrameLogGpuWaiter::Run(std::stop_token stop_token) {
    Common::SetCurrentThreadName("Nx2FrameLogGpu");
    while (!stop_token.stop_requested()) {
        std::pair<u64, u64> item;
        {
            std::unique_lock lock{mutex};
            cv.wait(lock, stop_token, [this] { return !queue.empty(); });
            if (stop_token.stop_requested()) {
                return;
            }
            item = queue.front();
            queue.pop_front();
        }
        // 2 ms slices, so a stop request is seen soon.
        while (!semaphore.WaitTick(item.second, 2'000'000)) {
            if (stop_token.stop_requested()) {
                return;
            }
        }
        Common::Nx2FrameLog::Stamp(item.first, Common::Nx2FrameLog::Stage::Gpu);
    }
}

} // namespace Vulkan
