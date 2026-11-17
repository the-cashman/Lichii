// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <chrono>
#include <cstdlib>

#include "common/logging.h"
#include "common/thread.h"
#include "video_core/renderer_vulkan/nx2_present_log.h"
#include "video_core/vulkan_common/vulkan_device.h"

namespace Vulkan {

bool Nx2PresentLog::Wanted() {
    const char* path = std::getenv("EDEN_NX2_PRESENT_LOG");
    return path != nullptr && *path != '\0';
}

std::unique_ptr<Nx2PresentLog> Nx2PresentLog::Create(const Device& device) {
    if (!Wanted()) {
        return nullptr;
    }
    const char* path = std::getenv("EDEN_NX2_PRESENT_LOG");
    std::FILE* file = std::fopen(path, "w");
    if (file == nullptr) {
        LOG_ERROR(Render_Vulkan, "EDEN_NX2_PRESENT_LOG: can't open {}", path);
        return nullptr;
    }
    std::fprintf(file, "seq,frame,composite_ns,ready_ns,release_ns,call_ns,return_ns,copy_ns,"
                       "interval_ns,late,present_id\n");
    // Ties the file's clock to the log's timestamps (the TAS lines of a bench).
    LOG_INFO(Render_Vulkan, "EDEN_NX2_PRESENT_LOG {} anchor steady_ns={}", path, Now());
    return std::make_unique<Nx2PresentLog>(device, file);
}

Nx2PresentLog::Nx2PresentLog(const Device& device_, std::FILE* file_)
    : device{device_}, file{file_} {
    thread = std::jthread([this](std::stop_token token) { Run(token); });
}

Nx2PresentLog::~Nx2PresentLog() {
    thread.request_stop();
    cv.notify_all();
    thread.join();
    std::fclose(file);
}

s64 Nx2PresentLog::Now() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

void Nx2PresentLog::Push(const Record& record) {
    if (record.fence != VK_NULL_HANDLE && record.ready_ns == 0) {
        pending[record.frame_index % pending.size()].store(true, std::memory_order_release);
    }
    {
        std::scoped_lock lock{mutex};
        records.push_back(record);
    }
    cv.notify_one();
}

void Nx2PresentLog::PushTiming(const Timing& timing) {
    {
        std::scoped_lock lock{mutex};
        timings.push_back(timing);
    }
    cv.notify_one();
}

void Nx2PresentLog::PushShown(u64 present_id, s64 shown_ns) {
    {
        std::scoped_lock lock{mutex};
        shown.emplace_back(present_id, shown_ns);
    }
    cv.notify_one();
}

void Nx2PresentLog::WaitFrame(u32 frame_index) {
    auto& flag = pending[frame_index % pending.size()];
    while (flag.load(std::memory_order_acquire)) {
        flag.wait(true, std::memory_order_acquire);
    }
}

void Nx2PresentLog::Run(std::stop_token token) {
    Common::SetCurrentThreadName("Nx2PresentLog");
    const auto& dld = device.GetDispatchLoader();
    const VkDevice logical = *device.GetLogical();
    while (true) {
        Record record;
        std::deque<Timing> new_timings;
        std::deque<std::pair<u64, s64>> new_shown;
        bool have_record = false;
        {
            std::unique_lock lock{mutex};
            cv.wait(lock, token, [this] {
                return !records.empty() || !timings.empty() || !shown.empty();
            });
            if (records.empty() && timings.empty() && shown.empty()) {
                break; // stop requested with nothing left
            }
            new_timings.swap(timings);
            new_shown.swap(shown);
            if (!records.empty()) {
                record = records.front();
                records.pop_front();
                have_record = true;
            }
        }
        for (const Timing& timing : new_timings) {
            std::fprintf(file, "T,%u,%llu,%llu,%llu,%llu\n", timing.present_id,
                         static_cast<unsigned long long>(timing.desired),
                         static_cast<unsigned long long>(timing.actual),
                         static_cast<unsigned long long>(timing.earliest),
                         static_cast<unsigned long long>(timing.margin));
        }
        for (const auto& [id, ns] : new_shown) {
            std::fprintf(file, "W,%llu,%lld\n", static_cast<unsigned long long>(id),
                         static_cast<long long>(ns));
        }
        if (!have_record) {
            continue;
        }
        s64 copy_ns = record.ready_ns;
        if (copy_ns == 0 && record.fence != VK_NULL_HANDLE) {
            // 1 s at most: a lost frame must not hang the presenter's next GetRenderFrame.
            const VkResult result =
                dld.vkWaitForFences(logical, 1, &record.fence, VK_TRUE, 1'000'000'000ULL);
            copy_ns = result == VK_SUCCESS ? Now() : -1;
            auto& flag = pending[record.frame_index % pending.size()];
            flag.store(false, std::memory_order_release);
            flag.notify_all();
        }
        std::fprintf(file, "%llu,%u,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%d,%llu\n",
                     static_cast<unsigned long long>(record.seq), record.frame_index,
                     static_cast<long long>(record.composite_ns),
                     static_cast<long long>(record.ready_ns),
                     static_cast<long long>(record.release_ns),
                     static_cast<long long>(record.call_ns),
                     static_cast<long long>(record.return_ns), static_cast<long long>(copy_ns),
                     static_cast<long long>(record.interval_ns), record.late ? 1 : 0,
                     static_cast<unsigned long long>(record.present_id));
    }
    std::fflush(file);
}

} // namespace Vulkan
