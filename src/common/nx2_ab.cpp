// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include <dirent.h>
#include <sys/stat.h>

#include "common/logging.h"
#include "common/nx2_ab.h"

namespace Common::Nx2Ab {

std::atomic<u32> g_arm{0};
std::atomic<u32> g_phase{0};

namespace {

struct Config {
    u64 period_frames{};
    u64 period_ns{};
    u32 arms{2};
    u64 start_frames{};
    std::string gate; ///< start/stop with this file (EDEN_NX2_PROBE_GATE)
    u64 settle_frames{}; ///< EDEN_NX2_AB_SETTLE: a "mark" line this many frames into each phase
    bool enabled{};
};

const Config& Cfg() {
    static const Config config = [] {
        Config c;
        const char* period = std::getenv("EDEN_NX2_AB_PERIOD");
        if (period == nullptr || period[0] == '\0') {
            return c;
        }
        char* end = nullptr;
        const u64 n = std::strtoull(period, &end, 10);
        if (end != nullptr && std::strcmp(end, "ms") == 0) {
            c.period_ns = n * 1'000'000;
        } else {
            c.period_frames = n;
        }
        if (const char* arms = std::getenv("EDEN_NX2_AB_ARMS")) {
            c.arms = (std::max)(2U, static_cast<u32>(std::strtoul(arms, nullptr, 10)));
        }
        if (const char* start = std::getenv("EDEN_NX2_AB_START")) {
            if (std::strcmp(start, "gate") == 0) {
                const char* gate = std::getenv("EDEN_NX2_PROBE_GATE");
                c.gate = gate != nullptr ? gate : "";
            } else {
                c.start_frames = std::strtoull(start, nullptr, 10);
            }
        }
        if (const char* settle = std::getenv("EDEN_NX2_AB_SETTLE")) {
            c.settle_frames = std::strtoull(settle, nullptr, 10);
        }
        c.enabled = n != 0;
        if (c.enabled) {
            LOG_INFO(HW_GPU, "nx2 AB: period {}{}, {} arms, start {}", n,
                     c.period_ns != 0 ? " ms" : " frames", c.arms,
                     c.gate.empty() ? std::to_string(c.start_frames) : "at the gate file");
        }
        return c;
    }();
    return config;
}

u64 NowNs() {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<u64>(ts.tv_sec) * 1'000'000'000ULL + static_cast<u64>(ts.tv_nsec);
}

u64 ProcessCpuNs() {
    timespec ts{};
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
    return static_cast<u64>(ts.tv_sec) * 1'000'000'000ULL + static_cast<u64>(ts.tv_nsec);
}

/// On-CPU ns of the process's threads by name (/proc/self/task/<tid>/schedstat), summed over the
/// threads of each name. Thread ids are cached; names are rescanned when one goes missing.
class ThreadClocks {
public:
    std::map<std::string, u64> Read() {
        std::map<std::string, u64> out;
        bool missing = tids.empty();
        for (const auto& [tid, name] : tids) {
            std::ifstream f("/proc/self/task/" + tid + "/schedstat");
            u64 ns = 0;
            if (!(f >> ns)) {
                missing = true;
                continue;
            }
            out[name] += ns;
        }
        if (missing) {
            Scan();
            out.clear();
            for (const auto& [tid, name] : tids) {
                std::ifstream f("/proc/self/task/" + tid + "/schedstat");
                u64 ns = 0;
                if (f >> ns) {
                    out[name] += ns;
                }
            }
        }
        return out;
    }

private:
    void Scan() {
        tids.clear();
        DIR* dir = opendir("/proc/self/task");
        if (dir == nullptr) {
            return;
        }
        while (const dirent* entry = readdir(dir)) {
            if (entry->d_name[0] == '.') {
                continue;
            }
            std::ifstream f(std::string{"/proc/self/task/"} + entry->d_name + "/comm");
            std::string name;
            std::getline(f, name);
            // Only the threads the analysis needs (a full scan per flip would be slow).
            if (name == "GPUBack" || name == "GPU" || name == "VulkanWorker" ||
                name.rfind("CPUCore_", 0) == 0) {
                tids.emplace_back(entry->d_name, name.rfind("CPUCore_", 0) == 0 ? "guest" : name);
            }
        }
        closedir(dir);
    }
    std::vector<std::pair<std::string, std::string>> tids;
};

struct State {
    bool running{};
    bool done{};
    u64 frames{};
    u64 phase{};
    u64 phase_start_frames{};
    u64 phase_start_ns{};
    ThreadClocks clocks;
};

State& S() {
    static State state;
    return state;
}

bool GateExists(const std::string& path) {
    struct stat st{};
    return ::stat(path.c_str(), &st) == 0;
}

u32 ArmOf(u64 phase, u32 arms) {
    const u64 cycle = phase / arms;
    const u32 index = static_cast<u32>(phase % arms);
    return cycle % 2 == 0 ? index : arms - 1 - index;
}

void LogFlip(State& s, const char* what, u32 arm, u64 now) {
    const auto threads = s.clocks.Read();
    const auto get = [&](const char* name) {
        const auto it = threads.find(name);
        return it != threads.end() ? it->second : 0;
    };
    LOG_INFO(HW_GPU,
             "nx2 AB {} phase={} arm={} cycle={} frames={} ns={} cpu_back={} cpu_front={} "
             "cpu_worker={} cpu_guest={} cpu_process={}",
             what, s.phase, arm, s.phase / Cfg().arms, s.frames, now, get("GPUBack"), get("GPU"),
             get("VulkanWorker"), get("guest"), ProcessCpuNs());
}

} // namespace

Switch Parse(const char* env_value) {
    Switch sw;
    if (env_value == nullptr || env_value[0] == '\0') {
        return sw;
    }
    const std::string text{env_value};
    if (text == "ab" || text.rfind("ab:", 0) == 0) {
        sw.kind = Switch::Kind::Ab;
        sw.arm = 1;
        if (text.size() > 3) {
            const std::string rest = text.substr(3);
            const auto colon = rest.find(':');
            sw.arm = static_cast<u32>(std::strtoul(rest.substr(0, colon).c_str(), nullptr, 10));
            sw.value = colon == std::string::npos ? "1" : rest.substr(colon + 1);
        } else {
            sw.value = "1";
        }
        return sw;
    }
    sw.kind = Switch::Kind::On;
    sw.value = text;
    return sw;
}

bool Enabled() {
    return Cfg().enabled;
}

void OnFrame() {
    const Config& c = Cfg();
    if (!c.enabled) {
        return;
    }
    State& s = S();
    if (s.done) {
        return;
    }
    ++s.frames;
    const u64 now = NowNs();
    if (!s.running) {
        const bool start = c.gate.empty() ? s.frames >= c.start_frames
                                          : (s.frames % 16 == 0 && GateExists(c.gate));
        if (!start) {
            return;
        }
        s.running = true;
        s.phase = 0;
        s.phase_start_frames = s.frames;
        s.phase_start_ns = now;
        const u32 arm = ArmOf(0, c.arms);
        g_arm.store(arm, std::memory_order_relaxed);
        g_phase.fetch_add(1, std::memory_order_relaxed);
        LogFlip(s, "start", arm, now);
        return;
    }
    if (!c.gate.empty() && s.frames % 16 == 0 && !GateExists(c.gate)) {
        g_arm.store(0, std::memory_order_relaxed);
        g_phase.fetch_add(1, std::memory_order_relaxed);
        LogFlip(s, "end", 0, now);
        s.done = true;
        return;
    }
    if (c.settle_frames != 0 && s.frames - s.phase_start_frames == c.settle_frames) {
        LogFlip(s, "mark", Arm(), now);
    }
    const bool due = c.period_ns != 0 ? now - s.phase_start_ns >= c.period_ns
                                      : s.frames - s.phase_start_frames >= c.period_frames;
    if (!due) {
        return;
    }
    ++s.phase;
    s.phase_start_frames = s.frames;
    s.phase_start_ns = now;
    const u32 arm = ArmOf(s.phase, c.arms);
    g_arm.store(arm, std::memory_order_relaxed);
    g_phase.fetch_add(1, std::memory_order_relaxed);
    LogFlip(s, "flip", arm, now);
}

} // namespace Common::Nx2Ab
