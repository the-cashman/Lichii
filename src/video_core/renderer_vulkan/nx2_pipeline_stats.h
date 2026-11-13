// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

#include "common/common_types.h"
#include "common/logging.h"

// EDEN_NX2_PIPELINE_STATS=1: where the time of each graphics pipeline built while playing goes
// (lowering, translation, SPIR-V emission, shader modules and layouts on the GPU thread; the
// driver compile on a worker; how long a frame waited for it) and how frames were paced. Logs a
// summary every 10 s and one at exit; =2 also logs one line per pipeline. Off: nothing is timed.
namespace Vulkan::Nx2Stats {

using Clock = std::chrono::steady_clock;

inline int Level() {
    static const int level = [] {
        const char* value = std::getenv("EDEN_NX2_PIPELINE_STATS");
        return value ? std::atoi(value) : 0;
    }();
    return level;
}

inline bool Enabled() {
    return Level() > 0;
}

enum class Phase : u32 {
    Lower,      ///< Native capture + lowering (RefreshStages, GPU thread)
    Env,        ///< GetGraphicsEnvironments
    Translate,  ///< CFG + TranslateProgram, all stages
    Spirv,      ///< ConvertLegacyToGeneric + EmitSPIRV, all stages
    Module,     ///< vkCreateShaderModule
    Layout,     ///< GraphicsPipeline ctor on the GPU thread (descriptor/pipeline layouts)
    GpuTotal,   ///< whole runtime CreateGraphicsPipeline() on the GPU thread
    Driver,     ///< vkCreateGraphicsPipelines (worker)
    Wait,       ///< a draw's command recording waited for an unbuilt pipeline (frame stall)
    IrCached,   ///< runtime pipelines whose every native stage came from the translation cache
    GplLibraries, ///< library pipeline: getting/creating its four libraries (worker)
    GplLink,      ///< library pipeline: the fast link (worker)
    GplLto,       ///< library pipeline: the optimized link (background worker)
    GplStageNew,  ///< a vertex/fragment library created at run time (cache hit or compile)
    RecordMono,   ///< per draw: binding a monolithic pipeline's state + descriptors (worker)
    RecordGpl,    ///< per draw: the same for a library pipeline (two descriptor sets)
    CsGpu,        ///< runtime compute pipeline: everything on the GPU thread (any source)
    CsTranslate,  ///< runtime compute pipeline: CFG + TranslateProgram (not cached)
    CsDriver,     ///< compute pipeline: vkCreateComputePipelines (worker, or the precompile)
    CsWait,       ///< a dispatch waited for an unbuilt compute pipeline (frame stall)
    CsPrebuild,   ///< precompile: one compute pipeline built from its cached translation
    Count,
};

inline constexpr std::array<const char*, static_cast<size_t>(Phase::Count)> phase_names{
    "lower", "env", "translate", "spirv", "module", "layout", "gpu_total", "driver", "wait",
    "all_stages_cached", "gpl_libs", "gpl_link", "gpl_lto", "gpl_stage_new", "record_mono",
    "record_gpl", "cs_gpu", "cs_translate", "cs_driver", "cs_wait", "cs_prebuild"};

struct Counter {
    std::atomic<u64> ns{};
    std::atomic<u64> count{};
    std::atomic<u64> max_ns{};
};

struct State {
    std::array<Counter, static_cast<size_t>(Phase::Count)> window;
    std::array<Counter, static_cast<size_t>(Phase::Count)> total;
    std::atomic<u64> driver_cache_hits{};
    std::atomic<u64> async_skipped_draws{};
    std::atomic<u64> boot_pipelines{};
    std::atomic<u64> boot_ns{};
    std::atomic<u64> ir_stage_hits{};       ///< runtime stages from the translation cache
    std::atomic<u64> ir_stage_misses{};     ///< runtime native stages translated
    std::atomic<u64> ir_env_mismatches{};   ///< cached but the environment inputs differ
    std::atomic<u64> ir_learned{};          ///< live translations stored (EDEN_NX2_IR_LEARN)
    std::atomic<u64> gpl_pipelines{};       ///< library pipelines linked
    std::atomic<u64> gpl_stage_reused{};    ///< stage libraries this session had already
    std::atomic<u64> gpl_stage_fast{};      ///< stage libraries created in < 1 ms (driver cache)
    std::atomic<u64> gpl_stage_slow{};      ///< stage libraries created in >= 1 ms (compiled)
    std::atomic<u64> gpl_monolithic{};      ///< native VS+PS pipelines not eligible (monolithic)
    std::atomic<u64> draws{};               ///< graphics draws recorded (window, reset each report)
    std::atomic<u64> gpl_draws{};           ///< of those, with a library pipeline
    std::atomic<u64> lto_draws{};           ///< of those, bound while its optimized link was done
    std::atomic<u64> cs_runtime{};          ///< compute pipelines needed at run time (new keys)
    std::atomic<u64> cs_prebuilt{};         ///< of those, a pipeline the precompile had built
    std::atomic<u64> cs_ir{};               ///< of those, built from a cached translation
    std::atomic<u64> cs_translated{};       ///< of those, translated (no usable record)
    std::atomic<u64> cs_prebuilt_mismatch{}; ///< a prebuilt pipeline whose record didn't match
    std::mutex frame_mutex;
    std::vector<float> frame_ms; ///< every presented frame's interval
    Clock::time_point last_frame{};
    Clock::time_point last_report{Clock::now()};
    size_t frames_reported{};
};

inline State& Get() {
    static State state;
    return state;
}

inline double Ms(Clock::duration d) {
    return std::chrono::duration<double, std::milli>(d).count();
}

inline void Max(std::atomic<u64>& target, u64 value) {
    u64 old = target.load(std::memory_order_relaxed);
    while (value > old && !target.compare_exchange_weak(old, value, std::memory_order_relaxed)) {
    }
}

inline void Add(Phase phase, Clock::duration d) {
    if (!Enabled()) {
        return;
    }
    const u64 ns = static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(d).count());
    for (auto* set : {&Get().window, &Get().total}) {
        Counter& c = (*set)[static_cast<size_t>(phase)];
        c.ns.fetch_add(ns, std::memory_order_relaxed);
        c.count.fetch_add(1, std::memory_order_relaxed);
        Max(c.max_ns, ns);
    }
}

/// Times one scope into a phase (does nothing when stats are off).
class Timer {
public:
    explicit Timer(Phase phase_) : phase{phase_}, begin{Enabled() ? Clock::now() : Clock::time_point{}} {}
    ~Timer() {
        if (Enabled()) {
            Add(phase, Clock::now() - begin);
        }
    }
    Timer(const Timer&) = delete;
    Timer& operator=(const Timer&) = delete;

private:
    Phase phase;
    Clock::time_point begin;
};

/// Per-thread phase durations of the pipeline being built (the runtime build reads them).
struct BuildTimes {
    Clock::duration translate{}, spirv{}, module{}, layout{};
    u32 cached_stages{};     ///< stages taken from the translation cache
    u32 translated_stages{}; ///< cacheable (native VS/PS) stages translated
};
inline thread_local BuildTimes build_times;
inline thread_local Clock::duration lowered_since_pipeline{};

/// A stage library a library pipeline got (GplLibraryCache::Result).
template <typename Result>
inline void CountGplLibrary(const Result& result) {
    State& s = Get();
    if (!result.created) {
        s.gpl_stage_reused.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    Add(Phase::GplStageNew, std::chrono::duration_cast<Clock::duration>(
                                std::chrono::duration<double, std::milli>(result.ms)));
    (result.ms < 1.0 ? s.gpl_stage_fast : s.gpl_stage_slow).fetch_add(1, std::memory_order_relaxed);
}

inline void Frame() {
    if (!Enabled()) {
        return;
    }
    State& s = Get();
    const auto now = Clock::now();
    std::scoped_lock lock{s.frame_mutex};
    if (s.last_frame.time_since_epoch().count() != 0) {
        s.frame_ms.push_back(static_cast<float>(Ms(now - s.last_frame)));
    }
    s.last_frame = now;
}

inline std::string FrameSummary(size_t from) {
    State& s = Get();
    std::vector<float> f;
    {
        std::scoped_lock lock{s.frame_mutex};
        if (from < s.frame_ms.size()) {
            f.assign(s.frame_ms.begin() + static_cast<std::ptrdiff_t>(from), s.frame_ms.end());
        }
    }
    if (f.empty()) {
        return "no frames";
    }
    double sum = 0, over50 = 0;
    size_t n33 = 0, n50 = 0, n100 = 0;
    for (const float ms : f) {
        sum += ms;
        n33 += ms > 33.4f;
        n50 += ms > 50.0f;
        n100 += ms > 100.0f;
        if (ms > 50.0f) {
            over50 += ms;
        }
    }
    std::vector<float> sorted = f;
    std::sort(sorted.begin(), sorted.end());
    const auto pct = [&](double p) {
        return sorted[std::min(sorted.size() - 1, static_cast<size_t>(p * sorted.size()))];
    };
    return fmt::format("{} frames in {:.1f} s, median {:.1f} ms, p99 {:.1f} ms, max {:.1f} ms, "
                       ">33ms {}, >50ms {} ({:.0f} ms total), >100ms {}",
                       f.size(), sum / 1000.0, pct(0.5), pct(0.99), sorted.back(), n33, n50, over50,
                       n100);
}

inline std::string PhaseSummary(std::array<Counter, static_cast<size_t>(Phase::Count)>& set,
                                bool reset) {
    std::string out;
    for (size_t i = 0; i < set.size(); ++i) {
        Counter& c = set[i];
        const u64 n = reset ? c.count.exchange(0) : c.count.load();
        const u64 ns = reset ? c.ns.exchange(0) : c.ns.load();
        const u64 mx = reset ? c.max_ns.exchange(0) : c.max_ns.load();
        if (n == 0) {
            continue;
        }
        out += fmt::format(" {}={}x/{:.1f}ms(max {:.1f})", phase_names[i], n, ns / 1e6, mx / 1e6);
    }
    return out.empty() ? std::string{" none"} : out;
}

/// From the GPU thread once per frame: a summary every 10 s.
inline void MaybeReport() {
    if (!Enabled()) {
        return;
    }
    State& s = Get();
    const auto now = Clock::now();
    if (now - s.last_report < std::chrono::seconds{10}) {
        return;
    }
    s.last_report = now;
    size_t from;
    {
        std::scoped_lock lock{s.frame_mutex};
        from = s.frames_reported;
        s.frames_reported = s.frame_ms.size();
    }
    LOG_INFO(Render_Vulkan,
             "NX2STATS window:{} | frames: {} | so far: stages cached {} translated {} env-mismatch "
             "{} learned {}, async-skipped draws {}, gpl pipelines {} (monolithic {}) stage libraries reused "
             "{} fast {} slow {} | draws {} (library {}, optimized {}) | compute so far: {} new "
             "(prebuilt {}, cached translation {}, translated {}, prebuilt mismatch {})",
             PhaseSummary(s.window, true), FrameSummary(from), s.ir_stage_hits.load(),
             s.ir_stage_misses.load(), s.ir_env_mismatches.load(), s.ir_learned.load(),
             s.async_skipped_draws.load(), s.gpl_pipelines.load(), s.gpl_monolithic.load(),
             s.gpl_stage_reused.load(), s.gpl_stage_fast.load(), s.gpl_stage_slow.load(),
             s.draws.exchange(0),
             s.gpl_draws.exchange(0), s.lto_draws.exchange(0), s.cs_runtime.load(),
             s.cs_prebuilt.load(), s.cs_ir.load(), s.cs_translated.load(),
             s.cs_prebuilt_mismatch.load());
}

inline void FinalReport() {
    if (!Enabled()) {
        return;
    }
    State& s = Get();
    LOG_INFO(Render_Vulkan, "NX2STATS total:{}", PhaseSummary(s.total, false));
    LOG_INFO(Render_Vulkan, "NX2STATS driver cache hits {}, async-skipped draws {}, boot-built {} in {:.0f} ms",
             s.driver_cache_hits.load(), s.async_skipped_draws.load(), s.boot_pipelines.load(),
             s.boot_ns.load() / 1e6);
    LOG_INFO(Render_Vulkan, "NX2STATS runtime stages: {} from the translation cache, {} translated "
             "({} cached with other environment inputs, {} stored for the next session)",
             s.ir_stage_hits.load(), s.ir_stage_misses.load(), s.ir_env_mismatches.load(),
             s.ir_learned.load());
    LOG_INFO(Render_Vulkan, "NX2STATS gpl: {} library pipelines, {} native VS+PS monolithic; stage "
             "libraries: {} reused, {} created fast (< 1 ms), {} created slow",
             s.gpl_pipelines.load(), s.gpl_monolithic.load(), s.gpl_stage_reused.load(),
             s.gpl_stage_fast.load(), s.gpl_stage_slow.load());
    LOG_INFO(Render_Vulkan, "NX2STATS compute: {} new at run time: {} prebuilt, {} from a cached "
             "translation, {} translated; {} prebuilt didn't match",
             s.cs_runtime.load(), s.cs_prebuilt.load(), s.cs_ir.load(), s.cs_translated.load(),
             s.cs_prebuilt_mismatch.load());
    LOG_INFO(Render_Vulkan, "NX2STATS frames: {}", FrameSummary(0));
}

} // namespace Vulkan::Nx2Stats
