// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <memory>
#include <mutex>
#include <span>
#include <thread>
#include <vector>

#include "common/common_types.h"
#include "common/container/unordered_set.h"
#include "core/hardware_properties.h"

namespace Dynarmic::A64 {
class Jit;
}

namespace Core {

/// JIT warm-up. The blocks execution compiles in the application's code image are recorded,
/// relative to the image, in <cache dir>/jit/<program id>-<build id>.blocks. The next launch
/// compiles them on each guest core's host thread while that core has nothing to run, before the
/// game reaches them. Blocks one core compiles are queued for the others as well.
class JitWarmup {
public:
    struct Image {
        u64 program_id{};
        std::array<u8, 0x20> build_id{};
        u64 base{}; ///< Start of the code image (rtld, main, subsdk*, sdk)
        u64 size{};
    };

    /// Longest one Run call compiles before it returns to the idle loop. The stop flag is checked
    /// before every block, whatever the slice.
    static constexpr u64 SliceNs = 2'000'000;
    /// Code cache left for the blocks execution compiles: each core stops precompiling there, so
    /// warm-up never fills the cache to the point where dynarmic clears it.
    static constexpr size_t KeepFreeBytes = size_t{128} << 20;
    /// Blocks listed per title (8 bytes each in the file and the list); Ocarina of Time has 646k.
    static constexpr size_t MaxEntries = 2'000'000;

    /// The block list file: a header naming the image, then append-only chunks of entries, each
    /// with its own checksum.
    struct File {
        static constexpr u32 Version = 1;
        static constexpr size_t HeaderSize = 64;
        static constexpr size_t ChunkHeaderSize = 16;
        static constexpr size_t MaxChunkEntries = 65536;
        /// Read no further: MaxEntries entries take 16 MiB plus their chunk headers.
        static constexpr u64 MaxBytes = u64{64} << 20;

        struct Contents {
            bool header_ok{};
            size_t valid_bytes{};    ///< Header and whole, valid chunks; anything after is dropped
            std::vector<u64> entries; ///< Unique, in file order, at most max_entries
        };

        [[nodiscard]] static std::vector<u8> MakeHeader(const Image& image);
        /// Appends entries to out as chunks of at most MaxChunkEntries.
        static void AppendChunks(std::vector<u8>& out, std::span<const u64> entries);
        /// Stops at the first chunk that is cut short, fails its checksum or holds an entry
        /// that is not a block of image.
        [[nodiscard]] static Contents Parse(std::span<const u8> data, const Image& image,
                                            size_t max_entries = MaxEntries);
    };

    /// nullptr when the image is unknown.
    [[nodiscard]] static std::shared_ptr<JitWarmup> Create(const Image& image);

    explicit JitWarmup(const Image& image);
    ~JitWarmup();
    JitWarmup(const JitWarmup&) = delete;
    JitWarmup& operator=(const JitWarmup&) = delete;

    /// Execution compiled the block at location (an IR::LocationDescriptor value) on core.
    void OnCompiled(size_t core, u64 location);

    /// On core's host thread while it has nothing to run: compiles listed blocks into jit until
    /// stop is set, SliceNs has passed or nothing is left. Returns whether there may be more to
    /// compile.
    bool Run(size_t core, Dynarmic::A64::Jit& jit, const std::atomic<bool>& stop);

private:
    struct alignas(64) CoreWork {
        size_t cursor{}; ///< Next index of list to compile (the core's thread only)
        bool full{};     ///< Out of code space: no more precompiles (the core's thread only)
        bool reported_done{};
        std::atomic<u64> compiled{};
        std::atomic<u64> present{};
        std::atomic<u64> unreadable{};
        std::atomic<u64> compile_ns{};
        std::atomic<u64> max_compile_ns{}; ///< Longest single Precompile
        std::atomic<u64> cache_bytes{};
        std::atomic<u64> late{};     ///< Execution compiled a listed block this core had not yet
        std::atomic<u64> unlisted{}; ///< Execution compiled a block no list had
        std::atomic<u64> outside{};  ///< Execution compiled a block outside the code image
    };

    void Load();
    void WriterLoop();
    void MaybeLog(bool force);

    [[nodiscard]] bool InImage(u64 location) const;
    [[nodiscard]] u64 ToEntry(u64 location) const;
    [[nodiscard]] u64 ToLocation(u64 entry) const;

    Image image;
    std::filesystem::path path;
    size_t loaded{};

    std::mutex mutex;
    Common::unordered_set<u64> known; ///< Entries listed or recorded
    std::deque<u64> list;             ///< Locations to precompile, in first-compiled order
    std::atomic<size_t> list_size{};
    std::vector<u64> pending;         ///< Entries to append to the file
    std::condition_variable pending_cv;
    bool file_ok{true};
    bool stopping{};

    std::array<CoreWork, Hardware::NUM_CPU_CORES> cores{};
    std::atomic<u64> last_log_ns{};
    std::atomic<u32> done_cores{};

    std::thread writer; ///< Joined (after a last write) by the destructor
};

} // namespace Core
