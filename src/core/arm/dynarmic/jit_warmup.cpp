// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <chrono>
#include <cstring>
#include <span>
#include <string>

#include <dynarmic/interface/A64/a64.h>
#include <fmt/format.h>

#include "common/cityhash.h"
#include "common/fs/file.h"
#include "common/fs/fs.h"
#include "common/fs/fs_util.h"
#include "common/fs/path_util.h"
#include "common/logging.h"
#include "common/thread.h"
#include "core/arm/dynarmic/jit_warmup.h"

namespace Core {

namespace {

// As Dynarmic::A64::LocationDescriptor: PC in bits 0-55, FPCR bits at 37 + 19..26, single
// stepping at 57. Entries keep the FPCR bits and replace the PC by its offset in the image.
constexpr u64 PcMask = (u64{1} << 56) - 1;
constexpr u64 FpcrBits = u64{0x07C8'0000} << 37;
constexpr u64 SingleStepBit = u64{1} << 57;

constexpr std::array<char, 8> FileMagic{'N', 'X', '2', 'J', 'I', 'T', 'B', '\0'};
constexpr u32 ChunkMagic = 0x4B4E4843; // "CHNK"

u64 NowNs() {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count());
}

template <typename T>
void Put(std::span<u8> out, size_t offset, T value) {
    std::memcpy(out.data() + offset, &value, sizeof(T));
}

template <typename T>
T Get(std::span<const u8> in, size_t offset) {
    T value;
    std::memcpy(&value, in.data() + offset, sizeof(T));
    return value;
}

} // Anonymous namespace

std::vector<u8> JitWarmup::File::MakeHeader(const Image& image) {
    std::vector<u8> header(HeaderSize);
    std::memcpy(header.data(), FileMagic.data(), FileMagic.size());
    Put<u32>(header, 8, Version);
    Put<u64>(header, 16, image.program_id);
    std::memcpy(header.data() + 24, image.build_id.data(), image.build_id.size());
    Put<u64>(header, 56, image.size);
    return header;
}

void JitWarmup::File::AppendChunks(std::vector<u8>& out, std::span<const u64> entries) {
    for (size_t first = 0; first < entries.size(); first += MaxChunkEntries) {
        const size_t count = std::min(MaxChunkEntries, entries.size() - first);
        const size_t payload = count * sizeof(u64);
        const size_t offset = out.size();
        out.resize(offset + ChunkHeaderSize + payload);
        const std::span<u8> chunk{out.data() + offset, ChunkHeaderSize + payload};
        std::memcpy(chunk.data() + ChunkHeaderSize, entries.data() + first, payload);
        Put<u32>(chunk, 0, ChunkMagic);
        Put<u32>(chunk, 4, static_cast<u32>(count));
        Put<u64>(chunk, 8,
                 Common::CityHash64(reinterpret_cast<const char*>(chunk.data() + ChunkHeaderSize),
                                    payload));
    }
}

JitWarmup::File::Contents JitWarmup::File::Parse(std::span<const u8> data, const Image& image,
                                                 size_t max_entries) {
    Contents contents;
    const std::vector<u8> header = MakeHeader(image);
    contents.header_ok =
        data.size() >= HeaderSize && std::memcmp(data.data(), header.data(), HeaderSize) == 0;
    if (!contents.header_ok) {
        return contents;
    }
    Common::unordered_set<u64> seen;
    size_t valid = HeaderSize;
    while (data.size() - valid >= ChunkHeaderSize) {
        const u32 magic = Get<u32>(data, valid);
        const u32 count = Get<u32>(data, valid + 4);
        const u64 hash = Get<u64>(data, valid + 8);
        const size_t bytes = size_t{count} * sizeof(u64);
        if (magic != ChunkMagic || count == 0 || count > MaxChunkEntries ||
            data.size() - valid - ChunkHeaderSize < bytes) {
            break;
        }
        const size_t payload = valid + ChunkHeaderSize;
        if (Common::CityHash64(reinterpret_cast<const char*>(data.data() + payload), bytes) !=
            hash) {
            break;
        }
        bool chunk_ok = true;
        for (u32 i = 0; i < count && chunk_ok; ++i) {
            const u64 entry = Get<u64>(data, payload + i * sizeof(u64));
            const u64 offset = entry & PcMask;
            chunk_ok = offset < image.size && offset % 4 == 0 &&
                       (entry & ~PcMask & ~FpcrBits) == 0;
        }
        if (!chunk_ok) {
            break;
        }
        for (u32 i = 0; i < count && contents.entries.size() < max_entries; ++i) {
            const u64 entry = Get<u64>(data, payload + i * sizeof(u64));
            if (seen.insert(entry).second) {
                contents.entries.push_back(entry);
            }
        }
        valid = payload + bytes;
    }
    contents.valid_bytes = valid;
    return contents;
}

std::shared_ptr<JitWarmup> JitWarmup::Create(const Image& image) {
    if (image.size == 0 || image.base == 0) {
        return nullptr;
    }
    return std::make_shared<JitWarmup>(image);
}

JitWarmup::JitWarmup(const Image& image_) : image{image_} {
    std::string build_id;
    for (size_t i = 0; i < 16; ++i) {
        build_id += fmt::format("{:02X}", image.build_id[i]);
    }
    path = Common::FS::GetEdenPath(Common::FS::EdenPath::CacheDir) / "jit" /
           fmt::format("{:016X}-{}.blocks", image.program_id, build_id);
    Load();
    list_size.store(list.size(), std::memory_order_release);

    LOG_INFO(Core_ARM, "JIT warm-up: {} blocks listed in {} (image {:#x}+{:#x})", loaded,
             Common::FS::PathToUTF8String(path), image.base, image.size);
    writer = std::thread([this] { WriterLoop(); });
}

JitWarmup::~JitWarmup() {
    if (writer.joinable()) {
        {
            std::scoped_lock lock{mutex};
            stopping = true;
        }
        pending_cv.notify_all();
        writer.join();
    }
    MaybeLog(true);
}

bool JitWarmup::InImage(u64 location) const {
    const u64 pc = location & PcMask;
    return (location & SingleStepBit) == 0 && pc >= image.base && pc - image.base < image.size;
}

u64 JitWarmup::ToEntry(u64 location) const {
    return (location & ~PcMask) | ((location & PcMask) - image.base);
}

u64 JitWarmup::ToLocation(u64 entry) const {
    return (entry & ~PcMask) | ((image.base + (entry & PcMask)) & PcMask);
}

void JitWarmup::Load() {
    std::vector<u8> data;
    {
        Common::FS::IOFile file{path, Common::FS::FileAccessMode::Read,
                                Common::FS::FileType::BinaryFile};
        if (file.IsOpen()) {
            data.resize(static_cast<size_t>(std::min(file.GetSize(), File::MaxBytes)));
            data.resize(file.ReadSpan(std::span<u8>{data}));
        }
    }
    const File::Contents contents = File::Parse(data, image);
    // Room to grow: OnCompiled inserts under the lock, on the thread that runs the guest.
    known.reserve(contents.entries.size() * 2 + File::MaxChunkEntries);
    for (const u64 entry : contents.entries) {
        known.insert(entry);
        list.push_back(ToLocation(entry));
    }
    loaded = list.size();
    if (!contents.header_ok && !data.empty()) {
        LOG_INFO(Core_ARM, "JIT warm-up: {} is from another build or version, starting over",
                 Common::FS::PathToUTF8String(path));
    } else if (contents.header_ok && contents.valid_bytes < data.size()) {
        LOG_WARNING(Core_ARM, "JIT warm-up: {} bytes after the last whole chunk of {} dropped",
                    data.size() - contents.valid_bytes, Common::FS::PathToUTF8String(path));
    }
    // Rewrite a foreign or empty file; cut a torn or corrupt tail so appends follow whole chunks.
    if (!contents.header_ok) {
        const std::vector<u8> header = File::MakeHeader(image);
        if (!Common::FS::CreateParentDirs(path)) {
            file_ok = false;
        } else {
            Common::FS::IOFile file{path, Common::FS::FileAccessMode::Write,
                                    Common::FS::FileType::BinaryFile};
            file_ok =
                file.IsOpen() && file.WriteSpan(std::span<const u8>{header}) == header.size();
        }
    } else if (contents.valid_bytes < data.size()) {
        Common::FS::IOFile file{path, Common::FS::FileAccessMode::ReadWrite,
                                Common::FS::FileType::BinaryFile};
        file_ok = file.IsOpen() && file.SetSize(contents.valid_bytes);
    }
    if (!file_ok) {
        LOG_WARNING(Core_ARM, "JIT warm-up: cannot write {}, not recording",
                    Common::FS::PathToUTF8String(path));
    }
}

void JitWarmup::WriterLoop() {
    Common::SetCurrentThreadName("JitWarmupWriter");
    std::vector<u64> batch;
    std::vector<u8> bytes;
    while (true) {
        bool last;
        {
            std::unique_lock lock{mutex};
            pending_cv.wait_for(lock, std::chrono::seconds{2},
                                [this] { return stopping || pending.size() >= 4096; });
            last = stopping;
            batch.swap(pending);
            if (!file_ok) {
                batch.clear();
            }
        }
        if (!batch.empty()) {
            Common::FS::IOFile file{path, Common::FS::FileAccessMode::Append,
                                    Common::FS::FileType::BinaryFile};
            bool ok = file.IsOpen();
            for (size_t first = 0; ok && first < batch.size(); first += File::MaxChunkEntries) {
                const size_t count = std::min(File::MaxChunkEntries, batch.size() - first);
                bytes.clear();
                File::AppendChunks(bytes, std::span<const u64>{batch.data() + first, count});
                // One write per chunk: a crash leaves at most one torn chunk, which Load drops.
                ok = file.WriteSpan(std::span<const u8>{bytes}) == bytes.size();
            }
            if (!ok) {
                LOG_WARNING(Core_ARM, "JIT warm-up: writing {} failed, not recording",
                            Common::FS::PathToUTF8String(path));
                std::scoped_lock lock{mutex};
                file_ok = false;
            }
            batch.clear();
        }
        if (last) {
            return;
        }
    }
}

void JitWarmup::OnCompiled(size_t core, u64 location) {
    CoreWork& work = cores[core];
    if (!InImage(location)) {
        work.outside.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    const u64 entry = ToEntry(location);
    bool listed;
    {
        std::scoped_lock lock{mutex};
        listed = known.count(entry) != 0;
        if (!listed && known.size() < MaxEntries) {
            known.insert(entry);
            pending.push_back(entry);
            list.push_back(location);
            list_size.store(list.size(), std::memory_order_release);
        }
    }
    if (listed) {
        work.late.fetch_add(1, std::memory_order_relaxed);
    } else {
        work.unlisted.fetch_add(1, std::memory_order_relaxed);
    }
}

bool JitWarmup::Run(size_t core, Dynarmic::A64::Jit& jit, const std::atomic<bool>& stop) {
    CoreWork& work = cores[core];
    if (work.full || work.cursor >= list_size.load(std::memory_order_acquire)) {
        return false;
    }
    const u64 end_ns = NowNs() + SliceNs;
    std::array<u64, 64> batch;
    bool more = true;
    // stop and the deadline are checked before every block: once stop is set, this returns
    // after at most the Precompile in progress.
    while (more && !stop.load(std::memory_order_acquire) && NowNs() < end_ns) {
        size_t count;
        {
            // Never waits for the lock: OnCompiled may hold it while the set grows.
            std::unique_lock lock{mutex, std::try_to_lock};
            if (!lock.owns_lock()) {
                break;
            }
            count = std::min(batch.size(), list.size() - std::min(work.cursor, list.size()));
            std::copy_n(list.begin() + static_cast<std::ptrdiff_t>(work.cursor), count,
                        batch.begin());
        }
        if (count == 0) {
            more = false;
            break;
        }
        for (size_t i = 0; i < count; ++i) {
            if (stop.load(std::memory_order_acquire) || NowNs() >= end_ns) {
                break;
            }
            const u64 compile_start = NowNs();
            const auto result = jit.Precompile(batch[i], KeepFreeBytes);
            if (result == Dynarmic::A64::PrecompileResult::Full) {
                work.full = true;
                more = false;
                LOG_WARNING(Core_ARM,
                            "JIT warm-up: core {} stops, less than {} MiB of code cache left",
                            core, KeepFreeBytes >> 20);
                break;
            }
            ++work.cursor;
            const u64 compile_ns = NowNs() - compile_start;
            switch (result) {
            case Dynarmic::A64::PrecompileResult::Compiled:
                work.compiled.fetch_add(1, std::memory_order_relaxed);
                work.compile_ns.fetch_add(compile_ns, std::memory_order_relaxed);
                if (compile_ns > work.max_compile_ns.load(std::memory_order_relaxed)) {
                    work.max_compile_ns.store(compile_ns, std::memory_order_relaxed);
                }
                break;
            case Dynarmic::A64::PrecompileResult::Present:
                work.present.fetch_add(1, std::memory_order_relaxed);
                break;
            default:
                work.unreadable.fetch_add(1, std::memory_order_relaxed);
                break;
            }
        }
    }
    work.cache_bytes.store(jit.CodeCacheUsed(), std::memory_order_relaxed);
    const bool done = work.full || work.cursor >= list_size.load(std::memory_order_acquire);
    if (done && (work.full || work.cursor >= loaded) && !work.reported_done) {
        work.reported_done = true;
        if (done_cores.fetch_add(1, std::memory_order_relaxed) + 1 == cores.size()) {
            LOG_INFO(Core_ARM, "JIT warm-up: every core has compiled the listed blocks");
            MaybeLog(true);
            return false;
        }
    }
    MaybeLog(false);
    return !done;
}

void JitWarmup::MaybeLog(bool force) {
    const u64 now = NowNs();
    u64 last = last_log_ns.load(std::memory_order_relaxed);
    if (!force && (now - last < 5'000'000'000ULL ||
                   !last_log_ns.compare_exchange_strong(last, now, std::memory_order_relaxed))) {
        return;
    }
    last_log_ns.store(now, std::memory_order_relaxed);
    std::string text;
    u64 late = 0;
    u64 unlisted = 0;
    u64 outside = 0;
    for (size_t core = 0; core < cores.size(); ++core) {
        const CoreWork& work = cores[core];
        late += work.late.load(std::memory_order_relaxed);
        unlisted += work.unlisted.load(std::memory_order_relaxed);
        outside += work.outside.load(std::memory_order_relaxed);
        text += fmt::format(
            "{}{}:{}/{:.0f}ms/max:{:.2f}ms/present:{}/skipped:{}/cache:{}MiB",
            text.empty() ? "" : ",", core, work.compiled.load(std::memory_order_relaxed),
            static_cast<double>(work.compile_ns.load(std::memory_order_relaxed)) / 1e6,
            static_cast<double>(work.max_compile_ns.load(std::memory_order_relaxed)) / 1e6,
            work.present.load(std::memory_order_relaxed),
            work.unreadable.load(std::memory_order_relaxed),
            work.cache_bytes.load(std::memory_order_relaxed) >> 20);
    }
    // No lock: Run logs from here, and must not wait for OnCompiled.
    LOG_INFO(Core_ARM,
             "JIT warm-up: precompiled {} (core:blocks/time/max/present/skipped/cache) listed={} "
             "loaded={} run-time compiles: listed_late={} new={} outside_image={}",
             text, list_size.load(std::memory_order_relaxed), loaded, late, unlisted, outside);
}

} // namespace Core
