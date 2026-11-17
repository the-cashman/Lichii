// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cstring>
#include <map>
#include <optional>
#include <vector>
#include <catch2/catch_test_macros.hpp>
#include "core/arm/dynarmic/jit_warmup.h"
#include "dynarmic/interface/A64/a64.h"

namespace {

using File = Core::JitWarmup::File;

// FPCR.RMode bit 22, where a location keeps it (37 + 22).
constexpr u64 FpcrRMode = u64{1} << 59;
constexpr u64 SingleStep = u64{1} << 57;

Core::JitWarmup::Image TestImage() {
    Core::JitWarmup::Image image{
        .program_id = 0x0100F2C0115B6000,
        .base = 0x80004000,
        .size = 0x100000,
    };
    for (size_t i = 0; i < image.build_id.size(); ++i) {
        image.build_id[i] = static_cast<u8>(i * 7 + 1);
    }
    return image;
}

std::vector<u8> FileWith(const Core::JitWarmup::Image& image,
                         const std::vector<std::vector<u64>>& chunks) {
    std::vector<u8> data = File::MakeHeader(image);
    for (const auto& chunk : chunks) {
        File::AppendChunks(data, chunk);
    }
    return data;
}

size_t ChunkBytes(size_t entries) {
    return File::ChunkHeaderSize + entries * sizeof(u64);
}

} // namespace

TEST_CASE("JIT warm-up file reads back its entries once each, in order", "[jit_warmup]") {
    const auto image = TestImage();
    const std::vector<u64> first{0, 4, 0x100};
    const std::vector<u64> second{4, 0x10 | FpcrRMode, 0xFFFFC};
    const auto data = FileWith(image, {first, second});
    REQUIRE(data.size() == File::HeaderSize + ChunkBytes(3) + ChunkBytes(3));

    const auto contents = File::Parse(data, image);
    REQUIRE(contents.header_ok);
    REQUIRE(contents.valid_bytes == data.size());
    REQUIRE(contents.entries == std::vector<u64>{0, 4, 0x100, 0x10 | FpcrRMode, 0xFFFFC});
}

TEST_CASE("JIT warm-up file from another title, build, image or version is not read",
          "[jit_warmup]") {
    const auto image = TestImage();
    const auto data = FileWith(image, {{0, 4}});

    auto other = image;
    other.build_id[31] ^= 1;
    REQUIRE_FALSE(File::Parse(data, other).header_ok);
    other = image;
    other.program_id += 1;
    REQUIRE_FALSE(File::Parse(data, other).header_ok);
    other = image;
    other.size += 0x1000;
    REQUIRE_FALSE(File::Parse(data, other).header_ok);
    // The image base moves every launch: it is not part of the header.
    other = image;
    other.base += 0x200000;
    REQUIRE(File::Parse(data, other).entries.size() == 2);

    auto old_version = data;
    old_version[8] ^= 1;
    for (const auto& bad : {old_version, std::vector<u8>{},
                            std::vector<u8>(data.begin(), data.begin() + File::HeaderSize - 1)}) {
        const auto contents = File::Parse(bad, image);
        REQUIRE_FALSE(contents.header_ok);
        REQUIRE(contents.valid_bytes == 0);
        REQUIRE(contents.entries.empty());
    }
}

TEST_CASE("JIT warm-up file cut anywhere keeps exactly the whole chunks before the cut",
          "[jit_warmup]") {
    const auto image = TestImage();
    const auto data = FileWith(image, {{0, 4}, {8, 12, 16}, {20}});
    const size_t end1 = File::HeaderSize + ChunkBytes(2);
    const size_t end2 = end1 + ChunkBytes(3);
    REQUIRE(data.size() == end2 + ChunkBytes(1));

    for (size_t size = File::HeaderSize; size <= data.size(); ++size) {
        const auto contents = File::Parse(std::span<const u8>{data.data(), size}, image);
        REQUIRE(contents.header_ok);
        if (size < end1) {
            REQUIRE(contents.valid_bytes == File::HeaderSize);
            REQUIRE(contents.entries.empty());
        } else if (size < end2) {
            REQUIRE(contents.valid_bytes == end1);
            REQUIRE(contents.entries == std::vector<u64>{0, 4});
        } else if (size < data.size()) {
            REQUIRE(contents.valid_bytes == end2);
            REQUIRE(contents.entries == std::vector<u64>{0, 4, 8, 12, 16});
        } else {
            REQUIRE(contents.valid_bytes == data.size());
            REQUIRE(contents.entries.size() == 6);
        }
    }
}

TEST_CASE("JIT warm-up file stops at a corrupt chunk and drops everything after it",
          "[jit_warmup]") {
    const auto image = TestImage();
    const auto data = FileWith(image, {{0, 4}, {8, 12, 16}, {20}});
    const size_t chunk2 = File::HeaderSize + ChunkBytes(2);

    const auto expect_first_chunk_only = [&](const std::vector<u8>& bad) {
        const auto contents = File::Parse(bad, image);
        REQUIRE(contents.header_ok);
        REQUIRE(contents.valid_bytes == chunk2);
        REQUIRE(contents.entries == std::vector<u64>{0, 4});
    };
    // Every byte of the second chunk: magic, count, checksum and payload.
    for (size_t offset = chunk2; offset < chunk2 + ChunkBytes(3); ++offset) {
        auto bad = data;
        bad[offset] ^= 0x40;
        expect_first_chunk_only(bad);
    }
    // A count of zero or over the chunk limit, even with a matching checksum.
    for (const u32 count : {0U, static_cast<u32>(File::MaxChunkEntries + 1)}) {
        auto bad = data;
        std::memcpy(bad.data() + chunk2 + 4, &count, sizeof(count));
        expect_first_chunk_only(bad);
    }
}

TEST_CASE("JIT warm-up file rejects a chunk with an entry that is not a block of the image",
          "[jit_warmup]") {
    const auto image = TestImage();
    for (const u64 entry : {image.size, image.size + 4, u64{2}, u64{4} | SingleStep,
                            u64{4} | (u64{1} << 58)}) {
        const auto data = FileWith(image, {{0}, {4, entry}, {8}});
        const auto contents = File::Parse(data, image);
        REQUIRE(contents.header_ok);
        REQUIRE(contents.valid_bytes == File::HeaderSize + ChunkBytes(1));
        REQUIRE(contents.entries == std::vector<u64>{0});
    }
}

TEST_CASE("JIT warm-up file keeps at most the entry cap", "[jit_warmup]") {
    const auto image = TestImage();
    const auto data = FileWith(image, {{0, 4, 4, 8}, {12, 16}});
    const auto contents = File::Parse(data, image, 3);
    REQUIRE(contents.entries == std::vector<u64>{0, 4, 8});
    // Whole chunks past the cap stay valid, so appends still follow a whole chunk.
    REQUIRE(contents.valid_bytes == data.size());
}

TEST_CASE("JIT warm-up file splits long appends into chunks the reader accepts",
          "[jit_warmup]") {
    const auto image = TestImage();
    std::vector<u64> entries(File::MaxChunkEntries + 5);
    for (size_t i = 0; i < entries.size(); ++i) {
        entries[i] = i * 4;
    }
    const auto data = FileWith(image, {entries});
    REQUIRE(data.size() == File::HeaderSize + ChunkBytes(File::MaxChunkEntries) + ChunkBytes(5));
    const auto contents = File::Parse(data, image);
    REQUIRE(contents.valid_bytes == data.size());
    REQUIRE(contents.entries == entries);
}

#ifdef ARCHITECTURE_x86_64

namespace {

constexpr u64 Entry = 0x10000000;
constexpr u32 Svc0 = 0xD4000001;

constexpr u32 MovX0(u32 value) {
    return 0xD2800000U | (value << 5);
}

class PrecompileEnvironment final : public Dynarmic::A64::UserCallbacks {
public:
    Dynarmic::A64::Jit* jit{};
    std::map<u64, u32> code;
    std::vector<u64> compiled; ///< BlockCompiled locations

    std::optional<u32> MemoryReadCode(u64 address) override {
        const auto it = code.find(address);
        return it == code.end() ? std::nullopt : std::optional<u32>{it->second};
    }
    u64 MemoryRead(u64, size_t) override {
        FAIL("Unexpected memory read");
        return 0;
    }
    Dynarmic::A64::Vector MemoryRead128(u64) override {
        FAIL("Unexpected memory read");
        return {};
    }
    void MemoryWrite(u64, u64, size_t) override {
        FAIL("Unexpected memory write");
    }
    void MemoryWrite128(u64, Dynarmic::A64::Vector) override {
        FAIL("Unexpected memory write");
    }
    void CallSVC(u32) override {
        jit->HaltExecution();
    }
    void ExceptionRaised(u64, Dynarmic::A64::Exception) override {
        FAIL("Unexpected exception");
    }
    void AddTicks(u64) override {}
    u64 GetTicksRemaining() override {
        return 1000;
    }
    u64 GetCNTPCT() override {
        return 0;
    }
    void BlockCompiled(u64 location) override {
        compiled.push_back(location);
    }
};

struct TestJit {
    PrecompileEnvironment env;
    std::optional<Dynarmic::A64::Jit> jit;

    TestJit() {
        env.code = {{Entry, MovX0(1)}, {Entry + 4, Svc0}};
        Dynarmic::A64::UserConfig config;
        config.callbacks = &env;
        config.enable_cycle_counting = false;
        jit.emplace(config);
        env.jit = &*jit;
        // FPCR 0: the location of Entry is Entry itself.
        jit->SetFpcr(0);
    }

    u64 RunEntry() {
        jit->SetPC(Entry);
        jit->SetRegister(0, 0);
        jit->Run();
        return jit->GetRegister(0);
    }
};

} // namespace

TEST_CASE("A precompiled block is the one execution runs", "[jit_warmup]") {
    TestJit t;
    REQUIRE(t.jit->Precompile(Entry, 0) == Dynarmic::A64::PrecompileResult::Compiled);
    REQUIRE(t.jit->Precompile(Entry, 0) == Dynarmic::A64::PrecompileResult::Present);
    // Changed without an invalidation: the precompiled block is what runs.
    t.env.code[Entry] = MovX0(2);
    REQUIRE(t.RunEntry() == 1);
    // BlockCompiled reports execution's compiles only.
    REQUIRE(std::find(t.env.compiled.begin(), t.env.compiled.end(), Entry) ==
            t.env.compiled.end());
}

TEST_CASE("Precompiled blocks are dropped by an invalidation requested before or after them",
          "[jit_warmup]") {
    SECTION("InvalidateCacheRange after the precompile") {
        TestJit t;
        REQUIRE(t.jit->Precompile(Entry, 0) == Dynarmic::A64::PrecompileResult::Compiled);
        t.env.code[Entry] = MovX0(2);
        t.jit->InvalidateCacheRange(Entry, 4);
        REQUIRE(t.RunEntry() == 2);
    }
    SECTION("ClearCache after the precompile") {
        TestJit t;
        REQUIRE(t.jit->Precompile(Entry, 0) == Dynarmic::A64::PrecompileResult::Compiled);
        t.env.code[Entry] = MovX0(2);
        t.jit->ClearCache();
        REQUIRE(t.RunEntry() == 2);
    }
    SECTION("InvalidateCacheRange still pending when the block is precompiled") {
        TestJit t;
        t.jit->InvalidateCacheRange(Entry, 4);
        REQUIRE(t.jit->Precompile(Entry, 0) == Dynarmic::A64::PrecompileResult::Compiled);
        t.env.code[Entry] = MovX0(2);
        REQUIRE(t.RunEntry() == 2);
    }
    SECTION("ClearCache still pending when the block is precompiled") {
        TestJit t;
        t.jit->ClearCache();
        REQUIRE(t.jit->Precompile(Entry, 0) == Dynarmic::A64::PrecompileResult::Compiled);
        t.env.code[Entry] = MovX0(2);
        REQUIRE(t.RunEntry() == 2);
    }
}

TEST_CASE("Precompile stops at keep_free and never clears the cache", "[jit_warmup]") {
    TestJit t;
    const size_t empty = t.jit->CodeCacheUsed();
    REQUIRE(t.jit->Precompile(Entry, size_t{1} << 40) == Dynarmic::A64::PrecompileResult::Full);
    REQUIRE(t.jit->CodeCacheUsed() == empty);

    REQUIRE(t.jit->Precompile(Entry, 0) == Dynarmic::A64::PrecompileResult::Compiled);
    const size_t used = t.jit->CodeCacheUsed();
    REQUIRE(used > empty);
    t.env.code[Entry + 0x100] = MovX0(3);
    t.env.code[Entry + 0x104] = Svc0;
    REQUIRE(t.jit->Precompile(Entry + 0x100, size_t{1} << 40) ==
            Dynarmic::A64::PrecompileResult::Full);
    REQUIRE(t.jit->CodeCacheUsed() == used);
    REQUIRE(t.jit->Precompile(Entry, 0) == Dynarmic::A64::PrecompileResult::Present);
}

TEST_CASE("Precompile emits nothing for unreadable or single-step locations", "[jit_warmup]") {
    TestJit t;
    const size_t empty = t.jit->CodeCacheUsed();
    // The second instruction is missing.
    t.env.code[Entry + 0x200] = MovX0(4);
    REQUIRE(t.jit->Precompile(Entry + 0x200, 0) == Dynarmic::A64::PrecompileResult::Unreadable);
    REQUIRE(t.jit->Precompile(Entry + 0x300, 0) == Dynarmic::A64::PrecompileResult::Unreadable);
    REQUIRE(t.jit->Precompile(Entry | SingleStep, 0) ==
            Dynarmic::A64::PrecompileResult::Unreadable);
    REQUIRE(t.jit->CodeCacheUsed() == empty);
}

TEST_CASE("Precompile leaves the guest state alone", "[jit_warmup]") {
    TestJit t;
    t.jit->SetPC(0x1234);
    t.jit->SetSP(0x20000000);
    for (size_t i = 0; i < 31; ++i) {
        t.jit->SetRegister(i, 0x5550000 + i);
    }
    t.jit->SetPstate(0x60000000);
    REQUIRE(t.jit->Precompile(Entry, 0) == Dynarmic::A64::PrecompileResult::Compiled);
    REQUIRE_FALSE(t.jit->IsExecuting());
    REQUIRE(t.jit->GetPC() == 0x1234);
    REQUIRE(t.jit->GetSP() == 0x20000000);
    for (size_t i = 0; i < 31; ++i) {
        REQUIRE(t.jit->GetRegister(i) == 0x5550000 + i);
    }
    REQUIRE(t.jit->GetPstate() == 0x60000000);
    REQUIRE(t.jit->GetFpcr() == 0);
}

#endif
