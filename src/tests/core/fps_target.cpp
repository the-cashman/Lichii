// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cstring>
#include <optional>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "core/loader/fps_target.h"

namespace {
using Loader::FpsTarget::Target;

constexpr u64 TextOffset = 0x1000;
constexpr u64 TextSize = 0xF00;
constexpr u64 DataOffset = 0x4000;
constexpr u64 Global = 0x8000;
constexpr u32 Field = 0x1DC;

u32 Adrp(u32 reg, u64 pc, u64 target) {
    const s64 pages =
        (static_cast<s64>(target & ~u64{0xFFF}) - static_cast<s64>(pc & ~u64{0xFFF})) / 0x1000;
    return 0x90000000 | ((static_cast<u32>(pages) & 3) << 29) |
           (((static_cast<u32>(pages) >> 2) & 0x7FFFF) << 5) | reg;
}

u32 AddImm(u32 rd, u32 rn, u32 imm) {
    return 0x91000000 | ((imm & 0xFFF) << 10) | (rn << 5) | rd;
}

u32 LdrX(u32 rt, u32 rn, u32 offset) {
    return 0xF9400000 | (((offset / 8) & 0xFFF) << 10) | (rn << 5) | rt;
}

u32 LdrS(u32 rt, u32 rn, u32 offset) {
    return 0xBD400000 | (((offset / 4) & 0xFFF) << 10) | (rn << 5) | rt;
}

u32 Scvtf(u32 rd, u32 rn) {
    return 0x5E21D800 | (rn << 5) | rd;
}

u32 Fdiv(u32 rd, u32 rn, u32 rm) {
    return 0x1E201800 | (rm << 16) | (rn << 5) | rd;
}

u32 FmovSw(u32 rd, u32 rn) {
    return 0x1E270000 | (rn << 5) | rd;
}

u32 Movz(u32 rd, u32 imm, u32 shift) {
    return 0x52800000 | ((shift / 16) << 21) | ((imm & 0xFFFF) << 5) | rd;
}

u32 StrW(u32 rt, u32 rn, u32 offset) {
    return 0xB9000000 | (((offset / 4) & 0xFFF) << 10) | (rn << 5) | rt;
}

class Image {
public:
    Image() : bytes(0x10000, 0) {}

    void Word(u64 offset, u32 value) {
        std::memcpy(bytes.data() + offset, &value, sizeof(value));
    }

    void DoubleWord(u64 offset, u64 value) {
        std::memcpy(bytes.data() + offset, &value, sizeof(value));
    }

    u64 Put(u32 value) {
        const u64 at = pc;
        Word(at, value);
        pc += sizeof(u32);
        return at;
    }

    void Gap(u32 words = 2) {
        pc += words * sizeof(u32);
    }

    void Reader(u64 global = Global, u32 field = Field, u32 constant = 0x4270) {
        Put(Adrp(1, pc, global));
        Put(AddImm(1, 1, static_cast<u32>(global & 0xFFF)));
        Put(LdrX(0, 1, 0));
        Put(Movz(4, constant, 16));
        Put(FmovSw(1, 4));
        Put(LdrS(2, 0, field));
        Put(Scvtf(3, 2));
        Put(Fdiv(3, 1, 3));
    }

    void Store(u32 value, u64 global = Global, u32 field = Field) {
        Put(Adrp(1, pc, global));
        Put(AddImm(1, 1, static_cast<u32>(global & 0xFFF)));
        Put(LdrX(0, 1, 0));
        Put(Movz(5, value, 0));
        Put(StrW(5, 0, field));
    }

    void GotReader(u64 global, u64 got_page, u32 slot, u32 field = Field) {
        Put(Adrp(1, pc, got_page));
        Put(LdrX(1, 1, slot * 8));
        Put(LdrX(0, 1, 0));
        Put(Movz(4, 0x4270, 16));
        Put(FmovSw(1, 4));
        Put(LdrS(2, 0, field));
        Put(Scvtf(3, 2));
        Put(Fdiv(3, 1, 3));
        DoubleWord(got_page + slot * 8, global);
    }

    [[nodiscard]] std::optional<Target> Find() const {
        return Loader::FpsTarget::Find(bytes, TextOffset, TextSize, DataOffset);
    }

    std::vector<u8> bytes;
    u64 pc = TextOffset;
};
} // namespace

TEST_CASE("FPS target finds the target frame rate field", "[core][fps_target]") {
    Image image;
    for (int i = 0; i < 8; ++i) {
        image.Reader();
        image.Gap();
    }
    image.Store(30);

    const auto found = image.Find();
    REQUIRE(found);
    CHECK(found->global_offset == Global);
    CHECK(found->field_offset == Field);
    CHECK(found->readers == 8);
    CHECK(found->stores_of_thirty == 1);
}

TEST_CASE("FPS target needs eight readers", "[core][fps_target]") {
    Image image;
    for (int i = 0; i < 7; ++i) {
        image.Reader();
        image.Gap();
    }
    image.Store(30);
    CHECK_FALSE(image.Find());
}

TEST_CASE("FPS target needs a store of 30", "[core][fps_target]") {
    Image image;
    for (int i = 0; i < 8; ++i) {
        image.Reader();
        image.Gap();
    }
    image.Store(60);
    CHECK_FALSE(image.Find());
}

TEST_CASE("FPS target ignores a store to another field", "[core][fps_target]") {
    Image image;
    for (int i = 0; i < 8; ++i) {
        image.Reader();
        image.Gap();
    }
    image.Store(30, Global, Field + 4);
    CHECK_FALSE(image.Find());
}

TEST_CASE("FPS target ignores another constant", "[core][fps_target]") {
    Image image;
    for (int i = 0; i < 8; ++i) {
        image.Reader(Global, Field, 0x41F0); // 30.0f
        image.Gap();
    }
    image.Store(30);
    CHECK_FALSE(image.Find());
}

TEST_CASE("FPS target ignores a global outside the data segment", "[core][fps_target]") {
    Image image;
    for (int i = 0; i < 8; ++i) {
        image.Reader(0x2000);
        image.Gap();
    }
    image.Store(30, 0x2000);
    CHECK_FALSE(image.Find());
}

TEST_CASE("FPS target refuses two qualifying fields", "[core][fps_target]") {
    Image image;
    for (int i = 0; i < 8; ++i) {
        image.Reader(Global, Field);
        image.Gap();
    }
    image.Store(30, Global, Field);
    for (int i = 0; i < 8; ++i) {
        image.Reader(Global, Field + 4);
        image.Gap();
    }
    image.Store(30, Global, Field + 4);
    CHECK_FALSE(image.Find());
}

TEST_CASE("FPS target follows a global through its GOT slot", "[core][fps_target]") {
    Image image;
    for (int i = 0; i < 8; ++i) {
        image.GotReader(Global, 0x6000, 0);
        image.Gap();
    }
    image.Store(30);
    const auto found = image.Find();
    REQUIRE(found);
    CHECK(found->global_offset == Global);
    CHECK(found->field_offset == Field);
}

TEST_CASE("FPS target ignores a GOT slot that points outside the module", "[core][fps_target]") {
    Image image;
    for (int i = 0; i < 8; ++i) {
        image.GotReader(0x20000, 0x6000, 0);
        image.Gap();
    }
    image.Store(30);
    CHECK_FALSE(image.Find());
}
