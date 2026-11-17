// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "video_core/texture_cache/image_info.h"
#include "video_core/texture_cache/util.h"

namespace {

using VideoCommon::ImageInfo;
using VideoCommon::ImageType;
using VideoCore::Surface::PixelFormat;

ImageInfo Depth(PixelFormat format, u32 width, u32 height) {
    ImageInfo info;
    info.format = format;
    info.type = ImageType::e2D;
    info.size = {width, height, 1};
    info.block = {0, 4, 0};
    return info;
}

} // namespace

TEST_CASE("A depth target padded to whole GOBs is a padded square", "[video_core][texture_cache]") {
    // Ocarina of Time's local shadow maps: 4 bytes a texel, 16 texels a GOB.
    REQUIRE(VideoCommon::IsPaddedSquare(Depth(PixelFormat::D32_FLOAT, 800, 793)));
    REQUIRE(VideoCommon::IsPaddedSquare(Depth(PixelFormat::D32_FLOAT, 1024, 1017)));
    REQUIRE(VideoCommon::IsPaddedSquare(Depth(PixelFormat::D32_FLOAT, 144, 129)));
    // 2 bytes a texel: 32 texels a GOB
    REQUIRE(VideoCommon::IsPaddedSquare(Depth(PixelFormat::D16_UNORM, 800, 793)));
}

TEST_CASE("Other depth targets are not padded squares", "[video_core][texture_cache]") {
    REQUIRE_FALSE(VideoCommon::IsPaddedSquare(Depth(PixelFormat::D32_FLOAT, 793, 793)));
    REQUIRE_FALSE(VideoCommon::IsPaddedSquare(Depth(PixelFormat::D32_FLOAT, 793, 800)));
    // One more texel needs another GOB column.
    REQUIRE_FALSE(VideoCommon::IsPaddedSquare(Depth(PixelFormat::D32_FLOAT, 801, 793)));
    REQUIRE_FALSE(VideoCommon::IsPaddedSquare(Depth(PixelFormat::D32_FLOAT, 816, 793)));
    REQUIRE_FALSE(VideoCommon::IsPaddedSquare(Depth(PixelFormat::D32_FLOAT, 1920, 1080)));
    // 2 bytes a texel: 816 texels need a 26th GOB column.
    REQUIRE_FALSE(VideoCommon::IsPaddedSquare(Depth(PixelFormat::D16_UNORM, 816, 793)));

    ImageInfo multisampled = Depth(PixelFormat::D32_FLOAT, 800, 793);
    multisampled.num_samples = 4;
    REQUIRE_FALSE(VideoCommon::IsPaddedSquare(multisampled));

    ImageInfo volume = Depth(PixelFormat::D32_FLOAT, 800, 793);
    volume.type = ImageType::e3D;
    REQUIRE_FALSE(VideoCommon::IsPaddedSquare(volume));
}

TEST_CASE("A target padded to whole GOBs is as wide as its clip", "[video_core][texture_cache]") {
    // Donkey Kong Bananza's character targets, with a 2095x1179 surface clip.
    REQUIRE(VideoCommon::IsPaddedWidth(Depth(PixelFormat::R16_FLOAT, 2112, 1179), 2095));
    REQUIRE(VideoCommon::IsPaddedWidth(Depth(PixelFormat::D32_FLOAT, 2096, 1179), 2095));
    REQUIRE(VideoCommon::IsPaddedWidth(Depth(PixelFormat::B10G11R11_FLOAT, 368, 204), 360));
    // 2 bytes a texel: 2081 texels are the first of the last GOB column.
    REQUIRE(VideoCommon::IsPaddedWidth(Depth(PixelFormat::R16_FLOAT, 2112, 1179), 2081));
}

TEST_CASE("A clip that ends before the last GOB column is no padding",
          "[video_core][texture_cache]") {
    REQUIRE_FALSE(VideoCommon::IsPaddedWidth(Depth(PixelFormat::R16_FLOAT, 2112, 1179), 2080));
    REQUIRE_FALSE(VideoCommon::IsPaddedWidth(Depth(PixelFormat::D32_FLOAT, 2096, 1179), 1440));
    // As wide as the target, wider, or empty
    REQUIRE_FALSE(VideoCommon::IsPaddedWidth(Depth(PixelFormat::D32_FLOAT, 2096, 1179), 2096));
    REQUIRE_FALSE(VideoCommon::IsPaddedWidth(Depth(PixelFormat::D32_FLOAT, 2096, 1179), 2100));
    REQUIRE_FALSE(VideoCommon::IsPaddedWidth(Depth(PixelFormat::D32_FLOAT, 2096, 1179), 0));

    ImageInfo multisampled = Depth(PixelFormat::D32_FLOAT, 2096, 1179);
    multisampled.num_samples = 4;
    REQUIRE_FALSE(VideoCommon::IsPaddedWidth(multisampled, 2095));
}
