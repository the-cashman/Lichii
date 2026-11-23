// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "core/hle/service/nx2_video_frame_queue.h"

namespace {
using Service::Nx2VideoFrameQueue;

// Only synthetic frame metadata is needed for scheduling. No compressed or game assets.
Nx2VideoFrameQueue::Frame Picture(u64 pts, bool end = false) {
    return {Tegra::Host1x::Nx2Picture{64, 48, pts, {}}, pts, end};
}
} // namespace

TEST_CASE("NX2 video only replaces a picture with a decoded successor that is due",
          "[nx2_video_frames]") {
    Nx2VideoFrameQueue queue;
    const auto generation = queue.Generation();
    REQUIRE(queue.Push(Picture(0), generation));
    // A slow decoder must not prevent the output worker from using this sole picture.
    REQUIRE(queue.DropOvertaken(50'000, 30) == 0);
    REQUIRE(queue.Front().pts == 0);
    REQUIRE(queue.Push(Picture(16'667), generation));
    REQUIRE(queue.DropOvertaken(16'666, 30) == 0);
    REQUIRE(queue.DropOvertaken(16'667, 30) == 1);
    REQUIRE(queue.Pop().pts == 16'667);
    REQUIRE(queue.Empty());
}

TEST_CASE("NX2 video preserves pictures around EOS and timestamp discontinuities",
          "[nx2_video_frames]") {
    Nx2VideoFrameQueue queue;
    const auto generation = queue.Generation();
    REQUIRE(queue.Push(Picture(33'334), generation));
    SECTION("An empty EOS is not a newer decoded picture") {
        REQUIRE(queue.Push({std::nullopt, 50'001, true}, generation));
    }
    SECTION("A repeated timestamp is not newer") {
        REQUIRE(queue.Push(Picture(33'334), generation));
    }
    SECTION("A backward timestamp must reach the client's clock reset") {
        REQUIRE(queue.Push(Picture(0), generation));
    }
    SECTION("A real final picture can replace an overdue picture") {
        REQUIRE(queue.Push(Picture(50'001, true), generation));
        REQUIRE(queue.DropOvertaken(50'001, 30) == 1);
        REQUIRE(queue.Front().end);
        REQUIRE(queue.Push(Picture(66'668), generation));
    }
    REQUIRE(queue.DropOvertaken(1'000'000, 30) == 0);
    REQUIRE(queue.Size() == 2);
}

TEST_CASE("NX2 video bounds decoded lookahead and rejects work from a discarded stream",
          "[nx2_video_frames]") {
    Nx2VideoFrameQueue queue;
    const auto old_generation = queue.Generation();
    for (size_t frame = 0; frame < queue.Capacity; ++frame) {
        REQUIRE(queue.Push(Picture(frame * 16'667), old_generation));
    }
    REQUIRE_FALSE(queue.HasRoom());
    REQUIRE_FALSE(queue.Push(Picture(200'000), old_generation));
    REQUIRE(queue.Size() == queue.Capacity);
    auto reserved_output = queue.Pop();
    REQUIRE(queue.HasRoom());
    REQUIRE(queue.Push(Picture(200'000), old_generation));

    // Flush, stop and reconfigure invalidate queued and in-flight decode work. An image already
    // reserved for output still owns its picture and can complete the client's promised fence.
    queue.Reset();
    REQUIRE(queue.Empty());
    REQUIRE(reserved_output.picture.has_value());
    REQUIRE(reserved_output.pts == 0);
    REQUIRE_FALSE(queue.Push(Picture(216'667, true), old_generation));
    REQUIRE(queue.Push(Picture(0), queue.Generation()));
    REQUIRE_FALSE(queue.Front().end);
    REQUIRE(queue.Size() == 1);
}

TEST_CASE("NX2 video bounded catch-up leaves a real picture and its final EOS",
          "[nx2_video_frames]") {
    Nx2VideoFrameQueue queue;
    for (u64 frame = 0; frame < queue.Capacity; ++frame) {
        REQUIRE(
            queue.Push(Picture(frame * 16'667, frame + 1 == queue.Capacity), queue.Generation()));
    }
    REQUIRE(queue.DropOvertaken(1'000'000, 3) == 3);
    REQUIRE(queue.Front().pts == 50'001);
    REQUIRE(queue.DropOvertaken(1'000'000, 30) == 4);
    REQUIRE(queue.Size() == 1);
    REQUIRE(queue.Front().end);
    REQUIRE(queue.Pop().pts == 116'669);
}
