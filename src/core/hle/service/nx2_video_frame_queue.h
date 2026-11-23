// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <deque>
#include <optional>
#include <utility>

#include "common/common_types.h"
#include "video_core/host1x/nx2_video_decoder.h"

namespace Service {

// The session mutex protects this queue and its generation. A decode started before Reset
// may finish afterwards, but must not enqueue into the new stream.
class Nx2VideoFrameQueue {
public:
    struct Frame {
        std::optional<Tegra::Host1x::Nx2Picture> picture;
        u64 pts;
        bool end;
    };

    static constexpr size_t Capacity = 8;

    u64 Generation() const {
        return generation;
    }
    bool Empty() const {
        return frames.empty();
    }
    size_t Size() const {
        return frames.size();
    }
    const Frame& Front() const {
        return frames.front();
    }
    Frame& Back() {
        return frames.back();
    }
    bool HasRoom(size_t capacity = Capacity) const {
        return frames.size() < capacity;
    }

    void Reset() {
        frames.clear();
        ++generation;
    }

    bool Push(Frame frame, u64 expected_generation, size_t capacity = Capacity) {
        if (expected_generation != generation || !HasRoom(capacity)) {
            return false;
        }
        frames.push_back(std::move(frame));
        return true;
    }

    Frame Pop() {
        auto frame = std::move(frames.front());
        frames.pop_front();
        return frame;
    }

    // An undecoded packet or an empty EOS marker cannot replace a picture. Keep the last
    // decoded picture available even when decoding its successor is taking too long.
    u32 DropOvertaken(u64 due, u32 limit) {
        u32 dropped{};
        while (frames.size() >= 2 && dropped < limit) {
            const auto& front = frames[0];
            const auto& next = frames[1];
            if (front.end || !front.picture || !next.picture || next.pts <= front.pts ||
                next.pts > due) {
                break;
            }
            frames.pop_front();
            ++dropped;
        }
        return dropped;
    }

private:
    std::deque<Frame> frames;
    u64 generation{};
};

} // namespace Service
