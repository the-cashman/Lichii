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
        u64 submission{};
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

    // A drain without a new picture associates its EOS with the last queued picture.
    // It still belongs to the input that ended the stream, even if another input follows.
    void MarkEnd(u64 pts, u64 submission) {
        if (frames.empty()) {
            frames.push_back({std::nullopt, pts, true, submission});
        } else {
            frames.back().end = true;
            frames.back().submission = submission;
        }
    }

    // The parallel output path requires a real decoded successor; an empty EOS cannot replace
    // its last picture. The serial path retains its existing handling of an EOS-only successor.
    u32 DropOvertaken(u64 due, u32 limit, bool require_picture = true) {
        u32 dropped{};
        while (frames.size() >= 2 && dropped < limit) {
            const auto& front = frames[0];
            const auto& next = frames[1];
            if (front.end || !front.picture || (require_picture && !next.picture) ||
                next.pts <= front.pts || next.pts > due) {
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
