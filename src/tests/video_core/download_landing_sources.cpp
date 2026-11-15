// SPDX-License-Identifier: GPL-3.0-or-later

// DownloadLanding with two publishers: the buffer cache's downloads and the texture cache's
// collector write-backs. A guest CPU write collected while either has a landing over its bytes
// stores them only once that landing has ended, so they land after the download's.

#include <atomic>
#include <chrono>
#include <optional>
#include <thread>

#include <catch2/catch_test_macros.hpp>

#include "video_core/rasterizer_download_area.h"

namespace {

using VideoCore::DownloadLanding;

/// A guest write of [addr, addr + size) on another thread: Await, then mark it stored.
class GuestWrite {
public:
    GuestWrite(DAddr addr, u64 size) {
        thread = std::thread([this, addr, size] {
            DownloadLanding::Await(addr, size);
            stored.store(true);
        });
    }
    ~GuestWrite() {
        thread.join();
    }

    /// Whether it stored within `wait`.
    bool StoredWithin(std::chrono::milliseconds wait) {
        const auto deadline = std::chrono::steady_clock::now() + wait;
        while (!stored.load() && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::yield();
        }
        return stored.load();
    }

private:
    std::atomic<bool> stored{};
    std::thread thread;
};

} // Anonymous namespace

TEST_CASE("A texture cache landing holds guest writes to its range", "[download_landing]") {
    const u64 waits = DownloadLanding::Waits();
    std::optional<GuestWrite> write;
    {
        const DownloadLanding landing{0x10000, 0x20000, DownloadLanding::Source::TextureCache};
        write.emplace(0x18000, 0x40);
        // Still held while the landing is published.
        REQUIRE_FALSE(write->StoredWithin(std::chrono::milliseconds{50}));
    }
    REQUIRE(write->StoredWithin(std::chrono::seconds{5}));
    REQUIRE(DownloadLanding::Waits() == waits + 1);
}

TEST_CASE("Guest writes outside a texture cache landing don't wait", "[download_landing]") {
    const u64 waits = DownloadLanding::Waits();
    const DownloadLanding landing{0x10000, 0x20000, DownloadLanding::Source::TextureCache};
    GuestWrite below{0xff00, 0x100};
    GuestWrite above{0x20000, 0x40};
    REQUIRE(below.StoredWithin(std::chrono::seconds{5}));
    REQUIRE(above.StoredWithin(std::chrono::seconds{5}));
    REQUIRE(DownloadLanding::Waits() == waits);
}

TEST_CASE("Buffer and texture cache landings are independent", "[download_landing]") {
    std::optional<GuestWrite> in_texture;
    std::optional<GuestWrite> in_buffer;
    {
        const DownloadLanding texture{0x10000, 0x20000, DownloadLanding::Source::TextureCache};
        {
            const DownloadLanding buffer{0x40000, 0x50000};
            in_texture.emplace(0x10000, 0x10);
            in_buffer.emplace(0x4fff0, 0x10);
            REQUIRE_FALSE(in_buffer->StoredWithin(std::chrono::milliseconds{50}));
        }
        // The buffer cache's landing ended; the texture cache's still holds its write.
        REQUIRE(in_buffer->StoredWithin(std::chrono::seconds{5}));
        REQUIRE_FALSE(in_texture->StoredWithin(std::chrono::milliseconds{50}));
    }
    REQUIRE(in_texture->StoredWithin(std::chrono::seconds{5}));
}
