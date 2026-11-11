// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "common/nx2_huge_pages.h"

#if defined(__linux__) && defined(__x86_64__)

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <string_view>
#include <thread>

#include <link.h>
#include <sys/mman.h>

#include "common/logging.h"

#ifndef MADV_COLLAPSE
#define MADV_COLLAPSE 25
#endif

namespace Common::Nx2HugePages {
namespace {

struct Range {
    std::uintptr_t start = 0;
    std::uintptr_t end = 0;
};

bool Wants(std::string_view value, std::string_view part) {
    if (value.empty() || value == "0") {
        return false;
    }
    if (value == "1" || value == "all") {
        return true;
    }
    while (!value.empty()) {
        const auto comma = value.find(',');
        if (value.substr(0, comma) == part) {
            return true;
        }
        if (comma == std::string_view::npos) {
            break;
        }
        value.remove_prefix(comma + 1);
    }
    return false;
}

bool ThpAvailable() {
    std::ifstream f("/sys/kernel/mm/transparent_hugepage/enabled");
    std::string line;
    return std::getline(f, line) && line.find("[never]") == std::string::npos;
}

std::size_t PmdSize() {
    std::ifstream f("/sys/kernel/mm/transparent_hugepage/hpage_pmd_size");
    std::size_t size = 0;
    if (!(f >> size) || size == 0 || (size & (size - 1)) != 0) {
        size = 2u << 20;
    }
    return size;
}

int FindText(dl_phdr_info* info, std::size_t, void* data) {
    // The first object is the main program.
    auto* range = static_cast<Range*>(data);
    for (int i = 0; i < info->dlpi_phnum; ++i) {
        const auto& ph = info->dlpi_phdr[i];
        if (ph.p_type == PT_LOAD && (ph.p_flags & PF_X) != 0) {
            range->start = info->dlpi_addr + ph.p_vaddr;
            range->end = range->start + ph.p_memsz;
            break;
        }
    }
    return 1;
}

std::size_t HugifyText(Range& moved) {
    Range text;
    dl_iterate_phdr(FindText, &text);
    const std::size_t huge = PmdSize();
    const std::uintptr_t a = (text.start + huge - 1) & ~(huge - 1);
    const std::uintptr_t b = text.end & ~(huge - 1);
    if (text.start == 0 || b <= a) {
        return 0;
    }
    const std::size_t len = b - a;
    void* raw = mmap(nullptr, len + huge, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (raw == MAP_FAILED) {
        return 0;
    }
    const auto raw_addr = reinterpret_cast<std::uintptr_t>(raw);
    const std::uintptr_t buf = (raw_addr + huge - 1) & ~(huge - 1);
    if (buf > raw_addr) {
        munmap(raw, buf - raw_addr);
    }
    if (const std::uintptr_t tail = raw_addr + len + huge - (buf + len); tail > 0) {
        munmap(reinterpret_cast<void*>(buf + len), tail);
    }
    void* const bufp = reinterpret_cast<void*>(buf);
    madvise(bufp, len, MADV_HUGEPAGE);
    std::memcpy(bufp, reinterpret_cast<const void*>(a), len);
    madvise(bufp, len, MADV_COLLAPSE); // fails harmlessly on kernels without it
    if (mprotect(bufp, len, PROT_READ | PROT_EXEC) != 0) {
        munmap(bufp, len);
        return 0;
    }
    // One syscall replaces the file mapping with identical bytes: no moment without code there,
    // so this function, the PLT and other threads may run from the moved range.
    if (mremap(bufp, len, len, MREMAP_MAYMOVE | MREMAP_FIXED, reinterpret_cast<void*>(a)) ==
        MAP_FAILED) {
        munmap(bufp, len);
        return 0;
    }
    moved = {a, b};
    return len;
}

struct Smaps {
    std::size_t text_huge = 0;  // AnonHugePages in the moved text range
    std::size_t exec_huge = 0;  // AnonHugePages of other executable anonymous mappings (JIT)
    std::size_t total_huge = 0; // AnonHugePages of the process
};

Smaps ReadSmaps(Range text) {
    Smaps s;
    std::ifstream f("/proc/self/smaps");
    std::string line;
    std::uintptr_t start = 0;
    bool exec = false;
    bool anon = false;
    while (std::getline(f, line)) {
        // Mapping header: "start-end perms offset dev inode [path]"; field lines: "Name: value kB".
        unsigned long long lo = 0;
        unsigned long long hi = 0;
        char perms[8] = {};
        unsigned long long inode = 0;
        int path_at = 0;
        if (std::sscanf(line.c_str(), "%llx-%llx %7s %*llx %*s %llu %n", &lo, &hi, perms, &inode,
                        &path_at) == 4) {
            start = lo;
            exec = perms[2] == 'x';
            anon = inode == 0 && line.find('/', path_at) == std::string::npos;
            continue;
        }
        if (line.rfind("AnonHugePages:", 0) == 0) {
            const std::size_t kb = std::strtoull(line.c_str() + 14, nullptr, 10);
            s.total_huge += kb << 10;
            if (start >= text.start && start < text.end) {
                s.text_huge += kb << 10;
            } else if (exec && anon) {
                s.exec_huge += kb << 10;
            }
        }
    }
    return s;
}

std::atomic_flag g_applied = ATOMIC_FLAG_INIT;

} // namespace

void Apply() {
    if (g_applied.test_and_set()) {
        return;
    }
    // Default on (text + jit); off with EDEN_NX2_HUGE_PAGES=0, and when unset while a profiler's
    // perf map is requested (PERF_BUILDID_DIR): perf/samply can't name the moved text.
    const char* env = std::getenv("EDEN_NX2_HUGE_PAGES");
    const std::string_view value =
        env ? env : (std::getenv("PERF_BUILDID_DIR") != nullptr ? "0" : "1");
    const bool text = Wants(value, "text");
    const bool jit = Wants(value, "jit");
    if ((!text && !jit) || !ThpAvailable()) {
        return;
    }
    Range moved;
    const std::size_t moved_bytes = text ? HugifyText(moved) : 0;
    const Smaps now = ReadSmaps(moved);
    std::fprintf(stderr, "Nx2HugePages: text %zu of %zu MiB moved, %zu MiB on huge pages\n",
                 now.text_huge >> 20, moved_bytes >> 20, now.text_huge >> 20);
    // The JIT caches fill later and the log starts after main: report from smaps once running.
    std::thread([moved, moved_bytes] {
        int elapsed_s = 0;
        for (const int delay_s : {20, 100}) {
            std::this_thread::sleep_for(std::chrono::seconds(delay_s));
            elapsed_s += delay_s;
            const Smaps s = ReadSmaps(moved);
            LOG_INFO(Common,
                     "Nx2HugePages (EDEN_NX2_HUGE_PAGES): text {} MiB on huge pages ({} MiB "
                     "moved), JIT/other executable anonymous {} MiB, process total {} MiB "
                     "(after {} s)",
                     s.text_huge >> 20, moved_bytes >> 20, s.exec_huge >> 20, s.total_huge >> 20,
                     elapsed_s);
        }
    }).detach();
}

} // namespace Common::Nx2HugePages

#else

namespace Common::Nx2HugePages {
void Apply() {}
} // namespace Common::Nx2HugePages

#endif
