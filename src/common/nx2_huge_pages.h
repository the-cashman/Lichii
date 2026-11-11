// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

/// EDEN_NX2_HUGE_PAGES (default on): puts the emulator's own code, and the JIT code caches, on
/// 2 MiB pages (transparent huge pages). Values: 0 = off; unset, 1 or "all" = text + jit; or a
/// comma list of "text", "jit". Unset with PERF_BUILDID_DIR set (dynarmic's perf map, i.e. a
/// profiling run) = off.
///
/// - text: the executable's RX segment (its 2 MiB-aligned part) is copied into an anonymous,
///   MADV_HUGEPAGE buffer, made read+execute and moved over the original with one
///   mremap(MREMAP_FIXED). The file mapping can't get huge pages itself here: btrfs has no
///   file-THP for executables (no CONFIG_READ_ONLY_THP_FOR_FS) and the segment's address isn't
///   2 MiB-congruent with its file offset. Side effect: the text shows as an anonymous mapping,
///   so perf/samply can't symbolize it; set EDEN_NX2_HUGE_PAGES=0 when profiling without a perf map.
/// - jit: dynarmic's code caches get MADV_HUGEPAGE (read in dynarmic's BlockOfCode).
///
/// Linux only. A silent no-op when transparent huge pages are off ("[never]") or a step fails;
/// the text then stays as it was.
namespace Common::Nx2HugePages {

/// Call first in main (before threads start; the move itself is atomic). Idempotent.
void Apply();

} // namespace Common::Nx2HugePages
