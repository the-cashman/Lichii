// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <filesystem>
#include <list>
#include <memory>
#include <fstream>
#include <mutex>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

#include "common/common_types.h"
#include "shader_recompiler/frontend/ir/program.h"
#include "shader_recompiler/object_pool.h"
#include "shader_recompiler/stage.h"

// Per-stage translation cache (Switch 2 native programs). TranslateProgram's result depends on the
// stage's own program and environment answers only (no pipeline key, no other stage), so it is
// cached per stage: <title>/vulkan_ir.bin holds the IR::Program after TranslateProgram, keyed by
// the stage's unique hash, with the environment answers the translation used. A pipeline whose
// stages are all in it skips CFG + TranslateProgram and only runs ConvertLegacyToGeneric +
// EmitSPIRV (which take the pipeline key) before the driver compile. Filled by the precompile pass
// (EDEN_NX2_PRECOMPILE, from game files alone); see docs/shader-precompile.md.
namespace VideoCommon::Nx2IrCache {

/// IR::Program <-> bytes. Deserialize throws Shader::LogicError on malformed data and checks every
/// instruction's use count against the serialized one.
[[nodiscard]] std::vector<u8> SerializeProgram(const Shader::IR::Program& program);
struct Record;
/// Decompresses a found record's program into record.program (no-op when already there); false
/// when it is damaged.

[[nodiscard]] Shader::IR::Program DeserializeProgram(std::span<const u8> data,
                                                     Shader::ObjectPool<Shader::IR::Inst>& inst_pool,
                                                     Shader::ObjectPool<Shader::IR::Block>& block_pool);

/// One cached stage translation.
struct Record {
    u64 unique_hash{};
    Shader::Stage stage{};
    /// Environment inputs that must match the live environment for the record to apply.
    bool texture_bound_used{};
    u32 texture_bound{};
    bool viewport_used{};
    u32 viewport_transform_state{1};
    /// Compute: the translation reads the local memory size only for programs that use local
    /// memory (then it must match); the work group and shared memory size are only copied into the
    /// program, so a compute record serves every launch and they are set from the live
    /// environment when it is used.
    bool local_memory_used{};
    u32 local_memory_size{};
    std::vector<std::pair<u64, u8>> storage_pointers; ///< (bank << 32 | offset) -> mapped
    /// Texture queries: the handle's cbuf words (index, offset, shift, secondary index, offset,
    /// shift), the query (0 texture type, 1 integer format) and the answer the translation got.
    struct TextureCheck {
        std::array<u32, 6> source;
        u32 kind;
        u32 answer;
    };
    std::vector<TextureCheck> texture_checks;
    std::vector<u8> answers;                          ///< GenericEnvironment::SerializeAnswers
    std::vector<u8> program;                          ///< SerializeProgram (uncompressed)
    /// As read by StageCache::Find: where the compressed program is in the file (LoadProgram
    /// reads and decompresses it into `program`). A lookup among many variant records reads only
    /// their checks and answers, and the program of the one whose answers match.
    u64 program_offset{};
    u32 compressed_size{};
    u32 raw_program_size{};
};

class StageCache {
public:
    /// Opens (or, with create, makes) the cache file. A file from another build is discarded
    /// when create is set and ignored otherwise. Returns false when there is no usable file.
    bool Open(const std::filesystem::path& path, u64 build_tag, bool create);

    [[nodiscard]] bool IsOpen() const noexcept {
        return is_open;
    }

    [[nodiscard]] bool Contains(u64 unique_hash) const;

    /// Every record for the stage (decompressed), empty when none.
    [[nodiscard]] std::vector<Record> Find(u64 unique_hash);
    /// Reads and decompresses a record's program (from Find) into record.program. False when the
    /// file can't be read or the data doesn't decompress to the recorded size.
    [[nodiscard]] bool LoadProgram(Record& record);

    /// Thread safe. Appends and indexes the record.
    void Append(const Record& record);

    [[nodiscard]] size_t NumRecords() const;
    [[nodiscard]] u64 FileSize() const;

private:
    mutable std::mutex mutex;
    std::filesystem::path path;
    std::fstream file;
    std::unordered_map<u64, std::vector<u64>> index; ///< unique hash -> record offsets
    /// Find's parsed records (checks and answers, no program) per unique hash, so a stage that
    /// several pipelines use reads its records from the file once per session. Append drops the
    /// hash's entry (a new record).
    std::unordered_map<u64, std::vector<Record>> parsed;
    /// Recently decompressed programs by file offset, least recently used last, up to
    /// ProgramCacheBytes: the stage's program for the next pipeline that uses it.
    using CachedProgram = std::pair<u64, std::shared_ptr<const std::vector<u8>>>;
    std::list<CachedProgram> program_lru;
    std::unordered_map<u64, std::list<CachedProgram>::iterator> program_lookup;
    size_t program_cache_bytes{};
    size_t num_records{};
    u64 end_offset{};
    bool is_open{};
};

} // namespace VideoCommon::Nx2IrCache
