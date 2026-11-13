// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// nx2 shaderwarm (docs/shader-precompile.md): the environment a program from a game file is
// translated with by the precompile (EDEN_NX2_PRECOMPILE).

#include <array>
#include <optional>
#include <span>
#include <vector>

#include "common/common_types.h"
#include "video_core/shader_environment.h"

namespace VideoCommon {

/// A native graphics program from a game file (no GPU state): the 0x80-byte header + code +
/// constant section bytes. Bank 1 (the program's constants) is read from the blob; every other
/// query gets the answer that holds for nearly all programs (2D float textures, mapped storage
/// pointers, viewport transform on) and is recorded like a live environment's (Serialize).
/// Lowering failures throw from the constructor.
class PrecompileEnvironment;

/// A compute program's launch (QMD): work group, shared and local memory size.
struct PrecompileComputeLaunch {
    std::array<u32, 3> workgroup{1, 1, 1};
    u32 shared_memory{};
    u32 local_memory{};
};

class PrecompileEnvironment final : public GenericEnvironment {
public:
    /// const_offset ~0U: unknown, taken as the 256-byte boundary after the captured code.
    /// Compute programs have no 0x80-byte header (code at blob + 0); their launch parameters
    /// (QMD: work group, shared memory, local memory) come from the NVN control section when the
    /// pack has it. The translation doesn't depend on the work group and shared memory size (they
    /// are only copied into the program), and on the local memory size only when the program uses
    /// local memory (Shader::Info::uses_local_memory).
    using ComputeLaunch = PrecompileComputeLaunch;
    explicit PrecompileEnvironment(std::span<const u8> blob, u32 const_offset, u32 const_size,
                                   Shader::Stage stage, u32 default_texture_bound,
                                   const ComputeLaunch& launch = {});
    ~PrecompileEnvironment() override = default;

    u32 ReadCbufValue(u32 cbuf_index, u32 cbuf_offset) override;
    Shader::TextureType ReadTextureType(u32 handle) override;
    Shader::TexturePixelFormat ReadTexturePixelFormat(u32 handle) override;
    bool IsTexturePixelFormatInteger(u32 handle) override;
    u32 ReadViewportTransformState() override;
    std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(u32 bank, u32 offset) override;
    bool IsCbufStorageBufferPointer(u32 index, u32 offset) override;
    // Overrides the base's on Lichii (pointer tables); plain Eglantine has none (unused there).
    bool IsCbufIndirectStorageBufferPointer(u32 index, u32 offset, u32 indirect);
    u32 TextureBoundBuffer() const override;

    void NoteTextureHandleSource(u32 index, u32 offset, u32 shift_left, u32 secondary_index,
                                 u32 secondary_offset, u32 secondary_shift_left) override;

    /// A texture query the translation made: the handle's constant buffer words, which query
    /// (0 type, 1 integer format) and the answer given. The runtime re-asks them with live words.
    struct TextureQuery {
        std::array<u32, 6> source; ///< index, offset, shift, secondary index, offset, shift
        u32 kind;
        u32 answer;
    };
    [[nodiscard]] const std::vector<TextureQuery>& TextureQueries() const noexcept {
        return texture_queries;
    }

    /// The answer to "is this constant buffer word a mapped storage buffer address" (direct and
    /// pointer-table, keys as StoragePointerAnswers): true except for the keys given here, which
    /// make a variant for pointers that are still null when the game first uses the program (the
    /// runtime rebuilds when they become mapped, so games use several combinations).
    void SetUnmappedPointers(std::vector<u64> keys) { unmapped_pointers = std::move(keys); }
    [[nodiscard]] bool TextureBoundUsed() const noexcept { return texture_bound_used; }
    [[nodiscard]] bool ViewportUsed() const noexcept { return viewport_used; }
    [[nodiscard]] u32 ConstOffset() const noexcept { return const_offset; }
    /// The constant offset the 256-byte rule gives (to check it against the control section's).
    [[nodiscard]] u32 DerivedConstOffset() const noexcept { return derived_const_offset; }

private:
    std::vector<u8> blob;
    u32 const_offset{};
    u32 const_size{};
    u32 derived_const_offset{};
    mutable bool texture_bound_used{};
    std::vector<u64> unmapped_pointers;
    bool viewport_used{};
    std::array<u32, 6> last_handle_source{};
    bool has_handle_source{};
    std::vector<TextureQuery> texture_queries;
};

} // namespace VideoCommon
