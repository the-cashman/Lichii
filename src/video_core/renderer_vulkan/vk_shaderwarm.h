// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// nx2 shaderwarm: precompiling Switch 2 shaders from the game files (docs/shader-precompile.md).
// Helpers shared by the hooks in vk_pipeline_cache.cpp and vk_shaderwarm.cpp.

#include <span>

#include "common/common_types.h"
#include "shader_recompiler/frontend/ir/program.h"
#include "shader_recompiler/runtime_info.h"
#include "shader_recompiler/stage.h"

namespace Vulkan {
class Device;
struct GraphicsPipelineCacheKey;
} // namespace Vulkan

namespace Vulkan::Nx2 {

/// Tag of vulkan_ir.bin files: valid only for the build and device that wrote them.
[[nodiscard]] u64 IrCacheBuildTag(const Device& device);

/// EDEN_NX2_PIPELINE_LOG=1: one line per runtime graphics pipeline.
[[nodiscard]] bool RuntimePipelineLog();
/// EDEN_NX2_IR_CACHE_TASKMESH=1: task/mesh stages in the translation cache (off by default).
[[nodiscard]] bool TaskMeshIrCacheEnabled();
/// EDEN_NX2_PRECOMPILE_POINTER_VARIANTS=0: no storage-pointer variant records (on by default).
[[nodiscard]] bool PointerVariantsEnabled();
/// EDEN_NX2_PRECOMPILE_POINTER_MAX: pointers up to which every combination is made (default 6).
[[nodiscard]] size_t PointerVariantsMax();
/// EDEN_NX2_IR_CACHE_COMPUTE=0: no compute stages in the translation cache (on by default).
[[nodiscard]] bool ComputeIrCacheEnabled();
/// EDEN_NX2_CS_LOG=1: one line per compute pipeline built (precompile and run time) with its launch
/// inputs and a hash of its SPIR-V, to tell why a precompiled compute pipeline misses the driver
/// cache at run time.
void LogComputeSpirv(const char* where, u64 unique_hash, const Shader::IR::Program& program,
                     std::span<const u32> spirv);
/// Geometry records only with EDEN_NX2_RELATIVE_LEPC=1.
[[nodiscard]] bool GeometryIrCacheEnabled();
/// Translation cache key of a task or mesh stage (the program's own hash, tagged by stage).
[[nodiscard]] u64 TaskMeshRecordHash(u64 program_hash, Shader::Stage stage);

/// Why a pipeline isn't built from graphics pipeline libraries; nullptr: it is.
[[nodiscard]] const char* GplIneligible(const Device& device, const GraphicsPipelineCacheKey& key,
                                        std::span<const Shader::IR::Program> programs,
                                        bool layer_emulation, bool all_native);
/// A vertex program that loads its position input (invalid SPIR-V as a library stage).
[[nodiscard]] bool VertexReadsPosition(const Shader::Info& info);

/// The runtime info a pipeline's stage is emitted with (defined in vk_pipeline_cache.cpp).
[[nodiscard]] Shader::RuntimeInfo MakeRuntimeInfo(std::span<const Shader::IR::Program> programs,
                                                  const GraphicsPipelineCacheKey& key,
                                                  const Shader::IR::Program& program,
                                                  const Shader::IR::Program* previous_program,
                                                  const Device& device);

} // namespace Vulkan::Nx2
