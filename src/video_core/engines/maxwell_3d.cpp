// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include "common/assert.h"
#include "common/bit_util.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "core/core.h"
#include "core/core_timing.h"
#include "video_core/dirty_flags.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/gpu.h"
#include "video_core/gpu_split.h"
#include "video_core/macro_hle_oot.h"
#include "video_core/memory_manager.h"
#include "video_core/nx2.h"
#include "video_core/nx2_cmdpage_diag.h"
#include "video_core/nx2_fast_draw.h"
#include "video_core/nx2_list_dispatch.h"
#include "video_core/nx2_list_draw.h"
#include "video_core/nx2_list_macro_trace.h"
#include "video_core/nx2_perf.h"
#include "video_core/nx2_profile.h"
#include "video_core/nx2_query_fast.h"
#include "video_core/query_cache/frame_trace.h"
#include "video_core/query_dma.h"
#include "video_core/rasterizer_interface.h"
#include "video_core/textures/texture.h"
#ifdef ARCHITECTURE_x86_64
#include "video_core/macro_turing_jit.h"
#endif

namespace Tegra::Engines {

/// First register id that is actually a Macro call.
constexpr u32 MacroRegistersStart = 0xE00;

namespace {
// The live Maxwell3D engines (one per channel), for IsMaxwell3D.
std::array<std::atomic<const EngineInterface*>, 32> g_maxwell3d_engines{};

// EDEN_NX2_CB_DIRECT=0: constant buffer words go through MemoryManager::WriteWordsCached's page
// walk (CallConstBufferWords).
const bool g_cb_direct = VideoCommon::Nx2Switch("EDEN_NX2_CB_DIRECT", true);
} // Anonymous namespace

bool Maxwell3D::IsMaxwell3D(const EngineInterface* engine) {
    for (const auto& slot : g_maxwell3d_engines) {
        if (slot.load(std::memory_order_relaxed) == engine) {
            return engine != nullptr;
        }
    }
    return false;
}

Maxwell3D::Maxwell3D(MemoryManager& memory_manager_)
    : draw_manager()
    , memory_manager{memory_manager_}
#ifdef ARCHITECTURE_x86_64
    , macro_engine(bool(Settings::values.disable_macro_jit))
#else
    , macro_engine(true)
#endif
    , upload_state{memory_manager, regs.upload}
{
    dirty.flags.flip();
    InitializeRegisterDefaults();
    reports_fast_draw_breaks = true;
    execution_mask.reset();
    for (size_t i = 0; i < execution_mask.size(); i++)
        execution_mask[i] = IsMethodExecutable(u32(i));
    fast_maxwell3d = !record_write_history &&
                     !trace_first_method_writes && !VideoCommon::Nx2Profile::Enabled();
    for (auto& slot : g_maxwell3d_engines) {
        const EngineInterface* expected = nullptr;
        if (slot.compare_exchange_strong(expected, this)) {
            break;
        }
    }
}

Maxwell3D::~Maxwell3D() {
    // Its held-back front state (FrontFlushPending) goes with it.
    VideoCommon::GpuSplit::Get().DropPendingFlush(this);
    for (auto& slot : g_maxwell3d_engines) {
        const EngineInterface* expected = this;
        if (slot.compare_exchange_strong(expected, nullptr)) {
            break;
        }
    }
}

void Maxwell3D::BindRasterizer(VideoCore::RasterizerInterface* rasterizer_) {
    rasterizer = rasterizer_;
    upload_state.BindRasterizer(rasterizer_);
}

void Maxwell3D::InitializeRegisterDefaults() {
    // Initializes registers to their default values - what games expect them to be at boot. This is
    // for certain registers that may not be explicitly set by games.

    // Reset all registers to zero
    std::memset(&regs, 0, sizeof(regs));

    // Depth range near/far is not always set, but is expected to be the default 0.0f, 1.0f. This is
    // needed for ARMS.
    for (auto& viewport : regs.viewports) {
        viewport.depth_range_near = 0.0f;
        viewport.depth_range_far = 1.0f;
    }
    for (auto& viewport : regs.viewport_transform) {
        viewport.swizzle.x.Assign(Regs::ViewportSwizzle::PositiveX);
        viewport.swizzle.y.Assign(Regs::ViewportSwizzle::PositiveY);
        viewport.swizzle.z.Assign(Regs::ViewportSwizzle::PositiveZ);
        viewport.swizzle.w.Assign(Regs::ViewportSwizzle::PositiveW);
    }

    // Doom and Bomberman seems to use the uninitialized registers and just enable blend
    // so initialize blend registers with sane values
    regs.blend.color_op = Regs::Blend::Equation::Add_D3D;
    regs.blend.color_source = Regs::Blend::Factor::One_D3D;
    regs.blend.color_dest = Regs::Blend::Factor::Zero_D3D;
    regs.blend.alpha_op = Regs::Blend::Equation::Add_D3D;
    regs.blend.alpha_source = Regs::Blend::Factor::One_D3D;
    regs.blend.alpha_dest = Regs::Blend::Factor::Zero_D3D;
    for (auto& blend : regs.blend_per_target) {
        blend.color_op = Regs::Blend::Equation::Add_D3D;
        blend.color_source = Regs::Blend::Factor::One_D3D;
        blend.color_dest = Regs::Blend::Factor::Zero_D3D;
        blend.alpha_op = Regs::Blend::Equation::Add_D3D;
        blend.alpha_source = Regs::Blend::Factor::One_D3D;
        blend.alpha_dest = Regs::Blend::Factor::Zero_D3D;
    }
    regs.stencil_front_op.fail = Regs::StencilOp::Op::Keep_D3D;
    regs.stencil_front_op.zfail = Regs::StencilOp::Op::Keep_D3D;
    regs.stencil_front_op.zpass = Regs::StencilOp::Op::Keep_D3D;
    regs.stencil_front_op.func = Regs::ComparisonOp::Always_GL;
    regs.stencil_front_func_mask = 0xFFFFFFFF;
    regs.stencil_front_mask = 0xFFFFFFFF;
    regs.stencil_two_side_enable = 1;
    regs.stencil_back_op.fail = Regs::StencilOp::Op::Keep_D3D;
    regs.stencil_back_op.zfail = Regs::StencilOp::Op::Keep_D3D;
    regs.stencil_back_op.zpass = Regs::StencilOp::Op::Keep_D3D;
    regs.stencil_back_op.func = Regs::ComparisonOp::Always_GL;
    regs.stencil_back_func_mask = 0xFFFFFFFF;
    regs.stencil_back_mask = 0xFFFFFFFF;

    regs.depth_test_func = Regs::ComparisonOp::Always_GL;
    regs.gl_front_face = Regs::FrontFace::CounterClockWise;
    regs.gl_cull_face = Regs::CullFace::Back;

    // TODO(Rodrigo): Most games do not set a point size. I think this is a case of a
    // register carrying a default value. Assume it's OpenGL's default (1).
    regs.point_size = 1.0f;

    // TODO(bunnei): Some games do not initialize the color masks (e.g. Sonic Mania). Assuming a
    // default of enabled fixes rendering here.
    for (auto& color_mask : regs.color_mask) {
        color_mask.R.Assign(1);
        color_mask.G.Assign(1);
        color_mask.B.Assign(1);
        color_mask.A.Assign(1);
    }

    for (auto& format : regs.vertex_attrib_format) {
        format.constant.Assign(1);
    }

    // NVN games expect these values to be enabled at boot
    regs.rasterize_enable = 1;
    regs.color_target_mrt_enable = 1;
    regs.framebuffer_srgb = 1;
    regs.line_width_aliased = 1.0f;
    regs.line_width_smooth = 1.0f;
    regs.gl_front_face = Maxwell3D::Regs::FrontFace::ClockWise;
    regs.polygon_mode_back = Maxwell3D::Regs::PolygonMode::Fill;
    regs.polygon_mode_front = Maxwell3D::Regs::PolygonMode::Fill;

    shadow_state = regs;
}

bool Maxwell3D::IsMethodExecutable(u32 method) {
    if (method >= MacroRegistersStart) {
        return true;
    }
    switch (method) {
    case MAXWELL3D_REG_INDEX(draw.end):
    case MAXWELL3D_REG_INDEX(draw.begin):
    case MAXWELL3D_REG_INDEX(vertex_buffer.first):
    case MAXWELL3D_REG_INDEX(vertex_buffer.count):
    case MAXWELL3D_REG_INDEX(index_buffer.first):
    case MAXWELL3D_REG_INDEX(index_buffer.count):
    case MAXWELL3D_REG_INDEX(draw_inline_index):
    case MAXWELL3D_REG_INDEX(index_buffer32_subsequent):
    case MAXWELL3D_REG_INDEX(index_buffer16_subsequent):
    case MAXWELL3D_REG_INDEX(index_buffer8_subsequent):
    case MAXWELL3D_REG_INDEX(index_buffer32_first):
    case MAXWELL3D_REG_INDEX(index_buffer16_first):
    case MAXWELL3D_REG_INDEX(index_buffer8_first):
    case MAXWELL3D_REG_INDEX(inline_index_2x16.even):
    case MAXWELL3D_REG_INDEX(inline_index_4x8.index0):
    case MAXWELL3D_REG_INDEX(vertex_array_instance_first):
    case MAXWELL3D_REG_INDEX(vertex_array_instance_subsequent):
    case MAXWELL3D_REG_INDEX(draw_texture.src_y0):
    case RootTables::SelectMethod:
    case RootTables::LoadMethod:
    case MAXWELL3D_REG_INDEX(wait_for_idle):
    case MAXWELL3D_REG_INDEX(shadow_ram_control):
    case MAXWELL3D_REG_INDEX(load_mme.instruction_ptr):
    case MAXWELL3D_REG_INDEX(load_mme.instruction):
    case MAXWELL3D_REG_INDEX(load_mme.start_address):
    case MAXWELL3D_REG_INDEX(falcon[4]):
    case MAXWELL3D_REG_INDEX(const_buffer.buffer):
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 1:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 2:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 3:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 4:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 5:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 6:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 7:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 8:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 9:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 10:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 11:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 12:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 13:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 14:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 15:
    case MAXWELL3D_REG_INDEX(bind_groups[0].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[1].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[2].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[3].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[4].raw_config):
    case MAXWELL3D_REG_INDEX(topology_override):
    case MAXWELL3D_REG_INDEX(clear_surface):
    case MAXWELL3D_REG_INDEX(report_semaphore.query):
    case MAXWELL3D_REG_INDEX(render_enable.mode):
    case MAXWELL3D_REG_INDEX(clear_report_value):
    case MAXWELL3D_REG_INDEX(sync_info):
    case MAXWELL3D_REG_INDEX(launch_dma):
    case MAXWELL3D_REG_INDEX(inline_data):
    case MAXWELL3D_REG_INDEX(fragment_barrier):
    case MAXWELL3D_REG_INDEX(invalidate_texture_data_cache):
    case MAXWELL3D_REG_INDEX(tiled_cache_barrier):
        return true;
    default:
        return false;
    }
}

void Maxwell3D::ProcessMacro(Core::System& system, u32 method, const u32* base_start, u32 amount, bool is_last_call) {
    if (executing_macro == 0) {
        // A macro call must begin by writing the macro method's register, not its argument.
        ASSERT((method % 2) == 0 && "Can't start macro execution by writing to the ARGS register");
        executing_macro = method;
    }

    macro_segments.emplace_back(current_dma_segment, amount);
    current_macro_dirty |= current_dirty;
    current_dirty = false;

    // Nearly every call arrives whole in one method run (title 1 makes ~1,000,000 a second):
    // copy it to the stack instead of growing the parameter vector. Still a copy: the command
    // list is often guest memory read in place, which the guest may reuse while the macro runs
    // (and the macro memo stores the parameters after it ran). Only accurate GPU mode rewrites
    // the parameters in place (RefreshParameters), so it keeps the vector.
    // EDEN_NX2_MACRO_COPY=1: always use the vector (for comparisons).
    static const bool always_copy = std::getenv("EDEN_NX2_MACRO_COPY") != nullptr;
    if (is_last_call && macro_params.empty() && amount <= 64 && !always_copy &&
        !Settings::IsGPULevelHigh()) {
        std::array<u32, 64> parameters;
        std::copy_n(base_start, amount, parameters.begin());
        ConsumeSink(system);
        CallMacroMethod(system, executing_macro, std::span<const u32>{parameters.data(), amount});
        macro_segments.clear();
        current_macro_dirty = false;
        return;
    }
    macro_params.insert(macro_params.end(), base_start, base_start + amount);

    // Call the macro when there are no more parameters in the command buffer
    if (is_last_call) {
        ConsumeSink(system);
        CallMacroMethod(system, executing_macro, macro_params);
        macro_params.clear();
        macro_segments.clear();
        current_macro_dirty = false;
    }
}

void Maxwell3D::RefreshParametersImpl() {
    if (!Settings::IsGPULevelHigh()) {
        return;
    }
    size_t current_index = 0;
    for (auto& segment : macro_segments) {
        if (segment.first == 0) {
            current_index += segment.second;
            continue;
        }
        memory_manager.ReadBlock(segment.first, &macro_params[current_index],
                                 sizeof(u32) * segment.second);
        current_index += segment.second;
    }
}

u32 Maxwell3D::GetMaxCurrentVertices() {
    u32 num_vertices = 0;
    for (size_t index = 0; index < Regs::NumVertexArrays; ++index) {
        const auto& array = regs.vertex_streams[index];
        if (array.enable == 0) {
            continue;
        }
        const auto& attribute = regs.vertex_attrib_format[index];
        if (attribute.constant) {
            num_vertices = (std::max)(num_vertices, 1U);
            continue;
        }
        const auto& limit = regs.vertex_stream_limits[index];
        const GPUVAddr gpu_addr_begin = array.Address();
        const GPUVAddr gpu_addr_end = limit.Address() + 1;
        const u64 stream_size = GetNativeMacroEngine().native_turing
                                    ? regs.NativeVertexStreamSize(index)
                                    : gpu_addr_end - gpu_addr_begin;
        const u32 address_size = static_cast<u32>((std::min<u64>)(stream_size, (std::numeric_limits<u32>::max)()));
        num_vertices = (std::max)(
            num_vertices, address_size / (std::max)(attribute.SizeInBytes(), array.stride.Value()));
        break;
    }
    return num_vertices;
}

u32 Maxwell3D::GetMaxStreamVertices() {
    u64 num_vertices = 0;
    for (size_t index = 0; index < Regs::NumVertexArrays; ++index) {
        const auto& array = regs.vertex_streams[index];
        if (array.enable == 0 || regs.vertex_stream_instances.IsInstancingEnabled(index)) {
            continue;
        }
        const u32 stride = array.stride.Value();
        if (stride == 0) {
            // Every vertex fetches the same element: the stream bounds nothing.
            return 0;
        }
        const u64 stream_size = GetNativeMacroEngine().native_turing
                                    ? regs.NativeVertexStreamSize(index)
                                    : regs.vertex_stream_limits[index].Address() + 1 -
                                          array.Address();
        num_vertices = (std::max)(num_vertices, stream_size / stride);
    }
    return static_cast<u32>((std::min<u64>)(num_vertices, (std::numeric_limits<u32>::max)()));
}

size_t Maxwell3D::EstimateIndexBufferSize() {
    GPUVAddr start_address = regs.index_buffer.StartAddress();
    GPUVAddr end_address = regs.index_buffer.EndAddress();
    auto const byte_size = regs.index_buffer.FormatSizeInBytes();
    auto const max_size = 1ull << (byte_size * CHAR_BIT);
    auto const upper_cap = GetMaxCurrentVertices() * 4 * byte_size;
    auto const lower_cap = std::min<size_t>(size_t(end_address - start_address), upper_cap);
    return std::min<size_t>(memory_manager.GetMemoryLayoutSize(start_address, byte_size * max_size) / byte_size, lower_cap);
}

void Maxwell3D::ConsumeSinkImpl(Core::System& system) {
    if (front_macros && !VideoCommon::GpuSplit::OnBackThread()) {
        ReleaseFront();
    }
    const auto control = shadow_state.shadow_ram_control;
    // Sunk methods are state writes only (not in execution_mask): skip unchanged values, as
    // CallMethod does.
    const bool skip_same = SkipSameValueWrites();
    const auto process = [&](u32 method, u32 value) {
        if (!skip_same || regs.reg_array[method] != value) {
            ProcessDirtyRegisters(method, value);
        }
    };
    if (control == Regs::ShadowRamControl::Track || control == Regs::ShadowRamControl::TrackWithFilter) {
        for (auto [method, value] : method_sink) {
            shadow_state.reg_array[method] = value;
            process(method, value);
        }
    } else if (control == Regs::ShadowRamControl::Replay) {
        for (auto [method, value] : method_sink)
            process(method, shadow_state.reg_array[method]);
    } else {
        for (auto [method, value] : method_sink)
            process(method, value);
    }
    method_sink.clear();
}

namespace {

// The occlusion query methods a delta draw may follow. Ocarina of Time resets the samples counter,
// sets the render condition and reports around most of its draws, and each of these sent the next
// draw to a full resolve. None of them touches what a delta draw keeps (pipeline, image views,
// samplers, texture buffer bindings): they record query and conditional rendering commands, and
// the counter is started again by the draw itself on both paths. The ones that end the render pass
// (a condition resolved on the GPU) or submit still end the chain through the rasterizer's own
// checks.
bool IsFastDrawQueryMethod(u32 method) {
    constexpr u32 render_enable = MAXWELL3D_REG_INDEX(render_enable);
    constexpr u32 report = MAXWELL3D_REG_INDEX(report_semaphore);
    return method == MAXWELL3D_REG_INDEX(clear_report_value) ||
           method - render_enable < sizeof(Maxwell3D::Regs::RenderEnable) / sizeof(u32) ||
           method - report < sizeof(Maxwell3D::Regs::ReportSemaphore) / sizeof(u32);
}

} // Anonymous namespace

// Whether a write to `method` leaves what a fast draw keeps from the draw before (pipeline, image
// views, samplers, texture buffer bindings): the draw reads the register again on both paths, or
// nothing a draw resolves from reads it.
bool Maxwell3D::IsFastDrawParameter(u32 method) {
    switch (method) {
    case MAXWELL3D_REG_INDEX(const_buffer.size):
    case MAXWELL3D_REG_INDEX(const_buffer.address_high):
    case MAXWELL3D_REG_INDEX(const_buffer.address_low):
    case MAXWELL3D_REG_INDEX(const_buffer.offset):
    case MAXWELL3D_REG_INDEX(global_base_vertex_index):
    case MAXWELL3D_REG_INDEX(global_base_instance_index):
    // Ocarina of Time's draw macros write it before each draw (~27,000 a second in its village,
    // nearly every break); nothing in video_core reads it.
    case MAXWELL3D_REG_INDEX(vertex_id_base):
    case MAXWELL3D_REG_INDEX(index_buffer.start_addr_high):
    case MAXWELL3D_REG_INDEX(index_buffer.start_addr_low):
    case MAXWELL3D_REG_INDEX(index_buffer.limit_addr_high):
    case MAXWELL3D_REG_INDEX(index_buffer.limit_addr_low):
    case MAXWELL3D_REG_INDEX(index_buffer.format):
        return true;
    case 0x154: // MME memory address A and B: where a macro's memory reads go. Ocarina of
    case 0x155: // Time sets B before the macro that reads each occlusion report; no draw
                // reads them.
        return true;
    default:
        return DrawManager::IsInstanceLoopMethod(method) || IsFastDrawDeltaMethod(method) ||
               IsFastDrawVertexBuffer(method) || IsFastDrawQueryMethod(method) ||
               IsFastDrawVertexFormat(method);
    }
}

bool Maxwell3D::IsFastDrawVertexFormat(u32 method) {
    // With dynamic vertex input only each attribute's type class is in the pipeline key, which
    // the reusable draw compares (TakeReusableDraw); the rest is recorded for every draw.
    constexpr u32 formats = MAXWELL3D_REG_INDEX(vertex_attrib_format);
    return VideoCommon::Nx2FastDraw::g_dynamic_vertex_input.load(std::memory_order_relaxed) &&
           method >= formats && method < formats + Regs::NumVertexAttributes;
}

// Whether `method` is a vertex stream register the draw binds again on both paths: its address,
// size and limit, and its stride and enable word where strides are dynamic state.
bool Maxwell3D::IsFastDrawVertexBuffer(u32 method) {
    constexpr u32 streams = MAXWELL3D_REG_INDEX(vertex_streams);
    constexpr u32 stream_words = sizeof(Regs::VertexStream) / sizeof(u32);
    if (method >= streams && method < streams + Regs::NumVertexArrays * stream_words) {
        const u32 word = (method - streams) % stream_words;
        // Word 0, the stride and the enable bit, where the host binds strides dynamically: the
        // buffers are bound again with their strides on both paths (UpdateGraphicsBuffers), and
        // the pipeline key holds strides only without extended dynamic state.
        if (word == 0) {
            return VideoCommon::Nx2FastDraw::g_dynamic_strides.load(std::memory_order_relaxed);
        }
        return word == 1 || word == 2; // address_high, address_low
    }
    constexpr u32 sizes = 0x600 / sizeof(u32); // SET_VERTEX_STREAM_SIZE_A/B (NativeVertexStreamSize)
    constexpr u32 limits = MAXWELL3D_REG_INDEX(vertex_stream_limits);
    constexpr u32 limit_words = sizeof(Regs::VertexStreamLimit) / sizeof(u32);
    return (method >= sizes && method < sizes + Regs::NumVertexArrays * 2) ||
           (method >= limits && method < limits + Regs::NumVertexArrays * limit_words);
}

bool Maxwell3D::IsFastDrawDeltaMethod(u32 method) {
    constexpr u32 cb_data = MAXWELL3D_REG_INDEX(const_buffer.buffer);
    if (method >= cb_data && method < cb_data + Regs::NumCBData) {
        return true;
    }
    switch (method) {
    case MAXWELL3D_REG_INDEX(bind_groups[0].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[1].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[2].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[3].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[4].raw_config):
        return true;
    default:
        return false;
    }
}

// Runs what a write to `method` does besides storing the register (draws, clears, queries,
// uploads, macros), and ends the fast draw chain unless the method cannot change a draw.
void Maxwell3D::ProcessMethodCall(u32 method, u32 argument, u32 nonshadow_argument, bool is_last_call) {
    if (!DrawManager::IsInstanceLoopMethod(method) && !IsFastDrawDeltaMethod(method)) {
        // A fence-flagged release signals a fence: never neutral for the fast draw path.
        const bool query_method =
            IsFastDrawQueryMethod(method) &&
            !(method == MAXWELL3D_REG_INDEX(report_semaphore.query) &&
              regs.report_semaphore.query.operation == Regs::ReportSemaphore::Operation::Release &&
              regs.report_semaphore.query.short_query != 0);
        if (!query_method) {
            VideoCommon::Nx2FastDraw::Break(VideoCommon::Nx2FastDraw::Reason::Method, method);
        }
    }
    if (GetNativeMacroEngine().native_turing && method == MAXWELL3D_REG_INDEX(clear_surface)) {
        LOG_DEBUG(HW_GPU, "Native GPU surface clear: value={:#x}", argument);
    }
    switch (method) {
    case RootTables::SelectMethod:
        root_tables.selected = argument & 7;
        root_tables.offset = (argument >> 8) & 0xff;
        return;
    case RootTables::LoadMethod:
        if (root_tables.offset < 256 && root_tables.offset % 4 == 0) {
            root_tables.tables[root_tables.selected][root_tables.offset / 4] = argument;
            root_tables.loaded = true;
        }
        root_tables.offset += 4;
        return;
    case MAXWELL3D_REG_INDEX(wait_for_idle): {
        VideoCommon::Nx2Profile::Scope zone{VideoCommon::Nx2Profile::Zone::WaitForIdle};
        return rasterizer->WaitForIdle();
    }
    case MAXWELL3D_REG_INDEX(shadow_ram_control):
        shadow_state.shadow_ram_control = static_cast<Regs::ShadowRamControl>(nonshadow_argument);
        return;
    // With macros on the split GPU thread's front (FrontWrite), the front loads and binds them.
    case MAXWELL3D_REG_INDEX(load_mme.instruction_ptr):
        if (front_macros) return;
        return GetNativeMacroEngine().ClearCode(regs.load_mme.instruction_ptr);
    case MAXWELL3D_REG_INDEX(load_mme.instruction):
        if (front_macros) return;
        return GetNativeMacroEngine().AddCode(regs.load_mme.instruction_ptr, argument);
    case MAXWELL3D_REG_INDEX(load_mme.start_address):
        if (front_macros) {
            ++regs.load_mme.start_address_ptr;
            return;
        }
        return ProcessMacroBind(argument);
    case MAXWELL3D_REG_INDEX(falcon[4]):
        return ProcessFirmwareCall4();
    case MAXWELL3D_REG_INDEX(const_buffer.buffer):
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 1:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 2:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 3:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 4:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 5:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 6:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 7:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 8:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 9:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 10:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 11:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 12:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 13:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 14:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 15:
        return ProcessCBData(argument);
    case MAXWELL3D_REG_INDEX(bind_groups[0].raw_config):
        return ProcessCBBind(0);
    case MAXWELL3D_REG_INDEX(bind_groups[1].raw_config):
        return ProcessCBBind(1);
    case MAXWELL3D_REG_INDEX(bind_groups[2].raw_config):
        return ProcessCBBind(2);
    case MAXWELL3D_REG_INDEX(bind_groups[3].raw_config):
        return ProcessCBBind(3);
    case MAXWELL3D_REG_INDEX(bind_groups[4].raw_config):
        return ProcessCBBind(4);
    case MAXWELL3D_REG_INDEX(report_semaphore.query): {
        VideoCommon::Nx2Profile::Scope zone{VideoCommon::Nx2Profile::Zone::QueryReport};
        return ProcessQueryGet();
    }
    case MAXWELL3D_REG_INDEX(render_enable.mode): {
        VideoCommon::Nx2Profile::Scope zone{VideoCommon::Nx2Profile::Zone::RenderEnable};
        return ProcessQueryCondition();
    }
    case MAXWELL3D_REG_INDEX(clear_report_value): {
        VideoCommon::Nx2Profile::Scope zone{VideoCommon::Nx2Profile::Zone::CounterReset};
        return ProcessCounterReset();
    }
    case MAXWELL3D_REG_INDEX(sync_info): {
        VideoCommon::Nx2Profile::Scope zone{VideoCommon::Nx2Profile::Zone::SyncPoint};
        return ProcessSyncPoint();
    }
    case MAXWELL3D_REG_INDEX(launch_dma):
        return upload_state.ProcessExec(regs.launch_dma.memory_layout.Value() == Regs::LaunchDMA::Layout::Pitch);
    case MAXWELL3D_REG_INDEX(inline_data):
        upload_state.ProcessData(argument, is_last_call);
        return;
    case MAXWELL3D_REG_INDEX(fragment_barrier): {
        VideoCommon::Nx2Profile::Scope zone{VideoCommon::Nx2Profile::Zone::CacheBarrier};
        return rasterizer->FragmentBarrier();
    }
    case MAXWELL3D_REG_INDEX(invalidate_texture_data_cache): {
        VideoCommon::Nx2Profile::Scope zone{VideoCommon::Nx2Profile::Zone::CacheBarrier};
        rasterizer->InvalidateGPUCache();
        return rasterizer->WaitForIdle();
    }
    case MAXWELL3D_REG_INDEX(tiled_cache_barrier): {
        VideoCommon::Nx2Profile::Scope zone{VideoCommon::Nx2Profile::Zone::CacheBarrier};
        return rasterizer->TiledCacheBarrier();
    }
    default:
        draw_manager.ProcessMethodCall(*this, method, argument);
        break;
    }
}

void Maxwell3D::CallMacroMethod(Core::System& system, u32 method, std::span<const u32> parameters) {
    // Reset the current macro.
    executing_macro = 0;

    // Lookup the macro offset
    const u32 entry =
        ((method - MacroRegistersStart) >> 1) % static_cast<u32>(GetNativeMacroPositions().size());

    // Execute the current macro.
    GetNativeMacroEngine().Execute(system, *this, GetNativeMacroPositions()[entry], parameters);
    draw_manager.DrawDeferred(*this);
}

void Maxwell3D::TraceFirstMethodWrite(u32 method, u32 argument) {
    static std::array<bool, Regs::NUM_REGS> seen{};
    if (method < seen.size() && !seen[method]) {
        seen[method] = true;
        LOG_INFO(HW_GPU, "3D method first write: {:#x} (byte {:#x}) = {:#x}", method, method * 4, argument);
    }
}

void Maxwell3D::CallMethod(Core::System& system, u32 method, u32 method_argument, bool is_last_call) {
    if (front_macros && !VideoCommon::GpuSplit::OnBackThread()) {
        ReleaseFront();
    }
    // About half of the split back's 3D writes (~5 million a second in Ocarina of Time) are state
    // registers: CallMethodFast's state write, here without the calls to it. A register not in
    // execution_mask is not SET_SHADOW_RAM_CONTROL either (CallMethodSlow's). The verify switch
    // takes the longer path below; the profile, the write history and the first-write trace
    // leave fast_maxwell3d off.
    if (fast_maxwell3d && executing_macro == 0 && method < MacroRegistersStart &&
        !execution_mask[method] && !perf_verify) {
        return WriteStateRegister(method, method_argument);
    }
    if (!fast_maxwell3d || executing_macro != 0 || method >= MacroRegistersStart ||
        method == MAXWELL3D_REG_INDEX(shadow_ram_control)) {
        return CallMethodSlow(system, method, method_argument, is_last_call);
    }
    if (perf_verify) {
        return VerifyCallMethod(system, method, method_argument, is_last_call);
    }
    CallMethodFast(method, method_argument, is_last_call);
}

namespace {

// Executable methods the draw manager does nothing for (ProcessDirtyRegisters is all they do).
bool IsStoreOnlyExecutable(u32 method) {
    switch (method) {
    case MAXWELL3D_REG_INDEX(vertex_buffer.first):
    case MAXWELL3D_REG_INDEX(vertex_buffer.count):
    case MAXWELL3D_REG_INDEX(index_buffer.first):
    case MAXWELL3D_REG_INDEX(topology_override):
        return true;
    default:
        return false;
    }
}

} // Anonymous namespace

// CallMethodSlow without the checks that cannot apply (CallMethod): no macro collects parameters,
// no write history, trace or profile, not a macro call or SET_SHADOW_RAM_CONTROL. The hot
// executable methods go straight to their handler.
void Maxwell3D::CallMethodFast(u32 method, u32 method_argument, bool is_last_call) {
    if (!execution_mask[method]) {
        return WriteStateRegister(method, method_argument);
    }
    const u32 argument = ProcessShadowRam(method, method_argument);
    ProcessDirtyRegisters(method, argument);
    constexpr u32 cb_data = MAXWELL3D_REG_INDEX(const_buffer.buffer);
    if (method - cb_data < 16) {
        return ProcessCBData(argument);
    }
    switch (method) {
    case MAXWELL3D_REG_INDEX(draw.begin):
        return draw_manager.DrawBegin(*this);
    case MAXWELL3D_REG_INDEX(draw.end):
        return draw_manager.DrawEnd(*this);
    case MAXWELL3D_REG_INDEX(vertex_buffer.first):
    case MAXWELL3D_REG_INDEX(vertex_buffer.count):
    case MAXWELL3D_REG_INDEX(index_buffer.first):
    case MAXWELL3D_REG_INDEX(topology_override):
        return;
    case MAXWELL3D_REG_INDEX(bind_groups[0].raw_config):
        return ProcessCBBind(0);
    case MAXWELL3D_REG_INDEX(bind_groups[1].raw_config):
        return ProcessCBBind(1);
    case MAXWELL3D_REG_INDEX(bind_groups[2].raw_config):
        return ProcessCBBind(2);
    case MAXWELL3D_REG_INDEX(bind_groups[3].raw_config):
        return ProcessCBBind(3);
    case MAXWELL3D_REG_INDEX(bind_groups[4].raw_config):
        return ProcessCBBind(4);
    default:
        return ProcessMethodCall(method, argument, method_argument, is_last_call);
    }
}

void Maxwell3D::VerifyCallMethod(Core::System& system, u32 method, u32 method_argument,
                                 bool is_last_call) {
    // Only register writes can run twice: handlers draw, upload and bind, and a deferred draw
    // ProcessDirtyRegisters may issue is a draw too.
    if ((execution_mask[method] && !IsStoreOnlyExecutable(method)) ||
        draw_manager.HasDeferredDraw()) {
        return CallMethodSlow(system, method, method_argument, is_last_call);
    }
    const u32 reg = regs.reg_array[method];
    const u32 shadow = shadow_state.reg_array[method];
    const DirtyState::Flags flags = dirty.flags;
    CallMethodFast(method, method_argument, is_last_call);
    const u32 fast_reg = regs.reg_array[method];
    const u32 fast_shadow = shadow_state.reg_array[method];
    const DirtyState::Flags fast_flags = dirty.flags;
    regs.reg_array[method] = reg;
    shadow_state.reg_array[method] = shadow;
    dirty.flags = flags;
    CallMethodSlow(system, method, method_argument, is_last_call);
    if (fast_reg == regs.reg_array[method] && fast_shadow == shadow_state.reg_array[method] &&
        fast_flags == dirty.flags) {
        return;
    }
    static std::atomic<u32> reports{};
    if (reports.fetch_add(1, std::memory_order_relaxed) >= 64) {
        return;
    }
    std::string flag_differences;
    const DirtyState::Flags differing = fast_flags ^ dirty.flags;
    for (size_t i = 0; i < differing.size(); ++i) {
        if (differing[i]) {
            flag_differences += fmt::format(" {}={}", i, fast_flags[i] ? 1 : 0);
        }
    }
    LOG_ERROR(HW_GPU,
              "Nx2Perf method: {:#x} = {:#x}: register {:#x} (old {:#x}), shadow {:#x} (old {:#x}), "
              "fast dirty flags differ:{}",
              method, method_argument, fast_reg, regs.reg_array[method], fast_shadow,
              shadow_state.reg_array[method], flag_differences);
}

void Maxwell3D::CallConstBufferWords(Core::System& system, const u32* methods,
                                     const u32* arguments, u32 count) {
    const u32 offset = regs.const_buffer.offset;
    const u64 bytes = u64{count} * sizeof(u32);
    const GPUVAddr address = regs.const_buffer.Address();
    // Keep per-word translation across device pages and per-word clipping at buffer limits.
    if (!fast_maxwell3d || executing_macro != 0 || address == 0 ||
        offset > regs.const_buffer.size || bytes > regs.const_buffer.size - offset ||
        bytes > Core::DEVICE_PAGESIZE - ((address + offset) & Core::DEVICE_PAGEMASK)) {
        for (u32 i = 0; i < count; ++i) {
            CallMethod(system, methods[i], arguments[i], true);
        }
        return;
    }
    std::array<u32, 64> values;
    const bool replay = shadow_state.shadow_ram_control == Regs::ShadowRamControl::Replay;
    for (u32 i = 0; i < count; ++i) {
        values[i] = replay ? shadow_state.reg_array[methods[i]] : arguments[i];
    }
    if (perf_verify) {
        for (u32 i = 0; i < count; ++i) {
            CallMethod(system, methods[i], arguments[i], true);
        }
        std::array<u32, 64> written;
        const bool mapped = memory_manager.IsFullyMappedRange(address + offset, bytes);
        if (mapped) {
            memory_manager.ReadBlockUnsafe(address + offset, written.data(), bytes);
        }
        if (regs.const_buffer.offset != offset + bytes ||
            (mapped && std::memcmp(written.data(), values.data(), bytes) != 0)) {
            static std::atomic<u32> reports{};
            if (reports.fetch_add(1, std::memory_order_relaxed) < 64) {
                LOG_ERROR(HW_GPU,
                          "Nx2Perf method: {} constant buffer words at {:#x}+{:#x}: offset {:#x}, "
                          "expected {:#x}, memory {}",
                          count, address, offset, regs.const_buffer.offset, offset + bytes,
                          mapped && std::memcmp(written.data(), values.data(), bytes) != 0
                              ? "differs"
                              : "matches");
            }
        }
        return;
    }
    // As CallMethodFast for each word: the first may issue a deferred draw, before any upload.
    for (u32 i = 0; i < count; ++i) {
        ProcessDirtyRegisters(methods[i], ProcessShadowRam(methods[i], arguments[i]));
    }
    const GPUVAddr destination = address + offset;
    // #1441: through the bound buffer's translation, its ranges invalidated apart
    // (InvalidateConstantBufferData). Where that can't, the general path below.
    if (!WriteBoundConstantBuffer(offset, values.data(), static_cast<u32>(bytes))) {
        if (VideoCommon::Nx2FastDraw::Enabled()) {
            NoteFastDrawConstantData(destination, destination + bytes);
        }
        // EDEN_NX2_CB_DIRECT: what WriteWordsCached does for words within one device page outside
        // Switch 2 sparse memory, with one translation (as MemoryManager::Write) instead of the page
        // walk: the words, then the invalidation accumulated word by word. Unmapped: the walk.
        u8* const pointer =
            g_cb_direct &&
                    (destination + bytes <= Tegra::g_nx2_sparse_low.load(std::memory_order_relaxed) ||
                     destination >= Tegra::g_nx2_sparse_high.load(std::memory_order_relaxed))
                ? memory_manager.GetPointer(destination)
                : nullptr;
        if (pointer != nullptr) [[likely]] {
            std::memcpy(pointer, values.data(), bytes);
            for (u32 i = 0; i < count; ++i) {
                memory_manager.AccumulateInvalidation(destination + u64{i} * sizeof(u32), sizeof(u32));
            }
        } else {
            memory_manager.WriteWordsCached(destination, values.data(), count);
        }
    }
    regs.const_buffer.offset += static_cast<u32>(bytes);
}

void Maxwell3D::CallMethodSlow(Core::System& system, u32 method, u32 method_argument, bool is_last_call) {
    // Every word of a macro register run is a parameter of the macro it started, whichever macro
    // register it writes (as on the split front, FrontCallMethod). Only a write to an ordinary
    // register before the macro has all its parameters is an error.
    if (executing_macro != 0 && method < MacroRegistersStart) {
        DumpRecentWrites(method);
    }
    if (executing_macro != 0) {
        ASSERT(method >= MacroRegistersStart);
    }
    RecordWrite(method, method_argument, 1, is_last_call ? 1 : 2);

    // Methods after 0xE00 are special, they're actually triggers for some microcode that was
    // uploaded to the GPU during initialization.
    if (method >= MacroRegistersStart) {
        ProcessMacro(system, method, &method_argument, 1, is_last_call);
        return;
    }
    if (trace_first_method_writes) [[unlikely]] {
        TraceFirstMethodWrite(method, method_argument);
    }

    ASSERT(method < Regs::NUM_REGS && "Invalid Maxwell3D register, increase the size of the Regs structure");
    if (VideoCommon::Nx2Profile::Enabled()) [[unlikely]] {
        VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::MethodWrites, 1);
        const bool same = regs.reg_array[method] == method_argument;
        if (same) {
            VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::SameValueWrites, 1);
        }
        VideoCommon::Nx2Profile::CountMethod(method, same);
    }
    if (method == MAXWELL3D_REG_INDEX(shadow_ram_control) && front_macros) [[unlikely]] {
        back_control_writes[back_control_count++ % back_control_writes.size()] = {
            method_argument, static_cast<u32>(shadow_state.shadow_ram_control), is_last_call ? 1U : 0U};
    }
    const u32 argument = ProcessShadowRam(method, method_argument);
    // A state register written with the value it holds changes nothing: no dirty flags (they
    // would make the next draw re-check that state), no method processing. Title 1 makes ~6
    // million 3D register writes a second, ~60% of them same-value, most from macro replays.
    // Methods with side effects (execution_mask: draws, clears, constant buffer data and
    // binds, semaphores, ...) always run. EDEN_NX2_SKIP_SAME=0 turns it off.
    if (!execution_mask[method]) {
        // ProcessMethodCall (and the draw manager behind it) acts only on executable methods.
        if (!SkipSameValueWrites() || regs.reg_array[method] != argument) {
            ProcessDirtyRegisters(method, argument);
        }
        return;
    }
    ProcessDirtyRegisters(method, argument);
    ProcessMethodCall(method, argument, method_argument, is_last_call);
}

void Maxwell3D::DumpRecentWrites(u32 method) {
    static u32 dumps{};
    if (dumps++ >= 4) {
        return;
    }
    LOG_ERROR(HW_GPU, "Write to {:#x} while macro {:#x} collects parameters ({} so far); last writes:",
              method, executing_macro, macro_params.size());
    if (!record_write_history) {
        LOG_ERROR(HW_GPU, "Register write history disabled; EDEN_NX2_3D_HISTORY=1 enables it");
        return;
    }
    for (size_t i = 0; i < recent_writes.size(); ++i) {
        const auto& w = recent_writes[(recent_write_index + i) % recent_writes.size()];
        if (w.amount != 0) {
            LOG_ERROR(HW_GPU, "  method {:#x} arg {:#x} amount {} pending {}", w.method, w.argument,
                      w.amount, w.pending);
        }
    }
}

void Maxwell3D::CallMultiMethod(Core::System& system, u32 method, const u32* base_start, u32 amount, u32 methods_pending) {
    if (front_macros && !VideoCommon::GpuSplit::OnBackThread()) {
        ReleaseFront();
    }
    RecordWrite(method, amount ? base_start[0] : 0, amount, methods_pending);
    // Methods after 0xE00 are special, they're actually triggers for some microcode that was
    // uploaded to the GPU during initialization.
    if (method >= MacroRegistersStart) {
        ProcessMacro(system, method, base_start, amount, amount == methods_pending);
        return;
    }
    // As in ProcessDirtyRegisters: the paths below write without it.
    if (draw_manager.HasDeferredDraw() && !DrawManager::IsInstanceLoopMethod(method)) [[unlikely]] {
        draw_manager.DrawDeferred(*this, true);
    }
    switch (method) {
    case MAXWELL3D_REG_INDEX(const_buffer.buffer):
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 1:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 2:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 3:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 4:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 5:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 6:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 7:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 8:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 9:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 10:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 11:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 12:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 13:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 14:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 15: {
        VideoCommon::Nx2Profile::Scope zone{VideoCommon::Nx2Profile::Zone::CbMulti};
        ProcessCBMultiData(base_start, amount);
        break;
    }
    case MAXWELL3D_REG_INDEX(inline_data): {
        ASSERT(methods_pending == amount);
        VideoCommon::Nx2FastDraw::Break(VideoCommon::Nx2FastDraw::Reason::InlineData);
        upload_state.ProcessData(base_start, amount);
        return;
    }
    case MAXWELL3D_REG_INDEX(draw_inline_index):
    case MAXWELL3D_REG_INDEX(inline_index_2x16.even):
    case MAXWELL3D_REG_INDEX(inline_index_4x8.index0):
        if (shadow_state.shadow_ram_control != Regs::ShadowRamControl::Replay) {
            ProcessInlineIndexMultiData(method, base_start, amount);
            break;
        }
        [[fallthrough]];
    default:
        for (u32 i = 0; i < amount; i++) {
            CallMethod(system, method, base_start[i], methods_pending - i <= 1);
        }
        break;
    }
}

void Maxwell3D::ProcessMacroUpload(u32 data) {
    GetNativeMacroEngine().AddCode(regs.load_mme.instruction_ptr++, data);
}

void Maxwell3D::ProcessMacroBind(u32 data) {
    const auto positions = GetNativeMacroPositions();
    const u32 index = regs.load_mme.start_address_ptr++;
    if (index < positions.size()) {
        positions[index] = data;
    }
}

void Maxwell3D::ProcessFirmwareCall4() {
    LOG_DEBUG(HW_GPU, "(STUBBED) called");

    // Firmware call 4 is a blob that changes some registers depending on its parameters.
    // These registers don't affect emulation and so are stubbed by setting 0xd00 to 1.
    regs.shadow_scratch[0] = 1;
}

void Maxwell3D::StampQueryResult(Core::System& system, u64 payload, bool long_query) {
    const GPUVAddr sequence_address{regs.report_semaphore.Address()};
    if (long_query) {
        memory_manager.Write<u64>(sequence_address + sizeof(u64), system.GPU().GetTicks());
        memory_manager.Write<u64>(sequence_address, payload);
    } else {
        memory_manager.Write<u32>(sequence_address, static_cast<u32>(payload));
    }
    if (const u64 trace = VideoCommon::QueryTrace::Sample(VideoCommon::QueryTrace::Event::Write)) {
        LOG_INFO(HW_GPU,
                 "DKB_QUERY write n={} path=stamp as={} gpu={:#x} bytes={} value={:#x}",
                 trace, memory_manager.GetID(), sequence_address, long_query ? 8 : 4,
                 long_query ? payload : static_cast<u32>(payload));
    }
}

void Maxwell3D::ProcessQueryGet() {
    VideoCommon::QueryPropertiesFlags flags{};
    if (regs.report_semaphore.query.short_query == 0) {
        flags |= VideoCommon::QueryPropertiesFlags::HasTimeout;
    }
    const GPUVAddr sequence_address{regs.report_semaphore.Address()};
    const VideoCommon::QueryType query_type =
        static_cast<VideoCommon::QueryType>(regs.report_semaphore.query.report.Value());
    const u32 payload = regs.report_semaphore.payload;
    const u32 subreport = regs.report_semaphore.query.sub_report;
    switch (regs.report_semaphore.query.operation) {
    case Regs::ReportSemaphore::Operation::Release:
        if (regs.report_semaphore.query.short_query != 0) {
            flags |= VideoCommon::QueryPropertiesFlags::IsAFence;
        }
        rasterizer->Query(sequence_address, query_type, flags, payload, subreport);
        break;
    case Regs::ReportSemaphore::Operation::Acquire:
        // TODO(Blinkhawk): Under this operation, the GPU waits for the CPU to write a value that
        // matches the current payload.
        UNIMPLEMENTED_MSG("Unimplemented query operation ACQUIRE");
        break;
    case Regs::ReportSemaphore::Operation::ReportOnly:
        rasterizer->Query(sequence_address, query_type, flags, payload, subreport);
        break;
    case Regs::ReportSemaphore::Operation::Trap:
        UNIMPLEMENTED_MSG("Unimplemented query operation TRAP");
        break;
    default:
        UNIMPLEMENTED_MSG("Unknown query operation");
        break;
    }
}

void Maxwell3D::ProcessQueryCondition() {
    if (rasterizer->AccelerateConditionalRendering()) {
        execute_on = true;
        return;
    }
    const GPUVAddr condition_address{regs.render_enable.Address()};
    Regs::ReportSemaphore::Compare cmp{};
    bool operands_read = false;
    switch (regs.render_enable_override) {
    case Regs::RenderEnable::Override::AlwaysRender:
        execute_on = true;
        break;
    case Regs::RenderEnable::Override::NeverRender:
        execute_on = false;
        break;
    case Regs::RenderEnable::Override::UseRenderEnable: {
        switch (regs.render_enable.mode) {
        case Regs::RenderEnable::Mode::True: {
            execute_on = true;
            break;
        }
        case Regs::RenderEnable::Mode::False: {
            execute_on = false;
            break;
        }
        case Regs::RenderEnable::Mode::Conditional: {
            memory_manager.ReadBlock(condition_address, &cmp, sizeof(cmp));
            operands_read = true;
            execute_on = cmp.initial_sequence != 0U && cmp.initial_mode != 0U;
            break;
        }
        case Regs::RenderEnable::Mode::IfEqual: {
            memory_manager.ReadBlock(condition_address, &cmp, sizeof(cmp));
            operands_read = true;
            execute_on = cmp.initial_sequence == cmp.current_sequence &&
                         cmp.initial_mode == cmp.current_mode;
            break;
        }
        case Regs::RenderEnable::Mode::IfNotEqual: {
            memory_manager.ReadBlock(condition_address, &cmp, sizeof(cmp));
            operands_read = true;
            execute_on = cmp.initial_sequence != cmp.current_sequence ||
                         cmp.initial_mode != cmp.current_mode;
            break;
        }
        default: {
            UNIMPLEMENTED_MSG("Uninplemented Condition Mode!");
            execute_on = true;
            break;
        }
        }
        break;
    }
    }
    if (const u64 trace =
            VideoCommon::QueryTrace::Sample(VideoCommon::QueryTrace::Event::CpuCondition)) {
        LOG_INFO(HW_GPU,
                 "DKB_CONDITION cpu n={} as={} gpu={:#x} mode={} override={} execute={} "
                 "operands_read={} initial={:#x}/{:#x} current={:#x}/{:#x}",
                 trace, memory_manager.GetID(), condition_address,
                 static_cast<u32>(regs.render_enable.mode),
                 static_cast<u32>(regs.render_enable_override), execute_on, operands_read,
                 cmp.initial_sequence, cmp.initial_mode, cmp.current_sequence, cmp.current_mode);
    }
}

void Maxwell3D::ProcessCounterReset() {
    const auto query_type = [clear_report = regs.clear_report_value]() {
        switch (clear_report) {
        case Tegra::Engines::Maxwell3D::Regs::ClearReport::ZPassPixelCount:
            return VideoCommon::QueryType::ZPassPixelCount64;
        case Tegra::Engines::Maxwell3D::Regs::ClearReport::StreamingPrimitivesSucceeded:
            return VideoCommon::QueryType::StreamingPrimitivesSucceeded;
        case Tegra::Engines::Maxwell3D::Regs::ClearReport::PrimitivesGenerated:
            return VideoCommon::QueryType::PrimitivesGenerated;
        case Tegra::Engines::Maxwell3D::Regs::ClearReport::VtgPrimitivesOut:
            return VideoCommon::QueryType::VtgPrimitivesOut;
        default:
            LOG_DEBUG(HW_GPU, "Unimplemented counter reset={}", clear_report);
            return VideoCommon::QueryType::Payload;
        }
    }();
    rasterizer->ResetCounter(query_type);
}

namespace {

// The split front's occlusion query fast path (EDEN_NX2_OOT_QUERY_FAST). Ocarina resets the
// ZPASS_PIXEL_CNT counter, reports it and sets render_enable around most draws (~40,000 of each
// a second); on the back each report allocates, takes the query cache's locks and makes the next
// fence real. A report address is handled on the front once a GPU condition read it (a
// render_enable operand, or a macro's MME DMA read: NVN2 decides conditional rendering in a
// macro): later reports to it write a "visible" result at once (counts growing by 0x1000 per
// report, so every result is nonzero and no two are equal) and never reach the back. Addresses
// only the CPU reads keep the real path (CPU reads of report memory can't be seen: the results
// are written to guest memory directly).
struct QueryFastState {
    enum : u8 {
        RealReported = 1, ///< A ZPass report to it went to the back
        Source = 2,       ///< A GPU condition read it
        FrontWritten = 4, ///< Holds the front's result: no report to it is queued for the back
        ShortResult = 8,  ///< The front's result is one word (a short query)
    };
    /// The real path's resets keep going for this many ZPass resets after a real ZPass report.
    static constexpr u32 RealWindow = 8192;

    std::unordered_map<GPUVAddr, u8> addresses;
    GPUVAddr min_address = ~GPUVAddr{0};
    GPUVAddr max_address = 0;
    u64 next_result = 0x100000;
    u32 real_window = RealWindow;
    // What the back's render_enable holds: valid in split session sent_session while the front's
    // register still holds last_mode (no other path wrote it).
    bool sent_valid = false;
    u32 sent_mode = 0;
    u32 sent_override = 0;
    u64 sent_session = 0;
    u32 last_mode = 0;

    u8& Note(GPUVAddr address) {
        min_address = (std::min)(min_address, address);
        max_address = (std::max)(max_address, address);
        return addresses[address];
    }
};

QueryFastState& QueryFastFor(const void* engine) {
    // Front thread only.
    static std::unordered_map<const void*, std::unique_ptr<QueryFastState>> states;
    static const void* last_engine = nullptr;
    static QueryFastState* last_state = nullptr;
    if (engine != last_engine) {
        auto& state = states[engine];
        if (!state) {
            state = std::make_unique<QueryFastState>();
        }
        last_engine = engine;
        last_state = state.get();
    }
    return *last_state;
}

/// The front's part of the query fast path for a write of `value` to `method`
/// (Nx2QueryFast::Watches). `regs`: the front's registers before the write. Returns true when
/// the write is done here and must not reach the back; may set `argument` (the render_enable
/// mode decided here).
bool FrontQueryFast(Tegra::MemoryManager& memory, VideoCommon::GpuSplit& split,
                    const void* engine, u32* regs, u32 method, u32 value, u32& argument,
                    bool replay) {
    auto& state = QueryFastFor(engine);
    const bool front_results = VideoCommon::Nx2QueryFast::FakeResults();
    if (replay) {
        // The back takes the shadow RAM's value instead of the argument.
        state.sent_valid = false;
        return false;
    }
    const auto find = [&state](GPUVAddr address) -> u8* {
        const auto it = state.addresses.find(address);
        return it != state.addresses.end() ? &it->second : nullptr;
    };
    switch (method) {
    case 0x157:   // MME_DMA_READ
    case 0x158: { // MME_DMA_READ_FIFOED
        if (!front_results || state.addresses.empty()) {
            return false;
        }
        const GPUVAddr source = (GPUVAddr{regs[0x154] & 0x1ffffff} << 32) | regs[0x155];
        const u64 bytes = u64{(std::min)(value, 0x1000U)} * sizeof(u32);
        if (bytes == 0 || source > state.max_address || source + bytes <= state.min_address) {
            return false;
        }
        const auto note = [](GPUVAddr, u8& flags) {
            flags |= QueryFastState::Source;
        };
        if (bytes <= 1024) {
            for (GPUVAddr address = source & ~GPUVAddr{3}; address < source + bytes;
                 address += sizeof(u32)) {
                if (u8* const flags = find(address)) {
                    note(address, *flags);
                }
            }
        } else {
            for (auto& [address, flags] : state.addresses) {
                if (address >= source && address < source + bytes) {
                    note(address, flags);
                }
            }
        }
        return false;
    }
    case 0x54c: { // clear_report_value
        if (value != static_cast<u32>(Maxwell3D::Regs::ClearReport::ZPassPixelCount)) {
            return false;
        }
        // A real report counts from its reset: keep the resets while real reports go on.
        if (!front_results || state.real_window > 0) {
            state.real_window -= state.real_window > 0 ? 1 : 0;
            return false;
        }
        regs[method] = value;
        return true;
    }
    case 0x6c3: { // report_semaphore.query
        const u32 operation = value & 3;
        const u32 report = (value >> 23) & 0x1f;
        const bool reduction = ((value >> 3) & 1) != 0;
        const bool short_query = ((value >> 28) & 1) != 0;
        const GPUVAddr address = (GPUVAddr{regs[0x6c0]} << 32) | regs[0x6c1];
        // ZPASS_PIXEL_CNT (2, 21 64-bit), release or report only.
        const bool zpass = (report == 2 || report == 21) && (operation == 0 || operation == 2) &&
                           !reduction;
        if (!zpass) {
            // Something else writes it now: its consumers see that, not a query result.
            if (!state.addresses.empty()) {
                state.addresses.erase(address);
            }
            return false;
        }
        u8& flags = state.Note(address);
        if (front_results && (flags & QueryFastState::Source) != 0) {
            const u64 result = state.next_result;
            state.next_result += 0x1000;
            if (state.next_result > 0x7ffff000) {
                state.next_result = 0x100000;
            }
            if (short_query) {
                const u32 word = static_cast<u32>(result);
                memory.WriteBlockUnsafe(address, &word, sizeof(word));
                flags |= QueryFastState::FrontWritten | QueryFastState::ShortResult;
            } else {
                // The time stamp word after it stays (no system clock on the front).
                memory.WriteBlockUnsafe(address, &result, sizeof(result));
                flags = static_cast<u8>((flags | QueryFastState::FrontWritten) &
                                        ~QueryFastState::ShortResult);
            }
            regs[method] = value;
            return true;
        }
        flags = static_cast<u8>((flags | QueryFastState::RealReported) &
                                ~QueryFastState::FrontWritten);
        state.real_window = QueryFastState::RealWindow;
        return false;
    }
    case 0x556: { // render_enable.mode
        const u32 override_mode = regs[0x651]; // render_enable_override
        u32 effective = value;
        if (override_mode == 0 && value >= 2 && value <= 4) {
            // Conditional (2): the 64-bit count at the address is nonzero; if equal (3) and if
            // not equal (4) compare it with the one 16 bytes after it.
            const GPUVAddr address = (GPUVAddr{regs[0x554]} << 32) | regs[0x555];
            const bool pair = value != 2;
            if (front_results) {
                u8* const first = &state.Note(address);
                *first |= QueryFastState::Source;
                u8* second = nullptr;
                if (pair) {
                    second = &state.Note(address + 16);
                    *second |= QueryFastState::Source;
                }
                const auto front_written = [](const u8* flags) {
                    return flags == nullptr || (*flags & QueryFastState::FrontWritten) != 0;
                };
                if (front_written(first) && front_written(second)) {
                    const bool narrow = (*first & QueryFastState::ShortResult) != 0 ||
                                        (second && (*second & QueryFastState::ShortResult) != 0);
                    u64 values[2]{};
                    memory.ReadBlockUnsafe(address, &values[0], narrow ? 4 : 8);
                    if (pair) {
                        memory.ReadBlockUnsafe(address + 16, &values[1], narrow ? 4 : 8);
                    }
                    const bool render = value == 2   ? values[0] != 0
                                        : value == 3 ? values[0] == values[1]
                                                     : values[0] != values[1];
                    effective = render ? 1 : 0;
                }
            }
        }
        // The back's decision depends only on the override and the mode unless the mode reads
        // memory: the same pair again changes nothing there.
        const u64 session = split.Session();
        const bool known = state.sent_valid && state.sent_session == session &&
                           regs[method] == state.last_mode;
        state.last_mode = value;
        const bool fixed = override_mode != 0 || effective <= 1;
        if (known && fixed && effective == state.sent_mode &&
            override_mode == state.sent_override) {
            regs[method] = value;
            return true;
        }
        argument = effective;
        state.sent_valid = fixed;
        state.sent_mode = effective;
        state.sent_override = override_mode;
        state.sent_session = session;
        return false;
    }
    default:
        return false;
    }
}

} // Anonymous namespace

void Maxwell3D::ProcessSyncPoint() {
    if (GetNativeMacroEngine().native_turing) {
        LOG_DEBUG(HW_GPU, "Native 3D syncpoint increment: id={}, raw={:#x}", regs.sync_info.sync_point.Value(), regs.reg_array[MAXWELL3D_REG_INDEX(sync_info)]);
    }
    const u32 sync_point = regs.sync_info.sync_point.Value();
    [[maybe_unused]] const u32 cache_flush = regs.sync_info.clean_l2.Value();
    rasterizer->SignalSyncPoint(sync_point);
}

void Maxwell3D::ProcessCBBind(size_t stage_index) {
    // Bind the buffer currently in CB_ADDRESS to the specified index in the desired shader
    // stage.
    const auto& bind_data = regs.bind_groups[stage_index];
    static std::array<std::array<bool, 32>, 5> seen{};
    if (!seen[stage_index][bind_data.shader_slot]) {
        seen[stage_index][bind_data.shader_slot] = true;
        LOG_INFO(HW_GPU, "Constant buffer bind: stage {} slot {} address {:#x} size {:#x}", stage_index,
                 bind_data.shader_slot.Value(), regs.const_buffer.Address(), regs.const_buffer.size);
    }
    if (bind_data.shader_slot >= Regs::MaxConstBuffers) {
        return; // slot field is 5 bits; the state array holds 18
    }
    auto& buffer = state.shader_stages[stage_index].const_buffers[bind_data.shader_slot];
    if (VideoCommon::Nx2FastDraw::Enabled() &&
        (buffer.enabled != (bind_data.valid.Value() != 0) ||
         buffer.address != regs.const_buffer.Address() || buffer.size != regs.const_buffer.size)) {
        VideoCommon::Nx2FastDraw::NoteBind(stage_index, bind_data.shader_slot);
    }
    buffer.enabled = bind_data.valid.Value() != 0;
    if (VideoCommon::ListMacroTrace::g_enabled && buffer.enabled) {
        VideoCommon::ListMacroTrace::NoteConstBufferBind(
            static_cast<u32>(stage_index), bind_data.shader_slot, regs.const_buffer.Address(),
            regs.const_buffer.size);
    }
    buffer.address = regs.const_buffer.Address();
    buffer.size = regs.const_buffer.size;

    const bool is_enabled = bind_data.valid.Value() != 0;
    if (!is_enabled) {
        rasterizer->DisableGraphicsUniformBuffer(stage_index, bind_data.shader_slot);
        return;
    }
    const GPUVAddr gpu_addr = regs.const_buffer.Address();
    const u32 size = regs.const_buffer.size;
    rasterizer->BindGraphicsUniformBuffer(stage_index, bind_data.shader_slot, gpu_addr, size);
}

namespace {
// Texture and sampler headers are read when image views are resolved, which delta draws skip.
constexpr u64 FastDrawHeaderSize = 0x20;

u64 FastDrawHeadersSize(const Maxwell3D::Regs& regs) {
    return (u64{regs.tex_header.limit} + 1) * FastDrawHeaderSize;
}

// As TextureCache::SynchronizeDescriptors sizes the sampler table.
u64 FastDrawSamplersSize(const Maxwell3D::Regs& regs) {
    return (u64{regs.sampler_binding == Maxwell3D::Regs::SamplerBinding::ViaHeaderBinding
                    ? regs.tex_header.limit
                    : regs.tex_sampler.limit} +
            1) *
           FastDrawHeaderSize;
}
} // Anonymous namespace

bool Maxwell3D::WriteBoundConstantBuffer(u32 offset, const void* data, u32 bytes) {
    namespace FastDraw = VideoCommon::Nx2FastDraw;
    if (!FastDraw::Enabled()) {
        return false;
    }
    const GPUVAddr address = regs.const_buffer.Address();
    const u32 size = regs.const_buffer.size;
    const u64 map_serial = memory_manager.MapSerial();
    const GPUVAddr tic = regs.tex_header.Address();
    const u64 tic_size = FastDrawHeadersSize(regs);
    const GPUVAddr tsc = regs.tex_sampler.Address();
    const u64 tsc_size = FastDrawSamplersSize(regs);
    BoundConstantBuffer& bound = bound_constant_buffer;
    if (bound.address != address || bound.size != size || bound.map_serial != map_serial ||
        bound.tic != tic || bound.tic_size != tic_size || bound.tsc != tsc ||
        bound.tsc_size != tsc_size) {
        bound = {address, size, map_serial, tic, tic_size, tsc, tsc_size, 0};
        const GPUVAddr end = address + size;
        const std::optional<DAddr> device =
            address != 0 && size != 0 && memory_manager.IsContinuousRange(address, size)
                ? memory_manager.GpuToCpuAddress(address)
                : std::nullopt;
        // Data anywhere in a buffer over the headers, at its GPU addresses or through another
        // mapping of their device memory, takes the general path, which checks each write.
        if (device && !(address < tic + tic_size && tic < end) &&
            !(address < tsc + tsc_size && tsc < end)) {
            fast_draw_header_spans.Update(memory_manager, tic, tic_size);
            fast_draw_sampler_spans.Update(memory_manager, tsc, tsc_size);
            if (!fast_draw_header_spans.Intersects(*device, *device + size) &&
                !fast_draw_sampler_spans.Intersects(*device, *device + size)) {
                bound.device = *device;
            }
        }
    }
    const GPUVAddr begin = address + offset;
    // Words on Switch 2 sparse memory go one at a time, as Nx2BackSparseWrite saw them.
    if (bound.device == 0 ||
        (begin + bytes > Tegra::g_nx2_sparse_low.load(std::memory_order_relaxed) &&
         begin < Tegra::g_nx2_sparse_high.load(std::memory_order_relaxed))) {
        return false;
    }
    const DAddr device = bound.device + offset;
    FastDraw::NoteConstantData(begin, begin + bytes);
    FastDraw::NoteDeviceData(device, device + bytes);
    memory_manager.WriteConstantBufferData(device, data, bytes);
    return true;
}

void Maxwell3D::NoteFastDrawConstantData(GPUVAddr begin, GPUVAddr end) {
    namespace FastDraw = VideoCommon::Nx2FastDraw;
    const GPUVAddr tic = regs.tex_header.Address();
    const u64 tic_size = FastDrawHeadersSize(regs);
    const GPUVAddr tsc = regs.tex_sampler.Address();
    const u64 tsc_size = FastDrawSamplersSize(regs);
    if ((begin < tic + tic_size && tic < end) || (begin < tsc + tsc_size && tsc < end)) {
        FastDraw::Break(FastDraw::Reason::TextureHeaders);
        return;
    }
    FastDraw::NoteConstantData(begin, end);
    // The same device memory can be mapped at other GPU addresses: compare where it lands.
    fast_draw_header_spans.Update(memory_manager, tic, tic_size);
    fast_draw_sampler_spans.Update(memory_manager, tsc, tsc_size);
    constexpr u64 PageSize = 0x1000; // the smallest GPU page
    for (GPUVAddr page = begin; page < end;) {
        const GPUVAddr page_end = (std::min)(end, (page & ~(PageSize - 1)) + PageSize);
        if (const std::optional<DAddr> device = memory_manager.GpuToCpuAddress(page)) {
            const DAddr device_end = *device + (page_end - page);
            if (fast_draw_header_spans.Intersects(*device, device_end) ||
                fast_draw_sampler_spans.Intersects(*device, device_end)) {
                FastDraw::Break(FastDraw::Reason::TextureHeaders);
                return;
            }
            FastDraw::NoteDeviceData(*device, device_end);
        }
        page = page_end;
    }
}

void Maxwell3D::ProcessCBMultiData(const u32* start_base, u32 amount) {
    // Write the input value to the current const buffer at the current position.
    const GPUVAddr buffer_address = regs.const_buffer.Address();
    ASSERT(buffer_address != 0);

    // Don't allow writing past the end of the buffer.
    const size_t copy_size = amount * sizeof(u32);
    const size_t room = regs.const_buffer.offset < regs.const_buffer.size
                            ? regs.const_buffer.size - regs.const_buffer.offset
                            : 0;
    const size_t write_size = std::min(copy_size, room / sizeof(u32) * sizeof(u32));
    if (write_size != 0) {
        const GPUVAddr address{buffer_address + regs.const_buffer.offset};
        if (!WriteBoundConstantBuffer(regs.const_buffer.offset, start_base,
                                      static_cast<u32>(write_size))) {
            if (VideoCommon::Nx2FastDraw::Enabled()) {
                NoteFastDrawConstantData(address, address + write_size);
            }
            memory_manager.WriteBlockCached(address, start_base, write_size);
        }
    }
    if (write_size != copy_size) {
        static std::atomic<bool> warned{};
        if (!warned.exchange(true, std::memory_order_relaxed)) {
            LOG_ERROR(HW_GPU,
                      "Dropping constant buffer words past the end: offset={:#x}, size={:#x}, "
                      "bytes={:#x}, written={:#x}",
                      regs.const_buffer.offset, regs.const_buffer.size, copy_size, write_size);
        }
    }

    // Increment the current buffer position.
    regs.const_buffer.offset += static_cast<u32>(copy_size);
}

void Maxwell3D::ProcessCBData(u32 value) {
    ProcessCBMultiData(&value, 1);
}

void Maxwell3D::ProcessInlineIndexMultiData(u32 method, const u32* start_base, u32 amount) {
    if (amount == 0) {
        return;
    }
    const u32 argument = ProcessShadowRam(method, start_base[amount - 1]);
    ProcessDirtyRegisters(method, argument);
    draw_manager.SetInlineIndexBuffer(*this, method, start_base, amount);
}

Texture::TICEntry Maxwell3D::GetTICEntry(u32 tic_index) const {
    const GPUVAddr tic_address_gpu{regs.tex_header.Address() +
                                   tic_index * sizeof(Texture::TICEntry)};
    Texture::TICEntry tic_entry;
    memory_manager.ReadBlockUnsafe(tic_address_gpu, &tic_entry, sizeof(Texture::TICEntry));
    return tic_entry;
}

Texture::TSCEntry Maxwell3D::GetTSCEntry(u32 tsc_index) const {
    const GPUVAddr tsc_address_gpu{regs.tex_sampler.Address() +
                                   tsc_index * sizeof(Texture::TSCEntry)};
    Texture::TSCEntry tsc_entry;
    memory_manager.ReadBlockUnsafe(tsc_address_gpu, &tsc_entry, sizeof(Texture::TSCEntry));
    return tsc_entry;
}

u32 Maxwell3D::GetRegisterValue(u32 method) const {
    ASSERT(method < Regs::NUM_REGS && "Invalid Maxwell3D register");
    return regs.reg_array[method];
}

void Maxwell3D::SetHLEReplacementAttributeType(u32 bank, u32 offset,
                                               HLEReplacementAttributeType name) {
    const u64 key = (static_cast<u64>(bank) << 32) | offset;
    replace_table.emplace(key, name);
}

// Split GPU thread with macros on its front half (gpu_split.h, EDEN_NX2_SPLIT=2). The front keeps
// what macros read: the registers as the back thread will hold them when it reaches the same
// point of the method stream. That is every write the front sends, with the shadow RAM applied,
// plus the few changes the back makes on its own (constant buffer offset, shadow_scratch[0] on
// falcon[4], the macro bind pointer); the front loads and binds the macro code itself.
struct Maxwell3D::FrontState {
    std::array<ControlWrite, 16> control_writes{};
    u32 control_count{};
    std::array<u32, Regs::NUM_REGS> regs{};
    std::array<u32, Regs::NUM_REGS> shadow{};
    Regs::ShadowRamControl control{};
    u32 executing_macro{};
    std::vector<u32> parameters;
    u64 calls{};
    /// A method the draw manager handles was sent since the back last ran DrawDeferred: only
    /// then can it have a deferred instanced draw to issue at the end of a macro.
    bool draw_pending{true};
    /// The back holds state writes in its method sink, which its next executable method consumes
    /// (issuing a deferred draw a changed one ends). Front() consumes it before copying the state.
    bool sink_pending{};
    u32 split_start{}; ///< GpuSplit::Starts() when this copy was made
    bool skip_same{}; ///< SkipSameValueWrites() && FrontSkipSame(), read once
    /// The inline upload being sent, when its words land linearly (FrontDmaRead).
    bool upload_known{};
    u64 upload_device{};
    u32 upload_size{};
    u32 upload_offset{};
    /// The back's draw manager is in an instance loop (DrawMode::Instance): draw.end does nothing.
    bool instance_loop{};
    /// EDEN_NX2_SKIP_SAME_DRAWREG: a draw that binds an index buffer other than the registers'
    /// (inline or small indexes) was sent since index_buffer.first or .count last went to the
    /// back. Until one of them goes again, a same-value index_buffer.first is sent anyway.
    bool index_foreign{true};
    /// What the back's inline upload took at its last launch_dma (Upload::State::ProcessExec):
    /// pitch layout and copy size, sent in split session upload_session (0: none yet).
    u64 upload_session{};
    bool upload_linear{};
    u32 upload_copy_size{};

    bool front_state{}; ///< EDEN_NX2_FRONT_STATE (off with EDEN_NX2_NARROW_DRAIN=2)
    bool draw_batch{};  ///< EDEN_NX2_DRAW_BATCH, with front_state
    bool oot_macros{};  ///< EDEN_NX2_OOT_MACROS and a replacement registered
    bool oot_dump{};    ///< EDEN_NX2_OOT_MACROS_DUMP
    /// State writes not sent yet (method | value << 32), in first-write order; a register's
    /// slot in it plus one (0: none). The back applies them before the next entry.
    u32 delta_count{};
    std::array<u64, Regs::NUM_REGS> delta{};
    std::array<u16, Regs::NUM_REGS> delta_slot{};
    /// Draw batches: the last complete instance-loop group (begin Subsequent .. end) held back
    /// and how many times in a row it came, and the next group while it is captured.
    struct BatchWrite {
        u32 method;
        u32 argument;
        bool is_last;
        bool from_macro;
    };
    static constexpr u32 MaxBatchWrites = 8;
    std::array<BatchWrite, MaxBatchWrites> batch{};
    u32 batch_size{};
    u32 batch_repeats{};
    std::array<BatchWrite, MaxBatchWrites> capture{};
    u32 capture_size{};
    /// Per register: a write to it is a plain state write (FrontStateWrite): front_state on, not
    /// executable, not watched by the query fast path.
    std::array<u8, Regs::NUM_REGS> plain{};
    /// EDEN_NX2_FRONT_CB_ENCODE (with front_state): constant buffer data writes held as one run
    /// (Kind::CbWrite) and bind group writes sent as Kind::CbBind. While a run is held nothing
    /// else is (delta, batch, capture): each flushes the others before it starts.
    bool cb_encode{};
    static constexpr u32 MaxCbWords = 4;
    u32 cb_count{};
    u32 cb_nibbles{};
    GPUVAddr cb_address{}; ///< Where the run's first word goes (for the narrow-drain record)
    std::array<u32, MaxCbWords> cb_words{};
    /// OoT macro replacements per macro slot, valid for oot_generation's code and oot_start.
    std::array<OotMacros::Id, 0x80> oot_macro{};
    std::array<u32, 0x80> oot_start{};
    std::array<u64, 0x80> oot_generation{};
    std::vector<u32> oot_writes; ///< EDEN_NX2_OOT_MACROS_DUMP: the logged call's writes
    bool oot_log_call{};
    /// List draws (nx2_list_draw.h, #1336 port): by start in the macro code of
    /// list_draw_generation, whether the macro is the array draw FrontTakeListDraw sends as an
    /// address.
    std::unordered_map<u32, bool> list_draw_macros;
    u64 list_draw_generation{~u64{0}};
    /// List guard (nx2_list_dispatch.h): the registers the last list draw left with their old
    /// values (bit 0: vertex_buffer.count, bit 1: vertex_buffer.first), on the front and the back
    /// alike, until they are written again; the list its words are in; whether an index count
    /// was written since the last draw.end (the draw then uses neither register).
    u32 list_draw_stale{};
    Nx2ListDispatch::ListReader list_draw_list;
    bool list_draw_indexed{};
    bool in_list_draw{};
    /// By start in the macro code of data_ram_generation, whether the macro uses the data RAM.
    std::unordered_map<u32, bool> data_ram_macros;
    u64 data_ram_generation{~u64{0}};
};

namespace {

// EDEN_NX2_FRONT_STATE=0: send every state write to the back as before (no delta, no batches).
bool FrontStateEnabled() {
    static const bool enabled = VideoCommon::Nx2Switch("EDEN_NX2_FRONT_STATE", true);
    return enabled;
}

// EDEN_NX2_DRAW_BATCH=0: send every instance of an instance loop (with EDEN_NX2_FRONT_STATE).
bool DrawBatchEnabled() {
    static const bool enabled = VideoCommon::Nx2Switch("EDEN_NX2_DRAW_BATCH", true);
    return enabled;
}

// EDEN_NX2_OOT_MACROS=0: run every macro through the JIT (no native replacements).
bool OotMacrosEnabled() {
    static const bool enabled = VideoCommon::Nx2Switch("EDEN_NX2_OOT_MACROS", true);
    return enabled;
}

// EDEN_NX2_OCCLUSION_HLE=0: the native occlusion test macro (0xe2c) reads the two counters with
// an MME DMA read as the code does, instead of writing the comparison as render_enable.
const bool g_occlusion_hle = VideoCommon::Nx2Switch("EDEN_NX2_OCCLUSION_HLE", true);

bool OcclusionHleEnabled() {
    return g_occlusion_hle;
}

// EDEN_NX2_FOLD_INSTANCE=0: an instance loop of the native draw macros draws its first instance
// on its own and again with the others in the loop's deferred draw (DrawManager).
const bool g_fold_instance = VideoCommon::Nx2Switch("EDEN_NX2_FOLD_INSTANCE", true);

bool FoldInstanceEnabled() {
    return g_fold_instance;
}

// EDEN_NX2_FRONT_CB_ENCODE=0: constant buffer data and bind group writes go as method entries.
bool FrontCbEncodeEnabled() {
    static const bool enabled = VideoCommon::Nx2Switch("EDEN_NX2_FRONT_CB_ENCODE", true);
    return enabled;
}

// EDEN_NX2_SKIP_SAME_DRAWREG=0: send executable draw registers written with the value they hold.
const bool g_skip_same_drawreg = VideoCommon::Nx2Switch("EDEN_NX2_SKIP_SAME_DRAWREG", true);

// Executable registers whose write with the value they hold does nothing on the back: the draw
// manager only stores them (IsStoreOnlyExecutable), no deferred draw goes before them
// (DrawManager::IsInstanceLoopMethod) and no pipeline state reads them. index_buffer.first raises
// Dirty::IndexBuffer, which index_buffer.count raises before every indexed draw from the
// registers too (it sets draw_indexed); only an inline or small-index draw since then needs it
// (FrontState::index_foreign).
bool IsSkippableSameValue(u32 method) {
    switch (method) {
    case MAXWELL3D_REG_INDEX(vertex_buffer.first):
    case MAXWELL3D_REG_INDEX(vertex_buffer.count):
    case MAXWELL3D_REG_INDEX(index_buffer.first):
        return true;
    default:
        return false;
    }
}

// Draws that bind an index buffer other than the index_buffer registers describe.
bool IsForeignIndexDraw(u32 method) {
    switch (method) {
    case MAXWELL3D_REG_INDEX(draw_inline_index):
    case MAXWELL3D_REG_INDEX(inline_index_2x16.even):
    case MAXWELL3D_REG_INDEX(inline_index_4x8.index0):
    case MAXWELL3D_REG_INDEX(index_buffer32_first):
    case MAXWELL3D_REG_INDEX(index_buffer16_first):
    case MAXWELL3D_REG_INDEX(index_buffer8_first):
    case MAXWELL3D_REG_INDEX(index_buffer32_subsequent):
    case MAXWELL3D_REG_INDEX(index_buffer16_subsequent):
    case MAXWELL3D_REG_INDEX(index_buffer8_subsequent):
        return true;
    default:
        return false;
    }
}

// Writes between an instance loop's draw.begin and draw.end whose repeats only count an
// instance on the back (state stores, draw_indexed); anything else ends a batch.
bool IsBatchMiddle(u32 method) {
    switch (method) {
    case MAXWELL3D_REG_INDEX(vertex_buffer.first):
    case MAXWELL3D_REG_INDEX(vertex_buffer.count):
    case MAXWELL3D_REG_INDEX(index_buffer.first):
    case MAXWELL3D_REG_INDEX(index_buffer.count):
        return true;
    default:
        return false;
    }
}

// EDEN_NX2_SPLIT_VERIFY=1: every 4096 front macro calls, wait for the back and compare the
// front's registers with the back's (logs and resynchronizes differences).
const bool g_front_verify = std::getenv("EDEN_NX2_SPLIT_VERIFY") != nullptr;

bool FrontVerifyEnabled() {
    return g_front_verify;
}

// Methods whose execution on the back writes guest memory (inline uploads, reports).
bool WritesGuestMemory(u32 method) {
    return method == MAXWELL3D_REG_INDEX(launch_dma) ||
           method == MAXWELL3D_REG_INDEX(inline_data) ||
           method == MAXWELL3D_REG_INDEX(report_semaphore.query);
}

// Executable methods ProcessMethodCall handles itself; the rest go to the draw manager.
bool FrontHandledOutsideDrawManager(u32 method) {
    constexpr u32 cb_data = MAXWELL3D_REG_INDEX(const_buffer.buffer);
    if (method >= cb_data && method < cb_data + 16) {
        return true;
    }
    switch (method) {
    case Maxwell3D::RootTables::SelectMethod:
    case Maxwell3D::RootTables::LoadMethod:
    case MAXWELL3D_REG_INDEX(wait_for_idle):
    case MAXWELL3D_REG_INDEX(shadow_ram_control):
    case MAXWELL3D_REG_INDEX(load_mme.instruction_ptr):
    case MAXWELL3D_REG_INDEX(load_mme.instruction):
    case MAXWELL3D_REG_INDEX(load_mme.start_address):
    case MAXWELL3D_REG_INDEX(falcon[4]):
    case MAXWELL3D_REG_INDEX(bind_groups[0].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[1].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[2].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[3].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[4].raw_config):
    case MAXWELL3D_REG_INDEX(report_semaphore.query):
    case MAXWELL3D_REG_INDEX(render_enable.mode):
    case MAXWELL3D_REG_INDEX(clear_report_value):
    case MAXWELL3D_REG_INDEX(sync_info):
    case MAXWELL3D_REG_INDEX(launch_dma):
    case MAXWELL3D_REG_INDEX(inline_data):
    case MAXWELL3D_REG_INDEX(fragment_barrier):
    case MAXWELL3D_REG_INDEX(invalidate_texture_data_cache):
    case MAXWELL3D_REG_INDEX(tiled_cache_barrier):
        return true;
    default:
        return false;
    }
}

// EDEN_NX2_SPLIT_SKIP=0: send state writes that keep the register's value too.
bool FrontSkipSame() {
    static const bool skip = [] {
        const char* value = std::getenv("EDEN_NX2_SPLIT_SKIP");
        return value == nullptr || value[0] != '0';
    }();
    return skip;
}

// Executable methods the front drops when they keep the register's value. Draw macros write them
// for every instance (~500,000 same values a second in Ocarina of Time); they only store the value
// and no dirty flag covers them. index_buffer.first marks the index buffer dirty, which replaces
// the binding an inline or small index draw left; topology_override is not part of an instance
// loop, so any write to it ends a deferred draw.
bool FrontDropsSameValue(u32 method) {
    return method == MAXWELL3D_REG_INDEX(vertex_buffer.first) ||
           method == MAXWELL3D_REG_INDEX(vertex_buffer.count);
}

} // Anonymous namespace

Maxwell3D::FrontState& Maxwell3D::Front(Core::System& system, VideoCommon::GpuSplit& split) {
    // Copied during an earlier start: while the split was stopped, state writes went to the
    // method sink without reaching it (they hand the front back only once consumed).
    if (front && front->split_start != split.Starts()) [[unlikely]] {
        split.Drain(VideoCommon::DrainSite::FrontStart); // The engine's state is the back's.
        ReleaseFront();
    }
    if (!front) [[unlikely]] {
        // Copy the back's state and take the macros over while it is idle (state writes it
        // still holds in its method sink first).
        split.Drain(VideoCommon::DrainSite::FrontStart);
        ConsumeSink(system);
        front = std::make_unique<FrontState>();
        std::memcpy(front->regs.data(), regs.reg_array.data(), sizeof(front->regs));
        std::memcpy(front->shadow.data(), shadow_state.reg_array.data(), sizeof(front->shadow));
        front->control = shadow_state.shadow_ram_control;
        front->skip_same = SkipSameValueWrites() && FrontSkipSame();
        // Strict narrow drains note each 3D write's position as it is sent: send them at once.
        front->front_state = FrontStateEnabled() && !split.NarrowStrict();
        front->draw_batch = front->front_state && DrawBatchEnabled();
        front->oot_macros = OotMacrosEnabled() && OotMacros::HasHandlers();
        front->oot_dump = OotMacros::DumpEnabled();
        front->oot_generation.fill(~u64{0});
        for (u32 method = 0; method < Regs::NUM_REGS; ++method) {
            front->plain[method] = front->front_state && !execution_mask[method] &&
                                   !VideoCommon::Nx2QueryFast::Watches(method);
        }
        front->cb_encode = front->front_state && FrontCbEncodeEnabled();
        front->split_start = split.Starts();
        LOG_INFO(HW_GPU, "Split front 3D state: delta {} draw batch {} OoT macros {}",
                 front->front_state, front->draw_batch, front->oot_macros);
        front->executing_macro = executing_macro;
        front->parameters = std::move(macro_params);
        macro_params.clear();
        executing_macro = 0;
        front_macros = true;
    }
    return *front;
}

// With the split on, engine methods run on its back only: one here on another thread means the
// split stopped (accuracy settings changed mid-game). Hand the macros back: the front's copy
// goes stale from now on, and a restart copies the registers again (Front).
void Maxwell3D::ReleaseFront() {
    executing_macro = front->executing_macro;
    macro_params = std::move(front->parameters);
    std::memcpy(shadow_state.reg_array.data(), front->shadow.data(), sizeof(front->shadow));
    shadow_state.shadow_ram_control = front->control;
    front.reset();
    front_macros = false;
    LOG_INFO(HW_GPU, "GPU thread split stopped: 3D macros back on the engine");
}

void Maxwell3D::FrontWrite(VideoCommon::GpuSplit& split, FrontState& fs, u32 method,
                           u32 argument, bool is_last_call, bool from_macro) {
    if (fs.list_draw_stale != 0) [[unlikely]] {
        FrontNoteListDrawWrite(split, fs, method);
    }
    if (fs.plain[method]) [[likely]] {
        FrontStateWrite(split, fs, method, argument);
        return;
    }
    FrontWriteFull(split, fs, method, argument, is_last_call, from_macro);
}

void Maxwell3D::FrontStateWrite(VideoCommon::GpuSplit& split, FrontState& fs, u32 method,
                                u32 argument) {
    // FrontWriteFull's path for a state register with front_state on.
    namespace Profile = VideoCommon::Nx2Profile;
    u32 value = argument;
    bool shadow_kept = true;
    if (fs.control == Regs::ShadowRamControl::Track ||
        fs.control == Regs::ShadowRamControl::TrackWithFilter) {
        shadow_kept = fs.shadow[method] == argument;
        fs.shadow[method] = argument;
    } else if (fs.control == Regs::ShadowRamControl::Replay) {
        value = fs.shadow[method];
    }
    if (Profile::Enabled()) [[unlikely]] {
        Profile::Add(Profile::Counter::MethodWrites, 1);
    }
    if (fs.regs[method] == value && shadow_kept && fs.skip_same) {
        if (Profile::Enabled()) [[unlikely]] {
            Profile::Add(Profile::Counter::SameValueWrites, 1);
        }
        return;
    }
    fs.regs[method] = value;
    FrontMarkDelta(split, fs, method, value);
}

void Maxwell3D::FrontWriteFull(VideoCommon::GpuSplit& split, FrontState& fs, u32 method,
                               u32 argument, bool is_last_call, bool from_macro) {
    namespace Profile = VideoCommon::Nx2Profile;
    // As ProcessShadowRam does on the back.
    u32 value = argument;
    bool shadow_kept = true;
    if (fs.control == Regs::ShadowRamControl::Track ||
        fs.control == Regs::ShadowRamControl::TrackWithFilter) {
        shadow_kept = fs.shadow[method] == argument;
        fs.shadow[method] = argument;
    } else if (fs.control == Regs::ShadowRamControl::Replay) {
        value = fs.shadow[method];
    }
    if (Profile::Enabled()) [[unlikely]] {
        Profile::Add(Profile::Counter::MethodWrites, 1);
    }
    if (method == MAXWELL3D_REG_INDEX(shadow_ram_control) && FrontVerifyEnabled()) [[unlikely]] {
        fs.control_writes[fs.control_count++ % fs.control_writes.size()] = {
            argument, static_cast<u32>(fs.control), from_macro ? 2U : 1U};
    }
    if (execution_mask[method]) {
        // Draw batches (EDEN_NX2_DRAW_BATCH) capture whole instance-loop groups, draw.end
        // included, and EDEN_NX2_SKIP_SAME_DRAWREG drops same-value vertex buffer writes outside
        // them: the loop-write skip only applies without batches.
        if (fs.draw_batch) {
            FrontTrackInstanceLoop(fs, method, value);
        } else if (FrontSkipsLoopWrite(fs, method, value, shadow_kept)) {
            if (Profile::Enabled()) [[unlikely]] {
                Profile::Add(Profile::Counter::SameValueWrites, 1);
            }
            return;
        }
    }
    if (VideoCommon::Nx2QueryFast::Watches(method) && VideoCommon::Nx2QueryFast::Enabled())
        [[unlikely]] {
        // Occlusion queries and render_enable (FrontQueryFast).
        if (FrontQueryFast(memory_manager, split, this, fs.regs.data(), method, value, argument,
                           fs.control == Regs::ShadowRamControl::Replay)) {
            return;
        }
    }
    if (!execution_mask[method]) {
        // Nothing would change on the back: CallMethod and ConsumeSinkImpl skip same-value state
        // writes, and the shadow RAM (tracked by default) holds the value already.
        if (fs.regs[method] == value && shadow_kept && fs.skip_same) {
            if (Profile::Enabled()) [[unlikely]] {
                Profile::Add(Profile::Counter::SameValueWrites, 1);
            }
            return;
        }
        fs.regs[method] = value;
        if (fs.front_state) {
            // Sent with the next entry (FrontFlushPending); NoteDraw only acts in strict mode,
            // which turns front_state off.
            FrontMarkDelta(split, fs, method, value);
            return;
        }
    } else {
        // EDEN_NX2_SKIP_SAME_DRAWREG (with front state deltas): index_buffer.first too, and
        // instance stores while nothing waits in the back's sink (deltas never do).
        const bool drawreg_skip =
            g_skip_same_drawreg && fs.front_state && IsSkippableSameValue(method);
        if (drawreg_skip) {
            // Not inside a captured instance-loop group: it is compared with the held batch
            // write by write.
            if (fs.regs[method] == value && shadow_kept && fs.skip_same && !fs.sink_pending &&
                fs.capture_size == 0 &&
                (method != MAXWELL3D_REG_INDEX(index_buffer.first) || !fs.index_foreign)) {
                fs.draw_pending = true;
                if (Profile::Enabled()) [[unlikely]] {
                    Profile::Add(Profile::Counter::SameValueWrites, 1);
                }
                return;
            }
            if (method == MAXWELL3D_REG_INDEX(index_buffer.first)) {
                fs.index_foreign = false;
            }
        } else if (method == MAXWELL3D_REG_INDEX(index_buffer.count)) {
            fs.index_foreign = false;
        } else if (IsForeignIndexDraw(method)) {
            fs.index_foreign = true;
        }
        // Kept while state writes wait in the back's sink: it would consume them before the next
        // engine's work.
        if (!drawreg_skip && FrontDropsSameValue(method) && fs.regs[method] == value &&
            shadow_kept && fs.skip_same && !fs.sink_pending) {
            if (Profile::Enabled()) [[unlikely]] {
                Profile::Add(Profile::Counter::SameValueWrites, 1);
            }
            return;
        }
        fs.regs[method] = value;
        constexpr u32 cb_data = MAXWELL3D_REG_INDEX(const_buffer.buffer);
        switch (method) {
        case MAXWELL3D_REG_INDEX(shadow_ram_control):
            // The back's mode is shadow_state.shadow_ram_control: the shadow copy of this very
            // register, which a later Replay write reads back.
            fs.control = static_cast<Regs::ShadowRamControl>(argument);
            fs.shadow[method] = argument;
            break;
        case MAXWELL3D_REG_INDEX(load_mme.instruction_ptr):
            GetNativeMacroEngine().ClearCode(fs.regs[MAXWELL3D_REG_INDEX(load_mme.instruction_ptr)]);
            break;
        case MAXWELL3D_REG_INDEX(load_mme.instruction):
            GetNativeMacroEngine().AddCode(fs.regs[MAXWELL3D_REG_INDEX(load_mme.instruction_ptr)], value);
            break;
        case MAXWELL3D_REG_INDEX(load_mme.start_address): {
            const auto positions = GetNativeMacroPositions();
            u32& pointer = fs.regs[MAXWELL3D_REG_INDEX(load_mme.start_address_ptr)];
            if (pointer < positions.size()) {
                positions[pointer] = value;
            }
            ++pointer;
            break;
        }
        case MAXWELL3D_REG_INDEX(falcon[4]):
            fs.regs[MAXWELL3D_REG_INDEX(shadow_scratch)] = 1;
            break;
        case MAXWELL3D_REG_INDEX(launch_dma):
            // As Upload::State::ProcessExec (bit 0: pitch layout).
            fs.upload_session = split.Session();
            fs.upload_linear = (value & 1) != 0;
            fs.upload_copy_size = fs.regs[MAXWELL3D_REG_INDEX(upload.line_length_in)] *
                                  fs.regs[MAXWELL3D_REG_INDEX(upload.line_count)];
            break;
        default:
            if (method >= cb_data && method < cb_data + 16) {
                fs.regs[MAXWELL3D_REG_INDEX(const_buffer.offset)] += sizeof(u32);
            } else if (!FrontHandledOutsideDrawManager(method)) {
                fs.draw_pending = true;
            }
            break;
        }
    }
    constexpr u32 cb_words = MAXWELL3D_REG_INDEX(const_buffer.buffer);
    constexpr u32 bind_first = MAXWELL3D_REG_INDEX(bind_groups[0].raw_config);
    constexpr u32 bind_stride =
        MAXWELL3D_REG_INDEX(bind_groups[1].raw_config) - MAXWELL3D_REG_INDEX(bind_groups[0].raw_config);
    if (fs.cb_encode && method - cb_words < 16) {
        // Recorded for narrow drains when the run is sent (FrontEmitCb); NoteDraw only acts in
        // strict mode, which turns cb_encode off.
        FrontCbWord(split, fs, method, argument);
        return;
    }
    if (fs.cb_encode && method - bind_first < 5 * bind_stride &&
        (method - bind_first) % bind_stride == 0) {
        split.CbBind(this, (method - bind_first) / bind_stride, argument);
    } else if (fs.draw_batch && execution_mask[method]) {
        FrontSend(split, fs, method, argument, value, is_last_call, from_macro);
    } else if (from_macro) {
        // Direct writes consume the sink first.
        split.MacroWrite(this, method, argument);
        fs.sink_pending = false;
    } else {
        split.Method(this, method, argument, is_last_call, nullptr);
        fs.sink_pending = !execution_mask[method];
    }
    if (WritesGuestMemory(method)) {
        FrontNoteMemoryWrite(split, fs, method, &value, 1);
    }
    if (split.NarrowDrains()) {
        if (execution_mask[method]) {
            FrontNoteWrites(split, fs, method, 1);
        }
        // A deferred instanced draw may run at any later 3D method (ProcessDirtyRegisters).
        split.NoteDraw();
    }
}

void Maxwell3D::FrontCallMethod(Core::System& system, VideoCommon::GpuSplit& split, u32 method,
                                u32 argument, bool is_last_call) {
    auto& fs = Front(system, split);
    if (method >= MacroRegistersStart) {
        if (fs.executing_macro == 0) {
            fs.executing_macro = method;
        }
        fs.parameters.push_back(argument);
        if (is_last_call) {
            FrontRunMacro(system, split, fs);
        }
        return;
    }
    FrontWrite(split, fs, method, argument, is_last_call, false);
}

void Maxwell3D::FrontCallMultiMethod(Core::System& system, VideoCommon::GpuSplit& split,
                                     u32 method, const u32* base_start, u32 amount,
                                     u32 methods_pending) {
    auto& fs = Front(system, split);
    if (fs.list_draw_stale != 0 && method < MacroRegistersStart) [[unlikely]] {
        // The same register, `amount` times (non-incrementing).
        for (u32 i = 0; i < amount && fs.list_draw_stale != 0; ++i) {
            FrontNoteListDrawWrite(split, fs, method);
        }
    }
    if (method >= MacroRegistersStart) {
        if (fs.executing_macro == 0) {
            fs.executing_macro = method;
        }
        fs.parameters.insert(fs.parameters.end(), base_start, base_start + amount);
        if (amount == methods_pending) {
            FrontRunMacro(system, split, fs);
        }
        return;
    }
    // What CallMultiMethod does to the registers on the back, then one entry for all the words.
    constexpr u32 cb_data = MAXWELL3D_REG_INDEX(const_buffer.buffer);
    const auto tracked = [&fs] {
        return fs.control == Regs::ShadowRamControl::Track ||
               fs.control == Regs::ShadowRamControl::TrackWithFilter;
    };
    if (method >= cb_data && method < cb_data + 16) {
        fs.regs[MAXWELL3D_REG_INDEX(const_buffer.offset)] += amount * sizeof(u32);
    } else if (method == MAXWELL3D_REG_INDEX(inline_data)) {
        // Upload data only.
    } else if ((method == MAXWELL3D_REG_INDEX(draw_inline_index) ||
                method == MAXWELL3D_REG_INDEX(inline_index_2x16.even) ||
                method == MAXWELL3D_REG_INDEX(inline_index_4x8.index0)) &&
               fs.control != Regs::ShadowRamControl::Replay) {
        FrontTrackInstanceLoop(fs, method, 0);
        if (amount != 0) {
            const u32 last = base_start[amount - 1];
            if (tracked()) {
                fs.shadow[method] = last;
            }
            fs.regs[method] = last;
        }
        fs.draw_pending = true;
        fs.index_foreign = true;
    } else {
        // CallMethod for each word, macro uploads among them (load_mme.instruction). A word may
        // change the shadow RAM mode for the next (a run of SET_SHADOW_RAM_CONTROL writes).
        const u32 regs_before = fs.regs[method];
        const u32 shadow_before = fs.shadow[method];
        for (u32 i = 0; i < amount; ++i) {
            const u32 argument = base_start[i];
            u32 value = argument;
            if (method == MAXWELL3D_REG_INDEX(shadow_ram_control) && FrontVerifyEnabled()) [[unlikely]] {
                fs.control_writes[fs.control_count++ % fs.control_writes.size()] = {
                    argument, static_cast<u32>(fs.control), 3U};
            }
            if (tracked()) {
                fs.shadow[method] = argument;
            } else if (fs.control == Regs::ShadowRamControl::Replay) {
                value = fs.shadow[method];
            }
            fs.regs[method] = value;
            FrontTrackInstanceLoop(fs, method, value); // DrawBegin runs for every word
            if (!execution_mask[method]) {
                continue;
            }
            switch (method) {
            case MAXWELL3D_REG_INDEX(shadow_ram_control):
                fs.control = static_cast<Regs::ShadowRamControl>(argument);
                fs.shadow[method] = argument; // The same word on the back (see FrontWrite)
                break;
            case MAXWELL3D_REG_INDEX(load_mme.instruction_ptr):
                GetNativeMacroEngine().ClearCode(value);
                break;
            case MAXWELL3D_REG_INDEX(load_mme.instruction):
                GetNativeMacroEngine().AddCode(fs.regs[MAXWELL3D_REG_INDEX(load_mme.instruction_ptr)], value);
                break;
            case MAXWELL3D_REG_INDEX(load_mme.start_address): {
                const auto positions = GetNativeMacroPositions();
                u32& pointer = fs.regs[MAXWELL3D_REG_INDEX(load_mme.start_address_ptr)];
                if (pointer < positions.size()) {
                    positions[pointer] = value;
                }
                ++pointer;
                break;
            }
            case MAXWELL3D_REG_INDEX(falcon[4]):
                fs.regs[MAXWELL3D_REG_INDEX(shadow_scratch)] = 1;
                break;
            case MAXWELL3D_REG_INDEX(launch_dma):
                // As in FrontWrite.
                fs.upload_session = split.Session();
                fs.upload_linear = (value & 1) != 0;
                fs.upload_copy_size = fs.regs[MAXWELL3D_REG_INDEX(upload.line_length_in)] *
                                      fs.regs[MAXWELL3D_REG_INDEX(upload.line_count)];
                break;
            default:
                if (!FrontHandledOutsideDrawManager(method)) {
                    fs.draw_pending = true;
                }
                if (IsForeignIndexDraw(method)) {
                    fs.index_foreign = true;
                }
                break;
            }
        }
        if (fs.front_state && !execution_mask[method]) {
            // A state register: only its last value reaches the back, with the next entry (the
            // shadow RAM mode can't change within the run).
            if (amount != 0 && (fs.regs[method] != regs_before ||
                                fs.shadow[method] != shadow_before || !fs.skip_same)) {
                FrontMarkDelta(split, fs, method, fs.regs[method]);
            }
            return;
        }
    }
    split.MultiMethod(this, method, base_start, amount, methods_pending, nullptr);
    if (WritesGuestMemory(method) && amount != 0) {
        FrontNoteMemoryWrite(split, fs, method, base_start, amount);
    }
    fs.sink_pending = false; // The back consumes the sink before a run.
    if (split.NarrowDrains()) {
        if (execution_mask[method]) {
            FrontNoteWrites(split, fs, method, amount);
        }
        split.NoteDraw(); // As in FrontWrite
    }
}

void Maxwell3D::FrontRunMacro(Core::System& system, VideoCommon::GpuSplit& split,
                              FrontState& fs) {
    const u32 method = fs.executing_macro;
    fs.executing_macro = 0;
    auto& engine = GetNativeMacroEngine();
    const auto positions = GetNativeMacroPositions();
    // Both binding tables have 0x80 slots: a mask instead of a division by positions.size().
    static_assert(sizeof(macro_positions) == 0x80 * sizeof(u32));
    static_assert(sizeof(NativeMacroContext::positions) == 0x80 * sizeof(u32));
    const u32 slot = ((method - MacroRegistersStart) >> 1) & 0x7f;
    const u32 start = positions[slot];
    VideoCommon::Nx2Profile::CountMethod(method, false);
    // EDEN_NX2_LIST_MACRO_TRACE (#1446/#1482). Our front runs every macro itself (no back macros),
    // so a traced call is never on the back; OoT macro replacements note no reads or writes.
    namespace ListTrace = VideoCommon::ListMacroTrace;
    ListTrace::Call* const trace =
        ListTrace::g_enabled
            ? ListTrace::BeginCall(ListTrace::Engine::Maxwell3D, static_cast<EngineInterface*>(this),
                                   method, engine.native_code, engine.native_code_generation, start,
                                   fs.parameters, native_data_ram, fs.regs)
            : nullptr;
    if (grid_marker.In(native_data_ram)) [[unlikely]] {
        FrontGuardMacro(fs, engine, start);
    }
    {
        VideoCommon::Nx2Profile::Scope profile_scope{VideoCommon::Nx2Profile::Zone::Macro};
        engine.native_interpreter.active_dram = &native_data_ram;
        // OoT macro replacements (macro_hle_oot.h), looked up again when the code or the
        // binding changed.
        OotMacros::Id replacement = OotMacros::Id::None;
        if ((fs.oot_macros || fs.oot_dump) && slot < fs.oot_macro.size()) {
            if (fs.oot_generation[slot] != engine.native_code_generation ||
                fs.oot_start[slot] != start) [[unlikely]] {
                fs.oot_generation[slot] = engine.native_code_generation;
                fs.oot_start[slot] = start;
                const OotMacros::Macro* const found =
                    fs.oot_macros ? OotMacros::Find(OotMacros::HashMacro(engine.native_code, start))
                                  : nullptr;
                fs.oot_macro[slot] = found != nullptr ? found->id : OotMacros::Id::None;
                if (fs.oot_dump) {
                    OotMacros::DumpCode(method, start, engine.native_code);
                }
            }
            replacement = fs.oot_macro[slot];
            if (fs.oot_dump) [[unlikely]] {
                fs.oot_log_call = OotMacros::WantCallLog(method);
                fs.oot_writes.clear();
            }
        }
        if (replacement == OotMacros::Id::None ||
            !FrontRunOotMacro(split, fs, method, replacement)) {
            const auto parameter_count = fs.parameters.size();
            const auto read = [&](u32 address) {
                if (fs.list_draw_stale != 0 && (address == Nx2ListDraw::VertexFirst ||
                                                address == Nx2ListDraw::VertexCount)) [[unlikely]] {
                    // List guard (nx2_list_dispatch.h): a register the last list draw left
                    // unwritten.
                    if (const int guard = Nx2ListDispatch::GuardMode(); guard != 0) {
                        Nx2ListDispatch::Count(Nx2ListDispatch::g_guard.stale_reads);
                        if (guard == 2) {
                            FrontResolveListDraw(split, fs);
                        }
                    }
                }
                const u32 value = address < Regs::NUM_REGS ? fs.regs[address] : 0U;
                ListTrace::NoteRead(trace, address, value);
                return value;
            };
            const auto emit = [&](u32 address, u32 value) {
                if (address >= MacroRegistersStart) {
                    return false;
                }
                if (fs.oot_log_call) [[unlikely]] {
                    fs.oot_writes.push_back(address);
                    fs.oot_writes.push_back(value);
                }
                ListTrace::NoteWrite(trace, address, value);
                FrontWrite(split, fs, address, value, true, true);
                // MME_DMA_READ (into data RAM) and MME_DMA_READ_FIFOED (into the load FIFO).
                if (address == 0x157 || address == 0x158) {
                    FrontDmaRead(split, fs, method, address, value);
                }
                return true;
            };
#ifdef ARCHITECTURE_x86_64
            if (!engine.native_jit) {
                engine.native_jit = std::make_shared<TuringMacroJit>();
            }
            engine.native_jit->Prepare(engine.native_code, engine.native_code_generation);
            // A traced call that follows data RAM runs on the interpreter, which counts its
            // data RAM reads and writes.
            auto result =
                trace != nullptr && trace->data_ram
                    ? engine.native_interpreter.Execute(engine.native_code, start, fs.parameters,
                                                        read, emit)
                    : engine.native_jit->Execute(start, fs.parameters,
                                                 std::span<u32>{native_data_ram}, read, emit);
            if (!result.success && !result.side_effects &&
                result.error == TuringMacroInterpreter::OutsideCodeError) {
                // The compiled program disagrees with the uploaded words about where this entry
                // point is and rejected the call (seen with the GPU thread split on Ocarina of
                // Time, where every macro failed and nothing drew). Run it on the interpreter
                // instead: it decodes the words as they are, and a macro that really is outside
                // the uploaded RAM fails there too and is reported as before. Only calls that
                // wrote nothing outside their registers are replayed; the FIFO is replayed from a
                // copy below.
                std::vector<u32> replay_parameters{fs.parameters};
                const auto replayed = engine.native_interpreter.Execute(
                    engine.native_code, start, replay_parameters, read, emit);
                if (replayed.success) {
                    fs.parameters = std::move(replay_parameters);
                    result = replayed;
                    static std::atomic<u32> fallbacks{};
                    if (fallbacks.fetch_add(1, std::memory_order_relaxed) % 4096 == 0) {
                        LOG_WARNING(HW_GPU,
                                    "Native MME (split front): the compiled program rejected "
                                    "start={:#x}; ran it on the interpreter ({} calls so far)",
                                    start, fallbacks.load(std::memory_order_relaxed));
                    }
                }
            }
#else
            const auto result =
                engine.native_interpreter.Execute(engine.native_code, start, fs.parameters, read, emit);
#endif
            if (!result.success) {
                engine.native_execution_failed = true;
                LOG_ERROR(HW_GPU, "Native MME (split front) failed: start={:#x}, consumed={}/{}, error={}",
                          start, result.consumed, parameter_count, result.error);
            }
        }
        if (fs.oot_log_call) [[unlikely]] {
            OotMacros::LogCall(method, start, fs.parameters, fs.oot_writes);
            fs.oot_log_call = false;
        }
    }
    if (trace != nullptr) {
        ListTrace::EndCall(*trace, false, engine.native_interpreter.data_ram_reads,
                           engine.native_interpreter.data_ram_writes);
    }
    fs.parameters.clear();
    if (fs.draw_pending) {
        fs.draw_pending = false;
        split.Defer([this, system_ptr = &system] { FinishFrontMacro(*system_ptr); },
                    VideoCommon::GpuSplit::FastDrawNeutral);
        split.NoteDraw();
    }
    if (FrontVerifyEnabled() && (++fs.calls & 4095) == 0) [[unlikely]] {
        FrontVerify(split, fs);
    }
}

void Maxwell3D::FrontNoteMemoryWrite(VideoCommon::GpuSplit& split, FrontState& fs, u32 method,
                                     const u32* words, u32 count) {
    const auto unknown = [&split](const char* what) { split.NoteMemoryWrite(0, ~u64{0}, what); };
    if (method == MAXWELL3D_REG_INDEX(report_semaphore.query)) {
        constexpr u32 base = MAXWELL3D_REG_INDEX(report_semaphore);
        const GPUVAddr address = (GPUVAddr{fs.regs[base]} << 32) | fs.regs[base + 1];
        Regs::ReportSemaphore query_regs{};
        query_regs.query.raw = words[count - 1];
        const auto device = memory_manager.GpuToCpuAddress(address);
        if (!device) {
            return; // CounterReport writes nothing at an unmapped address.
        }
        // A short payload release is a fence: its payload is known.
        if (query_regs.query.operation == Regs::ReportSemaphore::Operation::Release &&
            query_regs.query.short_query != 0 &&
            query_regs.query.report == Regs::ReportSemaphore::Report::Payload) {
            const u32 payload = fs.regs[base + 2];
            split.NoteMemoryWriteData(*device, &payload, 1);
            split.NoteReportWrite(*device, 4);
        } else {
            // Counters and timed releases are written when the host query completes, not when
            // the back runs them: reads of these bytes wait until the back ran the report and
            // the host query completed.
            split.NoteAsyncWrite(*device, 16);
            split.NoteMemoryWrite(*device, 16, "3D report");
        }
        return;
    }
    constexpr u32 base = MAXWELL3D_REG_INDEX(upload);
    if (method == MAXWELL3D_REG_INDEX(launch_dma)) {
        const u64 line_length = fs.regs[base];
        const u64 lines = (std::max)(fs.regs[base + 1], 1U);
        const u64 pitch = (std::max<u64>)(fs.regs[base + 4], line_length);
        const GPUVAddr address = (GPUVAddr{fs.regs[base + 2]} << 32) | fs.regs[base + 3];
        const u64 size = pitch * (lines - 1) + line_length;
        const auto device = memory_manager.GpuToCpuAddress(address);
        // LAUNCH_DMA: bit 0 pitch (linear) layout, bits 5:4 completion (2 releases a semaphore).
        const u32 launch = words[count - 1];
        const bool linear = (launch & 1) != 0;
        if (((launch >> 4) & 3) == 2) {
            unknown("upload release");
        }
        bool contiguous = device && linear && lines == 1 && (*device & 3) == 0 &&
                          size <= (u64{64} << 20);
        for (GPUVAddr page = (address & ~GPUVAddr{0xfff}) + 0x1000;
             contiguous && page < address + size; page += 0x1000) {
            contiguous = memory_manager.GpuToCpuAddress(page) == *device + (page - address);
        }
        fs.upload_known = contiguous;
        fs.upload_device = device.value_or(0);
        fs.upload_size = static_cast<u32>((std::min<u64>)(size, ~0U));
        fs.upload_offset = 0;
        if (!fs.upload_known) {
            if (!device || size == 0 || size > (u64{64} << 20)) {
                return unknown("upload unmapped or huge");
            }
            split.NoteMemoryWrite(*device, size, "upload (not linear)");
        }
        return;
    }
    // inline_data words of the current upload.
    if (!fs.upload_known) {
        const auto device = memory_manager.GpuToCpuAddress(
            (GPUVAddr{fs.regs[base + 2]} << 32) | fs.regs[base + 3]);
        if (!device) {
            return unknown("upload data unmapped");
        }
        split.NoteMemoryWrite(*device, fs.regs[base] * u64{(std::max)(fs.regs[base + 1], 1U)} +
                                           u64{fs.regs[base + 4]} * fs.regs[base + 1],
                              "upload data");
        return;
    }
    const u32 remaining_bytes =
        fs.upload_offset < fs.upload_size ? fs.upload_size - fs.upload_offset : 0;
    const u32 whole = (std::min<u32>)(count, remaining_bytes / 4);
    split.NoteMemoryWriteData(fs.upload_device + fs.upload_offset, words, whole);
    if (whole < count && remaining_bytes % 4 != 0) {
        split.NoteMemoryWrite(fs.upload_device + fs.upload_offset + whole * 4, remaining_bytes % 4);
    }
    fs.upload_offset += (std::min<u32>)(count * 4, remaining_bytes);
}

bool Maxwell3D::FrontDmaReadQueued(VideoCommon::GpuSplit& split, FrontState& fs, u32 macro,
                                   u32 method, GPUVAddr source, u32 words) {
    // Without draining the back, for reads of report memory only (pages 3D reports and
    // semaphore releases write, such as occlusion results): the words in guest memory, with the
    // queued writes the front knows the words of on top (fence releases, inline uploads), once
    // the back ran the queued writes it doesn't know that overlap the read. Other memory may be
    // written by queued draws or dispatches the front doesn't track; pages GPU shaders or copies
    // wrote, and counter reports whose host query hasn't completed, take the drained path too.
    const size_t read_bytes = size_t{words} * sizeof(u32);
    if (words == 0) {
        return false;
    }
    const auto device = memory_manager.GpuToCpuAddress(source);
    if (!device || !split.IsReportMemory(*device, read_bytes)) {
        return false;
    }
    // A queued dispatch or copy marks the pages it writes when the back runs it.
    split.WaitForWriters();
    // Every page where a contiguous device range puts it: the queued-write checks look at
    // [*device, *device + read_bytes).
    for (GPUVAddr page = source & ~GPUVAddr{0xfff}; page < source + read_bytes; page += 0x1000) {
        const GPUVAddr at = (std::max)(page, source);
        const auto page_device = memory_manager.GpuToCpuAddress(at);
        if (!page_device || *page_device != *device + (at - source) ||
            Tegra::Nx2IsGpuWritten(*page_device)) {
            return false;
        }
    }
    if (const u64 async_position = split.HasAsyncWrite(*device, read_bytes);
        async_position != 0 && (async_position > split.ConsumedNow() ||
                                !Tegra::Nx2AsyncReportsSettled(*device, read_bytes))) {
        return false;
    }
    thread_local std::vector<u32> fast;
    fast.resize(words);
    const u64 since = split.ConsumedNow();
    memory_manager.ReadBlockUnsafe(source, fast.data(), read_bytes);
    if (!split.OverlayQueuedWrites(since, *device, fast.data(), words)) {
        if (!split.WaitForWritesTo(*device, read_bytes)) {
            return false;
        }
        memory_manager.ReadBlockUnsafe(source, fast.data(), read_bytes);
    }
    // EDEN_NX2_DMA_READ_CHECK=1: also drain, read again and log reads that differ.
    static const bool check = std::getenv("EDEN_NX2_DMA_READ_CHECK") != nullptr;
    if (check) [[unlikely]] {
        static u64 checks{};
        static u64 differences{};
        split.Drain(VideoCommon::DrainSite::MacroDmaRead);
        if (memory_manager.IsMemoryDirty(source, read_bytes, VideoCommon::CacheType::BufferCache)) {
            split.Run(
                [memory = &memory_manager, source, read_bytes] {
                    memory->FlushRegion(source, read_bytes, VideoCommon::CacheType::BufferCache);
                },
                VideoCommon::DrainSite::MacroDmaFlush);
        }
        std::vector<u32> exact(words);
        memory_manager.ReadBlockUnsafe(source, exact.data(), read_bytes);
        ++checks;
        if (exact != fast) {
            if (differences++ < 16) {
                LOG_WARNING(HW_GPU,
                            "DMA read check: macro {:#x} read of {} words at {:#x} differs from "
                            "the drained read",
                            macro, words, source);
            }
            fast = std::move(exact);
        }
        if ((checks & 0xffff) == 0) {
            LOG_INFO(HW_GPU, "DMA read check: {} reads, {} differed", checks, differences);
        }
    }
    const u32 ram = fs.regs[0x156];
    for (u32 i = 0; i < words; ++i) {
        if (method == 0x158) {
            fs.parameters.push_back(fast[i]);
        } else if (ram + i < native_data_ram.size()) {
            native_data_ram[ram + i] = fast[i];
        }
    }
    return true;
}

void Maxwell3D::FrontTrackInstanceLoop(FrontState& fs, u32 method, u32 value) {
    switch (method) {
    case MAXWELL3D_REG_INDEX(draw.begin):
        // Instance id, bits 27:26: First leaves an instance loop, Subsequent enters one.
        if (((value >> 26) & 3) == 0) {
            fs.instance_loop = false;
        } else if (((value >> 26) & 3) == 1) {
            fs.instance_loop = true;
        }
        break;
    case MAXWELL3D_REG_INDEX(draw_inline_index):
    case MAXWELL3D_REG_INDEX(inline_index_2x16.even):
    case MAXWELL3D_REG_INDEX(inline_index_4x8.index0):
        fs.instance_loop = false; // DrawMode::InlineIndex: draw.end draws
        break;
    default:
        break;
    }
}

bool Maxwell3D::FrontSkipsLoopWrite(FrontState& fs, u32 method, u32 value, bool shadow_kept) {
    FrontTrackInstanceLoop(fs, method, value);
    switch (method) {
    case MAXWELL3D_REG_INDEX(draw.end):
    case MAXWELL3D_REG_INDEX(vertex_buffer.first):
    case MAXWELL3D_REG_INDEX(vertex_buffer.count):
        break;
    default:
        return false;
    }
    // Sent while state writes wait in the back's sink: it consumes them at the next executable
    // method, and a changed one ends a deferred draw that must issue before the next engine's work
    // (a compute dispatch after a dropped vertex_buffer.count would otherwise run first).
    if (!fs.skip_same || fs.sink_pending || !shadow_kept || fs.regs[method] != value) {
        return false;
    }
    // Instance loop methods flush no deferred draw, and the draw manager ignores these: DrawEnd
    // draws nothing in an instance loop, and it only reads the vertex buffer first and count
    // (index_buffer writes also mark the index buffer dirty, so they are always sent).
    return method != MAXWELL3D_REG_INDEX(draw.end) || fs.instance_loop;
}

void Maxwell3D::FrontNoteWrites(VideoCommon::GpuSplit& split, FrontState& fs, u32 method,
                                u32 words) {
    // The registers in fs are what the back holds when it runs the method.
    const auto address = [&fs](u32 high, u32 low) {
        return (GPUVAddr{fs.regs[high]} << 32) | GPUVAddr{fs.regs[low]};
    };
    constexpr u32 cb_data = MAXWELL3D_REG_INDEX(const_buffer.buffer);
    if (method >= cb_data && method < cb_data + 16) {
        // ProcessCBMultiData: at the offset before the words (FrontWrite has advanced it).
        const u32 bytes = words * static_cast<u32>(sizeof(u32));
        const u32 offset = fs.regs[MAXWELL3D_REG_INDEX(const_buffer.offset)] - bytes;
        split.NoteWrite(&memory_manager,
                        address(MAXWELL3D_REG_INDEX(const_buffer.address_high),
                                MAXWELL3D_REG_INDEX(const_buffer.address_low)) +
                            offset,
                        bytes);
        return;
    }
    switch (method) {
    case MAXWELL3D_REG_INDEX(inline_data): {
        if (fs.upload_session != split.Session()) {
            split.NoteUnboundedWrite(false, VideoCommon::UnboundedCause::Inline3D);
            return;
        }
        Upload::Registers upload{};
        std::memcpy(&upload, fs.regs.data() + MAXWELL3D_REG_INDEX(upload), sizeof(upload));
        const auto [begin, end] = Upload::WriteRange(upload, fs.upload_linear, fs.upload_copy_size);
        split.NoteWrite(&memory_manager, begin, end - begin);
        return;
    }
    case MAXWELL3D_REG_INDEX(report_semaphore.query): {
        // ProcessQueryGet: the report's value and time stamp.
        const GPUVAddr report = address(MAXWELL3D_REG_INDEX(report_semaphore.address_high),
                                        MAXWELL3D_REG_INDEX(report_semaphore.address_low));
        Regs::ReportSemaphore query{};
        query.query.raw = fs.regs[MAXWELL3D_REG_INDEX(report_semaphore.query)];
        // A short query writes one word, without the time stamp.
        split.NoteWrite(&memory_manager, report, query.query.short_query != 0 ? 4 : 16, true);
        if (split.SemaphorePredict()) {
            // A payload release (no counter, no reduction): its value is known now.
            if (query.query.operation == Regs::ReportSemaphore::Operation::Release &&
                query.query.report == Regs::ReportSemaphore::Report::Payload &&
                query.query.reduction_enable == 0) {
                split.NoteRelease(&memory_manager, report,
                                  fs.regs[MAXWELL3D_REG_INDEX(report_semaphore.payload)]);
            } else {
                split.ForgetRelease(&memory_manager, report);
            }
        }
        return;
    }
    case MAXWELL3D_REG_INDEX(wait_for_idle):
        // Host waits: the draws before them are done before the commands after them run.
        split.NoteUnboundedWrite(false, VideoCommon::UnboundedCause::Wfi3D);
        return;
    case MAXWELL3D_REG_INDEX(invalidate_texture_data_cache):
        // The back runs InvalidateGPUCache (the CPU's writes into the caches; GPU bytes next to
        // them are downloaded, only on GPU-written pages) and a WFI. A barrier: it still holds
        // reads after host waits, but releases no semaphore, so with EDEN_NX2_ACQUIRE_POLL it
        // doesn't hold acquires (GpuSplit::NoteUnboundedWrite, EDEN_NX2_ACQUIRE_SKIP_BARRIERS).
        split.NoteUnboundedWrite(false, VideoCommon::UnboundedCause::TexInvalidate);
        return;
    case MAXWELL3D_REG_INDEX(render_enable.mode):
        // Conditional rendering reads query results; it writes no guest memory
        // (EDEN_NX2_NARROW_RENDER_ENABLE=0: count it as an unbounded write, as before).
        if (!split.NarrowRenderEnable()) {
            split.NoteUnboundedWrite(false, VideoCommon::UnboundedCause::RenderEnable);
        }
        return;
    default:
        // Draws (GpuSplit::NoteDraw), clears, binds, barriers, syncpoints (the readers drain for
        // the syncpoint shim) and register-only methods.
        return;
    }
}

void Maxwell3D::FrontDmaRead(VideoCommon::GpuSplit& split, FrontState& fs, u32 macro,
                             u32 method, u32 count) {
    // The words may be written by methods still queued for the back (indirect draw arguments
    // from a culling dispatch), and the caches that may hold newer ones belong to it.
    const GPUVAddr source = (GPUVAddr{fs.regs[0x154] & 0x1ffffff} << 32) | fs.regs[0x155];
    const u32 ram = fs.regs[0x156];
    const u32 words = std::min<u32>(count, 0x1000);
    const size_t read_bytes = size_t{words} * sizeof(u32);
    std::vector<u32> diag_before;
    if (Tegra::Nx2CmdPageDiag::g_enabled) {
        diag_before.resize(words);
        memory_manager.ReadBlockUnsafe(source, diag_before.data(), read_bytes);
    }
    VideoCommon::Nx2Profile::CountDmaRead(macro);
    // Report memory first: read without draining, with the queued writes the front knows the
    // words of on top (FrontDmaReadQueued). Other reads take the narrow drains when on.
    if (diag_before.empty() && FrontDmaReadQueued(split, fs, macro, method, source, words)) {
        return;
    }
    // Narrow drains: wait only for the queued work that may write the words. On pages GPU work
    // wrote through the buffer cache newer words may be in its caches: drain and flush as before.
    bool narrowed = false;
    if (split.NarrowDrains()) {
        // EDEN_NX2_NARROW_DMA_RECORDED (Ocarina): its macros never read what unbounded GPU work
        // writes (no read on a GPU-written page, NARROW_VERIFY clean), so only the recorded
        // writes over the words count.
        split.WaitForWritesTo(&memory_manager, source, read_bytes,
                              VideoCommon::DrainSite::NarrowMacroDmaRead,
                              split.NarrowDmaRecorded() ? VideoCommon::NarrowBase::Recorded
                                                        : VideoCommon::NarrowBase::Read);
        narrowed = !memory_manager.Nx2ReadNeedsDrain(source, read_bytes);
    }
    std::vector<u32> narrowed_words;
    if (narrowed && VideoCommon::GpuSplit::NarrowVerify()) {
        // EDEN_NX2_NARROW_VERIFY: read the words now, then again after the full drain.
        narrowed_words.resize(words);
        for (u32 i = 0; i < words; ++i) {
            narrowed_words[i] = memory_manager.Read<u32>(source + u64{i} * sizeof(u32));
        }
        narrowed = false;
    }
    if (!narrowed) {
        split.Drain(VideoCommon::DrainSite::MacroDmaRead);
    }
    if (VideoCommon::QueryDma::TraceEnabled()) {
        split.Run(
            [this, macro, method, source, read_bytes] {
                if (VideoCommon::QueryDma::HasPendingQueries(memory_manager, *rasterizer, source,
                                                             read_bytes)) {
                    VideoCommon::QueryDma::Trace(memory_manager, *rasterizer, macro, method,
                                                 source, read_bytes);
                }
            },
            VideoCommon::DrainSite::MacroDmaFlush);
    }
    if (VideoCommon::Nx2Profile::Enabled()) {
        VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::DmaReadBytes, read_bytes);
    }
    if (!narrowed &&
        memory_manager.IsMemoryDirty(source, read_bytes, VideoCommon::CacheType::BufferCache)) {
        split.Run(
            [memory = &memory_manager, source, read_bytes] {
                memory->FlushRegion(source, read_bytes, VideoCommon::CacheType::BufferCache);
            },
            VideoCommon::DrainSite::MacroDmaFlush);
    }
    // Write out the reports held in the query cache, or the read returns stale occlusion counts.
    // Only pages GPU work reported or released to can hold them (GpuSplit::NoteGpuSemaphore,
    // which the narrow drains feed): a narrowed read of other pages skips the drain this takes.
    if (!narrowed || VideoCommon::GpuSplit::MayHoldReports(source, read_bytes)) {
        split.Run(
            [this, source, read_bytes] {
                VideoCommon::QueryDma::FlushPending(memory_manager, *rasterizer, source,
                                                    read_bytes);
            },
            VideoCommon::DrainSite::MacroDmaFlush);
    }
    u32 changed{};
    u32 mismatched{};
    for (u32 i = 0; i < words; ++i) {
        const u32 word = memory_manager.Read<u32>(source + u64{i} * sizeof(u32));
        if (method == 0x158) {
            fs.parameters.push_back(word);
        } else if (ram + i < native_data_ram.size()) {
            native_data_ram[ram + i] = word;
        }
        if (!diag_before.empty() && diag_before[i] != word) {
            ++changed;
        }
        if (!narrowed_words.empty() && narrowed_words[i] != word) {
            ++mismatched;
        }
    }
    if (!narrowed_words.empty()) {
        VideoCommon::GpuSplit::ReportNarrowCheck(VideoCommon::DrainSite::NarrowMacroDmaRead,
                                                 mismatched, macro, source);
    }
    if (Tegra::Nx2CmdPageDiag::g_enabled) {
        bool gpu_written = false;
        for (GPUVAddr page = source & ~GPUVAddr{0xfff}; page < source + read_bytes; page += 0x1000) {
            const auto device = memory_manager.GpuToCpuAddress(page);
            gpu_written |= device && Tegra::Nx2IsGpuWritten(*device);
        }
        Tegra::Nx2CmdPageDiag::RecordDmaRead(source, words, changed, gpu_written);
    }
}

void Maxwell3D::FrontVerify(VideoCommon::GpuSplit& split, FrontState& fs) {
    split.Drain(VideoCommon::DrainSite::FrontVerify);
    static u64 checks{};
    static u64 differences{};
    ++checks;
    // State writes the back still holds in its method sink count as done.
    std::array<u32, Regs::NUM_REGS> expected;
    std::memcpy(expected.data(), regs.reg_array.data(), sizeof(expected));
    for (const auto& [method, value] : method_sink) {
        expected[method] = shadow_state.shadow_ram_control == Regs::ShadowRamControl::Replay
                               ? shadow_state.reg_array[method]
                               : value;
    }
    static u64 control_differences{};
    if ((fs.control != shadow_state.shadow_ram_control ||
         fs.shadow[MAXWELL3D_REG_INDEX(shadow_ram_control)] != shadow_state.reg_array[MAXWELL3D_REG_INDEX(shadow_ram_control)] ||
         fs.regs[MAXWELL3D_REG_INDEX(shadow_ram_control)] != expected[MAXWELL3D_REG_INDEX(shadow_ram_control)]) &&
        control_differences < 4) {
        LOG_WARNING(HW_GPU, "Split shadow RAM control writes, front ({} total) / back ({} total):",
                    fs.control_count, back_control_count);
        for (u32 i = 0; i < 16; ++i) {
            const auto& f = fs.control_writes[(fs.control_count + i) % 16];
            const auto& b = back_control_writes[(back_control_count + i) % 16];
            LOG_WARNING(HW_GPU, "  front {:#x} (mode {}, path {}) | back {:#x} (mode {}, last {})",
                        f.argument, f.before, f.path, b.argument, b.before, b.path);
        }
    }
    for (u32 i = 0; i < Regs::NUM_REGS; ++i) {
        if (fs.regs[i] != expected[i]) {
            if (differences++ < 64) {
                LOG_WARNING(HW_GPU, "Split front register {:#x}: front {:#x}, back {:#x}", i,
                            fs.regs[i], expected[i]);
            }
            fs.regs[i] = expected[i];
        }
    }
    if (fs.control != shadow_state.shadow_ram_control) {
        if (control_differences++ < 16) {
            LOG_WARNING(HW_GPU, "Split front shadow RAM control {} vs back {}",
                        static_cast<u32>(fs.control),
                        static_cast<u32>(shadow_state.shadow_ram_control));
        }
        fs.control = shadow_state.shadow_ram_control;
    }
    for (u32 i = 0; i < Regs::NUM_REGS; ++i) {
        if (fs.shadow[i] != shadow_state.reg_array[i]) {
            if (control_differences++ < 16) {
                LOG_WARNING(HW_GPU, "Split front shadow register {:#x}: front {:#x}, back {:#x}", i,
                            fs.shadow[i], shadow_state.reg_array[i]);
            }
            fs.shadow[i] = shadow_state.reg_array[i];
        }
    }
    if ((checks & 63) == 1) {
        LOG_INFO(HW_GPU, "Split front register checks: {} done, {} differences", checks, differences);
    }
}

void Maxwell3D::FinishFrontMacro(Core::System& system) {
    // What ProcessMacro and CallMacroMethod do around the macro.
    VideoCommon::Nx2Profile::Scope zone{VideoCommon::Nx2Profile::Zone::FinishMacro};
    ConsumeSink(system);
    draw_manager.DrawDeferred(*this);
}

void Maxwell3D::ApplyFrontDelta(Core::System& system, const u64* entries, u32 count) {
    // What CallMethodFast and ConsumeSinkImpl did for each state register write: the shadow RAM
    // tracks it (the front applied a Replay already; the mode can't change within a delta, its
    // register is executable), and a value the register holds changes nothing.
    ConsumeSink(system);
    const auto control = shadow_state.shadow_ram_control;
    const bool track = control == Regs::ShadowRamControl::Track ||
                       control == Regs::ShadowRamControl::TrackWithFilter;
    const bool skip_same = SkipSameValueWrites();
    u32 i = 0;
    u32 same = 0;
    // A deferred instanced draw goes before the first write outside its loop
    // (ProcessDirtyRegisters); once it went, nothing in a delta defers another.
    for (; i < count && draw_manager.HasDeferredDraw(); ++i) {
        const u32 method = static_cast<u32>(entries[i]) & 0xffff;
        const u32 value = static_cast<u32>(entries[i] >> 32);
        if (track) {
            shadow_state.reg_array[method] = value;
        }
        if (!skip_same || regs.reg_array[method] != value) {
            ProcessDirtyRegisters(method, value);
        } else {
            ++same;
        }
    }
    static_assert(std::tuple_size_v<DirtyState::Tables> == 2,
                  "ApplyFrontDelta raises exactly two tables");
    // The rest: the register and its two dirty flags, the loop specialized on the shadow RAM
    // mode and the same-value skip (fixed for the delta). The flags are collected and raised
    // together at the end: nothing reads them in between.
    const auto& table0 = dirty.tables[0];
    const auto& table1 = dirty.tables[1];
    u32* const reg_array = regs.reg_array.data();
    u32* const shadow_array = shadow_state.reg_array.data();
    DirtyState::Flags raised;
    // Fast draw (nx2_fast_draw.h): as ProcessDirtyRegisters, a changed register that isn't a draw
    // parameter ends draw reuse (once for the delta).
    const bool fast_draw = VideoCommon::Nx2FastDraw::Enabled();
    u32 fast_draw_break = ~0U;
    const auto apply = [&]<bool Track, bool SkipSame>() {
        for (; i < count; ++i) {
            const u64 word = entries[i];
            const u32 method = static_cast<u32>(word) & 0xffff;
            const u32 value = static_cast<u32>(word >> 32);
            if constexpr (Track) {
                shadow_array[method] = value;
            }
            if constexpr (SkipSame) {
                if (reg_array[method] == value) {
                    ++same;
                    continue;
                }
            }
            if (fast_draw && fast_draw_break == ~0U && reg_array[method] != value &&
                !IsFastDrawParameter(method)) [[unlikely]] {
                fast_draw_break = method;
            }
            reg_array[method] = value;
            raised[table0[method]] = true;
            raised[table1[method]] = true;
        }
    };
    if (track) {
        if (skip_same) {
            apply.template operator()<true, true>();
        } else {
            apply.template operator()<true, false>();
        }
    } else if (skip_same) {
        apply.template operator()<false, true>();
    } else {
        apply.template operator()<false, false>();
    }
    dirty.flags |= raised;
    if (fast_draw_break != ~0U) {
        VideoCommon::Nx2FastDraw::Break(VideoCommon::Nx2FastDraw::Reason::Register,
                                        fast_draw_break);
    }
    if (VideoCommon::Nx2Profile::Enabled()) [[unlikely]] {
        VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::DeltaWords, count);
        VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::DeltaSameWords, same);
    }
}

void Maxwell3D::RunDrawBatch(Core::System& system, const u64* methods, u32 count, u32 repeats) {
    // The first instance as it was sent. Each repeat wrote the same values again: the registers,
    // shadow RAM and dirty flags stay, draw.begin (InstanceId::Subsequent) counts one more
    // instance, index_buffer.count sets draw_indexed again and draw.end waits for the loop to end.
    for (u32 i = 0; i < count; ++i) {
        ConsumeSink(system);
        CallMethod(system, static_cast<u32>(methods[i]) & 0xffff,
                   static_cast<u32>(methods[i] >> 32), true);
    }
    draw_manager.draw_state.instance_count += repeats - 1;
    if (VideoCommon::Nx2Profile::Enabled()) [[unlikely]] {
        VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::BatchRepeats,
                                     repeats > 0 ? repeats - 1 : 0);
    }
}

void Maxwell3D::FrontMarkDelta(VideoCommon::GpuSplit& split, FrontState& fs, u32 method,
                               u32 value) {
    if (fs.batch_size != 0 || fs.capture_size != 0 || fs.cb_count != 0) {
        // The held-back draws come before this write (a state change ends the instance loop).
        split.RunPendingFlush();
    }
    const u64 entry = u64{method} | (u64{value} << 32);
    u16& slot = fs.delta_slot[method];
    if (slot == 0) {
        fs.delta[fs.delta_count] = entry;
        slot = static_cast<u16>(++fs.delta_count);
    } else {
        fs.delta[slot - 1] = entry;
    }
    split.SetPendingFlush(&FrontFlushPending, this);
}

void Maxwell3D::FrontSend(VideoCommon::GpuSplit& split, FrontState& fs, u32 method,
                          u32 argument, u32 value, bool is_last_call, bool from_macro) {
    const FrontState::BatchWrite write{method, argument, is_last_call, from_macro};
    if (method == MAXWELL3D_REG_INDEX(draw.begin)) {
        // Regs::Draw::instance_id, bits 26-27 of the value the back will hold.
        constexpr u32 subsequent = static_cast<u32>(Regs::Draw::InstanceId::Subsequent);
        if (((value >> 26) & 3) == subsequent) {
            if (fs.capture_size != 0 || fs.delta_count != 0 || fs.cb_count != 0) {
                // A group that never ended, or state writes: they go first.
                split.RunPendingFlush();
            }
            fs.capture[0] = write;
            fs.capture_size = 1;
            split.SetPendingFlush(&FrontFlushPending, this);
            return;
        }
    } else if (fs.capture_size != 0) {
        if (IsBatchMiddle(method) && fs.capture_size + 1 < FrontState::MaxBatchWrites) {
            fs.capture[fs.capture_size++] = write;
            return;
        }
        if (method == MAXWELL3D_REG_INDEX(draw.end)) {
            fs.capture[fs.capture_size++] = write;
            bool same = fs.batch_size == fs.capture_size;
            for (u32 i = 0; same && i < fs.capture_size; ++i) {
                same = fs.batch[i].method == fs.capture[i].method &&
                       fs.batch[i].argument == fs.capture[i].argument;
            }
            if (same) {
                ++fs.batch_repeats;
                fs.capture_size = 0;
                return;
            }
            // Another group: the held one goes now, this one is held.
            split.ClearPendingFlush();
            FrontEmitBatch(split, fs);
            fs.batch = fs.capture;
            fs.batch_size = fs.capture_size;
            fs.batch_repeats = 1;
            fs.capture_size = 0;
            split.SetPendingFlush(&FrontFlushPending, this);
            return;
        }
    }
    // Anything held back goes first (GpuSplit::Reserve runs FrontFlushPending).
    if (from_macro) {
        split.MacroWrite(this, method, argument);
    } else {
        split.Method(this, method, argument, is_last_call, nullptr);
    }
}

void Maxwell3D::FrontEmitBatch(VideoCommon::GpuSplit& split, FrontState& fs) {
    if (fs.batch_size == 0) {
        return;
    }
    if (fs.batch_repeats > 1) {
        std::array<u64, FrontState::MaxBatchWrites> words;
        for (u32 i = 0; i < fs.batch_size; ++i) {
            words[i] = u64{fs.batch[i].method} | (u64{fs.batch[i].argument} << 32);
        }
        split.DrawBatch(this, words.data(), fs.batch_size, fs.batch_repeats);
    } else {
        for (u32 i = 0; i < fs.batch_size; ++i) {
            const auto& write = fs.batch[i];
            if (write.from_macro) {
                split.MacroWrite(this, write.method, write.argument);
            } else {
                split.Method(this, write.method, write.argument, write.is_last, nullptr);
            }
        }
    }
    fs.batch_size = 0;
    fs.batch_repeats = 0;
}

void Maxwell3D::FrontEmitCapture(VideoCommon::GpuSplit& split, FrontState& fs) {
    for (u32 i = 0; i < fs.capture_size; ++i) {
        const auto& write = fs.capture[i];
        if (write.from_macro) {
            split.MacroWrite(this, write.method, write.argument);
        } else {
            split.Method(this, write.method, write.argument, write.is_last, nullptr);
        }
    }
    fs.capture_size = 0;
}

void Maxwell3D::FrontFlushPending(void* maxwell3d) {
    // GpuSplit cleared the hook before calling: the entries below don't come back here.
    auto* const self = static_cast<Maxwell3D*>(maxwell3d);
    FrontState& fs = *self->front;
    auto& split = VideoCommon::GpuSplit::Get();
    self->FrontEmitCb(split, fs);
    self->FrontEmitBatch(split, fs);
    self->FrontEmitCapture(split, fs);
    const u32 count = fs.delta_count;
    if (count == 0) {
        return;
    }
    split.StateDelta(self, count, [&fs, count](u64* out) {
        std::memcpy(out, fs.delta.data(), size_t{count} * sizeof(u64));
    });
    for (u32 i = 0; i < count; ++i) {
        fs.delta_slot[static_cast<u32>(fs.delta[i]) & 0xffff] = 0;
    }
    fs.delta_count = 0;
}

void Maxwell3D::FrontCbWord(VideoCommon::GpuSplit& split, FrontState& fs, u32 method,
                            u32 argument) {
    if (fs.cb_count == 0 || fs.cb_count == FrontState::MaxCbWords) {
        // Whatever is held goes first (a full run too). FrontWriteFull has advanced the offset
        // past this word already.
        split.RunPendingFlush();
        FrontEmitCb(split, fs);
        const GPUVAddr buffer =
            (GPUVAddr{fs.regs[MAXWELL3D_REG_INDEX(const_buffer.address_high)]} << 32) |
            fs.regs[MAXWELL3D_REG_INDEX(const_buffer.address_low)];
        fs.cb_address =
            buffer + static_cast<u32>(fs.regs[MAXWELL3D_REG_INDEX(const_buffer.offset)] -
                                      sizeof(u32));
    }
    constexpr u32 cb_data = MAXWELL3D_REG_INDEX(const_buffer.buffer);
    fs.cb_nibbles |= (method - cb_data) << (fs.cb_count * 4);
    fs.cb_words[fs.cb_count++] = argument;
    split.SetPendingFlush(&FrontFlushPending, this);
}

void Maxwell3D::FrontEmitCb(VideoCommon::GpuSplit& split, FrontState& fs) {
    const u32 count = fs.cb_count;
    if (count == 0) {
        return;
    }
    const u32 nibbles = fs.cb_nibbles;
    fs.cb_count = 0;
    fs.cb_nibbles = 0;
    split.CbWrite(this, nibbles, fs.cb_words.data(), count);
    // As FrontNoteWrites for constant buffer data, once for the run.
    split.NoteWrite(&memory_manager, fs.cb_address, u64{count} * sizeof(u32));
}

void Maxwell3D::BackCbWrite(Core::System& system, u32 nibbles, u32 first, const u64* rest,
                            u32 count) {
    // The single constant buffer data entries the run stands for, as GpuSplit::Execute runs a
    // row of them (CallConstBufferWords).
    constexpr u32 cb_data = MAXWELL3D_REG_INDEX(const_buffer.buffer);
    ConsumeSink(system);
    if (VideoCommon::Nx2Profile::Enabled()) [[unlikely]] {
        VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::CbWriteWords, count);
    }
    // A lone word too: CallConstBufferWords runs it as CallMethod would (shadow RAM, dirty
    // flags, one cached write, the offset), and falls back to CallMethod at a buffer's end.
    std::array<u32, 4> methods;
    std::array<u32, 4> words{};
    count = std::clamp(count, u32{1}, static_cast<u32>(methods.size()));
    // GpuSplit::CbWrite copied words 1-3 into the u64s after the head: 2 or 3 words fill the
    // first, 4 words reach into the second.
    words[0] = first;
    if (count > 3) {
        std::memcpy(&words[1], rest, 3 * sizeof(u32));
    } else if (count > 1) {
        std::memcpy(&words[1], rest, 2 * sizeof(u32));
    }
    for (u32 i = 0; i < count; ++i) {
        methods[i] = cb_data + ((nibbles >> (i * 4)) & 15);
    }
    CallConstBufferWords(system, methods.data(), words.data(), count);
}

void Maxwell3D::BackCbBind(Core::System& system, u32 stage, u32 argument) {
    constexpr u32 bind_first = MAXWELL3D_REG_INDEX(bind_groups[0].raw_config);
    constexpr u32 bind_stride =
        MAXWELL3D_REG_INDEX(bind_groups[1].raw_config) - MAXWELL3D_REG_INDEX(bind_groups[0].raw_config);
    const u32 method = bind_first + stage * bind_stride;
    ConsumeSink(system);
    if (!fast_maxwell3d || executing_macro != 0 || VideoCommon::Nx2Perf::Verify()) {
        CallMethod(system, method, argument, true);
        return;
    }
    // CallMethodFast for a bind group: shadow RAM, the register and its dirty flags, the bind.
    ProcessDirtyRegisters(method, ProcessShadowRam(method, argument));
    ProcessCBBind(stage);
}

struct Maxwell3D::OotContext {
    Maxwell3D& engine;
    VideoCommon::GpuSplit& split;
    FrontState& fs;
    u32 macro_method;

    [[nodiscard]] u32 Count() const {
        return static_cast<u32>(fs.parameters.size());
    }
    [[nodiscard]] u32 Param(u32 index) const {
        return fs.parameters[index];
    }
    [[nodiscard]] u32* Dram() {
        return engine.native_data_ram.data();
    }
    [[nodiscard]] u32 Read(u32 method) const {
        if (fs.list_draw_stale != 0 && (method == Nx2ListDraw::VertexFirst ||
                                        method == Nx2ListDraw::VertexCount)) [[unlikely]] {
            // List guard (#1336 port): as the JIT's read in FrontRunMacro.
            if (const int guard = Nx2ListDispatch::GuardMode(); guard != 0) {
                Nx2ListDispatch::Count(Nx2ListDispatch::g_guard.stale_reads);
                if (guard == 2) {
                    engine.FrontResolveListDraw(split, fs);
                }
            }
        }
        return fs.regs[method];
    }
    [[nodiscard]] bool OcclusionHle() const {
        // Only with the query fast path's front results: it then decides the condition from the
        // same words the macro would read (FrontQueryFast, render_enable).
        return OcclusionHleEnabled() && VideoCommon::Nx2QueryFast::FakeResults();
    }
    void FoldFirstInstance() {
        if (FoldInstanceEnabled()) {
            split.Defer([maxwell3d = &engine] { maxwell3d->draw_manager.FoldFirstInstance(); });
        }
    }
    void Emit(u32 method, u32 value) {
        // As the JIT's emit in FrontRunMacro.
        if (method >= MacroRegistersStart) [[unlikely]] {
            Fail("method past the macro range");
            return;
        }
        if (fs.oot_log_call) [[unlikely]] {
            fs.oot_writes.push_back(method);
            fs.oot_writes.push_back(value);
        }
        engine.FrontWrite(split, fs, method, value, true, true);
        if (method == 0x157 || method == 0x158) [[unlikely]] {
            engine.FrontDmaRead(split, fs, macro_method, method, value);
        }
    }
    void EmitRepeated(const std::array<u32, 2>* writes, u32 count, u32 times) {
        for (u32 time = 0; time < times; ++time) {
            // From the third time on FrontSend only counts them when the batch holds them.
            if (time >= 2 &&
                engine.FrontBatchRepeats(split, fs, writes, count, times - time)) {
                return;
            }
            for (u32 i = 0; i < count; ++i) {
                Emit(writes[i][0], writes[i][1]);
            }
        }
    }
    void Fail(const char* error) {
        engine.GetNativeMacroEngine().native_execution_failed = true;
        LOG_ERROR(HW_GPU, "Native MME (split front, OoT macro {:#x}) failed: {}", macro_method,
                  error);
    }
};

bool Maxwell3D::FrontRunOotMacro(VideoCommon::GpuSplit& split, FrontState& fs, u32 method,
                                 OotMacros::Id id) {
    static_assert(sizeof(native_data_ram) == OotMacros::DataRamWords * sizeof(u32));
    OotContext context{*this, split, fs, method};
    return OotMacros::Run(id, context);
}

bool Maxwell3D::FrontBatchRepeats(VideoCommon::GpuSplit& split, FrontState& fs,
                                  const std::array<u32, 2>* writes, u32 count, u32 more) {
    // As FrontSend for each repeat: draw.begin (Subsequent) and the writes after it are
    // captured, draw.end finds the capture equal to the batch and counts one more. The registers,
    // shadow RAM, draw_pending and the ring stay as the first repeats left them.
    if (!fs.draw_batch || fs.oot_log_call || fs.capture_size != 0 || fs.delta_count != 0 ||
        fs.cb_count != 0 || fs.batch_size != count || count > FrontState::MaxBatchWrites) {
        return false;
    }
    for (u32 i = 0; i < count; ++i) {
        if (fs.batch[i].method != writes[i][0] || fs.batch[i].argument != writes[i][1]) {
            return false;
        }
    }
    fs.batch_repeats += more;
    if (split.NarrowDrains()) {
        // At the same ring position as the repeats before: once is as many times.
        for (u32 i = 0; i < count; ++i) {
            if (execution_mask[writes[i][0]]) {
                FrontNoteWrites(split, fs, writes[i][0], 1);
            }
        }
        split.NoteDraw();
    }
    if (VideoCommon::Nx2Profile::Enabled()) [[unlikely]] {
        VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::MethodWrites,
                                     u64{count} * more);
    }
    return true;
}


// List draws (nx2_list_draw.h, port of Lichii #1336 onto the #717 front): whether the macro at
// `start` does nothing but begin a draw, write its first parameter to vertex_buffer.count and its
// second to vertex_buffer.first, and end the draw.
bool Maxwell3D::FrontIsListDrawMacro(FrontState& fs, const MacroEngine& engine, u32 start) {
    if (fs.list_draw_generation != engine.native_code_generation) {
        fs.list_draw_macros.clear();
        fs.list_draw_generation = engine.native_code_generation;
    }
    const auto [it, inserted] = fs.list_draw_macros.try_emplace(start, false);
    if (!inserted) {
        return it->second;
    }
    static_assert(Nx2ListDraw::DrawBegin == MAXWELL3D_REG_INDEX(draw.begin) &&
                  Nx2ListDraw::DrawEnd == MAXWELL3D_REG_INDEX(draw.end) &&
                  Nx2ListDraw::VertexFirst == MAXWELL3D_REG_INDEX(vertex_buffer.first) &&
                  Nx2ListDraw::VertexCount == MAXWELL3D_REG_INDEX(vertex_buffer.count));
    if (!Nx2ListDraw::IsArrayDrawMacro(std::span<const u32>{engine.native_code}, start)) {
        return false;
    }
    LOG_INFO(HW_GPU, "List draw: the macro at {:#x} is an array draw of (count, first)", start);
    it->second = true;
    return true;
}

bool Maxwell3D::FrontMacroUsesDataRam(FrontState& fs, const MacroEngine& engine, u32 start) {
    if (fs.data_ram_generation != engine.native_code_generation) {
        fs.data_ram_macros.clear();
        fs.data_ram_generation = engine.native_code_generation;
    }
    const auto [it, inserted] = fs.data_ram_macros.try_emplace(start, true);
    if (inserted) {
        it->second = TuringMacroInterpreter::AnalyzeEffects(
                         std::span<const u32>{engine.native_code}, start)
                         .data_ram;
    }
    return it->second;
}

void Maxwell3D::FrontGuardMacro(FrontState& fs, const MacroEngine& engine, u32 start) {
    // List guard (nx2_list_dispatch.h): the data RAM holds the marker of a grid the front did not
    // read (the compute engine shares it). A 3D macro that uses the data RAM gets the real words,
    // whether it runs as the native code or as its OoT replacement (both see the same RAM).
    if (const int guard = Nx2ListDispatch::GuardMode();
        guard != 0 && FrontMacroUsesDataRam(fs, engine, start)) {
        Nx2ListDispatch::Count(Nx2ListDispatch::g_guard.dram_macros);
        if (guard == 2) {
            Nx2ListDispatch::ResolveGrid(grid_marker, native_data_ram);
        }
    }
}

void Maxwell3D::FrontNoteListDrawWrite(VideoCommon::GpuSplit& split, FrontState& fs, u32 method) {
    switch (method) {
    case MAXWELL3D_REG_INDEX(vertex_buffer.count):
        fs.list_draw_stale &= ~1U;
        break;
    case MAXWELL3D_REG_INDEX(vertex_buffer.first):
        fs.list_draw_stale &= ~2U;
        break;
    // The small index draws (index_buffer32_first and the like) draw at once, from their own
    // words: the draw.end after them is still an array draw.
    case MAXWELL3D_REG_INDEX(index_buffer.count):
    case MAXWELL3D_REG_INDEX(draw_inline_index):
    case MAXWELL3D_REG_INDEX(inline_index_2x16.even):
    case MAXWELL3D_REG_INDEX(inline_index_4x8.index0):
        fs.list_draw_indexed = true;
        break;
    case MAXWELL3D_REG_INDEX(draw.end):
        if (fs.in_list_draw) {
            break;
        }
        // An array draw takes its first vertex and its count from the two registers.
        if (!fs.list_draw_indexed) {
            if (const int guard = Nx2ListDispatch::GuardMode(); guard != 0) {
                Nx2ListDispatch::Count(Nx2ListDispatch::g_guard.stale_draws);
                if (guard == 2) {
                    FrontResolveListDraw(split, fs);
                }
            }
        }
        fs.list_draw_indexed = false;
        break;
    default:
        break;
    }
}

void Maxwell3D::FrontResolveListDraw(VideoCommon::GpuSplit& split, FrontState& fs) {
    const u32 stale = std::exchange(fs.list_draw_stale, 0);
    std::array<u32, 2> words{};
    if (!fs.list_draw_list.Read(words)) {
        Nx2ListDispatch::Count(Nx2ListDispatch::g_guard.failed);
        return;
    }
    // As the macro would have written them: the first vertex, then the count.
    if ((stale & 2) != 0) {
        FrontWrite(split, fs, MAXWELL3D_REG_INDEX(vertex_buffer.first), words[1], true, true);
    }
    if ((stale & 1) != 0) {
        FrontWrite(split, fs, MAXWELL3D_REG_INDEX(vertex_buffer.count), words[0], true, true);
    }
    Nx2ListDispatch::Count(Nx2ListDispatch::g_guard.draw_resolves);
}

bool Maxwell3D::FrontTakeListDraw(Core::System& system, VideoCommon::GpuSplit& split, u32 method,
                                  GPUVAddr address, const Nx2ListDispatch::ListReader& reader) {
    auto& fs = Front(system, split);
    // The registers must take the words themselves (no shadow RAM replay), and the two words
    // must be the whole call.
    if (fs.executing_macro != 0 || !fs.parameters.empty() ||
        fs.control == Regs::ShadowRamControl::Replay || rasterizer == nullptr ||
        !rasterizer->Nx2ListDrawSupported()) {
        return false;
    }
    const auto& engine = GetNativeMacroEngine();
    const auto positions = GetNativeMacroPositions();
    if (!engine.native_turing || positions.empty()) {
        return false;
    }
    const u32 start = positions[((method - MacroRegistersStart) >> 1) & 0x7f];
    if (!FrontIsListDrawMacro(fs, engine, start)) {
        return false;
    }
    VideoCommon::Nx2Profile::CountMethod(method, false);
    // As the writes of the macro, in its order. vertex_buffer.first and count keep the values of
    // the last draw that was not one of these, here and on the back: only this draw reads the
    // words.
    fs.in_list_draw = true;
    FrontWrite(split, fs, MAXWELL3D_REG_INDEX(draw.begin), 0, true, true);
    // Writes no guest memory itself (the draw.end after it notes the draw's writes).
    split.Defer([this, address, keep_id = reader.keep_id] {
        draw_manager.draw_state.list_draw_address = address;
        draw_manager.draw_state.list_keep_id = keep_id;
    });
    FrontWrite(split, fs, MAXWELL3D_REG_INDEX(draw.end), 0, true, true);
    fs.in_list_draw = false;
    // List guard (nx2_list_dispatch.h): the two registers keep their old values from here on.
    fs.list_draw_stale = 3;
    fs.list_draw_list = reader;
    fs.list_draw_indexed = false;
    if (fs.draw_pending) {
        fs.draw_pending = false;
        split.Defer([this, system_ptr = &system] { FinishFrontMacro(*system_ptr); },
                    VideoCommon::GpuSplit::FastDrawNeutral);
        split.NoteDraw();
    }
    return true;
}

} // namespace Tegra::Engines
