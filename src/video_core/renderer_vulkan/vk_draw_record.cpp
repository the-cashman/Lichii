// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <bit>
#include <string>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include "common/logging.h"
#include "common/settings.h"
#include "core/memory.h"
#include "video_core/dirty_flags.h"
#include "video_core/memory_manager.h"
#include "video_core/nx2_fast_draw.h"
#include "video_core/nx2_profile.h"
#include "video_core/renderer_vulkan/pipeline_helper.h"
#include "video_core/renderer_vulkan/vk_draw_record.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/shader_handle_program.h"
#include "video_core/textures/texture.h"
#include "video_core/texture_cache/texture_cache.h"
#include "video_core/vulkan_common/vulkan_device.h"

namespace Vulkan {

namespace {

namespace Record = VideoCommon::Nx2DrawRecord;
using Record::Mix;
using Record::MixBytes;

constexpr size_t SampleMask = 15; ///< time every 16th draw

u64 PointerBits(const void* pointer) {
    return static_cast<u64>(reinterpret_cast<uintptr_t>(pointer));
}

constexpr std::array<const char*, 13> stale_names{
    "image event",    "TIC/TSC data",      "rescale",       "buffer topology", "memory map",
    "uniform stream/classic flip", "HOLE handle", "HOLE render targets", "HOLE geometry",
    "HOLE uniform",   "HOLE storage",      "HOLE texture buffers", "HOLE payload",
};
constexpr std::array<const char*, 6> class_names{"mono",         "gpl",  "computed-mono",
                                                 "computed-gpl", "mesh", "writes-images"};

} // Anonymous namespace

DrawRecorder::DrawRecorder(const Device& device_, BufferCache& buffer_cache_,
                           TextureCache& texture_cache_, GuestDescriptorQueue& descriptor_queue_)
    : device{device_}, buffer_cache{buffer_cache_}, texture_cache{texture_cache_},
      descriptor_queue{descriptor_queue_} {
    report_frames = (std::max)(Record::EnvNumber("EDEN_NX2_DRAW_RECORD_REPORT", 60), u64{1});
    holes_left = Record::EnvNumber("EDEN_NX2_DRAW_RECORD_HOLES", 24);
    log_frames = Record::EnvNumber("EDEN_NX2_DRAW_RECORD_FRAMES", 0) != 0;
    storage_in_key = Record::EnvNumber("EDEN_NX2_DRAW_RECORD_STORAGE_KEY", 0) != 0;
    handles_in_key = Record::EnvNumber("EDEN_NX2_DRAW_RECORD_HANDLE_KEY", 0) != 0;
    mode = Record::GetMode();
    test_every = Record::EnvNumber("EDEN_NX2_DRAW_RECORD_TEST", 0);
    test_break = Record::EnvNumber("EDEN_NX2_DRAW_RECORD_TEST_BREAK", 0) != 0;
    stats = Record::Stats();
    LOG_INFO(Render_Vulkan, "Draw records: mode {}, report every {} frames",
             mode == Record::Mode::Dry ? "dry" : mode == Record::Mode::Verify ? "verify" : "on",
             report_frames);
    table.reserve(1 << 18);
    ResetWindow();
}

DrawRecorder::~DrawRecorder() = default;

void DrawRecorder::ResetWindow() {
    window = Window{};
    window.start_ticks = VideoCommon::Nx2Profile::Ticks();
    window.start_ns = VideoCommon::Nx2Profile::Now();
}

DrawRecorder::Plan& DrawRecorder::PlanFor(GraphicsPipeline& pipeline) {
    // As the hit path would keep it: on the pipeline object (a new object at a reused address
    // starts without one and gets a new generation).
    if (pipeline.record_plan != nullptr) [[likely]] {
        return *static_cast<Plan*>(pipeline.record_plan);
    }
    // A pipeline object freed and made again at this address gets a fresh plan; the old one
    // may still be pointed to by nothing else.
    std::unique_ptr<Plan>& slot = plans[&pipeline];
    slot = std::make_unique<Plan>();
    BuildPlan(pipeline, *slot);
    pipeline.record_plan = slot.get();
    return *slot;
}

void DrawRecorder::BuildPlan(GraphicsPipeline& pipeline, Plan& out) {
    out = Plan{};
    out.shader_hashes = pipeline.ShaderHashes();
    out.generation = ++plan_generation;
    out.gpl = pipeline.IsLibraryPipeline();
    bool computed = false;
    for (size_t stage = 0; stage < NUM_STAGES; ++stage) {
        const Shader::Info& info = pipeline.StageInfo(stage);
        u32 slots = info.constant_buffer_mask;
        for (const auto& desc : info.storage_buffers_descriptors) {
            // EDEN_NX2_DRAW_RECORD_STORAGE_KEY=1: storage pointer words in the key. Default: not
            // (per-object data moves per frame); storage bindings are resolved per draw instead.
            WordOp& op = out.params.emplace_back();
            op = WordOp{
                .kind = WordOp::Kind::Storage,
                .stage = static_cast<u8>(stage),
                .cbuf = static_cast<u8>(desc.cbuf_index),
                .written = desc.is_written,
                .offset = desc.cbuf_offset,
                .indirect = desc.indirect_offset,
            };
            slots |= 1U << desc.cbuf_index;
        }
        // As ConfigureImpl's read_handle, element by element, in its order.
        const auto add_handles = [&](const auto& desc, bool storage_pointer_check) {
            // One op per view element (ConfigureImpl's views), in its order.
            constexpr bool sampled =
                std::is_same_v<std::decay_t<decltype(desc)>, Shader::TextureDescriptor>;
            if (desc.handle_program != ~0U) {
                computed = true;
                // Since #1485 a program can compute a handle per element (HandleOp::Index, a
                // table of shadow maps the shader indexes): each element is evaluated, as
                // ConfigureImpl does.
                for (u32 index = 0; index < desc.count; ++index) {
                    out.words.push_back({
                        .kind = WordOp::Kind::Computed,
                        .stage = static_cast<u8>(stage),
                        .sampled = sampled,
                        .program = desc.handle_program,
                        .element = index,
                    });
                }
                return;
            }
            for (u32 index = 0; index < desc.count; ++index) {
                if constexpr (sampled) {
                    if (storage_pointer_check &&
                        VideoCommon::HandleIsStoragePointer(info, desc, index)) {
                        out.words.push_back({.kind = WordOp::Kind::Null,
                                             .stage = static_cast<u8>(stage),
                                             .sampled = true});
                        continue;
                    }
                }
                const u32 offset = desc.cbuf_offset + (index << desc.size_shift);
                WordOp op{
                    .kind = WordOp::Kind::Handle,
                    .stage = static_cast<u8>(stage),
                    .cbuf = static_cast<u8>(desc.cbuf_index),
                    .sampled = sampled,
                    .offset = offset,
                };
                slots |= 1U << desc.cbuf_index;
                if constexpr (requires { desc.has_secondary; }) {
                    if (desc.has_secondary) {
                        op.kind = WordOp::Kind::HandlePair;
                        op.shift = static_cast<u8>(desc.shift_left);
                        op.cbuf2 = static_cast<u8>(desc.secondary_cbuf_index);
                        op.offset2 = desc.secondary_cbuf_offset;
                        op.shift2 = static_cast<u8>(desc.secondary_shift_left);
                        slots |= 1U << desc.secondary_cbuf_index;
                    }
                }
                out.words.push_back(op);
            }
        };
        out.texture_buffers |= !info.texture_buffer_descriptors.empty() ||
                               !info.image_buffer_descriptors.empty();
        for (const auto& desc : info.texture_buffer_descriptors) {
            add_handles(desc, false);
        }
        for (const auto& desc : info.image_buffer_descriptors) {
            add_handles(desc, false);
        }
        for (const auto& desc : info.texture_descriptors) {
            add_handles(desc, true);
        }
        for (const auto& desc : info.image_descriptors) {
            add_handles(desc, false);
        }
        out.cbuf_slots[stage] = slots;
        // The descriptor payload's entries, in DescriptorLayoutBuilder::Add's order.
        const auto add_kinds = [&](EntryKind kind, const auto& descriptors) {
            for (const auto& desc : descriptors) {
                out.kinds.insert(out.kinds.end(), desc.count, kind);
            }
        };
        add_kinds(EntryKind::Uniform, info.constant_buffer_descriptors);
        add_kinds(EntryKind::Storage, info.storage_buffers_descriptors);
        add_kinds(EntryKind::Texel, info.texture_buffer_descriptors);
        add_kinds(EntryKind::ImageTexel, info.image_buffer_descriptors);
        add_kinds(EntryKind::Sampled, info.texture_descriptors);
        add_kinds(EntryKind::Image, info.image_descriptors);
        if (stage == 0) {
            out.vertex_entries = static_cast<u32>(out.kinds.size());
        }
    }
    out.layout_hash = MixBytes(Mix(31, out.vertex_entries), out.kinds.data(), out.kinds.size());
    s16 sampler_count = 0;
    for (const WordOp& op : out.words) {
        out.sampler_of.push_back(op.sampled ? sampler_count++ : s16{-1});
    }
    // Kept image descriptors: per stage, the texture elements then the image elements
    // (PushImageDescriptors' order, KeepImageDescriptors per stage in stage order).
    s16 kept = 0;
    for (size_t stage = 0; stage < NUM_STAGES; ++stage) {
        const Shader::Info& info = pipeline.StageInfo(stage);
        const auto add = [&](const auto& descriptors, bool with_entry, bool image) {
            for (const auto& desc : descriptors) {
                for (u32 index = 0; index < desc.count; ++index) {
                    Plan::ElementDescriptor element{.entry = with_entry ? kept++ : s16{-1},
                                                    .image = image};
                    if constexpr (requires { desc.type; }) {
                        element.type = desc.type;
                    }
                    if constexpr (requires { desc.is_depth; }) {
                        element.is_depth = desc.is_depth;
                    }
                    if constexpr (requires { desc.handle_program; }) {
                        element.read_from_memory = desc.handle_program != ~0U;
                    }
                    if constexpr (requires { desc.format; }) {
                        element.format = desc.format;
                    }
                    if constexpr (requires { desc.is_byte_addressed; }) {
                        element.byte_addressed = desc.is_byte_addressed;
                    }
                    out.element_descriptors.push_back(element);
                }
            }
        };
        add(info.texture_buffer_descriptors, false, false);
        add(info.image_buffer_descriptors, false, false);
        add(info.texture_descriptors, true, false);
        add(info.image_descriptors, true, true);
    }
    if (out.element_descriptors.size() != out.words.size()) {
        out.element_descriptors.clear(); // no patching for this pipeline
    }
    out.klass = pipeline.IsMeshPipeline() ? PipeClass::Mesh
                : pipeline.WritesImages() ? PipeClass::WritesImages
                : computed                ? (out.gpl ? PipeClass::ComputedGpl : PipeClass::ComputedMono)
                : out.gpl                 ? PipeClass::Gpl
                                          : PipeClass::Mono;
}

DrawRecorder::ChannelGroups& DrawRecorder::Channel(const Tegra::Engines::Maxwell3D& maxwell3d) {
    for (ChannelGroups& groups : channels) {
        if (groups.maxwell3d == &maxwell3d) {
            return groups;
        }
    }
    ChannelGroups& groups = channels[channel_cursor++ % channels.size()];
    groups = ChannelGroups{.maxwell3d = &maxwell3d};
    return groups;
}

void DrawRecorder::SnapshotRegs(const Tegra::Engines::Maxwell3D& maxwell3d, std::vector<u8>& out) {
    const auto& regs = maxwell3d.regs;
    out.clear();
    const auto add = [&](const auto& field) {
        const auto* const bytes = reinterpret_cast<const u8*>(&field);
        out.insert(out.end(), bytes, bytes + sizeof(field));
    };
    add(regs.rt);
    add(regs.rt_control);
    add(regs.zeta);
    add(regs.zeta_enable);
    add(regs.zeta_size);
    add(regs.surface_clip);
    add(regs.anti_alias_samples_mode);
    add(regs.tex_header);
    add(regs.tex_sampler);
    add(regs.sampler_binding);
    add(regs.transform_feedback_enabled);
}

u64 DrawRecorder::HashRegs(const Tegra::Engines::Maxwell3D& maxwell3d) const {
    const auto& regs = maxwell3d.regs;
    u64 hash = 7;
    const auto add = [&](const auto& field) { hash = MixBytes(hash, &field, sizeof(field)); };
    // Render targets (TextureCache::UpdateRenderTargets, render area push constant).
    add(regs.rt);
    add(regs.rt_control);
    add(regs.zeta);
    add(regs.zeta_enable);
    add(regs.zeta_size);
    add(regs.surface_clip);
    add(regs.anti_alias_samples_mode);
    // Descriptor pools and how handles name their entries (SynchronizeDescriptors, read_handle).
    add(regs.tex_header);
    add(regs.tex_sampler);
    add(regs.sampler_binding);
    add(regs.transform_feedback_enabled);
    return hash;
}

u64 DrawRecorder::HashStreams(Tegra::Engines::Maxwell3D& maxwell3d) const {
    const auto& regs = maxwell3d.regs;
    const bool native = maxwell3d.GetNativeMacroEngine().native_turing;
    u64 hash = 9;
    for (u32 index = 0; index < Tegra::Engines::Maxwell3D::Regs::NumVertexArrays; ++index) {
        const auto& stream = regs.vertex_streams[index];
        if (stream.enable == 0) {
            continue;
        }
        hash = Mix(hash, index);
        hash = MixBytes(hash, &stream, sizeof(stream));
        hash = MixBytes(hash, &regs.vertex_stream_limits[index],
                        sizeof(regs.vertex_stream_limits[index]));
        if (native) {
            hash = Mix(hash, regs.NativeVertexStreamSize(index));
        }
    }
    return hash;
}

const u8* DrawRecorder::PagePointer(Tegra::MemoryManager& gpu_memory, GPUVAddr address) {
    constexpr u64 page_mask = Core::Memory::LICHII_PAGEMASK;
    const GPUVAddr page = address & ~page_mask;
    PageSlot& slot = pages[(page >> Core::Memory::LICHII_PAGEBITS) & (pages.size() - 1)];
    const u64 serial = gpu_memory.MapSerial();
    if (slot.page != page || slot.memory != &gpu_memory || slot.map_serial != serial) {
        slot = PageSlot{&gpu_memory, serial, page, std::as_const(gpu_memory).GetPointer(page)};
    }
    return slot.host != nullptr ? slot.host + (address & page_mask) : nullptr;
}

template <typename T>
T DrawRecorder::ReadGuest(Tegra::MemoryManager& gpu_memory, GPUVAddr address) {
    constexpr u64 page_mask = Core::Memory::LICHII_PAGEMASK;
    if ((address & page_mask) + sizeof(T) <= Core::Memory::LICHII_PAGESIZE) {
        const GPUVAddr page = address & ~page_mask;
        PageSlot& slot = pages[(page >> Core::Memory::LICHII_PAGEBITS) & (pages.size() - 1)];
        const u64 serial = gpu_memory.MapSerial();
        if (slot.page != page || slot.memory != &gpu_memory || slot.map_serial != serial) {
            slot = PageSlot{&gpu_memory, serial, page, std::as_const(gpu_memory).GetPointer(page)};
        }
        if (slot.host != nullptr) {
            T value;
            std::memcpy(&value, slot.host + (address & page_mask), sizeof(T));
            return value;
        }
    }
    return gpu_memory.Read<T>(address);
}

/// EvaluateTextureHandle's memory: reads through the recorder's page pointers.
struct RecordMemory {
    DrawRecorder& recorder;
    Tegra::MemoryManager& memory;
    template <typename T>
    T Read(GPUVAddr address) const {
        return recorder.ReadGuest<T>(memory, address);
    }
};

u64 DrawRecorder::HashData(const Plan& draw_plan, GraphicsPipeline& pipeline,
                           Tegra::Engines::Maxwell3D& maxwell3d,
                           Tegra::MemoryManager& gpu_memory) {
    const bool native = maxwell3d.GetNativeMacroEngine().native_turing;
    const auto& stages = maxwell3d.state.shader_stages;
    cur_values.clear();
    for (const WordOp& op : draw_plan.words) {
        const auto& cbufs = stages[op.stage].const_buffers;
        switch (op.kind) {
        case WordOp::Kind::Handle:
            cur_values.push_back(ReadGuest<u32>(gpu_memory, cbufs[op.cbuf].address + op.offset));
            break;
        case WordOp::Kind::HandlePair: {
            const u32 lhs = ReadGuest<u32>(gpu_memory, cbufs[op.cbuf].address + op.offset)
                            << op.shift;
            const u32 rhs = ReadGuest<u32>(gpu_memory, cbufs[op.cbuf2].address + op.offset2)
                            << op.shift2;
            cur_values.push_back(lhs | rhs);
            break;
        }
        case WordOp::Kind::Null:
            cur_values.push_back(~0ULL);
            break;
        case WordOp::Kind::Computed: {
            if (op.repeat) {
                cur_values.push_back(cur_values.back());
                break;
            }
            RecordMemory memory{*this, gpu_memory};
            cur_values.push_back(VideoCommon::EvaluateTextureHandle(
                pipeline.StageInfo(op.stage), op.program, memory,
                [&](u32 cbuf) -> GPUVAddr {
                    return cbuf < cbufs.size() && cbufs[cbuf].enabled ? cbufs[cbuf].address
                                                                      : GPUVAddr{};
                },
                device.GetStorageBufferAlignment(), op.element));
            break;
        }
        case WordOp::Kind::Storage: {
            // BufferCache::BindGraphicsStorageBuffer and StorageBufferBinding's reads.
            GPUVAddr address = cbufs[op.cbuf].address + op.offset;
            u64 value = 41;
            if (op.indirect != ~0U) {
                const u64 holder = ReadGuest<u64>(gpu_memory, address);
                value = Mix(value, holder);
                address = holder + op.indirect;
                if (holder == 0) {
                    cur_values.push_back(value);
                    break;
                }
            }
            value = Mix(value, ReadGuest<u64>(gpu_memory, address));
            if (op.indirect == ~0U && !native) {
                value = Mix(value, ReadGuest<u64>(gpu_memory, address + 8));
            }
            cur_values.push_back(value);
            break;
        }
        }
    }
    return MixBytes(5, cur_values.data(), cur_values.size() * sizeof(u64));
}

void DrawRecorder::BeginDraw(GraphicsPipeline& pipeline, Tegra::Engines::Maxwell3D& maxwell3d,
                             Tegra::MemoryManager& gpu_memory, bool is_indexed, bool is_indirect,
                             bool repacked) {
    namespace Profile = VideoCommon::Nx2Profile;
    namespace FastDraw = VideoCommon::Nx2FastDraw;
    if (begun) {
        ++window.failed; // the last draw never reached EndDraw
    }
    if (const u32 serial = Common::Nx2Ab::PhaseSerial(); serial != phase_serial) [[unlikely]] {
        // An A/B flip (or the first draw): the mode of this arm. The table is dropped: its
        // records were not kept current while another arm ran.
        phase_serial = serial;
        mode = Record::CurrentMode();
        active = mode != Record::Mode::Off;
        level0_entry = nullptr;
        ++owner_epoch;
        table.clear();
        frames_since_flip = 0;
        ++window.flips;
    }
    if (!active) {
        begun = false; // A/B off phase: today's paths, nothing recorded
        return;
    }
    begun = true;
    sample = (++draw_counter & SampleMask) == 0;
    const u64 t0 = sample ? Profile::Ticks() : 0;
    Plan& draw_plan = PlanFor(pipeline);
    plan = &draw_plan;
    ChannelGroups& groups = Channel(maxwell3d);
    const auto& regs = maxwell3d.regs;
    auto& flags = maxwell3d.dirty.flags;
    // Render target registers: hashed again when Dirty::RenderTargets is set (the render target
    // update and clears consume it; a change it misses shows as a stale "render targets" hit).
    if (!groups.targets_valid || flags[VideoCommon::Dirty::RenderTargets]) {
        u64 hash = 7;
        const auto add = [&](const auto& field) { hash = MixBytes(hash, &field, sizeof(field)); };
        add(regs.rt);
        add(regs.rt_control);
        add(regs.zeta);
        add(regs.zeta_enable);
        add(regs.zeta_size);
        add(regs.surface_clip);
        groups.targets = hash;
        groups.targets_valid = true;
        ++window.regs_rehash;
    }
    ++window.regs_compared;
    // Vertex stream structure (which streams, stride/enable word): hashed again when
    // Dirty::VertexBuffers is set. Their addresses and limits are per-draw inputs (params).
    if (!groups.streams_valid || flags[VideoCommon::Dirty::VertexBuffers]) {
        u64 hash = 9;
        for (u32 index = 0; index < Tegra::Engines::Maxwell3D::Regs::NumVertexArrays; ++index) {
            const auto& stream = regs.vertex_streams[index];
            if (stream.enable != 0) {
                hash = Mix(hash, (u64{index} << 32) | stream.stride.Value());
            }
        }
        groups.streams = hash;
        groups.streams_valid = true;
        ++window.streams_rehash;
    }
    cur_regs = groups.targets;
    cur_maxwell3d = &maxwell3d;
    const auto& draw_state = maxwell3d.draw_manager.draw_state;
    cur_indirect = is_indirect;
    cur_inline = !draw_state.inline_index_draw_indexes.empty();
    cur_repacked = repacked;
    u64 kind = Mix(Mix(1, PointerBits(&maxwell3d)), PointerBits(&gpu_memory));
    kind = Mix(kind, (is_indexed ? 1U : 0U) | (is_indirect ? 2U : 0U) | (cur_inline ? 4U : 0U) |
                         (repacked ? 8U : 0U) | (static_cast<u64>(draw_state.topology) << 8));
    // Pools, handle split mode, samples, transform feedback: a few words, read per draw.
    u64 pools = 2;
    pools = MixBytes(pools, &regs.tex_header, sizeof(regs.tex_header));
    pools = MixBytes(pools, &regs.tex_sampler, sizeof(regs.tex_sampler));
    pools = Mix(pools, (u64{static_cast<u32>(regs.sampler_binding)} << 32) |
                           static_cast<u32>(regs.anti_alias_samples_mode));
    pools = Mix(pools, regs.transform_feedback_enabled);
    const u64 index_format = is_indexed ? static_cast<u64>(draw_state.index_buffer.format) + 1 : 0;
    u64 cbuf_sizes = 4;
    for (size_t stage = 0; stage < NUM_STAGES; ++stage) {
        const auto& cbufs = maxwell3d.state.shader_stages[stage].const_buffers;
        for (u32 mask = draw_plan.cbuf_slots[stage]; mask != 0; mask &= mask - 1) {
            // Not the address: streamed uniform data is placed per draw (rings), the uniform
            // binding is resolved per draw, and the words read through these buffers (handles,
            // storage pointers) are in the key by value or are per-draw inputs.
            const auto& cbuf = cbufs[std::countr_zero(mask)];
            cbuf_sizes = Mix(cbuf_sizes, (u64{cbuf.size} << 1) | (cbuf.enabled ? 1U : 0U));
        }
    }
    const u64 t1 = sample ? Profile::Ticks() : 0;
    // The texture handles: read here only when the key or the measurements need them; the hit
    // and the record read them when they do (EnsureHandles).
    values_read = false;
    u64 data = 0;
    if (stats || handles_in_key) {
        data = HashData(draw_plan, pipeline, maxwell3d, gpu_memory);
        values_read = true;
    }
    cur_pipeline = &pipeline;
    cur_memory = &gpu_memory;
    // Per-draw inputs (a record keeps the binding they gave and re-resolves only the moved ones).
    cur_params.clear();
    if (stats) {
    for (u32 index = 0; index < Tegra::Engines::Maxwell3D::Regs::NumVertexArrays; ++index) {
        const auto& stream = regs.vertex_streams[index];
        if (stream.enable != 0) {
            const u64 extent = maxwell3d.GetNativeMacroEngine().native_turing
                                   ? regs.NativeVertexStreamSize(index)
                                   : regs.vertex_stream_limits[index].Address();
            cur_params.push_back(Mix(stream.Address(), extent));
        }
    }
    if (is_indexed) {
        const auto& index = draw_state.index_buffer;
        cur_params.push_back(Mix(index.StartAddress(), index.EndAddress()));
    }
    {
        const bool native = maxwell3d.GetNativeMacroEngine().native_turing;
        const auto& stages = maxwell3d.state.shader_stages;
        for (const WordOp& op : draw_plan.params) {
            GPUVAddr address = stages[op.stage].const_buffers[op.cbuf].address + op.offset;
            u64 value = 41;
            if (op.indirect != ~0U) {
                const u64 holder = ReadGuest<u64>(gpu_memory, address);
                value = Mix(value, holder);
                address = holder + op.indirect;
                if (holder == 0) {
                    cur_params.push_back(value);
                    continue;
                }
            }
            value = Mix(value, ReadGuest<u64>(gpu_memory, address));
            if (op.indirect == ~0U && !native) {
                value = Mix(value, ReadGuest<u64>(gpu_memory, address + 8));
            }
            cur_params.push_back(value);
        }
    }
    } // stats
    const u64 t2 = sample ? Profile::Ticks() : 0;
    cur_groups = {kind, groups.targets, pools, groups.streams, index_format, cbuf_sizes, data};
    // Render targets are checked per draw (the render target update stays on the hit path) and
    // texture handles per binding (EDEN_NX2_DRAW_RECORD_HANDLE_KEY=1 keys them instead): only
    // the structure is key.
    std::array<u64, 7> keyed = cur_groups;
    keyed[1] = 0;
    if (storage_in_key) {
        keyed[5] = MixBytes(keyed[5], cur_params.data(), cur_params.size() * sizeof(u64));
    }
    if (!handles_in_key) {
        keyed[6] = 0;
    }
    const u64 pipe = Mix(PointerBits(&pipeline), draw_plan.generation);
    u64 lo = Mix(0x243F6A8885A308D3ULL, pipe);
    u64 hi = Mix(0x13198A2E03707344ULL, pipe);
    for (size_t group = 0; group < keyed.size(); ++group) {
        lo = Mix(lo, keyed[group]);
        hi = Mix(hi, keyed[keyed.size() - 1 - group] ^ 0xA4093822299F31D0ULL);
    }
    const Record::Key draw_key{.lo = lo, .hi = hi};
    // Level 0: the previous draw's key (no table lookup).
    if (level0_entry != nullptr && draw_key == key) {
        ++window.level0;
        entry = level0_entry;
        entry_new = !entry->valid;
    } else {
        auto [it, inserted] = table.try_emplace(draw_key);
        key = draw_key;
        entry = &it->second;
        entry_new = inserted || !it->second.valid;
        level0_entry = entry;
    }
    if (sample) {
        const u64 t3 = Profile::Ticks();
        begin_ticks_regs = t1 - t0;
        begin_ticks_data = t2 - t1;
        begin_ticks_lookup = t3 - t2;
        phase_stamps.fill(0);
        pipeline.record_phase_ticks = phase_stamps.data();
        begin_end = Profile::Ticks();
    }
}

DrawRecorder::Stale DrawRecorder::Classify(const Entry& old, const std::array<u64, NUM_OUT>& out,
                                           const std::vector<u64>& bindings, bool rescaling,
                                           u64 tex_clock, u64 scale, u64 buf_epoch,
                                           u64 map_serial) const {
    const auto differs = [&](Out part) {
        return old.out[part] != out[part] && part != Uniform && (part != Storage || storage_in_key);
    };
    const bool tex_event = old.tex_clock != tex_clock;
    const bool rescaled = old.scale_epoch != scale || old.rescaling != rescaling;
    if (differs(Views)) {
        if (tex_event) {
            return Stale::TexEvent;
        }
        if (rescaled) {
            return Stale::Rescale;
        }
        if (old.bindings.size() != bindings.size() || bindings.empty()) {
            return Stale::HoleHandle;
        }
        // A binding whose handle moved is resolved again (not stale); one with the same handle
        // and another view or sampler means the TIC/TSC entries changed under it.
        if (handles_in_key) {
            const size_t views = static_cast<size_t>(bindings[0]);
            for (size_t index = 1; index < bindings.size(); ++index) {
                if (old.bindings[index] != bindings[index] && index <= views &&
                    (old.bindings[index] >> 32) != (bindings[index] >> 32)) {
                    return Stale::HoleHandle;
                }
            }
        }
        return Stale::TicData;
    }
    if (differs(Targets)) {
        return tex_event ? Stale::TexEvent : rescaled ? Stale::Rescale : Stale::HoleTargets;
    }
    if (differs(Geometry) || differs(UniformStream) || differs(Storage) ||
        differs(TexBuffers)) {
        if (old.buf_epoch != buf_epoch) {
            return Stale::BufferTopology;
        }
        if (old.map_serial != map_serial) {
            return Stale::MemoryMap;
        }
        if (differs(Geometry)) {
            return Stale::HoleGeometry;
        }
        if (differs(Storage)) {
            return Stale::HoleStorage;
        }
        if (differs(TexBuffers)) {
            return tex_event ? Stale::TexEvent : Stale::HoleTexBuffers;
        }
        return Stale::UniformPath;
    }
    if (rescaled) {
        return Stale::Rescale;
    }
    if (tex_event) {
        return Stale::TexEvent;
    }
    if (old.buf_epoch != buf_epoch) {
        return Stale::BufferTopology;
    }
    return Stale::HolePayload;
}

void DrawRecorder::LogHole(Stale stale, const GraphicsPipeline& pipeline, const Entry& old,
                           const std::array<u64, NUM_OUT>& out, const std::vector<u64>& bindings) {
    if (holes_left == 0) {
        return;
    }
    --holes_left;
    u32 mask = 0;
    for (size_t part = 0; part < NUM_OUT; ++part) {
        mask |= old.out[part] != out[part] ? 1U << part : 0U;
    }
    std::string detail;
    if (stale == Stale::HoleHandle) {
        const size_t views = bindings.empty() ? 0 : static_cast<size_t>(bindings[0]);
        for (size_t index = 1; index < (std::min)(old.bindings.size(), bindings.size());
             ++index) {
            if (old.bindings[index] != bindings[index] && detail.size() < 400) {
                detail += fmt::format(" [{}{}] {:#x}->{:#x}", index <= views ? "view " : "sampler ",
                                      index <= views ? index - 1 : index - 1 - views,
                                      old.bindings[index], bindings[index]);
            }
        }
        if (old.bindings.size() != bindings.size()) {
            detail += fmt::format(" sizes {} {}", old.bindings.size(), bindings.size());
        }
    }
    const auto& hashes = pipeline.ShaderHashes();
    LOG_WARNING(Render_Vulkan,
                "Draw record key hole ({}): frame {} vs {:016x} ps {:016x} class {} differing "
                "parts {:#x} (views 1, geometry 2, uniform 4, stream 8, storage 16, texbuf 32, "
                "targets 64, payload 128){}",
                stale_names[static_cast<size_t>(stale)], frame, hashes[1], hashes[5],
                class_names[static_cast<size_t>(plan->klass)], mask, detail);
}

void DrawRecorder::EndDraw(GraphicsPipeline& pipeline, Tegra::MemoryManager& gpu_memory,
                           bool delta, u8 full_reason, u32 break_method, bool record_used) {
    namespace Profile = VideoCommon::Nx2Profile;
    if (!begun) {
        return;
    }
    begun = false;
    const u64 t0 = sample ? Profile::Ticks() : 0;
    Plan& draw_plan = *plan;
    if (sample) {
        pipeline.record_phase_ticks = nullptr;
        if (phase_stamps[6] != 0 && (delta || record_used)) {
            const size_t row = record_used ? 1 : 0;
            ++window.phase_samples[row];
            for (size_t i = 0; i < 6; ++i) {
                window.phases[row][i] += phase_stamps[i + 1] - phase_stamps[i];
            }
        }
    }
    if (!stats) {
        // EDEN_NX2_DRAW_RECORD=1 without the measurements: keep the record current.
        ++window.draws;
        ++frame_draws;
        if (!active) {
            hit_prepared = false;
            return;
        }
        if (mode == Record::Mode::On) {
            const size_t warm = frames_since_flip == 0 ? 0 : 1;
            ++window.warm_draws[warm];
            window.warm_hits[warm] += record_used ? 1 : 0;
        }
        Entry& stored = *entry;
        if (entry_new) {
            ++window.hits[static_cast<size_t>(Hit::Miss)];
            ++window.new_keys;
        } else if (record_used) {
            ++window.hits[static_cast<size_t>(Hit::Previous)];
        }
        // A hit that replayed the record unchanged leaves it as it is; a delta draw keeps the
        // views its chain's full draw captured (stored then, or by a later full draw).
        const bool unchanged_hit = record_used && hit_images_valid && hit_moved == 0 &&
                                   stored.record_tex_clock == texture_cache.BindingClock();
        if (!unchanged_hit && (!delta || !stored.has_record)) {
            StoreRecord(stored, pipeline, gpu_memory, gpu_memory.MapSerial(),
                        texture_cache.BindingClock());
        }
        stored.last_frame = frame;
        stored.valid = true;
        hit_prepared = false;
        if (sample) {
            ++window.samples;
            window.ticks_regs += begin_ticks_regs;
            window.ticks_data += begin_ticks_data;
            window.ticks_lookup += begin_ticks_lookup;
            if (record_used) {
                ++window.record_configure[0];
                window.record_configure[1] += t0 - begin_end;
            } else {
                window.configure_ticks[delta ? 1 : 0][0] += t0 - begin_end;
                ++window.configure_samples[delta ? 1 : 0][0];
            }
        }
        return;
    }
    // The outcome of the resolve that ran.
    std::array<u64, NUM_OUT> out{};
    const std::vector<u64>& bindings = pipeline.record_bindings;
    out[Views] = MixBytes(11, bindings.data(), bindings.size() * sizeof(u64));
    std::array<u64, 5> buffers{};
    buffer_cache.RecordGraphicsBindingsHashes(buffers);
    out[Geometry] = buffers[0];
    out[Uniform] = buffers[1];
    out[UniformStream] = buffers[2];
    out[Storage] = buffers[3];
    out[TexBuffers] = buffers[4];
    const auto& targets = texture_cache.GetRenderTargets();
    const bool rescaling = texture_cache.IsRescaling();
    u64 targets_hash = 13;
    for (const auto id : targets.color_buffer_ids) {
        targets_hash = Mix(targets_hash, id.index);
    }
    targets_hash = Mix(targets_hash, targets.depth_buffer_id.index);
    targets_hash = MixBytes(targets_hash, targets.draw_buffers.data(), targets.draw_buffers.size());
    targets_hash = Mix(targets_hash, (u64{targets.size.width} << 32) | targets.size.height);
    out[Targets] = Mix(targets_hash, (targets.is_rescaled ? 2U : 0U) | (rescaling ? 1U : 0U));
    const DescriptorUpdateEntry* const entries = descriptor_queue.UpdateData();
    const size_t count = descriptor_queue.UpdateCount();
    u64 payload = 17;
    for (size_t index = 0; index < count; ++index) {
        const EntryKind kind =
            index < draw_plan.kinds.size() ? draw_plan.kinds[index] : EntryKind::Other;
        const DescriptorUpdateEntry& item = entries[index];
        switch (kind) {
        case EntryKind::Uniform:
        case EntryKind::Other:
        case EntryKind::Storage:
            break; // per draw: uniforms always, storage through the storage binding hash

        case EntryKind::Texel:
        case EntryKind::ImageTexel:
            payload = Mix(payload, PointerBits(item.texel_buffer));
            break;
        case EntryKind::Sampled:
        case EntryKind::Image:
            payload = Mix(Mix(Mix(payload, PointerBits(item.image.sampler)),
                              PointerBits(item.image.imageView)),
                          item.image.imageLayout);
            break;
        }
    }
    out[Payload] = payload;
    const u64 tex_clock = texture_cache.BindingClock();
    const u64 scale = texture_cache.ScaleEpoch();
    const u64 buf_epoch = buffer_cache.BindingEpoch();
    const u64 map_serial = gpu_memory.MapSerial();
    const u64 gpu_modified = buffer_cache.gpu_modified_serial.load(std::memory_order_relaxed);

    Hit hit = Hit::Miss;
    if (!entry_new) {
        const u64 shift = frame - entry->last_frame;
        const u64 seen = shift >= 64 ? 0 : entry->seen << shift;
        hit = (seen & 2) != 0         ? Hit::Previous
              : (seen & ~u64{3}) != 0 ? Hit::Older
              : (seen & 1) != 0       ? Hit::SameFrame
                                      : Hit::Older;
    }
    const size_t klass = static_cast<size_t>(draw_plan.klass);
    bool correct = false;
    bool stale = false;
    std::vector<u64> storage_now;
    buffer_cache.RecordGraphicsStorageBindings(storage_now);
    u32 moved = 0;
    if (hit != Hit::Miss) {
        // Per-draw parts: uniform bindings always; vertex/index bindings when their inputs
        // (addresses, extents) moved; storage bindings when their pointer words moved.
        bool geometry_moved = false;
        bool storage_inputs_moved = false;
        const size_t geometry_params = cur_params.size() - draw_plan.params.size();
        if (entry->params.size() != cur_params.size()) {
            geometry_moved = storage_inputs_moved = true;
            moved = static_cast<u32>(cur_params.size());
        } else {
            for (size_t index = 0; index < cur_params.size(); ++index) {
                if (entry->params[index] != cur_params[index]) {
                    ++moved;
                    (index < geometry_params ? geometry_moved : storage_inputs_moved) = true;
                }
            }
        }
        // Texture/image bindings: one whose handle moved is resolved again (counted); one whose
        // handle stayed must give the same view. Samplers, texture buffers and the image part
        // of the payload are compared when no handle moved.
        u32 handles_moved = 0;
        bool views_stale = false;
        const std::vector<u64>& old_bindings = entry->bindings;
        if (old_bindings.size() != bindings.size() || bindings.empty()) {
            views_stale = true;
        } else {
            const size_t views = static_cast<size_t>(bindings[0]);
            const std::vector<u64>& older = entry->older_bindings;
            window.bindings_compared += views;
            for (size_t index = 1; index <= views; ++index) {
                if ((old_bindings[index] >> 32) != (bindings[index] >> 32)) {
                    ++handles_moved;
                    const bool same_view =
                        (old_bindings[index] & 0xffffffffULL) == (bindings[index] & 0xffffffffULL);
                    window.moved_same_view += same_view ? 1 : 0;
                    window.moved_new_view += same_view ? 0 : 1;
                    window.moved_cycle2 += older.size() == bindings.size() &&
                                                   (older[index] >> 32) == (bindings[index] >> 32)
                                               ? 1
                                               : 0;
                } else if (old_bindings[index] != bindings[index]) {
                    views_stale = true;
                }
            }
            if (handles_moved == 0) {
                views_stale |= old_bindings != bindings;
            }
        }
        moved += handles_moved;
        window.handle_moved_hits += handles_moved != 0 ? 1 : 0;
        const bool uniform_moved = entry->out[Uniform] != out[Uniform];
        const bool storage_moved = entry->out[Storage] != out[Storage];
        std::array<u64, NUM_OUT> compare = out;
        compare[Uniform] = entry->out[Uniform];
        compare[Targets] = entry->out[Targets];
        compare[Views] = views_stale ? ~entry->out[Views] : entry->out[Views];
        if (handles_moved != 0) {
            compare[TexBuffers] = entry->out[TexBuffers];
            compare[Payload] = entry->out[Payload];
        }
        window.targets_moved += entry->out[Targets] != out[Targets] ? 1 : 0;
        if (geometry_moved) {
            compare[Geometry] = entry->out[Geometry];
        }
        if (storage_inputs_moved) {
            compare[Storage] = entry->out[Storage];
        }
        correct = entry->out == compare;
        window.uniform_bindings_changed += correct && uniform_moved ? 1 : 0;
        window.storage_bindings_changed += correct && storage_moved ? 1 : 0;
        stale = !correct;
        if (correct) {
            ++window.hits[static_cast<size_t>(hit)];
            ++window.tiers[(std::min)(moved, 3U)];
            window.tiers_cross[(std::min)(moved, 3U)] +=
                hit == Hit::Previous || hit == Hit::Older ? 1 : 0;
            window.hit_tex_event += entry->tex_clock != tex_clock ? 1 : 0;
            window.hit_buf_epoch += entry->buf_epoch != buf_epoch ? 1 : 0;
            window.hit_map += entry->map_serial != map_serial ? 1 : 0;
            window.hit_gpu_modified += entry->gpu_modified != gpu_modified ? 1 : 0;
        } else {
            ++window.stale_hits[static_cast<size_t>(hit)];
            Entry masked = *entry;
            masked.out[Uniform] = out[Uniform];
            masked.out[Targets] = out[Targets];
            if (!views_stale) {
                masked.out[Views] = out[Views];
            }
            if (handles_moved != 0) {
                masked.out[TexBuffers] = out[TexBuffers];
                masked.out[Payload] = out[Payload];
            }
            if (geometry_moved) {
                masked.out[Geometry] = out[Geometry];
            }
            if (storage_inputs_moved) {
                masked.out[Storage] = out[Storage];
            }
            const Stale reason = Classify(masked, out, bindings, rescaling, tex_clock, scale,
                                          buf_epoch, map_serial);
            ++window.stale[static_cast<size_t>(reason)];
            if (reason >= Stale::HoleHandle) {
                LogHole(reason, pipeline, *entry, out, bindings);
                if (reason == Stale::HoleStorage && holes_left != 0) {
                    std::string detail;
                    for (size_t i = 0; i + 5 <= (std::max)(storage_now.size(), entry->storage_bindings.size()) &&
                                       detail.size() < 600;
                         i += 5) {
                        const auto word = [](const std::vector<u64>& v, size_t at) {
                            return at < v.size() ? v[at] : ~u64{};
                        };
                        bool differs = false;
                        for (size_t k = 0; k < 5; ++k) {
                            differs |= word(storage_now, i + k) != word(entry->storage_bindings, i + k);
                        }
                        if (differs) {
                            detail += fmt::format(
                                " [s{} slot {}: dev {:#x}->{:#x} gpu {:#x}->{:#x} size|id {:#x}->{:#x} "
                                "wr|seg {:#x}->{:#x}]",
                                word(storage_now, i) >> 8, word(storage_now, i) & 0xff,
                                word(entry->storage_bindings, i + 1), word(storage_now, i + 1),
                                word(entry->storage_bindings, i + 2), word(storage_now, i + 2),
                                word(entry->storage_bindings, i + 3), word(storage_now, i + 3),
                                word(entry->storage_bindings, i + 4), word(storage_now, i + 4));
                        }
                    }
                    LOG_WARNING(Render_Vulkan, "Draw record storage hole detail: params {}->{} moved {}{}",
                                entry->params.size(), cur_params.size(), moved, detail);
                }
            }
        }
    } else {
        ++window.hits[static_cast<size_t>(Hit::Miss)];
        ++window.new_keys;
        ++frame_new;
    }
    const bool cross = correct && (hit == Hit::Previous || hit == Hit::Older);
    ++window.draws;
    ++window.class_draws[klass];
    window.class_cross[klass] += cross ? 1 : 0;
    window.class_any[klass] += correct ? 1 : 0;
    window.class_stale[klass] += stale ? 1 : 0;
    ++window.overlap[delta ? 1 : 0][correct ? 1 : 0];
    window.indirect += cur_indirect ? 1 : 0;
    window.inline_index += cur_inline ? 1 : 0;
    window.repacked += cur_repacked ? 1 : 0;
    window.writes_storage += pipeline.WritesStorage() ? 1 : 0;
    {
        // 0-8 FullDraw, 9 delta, 10 state changed by a vertex attribute format register,
        // 11 by another register, 12 fast draws off.
        size_t cause = delta ? 9 : full_reason < 9 ? full_reason : 12;
        if (full_reason == 2) { // FullDraw::StateChanged
            const u32 reason = break_method & 0xff;
            const u32 method = break_method >> 8;
            if (reason == static_cast<u32>(VideoCommon::Nx2FastDraw::Reason::Register)) {
                cause = method >= 0x458 && method < 0x468 ? 10 : 11;
            }
        }
        if (full_reason == 2) {
            ++window.break_reasons[(std::min)(break_method & 0xffU, 15U)];
            const u32 reason = break_method & 0xff;
            if (reason == static_cast<u32>(VideoCommon::Nx2FastDraw::Reason::Method) ||
                reason == static_cast<u32>(VideoCommon::Nx2FastDraw::Reason::Register)) {
                ++break_methods[break_method];
            }
        }
        ++window.cause_draws[cause];
        window.cause_cross[cause] += cross ? 1 : 0;
        window.cause_any[cause] += correct ? 1 : 0;
        window.same_targets += have_prev && cur_regs == prev_regs && out[Targets] == prev_targets ? 1 : 0;
        window.same_pipeline_any += have_prev && prev_pipeline == &pipeline ? 1 : 0;
        prev_regs = cur_regs;
        prev_targets = out[Targets];
    }
    window.dynamic_dirty += dynamic_dirty ? 1 : 0;
    if (hit == Hit::Miss || hit == Hit::SameFrame) {
        Attribute(klass, &pipeline);
    }
    frame_sigs.push_back(Sig{
        .pipeline = &pipeline,
        .groups = cur_groups,
        .values_first = static_cast<u32>(frame_values.size()),
        .values_count = static_cast<u32>(cur_values.size()),
    });
    frame_values.insert(frame_values.end(), cur_values.begin(), cur_values.end());
    ++frame_draws;
    frame_cross += cross ? 1 : 0;
    frame_any += correct ? 1 : 0;
    frame_stale += stale ? 1 : 0;

    // Store this resolve's outcome as the record.
    Entry& stored = *entry;
    if (!entry_new && stored.bindings != bindings) {
        stored.older_bindings = std::move(stored.bindings);
    }
    stored.bindings = bindings;
    stored.out = out;
    stored.params = cur_params;
    stored.storage_bindings = std::move(storage_now);
    stored.tex_clock = tex_clock;
    stored.scale_epoch = scale;
    stored.buf_epoch = buf_epoch;
    stored.map_serial = map_serial;
    stored.gpu_modified = gpu_modified;
    stored.rescaling = rescaling;
    if (entry_new) {
        stored.seen = 1;
    } else {
        const u64 shift = frame - stored.last_frame;
        stored.seen = (shift >= 64 ? 0 : stored.seen << shift) | 1;
    }
    stored.last_frame = frame;
    stored.valid = true;
    if (mode >= Record::Mode::Verify) {
        if (hit_prepared && !record_used) {
            VerifyHit(pipeline);
        }
        // The pipeline holds this draw's bindings: captured by the full resolve, kept from the
        // last full one by a delta draw, or put there by the hit.
        if (!delta || pipeline.delta_views_valid) {
            StoreRecord(stored, pipeline, gpu_memory, map_serial, tex_clock);
        }
    }
    hit_prepared = false;

    TrackPayload(draw_plan, pipeline, entries, count);
    if (sample) {
        ++window.samples;
        window.ticks_regs += begin_ticks_regs;
        window.ticks_data += begin_ticks_data;
        window.ticks_lookup += begin_ticks_lookup;
        window.ticks_outcome += Profile::Ticks() - t0;
        window.data_words += draw_plan.words.size();
        if (record_used) {
            ++window.record_configure[0];
            window.record_configure[1] += t0 - begin_end;
        } else {
            window.configure_ticks[delta ? 1 : 0][correct ? 1 : 0] += t0 - begin_end;
            ++window.configure_samples[delta ? 1 : 0][correct ? 1 : 0];
        }
    }
}

size_t DrawRecorder::RunBucket(u64 length) noexcept {
    if (length <= 2) {
        return static_cast<size_t>(length == 0 ? 0 : length - 1);
    }
    // 3-4, 5-8, 9-16, 17-64, 65+
    if (length <= 4) {
        return 2;
    }
    if (length <= 8) {
        return 3;
    }
    if (length <= 16) {
        return 4;
    }
    if (length <= 64) {
        return 5;
    }
    return 6;
}

void DrawRecorder::TrackPayload(Plan& draw_plan, GraphicsPipeline& pipeline,
                                const DescriptorUpdateEntry* entries, size_t count) {
    window.payload_entries += count;
    if (count < draw_plan.kinds.size()) {
        ++window.unknown_layout;
    }
    u64 full = 19;
    std::array<u64, 2> sets{23, 29};
    u32 uniforms = 0;
    for (size_t index = 0; index < count; ++index) {
        const EntryKind kind =
            index < draw_plan.kinds.size() ? draw_plan.kinds[index] : EntryKind::Other;
        const DescriptorUpdateEntry& item = entries[index];
        u64& set = sets[draw_plan.gpl && index >= draw_plan.vertex_entries ? 1 : 0];
        switch (kind) {
        case EntryKind::Uniform:
        case EntryKind::Storage:
        case EntryKind::Other:
            full = Mix(Mix(Mix(full, PointerBits(item.buffer.buffer)), item.buffer.offset),
                       item.buffer.range);
            // Stage B1: uniforms from one stream ring and storage buffers as dynamic offsets.
            set = Mix(Mix(set, kind == EntryKind::Uniform ? 0 : PointerBits(item.buffer.buffer)),
                      item.buffer.range);
            uniforms += kind == EntryKind::Uniform ? 1 : 0;
            break;
        case EntryKind::Texel:
        case EntryKind::ImageTexel:
            full = Mix(full, PointerBits(item.texel_buffer));
            set = Mix(set, PointerBits(item.texel_buffer));
            break;
        case EntryKind::Sampled:
        case EntryKind::Image: {
            const u64 sampler = PointerBits(item.image.sampler);
            const u64 view = PointerBits(item.image.imageView);
            full = Mix(Mix(Mix(full, sampler), view), item.image.imageLayout);
            set = Mix(Mix(Mix(set, sampler), view), item.image.imageLayout);
            break;
        }
        }
    }
    window.uniform_entries += uniforms;
    const bool same_pipeline = have_prev && prev_pipeline == &pipeline;
    if (draw_plan.gpl) {
        ++window.gpl_draws;
        // GraphicsPipeline::ConfigureDraw: a library pipeline commits, writes and binds both
        // sets when it is bound again or its payload differs from its last draw's.
        if (!same_pipeline || !draw_plan.has_last_payload || draw_plan.last_payload != full) {
            ++window.gpl_updates;
        }
    }
    draw_plan.last_payload = full;
    draw_plan.has_last_payload = true;
    if (same_pipeline && prev_payload.size() == count) {
        ++window.same_pipeline;
        bool offsets = false;
        bool handles = false;
        bool views = false;
        bool texels = false;
        for (size_t index = 0; index < count; ++index) {
            const EntryKind kind =
                index < draw_plan.kinds.size() ? draw_plan.kinds[index] : EntryKind::Other;
            const DescriptorUpdateEntry& now = entries[index];
            const DescriptorUpdateEntry& was = prev_payload[index];
            switch (kind) {
            case EntryKind::Uniform:
            case EntryKind::Storage:
            case EntryKind::Other:
                if (now.buffer.buffer != was.buffer.buffer || now.buffer.range != was.buffer.range) {
                    handles = true;
                    ++window.changed_entries;
                } else if (now.buffer.offset != was.buffer.offset) {
                    offsets = true;
                    ++window.changed_entries;
                    window.changed_uniform_offsets += kind == EntryKind::Uniform ? 1 : 0;
                }
                break;
            case EntryKind::Texel:
            case EntryKind::ImageTexel:
                if (now.texel_buffer != was.texel_buffer) {
                    texels = true;
                    ++window.changed_entries;
                }
                break;
            case EntryKind::Sampled:
            case EntryKind::Image:
                if (now.image.sampler != was.image.sampler ||
                    now.image.imageView != was.image.imageView ||
                    now.image.imageLayout != was.image.imageLayout) {
                    views = true;
                    ++window.changed_entries;
                }
                break;
            }
        }
        if (views) {
            ++window.views_changed;
        } else if (texels) {
            ++window.texel_changed;
        } else if (handles) {
            ++window.buffer_handles;
        } else if (offsets) {
            ++window.offsets_only;
        } else {
            ++window.payload_identical;
        }
    }
    // Stage B1: a persistent set is bound again when its contents (or the layout) change.
    if (!(same_pipeline && prev_sets == sets)) {
        ++window.set_changes;
    }
    for (size_t index = 0; index < (draw_plan.gpl ? 2U : 1U); ++index) {
        const u64 set_key = Mix(sets[index], draw_plan.layout_hash + index);
        auto [it, inserted] = set_frames.try_emplace(set_key, frame);
        if (inserted) {
            ++window.sets_new;
            ++window.sets_first_use;
        } else if (it->second != frame) {
            ++window.sets_first_use;
            ++window.sets_cross_frame;
            it->second = frame;
        }
    }
    // Runs of consecutive draws (stage B2: multi-draw; B1: one set bind per run).
    const bool same_key = have_prev && prev_key == key;
    const auto run = [&](bool continues, u64& length, std::array<u64, 8>& histogram) {
        if (continues && length != 0) {
            ++length;
            return;
        }
        if (length != 0) {
            histogram[RunBucket(length)] += length;
        }
        length = 1;
    };
    run(same_key, run_key_len, window.run_key);
    run(same_key && full == prev_full, run_full_len, window.run_full);
    run(same_key && sets == prev_sets, run_set_len, window.run_set);
    prev_payload.assign(entries, entries + count);
    prev_pipeline = &pipeline;
    prev_key = key;
    prev_full = full;
    prev_sets = sets;
    have_prev = true;
}

u64 DrawRecorder::DescriptorHash(Tegra::MemoryManager& gpu_memory, u32 tic, u32 tsc,
                                 bool sampled) {
    // The TIC (and for textures the TSC) entry a binding was resolved from: a game that rewrites
    // a pool entry under the same handle raises no texture cache event.
    const auto& regs = cur_maxwell3d->regs;
    u64 hash = 61;
    const auto add_entry = [&](GPUVAddr address) {
        // Entries are 32-byte aligned: one page lookup, one copy.
        std::array<u64, 4> words;
        if (const u8* const host = PagePointer(gpu_memory, address)) {
            std::memcpy(words.data(), host, sizeof(words));
        } else {
            gpu_memory.ReadBlockUnsafe(address, words.data(), sizeof(words));
        }
        hash = MixBytes(hash, words.data(), sizeof(words));
    };
    const u32 tic_limit = regs.tex_header.limit;
    const u32 header = Tegra::Texture::TextureHeaderIndex(
        tic, tic_limit,
        Tegra::Texture::UsesHeaderByteOffsetHandles(gpu_memory.UsesNativeTiling(),
                                                    Settings::GetCurrentProgramID()));
    if (tic != ~0U && header <= tic_limit) {
        add_entry(regs.tex_header.Address() + u64{header} * 32);
    }
    if (sampled && tsc != ~0U && tsc <= regs.tex_sampler.limit) {
        add_entry(regs.tex_sampler.Address() + u64{tsc} * 32);
    }
    return hash;
}

void DrawRecorder::EnsureHandles() {
    if (!values_read) {
        HashData(*plan, *cur_pipeline, *cur_maxwell3d, *cur_memory);
        values_read = true;
    }
}

bool DrawRecorder::PatchImageEntries(const Plan& draw_plan,
                                     const std::vector<VideoCommon::ImageViewInOut>& views,
                                     const std::vector<VideoCommon::SamplerId>& samplers,
                                     std::vector<DescriptorUpdateEntry>& entries) {
    if (draw_plan.element_descriptors.size() != views.size() ||
        entry->rescaled.size() != views.size()) {
        return false;
    }
    for (const u16 index : moved_elements) {
        const Plan::ElementDescriptor& element = draw_plan.element_descriptors[index];
        if (element.entry < 0) {
            return false; // a texture buffer: not kept here
        }
        if (static_cast<size_t>(element.entry) >= entries.size()) {
            return false;
        }
        // As PushImageDescriptors: a depth-compare sample of an uncomparable view gets the null
        // view (#1485).
        const bool uncomparable = !element.image && element.is_depth &&
                                  views[index].id != VideoCommon::NULL_IMAGE_VIEW_ID &&
                                  IsUncomparable(texture_cache.GetImageView(views[index].id),
                                                 element.read_from_memory);
        ImageView& image_view = texture_cache.GetImageView(
            uncomparable ? VideoCommon::NULL_IMAGE_VIEW_ID : views[index].id);
        // The rescaling words are per descriptor array: a binding whose rescaling changed
        // needs them made again.
        if ((texture_cache.IsRescaling(image_view) ? 1 : 0) != entry->rescaled[index]) {
            return false;
        }
        DescriptorUpdateEntry& out = entries[static_cast<size_t>(element.entry)];
        if (element.image) {
            out = DescriptorUpdateEntry{VkDescriptorImageInfo{
                .sampler = VK_NULL_HANDLE,
                .imageView =
                    image_view.StorageView(element.type, element.format, element.byte_addressed),
                .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
            }};
        } else {
            VkImageView vk_image_view = image_view.Handle(element.type);
            if (vk_image_view == VK_NULL_HANDLE) {
                const VkImageView null_view =
                    texture_cache.GetImageView(VideoCommon::NULL_IMAGE_VIEW_ID).Handle(element.type);
                if (null_view != VK_NULL_HANDLE) {
                    vk_image_view = null_view;
                }
            }
            const s16 sampler = draw_plan.sampler_of[index];
            if (sampler < 0 || static_cast<size_t>(sampler) >= samplers.size()) {
                return false;
            }
            out = DescriptorUpdateEntry{VkDescriptorImageInfo{
                .sampler = texture_cache.GetSampler(samplers[sampler]).HandleFor(image_view,
                                                                                 element.is_depth),
                .imageView = vk_image_view,
                .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
            }};
        }
    }
    return true;
}

void DrawRecorder::BeforeFull(GraphicsPipeline& pipeline) const {
    if (mode >= Record::Mode::Verify && active) {
        pipeline.record_capture = true;
    }
}

bool DrawRecorder::PrepareHit(GraphicsPipeline& pipeline, Tegra::Engines::Maxwell3D& maxwell3d,
                              Tegra::MemoryManager& gpu_memory) {
    namespace Profile = VideoCommon::Nx2Profile;
    hit_prepared = false;
    if (mode < Record::Mode::Verify || !active) {
        return false;
    }
    const auto reject = [&](size_t reason) {
        ++window.hit_rejects[reason];
        return false;
    };
    if (entry_new) {
        return reject(5);
    }
    if (!entry->has_record) {
        return reject(0);
    }
    const Plan& draw_plan = *plan;
    if (cur_indirect) {
        // Indirect and GPU-read list draws record per-draw arguments in ConfigureDraw: not
        // taken by records yet.
        return reject(6);
    }
    if (draw_plan.klass == PipeClass::Mesh || draw_plan.klass == PipeClass::WritesImages ||
        draw_plan.texture_buffers || cur_inline || cur_repacked ||
        maxwell3d.regs.transform_feedback_enabled != 0) {
        return reject(1);
    }
    if (entry->record_map_serial != gpu_memory.MapSerial()) {
        return reject(2);
    }
    EnsureHandles();
    if (entry->views.size() != draw_plan.words.size() ||
        entry->handles.size() != cur_values.size() ||
        entry->views.size() != cur_values.size()) {
        return reject(3);
    }
    const u64 t0 = sample ? Profile::Ticks() : 0;
    // =1 works on the pipeline's own binding state: when it already holds this record's
    // bindings (it drew with this key last and nothing captured since), nothing is copied.
    const bool apply = mode == Record::Mode::On;
    const bool owned = apply && pipeline.record_owner == entry &&
                       pipeline.record_owner_version == entry->version &&
                       pipeline.record_owner_epoch == owner_epoch;
    std::vector<VideoCommon::ImageViewInOut>& views = apply ? pipeline.delta_views : hit_views;
    std::vector<VideoCommon::SamplerId>& samplers = apply ? pipeline.delta_samplers : hit_samplers;
    if (!owned) {
        views = entry->views;
        samplers = entry->samplers;
        if (apply) {
            pipeline.record_owner = nullptr;
        }
    }
    // EDEN_NX2_DRAW_RECORD_TEST=n: every n-th hit takes a fallback path on purpose (an image
    // event, a rescale, a changed descriptor entry), cycling through them.
    u32 forced = ~0U;
    if (test_every != 0 && ++test_counter % test_every == 0) {
        forced = static_cast<u32>((test_counter / test_every) % 3);
        ++window.tests[forced];
    }
    // Bindings whose handle moved are resolved again; after an image event, all of them.
    const bool all = texture_cache.BindingClock() != entry->record_tex_clock || forced == 0;
    const bool via_header_index =
        maxwell3d.regs.sampler_binding ==
        Tegra::Engines::Maxwell3D::Regs::SamplerBinding::ViaHeaderBinding;
    moved_elements.clear();
    bool synchronized = false;
    bool descriptor_moved = false;
    const bool check_entries = entry->entry_copies.size() == cur_values.size();
    for (size_t index = 0; index < cur_values.size(); ++index) {
        bool changed = all || entry->handles[index] != cur_values[index];
        if (!changed && check_entries &&
            (!EntryHolds(entry->entry_copies[index]) || (forced == 2 && index == 0))) {
            // The pool entry under the kept handle was rewritten (no texture cache event).
            changed = true;
            descriptor_moved = true;
        }
        if (changed && test_break && !all) {
            changed = false; // negative control: keep the record's binding
        }
        if (!changed) {
            continue;
        }
        if (!synchronized) {
            texture_cache.SynchronizeDescriptors(false);
            synchronized = true;
        }
        const u64 raw = cur_values[index];
        const std::pair<u32, u32> handle =
            raw == ~0ULL ? VideoCommon::NULL_TEXTURE_HANDLE
                         : Tegra::Texture::TexturePair(static_cast<u32>(raw), via_header_index);
        views[index].index = handle.first;
        views[index].id = {};
        texture_cache.FillImageViews(std::span(&views[index], 1), false, false);
        const s16 sampler = draw_plan.sampler_of[index];
        if (sampler >= 0 && static_cast<size_t>(sampler) < samplers.size()) {
            samplers[sampler] = texture_cache.GetSamplerId(handle.second, false);
        }
        moved_elements.push_back(static_cast<u16>(index));
    }
    const u32 moved = static_cast<u32>(moved_elements.size());
    if (texture_cache.SampledViewsChange(views)) {
        if (apply) {
            pipeline.record_owner = nullptr; // the views were changed for nothing
        }
        return reject(4);
    }
    hit_moved = moved;
    window.hit_descriptor_moved += descriptor_moved ? 1 : 0;
    hit_images_valid = entry->images_valid &&
                       entry->kept_scale_epoch == texture_cache.ScaleEpoch() && forced != 1;
    bool patched = false;
    // EDEN_NX2_DRAW_RECORD_PATCH: unset = on; "0" off; an A/B switch ("ab", ...) = on in its arm.
    static const Common::Nx2Ab::Switch patch_switch =
        Common::Nx2Ab::Parse(std::getenv("EDEN_NX2_DRAW_RECORD_PATCH"));
    const bool patch_on = patch_switch.kind == Common::Nx2Ab::Switch::Kind::Off ||
                          (patch_switch.Active() && patch_switch.value != "0");
    if (!patch_on && moved != 0) {
        hit_images_valid = false;
    }
    if (hit_images_valid && moved != 0) {
        // Make the moved bindings' kept descriptors again instead of pushing them all.
        if (apply) {
            if (!owned) {
                pipeline.kept_image_entries = entry->image_entries;
            }
            hit_images_valid = PatchImageEntries(draw_plan, views, samplers,
                                                 pipeline.kept_image_entries);
        } else {
            hit_entries = entry->image_entries;
            hit_images_valid = PatchImageEntries(draw_plan, views, samplers, hit_entries);
        }
        patched = hit_images_valid;
    } else if (!apply && hit_images_valid) {
        hit_entries = entry->image_entries;
    }
    window.hit_images_patched += patched ? 1 : 0;
    ++window.hit_ready;
    window.hit_bindings_moved += moved != 0 ? 1 : 0;
    window.hit_clock_moved += all ? 1 : 0;
    window.hit_images_replayed += hit_images_valid ? 1 : 0;
    hit_prepared = true;
    if (apply) {
        pipeline.configured_stages = entry->stages;
        pipeline.delta_views_valid = pipeline.delta_blocker == GraphicsPipeline::DeltaBlocker::None;
        if (!pipeline.delta_computed_handles.empty()) {
            pipeline.RecaptureComputedHandles();
        }
        pipeline.kept_images_valid = hit_images_valid;
        if (hit_images_valid) {
            if (!owned && !patched) {
                pipeline.kept_image_entries = entry->image_entries;
            }
            if (!owned) {
                pipeline.kept_image_ranges = entry->image_ranges;
                pipeline.kept_rescaling = entry->rescaling_words;
                pipeline.kept_scale_epoch = entry->kept_scale_epoch;
            }
        } else {
            pipeline.kept_image_entries.clear();
        }
        pipeline.record_owner = moved == 0 && hit_images_valid ? entry : nullptr;
        pipeline.record_owner_version = entry->version;
        pipeline.record_owner_epoch = owner_epoch;
    }
    if (sample) {
        ++window.prepare_ticks[0];
        window.prepare_ticks[1] += Profile::Ticks() - t0;
    }
    return true;
}

void DrawRecorder::VerifyHit(GraphicsPipeline& pipeline) {
    // The full resolve just ran (and captured): compare what the hit would have used.
    ++window.verified;
    ++total_verified;
    const auto same_view = [](const VideoCommon::ImageViewInOut& lhs,
                              const VideoCommon::ImageViewInOut& rhs) {
        return lhs.index == rhs.index && lhs.id == rhs.id;
    };
    std::array<bool, 5> bad{};
    bad[0] = !std::ranges::equal(hit_views, pipeline.delta_views, same_view);
    bad[4] = entry->stages != pipeline.configured_stages;
    bad[1] = hit_samplers != pipeline.delta_samplers;
    if (hit_images_valid) {
        const auto same_image = [](const DescriptorUpdateEntry& lhs,
                                   const DescriptorUpdateEntry& rhs) {
            return lhs.image.sampler == rhs.image.sampler &&
                   lhs.image.imageView == rhs.image.imageView &&
                   lhs.image.imageLayout == rhs.image.imageLayout;
        };
        bad[2] = entry->image_ranges != pipeline.kept_image_ranges ||
                 hit_entries.size() != pipeline.kept_image_entries.size();
        // Per entry against the descriptor made now (as PushImageDescriptors would make it from
        // the full resolve's view and sampler): the hit is wrong only where it differs from that;
        // a full-path entry that differs from it is a stale cached descriptor (binding memo).
        u32 full_stale = 0;
        const bool by_element = !plan->element_descriptors.empty() &&
                                plan->element_descriptors.size() == pipeline.delta_views.size();
        for (size_t i = 0; !bad[2] && i < hit_entries.size(); ++i) {
            if (same_image(hit_entries[i], pipeline.kept_image_entries[i])) {
                continue;
            }
            if (!by_element) {
                bad[2] = true;
                break;
            }
            s32 element = -1;
            for (size_t e = 0; e < plan->element_descriptors.size(); ++e) {
                if (plan->element_descriptors[e].entry == static_cast<s16>(i)) {
                    element = static_cast<s32>(e);
                    break;
                }
            }
            if (element < 0) {
                bad[2] = true;
                break;
            }
            const auto& desc = plan->element_descriptors[element];
            const VideoCommon::ImageViewId id_now = pipeline.delta_views[element].id;
            const bool uncomparable = !desc.image && desc.is_depth &&
                                      id_now != VideoCommon::NULL_IMAGE_VIEW_ID &&
                                      IsUncomparable(texture_cache.GetImageView(id_now),
                                                     desc.read_from_memory);
            ImageView& view_now = texture_cache.GetImageView(
                uncomparable ? VideoCommon::NULL_IMAGE_VIEW_ID : id_now);
            VkDescriptorImageInfo now{.sampler = VK_NULL_HANDLE,
                                      .imageView = VK_NULL_HANDLE,
                                      .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
            if (desc.image) {
                now.imageView = view_now.StorageView(desc.type, desc.format, desc.byte_addressed);
            } else {
                now.imageView = view_now.Handle(desc.type);
                if (now.imageView == VK_NULL_HANDLE) {
                    const VkImageView null_view = texture_cache
                                                      .GetImageView(VideoCommon::NULL_IMAGE_VIEW_ID)
                                                      .Handle(desc.type);
                    if (null_view != VK_NULL_HANDLE) {
                        now.imageView = null_view;
                    }
                }
                const s16 sampler_index = plan->sampler_of[element];
                if (sampler_index >= 0 &&
                    static_cast<size_t>(sampler_index) < pipeline.delta_samplers.size()) {
                    now.sampler = texture_cache.GetSampler(pipeline.delta_samplers[sampler_index])
                                      .HandleFor(view_now, desc.is_depth);
                }
            }
            const auto matches = [&](const VkDescriptorImageInfo& info) {
                return info.sampler == now.sampler && info.imageView == now.imageView &&
                       info.imageLayout == now.imageLayout;
            };
            if (!matches(hit_entries[i].image)) {
                bad[2] = true;
            } else if (!matches(pipeline.kept_image_entries[i].image)) {
                ++full_stale;
                // Could the stale descriptor change pixels? Only a different sampler (the FSR LOD
                // bias variant) on a view with mips, or with different min/mag filters (the bias
                // moves the min/mag switch), can.
                const VkDescriptorImageInfo& old_info = pipeline.kept_image_entries[i].image;
                const bool sampler_only = old_info.imageView == now.imageView &&
                                          old_info.imageLayout == now.imageLayout;
                const s16 sampler_index = desc.image ? s16{-1} : plan->sampler_of[element];
                const Tegra::Texture::TSCEntry* tsc =
                    sampler_index >= 0 &&
                            static_cast<size_t>(sampler_index) < pipeline.delta_samplers.size()
                        ? &texture_cache.GetSampler(pipeline.delta_samplers[sampler_index])
                               .Descriptor()
                        : nullptr;
                const u32 view_levels = static_cast<u32>(view_now.range.extent.levels);
                const bool filters_differ =
                    tsc != nullptr && tsc->mag_filter.Value() != tsc->min_filter.Value();
                const bool may_show = sampler_only && (view_levels > 1 || filters_differ);
                total_full_stale_visible += may_show ? 1 : 0;
                if (stale_logged < 64) {
                    ++stale_logged;
                    const Image* const backing = view_now.BackingImage();
                    const auto& hashes = pipeline.ShaderHashes();
                    LOG_INFO(Render_Vulkan,
                             "DRAWREC_STALE frame={} vs={:016x} ps={:016x} entry={} element={} "
                             "sampler_only={} may_show={} view_levels={} image={:#x} {}x{} "
                             "levels={} format={} gpu_modified={} gpu_written={} tsc_min={} "
                             "tsc_mag={} tsc_mip={}",
                             frame, hashes[1], hashes[5], i, element, sampler_only, may_show,
                             view_levels, backing != nullptr ? backing->gpu_addr : 0,
                             backing != nullptr ? backing->info.size.width : 0,
                             backing != nullptr ? backing->info.size.height : 0,
                             backing != nullptr ? backing->info.resources.levels : s32{0},
                             static_cast<u32>(view_now.format),
                             backing != nullptr &&
                                 True(backing->flags & VideoCommon::ImageFlagBits::GpuModified),
                             false, // GpuWritten (EDEN_NX2_OOT_LOD_STICKY) isn't tracked here
                             tsc != nullptr ? static_cast<u32>(tsc->min_filter.Value()) : 99,
                             tsc != nullptr ? static_cast<u32>(tsc->mag_filter.Value()) : 99,
                             tsc != nullptr ? static_cast<u32>(tsc->mipmap_filter.Value()) : 99);
                }
            }
        }
        window.verify_full_stale += full_stale;
        total_full_stale += full_stale;
        bad[3] = entry->rescaling_words != pipeline.kept_rescaling;
    }
    for (size_t part = 0; part < bad.size(); ++part) {
        window.verify_mismatch[part] += bad[part] ? 1 : 0;
        total_mismatches[part] += bad[part] ? 1 : 0;
    }
    if (bad[2] && descriptor_logged < 32) {
        // Which kept descriptor differs, and what the hit knew about its element.
        ++descriptor_logged;
        std::string detail;
        const size_t count = (std::min)(hit_entries.size(), pipeline.kept_image_entries.size());
        for (size_t i = 0; i < count && detail.size() < 700; ++i) {
            const VkDescriptorImageInfo& kept = hit_entries[i].image;
            const VkDescriptorImageInfo& fresh = pipeline.kept_image_entries[i].image;
            if (kept.sampler == fresh.sampler && kept.imageView == fresh.imageView &&
                kept.imageLayout == fresh.imageLayout) {
                continue;
            }
            // The view element of this entry.
            s32 element = -1;
            for (size_t e = 0; e < plan->element_descriptors.size(); ++e) {
                if (plan->element_descriptors[e].entry == static_cast<s16>(i)) {
                    element = static_cast<s32>(e);
                }
            }
            const bool moved = element >= 0 && std::ranges::find(moved_elements,
                                                                 static_cast<u16>(element)) !=
                                                    moved_elements.end();
            detail += fmt::format(
                " [entry {} element {} moved {} sampler {} view {} layout {} | view id {} -> {}]", i,
                element, moved, kept.sampler != fresh.sampler, kept.imageView != fresh.imageView,
                kept.imageLayout != fresh.imageLayout,
                element >= 0 && static_cast<size_t>(element) < hit_views.size()
                    ? hit_views[element].id.index
                    : ~0U,
                element >= 0 && static_cast<size_t>(element) < pipeline.delta_views.size()
                    ? pipeline.delta_views[element].id.index
                    : ~0U);
        }
        const auto& hashes = pipeline.ShaderHashes();
        LOG_ERROR(Render_Vulkan,
                  "Draw record verify: image descriptor mismatch, vs {:016x} ps {:016x} frame {} | "
                  "moved {} all-after-event {} | entries {} vs {} | scale epoch record {} now {} | "
                  "record clock {} now {} | ranges equal {}{}",
                  hashes[1], hashes[5], frame, moved_elements.size(),
                  entry->record_tex_clock != texture_cache.BindingClock(), hit_entries.size(),
                  pipeline.kept_image_entries.size(), entry->kept_scale_epoch,
                  texture_cache.ScaleEpoch(), entry->record_tex_clock, texture_cache.BindingClock(),
                  entry->image_ranges == pipeline.kept_image_ranges, detail);
    }
    if ((bad[0] || bad[1] || bad[2] || bad[3] || bad[4]) && verify_logged < 24) {
        ++verify_logged;
        std::string detail;
        for (size_t i = 0; i < (std::min)(hit_views.size(), pipeline.delta_views.size()); ++i) {
            if (!same_view(hit_views[i], pipeline.delta_views[i]) && detail.size() < 300) {
                detail += fmt::format(" view {} {}:{}->{}:{}", i, hit_views[i].index,
                                      hit_views[i].id.index, pipeline.delta_views[i].index,
                                      pipeline.delta_views[i].id.index);
            }
        }
        const auto& hashes = pipeline.ShaderHashes();
        LOG_ERROR(Render_Vulkan,
                  "Draw record verify mismatch: vs {:016x} ps {:016x} views {} samplers {} image "
                  "descriptors {} rescaling {} (images replayed {}){}",
                  hashes[1], hashes[5], bad[0], bad[1], bad[2], bad[3], hit_images_valid, detail);
    }
}

bool DrawRecorder::EntryHolds(const Entry::EntryCopy& copy) noexcept {
    return (copy.tic == nullptr || std::memcmp(copy.tic, copy.tic_words.data(), 32) == 0) &&
           (copy.tsc == nullptr || std::memcmp(copy.tsc, copy.tsc_words.data(), 32) == 0);
}

void DrawRecorder::CopyEntry(Tegra::MemoryManager& gpu_memory, size_t element, u32 tic, u32 tsc,
                             bool sampled, Entry::EntryCopy& out) {
    // The TIC (and for textures the TSC) entry a binding was resolved from; entries are
    // 32-byte aligned, so each lies in one page.
    const auto& regs = cur_maxwell3d->regs;
    out = Entry::EntryCopy{};
    const u32 tic_limit = regs.tex_header.limit;
    const u32 header = Tegra::Texture::TextureHeaderIndex(
        tic, tic_limit,
        Tegra::Texture::UsesHeaderByteOffsetHandles(gpu_memory.UsesNativeTiling(),
                                                    Settings::GetCurrentProgramID()));
    if (tic != ~0U && header <= tic_limit) {
        out.tic = PagePointer(gpu_memory, regs.tex_header.Address() + u64{header} * 32);
        if (out.tic != nullptr) {
            std::memcpy(out.tic_words.data(), out.tic, 32);
        }
    }
    if (sampled && tsc != ~0U && tsc <= regs.tex_sampler.limit) {
        out.tsc = PagePointer(gpu_memory, regs.tex_sampler.Address() + u64{tsc} * 32);
        if (out.tsc != nullptr) {
            std::memcpy(out.tsc_words.data(), out.tsc, 32);
        }
    }
    (void)element;
}

void DrawRecorder::StoreRecord(Entry& stored, GraphicsPipeline& pipeline,
                               Tegra::MemoryManager& gpu_memory_for_store, u64 map_serial,
                               u64 tex_clock) {
    EnsureHandles();
    if (pipeline.delta_views.size() != cur_values.size()) {
        stored.has_record = false;
        pipeline.record_owner = nullptr;
        return;
    }
    const bool via_header_index =
        cur_maxwell3d->regs.sampler_binding ==
        Tegra::Engines::Maxwell3D::Regs::SamplerBinding::ViaHeaderBinding;
    const auto store_element = [&](size_t index) {
        const u64 raw = cur_values[index];
        const std::pair<u32, u32> handle =
            raw == ~0ULL ? VideoCommon::NULL_TEXTURE_HANDLE
                         : Tegra::Texture::TexturePair(static_cast<u32>(raw), via_header_index);
        const s16 sampler = plan->sampler_of[index];
        if (sampler >= 0) {
            stored.sampler_handles[sampler] = handle.second;
        }
        CopyEntry(gpu_memory_for_store, index, stored.views[index].index, handle.second,
                  sampler >= 0, stored.entry_copies[index]);
        const VideoCommon::ImageViewId view_id = stored.views[index].id;
        stored.rescaled[index] =
            view_id && texture_cache.IsRescaling(texture_cache.GetImageView(view_id)) ? 1 : 0;
    };
    const bool partial = hit_prepared && mode == Record::Mode::On && stored.has_record &&
                         stored.views.size() == pipeline.delta_views.size() &&
                         stored.entry_copies.size() == cur_values.size() &&
                         stored.rescaled.size() == cur_values.size() &&
                         stored.record_map_serial == map_serial;
    if (partial) {
        // A hit: only the bindings it resolved again changed.
        for (const u16 index : moved_elements) {
            stored.views[index] = pipeline.delta_views[index];
            stored.handles[index] = cur_values[index];
            const s16 sampler = plan->sampler_of[index];
            if (sampler >= 0 && static_cast<size_t>(sampler) < stored.samplers.size()) {
                stored.samplers[sampler] = pipeline.delta_samplers[sampler];
            }
            store_element(index);
        }
    } else {
        stored.views = pipeline.delta_views;
        stored.samplers = pipeline.delta_samplers;
        stored.handles = cur_values;
        stored.sampler_handles.assign(stored.samplers.size(), ~0U);
        stored.entry_copies.resize(cur_values.size());
        stored.rescaled.resize(cur_values.size());
        for (size_t index = 0; index < cur_values.size(); ++index) {
            store_element(index);
        }
    }
    stored.images_valid = pipeline.kept_images_valid;
    if (stored.images_valid) {
        stored.image_entries = pipeline.kept_image_entries;
        stored.image_ranges = pipeline.kept_image_ranges;
        stored.rescaling_words = pipeline.kept_rescaling;
        stored.kept_scale_epoch = pipeline.kept_scale_epoch;
    }
    stored.stages = pipeline.configured_stages;
    stored.record_map_serial = map_serial;
    stored.record_tex_clock = tex_clock;
    stored.has_record = true;
    ++stored.version;
    pipeline.record_owner = &stored;
    pipeline.record_owner_version = stored.version;
    pipeline.record_owner_epoch = owner_epoch;
}

void DrawRecorder::Attribute(size_t klass, const GraphicsPipeline* pipeline) {
    // 0 no draw of the pipeline in the previous frame, 1 draw kind/channel, 2 render targets,
    // 3 pools/sampler mode/samples/xfb, 4 vertex stream structure, 5 index format, 6 constant
    // buffer sizes, 7 handle words, 8 computed handles, 9 storage words (key mode), 10 nothing
    // differs (key equal to that draw's), 11 data word count differs.
    const bool computed = klass == static_cast<size_t>(PipeClass::ComputedGpl) ||
                          klass == static_cast<size_t>(PipeClass::ComputedMono);
    std::array<bool, 12> differs{};
    const size_t position = frame_sigs.size();
    const auto found = prev_by_pipeline.find(pipeline);
    if (found == prev_by_pipeline.end() || found->second.empty()) {
        differs[0] = true;
    } else {
        const std::vector<u32>& candidates = found->second;
        // The previous frame's draw of this pipeline nearest in draw order.
        const auto upper = std::lower_bound(candidates.begin(), candidates.end(),
                                            static_cast<u32>(position));
        u32 best = upper == candidates.end() ? candidates.back() : *upper;
        if (upper != candidates.begin()) {
            const u32 before = *(upper - 1);
            if (upper == candidates.end() || position - before < best - position) {
                best = before;
            }
        }
        const Sig& old = prev_sigs[best];
        const std::array<size_t, 6> group_category{1, 2, 3, 4, 5, 6};
        for (size_t group = 0; group < 6; ++group) {
            differs[group_category[group]] = old.groups[group] != cur_groups[group];
        }
        if (old.values_count != cur_values.size()) {
            differs[11] = true;
        } else if (old.groups[6] != cur_groups[6]) {
            for (size_t index = 0; index < cur_values.size(); ++index) {
                if (prev_values[old.values_first + index] == cur_values[index]) {
                    continue;
                }
                switch (plan->words[index].kind) {
                case WordOp::Kind::Handle:
                case WordOp::Kind::HandlePair:
                    differs[7] = true;
                    break;
                case WordOp::Kind::Computed:
                    differs[8] = true;
                    break;
                case WordOp::Kind::Storage:
                    differs[9] = true;
                    break;
                case WordOp::Kind::Null:
                    break;
                }
            }
        }
        bool any = false;
        for (size_t category = 1; category < differs.size(); ++category) {
            any |= differs[category];
        }
        differs[10] = !any;
    }
    size_t count = 0;
    size_t last = 0;
    for (size_t category = 0; category < differs.size(); ++category) {
        if (differs[category]) {
            ++window.attr[category];
            window.attr_computed[category] += computed ? 1 : 0;
            ++count;
            last = category;
        }
    }
    if (count == 1) {
        ++window.attr_only[last];
    }
    ++window.attr_draws;
    window.attr_computed_draws += computed ? 1 : 0;
}

void DrawRecorder::EndRuns() {
    const auto flush = [](u64& length, std::array<u64, 8>& histogram) {
        if (length != 0) {
            histogram[RunBucket(length)] += length;
            length = 0;
        }
    };
    flush(run_key_len, window.run_key);
    flush(run_full_len, window.run_full);
    flush(run_set_len, window.run_set);
}

void DrawRecorder::EndFrame() {
    EndRuns();
    prev_sigs.swap(frame_sigs);
    prev_values.swap(frame_values);
    frame_sigs.clear();
    frame_values.clear();
    for (auto& [pipeline, list] : prev_by_pipeline) {
        list.clear();
    }
    for (u32 index = 0; index < prev_sigs.size(); ++index) {
        prev_by_pipeline[prev_sigs[index].pipeline].push_back(index);
    }
    if (frame_draws != 0) {
        const double cross = static_cast<double>(frame_cross) / static_cast<double>(frame_draws);
        window.min_frame_cross = (std::min)(window.min_frame_cross, cross);
        window.sum_frame_cross += cross;
        ++window.counted_frames;
        window.max_new_keys_frame = (std::max)(window.max_new_keys_frame, frame_new);
        if (log_frames) {
            LOG_INFO(Render_Vulkan,
                     "DRAWREC_FRAME frame={} draws={} cross={} any={} stale={} new={} keys={}",
                     frame, frame_draws, frame_cross, frame_any, frame_stale, frame_new,
                     table.size());
        }
    }
    frame_draws = frame_cross = frame_any = frame_stale = frame_new = 0;
    if (mode == Record::Mode::Verify) {
        // Whole-run totals as of this frame, logged whenever a mismatch or stale count grew: runs
        // are ended by a kill (no shutdown report), so the last line is the run's exact total.
        u64 sum = total_full_stale;
        for (const u64 value : total_mismatches) {
            sum += value;
        }
        if (sum != totals_logged_sum && totals_lines < 4096) {
            totals_logged_sum = sum;
            ++totals_lines;
            LOG_INFO(Render_Vulkan,
                     "DRAWREC_TOTALS frame={} verified={} views={} samplers={} images={} "
                     "rescaling={} stages={} full_stale={} full_stale_may_show={}",
                     frame, total_verified, total_mismatches[0], total_mismatches[1],
                     total_mismatches[2], total_mismatches[3], total_mismatches[4],
                     total_full_stale, total_full_stale_visible);
        }
    }
    ++frame;
    ++window.frames;
    ++frames_since_flip;
    if (table.size() > (4U << 20)) {
        level0_entry = nullptr;
        ++owner_epoch;
        table.clear();
        ++table_clears;
    }
    if (set_frames.size() > (4U << 20)) {
        set_frames.clear();
    }
    if (window.frames >= report_frames) {
        Report();
    }
}

void DrawRecorder::Report() {
    namespace Profile = VideoCommon::Nx2Profile;
    const Window& w = window;
    const double frames = static_cast<double>((std::max)(w.frames, u64{1}));
    const double draws = static_cast<double>((std::max)(w.draws, u64{1}));
    const auto pct = [&](u64 value) { return 100.0 * static_cast<double>(value) / draws; };
    const auto hits = [&](Hit hit) { return w.hits[static_cast<size_t>(hit)]; };
    const auto stales = [&](Hit hit) { return w.stale_hits[static_cast<size_t>(hit)]; };
    const u64 correct = hits(Hit::SameFrame) + hits(Hit::Previous) + hits(Hit::Older);
    const u64 stale = stales(Hit::SameFrame) + stales(Hit::Previous) + stales(Hit::Older);
    const u64 cross = hits(Hit::Previous) + hits(Hit::Older);
    const double mean_cross =
        w.counted_frames != 0 ? w.sum_frame_cross / static_cast<double>(w.counted_frames) : 0.0;
    LOG_INFO(Render_Vulkan,
             "Draw record dry: {} frames, {:.0f} draws/frame | key seen before {:.1f}% (correct "
             "{:.1f}%, stale {:.2f}%) | cross-frame correct {:.1f}% (previous frame {:.1f}%, "
             "older {:.1f}%), same frame only {:.1f}%, miss {:.1f}% | per frame cross-frame "
             "correct: min {:.1f}% mean {:.1f}%",
             w.frames, static_cast<double>(w.draws) / frames, pct(correct + stale), pct(correct),
             pct(stale), pct(cross), pct(hits(Hit::Previous)), pct(hits(Hit::Older)),
             pct(hits(Hit::SameFrame)), pct(hits(Hit::Miss)), 100.0 * (w.min_frame_cross > 1.0 ? 0.0 : w.min_frame_cross),
             100.0 * mean_cross);
    std::string stale_text;
    for (size_t reason = 0; reason < w.stale.size(); ++reason) {
        if (w.stale[reason] != 0) {
            stale_text += fmt::format(" {} {:.1f}", stale_names[reason],
                                      static_cast<double>(w.stale[reason]) / frames);
        }
    }
    LOG_INFO(Render_Vulkan,
             "Draw record stale hits per frame:{} | by age: same frame {:.1f}, previous {:.1f}, "
             "older {:.1f}",
             stale_text.empty() ? std::string{" none"} : stale_text,
             static_cast<double>(stales(Hit::SameFrame)) / frames,
             static_cast<double>(stales(Hit::Previous)) / frames,
             static_cast<double>(stales(Hit::Older)) / frames);
    std::string class_text;
    for (size_t klass = 0; klass < w.class_draws.size(); ++klass) {
        if (w.class_draws[klass] == 0) {
            continue;
        }
        const double class_draws = static_cast<double>(w.class_draws[klass]);
        class_text += fmt::format(" {} {:.0f}/frame cross {:.1f}% any {:.1f}% stale {:.2f}%;",
                                  class_names[klass], class_draws / frames,
                                  100.0 * static_cast<double>(w.class_cross[klass]) / class_draws,
                                  100.0 * static_cast<double>(w.class_any[klass]) / class_draws,
                                  100.0 * static_cast<double>(w.class_stale[klass]) / class_draws);
    }
    LOG_INFO(Render_Vulkan, "Draw record by pipeline class:{}", class_text);
    {
        static constexpr std::array<const char*, 12> attr_names{
            "no draw of the pipeline", "draw kind/channel", "render targets", "pools/modes",
            "stream structure", "index format", "cbuf sizes", "handle words", "computed handles",
            "storage words", "nothing (key equal)", "word count"};
        std::string attr_text;
        const double attr_draws = static_cast<double>((std::max)(w.attr_draws, u64{1}));
        const double attr_comp = static_cast<double>((std::max)(w.attr_computed_draws, u64{1}));
        for (size_t category = 0; category < attr_names.size(); ++category) {
            if (w.attr[category] == 0) {
                continue;
            }
            attr_text += fmt::format(" {} {:.1f}% (only it {:.1f}%, computed-handle draws {:.1f}%);",
                                     attr_names[category],
                                     100.0 * static_cast<double>(w.attr[category]) / attr_draws,
                                     100.0 * static_cast<double>(w.attr_only[category]) / attr_draws,
                                     100.0 * static_cast<double>(w.attr_computed[category]) / attr_comp);
        }
        LOG_INFO(Render_Vulkan,
                 "Draw record misses ({:.0f} draws/frame with a key no earlier frame had) vs the "
                 "nearest draw of the pipeline in the previous frame:{}",
                 static_cast<double>(w.attr_draws) / frames, attr_text);
        std::string reasons;
        for (size_t reason = 0; reason < w.break_reasons.size(); ++reason) {
            if (w.break_reasons[reason] != 0) {
                reasons += fmt::format(" {} {:.0f}",
                                       reason < VideoCommon::Nx2FastDraw::reason_names.size()
                                           ? VideoCommon::Nx2FastDraw::reason_names[reason]
                                           : "?",
                                       static_cast<double>(w.break_reasons[reason]) / frames);
            }
        }
        std::vector<std::pair<u64, u32>> top;
        for (const auto& [method, count] : break_methods) {
            top.emplace_back(count, method);
        }
        std::ranges::sort(top, std::greater{});
        std::string methods;
        for (size_t i = 0; i < (std::min)(top.size(), size_t{10}); ++i) {
            methods += fmt::format(" {}:{:#x}:{:.0f}", top[i].second & 0xff, top[i].second >> 8,
                                   static_cast<double>(top[i].first) / frames);
        }
        break_methods.clear();
        LOG_INFO(Render_Vulkan,
                 "Draw record: full draws after a state change per frame by break reason:{} | top "
                 "(reason:method:per frame):{} | correct hits by bindings re-resolved 0/1/2/3+: "
                 "{:.1f}/{:.1f}/{:.1f}/{:.1f}% of draws (cross-frame {:.1f}/{:.1f}/{:.1f}/{:.1f}%)",
                 reasons, methods, pct(w.tiers[0]), pct(w.tiers[1]), pct(w.tiers[2]),
                 pct(w.tiers[3]), pct(w.tiers_cross[0]), pct(w.tiers_cross[1]),
                 pct(w.tiers_cross[2]), pct(w.tiers_cross[3]));
    }
    static constexpr std::array<const char*, 13> cause_names{
        "after ineligible",    "other kind",         "state changed (method/other)",
        "after submit",        "render pass/pipe",   "pipeline takes no delta",
        "handle buffer bound", "handle data written", "sampled image to prepare",
        "delta today",         "attribute formats",  "other register",
        "fast draws off"};
    std::string cause_text;
    for (size_t cause = 0; cause < w.cause_draws.size(); ++cause) {
        if (w.cause_draws[cause] == 0) {
            continue;
        }
        const double cause_draws = static_cast<double>(w.cause_draws[cause]);
        cause_text += fmt::format(" {} {:.0f}/frame cross {:.1f}% any {:.1f}%;", cause_names[cause],
                                  cause_draws / frames,
                                  100.0 * static_cast<double>(w.cause_cross[cause]) / cause_draws,
                                  100.0 * static_cast<double>(w.cause_any[cause]) / cause_draws);
    }
    LOG_INFO(Render_Vulkan,
             "Draw record by today's path:{} | same render target group and targets as the "
             "previous draw {:.1f}%, same pipeline {:.1f}%",
             cause_text, pct(w.same_targets), pct(w.same_pipeline_any));
    const u64 now_ticks = Profile::Ticks();
    const u64 now_ns = Profile::Now();
    const double ns_per_tick =
        now_ticks > w.start_ticks
            ? static_cast<double>(now_ns - w.start_ns) / static_cast<double>(now_ticks - w.start_ticks)
            : 0.0;
    const double samples = static_cast<double>((std::max)(w.samples, u64{1}));
    const auto ns = [&](u64 ticks) { return static_cast<double>(ticks) * ns_per_tick / samples; };
    LOG_INFO(Render_Vulkan,
             "Draw record vs delta draws per frame: delta+hit {:.0f}, delta+no hit {:.0f}, full+hit "
             "{:.0f}, full+no hit {:.0f} | indirect {:.1f}%, inline index {:.1f}%, repacked "
             "{:.1f}%, writes storage {:.1f}%, failed {:.0f} | key ns: groups {:.1f}, data words {:.1f} ({:.1f} words), "
             "lookup {:.1f}, total {:.1f}; outcome and compare (dry only) {:.1f} | rehashed: "
             "registers {:.1f}% (compared {:.1f}%), streams {:.1f}% | dynamic state dirty {:.1f}% | correct hits "
             "after an image event {:.1f}%, buffer topology change {:.1f}%, map change {:.1f}%, "
             "new GPU-modified range {:.1f}%, uniform bindings moved {:.1f}%, storage bindings "
             "moved {:.1f}%, a texture handle moved {:.1f}%, render targets moved {:.1f}% of hits",
             static_cast<double>(w.overlap[1][1]) / frames,
             static_cast<double>(w.overlap[1][0]) / frames,
             static_cast<double>(w.overlap[0][1]) / frames,
             static_cast<double>(w.overlap[0][0]) / frames, pct(w.indirect), pct(w.inline_index),
             pct(w.repacked), pct(w.writes_storage), static_cast<double>(w.failed), ns(w.ticks_regs), ns(w.ticks_data),
             static_cast<double>(w.data_words) / samples, ns(w.ticks_lookup),
             ns(w.ticks_regs + w.ticks_data + w.ticks_lookup), ns(w.ticks_outcome),
             pct(w.regs_rehash), pct(w.regs_compared), pct(w.streams_rehash), pct(w.dynamic_dirty),
             correct ? 100.0 * static_cast<double>(w.hit_tex_event) / static_cast<double>(correct) : 0.0,
             correct ? 100.0 * static_cast<double>(w.hit_buf_epoch) / static_cast<double>(correct) : 0.0,
             correct ? 100.0 * static_cast<double>(w.hit_map) / static_cast<double>(correct) : 0.0,
             correct ? 100.0 * static_cast<double>(w.hit_gpu_modified) / static_cast<double>(correct) : 0.0,
             correct ? 100.0 * static_cast<double>(w.uniform_bindings_changed) / static_cast<double>(correct) : 0.0,
             correct ? 100.0 * static_cast<double>(w.storage_bindings_changed) / static_cast<double>(correct) : 0.0,
             correct ? 100.0 * static_cast<double>(w.handle_moved_hits) / static_cast<double>(correct) : 0.0,
             correct ? 100.0 * static_cast<double>(w.targets_moved) / static_cast<double>(correct + stale) : 0.0);
    const auto configure_ns = [&](size_t is_delta, size_t is_hit) {
        const u64 count = w.configure_samples[is_delta][is_hit];
        return count != 0 ? static_cast<double>(w.configure_ticks[is_delta][is_hit]) * ns_per_tick /
                                static_cast<double>(count)
                          : 0.0;
    };
    {
        const auto phase_ns = [&](size_t row, size_t i) {
            return w.phase_samples[row] != 0
                       ? static_cast<double>(w.phases[row][i]) * ns_per_tick /
                             static_cast<double>(w.phase_samples[row])
                       : 0.0;
        };
        LOG_INFO(Render_Vulkan,
                 "Draw record ConfigureDelta phases ns (delta draw | record hit): storage binds "
                 "{:.0f} | {:.0f}, buffer updates {:.0f} | {:.0f}, acquire+stage buffers+images "
                 "{:.0f} | {:.0f}, barrier {:.0f} | {:.0f}, render targets {:.0f} | {:.0f}, "
                 "ConfigureDraw {:.0f} | {:.0f} (samples {} | {}); Configure ns full {:.0f}, "
                 "delta {:.0f}",
                 phase_ns(0, 0), phase_ns(1, 0), phase_ns(0, 1), phase_ns(1, 1), phase_ns(0, 2),
                 phase_ns(1, 2), phase_ns(0, 3), phase_ns(1, 3), phase_ns(0, 4), phase_ns(1, 4),
                 phase_ns(0, 5), phase_ns(1, 5), w.phase_samples[0], w.phase_samples[1],
                 configure_ns(0, 0), configure_ns(1, 0));
    }
    if (w.flips != 0) {
        const auto share = [](u64 part, u64 whole) {
            return whole != 0 ? 100.0 * static_cast<double>(part) / static_cast<double>(whole) : 0.0;
        };
        LOG_INFO(Render_Vulkan,
                 "Draw record A/B: {} flips (table dropped at each) | hits in the first frame after "
                 "a flip {:.1f}% of its draws, later frames {:.1f}%",
                 w.flips, share(w.warm_hits[0], w.warm_draws[0]), share(w.warm_hits[1], w.warm_draws[1]));
    }
    if (Record::GetMode() >= Record::Mode::Verify) {
        const auto per_frame = [&](u64 value) { return static_cast<double>(value) / frames; };
        const auto avg_ns = [&](const std::array<u64, 2>& sampled) {
            return sampled[0] != 0 ? static_cast<double>(sampled[1]) * ns_per_tick /
                                         static_cast<double>(sampled[0])
                                   : 0.0;
        };
        LOG_INFO(Render_Vulkan,
                 "Draw record hits per frame: ready {:.0f} ({:.1f}% of draws; bindings resolved "
                 "again {:.0f}, all after an image event {:.0f}, image descriptors replayed {:.0f}, of "
                 "them patched {:.0f}) "
                 "| rejected: no record {:.0f}, ineligible {:.0f}, map {:.0f}, layout {:.0f}, "
                 "images to prepare {:.0f}, new key {:.0f}, indirect {:.0f} | verified {:.0f}, mismatches views {} "
                 "samplers {} image descriptors {} rescaling {} stages {} (whole run: {} {} {} {} {}); "
                 "full path's cached descriptors older than the hit's {} (whole run {}, verified {}) | "
                 "TIC/TSC words changed "
                 "under a kept handle {:.0f} | forced tests: image event {}, rescale {}, "
                 "descriptor {} | ns: prepare {:.0f}, ConfigureDelta of a hit {:.0f}",
                 per_frame(w.hit_ready), pct(w.hit_ready), per_frame(w.hit_bindings_moved),
                 per_frame(w.hit_clock_moved), per_frame(w.hit_images_replayed),
                 per_frame(w.hit_images_patched),
                 per_frame(w.hit_rejects[0]), per_frame(w.hit_rejects[1]),
                 per_frame(w.hit_rejects[2]), per_frame(w.hit_rejects[3]),
                 per_frame(w.hit_rejects[4]), per_frame(w.hit_rejects[5]),
                 per_frame(w.hit_rejects[6]), per_frame(w.verified),
                 w.verify_mismatch[0], w.verify_mismatch[1], w.verify_mismatch[2],
                 w.verify_mismatch[3], w.verify_mismatch[4], total_mismatches[0],
                 total_mismatches[1], total_mismatches[2], total_mismatches[3],
                 total_mismatches[4], w.verify_full_stale, total_full_stale, total_verified,
                 per_frame(w.hit_descriptor_moved),
                 w.tests[0], w.tests[1], w.tests[2], avg_ns(w.prepare_ticks),
                 avg_ns(w.record_configure));
    }
    LOG_INFO(Render_Vulkan,
             "Draw record: Configure ns per draw (sampled, dry hooks included): full+hit {:.0f}, "
             "full+no hit {:.0f}, delta+hit {:.0f}, delta+no hit {:.0f}",
             configure_ns(0, 1), configure_ns(0, 0), configure_ns(1, 1), configure_ns(1, 0));
    const auto runs = [&](const std::array<u64, 8>& histogram) {
        std::string text;
        for (size_t bucket = 0; bucket < 7; ++bucket) {
            text += fmt::format("{}{:.1f}", bucket == 0 ? "" : "/", pct(histogram[bucket]));
        }
        return text;
    };
    const double same = static_cast<double>((std::max)(w.same_pipeline, u64{1}));
    const auto same_pct = [&](u64 value) { return 100.0 * static_cast<double>(value) / same; };
    LOG_INFO(Render_Vulkan,
             "Draw record table: {} keys, new keys/frame {:.1f} (max {}), clears {} | M4: library "
             "draws {:.1f}%, library set writes today {:.0f}/frame ({:.1f}% of library draws) | "
             "payload entries/draw {:.1f}, uniform entries/draw {:.1f} | vs previous draw with "
             "the same pipeline ({:.1f}% of draws): identical {:.1f}%, buffer offsets only "
             "{:.1f}%, buffer handles {:.1f}%, texel views {:.1f}%, image views/samplers {:.1f}%; "
             "changed entries/draw {:.2f} (uniform offsets {:.2f}) | B1: set binds {:.0f}/frame, "
             "distinct sets/frame {:.0f} ({:.1f}% used in an earlier frame), new sets/frame "
             "{:.1f} | runs, % of draws in runs of 1/2/3-4/5-8/9-16/17-64/65+: same key {}, same "
             "key+payload {}, same key+sets {} | unknown layout {}",
             table.size(), static_cast<double>(w.new_keys) / frames, w.max_new_keys_frame,
             table_clears, pct(w.gpl_draws), static_cast<double>(w.gpl_updates) / frames,
             w.gpl_draws ? 100.0 * static_cast<double>(w.gpl_updates) / static_cast<double>(w.gpl_draws) : 0.0,
             static_cast<double>(w.payload_entries) / draws,
             static_cast<double>(w.uniform_entries) / draws, pct(w.same_pipeline),
             same_pct(w.payload_identical), same_pct(w.offsets_only), same_pct(w.buffer_handles),
             same_pct(w.texel_changed), same_pct(w.views_changed),
             static_cast<double>(w.changed_entries) / same,
             static_cast<double>(w.changed_uniform_offsets) / same,
             static_cast<double>(w.set_changes) / frames,
             static_cast<double>(w.sets_first_use) / frames,
             w.sets_first_use ? 100.0 * static_cast<double>(w.sets_cross_frame) / static_cast<double>(w.sets_first_use) : 0.0,
             static_cast<double>(w.sets_new) / frames, runs(w.run_key), runs(w.run_full),
             runs(w.run_set), w.unknown_layout);
    // Raw counts for scripts.
    std::string cause_raw;
    for (size_t cause = 0; cause < w.cause_draws.size(); ++cause) {
        cause_raw += fmt::format("{}{}:{}:{}", cause == 0 ? "" : ",", w.cause_draws[cause],
                                 w.cause_cross[cause], w.cause_any[cause]);
    }
    std::string stale_raw;
    for (size_t reason = 0; reason < w.stale.size(); ++reason) {
        stale_raw += fmt::format("{}{}", reason == 0 ? "" : ",", w.stale[reason]);
    }
    std::string class_raw;
    for (size_t klass = 0; klass < w.class_draws.size(); ++klass) {
        class_raw += fmt::format("{}{}:{}:{}:{}", klass == 0 ? "" : ",", w.class_draws[klass],
                                 w.class_cross[klass], w.class_any[klass], w.class_stale[klass]);
    }
    LOG_INFO(Render_Vulkan,
             "DRAWREC frame={} frames={} draws={} miss={} same={} prev={} older={} stale_same={} "
             "stale_prev={} stale_older={} stale_classes={} classes={} delta_hit={} delta_nohit={} "
             "full_hit={} full_nohit={} ns_groups={:.1f} ns_data={:.1f} ns_lookup={:.1f} "
             "ns_outcome={:.1f} words={:.1f} keys={} new_keys={} min_frame_cross={:.4f} "
             "mean_frame_cross={:.4f} gpl={} gpl_updates={} same_pipeline={} identical={} "
             "offsets_only={} handles={} views={} texels={} set_changes={} sets_first={} "
             "sets_cross={} run_key={} run_full={} run_set={} level0={} cfg_full_hit={:.1f} cfg_full_miss={:.1f} "
             "cfg_delta_hit={:.1f} cfg_delta_miss={:.1f} causes={} same_targets={} same_pipe={} "
             "uniform_moved={} storage_moved={} tex_event_hits={} buf_epoch_hits={} tiers={} "
             "tiers_cross={} attr={} attr_only={} attr_computed={} attr_draws={} "
             "attr_computed_draws={} handle_moved={} targets_moved={} moved_same_view={} "
             "moved_new_view={} moved_cycle2={} bindings_compared={}",
             frame, w.frames, w.draws, hits(Hit::Miss), hits(Hit::SameFrame), hits(Hit::Previous),
             hits(Hit::Older), stales(Hit::SameFrame), stales(Hit::Previous), stales(Hit::Older),
             stale_raw, class_raw, w.overlap[1][1], w.overlap[1][0], w.overlap[0][1],
             w.overlap[0][0], ns(w.ticks_regs), ns(w.ticks_data), ns(w.ticks_lookup),
             ns(w.ticks_outcome), static_cast<double>(w.data_words) / samples, table.size(),
             w.new_keys, w.min_frame_cross > 1.0 ? 0.0 : w.min_frame_cross, mean_cross,
             w.gpl_draws, w.gpl_updates, w.same_pipeline, w.payload_identical, w.offsets_only,
             w.buffer_handles, w.views_changed, w.texel_changed, w.set_changes, w.sets_first_use,
             w.sets_cross_frame, fmt::join(w.run_key, ":"), fmt::join(w.run_full, ":"),
             fmt::join(w.run_set, ":"), w.level0, configure_ns(0, 1), configure_ns(0, 0), configure_ns(1, 1),
             configure_ns(1, 0), cause_raw, w.same_targets, w.same_pipeline_any,
             w.uniform_bindings_changed, w.storage_bindings_changed, w.hit_tex_event,
             w.hit_buf_epoch, fmt::join(w.tiers, ":"), fmt::join(w.tiers_cross, ":"),
             fmt::join(w.attr, ":"), fmt::join(w.attr_only, ":"), fmt::join(w.attr_computed, ":"),
             w.attr_draws, w.attr_computed_draws, w.handle_moved_hits, w.targets_moved,
             w.moved_same_view, w.moved_new_view, w.moved_cycle2, w.bindings_compared);
    ResetWindow();
}

} // namespace Vulkan
