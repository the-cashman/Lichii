// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <memory>
#include <vector>

#include "common/common_types.h"
#include "common/container/unordered_map.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/nx2_draw_record.h"
#include "video_core/renderer_vulkan/vk_buffer_cache.h"
#include "video_core/renderer_vulkan/vk_texture_cache.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "video_core/renderer_vulkan/vk_update_descriptor.h"

namespace Tegra {
class MemoryManager;
}

namespace Vulkan {

class Device;
class GraphicsPipeline;

/// Draw-record cache, stage A1 step 1 (EDEN_NX2_DRAW_RECORD=dry, nx2_draw_record.h): next to
/// every full or delta resolve on the split's back, computes the draw key from the resolve's
/// inputs, looks it up in a table of the outcomes earlier draws with that key got, and counts
/// whether the stored outcome equals the one the resolve just gave (a hit that would have been
/// correct) or not (a hit that would have been stale: classified by the validation stamp that
/// moved, or logged as a key hole). Also counts what stage B needs (descriptor payload changes
/// per draw, runs of equal draws, distinct descriptor sets). Changes nothing the draw does.
///
/// Key (what a full resolve reads that isn't per-draw data):
///   G0 pipeline (GraphicsPipeline*, which stands for the shaders and the fixed pipeline state),
///      draw kind (indexed, indirect, topology, inline index, repacked vertices), channel (3D
///      engine), address space;
///   G1 render targets, depth target, surface clip, TIC/TSC pools and limits, sampler binding
///      mode, transform feedback enable, sample mode: hashed again only when the fast draw serial
///      (any non-parameter register write, side-effect method, frame, ...) moved;
///   G2 vertex streams (address, limit or native size, stride/enable word): hashed again only
///      when Dirty::VertexBuffers is set;
///   G3 index buffer range and format (not first/count: draw parameters);
///   G4 per stage the constant buffer table entries the pipeline reads (uniform, handle and
///      storage pointer slots): address, size, enabled;
///   G5 data words: the texture/image handle words read_handle reads (secondary words included),
///      the handles computed-handle programs evaluate to, the storage buffer pointer words (and
///      the table word for indirect ones, the size word where the binding reads it).
class DrawRecorder {
public:
    explicit DrawRecorder(const Device& device, BufferCache& buffer_cache,
                          TextureCache& texture_cache, GuestDescriptorQueue& descriptor_queue);
    ~DrawRecorder();

    /// With the caches locked, before Configure/ConfigureDelta: computes the key.
    void BeginDraw(GraphicsPipeline& pipeline, Tegra::Engines::Maxwell3D& maxwell3d,
                   Tegra::MemoryManager& gpu_memory, bool is_indexed, bool is_indirect,
                   bool repacked);
    /// Whether any dynamic state the draw sets is dirty (before UpdateDynamicStates).
    void NoteDynamicDirty(bool dirty) noexcept {
        dynamic_dirty = dirty;
    }
    /// After Configure/ConfigureDelta succeeded: compares and stores the outcome.
    /// full_reason: RasterizerVulkan::FullDraw of a full draw (9: delta, 0xff: fast draws off),
    /// break_method: the fast draw serial's last break (reason | method << 8).
    void EndDraw(GraphicsPipeline& pipeline, Tegra::MemoryManager& gpu_memory, bool delta,
                 u8 full_reason, u32 break_method, bool record_used);
    /// At the frame tick.
    void EndFrame();

    /// Verify and on: whether the key's record can stand for a full resolve of this draw. Reads
    /// the texture handles, resolves the bindings whose handle moved (or all after an image
    /// event) and checks the sampled images need no preparing. With EDEN_NX2_DRAW_RECORD=1 it
    /// also puts the record's bindings into the pipeline for ConfigureDelta.
    bool PrepareHit(GraphicsPipeline& pipeline, Tegra::Engines::Maxwell3D& maxwell3d,
                    Tegra::MemoryManager& gpu_memory);
    /// Before a full Configure: keep what it resolves (verify, on).
    void BeforeFull(GraphicsPipeline& pipeline) const;
    [[nodiscard]] bool UsesHits() const noexcept {
        return mode == VideoCommon::Nx2DrawRecord::Mode::On && active;
    }
    /// At a frame boundary (in-session A/B): with false, draws take today's paths and the
    /// recorder does nothing; the table is kept (every hit validates its record again).
    void SetActive(bool on) noexcept {
        active = on;
    }

private:
    static constexpr size_t NUM_STAGES = 5;

    enum class EntryKind : u8 { Uniform, Storage, Texel, ImageTexel, Sampled, Image, Other };

    struct WordOp {
        enum class Kind : u8 { Handle, HandlePair, Computed, Storage, Null };
        Kind kind{};
        u8 stage{};
        u8 cbuf{};
        u8 cbuf2{};
        u8 shift{};
        u8 shift2{};
        bool written{};
        bool sampled{}; ///< a texture element: its handle names a sampler too
        bool repeat{};  ///< unused since #1485 (each element is computed: HandleOp::Index)
        u32 offset{};
        u32 offset2{};
        u32 program{};
        u32 indirect{~0U};
        u32 element{}; ///< Computed: the descriptor array element (#1485's HandleOp::Index)
    };

    enum class PipeClass : u8 { Mono, Gpl, ComputedMono, ComputedGpl, Mesh, WritesImages, Count };

    struct Plan {
        std::array<u64, 6> shader_hashes{};
        u64 generation{};
        std::array<u32, NUM_STAGES> cbuf_slots{};
        std::vector<WordOp> words;
        std::vector<WordOp> params; ///< storage pointer reads that are per-draw inputs
        std::vector<EntryKind> kinds;
        u32 vertex_entries{}; ///< library pipelines: the vertex set's entries come first
        PipeClass klass{};
        bool gpl{};
        bool texture_buffers{}; ///< texture or image buffers (not taken by the hit path yet)
        u64 layout_hash{};  ///< the payload's entry kinds (sets of equal layouts can be shared)
        std::vector<s16> sampler_of; ///< per view element: its sampler index, -1 none
        /// Per view element: its kept image descriptor (index into the kept entries, -1 for
        /// texture/image buffers) and what PushImageDescriptors makes it from.
        struct ElementDescriptor {
            s16 entry{-1};
            bool image{};    ///< storage image (else combined image sampler)
            bool is_depth{};
            bool read_from_memory{}; ///< a handle program's result (#1485's IsUncomparable)
            bool byte_addressed{};
            Shader::TextureType type{};
            Shader::ImageFormat format{};
        };
        std::vector<ElementDescriptor> element_descriptors;
        u64 last_payload{}; ///< full payload hash of this pipeline's last draw
        bool has_last_payload{};
    };

    /// Outcome parts compared on a hit.
    enum Out : u8 {
        Views,
        Geometry,
        Uniform,
        UniformStream,
        Storage,
        TexBuffers,
        Targets,
        Payload,
        NUM_OUT
    };

    struct Entry {
        std::array<u64, NUM_OUT> out{};
        u64 tex_clock{};
        u64 scale_epoch{};
        u64 buf_epoch{};
        u64 map_serial{};
        u64 gpu_modified{};
        u64 seen{};       ///< bit n: used n frames before last_frame
        u64 last_frame{};
        std::vector<u64> bindings;
        std::vector<u64> older_bindings; ///< the bindings before the last change
        std::vector<u64> params;           ///< per-draw inputs, one hash per binding
        std::vector<u64> storage_bindings; ///< for storage hole diagnostics
        // The record proper (verify, on): what a hit puts back into the pipeline.
        std::vector<VideoCommon::ImageViewInOut> views;
        std::vector<VideoCommon::SamplerId> samplers;
        std::vector<u64> handles; ///< raw handle per view element (cur_values)
        std::vector<u64> descriptors; ///< hash of each element's TIC (+ TSC) entry words
        /// Per element: where its TIC (and TSC) entry is and the 32 bytes it held (a hit compares
        /// them; the pointers stand while the map serial does).
        struct EntryCopy {
            const u8* tic{};
            const u8* tsc{};
            std::array<u64, 4> tic_words{};
            std::array<u64, 4> tsc_words{};
        };
        std::vector<EntryCopy> entry_copies;
        std::vector<u8> rescaled; ///< per view element: its view was rescaled when stored
        u64 version{}; ///< bumped whenever the record changes (the pipeline's owner tag)
        std::vector<u32> sampler_handles; ///< TSC index of each sampled element
        std::vector<DescriptorUpdateEntry> image_entries;
        std::array<std::pair<u32, u32>, NUM_STAGES> image_ranges{};
        std::array<u32, Shader::Backend::SPIRV::NUM_TEXTURE_AND_IMAGE_SCALING_WORDS> rescaling_words{};
        u64 kept_scale_epoch{};
        std::array<bool, NUM_STAGES> stages{};
        bool images_valid{};
        bool has_record{};
        u64 record_map_serial{};
        u64 record_tex_clock{};
        bool rescaling{};
        bool valid{};
    };

    enum class Hit : u8 { Miss, SameFrame, Previous, Older, Count };

    enum class Stale : u8 {
        TexEvent,       ///< an image event (binding clock) since the record was stored
        TicData,        ///< same handles, other view or sampler, no image event: TIC/TSC words
        Rescale,        ///< scale epoch or rescaling moved
        BufferTopology, ///< a buffer created, deleted or joined
        MemoryMap,      ///< GPU mappings changed
        UniformPath,    ///< a uniform buffer flipped between streamed and classic
        HoleHandle,
        HoleTargets,
        HoleGeometry,
        HoleUniform,
        HoleStorage,
        HoleTexBuffers,
        HolePayload,
        Count,
    };

    struct ChannelGroups {
        const void* maxwell3d{};
        u64 serial{};
        u64 regs{};
        u64 streams{};
        u64 targets{};
        bool targets_valid{};
        bool regs_valid{};
        std::vector<u8> regs_snapshot; ///< the G1 register blocks last hashed
        bool streams_valid{};
    };

    struct Window {
        u64 frames{};
        u64 draws{};
        u64 failed{};
        std::array<u64, static_cast<size_t>(Hit::Count)> hits{};
        std::array<u64, static_cast<size_t>(Hit::Count)> stale_hits{};
        std::array<u64, static_cast<size_t>(Stale::Count)> stale{};
        std::array<u64, static_cast<size_t>(PipeClass::Count)> class_draws{};
        std::array<u64, static_cast<size_t>(PipeClass::Count)> class_cross{};
        std::array<u64, static_cast<size_t>(PipeClass::Count)> class_any{};
        std::array<u64, static_cast<size_t>(PipeClass::Count)> class_stale{};
        std::array<std::array<u64, 2>, 2> overlap{}; ///< [delta today][correct record hit]
        u64 indirect{};
        u64 inline_index{};
        u64 repacked{};
        u64 writes_storage{};
        /// By why today's path resolved in full (RasterizerVulkan::FullDraw, then delta, then
        /// "state changed" split: vertex attribute format registers / other registers / other
        /// breaks): draws, correct cross-frame hits, correct hits.
        std::array<u64, 13> cause_draws{};
        std::array<u64, 13> cause_cross{};
        std::array<u64, 13> cause_any{};
        /// Draws whose render target group (G1) and render targets equal the previous draw's
        /// (UpdateRenderTargets skippable), whose pipeline is the previous draw's.
        u64 same_targets{};
        u64 same_pipeline_any{};
        u64 level0{}; ///< draws with the previous draw's key (no lookup)
        u64 regs_rehash{};
        u64 regs_compared{};
        u64 streams_rehash{};
        u64 dynamic_dirty{};
        // Validation work a correct hit would have needed.
        u64 hit_tex_event{};
        u64 hit_buf_epoch{};
        u64 hit_map{};
        u64 hit_gpu_modified{};
        u64 uniform_bindings_changed{}; ///< correct hits whose uniform bindings (per draw) moved
        u64 storage_bindings_changed{}; ///< ... storage bindings (per draw unless in the key)
        /// Correct hits by bindings re-resolved because their per-draw inputs (vertex stream
        /// and index addresses, storage pointers) moved: 0, 1, 2, 3+; cross-frame ones apart.
        u64 handle_moved_hits{};
        /// Texture/image bindings of hits whose handle moved: same view id anyway, another
        /// view; the new handle equals the one before the record's last change (cycle 2).
        u64 moved_same_view{};
        u64 moved_new_view{};
        u64 moved_cycle2{};
        u64 bindings_compared{};
        u64 targets_moved{};
        std::array<u64, 4> tiers{};
        std::array<u64, 4> tiers_cross{};
        /// Draws whose key no earlier frame had, by what differs from the nearest draw of the
        /// same pipeline in the previous frame (MissAttr), all classes and computed-handle ones.
        std::array<u64, 12> attr{};
        std::array<u64, 12> attr_only{};
        std::array<u64, 12> attr_computed{};
        u64 attr_draws{};
        u64 attr_computed_draws{};
        /// Full draws after a fast draw state change, by Nx2FastDraw::Reason, and methods.
        std::array<u64, 16> break_reasons{};
        // Hit path (verify, on).
        u64 hit_ready{};          ///< PrepareHit gave a usable record
        u64 hit_bindings_moved{}; ///< of them, with bindings resolved again
        u64 hit_clock_moved{};    ///< of them, all bindings resolved again (image event)
        u64 hit_images_replayed{}; ///< kept image descriptors pushed as they were
        u64 hit_images_patched{};  ///< of them, with the moved bindings' descriptors made again
        u64 flips{};              ///< A/B phase changes (the table is dropped at each)
        std::array<u64, 2> warm_draws{}; ///< draws [first frame after a flip into a hit mode, later]
        std::array<u64, 2> warm_hits{};
        std::array<u64, 7> hit_rejects{}; ///< no record, ineligible, map, layout, prepare, new, indirect
        u64 verified{};
        std::array<u64, 5> verify_mismatch{}; ///< views, samplers, image descriptors, rescaling, stages
        u64 hit_descriptor_moved{}; ///< hits with a binding whose TIC/TSC words changed
        u64 verify_full_stale{};    ///< verify: full-path descriptors older than the hit's
        std::array<u64, 3> tests{}; ///< forced: image event, rescale, descriptor change
        std::array<u64, 2> prepare_ticks{}; ///< [samples, ticks]
        std::array<u64, 2> record_configure{}; ///< ConfigureDelta of record hits [samples, ticks]
        /// ConfigureDelta phases of sampled [delta draws, record hits]: storage binds, buffer
        /// updates, acquire + stage buffers + images, (barrier), render targets, ConfigureDraw.
        std::array<std::array<u64, 6>, 2> phases{};
        std::array<u64, 2> phase_samples{};
        // Timing (time stamp counter, every 16th draw).
        u64 samples{};
        u64 ticks_regs{};
        u64 ticks_data{};
        u64 ticks_lookup{};
        u64 ticks_outcome{};
        u64 data_words{};
        /// Configure/ConfigureDelta time of sampled draws: [delta today][correct record hit].
        std::array<std::array<u64, 2>, 2> configure_ticks{};
        std::array<std::array<u64, 2>, 2> configure_samples{};
        u64 start_ticks{};
        u64 start_ns{};
        // Table.
        u64 new_keys{};
        u64 max_new_keys_frame{};
        // Frames.
        double min_frame_cross{2.0};
        double sum_frame_cross{};
        u64 counted_frames{};
        // M4: descriptor payload and runs.
        u64 gpl_draws{};
        u64 gpl_updates{}; ///< draws where today's library path allocates and writes both sets
        u64 payload_entries{};
        u64 uniform_entries{};
        u64 same_pipeline{};
        u64 payload_identical{};
        u64 offsets_only{};
        u64 buffer_handles{};
        u64 views_changed{};
        u64 texel_changed{};
        u64 changed_entries{};
        u64 changed_uniform_offsets{};
        u64 unknown_layout{};
        std::array<u64, 8> run_key{};      ///< draws in runs of equal keys, by run length bucket
        std::array<u64, 8> run_full{};     ///< ... equal key and equal payload (multi-draw)
        std::array<u64, 8> run_set{};      ///< ... equal key and equal sets (offsets differ)
        u64 set_changes{};                 ///< draws needing another persistent set than the last
        u64 sets_first_use{};              ///< distinct sets per frame (summed)
        u64 sets_cross_frame{};            ///< of those: used in an earlier frame
        u64 sets_new{};
    };

    Plan& PlanFor(GraphicsPipeline& pipeline);
    void BuildPlan(GraphicsPipeline& pipeline, Plan& plan);
    ChannelGroups& Channel(const Tegra::Engines::Maxwell3D& maxwell3d);
    static size_t RunBucket(u64 length) noexcept;
    void ResetWindow();
    u64 HashRegs(const Tegra::Engines::Maxwell3D& maxwell3d) const;
    static void SnapshotRegs(const Tegra::Engines::Maxwell3D& maxwell3d, std::vector<u8>& out);
    u64 HashStreams(Tegra::Engines::Maxwell3D& maxwell3d) const;
    u64 HashData(const Plan& plan, GraphicsPipeline& pipeline, Tegra::Engines::Maxwell3D& maxwell3d,
                 Tegra::MemoryManager& gpu_memory);
    Stale Classify(const Entry& entry, const std::array<u64, NUM_OUT>& out,
                   const std::vector<u64>& bindings, bool rescaling, u64 tex_clock, u64 scale,
                   u64 buf_epoch, u64 map_serial) const;
    void LogHole(Stale stale, const GraphicsPipeline& pipeline, const Entry& entry,
                 const std::array<u64, NUM_OUT>& out, const std::vector<u64>& bindings);
    void TrackPayload(Plan& plan, GraphicsPipeline& pipeline,
                      const DescriptorUpdateEntry* entries, size_t count);
    void EndRuns();
    void Report();

    const Device& device;
    BufferCache& buffer_cache;
    TextureCache& texture_cache;
    GuestDescriptorQueue& descriptor_queue;

    /// Owned here, pointed to by GraphicsPipeline::record_plan (stable addresses).
    Common::unordered_map<const GraphicsPipeline*, std::unique_ptr<Plan>> plans;
    u64 plan_generation{};
    std::array<ChannelGroups, 8> channels{};
    /// Host pointers of guest pages the data words were read from (what a record would keep):
    /// direct mapped, by GPU page, valid for one memory manager and map serial.
    struct PageSlot {
        const Tegra::MemoryManager* memory{};
        u64 map_serial{};
        GPUVAddr page{~GPUVAddr{}};
        const u8* host{};
    };
    std::array<PageSlot, 256> pages{};
    template <typename T>
    T ReadGuest(Tegra::MemoryManager& gpu_memory, GPUVAddr address);
    const u8* PagePointer(Tegra::MemoryManager& gpu_memory, GPUVAddr address);
    friend struct RecordMemory;
    u32 channel_cursor{};
    std::vector<u8> scratch_regs;
    /// Per draw, for the miss attribution: the key's groups (MissAttr order), data word values.
    struct Sig {
        const GraphicsPipeline* pipeline{};
        std::array<u64, 7> groups{};
        u32 values_first{};
        u32 values_count{};
    };
    std::vector<Sig> frame_sigs;
    std::vector<Sig> prev_sigs;
    std::vector<u64> frame_values;
    std::vector<u64> prev_values;
    Common::unordered_map<const GraphicsPipeline*, std::vector<u32>> prev_by_pipeline;
    Common::unordered_map<u32, u64> break_methods;
    std::array<u64, 7> cur_groups{};
    std::vector<u64> cur_values;
    std::vector<u64> cur_params;
    void Attribute(size_t klass, const GraphicsPipeline* pipeline);
    Common::unordered_map<VideoCommon::Nx2DrawRecord::Key, Entry,
                          VideoCommon::Nx2DrawRecord::KeyHash>
        table;
    u64 table_clears{};

    // The draw between BeginDraw and EndDraw.
    VideoCommon::Nx2DrawRecord::Key key{};
    Entry* entry{};
    Entry* level0_entry{}; ///< the entry of `key` (the previous draw's)
    bool entry_new{};
    Plan* plan{};
    bool sample{};
    bool begun{};
    bool dynamic_dirty{};
    bool cur_indirect{};
    bool cur_inline{};
    bool cur_repacked{};
    u64 begin_ticks_regs{};
    u64 begin_ticks_data{};
    u64 begin_ticks_lookup{};
    u64 begin_end{};
    u64 cur_regs{};
    u64 prev_regs{};
    u64 prev_targets{};
    u64 draw_counter{};

    u64 frame{1};
    // Per frame.
    u64 frame_draws{};
    u64 frame_cross{};
    u64 frame_any{};
    u64 frame_stale{};
    u64 frame_new{};

    // M4 state: the previous draw.
    const GraphicsPipeline* prev_pipeline{};
    VideoCommon::Nx2DrawRecord::Key prev_key{};
    std::vector<DescriptorUpdateEntry> prev_payload;
    u64 prev_full{};
    std::array<u64, 2> prev_sets{};
    bool have_prev{};
    u64 run_key_len{};
    u64 run_full_len{};
    u64 run_set_len{};
    Common::unordered_map<u64, u64> set_frames; ///< set hash -> last frame used

    Window window;
    u64 report_frames{};
    u64 holes_left{};
    bool log_frames{};
    bool storage_in_key{};
    VideoCommon::Nx2DrawRecord::Mode mode{};
    bool stats{};
    bool active{true};
    u32 phase_serial{~0U}; ///< Common::Nx2Ab::PhaseSerial at the last mode refresh
    u64 frames_since_flip{};
    u64 owner_epoch{1}; ///< bumped when the table is cleared: owner tags of older records fail
    u32 hit_moved{};
    // PrepareHit's result for the current draw.
    bool hit_prepared{};
    bool hit_images_valid{};
    std::vector<VideoCommon::ImageViewInOut> hit_views;
    std::vector<VideoCommon::SamplerId> hit_samplers;
    std::vector<u16> moved_elements; ///< the hit's bindings resolved again
    std::vector<DescriptorUpdateEntry> hit_entries; ///< verify: the patched kept descriptors
    /// The kept image descriptors of the moved elements made again in `entries` (as
    /// PushImageDescriptors makes them); false when one can't be (its rescaling changed).
    bool PatchImageEntries(const Plan& draw_plan, const std::vector<VideoCommon::ImageViewInOut>& views,
                           const std::vector<VideoCommon::SamplerId>& samplers,
                           std::vector<DescriptorUpdateEntry>& entries);
    void CopyEntry(Tegra::MemoryManager& gpu_memory, size_t element, u32 tic, u32 tsc,
                   bool sampled, Entry::EntryCopy& out);
    [[nodiscard]] static bool EntryHolds(const Entry::EntryCopy& copy) noexcept;
    u32 verify_logged{};
    u32 descriptor_logged{};
    std::array<u64, 5> total_mismatches{}; ///< verify, whole run
    u64 total_full_stale{};
    u64 total_verified{};
    u64 total_full_stale_visible{}; ///< of them: sampler variant only, on a view with mips (or min != mag)
    u32 stale_logged{};
    u64 totals_logged_sum{}; ///< sum of the totals at the last "verify totals" line
    u32 totals_lines{};
    u64 test_every{};     ///< EDEN_NX2_DRAW_RECORD_TEST=<n>: force a fallback every n hits
    u64 test_counter{};
    bool test_break{};    ///< EDEN_NX2_DRAW_RECORD_TEST_BREAK=1: keep moved bindings (verify must fail)
    Tegra::Engines::Maxwell3D* cur_maxwell3d{};
    GraphicsPipeline* cur_pipeline{};
    Tegra::MemoryManager* cur_memory{};
    bool values_read{};
    std::array<u64, 7> phase_stamps{};
    void EnsureHandles();
    u64 DescriptorHash(Tegra::MemoryManager& gpu_memory, u32 tic, u32 tsc, bool sampled);
    void VerifyHit(GraphicsPipeline& pipeline);
    void StoreRecord(Entry& stored, GraphicsPipeline& pipeline,
                     Tegra::MemoryManager& gpu_memory_for_store, u64 map_serial, u64 tex_clock);
    bool handles_in_key{};
};

} // namespace Vulkan
