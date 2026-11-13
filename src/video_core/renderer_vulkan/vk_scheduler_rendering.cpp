// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// nx2 shaderwarm (docs/shader-precompile.md): dynamic rendering instances for the graphics pipeline
// library path (EDEN_NX2_GPL=1): its pipelines are made without a render pass, so every 3D draw is
// recorded inside vkCmdBeginRendering with the framebuffer's views. Not counted in Nx2Profile's
// render pass total.

#include <array>
#include <utility>

#include "video_core/renderer_vulkan/vk_query_cache.h" // first, as in vk_scheduler.cpp
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_texture_cache.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

// Lichii orders draws after compute writes with a barrier at render pass begin
// (compute_writes_pending); a base without it has nothing to take.
template <typename Self>
bool Scheduler::Nx2TakeComputeWrites(Self& self) {
    if constexpr (requires { self.compute_writes_pending; }) {
        return std::exchange(self.compute_writes_pending, false);
    } else {
        return false;
    }
}

void Scheduler::BeginRenderingImpl(const Framebuffer* framebuffer, u32 color_clear_mask,
                                   const std::array<VkClearValue, 8>* color_values,
                                   bool depth_stencil_clear, VkClearValue depth_stencil_value) {
    const VkExtent2D render_area = framebuffer->RenderArea();
    state.renderpass = framebuffer->RenderPass();
    state.framebuffer = framebuffer->Handle();
    state.render_area = render_area;
    state.dynamic_rendering = true;

    struct Attachments {
        std::array<VkRenderingAttachmentInfo, 8> colors;
        VkRenderingAttachmentInfo depth;
        VkRenderingAttachmentInfo stencil;
        u32 num_colors;
        u32 layers;
        bool has_depth;
        bool has_stencil;
    } att{};
    const auto make = [](VkImageView view, bool clear, const VkClearValue& value) {
        return VkRenderingAttachmentInfo{
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .pNext = nullptr,
            .imageView = view,
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
            .resolveMode = VK_RESOLVE_MODE_NONE,
            .resolveImageView = VK_NULL_HANDLE,
            .resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .loadOp = clear ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .clearValue = value,
        };
    };
    att.num_colors = framebuffer->NumColorSlots();
    for (u32 slot = 0; slot < att.num_colors; ++slot) {
        const bool clear = (color_clear_mask & (1u << slot)) != 0;
        att.colors[slot] = make(framebuffer->ColorView(slot), clear,
                                color_values ? (*color_values)[slot] : VkClearValue{});
    }
    att.has_depth = framebuffer->DepthView() != VK_NULL_HANDLE && framebuffer->FormatHasDepth();
    att.has_stencil = framebuffer->DepthView() != VK_NULL_HANDLE && framebuffer->FormatHasStencil();
    att.depth = make(framebuffer->DepthView(), depth_stencil_clear, depth_stencil_value);
    att.stencil = att.depth;
    att.layers = framebuffer->NumLayers();
    const bool after_compute = Nx2TakeComputeWrites(*this);
    Record([att, render_area, after_compute](vk::CommandBuffer cmdbuf) {
        if (after_compute) {
            // As in BeginRenderPassImpl: order the draws' reads after the compute writes.
            static constexpr VkMemoryBarrier COMPUTE_TO_GRAPHICS{
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
            };
            cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                   VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, 0, COMPUTE_TO_GRAPHICS);
        }
        const VkRenderingInfo rendering_info{
            .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
            .pNext = nullptr,
            .flags = 0,
            .renderArea = {.offset = {.x = 0, .y = 0}, .extent = render_area},
            .layerCount = att.layers,
            .viewMask = 0,
            .colorAttachmentCount = att.num_colors,
            .pColorAttachments = att.colors.data(),
            .pDepthAttachment = att.has_depth ? &att.depth : nullptr,
            .pStencilAttachment = att.has_stencil ? &att.stencil : nullptr,
        };
        cmdbuf.BeginRendering(rendering_info);
    });
    num_renderpass_images = framebuffer->NumImages();
    renderpass_images = framebuffer->Images();
    renderpass_image_ranges = framebuffer->ImageRanges();
}

} // namespace Vulkan
