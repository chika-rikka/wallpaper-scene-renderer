#include "CopyPass.hpp"
#include "SpecTexs.hpp"
#include "Utils/Logging.h"
#include "Utils/AutoDeletor.hpp"
#include "Resource.hpp"
#include "PassCommon.hpp"
#include "Msaa.hpp"

using namespace wallpaper::vulkan;

CopyPass::CopyPass(const Desc& desc): m_desc(desc) {}

CopyPass::~CopyPass() {};

std::string CopyPass::residencyKey() const {
    return "CopyPass|src=" + m_desc.src + "|dst=" + m_desc.dst;
}

void CopyPass::absorbResidencyGraphState(const VulkanPass& next_pass) {
    const auto* next = dynamic_cast<const CopyPass*>(&next_pass);
    if (next == nullptr) return;
    // Copy passes have no PSO or private buffers, but they do carry runtime visibility gates for
    // effect bypass paths. Reusing them keeps graph-diff accounting honest while allowing the newly
    // built topology to replace only the frame-local gate.
    m_desc.should_execute = next->m_desc.should_execute;
    m_desc.dest_draw_phase = next->m_desc.dest_draw_phase;
    m_desc.layer_id = next->m_desc.layer_id;
}

bool CopyPass::referencesRenderTarget(std::string_view render_target) const {
    // Copy passes only need a resource refresh when their source or destination render target was
    // resized/recreated. Skipping unrelated copies keeps text bridge updates from walking the
    // entire graph just to refresh handles that still point at valid images.
    return m_desc.src == render_target || m_desc.dst == render_target;
}

void CopyPass::prepare(Scene& scene, const Device& device, RenderingResources& rr) {
    (void)rr;
    if (scene.renderTargets.count(m_desc.src) == 0) {
        LOG_ERROR("%s not found", m_desc.src.c_str());
        return;
    }
    if (scene.renderTargets.count(m_desc.dst) == 0) {
        auto& rt                                   = scene.renderTargets.at(m_desc.src);
        scene.renderTargets[m_desc.dst]            = rt;
        scene.renderTargets[m_desc.dst].allowReuse = true;
    }

    std::array<std::string, 2>      textures    = { m_desc.src, m_desc.dst };
    std::array<ImageParameters*, 2> vk_textures = { &m_desc.vk_src, &m_desc.vk_dst };
    for (usize i = 0; i < textures.size(); i++) {
        auto& tex_name = textures[i];
        if (tex_name.empty()) continue;

        ImageParameters img;
        if (scene.renderTargets.count(tex_name) != 0) {
            // CopyPass operates on render targets, not material assets. Effect-local FBO names can
            // be plain authored names after uniquification, so the scene render-target table is the
            // correct capability check here rather than the `_rt_` prefix alone.
            auto& rt  = scene.renderTargets.at(tex_name);
            auto  opt = device.tex_cache().Query(tex_name, ToTexKey(rt), ! rt.allowReuse);
            if (opt.has_value())
                img = opt.value();
            else
                LOG_ERROR("query image from cache failed");
        } else {
            LOG_ERROR("can't copy image source");
            return;
        }
        *vk_textures[i] = img;
    }

    setPrepared();
};

void CopyPass::refreshResources(Scene& scene, const Device& device, RenderingResources&) {
    // Resource-only refreshes only need to look up the recreated source/destination render-target
    // handles. The copy pass has no pipeline state of its own, so re-querying the cache is enough
    // to make the next execute use the resized images without paying the full prepare cost.
    std::array<std::string, 2>      textures    = { m_desc.src, m_desc.dst };
    std::array<ImageParameters*, 2> vk_textures = { &m_desc.vk_src, &m_desc.vk_dst };
    for (usize i = 0; i < textures.size(); i++) {
        auto& tex_name = textures[i];
        if (tex_name.empty()) continue;

        if (scene.renderTargets.count(tex_name) == 0) {
            // Resource refresh follows the same render-target-table rule as prepare(), because
            // non-prefixed effect FBOs are still recreated and rebound through the texture cache.
            setPrepared(false);
            return;
        }
        auto& rt = scene.renderTargets.at(tex_name);
        auto  opt = device.tex_cache().Query(tex_name, ToTexKey(rt), !rt.allowReuse);
        if (!opt.has_value()) {
            setPrepared(false);
            return;
        }
        *vk_textures[i] = opt.value();
    }
}

void CopyPass::execute(const Device& device, RenderingResources& rr) {
    if (m_desc.should_execute && !m_desc.should_execute()) {
        // The render graph still declares this copy as an ordering edge, but runtime visibility can
        // make the actual transfer unnecessary for the current frame. Returning here is what keeps
        // visible toggles cheap and avoids rebuilding the graph just to skip one effect branch.
        releaseFinalReadTexs(device);
        return;
    }

    if (rr.scene != nullptr && m_desc.src == SpecTex_Default) {
        ResolveComposeMsaaIfNeeded(*rr.scene, device, rr);
    }

    auto& cmd = rr.command;
    auto& src = m_desc.vk_src;
    auto& dst = m_desc.vk_dst;

    if (! (src.handle && dst.handle)) {
        // A prepared copy with missing image handles cannot record a GPU read, but the frame has
        // still reached this pass's final-reader boundary. Release the graph-owned keys before the
        // diagnostic assert so non-assert builds do not pin reusable render targets indefinitely.
        releaseFinalReadTexs(device);
        assert(src.handle && dst.handle);
        return;
    }

    VkImageSubresourceRange srang {
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .baseMipLevel   = 0,
        .levelCount     = 1,
        .baseArrayLayer = 0,
        .layerCount     = 1,

    };
    VkImageCopy copy {
        .srcSubresource =
            VkImageSubresourceLayers {
                .aspectMask     = srang.aspectMask,
                .mipLevel       = 0,
                .baseArrayLayer = 0,
                .layerCount     = 1,
            },
        .dstSubresource =
            VkImageSubresourceLayers {
                .aspectMask     = srang.aspectMask,
                .mipLevel       = 0,
                .baseArrayLayer = 0,
                .layerCount     = 1,
            },
        .extent = { src.extent.width, src.extent.height, 1 },
    };
    {
        VkImageMemoryBarrier in_bar {
            .sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .pNext            = nullptr,
            .srcAccessMask    = VK_ACCESS_MEMORY_READ_BIT,
            .dstAccessMask    = VK_ACCESS_TRANSFER_READ_BIT,
            .oldLayout        = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            .newLayout        = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            .image            = src.handle,
            .subresourceRange = srang,
        };
        VkImageMemoryBarrier out_bar {
            .sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .pNext            = nullptr,
            .srcAccessMask    = VK_ACCESS_MEMORY_READ_BIT,
            .dstAccessMask    = VK_ACCESS_TRANSFER_WRITE_BIT,
            .oldLayout        = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout        = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .image            = dst.handle,
            .subresourceRange = srang,
        };

        cmd.PipelineBarrier(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                            VK_PIPELINE_STAGE_TRANSFER_BIT,
                            VK_DEPENDENCY_BY_REGION_BIT,
                            {},
                            {},
                            std::array { in_bar, out_bar });
    }
    cmd.CopyImage(src.handle,
                  VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                  dst.handle,
                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                  copy);
    {
        VkImageMemoryBarrier in_bar {
            .sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .pNext            = nullptr,
            .srcAccessMask    = VK_ACCESS_TRANSFER_READ_BIT,
            .dstAccessMask    = VK_ACCESS_MEMORY_READ_BIT,
            .oldLayout        = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            .newLayout        = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            .image            = src.handle,
            .subresourceRange = srang,
        };
        VkImageMemoryBarrier out_bar {
            .sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .pNext            = nullptr,
            .srcAccessMask    = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask    = VK_ACCESS_SHADER_READ_BIT,
            .oldLayout        = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .newLayout        = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            .image            = dst.handle,
            .subresourceRange = srang,
        };

        cmd.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                            VK_DEPENDENCY_BY_REGION_BIT,
                            {},
                            {},
                            std::array { in_bar, out_bar });
    }

    if (dst.mipmap_level > 1) {
        device.tex_cache().RecGenerateMipmaps(cmd, dst);
    }

    // CopyPass participates in the same temporary-resource lifetime contract as shader passes:
    // a key becomes reusable only after the command buffer has recorded the transfer that reads it.
    // Preparing copy handles is just binding metadata and must not free cache ownership.
    releaseFinalReadTexs(device);
};
void CopyPass::destory(const Device&, RenderingResources&) {
    // Copy passes also keep their pass objects across resource-only refreshes. Resetting the
    // prepared state guarantees that recreated render targets are rebound from the texture cache
    // instead of reusing stale source/destination image handles captured by the previous prepare.
    setPrepared(false);
}
