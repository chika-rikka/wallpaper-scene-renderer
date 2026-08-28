#include "ShaderDrawCore.hpp"
#include "Scene/Scene.h"
#include "Scene/SceneShader.h"
#include "Scene/SceneTextPrimitive.h"

#include "SpecTexs.hpp"
#include "Vulkan/Shader.hpp"
#include "Vulkan/VideoTextureCache.hpp"
#include "vvk/vma_wrapper.hpp"
#include "Utils/Logging.h"
#include "Utils/AutoDeletor.hpp"
#include "Resource.hpp"
#include "PassCommon.hpp"
#include "Msaa.hpp"
#include "Interface/IImageParser.h"

#include "Core/ArrayHelper.hpp"

#include <cassert>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

using namespace wallpaper::vulkan;

std::string wallpaper::vulkan::ShaderDrawPipelineCompatibilityKey(
    VkAttachmentLoadOp load_op, bool model_pass, VkAttachmentLoadOp depth_load_op,
    const ShaderDrawAttachmentDescription& attachment, VkSampleCountFlagBits samples,
    bool resolve_msaa, VkImageLayout color_initial_layout) {
    // Keep this key limited to Vulkan render-pass compatibility. GraphicsPipeline adds the shader,
    // descriptor, vertex-input, blend, depth, and topology state to the final cache key, matching
    // the descriptor-driven PSO caches used by larger renderers instead of tying immutable PSOs to
    // a transient layer/pass identity.
    return "ShaderDraw|format=rgba8|final=shader-read|store-vis=1|load=" +
           std::to_string(static_cast<int>(load_op)) +
           "|init=" + std::to_string(static_cast<int>(color_initial_layout)) +
           "|model=" + (model_pass ? std::string("1") : std::string("0")) +
           "|depth-format=d32|depth-load=" + std::to_string(static_cast<int>(depth_load_op)) +
           "|extra-tag=" + std::string(attachment.cache_tag) +
           "|extra-format=" + std::to_string(static_cast<int>(attachment.format)) +
           "|extra-depth-load=" +
           std::to_string(static_cast<int>(attachment.depth_load_op)) +
           "|extra-stencil-load=" +
           std::to_string(static_cast<int>(attachment.stencil_load_op)) +
           "|samples=" + std::to_string(static_cast<int>(samples)) +
           "|resolve=" + (resolve_msaa ? std::string("1") : std::string("0"));
}

namespace
{

void PopulateTextureBindingsFromReflection(wallpaper::vulkan::ShaderDrawData& desc,
                                           const wallpaper::vulkan::ShaderReflected& ref,
                                           size_t texture_count) {
    desc.vk_tex_binding.clear();
    desc.vk_tex_binding.reserve(texture_count);
    for (size_t i = 0; i < texture_count; i++) {
        wallpaper::i32 binding { -1 };
        if (i < wallpaper::WE_GLTEX_NAMES.size() &&
            wallpaper::exists(ref.binding_map, wallpaper::WE_GLTEX_NAMES[i])) {
            binding = static_cast<wallpaper::i32>(
                ref.binding_map.at(wallpaper::WE_GLTEX_NAMES[i]).binding);
        }
        desc.vk_tex_binding.push_back(binding);
    }
}

// Mesh primitive and index presence are the complete input-assembly contract. Warmup and real
// preparation must derive the same value so their pipeline keys match the pipeline actually used.
VkPrimitiveTopology ToTopology(const wallpaper::SceneMesh& mesh) {
    switch (mesh.Primitive()) {
    case wallpaper::MeshPrimitive::POINT: return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
    case wallpaper::MeshPrimitive::TRIANGLE:
        return mesh.IndexCount() > 0 ? VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST
                                     : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
    }
    assert(false);
    return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
}

// TEXT_E0_IDEST 0x1401e9681 / POSTFX_OMSET 0x1401ebf8c OMSet leftover /
// FullCompo as the color target before type-0 Draw. TREE dest-draw
// leftover and HORIZONTAL are Normal → DONT_CARE. Official D3D11 OMSet
// is a color attachment; Vulkan analog is COLOR_ATTACHMENT at
// BeginRenderPass, not UNDEFINED after a SHADER_READ bootstrap clear.
VkImageLayout DestDrawOmsetInitialLayout(const wallpaper::vulkan::ShaderDrawData& desc,
                                         VkAttachmentLoadOp load_op) {
    if (load_op == VK_ATTACHMENT_LOAD_OP_DONT_CARE &&
        (desc.dest_draw_phase == wallpaper::DestDrawPhase::Leftover ||
         desc.dest_draw_phase == wallpaper::DestDrawPhase::PostFx)) {
        return VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    }
    return VK_IMAGE_LAYOUT_UNDEFINED;
}

std::optional<VmaImageParameters> CreateModelDepthImage(const Device& device, VkExtent3D extent,
                                                        VkSampleCountFlagBits samples) {
    // Model depth is allocated only for opt-in 3D model passes. The existing 2D render-target cache
    // remains color-only, while separate model chunk passes can still behave like one depth-tested
    // scene when they share the same output texture.
    VmaImageParameters image;
    VkImageCreateInfo  info {
        .sType                 = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext                 = nullptr,
        .imageType             = VK_IMAGE_TYPE_2D,
        .format                = VK_FORMAT_D32_SFLOAT,
        .extent                = extent,
        .mipLevels             = 1,
        .arrayLayers           = 1,
        .samples               = samples,
        .tiling                = VK_IMAGE_TILING_OPTIMAL,
        // SAMPLED + TRANSFER_SRC: the volumetrics fill pass blits this scene depth
        // into `_rt_volumetricsSingle`.
        .usage                 = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                     VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                     VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        .sharingMode           = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .initialLayout         = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    image.extent       = extent;
    image.mipmap_level = 1;
    image.samples      = static_cast<uint>(samples);

    VmaAllocationCreateInfo vma_info {};
    vma_info.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    VVK_CHECK_ACT(return std::nullopt,
                         vvk::CreateImage(device.vma_allocator(), info, vma_info, image.handle));

    VkImageViewCreateInfo view_info {
        .sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext    = nullptr,
        .image    = *image.handle,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format   = VK_FORMAT_D32_SFLOAT,
        .subresourceRange =
            VkImageSubresourceRange {
                .aspectMask     = VK_IMAGE_ASPECT_DEPTH_BIT,
                .baseMipLevel   = 0,
                .levelCount     = 1,
                .baseArrayLayer = 0,
                .layerCount     = 1,
            },
    };
    VVK_CHECK_ACT(return std::nullopt, device.handle().CreateImageView(view_info, image.view));
    return image;
}

} // namespace

ShaderDrawCore::ShaderDrawCore(const ShaderDrawRequest& desc) {
    // The render graph builder already classifies hidden offscreen dependencies and gives passes
    // a live scene pointer for diagnostics. Preserve that prepared intent here; dropping these
    // fields forced text/effect passes back through generic visibility and null-scene behavior.
    m_desc.scene               = desc.scene;
    m_desc.node                = desc.node;
    m_desc.layer_id            = desc.layer_id;
    m_desc.execute_when_hidden = desc.execute_when_hidden;
    m_desc.should_execute      = desc.should_execute;
    m_desc.textures            = desc.textures;
    m_desc.output              = desc.output;
    m_desc.alpha_write_policy  = desc.alpha_write_policy;
    m_desc.premultiplied_source_blend = desc.premultiplied_source_blend;
    m_desc.clear_before_draw   = desc.clear_before_draw;
    m_desc.camera_override     = desc.camera_override;
    m_desc.use_active_camera_for_uniforms = desc.use_active_camera_for_uniforms;
    m_desc.use_active_camera_for_parallax = desc.use_active_camera_for_parallax;
    m_desc.use_identity_model  = desc.use_identity_model;
    // LEFTOVER_VS_DESTDRAW / PATH_B 0x14018b170: leftover then last-pass are
    // dest-draw phases on one object. IndexDestDrawPass reads destDrawPhase()
    // after construction; dropping this field left every CustomShaderPass and
    // MaskedMeshPass at DestDrawPhase::None.
    m_desc.dest_draw_phase     = desc.dest_draw_phase;
    m_desc.sprites_map         = desc.sprites_map;
    m_desc.model_pass          = desc.model_pass;
    m_desc.depth_test          = desc.depth_test;
    m_desc.depth_write         = desc.depth_write;
    m_desc.clear_depth         = desc.clear_depth;
    m_desc.depth_greater       = desc.depth_greater;
    m_desc.depth_clear         = desc.depth_clear;
};

std::string ShaderDrawCore::residencyKey(std::string_view pass_kind) const {
    return std::string(pass_kind) + "|layer=" + std::to_string(m_desc.layer_id) +
           "|node=" + std::to_string(reinterpret_cast<std::uintptr_t>(m_desc.node)) +
           "|output=" + m_desc.output;
}

static int IntendedShaderDrawSampleCount(const ShaderDrawData& desc) {
    if (desc.scene != nullptr && ShaderDrawCanUseMsaa(*desc.scene, desc.output, desc.node)) {
        return std::max(1, desc.scene->MsaaSampleCount());
    }
    return 1;
}

bool ShaderDrawCore::canReuseForResidency(const ShaderDrawCore& next) const {
    // A prepared pass may be reused only when its immutable GPU contract is the same. Runtime
    // visibility gates and descriptor texture keys can be refreshed in place, but changing model
    // depth/blend state or the owning SceneNode would require a different render pass/pipeline.
    const int this_samples = static_cast<int>(m_desc.sample_count);
    const int next_samples = IntendedShaderDrawSampleCount(next.m_desc);
    return m_desc.layer_id == next.m_desc.layer_id &&
           m_desc.node == next.m_desc.node &&
           m_desc.output == next.m_desc.output &&
           m_desc.execute_when_hidden == next.m_desc.execute_when_hidden &&
           m_desc.model_pass == next.m_desc.model_pass &&
           m_desc.depth_test == next.m_desc.depth_test &&
           m_desc.depth_write == next.m_desc.depth_write &&
           m_desc.clear_depth == next.m_desc.clear_depth &&
           m_desc.depth_greater == next.m_desc.depth_greater &&
           m_desc.depth_clear == next.m_desc.depth_clear &&
           // Alpha policy changes the prepared pipeline's color write mask, blend operation, and
           // factors. Reusing a pass across that boundary would keep stale composition coverage.
           m_desc.alpha_write_policy == next.m_desc.alpha_write_policy &&
           m_desc.premultiplied_source_blend ==
               next.m_desc.premultiplied_source_blend &&
           m_desc.clear_before_draw == next.m_desc.clear_before_draw &&
           // The uniform update lambda captures the pass camera route. Treat it as part of the
           // prepared uniform contract so a source-space composition route cannot reuse a screen-space
           // publisher after a topology rebuild.
           m_desc.camera_override == next.m_desc.camera_override &&
           m_desc.use_active_camera_for_uniforms ==
               next.m_desc.use_active_camera_for_uniforms &&
           m_desc.use_active_camera_for_parallax ==
               next.m_desc.use_active_camera_for_parallax &&
           m_desc.use_identity_model == next.m_desc.use_identity_model &&
           m_desc.dest_draw_phase == next.m_desc.dest_draw_phase &&
           this_samples == next_samples &&
           ! m_desc.resolve_msaa &&
           m_desc.textures.size() == next.m_desc.textures.size();
}

void ShaderDrawCore::absorbResidencyGraphState(const ShaderDrawCore& next) {
    // Render-graph diffing keeps this pass's expensive Vulkan objects alive while replacing only
    // the declarative state that can change as layers move between hidden and visible residency.
    // Texture handles are rebound by refreshResources()/prepare(), and the runtime gate must follow
    // the newly built graph so effect bypass/final-composite branches stay correct.
    m_desc.scene          = next.m_desc.scene;
    m_desc.layer_id       = next.m_desc.layer_id;
    m_desc.should_execute = next.m_desc.should_execute;
    m_desc.textures       = next.m_desc.textures;
    m_desc.output         = next.m_desc.output;
    m_desc.alpha_write_policy = next.m_desc.alpha_write_policy;
    m_desc.premultiplied_source_blend = next.m_desc.premultiplied_source_blend;
    m_desc.clear_before_draw = next.m_desc.clear_before_draw;
    m_desc.camera_override = next.m_desc.camera_override;
    m_desc.use_active_camera_for_uniforms = next.m_desc.use_active_camera_for_uniforms;
    m_desc.use_active_camera_for_parallax = next.m_desc.use_active_camera_for_parallax;
    m_desc.use_identity_model = next.m_desc.use_identity_model;
    m_desc.dest_draw_phase = next.m_desc.dest_draw_phase;
    m_desc.sprites_map    = next.m_desc.sprites_map;
}

bool ShaderDrawCore::referencesRenderTarget(std::string_view render_target) const {
    // Custom shader passes are affected when either their output framebuffer is the dirty target or
    // one of their descriptor inputs samples it. This lets a resized text bridge update the exact
    // effect chain that consumes it instead of refreshing every other shader in the wallpaper.
    if (m_desc.output == render_target) return true;
    return referencesImportedTexture(render_target);
}

bool ShaderDrawCore::referencesImportedTexture(std::string_view texture_key) const {
    for (const auto& texture : m_desc.textures) {
        if (texture == texture_key) return true;
    }
    if (m_extension != nullptr && m_desc.node != nullptr && m_desc.node->Mesh() != nullptr) {
        for (const auto texture : m_extension->resourceTextures(*m_desc.node->Mesh())) {
            if (texture == texture_key) return true;
        }
    }
    return false;
}

std::optional<vvk::RenderPass> wallpaper::vulkan::CreateShaderDrawRenderPass(
    const vvk::Device& device, VkFormat format, VkAttachmentLoadOp loadOp,
    VkImageLayout finalLayout, const ShaderDrawAttachmentDescription& extra_attachment,
    VkSampleCountFlagBits samples, bool resolve_msaa, VkImageLayout color_initial_layout) {
    const bool store_ms = samples > VK_SAMPLE_COUNT_1_BIT;
    const bool msaa     = store_ms && resolve_msaa;
    const VkSampleCountFlagBits color_samples =
        store_ms ? samples : VK_SAMPLE_COUNT_1_BIT;

    VkAttachmentDescription attachment {
        .format         = format,
        .samples        = color_samples,
        .loadOp         = loadOp,
        .storeOp        = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout  = color_initial_layout,
        .finalLayout    = store_ms ? VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL : finalLayout,
    };

    if (loadOp == VK_ATTACHMENT_LOAD_OP_LOAD &&
        color_initial_layout == VK_IMAGE_LAYOUT_UNDEFINED) {
        attachment.initialLayout = store_ms ? VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
                                            : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }

    VkAttachmentReference attachment_ref {
        .attachment = 0,
        .layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    };

    VkAttachmentDescription depth_attachment {
        .format         = extra_attachment.format,
        .samples        = color_samples,
        .loadOp         = extra_attachment.depth_load_op,
        .storeOp        = extra_attachment.depth_store_op,
        .stencilLoadOp  = extra_attachment.stencil_load_op,
        .stencilStoreOp = extra_attachment.stencil_store_op,
        .initialLayout  = extra_attachment.initial_layout,
        .finalLayout    = extra_attachment.final_layout,
    };
    VkAttachmentReference depth_attachment_ref {
        .attachment = 1,
        .layout     = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    };

    VkAttachmentDescription resolve_attachment {
        .format         = format,
        .samples        = VK_SAMPLE_COUNT_1_BIT,
        .loadOp         = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .storeOp        = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout    = finalLayout,
    };
    const uint32_t resolve_index = extra_attachment.enabled() ? 2u : 1u;
    VkAttachmentReference resolve_ref {
        .attachment = resolve_index,
        .layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    };

    std::array<VkAttachmentDescription, 3> attachments {
        attachment,
        extra_attachment.enabled() ? depth_attachment : resolve_attachment,
        resolve_attachment,
    };

    VkSubpassDescription subpass {
        .pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS,
        .colorAttachmentCount    = 1,
        .pColorAttachments       = &attachment_ref,
        .pResolveAttachments     = msaa ? &resolve_ref : nullptr,
        .pDepthStencilAttachment = extra_attachment.enabled() ? &depth_attachment_ref : nullptr,
    };

    // TEXT_E0_IDEST 0x1401e968a leftover named-RT store, then
    // POSTFX_OMSET HORIZONTAL samples that RT. IMAGE_VT_E8 leftover
    // Draw +0x2d8 then the next dest-draw pass samples it. Official
    // D3D11 implicit hazard. Implicit Vulkan EXTERNAL→BOTTOM_OF_PIPE
    // leaves COLOR_ATTACHMENT_WRITE invisible to FRAGMENT_SHADER, so
    // HORIZONTAL/VERTICAL can sample uninit white.
    std::array<VkSubpassDependency, 2> dependencies {
        VkSubpassDependency {
            .srcSubpass = VK_SUBPASS_EXTERNAL,
            .dstSubpass = 0,
            .srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                            VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
            .srcAccessMask = {},
            .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                             VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                             VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                             VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        },
        VkSubpassDependency {
            .srcSubpass = 0,
            .dstSubpass = VK_SUBPASS_EXTERNAL,
            .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
        },
    };

    uint32_t attachment_count = 1;
    if (extra_attachment.enabled()) attachment_count++;
    if (msaa) attachment_count++;

    VkRenderPassCreateInfo creatinfo {
        .sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .attachmentCount = attachment_count,
        .pAttachments    = attachments.data(),
        .subpassCount    = 1,
        .pSubpasses      = &subpass,
        .dependencyCount = static_cast<uint32_t>(dependencies.size()),
        .pDependencies   = dependencies.data(),
    };
    vvk::RenderPass pass;
    if (auto res = device.CreateRenderPass(creatinfo, pass); res == VK_SUCCESS) {
        return pass;
    } else {
        VVK_CHECK(res);
        return std::nullopt;
    }
}

void wallpaper::vulkan::UpdateShaderDrawUniform(StagingBuffer* buf,
                                                const StagingBufferRef& bufref,
                                                const ShaderReflected::Block& block,
                                                std::string_view name,
                                                const wallpaper::ShaderValue& value) {
    using namespace wallpaper;
    std::span<uint8_t> value_u8 { (uint8_t*)value.data(),
                                  value.size() * sizeof(ShaderValue::value_type) };
    auto               uni = block.member_map.find(name);
    if (uni == block.member_map.end()) {
        // log
        return;
    }

    const size_t offset        = uni->second.offset;
    const size_t reflectedSize = uni->second.size;
    const size_t packedSize    = value_u8.size();

    if (reflectedSize < packedSize) {
        // Some first-party model shaders declare compact matrix uniforms such as `mat3
        // g_ModelMatrix`, while the runtime updater naturally owns a full 4x4 scene transform.
        // Writing the whole packed matrix would overflow the reflected member slot and corrupt the
        // following uniforms. Clamp only at the upload boundary so 2D shader generation and the
        // shared ShaderValue representation do not need a model-specific branch.
        buf->writeToBuf(bufref, value_u8.subspan(0, reflectedSize), offset);
        return;
    }

    if (reflectedSize == packedSize || value.size() <= 1) {
        buf->writeToBuf(bufref, value_u8, offset);
        return;
    }

    // SPIR-V reflection reports std140 array sizes for uniforms such as
    // `float g_AudioSpectrum32Left[32]`, which occupy 16 bytes per element.
    // Our runtime values are stored densely as `float[N]`, so copy them using
    // the reflected stride instead of writing the packed blob directly.
    if (reflectedSize > packedSize && reflectedSize % value.size() == 0) {
        const size_t stride = reflectedSize / value.size();
        if (stride >= sizeof(ShaderValue::value_type)) {
            for (size_t i = 0; i < value.size(); i++) {
                std::span<uint8_t> elem { reinterpret_cast<uint8_t*>(
                                              const_cast<ShaderValue::value_type*>(&value[i])),
                                          sizeof(ShaderValue::value_type) };
                buf->writeToBuf(bufref, elem, offset + i * stride);
            }
            return;
        }
    }

    buf->writeToBuf(bufref, value_u8, offset);
}

static void WriteMaterialUniforms(StagingBuffer* buf, const StagingBufferRef& bufref,
                                  const ShaderReflected::Block&   block,
                                  const wallpaper::SceneMaterial& material) {
    auto write_values = [&](const auto& values) {
        for (const auto& [name, value] : values) {
            if (! wallpaper::exists(block.member_map, name)) continue;
            UpdateShaderDrawUniform(buf, bufref, block, name, value);
        }
    };

    if (material.customShader.shader != nullptr) {
        write_values(material.customShader.shader->default_uniforms);
    }
    write_values(material.customShader.constValues);
}

constexpr VkDeviceSize kInitialDynamicSuballocationSize = 64 * 1024;
constexpr VkDeviceSize kDynamicIndexQuadFloorSize       = sizeof(uint16_t) * 6;

VkDeviceSize InitialDynamicSuballocationSize(VkDeviceSize capacity, VkDeviceSize live_size,
                                             VkDeviceSize element_size) {
    if (capacity == 0) return 0;

    const VkDeviceSize non_empty_element = std::max<VkDeviceSize>(element_size, 1);
    const VkDeviceSize required_live     = std::max<VkDeviceSize>(live_size, non_empty_element);
    const VkDeviceSize bootstrap         = std::min<VkDeviceSize>(
        capacity, std::max<VkDeviceSize>(kInitialDynamicSuballocationSize, non_empty_element));

    // Dynamic particle meshes often advertise a very large theoretical capacity while starting
    // with zero live vertices. Reserve only a small bootstrap range up front, but never choose a
    // range smaller than the data that is already live and must be uploaded immediately.
    return std::min<VkDeviceSize>(capacity, std::max(required_live, bootstrap));
}

VkDeviceSize DynamicVertexUploadSize(const wallpaper::SceneVertexArray& vertex) {
    // Vertex arrays expose both live bytes and authored capacity. Use the live byte count for the
    // first upload so character-rain style particle systems do not reserve their entire theoretical
    // maximum before any spawned particles exist.
    return InitialDynamicSuballocationSize(static_cast<VkDeviceSize>(vertex.CapacitySizeOf()),
                                           static_cast<VkDeviceSize>(vertex.DataSizeOf()),
                                           static_cast<VkDeviceSize>(vertex.OneSizeOf()));
}

VkDeviceSize DynamicIndexUploadSize(const wallpaper::SceneIndexArray& indice) {
    // Index buffers follow the same bootstrap rule as vertices, but CustomShaderPass binds them as
    // VK_INDEX_TYPE_UINT16 at draw time. SceneIndexArray stores both 32-bit model indices and
    // packed 16-bit particle indices behind the same byte-count API, so the non-empty dynamic floor
    // must match the GPU binding size. The effect-dependency route added for private image
    // composites can expose one-quad particle helpers with only 12 bytes of authored capacity;
    // using a 24-byte uint32_t floor makes those valid helpers fail before their first dynamic
    // upload.
    return InitialDynamicSuballocationSize(static_cast<VkDeviceSize>(indice.CapacitySizeof()),
                                           static_cast<VkDeviceSize>(indice.DataSizeOf()),
                                           kDynamicIndexQuadFloorSize);
}

VkCullModeFlags ToVkCullMode(wallpaper::SceneCullMode mode) {
    switch (mode) {
    case wallpaper::SceneCullMode::None: return VK_CULL_MODE_NONE;
    case wallpaper::SceneCullMode::Back: return VK_CULL_MODE_BACK_BIT;
    case wallpaper::SceneCullMode::Front: return VK_CULL_MODE_FRONT_BIT;
    }
    return VK_CULL_MODE_NONE;
}

wallpaper::SceneCullMode ResolveModelCullMode(wallpaper::SceneCullMode mode,
                                              bool                     mirrored_handedness) {
    if (! mirrored_handedness) return mode;

    // A floor reflection uses a negative scale, so the model transform changes handedness and the
    // authored triangle winding is observed backwards by Vulkan. Flip only front/back model culling
    // here; `None` stays double-sided for receiver materials such as reflection grids, and
    // non-model custom shader passes never carry SceneModelRenderState at all.
    switch (mode) {
    case wallpaper::SceneCullMode::Back: return wallpaper::SceneCullMode::Front;
    case wallpaper::SceneCullMode::Front: return wallpaper::SceneCullMode::Back;
    case wallpaper::SceneCullMode::None: return wallpaper::SceneCullMode::None;
    }
    return mode;
}

bool ShouldWriteCustomShaderAlpha(const wallpaper::SceneMaterial& material,
                                  std::string_view                camera_name,
                                  wallpaper::AlphaWritePolicy      alpha_write_policy,
                                  std::string_view                output) {
    const bool is_model_pass = material.modelRenderState.has_value();
    // Model shaders may output non-opaque alpha for their own material math even when the authored
    // object is visually opaque. Allowing that alpha into `_rt_default` makes FinPass present a
    // translucent frame and visually crushes the lighting. Keep the RGB blend factors intact for
    // translucent model materials, but preserve the target alpha just like global 2D passes.
    // Offscreen targets such as `_rt_volumetricsLightBuffer` must keep the shader alpha:
    // volumetricsfront writes a=1 and the official additive combine (SRC_ALPHA, ONE) multiplies
    // by the sampled LightBuffer alpha. Suppressing A left the buffer at the clear value 0 and
    // the passthrough combine added nothing.
    if (is_model_pass && output == wallpaper::SpecTex_Default) return false;
    // Volumetric nodes have an empty camera name, so the 2D compositor gate below would still
    // drop A. Offscreen model targets must store the shader alpha: official additive combine
    // (SRC_ALPHA, ONE) samples LightBuffer.a, and volumetricsfront writes a=1.
    if (is_model_pass && output != wallpaper::SpecTex_Default) return true;

    // Explicit compositor policies opt in to camera-less alpha writes. Authored effect shaders such
    // as auto_sway otherwise retain the historical camera-derived mask because their helper regions
    // can legally output alpha=1 without representing final layer coverage.
    if (alpha_write_policy != wallpaper::AlphaWritePolicy::Preserve) return true;

    return ! (camera_name.empty() || wallpaper::sstart_with(camera_name, "global"));
}

std::string_view EffectiveCustomShaderCamera(
    const wallpaper::vulkan::ShaderDrawData& desc) {
    if (! desc.camera_override.empty()) return desc.camera_override;
    return desc.node != nullptr ? std::string_view(desc.node->Camera()) : std::string_view {};
}

void ApplyAlphaWritePolicy(wallpaper::AlphaWritePolicy                policy,
                           bool                                       writes_alpha,
                           VkPipelineColorBlendAttachmentState&       blend_state) {
    if (!writes_alpha || policy == wallpaper::AlphaWritePolicy::Preserve) return;

    // Alpha coverage is independent from the authored RGB equation. SourceOver publishes a
    // resolved private silhouette, while Max matches Wallpaper Engine's copybackground=false
    // attachment state: later transparent fragments may expand coverage but never reduce coverage
    // already written by an earlier attachment.
    if (!blend_state.blendEnable) {
        blend_state.blendEnable = true;
        blend_state.colorBlendOp = VK_BLEND_OP_ADD;
        blend_state.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        blend_state.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
    }
    blend_state.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    if (policy == wallpaper::AlphaWritePolicy::Max) {
        blend_state.alphaBlendOp = VK_BLEND_OP_MAX;
        blend_state.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        return;
    }
    blend_state.alphaBlendOp = VK_BLEND_OP_ADD;
    blend_state.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
}

void ApplyModelPassDesc(const wallpaper::SceneMaterial&            material,
                        wallpaper::vulkan::ShaderDrawData& desc,
                        VkAttachmentLoadOp&                        load_op) {
    const auto& model_state = material.modelRenderState;
    if (! model_state.has_value()) return;

    desc.model_pass    = true;
    desc.depth_test    = model_state->depthTest;
    desc.depth_write   = model_state->depthWrite;
    desc.depth_greater = model_state->depthGreater;
    desc.depth_clear   = model_state->depthClear;
    // Model passes are the only custom-shader passes allowed to override the historical load/cull
    // defaults. The parser chooses a color-load mode per output target, so offscreen model buffers
    // can be cleared once per frame before later chunks load and composite into the same image.
    switch (model_state->colorLoadMode) {
    case wallpaper::SceneModelColorLoadMode::DontCare: break;
    case wallpaper::SceneModelColorLoadMode::Load: load_op = VK_ATTACHMENT_LOAD_OP_LOAD; break;
    case wallpaper::SceneModelColorLoadMode::Clear: load_op = VK_ATTACHMENT_LOAD_OP_CLEAR; break;
    }
}

std::string_view ModelColorLoadModeName(wallpaper::SceneModelColorLoadMode mode) {
    switch (mode) {
    case wallpaper::SceneModelColorLoadMode::DontCare: return "dont-care";
    case wallpaper::SceneModelColorLoadMode::Load: return "load";
    case wallpaper::SceneModelColorLoadMode::Clear: return "clear";
    }
    return "unknown";
}

VkClearValue BuildCustomShaderClearValue(const wallpaper::Scene&         scene,
                                         const wallpaper::SceneMaterial& material,
                                         bool                            transparent_clear) {
    if (transparent_clear) {
        return VkClearValue {
            .color = { 0.0f, 0.0f, 0.0f, 0.0f },
        };
    }

    if (material.modelRenderState.has_value() &&
        material.modelRenderState->colorLoadMode == wallpaper::SceneModelColorLoadMode::Clear) {
        // Model-only offscreen targets are sampled as textures by later passes. Transparent black
        // is the neutral clear value for those buffers: uncovered pixels contribute no stale color,
        // no alpha, and no previous-frame reflection when the current model geometry shrinks.
        return VkClearValue {
            .color = { 0.0f, 0.0f, 0.0f, 0.0f },
        };
    }

    auto& sc = scene.clearColor;
    // Non-model and main-target custom shader passes retain the existing scene clear color
    // contract. Keeping this branch shared avoids changing ordinary image/effect behavior while
    // still letting model state opt into transparent offscreen clears explicitly.
    return VkClearValue {
        .color = { sc[0], sc[1], sc[2], 1.0f },
    };
}

void ApplyExplicitClearPolicy(const wallpaper::vulkan::ShaderDrawData& desc,
                              const wallpaper::SceneMaterial&                  material,
                              VkAttachmentLoadOp&                              load_op) {
    if (!desc.clear_before_draw) return;

    load_op = VK_ATTACHMENT_LOAD_OP_CLEAR;
    LOG_INFO("CustomShaderExplicitClearPolicy: layer=%d node='%s' material='%s' "
             "output='%s'",
             desc.layer_id,
             desc.node != nullptr ? desc.node->Name().c_str() : "",
             material.name.c_str(),
             desc.output.c_str());
}

ShaderDrawRenderState BuildCustomShaderRenderState(
    const wallpaper::SceneMaterial& material, wallpaper::vulkan::ShaderDrawData& desc) {
    ShaderDrawRenderState state;
    VkColorComponentFlags   color_mask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT;
    const auto camera_name = EffectiveCustomShaderCamera(desc);
    const bool writes_alpha =
        ShouldWriteCustomShaderAlpha(material, camera_name, desc.alpha_write_policy, desc.output);

    if (writes_alpha) color_mask |= VK_COLOR_COMPONENT_A_BIT;
    state.color_blend.colorWriteMask = color_mask;

    const auto blend_mode = material.blenmode;
    SetBlend(blend_mode, state.color_blend);
    // Official dest blend is material +0x1f0 → D3D cases in GFX_BLEND_DESC
    // (0x14009a0fa / 0x14009a32b / 0x14009a2fe). Premul ONE/INV_SRC_ALPHA is
    // gfx+0x28 bit7 (0x14009a12f); that bit stays ctor 0 (0x140098ed7).
    ApplyAlphaWritePolicy(desc.alpha_write_policy, writes_alpha, state.color_blend);
    desc.blending = state.color_blend.blendEnable;

    SetAttachmentLoadOp(blend_mode, state.color_load_op);
    ApplyModelPassDesc(material, desc, state.color_load_op);
    ApplyExplicitClearPolicy(desc, material, state.color_load_op);
    if (desc.sample_count > VK_SAMPLE_COUNT_1_BIT &&
        state.color_load_op == VK_ATTACHMENT_LOAD_OP_DONT_CARE) {
        // Compose shaders usually omit alpha. DONT_CARE would drop the opaque
        // MSAA clear and leave uncovered samples at A=0.
        state.color_load_op = VK_ATTACHMENT_LOAD_OP_LOAD;
    }
    if (desc.dest_draw_phase != wallpaper::DestDrawPhase::None) {
        LOG_INFO("DestDrawPipelineContract: layer=%d node='%s' material='%s' phase=%d "
                 "output='%s' blend-mode=%d blend-enable=%d "
                 "color-factors=[%d %d] alpha-factors=[%d %d] load=%d mask=0x%x",
                 desc.layer_id,
                 desc.node != nullptr ? desc.node->Name().c_str() : "",
                 material.name.c_str(),
                 static_cast<int>(desc.dest_draw_phase),
                 desc.output.c_str(),
                 static_cast<int>(blend_mode),
                 state.color_blend.blendEnable ? 1 : 0,
                 static_cast<int>(state.color_blend.srcColorBlendFactor),
                 static_cast<int>(state.color_blend.dstColorBlendFactor),
                 static_cast<int>(state.color_blend.srcAlphaBlendFactor),
                 static_cast<int>(state.color_blend.dstAlphaBlendFactor),
                 static_cast<int>(state.color_load_op),
                 static_cast<unsigned int>(state.color_blend.colorWriteMask));
        // A dest-draw effect can have correct geometry and blend state while still exposing a
        // hard card boundary when one of its small authored masks or private render targets is
        // sampled with the wrong resolution/filter contract. Report the scene-side descriptor
        // inputs next to the pipeline contract so the source of such a boundary is observable
        // without adding project- or layer-specific branches to the renderer.
        if (desc.scene != nullptr) {
            for (size_t slot = 0; slot < desc.textures.size(); ++slot) {
                const auto& texture_name = desc.textures[slot];
                const auto  rt_it = desc.scene->renderTargets.find(texture_name);
                if (rt_it != desc.scene->renderTargets.end()) {
                    const auto& sample = rt_it->second.sample;
                    const auto  resolution = rt_it->second.ResolutionVector();
                    LOG_INFO("DestDrawTextureContract: layer=%d phase=%d slot=%zu "
                             "texture='%s' source=render-target resolution=[%d %d %d %d] "
                             "sampler=[%.*s %.*s %.*s %.*s]",
                             desc.layer_id,
                             static_cast<int>(desc.dest_draw_phase),
                             slot,
                             texture_name.c_str(),
                             resolution[0],
                             resolution[1],
                             resolution[2],
                             resolution[3],
                             static_cast<int>(wallpaper::TextureWrapName(sample.wrapS).size()),
                             wallpaper::TextureWrapName(sample.wrapS).data(),
                             static_cast<int>(wallpaper::TextureWrapName(sample.wrapT).size()),
                             wallpaper::TextureWrapName(sample.wrapT).data(),
                             static_cast<int>(wallpaper::TextureFilterName(sample.magFilter).size()),
                             wallpaper::TextureFilterName(sample.magFilter).data(),
                             static_cast<int>(wallpaper::TextureFilterName(sample.minFilter).size()),
                             wallpaper::TextureFilterName(sample.minFilter).data());
                    continue;
                }
                const auto texture_it = desc.scene->textures.find(texture_name);
                if (texture_it == desc.scene->textures.end()) {
                    LOG_INFO("DestDrawTextureContract: layer=%d phase=%d slot=%zu "
                             "texture='%s' source=missing",
                             desc.layer_id,
                             static_cast<int>(desc.dest_draw_phase),
                             slot,
                             texture_name.c_str());
                    continue;
                }
                const auto& sample = texture_it->second.sample;
                const auto resolution =
                    desc.scene->EffectiveImportedTextureResolution(texture_it->second);
                LOG_INFO("DestDrawTextureContract: layer=%d phase=%d slot=%zu texture='%s' "
                         "source=imported resolution=[%d %d %d %d] "
                         "sampler=[%.*s %.*s %.*s %.*s]",
                         desc.layer_id,
                         static_cast<int>(desc.dest_draw_phase),
                         slot,
                         texture_name.c_str(),
                         resolution[0],
                         resolution[1],
                         resolution[2],
                         resolution[3],
                         static_cast<int>(wallpaper::TextureWrapName(sample.wrapS).size()),
                         wallpaper::TextureWrapName(sample.wrapS).data(),
                         static_cast<int>(wallpaper::TextureWrapName(sample.wrapT).size()),
                         wallpaper::TextureWrapName(sample.wrapT).data(),
                         static_cast<int>(wallpaper::TextureFilterName(sample.magFilter).size()),
                         wallpaper::TextureFilterName(sample.magFilter).data(),
                         static_cast<int>(wallpaper::TextureFilterName(sample.minFilter).size()),
                         wallpaper::TextureFilterName(sample.minFilter).data());
            }
        }
    }
    return state;
}

void ApplyModelPipelineState(const wallpaper::SceneMaterial&                  material,
                             const wallpaper::vulkan::ShaderDrawData& desc,
                             GraphicsPipeline&                                pipeline) {
    const auto& model_state = material.modelRenderState;
    if (! model_state.has_value()) return;

    // Only model materials can carry this optional state. Applying it here keeps culling separate
    // from the old 2D custom-shader defaults while still using the existing pipeline construction
    // path for shader reflection, descriptors, and mesh buffers.
    pipeline.depth.depthTestEnable       = desc.depth_test;
    pipeline.depth.depthWriteEnable      = desc.depth_write;
    pipeline.depth.depthCompareOp        = desc.depth_greater ? VK_COMPARE_OP_GREATER
                                                             : VK_COMPARE_OP_LESS_OR_EQUAL;
    pipeline.depth.depthBoundsTestEnable = false;
    pipeline.depth.stencilTestEnable     = false;
    const auto effective_cull_mode =
        ResolveModelCullMode(model_state->cullMode, model_state->mirroredHandedness);
    pipeline.raster.cullMode = ToVkCullMode(effective_cull_mode);
    LOG_INFO("ModelRenderStateBind: node='%s' shader='%s' output='%s' color-load=%s "
             "mirrored-handedness=%s depth-test=%s depth-write=%s depth-clear=%s "
             "depth-compare=%s depth-clear-z=%.3f cull=%u",
             desc.node != nullptr ? desc.node->Name().c_str() : "<null>",
             material.customShader.shader != nullptr ? material.customShader.shader->name.c_str()
                                                     : "<null>",
             desc.output.c_str(),
             ModelColorLoadModeName(model_state->colorLoadMode).data(),
             model_state->mirroredHandedness ? "true" : "false",
             desc.depth_test ? "true" : "false",
             desc.depth_write ? "true" : "false",
             desc.clear_depth ? "true" : "false",
             desc.depth_greater ? "greater" : "less-equal",
             desc.depth_clear,
             static_cast<unsigned>(pipeline.raster.cullMode));
}

VkDeviceSize GrowDynamicSuballocationSize(VkDeviceSize current_size,
                                          VkDeviceSize required_live_size, VkDeviceSize capacity,
                                          VkDeviceSize element_size) {
    if (capacity == 0) return 0;

    // Growth is geometric but clamped to authored capacity. That keeps normal particle expansion
    // amortized while still refusing to cross the renderer-side maximum promised by the scene data.
    VkDeviceSize next_size =
        current_size == 0
            ? InitialDynamicSuballocationSize(capacity, required_live_size, element_size)
            : current_size;
    const VkDeviceSize required = std::min<VkDeviceSize>(
        capacity,
        std::max<VkDeviceSize>(required_live_size, std::max<VkDeviceSize>(element_size, 1)));

    while (next_size < required && next_size < capacity) {
        const VkDeviceSize doubled = next_size > capacity / 2 ? capacity : next_size * 2;
        next_size                  = std::max<VkDeviceSize>(required, doubled);
        next_size                  = std::min<VkDeviceSize>(next_size, capacity);
    }
    return next_size;
}

bool RefreshCustomShaderPassTextures(wallpaper::Scene& scene, const Device& device,
                                     ShaderDrawData& desc) {
    desc.vk_textures.resize(desc.textures.size());
    for (wallpaper::usize i = 0; i < desc.textures.size(); i++) {
        auto& tex_name = desc.textures[i];
        if (tex_name.empty()) {
            desc.vk_textures[i] = {};
            continue;
        }

        ImageSlotsRef img_slots;
        const auto    render_target_it = scene.renderTargets.find(tex_name);
        if (render_target_it != scene.renderTargets.end()) {
            // The scene render-target table is the authoritative source for internal effect FBOs.
            // Some authored blur chains use plain names like `blur_start_2_<addr>`, so relying only
            // on the `_rt_` prefix would send valid runtime targets through the material-file
            // parser.
            auto& rt  = render_target_it->second;
            auto  opt = device.tex_cache().Query(
                tex_name, wallpaper::vulkan::ToTexKey(rt), ! rt.allowReuse);
            if ((desc.layer_id == 248 || desc.layer_id == 242) && i == 0) {
                LOG_INFO("DestDrawDateTex0: id=%d phase=%d tex='%s' rt=%dx%d ok=%d",
                         desc.layer_id,
                         static_cast<int>(desc.dest_draw_phase),
                         tex_name.c_str(),
                         rt.width,
                         rt.height,
                         opt.has_value() ? 1 : 0);
            }
            if (! opt.has_value()) {
                LOG_ERROR("CustomShaderPassRefresh: query input failed node='%s' output='%s' "
                          "slot=%zu texture='%s'",
                          desc.node != nullptr ? desc.node->Name().c_str() : "<null>",
                          desc.output.c_str(),
                          static_cast<size_t>(i),
                          tex_name.c_str());
                desc.vk_textures[i] = {};
                continue;
            }
            img_slots.slots.clear();
            img_slots.slots.push_back(opt.value());
        } else if (wallpaper::IsSpecTex(tex_name)) {
            LOG_ERROR("CustomShaderPassRefresh: missing input render target node='%s' "
                      "output='%s' slot=%zu texture='%s'",
                      desc.node != nullptr ? desc.node->Name().c_str() : "<null>",
                      desc.output.c_str(),
                      static_cast<size_t>(i),
                      tex_name.c_str());
            desc.vk_textures[i] = {};
            continue;
        } else {
            if (scene.dirtyImportedTextureKeys.count(tex_name) == 0) {
                if (auto cached_slots = device.tex_cache().FindTex(tex_name);
                    cached_slots.has_value()) {
                    desc.vk_textures[i] = *cached_slots;
                    continue;
                }
            } else {
                scene.DropParsedImageCache(tex_name);
            }

            const auto texture_it = scene.textures.find(tex_name);
            const bool static_scene_texture =
                texture_it != scene.textures.end() && ! texture_it->second.isVideo;
            auto image = static_scene_texture ? scene.GetParsedImageIfReady(tex_name) : nullptr;
            if (image == nullptr) {
                image = static_scene_texture
                            ? scene.ParseImageBlockingCached(tex_name)
                            : (scene.imageParser != nullptr ? scene.imageParser->Parse(tex_name)
                                                            : nullptr);
            }
            if (image) {
                if (scene.textures.count(tex_name) != 0 && scene.textures.at(tex_name).isVideo) {
                    const auto paused_it = scene.videoTexturePaused.find(tex_name);
                    const bool stopped   = scene.videoTextureStopped.count(tex_name) != 0;
                    // Hidden video passes are kept prepared so visibility flips are cheap, but the
                    // backing decoder should still start paused unless a scene script explicitly
                    // requested playback for this texture.
                    const bool initially_paused =
                        paused_it != scene.videoTexturePaused.end()
                            ? paused_it->second
                            : (desc.node != nullptr && ! desc.node->Visible());
                    const auto initial_state =
                        stopped
                            ? wallpaper::VideoTexturePlaybackState::Stopped
                            : (initially_paused ? wallpaper::VideoTexturePlaybackState::Paused
                                                : wallpaper::VideoTexturePlaybackState::Playing);
                    img_slots = device.video_tex_cache().Acquire(
                        tex_name, scene.textures.at(tex_name), *image, initial_state);
                } else {
                    img_slots = device.tex_cache().CreateTex(*image);
                    if (static_scene_texture) {
                        scene.DropParsedImageCache(tex_name);
                    }
                }
            } else {
                LOG_ERROR("parse tex \"%s\" failed", tex_name.c_str());
                desc.vk_textures[i] = {};
                continue;
            }
        }
        desc.vk_textures[i] = img_slots;
    }

    auto&      tex_name  = desc.output;
    const auto output_it = scene.renderTargets.find(tex_name);
    if (output_it == scene.renderTargets.end()) {
        // Outputs must be registered render targets, but they do not have to be `_rt_`-prefixed:
        // effect-local FBOs are uniquified from their authored names and are still valid Vulkan
        // framebuffer destinations once WPSceneParser has inserted them into scene.renderTargets.
        LOG_ERROR("CustomShaderPassRefresh: missing output render target node='%s' output='%s'",
                  desc.node != nullptr ? desc.node->Name().c_str() : "<null>",
                  tex_name.c_str());
        return false;
    }
    auto& rt = output_it->second;
    desc.sample_count      = VK_SAMPLE_COUNT_1_BIT;
    desc.resolve_msaa      = false;
    desc.alpha_to_coverage = false;
    desc.vk_resolve        = {};
    if (ShaderDrawCanUseMsaa(scene, tex_name, desc.node)) {
        const auto ms_name = std::string(wallpaper::SpecTex_DefaultMS);
        const auto ms_it   = scene.renderTargets.find(ms_name);
        if (ms_it != scene.renderTargets.end()) {
            if (auto ms_opt = device.tex_cache().Query(
                    ms_name, wallpaper::vulkan::ToTexKey(ms_it->second), ! ms_it->second.allowReuse);
                ms_opt.has_value()) {
                desc.vk_output    = ms_opt.value();
                desc.sample_count = static_cast<VkSampleCountFlagBits>(
                    std::max(1u, ms_it->second.sample_count > 0
                                     ? static_cast<uint>(ms_it->second.sample_count)
                                     : 1u));
                if (desc.node != nullptr && desc.node->Mesh() != nullptr &&
                    desc.node->Mesh()->Material() != nullptr) {
                    desc.alpha_to_coverage = desc.node->Mesh()->Material()->alpha_to_coverage;
                }
                return true;
            }
        }
        LOG_ERROR("CustomShaderPassRefresh: MSAA compose target missing node='%s' output='%s'",
                  desc.node != nullptr ? desc.node->Name().c_str() : "<null>",
                  tex_name.c_str());
    }
    if (auto opt =
            device.tex_cache().Query(tex_name, wallpaper::vulkan::ToTexKey(rt), ! rt.allowReuse);
        opt.has_value()) {
        desc.vk_output = opt.value();
        return true;
    }
    LOG_ERROR("CustomShaderPassRefresh: query output failed node='%s' output='%s'",
              desc.node != nullptr ? desc.node->Name().c_str() : "<null>",
              tex_name.c_str());
    return false;
}

bool StaticSceneTexturesResidentForDeferredPrepare(wallpaper::Scene& scene, const Device& device,
                                                   const ShaderDrawData& desc,
                                                   const ShaderDrawExtension* extension) {
    std::vector<std::string_view> missing_textures;
    for (const auto& tex_name : desc.textures) {
        if (tex_name.empty()) continue;
        if (scene.renderTargets.count(tex_name) != 0 || wallpaper::IsSpecTex(tex_name)) continue;
        if (scene.dirtyImportedTextureKeys.count(tex_name) != 0) continue;

        const auto texture_it = scene.textures.find(tex_name);
        if (texture_it == scene.textures.end() || texture_it->second.isVideo) continue;
        if (! device.tex_cache().FindTex(tex_name).has_value()) {
            missing_textures.push_back(tex_name);
        }
    }
    if (extension != nullptr && desc.node != nullptr && desc.node->Mesh() != nullptr) {
        for (const auto texture : extension->resourceTextures(*desc.node->Mesh())) {
            const std::string tex_name(texture);
            if (tex_name.empty()) continue;
            if (scene.dirtyImportedTextureKeys.count(tex_name) != 0) continue;

            const auto texture_it = scene.textures.find(tex_name);
            if (texture_it == scene.textures.end() || texture_it->second.isVideo) continue;
            if (! device.tex_cache().FindTex(tex_name).has_value()) {
                missing_textures.push_back(tex_name);
            }
        }
    }

    if (missing_textures.empty()) return true;

    // This is the guardrail that makes runtime visibility behave like a game-engine streaming
    // system: a deferred pass is not allowed to fall back to the blocking texture creation path
    // inside RefreshCustomShaderPassTextures(). It will stay off the render graph's executable set
    // until requestDeferredPrepareResources() has finished the background parse and the budgeted
    // GPU residency work.
    std::string missing;
    for (const auto texture : missing_textures) {
        if (! missing.empty()) missing += ",";
        missing += texture;
    }
    LOG_INFO("CustomShaderPassDeferredPrepareWaitTextures: node='%s' output='%s' missing='%s'",
             desc.node != nullptr ? desc.node->Name().c_str() : "<null>",
             desc.output.c_str(),
             missing.c_str());
    return false;
}

VmaImageParameters* QuerySharedModelDepthImage(const Device& device, RenderingResources& rr,
                                               ShaderDrawData& desc) {
    auto&      depth      = rr.model_depth_images[desc.output];
    const bool missing    = ! depth.view || ! depth.handle;
    const bool wrong_size = depth.extent.width != desc.vk_output.extent.width ||
                            depth.extent.height != desc.vk_output.extent.height ||
                            depth.extent.depth != desc.vk_output.extent.depth;
    const bool wrong_samples = depth.samples != static_cast<uint>(desc.sample_count);
    if (missing || wrong_size || wrong_samples) {
        // A single output can receive many model chunk passes. Recreate the shared depth image only
        // when the output extent or sample count changes, then later chunks can load the same depth
        // written by the earlier chunks in render-graph order.
        auto replacement = CreateModelDepthImage(device, desc.vk_output.extent, desc.sample_count);
        if (! replacement.has_value()) {
            LOG_ERROR("CustomShaderPassRefresh: cannot create shared model depth image node='%s' "
                      "output='%s' extent=[%u,%u] samples=%u",
                      desc.node != nullptr ? desc.node->Name().c_str() : "<null>",
                      desc.output.c_str(),
                      desc.vk_output.extent.width,
                      desc.vk_output.extent.height,
                      static_cast<unsigned>(desc.sample_count));
            return nullptr;
        }
        depth = std::move(replacement.value());
        rr.model_depth_resolved.erase(desc.output);
    }
    if (desc.sample_count > VK_SAMPLE_COUNT_1_BIT) {
        auto&      resolved      = rr.model_depth_resolved[desc.output];
        const bool res_missing   = ! resolved.view || ! resolved.handle;
        const bool res_wrong_size = resolved.extent.width != desc.vk_output.extent.width ||
                                    resolved.extent.height != desc.vk_output.extent.height;
        if (res_missing || res_wrong_size) {
            auto replacement =
                CreateModelDepthImage(device, desc.vk_output.extent, VK_SAMPLE_COUNT_1_BIT);
            if (replacement.has_value()) {
                resolved = std::move(replacement.value());
            }
        }
    }
    return &depth;
}

ShaderDrawAttachmentDescription ResolveShaderDrawAttachment(
    const ShaderDrawData& desc, const ShaderDrawExtension* extension) {
    if (desc.model_pass) {
        const auto depth_load_op = desc.clear_depth ? VK_ATTACHMENT_LOAD_OP_CLEAR
                                                    : VK_ATTACHMENT_LOAD_OP_LOAD;
        return ShaderDrawAttachmentDescription {
            .format           = VK_FORMAT_D32_SFLOAT,
            .depth_load_op    = depth_load_op,
            .depth_store_op   = VK_ATTACHMENT_STORE_OP_STORE,
            .stencil_load_op  = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencil_store_op = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initial_layout   = depth_load_op == VK_ATTACHMENT_LOAD_OP_LOAD
                                    ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
                                    : VK_IMAGE_LAYOUT_UNDEFINED,
            .final_layout     = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            .cache_tag        = "model-depth",
        };
    }
    return extension != nullptr ? extension->attachmentDescription()
                                : ShaderDrawAttachmentDescription {};
}

bool RecreateCustomShaderPassFramebuffer(const Device& device, RenderingResources& rr,
                                         ShaderDrawData& desc,
                                         ShaderDrawExtension* extension) {
    desc.fb.reset();
    if (! desc.pipeline.pass || desc.vk_output.view == VK_NULL_HANDLE ||
        desc.vk_output.extent.width == 0 || desc.vk_output.extent.height == 0) {
        LOG_ERROR("CustomShaderPassRefresh: cannot recreate framebuffer node='%s' output='%s' "
                  "hasRenderPass=%s hasView=%s extent=[%u,%u]",
                  desc.node != nullptr ? desc.node->Name().c_str() : "<null>",
                  desc.output.c_str(),
                  desc.pipeline.pass ? "true" : "false",
                  desc.vk_output.view != VK_NULL_HANDLE ? "true" : "false",
                  desc.vk_output.extent.width,
                  desc.vk_output.extent.height);
        return false;
    }
    const auto attachment = ResolveShaderDrawAttachment(desc, extension);
    if (desc.model_pass) {
        desc.depth_stencil_image_ref = QuerySharedModelDepthImage(device, rr, desc);
    } else if (attachment.enabled() && extension != nullptr) {
        desc.depth_stencil_image_ref = extension->acquireAttachment(device, rr, desc);
    } else {
        desc.depth_stencil_image_ref = nullptr;
    }
    if (attachment.enabled() && desc.depth_stencil_image_ref == nullptr) return false;

    std::array<VkImageView, 3> attachments {
        desc.vk_output.view,
        desc.depth_stencil_image_ref != nullptr && desc.depth_stencil_image_ref->view
            ? *desc.depth_stencil_image_ref->view
            : VK_NULL_HANDLE,
        desc.vk_resolve.view,
    };
    if (desc.resolve_msaa && desc.vk_resolve.view == VK_NULL_HANDLE) {
        LOG_ERROR("CustomShaderPassRefresh: missing MSAA resolve view node='%s' output='%s'",
                  desc.node != nullptr ? desc.node->Name().c_str() : "<null>",
                  desc.output.c_str());
        return false;
    }
    uint32_t attachment_count = 1;
    if (attachment.enabled()) attachment_count++;
    if (desc.resolve_msaa) {
        if (! attachment.enabled()) {
            attachments[1] = desc.vk_resolve.view;
        }
        attachment_count++;
    }
    VkFramebufferCreateInfo info {
        .sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
        .pNext           = nullptr,
        .renderPass      = *desc.pipeline.pass,
        .attachmentCount = attachment_count,
        .pAttachments    = attachments.data(),
        .width           = desc.vk_output.extent.width,
        .height          = desc.vk_output.extent.height,
        .layers          = 1,
    };
    return device.handle().CreateFramebuffer(info, desc.fb) == VK_SUCCESS;
}

bool ShaderDrawCore::prepare(Scene& scene, const Device& device, RenderingResources& rr) {
    // Prepared passes can survive resource-only refreshes, so keep the live scene pointer current
    // before binding render targets and before text/effect diagnostics read bridge metadata.
    m_desc.scene = &scene;
    // A retrying prepared pass may still own a framebuffer whose attachment points at the previous
    // TextureCache image view. Drop it before `Query()` can resize and destroy that output image,
    // otherwise Vulkan sees a framebuffer referencing a dead attachment during minute-rollover
    // text bridge updates.
    m_desc.fb.reset();
    m_desc.vk_tex_binding.clear();
    if (m_desc.node == nullptr || m_desc.node->Mesh() == nullptr ||
        m_desc.node->Mesh()->Material() == nullptr ||
        m_desc.node->Mesh()->Material()->customShader.shader == nullptr) {
        LOG_ERROR("ShaderDrawPrepare: incomplete scene contract node='%s' output='%s'",
                  m_desc.node != nullptr ? m_desc.node->Name().c_str() : "<null>",
                  m_desc.output.c_str());
        return false;
    }
    SceneMesh& mesh = *(m_desc.node->Mesh());
    if (m_extension != nullptr && ! m_extension->configure(device, m_desc, mesh)) return false;
    if (! RefreshCustomShaderPassTextures(scene, device, m_desc)) return false;
    if (m_extension != nullptr && ! m_extension->refreshTextures(scene, device, m_desc)) {
        return false;
    }

    std::vector<Uni_ShaderSpv> spvs;
    DescriptorSetInfo          descriptor_info;
    ShaderReflected            ref;
    {
        SceneShader& shader = *(mesh.Material()->customShader.shader);

        if (! GenReflect(shader.codes, spvs, ref)) {
            LOG_ERROR("gen spv reflect failed, %s", shader.name.c_str());
            return false;
        }

        auto& bindings = descriptor_info.bindings;
        bindings.resize(ref.binding_map.size());

        /*
        LOG_INFO("----shader------");
        LOG_INFO("%s", shader.name.c_str());
        LOG_INFO("--inputs:");
        for (auto& i : ref.input_location_map) {
            LOG_INFO("%d %s", i.second, i.first.c_str());
        }
        LOG_INFO("--bindings:");
        */

        std::transform(
            ref.binding_map.begin(), ref.binding_map.end(), bindings.begin(), [](auto& item) {
                // LOG_INFO("%d %s", item.second.binding, item.first.c_str());
                return item.second;
            });

        PopulateTextureBindingsFromReflection(m_desc, ref, m_desc.vk_textures.size());
    }

    m_desc.draw_count = 0;
    std::vector<VkVertexInputBindingDescription>   bind_descriptions;
    std::vector<VkVertexInputAttributeDescription> attr_descriptions;
    {
        m_desc.dyn_vertex = mesh.Dynamic();
        if (m_desc.dest_draw_phase == wallpaper::DestDrawPhase::Leftover ||
            m_desc.dest_draw_phase == wallpaper::DestDrawPhase::PostFx ||
            m_desc.dest_draw_phase == wallpaper::DestDrawPhase::LastPass) {
            // IMAGE_2D8_NOFULLFB 0x1401eb180 / IMAGE_VT_E8 0x140208067:
            // leftover +0x2d8 is the flags=0 0..AABB object card.
            // POSTFX_MESH 0x1401ea151 / 0x1401ede30: HORIZONTAL +0x2e0
            // ±1 and VERTICAL +0x2e8 ±half AABB are the same class of
            // vt+0xb0 cards, not TEXT_LAYOUT_VERTS Dirty glyphs.
            // ChangeMeshDataFrom onto an image/text-effect node leaves
            // SceneMesh::Dynamic() true and would put those verts on
            // dyn_buf with leftover TextPass glyphs/UBOs.
            m_desc.dyn_vertex = false;
        }
        // Dynamic meshes allocate fresh staging/GPU subranges every time the render graph is
        // recompiled. A static-looking text layer can therefore become "clean" long before a new
        // pass instance is created, which leaves the newly allocated buffer ranges uninitialized if
        // we only upload on `mesh.Dirty()`. Marking the pass for one mandatory upload keeps
        // long-lived text quads valid across unrelated render-graph rebuilds triggered by other
        // animated layers such as effect-backed clocks and dates.
        m_desc.force_dyn_upload = m_desc.dyn_vertex;
        m_desc.vertex_bufs.resize(mesh.VertexCount());

        for (uint i = 0; i < mesh.VertexCount(); i++) {
            const auto& vertex    = mesh.GetVertexArray(i);
            auto        attrs_map = vertex.GetAttrOffsetMap();

            VkVertexInputBindingDescription bind_desc {
                .binding   = i,
                .stride    = (uint32_t)vertex.OneSizeOf(),
                .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
            };
            bind_descriptions.push_back(bind_desc);

            for (auto& item : ref.input_location_map) {
                auto&      name     = item.first;
                auto&      input    = item.second;
                const bool has_attr = exists(attrs_map, name);
                usize      offset   = has_attr ? attrs_map[name].offset : 0;

                VkVertexInputAttributeDescription attr_desc {
                    .location = input.location,
                    .binding  = i,
                    .format   = input.format,
                    .offset   = (u32)offset,
                };
                attr_descriptions.push_back(attr_desc);
            }
            {
                auto& buf = m_desc.vertex_bufs[i];
                if (m_desc.dyn_vertex) {
                    const auto initial_size = DynamicVertexUploadSize(vertex);
                    if (! rr.dyn_buf->allocateSubRef(initial_size, buf)) return false;
                } else if (! mesh.FileImmutable()) {
                    if (! rr.vertex_buf->allocateSubRef(vertex.CapacitySizeOf(), buf)) return false;
                    if (! rr.vertex_buf->writeToBuf(buf, { (uint8_t*)vertex.Data(), buf.size }))
                        return false;
                }
            }
            m_desc.draw_count += (u32)(vertex.DataSize() / vertex.OneSize());
        }

        if (! m_desc.dyn_vertex && mesh.FileImmutable()) {
            m_desc.immutable_mesh = rr.immutable_meshes.getOrCreate(device, mesh);
            if (! m_desc.immutable_mesh) return false;
        }

        if (mesh.IndexCount() > 0) {
            m_desc.index_element_bytes = mesh.IndexElementBytes();
            m_desc.draw_count          = mesh.LogicalIndexCount();
            auto& buf                  = m_desc.index_buf;
            if (m_desc.dyn_vertex) {
                auto& indice = mesh.GetIndexArray(0);
                const auto initial_size = DynamicIndexUploadSize(indice);
                if (! rr.dyn_buf->allocateSubRef(initial_size, buf)) return false;
            } else if (! mesh.FileImmutable()) {
                auto& indice = mesh.GetIndexArray(0);
                if (! rr.vertex_buf->allocateSubRef(indice.CapacitySizeof(), buf)) return false;
                if (! rr.vertex_buf->writeToBuf(buf, { (uint8_t*)indice.Data(), buf.size })) {
                    return false;
                }
            }
        }
    }
    const auto render_state = BuildCustomShaderRenderState(*mesh.Material(), m_desc);
    {
        const auto attachment = ResolveShaderDrawAttachment(m_desc, m_extension);
        const auto color_initial =
            DestDrawOmsetInitialLayout(m_desc, render_state.color_load_op);
        auto opt = CreateShaderDrawRenderPass(device.handle(),
                                              VK_FORMAT_R8G8B8A8_UNORM,
                                              render_state.color_load_op,
                                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                              attachment,
                                              m_desc.sample_count,
                                              m_desc.resolve_msaa,
                                              color_initial);
        if (! opt.has_value()) return false;
        auto& pass = opt.value();

        descriptor_info.push_descriptor = true;
        GraphicsPipeline pipeline;
        pipeline.toDefault();
        pipeline.multisample.rasterizationSamples = m_desc.sample_count;
        pipeline.multisample.alphaToCoverageEnable =
            m_desc.alpha_to_coverage && m_desc.sample_count > VK_SAMPLE_COUNT_1_BIT;
        ApplyModelPipelineState(*mesh.Material(), m_desc, pipeline);
        m_desc.pipeline.debug_name =
            "CustomShaderPass[node=" +
            (m_desc.node != nullptr ? m_desc.node->Name() : std::string("(null)")) +
            ",output=" + m_desc.output + "]";
        pipeline.addDescriptorSetInfo(spanone { descriptor_info })
            .setColorBlendStates(spanone { render_state.color_blend })
            .setTopology(ToTopology(mesh))
            .addInputBindingDescription(bind_descriptions)
            .addInputAttributeDescription(attr_descriptions);
        for (auto& spv : spvs) pipeline.addStage(std::move(spv));

        m_desc.pipeline.cache_key = ShaderDrawPipelineCompatibilityKey(
            render_state.color_load_op,
            m_desc.model_pass,
            m_desc.clear_depth ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD,
            attachment,
            m_desc.sample_count,
            m_desc.resolve_msaa,
            color_initial);
        if (! pipeline.create(device, pass, m_desc.pipeline, rr.pipeline_cache.get())) return false;

        if (m_extension != nullptr &&
            ! m_extension->preparePipelines(
                device,
                rr,
                ShaderDrawPipelineContext {
                    .data                   = m_desc,
                    .mesh                   = mesh,
                    .material               = *mesh.Material(),
                    .descriptor_info        = descriptor_info,
                    .binding_descriptions   = bind_descriptions,
                    .attribute_descriptions = attr_descriptions,
                    .render_state           = render_state,
                })) {
            return false;
        }
    }
    {
        // The helper above already converts framebuffer creation into a plain success/failure
        // contract so that both the initial prepare path and the lightweight resource-refresh path
        // can share the same code. Keeping it as an explicit boolean check avoids routing a
        // non-VkResult helper through the `VVK_CHECK_*` macros, which only understand raw Vulkan
        // return codes.
        if (! RecreateCustomShaderPassFramebuffer(device, rr, m_desc, m_extension)) return false;
    }

    if (! ref.blocks.empty()) {
        auto& block = ref.blocks.front();
        if (! rr.dyn_buf->allocateSubRef(
                block.size, m_desc.ubo_buf, device.limits().minUniformBufferOffsetAlignment)) {
            return false;
        }
    }
    if (! ref.blocks.empty()) {
        std::function<void()> update_dyn_buf_op;
        if (m_desc.dyn_vertex) {
            auto&       mesh             = *m_desc.node->Mesh();
            auto*       dyn_buf          = rr.dyn_buf;
            auto&       vertex_bufs      = m_desc.vertex_bufs;
            auto&       draw_count       = m_desc.draw_count;
            auto&       index_buf        = m_desc.index_buf;
            auto&       force_dyn_upload = m_desc.force_dyn_upload;
            update_dyn_buf_op                = [&mesh,
                                                &vertex_bufs,
                                                &draw_count,
                                                &index_buf,
                                                dyn_buf,
                                                &force_dyn_upload]() {
                const bool dirty        = mesh.Dirty().load();
                const bool needs_upload = dirty || force_dyn_upload;
                if (needs_upload) {
                    auto ensure_vertex_subref = [&](usize                              array_index,
                                                    const wallpaper::SceneVertexArray& vertex) {
                        if (vertex_bufs.size() <= array_index) {
                            vertex_bufs.resize(array_index + 1);
                        }

                        auto&      buf                = vertex_bufs[array_index];
                        const auto required_live_size = static_cast<VkDeviceSize>(
                            std::max<usize>(vertex.DataSizeOf(), vertex.OneSizeOf()));
                        if (buf && buf.size >= required_live_size) return true;

                        // Dynamic custom-shader meshes may grow after a pass was prepared. Keep
                        // this as a renderer-level buffer refresh mechanism for authored
                        // dynamic meshes, while first-class text is handled by TextPass and
                        // never enters CustomShaderPass as glyph helper nodes.
                        const auto required_size = GrowDynamicSuballocationSize(
                            buf ? buf.size : 0,
                            required_live_size,
                            static_cast<VkDeviceSize>(vertex.CapacitySizeOf()),
                            static_cast<VkDeviceSize>(vertex.OneSizeOf()));
                        if (required_size < required_live_size) {
                            LOG_ERROR("DynamicVertexUpload: live data exceeds capacity node='%s' "
                                      "live=%zu capacity=%zu",
                                      mesh.Material() != nullptr ? mesh.Material()->name.c_str()
                                                                 : "<unknown>",
                                      static_cast<size_t>(required_live_size),
                                      static_cast<size_t>(vertex.CapacitySizeOf()));
                            return false;
                        }
                        if (buf) {
                            dyn_buf->unallocateSubRef(buf);
                            buf = {};
                        }
                        if (! dyn_buf->allocateSubRef(required_size, buf)) {
                            return false;
                        }
                        force_dyn_upload = true;
                        return true;
                    };

                    auto release_unused_vertex_subrefs = [&]() {
                        while (vertex_bufs.size() > mesh.VertexCount()) {
                            auto& stale_buf = vertex_bufs.back();
                            if (stale_buf) dyn_buf->unallocateSubRef(stale_buf);
                            vertex_bufs.pop_back();
                        }
                    };

                    auto ensure_index_subref = [&](const wallpaper::SceneIndexArray& indice) {
                        const auto required_live_size = static_cast<VkDeviceSize>(
                            std::max<usize>(indice.DataSizeOf(), kDynamicIndexQuadFloorSize));
                        if (index_buf && index_buf.size >= required_live_size) return true;

                        const auto required_size = GrowDynamicSuballocationSize(
                            index_buf ? index_buf.size : 0,
                            required_live_size,
                            static_cast<VkDeviceSize>(indice.CapacitySizeof()),
                            static_cast<VkDeviceSize>(sizeof(uint32_t) * 6));
                        if (required_size < required_live_size) {
                            LOG_ERROR("DynamicIndexUpload: live data exceeds capacity live=%zu "
                                      "capacity=%zu",
                                      static_cast<size_t>(required_live_size),
                                      static_cast<size_t>(indice.CapacitySizeof()));
                            return false;
                        }
                        if (index_buf) {
                            dyn_buf->unallocateSubRef(index_buf);
                            index_buf = {};
                        }
                        if (! dyn_buf->allocateSubRef(required_size, index_buf)) {
                            return false;
                        }
                        force_dyn_upload = true;
                        return true;
                    };

                    release_unused_vertex_subrefs();
                    for (usize i = 0; i < mesh.VertexCount(); i++) {
                        const auto& vertex = mesh.GetVertexArray(i);
                        if (! ensure_vertex_subref(i, vertex)) {
                            mesh.SetDirty();
                            return;
                        }
                        auto& buf = vertex_bufs[i];
                        if (! dyn_buf->writeToBuf(
                                buf, { (uint8_t*)vertex.Data(), vertex.DataSizeOf() })) {
                            mesh.SetDirty();
                            return;
                        }
                    }
                    if (mesh.IndexCount() > 0) {
                        auto& indice = mesh.GetIndexArray(0);
                        if (! ensure_index_subref(indice)) {
                            mesh.SetDirty();
                            return;
                        }
                        draw_count = mesh.LogicalIndexCount();
                        if (mesh.IndexElementBytes() == 4) {
                            draw_count = static_cast<u32>(indice.RenderDataCount());
                        } else {
                            const u32 count = (u32)((indice.RenderDataCount() * 2) / 3);
                            draw_count      = count * 3;
                        }
                        auto& buf  = index_buf;
                        if (! dyn_buf->writeToBuf(
                                buf, { (uint8_t*)indice.Data(), indice.DataSizeOf() })) {
                            mesh.SetDirty();
                            return;
                        }
                    } else {
                        // Dynamic non-indexed meshes are still drawable. Text effect outputs
                        // use a four-vertex triangle-strip card that is resized in place when
                        // Date/Day/ Clock content changes; clearing draw_count here made the
                        // bridge and effect passes execute successfully while submitting no
                        // final composite geometry at all. The first vertex binding defines the
                        // vertex count for non-indexed draws, matching the static prepare
                        // path's draw contract.
                        draw_count = mesh.VertexCount() > 0
                                         ? static_cast<u32>(mesh.GetVertexArray(0).DataSize() /
                                                            mesh.GetVertexArray(0).OneSize())
                                         : 0;
                        if (index_buf) {
                            dyn_buf->unallocateSubRef(index_buf);
                            index_buf = {};
                        }
                    }
                    // Clearing the pass-local bootstrap flag only after all writes succeed
                    // keeps a freshly compiled dynamic pass from getting stuck with empty GPU
                    // buffers if an earlier upload attempt bails out partway through due to an
                    // allocation/write failure. Subsequent frames will keep retrying until the
                    // first complete upload lands in the new subranges.
                    mesh.Dirty().store(false);
                    force_dyn_upload = false;
                }
            };
        }

        auto  block  = ref.blocks.front();
        auto* buf    = rr.dyn_buf;
        auto* bufref = &m_desc.ubo_buf;
        auto* node           = m_desc.node;
        auto* shader_updater = scene.shaderValueUpdater.get();
        auto* extension      = m_extension;
        auto& sprites        = m_desc.sprites_map;
        auto& vk_textures    = m_desc.vk_textures;
        // Keep the material dependency explicit in the capture list because the updater writes the
        // authored uniforms directly and should not rediscover the material through the scene node.
        auto* material = mesh.Material();

        // Keep dynamic mesh uploads separate from general pass updates because both operations own
        // independent subranges of the shared staging buffer. They are nevertheless dispatched by
        // updateBeforeUpload(): every CPU write that feeds the current draw must happen before
        // VulkanRender records and flushes m_dyn_buf->recordUpload().
        m_desc.update_dynamic_mesh_op = update_dyn_buf_op;
        m_ubo_block   = block;
        m_ubo_staging = buf;
        m_ubo_ready   = true;
        m_desc.update_op =
            [shader_updater, block, buf, bufref, extension,
             node, material, &sprites, &vk_textures,
             camera_override = m_desc.camera_override,
             use_active_camera_for_uniforms = m_desc.use_active_camera_for_uniforms,
             use_active_camera_for_parallax = m_desc.use_active_camera_for_parallax,
             use_identity_model = m_desc.use_identity_model]() {
                auto update_unf_op = [block, buf, bufref, extension](
                                         std::string_view name, wallpaper::ShaderValue value) {
                    UpdateShaderDrawUniform(buf, *bufref, block, name, value);
                    if (extension != nullptr) extension->updateUniform(buf, name, value);
                };
                if (material != nullptr) {
                    WriteMaterialUniforms(buf, *bufref, block, *material);
                }
                const ShaderUniformOverrides overrides {
                    .camera_name = camera_override,
                    .use_camera_override = !camera_override.empty(),
                    .use_active_camera_for_uniforms = use_active_camera_for_uniforms,
                    .use_active_camera_for_parallax = use_active_camera_for_parallax,
                    .use_identity_model = use_identity_model,
                };
                shader_updater->UpdateUniforms(
                    node,
                    sprites,
                    update_unf_op,
                    (overrides.use_camera_override ||
                     overrides.use_active_camera_for_uniforms ||
                     overrides.use_identity_model)
                        ? &overrides
                        : nullptr);
                // update image slot for sprites
                {
                    for (auto& [i, sp] : sprites) {
                        if (i >= vk_textures.size()) continue;
                        vk_textures.at(i).active = sp.GetCurFrame().imageId;
                    }
                }
            };

        auto exists_unf_op = [&block](std::string_view name) {
            return exists(block.member_map, name);
        };
        shader_updater->InitUniforms(node, exists_unf_op);

        // memset uniform buf
        buf->fillBuf(*bufref, 0, bufref->size, 0);
        if (m_extension != nullptr) m_extension->initializeUniforms(buf);
        WriteMaterialUniforms(buf, *bufref, block, *mesh.Material());
        m_desc.update_op();
        if (m_desc.update_dynamic_mesh_op) m_desc.update_dynamic_mesh_op();
    }

    {
        m_desc.clear_value =
            BuildCustomShaderClearValue(scene, *mesh.Material(), m_desc.clear_before_draw);
    }
    return true;
}

bool ShaderDrawCore::prepareDeferred(Scene& scene, const Device& device, RenderingResources& rr) {
    if (requestDeferredPrepareResources(scene, device) == DeferredPrepareResourcesState::Waiting) {
        return false;
    }
    if (! StaticSceneTexturesResidentForDeferredPrepare(scene, device, m_desc, m_extension)) {
        return false;
    }
    return prepare(scene, device, rr);
}

DeferredPrepareResourcesState
ShaderDrawCore::requestDeferredPrepareResources(Scene& scene, const Device& device) {
    constexpr std::size_t kDeferredStaticTextureStageBudgetBytes = 64u * 1024u * 1024u;
    bool                  waiting                                = false;

    const auto request_texture = [&](std::string_view tex_name_view,
                                     std::optional<usize> priority_slot) {
        const std::string tex_name(tex_name_view);
        if (tex_name.empty()) return;
        if (scene.renderTargets.count(tex_name) != 0 || wallpaper::IsSpecTex(tex_name)) return;
        if (scene.dirtyImportedTextureKeys.count(tex_name) != 0) return;
        if (device.tex_cache().FindTex(tex_name).has_value()) return;

        const auto pending_streaming_state = device.tex_cache().StagePendingTexUploads(
            tex_name, kDeferredStaticTextureStageBudgetBytes);
        if (pending_streaming_state == TextureCacheStreamingState::Waiting) {
            waiting = true;
            return;
        }
        if (pending_streaming_state == TextureCacheStreamingState::Ready &&
            device.tex_cache().FindTex(tex_name).has_value()) {
            return;
        }

        const auto texture_it = scene.textures.find(tex_name);
        if (texture_it == scene.textures.end() || texture_it->second.isVideo) return;

        // Deferred visibility prepare follows the same split as modern streaming renderers:
        // expensive disk/decompression work is requested from the scene asset cache first, and the
        // render thread only builds Vulkan residency after those CPU bytes are ready. This keeps a
        // newly visible deferred layer from blocking the whole frame on WPTexImageParser::Parse().
        const auto request = scene.RequestParsedImageAsync(tex_name);
        switch (request.state) {
        case Scene::ParsedImageRequestState::Ready:
            if (request.image != nullptr) {
                const auto streaming_state = device.tex_cache().StageTexUploads(
                    request.image, priority_slot, kDeferredStaticTextureStageBudgetBytes);
                scene.DropParsedImageCache(tex_name);
                if (streaming_state == TextureCacheStreamingState::Waiting) {
                    waiting = true;
                } else if (streaming_state == TextureCacheStreamingState::Failed) {
                    LOG_ERROR("CustomShaderPassDeferredResources: staging failed node='%s' "
                              "texture='%s'",
                              m_desc.node != nullptr ? m_desc.node->Name().c_str() : "<null>",
                              tex_name.c_str());
                }
            }
            break;
        case Scene::ParsedImageRequestState::Pending: waiting = true; break;
        case Scene::ParsedImageRequestState::Failed:
            LOG_ERROR("CustomShaderPassDeferredResources: parse failed node='%s' texture='%s'",
                      m_desc.node != nullptr ? m_desc.node->Name().c_str() : "<null>",
                      tex_name.c_str());
            break;
        }
    };

    for (usize texture_index = 0; texture_index < m_desc.textures.size(); texture_index++) {
        std::optional<usize> priority_slot;
        if (const auto sprite_it = m_desc.sprites_map.find(texture_index);
            sprite_it != m_desc.sprites_map.end()) {
            const auto image_id = sprite_it->second.GetCurFrame().imageId;
            if (image_id >= 0) priority_slot = static_cast<usize>(image_id);
        }
        request_texture(m_desc.textures[texture_index], priority_slot);
    }
    if (m_extension != nullptr && m_desc.node != nullptr && m_desc.node->Mesh() != nullptr) {
        for (const auto texture : m_extension->resourceTextures(*m_desc.node->Mesh())) {
            request_texture(texture, std::nullopt);
        }
    }

    return waiting ? DeferredPrepareResourcesState::Waiting : DeferredPrepareResourcesState::Ready;
}

bool ShaderDrawCore::warmupPipeline(Scene& scene, const Device& device, RenderingResources& rr) {
    m_desc.scene = &scene;
    if (m_desc.node == nullptr || m_desc.node->Mesh() == nullptr ||
        m_desc.node->Mesh()->Material() == nullptr) {
        return false;
    }

    SceneMesh& mesh = *(m_desc.node->Mesh());
    if (m_extension != nullptr && ! m_extension->configure(device, m_desc, mesh)) return false;

    std::vector<Uni_ShaderSpv> spvs;
    DescriptorSetInfo          descriptor_info;
    ShaderReflected            ref;
    {
        SceneShader& shader = *(mesh.Material()->customShader.shader);
        if (! GenReflect(shader.codes, spvs, ref)) {
            LOG_ERROR("pipeline warmup reflect failed, %s", shader.name.c_str());
            return false;
        }

        auto& bindings = descriptor_info.bindings;
        bindings.resize(ref.binding_map.size());
        std::transform(
            ref.binding_map.begin(), ref.binding_map.end(), bindings.begin(), [](auto& item) {
                return item.second;
            });
        PopulateTextureBindingsFromReflection(m_desc, ref, mesh.Material()->textures.size());
    }

    std::vector<VkVertexInputBindingDescription>   bind_descriptions;
    std::vector<VkVertexInputAttributeDescription> attr_descriptions;
    for (uint i = 0; i < mesh.VertexCount(); i++) {
        const auto& vertex    = mesh.GetVertexArray(i);
        auto        attrs_map = vertex.GetAttrOffsetMap();

        bind_descriptions.push_back(VkVertexInputBindingDescription {
            .binding   = i,
            .stride    = static_cast<uint32_t>(vertex.OneSizeOf()),
            .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
        });

        for (auto& item : ref.input_location_map) {
            const auto& name     = item.first;
            const auto& input    = item.second;
            const bool  has_attr = exists(attrs_map, name);
            const auto  offset   = has_attr ? attrs_map[name].offset : 0;
            attr_descriptions.push_back(VkVertexInputAttributeDescription {
                .location = input.location,
                .binding  = i,
                .format   = input.format,
                .offset   = static_cast<uint32_t>(offset),
            });
        }
    }
    if (ShaderDrawCanUseMsaa(scene, m_desc.output, m_desc.node)) {
        m_desc.sample_count = static_cast<VkSampleCountFlagBits>(
            std::max(1, scene.MsaaSampleCount()));
        m_desc.resolve_msaa = false;
        if (mesh.Material() != nullptr) {
            m_desc.alpha_to_coverage = mesh.Material()->alpha_to_coverage;
        }
    }
    auto render_state = BuildCustomShaderRenderState(*mesh.Material(), m_desc);
    const auto attachment = ResolveShaderDrawAttachment(m_desc, m_extension);
    const auto color_initial =
        DestDrawOmsetInitialLayout(m_desc, render_state.color_load_op);
    auto opt = CreateShaderDrawRenderPass(device.handle(),
                                          VK_FORMAT_R8G8B8A8_UNORM,
                                          render_state.color_load_op,
                                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                          attachment,
                                          m_desc.sample_count,
                                          m_desc.resolve_msaa,
                                          color_initial);
    if (! opt.has_value()) return false;
    auto& pass = opt.value();

    descriptor_info.push_descriptor = true;
    GraphicsPipeline pipeline;
    pipeline.toDefault();
    pipeline.multisample.rasterizationSamples = m_desc.sample_count;
    pipeline.multisample.alphaToCoverageEnable =
        m_desc.alpha_to_coverage && m_desc.sample_count > VK_SAMPLE_COUNT_1_BIT;
    ApplyModelPipelineState(*mesh.Material(), m_desc, pipeline);
    m_desc.pipeline.debug_name =
        "CustomShaderPassWarmup[node=" +
        (m_desc.node != nullptr ? m_desc.node->Name() : std::string("(null)")) +
        ",output=" + m_desc.output + "]";
    m_desc.pipeline.cache_key = ShaderDrawPipelineCompatibilityKey(
        render_state.color_load_op,
        m_desc.model_pass,
        m_desc.clear_depth ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD,
        attachment,
        m_desc.sample_count,
        m_desc.resolve_msaa,
        color_initial);
    pipeline.addDescriptorSetInfo(spanone { descriptor_info })
        .setColorBlendStates(spanone { render_state.color_blend })
        .setTopology(ToTopology(mesh))
        .addInputBindingDescription(bind_descriptions)
        .addInputAttributeDescription(attr_descriptions);
    for (auto& spv : spvs) pipeline.addStage(std::move(spv));

    return pipeline.create(device, pass, m_desc.pipeline, rr.pipeline_cache.get());
}

bool ShaderDrawCore::refreshResources(Scene& scene, const Device& device,
                                      RenderingResources& rr) {
    // Resource refreshes reuse the pass object. Refresh the scene pointer first so dependency
    // checks observe the current render-target table and text bridge state without logging.
    m_desc.scene = &scene;
    // Resource-only refreshes intentionally keep the compiled shader pipeline, reflected bindings,
    // and uploaded mesh/UBO allocations intact. The expensive part that changes for effect-backed
    // minute updates is the texture-cache-backed image handle set and the framebuffer that wraps
    // the resized render target. Rebinding only those pieces avoids recompiling every shader pass
    // in the scene when the clock/date text changes shape.
    if (m_desc.node != nullptr && m_desc.node->Mesh() != nullptr) {
        auto& mesh = *m_desc.node->Mesh();
        // POSTFX_MESH 0x1401ea151 / TEXT_2F0 0x1402589da: dest-draw
        // HORIZONTAL +0x2e0 and VERTICAL +0x2e8 are object cards. TREE
        // ChangeMeshDataFrom onto a text-effect node leaves
        // SceneMesh::Dynamic() true, but dest-draw prepare stores those
        // verts on vertex_buf (not dyn_buf). Dirty last-pass AABB growth
        // must re-prepare that card; mesh.Dynamic() alone would skip it
        // and leave GPU verts on the parse-time 846 card (±423).
        const bool dest_draw_object_card =
            ! m_desc.dyn_vertex &&
            (m_desc.dest_draw_phase == wallpaper::DestDrawPhase::Leftover ||
             m_desc.dest_draw_phase == wallpaper::DestDrawPhase::PostFx ||
             m_desc.dest_draw_phase == wallpaper::DestDrawPhase::LastPass);
        if ((! mesh.Dynamic() || dest_draw_object_card) && mesh.Dirty().load()) {
            // Resource-only refreshes were originally written for effects whose geometry never
            // changes after graph build. Refactored text effects break that assumption: runtime
            // updates now mutate the static blur/compose quads of already-compiled effect passes.
            // If we only recreate textures/framebuffers here, the pass keeps drawing the old GPU
            // vertex buffer even though the SceneMesh carries the new map-rate-adjusted quad. By
            // dropping back to the normal prepare path for dirty static meshes we force the pass
            // to re-upload its mesh data and make runtime text-effect geometry changes actually
            // visible on screen.
            destroy(rr);
            return false;
        }
    }

    const auto output_target_it = scene.renderTargets.find(m_desc.output);
    if (output_target_it == scene.renderTargets.end()) {
        LOG_ERROR(
            "CustomShaderPassRefresh: output target not found before refresh node='%s' output='%s'",
            m_desc.node != nullptr ? m_desc.node->Name().c_str() : "<null>",
            m_desc.output.c_str());
        return false;
    }
    const auto previous_output_view   = m_desc.vk_output.view;
    const auto previous_output_extent = m_desc.vk_output.extent;
    const auto previous_samples       = m_desc.sample_count;
    const auto previous_resolve       = m_desc.resolve_msaa;
    const auto desired_output_key     = wallpaper::vulkan::ToTexKey(output_target_it->second);
    const bool output_extent_changed =
        previous_output_extent.width != static_cast<uint32_t>(desired_output_key.width) ||
        previous_output_extent.height != static_cast<uint32_t>(desired_output_key.height);
    const int intended_samples = IntendedShaderDrawSampleCount(m_desc);
    if (output_extent_changed || intended_samples != static_cast<int>(previous_samples)) {
        // Drop the framebuffer before TextureCache replaces `_rt_FullFrameBufferMultiSampled`.
        // Keeping a live framebuffer across that resize leaves a destroyed MSAA image attached
        // and the next submit waits on the frame fence forever.
        m_desc.fb.reset();
    }
    if (! RefreshCustomShaderPassTextures(scene, device, m_desc)) {
        LOG_ERROR("CustomShaderPassRefresh: texture refresh failed node='%s' output='%s'",
                  m_desc.node != nullptr ? m_desc.node->Name().c_str() : "<null>",
                  m_desc.output.c_str());
        return false;
    }
    if (m_desc.sample_count != previous_samples || m_desc.resolve_msaa != previous_resolve) {
        destroy(rr);
        return false;
    }
    if (m_extension != nullptr && ! m_extension->refreshTextures(scene, device, m_desc)) {
        return false;
    }
    const bool output_view_changed = previous_output_view != m_desc.vk_output.view;
    const bool framebuffer_missing = ! m_desc.fb;
    if (framebuffer_missing || output_extent_changed || output_view_changed) {
        if (! RecreateCustomShaderPassFramebuffer(device, rr, m_desc, m_extension)) {
            return false;
        }
    }
    if (m_desc.dyn_vertex && m_desc.update_dynamic_mesh_op != nullptr && m_desc.node != nullptr &&
        m_desc.node->Mesh() != nullptr &&
        (m_desc.force_dyn_upload || m_desc.node->Mesh()->Dirty().load())) {
        // Text-backed effect passes keep their render-graph topology stable while the final
        // source quad changes size. Uploading the dirty dynamic mesh during the resource-refresh
        // phase lets the compile-time dynamic-buffer copy include the new quad before the first
        // post-refresh draw, instead of binding a fresh suballocation that still contains old data.
        m_desc.update_dynamic_mesh_op();
    }
    return true;
}

bool ShaderDrawCore::refreshImportedTextureBindings(Scene& scene, const Device& device) {
    m_desc.scene = &scene;

    for (usize texture_index = 0; texture_index < m_desc.textures.size(); ++texture_index) {
        const auto& texture_key = m_desc.textures[texture_index];
        if (scene.dirtyImportedTextureResourceKeys.count(texture_key) == 0) continue;

        const auto cached_slots = device.tex_cache().FindTex(texture_key);
        if (!cached_slots.has_value() || cached_slots->slots.empty()) {
            LOG_ERROR("ImportedTexturePassRebind: cached texture missing layer=%d node='%s' "
                      "output='%s' slot=%zu key='%s'",
                      m_desc.layer_id,
                      m_desc.node != nullptr ? m_desc.node->Name().c_str() : "<null>",
                      m_desc.output.c_str(),
                      static_cast<size_t>(texture_index),
                      texture_key.c_str());
            return false;
        }

        if (m_desc.vk_textures.size() < m_desc.textures.size()) {
            m_desc.vk_textures.resize(m_desc.textures.size());
        }
        m_desc.vk_textures[texture_index] = *cached_slots;
    }

    bool extension_affected = false;
    if (m_extension != nullptr && m_desc.node != nullptr && m_desc.node->Mesh() != nullptr) {
        for (const auto texture_key : m_extension->resourceTextures(*m_desc.node->Mesh())) {
            if (scene.dirtyImportedTextureResourceKeys.count(std::string(texture_key)) == 0) {
                continue;
            }
            extension_affected = true;
            break;
        }
        if (extension_affected && !m_extension->refreshTextures(scene, device, m_desc)) {
            return false;
        }
    }
    return true;
}

void ShaderDrawCore::dropOutputFramebuffers() { m_desc.fb.reset(); }

void ShaderDrawCore::updateBeforeUpload() {
    if (m_desc.should_execute && ! m_desc.should_execute()) {
        return;
    }

    if (m_desc.node != nullptr && ! m_desc.node->LocalVisible()) {
        return;
    }

    const bool node_visible = m_desc.node == nullptr ? true : m_desc.node->Visible();
    if (m_desc.node != nullptr && ! node_visible && ! m_desc.execute_when_hidden) {
        return;
    }

    // recordUpload() flushes the mapped staging allocation, records the staging-to-GPU copies, and
    // clears its dirty ranges before any pass executes. Writing uniforms from execute() therefore
    // made every animated material value arrive one submitted frame late: on a media switch the new
    // current texture was visible while the blend pass still read the previous animation endpoint.
    // Update both the pass UBO and dynamic geometry here, in render-graph order, so the buffer copy
    // and the draw recorded for this submit describe one coherent frame.
    if (m_desc.update_op) m_desc.update_op();
    if (m_desc.update_dynamic_mesh_op) m_desc.update_dynamic_mesh_op();
}

void ShaderDrawCore::WriteUniform(std::string_view name, const ShaderValue& value) {
    // VERTICAL_MVP_ID 0x1400d8676 copies +0x930 into g_MVP after ENGINE_FLUSH.
    if (! m_ubo_ready || m_ubo_staging == nullptr) return;
    UpdateShaderDrawUniform(m_ubo_staging, m_desc.ubo_buf, m_ubo_block, name, value);
}

bool ShaderDrawCore::HasUniform(std::string_view name) const {
    return wallpaper::exists(m_ubo_block.member_map, name);
}

void ShaderDrawCore::execute(const Device& device, RenderingResources& rr) {
    const bool leftover_417 =
        m_desc.layer_id == 417 && m_desc.dest_draw_phase == DestDrawPhase::Leftover;
    const bool leftover_date =
        (m_desc.layer_id == 248 || m_desc.layer_id == 242) &&
        m_desc.dest_draw_phase == DestDrawPhase::Leftover;
    const bool leftover_image =
        (m_desc.layer_id == 1175 || m_desc.layer_id == 173 || m_desc.layer_id == 751 ||
         m_desc.layer_id == 4350) &&
        m_desc.dest_draw_phase == DestDrawPhase::Leftover;
    const bool date_dest =
        ((m_desc.layer_id == 248 || m_desc.layer_id == 242) &&
         m_desc.dest_draw_phase != DestDrawPhase::None) ||
        leftover_image;
    auto log_date_dest = [&](const char* skip) {
        if (!date_dest) return;
        float min_x = 0.0f;
        float max_x = 0.0f;
        float min_y = 0.0f;
        float max_y = 0.0f;
        uint32_t verts = 0;
        if (m_desc.node != nullptr && m_desc.node->Mesh() != nullptr &&
            m_desc.node->Mesh()->VertexCount() > 0) {
            const auto& va = m_desc.node->Mesh()->GetVertexArray(0);
            const float* data = va.Data();
            verts = static_cast<uint32_t>(va.VertexCount());
            if (data != nullptr && va.VertexCount() > 0 && va.OneSize() >= 2) {
                min_x = max_x = data[0];
                min_y = max_y = data[1];
                for (uint32_t i = 1; i < va.VertexCount(); ++i) {
                    const float* p = data + i * va.OneSize();
                    min_x = std::min(min_x, p[0]);
                    max_x = std::max(max_x, p[0]);
                    min_y = std::min(min_y, p[1]);
                    max_y = std::max(max_y, p[1]);
                }
            }
        }
        void* tex0_view = nullptr;
        void* tex1_view = nullptr;
        void* tex2_view = nullptr;
        if (!m_desc.vk_textures.empty() && !m_desc.vk_textures[0].slots.empty()) {
            tex0_view = reinterpret_cast<void*>(m_desc.vk_textures[0].getActive().view);
        }
        if (m_desc.vk_textures.size() > 1 && !m_desc.vk_textures[1].slots.empty()) {
            tex1_view = reinterpret_cast<void*>(m_desc.vk_textures[1].getActive().view);
        }
        if (m_desc.vk_textures.size() > 2 && !m_desc.vk_textures[2].slots.empty()) {
            tex2_view = reinterpret_cast<void*>(m_desc.vk_textures[2].getActive().view);
        }
        int tex0_rw = 0;
        int tex0_rh = 0;
        if (m_desc.scene != nullptr && !m_desc.textures.empty()) {
            const auto rt_it = m_desc.scene->renderTargets.find(m_desc.textures[0]);
            if (rt_it != m_desc.scene->renderTargets.end()) {
                tex0_rw = rt_it->second.width;
                tex0_rh = rt_it->second.height;
            }
        }
        const uint32_t vbufs = static_cast<uint32_t>(m_desc.vertex_bufs.size());
        const int immutable = m_desc.immutable_mesh ? 1 : 0;
        void* gpu = nullptr;
        if (m_desc.immutable_mesh != nullptr && !m_desc.immutable_mesh->vertices.empty()) {
            gpu = reinterpret_cast<void*>(m_desc.immutable_mesh->vertices[0].handle());
        } else if (m_desc.dyn_vertex && rr.dyn_buf != nullptr) {
            gpu = reinterpret_cast<void*>(rr.dyn_buf->gpuBuf());
        } else if (rr.vertex_buf != nullptr) {
            gpu = reinterpret_cast<void*>(rr.vertex_buf->gpuBuf());
        }
        const int omset =
            DestDrawOmsetInitialLayout(m_desc, VK_ATTACHMENT_LOAD_OP_DONT_CARE) ==
                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
                ? 1
                : 0;
        const uint64_t voff = vbufs > 0 ? static_cast<uint64_t>(m_desc.vertex_bufs[0].offset) : 0;
        const uint64_t vsize = vbufs > 0 ? static_cast<uint64_t>(m_desc.vertex_bufs[0].size) : 0;
        const int vvalid = vbufs > 0 && m_desc.vertex_bufs[0] ? 1 : 0;
        float stage_xy[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
        int stage_ok = 0;
        StagingBuffer* stage = m_desc.dyn_vertex ? rr.dyn_buf : rr.vertex_buf;
        if (stage != nullptr && vbufs > 0 && m_desc.vertex_bufs[0]) {
            const auto& va = (m_desc.node != nullptr && m_desc.node->Mesh() != nullptr &&
                              m_desc.node->Mesh()->VertexCount() > 0)
                                 ? &m_desc.node->Mesh()->GetVertexArray(0)
                                 : nullptr;
            const usize stride = va != nullptr ? va->OneSizeOf() : sizeof(float) * 5;
            std::array<uint8_t, 128> raw {};
            if (stage->peekBytes(m_desc.vertex_bufs[0], raw) && stride >= 8) {
                stage_ok = 1;
                for (int i = 0; i < 4; ++i) {
                    const float* p = reinterpret_cast<const float*>(raw.data() + i * stride);
                    stage_xy[i * 2] = p[0];
                    stage_xy[i * 2 + 1] = p[1];
                }
            }
        }
        LOG_INFO("DestDrawDateExec: id=%d phase=%d skip='%s' draw=%u verts=%u "
                 "extent=%ux%u mesh=[%.1f %.1f %.1f %.1f] output='%s' "
                 "out_view=%p tex0=%p tex1=%p tex2=%p ntex=%zu fb=%d "
                 "tex0_rt=%dx%d vbufs=%u immutable=%d gpu=%p dyn=%d omset=%d "
                 "vvalid=%d voff=%llu vsize=%llu ubo_off=%llu stage_ok=%d "
                 "stage=[%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f] "
                 "out_img=%p pipe=%p pass=%p key='%s'",
                 m_desc.layer_id,
                 static_cast<int>(m_desc.dest_draw_phase),
                 skip,
                 m_desc.draw_count,
                 verts,
                 m_desc.vk_output.extent.width,
                 m_desc.vk_output.extent.height,
                 min_x,
                 max_x,
                 min_y,
                 max_y,
                 m_desc.output.c_str(),
                 reinterpret_cast<void*>(m_desc.vk_output.view),
                 tex0_view,
                 tex1_view,
                 tex2_view,
                 m_desc.vk_textures.size(),
                 m_desc.fb ? 1 : 0,
                 tex0_rw,
                 tex0_rh,
                 vbufs,
                 immutable,
                 gpu,
                 m_desc.dyn_vertex ? 1 : 0,
                 omset,
                 vvalid,
                 static_cast<unsigned long long>(voff),
                 static_cast<unsigned long long>(vsize),
                 static_cast<unsigned long long>(m_desc.ubo_buf.offset),
                 stage_ok,
                 stage_xy[0],
                 stage_xy[1],
                 stage_xy[2],
                 stage_xy[3],
                 stage_xy[4],
                 stage_xy[5],
                 stage_xy[6],
                 stage_xy[7],
                 reinterpret_cast<void*>(m_desc.vk_output.handle),
                 m_desc.pipeline.handle ? reinterpret_cast<void*>(*m_desc.pipeline.handle)
                                        : nullptr,
                 m_desc.pipeline.pass ? reinterpret_cast<void*>(*m_desc.pipeline.pass)
                                      : nullptr,
                 m_desc.pipeline.cache_key.c_str());
        float res0[4] = { 0, 0, 0, 0 };
        float scale[2] = { 0, 0 };
        float mvp00 = 0;
        float mvp03 = 0;
        float mvp11 = 0;
        float mvp13 = 0;
        int ubo_ok = 0;
        if (m_ubo_staging != nullptr && m_desc.ubo_buf) {
            std::vector<uint8_t> ubo(static_cast<size_t>(m_desc.ubo_buf.size), 0);
            if (m_ubo_staging->peekBytes(m_desc.ubo_buf, ubo)) {
                ubo_ok = 1;
                auto read_f = [&](std::string_view name, float* out, size_t n) {
                    const auto it = m_ubo_block.member_map.find(std::string(name));
                    if (it == m_ubo_block.member_map.end()) return;
                    const size_t off = it->second.offset;
                    const size_t bytes = std::min(n * sizeof(float), it->second.size);
                    if (off + bytes > ubo.size()) return;
                    std::memcpy(out, ubo.data() + off, bytes);
                };
                read_f("g_Texture0Resolution", res0, 4);
                read_f("g_Scale", scale, 2);
                float mvp[16] = {};
                read_f("g_ModelViewProjectionMatrix", mvp, 16);
                mvp00 = mvp[0];
                mvp11 = mvp[5];
                mvp03 = mvp[12];
                mvp13 = mvp[13];
            }
        }
        LOG_INFO("DestDrawUbo: id=%d phase=%d ubo_ok=%d size=%llu off=%llu "
                 "res0=[%.1f %.1f %.1f %.1f] scale=[%.3f %.3f] "
                 "mvp00=%.6f mvp11=%.6f mvpT=[%.3f %.3f]",
                 m_desc.layer_id,
                 static_cast<int>(m_desc.dest_draw_phase),
                 ubo_ok,
                 static_cast<unsigned long long>(m_desc.ubo_buf.size),
                 static_cast<unsigned long long>(m_desc.ubo_buf.offset),
                 res0[0],
                 res0[1],
                 res0[2],
                 res0[3],
                 scale[0],
                 scale[1],
                 mvp00,
                 mvp11,
                 mvp03,
                 mvp13);
    };
    if (m_desc.should_execute && ! m_desc.should_execute()) {
        // Runtime-gated helper passes stay in the render graph so visibility flips do not rebuild
        // framebuffer topology. Returning before uniform updates and draw submission makes the pass
        // a true no-op on frames where its fallback branch is not active.
        if (leftover_417) LOG_INFO("DestDrawLeftoverExec: id=417 skip=should_execute");
        if (leftover_date)
            LOG_INFO("DestDrawLeftoverExec: id=%d skip=should_execute", m_desc.layer_id);
        log_date_dest("should_execute");
        return;
    }

    if (m_desc.node != nullptr && ! m_desc.node->LocalVisible()) {
        // execute_when_hidden is only for layer-level invisibility, such as offscreen dependency
        // sources that must keep rendering while their authored layer is hidden in the main scene.
        // Effect-local visibility is a stricter contract: a hidden effect must not run its shader
        // pass, otherwise the hidden branch would still overwrite the ping-pong output that the
        // bypass copy is responsible for preserving.
        if (leftover_417) LOG_INFO("DestDrawLeftoverExec: id=417 skip=local_visible");
        if (leftover_date)
            LOG_INFO("DestDrawLeftoverExec: id=%d skip=local_visible", m_desc.layer_id);
        log_date_dest("local_visible");
        return;
    }

    const bool node_visible = m_desc.node == nullptr ? true : m_desc.node->Visible();
    if (m_desc.node != nullptr && ! node_visible && ! m_desc.execute_when_hidden) {
        // The render graph has still reached this pass's ordering point even when authored
        // visibility turns the shader into a no-op for the frame. Releasing final-read keys here
        // prevents temporary render targets from staying pinned only because no draw was recorded.
        if (leftover_417) LOG_INFO("DestDrawLeftoverExec: id=417 skip=visible");
        if (leftover_date)
            LOG_INFO("DestDrawLeftoverExec: id=%d skip=visible", m_desc.layer_id);
        log_date_dest("visible");
        return;
    }
    if (leftover_417) {
        LOG_INFO("DestDrawLeftoverExec: id=417 draw=%u extent=%ux%u fb=%d",
                 m_desc.draw_count,
                 m_desc.vk_output.extent.width,
                 m_desc.vk_output.extent.height,
                 m_desc.fb ? 1 : 0);
    }
    if (leftover_date) {
        LOG_INFO("DestDrawLeftoverExec: id=%d draw=%u extent=%ux%u fb=%d",
                 m_desc.layer_id,
                 m_desc.draw_count,
                 m_desc.vk_output.extent.width,
                 m_desc.vk_output.extent.height,
                 m_desc.fb ? 1 : 0);
    }
    log_date_dest("");

    if (auto* scene = m_desc.scene != nullptr ? m_desc.scene : rr.scene;
        scene != nullptr && ShaderDrawSamplesResolvedDefault(m_desc.textures)) {
        ResolveComposeMsaaIfNeeded(*scene, device, rr);
    }

    auto&                   cmd    = rr.command;
    auto&                   outext = m_desc.vk_output.extent;
    const auto is_comparison_depth = [&](usize i) {
        if (i >= m_desc.textures.size()) return false;
        const auto& name = m_desc.textures[i];
        if (name == SpecTex_ShadowAtlas) return true;
        if (m_desc.scene == nullptr) return false;
        const auto it = m_desc.scene->renderTargets.find(name);
        return it != m_desc.scene->renderTargets.end() && it->second.comparisonDepth;
    };
    const auto push_visible_descriptors = [&](VkPipelineLayout layout) {
        for (usize i = 0; i < m_desc.vk_textures.size(); i++) {
            const int binding = m_desc.vk_tex_binding[i];
            if (binding < 0 || m_desc.vk_textures[i].slots.empty()) continue;
            const auto& image = m_desc.vk_textures[i].getActive();
            VkDescriptorImageInfo image_info {
                image.sampler,
                image.view,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            };
            VkWriteDescriptorSet write {
                .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext           = nullptr,
                .dstSet          = {},
                .dstBinding      = static_cast<uint32_t>(binding),
                .descriptorCount = 1,
                .descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo      = &image_info,
            };
            cmd.PushDescriptorSetKHR(VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, write);
        }
        if (m_desc.ubo_buf) {
            VkDescriptorBufferInfo buffer_info {
                rr.dyn_buf->gpuBuf(),
                m_desc.ubo_buf.offset,
                m_desc.ubo_buf.size,
            };
            VkWriteDescriptorSet write {
                .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext           = nullptr,
                .dstSet          = {},
                .dstBinding      = 0,
                .descriptorCount = 1,
                .descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .pBufferInfo     = &buffer_info,
            };
            cmd.PushDescriptorSetKHR(VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, write);
        }
    };
    push_visible_descriptors(*m_desc.pipeline.layout);

    for (usize i = 0; i < m_desc.vk_textures.size(); i++) {
        auto& slot = m_desc.vk_textures[i];
        if (slot.slots.empty()) continue;
        auto& img = slot.getActive();
        const bool comparison_depth = is_comparison_depth(i);
        VkImageSubresourceRange srang {
            .aspectMask     = comparison_depth ? VK_IMAGE_ASPECT_DEPTH_BIT
                                               : VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel   = 0,
            .levelCount     = VK_REMAINING_ARRAY_LAYERS,
            .baseArrayLayer = 0,
            .layerCount     = VK_REMAINING_MIP_LEVELS,
        };

        VkImageMemoryBarrier imb {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .pNext = nullptr,
            // Color inputs wait on the previous color-attachment write. The shadow atlas is a
            // depth comparison target written by late fragment tests, then sampled with
            // texSample2DCompare.
            .srcAccessMask    = comparison_depth ? VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT
                                                 : VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            .dstAccessMask    = VK_ACCESS_SHADER_READ_BIT,
            .oldLayout        = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            .newLayout        = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            .image            = img.handle,
            .subresourceRange = srang,
        };

        cmd.PipelineBarrier(comparison_depth ? VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT
                                             : VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                            VK_DEPENDENCY_BY_REGION_BIT,
                            imb);
    }

    // TEXT_E0_IDEST 0x1401e9681 / POSTFX_OMSET 0x1401ebf8c: OMSet leftover /
    // FullCompo as the color target. Dest-draw leftover and HORIZONTAL are
    // DONT_CARE after TextureCache bootstrap (SHADER_READ). Transition the
    // output to COLOR_ATTACHMENT so BeginRenderPass matches the OMSet analog.
    if (m_desc.vk_output.handle != VK_NULL_HANDLE &&
        DestDrawOmsetInitialLayout(m_desc, VK_ATTACHMENT_LOAD_OP_DONT_CARE) ==
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) {
        VkImageMemoryBarrier to_color {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                             VK_ACCESS_COLOR_ATTACHMENT_READ_BIT,
            .oldLayout        = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            .newLayout        = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .image            = m_desc.vk_output.handle,
            .subresourceRange =
                VkImageSubresourceRange {
                    .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
                    .baseMipLevel   = 0,
                    .levelCount     = VK_REMAINING_MIP_LEVELS,
                    .baseArrayLayer = 0,
                    .layerCount     = VK_REMAINING_ARRAY_LAYERS,
                },
        };
        cmd.PipelineBarrier(VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                                VK_PIPELINE_STAGE_TRANSFER_BIT,
                            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                            VK_DEPENDENCY_BY_REGION_BIT,
                            to_color);
    }

    m_desc.depth_clear_value.depthStencil = { m_desc.depth_clear, 0 };
    std::array<VkClearValue, 3> clear_values { m_desc.clear_value, m_desc.depth_clear_value, {} };
    uint32_t clear_count = 1;
    if (ResolveShaderDrawAttachment(m_desc, m_extension).enabled()) clear_count++;
    if (m_desc.resolve_msaa) clear_count++;
    VkRenderPassBeginInfo       pass_begin_info {
        .sType       = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .pNext       = nullptr,
        .renderPass  = *m_desc.pipeline.pass,
        .framebuffer = *m_desc.fb,
        .renderArea =
            VkRect2D {
                .offset = { 0, 0 },
                .extent = { outext.width, outext.height },
            },
        .clearValueCount = clear_count,
        .pClearValues    = clear_values.data(),
    };
    cmd.BeginRenderPass(pass_begin_info, VK_SUBPASS_CONTENTS_INLINE);

    cmd.BindPipeline(VK_PIPELINE_BIND_POINT_GRAPHICS, *m_desc.pipeline.handle);
    VkViewport viewport {
        .x        = 0,
        .y        = (float)outext.height,
        .width    = (float)outext.width,
        .height   = -(float)outext.height,
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };
    VkRect2D scissor { { 0, 0 }, { outext.width, outext.height } };

    cmd.SetViewport(0, viewport);
    cmd.SetScissor(0, scissor);

    if (m_desc.immutable_mesh) {
        for (usize i = 0; i < m_desc.immutable_mesh->vertices.size(); i++) {
            VkBuffer           gpu = m_desc.immutable_mesh->vertices[i].handle();
            const VkDeviceSize off = 0;
            if (gpu == VK_NULL_HANDLE) continue;
            cmd.BindVertexBuffers((u32)i, 1, &gpu, &off);
        }
        if (m_desc.immutable_mesh->has_index) {
            const VkIndexType index_type = m_desc.immutable_mesh->index_element_bytes == 4
                                               ? VK_INDEX_TYPE_UINT32
                                               : VK_INDEX_TYPE_UINT16;
            cmd.BindIndexBuffer(m_desc.immutable_mesh->index.handle(), 0, index_type);
            if (m_extension == nullptr) {
                cmd.DrawIndexed(m_desc.draw_count, 1, 0, 0, 0);
            } else {
                m_extension->recordIndexed(ShaderDrawRecordContext {
                    .data                     = m_desc,
                    .device                   = device,
                    .resources                = rr,
                    .push_visible_descriptors = push_visible_descriptors,
                });
            }
        } else {
            cmd.Draw(m_desc.draw_count, 1, 0, 0);
        }
    } else {
        auto gpu_buf = m_desc.dyn_vertex ? rr.dyn_buf->gpuBuf() : rr.vertex_buf->gpuBuf();

        for (usize i = 0; i < m_desc.vertex_bufs.size(); i++) {
            auto& buf = m_desc.vertex_bufs[i];
            cmd.BindVertexBuffers((u32)i, 1, &gpu_buf, &buf.offset);
        }
        if (m_desc.index_buf) {
            const VkIndexType index_type = m_desc.index_element_bytes == 4
                                               ? VK_INDEX_TYPE_UINT32
                                               : VK_INDEX_TYPE_UINT16;
            cmd.BindIndexBuffer(gpu_buf, m_desc.index_buf.offset, index_type);
            if (m_extension == nullptr) {
                cmd.DrawIndexed(m_desc.draw_count, 1, 0, 0, 0);
            } else {
                m_extension->recordIndexed(ShaderDrawRecordContext {
                    .data                     = m_desc,
                    .device                   = device,
                    .resources                = rr,
                    .push_visible_descriptors = push_visible_descriptors,
                });
            }
        } else {
            cmd.Draw(m_desc.draw_count, 1, 0, 0);
        }
    }

    cmd.EndRenderPass();

    if (m_desc.sample_count > VK_SAMPLE_COUNT_1_BIT &&
        m_desc.output == wallpaper::SpecTex_Default) {
        NoteComposeMsaaDraw(rr, m_desc.sample_count);
    }

    if (m_desc.model_pass && m_desc.sample_count > VK_SAMPLE_COUNT_1_BIT &&
        m_desc.depth_stencil_image_ref != nullptr && m_desc.depth_stencil_image_ref->handle) {
        auto resolved_it = rr.model_depth_resolved.find(m_desc.output);
        if (resolved_it != rr.model_depth_resolved.end() && resolved_it->second.handle) {
            auto& src = *m_desc.depth_stencil_image_ref;
            auto& dst = resolved_it->second;
            VkImageSubresourceRange range {
                .aspectMask     = VK_IMAGE_ASPECT_DEPTH_BIT,
                .baseMipLevel   = 0,
                .levelCount     = 1,
                .baseArrayLayer = 0,
                .layerCount     = 1,
            };
            VkImageMemoryBarrier bars[2] {
                {
                    .sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    .srcAccessMask    = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                    .dstAccessMask    = VK_ACCESS_TRANSFER_READ_BIT,
                    .oldLayout        = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                    .newLayout        = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    .image            = *src.handle,
                    .subresourceRange = range,
                },
                {
                    .sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    .srcAccessMask    = 0,
                    .dstAccessMask    = VK_ACCESS_TRANSFER_WRITE_BIT,
                    .oldLayout        = VK_IMAGE_LAYOUT_UNDEFINED,
                    .newLayout        = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    .image            = *dst.handle,
                    .subresourceRange = range,
                },
            };
            cmd.PipelineBarrier(VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                                VK_PIPELINE_STAGE_TRANSFER_BIT,
                                VK_DEPENDENCY_BY_REGION_BIT,
                                {},
                                {},
                                std::array { bars[0], bars[1] });
            VkImageResolve region {
                .srcSubresource = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1 },
                .dstSubresource = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1 },
                .extent         = src.extent,
            };
            std::array<VkImageResolve, 1> regions { region };
            cmd.ResolveImage(*src.handle,
                             VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                             *dst.handle,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             regions);
            VkImageMemoryBarrier after[2] {
                {
                    .sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    .srcAccessMask    = VK_ACCESS_TRANSFER_READ_BIT,
                    .dstAccessMask    = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                                     VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                    .oldLayout        = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    .newLayout        = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                    .image            = *src.handle,
                    .subresourceRange = range,
                },
                {
                    .sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    .srcAccessMask    = VK_ACCESS_TRANSFER_WRITE_BIT,
                    .dstAccessMask    = VK_ACCESS_SHADER_READ_BIT,
                    .oldLayout        = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    .newLayout        = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    .image            = *dst.handle,
                    .subresourceRange = range,
                },
            };
            cmd.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                                VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                    VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                VK_DEPENDENCY_BY_REGION_BIT,
                                {},
                                {},
                                std::array { after[0], after[1] });
        }
    }
    // Temporary render targets may only be returned to TextureCache after the pass has actually
    // consumed them in the recorded frame. Releasing during prepare/resource-refresh is unsafe:
    // all passes are prepared before any pass executes, so a later pass can accidentally bind a
    // same-sized but unrelated physical image for a still-live logical key.
}

void ShaderDrawCore::destroy(RenderingResources& rr) {
    m_desc.update_op              = {};
    m_desc.update_dynamic_mesh_op = {};
    if (m_extension != nullptr) m_extension->destroy(rr);
    // Retiring a hidden layer must drop framebuffer/image/buffer residency while leaving cached
    // PSO ownership to GraphicsPipelineStateCache. This mirrors game-engine visibility handling:
    // textures and render targets can be evicted, but the immutable shader pipeline remains warm
    // for the next show transition instead of recompiling on the visible frame.
    m_desc.fb.reset();
    m_desc.vk_textures.clear();
    m_desc.vk_tex_binding.clear();
    m_desc.vk_output              = {};
    m_desc.depth_stencil_image_ref = nullptr;
    m_desc.immutable_mesh.reset();
    auto* mesh_buf = m_desc.dyn_vertex ? rr.dyn_buf : rr.vertex_buf;
    for (auto& bufref : m_desc.vertex_bufs) {
        if (mesh_buf) mesh_buf->unallocateSubRef(bufref);
    }
    m_desc.vertex_bufs.clear();
    if (m_desc.index_buf && mesh_buf) {
        mesh_buf->unallocateSubRef(m_desc.index_buf);
        m_desc.index_buf = {};
    }
    rr.dyn_buf->unallocateSubRef(m_desc.ubo_buf);
    m_ubo_staging = nullptr;
    m_ubo_ready   = false;
    m_desc.ubo_buf = {};
}

void ShaderDrawCore::setTexture(u32 index, std::string_view tex_key) {
    assert(index < m_desc.textures.size());
    if (index >= m_desc.textures.size()) return;
    m_desc.textures[index] = tex_key;
}
