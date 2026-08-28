#include "TextPass.hpp"

#include "Scene/Scene.h"
#include "Scene/SceneMesh.h"
#include "Scene/SceneNode.h"
#include "Scene/SceneObject.h"
#include "Scene/SceneTextPrimitive.h"
#include "SpecTexs.hpp"
#include "Utils/Logging.h"
#include "Vulkan/ShaderComp.hpp"
#include "PassCommon.hpp"
#include "Msaa.hpp"
#include "Resource.hpp"
#include "ShaderDrawCore.hpp"
#include "WPSceneScriptMedia.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cstdint>

using namespace wallpaper::vulkan;

namespace
{
constexpr std::string_view kTextBackgroundTextureKey { "__text_layer_background_white" };

bool LeftoverNamedDestDraw(const TextPass::Desc& desc) {
    // TEXT_CLEARALPHA / TEXT_E0_IDEST: Date +0x320>0 leftover OMSet is
    // the AABB named-RT, not FullFB. Clock +0x320==0 leftover is FullFB
    // and does not take this arm.
    return desc.dest_draw_phase == wallpaper::DestDrawPhase::Leftover &&
           desc.output != wallpaper::SpecTex_Default;
}

std::string TextPipelineCompatibilityKey(VkAttachmentLoadOp load_op,
                                         wallpaper::BlendMode blend_mode,
                                         wallpaper::AlphaWritePolicy alpha_write_policy,
                                         VkSampleCountFlagBits sample_count,
                                         bool resolve_msaa) {
    // Text PSOs are shared by render-pass compatibility plus the full GraphicsPipeline descriptor,
    // not by the layer that first requested them. This keeps visibility toggles on the same model
    // as engine-level PSO caches while still letting hidden text release atlas/framebuffer memory.
    return "TextPass|format=rgba8|final=shader-read|store-vis=1|load=" +
           std::to_string(static_cast<int>(load_op)) +
           "|blend=" + std::to_string(static_cast<int>(blend_mode)) +
           "|alpha-policy=" + std::to_string(static_cast<int>(alpha_write_policy)) +
           "|samples=" + std::to_string(static_cast<int>(sample_count)) +
           "|resolve=" + (resolve_msaa ? std::string("1") : std::string("0"));
}

int IntendedTextSampleCount(const wallpaper::Scene* scene, std::string_view output) {
    // Official compose text writes `_rt_FullFrameBufferMultiSampled`. Effect
    // ping-pong / text-bridge targets stay 1x: official effect shaders sample
    // them as Texture2D, and the exe has no multisampled ping-pong RT name.
    if (scene != nullptr && wallpaper::vulkan::ComposeOutputUsesMsaa(*scene, output)) {
        return std::max(1, scene->MsaaSampleCount());
    }
    return 1;
}

struct TextPassUniforms {
    float model_view_projection[16] {};
    float color[4] {};
};

struct PreparedTextShaders {
    std::vector<Uni_ShaderSpv> stages;
};

struct TextVertexInputLayout {
    VkVertexInputBindingDescription binding {};
    std::array<VkVertexInputAttributeDescription, 2> attributes {};
};

std::optional<TextVertexInputLayout> ResolveMeshVertexInputLayout(
    const wallpaper::SceneMesh& source_mesh) {
    if (source_mesh.VertexCount() == 0) return std::nullopt;

    const auto& vertex = source_mesh.GetVertexArray(0);
    const auto  attrs = vertex.GetAttrOffsetMap();
    const auto  position_it = attrs.find(std::string(wallpaper::WE_IN_POSITION));
    const auto  texcoord_it = attrs.find(std::string(wallpaper::WE_IN_TEXCOORD));
    if (position_it == attrs.end() || texcoord_it == attrs.end()) {
        LOG_ERROR("TextPass: generated text mesh is missing required position/texcoord attributes");
        return std::nullopt;
    }

    TextVertexInputLayout layout;
    // SceneVertexArray pads each attribute to Wallpaper Engine's vec4-style storage contract.
    // TextPass used to hardcode FLOAT3+FLOAT2 as a tightly packed 5-float vertex, but the actual
    // generated buffer is 8 floats per vertex. Reading the live SceneVertexArray stride/offsets
    // here keeps the dedicated text primitive on the same canonical mesh layout as generic image
    // passes and prevents every vertex after the first one from being fetched at the wrong byte.
    layout.binding = VkVertexInputBindingDescription {
        .binding = 0,
        .stride = static_cast<uint32_t>(vertex.OneSizeOf()),
        .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
    };
    layout.attributes = {
        VkVertexInputAttributeDescription {
            .location = 0,
            .binding = 0,
            .format = VK_FORMAT_R32G32B32_SFLOAT,
            .offset = static_cast<uint32_t>(position_it->second.offset),
        },
        VkVertexInputAttributeDescription {
            .location = 1,
            .binding = 0,
            .format = VK_FORMAT_R32G32_SFLOAT,
            .offset = static_cast<uint32_t>(texcoord_it->second.offset),
        },
    };
    return layout;
}

std::optional<TextVertexInputLayout> ResolveTextVertexInputLayout(
    const wallpaper::SceneTextPrimitive& primitive) {
    const wallpaper::SceneMesh* source_mesh { nullptr };
    for (const auto& page : primitive.glyph_pages) {
        if (page.mesh != nullptr && page.mesh->VertexCount() > 0) {
            source_mesh = page.mesh.get();
            break;
        }
    }
    if (source_mesh == nullptr && primitive.background_mesh != nullptr &&
        primitive.background_mesh->VertexCount() > 0) {
        source_mesh = primitive.background_mesh.get();
    }
    if (source_mesh == nullptr) return std::nullopt;
    return ResolveMeshVertexInputLayout(*source_mesh);
}

std::optional<PreparedTextShaders> CompileTextShaders() {
    static const char* kVertexSource = R"(
[[vk::binding(0, 0)]] cbuffer TextUniformBlock {
    column_major float4x4 g_ModelViewProjectionMatrix;
    float4 g_Color4;
};

struct VSInput {
    [[vk::location(0)]] float3 a_Position : A_POSITION;
    [[vk::location(1)]] float2 a_TexCoord : A_TEXCOORD;
};

struct VSOutput {
    float4 position : SV_Position;
    [[vk::location(0)]] float2 v_TexCoord : TEXCOORD0;
    [[vk::location(1)]] float4 v_Color : COLOR0;
};

VSOutput main_vs(VSInput input) {
    VSOutput output;
    output.position = mul(g_ModelViewProjectionMatrix, float4(input.a_Position, 1.0));
    output.v_TexCoord = input.a_TexCoord;
    output.v_Color = g_Color4;
    return output;
}
)";

    static const char* kFragmentSource = R"(
[[vk::combinedImageSampler]][[vk::binding(1, 0)]] Texture2D<float4> g_Texture0;
[[vk::combinedImageSampler]][[vk::binding(1, 0)]] SamplerState g_Texture0_ww_sampler;

struct PSInput {
    float4 position : SV_Position;
    [[vk::location(0)]] float2 v_TexCoord : TEXCOORD0;
    [[vk::location(1)]] float4 v_Color : COLOR0;
};

float4 main_ps(PSInput input) : SV_Target0 {
    // Glyph atlas pages use R8 coverage. The shared background texture is white in every channel,
    // so reading red keeps glyph and background draws on the same dedicated text pipeline.
    const float coverage = g_Texture0.Sample(g_Texture0_ww_sampler, input.v_TexCoord).r;
    return float4(input.v_Color.rgb, input.v_Color.a * coverage);
}
)";

    ShaderCompOpt options {};
    options.target_env = ShaderTargetEnv::VULKAN_1_0;
    options.auto_map_locations = false;
    options.auto_map_bindings = false;

    std::array<ShaderCompUnit, 2> units {
        ShaderCompUnit {
            .stage = wallpaper::ShaderType::VERTEX,
            .source_language = ShaderSourceLanguage::HLSL,
            .debug_name = "TextPass.vert",
            .entry_point = "main_vs",
            .src = kVertexSource,
        },
        ShaderCompUnit {
            .stage = wallpaper::ShaderType::FRAGMENT,
            .source_language = ShaderSourceLanguage::HLSL,
            .debug_name = "TextPass.frag",
            .entry_point = "main_ps",
            .src = kFragmentSource,
        },
    };

    PreparedTextShaders prepared;
    if (!CompileAndLinkShaderUnits(units, options, prepared.stages)) {
        LOG_ERROR("TextPass: failed to compile dedicated text shaders");
        return std::nullopt;
    }
    return prepared;
}

std::optional<PreparedTextShaders> CompileClearalphaShaders() {
    // TEXT_CLEARALPHA official composelayer.vert/frag + CLEARALPHA=1.
    // gl_Position from a_TexCoord*2-1 (named-RT NDC). g_MVP only for
    // v_ScreenCoord FullFB sample. Not font.vert.
    static const char* kVertexSource = R"(
[[vk::binding(0, 0)]] cbuffer TextUniformBlock {
    column_major float4x4 g_ModelViewProjectionMatrix;
    float4 g_Color4;
};

struct VSInput {
    [[vk::location(0)]] float3 a_Position : A_POSITION;
    [[vk::location(1)]] float2 a_TexCoord : A_TEXCOORD;
};

struct VSOutput {
    float4 position : SV_Position;
    [[vk::location(0)]] float2 v_TexCoord : TEXCOORD0;
    [[vk::location(1)]] float3 v_ScreenCoord : TEXCOORD1;
};

VSOutput main_vs(VSInput input) {
    VSOutput output;
    float4 clip = mul(g_ModelViewProjectionMatrix, float4(input.a_Position, 1.0));
    output.v_ScreenCoord = float3(clip.x, -clip.y, clip.w);
    float2 ndc = float2(input.a_TexCoord.x, 1.0 - input.a_TexCoord.y) * 2.0 - 1.0;
    output.position = float4(ndc, 0.0, 1.0);
    output.v_TexCoord = input.a_TexCoord;
    return output;
}
)";

    static const char* kFragmentSource = R"(
[[vk::combinedImageSampler]][[vk::binding(1, 0)]] Texture2D<float4> g_Texture0;
[[vk::combinedImageSampler]][[vk::binding(1, 0)]] SamplerState g_Texture0_ww_sampler;

struct PSInput {
    float4 position : SV_Position;
    [[vk::location(0)]] float2 v_TexCoord : TEXCOORD0;
    [[vk::location(1)]] float3 v_ScreenCoord : TEXCOORD1;
};

float4 main_ps(PSInput input) : SV_Target0 {
    float2 uv = input.v_ScreenCoord.xy / input.v_ScreenCoord.z * 0.5 + 0.5;
    float4 color = g_Texture0.Sample(g_Texture0_ww_sampler, uv);
    color.a = 0.0;
    return color;
}
)";

    ShaderCompOpt options {};
    options.target_env = ShaderTargetEnv::VULKAN_1_0;
    options.auto_map_locations = false;
    options.auto_map_bindings = false;

    std::array<ShaderCompUnit, 2> units {
        ShaderCompUnit {
            .stage = wallpaper::ShaderType::VERTEX,
            .source_language = ShaderSourceLanguage::HLSL,
            .debug_name = "TextPassClearalpha.vert",
            .entry_point = "main_vs",
            .src = kVertexSource,
        },
        ShaderCompUnit {
            .stage = wallpaper::ShaderType::FRAGMENT,
            .source_language = ShaderSourceLanguage::HLSL,
            .debug_name = "TextPassClearalpha.frag",
            .entry_point = "main_ps",
            .src = kFragmentSource,
        },
    };

    PreparedTextShaders prepared;
    if (!CompileAndLinkShaderUnits(units, options, prepared.stages)) {
        LOG_ERROR("TextPass: failed to compile TEXT_CLEARALPHA composelayer");
        return std::nullopt;
    }
    return prepared;
}

bool BindTextPassOutput(wallpaper::Scene& scene, const Device& device, TextPass::Desc& desc) {
    desc.sample_count = VK_SAMPLE_COUNT_1_BIT;
    desc.resolve_msaa = false;
    desc.vk_resolve   = {};

    auto output_it = scene.renderTargets.find(desc.output);
    if (output_it == scene.renderTargets.end()) return false;
    auto& rt = output_it->second;

    // Direct clock/text that composites to `_rt_default` writes the same MS color
    // target as CustomShaderPass. A 1x write into the resolved image is overwritten
    // by the later compose resolve of `_rt_FullFrameBufferMultiSampled`.
    if (ComposeOutputUsesMsaa(scene, desc.output)) {
        const auto ms_name = std::string(wallpaper::SpecTex_DefaultMS);
        const auto ms_it   = scene.renderTargets.find(ms_name);
        if (ms_it != scene.renderTargets.end()) {
            if (auto ms_opt = device.tex_cache().Query(
                    ms_name, ToTexKey(ms_it->second), ! ms_it->second.allowReuse);
                ms_opt.has_value()) {
                desc.vk_output    = ms_opt.value();
                desc.sample_count = static_cast<VkSampleCountFlagBits>(
                    std::max(1u, ms_it->second.sample_count > 0
                                     ? static_cast<uint>(ms_it->second.sample_count)
                                     : 1u));
                desc.resolve_msaa = false;
                return true;
            }
        }
        LOG_ERROR("TextPass: MSAA compose target missing node='%s' output='%s'",
                  desc.node != nullptr ? desc.node->Name().c_str() : "<null>",
                  desc.output.c_str());
    }

    if (auto opt = device.tex_cache().Query(desc.output, ToTexKey(rt), ! rt.allowReuse);
        opt.has_value()) {
        desc.vk_output = opt.value();
        if ((desc.layer_id == 248 || desc.layer_id == 242) &&
            desc.dest_draw_phase == wallpaper::DestDrawPhase::Leftover) {
            LOG_INFO("DestDrawLeftoverTextBind: id=%d output='%s' rt=%dx%d query=%ux%u",
                     desc.layer_id,
                     desc.output.c_str(),
                     rt.width,
                     rt.height,
                     desc.vk_output.extent.width,
                     desc.vk_output.extent.height);
        }
        return true;
    }
    return false;
}

void WriteMatrixToUniform(TextPassUniforms& uniforms, const Eigen::Matrix4f& matrix) {
    for (int column = 0; column < 4; column++) {
        for (int row = 0; row < 4; row++) {
            uniforms.model_view_projection[column * 4 + row] = matrix(row, column);
        }
    }
}

std::shared_ptr<wallpaper::Image> ResolveTextBackgroundImage() {
    // The direct text pipeline only needs one non-glyph texture: a 1x1 white coverage image for
    // the optional opaque background quad. Materializing it here keeps the text pass self-owned
    // and avoids routing primitive text rendering through unrelated image-parser infrastructure.
    static const std::shared_ptr<wallpaper::Image> image =
        wallpaper::CreateSceneScriptSolidImage(kTextBackgroundTextureKey, { 255, 255, 255, 255 });
    return image;
}

bool LoadTextPassTexture(const Device&                        device,
                         const std::shared_ptr<wallpaper::Image>& image,
                         ImageSlotsRef*                       out_slots) {
    if (out_slots == nullptr) return false;
    if (image == nullptr) {
        *out_slots = {};
        return true;
    }

    *out_slots = device.tex_cache().CreateTex(*image);
    return !out_slots->slots.empty();
}

bool CreateTextPipelineForPrimitive(const Device&                         device,
                                    RenderingResources&                   rr,
                                    const wallpaper::SceneTextPrimitive&  primitive,
                                    bool                                  offscreen_output,
                                    VkAttachmentLoadOp                    color_load_op,
                                    wallpaper::AlphaWritePolicy           alpha_write_policy,
                                    VkSampleCountFlagBits                 sample_count,
                                    bool                                  resolve_msaa,
                                    std::string                           debug_name,
                                    PipelineParameters&                   pipeline_parameters) {
    auto render_pass = CreateShaderDrawRenderPass(
        device.handle(),
        VK_FORMAT_R8G8B8A8_UNORM,
        color_load_op,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        {},
        sample_count,
        resolve_msaa);
    if (!render_pass.has_value()) return false;

    const auto compiled_shaders = CompileTextShaders();
    if (!compiled_shaders.has_value()) return false;

    DescriptorSetInfo descriptor_info;
    descriptor_info.push_descriptor = true;
    descriptor_info.bindings = {
        VkDescriptorSetLayoutBinding {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        },
        VkDescriptorSetLayoutBinding {
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
        },
    };

    const auto vertex_layout = ResolveTextVertexInputLayout(primitive);
    if (!vertex_layout.has_value()) return false;

    // colorBlendMode 31 is Wallpaper Engine's fixed-function additive case. Shader blend modes
    // 1..30 are handled later by the independent final passthrough, so their text source remains an
    // ordinary translucent offscreen raster.
    const auto blend_mode =
        !offscreen_output && primitive.object.colorBlendMode == 31
            ? wallpaper::BlendMode::Additive
            : wallpaper::BlendMode::Translucent;
    VkPipelineColorBlendAttachmentState blend_state {};
    blend_state.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT;
    if (offscreen_output) blend_state.colorWriteMask |= VK_COLOR_COMPONENT_A_BIT;
    SetBlend(blend_mode, blend_state);
    if (offscreen_output && alpha_write_policy != wallpaper::AlphaWritePolicy::Preserve) {
        // Composition attachments keep their authored RGB blend but use an explicit coverage
        // equation. copybackground=false selects Alpha-MAX so later transparent draws cannot erase
        // coverage accumulated by earlier routed children.
        blend_state.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        if (alpha_write_policy == wallpaper::AlphaWritePolicy::Max) {
            blend_state.alphaBlendOp = VK_BLEND_OP_MAX;
            blend_state.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        } else {
            blend_state.alphaBlendOp = VK_BLEND_OP_ADD;
            blend_state.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        }
    }

    GraphicsPipeline pipeline;
    pipeline.toDefault();
    pipeline.multisample.rasterizationSamples =
        sample_count > VK_SAMPLE_COUNT_1_BIT ? sample_count : VK_SAMPLE_COUNT_1_BIT;
    pipeline_parameters.debug_name = std::move(debug_name);
    pipeline_parameters.cache_key = TextPipelineCompatibilityKey(
        color_load_op, blend_mode, alpha_write_policy, sample_count, resolve_msaa);
    pipeline.addDescriptorSetInfo(std::span<const DescriptorSetInfo>(&descriptor_info, 1))
        .setColorBlendStates(std::span<const VkPipelineColorBlendAttachmentState>(&blend_state, 1))
        .setTopology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
        .addInputBindingDescription(
            std::span<const VkVertexInputBindingDescription>(&vertex_layout->binding, 1))
        .addInputAttributeDescription(vertex_layout->attributes);
    for (const auto& stage : compiled_shaders->stages) {
        if (!stage) continue;
        pipeline.addStage(Uni_ShaderSpv(new ShaderSpv(*stage)));
    }

    return pipeline.create(device, *render_pass, pipeline_parameters, rr.pipeline_cache.get());
}

bool CreateClearalphaPipeline(const Device&                        device,
                              const wallpaper::SceneMesh&          card_mesh,
                              VkSampleCountFlagBits                sample_count,
                              bool                                 resolve_msaa,
                              vvk::RenderPass&                     leftover_pass,
                              PipelineParameters&                  pipeline_parameters) {
    // TEXT_CLEARALPHA 0x140258a14 / TEXT_E0_IDEST 0x1401e968a: leftover
    // named-RT OMSet 0x1401e9681 then +0x5b0 card. Same VkRenderPass as
    // leftover TEXT_E8 glyphs. Vertex layout is the AABB card, not glyphs.
    (void)resolve_msaa;
    const auto compiled_shaders = CompileClearalphaShaders();
    if (!compiled_shaders.has_value()) return false;

    DescriptorSetInfo descriptor_info;
    descriptor_info.push_descriptor = true;
    descriptor_info.bindings = {
        VkDescriptorSetLayoutBinding {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        },
        VkDescriptorSetLayoutBinding {
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
        },
    };

    // TEXT_CLEARALPHA publishes +0x5b0 ±half AABB, not glyph verts.
    const auto vertex_layout = ResolveMeshVertexInputLayout(card_mesh);
    if (!vertex_layout.has_value()) return false;

    VkPipelineColorBlendAttachmentState blend_state {};
    blend_state.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                 VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    SetBlend(wallpaper::BlendMode::Normal, blend_state);

    GraphicsPipeline pipeline;
    pipeline.toDefault();
    pipeline.multisample.rasterizationSamples =
        sample_count > VK_SAMPLE_COUNT_1_BIT ? sample_count : VK_SAMPLE_COUNT_1_BIT;
    pipeline_parameters.debug_name = "TextPassClearalpha";
    pipeline_parameters.cache_key.clear();
    pipeline.addDescriptorSetInfo(std::span<const DescriptorSetInfo>(&descriptor_info, 1))
        .setColorBlendStates(std::span<const VkPipelineColorBlendAttachmentState>(&blend_state, 1))
        .setTopology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP)
        .addInputBindingDescription(
            std::span<const VkVertexInputBindingDescription>(&vertex_layout->binding, 1))
        .addInputAttributeDescription(vertex_layout->attributes);
    for (const auto& stage : compiled_shaders->stages) {
        if (!stage) continue;
        pipeline.addStage(Uni_ShaderSpv(new ShaderSpv(*stage)));
    }
    return pipeline.create(device, leftover_pass, pipeline_parameters, nullptr, false);
}

bool BindClearalphaFullfb(wallpaper::Scene& scene, const Device& device, ImageSlotsRef& slots) {
    // Official composelayer_clearalpha textures[0] is _rt_FullFrameBuffer.
    const auto name = std::string(wallpaper::SpecTex_Default);
    const auto it = scene.renderTargets.find(name);
    if (it == scene.renderTargets.end()) return false;
    auto opt = device.tex_cache().Query(name, ToTexKey(it->second), !it->second.allowReuse);
    if (!opt.has_value()) return false;
    slots.slots.clear();
    slots.slots.push_back(opt.value());
    slots.active = 0;
    return !slots.slots.empty();
}

std::array<float, 4> ResolveTextColor(const wallpaper::SceneTextPrimitive& primitive,
                                      bool                                 background) {
    if (background) {
        return {
            primitive.object.backgroundcolor[0] * primitive.object.backgroundbrightness,
            primitive.object.backgroundcolor[1] * primitive.object.backgroundbrightness,
            primitive.object.backgroundcolor[2] * primitive.object.backgroundbrightness,
            primitive.object.alpha,
        };
    }
    return {
        primitive.object.color[0],
        primitive.object.color[1],
        primitive.object.color[2],
        primitive.object.alpha,
    };
}
} // namespace

TextPass::TextPass(const Desc& desc) {
    // The pass description intentionally stores only the authored/runtime identity fields here.
    // Vulkan handles such as framebuffers and pipeline objects are non-copyable and must always be
    // created during `prepare()` against the live device, so the constructor avoids copying any of
    // the prepared-state members from the temporary render-graph description.
    m_desc.scene               = desc.scene;
    m_desc.node                = desc.node;
    m_desc.layer_id            = desc.layer_id;
    m_desc.execute_when_hidden = desc.execute_when_hidden;
    m_desc.dest_draw_phase     = desc.dest_draw_phase;
    m_desc.output              = desc.output;
    m_desc.alpha_write_policy  = desc.alpha_write_policy;
}
TextPass::~TextPass() = default;

std::string TextPass::residencyKey() const {
    return "TextPass|node=" + std::to_string(reinterpret_cast<std::uintptr_t>(m_desc.node)) +
           "|layer=" + std::to_string(m_desc.layer_id) + "|output=" + m_desc.output;
}

bool TextPass::canReuseForResidency(const VulkanPass& next_pass) const {
    const auto* next = dynamic_cast<const TextPass*>(&next_pass);
    if (next == nullptr) return false;
    // The text pipeline depends on the text primitive's vertex layout and output target, both
    // represented by the stable node/layer/output residency key. Visibility gates and the live
    // scene pointer are safe to absorb without recreating shader modules or descriptor layouts.
    const int this_samples = static_cast<int>(m_desc.sample_count);
    const int next_samples = IntendedTextSampleCount(next->m_desc.scene, next->m_desc.output);
    return residencyKey() == next->residencyKey() &&
           m_desc.execute_when_hidden == next->m_desc.execute_when_hidden &&
           m_desc.alpha_write_policy == next->m_desc.alpha_write_policy &&
           this_samples == next_samples &&
           ! m_desc.resolve_msaa;
}

void TextPass::absorbResidencyGraphState(const VulkanPass& next_pass) {
    const auto* next = dynamic_cast<const TextPass*>(&next_pass);
    if (next == nullptr) return;
    m_desc.scene               = next->m_desc.scene;
    m_desc.node                = next->m_desc.node;
    m_desc.layer_id            = next->m_desc.layer_id;
    m_desc.execute_when_hidden = next->m_desc.execute_when_hidden;
    m_desc.dest_draw_phase     = next->m_desc.dest_draw_phase;
    m_desc.output              = next->m_desc.output;
    m_desc.alpha_write_policy  = next->m_desc.alpha_write_policy;
}

void TextPass::writeLastPassMvp(const Eigen::Matrix4f& mvp) {
    // Date leftover DEST_ORTHO_TNF +0x930 is dest-ortho * I. Clock
    // TEXT_VT_F0 FONT_MVP +0x930 is FitOrtho * dest-STACK (ENGINE_FLUSH).
    m_dest_ortho_mvp = mvp;
    m_has_dest_ortho_mvp = true;
}

bool TextPass::referencesRenderTarget(std::string_view render_target) const {
    // A text pass only owns its compose/bridge output. Glyph atlas pages are imported texture-cache
    // entries, not render-graph targets. Compose text also writes `_rt_FullFrameBufferMultiSampled`
    // when MSAA is on, so that RT must refresh this pass.
    return m_desc.output == render_target ||
           (m_desc.output == wallpaper::SpecTex_Default &&
            render_target == wallpaper::SpecTex_DefaultMS);
}

bool TextPass::referencesTextLayer(int32_t layer_id) const {
    // Runtime text rerasters are scoped by authored layer id. Matching that id here lets a direct
    // Clock-style text pass refresh its atlas and mesh before command recording without touching
    // unrelated text layers that happen to draw to the same final render target.
    return layer_id != 0 && m_desc.layer_id == layer_id;
}

bool TextPass::refreshTextures(const Device& device) {
    const auto* primitive =
        m_desc.node != nullptr ? m_desc.node->Text() : nullptr;
    if (primitive == nullptr) return false;

    if (!LoadTextPassTexture(device, ResolveTextBackgroundImage(), &m_desc.background_texture)) {
        return false;
    }

    m_desc.page_textures.resize(primitive->glyph_pages.size());
    for (size_t page_index = 0; page_index < primitive->glyph_pages.size(); page_index++) {
        if (!LoadTextPassTexture(device,
                                 primitive->layout.glyph_pages[page_index].image,
                                 &m_desc.page_textures[page_index])) {
            return false;
        }
    }
    m_loaded_atlas_version = primitive->atlas_version;
    return true;
}

bool TextPass::recreateFramebuffer(const Device& device) {
    m_desc.framebuffer.reset();
    if (!m_desc.pipeline.pass || m_desc.vk_output.view == VK_NULL_HANDLE ||
        m_desc.vk_output.extent.width == 0 || m_desc.vk_output.extent.height == 0) {
        LOG_ERROR("TextPassRefresh: cannot recreate framebuffer node='%s' output='%s' "
                  "hasRenderPass=%s hasView=%s extent=[%u,%u]",
                  m_desc.node != nullptr ? m_desc.node->Name().c_str() : "<null>",
                  m_desc.output.c_str(),
                  m_desc.pipeline.pass ? "true" : "false",
                  m_desc.vk_output.view != VK_NULL_HANDLE ? "true" : "false",
                  m_desc.vk_output.extent.width,
                  m_desc.vk_output.extent.height);
        return false;
    }
    if (m_desc.resolve_msaa && m_desc.vk_resolve.view == VK_NULL_HANDLE) {
        LOG_ERROR("TextPassRefresh: missing MSAA resolve view node='%s' output='%s'",
                  m_desc.node != nullptr ? m_desc.node->Name().c_str() : "<null>",
                  m_desc.output.c_str());
        return false;
    }
    std::array<VkImageView, 2> attachments { m_desc.vk_output.view, m_desc.vk_resolve.view };
    VkFramebufferCreateInfo info {
        .sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
        .renderPass      = *m_desc.pipeline.pass,
        .attachmentCount = m_desc.resolve_msaa ? 2u : 1u,
        .pAttachments    = attachments.data(),
        .width           = m_desc.vk_output.extent.width,
        .height          = m_desc.vk_output.extent.height,
        .layers          = 1,
    };
    return device.handle().CreateFramebuffer(info, m_desc.framebuffer) == VK_SUCCESS;
}

bool TextPass::ensureMeshBuffers(SceneMesh& mesh, MeshBuffers& buffers, RenderingResources& rr) {
    auto* dyn_buf = rr.dyn_buf;
    if (dyn_buf == nullptr) return false;

    while (buffers.vertex_bufs.size() > mesh.VertexCount()) {
        dyn_buf->unallocateSubRef(buffers.vertex_bufs.back());
        buffers.vertex_bufs.pop_back();
    }
    buffers.vertex_bufs.resize(mesh.VertexCount());

    for (usize array_index = 0; array_index < mesh.VertexCount(); array_index++) {
        const auto& vertex = mesh.GetVertexArray(array_index);
        auto&       subref = buffers.vertex_bufs[array_index];
        const auto  required_size =
            static_cast<VkDeviceSize>(std::max<usize>(vertex.CapacitySizeOf(), vertex.OneSizeOf()));
        if (!subref || subref.size < required_size) {
            if (subref) dyn_buf->unallocateSubRef(subref);
            if (!dyn_buf->allocateSubRef(required_size, subref)) return false;
            buffers.force_upload = true;
        }
    }

    if (mesh.IndexCount() > 0) {
        const auto& index = mesh.GetIndexArray(0);
        const auto  required_size =
            static_cast<VkDeviceSize>(std::max<usize>(index.CapacitySizeof(), sizeof(uint16_t) * 6));
        if (!buffers.index_buf || buffers.index_buf.size < required_size) {
            if (buffers.index_buf) dyn_buf->unallocateSubRef(buffers.index_buf);
            if (!dyn_buf->allocateSubRef(required_size, buffers.index_buf)) return false;
            buffers.force_upload = true;
        }
    } else if (buffers.index_buf) {
        dyn_buf->unallocateSubRef(buffers.index_buf);
        buffers.index_buf = {};
    }

    const bool needs_upload = mesh.Dirty().load() || buffers.force_upload;
    if (!needs_upload) return true;

    for (usize array_index = 0; array_index < mesh.VertexCount(); array_index++) {
        const auto& vertex = mesh.GetVertexArray(array_index);
        auto&       subref = buffers.vertex_bufs[array_index];
        if (!dyn_buf->writeToBuf(
                subref,
                { reinterpret_cast<uint8_t*>(const_cast<float*>(vertex.Data())), vertex.DataSizeOf() })) {
            return false;
        }
    }

    if (mesh.IndexCount() > 0) {
        const auto& index = mesh.GetIndexArray(0);
        buffers.draw_count = static_cast<uint32_t>((index.RenderDataCount() * 2) / 3) * 3;
        if (!dyn_buf->writeToBuf(buffers.index_buf,
                                 { reinterpret_cast<uint8_t*>(const_cast<uint32_t*>(index.Data())),
                                   index.DataSizeOf() })) {
            return false;
        }
    } else {
        buffers.draw_count = mesh.VertexCount() > 0
            ? static_cast<uint32_t>(mesh.GetVertexArray(0).VertexCount())
            : 0;
    }

    mesh.Dirty().store(false);
    buffers.force_upload = false;
    return true;
}

void TextPass::prepare(Scene& scene, const Device& device, RenderingResources& rr) {
    const auto* primitive =
        m_desc.node != nullptr ? m_desc.node->Text() : nullptr;
    if (primitive == nullptr) return;

    if (!refreshTextures(device)) return;

    // Text bridge render targets can resize while the TextPass object is intentionally kept alive.
    // The existing framebuffer references the old TextureCache image view, so it must be released
    // before `Query()` is allowed to replace the backing image for this output.
    m_desc.framebuffer.reset();
    if (!BindTextPassOutput(scene, device, m_desc)) return;

    const bool offscreen_output = m_desc.output != wallpaper::SpecTex_Default;
    // TEXT_E0_IDEST then TEXT_E8: leftover named-RT glyphs LOAD after
    // clearalpha. Clock leftover FullFB stays LOAD (offscreen false).
    const bool leftover_named = LeftoverNamedDestDraw(m_desc);
    const VkAttachmentLoadOp color_load_op =
        leftover_named ? VK_ATTACHMENT_LOAD_OP_LOAD
                       : (offscreen_output ? VK_ATTACHMENT_LOAD_OP_CLEAR
                                           : VK_ATTACHMENT_LOAD_OP_LOAD);
    m_desc.clear_output = color_load_op == VK_ATTACHMENT_LOAD_OP_CLEAR;
    const auto debug_name =
        "TextPass[node=" + (m_desc.node != nullptr ? m_desc.node->Name() : std::string("(null)")) +
        ",output=" + m_desc.output + "]";
    if (!CreateTextPipelineForPrimitive(
            device,
            rr,
            *primitive,
            offscreen_output,
            color_load_op,
            m_desc.alpha_write_policy,
            m_desc.sample_count,
            m_desc.resolve_msaa,
            debug_name,
            m_desc.pipeline)) {
        return;
    }
    m_has_clearalpha = false;
    m_clearalpha_fullfb = {};
    if (leftover_named) {
        auto* dest_object = scene.FindSceneObject(m_desc.layer_id);
        wallpaper::SceneMesh* card =
            dest_object != nullptr ? dest_object->lastpass_mesh() : nullptr;
        if (card == nullptr ||
            m_desc.pipeline.cached_state == nullptr ||
            !m_desc.pipeline.cached_state->pass ||
            !CreateClearalphaPipeline(device, *card, m_desc.sample_count,
                                      m_desc.resolve_msaa,
                                      m_desc.pipeline.cached_state->pass,
                                      m_clearalpha_pipeline)) {
            LOG_ERROR("TextPass: TEXT_CLEARALPHA pipeline failed node='%s'",
                      m_desc.node != nullptr ? m_desc.node->Name().c_str() : "<null>");
            return;
        }
        if (!BindClearalphaFullfb(scene, device, m_clearalpha_fullfb)) {
            LOG_ERROR("TextPass: TEXT_CLEARALPHA FullFB missing node='%s'",
                      m_desc.node != nullptr ? m_desc.node->Name().c_str() : "<null>");
            return;
        }
        m_has_clearalpha = true;
    }
    if (!recreateFramebuffer(device)) return;

    rr.dyn_buf->allocateSubRef(sizeof(TextPassUniforms),
                               m_desc.ubo_buf,
                               device.limits().minUniformBufferOffsetAlignment);
    if (leftover_named) {
        // TEXT_E0_IDEST flush and TEXT_E8 glyphs share one OMSet. DestDraw
        // recordUpload runs before execute, so leftover uniforms must live
        // in a second slot: clearalpha g_MVP is LastPassMvp (+0x930),
        // glyphs are dest-ortho (DEST_ORTHO_TNF after TEXT_E0).
        rr.dyn_buf->allocateSubRef(sizeof(TextPassUniforms),
                                   m_clearalpha_ubo_buf,
                                   device.limits().minUniformBufferOffsetAlignment);
    }

    if (primitive->background_mesh != nullptr) {
        m_background_buffers.force_upload = true;
        if (!ensureMeshBuffers(*primitive->background_mesh, m_background_buffers, rr)) return;
    }
    m_page_buffers.resize(primitive->glyph_pages.size());
    for (size_t page_index = 0; page_index < primitive->glyph_pages.size(); page_index++) {
        m_page_buffers[page_index].force_upload = true;
        if (!ensureMeshBuffers(*primitive->glyph_pages[page_index].mesh,
                               m_page_buffers[page_index],
                               rr)) {
            return;
        }
    }

    // The dedicated text pass only requests the shared transform uniform contract. All visual text
    // state such as glyph color and background color comes directly from the text primitive, so no
    // generic image-material bootstrap is involved anymore.
    if (scene.shaderValueUpdater != nullptr && m_desc.node != nullptr) {
        scene.shaderValueUpdater->InitUniforms(
            m_desc.node,
            [](std::string_view uniform_name) {
                return uniform_name == wallpaper::G_MVP;
            });
    }

    m_desc.clear_value = VkClearValue {
        .color = {
            offscreen_output ? 0.0f : scene.clearColor[0],
            offscreen_output ? 0.0f : scene.clearColor[1],
            offscreen_output ? 0.0f : scene.clearColor[2],
            offscreen_output ? 0.0f : 1.0f,
        },
    };
    setPrepared();
}

bool TextPass::warmupPipeline(Scene& scene, const Device& device, RenderingResources& rr) {
    const auto* primitive =
        m_desc.node != nullptr ? m_desc.node->Text() : nullptr;
    if (primitive == nullptr) return false;

    const bool offscreen_output = m_desc.output != wallpaper::SpecTex_Default;
    const bool leftover_named = LeftoverNamedDestDraw(m_desc);
    const VkAttachmentLoadOp color_load_op =
        leftover_named ? VK_ATTACHMENT_LOAD_OP_LOAD
                       : (offscreen_output ? VK_ATTACHMENT_LOAD_OP_CLEAR
                                           : VK_ATTACHMENT_LOAD_OP_LOAD);
    const int intended_samples = IntendedTextSampleCount(&scene, m_desc.output);
    const auto sample_count = static_cast<VkSampleCountFlagBits>(intended_samples);
    const bool resolve_msaa = false;
    const auto debug_name =
        "TextPassWarmup[node=" +
        (m_desc.node != nullptr ? m_desc.node->Name() : std::string("(null)")) +
        ",output=" + m_desc.output + "]";
    return CreateTextPipelineForPrimitive(device,
                                          rr,
                                          *primitive,
                                          offscreen_output,
                                          color_load_op,
                                          m_desc.alpha_write_policy,
                                          sample_count,
                                          resolve_msaa,
                                          debug_name,
                                          m_desc.pipeline);
}

void TextPass::dropOutputFramebuffers() { m_desc.framebuffer.reset(); }

void TextPass::refreshResources(Scene& scene, const Device& device, RenderingResources& rr) {
    const int intended_samples = IntendedTextSampleCount(&scene, m_desc.output);
    if (static_cast<int>(m_desc.sample_count) != intended_samples || m_desc.resolve_msaa) {
        destory(device, rr);
        return;
    }
    if (!refreshTextures(device)) {
        LOG_ERROR("TextPassRefresh: texture refresh failed node='%s' output='%s'",
                  m_desc.node != nullptr ? m_desc.node->Name().c_str() : "<null>",
                  m_desc.output.c_str());
        setPrepared(false);
        return;
    }
    auto* primitive = m_desc.node != nullptr ? m_desc.node->Text() : nullptr;
    if (primitive == nullptr) {
        LOG_ERROR("TextPassRefresh: missing primitive node='%s' output='%s'",
                  m_desc.node != nullptr ? m_desc.node->Name().c_str() : "<null>",
                  m_desc.output.c_str());
        setPrepared(false);
        return;
    }

    // Resource refresh happens before the next draw command records its dynamic-buffer upload.
    // Rebuilding and writing text meshes here keeps resized bridge text from binding freshly
    // allocated subranges that have not been copied to the GPU yet, which was the reason
    // effect-backed Date/Clock/Day could disappear immediately after a layout update.
    if (primitive->background_mesh != nullptr &&
        !ensureMeshBuffers(*primitive->background_mesh, m_background_buffers, rr)) {
        LOG_ERROR("TextPassRefresh: background mesh upload failed node='%s' output='%s'",
                  m_desc.node != nullptr ? m_desc.node->Name().c_str() : "<null>",
                  m_desc.output.c_str());
        setPrepared(false);
        return;
    }
    if (m_page_buffers.size() != primitive->glyph_pages.size()) {
        m_page_buffers.resize(primitive->glyph_pages.size());
        for (auto& buffers : m_page_buffers) buffers.force_upload = true;
    }
    for (size_t page_index = 0; page_index < primitive->glyph_pages.size(); page_index++) {
        if (!ensureMeshBuffers(*primitive->glyph_pages[page_index].mesh,
                               m_page_buffers[page_index],
                               rr)) {
            LOG_ERROR("TextPassRefresh: glyph mesh upload failed node='%s' output='%s' page=%zu",
                      m_desc.node != nullptr ? m_desc.node->Name().c_str() : "<null>",
                      m_desc.output.c_str(),
                      page_index);
            setPrepared(false);
            return;
        }
    }
    const auto previous_output_view = m_desc.vk_output.view;
    const auto previous_output_extent = m_desc.vk_output.extent;
    const auto bind_name = ComposeOutputUsesMsaa(scene, m_desc.output)
                               ? std::string(wallpaper::SpecTex_DefaultMS)
                               : m_desc.output;
    if (const auto bind_it = scene.renderTargets.find(bind_name);
        bind_it != scene.renderTargets.end() &&
        (previous_output_extent.width != static_cast<uint32_t>(bind_it->second.width) ||
         previous_output_extent.height != static_cast<uint32_t>(bind_it->second.height))) {
        m_desc.framebuffer.reset();
    }
    if (!BindTextPassOutput(scene, device, m_desc)) {
        setPrepared(false);
        return;
    }
    const bool output_extent_changed =
        previous_output_extent.width != m_desc.vk_output.extent.width ||
        previous_output_extent.height != m_desc.vk_output.extent.height;
    const bool output_view_changed = previous_output_view != m_desc.vk_output.view;
    if (output_extent_changed || output_view_changed || !m_desc.framebuffer) {
        if (!recreateFramebuffer(device)) {
            setPrepared(false);
            return;
        }
    }
    if (m_has_clearalpha && !BindClearalphaFullfb(scene, device, m_clearalpha_fullfb)) {
        LOG_ERROR("TextPassRefresh: TEXT_CLEARALPHA FullFB missing node='%s'",
                  m_desc.node != nullptr ? m_desc.node->Name().c_str() : "<null>");
        setPrepared(false);
    }
}

void TextPass::execute(const Device& device, RenderingResources& rr) {
    auto* node = m_desc.node;
    auto* primitive = node != nullptr ? node->Text() : nullptr;
    const bool leftover_date =
        (m_desc.layer_id == 248 || m_desc.layer_id == 242) &&
        m_desc.dest_draw_phase == DestDrawPhase::Leftover;
    if (primitive == nullptr) {
        if (leftover_date) LOG_INFO("DestDrawLeftoverText: id=%d skip=no_primitive", m_desc.layer_id);
        return;
    }
    if (!m_desc.pipeline.handle || !m_desc.framebuffer) {
        if (leftover_date)
            LOG_INFO("DestDrawLeftoverText: id=%d skip=unprepared pipe=%d fb=%d",
                     m_desc.layer_id,
                     m_desc.pipeline.handle ? 1 : 0,
                     m_desc.framebuffer ? 1 : 0);
        return;
    }
    if (node != nullptr && !node->Visible() && !m_desc.execute_when_hidden) {
        if (leftover_date) LOG_INFO("DestDrawLeftoverText: id=%d skip=visible", m_desc.layer_id);
        return;
    }

    if (primitive->atlas_version != m_loaded_atlas_version ||
        m_desc.page_textures.size() != primitive->glyph_pages.size()) {
        // Text atlas content is owned by the scene primitive, not by render-graph pass creation.
        // Runtime text updates can therefore swap atlas pages or change page counts without a
        // graph rebuild. Refreshing the bound atlas images lazily here keeps the dedicated text
        // pass on the new scene-owned source of truth instead of depending on parser-time texture
        // registration.
        if (!refreshTextures(device)) {
            if (leftover_date)
                LOG_INFO("DestDrawLeftoverText: id=%d skip=refresh_textures", m_desc.layer_id);
            return;
        }
    }

    if (primitive->background_mesh != nullptr &&
        !ensureMeshBuffers(*primitive->background_mesh, m_background_buffers, rr)) {
        return;
    }
    // TEXT_LAYOUT_VERTS 0..AABB glyphs belong to the named-RT leftover only:
    // DEST_ORTHO_TNF maps 0..AABB into named-RT NDC. Clock TEXT_VT_F0
    // (+0x320==0) leftover stays on FullFB, where the Draw uses the compose
    // ±half glyph layout under the LastPassDrawMvp +0x8f0 stand-in; a 0..AABB
    // card there lands at the fit-ortho 0..AABB corner (bottom-left desktop).
    const bool leftover_layout_local =
        LeftoverNamedDestDraw(m_desc) &&
        !primitive->leftover_glyph_pages.empty();
    const auto& glyph_pages =
        leftover_layout_local ? primitive->leftover_glyph_pages : primitive->glyph_pages;
    if (m_page_buffers.size() != glyph_pages.size()) {
        m_page_buffers.resize(glyph_pages.size());
        for (auto& buffers : m_page_buffers) buffers.force_upload = true;
    }
    for (size_t page_index = 0; page_index < glyph_pages.size(); page_index++) {
        if (glyph_pages[page_index].mesh == nullptr) {
            if (leftover_date)
                LOG_INFO("DestDrawLeftoverText: id=%d skip=null_mesh page=%zu",
                         m_desc.layer_id,
                         page_index);
            return;
        }
        if (!ensureMeshBuffers(*glyph_pages[page_index].mesh, m_page_buffers[page_index], rr)) {
            if (leftover_date)
                LOG_INFO("DestDrawLeftoverText: id=%d skip=mesh_upload page=%zu",
                         m_desc.layer_id,
                         page_index);
            return;
        }
    }
    wallpaper::SceneObject* dest_object =
        m_desc.scene != nullptr ? m_desc.scene->FindSceneObject(m_desc.layer_id) : nullptr;
    wallpaper::SceneMesh* clearalpha_mesh =
        m_has_clearalpha && dest_object != nullptr ? dest_object->lastpass_mesh() : nullptr;
    if (clearalpha_mesh != nullptr &&
        !ensureMeshBuffers(*clearalpha_mesh, m_clearalpha_buffers, rr)) {
        if (leftover_date)
            LOG_INFO("DestDrawLeftoverText: id=%d skip=clearalpha_mesh", m_desc.layer_id);
        return;
    }
    if (leftover_date) {
        float uv_min_u = 0.0f;
        float uv_max_u = 0.0f;
        float uv_min_v = 0.0f;
        float uv_max_v = 0.0f;
        if (clearalpha_mesh != nullptr && clearalpha_mesh->VertexCount() > 0) {
            const auto& va = clearalpha_mesh->GetVertexArray(0);
            const auto attrs = va.GetAttrOffsetMap();
            const auto uv_it = attrs.find(std::string(wallpaper::WE_IN_TEXCOORD));
            const float* data = va.Data();
            if (uv_it != attrs.end() && data != nullptr && va.VertexCount() > 0) {
                const usize uv_off = uv_it->second.offset / sizeof(float);
                uv_min_u = uv_max_u = data[uv_off];
                uv_min_v = uv_max_v = data[uv_off + 1];
                for (uint32_t i = 1; i < va.VertexCount(); ++i) {
                    const float* p = data + i * va.OneSize() + uv_off;
                    uv_min_u = std::min(uv_min_u, p[0]);
                    uv_max_u = std::max(uv_max_u, p[0]);
                    uv_min_v = std::min(uv_min_v, p[1]);
                    uv_max_v = std::max(uv_max_v, p[1]);
                }
            }
        }
        void* fullfb_view = nullptr;
        void* fullfb_img = nullptr;
        uint32_t fullfb_w = 0;
        uint32_t fullfb_h = 0;
        if (!m_clearalpha_fullfb.slots.empty()) {
            const auto& image = m_clearalpha_fullfb.getActive();
            fullfb_view = reinterpret_cast<void*>(image.view);
            fullfb_img = reinterpret_cast<void*>(image.handle);
            fullfb_w = image.extent.width;
            fullfb_h = image.extent.height;
        }
        const uint64_t glyph_off =
            !m_page_buffers.empty() && !m_page_buffers[0].vertex_bufs.empty()
                ? static_cast<uint64_t>(m_page_buffers[0].vertex_bufs[0].offset)
                : 0;
        const uint64_t glyph_size =
            !m_page_buffers.empty() && !m_page_buffers[0].vertex_bufs.empty()
                ? static_cast<uint64_t>(m_page_buffers[0].vertex_bufs[0].size)
                : 0;
        const uint64_t ca_off =
            !m_clearalpha_buffers.vertex_bufs.empty()
                ? static_cast<uint64_t>(m_clearalpha_buffers.vertex_bufs[0].offset)
                : 0;
        const uint64_t ca_size =
            !m_clearalpha_buffers.vertex_bufs.empty()
                ? static_cast<uint64_t>(m_clearalpha_buffers.vertex_bufs[0].size)
                : 0;
        LOG_INFO("DestDrawLeftoverText: id=%d draw pages=%zu leftover_pages=%d "
                 "extent=%ux%u dest_ortho=%d clearalpha=%d clearalpha_draw=%u "
                 "uv=[%.2f %.2f %.2f %.2f] output='%s' view=%p out_img=%p "
                 "fullfb_view=%p fullfb_img=%p fullfb=%ux%u same_fullfb=%d "
                 "glyph_off=%llu glyph_size=%llu ca_off=%llu ca_size=%llu "
                 "ubo_off=%llu ca_ubo_off=%llu",
                 m_desc.layer_id,
                 glyph_pages.size(),
                 leftover_layout_local ? 1 : 0,
                 m_desc.vk_output.extent.width,
                 m_desc.vk_output.extent.height,
                 m_has_dest_ortho_mvp ? 1 : 0,
                 m_has_clearalpha && clearalpha_mesh != nullptr ? 1 : 0,
                 m_clearalpha_buffers.draw_count,
                 uv_min_u,
                 uv_max_u,
                 uv_min_v,
                 uv_max_v,
                 m_desc.output.c_str(),
                 reinterpret_cast<void*>(m_desc.vk_output.view),
                 reinterpret_cast<void*>(m_desc.vk_output.handle),
                 fullfb_view,
                 fullfb_img,
                 fullfb_w,
                 fullfb_h,
                 fullfb_view != nullptr &&
                         fullfb_view == reinterpret_cast<void*>(m_desc.vk_output.view)
                     ? 1
                     : 0,
                 static_cast<unsigned long long>(glyph_off),
                 static_cast<unsigned long long>(glyph_size),
                 static_cast<unsigned long long>(ca_off),
                 static_cast<unsigned long long>(ca_size),
                 static_cast<unsigned long long>(m_desc.ubo_buf.offset),
                 static_cast<unsigned long long>(
                     m_clearalpha_ubo_buf ? m_clearalpha_ubo_buf.offset : 0));
    }

    auto write_uniforms = [&](const std::array<float, 4>& color) {
        TextPassUniforms uniforms {};
        if (m_has_dest_ortho_mvp) {
            // DEST_ORTHO_TNF leftover dest=I dest-ortho, not private cam.
            WriteMatrixToUniform(uniforms, m_dest_ortho_mvp);
        } else if (m_desc.scene != nullptr && m_desc.scene->shaderValueUpdater != nullptr && node != nullptr) {
            sprite_map_t sprites;
            m_desc.scene->shaderValueUpdater->UpdateUniforms(
                node,
                sprites,
                [&uniforms](std::string_view name, wallpaper::ShaderValue value) {
                    if (name != wallpaper::G_MVP || value.size() < 16) return;
                    Eigen::Matrix4f matrix = Eigen::Matrix4f::Identity();
                    for (int column = 0; column < 4; column++) {
                        for (int row = 0; row < 4; row++) {
                            // ShaderValue uses a size_t index while the matrix loops are small
                            // signed integers; materializing the index keeps warning-clean builds
                            // without changing the column-major uniform contract.
                            const auto uniform_index = static_cast<size_t>(column * 4 + row);
                            matrix(row, column) = value[uniform_index];
                        }
                    }
                    WriteMatrixToUniform(uniforms, matrix);
                });
        }
        std::copy(color.begin(), color.end(), uniforms.color);
        rr.dyn_buf->writeToBuf(m_desc.ubo_buf,
                               { reinterpret_cast<uint8_t*>(const_cast<TextPassUniforms*>(&uniforms)),
                                 sizeof(uniforms) });
    };

    auto bind_uniforms = [&]() {
        VkDescriptorBufferInfo buffer_info {
            rr.dyn_buf->gpuBuf(),
            m_desc.ubo_buf.offset,
            m_desc.ubo_buf.size,
        };
        VkWriteDescriptorSet write {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .pBufferInfo = &buffer_info,
        };
        rr.command.PushDescriptorSetKHR(VK_PIPELINE_BIND_POINT_GRAPHICS, *m_desc.pipeline.layout, 0, write);
    };

    auto bind_texture = [&](const ImageSlotsRef& slots) {
        if (slots.slots.empty()) return;
        const auto& image = slots.getActive();
        VkDescriptorImageInfo image_info {
            image.sampler,
            image.view,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        };
        VkWriteDescriptorSet write {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo = &image_info,
        };
        rr.command.PushDescriptorSetKHR(VK_PIPELINE_BIND_POINT_GRAPHICS, *m_desc.pipeline.layout, 0, write);
    };

    const bool leftover_named = LeftoverNamedDestDraw(m_desc);
    const bool do_clearalpha =
        leftover_named && m_has_clearalpha && m_clearalpha_pipeline.handle &&
        m_clearalpha_pipeline.layout && m_clearalpha_ubo_buf &&
        clearalpha_mesh != nullptr && dest_object != nullptr && m_desc.scene != nullptr;
    if (do_clearalpha) {
        // TEXT_E0_FLUSH930 0x1401e968a: flush before I_SLOT 0x1401e96ac /
        // dest=I 0x1401e9702 / DEST_ORTHO_TNF 0x1401e9768. camera=FitOrtho
        // (0x1401e937e) dest=Path B dest-STACK (0x1401e935c). +0x1ca=1 so
        // ENGINE_FLUSH +0x930=camera*dest=LastPassMvp. composelayer g_MVP
        // is id 0xb copies +0x930 (UNIFORM_UPLOAD_MAP). Not +0x8f0
        // LastPassDrawMvp. FetchDest stays I-only (TEXT_E0_IDEST).
        TextPassUniforms uniforms {};
        WriteMatrixToUniform(uniforms, m_desc.scene->LastPassMvp());
        uniforms.color[0] = 1.0f;
        uniforms.color[1] = 1.0f;
        uniforms.color[2] = 1.0f;
        uniforms.color[3] = 0.0f;
        rr.dyn_buf->writeToBuf(
            m_clearalpha_ubo_buf,
            { reinterpret_cast<uint8_t*>(const_cast<TextPassUniforms*>(&uniforms)),
              sizeof(uniforms) });
    }
    if (leftover_named) {
        // DEST_ORTHO_TNF 0x1401e9768 then TEXT_E8: glyphs +0x930 is
        // dest-ortho * I. Same OMSet as TEXT_E0, so both UBOs flush
        // before BeginRenderPass (ENGINE_FLUSH 0x1400d4200 analog).
        write_uniforms(ResolveTextColor(*primitive, false));
        rr.dyn_buf->recordUpload(rr.command);
    }

    const VkExtent2D output_extent {
        .width = m_desc.vk_output.extent.width,
        .height = m_desc.vk_output.extent.height,
    };
    std::array<VkClearValue, 2> clear_values { m_desc.clear_value, {} };
    VkRenderPassBeginInfo begin_info {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .renderPass = *m_desc.pipeline.pass,
        .framebuffer = *m_desc.framebuffer,
        .renderArea = VkRect2D { .offset = { 0, 0 }, .extent = output_extent },
        .clearValueCount = m_desc.resolve_msaa ? 2u : 1u,
        .pClearValues = clear_values.data(),
    };
    rr.command.BeginRenderPass(begin_info, VK_SUBPASS_CONTENTS_INLINE);
    VkViewport viewport {
        .x = 0.0f,
        .y = static_cast<float>(m_desc.vk_output.extent.height),
        .width = static_cast<float>(m_desc.vk_output.extent.width),
        .height = -static_cast<float>(m_desc.vk_output.extent.height),
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };
    VkRect2D scissor { { 0, 0 }, output_extent };
    rr.command.SetViewport(0, viewport);
    rr.command.SetScissor(0, scissor);
    if (m_has_clearalpha && m_clearalpha_pipeline.handle && m_clearalpha_pipeline.layout &&
        clearalpha_mesh != nullptr && dest_object != nullptr && m_desc.scene != nullptr) {
        // TEXT_E0_IDEST 0x1401e968a: I=FetchDest (vt+0x80), flush
        // composelayer_clearalpha, publish +0x5b0 ±half AABB
        // (TEXT_CLEARALPHA_UV a_TexCoord 1,1). g_MVP is LastPassMvp
        // +0x930 (TEXT_E0_FLUSH930). Not leftover dest-ortho and not
        // LastPassDrawMvp +0x8f0.
        rr.command.BindPipeline(VK_PIPELINE_BIND_POINT_GRAPHICS, *m_clearalpha_pipeline.handle);
        {
            const StagingBufferRef& clearalpha_ubo =
                m_clearalpha_ubo_buf ? m_clearalpha_ubo_buf : m_desc.ubo_buf;
            VkDescriptorBufferInfo buffer_info {
                rr.dyn_buf->gpuBuf(),
                clearalpha_ubo.offset,
                clearalpha_ubo.size,
            };
            VkWriteDescriptorSet ubo_write {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstBinding = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .pBufferInfo = &buffer_info,
            };
            rr.command.PushDescriptorSetKHR(VK_PIPELINE_BIND_POINT_GRAPHICS,
                                            *m_clearalpha_pipeline.layout, 0, ubo_write);
            if (!m_clearalpha_fullfb.slots.empty()) {
                const auto& image = m_clearalpha_fullfb.getActive();
                VkDescriptorImageInfo image_info {
                    image.sampler,
                    image.view,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                };
                VkWriteDescriptorSet tex_write {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstBinding = 1,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .pImageInfo = &image_info,
                };
                rr.command.PushDescriptorSetKHR(VK_PIPELINE_BIND_POINT_GRAPHICS,
                                                *m_clearalpha_pipeline.layout, 0, tex_write);
            }
        }
        {
            auto gpu_buf = rr.dyn_buf->gpuBuf();
            for (usize binding_index = 0; binding_index < m_clearalpha_buffers.vertex_bufs.size();
                 binding_index++) {
                auto& subref = m_clearalpha_buffers.vertex_bufs[binding_index];
                rr.command.BindVertexBuffers(static_cast<uint32_t>(binding_index), 1, &gpu_buf,
                                             &subref.offset);
            }
            rr.command.Draw(m_clearalpha_buffers.draw_count, 1, 0, 0);
        }
    }
    rr.command.BindPipeline(VK_PIPELINE_BIND_POINT_GRAPHICS, *m_desc.pipeline.handle);

    auto draw_mesh = [&](MeshBuffers& buffers, const ImageSlotsRef& texture, const std::array<float, 4>& color) {
        if (buffers.draw_count == 0) return;
        if (!leftover_named) write_uniforms(color);
        bind_uniforms();
        bind_texture(texture);
        auto gpu_buf = rr.dyn_buf->gpuBuf();
        for (usize binding_index = 0; binding_index < buffers.vertex_bufs.size(); binding_index++) {
            auto& subref = buffers.vertex_bufs[binding_index];
            rr.command.BindVertexBuffers(static_cast<uint32_t>(binding_index), 1, &gpu_buf, &subref.offset);
        }
        // Glyph page meshes are indexed, while the optional opaque background is a plain strip.
        // Supporting both draw modes keeps the direct text primitive self-contained instead of
        // depending on the old generic image pass behavior for one half of the text renderable.
        if (buffers.index_buf) {
            rr.command.BindIndexBuffer(gpu_buf, buffers.index_buf.offset, VK_INDEX_TYPE_UINT16);
            rr.command.DrawIndexed(buffers.draw_count, 1, 0, 0, 0);
        } else {
            rr.command.Draw(buffers.draw_count, 1, 0, 0);
        }
    };

    if (primitive->object.opaquebackground && primitive->background_mesh != nullptr) {
        draw_mesh(m_background_buffers,
                  m_desc.background_texture,
                  ResolveTextColor(*primitive, true));
    }

    for (size_t page_index = 0; page_index < glyph_pages.size(); page_index++) {
        if (page_index >= m_desc.page_textures.size()) break;
        draw_mesh(m_page_buffers[page_index],
                  m_desc.page_textures[page_index],
                  ResolveTextColor(*primitive, false));
    }

    rr.command.EndRenderPass();
    if (leftover_named && m_desc.vk_output.handle != VK_NULL_HANDLE) {
        // TEXT_E0_IDEST 0x1401e968a leftover named-RT store, then
        // POSTFX_OMSET HORIZONTAL samples leftover. Official D3D11
        // implicit hazard. Make COLOR_ATTACHMENT_WRITE visible as
        // SHADER_READ before HORIZONTAL (same view as leftover bind).
        VkImageSubresourceRange range {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        };
        VkImageMemoryBarrier barrier {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            .image = m_desc.vk_output.handle,
            .subresourceRange = range,
        };
        rr.command.PipelineBarrier(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                   VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                   VK_DEPENDENCY_BY_REGION_BIT,
                                   barrier);
    }

    if (m_desc.sample_count > VK_SAMPLE_COUNT_1_BIT &&
        m_desc.output == wallpaper::SpecTex_Default) {
        NoteComposeMsaaDraw(rr, m_desc.sample_count);
    }
}

void TextPass::destory(const Device&, RenderingResources& rr) {
    // Keep the cached text PSO alive through PipelineStateCache, but release every residency-bound
    // object that points at hidden-layer textures, render targets, or dynamic-buffer suballocations.
    m_desc.framebuffer.reset();
    m_desc.vk_output = {};
    m_desc.vk_resolve = {};
    m_desc.sample_count = VK_SAMPLE_COUNT_1_BIT;
    m_desc.resolve_msaa = false;
    m_desc.background_texture = {};
    m_desc.page_textures.clear();
    for (auto& subref : m_background_buffers.vertex_bufs) {
        rr.dyn_buf->unallocateSubRef(subref);
    }
    if (m_background_buffers.index_buf) rr.dyn_buf->unallocateSubRef(m_background_buffers.index_buf);
    m_background_buffers = {};
    for (auto& page_buffers : m_page_buffers) {
        for (auto& subref : page_buffers.vertex_bufs) {
            rr.dyn_buf->unallocateSubRef(subref);
        }
        if (page_buffers.index_buf) rr.dyn_buf->unallocateSubRef(page_buffers.index_buf);
    }
    m_page_buffers.clear();
    rr.dyn_buf->unallocateSubRef(m_desc.ubo_buf);
    m_desc.ubo_buf = {};
    if (m_clearalpha_ubo_buf) rr.dyn_buf->unallocateSubRef(m_clearalpha_ubo_buf);
    m_clearalpha_ubo_buf = {};
    for (auto& subref : m_clearalpha_buffers.vertex_bufs) {
        rr.dyn_buf->unallocateSubRef(subref);
    }
    if (m_clearalpha_buffers.index_buf) rr.dyn_buf->unallocateSubRef(m_clearalpha_buffers.index_buf);
    m_clearalpha_buffers = {};
    m_clearalpha_pipeline.resetCachedState();
    m_clearalpha_pipeline.handle.reset();
    m_clearalpha_pipeline.layout.reset();
    m_clearalpha_pipeline.pass.reset();
    m_clearalpha_fullfb = {};
    m_has_clearalpha = false;
    setPrepared(false);
}
