#include "CustomShaderPass.hpp"

#include "Scene/Scene.h"
#include "Scene/SceneNode.h"
#include "Scene/SceneShader.h"
#include "SpecTexs.hpp"
#include "Type.hpp"

using namespace wallpaper::vulkan;

namespace
{
const char* BlendModeName(wallpaper::BlendMode mode) {
    switch (mode) {
    case wallpaper::BlendMode::Disable:     return "disable";
    case wallpaper::BlendMode::Translucent: return "translucent";
    case wallpaper::BlendMode::Additive:    return "additive";
    case wallpaper::BlendMode::Normal:           return "normal";
    case wallpaper::BlendMode::AlphaToCoverage:  return "alphatocoverage";
    }
    return "unknown";
}
} // namespace

CustomShaderPass::CustomShaderPass(const Desc& desc)
    : m_core(desc) {}

CustomShaderPass::~CustomShaderPass() = default;

void CustomShaderPass::prepare(Scene& scene, const Device& device, RenderingResources& resources) {
    setPrepared(m_core.prepare(scene, device, resources));
}

void CustomShaderPass::prepareDeferred(Scene& scene, const Device& device,
                                       RenderingResources& resources) {
    setPrepared(m_core.prepareDeferred(scene, device, resources));
}

void CustomShaderPass::refreshResources(Scene& scene, const Device& device,
                                        RenderingResources& resources) {
    if (! m_core.refreshResources(scene, device, resources)) setPrepared(false);
}

void CustomShaderPass::refreshImportedTextureBindings(Scene& scene, const Device& device) {
    if (!m_core.refreshImportedTextureBindings(scene, device)) setPrepared(false);
}

void CustomShaderPass::dropOutputFramebuffers() { m_core.dropOutputFramebuffers(); }

void CustomShaderPass::updateBeforeUpload() { m_core.updateBeforeUpload(); }

DeferredPrepareResourcesState
CustomShaderPass::requestDeferredPrepareResources(Scene& scene, const Device& device) {
    return m_core.requestDeferredPrepareResources(scene, device);
}

void CustomShaderPass::execute(const Device& device, RenderingResources& resources) {
    m_core.execute(device, resources);
    releaseFinalReadTexs(device);
}

void CustomShaderPass::destory(const Device&, RenderingResources& resources) {
    m_core.destroy(resources);
    setPrepared(false);
}

bool CustomShaderPass::warmupPipeline(Scene& scene, const Device& device,
                                      RenderingResources& resources) {
    return m_core.warmupPipeline(scene, device, resources);
}

std::string CustomShaderPass::residencyKey() const {
    return m_core.residencyKey("CustomShaderPass");
}

bool CustomShaderPass::canReuseForResidency(const VulkanPass& next_pass) const {
    const auto* next = dynamic_cast<const CustomShaderPass*>(&next_pass);
    return next != nullptr && m_core.canReuseForResidency(next->m_core);
}

void CustomShaderPass::absorbResidencyGraphState(const VulkanPass& next_pass) {
    const auto* next = dynamic_cast<const CustomShaderPass*>(&next_pass);
    if (next != nullptr) m_core.absorbResidencyGraphState(next->m_core);
}

bool CustomShaderPass::referencesRenderTarget(std::string_view target) const {
    return m_core.referencesRenderTarget(target);
}

bool CustomShaderPass::referencesImportedTexture(std::string_view texture_key) const {
    return m_core.referencesImportedTexture(texture_key);
}

void CustomShaderPass::setDescTex(u32 index, std::string_view tex_key) {
    m_core.setTexture(index, tex_key);
}

wallpaper::DestDrawPhase CustomShaderPass::destDrawPhase() const {
    return m_core.data().dest_draw_phase;
}

int32_t CustomShaderPass::destDrawLayerId() const { return m_core.data().layer_id; }

wallpaper::SceneNode* CustomShaderPass::destDrawNode() const { return m_core.data().node; }

void CustomShaderPass::writeLastPassMvp(const Eigen::Matrix4f& mvp) {
    m_core.WriteUniform(G_MVP, ShaderValue::fromMatrix(mvp));
}

void CustomShaderPass::writeLastPassInverseSlot(const Eigen::Matrix4f& mvp) {
    // IMAGE_VT_F0 / UNIFORM_UPLOAD_MAP 0x1400d8749: id 0xd copies +0x8f0.
    m_core.WriteUniform(G_MVPI, ShaderValue::fromMatrix(mvp));
}

bool CustomShaderPass::hasUniform(std::string_view name) const {
    return m_core.HasUniform(name);
}

bool CustomShaderPass::uboReady() const { return m_core.UboReady(); }
