#include "MaskedMeshPass.hpp"

#include "Scene/SceneShader.h"
#include "SpecTexs.hpp"

namespace wallpaper::vulkan
{

MaskedMeshPass::MaskedMeshPass(const Desc& desc)
    : m_visible_draw(desc) {
    m_visible_draw.setExtension(&m_masked_draw);
}

void MaskedMeshPass::prepare(Scene& scene, const Device& device, RenderingResources& resources) {
    setPrepared(m_visible_draw.prepare(scene, device, resources));
}

void MaskedMeshPass::prepareDeferred(Scene& scene, const Device& device,
                                     RenderingResources& resources) {
    setPrepared(m_visible_draw.prepareDeferred(scene, device, resources));
}

void MaskedMeshPass::refreshResources(Scene& scene, const Device& device,
                                      RenderingResources& resources) {
    if (! m_visible_draw.refreshResources(scene, device, resources)) setPrepared(false);
}

void MaskedMeshPass::refreshImportedTextureBindings(Scene& scene, const Device& device) {
    if (!m_visible_draw.refreshImportedTextureBindings(scene, device)) setPrepared(false);
}

void MaskedMeshPass::dropOutputFramebuffers() { m_visible_draw.dropOutputFramebuffers(); }

void MaskedMeshPass::updateBeforeUpload() { m_visible_draw.updateBeforeUpload(); }

DeferredPrepareResourcesState
MaskedMeshPass::requestDeferredPrepareResources(Scene& scene, const Device& device) {
    return m_visible_draw.requestDeferredPrepareResources(scene, device);
}

void MaskedMeshPass::execute(const Device& device, RenderingResources& resources) {
    m_visible_draw.execute(device, resources);
    releaseFinalReadTexs(device);
}

void MaskedMeshPass::destory(const Device&, RenderingResources& resources) {
    m_visible_draw.destroy(resources);
    setPrepared(false);
}

bool MaskedMeshPass::warmupPipeline(Scene& scene, const Device& device,
                                    RenderingResources& resources) {
    return m_visible_draw.warmupPipeline(scene, device, resources);
}

std::string MaskedMeshPass::residencyKey() const {
    return m_visible_draw.residencyKey("MaskedMeshPass");
}

bool MaskedMeshPass::canReuseForResidency(const VulkanPass& next_pass) const {
    const auto* next = dynamic_cast<const MaskedMeshPass*>(&next_pass);
    if (next == nullptr || ! m_visible_draw.canReuseForResidency(next->m_visible_draw)) {
        return false;
    }
    const auto* lhs_node = m_visible_draw.data().node;
    const auto* rhs_node = next->m_visible_draw.data().node;
    if (lhs_node == nullptr || rhs_node == nullptr || lhs_node->Mesh() == nullptr ||
        rhs_node->Mesh() == nullptr) {
        return lhs_node == rhs_node;
    }
    return MaskedDrawRenderer::SamePlan(lhs_node->Mesh()->MaskedDraw(),
                                        rhs_node->Mesh()->MaskedDraw());
}

void MaskedMeshPass::absorbResidencyGraphState(const VulkanPass& next_pass) {
    const auto* next = dynamic_cast<const MaskedMeshPass*>(&next_pass);
    if (next == nullptr) return;
    m_visible_draw.absorbResidencyGraphState(next->m_visible_draw);
}

bool MaskedMeshPass::referencesRenderTarget(std::string_view render_target) const {
    return m_visible_draw.referencesRenderTarget(render_target);
}

bool MaskedMeshPass::referencesImportedTexture(std::string_view texture_key) const {
    return m_visible_draw.referencesImportedTexture(texture_key);
}

void MaskedMeshPass::setDescTex(u32 index, std::string_view texture_key) {
    m_visible_draw.setTexture(index, texture_key);
}

wallpaper::DestDrawPhase MaskedMeshPass::destDrawPhase() const {
    return m_visible_draw.data().dest_draw_phase;
}

int32_t MaskedMeshPass::destDrawLayerId() const { return m_visible_draw.data().layer_id; }

void MaskedMeshPass::writeLastPassMvp(const Eigen::Matrix4f& mvp) {
    m_visible_draw.WriteUniform(G_MVP, ShaderValue::fromMatrix(mvp));
}

} // namespace wallpaper::vulkan
