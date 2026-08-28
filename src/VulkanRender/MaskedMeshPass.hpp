#pragma once

#include "MaskedDrawRenderer.hpp"
#include "ShaderDrawCore.hpp"
#include "VulkanPass.hpp"

namespace wallpaper::vulkan
{

class MaskedMeshPass final : public VulkanPass {
public:
    using Desc = ShaderDrawRequest;

    explicit MaskedMeshPass(const Desc&);
    ~MaskedMeshPass() override = default;

    void setDescTex(u32 index, std::string_view texture_key);
    DestDrawPhase destDrawPhase() const override;
    int32_t destDrawLayerId() const override;
    void writeLastPassMvp(const Eigen::Matrix4f&) override;

    void prepare(Scene&, const Device&, RenderingResources&) override;
    void prepareDeferred(Scene&, const Device&, RenderingResources&) override;
    void refreshResources(Scene&, const Device&, RenderingResources&) override;
    void refreshImportedTextureBindings(Scene&, const Device&) override;
    void dropOutputFramebuffers() override;
    void updateBeforeUpload() override;
    DeferredPrepareResourcesState requestDeferredPrepareResources(Scene&, const Device&) override;
    void execute(const Device&, RenderingResources&) override;
    void destory(const Device&, RenderingResources&) override;
    bool warmupPipeline(Scene&, const Device&, RenderingResources&) override;
    std::string residencyKey() const override;
    bool canReuseForResidency(const VulkanPass&) const override;
    void absorbResidencyGraphState(const VulkanPass&) override;
    bool referencesRenderTarget(std::string_view) const override;
    bool referencesImportedTexture(std::string_view) const override;

private:
    ShaderDrawCore     m_visible_draw;
    MaskedDrawRenderer m_masked_draw;
};

} // namespace wallpaper::vulkan
