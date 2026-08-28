#pragma once

#include "VulkanPass.hpp"

#include <Eigen/Dense>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include "Vulkan/Device.hpp"
#include "Vulkan/GraphicsPipeline.hpp"
#include "Vulkan/StagingBuffer.hpp"

namespace wallpaper
{
class SceneTextPrimitive;
class SceneNode;
class SceneMesh;

namespace vulkan
{

class TextPass : public VulkanPass {
public:
    struct Desc {
        Scene*      scene { nullptr };
        SceneNode*  node { nullptr };
        // Authored layer ownership is separate from the SceneNode pointer because runtime text
        // rebuilds swap primitives under stable nodes; the layer id remains the durable refresh key.
        int32_t     layer_id { 0 };
        bool        execute_when_hidden { false };
        DestDrawPhase dest_draw_phase { DestDrawPhase::None };
        std::string output;
        AlphaWritePolicy alpha_write_policy { AlphaWritePolicy::Preserve };

        ImageParameters          vk_output;
        ImageParameters          vk_resolve;
        VkSampleCountFlagBits    sample_count { VK_SAMPLE_COUNT_1_BIT };
        bool                     resolve_msaa { false };
        vvk::Framebuffer         framebuffer;
        PipelineParameters       pipeline;
        StagingBufferRef         ubo_buf;
        ImageSlotsRef            background_texture;
        std::vector<ImageSlotsRef> page_textures;
        VkClearValue             clear_value {};
        bool                     clear_output { false };
    };

    TextPass(const Desc&);
    ~TextPass() override;

    void prepare(Scene&, const Device&, RenderingResources&) override;
    void refreshResources(Scene&, const Device&, RenderingResources&) override;
    void dropOutputFramebuffers() override;
    void execute(const Device&, RenderingResources&) override;
    void destory(const Device&, RenderingResources&) override;
    bool warmupPipeline(Scene&, const Device&, RenderingResources&) override;
    std::string residencyKey() const override;
    bool canReuseForResidency(const VulkanPass& next_pass) const override;
    void absorbResidencyGraphState(const VulkanPass&) override;
    bool referencesRenderTarget(std::string_view) const override;
    bool referencesTextLayer(int32_t) const override;
    DestDrawPhase destDrawPhase() const override { return m_desc.dest_draw_phase; }
    int32_t destDrawLayerId() const override { return m_desc.layer_id; }
    wallpaper::SceneNode* destDrawNode() const override { return m_desc.node; }
    void writeLastPassMvp(const Eigen::Matrix4f& mvp) override;

private:
    struct MeshBuffers {
        std::vector<StagingBufferRef> vertex_bufs;
        StagingBufferRef              index_buf;
        uint32_t                      draw_count { 0 };
        bool                          force_upload { true };
    };

    // The direct text pass owns its own dynamic mesh uploads because text geometry can change
    // without changing render-graph topology. Keeping the buffers inside the pass lets a single
    // pass instance absorb atlas page count and quad changes in place.
    bool ensureMeshBuffers(SceneMesh&, MeshBuffers&, RenderingResources&);
    bool refreshTextures(const Device&);
    bool recreateFramebuffer(const Device&);

    Desc m_desc;
    MeshBuffers m_background_buffers;
    std::vector<MeshBuffers> m_page_buffers;
    MeshBuffers m_clearalpha_buffers;
    PipelineParameters m_clearalpha_pipeline;
    ImageSlotsRef m_clearalpha_fullfb;
    StagingBufferRef m_clearalpha_ubo_buf;
    uint32_t m_loaded_atlas_version { std::numeric_limits<uint32_t>::max() };
    Eigen::Matrix4f m_dest_ortho_mvp { Eigen::Matrix4f::Identity() };
    bool m_has_dest_ortho_mvp { false };
    bool m_has_clearalpha { false };
};

} // namespace vulkan
} // namespace wallpaper
