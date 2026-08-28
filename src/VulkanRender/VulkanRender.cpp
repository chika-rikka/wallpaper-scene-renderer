#include "VulkanRender.hpp"
#include <typeinfo>

#include "Utils/Logging.h"
#include "RenderGraph/RenderGraph.hpp"
#include "Scene/Scene.h"
#include "Scene/SceneMesh.h"
#include "Scene/SceneNode.h"
#include "SpecTexs.hpp"
#include "Interface/IImageParser.h"
#include "Interface/IShaderValueUpdater.h"

#include "Utils/Algorism.h"

#include "Vulkan/Device.hpp"
#include "Vulkan/TextureCache.hpp"
#include "Vulkan/Swapchain.hpp"
#include "Vulkan/VideoTextureCache.hpp"
#include "Vulkan/VulkanExSwapchain.hpp"

#include "Msaa.hpp"
#include "PassCommon.hpp"
#include "VulkanPass.hpp"
#include "PrePass.hpp"
#include "FinPass.hpp"
#include "CopyPass.hpp"
#include "Resource.hpp"
#include "Vulkan/Util.hpp"

#include "Core/ArrayHelper.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if ENABLE_RENDERDOC_API
#    include "RenderDoc.h"
#endif

using namespace wallpaper::vulkan;
using wallpaper::DestDrawGfx;
using wallpaper::DestDrawPhase;

constexpr uint64_t vk_wait_time { 10u * 1000u * 1000000u };
constexpr uint32_t vk_command_num { 1 };
constexpr std::size_t kDeferredPrepareMaxPassesPerFrame { 96 };
constexpr double      kDeferredPrepareFrameBudgetMs { 2.0 };

constexpr std::array base_inst_exts {
    Extension { false, VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME },
};
constexpr std::array base_device_exts {
    Extension { false, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME },
    Extension { false, VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME },
    Extension { false, VK_EXT_IMAGE_DRM_FORMAT_MODIFIER_EXTENSION_NAME },
    Extension { true, VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME },
    Extension { true, VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME },
    Extension { true, VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME },
    Extension { true, VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME },
    Extension { true, VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME }
};

namespace
{

std::mutex& VulkanInitMutex() {
    static std::mutex mutex;
    return mutex;
}

void RunComposeDrawWalker(wallpaper::Scene& scene, wallpaper::DestDrawGfx* gfx) {
    // FRAME_DEST_NO_RESET 0x14018aac0 / PATH_B 0x14018b170: dest-draw after
    // GPU record start, leftover then last-pass before dest pop
    // (LEFTOVER_VS_DESTDRAW).
    scene.BindDestDrawGfx(gfx);
    if (scene.shaderValueUpdater != nullptr)
        scene.shaderValueUpdater->ComposeDrawWalker();
    scene.BindDestDrawGfx(nullptr);
}

std::string MakeResidencyInstanceKey(
    const VulkanPass& pass, std::unordered_map<std::string, std::size_t>& occurrence_counts) {
    const auto base_key = pass.residencyKey();
    if (base_key.empty()) return {};

    const auto instance = occurrence_counts[base_key]++;
    return base_key + "|instance=" + std::to_string(instance);
}

void DestroyPassOnce(VulkanPass* pass, const Device& device, RenderingResources& resources,
                     std::unordered_set<VulkanPass*>& destroyed) {
    if (pass == nullptr || !destroyed.insert(pass).second) return;
    pass->destory(device, resources);
}

void WalkSceneMeshes(wallpaper::SceneNode* node,
                     const std::function<void(wallpaper::SceneMesh&)>& fn) {
    if (node == nullptr) return;
    if (auto* mesh = node->Mesh(); mesh != nullptr) fn(*mesh);
    for (auto& child : node->GetChildren()) {
        WalkSceneMeshes(child.get(), fn);
    }
}

void ReleaseUploadedFileMeshCpu(wallpaper::Scene& scene) {
    const auto release = [&](wallpaper::SceneMesh& mesh) {
        if (mesh.FileImmutable() && mesh.HasCpuPayload()) mesh.ReleaseCpuPayload();
    };
    WalkSceneMeshes(scene.sceneGraph.get(), release);
    for (auto& [_, nodes] : scene.objectRuntimeNodes) {
        for (auto* node : nodes) WalkSceneMeshes(node, release);
    }
}

const char* ExternalMemoryPreferenceName(wallpaper::ExternalFrameMemoryPreference preference) {
    switch (preference) {
    case wallpaper::ExternalFrameMemoryPreference::HostVisible: return "host-visible";
    case wallpaper::ExternalFrameMemoryPreference::DeviceLocal: return "device-local";
    case wallpaper::ExternalFrameMemoryPreference::Default:
    default: return "default";
    }
}

} // namespace

struct VulkanRender::Impl : wallpaper::DestDrawGfx {
    Impl()  = default;
    ~Impl() = default;

    void Record(wallpaper::SceneObject& object) override;
    void ClearDestDrawPasses();
    void IndexDestDrawPass(VulkanPass* pass);
    void PrepareDestDrawPasses(Scene&);
    void RefreshDestDrawPasses(Scene&, bool refresh_all,
                               const std::unordered_set<std::string>& dirty_render_targets,
                               const std::unordered_set<std::string>& dirty_imported_textures,
                               const std::unordered_set<int32_t>& dirty_text_layers);
    void DestroyDestDrawPasses();

    bool init(RenderInitInfo);
    void destroy();

    void drawFrame(Scene&);
    void setPaused(bool paused);
    void setOffscreenFrameReleaseCallback(OffscreenFrameReleaseCallback callback);
    bool reconfigureOffscreenExport(uint32_t width,
                                    uint32_t height,
                                    TexTiling tiling,
                                    ExternalFrameExportMode export_mode,
                                    uint32_t export_drm_fourcc,
                                    const std::vector<uint64_t>& export_drm_modifiers,
                                    ExternalFrameMemoryPreference memory_preference);

    bool CreateRenderingResource(RenderingResources&);
    void DestroyRenderingResource(RenderingResources&);

    void clearLastRenderGraph(bool clear_scene_caches);
    void clearRenderGraphResources();
    void releasePendingSceneResources(Scene&);
    void compileRenderGraph(Scene&, rg::RenderGraph&, bool refresh_resources_only);
    void warmupRenderGraphPipelines(Scene&, rg::RenderGraph&);
    void refreshImportedTextures(Scene&);
    void UpdateCameraFillMode(Scene&, wallpaper::FillMode);

    bool initRes();
    void drawFrameSwapchain();
    void drawFrameOffscreen(Scene&);
    int  m_compiled_msaa_samples { -1 };
    void processDeferredGraphPreparation(Scene&);
    void dropCompiledPassFramebuffers();
    void setRenderTargetSize(Scene&, rg::RenderGraph&);
    bool isDeviceFaultResult(VkResult) const;
    bool checkVkResult(VkResult, const char* operation);
    void abandonDeviceOwnedResourcesAfterFault();

    Instance                m_instance;
    std::unique_ptr<Device> m_device;

    std::unique_ptr<PrePass> m_prepass { nullptr };
    std::unique_ptr<FinPass> m_finpass { nullptr };

    std::unique_ptr<FinPass> m_testpass { nullptr };
    ReDrawCB                 m_redraw_cb;

    std::unique_ptr<StagingBuffer> m_vertex_buf { nullptr };
    std::unique_ptr<StagingBuffer> m_dyn_buf { nullptr };

    vvk::CommandBuffers m_cmds;
    vvk::CommandBuffer  m_render_cmd;

    bool m_with_surface { false };
    bool m_inited { false };
    bool m_pass_loaded { false };
    bool m_device_faulted { false };
    bool m_device_fault_log_emitted { false };
    std::deque<std::size_t> m_deferred_prepare_indices;
    std::unordered_set<std::size_t> m_deferred_waiting_indices_logged;

    std::unique_ptr<VulkanExSwapchain> m_ex_swapchain;
    RenderingResources                 m_rendering_resources;
    OffscreenFrameReleaseCallback      m_offscreen_frame_release_cb;

    std::vector<VulkanPass*> m_passes;
    std::vector<std::shared_ptr<rg::Pass>> m_compiled_pass_refs;
    std::unordered_map<int32_t, std::vector<VulkanPass*>> m_dest_leftover;
    std::unordered_map<int32_t, VulkanPass*> m_dest_leftover_mvp;
    std::unordered_map<int32_t, std::vector<VulkanPass*>> m_dest_postfx;
    std::unordered_map<int32_t, VulkanPass*> m_dest_lastpass;
    std::vector<VulkanPass*> m_dest_draw_all;
    std::unordered_set<int32_t> m_dest_draw_contracts_logged;

};

VulkanRender::VulkanRender(): pImpl(std::make_unique<Impl>()) {}
VulkanRender::~VulkanRender() { pImpl->destroy(); };

bool VulkanRender::inited() const { return pImpl->m_inited; }

bool VulkanRender::init(RenderInitInfo info) { return pImpl->init(info); }
void VulkanRender::destroy() { pImpl->destroy(); }
void VulkanRender::drawFrame(Scene& scene) { pImpl->drawFrame(scene); };
void VulkanRender::setPaused(bool paused) { pImpl->setPaused(paused); };
void VulkanRender::setOffscreenFrameReleaseCallback(OffscreenFrameReleaseCallback callback) {
    pImpl->setOffscreenFrameReleaseCallback(std::move(callback));
};
bool VulkanRender::reconfigureOffscreenExport(
    uint32_t width,
    uint32_t height,
    TexTiling tiling,
    ExternalFrameExportMode export_mode,
    uint32_t export_drm_fourcc,
    const std::vector<uint64_t>& export_drm_modifiers,
    ExternalFrameMemoryPreference memory_preference) {
    return pImpl->reconfigureOffscreenExport(width,
                                             height,
                                             tiling,
                                             export_mode,
                                             export_drm_fourcc,
                                             export_drm_modifiers,
                                             memory_preference);
};
void VulkanRender::clearLastRenderGraph(bool clear_scene_caches) {
    pImpl->clearLastRenderGraph(clear_scene_caches);
};
void VulkanRender::clearRenderGraphResources() { pImpl->clearRenderGraphResources(); };
void VulkanRender::compileRenderGraph(Scene& scene, rg::RenderGraph& rg, bool refresh_resources_only) {
    pImpl->compileRenderGraph(scene, rg, refresh_resources_only);
};
void VulkanRender::warmupRenderGraphPipelines(Scene& scene, rg::RenderGraph& rg) {
    pImpl->warmupRenderGraphPipelines(scene, rg);
};
void VulkanRender::refreshImportedTextures(Scene& scene) {
    pImpl->refreshImportedTextures(scene);
};
void VulkanRender::UpdateCameraFillMode(Scene& scene, wallpaper::FillMode fill) {
    pImpl->UpdateCameraFillMode(scene, fill);
};

wallpaper::ExSwapchain* VulkanRender::exSwapchain() const { return pImpl->m_ex_swapchain.get(); };

bool VulkanRender::Impl::init(RenderInitInfo info) {
    if (m_inited) return true;

    /*
     * Independent-display mode can start several scene renderers from the same
     * process. NVIDIA's Vulkan loader/ICD path has shown crashes when multiple
     * threads create instances and devices at the exact same time, so serialize
     * the one-time Vulkan bootstrap for this backend. The lock is deliberately
     * held only during init(); steady-state rendering remains fully parallel.
     */
    std::lock_guard<std::mutex> vulkan_init_lock(VulkanInitMutex());
    if (m_inited) return true;

    m_redraw_cb = info.redraw_callback;
    VkExtent2D extent { info.width, info.height };
    if (extent.width * extent.height < 500 * 500) {
        LOG_ERROR("too small swapchain image size: %dx%d", extent.width, extent.height);
    } else {
        LOG_INFO("set swapchain image size: %dx%d", extent.width, extent.height);
    }

    std::vector<Extension> inst_exts { base_inst_exts.begin(), base_inst_exts.end() };
    std::vector<Extension> device_exts { base_device_exts.begin(), base_device_exts.end() };

    if (! info.offscreen) {
        std::transform(info.surface_info.instanceExts.begin(),
                       info.surface_info.instanceExts.end(),
                       std::back_inserter(inst_exts),
                       [](const auto& s) {
                           return Extension { true, s.c_str() };
                       });
        device_exts.push_back({ true, VK_KHR_SWAPCHAIN_EXTENSION_NAME });
    }

    std::vector<InstanceLayer> inst_layers;
    // valid layer
    if (info.enable_valid_layer) {
        inst_layers.push_back({ true, VALIDATION_LAYER_NAME });
        LOG_INFO("vulkan valid layer \"%s\" enabled", VALIDATION_LAYER_NAME.data());
    }

    if (! Instance::Create(m_instance, inst_exts, inst_layers)) {
        LOG_ERROR("init vulkan failed");
        return false;
    }
    if (! info.offscreen) {
        VkSurfaceKHR surface;
        VVK_CHECK_ACT(
            {
                LOG_ERROR("create vulkan surface failed");
                return false;
            },
            info.surface_info.createSurfaceOp(*m_instance.inst(), &surface));
        m_instance.setSurface(VkSurfaceKHR(surface));
        m_with_surface = true;
    }
    {
        auto surface   = *m_instance.surface();
        auto check_gpu = [&device_exts, surface](const vvk::PhysicalDevice& gpu) {
            return Device::CheckGPU(gpu, device_exts, surface);
        };
        const auto preference =
            info.device_preference == wallpaper::VulkanDevicePreference::PreferIntegrated
                ? PhysicalDevicePreference::PreferIntegrated
                : (info.device_preference == wallpaper::VulkanDevicePreference::PreferDiscrete
                       ? PhysicalDevicePreference::PreferDiscrete
                       : PhysicalDevicePreference::Default);
        if (! m_instance.ChoosePhysicalDevice(check_gpu, info.uuid, preference)) return false;
    }

    {
        m_device = std::make_unique<Device>();
        const VideoTextureDecoderSettings video_texture_settings {
            .decoder_route =
                info.video_texture_decoder_route == wallpaper::VideoTextureDecoderRoute::Va
                    ? VideoTextureDecoderRoute::Va
                    : VideoTextureDecoderRoute::Nvidia,
            .render_node = info.render_node,
        };
        if (! Device::Create(m_instance, device_exts, extent, *m_device, video_texture_settings)) {
            LOG_ERROR("init vulkan device failed");
            return false;
        }
    }

    if (info.offscreen) {
        if (info.ex_swapchain_factory) {
            const RenderInitInfo::ExSwapchainHandles handles {
                .instance = *m_instance.inst(),
                .physical_device = *m_instance.gpu(),
                .device = *m_device->handle(),
                .graphics_queue = *m_device->graphics_queue().handle,
                .graphics_queue_family = m_device->graphics_queue().family_index,
                .renderer_device = m_device.get(),
            };
            m_ex_swapchain = info.ex_swapchain_factory(handles);
            if (!m_ex_swapchain) {
                LOG_ERROR("external offscreen swapchain factory returned null");
                return false;
            }
        } else {
            m_ex_swapchain = CreateExSwapchain(*m_device,
                                               extent.width,
                                               extent.height,
                                               (info.offscreen_tiling == TexTiling::OPTIMAL
                                                    ? VK_IMAGE_TILING_OPTIMAL
                                                    : VK_IMAGE_TILING_LINEAR),
                                               info.export_mode,
                                               info.export_drm_fourcc,
                                               info.export_drm_modifiers,
                                               info.export_memory_preference);
        }
        m_with_surface = false;
    }

    if (! initRes()) return false;
    ;

    m_inited = true;
    return m_inited;
}

bool VulkanRender::Impl::isDeviceFaultResult(VkResult result) const {
    return result == VK_ERROR_DEVICE_LOST || result == VK_TIMEOUT;
}

bool VulkanRender::Impl::checkVkResult(VkResult result, const char* operation) {
    if (result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR) return true;

    const char* operation_name = operation ? operation : "unknown operation";
    if (isDeviceFaultResult(result)) {
        m_device_faulted = true;
        m_pass_loaded = false;
        if (!m_device_fault_log_emitted) {
            // Device loss is terminal for this VulkanRender instance.  Continuing to submit frames
            // only repeats VK_ERROR_DEVICE_LOST, and destroying every pipeline after an NVIDIA Xid
            // can enter driver teardown paths that have appeared in the Arsenal crash stacks.
            LOG_ERROR("HanabiScene Vulkan: device became unhealthy during %s (%s); "
                      "suppressing future frame submissions and abandoning deep Vulkan teardown",
                      operation_name,
                      vvk::ToString(result));
            m_device_fault_log_emitted = true;
        }
        return false;
    }

    LOG_ERROR("HanabiScene Vulkan: %s failed with %s", operation_name, vvk::ToString(result));
    return false;
}

bool VulkanRender::Impl::initRes() {
    m_prepass = std::make_unique<PrePass>(PrePass::Desc {});
    m_finpass = std::make_unique<FinPass>(FinPass::Desc {});
    if (m_with_surface) {
        m_finpass->setPresentFormat(m_device->swapchain().format());
        m_finpass->setPresentQueueIndex(m_device->present_queue().family_index);
        m_finpass->setPresentLayout(VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
    } else {
        m_finpass->setPresentFormat(m_ex_swapchain->format());
        m_finpass->setPresentLayout(VK_IMAGE_LAYOUT_GENERAL);
        m_finpass->setPresentQueueIndex(VK_QUEUE_FAMILY_EXTERNAL);
    }
    /*
    m_testpass = std::make_unique<FinPass>(FinPass::Desc{});
    m_testpass->setPresentFormat(m_ex_swapchain->format());
    m_testpass->setPresentQueueIndex(m_device->graphics_queue().family_index);
    m_testpass->setPresentLayout(vk::ImageLayout::ePresentSrcKHR);
    */

    m_vertex_buf = std::make_unique<StagingBuffer>(*m_device,
                                                   2 * 1024 * 1024,
                                                   VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                                                       VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
    m_dyn_buf    = std::make_unique<StagingBuffer>(*m_device,
                                                2 * 1024 * 1024,
                                                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                                                    VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                                    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    if (! m_vertex_buf->allocate()) return false;
    if (! m_dyn_buf->allocate()) return false;
    {
        auto& pool = m_device->cmd_pool();
        VVK_CHECK_BOOL_RE(pool.Allocate(vk_command_num, VK_COMMAND_BUFFER_LEVEL_PRIMARY, m_cmds));
        m_render_cmd = vvk::CommandBuffer(m_cmds[0], m_device->handle().Dispatch());
    }
    if (! CreateRenderingResource(m_rendering_resources)) return false;

#if ENABLE_RENDERDOC_API
    load_renderdoc_api();
#endif
    return true;
}

void VulkanRender::Impl::abandonDeviceOwnedResourcesAfterFault() {
    // Once the GPU has reported timeout/device-lost, Vulkan object destruction is no longer a safe
    // cleanup mechanism on the observed NVIDIA path.  This routine intentionally abandons wrapper
    // ownership so process shutdown or backend replacement does not call back into driver destroy
    // functions with a broken device.  The leaked objects are bounded to this renderer instance and
    // are preferable to a SIGSEGV while switching away from the failed scene.
    m_rendering_resources.sem_swap_wait_image.abandon();
    m_rendering_resources.sem_swap_finish.abandon();
    m_rendering_resources.fence_frame.abandon();
    m_rendering_resources.command.abandon();
    for (auto& [_, image] : m_rendering_resources.model_depth_images) {
        image.sampler.abandon();
        image.view.abandon();
        image.handle.abandon();
    }
    m_rendering_resources.model_depth_images.clear();
    m_rendering_resources.model_depth_resolved.clear();
    m_rendering_resources.masked_draw_attachments.abandon();
    if (m_rendering_resources.pipeline_cache) {
        m_rendering_resources.pipeline_cache->abandon();
    }
    if (m_device) {
        m_device->tex_cache().CancelDeferredGraphActivation();
    }
    m_rendering_resources.immutable_meshes.abandon();
    m_rendering_resources.vertex_buf = nullptr;
    m_rendering_resources.dyn_buf = nullptr;
    m_deferred_prepare_indices.clear();
    m_deferred_waiting_indices_logged.clear();

    m_render_cmd.abandon();
    m_cmds.abandon();
    m_compiled_pass_refs.clear();
    m_passes.clear();
    ClearDestDrawPasses();
    (void)m_prepass.release();
    (void)m_finpass.release();
    (void)m_testpass.release();
    (void)m_vertex_buf.release();
    (void)m_dyn_buf.release();
    (void)m_ex_swapchain.release();
    (void)m_device.release();
    m_instance.Abandon();
    m_inited = false;
    m_pass_loaded = false;
}

void VulkanRender::Impl::destroy() {
    if (! m_inited && !m_device_faulted) return;
    if (m_device_faulted) {
        abandonDeviceOwnedResourcesAfterFault();
        return;
    }
    if (m_device && m_device->handle()) {
        if (!checkVkResult(m_device->handle().WaitIdle(), "device wait idle before destroy")) {
            abandonDeviceOwnedResourcesAfterFault();
            return;
        }

        // res
        DestroyDestDrawPasses();
        for (auto& p : m_passes) {
            p->destory(*m_device, m_rendering_resources);
        }
        m_compiled_pass_refs.clear();
        m_passes.clear();
        m_deferred_prepare_indices.clear();
        m_deferred_waiting_indices_logged.clear();
        m_device->tex_cache().CancelDeferredGraphActivation();
        if (m_rendering_resources.pipeline_cache) {
            m_rendering_resources.pipeline_cache->clear();
        }
        m_rendering_resources.pipeline_cache.reset();
        m_rendering_resources.model_depth_images.clear();
        m_rendering_resources.model_depth_resolved.clear();
        m_rendering_resources.masked_draw_attachments.clear();
        m_rendering_resources.immutable_meshes.clear();
        m_vertex_buf->destroy();
        m_dyn_buf->destroy();

        m_rendering_resources.command.reset();
        m_render_cmd.reset();
        m_cmds.reset();

        m_device->Destroy();
    }
    m_rendering_resources.sem_swap_wait_image.reset();
    m_rendering_resources.sem_swap_finish.reset();
    m_rendering_resources.fence_frame.reset();
    m_rendering_resources.vertex_buf = nullptr;
    m_rendering_resources.dyn_buf = nullptr;
    m_prepass.reset();
    m_finpass.reset();
    m_testpass.reset();
    m_vertex_buf.reset();
    m_dyn_buf.reset();
    m_ex_swapchain.reset();
    m_device.reset();
    m_instance.Destroy();
    m_with_surface = false;
    m_inited = false;
    m_pass_loaded = false;
}

bool VulkanRender::Impl::CreateRenderingResource(RenderingResources& rr) {
    rr.command = m_render_cmd;
    VVK_CHECK_BOOL_RE(m_device->handle().CreateFence(
        VkFenceCreateInfo {
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .pNext = nullptr,
            .flags = VK_FENCE_CREATE_SIGNALED_BIT,
        },
        rr.fence_frame));

    rr.fence_frame.Reset();

    if (m_with_surface) {
        VkSemaphoreCreateInfo ci { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
                                   .pNext = nullptr };
        VVK_CHECK_BOOL_RE(m_device->handle().CreateSemaphore(ci, rr.sem_swap_finish));
        VVK_CHECK_BOOL_RE(m_device->handle().CreateSemaphore(ci, rr.sem_swap_wait_image));
    }

    rr.vertex_buf = m_vertex_buf.get();
    rr.dyn_buf    = m_dyn_buf.get();
    rr.pipeline_cache = std::make_shared<GraphicsPipelineStateCache>();
    return true;
}

void VulkanRender::Impl::DestroyRenderingResource(RenderingResources& rr) {}

// VulkanExSwapchain* VulkanRender::exSwapchain() const { return m_ex_swapchain.get(); }

void VulkanRender::Impl::drawFrame(Scene& scene) {
    if (m_device_faulted) return;
    if (! (m_inited && m_pass_loaded)) return;
    // The QuickJS host records getVideoTexture().play()/pause() decisions on Scene before the
    // renderer polls GStreamer. Applying them here keeps hidden authored videos from decoding
    // while prepared passes can still reuse the last uploaded frame when they are invisible.
    m_device->video_tex_cache().ApplyPlaybackStates(scene.videoTexturePaused,
                                                    scene.videoTextureStopped,
                                                    scene.videoTextureRates);
    // setCurrentTime() requests are one-shot decoder commands, so the video cache consumes and
    // removes only the requests whose concrete GStreamer pipeline already exists.
    m_device->video_tex_cache().ApplySeekRequests(scene.videoTextureSeekRequests);
    m_device->video_tex_cache().Poll();
    m_device->video_tex_cache().PublishRuntimeStates(
        scene.videoTextureRuntimeStates, scene.videoTextureRuntimeStateRequests);
    processDeferredGraphPreparation(scene);
    m_rendering_resources.scene = &scene;

#if ENABLE_RENDERDOC_API
    if (rdoc_api)
        rdoc_api->StartFrameCapture(
            RENDERDOC_DEVICEPOINTER_FROM_VKINSTANCE((VkInstance)m_instance.inst()), NULL);
#endif

    if (m_instance.offscreen()) {
        drawFrameOffscreen(scene);
    } else {
        drawFrameSwapchain();
    }

    if (m_redraw_cb) m_redraw_cb();

#if ENABLE_RENDERDOC_API
    if (rdoc_api)
        rdoc_api->EndFrameCapture(
            RENDERDOC_DEVICEPOINTER_FROM_VKINSTANCE((VkInstance)m_instance.inst()), NULL);
#endif
}

void VulkanRender::Impl::setPaused(bool paused) {
    if (m_device == nullptr) return;
    m_device->video_tex_cache().SetGlobalPaused(paused);
}

void VulkanRender::Impl::setOffscreenFrameReleaseCallback(
    OffscreenFrameReleaseCallback callback) {
    m_offscreen_frame_release_cb = std::move(callback);
}

bool VulkanRender::Impl::reconfigureOffscreenExport(
    uint32_t width,
    uint32_t height,
    TexTiling tiling,
    ExternalFrameExportMode export_mode,
    uint32_t export_drm_fourcc,
    const std::vector<uint64_t>& export_drm_modifiers,
    ExternalFrameMemoryPreference memory_preference) {
    if (!m_inited || !m_device || !m_ex_swapchain || !m_instance.offscreen())
        return false;

    const bool ok = m_ex_swapchain->Reconfigure(*m_device,
                                                width,
                                                height,
                                                tiling == TexTiling::OPTIMAL
                                                    ? VK_IMAGE_TILING_OPTIMAL
                                                    : VK_IMAGE_TILING_LINEAR,
                                                export_mode,
                                                export_drm_fourcc,
                                                export_drm_modifiers,
                                                memory_preference);
    LOG_INFO("HanabiScene Vulkan: offscreen export reconfigure %s size=%ux%u "
             "fourcc=0x%08x modifier-count=%zu first-modifier=0x%016llx memory=%s",
             ok ? "succeeded" : "failed",
             width,
             height,
             export_drm_fourcc,
             export_drm_modifiers.size(),
             export_drm_modifiers.empty()
                 ? 0ull
                 : static_cast<unsigned long long>(export_drm_modifiers.front()),
             ExternalMemoryPreferenceName(memory_preference));
    return ok;
}

void VulkanRender::Impl::refreshImportedTextures(Scene& scene) {
    if (!m_device) return;

    for (const auto& key : scene.dirtyImportedTextureKeys) {
        scene.DropParsedImageCache(key);
        auto image = scene.ParseImageBlockingCached(key);
        if (!image) continue;
        m_device->tex_cache().CreateTex(*image);
        scene.DropParsedImageCache(key);
    }
    scene.dirtyImportedTextureKeys.clear();
}

void VulkanRender::Impl::processDeferredGraphPreparation(Scene& scene) {
    if (m_deferred_prepare_indices.empty()) return;
    if (m_device_faulted || !m_device) return;

    const auto batch_started_at = std::chrono::steady_clock::now();
    std::size_t attempted = 0;
    std::size_t prepared = 0;

    while (attempted < kDeferredPrepareMaxPassesPerFrame && !m_deferred_prepare_indices.empty()) {
        if (attempted != 0) {
            const auto batch_elapsed_ms =
                static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(
                                        std::chrono::steady_clock::now() - batch_started_at)
                                        .count()) /
                1000.0;
            if (batch_elapsed_ms >= kDeferredPrepareFrameBudgetMs) {
                break;
            }
        }

        const auto pass_index = m_deferred_prepare_indices.front();
        if (pass_index >= m_passes.size()) {
            m_deferred_prepare_indices.pop_front();
            m_deferred_waiting_indices_logged.erase(pass_index);
            continue;
        }

        auto* pass = m_passes[pass_index];
        if (pass == nullptr || pass->prepared()) {
            m_deferred_prepare_indices.pop_front();
            m_deferred_waiting_indices_logged.erase(pass_index);
            continue;
        }

        const auto key = pass->residencyKey();
        const auto resources_state = pass->requestDeferredPrepareResources(scene, *m_device);
        if (resources_state == DeferredPrepareResourcesState::Waiting) {
            if (m_deferred_waiting_indices_logged.insert(pass_index).second) {
                LOG_INFO("RenderGraphDeferredPrepareWait: index=%zu remaining=%zu key='%s'",
                         pass_index,
                         m_deferred_prepare_indices.size(),
                         key.c_str());
            }
            break;
        }

        m_deferred_waiting_indices_logged.erase(pass_index);

        const auto pass_started_at = std::chrono::steady_clock::now();
        pass->prepareDeferred(scene, *m_device, m_rendering_resources);
        const auto pass_elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                         std::chrono::steady_clock::now() - pass_started_at)
                                         .count();
        attempted++;
        if (pass->prepared()) {
            prepared++;
            m_deferred_prepare_indices.pop_front();
        }

        LOG_INFO("RenderGraphDeferredPreparePass: index=%zu prepared=%s remaining=%zu "
                 "duration=%.2fms key='%s'",
                 pass_index,
                 pass->prepared() ? "true" : "false",
                 m_deferred_prepare_indices.size(),
                 static_cast<double>(pass_elapsed_us) / 1000.0,
                 key.c_str());
        if (!pass->prepared()) {
            break;
        }
        const auto batch_elapsed_ms =
            static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(
                                    std::chrono::steady_clock::now() - batch_started_at)
                                    .count()) /
            1000.0;
        if (batch_elapsed_ms >= kDeferredPrepareFrameBudgetMs) {
            break;
        }
    }

    const auto batch_elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                      std::chrono::steady_clock::now() - batch_started_at)
                                      .count();
    if (attempted != 0) {
        LOG_INFO("RenderGraphDeferredPrepareBatch: attempted=%zu prepared=%zu remaining=%zu "
                 "duration=%.2fms",
                 attempted,
                 prepared,
                 m_deferred_prepare_indices.size(),
                 static_cast<double>(batch_elapsed_us) / 1000.0);
    }

    if (m_deferred_prepare_indices.empty()) {
        m_deferred_waiting_indices_logged.clear();
        m_device->tex_cache().EndDeferredGraphActivation();
        LOG_INFO("RenderGraphDeferredPrepareComplete");
    }
}

void VulkanRender::Impl::ClearDestDrawPasses() {
    m_dest_leftover.clear();
    m_dest_leftover_mvp.clear();
    m_dest_postfx.clear();
    m_dest_lastpass.clear();
    m_dest_draw_all.clear();
}

void VulkanRender::Impl::IndexDestDrawPass(VulkanPass* pass) {
    if (pass == nullptr) return;
    const DestDrawPhase phase = pass->destDrawPhase();
    if (phase == DestDrawPhase::None) return;
    const int32_t layer_id = pass->destDrawLayerId();
    m_dest_draw_all.push_back(pass);
    if (phase == DestDrawPhase::Leftover) {
        m_dest_leftover[layer_id].push_back(pass);
        return;
    }
    if (phase == DestDrawPhase::LeftoverMvp) {
        m_dest_leftover_mvp[layer_id] = pass;
        return;
    }
    if (phase == DestDrawPhase::PostFx) {
        m_dest_postfx[layer_id].push_back(pass);
        return;
    }
    m_dest_lastpass[layer_id] = pass;
}

void VulkanRender::Impl::PrepareDestDrawPasses(Scene& scene) {
    for (auto* pass : m_dest_draw_all) {
        if (pass == nullptr) continue;
        if (pass->prepared()) {
            pass->refreshResources(scene, *m_device, m_rendering_resources);
            continue;
        }
        pass->prepare(scene, *m_device, m_rendering_resources);
    }
}

void VulkanRender::Impl::RefreshDestDrawPasses(
    Scene& scene, bool refresh_all,
    const std::unordered_set<std::string>& dirty_render_targets,
    const std::unordered_set<std::string>& dirty_imported_textures,
    const std::unordered_set<int32_t>& dirty_text_layers) {
    for (auto* pass : m_dest_draw_all) {
        if (pass == nullptr) continue;
        const bool broader_resource_refresh =
            refresh_all || pass->referencesAnyRenderTarget(dirty_render_targets) ||
            pass->referencesAnyTextLayer(dirty_text_layers);
        const bool imported_binding_refresh =
            pass->referencesAnyImportedTexture(dirty_imported_textures);
        if (!broader_resource_refresh && !imported_binding_refresh) continue;
        if (pass->prepared()) {
            if (broader_resource_refresh) {
                pass->refreshResources(scene, *m_device, m_rendering_resources);
            } else {
                pass->refreshImportedTextureBindings(scene, *m_device);
            }
        }
        if (!pass->prepared()) {
            pass->prepare(scene, *m_device, m_rendering_resources);
        }
    }
}

void VulkanRender::Impl::DestroyDestDrawPasses() {
    std::unordered_set<VulkanPass*> destroyed;
    for (auto* pass : m_dest_draw_all) {
        DestroyPassOnce(pass, *m_device, m_rendering_resources, destroyed);
    }
    ClearDestDrawPasses();
}

void VulkanRender::Impl::Record(wallpaper::SceneObject& object) {
    // LEFTOVER_VS_DESTDRAW: leftover IMAGE_VT_E8 then POSTFX then last-pass
    // 0x1401ea151→0x1401ebf60 while dest-STACK is live. Last-pass g_MVP is
    // +0x930 (ENGINE_FLUSH / VERTICAL_MVP_ID 0x1400d8676).
    RenderingResources& rr = m_rendering_resources;
    const int32_t id = object.id();
    const auto leftover_it = m_dest_leftover.find(id);
    const auto leftover_mvp_it = m_dest_leftover_mvp.find(id);
    const auto postfx_it = m_dest_postfx.find(id);
    const auto last_it = m_dest_lastpass.find(id);
    const bool has_dest_draw_pass =
        leftover_it != m_dest_leftover.end() ||
        leftover_mvp_it != m_dest_leftover_mvp.end() ||
        postfx_it != m_dest_postfx.end() || last_it != m_dest_lastpass.end();
    if (has_dest_draw_pass && object.scene() != nullptr &&
        m_dest_draw_contracts_logged.insert(id).second) {
        /*
         * Emit the resolved contract once per layer instead of logging a selected workshop id
         * every frame. The local card bounds, FetchDest, live dest stack, and final draw matrix
         * are the four independent inputs that determine a dest-draw clip edge. Keeping them in
         * one record makes a visible rectangular seam attributable without project-specific
         * conditions or high-volume frame logging.
         */
        float min_x = 0.0f;
        float max_x = 0.0f;
        float min_y = 0.0f;
        float max_y = 0.0f;
        if (const auto* mesh = object.lastpass_mesh();
            mesh != nullptr && mesh->VertexCount() > 0) {
            const auto& va = mesh->GetVertexArray(0);
            const float* data = va.Data();
            if (data != nullptr && va.VertexCount() > 0 && va.OneSize() >= 2) {
                min_x = max_x = data[0];
                min_y = max_y = data[1];
                for (std::size_t i = 1; i < va.VertexCount(); ++i) {
                    const float x = data[i * va.OneSize()];
                    const float y = data[i * va.OneSize() + 1];
                    min_x = std::min(min_x, x);
                    max_x = std::max(max_x, x);
                    min_y = std::min(min_y, y);
                    max_y = std::max(max_y, y);
                }
            }
        }
        const Eigen::Matrix4f fetch = object.FetchDest();
        const Eigen::Matrix4f stack = object.scene()->DestStackTop();
        const Eigen::Matrix4f draw = object.scene()->LastPassDrawMvp(object);
        LOG_INFO("DestDrawContract: id=%d name='%s' kind=%d effects=%d "
                 "phases=[leftover=%zu leftover-mvp=%d postfx=%zu last=%d] "
                 "fetch-T=[%.3f %.3f] stack-T=[%.3f %.3f] "
                 "draw-T=[%.6f %.6f] draw-s=[%.9f %.9f] "
                 "mesh=[%.1f..%.1f, %.1f..%.1f] window=%dx%d",
                 id,
                 object.name().c_str(),
                 static_cast<int>(object.kind()),
                 object.effect_count(),
                 leftover_it != m_dest_leftover.end() ? leftover_it->second.size() : 0u,
                 leftover_mvp_it != m_dest_leftover_mvp.end() ? 1 : 0,
                 postfx_it != m_dest_postfx.end() ? postfx_it->second.size() : 0u,
                 last_it != m_dest_lastpass.end() ? 1 : 0,
                 fetch(0, 3),
                 fetch(1, 3),
                 stack(0, 3),
                 stack(1, 3),
                 draw(0, 3),
                 draw(1, 3),
                 draw(0, 0),
                 draw(1, 1),
                 min_x,
                 max_x,
                 min_y,
                 max_y,
                 object.scene()->window_width(),
                 object.scene()->window_height());
    }
    auto update_pass = [](VulkanPass* pass) {
        if (pass != nullptr && pass->prepared()) pass->updateBeforeUpload();
    };
    if (object.kind() == SceneObjectKind::Text && object.effect_count() > 0 &&
        object.scene() != nullptr) {
        // TEXT_2F0 0x140258a02 vt+0xb8 recreates leftover +0x2c8 / FullCompo
        // immediately before leftover Draw. Runtime dest_size can grow past
        // parse JSON+pad (live Date +0x2f0=1412). Refresh leftover TextPass
        // and HORIZONTAL so both Query the same AABB key this DestDraw.
        auto refresh_named = [this, &object, &rr](VulkanPass* pass) {
            if (pass == nullptr) return;
            // ShaderDrawCore::refreshResources returns false (and
            // CustomShaderPass unprepares) when a static dest card is
            // Dirty after TEXT_2F0 AABB growth. Official vt+0xb8 still
            // Draws this dest-draw; prepare() is the TREE recreate.
            if (pass->prepared()) {
                pass->refreshResources(*object.scene(), *m_device, rr);
            }
            if (!pass->prepared()) {
                pass->prepare(*object.scene(), *m_device, rr);
            }
        };
        if (leftover_it != m_dest_leftover.end()) {
            for (auto* pass : leftover_it->second) refresh_named(pass);
        }
        if (postfx_it != m_dest_postfx.end()) {
            for (auto* pass : postfx_it->second) refresh_named(pass);
        }
        if (last_it != m_dest_lastpass.end() && last_it->second != nullptr &&
            object.lastpass_mesh() != nullptr) {
            // TEXT_2F0 0x1402589da / POSTFX_MESH 0x1401ec667: last-pass
            // Draw uses live +0x2e8 from (int)+0x2f0, not a parse-time
            // snapshot. Graph build ChangeMeshDataFrom shares Data;
            // replacing lastpass_mesh unique_ptr used to leave VERTICAL
            // on the JSON+pad card (Date ±423 vs runtime ±723).
            if (auto* node = last_it->second->destDrawNode();
                node != nullptr && node->Mesh() != nullptr) {
                node->Mesh()->ChangeMeshDataFrom(*object.lastpass_mesh());
                node->Mesh()->SetDirty();
            }
        }
        if (last_it != m_dest_lastpass.end()) refresh_named(last_it->second);
    }
    if (object.kind() == SceneObjectKind::Image && object.scene() != nullptr &&
        object.leftover_mesh() != nullptr && leftover_it != m_dest_leftover.end()) {
        // IMAGE_2D8_NOFULLFB 0x1401eb180 / IMAGE_VT_E8 0x140208067:
        // leftover Draw +0x2d8 is flags=0 0..AABB, not last-pass
        // +0x2e8 ±half. Owner image Mesh() stays Dynamic() after
        // ChangeMeshDataFrom; rebind the live leftover card before
        // dest-ortho flush so IMAGE leftover is not a puppet/±half
        // card under dest-ortho (0,W,0,H).
        for (auto* pass : leftover_it->second) {
            if (auto* node = pass->destDrawNode();
                node != nullptr && node->Mesh() != nullptr) {
                node->Mesh()->ChangeMeshDataFrom(*object.leftover_mesh());
                node->Mesh()->SetDirty();
            }
            if (pass->prepared()) {
                pass->refreshResources(*object.scene(), *m_device, rr);
            }
            if (!pass->prepared()) {
                pass->prepare(*object.scene(), *m_device, rr);
            }
        }
    }
    if (leftover_it != m_dest_leftover.end()) {
        for (auto* pass : leftover_it->second) {
            update_pass(pass);
            // DEST_ORTHO_TNF leftover +0x320>0 +0x930 is dest-ortho * I.
            // IMAGE_VT_F0 leftover +0x320==0 live +0x110 id 0xd uploads
            // +0x8f0 (LastPassDrawMvp). Clock TEXT_VT_F0 is not this path.
            if (pass != nullptr && pass->prepared() && object.scene() != nullptr) {
                if (object.effect_count() > 0) {
                    // DEST_ORTHO_TNF dest=I. IMAGE leftover mesh is 0..AABB
                    // (IMAGE_2D8_NOFULLFB). TEXT leftover is TEXT_LAYOUT_VERTS
                    // 0..AABB, same dest-ortho, not ±half * to_dest_center.
                    const Eigen::Matrix4f leftover_mvp =
                        object.scene()->LeftoverDestOrthoMvp(object);
                    pass->writeLastPassMvp(leftover_mvp);
                    // Leftover I is identity (IMAGE_VT_E8 0x140207c74 skips
                    // I write). +0x8f0 = I * dest-ortho = dest-ortho
                    // (ENGINE_FLUSH 0x1400d4323). IMAGE leftover +0x110 is
                    // 0xd (LASTPASS_IMAGE_ID / IMAGE_VT_F0 0x1400d8749).
                    // Same dest-ortho matrix, not leftover +0x8f0 copied
                    // into last-pass +0x930.
                    if (object.kind() == SceneObjectKind::Image) {
                        pass->writeLastPassInverseSlot(leftover_mvp);
                    }
                } else if (object.kind() == SceneObjectKind::Image) {
                    const Eigen::Matrix4f draw =
                        object.scene()->LastPassDrawMvp(object);
                    pass->writeLastPassMvp(draw);
                    // IMAGE_VT_F0 live +0x110 id 0xd copies +0x8f0
                    // (UNIFORM_UPLOAD_MAP 0x1400d8749). Same matrix as
                    // LastPassDrawMvp, not inverse(+0x930).
                    pass->writeLastPassInverseSlot(draw);
                } else if (object.kind() == SceneObjectKind::Text) {
                    // Clock TEXT_VT_F0 +0x320==0 FONT_MVP_SLOT id 0xb
                    // copies +0x930 (0x1400d8676). TEXT_VT_F0 0x1402580b0
                    // sets +0x1ca=1 so flush ENGINE_FLUSH 0x1400d4264
                    // +0x930=camera*dest. Live 3219908811 Clock dest_p is
                    // BASE+0x40 Path B dest-STACK (T≈parallax, not ctor
                    // T=0; DEST_LIVE_WRITERS skip-Path-B is stale).
                    // camera is FitOrtho (LASTPASS_CAM_ORTHO). DestDraw
                    // already FlushLastPassMvp. +0x594 bit2 clear je
                    // 0x1402583a8 skips fontbackground +0x2d8. Do not
                    // write LastPassDrawMvp / +0x8f0 into +0x930.
                    pass->writeLastPassMvp(object.scene()->LastPassMvp());
                }
            }
        }
    }
    if (postfx_it != m_dest_postfx.end()) {
        for (auto* pass : postfx_it->second) {
            update_pass(pass);
            // GFX_ORTHO_CALLS 0x1401e9a1b: after leftover dest-ortho Draw,
            // leftover-inner type-0 (r15=0, +0x2e0 ±1) overwrites *camera
            // with clip ortho(-1,1,-1,1). dest is still I (DEST_ORTHO_TNF
            // 0x1401e9702; pop is 0x1401e9cd3 after this loop). +0x930 =
            // clip-ortho * I = identity 2D. DestStackTop pixel T would
            // throw the ±1 card off the named RT. Not FetchDest. Last-pass
            // after DEST_BLIT stays LastPassMvp / LastPassDrawMvp.
            if (pass != nullptr && pass->prepared()) {
                pass->writeLastPassMvp(Eigen::Matrix4f::Identity());
            }
        }
    }
    VulkanPass* leftover_mvp =
        leftover_mvp_it != m_dest_leftover_mvp.end() ? leftover_mvp_it->second : nullptr;
    if (leftover_mvp != nullptr && leftover_mvp->prepared() && object.scene() != nullptr) {
        leftover_mvp->updateBeforeUpload();
        // IMAGE_VT_F8 leftover-MVP: leftover material combo 0xd uploads
        // +0x8f0 (LastPassDrawMvp). Puppet Draws +0x490 (IMAGE_VT_F8_PUPPET
        // / PUPPET_490). No-puppet Draws +0x2e8 (IMAGE_VT_F8 0x1402090fd).
        // Not dest-ortho leftover +0x2d8 and not POSTFX last-pass leftover
        // RT dest-card (0x1401ea151).
        const Eigen::Matrix4f draw = object.scene()->LastPassDrawMvp(object);
        leftover_mvp->writeLastPassMvp(draw);
        leftover_mvp->writeLastPassInverseSlot(draw);
    }
    VulkanPass* last = last_it != m_dest_lastpass.end() ? last_it->second : nullptr;
    if (last != nullptr && last->prepared()) {
        last->updateBeforeUpload();
        if (object.scene() != nullptr) {
            // VERTICAL_MVP_ID 0x1400d8676: Date blur VERTICAL g_MVP is id
            // 0xb copies +0x930 (LastPassMvp). LASTPASS_IMAGE_ID /
            // UNIFORM_UPLOAD_MAP 0x1400d8749: IMAGE last-pass combo +0x110
            // is 0xd then 2; that upload is +0x8f0 = LastPassDrawMvp. Do
            // not copy +0x8f0 into +0x930 (LASTPASS_8F0_T). Clock leftover
            // is FONT_MVP_SLOT 0xb, not this.
            if (object.kind() == SceneObjectKind::Image) {
                const Eigen::Matrix4f draw =
                    object.scene()->LastPassDrawMvp(object);
                last->writeLastPassMvp(draw);
                last->writeLastPassInverseSlot(draw);
            } else {
                last->writeLastPassMvp(object.scene()->LastPassMvp());
            }
        }
    }
    if (m_vertex_buf) m_vertex_buf->recordUpload(rr.command);
    if (m_dyn_buf) m_dyn_buf->recordUpload(rr.command);
    rr.immutable_meshes.recordUploads(rr.command);
    // IMAGE_VT_E8 0x140207c74 leftover Draw +0x2d8 after flush of
    // [+0x4a0]/[+0x498]. Official layer albedo is already a D3D11 SRV.
    // TEXT_E0_IDEST 0x1401e9681 OMSet leftover named-RT then 0x1401e968a
    // TEXT_E0 samples FullFB. TEXT_E8 leftover glyphs follow. Official
    // leftover RT, FullFB, and glyph atlas exist at that Draw. TREE
    // Query during refresh queues UNDEFINED RTs and imported image
    // uploads; dest-draw runs in the compose walker before the later
    // frame RecordUploads. Flush both now so leftover IMAGE / TEXT_E0
    // sample a shader-readable resource, not an unuploaded image.
    // Bootstrap RT clear is still not official leftover content —
    // official first write is composelayer_clearalpha / IMAGE leftover
    // Draw.
    m_device->tex_cache().RecordUploads(rr.command);
    m_device->video_tex_cache().RecordUploads(rr.command);
    auto execute_pass = [this, &rr](VulkanPass* pass) {
        if (pass != nullptr && pass->prepared()) pass->execute(*m_device, rr);
    };
    if (leftover_it != m_dest_leftover.end()) {
        for (auto* pass : leftover_it->second) execute_pass(pass);
    }
    if (postfx_it != m_dest_postfx.end()) {
        for (auto* pass : postfx_it->second) execute_pass(pass);
    }
    if (leftover_mvp != nullptr && leftover_mvp->prepared()) {
        // IMAGE_VT_F8_PUPPET leftover-MVP node Mesh() is +0x490 at prepare
        // (PUPPET_490). Do not swap leftover dest-ortho +0x2d8 after upload.
        execute_pass(leftover_mvp);
    }
    execute_pass(last);
}

void VulkanRender::Impl::drawFrameSwapchain() {
    static size_t resource_index = 0;

    RenderingResources& rr = m_rendering_resources;
    resource_index         = (resource_index + 1) % 3;
    uint32_t image_index   = 0;
    {
        if (!checkVkResult(m_device->handle().AcquireNextImageKHR(*m_device->swapchain().handle(),
                                                                  vk_wait_time,
                                                                  *rr.sem_swap_wait_image,
                                                                  {},
                                                                  &image_index),
                           "acquire swapchain image"))
            return;
    }
    const auto& image = m_device->swapchain().images()[image_index];

    m_finpass->setPresent(image);

    for (auto* p : m_passes) {
        if (p->prepared()) {
            // Dynamic passes copy current CPU-side vertex/index bytes into staging before the
            // upload command is recorded. This keeps reused-source particle systems from binding a
            // freshly grown suballocation whose GPU contents have not been uploaded yet.
            p->updateBeforeUpload();
        }
    }

    if (!checkVkResult(rr.command.Begin(VkCommandBufferBeginInfo {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    }), "begin swapchain frame command buffer"))
        return;
    // Official last-pass writes leftover FullFB after the frame clear
    // (POSTFX_OMSET / LEFTOVER_VS_DESTDRAW). PrePass must run first;
    // executing it after DestDraw wipes Default.
    if (m_prepass != nullptr && m_prepass->prepared()) {
        m_prepass->execute(*m_device, rr);
    }
    if (rr.scene != nullptr) RunComposeDrawWalker(*rr.scene, this);
    // Deferred pass preparation can allocate and write static vertex/index subranges between
    // frames. Recording the static upload here keeps those newly resident passes drawable without
    // a compile-time WaitIdle, matching the frame-budgeted residency model used by streaming
    // renderers.
    m_vertex_buf->recordUpload(rr.command);
    m_dyn_buf->recordUpload(rr.command);
    rr.immutable_meshes.recordUploads(rr.command);
    m_device->tex_cache().RecordUploads(rr.command);
    m_device->video_tex_cache().RecordUploads(rr.command);
    for (auto* p : m_passes) {
        if (p == m_prepass.get()) continue;
        // LEFTOVER_VS_DESTDRAW 0x1401e8ed9 / 0x1401ea151: leftover then
        // POSTFX then last-pass are one dest-draw. A second DONT_CARE
        // HORIZONTAL / IMAGE leftover in m_passes discards that write
        // (FullCompo / leftover named-RT go uninit white). Official
        // walker does not run 0x1401ebf60 again after DEST_DRAW_JOIN.
        if (p->destDrawPhase() != DestDrawPhase::None) continue;
        if (p->prepared()) {
            p->execute(*m_device, rr);
        }
    }
    if (!checkVkResult(rr.command.End(), "end swapchain frame command buffer"))
        return;

    VkPipelineStageFlags wait_dst_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo         sub_info {
                .sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                .pNext                = nullptr,
                .waitSemaphoreCount   = 1,
                .pWaitSemaphores      = rr.sem_swap_wait_image.address(),
                .pWaitDstStageMask    = &wait_dst_stage,
                .commandBufferCount   = 1,
                .pCommandBuffers      = rr.command.address(),
                .signalSemaphoreCount = 1,
                .pSignalSemaphores    = rr.sem_swap_finish.address(),
    };

    if (!checkVkResult(m_device->present_queue().handle.Submit(sub_info, *rr.fence_frame),
                       "submit swapchain frame"))
        return;
    VkPresentInfoKHR present_info {
        .sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .pNext              = nullptr,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores    = rr.sem_swap_finish.address(),
        .swapchainCount     = 1,
        .pSwapchains        = m_device->swapchain().handle().address(),
        .pImageIndices      = &image_index,
    };
    if (!checkVkResult(m_device->present_queue().handle.Present(present_info),
                       "present swapchain frame"))
        return;

    if (!checkVkResult(rr.fence_frame.Wait(vk_wait_time), "wait swapchain frame fence"))
        return;
    m_device->tex_cache().RetireCompletedUploads();
    rr.immutable_meshes.retireStaging();
    if (rr.scene != nullptr) ReleaseUploadedFileMeshCpu(*rr.scene);
    if (!checkVkResult(rr.fence_frame.Reset(), "reset swapchain frame fence"))
        return;
}

void VulkanRender::Impl::drawFrameOffscreen(Scene& scene) {
    RenderingResources& rr = m_rendering_resources;
    if (!m_ex_swapchain) {
        return;
    }

    auto render_lock = m_ex_swapchain->acquireRenderLock();
    auto* inprogress_handle = m_ex_swapchain->getInprogress();
    if (!inprogress_handle) {
        return;
    }

    const uint32_t slot_id = static_cast<uint32_t>(inprogress_handle->id());
    if (m_offscreen_frame_release_cb) {
        /*
         * Vivid reuses the exported offscreen image ring directly as the display
         * transport. Mirror waywallen's BridgeProducerCore::acquireSlot(): wait
         * for the consumer release timeline immediately before recording GPU
         * writes into the selected slot. A timeout skips this render tick without
         * calling renderFrame(), so the ready slot remains the last fully
         * published image and no still-owned DMA-BUF is overwritten.
         */
        if (!m_offscreen_frame_release_cb(slot_id)) {
            return;
        }
    }

    ImageParameters image = m_ex_swapchain->GetInprogressImage();

    rr.msaa_compose_dirty = false;
    rr.scene = &scene;
    m_finpass->setPresent(image);

    for (auto* p : m_passes) {
        if (p->prepared()) {
            // Offscreen rendering exports the result to GTK, making stale particle bytes visible as
            // source-switch flicker. Pre-updating dynamic mesh data aligns the following m_dyn_buf
            // upload with the frame that will be exported.
            p->updateBeforeUpload();
        }
    }

    if (!checkVkResult(rr.command.Begin(VkCommandBufferBeginInfo {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    }), "begin offscreen frame command buffer"))
        return;
    // Official last-pass writes leftover FullFB after the frame clear
    // (POSTFX_OMSET / LEFTOVER_VS_DESTDRAW). Offscreen capture uses
    // this path; PrePass after DestDraw wipes 3363252053's Default.
    if (m_prepass != nullptr && m_prepass->prepared()) {
        m_prepass->execute(*m_device, rr);
    }
    RunComposeDrawWalker(scene, this);
    m_vertex_buf->recordUpload(rr.command);
    m_dyn_buf->recordUpload(rr.command);
    rr.immutable_meshes.recordUploads(rr.command);
    m_device->tex_cache().RecordUploads(rr.command);
    m_device->video_tex_cache().RecordUploads(rr.command);

    for (auto* p : m_passes) {
        if (p == m_prepass.get()) continue;
        if (! p->prepared()) continue;
        // LEFTOVER_VS_DESTDRAW 0x1401e8ed9 / 0x1401ea151: dest-draw
        // already ran in the compose walker. Re-executing DONT_CARE
        // HORIZONTAL / IMAGE leftover here discards FullCompo / leftover
        // (SceneToRenderGraph dest-draw copies in m_passes feed empty
        // FullCompo). Official join has no second 0x1401ebf60.
        if (p->destDrawPhase() != DestDrawPhase::None) continue;
        p->execute(*m_device, rr);
    }

    if (!checkVkResult(rr.command.End(), "end offscreen frame command buffer"))
        return;

    VkSubmitInfo sub_info {
        .sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext              = nullptr,
        .commandBufferCount = 1,
        .pCommandBuffers    = rr.command.address(),
    };
    if (!checkVkResult(m_device->graphics_queue().handle.Submit(sub_info, *rr.fence_frame),
                       "submit offscreen frame"))
        return;

    if (!checkVkResult(rr.fence_frame.Wait(vk_wait_time), "wait offscreen frame fence"))
        return;
    m_device->tex_cache().RetireCompletedUploads();
    rr.immutable_meshes.retireStaging();
    ReleaseUploadedFileMeshCpu(scene);
    if (!checkVkResult(rr.fence_frame.Reset(), "reset offscreen frame fence"))
        return;
    m_ex_swapchain->renderFrame();
}

void VulkanRender::Impl::setRenderTargetSize(Scene& scene, rg::RenderGraph& rg) {
    SyncSceneMsaa(scene, *m_device);
    auto& ext = m_device->out_extent();
    for (auto& item : scene.renderTargets) {
        auto& rt = item.second;
        if (rt.bind.enable && rt.bind.screen) {
            const i32 new_w = (i32)(rt.bind.scale * ext.width);
            const i32 new_h = (i32)(rt.bind.scale * ext.height);
            if (rt.width != new_w || rt.height != new_h) {
                LOG_INFO("SceneScreenBoundTarget: name='%s' previous=[%d %d] screen=[%d %d] "
                         "scale=%.3f extent=%ux%u",
                         item.first.c_str(),
                         rt.width,
                         rt.height,
                         new_w,
                         new_h,
                         rt.bind.scale,
                         ext.width,
                         ext.height);
            }
            rt.width  = new_w;
            rt.height = new_h;
            // Screen-sized render targets expose the full framebuffer as both their physical and
            // logical extent. Only text-owned runtime targets intentionally diverge these values.
            rt.mapWidth = rt.width;
            rt.mapHeight = rt.height;
        }
    }
    for (auto& item : scene.renderTargets) {
        auto& rt = item.second;
        if (rt.bind.screen || ! rt.bind.enable) continue;
        auto bind_rt = scene.renderTargets.find(rt.bind.name);
        if (rt.bind.name.empty() || bind_rt == scene.renderTargets.end()) {
            LOG_ERROR("unknonw render target bind: %s", rt.bind.name.c_str());
            continue;
        }
        rt.width  = (i32)(rt.bind.scale * bind_rt->second.width);
        rt.height = (i32)(rt.bind.scale * bind_rt->second.height);
        // Bound render targets inherit the sampled content rectangle from their source target so
        // shader uniforms continue to describe the authored image area rather than the raw backing
        // allocation. This keeps generic effect chains consistent whenever the source target uses
        // a logical content rectangle that differs from its physical allocation.
        rt.mapWidth = (i32)(rt.bind.scale * bind_rt->second.ContentWidth());
        rt.mapHeight = (i32)(rt.bind.scale * bind_rt->second.ContentHeight());
    }
    for (auto& item : scene.renderTargets) {
        auto& rt = item.second;
        if (! item.first.empty() && (rt.width * rt.height <= 4)) {
            LOG_ERROR("wrong size for render target: %s", item.first.c_str());
        } else if (rt.has_mipmap) {
            rt.mipmap_level =
                std::max(3u,
                         static_cast<uint>(std::floor(std::log2(std::min(rt.width, rt.height))))) -
                2u;
        }
    }
    scene.shaderValueUpdater->SetScreenSize((i32)ext.width, (i32)ext.height);
}

void VulkanRender::Impl::UpdateCameraFillMode(wallpaper::Scene&   scene,
                                              wallpaper::FillMode fillmode) {
    using namespace wallpaper;
    auto width  = m_device->out_extent().width;
    auto height = m_device->out_extent().height;

    if (width == 0 || height == 0) return;
    scene.physicalOutputExtent = { width, height };
    scene.ApplyTextureResolutionForCurrentOutput();
    double sw = scene.ortho[0], sh = scene.ortho[1];
    double fboAspect = width / (double)height, sAspect = sw / sh;
    auto&  gCam    = *scene.cameras.at("global");
    auto&  gPerCam = *scene.cameras.at("global_perspective");

    // Camera-layer animation mutates the same shared "global" camera object that fill mode uses to
    // adapt a 16:9 project to the monitor aspect. Preserve the live zoom value here so the render
    // side remains the single source of truth for framebuffer-relative width/height while authored
    // zoom still narrows that already aspect-correct view.
    double active_global_zoom = scene.defaultGlobalCameraZoom;
    if (!std::isfinite(active_global_zoom) || active_global_zoom <= 0.0001) {
        active_global_zoom = 1.0;
    }

    // Perspective camera layers can animate FOV directly. Keep that authored value when the active
    // layer explicitly targets the shared perspective camera. Otherwise use the scene's authored
    // particle-camera FOV. Derive an angle from the 1000-unit eye distance only when that override
    // is missing or non-finite.
    bool  use_active_global_perspective_fov = false;
    float active_global_perspective_fov = 50.0f;
    if (scene.activeCameraLayerId != 0) {
        auto active_layer_it = scene.cameraLayers.find(scene.activeCameraLayerId);
        if (active_layer_it != scene.cameraLayers.end()) {
            const auto& active_layer = active_layer_it->second;
            if (active_layer.camera_name.empty() || active_layer.camera_name == "global") {
                if (std::isfinite(active_layer.zoom) && active_layer.zoom > 0.0001) {
                    active_global_zoom = active_layer.zoom;
                } else {
                    active_global_zoom = 1.0;
                }
            } else if (active_layer.camera_name == "global_perspective" &&
                       std::isfinite(active_layer.fov) && active_layer.fov > 0.0001f) {
                use_active_global_perspective_fov = true;
                active_global_perspective_fov = active_layer.fov;
            }
        }
    }

    double framed_width = sw;
    double framed_height = sh;
    double perspective_aspect = sAspect;

    switch (fillmode) {
    case FillMode::STRETCH:
        framed_width = sw;
        framed_height = sh;
        perspective_aspect = sAspect;
        break;
    case FillMode::ASPECTFIT:
        if (fboAspect < sAspect) {
            // Preserve the full scene width and add vertical space when the output is taller.
            framed_width = sw;
            framed_height = sw / fboAspect;
        } else {
            framed_width = sh * fboAspect;
            framed_height = sh;
        }
        perspective_aspect = fboAspect;
        break;
    case FillMode::ASPECTCROP:
    default:
        if (fboAspect > sAspect) {
            // Preserve the full scene width and crop vertically when the output is wider.
            framed_width = sw;
            framed_height = sw / fboAspect;
        } else {
            framed_width = sh * fboAspect;
            framed_height = sh;
        }
        perspective_aspect = fboAspect;
        break;
    }

    gCam.SetWidth(std::max(1.0, framed_width / active_global_zoom));
    gCam.SetHeight(std::max(1.0, framed_height / active_global_zoom));
    gPerCam.SetAspect(perspective_aspect);
    gPerCam.SetFov(use_active_global_perspective_fov
                       ? active_global_perspective_fov
                       : algorism::ResolvePerspectiveFov(scene.perspectiveOverrideFov,
                                                         gCam.Height()));
    gCam.Update();
    gPerCam.Update();
    scene.UpdateLinkedCamera("global");

    if (!scene.modelPerspectiveCameraName.empty()) {
        auto model_camera_it = scene.cameras.find(scene.modelPerspectiveCameraName);
        if (model_camera_it != scene.cameras.end() && model_camera_it->second) {
            // 3D model chunks render through a camera that is intentionally isolated from the
            // legacy `global_perspective` camera, but its projection still has to follow the same
            // fill-mode-adjusted framebuffer aspect. Without this, a 16:9-authored model scene keeps
            // its native projection while Vulkan draws into a 16:10 or other non-native viewport,
            // which changes the apparent object proportions even though the model transform itself
            // is uniform. Only the aspect is synchronized here: the authored 3D FOV and the
            // camera-path eye/center/up basis remain owned by the scene data and path playback.
            model_camera_it->second->SetAspect(perspective_aspect);
            model_camera_it->second->Update();
            scene.UpdateLinkedCamera(scene.modelPerspectiveCameraName);
        } else {
            // A named model camera should exist whenever model nodes were materialized. Log this
            // explicitly instead of silently falling back to another camera, because choosing a
            // substitute would hide the real render-graph/parser state mismatch and make 3D aspect
            // regressions harder to diagnose from run.log.
            LOG_ERROR("Scene3DModelCameraAspect: missing model camera '%s' while applying "
                      "fill-mode perspective aspect %.6f",
                      scene.modelPerspectiveCameraName.c_str(),
                      perspective_aspect);
        }
    }

    // Text layers with Wallpaper Engine's screen-anchor property are authored against the project
    // canvas edge, but the active orthographic camera edge moves when aspect crop/fit changes the
    // visible frame. Re-apply those anchor transforms after camera framing so HUD-style text
    // remains inside the actual output instead of the uncropped project bounds.
    ApplyTextLayerScreenAnchorTransforms(scene);
    // Text bridge images are final-screen-density resources, not glyph-atlas-density resources.
    // Recompute them only after fill mode and screen anchors have produced the frame's authoritative
    // camera and world transforms.
    UpdateAllTextLayerBridgeBackings(scene);
}

void VulkanRender::Impl::clearLastRenderGraph(bool clear_scene_caches) {
    if (m_device_faulted) {
        // After device loss, pass destruction can call vkDestroyPipeline and friends on a driver
        // context that already timed out.  Leave the bounded stale graph abandoned with the renderer
        // instead of turning a recoverable backend replacement into a process crash.
        return;
    }

    // A topology rebuild invalidates the compiled pass list and the backing mesh buffers that were
    // uploaded for the previous graph. Reallocating those buffers keeps the full rebuild path
    // conservative and mirrors the historical behavior used when nodes were added or removed.
    DestroyDestDrawPasses();
    for (auto& p : m_passes) {
        p->destory(*m_device, m_rendering_resources);
    }
    m_passes.clear();
    m_compiled_pass_refs.clear();
    m_deferred_prepare_indices.clear();
    m_deferred_waiting_indices_logged.clear();
    m_device->tex_cache().CancelDeferredGraphActivation();
    if (clear_scene_caches) {
        // Scene switches and renderer shutdown still own a full cache teardown. Ordinary topology
        // rebuilds no longer do this: visibility-driven residency now releases only the keys that
        // became unreachable, so showing one deferred layer cannot evict every unrelated texture
        // and video decoder in the wallpaper.
        m_device->tex_cache().Clear();
        m_device->video_tex_cache().Clear();
        if (m_rendering_resources.pipeline_cache) {
            m_rendering_resources.pipeline_cache->clear();
        }
    }
    // Shared model depth images are tied to the compiled graph's output targets. Dropping them on
    // full graph rebuilds keeps 3D model depth opt-in and avoids stale depth attachments surviving
    // after scene topology or render-target ownership changes.
    m_rendering_resources.model_depth_images.clear();
    m_rendering_resources.model_depth_resolved.clear();
    m_rendering_resources.masked_draw_attachments.clear();
    if (clear_scene_caches) {
        // Scene switches drop GPU file meshes. Ordinary topology rebuilds keep them: the host
        // vertex bytes are released after the first upload, and the exact-size VB/IB pairs are
        // the scene-lifetime objects.
        m_rendering_resources.immutable_meshes.clear();
    }

    m_vertex_buf->destroy();
    m_dyn_buf->destroy();

    m_vertex_buf->allocate();
    m_dyn_buf->allocate();
}

void VulkanRender::Impl::releasePendingSceneResources(Scene& scene) {
    if (m_device_faulted || !m_device) return;
    if (scene.pendingStaticTextureReleaseKeys.empty() &&
        scene.pendingVideoTextureReleaseKeys.empty() &&
        scene.pendingRenderTargetReleaseKeys.empty()) {
        return;
    }

    const auto before_texture_bytes = m_device->tex_cache().GetTrackedBytes();
    const auto before_texture_count = m_device->tex_cache().GetTrackedImageCount();
    const auto before_video_bytes   = m_device->video_tex_cache().GetTrackedBytes();
    const auto before_video_count   = m_device->video_tex_cache().GetTrackedEntryCount();

    std::size_t released_static = 0;
    std::size_t released_render_targets = 0;
    std::size_t released_videos = 0;

    for (const auto& key : scene.pendingStaticTextureReleaseKeys) {
        if (m_device->tex_cache().ReleaseTexture(key)) released_static++;
        scene.DropParsedImageCache(key);
    }
    for (const auto& key : scene.pendingRenderTargetReleaseKeys) {
        if (m_device->tex_cache().ReleaseRenderTarget(key)) released_render_targets++;
    }
    for (const auto& key : scene.pendingVideoTextureReleaseKeys) {
        if (m_device->video_tex_cache().Release(key)) released_videos++;
    }

    LOG_INFO("SceneResidencyRelease: static=%zu/%zu render-target=%zu/%zu video=%zu/%zu "
             "texture-bytes-before=%zu texture-bytes-after=%zu texture-images-before=%zu "
             "texture-images-after=%zu video-bytes-before=%zu video-bytes-after=%zu "
             "video-entries-before=%zu video-entries-after=%zu",
             released_static,
             scene.pendingStaticTextureReleaseKeys.size(),
             released_render_targets,
             scene.pendingRenderTargetReleaseKeys.size(),
             released_videos,
             scene.pendingVideoTextureReleaseKeys.size(),
             before_texture_bytes,
             m_device->tex_cache().GetTrackedBytes(),
             before_texture_count,
             m_device->tex_cache().GetTrackedImageCount(),
             before_video_bytes,
             m_device->video_tex_cache().GetTrackedBytes(),
             before_video_count,
             m_device->video_tex_cache().GetTrackedEntryCount());

    scene.pendingStaticTextureReleaseKeys.clear();
    scene.pendingVideoTextureReleaseKeys.clear();
    scene.pendingRenderTargetReleaseKeys.clear();
}

void VulkanRender::Impl::clearRenderGraphResources() {
    // Resource-only rebuilds are hot resource refreshes, not a miniature full rebuild. Particle
    // effects already update every frame without clearing global caches; effect-backed text must
    // follow the same rule. TextureCache::Query now detects per-key TextureKey changes and
    // reallocates only the resized render target, so clearing the entire cache here would recreate
    // unrelated offscreen images and reintroduce the minute-rollover hitch.
}

void VulkanRender::Impl::dropCompiledPassFramebuffers() {
    for (auto& pass_ref : m_compiled_pass_refs) {
        if (auto* pass = dynamic_cast<VulkanPass*>(pass_ref.get())) {
            pass->dropOutputFramebuffers();
        }
    }
    for (auto* pass : m_passes) {
        if (pass != nullptr) pass->dropOutputFramebuffers();
    }
}

void VulkanRender::Impl::compileRenderGraph(Scene& scene, rg::RenderGraph& rg,
                                            bool refresh_resources_only) {
    if (m_device_faulted) return;
    if (! m_inited) return;
    SyncSceneMsaa(scene, *m_device);
    const int msaa_samples = scene.MsaaSampleCount();
    if (msaa_samples != m_compiled_msaa_samples) {
        dropCompiledPassFramebuffers();
        m_rendering_resources.model_depth_images.clear();
        m_rendering_resources.model_depth_resolved.clear();
        m_rendering_resources.masked_draw_attachments.clear();
        m_compiled_msaa_samples = msaa_samples;
    }
    m_pass_loaded = false;
    const bool had_resident_graph = !m_compiled_pass_refs.empty();

    if (refresh_resources_only && !m_passes.empty()) {
        setRenderTargetSize(scene, rg);

        const auto dirty_render_targets = scene.dirtyRenderTargetKeys;
        const auto dirty_imported_textures = scene.dirtyImportedTextureResourceKeys;
        const auto dirty_text_layers = scene.dirtyTextLayerIds;
        // Resource refreshes can be targeted by output dependency, imported descriptor input, or
        // direct text-layer ownership. Treating one empty set as "refresh everything" would turn
        // every media or clock update back into a full pass walk.
        const bool has_targeted_dirty_resources =
            !dirty_render_targets.empty() || !dirty_imported_textures.empty() ||
            !dirty_text_layers.empty();
        const bool refresh_all =
            scene.renderGraphAllResourcesDirty || !has_targeted_dirty_resources;
        RefreshDestDrawPasses(scene, refresh_all, dirty_render_targets, dirty_imported_textures,
                              dirty_text_layers);
        std::size_t refreshed_passes = 0;
        std::size_t prepared_passes = 0;

        for (size_t pass_index = 0; pass_index < m_passes.size(); ++pass_index) {
            auto* p = m_passes[pass_index];
            if (p == nullptr) continue;

            const bool broader_resource_refresh =
                refresh_all || p->referencesAnyRenderTarget(dirty_render_targets) ||
                p->referencesAnyTextLayer(dirty_text_layers);
            const bool imported_binding_refresh =
                p->referencesAnyImportedTexture(dirty_imported_textures);
            const bool affected = broader_resource_refresh || imported_binding_refresh;
            if (!affected) continue;

            if (p->prepared()) {
                if (broader_resource_refresh) {
                    // Text bridge updates are now target-scoped like particle resources: refresh
                    // only passes that touch resized targets. The broad hook may recreate output
                    // resources, so it is reserved for changes that actually affect those outputs.
                    p->refreshResources(scene, *m_device, m_rendering_resources);
                } else {
                    // Imported media replacement changes sampled image descriptors only. Rebinding
                    // through the broad hook also re-queries effect ping-pong outputs, which can
                    // disturb an otherwise resident transition chain and make its overlay vanish.
                    p->refreshImportedTextureBindings(scene, *m_device);
                }
                refreshed_passes++;
            }
            if (!p->prepared()) {
                p->prepare(scene, *m_device, m_rendering_resources);
                prepared_passes++;
            }
        }

        // Resource-only refreshes are intentionally silent in production; the counters stay local
        // so the branch preserves targeted text-bridge behavior without making minute rollovers
        // spend time formatting render-graph diagnostics.
        (void)refresh_all;
        (void)refreshed_passes;
        (void)prepared_passes;
        // Mature renderers do not submit a separate upload command and idle the whole device while
        // rebuilding resource bindings. The next draw command records all dirty vertex, dynamic,
        // and texture uploads before executing passes, preserving ordering without a render-thread
        // queue drain.
        m_pass_loaded = true;
        return;
    }

    auto nodes             = rg.topologicalOrder();
    auto node_release_texs = rg.getLastReadTexs(nodes);

    m_passes.clear();
    ClearDestDrawPasses();
    m_deferred_prepare_indices.clear();
    m_deferred_waiting_indices_logged.clear();
    m_device->tex_cache().CancelDeferredGraphActivation();
    std::vector<VulkanPass*> graph_passes;
    graph_passes.reserve(nodes.size());

    std::unordered_map<std::string, std::shared_ptr<rg::Pass>> reusable_passes;
    std::unordered_map<std::string, std::size_t> old_key_counts;
    for (const auto& old_pass_ref : m_compiled_pass_refs) {
        auto old_pass = std::dynamic_pointer_cast<VulkanPass>(old_pass_ref);
        if (!old_pass) continue;
        const auto key = MakeResidencyInstanceKey(*old_pass, old_key_counts);
        if (!key.empty()) reusable_passes.emplace(key, old_pass_ref);
    }

    std::unordered_map<std::string, std::size_t> new_key_counts;
    std::unordered_set<VulkanPass*> reused_passes;
    std::vector<std::shared_ptr<rg::Pass>> next_compiled_pass_refs;
    next_compiled_pass_refs.reserve(nodes.size());
    std::size_t reused_count = 0;
    std::size_t new_count = 0;

    for (std::size_t index = 0; index < nodes.size(); ++index) {
        const auto node_id = nodes[index];
        auto       pass_ref = rg.getPassShared(node_id);
        assert(pass_ref != nullptr);
        auto*      vpass = dynamic_cast<VulkanPass*>(pass_ref.get());
        assert(vpass != nullptr);

        const auto key = MakeResidencyInstanceKey(*vpass, new_key_counts);
        if (!key.empty()) {
            if (auto reusable_it = reusable_passes.find(key);
                reusable_it != reusable_passes.end()) {
                auto reusable_vpass = std::dynamic_pointer_cast<VulkanPass>(reusable_it->second);
                if (reusable_vpass && reusable_vpass->canReuseForResidency(*vpass)) {
                    // Mature renderers do not destroy every pipeline just because one layer toggled
                    // visibility. The new render graph describes the fresh topology, while this
                    // handoff keeps matching prepared pass objects alive and updates only their
                    // graph-local gates/texture declarations. Removed passes are retired below,
                    // which preserves hidden-layer resource release without a whole-scene PSO
                    // rebuild.
                    reusable_vpass->absorbResidencyGraphState(*vpass);
                    pass_ref = reusable_it->second;
                    vpass = reusable_vpass.get();
                    rg.replacePass(node_id, pass_ref);
                    reused_passes.insert(vpass);
                    reused_count++;
                    reusable_passes.erase(reusable_it);
                }
            }
        }
        if (reused_passes.count(vpass) == 0) {
            new_count++;
        }

        // Release ownership is compiled from the current render graph topology, not from pass
        // construction. Clear stale metadata before assigning this graph's final-reader keys so
        // reused pass objects keep an exact lifecycle contract.
        vpass->clearReleaseTexs();
        for (auto& tex : node_release_texs[index]) {
            vpass->addReleaseTexs(spanone<const std::string_view> { tex->key() });
        }
        if (vpass->destDrawPhase() != DestDrawPhase::None) {
            IndexDestDrawPass(vpass);
        } else {
            graph_passes.push_back(vpass);
        }
        next_compiled_pass_refs.push_back(std::move(pass_ref));
    }
    m_passes = std::move(graph_passes);

    std::unordered_set<VulkanPass*> destroyed_passes;
    std::size_t retired_count = 0;
    for (const auto& [_, stale_pass_ref] : reusable_passes) {
        auto stale_pass = std::dynamic_pointer_cast<VulkanPass>(stale_pass_ref);
        if (!stale_pass) continue;
        const auto before_destroy_count = destroyed_passes.size();
        DestroyPassOnce(stale_pass.get(), *m_device, m_rendering_resources, destroyed_passes);
        if (destroyed_passes.size() != before_destroy_count) retired_count++;
    }
    m_compiled_pass_refs = std::move(next_compiled_pass_refs);
    releasePendingSceneResources(scene);

    LOG_INFO("RenderGraphResidencyDiff: reused=%zu new=%zu retired=%zu graph-passes=%zu",
             reused_count,
             new_count,
             retired_count,
             nodes.size());

    m_passes.insert(m_passes.begin(), m_prepass.get());
    m_passes.push_back(m_finpass.get());

    // Dest-draw moved a large share of the compiled graph out of m_passes and into the
    // ComposeDrawWalker. Whatever stays behind still runs after the whole walker, so the two
    // lists together describe the frame's real submission order. Report the residue by identity
    // instead of by layer so a mis-partitioned pass is attributable without per-project logging.
    for (std::size_t residue_index = 0; residue_index < m_passes.size(); ++residue_index) {
        const auto* residue_pass = m_passes[residue_index];
        if (residue_pass == nullptr) continue;
        const auto residue_key = residue_pass->residencyKey();
        LOG_INFO("RenderGraphPassResidue: index=%zu key='%s' phase=%d layer=%d",
                 residue_index,
                 residue_key.empty() ? "(anonymous)" : residue_key.c_str(),
                 static_cast<int>(residue_pass->destDrawPhase()),
                 residue_pass->destDrawLayerId());
    }

    setRenderTargetSize(scene, rg);

    std::size_t reused_refreshed_count = 0;
    std::size_t refreshed_count = 0;
    std::size_t prepared_count = 0;
    std::size_t dependency_prepared_count = 0;
    std::size_t deferred_count = 0;
    std::size_t deferred_waiting_count = 0;
    std::size_t already_prepared_count = 0;
    // CopyPass is a lightweight graph-residency pass, not a heavy shader pass: it registers
    // dynamic copy render targets such as `_rt_default_pingpong` in Scene::renderTargets and binds
    // their TextureCache images. Reused shader passes can legitimately sample those copy targets
    // during the same topology compile, so deferring CopyPass creation lets refreshed passes see a
    // missing input and black out the frame. Prepare copy dependencies up front, then keep the
    // expensive shader/image passes on the deferred residency queue.
    for (size_t pass_index = 0; pass_index < m_passes.size(); ++pass_index) {
        auto* p = m_passes[pass_index];
        if (p == nullptr || p->prepared()) continue;
        if (dynamic_cast<CopyPass*>(p) == nullptr) continue;
        p->prepare(scene, *m_device, m_rendering_resources);
        dependency_prepared_count++;
    }
    PrepareDestDrawPasses(scene);
    for (size_t pass_index = 0; pass_index < m_passes.size(); ++pass_index) {
        auto* p = m_passes[pass_index];
        if (p != nullptr && reused_passes.count(p) != 0 && p->prepared()) {
            p->refreshResources(scene, *m_device, m_rendering_resources);
            reused_refreshed_count++;
        }
        if (refresh_resources_only && p != nullptr && p->prepared()) {
            p->refreshResources(scene, *m_device, m_rendering_resources);
            refreshed_count++;
        }
        if (p != nullptr && !p->prepared()) {
            const bool is_copy_dependency = dynamic_cast<CopyPass*>(p) != nullptr;
            const bool can_defer_runtime_prepare =
                had_resident_graph && !refresh_resources_only && p != m_prepass.get() &&
                p != m_finpass.get() && !is_copy_dependency;
            if (can_defer_runtime_prepare) {
                // Runtime visibility changes should not monopolize the render thread by making
                // every newly-visible layer allocate textures, framebuffers, and staging uploads in
                // one compile call. Queue cold passes and let drawFrame() advance residency a pass
                // at a time while already-prepared scene content keeps rendering.
                if (p->requestDeferredPrepareResources(scene, *m_device) ==
                    DeferredPrepareResourcesState::Waiting) {
                    deferred_waiting_count++;
                }
                m_deferred_prepare_indices.push_back(pass_index);
                deferred_count++;
            } else {
                p->prepare(scene, *m_device, m_rendering_resources);
                prepared_count++;
            }
        } else if (p != nullptr) {
            already_prepared_count++;
        }
    }

    LOG_INFO("RenderGraphCompileSummary: total=%zu reused-refreshed=%zu refreshed=%zu "
             "prepared=%zu dependency-prepared=%zu deferred=%zu already-prepared=%zu mode=%s",
             m_passes.size(),
             reused_refreshed_count,
             refreshed_count,
             prepared_count,
             dependency_prepared_count,
             deferred_count,
             already_prepared_count,
             refresh_resources_only ? "resources" : "topology");
    if (deferred_count > 0) {
        m_device->tex_cache().BeginDeferredGraphActivation();
        LOG_INFO("RenderGraphDeferredPrepareQueued: count=%zu max-passes-per-frame=%zu "
                 "frame-budget=%.2fms resource-waiting=%zu",
                 deferred_count,
                 kDeferredPrepareMaxPassesPerFrame,
                 kDeferredPrepareFrameBudgetMs,
                 deferred_waiting_count);
    }

    // Upload work queued by prepare() is recorded at the start of the next frame command buffer.
    // Avoiding a compile-time queue submit + DeviceWaitIdle is what keeps visibility-driven graph
    // changes from behaving like a scene load.
    m_pass_loaded = true;
};

void VulkanRender::Impl::warmupRenderGraphPipelines(Scene& scene, rg::RenderGraph& rg) {
    if (m_device_faulted) return;
    if (!m_inited || !m_device || !m_rendering_resources.pipeline_cache) return;

    const auto started_at = std::chrono::steady_clock::now();
    auto       nodes      = rg.topologicalOrder();

    setRenderTargetSize(scene, rg);

    std::size_t pipeline_passes = 0;
    std::size_t warmed_passes   = 0;

    for (const auto node_id : nodes) {
        auto pass_ref = rg.getPassShared(node_id);
        auto vpass = std::dynamic_pointer_cast<VulkanPass>(pass_ref);
        if (!vpass) continue;
        pipeline_passes++;
        if (vpass->warmupPipeline(scene, *m_device, m_rendering_resources)) {
            warmed_passes++;
        }
    }

    const auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                std::chrono::steady_clock::now() - started_at)
                                .count();
    LOG_INFO("RenderGraphPipelineWarmup: graph-passes=%zu pipeline-passes=%zu warmed=%zu "
             "cached-states=%zu duration=%.2fms",
             nodes.size(),
             pipeline_passes,
             warmed_passes,
             m_rendering_resources.pipeline_cache->size(),
             static_cast<double>(elapsed_us) / 1000.0);
}
