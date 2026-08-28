#include "SceneWallpaper.hpp"
#include "SceneWallpaperSurface.hpp"

#include "Utils/Logging.h"
#include "Looper/Looper.hpp"

#include "Timer/FrameTimer.hpp"
#include "Utils/FpsCounter.h"
#include "WPSceneParser.hpp"
#include "WPSceneScriptMedia.hpp"
#include "WPTextLayer.hpp"
#include "WPUserProperties.hpp"
#include "Scene/Scene.h"
#include "Particle/ParticleSystem.h"
#include "Interface/IShaderValueUpdater.h"

#include "Fs/VFS.h"
#include "Fs/PhysicalFs.h"
#include "WPPkgFs.hpp"

#include "Audio/SoundManager.h"

#include "RenderGraph/RenderGraph.hpp"

#include "VulkanRender/Msaa.hpp"
#include "VulkanRender/SceneToRenderGraph.hpp"
#include "VulkanRender/VulkanRender.hpp"
#include <algorithm>
#include <cmath>
#include <chrono>
#include <malloc.h>
#include <atomic>
#include <future>
#include <utility>
#include <vector>

using namespace wallpaper;

#define CASE_CMD(cmd) \
    case CMD::CMD_##cmd: handle_##cmd(msg); break;
#define MHANDLER_CMD(cmd) void handle_##cmd(const std::shared_ptr<looper::Message>& msg)
#define MHANDLER_CMD_IMPL(cl, cmd) \
    void impl_##cl::handle_##cmd(const std::shared_ptr<looper::Message>& msg)
#define CALL_MHANDLER_CMD(cmd, msg) handle_##cmd(msg)

namespace
{
constexpr int64_t SCENE_MEDIA_APPLY_SLOW_THRESHOLD_US  = 16000;
constexpr int64_t SCENE_RENDER_GRAPH_SLOW_THRESHOLD_US = 16000;

void TrimHeap() { malloc_trim(0); }

template<typename T>
void AddMsgCmd(looper::Message& msg, T cmd) {
    msg.setInt32("cmd", (int32_t)cmd);
}
template<typename T>
std::shared_ptr<looper::Message> CreateMsgWithCmd(const std::shared_ptr<looper::Handler>& handler,
                                                  T                                       cmd) {
    auto msg = looper::Message::create(0, handler);
    AddMsgCmd(*msg, cmd);
    return msg;
}

bool MediaThumbnailChanged(const std::shared_ptr<WPSceneScriptMediaState>& previous,
                           const std::shared_ptr<WPSceneScriptMediaState>& next) {
    if (previous == next) return false;
    if (! previous || ! next) return true;

    return previous->has_thumbnail != next->has_thumbnail ||
           previous->thumbnail_width != next->thumbnail_width ||
           previous->thumbnail_height != next->thumbnail_height ||
           previous->thumbnail_rgba != next->thumbnail_rgba ||
           previous->previous_thumbnail_width != next->previous_thumbnail_width ||
           previous->previous_thumbnail_height != next->previous_thumbnail_height ||
           previous->previous_thumbnail_rgba != next->previous_thumbnail_rgba;
}

void PopulatePreviousMediaThumbnail(const std::shared_ptr<WPSceneScriptMediaState>& previous,
                                    const std::shared_ptr<WPSceneScriptMediaState>& next) {
    if (! next) return;

    if (! next->has_thumbnail) {
        next->previous_thumbnail_width  = 0;
        next->previous_thumbnail_height = 0;
        next->previous_thumbnail_rgba.clear();
        return;
    }

    const bool previous_has_current_thumbnail =
        previous != nullptr && previous->has_thumbnail && previous->thumbnail_width > 0 &&
        previous->thumbnail_height > 0 && ! previous->thumbnail_rgba.empty();
    const bool current_thumbnail_changed = previous == nullptr ||
                                           previous->thumbnail_width != next->thumbnail_width ||
                                           previous->thumbnail_height != next->thumbnail_height ||
                                           previous->thumbnail_rgba != next->thumbnail_rgba;

    if (previous_has_current_thumbnail && current_thumbnail_changed) {
        // Wallpaper Engine's `$mediaPreviousThumbnail` is derived from the last committed current
        // thumbnail, not from an authored JSON field. Supplying it here gives blend-gradient
        // transition passes real old-cover pixels while keeping JavaScript payloads focused on the
        // active track metadata.
        next->previous_thumbnail_width  = previous->thumbnail_width;
        next->previous_thumbnail_height = previous->thumbnail_height;
        next->previous_thumbnail_rgba   = previous->thumbnail_rgba;
    } else if (previous != nullptr) {
        // Metadata-only updates should keep the old previous texture stable so a later thumbnail
        // change still transitions from the last visible cover instead of an empty transparent
        // placeholder.
        next->previous_thumbnail_width  = previous->previous_thumbnail_width;
        next->previous_thumbnail_height = previous->previous_thumbnail_height;
        next->previous_thumbnail_rgba   = previous->previous_thumbnail_rgba;
    }
}

std::string DescribeUserPropertyForLog(const UserPropertyMap& properties, std::string_view name) {
    // Reused scene switches are sensitive to the exact user-property snapshot
    // that MainHandler holds when PROPERTY_SOURCE triggers parsing.  This compact
    // formatter gives switch logs enough detail to prove whether a tracked combo
    // such as hrbigb2 was staged as the intended string value.
    const auto iter = properties.find(std::string(name));
    if (iter == properties.end()) return "missing";

    if (const auto* string_value = std::get_if<std::string>(&iter->second.value)) {
        return std::string("string:") + *string_value;
    }

    const auto* shader_value = std::get_if<ShaderValue>(&iter->second.value);
    if (shader_value == nullptr) return "unknown";

    std::string description = "shader:[";
    for (size_t index = 0; index < shader_value->size(); index++) {
        if (index != 0) description += ",";
        description += std::to_string((*shader_value)[index]);
        if (index >= 3 && shader_value->size() > 4) {
            description += ",...";
            break;
        }
    }
    description += "]";
    return description;
}

std::string DescribeUserPropertyKeysForLog(const UserPropertyMap& properties) {
    // Live user-property debugging must stay independent from individual wallpapers. A compact key
    // list at the SceneWallpaper boundary proves that the native bridge delivered a payload, while
    // the material-uniform logs prove which registered bindings consumed it.
    std::string description = "[";
    size_t      count       = 0;
    for (const auto& [name, _] : properties) {
        if (count != 0) description += ",";
        description += name;
        count++;
        if (count >= 12 && properties.size() > count) {
            description += ",...";
            break;
        }
    }
    description += "]";
    return description;
}

struct OffscreenExportReconfigureRequest {
    uint32_t                      width { 0 };
    uint32_t                      height { 0 };
    TexTiling                     tiling { TexTiling::LINEAR };
    ExternalFrameExportMode       export_mode { ExternalFrameExportMode::DMA_BUF };
    uint32_t                      export_drm_fourcc { 0 };
    std::vector<uint64_t>         export_drm_modifiers;
    ExternalFrameMemoryPreference memory_preference {
        ExternalFrameMemoryPreference::Default
    };
    std::promise<bool>            result;
};

} // namespace

namespace wallpaper
{
class RenderHandler;

class MainHandler : public looper::Handler {
public:
    enum class CMD
    {
        CMD_LOAD_SCENE,
        CMD_RENDER_READY,
        CMD_SET_PROPERTY,
        CMD_STOP,
        CMD_FIRST_FRAME,
        CMD_NO
    };

public:
    MainHandler();
    virtual ~MainHandler();

    bool init();
    auto renderHandler() const { return m_render_handler; }
    bool inited() const { return m_inited; }
    bool shuttingDown() const { return m_shutting_down.load(); }

public:
    void onMessageReceived(const std::shared_ptr<looper::Message>& msg) override {
        int32_t cmd_int = (int32_t)CMD::CMD_NO;
        if (msg->findInt32("cmd", &cmd_int)) {
            CMD cmd = static_cast<CMD>(cmd_int);
            switch (cmd) {
                CASE_CMD(SET_PROPERTY);
                CASE_CMD(LOAD_SCENE);
                CASE_CMD(RENDER_READY);
                CASE_CMD(STOP);
                CASE_CMD(FIRST_FRAME);
            default: break;
            }
        }
    }

    void        sendRenderReady();
    void        sendFirstFrameOk();
    bool        isGenGraphviz() const { return m_gen_graphviz; }

private:
    void loadScene();

    MHANDLER_CMD(LOAD_SCENE);
    MHANDLER_CMD(RENDER_READY);
    MHANDLER_CMD(SET_PROPERTY);
    MHANDLER_CMD(STOP);
    MHANDLER_CMD(FIRST_FRAME);

private:
    bool m_inited { false };
    bool m_render_ready { false };

    std::string                              m_assets;
    std::string                              m_source;
    std::string                              m_cache_path;
    UserPropertyMap                          m_user_properties;
    std::shared_ptr<WPSceneScriptMediaState> m_media_state;
    std::shared_ptr<std::vector<float>>      m_audio_samples;
    bool                                     m_gen_graphviz { false };
    bool                                     m_reflections_enabled { true };
    int32_t                                  m_volumetrics_quality { 2 };
    int32_t                                  m_shadows_quality { 2 };
    int32_t                                  m_postprocessing_quality { 1 };
    int32_t                                  m_antialiasing_quality { 1 };
    int32_t                                  m_texture_resolution_quality { 0 };

    WPSceneParser                        m_scene_parser;
    std::unique_ptr<audio::SoundManager> m_sound_manager;
    FirstFrameCallback                   m_first_frame_callback;

private:
    std::shared_ptr<looper::Looper> m_main_loop;
    std::shared_ptr<looper::Looper> m_render_loop;
    std::shared_ptr<RenderHandler>  m_render_handler;
    std::atomic<bool>               m_shutting_down { false };
};
// for macro
using impl_MainHandler = MainHandler;

class RenderHandler : public looper::Handler {
public:
    enum class CMD
    {
        CMD_INIT_VULKAN,
        CMD_MOUSE_INPUT,
        CMD_MOUSE_LEFT_BUTTON,
        CMD_SET_SCENE,
        CMD_APPLY_USER_PROPERTIES,
        CMD_APPLY_MEDIA_STATE,
        CMD_APPLY_AUDIO_SAMPLES,
        CMD_SET_FILLMODE,
        CMD_SET_REFLECTIONS,
        CMD_SET_VOLUMETRICS,
        CMD_SET_SHADOWS,
        CMD_SET_POSTPROCESSING,
        CMD_SET_ANTIALIASING,
        CMD_SET_TEXTURE_RESOLUTION,
        CMD_SET_SPEED,
        CMD_SET_OFFSCREEN_RELEASE_CALLBACK,
        CMD_RECONFIGURE_OFFSCREEN_EXPORT,
        CMD_STOP,
        CMD_DRAW,
        CMD_NO
    };
    MainHandler& main_handler;
    RenderHandler(MainHandler& m)
        : main_handler(m), m_render(std::make_unique<vulkan::VulkanRender>()) {}
    virtual ~RenderHandler() {
        frame_timer.Stop();
        if (m_render && m_rg) {
            m_render->clearLastRenderGraph(true);
        }
        m_rg.reset();
        m_scene.reset();
        m_applied_media_state.reset();
        if (m_render) {
            m_render->destroy();
            m_render.reset();
        }
        TrimHeap();
    }

    void onMessageReceived(const std::shared_ptr<looper::Message>& msg) override {
        int32_t cmd_int = (int32_t)CMD::CMD_NO;
        if (msg->findInt32("cmd", &cmd_int)) {
            CMD cmd = static_cast<CMD>(cmd_int);
            switch (cmd) {
                CASE_CMD(DRAW);
                CASE_CMD(STOP);
                CASE_CMD(MOUSE_INPUT);
                CASE_CMD(MOUSE_LEFT_BUTTON);
                CASE_CMD(SET_FILLMODE);
                CASE_CMD(SET_REFLECTIONS);
                CASE_CMD(SET_VOLUMETRICS);
                CASE_CMD(SET_SHADOWS);
                CASE_CMD(SET_POSTPROCESSING);
                CASE_CMD(SET_ANTIALIASING);
                CASE_CMD(SET_TEXTURE_RESOLUTION);
                CASE_CMD(SET_SCENE);
                CASE_CMD(APPLY_USER_PROPERTIES);
                CASE_CMD(APPLY_MEDIA_STATE);
                CASE_CMD(APPLY_AUDIO_SAMPLES);
                CASE_CMD(SET_SPEED);
                CASE_CMD(SET_OFFSCREEN_RELEASE_CALLBACK);
                CASE_CMD(RECONFIGURE_OFFSCREEN_EXPORT);
                CASE_CMD(INIT_VULKAN);
            default: break;
            }
        }
    }

    ExSwapchain* exSwapchain() const { return m_render->exSwapchain(); }

    bool renderInited() const { return m_render->inited(); }

    double textRenderScale() const {
        // Text stays in the authored letter box. Desktop render scale is applied when the
        // composited result is presented, not by enlarging atlases or effect ping-pong.
        return 1.0;
    }

private:
    void RefreshRenderGraphIfNeeded() {
        if (! m_scene || ! m_scene->renderGraphDirty) return;

        const auto started_at                = std::chrono::steady_clock::now();
        const bool requires_topology_rebuild = m_rg == nullptr || m_scene->renderGraphTopologyDirty;
        if (m_rg) {
            if (requires_topology_rebuild) {
                // Runtime visibility changes can alter graph topology, but treating that as a
                // scene switch destroys every prepared pass and recreates hundreds of pipelines on
                // the next frame. Mature game renderers diff the new graph against the resident one:
                // unchanged passes keep their PSO/descriptors, removed hidden branches are retired,
                // and the queued per-layer resource releases drain only after those old passes have
                // dropped their references. VulkanRender::compileRenderGraph owns that handoff.
            } else {
                // Minute-level effect text updates only resize existing offscreen resources.
                // Reusing the compiled graph topology while recreating pass-owned GPU resources
                // avoids the old "full graph rebuild" hitch where every static mesh buffer in the
                // scene was torn down even though the pass list itself had not changed.
                m_render->clearRenderGraphResources();
            }
        }
        if (requires_topology_rebuild) {
            m_rg = sceneToRenderGraph(*m_scene);
        }
        // Custom shader uniforms are written during pass preparation and uploaded before the first
        // draw after a graph rebuild. Keep the framebuffer-relative camera dimensions current
        // before preparing passes, otherwise the first uploaded UBO can still contain the project's
        // native orthographic aspect and only correct itself on the following frame.
        m_render->UpdateCameraFillMode(*m_scene, m_fillmode);
        m_render->compileRenderGraph(*m_scene, *m_rg, ! requires_topology_rebuild);
        m_scene->ClearRenderGraphDirty();
        const auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                    std::chrono::steady_clock::now() - started_at)
                                    .count();
        if (elapsed_us >= SCENE_RENDER_GRAPH_SLOW_THRESHOLD_US) {
            LOG_INFO("SceneWallpaper: render graph rebuild slow duration=%.2fms mode=%s",
                     elapsed_us / 1000.0,
                     requires_topology_rebuild ? "topology" : "resources");
        }
    }

    MHANDLER_CMD(MOUSE_INPUT) {
        float x { 0.5f };
        float y { 0.5f };
        if (! msg->findFloat("x", &x) || ! msg->findFloat("y", &y)) return;

        m_mouse_pos.store(std::array { x, y });
        if (! m_scene) return;

        m_scene->mousePositionNormalized = { x, y };
        m_scene->shaderValueUpdater->MouseInput(x, y);
        m_scene->paritileSys->SetMousePos(x, y);
        if (m_scene->scriptHost) {
            m_scene->scriptHost->HandleCursorMove();
        }
    }
    MHANDLER_CMD(MOUSE_LEFT_BUTTON) {
        bool down { false };
        if (! msg->findBool("down", &down)) return;

        m_cursor_left_down.store(down);
        if (! m_scene) return;

        m_scene->cursorLeftDown = down;
        if (m_scene->scriptHost) {
            m_scene->scriptHost->HandleCursorButton(down);
        }
    }
    MHANDLER_CMD(APPLY_AUDIO_SAMPLES) {
        std::shared_ptr<std::vector<float>> audio_samples;
        if (! msg->findObject("value", &audio_samples) || ! m_scene || ! m_scene->scriptHost ||
            ! audio_samples)
            return;

        m_scene->scriptHost->ApplyAudioSamples(*audio_samples);
    }
    MHANDLER_CMD(STOP) {
        bool stop { false };
        if (msg->findBool("value", &stop)) {
            if (m_render) {
                m_render->setPaused(stop);
            }
            if (stop)
                frame_timer.Stop();
            else
                frame_timer.Run();
        }
    }
    MHANDLER_CMD(APPLY_USER_PROPERTIES) {
        std::shared_ptr<UserPropertyMap> user_properties;
        if (! msg->findObject("value", &user_properties)) return;
        if (! m_scene || ! m_scene->scriptHost) return;

        if (user_properties) {
            m_scene->userProperties = *user_properties;
        } else {
            m_scene->userProperties.clear();
        }
        LOG_INFO("SceneWallpaper: render thread applying live user-properties count=%zu keys=%s",
                 m_scene->userProperties.size(),
                 DescribeUserPropertyKeysForLog(m_scene->userProperties).c_str());
        m_scene->scriptHost->ApplyUserProperties(m_scene->userProperties, false);
    }
    MHANDLER_CMD(APPLY_MEDIA_STATE) {
        std::shared_ptr<WPSceneScriptMediaState> media_state;
        if (! msg->findObject("value", &media_state)) return;
        if (! m_scene || ! m_scene->scriptHost || ! media_state) return;

        const bool thumbnail_changed = MediaThumbnailChanged(m_applied_media_state, media_state);
        const bool render_graph_dirty_before = m_scene->renderGraphDirty;
        const auto started_at                = std::chrono::steady_clock::now();
        m_scene->scriptHost->ApplyMediaState(*media_state);
        const bool render_graph_dirty_after = m_scene->renderGraphDirty;
        const auto elapsed_us               = std::chrono::duration_cast<std::chrono::microseconds>(
                                                  std::chrono::steady_clock::now() - started_at)
                                                  .count();
        m_applied_media_state               = media_state;

        if (thumbnail_changed && ! render_graph_dirty_after) {
            LOG_INFO("SceneScript: media state applied without render graph rebuild");
        } else if (! render_graph_dirty_before && render_graph_dirty_after) {
            LOG_INFO("SceneScript: media state requested render graph rebuild");
        }

        if (elapsed_us >= SCENE_MEDIA_APPLY_SLOW_THRESHOLD_US) {
            LOG_INFO("SceneScript: media state apply slow duration=%.2fms thumbnail-changed=%s "
                     "render-graph-dirty-before=%s after=%s title='%s' artist='%s'",
                     elapsed_us / 1000.0,
                     thumbnail_changed ? "true" : "false",
                     render_graph_dirty_before ? "true" : "false",
                     render_graph_dirty_after ? "true" : "false",
                     media_state->title.c_str(),
                     media_state->artist.c_str());
        }
    }
    MHANDLER_CMD(DRAW) {
        frame_timer.FrameBegin();
        if (m_rg) {
            const double frame_time = frame_timer.IdeaTime() * m_speed;
            m_scene->PassFrameTime(frame_time);
            // Scene camerashake is a view-space translation of the authored camera. Apply it after
            // the clock advances and before scripts/parallax/fill-mode read camera position.
            m_scene->UpdateCameraShake();
            if (m_scene->scriptHost) {
                m_scene->scriptHost->FrameBegin(frame_time);
            }
            // Parallax and puppet state can change the projected size of a text bridge. Advance that
            // resource-affecting state before UpdateCameraFillMode() recomputes backing extents and
            // before a resource-only graph refresh prepares resized framebuffers.
            m_scene->shaderValueUpdater->PrepareFrame();
            // Property animations update camera-layer zoom/origin through the script host before
            // shader uniforms are refreshed. Reapplying the renderer-side fill-mode camera
            // framing here keeps those animated camera values in monitor-relative coordinates
            // instead of letting the animation path snap the shared global camera back to the
            // project's native aspect until a later resize/fill-mode event repairs it.
            m_render->UpdateCameraFillMode(*m_scene, m_fillmode);
            // Imported images must finish their TextureCache allocation decision before prepared
            // passes copy the resulting Vulkan image/view handles. A media transition can change
            // either thumbnail's dimensions, which replaces that key's allocation. Refreshing the
            // pass first makes correctness depend on whichever pass happens to parse the dirty key
            // first and leaves the explicit imported-texture phase capable of replacing it again.
            // Keeping allocation, rebinding, upload recording, and execution in this order gives
            // every consumer one coherent resource generation in the submitted frame.
            m_render->refreshImportedTextures(*m_scene);
            RefreshRenderGraphIfNeeded();

            // LOG_INFO("frame info, fps: %.1f, frametime: %.1f", 1.0f, 1000.0f*m_scene->frameTime);
            // Path B walker is ComposeDrawWalker after command.Begin (0x14018aac0).
            m_scene->shaderValueUpdater->FrameBegin();
            {
                auto pos = m_mouse_pos.load();
                m_scene->paritileSys->SetMousePos(pos[0], pos[1]);
            }
            if (m_scene->paritileSys->NeedsAudioSpectrum()) {
                // Particle responses consume the same normalized 16-band snapshot as SceneScript
                // and stock audio shaders. A missing source is cleared to zero-strength response;
                // mode zero remains the explicit authored bypass.
                std::vector<float> audio_left, audio_right, audio_average;
                if (m_scene->scriptHost &&
                    m_scene->scriptHost->GetAudioSpectrum(
                        kParticleAudioBandCount, &audio_left, &audio_right, &audio_average)) {
                    m_scene->paritileSys->SetAudioSpectrum(audio_left, audio_right);
                } else {
                    m_scene->paritileSys->ClearAudioSpectrum();
                }
            }
            m_scene->paritileSys->Emitt();

            m_render->drawFrame(*m_scene);

            m_scene->shaderValueUpdater->FrameEnd();
            // fps_counter.RegisterFrame();

            if (! m_scene->first_frame_ok) {
                m_scene->first_frame_ok = true;
                main_handler.sendFirstFrameOk();
            }
        }
        frame_timer.FrameEnd();
    }
    MHANDLER_CMD(SET_FILLMODE) {
        int32_t value;
        if (msg->findInt32("value", &value)) {
            m_fillmode = (FillMode)value;
            if (m_scene && renderInited()) {
                m_render->UpdateCameraFillMode(*m_scene, m_fillmode);
            }
        }
    }
    MHANDLER_CMD(SET_REFLECTIONS) {
        bool enabled { true };
        if (! msg->findBool("value", &enabled) || ! m_scene) return;
        if (m_scene->reflectionsEnabled == enabled) return;
        m_scene->reflectionsEnabled = enabled;
        LOG_INFO("SceneWallpaper: reflections %s (live pass gate, RT kept)",
                 enabled ? "enabled" : "disabled");
    }
    MHANDLER_CMD(SET_VOLUMETRICS) {
        int32_t quality { 2 };
        if (! msg->findInt32("value", &quality) || ! m_scene) return;
        quality = std::clamp(quality, 0, 4);
        if (m_scene->volumetrics.quality == quality &&
            m_scene->volumetrics.built_quality == quality)
            return;
        m_scene->volumetrics.quality = quality;
        if (m_scene->vfs) {
            ConfigureSceneVolumetrics(*m_scene, *m_scene->vfs);
        }
        m_scene->MarkRenderGraphTopologyDirty();
        LOG_INFO("SceneWallpaper: volumetrics quality=%d (rebuild graph)", quality);
    }
    MHANDLER_CMD(SET_SHADOWS) {
        int32_t quality { 2 };
        if (! msg->findInt32("value", &quality) || ! m_scene) return;
        quality = std::clamp(quality, 0, 4);
        if (m_scene->shadows.quality == quality && m_scene->shadows.built_quality == quality &&
            m_scene->volumetrics.built_quality == m_scene->volumetrics.quality)
            return;
        m_scene->shadows.quality = quality;
        if (m_scene->vfs) {
            ConfigureSceneVolumetrics(*m_scene, *m_scene->vfs);
        }
        m_scene->MarkRenderGraphTopologyDirty();
        LOG_INFO("SceneWallpaper: shadows quality=%d (rebuild graph)", quality);
    }
    MHANDLER_CMD(SET_POSTPROCESSING) {
        int32_t quality { 1 };
        if (! msg->findInt32("value", &quality) || ! m_scene) return;
        quality = std::clamp(quality, 0, 3);
        if (m_scene->bloom.quality == quality && m_scene->bloom.built_quality == quality) return;
        m_scene->bloom.quality = quality;
        if (m_scene->vfs) {
            ConfigureSceneBloom(*m_scene, *m_scene->vfs);
        }
        m_scene->MarkRenderGraphTopologyDirty();
        LOG_INFO("SceneWallpaper: postprocessing quality=%d (rebuild graph)", quality);
    }
    MHANDLER_CMD(SET_ANTIALIASING) {
        int32_t quality { 1 };
        if (! msg->findInt32("value", &quality) || ! m_scene) return;
        quality = std::clamp(quality, 0, 3);
        const int effective = m_scene->has3dModels ? quality : 0;
        if (m_scene->msaa.quality == quality && m_scene->msaa.built_quality == effective) return;
        m_scene->msaa.quality        = quality;
        m_scene->msaa.device_samples = 0;
        ConfigureSceneMsaa(*m_scene);
        m_scene->MarkRenderGraphResourcesDirty();
        LOG_INFO("SceneWallpaper: antialiasing requested=%d effective=%d samples=%d has3d=%s "
                 "(refresh resources)",
                 quality,
                 m_scene->EffectiveMsaaQuality(),
                 m_scene->MsaaSampleCount(),
                 m_scene->has3dModels ? "true" : "false");
    }
    MHANDLER_CMD(SET_TEXTURE_RESOLUTION) {
        int32_t quality { 0 };
        if (! msg->findInt32("value", &quality) || ! m_scene) return;
        quality = std::clamp(quality, 0, 2);
        if (m_scene->textureResolution.quality == quality) return;
        m_scene->textureResolution.quality = quality;
        m_scene->ApplyTextureResolutionForCurrentOutput();
    }
    MHANDLER_CMD(SET_SCENE) {
        if (msg->findObject("scene", &m_scene)) {
            std::shared_ptr<WPSceneScriptMediaState> media_state;
            std::shared_ptr<std::vector<float>>      audio_samples;
            msg->findObject("media_state", &media_state);
            msg->findObject("audio_samples", &audio_samples);

            // Scene text arrives with authoring geometry already resolved by WPSceneParser. Keep
            // that authored density for later rerasterization; desktop render scale must not
            // rebuild glyphs or enlarge effect ping-pong.
            m_scene->textRenderScale = 1.0;
            m_scene->scriptHost = std::make_unique<WPSceneScriptHost>(m_scene.get());
            for (const auto& registration : m_scene->bindingRegistrations) {
                m_scene->scriptHost->RegisterPropertyBinding(registration);
            }
            for (const auto& registration : m_scene->propertyAnimationRegistrations) {
                m_scene->scriptHost->RegisterPropertyAnimation(registration);
            }
            for (const auto& registration : m_scene->scriptRegistrations) {
                m_scene->scriptHost->RegisterPropertyScript(registration);
            }
            m_scene->scriptHost->Initialize();
            if (media_state) {
                m_scene->scriptHost->ApplyMediaState(*media_state);
                m_applied_media_state = std::move(media_state);
            }
            if (audio_samples) {
                m_scene->scriptHost->ApplyAudioSamples(*audio_samples);
            }
            m_scene->scriptHost->MaterializeDeferredRuntimeLayersForResidency();

            if (m_rg) m_render->clearLastRenderGraph(true);
            {
                auto warmup_rg = sceneToPipelineWarmupRenderGraph(*m_scene);
                m_render->warmupRenderGraphPipelines(*m_scene, *warmup_rg);
            }
            m_rg = sceneToRenderGraph(*m_scene);

            if (main_handler.isGenGraphviz()) m_rg->ToGraphviz("graph.dot");
            // The initial render graph compile performs the first uniform write and immediately
            // submits the dynamic-buffer upload. Apply fill mode first so scene switches do not
            // show a one-frame native-aspect projection before draw-time uniform refreshes catch
            // up.
            m_render->UpdateCameraFillMode(*m_scene, m_fillmode);
            m_render->compileRenderGraph(*m_scene, *m_rg, false);
            m_scene->ClearRenderGraphDirty();

            auto pos                         = m_mouse_pos.load();
            m_scene->mousePositionNormalized = pos;
            m_scene->cursorLeftDown          = m_cursor_left_down.load();
            m_scene->shaderValueUpdater->MouseInput(pos[0], pos[1]);
            m_scene->paritileSys->SetMousePos(pos[0], pos[1]);
            m_scene->scriptHost->HandleCursorMove();
        }
    }
    MHANDLER_CMD(SET_SPEED) { msg->findFloat("value", &m_speed); }
    MHANDLER_CMD(SET_OFFSCREEN_RELEASE_CALLBACK) {
        std::shared_ptr<vulkan::OffscreenFrameReleaseCallback> callback;
        if (msg->findObject("value", &callback) && callback) {
            m_render->setOffscreenFrameReleaseCallback(*callback);
        } else {
            m_render->setOffscreenFrameReleaseCallback({});
        }
    }
    MHANDLER_CMD(RECONFIGURE_OFFSCREEN_EXPORT) {
        std::shared_ptr<OffscreenExportReconfigureRequest> request;
        if (!msg->findObject("request", &request) || !request) return;

        bool ok = false;
        try {
            ok = m_render->reconfigureOffscreenExport(request->width,
                                                      request->height,
                                                      request->tiling,
                                                      request->export_mode,
                                                      request->export_drm_fourcc,
                                                      request->export_drm_modifiers,
                                                      request->memory_preference);
        } catch (...) {
            ok = false;
        }
        request->result.set_value(ok);
    }
    MHANDLER_CMD(INIT_VULKAN) {
        std::shared_ptr<RenderInitInfo> info;
        if (msg->findObject("info", &info)) {
            m_render_scale = std::max(1.0, info->render_scale);
            m_render->init(*info);
            if (m_render->inited())
                main_handler.sendRenderReady();
        }
    }

public:
    FrameTimer frame_timer;
    FpsCounter fps_counter;

private:
    std::shared_ptr<Scene>                   m_scene { nullptr };
    std::shared_ptr<WPSceneScriptMediaState> m_applied_media_state;
    float                                    m_speed { 1.0f };
    double                                   m_render_scale { 1.0 };
    std::atomic<bool>                        m_cursor_left_down { false };

    std::unique_ptr<vulkan::VulkanRender> m_render;
    std::unique_ptr<rg::RenderGraph>      m_rg { nullptr };

    FillMode m_fillmode { FillMode::ASPECTCROP };

    std::atomic<std::array<float, 2>> m_mouse_pos { std::array { 0.5f, 0.5f } };
};
} // namespace wallpaper

SceneWallpaper::SceneWallpaper(): m_main_handler(std::make_shared<MainHandler>()) {}
SceneWallpaper::~SceneWallpaper() = default;

bool SceneWallpaper::inited() const { return m_main_handler->inited(); }

bool SceneWallpaper::init() { return m_main_handler->init(); }

void SceneWallpaper::initVulkan(const RenderInitInfo& info) {
    m_offscreen                             = info.offscreen;
    std::shared_ptr<RenderInitInfo> sp_info = std::make_shared<RenderInitInfo>(info);
    auto                            msg =
        CreateMsgWithCmd(m_main_handler->renderHandler(), RenderHandler::CMD::CMD_INIT_VULKAN);
    msg->setObject("info", sp_info);
    msg->post();
}

void SceneWallpaper::setOffscreenFrameReleaseCallback(
    vulkan::OffscreenFrameReleaseCallback callback) {
    auto msg = CreateMsgWithCmd(m_main_handler->renderHandler(),
                                RenderHandler::CMD::CMD_SET_OFFSCREEN_RELEASE_CALLBACK);
    msg->setObject("value",
                   std::make_shared<vulkan::OffscreenFrameReleaseCallback>(
                       std::move(callback)));
    msg->post();
}

bool SceneWallpaper::reconfigureOffscreenExport(
    uint32_t width,
    uint32_t height,
    TexTiling tiling,
    ExternalFrameExportMode export_mode,
    uint32_t export_drm_fourcc,
    const std::vector<uint64_t>& export_drm_modifiers,
    ExternalFrameMemoryPreference memory_preference) {
    auto request = std::make_shared<OffscreenExportReconfigureRequest>();
    request->width = width;
    request->height = height;
    request->tiling = tiling;
    request->export_mode = export_mode;
    request->export_drm_fourcc = export_drm_fourcc;
    request->export_drm_modifiers = export_drm_modifiers;
    request->memory_preference = memory_preference;

    auto future = request->result.get_future();
    auto msg = CreateMsgWithCmd(m_main_handler->renderHandler(),
                                RenderHandler::CMD::CMD_RECONFIGURE_OFFSCREEN_EXPORT);
    msg->setObject("request", request);
    if (msg->post() != looper::status_t::OK) return false;

    /*
     * Reconfiguration is part of the synchronous BIND_BUFFERS path.  Waiting
     * here keeps renderer-side allocation failures in the existing negotiation
     * flow, while the actual Vulkan image allocation still runs on the render
     * thread that owns TextureCache and the renderer Device.
     */
    if (future.wait_for(std::chrono::seconds(2)) != std::future_status::ready) {
        LOG_ERROR("SceneWallpaper: timed out waiting for offscreen export reconfigure");
        return false;
    }
    return future.get();
}

void SceneWallpaper::play() {
    auto msg = CreateMsgWithCmd(m_main_handler, MainHandler::CMD::CMD_STOP);
    msg->setBool("value", false);
    msg->post();
}
void SceneWallpaper::pause() {
    auto msg = CreateMsgWithCmd(m_main_handler, MainHandler::CMD::CMD_STOP);
    msg->setBool("value", true);
    msg->post();
}
void SceneWallpaper::requestFrame() {
    m_main_handler->renderHandler()->frame_timer.RequestFrame();
}

void SceneWallpaper::mouseInput(double x, double y) {
    auto msg =
        CreateMsgWithCmd(m_main_handler->renderHandler(), RenderHandler::CMD::CMD_MOUSE_INPUT);
    msg->setFloat("x", (float)x);
    msg->setFloat("y", (float)y);
    msg->post();
}

void SceneWallpaper::mouseLeftButton(bool down) {
    auto msg = CreateMsgWithCmd(m_main_handler->renderHandler(),
                                RenderHandler::CMD::CMD_MOUSE_LEFT_BUTTON);
    msg->setBool("down", down);
    msg->post();
}

#define BASIC_TYPE(NAME, TYPENAME)                                                       \
    void SceneWallpaper::setProperty##NAME(std::string_view name, TYPENAME value) {      \
        auto msg = CreateMsgWithCmd(m_main_handler, MainHandler::CMD::CMD_SET_PROPERTY); \
        msg->setString("property", std::string(name));                                   \
        msg->set##NAME("value", value);                                                  \
        msg->post();                                                                     \
    }

BASIC_TYPE(Bool, bool);
BASIC_TYPE(Int32, int32_t);
BASIC_TYPE(Float, float);
BASIC_TYPE(String, std::string);
BASIC_TYPE(Object, std::shared_ptr<void>);

ExSwapchain* SceneWallpaper::exSwapchain() const {
    return m_main_handler->renderHandler()->exSwapchain();
}

MHANDLER_CMD_IMPL(MainHandler, LOAD_SCENE) {
    if (m_render_ready) {
        loadScene();
    } else {
        LOG_INFO("SceneWallpaper: deferring scene load until renderer ready source=%s assets=%s",
                 m_source.empty() ? "missing" : "ready",
                 m_assets.empty() ? "missing" : "ready");
    }
}

MHANDLER_CMD_IMPL(MainHandler, RENDER_READY) {
    m_render_ready = true;
    LOG_INFO("SceneWallpaper: renderer ready source=%s assets=%s",
             m_source.empty() ? "missing" : "ready",
             m_assets.empty() ? "missing" : "ready");
    loadScene();
}

MHANDLER_CMD_IMPL(MainHandler, SET_PROPERTY) {
    std::string property;
    if (msg->findString("property", &property)) {
        if (property == PROPERTY_SOURCE) {
            msg->findString("value", &m_source);
            LOG_INFO("source: %s load-user-properties=%zu hrbigb2=%s",
                     m_source.c_str(),
                     m_user_properties.size(),
                     DescribeUserPropertyForLog(m_user_properties, "hrbigb2").c_str());
            CALL_MHANDLER_CMD(LOAD_SCENE, msg);
        } else if (property == PROPERTY_ASSETS) {
            // KDE keeps one SceneWallpaper alive and updates assets before source.
            // Treat assets as load context only; source is the authoritative reload
            // trigger so a project switch cannot parse old source with new assets.
            msg->findString("value", &m_assets);
        } else if (property == PROPERTY_FPS) {
            int32_t fps { 15 };
            msg->findInt32("value", &fps);
            if (fps >= 5) {
                m_render_handler->frame_timer.SetRequiredFps((uint8_t)fps);
            }
        } else if (property == PROPERTY_REFLECTIONS) {
            bool enabled { true };
            if (msg->findBool("value", &enabled)) {
                m_reflections_enabled = enabled;
                auto nmsg =
                    CreateMsgWithCmd(m_render_handler, RenderHandler::CMD::CMD_SET_REFLECTIONS);
                nmsg->setBool("value", enabled);
                nmsg->post();
            }
        } else if (property == PROPERTY_VOLUMETRICS) {
            int32_t quality { 2 };
            if (msg->findInt32("value", &quality)) {
                m_volumetrics_quality = std::clamp(quality, 0, 4);
                auto nmsg =
                    CreateMsgWithCmd(m_render_handler, RenderHandler::CMD::CMD_SET_VOLUMETRICS);
                nmsg->setInt32("value", m_volumetrics_quality);
                nmsg->post();
            }
        } else if (property == PROPERTY_SHADOWS) {
            int32_t quality { 2 };
            if (msg->findInt32("value", &quality)) {
                m_shadows_quality = std::clamp(quality, 0, 4);
                auto nmsg =
                    CreateMsgWithCmd(m_render_handler, RenderHandler::CMD::CMD_SET_SHADOWS);
                nmsg->setInt32("value", m_shadows_quality);
                nmsg->post();
            }
        } else if (property == PROPERTY_POSTPROCESSING) {
            int32_t quality { 1 };
            if (msg->findInt32("value", &quality)) {
                m_postprocessing_quality = std::clamp(quality, 0, 3);
                auto nmsg =
                    CreateMsgWithCmd(m_render_handler, RenderHandler::CMD::CMD_SET_POSTPROCESSING);
                nmsg->setInt32("value", m_postprocessing_quality);
                nmsg->post();
            }
        } else if (property == PROPERTY_ANTIALIASING) {
            int32_t quality { 1 };
            if (msg->findInt32("value", &quality)) {
                m_antialiasing_quality = std::clamp(quality, 0, 3);
                auto nmsg =
                    CreateMsgWithCmd(m_render_handler, RenderHandler::CMD::CMD_SET_ANTIALIASING);
                nmsg->setInt32("value", m_antialiasing_quality);
                nmsg->post();
            }
        } else if (property == PROPERTY_TEXTURE_RESOLUTION) {
            int32_t quality { 0 };
            if (msg->findInt32("value", &quality)) {
                m_texture_resolution_quality = std::clamp(quality, 0, 2);
                auto nmsg = CreateMsgWithCmd(m_render_handler,
                                             RenderHandler::CMD::CMD_SET_TEXTURE_RESOLUTION);
                nmsg->setInt32("value", m_texture_resolution_quality);
                nmsg->post();
            }
        } else if (property == PROPERTY_FILLMODE) {
            int32_t value;
            if (msg->findInt32("value", &value)) {
                auto nmsg =
                    CreateMsgWithCmd(m_render_handler, RenderHandler::CMD::CMD_SET_FILLMODE);
                nmsg->setInt32("value", value);
                nmsg->post();
            }
        } else if (property == PROPERTY_GRAPHIVZ) {
            msg->findBool("value", &m_gen_graphviz);
        } else if (property == PROPERTY_MUTED) {
            bool muted { false };
            msg->findBool("value", &muted);
            m_sound_manager->SetMuted(muted);
        } else if (property == PROPERTY_VOLUME) {
            float volume { 1.0f };
            msg->findFloat("value", &volume);
            m_sound_manager->SetVolume(volume);
        } else if (property == PROPERTY_CACHE_PATH) {
            std::string path;
            msg->findString("value", &path);
            m_cache_path = path;
        } else if (property == PROPERTY_FIRST_FRAME_CALLBACK) {
            std::shared_ptr<FirstFrameCallback> cb;
            msg->findObject("value", &cb);
            m_first_frame_callback = *cb;
        } else if (property == PROPERTY_LOAD_USER_PROPERTIES) {
            // Load-time user properties are committed to MainHandler only.  The
            // next PROPERTY_SOURCE will parse the new scene with these values,
            // while the outgoing render scene avoids a live ApplyUserProperties
            // pass that can create a visible switch-only intermediate state.
            std::shared_ptr<UserPropertyMap> user_properties;
            if (msg->findObject("value", &user_properties) && user_properties) {
                m_user_properties = *user_properties;
            } else {
                m_user_properties.clear();
            }
            LOG_INFO("SceneWallpaper: staged load user-properties count=%zu hrbigb2=%s",
                     m_user_properties.size(),
                     DescribeUserPropertyForLog(m_user_properties, "hrbigb2").c_str());
        } else if (property == PROPERTY_USER_PROPERTIES) {
            std::shared_ptr<UserPropertyMap> user_properties;
            if (msg->findObject("value", &user_properties) && user_properties) {
                m_user_properties = *user_properties;
            } else {
                m_user_properties.clear();
            }
            LOG_INFO("SceneWallpaper: live PROPERTY_USER_PROPERTIES count=%zu keys=%s",
                     m_user_properties.size(),
                     DescribeUserPropertyKeysForLog(m_user_properties).c_str());
            auto nmsg =
                CreateMsgWithCmd(m_render_handler, RenderHandler::CMD::CMD_APPLY_USER_PROPERTIES);
            nmsg->setObject("value", std::make_shared<UserPropertyMap>(m_user_properties));
            nmsg->post();
        } else if (property == PROPERTY_MEDIA_STATE) {
            std::shared_ptr<WPSceneScriptMediaState> next_media_state;
            msg->findObject("value", &next_media_state);
            PopulatePreviousMediaThumbnail(m_media_state, next_media_state);
            m_media_state = std::move(next_media_state);
            auto nmsg =
                CreateMsgWithCmd(m_render_handler, RenderHandler::CMD::CMD_APPLY_MEDIA_STATE);
            nmsg->setObject("value", m_media_state);
            nmsg->post();
        } else if (property == PROPERTY_AUDIO_SAMPLES) {
            msg->findObject("value", &m_audio_samples);
            auto nmsg =
                CreateMsgWithCmd(m_render_handler, RenderHandler::CMD::CMD_APPLY_AUDIO_SAMPLES);
            nmsg->setObject("value", m_audio_samples);
            nmsg->post();
        } else if (property == PROPERTY_SPEED) {
            float speed { 1.0f };
            if (msg->findFloat("value", &speed)) {
                auto nmsg = CreateMsgWithCmd(m_render_handler, RenderHandler::CMD::CMD_SET_SPEED);
                nmsg->setFloat("value", speed);
                nmsg->post();
            }
        }
    }
}

MHANDLER_CMD_IMPL(MainHandler, STOP) {
    bool stop { false };
    if (msg->findBool("value", &stop)) {
        if (stop) {
            m_sound_manager->Pause();
        } else {
            m_sound_manager->Play();
        }

        auto msg_r = CreateMsgWithCmd(m_render_handler, RenderHandler::CMD::CMD_STOP);
        msg_r->setBool("value", stop);
        msg_r->post();
    }
}

MHANDLER_CMD_IMPL(MainHandler, FIRST_FRAME) {
    if (m_first_frame_callback) m_first_frame_callback();
}

void MainHandler::loadScene() {
    if (m_source.empty() || m_assets.empty()) return;

    LOG_INFO("loading scene: %s user-properties=%zu hrbigb2=%s",
             m_source.c_str(),
             m_user_properties.size(),
             DescribeUserPropertyForLog(m_user_properties, "hrbigb2").c_str());

    if (! m_sound_manager->IsInited()) {
        m_sound_manager->Init();
        m_sound_manager->Play();
    } else {
        m_sound_manager->UnMountAll();
    }

    std::shared_ptr<Scene> scene { nullptr };

    // mount assets dir
    std::unique_ptr<fs::VFS> pVfs = std::make_unique<fs::VFS>();
    auto&                    vfs  = *pVfs;
    if (! vfs.IsMounted("assets")) {
        bool sus = vfs.Mount("/assets", fs::CreatePhysicalFs(m_assets), "assets");
        if (! sus) {
            LOG_ERROR("Mount assets dir failed");
            return;
        }
    }
    std::filesystem::path pkgPath_fs { m_source };
    pkgPath_fs.replace_extension("pkg");
    std::string pkgPath  = pkgPath_fs.native();
    std::string pkgEntry = pkgPath_fs.filename().replace_extension("json").native();
    std::string pkgDir   = pkgPath_fs.parent_path().native();
    std::string scene_id = pkgPath_fs.parent_path().filename().native();

    // Steam workshop scene projects usually ship as an unpacked directory while packaged
    // wallpapers use scene.pkg. Check the package path first so the expected directory fallback
    // does not make the low-level binary stream log a scary "can't open" error.
    const bool has_pkg_file = std::filesystem::is_regular_file(pkgPath_fs);
    if (! has_pkg_file || ! vfs.Mount("/assets", fs::WPPkgFs::CreatePkgFs(pkgPath))) {
        LOG_INFO("load pkg file %s %s, fallback to use dir",
                 pkgPath.c_str(),
                 has_pkg_file ? "failed" : "missing");
        // load pkg dir
        if (! vfs.Mount("/assets", fs::CreatePhysicalFs(pkgDir))) {
            LOG_ERROR("can't load pkg directory: %s", pkgDir.c_str());
            return;
        }
    }
    if (! m_cache_path.empty()) {
        if (! vfs.Mount("/cache", fs::CreatePhysicalFs(m_cache_path, true), "cache")) {
            LOG_ERROR("can't load cache folder: %s", m_cache_path.c_str());
        } else {
            LOG_INFO("cache folder: %s", m_cache_path.c_str());
        }
    }

    {
        std::string       scene_src;
        const std::string base { "/assets/" };
        {
            std::string scenePath = base + pkgEntry;
            if (vfs.Contains(scenePath)) {
                auto f = vfs.Open(scenePath);
                if (f) scene_src = f->ReadAllStr();
            }
        }
        if (scene_src.empty()) {
            LOG_ERROR("Not supported scene type");
            return;
        }
        scene = m_scene_parser.Parse(scene_id,
                                     scene_src,
                                     vfs,
                                     *m_sound_manager,
                                     &m_user_properties,
                                     m_render_handler->textRenderScale());
        scene->reflectionsEnabled = m_reflections_enabled;
        scene->volumetrics.quality = m_volumetrics_quality;
        scene->shadows.quality     = m_shadows_quality;
        scene->bloom.quality       = m_postprocessing_quality;
        scene->msaa.quality        = m_antialiasing_quality;
        scene->textureResolution.quality = m_texture_resolution_quality;
        ConfigureSceneMsaa(*scene);
        LOG_INFO("SceneWallpaper: antialiasing requested=%d effective=%d samples=%d has3d=%s",
                 m_antialiasing_quality,
                 scene->EffectiveMsaaQuality(),
                 scene->MsaaSampleCount(),
                 scene->has3dModels ? "true" : "false");
        scene->vfs.swap(pVfs);
        if (scene->vfs && (scene->volumetrics.built_quality != m_volumetrics_quality ||
                           scene->shadows.built_quality != m_shadows_quality)) {
            ConfigureSceneVolumetrics(*scene, *scene->vfs);
        }
        if (scene->vfs && scene->bloom.built_quality != m_postprocessing_quality) {
            ConfigureSceneBloom(*scene, *scene->vfs);
        }
    }

    {
        auto msg = CreateMsgWithCmd(m_render_handler, RenderHandler::CMD::CMD_SET_SCENE);
        msg->setObject("scene", scene);
        msg->setObject("media_state", m_media_state);
        msg->setObject("audio_samples", m_audio_samples);
        msg->post();
    }

    // The timer owns the single outstanding draw gate. Startup and producer-triggered requests use
    // that same path so scene parsing or initial Vulkan setup cannot accumulate dozens of stale
    // CMD_DRAW messages in front of live pointer input.
    m_render_handler->frame_timer.RequestFrame();
}
void MainHandler::sendRenderReady() {
    auto self = weak_from_this().lock();
    if (! self) return;

    auto msg = CreateMsgWithCmd(self, MainHandler::CMD::CMD_RENDER_READY);
    msg->post();
}
void MainHandler::sendFirstFrameOk() {
    auto self = weak_from_this().lock();
    if (! self) return;

    auto msg = CreateMsgWithCmd(self, MainHandler::CMD::CMD_FIRST_FRAME);
    msg->post();
}

bool MainHandler::init() {
    if (m_inited) return true;
    m_main_loop->setName("main");
    m_render_loop->setName("render");

    m_main_loop->start();
    m_render_loop->start();

    m_main_loop->registerHandler(shared_from_this());
    m_render_loop->registerHandler(m_render_handler);

    {
        auto& frameTimer = m_render_handler->frame_timer;
        std::weak_ptr<RenderHandler> render_handler = m_render_handler;
        frameTimer.SetCallback([render_handler]() {
            if (auto handler = render_handler.lock()) {
                auto msg = CreateMsgWithCmd(handler, RenderHandler::CMD::CMD_DRAW);
                msg->post();
            }
        });
        frameTimer.SetRequiredFps(15);
        frameTimer.Run();
    }

    m_inited = true;
    return true;
}
MainHandler::MainHandler()
    : m_sound_manager(std::make_unique<audio::SoundManager>()),
      m_main_loop(std::make_shared<looper::Looper>()),
      m_render_loop(std::make_shared<looper::Looper>()),
      m_render_handler(std::make_shared<RenderHandler>(*this)) {}

MainHandler::~MainHandler() {
    m_shutting_down.store(true);

    if (m_render_handler) {
        m_render_handler->frame_timer.Stop();
    }

    if (m_render_loop) {
        if (m_render_handler && m_render_handler->id() != looper::Handler::INVALID_HANDLER_ID) {
            m_render_loop->unregisterHandler(m_render_handler->id());
        }
        m_render_loop->stop();
    }

    if (m_main_loop) {
        if (id() != looper::Handler::INVALID_HANDLER_ID) {
            m_main_loop->unregisterHandler(id());
        }
        m_main_loop->stop();
    }

    m_media_state.reset();
    m_audio_samples.reset();
    m_user_properties.clear();
    m_sound_manager.reset();
    m_render_handler.reset();
    m_render_loop.reset();
    m_main_loop.reset();

    TrimHeap();
}
