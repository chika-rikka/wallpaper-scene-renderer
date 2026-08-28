#pragma once
#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <future>
#include <mutex>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "SceneTexture.h"
#include "SceneRenderTarget.h"
#include "SceneNode.h"
#include "SceneObject.h"
#include "SceneLight.hpp"
#include "WPSceneScriptHost.hpp"
#include "WPTextLayer.hpp"
#include "WPUserProperties.hpp"

#include "Core/NoCopyMove.hpp"

namespace wallpaper
{
class ParticleSystem;
class ParticleSubSystem;
class IShaderValueUpdater;
class IImageParser;
struct Image;
struct SceneImageEffect;
namespace audio
{
class SoundManager;
}

namespace fs
{
class VFS;
}
class Scene : NoCopy, NoMove {
public:
    struct CameraPathKeyframe {
        double timestamp { 0.0 };
        std::array<float, 3> eye { 0.0f, 0.0f, 1.0f };
        std::array<float, 3> center { 0.0f, 0.0f, 0.0f };
        std::array<float, 3> up { 0.0f, 1.0f, 0.0f };
    };

    struct CameraPathSegment {
        // Wallpaper Engine 3D camera paths are scene-level timelines, not 2D camera layers. The
        // parser stores them here but binds playback to a model-only camera name, so ordinary 2D
        // scenes never have their `global` or `global_perspective` camera semantics changed.
        std::string name;
        double      duration { 0.0 };
        std::vector<CameraPathKeyframe> keyframes;
    };

    struct ImageLayerRuntimeState {
        std::array<float, 2> size { 0.0f, 0.0f };
        std::string          alignment { "center" };
    };

    struct CameraLayerRuntimeState {
        // Wallpaper Engine camera layers are represented in scene.json as transform-only objects
        // with camera-specific properties. Keep the authored values beside the render node so
        // scripts and keyframe animations can round-trip the WE-facing origin/zoom values while
        // Hanabi stores the attached SceneCamera node in renderer coordinates.
        std::string                camera_name { "global" };
        std::shared_ptr<SceneNode> node;
        std::array<float, 3>       origin { 0.0f, 0.0f, 0.0f };
        std::array<float, 3>       angles { 0.0f, 0.0f, 0.0f };
        double                     zoom { 1.0 };
        float                      fov { 50.0f };
    };

    struct LayerParentBinding {
        int32_t     parent_id { 0 };
        std::string attachment;
    };

    struct BloomSettings {
        // Bloom is a scene option in Wallpaper Engine. It must run after the whole layer tree has
        // produced `_rt_default`, so the render graph owns it as a dedicated global post-process
        // node instead of attaching it to any authored layer.
        bool                       enabled { false };
        float                      strength { 0.0f };
        float                      threshold { 1.0f };
        std::array<float, 3>       tint { 1.0f, 1.0f, 1.0f };
        // Authored HDR bloom metadata. Ultra/displayhdr host quality selects the HDR chain
        // when this flag is set; enabled quality stays on the LDR chain.
        bool                       hdr { false };
        float                      hdrStrength { 0.0f };
        float                      hdrThreshold { 1.0f };
        float                      hdrScatter { 1.0f };
        float                      hdrFeather { 0.0f };
        int32_t                    hdrIterations { 0 };
        std::shared_ptr<SceneNode> node;
        // Scene Bloom is a small post-process chain in Wallpaper Engine assets, not a single
        // shader. Keep the legacy `node` as the runtime-uniform anchor, and store the ordered
        // pass nodes plus their output targets so SceneToRenderGraph can reproduce the chain.
        std::vector<std::shared_ptr<SceneNode>> nodes;
        std::vector<std::string>                outputs;
        // Host post-processing quality: 0 disabled, 1 enabled, 2 ultra, 3 displayhdr.
        int                        quality { 1 };
        int                        built_quality { -1 };
    };

    enum class ParsedImageRequestState
    {
        Ready,
        Pending,
        Failed,
    };

    struct ParsedImageRequest {
        ParsedImageRequestState state { ParsedImageRequestState::Failed };
        std::shared_ptr<Image>  image;
    };

    Scene();
    ~Scene();

    void                SetLayerParentBinding(int32_t layer_id, int32_t parent_id,
                                              std::string attachment = {});
    LayerParentBinding  GetLayerParentBinding(int32_t layer_id) const;
    void                ClearLayerParentBinding(int32_t layer_id);
    std::vector<int32_t> GetLayerChildren(int32_t layer_id) const;

    SceneObject*       FindSceneObject(int32_t layer_id);
    const SceneObject* FindSceneObject(int32_t layer_id) const;
    SceneObject*       FindSceneObjectForNode(const SceneNode* node);
    const SceneObject* FindSceneObjectForNode(const SceneNode* node) const;
    SceneObject&       EnsureSceneObject(int32_t layer_id);
    void               BindSceneObjectParent(int32_t layer_id, int32_t parent_id,
                                             std::string_view attachment);

    // Official engine dest [engine+0x38] (DEST_IDENTITY_CTOR dest=engine+0x4f0).
    // Path B push-copy 0x40, T+= ox·col0+oy·col1, dest-draw, pop (PATH_B).
    // TREE hosts that slot here (official scene+0xd8 engine). ComposeDrawWalker
    // is 0x14018aac0 (FRAME_DEST_NO_RESET). DestDraw is the vt+0x50 slot
    // (PATH_B 0x14018b170). Dest-draw GPU is DestDrawGfx (engine+0x1518).
    void                      DestStackResetIdentity();
    void                      DestStackPushCopy();
    void                      DestStackPop();
    void                      DestStackApplyPathB(SceneObject& object, float lookat_x,
                                                  float lookat_y, float amount);
    const Eigen::Matrix4f&    DestStackTop() const;
    bool                      DestStackAtBase() const;
    // FULLFB_SIZE engine+0x84/+0x88 (VIEW_ORTHO_LR). Last-pass camera uses this
    // window, not scene.ortho canvas.
    void                      SetWindowSize(int32_t w, int32_t h);
    int32_t                   window_width() const { return m_window_w; }
    int32_t                   window_height() const { return m_window_h; }
    // LASTPASS_CAM_ORTHO / VIEW_ORTHO_LR / GFX_ORTHO18 0x14009a630.
    Eigen::Matrix4f           FitOrthoCamera() const;
    // DEST_ORTHO_TNF 0x1401e9768 / GFX_ORTHO18: leftover named-RT
    // ortho(0,W,0,H,-1000,1000). dest=I (0x1401e9702 / LASTPASS_8F0_T).
    // Not dest-STACK, not last-pass fit-ortho.
    Eigen::Matrix4f           DestOrthoCamera(float width, float height) const;
    Eigen::Matrix4f           LeftoverDestOrthoMvp(const SceneObject& object) const;
    // DEST_ORTHO_TNF dest=I. TREE leftover text quads are ±half layout
    // (WPTextLayer), not leftover +0x2d8 0..AABB. Translate dest-ortho by
    // half dest so dest-local 0 lands at named-RT center.
    Eigen::Matrix4f           LeftoverTextDestOrthoMvp(const SceneObject& object) const;
    // ENGINE_FLUSH 0x1400d4264: +0x930 = camera * dest. DestDraw last-pass
    // only (LEFTOVER_VS_DESTDRAW). Not UpdateUniforms.
    void                      FlushLastPassMvp();
    const Eigen::Matrix4f&    LastPassMvp() const { return m_last_pass_mvp; }
    // DEST_BLIT 0x1401e9dd5 I*=FetchDest (I stack only). ENGINE_FLUSH
    // 0x1400d4323 +0x8f0 = I * (camera*dest). Date last-pass VERTICAL
    // g_MVP is id 0xb copies +0x930 (VERTICAL_MVP_ID). IMAGE leftover
    // +0x320==0 (IMAGE_VT_F0) and IMAGE last-pass (LASTPASS_IMAGE_ID)
    // upload this +0x8f0 stand-in. Do not copy it into +0x930.
    Eigen::Matrix4f           LastPassDrawMvp(SceneObject& object) const;
    // Official dest-draw gfx is [engine+0x1518]. TREE analog is this Scene
    // slot, bound after command.Begin (LEFTOVER_VS_DESTDRAW).
    void                      BindDestDrawGfx(DestDrawGfx* gfx) { m_dest_draw_gfx = gfx; }
    DestDrawGfx*              dest_draw_gfx() const { return m_dest_draw_gfx; }

    void SetLayerLocalVisibility(int32_t layer_id, bool visible);
    bool GetLayerLocalVisibility(int32_t layer_id) const;
    bool IsLayerVisible(int32_t layer_id) const;
    void ApplyLayerVisibility(int32_t layer_id);
    void ApplyAllLayerVisibility();
    void UpdateModelCameraPath();
    void UpdateCameraShake();
    Eigen::Vector3f ShadowCascadeCenter() const {
        if (! modelPerspectiveCameraName.empty()) {
            auto it = cameras.find(modelPerspectiveCameraName);
            if (it != cameras.end() && it->second) {
                return it->second->GetPosition().cast<float>();
            }
        }
        if (activeCamera != nullptr) return activeCamera->GetPosition().cast<float>();
        return Eigen::Vector3f::Zero();
    }
    Eigen::Vector3f ResolveCameraLayerNodeTranslation(
        const std::array<float, 3>& authored_origin) const;
    void UpdateActiveCameraLayer();

    // Official 2.3 MSAA: user `msaa.quality` is remembered, but sample count / MS RT
    // creation use 0 unless scene.json objects[] contains a non-null "model" key.
    int EffectiveMsaaQuality() const { return has3dModels ? msaa.quality : 0; }
    int MsaaSampleCount() const { return msaa.SampleCount(has3dModels); }

    SceneImageEffect*       FindImageEffect(int32_t owner_layer_id, uint32_t effect_index);
    const SceneImageEffect* FindImageEffect(int32_t owner_layer_id, uint32_t effect_index) const;
    SceneImageEffect*       FindImageEffectById(int32_t owner_layer_id, int32_t effect_id);
    const SceneImageEffect* FindImageEffectById(int32_t owner_layer_id, int32_t effect_id) const;
    bool                    SetEffectLocalVisibility(int32_t owner_layer_id,
                                                     uint32_t effect_index, bool visible);
    bool                    SetEffectLocalVisibilityById(int32_t owner_layer_id,
                                                         int32_t effect_id, bool visible);
    std::shared_ptr<Image>  GetParsedImageIfReady(const std::string& texture_key);
    std::shared_ptr<Image>  ParseImageBlockingCached(const std::string& texture_key);
    ParsedImageRequest      RequestParsedImageAsync(const std::string& texture_key);
    void                    DropParsedImageCache(std::string_view texture_key);
    void                    ClearParsedImageCache();

    std::unordered_map<std::string, SceneTexture>      textures;
    std::unordered_map<std::string, SceneRenderTarget> renderTargets;

    std::unordered_map<std::string, std::shared_ptr<SceneCamera>> cameras;
    std::unordered_map<std::string, std::vector<std::string>>     linkedCameras;

    std::vector<std::unique_ptr<SceneLight>> lights;

    std::shared_ptr<SceneNode>           sceneGraph;
    std::unique_ptr<IShaderValueUpdater> shaderValueUpdater;
    std::unique_ptr<IImageParser>        imageParser;
    std::unique_ptr<fs::VFS>             vfs;
    std::unique_ptr<WPSceneScriptHost>   scriptHost;
    std::vector<WPSceneScriptRegistration> bindingRegistrations;
    std::vector<WPSceneScriptRegistration> scriptRegistrations;
    std::vector<WPSceneScriptRegistration> propertyAnimationRegistrations;
    std::vector<int32_t>                 layerOrder;
    // Official [scene+0x158] parse-order object list (0x140190837).
    std::vector<SceneObject*>            objectList;
    // Official [scene+0x144] (0x1401850c9 / 0x14018512e). Ctor/heap leaves it 0;
    // .text has no add/inc of this slot. FetchDest copies it into object +0xd0.
    uint32_t                             destStamp { 0 };
    std::unordered_map<int32_t, std::unique_ptr<SceneObject>> sceneObjects;
    std::unordered_map<int32_t, SceneNode*> layerNodes;
    std::unordered_map<int32_t, LayerParentBinding> layerParentBindings;
    std::unordered_map<int32_t, std::vector<SceneNode*>> objectRuntimeNodes;
    std::unordered_map<int32_t, std::vector<std::string>> objectRuntimeCameraNames;
    std::unordered_map<int32_t, std::vector<std::string>> objectRuntimeRenderTargets;
    std::unordered_map<int32_t, std::vector<SceneLight*>> objectRuntimeLights;
    std::unordered_map<int32_t, std::vector<ParticleSubSystem*>> objectRuntimeParticleSubsystems;
    std::unordered_set<int32_t>                          deferredRuntimeImageLayerIds;
    std::unordered_set<int32_t>                          deferredRuntimeParticleLayerIds;
    std::unordered_set<int32_t>                          deferredRuntimeTextLayerIds;
    std::unordered_map<int32_t, uint32_t>                 objectRuntimeSoundHandles;
    std::unordered_map<int32_t, ImageLayerRuntimeState>   imageLayers;
    std::unordered_map<int32_t, TextLayerRuntimeState>    textLayers;
    std::unordered_map<int32_t, CameraLayerRuntimeState>  cameraLayers;
    std::vector<int32_t>                                  cameraLayerOrder;
    std::unordered_map<SceneNode*, int32_t> nodeOwners;
    std::unordered_map<int32_t, std::string> initialLayerConfigJson;
    std::unordered_map<std::string, int32_t> layerNameToId;
    std::unordered_set<int32_t>              offscreenDependencyLayerIds;
    // Some runtime nodes must stay root-owned for transform correctness, effect-camera routing, or
    // deferred materialization, but Wallpaper Engine still orders them as children of their
    // authored parent layer. These maps keep physical ownership separate from authored render
    // order so the render graph can emit passes at the correct sibling position without changing
    // the node's transform/output ownership model.
    std::unordered_map<SceneNode*, std::vector<SceneNode*>> renderOrderProxyChildren;
    std::unordered_set<SceneNode*>                          renderOrderProxyNodes;
    // Official passthrough (composelayer/projectlayer) sets +0x120 bit2
    // (0x1401faeb8 / 0x1401fb35f). Drawable children of those layers skip
    // `_rt_FullFrameBuffer` at 0x1401e8f6f. Empties are not in this set.
    std::unordered_set<int32_t>                             passthroughLayerIds;
    // Effect-backed image/text layers split into a visible world node plus a root-owned source
    // node that draws through the effect camera. The source must render at the world node's
    // authored position in sibling order, then be skipped when the physical root traversal reaches
    // the root-owned source node later.
    std::unordered_map<SceneNode*, std::vector<SceneNode*>> detachedEffectSourceNodesByLayerNode;
    std::unordered_set<SceneNode*>                          detachedEffectSourceNodes;
    UserPropertyMap                      userProperties;
    std::set<std::string>                dirtyImportedTextureKeys;
    std::unordered_set<std::string>      dirtyRenderTargetKeys;
    // Runtime imported images can replace their TextureCache allocation when dimensions or slot
    // layout change. Prepared passes keep copied Vulkan image/view handles, so they must rebind any
    // imported key whose CPU image changed before the next draw records uploads and executes.
    std::unordered_set<std::string>      dirtyImportedTextureResourceKeys;
    // Runtime visibility changes now use explicit resource residency instead of relying on a broad
    // texture-cache clear during every render-graph topology rebuild. Script/property code records
    // the concrete cache keys owned only by a hidden layer branch here, and the Vulkan render thread
    // drains the sets after old passes have released their descriptors but before the new graph is
    // prepared. Keeping the queue on Scene preserves thread ownership: scene mutation decides what
    // became unreachable, while Vulkan owns the actual GPU/video destruction.
    std::unordered_set<std::string>      pendingStaticTextureReleaseKeys;
    std::unordered_set<std::string>      pendingVideoTextureReleaseKeys;
    std::unordered_set<std::string>      pendingRenderTargetReleaseKeys;
    // Direct text rerastering changes pass-owned atlas and mesh resources without naming a
    // render-target dependency, so text layers need their own resource-refresh dirty key set.
    std::unordered_set<int32_t>          dirtyTextLayerIds;
    bool                                 renderGraphDirty { false };
    bool                                 renderGraphResourcesDirty { false };
    bool                                 renderGraphTopologyDirty { false };
    bool                                 renderGraphAllResourcesDirty { false };
    // Wallpaper Engine scene scripts can pause and resume video textures independently from layer
    // visibility. Keep the desired playback state on the scene so the script host can update it
    // during the QuickJS tick while the Vulkan video cache consumes the same state on the render
    // thread before polling GStreamer.
    std::unordered_map<std::string, bool> videoTexturePaused;
    // stop() is stronger than pause(): authors use it for finished intro videos that should stop
    // consuming decoder/upload work entirely. Keep it separate from pause so play() can clear the
    // stopped state while pause() can still preserve the current decoded frame.
    std::unordered_set<std::string>        videoTextureStopped;
    // Wallpaper Engine exposes getVideoTexture().setCurrentTime() as an imperative decoder
    // command rather than a persistent property. Store pending seeks separately from pause state so
    // scripts can queue a seek during init before the render graph has created the video cache
    // entry, and let the Vulkan render thread erase each request only after the matching decoder
    // has accepted it.
    std::unordered_map<std::string, double> videoTextureSeekRequests;
    // Playback rate is a persistent IVideoTexture property. QuickJS writes the desired value here,
    // and the Vulkan render thread applies it to the matching GStreamer entries before polling the
    // next decoded frame.
    std::unordered_map<std::string, double> videoTextureRates;
    // Decoder timing belongs to GStreamer, while scripts execute before the render-thread poll.
    // Publish a coherent previous-frame snapshot instead of exposing pipeline objects across the
    // script/render boundary. Only requested keys are queried so unrelated video wallpapers pay no
    // synchronous position/duration query cost.
    std::unordered_map<std::string, VideoTextureRuntimeState> videoTextureRuntimeStates;
    std::unordered_set<std::string> videoTextureRuntimeStateRequests;

    std::string scene_id { "unknown_id" };

    bool first_frame_ok { false };

    SceneMesh default_effect_mesh;

    std::unique_ptr<ParticleSystem> paritileSys;
    audio::SoundManager*            soundManager { nullptr };
    std::array<float, 2>            mousePositionNormalized { 0.5f, 0.5f };
    bool                            cursorLeftDown { false };

    SceneCamera* activeCamera;
    std::shared_ptr<SceneNode> defaultGlobalCameraNode;
    double                     defaultGlobalCameraZoom { 1.0 };
    int32_t                    activeCameraLayerId { 0 };
    std::string                modelPerspectiveCameraName;
    std::vector<CameraPathSegment> modelCameraPathSegments;
    bool                           modelCameraPathEnabled { false };
    int32_t                        activeModelCameraPathSegment { -1 };

    i32                  ortho[2] { 1920, 1080 }; // w, h
    // The authored canvas and the physical renderer output are independent. Text bridge
    // projection uses the final pixel extent after fill-mode framing, while layout, cameras, and
    // effect sampling continue to use authored scene units.
    std::array<uint32_t, 2> physicalOutputExtent { 0u, 0u };
    std::array<float, 3> clearColor { 1.0f, 1.0f, 1.0f };
    std::array<float, 3> ambientColor { 0.2f, 0.2f, 0.2f };
    std::array<float, 3> skylightColor { 0.3f, 0.3f, 0.3f };
    BloomSettings        bloom;
    // Official quality checkbox `reflection`. The RT stays registered when receivers exist;
    // this flag only gates the mirrored producer pass that populates `_rt_Reflection`.
    bool                 reflectionsEnabled { true };

    struct VolumetricLightPass {
        SceneLight*                light { nullptr };
        std::shared_ptr<SceneNode> back;
        std::shared_ptr<SceneNode> front;
        std::shared_ptr<SceneNode> fullscreen;
    };

    // volumetricsfront QUALITY + (SHADOW||COOKIE): 12/24/32/64 vs 2/3/5/8.
    static int VolumetricRaySampleCount(int quality, bool shadow_or_cookie) {
        if (quality <= 0) return 0;
        if (shadow_or_cookie) {
            if (quality >= 4) return 64;
            if (quality >= 3) return 32;
            if (quality >= 2) return 24;
            return 12;
        }
        if (quality >= 4) return 8;
        if (quality >= 3) return 5;
        if (quality >= 2) return 3;
        return 2;
    }

    struct VolumetricSettings {
        // QUALITY combo is the host enum 1–4. Disabled (0) skips the entire graph
        // and allocates no volumetric render targets.
        int  quality { 2 };
        int  built_quality { -1 };
        bool active { false };
        std::vector<VolumetricLightPass>          lights;
        std::vector<std::shared_ptr<SceneNode>>   nodes;
        std::vector<std::string>                  outputs;
        std::shared_ptr<SceneNode>                blur_h;
        std::shared_ptr<SceneNode>                blur_v;
        std::shared_ptr<SceneNode>                combine;
    };

    VolumetricSettings   volumetrics;

    struct ShadowSettings {
        // Shadows quality: disabled=0, low=1, medium=2 (default), high=3, ultra=4.
        int  quality { 2 };
        int  built_quality { -1 };
        bool atlas_active { false };
    };
    ShadowSettings       shadows;

    struct MsaaSettings {
        // Official msaa strings: none=0, x2=1, x4=2, x8=3. Sample count is 1 << quality.
        // `quality` is the user-requested value; pass has3dModels into SampleCount().
        int quality { 1 };
        int built_quality { -1 };
        // Device-clamped sample count written by the Vulkan backend. 0 means not yet queried.
        int device_samples { 0 };

        static int RequestedSampleCount(int quality) {
            if (quality <= 0) return 1;
            if (quality >= 3) return 8;
            return 1 << quality;
        }

        int SampleCount(bool has_3d_models) const {
            const int q = has_3d_models ? quality : 0;
            if (q <= 0) return 1;
            if (device_samples > 1) return device_samples;
            return RequestedSampleCount(q);
        }
    };
    MsaaSettings         msaa;

    struct TextureResolutionSettings {
        // Official Wallpaper Engine 2.8.42 `generalSettings.resolution`.
        // 0=full (High Quality), 1=half (High Performance), 2=auto.
        // Independent of Low/Medium/High/Ultra quality presets.
        int      quality { 0 };
        bool     drop_mip0 { false };
        uint32_t output_width { 0 };
        uint32_t output_height { 0 };
        uint64_t epoch { 1 };
    };
    TextureResolutionSettings textureResolution;

    void ApplyTextureResolution(int quality, uint32_t output_width, uint32_t output_height);
    void ApplyTextureResolutionForCurrentOutput();
    void PrepareParsedImageForGpu(Image& image);
    std::array<i32, 4> EffectiveImportedTextureResolution(const SceneTexture& texture) const;
    // Official 2.3 MSAA gate. True when any objects[] entry has a non-null "model"
    // field. `"image": "models/foo.json"` is not a model.
    bool                 has3dModels { false };

    struct LightingInventory {
        int point { 0 };
        int spot { 0 };
        int directional { 0 };
        int tube { 0 };
        std::vector<char> point_shadow;
        std::vector<char> spot_shadow;
        std::vector<char> spot_cookie;
        std::vector<char> directional_shadow;
    };
    LightingInventory    lighting;
    bool                 cameraParallax { false };
    float                cameraParallaxAmount { 0.0f };
    float                cameraParallaxDelay { 0.0f };
    float                cameraParallaxMouseInfluence { 0.0f };
    bool                 cameraOrthographic { true };
    // Particle perspective camera. Copied from scene `general.perspectiveoverridefov` (default 95).
    float                perspectiveOverrideFov { 95.0f };
    bool                 cameraShake { false };
    float                cameraShakeAmplitude { 0.5f };
    float                cameraShakeRoughness { 1.0f };
    float                cameraShakeSpeed { 3.0f };
    double               textRenderScale { 1.0 };

    double elapsingTime { 0.0f }, frameTime { 0.0f };
    void   PassFrameTime(double t) {
          frameTime = t;
          elapsingTime += t;
    }

    void MarkRenderGraphResourcesDirty() {
        // Global resource refreshes are still available for broad image-layer edits whose affected
        // render-target set is not known locally. Clearing the selective target set here makes the
        // renderer intentionally refresh every prepared pass instead of accidentally treating a
        // previous text-bridge target as the only dirty dependency.
        renderGraphDirty = true;
        renderGraphResourcesDirty = true;
        renderGraphAllResourcesDirty = true;
        dirtyRenderTargetKeys.clear();
        dirtyImportedTextureResourceKeys.clear();
        dirtyTextLayerIds.clear();
    }

    void MarkRenderTargetResourcesDirty(std::string render_target_key) {
        // Text bridges resize like particle-owned resources: the graph topology stays stable and
        // only passes that write or sample the changed render target need their framebuffer and
        // descriptor handles refreshed. Keeping this target list explicit prevents one clock tick
        // from refreshing every unrelated wallpaper effect pass.
        renderGraphDirty = true;
        renderGraphResourcesDirty = true;
        if (!renderGraphAllResourcesDirty && !render_target_key.empty()) {
            dirtyRenderTargetKeys.insert(std::move(render_target_key));
        }
    }

    void MarkImportedTextureResourcesDirty(std::string imported_texture_key) {
        // Keep imported-image rebinding selective. Media thumbnails affect only the passes that
        // sample their system texture keys; refreshing every prepared pass would turn a cover
        // change into a scene-wide resource rebuild.
        renderGraphDirty = true;
        renderGraphResourcesDirty = true;
        if (!renderGraphAllResourcesDirty && !imported_texture_key.empty()) {
            dirtyImportedTextureResourceKeys.insert(std::move(imported_texture_key));
        }
    }

    void MarkTextLayerResourcesDirty(int32_t layer_id) {
        // Direct text layers can change glyph atlas textures and vertex meshes while keeping the
        // same render target. Track the owning layer separately so the renderer refreshes the exact
        // TextPass before the next frame upload instead of discovering the atlas change during draw.
        renderGraphDirty = true;
        renderGraphResourcesDirty = true;
        if (!renderGraphAllResourcesDirty && layer_id != 0) {
            dirtyTextLayerIds.insert(layer_id);
        }
    }

    void MarkRenderGraphTopologyDirty() {
        // Topology rebuilds are the heavy path: the set of render passes or runtime scene nodes
        // changed, so the renderer must rebuild the graph structure and recreate its resources.
        // A topology change also implies a resource refresh, so both flags rise together here.
        renderGraphDirty = true;
        renderGraphResourcesDirty = true;
        renderGraphTopologyDirty = true;
    }

    void ClearRenderGraphDirty() {
        renderGraphDirty = false;
        renderGraphResourcesDirty = false;
        renderGraphTopologyDirty = false;
        renderGraphAllResourcesDirty = false;
        dirtyRenderTargetKeys.clear();
        dirtyImportedTextureResourceKeys.clear();
        dirtyTextLayerIds.clear();
    }

    void UpdateLinkedCamera(const std::string& name) {
        if (linkedCameras.count(name) != 0) {
            auto& cams = linkedCameras.at(name);
            for (auto& cam : cams) {
                if (cameras.count(cam) != 0) {
                    cameras.at(cam)->Clone(*cameras.at(name));
                    cameras.at(cam)->Update();
                }
            }
        }
    }

private:
    static constexpr size_t kDestStackSlots = 8;
    std::array<Eigen::Matrix4f, kDestStackSlots> m_dest_slots {};
    size_t m_dest_index { 0 };
    int32_t m_window_w { 0 };
    int32_t m_window_h { 0 };
    Eigen::Matrix4f m_last_pass_mvp { Eigen::Matrix4f::Identity() };
    DestDrawGfx* m_dest_draw_gfx { nullptr };

    struct PendingParsedImageRequest {
        std::future<std::shared_ptr<Image>>    future;
        std::chrono::steady_clock::time_point  started_at;
    };

    std::shared_ptr<Image> CacheParsedImageResultLocked(
        const std::string& texture_key,
        std::shared_ptr<Image> image,
        std::chrono::steady_clock::time_point started_at,
        const char* success_event,
        const char* failure_event);

    mutable std::mutex m_parsed_image_mutex;
    std::unordered_map<std::string, std::shared_ptr<Image>> m_parsed_image_cache;
    std::unordered_map<std::string, PendingParsedImageRequest> m_pending_parsed_images;
    std::unordered_set<std::string> m_failed_parsed_images;
};
} // namespace wallpaper
