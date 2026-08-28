#pragma once
#include <memory>
#include <vector>
#include <array>
#include <unordered_map>
#include <cstdint>

#include <Eigen/Dense>

#include "Core/Core.hpp"
#include "Interface/IShaderValueUpdater.h"
#include "Core/MapSet.hpp"
#include "SpriteAnimation.hpp"
#include "WPPuppet.hpp"
#include "Scene/SceneLight.hpp"

namespace wallpaper
{

class Scene;
class SceneNode;
class SceneMesh;
class SceneImageEffectLayer;

struct WPUniformInfo {
    bool has_MI { false };
    bool has_M { false };
    bool has_AM { false };
    bool has_AVP { false };
    bool has_EM { false };
    bool has_RV0 { false };
    bool has_RV1 { false };
    bool has_RV2 { false };
    bool has_RV3 { false };
    bool has_RV4 { false };
    bool has_MVP { false };
    // Wallpaper Engine effect passes can render through helper meshes while shader math still
    // depends on the authored layer transform. Track these matrix uniforms separately so effect
    // shaders get that contract without requiring a g_ModelViewProjectionMatrix declaration too.
    bool has_LMM { false };
    bool has_EMVP { false };
    bool has_MVPI { false };
    bool has_ETVP { false };
    bool has_ETVPI { false };
    bool has_VP { false };

    bool has_BONES { false };
    bool has_TIME { false };
    bool has_DAYTIME { false };
    // Cursor feedback shaders need these values as a coherent per-frame set. Tracking them beside
    // the older pointer position bit keeps UpdateUniforms data-driven: ordinary materials do not pay
    // for cursor state writes, while effects like cursorripple receive every uniform they declare.
    bool has_FRAMETIME { false };
    bool has_POINTERPOSITION { false };
    bool has_POINTERPOSITIONLAST { false };
    bool has_POINTERSTATE { false };
    bool has_PARALLAXPOSITION { false };
    bool has_TEXELSIZE { false };
    bool has_TEXELSIZEHALF { false };
    bool has_SCREEN { false };
    bool has_LP { false };
    // These extra light payloads and the camera basis uniforms are part of the 3D model shader
    // contract. InitUniforms enables them only on materialized model nodes so 2D shaders keep their
    // previous uniform surface even if they happen to declare similarly named values.
    bool has_model_LCP { false };
    bool has_LCR { false };
    bool has_LPOINT_ORIGIN { false };
    bool has_LPOINT_COLOR { false };
    bool has_LSPOT_ORIGIN { false };
    bool has_LSPOT_COLOR { false };
    bool has_LSPOT_DIRECTION { false };
    bool has_LSPOT_EXPONENT { false };
    bool has_LDIR_COLOR { false };
    bool has_LDIR_DIRECTION { false };
    bool has_LTUBE_ORIGINA { false };
    bool has_LTUBE_ORIGINB { false };
    bool has_LTUBE_COLOR { false };
    bool has_LFEAT_SHADOW_POINT_PROJ { false };
    bool has_LFEAT_SHADOW_POINT_XFORM { false };
    bool has_LFEAT_SHADOW_PROJ { false };
    bool has_LFEAT_SHADOW_PROJ_XFORM { false };
    bool has_EYE_POSITION { false };
    bool has_VIEWUP { false };
    bool has_VIEWRIGHT { false };
    bool has_VIEWFORWARD { false };
    std::array<bool, 3> has_audio_spectrum_left { false, false, false };
    std::array<bool, 3> has_audio_spectrum_right { false, false, false };

    struct Tex {
        bool has_resolution { false };
        bool has_mipmap { false };
    };
    std::array<Tex, 12> texs;
};

enum class WPNodeTransformBindingMode
{
    None,
    InheritParent,
    BoneAttachment,
};

struct WPNodeTransformBinding {
    WPNodeTransformBindingMode mode { WPNodeTransformBindingMode::None };
    SceneNode*                 parent { nullptr };
    uint32_t                   bone_index { 0xFFFFFFFFu };
    Eigen::Affine3f            bind_transform { Eigen::Affine3f::Identity() };
    Eigen::Affine3f            local_transform { Eigen::Affine3f::Identity() };

    bool InheritsParentTransform() const {
        return mode == WPNodeTransformBindingMode::InheritParent;
    }

    bool IsBoneAttachment() const {
        return mode == WPNodeTransformBindingMode::BoneAttachment;
    }
};

struct EffectTextureProjectionBinding {
    SceneNode* node { nullptr };
    SceneMesh* mesh { nullptr };
    // Official +0x2f0/+0x2f4. Leftover dest-draw +0x9b0 and g_ETVP take S(w/2)
    // at DEST_MVP 0x1401ec51f / G_ETVP 0x1401ec338. Last-pass g_MVP is +0x930
    // camera*dest, not +0x9b0. Dest 3×3 does not scale (0x140185150).
    float      width { 0.0f };
    float      height { 0.0f };
};

struct PuppetSurfaceBinding {
    SceneImageEffectLayer* layer { nullptr };
    SceneMesh*              skinned_mesh { nullptr };
};

struct WPShaderValueData {
    std::array<float, 2> parallaxDepth { 0.0f, 0.0f };
    // Presence does not change the runtime contract. Internal renderer nodes default to authored.
    bool                 parallaxDepthAuthored { true };
    // index + name
    std::vector<std::pair<usize, std::string>> renderTargets;

    WPPuppetLayer puppet_layer;
    SceneNode*     parallax_anchor { nullptr };
    WPNodeTransformBinding transform_binding {};
    EffectTextureProjectionBinding effect_texture_projection {};
    PuppetSurfaceBinding           puppet_surface {};
    // The scene transform already carries the required displacement for these nodes, or they are
    // private effect sources. Keep their parallax anchor for dependents without adding a second
    // local model offset.
    bool                  suppress_model_parallax { false };

    // Volumetric util materials pack g_RenderVar0–4 and the light-volume matrices per light.
    SceneLight*           volumetric_light { nullptr };
    bool                  volumetric_pass { false };

    void SetParallaxAnchor(SceneNode* parent) { parallax_anchor = parent; }

    void SetParallaxContract(const std::array<float, 2>& depth, SceneNode* anchor = nullptr,
                             bool suppress_own_model_parallax = false,
                             bool depth_authored = true) {
        parallaxDepth           = depth;
        parallaxDepthAuthored   = depth_authored;
        parallax_anchor         = anchor;
        suppress_model_parallax = suppress_own_model_parallax;
    }

    void SetEffectTextureProjection(SceneNode* projection_node, SceneMesh* projection_mesh,
                                    float width = 0.0f, float height = 0.0f) {
        effect_texture_projection.node   = projection_node;
        effect_texture_projection.mesh   = projection_mesh;
        effect_texture_projection.width  = width;
        effect_texture_projection.height = height;
    }

    void SetPuppetSurface(SceneImageEffectLayer* surface_layer, SceneMesh* skinned_mesh) {
        puppet_surface.layer = surface_layer;
        puppet_surface.skinned_mesh = skinned_mesh;
    }

    void SuppressOwnModelParallax() { suppress_model_parallax = true; }

    void CopyParallaxContractFrom(const WPShaderValueData& source) {
        SetParallaxContract(
            source.parallaxDepth,
            source.parallax_anchor,
            source.suppress_model_parallax,
            source.parallaxDepthAuthored);
    }

    void InheritParentTransform(SceneNode* parent, bool inherit_parent_parallax = true) {
        parallax_anchor        = inherit_parent_parallax ? parent : nullptr;
        transform_binding.mode = WPNodeTransformBindingMode::InheritParent;
        transform_binding.parent = parent;
    }

    void AttachToBone(SceneNode* parent, uint32_t bone_index,
                      const Eigen::Affine3f& bind_transform,
                      const Eigen::Affine3f& local_transform) {
        parallax_anchor                    = parent;
        transform_binding.mode             = WPNodeTransformBindingMode::BoneAttachment;
        transform_binding.parent           = parent;
        transform_binding.bone_index       = bone_index;
        transform_binding.bind_transform   = bind_transform;
        transform_binding.local_transform  = local_transform;
    }

    bool InheritsSceneParentTransform() const {
        return transform_binding.InheritsParentTransform();
    }

    bool IsBoneAttached() const { return transform_binding.IsBoneAttachment(); }

    bool AppliesModelParallax() const {
        // Official Path B (FUN_14018aac0) has no bone skip. Attach is dest-only.
        return ! suppress_model_parallax;
    }

    SceneNode* TransformParent() const { return transform_binding.parent; }
};

struct WPCameraParallax {
    bool  enable { false };
    float amount;
    float delay;
    float mouseinfluence;
};

class WPShaderValueUpdater : public IShaderValueUpdater {
public:
    WPShaderValueUpdater(Scene* scene): m_scene(scene) {}
    virtual ~WPShaderValueUpdater() {}

    void PrepareFrame() override;
    void FrameBegin() override;
    void ComposeDrawWalker() override;

    void InitUniforms(SceneNode*, const ExistsUniformOp&) override;
    void UpdateUniforms(SceneNode*, sprite_map_t&, const UpdateUniformOp&,
                        const ShaderUniformOverrides* overrides = nullptr) override;
    void FrameEnd() override;
    Eigen::Matrix4d ResolveModelTransformForProjection(
        SceneNode* node, const SceneCamera* camera, bool apply_parallax) override;
    void MouseInput(double, double) override;
    void SetTexelSize(float x, float y) override;

    void SetNodeData(void*, const WPShaderValueData&);
    const WPShaderValueData* GetNodeData(const void* node_addr) const;
    WPShaderValueData*       GetNodeData(const void* node_addr);
    void ReplaceNodeReferences(SceneNode* old_node, SceneNode* new_node);
    void SetCameraParallax(const WPCameraParallax& value) {
        m_parallax = value;
        // Camera parallax changes alter the derived model transforms even when the authored layer
        // transform data is unchanged. Clear the per-frame caches immediately so a runtime toggle
        // cannot leave puppet/model layers using offsets computed with the previous global state.
        m_modelTransformCache.clear();
        m_parallaxOffsetCache.clear();
        m_attachmentTransformCache.clear();
    }
    void     AdvanceAllPuppets();
    uint64_t NextPuppetFrameSerial() const noexcept { return m_puppet_frame_serial + 1; }

    void SetScreenSize(i32 w, i32 h) override;

private:
    Scene*               m_scene;
    WPCameraParallax     m_parallax;
    double               m_dayTime { 0.0f };
    std::array<float, 2> m_texelSize { 1.0f / 1920.0f, 1.0f / 1080.0f };

    std::array<float, 2> m_mousePos { 0.5f, 0.5f };
    // Stores the position published on the previous FrameBegin. Feedback effects compare this with
    // g_PointerPosition to decide whether to inject a new cursor impulse into their simulation FBOs.
    std::array<float, 2> m_mousePosLast { 0.5f, 0.5f };
    std::array<float, 2> m_mousePosInput { 0.5f, 0.5f };
    std::array<float, 2> m_parallaxLookat { 0.0f, 0.0f };
    bool                 m_parallaxLookatValid { false };

    std::array<float, 2> m_screen_size { 1920, 1080 };

    uint64_t                     m_puppet_frame_serial { 0 };
    Map<void*, Eigen::Matrix4d>  m_modelTransformCache;
    Map<void*, Eigen::Vector3f>  m_parallaxOffsetCache;
    Map<void*, Eigen::Affine3f>  m_attachmentTransformCache;
    Map<void*, WPShaderValueData> m_nodeDataMap;
    Map<void*, WPUniformInfo>     m_nodeUniformInfoMap;
};
} // namespace wallpaper
