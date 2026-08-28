#include "WPShaderValueUpdater.hpp"
#include "WPNodeTransformResolver.hpp"
#include "Eigen/src/Core/Matrix.h"
#include "Eigen/src/Geometry/Transform.h"
#include "Scene/Scene.h"
#include "Scene/SceneCamera.h"
#include "Scene/SceneImageEffectLayer.h"
#include "Scene/SceneNode.h"
#include "Scene/SceneObject.h"
#include "SpriteAnimation.hpp"
#include "SpecTexs.hpp"
#include "Core/ArrayHelper.hpp"
#include "Utils/Algorism.h"
#include "Utils/Logging.h"

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <iostream>
#include <ctime>
#include <cmath>
#include <algorithm>
#include <limits>
#include <numeric>
#include <unordered_set>
#include <vector>

using namespace wallpaper;
using namespace Eigen;

namespace
{
constexpr float kDefaultMouseCoord = 0.5f;
constexpr double kParallaxDelayRange = 3.0;
constexpr double kParallaxResponseRate = 10.0;
constexpr std::array<uint32_t, 3> kAudioSpectrumResolutions { 16, 32, 64 };
constexpr std::array<const char*, 3> kAudioSpectrumLeftUniforms {
    "g_AudioSpectrum16Left",
    "g_AudioSpectrum32Left",
    "g_AudioSpectrum64Left",
};
constexpr std::array<const char*, 3> kAudioSpectrumRightUniforms {
    "g_AudioSpectrum16Right",
    "g_AudioSpectrum32Right",
    "g_AudioSpectrum64Right",
};

struct MeshBounds2D {
    bool     valid { false };
    Vector3d center { Vector3d::Zero() };
    Vector2d halfExtent { Vector2d::Ones() };
};

Matrix4d ApplyMeshGeometryTransform(const Matrix4d& model, const SceneMesh* mesh) {
    if (mesh == nullptr) return model;
    return model * mesh->GeometryTransform().matrix().cast<double>();
}

float SanitizeMouseCoord(double value) {
    if (! std::isfinite(value)) return kDefaultMouseCoord;
    return std::clamp(static_cast<float>(value), 0.0f, 1.0f);
}

std::array<float, 2> ComposeParallaxLookatWorld(const Scene& scene,
                                                const WPCameraParallax& parallax,
                                                const std::array<float, 2>& mouse_input) {
    Vector2f ortho { (float)scene.ortho[0], (float)scene.ortho[1] };
    Vector2f mouse_vec =
        Scaling(1.0f, -1.0f) * (Vector2f { 0.5f, 0.5f } - Vector2f(&mouse_input[0]));
    mouse_vec = mouse_vec.cwiseProduct(ortho) * parallax.mouseinfluence;
    Vector2f cam_xy { ortho.x() * 0.5f, ortho.y() * 0.5f };
    if (scene.activeCamera != nullptr) {
        const auto cam = scene.activeCamera->GetPosition();
        cam_xy = Vector2f((float)cam.x(), (float)cam.y());
    }
    const Vector2f lookat = cam_xy - mouse_vec;
    return { lookat.x(), lookat.y() };
}

void DelayParallaxLookat(std::array<float, 2>& lookat, bool& valid,
                         const std::array<float, 2>& target, bool snap, double t) {
    if (snap || ! valid) {
        lookat = target;
        valid  = true;
        return;
    }
    lookat = { (float)algorism::lerp(t, lookat[0], target[0]),
               (float)algorism::lerp(t, lookat[1], target[1]) };
}

// Official lookat writer (0x140189c90–0x140189cc6) stores
// clamp(delayed_lookat / ortho, 0..1) at engine+0x9C/+0xA0 after the
// compose-then-delay lerp. Name-table id 0x6B is g_ParallaxPosition;
// scene ctor defaults those lanes to 0.5f when camera parallax is off
// (the writer skips the store). g_PointerPosition stays on delayed NDC
// mouse. Engine flag 0x800 then does x=1-x; Vivid has no equivalent.
std::array<float, 2> ComposeParallaxPositionNdc(const Scene& scene,
                                                const WPCameraParallax& parallax,
                                                const std::array<float, 2>& lookat,
                                                bool lookat_valid) {
    if (! parallax.enable || ! lookat_valid) {
        return { kDefaultMouseCoord, kDefaultMouseCoord };
    }
    const float ortho_w = static_cast<float>(scene.ortho[0]);
    const float ortho_h = static_cast<float>(scene.ortho[1]);
    if (!(ortho_w > 0.0f) || !(ortho_h > 0.0f)) {
        return { kDefaultMouseCoord, kDefaultMouseCoord };
    }
    return { std::clamp(lookat[0] / ortho_w, 0.0f, 1.0f),
             std::clamp(lookat[1] / ortho_h, 0.0f, 1.0f) };
}

MeshBounds2D ComputeMeshBounds2D(const SceneMesh* mesh) {
    if (mesh == nullptr || mesh->VertexCount() == 0) return {};

    if (mesh->HasBounds()) {
        const auto min_pos = mesh->BoundsMin().cast<double>();
        const auto max_pos = mesh->BoundsMax().cast<double>();
        const auto center  = (min_pos + max_pos) * 0.5;
        const auto halfExtent = Vector2d(std::max((max_pos.x() - min_pos.x()) * 0.5, 1e-6),
                                         std::max((max_pos.y() - min_pos.y()) * 0.5, 1e-6));
        return MeshBounds2D { .valid = true, .center = center, .halfExtent = halfExtent };
    }

    const auto& vertexArray = mesh->GetVertexArray(0);
    if (vertexArray.VertexCount() == 0) return {};

    const auto attrOffsets = vertexArray.GetAttrOffsetMap();
    if (!exists(attrOffsets, std::string(WE_IN_POSITION))) return {};

    const auto& posAttr     = attrOffsets.at(std::string(WE_IN_POSITION));
    const auto  components  = SceneVertexArray::TypeCount(posAttr.attr.type);
    const auto  stride      = vertexArray.OneSize();
    const auto  offset      = posAttr.offset / sizeof(float);
    const auto* vertexData  = vertexArray.Data();
    const auto  vertexCount = vertexArray.VertexCount();
    if (vertexData == nullptr || components < 2) return {};

    Vector3d minPos(std::numeric_limits<double>::infinity(),
                    std::numeric_limits<double>::infinity(),
                    std::numeric_limits<double>::infinity());
    Vector3d maxPos(-std::numeric_limits<double>::infinity(),
                    -std::numeric_limits<double>::infinity(),
                    -std::numeric_limits<double>::infinity());

    for (usize i = 0; i < vertexCount; ++i) {
        const auto base = i * stride + offset;
        const auto x    = static_cast<double>(vertexData[base + 0]);
        const auto y    = static_cast<double>(vertexData[base + 1]);
        const auto z    = components >= 3 ? static_cast<double>(vertexData[base + 2]) : 0.0;
        minPos = minPos.cwiseMin(Vector3d(x, y, z));
        maxPos = maxPos.cwiseMax(Vector3d(x, y, z));
    }

    const auto center     = (minPos + maxPos) * 0.5;
    const auto halfExtent = Vector2d(std::max((maxPos.x() - minPos.x()) * 0.5, 1e-6),
                                     std::max((maxPos.y() - minPos.y()) * 0.5, 1e-6));
    return MeshBounds2D { .valid = true, .center = center, .halfExtent = halfExtent };
}

bool IsModelRenderNode(SceneNode* node) {
    auto* mesh = node != nullptr ? node->Mesh() : nullptr;
    const auto* material = mesh != nullptr ? mesh->Material() : nullptr;
    // `g_EyePosition` updates are scoped to materials explicitly marked by WPModelObject
    // materialization. This prevents the new 3D camera uniform support from changing any legacy 2D
    // image, effect, text, or particle shader that happens to declare the same uniform name.
    return material != nullptr && material->modelRenderState.has_value();
}

Matrix4d ToD3dClipZViewProjection(const Matrix4d& view_projection) {
    // Vivid Ortho() maps near→NDC z=1 and far→NDC z=0 (GL-like reverse Z after the Vulkan
    // [0,1] remap). Official volumetricsfront.vert FULLSCREEN writes gl_Position.z=0 when
    // REVERSEDEPTH is off, which is D3D near. clip_z' = clip_w - clip_z makes NDC' = 1 - NDC
    // so the hull window Z and g_EffectModelMatrix unprojection share that D3D convention.
    Matrix4d d3d = view_projection;
    d3d.row(2)   = view_projection.row(3) - view_projection.row(2);
    return d3d;
}

ShaderValue ToDxcCBufferMatrixUniform(const Matrix4d& matrix) {
    // The DXC WE prologue maps authored `mul(v, M)` to native `mul(M, v)` so shader code observes
    // the same column-vector transform contract as the renderer. Keep Eigen's column-major matrix
    // bytes untouched; changing layout here would make uniform upload policy depend on the source
    // spelling of every shader expression instead of on the single language bridge in WPShaderParser.
    return ShaderValue::fromMatrix(matrix.cast<float>());
}

ShaderValue ToDxcRowVectorSkinningUniform(std::span<const Affine3f> matrices) {
    // WE authors skinning uniforms as GLSL `mat4x3`: four columns (xyz + translation) by three
    // rows, then multiplies `mul(float4(position, 1), g_Bones[i])` to get xyz. The DXC bridge spells
    // that type as HLSL `float3x4` and swaps the multiply to native `mul(M, v)`, which SPIR-V lowers
    // as a row-major matrix with ArrayStride 64. Therefore each bone needs four std140 vec4 slots:
    // the affine matrix's x/y/z rows in the first three lanes, plus a padded fourth lane that remains
    // zero. Packing compact 12-float matrices here misaligns every bone after the first one.
    std::vector<float> packed;
    packed.reserve(matrices.size() * 16);
    for (const auto& affine : matrices) {
        const Matrix4f matrix = affine.matrix();
        for (int column = 0; column < 4; ++column) {
            packed.push_back(matrix(0, column));
            packed.push_back(matrix(1, column));
            packed.push_back(matrix(2, column));
            packed.push_back(0.0f);
        }
    }
    return ShaderValue(std::span<const float>(packed.data(), packed.size()));
}

SceneNode* RemapSceneNodeReference(SceneNode* node, SceneNode* old_node, SceneNode* new_node) {
    return node == old_node ? new_node : node;
}

void PreserveDeferredRuntimeParallaxContract(const WPShaderValueData& old_data,
                                             WPShaderValueData&       new_data,
                                             SceneNode*               old_node,
                                             SceneNode*               new_node) {
    // Official keeps one SceneObject +0x170 across the object's life. Vivid rebuilds the
    // graph from JSON on materialize, so the placeholder's live depth (JSON or script) has
    // to land on the replacement the same way origin already does.
    new_data.parallaxDepth         = old_data.parallaxDepth;
    new_data.parallaxDepthAuthored = old_data.parallaxDepthAuthored;

    bool preserved = false;

    if (new_data.parallax_anchor == nullptr && old_data.parallax_anchor != nullptr) {
        new_data.parallax_anchor =
            RemapSceneNodeReference(old_data.parallax_anchor, old_node, new_node);
        preserved = true;
    }

    if (preserved && old_data.suppress_model_parallax) {
        new_data.suppress_model_parallax = true;
    }
}

Matrix4d ComputeEffectTextureProjection(const SceneNode* projectionNode,
                                        const SceneMesh* projectionMesh,
                                        const Matrix4d&  projectionModelTrans,
                                        const Matrix4d&  viewProjectionTrans,
                                        float authored_width = 0.0f,
                                        float authored_height = 0.0f) {
    if (projectionNode == nullptr) return Matrix4d::Identity();

    // Official leftover dest-draw flag0 writes dest*VP*S(w/2) at +0x9b0 (DEST_MVP
    // 0x1401ec51f). Date last-pass skips +0x9b0 (combo+0x14 bit0 clear). VERTICAL
    // g_MVP is +0x930 = camera*dest (ENGINE_FLUSH / LASTPASS_CAM_ORTHO /
    // LASTPASS_DEST_STACK), not this slot. Flag1 / I-slot g_ETVP still takes S
    // at 0x1401ec338. Dest 3×3 has no +0x2f0 (0x140185150).
    if (authored_width > 0.0f && authored_height > 0.0f) {
        const Affine3d scale(Eigen::Scaling(authored_width * 0.5, authored_height * 0.5, 1.0));
        return viewProjectionTrans * projectionModelTrans * scale.matrix();
    }

    if (projectionMesh == nullptr) return viewProjectionTrans * projectionModelTrans;
    const auto bounds = ComputeMeshBounds2D(projectionMesh);
    if (!bounds.valid) return viewProjectionTrans * projectionModelTrans;

    const auto localFromNormalized =
        (Affine3d(Eigen::Translation3d(bounds.center)) *
         Eigen::Scaling(bounds.halfExtent.x(), bounds.halfExtent.y(), 1.0))
            .matrix();
    return viewProjectionTrans * projectionModelTrans * localFromNormalized;
}

std::string_view ResolveEffectiveNodeCameraName(const SceneNode* node) {
    // Effect-backed text still contributes intermediate bridge-source quads to the generic image
    // path while the logical text owner carries the camera binding. Walking ancestors here lets
    // those bridge-source quads inherit the same offscreen camera contract as the owning text
    // primitive, so text effects stay synchronized without any text-specific fallback camera path.
    for (auto* current = node; current != nullptr; current = current->Parent()) {
        if (!current->Camera().empty()) return current->Camera();
    }
    return {};
}

} // namespace

void WPShaderValueUpdater::PrepareFrame() {
    m_puppet_frame_serial++;
    m_modelTransformCache.clear();
    m_parallaxOffsetCache.clear();
    m_attachmentTransformCache.clear();
    // 3D model camera paths are sampled before uniforms so the model-only camera projection,
    // g_EyePosition, and view-basis uniforms all describe the same frame. Scenes without model
    // camera paths return immediately inside Scene and keep the legacy 2D path untouched.
    if (m_scene != nullptr) {
        m_scene->UpdateModelCameraPath();
    }
    /*
        using namespace std::chrono;
        auto nowTime = system_clock::to_time_t(system_clock::now());
        auto cTime   = std::localtime(&nowTime);
        m_dayTime =
            (((cTime->tm_hour * 60) + cTime->tm_min) * 60 + cTime->tm_sec) / (24.0f * 60.0f
       * 60.0f);
    */
    const std::array<float, 2> previousMousePos = m_mousePos;
    const auto lookat_target =
        m_scene != nullptr ? ComposeParallaxLookatWorld(*m_scene, m_parallax, m_mousePosInput)
                           : m_parallaxLookat;
    // Official ctor 0x14018870c: ortho rest lookat = [scene+0xf0] + 0.5*ortho, then the
    // per-frame writer lerps from that stored lookat (0x140189c51). Do not start at (0,0)
    // and do not snap the first delayed frame to the current target.
    if (m_scene != nullptr && !m_parallaxLookatValid) {
        m_parallaxLookat = ComposeParallaxLookatWorld(
            *m_scene, m_parallax, { kDefaultMouseCoord, kDefaultMouseCoord });
        m_parallaxLookatValid = true;
    }
    if (!(m_parallax.delay > 0.0f) || !std::isfinite(m_parallax.delay)) {
        m_mousePosLast     = previousMousePos;
        m_mousePos         = m_mousePosInput;
        DelayParallaxLookat(m_parallaxLookat, m_parallaxLookatValid, lookat_target, true, 1.0);
        AdvanceAllPuppets();
        return;
    }

    const double frameTime = std::max(m_scene->frameTime, 0.0);
    // Wallpaper Engine maps the authored 0..3 delay setting to a response rate instead of
    // treating it as a settling duration. Keep that curve intact: scene authors tune the slider
    // against this exact relationship, and the per-frame clamp preserves the native fast path.
    const double responseRate =
        kParallaxResponseRate * (1.0 - static_cast<double>(m_parallax.delay) / kParallaxDelayRange);
    const double t = std::min(1.0, responseRate * frameTime);
    m_mousePosLast     = previousMousePos;
    m_mousePos         = std::array { (float)algorism::lerp(t, m_mousePos[0], m_mousePosInput[0]),
                                      (float)algorism::lerp(t, m_mousePos[1], m_mousePosInput[1]) };
    DelayParallaxLookat(m_parallaxLookat, m_parallaxLookatValid, lookat_target, false, t);
    AdvanceAllPuppets();
}

void WPShaderValueUpdater::SetScreenSize(i32 w, i32 h) {
    m_screen_size = { (float)w, (float)h };
    // VIEW_ORTHO_LR: last-pass camera uses FULLFB window, not canvas.
    if (m_scene != nullptr) m_scene->SetWindowSize(w, h);
}

void WPShaderValueUpdater::FrameBegin() {}

void WPShaderValueUpdater::ComposeDrawWalker() {
    // PATH_B 0x14018b022 / 0x14018b118 / 0x14018b170 / 0x14018b17a.
    // Dest-draw still runs when Path B is skipped (LEFTOVER_VS_DESTDRAW).
    if (m_scene == nullptr) return;
    for (SceneObject* object : m_scene->objectList) {
        if (object == nullptr) continue;
        m_scene->DestStackPushCopy();
        if (m_parallax.enable) {
            m_scene->DestStackApplyPathB(*object, m_parallaxLookat[0], m_parallaxLookat[1],
                                         m_parallax.amount);
        }
        object->DestDraw();
        m_scene->DestStackPop();
    }
}

void WPShaderValueUpdater::AdvanceAllPuppets() {
    if (!m_scene) return;
    const double frame_time = m_scene->frameTime;
    std::unordered_set<const void*> advanced_runtimes;
    std::vector<SceneNode*> notification_nodes;

    for (auto& [addr, nodeData] : m_nodeDataMap) {
        if (!nodeData.puppet_layer.hasPuppet()) continue;
        const void* runtime = nodeData.puppet_layer.RuntimeIdentity();
        if (!advanced_runtimes.insert(runtime).second) continue;
        nodeData.puppet_layer.AdvanceIfNeeded(frame_time, m_puppet_frame_serial);
        notification_nodes.push_back(static_cast<SceneNode*>(addr));
    }

    // Surface synchronization is a frame-preparation transaction. Every binding reads the pose
    // snapshot cached above, while SceneImageEffectLayer deduplicates multiple consumers of the
    // same private surface and owns all camera, target and publication-mesh changes.
    for (auto& [addr, nodeData] : m_nodeDataMap) {
        (void)addr;
        if (!nodeData.puppet_layer.hasPuppet() || nodeData.puppet_surface.layer == nullptr ||
            nodeData.puppet_surface.skinned_mesh == nullptr) {
            continue;
        }
        nodeData.puppet_surface.layer->PreparePuppetSurface(
            *m_scene,
            *nodeData.puppet_surface.skinned_mesh,
            nodeData.puppet_layer.PoseSnapshot(),
            m_puppet_frame_serial);
    }

    if (m_scene->scriptHost != nullptr) {
        for (auto* node : notification_nodes) {
            m_scene->scriptHost->NotifyAnimationLayersAdvanced(node);
        }
    }
}

void WPShaderValueUpdater::FrameEnd() {}

Matrix4d WPShaderValueUpdater::ResolveModelTransformForProjection(
    SceneNode* node, const SceneCamera* camera, bool apply_parallax) {
    if (m_scene == nullptr || node == nullptr) return Matrix4d::Identity();

    // Projection can run before render-graph refresh while ordinary uniform updates happen during
    // draw. Isolated caches make this query observe the current node graph without consuming a
    // matrix cached before a script changed an origin, scale, parent, or attachment in this frame.
    Map<void*, Matrix4d> local_model_cache;
    Map<void*, Vector3f> local_parallax_cache;
    Map<void*, Affine3f> local_attachment_cache;
    WPNodeTransformResolver transform_resolver(*m_scene,
                                               m_parallax,
                                               m_nodeDataMap,
                                               local_model_cache,
                                               local_parallax_cache,
                                               local_attachment_cache,
                                               camera,
                                               m_parallaxLookat,
                                               m_puppet_frame_serial);

    if (const auto* node_data = GetNodeData(node); node_data != nullptr) {
        transform_resolver.UpdateAttachmentParentIfNeeded(*node_data);
        if (const auto local_transform =
                transform_resolver.ResolveAttachmentLocalTransform(node);
            local_transform.has_value()) {
            node->SetLocalAffine(*local_transform);
        }
    }

    return transform_resolver.ResolveParallaxedModelTransform(node, camera, apply_parallax);
}

void WPShaderValueUpdater::MouseInput(double x, double y) {
    m_mousePosInput[0] = SanitizeMouseCoord(x);
    m_mousePosInput[1] = SanitizeMouseCoord(y);
}

void WPShaderValueUpdater::InitUniforms(SceneNode* pNode, const ExistsUniformOp& existsOp) {
    m_nodeUniformInfoMap[pNode] = WPUniformInfo();
    auto& info                  = m_nodeUniformInfoMap[pNode];
    info.has_MI                 = existsOp(G_MI);
    info.has_M                  = existsOp(G_M);
    info.has_AM                 = existsOp(G_AM);
    info.has_AVP                = existsOp(G_AVP);
    info.has_EM                 = existsOp(G_EM);
    info.has_RV0                = existsOp(G_RV0);
    info.has_RV1                = existsOp(G_RV1);
    info.has_RV2                = existsOp(G_RV2);
    info.has_RV3                = existsOp(G_RV3);
    info.has_RV4                = existsOp(G_RV4);
    info.has_MVP                = existsOp(G_MVP);
    info.has_LMM                = existsOp(G_LMM);
    info.has_EMVP               = existsOp(G_EMVP);
    info.has_MVPI               = existsOp(G_MVPI);
    info.has_ETVP               = existsOp(G_ETVP);
    info.has_ETVPI              = existsOp(G_ETVPI);

    info.has_VP = existsOp(G_VP);

    info.has_BONES            = existsOp(G_BONES);
    info.has_TIME             = existsOp(G_TIME);
    info.has_DAYTIME          = existsOp(G_DAYTIME);
    info.has_FRAMETIME        = existsOp(G_FRAMETIME);
    info.has_POINTERPOSITION  = existsOp(G_POINTERPOSITION);
    info.has_POINTERPOSITIONLAST = existsOp(G_POINTERPOSITIONLAST);
    info.has_POINTERSTATE     = existsOp(G_POINTERSTATE);
    info.has_PARALLAXPOSITION = existsOp(G_PARALLAXPOSITION);
    info.has_TEXELSIZE        = existsOp(G_TEXELSIZE);
    info.has_TEXELSIZEHALF    = existsOp(G_TEXELSIZEHALF);
    info.has_SCREEN           = existsOp(G_SCREEN);
    info.has_LP               = existsOp(G_LP);
    info.has_model_LCP        = IsModelRenderNode(pNode) && existsOp(G_LCP);
    info.has_LCR              = IsModelRenderNode(pNode) && existsOp(G_LCR);
    info.has_LPOINT_ORIGIN    = existsOp(G_LPOINT_ORIGIN);
    info.has_LPOINT_COLOR     = existsOp(G_LPOINT_COLOR);
    info.has_LSPOT_ORIGIN     = existsOp(G_LSPOT_ORIGIN);
    info.has_LSPOT_COLOR      = existsOp(G_LSPOT_COLOR);
    info.has_LSPOT_DIRECTION  = existsOp(G_LSPOT_DIRECTION);
    info.has_LSPOT_EXPONENT   = existsOp(G_LSPOT_EXPONENT);
    info.has_LDIR_COLOR       = existsOp(G_LDIR_COLOR);
    info.has_LDIR_DIRECTION   = existsOp(G_LDIR_DIRECTION);
    info.has_LTUBE_ORIGINA    = existsOp(G_LTUBE_ORIGINA);
    info.has_LTUBE_ORIGINB    = existsOp(G_LTUBE_ORIGINB);
    info.has_LTUBE_COLOR      = existsOp(G_LTUBE_COLOR);
    info.has_LFEAT_SHADOW_POINT_PROJ  = existsOp(G_LFEAT_SHADOW_POINT_PROJ);
    info.has_LFEAT_SHADOW_POINT_XFORM = existsOp(G_LFEAT_SHADOW_POINT_XFORM);
    info.has_LFEAT_SHADOW_PROJ        = existsOp(G_LFEAT_SHADOW_PROJ);
    info.has_LFEAT_SHADOW_PROJ_XFORM  = existsOp(G_LFEAT_SHADOW_PROJ_XFORM);
    info.has_EYE_POSITION     = IsModelRenderNode(pNode) && existsOp(G_EYE_POSITION);
    info.has_VIEWUP           = IsModelRenderNode(pNode) && existsOp(G_VIEWUP);
    info.has_VIEWRIGHT        = IsModelRenderNode(pNode) && existsOp(G_VIEWRIGHT);
    info.has_VIEWFORWARD      = IsModelRenderNode(pNode) && existsOp(G_VIEWFORWARD);
    for (size_t index = 0; index < kAudioSpectrumResolutions.size(); index++) {
        info.has_audio_spectrum_left[index] = existsOp(kAudioSpectrumLeftUniforms[index]);
        info.has_audio_spectrum_right[index] = existsOp(kAudioSpectrumRightUniforms[index]);
    }

    std::accumulate(begin(info.texs), end(info.texs), 0, [&existsOp](uint index, auto& value) {
        value.has_resolution = existsOp(WE_GLTEX_RESOLUTION_NAMES[index]);
        value.has_mipmap     = existsOp(WE_GLTEX_MIPMAPINFO_NAMES[index]);
        return index + 1;
    });
}

void WPShaderValueUpdater::UpdateUniforms(SceneNode* pNode, sprite_map_t& sprites,
                                          const UpdateUniformOp& updateOp,
                                          const ShaderUniformOverrides* overrides) {
    const auto node_cam_name = ResolveEffectiveNodeCameraName(pNode);
    const bool use_active_camera_for_uniforms =
        overrides != nullptr && overrides->use_active_camera_for_uniforms;
    const bool has_named_camera_override =
        overrides != nullptr && overrides->use_camera_override && !overrides->camera_name.empty();
    const bool has_camera_override = use_active_camera_for_uniforms || has_named_camera_override;
    const std::string_view uniform_cam_name =
        use_active_camera_for_uniforms
            ? std::string_view {}
            : (has_named_camera_override ? overrides->camera_name : node_cam_name);

    const SceneCamera* camera;
    if (! uniform_cam_name.empty()) {
        auto camera_it = m_scene->cameras.find(std::string(uniform_cam_name));
        if (camera_it != m_scene->cameras.end()) {
            camera = camera_it->second.get();
        } else {
            LOG_ERROR("ShaderUniformCameraOverride: camera '%.*s' not found for node '%s'",
                      static_cast<int>(uniform_cam_name.size()),
                      uniform_cam_name.data(),
                      pNode != nullptr ? pNode->Name().c_str() : "<null>");
            camera = m_scene->activeCamera;
        }
    } else {
        camera = m_scene->activeCamera;
    }

    if (! camera) return;

    const bool use_active_parallax_camera =
        has_camera_override && overrides->use_active_camera_for_parallax &&
        m_scene->activeCamera != nullptr;
    const SceneCamera* model_parallax_camera =
        use_active_parallax_camera ? m_scene->activeCamera : camera;
    const bool use_camera_local_transform_caches =
        has_camera_override && model_parallax_camera != m_scene->activeCamera;

    Map<void*, Matrix4d> localModelTransformCache;
    Map<void*, Vector3f> localParallaxOffsetCache;
    Map<void*, Affine3f> localAttachmentTransformCache;
    auto& modelTransformCache =
        use_camera_local_transform_caches ? localModelTransformCache : m_modelTransformCache;
    auto& parallaxOffsetCache =
        use_camera_local_transform_caches ? localParallaxOffsetCache : m_parallaxOffsetCache;
    auto& attachmentTransformCache =
        use_camera_local_transform_caches ? localAttachmentTransformCache
                                          : m_attachmentTransformCache;

    WPNodeTransformResolver transformResolver(*m_scene,
                                              m_parallax,
                                              m_nodeDataMap,
                                              modelTransformCache,
                                              parallaxOffsetCache,
                                              attachmentTransformCache,
                                              use_camera_local_transform_caches
                                                  ? model_parallax_camera
                                                  : m_scene->activeCamera,
                                              m_parallaxLookat,
                                              m_puppet_frame_serial);

    if (exists(m_nodeDataMap, pNode)) {
        auto& nodeData = m_nodeDataMap.at(pNode);
        transformResolver.UpdateAttachmentParentIfNeeded(nodeData);
        auto localTransform = transformResolver.ResolveAttachmentLocalTransform(pNode);
        if (localTransform.has_value()) {
            SceneImageEffectLayer* effectLayer { nullptr };
            if (!node_cam_name.empty()) {
                auto camera_it = m_scene->cameras.find(std::string(node_cam_name));
                if (camera_it != m_scene->cameras.end() && camera_it->second->HasImgEffect()) {
                    effectLayer = camera_it->second->GetImgEffect().get();
                }
            }

            if (effectLayer != nullptr) {
                if (auto* layer_node = effectLayer->LayerNode()) {
                    layer_node->SetLocalAffine(*localTransform);
                    layer_node->UpdateTrans();
                    // Official dest-draw last-pass T is dest-STACK, not a
                    // second FinalNode dest copy of this layer node.
                    const auto* object = m_scene != nullptr
                        ? m_scene->FindSceneObjectForNode(layer_node)
                        : nullptr;
                    if (object == nullptr || !object->DestDrawPublishesDefault()) {
                        effectLayer->SyncResolvedNodeToWorld();
                    }
                }
            } else {
                pNode->SetLocalAffine(*localTransform);
            }
        }
    }

    pNode->UpdateTrans();

    if (! node_cam_name.empty()) {
        auto camera_it = m_scene->cameras.find(std::string(node_cam_name));
        if (camera_it != m_scene->cameras.end() && camera_it->second->HasImgEffect()) {
            auto* effectLayer = camera_it->second->GetImgEffect().get();
            auto* layer_node  = effectLayer->LayerNode();
            if (layer_node != nullptr && exists(m_nodeDataMap, layer_node)) {
                auto& layer_node_data = m_nodeDataMap.at(layer_node);
                transformResolver.UpdateAttachmentParentIfNeeded(layer_node_data);
                if (layer_node_data.IsBoneAttached()) {
                    auto localTransform =
                        transformResolver.ResolveAttachmentLocalTransform(layer_node);
                    if (localTransform.has_value()) {
                        layer_node->SetLocalAffine(*localTransform);
                        layer_node->UpdateTrans();
                    }
                }
                const auto* object = m_scene != nullptr
                    ? m_scene->FindSceneObjectForNode(layer_node)
                    : nullptr;
                if ((object == nullptr || !object->DestDrawPublishesDefault()) &&
                    (layer_node_data.InheritsSceneParentTransform() ||
                     layer_node_data.IsBoneAttached())) {
                    const SceneCamera* displayCamera =
                        m_scene->activeCamera != nullptr ? m_scene->activeCamera : camera;
                    const auto worldModel =
                        effectLayer->PublishesPrivateFinalComposite()
                            ? transformResolver.ResolveRawModelTransform(layer_node)
                            : transformResolver.ResolveParallaxedModelTransform(
                                  layer_node, displayCamera, displayCamera != nullptr);
                    effectLayer->SyncResolvedNodeToMatrix(Affine3f(worldModel.cast<float>()));
                }
            }
        }
    }

    // Text is now allowed to be a first-class renderable without a backing SceneMesh material.
    // The old updater returned early here, which made transform uniforms unavailable to any
    // render path that was not disguised as a mesh/custom-shader node. Keeping material access
    // optional lets the dedicated text pass reuse the same attachment/parallax/camera transform
    // logic while still skipping mesh-only material uniform work when no mesh exists.
    auto* material = pNode->Mesh() != nullptr ? pNode->Mesh()->Material() : nullptr;
    // auto& shadervs = material->customShader.updateValueList;
    // const auto& valueSet = material->customShader.valueSet;

    assert(exists(m_nodeUniformInfoMap, pNode));
    const auto& info = m_nodeUniformInfoMap[pNode];

    bool hasNodeData = exists(m_nodeDataMap, pNode);
    if (hasNodeData) {
        auto& nodeData = m_nodeDataMap.at(pNode);
        for (const auto& el : nodeData.renderTargets) {
            if (m_scene->renderTargets.count(el.second) == 0) continue;
            const auto& rt = m_scene->renderTargets[el.second];

            const auto& unifrom_tex = info.texs[el.first];

            if (unifrom_tex.has_resolution) {
                // Runtime render targets expose one canonical resolution contract through
                // `ResolutionVector()`: physical size in `.xy`, logical content size in `.zw`.
                // Uniform updates should always forward that authoritative scene-side contract
                // directly instead of layering text-specific interpretation on top of it.
                std::array<i32, 4> resolution_uint(rt.ResolutionVector());
                updateOp(WE_GLTEX_RESOLUTION_NAMES[el.first],
                         ShaderValue(array_cast<float>(resolution_uint)));
            }
            if (unifrom_tex.has_mipmap) {
                updateOp(WE_GLTEX_MIPMAPINFO_NAMES[el.first], (float)rt.mipmap_level);
            }
        }
        if (nodeData.puppet_layer.hasPuppet() && info.has_BONES) {
            const auto pose = nodeData.puppet_layer.PoseSnapshot();
            // PrepareFrame() is the sole pose-advance boundary. Uniform consumers only publish the
            // immutable snapshot selected for this frame, so mask pre-passes, clipped main passes
            // and effect writers cannot independently advance animation or mutate render topology.
            assert(pose.frame_serial == m_puppet_frame_serial);
            updateOp(G_BONES, ToDxcRowVectorSkinningUniform(pose.skinning));
        }
    }

    if (material != nullptr) {
        const auto tex_count = std::min(material->textures.size(), info.texs.size());
        for (size_t i = 0; i < tex_count; i++) {
            if (! info.texs[i].has_resolution) continue;
            const auto& name = material->textures[i];
            if (name.empty() || m_scene->renderTargets.count(name) != 0) continue;
            const auto texture_it = m_scene->textures.find(name);
            if (texture_it == m_scene->textures.end()) continue;
            updateOp(WE_GLTEX_RESOLUTION_NAMES[i],
                     ShaderValue(array_cast<float>(
                         m_scene->EffectiveImportedTextureResolution(texture_it->second))));
        }
    }

    bool reqMI    = info.has_MI;
    bool reqM     = info.has_M;
    bool reqAM    = info.has_AM;
    bool reqMVP   = info.has_MVP;
    bool reqLMM   = info.has_LMM;
    bool reqEMVP  = info.has_EMVP;
    bool reqMVPI  = info.has_MVPI;
    bool reqETVP  = info.has_ETVP;
    bool reqETVPI = info.has_ETVPI;

    Matrix4d viewProTrans = camera->GetViewProjectionMatrix();

    if (info.has_VP) {
        updateOp(G_VP, ToDxcCBufferMatrixUniform(viewProTrans));
    }
    if (reqM || reqAM || reqMVP || reqLMM || reqEMVP || reqMI || reqMVPI || reqETVP ||
        reqETVPI) {
        // Official 0x1401ec799 writes I into [engine+0x30] only when +0x304 bit5
        // is clear (0x1401ec781 test; bit5 → 0x1401ec878 copies dest from rdi).
        // Dest blit after I pop is 0x1401e9dd5 / 0x1401e9cf9. Image draw
        // 0x1401e8f6f skips `_rt_FullFrameBuffer` for a passthrough parent and
        // then still loads scene+0x40/+0x38/+0x30 dest (0x1401e9029). The pass
        // sets use_identity_model for I-internal; a compose-source camera
        // override is dest, even though that camera HasImgEffect().
        const bool pass_requests_i =
            overrides != nullptr && overrides->use_identity_model;
        const bool pass_requests_dest =
            overrides != nullptr && overrides->use_camera_override &&
            ! overrides->use_identity_model;
        const bool official_i_slot =
            pass_requests_i ||
            uniform_cam_name == "effect" ||
            (camera != nullptr && camera->HasImgEffect() &&
             ! use_active_camera_for_uniforms && ! pass_requests_dest);
        Matrix4d modelTrans = official_i_slot
            ? Matrix4d::Identity()
            : transformResolver.ResolveParallaxedModelTransform(
                pNode, model_parallax_camera, true);

        modelTrans = ApplyMeshGeometryTransform(modelTrans, pNode->Mesh());

        if (reqM) updateOp(G_M, ToDxcCBufferMatrixUniform(modelTrans));
        if (reqAM) updateOp(G_AM, ToDxcCBufferMatrixUniform(modelTrans));
        if (reqLMM) updateOp(G_LMM, ToDxcCBufferMatrixUniform(modelTrans));
        if (reqMI) updateOp(G_MI, ToDxcCBufferMatrixUniform(modelTrans.inverse()));
        if (reqMVP || reqEMVP) {
            Matrix4d mvpTrans = viewProTrans * modelTrans;
            if (! official_i_slot) {
                // Dest-draw leftover g_MVP is dest-ortho * I (DEST_ORTHO_TNF).
                // Last-pass g_MVP is +0x930 fit-ortho * dest-STACK
                // (LIVE_LASTPASS_930). Both are Record writes, not this shared
                // UpdateUniforms path (LEFTOVER_VS_DESTDRAW). DEST_MVP +0x9b0
                // leftover scale is not g_MVP. Only the +0x2e0 ±1 card takes
                // this leftover scale. A ±size/2 mesh is already sized.
                float dest_w = 0.0f;
                float dest_h = 0.0f;
                if (hasNodeData) {
                    dest_w = m_nodeDataMap.at(pNode).effect_texture_projection.width;
                    dest_h = m_nodeDataMap.at(pNode).effect_texture_projection.height;
                }
                if ((!(dest_w > 0.0f) || !(dest_h > 0.0f)) && m_scene != nullptr) {
                    const auto owner = m_scene->nodeOwners.find(pNode);
                    if (owner != m_scene->nodeOwners.end()) {
                        const auto image = m_scene->imageLayers.find(owner->second);
                        if (image != m_scene->imageLayers.end()) {
                            dest_w = image->second.size[0];
                            dest_h = image->second.size[1];
                        }
                    }
                }
                const auto bounds = ComputeMeshBounds2D(pNode->Mesh());
                const bool unit_card =
                    bounds.valid && std::abs(bounds.halfExtent.x() - 1.0) < 0.05 &&
                    std::abs(bounds.halfExtent.y() - 1.0) < 0.05;
                if (unit_card && dest_w > 0.0f && dest_h > 0.0f) {
                    mvpTrans = mvpTrans *
                               Affine3d(Eigen::Scaling(
                                            static_cast<double>(dest_w) * 0.5,
                                            static_cast<double>(dest_h) * 0.5,
                                            1.0))
                                   .matrix();
                }
            }
            if (reqMVP) updateOp(G_MVP, ToDxcCBufferMatrixUniform(mvpTrans));
            if (reqEMVP) updateOp(G_EMVP, ToDxcCBufferMatrixUniform(mvpTrans));
            if (reqMVPI) updateOp(G_MVPI, ToDxcCBufferMatrixUniform(mvpTrans.inverse()));
        }
        if (reqETVP || reqETVPI) {
            const SceneNode* projectionNode      = pNode;
            const SceneMesh* projectionMesh      = pNode->Mesh();
            Matrix4d         projectionModelTrans = modelTrans;
            Matrix4d         projectionViewPro    = viewProTrans;

            const WPShaderValueData* nodeDataPtr = hasNodeData ? &m_nodeDataMap.at(pNode) : nullptr;
            float etvp_w = 0.0f;
            float etvp_h = 0.0f;
            if (nodeDataPtr != nullptr) {
                etvp_w = nodeDataPtr->effect_texture_projection.width;
                etvp_h = nodeDataPtr->effect_texture_projection.height;
            }
            if (nodeDataPtr != nullptr &&
                nodeDataPtr->effect_texture_projection.node != nullptr &&
                nodeDataPtr->effect_texture_projection.mesh != nullptr &&
                m_scene->activeCamera != nullptr) {
                projectionNode = nodeDataPtr->effect_texture_projection.node;
                projectionMesh = nodeDataPtr->effect_texture_projection.mesh;
                const_cast<SceneNode*>(projectionNode)->UpdateTrans();
                projectionModelTrans = ApplyMeshGeometryTransform(
                    projectionNode->ModelTrans(), projectionMesh);
                projectionViewPro    = m_scene->activeCamera->GetViewProjectionMatrix();
            }

            const auto etvpTrans = ComputeEffectTextureProjection(projectionNode,
                                                                  projectionMesh,
                                                                  projectionModelTrans,
                                                                  projectionViewPro,
                                                                  etvp_w,
                                                                  etvp_h);
            if (reqETVP) updateOp(G_ETVP, ToDxcCBufferMatrixUniform(etvpTrans));
            if (reqETVPI) {
                if (std::abs(etvpTrans.determinant()) > 1e-12) {
                    updateOp(G_ETVPI, ToDxcCBufferMatrixUniform(etvpTrans.inverse()));
                } else {
                    updateOp(G_ETVPI, ToDxcCBufferMatrixUniform(Matrix4d::Identity()));
                }
            }
        }
    }

    if (hasNodeData) {
        const auto& vol = m_nodeDataMap.at(pNode);
        if (vol.volumetric_pass && vol.volumetric_light != nullptr) {
            const SceneLight& light = *vol.volumetric_light;
            const Matrix4d    alt_vp = light.AltViewProjection().cast<double>();
            const Matrix4d    d3d_vp = ToD3dClipZViewProjection(viewProTrans);
            if (info.has_VP) updateOp(G_VP, ToDxcCBufferMatrixUniform(d3d_vp));
            if (info.has_AVP) updateOp(G_AVP, ToDxcCBufferMatrixUniform(alt_vp));
            if (info.has_EM) {
                updateOp(G_EM, ToDxcCBufferMatrixUniform(d3d_vp.inverse()));
            }
            if (info.has_AM) {
                if (light.type() == SceneLightType::Point) {
                    updateOp(G_AM, ToDxcCBufferMatrixUniform(Matrix4d::Identity()));
                } else {
                    updateOp(G_AM,
                             ToDxcCBufferMatrixUniform(light.WorldToLightClip().cast<double>()));
                }
            }
            const Vector3f origin  = light.WorldOrigin();
            const Vector3f forward = light.WorldForward();
            const Vector3f color   = light.color();
            // g_RenderVar1: radius*0.99, cos(inner), cos(outer), intensity.
            if (info.has_RV0) {
                const auto atlas = light.ShadowAtlasUv();
                updateOp(G_RV0, std::array<float, 4> { atlas.x(), atlas.y(), atlas.z(), atlas.w() });
            }
            if (info.has_RV1) {
                updateOp(G_RV1,
                         std::array<float, 4> { light.radius() * 0.9900000095367432f,
                                                std::cos(light.innerCone() * SceneLight::Deg2Rad()),
                                                std::cos(light.outerCone() * SceneLight::Deg2Rad()),
                                                light.intensity() });
            }
            if (info.has_RV2) {
                updateOp(G_RV2,
                         std::array<float, 4> {
                             origin.x(), origin.y(), origin.z(), light.density() });
            }
            if (info.has_RV3) {
                if (light.type() == SceneLightType::Point && light.castsShadows() &&
                    m_scene->shadows.quality != 0) {
                    const auto proj = light.ShadowProjectionInfo();
                    updateOp(G_RV3,
                             std::array<float, 4> { proj.x(), proj.y(), proj.z(), proj.w() });
                } else {
                    updateOp(G_RV3,
                             std::array<float, 4> { forward.x(), forward.y(), forward.z(), 0.0f });
                }
            }
            if (info.has_RV4) {
                updateOp(G_RV4,
                         std::array<float, 4> { color.x(),
                                                color.y(),
                                                color.z(),
                                                light.volumetricsExponent() });
            }
        }
    }

    //	g_EffectTextureProjectionMatrix
    // shadervs.push_back({"g_EffectTextureProjectionMatrixInverse",
    // ShaderValue::ValueOf(Eigen::Matrix4f::Identity())});
    if (info.has_TIME) updateOp(G_TIME, (float)m_scene->elapsingTime);

    if (info.has_DAYTIME) updateOp(G_DAYTIME, (float)m_dayTime);

    if (info.has_POINTERPOSITION) updateOp(G_POINTERPOSITION, m_mousePos);
    if (info.has_POINTERPOSITIONLAST) updateOp(G_POINTERPOSITIONLAST, m_mousePosLast);
    if (info.has_POINTERSTATE) {
        // Wallpaper Engine cursor ripple shaders treat `.z` as the left-button impulse term. Keep
        // the other lanes neutral because their exact editor-side meanings are effect-specific, and
        // writing arbitrary non-zero values would inject force into authored feedback buffers.
        updateOp(G_POINTERSTATE,
                 std::array<float, 4> { 0.0f, 0.0f, m_scene->cursorLeftDown ? 1.0f : 0.0f, 0.0f });
    }
    if (info.has_FRAMETIME) {
        // Feedback effects such as cursor ripple integrate per-frame decay from this uniform. The
        // parser already exposes the authored default, but runtime updates must overwrite it so the
        // simulation sees the same frame delta that drives timers and scripts.
        updateOp(G_FRAMETIME, static_cast<float>(std::max(m_scene->frameTime, 0.0)));
    }

    if (info.has_TEXELSIZE) updateOp(G_TEXELSIZE, m_texelSize);

    if (info.has_TEXELSIZEHALF)
        updateOp(G_TEXELSIZEHALF, std::array { m_texelSize[0] / 2.0f, m_texelSize[1] / 2.0f });

    if (info.has_SCREEN)
        updateOp(G_SCREEN,
                 std::array<float, 3> {
                     m_screen_size[0], m_screen_size[1], m_screen_size[0] / m_screen_size[1] });

    if (info.has_EYE_POSITION || info.has_VIEWUP || info.has_VIEWRIGHT || info.has_VIEWFORWARD) {
        // These camera basis uniforms are gated by IsModelRenderNode() during InitUniforms. Updating
        // them here gives 3D model shaders coherent camera-path lighting/reflection data without
        // introducing a new uniform contract for unrelated 2D image/effect/particle shaders.
        const auto eye = camera->GetPosition().cast<float>();
        Vector3f forward = camera->GetDirection().cast<float>();
        if (forward.norm() > 1e-6f) forward.normalize();
        Vector3f up = camera->GetUp().cast<float>();
        if (up.norm() > 1e-6f) up.normalize();
        Vector3f right = forward.cross(up);
        if (right.norm() > 1e-6f) right.normalize();

        if (info.has_EYE_POSITION)
            updateOp(G_EYE_POSITION, std::array<float, 3> { eye.x(), eye.y(), eye.z() });
        if (info.has_VIEWUP)
            updateOp(G_VIEWUP, std::array<float, 3> { up.x(), up.y(), up.z() });
        if (info.has_VIEWRIGHT)
            updateOp(G_VIEWRIGHT, std::array<float, 3> { right.x(), right.y(), right.z() });
        if (info.has_VIEWFORWARD)
            updateOp(G_VIEWFORWARD,
                     std::array<float, 3> { forward.x(), forward.y(), forward.z() });
    }

    if (info.has_PARALLAXPOSITION) {
        const auto para = ComposeParallaxPositionNdc(*m_scene, m_parallax, m_parallaxLookat,
                                                     m_parallaxLookatValid);
        updateOp(G_PARALLAXPOSITION, para);
    }

    for (size_t index = 0; index < kAudioSpectrumResolutions.size(); index++) {
        if (!info.has_audio_spectrum_left[index] && !info.has_audio_spectrum_right[index]) continue;

        const uint32_t     resolution = kAudioSpectrumResolutions[index];
        std::vector<float> left;
        std::vector<float> right;
        std::vector<float> average;
        if (m_scene->scriptHost == nullptr ||
            ! m_scene->scriptHost->GetAudioSpectrum(resolution, &left, &right, &average)) {
            left.assign(resolution, 0.0f);
            right.assign(resolution, 0.0f);
            average.assign(resolution, 0.0f);
        }

        if (info.has_audio_spectrum_left[index]) {
            updateOp(kAudioSpectrumLeftUniforms[index],
                     std::span<const float> { left.data(), left.size() });
        }
        if (info.has_audio_spectrum_right[index]) {
            updateOp(kAudioSpectrumRightUniforms[index],
                     std::span<const float> { right.data(), right.size() });
        }

    }

    if (m_scene->scriptHost) {
        m_scene->scriptHost->ApplyTextureAnimations(pNode, sprites, m_scene->frameTime);
    }

    for (auto& [i, sp] : sprites) {
        const auto& f      = sp.GetAnimateFrame(m_scene->frameTime);
        auto        grot   = WE_GLTEX_ROTATION_NAMES[i];
        auto        gtrans = WE_GLTEX_TRANSLATION_NAMES[i];
        updateOp(grot, std::array { f.xAxis[0], f.xAxis[1], f.yAxis[0], f.yAxis[1] });
        updateOp(gtrans, std::array { f.x, f.y });
    }

    if (info.has_LP || info.has_model_LCP || info.has_LCR) {
        std::array<float, 16> lights { 0 };
        std::array<float, 12> lights_color { 0 };
        std::array<float, 16> lights_color_radius { 0 };
        uint                  i = 0;
        for (auto& l : m_scene->lights) {
            if (i == 4) break;
            assert(l->node() != nullptr);
            l->node()->UpdateTrans();
            const auto modelTrans = l->node()->ModelTrans();
            lights[i * 4 + 0]     = (float)modelTrans(0, 3);
            lights[i * 4 + 1]     = (float)modelTrans(1, 3);
            lights[i * 4 + 2]     = (float)modelTrans(2, 3);
            // g_LightsColorRadius is distinct from g_LightsColorPremultiplied: Demon Core's
            // core.frag feeds rgb directly into ComputeLightSpecular and keeps the falloff radius in
            // w. Sending the radius-squared premultiplied payload here overdrives the sphere into a
            // clipped red/white blob, while color*intensity matches the shader's authored contract.
            const auto color_radius = l->colorIntensity();
            lights_color_radius[i * 4 + 0] = color_radius[0];
            lights_color_radius[i * 4 + 1] = color_radius[1];
            lights_color_radius[i * 4 + 2] = color_radius[2];
            lights_color_radius[i * 4 + 3] = l->radius();
            if (i < 3) {
                const auto& color = l->premultipliedColor();
                std::copy(color.begin(), color.end(), lights_color.begin() + i * 4);
            }
            i++;
        }
        if (info.has_LP) updateOp(G_LP, lights);
        if (info.has_LP || info.has_model_LCP) updateOp(G_LCP, lights_color);
        if (info.has_LCR) updateOp(G_LCR, lights_color_radius);
    }

    const bool has_lighting_v1 =
        info.has_LPOINT_ORIGIN || info.has_LPOINT_COLOR || info.has_LSPOT_ORIGIN ||
        info.has_LSPOT_COLOR || info.has_LSPOT_DIRECTION || info.has_LSPOT_EXPONENT ||
        info.has_LDIR_COLOR || info.has_LDIR_DIRECTION || info.has_LTUBE_ORIGINA ||
        info.has_LTUBE_ORIGINB || info.has_LTUBE_COLOR || info.has_LFEAT_SHADOW_POINT_PROJ ||
        info.has_LFEAT_SHADOW_POINT_XFORM || info.has_LFEAT_SHADOW_PROJ ||
        info.has_LFEAT_SHADOW_PROJ_XFORM;
    if (has_lighting_v1) {
        const bool shadows_on = m_scene->shadows.quality != 0;
        const Vector3f cascade_center = m_scene->ShadowCascadeCenter();
        std::vector<float> point_origin;
        std::vector<float> point_color;
        std::vector<float> point_proj;
        std::vector<float> point_xform;
        std::vector<float> spot_origin;
        std::vector<float> spot_color;
        std::vector<float> spot_direction;
        std::vector<float> spot_exponent;
        std::vector<float> dir_color;
        std::vector<float> dir_direction;
        std::vector<float> tube_a;
        std::vector<float> tube_b;
        std::vector<float> tube_color;
        std::vector<float> feat_proj;
        std::vector<float> feat_xform;

        auto append_vec4 = [](std::vector<float>& dst, float x, float y, float z, float w) {
            dst.insert(dst.end(), { x, y, z, w });
        };
        auto append_mat4 = [](std::vector<float>& dst, const Matrix4f& mat) {
            dst.insert(dst.end(), mat.data(), mat.data() + 16);
        };

        for (auto& light_ptr : m_scene->lights) {
            if (! light_ptr) continue;
            SceneLight& light = *light_ptr;
            const Vector3f origin  = light.WorldOrigin();
            const Vector3f forward = light.WorldForward();
            const Vector3f color   = light.colorIntensity();
            if (light.type() == SceneLightType::Point) {
                append_vec4(point_origin, origin.x(), origin.y(), origin.z(), light.radius());
                append_vec4(point_color, color.x(), color.y(), color.z(), light.intensity());
                if (shadows_on && light.castsShadows()) {
                    const auto proj = light.ShadowProjectionInfo();
                    const auto uv   = light.ShadowAtlasUv();
                    append_vec4(point_proj, proj.x(), proj.y(), proj.z(), proj.w());
                    append_vec4(point_xform, uv.x(), uv.y(), uv.z(), uv.w());
                } else if (shadows_on) {
                    append_vec4(point_proj, 0, 0, 0, 0);
                    append_vec4(point_xform, 0, 0, 0, 0);
                }
            } else if (light.type() == SceneLightType::Spot) {
                append_vec4(spot_origin, origin.x(), origin.y(), origin.z(),
                            std::cos(light.outerCone() * SceneLight::Deg2Rad()));
                append_vec4(spot_color, color.x(), color.y(), color.z(), light.intensity());
                append_vec4(spot_direction, forward.x(), forward.y(), forward.z(),
                            std::cos(light.innerCone() * SceneLight::Deg2Rad()));
                append_vec4(spot_exponent, light.exponent(), 0, 0, 0);
                if (shadows_on && (light.castsShadows() || light.hasCookie())) {
                    append_mat4(feat_proj, light.WorldToLightClip());
                    const auto uv = light.ShadowAtlasUv();
                    append_vec4(feat_xform, uv.x(), uv.y(), uv.z(), uv.w());
                }
            } else if (light.type() == SceneLightType::Directional) {
                append_vec4(dir_color, color.x(), color.y(), color.z(), light.intensity());
                append_vec4(dir_direction, forward.x(), forward.y(), forward.z(), 0);
                if (shadows_on && light.castsShadows()) {
                    for (int cascade = 0; cascade < 3; ++cascade) {
                        append_mat4(feat_proj,
                                    light.ShadowCascadeWorldToLightClip(
                                        cascade, cascade_center, false));
                        const auto uv = light.cascadeAtlasSlot(cascade).packed
                                            ? Eigen::Vector4f(
                                                  static_cast<float>(light.cascadeAtlasSlot(cascade).x) /
                                                      static_cast<float>(std::max(
                                                          light.cascadeAtlasSlot(cascade).atlas_w, 1)),
                                                  static_cast<float>(light.cascadeAtlasSlot(cascade).y) /
                                                      static_cast<float>(std::max(
                                                          light.cascadeAtlasSlot(cascade).atlas_h, 1)),
                                                  static_cast<float>(light.cascadeAtlasSlot(cascade).size) /
                                                      static_cast<float>(std::max(
                                                          light.cascadeAtlasSlot(cascade).atlas_w, 1)),
                                                  static_cast<float>(light.cascadeAtlasSlot(cascade).size) /
                                                      static_cast<float>(std::max(
                                                          light.cascadeAtlasSlot(cascade).atlas_h, 1)))
                                            : Eigen::Vector4f::Zero();
                        append_vec4(feat_xform, uv.x(), uv.y(), uv.z(), uv.w());
                    }
                }
            } else if (light.type() == SceneLightType::Tube) {
                append_vec4(tube_a, origin.x(), origin.y(), origin.z(), light.radius());
                append_vec4(tube_b, origin.x(), origin.y(), origin.z(), 0);
                append_vec4(tube_color, color.x(), color.y(), color.z(), light.intensity());
            }
        }

        auto push_if = [&](bool enabled, std::string_view name, const std::vector<float>& values) {
            if (! enabled || values.empty()) return;
            updateOp(name, std::span<const float> { values.data(), values.size() });
        };
        push_if(info.has_LPOINT_ORIGIN, G_LPOINT_ORIGIN, point_origin);
        push_if(info.has_LPOINT_COLOR, G_LPOINT_COLOR, point_color);
        push_if(info.has_LSPOT_ORIGIN, G_LSPOT_ORIGIN, spot_origin);
        push_if(info.has_LSPOT_COLOR, G_LSPOT_COLOR, spot_color);
        push_if(info.has_LSPOT_DIRECTION, G_LSPOT_DIRECTION, spot_direction);
        push_if(info.has_LSPOT_EXPONENT, G_LSPOT_EXPONENT, spot_exponent);
        push_if(info.has_LDIR_COLOR, G_LDIR_COLOR, dir_color);
        push_if(info.has_LDIR_DIRECTION, G_LDIR_DIRECTION, dir_direction);
        push_if(info.has_LTUBE_ORIGINA, G_LTUBE_ORIGINA, tube_a);
        push_if(info.has_LTUBE_ORIGINB, G_LTUBE_ORIGINB, tube_b);
        push_if(info.has_LTUBE_COLOR, G_LTUBE_COLOR, tube_color);
        push_if(info.has_LFEAT_SHADOW_POINT_PROJ, G_LFEAT_SHADOW_POINT_PROJ, point_proj);
        push_if(info.has_LFEAT_SHADOW_POINT_XFORM, G_LFEAT_SHADOW_POINT_XFORM, point_xform);
        push_if(info.has_LFEAT_SHADOW_PROJ, G_LFEAT_SHADOW_PROJ, feat_proj);
        push_if(info.has_LFEAT_SHADOW_PROJ_XFORM, G_LFEAT_SHADOW_PROJ_XFORM, feat_xform);
    }
}

void WPShaderValueUpdater::SetNodeData(void* nodeAddr, const WPShaderValueData& data) {
    m_nodeDataMap[nodeAddr] = data;
}

void WPShaderValueUpdater::ReplaceNodeReferences(SceneNode* old_node, SceneNode* new_node) {
    if (old_node == nullptr || new_node == nullptr || old_node == new_node) return;

    auto old_data_it = m_nodeDataMap.find(old_node);
    auto new_data_it = m_nodeDataMap.find(new_node);
    if (old_data_it != m_nodeDataMap.end() && new_data_it != m_nodeDataMap.end()) {
        PreserveDeferredRuntimeParallaxContract(
            old_data_it->second, new_data_it->second, old_node, new_node);
    }

    for (auto& [_, data] : m_nodeDataMap) {
        (void)_;
        if (data.parallax_anchor == old_node) data.parallax_anchor = new_node;
        if (data.transform_binding.parent == old_node) data.transform_binding.parent = new_node;
        if (data.effect_texture_projection.node == old_node) {
            data.effect_texture_projection.node = new_node;
        }
    }

    // Deferred materialization destroys the hidden placeholder after the real layer node is built.
    // Any remaining shader-data/cache entry keyed by the placeholder address can later match a
    // recycled allocation and send transform resolution through freed memory, so remove every
    // per-frame structure that treats the raw SceneNode pointer as an identity key.
    m_nodeDataMap.erase(old_node);
    m_nodeUniformInfoMap.erase(old_node);
    m_modelTransformCache.clear();
    m_parallaxOffsetCache.clear();
    m_attachmentTransformCache.clear();
}

const WPShaderValueData* WPShaderValueUpdater::GetNodeData(const void* node_addr) const {
    auto it = m_nodeDataMap.find(const_cast<void*>(node_addr));
    return it == m_nodeDataMap.end() ? nullptr : std::addressof(it->second);
}

WPShaderValueData* WPShaderValueUpdater::GetNodeData(const void* node_addr) {
    auto it = m_nodeDataMap.find(const_cast<void*>(node_addr));
    return it == m_nodeDataMap.end() ? nullptr : std::addressof(it->second);
}

void WPShaderValueUpdater::SetTexelSize(float x, float y) { m_texelSize = { x, y }; }
