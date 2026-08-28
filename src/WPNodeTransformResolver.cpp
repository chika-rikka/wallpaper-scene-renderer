#include "WPNodeTransformResolver.hpp"

#include "Scene/Scene.h"
#include "Scene/SceneCamera.h"
#include "Scene/SceneNode.h"
#include "Scene/SceneObject.h"
#include "WPImageAlignment.hpp"

#include <array>
#include <Eigen/Geometry>

using namespace wallpaper;
using namespace Eigen;

WPNodeTransformResolver::WPNodeTransformResolver(
    Scene& scene, const WPCameraParallax& parallax,
    Map<void*, WPShaderValueData>& node_data_map,
    Map<void*, Matrix4d>& model_transform_cache,
    Map<void*, Vector3f>& parallax_offset_cache,
    Map<void*, Affine3f>& attachment_transform_cache,
    const SceneCamera* parallax_camera,
    std::array<float, 2> parallax_lookat, uint64_t puppet_frame_serial)
    : m_scene(scene),
      m_parallax(parallax),
      m_node_data_map(node_data_map),
      m_model_transform_cache(model_transform_cache),
      m_parallax_offset_cache(parallax_offset_cache),
      m_attachment_transform_cache(attachment_transform_cache),
      m_parallax_camera(parallax_camera),
      m_parallax_lookat(parallax_lookat),
      m_puppet_frame_serial(puppet_frame_serial) {}

Matrix4d WPNodeTransformResolver::ResolveParallaxedModelTransform(SceneNode* node,
                                                                  const SceneCamera* camera,
                                                                  bool apply_parallax) {
    (void)camera;
    (void)apply_parallax;
    const auto* node_data = FindNodeData(node);
    // PATH_B / LASTPASS_DEST_STACK: Path B T is dest-STACK only
    // (0x14018b118 / 0x14018b170). Do not bake ox/oy onto FetchDest or
    // model.col(3). leftover_suppress is unofficial.
    return ResolveModelTransform(node, node_data);
}

void WPNodeTransformResolver::ApplyParallaxThroughLayerAxes(Matrix4d& model,
                                                            const Vector3f& offset) {
    if (offset.x() == 0.0f && offset.y() == 0.0f && offset.z() == 0.0f) return;
    // Not PATH_B. Official Path B T+= is dest-STACK 0x14018b118.
    model.col(3).x() += offset.x();
    model.col(3).y() += offset.y();
}

Matrix4d WPNodeTransformResolver::ResolveRawModelTransform(SceneNode* node) {
    const auto* node_data = FindNodeData(node);
    return ResolveModelTransform(node, node_data);
}

Vector3f WPNodeTransformResolver::ResolveParallaxOffset(SceneNode* node, const SceneCamera* camera) {
    // Official Path B 0x14018b062 reads ROOT +0x128/+0x170 on the object. It does
    // not consult a second node-data identity.
    const auto* node_data = FindNodeData(node);
    WPShaderValueData empty;
    return ComputeParallaxOffset(node, node_data != nullptr ? *node_data : empty, camera);
}

std::optional<Affine3f> WPNodeTransformResolver::ResolveAttachmentLocalTransform(SceneNode* node) {
    const auto* node_data = FindNodeData(node);
    if (node_data == nullptr) return std::nullopt;
    return ResolveAttachmentLocalTransform(node, *node_data);
}

bool WPNodeTransformResolver::ApplyAttachment(SceneNode* node) {
    auto local_transform = ResolveAttachmentLocalTransform(node);
    if (! local_transform.has_value()) return false;
    node->SetLocalAffine(*local_transform);
    return true;
}

void WPNodeTransformResolver::UpdateAttachmentParentIfNeeded(const WPShaderValueData& node_data) {
    if (node_data.TransformParent() == nullptr ||
        ! exists(m_node_data_map, node_data.TransformParent())) {
        return;
    }

    auto& parent_data = m_node_data_map.at(node_data.TransformParent());
    if (! parent_data.IsBoneAttached()) return;

    ApplyAttachment(node_data.TransformParent());
}

const WPShaderValueData* WPNodeTransformResolver::FindNodeData(SceneNode* node) const {
    if (node == nullptr || ! exists(m_node_data_map, node)) return nullptr;
    return std::addressof(m_node_data_map.at(node));
}

Matrix4d WPNodeTransformResolver::ResolveModelTransform(SceneNode* node,
                                                        const WPShaderValueData* node_data) {
    if (node == nullptr) return Matrix4d::Identity();
    if (exists(m_model_transform_cache, node)) return m_model_transform_cache.at(node);

    Matrix4d resolved = Matrix4d::Identity();
    if (auto* object = m_scene.FindSceneObjectForNode(node);
        object != nullptr && object->kind() != SceneObjectKind::Camera) {
        // 0x1401850a0 dest fetch. Camera +0x128 is not the view (0x14018870c).
        resolved = object->FetchDest().cast<double>();
        if (object->kind() == SceneObjectKind::Text) {
            // Text vt+0x80 0x140256e10: dest copy T += dest.R * local(+0x2f8).
            // +0x2f8 left is +0.5*(layout+0x98-layout+0x90) at 0x140257725.
            resolved = ApplyTextDestLocalOffset(resolved, node->AlignmentOffset());
        }
    } else if (node_data != nullptr && node_data->InheritsSceneParentTransform() &&
               node_data->TransformParent() != nullptr &&
               exists(m_node_data_map, node_data->TransformParent())) {
        const auto& parent_data = m_node_data_map.at(node_data->TransformParent());
        auto*       parent_node  = node_data->TransformParent();
        const auto  parent_model =
            RemoveImageAlignmentOffsetFromModel(ResolveModelTransform(parent_node, &parent_data),
                                                parent_node->AlignmentOffset());
        resolved = parent_model * node->GetLocalTrans();
    } else {
        node->UpdateTrans();
        resolved = node->ModelTrans();
    }

    m_model_transform_cache[node] = resolved;
    return resolved;
}

Vector3f WPNodeTransformResolver::ComputeParallaxOffset(SceneNode* node,
                                                        const WPShaderValueData& node_data,
                                                        const SceneCamera* camera) {
    (void)node_data;
    if (node == nullptr || ! m_parallax.enable) return Vector3f::Zero();
    // Scene camera-parallax uses the wallpaper camera for both the perspective skip and lookat.
    // An object's draw camera (including global_perspective particles) does not decide this.
    const SceneCamera* offset_camera =
        m_scene.activeCamera != nullptr ? m_scene.activeCamera : camera;
    if (offset_camera == nullptr || offset_camera->IsPerspective()) return Vector3f::Zero();
    if (exists(m_parallax_offset_cache, node)) return m_parallax_offset_cache.at(node);

    Vector3f offset = Vector3f::Zero();
    const SceneObject* object = m_scene.FindSceneObjectForNode(node);
    const SceneObject* root   = object != nullptr ? object->Root() : nullptr;
    if (root != nullptr) {
        // 0x14018b062: walk +0x180 to ROOT; (origin.xy - lookat) * amount * ROOT +0x170/+0x174
        const Vector2f origin(root->origin().x(), root->origin().y());
        const Vector2f depth(root->parallax_depth().x(), root->parallax_depth().y());
        const Vector2f lookat { m_parallax_lookat[0], m_parallax_lookat[1] };
        const Vector2f para_vec = (origin - lookat).cwiseProduct(depth) * m_parallax.amount;
        offset = Vector3f(para_vec.x(), para_vec.y(), 0.0f);
    }

    m_parallax_offset_cache[node] = offset;
    return offset;
}

std::optional<Affine3f> WPNodeTransformResolver::ResolveAttachmentLocalTransform(
    SceneNode* node, const WPShaderValueData& node_data) {
    if (node == nullptr || ! node_data.IsBoneAttached() || node_data.TransformParent() == nullptr ||
        ! exists(m_node_data_map, node_data.TransformParent())) {
        return std::nullopt;
    }

    if (exists(m_attachment_transform_cache, node)) {
        return m_attachment_transform_cache.at(node);
    }

    auto* parent_node = node_data.TransformParent();
    auto& parent_data = m_node_data_map.at(parent_node);
    if (parent_data.IsBoneAttached()) {
        auto parent_transform = ResolveAttachmentLocalTransform(parent_node, parent_data);
        if (! parent_transform.has_value()) return std::nullopt;
        parent_node->SetLocalAffine(*parent_transform);
    }

    if (! parent_data.puppet_layer.hasPuppet()) return std::nullopt;

    parent_data.puppet_layer.AdvanceIfNeeded(m_scene.frameTime, m_puppet_frame_serial);
    const auto* parent_puppet = parent_data.puppet_layer.Puppet();
    if (parent_puppet == nullptr) return std::nullopt;

    // MDAT attachment locators are bone-local. The multiplication order below is the coordinate
    // contract: the current animated bone frame moves the authored locator, then the child keeps
    // its own local origin/rotation/scale. Treating bind_transform as model space, or premultiplying
    // it by inverse(bindBoneModel), double-converts the locator and breaks non-root attachment points.
    Affine3f local_transform =
        parent_puppet->BoneModelTransform(node_data.transform_binding.bone_index) *
        node_data.transform_binding.bind_transform * node_data.transform_binding.local_transform;
    m_attachment_transform_cache[node] = local_transform;
    return local_transform;
}
