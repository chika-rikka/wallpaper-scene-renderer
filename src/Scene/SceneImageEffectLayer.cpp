#include "SceneImageEffectLayer.h"
#include "SceneNode.h"
#include "Scene.h"
#include "SceneMesh.h"

#include "SpecTexs.hpp"
#include "Core/StringHelper.hpp"
#include "Utils/Logging.h"

#include <algorithm>
#include <limits>

using namespace wallpaper;

namespace
{
std::string ResolvePingPongInputAlias(std::string_view value, std::string_view ppong_a,
                                      std::string_view ppong_b) {
    if (sstart_with(value, WE_EFFECT_PPONG_PREFIX_A)) return std::string(ppong_a);
    if (sstart_with(value, WE_EFFECT_PPONG_PREFIX_B)) return std::string(ppong_b);
    return std::string(value);
}

std::string ResolvePingPongOutputAlias(std::string_view value, std::string_view ppong_a,
                                       std::string_view ppong_b) {
    if (value == SpecTex_Default) return std::string(ppong_b);
    return ResolvePingPongInputAlias(value, ppong_a, ppong_b);
}

bool IsCurrentEffectOutput(std::string_view authored_output) {
    return authored_output == SpecTex_Default ||
        sstart_with(authored_output, WE_EFFECT_PPONG_PREFIX_B);
}

std::string_view ResolveTemplateOrCurrent(const std::string& authored_value,
                                          const std::string& current_value) {
    return authored_value.empty() ? std::string_view(current_value)
                                  : std::string_view(authored_value);
}
} // namespace

std::string_view wallpaper::FinalOutputCapabilityName(FinalOutputCapability capability) {
    switch (capability) {
    case FinalOutputCapability::PrivateDependency: return "private-dependency";
    case FinalOutputCapability::PrivatePuppetSurface: return "private-puppet-surface";
    case FinalOutputCapability::SceneAuthoredWriter: return "scene-authored-writer";
    case FinalOutputCapability::PrivateThenPublish: return "private-then-publish";
    }
    return "unknown";
}

namespace
{
struct EffectOutputDiagnosticRoles {
    std::string_view authored_writer;
    std::string_view publication_writer;
    std::string_view private_parallax_owner;
    std::string_view publication_parallax_owner;
    uint32_t         parallax_application_count { 1 };
};

EffectOutputDiagnosticRoles ResolveEffectOutputDiagnosticRoles(
    FinalOutputCapability capability,
    bool keep_authored_final_private,
    bool has_authored_final_output) {
    if (!has_authored_final_output) {
        return { "none", "neutral-composite", "none", "neutral-composite", 1 };
    }
    if (!keep_authored_final_private) {
        return { "scene-authored-final", "authored-final", "none", "authored-final", 1 };
    }
    if (capability == FinalOutputCapability::PrivatePuppetSurface) {
        return { "private-puppet-surface", "neutral-composite", "none", "neutral-composite", 1 };
    }
    return { "private-authored-final", "neutral-composite", "none", "neutral-composite", 1 };
}

std::shared_ptr<SceneMesh> BuildPuppetPublicationMesh(const PuppetSurfaceProjection& projection) {
    auto mesh = std::make_shared<SceneMesh>(true);
    const float left = projection.surface_bounds.min.x();
    const float right = projection.surface_bounds.max.x();
    const float bottom = projection.surface_bounds.min.y();
    const float top = projection.surface_bounds.max.y();
    const std::array<float, 12> positions {
        left, bottom, 0.0f, left, top, 0.0f, right, bottom, 0.0f, right, top, 0.0f,
    };
    const std::array<float, 8> texcoords { 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 0.0f };
    SceneVertexArray vertex(
        { { std::string(WE_IN_POSITION), VertexType::FLOAT3 },
          { std::string(WE_IN_TEXCOORD), VertexType::FLOAT2 } },
        4);
    vertex.SetVertex(WE_IN_POSITION, positions);
    vertex.SetVertex(WE_IN_TEXCOORD, texcoords);
    mesh->AddVertexArray(std::move(vertex));
    mesh->SetDirty();
    return mesh;
}

PuppetBounds3D TransformPuppetBounds(const PuppetBounds3D& bounds,
                                     const Eigen::Affine3f& transform) {
    PuppetBounds3D result;
    if (!bounds.IsFiniteAndOrdered()) return result;
    for (int x = 0; x < 2; x++) {
        for (int y = 0; y < 2; y++) {
            for (int z = 0; z < 2; z++) {
                result.Include(transform * Eigen::Vector3f {
                    x == 0 ? bounds.min.x() : bounds.max.x(),
                    y == 0 ? bounds.min.y() : bounds.max.y(),
                    z == 0 ? bounds.min.z() : bounds.max.z(),
                });
            }
        }
    }
    return result;
}

PuppetBounds3D ComputeSkinnedPuppetBounds(const SceneMesh& mesh,
                                          std::span<const Eigen::Affine3f> skinning) {
    PuppetBounds3D bounds;
    if (mesh.VertexCount() == 0 || skinning.empty()) return bounds;

    const auto& vertex_array = mesh.GetVertexArray(0);
    const auto offsets = vertex_array.GetAttrOffsetMap();
    const auto position_it = offsets.find(std::string(WE_IN_POSITION));
    const auto indices_it = offsets.find(std::string(WE_IN_BLENDINDICES));
    const auto weights_it = offsets.find(std::string(WE_IN_BLENDWEIGHTS));
    if (position_it == offsets.end() || indices_it == offsets.end() ||
        weights_it == offsets.end() || vertex_array.Data() == nullptr) {
        return bounds;
    }

    const usize stride = vertex_array.OneSize();
    const usize position_offset = position_it->second.offset / sizeof(float);
    const usize indices_offset = indices_it->second.offset / sizeof(float);
    const usize weights_offset = weights_it->second.offset / sizeof(float);
    const float* data = vertex_array.Data();
    for (usize vertex_index = 0; vertex_index < vertex_array.VertexCount(); vertex_index++) {
        const usize base = vertex_index * stride;
        const Eigen::Vector3f bind_position(data + base + position_offset);
        const auto* blend_indices = reinterpret_cast<const uint32_t*>(data + base + indices_offset);
        const float* blend_weights = data + base + weights_offset;
        Eigen::Vector3f skinned = Eigen::Vector3f::Zero();
        float total_weight = 0.0f;
        for (usize influence = 0; influence < 4; influence++) {
            const float weight = blend_weights[influence];
            if (weight <= 0.0f || blend_indices[influence] >= skinning.size()) continue;
            skinned += weight * (skinning[blend_indices[influence]] * bind_position);
            total_weight += weight;
        }
        bounds.Include(total_weight > 0.0f ? skinned : bind_position);
    }
    return bounds;
}

void UpdatePuppetProjectionDerivedValues(PuppetSurfaceProjection& projection) {
    const float minimum_extent = std::numeric_limits<float>::epsilon();
    const float authored_width = std::max(projection.authored_layer_size.x(), minimum_extent);
    const float authored_height = std::max(projection.authored_layer_size.y(), minimum_extent);
    const float camera_width = std::max(projection.surface_bounds.max.x() -
                                            projection.surface_bounds.min.x(),
                                        minimum_extent);
    const float camera_height = std::max(projection.surface_bounds.max.y() -
                                             projection.surface_bounds.min.y(),
                                         minimum_extent);
    const float authored_left = -authored_width * 0.5f;
    const float authored_top = authored_height * 0.5f;
    projection.source_to_layer = Eigen::Vector4f {
        (authored_left - projection.surface_bounds.min.x()) / camera_width,
        (projection.surface_bounds.max.y() - authored_top) / camera_height,
        authored_width / camera_width,
        authored_height / camera_height,
    };
    const float density = std::max(projection.render_density, minimum_extent);
    projection.target_extent = {
        std::max(1, static_cast<int32_t>(std::ceil(camera_width * density))),
        std::max(1, static_cast<int32_t>(std::ceil(camera_height * density))),
    };
    projection.publication_mesh = BuildPuppetPublicationMesh(projection);
}
} // namespace

// Official image +0x2f0/+0x2f4 (0x1401ec338). Keep authored size for g_ETVP.
SceneImageEffectLayer::SceneImageEffectLayer(SceneNode* node, float w, float h,
                                             std::string_view pingpong_a,
                                             std::string_view pingpong_b)
    : m_layer_node(node),
      m_pingpong_a(pingpong_a),
      m_pingpong_b(pingpong_b),
      m_authored_width(w),
      m_authored_height(h),
      m_source_mesh(std::make_unique<SceneMesh>()),
      m_final_mesh(std::make_unique<SceneMesh>()),
      m_final_node(std::make_unique<SceneNode>()) {};

FinalOutputCapability
SceneImageEffectLayer::ResolveFinalOutputCapability(bool dependency_route) const {
    // The order is intentional: a dependency must remain sampleable, and a puppet surface must
    // remain private until its neutral publisher applies scene-space transforms. Runtime/source-less
    // visibility is next because a direct writer can leave stale framebuffer contents when skipped.
    if (dependency_route || m_final_output_capability == FinalOutputCapability::PrivateDependency) {
        return FinalOutputCapability::PrivateDependency;
    }
    if (m_final_output_capability == FinalOutputCapability::PrivatePuppetSurface) {
        return FinalOutputCapability::PrivatePuppetSurface;
    }
    if (HasRuntimeVisibilityContract() ||
        m_final_composite.hidden_policy == HiddenFinalCompositePolicy::SuppressOutput) {
        return FinalOutputCapability::PrivateThenPublish;
    }
    return m_final_output_capability;
}

void SceneImageEffectLayer::SetPuppetSurfaceProjection(PuppetSurfaceProjection projection) {
    if (!projection.asset_bounds.IsFiniteAndOrdered() ||
        !projection.authored_pose_bounds.IsFiniteAndOrdered() ||
        !projection.surface_bounds.IsFiniteAndOrdered()) {
        LOG_ERROR("ScenePuppetProjectionInit: layer=%d has invalid asset/authored/surface bounds",
                  m_layer_node != nullptr ? m_layer_node->ID() : -1);
        return;
    }
    UpdatePuppetProjectionDerivedValues(projection);
    projection.surface_revision = 1;
    projection.camera_revision = projection.surface_revision;
    projection.target_revision = projection.surface_revision;
    projection.publication_mesh_revision = projection.surface_revision;
    m_puppet_surface_projection = std::move(projection);
}

bool SceneImageEffectLayer::PreparePuppetSurface(Scene& scene, const SceneMesh& skinned_mesh,
                                                 const PuppetPoseSnapshot& pose,
                                                 uint64_t frame_serial) {
    if (!m_puppet_surface_projection.has_value() ||
        pose.domain != PuppetPoseDomain::RuntimeMutable) {
        return false;
    }
    auto& projection = *m_puppet_surface_projection;
    if (projection.last_bounds_frame_serial == frame_serial &&
        projection.last_pose_revision == pose.revision) {
        return false;
    }

    projection.last_bounds_frame_serial = frame_serial;
    projection.last_pose_revision = pose.revision;
    const PuppetBounds3D observed = ComputeSkinnedPuppetBounds(skinned_mesh, pose.skinning);
    if (!observed.IsFiniteAndOrdered()) {
        LOG_ERROR("ScenePuppetSurfacePrepare: layer=%d frame=%llu revision=%llu produced invalid "
                  "runtime bounds",
                  m_layer_node != nullptr ? m_layer_node->ID() : -1,
                  static_cast<unsigned long long>(frame_serial),
                  static_cast<unsigned long long>(pose.revision));
        return false;
    }
    projection.observed_runtime_bounds = observed;

    const PuppetBounds3D transformed = TransformPuppetBounds(observed,
                                                              projection.geometry_transform);
    const float guard = 1.0f / std::max(projection.render_density,
                                        std::numeric_limits<float>::epsilon());
    const Eigen::Vector3f guarded_min = transformed.min - Eigen::Vector3f::Constant(guard);
    const Eigen::Vector3f guarded_max = transformed.max + Eigen::Vector3f::Constant(guard);
    const std::array<bool, 6> exceeded_axes {
        guarded_min.x() < projection.surface_bounds.min.x(),
        guarded_max.x() > projection.surface_bounds.max.x(),
        guarded_min.y() < projection.surface_bounds.min.y(),
        guarded_max.y() > projection.surface_bounds.max.y(),
        guarded_min.z() < projection.surface_bounds.min.z(),
        guarded_max.z() > projection.surface_bounds.max.z(),
    };
    if (std::none_of(exceeded_axes.begin(), exceeded_axes.end(), [](bool exceeded) {
            return exceeded;
        })) {
        return false;
    }

    const std::array<int32_t, 2> old_extent = projection.target_extent;
    projection.surface_bounds.min = projection.surface_bounds.min.cwiseMin(guarded_min);
    projection.surface_bounds.max = projection.surface_bounds.max.cwiseMax(guarded_max);
    projection.contract_exceeded = true;
    projection.surface_revision++;
    UpdatePuppetProjectionDerivedValues(projection);
    projection.camera_revision = projection.surface_revision;
    projection.target_revision = projection.surface_revision;
    projection.publication_mesh_revision = projection.surface_revision;

    auto camera_it = scene.cameras.find(projection.camera_name);
    if (camera_it != scene.cameras.end() && camera_it->second != nullptr) {
        camera_it->second->SetOrthographicViewRect(projection.surface_bounds.min.x(),
                                                   projection.surface_bounds.max.x(),
                                                   projection.surface_bounds.min.y(),
                                                   projection.surface_bounds.max.y());
    }
    const bool extent_changed = old_extent != projection.target_extent;
    auto target_it = scene.renderTargets.find(projection.target_name);
    if (target_it != scene.renderTargets.end()) {
        target_it->second.width = projection.target_extent[0];
        target_it->second.height = projection.target_extent[1];
        target_it->second.mapWidth = projection.target_extent[0];
        target_it->second.mapHeight = projection.target_extent[1];
        if (extent_changed) scene.MarkRenderTargetResourcesDirty(projection.target_name);
    }

    SyncResolvedOutputMesh();
    return true;
}

void SceneImageEffect::SetIdentity(int32_t owner_layer_id, int32_t effect_id,
                                   uint32_t effect_index, std::string effect_name) {
    m_owner_layer_id = owner_layer_id;
    m_effect_id      = effect_id;
    m_effect_index   = effect_index;
    m_effect_name    = std::move(effect_name);
}

void SceneImageEffect::SetLocalVisible(bool visible) {
    m_local_visible = visible;
    for (auto& node : nodes) {
        if (node.sceneNode != nullptr) {
            // Effect-local visibility is intentionally separate from layer visibility. The layer
            // tree still owns parent/child propagation through SceneNode::SetLayerVisible(), while
            // this flag lets scripts and animations disable only this effect without rebuilding
            // the render graph or hiding the owner layer.
            node.sceneNode->SetLocalVisible(visible);
        }
    }
}

void SceneImageEffect::SetBypassTargets(std::string src, std::string dst) {
    m_bypass_src = std::move(src);
    m_bypass_dst = std::move(dst);
}

bool SceneImageEffectLayer::HasFinalComposite() const {
    return m_final_node != nullptr && m_final_node->HasMaterial();
}

bool SceneImageEffectLayer::HasRuntimeVisibilityContract() const {
    for (const auto& effect : m_effects) {
        if (effect != nullptr && effect->HasRuntimeVisibilityContract()) return true;
    }
    return false;
}

bool SceneImageEffectLayer::HasVisibleRuntimeVisibilityContribution() const {
    for (const auto& effect : m_effects) {
        if (effect != nullptr && effect->HasRuntimeVisibilityContract() &&
            effect->LocalVisible()) {
            return true;
        }
    }
    return false;
}

bool SceneImageEffectLayer::HasVisibleSourceLessContribution() const {
    if (m_final_composite.output_effect == nullptr) return false;

    if (HasRuntimeVisibilityContract()) {
        // Source-less compose helpers often model a user-selected generator effect followed by
        // always-visible filters such as fisheye or scroll. The filters are not valid sources on
        // their own; when every runtime-selected source effect is hidden, publishing the final
        // filter output would expose an empty or stale helper target as a rectangular quad.
        return HasVisibleRuntimeVisibilityContribution();
    }

    // Static source-less helpers without runtime visibility still need to draw their authored
    // chain normally. In that simpler contract the final effect's own visibility remains the best
    // indication that the chain currently contributes visible pixels.
    return m_final_composite.output_effect->LocalVisible();
}

bool SceneImageEffectLayer::ShouldRunFinalComposite() const {
    if (!HasFinalComposite()) return false;

    // The neutral final composite has three explicit roles. It is the stable publisher for ordinary
    // private-authored chains, the conditional publisher for source-less generator helpers, and the
    // hidden-effect bypass for layers whose authored final shader normally writes the visible target.
    // Keeping those roles state-driven prevents an AuthoredWriter layer from being drawn twice while
    // retaining the independent publisher required by normal image/text and color-blend-only layers.
    if (m_final_composite.publishes_visible_output) {
        return HasVisibleSourceLessContribution();
    }
    if (m_final_composite.publishes_private_output) {
        if (m_final_composite.hidden_policy == HiddenFinalCompositePolicy::SuppressOutput) {
            return HasVisibleSourceLessContribution();
        }
        return true;
    }

    // A configured composite with no authored output node is itself the only publisher. This occurs
    // for contracts such as shader color blending without an authored effect list.
    if (m_final_composite.output_effect == nullptr) return true;
    if (m_final_composite.output_effect->LocalVisible()) return false;
    return m_final_composite.hidden_policy == HiddenFinalCompositePolicy::PreserveSource;
}

void SceneImageEffectLayer::SetFinalCompositeSource(std::string source) {
    if (!HasFinalComposite()) return;

    auto* material = m_final_node->Mesh()->Material();
    if (material == nullptr) return;

    // The final composite is deliberately separate from every authored effect shader. Depending on
    // the layer publication contract it is either the stable publication boundary for the resolved
    // ping-pong texture or the dormant bypass publisher used only when an AuthoredWriter is hidden.
    if (material->textures.empty()) material->textures.resize(1);
    material->textures[0] = std::move(source);
}

void SceneImageEffectLayer::SyncResolvedOutputMesh() {
    if (m_resolved_output_node != nullptr && m_resolved_output_node->Mesh() != nullptr) {
        if (m_resolved_output_mesh_follows_final_mesh) {
            // Resource-only render-graph refreshes keep the already-resolved effect nodes alive and
            // only recreate their GPU resources. Runtime text updates therefore cannot rely on
            // ResolveEffect() running again to copy `m_final_mesh` into the currently active output
            // node. Synchronizing the resolved node mesh here keeps effect-backed text quads
            // visually in lockstep with the latest runtime map-rate/size changes even when the pass
            // topology is intentionally reused.
            m_resolved_output_node->Mesh()->ChangeMeshDataFrom(*m_final_mesh);
            // `ChangeMeshDataFrom()` shares the CPU-side mesh payload but does not flip the
            // render-pass dirty flag. Resource-only refreshes look at the live pass mesh, not at
            // `m_final_mesh`, so the resolved output node must be marked dirty explicitly or Vulkan
            // keeps drawing the stale vertex buffer even though the debug logs already show the
            // updated quad geometry.
            m_resolved_output_node->Mesh()->SetDirty();
        } else {
            // Private dependency outputs deliberately use the effect camera's unit fullscreen mesh.
            // Runtime resize/transform refreshes should not copy the layer's world-space final mesh
            // into that node, otherwise offscreen dependencies render as if they were visible scene
            // quads. The neutral final composite below is a separate publisher and still has to
            // follow its own mesh policy.
        }
    }
    if (HasFinalComposite() && m_final_node->Mesh() != nullptr) {
        // The neutral final composite has its own node whether it is acting as a hidden fallback or
        // as the source-less helper publisher. Keep that mesh synchronized as well; otherwise a
        // runtime text/image resize could fix the authored path while leaving this publisher with
        // stale geometry.
        const SceneMesh* publication_mesh = nullptr;
        if (m_final_composite.uses_source_mesh && m_puppet_surface_projection.has_value() &&
            m_puppet_surface_projection->publication_mesh != nullptr) {
            publication_mesh = m_puppet_surface_projection->publication_mesh.get();
        } else if (m_final_composite.uses_source_mesh) {
            publication_mesh = m_source_mesh.get();
        } else {
            publication_mesh = m_final_mesh.get();
        }
        if (publication_mesh != nullptr) {
            m_final_node->Mesh()->ChangeMeshDataFrom(*publication_mesh);
        }
        m_final_node->Mesh()->SetDirty();
    }
}

void SceneImageEffectLayer::SyncResolvedNodeToWorld() {
    if (m_layer_node == nullptr) return;

    m_layer_node->UpdateTrans();
    // Final effect nodes are emitted as render-graph-only nodes, not as real children of the
    // authored scene node. Copying only the local TRS loses virtual parent transforms used by
    // render-order proxy groups such as Wallpaper Engine compose layers. Resolve the full world
    // matrix here so the final screen writer lands in the same place as the authored layer.
    Eigen::Affine3f world_affine;
    world_affine.matrix() = m_layer_node->ModelTrans().cast<float>();
    m_final_node->SetLocalAffine(world_affine);
    m_final_node->UpdateTrans();

    if (m_resolved_output_node != nullptr && m_resolved_output_follows_world) {
        m_resolved_output_node->CopyTrans(*m_final_node);
        m_resolved_output_node->UpdateTrans();
    }
}

void SceneImageEffectLayer::SyncResolvedNodeToMatrix(const Eigen::Affine3f& world_affine) {
    m_final_node->SetLocalAffine(world_affine);
    m_final_node->UpdateTrans();

    if (m_resolved_output_node != nullptr && m_resolved_output_follows_world) {
        m_resolved_output_node->CopyTrans(*m_final_node);
        m_resolved_output_node->UpdateTrans();
    }
}

void SceneImageEffectLayer::SyncResolvedNodeForRoute(
    const Eigen::Affine3f* resolved_world_affine) {
    if (resolved_world_affine != nullptr) {
        // Render-order proxy routes keep some authored children root-owned in the physical
        // SceneNode tree while still drawing them under a virtual parent. When the render graph
        // already resolved that routed world matrix, trust it here instead of asking the node's
        // physical parent chain, which would drop the virtual parent transform.
        SyncResolvedNodeToMatrix(*resolved_world_affine);
        return;
    }
    SyncResolvedNodeToWorld();
}

SceneImageEffectNode* SceneImageEffectLayer::ResolveEffectPingPongChain(
    const SceneMesh& default_mesh,
    SceneNode& default_node,
    std::string_view effect_cam,
    std::string_view& ppong_a,
    std::string_view& ppong_b) {
    SceneImageEffectNode* fallback_last_output { nullptr };

    for (auto& eff : m_effects) {
        // Each effect consumes the current input ping-pong target and normally writes the next
        // output ping-pong target. Capturing that pair after alias resolution gives the renderer a
        // topology-stable hidden path: when this effect is locally hidden, a conditional copy moves
        // input to output so later effects observe the correct current frame instead of the last
        // frame produced while the effect was visible.
        eff->SetBypassTargets(std::string(ppong_a), std::string(ppong_b));
        for (auto& cmd : eff->commands) {
            const auto authored_src = ResolveTemplateOrCurrent(cmd.authored_src, cmd.src);
            const auto authored_dst = ResolveTemplateOrCurrent(cmd.authored_dst, cmd.dst);
            cmd.src                 = ResolvePingPongInputAlias(authored_src, ppong_a, ppong_b);
            cmd.dst                 = ResolvePingPongInputAlias(authored_dst, ppong_a, ppong_b);
        }

        for (auto it = eff->nodes.begin(); it != eff->nodes.end(); it++) {
            const auto authored_output = ResolveTemplateOrCurrent(it->authored_output, it->output);
            it->output                 = ResolvePingPongOutputAlias(authored_output, ppong_a, ppong_b);
            if (IsCurrentEffectOutput(authored_output)) {
                fallback_last_output = &(*it);
                m_final_composite.output_effect = eff.get();
            }

            assert(it->sceneNode->HasMaterial());

            auto& material = *(it->sceneNode->Mesh()->Material());
            material.blenmode = BlendMode::Normal;
            it->sceneNode->SetCamera(effect_cam.data());
            it->camera_override.clear();
            it->use_active_camera_for_parallax = false;
            it->use_identity_model             = false;
            it->clear_before_draw              = false;
            it->alpha_write_policy             = AlphaWritePolicy::Preserve;
            it->sceneNode->CopyTrans(default_node);
            it->sceneNode->Mesh()->ChangeMeshDataFrom(default_mesh);

            auto& texs = material.textures;
            if (!it->authored_textures.empty()) {
                texs = it->authored_textures;
            }
            for (auto& texture : texs) {
                texture = ResolvePingPongInputAlias(texture, ppong_a, ppong_b);
            }
        }

        std::swap(ppong_a, ppong_b);
    }

    return fallback_last_output;
}

SceneImageEffectLayer::FinalOutputResolveDecision
SceneImageEffectLayer::ResolveFinalOutputDecision(
    SceneImageEffectNode* fallback_last_output,
    std::string_view layer_surface_cam,
    bool keep_final_output_private,
    FinalOutputCapability output_capability) {
    FinalOutputResolveDecision decision;

    const bool source_less_final_output =
        m_final_composite.hidden_policy == HiddenFinalCompositePolicy::SuppressOutput &&
        m_final_composite.output_effect != nullptr;
    const bool publish_private_final_composite = keep_final_output_private ||
        output_capability != FinalOutputCapability::SceneAuthoredWriter;
    decision.keep_authored_final_private =
        keep_final_output_private || source_less_final_output || publish_private_final_composite;

    m_final_composite.publishes_visible_output =
        source_less_final_output && !keep_final_output_private;
    m_final_composite.publishes_private_output = publish_private_final_composite;
    decision.private_final_uses_layer_surface =
        m_final_output_capability == FinalOutputCapability::PrivatePuppetSurface &&
        fallback_last_output != nullptr &&
        fallback_last_output->private_final_output_uses_layer_surface && !m_fullscreen &&
        !layer_surface_cam.empty();
    m_final_composite.uses_source_mesh = decision.private_final_uses_layer_surface;
    // Official translucent dest is SRC_ALPHA/INV_SRC_ALPHA (GFX_BLEND_DESC
    // 0x14009a32b), not a second premul publisher.
    m_final_composite.samples_premultiplied_source = false;

    return decision;
}

void SceneImageEffectLayer::ResolveFinalCompositeNode(
    const SceneMesh& default_mesh,
    SceneNode& default_node,
    std::string_view effect_cam,
    std::string_view final_output,
    std::string_view final_composite_source,
    const Eigen::Affine3f* resolved_world_affine) {
    if (!HasFinalComposite()) return;

    // The independent material is prepared for both publication contracts. Private-authored layers
    // use it as their normal layer-surface publisher; AuthoredWriter layers keep it dormant until a
    // hidden final effect needs the current bypassed ping-pong source to remain visible.
    SetFinalCompositeSource(std::string(final_composite_source));
    auto& mesh     = *m_final_node->Mesh();
    auto& material = *mesh.Material();
    if (m_fullscreen) {
        // The synthetic fallback uses a generic image shader with an MVP uniform. Fullscreen
        // postprocess layers are 2x2 clip-space quads, so the fallback must stay on the effect
        // camera/default mesh path; routing it through the active scene camera would shrink the
        // hidden-effect passthrough into world units instead of covering the framebuffer.
        material.blenmode = BlendMode::Normal;
        m_final_node->SetCamera(effect_cam.data());
        m_final_node->CopyTrans(default_node);
        mesh.ChangeMeshDataFrom(default_mesh);
        LOG_INFO("SceneEffectFinalCompositeResolve: layer=%d name='%s' fullscreen=true "
                 "camera='%.*s' output='%s' source='%s' blend=%d",
                 m_layer_node != nullptr ? m_layer_node->ID() : -1,
                 m_layer_node != nullptr ? m_layer_node->Name().c_str() : "",
                 static_cast<int>(effect_cam.size()),
                 effect_cam.data(),
                 std::string(final_output).c_str(),
                 std::string(final_composite_source).c_str(),
                 static_cast<int>(material.blenmode));
        return;
    }

    SyncResolvedNodeForRoute(resolved_world_affine);
    material.blenmode = m_final_blend;
    m_final_node->SetCamera(std::string());
    const SceneMesh* publication_mesh = nullptr;
    if (m_final_composite.uses_source_mesh && m_puppet_surface_projection.has_value() &&
        m_puppet_surface_projection->publication_mesh != nullptr) {
        publication_mesh = m_puppet_surface_projection->publication_mesh.get();
    } else if (m_final_composite.uses_source_mesh) {
        publication_mesh = m_source_mesh.get();
    } else {
        publication_mesh = m_final_mesh.get();
    }
    if (publication_mesh != nullptr) mesh.ChangeMeshDataFrom(*publication_mesh);
    LOG_INFO("SceneEffectFinalCompositeResolve: layer=%d name='%s' fullscreen=false "
             "camera='' output='%s' source='%s' blend=%d publish=%s "
             "publish-private=%s source-mesh=%s policy=%d",
             m_layer_node != nullptr ? m_layer_node->ID() : -1,
             m_layer_node != nullptr ? m_layer_node->Name().c_str() : "",
             std::string(final_output).c_str(),
             std::string(final_composite_source).c_str(),
             static_cast<int>(material.blenmode),
             m_final_composite.publishes_visible_output ? "true" : "false",
             m_final_composite.publishes_private_output ? "true" : "false",
             m_final_composite.uses_source_mesh ? "true" : "false",
             static_cast<int>(m_final_composite.hidden_policy));
}

void SceneImageEffectLayer::ResolveVisibleFinalOutput(
    SceneImageEffectNode& final_output_node,
    const SceneMesh& default_mesh,
    SceneNode& default_node,
    std::string_view effect_cam,
    std::string_view final_output,
    const Eigen::Affine3f* resolved_world_affine) {
    // Keep the historical visible path: the final authored shader writes the screen/default
    // output directly, preserving shaders whose visual result depends on being the final
    // compositor. The synthetic final composite remains dormant unless this effect is hidden.
    m_resolved_output_node = final_output_node.sceneNode.get();
    final_output_node.output = std::string(final_output);
    auto& mesh               = *(final_output_node.sceneNode->Mesh());
    auto& material           = *mesh.Material();
    if (m_fullscreen) {
        // Fullscreen postprocess final passes already draw the unit effect mesh and may still
        // multiply by g_ModelViewProjectionMatrix. Keep that final pass on the effect camera so
        // the 2x2 utility quad remains a full-frame clip-space composite instead of becoming a
        // tiny active-camera world quad that leaves the effect effectively invisible.
        m_resolved_output_follows_world = false;
        m_resolved_output_mesh_follows_final_mesh = false;
        material.blenmode = BlendMode::Normal;
        final_output_node.sceneNode->SetCamera(effect_cam.data());
        final_output_node.sceneNode->CopyTrans(default_node);
        mesh.ChangeMeshDataFrom(default_mesh);
        LOG_INFO("SceneEffectFinalOutputResolve: layer=%d name='%s' fullscreen=true "
                 "camera='%.*s' output='%s' material='%s' blend=%d",
                 m_layer_node != nullptr ? m_layer_node->ID() : -1,
                 m_layer_node != nullptr ? m_layer_node->Name().c_str() : "",
                 static_cast<int>(effect_cam.size()),
                 effect_cam.data(),
                 std::string(final_output).c_str(),
                 material.name.c_str(),
                 static_cast<int>(material.blenmode));
        return;
    }

    SyncResolvedNodeForRoute(resolved_world_affine);
    material.blenmode = m_final_blend;
    final_output_node.sceneNode->SetCamera(std::string());
    final_output_node.sceneNode->CopyTrans(*m_final_node);
    mesh.ChangeMeshDataFrom(*m_final_mesh);
    LOG_INFO("SceneEffectFinalOutputResolve: layer=%d name='%s' fullscreen=false "
             "camera='' output='%s' material='%s' blend=%d private=false",
             m_layer_node != nullptr ? m_layer_node->ID() : -1,
             m_layer_node != nullptr ? m_layer_node->Name().c_str() : "",
             std::string(final_output).c_str(),
             material.name.c_str(),
             static_cast<int>(material.blenmode));
}

void SceneImageEffectLayer::ResolvePrivateFinalOutput(
    SceneImageEffectNode& final_output_node,
    const SceneMesh& default_mesh,
    SceneNode& default_node,
    std::string_view effect_cam,
    std::string_view layer_surface_cam,
    bool private_final_uses_layer_surface,
    const Eigen::Affine3f* resolved_world_affine) {
    // Some final authored effects must remain private even when their owner is a visible screen
    // layer. Hidden dependency sources need this because consumers sample the resolved private
    // texture. Source-less passthrough helpers need the same topology because their base image
    // is intentionally empty; a stable neutral composite can then publish only real visible
    // source contributions and suppress the chain when user-selected generator effects are off.
    m_resolved_output_node = final_output_node.sceneNode.get();
    m_resolved_output_follows_world = false;
    m_resolved_output_mesh_follows_final_mesh = private_final_uses_layer_surface;
    SyncResolvedNodeForRoute(resolved_world_affine);
    auto& mesh     = *(final_output_node.sceneNode->Mesh());
    auto& material = *mesh.Material();
    if (private_final_uses_layer_surface) {
        // Composition layers publish child effects through a parent source texture before the
        // parent distortion/opacity chain runs. For animated puppet image layers, the last
        // authored pass is not a fullscreen postprocess: it is a synthetic layer-surface writer
        // that applies skinning while sampling the previous private effect result. Keep that
        // pass in the child layer's local source camera so the animated pose is baked into the
        // private texture, then let the neutral composite place that texture into the parent
        // composition source. This keeps parent distortion/opacity effects from flattening
        // layer-surface puppet motion.
        // The private layer-surface writer draws the authored puppet mesh into a freshly
        // cleared local target. Keeping the original layer blend is important for meshes with
        // overlapping translucent triangles: a forced Normal blend lets later transparent
        // fragments overwrite already rendered puppet fragments. Source-over alpha keeps the
        // cleared target deterministic without making transparent fragments erase earlier
        // fragments from the same draw.
        material.blenmode = m_final_blend;
        if (m_puppet_surface_projection.has_value() &&
            !m_puppet_surface_projection->target_name.empty()) {
            // The skinned layer surface owns a target distinct from the effect ping-pong chain.
            // Intermediate effects retain the authored source resolution and UV domain, while this
            // final raster alone expands to the animation envelope. Auxiliary puppet textures and
            // masks therefore keep their authored UVs unchanged.
            final_output_node.output = m_puppet_surface_projection->target_name;
        }
        final_output_node.sceneNode->SetCamera(std::string());
        final_output_node.camera_override = std::string(layer_surface_cam);
        final_output_node.use_active_camera_for_parallax = false;
        // Official 0x1401ec799: I-internal writes I. This writer is that pass
        // (0x14020805e [instance+0x400]); dest publication stays FetchDest.
        final_output_node.use_identity_model = true;
        final_output_node.alpha_write_policy = AlphaWritePolicy::SourceOver;
        // The private layer-surface target is freshly cleared before the skinned draw so coverage
        // outside the current pose never survives from an earlier animation frame.
        final_output_node.clear_before_draw = true;
        final_output_node.sceneNode->CopyTrans(default_node);
        mesh.ChangeMeshDataFrom(*m_final_mesh);
        LOG_INFO("SceneEffectFinalOutputResolve: layer=%d name='%s' fullscreen=false "
                 "camera-override='%.*s' output='%s' material='%s' blend=%d private=true "
                 "publish-composite=%s publish-private=%s layer-surface=true "
                 "identity-model=true",
                 m_layer_node != nullptr ? m_layer_node->ID() : -1,
                 m_layer_node != nullptr ? m_layer_node->Name().c_str() : "",
                 static_cast<int>(layer_surface_cam.size()),
                 layer_surface_cam.data(),
                 final_output_node.output.c_str(),
                 material.name.c_str(),
                 static_cast<int>(material.blenmode),
                 m_final_composite.publishes_visible_output ? "true" : "false",
                 m_final_composite.publishes_private_output ? "true" : "false");
        return;
    }

    material.blenmode = BlendMode::Normal;
    final_output_node.sceneNode->SetCamera(effect_cam.data());
    final_output_node.sceneNode->CopyTrans(default_node);
    mesh.ChangeMeshDataFrom(default_mesh);
    LOG_INFO("SceneEffectFinalOutputResolve: layer=%d name='%s' fullscreen=false "
             "camera='%.*s' output='%s' material='%s' blend=%d private=true "
             "publish-composite=%s publish-private=%s",
             m_layer_node != nullptr ? m_layer_node->ID() : -1,
             m_layer_node != nullptr ? m_layer_node->Name().c_str() : "",
             static_cast<int>(effect_cam.size()),
             effect_cam.data(),
             final_output_node.output.c_str(),
             material.name.c_str(),
             static_cast<int>(material.blenmode),
             m_final_composite.publishes_visible_output ? "true" : "false",
             m_final_composite.publishes_private_output ? "true" : "false");
}

void SceneImageEffectLayer::ResolveEffect(const SceneMesh& default_mesh,
                                          std::string_view effect_cam,
                                          std::string_view layer_surface_cam,
                                          std::string_view final_output,
                                          bool keep_final_output_private,
                                          const Eigen::Affine3f* resolved_world_affine,
                                          FinalOutputCapability output_capability) {
    std::string_view ppong_a = m_pingpong_a, ppong_b = m_pingpong_b;
    auto             default_node = SceneNode();

    m_resolved_output_node = nullptr;
    m_resolved_output_follows_world = true;
    m_resolved_output_mesh_follows_final_mesh = true;
    m_final_composite.ResetForResolve();

    auto* fallback_last_output =
        ResolveEffectPingPongChain(default_mesh, default_node, effect_cam, ppong_a, ppong_b);
    if (EffectCount() > 0) {
        // Official dest-draw leftover then last-pass 0x1401ebf60 on one +0x158
        // object. ResolveFinalComposite / ResolveVisibleFinalOutput copy
        // FinalNode dest + FinalMesh onto the last pass — that is the
        // WorldNode/FinalNode split, not 0x1401ebf60.
        return;
    }
    SyncResolvedNodeForRoute(resolved_world_affine);
    const auto final_decision = ResolveFinalOutputDecision(fallback_last_output,
                                                           layer_surface_cam,
                                                           keep_final_output_private,
                                                           output_capability);

    std::string_view final_composite_source = ppong_a;
    if (final_decision.private_final_uses_layer_surface &&
        m_puppet_surface_projection.has_value() &&
        !m_puppet_surface_projection->target_name.empty()) {
        final_composite_source = m_puppet_surface_projection->target_name;
    }

    const auto diagnostic_roles = ResolveEffectOutputDiagnosticRoles(
        output_capability,
        final_decision.keep_authored_final_private,
        fallback_last_output != nullptr);
    const std::string_view authored_output_target =
        fallback_last_output != nullptr ? std::string_view(fallback_last_output->output)
                                        : std::string_view {};
    LOG_INFO("SceneEffectOutputContract: layer=%d name='%s' declared-capability=%.*s "
             "resolved-capability=%.*s authored-writer-role='%.*s' "
             "publication-writer-role='%.*s' effect-camera='%.*s' "
             "layer-surface-camera='%.*s' authored-output-target='%.*s' "
             "publication-source='%.*s' final-output-target='%.*s' "
             "private-parallax-owner='%.*s' publication-parallax-owner='%.*s' "
             "parallax-application-count=%u keep-authored-final-private=%s "
             "copybackground=%s",
             m_layer_node != nullptr ? m_layer_node->ID() : -1,
             m_layer_node != nullptr ? m_layer_node->Name().c_str() : "",
             static_cast<int>(FinalOutputCapabilityName(m_final_output_capability).size()),
             FinalOutputCapabilityName(m_final_output_capability).data(),
             static_cast<int>(FinalOutputCapabilityName(output_capability).size()),
             FinalOutputCapabilityName(output_capability).data(),
             static_cast<int>(diagnostic_roles.authored_writer.size()),
             diagnostic_roles.authored_writer.data(),
             static_cast<int>(diagnostic_roles.publication_writer.size()),
             diagnostic_roles.publication_writer.data(),
             static_cast<int>(effect_cam.size()),
             effect_cam.data(),
             static_cast<int>(layer_surface_cam.size()),
             layer_surface_cam.data(),
             static_cast<int>(authored_output_target.size()),
             authored_output_target.data(),
             static_cast<int>(final_composite_source.size()),
             final_composite_source.data(),
             static_cast<int>(final_output.size()),
             final_output.data(),
             static_cast<int>(diagnostic_roles.private_parallax_owner.size()),
             diagnostic_roles.private_parallax_owner.data(),
             static_cast<int>(diagnostic_roles.publication_parallax_owner.size()),
             diagnostic_roles.publication_parallax_owner.data(),
             diagnostic_roles.parallax_application_count,
             final_decision.keep_authored_final_private ? "true" : "false",
             m_copy_background ? "true" : "false");

    ResolveFinalCompositeNode(default_mesh,
                              default_node,
                              effect_cam,
                              final_output,
                              final_composite_source,
                              resolved_world_affine);

    if (fallback_last_output == nullptr) return;
    if (!final_decision.keep_authored_final_private) {
        ResolveVisibleFinalOutput(*fallback_last_output,
                                  default_mesh,
                                  default_node,
                                  effect_cam,
                                  final_output,
                                  resolved_world_affine);
        return;
    }

    ResolvePrivateFinalOutput(*fallback_last_output,
                              default_mesh,
                              default_node,
                              effect_cam,
                              layer_surface_cam,
                              final_decision.private_final_uses_layer_surface,
                              resolved_world_affine);
}
