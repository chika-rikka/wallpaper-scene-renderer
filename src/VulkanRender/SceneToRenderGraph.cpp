#include "SceneToRenderGraph.hpp"

#include "Scene/Scene.h"
#include "Scene/SceneMesh.h"
#include "RenderGraph/RenderGraph.hpp"
#include "SpecTexs.hpp"
#include "Core/MapSet.hpp"
#include "Utils/Logging.h"
#include "WPImageAlignment.hpp"

#include "VulkanRender/AllPasses.hpp"

#include <algorithm>
#include <array>
#include <functional>
#include <limits>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace wallpaper;
namespace wallpaper::rg
{

void doCopy(RenderGraphBuilder& builder, vulkan::CopyPass::Desc& desc, TexNode* in, TexNode* out) {
    builder.read(in);
    builder.write(out);

    desc.src = in->key();
    desc.dst = out->key();
}
void addCopyPass(RenderGraph& rgraph, TexNode* in, TexNode* out,
                 std::function<bool()> should_execute = {}) {
    rgraph.addPass<vulkan::CopyPass>(
        "copy",
        PassNode::Type::Copy,
        [in, out, should_execute = std::move(should_execute)](
            RenderGraphBuilder& builder, vulkan::CopyPass::Desc& desc) {
            doCopy(builder, desc, in, out);
            desc.should_execute = should_execute;
        });
}

void addCopyPass(RenderGraph& rgraph, const TexNode::Desc& in, const TexNode::Desc& out,
                 std::function<bool()> should_execute = {},
                 DestDrawPhase dest_draw_phase = DestDrawPhase::None, i32 layer_id = 0) {
    rgraph.addPass<vulkan::CopyPass>(
        "copy",
        PassNode::Type::Copy,
        [in, out, should_execute = std::move(should_execute), dest_draw_phase, layer_id](
            RenderGraphBuilder& builder, vulkan::CopyPass::Desc& desc) {
            auto* in_node  = builder.createTexNode(in);
            auto* out_node = builder.createTexNode(out, true);
            doCopy(builder, desc, in_node, out_node);
            desc.should_execute = should_execute;
            desc.dest_draw_phase = dest_draw_phase;
            desc.layer_id = layer_id;
        });
}

TexNode* addCopyPass(RenderGraph& rgraph, TexNode* in, TexNode::Desc* out_desc = nullptr,
                     std::function<bool()> should_execute = {}) {
    TexNode* copy { nullptr };
    rgraph.addPass<vulkan::CopyPass>(
        "copy",
        PassNode::Type::Copy,
        [&copy, in, out_desc, should_execute = std::move(should_execute)](
            RenderGraphBuilder& builder, vulkan::CopyPass::Desc& pdesc) {
            auto desc = out_desc == nullptr ? in->genDesc() : *out_desc;
            if (out_desc == nullptr) {
                desc.key += "_" + std::to_string(in->version()) + "_copy";
                desc.name += "_" + std::to_string(in->version()) + "_copy";
            }
            copy = builder.createTexNode(desc, true);
            doCopy(builder, pdesc, in, copy);
            pdesc.should_execute = should_execute;
        });
    return copy;
}

TexNode* addDefaultComposeSnapshot(RenderGraph& rgraph, TexNode* in) {
    // `_rt_default` is both the compose write target and the FullFrameBuffer sampler. A unique
    // `_rt_default_<version>_copy` per self-write keeps a screen-sized image alive for every
    // overlapping letter. Reuse one screen-sized partner for that snapshot instead.
    //
    // The transfer itself stays. Shader color-blend and refraction must sample the current
    // compose, including earlier letters, while Preserve/LOAD keeps uncovered pixels. Writing
    // an unseeded partner and publishing a mesh from it drops or smears those pixels.
    TexNode::Desc snapshot = in->genDesc();
    snapshot.key           = std::string(SpecTex_DefaultPingPong);
    snapshot.name          = snapshot.key;
    return addCopyPass(rgraph, in, &snapshot);
}

void addClearPass(RenderGraph& rgraph, const TexNode::Desc& target,
                  std::array<float, 4> color = { 0.0f, 0.0f, 0.0f, 0.0f },
                  std::function<bool()> should_execute = {}) {
    rgraph.addPass<vulkan::ClearPass>(
        "clear",
        PassNode::Type::Clear,
        [target, color, should_execute = std::move(should_execute)](
            RenderGraphBuilder& builder, vulkan::ClearPass::Desc& desc) {
            auto* target_node = builder.createTexNode(target, true);
            builder.write(target_node);
            desc.target         = target_node->key();
            desc.clear_value    = VkClearValue { .color = { color[0], color[1], color[2], color[3] } };
            desc.should_execute = should_execute;
        });
}

static bool IsRuntimeRenderTarget(const Scene* scene, const std::string& path) {
    // Authored effect FBO names are not guaranteed to carry Wallpaper Engine's `_rt_` prefix after
    // the parser uniquifies them with the effect-layer address. The render-target table is the
    // authoritative runtime contract, so graph construction must consult it before classifying a
    // texture edge as an imported asset.
    return IsSpecTex(path) || (scene != nullptr && scene->renderTargets.count(path) != 0);
}

static TexNode::Desc createTexDesc(std::string path, const Scene* scene = nullptr) {
    return TexNode::Desc { .name = path,
                           .key  = path,
                           .type = IsRuntimeRenderTarget(scene, path) ? TexNode::TexType::Temp
                                                                      : TexNode::TexType::Imported };
}

void addShadowAtlasPass(RenderGraph& rgraph, Scene* scene) {
    rgraph.addPass<vulkan::ShadowAtlasPass>(
        "shadow_atlas",
        PassNode::Type::CustomShader,
        [scene](RenderGraphBuilder& builder, vulkan::ShadowAtlasPass::Desc& desc) {
            auto* dst =
                builder.createTexNode(createTexDesc(std::string(SpecTex_ShadowAtlas), scene), true);
            builder.write(dst);
            desc.target = dst->key();
            desc.scene  = scene;
        });
}

void addVolumetricsSingleFillPass(RenderGraph& rgraph, const Scene* scene) {
    // Scene-depth blit into `_rt_volumetricsSingle`.
    // Read `_rt_default` so this runs after model chunks have written shared scene depth.
    rgraph.addPass<vulkan::VolumetricsSingleFillPass>(
        "volumetrics_single_fill",
        PassNode::Type::CustomShader,
        [scene](RenderGraphBuilder& builder, vulkan::VolumetricsSingleFillPass::Desc& desc) {
            auto* src = builder.createTexNode(createTexDesc(std::string(SpecTex_Default), scene));
            auto* dst =
                builder.createTexNode(createTexDesc(std::string(SpecTex_VolumetricsSingle), scene),
                                      true);
            builder.read(src);
            builder.write(dst);
            desc.dst          = dst->key();
            desc.scene_output = src->key();
        });
}
} // namespace wallpaper::rg

static void CheckAndSetSprite(Scene& scene, vulkan::ShaderDrawRequest& desc,
                              std::span<const std::string> texs) {
    for (usize i = 0; i < texs.size(); i++) {
        auto& tex = texs[i];
        if (! tex.empty() && ! IsSpecTex(tex) && scene.textures.count(tex) != 0) {
            const auto& stex = scene.textures.at(tex);
            if (stex.isSprite) {
                desc.sprites_map[i] = stex.spriteAnim;
            }
        }
    }
}

static bool ShouldExecuteHiddenDependency(Scene& scene, SceneNode* node, std::string_view output) {
    const auto owner_it = scene.nodeOwners.find(node);
    if (owner_it == scene.nodeOwners.end()) return false;
    if (scene.offscreenDependencyLayerIds.count(owner_it->second) == 0) return false;

    // Hidden dependency layers are allowed to keep rendering only into private offscreen targets that
    // another effect samples. They must never use that exemption for `_rt_default`, because that
    // would make an invisible helper layer composite directly onto the wallpaper and create the large
    // tinted rectangles seen in xray-style scenes.
    return output != SpecTex_Default;
}

struct DelayLinkInfo {
    using BindTexture = void (*)(rg::Pass&, u32, std::string_view);

    rg::NodeID id;
    rg::NodeID link_id;
    i32        tex_index;
    BindTexture bind_texture { nullptr };
};

template <typename PassT>
void BindShaderDrawTexture(rg::Pass& pass, u32 index, std::string_view texture_key) {
    static_cast<PassT&>(pass).setDescTex(index, texture_key);
}

struct ExtraInfo {
    Map<size_t, rg::TexNode*>  id_link_map {};
    std::vector<DelayLinkInfo> link_info {};
    rg::RenderGraph*           rgraph { nullptr };
    Scene*                     scene { nullptr };
    std::unordered_map<int32_t, size_t> layer_order_index {};
    // Model depth is shared per output target. Tracking the first model pass here lets the graph
    // clear depth once for each target, then load it for later chunks without touching 2D passes.
    std::unordered_set<std::string> model_depth_outputs_seen {};
    bool                       use_mipmap_framebuffer { false };
    bool                       include_hidden_for_pipeline_warmup { false };
    std::vector<SceneRenderGraphPassRecord>* inventory { nullptr };
};

static void RecordGraphPass(ExtraInfo& extra, SceneNode* node, i32 imgId,
                            std::string_view output, std::string_view camera,
                            DestDrawPhase dest_draw_phase) {
    if (extra.inventory == nullptr || node == nullptr) return;
    extra.inventory->push_back(SceneRenderGraphPassRecord {
        .layer_id         = imgId,
        .node_name        = node->Name(),
        .output           = std::string(output),
        .camera           = std::string(camera),
        .dest_draw_phase  = dest_draw_phase,
    });
}

static bool IsOffscreenDependencyLayer(const ExtraInfo& extra, i32 imgId) {
    return extra.scene != nullptr && imgId != 0 &&
        extra.scene->offscreenDependencyLayerIds.count(imgId) != 0;
}

static bool ShouldPublishLayerLinkOutput(const ExtraInfo& extra, i32 imgId,
                                         std::string_view output) {
    if (!IsOffscreenDependencyLayer(extra, imgId)) return true;
    if (output != SpecTex_Default) return true;

    // `_rt_imageLayerComposite_<id>` is a source-texture contract, not a screen-composite contract.
    // Hidden dependency layers may still contain historical final passes that target `_rt_default`,
    // but those passes are deliberately blocked from executing while the layer is invisible. Letting
    // such a skipped screen pass replace the layer's link source makes the consumer sample the
    // wallpaper/default target instead of the dependency's raw or effect-resolved offscreen image.
    return false;
}

static bool ShouldKeepEffectFinalOutputPrivate(const ExtraInfo& extra, SceneNode* node, i32 imgId,
                                               std::string_view inherited_output) {
    if (!IsOffscreenDependencyLayer(extra, imgId)) return false;

    const bool visible_default_route =
        node != nullptr && node->Visible() && inherited_output == SpecTex_Default;
    // A layer can be both an offscreen dependency source and a normal visible wallpaper layer. The
    // dependency route needs the final authored effect to stay in ping-pong space so
    // `_rt_imageLayerComposite_<id>` samples a resolved private texture. The visible default route is
    // the opposite contract: the authored final effect is the screen writer, while the synthetic
    // fallback is gated off whenever that final effect is visible. Keeping this visible route private
    // leaves `_rt_default` untouched and produces the gray frame reported by Rika/943626357.
    return !visible_default_route;
}

struct OrderedRenderGraphChild {
    SceneNode* node { nullptr };
    bool       proxy { false };
    size_t     sequence { 0 };
};

struct NodePassOptions {
    AlphaWritePolicy alpha_write_policy { AlphaWritePolicy::Preserve };
    bool        clear_before_draw { false };
    std::string camera_override;
    bool        use_active_camera_for_parallax { false };
    bool        premultiplied_source_blend { false };
    bool        use_active_camera_for_uniforms { false };
    bool        use_identity_model { false };
    DestDrawPhase dest_draw_phase { DestDrawPhase::None };
    // LASTPASS_DEST_STACK: leftover dest-ortho / last-pass fit-ortho are
    // not the private WorldNode camera and not I-slot "effect".
    bool        omit_layer_camera { false };
};

static bool IsDestDrawObject(const SceneObject* object) {
    // Official dest-draw walker is ImageLayer / text +0x158 (LEFTOVER_VS_DESTDRAW).
    return object != nullptr &&
           (object->kind() == SceneObjectKind::Image || object->kind() == SceneObjectKind::Text);
}

static bool DestDrawPublishesDefault(const SceneObject* object) {
    return object != nullptr && object->DestDrawPublishesDefault();
}

static void ApplyDestDrawLeftover(Scene& scene, SceneNode* node, i32 imgId,
                                  NodePassOptions& options) {
    auto* object = scene.FindSceneObject(imgId);
    if (!IsDestDrawObject(object)) return;
    options.dest_draw_phase = DestDrawPhase::Leftover;
    options.use_identity_model = true;
    options.camera_override.clear();
    options.omit_layer_camera = true;
    options.use_active_camera_for_uniforms = false;
    options.use_active_camera_for_parallax = false;
    if (object->effect_count() <= 0) {
        // IMAGE_VT_F0 leftover +0x320==0 Draw [+0x490]. Puppet leftover-card
        // writes puppet+0x18 verts (PUPPET_490). No-puppet +0x490 is the
        // MESH_FACTORY card already bound on the node: it carries the
        // authored UV contract (sprite frames stay frame-local [0,1],
        // nopadding samples the full texture). The +0x2e8 last-pass card
        // uses leftover UV (content/physical) and must not replace it.
        // Live +0x110 id 0xd uploads +0x8f0 (LastPassDrawMvp).
        const SceneMesh* leftover_f0 = object->image_490_mesh();
        if (leftover_f0 != nullptr && node != nullptr && node->Mesh() != nullptr) {
            node->Mesh()->ChangeMeshDataFrom(*leftover_f0);
            node->Mesh()->SetDirty();
        }
        return;
    }
    // IMAGE_VT_E8 dest=I 0x1401e9702. Mesh +0x2d8 IMAGE_2D8_NOFULLFB.
    // LASTPASS_DEST_STACK: leftover private WorldNode camera is not dest-ortho.
    object->SizeDestDrawNamedRts();
    if (object->leftover_mesh() != nullptr && node != nullptr && node->Mesh() != nullptr) {
        node->Mesh()->ChangeMeshDataFrom(*object->leftover_mesh());
        // ChangeMeshDataFrom shares CPU payload only (SceneImageEffectLayer).
        // Official leftover Draw is +0x2d8 (IMAGE_2D8_NOFULLFB 0x140208067).
        node->Mesh()->SetDirty();
    }
}

struct TraversalRoute {
    bool                           routed_node { false };
    std::optional<Eigen::Matrix4d> model;
    bool                           compose_source { false };
    std::string                    compose_source_camera;
    AlphaWritePolicy               compose_source_alpha_write_policy {
        AlphaWritePolicy::Preserve
    };
    bool                           premultiplied_source_blend { false };
};

static bool HasRenderableMeshMaterial(SceneNode* node) {
    return node != nullptr && node->Mesh() != nullptr && node->Mesh()->Material() != nullptr;
}

static int32_t NodeLayerId(const Scene& scene, SceneNode* node) {
    if (node == nullptr) return 0;
    if (auto owner_it = scene.nodeOwners.find(node); owner_it != scene.nodeOwners.end()) {
        return owner_it->second;
    }
    return node->ID();
}

static bool ShouldEmitLayerNodeForResidency(Scene& scene, SceneNode* node, const ExtraInfo& extra) {
    if (node == nullptr || node == scene.sceneGraph.get()) return true;
    if (extra.include_hidden_for_pipeline_warmup) return true;

    const int32_t layer_id = NodeLayerId(scene, node);
    if (layer_id == 0) return true;
    if (scene.IsLayerVisible(layer_id)) return true;

    // Dependency-source layers are the one hidden case that must stay resident in the graph: other
    // visible effects can sample their private offscreen outputs. Ordinary hidden layers are
    // pruned so their passes, framebuffers, descriptors, imported textures, and video decoders can
    // be released until the layer becomes visible again.
    return scene.offscreenDependencyLayerIds.count(layer_id) != 0;
}

static size_t NodeLayerOrderIndex(SceneNode* node, const ExtraInfo& extra) {
    if (extra.scene == nullptr || node == nullptr) return std::numeric_limits<size_t>::max();
    const int32_t layer_id = NodeLayerId(*extra.scene, node);
    if (auto it = extra.layer_order_index.find(layer_id); it != extra.layer_order_index.end()) {
        return it->second;
    }
    return std::numeric_limits<size_t>::max();
}

static bool IsEffectLocalProxyDependency(SceneNode* node, const ExtraInfo& extra) {
    if (extra.scene == nullptr || node == nullptr) return false;
    const int32_t layer_id = NodeLayerId(*extra.scene, node);
    return layer_id != 0 && extra.scene->offscreenDependencyLayerIds.count(layer_id) != 0;
}

static bool EffectSourceUsesProxyChildren(SceneImageEffectLayer* imgeff) {
    if (imgeff == nullptr) return false;
    const auto policy = imgeff->SourceContributionPolicy();
    return policy == SceneImageEffectLayer::SourcePolicy::OwnerNodeAndProxyChildren ||
        policy == SceneImageEffectLayer::SourcePolicy::ProxyChildrenOnly;
}

static bool EffectSourceUsesOwnerNode(SceneImageEffectLayer* imgeff) {
    if (imgeff == nullptr) return true;
    const auto policy = imgeff->SourceContributionPolicy();
    return policy == SceneImageEffectLayer::SourcePolicy::OwnerNode ||
        policy == SceneImageEffectLayer::SourcePolicy::OwnerNodeAndProxyChildren;
}

static bool HasRenderOrderProxyChildren(SceneNode* node, const ExtraInfo& extra) {
    if (node == nullptr || extra.scene == nullptr) return false;
    const auto proxy_it = extra.scene->renderOrderProxyChildren.find(node);
    return proxy_it != extra.scene->renderOrderProxyChildren.end() && !proxy_it->second.empty();
}

static bool ShouldSeedEmptyProxyComposeFromFramebuffer(SceneNode* node,
                                                       SceneImageEffectLayer* imgeff,
                                                       i32 imgId,
                                                       const ExtraInfo& extra,
                                                       std::string_view inherited_output,
                                                       bool compose_source_route) {
    if (node == nullptr || imgeff == nullptr) return false;
    if (compose_source_route || inherited_output != SpecTex_Default) return false;
    if (!node->Visible()) return false;
    if (IsOffscreenDependencyLayer(extra, imgId)) return false;
    if (imgeff->SourceContributionPolicy() !=
        SceneImageEffectLayer::SourcePolicy::ProxyChildrenOnly) {
        return false;
    }
    return !HasRenderOrderProxyChildren(node, extra);
}

enum class OwnerNodeSourceFallbackReason
{
    None,
    EmptyProxyDependencySource,
};

static OwnerNodeSourceFallbackReason ResolveOwnerNodeSourceFallbackReason(
    SceneNode* node, SceneImageEffectLayer* imgeff, i32 imgId, const ExtraInfo& extra) {
    if (node == nullptr || imgeff == nullptr) return OwnerNodeSourceFallbackReason::None;
    if (imgeff->SourceContributionPolicy() !=
        SceneImageEffectLayer::SourcePolicy::ProxyChildrenOnly) {
        return OwnerNodeSourceFallbackReason::None;
    }
    if (HasRenderOrderProxyChildren(node, extra)) return OwnerNodeSourceFallbackReason::None;
    if (IsOffscreenDependencyLayer(extra, imgId)) {
        return OwnerNodeSourceFallbackReason::EmptyProxyDependencySource;
    }
    return OwnerNodeSourceFallbackReason::None;
}

static std::string_view OwnerNodeSourceFallbackReasonName(
    OwnerNodeSourceFallbackReason reason) {
    switch (reason) {
    case OwnerNodeSourceFallbackReason::None: return "none";
    case OwnerNodeSourceFallbackReason::EmptyProxyDependencySource:
        return "empty-proxy-dependency-source";
    }
    return "unknown";
}

static bool NodeMaterialSamplesFramebuffer(SceneNode* node) {
    if (node == nullptr || node->Mesh() == nullptr || node->Mesh()->Material() == nullptr) {
        return false;
    }

    const auto& textures = node->Mesh()->Material()->textures;
    return std::find(textures.begin(), textures.end(), std::string(SpecTex_Default)) !=
        textures.end();
}

static bool NodeUsesPerspectiveCamera(const Scene& scene, SceneNode* node) {
    // Uniform updates walk ancestors for the effective camera. Keep the same walk here so a
    // child that inherits a perspective camera is not flattened back to the active ortho view.
    for (auto* current = node; current != nullptr; current = current->Parent()) {
        if (current->Camera().empty()) continue;
        const auto camera_it = scene.cameras.find(current->Camera());
        if (camera_it != scene.cameras.end() && camera_it->second != nullptr &&
            camera_it->second->IsPerspective()) {
            return true;
        }
    }
    return false;
}

struct EffectSourceRoutingDecision {
    OwnerNodeSourceFallbackReason owner_node_fallback_reason {
        OwnerNodeSourceFallbackReason::None
    };
    bool        owner_node_source_fallback { false };
    bool        owner_node_source_fallback_samples_framebuffer { false };
    bool        owner_node_samples_framebuffer { false };
    bool        owner_node_uses_perspective_camera { false };
    bool        owner_node_contributes_to_effect_source { false };
    bool        proxy_children_contribute_to_effect_source { false };
    std::string active_compose_source_camera;
    bool        use_compose_camera_override { false };
    bool        seed_empty_proxy_compose_from_framebuffer { false };
};

static EffectSourceRoutingDecision ResolveEffectSourceRouting(SceneNode* node,
                                                              SceneImageEffectLayer* imgeff,
                                                              i32 imgId,
                                                              const ExtraInfo& extra,
                                                              std::string_view inherited_output,
                                                              const TraversalRoute& route) {
    EffectSourceRoutingDecision decision;

    // Compose source routing is the policy boundary between traversal and pass emission. Keeping the
    // derived booleans together makes each Wallpaper Engine source contract explicit: owner-card
    // contribution, proxy-child contribution, empty dependency fallback, and framebuffer seeding are
    // separate decisions even though the render graph eventually emits them in one traversal pass.
    decision.owner_node_fallback_reason =
        ResolveOwnerNodeSourceFallbackReason(node, imgeff, imgId, extra);
    decision.owner_node_source_fallback =
        decision.owner_node_fallback_reason != OwnerNodeSourceFallbackReason::None;
    decision.owner_node_source_fallback_samples_framebuffer =
        decision.owner_node_source_fallback && NodeMaterialSamplesFramebuffer(node);
    decision.owner_node_samples_framebuffer = NodeMaterialSamplesFramebuffer(node);
    decision.owner_node_uses_perspective_camera =
        extra.scene != nullptr && NodeUsesPerspectiveCamera(*extra.scene, node);
    decision.owner_node_contributes_to_effect_source =
        EffectSourceUsesOwnerNode(imgeff) || decision.owner_node_source_fallback;
    decision.proxy_children_contribute_to_effect_source = EffectSourceUsesProxyChildren(imgeff);
    decision.active_compose_source_camera =
        route.compose_source
            ? route.compose_source_camera
            : (decision.proxy_children_contribute_to_effect_source && node != nullptr
                   ? std::string(node->Camera())
                   : std::string());
    decision.use_compose_camera_override =
        route.compose_source && imgeff == nullptr && !decision.active_compose_source_camera.empty();
    decision.seed_empty_proxy_compose_from_framebuffer =
        ShouldSeedEmptyProxyComposeFromFramebuffer(node,
                                                  imgeff,
                                                  imgId,
                                                  extra,
                                                  inherited_output,
                                                  route.compose_source);
    return decision;
}

static NodePassOptions BuildOwnerSourcePassOptions(
    SceneImageEffectLayer* imgeff,
    std::string_view output,
    std::string_view inherited_output,
    const TraversalRoute& route,
    const EffectSourceRoutingDecision& source_route) {
    const bool clear_private_effect_source =
        route.compose_source && imgeff != nullptr && output != inherited_output;
    const bool evaluate_framebuffer_source_with_active_camera =
        source_route.owner_node_samples_framebuffer &&
        !source_route.owner_node_uses_perspective_camera;

    // Owner-source emission is the one place where source routing affects actual pass state. Keep
    // these side effects grouped so future route types can extend the pass contract without adding
    // another cluster of loosely related booleans inside ToGraphPass().
    //
    // - Composition source routes write child layers into a parent-local source target. Their alpha
    //   policy comes from the parent composition layer's copybackground contract.
    // - An owner material that samples the live framebuffer normally evaluates world geometry
    //   against the active scene camera, including while writing a private source target. The
    //   composelayer vertex shader separately turns authored texture coordinates into the fullscreen
    //   output quad, so using the private effect camera for both roles creates the cursor-following
    //   rectangular region seen in incorrect implementations. An explicitly perspective node is
    //   the exception: its authored camera remains the draw camera even when its material samples
    //   the framebuffer. Forcing refractive perspective particles through the active orthographic
    //   camera removes all authored depth scaling and makes foreground rain appear uniformly small.
    // - Private effect source targets are cleared only at the seed step; clearing later authored
    //   effect passes would erase intermediate waterwaves/foliagesway/opacity results.
    return NodePassOptions {
        .alpha_write_policy = route.compose_source
            ? route.compose_source_alpha_write_policy
            : AlphaWritePolicy::Preserve,
        .clear_before_draw = clear_private_effect_source,
        .camera_override = source_route.use_compose_camera_override
            ? source_route.active_compose_source_camera
            : std::string(),
        .use_active_camera_for_parallax = source_route.use_compose_camera_override,
        .premultiplied_source_blend = route.premultiplied_source_blend,
        .use_active_camera_for_uniforms =
            evaluate_framebuffer_source_with_active_camera,
    };
}

static std::string_view EffectSourcePolicyName(SceneImageEffectLayer::SourcePolicy policy) {
    switch (policy) {
    case SceneImageEffectLayer::SourcePolicy::OwnerNode: return "owner-node";
    case SceneImageEffectLayer::SourcePolicy::OwnerNodeAndProxyChildren:
        return "owner-node-and-proxy-children";
    case SceneImageEffectLayer::SourcePolicy::ProxyChildrenOnly: return "proxy-children-only";
    }
    return "unknown";
}

static Eigen::Matrix4d ResolveRouteModel(SceneNode* node,
                                         const std::optional<Eigen::Matrix4d>& route_model) {
    if (route_model.has_value()) return *route_model;
    if (node == nullptr) return Eigen::Matrix4d::Identity();

    node->UpdateTrans();
    return node->ModelTrans();
}

static std::optional<Eigen::Matrix4d> BuildChildRouteModel(
    SceneNode* parent, SceneNode* child, bool routed_child,
    const std::optional<Eigen::Matrix4d>& parent_route_model) {
    if (parent == nullptr || child == nullptr) return std::nullopt;
    if (!routed_child && !parent_route_model.has_value()) return std::nullopt;

    // Proxy routing is order-only in the physical SceneNode tree, but render-time transforms still
    // need to follow Wallpaper Engine's authored parent. Propagating a resolved route matrix lets
    // effect final passes use the same virtual parent chain that shader uniforms use later.
    const auto parent_route =
        RemoveImageAlignmentOffsetFromModel(ResolveRouteModel(parent, parent_route_model),
                                            parent->AlignmentOffset());
    return parent_route * child->GetLocalTrans();
}

static std::vector<OrderedRenderGraphChild> OrderedRenderGraphChildren(SceneNode* node,
                                                                       ExtraInfo& extra) {
    std::vector<OrderedRenderGraphChild> children;
    if (node == nullptr || extra.scene == nullptr) return children;

    std::unordered_set<SceneNode*> seen;
    size_t sequence = 0;
    for (auto& child : node->GetChildren()) {
        if (!child || !seen.insert(child.get()).second) continue;
        children.push_back(OrderedRenderGraphChild {
            .node = child.get(),
            .proxy = false,
            .sequence = sequence++,
        });
    }

    if (auto proxy_it = extra.scene->renderOrderProxyChildren.find(node);
        proxy_it != extra.scene->renderOrderProxyChildren.end()) {
        for (auto* proxy_child : proxy_it->second) {
            if (proxy_child == nullptr || !seen.insert(proxy_child).second) continue;
            children.push_back(OrderedRenderGraphChild {
                .node = proxy_child,
                .proxy = true,
                .sequence = sequence++,
            });
        }
    }

    // The scene tree still owns lifetime and transforms, but the render graph needs Wallpaper
    // Engine's authored layer order. Sorting only by known layer order and then by insertion
    // sequence keeps helper nodes deterministic without forcing every runtime node to have an
    // authored layer id.
    std::stable_sort(children.begin(), children.end(), [&extra](const auto& lhs, const auto& rhs) {
        const auto lhs_index = NodeLayerOrderIndex(lhs.node, extra);
        const auto rhs_index = NodeLayerOrderIndex(rhs.node, extra);
        if (lhs_index != rhs_index) return lhs_index < rhs_index;
        return lhs.sequence < rhs.sequence;
    });
    return children;
}

template <typename PassT>
static void AddNodePassImpl(SceneNode* node, std::string_view output, i32 imgId, ExtraInfo& extra,
                            std::function<bool()> should_execute, NodePassOptions options) {
    auto& rgraph = *extra.rgraph;
    auto& scene  = *extra.scene;

    if (node->Mesh() == nullptr) {
        return;
    }
    auto* mesh = node->Mesh();
    if (mesh->Material() == nullptr) {
        return;
    }
    auto* material = mesh->Material();
    const std::string output_key =
        material->modelRenderState.has_value() && !material->modelRenderState->outputOverride.empty()
            ? material->modelRenderState->outputOverride
            : std::string(output);
    const bool is_model_pass = material->modelRenderState.has_value();
    const bool clear_model_depth = is_model_pass &&
        extra.model_depth_outputs_seen.insert(output_key).second;

    // Official quality `reflection=false` keeps `_rt_Reflection` allocated and sampled, but
    // skips the mirrored producer draw. Gate execution here so the checkbox is live without a
    // topology rebuild; a companion clear pass empties the RT while the switch is off.
    if (output_key == SpecTex_Reflection) {
        should_execute = [inner = std::move(should_execute), &scene]() {
            if (!scene.reflectionsEnabled) return false;
            return !inner || inner();
        };
    }

    std::string passName = material->name;
    const std::string_view pass_camera = options.omit_layer_camera
        ? std::string_view {}
        : (! options.camera_override.empty() ? std::string_view(options.camera_override)
                                             : std::string_view(node->Camera()));
    RecordGraphPass(extra, node, imgId, output_key, pass_camera, options.dest_draw_phase);
    rgraph.addPass<PassT>(
        passName,
        rg::PassNode::Type::CustomShader,
        [material, node, output_key, imgId, &rgraph, &scene, &extra,
         clear_model_depth, options = std::move(options),
         should_execute = std::move(should_execute)](
            rg::RenderGraphBuilder& builder, typename PassT::Desc& pdesc) {
            const auto& pass = builder.workPassNode();
            // Passing the live scene into the prepared pass lets resource refreshes resolve current
            // render-target dependencies directly, which is what keeps first-class text bridges and
            // ordinary effect passes on the same stable render-graph contract.
            pdesc.scene      = &scene;
            pdesc.node       = node;
            pdesc.layer_id   = imgId;
            pdesc.execute_when_hidden = ShouldExecuteHiddenDependency(scene, node, output_key);
            pdesc.should_execute      = should_execute;
            pdesc.output     = output_key;
            pdesc.alpha_write_policy = output_key != SpecTex_Default
                ? options.alpha_write_policy
                : AlphaWritePolicy::Preserve;
            pdesc.premultiplied_source_blend = options.premultiplied_source_blend;
            pdesc.clear_before_draw = output_key != SpecTex_Default && options.clear_before_draw;
            pdesc.camera_override = options.camera_override;
            pdesc.use_active_camera_for_uniforms = options.use_active_camera_for_uniforms;
            pdesc.use_active_camera_for_parallax =
                !pdesc.camera_override.empty() && options.use_active_camera_for_parallax;
            pdesc.use_identity_model = options.use_identity_model;
            pdesc.dest_draw_phase = options.dest_draw_phase;
            if (!pdesc.camera_override.empty()) {
                LOG_INFO("SceneRenderGraphComposeCameraOverride: layer=%d node='%s' "
                         "output='%s' camera='%s' active-parallax=%s",
                         imgId,
                         node != nullptr ? node->Name().c_str() : "",
                         output_key.c_str(),
                         pdesc.camera_override.c_str(),
                         pdesc.use_active_camera_for_parallax ? "true" : "false");
            }
            if (pdesc.use_active_camera_for_uniforms) {
                LOG_INFO("SceneRenderGraphActiveCameraUniformOverride: layer=%d node='%s' "
                         "output='%s'",
                         imgId,
                         node != nullptr ? node->Name().c_str() : "",
                         output_key.c_str());
            }
            if (const auto& model_state = material->modelRenderState; model_state.has_value()) {
                // Depth state is transported through the pass description instead of inferred from
                // camera names, so adding model rendering cannot alter ordinary 2D custom shaders.
                pdesc.model_pass = true;
                pdesc.depth_test = model_state->depthTest;
                pdesc.depth_write = model_state->depthWrite;
                pdesc.depth_greater = model_state->depthGreater;
                pdesc.depth_clear = model_state->depthClear;
                // The shared Back RT is rebound per light, so each hull must start
                // from a cleared depth instead of the first-writer-wins model-chunk rule.
                pdesc.clear_depth = clear_model_depth ||
                    output_key == SpecTex_VolumetricsBack;
            }
            CheckAndSetSprite(scene, pdesc, material->textures);
            for (usize i = 0; i < material->textures.size(); i++) {
                const auto&  url = material->textures[i];
                rg::TexNode* input { nullptr };
                if (url.empty()) {
                    pdesc.textures.emplace_back("");
                    continue;
                } else if (IsSpecLinkTex(url)) {
                    auto id = ParseLinkTex(url);
                    extra.link_info.push_back(DelayLinkInfo {
                        .id           = pass.ID(),
                        .link_id      = id,
                        .tex_index    = (i32)i,
                        .bind_texture = &BindShaderDrawTexture<PassT>,
                    });
                    pdesc.textures.emplace_back("");
                    continue;
                } else {
                    rg::TexNode::Desc desc;
                    desc.key  = url;
                    desc.name = url;
                    // Some effect-local FBOs use plain names such as `blur_start_2_<addr>`.
                    // Treat any key already registered in Scene::renderTargets as temporary graph
                    // storage so those edges order like internal render targets, not external
                    // material textures.
                    desc.type = ! rg::IsRuntimeRenderTarget(&scene, url)
                        ? rg::TexNode::TexType::Imported
                        : rg::TexNode::TexType::Temp;
                    input     = builder.createTexNode(desc);
                    if (rg::IsRuntimeRenderTarget(&scene, url)) builder.markVirtualWrite(input);
                    if (sstart_with(url, WE_MIP_MAPPED_FRAME_BUFFER))
                        extra.use_mipmap_framebuffer = true;
                }

                if (url == output_key) {
                    builder.markSelfWrite(input);
                    // Compose self-writes reuse one screen-sized snapshot. Other self-writes
                    // still get a versioned copy name because their destinations are not a
                    // shared compose buffer.
                    input = url == SpecTex_Default ? rg::addDefaultComposeSnapshot(rgraph, input)
                                                   : rg::addCopyPass(rgraph, input);
                }
                builder.read(input);
                pdesc.textures.emplace_back(input->key());
            }

            // Mask textures belong to the mesh draw plan rather than to the visible material. The
            // selected masked pass declares them as ordinary imported reads, while an empty plan
            // leaves the regular CustomShaderPass with no mask-related graph resources at all.
            for (const auto& group : node->Mesh()->MaskedDraw().groups) {
                rg::TexNode::Desc desc {
                    .name = group.maskTexture,
                    .key  = group.maskTexture,
                    .type = rg::TexNode::TexType::Imported,
                };
                auto* input = builder.createTexNode(desc);
                builder.read(input);
            }

            rg::TexNode* output_node { nullptr };
            output_node =
                builder.createTexNode(rg::TexNode::Desc { .name = output_key,
                                                          .key  = output_key,
                                                          .type = rg::TexNode::TexType::Temp },
                                      true);
            builder.write(output_node);
            if (ShouldPublishLayerLinkOutput(extra, imgId, output_key)) {
                extra.id_link_map[(usize)imgId] = output_node;
            }
        });
}

static void AddNodePass(SceneNode* node, std::string_view output, i32 imgId, ExtraInfo& extra,
                        std::function<bool()> should_execute = {},
                        NodePassOptions options = {}) {
    if (node != nullptr && node->Mesh() != nullptr && ! node->Mesh()->MaskedDraw().empty()) {
        AddNodePassImpl<vulkan::MaskedMeshPass>(node,
                                                output,
                                                imgId,
                                                extra,
                                                std::move(should_execute),
                                                std::move(options));
        return;
    }
    AddNodePassImpl<vulkan::CustomShaderPass>(node,
                                              output,
                                              imgId,
                                              extra,
                                              std::move(should_execute),
                                              std::move(options));
}

static void AddTextNodePass(SceneNode* node, std::string_view output, i32 imgId, ExtraInfo& extra,
                            AlphaWritePolicy alpha_write_policy,
                            DestDrawPhase dest_draw_phase = DestDrawPhase::None,
                            bool omit_layer_camera = false) {
    auto& rgraph = *extra.rgraph;
    auto& scene  = *extra.scene;

    if (node == nullptr || node->Text() == nullptr) {
        return;
    }

    const std::string output_key(output);
    std::string pass_name = node->Name().empty() ? std::string("text") : node->Name();
    const std::string_view pass_camera =
        omit_layer_camera ? std::string_view {} : std::string_view(node->Camera());
    RecordGraphPass(extra, node, imgId, output_key, pass_camera, dest_draw_phase);
    rgraph.addPass<vulkan::TextPass>(
        pass_name,
        rg::PassNode::Type::Text,
        [node, output_key, imgId, alpha_write_policy, dest_draw_phase, &scene, &extra](
            rg::RenderGraphBuilder& builder, vulkan::TextPass::Desc& pdesc) {
            const auto& pass = builder.workPassNode();
            // Text is now emitted as its own render-graph pass. It shares the same constrained
            // hidden-dependency rule as mesh passes: invisible helper layers may render private
            // offscreen sources, but they must not composite text directly into `_rt_default`.
            pdesc.scene = &scene;
            pdesc.node = node;
            // Keep the authored layer id on the prepared pass so runtime text rerasters can
            // refresh the exact Clock/TextPass resources without broadening the dirty target set.
            pdesc.layer_id = imgId;
            pdesc.execute_when_hidden = ShouldExecuteHiddenDependency(scene, node, output_key);
            pdesc.dest_draw_phase = dest_draw_phase;
            pdesc.output = output_key;
            pdesc.alpha_write_policy = output_key != SpecTex_Default
                ? alpha_write_policy
                : AlphaWritePolicy::Preserve;

            auto* output_node =
                builder.createTexNode(rg::TexNode::Desc { .name = output_key,
                                                          .key = output_key,
                                                          .type = rg::TexNode::TexType::Temp },
                                      true);
            builder.write(output_node);
            if (ShouldPublishLayerLinkOutput(extra, imgId, output_key)) {
                extra.id_link_map[(usize)imgId] = output_node;
            }
            (void)pass;
        });
}

static void ToGraphPass(SceneNode* node, std::string_view inherited_output, i32 imgId,
                        ExtraInfo& extra, std::function<bool()> node_execute_gate = {},
                        TraversalRoute route = {}) {
    auto& scene = *extra.scene;

    if (node != nullptr && !route.routed_node) {
        const bool proxy_node = scene.renderOrderProxyNodes.count(node) != 0;
        const bool detached_source_node = scene.detachedEffectSourceNodes.count(node) != 0;
        if (proxy_node || detached_source_node) {
            // Root-owned proxy/source nodes are emitted through explicit authored-order routes.
            // Reaching them through the physical tree means the root traversal is at the wrong
            // sibling position, so skip this visit to avoid late duplicate composites.
            LOG_INFO("SceneRenderGraphNodeRouteSkip: layer=%d name='%s' reason='%s'",
                     NodeLayerId(scene, node),
                     node->Name().c_str(),
                     detached_source_node ? "detached-source" : "proxy");
            return;
        }
    }

    if (node != nullptr && !ShouldEmitLayerNodeForResidency(scene, node, extra)) {
        LOG_INFO("SceneRenderGraphResidencySkip: layer=%d name='%s' local-visible=%s "
                 "layer-visible=%s",
                 NodeLayerId(scene, node),
                 node->Name().c_str(),
                 scene.GetLayerLocalVisibility(NodeLayerId(scene, node)) ? "true" : "false",
                 node->LayerVisible() ? "true" : "false");
        return;
    }

    std::string_view         output = inherited_output;
    SceneImageEffectLayer*   imgeff { nullptr };
    const auto resolved_route_model = ResolveRouteModel(node, route.model);
    if (node != nullptr && !node->Camera().empty()) {
        auto camera_it = scene.cameras.find(node->Camera());
        if (camera_it != scene.cameras.end() && camera_it->second->HasImgEffect()) {
            imgeff = camera_it->second->GetImgEffect().get();
            output = imgeff->FirstTarget();
        }
    }

    const auto source_route =
        ResolveEffectSourceRouting(node, imgeff, imgId, extra, inherited_output, route);
    if (imgeff != nullptr &&
        imgeff->SourceContributionPolicy() == SceneImageEffectLayer::SourcePolicy::ProxyChildrenOnly) {
        LOG_INFO("SceneRenderGraphComposeSourceClear: layer=%d name='%s' output='%.*s' "
                 "camera='%s'",
                 NodeLayerId(scene, node),
                 node != nullptr ? node->Name().c_str() : "",
                 static_cast<int>(output.size()),
                 output.data(),
                 source_route.active_compose_source_camera.c_str());
        rg::addClearPass(*extra.rgraph, rg::createTexDesc(std::string(output), extra.scene));
    }

    auto* dest_object = scene.FindSceneObject(imgId);
    const bool dest_leftover_required =
        IsDestDrawObject(dest_object) && dest_object->leftover_mesh() != nullptr &&
        (node == nullptr || node->Text() == nullptr) &&
        (dest_object->effect_count() > 0 ||
         (dest_object->kind() == SceneObjectKind::Image && !route.compose_source));

    if (source_route.seed_empty_proxy_compose_from_framebuffer &&
        HasRenderableMeshMaterial(node) && !dest_leftover_required) {
        LOG_INFO("SceneRenderGraphComposeFramebufferSeed: layer=%d name='%s' output='%.*s' "
                 "policy=%.*s reason='empty-proxy-visible-framebuffer-source'",
                 NodeLayerId(scene, node),
                 node->Name().c_str(),
                 static_cast<int>(output.size()),
                 output.data(),
                 static_cast<int>(EffectSourcePolicyName(imgeff->SourceContributionPolicy()).size()),
                 EffectSourcePolicyName(imgeff->SourceContributionPolicy()).data());
        // Empty visible compose layers are framebuffer filters: their first effect texture is the
        // current screen image, not the composelayer card itself. The composelayer shader writes a
        // full offscreen source target while sampling `_rt_default` through the active camera, so
        // the later world-space effect writer replaces exactly the same screen region without
        // exposing a rectangular owner mesh.
        AddNodePass(node,
                    output,
                    imgId,
                    extra,
                    node_execute_gate,
                    NodePassOptions { .use_active_camera_for_uniforms = true });
    }

    if (HasRenderableMeshMaterial(node) &&
        (source_route.owner_node_contributes_to_effect_source || dest_leftover_required)) {
        if (source_route.owner_node_source_fallback) {
            LOG_INFO("SceneRenderGraphComposeOwnerSourceFallback: layer=%d name='%s' output='%.*s' "
                     "policy=%.*s reason='%.*s' framebuffer-uniforms=%s",
                     NodeLayerId(scene, node),
                     node->Name().c_str(),
                     static_cast<int>(output.size()),
                     output.data(),
                     static_cast<int>(EffectSourcePolicyName(imgeff->SourceContributionPolicy()).size()),
                     EffectSourcePolicyName(imgeff->SourceContributionPolicy()).data(),
                     static_cast<int>(
                         OwnerNodeSourceFallbackReasonName(
                             source_route.owner_node_fallback_reason)
                             .size()),
                     OwnerNodeSourceFallbackReasonName(source_route.owner_node_fallback_reason)
                         .data(),
                     source_route.owner_node_source_fallback_samples_framebuffer ? "true"
                                                                                 : "false");
        }
        auto leftover_options = BuildOwnerSourcePassOptions(
            imgeff, output, inherited_output, route, source_route);
        ApplyDestDrawLeftover(scene, node, imgId, leftover_options);
        // IMAGE_VT_F0 leftover +0x320==0 OMSet is FullFB, not +0x2c8
        // named-RT (IMAGE_DRAW_PASS bit2 clear).
        const std::string_view leftover_output =
            dest_object != nullptr && dest_object->effect_count() <= 0 &&
                    dest_object->kind() == SceneObjectKind::Image && !route.compose_source
                ? std::string_view(SpecTex_Default)
                : output;
        AddNodePass(node,
                    leftover_output,
                    imgId,
                    extra,
                    node_execute_gate,
                    leftover_options);
    } else if (HasRenderableMeshMaterial(node) && imgeff != nullptr &&
               !source_route.seed_empty_proxy_compose_from_framebuffer) {
        LOG_INFO("SceneRenderGraphComposeOwnerSourceSkip: layer=%d name='%s' output='%.*s' "
                 "policy=%.*s",
                 NodeLayerId(scene, node),
                 node->Name().c_str(),
                 static_cast<int>(output.size()),
                 output.data(),
                 static_cast<int>(EffectSourcePolicyName(imgeff->SourceContributionPolicy()).size()),
                 EffectSourcePolicyName(imgeff->SourceContributionPolicy()).data());
    }
    // Text is now a first-class scene primitive. Whenever a node owns text we emit the dedicated
    // text pass directly from that primitive, keeping the render graph aligned with the same
    // authoritative text object that parser and runtime updates mutate.
    if (node != nullptr && node->Text() != nullptr) {
        const auto text_alpha_write_policy =
            imgeff == nullptr && route.compose_source
                ? route.compose_source_alpha_write_policy
                : AlphaWritePolicy::Preserve;
        DestDrawPhase text_phase = DestDrawPhase::None;
        bool text_omit_camera = false;
        if (auto* object = scene.FindSceneObject(imgId); IsDestDrawObject(object)) {
            text_phase = DestDrawPhase::Leftover;
            if (object->effect_count() > 0) {
                // Date +0x320>0 leftover is dest-ortho named-RT dest=I
                // (DEST_ORTHO_TNF), not TEXT_VT_F0 and not private cam.
                object->SizeDestDrawNamedRts();
                text_omit_camera = true;
            }
        }
        AddTextNodePass(node, output, imgId, extra, text_alpha_write_policy, text_phase,
                        text_omit_camera);
    }

    if (node != nullptr) {
        if (auto detached_it = scene.detachedEffectSourceNodesByLayerNode.find(node);
            detached_it != scene.detachedEffectSourceNodesByLayerNode.end()) {
            for (auto* source_node : detached_it->second) {
                if (source_node == nullptr) continue;
                LOG_INFO("SceneRenderGraphDetachedSourceRoute: world-layer=%d source-layer=%d "
                         "world-name='%s' source-name='%s' output='%.*s'",
                         NodeLayerId(scene, node),
                         NodeLayerId(scene, source_node),
                         node->Name().c_str(),
                         source_node->Name().c_str(),
                         static_cast<int>(output.size()),
                         output.data());
                ToGraphPass(source_node,
                            output,
                            NodeLayerId(scene, source_node),
                            extra,
                            {},
                            TraversalRoute {
                                .routed_node = true,
                                // Detached source nodes render through their own effect camera, but
                                // the image-effect final writer belongs to the visible world node.
                                // Forward the world route matrix so the final pass inherits virtual
                                // render-order parents while intermediate effect passes stay local.
                                .model = std::optional<Eigen::Matrix4d> { resolved_route_model },
                                .compose_source = route.compose_source,
                                .compose_source_camera =
                                    source_route.active_compose_source_camera,
                                .compose_source_alpha_write_policy =
                                    route.compose_source_alpha_write_policy });
            }
        }
    }

    std::vector<OrderedRenderGraphChild> deferred_proxy_children;
    if (node != nullptr) {
        for (const auto& child : OrderedRenderGraphChildren(node, extra)) {
            if (child.node == nullptr) continue;
            if (child.proxy && imgeff != nullptr &&
                !source_route.proxy_children_contribute_to_effect_source &&
                !IsEffectLocalProxyDependency(child.node, extra)) {
                // A render-order proxy edge only restores authored sibling order. It does not make
                // the proxied world-space node a real child of this image-effect source target, so
                // defer ordinary proxies until the parent effect has resolved back to the inherited
                // output space.
                LOG_INFO("SceneRenderGraphProxyChildDefer: parent-layer=%d proxy-layer=%d "
                         "inherited-output='%.*s' parent-effect-output='%.*s'",
                         NodeLayerId(scene, node),
                         NodeLayerId(scene, child.node),
                         static_cast<int>(inherited_output.size()),
                         inherited_output.data(),
                         static_cast<int>(output.size()),
                         output.data());
                deferred_proxy_children.push_back(child);
                continue;
            }

            if (child.proxy) {
                if (imgeff != nullptr) {
                    if (source_route.proxy_children_contribute_to_effect_source &&
                        !IsEffectLocalProxyDependency(child.node, extra)) {
                        LOG_INFO("SceneRenderGraphProxyComposeSourceRoute: parent-layer=%d "
                                 "proxy-layer=%d output='%.*s' policy=%.*s camera='%s'",
                                 NodeLayerId(scene, node),
                                 NodeLayerId(scene, child.node),
                                 static_cast<int>(output.size()),
                                 output.data(),
                                 static_cast<int>(
                                     EffectSourcePolicyName(imgeff->SourceContributionPolicy()).size()),
                                 EffectSourcePolicyName(imgeff->SourceContributionPolicy()).data(),
                                 source_route.active_compose_source_camera.c_str());
                    } else {
                        // Wallpaper Engine `dependencies` are effect-local inputs. These proxies
                        // must stay inside the parent effect phase because the parent shader samples
                        // their private source target while resolving the effect chain.
                        LOG_INFO("SceneRenderGraphProxyInlineEffectRoute: parent-layer=%d "
                                 "proxy-layer=%d output='%.*s'",
                                 NodeLayerId(scene, node),
                                 NodeLayerId(scene, child.node),
                                 static_cast<int>(output.size()),
                                 output.data());
                    }
                } else {
                    LOG_INFO("SceneRenderGraphProxyChildRoute: parent-layer=%d proxy-layer=%d "
                             "name='%s' output='%.*s'",
                             NodeLayerId(scene, node),
                             NodeLayerId(scene, child.node),
                             child.node->Name().c_str(),
                             static_cast<int>(output.size()),
                             output.data());
                }
            }
            const bool child_compose_source_route =
                route.compose_source ||
                (child.proxy && imgeff != nullptr &&
                 source_route.proxy_children_contribute_to_effect_source &&
                 !IsEffectLocalProxyDependency(child.node, extra));
            const std::string child_compose_source_camera =
                child_compose_source_route ? source_route.active_compose_source_camera
                                           : std::string();
            const AlphaWritePolicy child_compose_source_alpha_write_policy =
                route.compose_source
                    ? route.compose_source_alpha_write_policy
                    : (child_compose_source_route && imgeff != nullptr
                           ? imgeff->CompositionChildAlphaWritePolicy()
                           : AlphaWritePolicy::Preserve);
            ToGraphPass(child.node,
                        child_compose_source_route ? output : inherited_output,
                        NodeLayerId(scene, child.node),
                        extra,
                        {},
                        TraversalRoute {
                            .routed_node = child.proxy,
                            .model = BuildChildRouteModel(node, child.node, child.proxy, route.model),
                            .compose_source = child_compose_source_route,
                            .compose_source_camera = child_compose_source_camera,
                            .compose_source_alpha_write_policy =
                                child_compose_source_alpha_write_policy });
        }
    }

    auto* dest_object_for_resolve = scene.FindSceneObject(imgId);
    const bool leftover_only_image =
        dest_object_for_resolve != nullptr &&
        dest_object_for_resolve->kind() == SceneObjectKind::Image &&
        dest_object_for_resolve->effect_count() <= 0 && !route.compose_source;
    if (imgeff != nullptr && leftover_only_image) {
        // IMAGE_VT_F0 leftover +0x320==0 already published Default. Skip
        // ResolveEffect / FinalNode so the layer mesh stays +0x490 +/-half
        // (IMAGE_490_MESH). No FinalNode/HEAD restore.
        imgeff = nullptr;
    }
    if (imgeff != nullptr) {
        // Composite source nodes may now be transform-only containers whose children draw into the
        // effect source target. Resolving the effect after all descendants have emitted their passes
        // keeps those composite layers correct without requiring each child to manage effect timing.
        // Hidden dependency layers are not visible screen compositors. When a later effect samples
        // `_rt_imageLayerComposite_<id>`, the dependency's final authored effect must still resolve
        // into its private ping-pong target; otherwise the generic visible-layer rewrite would move
        // that final pass to `_rt_default`, where the hidden-layer execution gate correctly skips it.
        const auto resolved_effect_world_affine =
            Eigen::Affine3f(resolved_route_model.cast<float>());
        const bool keep_final_output_private =
            ShouldKeepEffectFinalOutputPrivate(extra, node, imgId, inherited_output);
        // Publication policy is authored by the parsed render topology, not inferred from a shader
        // name. Ordinary image/text layers keep the complete authored chain private and publish it
        // through the neutral layer-surface pass. A DIRECTDRAW shape explicitly declares its final
        // authored shader as the visible writer. Composition-source routing always overrides that
        // declaration because the child result must remain sampleable before it is placed into the
        // parent source target.
        const bool dependency_route = route.compose_source || keep_final_output_private;
        const auto final_output_capability =
            imgeff->ResolveFinalOutputCapability(dependency_route);
        const std::string_view layer_surface_camera =
            !imgeff->LayerSurfaceCamera().empty()
                ? std::string_view(imgeff->LayerSurfaceCamera())
                : (node != nullptr ? std::string_view(node->Camera()) : std::string_view());
        const auto private_target_it = scene.renderTargets.find(imgeff->FirstTarget());
        const auto final_target_it = scene.renderTargets.find(std::string(inherited_output));
        const SceneRenderTarget* private_target =
            private_target_it != scene.renderTargets.end() ? &private_target_it->second : nullptr;
        const SceneRenderTarget* final_target =
            final_target_it != scene.renderTargets.end() ? &final_target_it->second : nullptr;
        const TextureSample private_sample =
            private_target != nullptr ? private_target->sample : TextureSample {};
        const TextureSample final_sample =
            final_target != nullptr ? final_target->sample : TextureSample {};
        LOG_INFO("SceneRenderGraphEffectResolve: layer=%d name='%s' inherited-output='%.*s' "
                 "effect-output='%.*s' offscreen-dependency=%s visible=%s local-visible=%s "
                 "routed=%s keep-final-private=%s compose-source-route=%s compose-camera='%s' "
                 "capability=%.*s layer-surface-camera='%.*s' private-target='%s' "
                 "private-target-size=[%d %d] "
                 "private-sampler=[wrap-s=%.*s wrap-t=%.*s mag=%.*s min=%.*s] "
                 "final-target='%.*s' final-target-size=[%d %d] "
                 "final-sampler=[wrap-s=%.*s wrap-t=%.*s mag=%.*s min=%.*s]",
                 NodeLayerId(scene, node),
                 node != nullptr ? node->Name().c_str() : "",
                 static_cast<int>(inherited_output.size()),
                 inherited_output.data(),
                 static_cast<int>(output.size()),
                 output.data(),
                 IsOffscreenDependencyLayer(extra, imgId) ? "true" : "false",
                 node != nullptr && node->Visible() ? "true" : "false",
                 node != nullptr && node->LocalVisible() ? "true" : "false",
                 route.routed_node ? "true" : "false",
                 keep_final_output_private ? "true" : "false",
                 route.compose_source ? "true" : "false",
                 source_route.active_compose_source_camera.c_str(),
                 static_cast<int>(FinalOutputCapabilityName(final_output_capability).size()),
                 FinalOutputCapabilityName(final_output_capability).data(),
                 static_cast<int>(layer_surface_camera.size()),
                 layer_surface_camera.data(),
                 imgeff->FirstTarget().c_str(),
                 private_target != nullptr ? private_target->width : 0,
                 private_target != nullptr ? private_target->height : 0,
                 static_cast<int>(TextureWrapName(private_sample.wrapS).size()),
                 TextureWrapName(private_sample.wrapS).data(),
                 static_cast<int>(TextureWrapName(private_sample.wrapT).size()),
                 TextureWrapName(private_sample.wrapT).data(),
                 static_cast<int>(TextureFilterName(private_sample.magFilter).size()),
                 TextureFilterName(private_sample.magFilter).data(),
                 static_cast<int>(TextureFilterName(private_sample.minFilter).size()),
                 TextureFilterName(private_sample.minFilter).data(),
                 static_cast<int>(inherited_output.size()),
                 inherited_output.data(),
                 final_target != nullptr ? final_target->width : 0,
                 final_target != nullptr ? final_target->height : 0,
                 static_cast<int>(TextureWrapName(final_sample.wrapS).size()),
                 TextureWrapName(final_sample.wrapS).data(),
                 static_cast<int>(TextureWrapName(final_sample.wrapT).size()),
                 TextureWrapName(final_sample.wrapT).data(),
                 static_cast<int>(TextureFilterName(final_sample.magFilter).size()),
                 TextureFilterName(final_sample.magFilter).data(),
                 static_cast<int>(TextureFilterName(final_sample.minFilter).size()),
                 TextureFilterName(final_sample.minFilter).data());
        imgeff->ResolveEffect(scene.default_effect_mesh,
                              "effect",
                              layer_surface_camera,
                              inherited_output,
                              keep_final_output_private,
                              &resolved_effect_world_affine,
                              final_output_capability);

        auto* dest_object = scene.FindSceneObject(imgId);
        const bool dest_draw_effects = dest_object != nullptr && dest_object->DestDrawHasEffects();
        const bool dest_draw_last_pass = DestDrawPublishesDefault(dest_object);

        for (usize i = 0; i < imgeff->EffectCount(); i++) {
            auto& eff     = imgeff->GetEffect(i);
            auto  cmdItor = eff->commands.begin();
            auto  cmdEnd  = eff->commands.end();
            int   nodePos = 0;
            auto  effect_visible_gate = [eff]() {
                return eff == nullptr || eff->LocalVisible();
            };
            auto effect_hidden_gate = [eff]() {
                return eff != nullptr && !eff->LocalVisible();
            };
            const bool last_effect = (i + 1 == imgeff->EffectCount());
            for (auto& effect_node : eff->nodes) {
                if (cmdItor != cmdEnd && nodePos == cmdItor->afterpos) {
                    // LEFTOVER_VS_DESTDRAW: leftover then POSTFX copies then
                    // last-pass are one dest-draw. Copies left in m_passes run
                    // after DestDraw last-pass and feed empty FullCompo.
                    rg::addCopyPass(*extra.rgraph,
                                    rg::createTexDesc(cmdItor->src, extra.scene),
                                    rg::createTexDesc(cmdItor->dst, extra.scene),
                                    effect_visible_gate,
                                    dest_draw_effects ? DestDrawPhase::PostFx
                                                      : DestDrawPhase::None,
                                    imgId);
                    cmdItor++;
                }
                // Effect material nodes are private render-graph passes. They can carry a camera
                // override for layer-surface publication, but they must not be traversed through
                // ToGraphPass(): an override camera may itself own the same SceneImageEffectLayer,
                // and treating this internal pass as another image-effect owner recursively
                // rebuilds the chain while it is being emitted, overwriting the already-resolved
                // puppet and layer-surface publication route.
                std::string_view effect_output = effect_node.output;
                NodePassOptions effect_options {
                    .alpha_write_policy = effect_node.alpha_write_policy,
                    .clear_before_draw = effect_node.clear_before_draw,
                    .camera_override = effect_node.camera_override,
                    .use_active_camera_for_parallax =
                        effect_node.use_active_camera_for_parallax,
                    .use_identity_model = effect_node.use_identity_model };
                if (dest_draw_effects) {
                    const bool last_node = (&effect_node == &eff->nodes.back());
                    if (dest_draw_last_pass && last_effect && last_node) {
                        // POSTFX_OMSET: VERTICAL no-target keeps leftover FullFB
                        // after HORIZONTAL pop (compose Default). POSTFX_MESH
                        // +0x2e8. Date last-pass g_MVP is +0x930 (VERTICAL_MVP_ID).
                        // IMAGE last-pass combo uploads +0x8f0 (LASTPASS_IMAGE_ID
                        // / 0x1400d8749). Not I-slot ping-pong / FinalNode blit.
                        effect_options.dest_draw_phase = DestDrawPhase::LastPass;
                        effect_options.camera_override.clear();
                        effect_options.omit_layer_camera = true;
                        effect_options.use_identity_model = true;
                        effect_output = SpecTex_Default;
                        // POSTFX_OMSET pass1 bind is HORIZONTAL
                        // `_rt_FullCompoBuffer1`. Remapping last output to
                        // Default must not leave g_Texture0 as Default (PrePass
                        // clear) or the unwritten last-node ping-pong.
                        std::string previous;
                        for (auto it = eff->nodes.begin(); it != eff->nodes.end(); ++it) {
                            if (&*it == &effect_node) break;
                            if (!it->output.empty()) previous = it->output;
                        }
                        if (previous.empty()) previous = imgeff->FirstTarget();
                        if (effect_node.sceneNode != nullptr &&
                            effect_node.sceneNode->Mesh() != nullptr &&
                            effect_node.sceneNode->Mesh()->Material() != nullptr) {
                            auto* last_mat = effect_node.sceneNode->Mesh()->Material();
                            // DEST_1F0_WRITERS 0x1401ea0b6: last-pass Draw uses
                            // dest +0x1f0 (IMAGE_DEST_BLEND / FinalBlend), not
                            // ResolveEffectPingPongChain Normal. LASTPASS_WRITES
                            // 0x1401ec209 restores combo+0x1f0 after Draw.
                            // Forced Normal is ONE/ZERO/DONT_CARE and punches
                            // leftover FullFB with transparent-black dest-size.
                            last_mat->blenmode = imgeff->FinalBlend();
                            if (!previous.empty()) {
                                const std::string last_out = effect_node.output;
                                for (auto& tex : last_mat->textures) {
                                    if (tex == SpecTex_Default || tex == last_out) {
                                        tex = previous;
                                    }
                                }
                            }
                        }
                        if (dest_object->lastpass_mesh() != nullptr &&
                            effect_node.sceneNode != nullptr &&
                            effect_node.sceneNode->Mesh() != nullptr) {
                            effect_node.sceneNode->Mesh()->ChangeMeshDataFrom(
                                *dest_object->lastpass_mesh());
                            effect_node.sceneNode->Mesh()->SetDirty();
                        }
                    } else {
                        // POSTFX_MESH: HORIZONTAL esi!=+0x144 → +0x2e0 ±1.
                        // I-internal (I_SLOT). omit_layer_camera: leftover
                        // WorldNode camera is not dest-ortho / not I.
                        effect_options.dest_draw_phase = DestDrawPhase::PostFx;
                        effect_options.camera_override.clear();
                        effect_options.omit_layer_camera = true;
                        effect_options.use_identity_model = true;
                        if (dest_object->postfx_mesh() != nullptr &&
                            effect_node.sceneNode != nullptr &&
                            effect_node.sceneNode->Mesh() != nullptr) {
                            effect_node.sceneNode->Mesh()->ChangeMeshDataFrom(
                                *dest_object->postfx_mesh());
                            effect_node.sceneNode->Mesh()->SetDirty();
                        }
                    }
                }
                AddNodePass(effect_node.sceneNode.get(),
                            effect_output,
                            imgId,
                            extra,
                            effect_visible_gate,
                            effect_options);
                nodePos++;
            }
            if (!eff->BypassSource().empty() && !eff->BypassTarget().empty() &&
                eff->BypassSource() != eff->BypassTarget()) {
                // Hidden effects must still advance the ping-pong chain. The shader and authored
                // effect commands become no-ops through their local visibility, then this gated copy
                // forwards the current input target to the effect output target so downstream
                // effects sample the current frame, not the stale texture from the last visible
                // frame.
                rg::addCopyPass(*extra.rgraph,
                                rg::createTexDesc(eff->BypassSource(), extra.scene),
                                rg::createTexDesc(eff->BypassTarget(), extra.scene),
                                effect_hidden_gate,
                                dest_draw_effects ? DestDrawPhase::PostFx
                                                  : DestDrawPhase::None,
                                imgId);
            }
        }

        if (dest_object != nullptr && dest_object->Flag304Bit4() && dest_draw_effects &&
            !route.compose_source) {
            // IMAGE_VT_F8 / DRAW_FLAG304: bit4 leftover-MVP onto leftover
            // FullFB after leftover dest-ortho + in-loop dest-draw. Not
            // POSTFX last-pass dest-card of leftover RT (0x1401ea151).
            // Official leftover dest-ortho Draws +0x2d8; leftover-MVP Draws
            // +0x490 (PUPPET_490 / IMAGE_VT_F8_PUPPET). Bind +0x490 on a
            // leftover-MVP node so prepare/upload is not leftover dest card.
            NodePassOptions leftover_mvp {
                .use_identity_model = true,
                .dest_draw_phase = DestDrawPhase::LeftoverMvp,
                .omit_layer_camera = true,
            };
            SceneNode* leftover_mvp_node = dest_object->EnsureLeftoverMvpNode(node);
            if (leftover_mvp_node != nullptr) {
                AddNodePass(leftover_mvp_node,
                            SpecTex_Default,
                            imgId,
                            extra,
                            node_execute_gate,
                            leftover_mvp);
            }
        }

        if (imgeff->HasFinalComposite() && !dest_draw_effects) {
            auto final_composite_gate = [imgeff]() {
                return imgeff != nullptr && imgeff->ShouldRunFinalComposite();
            };
            ToGraphPass(&imgeff->FinalNode(),
                        inherited_output,
                        imgId,
                        extra,
                        final_composite_gate,
                        TraversalRoute {
                            // The synthetic fallback is another detached final writer for the same
                            // image effect, so it must share the resolved route matrix used by the
                            // authored final output instead of recomputing from its physical tree.
                            .model = std::optional<Eigen::Matrix4d> { resolved_route_model },
                            .compose_source = route.compose_source,
                            .compose_source_camera = source_route.active_compose_source_camera,
                            .compose_source_alpha_write_policy =
                                route.compose_source_alpha_write_policy });
        }
    }

    for (const auto& child : deferred_proxy_children) {
        if (child.node == nullptr) continue;
        LOG_INFO("SceneRenderGraphProxyOutputRoute: parent-layer=%d proxy-layer=%d output='%.*s'",
                 NodeLayerId(scene, node),
                 NodeLayerId(scene, child.node),
                 static_cast<int>(inherited_output.size()),
                 inherited_output.data());
        ToGraphPass(child.node,
                    inherited_output,
                    NodeLayerId(scene, child.node),
                    extra,
                    {},
                    TraversalRoute {
                        .routed_node = true,
                        .model = BuildChildRouteModel(node, child.node, true, route.model),
                        .compose_source = route.compose_source,
                        .compose_source_camera = source_route.active_compose_source_camera,
                        .compose_source_alpha_write_policy =
                            route.compose_source_alpha_write_policy });
    }
}

static std::unique_ptr<rg::RenderGraph> SceneToRenderGraphImpl(
    Scene& scene, bool include_hidden_for_pipeline_warmup,
    std::vector<SceneRenderGraphPassRecord>* inventory) {
    std::unique_ptr<rg::RenderGraph> rgraph = std::make_unique<rg::RenderGraph>();
    ExtraInfo                        extra { .rgraph = rgraph.get(),
                                             .scene = &scene,
                                             .include_hidden_for_pipeline_warmup =
                                                 include_hidden_for_pipeline_warmup,
                                             .inventory = inventory };
    for (size_t index = 0; index < scene.layerOrder.size(); index++) {
        extra.layer_order_index[scene.layerOrder[index]] = index;
    }
    LOG_INFO("SceneRenderGraphOrderInit: layer-count=%zu proxy-parent-count=%zu proxy-node-count=%zu "
             "detached-anchor-count=%zu detached-source-count=%zu warmup-hidden=%s",
             scene.layerOrder.size(),
             scene.renderOrderProxyChildren.size(),
             scene.renderOrderProxyNodes.size(),
             scene.detachedEffectSourceNodesByLayerNode.size(),
             scene.detachedEffectSourceNodes.size(),
             include_hidden_for_pipeline_warmup ? "true" : "false");
    if (scene.renderTargets.count(std::string(SpecTex_Reflection)) != 0) {
        // Keep the official empty-buffer contract when reflections are off: receivers still
        // sample `_rt_Reflection`, so the target stays registered and is cleared instead of
        // destroyed. The clear is a no-op while the quality checkbox is on.
        rg::addClearPass(*rgraph,
                         rg::createTexDesc(std::string(SpecTex_Reflection), &scene),
                         { 0.0f, 0.0f, 0.0f, 0.0f },
                         [&scene]() { return !scene.reflectionsEnabled; });
    }
    ToGraphPass(scene.sceneGraph.get(), SpecTex_Default, scene.sceneGraph->ID(), extra);

    for (auto& info : extra.link_info) {
        if (! exists(extra.id_link_map, info.link_id)) {
            LOG_ERROR("link tex %d not found", info.link_id);
            continue;
        }
        rgraph->afterBuild(
            info.id, [&rgraph, &extra, &info](rg::RenderGraphBuilder& builder, rg::Pass& rgpass) {
                auto* link_tex_node = extra.id_link_map.at(info.link_id);
                auto  copy_desc     = link_tex_node->genDesc();
                copy_desc.key       = GenLinkTex((idx)info.link_id);
                copy_desc.name      = copy_desc.key;

                auto new_in = rg::addCopyPass(*rgraph, link_tex_node, &copy_desc);
                builder.read(new_in);
                if (info.bind_texture != nullptr) {
                    info.bind_texture(rgpass, (u32)info.tex_index, new_in->key());
                }
                return true;
            });
    }

    if (extra.use_mipmap_framebuffer) {
        rg::addCopyPass(*rgraph,
                        rg::TexNode::Desc { .name = SpecTex_Default.data(),
                                            .key  = SpecTex_Default.data(),
                                            .type = rg::TexNode::TexType::Temp },
                        rg::TexNode::Desc { .name = WE_MIP_MAPPED_FRAME_BUFFER.data(),
                                            .key  = WE_MIP_MAPPED_FRAME_BUFFER.data(),
                                            .type = rg::TexNode::TexType::Temp });
    }

    if (scene.shadows.quality != 0) {
        rg::addShadowAtlasPass(*rgraph, &scene);
    }

    if (scene.volumetrics.active && !scene.volumetrics.lights.empty()) {
        // Volumetrics then bloom/HDR. Each light writes LightBuffer additively; blur/combine
        // run after every light has finished.
        rg::addClearPass(*rgraph,
                         rg::createTexDesc(std::string(SpecTex_VolumetricsLightBuffer), &scene),
                         { 0.0f, 0.0f, 0.0f, 0.0f });
        rg::addVolumetricsSingleFillPass(*rgraph, &scene);
        for (const auto& pass : scene.volumetrics.lights) {
            if (pass.light == nullptr) continue;
            SceneLight* light = pass.light;
            auto camera_inside = [&scene, light]() {
                if (scene.activeCamera == nullptr || light == nullptr) return false;
                return light->CameraInsideVolume(scene.activeCamera->GetPosition().cast<float>(),
                                                 scene.activeCamera->GetDirection().cast<float>());
            };
            if (pass.back) {
                AddNodePass(pass.back.get(), SpecTex_VolumetricsBack, 0, extra);
            }
            if (pass.front) {
                AddNodePass(pass.front.get(),
                            SpecTex_VolumetricsLightBuffer,
                            0,
                            extra,
                            [camera_inside]() { return ! camera_inside(); });
            }
            if (pass.fullscreen) {
                AddNodePass(pass.fullscreen.get(),
                            SpecTex_VolumetricsLightBuffer,
                            0,
                            extra,
                            camera_inside);
            }
        }
        if (scene.volumetrics.quality < 3 && scene.volumetrics.blur_h &&
            scene.volumetrics.blur_v) {
            AddNodePass(scene.volumetrics.blur_h.get(), SpecTex_VolumetricsLightBufferB, 0, extra);
            AddNodePass(scene.volumetrics.blur_v.get(), SpecTex_VolumetricsLightBuffer, 0, extra);
        }
        if (scene.volumetrics.combine) {
            AddNodePass(scene.volumetrics.combine.get(), SpecTex_Default, 0, extra);
        }
    }

    if (scene.bloom.quality > 0 && scene.bloom.enabled && !scene.bloom.nodes.empty()) {
        // Wallpaper Engine treats `general.bloom` as the execution switch for the complete Bloom
        // chain. Authored strength and threshold values remain stored while the switch is off, but
        // they do not make the post-process execute. Keeping disabled Bloom out of the graph avoids
        // three private blur targets, one full-frame feedback copy, and four GPU passes while still
        // retaining the parsed nodes for a later topology rebuild when a runtime binding enables it.
        // Scene Bloom is authored in `general`, so an enabled chain belongs after the complete layer
        // traversal, with each synthetic node bound to its explicit quarter/eighth output target.
        if (scene.bloom.nodes.size() != scene.bloom.outputs.size()) {
            LOG_ERROR("SceneBloomGraphBind: pass/output mismatch passes=%zu outputs=%zu",
                      scene.bloom.nodes.size(),
                      scene.bloom.outputs.size());
        } else {
            LOG_INFO("SceneBloomGraphBind: passes=%zu enabled=%s strength=%.3f threshold=%.3f",
                     scene.bloom.nodes.size(),
                     scene.bloom.enabled ? "true" : "false",
                     scene.bloom.strength,
                     scene.bloom.threshold);
            for (usize i = 0; i < scene.bloom.nodes.size(); ++i) {
                if (scene.bloom.nodes[i] == nullptr) {
                    LOG_ERROR("SceneBloomGraphBind: missing pass index=%zu output='%s'",
                              i,
                              scene.bloom.outputs[i].c_str());
                    continue;
                }
                LOG_INFO("SceneBloomGraphBind: pass=%zu node='%s' output='%s'",
                         i,
                         scene.bloom.nodes[i]->Name().c_str(),
                         scene.bloom.outputs[i].c_str());
                AddNodePass(scene.bloom.nodes[i].get(), scene.bloom.outputs[i], 0, extra);
            }
        }
    } else if (scene.bloom.quality > 0 && scene.bloom.enabled && scene.bloom.node != nullptr) {
        // This fallback is intentionally retained for older parsed scene objects that may still
        // populate only the legacy single-node field before a full reparse has occurred.
        LOG_INFO("SceneBloomGraphBind: legacy-output='%s' enabled=%s strength=%.3f threshold=%.3f",
                 SpecTex_Default.data(),
                 scene.bloom.enabled ? "true" : "false",
                 scene.bloom.strength,
                 scene.bloom.threshold);
        AddNodePass(scene.bloom.node.get(), SpecTex_Default, 0, extra);
    }

    return rgraph;
}

std::unique_ptr<rg::RenderGraph> wallpaper::sceneToRenderGraph(Scene& scene) {
    return SceneToRenderGraphImpl(scene, false, nullptr);
}

std::unique_ptr<rg::RenderGraph> wallpaper::sceneToRenderGraph(
    Scene& scene, std::vector<SceneRenderGraphPassRecord>* inventory) {
    return SceneToRenderGraphImpl(scene, false, inventory);
}

std::unique_ptr<rg::RenderGraph> wallpaper::sceneToPipelineWarmupRenderGraph(Scene& scene) {
    return SceneToRenderGraphImpl(scene, true, nullptr);
}
