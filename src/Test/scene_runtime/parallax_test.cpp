#include "test_harness.h"

#include "Scene/Scene.h"
#include "Scene/SceneCamera.h"
#include "Scene/SceneMesh.h"
#include "Scene/SceneNode.h"
#include "Scene/SceneObject.h"
#include "Scene/SceneShader.h"
#include "SpecTexs.hpp"
#include "WPNodeTransformResolver.hpp"
#include "WPShaderValueUpdater.hpp"

#include <array>
#include <cmath>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace wallpaper;

namespace
{

constexpr float kOrthoW     = 3840.0f;
constexpr float kOrthoH     = 2160.0f;
constexpr float kCamX       = 1920.0f;
constexpr float kCamY       = 1080.0f;
constexpr float kAmount     = 0.2f;
constexpr float kDelay      = 2.0f;
constexpr float kInfluence  = 0.3f;
constexpr float kBgX        = 1964.01025f;
constexpr float kBgY        = 1211.59460f;
constexpr float kBgDepth    = -0.92f;
constexpr float kFwX        = 619.40967f;
constexpr float kFwY        = 31.15491f;
constexpr float kBodyX      = 1691.51270f;
constexpr float kBodyY      = 1305.29688f;
constexpr float kBodyDepthX = -0.52f;
constexpr float kBodyDepthY = -0.16f;

struct ParallaxFixture {
    Scene                             scene;
    std::shared_ptr<SceneNode>        camera_node;
    std::shared_ptr<SceneCamera>      camera;
    Map<void*, WPShaderValueData>     node_data;
    Map<void*, Eigen::Matrix4d>       model_cache;
    Map<void*, Eigen::Vector3f>       parallax_cache;
    Map<void*, Eigen::Affine3f>       attach_cache;
    WPCameraParallax                  parallax { true, kAmount, kDelay, kInfluence };
    std::array<float, 2>              mouse { 0.5f, 0.5f };
    std::array<float, 2>              lookat { kCamX, kCamY };

    ParallaxFixture() {
        scene.ortho[0] = static_cast<i32>(kOrthoW);
        scene.ortho[1] = static_cast<i32>(kOrthoH);
        camera_node    = std::make_shared<SceneNode>(Eigen::Vector3f { kCamX, kCamY, 0.0f },
                                                  Eigen::Vector3f { 1.0f, 1.0f, 1.0f },
                                                  Eigen::Vector3f::Zero());
        camera = std::make_shared<SceneCamera>(static_cast<i32>(kOrthoW),
                                               static_cast<i32>(kOrthoH),
                                               -1.0f,
                                               1.0f);
        camera->AttatchNode(camera_node);
        camera->Update();
        scene.cameras["global"] = camera;
        scene.activeCamera      = camera.get();
    }

    std::shared_ptr<SceneNode> MakeNode(float x, float y) {
        return std::make_shared<SceneNode>(Eigen::Vector3f { x, y, 0.0f },
                                           Eigen::Vector3f { 1.0f, 1.0f, 1.0f },
                                           Eigen::Vector3f::Zero());
    }

    void Bind(SceneNode* node, WPShaderValueData data) {
        if (node == nullptr) return;
        if (node->ID() == 0) {
            static int32_t next_id = 900000;
            node->ID()             = next_id++;
        }
        node_data[node]          = std::move(data);
        scene.nodeOwners[node]   = node->ID();
        auto& object             = scene.EnsureSceneObject(node->ID());
        object.set_origin(node->Translate());
        object.set_parallax_depth(Eigen::Vector2f(node_data[node].parallaxDepth[0],
                                                  node_data[node].parallaxDepth[1]));
        object.set_parallax_depth_authored(node_data[node].parallaxDepthAuthored);
        object.set_source_node(node);
        if (node_data[node].parallax_anchor != nullptr) {
            const auto parent_it = scene.nodeOwners.find(node_data[node].parallax_anchor);
            if (parent_it != scene.nodeOwners.end()) {
                scene.BindSceneObjectParent(node->ID(), parent_it->second, {});
            }
        }
    }

    void ClearCaches() {
        model_cache.clear();
        parallax_cache.clear();
        attach_cache.clear();
    }

    WPNodeTransformResolver Resolver() {
        const auto world = OfficialLookatFromViewCamera(
            { kCamX, kCamY }, { mouse[0], mouse[1] }, { kOrthoW, kOrthoH },
            parallax.mouseinfluence);
        const std::array<float, 2> parallax_lookat { world.x(), world.y() };
        return WPNodeTransformResolver(scene,
                                       parallax,
                                       node_data,
                                       model_cache,
                                       parallax_cache,
                                       attach_cache,
                                       camera.get(),
                                       parallax_lookat,
                                       1);
    }

    Eigen::Vector3f Offset(SceneNode* node) {
        auto resolver = Resolver();
        return resolver.ResolveParallaxOffset(node, camera.get());
    }

    Eigen::Matrix4d ParallaxedModel(SceneNode* node) {
        auto resolver = Resolver();
        return resolver.ResolveParallaxedModelTransform(node, camera.get(), true);
    }
};

void ExpectVec2(const Eigen::Vector3f& got, float x, float y, float eps = 1e-4f) {
    SCENE_CHECK_NEAR(got.x(), x, eps);
    SCENE_CHECK_NEAR(got.y(), y, eps);
    SCENE_CHECK_NEAR(got.z(), 0.0f, eps);
}

void ExpectVec2(const Eigen::Vector3f& got, const Eigen::Vector2f& expected, float eps = 1e-4f) {
    ExpectVec2(got, expected.x(), expected.y(), eps);
}

} // namespace

void TestParallaxDisabledIsZero() {
    SceneTestBegin("Parallax.DisabledIsZero");
    ParallaxFixture fx;
    fx.parallax.enable = false;
    auto node          = fx.MakeNode(kBgX, kBgY);
    WPShaderValueData data;
    data.SetParallaxContract({ kBgDepth, kBgDepth }, nullptr, false, true);
    fx.Bind(node.get(), data);
    fx.mouse = { 0.0f, 1.0f };
    ExpectVec2(fx.Offset(node.get()), 0.0f, 0.0f);
}

void TestParallaxAmountZeroIsZero() {
    SceneTestBegin("Parallax.AmountZeroIsZero");
    ParallaxFixture fx;
    fx.parallax.amount = 0.0f;
    auto node          = fx.MakeNode(kBgX, kBgY);
    WPShaderValueData data;
    data.SetParallaxContract({ kBgDepth, kBgDepth }, nullptr, false, true);
    fx.Bind(node.get(), data);
    fx.mouse = { 0.0f, 1.0f };
    ExpectVec2(fx.Offset(node.get()), 0.0f, 0.0f);
}

void TestParallaxRestMouseMatchesCenteredCamera() {
    SceneTestBegin("Parallax.RestMouseUsesNodeMinusCamera");
    ParallaxFixture fx;
    auto            node = fx.MakeNode(kBgX, kBgY);
    WPShaderValueData data;
    data.SetParallaxContract({ kBgDepth, kBgDepth }, nullptr, false, true);
    fx.Bind(node.get(), data);
    fx.mouse = { 0.5f, 0.5f };

    const auto expected = ExpectedPathBOffset({ kBgX, kBgY },
                                              { kCamX, kCamY },
                                              { 0.5f, 0.5f },
                                              { kOrthoW, kOrthoH },
                                              { kBgDepth, kBgDepth },
                                              kAmount,
                                              kInfluence);
    const auto got = fx.Offset(node.get());
    SceneReportVec2("rest mouse PathB", got.x(), got.y(), expected.x(), expected.y());
    ExpectVec2(got, expected);
    SCENE_CHECK(std::fabs(expected.x()) > 1.0f || std::fabs(expected.y()) > 1.0f);
}

void TestParallaxMouseInfluenceAndYFlip() {
    SceneTestBegin("Parallax.MouseInfluenceAndYFlip");
    ParallaxFixture fx;
    auto            node = fx.MakeNode(kBgX, kBgY);
    WPShaderValueData data;
    data.SetParallaxContract({ kBgDepth, kBgDepth }, nullptr, false, true);
    fx.Bind(node.get(), data);

    const Eigen::Vector2f left  { 0.0f, 0.5f };
    const Eigen::Vector2f right { 1.0f, 0.5f };
    const Eigen::Vector2f top   { 0.5f, 0.0f };
    const Eigen::Vector2f bot   { 0.5f, 1.0f };

    fx.mouse = { left.x(), left.y() };
    const auto left_off = fx.Offset(node.get());
    const auto left_exp = ExpectedPathBOffset({ kBgX, kBgY },
                                              { kCamX, kCamY },
                                              left,
                                              { kOrthoW, kOrthoH },
                                              { kBgDepth, kBgDepth },
                                              kAmount,
                                              kInfluence);
    SceneReportVec2("mouse left PathB", left_off.x(), left_off.y(), left_exp.x(), left_exp.y());
    ExpectVec2(left_off, left_exp);

    fx.ClearCaches();
    fx.mouse = { right.x(), right.y() };
    const auto right_off = fx.Offset(node.get());
    ExpectVec2(right_off,
               ExpectedPathBOffset({ kBgX, kBgY },
                                   { kCamX, kCamY },
                                   right,
                                   { kOrthoW, kOrthoH },
                                   { kBgDepth, kBgDepth },
                                   kAmount,
                                   kInfluence));
    SCENE_CHECK(left_off.x() != right_off.x());

    fx.ClearCaches();
    fx.mouse = { top.x(), top.y() };
    const auto top_off = fx.Offset(node.get());
    fx.ClearCaches();
    fx.mouse = { bot.x(), bot.y() };
    const auto bot_off = fx.Offset(node.get());
    ExpectVec2(top_off,
               ExpectedPathBOffset({ kBgX, kBgY },
                                   { kCamX, kCamY },
                                   top,
                                   { kOrthoW, kOrthoH },
                                   { kBgDepth, kBgDepth },
                                   kAmount,
                                   kInfluence));
    ExpectVec2(bot_off,
               ExpectedPathBOffset({ kBgX, kBgY },
                                   { kCamX, kCamY },
                                   bot,
                                   { kOrthoW, kOrthoH },
                                   { kBgDepth, kBgDepth },
                                   kAmount,
                                   kInfluence));
    SCENE_CHECK(top_off.y() != bot_off.y());

    fx.ClearCaches();
    fx.parallax.mouseinfluence = 0.0f;
    fx.mouse                   = { 0.0f, 1.0f };
    const auto no_mouse        = fx.Offset(node.get());
    const auto rest            = ExpectedPathBOffset({ kBgX, kBgY },
                                          { kCamX, kCamY },
                                          { 0.5f, 0.5f },
                                          { kOrthoW, kOrthoH },
                                          { kBgDepth, kBgDepth },
                                          kAmount,
                                          0.0f);
    ExpectVec2(no_mouse, rest);
}

void TestParallaxAsymmetricDepth() {
    SceneTestBegin("Parallax.AsymmetricDepthUsesPerAxis");
    ParallaxFixture fx;
    auto            node = fx.MakeNode(kBodyX, kBodyY);
    WPShaderValueData data;
    data.SetParallaxContract({ kBodyDepthX, kBodyDepthY }, nullptr, false, true);
    fx.Bind(node.get(), data);
    fx.mouse = { 0.0f, 1.0f };
    const auto expected = ExpectedPathBOffset({ kBodyX, kBodyY },
                                              { kCamX, kCamY },
                                              { 0.0f, 1.0f },
                                              { kOrthoW, kOrthoH },
                                              { kBodyDepthX, kBodyDepthY },
                                              kAmount,
                                              kInfluence);
    const auto got = fx.Offset(node.get());
    SceneReportVec2("asymmetric PathB", got.x(), got.y(), expected.x(), expected.y());
    ExpectVec2(got, expected);
}

void TestParallaxZeroDepthStaysPut() {
    SceneTestBegin("Parallax.ZeroDepthStaysPut");
    ParallaxFixture fx;
    auto            node = fx.MakeNode(kBgX, kBgY);
    WPShaderValueData data;
    data.SetParallaxContract({ 0.0f, 0.0f }, nullptr, false, true);
    fx.Bind(node.get(), data);
    fx.mouse = { 0.0f, 1.0f };
    ExpectVec2(fx.Offset(node.get()), 0.0f, 0.0f);
}

void TestParallaxOmittedChildEqualsRootNotOwnOrigin() {
    SceneTestBegin("Parallax.OmittedChildEqualsRootNotOwnOrigin");
    ParallaxFixture fx;
    auto            root  = fx.MakeNode(kBgX, kBgY);
    auto            child = fx.MakeNode(kFwX, kFwY);
    root->AppendChild(child);

    WPShaderValueData root_data;
    root_data.SetParallaxContract({ kBgDepth, kBgDepth }, nullptr, false, true);
    WPShaderValueData child_data;
    child_data.SetParallaxContract({ 1.0f, 1.0f }, root.get(), false, false);
    fx.Bind(root.get(), root_data);
    fx.Bind(child.get(), child_data);
    fx.mouse = { 0.0f, 1.0f };

    const auto root_off  = fx.Offset(root.get());
    const auto child_off = fx.Offset(child.get());
    const auto expected  = ExpectedPathBOffset({ kBgX, kBgY },
                                              { kCamX, kCamY },
                                              { 0.0f, 1.0f },
                                              { kOrthoW, kOrthoH },
                                              { kBgDepth, kBgDepth },
                                              kAmount,
                                              kInfluence);
    SceneReportVec2("root PathB", root_off.x(), root_off.y(), expected.x(), expected.y());
    SceneReportVec2("omitted child inherit", child_off.x(), child_off.y(), expected.x(), expected.y());
    ExpectVec2(root_off, expected);
    ExpectVec2(child_off, expected);

    const auto unofficial = ExpectedPathBOffset({ kFwX, kFwY },
                                                { kCamX, kCamY },
                                                { 0.0f, 1.0f },
                                                { kOrthoW, kOrthoH },
                                                { 1.0f, 1.0f },
                                                kAmount,
                                                kInfluence);
    SCENE_CHECK(std::fabs(child_off.x() - unofficial.x()) > 1.0f ||
                std::fabs(child_off.y() - unofficial.y()) > 1.0f);
}

void TestParallaxAuthoredChildKeepsOwnOrigin() {
    SceneTestBegin("Parallax.AuthoredChildKeepsOwnOrigin");
    ParallaxFixture fx;
    auto            root  = fx.MakeNode(kBgX, kBgY);
    auto            child = fx.MakeNode(kFwX, kFwY);
    WPShaderValueData root_data;
    root_data.SetParallaxContract({ kBgDepth, kBgDepth }, nullptr, false, true);
    WPShaderValueData child_data;
    child_data.SetParallaxContract({ -0.4f, -0.4f }, nullptr, false, true);
    fx.Bind(root.get(), root_data);
    fx.Bind(child.get(), child_data);
    fx.mouse = { 0.0f, 1.0f };

    ExpectVec2(fx.Offset(child.get()),
               ExpectedPathBOffset({ kFwX, kFwY },
                                   { kCamX, kCamY },
                                   { 0.0f, 1.0f },
                                   { kOrthoW, kOrthoH },
                                   { -0.4f, -0.4f },
                                   kAmount,
                                   kInfluence));
    fx.ClearCaches();
    const auto root_off = fx.Offset(root.get());
    SCENE_CHECK(std::fabs(fx.Offset(child.get()).x() - root_off.x()) > 1.0f ||
                std::fabs(fx.Offset(child.get()).y() - root_off.y()) > 1.0f);
}

void TestParallaxGrandchildWalksToRoot() {
    SceneTestBegin("Parallax.GrandchildWalksToRoot");
    ParallaxFixture fx;
    auto            root  = fx.MakeNode(kBgX, kBgY);
    auto            mid   = fx.MakeNode(100.0f, 200.0f);
    auto            leaf  = fx.MakeNode(kFwX, kFwY);
    WPShaderValueData root_data;
    root_data.SetParallaxContract({ kBgDepth, kBgDepth }, nullptr, false, true);
    WPShaderValueData mid_data;
    mid_data.SetParallaxContract({ 1.0f, 1.0f }, root.get(), false, false);
    WPShaderValueData leaf_data;
    leaf_data.SetParallaxContract({ 1.0f, 1.0f }, mid.get(), false, false);
    fx.Bind(root.get(), root_data);
    fx.Bind(mid.get(), mid_data);
    fx.Bind(leaf.get(), leaf_data);
    fx.mouse = { 0.25f, 0.75f };

    const auto expected = ExpectedPathBOffset({ kBgX, kBgY },
                                              { kCamX, kCamY },
                                              { 0.25f, 0.75f },
                                              { kOrthoW, kOrthoH },
                                              { kBgDepth, kBgDepth },
                                              kAmount,
                                              kInfluence);
    ExpectVec2(fx.Offset(leaf.get()), expected);
    ExpectVec2(fx.Offset(mid.get()), expected);
    ExpectVec2(fx.Offset(root.get()), expected);
}

void TestParallaxBoneAttachedAnchorDoesNotRecurse() {
    SceneTestBegin("Parallax.BoneAttachedAnchorDoesNotRecurse");
    ParallaxFixture fx;
    auto            parent = fx.MakeNode(kBodyX, kBodyY);
    auto            child  = fx.MakeNode(kFwX, kFwY);
    WPShaderValueData parent_data;
    parent_data.SetParallaxContract({ kBodyDepthX, kBodyDepthY }, nullptr, false, true);
    parent_data.AttachToBone(parent.get(), 4u, Eigen::Affine3f::Identity(), Eigen::Affine3f::Identity());
    WPShaderValueData child_data;
    child_data.SetParallaxContract({ 1.0f, 1.0f }, parent.get(), false, false);
    fx.Bind(parent.get(), parent_data);
    fx.Bind(child.get(), child_data);
    fx.mouse = { 0.0f, 1.0f };
    // Official Path B has no bone skip (0x14018b062). Child walks +0x180 to ROOT parent.
    const auto expected = ExpectedPathBOffset({ kBodyX, kBodyY },
                                              { kCamX, kCamY },
                                              { 0.0f, 1.0f },
                                              { kOrthoW, kOrthoH },
                                              { kBodyDepthX, kBodyDepthY },
                                              kAmount,
                                              kInfluence);
    ExpectVec2(fx.Offset(child.get()), expected);
}

void TestParallaxSuppressKeepsModelTranslation() {
    SceneTestBegin("Parallax.SuppressKeepsModelTranslation");
    ParallaxFixture fx;
    auto            node = fx.MakeNode(kBgX, kBgY);
    WPShaderValueData data;
    data.SetParallaxContract({ kBgDepth, kBgDepth }, nullptr, true, true);
    fx.Bind(node.get(), data);
    fx.mouse = { 0.0f, 1.0f };

    const auto offset = fx.Offset(node.get());
    const auto expected = ExpectedPathBOffset({ kBgX, kBgY },
                                              { kCamX, kCamY },
                                              { 0.0f, 1.0f },
                                              { kOrthoW, kOrthoH },
                                              { kBgDepth, kBgDepth },
                                              kAmount,
                                              kInfluence);
    ExpectVec2(offset, expected);
    SCENE_CHECK(data.AppliesModelParallax() == false);

    // leftover_suppress is unofficial. Path B is dest-STACK, not model.
    const auto model = fx.ParallaxedModel(node.get());
    SCENE_CHECK_NEAR(model(0, 3), static_cast<double>(kBgX), 1e-3);
    SCENE_CHECK_NEAR(model(1, 3), static_cast<double>(kBgY), 1e-3);
}

void TestParallaxAppliedModelAddsOffset() {
    // PATH_B / LASTPASS_DEST_STACK: ox/oy stay on dest-STACK. FetchDest /
    // model.col(3) keep +0x128 origin.
    SceneTestBegin("Parallax.AppliedModelStaysFetchDest");
    ParallaxFixture fx;
    auto            node = fx.MakeNode(kBgX, kBgY);
    WPShaderValueData data;
    data.SetParallaxContract({ kBgDepth, kBgDepth }, nullptr, false, true);
    fx.Bind(node.get(), data);
    fx.mouse = { 0.0f, 1.0f };

    const auto offset = fx.Offset(node.get());
    const auto model  = fx.ParallaxedModel(node.get());
    SCENE_CHECK(data.AppliesModelParallax());
    SCENE_CHECK_NEAR(model(0, 3), static_cast<double>(kBgX), 1e-3);
    SCENE_CHECK_NEAR(model(1, 3), static_cast<double>(kBgY), 1e-3);
    auto* object = fx.scene.FindSceneObject(node->ID());
    SCENE_CHECK(object != nullptr);
    const auto lookat = OfficialLookatFromViewCamera(
        { kCamX, kCamY }, { fx.mouse[0], fx.mouse[1] }, { kOrthoW, kOrthoH },
        fx.parallax.mouseinfluence);
    fx.scene.DestStackPushCopy();
    fx.scene.DestStackApplyPathB(*object, lookat.x(), lookat.y(), kAmount);
    SCENE_CHECK_NEAR(fx.scene.DestStackTop()(0, 3), offset.x(), 1e-3f);
    SCENE_CHECK_NEAR(fx.scene.DestStackTop()(1, 3), offset.y(), 1e-3f);
    fx.scene.DestStackPop();
}

void TestParallaxScaleDoesNotMultiplyWorldOffset() {
    // PATH_B dest-STACK 3x3 is identity (DEST_IDENTITY_CTOR). FetchDest scale
    // must not multiply dest T.
    SceneTestBegin("Parallax.ScaleDoesNotMultiplyWorldOffset");
    ParallaxFixture fx;
    auto            node = fx.MakeNode(kBgX, kBgY);
    node->SetScale({ 2.0f, 2.0f, 1.0f });
    WPShaderValueData data;
    data.SetParallaxContract({ kBgDepth, kBgDepth }, nullptr, false, true);
    fx.Bind(node.get(), data);
    fx.mouse = { 0.0f, 1.0f };

    const auto offset = fx.Offset(node.get());
    const auto model  = fx.ParallaxedModel(node.get());
    SCENE_CHECK(std::fabs(offset.x()) > 1.0f || std::fabs(offset.y()) > 1.0f);
    SCENE_CHECK_NEAR(model(0, 3), static_cast<double>(kBgX), 1e-3);
    SCENE_CHECK_NEAR(model(1, 3), static_cast<double>(kBgY), 1e-3);
    auto* object = fx.scene.FindSceneObject(node->ID());
    SCENE_CHECK(object != nullptr);
    const auto lookat = OfficialLookatFromViewCamera(
        { kCamX, kCamY }, { fx.mouse[0], fx.mouse[1] }, { kOrthoW, kOrthoH },
        fx.parallax.mouseinfluence);
    fx.scene.DestStackPushCopy();
    fx.scene.DestStackApplyPathB(*object, lookat.x(), lookat.y(), kAmount);
    SCENE_CHECK_NEAR(fx.scene.DestStackTop()(0, 3), offset.x(), 1e-3f);
    SCENE_CHECK_NEAR(fx.scene.DestStackTop()(1, 3), offset.y(), 1e-3f);
    SCENE_CHECK(std::fabs(fx.scene.DestStackTop()(0, 3) - 2.0f * offset.x()) > 1.0f ||
                std::fabs(fx.scene.DestStackTop()(1, 3) - 2.0f * offset.y()) > 1.0f);
    fx.scene.DestStackPop();
}

void TestParallaxOffsetCacheDoesNotStaleAcrossMouse() {
    SceneTestBegin("Parallax.CacheIsPerResolver");
    ParallaxFixture fx;
    auto            node = fx.MakeNode(kBgX, kBgY);
    WPShaderValueData data;
    data.SetParallaxContract({ kBgDepth, kBgDepth }, nullptr, false, true);
    fx.Bind(node.get(), data);

    fx.mouse        = { 0.5f, 0.5f };
    const auto rest = fx.Offset(node.get());
    fx.mouse        = { 0.0f, 1.0f };
    const auto moved_cached = fx.Offset(node.get());
    SCENE_CHECK_NEAR(moved_cached.x(), rest.x(), 1e-5f);
    SCENE_CHECK_NEAR(moved_cached.y(), rest.y(), 1e-5f);

    fx.ClearCaches();
    const auto moved = fx.Offset(node.get());
    SCENE_CHECK(std::fabs(moved.x() - rest.x()) > 1.0f || std::fabs(moved.y() - rest.y()) > 1.0f);
}

void TestParallaxDelayLerpsMouseBeforePathB() {
    // PATH_B / FETCH_DEST_LOCAL: delay lerps lookat, then dest-STACK Path B
    // uses that lookat. FetchDest / model.col(3) stay +0x128.
    SceneTestBegin("Parallax.DelayLerpsMouseBeforePathB");
    ParallaxFixture fx;
    auto            node = fx.MakeNode(kBgX, kBgY);
    WPShaderValueData data;
    data.SetParallaxContract({ kBgDepth, kBgDepth }, nullptr, false, true);

    WPShaderValueUpdater updater(&fx.scene);
    updater.SetCameraParallax(fx.parallax);
    fx.Bind(node.get(), data);
    updater.SetNodeData(node.get(), data);
    updater.MouseInput(0.0, 1.0);

    fx.scene.PassFrameTime(0.1);
    updater.PrepareFrame();
    updater.FrameBegin();
    updater.ComposeDrawWalker();
    const Eigen::Matrix4d delayed =
        updater.ResolveModelTransformForProjection(node.get(), fx.camera.get(), true);

    constexpr double kDelayRange    = 3.0;
    constexpr double kResponseRate  = 10.0;
    const double     t = std::min(1.0, kResponseRate * (1.0 - kDelay / kDelayRange) * 0.1);
    const float      mx = static_cast<float>(0.5 + t * (0.0 - 0.5));
    const float      my = static_cast<float>(0.5 + t * (1.0 - 0.5));
    SCENE_CHECK_NEAR(t, 1.0 / 3.0, 1e-6);

    const auto expected_mouse = ExpectedPathBOffset({ kBgX, kBgY },
                                                    { kCamX, kCamY },
                                                    { mx, my },
                                                    { kOrthoW, kOrthoH },
                                                    { kBgDepth, kBgDepth },
                                                    kAmount,
                                                    kInfluence);
    SCENE_CHECK_NEAR(delayed(0, 3), static_cast<double>(kBgX), 1e-3);
    SCENE_CHECK_NEAR(delayed(1, 3), static_cast<double>(kBgY), 1e-3);
    auto* object = fx.scene.FindSceneObject(node->ID());
    SCENE_CHECK(object != nullptr);
    SCENE_CHECK_NEAR(object->leftover_parallax().x(), expected_mouse.x(), 1e-3f);
    SCENE_CHECK_NEAR(object->leftover_parallax().y(), expected_mouse.y(), 1e-3f);

    const auto instant = ExpectedPathBOffset({ kBgX, kBgY },
                                             { kCamX, kCamY },
                                             { 0.0f, 1.0f },
                                             { kOrthoW, kOrthoH },
                                             { kBgDepth, kBgDepth },
                                             kAmount,
                                             kInfluence);
    SCENE_CHECK(std::fabs(object->leftover_parallax().x() - instant.x()) > 0.5f ||
                std::fabs(object->leftover_parallax().y() - instant.y()) > 0.5f);
}

void TestParallaxDelayOffSnapsMouse() {
    // PATH_B delay==0 snaps lookat; ox/oy still dest-STACK, not FetchDest.
    SceneTestBegin("Parallax.DelayOffSnapsMouse");
    ParallaxFixture fx;
    fx.parallax.delay = 0.0f;
    auto node         = fx.MakeNode(kBgX, kBgY);
    WPShaderValueData data;
    data.SetParallaxContract({ kBgDepth, kBgDepth }, nullptr, false, true);

    WPShaderValueUpdater updater(&fx.scene);
    updater.SetCameraParallax(fx.parallax);
    fx.Bind(node.get(), data);
    updater.SetNodeData(node.get(), data);
    updater.MouseInput(0.0, 1.0);
    fx.scene.PassFrameTime(0.1);
    updater.PrepareFrame();
    updater.FrameBegin();
    updater.ComposeDrawWalker();

    const Eigen::Matrix4d model =
        updater.ResolveModelTransformForProjection(node.get(), fx.camera.get(), true);
    const auto expected = ExpectedPathBOffset({ kBgX, kBgY },
                                              { kCamX, kCamY },
                                              { 0.0f, 1.0f },
                                              { kOrthoW, kOrthoH },
                                              { kBgDepth, kBgDepth },
                                              kAmount,
                                              kInfluence);
    SCENE_CHECK_NEAR(model(0, 3), static_cast<double>(kBgX), 1e-3);
    SCENE_CHECK_NEAR(model(1, 3), static_cast<double>(kBgY), 1e-3);
    auto* object = fx.scene.FindSceneObject(node->ID());
    SCENE_CHECK(object != nullptr);
    SCENE_CHECK_NEAR(object->leftover_parallax().x(), expected.x(), 1e-3f);
    SCENE_CHECK_NEAR(object->leftover_parallax().y(), expected.y(), 1e-3f);
}

void TestParallaxPositionUniform() {
    SceneTestBegin("Parallax.GParallaxPositionUniform");
    ParallaxFixture fx;
    auto            node = fx.MakeNode(kBgX, kBgY);
    WPShaderValueData data;
    data.SetParallaxContract({ kBgDepth, kBgDepth }, nullptr, false, true);

    WPShaderValueUpdater updater(&fx.scene);
    updater.SetCameraParallax(fx.parallax);
    updater.SetNodeData(node.get(), data);
    updater.InitUniforms(node.get(), [](std::string_view name) {
        return name == G_PARALLAXPOSITION;
    });

    sprite_map_t sprites;
    ShaderValue  written;
    bool         saw = false;
    updater.MouseInput(0.0, 1.0);
    fx.parallax.delay = 0.0f;
    updater.SetCameraParallax(fx.parallax);
    fx.scene.PassFrameTime(0.0);
    updater.PrepareFrame();
    updater.UpdateUniforms(node.get(), sprites, [&](std::string_view name, ShaderValue value) {
        if (name == G_PARALLAXPOSITION) {
            written = std::move(value);
            saw     = true;
        }
    });
    SCENE_CHECK(saw);
    SCENE_CHECK(written.size() >= 2u);
    const auto expected = ExpectedParallaxPositionNdc({ 0.0f, 1.0f }, kInfluence, true);
    SceneReportVec2("g_ParallaxPosition enable",
                    written[0],
                    written[1],
                    expected.x(),
                    expected.y());
    SCENE_CHECK_NEAR(written[0], expected.x(), 1e-5f);
    SCENE_CHECK_NEAR(written[1], expected.y(), 1e-5f);

    fx.parallax.enable = false;
    updater.SetCameraParallax(fx.parallax);
    saw = false;
    updater.UpdateUniforms(node.get(), sprites, [&](std::string_view name, ShaderValue value) {
        if (name == G_PARALLAXPOSITION) {
            written = std::move(value);
            saw     = true;
        }
    });
    SCENE_CHECK(saw);
    SCENE_CHECK_NEAR(written[0], 0.5f, 1e-5f);
    SCENE_CHECK_NEAR(written[1], 0.5f, 1e-5f);
}

void TestDestStackPathBIdentityT() {
    // PATH_B 0x14018b118 on DEST_IDENTITY_CTOR dest: T=(ox,oy). leftover +0x178.
    SceneTestBegin("DestStack.PathBIdentityT");
    ParallaxFixture fx;
    auto            node = fx.MakeNode(kBgX, kBgY);
    WPShaderValueData data;
    data.SetParallaxContract({ kBgDepth, kBgDepth }, nullptr, false, true);
    fx.Bind(node.get(), data);
    auto* object = fx.scene.FindSceneObject(node->ID());
    SCENE_CHECK(object != nullptr);
    fx.scene.DestStackPushCopy();
    fx.scene.DestStackApplyPathB(*object, kCamX, kCamY, kAmount);
    const auto& dest   = fx.scene.DestStackTop();
    const auto  expect = ExpectedPathBFromLookat({ kBgX, kBgY },
                                                { kCamX, kCamY },
                                                { kBgDepth, kBgDepth },
                                                kAmount);
    SCENE_CHECK_NEAR(dest(0, 3), expect.x(), 1e-4f);
    SCENE_CHECK_NEAR(dest(1, 3), expect.y(), 1e-4f);
    SCENE_CHECK_NEAR(dest(0, 0), 1.0f, 1e-6f);
    SCENE_CHECK_NEAR(dest(1, 1), 1.0f, 1e-6f);
    SCENE_CHECK_NEAR(object->leftover_parallax().x(), expect.x(), 1e-4f);
    SCENE_CHECK_NEAR(object->leftover_parallax().y(), expect.y(), 1e-4f);
    fx.scene.DestStackPop();
    SCENE_CHECK(fx.scene.DestStackAtBase());
    SCENE_CHECK_NEAR(fx.scene.DestStackTop()(0, 3), 0.0f, 1e-6f);
}

void TestDestStackChildUsesRootOrigin() {
    // PATH_B 0x14018b047 walks +0x180 to ROOT; ox/oy from ROOT +0x128/+0x170.
    SceneTestBegin("DestStack.ChildUsesRootOrigin");
    ParallaxFixture fx;
    auto            root  = fx.MakeNode(kBodyX, kBodyY);
    auto            child = fx.MakeNode(kFwX, kFwY);
    WPShaderValueData root_data;
    root_data.SetParallaxContract({ kBodyDepthX, kBodyDepthY }, nullptr, false, true);
    WPShaderValueData child_data;
    child_data.SetParallaxContract({ 1.0f, 1.0f }, root.get(), false, false);
    fx.Bind(root.get(), root_data);
    fx.Bind(child.get(), child_data);
    auto* child_object = fx.scene.FindSceneObject(child->ID());
    SCENE_CHECK(child_object != nullptr);
    fx.scene.DestStackPushCopy();
    fx.scene.DestStackApplyPathB(*child_object, kCamX, kCamY, kAmount);
    const auto& dest   = fx.scene.DestStackTop();
    const auto  expect = ExpectedPathBFromLookat({ kBodyX, kBodyY },
                                                { kCamX, kCamY },
                                                { kBodyDepthX, kBodyDepthY },
                                                kAmount);
    SCENE_CHECK_NEAR(dest(0, 3), expect.x(), 1e-4f);
    SCENE_CHECK_NEAR(dest(1, 3), expect.y(), 1e-4f);
    SCENE_CHECK_NEAR(child_object->leftover_parallax().x(), expect.x(), 1e-4f);
    SCENE_CHECK_NEAR(child_object->leftover_parallax().y(), expect.y(), 1e-4f);
    fx.scene.DestStackPop();
}

void TestLeftoverDestOrthoMvpIsNamedRtPixelOrtho() {
    // DEST_ORTHO_TNF / LIVE_LASTPASS_930 leftover PRE +0x930: 2/W, 2/H, Tx=Ty=-1.
    // dest=I. Not dest-STACK.
    SceneTestBegin("DestStack.LeftoverDestOrthoMvp");
    ParallaxFixture fx;
    auto            node = fx.MakeNode(kBgX, kBgY);
    WPShaderValueData data;
    data.SetParallaxContract({ kBgDepth, kBgDepth }, nullptr, false, true);
    fx.Bind(node.get(), data);
    auto* object = fx.scene.FindSceneObject(node->ID());
    SCENE_CHECK(object != nullptr);
    object->set_dest_size(1400.0f, 307.0f);
    fx.scene.DestStackPushCopy();
    fx.scene.DestStackApplyPathB(*object, kCamX, kCamY, kAmount);
    const Eigen::Matrix4f leftover = fx.scene.LeftoverDestOrthoMvp(*object);
    const Eigen::Matrix4f last     = fx.scene.FitOrthoCamera() * fx.scene.DestStackTop();
    SCENE_CHECK_NEAR(leftover(0, 0), 2.0f / 1400.0f, 1e-6f);
    SCENE_CHECK_NEAR(leftover(1, 1), 2.0f / 307.0f, 1e-6f);
    SCENE_CHECK_NEAR(leftover(0, 3), -1.0f, 1e-6f);
    SCENE_CHECK_NEAR(leftover(1, 3), -1.0f, 1e-6f);
    SCENE_CHECK(std::fabs(leftover(0, 3) - last(0, 3)) > 1e-3f ||
                std::fabs(leftover(0, 0) - last(0, 0)) > 1e-6f);
    fx.scene.DestStackPop();
}

void TestDestStackLastPassMvpIsCameraTimesDest() {
    // ENGINE_FLUSH 0x1400d4264 / LASTPASS_CAM_ORTHO / LASTPASS_DEST_STACK:
    // +0x930 = fit-ortho * Path B dest-STACK. Not FetchDest, not TREE Ortho().
    SceneTestBegin("DestStack.LastPassMvpIsCameraTimesDest");
    ParallaxFixture fx;
    auto            node = fx.MakeNode(kBgX, kBgY);
    WPShaderValueData data;
    data.SetParallaxContract({ kBgDepth, kBgDepth }, nullptr, false, true);
    fx.Bind(node.get(), data);
    auto* object = fx.scene.FindSceneObject(node->ID());
    SCENE_CHECK(object != nullptr);
    fx.scene.SetWindowSize(1600, 1000);
    fx.scene.DestStackPushCopy();
    fx.scene.DestStackApplyPathB(*object, kCamX, kCamY, kAmount);
    fx.scene.FlushLastPassMvp();
    const Eigen::Matrix4f cam  = fx.scene.FitOrthoCamera();
    const Eigen::Matrix4f dest = fx.scene.DestStackTop();
    const Eigen::Matrix4f got  = fx.scene.LastPassMvp();
    const Eigen::Matrix4f want = cam * dest;
    const float           vw   = kOrthoH * 1600.0f / 1000.0f;
    SCENE_CHECK_NEAR(cam(0, 0), 2.0f / vw, 1e-5f);
    SCENE_CHECK_NEAR(cam(1, 1), 2.0f / kOrthoH, 1e-5f);
    SCENE_CHECK_NEAR(cam(0, 3), -kOrthoW / vw, 1e-5f);
    SCENE_CHECK_NEAR(cam(1, 3), -1.0f, 1e-5f);
    SCENE_CHECK_NEAR(got(0, 3), want(0, 3), 1e-4f);
    SCENE_CHECK_NEAR(got(1, 3), want(1, 3), 1e-4f);
    SCENE_CHECK_NEAR(got(0, 0), want(0, 0), 1e-5f);
    const Eigen::Matrix4f fetch = object->FetchDest();
    const Eigen::Matrix4f draw  = fx.scene.LastPassDrawMvp(*object);
    const Eigen::Matrix4f want_draw = got * fetch;
    SCENE_CHECK_NEAR(draw(0, 3), want_draw(0, 3), 1e-3f);
    SCENE_CHECK_NEAR(draw(1, 3), want_draw(1, 3), 1e-3f);
    fx.scene.DestStackPop();
}

void TestDestStackFrameBeginWalkerStoresLeftover() {
    SceneTestBegin("DestStack.FrameBeginWalkerStoresLeftover");
    ParallaxFixture fx;
    auto            node = fx.MakeNode(kBgX, kBgY);
    WPShaderValueData data;
    data.SetParallaxContract({ kBgDepth, kBgDepth }, nullptr, false, true);
    WPShaderValueUpdater updater(&fx.scene);
    updater.SetCameraParallax(fx.parallax);
    fx.Bind(node.get(), data);
    updater.SetNodeData(node.get(), data);
    updater.SetScreenSize(1600, 1000);
    updater.MouseInput(0.5, 0.5);
    fx.scene.PassFrameTime(0.0);
    updater.PrepareFrame();
    updater.FrameBegin();
    updater.ComposeDrawWalker();
    auto* object = fx.scene.FindSceneObject(node->ID());
    SCENE_CHECK(object != nullptr);
    SCENE_CHECK(fx.scene.DestStackAtBase());
    SCENE_CHECK_NEAR(fx.scene.LastPassMvp()(0, 0), fx.scene.FitOrthoCamera()(0, 0), 1e-5f);
    const auto lookat = OfficialLookatFromViewCamera({ kCamX, kCamY },
                                                     { 0.5f, 0.5f },
                                                     { kOrthoW, kOrthoH },
                                                     kInfluence);
    const auto expect = ExpectedPathBFromLookat({ kBgX, kBgY },
                                                lookat,
                                                { kBgDepth, kBgDepth },
                                                kAmount);
    SCENE_CHECK_NEAR(object->leftover_parallax().x(), expect.x(), 1e-3f);
    SCENE_CHECK_NEAR(object->leftover_parallax().y(), expect.y(), 1e-3f);
    const Eigen::Matrix4f cam = fx.scene.FitOrthoCamera();
    SCENE_CHECK_NEAR(fx.scene.LastPassMvp()(0, 3),
                     cam(0, 0) * expect.x() + cam(0, 3), 1e-3f);
    SCENE_CHECK_NEAR(fx.scene.LastPassMvp()(1, 3),
                     cam(1, 1) * expect.y() + cam(1, 3), 1e-3f);
}

void ReadCardAttr(const SceneMesh& mesh, std::string_view name, Eigen::Vector2f out[4]) {
    SCENE_CHECK(mesh.VertexCount() > 0);
    const auto& vertex = mesh.GetVertexArray(0);
    SCENE_CHECK(vertex.VertexCount() == 4u);
    const auto offsets = vertex.GetAttrOffsetMap();
    const auto attr    = offsets.find(std::string(name));
    SCENE_CHECK(attr != offsets.end());
    const float* data   = vertex.Data();
    const usize  stride = vertex.OneSize();
    for (int i = 0; i < 4; ++i) {
        const float* p = data + i * static_cast<int>(stride) +
                         static_cast<int>(attr->second.offset / sizeof(float));
        out[i] = { p[0], p[1] };
    }
}

void ReadCardXY(const SceneMesh& mesh, Eigen::Vector2f out[4]) {
    ReadCardAttr(mesh, WE_IN_POSITION, out);
}

void TestDestDrawMeshesCenterFlags() {
    // IMAGE_2D8_NOFULLFB leftover flags=0 → 0..max(4,AABB).
    // IMAGE_VT_B0 / POSTFX_MESH last-pass bit0 → ±half (int)+0x2f0.
    // DEST_DRAW_VERTS order: (L,T), (R,T), (L,B), (R,B).
    SceneTestBegin("DestDrawMeshes.CenterFlags");
    Scene       scene;
    SceneObject object(&scene, 417);
    object.set_dest_size(787.0f, 743.0f);
    object.PublishDestDrawMeshes();
    SCENE_CHECK(object.leftover_mesh() != nullptr);
    SCENE_CHECK(object.postfx_mesh() != nullptr);
    SCENE_CHECK(object.lastpass_mesh() != nullptr);
    Eigen::Vector2f leftover[4];
    Eigen::Vector2f postfx[4];
    Eigen::Vector2f lastpass[4];
    ReadCardXY(*object.leftover_mesh(), leftover);
    ReadCardXY(*object.postfx_mesh(), postfx);
    ReadCardXY(*object.lastpass_mesh(), lastpass);
    SCENE_CHECK_NEAR(leftover[0].x(), 0.0f, 1e-5f);
    SCENE_CHECK_NEAR(leftover[0].y(), 743.0f, 1e-5f);
    SCENE_CHECK_NEAR(leftover[1].x(), 787.0f, 1e-5f);
    SCENE_CHECK_NEAR(leftover[1].y(), 743.0f, 1e-5f);
    SCENE_CHECK_NEAR(leftover[2].x(), 0.0f, 1e-5f);
    SCENE_CHECK_NEAR(leftover[2].y(), 0.0f, 1e-5f);
    SCENE_CHECK_NEAR(leftover[3].x(), 787.0f, 1e-5f);
    SCENE_CHECK_NEAR(leftover[3].y(), 0.0f, 1e-5f);
    // POSTFX_MESH / CARD_CENTER: +0x2e0 {2,2} bit0 → ±1.
    SCENE_CHECK_NEAR(postfx[0].x(), -1.0f, 1e-5f);
    SCENE_CHECK_NEAR(postfx[0].y(), 1.0f, 1e-5f);
    SCENE_CHECK_NEAR(postfx[1].x(), 1.0f, 1e-5f);
    SCENE_CHECK_NEAR(postfx[1].y(), 1.0f, 1e-5f);
    SCENE_CHECK_NEAR(postfx[2].x(), -1.0f, 1e-5f);
    SCENE_CHECK_NEAR(postfx[2].y(), -1.0f, 1e-5f);
    SCENE_CHECK_NEAR(postfx[3].x(), 1.0f, 1e-5f);
    SCENE_CHECK_NEAR(postfx[3].y(), -1.0f, 1e-5f);
    SCENE_CHECK_NEAR(lastpass[0].x(), -393.5f, 1e-5f);
    SCENE_CHECK_NEAR(lastpass[0].y(), 371.5f, 1e-5f);
    SCENE_CHECK_NEAR(lastpass[1].x(), 393.5f, 1e-5f);
    SCENE_CHECK_NEAR(lastpass[1].y(), 371.5f, 1e-5f);
    SCENE_CHECK_NEAR(lastpass[2].x(), -393.5f, 1e-5f);
    SCENE_CHECK_NEAR(lastpass[2].y(), -371.5f, 1e-5f);
    SCENE_CHECK_NEAR(lastpass[3].x(), 393.5f, 1e-5f);
    SCENE_CHECK_NEAR(lastpass[3].y(), -371.5f, 1e-5f);

    // IMAGE_VT_128 0x140209290: leftover UV max is content/physical.
    // Last-pass +0x2e8 UV stays 1,1 when +0x320>0 (TEXT_2F0). Leftover-only
    // +0x490 Draw uses that leftover mapRate (IMAGE_490_MESH).
    object.set_leftover_uv(4200.0f / 8192.0f, 2500.0f / 4096.0f);
    Eigen::Vector2f leftover_uv[4];
    Eigen::Vector2f lastpass_uv[4];
    ReadCardAttr(*object.leftover_mesh(), WE_IN_TEXCOORD, leftover_uv);
    ReadCardAttr(*object.lastpass_mesh(), WE_IN_TEXCOORD, lastpass_uv);
    SCENE_CHECK_NEAR(leftover_uv[1].x(), 4200.0f / 8192.0f, 1e-5f);
    SCENE_CHECK_NEAR(leftover_uv[2].y(), 2500.0f / 4096.0f, 1e-5f);
    SCENE_CHECK_NEAR(lastpass_uv[1].x(), 4200.0f / 8192.0f, 1e-5f);
    SCENE_CHECK_NEAR(lastpass_uv[2].y(), 2500.0f / 4096.0f, 1e-5f);
    object.set_effect_count(1);
    object.PublishDestDrawMeshes();
    ReadCardAttr(*object.lastpass_mesh(), WE_IN_TEXCOORD, lastpass_uv);
    SCENE_CHECK_NEAR(lastpass_uv[1].x(), 1.0f, 1e-5f);
    SCENE_CHECK_NEAR(lastpass_uv[2].y(), 1.0f, 1e-5f);

    SceneObject tiny(&scene, 1);
    tiny.set_dest_size(2.0f, 2.0f);
    tiny.PublishDestDrawMeshes();
    Eigen::Vector2f tiny_leftover[4];
    ReadCardXY(*tiny.leftover_mesh(), tiny_leftover);
    SCENE_CHECK_NEAR(tiny_leftover[1].x(), 4.0f, 1e-5f);
    SCENE_CHECK_NEAR(tiny_leftover[0].y(), 4.0f, 1e-5f);
}

void TestText2F0AddsPadWhenEffects() {
    // TEXT_2F0 0x140258986: +0x320>0 → +0x2f0 = AABB + 2*min(pad,512).
    SceneTestBegin("DestDrawMeshes.Text2F0AddsPadWhenEffects");
    Scene       scene;
    SceneObject dated(&scene, 248);
    dated.set_effect_count(1);
    dated.ApplyTextDestSize(100.0f, 40.0f, 10.0f, 10.0f);
    SCENE_CHECK_NEAR(dated.dest_size().x(), 120.0f, 1e-5f);
    SCENE_CHECK_NEAR(dated.dest_size().y(), 60.0f, 1e-5f);
    Eigen::Vector2f lastpass[4];
    ReadCardXY(*dated.lastpass_mesh(), lastpass);
    SCENE_CHECK_NEAR(lastpass[0].x(), -60.0f, 1e-5f);
    SCENE_CHECK_NEAR(lastpass[0].y(), 30.0f, 1e-5f);
    // TEXT_2F0 0x1402589da / POSTFX_MESH: +0x2e8 republish keeps the
    // Data last-pass Draw already shares. A new unique_ptr left VERTICAL
    // on the parse-time card (Date ±423 vs runtime ±723).
    SceneMesh vertical_share;
    vertical_share.ChangeMeshDataFrom(*dated.lastpass_mesh());
    dated.ApplyTextDestSize(1382.0f, 307.0f, 32.0f, 32.0f);
    ReadCardXY(*dated.lastpass_mesh(), lastpass);
    SCENE_CHECK_NEAR(lastpass[0].x(), -723.0f, 1e-5f);
    Eigen::Vector2f shared[4];
    ReadCardXY(vertical_share, shared);
    SCENE_CHECK_NEAR(shared[0].x(), lastpass[0].x(), 1e-5f);
    SCENE_CHECK_NEAR(shared[1].x(), lastpass[1].x(), 1e-5f);

    SceneObject clock(&scene, 315);
    clock.set_effect_count(0);
    clock.ApplyTextDestSize(100.0f, 40.0f, 10.0f, 10.0f);
    SCENE_CHECK_NEAR(clock.dest_size().x(), 100.0f, 1e-5f);
    SCENE_CHECK_NEAR(clock.dest_size().y(), 40.0f, 1e-5f);

    SceneObject clamped(&scene, 1);
    clamped.set_effect_count(1);
    clamped.ApplyTextDestSize(10.0f, 10.0f, 600.0f, 600.0f);
    SCENE_CHECK_NEAR(clamped.dest_size().x(), 10.0f + 1024.0f, 1e-5f);
    SCENE_CHECK_NEAR(clamped.dest_size().y(), 10.0f + 1024.0f, 1e-5f);
}

void TestParallaxDoubleHalfOrthoIsWrong() {
    SceneTestBegin("Parallax.CameraAlreadyAtHalfOrtho");
    ParallaxFixture fx;
    auto            node = fx.MakeNode(kBgX, kBgY);
    WPShaderValueData data;
    data.SetParallaxContract({ kBgDepth, kBgDepth }, nullptr, false, true);
    fx.Bind(node.get(), data);
    fx.mouse = { 0.5f, 0.5f };

    const auto got = fx.Offset(node.get());
    const auto doubled = ExpectedPathBOffset({ kBgX, kBgY },
                                             { kCamX + kOrthoW * 0.5f, kCamY + kOrthoH * 0.5f },
                                             { 0.5f, 0.5f },
                                             { kOrthoW, kOrthoH },
                                             { kBgDepth, kBgDepth },
                                             kAmount,
                                             kInfluence);
    SCENE_CHECK(std::fabs(got.x() - doubled.x()) > 1.0f || std::fabs(got.y() - doubled.y()) > 1.0f);
}

void RunParallaxTests() {
    TestParallaxDisabledIsZero();
    TestParallaxAmountZeroIsZero();
    TestParallaxRestMouseMatchesCenteredCamera();
    TestParallaxMouseInfluenceAndYFlip();
    TestParallaxAsymmetricDepth();
    TestParallaxZeroDepthStaysPut();
    TestParallaxOmittedChildEqualsRootNotOwnOrigin();
    TestParallaxAuthoredChildKeepsOwnOrigin();
    TestParallaxGrandchildWalksToRoot();
    TestParallaxBoneAttachedAnchorDoesNotRecurse();
    TestParallaxSuppressKeepsModelTranslation();
    TestParallaxAppliedModelAddsOffset();
    TestParallaxScaleDoesNotMultiplyWorldOffset();
    TestParallaxOffsetCacheDoesNotStaleAcrossMouse();
    TestParallaxDelayLerpsMouseBeforePathB();
    TestParallaxDelayOffSnapsMouse();
    TestParallaxPositionUniform();
    TestParallaxDoubleHalfOrthoIsWrong();
    TestDestStackPathBIdentityT();
    TestDestStackChildUsesRootOrigin();
    TestLeftoverDestOrthoMvpIsNamedRtPixelOrtho();
    TestDestStackLastPassMvpIsCameraTimesDest();
    TestDestStackFrameBeginWalkerStoresLeftover();
    TestDestDrawMeshesCenterFlags();
    TestText2F0AddsPadWhenEffects();
}
