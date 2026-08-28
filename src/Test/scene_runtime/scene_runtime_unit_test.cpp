#include "test_harness.h"

#include "Particle/ParticleEmitter.h"
#include "Scene/Scene.h"
#include "Scene/SceneMesh.h"
#include "Scene/SceneNode.h"
#include "Scene/SceneObject.h"
#include "SpriteAnimation.hpp"
#include "WPImageAlignment.hpp"
#include "WPNodeTransformResolver.hpp"
#include "WPShaderValueUpdater.hpp"
#include "WPTextLayer.hpp"

#include <memory>
#include <vector>

using namespace wallpaper;

void TestSpriteAnimationAdvancesOneFramePerWrap() {
    SceneTestBegin("SpriteAnimation.AdvancesOneFramePerWrap");

    SpriteAnimation anim;
    for (int i = 0; i < 151; ++i) {
        SpriteFrame frame;
        frame.frametime = 0.04f;
        frame.x         = static_cast<float>(i);
        anim.AppendFrame(frame);
    }
    SCENE_CHECK(anim.numFrames() == 151u);
    SCENE_CHECK(anim.CurrentFrameIndex() == 0);

    (void)anim.GetAnimateFrame(0.0);
    SCENE_CHECK(anim.CurrentFrameIndex() == 0);

    (void)anim.GetAnimateFrame(0.04);
    SCENE_CHECK(anim.CurrentFrameIndex() == 1);
    (void)anim.GetAnimateFrame(0.04);
    SCENE_CHECK(anim.CurrentFrameIndex() == 2);

    anim.SetCurrentFrame(150);
    (void)anim.GetAnimateFrame(0.04);
    SCENE_CHECK(anim.CurrentFrameIndex() == 0);
}

void TestParticleEmitDelayArmsNextFrame() {
    SceneTestBegin("ParticleEmit.DelayBlocksCurrentFrame");

    ParticleBoxEmitterArgs args {};
    args.directions  = { 1.0f, 1.0f, 1.0f };
    args.minDistance = { 0.0f, 0.0f, 0.0f };
    args.maxDistance = { 0.0f, 0.0f, 0.0f };
    args.orgin       = { 0.0f, 0.0f, 0.0f };
    args.minSpeed    = 0.0f;
    args.maxSpeed    = 0.0f;
    args.timing.emit_speed    = 10.0f;
    args.timing.instantaneous = 3;
    args.timing.delay         = 1.0f;
    args.timing.duration      = 0.0f;

    auto emit = ParticleBoxEmitterArgs::MakeEmittOp(args);
    std::vector<Particle>        particles;
    std::vector<ParticleInitOp>  inits;
    std::vector<ParticleControlpoint> cps;
    uint64_t                     seq = 0;
    auto                         runtime = MakeParticleEmitRuntime(args.timing);

    emit(particles, inits, cps, 16, 0.5, 0.0, seq, runtime);
    SCENE_CHECK(particles.empty());
    SCENE_CHECK_NEAR(runtime.delay_remaining, 0.5f, 1e-5f);

    emit(particles, inits, cps, 16, 0.6, 0.5, seq, runtime);
    SCENE_CHECK(particles.empty());

    emit(particles, inits, cps, 16, 0.1, 1.1, seq, runtime);
    SCENE_CHECK(particles.size() == 3u + 1u);
}

void TestParticleEmitRateCredit() {
    SceneTestBegin("ParticleEmit.RateCreditEmitsWholeParticles");

    ParticleBoxEmitterArgs args {};
    args.directions  = { 1.0f, 1.0f, 1.0f };
    args.minDistance = { 0.0f, 0.0f, 0.0f };
    args.maxDistance = { 0.0f, 0.0f, 0.0f };
    args.orgin       = { 10.0f, 20.0f, 0.0f };
    args.minSpeed    = 0.0f;
    args.maxSpeed    = 0.0f;
    args.timing.emit_speed = 10.0f;

    auto emit = ParticleBoxEmitterArgs::MakeEmittOp(args);
    std::vector<Particle>        particles;
    std::vector<ParticleInitOp>  inits;
    std::vector<ParticleControlpoint> cps;
    uint64_t                     seq = 0;
    auto                         runtime = MakeParticleEmitRuntime(args.timing);

    emit(particles, inits, cps, 16, 0.05, 0.0, seq, runtime);
    SCENE_CHECK(particles.empty());
    SCENE_CHECK_NEAR(runtime.credit, 0.5f, 1e-5f);

    emit(particles, inits, cps, 16, 0.05, 0.05, seq, runtime);
    SCENE_CHECK(particles.size() == 1u);
    SCENE_CHECK_NEAR(particles[0].position.x(), 10.0f, 1e-4f);
    SCENE_CHECK_NEAR(particles[0].position.y(), 20.0f, 1e-4f);
}

void TestGeometryTransformComposesAfterNodeLocal() {
    SceneTestBegin("Transform.GeometryAfterNodeLocal");

    SceneNode node(Eigen::Vector3f { 100.0f, 200.0f, 0.0f },
                   Eigen::Vector3f { 2.0f, 3.0f, 1.0f },
                   Eigen::Vector3f::Zero());
    auto mesh = std::make_shared<SceneMesh>();
    mesh->SetGeometryTransform(
        Eigen::Affine3f(Eigen::Translation3f(Eigen::Vector3f { 10.0f, -20.0f, 0.0f })));
    node.AddMesh(mesh);
    node.UpdateTrans();

    const Eigen::Matrix4d composed =
        node.GetLocalTrans() * mesh->GeometryTransform().matrix().cast<double>();
    SCENE_CHECK_NEAR(composed(0, 3), 120.0, 1e-4);
    SCENE_CHECK_NEAR(composed(1, 3), 140.0, 1e-4);
}

void TestTextDestAppliesOfficialLocalOffsetAfterFetchDest() {
    SceneTestBegin("TextDest.Vt80AppliesLocalOffsetAfterFetchDest");

    Scene scene;
    auto node = std::make_shared<SceneNode>(Eigen::Vector3f { 2934.68628f, 1201.71655f, 0.0f },
                                            Eigen::Vector3f { 1.0f, 1.0f, 1.0f },
                                            Eigen::Vector3f::Zero());
    node->ID() = 315;
    TextLayerRuntimeState clock;
    clock.object.horizontalalign = "left";
    clock.object.verticalalign   = "center";
    clock.object.anchor          = "none";
    clock.object.size            = { 328.0f, 83.0f };
    ApplyTextLayerNodePlacement(node.get(), clock, { 2934.68628f, 1201.71655f, 0.0f });
    // Official +0x2f8 left is +0.5*(layout+0x98-layout+0x90) at 0x140257725.
    SCENE_CHECK_NEAR(node->AlignmentOffset().x(), 164.0f, 1e-3f);
    SCENE_CHECK_NEAR(node->AlignmentOffset().y(), 0.0f, 1e-3f);
    scene.nodeOwners[node.get()] = 315;

    auto& text = scene.EnsureSceneObject(315);
    text.set_kind(SceneObjectKind::Text);
    text.set_origin(node->Translate());
    text.set_source_node(node.get());

    auto& image = scene.EnsureSceneObject(316);
    image.set_kind(SceneObjectKind::Image);
    image.set_origin(node->Translate());

    Map<void*, WPShaderValueData> node_data;
    Map<void*, Eigen::Matrix4d>   model_cache;
    Map<void*, Eigen::Vector3f>   parallax_cache;
    Map<void*, Eigen::Affine3f>   attach_cache;
    WPCameraParallax              parallax {};
    WPNodeTransformResolver resolver(scene,
                                     parallax,
                                     node_data,
                                     model_cache,
                                     parallax_cache,
                                     attach_cache,
                                     nullptr,
                                     { 0.0f, 0.0f },
                                     1);

    const auto text_dest = text.FetchDest();
    SCENE_CHECK_NEAR(text_dest(0, 3), 2934.68628f, 1e-3f);
    SCENE_CHECK_NEAR(text_dest(1, 3), 1201.71655f, 1e-3f);

    const auto text_model = resolver.ResolveRawModelTransform(node.get());
    const auto expect = ApplyTextDestLocalOffset(text_dest.cast<double>(), node->AlignmentOffset());
    SCENE_CHECK_NEAR(text_model(0, 3), expect(0, 3), 1e-3);
    SCENE_CHECK_NEAR(text_model(1, 3), expect(1, 3), 1e-3);
    SCENE_CHECK_NEAR(text_model(0, 3), 3098.68628, 1e-3);

    scene.nodeOwners[node.get()] = 316;
    image.set_source_node(node.get());
    model_cache.clear();
    const auto image_model = resolver.ResolveRawModelTransform(node.get());
    SCENE_CHECK_NEAR(image_model(0, 3), 2934.68628, 1e-3);
    SCENE_CHECK_NEAR(image_model(1, 3), 1201.71655, 1e-3);
}

void RunParallaxTests();

void RunSceneRuntimeUnitTests() {
    TestSpriteAnimationAdvancesOneFramePerWrap();
    TestParticleEmitDelayArmsNextFrame();
    TestParticleEmitRateCredit();
    TestGeometryTransformComposesAfterNodeLocal();
    TestTextDestAppliesOfficialLocalOffsetAfterFetchDest();
    RunParallaxTests();
}
