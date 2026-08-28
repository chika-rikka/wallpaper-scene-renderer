#include "test_harness.h"

#include "Audio/SoundManager.h"
#include "Fs/PhysicalFs.h"
#include "Fs/VFS.h"
#include "Scene/Scene.h"
#include "Scene/SceneCamera.h"
#include "Scene/SceneImageEffectLayer.h"
#include "Scene/SceneMesh.h"
#include "Scene/SceneObject.h"
#include "Scene/SceneShader.h"
#include "SpecTexs.hpp"
#include "RenderGraph/RenderGraph.hpp"
#include "VulkanRender/SceneToRenderGraph.hpp"
#include "WPSceneParser.hpp"
#include "WPShaderValueUpdater.hpp"
#include "WPPuppet.hpp"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

using namespace wallpaper;

void RunWorkshop3462491575Tests();

namespace
{

#ifndef WESCENE_WORKSHOP_3462491575
#define WESCENE_WORKSHOP_3462491575 \
    "/media/rikka/Data/steam/steamapps/workshop/content/431960/3462491575/output"
#endif

#ifndef WESCENE_WE_ASSETS
#define WESCENE_WE_ASSETS "/media/rikka/Data/steam/steamapps/common/wallpaper_engine/assets"
#endif

std::string WorkshopDir() {
    if (const char* env = std::getenv("WESCENE_WORKSHOP_3462491575"); env && env[0] != '\0') {
        return env;
    }
    return WESCENE_WORKSHOP_3462491575;
}

std::string AssetsDir() {
    if (const char* env = std::getenv("WESCENE_WE_ASSETS"); env && env[0] != '\0') {
        return env;
    }
    return WESCENE_WE_ASSETS;
}

std::shared_ptr<Scene> ParseWorkshopScene() {
    const auto workshop = WorkshopDir();
    if (! std::filesystem::is_regular_file(std::filesystem::path(workshop) / "scene.json")) {
        return nullptr;
    }
    if (const char* skip = std::getenv("WESCENE_SKIP_FULL_PARSE"); skip && skip[0] == '1') {
        return nullptr;
    }
    auto vfs = std::make_unique<fs::VFS>();
    const auto assets = AssetsDir();
    if (std::filesystem::is_directory(assets)) {
        if (! vfs->Mount("/assets", fs::CreatePhysicalFs(assets), "assets")) return nullptr;
    }
    if (! vfs->Mount("/assets", fs::CreatePhysicalFs(workshop), "workshop")) return nullptr;
    const std::string scene_src = fs::GetFileContent(*vfs, "/assets/scene.json");
    if (scene_src.empty()) return nullptr;
    audio::SoundManager sound;
    WPSceneParser       parser;
    return parser.Parse("3462491575", scene_src, *vfs, sound, nullptr, 1.0);
}

SceneImageEffectLayer* FindEffectLayerForId(Scene& scene, int32_t id) {
    for (const auto& [name, cam] : scene.cameras) {
        if (cam == nullptr || ! cam->HasImgEffect()) continue;
        auto* layer = cam->GetImgEffect().get();
        if (layer == nullptr || layer->LayerNode() == nullptr) continue;
        const auto owner = scene.nodeOwners.find(layer->LayerNode());
        if (owner != scene.nodeOwners.end() && owner->second == id) return layer;
    }
    return nullptr;
}

bool LayerWritesOutput(const std::vector<SceneRenderGraphPassRecord>& inventory, int32_t id,
                       std::string_view output) {
    for (const auto& pass : inventory) {
        if (pass.layer_id == id && pass.output == output) return true;
    }
    return false;
}

bool NodeIsProxy(const Scene& scene, SceneNode* node) {
    return node != nullptr && scene.renderOrderProxyNodes.count(node) != 0;
}

void CheckOfficialDestMultiply(Scene& scene, int32_t child_id, int32_t parent_id) {
    auto* child  = scene.FindSceneObject(child_id);
    auto* parent = scene.FindSceneObject(parent_id);
    SCENE_CHECK(child != nullptr && parent != nullptr);
    if (child == nullptr || parent == nullptr) return;
    const auto dest        = child->FetchDest();
    const auto parent_dest = parent->FetchDest();
    const auto local       = child->ComposeLocalDest();
    const Eigen::Matrix4f expected = parent_dest * local;
    SCENE_CHECK_NEAR(dest(0, 3), expected(0, 3), 1e-3f);
    SCENE_CHECK_NEAR(dest(1, 3), expected(1, 3), 1e-3f);
}

} // namespace

void TestWorkshop3462491575OfficialFlatDrawList() {
    // Official factory 0x140190837 pushes every object onto +0x158, including
    // empty 0x2c0 objects (0x1401907e0). Walker 0x14018aef0 calls vt+0x50 then
    // add r14,8. Image draw 0x1401e8aa0 does not recurse +0x198. Empty vt+0x50
    // is ret (0x14000ec30). Dest is FetchDest 0x1401850a0 / 0x1401852c0.
    SceneTestBegin("Workshop3462491575.OfficialFlatDrawList");

    auto parsed = ParseWorkshopScene();
    if (parsed == nullptr) {
        std::fprintf(stderr, "SKIP full parse for 3462491575 official flat list\n");
        return;
    }

    SCENE_CHECK(parsed->GetLayerParentBinding(489).parent_id == 588);
    SCENE_CHECK(parsed->GetLayerParentBinding(1607).parent_id == 489);
    SCENE_CHECK(parsed->GetLayerParentBinding(681).parent_id == 1489);
    SCENE_CHECK(parsed->GetLayerParentBinding(5174).parent_id == 681);
    SCENE_CHECK(parsed->GetLayerParentBinding(504).parent_id == 4881);

    for (int32_t id : { 489, 1607, 681, 5174, 504, 4881 }) {
        const auto nodes = parsed->objectRuntimeNodes.find(id);
        SCENE_CHECK(nodes != parsed->objectRuntimeNodes.end());
        if (nodes == parsed->objectRuntimeNodes.end()) continue;
        for (auto* node : nodes->second) {
            SCENE_CHECK(! NodeIsProxy(*parsed, node));
        }
    }

    CheckOfficialDestMultiply(*parsed, 489, 588);
    CheckOfficialDestMultiply(*parsed, 1607, 489);
    CheckOfficialDestMultiply(*parsed, 681, 1489);
    CheckOfficialDestMultiply(*parsed, 5174, 681);
    CheckOfficialDestMultiply(*parsed, 504, 4881);

    std::vector<SceneRenderGraphPassRecord> inventory;
    auto graph = sceneToRenderGraph(*parsed, &inventory);
    SCENE_CHECK(graph != nullptr);

    auto* kaltsit_compose = FindEffectLayerForId(*parsed, 588);
    auto* m3_compose      = FindEffectLayerForId(*parsed, 1489);
    SCENE_CHECK(kaltsit_compose != nullptr && m3_compose != nullptr);
    if (kaltsit_compose == nullptr || m3_compose == nullptr) return;

    SCENE_CHECK(LayerWritesOutput(inventory, 1607, SpecTex_Default));
    SCENE_CHECK(LayerWritesOutput(inventory, 5174, SpecTex_Default));
    SCENE_CHECK(! LayerWritesOutput(inventory, 1607, kaltsit_compose->FirstTarget()));
    SCENE_CHECK(! LayerWritesOutput(inventory, 5174, m3_compose->FirstTarget()));
}

void TestWorkshop3462491575OfficialComposePassthroughChild() {
    // Official composelayer.json passthrough → 0x1401faeb8 +0x304 bit5 →
    // 0x1401fb35f +0x120 bit2. Image draw 0x1401e8f6f skips
    // `_rt_FullFrameBuffer` when the parent has that bit. Particle 4183's
    // parent is compose 1322. Empty 489/681 stay on the flat list.
    SceneTestBegin("Workshop3462491575.OfficialComposePassthroughChild");

    auto parsed = ParseWorkshopScene();
    if (parsed == nullptr) {
        std::fprintf(stderr, "SKIP full parse for 3462491575 compose passthrough child\n");
        return;
    }

    SCENE_CHECK(parsed->passthroughLayerIds.count(1322) == 1);
    SCENE_CHECK(parsed->passthroughLayerIds.count(588) == 1);
    SCENE_CHECK(parsed->passthroughLayerIds.count(1489) == 1);
    SCENE_CHECK(parsed->GetLayerParentBinding(4183).parent_id == 1322);

    const auto rain_nodes = parsed->objectRuntimeNodes.find(4183);
    SCENE_CHECK(rain_nodes != parsed->objectRuntimeNodes.end());
    if (rain_nodes != parsed->objectRuntimeNodes.end()) {
        bool rain_is_proxy = false;
        for (auto* node : rain_nodes->second) {
            rain_is_proxy = rain_is_proxy || NodeIsProxy(*parsed, node);
        }
        SCENE_CHECK(rain_is_proxy);
    }

    auto* rain_compose = FindEffectLayerForId(*parsed, 1322);
    SCENE_CHECK(rain_compose != nullptr);
    if (rain_compose == nullptr) return;

    std::vector<SceneRenderGraphPassRecord> inventory;
    auto graph = sceneToRenderGraph(*parsed, &inventory);
    SCENE_CHECK(graph != nullptr);
    SCENE_CHECK(LayerWritesOutput(inventory, 4183, rain_compose->FirstTarget()));
    SCENE_CHECK(! LayerWritesOutput(inventory, 4183, SpecTex_Default));

    // Compose-source override uses the parent effect camera (HasImgEffect) but
    // dest, not I. Official 0x1401e8f6f only skips FBO; 0x1401ec781 bit5 /
    // 0x1401e9dd5 keep dest. HasImgEffect() must not force I here.
    auto* updater = dynamic_cast<WPShaderValueUpdater*>(parsed->shaderValueUpdater.get());
    auto* rain_obj = parsed->FindSceneObject(4183);
    SCENE_CHECK(updater != nullptr && rain_obj != nullptr);
    if (updater == nullptr || rain_obj == nullptr || rain_nodes == parsed->objectRuntimeNodes.end() ||
        rain_nodes->second.empty() || rain_nodes->second.front() == nullptr) {
        return;
    }
    SceneNode* rain_node = rain_nodes->second.front();
    const std::string compose_camera { rain_compose->LayerNode() != nullptr
                                           ? rain_compose->LayerNode()->Camera()
                                           : std::string() };
    SCENE_CHECK(! compose_camera.empty());
    const auto compose_cam_it = parsed->cameras.find(compose_camera);
    SCENE_CHECK(compose_cam_it != parsed->cameras.end() && compose_cam_it->second != nullptr &&
                compose_cam_it->second->HasImgEffect());

    ShaderUniformOverrides compose_dest {
        .camera_name = compose_camera,
        .use_camera_override = true,
        .use_active_camera_for_parallax = true,
        .use_identity_model = false,
    };
    updater->InitUniforms(rain_node, [](std::string_view name) { return name == G_M; });
    sprite_map_t sprites;
    Eigen::Vector2f model_t { 0.0f, 0.0f };
    bool saw_model = false;
    updater->UpdateUniforms(
        rain_node,
        sprites,
        [&](std::string_view name, ShaderValue value) {
            if (name != G_M || value.size() < 16u) return;
            model_t = { value[12], value[13] };
            saw_model = true;
        },
        &compose_dest);
    SCENE_CHECK(saw_model);
    const auto dest = rain_obj->FetchDest();
    std::fprintf(stderr,
                 "  rain 4183 compose-dest g_M T=(%.3f,%.3f) FetchDest T=(%.3f,%.3f)\n",
                 model_t.x(),
                 model_t.y(),
                 dest(0, 3),
                 dest(1, 3));
    SCENE_CHECK(std::abs(model_t.x()) > 1.0f || std::abs(model_t.y()) > 1.0f);
    SCENE_CHECK(std::abs(model_t.x() - dest(0, 3)) < 2.0f);
    SCENE_CHECK(std::abs(model_t.y() - dest(1, 3)) < 2.0f);
}

void TestWorkshop3462491575HairAttachDest() {
    // Factory image 0x1401902b6 → ctor 0x1401fac50 → vt 0x1404911a8.
    // vt+0x70 0x1401fd510 writes +0x190 from the puppet MDAT table. FetchDest
    // 0x140185296 then calls vt+0x78 0x1401fd5c0: dest = (bone*bind)*local,
    // then 0x1401852c0 parent_dest * dest. Empty 2754 attachment=头发.
    SceneTestBegin("Workshop3462491575.HairAttachDest");

    auto parsed = ParseWorkshopScene();
    if (parsed == nullptr) {
        std::fprintf(stderr, "SKIP full parse for 3462491575 hair attach dest\n");
        return;
    }

    auto* body = parsed->FindSceneObject(1547);
    auto* hair = parsed->FindSceneObject(2754);
    auto* bangs = parsed->FindSceneObject(2548);
    SCENE_CHECK(body != nullptr && hair != nullptr && bangs != nullptr);
    if (body == nullptr || hair == nullptr || bangs == nullptr) return;

    SCENE_CHECK(hair->parent() == body);
    SCENE_CHECK(bangs->parent() == hair);
    SCENE_CHECK(body->puppet_layer() != nullptr && body->puppet_layer()->hasPuppet());
    if (body->puppet_layer() == nullptr || ! body->puppet_layer()->hasPuppet()) return;

    const auto* puppet = body->puppet_layer()->Puppet();
    SCENE_CHECK(puppet != nullptr);
    if (puppet == nullptr) return;
    const auto* named = puppet->FindAttachment("头发");
    SCENE_CHECK(named != nullptr);
    if (named != nullptr) {
        std::fprintf(stderr,
                     "  attach 头发 bone=%u bindT=(%.3f,%.3f)\n",
                     named->bone_index,
                     named->bind_transform.translation().x(),
                     named->bind_transform.translation().y());
        SCENE_CHECK(hair->attach_bone_index() == named->bone_index);
    }
    for (uint32_t i = 0; i < puppet->bones.size(); ++i) {
        const auto& b = puppet->bones[i];
        std::fprintf(stderr,
                     "  bone[%u] name='%s' parent=%u restT=(%.3f,%.3f)\n",
                     i,
                     b.name.c_str(),
                     b.parent,
                     b.transform.translation().x(),
                     b.transform.translation().y());
    }

    auto* updater = dynamic_cast<WPShaderValueUpdater*>(parsed->shaderValueUpdater.get());
    SCENE_CHECK(updater != nullptr);
    if (updater == nullptr) return;
    if (auto* hair_node = hair->source_node()) {
        const auto* hair_data = updater->GetNodeData(hair_node);
        SCENE_CHECK(hair_data != nullptr && hair_data->IsBoneAttached());
    }

    parsed->frameTime = 1.0 / 30.0;
    const uint32_t hair_bone = named != nullptr ? named->bone_index : 0xFFFFFFFFu;
    for (usize i = 0; i < body->puppet_layer()->AnimationLayerCount(); ++i) {
        const auto* layer = body->puppet_layer()->AnimationLayerState(i);
        const auto* anim  = body->puppet_layer()->AnimationDefinition(i);
        if (layer == nullptr || anim == nullptr) continue;
        std::fprintf(stderr,
                     "  anim[%zu] id=%d mode=%d playing=%d visible=%d t=%.3f len=%d\n",
                     i,
                     layer->id,
                     static_cast<int>(anim->mode),
                     layer->playing ? 1 : 0,
                     layer->visible ? 1 : 0,
                     layer->cur_time,
                     anim->length);
    }
    updater->PrepareFrame();
    const auto dest0 = hair->FetchDest();
    const auto bangs0 = bangs->FetchDest();
    const auto parent_times_local = body->FetchDest() * hair->ComposeLocalDest();
    Eigen::Vector3f bone0 = Eigen::Vector3f::Zero();
    if (hair_bone != 0xFFFFFFFFu) {
        bone0 = puppet->BoneModelTransform(hair_bone).translation();
    }
    for (int frame = 0; frame < 90; ++frame) updater->PrepareFrame();
    // 2548 is unattached. Official +0xd0==0 (0x1401850a9) recomputes and
    // walks 2754 vt+0x78. A nonzero destStamp would cache 2548 and pin hair.
    const auto bangs1 = bangs->FetchDest();
    const auto dest1 = hair->FetchDest();
    Eigen::Vector3f bone1 = Eigen::Vector3f::Zero();
    if (hair_bone != 0xFFFFFFFFu) {
        bone1 = puppet->BoneModelTransform(hair_bone).translation();
    }
    std::fprintf(stderr,
                 "  hair dest0=(%.3f,%.3f) dest1=(%.3f,%.3f) parent*local=(%.3f,%.3f)\n"
                 "  bangs dest0=(%.3f,%.3f) dest1=(%.3f,%.3f) (bangs FetchDest first)\n"
                 "  bone5 T0=(%.3f,%.3f) T1=(%.3f,%.3f)\n",
                 dest0(0, 3),
                 dest0(1, 3),
                 dest1(0, 3),
                 dest1(1, 3),
                 parent_times_local(0, 3),
                 parent_times_local(1, 3),
                 bangs0(0, 3),
                 bangs0(1, 3),
                 bangs1(0, 3),
                 bangs1(1, 3),
                 bone0.x(),
                 bone0.y(),
                 bone1.x(),
                 bone1.y());
    const float bangs_delta = std::hypot(bangs1(0, 3) - bangs0(0, 3), bangs1(1, 3) - bangs0(1, 3));
    const float bone_delta = std::hypot(dest1(0, 3) - dest0(0, 3), dest1(1, 3) - dest0(1, 3));
    const float vs_parent = std::hypot(dest0(0, 3) - parent_times_local(0, 3),
                                       dest0(1, 3) - parent_times_local(1, 3));
    const auto bangs_from_hair = dest1 * bangs->ComposeLocalDest();
    const float bangs_follow = std::hypot(bangs1(0, 3) - bangs_from_hair(0, 3),
                                          bangs1(1, 3) - bangs_from_hair(1, 3));
    SCENE_CHECK(vs_parent > 1.0f);
    SCENE_CHECK(bangs_delta > 0.05f);
    SCENE_CHECK(bone_delta > 0.05f);
    SCENE_CHECK(bangs_follow < 0.05f);
}

void TestWorkshop3462491575OfficialLightshaftsDestMesh() {
    // Same official +0x2e8 path as 3409595232: 314 has point0-3, so dest is
    // the gizmo quad (0x140260441), not the {2,2} card.
    SceneTestBegin("Workshop3462491575.OfficialLightshaftsDestMesh");

    auto parsed = ParseWorkshopScene();
    if (parsed == nullptr) {
        std::fprintf(stderr, "SKIP full parse for 3462491575 lightshafts dest mesh\n");
        return;
    }
    auto* layer = FindEffectLayerForId(*parsed, 314);
    SCENE_CHECK(layer != nullptr);
    if (layer == nullptr || layer->FinalMesh().VertexCount() == 0) return;
    const auto& va = layer->FinalMesh().GetVertexArray(0);
    SCENE_CHECK(va.Data() != nullptr && va.VertexCount() >= 4 && va.OneSize() >= 3);
    if (va.Data() == nullptr || va.VertexCount() < 4) return;
    const float x0 = va.Data()[0];
    const float y0 = va.Data()[1];
    constexpr float w = 2160.0f;
    constexpr float h = 2160.0f;
    SCENE_CHECK_NEAR(x0, w * -0.09780f - w * 0.5f, 0.05f);
    SCENE_CHECK_NEAR(y0, h * (1.0f - 0.34117f) - h * 0.5f, 0.05f);
    // Official shape dest is enum 2 additive (0x1402607a0), 4 verts + 6 indices
    // (0x140260630). Same +0x2e8 path as 3409595232.
    SCENE_CHECK(layer->FinalBlend() == BlendMode::Additive);
    SCENE_CHECK(! layer->FinalPremultipliedSourceBlend());
    SCENE_CHECK(layer->FinalMesh().IndexCount() > 0);
}

void RunWorkshop3462491575Tests() {
    TestWorkshop3462491575OfficialFlatDrawList();
    TestWorkshop3462491575OfficialComposePassthroughChild();
    TestWorkshop3462491575HairAttachDest();
    TestWorkshop3462491575OfficialLightshaftsDestMesh();
}
