#include "test_harness.h"

#include "Audio/SoundManager.h"
#include "Fs/PhysicalFs.h"
#include "Fs/VFS.h"
#include "Scene/Scene.h"
#include "Scene/SceneCamera.h"
#include "Scene/SceneImageEffectLayer.h"
#include "Scene/SceneMesh.h"
#include "Scene/SceneObject.h"
#include "SpecTexs.hpp"
#include "RenderGraph/RenderGraph.hpp"
#include "VulkanRender/SceneToRenderGraph.hpp"
#include "WPMdlParser.hpp"
#include "WPNodeTransformResolver.hpp"
#include "WPPuppet.hpp"
#include "WPSceneParser.hpp"
#include "WPShaderValueUpdater.hpp"
#include "WPTexImageParser.hpp"
#include "wpscene/WPImageObject.h"
#include "wpscene/WPParallaxDepth.hpp"
#include "wpscene/WPScene.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <set>
#include <string>
#include <string_view>
#include <vector>

using namespace wallpaper;

void RunSceneRuntimeUnitTests();
void RunWorkshop3219908811Tests();
void RunWorkshop3462491575Tests();
void RunWorkshop3409595232Tests();

namespace
{

#ifndef WESCENE_WORKSHOP_3363252053
#define WESCENE_WORKSHOP_3363252053 \
    "/media/rikka/Data/steam/steamapps/workshop/content/431960/3363252053/output"
#endif

#ifndef WESCENE_WE_ASSETS
#define WESCENE_WE_ASSETS "/media/rikka/Data/steam/steamapps/common/wallpaper_engine/assets"
#endif

const nlohmann::json* FindObject(const nlohmann::json& scene, int32_t id) {
    if (! scene.contains("objects") || ! scene.at("objects").is_array()) return nullptr;
    for (const auto& object : scene.at("objects")) {
        if (object.contains("id") && object.at("id").is_number_integer() &&
            object.at("id").get<int32_t>() == id) {
            return &object;
        }
    }
    return nullptr;
}

std::string WorkshopDir() {
    if (const char* env = std::getenv("WESCENE_WORKSHOP_3363252053"); env && env[0] != '\0') {
        return env;
    }
    return WESCENE_WORKSHOP_3363252053;
}

std::string AssetsDir() {
    if (const char* env = std::getenv("WESCENE_WE_ASSETS"); env && env[0] != '\0') {
        return env;
    }
    return WESCENE_WE_ASSETS;
}

bool WorkshopAvailable(const std::string& dir) {
    return std::filesystem::is_regular_file(std::filesystem::path(dir) / "scene.json");
}

std::unique_ptr<fs::VFS> MountWorkshop(const std::string& workshop, const std::string& assets) {
    auto vfs = std::make_unique<fs::VFS>();
    if (std::filesystem::is_directory(assets)) {
        if (! vfs->Mount("/assets", fs::CreatePhysicalFs(assets), "assets")) return nullptr;
    }
    if (! vfs->Mount("/assets", fs::CreatePhysicalFs(workshop), "workshop")) return nullptr;
    return vfs;
}

nlohmann::json LoadSceneJson(const std::string& workshop) {
    std::ifstream in(std::filesystem::path(workshop) / "scene.json");
    nlohmann::json json;
    in >> json;
    return json;
}

wpscene::WPImageObject ParseImageObject(const nlohmann::json& scene, fs::VFS& vfs, int32_t id) {
    wpscene::WPImageObject image;
    const auto*            object = FindObject(scene, id);
    SCENE_CHECK(object != nullptr);
    if (object == nullptr) return image;
    SCENE_CHECK(image.FromJson(*object, vfs));
    return image;
}

std::shared_ptr<Scene> ParseWorkshopScene() {
    const auto workshop = WorkshopDir();
    if (! WorkshopAvailable(workshop)) return nullptr;
    if (const char* skip = std::getenv("WESCENE_SKIP_FULL_PARSE"); skip && skip[0] == '1') {
        return nullptr;
    }
    auto vfs = MountWorkshop(workshop, AssetsDir());
    if (vfs == nullptr) return nullptr;
    const std::string scene_src = fs::GetFileContent(*vfs, "/assets/scene.json");
    if (scene_src.empty()) return nullptr;
    audio::SoundManager sound;
    WPSceneParser       parser;
    return parser.Parse("3363252053", scene_src, *vfs, sound, nullptr, 1.0);
}

bool LayerInGraph(const std::vector<SceneRenderGraphPassRecord>& inventory, int32_t id) {
    for (const auto& pass : inventory) {
        if (pass.layer_id == id) return true;
    }
    return false;
}

Eigen::Vector2f ReadModelTranslation(WPShaderValueUpdater& updater, SceneNode* node,
                                     const ShaderUniformOverrides* overrides) {
    updater.InitUniforms(node, [](std::string_view name) { return name == G_M; });
    sprite_map_t sprites;
    Eigen::Vector2f translation { 0.0f, 0.0f };
    bool            saw = false;
    updater.UpdateUniforms(
        node,
        sprites,
        [&](std::string_view name, ShaderValue value) {
            if (name != G_M || value.size() < 16u) return;
            // Eigen / ToDxcCBufferMatrixUniform is column-major; translation is [12],[13].
            translation = { value[12], value[13] };
            saw         = true;
        },
        overrides);
    SCENE_CHECK(saw);
    return translation;
}

Eigen::Vector2f VisibleParallaxDelta(WPShaderValueUpdater& updater, SceneNode* node,
                                     const SceneCamera* camera) {
    const auto raw  = updater.ResolveModelTransformForProjection(node, camera, false);
    const auto para = updater.ResolveModelTransformForProjection(node, camera, true);
    return { static_cast<float>(para(0, 3) - raw(0, 3)),
             static_cast<float>(para(1, 3) - raw(1, 3)) };
}

bool AabbOverlap2D(float a_min_x, float a_max_x, float a_min_y, float a_max_y, float b_min_x,
                   float b_max_x, float b_min_y, float b_max_y) {
    return a_max_x >= b_min_x && b_max_x >= a_min_x && a_max_y >= b_min_y && b_max_y >= a_min_y;
}

} // namespace

void TestSceneDocumentAuthoredIdentity() {
    SceneTestBegin("Workshop3363252053.SceneDocumentAuthoredIdentity");

    const auto workshop = WorkshopDir();
    if (! WorkshopAvailable(workshop)) {
        std::fprintf(stderr, "SKIP workshop missing: %s\n", workshop.c_str());
        return;
    }

    const auto scene = LoadSceneJson(workshop);
    SCENE_CHECK(scene.contains("objects"));
    SCENE_CHECK(scene.at("objects").size() == 48u);

    const auto* background = FindObject(scene, 157);
    const auto* body       = FindObject(scene, 751);
    const auto* head       = FindObject(scene, 4350);
    SCENE_CHECK(background != nullptr && body != nullptr && head != nullptr);
    if (background == nullptr || body == nullptr || head == nullptr) return;

    SCENE_CHECK_STREQ((*background)["name"].get<std::string>(), "背景");
    SCENE_CHECK((*background).contains("parallaxDepth"));
    SCENE_CHECK(! (*background).contains("parent"));
    SCENE_CHECK(! (*background).contains("attachment"));

    SCENE_CHECK_STREQ((*body)["name"].get<std::string>(), "身体部件");
    SCENE_CHECK((*body).contains("parallaxDepth"));
    SCENE_CHECK(! (*body).contains("attachment"));

    SCENE_CHECK_STREQ((*head)["name"].get<std::string>(), "头");
    SCENE_CHECK((*head)["parent"].get<int32_t>() == 751);
    SCENE_CHECK_STREQ((*head)["attachment"].get<std::string>(), "头");
    SCENE_CHECK(! (*head).contains("parallaxDepth"));
    SCENE_CHECK((*head).contains("effects"));
    SCENE_CHECK((*head)["effects"].size() == 2u);

    const int32_t fireworks[] = { 2553, 228, 432, 256, 430 };
    for (int32_t id : fireworks) {
        const auto* firework = FindObject(scene, id);
        SCENE_CHECK(firework != nullptr);
        if (firework == nullptr) continue;
        SCENE_CHECK((*firework)["parent"].get<int32_t>() == 157);
        SCENE_CHECK(! (*firework).contains("attachment"));
        SCENE_CHECK(! (*firework).contains("parallaxDepth"));
        SCENE_CHECK((*firework)["image"].get<std::string>() == "models/合成 1_00000.json");
    }
}

void TestWPSceneCameraParallaxBindings() {
    SceneTestBegin("Workshop3363252053.WPSceneCameraParallax");

    const auto workshop = WorkshopDir();
    if (! WorkshopAvailable(workshop)) {
        std::fprintf(stderr, "SKIP workshop missing: %s\n", workshop.c_str());
        return;
    }

    wpscene::WPScene scene;
    SCENE_CHECK(scene.FromJson(LoadSceneJson(workshop)));
    SCENE_CHECK(scene.general.cameraparallax);
    SCENE_CHECK_NEAR(scene.general.cameraparallaxamount, 0.2f, 1e-5f);
    SCENE_CHECK_NEAR(scene.general.cameraparallaxdelay, 2.0f, 1e-5f);
    SCENE_CHECK_NEAR(scene.general.cameraparallaxmouseinfluence, 0.3f, 1e-5f);
    SCENE_CHECK(scene.general.orthogonalprojection.width == 3840);
    SCENE_CHECK(scene.general.orthogonalprojection.height == 2160);
}

void TestWPImageObjectKeyLayers() {
    SceneTestBegin("Workshop3363252053.WPImageObjectKeyLayers");

    const auto workshop = WorkshopDir();
    if (! WorkshopAvailable(workshop)) {
        std::fprintf(stderr, "SKIP workshop missing: %s\n", workshop.c_str());
        return;
    }

    auto vfs = MountWorkshop(workshop, AssetsDir());
    SCENE_CHECK(vfs != nullptr);
    if (vfs == nullptr) return;
    const auto scene = LoadSceneJson(workshop);

    auto background = ParseImageObject(scene, *vfs, 157);
    SCENE_CHECK(background.id == 157);
    SCENE_CHECK(background.parent == 0);
    SCENE_CHECK(background.attachment.empty());
    SCENE_CHECK(background.parallaxDepthAuthored);
    SCENE_CHECK_NEAR(background.parallaxDepth[0], -0.92f, 1e-5f);
    SCENE_CHECK_NEAR(background.parallaxDepth[1], -0.92f, 1e-5f);
    SCENE_CHECK_NEAR(background.origin[0], 1964.01025f, 1e-4f);
    SCENE_CHECK_NEAR(background.origin[1], 1211.59460f, 1e-4f);
    SCENE_CHECK(background.effects.size() == 1u);

    auto body = ParseImageObject(scene, *vfs, 751);
    SCENE_CHECK(body.id == 751);
    SCENE_CHECK(body.puppet == "models/身体部件_puppet.mdl");
    SCENE_CHECK(body.parallaxDepthAuthored);
    SCENE_CHECK_NEAR(body.parallaxDepth[0], -0.52f, 1e-5f);
    SCENE_CHECK_NEAR(body.parallaxDepth[1], -0.16f, 1e-5f);
    SCENE_CHECK(body.effects.size() == 3u);
    SCENE_CHECK(body.puppet_layers.size() == 2u);

    auto head = ParseImageObject(scene, *vfs, 4350);
    SCENE_CHECK(head.id == 4350);
    SCENE_CHECK(head.parent == 751);
    SCENE_CHECK_STREQ(head.attachment, "头");
    SCENE_CHECK(! head.parallaxDepthAuthored);
    SCENE_CHECK_NEAR(head.parallaxDepth[0], wpscene::kDefaultParallaxDepth[0], 1e-6f);
    SCENE_CHECK_NEAR(head.size[0], 2907.0f, 1e-3f);
    SCENE_CHECK_NEAR(head.size[1], 872.0f, 1e-3f);
    SCENE_CHECK_NEAR(head.origin[0], 1297.48083f, 1e-4f);
    SCENE_CHECK_NEAR(head.origin[1], 221.27478f, 1e-4f);
    SCENE_CHECK(head.effects.size() == 2u);

    auto firework = ParseImageObject(scene, *vfs, 2553);
    SCENE_CHECK(firework.parent == 157);
    SCENE_CHECK(firework.attachment.empty());
    SCENE_CHECK(! firework.parallaxDepthAuthored);
    SCENE_CHECK_NEAR(firework.size[0], 1500.0f, 1e-3f);
    SCENE_CHECK_NEAR(firework.size[1], 1500.0f, 1e-3f);
    SCENE_CHECK(firework.effects.size() == 1u);
}

void TestWorkshopParallaxFromParsedLayers() {
    SceneTestBegin("Workshop3363252053.ParallaxFromParsedLayers");

    const auto workshop = WorkshopDir();
    if (! WorkshopAvailable(workshop)) {
        std::fprintf(stderr, "SKIP workshop missing: %s\n", workshop.c_str());
        return;
    }

    auto vfs = MountWorkshop(workshop, AssetsDir());
    SCENE_CHECK(vfs != nullptr);
    if (vfs == nullptr) return;
    const auto scenej = LoadSceneJson(workshop);

    wpscene::WPScene scene;
    SCENE_CHECK(scene.FromJson(scenej));
    const auto background = ParseImageObject(scenej, *vfs, 157);
    const auto body       = ParseImageObject(scenej, *vfs, 751);
    const auto firework   = ParseImageObject(scenej, *vfs, 2553);
    const auto head       = ParseImageObject(scenej, *vfs, 4350);

    Scene runtime;
    runtime.ortho[0] = scene.general.orthogonalprojection.width;
    runtime.ortho[1] = scene.general.orthogonalprojection.height;
    const float cam_x = static_cast<float>(runtime.ortho[0]) * 0.5f;
    const float cam_y = static_cast<float>(runtime.ortho[1]) * 0.5f;

    auto camera_node = std::make_shared<SceneNode>(Eigen::Vector3f { cam_x, cam_y, 0.0f },
                                                   Eigen::Vector3f { 1.0f, 1.0f, 1.0f },
                                                   Eigen::Vector3f::Zero());
    auto camera = std::make_shared<SceneCamera>(runtime.ortho[0], runtime.ortho[1], -1.0f, 1.0f);
    camera->AttatchNode(camera_node);
    camera->Update();
    runtime.cameras["global"] = camera;
    runtime.activeCamera      = camera.get();

    auto bg_node = std::make_shared<SceneNode>(
        Eigen::Vector3f { background.origin[0], background.origin[1], 0.0f },
        Eigen::Vector3f { 1.0f, 1.0f, 1.0f },
        Eigen::Vector3f::Zero());
    auto body_node = std::make_shared<SceneNode>(
        Eigen::Vector3f { body.origin[0], body.origin[1], 0.0f },
        Eigen::Vector3f { 1.0f, 1.0f, 1.0f },
        Eigen::Vector3f::Zero());
    auto fw_node = std::make_shared<SceneNode>(
        Eigen::Vector3f { firework.origin[0], firework.origin[1], 0.0f },
        Eigen::Vector3f { 1.0f, 1.0f, 1.0f },
        Eigen::Vector3f::Zero());
    auto head_node = std::make_shared<SceneNode>(
        Eigen::Vector3f { head.origin[0], head.origin[1], 0.0f },
        Eigen::Vector3f { 1.0f, 1.0f, 1.0f },
        Eigen::Vector3f::Zero());

    WPShaderValueData bg_data;
    bg_data.SetParallaxContract({ background.parallaxDepth[0], background.parallaxDepth[1] },
                                nullptr,
                                false,
                                background.parallaxDepthAuthored);
    WPShaderValueData body_data;
    body_data.SetParallaxContract({ body.parallaxDepth[0], body.parallaxDepth[1] },
                                  nullptr,
                                  false,
                                  body.parallaxDepthAuthored);
    WPShaderValueData fw_data;
    fw_data.SetParallaxContract({ firework.parallaxDepth[0], firework.parallaxDepth[1] },
                                bg_node.get(),
                                false,
                                firework.parallaxDepthAuthored);
    WPShaderValueData head_data;
    head_data.SetParallaxContract({ head.parallaxDepth[0], head.parallaxDepth[1] },
                                  nullptr,
                                  false,
                                  head.parallaxDepthAuthored);
    head_data.AttachToBone(body_node.get(),
                           4u,
                           Eigen::Affine3f::Identity(),
                           Eigen::Affine3f::Identity());

    Map<void*, WPShaderValueData> node_data;
    Map<void*, Eigen::Matrix4d>   model_cache;
    Map<void*, Eigen::Vector3f>   parallax_cache;
    Map<void*, Eigen::Affine3f>   attach_cache;
    node_data[bg_node.get()]   = bg_data;
    node_data[body_node.get()] = body_data;
    node_data[fw_node.get()]   = fw_data;
    node_data[head_node.get()] = head_data;

    bg_node->ID()   = 157;
    body_node->ID() = 751;
    fw_node->ID()   = 2553;
    head_node->ID() = 4350;
    runtime.nodeOwners[bg_node.get()]   = 157;
    runtime.nodeOwners[body_node.get()] = 751;
    runtime.nodeOwners[fw_node.get()]   = 2553;
    runtime.nodeOwners[head_node.get()] = 4350;
    auto fill_object = [&](int32_t id, SceneNode* node, const WPShaderValueData& data,
                           const std::array<float, 3>& origin, int32_t parent_id) {
        auto& object = runtime.EnsureSceneObject(id);
        object.set_origin(Eigen::Vector3f(origin[0], origin[1], origin[2]));
        object.set_parallax_depth(Eigen::Vector2f(data.parallaxDepth[0], data.parallaxDepth[1]));
        object.set_parallax_depth_authored(data.parallaxDepthAuthored);
        object.set_source_node(node);
        if (parent_id != 0) runtime.BindSceneObjectParent(id, parent_id, {});
    };
    fill_object(157, bg_node.get(), bg_data, background.origin, 0);
    fill_object(751, body_node.get(), body_data, body.origin, 0);
    fill_object(2553, fw_node.get(), fw_data, firework.origin, 157);
    fill_object(4350, head_node.get(), head_data, { 0.0f, 0.0f, 0.0f }, 751);

    WPCameraParallax parallax { scene.general.cameraparallax,
                                scene.general.cameraparallaxamount,
                                scene.general.cameraparallaxdelay,
                                scene.general.cameraparallaxmouseinfluence };
    std::array<float, 2> mouse { 0.0f, 1.0f };
    const auto world_lookat = OfficialLookatFromViewCamera(
        { cam_x, cam_y }, { mouse[0], mouse[1] },
        { static_cast<float>(runtime.ortho[0]), static_cast<float>(runtime.ortho[1]) },
        parallax.mouseinfluence);
    // Official lookat is [scene+0x340/+0x344] (0x14018b07d). Keep a named
    // by-value snapshot; a brace temporary must not dangle.
    const std::array<float, 2> parallax_lookat { world_lookat.x(), world_lookat.y() };
    WPNodeTransformResolver resolver(runtime,
                                    parallax,
                                    node_data,
                                    model_cache,
                                    parallax_cache,
                                    attach_cache,
                                    camera.get(),
                                    parallax_lookat,
                                    1);

    const auto bg_off   = resolver.ResolveParallaxOffset(bg_node.get(), camera.get());
    const auto fw_off   = resolver.ResolveParallaxOffset(fw_node.get(), camera.get());
    const auto body_off = resolver.ResolveParallaxOffset(body_node.get(), camera.get());
    const auto head_off = resolver.ResolveParallaxOffset(head_node.get(), camera.get());

    const auto bg_expected = ExpectedPathBOffset(
        { background.origin[0], background.origin[1] },
        { cam_x, cam_y },
        { mouse[0], mouse[1] },
        { static_cast<float>(runtime.ortho[0]), static_cast<float>(runtime.ortho[1]) },
        { background.parallaxDepth[0], background.parallaxDepth[1] },
        parallax.amount,
        parallax.mouseinfluence);
    const auto body_expected = ExpectedPathBOffset(
        { body.origin[0], body.origin[1] },
        { cam_x, cam_y },
        { mouse[0], mouse[1] },
        { static_cast<float>(runtime.ortho[0]), static_cast<float>(runtime.ortho[1]) },
        { body.parallaxDepth[0], body.parallaxDepth[1] },
        parallax.amount,
        parallax.mouseinfluence);
    const auto fw_own_wrong = ExpectedPathBOffset(
        { firework.origin[0], firework.origin[1] },
        { cam_x, cam_y },
        { mouse[0], mouse[1] },
        { static_cast<float>(runtime.ortho[0]), static_cast<float>(runtime.ortho[1]) },
        { wpscene::kDefaultParallaxDepth[0], wpscene::kDefaultParallaxDepth[1] },
        parallax.amount,
        parallax.mouseinfluence);

    SceneReportVec2("bg PathB", bg_off.x(), bg_off.y(), bg_expected.x(), bg_expected.y());
    SceneReportVec2("fw inherit", fw_off.x(), fw_off.y(), bg_off.x(), bg_off.y());
    SceneReportVec2("body PathB", body_off.x(), body_off.y(), body_expected.x(), body_expected.y());
    SceneReportVec2("head inherit", head_off.x(), head_off.y(), body_off.x(), body_off.y());
    SCENE_CHECK_NEAR(bg_off.x(), bg_expected.x(), 1e-4f);
    SCENE_CHECK_NEAR(bg_off.y(), bg_expected.y(), 1e-4f);
    SCENE_CHECK_NEAR(fw_off.x(), bg_off.x(), 1e-5f);
    SCENE_CHECK_NEAR(fw_off.y(), bg_off.y(), 1e-5f);
    SCENE_CHECK_NEAR(body_off.x(), body_expected.x(), 1e-4f);
    SCENE_CHECK_NEAR(body_off.y(), body_expected.y(), 1e-4f);
    SCENE_CHECK_NEAR(head_off.x(), body_off.x(), 1e-5f);
    SCENE_CHECK_NEAR(head_off.y(), body_off.y(), 1e-5f);
    // Official Path B has no bone skip (FUN_14018aac0). Head uses ROOT 751.
    SCENE_CHECK(head_data.AppliesModelParallax());
    const auto head_model =
        resolver.ResolveParallaxedModelTransform(head_node.get(), camera.get(), true);
    // This mock fills origin (0,0). Live JSON origin stays at +0x128; attach-zero
    // is setParent(adjust=1) only. Path B is dest-STACK, not model.col(3).
    const auto body_model =
        resolver.ResolveParallaxedModelTransform(body_node.get(), camera.get(), true);
    SCENE_CHECK_NEAR(head_model(0, 3), body_model(0, 3), 1e-3);
    SCENE_CHECK_NEAR(head_model(1, 3), body_model(1, 3), 1e-3);
    SCENE_CHECK(std::fabs(fw_off.x() - fw_own_wrong.x()) > 1.0f ||
                std::fabs(fw_off.y() - fw_own_wrong.y()) > 1.0f);
}

void TestPuppetHeadAttachmentMdat() {
    SceneTestBegin("Workshop3363252053.PuppetHeadAttachmentMdat");

    const auto workshop = WorkshopDir();
    if (! WorkshopAvailable(workshop)) {
        std::fprintf(stderr, "SKIP workshop missing: %s\n", workshop.c_str());
        return;
    }

    auto vfs = MountWorkshop(workshop, AssetsDir());
    SCENE_CHECK(vfs != nullptr);
    if (vfs == nullptr) return;

    WPMdl mdl;
    SCENE_CHECK(WPMdlParser::Parse("models/身体部件_puppet.mdl", *vfs, mdl));
    SCENE_CHECK(mdl.puppet != nullptr);
    if (mdl.puppet == nullptr) return;

    const auto* attachment = mdl.puppet->FindAttachment("头");
    SCENE_CHECK(attachment != nullptr);
    if (attachment == nullptr) return;
    SCENE_CHECK(attachment->bone_index == 4u);
    SCENE_CHECK_NEAR(attachment->transform.translation().x(), -6.8774f, 1e-3f);
    SCENE_CHECK_NEAR(attachment->transform.translation().y(), 52.1063f, 1e-3f);
    SCENE_CHECK_NEAR(attachment->transform.translation().z(), 0.0f, 1e-3f);
    SCENE_CHECK_NEAR(attachment->transform.linear()(0, 0), 1.0f, 1e-4f);
    SCENE_CHECK_NEAR(attachment->transform.linear()(1, 1), 1.0f, 1e-4f);
    SCENE_CHECK_NEAR(attachment->bind_transform.translation().x(),
                     attachment->transform.translation().x(),
                     1e-6f);

    SCENE_CHECK(mdl.puppet->bones.size() > 4u);
    SCENE_CHECK(mdl.puppet->bones[4].parent == 3u);
    SCENE_CHECK_NEAR(mdl.puppet->bones[4].transform.translation().x(), -14.147f, 1e-2f);
    SCENE_CHECK_NEAR(mdl.puppet->bones[4].transform.translation().y(), 52.135f, 1e-2f);
    SCENE_CHECK(mdl.puppet->FindBoneIndex("头") == 0xFFFFFFFFu);
}

void TestFireworkSpriteFrameCount() {
    SceneTestBegin("Workshop3363252053.FireworkSpriteFrameCount");

    const auto workshop = WorkshopDir();
    if (! WorkshopAvailable(workshop)) {
        std::fprintf(stderr, "SKIP workshop missing: %s\n", workshop.c_str());
        return;
    }

    auto vfs = MountWorkshop(workshop, AssetsDir());
    SCENE_CHECK(vfs != nullptr);
    if (vfs == nullptr) return;

    WPTexImageParser parser(vfs.get());
    const auto       header = parser.ParseHeader("合成 1_00000");
    SCENE_CHECK(header.isSprite);
    SCENE_CHECK(header.spriteAnim.numFrames() == 146u);
}

void TestWPSceneParserRuntimeIdentity() {
    SceneTestBegin("Workshop3363252053.WPSceneParserRuntimeIdentity");

    const auto workshop = WorkshopDir();
    if (! WorkshopAvailable(workshop)) {
        std::fprintf(stderr, "SKIP workshop missing: %s\n", workshop.c_str());
        return;
    }
    if (const char* skip = std::getenv("WESCENE_SKIP_FULL_PARSE"); skip && skip[0] == '1') {
        std::fprintf(stderr, "SKIP full parse (WESCENE_SKIP_FULL_PARSE=1)\n");
        return;
    }

    auto vfs = MountWorkshop(workshop, AssetsDir());
    SCENE_CHECK(vfs != nullptr);
    if (vfs == nullptr) return;

    const std::string scene_src = fs::GetFileContent(*vfs, "/assets/scene.json");
    SCENE_CHECK(! scene_src.empty());
    if (scene_src.empty()) return;

    audio::SoundManager sound;
    WPSceneParser       parser;
    auto                parsed = parser.Parse("3363252053", scene_src, *vfs, sound, nullptr, 1.0);
    SCENE_CHECK(parsed != nullptr);
    if (parsed == nullptr) return;

    SCENE_CHECK(parsed->ortho[0] == 3840);
    SCENE_CHECK(parsed->ortho[1] == 2160);
    SCENE_CHECK(parsed->cameraParallax);
    SCENE_CHECK_NEAR(parsed->cameraParallaxAmount, 0.2f, 1e-5f);

    const auto head_bind = parsed->GetLayerParentBinding(4350);
    SCENE_CHECK(head_bind.parent_id == 751);
    SCENE_CHECK_STREQ(head_bind.attachment, "头");

    for (int32_t id : { 2553, 228, 432, 256, 430 }) {
        const auto bind = parsed->GetLayerParentBinding(id);
        SCENE_CHECK(bind.parent_id == 157);
        SCENE_CHECK(bind.attachment.empty());
    }

    auto* updater = dynamic_cast<WPShaderValueUpdater*>(parsed->shaderValueUpdater.get());
    SCENE_CHECK(updater != nullptr);
    if (updater == nullptr) return;

    const auto head_nodes = parsed->objectRuntimeNodes.find(4350);
    SCENE_CHECK(head_nodes != parsed->objectRuntimeNodes.end());
    if (head_nodes == parsed->objectRuntimeNodes.end()) return;
    SCENE_CHECK(! head_nodes->second.empty());

    bool head_attached = false;
    for (auto* node : head_nodes->second) {
        const auto* data = updater->GetNodeData(node);
        if (data != nullptr && data->IsBoneAttached()) {
            head_attached = true;
            SCENE_CHECK(data->transform_binding.bone_index == 4u);
            SCENE_CHECK_NEAR(data->transform_binding.bind_transform.translation().x(),
                             -6.8774f,
                             1e-3f);
            SCENE_CHECK_NEAR(data->transform_binding.bind_transform.translation().y(),
                             52.1063f,
                             1e-3f);
        }
    }
    SCENE_CHECK(head_attached);

    const auto firework_nodes = parsed->objectRuntimeNodes.find(2553);
    SCENE_CHECK(firework_nodes != parsed->objectRuntimeNodes.end());
    SceneNode* firework_world = nullptr;
    if (firework_nodes != parsed->objectRuntimeNodes.end() && ! firework_nodes->second.empty()) {
        firework_world = firework_nodes->second.front();
        const auto* data = updater->GetNodeData(firework_world);
        SCENE_CHECK(data != nullptr);
        if (data != nullptr) {
            SCENE_CHECK(! data->IsBoneAttached());
            SCENE_CHECK(! data->parallaxDepthAuthored);
        }
    }

    const auto background_nodes = parsed->objectRuntimeNodes.find(157);
    SCENE_CHECK(background_nodes != parsed->objectRuntimeNodes.end());
    SceneNode* background_world = nullptr;
    if (background_nodes != parsed->objectRuntimeNodes.end() && ! background_nodes->second.empty()) {
        background_world = background_nodes->second.front();
        const auto* data = updater->GetNodeData(background_world);
        SCENE_CHECK(data != nullptr);
        if (data != nullptr) {
            SCENE_CHECK(data->parallaxDepthAuthored);
            SCENE_CHECK_NEAR(data->parallaxDepth[0], -0.92f, 1e-5f);
        }
    }

    if (firework_world != nullptr && background_world != nullptr) {
        const auto* fw_data = updater->GetNodeData(firework_world);
        SCENE_CHECK(fw_data != nullptr);
        if (fw_data != nullptr) {
            SCENE_CHECK(fw_data->parallax_anchor == background_world);
        }

        Map<void*, WPShaderValueData> node_data;
        Map<void*, Eigen::Matrix4d>   model_cache;
        Map<void*, Eigen::Vector3f>   parallax_cache;
        Map<void*, Eigen::Affine3f>   attach_cache;
        auto copy_data = [&](SceneNode* node) {
            if (node == nullptr) return;
            if (const auto* data = updater->GetNodeData(node); data != nullptr) {
                node_data[node] = *data;
            }
        };
        copy_data(background_world);
        copy_data(firework_world);
        if (fw_data != nullptr) copy_data(fw_data->parallax_anchor);

        WPCameraParallax parallax { parsed->cameraParallax,
                                    parsed->cameraParallaxAmount,
                                    parsed->cameraParallaxDelay,
                                    parsed->cameraParallaxMouseInfluence };
        std::array<float, 2> mouse { 0.0f, 1.0f };
        const Eigen::Vector2f ortho { static_cast<float>(parsed->ortho[0]),
                                      static_cast<float>(parsed->ortho[1]) };
        const Eigen::Vector3f cam_pos = parsed->activeCamera != nullptr
                                            ? parsed->activeCamera->GetPosition().cast<float>()
                                            : Eigen::Vector3f { 0.0f, 0.0f, 0.0f };
        const auto world_lookat = OfficialLookatFromViewCamera(
            { cam_pos.x(), cam_pos.y() }, { mouse[0], mouse[1] }, ortho,
            parsed->cameraParallaxMouseInfluence);
        // Official lookat is [scene+0x340/+0x344] (0x14018b07d). Named copy.
        const std::array<float, 2> parallax_lookat { world_lookat.x(), world_lookat.y() };
        WPNodeTransformResolver resolver(*parsed,
                                         parallax,
                                         node_data,
                                         model_cache,
                                         parallax_cache,
                                         attach_cache,
                                         parsed->activeCamera,
                                         parallax_lookat,
                                         1);
        const auto bg_off = resolver.ResolveParallaxOffset(background_world, parsed->activeCamera);
        const auto fw_off = resolver.ResolveParallaxOffset(firework_world, parsed->activeCamera);
        float expect_f0_x = 0.0f;
        float expect_f0_y = 0.0f;
        if (const auto layer_it = parsed->cameraLayers.find(1297271);
            layer_it != parsed->cameraLayers.end()) {
            expect_f0_x = layer_it->second.origin[0];
            expect_f0_y = layer_it->second.origin[1];
        }
        const float expect_view_x = ortho.x() * 0.5f + expect_f0_x;
        const float expect_view_y = ortho.y() * 0.5f + expect_f0_y;
        SceneReportVec2("fullparse camera view", cam_pos.x(), cam_pos.y(), expect_view_x,
                        expect_view_y);
        SCENE_CHECK_NEAR(cam_pos.x(), expect_view_x, 1e-4f);
        SCENE_CHECK_NEAR(cam_pos.y(), expect_view_y, 1e-4f);
        const auto bg_expected = ExpectedPathBOffset({ 1964.01025f, 1211.59460f },
                                                     { cam_pos.x(), cam_pos.y() },
                                                     { mouse[0], mouse[1] },
                                                     ortho,
                                                     { -0.92f, -0.92f },
                                                     parsed->cameraParallaxAmount,
                                                     parsed->cameraParallaxMouseInfluence);
        SceneReportVec2("fullparse bg PathB", bg_off.x(), bg_off.y(), bg_expected.x(), bg_expected.y());
        SceneReportVec2("fullparse fw inherit", fw_off.x(), fw_off.y(), bg_off.x(), bg_off.y());
        SCENE_CHECK_NEAR(bg_off.x(), bg_expected.x(), 1e-4f);
        SCENE_CHECK_NEAR(bg_off.y(), bg_expected.y(), 1e-4f);
        SCENE_CHECK_NEAR(fw_off.x(), bg_off.x(), 1e-4f);
        SCENE_CHECK_NEAR(fw_off.y(), bg_off.y(), 1e-4f);

        const auto* stars_object = parsed->FindSceneObject(200);
        SCENE_CHECK(stars_object != nullptr);
        if (stars_object != nullptr) {
            const auto* stars_root = stars_object->Root();
            std::fprintf(stderr,
                         "  stars SO200 parent=%d root=%d origin=(%.5f,%.5f) depth=(%.5f,%.5f)\n",
                         stars_object->parent() != nullptr ? stars_object->parent()->id() : 0,
                         stars_root != nullptr ? stars_root->id() : 0,
                         stars_object->origin().x(),
                         stars_object->origin().y(),
                         stars_object->parallax_depth().x(),
                         stars_object->parallax_depth().y());
            SCENE_CHECK(stars_root != nullptr && stars_root->id() == 157);
            SceneNode* stars = stars_object->source_node();
            if (stars == nullptr) {
                const auto stars_nodes = parsed->objectRuntimeNodes.find(200);
                if (stars_nodes != parsed->objectRuntimeNodes.end() &&
                    ! stars_nodes->second.empty()) {
                    stars = stars_nodes->second.front();
                }
            }
            SCENE_CHECK(stars != nullptr);
            if (stars != nullptr) {
                const auto* found = parsed->FindSceneObjectForNode(stars);
                std::fprintf(stderr,
                             "  stars node='%s' id=%d owner_obj=%d found_root=%d "
                             "found_origin=(%.5f,%.5f) found_depth=(%.5f,%.5f)\n",
                             stars->Name().c_str(),
                             stars->ID(),
                             found != nullptr ? found->id() : 0,
                             found != nullptr && found->Root() != nullptr ? found->Root()->id() : 0,
                             found != nullptr ? found->origin().x() : 0.0f,
                             found != nullptr ? found->origin().y() : 0.0f,
                             found != nullptr ? found->parallax_depth().x() : 0.0f,
                             found != nullptr ? found->parallax_depth().y() : 0.0f);
                copy_data(stars);
                const auto stars_off = resolver.ResolveParallaxOffset(stars, parsed->activeCamera);
                SceneReportVec2("fullparse stars inherit ROOT 157",
                                stars_off.x(),
                                stars_off.y(),
                                bg_off.x(),
                                bg_off.y());
                SCENE_CHECK_NEAR(stars_off.x(), bg_off.x(), 1e-4f);
                SCENE_CHECK_NEAR(stars_off.y(), bg_off.y(), 1e-4f);
            }
        }
    }
}

void TestT1GraphPassInventory() {
    // T1: CPU graph inventory from sceneToRenderGraph. This binary does not sample pixels.
    // TREE snapshot: today's pass list on 3363252053/output. Stars-only would drop 157/751/4350/烟花.
    SceneTestBegin("Workshop3363252053.T1GraphPassInventory");

    auto parsed = ParseWorkshopScene();
    if (parsed == nullptr) {
        std::fprintf(stderr, "SKIP full parse for T1\n");
        return;
    }

    std::vector<SceneRenderGraphPassRecord> inventory;
    auto graph = sceneToRenderGraph(*parsed, &inventory);
    SCENE_CHECK(graph != nullptr);
    SCENE_CHECK(! inventory.empty());

    std::set<int32_t> layers;
    bool              bg_has_offscreen = false;
    for (const auto& pass : inventory) {
        layers.insert(pass.layer_id);
        if (pass.layer_id == 157 && pass.output != std::string(SpecTex_Default)) {
            bg_has_offscreen = true;
        }
        std::fprintf(stderr,
                     "  GRAPH layer=%d name='%s' output='%s' camera='%s'\n",
                     pass.layer_id,
                     pass.node_name.c_str(),
                     pass.output.c_str(),
                     pass.camera.c_str());
    }

    SCENE_CHECK(LayerInGraph(inventory, 157));
    SCENE_CHECK(LayerInGraph(inventory, 173));
    SCENE_CHECK(LayerInGraph(inventory, 200));
    SCENE_CHECK(LayerInGraph(inventory, 2553));
    SCENE_CHECK(LayerInGraph(inventory, 228));
    SCENE_CHECK(LayerInGraph(inventory, 432));
    SCENE_CHECK(LayerInGraph(inventory, 256));
    SCENE_CHECK(LayerInGraph(inventory, 430));
    SCENE_CHECK(LayerInGraph(inventory, 751));
    SCENE_CHECK(LayerInGraph(inventory, 4350));
    SCENE_CHECK(LayerInGraph(inventory, 1175));
    SCENE_CHECK(bg_has_offscreen);
    SCENE_CHECK(layers.size() > 1u);
    SCENE_CHECK(! (layers.size() == 1u && layers.count(200) == 1u));
}

void TestT2EffectInternalVsSceneBlit() {
    // T2: official I-slot internals (0x1401e96ac) vs dest*stack scene blit (0x1401e9dd5).
    SceneTestBegin("Workshop3363252053.T2EffectInternalVsSceneBlit");

    auto parsed = ParseWorkshopScene();
    if (parsed == nullptr) {
        std::fprintf(stderr, "SKIP full parse for T2\n");
        return;
    }
    auto* updater = dynamic_cast<WPShaderValueUpdater*>(parsed->shaderValueUpdater.get());
    SCENE_CHECK(updater != nullptr);
    if (updater == nullptr) return;
    updater->PrepareFrame();
    updater->FrameBegin();
    updater->ComposeDrawWalker();

    SceneImageEffectLayer* bg_effect = nullptr;
    std::string            effect_cam;
    for (const auto& [name, cam] : parsed->cameras) {
        if (cam == nullptr || ! cam->HasImgEffect()) continue;
        auto* layer = cam->GetImgEffect().get();
        if (layer == nullptr || layer->LayerNode() == nullptr) continue;
        const auto owner = parsed->nodeOwners.find(layer->LayerNode());
        if (owner != parsed->nodeOwners.end() && owner->second == 157) {
            bg_effect  = layer;
            effect_cam = name;
            break;
        }
    }
    SCENE_CHECK(bg_effect != nullptr);
    if (bg_effect == nullptr) return;

    SceneNode* internal_node = nullptr;
    if (bg_effect->EffectCount() > 0) {
        auto& effect = bg_effect->GetEffect(0);
        if (effect != nullptr) {
            for (auto& effect_node : effect->nodes) {
                if (effect_node.sceneNode != nullptr) {
                    internal_node = effect_node.sceneNode.get();
                    break;
                }
            }
        }
    }
    SCENE_CHECK(internal_node != nullptr);
    if (internal_node == nullptr) return;

    ShaderUniformOverrides effect_override;
    // Official I-slot (0x1401e96ac) is the identity effect camera, not the private
    // HasImgEffect source camera whose name is a node address.
    effect_override.camera_name         = "effect";
    effect_override.use_camera_override = true;
    const auto internal_xy = ReadModelTranslation(*updater, internal_node, &effect_override);
    (void)effect_cam;

    auto* bg_object = parsed->FindSceneObject(157);
    SCENE_CHECK(bg_object != nullptr);
    if (bg_object == nullptr) return;
    const auto dest = bg_object->FetchDest();

    SceneReportVec2("T2 effect internal g_M", internal_xy.x(), internal_xy.y(), 0.0f, 0.0f);
    SceneReportVec2("T2 dest FetchDest", dest(0, 3), dest(1, 3), 1964.010254f, 1211.594604f);

    // Official 0x1401e96ac writes I for internals. Dest is FetchDest
    // 0x140185150 (no Path B). Path B T+= is dest-STACK (PATH_B).
    SCENE_CHECK_NEAR(internal_xy.x(), 0.0f, 1e-4f);
    SCENE_CHECK_NEAR(internal_xy.y(), 0.0f, 1e-4f);
    SCENE_CHECK_NEAR(dest(0, 3), 1964.010254f, 1e-3f);
    SCENE_CHECK_NEAR(dest(1, 3), 1211.594604f, 1e-3f);
}

void TestT4OneObjectManyDraws() {
    // T4: behavior snapshot. Reset TREE may split WorldNode/source/FinalNode. Do not require
    // wallpaper::SceneObject. Lock current runtime-node / I-vs-dest / Path B / bone-local.
    SceneTestBegin("Workshop3363252053.T4OneObjectManyDraws");

    auto parsed = ParseWorkshopScene();
    if (parsed == nullptr) {
        std::fprintf(stderr, "SKIP full parse for T4\n");
        return;
    }
    auto* updater = dynamic_cast<WPShaderValueUpdater*>(parsed->shaderValueUpdater.get());
    SCENE_CHECK(updater != nullptr);
    if (updater == nullptr) return;

    const auto bg_it = parsed->objectRuntimeNodes.find(157);
    SCENE_CHECK(bg_it != parsed->objectRuntimeNodes.end());
    if (bg_it == parsed->objectRuntimeNodes.end()) return;
    SCENE_CHECK(! bg_it->second.empty());

    std::fprintf(stderr, "  T4 157 runtime-nodes=%zu scene-objects=%zu detached-sources=%zu\n",
                 bg_it->second.size(),
                 parsed->sceneObjects.size(),
                 parsed->detachedEffectSourceNodes.size());
    // Official SO1/SO5: one object per objects[] id. I is a matrix slot, not a
    // second SceneObject / detached source identity.
    SCENE_CHECK(parsed->FindSceneObject(157) != nullptr);
    SCENE_CHECK(bg_it->second.size() >= 1u);

    const auto head_it = parsed->objectRuntimeNodes.find(4350);
    SCENE_CHECK(head_it != parsed->objectRuntimeNodes.end());
    if (head_it == parsed->objectRuntimeNodes.end()) return;
    bool head_bone = false;
    for (auto* node : head_it->second) {
        const auto* data = updater->GetNodeData(node);
        if (data != nullptr && data->IsBoneAttached()) {
            // Official Path B has no bone skip (FUN_14018aac0 / 0x14018b062).
            SCENE_CHECK(data->AppliesModelParallax());
            head_bone = true;
        }
    }
    SCENE_CHECK(head_bone);
}

SceneImageEffectLayer* FindEffectLayerForId(Scene& scene, int32_t layer_id) {
    for (const auto& [name, cam] : scene.cameras) {
        if (cam == nullptr || ! cam->HasImgEffect()) continue;
        auto* layer = cam->GetImgEffect().get();
        if (layer == nullptr) continue;
        if (layer->LayerNode() != nullptr) {
            const auto owner = scene.nodeOwners.find(layer->LayerNode());
            if (owner != scene.nodeOwners.end() && owner->second == layer_id) return layer;
        }
        const auto cams = scene.objectRuntimeCameraNames.find(layer_id);
        if (cams != scene.objectRuntimeCameraNames.end()) {
            for (const auto& cam_name : cams->second) {
                if (cam_name == name) return layer;
            }
        }
    }
    return nullptr;
}

void MeshLocalAabb(const SceneMesh* mesh, float& min_x, float& max_x, float& min_y, float& max_y) {
    min_x = max_x = min_y = max_y = 0.0f;
    if (mesh == nullptr || mesh->VertexCount() == 0) return;
    const auto& va = mesh->GetVertexArray(0);
    const float* data = va.Data();
    if (data == nullptr || va.VertexCount() == 0 || va.OneSize() < 2) return;
    min_x = max_x = data[0];
    min_y = max_y = data[1];
    for (std::size_t i = 0; i < va.VertexCount(); ++i) {
        const float x = data[i * va.OneSize()];
        const float y = data[i * va.OneSize() + 1];
        min_x = std::min(min_x, x);
        max_x = std::max(max_x, x);
        min_y = std::min(min_y, y);
        max_y = std::max(max_y, y);
    }
}

const SceneRenderGraphPassRecord* FindDestDrawPhase(
    const std::vector<SceneRenderGraphPassRecord>& inventory, int32_t layer_id,
    DestDrawPhase phase) {
    const SceneRenderGraphPassRecord* last = nullptr;
    for (const auto& pass : inventory) {
        if (pass.layer_id != layer_id) continue;
        if (pass.dest_draw_phase != phase) continue;
        last = &pass;
    }
    return last;
}

const SceneRenderGraphPassRecord* FindDestDrawLastPass(
    const std::vector<SceneRenderGraphPassRecord>& inventory, int32_t layer_id) {
    // Official last-pass 0x1401ebf60 writes leftover FullFB. TREE FinalNode
    // dest blit is not that publisher.
    return FindDestDrawPhase(inventory, layer_id, DestDrawPhase::LastPass);
}

void TestT6LiveMissCurrentTree() {
    // Live-miss harness for 3363252053 鲸鱼/身体/头/地面. Publisher is dest-draw
    // last-pass 0x1401ebf60 writing leftover FullFB, not FinalNode dest blit.
    SceneTestBegin("Workshop3363252053.T6LiveMissCurrentTree");

    auto parsed = ParseWorkshopScene();
    if (parsed == nullptr) {
        std::fprintf(stderr, "SKIP full parse for T6\n");
        return;
    }
    auto* updater = dynamic_cast<WPShaderValueUpdater*>(parsed->shaderValueUpdater.get());
    SCENE_CHECK(updater != nullptr);
    if (updater == nullptr) return;
    updater->PrepareFrame();
    updater->FrameBegin();
    updater->ComposeDrawWalker();

    std::vector<SceneRenderGraphPassRecord> inventory;
    auto graph = sceneToRenderGraph(*parsed, &inventory);
    SCENE_CHECK(graph != nullptr);

    auto* whale  = FindEffectLayerForId(*parsed, 173);
    auto* body   = FindEffectLayerForId(*parsed, 751);
    auto* head   = FindEffectLayerForId(*parsed, 4350);
    auto* ground = FindEffectLayerForId(*parsed, 1175);
    SCENE_CHECK(whale != nullptr && body != nullptr && head != nullptr && ground != nullptr);
    if (whale == nullptr || body == nullptr || head == nullptr || ground == nullptr) return;

    const auto* whale_final  = FindDestDrawLastPass(inventory, 173);
    const auto* body_final   = FindDestDrawPhase(inventory, 751, DestDrawPhase::LeftoverMvp);
    const auto* head_final   = FindDestDrawLastPass(inventory, 4350);
    const auto* ground_final = FindDestDrawLastPass(inventory, 1175);
    SCENE_CHECK(whale_final != nullptr && body_final != nullptr && head_final != nullptr &&
                ground_final != nullptr);
    if (whale_final == nullptr || body_final == nullptr || head_final == nullptr ||
        ground_final == nullptr) {
        return;
    }
    auto* body_object = parsed->FindSceneObject(751);
    SCENE_CHECK(body_object != nullptr && body_object->Flag304Bit4());
    SCENE_CHECK(body_object != nullptr && !body_object->DestDrawPublishesDefault());
    SCENE_CHECK(FindDestDrawLastPass(inventory, 751) == nullptr);
    // PUPPET_490 / IMAGE_VT_F8_PUPPET: leftover-MVP Draws +0x490 puppet
    // verts, not +0x2e8 ±half dest AABB.
    SCENE_CHECK(body_object != nullptr && body_object->image_490_mesh() != nullptr);
    SCENE_CHECK(body_object != nullptr && body_object->leftover_mvp_node() != nullptr &&
                body_object->leftover_mvp_node()->Mesh() != nullptr);
    if (body_object != nullptr && body_object->image_490_mesh() != nullptr &&
        body_object->image_490_mesh()->VertexCount() > 0) {
        SCENE_CHECK(body_object->image_490_mesh()->GetVertexArray(0).VertexCount() > 4);
    }

    // Official SO2 [scene+0x158] is a flat draw list (0x140190837); walker
    // 0x14018aebc calls each object's vt+0x50. 751 and 4350 both default
    // sortorder 0, parse order 751 then 4350, so 4350 dest blit is later.
    SCENE_CHECK_STREQ(whale_final->output, std::string(SpecTex_Default));
    SCENE_CHECK(whale_final->dest_draw_phase == DestDrawPhase::LastPass);
    SCENE_CHECK_STREQ(body_final->output, std::string(SpecTex_Default));
    SCENE_CHECK(body_final->dest_draw_phase == DestDrawPhase::LeftoverMvp);
    SCENE_CHECK_STREQ(ground_final->output, std::string(SpecTex_Default));
    SCENE_CHECK(ground_final->dest_draw_phase == DestDrawPhase::LastPass);
    SCENE_CHECK_STREQ(head_final->output, std::string(SpecTex_Default));
    SCENE_CHECK(head_final->dest_draw_phase == DestDrawPhase::LastPass);
    SCENE_CHECK(head_final->output != body->FirstTarget());
    SCENE_CHECK(head->FirstTarget() != body->FirstTarget());
    bool head_wrote_parent_pingpong = false;
    int  body_final_index           = -1;
    int  head_final_index           = -1;
    for (int i = 0; i < static_cast<int>(inventory.size()); i++) {
        const auto& pass = inventory[static_cast<size_t>(i)];
        if (pass.layer_id == 4350 && pass.output == body->FirstTarget()) {
            head_wrote_parent_pingpong = true;
        }
        if (pass.layer_id == 751 && pass.output == std::string(SpecTex_Default) &&
            pass.node_name.find("final_composite") == std::string::npos) {
            body_final_index = i;
        }
        if (pass.layer_id == 4350 && pass.output == std::string(SpecTex_Default) &&
            pass.node_name.find("final_composite") == std::string::npos) {
            head_final_index = i;
        }
    }
    SCENE_CHECK(! head_wrote_parent_pingpong);
    SCENE_CHECK(body_final_index >= 0 && head_final_index >= 0);
    SCENE_CHECK(head_final_index > body_final_index);

    struct LiveBlit {
        int32_t                id;
        const char*            name;
        SceneImageEffectLayer* effect;
        float                  expect_dest_x;
        float                  expect_dest_y;
    };
    // Official dest 0x140185150 is FetchDest from +0x128. Path B 0x14018b118
    // is dest-STACK, not FetchDest. Last-pass +0x930 is fit-ortho * dest-STACK
    // (LASTPASS_DEST_STACK). Do not add FetchDest onto last-pass mesh.
    LiveBlit layers[] = {
        { 173, "鲸鱼", whale, 2032.500732f, 2182.060791f },
        { 751, "身体部件", body, 1691.512695f, 1305.296875f },
        { 4350, "头", head, 2083.434570f, 1834.435547f },
        { 1175, "地面", ground, 1942.174072f, 654.348206f },
    };

    ShaderUniformOverrides effect_i;
    effect_i.camera_name         = "effect";
    effect_i.use_camera_override = true;

    float body_min_x = 0, body_max_x = 0, body_min_y = 0, body_max_y = 0;
    float head_min_x = 0, head_max_x = 0, head_min_y = 0, head_max_y = 0;
    bool  have_body_box = false;
    bool  have_head_box = false;

    for (auto& layer : layers) {
        SceneNode* layer_node = layer.effect->LayerNode();
        auto* object = parsed->FindSceneObject(layer.id);
        SCENE_CHECK(layer_node != nullptr && object != nullptr);
        if (layer_node == nullptr || object == nullptr) continue;

        const auto leftover_i = ReadModelTranslation(*updater, layer_node, &effect_i);
        SceneReportVec2((std::string("T6 ") + layer.name + " leftover dest=I").c_str(),
                        leftover_i.x(),
                        leftover_i.y(),
                        0.0f,
                        0.0f);
        SCENE_CHECK_NEAR(leftover_i.x(), 0.0f, 1e-4f);
        SCENE_CHECK_NEAR(leftover_i.y(), 0.0f, 1e-4f);

        const auto leftover_raw = ReadModelTranslation(*updater, layer_node, nullptr);
        SCENE_CHECK_NEAR(leftover_raw.x(), 0.0f, 1e-4f);
        SCENE_CHECK_NEAR(leftover_raw.y(), 0.0f, 1e-4f);

        const auto dest = object->FetchDest();
        SceneReportVec2((std::string("T6 ") + layer.name + " FetchDest").c_str(),
                        dest(0, 3),
                        dest(1, 3),
                        layer.expect_dest_x,
                        layer.expect_dest_y);
        SCENE_CHECK_NEAR(dest(0, 3), layer.expect_dest_x, 1e-3f);
        SCENE_CHECK_NEAR(dest(1, 3), layer.expect_dest_y, 1e-3f);

        const int32_t dest_w = static_cast<int32_t>(object->dest_size().x());
        const int32_t dest_h = static_cast<int32_t>(object->dest_size().y());
        const Eigen::Matrix4f leftover_ortho = parsed->LeftoverDestOrthoMvp(*object);
        SCENE_CHECK_NEAR(leftover_ortho(0, 0),
                         2.0f / static_cast<float>(std::max(4, dest_w)), 1e-6f);
        SCENE_CHECK_NEAR(leftover_ortho(0, 3), -1.0f, 1e-6f);
        SCENE_CHECK(std::fabs(leftover_ortho(0, 3) - dest(0, 3)) > 1.0f);
        parsed->DestStackPushCopy();
        parsed->FlushLastPassMvp();
        const Eigen::Matrix4f mvp = parsed->LastPassMvp();
        const Eigen::Matrix4f cam = parsed->FitOrthoCamera();
        SCENE_CHECK_NEAR(mvp(0, 3), cam(0, 3), 1e-4f);
        SCENE_CHECK(std::fabs(mvp(0, 3) - dest(0, 3)) > 1.0f);
        const Eigen::Matrix4f draw = parsed->LastPassDrawMvp(*object);
        const Eigen::Matrix4f expect_draw = mvp * dest;
        SCENE_CHECK_NEAR(draw(0, 3), expect_draw(0, 3), 1e-3f);
        SCENE_CHECK(std::fabs(draw(0, 3) - mvp(0, 3)) > 1.0f);
        parsed->DestStackPop();

        float mesh_min_x = 0, mesh_max_x = 0, mesh_min_y = 0, mesh_max_y = 0;
        MeshLocalAabb(object->lastpass_mesh(), mesh_min_x, mesh_max_x, mesh_min_y, mesh_max_y);
        SCENE_CHECK_NEAR(mesh_min_x, -0.5f * static_cast<float>(dest_w), 1e-3f);
        SCENE_CHECK_NEAR(mesh_max_x, 0.5f * static_cast<float>(dest_w), 1e-3f);
        SCENE_CHECK_NEAR(mesh_min_y, -0.5f * static_cast<float>(dest_h), 1e-3f);
        SCENE_CHECK_NEAR(mesh_max_y, 0.5f * static_cast<float>(dest_h), 1e-3f);
        if (layer.id == 751) {
            body_min_x = mesh_min_x;
            body_max_x = mesh_max_x;
            body_min_y = mesh_min_y;
            body_max_y = mesh_max_y;
            have_body_box = true;
        }
        if (layer.id == 4350) {
            head_min_x = mesh_min_x;
            head_max_x = mesh_max_x;
            head_min_y = mesh_min_y;
            head_max_y = mesh_max_y;
            have_head_box = true;
        }
        std::fprintf(stderr,
                     "  T6 LIVE id=%d name='%s' origin=(%.3f,%.3f) dest=(%.3f,%.3f) "
                     "lastpassXY=[%.1f,%.1f]x[%.1f,%.1f] out='%s'\n",
                     layer.id,
                     layer.name,
                     object->origin().x(),
                     object->origin().y(),
                     dest(0, 3),
                     dest(1, 3),
                     mesh_min_x,
                     mesh_max_x,
                     mesh_min_y,
                     mesh_max_y,
                     layer.id == 4350 ? head_final->output.c_str() : SpecTex_Default.data());
        SCENE_CHECK(object->lastpass_mesh() != nullptr);
    }

    SCENE_CHECK(have_body_box && have_head_box);
    if (have_body_box && have_head_box) {
        std::fprintf(stderr,
                     "  T6 head/body last-pass +0x2e8 body=[%.1f,%.1f]x[%.1f,%.1f] "
                     "head=[%.1f,%.1f]x[%.1f,%.1f]\n",
                     body_min_x,
                     body_max_x,
                     body_min_y,
                     body_max_y,
                     head_min_x,
                     head_max_x,
                     head_min_y,
                     head_max_y);
        SCENE_CHECK(AabbOverlap2D(body_min_x, body_max_x, body_min_y, body_max_y, head_min_x,
                                 head_max_x, head_min_y, head_max_y));
        const float body_cx = 0.5f * (body_min_x + body_max_x);
        const float body_cy = 0.5f * (body_min_y + body_max_y);
        const float head_cx = 0.5f * (head_min_x + head_max_x);
        const float head_cy = 0.5f * (head_min_y + head_max_y);
        SCENE_CHECK_NEAR(body_cx, 0.0f, 1e-3f);
        SCENE_CHECK_NEAR(body_cy, 0.0f, 1e-3f);
        SCENE_CHECK_NEAR(head_cx, 0.0f, 1e-3f);
        SCENE_CHECK_NEAR(head_cy, 0.0f, 1e-3f);
    }

    // Official setter 0x1401a4530 writes JSON origin to +0x128; load-time
    // parent/attach does not clear it. Dest is FetchDest (0x1401850a0).
    const auto* head_object = parsed->FindSceneObject(4350);
    SCENE_CHECK(head_object != nullptr);
    if (head_object != nullptr) {
        SceneReportVec2("T6 头 SceneObject origin",
                        head_object->origin().x(),
                        head_object->origin().y(),
                        1297.48083f,
                        221.27478f);
        SCENE_CHECK_NEAR(head_object->origin().x(), 1297.48083f, 1e-3f);
        SCENE_CHECK_NEAR(head_object->origin().y(), 221.27478f, 1e-3f);
        SceneNode* head_world = head->LayerNode();
        SCENE_CHECK(head_world != nullptr);
        if (head_world != nullptr) {
            const auto* found_world = parsed->FindSceneObjectForNode(head_world);
            Eigen::Vector2f fetch_xy { 0.0f, 0.0f };
            if (found_world != nullptr) {
                const auto dest = const_cast<SceneObject*>(found_world)->FetchDest();
                fetch_xy        = { dest(0, 3), dest(1, 3) };
            }
            head_world->UpdateTrans();
            std::fprintf(stderr,
                         "  T6 头 WorldNode id=%d find=%d fetch=(%.5f,%.5f) "
                         "model=(%.5f,%.5f) align=(%.5f,%.5f) geomT=(%.5f,%.5f)\n",
                         head_world->ID(),
                         found_world != nullptr ? found_world->id() : 0,
                         fetch_xy.x(),
                         fetch_xy.y(),
                         static_cast<float>(head_world->ModelTrans()(0, 3)),
                         static_cast<float>(head_world->ModelTrans()(1, 3)),
                         head_world->AlignmentOffset().x(),
                         head_world->AlignmentOffset().y(),
                         head_world->Mesh() != nullptr
                             ? head_world->Mesh()->GeometryTransform().translation().x()
                             : 0.0f,
                         head_world->Mesh() != nullptr
                             ? head_world->Mesh()->GeometryTransform().translation().y()
                             : 0.0f);
            SceneReportVec2("T6 头 FetchDest",
                            fetch_xy.x(),
                            fetch_xy.y(),
                            2083.434570f,
                            1834.435547f);
            SCENE_CHECK_NEAR(fetch_xy.x(), 2083.434570f, 1e-3f);
            SCENE_CHECK_NEAR(fetch_xy.y(), 1834.435547f, 1e-3f);
        }
    }
}

void TestVisibleParallaxApplyInherit() {
    // Offset() inherit can stay equal while dest-axes apply 2× Stars / rotate 烟花.
    // Lock the translation actually written into the model (T.xy +=).
    SceneTestBegin("Workshop3363252053.VisibleParallaxApplyInherit");

    auto parsed = ParseWorkshopScene();
    if (parsed == nullptr) {
        std::fprintf(stderr, "SKIP full parse for visible Path B apply\n");
        return;
    }
    auto* updater = dynamic_cast<WPShaderValueUpdater*>(parsed->shaderValueUpdater.get());
    SCENE_CHECK(updater != nullptr);
    if (updater == nullptr) return;
    updater->PrepareFrame();
    updater->FrameBegin();
    updater->ComposeDrawWalker();

    auto* bg_effect = FindEffectLayerForId(*parsed, 157);
    auto* fw_effect = FindEffectLayerForId(*parsed, 2553);
    SCENE_CHECK(bg_effect != nullptr && fw_effect != nullptr);
    if (bg_effect == nullptr || fw_effect == nullptr) return;

    const SceneCamera* cam = parsed->activeCamera;
    auto* bg_object = parsed->FindSceneObject(157);
    auto* fw_object = parsed->FindSceneObject(2553);
    SCENE_CHECK(bg_object != nullptr && fw_object != nullptr);
    if (bg_object == nullptr || fw_object == nullptr) return;
    const auto bg_delta = bg_object->leftover_parallax();
    const auto fw_delta = fw_object->leftover_parallax();
    (void)cam;
    SceneReportVec2("visible apply 烟花 vs 背景", fw_delta.x(), fw_delta.y(), bg_delta.x(),
                    bg_delta.y());
    SCENE_CHECK(std::fabs(bg_delta.x()) > 1.0f || std::fabs(bg_delta.y()) > 1.0f);
    SCENE_CHECK_NEAR(fw_delta.x(), bg_delta.x(), 1e-3f);
    SCENE_CHECK_NEAR(fw_delta.y(), bg_delta.y(), 1e-3f);

    auto* stars_object = parsed->FindSceneObject(200);
    SCENE_CHECK(stars_object != nullptr);
    if (stars_object == nullptr) return;
    const auto stars_delta = stars_object->leftover_parallax();
    SceneReportVec2("visible apply Stars vs 背景", stars_delta.x(), stars_delta.y(), bg_delta.x(),
                    bg_delta.y());
    SCENE_CHECK_NEAR(stars_delta.x(), bg_delta.x(), 1e-3f);
    SCENE_CHECK_NEAR(stars_delta.y(), bg_delta.y(), 1e-3f);
    SCENE_CHECK(std::fabs(stars_delta.x() - 2.0f * bg_delta.x()) > 1.0f ||
                std::fabs(stars_delta.y() - 2.0f * bg_delta.y()) > 1.0f);
}

void TestT5HarnessDoesNotSampleFramebuffer() {
    SceneTestBegin("Workshop3363252053.T5HarnessHonesty");
    // T5: this binary never reads a color attachment. T1 locks graph inventory; T2 locks
    // updater matrices. A stars-only framebuffer can still pass Path B numbers alone.
    SCENE_CHECK(true);
}

int main() {
    RunSceneRuntimeUnitTests();
    TestSceneDocumentAuthoredIdentity();
    TestWPSceneCameraParallaxBindings();
    TestWPImageObjectKeyLayers();
    TestWorkshopParallaxFromParsedLayers();
    TestPuppetHeadAttachmentMdat();
    TestFireworkSpriteFrameCount();
    TestWPSceneParserRuntimeIdentity();
    TestT1GraphPassInventory();
    TestT2EffectInternalVsSceneBlit();
    TestT4OneObjectManyDraws();
    TestT6LiveMissCurrentTree();
    TestVisibleParallaxApplyInherit();
    TestT5HarnessDoesNotSampleFramebuffer();
    RunWorkshop3219908811Tests();
    RunWorkshop3462491575Tests();
    RunWorkshop3409595232Tests();
    return SceneTestSummary();
}
