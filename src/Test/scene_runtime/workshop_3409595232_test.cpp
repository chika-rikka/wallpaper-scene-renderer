#include "test_harness.h"

#include "Audio/SoundManager.h"
#include "Fs/PhysicalFs.h"
#include "Fs/VFS.h"
#include "Scene/Scene.h"
#include "Scene/SceneCamera.h"
#include "Scene/SceneImageEffectLayer.h"
#include "Scene/SceneMesh.h"
#include "WPSceneParser.hpp"

#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>

using namespace wallpaper;

void RunWorkshop3409595232Tests();

namespace
{

#ifndef WESCENE_WORKSHOP_3409595232
#define WESCENE_WORKSHOP_3409595232 \
    "/media/rikka/Data/steam/steamapps/workshop/content/431960/3409595232/output"
#endif

#ifndef WESCENE_WE_ASSETS
#define WESCENE_WE_ASSETS "/media/rikka/Data/steam/steamapps/common/wallpaper_engine/assets"
#endif

std::string WorkshopDir() {
    if (const char* env = std::getenv("WESCENE_WORKSHOP_3409595232"); env && env[0] != '\0') {
        return env;
    }
    return WESCENE_WORKSHOP_3409595232;
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
    return parser.Parse("3409595232", scene_src, *vfs, sound, nullptr, 1.0);
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

bool ReadDestPositions(const SceneMesh& mesh, std::array<float, 12>& pos) {
    if (mesh.VertexCount() == 0) return false;
    const auto& va = mesh.GetVertexArray(0);
    if (va.Data() == nullptr || va.VertexCount() < 4 || va.OneSize() < 3) return false;
    const float* data = va.Data();
    const auto   stride = va.OneSize();
    for (int i = 0; i < 4; ++i) {
        pos[static_cast<std::size_t>(i) * 3 + 0] = data[static_cast<std::size_t>(i) * stride + 0];
        pos[static_cast<std::size_t>(i) * 3 + 1] = data[static_cast<std::size_t>(i) * stride + 1];
        pos[static_cast<std::size_t>(i) * 3 + 2] = data[static_cast<std::size_t>(i) * stride + 2];
    }
    return true;
}

} // namespace

void TestWorkshop3409595232OfficialLightshaftsDestMesh() {
    // Official shape vt+0x110 0x140260441: dest +0x2e8 vertices are
    // x=w*u-w/2, y=h*(1-v)-h/2 for point0..3. Unsized shape +0x2f0 is
    // height×height (0x14025fac0). 1511 has no size on a 3840×2160 ortho, so
    // w=h=2160. The {2,2} dest card left a vertical clip at the 2160 edge.
    SceneTestBegin("Workshop3409595232.OfficialLightshaftsDestMesh");

    auto parsed = ParseWorkshopScene();
    if (parsed == nullptr) {
        std::fprintf(stderr, "SKIP full parse for 3409595232 lightshafts dest mesh\n");
        return;
    }

    auto* layer = FindEffectLayerForId(*parsed, 1511);
    SCENE_CHECK(layer != nullptr);
    if (layer == nullptr) return;

    std::array<float, 12> pos {};
    SCENE_CHECK(ReadDestPositions(layer->FinalMesh(), pos));

    constexpr float w = 2160.0f;
    constexpr float h = 2160.0f;
    const std::array<std::array<float, 2>, 4> points {{
        { -0.36761f, 0.96988f },
        { -0.80924f, -0.18247f },
        { 1.50806f, -0.63106f },
        { 1.11470f, 0.93796f },
    }};
    for (int i = 0; i < 4; ++i) {
        const float u        = points[static_cast<std::size_t>(i)][0];
        const float v        = points[static_cast<std::size_t>(i)][1];
        const float expect_x = w * u - w * 0.5f;
        const float expect_y = h * (1.0f - v) - h * 0.5f;
        SCENE_CHECK_NEAR(pos[static_cast<std::size_t>(i) * 3 + 0], expect_x, 0.05f);
        SCENE_CHECK_NEAR(pos[static_cast<std::size_t>(i) * 3 + 1], expect_y, 0.05f);
    }
    SCENE_CHECK(pos[3] < -2000.0f);
    // Official dest create passes 6 indices (0x140260630). Strip order in
    // point0..3 leaves triangle (0,2,3) empty.
    SCENE_CHECK(layer->FinalMesh().IndexCount() > 0);
    SCENE_CHECK(layer->FinalMesh().LogicalIndexCount() == 6);
    if (layer->FinalMesh().IndexCount() > 0) {
        const auto& ia = layer->FinalMesh().GetIndexArray(0);
        SCENE_CHECK(ia.Data() != nullptr);
        if (ia.Data() != nullptr) {
            const auto* packed = reinterpret_cast<const uint16_t*>(ia.Data());
            SCENE_CHECK(packed[0] == 0);
            SCENE_CHECK(packed[1] == 1);
            SCENE_CHECK(packed[2] == 2);
            SCENE_CHECK(packed[3] == 0);
            SCENE_CHECK(packed[4] == 2);
            SCENE_CHECK(packed[5] == 3);
        }
    }
    // Official shape dest blend is enum 2 (0x1402607a0), not premul source-over.
    SCENE_CHECK(layer->FinalBlend() == BlendMode::Additive);
    SCENE_CHECK(! layer->FinalPremultipliedSourceBlend());
    if (layer->EffectCount() > 0) {
        auto& effect = layer->GetEffect(0);
        if (! effect->nodes.empty() && effect->nodes.front().sceneNode != nullptr &&
            effect->nodes.front().sceneNode->Mesh() != nullptr &&
            effect->nodes.front().sceneNode->Mesh()->Material() != nullptr) {
            const auto& consts =
                effect->nodes.front().sceneNode->Mesh()->Material()->customShader.constValues;
            const auto it = consts.find("g_Point0");
            SCENE_CHECK(it != consts.end());
            if (it != consts.end()) {
                SCENE_CHECK(it->second.size() >= 2);
                SCENE_CHECK_NEAR(it->second[0], -0.36761f, 0.0001f);
                SCENE_CHECK_NEAR(it->second[1], 0.96988f, 0.0001f);
            }
        }
    }
}

void RunWorkshop3409595232Tests() {
    TestWorkshop3409595232OfficialLightshaftsDestMesh();
}
