#include "test_harness.h"

#include "Audio/SoundManager.h"
#include "Fs/PhysicalFs.h"
#include "Fs/VFS.h"
#include "Interface/IShaderValueUpdater.h"
#include "Scene/Scene.h"
#include "Scene/SceneCamera.h"
#include "Scene/SceneImageEffectLayer.h"
#include "Scene/SceneMesh.h"
#include "SpecTexs.hpp"
#include "RenderGraph/RenderGraph.hpp"
#include "VulkanRender/SceneToRenderGraph.hpp"
#include "WPSceneParser.hpp"
#include "WPShaderValueUpdater.hpp"
#include "Core/MapSet.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

using namespace wallpaper;

#ifndef WESCENE_WORKSHOP_3219908811
#define WESCENE_WORKSHOP_3219908811 \
    "/media/rikka/Data/steam/steamapps/workshop/content/431960/3219908811/output"
#endif

#ifndef WESCENE_WE_ASSETS
#define WESCENE_WE_ASSETS "/media/rikka/Data/steam/steamapps/common/wallpaper_engine/assets"
#endif

namespace
{

constexpr int32_t kDateId  = 248;
constexpr int32_t kDayId   = 242;
constexpr int32_t kClockId = 315;

std::string WorkshopDir() {
    if (const char* env = std::getenv("WESCENE_WORKSHOP_3219908811"); env && env[0] != '\0') {
        return env;
    }
    return WESCENE_WORKSHOP_3219908811;
}

std::string AssetsDir() {
    if (const char* env = std::getenv("WESCENE_WE_ASSETS"); env && env[0] != '\0') {
        return env;
    }
    return WESCENE_WE_ASSETS;
}

std::shared_ptr<Scene> ParseDateWorkshop() {
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
    return parser.Parse("3219908811", scene_src, *vfs, sound, nullptr, 1.0);
}

std::vector<SceneRenderGraphPassRecord> LayerPasses(
    const std::vector<SceneRenderGraphPassRecord>& inventory, int32_t id) {
    std::vector<SceneRenderGraphPassRecord> out;
    for (const auto& pass : inventory) {
        if (pass.layer_id == id) out.push_back(pass);
    }
    return out;
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
            translation = { value[12], value[13] };
            saw         = true;
        },
        overrides);
    SCENE_CHECK(saw);
    return translation;
}

void MeshLocalAabb(const SceneMesh* mesh, float& min_x, float& max_x, float& min_y,
                   float& max_y) {
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

void CheckLastPassIsDestStackNotFetchDest(Scene& scene, SceneObject& object) {
    // ENGINE_FLUSH / LASTPASS_DEST_STACK: +0x930 = fit-ortho * dest-STACK.
    // FetchDest is I-only (DEST_LIVE_WRITERS). Do not put it on +0x930.
    const auto dest = object.FetchDest();
    scene.DestStackPushCopy();
    scene.FlushLastPassMvp();
    const Eigen::Matrix4f mvp = scene.LastPassMvp();
    const Eigen::Matrix4f cam = scene.FitOrthoCamera();
    SCENE_CHECK_NEAR(mvp(0, 0), cam(0, 0), 1e-5f);
    SCENE_CHECK_NEAR(mvp(0, 3), cam(0, 3), 1e-4f);
    SCENE_CHECK_NEAR(mvp(1, 3), cam(1, 3), 1e-4f);
    SCENE_CHECK(std::fabs(dest(0, 3)) > 1.0f || std::fabs(dest(1, 3)) > 1.0f);
    SCENE_CHECK(std::fabs(mvp(0, 3) - dest(0, 3)) > 1.0f ||
                std::fabs(mvp(1, 3) - dest(1, 3)) > 1.0f);
    const Eigen::Matrix4f draw = scene.LastPassDrawMvp(object);
    const Eigen::Matrix4f expect_draw = mvp * dest;
    SCENE_CHECK_NEAR(draw(0, 3), expect_draw(0, 3), 1e-3f);
    SCENE_CHECK_NEAR(draw(1, 3), expect_draw(1, 3), 1e-3f);
    SCENE_CHECK(std::fabs(draw(0, 3) - mvp(0, 3)) > 1.0f ||
                std::fabs(draw(1, 3) - mvp(1, 3)) > 1.0f);
    scene.DestStackPop();

    const int32_t dest_w = static_cast<int32_t>(object.dest_size().x());
    const int32_t dest_h = static_cast<int32_t>(object.dest_size().y());
    const Eigen::Matrix4f leftover = scene.LeftoverDestOrthoMvp(object);
    const float named_w = static_cast<float>(std::max(4, dest_w));
    const float named_h = static_cast<float>(std::max(4, dest_h));
    SCENE_CHECK_NEAR(leftover(0, 0), 2.0f / named_w, 1e-6f);
    SCENE_CHECK_NEAR(leftover(1, 1), 2.0f / named_h, 1e-6f);
    SCENE_CHECK_NEAR(leftover(0, 3), -1.0f, 1e-6f);
    SCENE_CHECK_NEAR(leftover(1, 3), -1.0f, 1e-6f);
    SCENE_CHECK(std::fabs(leftover(0, 3) - dest(0, 3)) > 1.0f);

    float min_x = 0, max_x = 0, min_y = 0, max_y = 0;
    MeshLocalAabb(object.lastpass_mesh(), min_x, max_x, min_y, max_y);
    SCENE_CHECK_NEAR(min_x, -0.5f * static_cast<float>(dest_w), 1e-3f);
    SCENE_CHECK_NEAR(max_x, 0.5f * static_cast<float>(dest_w), 1e-3f);
    SCENE_CHECK_NEAR(min_y, -0.5f * static_cast<float>(dest_h), 1e-3f);
    SCENE_CHECK_NEAR(max_y, 0.5f * static_cast<float>(dest_h), 1e-3f);
}

SceneImageEffectLayer* FindEffectLayer(Scene& scene, int32_t layer_id) {
    for (const auto& [name, cam] : scene.cameras) {
        if (cam == nullptr || ! cam->HasImgEffect()) continue;
        auto* layer = cam->GetImgEffect().get();
        if (layer == nullptr || layer->LayerNode() == nullptr) continue;
        const auto owner = scene.nodeOwners.find(layer->LayerNode());
        if (owner != scene.nodeOwners.end() && owner->second == layer_id) return layer;
    }
    return nullptr;
}

} // namespace

void RunWorkshop3219908811Tests() {
    // LEFTOVER_VS_DESTDRAW: Date 248 blurprecise y.json VERTICAL=1 is official last-pass
    // shader/combo, TREE emits it as AddNodePass. Clock 315 has no effect (no 0x1401ebf60).
    SceneTestBegin("Workshop3219908811.DateLastPassIsAddNodePass");

    auto parsed = ParseDateWorkshop();
    if (parsed == nullptr) {
        std::fprintf(stderr, "SKIP full parse for 3219908811 Date last-pass inventory\n");
        return;
    }

    std::vector<SceneRenderGraphPassRecord> inventory;
    auto graph = sceneToRenderGraph(*parsed, &inventory);
    SCENE_CHECK(graph != nullptr);

    const auto date  = LayerPasses(inventory, kDateId);
    const auto day   = LayerPasses(inventory, kDayId);
    const auto clock = LayerPasses(inventory, kClockId);
    for (const auto& pass : date) {
        std::fprintf(stderr, "  DATE layer=%d name='%s' output='%s' camera='%s'\n",
                     pass.layer_id, pass.node_name.c_str(), pass.output.c_str(),
                     pass.camera.c_str());
    }
    for (const auto& pass : day) {
        std::fprintf(stderr, "  DAY layer=%d name='%s' output='%s' camera='%s'\n",
                     pass.layer_id, pass.node_name.c_str(), pass.output.c_str(),
                     pass.camera.c_str());
    }
    for (const auto& pass : clock) {
        std::fprintf(stderr, "  CLOCK layer=%d name='%s' output='%s' camera='%s'\n",
                     pass.layer_id, pass.node_name.c_str(), pass.output.c_str(),
                     pass.camera.c_str());
    }

    // Official Date dest-draw: leftover vt+0xe8 dest=I then type-0 POSTFX;
    // VERTICAL no-target FullFB is 0x1401ebf60 dest=Path B dest-STACK
    // (LASTPASS_DEST_STACK). FinalNode dest blit is not 0x1401ebf60.
    SCENE_CHECK(date.size() == 3u);
    SCENE_CHECK(date[0].node_name == "Date");
    // DEST_ORTHO_TNF leftover named-RT is +0x2c8 (TREE ping-pong stand-in).
    // LASTPASS_DEST_STACK: leftover dest=I dest-ortho, not private cam.
    SCENE_CHECK(date[0].output.find("_rt_effect_pingpong") != std::string::npos);
    SCENE_CHECK(date[0].camera.empty());
    SCENE_CHECK(date[0].dest_draw_phase == DestDrawPhase::Leftover);
    // TEXT_CLEARALPHA / TEXT_E0_IDEST 0x1401e968a: leftover named-RT first
    // Draw is +0x5b0 ±half AABB (lastpass_mesh), then TEXT_E8 glyphs LOAD.
    SCENE_CHECK(date[1].output.find("_rt_FullCompoBuffer1") != std::string::npos);
    // POSTFX_MVP930 / I_SLOT: HORIZONTAL I-internal. leftover WorldNode
    // "effect" camera is not dest-ortho and not +0x930 (VERTICAL_VP).
    SCENE_CHECK(date[1].camera.empty());
    SCENE_CHECK(date[1].dest_draw_phase == DestDrawPhase::PostFx);
    // POSTFX_OMSET VERTICAL no-target leftover FullFB (TREE Default).
    SCENE_CHECK(date[2].output == std::string(SpecTex_Default));
    SCENE_CHECK(date[2].camera.empty());
    SCENE_CHECK(date[2].dest_draw_phase == DestDrawPhase::LastPass);
    SCENE_CHECK(date[2].node_name.find("__hanabi_effect_final_composite") == std::string::npos);
    SCENE_CHECK(day.size() == 3u);
    SCENE_CHECK(day.back().output == std::string(SpecTex_Default));
    SCENE_CHECK(day.back().dest_draw_phase == DestDrawPhase::LastPass);
    SCENE_CHECK(day.back().node_name.find("__hanabi_effect_final_composite") ==
                std::string::npos);
    // Clock +0x320==0: leftover only, no 0x1401ebf60.
    SCENE_CHECK(clock.size() == 1u);
    SCENE_CHECK(clock[0].output == std::string(SpecTex_Default));
    SCENE_CHECK(clock[0].dest_draw_phase == DestDrawPhase::Leftover);

    // IMAGE_VT_F0 leftover 417: +0x490 +/-half AABB onto FullFB. Live
    // +0x110 id 0xd uploads +0x8f0 (LastPassDrawMvp). No FinalNode.
    const auto lantern_passes = LayerPasses(inventory, 417);
    SCENE_CHECK(lantern_passes.size() == 1u);
    SCENE_CHECK(lantern_passes[0].output == std::string(SpecTex_Default));
    SCENE_CHECK(lantern_passes[0].camera.empty());
    SCENE_CHECK(lantern_passes[0].dest_draw_phase == DestDrawPhase::Leftover);

    // DEST_1F0_WRITERS: 人物 last-pass Draw uses dest +0x1f0, not ping-pong
    // Normal. FetchDest is I-only (LASTPASS_DEST_STACK); do not put it on
    // +0x930.
    SceneTestBegin("Workshop3219908811.CharacterLastPassUsesDestBlend");
    constexpr int32_t kHairId = 512;
    auto* hair = FindEffectLayer(*parsed, kHairId);
    SCENE_CHECK(hair != nullptr);
    if (hair == nullptr || hair->EffectCount() == 0) return;
    auto& last_effect = hair->GetEffect(hair->EffectCount() - 1);
    SCENE_CHECK(last_effect != nullptr && !last_effect->nodes.empty());
    if (last_effect == nullptr || last_effect->nodes.empty()) return;
    auto* last_mat = last_effect->nodes.back().sceneNode != nullptr &&
                             last_effect->nodes.back().sceneNode->Mesh() != nullptr
                         ? last_effect->nodes.back().sceneNode->Mesh()->Material()
                         : nullptr;
    SCENE_CHECK(last_mat != nullptr);
    if (last_mat != nullptr) {
        SCENE_CHECK(last_mat->blenmode == hair->FinalBlend());
        SCENE_CHECK(last_mat->blenmode != BlendMode::Normal);
    }
    const auto hair_passes = LayerPasses(inventory, kHairId);
    SCENE_CHECK(!hair_passes.empty());
    SCENE_CHECK(hair_passes.front().dest_draw_phase == DestDrawPhase::Leftover);
    SCENE_CHECK(hair_passes.back().output == std::string(SpecTex_Default));
    SCENE_CHECK(hair_passes.back().dest_draw_phase == DestDrawPhase::LastPass);
    SCENE_CHECK(hair_passes.back().node_name.find("__hanabi_effect_final_composite") ==
                std::string::npos);
    if (auto* hair_object = parsed->FindSceneObject(kHairId); hair_object != nullptr) {
        CheckLastPassIsDestStackNotFetchDest(*parsed, *hair_object);
    }

    // LASTPASS_DEST_STACK / I_SLOT 0x1401ec799 / leftover dest=I 0x1401e9702:
    // leftover HasImgEffect private cam and VERTICAL effect cam are I. FinalNode
    // dest blit is dest, not last-pass +0x930. ComposeDrawWalker already popped dest.
    SceneTestBegin("Workshop3219908811.DateLeftoverIsISlotDestStackDead");
    auto* updater = dynamic_cast<WPShaderValueUpdater*>(parsed->shaderValueUpdater.get());
    SCENE_CHECK(updater != nullptr);
    if (updater == nullptr) return;
    updater->PrepareFrame();
    updater->FrameBegin();
    updater->ComposeDrawWalker();
    SCENE_CHECK(parsed->DestStackAtBase());
    SCENE_CHECK_NEAR(parsed->DestStackTop()(0, 3), 0.0f, 1e-6f);
    SCENE_CHECK_NEAR(parsed->DestStackTop()(1, 3), 0.0f, 1e-6f);

    auto* date_layer = FindEffectLayer(*parsed, kDateId);
    SCENE_CHECK(date_layer != nullptr);
    if (date_layer == nullptr) return;
    SCENE_CHECK(date_layer->LayerNode() != nullptr);
    // POSTFX_OMSET: Date VERTICAL bind is HORIZONTAL FullCompo, not Default.
    // VERTICAL y.json also binds previous=leftover named-RT (TEXT_E8 dest).
    if (date_layer->EffectCount() > 0) {
        auto& effect = date_layer->GetEffect(0);
        if (effect != nullptr && !effect->nodes.empty()) {
            auto* horiz_mat = effect->nodes.front().sceneNode != nullptr &&
                                      effect->nodes.front().sceneNode->Mesh() != nullptr
                                  ? effect->nodes.front().sceneNode->Mesh()->Material()
                                  : nullptr;
            SCENE_CHECK(horiz_mat != nullptr);
            if (horiz_mat != nullptr) {
                for (size_t i = 0; i < horiz_mat->textures.size(); ++i) {
                    std::fprintf(stderr, "  DATE HORIZONTAL tex[%zu]='%s'\n", i,
                                 horiz_mat->textures[i].c_str());
                }
                SCENE_CHECK(!horiz_mat->textures.empty());
                if (!horiz_mat->textures.empty()) {
                    SCENE_CHECK(horiz_mat->textures[0].find("_rt_effect_pingpong") !=
                                std::string::npos);
                }
            }
            if (effect->nodes.size() >= 2u) {
                auto it = effect->nodes.begin();
                ++it;
                auto* last_mat = it->sceneNode != nullptr && it->sceneNode->Mesh() != nullptr
                    ? it->sceneNode->Mesh()->Material()
                    : nullptr;
                SCENE_CHECK(last_mat != nullptr);
                if (last_mat != nullptr) {
                    bool samples_fullcompo = false;
                    bool samples_default = false;
                    bool samples_leftover = false;
                    for (size_t i = 0; i < last_mat->textures.size(); ++i) {
                        std::fprintf(stderr, "  DATE VERTICAL tex[%zu]='%s' blend=%d\n", i,
                                     last_mat->textures[i].c_str(),
                                     static_cast<int>(last_mat->blenmode));
                        if (last_mat->textures[i].find("_rt_FullCompoBuffer1") !=
                            std::string::npos) {
                            samples_fullcompo = true;
                        }
                        if (last_mat->textures[i] == SpecTex_Default) samples_default = true;
                        if (last_mat->textures[i].find("_rt_effect_pingpong") !=
                            std::string::npos) {
                            samples_leftover = true;
                        }
                    }
                    SCENE_CHECK(samples_fullcompo);
                    SCENE_CHECK(!samples_default);
                    // DEST_1F0_WRITERS text vt+0x108 0x1402585a3 writes 1.
                    SCENE_CHECK(last_mat->blenmode == BlendMode::Translucent);
                    SCENE_CHECK(samples_leftover);
                }
            }
        }
    }
    const auto leftover_xy = ReadModelTranslation(*updater, date_layer->LayerNode(), nullptr);
    SCENE_CHECK_NEAR(leftover_xy.x(), 0.0f, 1e-4f);
    SCENE_CHECK_NEAR(leftover_xy.y(), 0.0f, 1e-4f);

    SCENE_CHECK(date_layer->EffectCount() > 0);
    SceneNode* vertical = nullptr;
    if (date_layer->EffectCount() > 0) {
        auto& effect = date_layer->GetEffect(0);
        if (effect != nullptr && effect->nodes.size() >= 2u) {
            auto it = effect->nodes.begin();
            ++it;
            vertical = it->sceneNode.get();
        }
    }
    SCENE_CHECK(vertical != nullptr);
    if (vertical == nullptr) return;
    ShaderUniformOverrides effect_i;
    effect_i.camera_name         = "effect";
    effect_i.use_camera_override = true;
    const auto vertical_xy = ReadModelTranslation(*updater, vertical, &effect_i);
    SCENE_CHECK_NEAR(vertical_xy.x(), 0.0f, 1e-4f);
    SCENE_CHECK_NEAR(vertical_xy.y(), 0.0f, 1e-4f);

    auto* date_object = parsed->FindSceneObject(kDateId);
    SCENE_CHECK(date_object != nullptr);
    if (date_object == nullptr) return;
    CheckLastPassIsDestStackNotFetchDest(*parsed, *date_object);
    const auto dest = date_object->FetchDest();
    SCENE_CHECK(std::fabs(dest(0, 3)) > 1.0f || std::fabs(dest(1, 3)) > 1.0f);
    {
        // TEXT_E0_FLUSH930 0x1401e968a: clearalpha g_MVP is +0x930 =
        // FitOrtho*dest-STACK (LastPassMvp) before I_SLOT / dest-ortho.
        // LastPassDrawMvp is +0x8f0 = I*that. FetchDest stays I-only.
        parsed->FlushLastPassMvp();
        const Eigen::Matrix4f clearalpha_mvp = parsed->LastPassMvp();
        const Eigen::Matrix4f draw_mvp = parsed->LastPassDrawMvp(*date_object);
        SCENE_CHECK_NEAR(clearalpha_mvp(0, 3),
                         (parsed->FitOrthoCamera() * parsed->DestStackTop())(0, 3),
                         1e-4f);
        SCENE_CHECK(std::fabs(clearalpha_mvp(0, 3) - draw_mvp(0, 3)) > 1e-3f ||
                    std::fabs(clearalpha_mvp(1, 3) - draw_mvp(1, 3)) > 1e-3f);
    }

    // EFFECT_COUNT +0x320. Date/Day IMAGE_PARSE +1 per effect; Clock +0x320==0
    // (IMAGE_DRAW_PASS no 0x1401ebf60). Text +0x2f0 is TEXT_2F0 layout AABB.
    SceneTestBegin("Workshop3219908811.EffectCountAndImageMeshes");
    const auto* day_object   = parsed->FindSceneObject(kDayId);
    const auto* clock_object = parsed->FindSceneObject(kClockId);
    SCENE_CHECK(date_object != nullptr);
    SCENE_CHECK(day_object != nullptr);
    SCENE_CHECK(clock_object != nullptr);
    SCENE_CHECK(date_object->effect_count() > 0);
    SCENE_CHECK(day_object->effect_count() > 0);
    SCENE_CHECK(clock_object->effect_count() == 0);
    {
        // Clock TEXT_VT_F0 FONT_MVP is +0x930 = FitOrtho * dest-STACK
        // (ENGINE_FLUSH / live dest_p=BASE+0x40). Not FetchDest / +0x8f0
        // (FONT_MVP_SLOT id 0xb). DestDraw FlushLastPassMvp first.
        auto* clock_mut = parsed->FindSceneObject(kClockId);
        SCENE_CHECK(clock_mut != nullptr);
        if (clock_mut != nullptr) {
            parsed->DestStackPushCopy();
            parsed->FlushLastPassMvp();
            const Eigen::Matrix4f font_mvp = parsed->LastPassMvp();
            const Eigen::Matrix4f cam      = parsed->FitOrthoCamera();
            const Eigen::Matrix4f dest     = parsed->DestStackTop();
            const Eigen::Matrix4f fetch    = clock_mut->FetchDest();
            SCENE_CHECK_NEAR(font_mvp(0, 0), cam(0, 0), 1e-5f);
            SCENE_CHECK_NEAR(font_mvp(0, 3), (cam * dest)(0, 3), 1e-4f);
            SCENE_CHECK_NEAR(font_mvp(1, 3), (cam * dest)(1, 3), 1e-4f);
            SCENE_CHECK(std::fabs(font_mvp(0, 3) - fetch(0, 3)) > 1.0f ||
                        std::fabs(font_mvp(1, 3) - fetch(1, 3)) > 1.0f);
            parsed->DestStackPop();
        }
    }
    // TEXT_LAYOUT_VERTS / CLOCK_VERT_ADD: leftover a_Position is 0..AABB
    // (ox/oy start 0, addss/subss only). DEST_ORTHO_TNF maps that into
    // named-RT NDC. Compose glyph_pages stay ±half.
    auto leftover_glyph_aabb_ok = [&](int32_t layer_id, const SceneObject& object) {
        auto node_it = parsed->layerNodes.find(layer_id);
        SCENE_CHECK(node_it != parsed->layerNodes.end());
        if (node_it == parsed->layerNodes.end() || node_it->second == nullptr) return;
        const auto* primitive = node_it->second->Text();
        SCENE_CHECK(primitive != nullptr);
        if (primitive == nullptr || primitive->leftover_glyph_pages.empty()) return;
        SCENE_CHECK(primitive->leftover_glyph_pages[0].mesh != nullptr);
        if (primitive->leftover_glyph_pages[0].mesh == nullptr) return;
        float min_x = 0, max_x = 0, min_y = 0, max_y = 0;
        MeshLocalAabb(primitive->leftover_glyph_pages[0].mesh.get(), min_x, max_x, min_y,
                      max_y);
        SCENE_CHECK(min_x >= -1.0f);
        SCENE_CHECK(min_y >= -1.0f);
        SCENE_CHECK(max_x > 1.0f);
        SCENE_CHECK(max_y > 1.0f);
        const float dest_w = object.dest_size().x();
        const float dest_h = object.dest_size().y();
        SCENE_CHECK(max_x <= dest_w + 1.0f);
        SCENE_CHECK(max_y <= dest_h + 1.0f);
        SCENE_CHECK(std::fabs(min_x + 0.5f * dest_w) > 1.0f);
        const Eigen::Matrix4f leftover_mvp = parsed->LeftoverDestOrthoMvp(object);
        std::fprintf(stderr,
                     "LeftoverGlyph id=%d dest=[%.1f %.1f] mesh=[%.1f..%.1f, %.1f..%.1f] "
                     "ortho sx=%.6f sy=%.6f T=[%.3f %.3f] glyph_display=[%.1f %.1f] "
                     "color=[%.3f %.3f %.3f] opaque=%d pages=%zu leftover_pages=%zu\n",
                     layer_id, dest_w, dest_h, min_x, max_x, min_y, max_y, leftover_mvp(0, 0),
                     leftover_mvp(1, 1), leftover_mvp(0, 3), leftover_mvp(1, 3),
                     primitive->layout.glyph_display_size[0],
                     primitive->layout.glyph_display_size[1], primitive->object.color[0],
                     primitive->object.color[1], primitive->object.color[2],
                     primitive->object.opaquebackground ? 1 : 0, primitive->glyph_pages.size(),
                     primitive->leftover_glyph_pages.size());
    };
    leftover_glyph_aabb_ok(kDateId, *date_object);
    leftover_glyph_aabb_ok(kClockId, *clock_object);
    // we_live_lastpass_draw.py aabb is obj+0x2F0 (TEXT_2F0). Live Date
    // 0x1401EC667: +0x2f0=(1412,307). TREE dest_size is JSON size+2*pad.
    std::fprintf(stderr,
                 "  Date TEXT_2F0 dest_size=[%.1f %.1f] official_live=+0x2f0=(1412,307)\n",
                 date_object->dest_size().x(), date_object->dest_size().y());
    SCENE_CHECK(date_object->leftover_mesh() != nullptr);
    SCENE_CHECK(date_object->lastpass_mesh() != nullptr);
    date_object->SizeDestDrawNamedRts();
    if (date_layer != nullptr) {
        const auto named = parsed->renderTargets.find(date_layer->FirstTarget());
        SCENE_CHECK(named != parsed->renderTargets.end());
        if (named != parsed->renderTargets.end()) {
            const int32_t expect_w =
                std::max(4, static_cast<int32_t>(date_object->dest_size().x()));
            const int32_t expect_h =
                std::max(4, static_cast<int32_t>(date_object->dest_size().y()));
            SCENE_CHECK(named->second.width == expect_w);
            SCENE_CHECK(named->second.height == expect_h);
            // UpdateAllTextLayerBridgeBackings letter-box must not win.
            // TEXT_2F0 0x140258a02 / NAMED_RT_VPSIZE leftover +0x2c8 stays
            // max(4,AABB) after SizeDestDrawNamedRts.
            named->second.width  = 64;
            named->second.height = 64;
            date_object->SizeDestDrawNamedRts();
            SCENE_CHECK(named->second.width == expect_w);
            SCENE_CHECK(named->second.height == expect_h);
        }
    }
    const auto* lantern = parsed->FindSceneObject(417);
    SCENE_CHECK(lantern != nullptr);
    if (lantern == nullptr) return;
    SCENE_CHECK(lantern->effect_count() == 0);
    SCENE_CHECK(lantern->leftover_mesh() != nullptr);
    SCENE_CHECK(lantern->lastpass_mesh() != nullptr);
    SCENE_CHECK(lantern->dest_size().x() > 1.0f);
    SCENE_CHECK(lantern->dest_size().y() > 1.0f);
    {
        auto* lantern_mut = parsed->FindSceneObject(417);
        SCENE_CHECK(lantern_mut != nullptr);
        if (lantern_mut != nullptr) {
            CheckLastPassIsDestStackNotFetchDest(*parsed, *lantern_mut);
        }
    }
    // IMAGE_490_MESH leftover +0x490 is +/-half dest, same as last-pass
    // +0x2e8. IMAGE_VT_F0 Draw uses that card, not leftover 0..AABB.
    {
        float min_x = 0, max_x = 0, min_y = 0, max_y = 0;
        MeshLocalAabb(lantern->lastpass_mesh(), min_x, max_x, min_y, max_y);
        const float dest_w = lantern->dest_size().x();
        const float dest_h = lantern->dest_size().y();
        SCENE_CHECK_NEAR(min_x, -0.5f * dest_w, 1e-3f);
        SCENE_CHECK_NEAR(max_x, 0.5f * dest_w, 1e-3f);
        SCENE_CHECK_NEAR(min_y, -0.5f * dest_h, 1e-3f);
        SCENE_CHECK_NEAR(max_y, 0.5f * dest_h, 1e-3f);
        // IMAGE_VT_128 leftover-only +0x490 a_TexCoord max is mapRate.
        const auto& va = lantern->lastpass_mesh()->GetVertexArray(0);
        const auto offsets = va.GetAttrOffsetMap();
        SCENE_CHECK(exists(offsets, WE_IN_TEXCOORD));
        const float* data = va.Data();
        const auto tex = offsets.at(std::string(WE_IN_TEXCOORD));
        float max_u = 0.0f;
        float max_v = 0.0f;
        for (std::size_t i = 0; i < va.VertexCount(); ++i) {
            const float* v = data + i * va.OneSize() + tex.offset / sizeof(float);
            max_u = std::max(max_u, v[0]);
            max_v = std::max(max_v, v[1]);
        }
        SCENE_CHECK_NEAR(max_u, lantern->leftover_uv().x(), 1e-4f);
        SCENE_CHECK_NEAR(max_v, lantern->leftover_uv().y(), 1e-4f);
        SCENE_CHECK(max_u < 0.95f || max_v < 0.95f);
    }

    // IMAGE_VT_128: 头 leftover UV is 4200/8192 × 2500/4096, not 1,1.
    const auto* head = parsed->FindSceneObject(477);
    SCENE_CHECK(head != nullptr);
    if (head == nullptr) return;
    SCENE_CHECK_NEAR(head->leftover_uv().x(), 4200.0f / 8192.0f, 1e-4f);
    SCENE_CHECK_NEAR(head->leftover_uv().y(), 2500.0f / 4096.0f, 1e-4f);
}
