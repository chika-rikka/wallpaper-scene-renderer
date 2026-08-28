#pragma once
#include "Scene/SceneObject.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace wallpaper
{

class Scene;
namespace rg
{
class RenderGraph;
}

// Test-only inventory of passes sceneToRenderGraph actually emits. Pass names are
// material names, so layer_id / node_name / output / dest_draw_phase are what
// dest-draw leftover / HORIZONTAL / last-pass can assert.
struct SceneRenderGraphPassRecord {
    int32_t       layer_id { 0 };
    std::string   node_name;
    std::string   output;
    std::string   camera;
    DestDrawPhase dest_draw_phase { DestDrawPhase::None };
};

std::unique_ptr<rg::RenderGraph> sceneToRenderGraph(Scene&);
std::unique_ptr<rg::RenderGraph> sceneToRenderGraph(
    Scene&, std::vector<SceneRenderGraphPassRecord>* inventory);
std::unique_ptr<rg::RenderGraph> sceneToPipelineWarmupRenderGraph(Scene&);
} // namespace wallpaper
