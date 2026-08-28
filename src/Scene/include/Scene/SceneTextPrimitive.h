#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "Image.hpp"
#include "SceneMesh.h"
#include "wpscene/WPTextObject.h"

namespace wallpaper
{

struct TextGlyphRun {
    uint32_t             page_index { 0 };
    std::array<float, 4> source_rect { 0.0f, 0.0f, 0.0f, 0.0f };
    std::array<float, 4> atlas_rect { 0.0f, 0.0f, 0.0f, 0.0f };
};

struct TextGlyphAtlasPage {
    std::string            texture_key;
    std::shared_ptr<Image> image;
    std::array<float, 2>   source_size { 0.0f, 0.0f };
};

struct TextLayoutResult {
    // The canonical logical box always tracks the authored text rectangle after shaping. This is
    // the box used by opaque-background rendering and by exact-size text bridges.
    std::array<float, 2> logical_size { 0.0f, 0.0f };
    std::array<float, 2> logical_source_size { 0.0f, 0.0f };

    // Glyph-only text can expose cropped glyph bounds as visible geometry, but placement still
    // belongs to `logical_size`. The cropped visible quad starts from its measured local offset from
    // that logical rectangle; WPTextLayer resolves the final mesh center from these crop metrics and
    // the authored alignment/origin.
    std::array<float, 2> glyph_display_size { 0.0f, 0.0f };
    std::array<float, 2> glyph_source_size { 0.0f, 0.0f };
    std::array<float, 2> glyph_offset { 0.0f, 0.0f };
    std::array<float, 4> glyph_source_crop { 0.0f, 0.0f, 0.0f, 0.0f };
    std::array<float, 2> visible_display_size { 0.0f, 0.0f };
    std::array<float, 2> visible_source_size { 0.0f, 0.0f };
    std::array<float, 2> visible_display_offset { 0.0f, 0.0f };

    // These values describe independent renderer contracts. Point-size conversion changes
    // authored glyph geometry, while backing density only changes atlas resolution; neither object
    // scale nor the scene camera is folded into either value.
    float point_size_authoring_units { 0.0f };
    float backing_density { 1.0f };

    std::vector<TextGlyphAtlasPage> glyph_pages;
    std::vector<TextGlyphRun>       glyph_runs;
};

struct TextBridgeRenderTarget {
    // A first-class text bridge owns projected-density offscreen targets that image effects sample.
    // `scale` and `fit` mirror Wallpaper Engine's authored effect FBO sizing rules so runtime text
    // updates can recompute the same target dimensions that parse-time image/text effect material
    // construction used. Keeping the sizing metadata with the bridge avoids texture-resolution
    // hacks inside shader passes when an effect target is resized after text content changes.
    std::string name;
    uint32_t    scale { 1 };
    uint32_t    fit { 0 };
    // Feedback targets retain history across frames. Their physical grid must remain in the
    // authored effect domain instead of following transient projected text scale changes that
    // would recreate the image and discard the accumulated state.
    bool        persistent_feedback { false };
};

struct TextSourceBridge {
    std::string camera_name;
    std::string pingpong_a;
    std::string pingpong_b;

    // The Vulkan backing follows final projected screen coverage at the glyph raster density.
    // Authored local geometry and the logical effect sampling grid remain owned by TextLayoutResult
    // and the bridge camera; the backing therefore retains text antialiasing headroom without
    // reverting to the full authored source rectangle for a heavily downscaled layer.
    std::array<uint32_t, 2> bridge_backing_extent { 1u, 1u };
    std::array<uint32_t, 2> projected_output_extent { 0u, 0u };
    std::vector<TextBridgeRenderTarget> render_targets;
};

struct TextLayerRenderContract {
    bool has_materialized_authored_effects { false };
    bool uses_shader_color_blend_bridge { false };

    [[nodiscard]] bool RequiresBridge() const {
        return has_materialized_authored_effects || uses_shader_color_blend_bridge;
    }
};

class SceneTextPrimitive {
public:
    struct GlyphPageRenderable {
        uint32_t                 page_index { 0 };
        std::string              texture_key;
        std::array<float, 2>     source_size { 0.0f, 0.0f };
        std::shared_ptr<SceneMesh> mesh;
    };

    // The primitive is scene-owned state: authored text properties, canonical layout, atlas page
    // meshes, material data, and optional bridge metadata all live here so parser, runtime, and
    // render graph consume one final representation instead of synthetic image-layer sidecars.
    wpscene::WPTextObject object;
    TextLayerRenderContract render_contract;
    TextLayoutResult      layout;
    TextSourceBridge      bridge;
    std::shared_ptr<SceneMesh> background_mesh;
    std::vector<GlyphPageRenderable> glyph_pages;
    // TEXT_LAYOUT_VERTS leftover dest-draw: layout-local 0..AABB, not
    // ±half. TEXT_E8 / DEST_ORTHO_TNF maps (0..W,0..H) to NDC.
    std::vector<GlyphPageRenderable> leftover_glyph_pages;
    uint32_t              atlas_version { 0 };

    [[nodiscard]] std::array<float, 2> VisibleDisplaySize() const { return layout.visible_display_size; }
    [[nodiscard]] std::array<float, 2> VisibleSourceSize() const { return layout.visible_source_size; }
    [[nodiscard]] std::array<float, 2> VisibleDisplayOffset() const { return layout.visible_display_offset; }
    [[nodiscard]] std::array<float, 2> BackgroundLocalOffset() const {
        return object.opaquebackground
            ? std::array<float, 2> { 0.0f, 0.0f }
            : std::array<float, 2> { -layout.visible_display_offset[0], -layout.visible_display_offset[1] };
    }
};

} // namespace wallpaper
