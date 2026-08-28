#include "WPSceneParser.hpp"
#include "WPJson.hpp"
#include "WPUserProperties.hpp"

#include "Utils/String.h"
#include "Utils/Logging.h"
#include "Utils/Algorism.h"
#include "Utils/Eigen.h"
#include "Core/Visitors.hpp"
#include "Core/StringHelper.hpp"
#include "Core/ArrayHelper.hpp"
#include "SpecTexs.hpp"
#include "VulkanRender/Msaa.hpp"
#include "Scene/ShadowAtlas.hpp"
#include "Scene/LightingV1.hpp"
#include "Scene/SceneTexture.h"
#include "Scene/SceneIndexArray.h"

#include "WPShaderParser.hpp"
#include "WPTexImageParser.hpp"
#include "WPSyntheticImageParser.hpp"
#include "WPParticleParser.hpp"
#include "WPSoundParser.hpp"
#include "WPMdlParser.hpp"
#include "WPPropertyAnimation.hpp"
#include "WPSceneScriptMedia.hpp"
#include "WPTextLayer.hpp"
#include "WPUserSetting.hpp"
#include "WPImageAlignment.hpp"

#include "Particle/ParticleRenderPlan.h"
#include "Particle/WPParticleRawGener.h"
#include "Particle/ParticleSystem.h"

#include "WPShaderValueUpdater.hpp"
#include "wpscene/WPImageObject.h"
#include "wpscene/WPParallaxDepth.hpp"
#include "wpscene/WPParticleObject.h"
#include "wpscene/WPSoundObject.h"
#include "wpscene/WPLightObject.hpp"
#include "wpscene/WPScene.h"

#include "Fs/VFS.h"

#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <cctype>
#include <random>
#include <cmath>
#include <functional>
#include <regex>
#include <optional>
#include <type_traits>
#include <variant>
#include <limits>
#include <cstring>
#include <span>
#include <Eigen/Dense>

using namespace wallpaper;
using namespace Eigen;

std::string getAddr(void* p) { return std::to_string(reinterpret_cast<intptr_t>(p)); }

uint32_t HashParticleFrameU32(uint32_t seed, uint32_t bits) {
    seed ^= bits + 0x9e3779b9u + (seed << 6u) + (seed >> 2u);
    return seed;
}

uint32_t HashParticleFrameFloat(uint32_t seed, float value) {
    uint32_t bits { 0 };
    std::memcpy(&bits, &value, sizeof(bits));
    return HashParticleFrameU32(seed, bits);
}

float RandomParticleFrameLifetime(const Particle& p, float sprite_frame_count_value) {
    const auto sprite_frame_count =
        static_cast<uint32_t>(std::max(1.0f, std::round(sprite_frame_count_value)));
    if (sprite_frame_count <= 1u) return 0.0f;

    // The shader only receives one float named "lifetime", so randomframe encodes a stable
    // atlas cell into that float. Hash spawn-time identity rather than live velocity, which
    // changes every operator tick.
    uint32_t seed = 2166136261u;
    seed          = HashParticleFrameFloat(seed, p.init.lifetime);
    seed          = HashParticleFrameFloat(seed, p.init.size);
    seed          = HashParticleFrameFloat(seed, p.init.color.x());
    seed          = HashParticleFrameFloat(seed, p.init.color.y());
    seed          = HashParticleFrameFloat(seed, p.init.color.z());
    seed          = HashParticleFrameU32(seed, static_cast<uint32_t>(p.spawnSequence));
    seed          = HashParticleFrameU32(seed, static_cast<uint32_t>(p.spawnSequence >> 32));

    const uint32_t frame = seed % sprite_frame_count;
    return (static_cast<float>(frame) + 0.5f) / static_cast<float>(sprite_frame_count);
}

std::string DescribeIndexVec(const std::vector<usize>& values) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < values.size(); i++) {
        if (i != 0) oss << ", ";
        oss << values[i];
    }
    oss << "]";
    return oss.str();
}

std::optional<std::string_view> TextureFormatShaderDefine(TextureFormat format) {
    // Wallpaper Engine's stock shader headers use TEXnFORMAT preprocessor symbols to choose the
    // correct channel layout for compressed normal maps, R/RG masks, and light-gradient textures.
    // The old path only populated TEX0FORMAT, which left later slots such as normal maps
    // (TEX1FORMAT), PBR gradients (TEX4FORMAT), and fur/detail masks (TEX8FORMAT) to compile as
    // the implicit RGBA default under DXC. Keep the policy here at the material/texture binding
    // boundary: the shader remains authored WE code, while the parser supplies the real texture
    // metadata for every bound slot.
    switch (format) {
    case TextureFormat::RGBA8: return "FORMAT_RGBA8888";
    case TextureFormat::RGB8: return "FORMAT_RGB888";
    case TextureFormat::BC1: return "FORMAT_DXT1";
    case TextureFormat::BC2: return "FORMAT_DXT3";
    case TextureFormat::BC3: return "FORMAT_DXT5";
    case TextureFormat::RG8: return "FORMAT_RG88";
    case TextureFormat::R8: return "FORMAT_R8";
    }
    return std::nullopt;
}

void SetTextureFormatShaderDefine(WPShaderInfo& shader_info, usize slot, TextureFormat format) {
    const auto define = TextureFormatShaderDefine(format);
    if (! define.has_value()) return;
    shader_info.combos["TEX" + std::to_string(slot) + "FORMAT"] = std::string(*define);
}

void ExtractReferencedLayerNamesFromScript(std::string_view                 script,
                                           std::unordered_set<std::string>& out_names) {
    auto is_identifier_char = [](char ch) {
        const unsigned char uch = static_cast<unsigned char>(ch);
        return std::isalnum(uch) != 0 || ch == '_' || ch == '$';
    };

    auto skip_whitespace_backward = [&script](size_t index) {
        while (index > 0 && std::isspace(static_cast<unsigned char>(script[index - 1])) != 0) {
            index--;
        }
        return index;
    };

    auto is_layer_lookup_literal = [&](size_t literal_start) {
        // Large Wallpaper Engine scripts can contain hundreds of kilobytes of text.  The previous
        // std::regex-based scan for generic string literals could recurse deeply enough inside
        // libstdc++ to segfault before the scene parser even finished loading.  This lightweight
        // backwards check keeps the same "string literal immediately inside getLayer(...)" signal
        // without entering the catastrophic regex executor path.
        size_t cursor = skip_whitespace_backward(literal_start);
        if (cursor == 0 || script[cursor - 1] != '(') return false;

        cursor          = skip_whitespace_backward(cursor - 1);
        size_t name_end = cursor;
        while (cursor > 0 && is_identifier_char(script[cursor - 1])) {
            cursor--;
        }

        const std::string_view fn_name = script.substr(cursor, name_end - cursor);
        return fn_name == "getLayer" || fn_name == "getLayerIndex";
    };

    for (size_t index = 0; index < script.size(); index++) {
        const char quote = script[index];
        if (quote != '\'' && quote != '"') continue;

        std::string literal;
        literal.reserve(32);

        bool   escaped = false;
        size_t cursor  = index + 1;
        for (; cursor < script.size(); cursor++) {
            const char ch = script[cursor];
            if (escaped) {
                // The script scanner only needs a stable best-effort string token for layer-name
                // discovery, so keeping escaped characters as their literal payload is sufficient
                // and avoids building a full JavaScript parser in the hot scene-load path.
                literal.push_back(ch);
                escaped = false;
                continue;
            }

            if (ch == '\\') {
                escaped = true;
                continue;
            }

            if (ch == quote) break;
            literal.push_back(ch);
        }

        if (cursor >= script.size()) break;
        if (! literal.empty()) out_names.insert(literal);

        // Some scene scripts keep layer names in string arrays first and only call getLayer(name)
        // later via map/forEach helpers, so we still collect every plain string literal.  We also
        // preserve the explicit getLayer/getLayerIndex signal because it helps future logging and
        // debugging distinguish direct layer lookups from unrelated strings without any regex use.
        if (is_layer_lookup_literal(index) && ! literal.empty()) {
            out_names.insert(literal);
        }

        index = cursor;
    }
}

void CollectScriptReferencedLayerNames(const nlohmann::json&            value,
                                       std::unordered_set<std::string>& out_names) {
    if (value.is_object()) {
        for (auto it = value.begin(); it != value.end(); ++it) {
            if (it.key() == "script" && it->is_string()) {
                ExtractReferencedLayerNamesFromScript(it->get_ref<const std::string&>(), out_names);
                continue;
            }
            CollectScriptReferencedLayerNames(*it, out_names);
        }
        return;
    }

    if (value.is_array()) {
        for (const auto& item : value) {
            CollectScriptReferencedLayerNames(item, out_names);
        }
    }
}

// Text-layer bindings and deferred materialization span authored JSON, runtime state, and script
// registration data. Keeping these small predicates centralized avoids duplicating the text-layer
// identification rules across parser entry points that need to make the same decisions.
bool IsTextLayerObjectJson(const nlohmann::json& object_json) {
    return object_json.contains("text") && ! object_json.at("text").is_null();
}

const char* DynamicValueTypeName(WPDynamicValue::Type hint) {
    switch (hint) {
    case WPDynamicValue::Type::Null: return "null";
    case WPDynamicValue::Type::Boolean: return "bool";
    case WPDynamicValue::Type::Int32: return "int32";
    case WPDynamicValue::Type::UInt32: return "uint32";
    case WPDynamicValue::Type::Float: return "float";
    case WPDynamicValue::Type::Double: return "double";
    case WPDynamicValue::Type::String: return "string";
    case WPDynamicValue::Type::FloatVector: return "floatVector";
    case WPDynamicValue::Type::Int3: return "int3";
    case WPDynamicValue::Type::Float2: return "float2";
    case WPDynamicValue::Type::Float3: return "float3";
    case WPDynamicValue::Type::Float4: return "float4";
    }
    return "unknown";
}

bool IsParserOpacityUniformName(std::string_view uniform_name) {
    return uniform_name == "alpha" || uniform_name == "g_Alpha" ||
           uniform_name == "g_UserAlpha";
}

float ClampParserOpacityScalar(float opacity) {
    if (! std::isfinite(opacity)) return 0.0f;
    return std::clamp(opacity, 0.0f, 1.0f);
}

std::array<i32, 4> ResolvePaddedSpriteSheetResolution(const ImageHeader& texh,
                                                       const SpriteFrame& frame) {
    const auto physical_width  = texh.width > 0 ? texh.width : texh.mapWidth;
    const auto physical_height = texh.height > 0 ? texh.height : texh.mapHeight;
    auto       content_width   = texh.mapWidth > 0 ? texh.mapWidth : physical_width;
    auto       content_height  = texh.mapHeight > 0 ? texh.mapHeight : physical_height;

    const auto frame_width = static_cast<i32>(std::lround(frame.width));
    if (frame_width > 0) content_width -= content_width % frame_width;
    const auto frame_height = static_cast<i32>(std::lround(frame.height));
    if (frame_height > 0) content_height -= content_height % frame_height;

    return { physical_width, physical_height, content_width, content_height };
}

ShaderValue ClampParserOpacityUniformValue(std::string_view uniform_name,
                                           const ShaderValue& value) {
    if (! IsParserOpacityUniformName(uniform_name) || value.size() == 0) return value;

    // Parser-time material constants can be consumed before the script host advances animations.
    // Clamp only normalized opacity uniforms here so cold-start alpha writes match the runtime
    // registration boundary without flattening intentionally overshooting non-opacity curves.
    ShaderValue clamped = value;
    clamped[0]          = ClampParserOpacityScalar(clamped[0]);
    return clamped;
}

void LogTextLayerRegistration(const char* event_name, int32_t object_id,
                              const std::string& object_name, std::string_view property_name,
                              WPDynamicValue::Type hint, const WPUserSetting& setting,
                              const std::optional<WPDynamicValue>& base_value) {}

enum class LazyMaterializeKind
{
    None,
    Image,
    Particle,
    Text,
};

struct VisibilityContract {
    bool                authored_visible { true };
    bool                initial_visible { true };
    bool                has_user_binding { false };
    bool                has_script { false };
    bool                has_animation { false };
    bool                referenced_by_script { false };
    bool                dependency_source { false };
    bool                requires_runtime_contract { false };
    bool                can_prune_at_parse_time { false };
    LazyMaterializeKind lazy_materialize_kind { LazyMaterializeKind::None };
};

struct ParseContext {
    std::shared_ptr<Scene> scene;
    WPShaderValueUpdater*  shader_updater;
    i32                    ortho_w;
    i32                    ortho_h;
    fs::VFS*               vfs;
    const UserPropertyMap* user_properties;

    ShaderValueMap                                          global_base_uniforms;
    std::shared_ptr<SceneNode>                              effect_camera_node;
    std::shared_ptr<SceneNode>                              global_camera_node;
    std::shared_ptr<SceneNode>                              global_perspective_camera_node;
    std::unordered_set<int32_t>                             dependent_parent_ids;
    std::unordered_map<int32_t, VisibilityContract>         layer_visibility_contracts;
    std::unordered_map<int32_t, int32_t>                    initial_parent_by_layer_id;
    std::unordered_map<int32_t, std::shared_ptr<SceneNode>> object_nodes;
    std::unordered_map<int32_t, const WPPuppet*>            object_puppets;
    // Official passthrough layers: model key at 0x1401fae95, +0x304 bit5 at
    // 0x1401faeb8, +0x120 bit2 at 0x1401fb35f.
    std::unordered_set<int32_t>                             passthrough_layer_ids;
    // Model chunk passes share the main scene target. Tracking their parse order here lets the
    // model-only material state preserve color after the first model pass without changing the
    // legacy load-op behavior of 2D image/effect passes.
    std::unordered_map<std::string, usize> model_pass_count_by_output;
};

std::array<float, 2> ImageObjectParallaxDepth(const wpscene::WPImageObject& object) {
    return { object.parallaxDepth[0], object.parallaxDepth[1] };
}

std::array<float, 2> TextObjectParallaxDepth(const wpscene::WPTextObject& object) {
    return { object.parallaxDepth[0], object.parallaxDepth[1] };
}

void NotePassthroughLayer(ParseContext& context, const wpscene::WPImageObject& image) {
    if (image.id == 0 || ! image.config.passthrough) return;
    context.passthrough_layer_ids.insert(image.id);
    if (context.scene != nullptr) context.scene->passthroughLayerIds.insert(image.id);
}

bool LayerUsesRoutedParent(const ParseContext& context, int32_t parent_id,
                           std::string_view attachment) {
    (void)attachment;
    // Official image draw 0x1401e8f63 loads parent at +0x180. If that parent
    // has +0x120 bit2 it skips the `_rt_FullFrameBuffer` bind at 0x1401e8f78
    // (0x1401e8f6f jne 0x1401e8f9b). Bit2 is written only for passthrough:
    // 0x1401fae95 key, 0x1401faeb8 or +0x304 0x20, 0x1401fb35f or +0x120 4.
    // models/util/composelayer.json and projectlayer.json both set
    // passthrough. Empty 0x2c0 objects still draw nothing (vt+0x50
    // 0x14000ec30) and are not registered as proxies.
    return parent_id != 0 && context.passthrough_layer_ids.count(parent_id) != 0;
}

struct WPEmptyObject {
    int32_t              id { 0 };
    std::string          name;
    std::array<float, 3> origin { 0.0f, 0.0f, 0.0f };
    std::array<float, 3> scale { 1.0f, 1.0f, 1.0f };
    std::array<float, 3> angles { 0.0f, 0.0f, 0.0f };
    std::array<float, 2> parallaxDepth { wpscene::kDefaultParallaxDepth };
    bool                 parallaxDepthAuthored { false };
    bool                 visible { true };
    VisibleBinding       visible_binding;
    int32_t              parent { 0 };
    std::string          attachment;
    bool                 is_camera_layer { false };
    std::string          camera_name;
    std::string          camera_path;
    float                fov { 50.0f };
    float                zoom { 1.0f };

    bool FromJson(const nlohmann::json& json, fs::VFS&) {
        GET_JSON_NAME_VALUE_NOWARN(json, "name", name);
        GET_JSON_NAME_VALUE_NOWARN(json, "id", id);
        GET_JSON_NAME_VALUE_NOWARN(json, "origin", origin);
        GET_JSON_NAME_VALUE_NOWARN(json, "scale", scale);
        GET_JSON_NAME_VALUE_NOWARN(json, "angles", angles);
        parallaxDepthAuthored =
            json.contains("parallaxDepth") && ! json.at("parallaxDepth").is_null();
        GET_JSON_NAME_VALUE_NOWARN(json, "parallaxDepth", parallaxDepth);
        GET_JSON_NAME_VALUE_NOWARN(json, "visible", visible);
        if (json.contains("visible")) {
            GET_JSON_NAME_VALUE_NOWARN(json.at("visible"), "value", visible_binding.value);
            if (json.at("visible").contains("user") && ! json.at("visible").at("user").is_null()) {
                const auto& user = json.at("visible").at("user");
                if (user.is_string()) {
                    GET_JSON_VALUE(user, visible_binding.user.name);
                } else if (user.is_object()) {
                    GET_JSON_NAME_VALUE_NOWARN(user, "name", visible_binding.user.name);
                    GET_JSON_NAME_VALUE_NOWARN(user, "condition", visible_binding.user.condition);
                }
            }
        }
        GET_JSON_NAME_VALUE_NOWARN(json, "parent", parent);
        GET_JSON_NAME_VALUE_NOWARN(json, "attachment", attachment);
        if (json.contains("camera") && json.at("camera").is_string()) {
            // Camera assets do not have an image/particle/text discriminator, so they otherwise
            // arrive here as empty layers. Preserve their camera marker and authored projection
            // values so ParseEmptyObj can register a runtime camera target instead of a harmless
            // transform-only placeholder.
            GET_JSON_NAME_VALUE_NOWARN(json, "camera", camera_name);
            is_camera_layer = true;
        }
        if (json.contains("path") && json.at("path").is_string()) {
            GET_JSON_NAME_VALUE_NOWARN(json, "path", camera_path);
            if (camera_path.find("scripts/camera_paths_") == 0) {
                is_camera_layer = true;
            }
        }
        GET_JSON_NAME_VALUE_NOWARN(json, "fov", fov);
        GET_JSON_NAME_VALUE_NOWARN(json, "zoom", zoom);
        return true;
    }
};

struct WPShapeObject {
    int32_t                             id { 0 };
    std::string                         name;
    std::string                         shape;
    std::array<float, 3>                origin { 0.0f, 0.0f, 0.0f };
    std::array<float, 3>                scale { 1.0f, 1.0f, 1.0f };
    std::array<float, 3>                angles { 0.0f, 0.0f, 0.0f };
    std::array<float, 2>                size { 0.0f, 0.0f };
    std::array<float, 2>                parallaxDepth { wpscene::kDefaultParallaxDepth };
    bool                                parallaxDepthAuthored { false };
    std::array<float, 3>                color { 1.0f, 1.0f, 1.0f };
    float                               alpha { 1.0f };
    float                               brightness { 1.0f };
    bool                                visible { true };
    bool                                has_size { false };
    VisibleBinding                      visible_binding;
    int32_t                             parent { 0 };
    std::string                         attachment;
    std::vector<wpscene::WPImageEffect> effects;

    bool FromJson(const nlohmann::json& json, fs::VFS& vfs) {
        GET_JSON_NAME_VALUE_NOWARN(json, "name", name);
        GET_JSON_NAME_VALUE_NOWARN(json, "id", id);
        GET_JSON_NAME_VALUE_NOWARN(json, "shape", shape);
        GET_JSON_NAME_VALUE_NOWARN(json, "origin", origin);
        GET_JSON_NAME_VALUE_NOWARN(json, "scale", scale);
        GET_JSON_NAME_VALUE_NOWARN(json, "angles", angles);
        if (json.contains("size") && ! json.at("size").is_null()) {
            GET_JSON_NAME_VALUE_NOWARN(json, "size", size);
            has_size = size[0] > 0.0f && size[1] > 0.0f;
        }
        parallaxDepthAuthored =
            json.contains("parallaxDepth") && ! json.at("parallaxDepth").is_null();
        GET_JSON_NAME_VALUE_NOWARN(json, "parallaxDepth", parallaxDepth);
        GET_JSON_NAME_VALUE_NOWARN(json, "color", color);
        GET_JSON_NAME_VALUE_NOWARN(json, "alpha", alpha);
        GET_JSON_NAME_VALUE_NOWARN(json, "brightness", brightness);

        if (json.contains("visible") && ! json.at("visible").is_null()) {
            const auto& visible_json = json.at("visible");
            if (visible_json.is_boolean()) {
                GET_JSON_VALUE_NOWARN(visible_json, visible);
                visible_binding.value = visible;
            } else if (visible_json.is_object()) {
                // Shape layers can be driven by the same user/script visibility contracts as image
                // layers. Preserve the authored fallback and binding instead of treating the whole
                // object as a boolean, otherwise direct-draw effects would materialize with the
                // wrong initial visibility and runtime toggles would have no stable target.
                GET_JSON_NAME_VALUE_NOWARN(visible_json, "value", visible_binding.value);
                visible = visible_binding.value;
                if (visible_json.contains("user") && ! visible_json.at("user").is_null()) {
                    const auto& user = visible_json.at("user");
                    if (user.is_string()) {
                        GET_JSON_VALUE(user, visible_binding.user.name);
                    } else if (user.is_object()) {
                        GET_JSON_NAME_VALUE_NOWARN(user, "name", visible_binding.user.name);
                        GET_JSON_NAME_VALUE_NOWARN(
                            user, "condition", visible_binding.user.condition);
                    }
                }
            }
        }

        GET_JSON_NAME_VALUE_NOWARN(json, "parent", parent);
        GET_JSON_NAME_VALUE_NOWARN(json, "attachment", attachment);

        if (json.contains("effects") && json.at("effects").is_array()) {
            for (const auto& effect_json : json.at("effects")) {
                wpscene::WPImageEffect effect;
                if (effect.FromJson(effect_json, vfs)) {
                    effects.push_back(std::move(effect));
                } else {
                    LOG_ERROR(
                        "ShapeObject: effect parse failed, layer=%d name='%s'", id, name.c_str());
                }
            }
        }
        return true;
    }
};

struct WPModelObject {
    int32_t              id { 0 };
    std::string          name;
    std::array<float, 3> origin { 0.0f, 0.0f, 0.0f };
    std::array<float, 3> scale { 1.0f, 1.0f, 1.0f };
    std::array<float, 3> angles { 0.0f, 0.0f, 0.0f };
    bool                 visible { true };
    VisibleBinding       visible_binding;
    int32_t              parent { 0 };
    std::string          attachment;
    std::string          model;
    int32_t              skin { 0 };
    bool                 reflected { false };
    // 3D models omit this key when they should cast. Non-casters write false.
    bool                 castshadow { true };

    bool FromJson(const nlohmann::json& json, fs::VFS&) {
        GET_JSON_NAME_VALUE_NOWARN(json, "name", name);
        GET_JSON_NAME_VALUE_NOWARN(json, "id", id);
        GET_JSON_NAME_VALUE_NOWARN(json, "origin", origin);
        GET_JSON_NAME_VALUE_NOWARN(json, "scale", scale);
        GET_JSON_NAME_VALUE_NOWARN(json, "angles", angles);
        GET_JSON_NAME_VALUE_NOWARN(json, "visible", visible);
        if (json.contains("visible") && json.at("visible").is_object()) {
            GET_JSON_NAME_VALUE_NOWARN(json.at("visible"), "value", visible_binding.value);
            if (json.at("visible").contains("user") && ! json.at("visible").at("user").is_null()) {
                const auto& user = json.at("visible").at("user");
                if (user.is_string()) {
                    GET_JSON_VALUE(user, visible_binding.user.name);
                } else if (user.is_object()) {
                    GET_JSON_NAME_VALUE_NOWARN(user, "name", visible_binding.user.name);
                    GET_JSON_NAME_VALUE_NOWARN(user, "condition", visible_binding.user.condition);
                }
            }
        }
        GET_JSON_NAME_VALUE_NOWARN(json, "parent", parent);
        GET_JSON_NAME_VALUE_NOWARN(json, "attachment", attachment);
        GET_JSON_NAME_VALUE_NOWARN(json, "model", model);
        GET_JSON_NAME_VALUE_NOWARN(json, "skin", skin);
        GET_JSON_NAME_VALUE_NOWARN(json, "reflected", reflected);
        GET_JSON_NAME_VALUE_NOWARN(json, "castshadow", castshadow);
        return ! model.empty();
    }
};

using WPObjectVar =
    std::variant<wpscene::WPImageObject, wpscene::WPParticleObject, wpscene::WPSoundObject,
                 wpscene::WPLightObject, wpscene::WPTextObject, WPModelObject, WPShapeObject,
                 WPEmptyObject>;

constexpr std::string_view kSceneModelPerspectiveCameraName { "__hanabi_model_perspective" };
constexpr std::string_view kModelReflectionTargetName { "_rt_Reflection" };

namespace
{
bool           SceneHasShadowLights(const Scene& scene);
LightingV1Desc LightingDescFromScene(const Scene& scene);

constexpr std::string_view kSyntheticDirectDrawShapeTextureName {
    "__hanabi_shape_directdraw_transparent_source"
};

bool UsesShaderColorBlendMode(int32_t color_blend_mode) {
    return color_blend_mode >= 1 && color_blend_mode <= 30;
}

struct FinalShaderCapabilityEntry {
    std::string_view shader;
    FinalOutputCapability capability;
};

constexpr std::array<FinalShaderCapabilityEntry, 2> kFinalShaderCapabilities {{
    // Scroll evaluates UV/time in the authored layer projection. Running its visible raster in a
    // source-sized private target quantizes repeated pixel art before the layer is scaled to the
    // display. The capability is attached to the shader contract itself, never to a wallpaper,
    // texture name, or nearest-sampler heuristic.
    { "effects/scroll", FinalOutputCapability::SceneAuthoredWriter },
    // X-Ray unprojects the normalized desktop cursor through the inverse authored layer MVP and
    // divides the resulting local position by the source texture extent. A private effect-camera
    // pass uses a 2x2 clip-space helper mesh whose XY MVP is identity, collapsing cursor movement
    // to roughly one source texel before the neutral publication pass. Keep the authored final
    // shader in scene space so its matrix, pointer coordinates, and visible layer geometry remain
    // one coherent projection contract.
    { "effects/xray", FinalOutputCapability::SceneAuthoredWriter },
}};

FinalOutputCapability ResolveFinalShaderCapability(std::string_view shader) {
    const auto it = std::find_if(kFinalShaderCapabilities.begin(),
                                 kFinalShaderCapabilities.end(),
                                 [shader](const auto& entry) { return entry.shader == shader; });
    return it == kFinalShaderCapabilities.end()
        ? FinalOutputCapability::PrivateThenPublish
        : it->capability;
}

bool IsCurrentEffectWriterTarget(std::string_view target) {
    return target == SpecTex_Default || sstart_with(target, WE_EFFECT_PPONG_PREFIX_B);
}

BlendMode ResolveObjectFinalBlend(BlendMode authored_blend, int32_t color_blend_mode) {
    return color_blend_mode == 31 ? BlendMode::Additive : authored_blend;
}

void EnsureSystemTextureRegistered(Scene& scene, std::string_view texture_key) {
    auto* synthetic_parser = AsSyntheticImageParser(scene.imageParser.get());
    if (synthetic_parser == nullptr) return;

    const std::string key(texture_key);
    if (scene.textures.count(key) != 0) return;

    synthetic_parser->RegisterImage(key, CreateSceneScriptSolidImage(texture_key, { 0, 0, 0, 0 }));
    scene.textures[key] = SceneTexture {
        .url = key,
        .sample =
            TextureSample {
                .wrapS     = TextureWrap::CLAMP_TO_EDGE,
                .wrapT     = TextureWrap::CLAMP_TO_EDGE,
                .magFilter = TextureFilter::LINEAR,
                .minFilter = TextureFilter::LINEAR,
            },
        .format    = TextureFormat::RGBA8,
        .isVideo   = false,
        .isSprite  = false,
        .width     = 1,
        .height    = 1,
        .mapWidth     = 1,
        .mapHeight    = 1,
        .mipmapCount  = 1,
    };
    scene.dirtyImportedTextureKeys.insert(key);
}

bool ResolveObjectVisibility(bool raw_visible, const VisibleBinding& binding,
                             const UserPropertyMap* user_properties) {
    if (! binding.hasUserBinding()) return raw_visible;
    return EvaluateVisibleBinding(binding, user_properties);
}

bool JsonPropertyHasRuntimeMember(const nlohmann::json* property_json, const char* member_name) {
    return property_json != nullptr && property_json->is_object() &&
           property_json->contains(member_name) && ! property_json->at(member_name).is_null();
}

const nlohmann::json* FindVisibleProperty(const nlohmann::json& json) {
    if (! json.is_object() || ! json.contains("visible")) return nullptr;
    return &json.at("visible");
}

bool IsCameraLayerObjectJson(const nlohmann::json& object_json) {
    if (! object_json.is_object()) return false;
    if (object_json.contains("camera") && object_json.at("camera").is_string()) return true;
    if (! object_json.contains("path") || ! object_json.at("path").is_string()) return false;

    // Camera assets can be serialized as otherwise-empty objects that only carry a camera path.
    // Treat those as camera layers too, because their zoom/origin properties still drive the
    // active view even when the path file itself contains no authored points.
    const auto path = object_json.at("path").get<std::string>();
    return path.find("scripts/camera_paths_") == 0;
}

bool IsCameraLayerRuntimeProperty(std::string_view property_name) {
    return property_name == "visible" || property_name == "origin" || property_name == "angles" ||
           property_name == "zoom" || property_name == "fov";
}

bool ReadAuthoredVisibleValue(const nlohmann::json& json, bool default_visible = true) {
    const auto* visible_json = FindVisibleProperty(json);
    if (visible_json == nullptr) return default_visible;
    if (visible_json->is_boolean()) return visible_json->get<bool>();
    if (visible_json->is_object() && visible_json->contains("value") &&
        visible_json->at("value").is_boolean()) {
        return visible_json->at("value").get<bool>();
    }
    return default_visible;
}

VisibleBinding ReadVisibleBindingFromJson(const nlohmann::json* visible_json,
                                          bool                  authored_visible) {
    VisibleBinding binding;
    binding.value = authored_visible;
    if (visible_json == nullptr || ! visible_json->is_object()) return binding;

    GET_JSON_NAME_VALUE_NOWARN(*visible_json, "value", binding.value);
    if (! visible_json->contains("user") || visible_json->at("user").is_null()) return binding;

    const auto& user = visible_json->at("user");
    if (user.is_string()) {
        GET_JSON_VALUE(user, binding.user.name);
    } else if (user.is_object()) {
        GET_JSON_NAME_VALUE_NOWARN(user, "name", binding.user.name);
        GET_JSON_NAME_VALUE_NOWARN(user, "condition", binding.user.condition);
    }
    return binding;
}

VisibilityContract
BuildVisibilityContract(bool authored_visible, const VisibleBinding& binding,
                        const nlohmann::json* visible_json, const UserPropertyMap* user_properties,
                        bool dependency_source = false, bool referenced_by_script = false,
                        LazyMaterializeKind lazy_kind = LazyMaterializeKind::None) {
    VisibilityContract contract;
    contract.authored_visible = authored_visible;
    contract.initial_visible  = ResolveObjectVisibility(authored_visible, binding, user_properties);
    contract.has_user_binding =
        binding.hasUserBinding() || JsonPropertyHasRuntimeMember(visible_json, "user");
    contract.has_script                = JsonPropertyHasRuntimeMember(visible_json, "script");
    contract.has_animation             = JsonPropertyHasRuntimeMember(visible_json, "animation");
    contract.referenced_by_script      = referenced_by_script;
    contract.dependency_source         = dependency_source;
    contract.lazy_materialize_kind     = lazy_kind;
    contract.requires_runtime_contract = contract.has_user_binding || contract.has_script ||
                                         contract.has_animation || referenced_by_script ||
                                         dependency_source;
    // Parse-time pruning is reserved for purely static invisible authored content. Anything that
    // can be driven by scripts, user properties, animations, dependencies, or name references keeps
    // a runtime contract even if it begins hidden.
    contract.can_prune_at_parse_time =
        ! contract.initial_visible && ! contract.requires_runtime_contract;
    return contract;
}

VisibilityContract BuildObjectVisibilityContractFromJson(
    const nlohmann::json& obj, const UserPropertyMap* user_properties, bool dependency_source,
    bool referenced_by_script, LazyMaterializeKind lazy_kind = LazyMaterializeKind::None) {
    const auto* visible_json     = FindVisibleProperty(obj);
    const bool  authored_visible = ReadAuthoredVisibleValue(obj);
    const auto  binding          = ReadVisibleBindingFromJson(visible_json, authored_visible);
    return BuildVisibilityContract(authored_visible,
                                   binding,
                                   visible_json,
                                   user_properties,
                                   dependency_source,
                                   referenced_by_script,
                                   lazy_kind);
}

VisibilityContract BuildEffectVisibilityContract(const wpscene::WPImageEffect& effect,
                                                 const UserPropertyMap*        user_properties) {
    const nlohmann::json* visible_json =
        effect.visible_json.is_null() ? nullptr : &effect.visible_json;
    return BuildVisibilityContract(
        effect.visible, effect.visible_binding, visible_json, user_properties);
}

const VisibilityContract* FindLayerVisibilityContract(const ParseContext& context,
                                                      int32_t             layer_id) {
    const auto it = context.layer_visibility_contracts.find(layer_id);
    return it == context.layer_visibility_contracts.end() ? nullptr : &it->second;
}

int32_t FindInitialParentLayerId(const ParseContext& context, int32_t layer_id) {
    const auto it = context.initial_parent_by_layer_id.find(layer_id);
    return it == context.initial_parent_by_layer_id.end() ? 0 : it->second;
}

bool LayerHasRuntimeVisibilityInInitialAncestry(const ParseContext& context, int32_t layer_id) {
    std::unordered_set<int32_t> visited;
    int32_t current = layer_id;
    while (current != 0 && visited.insert(current).second) {
        if (const auto* contract = FindLayerVisibilityContract(context, current);
            contract != nullptr && contract->requires_runtime_contract) {
            return true;
        }
        current = FindInitialParentLayerId(context, current);
    }
    return false;
}

bool LayerInitiallyVisibleThroughInitialAncestry(const ParseContext& context, int32_t layer_id,
                                                bool local_initial_visible) {
    if (! local_initial_visible) return false;

    std::unordered_set<int32_t> visited;
    int32_t current = layer_id;
    while (current != 0 && visited.insert(current).second) {
        if (const auto* contract = FindLayerVisibilityContract(context, current);
            contract != nullptr && ! contract->initial_visible) {
            return false;
        }
        current = FindInitialParentLayerId(context, current);
    }
    return true;
}

bool ShouldDeferRuntimeLayerMaterialization(const ParseContext& context, int32_t layer_id,
                                            bool local_initial_visible,
                                            const VisibilityContract* contract,
                                            bool force_runtime_materialization) {
    if (force_runtime_materialization) return false;
    if (contract != nullptr && contract->dependency_source) return false;
    if (LayerInitiallyVisibleThroughInitialAncestry(context, layer_id, local_initial_visible)) {
        return false;
    }

    // A multilingual branch often keeps each child layer locally visible and hides the branch at an
    // ancestor controlled by user settings or scripts. Looking only at the child authored flag
    // materializes every hidden language subtree. The initial parent map lets parser-side lazy
    // materialization follow the same effective-visibility contract that Scene later applies to
    // runtime nodes, while dependency-source layers still stay concrete for consumers that sample
    // their render targets.
    return LayerHasRuntimeVisibilityInInitialAncestry(context, layer_id);
}

void PopulateGlobalBaseUniforms(ParseContext& context, const Scene& scene) {
    auto& gb                   = context.global_base_uniforms;
    gb["g_ViewUp"]             = std::array { 0.0f, 1.0f, 0.0f };
    gb["g_ViewRight"]          = std::array { 1.0f, 0.0f, 0.0f };
    gb["g_ViewForward"]        = std::array { 0.0f, 0.0f, -1.0f };
    gb["g_TexelSize"]          = std::array { 1.0f / 1920.0f, 1.0f / 1080.0f };
    gb["g_TexelSizeHalf"]      = std::array { 1.0f / 1920.0f / 2.0f, 1.0f / 1080.0f / 2.0f };
    gb["g_LightAmbientColor"]  = scene.ambientColor;
    gb["g_LightSkylightColor"] = scene.skylightColor;
    gb["g_NormalModelMatrix"]  = ShaderValue::fromMatrix(Matrix4f::Identity());

    if (context.ortho_w > 0 && context.ortho_h > 0) {
        gb["g_TexelSize"]     = std::array { 1.0f / static_cast<float>(context.ortho_w),
                                             1.0f / static_cast<float>(context.ortho_h) };
        gb["g_TexelSizeHalf"] = std::array { 1.0f / static_cast<float>(context.ortho_w) / 2.0f,
                                             1.0f / static_cast<float>(context.ortho_h) / 2.0f };
    }
}

void CollectSceneNodeRefs(const std::shared_ptr<SceneNode>&                           node,
                          std::unordered_map<SceneNode*, std::shared_ptr<SceneNode>>& refs) {
    if (! node) return;
    refs[node.get()] = node;
    for (const auto& child : node->GetChildren()) {
        CollectSceneNodeRefs(child, refs);
    }
}

int32_t AllocateDynamicLayerId(const Scene& scene) {
    int32_t max_id = 0;
    for (const auto layer_id : scene.layerOrder) {
        max_id = std::max(max_id, layer_id);
    }
    for (const auto& [layer_id, _] : scene.layerNodes) {
        (void)_;
        max_id = std::max(max_id, layer_id);
    }
    for (const auto& [layer_id, _] : scene.objectRuntimeNodes) {
        (void)_;
        max_id = std::max(max_id, layer_id);
    }
    return max_id + 1;
}

// mapRate < 1.0
void GenCardMesh(SceneMesh& mesh, const std::array<uint16_t, 2> size,
                 const std::array<float, 2> mapRate = { 1.0f, 1.0f },
                 bool centered = true) {
    // Official 0x1401ede30: flags bit0 set → ±extent/2; bit0 clear → 0..extent
    // (0x1401eb19f flags=0 for +0x2d8). {2,2}+centered is +0x2e0 ±1 (0x14020679c).
    float left, right, bottom, top;
    if (centered) {
        left   = -(size[0] / 2.0f);
        right  = size[0] / 2.0f;
        bottom = -(size[1] / 2.0f);
        top    = size[1] / 2.0f;
    } else {
        left   = 0.0f;
        right  = static_cast<float>(size[0]);
        bottom = 0.0f;
        top    = static_cast<float>(size[1]);
    }
    float z = 0.0f;

    float tw = mapRate[0], th = mapRate[1];

    // clang-format off
	const std::array pos = {
		left, bottom, z,
		left,  top, z,
		right, bottom, z,
		right,  top, z,
	};
	const std::array texCoord = {
		0.0f, th,
		0.0f, 0.0f,
		tw, th,
		tw, 0.0f,
	};
    // clang-format on

    SceneVertexArray vertex(
        {
            { WE_IN_POSITION.data(), VertexType::FLOAT3 },
            { WE_IN_TEXCOORD.data(), VertexType::FLOAT2 },
        },
        4);
    vertex.SetVertex(WE_IN_POSITION, pos);
    vertex.SetVertex(WE_IN_TEXCOORD, texCoord);
    mesh.AddVertexArray(std::move(vertex));
}

bool ReadNamedVec2Constant(const std::unordered_map<std::string, std::vector<float>>& values,
                           const char* name, std::array<float, 2>& out) {
    const auto it = values.find(name);
    if (it == values.end() || it->second.size() < 2) return false;
    out[0] = it->second[0];
    out[1] = it->second[1];
    return true;
}

bool LoadShapePerspectivePoints(const wpscene::WPImageObject& image,
                                std::array<std::array<float, 2>, 4>& points) {
    // Official shape vt+0x110 0x140260190 looks up point0..point3 on the first
    // effect (0x14026025e / 0x140260383). Any miss jumps to 0x14026072d, the
    // default +0x2e8 size card. Image vt+0x110 is ret (0x14000ec30).
    if (image.image != "__hanabi_shape_directdraw") return false;
    auto try_map = [&](const std::unordered_map<std::string, std::vector<float>>& values) {
        return ReadNamedVec2Constant(values, "point0", points[0]) &&
               ReadNamedVec2Constant(values, "point1", points[1]) &&
               ReadNamedVec2Constant(values, "point2", points[2]) &&
               ReadNamedVec2Constant(values, "point3", points[3]);
    };
    for (const auto& effect : image.effects) {
        for (const auto& pass : effect.passes) {
            if (try_map(pass.constantshadervalues)) return true;
        }
        for (const auto& material : effect.materials) {
            if (try_map(material.constantshadervalues)) return true;
        }
    }
    return false;
}

void GenOfficialShapePerspectiveDestMesh(SceneMesh& mesh, const std::array<float, 2>& size,
                                         const std::array<std::array<float, 2>, 4>& points) {
    // Official 0x140260441–0x140260644. xmm11/xmm13 are +0x2f0/+0x2f4. 0x1404926c0
    // is 0.5, 0x140492778 is 1.0 double. Vertex i: x=w*u-w/2, y=h*(1-v)-h/2, z=0,
    // uv=(u,v). Stored at +0x2e8. Dest then uses dest*VP without S(w/2) because
    // the mesh is already sized (TREE unit_card check / official r15!=0).
    const float w      = std::max(1.0f, size[0]);
    const float h      = std::max(1.0f, size[1]);
    const float half_w = w * 0.5f;
    const float half_h = h * 0.5f;
    std::array<float, 12> pos {};
    std::array<float, 8>  texCoord {};
    for (int i = 0; i < 4; ++i) {
        const float u = points[static_cast<std::size_t>(i)][0];
        const float v = points[static_cast<std::size_t>(i)][1];
        pos[static_cast<std::size_t>(i) * 3 + 0] = w * u - half_w;
        pos[static_cast<std::size_t>(i) * 3 + 1] = h * (1.0f - v) - half_h;
        pos[static_cast<std::size_t>(i) * 3 + 2] = 0.0f;
        texCoord[static_cast<std::size_t>(i) * 2 + 0] = u;
        texCoord[static_cast<std::size_t>(i) * 2 + 1] = v;
    }

    SceneVertexArray vertex(
        {
            { WE_IN_POSITION.data(), VertexType::FLOAT3 },
            { WE_IN_TEXCOORD.data(), VertexType::FLOAT2 },
        },
        4);
    vertex.SetVertex(WE_IN_POSITION, pos);
    vertex.SetVertex(WE_IN_TEXCOORD, texCoord);
    mesh.AddVertexArray(std::move(vertex));
    // Official dest create 0x1402605f9 r9d=4 verts, 0x140260630 [rsp+0x28]=6.
    // The same slot is the generated index count at 0x14026135d, advanced by 6
    // per quad (0x140261327). Verts are point0..3 in cyclic order, so the two
    // triangles are 0,1,2 and 0,2,3. A 4-vert strip in that order leaves
    // (0,2,3) unfilled.
    const uint16_t dest_indices[] = { 0, 1, 2, 0, 2, 3 };
    SceneIndexArray dest_index_array((sizeof(dest_indices) / sizeof(dest_indices[0]) + 1) / 2);
    dest_index_array.AssignHalf(
        0, std::span<const uint16_t>(dest_indices, sizeof(dest_indices) / sizeof(dest_indices[0])));
    mesh.AddIndexArray(std::move(dest_index_array));
}

void AppendPuppetSurfaceWriter(wpscene::WPImageObject& image, const WPMdl& mdl) {
    // Official I-internal binds [instance+0x400] when +0x4b0 (0x14020805e). TREE keeps
    // that skinned draw as a private layer-surface pass after authored ping-pong, then
    // dest blit publishes the surface RT (0x1401ec6b2). Do not fold this mesh into dest.
    wpscene::WPImageEffect puppet_effect;
    wpscene::WPMaterial    puppet_mat = image.material;
    if (puppet_mat.textures.empty()) puppet_mat.textures.resize(1);
    puppet_mat.textures[0] = "";
    WPMdlParser::AddPuppetMatInfo(puppet_mat, mdl);
    puppet_effect.materials.push_back(std::move(puppet_mat));
    image.effects.push_back(std::move(puppet_effect));
}

void GenCardMeshWithTexCoordBounds(SceneMesh& mesh, const std::array<float, 2>& size,
                                   const std::array<float, 4>& texcoord_bounds) {
    const float width  = std::max(1.0f, size[0]);
    const float height = std::max(1.0f, size[1]);
    const float min_u  = texcoord_bounds[0];
    const float min_v  = texcoord_bounds[1];
    const float max_u  = texcoord_bounds[2];
    const float max_v  = texcoord_bounds[3];
    const float z      = 0.0f;

    auto local_x = [width](float u) {
        return (u - 0.5f) * width;
    };
    auto local_y = [height](float v) {
        return (0.5f - v) * height;
    };

    // The final writer may need to cover UVs outside [0, 1], but the effect shader still expects
    // its authored domain to be the original layer size. Expanding positions from UV bounds keeps
    // those two contracts independent: shader math stays stable, while the final quad no longer
    // clips generated DIRECTDRAW pixels at the canonical card edge.
    const std::array pos = {
        local_x(min_u), local_y(max_v), z,
        local_x(min_u), local_y(min_v), z,
        local_x(max_u), local_y(max_v), z,
        local_x(max_u), local_y(min_v), z,
    };
    const std::array texCoord = {
        min_u, max_v,
        min_u, min_v,
        max_u, max_v,
        max_u, min_v,
    };

    SceneVertexArray vertex(
        {
            { WE_IN_POSITION.data(), VertexType::FLOAT3 },
            { WE_IN_TEXCOORD.data(), VertexType::FLOAT2 },
        },
        4);
    vertex.SetVertex(WE_IN_POSITION, pos);
    vertex.SetVertex(WE_IN_TEXCOORD, texCoord);
    mesh.AddVertexArray(std::move(vertex));
}

void ConfigureParticleMesh(SceneMesh& mesh, uint32_t count, const ParticleRenderPlan& plan) {
    if (plan.IsRope()) {
        std::vector<SceneVertexArray::SceneVertexAttribute> attrs {
            { WE_IN_POSITIONVEC4.data(), VertexType::FLOAT4 },
            { WE_IN_TEXCOORDVEC4.data(), VertexType::FLOAT4 },
            { WE_IN_TEXCOORDVEC4C1.data(), VertexType::FLOAT4 },
        };
        if (plan.thick_format) {
            attrs.push_back({ WE_IN_TEXCOORDVEC4C2.data(), VertexType::FLOAT4 });
            attrs.push_back({ WE_IN_TEXCOORDVEC4C3.data(), VertexType::FLOAT4 });
        } else {
            attrs.push_back({ WE_IN_TEXCOORDVEC3C2.data(), VertexType::FLOAT3 });
        }
        attrs.push_back({ WE_IN_COLOR.data(), VertexType::FLOAT4 });

        // Rope renderers always submit one point per logical segment. Their stock geometry stage
        // owns spline expansion, so an index buffer or CPU-generated ribbon would describe a
        // different shader ABI than the one resolved in ParticleRenderPlan.
        mesh.AddVertexArray(SceneVertexArray(attrs, count));
        mesh.SetPrimitive(MeshPrimitive::POINT);
        return;
    }

    std::vector<SceneVertexArray::SceneVertexAttribute> attrs {
        { WE_IN_POSITION.data(), VertexType::FLOAT3 },
        { WE_IN_TEXCOORDVEC4.data(), VertexType::FLOAT4 },
        { WE_IN_COLOR.data(), VertexType::FLOAT4 },
    };
    if (plan.thick_format) {
        attrs.push_back({ WE_IN_TEXCOORDVEC4C1.data(), VertexType::FLOAT4 });
    }

    if (plan.expansion == ParticleExpansionMode::GeometryPoint) {
        mesh.AddVertexArray(SceneVertexArray(attrs, count));
        mesh.SetPrimitive(MeshPrimitive::POINT);
        return;
    }

    // Authored particle materials without a geometry stage consume four vertex invocations per
    // live particle. The extra attribute carries the authored rotation pair, while TexCoordVec4.xy
    // carries the individual quad corner expected by the original vertex shader.
    attrs.push_back({ WE_IN_TEXCOORDC2.data(), VertexType::FLOAT2 });
    mesh.AddVertexArray(
        SceneVertexArray(attrs, static_cast<size_t>(count) * static_cast<size_t>(4)));
    mesh.AddIndexArray(SceneIndexArray(count));
    mesh.SetPrimitive(MeshPrimitive::TRIANGLE);
}

constexpr u32 kMaxParticleCount = 20000;
constexpr i32 kMaxParticleTrailSegments = 256;

struct ParticleRendererSpec {
    std::string_view         name;
    ParticleRendererKind     kind;
    ParticleShaderSelection shader_selection;
    bool                     thick_format;

    constexpr bool IsRope() const noexcept {
        return kind == ParticleRendererKind::Rope || kind == ParticleRendererKind::RopeTrail;
    }
    constexpr bool IsTrail() const noexcept {
        return kind == ParticleRendererKind::SpriteTrail ||
            kind == ParticleRendererKind::RopeTrail;
    }
    constexpr bool UsesHistory() const noexcept {
        return kind == ParticleRendererKind::RopeTrail;
    }
};

constexpr std::array kParticleRendererSpecs {
    ParticleRendererSpec { "sprite",
                           ParticleRendererKind::Sprite,
                           ParticleShaderSelection::Authored,
                           false },
    ParticleRendererSpec { "spritetrail",
                           ParticleRendererKind::SpriteTrail,
                           ParticleShaderSelection::Authored,
                           true },
    ParticleRendererSpec { "rope",
                           ParticleRendererKind::Rope,
                           ParticleShaderSelection::StockRope,
                           true },
    ParticleRendererSpec { "ropetrail",
                           ParticleRendererKind::RopeTrail,
                           ParticleShaderSelection::StockRope,
                           false },
};

std::optional<ParticleRendererSpec> DescribeParticleRender(std::string_view renderer_name) {
    // Renderer classification is an ABI decision: it controls shader stages, mesh layout, CPU
    // extraction, and history ownership. Exact names keep rope and ropetrail from drifting into
    // the same runtime path merely because they share a prefix.
    for (const auto& spec : kParticleRendererSpecs) {
        if (spec.name == renderer_name) return spec;
    }
    return std::nullopt;
}

std::optional<u32> CheckedParticleMeshCapacity(std::string_view object_name,
                                               std::string_view renderer_name,
                                               std::initializer_list<uint64_t> factors) {
    uint64_t capacity { 1 };
    for (const uint64_t factor : factors) {
        if (factor != 0 && capacity > std::numeric_limits<uint64_t>::max() / factor) {
            LOG_ERROR("particle mesh capacity overflow object='%.*s' renderer='%.*s'",
                      static_cast<int>(object_name.size()),
                      object_name.data(),
                      static_cast<int>(renderer_name.size()),
                      renderer_name.data());
            return std::nullopt;
        }
        capacity *= factor;
    }
    if (capacity > std::numeric_limits<u32>::max()) {
        LOG_ERROR("particle mesh capacity exceeds uint32 object='%.*s' renderer='%.*s' "
                  "capacity=%llu",
                  static_cast<int>(object_name.size()),
                  object_name.data(),
                  static_cast<int>(renderer_name.size()),
                  renderer_name.data(),
                  static_cast<unsigned long long>(capacity));
        return std::nullopt;
    }
    return static_cast<u32>(capacity);
}

ParticleAnimationMode ToAnimMode(const std::string& str) {
    if (str == "randomframe")
        return ParticleAnimationMode::RANDOMONE;
    else if (str == "sequence")
        return ParticleAnimationMode::SEQUENCE;
    else {
        return ParticleAnimationMode::SEQUENCE;
    }
}

void ApplyLayerControlPointOverrides(ParticleSubSystem&                       pSys,
                                     const wpscene::ParticleInstanceoverride& over) {
    std::span<ParticleControlpoint> pcs = pSys.Controlpoints();
    const usize override_count          = std::min(pcs.size(), over.controlpointOffsets.size());
    for (usize i = 0; i < override_count; i++) {
        if (! over.controlpointOffsets[i].has_value()) continue;

        // Instance overrides are authored at the scene layer level, after the particle asset has
        // supplied default control point flags. Only the offset is replaced here so link_mouse and
        // worldspace semantics continue to come from the particle asset definition.
        pcs[i].base_offset =
            Eigen::Vector3d { array_cast<double>(*over.controlpointOffsets[i]).data() };
        pcs[i].offset = pcs[i].base_offset;
    }
}

void LoadControlPoint(ParticleSubSystem& pSys, const wpscene::Particle& wp,
                      const wpscene::ParticleInstanceoverride& over) {
    std::span<ParticleControlpoint> pcs = pSys.Controlpoints();
    usize                           s   = std::min(pcs.size(), wp.controlpoints.size());
    for (usize i = 0; i < s; i++) {
        pcs[i].base_offset =
            Eigen::Vector3d { array_cast<double>(wp.controlpoints[i].offset).data() };
        pcs[i].offset = pcs[i].base_offset;
        pcs[i].link_mouse =
            wp.controlpoints[i].flags[wpscene::ParticleControlpoint::FlagEnum::link_mouse];
        pcs[i].worldspace =
            wp.controlpoints[i].flags[wpscene::ParticleControlpoint::FlagEnum::worldspace];
    }
    ApplyLayerControlPointOverrides(pSys, over);
}

// A scene layer's instanceoverride belongs to the particle asset instantiated directly by that
// layer. Nested particle assets are separate authored systems: their own colorrandom ranges must
// remain intact so a root-layer colorn value does not flatten every child gradient (for example,
// the Matrix rain's blue glyph stream) before the scene compositor applies its parent effects.
wpscene::ParticleInstanceoverride ResolveParticleSubsystemOverride(
    const wpscene::ParticleInstanceoverride& layer_override, bool is_child_subsystem) {
    if (! is_child_subsystem) return layer_override;

    auto child_override = layer_override;
    child_override.overColor  = false;
    child_override.overColorn = false;
    return child_override;
}

ParticleAudioResponseParams ReadParticleAudioResponse(const nlohmann::json& json) {
    /*
     * Keep audio-response parsing in one helper so emitter, initializer, and operator paths share
     * the same defaults, endpoint clamping, and reversed frequency-interval handling.
     */
    ParticleAudioResponseParams response;
    GET_JSON_NAME_VALUE_NOWARN(json, "audioprocessingmode", response.mode);
    GET_JSON_NAME_VALUE_NOWARN(json, "audioprocessingexponent", response.exponent);
    if (json.contains("audioprocessingbounds") && json.at("audioprocessingbounds").is_string()) {
        AssignAudioBoundsFromAuthoredString(json.at("audioprocessingbounds").get<std::string>(),
                                            response.bounds);
    } else {
        GET_JSON_NAME_VALUE_NOWARN(json, "audioprocessingbounds", response.bounds);
    }
    GET_JSON_NAME_VALUE_NOWARN(json, "audioprocessingfrequencystart", response.frequency_start);
    GET_JSON_NAME_VALUE_NOWARN(json, "audioprocessingfrequencyend", response.frequency_end);
    response.frequency_start = std::min<uint32_t>(response.frequency_start, 15);
    response.frequency_end   = std::min<uint32_t>(response.frequency_end, 15);
    if (response.frequency_end < response.frequency_start)
        std::swap(response.frequency_start, response.frequency_end);
    return response;
}

ParticleAudioResponseFactor BindParticleAudioResponse(ParticleSystem& system,
                                                      ParticleAudioResponseParams response) {
    if (response.mode == 0) return {};

    system.MarkNeedsAudioSpectrum();
    ParticleSystem* system_ptr = &system;
    return [system_ptr, response]() { return system_ptr->AudioResponseFactor(response); };
}

void LoadInitializer(ParticleSubSystem& pSys, const wpscene::Particle& wp,
                     const wpscene::ParticleInstanceoverride& over, ParticleSystem& system) {
    for (const auto& ini : wp.initializers) {
        // A layer color override depends on colorrandom's authored endpoints and their RGB midpoint.
        // Wallpaper Engine shifts those endpoints by the override/reference delta in HSV space and
        // performs the existing per-particle interpolation afterward, so the initializer must remain
        // in the stream instead of being replaced by the requested layer color.
        ParticleAudioResponseFactor audio_factor =
            BindParticleAudioResponse(system, ReadParticleAudioResponse(ini));
        pSys.AddInitializer(WPParticleParser::genParticleInitOp(ini, std::move(audio_factor)));
    }
    if (over.enabled) pSys.AddInitializer(WPParticleParser::genOverrideInitOp(over));
}
void LoadOperator(ParticleSystem& system, ParticleSubSystem& pSys, const wpscene::Particle& wp,
                  const wpscene::ParticleInstanceoverride& over) {
    for (const auto& op : wp.operators) {
        ParticleAudioResponseFactor audio_factor =
            BindParticleAudioResponse(system, ReadParticleAudioResponse(op));
        pSys.AddOperator(WPParticleParser::genParticleOperatorOp(op, over, std::move(audio_factor)));
    }
}
void LoadEmitter(ParticleSystem& system, ParticleSubSystem& pSys, const wpscene::Particle& wp,
                 float count) {
    for (const auto& em : wp.emitters) {
        auto newEm = em;
        newEm.rate *= count;
        // newEm.origin[2] -= perspectiveZ;

        /*
         * Wallpaper Engine's audio-responsive emitters ("Custom BEAT Particle" style assets) are
         * only active while audio is playing: the authored rate is scaled by the live loudness of
         * the selected channel through the authored bounds window. Without this factor such
         * emitters run at full authored rate forever; assets author beat-burst rates in the
         * thousands per second, which floods the simulation with tens of thousands of permanent
         * particles and dominates both CPU and GPU frame time.
         */
        ParticleAudioResponseFactor audio_rate_factor =
            BindParticleAudioResponse(system, em.audio_response);
        ParticleEmitterTiming timing;
        auto op = WPParticleParser::genParticleEmittOp(newEm, std::move(audio_rate_factor), timing);
        pSys.AddEmitter(std::move(op), std::move(timing));
    }
}

ParticleSubSystem::SpawnType ParseSpawnType(std::string_view str) {
    using ST = ParticleSubSystem::SpawnType;
    ST type { ST::STATIC };
    if (str == "eventfollow") {
        type = ST::EVENT_FOLLOW;
    } else if (str == "eventspawn") {
        type = ST::EVENT_SPAWN;
    } else if (str == "eventdeath") {
        type = ST::EVENT_DEATH;
    }
    return type;
};

BlendMode ParseBlendMode(std::string_view str) {
    BlendMode bm;
    if (str == "translucent") {
        bm = BlendMode::Translucent;
    } else if (str == "additive") {
        bm = BlendMode::Additive;
    } else if (str == "normal") {
        bm = BlendMode::Normal;
    } else if (str == "alphatocoverage") {
        bm = BlendMode::AlphaToCoverage;
    } else if (str == "disabled") {
        // seems disabled is normal
        bm = BlendMode::Normal;
    } else {
        bm = BlendMode::Normal;
        LOG_ERROR("unknown blending: %s", str.data());
    }
    return bm;
}

void ParseSpecTexName(std::string& name, const wpscene::WPMaterial& wpmat, const Scene* scene,
                      const WPShaderInfo& sinfo) {
    if (IsSpecTex(name)) {
        if (name == "_rt_FullFrameBuffer") {
            name = SpecTex_Default;
            if (wpmat.shader == "genericimage2" && ! exists(sinfo.combos, "BLENDMODE")) name = "";
            /*
            if(wpmat.shader == "genericparticle") {
                name = "_rt_ParticleRefract";
            }
            */
        } else if (sstart_with(name, WE_IMAGE_LAYER_COMPOSITE_PREFIX)) {
            LOG_INFO("link tex \"%s\"", name.c_str());
            int         wpid { -1 };
            std::regex  reImgId { R"(_rt_imageLayerComposite_([0-9]+))" };
            std::smatch match;
            if (std::regex_search(name, match, reImgId)) {
                STRTONUM(std::string(match[1]), wpid);
            }
            name = GenLinkTex((u32)wpid);
        } else if (name == SpecTex_DefaultPingPong) {
        } else if (sstart_with(name, WE_MIP_MAPPED_FRAME_BUFFER)) {
        } else if (sstart_with(name, WE_EFFECT_PPONG_PREFIX)) {
        } else if (sstart_with(name, WE_HALF_COMPO_BUFFER_PREFIX)) {
        } else if (sstart_with(name, WE_QUARTER_COMPO_BUFFER_PREFIX)) {
        } else if (sstart_with(name, WE_FULL_COMPO_BUFFER_PREFIX)) {
        } else if (name == "_rt_shadowAtlas") {
        } else if (name == kModelReflectionTargetName) {
            // 3D model reflection targets are registered lazily by model materialization, but
            // first-party grid materials may reference the sampler while their own model is not
            // reflected. Accept the name here as a model/runtime target instead of treating it as a
            // missing 2D texture.
        } else if (scene != nullptr && scene->renderTargets.count(name) != 0) {
            // Effect-local feedback buffers such as `_rt_EightBuffer1_<effect-layer-address>` are
            // registered dynamically from the authored FBO table. They still use Wallpaper Engine's
            // `_rt_` prefix, so the generic special-texture parser sees them, but the scene render
            // target table is the authoritative contract for whether they are valid runtime FBOs.
        } else {
            LOG_ERROR("unknown tex \"%s\"", name.c_str());
        }
    }
}

void ApplyKnownShaderSourceFixes(std::string_view shader_name, ShaderType stage,
                                 std::string& source) {
    // Wallpaper Engine's stock `genericropeparticle.vert` expands the ribbon with
    // `position += right * uvs.x * 2.0 - 1.0;`, which offsets the whole rope instead
    // of scaling `right` symmetrically around the centerline.
    if (shader_name == "genericropeparticle" && stage == ShaderType::VERTEX) {
        const std::string_view broken = "position += right * uvs.x * 2.0 - 1.0;";
        const auto             pos    = source.find(broken);
        if (pos != std::string::npos) {
            source.replace(pos, broken.size(), "position += right * (uvs.x * 2.0 - 1.0);");
        }
    }

    // Stock volumetricsback.frag is empty. The engine writes window Z into a
    // depth RT and volumetricsfront samples it with texSample2DBackBuffer.
    // This renderer exposes RTs as color images, so publish the same gl_FragCoord.z into .r.
    // The Steam copy is CRLF (`void main() {\r\n}`), so a LF-only search never matched and the
    // empty main left glOutColor at 0 — every ray then had length 0.
    if (shader_name == "volumetricsback" && stage == ShaderType::FRAGMENT &&
        source.find("gl_FragCoord") == std::string::npos) {
        source = "void main() {\n"
                 "\tgl_FragColor = vec4(gl_FragCoord.z, gl_FragCoord.z, gl_FragCoord.z, 1.0);\n"
                 "}\n";
        LOG_INFO("SceneVolumetrics: rewrote volumetricsback.frag to publish window Z");
    }

}

bool IsMaterialRuntimeRenderTarget(const Scene* scene, const std::string& name) {
    // Wallpaper Engine effect FBOs are runtime render targets even when their authored names do not
    // use the `_rt_` prefix. Checking the scene table keeps names like `blur_start_2_<addr>` on the
    // render-target path instead of probing `/assets/materials/<name>.tex` and logging false VFS
    // errors.
    return scene != nullptr && scene->renderTargets.count(name) != 0;
}

void RegisterSceneTextureFromHeader(Scene& scene, const std::string& name,
                                    const ImageHeader& header) {
    if (scene.textures.count(name) != 0) return;

    SceneTexture texture;
    texture.sample    = header.sample;
    texture.url       = name;
    texture.format    = header.format;
    texture.isVideo   = header.isVideoTexture;
    texture.width     = header.width;
    texture.height    = header.height;
    texture.mapWidth  = header.mapWidth;
    texture.mapHeight = header.mapHeight;
    texture.mipmapCount   = header.mipmapCount;
    texture.mipmap_larger = header.mipmap_larger;
    if (header.isSprite) {
        texture.isSprite   = true;
        texture.spriteAnim = header.spriteAnim;
    }
    scene.textures[name] = std::move(texture);
}

enum class GeometryStagePolicy {
    Disabled,
    MatchMaterial,
    Required,
};

struct MaterialLoadResult {
    bool geometry_stage_loaded { false };
};

std::optional<MaterialLoadResult>
LoadMaterial(fs::VFS& vfs, const wpscene::WPMaterial& wpmat, Scene* pScene, SceneNode* pNode,
             SceneMaterial* pMaterial, WPShaderValueData* pSvData,
             const UserPropertyMap* user_properties = nullptr,
             WPShaderInfo*          pWPShaderInfo   = nullptr,
             GeometryStagePolicy geometry_stage = GeometryStagePolicy::Disabled) {
    bool geometry_stage_loaded { false };

    auto& svData   = *pSvData;
    auto& material = *pMaterial;

    std::unique_ptr<WPShaderInfo> upWPShaderInfo(nullptr);
    if (pWPShaderInfo == nullptr) {
        upWPShaderInfo = std::make_unique<WPShaderInfo>();
        pWPShaderInfo  = upWPShaderInfo.get();
    }

    SceneMaterialCustomShader materialShader;

    auto& shader = materialShader.shader;
    shader       = std::make_shared<SceneShader>();
    shader->name = wpmat.shader;

    std::string shaderPath("/assets/shaders/" + wpmat.shader);

    std::vector<WPShaderUnit> sd_units;
    sd_units.push_back(WPShaderUnit {
        .stage           = ShaderType::VERTEX,
        .src             = fs::GetFileContent(vfs, shaderPath + ".vert"),
        .preprocess_info = {},
        .debug_name      = wpmat.shader + ".vert",
    });
    if (geometry_stage != GeometryStagePolicy::Disabled) {
        const auto geometry_path = shaderPath + ".geom";
        const bool geometry_source_exists = vfs.Contains(geometry_path);
        if (! geometry_source_exists && geometry_stage == GeometryStagePolicy::Required) {
            LOG_ERROR("material '%s' required geometry shader source missing shader='%s' path='%s'",
                      wpmat.shader.c_str(),
                      wpmat.shader.c_str(),
                      geometry_path.c_str());
            return std::nullopt;
        }
        if (geometry_source_exists) {
            sd_units.push_back(WPShaderUnit {
                .stage           = ShaderType::GEOMETRY,
                .src             = fs::GetFileContent(vfs, geometry_path),
                .preprocess_info = {},
                .debug_name      = wpmat.shader + ".geom",
            });
            // GS_ENABLED describes a stage that was successfully materialized. It is deliberately
            // set after the VFS check so combo state and mesh topology cannot claim a geometry ABI
            // that the selected material does not actually provide.
            pWPShaderInfo->combos["GS_ENABLED"] = "1";
            geometry_stage_loaded = true;
        }
    }
    sd_units.push_back(WPShaderUnit {
        .stage           = ShaderType::FRAGMENT,
        .src             = fs::GetFileContent(vfs, shaderPath + ".frag"),
        .preprocess_info = {},
        .debug_name      = wpmat.shader + ".frag",
    });

    for (auto& unit : sd_units) {
        ApplyKnownShaderSourceFixes(wpmat.shader, unit.stage, unit.src);
    }

    bool lighting_v1 { false };
    if (pScene != nullptr) {
        const auto lighting_desc = LightingDescFromScene(*pScene);
        for (auto& unit : sd_units) {
            if (ExpandRequireLightingV1(unit.src, lighting_desc)) lighting_v1 = true;
        }
    }

    auto textures = wpmat.textures;
    if (wpmat.usertextures.size() > textures.size()) {
        textures.resize(wpmat.usertextures.size());
    }
    for (usize i = 0; i < wpmat.usertextures.size(); i++) {
        const auto& binding = wpmat.usertextures[i];
        if (binding.empty()) continue;
        if (binding.type == "system") {
            if (binding.name == "$mediaThumbnail") {
                textures[i] = std::string(WP_SCENE_SCRIPT_MEDIA_THUMBNAIL_TEXTURE);
                EnsureSystemTextureRegistered(*pScene, WP_SCENE_SCRIPT_MEDIA_THUMBNAIL_TEXTURE);
            } else if (binding.name == "$mediaPreviousThumbnail") {
                textures[i] = std::string(WP_SCENE_SCRIPT_MEDIA_PREVIOUS_THUMBNAIL_TEXTURE);
                EnsureSystemTextureRegistered(*pScene,
                                              WP_SCENE_SCRIPT_MEDIA_PREVIOUS_THUMBNAIL_TEXTURE);
            }
            continue;
        }

        const auto* property = LookupUserPropertyString(user_properties, binding.name);
        if (property == nullptr || property->empty()) continue;

        textures[i] = *property;
    }

    std::vector<WPShaderTexInfo>                 texinfos;
    std::unordered_map<std::string, ImageHeader> texHeaders;
    // The shader parser uses this effective texture list to decide whether a texture-driven combo
    // may expose a sampler branch. User texture bindings must be resolved before this point so a
    // real selected texture still enables its combo, while an empty optional mask slot stays off.
    for (const auto& el : textures) {
        if (el.empty()) {
            texinfos.push_back({ false });
        } else if (! IsSpecTex(el) && ! IsMaterialRuntimeRenderTarget(pScene, el)) {
            const auto& texh = pScene->imageParser->ParseHeader(el);
            texHeaders[el]   = texh;
            if (texh.extraHeader.count("compo1") == 0) {
                texinfos.push_back({ false });
                continue;
            }
            texinfos.push_back({ true,
                                 {
                                     (bool)texh.extraHeader.at("compo1").val,
                                     (bool)texh.extraHeader.at("compo2").val,
                                     (bool)texh.extraHeader.at("compo3").val,
                                 } });
        } else {
            // Runtime render targets should expose sampler metadata to the shader preprocessor just
            // like `_rt_` textures. Their exact dimensions are resolved below from
            // SceneRenderTarget, so no material header lookup is needed here.
            WPShaderTexInfo texinfo { .enabled = true };
            if (pScene != nullptr) {
                const auto rt_it = pScene->renderTargets.find(el);
                if (rt_it != pScene->renderTargets.end()) {
                    // Render-target sampling conventions travel with the bound texture slot. This
                    // keeps screen-space reflection correction generic and avoids changing the
                    // producer pass viewport, which would also affect geometry and culling.
                    texinfo.screenSpaceSampleYFlip = rt_it->second.screenSpaceSampleYFlip;
                }
            }
            texinfos.push_back(texinfo);
        }
    }

    for (auto& unit : sd_units) {
        unit.src = WPShaderParser::PreShaderSrc(vfs, unit.src, pWPShaderInfo, texinfos);
    }

    shader->default_uniforms = pWPShaderInfo->svs;

    for (const auto& el : wpmat.combos) {
        pWPShaderInfo->combos[el.first] = std::to_string(el.second);
    }

    if (lighting_v1 && pScene != nullptr) {
        if (pScene->shadows.quality != 0 && SceneHasShadowLights(*pScene)) {
            pWPShaderInfo->combos["LIGHTS_SHADOW_MAPPING"]         = "1";
            pWPShaderInfo->combos["LIGHTS_SHADOW_MAPPING_QUALITY"] =
                std::to_string(pScene->shadows.quality);
        }
    }

    // ALPHATOCOVERAGE is a compile-time combo injected when the material blending
    // mode is alphatocoverage. Live msaa changes do not recompile materials;
    // rasterizer A2C is a separate draw-state bit and follows the current sample
    // count. Shaders may contain an unused #if ALPHATOCOVERAGE block without
    // this blending mode; those stay combo-off so 2D translucent layers are not
    // coverage-tested.
    const bool blending_alpha_to_coverage = wpmat.blending == "alphatocoverage";
    if (blending_alpha_to_coverage) {
        pWPShaderInfo->combos["ALPHATOCOVERAGE"] = "1";
    }

    if (pWPShaderInfo->defTexs.size() > 0) {
        for (auto& t : pWPShaderInfo->defTexs) {
            if (textures.size() > t.first) {
                if (! textures.at(t.first).empty()) continue;
            } else {
                textures.resize(t.first + 1);
            }
            textures[t.first] = t.second;
        }
    }

    for (usize i = 0; i < textures.size(); i++) {
        std::string name = textures.at(i);
        if (name == "_alias_lightCookie") name = "cookie/flashlight1";
        ParseSpecTexName(name, wpmat, pScene, *pWPShaderInfo);
        material.textures.push_back(name);
        material.defines.push_back("g_Texture" + std::to_string(i));
        if (name.empty()) {
            continue;
        }

        std::array<i32, 4> resolution {};
        if (IsSpecTex(name) || IsMaterialRuntimeRenderTarget(pScene, name)) {
            if (IsSpecLinkTex(name)) {
                svData.renderTargets.push_back({ i, name });
            } else if (pScene->renderTargets.count(name) == 0) {
                LOG_ERROR("%s not found in render targes", name.c_str());
            } else {
                svData.renderTargets.push_back({ i, name });
                const auto& rt = pScene->renderTargets.at(name);
                // Runtime render targets may keep a larger physical allocation than the logical
                // content they currently store. Forwarding the authored content extent through
                // `.zw` preserves Wallpaper Engine's original "sample area" contract for effects
                // that distinguish between allocated size and meaningful image size.
                resolution = rt.ResolutionVector();
            }
        } else {
            const ImageHeader& texh = texHeaders.count(name) == 0
                                          ? pScene->imageParser->ParseHeader(name)
                                          : texHeaders.at(name);
            SetTextureFormatShaderDefine(*pWPShaderInfo, i, texh.format);
            if (texh.mipmap_larger) {
                resolution = { texh.width, texh.height, texh.mapWidth, texh.mapHeight };
            } else {
                resolution = { texh.mapWidth, texh.mapHeight, texh.mapWidth, texh.mapHeight };
            }
            // Parse-time values use the authored header. After GPU upload the
            // shader updater overwrites g_TextureNResolution from
            // EffectiveImportedTextureResolution() so half/auto follow the
            // official bind-path GPU extent.

            RegisterSceneTextureFromHeader(*pScene, name, texh);
            if ((pScene->textures.at(name)).isSprite) {
                material.hasSprite = true;
                const auto& f1     = texh.spriteAnim.GetCurFrame();
                if (wpmat.shader == "genericparticle" || wpmat.shader == "genericropeparticle") {
                    pWPShaderInfo->combos["SPRITESHEET"] = "1";
                    pWPShaderInfo->combos["THICKFORMAT"] = "1";
                    if (algorism::IsPowOfTwo((u32)texh.width) &&
                        algorism::IsPowOfTwo((u32)texh.height)) {
                        pWPShaderInfo->combos["SPRITESHEETBLENDNPOT"] = "1";
                        resolution = ResolvePaddedSpriteSheetResolution(texh, f1);
                    }
                    materialShader.constValues["g_RenderVar1"] = std::array {
                        f1.xAxis[0], f1.yAxis[1], (float)(texh.spriteAnim.numFrames()), f1.rate
                    };
                }
            }
        }
        if (! resolution.empty()) {
            const std::string gResolution = WE_GLTEX_RESOLUTION_NAMES[i];

            materialShader.constValues[gResolution] = array_cast<float>(resolution);
        }
    }
    if (exists(pWPShaderInfo->combos, "LIGHTING")) {
        // pWPShaderInfo->combos["PRELIGHTING"] =
        // pWPShaderInfo->combos.at("LIGHTING");
    }

    if (! WPShaderParser::CompileToSpv(
            pScene->scene_id, sd_units, shader->codes, vfs, pWPShaderInfo, texinfos)) {
        return std::nullopt;
    }

    material.blenmode = ParseBlendMode(wpmat.blending);
    material.alpha_to_coverage = blending_alpha_to_coverage;

    const auto& fragment_unit = sd_units.back();
    assert(fragment_unit.stage == ShaderType::FRAGMENT);
    for (uint i = 0; i < material.textures.size(); i++) {
        if (! exists(fragment_unit.preprocess_info.active_tex_slots, i)) {
            material.textures[i].clear();
        }
    }

    for (const auto& el : pWPShaderInfo->baseConstSvs) {
        materialShader.constValues[el.first] = el.second;
    }
    material.customShader = materialShader;
    material.name         = wpmat.shader;
    // Store the material-name to GLSL-uniform alias table on the live SceneMaterial. Runtime WE
    // scripts can then write properties such as `thisObject.getMaterial(0).raythreshold` and have
    // the script bridge resolve them to the actual shader uniform (`g_Threshold`) that this parse
    // pass discovered from the shader metadata comments.
    material.uniformAliases = pWPShaderInfo->alias;

    return MaterialLoadResult { .geometry_stage_loaded = geometry_stage_loaded };
}

bool ConfigureEffectFinalComposite(ParseContext& context, SceneImageEffectLayer& effect_layer,
                                   std::string_view initial_source, int32_t owner_layer_id,
                                   std::string_view         owner_name,
                                   int32_t                  color_blend_mode,
                                   const WPShaderValueData* final_transform_data = nullptr) {
    auto& vfs = *context.vfs;

    wpscene::WPMaterial composite_source;
    nlohmann::json      composite_json;
    if (! PARSE_JSON(fs::GetFileContent(vfs, "/assets/materials/util/effectpassthrough.json"),
                     composite_json) ||
        ! composite_source.FromJson(composite_json)) {
        LOG_ERROR(
            "SceneEffectFinalComposite: layer=%d name='%.*s' failed to load passthrough material",
            owner_layer_id,
            static_cast<int>(owner_name.size()),
            owner_name.data());
        return false;
    }

    if (composite_source.textures.empty()) composite_source.textures.resize(1);
    composite_source.textures[0] = std::string(initial_source);
    // Modes 1..30 are framebuffer-aware shader blend equations. Modes 0 and 31 use the neutral
    // shader variant; mode 31 is expressed by the final fixed-function additive state instead.
    composite_source.combos["BLENDMODE"] =
        UsesShaderColorBlendMode(color_blend_mode) ? color_blend_mode : 0;

    WPShaderInfo composite_shader_info;
    composite_shader_info.baseConstSvs = context.global_base_uniforms;
    // The source object/effect chain has already baked authored color, alpha, brightness, and
    // effect output into the ping-pong texture. The final composite must therefore be a neutral
    // sampler: it applies the layer's final mesh and blend state, but it must not tint or fade the
    // resolved texture a second time.
    composite_shader_info.baseConstSvs["g_Color4"] =
        std::array<float, 4> { 1.0f, 1.0f, 1.0f, 1.0f };
    composite_shader_info.baseConstSvs["g_Color"]      = std::array<float, 3> { 1.0f, 1.0f, 1.0f };
    composite_shader_info.baseConstSvs["g_Alpha"]      = 1.0f;
    composite_shader_info.baseConstSvs["g_UserAlpha"]  = 1.0f;
    composite_shader_info.baseConstSvs["g_Brightness"] = 1.0f;

    SceneMaterial     composite_material;
    WPShaderValueData composite_data;
    if (! LoadMaterial(vfs,
                       composite_source,
                       context.scene.get(),
                       &effect_layer.FinalNode(),
                       &composite_material,
                       &composite_data,
                       context.user_properties,
                       &composite_shader_info)) {
        LOG_ERROR("SceneEffectFinalComposite: layer=%d name='%.*s' material compile failed",
                  owner_layer_id,
                  static_cast<int>(owner_name.size()),
                  owner_name.data());
        return false;
    }
    if (final_transform_data != nullptr) {
        // The fallback final composite is the screen-space writer used when the last authored
        // effect in a layer is hidden. It is not an authored child node, so copying the full parent
        // transform binding would multiply route matrices twice; only the parallax contract is
        // mirrored from the visible world node so hidden-effect fallbacks keep moving with their
        // compose/text layer.
        composite_data.CopyParallaxContractFrom(*final_transform_data);
        // Official Path B 0x14018b062 has no attach/final suppress. The visible
        // dest blit after I pop (0x1401e9dd5) always applies ROOT offset.
        composite_data.SetParallaxContract(composite_data.parallaxDepth,
                                           composite_data.parallax_anchor,
                                           false,
                                           composite_data.parallaxDepthAuthored);
    }

    // Leftover dest-draw +0x9b0 is dest*VP*S(w/2) (DEST_MVP 0x1401ec51f). Last-pass
    // g_MVP is +0x930 camera*dest. FinalNode is not 0x1401ebf60 (LEFTOVER_VS_DESTDRAW).
    composite_data.SetEffectTextureProjection(&effect_layer.FinalNode(),
                                              &effect_layer.FinalMesh(),
                                              effect_layer.AuthoredWidth(),
                                              effect_layer.AuthoredHeight());

    auto composite_mesh = std::make_shared<SceneMesh>();
    composite_mesh->AddMaterial(std::move(composite_material));

    auto& final_node = effect_layer.FinalNode();
    final_node.SetName(owner_name.empty()
                           ? std::string("__hanabi_effect_final_composite")
                           : std::string(owner_name) + "::__hanabi_effect_final_composite");
    final_node.ID() = owner_layer_id;
    final_node.SetCamera(std::string());
    final_node.AddMesh(composite_mesh);

    context.shader_updater->SetNodeData(&final_node, composite_data);
    context.scene->nodeOwners[&final_node] = owner_layer_id;
    effect_layer.SetFinalCompositeSource(std::string(initial_source));
    return true;
}

SceneImageEffectLayer::HiddenFinalCompositePolicy
ResolveHiddenFinalCompositePolicy(const wpscene::WPImageObject& image) {
    // This is a source-contract decision rather than a shader or layer-name decision. Normal image
    // layers have a meaningful pre-effect source, so disabling the final effect should reveal that
    // source. Passthrough compose helpers are source-less routing layers; when their final effect is
    // hidden they should publish nothing instead of preserving a helper render target that may only
    // contain previous framebuffer contents.
    return image.config.passthrough
        ? SceneImageEffectLayer::HiddenFinalCompositePolicy::SuppressOutput
        : SceneImageEffectLayer::HiddenFinalCompositePolicy::PreserveSource;
}

SceneImageEffectLayer::SourcePolicy ResolveImageEffectSourcePolicy(
    bool is_compose_layer, const wpscene::WPImageObject& image) {
    if (!is_compose_layer) return SceneImageEffectLayer::SourcePolicy::OwnerNode;
    // copybackground composition layers seed their private source from the owner framebuffer
    // material before routed children are accumulated. Ordinary composition layers instead author
    // their source exclusively through routed children; drawing the owner framebuffer again would
    // composite the already-rendered scene a second time and brighten the complete wallpaper.
    return image.copybackground ? SceneImageEffectLayer::SourcePolicy::OwnerNodeAndProxyChildren
                                : SceneImageEffectLayer::SourcePolicy::ProxyChildrenOnly;
}

std::string_view ImageEffectSourcePolicyName(SceneImageEffectLayer::SourcePolicy policy) {
    switch (policy) {
    case SceneImageEffectLayer::SourcePolicy::OwnerNode: return "owner-node";
    case SceneImageEffectLayer::SourcePolicy::OwnerNodeAndProxyChildren:
        return "owner-node-and-proxy-children";
    case SceneImageEffectLayer::SourcePolicy::ProxyChildrenOnly: return "proxy-children-only";
    }
    return "unknown";
}

struct ImageEffectCameraClipRange {
    float near_clip { -1.0f };
    float far_clip { 1.0f };
};

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

PuppetSurfaceProjection BuildPuppetSurfaceProjection(
    const wpscene::WPImageObject& image,
    const WPMdl& mdl,
    const SceneMesh& puppet_mesh,
    std::string camera_name,
    std::string target_name,
    const std::array<float, 2>& effect_source_size) {
    PuppetSurfaceProjection projection;
    projection.authored_layer_size = { image.size[0], image.size[1] };
    projection.geometry_transform = puppet_mesh.GeometryTransform();
    projection.camera_name = std::move(camera_name);
    projection.target_name = std::move(target_name);
    projection.render_density = std::max(
        effect_source_size[0] / std::max(image.size[0], 1.0f),
        effect_source_size[1] / std::max(image.size[1], 1.0f));
    projection.asset_bounds = mdl.asset_bounds;
    projection.authored_pose_bounds = mdl.authored_pose_bounds.IsFiniteAndOrdered()
                                          ? mdl.authored_pose_bounds
                                          : mdl.asset_bounds;

    PuppetBounds3D authored_surface = TransformPuppetBounds(
        projection.authored_pose_bounds, projection.geometry_transform);
    authored_surface.Include(Eigen::Vector3f {
        -image.size[0] * 0.5f, -image.size[1] * 0.5f, 0.0f
    });
    authored_surface.Include(Eigen::Vector3f {
        image.size[0] * 0.5f, image.size[1] * 0.5f, 0.0f
    });
    // One physical target pixel protects raster-edge precision while keeping the guard tied to the
    // actual render density. The resulting surface remains immutable for authored animation.
    const float guard = 1.0f / std::max(projection.render_density,
                                        std::numeric_limits<float>::epsilon());
    projection.surface_bounds = authored_surface;
    projection.surface_bounds.min -= Eigen::Vector3f::Constant(guard);
    projection.surface_bounds.max += Eigen::Vector3f::Constant(guard);
    return projection;
}

TextureSample ResolvePrimaryMaterialSampler(const Scene& scene, const SceneMaterial& material) {
    if (material.textures.empty() || material.textures.front().empty()) return {};
    const auto& texture_name = material.textures.front();
    if (const auto texture_it = scene.textures.find(texture_name);
        texture_it != scene.textures.end()) {
        return texture_it->second.sample;
    }
    if (const auto target_it = scene.renderTargets.find(texture_name);
        target_it != scene.renderTargets.end()) {
        return target_it->second.sample;
    }
    return {};
}

bool PrimaryMaterialTextureIsSprite(const Scene& scene, const SceneMaterial& material) {
    if (material.textures.empty() || material.textures.front().empty()) return false;
    const auto texture_it = scene.textures.find(material.textures.front());
    return texture_it != scene.textures.end() && texture_it->second.isSprite;
}

ImageEffectCameraClipRange ResolveImageEffectCameraClipRange(bool has_animated_puppet_mesh) {
    if (! has_animated_puppet_mesh) return {};

    // Animated puppet meshes are still rendered by 2D image-effect cameras, but their bone
    // animation is not limited to the flat source-card z range. Keep ordinary non-puppet effects on
    // the tight range and widen only puppet-capable layer-surface paths that need authored z to
    // survive the final synthetic writer.
    return { -1024.0f, 1024.0f };
}

void LoadAlignment(SceneNode& node, std::string_view align, Vector2f size) {
    // Alignment changes where the centered quad is drawn relative to the authored origin. Store it
    // as a local mesh offset instead of mutating translation, because translation is the pivot that
    // Wallpaper Engine scripts read and rotate around.
    node.SetAlignmentOffset(ResolveImageAlignmentOffset(align, size));
}

std::shared_ptr<SceneNode> FindParentNode(ParseContext& context, int32_t parent_id) {
    auto it = context.object_nodes.find(parent_id);
    return it == context.object_nodes.end() ? nullptr : it->second;
}

void RegisterRenderOrderProxyChild(ParseContext& context, int32_t parent_id,
                                   const std::shared_ptr<SceneNode>& child,
                                   int32_t                           child_layer_id) {
    if (context.scene == nullptr || parent_id == 0 || child == nullptr) return;

    auto parent = FindParentNode(context, parent_id);
    if (! parent) return;

    // This edge is intentionally order-only. The child may remain root-owned for inherited
    // transform, attachment, or effect-camera correctness, while render-graph traversal later emits
    // it at the authored parent's visual sibling position.
    auto& routed_children = context.scene->renderOrderProxyChildren[parent.get()];
    if (std::find(routed_children.begin(), routed_children.end(), child.get()) ==
        routed_children.end()) {
        routed_children.push_back(child.get());
    }
    context.scene->renderOrderProxyNodes.insert(child.get());
    LOG_INFO("SceneRenderOrderProxyRegister: parent-layer=%d proxy-layer=%d parent-name='%s' "
             "proxy-name='%s'",
             parent_id,
             child_layer_id,
             parent->Name().c_str(),
             child->Name().c_str());
}

void RegisterDetachedRenderOrderSource(ParseContext&                     context,
                                       const std::shared_ptr<SceneNode>& world_node,
                                       const std::shared_ptr<SceneNode>& source_node,
                                       int32_t                           layer_id) {
    if (context.scene == nullptr || world_node == nullptr || source_node == nullptr ||
        world_node.get() == source_node.get()) {
        return;
    }

    // Effect-backed layers split into a visible world transform node and a root-owned source node.
    // Recording the pair here lets render-graph construction draw the source exactly where the
    // world node appears in authored layer order without duplicating it during root traversal.
    auto& sources = context.scene->detachedEffectSourceNodesByLayerNode[world_node.get()];
    if (std::find(sources.begin(), sources.end(), source_node.get()) == sources.end()) {
        sources.push_back(source_node.get());
    }
    context.scene->detachedEffectSourceNodes.insert(source_node.get());
    LOG_INFO("SceneRenderOrderDetachedSourceRegister: layer=%d world-name='%s' source-name='%s'",
             layer_id,
             world_node->Name().c_str(),
             source_node->Name().c_str());
}

void RemoveRenderOrderNodeReferences(Scene& scene, SceneNode* node) {
    if (node == nullptr) return;

    // Deferred materialization replaces placeholder nodes with real particle/text nodes. Any
    // order-routing references to the placeholder must be removed first so the new node becomes the
    // only authored-order participant for that layer.
    const auto erase_node_from_vector = [node](std::vector<SceneNode*>& nodes) {
        nodes.erase(std::remove(nodes.begin(), nodes.end(), node), nodes.end());
    };

    scene.renderOrderProxyNodes.erase(node);
    scene.detachedEffectSourceNodes.erase(node);
    for (auto it = scene.renderOrderProxyChildren.begin();
         it != scene.renderOrderProxyChildren.end();) {
        if (it->first == node) {
            it = scene.renderOrderProxyChildren.erase(it);
            continue;
        }
        erase_node_from_vector(it->second);
        if (it->second.empty()) {
            it = scene.renderOrderProxyChildren.erase(it);
        } else {
            ++it;
        }
    }
    for (auto it = scene.detachedEffectSourceNodesByLayerNode.begin();
         it != scene.detachedEffectSourceNodesByLayerNode.end();) {
        if (it->first == node) {
            it = scene.detachedEffectSourceNodesByLayerNode.erase(it);
            continue;
        }
        erase_node_from_vector(it->second);
        if (it->second.empty()) {
            it = scene.detachedEffectSourceNodesByLayerNode.erase(it);
        } else {
            ++it;
        }
    }
}

void RestoreRenderOrderProxyChildrenForLayer(Scene& scene, int32_t parent_layer_id,
                                             SceneNode* parent_node) {
    if (parent_layer_id == 0 || parent_node == nullptr) return;

    auto& routed_children = scene.renderOrderProxyChildren[parent_node];
    for (const auto child_layer_id : scene.GetLayerChildren(parent_layer_id)) {
        const auto binding = scene.GetLayerParentBinding(child_layer_id);
        if (binding.parent_id != parent_layer_id || ! binding.attachment.empty()) continue;

        const auto child_node_it = scene.layerNodes.find(child_layer_id);
        if (child_node_it == scene.layerNodes.end() || child_node_it->second == nullptr) continue;

        // Deferred parent materialization replaces the placeholder node that owned the authored
        // render-order proxy bucket. Existing children may be lightweight containers that are not
        // materialized themselves, so rebuild the parent->child route from layer bindings instead
        // of waiting for every child parser to run again.
        if (std::find(routed_children.begin(), routed_children.end(), child_node_it->second) ==
            routed_children.end()) {
            routed_children.push_back(child_node_it->second);
        }
        scene.renderOrderProxyNodes.insert(child_node_it->second);
        LOG_INFO("SceneRenderOrderProxyRestore: parent-layer=%d proxy-layer=%d parent-name='%s' "
                 "proxy-name='%s'",
                 parent_layer_id,
                 child_layer_id,
                 parent_node->Name().c_str(),
                 child_node_it->second->Name().c_str());
    }

    if (routed_children.empty()) {
        scene.renderOrderProxyChildren.erase(parent_node);
    }
}

struct AttachmentBinding {
    uint32_t        bone_index { 0xFFFFFFFFu };
    Eigen::Affine3f transform { Eigen::Affine3f::Identity() };
};

std::optional<AttachmentBinding> ResolveAttachmentBinding(const ParseContext& context,
                                                          int32_t             parent_id,
                                                          std::string_view    attachment) {
    auto pit = context.object_puppets.find(parent_id);
    if (pit == context.object_puppets.end() || pit->second == nullptr) return std::nullopt;

    const auto& puppet = *pit->second;
    if (const auto* named_attachment = puppet.FindAttachment(attachment)) {
        return AttachmentBinding {
            .bone_index = named_attachment->bone_index,
            .transform  = named_attachment->bind_transform,
        };
    }

    auto bone_index = puppet.FindBoneIndex(attachment);
    if (bone_index == 0xFFFFFFFFu) return std::nullopt;
    return AttachmentBinding {
        .bone_index = bone_index,
        .transform  = Eigen::Affine3f::Identity(),
    };
}

bool ConfigureBoneAttachment(ParseContext& context, int32_t layer_id, int32_t parent_id,
                             std::string_view attachment, const Eigen::Affine3f& local_transform,
                             std::string_view object_kind, std::string_view object_name,
                             WPShaderValueData& node_data) {
    if (parent_id == 0 || attachment.empty()) return false;

    auto parent_node = FindParentNode(context, parent_id);
    if (! parent_node) {
        LOG_ERROR("parent id %d for %s '%s' not found while resolving attachment '%s'",
                  (int)parent_id,
                  std::string(object_kind).c_str(),
                  std::string(object_name).c_str(),
                  std::string(attachment).c_str());
        return false;
    }

    auto attachment_binding = ResolveAttachmentBinding(context, parent_id, attachment);
    if (! attachment_binding.has_value()) {
        LOG_ERROR("attachment '%s' not found for %s '%s'",
                  std::string(attachment).c_str(),
                  std::string(object_kind).c_str(),
                  std::string(object_name).c_str());
        return false;
    }

    Affine3f dest_local = local_transform;
    if (auto* object = context.scene != nullptr ? context.scene->FindSceneObject(layer_id)
                                                : nullptr) {
        // Official image parse 0x1401e6f50 → 0x1401de470 writes +0x180/+0x190
        // and 0x1401dda60 child-list only. Origin stays +0x128 from setter
        // 0x1401a4530 (callback 0x1401e0520 only clears +0xd0). Attach-zero
        // 0x1401de962 runs only when setParent r8 adjustTransforms != 0
        // (0x1401de925); scene bind 0x14018802c uses r8=0.
        object->SetAttachBinding(attachment_binding->bone_index, attachment_binding->transform);
        dest_local = Affine3f(object->ComposeLocalDest());
    }
    node_data.AttachToBone(parent_node.get(),
                           attachment_binding->bone_index,
                           attachment_binding->transform,
                           dest_local);
    return true;
}

void AttachNodeToScene(ParseContext& context, const std::shared_ptr<SceneNode>& node,
                       int32_t parent_id, const std::string& object_name,
                       WPShaderValueData* node_data = nullptr) {
    if (parent_id == 0) {
        context.scene->sceneGraph->AppendChild(node);
        return;
    }

    auto parent = FindParentNode(context, parent_id);
    if (! parent) {
        LOG_ERROR("parent id %d for object '%s' not found, attaching to scene root",
                  (int)parent_id,
                  object_name.c_str());
        context.scene->sceneGraph->AppendChild(node);
        return;
    }

    parent->AppendChild(node);
    if (node_data != nullptr) {
        node_data->SetParallaxAnchor(parent.get());
    }
}

enum class ParentTransformBindingContract
{
    None,
    InheritAuthoredParent,
};

enum class ParentParallaxAnchorContract
{
    None,
    InheritWhenCompatible,
    ForceAuthoredParent,
};

struct ParentTransformContract {
    int32_t                         parent_id { 0 };
    ParentTransformBindingContract  transform_binding {
        ParentTransformBindingContract::None
    };
    ParentParallaxAnchorContract    parallax_anchor {
        ParentParallaxAnchorContract::None
    };

    static ParentTransformContract InheritAuthoredParentTransform(int32_t parent_id) {
        return { .parent_id         = parent_id,
                 .transform_binding = ParentTransformBindingContract::InheritAuthoredParent,
                 .parallax_anchor   = ParentParallaxAnchorContract::InheritWhenCompatible };
    }

    static ParentTransformContract RoutedEffectWorldTransform(int32_t parent_id) {
        return { .parent_id         = parent_id,
                 .transform_binding = ParentTransformBindingContract::InheritAuthoredParent,
                 .parallax_anchor   = ParentParallaxAnchorContract::ForceAuthoredParent };
    }

    static ParentTransformContract ParallaxAnchorWhenCompatible(int32_t parent_id) {
        return { .parent_id       = parent_id,
                 .parallax_anchor = ParentParallaxAnchorContract::InheritWhenCompatible };
    }

    static ParentTransformContract ForceParallaxAnchor(int32_t parent_id) {
        return { .parent_id       = parent_id,
                 .parallax_anchor = ParentParallaxAnchorContract::ForceAuthoredParent };
    }
};

void ApplyParentTransformContract(ParseContext& context, const ParentTransformContract& contract,
                                  WPShaderValueData& node_data) {
    if (contract.parent_id == 0) return;
    auto parent = FindParentNode(context, contract.parent_id);
    if (! parent) return;

    // PATH_B has no attach/final suppress gate. Parent list +0x158 is flat
    // (0x140190837); attach writes +0x180/+0x190 only (ATTACH_WRITE).
    const bool inherit_parent_parallax =
        contract.parallax_anchor != ParentParallaxAnchorContract::None;

    if (contract.transform_binding ==
        ParentTransformBindingContract::InheritAuthoredParent) {
        node_data.InheritParentTransform(parent.get(), inherit_parent_parallax);
        return;
    }

    if (inherit_parent_parallax) {
        node_data.SetParallaxAnchor(parent.get());
    }
}

void ConfigureInheritedParentBinding(ParseContext& context, int32_t parent_id,
                                     WPShaderValueData& node_data) {
    ApplyParentTransformContract(
        context, ParentTransformContract::InheritAuthoredParentTransform(parent_id), node_data);
}

void ConfigureRoutedEffectWorldParentBinding(ParseContext& context, int32_t parent_id,
                                             WPShaderValueData& node_data) {
    // Effect-backed image layers split into a routed world node and private effect nodes. The world
    // node is the scene-space writer that ResolveEffect()/UpdateUniforms use to synchronize the
    // final output, so it must keep the authored parent as its camera-parallax anchor even when both
    // parent and child authored non-zero parallaxDepth. Authored effect materials still keep their
    // own child parallax contract; this binding only preserves the parent component that would
    // otherwise be lost before the final writer is synchronized.
    ApplyParentTransformContract(
        context, ParentTransformContract::RoutedEffectWorldTransform(parent_id), node_data);
}

struct EffectWriterTransformContract {
    std::array<float, 2>       parallax_depth { 0.0f, 0.0f };
    bool                       parallax_depth_authored { true };
    SceneImageEffectLayer*     projection_layer { nullptr };
    bool                       binds_puppet_surface { false };
    ParentTransformContract    parent_transform {};
    bool suppress_own_model_parallax { false };
};

void ApplyEffectWriterTransformContract(ParseContext& context,
                                        const EffectWriterTransformContract& contract,
                                        WPShaderValueData& data) {
    data.SetParallaxContract(contract.parallax_depth,
                             nullptr,
                             false,
                             contract.parallax_depth_authored);

    if (contract.projection_layer != nullptr) {
        data.SetEffectTextureProjection(&contract.projection_layer->FinalNode(),
                                        &contract.projection_layer->FinalMesh(),
                                        contract.projection_layer->AuthoredWidth(),
                                        contract.projection_layer->AuthoredHeight());
        if (contract.binds_puppet_surface) {
            data.SetPuppetSurface(contract.projection_layer,
                                  &contract.projection_layer->FinalMesh());
        }
    }

    ApplyParentTransformContract(context, contract.parent_transform, data);

    if (contract.suppress_own_model_parallax) {
        data.SuppressOwnModelParallax();
    }
}

WPShaderValueData BuildEffectWriterTransformData(ParseContext& context,
                                                 const EffectWriterTransformContract& contract) {
    WPShaderValueData data;
    ApplyEffectWriterTransformContract(context, contract, data);
    return data;
}

EffectWriterTransformContract BuildImageEffectFinalCompositeContract(
    const wpscene::WPImageObject& image, bool uses_routed_parent) {
    EffectWriterTransformContract contract;
    contract.parallax_depth          = ImageObjectParallaxDepth(image);
    contract.parallax_depth_authored = image.parallaxDepthAuthored;

    if (uses_routed_parent) {
        // Effect-backed image world nodes deliberately force the authored parent as their parallax
        // anchor because their physical SceneNode is detached and routed only for render order. The
        // private final composite is the actual visible writer, so it must mirror that same anchor
        // rather than re-running compatibility selection against its own synthetic node data. This
        // keeps parented media covers and other routed effect layers moving with their authored
        // parent while the route matrix continues to supply only the raw transform hierarchy.
        contract.parent_transform = ParentTransformContract::ForceParallaxAnchor(image.parent);
    }
    return contract;
}

enum class ImageEffectWriterRole
{
    AuthoredEffectProjection,
    LayerSurfaceProxy,
};

struct ImageEffectMaterialTopology {
    bool                  uses_routed_parent { false };
    ImageEffectWriterRole writer_role { ImageEffectWriterRole::AuthoredEffectProjection };

    bool NeedsLayerSurfaceParentParallax() const {
        return uses_routed_parent && writer_role == ImageEffectWriterRole::LayerSurfaceProxy;
    }
};

ImageEffectWriterRole ResolveImageEffectWriterRole(bool is_compose_layer) {
    if (is_compose_layer) return ImageEffectWriterRole::LayerSurfaceProxy;
    return ImageEffectWriterRole::AuthoredEffectProjection;
}

EffectWriterTransformContract BuildTextEffectFinalCompositeContract(
    const ParseContext& context, const wpscene::WPTextObject& text) {
    EffectWriterTransformContract contract;
    contract.parallax_depth          = TextObjectParallaxDepth(text);
    contract.parallax_depth_authored = text.parallaxDepthAuthored;
    if (LayerUsesRoutedParent(context, text.parent, text.attachment)) {
        // The route matrix contains the authored parent transform but never shader-time mouse
        // parallax. A parented text writer always inherits that parent camera-parallax source so
        // the final composite applies exactly one offset.
        contract.parent_transform =
            ParentTransformContract::ParallaxAnchorWhenCompatible(text.parent);
    }
    return contract;
}

EffectWriterTransformContract BuildImageEffectMaterialContract(
    const wpscene::WPImageObject& image, SceneImageEffectLayer& effect_layer,
    const ImageEffectMaterialTopology& topology, bool private_layer_surface_writer) {
    EffectWriterTransformContract contract;
    contract.parallax_depth          = ImageObjectParallaxDepth(image);
    contract.parallax_depth_authored = image.parallaxDepthAuthored;
    contract.projection_layer        = &effect_layer;
    contract.binds_puppet_surface    = private_layer_surface_writer;

    if (private_layer_surface_writer) {
        // An animated puppet surface writer rasterizes the skinned mesh through the layer-local
        // source camera, then the neutral final composite places that resolved texture in scene
        // space. Camera parallax belongs exclusively to that final scene-space placement. Applying
        // it during the private puppet draw as well shifts the body inside its texture, while a
        // bone-attached child receives the parent's parallax only through its attachment transform;
        // the two pieces therefore separate as the pointer moves. Keep the private rasterization
        // local so the final composite and every attachment observe one shared parallax transform.
        contract.suppress_own_model_parallax = true;
    }

    if (topology.NeedsLayerSurfaceParentParallax()) {
        // This decision is based on render topology, not on authored parallax values. Compose
        // layers have no authored effect projection that should own a separate child-space parallax
        // result; their resolved screen writer is the layer image itself, merely routed through the
        // image-effect path. Match the no-effect
        // image-layer contract by inheriting the authored parent's parallax anchor, so virtual
        // render-order parents keep routed visual layers locked together.
        //
        // Detached chains with real authored effects intentionally do not enter this branch. Their
        // final writer belongs to the effect pipeline, and the world route matrix already supplies
        // the parent transform. Re-anchoring that authored final writer to the parent changes the
        // effect-chain projection contract and moves layers such as the lantern media cover and
        // audio rings away from their authored local center.
        contract.parent_transform = ParentTransformContract::ForceParallaxAnchor(image.parent);
    }
    return contract;
}

EffectWriterTransformContract BuildTextEffectMaterialContract(
    const ParseContext& context, const wpscene::WPTextObject& text,
    SceneImageEffectLayer& effect_layer) {
    EffectWriterTransformContract contract;
    contract.parallax_depth          = TextObjectParallaxDepth(text);
    contract.parallax_depth_authored = text.parallaxDepthAuthored;
    contract.projection_layer        = &effect_layer;
    if (LayerUsesRoutedParent(context, text.parent, text.attachment)) {
        // Text effect nodes start as private bridge passes, but ResolveEffect() may turn the last
        // authored effect node into the visible scene-space writer. Parent-routed text already
        // receives the visual parent chain, including parent camera parallax, through the render
        // graph route matrix. Suppressing this node's own model parallax prevents two failure modes:
        // non-zero child text depths drifting inside zero-parallax HUD groups, and zero-depth
        // effect-backed date labels receiving an extra copy of their moving parent's parallax.
        contract.suppress_own_model_parallax = true;
    }
    return contract;
}

void RegisterLayerSceneState(ParseContext& context, int32_t layer_id, int32_t parent_id,
                             std::string_view attachment, bool visible) {
    if (context.scene == nullptr || layer_id == 0) return;
    context.scene->EnsureSceneObject(layer_id);
    context.scene->SetLayerParentBinding(layer_id, parent_id, std::string(attachment));
    context.scene->SetLayerLocalVisibility(layer_id, visible);
}

void RegisterLogicalImageLayer(ParseContext& context, const wpscene::WPImageObject& wpimgobj,
                               bool defer_runtime_materialization) {
    NotePassthroughLayer(context, wpimgobj);
    auto node = std::make_shared<SceneNode>(Vector3f(wpimgobj.origin.data()),
                                            Vector3f(wpimgobj.scale.data()),
                                            Vector3f(wpimgobj.angles.data()),
                                            wpimgobj.name);
    LoadAlignment(*node, wpimgobj.alignment, { wpimgobj.size[0], wpimgobj.size[1] });
    node->ID() = wpimgobj.id;

    WPShaderValueData node_data;
    node_data.parallaxDepth = { wpimgobj.parallaxDepth[0], wpimgobj.parallaxDepth[1] };
    node_data.parallaxDepthAuthored = wpimgobj.parallaxDepthAuthored;
    ConfigureBoneAttachment(context,
                            wpimgobj.id,
                            wpimgobj.parent,
                            wpimgobj.attachment,
                            Eigen::Affine3f(node->GetLocalTrans().cast<float>()),
                            "image layer",
                            wpimgobj.name,
                            node_data);

    if (LayerUsesRoutedParent(context, wpimgobj.parent, wpimgobj.attachment)) {
        ConfigureInheritedParentBinding(context, wpimgobj.parent, node_data);
        context.scene->sceneGraph->AppendChild(node);
        RegisterRenderOrderProxyChild(context, wpimgobj.parent, node, wpimgobj.id);
    } else if (wpimgobj.parent != 0) {
        if (wpimgobj.attachment.empty()) {
            ConfigureInheritedParentBinding(context, wpimgobj.parent, node_data);
        }
        context.scene->sceneGraph->AppendChild(node);
    } else {
        AttachNodeToScene(context, node, wpimgobj.parent, wpimgobj.name, &node_data);
    }

    context.object_nodes[wpimgobj.id]       = node;
    context.scene->imageLayers[wpimgobj.id] = Scene::ImageLayerRuntimeState {
        .size      = wpimgobj.size,
        .alignment = wpimgobj.alignment,
    };
    context.scene->objectRuntimeNodes[wpimgobj.id].push_back(node.get());
    context.scene->nodeOwners[node.get()] = wpimgobj.id;
    context.shader_updater->SetNodeData(node.get(), node_data);
    RegisterLayerSceneState(
        context, wpimgobj.id, wpimgobj.parent, wpimgobj.attachment, wpimgobj.visible);
    context.scene->ApplyLayerVisibility(wpimgobj.id);

    if (defer_runtime_materialization) {
        // Hidden user/script-controlled image layers can be very expensive: materializing their
        // complete effect chains creates render targets, pipelines, and descriptors even though no
        // pass can execute while the layer is invisible. Keep only the transform/runtime contract
        // until the visibility property first turns true, then rebuild the graph with real passes.
        context.scene->deferredRuntimeImageLayerIds.insert(wpimgobj.id);
    }

    LOG_INFO("SceneObjectMaterialize: mode=image-logical-only id=%d name='%s' image='%s' "
             "fullscreen=%s autosize=%s projectlayer=%s effects=%zu dependency-source=%s "
             "deferred-runtime=%s",
             wpimgobj.id,
             wpimgobj.name.c_str(),
             wpimgobj.image.c_str(),
             wpimgobj.fullscreen ? "true" : "false",
             wpimgobj.autosize ? "true" : "false",
             wpimgobj.projectlayer ? "true" : "false",
             wpimgobj.effects.size(),
             context.scene != nullptr &&
                     context.scene->offscreenDependencyLayerIds.count(wpimgobj.id) != 0
                 ? "true"
                 : "false",
             defer_runtime_materialization ? "true" : "false");
}

void RegisterLogicalParticleLayer(ParseContext& context, wpscene::WPParticleObject& wppartobj) {
    auto node  = std::make_shared<SceneNode>(Vector3f(wppartobj.origin.data()),
                                             Vector3f(wppartobj.scale.data()),
                                             Vector3f(wppartobj.angles.data()),
                                             wppartobj.name);
    node->ID() = wppartobj.id;

    WPShaderValueData node_data;
    node_data.parallaxDepth = { wppartobj.parallaxDepth[0], wppartobj.parallaxDepth[1] };
    node_data.parallaxDepthAuthored = wppartobj.parallaxDepthAuthored;

    ConfigureBoneAttachment(context,
                            wppartobj.id,
                            wppartobj.parent,
                            wppartobj.attachment,
                            Eigen::Affine3f(node->GetLocalTrans().cast<float>()),
                            "particle object",
                            wppartobj.name,
                            node_data);

    if (LayerUsesRoutedParent(context, wppartobj.parent, wppartobj.attachment)) {
        ConfigureInheritedParentBinding(context, wppartobj.parent, node_data);
        context.scene->sceneGraph->AppendChild(node);
        RegisterRenderOrderProxyChild(context, wppartobj.parent, node, wppartobj.id);
    } else if (wppartobj.parent != 0) {
        if (wppartobj.attachment.empty()) {
            ConfigureInheritedParentBinding(context, wppartobj.parent, node_data);
        }
        context.scene->sceneGraph->AppendChild(node);
    } else {
        AttachNodeToScene(context, node, wppartobj.parent, wppartobj.name, &node_data);
    }

    context.object_nodes[wppartobj.id] = node;
    context.scene->objectRuntimeNodes[wppartobj.id].push_back(node.get());
    context.scene->nodeOwners[node.get()] = wppartobj.id;
    if (auto* object = context.scene->FindSceneObject(wppartobj.id)) {
        object->set_source_node(node.get());
    }
    context.shader_updater->SetNodeData(node.get(), node_data);
    RegisterLayerSceneState(
        context, wppartobj.id, wppartobj.parent, wppartobj.attachment, wppartobj.visible);
    context.scene->ApplyLayerVisibility(wppartobj.id);
    context.scene->deferredRuntimeParticleLayerIds.insert(wppartobj.id);
}

void RegisterLogicalTextLayer(ParseContext& context,
                              wpscene::WPTextObject& text_obj,
                              const TextLayerRenderContract& render_contract) {
    auto node  = std::make_shared<SceneNode>(Vector3f(text_obj.origin.data()),
                                             Vector3f(text_obj.scale.data()),
                                             Vector3f(text_obj.angles.data()),
                                             text_obj.name);
    node->ID() = text_obj.id;

    WPShaderValueData node_data;
    node_data.parallaxDepth = { text_obj.parallaxDepth[0], text_obj.parallaxDepth[1] };
    node_data.parallaxDepthAuthored = text_obj.parallaxDepthAuthored;

    ConfigureBoneAttachment(context,
                            text_obj.id,
                            text_obj.parent,
                            text_obj.attachment,
                            Eigen::Affine3f(node->GetLocalTrans().cast<float>()),
                            "text object",
                            text_obj.name,
                            node_data);

    if (LayerUsesRoutedParent(context, text_obj.parent, text_obj.attachment)) {
        ConfigureInheritedParentBinding(context, text_obj.parent, node_data);
        context.scene->sceneGraph->AppendChild(node);
        RegisterRenderOrderProxyChild(context, text_obj.parent, node, text_obj.id);
    } else if (text_obj.parent != 0) {
        if (text_obj.attachment.empty()) {
            ConfigureInheritedParentBinding(context, text_obj.parent, node_data);
        }
        context.scene->sceneGraph->AppendChild(node);
    } else {
        AttachNodeToScene(context, node, text_obj.parent, text_obj.name, &node_data);
    }

    context.object_nodes[text_obj.id] = node;
    context.scene->objectRuntimeNodes[text_obj.id].push_back(node.get());
    context.scene->nodeOwners[node.get()] = text_obj.id;
    context.shader_updater->SetNodeData(node.get(), node_data);
    context.scene->textLayers[text_obj.id] = TextLayerRuntimeState {
        .object            = text_obj,
        .primitive         = nullptr,
        .render_contract   = render_contract,
        .applied_alignment = ResolveTextLayerSceneAlignment(text_obj),
    };
    RegisterLayerSceneState(
        context, text_obj.id, text_obj.parent, text_obj.attachment, text_obj.visible);
    context.scene->ApplyLayerVisibility(text_obj.id);
    context.scene->deferredRuntimeTextLayerIds.insert(text_obj.id);
}

bool DetachNodeFromTree(const std::shared_ptr<SceneNode>& parent, SceneNode* target) {
    if (! parent || target == nullptr) return false;
    if (parent->RemoveChild(target)) return true;
    for (auto& child : parent->GetChildren()) {
        if (DetachNodeFromTree(child, target)) return true;
    }
    return false;
}

std::vector<std::shared_ptr<SceneNode>> ExtractDirectChildren(SceneNode* node) {
    std::vector<std::shared_ptr<SceneNode>> children;
    if (node == nullptr) return children;

    while (! node->GetChildren().empty()) {
        auto child = node->GetChildren().front();
        if (! node->RemoveChild(child.get())) break;
        children.push_back(std::move(child));
    }

    return children;
}

int32_t ResolveLayerIdForRuntimeNode(const Scene& scene, SceneNode* node) {
    if (node == nullptr) return 0;
    if (auto owner_it = scene.nodeOwners.find(node);
        owner_it != scene.nodeOwners.end()) {
        return owner_it->second;
    }
    return node->ID();
}

void RebindAdoptedAttachmentChildren(ParseContext& context, int32_t parent_layer_id,
                                     SceneNode* parent_node,
                                     const std::vector<std::shared_ptr<SceneNode>>& children) {
    if (context.scene == nullptr || context.shader_updater == nullptr || parent_layer_id == 0 ||
        parent_node == nullptr) {
        return;
    }

    for (const auto& child : children) {
        if (! child) continue;

        const int32_t child_layer_id = ResolveLayerIdForRuntimeNode(*context.scene, child.get());
        if (child_layer_id == 0) continue;

        const auto binding = context.scene->GetLayerParentBinding(child_layer_id);
        if (binding.parent_id != parent_layer_id || binding.attachment.empty()) continue;

        auto attachment_binding =
            ResolveAttachmentBinding(context, parent_layer_id, binding.attachment);
        if (! attachment_binding.has_value()) continue;

        auto* child_data = context.shader_updater->GetNodeData(child.get());
        if (child_data == nullptr) continue;

        // Attachment children can be parsed while their hidden parent is still only a lightweight
        // placeholder, so the original bone lookup may fail and the child is temporarily kept as a
        // normal scene child. Once the parent is materialized and its puppet is available, promote
        // that child back to the authored bone-attachment contract before the old placeholder is
        // destroyed.
        child_data->AttachToBone(parent_node,
                                 attachment_binding->bone_index,
                                 attachment_binding->transform,
                                 Eigen::Affine3f(child->GetLocalTrans().cast<float>()));
    }
}

void ReplaceDeferredPlaceholderNode(ParseContext& context, int32_t layer_id,
                                    const std::shared_ptr<SceneNode>& placeholder_node,
                                    const std::shared_ptr<SceneNode>& replacement_node) {
    if (context.scene == nullptr || ! placeholder_node || ! replacement_node ||
        placeholder_node.get() == replacement_node.get()) {
        return;
    }

    // Hidden runtime layers are first represented by lightweight placeholder nodes so scripts and
    // user bindings have a stable layer identity. Some authored attachment children are real
    // SceneNode children of that placeholder. Destroying the placeholder without first adopting
    // those children leaves scene.layerNodes, objectRuntimeNodes, and shader-data parent pointers
    // aimed at freed memory; later allocations can reuse the same address and create recursive or
    // dangling transform chains during uniform updates.
    auto adopted_children = ExtractDirectChildren(placeholder_node.get());
    RemoveRenderOrderNodeReferences(*context.scene, placeholder_node.get());
    if (context.scene->sceneGraph != nullptr) {
        (void)DetachNodeFromTree(context.scene->sceneGraph, placeholder_node.get());
    }
    context.scene->nodeOwners.erase(placeholder_node.get());
    if (context.shader_updater != nullptr) {
        context.shader_updater->ReplaceNodeReferences(placeholder_node.get(), replacement_node.get());
    }

    RebindAdoptedAttachmentChildren(
        context, layer_id, replacement_node.get(), adopted_children);
    for (auto& child : adopted_children) {
        if (! child) continue;
        replacement_node->AppendChild(child);
    }
}

struct ResolvedUserShaderValueBinding {
    std::string        user_property_name;
    std::string        material_value_name;
    std::string        gl_uniform_name;
    const ShaderValue* property { nullptr };
    bool               legacy_reversed { false };
};

enum class MaterialValueUniformResolutionKind
{
    ExactAlias,
    UniformName,
    UniformSuffix,
    NormalizedAlias,
    AmbiguousNormalizedAlias,
    Unresolved,
};

struct MaterialValueUniformResolution {
    std::string                        uniform_name;
    std::string                        matched_alias;
    MaterialValueUniformResolutionKind kind {
        MaterialValueUniformResolutionKind::Unresolved
    };

    bool resolved() const noexcept {
        return kind != MaterialValueUniformResolutionKind::Unresolved &&
               kind != MaterialValueUniformResolutionKind::AmbiguousNormalizedAlias;
    }
};

const char* MaterialValueUniformResolutionKindName(MaterialValueUniformResolutionKind kind) {
    switch (kind) {
    case MaterialValueUniformResolutionKind::ExactAlias: return "exact-alias";
    case MaterialValueUniformResolutionKind::UniformName: return "uniform-name";
    case MaterialValueUniformResolutionKind::UniformSuffix: return "uniform-suffix";
    case MaterialValueUniformResolutionKind::NormalizedAlias: return "normalized-alias";
    case MaterialValueUniformResolutionKind::AmbiguousNormalizedAlias:
        return "ambiguous-normalized-alias";
    case MaterialValueUniformResolutionKind::Unresolved: return "unresolved";
    }
    return "unknown";
}

bool IsDirectMaterialValueResolution(MaterialValueUniformResolutionKind kind) {
    return kind == MaterialValueUniformResolutionKind::ExactAlias ||
           kind == MaterialValueUniformResolutionKind::UniformName ||
           kind == MaterialValueUniformResolutionKind::UniformSuffix;
}

std::string NormalizeMaterialValueAlias(std::string_view name) {
    std::string normalized;
    int         parenthetical_depth = 0;
    for (unsigned char raw_ch : name) {
        const char ch = static_cast<char>(raw_ch);
        if (ch == '(') {
            parenthetical_depth++;
            continue;
        }
        if (ch == ')') {
            if (parenthetical_depth > 0) parenthetical_depth--;
            continue;
        }
        if (parenthetical_depth > 0) continue;

        if (std::isalnum(raw_ch)) {
            normalized.push_back(static_cast<char>(std::tolower(raw_ch)));
        }
    }

    // Wallpaper Engine sometimes serializes constants by editor label ("Texture parallax depth")
    // while the shader metadata only exposes numbered material keys ("4textureParallaxDepth").
    // Dropping only leading digits lets those two forms meet without treating unrelated numeric
    // suffixes as equivalent.
    const auto first_non_digit =
        std::find_if(normalized.begin(), normalized.end(), [](unsigned char ch) {
            return ! std::isdigit(ch);
        });
    normalized.erase(normalized.begin(), first_non_digit);
    return normalized;
}

MaterialValueUniformResolution
ResolveMaterialValueUniform(const WPShaderInfo& info, std::string_view material_value_name,
                            bool allow_normalized_alias) {
    const std::string material_value_key(material_value_name);
    if (const auto alias_it = info.alias.find(material_value_key); alias_it != info.alias.end()) {
        return {
            .uniform_name  = alias_it->second,
            .matched_alias = alias_it->first,
            .kind          = MaterialValueUniformResolutionKind::ExactAlias,
        };
    }

    for (const auto& [alias_name, uniform_name] : info.alias) {
        if (uniform_name == material_value_key) {
            return {
                .uniform_name  = uniform_name,
                .matched_alias = alias_name,
                .kind          = MaterialValueUniformResolutionKind::UniformName,
            };
        }

        // Some shader metadata stores material aliases like `color1`, while the parsed GLSL
        // uniform is named `g_Color1`. Keep this suffix match so user-facing project properties
        // can still target old stock shaders whose material JSON uses the shorter alias instead
        // of the final GLSL symbol.
        if (uniform_name.size() > 2 && uniform_name.substr(2) == material_value_key) {
            return {
                .uniform_name  = uniform_name,
                .matched_alias = alias_name,
                .kind          = MaterialValueUniformResolutionKind::UniformSuffix,
            };
        }
    }

    if (! allow_normalized_alias) {
        return {
            .uniform_name = material_value_key,
            .kind         = MaterialValueUniformResolutionKind::Unresolved,
        };
    }

    const auto normalized_key = NormalizeMaterialValueAlias(material_value_key);
    if (normalized_key.empty()) {
        return {
            .uniform_name = material_value_key,
            .kind         = MaterialValueUniformResolutionKind::Unresolved,
        };
    }

    std::optional<MaterialValueUniformResolution> candidate;
    for (const auto& [alias_name, uniform_name] : info.alias) {
        if (NormalizeMaterialValueAlias(alias_name) != normalized_key) continue;

        if (candidate.has_value() && candidate->uniform_name != uniform_name) {
            return {
                .uniform_name  = material_value_key,
                .matched_alias = alias_name,
                .kind = MaterialValueUniformResolutionKind::AmbiguousNormalizedAlias,
            };
        }

        candidate = MaterialValueUniformResolution {
            .uniform_name  = uniform_name,
            .matched_alias = alias_name,
            .kind          = MaterialValueUniformResolutionKind::NormalizedAlias,
        };
    }

    if (candidate.has_value()) return *candidate;
    return {
        .uniform_name = material_value_key,
        .kind         = MaterialValueUniformResolutionKind::Unresolved,
    };
}

std::string ResolveMaterialValueUniformName(const WPShaderInfo& info,
                                            const std::string&  material_value_name) {
    const auto resolution = ResolveMaterialValueUniform(info, material_value_name, true);
    return resolution.resolved() ? resolution.uniform_name : material_value_name;
}

void ApplyResolvedConstvalue(SceneMaterial& material, const std::string& material_value_name,
                             const std::vector<float>&                 value,
                             const MaterialValueUniformResolution&     resolution) {
    if (! resolution.resolved()) return;
    if (resolution.kind == MaterialValueUniformResolutionKind::NormalizedAlias) {
        LOG_INFO("ShaderValueAliasFallback: material-value='%s' alias='%s' uniform='%s'",
                 material_value_name.c_str(),
                 resolution.matched_alias.c_str(),
                 resolution.uniform_name.c_str());
    }

    material.customShader.constValues[resolution.uniform_name] =
        ClampParserOpacityUniformValue(resolution.uniform_name, ShaderValue(value));
}

void LoadConstvalue(SceneMaterial& material, const wpscene::WPMaterial& wpmat,
                    const WPShaderInfo& info) {
    // Apply exact authored material keys before display-name fallbacks. Some Wallpaper Engine
    // projects serialize both forms in one pass; the display-name value is the editor-visible
    // override and must be allowed to replace the internal default key deterministically.
    for (const auto& [name, value] : wpmat.constantshadervalues) {
        const auto resolution = ResolveMaterialValueUniform(info, name, false);
        if (! resolution.resolved()) continue;
        ApplyResolvedConstvalue(material, name, value, resolution);
    }

    for (const auto& [name, value] : wpmat.constantshadervalues) {
        const auto direct_resolution = ResolveMaterialValueUniform(info, name, false);
        if (direct_resolution.resolved() &&
            IsDirectMaterialValueResolution(direct_resolution.kind)) {
            continue;
        }

        const auto resolution = ResolveMaterialValueUniform(info, name, true);
        if (resolution.resolved()) {
            ApplyResolvedConstvalue(material, name, value, resolution);
            continue;
        }

        LOG_WARN("ShaderValue: material-value='%s' skipped reason=%s",
                 name.c_str(),
                 MaterialValueUniformResolutionKindName(resolution.kind));
    }
}

std::vector<ResolvedUserShaderValueBinding>
ResolveUserShaderValueBindings(const wpscene::WPMaterial& wpmat, const WPShaderInfo& info,
                               const UserPropertyMap* user_properties, bool log_missing) {
    std::vector<ResolvedUserShaderValueBinding> bindings;
    if (user_properties == nullptr) return bindings;

    bindings.reserve(wpmat.usershadervalues.size());
    for (const auto& us : wpmat.usershadervalues) {
        // Wallpaper Engine writes `usershadervalues` as
        // `{ "<project user property>": "<shader material value>" }`. Eagle Flag is a compact
        // example: `schemecolor -> color1`, `flagcolor1 -> color2`, and `flagcolor2 -> color3`.
        // Looking up the value side as a user property misses the authored colors and leaves the
        // shader on its black/white defaults, which makes the red and green flag regions vanish.
        std::string user_property_name  = us.first;
        std::string material_value_name = us.second;
        bool        legacy_reversed     = false;
        const auto* property = LookupUserPropertyShaderValue(user_properties, user_property_name);
        if (property == nullptr) {
            // Older local builds interpreted the mapping in the opposite direction. This fallback
            // keeps any locally-authored scenes that accidentally depended on that reversed
            // behavior visible, while logging the mismatch so the material JSON can be fixed.
            const auto* legacy_property =
                LookupUserPropertyShaderValue(user_properties, material_value_name);
            if (legacy_property != nullptr) {
                legacy_reversed = true;
                std::swap(user_property_name, material_value_name);
                property = legacy_property;
            } else {
                if (log_missing) {
                    LOG_INFO("UserShaderValue: property '%s' not provided for material value '%s'",
                             user_property_name.c_str(),
                             material_value_name.c_str());
                }
                continue;
            }
        }

        const auto gl_uniform_name = ResolveMaterialValueUniformName(info, material_value_name);
        bindings.push_back(ResolvedUserShaderValueBinding {
            .user_property_name  = std::move(user_property_name),
            .material_value_name = std::move(material_value_name),
            .gl_uniform_name     = gl_uniform_name,
            .property            = property,
            .legacy_reversed     = legacy_reversed,
        });
    }

    return bindings;
}

WPDynamicValue::Type DynamicTypeForShaderValue(const ShaderValue& value) {
    switch (value.size()) {
    case 2: return WPDynamicValue::Type::Float2;
    case 3: return WPDynamicValue::Type::Float3;
    case 4: return WPDynamicValue::Type::Float4;
    case 1: return WPDynamicValue::Type::Float;
    default: return WPDynamicValue::Type::FloatVector;
    }
}

bool SceneMaterialHasUniform(const SceneMaterial& material, std::string_view uniform_name) {
    const std::string uniform_key(uniform_name);
    if (material.customShader.constValues.count(uniform_key) != 0) return true;
    return material.customShader.shader != nullptr &&
           material.customShader.shader->default_uniforms.count(uniform_key) != 0;
}

void RegisterUserShaderValueBindings(ParseContext& context, const wpscene::WPMaterial& wpmat,
                                     const WPShaderInfo& info, SceneNode* node, int32_t object_id,
                                     std::string_view object_name) {
    if (context.scene == nullptr || node == nullptr || node->Mesh() == nullptr ||
        node->Mesh()->Material() == nullptr) {
        return;
    }

    for (const auto& binding :
         ResolveUserShaderValueBindings(wpmat, info, context.user_properties, false)) {
        if (binding.property == nullptr) continue;

        const auto value_type = DynamicTypeForShaderValue(*binding.property);
        auto       base_value =
            WPDynamicValue::FromUserPropertyValue(UserPropertyValue(*binding.property), value_type)
                .value_or(WPDynamicValue {});

        WPUserSetting setting;
        setting.value    = base_value;
        setting.property = UserPropertyBinding {
            .name      = binding.user_property_name,
            .condition = {},
        };

        // `usershadervalues` bindings are not layer properties: they write directly into the
        // material uniform map. Registering them after the material has been attached makes 2D
        // layers and 3D model chunks share the same live-update contract: the dispatcher resolves
        // node->Mesh()->Material() immediately and writes the same GLSL uniform that the cold parse
        // resolved from shader metadata.
        context.scene->bindingRegistrations.push_back(WPSceneScriptRegistration {
            .object_id     = object_id,
            .object_name   = std::string(object_name),
            .property_name = binding.gl_uniform_name,
            .node          = node,
            .target_kind   = WPSceneScriptTargetKind::MaterialUniform,
            .target_index  = 0,
            .value_type    = value_type,
            .base_value    = base_value,
            .setting       = std::move(setting),
        });

        LOG_INFO("UserShaderValueRegister: layer=%d name='%.*s' user-property='%s' "
                 "material-value='%s' uniform='%s' components=%zu legacy-reversed=%s",
                 object_id,
                 static_cast<int>(object_name.size()),
                 object_name.data(),
                 binding.user_property_name.c_str(),
                 binding.material_value_name.c_str(),
                 binding.gl_uniform_name.c_str(),
                 binding.property->size(),
                 binding.legacy_reversed ? "true" : "false");
    }
}

void RegisterConstantShaderValueBindings(ParseContext& context, const wpscene::WPMaterial& wpmat,
                                         const WPShaderInfo& info, SceneNode* node,
                                         int32_t object_id, std::string_view object_name,
                                         int32_t effect_id, int32_t effect_index,
                                         usize material_index) {
    if (context.scene == nullptr || node == nullptr || node->Mesh() == nullptr ||
        node->Mesh()->Material() == nullptr) {
        return;
    }

    for (const auto& [material_value_name, binding] : wpmat.constantshadervaluebindings) {
        const auto& setting       = binding.setting;
        const bool  has_animation = binding.animation != nullptr && binding.animation->valid();
        if (! setting.hasUserBinding() && ! setting.hasScript() && ! has_animation) continue;

        const auto  resolution      = ResolveMaterialValueUniform(info, material_value_name, true);
        const auto& gl_uniform_name = resolution.uniform_name;
        if (! SceneMaterialHasUniform(*node->Mesh()->Material(), gl_uniform_name)) {
            LOG_INFO("ConstantShaderValueRegister: layer=%d effect-id=%d effect-index=%d "
                     "material-index=%zu material-value='%s' unresolved uniform='%s' reason=%s",
                     object_id,
                     effect_id,
                     effect_index,
                     material_index,
                     material_value_name.c_str(),
                     gl_uniform_name.c_str(),
                     MaterialValueUniformResolutionKindName(resolution.kind));
            continue;
        }

        // Effect pass constants are parsed into SceneMaterial::constValues for cold start, but
        // dynamic constants also need a live target on the concrete pass node. User bindings and
        // scripts both reuse the MaterialUniform dispatcher so album-art color scripts can update
        // Gradient Color uniforms without rebuilding the post-process chain.
        WPSceneScriptRegistration registration {
            .object_id     = object_id,
            .object_name   = std::string(object_name),
            .property_name = gl_uniform_name,
            .node          = node,
            .target_kind   = WPSceneScriptTargetKind::MaterialUniform,
            .target_index  = static_cast<uint32_t>(material_index),
            .target_id     = effect_id,
            .value_type    = setting.value.type(),
            .base_value    = setting.value,
            .setting       = setting,
        };

        std::string registration_kind;
        if (has_animation) {
            // Material-uniform animations are registered against the same target descriptor as
            // their sibling script/user binding. This keeps thisObject.getAnimation() resolvable
            // for effect scripts that replay cover-transition timelines during media changes.
            auto animation_registration      = registration;
            animation_registration.animation = binding.animation;
            context.scene->propertyAnimationRegistrations.push_back(
                std::move(animation_registration));
            registration_kind = "animation";
        }
        if (setting.hasScript()) {
            context.scene->scriptRegistrations.push_back(registration);
            registration_kind += registration_kind.empty() ? "script" : "+script";
        } else if (setting.hasUserBinding()) {
            context.scene->bindingRegistrations.push_back(registration);
            registration_kind += registration_kind.empty() ? "user" : "+user";
        }

        LOG_INFO("ConstantShaderValueRegister: layer=%d name='%.*s' effect-id=%d "
                 "effect-index=%d material-index=%zu kind=%s user-property='%s' "
                 "material-value='%s' uniform='%s' value-type=%s",
                 object_id,
                 static_cast<int>(object_name.size()),
                 object_name.data(),
                 effect_id,
                 effect_index,
                 material_index,
                 registration_kind.c_str(),
                 setting.property.has_value() ? setting.property->name.c_str() : "",
                 material_value_name.c_str(),
                 gl_uniform_name.c_str(),
                 DynamicValueTypeName(setting.value.type()));
    }
}

void LoadUserShaderValue(SceneMaterial& material, const wpscene::WPMaterial& wpmat,
                         const WPShaderInfo& info, const UserPropertyMap* user_properties) {
    for (const auto& binding : ResolveUserShaderValueBindings(wpmat, info, user_properties, true)) {
        if (binding.legacy_reversed) {
            LOG_INFO("UserShaderValue: legacy reversed mapping user-property '%s' -> material "
                     "value '%s'",
                     binding.user_property_name.c_str(),
                     binding.material_value_name.c_str());
        }

        LOG_INFO("UserShaderValue: property '%s' -> material value '%s' -> uniform '%s' (%zu)",
                 binding.user_property_name.c_str(),
                 binding.material_value_name.c_str(),
                 binding.gl_uniform_name.c_str(),
                 binding.property != nullptr ? binding.property->size() : 0);
        if (binding.property != nullptr)
            material.customShader.constValues[binding.gl_uniform_name] =
                ClampParserOpacityUniformValue(binding.gl_uniform_name, *binding.property);
    }
}

// parse

bool ParseModelCameraPathKeyframe(const nlohmann::json& json, Scene::CameraPathKeyframe& keyframe) {
    if (! json.is_object()) return false;
    GET_JSON_NAME_VALUE_NOWARN(json, "timestamp", keyframe.timestamp);
    GET_JSON_NAME_VALUE_NOWARN(json, "eye", keyframe.eye);
    GET_JSON_NAME_VALUE_NOWARN(json, "center", keyframe.center);
    GET_JSON_NAME_VALUE_NOWARN(json, "up", keyframe.up);
    return true;
}

void LoadModelCameraPaths(ParseContext& context, const wpscene::WPSceneCamera& authored_camera) {
    auto& scene = *context.scene;
    scene.modelCameraPathSegments.clear();
    scene.modelCameraPathEnabled       = false;
    scene.activeModelCameraPathSegment = -1;

    if (authored_camera.paths.empty()) return;

    for (const auto& relative_path : authored_camera.paths) {
        const std::string asset_path = "/assets/" + relative_path;
        nlohmann::json    camera_path_json;
        if (! context.vfs->Contains(asset_path) ||
            ! PARSE_JSON(fs::GetFileContent(*context.vfs, asset_path), camera_path_json)) {
            LOG_ERROR("Scene3DModelCameraPath: failed to read path='%s'", relative_path.c_str());
            continue;
        }
        if (! camera_path_json.is_object() || ! camera_path_json.contains("paths") ||
            ! camera_path_json.at("paths").is_array()) {
            LOG_ERROR("Scene3DModelCameraPath: path file has no paths array path='%s'",
                      relative_path.c_str());
            continue;
        }

        for (const auto& segment_json : camera_path_json.at("paths")) {
            if (! segment_json.is_object()) continue;

            Scene::CameraPathSegment segment;
            GET_JSON_NAME_VALUE_NOWARN(segment_json, "name", segment.name);
            GET_JSON_NAME_VALUE_NOWARN(segment_json, "duration", segment.duration);
            if (segment_json.contains("transforms") && segment_json.at("transforms").is_array()) {
                for (const auto& keyframe_json : segment_json.at("transforms")) {
                    Scene::CameraPathKeyframe keyframe;
                    if (ParseModelCameraPathKeyframe(keyframe_json, keyframe)) {
                        segment.keyframes.push_back(keyframe);
                    }
                }
            }

            if (segment.duration <= 0.0 && ! segment.keyframes.empty()) {
                segment.duration = segment.keyframes.back().timestamp;
            }
            if (segment.duration <= 0.0 || segment.keyframes.empty()) {
                LOG_ERROR("Scene3DModelCameraPath: ignore empty segment path='%s' duration=%.3f "
                          "keyframes=%zu",
                          relative_path.c_str(),
                          segment.duration,
                          segment.keyframes.size());
                continue;
            }

            LOG_INFO("Scene3DModelCameraPath: segment parsed path='%s' index=%zu duration=%.3f "
                     "keyframes=%zu",
                     relative_path.c_str(),
                     scene.modelCameraPathSegments.size(),
                     segment.duration,
                     segment.keyframes.size());
            scene.modelCameraPathSegments.push_back(std::move(segment));
        }
    }

    scene.modelCameraPathEnabled = ! scene.modelCameraPathSegments.empty();
    if (scene.modelCameraPathEnabled) {
        const auto& first     = scene.modelCameraPathSegments.front().keyframes.front();
        auto        camera_it = scene.cameras.find(std::string(kSceneModelPerspectiveCameraName));
        if (camera_it != scene.cameras.end() && camera_it->second) {
            // Seed frame zero on the model-only camera. The legacy 2D `global_perspective` camera
            // is intentionally not touched here, because 2D particle scenes depend on its old
            // screen center transform.
            camera_it->second->SetExplicitView(
                Vector3d(first.eye[0], first.eye[1], first.eye[2]),
                Vector3d(first.center[0], first.center[1], first.center[2]),
                Vector3d(first.up[0], first.up[1], first.up[2]));
        }
        LOG_INFO("Scene3DModelCameraPath: enabled segments=%zu first-eye=[%.3f, %.3f, %.3f] "
                 "first-center=[%.3f, %.3f, %.3f]",
                 scene.modelCameraPathSegments.size(),
                 first.eye[0],
                 first.eye[1],
                 first.eye[2],
                 first.center[0],
                 first.center[1],
                 first.center[2]);
    }
}

void ParseCamera(ParseContext& context, const wpscene::WPScene& scene_config) {
    auto&       scene   = *context.scene;
    const auto& general = scene_config.general;
    // effect camera
    scene.cameras["effect"]    = std::make_shared<SceneCamera>(2, 2, -1.0f, 1.0f);
    context.effect_camera_node = std::make_shared<SceneNode>(); // at 0,0,0
    scene.cameras.at("effect")->AttatchNode(context.effect_camera_node);
    scene.sceneGraph->AppendChild(context.effect_camera_node);

    // global camera
    scene.cameras["global"] = std::make_shared<SceneCamera>((context.ortho_w / (i32)general.zoom),
                                                            (context.ortho_h / (i32)general.zoom),
                                                            -5000.0f,
                                                            5000.0f);
    scene.activeCamera      = scene.cameras.at("global").get();
    Vector3f cori { (float)context.ortho_w / 2.0f, (float)context.ortho_h / 2.0f, 0 },
        cscale { 1.0f, 1.0f, 1.0f }, cangle(Vector3f::Zero());

    context.global_camera_node = std::make_shared<SceneNode>(cori, cscale, cangle);
    scene.activeCamera->AttatchNode(context.global_camera_node);
    scene.sceneGraph->AppendChild(context.global_camera_node);
    scene.defaultGlobalCameraNode = context.global_camera_node;
    scene.defaultGlobalCameraZoom = std::max(0.0001f, general.zoom);

    scene.cameras["global_perspective"] =
        std::make_shared<SceneCamera>((float)context.ortho_w / (float)context.ortho_h,
                                      general.nearz,
                                      general.farz,
                                      algorism::ResolvePerspectiveFov(scene.perspectiveOverrideFov,
                                                                      context.ortho_h));

    Vector3f cperori                       = cori;
    cperori[2]                             = 1000.0f;
    context.global_perspective_camera_node = std::make_shared<SceneNode>(cperori, cscale, cangle);
    scene.cameras["global_perspective"]->AttatchNode(context.global_perspective_camera_node);
    scene.sceneGraph->AppendChild(context.global_perspective_camera_node);

    const Vector3d eye(
        scene_config.camera.eye[0], scene_config.camera.eye[1], scene_config.camera.eye[2]);
    const Vector3d center(scene_config.camera.center[0],
                          scene_config.camera.center[1],
                          scene_config.camera.center[2]);
    const Vector3d up(
        scene_config.camera.up[0], scene_config.camera.up[1], scene_config.camera.up[2]);
    scene.cameras[std::string(kSceneModelPerspectiveCameraName)] = std::make_shared<SceneCamera>(
        (float)context.ortho_w / (float)context.ortho_h, general.nearz, general.farz, general.fov);
    auto model_camera_node = std::make_shared<SceneNode>();
    // 3D model support must not reuse `global_perspective`: existing 2D particle systems and
    // camera-layer scenes already depend on that camera's historical centered-at-screen transform.
    // The authored scene camera is therefore installed under a model-only name and consumed only by
    // WPModelObject materialization and model camera-path playback.
    scene.cameras[std::string(kSceneModelPerspectiveCameraName)]->AttatchNode(model_camera_node);
    scene.cameras[std::string(kSceneModelPerspectiveCameraName)]->SetExplicitView(eye, center, up);
    scene.modelPerspectiveCameraName = std::string(kSceneModelPerspectiveCameraName);
    scene.sceneGraph->AppendChild(model_camera_node);
    LoadModelCameraPaths(context, scene_config.camera);
}

void InitContext(ParseContext& context, fs::VFS& vfs, wpscene::WPScene& sc,
                 std::string_view scene_id) {
    context.scene     = std::make_shared<Scene>();
    context.vfs       = &vfs;
    auto& scene       = *context.scene;
    scene.imageParser = std::make_unique<WPSyntheticImageParser>(
        std::make_unique<WPTexImageParser>(&vfs, std::string(scene_id)));
    scene.paritileSys->gener = std::make_unique<WPParticleRawGener>();
    scene.shaderValueUpdater = std::make_unique<WPShaderValueUpdater>(&scene);
    GenCardMesh(scene.default_effect_mesh, { 2, 2 });
    context.shader_updater = static_cast<WPShaderValueUpdater*>(scene.shaderValueUpdater.get());

    scene.clearColor                   = sc.general.clearcolor;
    scene.ambientColor                 = sc.general.ambientcolor;
    scene.skylightColor                = sc.general.skylightcolor;
    scene.bloom.enabled                = sc.general.bloom;
    scene.bloom.strength               = sc.general.bloomstrength;
    scene.bloom.threshold              = sc.general.bloomthreshold;
    scene.bloom.tint                   = sc.general.bloomtint;
    scene.bloom.hdr                    = sc.general.hdr;
    scene.bloom.hdrStrength            = sc.general.bloomhdrstrength;
    scene.bloom.hdrThreshold           = sc.general.bloomhdrthreshold;
    scene.bloom.hdrScatter             = sc.general.bloomhdrscatter;
    scene.bloom.hdrFeather             = sc.general.bloomhdrfeather;
    scene.bloom.hdrIterations          = sc.general.bloomhdriterations;
    scene.cameraParallax               = sc.general.cameraparallax;
    scene.cameraParallaxAmount         = sc.general.cameraparallaxamount;
    scene.cameraParallaxDelay          = sc.general.cameraparallaxdelay;
    scene.cameraParallaxMouseInfluence = sc.general.cameraparallaxmouseinfluence;
    scene.cameraOrthographic           = sc.general.isOrtho;
    scene.perspectiveOverrideFov       = sc.general.perspectiveoverridefov;
    scene.cameraShake                  = sc.general.camerashake;
    scene.cameraShakeAmplitude         = sc.general.camerashakeamplitude;
    scene.cameraShakeRoughness         = sc.general.camerashakeroughness;
    scene.cameraShakeSpeed             = sc.general.camerashakespeed;
    scene.ortho[0]                     = sc.general.orthogonalprojection.width;
    scene.ortho[1]                     = sc.general.orthogonalprojection.height;
    context.ortho_w                    = scene.ortho[0];
    context.ortho_h                    = scene.ortho[1];

    PopulateGlobalBaseUniforms(context, scene);

    {
        WPCameraParallax cam_para;
        cam_para.enable         = sc.general.cameraparallax;
        cam_para.amount         = sc.general.cameraparallaxamount;
        cam_para.delay          = sc.general.cameraparallaxdelay;
        cam_para.mouseinfluence = sc.general.cameraparallaxmouseinfluence;
        context.shader_updater->SetCameraParallax(cam_para);
    }
    if (scene.cameraShake) {
        LOG_INFO("SceneCameraShake: enabled=true amplitude=%.3f roughness=%.3f speed=%.3f "
                 "ortho=%s height=%d",
                 scene.cameraShakeAmplitude,
                 scene.cameraShakeRoughness,
                 scene.cameraShakeSpeed,
                 scene.cameraOrthographic ? "true" : "false",
                 scene.ortho[1]);
    }
}

void ClearSceneBloomGraph(Scene& scene) {
    for (const auto& output : scene.bloom.outputs) {
        if (output.rfind("__hanabi_scene_bloom", 0) == 0 || output.rfind("_rt_hdr_bloom", 0) == 0 ||
            output == "_rt_Bloom") {
            scene.renderTargets.erase(output);
            scene.pendingRenderTargetReleaseKeys.insert(output);
        }
    }
    scene.bloom.nodes.clear();
    scene.bloom.outputs.clear();
    scene.bloom.node.reset();
    scene.bloom.built_quality = 0;
}

bool ConfigureSceneHdrBloomPass(ParseContext& context) {
    auto& scene = *context.scene;
    const i32 scene_width  = std::max(1, context.ortho_w);
    const i32 scene_height = std::max(1, context.ortho_h);
    const int levels =
        std::max(2, scene.bloom.hdrIterations > 0 ? scene.bloom.hdrIterations : 2);
    const bool display_hdr = scene.bloom.quality >= 3;
    ClearSceneBloomGraph(scene);

    constexpr std::string_view fullscreen_vertex_source = R"(
attribute vec3 a_Position;
attribute vec2 a_TexCoord;
varying vec2 v_TexCoord;
void main() {
    gl_Position = vec4(a_Position, 1.0);
    v_TexCoord = a_TexCoord;
}
)";
    constexpr std::string_view hdr_downsample_fragment_source = R"(
varying vec2 v_TexCoord;
uniform sampler2D g_Texture0;
uniform vec4 g_RenderVar0;
#if BICUBIC
vec4 cubic(float v)
{
    vec4 n = vec4(1.0, 2.0, 3.0, 4.0) - v;
    vec4 s = n * n * n;
    float x = s.x;
    float y = s.y - 4.0 * s.x;
    float z = s.z - 4.0 * s.y + 6.0 * s.x;
    float w = 6.0 - x - y - z;
    return vec4(x, y, z, w) * (1.0/6.0);
}
vec4 textureBicubic(vec2 texCoords)
{
    float sc = 0.5;
    vec2 texSize = sc / g_RenderVar0.xy;
    vec2 invTexSize = g_RenderVar0.xy / sc;
    texCoords = texCoords * texSize - 0.5;
    vec2 fxy = frac(texCoords);
    texCoords -= fxy;
    vec4 xcubic = cubic(fxy.x);
    vec4 ycubic = cubic(fxy.y);
    vec4 c = texCoords.xxyy + vec2 (-0.5, +1.5).xyxy;
    vec4 s = vec4(xcubic.xz + xcubic.yw, ycubic.xz + ycubic.yw);
    vec4 offset = c + vec4 (xcubic.yw, ycubic.yw) / s;
    offset *= invTexSize.xxyy;
    vec4 sample0 = texSample2D(g_Texture0, offset.xz);
    vec4 sample1 = texSample2D(g_Texture0, offset.yz);
    vec4 sample2 = texSample2D(g_Texture0, offset.xw);
    vec4 sample3 = texSample2D(g_Texture0, offset.yw);
    float sx = s.x / (s.x + s.y);
    float sy = s.z / (s.z + s.w);
    return mix(mix(sample3, sample2, sx), mix(sample1, sample0, sx), sy);
}
#endif
#if BLOOM
uniform float g_BloomStrength;
uniform vec4 g_BloomBlendParams;
uniform vec3 g_BloomTint;
#endif
#if UPSAMPLE
uniform float g_BloomScatter;
#endif
void main() {
#if BICUBIC
    vec3 albedo = textureBicubic(v_TexCoord + g_RenderVar0.xy).rgb +
                    textureBicubic(v_TexCoord + g_RenderVar0.zy).rgb +
                    textureBicubic(v_TexCoord + g_RenderVar0.xw).rgb +
                    textureBicubic(v_TexCoord + g_RenderVar0.zw).rgb;
#else
    vec3 albedo = texSample2D(g_Texture0, v_TexCoord + g_RenderVar0.xy).rgb +
                    texSample2D(g_Texture0, v_TexCoord + g_RenderVar0.zy).rgb +
                    texSample2D(g_Texture0, v_TexCoord + g_RenderVar0.xw).rgb +
                    texSample2D(g_Texture0, v_TexCoord + g_RenderVar0.zw).rgb;
#endif
#if UPSAMPLE
    albedo *= 0.25 * g_BloomScatter;
#else
    albedo *= 0.25;
#endif
#if BLOOM
    albedo = max(CAST3(0), albedo);
    float brightness = max(albedo.r, max(albedo.g, albedo.b));
    float soft = brightness - g_BloomBlendParams.y;
    soft = clamp(soft, 0.0, g_BloomBlendParams.z);
    soft = soft * soft * g_BloomBlendParams.w;
    float contribution = max(soft, brightness - g_BloomBlendParams.x);
    contribution /= max(brightness, 0.00001);
    albedo *= contribution * g_BloomStrength * g_BloomTint;
#endif
    gl_FragColor = vec4(albedo, 1.0);
}
)";
    constexpr std::string_view combine_hdr_fragment_source = R"(
varying vec2 v_TexCoord;
uniform sampler2D g_Texture0;
uniform sampler2D g_Texture1;
uniform vec2 g_TexelSize;
uniform vec4 g_RenderVar0;
vec3 lin(vec3 v)
{
    vec3 c = step(0.04045, v);
    return c * (pow((v + 0.055) / 1.055, CAST3(2.4))) + (1.0 - c) * (v / 12.92);
}
void main()
{
    vec3 albedo = texSample2D(g_Texture0, v_TexCoord).rgb;
    vec3 bloom1 = texSample2D(g_Texture1, v_TexCoord + g_TexelSize).rgb +
                texSample2D(g_Texture1, v_TexCoord - g_TexelSize).rgb +
                texSample2D(g_Texture1, v_TexCoord + vec2(g_TexelSize.x, -g_TexelSize.y)).rgb +
                texSample2D(g_Texture1, v_TexCoord + vec2(-g_TexelSize.x, g_TexelSize.y)).rgb;
    bloom1 *= 0.25;
#if DISPLAYHDR == 1
    albedo = saturate(albedo);
    albedo += bloom1;
    float hdrFactors = g_RenderVar0.y * smoothstep(1.0, 5.0, dot(vec3(0.299, 0.587, 0.114), albedo)) + g_RenderVar0.x;
    gl_FragColor = vec4(lin((max(CAST3(0.0), albedo))) * hdrFactors, 1.0);
#else
    albedo += bloom1;
    gl_FragColor = vec4(saturate(lin(albedo)) * g_RenderVar0.x, 1.0);
#endif
}
)";

    const auto compile_shader = [&](std::string      name,
                                    std::string_view vertex_source,
                                    std::string_view fragment_source,
                                    usize            texture_count,
                                    Combos           combos) -> std::shared_ptr<SceneShader> {
        auto shader  = std::make_shared<SceneShader>();
        shader->name = std::move(name);
        WPShaderInfo                 shader_info;
        shader_info.combos           = std::move(combos);
        std::array                   units { WPShaderUnit {
                                                 .stage           = ShaderType::VERTEX,
                                                 .src             = std::string(vertex_source),
                                                 .preprocess_info = {},
                                                 .debug_name      = shader->name + ".vert",
                                             },
                                             WPShaderUnit {
                                                 .stage           = ShaderType::FRAGMENT,
                                                 .src             = std::string(fragment_source),
                                                 .preprocess_info = {},
                                                 .debug_name      = shader->name + ".frag",
                                             } };
        std::vector<WPShaderTexInfo> texinfos(texture_count, WPShaderTexInfo { .enabled = true });
        for (auto& unit : units) {
            unit.src = WPShaderParser::PreShaderSrc(*context.vfs, unit.src, &shader_info, texinfos);
        }
        shader->default_uniforms = shader_info.svs;
        if (! WPShaderParser::CompileToSpv(
                scene.scene_id, units, shader->codes, *context.vfs, &shader_info, texinfos)) {
            LOG_ERROR("SceneBloomConfig: HDR compile failed pass='%s'", shader->name.c_str());
            return nullptr;
        }
        return shader;
    };

    const auto make_node = [&](std::string name,
                               std::shared_ptr<SceneShader>
                                   shader,
                               std::vector<std::string>
                                            textures,
                               ShaderValues const_values,
                               BlendMode    blend) -> std::shared_ptr<SceneNode> {
        SceneMaterial material;
        material.name     = name;
        material.textures = std::move(textures);
        material.defines.reserve(material.textures.size());
        for (usize i = 0; i < material.textures.size(); ++i) {
            material.defines.push_back("g_Texture" + std::to_string(i));
        }
        material.customShader.shader      = std::move(shader);
        material.customShader.constValues = std::move(const_values);
        material.blenmode                 = blend;
        auto mesh = std::make_shared<SceneMesh>();
        mesh->ChangeMeshDataFrom(scene.default_effect_mesh);
        mesh->AddMaterial(std::move(material));
        auto node = std::make_shared<SceneNode>();
        node->SetName(name);
        node->AddMesh(mesh);
        scene.nodeOwners[node.get()] = 0;
        context.shader_updater->SetNodeData(node.get(), WPShaderValueData {});
        return node;
    };

    auto extract_shader = compile_shader("__hanabi_scene_hdr_extract",
                                         fullscreen_vertex_source,
                                         hdr_downsample_fragment_source,
                                         1,
                                         {{"BLOOM", "1"}});
    auto down_shader    = compile_shader("__hanabi_scene_hdr_down",
                                         fullscreen_vertex_source,
                                         hdr_downsample_fragment_source,
                                         1,
                                         {});
    auto up_shader      = compile_shader("__hanabi_scene_hdr_up",
                                         fullscreen_vertex_source,
                                         hdr_downsample_fragment_source,
                                         1,
                                         {{"UPSAMPLE", "1"}});
    auto up_cubic_shader = compile_shader("__hanabi_scene_hdr_up_cubic",
                                          fullscreen_vertex_source,
                                          hdr_downsample_fragment_source,
                                          1,
                                          {{"UPSAMPLE", "1"}, {"BICUBIC", "1"}});
    auto combine_shader = compile_shader("__hanabi_scene_hdr_combine",
                                         fullscreen_vertex_source,
                                         combine_hdr_fragment_source,
                                         2,
                                         {{"DISPLAYHDR", display_hdr ? "1" : "0"}});
    if (extract_shader == nullptr || down_shader == nullptr || up_shader == nullptr ||
        up_cubic_shader == nullptr || combine_shader == nullptr) {
        ClearSceneBloomGraph(scene);
        return false;
    }

    const float threshold = scene.bloom.hdrThreshold;
    const float feather   = scene.bloom.hdrFeather;
    const float scatter   = scene.bloom.hdrScatter;
    const float knee      = threshold * feather;
    const std::array<float, 4> blend_params {
        threshold,
        threshold - knee,
        knee + knee,
        0.25f / (knee + 1.0e-5f),
    };
    // Dual-filter upsample is additive. Extract gain is
    // strength / (scatter^(levels-2) + 1) so scatter > 1 does not stack raw energy.
    const float extract_strength =
        scene.bloom.hdrStrength / (std::pow(scatter, static_cast<float>(levels - 2)) + 1.0f);

    std::vector<std::string> mip_targets;
    mip_targets.reserve(static_cast<usize>(levels));
    for (int i = 0; i < levels; ++i) {
        const int   denom  = 2 << i;
        const i32   width  = std::max(1, scene_width / denom);
        const i32   height = std::max(1, scene_height / denom);
        const float scale  = 1.0f / static_cast<float>(denom);
        std::string target = (i == 0) ? std::string("_rt_Bloom")
                                      : ("_rt_hdr_bloom_" + std::to_string(i));
        scene.renderTargets[target] = {
            .width     = width,
            .height    = height,
            .mapWidth  = width,
            .mapHeight = height,
            .bind      = { .enable = true, .name = SpecTex_Default.data(), .scale = scale },
        };
        mip_targets.push_back(std::move(target));
    }

    const auto texel4 = [](i32 w, i32 h) -> std::array<float, 4> {
        const float tx = 1.0f / static_cast<float>(std::max(1, w));
        const float ty = 1.0f / static_cast<float>(std::max(1, h));
        return { tx, ty, -tx, -ty };
    };

    scene.bloom.nodes.clear();
    scene.bloom.outputs.clear();

    ShaderValues extract_values;
    extract_values["g_RenderVar0"]       = ShaderValue(texel4(scene_width, scene_height));
    extract_values["g_BloomStrength"]    = ShaderValue(extract_strength);
    extract_values["g_BloomBlendParams"] = ShaderValue(blend_params);
    extract_values["g_BloomTint"]        = ShaderValue(scene.bloom.tint);
    auto extract_node                    = make_node("__hanabi_scene_hdr_extract",
                                  extract_shader,
                                  { SpecTex_Default.data() },
                                  std::move(extract_values),
                                  BlendMode::Disable);
    scene.bloom.nodes.push_back(extract_node);
    scene.bloom.outputs.push_back(mip_targets[0]);
    scene.bloom.node = extract_node;

    for (int i = 1; i < levels; ++i) {
        const i32 src_w = std::max(1, scene_width / (2 << (i - 1)));
        const i32 src_h = std::max(1, scene_height / (2 << (i - 1)));
        ShaderValues down_values;
        down_values["g_RenderVar0"] = ShaderValue(texel4(src_w, src_h));
        auto down_node              = make_node("__hanabi_scene_hdr_down_" + std::to_string(i),
                                   down_shader,
                                   { mip_targets[static_cast<usize>(i - 1)] },
                                   std::move(down_values),
                                   BlendMode::Disable);
        scene.bloom.nodes.push_back(down_node);
        scene.bloom.outputs.push_back(mip_targets[static_cast<usize>(i)]);
    }

    for (int i = levels - 2; i >= 0; --i) {
        const i32 src_w = std::max(1, scene_width / (2 << (i + 1)));
        const i32 src_h = std::max(1, scene_height / (2 << (i + 1)));
        ShaderValues up_values;
        up_values["g_RenderVar0"]    = ShaderValue(texel4(src_w, src_h));
        up_values["g_BloomScatter"]  = ShaderValue(scatter);
        const bool cubic             = (i == 0);
        auto up_node                 = make_node(cubic ? "__hanabi_scene_hdr_up_cubic"
                                                      : ("__hanabi_scene_hdr_up_" + std::to_string(i)),
                                cubic ? up_cubic_shader : up_shader,
                                { mip_targets[static_cast<usize>(i + 1)] },
                                std::move(up_values),
                                BlendMode::Additive);
        scene.bloom.nodes.push_back(up_node);
        scene.bloom.outputs.push_back(mip_targets[static_cast<usize>(i)]);
    }

    const i32 bloom_w = std::max(1, scene_width / 2);
    const i32 bloom_h = std::max(1, scene_height / 2);
    ShaderValues combine_values;
    combine_values["g_TexelSize"]   = ShaderValue(std::array<float, 2> {
        1.0f / static_cast<float>(bloom_w),
        1.0f / static_cast<float>(bloom_h),
    });
    combine_values["g_RenderVar0"]  = ShaderValue(std::array<float, 4> { 1.0f, 0.0f, 0.0f, 0.0f });
    auto combine_node               = make_node("__hanabi_scene_hdr_combine",
                                  combine_shader,
                                  { SpecTex_Default.data(), mip_targets[0] },
                                  std::move(combine_values),
                                  BlendMode::Disable);
    scene.bloom.nodes.push_back(combine_node);
    scene.bloom.outputs.push_back(SpecTex_Default.data());
    scene.bloom.built_quality = scene.bloom.quality;

    LOG_INFO("ScenePostProcess: quality=%d path=%s passes=%zu levels=%d "
             "mip0=%dx%d strength=%.3f extract=%.5f threshold=%.3f "
             "scatter=%.3f feather=%.3f",
             scene.bloom.quality,
             display_hdr ? "displayhdr" : "hdr",
             scene.bloom.nodes.size(),
             levels,
             bloom_w,
             bloom_h,
             scene.bloom.hdrStrength,
             extract_strength,
             scene.bloom.hdrThreshold,
             scatter,
             scene.bloom.hdrFeather);
    return true;
}

bool ConfigureSceneBloomPass(ParseContext& context) {
    if (context.scene == nullptr || context.vfs == nullptr) return false;

    auto& scene = *context.scene;
    scene.bloom.quality = std::clamp(scene.bloom.quality, 0, 3);
    if (scene.bloom.quality <= 0) {
        ClearSceneBloomGraph(scene);
        LOG_INFO("ScenePostProcess: quality=0 path=none passes=0");
        return false;
    }
    const bool use_hdr = scene.bloom.quality >= 2 && scene.bloom.hdr && scene.bloom.enabled;
    if (use_hdr) {
        return ConfigureSceneHdrBloomPass(context);
    }
    // Build the LDR Bloom node even when the authored user toggle currently disables it, as long
    // as the scene carries non-zero Bloom settings. Runtime toggles can then update `u_enabled`
    // in place. Ultra/displayhdr with authored HDR uses the separate HDR chain above.
    const bool has_ldr_bloom_work = scene.bloom.enabled || scene.bloom.strength > 0.0f;
    if (! has_ldr_bloom_work) {
        LOG_INFO("SceneBloomConfig: enabled=%s strength=%.3f threshold=%.3f "
                 "hdr-requested=%s render-hdr=false active=false",
                 scene.bloom.enabled ? "true" : "false",
                 scene.bloom.strength,
                 scene.bloom.threshold,
                 scene.bloom.hdr ? "true" : "false");
        return false;
    }

    ClearSceneBloomGraph(scene);
    const i32 scene_width    = std::max(1, context.ortho_w);
    const i32 scene_height   = std::max(1, context.ortho_h);
    const i32 quarter_width  = std::max(1, scene_width / 4);
    const i32 quarter_height = std::max(1, scene_height / 4);
    const i32 eighth_width   = std::max(1, scene_width / 8);
    const i32 eighth_height  = std::max(1, scene_height / 8);

    constexpr std::string_view quarter_target = "__hanabi_scene_bloom_quarter";
    constexpr std::string_view eighth_target  = "__hanabi_scene_bloom_eighth";
    constexpr std::string_view blur_target    = "__hanabi_scene_bloom_blur";

    // Wallpaper Engine's scene Bloom is implemented by the stock utility assets as a four-pass
    // render-target chain: quarter-size extraction, eighth-size horizontal blur, eighth-size
    // vertical blur, then additive combine into `_rt_default`. Rebuilding that topology here keeps
    // the high-channel clipping and pink highlight rolloff aligned with the Windows renderer,
    // instead of relying on a hand-tuned single-pass approximation.
    scene.renderTargets[std::string(quarter_target)] = {
        .width     = quarter_width,
        .height    = quarter_height,
        .mapWidth  = quarter_width,
        .mapHeight = quarter_height,
        .bind      = { .enable = true, .name = SpecTex_Default.data(), .scale = 0.25 },
    };
    scene.renderTargets[std::string(eighth_target)] = {
        .width     = eighth_width,
        .height    = eighth_height,
        .mapWidth  = eighth_width,
        .mapHeight = eighth_height,
        .bind      = { .enable = true, .name = SpecTex_Default.data(), .scale = 0.125 },
    };
    scene.renderTargets[std::string(blur_target)] = {
        .width     = eighth_width,
        .height    = eighth_height,
        .mapWidth  = eighth_width,
        .mapHeight = eighth_height,
        .bind      = { .enable = true, .name = SpecTex_Default.data(), .scale = 0.125 },
    };

    constexpr std::string_view fullscreen_vertex_source           = R"(
attribute vec3 a_Position;
attribute vec2 a_TexCoord;

varying vec2 v_TexCoord;

void main() {
    gl_Position = vec4(a_Position, 1.0);
    v_TexCoord = a_TexCoord;
}
)";
    constexpr std::string_view downsample_quarter_vertex_source   = R"(
attribute vec3 a_Position;
attribute vec2 a_TexCoord;

uniform vec2 g_TexelSize;

varying vec2 v_TexCoord[4];

void main() {
    gl_Position = vec4(a_Position, 1.0);
    v_TexCoord[0] = a_TexCoord - g_TexelSize;
    v_TexCoord[1] = a_TexCoord + g_TexelSize;
    v_TexCoord[2] = a_TexCoord + vec2(-g_TexelSize.x, g_TexelSize.y);
    v_TexCoord[3] = a_TexCoord + vec2(g_TexelSize.x, -g_TexelSize.y);
}
)";
    constexpr std::string_view downsample_quarter_fragment_source = R"(
varying vec2 v_TexCoord[4];

uniform sampler2D g_Texture0;

uniform float u_enabled; // {"material":"Bloom enabled","default":0,"range":[0,1]}
uniform float g_BloomStrength; // {"material":"bloomstrength","default":2,"range":[0,4]}
uniform float g_BloomThreshold; // {"material":"bloomthreshold","default":0.65,"range":[0,0.999]}
uniform vec3 g_BloomTint; // {"material":"bloomtint","default":"1 1 1"}

void main() {
    // Keep the generated render graph stable for runtime user-property toggles. A disabled scene
    // Bloom still writes black into the private Bloom targets so the later additive combine becomes
    // visually neutral without needing to destroy and rebuild graph nodes.
    if (u_enabled <= 0.0 || g_BloomStrength <= 0.0) {
        gl_FragColor = vec4(CAST3(0), 1.0);
        return;
    }

    vec3 albedo = texSample2D(g_Texture0, v_TexCoord[0]).rgb +
                    texSample2D(g_Texture0, v_TexCoord[1]).rgb +
                    texSample2D(g_Texture0, v_TexCoord[2]).rgb +
                    texSample2D(g_Texture0, v_TexCoord[3]).rgb;
    albedo *= 0.25;

    float scale = max(max(albedo.x, albedo.y), albedo.z);
    albedo *= saturate(scale - g_BloomThreshold);

    float grayscale = dot(vec3(0.2989, 0.5870, 0.1140), albedo);
    float sat = 1.0;
    albedo = -grayscale * sat + albedo * (1.0 + sat);

    gl_FragColor = vec4(max(CAST3(0), albedo * g_BloomStrength * g_BloomTint), 1.0);
}
)";
    constexpr std::string_view blur_x_vertex_source               = R"(
attribute vec3 a_Position;
attribute vec2 a_TexCoord;

uniform vec2 g_TexelSize;

varying vec2 v_TexCoord[13];

void main() {
    gl_Position = vec4(a_Position, 1);

    float localTexel = g_TexelSize.x * 8.0;
    v_TexCoord[0] = vec2(a_TexCoord.x - localTexel * 6.0, a_TexCoord.y);
    v_TexCoord[1] = vec2(a_TexCoord.x - localTexel * 5.0, a_TexCoord.y);
    v_TexCoord[2] = vec2(a_TexCoord.x - localTexel * 4.0, a_TexCoord.y);
    v_TexCoord[3] = vec2(a_TexCoord.x - localTexel * 3.0, a_TexCoord.y);
    v_TexCoord[4] = vec2(a_TexCoord.x - localTexel * 2.0, a_TexCoord.y);
    v_TexCoord[5] = vec2(a_TexCoord.x - localTexel * 1.0, a_TexCoord.y);
    v_TexCoord[6] = a_TexCoord;
    v_TexCoord[7] = vec2(a_TexCoord.x + localTexel * 1.0, a_TexCoord.y);
    v_TexCoord[8] = vec2(a_TexCoord.x + localTexel * 2.0, a_TexCoord.y);
    v_TexCoord[9] = vec2(a_TexCoord.x + localTexel * 3.0, a_TexCoord.y);
    v_TexCoord[10] = vec2(a_TexCoord.x + localTexel * 4.0, a_TexCoord.y);
    v_TexCoord[11] = vec2(a_TexCoord.x + localTexel * 5.0, a_TexCoord.y);
    v_TexCoord[12] = vec2(a_TexCoord.x + localTexel * 6.0, a_TexCoord.y);
}
)";
    constexpr std::string_view blur_y_vertex_source               = R"(
attribute vec3 a_Position;
attribute vec2 a_TexCoord;

uniform vec2 g_TexelSize;

varying vec2 v_TexCoord[13];

void main() {
    gl_Position = vec4(a_Position, 1);

    float localTexel = g_TexelSize.y * 8.0;
    v_TexCoord[0] = vec2(a_TexCoord.x, a_TexCoord.y - localTexel * 6.0);
    v_TexCoord[1] = vec2(a_TexCoord.x, a_TexCoord.y - localTexel * 5.0);
    v_TexCoord[2] = vec2(a_TexCoord.x, a_TexCoord.y - localTexel * 4.0);
    v_TexCoord[3] = vec2(a_TexCoord.x, a_TexCoord.y - localTexel * 3.0);
    v_TexCoord[4] = vec2(a_TexCoord.x, a_TexCoord.y - localTexel * 2.0);
    v_TexCoord[5] = vec2(a_TexCoord.x, a_TexCoord.y - localTexel * 1.0);
    v_TexCoord[6] = a_TexCoord;
    v_TexCoord[7] = vec2(a_TexCoord.x, a_TexCoord.y + localTexel * 1.0);
    v_TexCoord[8] = vec2(a_TexCoord.x, a_TexCoord.y + localTexel * 2.0);
    v_TexCoord[9] = vec2(a_TexCoord.x, a_TexCoord.y + localTexel * 3.0);
    v_TexCoord[10] = vec2(a_TexCoord.x, a_TexCoord.y + localTexel * 4.0);
    v_TexCoord[11] = vec2(a_TexCoord.x, a_TexCoord.y + localTexel * 5.0);
    v_TexCoord[12] = vec2(a_TexCoord.x, a_TexCoord.y + localTexel * 6.0);
}
)";
    constexpr std::string_view blur_fragment_source               = R"(
varying vec2 v_TexCoord[13];

uniform sampler2D g_Texture0;

void main() {
    vec3 albedo = texSample2D(g_Texture0, v_TexCoord[0]).rgb * 0.006299 +
                    texSample2D(g_Texture0, v_TexCoord[1]).rgb * 0.017298 +
                    texSample2D(g_Texture0, v_TexCoord[2]).rgb * 0.039533 +
                    texSample2D(g_Texture0, v_TexCoord[3]).rgb * 0.075189 +
                    texSample2D(g_Texture0, v_TexCoord[4]).rgb * 0.119007 +
                    texSample2D(g_Texture0, v_TexCoord[5]).rgb * 0.156756 +
                    texSample2D(g_Texture0, v_TexCoord[6]).rgb * 0.171834 +
                    texSample2D(g_Texture0, v_TexCoord[7]).rgb * 0.156756 +
                    texSample2D(g_Texture0, v_TexCoord[8]).rgb * 0.119007 +
                    texSample2D(g_Texture0, v_TexCoord[9]).rgb * 0.075189 +
                    texSample2D(g_Texture0, v_TexCoord[10]).rgb * 0.039533 +
                    texSample2D(g_Texture0, v_TexCoord[11]).rgb * 0.017298 +
                    texSample2D(g_Texture0, v_TexCoord[12]).rgb * 0.006299;

    gl_FragColor = vec4(albedo, 1.0);
}
)";
    constexpr std::string_view combine_fragment_source            = R"(
varying vec2 v_TexCoord;

uniform sampler2D g_Texture0;
uniform sampler2D g_Texture1;

void main() {
    vec3 albedo = texSample2D(g_Texture0, v_TexCoord).rgb;
    vec3 bloom = texSample2D(g_Texture1, v_TexCoord).rgb;
    albedo += bloom;

    gl_FragColor = vec4(albedo, 1.0);
}
)";

    const auto compile_shader = [&](std::string      name,
                                    std::string_view vertex_source,
                                    std::string_view fragment_source,
                                    usize texture_count) -> std::shared_ptr<SceneShader> {
        auto shader  = std::make_shared<SceneShader>();
        shader->name = std::move(name);

        WPShaderInfo                 shader_info;
        std::array                   units { WPShaderUnit {
                                                 .stage           = ShaderType::VERTEX,
                                                 .src             = std::string(vertex_source),
                                                 .preprocess_info = {},
                                                 .debug_name      = shader->name + ".vert",
                                             },
                                             WPShaderUnit {
                                                 .stage           = ShaderType::FRAGMENT,
                                                 .src             = std::string(fragment_source),
                                                 .preprocess_info = {},
                                                 .debug_name      = shader->name + ".frag",
                                             } };
        std::vector<WPShaderTexInfo> texinfos(texture_count, WPShaderTexInfo { .enabled = true });
        for (auto& unit : units) {
            unit.src = WPShaderParser::PreShaderSrc(*context.vfs, unit.src, &shader_info, texinfos);
        }
        shader->default_uniforms = shader_info.svs;
        if (! WPShaderParser::CompileToSpv(
                scene.scene_id, units, shader->codes, *context.vfs, &shader_info, texinfos)) {
            LOG_ERROR("SceneBloomConfig: compile failed pass='%s'", shader->name.c_str());
            return nullptr;
        }
        return shader;
    };

    const std::array<float, 2> scene_texel_size {
        1.0f / static_cast<float>(scene_width),
        1.0f / static_cast<float>(scene_height),
    };

    auto downsample_shader = compile_shader("__hanabi_scene_bloom_downsample_quarter",
                                            downsample_quarter_vertex_source,
                                            downsample_quarter_fragment_source,
                                            1);
    auto blur_x_shader     = compile_shader("__hanabi_scene_bloom_downsample_eighth_blur",
                                            blur_x_vertex_source,
                                            blur_fragment_source,
                                            1);
    auto blur_y_shader =
        compile_shader("__hanabi_scene_bloom_blur", blur_y_vertex_source, blur_fragment_source, 1);
    auto combine_shader = compile_shader(
        "__hanabi_scene_bloom_combine", fullscreen_vertex_source, combine_fragment_source, 2);
    if (downsample_shader == nullptr || blur_x_shader == nullptr || blur_y_shader == nullptr ||
        combine_shader == nullptr) {
        return false;
    }

    const auto make_node = [&](std::string name,
                               std::shared_ptr<SceneShader>
                                   shader,
                               std::vector<std::string>
                                            textures,
                               ShaderValues const_values) -> std::shared_ptr<SceneNode> {
        SceneMaterial material;
        material.name     = name;
        material.textures = std::move(textures);
        material.defines.reserve(material.textures.size());
        for (usize i = 0; i < material.textures.size(); ++i) {
            material.defines.push_back("g_Texture" + std::to_string(i));
        }
        material.customShader.shader      = std::move(shader);
        material.customShader.constValues = std::move(const_values);
        material.blenmode                 = BlendMode::Disable;

        auto mesh = std::make_shared<SceneMesh>();
        mesh->ChangeMeshDataFrom(scene.default_effect_mesh);
        mesh->AddMaterial(std::move(material));

        auto node = std::make_shared<SceneNode>();
        node->SetName(name);
        node->AddMesh(mesh);
        scene.nodeOwners[node.get()] = 0;
        context.shader_updater->SetNodeData(node.get(), WPShaderValueData {});
        return node;
    };

    ShaderValues downsample_values;
    downsample_values["g_TexelSize"]      = ShaderValue(scene_texel_size);
    downsample_values["u_enabled"]        = ShaderValue(scene.bloom.enabled ? 1.0f : 0.0f);
    downsample_values["g_BloomStrength"]  = ShaderValue(scene.bloom.strength);
    downsample_values["g_BloomThreshold"] = ShaderValue(scene.bloom.threshold);
    downsample_values["g_BloomTint"]      = ShaderValue(scene.bloom.tint);

    ShaderValues blur_values;
    blur_values["g_TexelSize"] = ShaderValue(scene_texel_size);

    auto downsample_node = make_node("__hanabi_scene_bloom_downsample_quarter",
                                     downsample_shader,
                                     { SpecTex_Default.data() },
                                     std::move(downsample_values));
    auto blur_x_node     = make_node("__hanabi_scene_bloom_downsample_eighth_blur",
                                     blur_x_shader,
                                     { std::string(quarter_target) },
                                     blur_values);
    auto blur_y_node     = make_node(
        "__hanabi_scene_bloom_blur", blur_y_shader, { std::string(eighth_target) }, blur_values);
    auto combine_node = make_node("__hanabi_scene_bloom_combine",
                                  combine_shader,
                                  { SpecTex_Default.data(), std::string(blur_target) },
                                  {});

    scene.bloom.node    = downsample_node;
    scene.bloom.nodes   = { downsample_node, blur_x_node, blur_y_node, combine_node };
    scene.bloom.outputs = {
        std::string(quarter_target),
        std::string(eighth_target),
        std::string(blur_target),
        SpecTex_Default.data(),
    };

    scene.bloom.built_quality = scene.bloom.quality;
    LOG_INFO("ScenePostProcess: quality=%d path=ldr passes=%zu quarter=%dx%d eighth=%dx%d",
             scene.bloom.quality,
             scene.bloom.nodes.size(),
             quarter_width,
             quarter_height,
             eighth_width,
             eighth_height);
    LOG_INFO("SceneBloomConfig: enabled=%s strength=%.3f threshold=%.3f tint=[%.3f,%.3f,%.3f] "
             "hdr-requested=%s render-hdr=false hdr-strength=%.3f hdr-threshold=%.3f "
             "hdr-scatter=%.3f hdr-feather=%.3f hdr-iterations=%d active=true passes=%zu "
             "quarter=%dx%d eighth=%dx%d",
             scene.bloom.enabled ? "true" : "false",
             scene.bloom.strength,
             scene.bloom.threshold,
             scene.bloom.tint[0],
             scene.bloom.tint[1],
             scene.bloom.tint[2],
             scene.bloom.hdr ? "true" : "false",
             scene.bloom.hdrStrength,
             scene.bloom.hdrThreshold,
             scene.bloom.hdrScatter,
             scene.bloom.hdrFeather,
             scene.bloom.hdrIterations,
             scene.bloom.nodes.size(),
             quarter_width,
             quarter_height,
             eighth_width,
             eighth_height);
    return true;
}

bool SceneHasShadowLights(const Scene& scene) {
    for (char flag : scene.lighting.point_shadow) {
        if (flag) return true;
    }
    for (char flag : scene.lighting.spot_shadow) {
        if (flag) return true;
    }
    for (char flag : scene.lighting.directional_shadow) {
        if (flag) return true;
    }
    return false;
}

LightingV1Desc LightingDescFromScene(const Scene& scene) {
    LightingV1Desc desc;
    desc.point               = scene.lighting.point;
    desc.spot                = scene.lighting.spot;
    desc.directional         = scene.lighting.directional;
    desc.tube                = scene.lighting.tube;
    desc.shadows             = scene.shadows.quality != 0;
    desc.point_shadow        = scene.lighting.point_shadow;
    desc.spot_shadow         = scene.lighting.spot_shadow;
    desc.spot_cookie         = scene.lighting.spot_cookie;
    desc.directional_shadow  = scene.lighting.directional_shadow;
    return desc;
}

void ConfigureSceneShadows(Scene& scene) {
    for (auto& light : scene.lights) {
        if (light) light->clearShadowAtlasSlot();
    }
    const int quality = scene.shadows.quality;
    scene.shadows.built_quality = quality;
    scene.shadows.atlas_active  = false;
    if (quality <= 0) {
        scene.renderTargets[std::string(SpecTex_ShadowAtlas)] = {
            .width           = 2,
            .height          = 2,
            .mapWidth        = 2,
            .mapHeight       = 2,
            .allowReuse      = false,
            .comparisonDepth = true,
        };
        return;
    }

    const int face = ShadowFaceSize(quality);
    std::vector<SceneLight*> point_casters;
    std::vector<SceneLight*> spot_casters;
    std::vector<SceneLight*> dir_casters;
    for (auto& light : scene.lights) {
        if (! light || ! light->castsShadows()) continue;
        if (light->type() == SceneLightType::Point) point_casters.push_back(light.get());
        else if (light->type() == SceneLightType::Spot) spot_casters.push_back(light.get());
        else if (light->type() == SceneLightType::Directional) dir_casters.push_back(light.get());
    }
    const int point_n    = static_cast<int>(point_casters.size());
    const int spot_n     = static_cast<int>(spot_casters.size());
    const int dir_n      = static_cast<int>(dir_casters.size());
    const int cascade_n  = dir_n * 3;
    const int slots      = point_n + spot_n + cascade_n;
    const int n          = std::max(1, slots);
    const int w          = std::max(2, n * face);
    const int h          = std::max(2, face);
    int       cursor     = 0;
    auto pack = [&](SceneLight& light, bool point, bool directional, int cascade_index) {
        SceneLight::ShadowAtlasSlot slot;
        slot.packed         = true;
        slot.point          = point;
        slot.directional    = directional;
        slot.cascade_index  = cascade_index;
        slot.quality        = quality;
        slot.x              = cursor * face;
        slot.y              = 0;
        slot.size           = face;
        slot.atlas_w        = w;
        slot.atlas_h        = h;
        if (directional) light.setCascadeAtlasSlot(cascade_index, slot);
        else light.setShadowAtlasSlot(slot);
        cursor++;
    };
    for (SceneLight* light : point_casters) pack(*light, true, false, 0);
    for (SceneLight* light : spot_casters) pack(*light, false, false, 0);
    for (SceneLight* light : dir_casters) {
        for (int cascade = 0; cascade < 3; ++cascade) {
            pack(*light, false, true, cascade);
        }
    }
    scene.renderTargets[std::string(SpecTex_ShadowAtlas)] = {
        .width           = w,
        .height          = h,
        .mapWidth        = w,
        .mapHeight       = h,
        .allowReuse      = false,
        .comparisonDepth = true,
        .sample          = { TextureWrap::CLAMP_TO_EDGE,
                             TextureWrap::CLAMP_TO_EDGE,
                             TextureFilter::LINEAR,
                             TextureFilter::LINEAR },
    };
    scene.shadows.atlas_active = true;
}

void ClearSceneVolumetrics(Scene& scene) {
    scene.renderTargets.erase(std::string(SpecTex_VolumetricsBack));
    scene.renderTargets.erase(std::string(SpecTex_VolumetricsSingle));
    scene.renderTargets.erase(std::string(SpecTex_VolumetricsLightBuffer));
    scene.renderTargets.erase(std::string(SpecTex_VolumetricsLightBufferB));
    scene.volumetrics.lights.clear();
    scene.volumetrics.nodes.clear();
    scene.volumetrics.outputs.clear();
    scene.volumetrics.blur_h.reset();
    scene.volumetrics.blur_v.reset();
    scene.volumetrics.combine.reset();
    scene.volumetrics.active        = false;
    scene.volumetrics.built_quality = scene.volumetrics.quality == 0 ? 0 : -1;
}

void AddInwardUnitCube(SceneMesh& mesh) {
    // Cookie hull: eight float3 corners. Combined with AltVP = inverse(light clip)
    // this is the unit cube in [-1,1]. Inward winding + cullmode "normal" writes
    // the far faces that volumetricsfront treats as backDepth.
    constexpr float p[8 * 3] = {
        -1, -1, -1,  1, -1, -1,  1,  1, -1, -1,  1, -1,
        -1, -1,  1,  1, -1,  1,  1,  1,  1, -1,  1,  1,
    };
    const uint16_t idx[] = {
        0, 3, 2, 0, 2, 1, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4,
        3, 7, 6, 3, 6, 2, 0, 4, 7, 0, 7, 3, 1, 2, 6, 1, 6, 5,
    };
    SceneVertexArray vertex({ { WE_IN_POSITION.data(), VertexType::FLOAT3 } }, 8);
    vertex.SetVertex(WE_IN_POSITION, std::span<const float>(p, 24));
    mesh.AddVertexArray(std::move(vertex));
    SceneIndexArray indices((sizeof(idx) / sizeof(idx[0]) + 1) / 2);
    indices.AssignHalf(0, std::span<const uint16_t>(idx, sizeof(idx) / sizeof(idx[0])));
    mesh.AddIndexArray(std::move(indices));
}

void AddInwardPointSphere(SceneMesh& mesh) {
    // Point hull: unit sphere. AltVP is T(origin)*S(radius).
    // Latitude step π/24, longitude step 2π/24.
    // Poles at (0, ±1, 0), then 23 rings × 25 verts (last duplicates first).
    // Inward winding: (north[j], north[j+1], south[j]) is CW from outside.
    constexpr int    rings       = 23;
    constexpr int    slices      = 24;
    constexpr int    ring_stride = slices + 1;
    constexpr float  lat_step    = 0.1308997f;
    constexpr float  lon_step    = 0.2617994f;
    std::vector<float>    pos;
    std::vector<uint16_t> idx;
    pos.reserve(static_cast<size_t>((2 + rings * ring_stride) * 3));
    pos.insert(pos.end(), { 0.0f, 1.0f, 0.0f });
    pos.insert(pos.end(), { 0.0f, -1.0f, 0.0f });
    for (int ring = 1; ring <= rings; ++ring) {
        const float lat = static_cast<float>(ring) * lat_step;
        const float sl  = std::sin(lat);
        const float cl  = std::cos(lat);
        for (int slice = 0; slice < ring_stride; ++slice) {
            const float lon = static_cast<float>(slice) * lon_step;
            pos.insert(pos.end(), { std::cos(lon) * sl, cl, std::sin(lon) * sl });
        }
    }
    const auto ring_start = [](int ring_index) -> uint16_t {
        return static_cast<uint16_t>(2 + ring_index * ring_stride);
    };
    for (int ring = 0; ring < rings - 1; ++ring) {
        const uint16_t curr = ring_start(ring);
        const uint16_t next = ring_start(ring + 1);
        for (int slice = 0; slice < slices; ++slice) {
            const uint16_t a = static_cast<uint16_t>(curr + slice);
            const uint16_t b = static_cast<uint16_t>(curr + slice + 1);
            const uint16_t c = static_cast<uint16_t>(next + slice);
            const uint16_t d = static_cast<uint16_t>(next + slice + 1);
            idx.insert(idx.end(), { a, b, c, b, d, c });
        }
    }
    const uint16_t first = ring_start(0);
    const uint16_t last  = ring_start(rings - 1);
    for (int slice = 0; slice < slices; ++slice) {
        idx.insert(idx.end(),
                   { 0, static_cast<uint16_t>(first + slice + 1),
                     static_cast<uint16_t>(first + slice) });
        idx.insert(idx.end(),
                   { 1, static_cast<uint16_t>(last + slice),
                     static_cast<uint16_t>(last + slice + 1) });
    }
    SceneVertexArray vertex({ { WE_IN_POSITION.data(), VertexType::FLOAT3 } },
                            static_cast<uint32_t>(pos.size() / 3));
    vertex.SetVertex(WE_IN_POSITION, std::span<const float>(pos.data(), pos.size()));
    mesh.AddVertexArray(std::move(vertex));
    SceneIndexArray indices((idx.size() + 1) / 2);
    indices.AssignHalf(0, std::span<const uint16_t>(idx.data(), idx.size()));
    mesh.AddIndexArray(std::move(indices));
}

void AddInwardSpotCone(SceneMesh& mesh) {
    // Spot hull: 32 slices, rings at z=1 and z=0 when reverse-depth is off.
    // Apex at z=0, unit base at z=1.
    constexpr int slices = 32;
    std::vector<float>    pos;
    std::vector<uint16_t> idx;
    pos.reserve(static_cast<size_t>((slices + 2) * 3));
    pos.insert(pos.end(), { 0.0f, 0.0f, 0.0f });
    pos.insert(pos.end(), { 0.0f, 0.0f, 1.0f });
    for (int i = 0; i < slices; ++i) {
        const float a = (static_cast<float>(i) / static_cast<float>(slices)) * 6.283185307179586f;
        pos.insert(pos.end(), { std::cos(a), std::sin(a), 1.0f });
    }
    for (int i = 0; i < slices; ++i) {
        const uint16_t a = static_cast<uint16_t>(2 + i);
        const uint16_t b = static_cast<uint16_t>(2 + ((i + 1) % slices));
        idx.insert(idx.end(), { 0, b, a });
        idx.insert(idx.end(), { 1, a, b });
    }
    SceneVertexArray vertex({ { WE_IN_POSITION.data(), VertexType::FLOAT3 } },
                            static_cast<uint32_t>(pos.size() / 3));
    vertex.SetVertex(WE_IN_POSITION, std::span<const float>(pos.data(), pos.size()));
    mesh.AddVertexArray(std::move(vertex));
    SceneIndexArray indices((idx.size() + 1) / 2);
    indices.AssignHalf(0, std::span<const uint16_t>(idx.data(), idx.size()));
    mesh.AddIndexArray(std::move(indices));
}

bool LoadVolumetricUtilMaterial(fs::VFS& vfs, Scene& scene, WPShaderValueUpdater* updater,
                                SceneNode& node, std::string_view json_path,
                                const std::unordered_map<std::string, int32_t>& extra_combos,
                                SceneLight* light, const std::string& cookie,
                                const std::optional<SceneModelRenderState>& render_state,
                                std::shared_ptr<SceneMesh> mesh) {
    nlohmann::json json;
    if (! PARSE_JSON(fs::GetFileContent(vfs, std::string(json_path)), json)) {
        LOG_ERROR("SceneVolumetrics: failed to parse '%.*s'",
                  static_cast<int>(json_path.size()),
                  json_path.data());
        return false;
    }
    wpscene::WPMaterial wpmat;
    if (! wpmat.FromJson(json)) {
        LOG_ERROR("SceneVolumetrics: invalid material '%.*s'",
                  static_cast<int>(json_path.size()),
                  json_path.data());
        return false;
    }
    for (const auto& [name, value] : extra_combos) {
        wpmat.combos[name] = value;
    }
    if (! cookie.empty()) {
        if (wpmat.textures.size() < 3) wpmat.textures.resize(3);
        wpmat.textures[2] = cookie;
    }

    SceneMaterial     material;
    WPShaderValueData sv_data;
    sv_data.volumetric_light = light;
    sv_data.volumetric_pass  = light != nullptr;
    WPShaderInfo shader_info;
    if (! LoadMaterial(vfs, wpmat, &scene, &node, &material, &sv_data, nullptr, &shader_info)) {
        LOG_ERROR("SceneVolumetrics: compile failed '%.*s'",
                  static_cast<int>(json_path.size()),
                  json_path.data());
        return false;
    }
    if (render_state.has_value()) material.modelRenderState = *render_state;
    mesh->AddMaterial(std::move(material));
    node.AddMesh(mesh);
    updater->SetNodeData(&node, sv_data);
    scene.nodeOwners[&node] = 0;
    return true;
}

bool ConfigureSceneVolumetricsImpl(Scene& scene, fs::VFS& vfs) {
    auto* updater = dynamic_cast<WPShaderValueUpdater*>(scene.shaderValueUpdater.get());
    if (updater == nullptr) return false;

    ConfigureSceneShadows(scene);
    ClearSceneVolumetrics(scene);
    const int quality = scene.volumetrics.quality;
    if (quality <= 0) {
        scene.volumetrics.built_quality = 0;
        return true;
    }

    std::vector<SceneLight*> lights;
    for (auto& light : scene.lights) {
        if (light && light->castVolumetrics() && light->typeSupportsVolumetrics()) {
            lights.push_back(light.get());
        }
    }
    if (lights.empty()) {
        scene.volumetrics.built_quality = quality;
        return true;
    }

    // LightBuffer scale: 1/8 below high quality, 1/4 at high/ultra. Back stays full-res.
    const double scale = quality < 3 ? 0.125 : 0.25;
    const i32    full_w = std::max(1, scene.ortho[0]);
    const i32    full_h = std::max(1, scene.ortho[1]);
    const i32    buf_w  = std::max(1, static_cast<i32>(full_w * scale));
    const i32    buf_h  = std::max(1, static_cast<i32>(full_h * scale));

    scene.renderTargets[std::string(SpecTex_VolumetricsBack)] = {
        .width     = full_w,
        .height    = full_h,
        .mapWidth  = full_w,
        .mapHeight = full_h,
        .withDepth = true,
        .bind      = { .enable = true, .screen = true, .scale = 1.0 },
    };
    scene.renderTargets[std::string(SpecTex_VolumetricsSingle)] = {
        .width     = buf_w,
        .height    = buf_h,
        .mapWidth  = buf_w,
        .mapHeight = buf_h,
        .bind      = { .enable = true, .screen = true, .scale = scale },
    };
    scene.renderTargets[std::string(SpecTex_VolumetricsLightBuffer)] = {
        .width     = buf_w,
        .height    = buf_h,
        .mapWidth  = buf_w,
        .mapHeight = buf_h,
        .bind      = { .enable = true, .screen = true, .scale = scale },
    };
    if (quality < 3) {
        scene.renderTargets[std::string(SpecTex_VolumetricsLightBufferB)] = {
            .width     = buf_w,
            .height    = buf_h,
            .mapWidth  = buf_w,
            .mapHeight = buf_h,
            .bind      = { .enable = true, .screen = true, .scale = scale },
        };
    }

    SceneModelRenderState back_state;
    back_state.depthTest     = true;
    back_state.depthWrite    = true;
    back_state.cullMode      = SceneCullMode::Back;
    back_state.colorLoadMode = SceneModelColorLoadMode::Clear;
    // The hull is drawn before the camera-inside test, so both hemispheres rasterize
    // when the eye is inside the volume. Default LESS keeps the near wall and the
    // fullscreen ray (shader z=0 = D3D near) never enters the light. GREATER + clear 0
    // keeps max window Z, matching a D3D depth RT sampled as backDepth.
    back_state.depthGreater  = true;
    back_state.depthClear    = 0.0f;

    SceneModelRenderState front_hull_state;
    front_hull_state.depthTest     = false;
    front_hull_state.depthWrite    = false;
    front_hull_state.cullMode      = SceneCullMode::Back;
    front_hull_state.colorLoadMode = SceneModelColorLoadMode::Load;

    SceneModelRenderState fullscreen_state;
    fullscreen_state.depthTest     = false;
    fullscreen_state.depthWrite    = false;
    fullscreen_state.cullMode      = SceneCullMode::None;
    fullscreen_state.colorLoadMode = SceneModelColorLoadMode::Load;

    for (SceneLight* light : lights) {
        std::unordered_map<std::string, int32_t> combos;
        const bool shadow =
            light->castsShadows() && scene.shadows.quality != 0;
        combos["QUALITY"] = quality;
        combos["SHADOW"]  = shadow ? 1 : 0;
        if (light->hasCookie()) combos["COOKIE"] = 1;
        if (light->type() == SceneLightType::Point) {
            combos["POINTLIGHT"]                    = 1;
            combos["LIGHTS_SHADOW_MAPPING_QUALITY"] = scene.shadows.quality;
        }
        const std::string cookie = light->hasCookie() ? light->cookie() : std::string();

        auto back_node        = std::make_shared<SceneNode>();
        auto front_node       = std::make_shared<SceneNode>();
        auto fullscreen_node  = std::make_shared<SceneNode>();
        back_node->SetName("volumetrics_back");
        front_node->SetName("volumetrics_front");
        fullscreen_node->SetName("volumetrics_fullscreen");

        auto back_mesh  = std::make_shared<SceneMesh>();
        auto front_mesh = std::make_shared<SceneMesh>();
        auto fs_mesh    = std::make_shared<SceneMesh>();
        if (light->type() == SceneLightType::Point) {
            AddInwardPointSphere(*back_mesh);
            AddInwardPointSphere(*front_mesh);
        } else if (light->hasCookie()) {
            AddInwardUnitCube(*back_mesh);
            AddInwardUnitCube(*front_mesh);
        } else {
            AddInwardSpotCone(*back_mesh);
            AddInwardSpotCone(*front_mesh);
        }
        fs_mesh->ChangeMeshDataFrom(scene.default_effect_mesh);

        if (! LoadVolumetricUtilMaterial(vfs,
                                         scene,
                                         updater,
                                         *back_node,
                                         "/assets/materials/util/volumetrics_back.json",
                                         {},
                                         light,
                                         {},
                                         back_state,
                                         back_mesh) ||
            ! LoadVolumetricUtilMaterial(vfs,
                                         scene,
                                         updater,
                                         *front_node,
                                         "/assets/materials/util/volumetrics_front.json",
                                         combos,
                                         light,
                                         cookie,
                                         front_hull_state,
                                         front_mesh) ||
            ! LoadVolumetricUtilMaterial(vfs,
                                         scene,
                                         updater,
                                         *fullscreen_node,
                                         "/assets/materials/util/volumetrics_fullscreen.json",
                                         combos,
                                         light,
                                         cookie,
                                         fullscreen_state,
                                         fs_mesh)) {
            ClearSceneVolumetrics(scene);
            return false;
        }

        Scene::VolumetricLightPass pass;
        pass.light      = light;
        pass.back       = back_node;
        pass.front      = front_node;
        pass.fullscreen = fullscreen_node;
        scene.volumetrics.lights.push_back(std::move(pass));
        scene.volumetrics.nodes.push_back(back_node);
        scene.volumetrics.nodes.push_back(front_node);
        scene.volumetrics.nodes.push_back(fullscreen_node);
    }

    if (quality < 3) {
        auto blur_h = std::make_shared<SceneNode>();
        auto blur_v = std::make_shared<SceneNode>();
        blur_h->SetName("volumetrics_blur_h");
        blur_v->SetName("volumetrics_blur_v");
        auto blur_h_mesh = std::make_shared<SceneMesh>();
        auto blur_v_mesh = std::make_shared<SceneMesh>();
        blur_h_mesh->ChangeMeshDataFrom(scene.default_effect_mesh);
        blur_v_mesh->ChangeMeshDataFrom(scene.default_effect_mesh);
        if (! LoadVolumetricUtilMaterial(vfs,
                                         scene,
                                         updater,
                                         *blur_h,
                                         "/assets/materials/util/volumetrics_blur_h.json",
                                         {},
                                         nullptr,
                                         {},
                                         std::nullopt,
                                         blur_h_mesh) ||
            ! LoadVolumetricUtilMaterial(vfs,
                                         scene,
                                         updater,
                                         *blur_v,
                                         "/assets/materials/util/volumetrics_blur_v.json",
                                         {},
                                         nullptr,
                                         {},
                                         std::nullopt,
                                         blur_v_mesh)) {
            ClearSceneVolumetrics(scene);
            return false;
        }
        scene.volumetrics.blur_h = blur_h;
        scene.volumetrics.blur_v = blur_v;
        scene.volumetrics.nodes.push_back(blur_h);
        scene.volumetrics.nodes.push_back(blur_v);
    }

    auto combine = std::make_shared<SceneNode>();
    combine->SetName("volumetrics_combine");
    auto combine_mesh = std::make_shared<SceneMesh>();
    combine_mesh->ChangeMeshDataFrom(scene.default_effect_mesh);
    if (! LoadVolumetricUtilMaterial(vfs,
                                     scene,
                                     updater,
                                     *combine,
                                     "/assets/materials/util/volumetrics_combine.json",
                                     {},
                                     nullptr,
                                     {},
                                     std::nullopt,
                                     combine_mesh)) {
        ClearSceneVolumetrics(scene);
        return false;
    }
    scene.volumetrics.combine = combine;
    scene.volumetrics.nodes.push_back(combine);
    scene.volumetrics.active        = true;
    scene.volumetrics.built_quality = quality;
    return true;
}

std::string StripJsonExtension(std::string value) {
    constexpr std::string_view extension { ".json" };
    if (value.size() >= extension.size() &&
        value.compare(value.size() - extension.size(), extension.size(), extension) == 0) {
        value.resize(value.size() - extension.size());
    }
    return value;
}

std::string MaterialStem(std::string_view material_path) {
    const auto slash = material_path.find_last_of('/');
    const auto start = slash == std::string_view::npos ? 0 : slash + 1;
    return StripJsonExtension(std::string(material_path.substr(start)));
}

std::string MaterialDirectory(std::string_view material_path) {
    const auto slash = material_path.find_last_of('/');
    if (slash == std::string_view::npos) return {};
    return std::string(material_path.substr(0, slash + 1));
}

std::string ResolveModelMaterialPath(const std::string&    material_path,
                                     const nlohmann::json* sidecar_json, int32_t skin) {
    if (sidecar_json == nullptr || ! sidecar_json->is_object() ||
        ! sidecar_json->contains("skins") || ! sidecar_json->at("skins").is_array()) {
        return material_path;
    }

    const auto& skins = sidecar_json->at("skins");
    if (skin < 0 || static_cast<size_t>(skin) >= skins.size() || ! skins.at(skin).is_object()) {
        return material_path;
    }

    const auto  stem   = MaterialStem(material_path);
    const auto& remaps = skins.at(skin);
    if (! remaps.contains(stem) || ! remaps.at(stem).is_string()) return material_path;

    auto remapped = remaps.at(stem).get<std::string>();
    if (remapped.find('/') == std::string::npos) {
        remapped = MaterialDirectory(material_path) + StripJsonExtension(remapped) + ".json";
    } else if (remapped.size() < 5 || remapped.substr(remapped.size() - 5) != ".json") {
        remapped += ".json";
    }
    return remapped;
}

std::string ResolveStaticChunkMaterialPath(const WPMdl::StaticChunk& chunk, int32_t skin) {
    if (chunk.material_json_variants.empty()) return chunk.material_json_file;
    if (skin >= 0 && static_cast<size_t>(skin) < chunk.material_json_variants.size()) {
        // MDLV0004 static models can carry several material paths for one geometry payload. The
        // scene object owns the skin index, so material selection belongs here rather than in the
        // low-level binary parser that only knows the model file bytes.
        return chunk.material_json_variants[skin];
    }

    LOG_ERROR("ModelMaterialSkin: skin=%d out of range variants=%zu fallback='%s'",
              skin,
              chunk.material_json_variants.size(),
              chunk.material_json_file.c_str());
    return chunk.material_json_file;
}

std::optional<nlohmann::json> LoadModelSidecarJson(fs::VFS& vfs, std::string_view model_path) {
    auto path = std::string(model_path);
    if (path.size() >= 4 && path.substr(path.size() - 4) == ".mdl") {
        path.resize(path.size() - 4);
        path += ".json";
    }

    const auto asset_path = "/assets/" + path;
    if (! vfs.Contains(asset_path)) return std::nullopt;

    const auto content = fs::GetFileContent(vfs, asset_path);
    if (content.empty()) return std::nullopt;

    nlohmann::json json;
    if (! PARSE_JSON(content, json)) {
        LOG_ERROR("ModelSidecar: parse failed path='%s'", path.c_str());
        return std::nullopt;
    }
    return json;
}

std::optional<SceneCullMode> ParseModelCullModeValue(std::string_view value) {
    static const std::unordered_map<std::string_view, SceneCullMode> modes {
        { "nocull", SceneCullMode::None }, { "none", SceneCullMode::None },
        { "normal", SceneCullMode::Back }, { "back", SceneCullMode::Back },
        { "front", SceneCullMode::Front },
    };
    if (auto it = modes.find(value); it != modes.end()) return it->second;
    return std::nullopt;
}

std::string_view ModelCullModeMaterialString(SceneCullMode mode) {
    switch (mode) {
    case SceneCullMode::None: return "nocull";
    case SceneCullMode::Back: return "back";
    case SceneCullMode::Front: return "front";
    }
    return "back";
}

std::optional<bool> ParseModelRenderStateSwitch(std::string_view value) {
    if (value == "enabled" || value == "enable" || value == "true" || value == "1") return true;
    if (value == "disabled" || value == "disable" || value == "false" || value == "0")
        return false;
    return std::nullopt;
}

bool ModelBlendUsesTransparency(std::string_view blending) {
    return blending == "translucent" || blending == "additive";
}

bool ModelMaterialSamplesReflection(const wpscene::WPMaterial& material) {
    return std::any_of(
        material.textures.begin(), material.textures.end(), [](const std::string& texture) {
            return texture == kModelReflectionTargetName;
        });
}

struct ModelMaterialRenderPolicy {
    std::string   blending;
    bool          transparent { false };
    bool          depthTest { true };
    bool          depthWrite { true };
    SceneCullMode cullMode { SceneCullMode::Back };
};

std::optional<ModelMaterialRenderPolicy>
BuildModelMaterialRenderPolicy(const wpscene::WPMaterial& material,
                               const std::string&         material_path) {
    ModelMaterialRenderPolicy policy {};
    policy.blending    = material.blendingAuthored ? material.blending : "normal";
    policy.transparent = ModelBlendUsesTransparency(policy.blending);

    if (material.depthtestAuthored) {
        const auto depth_test = ParseModelRenderStateSwitch(material.depthtest);
        if (! depth_test.has_value()) {
            LOG_ERROR("ModelMaterialState: invalid depthtest '%s' path='%s'",
                      material.depthtest.c_str(),
                      material_path.c_str());
            return std::nullopt;
        }
        policy.depthTest = *depth_test;
    }

    if (material.depthwriteAuthored) {
        const auto depth_write = ParseModelRenderStateSwitch(material.depthwrite);
        if (! depth_write.has_value()) {
            LOG_ERROR("ModelMaterialState: invalid depthwrite '%s' path='%s'",
                      material.depthwrite.c_str(),
                      material_path.c_str());
            return std::nullopt;
        }
        policy.depthWrite = *depth_write;
    }
    if (policy.transparent) {
        // Alpha-blended model chunks should test against opaque geometry but avoid writing depth;
        // otherwise transparent quads such as glass, shadow blobs, or reflection grids can occlude
        // later model chunks. This is a model-material rule, not a parser recovery path.
        policy.depthWrite = false;
    }

    if (material.cullmodeAuthored) {
        const auto cull_mode = ParseModelCullModeValue(material.cullmode);
        if (! cull_mode.has_value()) {
            LOG_ERROR("ModelMaterialState: invalid cullmode '%s' path='%s'",
                      material.cullmode.c_str(),
                      material_path.c_str());
            return std::nullopt;
        }
        policy.cullMode = *cull_mode;
    } else if (ModelMaterialSamplesReflection(material) || policy.transparent) {
        // Model reflection and alpha-blended shell surfaces are intentionally double-sided when the
        // material omits culling. Opaque model chunks keep the stricter back-face default, and the
        // policy is scoped here so the legacy 2D WPMaterial defaults remain unchanged.
        policy.cullMode = SceneCullMode::None;
    }

    return policy;
}

wpscene::WPMaterial BuildEffectiveModelMaterial(wpscene::WPMaterial         material,
                                                const ModelMaterialRenderPolicy& policy) {
    // LoadMaterial still consumes WPMaterial because shader, texture, combo, and binding parsing is
    // shared with 2D layers. This effective copy writes the already-validated model policy into the
    // string fields before that shared loader runs, so missing model fields are resolved once at the
    // model boundary instead of being patched later by renderer code.
    material.blending   = policy.blending;
    material.depthtest  = policy.depthTest ? "enabled" : "disabled";
    material.depthwrite = policy.depthWrite ? "enabled" : "disabled";
    material.cullmode   = std::string(ModelCullModeMaterialString(policy.cullMode));
    return material;
}

bool LoadModelMaterialJson(ParseContext& context, const std::string& material_path,
                           nlohmann::json& material_json) {
    const auto material_source = fs::GetFileContent(*context.vfs, "/assets/" + material_path);
    if (material_source.empty()) return false;

    // Wallpaper Engine model materials are JSON-with-comments in practice. Parse that dialect
    // directly with nlohmann's comment-aware mode so valid first-party materials are accepted on the
    // first pass and malformed assets still produce one clear diagnostic.
    material_json = nlohmann::json::parse(material_source, nullptr, false, true);
    if (! material_json.is_discarded()) return true;

    LOG_ERROR("ModelMaterialJson: parse failed path='%s'", material_path.c_str());
    return false;
}

void SeedModelCameraUniforms(ParseContext& context, WPShaderInfo& shader_info) {
    auto& scene     = *context.scene;
    auto  camera_it = scene.cameras.find(std::string(kSceneModelPerspectiveCameraName));
    if (camera_it == scene.cameras.end() || ! camera_it->second) return;

    const auto eye     = camera_it->second->GetPosition().cast<float>();
    Vector3f   forward = camera_it->second->GetDirection().cast<float>();
    if (forward.norm() > 1e-6f) forward.normalize();
    Vector3f up = camera_it->second->GetUp().cast<float>();
    if (up.norm() > 1e-6f) up.normalize();
    Vector3f right = forward.cross(up);
    if (right.norm() > 1e-6f) right.normalize();

    // These constants are seeded only for model materials. Runtime updates in WPShaderValueUpdater
    // keep them animated for camera paths, while 2D materials never receive this 3D camera
    // contract.
    shader_info.baseConstSvs[std::string(G_EYE_POSITION)] =
        std::array<float, 3> { eye.x(), eye.y(), eye.z() };
    shader_info.baseConstSvs[std::string(G_VIEWUP)] =
        std::array<float, 3> { up.x(), up.y(), up.z() };
    shader_info.baseConstSvs[std::string(G_VIEWRIGHT)] =
        std::array<float, 3> { right.x(), right.y(), right.z() };
    shader_info.baseConstSvs[std::string(G_VIEWFORWARD)] =
        std::array<float, 3> { forward.x(), forward.y(), forward.z() };
}

struct ModelMaterialSource {
    std::string               path;
    wpscene::WPMaterial       material;
    ModelMaterialRenderPolicy renderPolicy;
};

class ModelMaterialLoader {
public:
    ModelMaterialLoader(ParseContext& context, const WPModelObject& model_obj,
                        const nlohmann::json* sidecar_json)
        : context_(context), model_obj_(model_obj), sidecar_json_(sidecar_json) {}

    bool AnyChunkSamplesReflection(const WPMdl& mdl) const {
        return std::any_of(
            mdl.static_chunks.begin(), mdl.static_chunks.end(), [this](const auto& chunk) {
                return ChunkSamplesReflection(chunk);
            });
    }

    bool ChunkSamplesReflection(const WPMdl::StaticChunk& chunk) const {
        const auto source = LoadSource(chunk);
        if (!source.has_value()) {
            LOG_ERROR("ModelReflectionMaterial: failed to inspect layer=%d name='%s' material='%s'",
                      model_obj_.id,
                      model_obj_.name.c_str(),
                      ResolvePath(chunk).c_str());
            return false;
        }

        return ModelMaterialSamplesReflection(source->material);
    }

    bool UsesTransparentBlend(const WPMdl::StaticChunk& chunk) const {
        const auto source = LoadSource(chunk);
        if (! source.has_value()) {
            LOG_ERROR("ModelRenderOrder: failed to inspect layer=%d name='%s' material='%s'",
                      model_obj_.id,
                      model_obj_.name.c_str(),
                      ResolvePath(chunk).c_str());
            return false;
        }

        return source->renderPolicy.transparent;
    }

    bool LoadChunkMaterial(const WPMdl::StaticChunk& chunk, SceneNode* chunk_node,
                           SceneMaterial& material, WPShaderValueData& node_data,
                           wpscene::WPMaterial& resolved_wp_material,
                           WPShaderInfo& resolved_shader_info,
                           SceneModelColorLoadMode color_load_mode,
                           bool mirrored_handedness, std::string output_override) const {
        const auto source = LoadSource(chunk);
        if (! source.has_value()) {
            LOG_ERROR("ModelMaterialLoad: failed to parse layer=%d name='%s' material='%s'",
                      model_obj_.id,
                      model_obj_.name.c_str(),
                      ResolvePath(chunk).c_str());
            return false;
        }

        WPShaderInfo shader_info;
        shader_info.baseConstSvs = context_.global_base_uniforms;
        SeedModelCameraUniforms(context_, shader_info);
        if (! LoadMaterial(*context_.vfs,
                           source->material,
                           context_.scene.get(),
                           chunk_node,
                           &material,
                           &node_data,
                           context_.user_properties,
                           &shader_info)) {
            LOG_ERROR("ModelMaterialLoad: shader load failed layer=%d name='%s' material='%s'",
                      model_obj_.id,
                      model_obj_.name.c_str(),
                      source->path.c_str());
            return false;
        }

        LoadConstvalue(material, source->material, shader_info);
        LoadUserShaderValue(material, source->material, shader_info, context_.user_properties);
        const auto render_state =
            BuildRenderState(color_load_mode,
                             mirrored_handedness,
                             std::move(output_override),
                             source->renderPolicy);
        material.modelRenderState = render_state;
        // Model material JSON and shader metadata are returned to the caller so binding
        // registration can happen after mesh->AddMaterial() and node->AddMesh(). That keeps 3D
        // model chunks on the same material-ready registration path as ordinary scene layers.
        resolved_wp_material = source->material;
        resolved_shader_info = shader_info;
        return true;
    }

private:
    SceneModelRenderState BuildRenderState(SceneModelColorLoadMode color_load_mode,
                                           bool                    mirrored_handedness,
                                           std::string             output_override,
                                           const ModelMaterialRenderPolicy& policy) const {
        // The renderer-facing model state is derived from the same validated policy used to build
        // the effective WPMaterial, keeping shader loading, depth rules, culling, and reflection
        // output routing on one explicit model-material contract.
        return SceneModelRenderState {
            .colorLoadMode      = color_load_mode,
            .depthTest          = policy.depthTest,
            .depthWrite         = policy.depthWrite,
            .cullMode           = policy.cullMode,
            .mirroredHandedness = mirrored_handedness,
            .outputOverride     = std::move(output_override),
        };
    }

    std::optional<ModelMaterialSource> LoadSource(const WPMdl::StaticChunk& chunk) const {
        const auto material_path = ResolvePath(chunk);
        if (const auto cached = source_cache_.find(material_path); cached != source_cache_.end()) {
            return cached->second;
        }
        if (failed_sources_.count(material_path) != 0) return std::nullopt;

        auto source = LoadSourceFromPath(material_path);
        if (! source.has_value()) {
            failed_sources_.insert(material_path);
            return std::nullopt;
        }

        const auto [cached, inserted] = source_cache_.emplace(material_path, std::move(*source));
        return cached->second;
    }

    std::optional<ModelMaterialSource> LoadSourceFromPath(const std::string& material_path) const {
        nlohmann::json material_json;
        if (! LoadModelMaterialJson(context_, material_path, material_json)) return std::nullopt;

        wpscene::WPMaterial wp_material;
        if (! wp_material.FromJson(material_json)) return std::nullopt;

        const auto render_policy = BuildModelMaterialRenderPolicy(wp_material, material_path);
        if (! render_policy.has_value()) return std::nullopt;

        return ModelMaterialSource {
            .path         = material_path,
            .material     = BuildEffectiveModelMaterial(std::move(wp_material), *render_policy),
            .renderPolicy = *render_policy,
        };
    }

    std::string ResolvePath(const WPMdl::StaticChunk& chunk) const {
        return ResolveModelMaterialPath(
            ResolveStaticChunkMaterialPath(chunk, model_obj_.skin), sidecar_json_, model_obj_.skin);
    }

    ParseContext&         context_;
    const WPModelObject&  model_obj_;
    const nlohmann::json* sidecar_json_ { nullptr };
    mutable std::unordered_map<std::string, ModelMaterialSource> source_cache_;
    mutable std::unordered_set<std::string>                      failed_sources_;
};

void EnsureModelReflectionTarget(ParseContext& context) {
    auto& scene = *context.scene;
    if (scene.renderTargets.count(std::string(kModelReflectionTargetName)) != 0) return;

    // Reflection is a model-only render target. It is registered lazily when a model layer requests
    // reflection so ordinary 2D scenes do not gain another render target or graph edge.
    scene.renderTargets[std::string(kModelReflectionTargetName)] = {
        .width                  = context.ortho_w,
        .height                 = context.ortho_h,
        .mapWidth               = context.ortho_w,
        .mapHeight              = context.ortho_h,
        .allowReuse             = true,
        .withDepth              = true,
        .screenSpaceSampleYFlip = true,
        .bind                   = { .enable = true, .screen = true },
    };
    LOG_INFO("ModelReflectionTarget: registered name='_rt_Reflection' size=%ux%u map-size=%ux%u "
             "with-depth=true screen-aligned=true screen-space-sample-y-flip=true",
             context.ortho_w,
             context.ortho_h,
             context.ortho_w,
             context.ortho_h);
}

struct ModelChunkOrder {
    std::vector<usize> opaque;
    std::vector<usize> transparent;
    std::vector<usize> ordered;

    static ModelChunkOrder Build(const WPMdl& mdl, const ModelMaterialLoader& material_loader,
                                 const WPModelObject& model_obj) {
        ModelChunkOrder order;
        order.opaque.reserve(mdl.static_chunks.size());
        order.transparent.reserve(mdl.static_chunks.size());

        for (usize chunk_index = 0; chunk_index < mdl.static_chunks.size(); chunk_index++) {
            const auto& chunk = mdl.static_chunks[chunk_index];
            if (material_loader.UsesTransparentBlend(chunk)) {
                order.transparent.push_back(chunk_index);
            } else {
                order.opaque.push_back(chunk_index);
            }
        }

        order.ordered.reserve(mdl.static_chunks.size());
        order.ordered.insert(order.ordered.end(), order.opaque.begin(), order.opaque.end());
        order.ordered.insert(
            order.ordered.end(), order.transparent.begin(), order.transparent.end());
        order.Log(model_obj);
        return order;
    }

    void Log(const WPModelObject& model_obj) const {
        if (transparent.empty()) return;

        // Transparent model chunks must be appended after opaque chunks from the same authored
        // model. They still depth-test against the opaque depth buffer, but drawing them last keeps
        // later opaque chunks from overwriting glass that does not write depth. The diagnostic
        // records the exact parser-side order used by run.log.
        LOG_INFO("ModelRenderOrder: layer=%d name='%s' opaque=%s transparent=%s final=%s",
                 model_obj.id,
                 model_obj.name.c_str(),
                 DescribeIndexVec(opaque).c_str(),
                 DescribeIndexVec(transparent).c_str(),
                 DescribeIndexVec(ordered).c_str());
    }
};

struct ModelChunkNodeRequest {
    std::string name;
    std::string output_override;
    SceneModelColorLoadMode color_load_mode { SceneModelColorLoadMode::DontCare };
    bool        mirrored_handedness { false };
};

class ModelLayerMaterializer {
public:
    ModelLayerMaterializer(ParseContext& context, const WPModelObject& model_obj)
        : context_(context),
          model_obj_(model_obj),
          sidecar_json_(LoadModelSidecarJson(*context.vfs, model_obj.model)),
          material_loader_(context_, model_obj_, SidecarJson()) {}

    void Materialize(const WPMdl& mdl) {
        root_ = CreateRootNode();
        RegisterRootNode();
        const bool material_samples_reflection = material_loader_.AnyChunkSamplesReflection(mdl);
        if (model_obj_.reflected || material_samples_reflection) {
            EnsureModelReflectionTarget(context_);
        }

        const auto order = ModelChunkOrder::Build(mdl, material_loader_, model_obj_);
        AppendChunks(mdl, order);
        ApplyCastsShadows(root_.get(), model_obj_.castshadow);

        context_.scene->ApplyLayerVisibility(model_obj_.id);
    }

private:
    const nlohmann::json* SidecarJson() const {
        return sidecar_json_.has_value() ? &*sidecar_json_ : nullptr;
    }

    std::shared_ptr<SceneNode> CreateRootNode() const {
        auto root  = std::make_shared<SceneNode>(Vector3f(model_obj_.origin.data()),
                                                 Vector3f(model_obj_.scale.data()),
                                                 Vector3f(model_obj_.angles.data()),
                                                 model_obj_.name);
        root->ID() = model_obj_.id;
        return root;
    }

    void RegisterRootNode() {
        WPShaderValueData root_data;
        ConfigureBoneAttachment(context_,
                                model_obj_.id,
                                model_obj_.parent,
                                model_obj_.attachment,
                                Eigen::Affine3f(root_->GetLocalTrans().cast<float>()),
                                "model",
                                model_obj_.name,
                                root_data);
        AttachNodeToScene(context_, root_, model_obj_.parent, model_obj_.name, &root_data);

        context_.object_nodes[model_obj_.id] = root_;
        context_.scene->objectRuntimeNodes[model_obj_.id].push_back(root_.get());
        context_.scene->nodeOwners[root_.get()] = model_obj_.id;
        context_.shader_updater->SetNodeData(root_.get(), root_data);
        RegisterLayerSceneState(
            context_, model_obj_.id, model_obj_.parent, model_obj_.attachment, model_obj_.visible);
    }

    void AppendChunks(const WPMdl& mdl, const ModelChunkOrder& order) {
        // Receiver materials and producer passes are separate contracts. A material may reference
        // `_rt_Reflection` only so its shader can bind the runtime target, while mirrored producer
        // geometry is authored by the model object's `"reflected": true` flag. Keeping those paths
        // separate prevents receiver-only models from drawing phantom mirrored geometry, without
        // disabling reflected scenes that intentionally populate the target.
        if (model_obj_.reflected) {
            for (usize chunk_index : order.ordered) {
                const auto& chunk = mdl.static_chunks[chunk_index];
                AppendReflectionChunk(chunk, chunk_index);
            }
        }

        for (usize chunk_index : order.ordered) {
            const auto& chunk = mdl.static_chunks[chunk_index];
            AppendMainChunk(chunk, chunk_index);
        }
    }

    void AppendReflectionChunk(const WPMdl::StaticChunk& chunk, usize chunk_index) {
        if (material_loader_.ChunkSamplesReflection(chunk)) {
            // A material that samples `_rt_Reflection` is the receiver surface, not a producer for
            // that same target. Skipping it avoids feedback/self-copy edges while still allowing
            // authored reflected models to populate the target before the receiver draws.
            return;
        }

        auto reflection_node = MakeChunkNode(
            chunk,
            ModelChunkNodeRequest {
                .name                = model_obj_.name + "::__hanabi_model_reflection_chunk_" +
                                       std::to_string(chunk_index),
                .output_override     = std::string(kModelReflectionTargetName),
                .color_load_mode     = NextModelOutputColorLoadMode(kModelReflectionTargetName),
                .mirrored_handedness = true,
            });
        if (reflection_node == nullptr) return;

        // Some authored reflection receivers sample `_rt_Reflection` as a screen-space floor
        // mirror. Mirroring reflected chunks across the authored Y=0 floor plane gives the target
        // the expected geometry, while `mirroredHandedness` above lets the render pass fix
        // winding/culling without weakening cull behavior for normal 3D or any 2D scene.
        reflection_node->SetScale(Vector3f { 1.0f, -1.0f, 1.0f });
        root_->AppendChild(reflection_node);
    }

    void AppendMainChunk(const WPMdl::StaticChunk& chunk, usize chunk_index) {
        auto node = MakeChunkNode(
            chunk,
            ModelChunkNodeRequest {
                .name = model_obj_.name + "::__hanabi_model_chunk_" + std::to_string(chunk_index),
                .color_load_mode = NextModelOutputColorLoadMode(SpecTex_Default),
            });
        if (node != nullptr) root_->AppendChild(node);
    }

    std::shared_ptr<SceneNode> MakeChunkNode(const WPMdl::StaticChunk& chunk,
                                             ModelChunkNodeRequest     request) {
        auto node = std::make_shared<SceneNode>();
        node->SetName(std::move(request.name));
        node->ID() = model_obj_.id;
        // Model chunks use the isolated model camera so authored 3D view transforms cannot move
        // legacy 2D perspective particles that still render through `global_perspective`.
        node->SetCamera(std::string(kSceneModelPerspectiveCameraName));

        auto mesh = std::make_shared<SceneMesh>();
        WPMdlParser::GenStaticMesh(*mesh, chunk);

        SceneMaterial       material;
        WPShaderValueData   node_data;
        wpscene::WPMaterial wp_material;
        WPShaderInfo        shader_info;
        if (! material_loader_.LoadChunkMaterial(chunk,
                                                 node.get(),
                                                 material,
                                                 node_data,
                                                 wp_material,
                                                 shader_info,
                                                 request.color_load_mode,
                                                 request.mirrored_handedness,
                                                 std::move(request.output_override))) {
            return nullptr;
        }

        mesh->AddMaterial(std::move(material));
        node->AddMesh(mesh);
        RegisterUserShaderValueBindings(
            context_, wp_material, shader_info, node.get(), model_obj_.id, model_obj_.name);
        context_.shader_updater->SetNodeData(node.get(), node_data);
        context_.scene->nodeOwners[node.get()] = model_obj_.id;
        context_.scene->objectRuntimeNodes[model_obj_.id].push_back(node.get());
        return node;
    }

    void ApplyCastsShadows(SceneNode* node, bool value) {
        if (node == nullptr) return;
        if (node->Name().find("__hanabi_model_reflection") != std::string::npos) return;
        node->SetCastsShadows(value);
        for (auto& child : node->GetChildren()) {
            ApplyCastsShadows(child.get(), value);
        }
    }

    SceneModelColorLoadMode NextModelOutputColorLoadMode(std::string_view output) {
        auto  key   = output.empty() ? std::string(SpecTex_Default) : std::string(output);
        auto& count = context_.model_pass_count_by_output[key];
        if (count++ > 0) return SceneModelColorLoadMode::Load;

        // The default scene target is already owned by the renderer pre-pass, while private model
        // render targets have no standalone clear pass. Clearing the first model writer to an
        // offscreen target prevents transparent pixels from loading the previous frame, and later
        // writers still load so multi-chunk models compose into the same target.
        return key == SpecTex_Default ? SceneModelColorLoadMode::DontCare
                                      : SceneModelColorLoadMode::Clear;
    }

    ParseContext&                 context_;
    const WPModelObject&          model_obj_;
    std::optional<nlohmann::json> sidecar_json_;
    ModelMaterialLoader           material_loader_;
    std::shared_ptr<SceneNode>    root_;
};

void ParseModelObj(ParseContext& context, WPModelObject& model_obj) {
    if (! model_obj.visible) return;

    WPMdl mdl;
    if (! WPMdlParser::ParseStaticModel(model_obj.model, *context.vfs, mdl)) {
        LOG_ERROR("ModelObjectParse: static mdl parse failed layer=%d name='%s' model='%s'",
                  model_obj.id,
                  model_obj.name.c_str(),
                  model_obj.model.c_str());
        return;
    }

    ModelLayerMaterializer(context, model_obj).Materialize(mdl);
}

void ParseImageObj(ParseContext& context, wpscene::WPImageObject& img_obj,
                   bool force_runtime_materialization = false) {
    auto& wpimgobj = img_obj;
    NotePassthroughLayer(context, wpimgobj);
    const auto* visibility_contract = FindLayerVisibilityContract(context, wpimgobj.id);
    const bool  can_defer_hidden_runtime_image = ShouldDeferRuntimeLayerMaterialization(
        context, wpimgobj.id, wpimgobj.visible, visibility_contract, force_runtime_materialization);
    if (can_defer_hidden_runtime_image) {
        RegisterLogicalImageLayer(context, wpimgobj, true);
        return;
    }
    if (! wpimgobj.visible && ! force_runtime_materialization) return;

    auto& vfs = *context.vfs;

    const auto register_logical_only_layer = [&]() {
        RegisterLogicalImageLayer(context, wpimgobj, false);
    };

    int32_t count_eff = 0;
    for (const auto& wpeffobj : wpimgobj.effects) {
        const auto effect_visibility =
            BuildEffectVisibilityContract(wpeffobj, context.user_properties);
        if (! effect_visibility.can_prune_at_parse_time) count_eff++;
    }
    const bool hasAuthoredEffect = count_eff > 0;
    bool       isCompose         = (wpimgobj.image == "models/util/composelayer.json");
    const bool isProjectLayer =
        wpimgobj.projectlayer || wpimgobj.image == "models/util/projectlayer.json";
    const bool is_offscreen_dependency_source =
        context.scene != nullptr &&
        context.scene->offscreenDependencyLayerIds.count(wpimgobj.id) != 0;
    const bool has_shader_color_blend = UsesShaderColorBlendMode(wpimgobj.colorBlendMode);
    // Wallpaper Engine `dependencies` expose a layer through `_rt_imageLayerComposite_<id>`
    // even when the source layer has no authored effects. Such layers still need a private source
    // render target because the visible consumer samples that source while the layer itself remains
    // hidden in the main scene. Treating dependency-only image layers as effect-backed sources lets
    // the existing effect camera/ping-pong path materialize the raw image or mask without drawing
    // it directly into `_rt_default`.
    bool hasEffect =
        hasAuthoredEffect || has_shader_color_blend || is_offscreen_dependency_source;
    const bool uses_routed_parent = LayerUsesRoutedParent(context, wpimgobj.parent, wpimgobj.attachment);
    // Official one object / +0x158 (0x140190837). I is a matrix slot at 0x1401e96ac,
    // not a second SceneNode identity (SO5 / 0x1401ec720).
    // Screen-bound effect sources are only the fullscreen flag and assets that already mark
    // themselves that way (direct-draw shapes without an authored size). An ortho-sized image
    // layer keeps its authored ping-pong; object scale is applied when that result is composited.
    const bool effect_source_screen_bound =
        wpimgobj.fullscreen || wpimgobj.effectSourceScreenBound;
    const std::array<float, 2> effect_source_size =
        wpimgobj.effectSourceSize[0] > 0.0f && wpimgobj.effectSourceSize[1] > 0.0f
            ? wpimgobj.effectSourceSize
            : wpimgobj.size;
    // skip no effect fullscreen layer
    if (! hasEffect && wpimgobj.fullscreen) {
        register_logical_only_layer();
        return;
    }

    const bool hasAuthoredPuppet = ! wpimgobj.puppet.empty();
    // No-effect compose/project layers are logical framebuffer helpers. Drawing them as regular
    // image meshes can sample `_rt_default` and write it back through the scene camera, which
    // applies a second projection to the already-composited frame on non-authored output aspects.
    if (! hasEffect && (isCompose || isProjectLayer) && ! is_offscreen_dependency_source) {
        register_logical_only_layer();
        return;
    }

    std::unique_ptr<WPMdl> puppet;
    if (hasAuthoredPuppet) {
        puppet = std::make_unique<WPMdl>();
        if (! WPMdlParser::Parse(wpimgobj.puppet, vfs, *puppet)) {
            LOG_ERROR("parse puppet failed: %s", wpimgobj.puppet.c_str());
            puppet = nullptr;
        } else if (puppet->kind == WPMdl::MeshKind::Puppet &&
                   (puppet->puppet == nullptr || puppet->puppet->bones.empty())) {
            LOG_ERROR("puppet has no bones: %s", wpimgobj.puppet.c_str());
            puppet = nullptr;
        }
    }
    if (puppet != nullptr) {
        // Puppet clipping masks are independent imported textures referenced by MDLV0022+
        // metadata rather than by the visible material JSON. Register them with the same scene
        // texture contract as authored material slots so deferred visibility preparation can stage
        // their bytes before the masked mesh becomes executable.
        for (const auto& mask : puppet->masks) {
            if (context.scene->textures.count(mask.material) != 0) continue;
            const auto& header = context.scene->imageParser->ParseHeader(mask.material);
            RegisterSceneTextureFromHeader(*context.scene, mask.material, header);
        }
    }
    const bool hasAnimatedPuppetMesh =
        puppet != nullptr && puppet->kind == WPMdl::MeshKind::Puppet && puppet->puppet != nullptr;
    const bool hasStaticImageMesh =
        puppet != nullptr && puppet->kind == WPMdl::MeshKind::StaticImage;

    // wpimgobj.origin[1] = context.ortho_h - wpimgobj.origin[1];
    auto spLayerNode = std::make_shared<SceneNode>(Vector3f(wpimgobj.origin.data()),
                                                   Vector3f(wpimgobj.scale.data()),
                                                   Vector3f(wpimgobj.angles.data()),
                                                   wpimgobj.name);
    LoadAlignment(*spLayerNode, wpimgobj.alignment, { wpimgobj.size[0], wpimgobj.size[1] });
    spLayerNode->ID() = wpimgobj.id;
    auto spImgNode = spLayerNode;
    spImgNode->SetName(wpimgobj.name);
    spImgNode->ID() = wpimgobj.id;

    SceneMaterial     material;
    WPShaderValueData svData;
    WPShaderValueData worldNodeData;
    TextureSample     source_sampler;
    std::string       primary_source_texture;
    WPPuppetLayer     shared_puppet_pose;
    if (hasAnimatedPuppetMesh) {
        shared_puppet_pose = WPPuppetLayer(puppet->puppet);
        shared_puppet_pose.prepared(wpimgobj.puppet_layers);
    }

    ShaderValueMap baseConstSvs = context.global_base_uniforms;
    const float    initial_alpha = ClampParserOpacityScalar(wpimgobj.alpha);
    WPShaderInfo   shaderInfo;
    {
        if (! hasEffect) {
            svData.parallaxDepth = { wpimgobj.parallaxDepth[0], wpimgobj.parallaxDepth[1] };
            svData.parallaxDepthAuthored = wpimgobj.parallaxDepthAuthored;
            if (hasAnimatedPuppetMesh) {
                WPMdlParser::AddPuppetShaderInfo(shaderInfo, *puppet);
            }
        }

        baseConstSvs["g_Color4"] = std::array<float, 4> {
            wpimgobj.color[0], wpimgobj.color[1], wpimgobj.color[2], initial_alpha
        };
        baseConstSvs["g_Color"] =
            std::array<float, 3> { wpimgobj.color[0], wpimgobj.color[1], wpimgobj.color[2] };
        baseConstSvs["g_Alpha"]      = initial_alpha;
        baseConstSvs["g_UserAlpha"]  = initial_alpha;
        baseConstSvs["g_Brightness"] = wpimgobj.brightness;

        shaderInfo.baseConstSvs = baseConstSvs;

        if (! LoadMaterial(vfs,
                           wpimgobj.material,
                           context.scene.get(),
                           spImgNode.get(),
                           &material,
                           &svData,
                           context.user_properties,
                           &shaderInfo)) {
            LOG_ERROR("load imageobj '%s' material faild", wpimgobj.name.c_str());
            return;
        };
        LoadConstvalue(material, wpimgobj.material, shaderInfo);
        LoadUserShaderValue(material, wpimgobj.material, shaderInfo, context.user_properties);
        source_sampler = ResolvePrimaryMaterialSampler(*context.scene, material);
        if (!material.textures.empty()) primary_source_texture = material.textures.front();
    }

    // mesh
    SceneMesh effct_final_mesh {};
    SceneMesh puppet_surface_mesh {};
    auto      spMesh = std::make_shared<SceneMesh>();
    auto&     mesh   = *spMesh;

    {
        const bool primary_texture_is_sprite =
            PrimaryMaterialTextureIsSprite(*context.scene, material);
        std::array<float, 2> mapRate { 1.0f, 1.0f };
        if (! wpimgobj.nopadding &&
            exists(material.customShader.constValues, WE_GLTEX_RESOLUTION_NAMES[0])) {
            const auto& r = material.customShader.constValues.at(WE_GLTEX_RESOLUTION_NAMES[0]);
            const std::array<float, 2> padded_map_rate { r[2] / r[0], r[3] / r[1] };

            /*
             * Ordinary padded images sample the card UV directly, so their base coordinates must
             * be cropped to the logical content extent. Sprite frames already encode rotation and
             * translation normalized against the physical mip0 atlas. Cropping the card before
             * that frame transform applies the same content-to-physical ratio twice and stretches
             * a partial frame over the authored layer. Keep sprite cards in frame-local [0, 1]
             * space while preserving the physical/content resolution contract for shader users.
             */
            if (primary_texture_is_sprite) {
                if (padded_map_rate[0] != 1.0f || padded_map_rate[1] != 1.0f) {
                    LOG_INFO("SceneImageCardUvContract: layer=%d name='%s' texture='%s' "
                             "physical=[%.0f %.0f] content=[%.0f %.0f] "
                             "skipped-card-uv=[%.4f %.4f] "
                             "final-card-uv=[1.0000 1.0000] owner=sprite-frame",
                             wpimgobj.id,
                             wpimgobj.name.c_str(),
                             primary_source_texture.c_str(),
                             r[0],
                             r[1],
                             r[2],
                             r[3],
                             padded_map_rate[0],
                             padded_map_rate[1]);
                }
            } else {
                mapRate = padded_map_rate;
            }
        }
        // IMAGE_VT_128 0x140209290: leftover +0x2d8 a_TexCoord max is
        // (tex+0x2c)/(tex+0x20) from the texture header, not g_Texture0Resolution
        // (that vector is {map,map,map,map} when !mipmap_larger) and not the
        // sprite card [0,1] contract. Last-pass +0x2e8 UV stays 1,1 (TEXT_2F0).
        if (auto* object = context.scene->FindSceneObject(wpimgobj.id);
            object != nullptr && !primary_source_texture.empty()) {
            auto apply_header = [&](i32 w, i32 h, i32 mw, i32 mh) {
                if (w <= 0 || h <= 0) return;
                object->set_leftover_uv(
                    static_cast<float>(mw > 0 ? mw : w) / static_cast<float>(w),
                    static_cast<float>(mh > 0 ? mh : h) / static_cast<float>(h));
            };
            if (const auto tex_it = context.scene->textures.find(primary_source_texture);
                tex_it != context.scene->textures.end()) {
                apply_header(tex_it->second.width, tex_it->second.height, tex_it->second.mapWidth,
                             tex_it->second.mapHeight);
            } else if (context.scene->imageParser != nullptr) {
                const auto header = context.scene->imageParser->ParseHeader(primary_source_texture);
                apply_header(header.width, header.height, header.mapWidth, header.mapHeight);
            }
        }

        if (hasAnimatedPuppetMesh) {
            if (hasEffect) {
                GenCardMesh(
                    mesh, { (uint16_t)wpimgobj.size[0], (uint16_t)wpimgobj.size[1] }, mapRate);
                WPMdlParser::GenPuppetMesh(puppet_surface_mesh, *puppet);
                effct_final_mesh.ChangeMeshDataFrom(puppet_surface_mesh);
                if (auto* object = context.scene->FindSceneObject(wpimgobj.id)) {
                    object->CaptureImage490Mesh(puppet_surface_mesh);
                }
                AppendPuppetSurfaceWriter(wpimgobj, *puppet);
            } else {
                svData.puppet_layer = shared_puppet_pose;
                WPMdlParser::GenPuppetMesh(mesh, *puppet);
                if (auto* object = context.scene->FindSceneObject(wpimgobj.id)) {
                    object->CaptureImage490Mesh(mesh);
                }
            }
        } else if (hasStaticImageMesh) {
            if (hasEffect) {
                GenCardMesh(
                    mesh, { (uint16_t)wpimgobj.size[0], (uint16_t)wpimgobj.size[1] }, mapRate);
                WPMdlParser::GenPuppetMesh(effct_final_mesh, *puppet);
            } else {
                WPMdlParser::GenPuppetMesh(mesh, *puppet);
            }
        } else {
            if (hasAuthoredPuppet) {
                LOG_INFO("ImagePuppetFallback: layer=%d name='%s' puppet='%s' using rectangular "
                         "card mesh",
                         wpimgobj.id,
                         wpimgobj.name.c_str(),
                         wpimgobj.puppet.c_str());
            }
            const auto source_mesh_size = wpimgobj.size;
            GenCardMesh(
                mesh, { (uint16_t)source_mesh_size[0], (uint16_t)source_mesh_size[1] }, mapRate);
            std::array<std::array<float, 2>, 4> perspective_points {};
            if (LoadShapePerspectivePoints(wpimgobj, perspective_points)) {
                GenOfficialShapePerspectiveDestMesh(
                    effct_final_mesh, wpimgobj.size, perspective_points);
            } else {
                // DEST_DRAW_VERTS / CARD_CENTER: leftover dest-draw +0x2e0 is {2,2} ±1.
                GenCardMesh(effct_final_mesh, { 2, 2 });
            }
        }
    }
    // material blendmode for last step to use
    auto imgBlendMode = ResolveObjectFinalBlend(material.blenmode, wpimgobj.colorBlendMode);
    if (wpimgobj.image == "__hanabi_shape_directdraw") {
        // SHAPE_BLEND / DEST_1F0_WRITERS: shape dest +0x1f0 is enum 2
        // (0x1402607a0 / 0x140260790). DIRECTDRAW string xrefs only shape
        // parse 0x140260088. Non-shape dest stays authored unless
        // IMAGE_DEST_BLEND (colorBlendMode 0x1f or +0x304&0x60==0x20).
        imgBlendMode = BlendMode::Additive;
    }
    // disable img material blend, as it's the first effect node now
    if (hasEffect) {
        material.blenmode = BlendMode::Normal;
    } else {
        material.blenmode = imgBlendMode;
    }
    mesh.AddMaterial(std::move(material));
    spImgNode->AddMesh(spMesh);
    RegisterUserShaderValueBindings(
        context, wpimgobj.material, shaderInfo, spImgNode.get(), wpimgobj.id, wpimgobj.name);

    if (hasAnimatedPuppetMesh) {
        svData.puppet_layer = shared_puppet_pose;
    }

    ConfigureBoneAttachment(context,
                            wpimgobj.id,
                            wpimgobj.parent,
                            wpimgobj.attachment,
                            Eigen::Affine3f(spLayerNode->GetLocalTrans().cast<float>()),
                            "object",
                            wpimgobj.name,
                            svData);

    worldNodeData               = svData;
    worldNodeData.parallaxDepth = { wpimgobj.parallaxDepth[0], wpimgobj.parallaxDepth[1] };
    worldNodeData.parallaxDepthAuthored = wpimgobj.parallaxDepthAuthored;

    if (hasEffect) {
        auto& scene = *context.scene;
        FinalOutputCapability final_shader_capability =
            wpimgobj.config.finalOutputCapability;
        // currently use addr for unique
        std::string nodeAddr = getAddr(spImgNode.get());
        const auto  effect_camera_clip = ResolveImageEffectCameraClipRange(hasAnimatedPuppetMesh);
        // set camera to attatch effect
        if (isCompose) {
            const int32_t source_camera_width =
                std::max<int32_t>(1, static_cast<int32_t>(std::lround(effect_source_size[0])));
            const int32_t source_camera_height =
                std::max<int32_t>(1, static_cast<int32_t>(std::lround(effect_source_size[1])));
            scene.cameras[nodeAddr] = std::make_shared<SceneCamera>(
                source_camera_width,
                source_camera_height,
                effect_camera_clip.near_clip,
                effect_camera_clip.far_clip);
            scene.cameras.at(nodeAddr)->AttatchNode(spLayerNode);
            LOG_INFO("SceneCompositionLayerSourceCamera: layer=%d name='%s' camera='%s' "
                     "size=[%d, %d] source-target=[%.3f, %.3f] near=%.3f far=%.3f "
                     "animated-puppet=%s",
                     wpimgobj.id,
                     wpimgobj.name.c_str(),
                     nodeAddr.c_str(),
                     source_camera_width,
                     source_camera_height,
                     effect_source_size[0],
                     effect_source_size[1],
                     effect_camera_clip.near_clip,
                     effect_camera_clip.far_clip,
                     hasAnimatedPuppetMesh ? "true" : "false");
        } else {
            // Keep the effect camera extents in display units. The render target
            // resolution below may still be reduced independently.
            i32 w                   = (i32)wpimgobj.size[0];
            i32 h                   = (i32)wpimgobj.size[1];
            scene.cameras[nodeAddr] = std::make_shared<SceneCamera>(
                w, h, effect_camera_clip.near_clip, effect_camera_clip.far_clip);
            scene.cameras.at(nodeAddr)->AttatchNode(context.effect_camera_node);
            LOG_INFO("SceneImageEffectSourceCamera: layer=%d name='%s' camera='%s' "
                     "size=[%d, %d] near=%.3f far=%.3f animated-puppet=%s",
                     wpimgobj.id,
                     wpimgobj.name.c_str(),
                     nodeAddr.c_str(),
                     w,
                     h,
                     effect_camera_clip.near_clip,
                     effect_camera_clip.far_clip,
                     hasAnimatedPuppetMesh ? "true" : "false");
        }
        scene.objectRuntimeCameraNames[wpimgobj.id].push_back(nodeAddr);
        spImgNode->SetCamera(nodeAddr);
        std::string effect_ppong_a, effect_ppong_b;
        effect_ppong_a = WE_EFFECT_PPONG_PREFIX_A.data() + nodeAddr;
        effect_ppong_b = WE_EFFECT_PPONG_PREFIX_B.data() + nodeAddr;
        // set image effect
        // Official one object / +0x158 (0x140190837). This SceneNode is the
        // layer handle for leftover dest-draw, not a WorldNode leftover sibling.
        auto imgEffectLayer = std::make_shared<SceneImageEffectLayer>(
            spLayerNode.get(), wpimgobj.size[0], wpimgobj.size[1], effect_ppong_a, effect_ppong_b);
        {
            // Fullscreen image-effect layers are postprocess-style framebuffer passes. Remember
            // that authored shape here so ResolveEffect() can keep their final shader on the
            // effect-camera fullscreen quad instead of projecting the 2x2 utility mesh through the
            // active scene camera.
            imgEffectLayer->SetFullscreen(wpimgobj.fullscreen);
            imgEffectLayer->SetFinalBlend(imgBlendMode);
            // Official dest factors are +0x1f0 cases only (GFX_BLEND_DESC
            // 0x14009a32b / 0x14009a2fe). gfx+0x28 bit7 premul stays 0.
            imgEffectLayer->SetCopyBackground(wpimgobj.copybackground);
            const auto source_policy = ResolveImageEffectSourcePolicy(isCompose, wpimgobj);
            imgEffectLayer->SetSourceContributionPolicy(source_policy);
            if (isCompose) {
                LOG_INFO("SceneCompositionLayerSourcePolicy: layer=%d name='%s' "
                         "copybackground=%s policy=%.*s",
                         wpimgobj.id,
                         wpimgobj.name.c_str(),
                         wpimgobj.copybackground ? "true" : "false",
                         static_cast<int>(ImageEffectSourcePolicyName(source_policy).size()),
                         ImageEffectSourcePolicyName(source_policy).data());
            }
            imgEffectLayer->SetHiddenFinalCompositePolicy(
                ResolveHiddenFinalCompositePolicy(wpimgobj));
            imgEffectLayer->SourceMesh().ChangeMeshDataFrom(mesh);
            imgEffectLayer->FinalMesh().ChangeMeshDataFrom(effct_final_mesh);
            if (wpimgobj.effects.empty()) {
                imgEffectLayer->FinalNode().CopyTrans(*spLayerNode);
                imgEffectLayer->FinalNode().ID() = wpimgobj.id;
                if (context.scene != nullptr) {
                    context.scene->nodeOwners[&imgEffectLayer->FinalNode()] = wpimgobj.id;
                }
            }
            // Official dest stays on the object (0x140185150 origin→+0x110).
            // Do not wipe the shared node. I is written at 0x1401ec799 / 0x1401e96ac.
            scene.cameras.at(nodeAddr)->AttatchImgEffect(imgEffectLayer);
        }
        if (hasAnimatedPuppetMesh && puppet->asset_bounds.IsFiniteAndOrdered()) {
            const std::string puppet_surface_camera = nodeAddr + "__puppet_surface_camera";
            const std::string puppet_surface_target =
                "_rt_puppet_surface_" + nodeAddr;
            imgEffectLayer->SetPuppetSurfaceProjection(BuildPuppetSurfaceProjection(
                wpimgobj,
                *puppet,
                puppet_surface_mesh,
                puppet_surface_camera,
                puppet_surface_target,
                effect_source_size));
            imgEffectLayer->SetLayerSurfaceCamera(puppet_surface_camera);

            const auto* projection = imgEffectLayer->GetPuppetSurfaceProjection();
            if (projection != nullptr) {
                scene.cameras[puppet_surface_camera] = std::make_shared<SceneCamera>(
                    1, 1, effect_camera_clip.near_clip, effect_camera_clip.far_clip);
                scene.cameras.at(puppet_surface_camera)->AttatchNode(context.effect_camera_node);
                scene.cameras.at(puppet_surface_camera)->SetOrthographicViewRect(
                    projection->surface_bounds.min.x(),
                    projection->surface_bounds.max.x(),
                    projection->surface_bounds.min.y(),
                    projection->surface_bounds.max.y());
                scene.objectRuntimeCameraNames[wpimgobj.id].push_back(puppet_surface_camera);

                scene.renderTargets[puppet_surface_target] = SceneRenderTarget {
                    .width = projection->target_extent[0],
                    .height = projection->target_extent[1],
                    .mapWidth = projection->target_extent[0],
                    .mapHeight = projection->target_extent[1],
                    .allowReuse = true,
                    .sample = source_sampler,
                };
                scene.objectRuntimeRenderTargets[wpimgobj.id].push_back(puppet_surface_target);
            }
        }
        // set renderTarget for ping-pong operate
        {
            scene.renderTargets[effect_ppong_a] = {
                .width      = (uint16_t)effect_source_size[0],
                .height     = (uint16_t)effect_source_size[1],
                .mapWidth   = (uint16_t)effect_source_size[0],
                .mapHeight  = (uint16_t)effect_source_size[1],
                .allowReuse = true,
                .sample     = source_sampler,
            };
            if (effect_source_screen_bound) {
                scene.renderTargets[effect_ppong_a].bind = { .enable = true, .screen = true };
            }
            scene.renderTargets[effect_ppong_b] = scene.renderTargets.at(effect_ppong_a);
            // Intermediate effect output is a separate sampling contract. Point-preserving source
            // passes read ping-pong A with the authored source sampler; generic downstream filters
            // read ping-pong B linearly unless their own FBO contract says otherwise.
            scene.renderTargets[effect_ppong_b].sample = TextureSample {
                .wrapS = TextureWrap::CLAMP_TO_EDGE,
                .wrapT = TextureWrap::CLAMP_TO_EDGE,
                .magFilter = TextureFilter::LINEAR,
                .minFilter = TextureFilter::LINEAR,
            };
            scene.objectRuntimeRenderTargets[wpimgobj.id].push_back(effect_ppong_a);
            scene.objectRuntimeRenderTargets[wpimgobj.id].push_back(effect_ppong_b);
            if (effect_source_screen_bound) {
                LOG_INFO("SceneEffectPingPongTargetResolve: layer=%d name='%s' "
                         "pingpong-a='%s' pingpong-b='%s' authored-size=[%.3f, %.3f] "
                         "screen-bound=true fullscreen=%s",
                         wpimgobj.id,
                         wpimgobj.name.c_str(),
                         effect_ppong_a.c_str(),
                         effect_ppong_b.c_str(),
                         effect_source_size[0],
                         effect_source_size[1],
                         wpimgobj.fullscreen ? "true" : "false");
            }
        }
        if (hasAuthoredEffect || has_shader_color_blend) {
            // Dependency-only sources intentionally stop at the first ping-pong target so
            // `_rt_imageLayerComposite_<id>` samples the raw source texture. Every real authored
            // chain and every framebuffer-aware color blend instead uses Wallpaper Engine's
            // independent final passthrough publisher.
            const auto finalCompositeTransformData = BuildEffectWriterTransformData(
                context,
                BuildImageEffectFinalCompositeContract(wpimgobj,
                                                       uses_routed_parent));
            ConfigureEffectFinalComposite(context,
                                          *imgEffectLayer,
                                          effect_ppong_a,
                                          wpimgobj.id,
                                          wpimgobj.name,
                                          wpimgobj.colorBlendMode,
                                          &finalCompositeTransformData);
        }
        int32_t i_eff = -1;
        for (const auto& wpeffobj : wpimgobj.effects) {
            i_eff++;
            const auto effect_visibility =
                BuildEffectVisibilityContract(wpeffobj, context.user_properties);
            if (effect_visibility.can_prune_at_parse_time) {
                i_eff--;
                continue;
            }
            std::shared_ptr<SceneImageEffect> imgEffect = std::make_shared<SceneImageEffect>();
            imgEffect->SetIdentity(
                wpimgobj.id, wpeffobj.id, static_cast<uint32_t>(i_eff), wpeffobj.name);
            imgEffect->SetRuntimeVisibilityContract(effect_visibility.requires_runtime_contract);
            if (effect_visibility.requires_runtime_contract) {
                LOG_INFO("SceneVisibilityEffectMaterialize: layer=%d effect-id=%d effect-index=%d "
                         "name='%s' initial-visible=%s authored-visible=%s",
                         wpimgobj.id,
                         wpeffobj.id,
                         i_eff,
                         wpeffobj.name.c_str(),
                         effect_visibility.initial_visible ? "true" : "false",
                         effect_visibility.authored_visible ? "true" : "false");
            }
            const ImageEffectMaterialTopology effect_material_topology {
                .uses_routed_parent = uses_routed_parent,
                .writer_role        = ResolveImageEffectWriterRole(isCompose),
            };

            // this will be replace when resolve, use here to get rt info
            const std::string inRT { effect_ppong_a };

            // fbo name map and effect command
            std::string effaddr = getAddr(imgEffectLayer.get());
            const auto  feedback_fbos = wpeffobj.FeedbackFboNames();

            std::unordered_map<std::string, std::string> fboMap;
            {
                fboMap["previous"] = inRT;
                for (usize i = 0; i < wpeffobj.fbos.size(); i++) {
                    const auto& wpfbo  = wpeffobj.fbos.at(i);
                    std::string rtname = wpfbo.name + "_" + effaddr;
                    const auto  fbo_size = wpfbo.ResolveSize(effect_source_size);
                    const bool  persistent_feedback_fbo =
                        feedback_fbos.count(wpfbo.name) != 0;
                    if (effect_source_screen_bound && wpfbo.fit == 0) {
                        scene.renderTargets[rtname] = {
                            .width      = 2,
                            .height     = 2,
                            .mapWidth   = 2,
                            .mapHeight  = 2,
                            .allowReuse = ! persistent_feedback_fbo,
                        };
                        scene.renderTargets[rtname].bind = {
                            .enable = true,
                            .screen = true,
                            .scale  = 1.0 / wpfbo.scale,
                        };
                    } else {
                        // `fit`-sized feedback buffers are authored simulation textures, not
                        // display-space framebuffers. Resolve them directly here so shaders see the
                        // intended texel size through g_TextureNResolution and their diffusion
                        // kernels advance at Wallpaper Engine's authored rate.
                        scene.renderTargets[rtname] = {
                            .width      = fbo_size[0],
                            .height     = fbo_size[1],
                            .mapWidth   = fbo_size[0],
                            .mapHeight  = fbo_size[1],
                            .allowReuse = ! persistent_feedback_fbo,
                        };
                    }
                    if (wpfbo.fit > 0 || persistent_feedback_fbo) {
                        LOG_INFO("SceneEffectFboResolve: layer=%d effect-id=%d effect='%s' "
                                 "fbo='%s' target='%s' size=%dx%d scale=%u fit=%u "
                                 "persistent-feedback=%s",
                                 wpimgobj.id,
                                 wpeffobj.id,
                                 wpeffobj.name.c_str(),
                                 wpfbo.name.c_str(),
                                 rtname.c_str(),
                                 fbo_size[0],
                                 fbo_size[1],
                                 wpfbo.scale,
                                 wpfbo.fit,
                                 persistent_feedback_fbo ? "true" : "false");
                    }
                    scene.objectRuntimeRenderTargets[wpimgobj.id].push_back(rtname);
                    fboMap[wpfbo.name] = rtname;
                }
            }
            // load! effect commands
            {
                for (const auto& el : wpeffobj.commands) {
                    if (el.command != "copy") {
                        LOG_ERROR("Unknown effect command: %s", el.command.c_str());
                        continue;
                    }
                    if (fboMap.count(el.target) + fboMap.count(el.source) < 2) {
                        LOG_ERROR("Unknown effect command dst or src: %s %s",
                                  el.target.c_str(),
                                  el.source.c_str());
                        continue;
                    }
                    const auto resolved_dst = fboMap[el.target];
                    const auto resolved_src = fboMap[el.source];
                    imgEffect->commands.push_back({ .cmd          = SceneImageEffect::CmdType::Copy,
                                                    .authored_dst = resolved_dst,
                                                    .authored_src = resolved_src,
                                                    .dst          = resolved_dst,
                                                    .src          = resolved_src,
                                                    .afterpos     = el.afterpos });
                }
            }

            bool eff_mat_ok { true };

            for (usize i_mat = 0; i_mat < wpeffobj.materials.size(); i_mat++) {
                wpscene::WPMaterial wpmat = wpeffobj.materials.at(i_mat);
                std::string         matOutRT { WE_EFFECT_PPONG_PREFIX_B };
                if (wpeffobj.passes.size() > i_mat) {
                    const auto& wppass = wpeffobj.passes.at(i_mat);
                    wpmat.MergePass(wppass);
                    // Set rendertarget, in and out
                    for (const auto& el : wppass.bind) {
                        if (fboMap.count(el.name) == 0) {
                            LOG_ERROR("fbo %s not found", el.name.c_str());
                            continue;
                        }
                        if (wpmat.textures.size() <= (usize)el.index)
                            wpmat.textures.resize((usize)el.index + 1);
                        wpmat.textures[(usize)el.index] = fboMap[el.name];
                    }
                    if (! wppass.target.empty()) {
                        if (fboMap.count(wppass.target) == 0) {
                            LOG_ERROR("fbo %s not found", wppass.target.c_str());
                        } else {
                            matOutRT = fboMap.at(wppass.target);
                        }
                    }
                }
                if (wpmat.textures.size() == 0) wpmat.textures.resize(1);
                if (wpmat.textures.at(0).empty()) {
                    wpmat.textures[0] = inRT;
                }
                auto spEffNode = std::make_shared<SceneNode>();
                ShaderValueMap effectBaseConstSvs = baseConstSvs;
                WPShaderInfo wpEffShaderInfo;
                wpEffShaderInfo.baseConstSvs = std::move(effectBaseConstSvs);
                wpEffShaderInfo.baseConstSvs["g_EffectTextureProjectionMatrix"] =
                    ShaderValue::fromMatrix(Eigen::Matrix4f::Identity());
                wpEffShaderInfo.baseConstSvs["g_EffectTextureProjectionMatrixInverse"] =
                    ShaderValue::fromMatrix(Eigen::Matrix4f::Identity());
                SceneMaterial     material;
                WPShaderValueData svData;
                if (! LoadMaterial(vfs,
                                   wpmat,
                                   context.scene.get(),
                                   spEffNode.get(),
                                   &material,
                                   &svData,
                                   context.user_properties,
                                   &wpEffShaderInfo)) {
                    LOG_ERROR(
                        "SceneEffectLoad: layer=%d effect='%s' material-index=%zu load failed",
                        wpimgobj.id,
                        wpeffobj.name.c_str(),
                        i_mat);
                    eff_mat_ok = false;
                    break;
                }

                if (IsCurrentEffectWriterTarget(matOutRT)) {
                    final_shader_capability = ResolveFinalShaderCapability(wpmat.shader);
                }

                // load glname from alias and load to constvalue
                LoadConstvalue(material, wpmat, wpEffShaderInfo);
                LoadUserShaderValue(material, wpmat, wpEffShaderInfo, context.user_properties);
                auto spMesh = std::make_shared<SceneMesh>();
                {
                    ApplyEffectWriterTransformContract(
                        context,
                        BuildImageEffectMaterialContract(wpimgobj,
                                                         *imgEffectLayer,
                                                         effect_material_topology,
                                                         hasAnimatedPuppetMesh && wpmat.use_puppet),
                        svData);
                    if (hasAnimatedPuppetMesh && wpmat.use_puppet) {
                        svData.puppet_layer = shared_puppet_pose;
                    }
                }
                const auto authored_textures = material.textures;
                spMesh->AddMaterial(std::move(material));
                spEffNode->AddMesh(spMesh);
                RegisterUserShaderValueBindings(
                    context, wpmat, wpEffShaderInfo, spEffNode.get(), wpimgobj.id, wpimgobj.name);
                RegisterConstantShaderValueBindings(context,
                                                    wpmat,
                                                    wpEffShaderInfo,
                                                    spEffNode.get(),
                                                    wpimgobj.id,
                                                    wpimgobj.name,
                                                    wpeffobj.id,
                                                    i_eff,
                                                    i_mat);

                context.shader_updater->SetNodeData(spEffNode.get(), svData);
                context.scene->nodeOwners[spEffNode.get()] = wpimgobj.id;
                imgEffect->nodes.push_back({ .authored_output = matOutRT,
                                             .output = matOutRT,
                                             .authored_textures = authored_textures,
                                             .sceneNode = spEffNode,
                                             .private_final_output_uses_layer_surface =
                                                 hasAnimatedPuppetMesh && wpmat.use_puppet });
            }

            if (eff_mat_ok) {
                imgEffect->SetLocalVisible(effect_visibility.initial_visible);
                imgEffectLayer->AddEffect(imgEffect);
            } else {
                LOG_ERROR("effect \'%s\' failed to load", wpeffobj.name.c_str());
            }
        }

        // Capability priority is structural. Dependency sources must remain private, animated
        // puppets must keep their private skinned surface, and only then may the last authored
        // shader publish directly in scene space.
        if (is_offscreen_dependency_source) {
            imgEffectLayer->SetFinalOutputCapability(
                FinalOutputCapability::PrivateDependency);
        } else if (hasAnimatedPuppetMesh) {
            imgEffectLayer->SetFinalOutputCapability(
                FinalOutputCapability::PrivatePuppetSurface);
        } else if (wpimgobj.config.finalOutputCapability ==
                   FinalOutputCapability::SceneAuthoredWriter) {
            imgEffectLayer->SetFinalOutputCapability(
                FinalOutputCapability::SceneAuthoredWriter);
        } else {
            imgEffectLayer->SetFinalOutputCapability(final_shader_capability);
        }
        LOG_INFO("SceneEffectOutputCapability: layer=%d name='%s' capability=%.*s "
                 "dependency=%s puppet=%s source-policy=%.*s",
                 wpimgobj.id,
                 wpimgobj.name.c_str(),
                 static_cast<int>(FinalOutputCapabilityName(
                     imgEffectLayer->DeclaredFinalOutputCapability()).size()),
                 FinalOutputCapabilityName(
                     imgEffectLayer->DeclaredFinalOutputCapability()).data(),
                 is_offscreen_dependency_source ? "true" : "false",
                 hasAnimatedPuppetMesh ? "true" : "false",
                 static_cast<int>(ImageEffectSourcePolicyName(
                     imgEffectLayer->SourceContributionPolicy()).size()),
                 ImageEffectSourcePolicyName(
                     imgEffectLayer->SourceContributionPolicy()).data());

        int32_t source_width = 0;
        int32_t source_height = 0;
        if (const auto source_texture_it = scene.textures.find(primary_source_texture);
            source_texture_it != scene.textures.end()) {
            source_width = source_texture_it->second.width;
            source_height = source_texture_it->second.height;
        } else if (const auto source_target_it = scene.renderTargets.find(primary_source_texture);
                   source_target_it != scene.renderTargets.end()) {
            source_width = source_target_it->second.ContentWidth();
            source_height = source_target_it->second.ContentHeight();
        }
        const auto& pingpong_a = scene.renderTargets.at(effect_ppong_a);
        const auto& pingpong_b = scene.renderTargets.at(effect_ppong_b);
        const auto& display_target = scene.renderTargets.at(SpecTex_Default.data());
        LOG_INFO("SceneEffectTextureContract: layer=%d name='%s' source='%s' "
                 "source-size=[%d %d] authored-layer-size=[%.3f %.3f] "
                 "source-sampler=[wrap-s=%.*s wrap-t=%.*s mag=%.*s min=%.*s] "
                 "private-target-a='%s' private-target-a-size=[%d %d] "
                 "intermediate-sampler-a=[wrap-s=%.*s wrap-t=%.*s mag=%.*s min=%.*s] "
                 "private-target-b='%s' private-target-b-size=[%d %d] "
                 "intermediate-sampler-b=[wrap-s=%.*s wrap-t=%.*s mag=%.*s min=%.*s] "
                 "display-target='%.*s' display-target-size=[%d %d] output-policy=%.*s",
                 wpimgobj.id,
                 wpimgobj.name.c_str(),
                 primary_source_texture.c_str(),
                 source_width,
                 source_height,
                 wpimgobj.size[0],
                 wpimgobj.size[1],
                 static_cast<int>(TextureWrapName(source_sampler.wrapS).size()),
                 TextureWrapName(source_sampler.wrapS).data(),
                 static_cast<int>(TextureWrapName(source_sampler.wrapT).size()),
                 TextureWrapName(source_sampler.wrapT).data(),
                 static_cast<int>(TextureFilterName(source_sampler.magFilter).size()),
                 TextureFilterName(source_sampler.magFilter).data(),
                 static_cast<int>(TextureFilterName(source_sampler.minFilter).size()),
                 TextureFilterName(source_sampler.minFilter).data(),
                 effect_ppong_a.c_str(),
                 pingpong_a.width,
                 pingpong_a.height,
                 static_cast<int>(TextureWrapName(pingpong_a.sample.wrapS).size()),
                 TextureWrapName(pingpong_a.sample.wrapS).data(),
                 static_cast<int>(TextureWrapName(pingpong_a.sample.wrapT).size()),
                 TextureWrapName(pingpong_a.sample.wrapT).data(),
                 static_cast<int>(TextureFilterName(pingpong_a.sample.magFilter).size()),
                 TextureFilterName(pingpong_a.sample.magFilter).data(),
                 static_cast<int>(TextureFilterName(pingpong_a.sample.minFilter).size()),
                 TextureFilterName(pingpong_a.sample.minFilter).data(),
                 effect_ppong_b.c_str(),
                 pingpong_b.width,
                 pingpong_b.height,
                 static_cast<int>(TextureWrapName(pingpong_b.sample.wrapS).size()),
                 TextureWrapName(pingpong_b.sample.wrapS).data(),
                 static_cast<int>(TextureWrapName(pingpong_b.sample.wrapT).size()),
                 TextureWrapName(pingpong_b.sample.wrapT).data(),
                 static_cast<int>(TextureFilterName(pingpong_b.sample.magFilter).size()),
                 TextureFilterName(pingpong_b.sample.magFilter).data(),
                 static_cast<int>(TextureFilterName(pingpong_b.sample.minFilter).size()),
                 TextureFilterName(pingpong_b.sample.minFilter).data(),
                 static_cast<int>(SpecTex_Default.size()),
                 SpecTex_Default.data(),
                 display_target.width,
                 display_target.height,
                 static_cast<int>(FinalOutputCapabilityName(
                     imgEffectLayer->DeclaredFinalOutputCapability()).size()),
                 FinalOutputCapabilityName(
                     imgEffectLayer->DeclaredFinalOutputCapability()).data());
    }
    if (uses_routed_parent) {
        if (hasEffect) {
            ConfigureRoutedEffectWorldParentBinding(context, wpimgobj.parent, worldNodeData);
        } else {
            ConfigureInheritedParentBinding(context, wpimgobj.parent, svData);
        }
        context.scene->sceneGraph->AppendChild(spLayerNode);
        RegisterRenderOrderProxyChild(context, wpimgobj.parent, spLayerNode, wpimgobj.id);
    } else if (wpimgobj.parent != 0) {
        // Official attach writes +0x180/+0x190 only (0x1401de470). InheritParent
        // would replace BoneAttachment and drop 0x1401852a6.
        if (wpimgobj.attachment.empty()) {
            if (hasEffect) {
                ConfigureRoutedEffectWorldParentBinding(context, wpimgobj.parent, worldNodeData);
            } else {
                ConfigureInheritedParentBinding(context, wpimgobj.parent, svData);
            }
        }
        context.scene->sceneGraph->AppendChild(spLayerNode);
    } else {
        AttachNodeToScene(context, spLayerNode, wpimgobj.parent, wpimgobj.name, &svData);
    }
    context.object_nodes[wpimgobj.id]       = spLayerNode;
    context.scene->imageLayers[wpimgobj.id] = Scene::ImageLayerRuntimeState {
        .size      = wpimgobj.size,
        .alignment = wpimgobj.alignment,
    };
    context.scene->objectRuntimeNodes[wpimgobj.id].push_back(spLayerNode.get());
    context.scene->nodeOwners[spLayerNode.get()] = wpimgobj.id;
    if (auto* object = context.scene->FindSceneObject(wpimgobj.id)) {
        object->set_source_node(spLayerNode.get());
        if (hasAnimatedPuppetMesh) object->set_puppet_layer(shared_puppet_pose);
    }
    if (hasAnimatedPuppetMesh) {
        context.object_puppets[wpimgobj.id] = puppet->puppet.get();
    }
    // Official one object (0x140190837). Effect internals write I (0x1401e96ac);
    // visible blit uses dest after I pop (0x1401e9dd5).
    context.shader_updater->SetNodeData(spImgNode.get(), hasEffect ? worldNodeData : svData);
    RegisterLayerSceneState(
        context, wpimgobj.id, wpimgobj.parent, wpimgobj.attachment, wpimgobj.visible);
    context.scene->ApplyLayerVisibility(wpimgobj.id);
}

void ParseTextObj(ParseContext& context, wpscene::WPTextObject& text_obj) {
    const auto* visibility_contract = FindLayerVisibilityContract(context, text_obj.id);
    TextLayerRenderContract render_contract;
    for (const auto& wp_effect : text_obj.effects) {
        const auto effect_visibility =
            BuildEffectVisibilityContract(wp_effect, context.user_properties);
        if (!effect_visibility.can_prune_at_parse_time) {
            render_contract.has_materialized_authored_effects = true;
            break;
        }
    }
    render_contract.uses_shader_color_blend_bridge =
        UsesShaderColorBlendMode(text_obj.colorBlendMode);

    // This immutable contract is resolved before either deferred or concrete materialization. Text
    // rasterization, logical-box preservation, camera/target sizing, glyph placement, and final
    // publication must all agree on the same bridge decision; consulting effects or blend mode again
    // in any downstream path recreates the crop/offset mismatch this contract exists to prevent.
    if (ShouldDeferRuntimeLayerMaterialization(
            context, text_obj.id, text_obj.visible, visibility_contract, false)) {
        // Hidden dynamic text, including text hidden only by a runtime-controlled parent branch,
        // stays as a lightweight logical layer until first effective visibility. This keeps startup
        // cheap while preserving the same runtime target that scripts and user bindings will
        // materialize later through MaterializeDeferredTextLayer().
        RegisterLogicalTextLayer(context, text_obj, render_contract);
        return;
    }
    if (! text_obj.visible) {
        return;
    }

    std::shared_ptr<SceneTextPrimitive> primitive;
    std::string                         error;
    if (! BuildSceneTextPrimitive(
            *context.vfs,
            text_obj,
            render_contract,
            0,
            context.scene->textRenderScale,
            &primitive,
            &error)) {
        LOG_ERROR("build text primitive '%s' failed: %s", text_obj.name.c_str(), error.c_str());
        return;
    }

    const bool has_effect = render_contract.RequiresBridge();
    auto       spLayerNode = std::make_shared<SceneNode>(Vector3f(text_obj.origin.data()),
                                                         Vector3f(text_obj.scale.data()),
                                                         Vector3f(text_obj.angles.data()),
                                                         text_obj.name);
    spLayerNode->ID()      = text_obj.id;
    // Official text +0x50 is 0x1401e8aa0 (same image draw). One object; I is
    // the matrix slot at 0x1401e96ac, not a second SceneNode (SO5).
    auto spTextNode        = spLayerNode;
    spTextNode->SetName(text_obj.name);
    spTextNode->ID() = text_obj.id;
    spTextNode->AddText(primitive);

    WPShaderValueData worldNodeData;
    worldNodeData.parallaxDepth = { text_obj.parallaxDepth[0], text_obj.parallaxDepth[1] };
    worldNodeData.parallaxDepthAuthored = text_obj.parallaxDepthAuthored;
    ConfigureBoneAttachment(context,
                            text_obj.id,
                            text_obj.parent,
                            text_obj.attachment,
                            Eigen::Affine3f(spLayerNode->GetLocalTrans().cast<float>()),
                            "text object",
                            text_obj.name,
                            worldNodeData);

    if (has_effect) {
        auto&             scene       = *context.scene;
        const std::string camera_name = getAddr(spTextNode.get());
        primitive->bridge.camera_name = camera_name;
        primitive->bridge.pingpong_a  = WE_EFFECT_PPONG_PREFIX_A.data() + camera_name;
        primitive->bridge.pingpong_b  = WE_EFFECT_PPONG_PREFIX_B.data() + camera_name;
        primitive->bridge.bridge_backing_extent = {
            static_cast<uint32_t>(std::max(
                1, static_cast<int32_t>(std::lround(primitive->VisibleDisplaySize()[0])))),
            static_cast<uint32_t>(std::max(
                1, static_cast<int32_t>(std::lround(primitive->VisibleDisplaySize()[1])))),
        };
        primitive->bridge.render_targets.push_back(
            TextBridgeRenderTarget { .name = primitive->bridge.pingpong_a, .scale = 1 });
        primitive->bridge.render_targets.push_back(
            TextBridgeRenderTarget { .name = primitive->bridge.pingpong_b, .scale = 1 });

        const auto display_size = primitive->VisibleDisplaySize();
        SceneMesh  effect_final_mesh {};
        RebuildTextPrimitiveVisibleMesh(&effect_final_mesh, *primitive);

        scene.cameras[camera_name] = std::make_shared<SceneCamera>(
            std::max(1, static_cast<int32_t>(std::lround(display_size[0]))),
            std::max(1, static_cast<int32_t>(std::lround(display_size[1]))),
            -1.0f,
            1.0f);
        scene.cameras.at(camera_name)->AttatchNode(context.effect_camera_node);
        // Official internals write I into [engine+0x30] (0x1401e96ac). Same object,
        // not a second source identity.
        scene.objectRuntimeCameraNames[text_obj.id].push_back(camera_name);
        spTextNode->SetCamera(camera_name);

        auto imgEffectLayer = std::make_shared<SceneImageEffectLayer>(spLayerNode.get(),
                                                                      display_size[0],
                                                                      display_size[1],
                                                                      primitive->bridge.pingpong_a,
                                                                      primitive->bridge.pingpong_b);
        imgEffectLayer->SetFinalBlend(
            ResolveObjectFinalBlend(BlendMode::Translucent, text_obj.colorBlendMode));
        imgEffectLayer->FinalMesh().ChangeMeshDataFrom(effect_final_mesh);
        if (text_obj.effects.empty()) {
            imgEffectLayer->FinalNode().CopyTrans(*spLayerNode);
            imgEffectLayer->FinalNode().ID() = text_obj.id;
            scene.nodeOwners[&imgEffectLayer->FinalNode()] = text_obj.id;
        }
        scene.cameras.at(camera_name)->AttatchImgEffect(imgEffectLayer);

        scene.renderTargets[primitive->bridge.pingpong_a] = SceneRenderTarget {
            .width = static_cast<int32_t>(primitive->bridge.bridge_backing_extent[0]),
            .height = static_cast<int32_t>(primitive->bridge.bridge_backing_extent[1]),
            .mapWidth = std::max(
                1, static_cast<int32_t>(std::lround(primitive->VisibleDisplaySize()[0]))),
            .mapHeight = std::max(
                1, static_cast<int32_t>(std::lround(primitive->VisibleDisplaySize()[1]))),
            // Text targets keep the authored letter box for both the physical backing and the
            // logical effect grid. Persisting the cache entry replaces the old image in place.
            .allowReuse = false,
        };
        scene.renderTargets[primitive->bridge.pingpong_b] =
            scene.renderTargets.at(primitive->bridge.pingpong_a);
        scene.objectRuntimeRenderTargets[text_obj.id].push_back(primitive->bridge.pingpong_a);
        scene.objectRuntimeRenderTargets[text_obj.id].push_back(primitive->bridge.pingpong_b);
        const auto source_size = primitive->VisibleSourceSize();
        const auto& bridge_camera = *scene.cameras.at(camera_name);
        const auto& bridge_target = scene.renderTargets.at(primitive->bridge.pingpong_a);
        LOG_INFO("SceneTextBridgeContract: layer=%d name='%s' camera='%s' "
                 "camera-size=[%.3f %.3f] target='%s' target-size=[%d %d] "
                 "logical-display=[%.3f %.3f] logical-source=[%.3f %.3f] "
                 "glyph-display=[%.3f %.3f] glyph-source=[%.3f %.3f] "
                 "glyph-offset=[%.3f %.3f] display-offset=[%.3f %.3f] "
                 "source-crop=[%.3f %.3f %.3f %.3f] "
                 "final-mesh-bounds=[%.3f %.3f]-[%.3f %.3f]",
                 text_obj.id,
                 text_obj.name.c_str(),
                 camera_name.c_str(),
                 bridge_camera.Width(),
                 bridge_camera.Height(),
                 primitive->bridge.pingpong_a.c_str(),
                 bridge_target.width,
                 bridge_target.height,
                 display_size[0],
                 display_size[1],
                 source_size[0],
                 source_size[1],
                 primitive->layout.glyph_display_size[0],
                 primitive->layout.glyph_display_size[1],
                 primitive->layout.glyph_source_size[0],
                 primitive->layout.glyph_source_size[1],
                 primitive->layout.glyph_offset[0],
                 primitive->layout.glyph_offset[1],
                 primitive->layout.visible_display_offset[0],
                 primitive->layout.visible_display_offset[1],
                 primitive->layout.glyph_source_crop[0],
                 primitive->layout.glyph_source_crop[1],
                 primitive->layout.glyph_source_crop[2],
                 primitive->layout.glyph_source_crop[3],
                 -display_size[0] * 0.5f,
                 -display_size[1] * 0.5f,
                 display_size[0] * 0.5f,
                 display_size[1] * 0.5f);
        const auto finalCompositeTransformData = BuildEffectWriterTransformData(
            context, BuildTextEffectFinalCompositeContract(context, text_obj));
        ConfigureEffectFinalComposite(context,
                                      *imgEffectLayer,
                                      primitive->bridge.pingpong_a,
                                      text_obj.id,
                                      text_obj.name,
                                      text_obj.colorBlendMode,
                                      &finalCompositeTransformData);

        const std::string in_rt        = primitive->bridge.pingpong_a;
        const std::string effect_addr  = getAddr(imgEffectLayer.get());
        int32_t           effect_index = -1;
        for (const auto& wp_effect : text_obj.effects) {
            effect_index++;
            const auto effect_visibility =
                BuildEffectVisibilityContract(wp_effect, context.user_properties);
            if (effect_visibility.can_prune_at_parse_time) {
                effect_index--;
                continue;
            }
            std::shared_ptr<SceneImageEffect> img_effect = std::make_shared<SceneImageEffect>();
            img_effect->SetIdentity(
                text_obj.id, wp_effect.id, static_cast<uint32_t>(effect_index), wp_effect.name);
            img_effect->SetRuntimeVisibilityContract(effect_visibility.requires_runtime_contract);
            if (effect_visibility.requires_runtime_contract) {
                LOG_INFO("SceneVisibilityEffectMaterialize: layer=%d effect-id=%d effect-index=%d "
                         "name='%s' initial-visible=%s authored-visible=%s",
                         text_obj.id,
                         wp_effect.id,
                         effect_index,
                         wp_effect.name.c_str(),
                         effect_visibility.initial_visible ? "true" : "false",
                         effect_visibility.authored_visible ? "true" : "false");
            }
            std::unordered_map<std::string, std::string> fbo_map;
            fbo_map["previous"] = in_rt;
            const auto feedback_fbos = wp_effect.FeedbackFboNames();

            for (const auto& wp_fbo : wp_effect.fbos) {
                const std::string rt_name = wp_fbo.name + "_" + effect_addr;
                const auto fbo_size = wp_fbo.ResolveSize(primitive->VisibleDisplaySize());
                const bool        persistent_feedback_fbo =
                    feedback_fbos.count(wp_fbo.name) != 0;
                scene.renderTargets[rt_name] = SceneRenderTarget {
                    .width      = fbo_size[0],
                    .height     = fbo_size[1],
                    .mapWidth   = fbo_size[0],
                    .mapHeight  = fbo_size[1],
                    // Text-owned effect targets are runtime-sized under stable names. Keeping each
                    // entry persistent prevents the reusable image pool from retaining every old
                    // size after audio-driven scale or output changes.
                    .allowReuse = false,
                };
                if (wp_fbo.fit > 0 || persistent_feedback_fbo) {
                    LOG_INFO("SceneTextEffectFboResolve: layer=%d effect-id=%d effect='%s' "
                             "fbo='%s' target='%s' size=%dx%d scale=%u fit=%u "
                             "persistent-feedback=%s",
                             text_obj.id,
                             wp_effect.id,
                             wp_effect.name.c_str(),
                             wp_fbo.name.c_str(),
                             rt_name.c_str(),
                             fbo_size[0],
                             fbo_size[1],
                             wp_fbo.scale,
                             wp_fbo.fit,
                             persistent_feedback_fbo ? "true" : "false");
                }
                scene.objectRuntimeRenderTargets[text_obj.id].push_back(rt_name);
                primitive->bridge.render_targets.push_back(TextBridgeRenderTarget {
                    .name = rt_name,
                    .scale = std::max<uint32_t>(1u, wp_fbo.scale),
                    .fit = wp_fbo.fit,
                    .persistent_feedback = persistent_feedback_fbo,
                });
                fbo_map[wp_fbo.name] = rt_name;
            }

            for (const auto& command : wp_effect.commands) {
                if (command.command != "copy") continue;
                if (fbo_map.count(command.target) == 0 || fbo_map.count(command.source) == 0)
                    continue;
                const auto resolved_dst = fbo_map.at(command.target);
                const auto resolved_src = fbo_map.at(command.source);
                img_effect->commands.push_back({ .cmd          = SceneImageEffect::CmdType::Copy,
                                                 .authored_dst = resolved_dst,
                                                 .authored_src = resolved_src,
                                                 .dst          = resolved_dst,
                                                 .src          = resolved_src,
                                                 .afterpos     = command.afterpos });
            }

            bool effect_materials_ok = true;
            for (usize material_index = 0; material_index < wp_effect.materials.size();
                 material_index++) {
                wpscene::WPMaterial material_source = wp_effect.materials.at(material_index);
                std::string         material_output { WE_EFFECT_PPONG_PREFIX_B };
                if (wp_effect.passes.size() > material_index) {
                    const auto& wp_pass = wp_effect.passes.at(material_index);
                    material_source.MergePass(wp_pass);
                    for (const auto& bind : wp_pass.bind) {
                        if (fbo_map.count(bind.name) == 0) continue;
                        if (material_source.textures.size() <= static_cast<usize>(bind.index)) {
                            material_source.textures.resize(static_cast<usize>(bind.index) + 1);
                        }
                        material_source.textures[static_cast<usize>(bind.index)] =
                            fbo_map.at(bind.name);
                    }
                    if (! wp_pass.target.empty() && fbo_map.count(wp_pass.target) != 0) {
                        material_output = fbo_map.at(wp_pass.target);
                    }
                }
                if (material_source.textures.empty()) material_source.textures.resize(1);
                if (material_source.textures[0].empty()) material_source.textures[0] = in_rt;

                auto         spEffectNode = std::make_shared<SceneNode>();
                WPShaderInfo effect_shader_info;
                effect_shader_info.baseConstSvs = context.global_base_uniforms;
                effect_shader_info.baseConstSvs["g_EffectTextureProjectionMatrix"] =
                    ShaderValue::fromMatrix(Eigen::Matrix4f::Identity());
                effect_shader_info.baseConstSvs["g_EffectTextureProjectionMatrixInverse"] =
                    ShaderValue::fromMatrix(Eigen::Matrix4f::Identity());

                SceneMaterial     effect_material;
                WPShaderValueData effect_node_data;
                if (! LoadMaterial(*context.vfs,
                                   material_source,
                                   context.scene.get(),
                                   spEffectNode.get(),
                                   &effect_material,
                                   &effect_node_data,
                                   context.user_properties,
                                   &effect_shader_info)) {
                    effect_materials_ok = false;
                    break;
                }
                LoadConstvalue(effect_material, material_source, effect_shader_info);
                LoadUserShaderValue(
                    effect_material, material_source, effect_shader_info, context.user_properties);

                // Text effect passes can receive a new bridge quad whenever the shaped text bounds
                // change, for example on a minute rollover from "12:47" to "12:48". Treating these
                // effect meshes as dynamic from graph construction keeps the already-compiled
                // CustomShaderPass hot and lets the renderer upload the changed quad through the
                // same dynamic-buffer path used by particle-like geometry, instead of destroying
                // and rebuilding the shader pass for every text size change.
                auto spMesh                             = std::make_shared<SceneMesh>(true);
                ApplyEffectWriterTransformContract(
                    context,
                    BuildTextEffectMaterialContract(context, text_obj, *imgEffectLayer),
                    effect_node_data);
                const auto authored_textures = effect_material.textures;
                spMesh->AddMaterial(std::move(effect_material));
                spEffectNode->AddMesh(spMesh);
                RegisterUserShaderValueBindings(context,
                                                material_source,
                                                effect_shader_info,
                                                spEffectNode.get(),
                                                text_obj.id,
                                                text_obj.name);
                // Text layers build their effect materials through this separate bridge path rather
                // than the image-layer effect parser. Register constant-shader scripts here as well
                // so media-driven opacity fades on dynamic song-title/artist text receive playback
                // events and can reveal the updated text layer over the authored placeholder layer.
                RegisterConstantShaderValueBindings(context,
                                                    material_source,
                                                    effect_shader_info,
                                                    spEffectNode.get(),
                                                    text_obj.id,
                                                    text_obj.name,
                                                    wp_effect.id,
                                                    effect_index,
                                                    material_index);
                context.shader_updater->SetNodeData(spEffectNode.get(), effect_node_data);
                context.scene->nodeOwners[spEffectNode.get()] = text_obj.id;
                img_effect->nodes.push_back({ .authored_output = material_output,
                                              .output = material_output,
                                              .authored_textures = authored_textures,
                                              .sceneNode = spEffectNode });
            }

            if (effect_materials_ok) {
                img_effect->SetLocalVisible(effect_visibility.initial_visible);
                imgEffectLayer->AddEffect(img_effect);
            }
        }
    }

    if (LayerUsesRoutedParent(context, text_obj.parent, text_obj.attachment)) {
        ConfigureInheritedParentBinding(context, text_obj.parent, worldNodeData);
        context.scene->sceneGraph->AppendChild(spLayerNode);
        RegisterRenderOrderProxyChild(context, text_obj.parent, spLayerNode, text_obj.id);
    } else if (text_obj.parent != 0) {
        if (text_obj.attachment.empty()) {
            ConfigureInheritedParentBinding(context, text_obj.parent, worldNodeData);
        }
        context.scene->sceneGraph->AppendChild(spLayerNode);
    } else {
        AttachNodeToScene(context, spLayerNode, text_obj.parent, text_obj.name, &worldNodeData);
    }

    context.object_nodes[text_obj.id] = spLayerNode;
    context.scene->objectRuntimeNodes[text_obj.id].push_back(spLayerNode.get());
    context.scene->nodeOwners[spLayerNode.get()] = text_obj.id;
    if (auto* object = context.scene->FindSceneObject(text_obj.id)) {
        object->set_source_node(spLayerNode.get());
    }
    context.shader_updater->SetNodeData(spLayerNode.get(), worldNodeData);

    context.scene->textLayers[text_obj.id] = TextLayerRuntimeState {
        .object            = text_obj,
        .primitive         = primitive,
        .render_contract   = render_contract,
        .applied_alignment = ResolveTextLayerSceneAlignment(text_obj),
    };
    PublishTextDestDrawMeshes(*context.scene, text_obj.id);

    ApplyTextLayerNodePlacement(spLayerNode.get(),
                                context.scene->textLayers[text_obj.id],
                                text_obj.origin);

    RegisterLayerSceneState(
        context, text_obj.id, text_obj.parent, text_obj.attachment, text_obj.visible);
    context.scene->ApplyLayerVisibility(text_obj.id);
    context.scene->deferredRuntimeTextLayerIds.erase(text_obj.id);
}

struct ParticleChildPtr {
    wpscene::ParticleChild* child { nullptr };
    SceneNode*              node_parent { nullptr };
    ParticleSubSystem*      particle_parent { nullptr };

    // This is the mesh-capacity estimate for the subsystem currently being parsed, not the
    // authored child `maxcount` blindly multiplied through the whole child chain. Runtime child
    // subsystem instances are globally capped by each child relationship, so recursive products
    // overestimate dynamic particle meshes by orders of magnitude.
    i32 max_instancecount { 1 };
    // Event-follow children can only follow live parent particles. Carrying the parent's live slot
    // count lets mesh capacity match that runtime constraint instead of reserving the authored
    // default follow count when the parent can only ever expose fewer live particles.
    u32 parent_live_particle_slots { 1 };
};

void AttachExtraParticleRenderer(ParseContext& context, wpscene::WPParticleObject& wppartobj,
                                 const wpscene::Particle& particle_obj,
                                 const wpscene::ParticleRender& particle_renderer,
                                 SceneNode& parent_node, ParticleSubSystem& subsystem,
                                 const WPShaderValueData& parent_sv,
                                 const ParticleRawGenSpecOp& spec_op, u32 maxcount,
                                 u32 mesh_instancecount, std::string_view particle_debug_name,
                                 bool register_layer_nodes) {
    const auto renderer_spec = DescribeParticleRender(particle_renderer.name);
    if (! renderer_spec.has_value()) {
        LOG_ERROR("particle object '%s' has unsupported extra renderer '%s'",
                  wppartobj.name.c_str(),
                  particle_renderer.name.c_str());
        return;
    }
    const auto& renderer          = *renderer_spec;
    const bool  render_rope_trail = renderer.UsesHistory();
    const bool  rope_shader       = renderer.IsRope();
    const bool  has_trail         = renderer.IsTrail();

    auto extra_node = std::make_shared<SceneNode>(
        Vector3f::Zero(), Vector3f::Ones(), Vector3f::Zero(),
        wppartobj.name + " " + particle_renderer.name);
    extra_node->ID() = wppartobj.id;
    extra_node->SetCamera(parent_node.Camera());

    auto effective_material = particle_obj.material;
    if (renderer.shader_selection == ParticleShaderSelection::StockRope) {
        effective_material.shader = "genericropeparticle";
    }

    WPShaderValueData svData;
    svData.CopyParallaxContractFrom(parent_sv);

    WPShaderInfo shaderInfo;
    shaderInfo.baseConstSvs                         = context.global_base_uniforms;
    shaderInfo.baseConstSvs["g_OrientationUp"]      = std::array { 0.0f, 1.0f, 0.0f };
    shaderInfo.baseConstSvs["g_OrientationRight"]   = std::array { 1.0f, 0.0f, 0.0f };
    shaderInfo.baseConstSvs["g_OrientationForward"] = std::array { 0.0f, 0.0f, 1.0f };
    shaderInfo.baseConstSvs["g_ViewUp"]             = std::array { 0.0f, 1.0f, 0.0f };
    shaderInfo.baseConstSvs["g_ViewRight"]          = std::array { 1.0f, 0.0f, 0.0f };
    shaderInfo.baseConstSvs["g_EyePosition"]        = std::array {
        static_cast<float>(context.ortho_w) / 2.0f,
        static_cast<float>(context.ortho_h) / 2.0f,
        1000.0f,
    };

    const uint16_t trail_length = render_rope_trail
        ? static_cast<uint16_t>(
              std::clamp(particle_renderer.segments, 1, kMaxParticleTrailSegments))
        : 0;
    auto mesh_capacity = CheckedParticleMeshCapacity(
        particle_debug_name,
        particle_renderer.name,
        { static_cast<uint64_t>(maxcount), static_cast<uint64_t>(mesh_instancecount) });
    if (! mesh_capacity.has_value()) return;
    if (render_rope_trail) {
        mesh_capacity = CheckedParticleMeshCapacity(
            particle_debug_name,
            particle_renderer.name,
            { static_cast<uint64_t>(maxcount),
              static_cast<uint64_t>(mesh_instancecount),
              static_cast<uint64_t>(trail_length) });
        if (! mesh_capacity.has_value()) return;
    }

    if (has_trail) {
        shaderInfo.baseConstSvs["g_RenderVar0"] = std::array {
            particle_renderer.length,
            particle_renderer.maxlength,
            render_rope_trail ? 0.0f : particle_renderer.minlength,
            static_cast<float>(maxcount - 1),
        };
        shaderInfo.combos["TRAILRENDERER"] = "1";
    }
    if (renderer.thick_format) shaderInfo.combos["THICKFORMAT"] = "1";
    if (rope_shader) {
        const uint32_t subdivision = static_cast<uint32_t>(
            std::lround(std::max(0.0f, particle_renderer.subdivision)));
        shaderInfo.combos["TRAILSUBDIVISION"] = std::to_string(subdivision);
    }
    if (! particle_obj.flags[wpscene::Particle::FlagEnum::spritenoframeblending] &&
        particle_obj.animationmode != "randomframe") {
        shaderInfo.combos["SPRITESHEETBLEND"] = "1";
    }

    SceneMaterial material;
    std::optional<MaterialLoadResult> material_result;
    try {
        material_result = LoadMaterial(*context.vfs,
                                       effective_material,
                                       context.scene.get(),
                                       extra_node.get(),
                                       &material,
                                       &svData,
                                       context.user_properties,
                                       &shaderInfo,
                                       rope_shader ? GeometryStagePolicy::Required
                                                   : GeometryStagePolicy::MatchMaterial);
    } catch (const std::exception& e) {
        LOG_ERROR("load particleobj '%s' extra renderer '%s' material exception: %s",
                  wppartobj.name.c_str(),
                  particle_renderer.name.c_str(),
                  e.what());
    }
    if (! material_result.has_value()) {
        LOG_ERROR("load particleobj '%s' extra renderer '%s' material faild",
                  wppartobj.name.c_str(),
                  particle_renderer.name.c_str());
        return;
    }
    LoadConstvalue(material, effective_material, shaderInfo);
    LoadUserShaderValue(material, effective_material, shaderInfo, context.user_properties);

    auto  spMesh = std::make_shared<SceneMesh>(true);
    const ParticleRenderPlan render_plan {
        .renderer         = renderer.kind,
        .shader_selection = renderer.shader_selection,
        .expansion        = material_result->geometry_stage_loaded
                    ? ParticleExpansionMode::GeometryPoint
                    : ParticleExpansionMode::IndexedQuad,
        .thick_format     = renderer.thick_format || material.hasSprite,
    };
    ConfigureParticleMesh(*spMesh, *mesh_capacity, render_plan);
    spMesh->AddMaterial(std::move(material));
    extra_node->AddMesh(spMesh);
    RegisterUserShaderValueBindings(
        context, effective_material, shaderInfo, extra_node.get(), wppartobj.id, wppartobj.name);

    subsystem.AddRenderOutput(spMesh, render_plan, spec_op);
    parent_node.AppendChild(extra_node);
    if (register_layer_nodes) {
        context.scene->objectRuntimeNodes[wppartobj.id].push_back(extra_node.get());
        context.scene->nodeOwners[extra_node.get()] = wppartobj.id;
    }
    context.shader_updater->SetNodeData(extra_node.get(), svData);
}

void ParseParticleObj(ParseContext& context, wpscene::WPParticleObject& wppartobj,
                      ParticleChildPtr child_ptr = {}) {
    struct ChildData {
        ChildData() = default;
        ChildData(const wpscene::ParticleChild& o)
            : type(o.type),
              maxcount(o.maxcount),
              controlpointstartindex(o.controlpointstartindex),
              probability(o.probability) {}
        std::string type { "static" };
        i32         maxcount { 20 };
        i32         controlpointstartindex { 0 };
        float       probability { 1.0f };
    };

    wpscene::Particle*         p_particle_obj { nullptr };
    std::shared_ptr<SceneNode> spNode;
    ChildData                  child_data;

    bool        is_child            = child_ptr.child != nullptr;
    const auto* visibility_contract = FindLayerVisibilityContract(context, wppartobj.id);
    if (! is_child &&
        ShouldDeferRuntimeLayerMaterialization(
            context, wppartobj.id, wppartobj.visible, visibility_contract, false)) {
        // Hidden dynamic particles keep only their logical registration at parse time. The check
        // uses effective initial visibility through the parent chain, so language branches hidden
        // by a runtime-controlled container do not allocate particle systems until that branch is
        // actually shown.
        RegisterLogicalParticleLayer(context, wppartobj);
        return;
    }

    if (is_child) {
        p_particle_obj = &(child_ptr.child->obj);
        spNode         = std::make_shared<SceneNode>(Vector3f(child_ptr.child->origin.data()),
                                                     Vector3f(child_ptr.child->scale.data()),
                                                     Vector3f(child_ptr.child->angles.data()),
                                                     child_ptr.child->name);
        child_data     = ChildData(*child_ptr.child);

    } else {
        p_particle_obj = &wppartobj.particleObj;
        spNode         = std::make_shared<SceneNode>(Vector3f(wppartobj.origin.data()),
                                                     Vector3f(wppartobj.scale.data()),
                                                     Vector3f(wppartobj.angles.data()),
                                                     wppartobj.name);
        spNode->ID()   = wppartobj.id;
    }

    const auto override =
        ResolveParticleSubsystemOverride(wppartobj.instanceoverride, is_child);

    auto& particle_obj = *p_particle_obj;
    auto& vfs          = *context.vfs;

    const auto& particle_renderer = particle_obj.renderers.at(0);
    const auto  renderer_spec     = DescribeParticleRender(particle_renderer.name);
    if (! renderer_spec.has_value()) {
        LOG_ERROR("particle object '%s' has unsupported renderer '%s'",
                  wppartobj.name.c_str(),
                  particle_renderer.name.c_str());
        return;
    }
    const auto& renderer          = *renderer_spec;
    const bool  render_rope_trail = renderer.UsesHistory();
    const bool  rope_shader       = renderer.IsRope();
    const bool  has_trail         = renderer.IsTrail();
    const std::string_view particle_debug_name =
        is_child ? std::string_view(child_ptr.child->name) : std::string_view(wppartobj.name);

    // Resolve the effective material without modifying the parsed particle asset. Child parsing,
    // runtime property registration, and diagnostics can therefore continue to inspect authored
    // data, while rope renderers receive the stock shader required by their segment ABI.
    auto effective_material = particle_obj.material;
    if (renderer.shader_selection == ParticleShaderSelection::StockRope) {
        effective_material.shader = "genericropeparticle";
    }

    // wppartobj.origin[1] = context.ortho_h - wppartobj.origin[1];

    if (particle_obj.flags[wpscene::Particle::FlagEnum::perspective]) {
        spNode->SetCamera("global_perspective");
    }

    SceneMaterial     material;
    WPShaderValueData svData;

    if (! is_child) {
        svData.parallaxDepth = { wppartobj.parallaxDepth[0], wppartobj.parallaxDepth[1] };
        svData.parallaxDepthAuthored = wppartobj.parallaxDepthAuthored;
    }

    WPShaderInfo shaderInfo;
    shaderInfo.baseConstSvs                         = context.global_base_uniforms;
    shaderInfo.baseConstSvs["g_OrientationUp"]      = std::array { 0.0f, 1.0f, 0.0f };
    shaderInfo.baseConstSvs["g_OrientationRight"]   = std::array { 1.0f, 0.0f, 0.0f };
    shaderInfo.baseConstSvs["g_OrientationForward"] = std::array { 0.0f, 0.0f, 1.0f };
    shaderInfo.baseConstSvs["g_ViewUp"]             = std::array { 0.0f, 1.0f, 0.0f };
    shaderInfo.baseConstSvs["g_ViewRight"]          = std::array { 1.0f, 0.0f, 0.0f };
    shaderInfo.baseConstSvs["g_EyePosition"]        = std::array {
        static_cast<float>(context.ortho_w) / 2.0f,
        static_cast<float>(context.ortho_h) / 2.0f,
        1000.0f,
    };

    const auto spawn_type              = ParseSpawnType(child_data.type);
    const u32  authored_child_maxcount = static_cast<u32>(std::max<i32>(1, child_data.maxcount));
    u32        mesh_instancecount = static_cast<u32>(std::max<i32>(1, child_ptr.max_instancecount));
    if (is_child) {
        switch (spawn_type) {
        case ParticleSubSystem::SpawnType::STATIC:
            // A static child relationship creates one persistent child emitter at this authored
            // transform. Its `maxcount` is a runtime cap for reusable child instances, not a request
            // to pre-seed N identical emitters. Pre-seeding duplicates this transform exactly; for
            // additive sprite effects such as the Matrix rain that stacks the same glyph stream on
            // itself, over-brightens the result, and washes the parent blue tint toward white.
            mesh_instancecount = 1;
            break;
        case ParticleSubSystem::SpawnType::EVENT_FOLLOW:
            // Follow instances are bounded by live parent particles as well as by authored child
            // maxcount. This keeps one live parent particle from reserving the default 20 follow
            // instances that Wallpaper Engine cannot actually use at the same time.
            mesh_instancecount = std::max<u32>(
                1, std::min<u32>(authored_child_maxcount, child_ptr.parent_live_particle_slots));
            break;
        case ParticleSubSystem::SpawnType::EVENT_SPAWN:
        case ParticleSubSystem::SpawnType::EVENT_DEATH:
            // Spawn/death children can outlive the triggering parent particle, so keep their local
            // authored cap. The cap is local to this child subsystem, not multiplied by ancestor
            // static child counts.
            mesh_instancecount = authored_child_maxcount;
            break;
        }
    }

    const u32 maxcount = std::clamp(particle_obj.maxcount, 1u, kMaxParticleCount);
    const uint16_t trail_length = render_rope_trail
        ? static_cast<uint16_t>(
              std::clamp(particle_renderer.segments, 1, kMaxParticleTrailSegments))
        : 0;
    const auto live_particle_slots = CheckedParticleMeshCapacity(
        particle_debug_name,
        particle_renderer.name,
        { static_cast<uint64_t>(maxcount), static_cast<uint64_t>(mesh_instancecount) });
    if (! live_particle_slots.has_value()) return;
    auto mesh_capacity = live_particle_slots;
    if (render_rope_trail) {
        mesh_capacity = CheckedParticleMeshCapacity(
            particle_debug_name,
            particle_renderer.name,
            { static_cast<uint64_t>(maxcount),
              static_cast<uint64_t>(mesh_instancecount),
              static_cast<uint64_t>(trail_length) });
        if (! mesh_capacity.has_value()) return;
    }

    if (has_trail) {
        shaderInfo.baseConstSvs["g_RenderVar0"] = std::array {
            particle_renderer.length,
            particle_renderer.maxlength,
            // Sprite trails clamp length to this authored floor. Rope trails overwrite z each
            // frame with the normalized remainder, so they keep a parse-time zero here.
            render_rope_trail ? 0.0f : particle_renderer.minlength,
            static_cast<float>(maxcount - 1),
        };
        shaderInfo.combos["TRAILRENDERER"] = "1";
    }
    if (renderer.thick_format) {
        // Wallpaper Engine unconditionally selects THICKFORMAT for the stock rope renderer. The
        // endpoint size and color are part of that ABI and must reach the geometry shader so each
        // segment interpolates into the next particle instead of changing abruptly at its boundary.
        shaderInfo.combos["THICKFORMAT"] = "1";
    }
    if (rope_shader) {
        // Wallpaper Engine selects the GS-enabled vertex branch and expands each rope segment in
        // genericropeparticle.geom. Its subdivision value is the count of additional
        // pairs emitted between endpoints, so forward the authored value verbatim to the shader
        // combo rather than pre-tessellating it in the particle uploader.
        const uint32_t subdivision = static_cast<uint32_t>(
            std::lround(std::max(0.0f, particle_renderer.subdivision)));
        shaderInfo.combos["TRAILSUBDIVISION"] = std::to_string(subdivision);
    }

    if (! particle_obj.flags[wpscene::Particle::FlagEnum::spritenoframeblending] &&
        particle_obj.animationmode != "randomframe") {
        // randomframe encodes one atlas cell per particle. Frame blending would mix that cell with
        // the next one, so keep SPRITESHEETBLEND for sequenced sprite playback only.
        shaderInfo.combos["SPRITESHEETBLEND"] = "1";
    }

    std::optional<MaterialLoadResult> material_result;
    try {
        material_result = LoadMaterial(vfs,
                                       effective_material,
                                       context.scene.get(),
                                       spNode.get(),
                                       &material,
                                       &svData,
                                       context.user_properties,
                                       &shaderInfo,
                                       rope_shader ? GeometryStagePolicy::Required
                                                   : GeometryStagePolicy::MatchMaterial);
    } catch (const std::exception& e) {
        LOG_ERROR("load particleobj '%s' material exception: %s", wppartobj.name.c_str(), e.what());
    }
    if (! material_result.has_value()) {
        LOG_ERROR("load particleobj '%s' material faild", wppartobj.name.c_str());
        return;
    }
    LoadConstvalue(material, effective_material, shaderInfo);
    LoadUserShaderValue(material, effective_material, shaderInfo, context.user_properties);
    auto  spMesh             = std::make_shared<SceneMesh>(true);
    auto& mesh               = *spMesh;
    auto  animationmode      = ToAnimMode(particle_obj.animationmode);
    auto  sequencemultiplier = particle_obj.sequencemultiplier;
    float sprite_frame_count = 0.0f;
    if (const auto it = material.customShader.constValues.find("g_RenderVar1");
        it != material.customShader.constValues.end() && it->second.size() >= 3) {
        // g_RenderVar1.z is the SpriteAnimation frame count of the current particle texture.
        sprite_frame_count = it->second[2];
    }

    const ParticleRenderPlan render_plan {
        .renderer         = renderer.kind,
        .shader_selection = renderer.shader_selection,
        .expansion        = material_result->geometry_stage_loaded
                    ? ParticleExpansionMode::GeometryPoint
                    : ParticleExpansionMode::IndexedQuad,
        .thick_format     = renderer.thick_format || material.hasSprite,
    };
    ConfigureParticleMesh(mesh, *mesh_capacity, render_plan);

    const std::string geometry_shader = material_result->geometry_stage_loaded
        ? effective_material.shader + ".geom"
        : "(none)";
    LOG_INFO("ParticleRenderPlan: object='%.*s' renderer='%s' shader='%s' vertex='%s.vert' "
             "geometry='%s' fragment='%s.frag' expansion=%s primitive=%s thick=%s history=%s",
             static_cast<int>(particle_debug_name.size()),
             particle_debug_name.data(),
             particle_renderer.name.c_str(),
             effective_material.shader.c_str(),
             effective_material.shader.c_str(),
             geometry_shader.c_str(),
             effective_material.shader.c_str(),
             render_plan.expansion == ParticleExpansionMode::GeometryPoint ? "GeometryPoint"
                                                                            : "IndexedQuad",
             mesh.Primitive() == MeshPrimitive::POINT ? "point" : "triangle",
             render_plan.thick_format ? "true" : "false",
             render_plan.UsesHistory() ? "true" : "false");

    std::function<void(float)> trail_uniform_update;
    if (render_rope_trail) {
        std::weak_ptr<SceneMesh> weak_mesh = spMesh;
        trail_uniform_update = [weak_mesh](float normalized_remainder) {
            const auto mesh = weak_mesh.lock();
            if (! mesh || mesh->Material() == nullptr) return;
            auto& values = mesh->Material()->customShader.constValues;
            const auto render_var = values.find("g_RenderVar0");
            if (render_var == values.end() || render_var->second.size() < 3) return;
            render_var->second[2] = normalized_remainder;
        };
    }

    const ParticleRawGenSpecOp spec_op = [=](const Particle& p, const ParticleRawGenSpec& spec) {
        auto& lifetime = *(spec.lifetime);
        if (lifetime <= 0.0f) {
            lifetime = 0.0f;
            return;
        }
        switch (animationmode) {
        case ParticleAnimationMode::RANDOMONE:
            lifetime = sprite_frame_count > 1.0f
                           ? RandomParticleFrameLifetime(p, sprite_frame_count)
                           : std::floor(p.init.lifetime);
            break;
        case ParticleAnimationMode::SEQUENCE:
            lifetime = (1.0f - (p.lifetime / p.init.lifetime)) * sequencemultiplier;
            break;
        }
    };
    auto particleSub = std::make_unique<ParticleSubSystem>(
        *context.scene->paritileSys,
        spMesh,
        maxcount,
        override.rate,
        child_data.maxcount,
        child_data.probability,
        spawn_type,
        render_plan,
        spec_op,
        trail_length,
        render_rope_trail ? static_cast<double>(particle_renderer.length) : 0.0,
        std::move(trail_uniform_update));
    auto* particle_subsystem = particleSub.get();
    particleSub->SetSceneNode(spNode.get());
    // instanceoverride.size is baked into initializer output during cold parse. Keep that parsed
    // multiplier as the runtime reference so later user-property size edits can rescale live
    // particles by ratio instead of mistaking the multiplier for an absolute particle size.
    particleSub->SetRuntimeSizeReference(override.size);

    LoadEmitter(*context.scene->paritileSys, *particleSub, particle_obj, override.count);
    LoadInitializer(*particleSub, particle_obj, override, *context.scene->paritileSys);
    LoadOperator(*context.scene->paritileSys, *particleSub, particle_obj, override);
    LoadControlPoint(*particleSub, particle_obj, override);

    mesh.AddMaterial(std::move(material));
    spNode->AddMesh(spMesh);
    RegisterUserShaderValueBindings(
        context, effective_material, shaderInfo, spNode.get(), wppartobj.id, wppartobj.name);

    if (! is_child) {
        ConfigureBoneAttachment(context,
                                wppartobj.id,
                                wppartobj.parent,
                                wppartobj.attachment,
                                Eigen::Affine3f(spNode->GetLocalTrans().cast<float>()),
                                "particle object",
                                wppartobj.name,
                                svData);
    }

    for (auto& child : particle_obj.children) {
        ParseParticleObj(context,
                         wppartobj,
                         {
                             .child                      = &child,
                             .node_parent                = spNode.get(),
                             .particle_parent            = particleSub.get(),
                             .max_instancecount          = static_cast<i32>(mesh_instancecount),
                             .parent_live_particle_slots = *live_particle_slots,
                         });
    }

    for (size_t renderer_index = 1; renderer_index < particle_obj.renderers.size();
         renderer_index++) {
        AttachExtraParticleRenderer(context,
                                    wppartobj,
                                    particle_obj,
                                    particle_obj.renderers[renderer_index],
                                    *spNode,
                                    *particleSub,
                                    svData,
                                    spec_op,
                                    maxcount,
                                    mesh_instancecount,
                                    particle_debug_name,
                                    ! is_child);
    }

    if (is_child)
        child_ptr.particle_parent->AddChild(std::move(particleSub));
    else {
        context.scene->paritileSys->subsystems.emplace_back(std::move(particleSub));
        context.scene->objectRuntimeParticleSubsystems[wppartobj.id].push_back(particle_subsystem);
    }

    if (is_child) {
        child_ptr.node_parent->AppendChild(spNode);
        svData.InheritParentTransform(child_ptr.node_parent);
    } else {
        if (LayerUsesRoutedParent(context, wppartobj.parent, wppartobj.attachment)) {
            ConfigureInheritedParentBinding(context, wppartobj.parent, svData);
            context.scene->sceneGraph->AppendChild(spNode);
            RegisterRenderOrderProxyChild(context, wppartobj.parent, spNode, wppartobj.id);
        } else if (wppartobj.parent != 0) {
            if (wppartobj.attachment.empty()) {
                ConfigureInheritedParentBinding(context, wppartobj.parent, svData);
            }
            context.scene->sceneGraph->AppendChild(spNode);
        } else {
            AttachNodeToScene(context, spNode, wppartobj.parent, wppartobj.name, &svData);
        }
        context.object_nodes[wppartobj.id] = spNode;
        context.scene->objectRuntimeNodes[wppartobj.id].push_back(spNode.get());
        context.scene->nodeOwners[spNode.get()] = wppartobj.id;
        if (auto* object = context.scene->FindSceneObject(wppartobj.id)) {
            object->set_source_node(spNode.get());
        }
        RegisterLayerSceneState(
            context, wppartobj.id, wppartobj.parent, wppartobj.attachment, wppartobj.visible);
        context.scene->ApplyLayerVisibility(wppartobj.id);
    }
    context.shader_updater->SetNodeData(spNode.get(), svData);

    if (! is_child && particle_obj.starttime > 0) {
        particle_subsystem->Prewarm(static_cast<double>(particle_obj.starttime));
    }
}

void ParseLightObj(ParseContext& context, wpscene::WPLightObject& light_obj) {
    auto node = std::make_shared<SceneNode>(Vector3f(light_obj.origin.data()),
                                            Vector3f(light_obj.scale.data()),
                                            Vector3f(light_obj.angles.data()),
                                            light_obj.name);

    context.scene->lights.emplace_back(std::make_unique<SceneLight>(
        Vector3f(light_obj.color.data()), light_obj.radius, light_obj.intensity));

    auto& light = *(context.scene->lights.back());
    // Workshop scene.pkg and editor assets write prefixed tokens (lpoint/lspot/
    // ldirectional/ltube). Unpacked defaultprojects and older scene.json keep the
    // unprefixed names. Accept both.
    if (light_obj.light == "lpoint" || light_obj.light == "point") {
        light.setType(SceneLightType::Point);
    } else if (light_obj.light == "lspot" || light_obj.light == "spot") {
        light.setType(SceneLightType::Spot);
    } else if (light_obj.light == "ldirectional" || light_obj.light == "directional") {
        light.setType(SceneLightType::Directional);
    } else if (light_obj.light == "ltube" || light_obj.light == "tube") {
        light.setType(SceneLightType::Tube);
    } else {
        light.setType(SceneLightType::Other);
    }
    light.setCastVolumetrics(light_obj.castvolumetrics);
    light.setDensity(light_obj.density);
    light.setVolumetricsExponent(light_obj.volumetricsexponent);
    light.setInnerCone(light_obj.innercone);
    light.setOuterCone(light_obj.outercone);
    light.setCastsShadows(light_obj.castshadows);
    light.setExponent(light_obj.exponent);
    light.setCascadeDistances(light_obj.cascadedistance0, light_obj.cascadedistance1,
                              light_obj.cascadedistance2);
    if (light_obj.usecookie && ! light_obj.cookie.empty()) {
        light.setCookie(light_obj.cookie);
    }

    WPShaderValueData svData;
    svData.parallaxDepth = light_obj.parallaxDepth;
    if (LayerUsesRoutedParent(context, light_obj.parent, {})) {
        ConfigureInheritedParentBinding(context, light_obj.parent, svData);
    } else if (light_obj.parent != 0) {
        ApplyParentTransformContract(
            context, ParentTransformContract::ParallaxAnchorWhenCompatible(light_obj.parent), svData);
    }
    // Keep the node as a parent source for children; do not shift the light itself.
    svData.SuppressOwnModelParallax();

    context.scene->objectRuntimeLights[light_obj.id].push_back(&light);
    light.setNode(node);

    if (light_obj.parent != 0) {
        context.scene->sceneGraph->AppendChild(node);
    } else {
        AttachNodeToScene(context, node, light_obj.parent, light_obj.name, &svData);
    }
    context.shader_updater->SetNodeData(node.get(), svData);
    context.object_nodes[light_obj.id] = node;
    context.scene->objectRuntimeNodes[light_obj.id].push_back(node.get());
    context.scene->nodeOwners[node.get()] = light_obj.id;
    if (auto* object = context.scene->FindSceneObject(light_obj.id)) {
        object->set_source_node(node.get());
    }
    RegisterLayerSceneState(context, light_obj.id, light_obj.parent, {}, light_obj.visible);
    context.scene->ApplyLayerVisibility(light_obj.id);
}

void ParseEmptyObj(ParseContext& context, WPEmptyObject& empty_obj) {
    const auto node_origin =
        empty_obj.is_camera_layer
            ? context.scene->ResolveCameraLayerNodeTranslation(empty_obj.origin)
            : Vector3f(empty_obj.origin.data());
    auto node  = std::make_shared<SceneNode>(node_origin,
                                             Vector3f(empty_obj.scale.data()),
                                             Vector3f(empty_obj.angles.data()),
                                             empty_obj.name);
    node->ID() = empty_obj.id;

    WPShaderValueData svData;
    svData.parallaxDepth = empty_obj.parallaxDepth;
    svData.parallaxDepthAuthored = empty_obj.parallaxDepthAuthored;
    ConfigureBoneAttachment(context,
                            empty_obj.id,
                            empty_obj.parent,
                            empty_obj.attachment,
                            Eigen::Affine3f(node->GetLocalTrans().cast<float>()),
                            "object",
                            empty_obj.name,
                            svData);

    if (empty_obj.parent != 0) {
        // Official attach writes +0x180/+0x190 only (0x1401de470). Image factory
        // 0x1401902b6 constructs 0x4d0 objects at 0x1401fac50 and installs
        // vt 0x1404911a8: vt+0x70 0x1401fd510 / vt+0x78 0x1401fd5c0. Inherit
        // would replace BoneAttachment and drop that dest path. Same guard as
        // the image flat list at the attachment.empty() check.
        if (empty_obj.attachment.empty()) {
            ConfigureInheritedParentBinding(context, empty_obj.parent, svData);
        }
        context.scene->sceneGraph->AppendChild(node);
    } else {
        AttachNodeToScene(context, node, empty_obj.parent, empty_obj.name, &svData);
    }
    context.object_nodes[empty_obj.id] = node;
    context.scene->objectRuntimeNodes[empty_obj.id].push_back(node.get());
    context.scene->nodeOwners[node.get()] = empty_obj.id;
    if (auto* object = context.scene->FindSceneObject(empty_obj.id)) {
        object->set_source_node(node.get());
    }
    context.shader_updater->SetNodeData(node.get(), svData);
    RegisterLayerSceneState(
        context, empty_obj.id, empty_obj.parent, empty_obj.attachment, empty_obj.visible);
    context.scene->ApplyLayerVisibility(empty_obj.id);

    if (empty_obj.is_camera_layer) {
        Scene::CameraLayerRuntimeState camera_layer;
        // Wallpaper Engine writes "default" for the normal scene camera. Hanabi's matching
        // orthographic camera is named "global", so normalize the authored token once and keep
        // the rest of the runtime path name-based for future camera targets.
        camera_layer.camera_name =
            empty_obj.camera_name.empty() || empty_obj.camera_name == "default"
                ? "global"
                : empty_obj.camera_name;
        camera_layer.node                    = node;
        camera_layer.origin                  = empty_obj.origin;
        camera_layer.angles                  = empty_obj.angles;
        camera_layer.zoom                    = empty_obj.zoom;
        camera_layer.fov                     = empty_obj.fov;
        const bool first_camera_registration = context.scene->cameraLayers.count(empty_obj.id) == 0;
        context.scene->cameraLayers[empty_obj.id] = camera_layer;
        if (first_camera_registration) context.scene->cameraLayerOrder.push_back(empty_obj.id);
        context.scene->UpdateActiveCameraLayer();

        size_t camera_path_count = 0;
        if (! empty_obj.camera_path.empty()) {
            nlohmann::json    path_json;
            const std::string asset_path = "/assets/" + empty_obj.camera_path;
            if (context.vfs != nullptr && context.vfs->Contains(asset_path) &&
                PARSE_JSON(fs::GetFileContent(*context.vfs, asset_path), path_json) &&
                path_json.contains("paths") && path_json.at("paths").is_array()) {
                camera_path_count = path_json.at("paths").size();
            }
        }
        LOG_INFO("SceneCameraLayerParsed: id=%d name='%s' camera='%s' origin=[%.3f, %.3f, %.3f] "
                 "zoom=%.3f fov=%.3f visible=%s path='%s' path-count=%zu",
                 empty_obj.id,
                 empty_obj.name.c_str(),
                 camera_layer.camera_name.c_str(),
                 empty_obj.origin[0],
                 empty_obj.origin[1],
                 empty_obj.origin[2],
                 empty_obj.zoom,
                 empty_obj.fov,
                 context.scene->IsLayerVisible(empty_obj.id) ? "true" : "false",
                 empty_obj.camera_path.c_str(),
                 camera_path_count);
    }
}

bool ShapeEffectRequestsDirectDraw(const WPShapeObject& shape_obj) {
    // Wallpaper Engine marks shader-authored shape output with the DIRECTDRAW combo on the effect
    // chain. Ask the parsed effect model for that shader contract instead of inferring it from the
    // outer shape geometry string; the geometry name only says what primitive the editor displayed,
    // while DIRECTDRAW is the actual render-path switch that means no image/model source exists.
    return std::any_of(shape_obj.effects.begin(), shape_obj.effects.end(), [](const auto& effect) {
        return effect.HasEnabledCombo("DIRECTDRAW");
    });
}

std::array<float, 2> ResolveImplicitDirectDrawShapeVisualSize(const ParseContext& context) {
    // Shape parse vt+0x40 0x14025fac0: movd xmm0, [scene+0x88]; pshufd 0;
    // cvtdq2ps; movsd [obj+0x2f0], xmm1. scene+0x88 is orthogonalprojection
    // height (0x1401875f2 / 0x1401875fb). Both +0x2f0 and +0x2f4 get that
    // integer as float, so the implicit card is height×height.
    const float height = static_cast<float>(std::max(1, context.ortho_h));
    return { height, height };
}

struct DirectDrawShapeMetrics {
    std::array<float, 2> visual_size { 1.0f, 1.0f };
    std::array<float, 2> effect_source_size { 0.0f, 0.0f };
    bool                 effect_source_screen_bound { false };
    bool                 final_texcoord_bounds_enabled { false };
    std::array<float, 4> final_texcoord_bounds { 0.0f, 0.0f, 1.0f, 1.0f };
    const char*          visual_policy { "authored-size" };
    const char*          effect_source_policy { "layer-size" };
    const char*          final_uv_policy { "default" };
};

DirectDrawShapeMetrics ResolveDirectDrawShapeMetrics(const ParseContext& context,
                                                     const WPShapeObject& shape_obj) {
    DirectDrawShapeMetrics metrics;
    if (shape_obj.has_size) {
        metrics.visual_size        = shape_obj.size;
        metrics.effect_source_size = shape_obj.size;
    } else {
        metrics.visual_size        = ResolveImplicitDirectDrawShapeVisualSize(context);
        metrics.effect_source_size = metrics.visual_size;
        metrics.visual_policy      = "implicit-ortho-height-square";
        metrics.effect_source_policy = "layer-size";
    }

    return metrics;
}

void ParseShapeObj(ParseContext& context, WPShapeObject& shape_obj,
                   bool force_runtime_materialization = false) {
    if (! shape_obj.visible && ! force_runtime_materialization) return;

    const bool direct_draw_shape = ShapeEffectRequestsDirectDraw(shape_obj);
    if (! direct_draw_shape) {
        // Unsupported or effect-less shape layers still need to behave like transform containers.
        // Registering a normal empty object preserves parent bindings, scripts, and child ordering
        // while making the missing drawable path explicit in the log instead of silently dropping
        // the authored layer.
        LOG_INFO("SceneShapeObjectFallback: id=%d name='%s' shape='%s' effects=%zu",
                 shape_obj.id,
                 shape_obj.name.c_str(),
                 shape_obj.shape.c_str(),
                 shape_obj.effects.size());

        WPEmptyObject empty_obj;
        empty_obj.id              = shape_obj.id;
        empty_obj.name            = shape_obj.name;
        empty_obj.origin          = shape_obj.origin;
        empty_obj.scale           = shape_obj.scale;
        empty_obj.angles          = shape_obj.angles;
        empty_obj.parallaxDepth   = shape_obj.parallaxDepth;
        empty_obj.parallaxDepthAuthored = shape_obj.parallaxDepthAuthored;
        empty_obj.visible         = shape_obj.visible;
        empty_obj.visible_binding = shape_obj.visible_binding;
        empty_obj.parent          = shape_obj.parent;
        empty_obj.attachment      = shape_obj.attachment;
        ParseEmptyObj(context, empty_obj);
        return;
    }

    auto& vfs = *context.vfs;

    wpscene::WPMaterial transparent_source_material;
    nlohmann::json      transparent_source_json;
    if (! PARSE_JSON(fs::GetFileContent(vfs, "/assets/materials/util/effectpassthrough.json"),
                     transparent_source_json) ||
        ! transparent_source_material.FromJson(transparent_source_json)) {
        LOG_ERROR(
            "SceneShapeDirectDraw: layer=%d name='%s' failed to load transparent source material",
            shape_obj.id,
            shape_obj.name.c_str());
        return;
    }

    EnsureSystemTextureRegistered(*context.scene, kSyntheticDirectDrawShapeTextureName);
    if (transparent_source_material.textures.empty()) {
        transparent_source_material.textures.resize(1);
    }
    transparent_source_material.textures[0] = std::string(kSyntheticDirectDrawShapeTextureName);
    // Official assets/materials/util/effectpassthrough.json and
    // materials/effects/lightshafts.json both author blending "normal".
    // Image ctor / shape parse do not rewrite that to additive.

    const DirectDrawShapeMetrics metrics = ResolveDirectDrawShapeMetrics(context, shape_obj);

    // Shape direct-draw layers have no image asset because the effect shader owns the visible
    // pixels (`DIRECTDRAW=1`). Synthesize a fully transparent image source only to reuse the
    // established image-effect camera, ping-pong render targets, visibility contracts, and final
    // composite path; the authored shape string stays metadata, not a render-path discriminator.
    wpscene::WPImageObject image_obj;
    image_obj.id               = shape_obj.id;
    image_obj.name             = shape_obj.name;
    image_obj.origin           = shape_obj.origin;
    image_obj.scale            = shape_obj.scale;
    image_obj.angles           = shape_obj.angles;
    image_obj.size             = metrics.visual_size;
    image_obj.parallaxDepth    = shape_obj.parallaxDepth;
    image_obj.parallaxDepthAuthored = shape_obj.parallaxDepthAuthored;
    image_obj.color            = shape_obj.color;
    image_obj.alpha            = shape_obj.alpha;
    image_obj.brightness       = shape_obj.brightness;
    image_obj.visible          = shape_obj.visible;
    image_obj.visible_binding  = shape_obj.visible_binding;
    image_obj.image            = "__hanabi_shape_directdraw";
    image_obj.parent           = shape_obj.parent;
    image_obj.attachment       = shape_obj.attachment;
    image_obj.effectSourceSize = metrics.effect_source_size;
    image_obj.effectSourceScreenBound = metrics.effect_source_screen_bound;
    image_obj.effectFinalTexCoordBoundsEnabled = metrics.final_texcoord_bounds_enabled;
    image_obj.effectFinalTexCoordBounds        = metrics.final_texcoord_bounds;
    image_obj.material         = std::move(transparent_source_material);
    image_obj.effects          = std::move(shape_obj.effects);
    image_obj.nopadding        = true;
    image_obj.config.finalOutputCapability = FinalOutputCapability::SceneAuthoredWriter;

    LOG_INFO("SceneShapeDirectDraw: materialize layer=%d name='%s' shape='%s' effects=%zu "
             "visual-size=[%.3f, %.3f] visual-policy=%s effect-source-policy=%s "
             "effect-source-size=[%.3f, %.3f] authored-size=%s transparent-texture='%.*s' "
             "source-blend='%s'",
             image_obj.id,
             image_obj.name.c_str(),
             shape_obj.shape.c_str(),
             image_obj.effects.size(),
             image_obj.size[0],
             image_obj.size[1],
             metrics.visual_policy,
             metrics.effect_source_policy,
             image_obj.effectSourceSize[0],
             image_obj.effectSourceSize[1],
             shape_obj.has_size ? "true" : "false",
             static_cast<int>(kSyntheticDirectDrawShapeTextureName.size()),
             kSyntheticDirectDrawShapeTextureName.data(),
             image_obj.material.blending.c_str());

    ParseImageObj(context, image_obj, force_runtime_materialization);
}

template<typename T>
void AddWPObject(std::vector<WPObjectVar>& objs, const nlohmann::json& json_obj, fs::VFS& vfs,
                 const UserPropertyMap* user_properties) {
    T wpobj;
    if (! wpobj.FromJson(json_obj, vfs)) {
        LOG_ERROR("parse scene object failed, name: %s", wpobj.name.c_str());
        return;
    }

    const auto* visible_json     = FindVisibleProperty(json_obj);
    const auto  visible_contract = BuildVisibilityContract(
        wpobj.visible, wpobj.visible_binding, visible_json, user_properties);
    const bool has_visible_runtime_binding = visible_contract.has_script ||
                                             visible_contract.has_user_binding ||
                                             visible_contract.has_animation;
    wpobj.visible                          = visible_contract.initial_visible;
    if (has_visible_runtime_binding && ! std::is_same_v<T, wpscene::WPImageObject> &&
        ! std::is_same_v<T, wpscene::WPParticleObject> &&
        ! std::is_same_v<T, wpscene::WPTextObject> && ! std::is_same_v<T, WPShapeObject>) {
        // Legacy non-lazy layer types still need a concrete runtime node to receive visibility
        // writes. Image and shape-direct-draw layers now use explicit
        // materialize-but-apply-initial-visibility handling, and particle/text layers keep their
        // existing lightweight lazy nodes until first shown.
        wpobj.visible = true;
    }
    if constexpr (std::is_same_v<T, wpscene::WPImageObject>) {
        // This image parse log intentionally mirrors the runtime geometry fields that decide
        // whether utility layers become drawable. It makes project-layer regressions visible in
        // run logs before the render graph has a chance to create a framebuffer feedback pass.
        LOG_INFO("SceneObjectParsed: source=scene-load kind=image id=%d name='%s' "
                 "origin=[%.3f, %.3f, %.3f] size=[%.3f, %.3f] fullscreen=%s autosize=%s "
                 "projectlayer=%s image='%s' effects=%zu",
                 wpobj.id,
                 wpobj.name.c_str(),
                 wpobj.origin[0],
                 wpobj.origin[1],
                 wpobj.origin[2],
                 wpobj.size[0],
                 wpobj.size[1],
                 wpobj.fullscreen ? "true" : "false",
                 wpobj.autosize ? "true" : "false",
                 wpobj.projectlayer ? "true" : "false",
                 wpobj.image.c_str(),
                 wpobj.effects.size());
    } else if constexpr (std::is_same_v<T, WPShapeObject>) {
        // Shape/direct-draw layers do not carry an image model, so this parser log is the only
        // early proof that the layer was classified as drawable effect content rather than as a
        // transform-only empty object.
        LOG_INFO("SceneObjectParsed: source=scene-load kind=shape id=%d name='%s' "
                 "origin=[%.3f, %.3f, %.3f] size=[%.3f, %.3f] has-size=%s shape='%s' effects=%zu",
                 wpobj.id,
                 wpobj.name.c_str(),
                 wpobj.origin[0],
                 wpobj.origin[1],
                 wpobj.origin[2],
                 wpobj.size[0],
                 wpobj.size[1],
                 wpobj.has_size ? "true" : "false",
                 wpobj.shape.c_str(),
                 wpobj.effects.size());
    } else if constexpr (std::is_same_v<T, WPModelObject>) {
        // Model objects are intentionally logged before materialization because a missing model
        // parse would otherwise look like an empty scene: the object has no image/particle/text
        // discriminator, so this line proves the 3D-specific dispatch path claimed it.
        LOG_INFO("SceneObjectParsed: source=scene-load kind=model id=%d name='%s' model='%s' "
                 "origin=[%.3f, %.3f, %.3f] reflected=%s skin=%d",
                 wpobj.id,
                 wpobj.name.c_str(),
                 wpobj.model.c_str(),
                 wpobj.origin[0],
                 wpobj.origin[1],
                 wpobj.origin[2],
                 wpobj.reflected ? "true" : "false",
                 wpobj.skin);
    }
    objs.push_back(wpobj);
}

std::optional<int32_t> GetObjectId(const WPObjectVar& obj) {
    return std::visit(
        visitor::overload {
            [](const auto& value) -> std::optional<int32_t> {
                return value.id;
            },
        },
        obj);
}

std::string GetObjectName(const WPObjectVar& obj) {
    return std::visit(
        [](const auto& value) {
            return value.name;
        },
        obj);
}

int32_t GetObjectParentId(const WPObjectVar& obj) {
    return std::visit(visitor::overload {
                          [](const wpscene::WPSoundObject&) {
                              return 0;
                          },
                          [](const auto& value) {
                              return value.parent;
                          },
                      },
                      obj);
}

std::string GetObjectAttachment(const WPObjectVar& obj) {
    return std::visit(visitor::overload {
                          [](const wpscene::WPSoundObject&) {
                              return std::string {};
                          },
                          [](const wpscene::WPLightObject&) {
                              return std::string {};
                          },
                          [](const auto& value) {
                              return value.attachment;
                          },
                      },
                      obj);
}

void FillSceneObjectIdentity(Scene& scene, const WPObjectVar& obj) {
    std::visit(
        visitor::overload {
            [&scene](const wpscene::WPSoundObject& value) {
                if (value.id == 0) return;
                auto& object = scene.EnsureSceneObject(value.id);
                object.set_kind(SceneObjectKind::Sound);
                object.set_name(value.name);
                object.set_local_visible(value.visible);
            },
            [&scene](const auto& value) {
                if (value.id == 0) return;
                auto& object = scene.EnsureSceneObject(value.id);
                object.set_name(value.name);
                object.set_origin(Vector3f(value.origin.data()));
                object.set_scale(Vector3f(value.scale.data()));
                object.set_angles(Vector3f(value.angles.data()));
                object.set_local_visible(value.visible);
                if constexpr (requires { value.parallaxDepth; }) {
                    object.set_parallax_depth(
                        Vector2f(value.parallaxDepth[0], value.parallaxDepth[1]));
                }
                if constexpr (requires { value.parallaxDepthAuthored; }) {
                    object.set_parallax_depth_authored(value.parallaxDepthAuthored);
                }
                if constexpr (std::is_same_v<std::decay_t<decltype(value)>, wpscene::WPImageObject>) {
                    object.set_kind(SceneObjectKind::Image);
                    // IMAGE_SIZE_PROP 0x1401a4046 writes +0x2f0. Image vt+0x110
                    // 0x1402065e0 then vt+0xb0 (IMAGE_VT_B0). EFFECT_COUNT +0x320
                    // is +1 per JSON effect (0x1401e893d, ctor +0x140=0).
                    object.set_dest_size(value.size[0], value.size[1]);
                    object.set_effect_count(static_cast<int32_t>(value.effects.size()));
                    // PARSE_304_BIT4 / PUPPET_304_BIT4: bit4 leftover-MVP
                    // (IMAGE_VT_F8), not POSTFX last-pass dest-card.
                    object.ApplyParse304Bit4(value.colorBlendMode);
                    object.ApplyPuppet304Bit4();
                    object.PublishDestDrawMeshes();
                } else if constexpr (std::is_same_v<std::decay_t<decltype(value)>,
                                                    wpscene::WPParticleObject>) {
                    object.set_kind(SceneObjectKind::Particle);
                } else if constexpr (std::is_same_v<std::decay_t<decltype(value)>,
                                                    wpscene::WPTextObject>) {
                    object.set_kind(SceneObjectKind::Text);
                    // EFFECT_COUNT: text vt+0x40 IMAGE_PARSE. +0x2f0 is TEXT_2F0
                    // layout AABB, not JSON size (TEXT_AABB_FILL).
                    object.set_effect_count(static_cast<int32_t>(value.effects.size()));
                    object.ApplyParse304Bit4(value.colorBlendMode);
                } else if constexpr (std::is_same_v<std::decay_t<decltype(value)>,
                                                    wpscene::WPLightObject>) {
                    object.set_kind(SceneObjectKind::Light);
                } else if constexpr (std::is_same_v<std::decay_t<decltype(value)>, WPModelObject>) {
                    object.set_kind(SceneObjectKind::Model);
                } else if constexpr (std::is_same_v<std::decay_t<decltype(value)>, WPShapeObject>) {
                    object.set_kind(SceneObjectKind::Shape);
                } else if constexpr (std::is_same_v<std::decay_t<decltype(value)>, WPEmptyObject>) {
                    object.set_kind(value.is_camera_layer ? SceneObjectKind::Camera
                                                          : SceneObjectKind::Empty);
                }
            },
        },
        obj);
}

bool IsObjectVisible(const WPObjectVar& obj) {
    return std::visit(
        [](const auto& value) {
            return value.visible;
        },
        obj);
}

std::optional<WPDynamicValue> ParsePropertyBaseValue(const nlohmann::json& property_json,
                                                     WPDynamicValue::Type  hint) {
    if (! property_json.is_object() || ! property_json.contains("value")) return std::nullopt;
    return WPDynamicValue::FromJsonLiteral(property_json.at("value"), hint);
}

void RegisterScenePropertyAnimationBinding(ParseContext& context, const nlohmann::json& object_json,
                                           std::string_view     property_name,
                                           WPDynamicValue::Type hint) {
    if (! object_json.is_object() || ! object_json.contains("id") ||
        ! object_json.contains("name") || ! object_json.contains(property_name)) {
        return;
    }

    const auto& property_json = object_json.at(property_name);
    if (! property_json.is_object() || ! property_json.contains("animation")) return;

    int32_t object_id { 0 };
    if (! GetJsonValue(__SHORT_FILE__,
                       __FUNCTION__,
                       __LINE__,
                       object_json.at("id"),
                       object_id,
                       false,
                       "",
                       false)) {
        return;
    }

    // Property animation registrations still target SceneNode-backed values. Camera layers keep a
    // node too, but they must dispatch through the camera target so origin/zoom keyframes update
    // the active SceneCamera instead of only the invisible layer node.
    const bool camera_registration =
        IsCameraLayerObjectJson(object_json) && IsCameraLayerRuntimeProperty(property_name);
    const auto object_node_it = context.object_nodes.find(object_id);
    if (object_node_it == context.object_nodes.end()) return;

    WPPropertyAnimationDefinition animation_definition;
    if (! ParsePropertyAnimationDefinition(property_json, hint, animation_definition)) return;

    WPUserSetting setting;
    if (! ParseUserSetting(property_json, setting, hint)) return;

    std::string object_name;
    if (! GetJsonValue(__SHORT_FILE__,
                       __FUNCTION__,
                       __LINE__,
                       object_json.at("name"),
                       object_name,
                       false,
                       "",
                       false)) {
        object_name = std::to_string(object_id);
    }

    context.scene->propertyAnimationRegistrations.push_back(WPSceneScriptRegistration {
        .object_id     = object_id,
        .object_name   = std::move(object_name),
        .property_name = std::string(property_name),
        .node          = object_node_it->second.get(),
        .target_kind =
            camera_registration ? WPSceneScriptTargetKind::Camera : WPSceneScriptTargetKind::Layer,
        .target_index = 0,
        .value_type   = hint,
        .base_value   = ParsePropertyBaseValue(property_json, hint).value_or(setting.value),
        .animation =
            std::make_shared<WPPropertyAnimationDefinition>(std::move(animation_definition)),
        .setting = std::move(setting),
    });
    if (IsTextLayerObjectJson(object_json)) {
        const auto& registration = context.scene->propertyAnimationRegistrations.back();
        LogTextLayerRegistration("register-property-animation",
                                 registration.object_id,
                                 registration.object_name,
                                 registration.property_name,
                                 registration.value_type,
                                 registration.setting,
                                 registration.base_value);
    }
    if (camera_registration) {
        LOG_INFO("SceneCameraLayerRegister: layer=%d property='%.*s' kind=animation target=camera",
                 object_id,
                 static_cast<int>(property_name.size()),
                 property_name.data());
    }
}

void RegisterSceneScriptBinding(ParseContext& context, const nlohmann::json& object_json,
                                std::string_view property_name, WPDynamicValue::Type hint) {
    if (! object_json.is_object() || ! object_json.contains("id") ||
        ! object_json.contains("name") || ! object_json.contains(property_name)) {
        return;
    }

    const auto& property_json = object_json.at(property_name);
    if (! property_json.is_object() || ! property_json.contains("script")) return;

    int32_t object_id { 0 };
    if (! GetJsonValue(__SHORT_FILE__,
                       __FUNCTION__,
                       __LINE__,
                       object_json.at("id"),
                       object_id,
                       false,
                       "",
                       false)) {
        return;
    }

    WPUserSetting setting;
    if (! ParseUserSetting(property_json, setting, hint) || ! setting.hasScript()) return;

    std::string object_name;
    if (! GetJsonValue(__SHORT_FILE__,
                       __FUNCTION__,
                       __LINE__,
                       object_json.at("name"),
                       object_name,
                       false,
                       "",
                       false)) {
        object_name = std::to_string(object_id);
    }

    // Script bindings still dispatch through a SceneNode-backed target. Camera layers use the
    // camera target so parser/runtime scripts see normal layer objects while writes are routed to
    // the active SceneCamera state.
    const bool camera_registration =
        IsCameraLayerObjectJson(object_json) && IsCameraLayerRuntimeProperty(property_name);
    const auto object_node_it = context.object_nodes.find(object_id);
    if (object_node_it == context.object_nodes.end()) return;

    context.scene->scriptRegistrations.push_back(WPSceneScriptRegistration {
        .object_id     = object_id,
        .object_name   = std::move(object_name),
        .property_name = std::string(property_name),
        .node          = object_node_it->second.get(),
        .target_kind =
            camera_registration ? WPSceneScriptTargetKind::Camera : WPSceneScriptTargetKind::Layer,
        .target_index = 0,
        .value_type   = hint,
        .base_value   = setting.value,
        .setting      = std::move(setting),
    });
    if (IsTextLayerObjectJson(object_json)) {
        const auto& registration = context.scene->scriptRegistrations.back();
        LogTextLayerRegistration("register-script-binding",
                                 registration.object_id,
                                 registration.object_name,
                                 registration.property_name,
                                 registration.value_type,
                                 registration.setting,
                                 registration.base_value);
    }
    if (camera_registration) {
        LOG_INFO("SceneCameraLayerRegister: layer=%d property='%.*s' kind=script target=camera",
                 object_id,
                 static_cast<int>(property_name.size()),
                 property_name.data());
    }
}

void RegisterScenePropertyBinding(ParseContext& context, const nlohmann::json& object_json,
                                  std::string_view property_name, WPDynamicValue::Type hint) {
    if (! object_json.is_object() || ! object_json.contains("id") ||
        ! object_json.contains("name") || ! object_json.contains(property_name)) {
        return;
    }

    const auto& property_json = object_json.at(property_name);
    if (! property_json.is_object()) return;
    if (property_json.contains("script") || property_json.contains("animation")) return;

    int32_t object_id { 0 };
    if (! GetJsonValue(__SHORT_FILE__,
                       __FUNCTION__,
                       __LINE__,
                       object_json.at("id"),
                       object_id,
                       false,
                       "",
                       false)) {
        return;
    }

    // Sound objects are mounted as SoundManager streams and intentionally have no SceneNode in
    // context.object_nodes. Treat volume as a first-class runtime binding anyway so live user
    // property edits follow the same value path as the parse-time WPSoundParser::MountStream call.
    const bool sound_volume_binding =
        property_name == "volume" && context.scene != nullptr &&
        context.scene->objectRuntimeSoundHandles.count(object_id) != 0;
    const bool camera_registration =
        IsCameraLayerObjectJson(object_json) && IsCameraLayerRuntimeProperty(property_name);
    const auto object_node_it = context.object_nodes.find(object_id);
    if (object_node_it == context.object_nodes.end() && ! sound_volume_binding) return;

    WPUserSetting setting;
    if (! ParseUserSetting(property_json, setting, hint) || ! setting.hasUserBinding()) return;

    std::string object_name;
    if (! GetJsonValue(__SHORT_FILE__,
                       __FUNCTION__,
                       __LINE__,
                       object_json.at("name"),
                       object_name,
                       false,
                       "",
                       false)) {
        object_name = std::to_string(object_id);
    }

    context.scene->bindingRegistrations.push_back(WPSceneScriptRegistration {
        .object_id     = object_id,
        .object_name   = std::move(object_name),
        .property_name = std::string(property_name),
        .node          = sound_volume_binding ? nullptr : object_node_it->second.get(),
        .target_kind   = sound_volume_binding
                             ? WPSceneScriptTargetKind::Sound
                             : (camera_registration ? WPSceneScriptTargetKind::Camera
                                                    : WPSceneScriptTargetKind::Layer),
        .target_index  = 0,
        .value_type    = hint,
        .base_value    = setting.value,
        .setting       = std::move(setting),
    });
    if (sound_volume_binding) {
        LOG_INFO("SceneSoundRegister: layer=%d property='%.*s' kind=user target=sound",
                 object_id,
                 static_cast<int>(property_name.size()),
                 property_name.data());
    }
    if (IsTextLayerObjectJson(object_json)) {
        const auto& registration = context.scene->bindingRegistrations.back();
        LogTextLayerRegistration("register-user-binding",
                                 registration.object_id,
                                 registration.object_name,
                                 registration.property_name,
                                 registration.value_type,
                                 registration.setting,
                                 registration.base_value);
    }
    if (camera_registration) {
        LOG_INFO("SceneCameraLayerRegister: layer=%d property='%.*s' kind=user target=camera",
                 object_id,
                 static_cast<int>(property_name.size()),
                 property_name.data());
    }
}

void RegisterSceneParticleOverridePropertyBinding(ParseContext&         context,
                                                  const nlohmann::json& object_json,
                                                  std::string_view      property_name,
                                                  WPDynamicValue::Type  hint) {
    if (! object_json.is_object() || ! object_json.contains("id") ||
        ! object_json.contains("name") || ! object_json.contains("particle") ||
        object_json.at("particle").is_null() || ! object_json.contains("instanceoverride") ||
        ! object_json.at("instanceoverride").is_object()) {
        return;
    }

    const auto& override_json = object_json.at("instanceoverride");
    if (! override_json.contains(property_name)) return;

    const auto& property_json = override_json.at(property_name);
    if (! property_json.is_object()) return;
    if (property_json.contains("script") || property_json.contains("animation")) return;

    int32_t object_id { 0 };
    if (! GetJsonValue(__SHORT_FILE__,
                       __FUNCTION__,
                       __LINE__,
                       object_json.at("id"),
                       object_id,
                       false,
                       "",
                       false)) {
        return;
    }

    const auto object_node_it = context.object_nodes.find(object_id);
    if (object_node_it == context.object_nodes.end()) return;

    WPUserSetting setting;
    if (! ParseUserSetting(property_json, setting, hint) || ! setting.hasUserBinding()) return;

    std::string object_name;
    if (! GetJsonValue(__SHORT_FILE__,
                       __FUNCTION__,
                       __LINE__,
                       object_json.at("name"),
                       object_name,
                       false,
                       "",
                       false)) {
        object_name = std::to_string(object_id);
    }

    // Particle authoring stores trail/sprite overrides under `instanceoverride`, not beside
    // ordinary layer material or image-size properties. Registering the nested value as a layer
    // target lets the runtime dispatch reuse the existing user-property pipeline while
    // ApplyLayerPropertyValue() routes `colorn` and scalar `size` to ParticleSubSystem.
    context.scene->bindingRegistrations.push_back(WPSceneScriptRegistration {
        .object_id     = object_id,
        .object_name   = std::move(object_name),
        .property_name = std::string(property_name),
        .node          = object_node_it->second.get(),
        .target_kind   = WPSceneScriptTargetKind::Layer,
        .target_index  = 0,
        .value_type    = hint,
        .base_value    = ParsePropertyBaseValue(property_json, hint).value_or(setting.value),
        .setting       = std::move(setting),
    });

}

void RegisterSceneParticleOverrideScriptBinding(ParseContext&         context,
                                                const nlohmann::json& object_json,
                                                std::string_view      property_name,
                                                WPDynamicValue::Type  hint) {
    if (! object_json.is_object() || ! object_json.contains("id") ||
        ! object_json.contains("name") || ! object_json.contains("particle") ||
        object_json.at("particle").is_null() || ! object_json.contains("instanceoverride") ||
        ! object_json.at("instanceoverride").is_object()) {
        return;
    }

    const auto& override_json = object_json.at("instanceoverride");
    if (! override_json.contains(property_name)) return;

    const auto& property_json = override_json.at(property_name);
    if (! property_json.is_object() || ! property_json.contains("script")) return;

    int32_t object_id { 0 };
    if (! GetJsonValue(__SHORT_FILE__,
                       __FUNCTION__,
                       __LINE__,
                       object_json.at("id"),
                       object_id,
                       false,
                       "",
                       false)) {
        return;
    }

    const auto object_node_it = context.object_nodes.find(object_id);
    if (object_node_it == context.object_nodes.end()) return;

    WPUserSetting setting;
    if (! ParseUserSetting(property_json, setting, hint) || ! setting.hasScript()) return;

    std::string object_name;
    if (! GetJsonValue(__SHORT_FILE__,
                       __FUNCTION__,
                       __LINE__,
                       object_json.at("name"),
                       object_name,
                       false,
                       "",
                       false)) {
        object_name = std::to_string(object_id);
    }

    // Particle override scripts are authored under instanceoverride, but the script runtime only
    // knows how to dispatch persistent scripts to concrete scene targets. Register the nested
    // value as a layer target with the bare override name so ApplyParticlePropertyValue() can
    // forward it to ParticleSubSystem instead of losing audio-reactive particle clocks after parse.
    context.scene->scriptRegistrations.push_back(WPSceneScriptRegistration {
        .object_id     = object_id,
        .object_name   = std::move(object_name),
        .property_name = std::string(property_name),
        .node          = object_node_it->second.get(),
        .target_kind   = WPSceneScriptTargetKind::Layer,
        .target_index  = 0,
        .value_type    = hint,
        .base_value    = setting.value,
        .setting       = std::move(setting),
    });

}

void RegisterEffectVisibilityBinding(ParseContext& context, const nlohmann::json& object_json,
                                     const nlohmann::json& effect_json,
                                     uint32_t              authored_effect_index) {
    if (! object_json.is_object() || ! effect_json.is_object() ||
        ! effect_json.contains("visible") || ! effect_json.at("visible").is_object()) {
        return;
    }

    int32_t object_id { 0 };
    GET_JSON_NAME_VALUE_NOWARN(object_json, "id", object_id);
    if (object_id == 0 || context.object_nodes.count(object_id) == 0 || context.scene == nullptr) {
        return;
    }

    int32_t     effect_id { 0 };
    std::string authored_effect_name;
    GET_JSON_NAME_VALUE_NOWARN(effect_json, "id", effect_id);
    GET_JSON_NAME_VALUE_NOWARN(effect_json, "name", authored_effect_name);

    SceneImageEffect* effect =
        effect_id != 0 ? context.scene->FindImageEffectById(object_id, effect_id)
                       : context.scene->FindImageEffect(object_id, authored_effect_index);
    if (effect == nullptr) return;

    const auto&       visible_json = effect_json.at("visible");
    const std::string effect_name =
        ! authored_effect_name.empty() ? authored_effect_name : effect->EffectName();
    WPUserSetting setting;
    if (! ParseUserSetting(visible_json, setting, WPDynamicValue::Type::Boolean)) return;

    WPSceneScriptRegistration registration {
        .object_id     = object_id,
        .object_name   = effect_name,
        .property_name = "visible",
        .node          = context.object_nodes.at(object_id).get(),
        .target_kind   = WPSceneScriptTargetKind::Effect,
        .target_index  = effect->EffectIndex(),
        .target_id     = effect_id,
        .value_type    = WPDynamicValue::Type::Boolean,
        .base_value    = ParsePropertyBaseValue(visible_json, WPDynamicValue::Type::Boolean)
                             .value_or(setting.value),
        .setting       = std::move(setting),
    };

    const char* registration_kind = nullptr;
    if (visible_json.contains("animation") && ! visible_json.at("animation").is_null()) {
        WPPropertyAnimationDefinition animation_definition;
        if (! ParsePropertyAnimationDefinition(
                visible_json, WPDynamicValue::Type::Boolean, animation_definition)) {
            return;
        }
        registration.animation =
            std::make_shared<WPPropertyAnimationDefinition>(std::move(animation_definition));
        context.scene->propertyAnimationRegistrations.push_back(std::move(registration));
        registration_kind = "animation";
    } else if (registration.setting.hasScript()) {
        context.scene->scriptRegistrations.push_back(std::move(registration));
        registration_kind = "script";
    } else if (registration.setting.hasUserBinding()) {
        context.scene->bindingRegistrations.push_back(std::move(registration));
        registration_kind = "user";
    }

    if (registration_kind != nullptr) {
        LOG_INFO("SceneVisibilityEffectRegister: layer=%d effect-id=%d effect-index=%u name='%s' "
                 "kind=%s initial-visible=%s",
                 object_id,
                 effect_id,
                 effect->EffectIndex(),
                 effect_name.c_str(),
                 registration_kind,
                 effect->LocalVisible() ? "true" : "false");
    }
}

void RegisterEffectVisibilityBindings(ParseContext& context, const nlohmann::json& object_json) {
    if (! object_json.is_object() || ! object_json.contains("effects") ||
        ! object_json.at("effects").is_array()) {
        return;
    }

    uint32_t effect_index = 0;
    for (const auto& effect_json : object_json.at("effects")) {
        // Effect material scripts are registered later while each concrete pass material is being
        // built, because only that stage knows the resolved GLSL uniform name and pass SceneNode.
        // Visibility bindings still stay here where the authored effect index is available.
        RegisterEffectVisibilityBinding(context, object_json, effect_json, effect_index);
        effect_index++;
    }
}

void RegisterAnimationLayerSceneScriptBinding(ParseContext&         context,
                                              const nlohmann::json& object_json,
                                              const nlohmann::json& layer_json,
                                              uint32_t layer_index, std::string_view property_name,
                                              WPDynamicValue::Type hint) {
    if (! object_json.is_object() || ! object_json.contains("id") ||
        ! object_json.contains("name") || ! layer_json.is_object() ||
        ! layer_json.contains(property_name)) {
        return;
    }

    const auto& property_json = layer_json.at(property_name);
    if (! property_json.is_object() || ! property_json.contains("script")) return;

    int32_t object_id { 0 };
    if (! GetJsonValue(__SHORT_FILE__,
                       __FUNCTION__,
                       __LINE__,
                       object_json.at("id"),
                       object_id,
                       false,
                       "",
                       false)) {
        return;
    }

    if (context.object_nodes.count(object_id) == 0) return;

    WPUserSetting setting;
    if (! ParseUserSetting(property_json, setting, hint) || ! setting.hasScript()) return;

    std::string object_name;
    if (! GetJsonValue(__SHORT_FILE__,
                       __FUNCTION__,
                       __LINE__,
                       object_json.at("name"),
                       object_name,
                       false,
                       "",
                       false)) {
        object_name = std::to_string(object_id);
    }

    context.scene->scriptRegistrations.push_back(WPSceneScriptRegistration {
        .object_id     = object_id,
        .object_name   = std::move(object_name),
        .property_name = std::string(property_name),
        .node          = context.object_nodes.at(object_id).get(),
        .target_kind   = WPSceneScriptTargetKind::AnimationLayer,
        .target_index  = layer_index,
        .value_type    = hint,
        .base_value    = setting.value,
        .setting       = std::move(setting),
    });
}

void RegisterAnimationLayerPropertyBinding(ParseContext& context, const nlohmann::json& object_json,
                                           const nlohmann::json& layer_json, uint32_t layer_index,
                                           std::string_view     property_name,
                                           WPDynamicValue::Type hint) {
    if (! object_json.is_object() || ! object_json.contains("id") ||
        ! object_json.contains("name") || ! layer_json.is_object() ||
        ! layer_json.contains(property_name)) {
        return;
    }

    const auto& property_json = layer_json.at(property_name);
    if (! property_json.is_object()) return;
    // A script on an animation-layer property can initialize timing or shared state while the same
    // property still declares a user binding for live visibility/blend control. Only authored
    // property animations own the value path exclusively; scripts and direct user bindings must
    // coexist so runtime project-property edits are applied after script initialization.
    if (property_json.contains("animation")) return;

    int32_t object_id { 0 };
    if (! GetJsonValue(__SHORT_FILE__,
                       __FUNCTION__,
                       __LINE__,
                       object_json.at("id"),
                       object_id,
                       false,
                       "",
                       false)) {
        return;
    }

    if (context.object_nodes.count(object_id) == 0) return;

    WPUserSetting setting;
    if (! ParseUserSetting(property_json, setting, hint) || ! setting.hasUserBinding()) return;

    std::string object_name;
    if (! GetJsonValue(__SHORT_FILE__,
                       __FUNCTION__,
                       __LINE__,
                       object_json.at("name"),
                       object_name,
                       false,
                       "",
                       false)) {
        object_name = std::to_string(object_id);
    }

    context.scene->bindingRegistrations.push_back(WPSceneScriptRegistration {
        .object_id     = object_id,
        .object_name   = std::move(object_name),
        .property_name = std::string(property_name),
        .node          = context.object_nodes.at(object_id).get(),
        .target_kind   = WPSceneScriptTargetKind::AnimationLayer,
        .target_index  = layer_index,
        .value_type    = hint,
        .base_value    = setting.value,
        .setting       = std::move(setting),
    });

    LOG_INFO("SceneAnimationLayerRegister: layer=%d animation-index=%u property='%.*s' kind=user "
             "target=animationLayer script-present=%s",
             object_id,
             layer_index,
             static_cast<int>(property_name.size()),
             property_name.data(),
             property_json.contains("script") ? "true" : "false");
}

void RegisterSceneGeneralPropertyBinding(ParseContext& context, const nlohmann::json& general_json,
                                         std::string_view     property_name,
                                         WPDynamicValue::Type hint) {
    if (context.scene == nullptr || ! general_json.is_object() ||
        ! general_json.contains(property_name)) {
        return;
    }

    const auto& property_json = general_json.at(property_name);
    if (! property_json.is_object()) return;
    // General properties do not have SceneNode owners. Keep authored animations out of the direct
    // binding table, but allow user-bound values to drive global runtime state such as camera
    // parallax through the same dispatch path as layer and effect properties.
    if (property_json.contains("animation")) return;

    WPUserSetting setting;
    if (! ParseUserSetting(property_json, setting, hint) || ! setting.hasUserBinding()) return;

    context.scene->bindingRegistrations.push_back(WPSceneScriptRegistration {
        .object_id     = 0,
        .object_name   = "scene.general",
        .property_name = std::string(property_name),
        .node          = nullptr,
        .target_kind   = WPSceneScriptTargetKind::Scene,
        .target_index  = 0,
        .value_type    = hint,
        .base_value    = ParsePropertyBaseValue(property_json, hint).value_or(setting.value),
        .setting       = std::move(setting),
    });

    LOG_INFO("SceneGeneralRegister: property='%.*s' kind=user target=scene.general",
             static_cast<int>(property_name.size()),
             property_name.data());
}

void RegisterSceneScripts(ParseContext& context, const nlohmann::json& json) {
    if (json.contains("general") && json.at("general").is_object()) {
        const auto& general_json = json.at("general");
        RegisterSceneGeneralPropertyBinding(
            context, general_json, "clearcolor", WPDynamicValue::Type::Float3);
        RegisterSceneGeneralPropertyBinding(
            context, general_json, "ambientcolor", WPDynamicValue::Type::Float3);
        RegisterSceneGeneralPropertyBinding(
            context, general_json, "skylightcolor", WPDynamicValue::Type::Float3);
        RegisterSceneGeneralPropertyBinding(
            context, general_json, "bloom", WPDynamicValue::Type::Boolean);
        RegisterSceneGeneralPropertyBinding(
            context, general_json, "bloomstrength", WPDynamicValue::Type::Float);
        RegisterSceneGeneralPropertyBinding(
            context, general_json, "bloomthreshold", WPDynamicValue::Type::Float);
        RegisterSceneGeneralPropertyBinding(
            context, general_json, "bloomtint", WPDynamicValue::Type::Float3);
        RegisterSceneGeneralPropertyBinding(
            context, general_json, "cameraparallax", WPDynamicValue::Type::Boolean);
        RegisterSceneGeneralPropertyBinding(
            context, general_json, "cameraparallaxamount", WPDynamicValue::Type::Float);
        RegisterSceneGeneralPropertyBinding(
            context, general_json, "cameraparallaxdelay", WPDynamicValue::Type::Float);
        RegisterSceneGeneralPropertyBinding(
            context, general_json, "cameraparallaxmouseinfluence", WPDynamicValue::Type::Float);
        RegisterSceneGeneralPropertyBinding(
            context, general_json, "camerashake", WPDynamicValue::Type::Boolean);
        RegisterSceneGeneralPropertyBinding(
            context, general_json, "camerashakeamplitude", WPDynamicValue::Type::Float);
        RegisterSceneGeneralPropertyBinding(
            context, general_json, "camerashakeroughness", WPDynamicValue::Type::Float);
        RegisterSceneGeneralPropertyBinding(
            context, general_json, "camerashakespeed", WPDynamicValue::Type::Float);
        RegisterSceneGeneralPropertyBinding(
            context, general_json, "fov", WPDynamicValue::Type::Float);
        RegisterSceneGeneralPropertyBinding(
            context, general_json, "nearz", WPDynamicValue::Type::Float);
        RegisterSceneGeneralPropertyBinding(
            context, general_json, "farz", WPDynamicValue::Type::Float);
    }

    if (! json.contains("objects") || ! json.at("objects").is_array()) return;

    for (const auto& object_json : json.at("objects")) {
        RegisterScenePropertyBinding(
            context, object_json, "visible", WPDynamicValue::Type::Boolean);
        RegisterScenePropertyBinding(context, object_json, "origin", WPDynamicValue::Type::Float3);
        RegisterScenePropertyBinding(context, object_json, "angles", WPDynamicValue::Type::Float3);
        RegisterScenePropertyBinding(context, object_json, "scale", WPDynamicValue::Type::Float3);
        RegisterScenePropertyBinding(
            context, object_json, "parallaxDepth", WPDynamicValue::Type::Float2);
        if ((object_json.contains("image") && ! object_json.at("image").is_null()) ||
            (object_json.contains("text") && ! object_json.at("text").is_null())) {
            RegisterScenePropertyBinding(
                context, object_json, "size", WPDynamicValue::Type::Float2);
        }
        RegisterScenePropertyBinding(context, object_json, "text", WPDynamicValue::Type::String);
        RegisterScenePropertyBinding(context, object_json, "font", WPDynamicValue::Type::String);
        RegisterScenePropertyBinding(context, object_json, "color", WPDynamicValue::Type::Float3);
        // Particle controls such as mouse-tail `colorn`, scalar `size`, and audio-reactive
        // `rate` are nested below instanceoverride, so the generic top-level scans above never see
        // them. Register the nested override names here so user-property edits and persistent
        // scripts can reach live particles instead of being frozen at parse-time values.
        RegisterSceneParticleOverridePropertyBinding(
            context, object_json, "colorn", WPDynamicValue::Type::Float3);
        RegisterSceneParticleOverridePropertyBinding(
            context, object_json, "color", WPDynamicValue::Type::Float3);
        RegisterSceneParticleOverridePropertyBinding(
            context, object_json, "size", WPDynamicValue::Type::Float);
        RegisterSceneParticleOverridePropertyBinding(
            context, object_json, "rate", WPDynamicValue::Type::Float);
        RegisterSceneParticleOverrideScriptBinding(
            context, object_json, "rate", WPDynamicValue::Type::Float);
        RegisterScenePropertyBinding(context, object_json, "alpha", WPDynamicValue::Type::Float);
        RegisterScenePropertyBinding(
            context, object_json, "brightness", WPDynamicValue::Type::Float);
        // Sound-layer volume is authored beside visual layer properties, but its runtime target is
        // the mounted audio stream instead of a SceneNode material.
        RegisterScenePropertyBinding(context, object_json, "volume", WPDynamicValue::Type::Float);
        RegisterScenePropertyBinding(
            context, object_json, "backgroundcolor", WPDynamicValue::Type::Float3);
        RegisterScenePropertyBinding(
            context, object_json, "backgroundbrightness", WPDynamicValue::Type::Float);
        RegisterScenePropertyBinding(
            context, object_json, "opaquebackground", WPDynamicValue::Type::Boolean);
        RegisterScenePropertyBinding(
            context, object_json, "pointsize", WPDynamicValue::Type::Float);
        RegisterScenePropertyBinding(context, object_json, "padding", WPDynamicValue::Type::Int32);
        RegisterScenePropertyBinding(
            context, object_json, "horizontalalign", WPDynamicValue::Type::String);
        RegisterScenePropertyBinding(
            context, object_json, "verticalalign", WPDynamicValue::Type::String);
        RegisterScenePropertyBinding(context, object_json, "anchor", WPDynamicValue::Type::String);
        RegisterScenePropertyBinding(
            context, object_json, "limitrows", WPDynamicValue::Type::Boolean);
        RegisterScenePropertyBinding(context, object_json, "maxrows", WPDynamicValue::Type::Int32);
        RegisterScenePropertyBinding(
            context, object_json, "limitwidth", WPDynamicValue::Type::Boolean);
        RegisterScenePropertyBinding(context, object_json, "maxwidth", WPDynamicValue::Type::Float);
        RegisterScenePropertyAnimationBinding(
            context, object_json, "visible", WPDynamicValue::Type::Boolean);
        RegisterScenePropertyAnimationBinding(
            context, object_json, "origin", WPDynamicValue::Type::Float3);
        RegisterScenePropertyAnimationBinding(
            context, object_json, "angles", WPDynamicValue::Type::Float3);
        RegisterScenePropertyAnimationBinding(
            context, object_json, "scale", WPDynamicValue::Type::Float3);
        RegisterScenePropertyAnimationBinding(
            context, object_json, "parallaxDepth", WPDynamicValue::Type::Float2);
        if ((object_json.contains("image") && ! object_json.at("image").is_null()) ||
            (object_json.contains("text") && ! object_json.at("text").is_null())) {
            RegisterScenePropertyAnimationBinding(
                context, object_json, "size", WPDynamicValue::Type::Float2);
        }
        RegisterScenePropertyAnimationBinding(
            context, object_json, "color", WPDynamicValue::Type::Float3);
        RegisterScenePropertyAnimationBinding(
            context, object_json, "alpha", WPDynamicValue::Type::Float);
        RegisterScenePropertyAnimationBinding(
            context, object_json, "brightness", WPDynamicValue::Type::Float);
        RegisterScenePropertyAnimationBinding(
            context, object_json, "backgroundcolor", WPDynamicValue::Type::Float3);
        RegisterScenePropertyAnimationBinding(
            context, object_json, "backgroundbrightness", WPDynamicValue::Type::Float);
        RegisterScenePropertyAnimationBinding(
            context, object_json, "opaquebackground", WPDynamicValue::Type::Boolean);
        RegisterScenePropertyAnimationBinding(
            context, object_json, "pointsize", WPDynamicValue::Type::Float);
        RegisterScenePropertyAnimationBinding(
            context, object_json, "padding", WPDynamicValue::Type::Int32);
        RegisterScenePropertyAnimationBinding(
            context, object_json, "limitrows", WPDynamicValue::Type::Boolean);
        RegisterScenePropertyAnimationBinding(
            context, object_json, "maxrows", WPDynamicValue::Type::Int32);
        RegisterScenePropertyAnimationBinding(
            context, object_json, "limitwidth", WPDynamicValue::Type::Boolean);
        RegisterScenePropertyAnimationBinding(
            context, object_json, "maxwidth", WPDynamicValue::Type::Float);

        RegisterSceneScriptBinding(context, object_json, "visible", WPDynamicValue::Type::Boolean);
        RegisterSceneScriptBinding(context, object_json, "origin", WPDynamicValue::Type::Float3);
        RegisterSceneScriptBinding(context, object_json, "angles", WPDynamicValue::Type::Float3);
        RegisterSceneScriptBinding(context, object_json, "scale", WPDynamicValue::Type::Float3);
        RegisterSceneScriptBinding(
            context, object_json, "parallaxDepth", WPDynamicValue::Type::Float2);
        if ((object_json.contains("image") && ! object_json.at("image").is_null()) ||
            (object_json.contains("text") && ! object_json.at("text").is_null())) {
            RegisterSceneScriptBinding(context, object_json, "size", WPDynamicValue::Type::Float2);
        }
        RegisterSceneScriptBinding(context, object_json, "text", WPDynamicValue::Type::String);
        RegisterSceneScriptBinding(context, object_json, "font", WPDynamicValue::Type::String);
        RegisterSceneScriptBinding(context, object_json, "color", WPDynamicValue::Type::Float3);
        RegisterSceneScriptBinding(context, object_json, "alpha", WPDynamicValue::Type::Float);
        RegisterSceneScriptBinding(context, object_json, "brightness", WPDynamicValue::Type::Float);
        RegisterSceneScriptBinding(
            context, object_json, "backgroundcolor", WPDynamicValue::Type::Float3);
        RegisterSceneScriptBinding(
            context, object_json, "backgroundbrightness", WPDynamicValue::Type::Float);
        RegisterSceneScriptBinding(
            context, object_json, "opaquebackground", WPDynamicValue::Type::Boolean);
        RegisterSceneScriptBinding(context, object_json, "pointsize", WPDynamicValue::Type::Float);
        RegisterSceneScriptBinding(context, object_json, "padding", WPDynamicValue::Type::Int32);
        RegisterSceneScriptBinding(
            context, object_json, "horizontalalign", WPDynamicValue::Type::String);
        RegisterSceneScriptBinding(
            context, object_json, "verticalalign", WPDynamicValue::Type::String);
        RegisterSceneScriptBinding(context, object_json, "anchor", WPDynamicValue::Type::String);
        RegisterSceneScriptBinding(
            context, object_json, "limitrows", WPDynamicValue::Type::Boolean);
        RegisterSceneScriptBinding(context, object_json, "maxrows", WPDynamicValue::Type::Int32);
        RegisterSceneScriptBinding(
            context, object_json, "limitwidth", WPDynamicValue::Type::Boolean);
        RegisterSceneScriptBinding(context, object_json, "maxwidth", WPDynamicValue::Type::Float);

        if (object_json.contains("light") && ! object_json.at("light").is_null()) {
            // Official lighting docs animate Intensity (and optionally Radius) on the light
            // object itself. These are not material uniforms, so they are scanned only for
            // authored lights and applied to SceneLight each frame.
            RegisterScenePropertyBinding(
                context, object_json, "intensity", WPDynamicValue::Type::Float);
            RegisterScenePropertyBinding(
                context, object_json, "radius", WPDynamicValue::Type::Float);
            RegisterScenePropertyAnimationBinding(
                context, object_json, "intensity", WPDynamicValue::Type::Float);
            RegisterScenePropertyAnimationBinding(
                context, object_json, "radius", WPDynamicValue::Type::Float);
            RegisterSceneScriptBinding(
                context, object_json, "intensity", WPDynamicValue::Type::Float);
            RegisterSceneScriptBinding(context, object_json, "radius", WPDynamicValue::Type::Float);
        }

        if (IsCameraLayerObjectJson(object_json)) {
            // Camera zoom/fov are not normal drawable layer properties, so they are scanned only
            // for authored camera layers and routed through the camera target kind registered
            // above. Origin/visible are already part of the shared layer scan and are retargeted
            // by RegisterScene*Binding when the object is a camera layer.
            RegisterScenePropertyBinding(context, object_json, "zoom", WPDynamicValue::Type::Float);
            RegisterScenePropertyBinding(context, object_json, "fov", WPDynamicValue::Type::Float);
            RegisterScenePropertyAnimationBinding(
                context, object_json, "zoom", WPDynamicValue::Type::Float);
            RegisterScenePropertyAnimationBinding(
                context, object_json, "fov", WPDynamicValue::Type::Float);
            RegisterSceneScriptBinding(context, object_json, "zoom", WPDynamicValue::Type::Float);
            RegisterSceneScriptBinding(context, object_json, "fov", WPDynamicValue::Type::Float);
        }

        RegisterEffectVisibilityBindings(context, object_json);

        if (! object_json.contains("animationlayers") ||
            ! object_json.at("animationlayers").is_array()) {
            continue;
        }

        uint32_t layer_index = 0;
        for (const auto& animation_layer_json : object_json.at("animationlayers")) {
            RegisterAnimationLayerPropertyBinding(context,
                                                  object_json,
                                                  animation_layer_json,
                                                  layer_index,
                                                  "visible",
                                                  WPDynamicValue::Type::Boolean);
            RegisterAnimationLayerPropertyBinding(context,
                                                  object_json,
                                                  animation_layer_json,
                                                  layer_index,
                                                  "rate",
                                                  WPDynamicValue::Type::Float);
            RegisterAnimationLayerPropertyBinding(context,
                                                  object_json,
                                                  animation_layer_json,
                                                  layer_index,
                                                  "blend",
                                                  WPDynamicValue::Type::Float);
            RegisterAnimationLayerSceneScriptBinding(context,
                                                     object_json,
                                                     animation_layer_json,
                                                     layer_index,
                                                     "visible",
                                                     WPDynamicValue::Type::Boolean);
            RegisterAnimationLayerSceneScriptBinding(context,
                                                     object_json,
                                                     animation_layer_json,
                                                     layer_index,
                                                     "rate",
                                                     WPDynamicValue::Type::Float);
            RegisterAnimationLayerSceneScriptBinding(context,
                                                     object_json,
                                                     animation_layer_json,
                                                     layer_index,
                                                     "blend",
                                                     WPDynamicValue::Type::Float);
            layer_index++;
        }
    }
}

void RegisterSceneScriptsForObject(ParseContext& context, const nlohmann::json& object_json) {
    RegisterScenePropertyBinding(context, object_json, "visible", WPDynamicValue::Type::Boolean);
    RegisterScenePropertyBinding(context, object_json, "origin", WPDynamicValue::Type::Float3);
    RegisterScenePropertyBinding(context, object_json, "angles", WPDynamicValue::Type::Float3);
    RegisterScenePropertyBinding(context, object_json, "scale", WPDynamicValue::Type::Float3);
    RegisterScenePropertyBinding(
        context, object_json, "parallaxDepth", WPDynamicValue::Type::Float2);
    if ((object_json.contains("image") && ! object_json.at("image").is_null()) ||
        (object_json.contains("text") && ! object_json.at("text").is_null())) {
        RegisterScenePropertyBinding(context, object_json, "size", WPDynamicValue::Type::Float2);
    }
    RegisterScenePropertyBinding(context, object_json, "text", WPDynamicValue::Type::String);
    RegisterScenePropertyBinding(context, object_json, "font", WPDynamicValue::Type::String);
    RegisterScenePropertyBinding(context, object_json, "color", WPDynamicValue::Type::Float3);
    // Dynamic layer materialization uses this per-object registration path, so particle
    // instanceoverride color, size, and rate bindings must be added here as well as during the
    // initial full-scene scan above.
    RegisterSceneParticleOverridePropertyBinding(
        context, object_json, "colorn", WPDynamicValue::Type::Float3);
    RegisterSceneParticleOverridePropertyBinding(
        context, object_json, "color", WPDynamicValue::Type::Float3);
    RegisterSceneParticleOverridePropertyBinding(
        context, object_json, "size", WPDynamicValue::Type::Float);
    RegisterSceneParticleOverridePropertyBinding(
        context, object_json, "rate", WPDynamicValue::Type::Float);
    RegisterSceneParticleOverrideScriptBinding(
        context, object_json, "rate", WPDynamicValue::Type::Float);
    RegisterScenePropertyBinding(context, object_json, "alpha", WPDynamicValue::Type::Float);
    RegisterScenePropertyBinding(context, object_json, "brightness", WPDynamicValue::Type::Float);
    // Dynamic materialization reuses the same registration helper, so keep sound volume in this
    // per-object path as well as the initial full-scene scan.
    RegisterScenePropertyBinding(context, object_json, "volume", WPDynamicValue::Type::Float);
    RegisterScenePropertyBinding(
        context, object_json, "backgroundcolor", WPDynamicValue::Type::Float3);
    RegisterScenePropertyBinding(
        context, object_json, "backgroundbrightness", WPDynamicValue::Type::Float);
    RegisterScenePropertyBinding(
        context, object_json, "opaquebackground", WPDynamicValue::Type::Boolean);
    RegisterScenePropertyBinding(context, object_json, "pointsize", WPDynamicValue::Type::Float);
    RegisterScenePropertyBinding(context, object_json, "padding", WPDynamicValue::Type::Int32);
    RegisterScenePropertyBinding(
        context, object_json, "horizontalalign", WPDynamicValue::Type::String);
    RegisterScenePropertyBinding(
        context, object_json, "verticalalign", WPDynamicValue::Type::String);
    RegisterScenePropertyBinding(context, object_json, "anchor", WPDynamicValue::Type::String);
    RegisterScenePropertyBinding(context, object_json, "limitrows", WPDynamicValue::Type::Boolean);
    RegisterScenePropertyBinding(context, object_json, "maxrows", WPDynamicValue::Type::Int32);
    RegisterScenePropertyBinding(context, object_json, "limitwidth", WPDynamicValue::Type::Boolean);
    RegisterScenePropertyBinding(context, object_json, "maxwidth", WPDynamicValue::Type::Float);
    RegisterScenePropertyAnimationBinding(
        context, object_json, "visible", WPDynamicValue::Type::Boolean);
    RegisterScenePropertyAnimationBinding(
        context, object_json, "origin", WPDynamicValue::Type::Float3);
    RegisterScenePropertyAnimationBinding(
        context, object_json, "angles", WPDynamicValue::Type::Float3);
    RegisterScenePropertyAnimationBinding(
        context, object_json, "scale", WPDynamicValue::Type::Float3);
    RegisterScenePropertyAnimationBinding(
        context, object_json, "parallaxDepth", WPDynamicValue::Type::Float2);
    if ((object_json.contains("image") && ! object_json.at("image").is_null()) ||
        (object_json.contains("text") && ! object_json.at("text").is_null())) {
        RegisterScenePropertyAnimationBinding(
            context, object_json, "size", WPDynamicValue::Type::Float2);
    }
    RegisterScenePropertyAnimationBinding(
        context, object_json, "color", WPDynamicValue::Type::Float3);
    RegisterScenePropertyAnimationBinding(
        context, object_json, "alpha", WPDynamicValue::Type::Float);
    RegisterScenePropertyAnimationBinding(
        context, object_json, "brightness", WPDynamicValue::Type::Float);
    RegisterScenePropertyAnimationBinding(
        context, object_json, "backgroundcolor", WPDynamicValue::Type::Float3);
    RegisterScenePropertyAnimationBinding(
        context, object_json, "backgroundbrightness", WPDynamicValue::Type::Float);
    RegisterScenePropertyAnimationBinding(
        context, object_json, "opaquebackground", WPDynamicValue::Type::Boolean);
    RegisterScenePropertyAnimationBinding(
        context, object_json, "pointsize", WPDynamicValue::Type::Float);
    RegisterScenePropertyAnimationBinding(
        context, object_json, "padding", WPDynamicValue::Type::Int32);
    RegisterScenePropertyAnimationBinding(
        context, object_json, "limitrows", WPDynamicValue::Type::Boolean);
    RegisterScenePropertyAnimationBinding(
        context, object_json, "maxrows", WPDynamicValue::Type::Int32);
    RegisterScenePropertyAnimationBinding(
        context, object_json, "limitwidth", WPDynamicValue::Type::Boolean);
    RegisterScenePropertyAnimationBinding(
        context, object_json, "maxwidth", WPDynamicValue::Type::Float);

    RegisterSceneScriptBinding(context, object_json, "visible", WPDynamicValue::Type::Boolean);
    RegisterSceneScriptBinding(context, object_json, "origin", WPDynamicValue::Type::Float3);
    RegisterSceneScriptBinding(context, object_json, "angles", WPDynamicValue::Type::Float3);
    RegisterSceneScriptBinding(context, object_json, "scale", WPDynamicValue::Type::Float3);
    RegisterSceneScriptBinding(context, object_json, "parallaxDepth", WPDynamicValue::Type::Float2);
    if ((object_json.contains("image") && ! object_json.at("image").is_null()) ||
        (object_json.contains("text") && ! object_json.at("text").is_null())) {
        RegisterSceneScriptBinding(context, object_json, "size", WPDynamicValue::Type::Float2);
    }
    RegisterSceneScriptBinding(context, object_json, "text", WPDynamicValue::Type::String);
    RegisterSceneScriptBinding(context, object_json, "font", WPDynamicValue::Type::String);
    RegisterSceneScriptBinding(context, object_json, "color", WPDynamicValue::Type::Float3);
    RegisterSceneScriptBinding(context, object_json, "alpha", WPDynamicValue::Type::Float);
    RegisterSceneScriptBinding(context, object_json, "brightness", WPDynamicValue::Type::Float);
    RegisterSceneScriptBinding(
        context, object_json, "backgroundcolor", WPDynamicValue::Type::Float3);
    RegisterSceneScriptBinding(
        context, object_json, "backgroundbrightness", WPDynamicValue::Type::Float);
    RegisterSceneScriptBinding(
        context, object_json, "opaquebackground", WPDynamicValue::Type::Boolean);
    RegisterSceneScriptBinding(context, object_json, "pointsize", WPDynamicValue::Type::Float);
    RegisterSceneScriptBinding(context, object_json, "padding", WPDynamicValue::Type::Int32);
    RegisterSceneScriptBinding(
        context, object_json, "horizontalalign", WPDynamicValue::Type::String);
    RegisterSceneScriptBinding(context, object_json, "verticalalign", WPDynamicValue::Type::String);
    RegisterSceneScriptBinding(context, object_json, "anchor", WPDynamicValue::Type::String);
    RegisterSceneScriptBinding(context, object_json, "limitrows", WPDynamicValue::Type::Boolean);
    RegisterSceneScriptBinding(context, object_json, "maxrows", WPDynamicValue::Type::Int32);
    RegisterSceneScriptBinding(context, object_json, "limitwidth", WPDynamicValue::Type::Boolean);
    RegisterSceneScriptBinding(context, object_json, "maxwidth", WPDynamicValue::Type::Float);

    if (object_json.contains("light") && ! object_json.at("light").is_null()) {
        RegisterScenePropertyBinding(
            context, object_json, "intensity", WPDynamicValue::Type::Float);
        RegisterScenePropertyBinding(context, object_json, "radius", WPDynamicValue::Type::Float);
        RegisterScenePropertyAnimationBinding(
            context, object_json, "intensity", WPDynamicValue::Type::Float);
        RegisterScenePropertyAnimationBinding(
            context, object_json, "radius", WPDynamicValue::Type::Float);
        RegisterSceneScriptBinding(context, object_json, "intensity", WPDynamicValue::Type::Float);
        RegisterSceneScriptBinding(context, object_json, "radius", WPDynamicValue::Type::Float);
    }

    if (IsCameraLayerObjectJson(object_json)) {
        // Dynamic camera layers need the same camera-only zoom/fov registration as scene-load
        // camera layers; otherwise scripts that create or re-materialize camera assets would keep
        // origin live but leave zoom frozen at the authored parse value.
        RegisterScenePropertyBinding(context, object_json, "zoom", WPDynamicValue::Type::Float);
        RegisterScenePropertyBinding(context, object_json, "fov", WPDynamicValue::Type::Float);
        RegisterScenePropertyAnimationBinding(
            context, object_json, "zoom", WPDynamicValue::Type::Float);
        RegisterScenePropertyAnimationBinding(
            context, object_json, "fov", WPDynamicValue::Type::Float);
        RegisterSceneScriptBinding(context, object_json, "zoom", WPDynamicValue::Type::Float);
        RegisterSceneScriptBinding(context, object_json, "fov", WPDynamicValue::Type::Float);
    }

    RegisterEffectVisibilityBindings(context, object_json);

    if (! object_json.contains("animationlayers") ||
        ! object_json.at("animationlayers").is_array()) {
        return;
    }

    uint32_t layer_index = 0;
    for (const auto& animation_layer_json : object_json.at("animationlayers")) {
        RegisterAnimationLayerPropertyBinding(context,
                                              object_json,
                                              animation_layer_json,
                                              layer_index,
                                              "visible",
                                              WPDynamicValue::Type::Boolean);
        RegisterAnimationLayerPropertyBinding(context,
                                              object_json,
                                              animation_layer_json,
                                              layer_index,
                                              "rate",
                                              WPDynamicValue::Type::Float);
        RegisterAnimationLayerPropertyBinding(context,
                                              object_json,
                                              animation_layer_json,
                                              layer_index,
                                              "blend",
                                              WPDynamicValue::Type::Float);
        RegisterAnimationLayerSceneScriptBinding(context,
                                                 object_json,
                                                 animation_layer_json,
                                                 layer_index,
                                                 "visible",
                                                 WPDynamicValue::Type::Boolean);
        RegisterAnimationLayerSceneScriptBinding(context,
                                                 object_json,
                                                 animation_layer_json,
                                                 layer_index,
                                                 "rate",
                                                 WPDynamicValue::Type::Float);
        RegisterAnimationLayerSceneScriptBinding(context,
                                                 object_json,
                                                 animation_layer_json,
                                                 layer_index,
                                                 "blend",
                                                 WPDynamicValue::Type::Float);
        layer_index++;
    }
}

bool InitDynamicParseContext(ParseContext& context, Scene& scene,
                             const UserPropertyMap* user_properties) {
    if (scene.shaderValueUpdater == nullptr || scene.vfs == nullptr) return false;

    auto* shader_updater = dynamic_cast<WPShaderValueUpdater*>(scene.shaderValueUpdater.get());
    if (shader_updater == nullptr) return false;

    context.scene           = std::shared_ptr<Scene>(&scene, [](Scene*) {
    });
    context.shader_updater  = shader_updater;
    context.vfs             = scene.vfs.get();
    context.user_properties = user_properties;
    context.ortho_w         = scene.ortho[0];
    context.ortho_h         = scene.ortho[1];
    context.passthrough_layer_ids = scene.passthroughLayerIds;
    PopulateGlobalBaseUniforms(context, scene);

    auto effect_it      = scene.cameras.find("effect");
    auto global_it      = scene.cameras.find("global");
    auto perspective_it = scene.cameras.find("global_perspective");
    if (effect_it != scene.cameras.end())
        context.effect_camera_node = effect_it->second->GetAttachedNode();
    if (global_it != scene.cameras.end())
        context.global_camera_node = global_it->second->GetAttachedNode();
    if (perspective_it != scene.cameras.end()) {
        context.global_perspective_camera_node = perspective_it->second->GetAttachedNode();
    }

    std::unordered_map<SceneNode*, std::shared_ptr<SceneNode>> shared_nodes;
    CollectSceneNodeRefs(scene.sceneGraph, shared_nodes);
    for (const auto& [layer_id, node] : scene.layerNodes) {
        auto node_it = shared_nodes.find(node);
        if (node_it != shared_nodes.end()) {
            context.object_nodes[layer_id] = node_it->second;
        }

        if (const auto* node_data = shader_updater->GetNodeData(node);
            node_data != nullptr && node_data->puppet_layer.Puppet() != nullptr) {
            context.object_puppets[layer_id] = node_data->puppet_layer.Puppet();
        }
    }
    return true;
}

bool ParseDynamicSceneObject(ParseContext& context, const nlohmann::json& object_json,
                             const UserPropertyMap* user_properties, int32_t* out_layer_id) {
    const auto register_visibility_contract = [&](int32_t layer_id, LazyMaterializeKind lazy_kind) {
        if (layer_id == 0) return;
        // Dynamically-created layers go through the same runtime visibility policy as scene-load
        // layers. Without this, hidden text/particle objects created from scripts would fall back
        // to scattered type checks instead of the unified contract used by the main parser.
        context.layer_visibility_contracts[layer_id] = BuildObjectVisibilityContractFromJson(
            object_json, user_properties, false, false, lazy_kind);
    };
    const auto resolve_visibility = [&](auto& object) {
        object.visible =
            ResolveObjectVisibility(object.visible, object.visible_binding, user_properties);
    };

    if (object_json.contains("image") && ! object_json.at("image").is_null()) {
        wpscene::WPImageObject object;
        if (! object.FromJson(object_json, *context.vfs)) return false;
        register_visibility_contract(object.id, LazyMaterializeKind::Image);
        resolve_visibility(object);

        const bool requested_visible = object.visible;
        ParseImageObj(context, object, true);
        context.scene->SetLayerLocalVisibility(object.id, requested_visible);
        context.scene->ApplyLayerVisibility(object.id);
        if (out_layer_id) *out_layer_id = object.id;
        return context.object_nodes.count(object.id) != 0;
    }

    if (object_json.contains("particle") && ! object_json.at("particle").is_null()) {
        wpscene::WPParticleObject object;
        if (! object.FromJson(object_json, *context.vfs)) return false;
        register_visibility_contract(object.id, LazyMaterializeKind::Particle);
        resolve_visibility(object);
        ParseParticleObj(context, object);
        context.scene->SetLayerLocalVisibility(object.id, object.visible);
        context.scene->ApplyLayerVisibility(object.id);
        if (out_layer_id) *out_layer_id = object.id;
        return context.object_nodes.count(object.id) != 0;
    }

    if (object_json.contains("light") && ! object_json.at("light").is_null()) {
        wpscene::WPLightObject object;
        if (! object.FromJson(object_json, *context.vfs)) return false;
        register_visibility_contract(object.id, LazyMaterializeKind::None);
        resolve_visibility(object);
        ParseLightObj(context, object);
        context.scene->SetLayerLocalVisibility(object.id, object.visible);
        context.scene->ApplyLayerVisibility(object.id);
        if (out_layer_id) *out_layer_id = object.id;
        return context.object_nodes.count(object.id) != 0;
    }

    if (object_json.contains("sound") && ! object_json.at("sound").is_null()) {
        wpscene::WPSoundObject object;
        if (! object.FromJson(object_json, *context.vfs)) return false;
        resolve_visibility(object);
        if (context.scene->soundManager == nullptr) return false;
        const auto sound_handle =
            WPSoundParser::Parse(object, *context.vfs, *context.scene->soundManager);
        if (sound_handle == 0) return false;
        context.scene->objectRuntimeSoundHandles[object.id] = sound_handle;
        if (out_layer_id) *out_layer_id = object.id;
        return true;
    }

    if (object_json.contains("text") && ! object_json.at("text").is_null()) {
        wpscene::WPTextObject object;
        if (! object.FromJson(object_json, *context.vfs)) return false;
        register_visibility_contract(object.id, LazyMaterializeKind::Text);
        resolve_visibility(object);
        ParseTextObj(context, object);
        context.scene->SetLayerLocalVisibility(object.id, object.visible);
        context.scene->ApplyLayerVisibility(object.id);
        if (out_layer_id) *out_layer_id = object.id;
        return context.object_nodes.count(object.id) != 0;
    }

    if (object_json.contains("model") && ! object_json.at("model").is_null()) {
        WPModelObject object;
        if (! object.FromJson(object_json, *context.vfs)) return false;
        register_visibility_contract(object.id, LazyMaterializeKind::None);
        resolve_visibility(object);
        ParseModelObj(context, object);
        context.scene->SetLayerLocalVisibility(object.id, object.visible);
        context.scene->ApplyLayerVisibility(object.id);
        if (out_layer_id) *out_layer_id = object.id;
        return context.object_nodes.count(object.id) != 0;
    }

    WPEmptyObject object;
    if (! object.FromJson(object_json, *context.vfs)) return false;
    register_visibility_contract(object.id, LazyMaterializeKind::None);
    resolve_visibility(object);
    ParseEmptyObj(context, object);
    context.scene->SetLayerLocalVisibility(object.id, object.visible);
    context.scene->ApplyLayerVisibility(object.id);
    if (out_layer_id) *out_layer_id = object.id;
    return context.object_nodes.count(object.id) != 0;
}
} // namespace

bool wallpaper::ConfigureSceneVolumetrics(Scene& scene, fs::VFS& vfs) {
    return ConfigureSceneVolumetricsImpl(scene, vfs);
}

bool wallpaper::ConfigureSceneBloom(Scene& scene, fs::VFS& vfs) {
    auto* updater = dynamic_cast<WPShaderValueUpdater*>(scene.shaderValueUpdater.get());
    if (updater == nullptr) return false;
    ParseContext context;
    context.scene          = std::shared_ptr<Scene>(&scene, [](Scene*) {});
    context.shader_updater = updater;
    context.ortho_w        = std::max(1, scene.ortho[0]);
    context.ortho_h        = std::max(1, scene.ortho[1]);
    context.vfs            = &vfs;
    return ConfigureSceneBloomPass(context);
}

bool wallpaper::MaterializeDeferredImageLayer(Scene& scene, int32_t layer_id,
                                              const UserPropertyMap* user_properties) {
    if (scene.deferredRuntimeImageLayerIds.count(layer_id) == 0) return false;

    const auto config_it = scene.initialLayerConfigJson.find(layer_id);
    if (config_it == scene.initialLayerConfigJson.end()) return false;

    nlohmann::json object_json;
    if (! PARSE_JSON(config_it->second, object_json)) return false;

    ParseContext context {};
    if (! InitDynamicParseContext(context, scene, user_properties)) return false;

    std::shared_ptr<SceneNode> placeholder_node;
    if (auto node_it = context.object_nodes.find(layer_id); node_it != context.object_nodes.end()) {
        placeholder_node = node_it->second;
    }

    // Deferred materialization may now run as a residency warm-up while the authored layer is still
    // hidden. The parser entry points still require `visible=true` to build the full runtime node,
    // so preserve the logical visibility first and restore it after the placeholder swap.
    const bool local_visible = scene.GetLayerLocalVisibility(layer_id);
    const auto binding = scene.GetLayerParentBinding(layer_id);
    const auto apply_placeholder_transform = [layer_id, &binding, &placeholder_node,
                                              &context](auto& object) {
        object.id         = layer_id;
        object.visible    = true;
        object.parent     = binding.parent_id;
        object.attachment = binding.attachment;
        if (placeholder_node == nullptr) return;

        // Runtime materialization must preserve the script-mutated logical transform carried by
        // the placeholder. Reusing the initial JSON transform would make late-visible multilingual
        // branches snap back to authored values after user/script updates.
        object.name           = placeholder_node->Name();
        const auto& translate = placeholder_node->Translate();
        const auto& scale     = placeholder_node->Scale();
        const auto& rotation  = placeholder_node->Rotation();
        object.origin         = { translate.x(), translate.y(), translate.z() };
        object.scale          = { scale.x(), scale.y(), scale.z() };
        object.angles         = { rotation.x(), rotation.y(), rotation.z() };
        if (context.shader_updater != nullptr) {
            if (const auto* data = context.shader_updater->GetNodeData(placeholder_node.get())) {
                object.parallaxDepth         = data->parallaxDepth;
                object.parallaxDepthAuthored = data->parallaxDepthAuthored;
            }
        }
    };

    scene.objectRuntimeNodes.erase(layer_id);
    scene.imageLayers.erase(layer_id);
    scene.layerNodes[layer_id] = nullptr;

    if (object_json.contains("shape") && ! object_json.at("shape").is_null()) {
        WPShapeObject object;
        if (! object.FromJson(object_json, *context.vfs)) return false;
        apply_placeholder_transform(object);
        // Direct-draw shapes are stored in the same deferred image set because their runtime path
        // deliberately reuses image-effect materialization. On wake-up they still need to be parsed
        // from the authored shape JSON first, otherwise WPImageObject would try to load an empty
        // image path and abort the visible subtree before clock text can materialize.
        ParseShapeObj(context, object, true);
    } else {
        wpscene::WPImageObject object;
        if (! object.FromJson(object_json, *context.vfs)) return false;
        apply_placeholder_transform(object);
        ParseImageObj(context, object, true);
    }
    auto node_it = context.object_nodes.find(layer_id);
    if (node_it == context.object_nodes.end() || ! node_it->second) return false;

    // The initial scene script scan can only register layer-level properties for deferred image
    // placeholders. Effect visibility targets do not exist until this late materialization pass builds
    // the actual effect chain, so register them here while the authored effect ids and the concrete
    // SceneImageEffect instances are both available.
    RegisterEffectVisibilityBindings(context, object_json);

    ReplaceDeferredPlaceholderNode(context, layer_id, placeholder_node, node_it->second);
    scene.layerNodes[layer_id] = node_it->second.get();
    RestoreRenderOrderProxyChildrenForLayer(scene, layer_id, node_it->second.get());
    for (auto it = scene.layerNameToId.begin(); it != scene.layerNameToId.end();) {
        if (it->second == layer_id) {
            it = scene.layerNameToId.erase(it);
        } else {
            ++it;
        }
    }
    if (! node_it->second->Name().empty()) {
        scene.layerNameToId[node_it->second->Name()] = layer_id;
    }
    scene.SetLayerLocalVisibility(layer_id, local_visible);
    scene.ApplyLayerVisibility(layer_id);
    scene.deferredRuntimeImageLayerIds.erase(layer_id);
    return true;
}

bool wallpaper::MaterializeDeferredParticleLayer(Scene& scene, int32_t layer_id,
                                                 const UserPropertyMap* user_properties) {
    if (scene.deferredRuntimeParticleLayerIds.count(layer_id) == 0) return false;

    const auto config_it = scene.initialLayerConfigJson.find(layer_id);
    if (config_it == scene.initialLayerConfigJson.end()) return false;

    nlohmann::json object_json;
    if (! PARSE_JSON(config_it->second, object_json)) return false;

    ParseContext context {};
    if (! InitDynamicParseContext(context, scene, user_properties)) return false;

    std::shared_ptr<SceneNode> placeholder_node;
    if (auto node_it = context.object_nodes.find(layer_id); node_it != context.object_nodes.end()) {
        placeholder_node = node_it->second;
    }

    wpscene::WPParticleObject object;
    if (! object.FromJson(object_json, *context.vfs)) return false;

    // See image deferred materialization above: build the concrete runtime object with parser-local
    // visibility enabled, then restore the Scene-owned logical visibility contract.
    const bool local_visible = scene.GetLayerLocalVisibility(layer_id);
    object.id          = layer_id;
    object.visible     = true;
    const auto binding = scene.GetLayerParentBinding(layer_id);
    object.parent      = binding.parent_id;
    object.attachment  = binding.attachment;
    if (placeholder_node != nullptr) {
        object.name           = placeholder_node->Name();
        const auto& translate = placeholder_node->Translate();
        const auto& scale     = placeholder_node->Scale();
        const auto& rotation  = placeholder_node->Rotation();
        object.origin         = { translate.x(), translate.y(), translate.z() };
        object.scale          = { scale.x(), scale.y(), scale.z() };
        object.angles         = { rotation.x(), rotation.y(), rotation.z() };
        if (context.shader_updater != nullptr) {
            if (const auto* data = context.shader_updater->GetNodeData(placeholder_node.get())) {
                object.parallaxDepth         = data->parallaxDepth;
                object.parallaxDepthAuthored = data->parallaxDepthAuthored;
            }
        }
    }

    scene.objectRuntimeNodes.erase(layer_id);
    scene.layerNodes[layer_id] = nullptr;

    ParseParticleObj(context, object);
    auto node_it = context.object_nodes.find(layer_id);
    if (node_it == context.object_nodes.end() || ! node_it->second) return false;

    ReplaceDeferredPlaceholderNode(context, layer_id, placeholder_node, node_it->second);
    scene.layerNodes[layer_id] = node_it->second.get();
    for (auto it = scene.layerNameToId.begin(); it != scene.layerNameToId.end();) {
        if (it->second == layer_id) {
            it = scene.layerNameToId.erase(it);
        } else {
            ++it;
        }
    }
    if (! node_it->second->Name().empty()) {
        scene.layerNameToId[node_it->second->Name()] = layer_id;
    }
    scene.SetLayerLocalVisibility(layer_id, local_visible);
    scene.ApplyLayerVisibility(layer_id);
    scene.deferredRuntimeParticleLayerIds.erase(layer_id);
    return true;
}

bool wallpaper::MaterializeDeferredTextLayer(Scene& scene, int32_t layer_id,
                                             const UserPropertyMap* user_properties) {
    if (scene.deferredRuntimeTextLayerIds.count(layer_id) == 0) return false;

    const auto config_it = scene.initialLayerConfigJson.find(layer_id);
    if (config_it == scene.initialLayerConfigJson.end()) return false;

    nlohmann::json object_json;
    if (! PARSE_JSON(config_it->second, object_json)) return false;

    ParseContext context {};
    if (! InitDynamicParseContext(context, scene, user_properties)) return false;

    std::shared_ptr<SceneNode> placeholder_node;
    if (auto node_it = context.object_nodes.find(layer_id); node_it != context.object_nodes.end()) {
        placeholder_node = node_it->second;
    }

    wpscene::WPTextObject object;
    if (const auto state_it = scene.textLayers.find(layer_id); state_it != scene.textLayers.end()) {
        object = state_it->second.object;
    } else {
        if (! object.FromJson(object_json, *context.vfs)) return false;
    }

    // Text parsing uses `visible` as an early materialization gate, so residency warm-up must keep
    // the parser visible while preserving the Scene-visible state externally.
    object.id          = layer_id;
    object.visible     = true;
    const bool local_visible = scene.GetLayerLocalVisibility(layer_id);
    const auto binding = scene.GetLayerParentBinding(layer_id);
    object.parent      = binding.parent_id;
    object.attachment  = binding.attachment;
    if (placeholder_node != nullptr) {
        object.name           = placeholder_node->Name();
        const auto& translate = placeholder_node->Translate();
        const auto& scale     = placeholder_node->Scale();
        const auto& rotation  = placeholder_node->Rotation();
        object.origin         = { translate.x(), translate.y(), translate.z() };
        object.scale          = { scale.x(), scale.y(), scale.z() };
        object.angles         = { rotation.x(), rotation.y(), rotation.z() };
        if (context.shader_updater != nullptr) {
            if (const auto* data = context.shader_updater->GetNodeData(placeholder_node.get())) {
                object.parallaxDepth         = data->parallaxDepth;
                object.parallaxDepthAuthored = data->parallaxDepthAuthored;
            }
        }
    }

    scene.objectRuntimeNodes.erase(layer_id);
    scene.layerNodes[layer_id] = nullptr;

    ParseTextObj(context, object);
    auto node_it = context.object_nodes.find(layer_id);
    if (node_it == context.object_nodes.end() || ! node_it->second) return false;

    // Deferred text effects follow the same late-target rule as image effects: their concrete effect
    // passes are absent during the initial placeholder scan, so user-bound effect visibility must be
    // registered after ParseTextObj() creates the runtime effect chain.
    RegisterEffectVisibilityBindings(context, object_json);

    ReplaceDeferredPlaceholderNode(context, layer_id, placeholder_node, node_it->second);
    scene.layerNodes[layer_id] = node_it->second.get();
    for (auto it = scene.layerNameToId.begin(); it != scene.layerNameToId.end();) {
        if (it->second == layer_id) {
            it = scene.layerNameToId.erase(it);
        } else {
            ++it;
        }
    }
    if (! node_it->second->Name().empty()) {
        scene.layerNameToId[node_it->second->Name()] = layer_id;
    }
    scene.SetLayerLocalVisibility(layer_id, local_visible);
    scene.ApplyLayerVisibility(layer_id);
    scene.deferredRuntimeTextLayerIds.erase(layer_id);
    if (const auto state_it = scene.textLayers.find(layer_id); state_it != scene.textLayers.end()) {
    } else {
    }
    return true;
}

bool wallpaper::CreateDynamicSceneLayer(
    Scene& scene, const nlohmann::json& object_json, const UserPropertyMap* user_properties,
    std::vector<WPSceneScriptRegistration>* out_binding_registrations,
    std::vector<WPSceneScriptRegistration>* out_script_registrations,
    std::vector<WPSceneScriptRegistration>* out_property_animation_registrations,
    std::string* out_initial_config_json, int32_t* out_layer_id) {
    if (! object_json.is_object()) return false;

    ParseContext context {};
    if (! InitDynamicParseContext(context, scene, user_properties)) return false;

    nlohmann::json normalized_object_json = object_json;
    int32_t        layer_id               = 0;
    GET_JSON_NAME_VALUE_NOWARN(normalized_object_json, "id", layer_id);
    if (layer_id <= 0 || scene.layerNodes.count(layer_id) != 0 ||
        scene.objectRuntimeNodes.count(layer_id) != 0) {
        layer_id                     = AllocateDynamicLayerId(scene);
        normalized_object_json["id"] = layer_id;
    }

    if (! ParseDynamicSceneObject(context, normalized_object_json, user_properties, &layer_id)) {
        return false;
    }

    auto       node_it = context.object_nodes.find(layer_id);
    SceneNode* layer_node =
        node_it != context.object_nodes.end() && node_it->second ? node_it->second.get() : nullptr;
    const bool has_sound_runtime = scene.objectRuntimeSoundHandles.count(layer_id) != 0;
    if (layer_node == nullptr && ! has_sound_runtime) return false;

    const auto binding_start            = scene.bindingRegistrations.size();
    const auto property_animation_start = scene.propertyAnimationRegistrations.size();
    const auto script_start             = scene.scriptRegistrations.size();
    RegisterSceneScriptsForObject(context, normalized_object_json);

    scene.layerOrder.push_back(layer_id);
    scene.layerNodes[layer_id]             = layer_node;
    scene.initialLayerConfigJson[layer_id] = normalized_object_json.dump();
    std::string layer_name = layer_node != nullptr
                                 ? layer_node->Name()
                                 : normalized_object_json.value("name", std::string {});
    if (! layer_name.empty()) {
        scene.layerNameToId[layer_name] = layer_id;
    }

    if (out_binding_registrations != nullptr) {
        out_binding_registrations->assign(scene.bindingRegistrations.begin() + binding_start,
                                          scene.bindingRegistrations.end());
    }
    if (out_property_animation_registrations != nullptr) {
        out_property_animation_registrations->assign(scene.propertyAnimationRegistrations.begin() +
                                                         property_animation_start,
                                                     scene.propertyAnimationRegistrations.end());
    }
    if (out_script_registrations != nullptr) {
        out_script_registrations->assign(scene.scriptRegistrations.begin() + script_start,
                                         scene.scriptRegistrations.end());
    }
    if (out_initial_config_json != nullptr) {
        *out_initial_config_json = scene.initialLayerConfigJson.at(layer_id);
    }
    if (out_layer_id != nullptr) {
        *out_layer_id = layer_id;
    }
    return true;
}

std::shared_ptr<Scene> WPSceneParser::Parse(std::string_view scene_id, const std::string& buf,
                                            fs::VFS& vfs, audio::SoundManager& sm,
                                            const UserPropertyMap* user_properties,
                                            double                 text_render_scale) {
    nlohmann::json json;
    if (! PARSE_JSON(buf, json)) return nullptr;

    ScopedJsonUserProperties json_user_scope(user_properties, &json);

    wpscene::WPScene sc;
    sc.FromJson(json);
    //	LOG_INFO(nlohmann::json(sc).dump(4));

    ParseContext context {};
    context.user_properties = user_properties;

    std::vector<WPObjectVar>                        wp_objs;
    std::unordered_map<int32_t, VisibilityContract> layer_visibility_contracts;
    std::unordered_map<int32_t, std::string>        initial_layer_config_json_by_id;
    std::unordered_set<int32_t>                     dependency_source_ids;
    std::unordered_set<std::string>                 script_referenced_layer_names;

    CollectScriptReferencedLayerNames(json, script_referenced_layer_names);

    for (auto& obj : json.at("objects")) {
        if (obj.contains("dependencies") && obj.at("dependencies").is_array()) {
            for (const auto& dependency : obj.at("dependencies")) {
                int32_t dependency_id = 0;
                GET_JSON_VALUE_NOWARN(dependency, dependency_id);
                if (dependency_id != 0) dependency_source_ids.insert(dependency_id);
            }
        }
    }

    bool has_3d_models = false;
    for (auto& obj : json.at("objects")) {
        int32_t     object_id = 0;
        std::string object_name;
        GET_JSON_NAME_VALUE_NOWARN(obj, "id", object_id);
        GET_JSON_NAME_VALUE_NOWARN(obj, "name", object_name);

        // Official 2.3 MSAA gate: any non-null objects[].model counts, including
        // hidden/pruned layers. An image path of models/foo.json does not.
        if (obj.contains("model") && ! obj.at("model").is_null()) {
            has_3d_models = true;
        }

        const bool is_image_layer = obj.contains("image") && ! obj.at("image").is_null();
        const bool script_referenced_hidden_image =
            is_image_layer && object_id != 0 && ! object_name.empty() &&
            ! ReadAuthoredVisibleValue(obj) &&
            script_referenced_layer_names.count(object_name) != 0;

        LazyMaterializeKind lazy_kind = LazyMaterializeKind::None;
        if (obj.contains("image") && ! obj.at("image").is_null()) {
            lazy_kind = LazyMaterializeKind::Image;
        } else if (obj.contains("particle") && ! obj.at("particle").is_null()) {
            lazy_kind = LazyMaterializeKind::Particle;
        } else if (obj.contains("text") && ! obj.at("text").is_null()) {
            lazy_kind = LazyMaterializeKind::Text;
        }

        if (object_id != 0) {
            // Deferred runtime materialization must keep the exact authored JSON for this layer id.
            // The parser prunes static hidden layers before building Scene::layerOrder, so using
            // the surviving vector index later can point a hidden dynamic particle/text layer at a
            // different original object and make false->true user-property toggles realize the
            // wrong runtime layer.
            initial_layer_config_json_by_id[object_id] = obj.dump();

            auto visibility_contract =
                BuildObjectVisibilityContractFromJson(obj,
                                                      user_properties,
                                                      dependency_source_ids.count(object_id) != 0,
                                                      script_referenced_hidden_image,
                                                      lazy_kind);
            layer_visibility_contracts[object_id] = visibility_contract;

        }

        if (obj.contains("image") && ! obj.at("image").is_null()) {
            AddWPObject<wpscene::WPImageObject>(wp_objs, obj, vfs, user_properties);
        } else if (obj.contains("particle") && ! obj.at("particle").is_null()) {
            AddWPObject<wpscene::WPParticleObject>(wp_objs, obj, vfs, user_properties);
        } else if (obj.contains("sound") && ! obj.at("sound").is_null()) {
            AddWPObject<wpscene::WPSoundObject>(wp_objs, obj, vfs, user_properties);
        } else if (obj.contains("light") && ! obj.at("light").is_null()) {
            AddWPObject<wpscene::WPLightObject>(wp_objs, obj, vfs, user_properties);
        } else if (obj.contains("text") && ! obj.at("text").is_null()) {
            AddWPObject<wpscene::WPTextObject>(wp_objs, obj, vfs, user_properties);
        } else if (obj.contains("model") && ! obj.at("model").is_null()) {
            // Scene-level 3D models have their own parser/materialization path. Classify them
            // before the generic empty fallback so a model-only layer cannot silently become a
            // transform placeholder.
            AddWPObject<WPModelObject>(wp_objs, obj, vfs, user_properties);
        } else if (obj.contains("shape") && ! obj.at("shape").is_null()) {
            // Wallpaper Engine uses shape objects for direct-draw effects such as Light Shafts.
            // They have no `image` field, so they must be classified before the generic empty
            // object fallback or their effect chain never reaches the render graph.
            AddWPObject<WPShapeObject>(wp_objs, obj, vfs, user_properties);
        } else {
            AddWPObject<WPEmptyObject>(wp_objs, obj, vfs, user_properties);
        }
    }

    context.initial_parent_by_layer_id.clear();
    for (const auto& obj : wp_objs) {
        const auto id = GetObjectId(obj);
        if (! id.has_value() || *id == 0) continue;
        // Effective visibility is a parent-chain property, not just a layer-local property. Keep
        // the authored parse-time parent table before any pruning/materialization so lazy layer
        // decisions can mirror Scene::IsLayerVisible() without needing concrete runtime nodes.
        context.initial_parent_by_layer_id[*id] = GetObjectParentId(obj);
    }
    const std::vector<WPObjectVar> authored_objects = wp_objs;

    std::unordered_set<int32_t> hidden_object_ids;
    for (const auto& obj : wp_objs) {
        const auto id = GetObjectId(obj);
        if (id.has_value()) {
            const auto contract_it = layer_visibility_contracts.find(*id);
            if (contract_it != layer_visibility_contracts.end()) {
                if (contract_it->second.requires_runtime_contract) continue;
                if (contract_it->second.can_prune_at_parse_time) {
                    hidden_object_ids.insert(*id);
                    continue;
                }
            }
        }
        if (id.has_value() && ! IsObjectVisible(obj)) hidden_object_ids.insert(*id);
    }

    bool changed = true;
    while (changed) {
        changed = false;
        for (const auto& obj : wp_objs) {
            const auto parent_id = GetObjectParentId(obj);
            if (parent_id == 0) continue;
            if (! hidden_object_ids.count(parent_id)) continue;

            const auto id = GetObjectId(obj);
            if (id.has_value()) {
                const auto contract_it = layer_visibility_contracts.find(*id);
                if (contract_it != layer_visibility_contracts.end() &&
                    contract_it->second.requires_runtime_contract) {
                    continue;
                }
            }
            if (id.has_value() && hidden_object_ids.insert(*id).second) changed = true;
        }
    }

    std::erase_if(wp_objs, [&hidden_object_ids, &layer_visibility_contracts](const auto& obj) {
        const auto id = GetObjectId(obj);
        if (id.has_value()) {
            const auto contract_it = layer_visibility_contracts.find(*id);
            if (contract_it != layer_visibility_contracts.end() &&
                contract_it->second.requires_runtime_contract) {
                return false;
            }
        }
        return id.has_value() && hidden_object_ids.count(*id) != 0;
    });

    for (const auto& obj : wp_objs) {
        std::visit(visitor::overload {
                       [&context](const wpscene::WPImageObject& obj) {
                           if (obj.parent != 0) context.dependent_parent_ids.insert(obj.parent);
                       },
                       [&context](const wpscene::WPParticleObject& obj) {
                           if (obj.parent != 0) context.dependent_parent_ids.insert(obj.parent);
                       },
                       [&context](const wpscene::WPLightObject& obj) {
                           if (obj.parent != 0) context.dependent_parent_ids.insert(obj.parent);
                       },
                       [&context](const wpscene::WPTextObject& obj) {
                           if (obj.parent != 0) context.dependent_parent_ids.insert(obj.parent);
                       },
                       [&context](const WPModelObject& obj) {
                           if (obj.parent != 0) context.dependent_parent_ids.insert(obj.parent);
                       },
                       [&context](const WPShapeObject& obj) {
                           // Shape effect layers participate in the same inherited-parent ordering
                           // as images; otherwise a parented direct-draw light shaft would resolve
                           // outside the transform that authored its final screen position.
                           if (obj.parent != 0) context.dependent_parent_ids.insert(obj.parent);
                       },
                       [&context](const WPEmptyObject& obj) {
                           if (obj.parent != 0) context.dependent_parent_ids.insert(obj.parent);
                       },
                       [](const wpscene::WPSoundObject&) {
                       },
                   },
                   obj);
    }

    if (sc.general.orthogonalprojection.auto_) {
        i32 w = 0, h = 0;
        for (auto& obj : wp_objs) {
            auto*                       img            = std::get_if<wpscene::WPImageObject>(&obj);
            auto*                       shape          = std::get_if<WPShapeObject>(&obj);
            const std::array<float, 2>* candidate_size = nullptr;
            if (img != nullptr) {
                candidate_size = &img->size;
            } else if (shape != nullptr && shape->has_size) {
                // Authored shape sizes can define the projection when no larger image layer exists;
                // implicit full-screen shape sizes are resolved later from the final projection and
                // must not feed this auto-projection bootstrap loop.
                candidate_size = &shape->size;
            }
            if (candidate_size == nullptr) continue;
            i32 size = (i32)(candidate_size->at(0) * candidate_size->at(1));
            if (size > w * h) {
                w = (i32)candidate_size->at(0);
                h = (i32)candidate_size->at(1);
            }
        }
        sc.general.orthogonalprojection.width  = w;
        sc.general.orthogonalprojection.height = h;
    }

    InitContext(context, vfs, sc, scene_id);
    for (const auto& obj : authored_objects) {
        if (const auto* image = std::get_if<wpscene::WPImageObject>(&obj)) {
            NotePassthroughLayer(context, *image);
        }
    }
    for (const auto& obj : authored_objects) {
        FillSceneObjectIdentity(*context.scene, obj);
    }
    for (const auto& obj : authored_objects) {
        const auto id = GetObjectId(obj);
        if (! id.has_value() || *id == 0) continue;
        // SO7 0x1401de931 / 0x14018802c: +0x180 parent after factory parse,
        // adjustTransforms=0 so origin +0x128 is left as 0x1401a4530 wrote it.
        context.scene->BindSceneObjectParent(*id, GetObjectParentId(obj), GetObjectAttachment(obj));
    }
    context.scene->has3dModels         = has_3d_models;
    context.layer_visibility_contracts = std::move(layer_visibility_contracts);
    context.scene->soundManager        = &sm;
    // Text atlases and effect ping-pong stay in the authored letter box. Desktop density is
    // applied when those results are composited, not by rebuilding glyphs at the output scale.
    (void)text_render_scale;
    context.scene->textRenderScale             = 1.0;
    context.scene->offscreenDependencyLayerIds = dependency_source_ids;
    if (user_properties) {
        context.scene->userProperties = *user_properties;
    } else {
        context.scene->userProperties.clear();
    }
    ParseCamera(context, sc);

    {
        context.scene->renderTargets[SpecTex_Default.data()] = {
            .width     = context.ortho_w,
            .height    = context.ortho_h,
            .mapWidth  = context.ortho_w,
            .mapHeight = context.ortho_h,
            .bind      = { .enable = true, .screen = true },
        };
        // Stable compose snapshot for `_rt_default` self-writes. Screen-bound so a live output
        // resize keeps the snapshot the same extent as the compose target it mirrors.
        context.scene->renderTargets[SpecTex_DefaultPingPong.data()] = {
            .width     = context.ortho_w,
            .height    = context.ortho_h,
            .mapWidth  = context.ortho_w,
            .mapHeight = context.ortho_h,
            .bind      = { .enable = true, .screen = true },
        };
        ConfigureSceneMsaa(*context.scene);
        context.scene->renderTargets[WE_MIP_MAPPED_FRAME_BUFFER.data()] = {
            .width      = context.ortho_w,
            .height     = context.ortho_h,
            .mapWidth   = context.ortho_w,
            .mapHeight  = context.ortho_h,
            .has_mipmap = true,
            .bind       = { .enable = true, .name = SpecTex_Default.data() }
        };
        context.scene->renderTargets[std::string(SpecTex_ShadowAtlas)] = {
            .width           = 2,
            .height          = 2,
            .mapWidth        = 2,
            .mapHeight       = 2,
            .allowReuse      = false,
            .comparisonDepth = true,
        };
    }
    context.scene->scene_id = scene_id;
    // Scene Bloom owns a synthetic shader and its cache keys include the scene id, so it must be
    // built only after the parse context has reached the same identity state as authored shaders.
    ConfigureSceneBloomPass(context);

    context.scene->lighting = {};
    for (const auto& obj : wp_objs) {
        const auto* light_obj = std::get_if<wpscene::WPLightObject>(&obj);
        if (light_obj == nullptr) continue;
        const auto& token = light_obj->light;
        if (token == "lpoint" || token == "point") {
            context.scene->lighting.point++;
            context.scene->lighting.point_shadow.push_back(light_obj->castshadows ? 1 : 0);
        } else if (token == "lspot" || token == "spot") {
            context.scene->lighting.spot++;
            context.scene->lighting.spot_shadow.push_back(light_obj->castshadows ? 1 : 0);
            context.scene->lighting.spot_cookie.push_back(
                (light_obj->usecookie && ! light_obj->cookie.empty()) ? 1 : 0);
        } else if (token == "ldirectional" || token == "directional") {
            context.scene->lighting.directional++;
            context.scene->lighting.directional_shadow.push_back(light_obj->castshadows ? 1 : 0);
        } else if (token == "ltube" || token == "tube") {
            context.scene->lighting.tube++;
        }
    }

    for (WPObjectVar& obj : wp_objs) {
        std::visit(visitor::overload {
                       [&context](wpscene::WPImageObject& obj) {
                           const auto* visibility_contract =
                               FindLayerVisibilityContract(context, obj.id);
                           const bool force_runtime_materialization =
                               visibility_contract != nullptr &&
                               visibility_contract->dependency_source;

                           // Image layers now keep the authored initial visibility intact while
                           // deferring runtime-controlled hidden nodes. Dependency sources remain
                           // concrete because another visible layer may sample their render target
                           // even while their own visibility contract begins false.
                           ParseImageObj(context, obj, force_runtime_materialization);
                       },
                       [&context](wpscene::WPParticleObject& obj) {
                           ParseParticleObj(context, obj);
                       },
                       [&context, &sm](wpscene::WPSoundObject& obj) {
                           context.scene->objectRuntimeSoundHandles[obj.id] =
                               WPSoundParser::Parse(obj, *context.vfs, sm);
                       },
                       [&context](wpscene::WPLightObject& obj) {
                           ParseLightObj(context, obj);
                       },
                       [&context](wpscene::WPTextObject& obj) {
                           ParseTextObj(context, obj);
                       },
                       [&context](WPModelObject& obj) {
                           ParseModelObj(context, obj);
                       },
                       [&context](WPShapeObject& obj) {
                           const auto* visibility_contract =
                               FindLayerVisibilityContract(context, obj.id);
                           const bool force_runtime_materialization =
                               visibility_contract != nullptr &&
                               visibility_contract->dependency_source;

                           // Shape direct-draw effects mirror image-layer visibility behavior:
                           // dependency-source shapes stay materialized for consumers, while
                           // hidden runtime branches can stay logical until effective visibility.
                           ParseShapeObj(context, obj, force_runtime_materialization);
                       },
                       [&context](WPEmptyObject& obj) {
                           ParseEmptyObj(context, obj);
                       },
                   },
                   obj);
    }

    RegisterSceneScripts(context, json);

    context.scene->layerOrder.clear();
    context.scene->layerNodes.clear();
    context.scene->initialLayerConfigJson.clear();
    context.scene->layerNameToId.clear();
    for (const auto& obj : wp_objs) {
        const auto object_id = GetObjectId(obj);
        if (! object_id.has_value()) continue;

        auto node_it = context.object_nodes.find(*object_id);
        context.scene->layerOrder.push_back(*object_id);
        context.scene->layerNodes[*object_id] =
            node_it != context.object_nodes.end() && node_it->second ? node_it->second.get()
                                                                     : nullptr;
        if (auto config_it = initial_layer_config_json_by_id.find(*object_id);
            config_it != initial_layer_config_json_by_id.end()) {
            context.scene->initialLayerConfigJson[*object_id] = config_it->second;
        }

        const auto node_name = node_it != context.object_nodes.end() && node_it->second
                                   ? node_it->second->Name()
                                   : GetObjectName(obj);
        if (! node_name.empty()) {
            context.scene->layerNameToId[node_name] = *object_id;
        }
    }

    context.scene->ApplyAllLayerVisibility();

    ConfigureSceneVolumetricsImpl(*context.scene, *context.vfs);

    return context.scene;
}

std::shared_ptr<Scene> WPSceneParser::Parse(std::string_view scene_id, const std::string& buf,
                                            fs::VFS& vfs, audio::SoundManager& sm) {
    return Parse(scene_id, buf, vfs, sm, nullptr);
}
