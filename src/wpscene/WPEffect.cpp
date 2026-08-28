#include "wpscene/WPEffect.h"

#include <algorithm>
#include <cmath>
#include <filesystem>

#include "Fs/VFS.h"
#include "Utils/Logging.h"
#include "WPJson.hpp"

using namespace wallpaper::wpscene;

namespace
{

void ReadVisibleBinding(const nlohmann::json& json, wallpaper::VisibleBinding* binding) {
    if (! json.is_object()) return;

    GET_JSON_NAME_VALUE_NOWARN(json, "value", binding->value);
    if (! json.contains("user") || json.at("user").is_null()) return;

    const auto& user = json.at("user");
    if (user.is_string()) {
        GET_JSON_VALUE(user, binding->user.name);
        return;
    }

    if (! user.is_object()) return;

    GET_JSON_NAME_VALUE_NOWARN(user, "name", binding->user.name);
    GET_JSON_NAME_VALUE_NOWARN(user, "condition", binding->user.condition);
}

int32_t PositiveRoundedPixel(float value) {
    return std::max(1, static_cast<int32_t>(std::lround(std::max(1.0f, value))));
}

} // namespace

bool WPEffectCommand::FromJson(const nlohmann::json& json) {
    GET_JSON_NAME_VALUE(json, "command", command);
    GET_JSON_NAME_VALUE(json, "target", target);
    GET_JSON_NAME_VALUE(json, "source", source);
    return true;
}

bool WPEffectFbo::FromJson(const nlohmann::json& json) {
    GET_JSON_NAME_VALUE(json, "name", name);
    GET_JSON_NAME_VALUE(json, "format", format);

    GET_JSON_NAME_VALUE_NOWARN(json, "scale", scale);
    GET_JSON_NAME_VALUE_NOWARN(json, "fit", fit);
    if(scale == 0) { 
        LOG_ERROR("fbo scale can't be 0");
        scale = 1;
    }
    return true;
}

std::array<int32_t, 2> WPEffectFbo::ResolveSize(std::array<float, 2> source_size) const {
    const float source_width  = std::max(1.0f, source_size[0]);
    const float source_height = std::max(1.0f, source_size[1]);

    if (fit > 0) {
        const float longest_edge = std::max(source_width, source_height);
        const float fit_scale    = static_cast<float>(fit) / longest_edge;
        return {
            PositiveRoundedPixel(source_width * fit_scale),
            PositiveRoundedPixel(source_height * fit_scale),
        };
    }

    const float divisor = static_cast<float>(std::max<uint32_t>(1u, scale));
    return {
        PositiveRoundedPixel(source_width / divisor),
        PositiveRoundedPixel(source_height / divisor),
    };
}

// The blacklist belongs to the shared effect parser, not to WPImageObject. Text effects and image
// effects must make the same safety decision from the same data so both paths stay behaviorally
// identical after the text primitive split.
const std::unordered_set<std::string> WPImageEffect::BLACKLISTED_WORKSHOP_EFFECTS =
{
    // Keep this quarantine list evidence-driven: a blocked effect disappears before the scene
    // parser can build render passes or visibility bindings for it. Audio Responsive
    // Oscilloscope (2799421411) must remain enabled because it is the authored source of the
    // user-controlled music halo in scene 3585875739, and hiding it here makes the `osc` toggle
    // impossible to observe even when audio samples are flowing correctly.
};

bool WPImageEffect::IsEffectBlacklisted(const std::string& filePath) {
    std::filesystem::path path(filePath);
    // Workshop effect paths encode the item id in the grandparent directory. Keep the path walk
    // local to the effect parser so object types only receive a parsed visibility result.
    if (path.has_parent_path()) {
        path = path.parent_path();
        if(path.has_parent_path()) {
            std::string effectId = path.parent_path().filename().string();
            std::string parentPath = path.parent_path().string();
            return WPImageEffect::BLACKLISTED_WORKSHOP_EFFECTS.find(effectId) != WPImageEffect::BLACKLISTED_WORKSHOP_EFFECTS.end();
        }
    }
    return false;
}

bool WPImageEffect::HasEnabledCombo(const std::string& combo_name) const {
    const auto combo_is_enabled = [&combo_name](const auto& combos) {
        const auto combo_it = combos.find(combo_name);
        return combo_it != combos.end() && combo_it->second != 0;
    };

    // Effect pass overrides are the author-facing place where Wallpaper Engine stores switches such
    // as DIRECTDRAW. Keep that lookup inside the parsed effect model so object parsers can ask for a
    // semantic capability without knowing whether the switch came from the scene override or from
    // the resolved material data.
    for (const auto& pass : passes) {
        if (combo_is_enabled(pass.combos)) return true;
    }

    // Some packed workshop assets can bake combo state into the resolved material rather than the
    // scene-level pass override. Treat that as the same enabled combo so callers do not have to
    // duplicate the fallback search whenever they need to identify a shader feature.
    for (const auto& material : materials) {
        if (combo_is_enabled(material.combos)) return true;
    }

    return false;
}

std::unordered_set<std::string> WPImageEffect::FeedbackFboNames() const {
    // A normal transient effect FBO only needs to survive until its final reader in the same frame.
    // Feedback simulations are different: they sample one of their own FBOs before any pass has
    // written that FBO in the current frame, so the sampler intentionally reads last frame's state.
    // Cursor ripple is the canonical case: pass 0 reads `_rt_EightBuffer2`, pass 1 writes it back
    // after diffusion, and the next frame starts from that stored wave field. Detecting the
    // read-before-write contract from the authored pass order keeps this generic and avoids
    // hard-coding cursor-ripple resource names in the renderer.
    std::unordered_set<std::string> fbo_names;
    for (const auto& fbo : fbos) {
        if (! fbo.name.empty()) fbo_names.insert(fbo.name);
    }

    std::unordered_set<std::string> written_this_frame;
    std::unordered_set<std::string> feedback_fbos;

    auto read_fbo = [&](const std::string& name) {
        if (fbo_names.count(name) == 0) return;
        if (written_this_frame.count(name) == 0) feedback_fbos.insert(name);
    };

    auto write_fbo = [&](const std::string& name) {
        if (fbo_names.count(name) != 0) written_this_frame.insert(name);
    };

    auto apply_commands_at = [&](int32_t afterpos) {
        for (const auto& command : commands) {
            if (command.afterpos != afterpos) continue;
            read_fbo(command.source);
            write_fbo(command.target);
        }
    };

    for (std::size_t pass_index = 0; pass_index < passes.size(); pass_index++) {
        apply_commands_at(static_cast<int32_t>(pass_index));
        const auto& pass = passes[pass_index];
        for (const auto& binding : pass.bind) {
            read_fbo(binding.name);
        }
        write_fbo(pass.target);
    }
    apply_commands_at(static_cast<int32_t>(passes.size()));

    return feedback_fbos;
}

bool WPImageEffect::FromJson(const nlohmann::json& json, fs::VFS& vfs) {
    std::string filePath;
    GET_JSON_NAME_VALUE(json, "file", filePath);
    GET_JSON_NAME_VALUE_NOWARN(json, "visible", visible);
    if (json.contains("visible")) {
        visible_json             = json.at("visible");
        ReadVisibleBinding(visible_json, &visible_binding);
        // Object-shaped visibility carries the authored start value in
        // "value"; the plain NOWARN read above cannot extract a bool from an
        // object and silently leaves the default. Script-driven effects such
        // as media album covers author {script, value:false} and must start
        // hidden until the script shows them, exactly like WPImageObject's
        // ReadVisibleProperty contract.
        if (visible_json.is_object()) {
            visible = visible_binding.value;
        }
    }
    if(this->IsEffectBlacklisted(filePath)) {
        // Hide blacklisted effects at parse time so every object type consumes the same sanitized
        // shared effect representation. Log the exact resource path because a blacklist hit is
        // otherwise silent and looks identical to a missing or unsupported effect in render logs.
        LOG_INFO("SceneEffectBlacklist: file='%s' hidden-at-parse-time=true", filePath.c_str());
        visible = false;
        visible_json = nlohmann::json();
        visible_binding = {};
    }
	GET_JSON_NAME_VALUE_NOWARN(json, "id", id);
    nlohmann::json jEffect;
    if(!PARSE_JSON(fs::GetFileContent(vfs, "/assets/" + filePath), jEffect))
        return false;
    if(!FromFileJson(jEffect, vfs))
        return false;

    if(json.contains("passes")) {
        const auto& jPasses = json.at("passes");
        if(jPasses.size() > passes.size()) {
            LOG_ERROR("passes is not injective");
            return false;
        }
        int32_t i = 0;
        for(const auto& jP:jPasses) {
            WPMaterialPass pass;
            pass.FromJson(jP);
            passes[i++].Update(pass);
        }
    }
    return true;
}

bool WPImageEffect::FromFileJson(const nlohmann::json& json, fs::VFS& vfs) {
	GET_JSON_NAME_VALUE_NOWARN(json, "version", version);
    GET_JSON_NAME_VALUE(json, "name", name);
    if(json.contains("fbos")) {
        for(auto& jF:json.at("fbos")) {
            WPEffectFbo fbo;
            fbo.FromJson(jF);
            fbos.push_back(std::move(fbo));
        }
    }
    if(json.contains("passes")) {
        const auto& jEPasses = json.at("passes");
        bool compose {false};
        for(const auto& jP:jEPasses) {
            if(!jP.contains("material")) {
                if(jP.contains("command")) {
                    WPEffectCommand cmd;
                    cmd.FromJson(jP);
                    cmd.afterpos = passes.size();
                    commands.push_back(cmd);
                    continue;
                }
                LOG_ERROR("no material in effect pass");
                return false;
            }
            std::string matPath;
            GET_JSON_NAME_VALUE(jP, "material", matPath);
            nlohmann::json jMat;
            if(!PARSE_JSON(fs::GetFileContent(vfs, "/assets/" + matPath), jMat))
                return false;
            WPMaterial material;
            material.FromJson(jMat);
            materials.push_back(std::move(material));
            WPMaterialPass pass;
            pass.FromJson(jP);
            passes.push_back(std::move(pass));
            if(jP.contains("compose"))
	            GET_JSON_NAME_VALUE(jP, "compose", compose);
        }
        if(compose) {
            if(passes.size() != 2) {
                LOG_ERROR("effect compose option error");
                return false;
            }
            WPEffectFbo fbo; {fbo.name = "_rt_FullCompoBuffer1"; fbo.scale = 1;}
            fbos.push_back(fbo);
            passes.at(0).bind.push_back({ "previous", 0});
            passes.at(0).target = "_rt_FullCompoBuffer1";
            passes.at(1).bind.push_back({"_rt_FullCompoBuffer1", 0});
        }
    } else {
        LOG_ERROR("no passes in effect file");
        return false;
    }
    return true;
}
