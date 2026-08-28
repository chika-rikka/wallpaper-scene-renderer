#pragma once
#include "Core/Literals.hpp"
#include "Core/NoCopyMove.hpp"
#include "Core/MapSet.hpp"

#include <Eigen/Dense>

#include <functional>
#include <string_view>

namespace wallpaper
{
class SceneNode;
class SceneCamera;
class SceneShader;
class ShaderValue;
class SpriteAnimation;

using sprite_map_t    = Map<usize, SpriteAnimation>;
using UpdateUniformOp = std::function<void(std::string_view, ShaderValue)>;
using ExistsUniformOp = std::function<bool(std::string_view)>;

struct ShaderUniformOverrides {
    std::string_view camera_name;
    bool             use_camera_override { false };
    // Some source-seed passes still write into an offscreen effect target owned by the node's
    // camera, but their shader samples screen-space data. Those passes need uniforms evaluated
    // against the live active camera without permanently changing the node's authored camera.
    bool             use_active_camera_for_uniforms { false };
    bool             use_active_camera_for_parallax { false };
    // Official 0x1401ec799: I-internal writes identity into [engine+0x30].
    // Dest blit after I pop uses dest (0x1401e9dd5). This is the pass domain,
    // not a camera-name guess.
    bool             use_identity_model { false };
};

class IShaderValueUpdater : NoCopy, NoMove {
public:
    IShaderValueUpdater()          = default;
    virtual ~IShaderValueUpdater() = default;

    // Resource-affecting pose/surface preparation must finish before render-graph refresh. The
    // later FrameBegin hook remains the draw-phase boundary for uniform/cache consumers.
    virtual void PrepareFrame()                                                    = 0;
    virtual void FrameBegin()                                                      = 0;
    // Official frame draw 0x14018aac0 (FRAME_DEST_NO_RESET / COMPOSE_WRAP) hosts
    // Path B push/T+=/vt+0x50/pop. Not FrameBegin.
    virtual void ComposeDrawWalker()                                               = 0;
    virtual void InitUniforms(SceneNode*, const ExistsUniformOp&)                  = 0;
    virtual void UpdateUniforms(SceneNode*, sprite_map_t&, const UpdateUniformOp&,
                                const ShaderUniformOverrides* overrides = nullptr) = 0;
    virtual void FrameEnd()                                                        = 0;

    // Projection-driven resources must use the same inherited-parent, attachment, and camera
    // parallax transform contract as shader uniforms. Exposing that matrix here keeps sizing code
    // independent from the concrete Wallpaper Engine updater implementation.
    virtual Eigen::Matrix4d ResolveModelTransformForProjection(
        SceneNode* node, const SceneCamera* camera, bool apply_parallax) = 0;

    virtual void MouseInput(double x, double y) = 0;
    virtual void SetTexelSize(float x, float y) = 0;
    virtual void SetScreenSize(i32 w, i32 h)    = 0;
};
} // namespace wallpaper
