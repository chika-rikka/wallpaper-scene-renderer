#pragma once
#include "Particle.h"
#include "ParticleAudioResponse.h"

#include <vector>
#include <random>
#include <memory>
#include <array>
#include <span>
#include <cstdint>
#include <functional>

#include "Core/Literals.hpp"

namespace wallpaper
{

struct ParticleControlpoint {
    bool            link_mouse { false };
    bool            worldspace { false };
    Eigen::Vector3d offset { 0, 0, 0 };
    Eigen::Vector3d base_offset { 0, 0, 0 };
};

struct ParticleInfo {
    std::span<Particle>                   particles;
    std::span<const ParticleControlpoint> controlpoints;
    double                                time;
    double                                time_pass;
};

struct ParticleInitInfo {
    // Initializers that sequence around a control point need the live control-point snapshot and a
    // monotonic spawn counter. Operators keep using ParticleInfo.
    double                                duration { 0.0 };
    double                                time { 0.0 };
    std::span<const ParticleControlpoint> controlpoints;
    uint64_t                              sequence { 0 };
};

using ParticleInitOp = std::function<void(Particle&, const ParticleInitInfo&)>;
// particle index lifetime-percent passTime
using ParticleOperatorOp = std::function<void(const ParticleInfo&)>;

struct ParticleEmitterTiming {
    float emit_speed { 0.0f };
    bool  one_per_frame { false };
    bool  periodic { false };
    u32   instantaneous { 0 };
    u32   max_to_emit_per_period { 0 };
    float min_periodic_duration { 0.0f };
    float max_periodic_duration { 0.0f };
    float min_periodic_delay { 0.0f };
    float max_periodic_delay { 0.0f };
    float duration { 0.0f };
    float delay { 0.0f };
    ParticleAudioResponseFactor audio_rate_factor;
};

// Per particle-system instance. Recycle reloads instantaneous, delay, duration,
// clears credit, and zeroes the periodic timer/count. Shared emitter lambdas must not
// keep this state.
struct ParticleEmitRuntime {
    float    credit { 0.0f };
    u32      instantaneous { 0 };
    float    periodic_timer { 0.0f };
    u32      emitted_this_period { 0 };
    float    delay_remaining { 0.0f };
    float    duration_remaining { 0.0f };
    bool     duration_limited { false };
    bool     inactive { false };
};

inline ParticleEmitRuntime MakeParticleEmitRuntime(const ParticleEmitterTiming& timing) {
    ParticleEmitRuntime runtime;
    runtime.instantaneous      = timing.instantaneous;
    runtime.delay_remaining    = timing.delay;
    runtime.duration_remaining = timing.duration;
    runtime.duration_limited   = timing.duration > 0.0f;
    return runtime;
}

using ParticleEmittOp =
    std::function<void(std::vector<Particle>&, std::vector<ParticleInitOp>&,
                       std::span<const ParticleControlpoint>, uint32_t maxcount, double timepass,
                       double time, uint64_t& next_spawn_sequence, ParticleEmitRuntime& runtime)>;

struct ParticleBoxEmitterArgs {
    std::array<float, 3> directions;
    std::array<float, 3> minDistance;
    std::array<float, 3> maxDistance;
    std::array<float, 3> orgin;
    i32                  controlpoint { 0 };
    float                minSpeed;
    float                maxSpeed;
    ParticleEmitterTiming timing;

    static ParticleEmittOp MakeEmittOp(ParticleBoxEmitterArgs);
};

struct ParticleSphereEmitterArgs {
    std::array<float, 3>   directions;
    float                  minDistance;
    float                  maxDistance;
    std::array<float, 3>   orgin;
    i32                    controlpoint { 0 };
    std::array<int32_t, 3> sign;
    float                  minSpeed;
    float                  maxSpeed;
    ParticleEmitterTiming timing;

    static ParticleEmittOp MakeEmittOp(ParticleSphereEmitterArgs);
};

} // namespace wallpaper
