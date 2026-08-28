#include "ParticleEmitter.h"
#include "ParticleModify.h"
#include "Utils/Algorism.h"
#include "Core/Random.hpp"

#include <Eigen/src/Core/Matrix.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <random>
#include <tuple>

using namespace wallpaper;

namespace
{

inline std::tuple<u32, bool> FindDeadParticle(std::span<const Particle> particles, u32 start) {
    for (u32 i = start; i < particles.size(); i++) {
        if (! ParticleModify::LifetimeOk(particles[i])) return { i, true };
    }
    return { 0, false };
}

inline float SampleInclusiveRange(float min_value, float max_value) {
    return min_value + Random::get(0.0f, 1.0f) * (max_value - min_value);
}

inline u32 ResolveEmitNum(ParticleEmitRuntime& runtime, const ParticleEmitterTiming& timing,
                          double time_pass) {
    if (runtime.inactive) return 0;

    const bool  valid_time = std::isfinite(time_pass);
    const float frame_time = valid_time ? static_cast<float>(time_pass) : 0.0f;

    // A frame which starts inside the authored delay never emits. Crossing zero only arms the
    // following frame.
    if (runtime.delay_remaining > 0.0f) {
        runtime.delay_remaining -= frame_time;
        return 0;
    }

    if (runtime.duration_limited && runtime.duration_remaining < 0.0f) {
        runtime.inactive = true;
        return 0;
    }

    const float factor =
        static_cast<float>(timing.audio_rate_factor ? timing.audio_rate_factor() : 1.0);
    float rate = timing.emit_speed * factor;

    if (timing.periodic) {
        if (runtime.periodic_timer > 0.0f) {
            runtime.periodic_timer -= frame_time;
            if (runtime.periodic_timer < 0.0f) {
                runtime.periodic_timer =
                    -SampleInclusiveRange(timing.min_periodic_delay, timing.max_periodic_delay);
            }
        } else {
            runtime.periodic_timer += frame_time;
            if (runtime.periodic_timer < 0.0f) {
                rate = 0.0f;
            } else {
                runtime.instantaneous        = timing.instantaneous;
                runtime.emitted_this_period  = 0;
                runtime.periodic_timer       = SampleInclusiveRange(
                    timing.min_periodic_duration, timing.max_periodic_duration);
            }
        }
    }

    const u32 instantaneous_count = runtime.instantaneous;
    runtime.instantaneous         = 0;

    if (std::isfinite(rate) && valid_time) runtime.credit += rate * frame_time;

    u32 continuous_count = 0;
    if (std::isfinite(runtime.credit) && runtime.credit >= 1.0f) {
        const float integral_credit = std::floor(runtime.credit);
        continuous_count           = integral_credit > 0.0f
            ? static_cast<u32>(std::min(
                  integral_credit, static_cast<float>(std::numeric_limits<i32>::max())))
            : 0;

        runtime.credit -=
            static_cast<float>(continuous_count) + static_cast<float>(instantaneous_count);
        if (timing.one_per_frame) continuous_count = std::min<u32>(continuous_count, 1);

        if (timing.periodic && timing.max_to_emit_per_period != 0) {
            const i32 remaining = static_cast<i32>(timing.max_to_emit_per_period) -
                                  static_cast<i32>(runtime.emitted_this_period);
            const u32 cap = remaining > 0 ? static_cast<u32>(remaining) : 0;
            if (continuous_count > cap) continuous_count = cap;
            runtime.emitted_this_period += continuous_count;
        }
    } else {
        runtime.credit -= static_cast<float>(instantaneous_count);
    }

    const u32 count = continuous_count > std::numeric_limits<u32>::max() - instantaneous_count
        ? std::numeric_limits<u32>::max()
        : continuous_count + instantaneous_count;

    // Duration is tested at frame start and consumed after this frame's emission. Once it expires,
    // the whole emitter stays inactive, including future periodic instantaneous bursts.
    if (runtime.duration_limited && valid_time) {
        runtime.duration_remaining -= frame_time;
        if (runtime.duration_remaining < 0.0f) runtime.inactive = true;
    }
    return count;
}

inline bool HasParticleCapacity(std::span<const Particle> particles, u32 maxcount) {
    if (particles.size() < maxcount) return true;
    return std::any_of(particles.begin(), particles.end(), [](const Particle& particle) {
        return ! ParticleModify::LifetimeOk(particle);
    });
}

template<typename SpawnOp>
void EmitParticles(std::vector<Particle>& particles, u32 num, u32 maxcount,
                   uint64_t& next_spawn_sequence, SpawnOp&& spawn_particle) {
    u32  next_search_index = 0;
    bool has_dead          = true;

    for (u32 i = 0; i < num; i++) {
        if (has_dead) {
            auto [dead_index, found] = FindDeadParticle(particles, next_search_index);
            next_search_index        = dead_index;
            has_dead                 = found;
        }
        if (! has_dead && maxcount == particles.size()) break;

        Particle spawned      = spawn_particle();
        spawned.spawnSequence = next_spawn_sequence++;
        if (has_dead) {
            particles[next_search_index] = std::move(spawned);
        } else {
            particles.push_back(std::move(spawned));
        }
    }
}

template<typename GenerateOp>
Particle SpawnParticle(GenerateOp&& generate, std::vector<ParticleInitOp>& initializers,
                       const ParticleInitInfo& info) {
    const float    unit = Random::get(0.0f, 1.0f);
    auto           particle = generate();
    particle.remap_seed     = std::bit_cast<uint32_t>(unit);
    for (auto& initializer : initializers) initializer(particle, info);
    return particle;
}

inline void ApplySign(Eigen::Vector3d& p, int32_t x, int32_t y, int32_t z) noexcept {
    if (x != 0) {
        p.x() = std::abs(p.x()) * (float)x;
    }
    if (y != 0) {
        p.y() = std::abs(p.y()) * (float)y;
    }
    if (z != 0) {
        p.z() = std::abs(p.z()) * (float)z;
    }
}
} // namespace

ParticleEmittOp ParticleBoxEmitterArgs::MakeEmittOp(ParticleBoxEmitterArgs a) {
    // PARTICLE_SEQ_RESET: phase is packed [system+0x80]+8, not runtime.sequence.
    uint64_t sequence { 0 };
    return [a, sequence](std::vector<Particle>&       ps,
                         std::vector<ParticleInitOp>& inis,
                         std::span<const ParticleControlpoint> controlpoints,
                         u32                          maxcount,
                         double                       timepass,
                         double                       time,
                         uint64_t&                    next_spawn_sequence,
                         ParticleEmitRuntime&         runtime) mutable {
        auto GenBox = [&]() {
            Eigen::Vector3d pos;
            for (int32_t i = 0; i < 3; i++)
                pos[i] = algorism::lerp(Random::get(-1.0, 1.0), a.minDistance[i], a.maxDistance[i]);
            auto p = Particle();
            pos    = pos.cwiseProduct(Eigen::Vector3f { a.directions.data() }.cast<double>());
            ParticleModify::MoveTo(p, pos);
            ParticleModify::ChangeVelocity(p,
                                           Random::get(a.minSpeed, a.maxSpeed) * pos.normalized());

            Eigen::Vector3d origin = Eigen::Vector3f { a.orgin.data() }.cast<double>();
            if (a.controlpoint >= 0 && (usize)a.controlpoint < controlpoints.size()) {
                origin += controlpoints[(usize)a.controlpoint].offset;
            }
            ParticleModify::Move(p, origin);
            return p;
        };
        if (! HasParticleCapacity(ps, maxcount)) return;
        const u32 emission_count = ResolveEmitNum(runtime, a.timing, timepass);
        EmitParticles(ps, emission_count, maxcount, next_spawn_sequence, [&]() {
            ParticleInitInfo init_info;
            init_info.duration      = a.timing.emit_speed > 0.0f ? 1.0 / a.timing.emit_speed : 0.0;
            init_info.time          = time;
            init_info.controlpoints = controlpoints;
            init_info.sequence      = sequence++;
            return SpawnParticle(GenBox, inis, init_info);
        });
    };
}

ParticleEmittOp ParticleSphereEmitterArgs::MakeEmittOp(ParticleSphereEmitterArgs a) {
    using namespace Eigen;
    uint64_t sequence { 0 };
    return [a, sequence](std::vector<Particle>&       ps,
                         std::vector<ParticleInitOp>& inis,
                         std::span<const ParticleControlpoint> controlpoints,
                         u32                          maxcount,
                         double                       timepass,
                         double                       time,
                         uint64_t&                    next_spawn_sequence,
                         ParticleEmitRuntime&         runtime) mutable {
        auto GenSphere = [&]() {
            auto   p = Particle();
            double r = algorism::lerp(
                std::pow(Random::get(0.0, 1.0), 1.0 / 3.0), a.minDistance, a.maxDistance);
            Eigen::Vector3d sp = r * algorism::GenSphereSurfaceNormal(
                                         [](double u, double o) {
                                             return Random::get<std::normal_distribution<>>(u, o);
                                         },
                                         Eigen::Vector3f { a.directions.data() }.cast<double>());
            ApplySign(sp, a.sign[0], a.sign[1], a.sign[2]);

            ParticleModify::MoveTo(p, sp);
            ParticleModify::ChangeVelocity(p,
                                           Random::get(a.minSpeed, a.maxSpeed) * sp.normalized());

            Eigen::Vector3d origin = Eigen::Vector3f { a.orgin.data() }.cast<double>();
            if (a.controlpoint >= 0 && (usize)a.controlpoint < controlpoints.size()) {
                origin += controlpoints[(usize)a.controlpoint].offset;
            }
            ParticleModify::Move(p, origin);
            return p;
        };
        if (! HasParticleCapacity(ps, maxcount)) return;
        const u32 emission_count = ResolveEmitNum(runtime, a.timing, timepass);
        EmitParticles(ps, emission_count, maxcount, next_spawn_sequence, [&]() {
            ParticleInitInfo init_info;
            init_info.duration      = a.timing.emit_speed > 0.0f ? 1.0 / a.timing.emit_speed : 0.0;
            init_info.time          = time;
            init_info.controlpoints = controlpoints;
            init_info.sequence      = sequence++;
            return SpawnParticle(GenSphere, inis, init_info);
        });
    };
}
