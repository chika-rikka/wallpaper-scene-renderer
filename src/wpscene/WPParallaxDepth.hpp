#pragma once

#include <array>

namespace wallpaper
{
namespace wpscene
{

// Wallpaper Engine omits `parallaxDepth` when a layer uses its 1.0 ctor default. Omitted, the
// default, and an explicit `"1 1"` are the same value; presence does not change inheritance.
inline constexpr std::array<float, 2> kDefaultParallaxDepth { 1.0f, 1.0f };

// Explicit authored zero. Fullscreen is a model/material flag, not this depth.
inline constexpr std::array<float, 2> kScreenSpaceParallaxDepth { 0.0f, 0.0f };

} // namespace wpscene
} // namespace wallpaper
