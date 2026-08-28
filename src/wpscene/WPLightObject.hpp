#pragma once
#include "WPParallaxDepth.hpp"
#include "WPJson.hpp"
#include "WPUserProperties.hpp"
#include <nlohmann/json.hpp>
#include <vector>
#include "WPPuppet.hpp"

namespace wallpaper
{
namespace fs
{
class VFS;
}

namespace wpscene
{

class WPLightObject {
public:
    bool                 FromJson(const nlohmann::json&, fs::VFS&);
    int32_t              id { 0 };
    std::string          name;
    std::array<float, 3> origin { 0.0f, 0.0f, 0.0f };
    std::array<float, 3> scale { 1.0f, 1.0f, 1.0f };
    std::array<float, 3> angles { 0.0f, 0.0f, 0.0f };
    std::array<float, 2> parallaxDepth { kDefaultParallaxDepth };
    std::array<float, 3> color { 1.0f, 1.0f, 1.0f };
    std::string          light;
    float                radius { 1000.0f };
    float                intensity { 1.0f };
    bool                 castvolumetrics { false };
    float                density { 1.0f };
    float                volumetricsexponent { 1.0f };
    float                innercone { 0.0f };
    float                outercone { 45.0f };
    bool                 usecookie { false };
    std::string          cookie;
    bool                 castshadows { false }; // scene.json key is `castshadow`
    float                exponent { 1.0f };
    float                cascadedistance0 { 25.0f };
    float                cascadedistance1 { 50.0f };
    float                cascadedistance2 { 200.0f };
    bool                 visible { true };
    VisibleBinding       visible_binding;
    int32_t              parent { 0 };
};

} // namespace wpscene
} // namespace wallpaper
  /*
          {
              "angles" : "0.00000 0.00000 0.00000",
              "color" : "1.00000 0.95686 0.87451",
              "id" : 237,
              "intensity" : 0.5,
              "light" : "point",
              "locktransforms" : false,
              "name" : "",
              "origin" : "611.30676 302.13736 2000.00000",
              "parallaxDepth" : "0.00000 0.00000",
              "radius" : 3000.0,
              "scale" : "1.00000 1.00000 1.00000",
              "visible" : true
          },
  */
