#pragma once

#include <limits>

namespace raydi {

constexpr float Epsilon = 1e-5f;
constexpr float RayEpsilon = 1e-3f;
constexpr float ShadowEpsilon = 1e-3f;
constexpr float Pi = 3.14159265358979323846f;
constexpr float Infinity = std::numeric_limits<float>::infinity();

} // namespace raydi
