#pragma once

#include <rayd/shared/reflection/epc_params.h>

namespace rayd {

using ReflEpcParams = shared::optix::ReflEpcParams;
inline constexpr int ReflEpcMaxBounces = shared::optix::ReflEpcMaxBounces;
inline constexpr int ReflEpcVisibilityIgnorePrimitive =
    shared::optix::ReflEpcVisibilityIgnorePrimitive;
inline constexpr int ReflEpcVisibilityIgnoreSurfaceGroup =
    shared::optix::ReflEpcVisibilityIgnoreSurfaceGroup;

} // namespace rayd
