#pragma once

#include <cstdint>
#include <string_view>

#include <rayd/diffraction.h>
#include <rayd/penetration.h>
#include <rayd/reflection.h>
#include <rayd/scattering.h>
#include <rayd/scene.h>
#include <rayd/transmission.h>
#include <rayd/visibility.h>

namespace rayd::torch {

inline constexpr std::uint32_t kIntegrationApiVersion = 8;
inline constexpr std::string_view kIntegrationHeaderIdentity =
    "rayd.torch.integration";

} // namespace rayd::torch
