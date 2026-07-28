#pragma once

#include <cstdint>
#include <string_view>

#include <rayd/diffraction/torch.h>
#include <rayd/penetration/torch.h>
#include <rayd/reflection/torch.h>
#include <rayd/scattering/torch.h>
#include <rayd/scene/torch.h>
#include <rayd/transmission/torch.h>
#include <rayd/visibility/torch.h>

namespace rayd::torch {

inline constexpr std::uint32_t kIntegrationApiVersion = 7;
inline constexpr std::string_view kIntegrationHeaderIdentity =
    "rayd.torch.integration";

} // namespace rayd::torch
