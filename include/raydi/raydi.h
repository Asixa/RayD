#pragma once

#include <stdexcept>
#include <string>
#include <string_view>

constexpr int RAYDI_NUM_CHANNELS = 3;

#include <raydi/constants.h>
#include <raydi/types.h>
#include <raydi/fwd.h>
#include <raydi/utils.h>

namespace raydi {

inline void require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

} // namespace raydi
