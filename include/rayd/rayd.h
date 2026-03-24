#pragma once

#include <stdexcept>
#include <string>
#include <string_view>

constexpr int RAYD_NUM_CHANNELS = 3;

#include <rayd/constants.h>
#include <rayd/types.h>
#include <rayd/fwd.h>
#include <rayd/utils.h>

namespace rayd {

inline void require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

} // namespace rayd
