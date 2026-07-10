#pragma once

namespace rayd::torch_backend {

enum DfrStrategyMask {
    RAYD_TORCH_DFR_DIRECT = 1 << 0,
    RAYD_TORCH_DFR_KELLER = 1 << 1,
    RAYD_TORCH_DFR_SUFFIX_REFL = 1 << 2
};

enum DfrSampleSequence {
    RAYD_TORCH_DFR_HASH = 0,
    RAYD_TORCH_DFR_SOBOL = 1
};

enum DfrReceiverModel {
    RAYD_TORCH_DFR_MATCHED_ISO = 0
};

} // namespace rayd::torch_backend
