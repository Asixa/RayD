// Copyright Xingyu Chen.
// Declares internal diffraction support for common.

#pragma once

#include <rayd/diffraction/contracts.h>

namespace rayd::torch_backend {

enum DfrStrategyMask {
    RAYD_TORCH_DFR_DIRECT = static_cast<int>(shared::optix::DiffractionStrategyBit::Direct),
    RAYD_TORCH_DFR_KELLER = static_cast<int>(shared::optix::DiffractionStrategyBit::Keller),
    RAYD_TORCH_DFR_SUFFIX_REFL = static_cast<int>(shared::optix::DiffractionStrategyBit::SuffixReflection)
};

enum DfrSampleSequence {
    RAYD_TORCH_DFR_HASH = static_cast<int>(shared::optix::DiffractionSampleSequence::Hash),
    RAYD_TORCH_DFR_SOBOL = static_cast<int>(shared::optix::DiffractionSampleSequence::Sobol)
};

enum DfrReceiverModel {
    RAYD_TORCH_DFR_MATCHED_ISO = static_cast<int>(shared::optix::DiffractionReceiverModel::MatchedIsotropic)
};

} // namespace rayd::torch_backend
