// Copyright Xingyu Chen.
// Defines shared rt support for optix sbt.

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace rayd::shared::optix {

// OptiX fixes these values as part of the shader binding table ABI. Keeping
// them here lets both backends share record layouts without including either
// backend's OptiX host declarations.
inline constexpr std::size_t SbtRecordAlignment = 16u;
inline constexpr std::size_t SbtRecordHeaderSize = 32u;

template <typename T> struct alignas(SbtRecordAlignment) SbtRecord {
    std::byte header[SbtRecordHeaderSize];
    T data;
};

struct alignas(SbtRecordAlignment) EmptySbtRecord {
    std::byte header[SbtRecordHeaderSize];
};

static_assert(std::is_standard_layout_v<EmptySbtRecord>);
static_assert(std::is_trivially_copyable_v<EmptySbtRecord>);
static_assert(alignof(EmptySbtRecord) == SbtRecordAlignment);
static_assert(sizeof(EmptySbtRecord) == SbtRecordHeaderSize);
static_assert(offsetof(SbtRecord<std::uint32_t>, data) == SbtRecordHeaderSize);

} // namespace rayd::shared::optix