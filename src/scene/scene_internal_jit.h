// Copyright Xingyu Chen.
// Declares internal scene support for scene internal Dr.Jit.

#pragma once

#include <rayd/jit/scene.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace rayd {

/// Element count of a count_a x count_b slot grid. Dr.Jit index arrays and the native kernels address these grids
/// with 32-bit signed ints, so a grid whose largest slot index does not fit throws here instead of wrapping into a
/// small or negative size and an out-of-bounds launch.
inline int checked_slot_count(int64_t count_a, int64_t count_b, std::string_view what) {
    constexpr int64_t kLimit = std::numeric_limits<int32_t>::max();
    if (count_a < 0 || count_b < 0) {
        throw std::runtime_error(std::string(what) + ": element counts must be non-negative (got " +
                                 std::to_string(count_a) + " x " + std::to_string(count_b) + ").");
    }
    if (count_a != 0 && count_b > kLimit / count_a) {
        throw std::runtime_error(std::string(what) + ": " + std::to_string(count_a) + " x " + std::to_string(count_b) +
                                 " elements exceed the 32-bit index range (" + std::to_string(kLimit) +
                                 "); split the batch into smaller launches.");
    }
    return static_cast<int>(count_a * count_b);
}

/// Batch width of a Dr.Jit array as an int; Dr.Jit arrays may hold up to 2^32 - 1 entries, which int cannot.
inline int checked_batch_width(size_t width, std::string_view what) {
    return checked_slot_count(static_cast<int64_t>(width), 1, what);
}

struct DfrDirectTapeCapture {
    int launch_count = 0;
    Mask active;
    Int state_idx;
    Int cell;
    Int material_idx;
    Float edge_u;
};

extern thread_local DfrDirectTapeCapture* active_dfr_direct_tape_capture;

/// Tell the user once per process why trace_backend='auto' resolved to the CUDA trace backend. Call it only after
/// optix_available() returned false; RAYD_DISABLE_OPTIX keeps it silent.
void warn_optix_auto_fallback_once();

void require_dfr_direct_custom_ad_supported(const DfrOptions& options);
void require_dfr_chain_custom_ad_supported(const DfrOptions& options);

DfrAccumAD dfr_direct_accum_custom_op(const Scene* scene, const DfrStatesAD& states, const DfrGrid& grid,
                                      const DfrMaterialAD& material, const DfrOptions& options,
                                      const Vector3fAD& suffix_tri_p0, const Vector3fAD& suffix_tri_face_normal,
                                      const Vector3fAD& suffix_vertices, const Vector3i& suffix_faces,
                                      const MaskAD& active);

DfrAccumAD dfr_chain_accum_custom_op(const Scene* scene, const DfrStatesAD& initial_states,
                                     const DfrStatesAD& recursive_states, const DfrGrid& grid,
                                     const DfrMaterialAD& material, const DfrOptions& options,
                                     const Vector3fAD& suffix_tri_p0, const Vector3fAD& suffix_tri_face_normal,
                                     const Vector3fAD& suffix_vertices, const Vector3i& suffix_faces,
                                     const MaskAD& active);

} // namespace rayd
