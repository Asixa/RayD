// Copyright Xingyu Chen.
// Declares internal scene support for scene internal Dr.Jit.

#pragma once

#include <rayd/jit/scene.h>

namespace rayd {

struct DfrDirectTapeCapture {
    int launch_count = 0;
    Mask active;
    Int state_idx;
    Int cell;
    Int material_idx;
    Float edge_u;
};

extern thread_local DfrDirectTapeCapture* active_dfr_direct_tape_capture;

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
