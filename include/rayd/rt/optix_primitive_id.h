// Copyright Xingyu Chen.
// Defines shared rt support for optix primitive id.

#pragma once

#ifdef __CUDACC__

namespace rayd::shared::optix {

static __forceinline__ __device__ int global_primitive_id(int shape_id, int local_primitive, const int* face_offsets,
                                                          int mesh_count) {
    const int face_offset = (shape_id >= 0 && shape_id < mesh_count) ? face_offsets[shape_id] : 0;
    return face_offset + local_primitive;
}

} // namespace rayd::shared::optix

#endif
