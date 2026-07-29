// Copyright Xingyu Chen.
// Defines shared rt support for qualifiers.

#pragma once

// Backend-neutral device/host qualifier macros (RAY_TRACING_BACKEND_ARCHITECTURE
// .md §7). Promotes the ad-hoc `__CUDACC__` toggle that rayd/math.h wrote
// inline into a single shared spelling, so device traversal cores and the
// host-compilable algorithm bodies can share one convention:
//
//   RAYD_DEVICE       - device-only inline (host builds see plain `inline`).
//   RAYD_HOST_DEVICE  - host+device inline (host builds see plain `inline`).
//
// Host-safe: this header pulls in no CUDA/OptiX SDK header. Under a pure host
// compiler both macros collapse to `inline`, so an algorithm header that only
// spells RAYD_* qualifiers (never a raw `__device__`) still parses and compiles
// with the plain host toolchain. Under nvcc they expand to the CUDA forms and
// device code is generated exactly as before.

#if defined(__CUDACC__)
#define RAYD_DEVICE __device__ __forceinline__
#define RAYD_HOST_DEVICE __host__ __device__ __forceinline__
#else
#define RAYD_DEVICE inline
#define RAYD_HOST_DEVICE inline
#endif
