# Multi-Bounce Reflection Trace — OptiX Raygen Program Design

## Overview

Add a `scene.trace_reflections()` API to RayDi that performs specular multi-bounce
ray tracing in a **single GPU dispatch** via a custom OptiX raygen program.  The
entire bounce loop runs on-device — zero Python round-trips, zero CPU-GPU sync
between bounces.

### Why

The current single-query `scene.intersect()` requires one DrJit kernel launch per
bounce.  For 3-bounce reflection tracing this means 3 OptiX launches plus ~50
intermediate DrJit operations per bounce orchestrated from Python.  Profiling shows
this takes ~90 ms in witwin.channel even for simple scenes.  A fused raygen program
does the same work in ~3-5 ms.

---

## API

### C++ (scene.h)

```cpp
static constexpr int kMaxReflectionBounces = 8;

/// Output of trace_reflections().  All flat arrays with stride = max_bounces.
/// Element [ray_index * max_bounces + bounce] for per-bounce fields.
struct ReflectionChain {
    int max_bounces = 0;
    int ray_count   = 0;

    IntDetached      bounce_count;    // [N]       — actual bounces per ray (0..B)
    IntDetached      prim_ids;        // [N * B]   — global triangle id (-1 if no hit)
    IntDetached      shape_ids;       // [N * B]   — mesh id
    Vector3fDetached hit_points;      // [N * B]   — world hit position
    Vector3fDetached geo_normals;     // [N * B]   — geometric face normal (unnormalized ok)
    Vector3fDetached image_sources;   // [N * B]   — cumulative image source after this bounce
    Vector3fDetached plane_points;    // [N * B]   — reflection plane point (= hit_point)
    Vector3fDetached plane_normals;   // [N * B]   — reflection plane normal (= geo_normal)
};

// On Scene class:
ReflectionChain trace_reflections(const RayDetached &ray,
                                  int max_bounces,
                                  MaskDetached active = true) const;
```

Current API note:

- `prim_ids` is mesh-local for compatibility.
- `local_prim_ids` is identical to `prim_ids`.
- `global_prim_ids` is the scene-global triangle-id chain.

### Python (nanobind)

```python
chain = scene.trace_reflections(ray, max_bounces=3)
chain.bounce_count   # drjit.cuda.Int    [N]
chain.prim_ids       # drjit.cuda.Int    [N*B]
chain.global_prim_ids# drjit.cuda.Int    [N*B]
chain.hit_points     # drjit.cuda.Array3f [N*B]
chain.image_sources  # drjit.cuda.Array3f [N*B]
# ...
```

### PyTorch wrapper (rayd.torch)

```python
chain = scene.trace_reflections(ray, max_bounces=3)
chain.bounce_count   # torch.Tensor int32   [N]
chain.prim_ids       # torch.Tensor int32   [N, B]
chain.global_prim_ids# torch.Tensor int32   [N, B]
chain.hit_points     # torch.Tensor float32 [N, B, 3]
chain.image_sources  # torch.Tensor float32 [N, B, 3]
# ...
```

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Python call                        │
│  chain = scene.trace_reflections(ray, max_bounces)   │
└────────────────────┬────────────────────────────────┘
                     │  one C++ call
                     ▼
┌─────────────────────────────────────────────────────┐
│               Scene::trace_reflections()             │
│  1. Allocate GPU output buffers                      │
│  2. Set launch params (handle, buffers, ray data)    │
│  3. optixLaunch() — single dispatch, N threads       │
│  4. Wrap output buffers as DrJit arrays              │
└────────────────────┬────────────────────────────────┘
                     │  one GPU kernel
                     ▼
┌─────────────────────────────────────────────────────┐
│         __raygen__reflection_trace (GPU)              │
│  per thread (one ray):                               │
│    for b in 0..max_bounces:                          │
│      optixTrace() → __closesthit__reflection         │
│      if miss: break                                  │
│      read payload (t, prim_id, bary_u, bary_v)       │
│      reconstruct hit_point from triangle_info         │
│      compute geo_normal from triangle_info            │
│      reflect image_source across hit plane            │
│      write output[ray * B + b]                        │
│      reflect ray direction, advance origin             │
│    write bounce_count[ray]                            │
└─────────────────────────────────────────────────────┘
```

---

## New Files

```
include/rayd/
    reflection_chain.h          — ReflectionChain struct definition

src/scene/
    reflection_trace.cu         — OptiX programs (__raygen__, __closesthit__, __miss__)
    reflection_trace_host.cpp   — Host-side pipeline build + launch
    reflection_trace_host.h     — Internal header

src/rayd.cpp                    — Add nanobind bindings (modify)
rayd/torch/scene.py             — Add PyTorch wrapper  (modify)
CMakeLists.txt                  — Add PTX compilation   (modify)
```

---

## CUDA / OptiX Programs

### reflection_trace.cu

This file compiles to PTX.  It uses the real OptiX SDK headers (provided by the
CUDA toolkit or OptiX SDK install).

```cuda
// src/scene/reflection_trace.cu
//
// Compile: nvcc -ptx -O3 --use_fast_math -std=c++17
//          -I<optix_include> -I<rayd_include>
//          -arch=compute_70   (or appropriate SM)
//          reflection_trace.cu -o reflection_trace.ptx

#include <optix.h>
#include <optix_device.h>

// ---------------------------------------------------------------
//  Launch parameters — set on host before optixLaunch
// ---------------------------------------------------------------

struct TriangleData {
    float3 *p0;        // [T] vertex 0
    float3 *e1;        // [T] edge 1 (v1 - v0)
    float3 *e2;        // [T] edge 2 (v2 - v0)
    float3 *face_n;    // [T] face normal (unnormalized cross(e1, e2))
};

struct ReflectionTraceParams {
    OptixTraversableHandle handle;

    // Per-mesh SBT data is not needed — we read from triangle_info directly.
    // The global prim_id is recovered from the SBT hit-group record.

    // Triangle geometry (scene-global, indexed by global_prim_id)
    TriangleData tri;
    int          n_triangles;

    // Per-mesh face offsets for global prim_id computation
    int         *face_offsets;     // [n_meshes]
    int          n_meshes;

    // Input rays
    float3      *ray_origins;      // [N]
    float3      *ray_directions;   // [N]
    int          n_rays;
    int          max_bounces;

    // Output buffers (all device pointers)
    int         *out_bounce_count; // [N]
    int         *out_prim_ids;     // [N * B]
    int         *out_shape_ids;    // [N * B]
    float3      *out_hit_points;   // [N * B]
    float3      *out_geo_normals;  // [N * B]
    float3      *out_image_sources;// [N * B]
    float3      *out_plane_points; // [N * B]
    float3      *out_plane_normals;// [N * B]
};

extern "C" {
    __constant__ ReflectionTraceParams params;
}


// ---------------------------------------------------------------
//  Payload: closesthit → raygen communication (7 registers)
// ---------------------------------------------------------------
//  p0: hit flag      (0 or 1)
//  p1: t             (float bits)
//  p2: bary_u        (float bits)
//  p3: bary_v        (float bits)
//  p4: prim_index    (local to GAS)
//  p5: sbt_offset    (to recover shape_id and face_offset)
//  p6: (reserved)

static __forceinline__ __device__
void set_payload(unsigned int p0, unsigned int p1, unsigned int p2,
                 unsigned int p3, unsigned int p4, unsigned int p5)
{
    optixSetPayload_0(p0);
    optixSetPayload_1(p1);
    optixSetPayload_2(p2);
    optixSetPayload_3(p3);
    optixSetPayload_4(p4);
    optixSetPayload_5(p5);
}


// ---------------------------------------------------------------
//  Closest-hit program
// ---------------------------------------------------------------

extern "C" __global__ void __closesthit__reflection()
{
    const float t = optixGetRayTmax();
    const float2 bary = optixGetTriangleBarycentrics();
    const unsigned int prim_idx = optixGetPrimitiveIndex();
    const unsigned int sbt_idx = optixGetSbtGASIndex();

    set_payload(
        1u,
        __float_as_uint(t),
        __float_as_uint(bary.x),
        __float_as_uint(bary.y),
        prim_idx,
        sbt_idx
    );
}


// ---------------------------------------------------------------
//  Miss program
// ---------------------------------------------------------------

extern "C" __global__ void __miss__reflection()
{
    optixSetPayload_0(0u);  // miss flag
}


// ---------------------------------------------------------------
//  Helper math
// ---------------------------------------------------------------

static __forceinline__ __device__
float3 operator+(float3 a, float3 b) { return make_float3(a.x+b.x, a.y+b.y, a.z+b.z); }

static __forceinline__ __device__
float3 operator-(float3 a, float3 b) { return make_float3(a.x-b.x, a.y-b.y, a.z-b.z); }

static __forceinline__ __device__
float3 operator*(float s, float3 v) { return make_float3(s*v.x, s*v.y, s*v.z); }

static __forceinline__ __device__
float dot3(float3 a, float3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }

static __forceinline__ __device__
float3 cross3(float3 a, float3 b) {
    return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

static __forceinline__ __device__
float3 normalize3(float3 v) {
    float inv_len = rsqrtf(fmaxf(dot3(v, v), 1e-12f));
    return inv_len * v;
}


// ---------------------------------------------------------------
//  Raygen program — the core multi-bounce loop
// ---------------------------------------------------------------

extern "C" __global__ void __raygen__reflection_trace()
{
    const uint3 launch_idx = optixGetLaunchIndex();
    const int ray_idx = static_cast<int>(launch_idx.x);
    if (ray_idx >= params.n_rays) return;

    const int B = params.max_bounces;
    const int base = ray_idx * B;

    float3 origin = params.ray_origins[ray_idx];
    float3 dir    = params.ray_directions[ray_idx];
    float3 img_src = origin;  // cumulative image source (starts at TX)

    int bounce_count = 0;

    for (int b = 0; b < B; ++b) {
        // ---- Trace one bounce via OptiX ----
        unsigned int p0 = 0, p1, p2, p3, p4, p5;

        optixTrace(
            params.handle,
            origin,                           // ray origin
            dir,                              // ray direction
            1e-5f,                            // tmin (small bias)
            1e8f,                             // tmax
            0.0f,                             // ray time
            255u,                             // visibility mask
            OPTIX_RAY_FLAG_DISABLE_ANYHIT,    // ray flags
            0,                                // SBT offset
            1,                                // SBT stride
            0,                                // miss SBT index
            p0, p1, p2, p3, p4, p5           // payload
        );

        // ---- Check hit ----
        if (p0 == 0u) break;  // miss — stop bouncing

        // ---- Decode payload ----
        const float t      = __uint_as_float(p1);
        const float bary_u = __uint_as_float(p2);
        const float bary_v = __uint_as_float(p3);
        const int   local_prim = static_cast<int>(p4);
        const int   sbt_idx    = static_cast<int>(p5);

        // Recover global prim_id and shape_id from SBT index.
        // SBT index corresponds to mesh index in the scene.
        const int shape_id = sbt_idx;
        const int face_offset = (sbt_idx < params.n_meshes)
                                    ? params.face_offsets[sbt_idx]
                                    : 0;
        const int global_prim = local_prim + face_offset;

        // ---- Reconstruct hit geometry from triangle_info ----
        float3 hit_p, geo_n;

        if (global_prim >= 0 && global_prim < params.n_triangles) {
            const float3 p0_vtx = params.tri.p0[global_prim];
            const float3 e1     = params.tri.e1[global_prim];
            const float3 e2     = params.tri.e2[global_prim];

            hit_p = p0_vtx + bary_u * e1 + bary_v * e2;
            geo_n = normalize3(params.tri.face_n[global_prim]);
        } else {
            hit_p = origin + t * dir;
            geo_n = make_float3(0.f, 0.f, 1.f);
        }

        // Ensure normal faces against incoming ray
        if (dot3(dir, geo_n) > 0.f) geo_n = -1.f * geo_n;

        // ---- Reflect image source across hit plane ----
        float d = dot3(img_src - hit_p, geo_n);
        img_src = img_src - 2.0f * d * geo_n;

        // ---- Write output ----
        const int slot = base + b;
        params.out_prim_ids[slot]      = global_prim;
        params.out_shape_ids[slot]     = shape_id;
        params.out_hit_points[slot]    = hit_p;
        params.out_geo_normals[slot]   = geo_n;
        params.out_image_sources[slot] = img_src;
        params.out_plane_points[slot]  = hit_p;
        params.out_plane_normals[slot] = geo_n;

        // ---- Reflect ray for next bounce ----
        float dot_dn = dot3(dir, geo_n);
        dir    = dir - 2.0f * dot_dn * geo_n;
        origin = hit_p + 1e-5f * dir;   // bias to avoid self-intersection
        bounce_count = b + 1;
    }

    params.out_bounce_count[ray_idx] = bounce_count;
}
```

---

## Host-Side Pipeline & Launch

### reflection_trace_host.h

```cpp
// src/scene/reflection_trace_host.h
#pragma once

#include <rayd/rayd.h>
#include <rayd/optix.h>

namespace rayd {

struct TriangleInfoDetached;

/// Manages a separate OptiX pipeline for multi-bounce reflection tracing.
class ReflectionTracePipeline {
public:
    ReflectionTracePipeline();
    ~ReflectionTracePipeline();

    /// Build the pipeline (call once after OptiX context is initialized).
    void build(OptixDeviceContext context);
    bool is_ready() const { return ready_; }

    /// Launch the multi-bounce trace.
    /// - ias_handle: scene IAS traversable
    /// - tri: scene-global triangle info (detached)
    /// - face_offsets: per-mesh face offset buffer (device ptr)
    /// - n_meshes: number of meshes
    /// - ray_o, ray_d: ray arrays (device ptrs, float3 SOA)
    /// - n_rays, max_bounces: launch dimensions
    /// - output buffers: all device ptrs, pre-allocated
    void launch(
        OptixTraversableHandle ias_handle,
        const TriangleInfoDetached &tri,
        const void *face_offsets_ptr,
        int n_meshes,
        int n_triangles,
        const void *ray_o_ptr,
        const void *ray_d_ptr,
        int n_rays,
        int max_bounces,
        // output device pointers
        void *out_bounce_count,
        void *out_prim_ids,
        void *out_shape_ids,
        void *out_hit_points,
        void *out_geo_normals,
        void *out_image_sources,
        void *out_plane_points,
        void *out_plane_normals
    );

private:
    bool ready_ = false;
    OptixModule module_ = nullptr;
    OptixProgramGroup pg_raygen_ = nullptr;
    OptixProgramGroup pg_miss_ = nullptr;
    OptixProgramGroup pg_hitgroup_ = nullptr;
    OptixPipeline pipeline_ = nullptr;
    void *sbt_raygen_record_ = nullptr;
    void *sbt_miss_record_ = nullptr;
    void *sbt_hitgroup_record_ = nullptr;
    void *params_buffer_ = nullptr;
};

}  // namespace rayd
```

### reflection_trace_host.cpp

```cpp
// src/scene/reflection_trace_host.cpp
#include "reflection_trace_host.h"

#include <cstring>
#include <stdexcept>
#include <rayd/mesh.h>

// PTX is embedded at compile time (see CMake).
// The build system generates: extern const char reflection_trace_ptx[];
//                             extern size_t   reflection_trace_ptx_size;
#include "reflection_trace_ptx.h"

namespace rayd {

namespace {

// Minimal SBT record (header only — no per-record data for raygen/miss).
struct EmptySbtRecord {
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

// Hitgroup SBT record (must match existing scene's per-mesh SBT layout
// so that SBT index == mesh index).
struct HitGroupSbtRecord {
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

void check_optix(OptixResult result, const char *msg) {
    if (result != 0)
        throw std::runtime_error(std::string("OptiX error in ") + msg);
}

}  // namespace

ReflectionTracePipeline::ReflectionTracePipeline() = default;

ReflectionTracePipeline::~ReflectionTracePipeline() {
    // Cleanup via OptiX API (optixPipelineDestroy, optixModuleDestroy, etc.)
    // and jit_free for device buffers.
    if (params_buffer_)       jit_free(params_buffer_);
    if (sbt_raygen_record_)   jit_free(sbt_raygen_record_);
    if (sbt_miss_record_)     jit_free(sbt_miss_record_);
    if (sbt_hitgroup_record_) jit_free(sbt_hitgroup_record_);
    // NOTE: pipeline, module, program group destruction requires
    //       optixPipelineDestroy etc. looked up via jit_optix_lookup.
}

void ReflectionTracePipeline::build(OptixDeviceContext context) {
    // ---- 1. Create module from embedded PTX ----
    OptixModuleCompileOptions module_options = {};
    module_options.maxRegisterCount = 0;  // let compiler decide
    module_options.optLevel  = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
    module_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    OptixPipelineCompileOptions pipeline_options = {};
    pipeline_options.usesMotionBlur = 0;
    pipeline_options.traversableGraphFlags =
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    pipeline_options.numPayloadValues = 6;      // p0..p5
    pipeline_options.numAttributeValues = 2;    // bary u,v (built-in triangle)
    pipeline_options.pipelineLaunchParamsVariableName = "params";

    auto optixModuleCreate_ = (decltype(&optixModuleCreate))
        jit_optix_lookup("optixModuleCreate");

    char log[2048]; size_t log_size = sizeof(log);
    check_optix(
        optixModuleCreate_(context, &module_options, &pipeline_options,
                           reflection_trace_ptx, reflection_trace_ptx_size,
                           log, &log_size, &module_),
        "optixModuleCreate");

    // ---- 2. Create program groups ----
    auto optixProgramGroupCreate_ = (decltype(&optixProgramGroupCreate))
        jit_optix_lookup("optixProgramGroupCreate");

    // Raygen
    OptixProgramGroupDesc pg_raygen_desc = {};
    pg_raygen_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pg_raygen_desc.raygen.module = module_;
    pg_raygen_desc.raygen.entryFunctionName = "__raygen__reflection_trace";

    log_size = sizeof(log);
    OptixProgramGroupOptions pg_options = {};
    check_optix(
        optixProgramGroupCreate_(context, &pg_raygen_desc, 1, &pg_options,
                                 log, &log_size, &pg_raygen_),
        "raygen program group");

    // Miss
    OptixProgramGroupDesc pg_miss_desc = {};
    pg_miss_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    pg_miss_desc.miss.module = module_;
    pg_miss_desc.miss.entryFunctionName = "__miss__reflection";

    log_size = sizeof(log);
    check_optix(
        optixProgramGroupCreate_(context, &pg_miss_desc, 1, &pg_options,
                                 log, &log_size, &pg_miss_),
        "miss program group");

    // Hitgroup (closesthit)
    OptixProgramGroupDesc pg_hit_desc = {};
    pg_hit_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pg_hit_desc.hitgroup.moduleCH = module_;
    pg_hit_desc.hitgroup.entryFunctionNameCH = "__closesthit__reflection";

    log_size = sizeof(log);
    check_optix(
        optixProgramGroupCreate_(context, &pg_hit_desc, 1, &pg_options,
                                 log, &log_size, &pg_hitgroup_),
        "hitgroup program group");

    // ---- 3. Create pipeline ----
    auto optixPipelineCreate_ = (decltype(&optixPipelineCreate))
        jit_optix_lookup("optixPipelineCreate");

    OptixProgramGroup groups[] = { pg_raygen_, pg_miss_, pg_hitgroup_ };
    OptixPipelineLinkOptions link_options = {};
    link_options.maxTraceDepth = 1;  // single-level trace per optixTrace call

    log_size = sizeof(log);
    check_optix(
        optixPipelineCreate_(context, &pipeline_options, &link_options,
                             groups, 3, log, &log_size, &pipeline_),
        "optixPipelineCreate");

    // Set stack sizes
    auto optixPipelineSetStackSize_ = (decltype(&optixPipelineSetStackSize))
        jit_optix_lookup("optixPipelineSetStackSize");
    optixPipelineSetStackSize_(pipeline_,
                               2048,   // direct callable stack
                               2048,   // continuation callable stack
                               2048,   // traversal stack
                               2);     // max traversal depth (IAS → GAS)

    // ---- 4. Build SBT ----
    auto optixSbtRecordPackHeader_ = (decltype(&optixSbtRecordPackHeader))
        jit_optix_lookup("optixSbtRecordPackHeader");

    // Raygen record
    EmptySbtRecord raygen_record = {};
    optixSbtRecordPackHeader_(pg_raygen_, &raygen_record);
    sbt_raygen_record_ = jit_malloc(AllocType::Device, sizeof(raygen_record));
    jit_memcpy(JitBackend::CUDA, sbt_raygen_record_, &raygen_record,
               sizeof(raygen_record));

    // Miss record
    EmptySbtRecord miss_record = {};
    optixSbtRecordPackHeader_(pg_miss_, &miss_record);
    sbt_miss_record_ = jit_malloc(AllocType::Device, sizeof(miss_record));
    jit_memcpy(JitBackend::CUDA, sbt_miss_record_, &miss_record,
               sizeof(miss_record));

    // Hitgroup record — one per mesh is needed for correct SBT indexing.
    // For now allocate a single record; Scene::trace_reflections() will
    // rebuild the hitgroup SBT when mesh count changes.
    HitGroupSbtRecord hitgroup_record = {};
    optixSbtRecordPackHeader_(pg_hitgroup_, &hitgroup_record);
    sbt_hitgroup_record_ = jit_malloc(AllocType::Device, sizeof(hitgroup_record));
    jit_memcpy(JitBackend::CUDA, sbt_hitgroup_record_, &hitgroup_record,
               sizeof(hitgroup_record));

    // Params buffer
    params_buffer_ = jit_malloc(AllocType::Device, sizeof(ReflectionTraceParams));

    ready_ = true;
}

void ReflectionTracePipeline::launch(
    OptixTraversableHandle ias_handle,
    const TriangleInfoDetached &tri,
    const void *face_offsets_ptr,
    int n_meshes,
    int n_triangles,
    const void *ray_o_ptr,
    const void *ray_d_ptr,
    int n_rays,
    int max_bounces,
    void *out_bounce_count,
    void *out_prim_ids,
    void *out_shape_ids,
    void *out_hit_points,
    void *out_geo_normals,
    void *out_image_sources,
    void *out_plane_points,
    void *out_plane_normals)
{
    if (!ready_)
        throw std::runtime_error("ReflectionTracePipeline not built.");

    // NOTE: tri.p0, tri.e1, tri.e2, tri.face_normal are Vector3fDetached.
    // Extract device pointers via drjit::data_ptr<FloatDetached>(array).
    // These point to interleaved float SOA managed by DrJit.

    // Assemble host-side params struct
    ReflectionTraceParams host_params = {};
    host_params.handle       = ias_handle;
    host_params.tri.p0       = (float3 *)drjit::data_ptr(tri.p0);
    host_params.tri.e1       = (float3 *)drjit::data_ptr(tri.e1);
    host_params.tri.e2       = (float3 *)drjit::data_ptr(tri.e2);
    host_params.tri.face_n   = (float3 *)drjit::data_ptr(tri.face_normal);
    host_params.n_triangles  = n_triangles;
    host_params.face_offsets = (int *)face_offsets_ptr;
    host_params.n_meshes     = n_meshes;
    host_params.ray_origins    = (float3 *)ray_o_ptr;
    host_params.ray_directions = (float3 *)ray_d_ptr;
    host_params.n_rays       = n_rays;
    host_params.max_bounces  = max_bounces;
    host_params.out_bounce_count  = (int *)out_bounce_count;
    host_params.out_prim_ids      = (int *)out_prim_ids;
    host_params.out_shape_ids     = (int *)out_shape_ids;
    host_params.out_hit_points    = (float3 *)out_hit_points;
    host_params.out_geo_normals   = (float3 *)out_geo_normals;
    host_params.out_image_sources = (float3 *)out_image_sources;
    host_params.out_plane_points  = (float3 *)out_plane_points;
    host_params.out_plane_normals = (float3 *)out_plane_normals;

    // Upload params to device
    jit_memcpy(JitBackend::CUDA, params_buffer_, &host_params,
               sizeof(host_params));

    // Build SBT descriptor
    OptixShaderBindingTable sbt = {};
    sbt.raygenRecord            = (CUdeviceptr)sbt_raygen_record_;
    sbt.missRecordBase          = (CUdeviceptr)sbt_miss_record_;
    sbt.missRecordStrideInBytes = sizeof(EmptySbtRecord);
    sbt.missRecordCount         = 1;
    sbt.hitgroupRecordBase          = (CUdeviceptr)sbt_hitgroup_record_;
    sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
    sbt.hitgroupRecordCount         = n_meshes;

    // Launch
    auto optixLaunch_ = (decltype(&optixLaunch))
        jit_optix_lookup("optixLaunch");

    CUstream stream = (CUstream)jit_cuda_stream();
    check_optix(
        optixLaunch_(pipeline_, stream,
                     (CUdeviceptr)params_buffer_,
                     sizeof(ReflectionTraceParams),
                     &sbt,
                     n_rays,  // width
                     1,       // height
                     1),      // depth
        "optixLaunch (reflection trace)");
}

}  // namespace rayd
```

---

## Scene Integration

### scene.h addition

```cpp
#include <rayd/reflection_chain.h>

class Scene {
    // ... existing members ...

    ReflectionChain trace_reflections(const RayDetached &ray,
                                     int max_bounces,
                                     MaskDetached active = true) const;
private:
    // ... existing members ...
    mutable std::unique_ptr<ReflectionTracePipeline> reflection_pipeline_;
};
```

### scene.cpp — trace_reflections implementation

```cpp
#include "scene/reflection_trace_host.h"

ReflectionChain Scene::trace_reflections(
    const RayDetached &ray, int max_bounces, MaskDetached active) const
{
    require(is_ready(), "Scene::trace_reflections(): scene not built.");
    require(!pending_updates_, "Scene::trace_reflections(): pending updates.");
    require(max_bounces > 0 && max_bounces <= kMaxReflectionBounces,
            "max_bounces out of range");

    namespace dr = drjit;

    // ---- Lazy pipeline initialization ----
    if (!reflection_pipeline_) {
        reflection_pipeline_ = std::make_unique<ReflectionTracePipeline>();
        reflection_pipeline_->build(m_accel->context);
    }

    // ---- Evaluate all DrJit inputs (force to device) ----
    const int N = static_cast<int>(slices(ray.o));
    const int B = max_bounces;
    const int NB = N * B;

    // Compact active rays (remove masked-out lanes).
    // For simplicity, first implementation: trace all N rays,
    // apply active mask to output afterward.
    dr::eval(ray.o, ray.d, active);
    dr::eval(triangle_info_detached_.p0, triangle_info_detached_.e1,
             triangle_info_detached_.e2, triangle_info_detached_.face_normal);
    dr::eval(face_offsets_);
    dr::sync_thread();

    // ---- Allocate output buffers ----
    void *buf_bounce_count  = jit_malloc(AllocType::Device, N * sizeof(int));
    void *buf_prim_ids      = jit_malloc(AllocType::Device, NB * sizeof(int));
    void *buf_shape_ids     = jit_malloc(AllocType::Device, NB * sizeof(int));
    void *buf_hit_points    = jit_malloc(AllocType::Device, NB * sizeof(float) * 3);
    void *buf_geo_normals   = jit_malloc(AllocType::Device, NB * sizeof(float) * 3);
    void *buf_image_sources = jit_malloc(AllocType::Device, NB * sizeof(float) * 3);
    void *buf_plane_points  = jit_malloc(AllocType::Device, NB * sizeof(float) * 3);
    void *buf_plane_normals = jit_malloc(AllocType::Device, NB * sizeof(float) * 3);

    // Zero-initialize output
    jit_memset_async(JitBackend::CUDA, buf_bounce_count,  0, N * sizeof(int));
    jit_memset_async(JitBackend::CUDA, buf_prim_ids, 0xFF, NB * sizeof(int)); // -1

    // ---- Convert ray SOA to float3* AOS ----
    // DrJit Vector3fDetached stores x,y,z as separate flat arrays.
    // OptiX expects interleaved float3.  Two options:
    //   A) Pack to AOS before launch (extra kernel + memory)
    //   B) Pass x/y/z pointers separately in params and unpack in raygen
    //
    // We choose (B): modify ReflectionTraceParams to take separate x/y/z
    // pointers and reconstruct float3 in the raygen shader.  This avoids
    // an extra copy.
    //
    // See below for the adjusted param struct and raygen code.

    // Extract device pointers from DrJit arrays
    dr::eval(face_offsets_);
    dr::sync_thread();

    const int n_tris = static_cast<int>(slices(triangle_info_detached_.p0));

    reflection_pipeline_->launch(
        m_accel->ias_handle,
        triangle_info_detached_,
        dr::data_ptr(face_offsets_),
        mesh_count_,
        n_tris,
        dr::data_ptr(ray.o),    // SOA float3 (DrJit interleaved)
        dr::data_ptr(ray.d),
        N, B,
        buf_bounce_count, buf_prim_ids, buf_shape_ids,
        buf_hit_points, buf_geo_normals, buf_image_sources,
        buf_plane_points, buf_plane_normals
    );

    // ---- Wrap output buffers as DrJit arrays ----
    // Use IntDetached::map() / Vector3fDetached::map() to wrap existing
    // device memory without copying.

    ReflectionChain result;
    result.max_bounces = B;
    result.ray_count = N;
    result.bounce_count  = IntDetached::steal(
        jit_var_mem_map(JitBackend::CUDA, VarType::Int32, buf_bounce_count, N, 1));
    result.prim_ids      = IntDetached::steal(
        jit_var_mem_map(JitBackend::CUDA, VarType::Int32, buf_prim_ids, NB, 1));
    result.shape_ids     = IntDetached::steal(
        jit_var_mem_map(JitBackend::CUDA, VarType::Int32, buf_shape_ids, NB, 1));
    // Vector3f: wrap as three separate Float arrays, then construct Vector3f
    // (depends on DrJit memory layout convention)
    // ... similar for hit_points, geo_normals, image_sources, etc.

    // Apply active mask: zero out bounce_count for inactive rays
    const MaskDetached inactive = !active;
    if (dr::any(inactive)) {
        result.bounce_count = dr::select(active, result.bounce_count, IntDetached(0));
    }

    return result;
}
```

---

## SOA ↔ AOS Data Layout

DrJit stores `Vector3fDetached` as three separate float arrays (SOA).
OptiX `float3` is interleaved (AOS).  Two approaches:

### Option A: Pack/Unpack Kernel (simple, +1 kernel launch)

Write a small CUDA kernel in `reflection_trace.cu` to pack SOA→AOS before
`optixLaunch` and unpack AOS→SOA afterward.

### Option B: SOA params in raygen (zero-copy, recommended)

Change `ReflectionTraceParams` to accept SOA pointers:

```cuda
struct ReflectionTraceParams {
    OptixTraversableHandle handle;

    // Triangle data (SOA — DrJit native layout)
    float *tri_p0_x, *tri_p0_y, *tri_p0_z;
    float *tri_e1_x, *tri_e1_y, *tri_e1_z;
    float *tri_e2_x, *tri_e2_y, *tri_e2_z;
    float *tri_fn_x, *tri_fn_y, *tri_fn_z;
    int    n_triangles;

    int   *face_offsets;
    int    n_meshes;

    // Input rays (SOA)
    float *ray_ox, *ray_oy, *ray_oz;
    float *ray_dx, *ray_dy, *ray_dz;
    int    n_rays;
    int    max_bounces;

    // Output (AOS float3 — written directly, wrapped as DrJit after)
    int    *out_bounce_count;
    int    *out_prim_ids;
    int    *out_shape_ids;
    float  *out_hit_x, *out_hit_y, *out_hit_z;
    float  *out_norm_x, *out_norm_y, *out_norm_z;
    float  *out_img_x, *out_img_y, *out_img_z;
    float  *out_plane_pt_x, *out_plane_pt_y, *out_plane_pt_z;
    float  *out_plane_n_x, *out_plane_n_y, *out_plane_n_z;
};
```

And in the raygen program:

```cuda
extern "C" __global__ void __raygen__reflection_trace()
{
    const int i = optixGetLaunchIndex().x;
    if (i >= params.n_rays) return;

    const int B = params.max_bounces;
    float ox = params.ray_ox[i], oy = params.ray_oy[i], oz = params.ray_oz[i];
    float dx = params.ray_dx[i], dy = params.ray_dy[i], dz = params.ray_dz[i];
    float ix = ox, iy = oy, iz = oz;  // image source

    int bounces = 0;

    for (int b = 0; b < B; ++b) {
        unsigned int p0 = 0, p1, p2, p3, p4, p5;
        optixTrace(params.handle,
                   make_float3(ox, oy, oz), make_float3(dx, dy, dz),
                   1e-5f, 1e8f, 0.f, 255u,
                   OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                   0, 1, 0,
                   p0, p1, p2, p3, p4, p5);

        if (p0 == 0u) break;

        float t     = __uint_as_float(p1);
        float bary_u = __uint_as_float(p2);
        float bary_v = __uint_as_float(p3);
        int   lprim  = (int)p4;
        int   sbt_id = (int)p5;

        int shape_id = sbt_id;
        int foff = (sbt_id < params.n_meshes) ? params.face_offsets[sbt_id] : 0;
        int gprim = lprim + foff;

        // Reconstruct from SOA triangle data
        float hx, hy, hz, nx, ny, nz;
        if (gprim >= 0 && gprim < params.n_triangles) {
            float p0x = params.tri_p0_x[gprim], p0y = params.tri_p0_y[gprim], p0z = params.tri_p0_z[gprim];
            float e1x = params.tri_e1_x[gprim], e1y = params.tri_e1_y[gprim], e1z = params.tri_e1_z[gprim];
            float e2x = params.tri_e2_x[gprim], e2y = params.tri_e2_y[gprim], e2z = params.tri_e2_z[gprim];
            hx = p0x + bary_u * e1x + bary_v * e2x;
            hy = p0y + bary_u * e1y + bary_v * e2y;
            hz = p0z + bary_u * e1z + bary_v * e2z;
            nx = params.tri_fn_x[gprim]; ny = params.tri_fn_y[gprim]; nz = params.tri_fn_z[gprim];
            float inv_len = rsqrtf(fmaxf(nx*nx + ny*ny + nz*nz, 1e-12f));
            nx *= inv_len; ny *= inv_len; nz *= inv_len;
        } else {
            hx = ox + t*dx; hy = oy + t*dy; hz = oz + t*dz;
            nx = 0.f; ny = 0.f; nz = 1.f;
        }

        // Flip normal to face ray
        if (dx*nx + dy*ny + dz*nz > 0.f) { nx = -nx; ny = -ny; nz = -nz; }

        // Reflect image source
        float d_img = (ix-hx)*nx + (iy-hy)*ny + (iz-hz)*nz;
        ix -= 2.f * d_img * nx;
        iy -= 2.f * d_img * ny;
        iz -= 2.f * d_img * nz;

        // Write output (SOA)
        int s = i * B + b;
        params.out_prim_ids[s] = gprim;
        params.out_shape_ids[s] = shape_id;
        params.out_hit_x[s] = hx; params.out_hit_y[s] = hy; params.out_hit_z[s] = hz;
        params.out_norm_x[s] = nx; params.out_norm_y[s] = ny; params.out_norm_z[s] = nz;
        params.out_img_x[s] = ix; params.out_img_y[s] = iy; params.out_img_z[s] = iz;
        params.out_plane_pt_x[s] = hx; params.out_plane_pt_y[s] = hy; params.out_plane_pt_z[s] = hz;
        params.out_plane_n_x[s] = nx; params.out_plane_n_y[s] = ny; params.out_plane_n_z[s] = nz;

        // Reflect ray
        float dot_dn = dx*nx + dy*ny + dz*nz;
        dx -= 2.f * dot_dn * nx;
        dy -= 2.f * dot_dn * ny;
        dz -= 2.f * dot_dn * nz;
        ox = hx + 1e-5f * dx;
        oy = hy + 1e-5f * dy;
        oz = hz + 1e-5f * dz;
        bounces = b + 1;
    }

    params.out_bounce_count[i] = bounces;
}
```

This reads and writes SOA directly — zero data conversion.

---

## Build System Changes

### CMakeLists.txt additions

```cmake
# ---- PTX compilation for OptiX raygen program ----
# Requires OptiX SDK headers on include path.
# Set OPTIX_INCLUDE_DIR via env or cmake variable.

find_path(OPTIX_INCLUDE_DIR optix.h
    HINTS
        ENV OPTIX_PATH
        "$ENV{PROGRAMDATA}/NVIDIA Corporation/OptiX SDK 8.1.0/include"
        "/usr/local/NVIDIA-OptiX-SDK-8.1.0/include"
)

set(REFLECTION_TRACE_CU "${CMAKE_CURRENT_SOURCE_DIR}/src/scene/reflection_trace.cu")
set(REFLECTION_TRACE_PTX "${CMAKE_CURRENT_BINARY_DIR}/reflection_trace.ptx")

add_custom_command(
    OUTPUT "${REFLECTION_TRACE_PTX}"
    COMMAND "${CUDA_NVCC_EXECUTABLE}"
        -ptx -O3 --use_fast_math -std=c++17
        -I"${OPTIX_INCLUDE_DIR}"
        -I"${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${REFLECTION_TRACE_CU}"
        -o "${REFLECTION_TRACE_PTX}"
    DEPENDS "${REFLECTION_TRACE_CU}"
    COMMENT "Compiling reflection_trace.cu → PTX"
    VERBATIM
)

# Embed PTX as a C++ string via bin2c or xxd
set(REFLECTION_TRACE_PTX_H "${CMAKE_CURRENT_BINARY_DIR}/reflection_trace_ptx.h")
add_custom_command(
    OUTPUT "${REFLECTION_TRACE_PTX_H}"
    COMMAND ${CMAKE_COMMAND}
        -DPTX_FILE="${REFLECTION_TRACE_PTX}"
        -DOUTPUT_FILE="${REFLECTION_TRACE_PTX_H}"
        -DVAR_NAME="reflection_trace_ptx"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_ptx.cmake"
    DEPENDS "${REFLECTION_TRACE_PTX}"
    COMMENT "Embedding PTX as C++ header"
    VERBATIM
)

# Add host implementation to rayd_core
list(APPEND RAYD_CORE_SOURCE_FILES
    src/scene/reflection_trace_host.cpp
    "${REFLECTION_TRACE_PTX_H}"
)
```

### cmake/embed_ptx.cmake

```cmake
# Read PTX file and generate a C++ header with embedded string.
file(READ "${PTX_FILE}" PTX_CONTENT)
string(LENGTH "${PTX_CONTENT}" PTX_SIZE)
file(WRITE "${OUTPUT_FILE}"
    "// Auto-generated — do not edit.\n"
    "#pragma once\n"
    "#include <cstddef>\n"
    "static const char ${VAR_NAME}[] = R\"PTX(\n"
    "${PTX_CONTENT}"
    ")PTX\";\n"
    "static const size_t ${VAR_NAME}_size = ${PTX_SIZE};\n"
)
```

