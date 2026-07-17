// Host-compile smoke for the migrated multipath algorithm headers (P4 key gate).
//
// This translation unit is compiled by tests/test_rt_host_compile.py with a pure
// host C++ compiler (cl.exe / g++), no CUDA or OptiX device compiler. It proves
// that the de-CUDA-ised algorithm bodies parse and fully type-check off-device:
// they must spell no optixTrace / payload register / launch index, and every ray
// cast must go through the rt::Traverser concept, which a plain host struct can
// satisfy. `<vector_types.h>` (a header-only POD from the CUDA toolkit, pulled in
// by reflection_trace_params.h for the optional packed-triangle inputs) is the
// only CUDA include on the host path.

#include <cstdint>

#include <rayd/shared/multipath/reflection_epc_algo.h>
#include <rayd/shared/multipath/reflection_trace_algo.h>
#include <rayd/shared/multipath/segment_visibility_algo.h>
#include <rayd/shared/rt/qualifiers.h>
#include <rayd/shared/rt/traverser.h>

namespace {

using rayd::shared::math::Vec3f;
using rayd::shared::rt::TriangleHit;

// A host layout policy exercising every compile-time layout branch (the Torch
// superset). Instantiating the algorithm against it type-checks the AoS-input,
// packed-triangle, output-layout, empty-slot, nullable-tmax, and extended-output
// paths under the host compiler.
struct HostLayoutPolicy {
    static constexpr bool allow_aos_inputs = true;
    static constexpr bool allow_packed_triangles = true;
    static constexpr bool honor_output_layout = true;
    static constexpr bool clear_empty_slots = true;
    static constexpr bool nullable_ray_tmax = true;
    static constexpr bool allow_extended_outputs = true;
};

// A trivial host traverser satisfying rt::is_traverser: enough for the algorithm
// body to compile and route all four traversal calls off-device.
struct HostTraverser {
    TriangleHit trace_closest(Vec3f, Vec3f, float, float) const {
        return TriangleHit{0.0f, 0.0f, 0.0f, -1, -1, 0u};
    }
    bool trace_occluded(Vec3f, Vec3f, float, float) const { return false; }
    bool trace_occluded_ignore(Vec3f, Vec3f, float, float,
                               const std::int32_t *, int) const {
        return false;
    }
    TriangleHit trace_first_blocker(Vec3f, Vec3f, float, float,
                                    const std::int32_t *, int) const {
        return TriangleHit{0.0f, 0.0f, 0.0f, -1, -1, 0u};
    }
};

static_assert(rayd::shared::rt::is_traverser_v<HostTraverser>,
              "HostTraverser must satisfy the rt::Traverser concept off-device.");

using HostConfig = rayd::shared::rt::TraceConfig<HostLayoutPolicy, HostTraverser>;

using AlgoFn = void (*)(const rayd::shared::optix::ReflectionTraceParams &,
                        std::uint32_t, const HostTraverser &, const HostTraverser &);

// Segment-visibility layout policy (both compile-time knobs on) and its config.
struct HostSegmentLayoutPolicy {
    static constexpr bool disable_anyhit_without_ignore = true;
    static constexpr bool write_output_t = true;
};
using HostSegmentConfig =
    rayd::shared::rt::TraceConfig<HostSegmentLayoutPolicy, HostTraverser>;
using SegmentAlgoFn = void (*)(const rayd::shared::optix::SegmentVisibilityParams &,
                               std::uint32_t, const HostTraverser &);

// Reflection-EPC layout policy and its config. The EPC algorithm reads only the
// traverser axis; the layout only carries the anyhit-disable knob the OptiX shim's
// traverser template consumes.
struct HostEpcLayoutPolicy {
    static constexpr bool DisableAnyHitWithoutIgnore = true;
};
using HostEpcConfig = rayd::shared::rt::TraceConfig<HostEpcLayoutPolicy, HostTraverser>;
using EpcAlgoFn = void (*)(const rayd::shared::optix::ReflEpcParams &,
                           std::uint32_t, const HostTraverser &, const HostTraverser &);

}  // namespace

// Returning the address forces full host instantiation of each algorithm body.
AlgoFn rt_host_compile_smoke_reflection_trace() {
    return &rayd::shared::multipath::reflection_trace_algo<HostConfig>;
}

SegmentAlgoFn rt_host_compile_smoke_segment_visibility() {
    // Instantiating one entry pulls in the shared trace_segment core; take the
    // address of every launch variant so all four fully type-check off-device.
    volatile SegmentAlgoFn pair =
        &rayd::shared::multipath::segment_pair_visibility_algo<HostSegmentConfig>;
    volatile SegmentAlgoFn axial =
        &rayd::shared::multipath::axial_edge_visibility_algo<HostSegmentConfig>;
    volatile SegmentAlgoFn chain =
        &rayd::shared::multipath::segment_chain_visibility_algo<HostSegmentConfig>;
    (void)pair;
    (void)axial;
    (void)chain;
    return &rayd::shared::multipath::segment_visibility_algo<HostSegmentConfig>;
}

EpcAlgoFn rt_host_compile_smoke_reflection_epc() {
    // Exercise the DirectOnly / PrimaryVisibilityOnly template axes off-device.
    volatile EpcAlgoFn direct =
        &rayd::shared::multipath::run_reflection_epc_algo<HostEpcConfig, true, false>;
    volatile EpcAlgoFn primary_only =
        &rayd::shared::multipath::run_reflection_epc_algo<HostEpcConfig, false, true>;
    (void)direct;
    (void)primary_only;
    return &rayd::shared::multipath::run_reflection_epc_algo<HostEpcConfig, false, false>;
}
