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

#include <rayd/shared/multipath/reflection_trace_algo.h>
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

}  // namespace

// Returning the address forces full host instantiation of the algorithm body.
AlgoFn rt_host_compile_smoke_reflection_trace() {
    return &rayd::shared::multipath::reflection_trace_algo<HostConfig>;
}
