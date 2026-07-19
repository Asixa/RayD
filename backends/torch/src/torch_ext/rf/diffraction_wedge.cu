#include <rayd/torch/rf/diffraction.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/complex.h>
#include <rayd/shared/rf/field_transport.cuh>
#include <rayd/torch/common/tensor_check.h>
#include <rayd/torch/rf/field_transport_ad.cuh>

#include <array>
#include <cmath>
#include <utility>
#include <vector>

// Plan 07 AD-4a: differentiable UTD wedge diffraction and coupled
// reflection-diffraction.
//
// The wedge field is RayD's own templated forward
// (rayd/shared/utd/utd_math.h): instantiated with float it IS the production
// forward, instantiated with utd::Dual the same pass carries an exact
// directional derivative (host-FD validated in both channel conventions).
// Reverse mode runs one seeded dual pass per requested input scalar and
// contracts the output tangents with the cotangents; the
// torch.autograd.Function layer in ops.py is dispatch only.
//
// The coupled row dual below mirrors coupled_rd_field_kernel
// (field_transport.cu) step by step, reusing the validated AD-1/AD-2 duals
// (slab_fresnel_dual, dual_reflect_frame) for the slab legs and the RayD
// templates for everything else. Edit the primal kernel and this mirror
// TOGETHER.

namespace {

constexpr int kBlockSize = 128;
namespace field = rayd::shared::utd;
namespace transport = rayd::shared::rf::field_transport;
namespace ad = rayd::torch::rf::field_transport_ad;

using Dual = field::Dual;

__device__ __forceinline__ field::float3a load3f(
    const float* values, int64_t index) {
    const int64_t base = index * 3;
    return field::make_f3(values[base], values[base + 1], values[base + 2]);
}

__device__ __forceinline__ c10::complex<float> to_c10(
    field::Complex value) {
    return c10::complex<float>(value.re, value.im);
}

__device__ __forceinline__ field::Complex from_c10(
    c10::complex<float> value) {
    return field::cplx(value.real(), value.imag());
}

int launch_blocks(int64_t count) {
    return static_cast<int>((count + kBlockSize - 1) / kBlockSize);
}

at::Tensor zero_scalar(const at::TensorOptions& options) {
    auto tensor = at::empty({1}, options);
    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(tensor.get_device()).stream();
    C10_CUDA_CHECK(cudaMemsetAsync(
        tensor.data_ptr(), 0, tensor.element_size(), stream));
    return tensor;
}

// Scalar/vector seeding shims shared with the diffraction-map companions
// (field_transport_ad.cuh).
using ad::seeded;
using ad::seeded3;
// ---------------------------------------------------------------------------
// Pure diffraction (component 2): re-evaluate RayD's order-1 wedge export
// from the frozen topology. One templated row serves the forward (float) and
// the derivative (Dual). The conventions below reproduce what the export
// actually does (paths_optix.cu + rayd/diffraction.cpp): stationary-point
// selection inside the pair, half-space Fresnel from the face materials
// (mat.omega = 2*pi*f > 0), the +z hard-coded tx polarization the bridge
// hands to RayD, and the sqrt(tx_power) amplitude scale. The forward-parity
// test against topology.field_xyz is the gate for these conventions.
// ---------------------------------------------------------------------------

struct WedgeRowInputs {
    field::float3a source;
    field::float3a target;
    field::float3a edge_pos;
    field::float3a edge_dir;
    float t_min;
    float t_max;
    field::float3a n0;
    field::float3a n1;
    float exterior_angle;
    float eps0, sigma0, mu0, gain0;
    float eps1, sigma1, mu1, gain1;
    bool valid0;
    bool valid1;
    float tx_power;
    float frequency;
    // ISB boundary taper (ADR-017), D member. Plain (non-differentiated) config
    // scalar carried into pair.isbTaperWidthScale so the shared UTD header
    // notches the incident-boundary odd part. 0 reproduces the hard GO step.
    // Taper + AD is refused upstream (deterministic/path pipelines), so this is
    // always 0 on the live AD path; it is threaded for lockstep completeness of
    // the guarded path only.
    float isb_taper_width_scale;
    // Optional winner vertices (plan 07 section 9.3 mesh-vertex x
    // diffraction). When present, the row rebuilds the edge tables from them
    // so vertex seeds reach the edge geometry; the frozen tables above stay
    // the winner reference for sign/plane assignment.
    bool has_vertices;
    bool edge_boundary;
    field::float3a v0, v1, opp0, opp1;
};

struct WedgeRowSeeds {
    field::float3a source;
    field::float3a target;
    float eps0, sigma0, gain0;
    float eps1, sigma1, gain1;
    float frequency;
    field::float3a v0, v1, opp0, opp1;
};

__device__ __forceinline__ WedgeRowSeeds wedge_seeds_zero() {
    WedgeRowSeeds seeds;
    seeds.source = field::f3_zero();
    seeds.target = field::f3_zero();
    seeds.eps0 = 0.f; seeds.sigma0 = 0.f; seeds.gain0 = 0.f;
    seeds.eps1 = 0.f; seeds.sigma1 = 0.f; seeds.gain1 = 0.f;
    seeds.frequency = 0.f;
    seeds.v0 = field::f3_zero();
    seeds.v1 = field::f3_zero();
    seeds.opp0 = field::f3_zero();
    seeds.opp1 = field::f3_zero();
    return seeds;
}
// acos on the templated scalar via atan2 (utd has no acos shim); equals
// ::acosf to float rounding on [-1, 1].
template <typename T>
__device__ __forceinline__ T wedge_acos(T x) {
    return field::atan2f(field::sqrtf(field::fmaxf(1.0f - x * x, T(0.f))), x);
}

// Primal part of a templated vector (identity for float).
template <typename T>
__device__ __forceinline__ field::float3a primal3(field::Vec3T<T> v) {
    return field::make_f3(
        field::scalar_value(v.x), field::scalar_value(v.y),
        field::scalar_value(v.z));
}

// Differentiable edge tables rebuilt from the winner vertices. The frozen
// discovery tables (in.n0 / in.n1) pick the plane assignment and the normal
// signs, so RayD's winding/ordering conventions cannot drift the primal
// values; the derivative flows through the aligned smooth normal. Mirrors
// diffraction_edge_geometry_kernel (kernels/diffraction.cu) row math; edit
// the discovery kernel and this rebuild TOGETHER.
template <typename T>
struct WedgeEdgeTables {
    field::Vec3T<T> edge_pos;
    field::Vec3T<T> edge_dir;
    T t_min;
    T t_max;
    field::Vec3T<T> n0;
    field::Vec3T<T> n1;
    T wedge_n;  // exterior_angle / pi
};

template <typename T>
__device__ WedgeEdgeTables<T> wedge_edge_tables_from_vertices(
    const WedgeRowInputs& in, const WedgeRowSeeds& seeds) {
    WedgeEdgeTables<T> tables;
    const field::Vec3T<T> v0 = seeded3<T>(in.v0, seeds.v0);
    const field::Vec3T<T> v1 = seeded3<T>(in.v1, seeds.v1);
    const field::Vec3T<T> vector = field::f3_sub(v1, v0);
    const T length = field::fmaxf(field::safe_length(vector), T(1.0e-12f));
    tables.edge_dir = field::f3_mul(vector, 1.0f / length);
    tables.edge_pos = field::f3_mul(field::f3_add(v0, v1), 0.5f);
    tables.t_min = -0.5f * length;
    tables.t_max = 0.5f * length;

    const field::Vec3T<T> opp0 = seeded3<T>(in.opp0, seeds.opp0);
    const field::Vec3T<T> candidate_a = field::safe_normalize(
        field::f3_cross(vector, field::f3_sub(opp0, v0)),
        field::v3_const<T>(0.f, 0.f, 1.f));
    field::Vec3T<T> pick0 = candidate_a;
    field::Vec3T<T> pick1 = candidate_a;
    if (!in.edge_boundary) {
        const field::Vec3T<T> opp1 = seeded3<T>(in.opp1, seeds.opp1);
        const field::Vec3T<T> candidate_b = field::safe_normalize(
            field::f3_cross(vector, field::f3_sub(opp1, v0)),
            field::v3_const<T>(0.f, 0.f, 1.f));
        // Frozen plane assignment: match each candidate to the discovery
        // table slot it is (anti)parallel to (primal values only).
        const field::float3a a_val = primal3<T>(candidate_a);
        const bool a_is_n0 =
            ::fabsf(field::f3_dot(a_val, in.n0)) >=
            ::fabsf(field::f3_dot(a_val, in.n1));
        pick0 = a_is_n0 ? candidate_a : candidate_b;
        pick1 = a_is_n0 ? candidate_b : candidate_a;
    }
    // Frozen sign alignment against the discovery normals.
    const float sign0 =
        field::f3_dot(primal3<T>(pick0), in.n0) < 0.f ? -1.f : 1.f;
    tables.n0 = field::f3_mul(pick0, sign0);
    if (in.edge_boundary) {
        tables.n1 = field::f3_neg(tables.n0);
        tables.wedge_n = T(2.f);
    } else {
        const float sign1 =
            field::f3_dot(primal3<T>(pick1), in.n1) < 0.f ? -1.f : 1.f;
        tables.n1 = field::f3_mul(pick1, sign1);
        const T neg_dot = field::fminf(
            field::fmaxf(-field::f3_dot(tables.n0, tables.n1), T(-1.f)), T(1.f));
        const T exterior = 2.0f * field::UTD_PI - wedge_acos(neg_dot);
        tables.wedge_n = exterior * (1.0f / field::UTD_PI);
    }
    return tables;
}

template <typename T>
struct WedgeRowOutputs {
    field::Complex3T<T> field_vector;  // includes the sqrt(tx_power) scale
    field::Vec3T<T> direction;         // arrival from the clamped edge point
};

// RayD's face_material_params: absent faces keep the default material and
// present = 0 (the pair then treats the face operator as zero).
template <typename T>
__device__ __forceinline__ field::FaceMaterialParamsT<T> wedge_face_material(
    bool valid, T eps, T sigma, float mu, T gain) {
    if (!valid) {
        return {T(1.f), T(1.f), T(0.f), T(1.f), 1.f, 0.f};
    }
    return {eps, T(mu), sigma, field::fmaxf(gain, T(0.f)), 1.f, 1.f};
}

template <typename T>
__device__ WedgeRowOutputs<T> wedge_row_eval(
    const WedgeRowInputs& in, const WedgeRowSeeds& seeds) {
    const field::Vec3T<T> source = seeded3<T>(in.source, seeds.source);
    const field::Vec3T<T> target = seeded3<T>(in.target, seeds.target);
    const T frequency = seeded<T>(in.frequency, seeds.frequency);
    const T wave_number =
        2.0f * field::UTD_PI * frequency / transport::kSpeedOfLight;

    const field::float3a zero3 = field::f3_zero();
    field::PairInputsT<T> pair{};
    T edge_t_min;
    T edge_t_max;
    if (in.has_vertices) {
        const WedgeEdgeTables<T> tables =
            wedge_edge_tables_from_vertices<T>(in, seeds);
        pair.edgePos = tables.edge_pos;
        pair.edgeDir = tables.edge_dir;
        pair.n0 = tables.n0;
        pair.nn = tables.n1;
        pair.wedgeN = tables.wedge_n;
        edge_t_min = tables.t_min;
        edge_t_max = tables.t_max;
    } else {
        pair.edgePos = seeded3<T>(in.edge_pos, zero3);
        pair.edgeDir = seeded3<T>(in.edge_dir, zero3);
        pair.n0 = seeded3<T>(in.n0, zero3);
        pair.nn = seeded3<T>(in.n1, zero3);
        pair.wedgeN = T(in.exterior_angle / field::UTD_PI);
        edge_t_min = T(in.t_min);
        edge_t_max = T(in.t_max);
    }
    pair.edgeLineMin = edge_t_min;
    pair.edgeLineMax = edge_t_max;
    pair.sourcePos = source;
    pair.selectStationaryPoint = 1.f;
    // ISB boundary taper (ADR-017), D member. isbTaperWidthScale is a plain
    // float in PairInputsT (a config scalar like selectStationaryPoint), so it
    // is assigned directly and carries no tangent; the header derives w_F / s2
    // internally, so the kernel must not precompute them. 0 = hard GO step
    // (bit-identical to the pre-ADR-017 twin).
    pair.isbTaperWidthScale = in.isb_taper_width_scale;
    pair.face0Material = wedge_face_material(
        in.valid0, seeded<T>(in.eps0, seeds.eps0),
        seeded<T>(in.sigma0, seeds.sigma0), in.mu0,
        seeded<T>(in.gain0, seeds.gain0));
    pair.face1Material = wedge_face_material(
        in.valid1, seeded<T>(in.eps1, seeds.eps1),
        seeded<T>(in.sigma1, seeds.sigma1), in.mu1,
        seeded<T>(in.gain1, seeds.gain1));

    field::MaterialParamsT<T> mat{};
    mat.useFresnel = 1;
    mat.etaR = T(1.f);
    mat.muR = T(1.f);
    mat.sigma = T(0.f);
    mat.gain = T(1.f);
    mat.omega = 2.0f * field::UTD_PI * frequency;
    // rayd/diffraction.cpp hands RayD a hard-coded +z tx polarization for the
    // order-1 diffraction export; reproduce it (forward-parity gate).
    mat.txPolX = T(0.f);
    mat.txPolY = T(0.f);
    mat.txPolZ = T(1.f);

    WedgeRowOutputs<T> out;
    const field::Complex3T<T> vec =
        field::compute_pair_vector_contribution(pair, target, wave_number, mat);
    const T amplitude = sqrtf(T(fmaxf(in.tx_power, 0.f)));
    out.field_vector = field::c3_scale_real(vec, amplitude);

    // Arrival direction from the clamped stationary point (the export's p0).
    const field::Vec3T<T> edge_hat = field::safe_normalize(
        pair.edgeDir, field::v3_const<T>(0.f, 0.f, 1.f));
    const T edge_length = edge_t_max - edge_t_min;
    const field::Vec3T<T> edge_origin = field::f3_add(
        pair.edgePos, field::f3_mul(edge_hat, edge_t_min));
    const T parameter = field::first_order_diffraction_parameter(
        source, target, edge_origin, edge_hat);
    const T clamped = field::fminf(field::fmaxf(parameter, T(0.f)), edge_length);
    const field::Vec3T<T> point = field::f3_add(
        edge_origin, field::f3_mul(edge_hat, clamped));
    out.direction = field::safe_normalize(
        field::f3_sub(target, point), field::v3_const<T>(0.f, 0.f, 1.f));
    return out;
}

__device__ __forceinline__ WedgeRowInputs load_wedge_row(
    int64_t index,
    const float* source,
    const float* target,
    const float* edge_position,
    const float* edge_direction,
    const float* edge_t_min,
    const float* edge_t_max,
    const float* edge_n0,
    const float* edge_n1,
    const float* exterior_angle,
    const bool* face0_valid,
    const float* face0_eps_r,
    const float* face0_sigma_e,
    const float* face0_mu_r,
    const float* face0_gain,
    const bool* face1_valid,
    const float* face1_eps_r,
    const float* face1_sigma_e,
    const float* face1_mu_r,
    const float* face1_gain,
    const float* tx_power,
    float frequency_hz,
    const float* vertex_v0,
    const float* vertex_v1,
    const float* vertex_opp0,
    const float* vertex_opp1,
    const bool* edge_boundary,
    float isb_taper_width) {
    WedgeRowInputs in;
    in.source = load3f(source, index);
    in.target = load3f(target, index);
    in.edge_pos = load3f(edge_position, index);
    in.edge_dir = load3f(edge_direction, index);
    in.t_min = edge_t_min[index];
    in.t_max = edge_t_max[index];
    in.n0 = load3f(edge_n0, index);
    in.n1 = load3f(edge_n1, index);
    in.exterior_angle = exterior_angle[index];
    in.valid0 = face0_valid[index];
    in.eps0 = face0_eps_r[index];
    in.sigma0 = face0_sigma_e[index];
    in.mu0 = face0_mu_r[index];
    in.gain0 = face0_gain[index];
    in.valid1 = face1_valid[index];
    in.eps1 = face1_eps_r[index];
    in.sigma1 = face1_sigma_e[index];
    in.mu1 = face1_mu_r[index];
    in.gain1 = face1_gain[index];
    in.tx_power = tx_power[index];
    in.frequency = frequency_hz;
    in.isb_taper_width_scale = isb_taper_width;
    in.has_vertices = vertex_v0 != nullptr;
    if (in.has_vertices) {
        in.edge_boundary = edge_boundary[index];
        in.v0 = load3f(vertex_v0, index);
        in.v1 = load3f(vertex_v1, index);
        in.opp0 = load3f(vertex_opp0, index);
        in.opp1 = load3f(vertex_opp1, index);
    } else {
        in.edge_boundary = false;
        in.v0 = field::f3_zero();
        in.v1 = field::f3_zero();
        in.opp0 = field::f3_zero();
        in.opp1 = field::f3_zero();
    }
    return in;
}

#define WEDGE_ROW_PARAMS                                                      \
    const float* source, const float* target, const float* edge_position,    \
        const float* edge_direction, const float* edge_t_min,                 \
        const float* edge_t_max, const float* edge_n0, const float* edge_n1,  \
        const float* exterior_angle, const bool* face0_valid,                 \
        const float* face0_eps_r, const float* face0_sigma_e,                 \
        const float* face0_mu_r, const float* face0_gain,                     \
        const bool* face1_valid, const float* face1_eps_r,                    \
        const float* face1_sigma_e, const float* face1_mu_r,                  \
        const float* face1_gain, const float* tx_power, float frequency_hz,   \
        const float* vertex_v0, const float* vertex_v1,                       \
        const float* vertex_opp0, const float* vertex_opp1,                   \
        const bool* edge_boundary, float isb_taper_width

#define WEDGE_ROW_ARGS(index)                                                 \
    index, source, target, edge_position, edge_direction, edge_t_min,         \
        edge_t_max, edge_n0, edge_n1, exterior_angle, face0_valid,            \
        face0_eps_r, face0_sigma_e, face0_mu_r, face0_gain, face1_valid,      \
        face1_eps_r, face1_sigma_e, face1_mu_r, face1_gain, tx_power,         \
        frequency_hz, vertex_v0, vertex_v1, vertex_opp0, vertex_opp1,         \
        edge_boundary, isb_taper_width

__global__ void diffraction_wedge_forward_kernel(
    int64_t count,
    WEDGE_ROW_PARAMS,
    c10::complex<float>* field_vector,
    float* direction) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const WedgeRowInputs in = load_wedge_row(WEDGE_ROW_ARGS(index));
        const WedgeRowOutputs<float> out =
            wedge_row_eval<float>(in, wedge_seeds_zero());
        const int64_t base = index * 3;
        field_vector[base] = to_c10(out.field_vector.x);
        field_vector[base + 1] = to_c10(out.field_vector.y);
        field_vector[base + 2] = to_c10(out.field_vector.z);
        direction[base] = out.direction.x;
        direction[base + 1] = out.direction.y;
        direction[base + 2] = out.direction.z;
    }
}

__device__ __forceinline__ float wedge_contract(
    int64_t index,
    const c10::complex<float>* grad_field_vector,
    const float* grad_direction,
    const WedgeRowOutputs<Dual>& out) {
    float acc = 0.f;
    const int64_t base = index * 3;
    if (grad_field_vector != nullptr) {
        const field::Complex3 tangent = field::dual_tangent(out.field_vector);
        acc += field::cplx_adj_dot(from_c10(grad_field_vector[base]), tangent.x);
        acc += field::cplx_adj_dot(from_c10(grad_field_vector[base + 1]), tangent.y);
        acc += field::cplx_adj_dot(from_c10(grad_field_vector[base + 2]), tangent.z);
    }
    if (grad_direction != nullptr) {
        acc += grad_direction[base] * out.direction.x.d;
        acc += grad_direction[base + 1] * out.direction.y.d;
        acc += grad_direction[base + 2] * out.direction.z.d;
    }
    return acc;
}

__global__ void diffraction_wedge_backward_kernel(
    int64_t count,
    WEDGE_ROW_PARAMS,
    const c10::complex<float>* grad_field_vector,
    const float* grad_direction,
    float* grad_source,
    float* grad_target,
    float* grad_face0_eps_r,
    float* grad_face0_sigma_e,
    float* grad_face0_gain,
    float* grad_face1_eps_r,
    float* grad_face1_sigma_e,
    float* grad_face1_gain,
    float* grad_frequency,
    float* grad_vertex_v0,
    float* grad_vertex_v1,
    float* grad_vertex_opp0,
    float* grad_vertex_opp1) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const WedgeRowInputs in = load_wedge_row(WEDGE_ROW_ARGS(index));
        WedgeRowSeeds seeds = wedge_seeds_zero();
        const int64_t base = index * 3;
        if (grad_source != nullptr) {
            float* slots[3] = {&seeds.source.x, &seeds.source.y, &seeds.source.z};
            for (int axis = 0; axis < 3; ++axis) {
                *slots[axis] = 1.f;
                const WedgeRowOutputs<Dual> out = wedge_row_eval<Dual>(in, seeds);
                *slots[axis] = 0.f;
                grad_source[base + axis] = wedge_contract(
                    index, grad_field_vector, grad_direction, out);
            }
        }
        if (grad_target != nullptr) {
            float* slots[3] = {&seeds.target.x, &seeds.target.y, &seeds.target.z};
            for (int axis = 0; axis < 3; ++axis) {
                *slots[axis] = 1.f;
                const WedgeRowOutputs<Dual> out = wedge_row_eval<Dual>(in, seeds);
                *slots[axis] = 0.f;
                grad_target[base + axis] = wedge_contract(
                    index, grad_field_vector, grad_direction, out);
            }
        }
        if (grad_vertex_v0 != nullptr) {
            struct VertexSlot {
                field::float3a* seed;
                float* grad;
            };
            VertexSlot vertex_slots[4] = {
                {&seeds.v0, grad_vertex_v0},
                {&seeds.v1, grad_vertex_v1},
                {&seeds.opp0, grad_vertex_opp0},
                {&seeds.opp1, grad_vertex_opp1},
            };
            for (int slot = 0; slot < 4; ++slot) {
                float* components[3] = {
                    &vertex_slots[slot].seed->x,
                    &vertex_slots[slot].seed->y,
                    &vertex_slots[slot].seed->z,
                };
                for (int axis = 0; axis < 3; ++axis) {
                    *components[axis] = 1.f;
                    const WedgeRowOutputs<Dual> out =
                        wedge_row_eval<Dual>(in, seeds);
                    *components[axis] = 0.f;
                    vertex_slots[slot].grad[base + axis] = wedge_contract(
                        index, grad_field_vector, grad_direction, out);
                }
            }
        }
        struct MaterialSlot {
            float* seed;
            float* grad;
        };
        MaterialSlot material_slots[6] = {
            {&seeds.eps0, grad_face0_eps_r},
            {&seeds.sigma0, grad_face0_sigma_e},
            {&seeds.gain0, grad_face0_gain},
            {&seeds.eps1, grad_face1_eps_r},
            {&seeds.sigma1, grad_face1_sigma_e},
            {&seeds.gain1, grad_face1_gain},
        };
        for (int slot = 0; slot < 6; ++slot) {
            if (material_slots[slot].grad == nullptr)
                continue;
            *material_slots[slot].seed = 1.f;
            const WedgeRowOutputs<Dual> out = wedge_row_eval<Dual>(in, seeds);
            *material_slots[slot].seed = 0.f;
            material_slots[slot].grad[index] = wedge_contract(
                index, grad_field_vector, grad_direction, out);
        }
        if (grad_frequency != nullptr) {
            seeds.frequency = 1.f;
            const WedgeRowOutputs<Dual> out = wedge_row_eval<Dual>(in, seeds);
            seeds.frequency = 0.f;
            atomicAdd(grad_frequency, wedge_contract(
                index, grad_field_vector, grad_direction, out));
        }
    }
}

__global__ void diffraction_wedge_jvp_kernel(
    int64_t count,
    WEDGE_ROW_PARAMS,
    const float* tangent_source,
    const float* tangent_target,
    const float* tangent_face0_eps_r,
    const float* tangent_face0_sigma_e,
    const float* tangent_face0_gain,
    const float* tangent_face1_eps_r,
    const float* tangent_face1_sigma_e,
    const float* tangent_face1_gain,
    float tangent_frequency,
    const float* tangent_vertex_v0,
    const float* tangent_vertex_v1,
    const float* tangent_vertex_opp0,
    const float* tangent_vertex_opp1,
    c10::complex<float>* tangent_field_vector,
    float* tangent_direction) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const WedgeRowInputs in = load_wedge_row(WEDGE_ROW_ARGS(index));
        WedgeRowSeeds seeds = wedge_seeds_zero();
        if (tangent_source != nullptr)
            seeds.source = load3f(tangent_source, index);
        if (tangent_target != nullptr)
            seeds.target = load3f(tangent_target, index);
        if (tangent_face0_eps_r != nullptr)
            seeds.eps0 = tangent_face0_eps_r[index];
        if (tangent_face0_sigma_e != nullptr)
            seeds.sigma0 = tangent_face0_sigma_e[index];
        if (tangent_face0_gain != nullptr)
            seeds.gain0 = tangent_face0_gain[index];
        if (tangent_face1_eps_r != nullptr)
            seeds.eps1 = tangent_face1_eps_r[index];
        if (tangent_face1_sigma_e != nullptr)
            seeds.sigma1 = tangent_face1_sigma_e[index];
        if (tangent_face1_gain != nullptr)
            seeds.gain1 = tangent_face1_gain[index];
        if (tangent_vertex_v0 != nullptr)
            seeds.v0 = load3f(tangent_vertex_v0, index);
        if (tangent_vertex_v1 != nullptr)
            seeds.v1 = load3f(tangent_vertex_v1, index);
        if (tangent_vertex_opp0 != nullptr)
            seeds.opp0 = load3f(tangent_vertex_opp0, index);
        if (tangent_vertex_opp1 != nullptr)
            seeds.opp1 = load3f(tangent_vertex_opp1, index);
        seeds.frequency = tangent_frequency;
        const WedgeRowOutputs<Dual> out = wedge_row_eval<Dual>(in, seeds);
        const field::Complex3 tangent = field::dual_tangent(out.field_vector);
        const int64_t base = index * 3;
        tangent_field_vector[base] = to_c10(tangent.x);
        tangent_field_vector[base + 1] = to_c10(tangent.y);
        tangent_field_vector[base + 2] = to_c10(tangent.z);
        tangent_direction[base] = out.direction.x.d;
        tangent_direction[base + 1] = out.direction.y.d;
        tangent_direction[base + 2] = out.direction.z.d;
    }
}

void require_same_device(
    const at::Tensor& tensor,
    const at::Tensor& reference,
    const char* name) {
    TORCH_CHECK(
        tensor.get_device() == reference.get_device(),
        name, " must share the primal device");
}

void require_rows(
    const at::Tensor& tensor,
    int64_t count,
    const char* name) {
    TORCH_CHECK(tensor.size(0) == count, name, " must match source rows");
}

void check_wedge_primal(const rayd::torch::DiffractionWedgeRequest& request) {
    using rayd::torch_backend::require_mask;
    using rayd::torch_backend::require_scalar_f;
    using rayd::torch_backend::require_vec3f;

    require_vec3f(request.source, "source");
    const int64_t count = request.source.size(0);
    for (const auto& named :
         std::array<std::pair<const at::Tensor*, const char*>, 5>{{
             {&request.target, "target"},
             {&request.edge_position, "edge_position"},
             {&request.edge_direction, "edge_direction"},
             {&request.edge_n0, "edge_n0"},
             {&request.edge_n1, "edge_n1"}}}) {
        require_vec3f(*named.first, named.second);
        require_rows(*named.first, count, named.second);
        require_same_device(*named.first, request.source, named.second);
    }
    for (const auto& named :
         std::array<std::pair<const at::Tensor*, const char*>, 12>{{
             {&request.edge_t_min, "edge_t_min"},
             {&request.edge_t_max, "edge_t_max"},
             {&request.exterior_angle, "exterior_angle"},
             {&request.face0_eps_r, "face0_eps_r"},
             {&request.face0_sigma_e, "face0_sigma_e"},
             {&request.face0_mu_r, "face0_mu_r"},
             {&request.face0_gain, "face0_gain"},
             {&request.face1_eps_r, "face1_eps_r"},
             {&request.face1_sigma_e, "face1_sigma_e"},
             {&request.face1_mu_r, "face1_mu_r"},
             {&request.face1_gain, "face1_gain"},
             {&request.tx_power, "tx_power"}}}) {
        require_scalar_f(*named.first, named.second);
        require_rows(*named.first, count, named.second);
        require_same_device(*named.first, request.source, named.second);
    }
    for (const auto& named :
         std::array<std::pair<const at::Tensor*, const char*>, 2>{{
             {&request.face0_valid, "face0_valid"},
             {&request.face1_valid, "face1_valid"}}}) {
        require_mask(*named.first, named.second);
        require_rows(*named.first, count, named.second);
        require_same_device(*named.first, request.source, named.second);
    }
    TORCH_CHECK(
        request.frequency_hz > 0.0,
        "frequency_hz must be positive");
}

struct WedgeVertexArgs {
    const at::Tensor* v0 = nullptr;
    const at::Tensor* v1 = nullptr;
    const at::Tensor* opp0 = nullptr;
    const at::Tensor* opp1 = nullptr;
    const at::Tensor* boundary = nullptr;
};

WedgeVertexArgs resolve_wedge_vertices(
    const rayd::torch::DiffractionWedgeRequest& request) {
    const bool any = request.vertex_v0.has_value() ||
        request.vertex_v1.has_value() || request.vertex_opp0.has_value() ||
        request.vertex_opp1.has_value() || request.edge_boundary.has_value();
    if (!any)
        return {};
    TORCH_CHECK(
        request.vertex_v0.has_value() && request.vertex_v1.has_value() &&
            request.vertex_opp0.has_value() &&
            request.vertex_opp1.has_value() &&
            request.edge_boundary.has_value(),
        "wedge vertex inputs must be supplied together");

    const int64_t count = request.source.size(0);
    for (const auto& named :
         std::array<std::pair<const at::Tensor*, const char*>, 4>{{
             {&*request.vertex_v0, "vertex_v0"},
             {&*request.vertex_v1, "vertex_v1"},
             {&*request.vertex_opp0, "vertex_opp0"},
             {&*request.vertex_opp1, "vertex_opp1"}}}) {
        rayd::torch_backend::require_vec3f(*named.first, named.second);
        require_rows(*named.first, count, named.second);
        require_same_device(*named.first, request.source, named.second);
    }
    rayd::torch_backend::require_mask(
        *request.edge_boundary, "edge_boundary");
    require_rows(*request.edge_boundary, count, "edge_boundary");
    require_same_device(
        *request.edge_boundary, request.source, "edge_boundary");
    return {
        &*request.vertex_v0,
        &*request.vertex_v1,
        &*request.vertex_opp0,
        &*request.vertex_opp1,
        &*request.edge_boundary};
}

const at::Tensor* optional_tensor(
    const std::optional<at::Tensor>& value,
    const char* name,
    at::ScalarType dtype,
    at::IntArrayRef sizes,
    const at::Tensor& reference) {
    if (!value.has_value())
        return nullptr;
    rayd::torch_backend::require_cuda(*value, name);
    rayd::torch_backend::require_contiguous(*value, name);
    rayd::torch_backend::require_dtype(*value, dtype, name);
    TORCH_CHECK(value->sizes() == sizes, name, " has the wrong shape");
    require_same_device(*value, reference, name);
    return &*value;
}

template <typename T>
const T* opt_ptr(const at::Tensor* tensor) {
    return tensor == nullptr ? nullptr : tensor->data_ptr<T>();
}

template <typename T>
T* opt_mut_ptr(at::Tensor* tensor) {
    return tensor == nullptr ? nullptr : tensor->data_ptr<T>();
}

} // namespace

#define WEDGE_HOST_ARGS                                                       \
    primal.source.data_ptr<float>(), primal.target.data_ptr<float>(),         \
        primal.edge_position.data_ptr<float>(),                               \
        primal.edge_direction.data_ptr<float>(),                              \
        primal.edge_t_min.data_ptr<float>(),                                  \
        primal.edge_t_max.data_ptr<float>(), primal.edge_n0.data_ptr<float>(), \
        primal.edge_n1.data_ptr<float>(),                                     \
        primal.exterior_angle.data_ptr<float>(),                              \
        primal.face0_valid.data_ptr<bool>(),                                  \
        primal.face0_eps_r.data_ptr<float>(),                                 \
        primal.face0_sigma_e.data_ptr<float>(),                               \
        primal.face0_mu_r.data_ptr<float>(),                                  \
        primal.face0_gain.data_ptr<float>(),                                  \
        primal.face1_valid.data_ptr<bool>(),                                  \
        primal.face1_eps_r.data_ptr<float>(),                                 \
        primal.face1_sigma_e.data_ptr<float>(),                               \
        primal.face1_mu_r.data_ptr<float>(),                                  \
        primal.face1_gain.data_ptr<float>(), primal.tx_power.data_ptr<float>(), \
        static_cast<float>(primal.frequency_hz),                              \
        opt_ptr<float>(vertex_args.v0), opt_ptr<float>(vertex_args.v1),       \
        opt_ptr<float>(vertex_args.opp0), opt_ptr<float>(vertex_args.opp1),   \
        opt_ptr<bool>(vertex_args.boundary),                                  \
        static_cast<float>(primal.isb_boundary_taper_width)

rayd::torch::DiffractionWedgeResult rayd::torch::field_diffraction_wedge(
    const DiffractionWedgeRequest& primal) {
    check_wedge_primal(primal);
    const WedgeVertexArgs vertex_args = resolve_wedge_vertices(primal);
    const int64_t count = primal.source.size(0);
    auto field_vector = at::empty(
        {count, 3}, primal.source.options().dtype(at::kComplexFloat));
    auto direction = at::empty({count, 3}, primal.source.options());
    if (count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(
            primal.source.get_device()).stream();
        diffraction_wedge_forward_kernel<<<
            launch_blocks(count), kBlockSize, 0, stream>>>(
                count,
                WEDGE_HOST_ARGS,
                field_vector.data_ptr<c10::complex<float>>(),
                direction.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {field_vector, direction};
}

rayd::torch::DiffractionWedgeBackwardResult
rayd::torch::field_diffraction_wedge_backward(
    const DiffractionWedgeBackwardRequest& request) {
    const auto& primal = request.primal;
    check_wedge_primal(primal);
    const WedgeVertexArgs vertex_args = resolve_wedge_vertices(primal);
    TORCH_CHECK(
        !request.need_grad_vertices || vertex_args.v0 != nullptr,
        "vertex gradients require the wedge vertex inputs");
    const int64_t count = primal.source.size(0);
    const at::Tensor* g_field = optional_tensor(
        request.grad_field_vector, "grad_field_vector", at::kComplexFloat,
        {count, 3}, primal.source);
    const at::Tensor* g_direction = optional_tensor(
        request.grad_direction, "grad_direction", at::kFloat,
        {count, 3}, primal.source);

    const auto options = primal.source.options();
    at::Tensor grad_source, grad_target;
    at::Tensor grad_face0_eps, grad_face0_sigma, grad_face0_gain;
    at::Tensor grad_face1_eps, grad_face1_sigma, grad_face1_gain;
    at::Tensor grad_frequency;
    at::Tensor grad_vertices[4];
    at::Tensor* grad_source_ptr = nullptr;
    at::Tensor* grad_target_ptr = nullptr;
    at::Tensor* material_ptrs[6] = {
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    at::Tensor* grad_frequency_ptr = nullptr;
    at::Tensor* grad_vertex_ptrs[4] = {nullptr, nullptr, nullptr, nullptr};
    if (request.need_grad_geometry) {
        grad_source = at::empty({count, 3}, options);
        grad_target = at::empty({count, 3}, options);
        grad_source_ptr = &grad_source;
        grad_target_ptr = &grad_target;
    }
    if (request.need_grad_material) {
        grad_face0_eps = at::empty({count}, options);
        grad_face0_sigma = at::empty({count}, options);
        grad_face0_gain = at::empty({count}, options);
        grad_face1_eps = at::empty({count}, options);
        grad_face1_sigma = at::empty({count}, options);
        grad_face1_gain = at::empty({count}, options);
        material_ptrs[0] = &grad_face0_eps;
        material_ptrs[1] = &grad_face0_sigma;
        material_ptrs[2] = &grad_face0_gain;
        material_ptrs[3] = &grad_face1_eps;
        material_ptrs[4] = &grad_face1_sigma;
        material_ptrs[5] = &grad_face1_gain;
    }
    if (request.need_grad_frequency) {
        grad_frequency = zero_scalar(options);
        grad_frequency_ptr = &grad_frequency;
    }
    if (request.need_grad_vertices) {
        for (int slot = 0; slot < 4; ++slot) {
            grad_vertices[slot] = at::empty({count, 3}, options);
            grad_vertex_ptrs[slot] = &grad_vertices[slot];
        }
    }
    if (count > 0 && (g_field != nullptr || g_direction != nullptr)) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(
            primal.source.get_device()).stream();
        diffraction_wedge_backward_kernel<<<
            launch_blocks(count), kBlockSize, 0, stream>>>(
                count,
                WEDGE_HOST_ARGS,
                opt_ptr<c10::complex<float>>(g_field),
                opt_ptr<float>(g_direction),
                opt_mut_ptr<float>(grad_source_ptr),
                opt_mut_ptr<float>(grad_target_ptr),
                opt_mut_ptr<float>(material_ptrs[0]),
                opt_mut_ptr<float>(material_ptrs[1]),
                opt_mut_ptr<float>(material_ptrs[2]),
                opt_mut_ptr<float>(material_ptrs[3]),
                opt_mut_ptr<float>(material_ptrs[4]),
                opt_mut_ptr<float>(material_ptrs[5]),
                opt_mut_ptr<float>(grad_frequency_ptr),
                opt_mut_ptr<float>(grad_vertex_ptrs[0]),
                opt_mut_ptr<float>(grad_vertex_ptrs[1]),
                opt_mut_ptr<float>(grad_vertex_ptrs[2]),
                opt_mut_ptr<float>(grad_vertex_ptrs[3]));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        for (at::Tensor* tensor : {grad_source_ptr, grad_target_ptr}) {
            if (tensor != nullptr)
                tensor->zero_();
        }
        for (at::Tensor* tensor : material_ptrs) {
            if (tensor != nullptr)
                tensor->zero_();
        }
        for (at::Tensor* tensor : grad_vertex_ptrs) {
            if (tensor != nullptr)
                tensor->zero_();
        }
    }

    return {
        grad_source_ptr ? std::optional<at::Tensor>(grad_source) : std::nullopt,
        grad_target_ptr ? std::optional<at::Tensor>(grad_target) : std::nullopt,
        material_ptrs[0]
            ? std::optional<at::Tensor>(grad_face0_eps) : std::nullopt,
        material_ptrs[1]
            ? std::optional<at::Tensor>(grad_face0_sigma) : std::nullopt,
        material_ptrs[2]
            ? std::optional<at::Tensor>(grad_face0_gain) : std::nullopt,
        material_ptrs[3]
            ? std::optional<at::Tensor>(grad_face1_eps) : std::nullopt,
        material_ptrs[4]
            ? std::optional<at::Tensor>(grad_face1_sigma) : std::nullopt,
        material_ptrs[5]
            ? std::optional<at::Tensor>(grad_face1_gain) : std::nullopt,
        grad_frequency_ptr
            ? std::optional<at::Tensor>(grad_frequency) : std::nullopt,
        grad_vertex_ptrs[0]
            ? std::optional<at::Tensor>(grad_vertices[0]) : std::nullopt,
        grad_vertex_ptrs[1]
            ? std::optional<at::Tensor>(grad_vertices[1]) : std::nullopt,
        grad_vertex_ptrs[2]
            ? std::optional<at::Tensor>(grad_vertices[2]) : std::nullopt,
        grad_vertex_ptrs[3]
            ? std::optional<at::Tensor>(grad_vertices[3]) : std::nullopt};
}

rayd::torch::DiffractionWedgeJvpResult
rayd::torch::field_diffraction_wedge_jvp(
    const DiffractionWedgeJvpRequest& request) {
    const auto& primal = request.primal;
    check_wedge_primal(primal);
    const WedgeVertexArgs vertex_args = resolve_wedge_vertices(primal);
    const int64_t count = primal.source.size(0);
    const at::Tensor* t_source = optional_tensor(
        request.tangent_source, "tangent_source", at::kFloat,
        {count, 3}, primal.source);
    const at::Tensor* t_target = optional_tensor(
        request.tangent_target, "tangent_target", at::kFloat,
        {count, 3}, primal.source);
    const at::Tensor* t_f0_eps = optional_tensor(
        request.tangent_face0_eps_r, "tangent_face0_eps_r", at::kFloat,
        {count}, primal.source);
    const at::Tensor* t_f0_sigma = optional_tensor(
        request.tangent_face0_sigma_e, "tangent_face0_sigma_e", at::kFloat,
        {count}, primal.source);
    const at::Tensor* t_f0_gain = optional_tensor(
        request.tangent_face0_gain, "tangent_face0_gain", at::kFloat,
        {count}, primal.source);
    const at::Tensor* t_f1_eps = optional_tensor(
        request.tangent_face1_eps_r, "tangent_face1_eps_r", at::kFloat,
        {count}, primal.source);
    const at::Tensor* t_f1_sigma = optional_tensor(
        request.tangent_face1_sigma_e, "tangent_face1_sigma_e", at::kFloat,
        {count}, primal.source);
    const at::Tensor* t_f1_gain = optional_tensor(
        request.tangent_face1_gain, "tangent_face1_gain", at::kFloat,
        {count}, primal.source);
    const at::Tensor* t_v0 = optional_tensor(
        request.tangent_vertex_v0, "tangent_vertex_v0", at::kFloat,
        {count, 3}, primal.source);
    const at::Tensor* t_v1 = optional_tensor(
        request.tangent_vertex_v1, "tangent_vertex_v1", at::kFloat,
        {count, 3}, primal.source);
    const at::Tensor* t_opp0 = optional_tensor(
        request.tangent_vertex_opp0, "tangent_vertex_opp0", at::kFloat,
        {count, 3}, primal.source);
    const at::Tensor* t_opp1 = optional_tensor(
        request.tangent_vertex_opp1, "tangent_vertex_opp1", at::kFloat,
        {count, 3}, primal.source);
    TORCH_CHECK(
        (t_v0 == nullptr && t_v1 == nullptr && t_opp0 == nullptr &&
         t_opp1 == nullptr) || vertex_args.v0 != nullptr,
        "vertex tangents require the wedge vertex inputs");

    auto tangent_field_vector = at::empty(
        {count, 3}, primal.source.options().dtype(at::kComplexFloat));
    auto tangent_direction = at::empty(
        {count, 3}, primal.source.options());
    if (count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(
            primal.source.get_device()).stream();
        diffraction_wedge_jvp_kernel<<<
            launch_blocks(count), kBlockSize, 0, stream>>>(
                count,
                WEDGE_HOST_ARGS,
                opt_ptr<float>(t_source),
                opt_ptr<float>(t_target),
                opt_ptr<float>(t_f0_eps),
                opt_ptr<float>(t_f0_sigma),
                opt_ptr<float>(t_f0_gain),
                opt_ptr<float>(t_f1_eps),
                opt_ptr<float>(t_f1_sigma),
                opt_ptr<float>(t_f1_gain),
                static_cast<float>(request.tangent_frequency),
                opt_ptr<float>(t_v0),
                opt_ptr<float>(t_v1),
                opt_ptr<float>(t_opp0),
                opt_ptr<float>(t_opp1),
                tangent_field_vector.data_ptr<c10::complex<float>>(),
                tangent_direction.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {tangent_field_vector, tangent_direction};
}

#undef WEDGE_HOST_ARGS
