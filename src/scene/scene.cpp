#include <array>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <drjit/custom.h>
#include <nanobind/nanobind.h>

#include <rayd/ray.h>
#include <rayd/scene/scene.h>
#include <rayd/edge/scene_edge.h>

#include <rayd/multipath/diffraction_accumulation_ad.h>
#include <rayd/multipath/reflection_dedup.h>
#include <rayd/multipath/reflection_epc_field.h>
#include <rayd/multipath/pipelines.h>
#include <rayd/native_launch_audit.h>

namespace drjit {

template <typename T>
struct struct_support {
    using Traversable = traversable_t<T>;

    template <typename T1, typename F>
    static void apply_1(T1 &&value, F &&f) {
        auto fields = Traversable::fields(value);
        traverse_1(fields, std::forward<F>(f));
    }

    template <typename T1, typename T2, typename F>
    static void apply_2(T1 &&value_1, T2 &&value_2, F &&f) {
        auto fields_1 = Traversable::fields(value_1);
        auto fields_2 = Traversable::fields(value_2);
        traverse_2(fields_1, fields_2, std::forward<F>(f));
    }
};

} // namespace drjit

namespace rayd {

namespace {

namespace nb = nanobind;

template <typename Output, typename Input>
class RaydCustomOp : public drjit::detail::CustomOpBase {
public:
    explicit RaydCustomOp(const Input &input)
        : registered_input(drjit::detail::ad_scan(*this, input, true)) {}

    void register_output(const Output &output) {
        registered_output = drjit::detail::ad_scan(*this, output, false);
    }

protected:
    Input registered_input;
    Output registered_output;
};

struct DfrDirectAccumOpInput {
    DfrStatesAD states;
    DfrMaterialAD material;
    Vector3fAD suffix_tri_p0;
    Vector3fAD suffix_tri_face_normal;
    Vector3fAD suffix_vertices;
    Vector3i suffix_faces;
    MaskAD active;

    DRJIT_STRUCT(DfrDirectAccumOpInput,
                 states,
                 material,
                 suffix_tri_p0,
                 suffix_tri_face_normal,
                 suffix_vertices,
                 active)
};

struct DfrDirectAccumOpInputDetached {
    DfrStates states;
    DfrMaterial material;
    Vector3f suffix_tri_p0;
    Vector3f suffix_tri_face_normal;
    Vector3f suffix_vertices;
    Vector3i suffix_faces;
    Mask active;
};

DfrDirectAccumOpInputDetached detach_dfr_direct_input(
    const DfrDirectAccumOpInput &input) {
    DfrDirectAccumOpInputDetached detached;
    detached.states.count = input.states.count;
    detached.states.edge_index = detach<false>(input.states.edge_index);
    detached.states.edge_pos = detach<false>(input.states.edge_pos);
    detached.states.edge_dir = detach<false>(input.states.edge_dir);
    detached.states.edge_t_min = detach<false>(input.states.edge_t_min);
    detached.states.edge_t_max = detach<false>(input.states.edge_t_max);
    detached.states.n0 = detach<false>(input.states.n0);
    detached.states.n1 = detach<false>(input.states.n1);
    detached.states.prim0 = detach<false>(input.states.prim0);
    detached.states.prim1 = detach<false>(input.states.prim1);
    detached.states.exterior_angle = detach<false>(input.states.exterior_angle);
    detached.states.src = detach<false>(input.states.src);
    detached.states.src_power = detach<false>(input.states.src_power);
    detached.states.wi = detach<false>(input.states.wi);
    detached.states.d0 = detach<false>(input.states.d0);
    detached.states.prefix_depth = detach<false>(input.states.prefix_depth);
    detached.material.eta_r = detach<false>(input.material.eta_r);
    detached.material.sigma = detach<false>(input.material.sigma);
    detached.material.mu_r = detach<false>(input.material.mu_r);
    detached.material.gain = detach<false>(input.material.gain);
    detached.material.valid = detach<false>(input.material.valid);
    detached.suffix_tri_p0 = detach<false>(input.suffix_tri_p0);
    detached.suffix_tri_face_normal = detach<false>(input.suffix_tri_face_normal);
    detached.suffix_vertices = detach<false>(input.suffix_vertices);
    detached.suffix_faces = input.suffix_faces;
    detached.active = detach<false>(input.active);
    detached.states.count = input.states.count;
    return detached;
}

DfrStates detach_dfr_states_input(const DfrStatesAD &input) {
    DfrStates detached;
    detached.count = input.count;
    detached.edge_index = detach<false>(input.edge_index);
    detached.edge_pos = detach<false>(input.edge_pos);
    detached.edge_dir = detach<false>(input.edge_dir);
    detached.edge_t_min = detach<false>(input.edge_t_min);
    detached.edge_t_max = detach<false>(input.edge_t_max);
    detached.n0 = detach<false>(input.n0);
    detached.n1 = detach<false>(input.n1);
    detached.prim0 = detach<false>(input.prim0);
    detached.prim1 = detach<false>(input.prim1);
    detached.exterior_angle = detach<false>(input.exterior_angle);
    detached.src = detach<false>(input.src);
    detached.src_power = detach<false>(input.src_power);
    detached.wi = detach<false>(input.wi);
    detached.d0 = detach<false>(input.d0);
    detached.prefix_depth = detach<false>(input.prefix_depth);
    return detached;
}

DfrAccumAD dfr_accum_to_ad(const DfrAccum &input) {
    DfrAccumAD output;
    output.grid_cell_count = input.grid_cell_count;
    output.power = FloatAD(input.power);
    output.field_x = drjit::Complex<FloatAD>(
        FloatAD(input.field_x.x()),
        FloatAD(input.field_x.y()));
    output.field_y = drjit::Complex<FloatAD>(
        FloatAD(input.field_y.x()),
        FloatAD(input.field_y.y()));
    output.field_z = drjit::Complex<FloatAD>(
        FloatAD(input.field_z.x()),
        FloatAD(input.field_z.y()));
    output.direct_count = IntAD(input.direct_count);
    output.keller_count = IntAD(input.keller_count);
    output.suffix_count = IntAD(input.suffix_count);
    output.vis_rejects = IntAD(input.vis_rejects);
    output.edge_vis_rejects = IntAD(input.edge_vis_rejects);
    output.utd_rejects = IntAD(input.utd_rejects);
    output.edge_uses = IntAD(input.edge_uses);
    return output;
}

DfrAccum zero_dfr_accum_grad(int grid_cell_count) {
    DfrAccum output;
    output.grid_cell_count = grid_cell_count;
    output.power = zeros<Float>(grid_cell_count);
    output.field_x =
        drjit::Complex<Float>(zeros<Float>(grid_cell_count),
                              zeros<Float>(grid_cell_count));
    output.field_y =
        drjit::Complex<Float>(zeros<Float>(grid_cell_count),
                              zeros<Float>(grid_cell_count));
    output.field_z =
        drjit::Complex<Float>(zeros<Float>(grid_cell_count),
                              zeros<Float>(grid_cell_count));
    output.direct_count = full<Int>(0, 1);
    output.keller_count = full<Int>(0, 1);
    output.suffix_count = full<Int>(0, 1);
    output.vis_rejects = full<Int>(0, 1);
    output.edge_vis_rejects = full<Int>(0, 1);
    output.utd_rejects = full<Int>(0, 1);
    output.edge_uses = full<Int>(0, 1);
    return output;
}

void set_dfr_accum_output_grad(DfrAccumAD &registered_output,
                               const DfrAccum &grad_output) {
    drjit::set_grad(registered_output.power, grad_output.power);
    drjit::set_grad(registered_output.field_x.x(), grad_output.field_x.x());
    drjit::set_grad(registered_output.field_x.y(), grad_output.field_x.y());
    drjit::set_grad(registered_output.field_y.x(), grad_output.field_y.x());
    drjit::set_grad(registered_output.field_y.y(), grad_output.field_y.y());
    drjit::set_grad(registered_output.field_z.x(), grad_output.field_z.x());
    drjit::set_grad(registered_output.field_z.y(), grad_output.field_z.y());
}

struct DfrDirectTapeCapture {
    int launch_count = 0;
    Mask active;
    Int state_idx;
    Int cell;
    Int material_idx;
    Float edge_u;
};

thread_local DfrDirectTapeCapture *active_dfr_direct_tape_capture = nullptr;

class ScopedDfrDirectTapeCapture {
public:
    explicit ScopedDfrDirectTapeCapture(DfrDirectTapeCapture *capture)
        : previous_(active_dfr_direct_tape_capture) {
        active_dfr_direct_tape_capture = capture;
    }

    ~ScopedDfrDirectTapeCapture() {
        active_dfr_direct_tape_capture = previous_;
    }

private:
    DfrDirectTapeCapture *previous_ = nullptr;
};

int dfr_direct_sample_count(const DfrOptions &options) {
    return (options.strategy_mask & RAYD_DFR_DIRECT) != 0
               ? (options.direct_samples > 0 ? options.direct_samples : options.samples)
               : 0;
}

int dfr_keller_sample_count(const DfrOptions &options) {
    return (options.strategy_mask & RAYD_DFR_KELLER) != 0
               ? options.keller_samples
               : 0;
}

int dfr_suffix_sample_count(const DfrOptions &options) {
    return (options.strategy_mask & RAYD_DFR_SUFFIX_REFL) != 0
               ? options.suffix_samples
               : 0;
}

int dfr_direct_custom_ad_sample_count(const DfrOptions &options) {
    return dfr_direct_sample_count(options) +
           dfr_keller_sample_count(options) +
           dfr_suffix_sample_count(options);
}

void require_dfr_direct_custom_ad_supported(const DfrOptions &options) {
    require(options.max_order == 1,
            "Scene::accum_dfr_direct(): native AD currently supports max_order == 1.");
}

void require_dfr_chain_custom_ad_supported(const DfrOptions &options) {
    require(options.max_order == 2 || options.max_order == 3,
            "Scene::accum_dfr(): native AD currently supports max_order 2 or 3.");
}

template <typename FloatLike>
Float coerce_float_grad(const FloatLike &value, size_t width) {
    Float detached = detach<false>(value);
    return detached.size() == width ? detached : zeros<Float>(width);
}

template <typename VecLike>
Vector3f coerce_vec3_grad(const VecLike &value, size_t width) {
    Vector3f detached = detach<false>(value);
    return slices(detached) == width ? detached : zeros<Vector3f>(width);
}

struct DfrSuffixTriangleJvp {
    Vector3f p0;
    Vector3f face_normal;
};

DfrSuffixTriangleJvp dfr_suffix_triangle_jvp_from_vertices(
    const Vector3f &vertices,
    const Vector3i &faces,
    const Vector3f &dot_vertices,
    size_t triangle_width) {
    DfrSuffixTriangleJvp result;
    result.p0 = zeros<Vector3f>(triangle_width);
    result.face_normal = zeros<Vector3f>(triangle_width);
    if (triangle_width == 0 || slices(vertices) == 0 || slices(dot_vertices) == 0) {
        return result;
    }

    const Vector3f v0 = gather<Vector3f>(vertices, faces[0]);
    const Vector3f v1 = gather<Vector3f>(vertices, faces[1]);
    const Vector3f v2 = gather<Vector3f>(vertices, faces[2]);
    const Vector3f dot_v0 = gather<Vector3f>(dot_vertices, faces[0]);
    const Vector3f dot_v1 = gather<Vector3f>(dot_vertices, faces[1]);
    const Vector3f dot_v2 = gather<Vector3f>(dot_vertices, faces[2]);

    const Vector3f e1 = v1 - v0;
    const Vector3f e2 = v2 - v0;
    const Vector3f dot_e1 = dot_v1 - dot_v0;
    const Vector3f dot_e2 = dot_v2 - dot_v0;
    const Vector3f raw_normal = cross(e1, e2);
    const Vector3f dot_raw_normal = cross(dot_e1, e2) + cross(e1, dot_e2);
    const Float raw_normal_norm = norm(raw_normal);
    const Mask valid = raw_normal_norm > Epsilon;
    const Vector3f face_normal = select(valid,
                                        raw_normal / raw_normal_norm,
                                        Vector3f(0.f, 0.f, 1.f));
    result.p0 = dot_v0;
    result.face_normal = select(
        valid,
        (dot_raw_normal - face_normal * dot(face_normal, dot_raw_normal)) / raw_normal_norm,
        zeros<Vector3f>(triangle_width));
    return result;
}

Vector3f dfr_suffix_triangle_vertex_vjp(const Vector3f &vertices,
                                        const Vector3i &faces,
                                        const Vector3f &grad_tri_p0,
                                        const Vector3f &grad_tri_face_normal,
                                        size_t vertex_width) {
    Vector3f grad_vertices = zeros<Vector3f>(vertex_width);
    if (vertex_width == 0 || slices(faces[0]) == 0) {
        return grad_vertices;
    }

    const Vector3f v0 = gather<Vector3f>(vertices, faces[0]);
    const Vector3f v1 = gather<Vector3f>(vertices, faces[1]);
    const Vector3f v2 = gather<Vector3f>(vertices, faces[2]);
    const Vector3f e1 = v1 - v0;
    const Vector3f e2 = v2 - v0;
    const Vector3f raw_normal = cross(e1, e2);
    const Float raw_normal_norm = norm(raw_normal);
    const Mask valid = raw_normal_norm > Epsilon;
    const Vector3f face_normal = select(valid,
                                        raw_normal / raw_normal_norm,
                                        Vector3f(0.f, 0.f, 1.f));
    const Vector3f grad_raw_normal = select(
        valid,
        (grad_tri_face_normal -
         face_normal * dot(face_normal, grad_tri_face_normal)) / raw_normal_norm,
        zeros<Vector3f>(slices(faces[0])));
    const Vector3f grad_e1 = cross(e2, grad_raw_normal);
    const Vector3f grad_e2 = cross(grad_raw_normal, e1);
    const Vector3f grad_v0 = grad_tri_p0 - grad_e1 - grad_e2;
    const Vector3f grad_v1 = grad_e1;
    const Vector3f grad_v2 = grad_e2;

    for (int axis = 0; axis < 3; ++axis) {
        scatter_reduce(ReduceOp::Add, grad_vertices[axis], grad_v0[axis], faces[0]);
        scatter_reduce(ReduceOp::Add, grad_vertices[axis], grad_v1[axis], faces[1]);
        scatter_reduce(ReduceOp::Add, grad_vertices[axis], grad_v2[axis], faces[2]);
    }
    return grad_vertices;
}

template <typename States>
int dfr_state_count_for(const States &states) {
    const int state_width = static_cast<int>(slices(states.edge_index));
    return states.count > 0 ? states.count : state_width;
}

class DfrDirectAccumOp : public RaydCustomOp<DfrAccumAD, DfrDirectAccumOpInput> {
public:
    using Base = RaydCustomOp<DfrAccumAD, DfrDirectAccumOpInput>;
    using OutputType = DfrAccumAD;

    DfrDirectAccumOp(const DfrDirectAccumOpInput &input,
                     const Scene *scene,
                     const DfrGrid &grid,
                     const DfrOptions &options)
        : Base(input),
          scene_(scene),
          grid_(grid),
          options_(options) {}

    OutputType eval(DfrDirectAccumOpInputDetached input) {
        m_input_ = input;
        const int launch_count = dfr_direct_custom_ad_sample_count(options_);
        if (launch_count > 0) {
            m_tape_.launch_count = launch_count;
            m_tape_.active = full<Mask>(false, launch_count);
            m_tape_.state_idx = full<Int>(-1, launch_count);
            m_tape_.cell = full<Int>(-1, launch_count);
            m_tape_.material_idx = full<Int>(-1, launch_count);
            m_tape_.edge_u = zeros<Float>(launch_count);
            drjit::eval(m_tape_.active,
                        m_tape_.state_idx,
                        m_tape_.cell,
                        m_tape_.material_idx,
                        m_tape_.edge_u);
        }

        ScopedDfrDirectTapeCapture tape_scope(
            launch_count > 0 ? &m_tape_ : nullptr);
        DfrAccum primal = scene_->accum_dfr_direct<true>(
            input.states,
            grid_,
            input.material,
            options_,
            input.active);
        return dfr_accum_to_ad(primal);
    }

    void forward() override {
        const int grid_cell_count = grid_.resolution0 * grid_.resolution1;
        DfrAccum output = zero_dfr_accum_grad(grid_cell_count);
        if (m_tape_.launch_count <= 0) {
            set_dfr_accum_output_grad(this->registered_output, output);
            return;
        }

        const size_t state_width = slices(m_input_.states.edge_index);
        const size_t material_width = slices(m_input_.material.gain);
        const size_t triangle_width = slices(m_input_.suffix_tri_p0);
        const size_t vertex_width = slices(m_input_.suffix_vertices);

        const Vector3f dot_edge_pos =
            coerce_vec3_grad(drjit::grad<false>(this->registered_input.states.edge_pos), state_width);
        const Vector3f dot_edge_dir =
            coerce_vec3_grad(drjit::grad<false>(this->registered_input.states.edge_dir), state_width);
        const Float dot_edge_t_min =
            coerce_float_grad(drjit::grad<false>(this->registered_input.states.edge_t_min), state_width);
        const Float dot_edge_t_max =
            coerce_float_grad(drjit::grad<false>(this->registered_input.states.edge_t_max), state_width);
        const Vector3f dot_src =
            coerce_vec3_grad(drjit::grad<false>(this->registered_input.states.src), state_width);
        const Vector3f dot_wi =
            coerce_vec3_grad(drjit::grad<false>(this->registered_input.states.wi), state_width);
        const Float dot_src_power =
            coerce_float_grad(drjit::grad<false>(this->registered_input.states.src_power), state_width);
        const Float dot_exterior_angle =
            coerce_float_grad(drjit::grad<false>(this->registered_input.states.exterior_angle), state_width);
        const Float dot_material_gain =
            coerce_float_grad(drjit::grad<false>(this->registered_input.material.gain), material_width);
        const Vector3f dot_suffix_vertices =
            coerce_vec3_grad(drjit::grad<false>(scene_->global_geometry().vertices), vertex_width);
        const DfrSuffixTriangleJvp dot_suffix_triangles =
            dfr_suffix_triangle_jvp_from_vertices(m_input_.suffix_vertices,
                                                  m_input_.suffix_faces,
                                                  dot_suffix_vertices,
                                                  triangle_width);
        const Vector3f &dot_tri_p0 = dot_suffix_triangles.p0;
        const Vector3f &dot_tri_face_normal = dot_suffix_triangles.face_normal;

        drjit::eval(dot_edge_pos,
                    dot_edge_dir,
                    dot_edge_t_min,
                    dot_edge_t_max,
                    dot_src,
                    dot_wi,
                    dot_src_power,
                    dot_exterior_angle,
                    dot_material_gain,
                    dot_suffix_vertices,
                    dot_tri_p0,
                    dot_tri_face_normal,
                    output.power,
                    output.field_x.x());

        DfrDirectAccumADParams params = base_ad_params();
        params.dot_state_edge_pos_x = dot_edge_pos.x().data();
        params.dot_state_edge_pos_y = dot_edge_pos.y().data();
        params.dot_state_edge_pos_z = dot_edge_pos.z().data();
        params.dot_state_edge_dir_x = dot_edge_dir.x().data();
        params.dot_state_edge_dir_y = dot_edge_dir.y().data();
        params.dot_state_edge_dir_z = dot_edge_dir.z().data();
        params.dot_state_edge_t_min = dot_edge_t_min.data();
        params.dot_state_edge_t_max = dot_edge_t_max.data();
        params.dot_state_src_x = dot_src.x().data();
        params.dot_state_src_y = dot_src.y().data();
        params.dot_state_src_z = dot_src.z().data();
        params.dot_state_wi_x = dot_wi.x().data();
        params.dot_state_wi_y = dot_wi.y().data();
        params.dot_state_wi_z = dot_wi.z().data();
        params.dot_state_src_power = dot_src_power.data();
        params.dot_state_exterior_angle = dot_exterior_angle.data();
        params.dot_material_gain = dot_material_gain.data();
        params.dot_tri_p0_x = dot_tri_p0.x().data();
        params.dot_tri_p0_y = dot_tri_p0.y().data();
        params.dot_tri_p0_z = dot_tri_p0.z().data();
        params.dot_tri_fn_x = dot_tri_face_normal.x().data();
        params.dot_tri_fn_y = dot_tri_face_normal.y().data();
        params.dot_tri_fn_z = dot_tri_face_normal.z().data();
        params.dot_out_power = output.power.data();
        params.dot_out_field_x_re = output.field_x.x().data();
        dfr_direct_accum_jvp_gpu(params);
        set_dfr_accum_output_grad(this->registered_output, output);
    }

    void backward() override {
        if (m_tape_.launch_count <= 0) {
            return;
        }

        const size_t state_width = slices(m_input_.states.edge_index);
        const size_t material_width = slices(m_input_.material.gain);
        const size_t triangle_width = slices(m_input_.suffix_tri_p0);
        const size_t vertex_width = slices(m_input_.suffix_vertices);
        Vector3f grad_edge_pos = zeros<Vector3f>(state_width);
        Vector3f grad_edge_dir = zeros<Vector3f>(state_width);
        Float grad_edge_t_min = zeros<Float>(state_width);
        Float grad_edge_t_max = zeros<Float>(state_width);
        Vector3f grad_src = zeros<Vector3f>(state_width);
        Vector3f grad_wi = zeros<Vector3f>(state_width);
        Float grad_src_power = zeros<Float>(state_width);
        Float grad_exterior_angle = zeros<Float>(state_width);
        Float grad_material_gain = zeros<Float>(material_width);
        Vector3f grad_tri_p0 = zeros<Vector3f>(triangle_width);
        Vector3f grad_tri_face_normal = zeros<Vector3f>(triangle_width);
        Vector3f grad_suffix_vertices = zeros<Vector3f>(vertex_width);
        Float grad_power =
            coerce_float_grad(drjit::grad<false>(this->registered_output.power),
                              grid_.resolution0 * grid_.resolution1);
        Float grad_field_x_re =
            coerce_float_grad(drjit::grad<false>(this->registered_output.field_x.x()),
                              grid_.resolution0 * grid_.resolution1);

        drjit::eval(grad_edge_pos,
                    grad_edge_dir,
                    grad_edge_t_min,
                    grad_edge_t_max,
                    grad_src,
                    grad_wi,
                    grad_src_power,
                    grad_exterior_angle,
                    grad_material_gain,
                    grad_tri_p0,
                    grad_tri_face_normal,
                    grad_suffix_vertices,
                    grad_power,
                    grad_field_x_re);

        DfrDirectAccumADParams params = base_ad_params();
        params.grad_out_power = grad_power.data();
        params.grad_out_field_x_re = grad_field_x_re.data();
        params.grad_state_edge_pos_x = grad_edge_pos.x().data();
        params.grad_state_edge_pos_y = grad_edge_pos.y().data();
        params.grad_state_edge_pos_z = grad_edge_pos.z().data();
        params.grad_state_edge_dir_x = grad_edge_dir.x().data();
        params.grad_state_edge_dir_y = grad_edge_dir.y().data();
        params.grad_state_edge_dir_z = grad_edge_dir.z().data();
        params.grad_state_edge_t_min = grad_edge_t_min.data();
        params.grad_state_edge_t_max = grad_edge_t_max.data();
        params.grad_state_src_x = grad_src.x().data();
        params.grad_state_src_y = grad_src.y().data();
        params.grad_state_src_z = grad_src.z().data();
        params.grad_state_wi_x = grad_wi.x().data();
        params.grad_state_wi_y = grad_wi.y().data();
        params.grad_state_wi_z = grad_wi.z().data();
        params.grad_state_src_power = grad_src_power.data();
        params.grad_state_exterior_angle = grad_exterior_angle.data();
        params.grad_material_gain = grad_material_gain.data();
        params.grad_tri_p0_x = grad_tri_p0.x().data();
        params.grad_tri_p0_y = grad_tri_p0.y().data();
        params.grad_tri_p0_z = grad_tri_p0.z().data();
        params.grad_tri_fn_x = grad_tri_face_normal.x().data();
        params.grad_tri_fn_y = grad_tri_face_normal.y().data();
        params.grad_tri_fn_z = grad_tri_face_normal.z().data();
        dfr_direct_accum_vjp_gpu(params);
        grad_suffix_vertices = dfr_suffix_triangle_vertex_vjp(m_input_.suffix_vertices,
                                                              m_input_.suffix_faces,
                                                              grad_tri_p0,
                                                              grad_tri_face_normal,
                                                              vertex_width);
        drjit::eval(grad_suffix_vertices);

        drjit::accum_grad(this->registered_input.states.edge_pos,
                          drjit::detach<false>(grad_edge_pos));
        drjit::accum_grad(this->registered_input.states.edge_dir,
                          drjit::detach<false>(grad_edge_dir));
        drjit::accum_grad(this->registered_input.states.edge_t_min,
                          drjit::detach<false>(grad_edge_t_min));
        drjit::accum_grad(this->registered_input.states.edge_t_max,
                          drjit::detach<false>(grad_edge_t_max));
        drjit::accum_grad(this->registered_input.states.src,
                          drjit::detach<false>(grad_src));
        drjit::accum_grad(this->registered_input.states.wi,
                          drjit::detach<false>(grad_wi));
        drjit::accum_grad(this->registered_input.states.src_power,
                          drjit::detach<false>(grad_src_power));
        drjit::accum_grad(this->registered_input.states.exterior_angle,
                          drjit::detach<false>(grad_exterior_angle));
        drjit::accum_grad(this->registered_input.material.gain,
                          drjit::detach<false>(grad_material_gain));
        drjit::accum_grad(this->registered_input.suffix_vertices,
                          drjit::detach<false>(grad_suffix_vertices));
    }

    const char *name() const override { return "DfrDirectAccum"; }

private:
    DfrDirectAccumADParams base_ad_params() const {
        DfrDirectAccumADParams params = {};
        params.n_rays = m_tape_.launch_count;
        params.state_count = dfr_state_count_for(m_input_.states);
        params.material_count = static_cast<int>(slices(m_input_.material.gain));
        params.grid_axis = grid_.axis;
        params.grid_position = grid_.position;
        params.grid_coord0_min = grid_.coord0_min;
        params.grid_coord0_max = grid_.coord0_max;
        params.grid_coord1_min = grid_.coord1_min;
        params.grid_coord1_max = grid_.coord1_max;
        params.grid_resolution0 = grid_.resolution0;
        params.grid_resolution1 = grid_.resolution1;
        params.grid_cell_area = grid_.cell_area;
        params.direct_samples = dfr_direct_sample_count(options_);
        params.keller_samples = dfr_keller_sample_count(options_);
        params.suffix_samples = dfr_suffix_sample_count(options_);
        params.wavelength = options_.wavelength;
        params.seed = options_.seed;
        const TriangleInfo &triangles = scene_->triangle_info_detached();
        const bool suffix_enabled = params.suffix_samples > 0;
        params.n_triangles = suffix_enabled
                                 ? static_cast<int>(slices(m_input_.suffix_tri_p0))
                                 : 0;
        params.tape_active =
            reinterpret_cast<const uint8_t *>(m_tape_.active.data());
        params.tape_state_idx = m_tape_.state_idx.data();
        params.tape_cell = m_tape_.cell.data();
        params.tape_material_idx = m_tape_.material_idx.data();
        params.tape_edge_u = m_tape_.edge_u.data();
        params.state_edge_pos_x = m_input_.states.edge_pos.x().data();
        params.state_edge_pos_y = m_input_.states.edge_pos.y().data();
        params.state_edge_pos_z = m_input_.states.edge_pos.z().data();
        params.state_edge_dir_x = m_input_.states.edge_dir.x().data();
        params.state_edge_dir_y = m_input_.states.edge_dir.y().data();
        params.state_edge_dir_z = m_input_.states.edge_dir.z().data();
        params.state_edge_t_min = m_input_.states.edge_t_min.data();
        params.state_edge_t_max = m_input_.states.edge_t_max.data();
        params.state_src_x = m_input_.states.src.x().data();
        params.state_src_y = m_input_.states.src.y().data();
        params.state_src_z = m_input_.states.src.z().data();
        params.state_wi_x = m_input_.states.wi.x().data();
        params.state_wi_y = m_input_.states.wi.y().data();
        params.state_wi_z = m_input_.states.wi.z().data();
        params.state_src_power = m_input_.states.src_power.data();
        params.state_exterior_angle = m_input_.states.exterior_angle.data();
        params.state_prim0 = m_input_.states.prim0.data();
        params.state_prim1 = m_input_.states.prim1.data();
        params.tri_p0_x = suffix_enabled ? m_input_.suffix_tri_p0.x().data() : nullptr;
        params.tri_p0_y = suffix_enabled ? m_input_.suffix_tri_p0.y().data() : nullptr;
        params.tri_p0_z = suffix_enabled ? m_input_.suffix_tri_p0.z().data() : nullptr;
        params.tri_e1_x = suffix_enabled ? triangles.e1.x().data() : nullptr;
        params.tri_e1_y = suffix_enabled ? triangles.e1.y().data() : nullptr;
        params.tri_e1_z = suffix_enabled ? triangles.e1.z().data() : nullptr;
        params.tri_e2_x = suffix_enabled ? triangles.e2.x().data() : nullptr;
        params.tri_e2_y = suffix_enabled ? triangles.e2.y().data() : nullptr;
        params.tri_e2_z = suffix_enabled ? triangles.e2.z().data() : nullptr;
        params.tri_fn_x = suffix_enabled ? m_input_.suffix_tri_face_normal.x().data() : nullptr;
        params.tri_fn_y = suffix_enabled ? m_input_.suffix_tri_face_normal.y().data() : nullptr;
        params.tri_fn_z = suffix_enabled ? m_input_.suffix_tri_face_normal.z().data() : nullptr;
        params.material_gain = m_input_.material.gain.data();
        params.material_valid =
            reinterpret_cast<const uint8_t *>(m_input_.material.valid.data());
        return params;
    }

    const Scene *scene_ = nullptr;
    DfrGrid grid_;
    DfrOptions options_;
    DfrDirectAccumOpInputDetached m_input_;
    DfrDirectTapeCapture m_tape_;
};

DfrAccumAD dfr_direct_accum_custom_op(const Scene *scene,
                                      const DfrStatesAD &states,
                                      const DfrGrid &grid,
                                      const DfrMaterialAD &material,
                                      const DfrOptions &options,
                                      const Vector3fAD &suffix_tri_p0,
                                      const Vector3fAD &suffix_tri_face_normal,
                                      const Vector3fAD &suffix_vertices,
                                      const Vector3i &suffix_faces,
                                      const MaskAD &active) {
    DfrDirectAccumOpInput input;
    input.states = states;
    input.material = material;
    input.suffix_tri_p0 = suffix_tri_p0;
    input.suffix_tri_face_normal = suffix_tri_face_normal;
    input.suffix_vertices = suffix_vertices;
    input.suffix_faces = suffix_faces;
    input.active = active;
    nb::ref<DfrDirectAccumOp> op =
        new DfrDirectAccumOp(input, scene, grid, options);
    DfrAccumAD output = op->eval(detach_dfr_direct_input(input));
    drjit::detail::new_grad(output);
    op->register_output(output);
    if (!ad_custom_op(op.get())) {
        drjit::disable_grad(output);
    }
    return output;
}

struct DfrChainAccumOpInput {
    DfrStatesAD initial_states;
    DfrStatesAD recursive_states;
    DfrMaterialAD material;
    MaskAD active;
    Vector3fAD suffix_tri_p0;
    Vector3fAD suffix_tri_face_normal;
    Vector3fAD suffix_vertices;
    Vector3i suffix_faces;

    DRJIT_STRUCT(DfrChainAccumOpInput,
                 initial_states,
                 recursive_states,
                 material,
                 active,
                 suffix_tri_p0,
                 suffix_tri_face_normal,
                 suffix_vertices)
};

struct DfrChainAccumOpInputDetached {
    DfrStates initial_states;
    DfrStates recursive_states;
    DfrMaterial material;
    Mask active;
    Vector3f suffix_tri_p0;
    Vector3f suffix_tri_face_normal;
    Vector3f suffix_vertices;
    Vector3i suffix_faces;
};

DfrChainAccumOpInputDetached detach_dfr_chain_input(
    const DfrChainAccumOpInput &input) {
    DfrChainAccumOpInputDetached detached;
    detached.initial_states = detach_dfr_states_input(input.initial_states);
    detached.recursive_states = detach_dfr_states_input(input.recursive_states);
    detached.material.eta_r = detach<false>(input.material.eta_r);
    detached.material.sigma = detach<false>(input.material.sigma);
    detached.material.mu_r = detach<false>(input.material.mu_r);
    detached.material.gain = detach<false>(input.material.gain);
    detached.material.valid = detach<false>(input.material.valid);
    detached.active = detach<false>(input.active);
    detached.suffix_tri_p0 = detach<false>(input.suffix_tri_p0);
    detached.suffix_tri_face_normal = detach<false>(input.suffix_tri_face_normal);
    detached.suffix_vertices = detach<false>(input.suffix_vertices);
    detached.suffix_faces = input.suffix_faces;
    return detached;
}

class DfrChainAccumOp : public RaydCustomOp<DfrAccumAD, DfrChainAccumOpInput> {
public:
    using Base = RaydCustomOp<DfrAccumAD, DfrChainAccumOpInput>;
    using OutputType = DfrAccumAD;

    DfrChainAccumOp(const DfrChainAccumOpInput &input,
                    const Scene *scene,
                    const DfrGrid &grid,
                    const DfrOptions &options)
        : Base(input),
          scene_(scene),
          grid_(grid),
          options_(options) {}

    OutputType eval(DfrChainAccumOpInputDetached input) {
        m_input_ = input;
        const int launch_count = dfr_direct_custom_ad_sample_count(options_);
        if (launch_count > 0) {
            m_tape_.launch_count = launch_count;
            m_tape_.active = full<Mask>(false, launch_count);
            m_tape_.state_idx = full<Int>(-1, launch_count);
            m_tape_.cell = full<Int>(-1, launch_count);
            m_tape_.material_idx = full<Int>(-1, launch_count);
            m_tape_.edge_u = zeros<Float>(launch_count);
            drjit::eval(m_tape_.active,
                        m_tape_.state_idx,
                        m_tape_.cell,
                        m_tape_.material_idx,
                        m_tape_.edge_u);
        }

        ScopedDfrDirectTapeCapture tape_scope(
            launch_count > 0 ? &m_tape_ : nullptr);
        DfrAccum primal = scene_->accum_dfr<true>(
            input.initial_states,
            input.recursive_states,
            grid_,
            input.material,
            options_,
            input.active);
        return dfr_accum_to_ad(primal);
    }

    void forward() override {
        const int grid_cell_count = grid_.resolution0 * grid_.resolution1;
        DfrAccum output = zero_dfr_accum_grad(grid_cell_count);
        if (m_tape_.launch_count <= 0) {
            set_dfr_accum_output_grad(this->registered_output, output);
            return;
        }

        const size_t initial_width = slices(m_input_.initial_states.edge_index);
        const size_t recursive_width = slices(m_input_.recursive_states.edge_index);
        const size_t material_width = slices(m_input_.material.gain);
        const size_t triangle_width = slices(m_input_.suffix_tri_p0);
        const size_t vertex_width = slices(m_input_.suffix_vertices);
        const Vector3f dot_initial_edge_pos =
            coerce_vec3_grad(drjit::grad<false>(this->registered_input.initial_states.edge_pos),
                             initial_width);
        const Vector3f dot_initial_edge_dir =
            coerce_vec3_grad(drjit::grad<false>(this->registered_input.initial_states.edge_dir),
                             initial_width);
        const Float dot_initial_edge_t_min =
            coerce_float_grad(drjit::grad<false>(this->registered_input.initial_states.edge_t_min),
                              initial_width);
        const Float dot_initial_edge_t_max =
            coerce_float_grad(drjit::grad<false>(this->registered_input.initial_states.edge_t_max),
                              initial_width);
        const Vector3f dot_initial_src =
            coerce_vec3_grad(drjit::grad<false>(this->registered_input.initial_states.src),
                             initial_width);
        const Float dot_initial_src_power =
            coerce_float_grad(drjit::grad<false>(this->registered_input.initial_states.src_power),
                              initial_width);
        const Float dot_initial_exterior =
            coerce_float_grad(drjit::grad<false>(this->registered_input.initial_states.exterior_angle),
                              initial_width);
        const Vector3f dot_recursive_edge_pos =
            coerce_vec3_grad(drjit::grad<false>(this->registered_input.recursive_states.edge_pos),
                             recursive_width);
        const Vector3f dot_recursive_edge_dir =
            coerce_vec3_grad(drjit::grad<false>(this->registered_input.recursive_states.edge_dir),
                             recursive_width);
        const Float dot_recursive_edge_t_min =
            coerce_float_grad(drjit::grad<false>(this->registered_input.recursive_states.edge_t_min),
                              recursive_width);
        const Float dot_recursive_edge_t_max =
            coerce_float_grad(drjit::grad<false>(this->registered_input.recursive_states.edge_t_max),
                              recursive_width);
        const Float dot_recursive_exterior =
            coerce_float_grad(drjit::grad<false>(this->registered_input.recursive_states.exterior_angle),
                              recursive_width);
        const Float dot_material_gain =
            coerce_float_grad(drjit::grad<false>(this->registered_input.material.gain),
                              material_width);
        const Vector3f dot_suffix_vertices =
            coerce_vec3_grad(drjit::grad<false>(scene_->global_geometry().vertices), vertex_width);
        const DfrSuffixTriangleJvp dot_suffix_triangles =
            dfr_suffix_triangle_jvp_from_vertices(m_input_.suffix_vertices,
                                                  m_input_.suffix_faces,
                                                  dot_suffix_vertices,
                                                  triangle_width);
        const Vector3f &dot_suffix_tri_p0 = dot_suffix_triangles.p0;
        const Vector3f &dot_suffix_tri_face_normal = dot_suffix_triangles.face_normal;

        drjit::eval(dot_initial_edge_pos,
                    dot_initial_edge_dir,
                    dot_initial_edge_t_min,
                    dot_initial_edge_t_max,
                    dot_initial_src,
                    dot_initial_src_power,
                    dot_initial_exterior,
                    dot_recursive_edge_pos,
                    dot_recursive_edge_dir,
                    dot_recursive_edge_t_min,
                    dot_recursive_edge_t_max,
                    dot_recursive_exterior,
                    dot_material_gain,
                    dot_suffix_vertices,
                    dot_suffix_tri_p0,
                    dot_suffix_tri_face_normal,
                    output.power,
                    output.field_x.x());

        DfrChainAccumADParams params = base_ad_params();
        params.dot_state_edge_pos_x = dot_initial_edge_pos.x().data();
        params.dot_state_edge_pos_y = dot_initial_edge_pos.y().data();
        params.dot_state_edge_pos_z = dot_initial_edge_pos.z().data();
        params.dot_state_edge_dir_x = dot_initial_edge_dir.x().data();
        params.dot_state_edge_dir_y = dot_initial_edge_dir.y().data();
        params.dot_state_edge_dir_z = dot_initial_edge_dir.z().data();
        params.dot_state_edge_t_min = dot_initial_edge_t_min.data();
        params.dot_state_edge_t_max = dot_initial_edge_t_max.data();
        params.dot_state_src_x = dot_initial_src.x().data();
        params.dot_state_src_y = dot_initial_src.y().data();
        params.dot_state_src_z = dot_initial_src.z().data();
        params.dot_state_src_power = dot_initial_src_power.data();
        params.dot_state_exterior_angle = dot_initial_exterior.data();
        params.dot_recursive_state_edge_pos_x = dot_recursive_edge_pos.x().data();
        params.dot_recursive_state_edge_pos_y = dot_recursive_edge_pos.y().data();
        params.dot_recursive_state_edge_pos_z = dot_recursive_edge_pos.z().data();
        params.dot_recursive_state_edge_dir_x = dot_recursive_edge_dir.x().data();
        params.dot_recursive_state_edge_dir_y = dot_recursive_edge_dir.y().data();
        params.dot_recursive_state_edge_dir_z = dot_recursive_edge_dir.z().data();
        params.dot_recursive_state_edge_t_min = dot_recursive_edge_t_min.data();
        params.dot_recursive_state_edge_t_max = dot_recursive_edge_t_max.data();
        params.dot_recursive_state_exterior_angle = dot_recursive_exterior.data();
        params.dot_material_gain = dot_material_gain.data();
        params.dot_tri_p0_x = dot_suffix_tri_p0.x().data();
        params.dot_tri_p0_y = dot_suffix_tri_p0.y().data();
        params.dot_tri_p0_z = dot_suffix_tri_p0.z().data();
        params.dot_tri_fn_x = dot_suffix_tri_face_normal.x().data();
        params.dot_tri_fn_y = dot_suffix_tri_face_normal.y().data();
        params.dot_tri_fn_z = dot_suffix_tri_face_normal.z().data();
        params.dot_out_power = output.power.data();
        params.dot_out_field_x_re = output.field_x.x().data();
        dfr_chain_accum_jvp_gpu(params);
        set_dfr_accum_output_grad(this->registered_output, output);
    }

    void backward() override {
        if (m_tape_.launch_count <= 0) {
            return;
        }

        const size_t initial_width = slices(m_input_.initial_states.edge_index);
        const size_t recursive_width = slices(m_input_.recursive_states.edge_index);
        const size_t material_width = slices(m_input_.material.gain);
        const size_t triangle_width = slices(m_input_.suffix_tri_p0);
        const size_t vertex_width = slices(m_input_.suffix_vertices);
        Vector3f grad_initial_edge_pos = zeros<Vector3f>(initial_width);
        Vector3f grad_initial_edge_dir = zeros<Vector3f>(initial_width);
        Float grad_initial_edge_t_min = zeros<Float>(initial_width);
        Float grad_initial_edge_t_max = zeros<Float>(initial_width);
        Vector3f grad_initial_src = zeros<Vector3f>(initial_width);
        Float grad_initial_src_power = zeros<Float>(initial_width);
        Float grad_initial_exterior = zeros<Float>(initial_width);
        Vector3f grad_recursive_edge_pos = zeros<Vector3f>(recursive_width);
        Vector3f grad_recursive_edge_dir = zeros<Vector3f>(recursive_width);
        Float grad_recursive_edge_t_min = zeros<Float>(recursive_width);
        Float grad_recursive_edge_t_max = zeros<Float>(recursive_width);
        Float grad_recursive_exterior = zeros<Float>(recursive_width);
        Float grad_material_gain = zeros<Float>(material_width);
        Vector3f grad_suffix_tri_p0 = zeros<Vector3f>(triangle_width);
        Vector3f grad_suffix_tri_face_normal = zeros<Vector3f>(triangle_width);
        Vector3f grad_suffix_vertices = zeros<Vector3f>(vertex_width);
        Float grad_power =
            coerce_float_grad(drjit::grad<false>(this->registered_output.power),
                              grid_.resolution0 * grid_.resolution1);
        Float grad_field_x_re =
            coerce_float_grad(drjit::grad<false>(this->registered_output.field_x.x()),
                              grid_.resolution0 * grid_.resolution1);

        drjit::eval(grad_initial_edge_pos,
                    grad_initial_edge_dir,
                    grad_initial_edge_t_min,
                    grad_initial_edge_t_max,
                    grad_initial_src,
                    grad_initial_src_power,
                    grad_initial_exterior,
                    grad_recursive_edge_pos,
                    grad_recursive_edge_dir,
                    grad_recursive_edge_t_min,
                    grad_recursive_edge_t_max,
                    grad_recursive_exterior,
                    grad_material_gain,
                    grad_suffix_tri_p0,
                    grad_suffix_tri_face_normal,
                    grad_suffix_vertices,
                    grad_power,
                    grad_field_x_re);

        DfrChainAccumADParams params = base_ad_params();
        params.grad_out_power = grad_power.data();
        params.grad_out_field_x_re = grad_field_x_re.data();
        params.grad_state_edge_pos_x = grad_initial_edge_pos.x().data();
        params.grad_state_edge_pos_y = grad_initial_edge_pos.y().data();
        params.grad_state_edge_pos_z = grad_initial_edge_pos.z().data();
        params.grad_state_edge_dir_x = grad_initial_edge_dir.x().data();
        params.grad_state_edge_dir_y = grad_initial_edge_dir.y().data();
        params.grad_state_edge_dir_z = grad_initial_edge_dir.z().data();
        params.grad_state_edge_t_min = grad_initial_edge_t_min.data();
        params.grad_state_edge_t_max = grad_initial_edge_t_max.data();
        params.grad_state_src_x = grad_initial_src.x().data();
        params.grad_state_src_y = grad_initial_src.y().data();
        params.grad_state_src_z = grad_initial_src.z().data();
        params.grad_state_src_power = grad_initial_src_power.data();
        params.grad_state_exterior_angle = grad_initial_exterior.data();
        params.grad_recursive_state_edge_pos_x = grad_recursive_edge_pos.x().data();
        params.grad_recursive_state_edge_pos_y = grad_recursive_edge_pos.y().data();
        params.grad_recursive_state_edge_pos_z = grad_recursive_edge_pos.z().data();
        params.grad_recursive_state_edge_dir_x = grad_recursive_edge_dir.x().data();
        params.grad_recursive_state_edge_dir_y = grad_recursive_edge_dir.y().data();
        params.grad_recursive_state_edge_dir_z = grad_recursive_edge_dir.z().data();
        params.grad_recursive_state_edge_t_min = grad_recursive_edge_t_min.data();
        params.grad_recursive_state_edge_t_max = grad_recursive_edge_t_max.data();
        params.grad_recursive_state_exterior_angle = grad_recursive_exterior.data();
        params.grad_material_gain = grad_material_gain.data();
        params.grad_tri_p0_x = grad_suffix_tri_p0.x().data();
        params.grad_tri_p0_y = grad_suffix_tri_p0.y().data();
        params.grad_tri_p0_z = grad_suffix_tri_p0.z().data();
        params.grad_tri_fn_x = grad_suffix_tri_face_normal.x().data();
        params.grad_tri_fn_y = grad_suffix_tri_face_normal.y().data();
        params.grad_tri_fn_z = grad_suffix_tri_face_normal.z().data();
        dfr_chain_accum_vjp_gpu(params);
        grad_suffix_vertices = dfr_suffix_triangle_vertex_vjp(m_input_.suffix_vertices,
                                                              m_input_.suffix_faces,
                                                              grad_suffix_tri_p0,
                                                              grad_suffix_tri_face_normal,
                                                              vertex_width);
        drjit::eval(grad_suffix_vertices);

        drjit::accum_grad(this->registered_input.initial_states.edge_pos,
                          drjit::detach<false>(grad_initial_edge_pos));
        drjit::accum_grad(this->registered_input.initial_states.edge_dir,
                          drjit::detach<false>(grad_initial_edge_dir));
        drjit::accum_grad(this->registered_input.initial_states.edge_t_min,
                          drjit::detach<false>(grad_initial_edge_t_min));
        drjit::accum_grad(this->registered_input.initial_states.edge_t_max,
                          drjit::detach<false>(grad_initial_edge_t_max));
        drjit::accum_grad(this->registered_input.initial_states.src,
                          drjit::detach<false>(grad_initial_src));
        drjit::accum_grad(this->registered_input.initial_states.src_power,
                          drjit::detach<false>(grad_initial_src_power));
        drjit::accum_grad(this->registered_input.initial_states.exterior_angle,
                          drjit::detach<false>(grad_initial_exterior));
        drjit::accum_grad(this->registered_input.recursive_states.edge_pos,
                          drjit::detach<false>(grad_recursive_edge_pos));
        drjit::accum_grad(this->registered_input.recursive_states.edge_dir,
                          drjit::detach<false>(grad_recursive_edge_dir));
        drjit::accum_grad(this->registered_input.recursive_states.edge_t_min,
                          drjit::detach<false>(grad_recursive_edge_t_min));
        drjit::accum_grad(this->registered_input.recursive_states.edge_t_max,
                          drjit::detach<false>(grad_recursive_edge_t_max));
        drjit::accum_grad(this->registered_input.recursive_states.exterior_angle,
                          drjit::detach<false>(grad_recursive_exterior));
        drjit::accum_grad(this->registered_input.material.gain,
                          drjit::detach<false>(grad_material_gain));
        drjit::accum_grad(this->registered_input.suffix_vertices,
                          drjit::detach<false>(grad_suffix_vertices));
    }

    const char *name() const override { return "DfrChainAccum"; }

private:
    DfrChainAccumADParams base_ad_params() const {
        DfrChainAccumADParams params = {};
        params.n_rays = m_tape_.launch_count;
        params.state_count = dfr_state_count_for(m_input_.initial_states);
        params.recursive_state_count = dfr_state_count_for(m_input_.recursive_states);
        params.material_count = static_cast<int>(slices(m_input_.material.gain));
        params.grid_axis = grid_.axis;
        params.grid_position = grid_.position;
        params.grid_coord0_min = grid_.coord0_min;
        params.grid_coord0_max = grid_.coord0_max;
        params.grid_coord1_min = grid_.coord1_min;
        params.grid_coord1_max = grid_.coord1_max;
        params.grid_resolution0 = grid_.resolution0;
        params.grid_resolution1 = grid_.resolution1;
        params.grid_cell_area = grid_.cell_area;
        params.direct_samples = dfr_direct_sample_count(options_);
        params.keller_samples = dfr_keller_sample_count(options_);
        params.suffix_samples = dfr_suffix_sample_count(options_);
        params.max_order = options_.max_order;
        params.wavelength = options_.wavelength;
        params.seed = options_.seed;
        const TriangleInfo &triangles = scene_->triangle_info_detached();
        const bool suffix_enabled = params.suffix_samples > 0;
        params.n_triangles = suffix_enabled
                                 ? static_cast<int>(slices(m_input_.suffix_tri_p0))
                                 : 0;
        params.tape_active =
            reinterpret_cast<const uint8_t *>(m_tape_.active.data());
        params.tape_cell = m_tape_.cell.data();
        params.state_edge_index = m_input_.initial_states.edge_index.data();
        params.state_edge_pos_x = m_input_.initial_states.edge_pos.x().data();
        params.state_edge_pos_y = m_input_.initial_states.edge_pos.y().data();
        params.state_edge_pos_z = m_input_.initial_states.edge_pos.z().data();
        params.state_edge_dir_x = m_input_.initial_states.edge_dir.x().data();
        params.state_edge_dir_y = m_input_.initial_states.edge_dir.y().data();
        params.state_edge_dir_z = m_input_.initial_states.edge_dir.z().data();
        params.state_edge_t_min = m_input_.initial_states.edge_t_min.data();
        params.state_edge_t_max = m_input_.initial_states.edge_t_max.data();
        params.state_src_x = m_input_.initial_states.src.x().data();
        params.state_src_y = m_input_.initial_states.src.y().data();
        params.state_src_z = m_input_.initial_states.src.z().data();
        params.state_src_power = m_input_.initial_states.src_power.data();
        params.state_exterior_angle = m_input_.initial_states.exterior_angle.data();
        params.state_prim0 = m_input_.initial_states.prim0.data();
        params.state_prim1 = m_input_.initial_states.prim1.data();
        params.recursive_state_edge_index = m_input_.recursive_states.edge_index.data();
        params.recursive_state_edge_pos_x = m_input_.recursive_states.edge_pos.x().data();
        params.recursive_state_edge_pos_y = m_input_.recursive_states.edge_pos.y().data();
        params.recursive_state_edge_pos_z = m_input_.recursive_states.edge_pos.z().data();
        params.recursive_state_edge_dir_x = m_input_.recursive_states.edge_dir.x().data();
        params.recursive_state_edge_dir_y = m_input_.recursive_states.edge_dir.y().data();
        params.recursive_state_edge_dir_z = m_input_.recursive_states.edge_dir.z().data();
        params.recursive_state_edge_t_min = m_input_.recursive_states.edge_t_min.data();
        params.recursive_state_edge_t_max = m_input_.recursive_states.edge_t_max.data();
        params.recursive_state_exterior_angle =
            m_input_.recursive_states.exterior_angle.data();
        params.recursive_state_prim0 = m_input_.recursive_states.prim0.data();
        params.recursive_state_prim1 = m_input_.recursive_states.prim1.data();
        params.tri_p0_x = suffix_enabled ? m_input_.suffix_tri_p0.x().data() : nullptr;
        params.tri_p0_y = suffix_enabled ? m_input_.suffix_tri_p0.y().data() : nullptr;
        params.tri_p0_z = suffix_enabled ? m_input_.suffix_tri_p0.z().data() : nullptr;
        params.tri_e1_x = suffix_enabled ? triangles.e1.x().data() : nullptr;
        params.tri_e1_y = suffix_enabled ? triangles.e1.y().data() : nullptr;
        params.tri_e1_z = suffix_enabled ? triangles.e1.z().data() : nullptr;
        params.tri_e2_x = suffix_enabled ? triangles.e2.x().data() : nullptr;
        params.tri_e2_y = suffix_enabled ? triangles.e2.y().data() : nullptr;
        params.tri_e2_z = suffix_enabled ? triangles.e2.z().data() : nullptr;
        params.tri_fn_x = suffix_enabled ? m_input_.suffix_tri_face_normal.x().data() : nullptr;
        params.tri_fn_y = suffix_enabled ? m_input_.suffix_tri_face_normal.y().data() : nullptr;
        params.tri_fn_z = suffix_enabled ? m_input_.suffix_tri_face_normal.z().data() : nullptr;
        params.material_gain = m_input_.material.gain.data();
        params.material_valid =
            reinterpret_cast<const uint8_t *>(m_input_.material.valid.data());
        return params;
    }

    const Scene *scene_ = nullptr;
    DfrGrid grid_;
    DfrOptions options_;
    DfrChainAccumOpInputDetached m_input_;
    DfrDirectTapeCapture m_tape_;
};

DfrAccumAD dfr_chain_accum_custom_op(const Scene *scene,
                                     const DfrStatesAD &initial_states,
                                     const DfrStatesAD &recursive_states,
                                     const DfrGrid &grid,
                                     const DfrMaterialAD &material,
                                     const DfrOptions &options,
                                     const Vector3fAD &suffix_tri_p0,
                                     const Vector3fAD &suffix_tri_face_normal,
                                     const Vector3fAD &suffix_vertices,
                                     const Vector3i &suffix_faces,
                                     const MaskAD &active) {
    DfrChainAccumOpInput input;
    input.initial_states = initial_states;
    input.recursive_states = recursive_states;
    input.material = material;
    input.active = active;
    input.suffix_tri_p0 = suffix_tri_p0;
    input.suffix_tri_face_normal = suffix_tri_face_normal;
    input.suffix_vertices = suffix_vertices;
    input.suffix_faces = suffix_faces;
    nb::ref<DfrChainAccumOp> op =
        new DfrChainAccumOp(input, scene, grid, options);
    DfrAccumAD output = op->eval(detach_dfr_chain_input(input));
    drjit::detail::new_grad(output);
    op->register_output(output);
    if (!ad_custom_op(op.get())) {
        drjit::disable_grad(output);
    }
    return output;
}

/// Whether to split static and dynamic meshes into separate OptiX scenes (env RAYD_OPTIX_SPLIT_MODE).
enum class OptixSplitMode {
    Auto,
    Off,
    On
};

/// Backend for segment-visibility traces (env RAYD_TRACE_VISIBILITY_BACKEND): Dr.Jit HitObject vs. native optixLaunch.
enum class TraceVisibilityBackend {
    Auto,
    Jit,
    Native
};

/// How EPC visibility decides which primitives to ignore: exact primitive ids vs. surface groups.
enum class ReflEpcVisibilityIgnoreMode {
    Primitive,
    SurfaceGroup
};

std::string normalize_optix_split_mode_value(const char *value) {
    std::string normalized = value != nullptr ? std::string(value) : std::string();
    std::transform(normalized.begin(),
                   normalized.end(),
                   normalized.begin(),
                   [](unsigned char ch) -> char {
                       return static_cast<char>(std::tolower(ch));
                   });
    return normalized;
}

OptixSplitMode active_optix_split_mode() {
    static const OptixSplitMode value = []() {
        const char *raw = std::getenv("RAYD_OPTIX_SPLIT_MODE");
        const std::string normalized = normalize_optix_split_mode_value(raw);
        if (normalized.empty() || normalized == "auto") {
            return normalized.empty() ? OptixSplitMode::Off : OptixSplitMode::Auto;
        }
        if (normalized == "off" || normalized == "false" || normalized == "0") {
            return OptixSplitMode::Off;
        }
        if (normalized == "on" || normalized == "true" || normalized == "1") {
            return OptixSplitMode::On;
        }
        throw std::runtime_error(
            "Invalid RAYD_OPTIX_SPLIT_MODE. Expected one of: auto, off, on.");
    }();
    return value;
}

TraceVisibilityBackend active_trace_visibility_backend() {
    static const TraceVisibilityBackend value = []() {
        const char *raw = std::getenv("RAYD_TRACE_VISIBILITY_BACKEND");
        const std::string normalized = normalize_optix_split_mode_value(raw);
        if (normalized.empty() || normalized == "auto") {
            return TraceVisibilityBackend::Auto;
        }
        if (normalized == "jit" || normalized == "drjit" ||
            normalized == "hitobject" || normalized == "hit_object") {
            return TraceVisibilityBackend::Jit;
        }
        if (normalized == "native" || normalized == "optixlaunch" ||
            normalized == "optix_launch") {
            return TraceVisibilityBackend::Native;
        }
        throw std::runtime_error(
            "Invalid RAYD_TRACE_VISIBILITY_BACKEND. Expected one of: auto, jit, native.");
    }();
    return value;
}

ReflEpcVisibilityIgnoreMode parse_refl_epc_vis_ignore(
    const std::string &value) {
    const std::string normalized = normalize_optix_split_mode_value(value.c_str());
    if (normalized.empty() || normalized == "primitive" ||
        normalized == "prim" || normalized == "exact") {
        return ReflEpcVisibilityIgnoreMode::Primitive;
    }
    if (normalized == "surface_group" || normalized == "surface-group" ||
        normalized == "group") {
        return ReflEpcVisibilityIgnoreMode::SurfaceGroup;
    }
    throw std::runtime_error(
        "Invalid ReflEpcOptions.visibility_ignore_mode. "
        "Expected one of: 'primitive', 'surface_group'.");
}

bool use_jit_trace_visibility_path(int ignore_k) {
    const TraceVisibilityBackend backend = active_trace_visibility_backend();
    if (backend == TraceVisibilityBackend::Native) {
        return false;
    }
    if (backend == TraceVisibilityBackend::Jit) {
        require(ignore_k == 0,
                "RAYD_TRACE_VISIBILITY_BACKEND=jit does not support ignore lists yet.");
        return true;
    }
    return ignore_k == 0;
}

std::string normalize_edge_backend_value(const std::string &value) {
    std::string normalized = value;
    std::transform(normalized.begin(),
                   normalized.end(),
                   normalized.begin(),
                   [](unsigned char ch) -> char {
                       return static_cast<char>(std::tolower(ch));
                   });
    return normalized;
}

/// Map a backend name ("drjit"/"optix"/"hybrid" and aliases) to EdgeBVHBackend; throws on unknown.
EdgeBVHBackend parse_edge_backend(const std::string &value) {
    const std::string normalized = normalize_edge_backend_value(value);
    if (normalized.empty() || normalized == "optix" ||
        normalized == "custom_aabb") {
        return EdgeBVHBackend::Optix;
    }
    if (normalized == "drjit" || normalized == "dr_jit" ||
        normalized == "software") {
        return EdgeBVHBackend::DrJit;
    }
    if (normalized == "hybrid" || normalized == "mixed" ||
        normalized == "optix_ray" || normalized == "ray_optix") {
        return EdgeBVHBackend::Hybrid;
    }
    throw std::runtime_error(
        "Invalid edge_bvh_backend. Expected one of: 'drjit', 'optix', 'hybrid'.");
}

const char *edge_backend_name(EdgeBVHBackend backend) {
    switch (backend) {
    case EdgeBVHBackend::DrJit:
        return "drjit";
    case EdgeBVHBackend::Optix:
        return "optix";
    case EdgeBVHBackend::Hybrid:
        return "hybrid";
    }
    return "drjit";
}

bool edge_backend_builds_drjit(EdgeBVHBackend backend) {
    return backend == EdgeBVHBackend::DrJit ||
           backend == EdgeBVHBackend::Hybrid;
}

bool edge_backend_builds_optix(EdgeBVHBackend backend) {
    return backend == EdgeBVHBackend::Optix ||
           backend == EdgeBVHBackend::Hybrid;
}

bool edge_backend_uses_optix_point(EdgeBVHBackend backend) {
    return backend == EdgeBVHBackend::Optix;
}

bool edge_backend_uses_optix_ray(EdgeBVHBackend backend) {
    return backend == EdgeBVHBackend::Optix ||
           backend == EdgeBVHBackend::Hybrid;
}

bool edge_backend_uses_optix_topk(EdgeBVHBackend backend) {
    return backend == EdgeBVHBackend::Optix;
}

bool should_split_optix_scene(OptixSplitMode mode,
                              int static_mesh_count,
                              int dynamic_mesh_count) {
    if (static_mesh_count == 0 || dynamic_mesh_count == 0) {
        return false;
    }
    if (mode == OptixSplitMode::On) {
        return true;
    }
    if (mode == OptixSplitMode::Off) {
        return false;
    }

    // The measured mixed-scene query tax is still too large to justify enabling
    // split mode automatically. Keep "on" available for calibration, but bias
    // "auto" to the stable single-scene path until a better heuristic exists.
    return false;
}

bool uses_symbolic_optix_query_path() {
    // Dr.Jit symbolic recording cannot mix multiple OptiX pipelines/SBTs
    // within a single captured kernel. Fall back to the unified scene path.
    return jit_flag(JitFlag::Recording);
}

bool recording_reflections() {
    return jit_flag(JitFlag::Recording);
}

template <bool Detached>
NearestPointEdgeT<Detached> initialize_nearest_point_edge_result(int query_count) {
    NearestPointEdgeT<Detached> result;
    result.distance = full<FloatT<Detached>>(Infinity, query_count);
    result.point = zeros<Vector3fT<Detached>>(query_count);
    result.edge_t = zeros<FloatT<Detached>>(query_count);
    result.edge_point = zeros<Vector3fT<Detached>>(query_count);
    result.shape_id = full<IntT<Detached>>(-1, query_count);
    result.edge_id = full<IntT<Detached>>(-1, query_count);
    result.global_edge_id = full<IntT<Detached>>(-1, query_count);
    result.is_boundary = full<MaskT<Detached>>(false, query_count);
    return result;
}

template <bool Detached>
NearestRayEdgeT<Detached> initialize_nearest_ray_edge_result(int query_count) {
    NearestRayEdgeT<Detached> result;
    result.distance = full<FloatT<Detached>>(Infinity, query_count);
    result.ray_t = zeros<FloatT<Detached>>(query_count);
    result.point = zeros<Vector3fT<Detached>>(query_count);
    result.edge_t = zeros<FloatT<Detached>>(query_count);
    result.edge_point = zeros<Vector3fT<Detached>>(query_count);
    result.shape_id = full<IntT<Detached>>(-1, query_count);
    result.edge_id = full<IntT<Detached>>(-1, query_count);
    result.global_edge_id = full<IntT<Detached>>(-1, query_count);
    result.is_boundary = full<MaskT<Detached>>(false, query_count);
    return result;
}

struct ReflectionTraceRaw {
    int max_bounces = 0;
    int ray_count = 0;
    Int bounce_count;
    Int discovery_count;
    Int representative_ray_index;
    Int shape_ids;
    Int prim_ids;
    Float t;
    Float bary_u;
    Float bary_v;
    Float hit_x;
    Float hit_y;
    Float hit_z;
    Float norm_x;
    Float norm_y;
    Float norm_z;
    Float img_x;
    Float img_y;
    Float img_z;
    Float trailing_t;
    Int trailing_prim;
    Float trailing_dir_x;
    Float trailing_dir_y;
    Float trailing_dir_z;
    Float trailing_origin_x;
    Float trailing_origin_y;
    Float trailing_origin_z;
};

struct ReflEpcRaw {
    int ray_count = 0;
    int max_bounces = 0;
    Mask valid;
    Int bounce_count;
    Float path_length;
    Float point_x;
    Float point_y;
    Float point_z;
    Int trace_prim_ids;
    Int resolved_prim_ids;
    Int surface_group_ids;
    Float plane_normal_x;
    Float plane_normal_y;
    Float plane_normal_z;
    Int first_blocked_segment;
    Int first_blocked_prim;
    Int first_blocked_group;
};

struct AccumRaw {
    int ray_count = 0;
    int max_bounces = 0;
    int grid_cell_count = 0;
    int wedge_capacity = 0;
    Float reflection_power;
    Float field_x_re;
    Float field_x_im;
    Float field_y_re;
    Float field_y_im;
    Float field_z_re;
    Float field_z_im;
    Int reflection_count;
    Int wedge_count;
    Int wedge_ray_index;
    Float wedge_hit_x;
    Float wedge_hit_y;
    Float wedge_hit_z;
    Float wedge_normal_x;
    Float wedge_normal_y;
    Float wedge_normal_z;
    Int wedge_prim_id;
    Float wedge_dir_x;
    Float wedge_dir_y;
    Float wedge_dir_z;
    Float wedge_source_x;
    Float wedge_source_y;
    Float wedge_source_z;
    Float wedge_source_power;
    Float wedge_initial_dir_x;
    Float wedge_initial_dir_y;
    Float wedge_initial_dir_z;
    Int wedge_bounce_depth;
};

struct DfrAccumRaw {
    int grid_cell_count = 0;
    Float power;
    Float field_x_re;
    Float field_x_im;
    Float field_y_re;
    Float field_y_im;
    Float field_z_re;
    Float field_z_im;
    Int direct_count;
    Int keller_count;
    Int suffix_count;
    Int vis_rejects;
    Int edge_vis_rejects;
    Int utd_rejects;
    Int edge_uses;
};

struct DfrPathsRaw {
    int capacity = 0;
    Int count;
    Mask valid;
    Int tx_id;
    Int rx_id;
    Int order;
    Int edge0;
    Int edge1;
    Int edge2;
    Float delay;
    Float field_x_re;
    Float field_x_im;
    Float field_y_re;
    Float field_y_im;
    Float field_z_re;
    Float field_z_im;
    Vector3f p0;
    Vector3f p1;
    Vector3f p2;
};

/// Convert per-mesh (shape_id, local primitive id) pairs into scene-global primitive ids;
/// invalid or out-of-range inputs map to -1.
Int globalize_primitive_ids(const Int &local_prim_ids,
                                    const Int &shape_ids,
                                    const Int &face_offsets) {
    const int ray_count = static_cast<int>(slices(local_prim_ids));
    if (ray_count == 0) {
        return Int();
    }

    const int mesh_count = std::max(0, static_cast<int>(slices(face_offsets)) - 1);
    const Mask valid =
        (local_prim_ids >= 0) && (shape_ids >= 0) && (shape_ids < mesh_count);
    const Int safe_shape_ids = select(valid, shape_ids, zeros<Int>(ray_count));
    const Int mesh_face_offsets =
        gather<Int>(face_offsets, safe_shape_ids, valid);
    return select(valid,
                  local_prim_ids + mesh_face_offsets,
                  full<Int>(-1, ray_count));
}

template <bool Detached>
ReflEpcT<Detached> init_refl_epc(int ray_count,
                                                                int max_bounces) {
    ReflEpcT<Detached> result;
    result.ray_count = ray_count;
    result.max_bounces = max_bounces;
    const int slot_count = ray_count * max_bounces;
    result.valid = full<MaskT<Detached>>(false, ray_count);
    result.bounce_count = full<IntT<Detached>>(0, ray_count);
    result.path_length = full<FloatT<Detached>>(Infinity, ray_count);
    result.reflection_points = zeros<Vector3fT<Detached>>(slot_count);
    result.prim_ids = full<IntT<Detached>>(-1, slot_count);
    result.trace_prim_ids = full<IntT<Detached>>(-1, slot_count);
    result.resolved_prim_ids = full<IntT<Detached>>(-1, slot_count);
    result.surface_group_ids = full<IntT<Detached>>(-1, slot_count);
    result.plane_normals = zeros<Vector3fT<Detached>>(slot_count);
    result.first_blocked_segment = full<IntT<Detached>>(-1, ray_count);
    result.first_blocked_prim = full<IntT<Detached>>(-1, ray_count);
    result.first_blocked_group = full<IntT<Detached>>(-1, ray_count);
    return result;
}

ReflEpcRaw alloc_refl_epc_raw(int ray_count, int max_bounces) {
    const int slot_count = ray_count * max_bounces;
    ReflEpcRaw raw;
    raw.ray_count = ray_count;
    raw.max_bounces = max_bounces;
    raw.valid = empty<Mask>(ray_count);
    raw.bounce_count = empty<Int>(ray_count);
    raw.path_length = empty<Float>(ray_count);
    raw.point_x = empty<Float>(slot_count);
    raw.point_y = empty<Float>(slot_count);
    raw.point_z = empty<Float>(slot_count);
    raw.trace_prim_ids = empty<Int>(slot_count);
    raw.resolved_prim_ids = empty<Int>(slot_count);
    raw.surface_group_ids = empty<Int>(slot_count);
    raw.plane_normal_x = empty<Float>(slot_count);
    raw.plane_normal_y = empty<Float>(slot_count);
    raw.plane_normal_z = empty<Float>(slot_count);
    raw.first_blocked_segment = empty<Int>(ray_count);
    raw.first_blocked_prim = empty<Int>(ray_count);
    raw.first_blocked_group = empty<Int>(ray_count);
    return raw;
}

template <bool Detached>
ReflEpcFieldT<Detached> init_refl_epc_field(
    int ray_count,
    int max_bounces,
    const ReflEpcFieldOptionsT<Detached> &options) {
    ReflEpcFieldT<Detached> result;
    result.ray_count = ray_count;
    result.max_bounces = max_bounces;
    const int slot_count = ray_count * max_bounces;
    const bool return_geom = options.return_geom;
    const bool return_endpoints = options.return_endpoints;
    const bool return_hit_points =
        return_geom && options.return_hit_points;
    const bool return_normals = return_geom && options.return_normals;
    const bool return_resolved_prim_ids =
        return_geom && options.return_resolved_prim_ids;
    const bool return_surface_group_ids =
        return_geom && options.return_surface_group_ids;

    result.valid = empty<MaskT<Detached>>(ray_count);
    result.bounce_count = empty<IntT<Detached>>(ray_count);
    result.path_length = empty<FloatT<Detached>>(ray_count);
    result.field_x_re = empty<FloatT<Detached>>(ray_count);
    result.field_x_im = empty<FloatT<Detached>>(ray_count);
    result.field_y_re = empty<FloatT<Detached>>(ray_count);
    result.field_y_im = empty<FloatT<Detached>>(ray_count);
    result.field_z_re = empty<FloatT<Detached>>(ray_count);
    result.field_z_im = empty<FloatT<Detached>>(ray_count);

    if (return_endpoints) {
        result.tx_pos =
            Vector3fT<Detached>(empty<FloatT<Detached>>(ray_count),
                                empty<FloatT<Detached>>(ray_count),
                                empty<FloatT<Detached>>(ray_count));
        result.first_hit =
            Vector3fT<Detached>(empty<FloatT<Detached>>(ray_count),
                                empty<FloatT<Detached>>(ray_count),
                                empty<FloatT<Detached>>(ray_count));
        result.last_hit =
            Vector3fT<Detached>(empty<FloatT<Detached>>(ray_count),
                                empty<FloatT<Detached>>(ray_count),
                                empty<FloatT<Detached>>(ray_count));
    } else {
        result.tx_pos = zeros<Vector3fT<Detached>>(0);
        result.first_hit = zeros<Vector3fT<Detached>>(0);
        result.last_hit = zeros<Vector3fT<Detached>>(0);
    }

    if (return_hit_points) {
        result.hit_points =
            Vector3fT<Detached>(empty<FloatT<Detached>>(slot_count),
                                empty<FloatT<Detached>>(slot_count),
                                empty<FloatT<Detached>>(slot_count));
    } else {
        result.hit_points = zeros<Vector3fT<Detached>>(0);
    }
    if (return_normals) {
        result.normals =
            Vector3fT<Detached>(empty<FloatT<Detached>>(slot_count),
                                empty<FloatT<Detached>>(slot_count),
                                empty<FloatT<Detached>>(slot_count));
    } else {
        result.normals = zeros<Vector3fT<Detached>>(0);
    }
    if (return_resolved_prim_ids) {
        result.resolved_prim_ids = empty<IntT<Detached>>(slot_count);
    } else {
        result.resolved_prim_ids = full<IntT<Detached>>(-1, 0);
    }
    if (return_surface_group_ids) {
        result.surface_group_ids = empty<IntT<Detached>>(slot_count);
    } else {
        result.surface_group_ids = full<IntT<Detached>>(-1, 0);
    }

    return result;
}

ReflEpcOptions epc_options_from_field_options(
    const ReflEpcFieldOptions &options) {
    ReflEpcOptions epc_options;
    epc_options.expected_prim_ids = options.expected_prim_ids;
    epc_options.surface_group_id = options.surface_group_id;
    epc_options.surface_group_size = options.surface_group_size;
    epc_options.surface_group_members = options.surface_group_members;
    epc_options.surface_max_group_size = options.surface_max_group_size;
    epc_options.visibility_ignore_mode = options.visibility_ignore_mode;
    epc_options.final_ignore_group_ids = options.final_ignore_group_ids;
    return epc_options;
}

template <bool Detached>
ReflectionChainT<Detached> initialize_reflection_chain_result(int ray_count,
                                                              int max_bounces) {
    ReflectionChainT<Detached> result;
    result.max_bounces = max_bounces;
    result.ray_count = ray_count;

    const int slot_count = ray_count * max_bounces;
    result.bounce_count = full<IntT<Detached>>(0, ray_count);
    result.discovery_count = full<IntT<Detached>>(0, ray_count);
    result.representative_ray_index = full<IntT<Detached>>(-1, ray_count);
    result.t = full<FloatT<Detached>>(Infinity, slot_count);
    result.hit_points = zeros<Vector3fT<Detached>>(slot_count);
    result.geo_normals = zeros<Vector3fT<Detached>>(slot_count);
    result.image_sources = zeros<Vector3fT<Detached>>(slot_count);
    result.plane_points = zeros<Vector3fT<Detached>>(slot_count);
    result.plane_normals = zeros<Vector3fT<Detached>>(slot_count);
    result.shape_ids = full<IntT<Detached>>(-1, slot_count);
    result.prim_ids = full<IntT<Detached>>(-1, slot_count);
    result.local_prim_ids = full<IntT<Detached>>(-1, slot_count);
    result.global_prim_ids = full<IntT<Detached>>(-1, slot_count);
    result.trailing_t = full<FloatT<Detached>>(Infinity, ray_count);
    result.trailing_prim = full<IntT<Detached>>(-1, ray_count);
    result.trailing_dir = zeros<Vector3fT<Detached>>(ray_count);
    result.trailing_origin = zeros<Vector3fT<Detached>>(ray_count);
    return result;
}

template <bool Detached>
ReflectionBounceT<Detached> initialize_reflection_bounce_result(int ray_count) {
    ReflectionBounceT<Detached> result;
    result.t = full<FloatT<Detached>>(Infinity, ray_count);
    result.hit_points = zeros<Vector3fT<Detached>>(ray_count);
    result.geo_normals = zeros<Vector3fT<Detached>>(ray_count);
    result.image_sources = zeros<Vector3fT<Detached>>(ray_count);
    result.plane_points = zeros<Vector3fT<Detached>>(ray_count);
    result.plane_normals = zeros<Vector3fT<Detached>>(ray_count);
    result.shape_ids = full<IntT<Detached>>(-1, ray_count);
    result.prim_ids = full<IntT<Detached>>(-1, ray_count);
    result.local_prim_ids = full<IntT<Detached>>(-1, ray_count);
    result.global_prim_ids = full<IntT<Detached>>(-1, ray_count);
    return result;
}

template <bool Detached>
ReflectionTraceT<Detached> initialize_reflection_trace_result(
    int ray_count,
    int max_bounces) {
    ReflectionTraceT<Detached> result;
    result.max_bounces = max_bounces;
    result.ray_count = ray_count;
    result.bounce_count = full<IntT<Detached>>(0, ray_count);
    result.discovery_count = full<IntT<Detached>>(0, ray_count);
    result.representative_ray_index = full<IntT<Detached>>(-1, ray_count);
    result.dedup_keep_mask = full<MaskT<Detached>>(false, ray_count);
    result.bounces.reserve(static_cast<size_t>(max_bounces));
    return result;
}

ReflectionTraceRaw allocate_reflection_trace_raw(int ray_count, int max_bounces) {
    const int slot_count = ray_count * max_bounces;

    ReflectionTraceRaw raw;
    raw.max_bounces = max_bounces;
    raw.ray_count = ray_count;
    raw.bounce_count = empty<Int>(ray_count);
    raw.discovery_count = empty<Int>(ray_count);
    raw.representative_ray_index = empty<Int>(ray_count);
    raw.shape_ids = empty<Int>(slot_count);
    raw.prim_ids = empty<Int>(slot_count);
    raw.t = empty<Float>(slot_count);
    raw.bary_u = empty<Float>(slot_count);
    raw.bary_v = empty<Float>(slot_count);
    raw.hit_x = empty<Float>(slot_count);
    raw.hit_y = empty<Float>(slot_count);
    raw.hit_z = empty<Float>(slot_count);
    raw.norm_x = empty<Float>(slot_count);
    raw.norm_y = empty<Float>(slot_count);
    raw.norm_z = empty<Float>(slot_count);
    raw.img_x = empty<Float>(slot_count);
    raw.img_y = empty<Float>(slot_count);
    raw.img_z = empty<Float>(slot_count);
    raw.trailing_t = empty<Float>(ray_count);
    raw.trailing_prim = empty<Int>(ray_count);
    raw.trailing_dir_x = empty<Float>(ray_count);
    raw.trailing_dir_y = empty<Float>(ray_count);
    raw.trailing_dir_z = empty<Float>(ray_count);
    raw.trailing_origin_x = empty<Float>(ray_count);
    raw.trailing_origin_y = empty<Float>(ray_count);
    raw.trailing_origin_z = empty<Float>(ray_count);
    return raw;
}

void initialize_reflection_trace_raw(ReflectionTraceRaw &raw) {
    const int ray_count = raw.ray_count;
    const int slot_count = raw.ray_count * raw.max_bounces;
    const int zero_i = 0;
    const int minus_one_i = -1;
    const float zero_f = 0.f;
    const float inf_f = Infinity;

    jit_memset_async(JitBackend::CUDA, raw.bounce_count.data(), ray_count, sizeof(int), &zero_i);
    jit_memset_async(JitBackend::CUDA, raw.discovery_count.data(), ray_count, sizeof(int), &zero_i);
    jit_memset_async(JitBackend::CUDA,
                     raw.representative_ray_index.data(),
                     ray_count,
                     sizeof(int),
                     &minus_one_i);
    jit_memset_async(JitBackend::CUDA, raw.shape_ids.data(), slot_count, sizeof(int), &minus_one_i);
    jit_memset_async(JitBackend::CUDA, raw.prim_ids.data(), slot_count, sizeof(int), &minus_one_i);
    jit_memset_async(JitBackend::CUDA, raw.t.data(), slot_count, sizeof(float), &inf_f);
    jit_memset_async(JitBackend::CUDA, raw.bary_u.data(), slot_count, sizeof(float), &zero_f);
    jit_memset_async(JitBackend::CUDA, raw.bary_v.data(), slot_count, sizeof(float), &zero_f);
    jit_memset_async(JitBackend::CUDA, raw.hit_x.data(), slot_count, sizeof(float), &zero_f);
    jit_memset_async(JitBackend::CUDA, raw.hit_y.data(), slot_count, sizeof(float), &zero_f);
    jit_memset_async(JitBackend::CUDA, raw.hit_z.data(), slot_count, sizeof(float), &zero_f);
    jit_memset_async(JitBackend::CUDA, raw.norm_x.data(), slot_count, sizeof(float), &zero_f);
    jit_memset_async(JitBackend::CUDA, raw.norm_y.data(), slot_count, sizeof(float), &zero_f);
    jit_memset_async(JitBackend::CUDA, raw.norm_z.data(), slot_count, sizeof(float), &zero_f);
    jit_memset_async(JitBackend::CUDA, raw.img_x.data(), slot_count, sizeof(float), &zero_f);
    jit_memset_async(JitBackend::CUDA, raw.img_y.data(), slot_count, sizeof(float), &zero_f);
    jit_memset_async(JitBackend::CUDA, raw.img_z.data(), slot_count, sizeof(float), &zero_f);
    jit_memset_async(JitBackend::CUDA, raw.trailing_t.data(), ray_count, sizeof(float), &inf_f);
    jit_memset_async(JitBackend::CUDA, raw.trailing_prim.data(), ray_count, sizeof(int), &minus_one_i);
    jit_memset_async(JitBackend::CUDA, raw.trailing_dir_x.data(), ray_count, sizeof(float), &zero_f);
    jit_memset_async(JitBackend::CUDA, raw.trailing_dir_y.data(), ray_count, sizeof(float), &zero_f);
    jit_memset_async(JitBackend::CUDA, raw.trailing_dir_z.data(), ray_count, sizeof(float), &zero_f);
    jit_memset_async(JitBackend::CUDA, raw.trailing_origin_x.data(), ray_count, sizeof(float), &zero_f);
    jit_memset_async(JitBackend::CUDA, raw.trailing_origin_y.data(), ray_count, sizeof(float), &zero_f);
    jit_memset_async(JitBackend::CUDA, raw.trailing_origin_z.data(), ray_count, sizeof(float), &zero_f);
}

AccumRaw allocate_reflection_accumulation_raw(int ray_count,
                                                               int max_bounces,
                                                               int grid_cell_count,
                                                               int wedge_capacity) {
    AccumRaw raw;
    raw.ray_count = ray_count;
    raw.max_bounces = max_bounces;
    raw.grid_cell_count = grid_cell_count;
    raw.wedge_capacity = wedge_capacity;
    raw.reflection_power = empty<Float>(grid_cell_count);
    raw.field_x_re = empty<Float>(grid_cell_count);
    raw.field_x_im = empty<Float>(grid_cell_count);
    raw.field_y_re = empty<Float>(grid_cell_count);
    raw.field_y_im = empty<Float>(grid_cell_count);
    raw.field_z_re = empty<Float>(grid_cell_count);
    raw.field_z_im = empty<Float>(grid_cell_count);
    raw.reflection_count = empty<Int>(1);
    raw.wedge_count = empty<Int>(1);
    const int event_count = std::max(1, wedge_capacity);
    raw.wedge_ray_index = empty<Int>(event_count);
    raw.wedge_hit_x = empty<Float>(event_count);
    raw.wedge_hit_y = empty<Float>(event_count);
    raw.wedge_hit_z = empty<Float>(event_count);
    raw.wedge_normal_x = empty<Float>(event_count);
    raw.wedge_normal_y = empty<Float>(event_count);
    raw.wedge_normal_z = empty<Float>(event_count);
    raw.wedge_prim_id = empty<Int>(event_count);
    raw.wedge_dir_x = empty<Float>(event_count);
    raw.wedge_dir_y = empty<Float>(event_count);
    raw.wedge_dir_z = empty<Float>(event_count);
    raw.wedge_source_x = empty<Float>(event_count);
    raw.wedge_source_y = empty<Float>(event_count);
    raw.wedge_source_z = empty<Float>(event_count);
    raw.wedge_source_power = empty<Float>(event_count);
    raw.wedge_initial_dir_x = empty<Float>(event_count);
    raw.wedge_initial_dir_y = empty<Float>(event_count);
    raw.wedge_initial_dir_z = empty<Float>(event_count);
    raw.wedge_bounce_depth = empty<Int>(event_count);
    return raw;
}

void initialize_reflection_accumulation_raw(AccumRaw &raw) {
    const int zero_i = 0;
    const int minus_one_i = -1;
    const float zero_f = 0.f;
    const int event_count = std::max(1, raw.wedge_capacity);

    jit_memset_async(JitBackend::CUDA,
                     raw.reflection_power.data(),
                     raw.grid_cell_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.field_x_re.data(),
                     raw.grid_cell_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.field_x_im.data(),
                     raw.grid_cell_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.field_y_re.data(),
                     raw.grid_cell_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.field_y_im.data(),
                     raw.grid_cell_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.field_z_re.data(),
                     raw.grid_cell_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.field_z_im.data(),
                     raw.grid_cell_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA, raw.reflection_count.data(), 1, sizeof(int), &zero_i);
    jit_memset_async(JitBackend::CUDA, raw.wedge_count.data(), 1, sizeof(int), &zero_i);
    jit_memset_async(JitBackend::CUDA,
                     raw.wedge_ray_index.data(),
                     event_count,
                     sizeof(int),
                     &minus_one_i);
    jit_memset_async(JitBackend::CUDA,
                     raw.wedge_hit_x.data(),
                     event_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.wedge_hit_y.data(),
                     event_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.wedge_hit_z.data(),
                     event_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.wedge_normal_x.data(),
                     event_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.wedge_normal_y.data(),
                     event_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.wedge_normal_z.data(),
                     event_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.wedge_prim_id.data(),
                     event_count,
                     sizeof(int),
                     &minus_one_i);
    jit_memset_async(JitBackend::CUDA,
                     raw.wedge_dir_x.data(),
                     event_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.wedge_dir_y.data(),
                     event_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.wedge_dir_z.data(),
                     event_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.wedge_source_x.data(),
                     event_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.wedge_source_y.data(),
                     event_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.wedge_source_z.data(),
                     event_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.wedge_source_power.data(),
                     event_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.wedge_initial_dir_x.data(),
                     event_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.wedge_initial_dir_y.data(),
                     event_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.wedge_initial_dir_z.data(),
                     event_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.wedge_bounce_depth.data(),
                     event_count,
                     sizeof(int),
                     &minus_one_i);
}

DfrAccumRaw alloc_dfr_accum_raw(int grid_cell_count) {
    DfrAccumRaw raw;
    raw.grid_cell_count = grid_cell_count;
    raw.power = empty<Float>(grid_cell_count);
    raw.field_x_re = empty<Float>(grid_cell_count);
    raw.field_x_im = empty<Float>(grid_cell_count);
    raw.field_y_re = empty<Float>(grid_cell_count);
    raw.field_y_im = empty<Float>(grid_cell_count);
    raw.field_z_re = empty<Float>(grid_cell_count);
    raw.field_z_im = empty<Float>(grid_cell_count);
    raw.direct_count = empty<Int>(1);
    raw.keller_count = empty<Int>(1);
    raw.suffix_count = empty<Int>(1);
    raw.vis_rejects = empty<Int>(1);
    raw.edge_vis_rejects = empty<Int>(1);
    raw.utd_rejects = empty<Int>(1);
    raw.edge_uses = empty<Int>(1);
    return raw;
}

void init_dfr_accum_raw(DfrAccumRaw &raw) {
    const int zero_i = 0;
    const float zero_f = 0.f;
    jit_memset_async(JitBackend::CUDA,
                     raw.power.data(),
                     raw.grid_cell_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.field_x_re.data(),
                     raw.grid_cell_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.field_x_im.data(),
                     raw.grid_cell_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.field_y_re.data(),
                     raw.grid_cell_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.field_y_im.data(),
                     raw.grid_cell_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.field_z_re.data(),
                     raw.grid_cell_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA,
                     raw.field_z_im.data(),
                     raw.grid_cell_count,
                     sizeof(float),
                     &zero_f);
    jit_memset_async(JitBackend::CUDA, raw.direct_count.data(), 1, sizeof(int), &zero_i);
    jit_memset_async(JitBackend::CUDA, raw.keller_count.data(), 1, sizeof(int), &zero_i);
    jit_memset_async(JitBackend::CUDA, raw.suffix_count.data(), 1, sizeof(int), &zero_i);
    jit_memset_async(JitBackend::CUDA,
                     raw.vis_rejects.data(),
                     1,
                     sizeof(int),
                     &zero_i);
    jit_memset_async(JitBackend::CUDA,
                     raw.edge_vis_rejects.data(),
                     1,
                     sizeof(int),
                     &zero_i);
    jit_memset_async(JitBackend::CUDA, raw.utd_rejects.data(), 1, sizeof(int), &zero_i);
    jit_memset_async(JitBackend::CUDA, raw.edge_uses.data(), 1, sizeof(int), &zero_i);
}

DfrPathsRaw alloc_dfr_paths_raw(int capacity) {
    DfrPathsRaw raw;
    raw.capacity = capacity;
    raw.count = empty<Int>(1);
    raw.valid = empty<Mask>(capacity);
    raw.tx_id = empty<Int>(capacity);
    raw.rx_id = empty<Int>(capacity);
    raw.order = empty<Int>(capacity);
    raw.edge0 = empty<Int>(capacity);
    raw.edge1 = empty<Int>(capacity);
    raw.edge2 = empty<Int>(capacity);
    raw.delay = empty<Float>(capacity);
    raw.field_x_re = empty<Float>(capacity);
    raw.field_x_im = empty<Float>(capacity);
    raw.field_y_re = empty<Float>(capacity);
    raw.field_y_im = empty<Float>(capacity);
    raw.field_z_re = empty<Float>(capacity);
    raw.field_z_im = empty<Float>(capacity);
    raw.p0 =
        Vector3f(empty<Float>(capacity), empty<Float>(capacity), empty<Float>(capacity));
    raw.p1 =
        Vector3f(empty<Float>(capacity), empty<Float>(capacity), empty<Float>(capacity));
    raw.p2 =
        Vector3f(empty<Float>(capacity), empty<Float>(capacity), empty<Float>(capacity));
    return raw;
}

void init_dfr_paths_raw(DfrPathsRaw &raw) {
    const int zero_i = 0;
    const int minus_one_i = -1;
    const uint8_t zero_b = 0u;
    const float zero_f = 0.f;
    jit_memset_async(JitBackend::CUDA, raw.count.data(), 1, sizeof(int), &zero_i);
    jit_memset_async(JitBackend::CUDA, raw.valid.data(), raw.capacity, sizeof(uint8_t), &zero_b);
    jit_memset_async(JitBackend::CUDA, raw.tx_id.data(), raw.capacity, sizeof(int), &minus_one_i);
    jit_memset_async(JitBackend::CUDA, raw.rx_id.data(), raw.capacity, sizeof(int), &minus_one_i);
    jit_memset_async(JitBackend::CUDA, raw.order.data(), raw.capacity, sizeof(int), &zero_i);
    jit_memset_async(JitBackend::CUDA, raw.edge0.data(), raw.capacity, sizeof(int), &minus_one_i);
    jit_memset_async(JitBackend::CUDA, raw.edge1.data(), raw.capacity, sizeof(int), &minus_one_i);
    jit_memset_async(JitBackend::CUDA, raw.edge2.data(), raw.capacity, sizeof(int), &minus_one_i);
    jit_memset_async(JitBackend::CUDA, raw.delay.data(), raw.capacity, sizeof(float), &zero_f);
    jit_memset_async(JitBackend::CUDA, raw.field_x_re.data(), raw.capacity, sizeof(float), &zero_f);
    jit_memset_async(JitBackend::CUDA, raw.field_x_im.data(), raw.capacity, sizeof(float), &zero_f);
    jit_memset_async(JitBackend::CUDA, raw.field_y_re.data(), raw.capacity, sizeof(float), &zero_f);
    jit_memset_async(JitBackend::CUDA, raw.field_y_im.data(), raw.capacity, sizeof(float), &zero_f);
    jit_memset_async(JitBackend::CUDA, raw.field_z_re.data(), raw.capacity, sizeof(float), &zero_f);
    jit_memset_async(JitBackend::CUDA, raw.field_z_im.data(), raw.capacity, sizeof(float), &zero_f);
    jit_memset_async(JitBackend::CUDA, raw.p0.x().data(), raw.capacity, sizeof(float), &zero_f);
    jit_memset_async(JitBackend::CUDA, raw.p0.y().data(), raw.capacity, sizeof(float), &zero_f);
    jit_memset_async(JitBackend::CUDA, raw.p0.z().data(), raw.capacity, sizeof(float), &zero_f);
    jit_memset_async(JitBackend::CUDA, raw.p1.x().data(), raw.capacity, sizeof(float), &zero_f);
    jit_memset_async(JitBackend::CUDA, raw.p1.y().data(), raw.capacity, sizeof(float), &zero_f);
    jit_memset_async(JitBackend::CUDA, raw.p1.z().data(), raw.capacity, sizeof(float), &zero_f);
    jit_memset_async(JitBackend::CUDA, raw.p2.x().data(), raw.capacity, sizeof(float), &zero_f);
    jit_memset_async(JitBackend::CUDA, raw.p2.y().data(), raw.capacity, sizeof(float), &zero_f);
    jit_memset_async(JitBackend::CUDA, raw.p2.z().data(), raw.capacity, sizeof(float), &zero_f);
}

template <typename ArrayD>
ArrayD prefix_array(const ArrayD &value, int count) {
    return gather<ArrayD>(value, arange<Int>(count));
}

template <typename ArrayD>
ArrayD concat_array_sequence(const std::vector<ArrayD> &parts) {
    require(!parts.empty(),
            "concat_array_sequence(): at least one array is required.");
    ArrayD result = parts.front();
    for (size_t i = 1; i < parts.size(); ++i) {
        result = concat(result, parts[i]);
    }
    return result;
}

Int reflection_trace_ray_major_indices(int ray_count, int max_bounces) {
    const Int slot = arange<Int>(ray_count * max_bounces);
    const Int ray_index = slot / Int(max_bounces);
    const Int bounce_index = slot - ray_index * Int(max_bounces);
    return bounce_index * Int(ray_count) + ray_index;
}

template <bool Detached>
Mask sanitize_reflection_active(const RayT<Detached> &ray,
                                        MaskT<Detached> active) {
    Mask active_detached;
    if constexpr (!Detached) {
        active_detached = detach<false>(active);
        active_detached &= drjit::isfinite(detach<false>(ray.o.x())) &&
                           drjit::isfinite(detach<false>(ray.o.y())) &&
                           drjit::isfinite(detach<false>(ray.o.z()));
        active_detached &= drjit::isfinite(detach<false>(ray.d.x())) &&
                           drjit::isfinite(detach<false>(ray.d.y())) &&
                           drjit::isfinite(detach<false>(ray.d.z()));
        active_detached &= squared_norm(Vector3f(detach<false>(ray.d.x()),
                                                        detach<false>(ray.d.y()),
                                                        detach<false>(ray.d.z()))) > 0.f;
        active_detached &= ~drjit::isfinite(detach<false>(ray.tmax)) ||
                           (detach<false>(ray.tmax) > 0.f);
    } else {
        active_detached = active;
        active_detached &= drjit::isfinite(ray.o.x()) &&
                           drjit::isfinite(ray.o.y()) &&
                           drjit::isfinite(ray.o.z());
        active_detached &= drjit::isfinite(ray.d.x()) &&
                           drjit::isfinite(ray.d.y()) &&
                           drjit::isfinite(ray.d.z());
        active_detached &= squared_norm(ray.d) > 0.f;
        active_detached &= ~drjit::isfinite(ray.tmax) || (ray.tmax > 0.f);
    }
    return active_detached;
}

template <bool Detached>
Mask sanitize_segment_active(const Vector3fT<Detached> &start,
                                     const Vector3fT<Detached> &end,
                                     MaskT<Detached> active) {
    Mask active_detached;
    if constexpr (!Detached) {
        active_detached = detach<false>(active);
        active_detached &= drjit::isfinite(detach<false>(start.x())) &&
                           drjit::isfinite(detach<false>(start.y())) &&
                           drjit::isfinite(detach<false>(start.z()));
        active_detached &= drjit::isfinite(detach<false>(end.x())) &&
                           drjit::isfinite(detach<false>(end.y())) &&
                           drjit::isfinite(detach<false>(end.z()));
    } else {
        active_detached = active;
        active_detached &= drjit::isfinite(start.x()) &&
                           drjit::isfinite(start.y()) &&
                           drjit::isfinite(start.z());
        active_detached &= drjit::isfinite(end.x()) &&
                           drjit::isfinite(end.y()) &&
                           drjit::isfinite(end.z());
    }
    return active_detached;
}

void ensure_pipeline(std::shared_ptr<OptixLaunchPipeline> &pipeline,
                     OptixDeviceContext context,
                     int hitgroup_record_count,
                     const OptixPipelineConfig &config) {
    if (!pipeline) {
        pipeline = shared_optix_launch_pipeline(context, hitgroup_record_count, config);
    }
}

void eval_segment_visibility_common(const Vector3f &start,
                                    const Int &face_offsets,
                                    const Int &ignore_prim_ids,
                                    int ignore_k,
                                    const Mask &active_detached) {
    if (ignore_k > 0) {
        drjit::eval(start, face_offsets, ignore_prim_ids, active_detached);
    } else {
        drjit::eval(start, face_offsets, active_detached);
    }
}

SegmentVisibilityParams make_segment_visibility_params(
    const OptixScene &optix_scene,
    const Int &face_offsets,
    int mesh_count,
    const Vector3f &start,
    const Int &ignore_prim_ids,
    int ignore_k,
    const Mask &active_detached,
    int ray_count) {
    SegmentVisibilityParams params = {};
    params.handle = optix_scene.ias_handle();
    params.face_offsets = face_offsets.data();
    params.n_meshes = mesh_count;
    params.start_x = start.x().data();
    params.start_y = start.y().data();
    params.start_z = start.z().data();
    params.ignore_prim_ids = ignore_k > 0 ? ignore_prim_ids.data() : nullptr;
    params.ignore_k = ignore_k;
    params.active_mask = reinterpret_cast<const uint8_t *>(active_detached.data());
    params.n_rays = ray_count;
    return params;
}

Mask launch_segment_visibility_detached(
    const OptixScene &optix_scene,
    const OptixLaunchPipeline &pipeline,
    const Int &face_offsets,
    int mesh_count,
    const Vector3f &start,
    const Vector3f &end,
    const Int &ignore_prim_ids,
    int ignore_k,
    const Mask &active_detached) {
    const int ray_count = static_cast<int>(slices(start));
    if (ray_count == 0) {
        return Mask();
    }

    Mask visible = empty<Mask>(ray_count);
    eval_segment_visibility_common(start, face_offsets, ignore_prim_ids, ignore_k, active_detached);
    drjit::eval(end);

    SegmentVisibilityParams params =
        make_segment_visibility_params(optix_scene,
                                       face_offsets,
                                       mesh_count,
                                       start,
                                       ignore_prim_ids,
                                       ignore_k,
                                       active_detached,
                                       ray_count);
    params.end_x = end.x().data();
    params.end_y = end.y().data();
    params.end_z = end.z().data();
    params.out_visible = reinterpret_cast<uint8_t *>(visible.data());
    pipeline.launch(0, params);
    return visible;
}

template <bool Detached>
SegmentVisibilityT<Detached> trace_segment_visibility_jit_no_ignore(
    const OptixScene &optix_scene,
    const Vector3f &start,
    const Vector3f &end,
    const Mask &active_detached) {
    const int ray_count = static_cast<int>(slices(start));
    SegmentVisibilityT<Detached> result;
    result.ray_count = ray_count;

    const OptixSegmentHit hit =
        optix_scene.segment_hit<true>(start, end, active_detached);
    if constexpr (!Detached) {
        result.visible = MaskAD(hit.visible);
    } else {
        result.visible = hit.visible;
    }
    return result;
}

template <bool Detached>
SegmentVisibilityT<Detached> trace_segment_visibility_native(
    const OptixScene &optix_scene,
    const OptixLaunchPipeline &pipeline,
    const Int &face_offsets,
    int mesh_count,
    const Vector3f &start,
    const Vector3f &end,
    const Int &ignore_prim_ids,
    int ignore_k,
    const Mask &active_detached) {
    const int ray_count = static_cast<int>(slices(start));
    SegmentVisibilityT<Detached> result;
    result.ray_count = ray_count;

    const Mask visible_detached =
        launch_segment_visibility_detached(optix_scene,
                                           pipeline,
                                           face_offsets,
                                           mesh_count,
                                           start,
                                           end,
                                           ignore_prim_ids,
                                           ignore_k,
                                           active_detached);
    if constexpr (!Detached) {
        result.visible = MaskAD(visible_detached);
    } else {
        result.visible = visible_detached;
    }
    return result;
}

template <bool Detached>
SegmentPairVisibilityT<Detached> trace_segment_pair_visibility_jit_no_ignore(
    const OptixScene &optix_scene,
    const Vector3f &start,
    const Vector3f &end_a,
    const Vector3f &end_b,
    const Mask &active_detached) {
    const int ray_count = static_cast<int>(slices(start));
    SegmentPairVisibilityT<Detached> result;
    result.ray_count = ray_count;

    const OptixSegmentHit hit_a =
        optix_scene.segment_hit<true>(start, end_a, active_detached);
    const OptixSegmentHit hit_b =
        optix_scene.segment_hit<true>(start, end_b, active_detached);
    if constexpr (!Detached) {
        result.visible_a = MaskAD(hit_a.visible);
        result.visible_b = MaskAD(hit_b.visible);
    } else {
        result.visible_a = hit_a.visible;
        result.visible_b = hit_b.visible;
    }
    return result;
}

template <bool Detached>
SegmentPairVisibilityT<Detached> trace_segment_pair_visibility_native(
    const OptixScene &optix_scene,
    const OptixLaunchPipeline &pipeline,
    const Int &face_offsets,
    int mesh_count,
    const Vector3f &start,
    const Vector3f &end_a,
    const Vector3f &end_b,
    const Int &ignore_prim_ids,
    int ignore_k,
    const Mask &active_detached) {
    const int ray_count = static_cast<int>(slices(start));
    SegmentPairVisibilityT<Detached> result;
    result.ray_count = ray_count;

    Mask visible_a = empty<Mask>(ray_count);
    Mask visible_b = empty<Mask>(ray_count);
    eval_segment_visibility_common(start, face_offsets, ignore_prim_ids, ignore_k, active_detached);
    drjit::eval(end_a, end_b);

    SegmentVisibilityParams params =
        make_segment_visibility_params(optix_scene,
                                       face_offsets,
                                       mesh_count,
                                       start,
                                       ignore_prim_ids,
                                       ignore_k,
                                       active_detached,
                                       ray_count);
    params.end_x = end_a.x().data();
    params.end_y = end_a.y().data();
    params.end_z = end_a.z().data();
    params.end_b_x = end_b.x().data();
    params.end_b_y = end_b.y().data();
    params.end_b_z = end_b.z().data();
    params.out_visible = reinterpret_cast<uint8_t *>(visible_a.data());
    params.out_visible_b = reinterpret_cast<uint8_t *>(visible_b.data());
    pipeline.launch(0, params);

    if constexpr (!Detached) {
        result.visible_a = MaskAD(visible_a);
        result.visible_b = MaskAD(visible_b);
    } else {
        result.visible_a = visible_a;
        result.visible_b = visible_b;
    }
    return result;
}

template <bool Detached>
AxialEdgeVisibilityT<Detached> trace_axial_edge_visibility_jit(
    const OptixScene &optix_scene,
    const Vector3f &src,
    const Vector3f &edge_pos,
    const Vector3f &edge_dir,
    const Float &edge_t_min,
    const Float &edge_t_max,
    const std::vector<float> &sample_fractions,
    const Mask &active_detached) {
    const int state_count = static_cast<int>(slices(src));
    AxialEdgeVisibilityT<Detached> result;
    result.state_count = state_count;

    Mask any_visible = full<Mask>(false, state_count);
    const Float span =
        maximum(edge_t_max - edge_t_min, Float(0.f));
    for (float fraction : sample_fractions) {
        const Float sample_t = edge_t_min + fraction * span;
        const Vector3f sample_pos = edge_pos + sample_t * edge_dir;
        const OptixSegmentHit hit =
            optix_scene.segment_hit<true>(src, sample_pos, active_detached);
        any_visible = any_visible || hit.visible;
    }

    if constexpr (!Detached) {
        result.any_visible = MaskAD(any_visible);
    } else {
        result.any_visible = any_visible;
    }
    return result;
}

template <bool Detached>
AxialEdgeVisibilityT<Detached> trace_axial_edge_visibility_native(
    const OptixScene &optix_scene,
    const OptixLaunchPipeline &pipeline,
    const Int &face_offsets,
    int mesh_count,
    const Vector3f &src,
    const Vector3f &edge_pos,
    const Vector3f &edge_dir,
    const Float &edge_t_min,
    const Float &edge_t_max,
    const std::vector<float> &sample_fractions,
    const Mask &active_detached) {
    const int state_count = static_cast<int>(slices(src));
    AxialEdgeVisibilityT<Detached> result;
    result.state_count = state_count;

    Mask any_visible = empty<Mask>(state_count);
    drjit::eval(src,
                edge_pos,
                edge_dir,
                edge_t_min,
                edge_t_max,
                face_offsets,
                active_detached);

    SegmentVisibilityParams params =
        make_segment_visibility_params(optix_scene,
                                       face_offsets,
                                       mesh_count,
                                       src,
                                       Int(),
                                       0,
                                       active_detached,
                                       state_count);
    params.end_x = edge_pos.x().data();
    params.end_y = edge_pos.y().data();
    params.end_z = edge_pos.z().data();
    params.edge_dir_x = edge_dir.x().data();
    params.edge_dir_y = edge_dir.y().data();
    params.edge_dir_z = edge_dir.z().data();
    params.edge_t_min = edge_t_min.data();
    params.edge_t_max = edge_t_max.data();
    params.sample_count = static_cast<int>(sample_fractions.size());
    for (size_t i = 0; i < sample_fractions.size(); ++i) {
        params.sample_fractions[i] = sample_fractions[i];
    }
    params.out_visible = reinterpret_cast<uint8_t *>(any_visible.data());
    pipeline.launch(0, params);

    if constexpr (!Detached) {
        result.any_visible = MaskAD(any_visible);
    } else {
        result.any_visible = any_visible;
    }
    return result;
}

template <bool Detached>
SegmentChainVisibilityT<Detached> trace_segment_chain_visibility_jit_no_ignore(
    const OptixScene &optix_scene,
    const Vector3f &points,
    const Int &chain_length,
    int chain_count,
    int max_points,
    int max_segments,
    const Mask &active_detached) {
    SegmentChainVisibilityT<Detached> result;
    result.chain_count = chain_count;
    result.max_segments = max_segments;

    const Int chain_index = arange<Int>(chain_count);
    const Int chain_base = chain_index * max_points;
    Mask all_visible = active_detached;
    Int first_blocked_segment = full<Int>(-1, chain_count);
    Int first_blocked_prim = full<Int>(-1, chain_count);

    for (int segment = 0; segment < max_segments; ++segment) {
        const Mask segment_active =
            active_detached && all_visible && (chain_length > segment);
        const Int start_index = chain_base + segment;
        const Vector3f start_point =
            gather<Vector3f>(points, start_index, segment_active);
        const Vector3f end_point =
            gather<Vector3f>(points, start_index + 1, segment_active);
        const OptixSegmentHit hit =
            optix_scene.segment_hit<true>(start_point, end_point, segment_active);
        const Mask blocked = segment_active && !hit.visible;
        all_visible &= !blocked;
        first_blocked_segment =
            select(blocked, Int(segment), first_blocked_segment);
        first_blocked_prim =
            select(blocked, hit.global_prim_id, first_blocked_prim);
    }

    if constexpr (!Detached) {
        result.all_visible = MaskAD(all_visible);
        result.first_blocked_segment = IntAD(first_blocked_segment);
        result.first_blocked_prim = IntAD(first_blocked_prim);
    } else {
        result.all_visible = all_visible;
        result.first_blocked_segment = first_blocked_segment;
        result.first_blocked_prim = first_blocked_prim;
    }
    return result;
}

template <bool Detached>
SegmentChainVisibilityT<Detached> trace_segment_chain_visibility_native(
    const OptixScene &optix_scene,
    const OptixLaunchPipeline &pipeline,
    const Int &face_offsets,
    int mesh_count,
    const Vector3f &points,
    const Int &chain_length,
    const Int &ignore_prim_per_segment,
    int ignore_k,
    int chain_count,
    int max_points,
    int max_segments,
    const Mask &active_detached) {
    SegmentChainVisibilityT<Detached> result;
    result.chain_count = chain_count;
    result.max_segments = max_segments;

    Mask all_visible = empty<Mask>(chain_count);
    Int first_blocked_segment = empty<Int>(chain_count);
    Int first_blocked_prim = empty<Int>(chain_count);
    if (ignore_k > 0) {
        drjit::eval(points,
                    chain_length,
                    ignore_prim_per_segment,
                    face_offsets,
                    active_detached);
    } else {
        drjit::eval(points, chain_length, face_offsets, active_detached);
    }

    SegmentVisibilityParams params = {};
    params.handle = optix_scene.ias_handle();
    params.face_offsets = face_offsets.data();
    params.n_meshes = mesh_count;
    params.chain_point_x = points.x().data();
    params.chain_point_y = points.y().data();
    params.chain_point_z = points.z().data();
    params.chain_length = chain_length.data();
    params.max_points = max_points;
    params.max_segments = max_segments;
    params.ignore_prim_ids = ignore_k > 0 ? ignore_prim_per_segment.data() : nullptr;
    params.ignore_k = ignore_k;
    params.active_mask = reinterpret_cast<const uint8_t *>(active_detached.data());
    params.n_rays = chain_count;
    params.out_visible = reinterpret_cast<uint8_t *>(all_visible.data());
    params.out_first_blocked_segment = first_blocked_segment.data();
    params.out_first_blocked_prim = first_blocked_prim.data();
    pipeline.launch(0, params);

    if constexpr (!Detached) {
        result.all_visible = MaskAD(all_visible);
        result.first_blocked_segment = IntAD(first_blocked_segment);
        result.first_blocked_prim = IntAD(first_blocked_prim);
    } else {
        result.all_visible = all_visible;
        result.first_blocked_segment = first_blocked_segment;
        result.first_blocked_prim = first_blocked_prim;
    }
    return result;
}

template <bool Detached>
ReflectionTraceT<Detached> trace_bounces_impl(
    const Scene &scene,
    const RayT<Detached> &ray,
    int max_bounces,
    const ReflectionTraceOptions &options,
    MaskT<Detached> active) {
    require(!options.deduplicate,
            "Scene::trace_reflections(): deduplicate=true is not implemented with symbolic=true yet.");

    const int ray_count = static_cast<int>(slices(ray.o));
    ReflectionTraceT<Detached> result =
        initialize_reflection_trace_result<Detached>(ray_count, max_bounces);
    result.deduplicate_requested = options.deduplicate;
    if (ray_count == 0) {
        return result;
    }

    const Mask sanitized_active_detached =
        sanitize_reflection_active<Detached>(ray, active);

    RayT<Detached> current_ray = ray;
    MaskT<Detached> current_active;
    if constexpr (Detached) {
        current_active = sanitized_active_detached;
        result.representative_ray_index = arange<Int>(ray_count);
    } else {
        current_active = MaskAD(sanitized_active_detached);
        result.representative_ray_index = IntAD(arange<Int>(ray_count));
    }
    Vector3fT<Detached> current_image_source = ray.o;

    const FloatT<Detached> miss_t = full<FloatT<Detached>>(Infinity, ray_count);
    const IntT<Detached> miss_id = full<IntT<Detached>>(-1, ray_count);
    const Vector3fT<Detached> zero_v = zeros<Vector3fT<Detached>>(ray_count);
    const IntT<Detached> one_i = full<IntT<Detached>>(1, ray_count);
    const IntT<Detached> zero_i = full<IntT<Detached>>(0, ray_count);

    for (int bounce = 0; bounce < max_bounces; ++bounce) {
        const IntersectionT<Detached> its =
            scene.template intersect<Detached>(current_ray, current_active, RayFlags::Geometric);
        const MaskT<Detached> bounce_hit = current_active && its.is_valid();

        Vector3fT<Detached> geo_normal = its.geo_n;
        geo_normal = select(dot(current_ray.d, geo_normal) > 0.f, -geo_normal, geo_normal);
        const FloatT<Detached> plane_distance =
            dot(current_image_source - its.p, geo_normal);
        const Vector3fT<Detached> reflected_image_source =
            current_image_source - 2.f * plane_distance * geo_normal;

        ReflectionBounceT<Detached> bounce_result =
            initialize_reflection_bounce_result<Detached>(ray_count);
        bounce_result.t = select(bounce_hit, its.t, miss_t);
        bounce_result.hit_points = select(bounce_hit, its.p, zero_v);
        bounce_result.geo_normals = select(bounce_hit, geo_normal, zero_v);
        bounce_result.image_sources = select(bounce_hit, reflected_image_source, zero_v);
        bounce_result.plane_points = bounce_result.hit_points;
        bounce_result.plane_normals = bounce_result.geo_normals;
        bounce_result.shape_ids = select(bounce_hit, its.shape_id, miss_id);
        bounce_result.prim_ids = select(bounce_hit, its.prim_id, miss_id);
        bounce_result.local_prim_ids = select(bounce_hit, its.local_prim_id, miss_id);
        bounce_result.global_prim_ids = select(bounce_hit, its.global_prim_id, miss_id);
        result.bounces.push_back(std::move(bounce_result));

        result.bounce_count += select(bounce_hit, one_i, zero_i);

        const FloatT<Detached> ray_dot_normal = dot(current_ray.d, geo_normal);
        const Vector3fT<Detached> reflected_direction =
            current_ray.d - 2.f * ray_dot_normal * geo_normal;
        current_ray.o = select(bounce_hit,
                               its.p + Epsilon * reflected_direction,
                               current_ray.o);
        current_ray.d = select(bounce_hit, reflected_direction, current_ray.d);
        current_ray.tmax = select(bounce_hit,
                                  full<FloatT<Detached>>(Infinity, ray_count),
                                  current_ray.tmax);
        current_image_source =
            select(bounce_hit, reflected_image_source, current_image_source);
        current_active = bounce_hit;
    }

    result.dedup_keep_mask = result.bounce_count > 0;
    result.discovery_count = select(result.dedup_keep_mask, one_i, zero_i);
    result.representative_ray_index =
        select(result.dedup_keep_mask,
               result.representative_ray_index,
               full<IntT<Detached>>(-1, ray_count));
    return result;
}

int face_edge_slot(const std::array<int, 3> &face_vertices, int v0, int v1) {
    auto matches = [v0, v1](int a, int b) {
        return (a == v0 && b == v1) || (a == v1 && b == v0);
    };

    if (matches(face_vertices[0], face_vertices[1])) {
        return 0;
    }
    if (matches(face_vertices[1], face_vertices[2])) {
        return 1;
    }
    if (matches(face_vertices[2], face_vertices[0])) {
        return 2;
    }
    return -1;
}

int face_opposite_vertex(const std::array<int, 3> &face_vertices, int v0, int v1) {
    for (int vertex : face_vertices) {
        if (vertex != v0 && vertex != v1) {
            return vertex;
        }
    }
    return -1;
}

} // namespace

void Scene::reset_multipath_pipelines() {
    reflection_pipeline_.reset();
    reflection_accumulation_pipeline_.reset();
    diffraction_accumulation_pipeline_.reset();
    diffraction_paths_pipeline_.reset();
    reflection_epc_pipeline_.reset();
    reflection_epc_geometry_ready_ = false;
    segment_visibility_pipeline_.reset();
    segment_pair_visibility_pipeline_.reset();
    axial_edge_visibility_pipeline_.reset();
    segment_chain_visibility_pipeline_.reset();
}

void Scene::prewarm_path_multipath_pipelines(int hitgroup_record_count) {
    const int record_count = std::max(1, hitgroup_record_count);
    OptixDeviceContext context = jit_optix_context();
    ensure_pipeline(segment_visibility_pipeline_, context,
                    record_count, segment_visibility_pipeline_config());
    ensure_pipeline(segment_pair_visibility_pipeline_, context,
                    record_count, segment_pair_visibility_pipeline_config());
    ensure_pipeline(axial_edge_visibility_pipeline_, context,
                    record_count, axial_edge_visibility_pipeline_config());
    ensure_pipeline(segment_chain_visibility_pipeline_, context,
                    record_count, segment_chain_visibility_pipeline_config());
    ensure_pipeline(reflection_accumulation_pipeline_, context,
                    record_count, reflection_accumulation_pipeline_config());
    ensure_pipeline(reflection_epc_pipeline_, context,
                    record_count, reflection_epc_pipeline_config());
    ensure_pipeline(diffraction_accumulation_pipeline_, context,
                    record_count, diffraction_accumulation_pipeline_config());
    ensure_pipeline(diffraction_paths_pipeline_, context,
                    record_count, diffraction_paths_pipeline_config());
}

Scene::Scene(const std::string &edge_bvh_backend)
    : optix_scene_(std::make_unique<OptixScene>()),
      optix_static_scene_(std::make_unique<OptixScene>()),
      optix_dynamic_scene_(std::make_unique<OptixScene>()),
      edge_bvh_(std::make_unique<SceneEdge>()),
      edge_optix_(std::make_unique<SceneEdgeOptix>()),
      edge_bvh_backend_(parse_edge_backend(edge_bvh_backend)) {}

Scene::~Scene() = default;

std::string Scene::to_string() const {
    std::stringstream stream;
    stream << "Scene[num_meshes=" << mesh_count_
           << ", ready=" << is_ready()
           << ", pending_updates=" << pending_updates_
           << "]";
    return stream.str();
}

std::vector<const Mesh *> Scene::meshes() const {
    std::vector<const Mesh *> result;
    result.reserve(mesh_records_.size());
    for (const SceneMeshRecord &record : mesh_records_) {
        result.push_back(record.mesh.get());
    }
    return result;
}

int Scene::add_mesh(const Mesh &mesh, bool dynamic) {
    SceneMeshRecord record;
    record.mesh = std::make_unique<Mesh>(mesh);
    record.mesh->set_mesh_id(static_cast<int>(mesh_records_.size()));
    record.dynamic = dynamic;
    mesh_records_.push_back(std::move(record));

    mesh_count_ = static_cast<int>(mesh_records_.size());
    is_ready_ = false;
    pending_updates_ = false;
    vertex_offsets_ = Int();
    global_geometry_ = SceneGeometry();
    edge_mask_ = Mask();
    pending_edge_bvh_dirty_ranges_.clear();
    edge_bvh_dirty_ = false;
    mask_dirty_ = false;
    optix_split_active_ = false;
    optix_static_mesh_indices_.clear();
    optix_dynamic_mesh_indices_.clear();
    optix_dynamic_mesh_local_index_.clear();
    reset_multipath_pipelines();
    return mesh_count_ - 1;
}

Scene::SceneMeshRecord &Scene::mesh_record(int mesh_id) {
    require(mesh_id >= 0 && mesh_id < static_cast<int>(mesh_records_.size()),
            "Scene: mesh_id is out of range.");
    return mesh_records_[static_cast<size_t>(mesh_id)];
}

const Scene::SceneMeshRecord &Scene::mesh_record(int mesh_id) const {
    require(mesh_id >= 0 && mesh_id < static_cast<int>(mesh_records_.size()),
            "Scene: mesh_id is out of range.");
    return mesh_records_[static_cast<size_t>(mesh_id)];
}

void Scene::scatter_mesh_data(const SceneMeshRecord &record, bool include_static) {
    const Mesh &mesh = *record.mesh;
    const int mesh_face_count = mesh.face_count();
    if (mesh_face_count == 0) {
        return;
    }

    const TriangleInfoAD *mesh_triangle_info = mesh.triangle_info();
    const IntAD scatter_indices = arange<IntAD>(mesh_face_count) + record.face_offset;
    const Int scatter_indices_detached =
        arange<Int>(mesh_face_count) + record.face_offset;

    scatter(triangle_info_.p0, mesh_triangle_info->p0, scatter_indices);
    scatter(triangle_info_.e1, mesh_triangle_info->e1, scatter_indices);
    scatter(triangle_info_.e2, mesh_triangle_info->e2, scatter_indices);
    scatter(triangle_info_.n0, mesh_triangle_info->n0, scatter_indices);
    scatter(triangle_info_.n1, mesh_triangle_info->n1, scatter_indices);
    scatter(triangle_info_.n2, mesh_triangle_info->n2, scatter_indices);
    scatter(triangle_info_.face_normal, mesh_triangle_info->face_normal, scatter_indices);
    scatter(triangle_info_.face_area, mesh_triangle_info->face_area, scatter_indices);

    scatter(triangle_info_detached_.p0, detach<false>(mesh_triangle_info->p0), scatter_indices_detached);
    scatter(triangle_info_detached_.e1, detach<false>(mesh_triangle_info->e1), scatter_indices_detached);
    scatter(triangle_info_detached_.e2, detach<false>(mesh_triangle_info->e2), scatter_indices_detached);
    scatter(triangle_info_detached_.n0, detach<false>(mesh_triangle_info->n0), scatter_indices_detached);
    scatter(triangle_info_detached_.n1, detach<false>(mesh_triangle_info->n1), scatter_indices_detached);
    scatter(triangle_info_detached_.n2, detach<false>(mesh_triangle_info->n2), scatter_indices_detached);
    scatter(triangle_info_detached_.face_normal,
            detach<false>(mesh_triangle_info->face_normal),
            scatter_indices_detached);
    scatter(triangle_info_detached_.face_area,
            detach<false>(mesh_triangle_info->face_area),
            scatter_indices_detached);

    if (!include_static) {
        return;
    }

    scatter(triangle_info_.face_indices, mesh_triangle_info->face_indices, scatter_indices);
    scatter(triangle_info_detached_.face_indices,
            detach<false>(mesh_triangle_info->face_indices),
            scatter_indices_detached);
    scatter(triangle_face_normal_mask_,
            full<MaskAD>(mesh.use_face_normals(), mesh_face_count),
            scatter_indices);
    scatter(triangle_face_normal_mask_detached_,
            full<Mask>(mesh.use_face_normals(), mesh_face_count),
            scatter_indices_detached);

    if (mesh.has_uv() && mesh.triangle_uv() != nullptr) {
        scatter(triangle_uv_[0], (*mesh.triangle_uv())[0], scatter_indices);
        scatter(triangle_uv_[1], (*mesh.triangle_uv())[1], scatter_indices);
        scatter(triangle_uv_[2], (*mesh.triangle_uv())[2], scatter_indices);

        scatter(triangle_uv_detached_[0], detach<false>((*mesh.triangle_uv())[0]), scatter_indices_detached);
        scatter(triangle_uv_detached_[1], detach<false>((*mesh.triangle_uv())[1]), scatter_indices_detached);
        scatter(triangle_uv_detached_[2], detach<false>((*mesh.triangle_uv())[2]), scatter_indices_detached);
    }
}

void Scene::scatter_mesh_edge_data(const SceneMeshRecord &record, bool include_static_ids) {
    const Mesh &mesh = *record.mesh;
    const SecondaryEdgeInfoAD *mesh_edge_info = mesh.secondary_edge_info();
    const int mesh_edge_count = mesh_edge_info != nullptr ? mesh_edge_info->size() : 0;
    if (mesh_edge_count == 0) {
        return;
    }

    const IntAD scatter_indices = arange<IntAD>(mesh_edge_count) + record.edge_offset;
    scatter(edge_info_.start, mesh_edge_info->start, scatter_indices);
    scatter(edge_info_.edge, mesh_edge_info->edge, scatter_indices);
    scatter(edge_info_.normal0, mesh_edge_info->normal0, scatter_indices);
    scatter(edge_info_.normal1, mesh_edge_info->normal1, scatter_indices);
    scatter(edge_info_.opposite, mesh_edge_info->opposite, scatter_indices);
    scatter(edge_info_.is_boundary, mesh_edge_info->is_boundary, scatter_indices);

    if (!include_static_ids) {
        return;
    }

    const Int scatter_indices_detached = arange<Int>(mesh_edge_count) + record.edge_offset;
    scatter(edge_shape_ids_,
            full<Int>(mesh.mesh_id(), mesh_edge_count),
            scatter_indices_detached);
    scatter(edge_local_ids_,
            arange<Int>(mesh_edge_count),
            scatter_indices_detached);
}

void Scene::ensure_scene_edge_data_ready() const {
    if (!edge_bvh_dirty_) {
        return;
    }

    for (const SceneMeshRecord &record : mesh_records_) {
        if (!record.edge_dirty) {
            continue;
        }

        const_cast<Scene *>(this)->scatter_mesh_edge_data(record, false);
        record.edge_dirty = false;
    }

    ensure_edge_bvh_ready();
}

void Scene::ensure_edge_bvh_ready() const {
    if (!edge_bvh_dirty_) {
        return;
    }

    Scene *scene = const_cast<Scene *>(this);
    if (mask_dirty_) {
        if (edge_backend_builds_drjit(edge_bvh_backend_)) {
            scene->edge_bvh_->set_mask(scene->edge_mask_);
            if (edge_backend_builds_optix(edge_bvh_backend_)) {
                scene->edge_bvh_->materialize();
            }
        }
        if (edge_backend_builds_optix(edge_bvh_backend_)) {
            scene->edge_optix_->set_mask(scene->edge_mask_);
            if (edge_backend_builds_drjit(edge_bvh_backend_)) {
                scene->edge_bvh_->materialize();
            }
        }
        scene->mask_dirty_ = false;
    }

    if (pending_edge_bvh_dirty_ranges_.empty()) {
        scene->edge_bvh_dirty_ = false;
        return;
    }

    if (edge_backend_builds_drjit(edge_bvh_backend_)) {
        scene->edge_bvh_->refit(scene->edge_info_, scene->pending_edge_bvh_dirty_ranges_);
        if (edge_backend_builds_optix(edge_bvh_backend_)) {
            scene->edge_bvh_->materialize();
        }
    }
    if (edge_backend_builds_optix(edge_bvh_backend_)) {
        scene->edge_optix_->refit(scene->edge_info_, scene->pending_edge_bvh_dirty_ranges_);
        if (edge_backend_builds_drjit(edge_bvh_backend_)) {
            scene->edge_bvh_->materialize();
        }
    }
    scene->pending_edge_bvh_dirty_ranges_.clear();
    scene->edge_bvh_dirty_ = false;
}

void Scene::ensure_reflection_epc_geometry_ready() const {
    if (reflection_epc_geometry_ready_) {
        return;
    }

    drjit::eval(triangle_info_detached_.p0,
                triangle_info_detached_.e1,
                triangle_info_detached_.e2,
                triangle_info_detached_.face_normal,
                face_offsets_);
    reflection_epc_geometry_ready_ = true;
}

void Scene::build() {
    ScopedNativeLaunchStage native_launch_stage(NativeLaunchStage::Build);
    require(!mesh_records_.empty(), "Scene::build(): missing meshes.");

    std::vector<int> face_offsets;
    face_offsets.reserve(mesh_records_.size() + 1);
    face_offsets.push_back(0);

    std::vector<int> vertex_offsets;
    vertex_offsets.reserve(mesh_records_.size() + 1);
    vertex_offsets.push_back(0);

    std::vector<int> edge_offsets;
    edge_offsets.reserve(mesh_records_.size() + 1);
    edge_offsets.push_back(0);

    std::vector<OptixSceneMeshDesc> mesh_descs;
    mesh_descs.reserve(mesh_records_.size());

    std::vector<int> topology_v0;
    std::vector<int> topology_v1;
    std::vector<int> topology_v0_global;
    std::vector<int> topology_v1_global;
    std::vector<int> topology_face0_local;
    std::vector<int> topology_face1_local;
    std::vector<int> topology_face0_global;
    std::vector<int> topology_face1_global;
    std::vector<int> topology_opposite0;
    std::vector<int> topology_opposite1;
    std::vector<int> topology_opposite0_global;
    std::vector<int> topology_opposite1_global;

    for (size_t mesh_index = 0; mesh_index < mesh_records_.size(); ++mesh_index) {
        SceneMeshRecord &record = mesh_records_[mesh_index];
        Mesh &mesh = *record.mesh;
        mesh.set_mesh_id(static_cast<int>(mesh_index));
        mesh.build();
        record.vertex_offset = vertex_offsets.back();
        record.face_offset = face_offsets.back();
        const SecondaryEdgeInfoAD *mesh_edge_info = mesh.secondary_edge_info();
        const int mesh_edge_count = mesh_edge_info != nullptr ? mesh_edge_info->size() : 0;
        record.edge_offset = edge_offsets.back();
        record.vertices_dirty = false;
        record.transform_dirty = false;
        record.edge_dirty = false;

        vertex_offsets.push_back(vertex_offsets.back() + mesh.vertex_count());
        face_offsets.push_back(face_offsets.back() + mesh.face_count());
        edge_offsets.push_back(edge_offsets.back() + mesh_edge_count);
        mesh_descs.push_back({ &mesh, record.dynamic, record.face_offset, static_cast<int>(mesh_index) });
    }

    mesh_count_ = static_cast<int>(mesh_records_.size());
    const int total_vertex_count = vertex_offsets.back();
    const int total_face_count = face_offsets.back();
    require(total_face_count > 0, "Scene::build(): scene has no triangles.");

    edge_count_ = edge_offsets.back();
    topology_v0.reserve(edge_count_);
    topology_v1.reserve(edge_count_);
    topology_v0_global.reserve(edge_count_);
    topology_v1_global.reserve(edge_count_);
    topology_face0_local.reserve(edge_count_);
    topology_face1_local.reserve(edge_count_);
    topology_face0_global.reserve(edge_count_);
    topology_face1_global.reserve(edge_count_);
    topology_opposite0.reserve(edge_count_);
    topology_opposite1.reserve(edge_count_);
    topology_opposite0_global.reserve(edge_count_);
    topology_opposite1_global.reserve(edge_count_);

    std::array<std::vector<int>, 3> global_face_indices_cpu;
    for (auto &global_face_indices : global_face_indices_cpu) {
        global_face_indices.reserve(total_face_count);
    }
    std::vector<int> global_shape_ids_cpu;
    std::vector<int> global_local_prim_ids_cpu;
    std::vector<int> global_prim_ids_cpu;
    global_shape_ids_cpu.reserve(total_face_count);
    global_local_prim_ids_cpu.reserve(total_face_count);
    global_prim_ids_cpu.reserve(total_face_count);

    std::array<std::vector<int>, 3> triangle_edge_ids_cpu;
    for (auto &triangle_edge_ids : triangle_edge_ids_cpu) {
        triangle_edge_ids.assign(total_face_count, -1);
    }

    for (const SceneMeshRecord &record : mesh_records_) {
        const Mesh &mesh = *record.mesh;
        const auto &mesh_edge_indices = mesh.edge_indices();
        const int mesh_edge_count = mesh.edges_enabled() ? static_cast<int>(slices(mesh_edge_indices)) : 0;
        const Vector3i mesh_face_indices(detach<false>(mesh.face_indices()[0]),
                                                 detach<false>(mesh.face_indices()[1]),
                                                 detach<false>(mesh.face_indices()[2]));
        std::array<std::vector<int>, 3> mesh_face_cpu;
        copy_cuda_array(mesh_face_indices, mesh_face_cpu);
        for (int local_face_id = 0; local_face_id < mesh.face_count(); ++local_face_id) {
            global_face_indices_cpu[0].push_back(record.vertex_offset + mesh_face_cpu[0][local_face_id]);
            global_face_indices_cpu[1].push_back(record.vertex_offset + mesh_face_cpu[1][local_face_id]);
            global_face_indices_cpu[2].push_back(record.vertex_offset + mesh_face_cpu[2][local_face_id]);
            global_shape_ids_cpu.push_back(mesh.mesh_id());
            global_local_prim_ids_cpu.push_back(local_face_id);
            global_prim_ids_cpu.push_back(record.face_offset + local_face_id);
        }

        if (mesh_edge_count == 0) {
            continue;
        }

        std::array<std::vector<int>, 5> mesh_edge_cpu;
        copy_cuda_array(mesh_edge_indices, mesh_edge_cpu);

        for (int local_edge_id = 0; local_edge_id < mesh_edge_count; ++local_edge_id) {
            const int v0 = mesh_edge_cpu[0][local_edge_id];
            const int v1 = mesh_edge_cpu[1][local_edge_id];
            const int v0_global = record.vertex_offset + v0;
            const int v1_global = record.vertex_offset + v1;
            const int face0_local = mesh_edge_cpu[2][local_edge_id];
            const int face1_local = mesh_edge_cpu[3][local_edge_id];
            const int face0_global = record.face_offset + face0_local;
            const int face1_global = face1_local >= 0 ? record.face_offset + face1_local : -1;
            const int opposite0 = mesh_edge_cpu[4][local_edge_id];
            const int opposite0_global = opposite0 >= 0 ? record.vertex_offset + opposite0 : -1;
            const int global_edge_id = record.edge_offset + local_edge_id;

            const std::array<int, 3> face0_vertices {
                mesh_face_cpu[0][face0_local],
                mesh_face_cpu[1][face0_local],
                mesh_face_cpu[2][face0_local]
            };

            int opposite1 = -1;
            if (face1_local >= 0) {
                const std::array<int, 3> face1_vertices {
                    mesh_face_cpu[0][face1_local],
                    mesh_face_cpu[1][face1_local],
                    mesh_face_cpu[2][face1_local]
                };
                opposite1 = face_opposite_vertex(face1_vertices, v0, v1);
                const int opposite1_global = opposite1 >= 0 ? record.vertex_offset + opposite1 : -1;
                const int face1_slot = face_edge_slot(face1_vertices, v0, v1);
                if (face1_slot >= 0) {
                    triangle_edge_ids_cpu[face1_slot][face1_global] = global_edge_id;
                }
                topology_opposite1_global.push_back(opposite1_global);
            } else {
                topology_opposite1_global.push_back(-1);
            }

            const int face0_slot = face_edge_slot(face0_vertices, v0, v1);
            if (face0_slot >= 0) {
                triangle_edge_ids_cpu[face0_slot][face0_global] = global_edge_id;
            }

            topology_v0.push_back(v0);
            topology_v1.push_back(v1);
            topology_v0_global.push_back(v0_global);
            topology_v1_global.push_back(v1_global);
            topology_face0_local.push_back(face0_local);
            topology_face1_local.push_back(face1_local);
            topology_face0_global.push_back(face0_global);
            topology_face1_global.push_back(face1_global);
            topology_opposite0.push_back(opposite0);
            topology_opposite1.push_back(opposite1);
            topology_opposite0_global.push_back(opposite0_global);
        }
    }

    auto load_or_empty = [](const std::vector<int> &values) {
        return values.empty() ? Int() : load<Int>(values.data(), values.size());
    };

    face_offsets_ = load<Int>(face_offsets.data(), face_offsets.size());
    edge_offsets_ = load<Int>(edge_offsets.data(), edge_offsets.size());
    vertex_offsets_ = load<Int>(vertex_offsets.data(), vertex_offsets.size());
    triangle_info_ = empty<TriangleInfoAD>(total_face_count);
    triangle_info_detached_ = empty<TriangleInfo>(total_face_count);
    triangle_uv_ = zeros<TriangleUVAD>(total_face_count);
    triangle_uv_detached_ = zeros<TriangleUV>(total_face_count);
    triangle_face_normal_mask_ = empty<MaskAD>(total_face_count);
    triangle_face_normal_mask_detached_ = empty<Mask>(total_face_count);
    global_geometry_.vertices = total_vertex_count > 0 ? empty<Vector3fAD>(total_vertex_count) : Vector3fAD();
    global_geometry_.faces = Vector3i(
        load<Int>(global_face_indices_cpu[0].data(), total_face_count),
        load<Int>(global_face_indices_cpu[1].data(), total_face_count),
        load<Int>(global_face_indices_cpu[2].data(), total_face_count));
    global_geometry_.shape_id = load<Int>(global_shape_ids_cpu.data(), total_face_count);
    global_geometry_.local_prim_id =
        load<Int>(global_local_prim_ids_cpu.data(), total_face_count);
    global_geometry_.global_prim_id = load<Int>(global_prim_ids_cpu.data(), total_face_count);
    triangle_edge_ids_ = VectoriT<3, true>(load<Int>(triangle_edge_ids_cpu[0].data(), total_face_count),
                                           load<Int>(triangle_edge_ids_cpu[1].data(), total_face_count),
                                           load<Int>(triangle_edge_ids_cpu[2].data(), total_face_count));
    if (edge_count_ > 0) {
        edge_info_ = empty<SecondaryEdgeInfoAD>(edge_count_);
        edge_topology_ = SceneEdgeTopology {
            load_or_empty(topology_v0),
            load_or_empty(topology_v1),
            load_or_empty(topology_v0_global),
            load_or_empty(topology_v1_global),
            load_or_empty(topology_face0_local),
            load_or_empty(topology_face1_local),
            load_or_empty(topology_face0_global),
            load_or_empty(topology_face1_global),
            load_or_empty(topology_opposite0),
            load_or_empty(topology_opposite1),
            load_or_empty(topology_opposite0_global),
            load_or_empty(topology_opposite1_global)
        };
        edge_shape_ids_ = empty<Int>(edge_count_);
        edge_local_ids_ = empty<Int>(edge_count_);
        edge_mask_ = full<Mask>(true, edge_count_);
    } else {
        edge_info_ = SecondaryEdgeInfoAD();
        edge_topology_ = SceneEdgeTopology();
        edge_shape_ids_ = Int();
        edge_local_ids_ = Int();
        edge_mask_ = Mask();
    }

    for (const SceneMeshRecord &record : mesh_records_) {
        scatter_mesh_data(record, true);
        scatter_mesh_edge_data(record, true);
        const Mesh &mesh = *record.mesh;
        const int mesh_vertex_count = mesh.vertex_count();
        if (mesh_vertex_count > 0) {
            const IntAD vertex_scatter_indices = arange<IntAD>(mesh_vertex_count) + record.vertex_offset;
            scatter(global_geometry_.vertices, mesh.vertex_positions_world(), vertex_scatter_indices);
        }
    }
    global_geometry_.face_normal = triangle_info_.face_normal;

    int static_mesh_count = 0;
    int dynamic_mesh_count = 0;
    for (const SceneMeshRecord &record : mesh_records_) {
        if (record.dynamic) {
            ++dynamic_mesh_count;
        } else {
            ++static_mesh_count;
        }
    }

    optix_split_active_ =
        should_split_optix_scene(active_optix_split_mode(), static_mesh_count, dynamic_mesh_count);
    optix_static_mesh_indices_.clear();
    optix_dynamic_mesh_indices_.clear();
    optix_dynamic_mesh_local_index_.assign(mesh_records_.size(), -1);
    reset_multipath_pipelines();
    prewarm_path_multipath_pipelines(
        optix_split_active_ ? std::max(static_mesh_count, dynamic_mesh_count) : mesh_count_);

    if (optix_split_active_) {
        std::vector<OptixSceneMeshDesc> static_mesh_descs;
        std::vector<OptixSceneMeshDesc> dynamic_mesh_descs;
        static_mesh_descs.reserve(static_mesh_count);
        dynamic_mesh_descs.reserve(dynamic_mesh_count);

        for (size_t mesh_index = 0; mesh_index < mesh_records_.size(); ++mesh_index) {
            if (mesh_records_[mesh_index].dynamic) {
                optix_dynamic_mesh_local_index_[mesh_index] =
                    static_cast<int>(dynamic_mesh_descs.size());
                optix_dynamic_mesh_indices_.push_back(static_cast<int>(mesh_index));
                dynamic_mesh_descs.push_back(mesh_descs[mesh_index]);
            } else {
                optix_static_mesh_indices_.push_back(static_cast<int>(mesh_index));
                static_mesh_descs.push_back(mesh_descs[mesh_index]);
            }
        }

        optix_scene_ = std::make_unique<OptixScene>();
        optix_static_scene_ = std::make_unique<OptixScene>();
        optix_dynamic_scene_ = std::make_unique<OptixScene>();
        optix_scene_->build(mesh_descs);
        optix_static_scene_->build(static_mesh_descs, optix_scene_.get());
        optix_dynamic_scene_->build(dynamic_mesh_descs, optix_scene_.get());
    } else {
        optix_scene_ = std::make_unique<OptixScene>();
        optix_static_scene_ = std::make_unique<OptixScene>();
        optix_dynamic_scene_ = std::make_unique<OptixScene>();
        optix_scene_->build(mesh_descs);
    }
    mask_dirty_ = false;
    edge_bvh_ = std::make_unique<SceneEdge>();
    edge_optix_ = std::make_unique<SceneEdgeOptix>();
    if (edge_backend_builds_drjit(edge_bvh_backend_)) {
        edge_bvh_->build(edge_info_, edge_mask_);
        if (edge_backend_builds_optix(edge_bvh_backend_)) {
            edge_bvh_->materialize();
        }
    }
    if (edge_backend_builds_optix(edge_bvh_backend_)) {
        edge_optix_->build(edge_info_, edge_mask_);
        if (edge_backend_builds_drjit(edge_bvh_backend_)) {
            edge_bvh_->materialize();
        }
    }
    is_ready_ = true;
    pending_updates_ = false;
    ++scene_version_;
    ++edge_version_;
}

void Scene::update_mesh_vertices(int mesh_id, const Vector3fAD &positions) {
    require(is_ready(), "Scene::update_mesh_vertices(): scene is not built.");

    SceneMeshRecord &record = mesh_record(mesh_id);
    require(record.dynamic, "Scene::update_mesh_vertices(): target mesh is not dynamic.");
    require(static_cast<int>(slices(positions)) == record.mesh->vertex_count(),
            "Scene::update_mesh_vertices(): vertex count must remain unchanged.");

    record.mesh->set_vertex_positions(positions);
    record.vertices_dirty = true;
    pending_updates_ = true;
}

void Scene::set_mesh_transform(int mesh_id, const Matrix4fAD &matrix, bool set_left) {
    require(is_ready(), "Scene::set_mesh_transform(): scene is not built.");

    SceneMeshRecord &record = mesh_record(mesh_id);
    require(record.dynamic, "Scene::set_mesh_transform(): target mesh is not dynamic.");

    record.mesh->set_transform(matrix, set_left);
    record.transform_dirty = true;
    pending_updates_ = true;
}

void Scene::append_mesh_transform(int mesh_id, const Matrix4fAD &matrix, bool append_left) {
    require(is_ready(), "Scene::append_mesh_transform(): scene is not built.");

    SceneMeshRecord &record = mesh_record(mesh_id);
    require(record.dynamic, "Scene::append_mesh_transform(): target mesh is not dynamic.");

    record.mesh->append_transform(matrix, append_left);
    record.transform_dirty = true;
    pending_updates_ = true;
}

void Scene::set_edge_mask(const Mask &mask) {
    require(is_ready(), "Scene::set_edge_mask(): scene is not built.");
    require(static_cast<int>(mask.size()) == edge_count_,
            "Scene::set_edge_mask(): mask size must match the scene edge count.");

    if (mask.size() == edge_mask_.size() && drjit::all(mask == edge_mask_)) {
        return;
    }

    edge_mask_ = mask;
    mask_dirty_ = true;
    edge_bvh_dirty_ = true;
    pending_updates_ = true;
}

void Scene::sync() {
    ScopedNativeLaunchStage native_launch_stage(NativeLaunchStage::Sync);
    require(is_ready(), "Scene::sync(): scene is not built.");
    last_sync_profile_ = SceneSyncProfile();

    if (!pending_updates_) {
        return;
    }

    using Clock = std::chrono::steady_clock;
    const auto total_start = Clock::now();
    const bool mask_dirty_before = mask_dirty_;

    std::vector<OptixSceneMeshDesc> mesh_descs;
    mesh_descs.reserve(mesh_records_.size());

    std::vector<OptixSceneMeshUpdate> updates;
    updates.reserve(mesh_records_.size());

    for (size_t mesh_index = 0; mesh_index < mesh_records_.size(); ++mesh_index) {
        SceneMeshRecord &record = mesh_records_[mesh_index];
        mesh_descs.push_back({ record.mesh.get(), record.dynamic, record.face_offset, static_cast<int>(mesh_index) });

        if (!record.vertices_dirty && !record.transform_dirty) {
            continue;
        }

        const auto mesh_update_start = Clock::now();
        record.mesh->update_runtime_data(record.vertices_dirty, record.transform_dirty);
        last_sync_profile_.mesh_update_ms += std::chrono::duration<double, std::milli>(
            Clock::now() - mesh_update_start).count();

        const auto scatter_start = Clock::now();
        scatter_mesh_data(record, false);
        const int mesh_vertex_count = record.mesh->vertex_count();
        if (mesh_vertex_count > 0) {
            const IntAD vertex_scatter_indices = arange<IntAD>(mesh_vertex_count) + record.vertex_offset;
            scatter(global_geometry_.vertices,
                    record.mesh->vertex_positions_world(),
                    vertex_scatter_indices);
        }
        last_sync_profile_.triangle_scatter_ms += std::chrono::duration<double, std::milli>(
            Clock::now() - scatter_start).count();

        const int mesh_edge_count =
            record.mesh->edges_enabled() ? static_cast<int>(slices(record.mesh->edge_indices())) : 0;
        if (mesh_edge_count > 0 && !record.edge_dirty) {
            pending_edge_bvh_dirty_ranges_.push_back({ record.edge_offset, mesh_edge_count });
            record.edge_dirty = true;
            edge_bvh_dirty_ = true;
            ++last_sync_profile_.updated_edge_meshes;
            last_sync_profile_.updated_edges += mesh_edge_count;
        }

        updates.push_back({ static_cast<int>(mesh_index), record.vertices_dirty, record.transform_dirty });
        ++last_sync_profile_.updated_meshes;
        if (record.vertices_dirty) {
            ++last_sync_profile_.updated_vertex_meshes;
        }
        if (record.transform_dirty) {
            ++last_sync_profile_.updated_transform_meshes;
        }
        record.vertices_dirty = false;
        record.transform_dirty = false;
    }
    if (!updates.empty()) {
        global_geometry_.face_normal = triangle_info_.face_normal;
    }

    if (edge_bvh_dirty_) {
        const auto edge_scatter_start = Clock::now();
        for (SceneMeshRecord &record : mesh_records_) {
            if (!record.edge_dirty) {
                continue;
            }

            scatter_mesh_edge_data(record, false);
            record.edge_dirty = false;
        }
        last_sync_profile_.edge_scatter_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - edge_scatter_start).count();

        const auto edge_refit_start = Clock::now();
        ensure_edge_bvh_ready();
        last_sync_profile_.edge_refit_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - edge_refit_start).count();
    }

    const auto optix_start = Clock::now();
    if (optix_split_active_) {
        if (!updates.empty()) {
            optix_scene_->sync(mesh_descs, updates);
        }

        std::vector<OptixSceneMeshDesc> dynamic_mesh_descs;
        dynamic_mesh_descs.reserve(optix_dynamic_mesh_indices_.size());
        for (int mesh_index : optix_dynamic_mesh_indices_) {
            dynamic_mesh_descs.push_back(mesh_descs[static_cast<size_t>(mesh_index)]);
        }

        std::vector<OptixSceneMeshUpdate> dynamic_updates;
        dynamic_updates.reserve(updates.size());
        for (const OptixSceneMeshUpdate &update : updates) {
            const int dynamic_local_index =
                optix_dynamic_mesh_local_index_[static_cast<size_t>(update.mesh_id)];
            if (dynamic_local_index < 0) {
                continue;
            }
            dynamic_updates.push_back(
                { dynamic_local_index, update.vertices_dirty, update.transform_dirty });
        }

        if (!dynamic_updates.empty()) {
            optix_dynamic_scene_->sync(dynamic_mesh_descs, dynamic_updates);
        }
        last_sync_profile_.optix_sync_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - optix_start).count();
        if (!updates.empty()) {
            const OptixSyncProfile &optix_profile = optix_scene_->last_sync_profile();
            last_sync_profile_.optix_gas_update_ms += optix_profile.gas_update_ms;
            last_sync_profile_.optix_ias_update_ms += optix_profile.ias_update_ms;
        }
        if (!dynamic_updates.empty()) {
            const OptixSyncProfile &optix_profile = optix_dynamic_scene_->last_sync_profile();
            last_sync_profile_.optix_gas_update_ms += optix_profile.gas_update_ms;
            last_sync_profile_.optix_ias_update_ms += optix_profile.ias_update_ms;
        }
    } else {
        optix_scene_->sync(mesh_descs, updates);
        last_sync_profile_.optix_sync_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - optix_start).count();
        const OptixSyncProfile &optix_profile = optix_scene_->last_sync_profile();
        last_sync_profile_.optix_gas_update_ms = optix_profile.gas_update_ms;
        last_sync_profile_.optix_ias_update_ms = optix_profile.ias_update_ms;
    }
    pending_updates_ = false;
    if (!updates.empty()) {
        reflection_epc_geometry_ready_ = false;
    }
    if (!updates.empty()) {
        ++scene_version_;
    }
    if (mask_dirty_before || last_sync_profile_.updated_edge_meshes > 0) {
        ++edge_version_;
    }
    last_sync_profile_.total_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - total_start).count();
}

SceneEdgeInfo Scene::edge_info() const {
    require(is_ready(), "Scene::edge_info(): scene is not built.");
    require(!pending_updates_, "Scene::edge_info(): scene has pending updates. Call Scene::sync() first.");

    ensure_scene_edge_data_ready();

    SceneEdgeInfo info;
    info.start = edge_info_.start;
    info.edge = edge_info_.edge;
    info.end = edge_info_.start + edge_info_.edge;
    info.length = norm(edge_info_.edge);
    info.normal0 = edge_info_.normal0;
    info.normal1 = edge_info_.normal1;
    info.is_boundary = edge_info_.is_boundary;
    info.shape_id = edge_shape_ids_;
    info.local_edge_id = edge_local_ids_;
    info.global_edge_id = arange<Int>(edge_count_);
    return info;
}

std::string Scene::edge_bvh_backend() const {
    return edge_backend_name(edge_bvh_backend_);
}

SceneEdgeBVHStats Scene::edge_bvh_stats() const {
    require(is_ready(), "Scene::edge_bvh_stats(): scene is not built.");
    require(!pending_updates_,
            "Scene::edge_bvh_stats(): scene has pending updates. Call Scene::sync() first.");
    ensure_edge_bvh_ready();
    return edge_bvh_backend_ == EdgeBVHBackend::Optix ? edge_optix_->stats() : edge_bvh_->stats();
}

const SceneEdgeTopology &Scene::edge_topology() const {
    require(is_ready(), "Scene::edge_topology(): scene is not built.");
    return edge_topology_;
}

const Mask &Scene::edge_mask() const {
    require(is_ready(), "Scene::edge_mask(): scene is not built.");
    return edge_mask_;
}

const SceneGeometry &Scene::global_geometry() const {
    require(is_ready(), "Scene::global_geometry(): scene is not built.");
    require(!pending_updates_,
            "Scene::global_geometry(): scene has pending updates. Call Scene::sync() first.");
    return global_geometry_;
}

VectoriT<3, true> Scene::triangle_edge_indices(const Int &prim_id, bool global) const {
    require(is_ready(), "Scene::triangle_edge_indices(): scene is not built.");

    const int query_count = static_cast<int>(slices(prim_id));
    VectoriT<3, true> result(full<Int>(-1, query_count),
                             full<Int>(-1, query_count),
                             full<Int>(-1, query_count));
    if (query_count == 0) {
        return result;
    }

    const int face_count = static_cast<int>(slices(triangle_edge_ids_[0]));
    const Mask valid = prim_id >= 0 && prim_id < face_count;
    const Int edge0 = gather<Int>(triangle_edge_ids_[0], prim_id, valid);
    const Int edge1 = gather<Int>(triangle_edge_ids_[1], prim_id, valid);
    const Int edge2 = gather<Int>(triangle_edge_ids_[2], prim_id, valid);

    if (global) {
        result[0] = select(valid, edge0, result[0]);
        result[1] = select(valid, edge1, result[1]);
        result[2] = select(valid, edge2, result[2]);
        return result;
    }

    const Mask valid0 = valid && edge0 >= 0;
    const Mask valid1 = valid && edge1 >= 0;
    const Mask valid2 = valid && edge2 >= 0;
    result[0] = select(valid0, gather<Int>(edge_local_ids_, edge0, valid0), result[0]);
    result[1] = select(valid1, gather<Int>(edge_local_ids_, edge1, valid1), result[1]);
    result[2] = select(valid2, gather<Int>(edge_local_ids_, edge2, valid2), result[2]);
    return result;
}

VectoriT<2, true> Scene::edge_adjacent_faces(const Int &edge_id, bool global) const {
    require(is_ready(), "Scene::edge_adjacent_faces(): scene is not built.");

    const int query_count = static_cast<int>(slices(edge_id));
    VectoriT<2, true> result(full<Int>(-1, query_count),
                             full<Int>(-1, query_count));
    if (query_count == 0 || edge_count_ == 0) {
        return result;
    }

    const Mask valid = edge_id >= 0 && edge_id < edge_count_;
    const Int face0 = global
        ? gather<Int>(edge_topology_.face0_global, edge_id, valid)
        : gather<Int>(edge_topology_.face0_local, edge_id, valid);
    const Int face1 = global
        ? gather<Int>(edge_topology_.face1_global, edge_id, valid)
        : gather<Int>(edge_topology_.face1_local, edge_id, valid);
    result[0] = select(valid, face0, result[0]);
    result[1] = select(valid, face1, result[1]);
    return result;
}

bool Scene::is_ready() const {
    const bool optix_ready =
        optix_split_active_
            ? (optix_scene_ != nullptr && optix_static_scene_ != nullptr &&
               optix_dynamic_scene_ != nullptr && optix_scene_->is_ready() &&
               optix_static_scene_->is_ready() && optix_dynamic_scene_->is_ready())
            : (optix_scene_ != nullptr && optix_scene_->is_ready());
    bool edge_ready = true;
    if (edge_backend_builds_optix(edge_bvh_backend_)) {
        edge_ready &= edge_optix_ != nullptr && edge_optix_->is_ready();
    }
    if (edge_backend_builds_drjit(edge_bvh_backend_)) {
        edge_ready &= edge_bvh_ != nullptr && edge_bvh_->is_ready();
    }
    return is_ready_ && edge_ready && optix_ready;
}

template <bool Detached>
IntersectionT<Detached> Scene::intersect(const RayT<Detached> &ray, MaskT<Detached> active, RayFlags flags) const {
    require(is_ready(), "Scene::intersect(): scene is not built.");
    require(!pending_updates_, "Scene::intersect(): scene has pending updates. Call Scene::sync() first.");

    const int ray_count = static_cast<int>(slices(ray.o));
    const bool want_geo_n   = has_flag(flags, RayFlags::Geometric);
    const bool want_shading = has_flag(flags, RayFlags::ShadingN);
    const bool want_uv      = has_flag(flags, RayFlags::UV);
    const bool symbolic_optix_query = optix_split_active_ && uses_symbolic_optix_query_path();

    IntersectionT<Detached> intersection;
    intersection.t = full<FloatT<Detached>>(Infinity, ray_count);
    intersection.p = zeros<Vector3fT<Detached>>(ray_count);
    intersection.n = zeros<Vector3fT<Detached>>(ray_count);
    intersection.geo_n = zeros<Vector3fT<Detached>>(ray_count);
    intersection.uv = zeros<Vector2fT<Detached>>(ray_count);
    intersection.barycentric = zeros<Vector3fT<Detached>>(ray_count);
    intersection.shape_id = full<IntT<Detached>>(-1, ray_count);
    intersection.prim_id = full<IntT<Detached>>(-1, ray_count);

    MaskT<Detached> hit_mask = active;
    OptixIntersection optix_hit;
    if (optix_split_active_ && !symbolic_optix_query) {
        MaskT<Detached> static_hit_mask = active;
        MaskT<Detached> dynamic_hit_mask = active;
        const OptixIntersection static_hit =
            optix_static_scene_->template intersect<Detached>(ray, static_hit_mask);
        const OptixIntersection dynamic_hit =
            optix_dynamic_scene_->template intersect<Detached>(ray, dynamic_hit_mask);

        const Mask static_hit_mask_detached = detach<false>(static_hit_mask);
        const Mask dynamic_hit_mask_detached = detach<false>(dynamic_hit_mask);
        const Mask choose_dynamic =
            dynamic_hit_mask_detached &&
            (!static_hit_mask_detached || (dynamic_hit.t < static_hit.t));
        const Mask any_hit = static_hit_mask_detached || dynamic_hit_mask_detached;

        optix_hit.reserve(ray_count);
        optix_hit.t = select(choose_dynamic, dynamic_hit.t, static_hit.t);
        optix_hit.barycentric[0] =
            select(choose_dynamic, dynamic_hit.barycentric[0], static_hit.barycentric[0]);
        optix_hit.barycentric[1] =
            select(choose_dynamic, dynamic_hit.barycentric[1], static_hit.barycentric[1]);
        optix_hit.shape_id = select(choose_dynamic, dynamic_hit.shape_id, static_hit.shape_id);
        optix_hit.global_prim_id =
            select(choose_dynamic, dynamic_hit.global_prim_id, static_hit.global_prim_id);

        if constexpr (!Detached) {
            hit_mask = MaskAD(any_hit);
        } else {
            hit_mask = any_hit;
        }
    } else {
        optix_hit = optix_scene_->template intersect<Detached>(ray, hit_mask);
    }

    const Int shape_id = optix_hit.shape_id;
    const Int global_primitive_id = optix_hit.global_prim_id;
    const Mask hit_mask_detached = detach<false>(hit_mask);
    const Int mesh_face_offset = gather<Int>(face_offsets_, shape_id, hit_mask_detached);
    const Int local_primitive_id = global_primitive_id - mesh_face_offset;

    Vector2fT<Detached> triangle_uv_coords;
    FloatT<Detached> hit_distance;

    if constexpr (!Detached) {
        // AD path: re-gather vertex data and recompute intersection for gradients.
        const IntAD global_primitive_id_ad = IntAD(global_primitive_id);
        const Vector3fAD triangle_p0 = gather<Vector3fAD>(triangle_info_.p0, global_primitive_id_ad, hit_mask);
        const Vector3fAD triangle_e1 = gather<Vector3fAD>(triangle_info_.e1, global_primitive_id_ad, hit_mask);
        const Vector3fAD triangle_e2 = gather<Vector3fAD>(triangle_info_.e2, global_primitive_id_ad, hit_mask);
        std::tie(triangle_uv_coords, hit_distance) = ray_intersect_triangle<Detached>(triangle_p0, triangle_e1, triangle_e2, ray);

        if (want_geo_n || want_shading) {
            Vector3fT<Detached> geometric_normal = gather<Vector3fAD>(triangle_info_.face_normal, global_primitive_id_ad, hit_mask);

            if (want_shading) {
                Vector3fT<Detached> shading_n0 = gather<Vector3fAD>(triangle_info_.n0, global_primitive_id_ad, hit_mask);
                Vector3fT<Detached> shading_n1 = gather<Vector3fAD>(triangle_info_.n1, global_primitive_id_ad, hit_mask);
                Vector3fT<Detached> shading_n2 = gather<Vector3fAD>(triangle_info_.n2, global_primitive_id_ad, hit_mask);
                MaskT<Detached> use_face_normal_mask = gather<MaskAD>(triangle_face_normal_mask_, global_primitive_id_ad, hit_mask);
                const Vector2fT<Detached> safe_uv = select(hit_mask, triangle_uv_coords, zeros<Vector2fT<Detached>>(ray_count));
                Vector3fT<Detached> shading_normal =
                    normalize(bilinear<Detached>(shading_n0, shading_n1 - shading_n0, shading_n2 - shading_n0, safe_uv));
                shading_normal = select(use_face_normal_mask, geometric_normal, shading_normal);
                intersection.n = select(hit_mask, shading_normal, intersection.n);
            }
            if (want_geo_n) {
                intersection.geo_n = select(hit_mask, geometric_normal, intersection.geo_n);
            }
        }

        if (want_uv) {
            TriangleUVT<Detached> triangle_uv_data = gather<TriangleUVAD>(triangle_uv_, global_primitive_id_ad, hit_mask);
            const Vector2fT<Detached> safe_uv = select(hit_mask, triangle_uv_coords, zeros<Vector2fT<Detached>>(ray_count));
            const Vector2fT<Detached> uv =
                bilinear2<Detached>(triangle_uv_data[0], triangle_uv_data[1] - triangle_uv_data[0], triangle_uv_data[2] - triangle_uv_data[0], safe_uv);
            intersection.uv = select(hit_mask, uv, intersection.uv);
        }
    } else {
        // Detached path: use OptiX results directly, gather only what is needed.
        triangle_uv_coords = optix_hit.barycentric;
        hit_distance = optix_hit.t;

        if (want_geo_n || want_shading) {
            Vector3fT<Detached> geometric_normal = gather<Vector3f>(triangle_info_detached_.face_normal, global_primitive_id, hit_mask_detached);

            if (want_shading) {
                Vector3fT<Detached> shading_n0 = gather<Vector3f>(triangle_info_detached_.n0, global_primitive_id, hit_mask_detached);
                Vector3fT<Detached> shading_n1 = gather<Vector3f>(triangle_info_detached_.n1, global_primitive_id, hit_mask_detached);
                Vector3fT<Detached> shading_n2 = gather<Vector3f>(triangle_info_detached_.n2, global_primitive_id, hit_mask_detached);
                MaskT<Detached> use_face_normal_mask = gather<Mask>(triangle_face_normal_mask_detached_, global_primitive_id, hit_mask_detached);
                const Vector2fT<Detached> safe_uv = select(hit_mask_detached, triangle_uv_coords, zeros<Vector2fT<Detached>>(ray_count));
                Vector3fT<Detached> shading_normal =
                    normalize(bilinear<Detached>(shading_n0, shading_n1 - shading_n0, shading_n2 - shading_n0, safe_uv));
                shading_normal = select(use_face_normal_mask, geometric_normal, shading_normal);
                intersection.n = select(hit_mask_detached, shading_normal, intersection.n);
            }
            if (want_geo_n) {
                intersection.geo_n = select(hit_mask_detached, geometric_normal, intersection.geo_n);
            }
        }

        if (want_uv) {
            TriangleUVT<Detached> triangle_uv_data = gather<TriangleUV>(triangle_uv_detached_, global_primitive_id, hit_mask_detached);
            const Vector2fT<Detached> safe_uv = select(hit_mask_detached, triangle_uv_coords, zeros<Vector2fT<Detached>>(ray_count));
            const Vector2fT<Detached> uv =
                bilinear2<Detached>(triangle_uv_data[0], triangle_uv_data[1] - triangle_uv_data[0], triangle_uv_data[2] - triangle_uv_data[0], safe_uv);
            intersection.uv = select(hit_mask_detached, uv, intersection.uv);
        }
    }

    hit_mask &= drjit::isfinite(hit_distance) && (hit_distance < ray.tmax);

    const FloatT<Detached> safe_hit_distance = select(hit_mask, hit_distance, zeros<FloatT<Detached>>(ray_count));
    const Vector2fT<Detached> safe_triangle_uv = select(hit_mask, triangle_uv_coords, zeros<Vector2fT<Detached>>(ray_count));

    const Vector3fT<Detached> barycentric_coordinates(1.f - safe_triangle_uv.x() - safe_triangle_uv.y(),
                                                      safe_triangle_uv.x(),
                                                      safe_triangle_uv.y());
    const Vector3fT<Detached> hit_position = ray(safe_hit_distance);

    intersection.t = select(hit_mask, safe_hit_distance, intersection.t);
    intersection.p = select(hit_mask, hit_position, intersection.p);
    intersection.barycentric = select(hit_mask, barycentric_coordinates, intersection.barycentric);
    intersection.shape_id = select(hit_mask, IntT<Detached>(shape_id), intersection.shape_id);
    const IntT<Detached> local_primitive_id_t = IntT<Detached>(local_primitive_id);
    const IntT<Detached> global_primitive_id_t = IntT<Detached>(global_primitive_id);
    intersection.prim_id = select(hit_mask, local_primitive_id_t, intersection.prim_id);
    intersection.local_prim_id =
        select(hit_mask, local_primitive_id_t, intersection.local_prim_id);
    intersection.global_prim_id =
        select(hit_mask, global_primitive_id_t, intersection.global_prim_id);
    return intersection;
}

Scene::OptixSceneSelection Scene::select_optix_scenes() const {
    OptixSceneSelection selection;
    selection.hitgroup_record_count = mesh_count_;
    if (optix_split_active_) {
        selection.primary = optix_static_scene_.get();
        selection.secondary = optix_dynamic_scene_.get();
        selection.split_mode = 1;
        selection.hitgroup_record_count = static_cast<int>(
            std::max(optix_static_mesh_indices_.size(), optix_dynamic_mesh_indices_.size()));
    } else {
        selection.primary = optix_scene_.get();
    }
    return selection;
}

template <bool Detached>
ReflectionChainT<Detached> Scene::trace_reflections(const RayT<Detached> &ray,
                                                    int max_bounces,
                                                    MaskT<Detached> active) const {
    return this->template trace_reflections<Detached>(
        ray, max_bounces, ReflectionTraceOptions(), active);
}

template <bool Detached>
ReflectionTraceT<Detached> Scene::trace_bounces(
    const RayT<Detached> &ray,
    int max_bounces,
    MaskT<Detached> active) const {
    return this->template trace_bounces<Detached>(
        ray, max_bounces, ReflectionTraceOptions(), active);
}

template <bool Detached>
ReflectionTraceT<Detached> Scene::trace_bounces(
    const RayT<Detached> &ray,
    int max_bounces,
    const ReflectionTraceOptions &options,
    MaskT<Detached> active) const {
    ScopedNativeLaunchStage native_launch_stage(NativeLaunchStage::TraceReflections);
    require(is_ready(), "Scene::trace_reflections(): scene is not built.");
    require(!pending_updates_,
            "Scene::trace_reflections(): scene has pending updates. Call Scene::sync() first.");
    require(max_bounces > 0,
            "Scene::trace_reflections(): max_bounces must be positive.");
    return trace_bounces_impl<Detached>(
        *this, ray, max_bounces, options, active);
}

template <bool Detached>
ReflectionChainT<Detached> Scene::trace_reflections(const RayT<Detached> &ray,
                                                    int max_bounces,
                                                    const ReflectionTraceOptions &options,
                                                    MaskT<Detached> active) const {
    ScopedNativeLaunchStage native_launch_stage(NativeLaunchStage::TraceReflections);
    require(is_ready(), "Scene::trace_reflections(): scene is not built.");
    require(!pending_updates_,
            "Scene::trace_reflections(): scene has pending updates. Call Scene::sync() first.");
    require(max_bounces > 0, "Scene::trace_reflections(): max_bounces must be positive.");

    const int ray_count = static_cast<int>(slices(ray.o));
    ReflectionChainT<Detached> result =
        initialize_reflection_chain_result<Detached>(ray_count, max_bounces);
    if (ray_count == 0) {
        return result;
    }

    const bool symbolic_reflection_trace = recording_reflections();
    require(!symbolic_reflection_trace || !options.deduplicate,
            "Scene::trace_reflections(): symbolic recording does not support deduplicate=true yet.");
    if (symbolic_reflection_trace) {
        require(max_bounces == 1,
                "Scene::trace_reflections(): symbolic recording currently supports max_bounces=1 only.");
        const ReflectionTraceT<Detached> trace =
            this->template trace_bounces<Detached>(ray, 1, options, active);
        result.bounce_count = trace.bounce_count;
        result.discovery_count = trace.discovery_count;
        result.representative_ray_index = trace.representative_ray_index;
        if (!trace.bounces.empty()) {
            const ReflectionBounceT<Detached> &bounce = trace.bounces.front();
            result.t = bounce.t;
            result.hit_points = bounce.hit_points;
            result.geo_normals = bounce.geo_normals;
            result.image_sources = bounce.image_sources;
            result.plane_points = bounce.plane_points;
            result.plane_normals = bounce.plane_normals;
            result.shape_ids = bounce.shape_ids;
            result.prim_ids = bounce.prim_ids;
            result.local_prim_ids = bounce.local_prim_ids;
            result.global_prim_ids = bounce.global_prim_ids;

            const MaskT<Detached> trailing_active = trace.bounce_count > 0;
            const Vector3fT<Detached> reflected_direction =
                ray.d - 2.f * dot(ray.d, bounce.geo_normals) * bounce.geo_normals;
            const Vector3fT<Detached> trailing_origin =
                bounce.hit_points + Epsilon * reflected_direction;
            RayT<Detached> trailing_ray(
                trailing_origin,
                reflected_direction,
                full<FloatT<Detached>>(Infinity, ray_count));
            const IntersectionT<Detached> trailing =
                this->template intersect<Detached>(
                    trailing_ray, trailing_active, RayFlags::Geometric);
            const MaskT<Detached> trailing_hit =
                trailing_active && trailing.is_valid();
            result.trailing_t =
                select(trailing_hit,
                       trailing.t,
                       full<FloatT<Detached>>(Infinity, ray_count));
            result.trailing_prim =
                select(trailing_hit,
                       trailing.global_prim_id,
                       full<IntT<Detached>>(-1, ray_count));
            result.trailing_dir =
                select(trailing_active,
                       reflected_direction,
                       zeros<Vector3fT<Detached>>(ray_count));
            result.trailing_origin =
                select(trailing_active,
                       trailing_origin,
                       zeros<Vector3fT<Detached>>(ray_count));
        }
        return result;
    }

    const Mask active_detached = sanitize_reflection_active<Detached>(ray, active);
    if (drjit::none(active_detached)) {
        return result;
    }
    if constexpr (!Detached) {
        drjit::eval(triangle_info_.p0,
                    triangle_info_.e1,
                    triangle_info_.e2,
                    triangle_info_.face_normal);
    }

    const OptixSceneSelection scenes = select_optix_scenes();
    const OptixScene *primary_scene = scenes.primary;
    const OptixScene *secondary_scene = scenes.secondary;
    int split_mode = scenes.split_mode;
    int hitgroup_record_count = scenes.hitgroup_record_count;

    require(primary_scene != nullptr && primary_scene->is_ready(),
            "Scene::trace_reflections(): OptiX scene is not ready.");
    require(hitgroup_record_count > 0,
            "Scene::trace_reflections(): invalid hitgroup record count.");

    ensure_pipeline(reflection_pipeline_, primary_scene->context(),
                    hitgroup_record_count, reflection_trace_pipeline_config());

    Ray broadphase_ray;
    if constexpr (!Detached) {
        broadphase_ray = Ray(detach<false>(ray.o),
                                     detach<false>(ray.d),
                                     detach<false>(ray.tmax));
    } else {
        broadphase_ray = ray;
    }

    drjit::eval(broadphase_ray.o,
                broadphase_ray.d,
                broadphase_ray.tmax,
                active_detached,
                triangle_info_detached_.p0,
                triangle_info_detached_.e1,
                triangle_info_detached_.e2,
                triangle_info_detached_.face_normal,
                face_offsets_);
    if (options.deduplicate && slices(options.canonical_prim_table) > 0) {
        drjit::eval(options.canonical_prim_table);
    }

    ReflectionTraceRaw raw = allocate_reflection_trace_raw(ray_count, max_bounces);
    initialize_reflection_trace_raw(raw);

    ReflectionTraceParams params = {};
    params.primary_handle = primary_scene->ias_handle();
    params.secondary_handle =
        secondary_scene != nullptr && secondary_scene->is_ready() ? secondary_scene->ias_handle() : 0ull;
    params.split_mode = split_mode;
    params.tri_p0_x = triangle_info_detached_.p0.x().data();
    params.tri_p0_y = triangle_info_detached_.p0.y().data();
    params.tri_p0_z = triangle_info_detached_.p0.z().data();
    params.tri_e1_x = triangle_info_detached_.e1.x().data();
    params.tri_e1_y = triangle_info_detached_.e1.y().data();
    params.tri_e1_z = triangle_info_detached_.e1.z().data();
    params.tri_e2_x = triangle_info_detached_.e2.x().data();
    params.tri_e2_y = triangle_info_detached_.e2.y().data();
    params.tri_e2_z = triangle_info_detached_.e2.z().data();
    params.tri_fn_x = triangle_info_detached_.face_normal.x().data();
    params.tri_fn_y = triangle_info_detached_.face_normal.y().data();
    params.tri_fn_z = triangle_info_detached_.face_normal.z().data();
    params.face_offsets = face_offsets_.data();
    params.n_meshes = mesh_count_;
    params.n_triangles = static_cast<int>(slices(triangle_info_detached_.p0));
    params.ray_ox = broadphase_ray.o.x().data();
    params.ray_oy = broadphase_ray.o.y().data();
    params.ray_oz = broadphase_ray.o.z().data();
    params.ray_dx = broadphase_ray.d.x().data();
    params.ray_dy = broadphase_ray.d.y().data();
    params.ray_dz = broadphase_ray.d.z().data();
    params.ray_tmax = broadphase_ray.tmax.data();
    params.active_mask = reinterpret_cast<const uint8_t *>(active_detached.data());
    params.n_rays = ray_count;
    params.max_bounces = max_bounces;
    params.out_bounce_count = raw.bounce_count.data();
    params.out_shape_ids = raw.shape_ids.data();
    params.out_prim_ids = raw.prim_ids.data();
    params.out_t = raw.t.data();
    params.out_bary_u = raw.bary_u.data();
    params.out_bary_v = raw.bary_v.data();
    params.out_hit_x = raw.hit_x.data();
    params.out_hit_y = raw.hit_y.data();
    params.out_hit_z = raw.hit_z.data();
    params.out_norm_x = raw.norm_x.data();
    params.out_norm_y = raw.norm_y.data();
    params.out_norm_z = raw.norm_z.data();
    params.out_img_x = raw.img_x.data();
    params.out_img_y = raw.img_y.data();
    params.out_img_z = raw.img_z.data();
    params.out_trailing_t = raw.trailing_t.data();
    params.out_trailing_prim = raw.trailing_prim.data();
    params.out_trailing_dir_x = raw.trailing_dir_x.data();
    params.out_trailing_dir_y = raw.trailing_dir_y.data();
    params.out_trailing_dir_z = raw.trailing_dir_z.data();
    params.out_trailing_origin_x = raw.trailing_origin_x.data();
    params.out_trailing_origin_y = raw.trailing_origin_y.data();
    params.out_trailing_origin_z = raw.trailing_origin_z.data();

    reflection_pipeline_->launch(0, params);

    int trace_ray_count = ray_count;
    Int trace_bounce_count = raw.bounce_count;
    Int trace_discovery_count =
        select(raw.bounce_count > 0,
               full<Int>(1, ray_count),
               full<Int>(0, ray_count));
    Int trace_representative_ray_index = arange<Int>(ray_count);
    Int trace_shape_ids = raw.shape_ids;
    Int trace_prim_ids = raw.prim_ids;
    Float trace_t = raw.t;
    Float trace_hit_x = raw.hit_x;
    Float trace_hit_y = raw.hit_y;
    Float trace_hit_z = raw.hit_z;
    Float trace_norm_x = raw.norm_x;
    Float trace_norm_y = raw.norm_y;
    Float trace_norm_z = raw.norm_z;
    Float trace_img_x = raw.img_x;
    Float trace_img_y = raw.img_y;
    Float trace_img_z = raw.img_z;
    Float trace_trailing_t = raw.trailing_t;
    Int trace_trailing_prim = raw.trailing_prim;
    Float trace_trailing_dir_x = raw.trailing_dir_x;
    Float trace_trailing_dir_y = raw.trailing_dir_y;
    Float trace_trailing_dir_z = raw.trailing_dir_z;
    Float trace_trailing_origin_x = raw.trailing_origin_x;
    Float trace_trailing_origin_y = raw.trailing_origin_y;
    Float trace_trailing_origin_z = raw.trailing_origin_z;

    if (options.deduplicate) {
        ReflectionTraceRaw compacted = allocate_reflection_trace_raw(ray_count, max_bounces);
        initialize_reflection_trace_raw(compacted);

        const Int canonical_table = options.canonical_prim_table;
        const int canonical_table_size = static_cast<int>(slices(canonical_table));
        const int n_unique = reflection_dedup_gpu(
            ray_count,
            max_bounces,
            raw.bounce_count.data(),
            raw.shape_ids.data(),
            raw.prim_ids.data(),
            raw.t.data(),
            raw.bary_u.data(),
            raw.bary_v.data(),
            raw.hit_x.data(),
            raw.hit_y.data(),
            raw.hit_z.data(),
            raw.norm_x.data(),
            raw.norm_y.data(),
            raw.norm_z.data(),
            raw.img_x.data(),
            raw.img_y.data(),
            raw.img_z.data(),
            face_offsets_.data(),
            mesh_count_,
            canonical_table_size > 0 ? canonical_table.data() : nullptr,
            canonical_table_size,
            options.image_source_tolerance,
            compacted.bounce_count.data(),
            compacted.shape_ids.data(),
            compacted.prim_ids.data(),
            compacted.t.data(),
            compacted.bary_u.data(),
            compacted.bary_v.data(),
            compacted.hit_x.data(),
            compacted.hit_y.data(),
            compacted.hit_z.data(),
            compacted.norm_x.data(),
            compacted.norm_y.data(),
            compacted.norm_z.data(),
            compacted.img_x.data(),
            compacted.img_y.data(),
            compacted.img_z.data(),
            compacted.discovery_count.data(),
            compacted.representative_ray_index.data());

        trace_ray_count = n_unique;
        const int unique_slot_count = trace_ray_count * max_bounces;
        trace_bounce_count = prefix_array(compacted.bounce_count, trace_ray_count);
        trace_discovery_count = prefix_array(compacted.discovery_count, trace_ray_count);
        trace_representative_ray_index =
            prefix_array(compacted.representative_ray_index, trace_ray_count);
        trace_shape_ids = prefix_array(compacted.shape_ids, unique_slot_count);
        trace_prim_ids = prefix_array(compacted.prim_ids, unique_slot_count);
        trace_t = prefix_array(compacted.t, unique_slot_count);
        trace_hit_x = prefix_array(compacted.hit_x, unique_slot_count);
        trace_hit_y = prefix_array(compacted.hit_y, unique_slot_count);
        trace_hit_z = prefix_array(compacted.hit_z, unique_slot_count);
        trace_norm_x = prefix_array(compacted.norm_x, unique_slot_count);
        trace_norm_y = prefix_array(compacted.norm_y, unique_slot_count);
        trace_norm_z = prefix_array(compacted.norm_z, unique_slot_count);
        trace_img_x = prefix_array(compacted.img_x, unique_slot_count);
        trace_img_y = prefix_array(compacted.img_y, unique_slot_count);
        trace_img_z = prefix_array(compacted.img_z, unique_slot_count);
        const Mask unique_mask = full<Mask>(true, trace_ray_count);
        trace_trailing_t =
            gather<Float>(raw.trailing_t, trace_representative_ray_index, unique_mask);
        trace_trailing_prim =
            gather<Int>(raw.trailing_prim, trace_representative_ray_index, unique_mask);
        trace_trailing_dir_x =
            gather<Float>(raw.trailing_dir_x, trace_representative_ray_index, unique_mask);
        trace_trailing_dir_y =
            gather<Float>(raw.trailing_dir_y, trace_representative_ray_index, unique_mask);
        trace_trailing_dir_z =
            gather<Float>(raw.trailing_dir_z, trace_representative_ray_index, unique_mask);
        trace_trailing_origin_x =
            gather<Float>(raw.trailing_origin_x, trace_representative_ray_index, unique_mask);
        trace_trailing_origin_y =
            gather<Float>(raw.trailing_origin_y, trace_representative_ray_index, unique_mask);
        trace_trailing_origin_z =
            gather<Float>(raw.trailing_origin_z, trace_representative_ray_index, unique_mask);
        result.ray_count = trace_ray_count;
    }

    const Int trace_global_prim_ids =
        globalize_primitive_ids(trace_prim_ids, trace_shape_ids, face_offsets_);

    if constexpr (Detached) {
        const Vector3f hit_points(trace_hit_x, trace_hit_y, trace_hit_z);
        const Vector3f plane_normals(trace_norm_x, trace_norm_y, trace_norm_z);
        result.bounce_count = trace_bounce_count;
        result.discovery_count = trace_discovery_count;
        result.representative_ray_index = trace_representative_ray_index;
        result.t = trace_t;
        result.hit_points = hit_points;
        result.geo_normals = plane_normals;
        result.image_sources = Vector3f(trace_img_x, trace_img_y, trace_img_z);
        result.plane_points = hit_points;
        result.plane_normals = plane_normals;
        result.shape_ids = trace_shape_ids;
        result.prim_ids = trace_prim_ids;
        result.local_prim_ids = trace_prim_ids;
        result.global_prim_ids = trace_global_prim_ids;
        result.trailing_t = trace_trailing_t;
        result.trailing_prim = trace_trailing_prim;
        result.trailing_dir = Vector3f(trace_trailing_dir_x,
                                               trace_trailing_dir_y,
                                               trace_trailing_dir_z);
        result.trailing_origin = Vector3f(trace_trailing_origin_x,
                                                  trace_trailing_origin_y,
                                                  trace_trailing_origin_z);
        return result;
    } else {
        result = initialize_reflection_chain_result<false>(trace_ray_count, max_bounces);
        result.bounce_count = IntAD(trace_bounce_count);
        result.discovery_count = IntAD(trace_discovery_count);
        result.representative_ray_index = IntAD(trace_representative_ray_index);
        result.shape_ids = IntAD(trace_shape_ids);
        result.prim_ids = IntAD(trace_prim_ids);
        result.local_prim_ids = IntAD(trace_prim_ids);
        result.global_prim_ids = IntAD(trace_global_prim_ids);
        result.trailing_t = FloatAD(trace_trailing_t);
        result.trailing_prim = IntAD(trace_trailing_prim);
        result.trailing_dir = Vector3fAD(FloatAD(trace_trailing_dir_x),
                                       FloatAD(trace_trailing_dir_y),
                                       FloatAD(trace_trailing_dir_z));
        result.trailing_origin = Vector3fAD(FloatAD(trace_trailing_origin_x),
                                          FloatAD(trace_trailing_origin_y),
                                          FloatAD(trace_trailing_origin_z));

        if (trace_ray_count == 0) {
            return result;
        }

        const MaskAD representative_mask = full<MaskAD>(true, trace_ray_count);
        const Mask representative_mask_detached =
            full<Mask>(true, trace_ray_count);
        const IntAD representative_ray_index = IntAD(trace_representative_ray_index);
        RayAD current_ray(
            gather<Vector3fAD>(ray.o, representative_ray_index, representative_mask),
            gather<Vector3fAD>(ray.d, representative_ray_index, representative_mask),
            gather<FloatAD>(ray.tmax, representative_ray_index, representative_mask));
        Mask current_active_detached =
            gather<Mask>(active_detached,
                                 trace_representative_ray_index,
                                 representative_mask_detached);
        Vector3fAD current_image_source = current_ray.o;
        const Int bounce_slots =
            arange<Int>(trace_ray_count) * Int(max_bounces);

        for (int bounce = 0; bounce < max_bounces; ++bounce) {
            const Int slot_detached = bounce_slots + bounce;
            const IntAD slot = IntAD(slot_detached);
            const Int shape_id_detached =
                gather<Int>(trace_shape_ids, slot_detached, current_active_detached);
            const Int prim_id_detached =
                gather<Int>(trace_prim_ids, slot_detached, current_active_detached);
            const Mask broadphase_hit =
                current_active_detached && (shape_id_detached >= 0) && (prim_id_detached >= 0);
            if (drjit::none(broadphase_hit)) {
                break;
            }

            const Int mesh_face_offset =
                gather<Int>(face_offsets_, shape_id_detached, broadphase_hit);
            const Int global_prim_detached = mesh_face_offset + prim_id_detached;
            const IntAD global_prim = IntAD(global_prim_detached);
            const MaskAD hit_mask = MaskAD(broadphase_hit);

            const Vector3fAD triangle_p0 = gather<Vector3fAD>(triangle_info_.p0, global_prim, hit_mask);
            const Vector3fAD triangle_e1 = gather<Vector3fAD>(triangle_info_.e1, global_prim, hit_mask);
            const Vector3fAD triangle_e2 = gather<Vector3fAD>(triangle_info_.e2, global_prim, hit_mask);

            Vector2fAD triangle_barycentric;
            FloatAD hit_distance;
            std::tie(triangle_barycentric, hit_distance) =
                ray_intersect_triangle<false>(triangle_p0, triangle_e1, triangle_e2, current_ray);

            MaskAD bounce_hit =
                hit_mask && drjit::isfinite(hit_distance) && (hit_distance < current_ray.tmax);
            const FloatAD safe_t =
                select(bounce_hit, hit_distance, full<FloatAD>(Infinity, trace_ray_count));
            Vector3fAD geo_normal = gather<Vector3fAD>(triangle_info_.face_normal, global_prim, hit_mask);
            geo_normal = normalize(select(hit_mask, geo_normal, Vector3fAD(0.f, 0.f, 1.f)));
            geo_normal = select(dot(current_ray.d, geo_normal) > 0.f, -geo_normal, geo_normal);
            const Vector3fAD hit_point =
                current_ray(select(bounce_hit, safe_t, zeros<FloatAD>(trace_ray_count)));
            const FloatAD plane_distance = dot(current_image_source - hit_point, geo_normal);
            const Vector3fAD reflected_image_source =
                current_image_source - 2.f * plane_distance * geo_normal;

            scatter(result.t, safe_t, slot, bounce_hit);
            scatter(result.hit_points, hit_point, slot, bounce_hit);
            scatter(result.geo_normals, geo_normal, slot, bounce_hit);
            scatter(result.image_sources, reflected_image_source, slot, bounce_hit);
            scatter(result.plane_points, hit_point, slot, bounce_hit);
            scatter(result.plane_normals, geo_normal, slot, bounce_hit);

            const FloatAD ray_dot_normal = dot(current_ray.d, geo_normal);
            const Vector3fAD reflected_direction =
                current_ray.d - 2.f * ray_dot_normal * geo_normal;
            current_ray.o = select(bounce_hit,
                                   hit_point + Epsilon * reflected_direction,
                                   current_ray.o);
            current_ray.d = select(bounce_hit, reflected_direction, current_ray.d);
            current_ray.tmax = select(bounce_hit,
                                      full<FloatAD>(Infinity, trace_ray_count),
                                      current_ray.tmax);
            current_image_source =
                select(bounce_hit, reflected_image_source, current_image_source);
            current_active_detached = detach<false>(bounce_hit);
        }

        const MaskAD trailing_active = result.bounce_count > 0;
        const IntersectionAD trailing =
            this->template intersect<false>(
                current_ray, trailing_active, RayFlags::Geometric);
        const MaskAD trailing_hit = trailing_active && trailing.is_valid();
        result.trailing_t =
            select(trailing_hit,
                   trailing.t,
                   full<FloatAD>(Infinity, trace_ray_count));
        result.trailing_prim =
            select(trailing_hit,
                   trailing.global_prim_id,
                   full<IntAD>(-1, trace_ray_count));
        result.trailing_dir =
            select(trailing_active,
                   current_ray.d,
                   zeros<Vector3fAD>(trace_ray_count));
        result.trailing_origin =
            select(trailing_active,
                   current_ray.o,
                   zeros<Vector3fAD>(trace_ray_count));

        return result;
    }
}

template <bool Detached>
DfrPathsT<Detached> Scene::trace_dfr_paths(
    const Vector3fT<Detached> &tx_positions,
    const Vector3fT<Detached> &rx_positions,
    const DfrStatesT<Detached> &states,
    const DfrMaterialT<Detached> &material,
    const DfrPathOptions &options,
    MaskT<Detached> active) const {
    ScopedNativeLaunchStage native_launch_stage(
        NativeLaunchStage::AccumDfr);
    require(is_ready(), "Scene::trace_dfr_paths(): scene is not built.");
    require(!pending_updates_,
            "Scene::trace_dfr_paths(): scene has pending updates. Call Scene::sync() first.");
    require(options.wavelength > 0.f,
            "Scene::trace_dfr_paths(): wavelength must be positive.");
    require(options.max_order == 1,
            "Scene::trace_dfr_paths(): only max_order == 1 is supported.");
    require(options.max_paths > 0,
            "Scene::trace_dfr_paths(): max_paths must be positive.");
    require((options.strategy_mask & RAYD_DFR_DIRECT) != 0,
            "Scene::trace_dfr_paths(): first-order path export requires direct diffraction.");

    DfrPathsT<Detached> result;
    if constexpr (!Detached) {
        const int tx_count = static_cast<int>(slices(tx_positions));
        const int rx_width = static_cast<int>(slices(rx_positions));
        const int rx_count = options.max_rx > 0
                                 ? std::min(rx_width, options.max_rx)
                                 : rx_width;
        const int state_width = static_cast<int>(slices(states.edge_index));
        const int state_count = states.count > 0 ? states.count : state_width;
        if (tx_count == 0 || rx_count == 0 || state_count == 0) {
            result.capacity = 0;
            result.count = full<IntAD>(0, 1);
            result.valid = full<MaskAD>(false, 0);
            result.tx_id = full<IntAD>(-1, 0);
            result.rx_id = full<IntAD>(-1, 0);
            result.order = full<IntAD>(0, 0);
            result.edge0 = full<IntAD>(-1, 0);
            result.edge1 = full<IntAD>(-1, 0);
            result.edge2 = full<IntAD>(-1, 0);
            result.delay = zeros<FloatAD>(0);
            result.field_x = drjit::Complex<FloatAD>(zeros<FloatAD>(0), zeros<FloatAD>(0));
            result.field_y = drjit::Complex<FloatAD>(zeros<FloatAD>(0), zeros<FloatAD>(0));
            result.field_z = drjit::Complex<FloatAD>(zeros<FloatAD>(0), zeros<FloatAD>(0));
            result.p0 = zeros<Vector3fAD>(0);
            result.p1 = zeros<Vector3fAD>(0);
            result.p2 = zeros<Vector3fAD>(0);
            return result;
        }
        require(state_count > 0 && state_count <= state_width,
                "Scene::trace_dfr_paths(): invalid state count.");
        require(static_cast<int>(slices(states.edge_pos)) >= state_count &&
                    static_cast<int>(slices(states.edge_dir)) >= state_count &&
                    static_cast<int>(slices(states.edge_t_min)) >= state_count &&
                    static_cast<int>(slices(states.edge_t_max)) >= state_count &&
                    static_cast<int>(slices(states.n0)) >= state_count &&
                    static_cast<int>(slices(states.n1)) >= state_count &&
                    static_cast<int>(slices(states.prim0)) >= state_count &&
                    static_cast<int>(slices(states.prim1)) >= state_count &&
                    static_cast<int>(slices(states.exterior_angle)) >= state_count &&
                    static_cast<int>(slices(states.src)) >= state_count &&
                    static_cast<int>(slices(states.src_power)) >= state_count,
                "Scene::trace_dfr_paths(): state fields must cover state count.");
        require(rx_count <= rx_width,
                "Scene::trace_dfr_paths(): invalid receiver count.");

        const int material_count = static_cast<int>(slices(material.eta_r));
        require(material_count > 0,
                "Scene::trace_dfr_paths(): material payload must not be empty.");
        require(static_cast<int>(slices(material.sigma)) == material_count &&
                    static_cast<int>(slices(material.mu_r)) == material_count &&
                    static_cast<int>(slices(material.gain)) == material_count &&
                    static_cast<int>(slices(material.valid)) == material_count,
                "Scene::trace_dfr_paths(): material payload fields must have matching widths.");

        MaskAD active_ad = active;
        int active_width = static_cast<int>(slices(active_ad));
        if (active_width == 1 && state_count > 1) {
            active_ad = gather<MaskAD>(active_ad, zeros<IntAD>(state_count));
            active_width = state_count;
        } else {
            require(active_width == state_count,
                    "Scene::trace_dfr_paths(): active width must be 1 or match state count.");
        }

        const int state_limit = std::min(state_count, options.max_paths);
        const int64_t capacity64 =
            static_cast<int64_t>(tx_count) *
            static_cast<int64_t>(rx_count) *
            static_cast<int64_t>(state_limit);
        require(capacity64 <= static_cast<int64_t>(std::numeric_limits<int>::max()),
                "Scene::trace_dfr_paths(): requested path capacity exceeds int range.");
        const int capacity = static_cast<int>(capacity64);
        result.capacity = capacity;
        result.count = full<IntAD>(0, 1);
        result.valid = full<MaskAD>(false, capacity);
        result.tx_id = full<IntAD>(-1, capacity);
        result.rx_id = full<IntAD>(-1, capacity);
        result.order = full<IntAD>(0, capacity);
        result.edge0 = full<IntAD>(-1, capacity);
        result.edge1 = full<IntAD>(-1, capacity);
        result.edge2 = full<IntAD>(-1, capacity);
        result.delay = zeros<FloatAD>(capacity);
        result.field_x = drjit::Complex<FloatAD>(zeros<FloatAD>(capacity), zeros<FloatAD>(capacity));
        result.field_y = drjit::Complex<FloatAD>(zeros<FloatAD>(capacity), zeros<FloatAD>(capacity));
        result.field_z = drjit::Complex<FloatAD>(zeros<FloatAD>(capacity), zeros<FloatAD>(capacity));
        result.p0 = zeros<Vector3fAD>(capacity);
        result.p1 = zeros<Vector3fAD>(capacity);
        result.p2 = zeros<Vector3fAD>(capacity);
        if (capacity == 0) {
            return result;
        }

        auto material_gain_for_faces = [&](const IntAD &prim0,
                                           const IntAD &prim1,
                                           const MaskAD &path_active) -> FloatAD {
            const MaskAD prim0_in_range =
                path_active && (prim0 >= IntAD(0)) && (prim0 < IntAD(material_count));
            const IntAD safe0 = select(prim0_in_range, prim0, IntAD(0));
            const MaskAD prim0_valid =
                prim0_in_range && gather<MaskAD>(material.valid, safe0, prim0_in_range);
            const MaskAD prim1_in_range =
                path_active && (prim1 >= IntAD(0)) && (prim1 < IntAD(material_count));
            const IntAD safe1 = select(prim1_in_range, prim1, IntAD(0));
            const MaskAD prim1_valid =
                prim1_in_range && gather<MaskAD>(material.valid, safe1, prim1_in_range);
            const IntAD chosen = select(prim0_valid, safe0, safe1);
            const MaskAD chosen_valid = prim0_valid || prim1_valid;
            const FloatAD gain =
                gather<FloatAD>(material.gain, chosen, chosen_valid);
            return select(chosen_valid, maximum(gain, FloatAD(0.f)), FloatAD(1.f));
        };

        const UIntAD lane_u = arange<UIntAD>(capacity);
        const IntAD lane = IntAD(lane_u);
        const IntAD state_idx = IntAD(lane_u % UIntAD(state_limit));
        const UIntAD pair_idx = lane_u / UIntAD(state_limit);
        const IntAD rx_idx = IntAD(pair_idx % UIntAD(rx_count));
        const IntAD tx_idx = IntAD(pair_idx / UIntAD(rx_count));
        const MaskAD lane_active = full<MaskAD>(true, capacity);
        const MaskAD state_active =
            gather<MaskAD>(active_ad, state_idx, lane_active);

        const Vector3fAD edge_pos =
            gather<Vector3fAD>(states.edge_pos, state_idx, state_active);
        const Vector3fAD edge_dir =
            normalize(gather<Vector3fAD>(states.edge_dir, state_idx, state_active));
        const FloatAD edge_t_min =
            gather<FloatAD>(states.edge_t_min, state_idx, state_active);
        const FloatAD edge_t_max =
            gather<FloatAD>(states.edge_t_max, state_idx, state_active);
        const FloatAD edge_t = FloatAD(0.5f) * (edge_t_min + edge_t_max);
        const Vector3fAD edge_point = edge_pos + edge_t * edge_dir;
        const Vector3fAD source =
            gather<Vector3fAD>(states.src, state_idx, state_active);
        const Vector3fAD receiver =
            gather<Vector3fAD>(rx_positions, rx_idx, lane_active);
        const FloatAD src_power =
            gather<FloatAD>(states.src_power, state_idx, state_active);
        const IntAD prim0 = gather<IntAD>(states.prim0, state_idx, state_active);
        const IntAD prim1 = gather<IntAD>(states.prim1, state_idx, state_active);
        const FloatAD exterior_angle =
            gather<FloatAD>(states.exterior_angle, state_idx, state_active);
        const IntAD edge_index =
            gather<IntAD>(states.edge_index, state_idx, state_active);

        const MaskAD finite_active =
            state_active &&
            drjit::isfinite(source.x()) &&
            drjit::isfinite(source.y()) &&
            drjit::isfinite(source.z()) &&
            drjit::isfinite(edge_point.x()) &&
            drjit::isfinite(edge_point.y()) &&
            drjit::isfinite(edge_point.z()) &&
            drjit::isfinite(receiver.x()) &&
            drjit::isfinite(receiver.y()) &&
            drjit::isfinite(receiver.z()) &&
            drjit::isfinite(src_power);
        const SegmentPairVisibilityAD visibility =
            this->template visible_pair<false>(
                edge_point,
                source,
                receiver,
                Int(),
                finite_active);
        const MaskAD visible = visibility.visible_a && visibility.visible_b;
        const FloatAD source_distance =
            maximum(norm(edge_point - source), FloatAD(Epsilon));
        const FloatAD receiver_distance =
            maximum(norm(receiver - edge_point), FloatAD(Epsilon));
        const FloatAD edge_length =
            maximum(edge_t_max - edge_t_min, FloatAD(0.f));
        const FloatAD wedge_scale =
            minimum(
                maximum(exterior_angle, FloatAD(0.25f * Pi)) / FloatAD(2.f * Pi),
                FloatAD(2.f));
        const FloatAD material_gain =
            material_gain_for_faces(prim0, prim1, finite_active);
        const FloatAD wave_gain =
            FloatAD(options.wavelength) / FloatAD(4.f * Pi);
        const FloatAD contribution =
            src_power *
            material_gain *
            edge_length *
            wedge_scale *
            wave_gain *
            wave_gain /
            (source_distance * source_distance * receiver_distance * receiver_distance);
        const MaskAD path_active =
            visible && (contribution > FloatAD(0.f)) && drjit::isfinite(contribution);
        const FloatAD path_length = source_distance + receiver_distance;
        const FloatAD phase = -FloatAD(options.k) * path_length;
        const FloatAD amplitude = sqrt(maximum(contribution, FloatAD(0.f)));

        result.valid = path_active;
        result.tx_id = select(path_active, tx_idx, IntAD(-1));
        result.rx_id = select(path_active, rx_idx, IntAD(-1));
        result.order = select(path_active, IntAD(1), IntAD(0));
        result.edge0 = select(path_active, edge_index, IntAD(-1));
        result.edge1 = full<IntAD>(-1, capacity);
        result.edge2 = full<IntAD>(-1, capacity);
        result.delay =
            select(path_active, path_length / FloatAD(299792458.f), FloatAD(0.f));
        result.field_x =
            drjit::Complex<FloatAD>(
                select(path_active, amplitude * cos(phase), FloatAD(0.f)),
                select(path_active, amplitude * sin(phase), FloatAD(0.f)));
        result.field_y =
            drjit::Complex<FloatAD>(zeros<FloatAD>(capacity), zeros<FloatAD>(capacity));
        result.field_z =
            drjit::Complex<FloatAD>(zeros<FloatAD>(capacity), zeros<FloatAD>(capacity));
        result.p0 = select(path_active, edge_point, zeros<Vector3fAD>(capacity));
        result.p1 = zeros<Vector3fAD>(capacity);
        result.p2 = zeros<Vector3fAD>(capacity);
        scatter_reduce(
            ReduceOp::Add,
            result.count,
            IntAD(1),
            zeros<IntAD>(capacity),
            path_active);
        return result;
    } else {
        const int tx_count = static_cast<int>(slices(tx_positions));
        const int rx_width = static_cast<int>(slices(rx_positions));
        const int rx_count = options.max_rx > 0
                                 ? std::min(rx_width, options.max_rx)
                                 : rx_width;
        const int state_width = static_cast<int>(slices(states.edge_index));
        const int state_count = states.count > 0 ? states.count : state_width;
        if (tx_count == 0 || rx_count == 0 || state_count == 0) {
            result.capacity = 0;
            result.count = full<Int>(0, 1);
            result.valid = full<Mask>(false, 0);
            result.tx_id = full<Int>(-1, 0);
            result.rx_id = full<Int>(-1, 0);
            result.order = full<Int>(0, 0);
            result.edge0 = full<Int>(-1, 0);
            result.edge1 = full<Int>(-1, 0);
            result.edge2 = full<Int>(-1, 0);
            result.delay = zeros<Float>(0);
            result.field_x = drjit::Complex<Float>(zeros<Float>(0), zeros<Float>(0));
            result.field_y = drjit::Complex<Float>(zeros<Float>(0), zeros<Float>(0));
            result.field_z = drjit::Complex<Float>(zeros<Float>(0), zeros<Float>(0));
            result.p0 = zeros<Vector3f>(0);
            result.p1 = zeros<Vector3f>(0);
            result.p2 = zeros<Vector3f>(0);
            return result;
        }
        require(state_count > 0 && state_count <= state_width,
                "Scene::trace_dfr_paths(): invalid state count.");
        require(static_cast<int>(slices(states.edge_pos)) >= state_count &&
                    static_cast<int>(slices(states.edge_dir)) >= state_count &&
                    static_cast<int>(slices(states.edge_t_min)) >= state_count &&
                    static_cast<int>(slices(states.edge_t_max)) >= state_count &&
                    static_cast<int>(slices(states.n0)) >= state_count &&
                    static_cast<int>(slices(states.n1)) >= state_count &&
                    static_cast<int>(slices(states.prim0)) >= state_count &&
                    static_cast<int>(slices(states.prim1)) >= state_count &&
                    static_cast<int>(slices(states.exterior_angle)) >= state_count &&
                    static_cast<int>(slices(states.src)) >= state_count &&
                    static_cast<int>(slices(states.src_power)) >= state_count,
                "Scene::trace_dfr_paths(): state fields must cover state count.");
        require(rx_count <= rx_width,
                "Scene::trace_dfr_paths(): invalid receiver count.");

        const int material_count = static_cast<int>(slices(material.eta_r));
        require(material_count > 0,
                "Scene::trace_dfr_paths(): material payload must not be empty.");
        require(static_cast<int>(slices(material.sigma)) == material_count &&
                    static_cast<int>(slices(material.mu_r)) == material_count &&
                    static_cast<int>(slices(material.gain)) == material_count &&
                    static_cast<int>(slices(material.valid)) == material_count,
                "Scene::trace_dfr_paths(): material payload fields must have matching widths.");

        Mask active_detached = active;
        int active_width = static_cast<int>(slices(active_detached));
        if (active_width == 1 && state_count > 1) {
            active_detached = gather<Mask>(active_detached, zeros<Int>(state_count));
            active_width = state_count;
        } else {
            require(active_width == state_count,
                    "Scene::trace_dfr_paths(): active width must be 1 or match state count.");
        }
        active_detached &= drjit::isfinite(states.src.x()) &&
                           drjit::isfinite(states.src.y()) &&
                           drjit::isfinite(states.src.z()) &&
                           drjit::isfinite(states.edge_pos.x()) &&
                           drjit::isfinite(states.edge_pos.y()) &&
                           drjit::isfinite(states.edge_pos.z()) &&
                           drjit::isfinite(states.src_power);

        const int state_limit = std::min(state_count, options.max_paths);
        const int64_t capacity64 =
            static_cast<int64_t>(tx_count) *
            static_cast<int64_t>(rx_count) *
            static_cast<int64_t>(state_limit);
        require(capacity64 <= static_cast<int64_t>(std::numeric_limits<int>::max()),
                "Scene::trace_dfr_paths(): requested path capacity exceeds int range.");
        const int capacity = static_cast<int>(capacity64);

        const OptixSceneSelection scenes = select_optix_scenes();
        const OptixScene *primary_scene = scenes.primary;
        const OptixScene *secondary_scene = scenes.secondary;
        const int split_mode = scenes.split_mode;
        const int hitgroup_record_count = scenes.hitgroup_record_count;
        require(primary_scene != nullptr && primary_scene->is_ready(),
                "Scene::trace_dfr_paths(): OptiX scene is not ready.");
        require(hitgroup_record_count > 0,
                "Scene::trace_dfr_paths(): invalid hitgroup record count.");

        ensure_pipeline(diffraction_paths_pipeline_,
                        primary_scene->context(),
                        hitgroup_record_count,
                        diffraction_paths_pipeline_config());

        drjit::eval(tx_positions,
                    rx_positions,
                    states.edge_index,
                    states.edge_pos,
                    states.edge_dir,
                    states.edge_t_min,
                    states.edge_t_max,
                    states.n0,
                    states.n1,
                    states.prim0,
                    states.prim1,
                    states.exterior_angle,
                    states.src,
                    states.src_power,
                    active_detached,
                    material.eta_r,
                    material.sigma,
                    material.mu_r,
                    material.gain,
                    material.valid);

        DfrPathsRaw raw = alloc_dfr_paths_raw(capacity);
        init_dfr_paths_raw(raw);

        DfrPathParams params = {};
        params.primary_handle = primary_scene->ias_handle();
        params.secondary_handle =
            secondary_scene != nullptr && secondary_scene->is_ready() ? secondary_scene->ias_handle() : 0ull;
        params.split_mode = split_mode;
        params.n_rays = capacity;
        params.capacity = capacity;
        params.tx_pos_x = tx_positions.x().data();
        params.tx_pos_y = tx_positions.y().data();
        params.tx_pos_z = tx_positions.z().data();
        params.tx_count = tx_count;
        params.rx_pos_x = rx_positions.x().data();
        params.rx_pos_y = rx_positions.y().data();
        params.rx_pos_z = rx_positions.z().data();
        params.rx_count = rx_count;
        params.active_mask = reinterpret_cast<const uint8_t *>(active_detached.data());
        params.active_width = active_width;
        params.state_count = state_count;
        params.state_limit = state_limit;
        params.state_edge_index = states.edge_index.data();
        params.state_edge_pos_x = states.edge_pos.x().data();
        params.state_edge_pos_y = states.edge_pos.y().data();
        params.state_edge_pos_z = states.edge_pos.z().data();
        params.state_edge_dir_x = states.edge_dir.x().data();
        params.state_edge_dir_y = states.edge_dir.y().data();
        params.state_edge_dir_z = states.edge_dir.z().data();
        params.state_edge_t_min = states.edge_t_min.data();
        params.state_edge_t_max = states.edge_t_max.data();
        params.state_n0_x = states.n0.x().data();
        params.state_n0_y = states.n0.y().data();
        params.state_n0_z = states.n0.z().data();
        params.state_n1_x = states.n1.x().data();
        params.state_n1_y = states.n1.y().data();
        params.state_n1_z = states.n1.z().data();
        params.state_prim0 = states.prim0.data();
        params.state_prim1 = states.prim1.data();
        params.state_exterior_angle = states.exterior_angle.data();
        params.state_src_x = states.src.x().data();
        params.state_src_y = states.src.y().data();
        params.state_src_z = states.src.z().data();
        params.state_src_power = states.src_power.data();
        params.material_gain = material.gain.data();
        params.material_valid = reinterpret_cast<const uint8_t *>(material.valid.data());
        params.material_count = material_count;
        params.wavelength = options.wavelength;
        params.k = options.k;
        params.seed = options.seed;
        params.max_order = options.max_order;
        params.strategy_mask = options.strategy_mask;
        params.sample_count = options.sample_count;
        params.return_geom = options.return_geom;
        params.receiver_model = options.receiver_model;
        params.out_count = raw.count.data();
        params.out_valid = reinterpret_cast<uint8_t *>(raw.valid.data());
        params.out_tx_id = raw.tx_id.data();
        params.out_rx_id = raw.rx_id.data();
        params.out_order = raw.order.data();
        params.out_edge0 = raw.edge0.data();
        params.out_edge1 = raw.edge1.data();
        params.out_edge2 = raw.edge2.data();
        params.out_delay = raw.delay.data();
        params.out_field_x_re = raw.field_x_re.data();
        params.out_field_x_im = raw.field_x_im.data();
        params.out_field_y_re = raw.field_y_re.data();
        params.out_field_y_im = raw.field_y_im.data();
        params.out_field_z_re = raw.field_z_re.data();
        params.out_field_z_im = raw.field_z_im.data();
        params.out_p0_x = raw.p0.x().data();
        params.out_p0_y = raw.p0.y().data();
        params.out_p0_z = raw.p0.z().data();
        params.out_p1_x = raw.p1.x().data();
        params.out_p1_y = raw.p1.y().data();
        params.out_p1_z = raw.p1.z().data();
        params.out_p2_x = raw.p2.x().data();
        params.out_p2_y = raw.p2.y().data();
        params.out_p2_z = raw.p2.z().data();

        diffraction_paths_pipeline_->launch(0, params);

        result.capacity = capacity;
        result.count = raw.count;
        result.valid = raw.valid;
        result.tx_id = raw.tx_id;
        result.rx_id = raw.rx_id;
        result.order = raw.order;
        result.edge0 = raw.edge0;
        result.edge1 = raw.edge1;
        result.edge2 = raw.edge2;
        result.delay = raw.delay;
        result.field_x = drjit::Complex<Float>(raw.field_x_re, raw.field_x_im);
        result.field_y = drjit::Complex<Float>(raw.field_y_re, raw.field_y_im);
        result.field_z = drjit::Complex<Float>(raw.field_z_re, raw.field_z_im);
        result.p0 = raw.p0;
        result.p1 = raw.p1;
        result.p2 = raw.p2;
        return result;
    }
}

template <bool Detached>
ReflEpcT<Detached> Scene::trace_refl_epc(
    const RayT<Detached> &ray,
    const Vector3fT<Detached> &receiver,
    int max_bounces,
    const ReflEpcOptions &options,
    MaskT<Detached> active) const {
    ScopedNativeLaunchStage native_launch_stage(NativeLaunchStage::TraceReflections);
    require(is_ready(), "Scene::trace_refl_epc(): scene is not built.");
    require(!pending_updates_,
            "Scene::trace_refl_epc(): scene has pending updates. Call Scene::sync() first.");
    require(max_bounces > 0,
            "Scene::trace_refl_epc(): max_bounces must be positive.");
    require(max_bounces <= ReflEpcMaxBounces,
            "Scene::trace_refl_epc(): max_bounces exceeds the native EPC limit.");

    const int ray_count = static_cast<int>(slices(ray.o));
    ReflEpcT<Detached> result =
        init_refl_epc<Detached>(ray_count, max_bounces);
    if (ray_count == 0) {
        return result;
    }

    if constexpr (!Detached) {
        require(false,
                "Scene::trace_refl_epc(): native EPC is a non-AD native fast path. "
                "Pass a non-AD Ray and detached receiver positions.");
        return result;
    } else {
        const int receiver_count = static_cast<int>(slices(receiver));
        require(receiver_count == 1 || receiver_count == ray_count,
                "Scene::trace_refl_epc(): receiver width must be 1 or match ray count.");
        const ReflEpcVisibilityIgnoreMode visibility_ignore_mode =
            parse_refl_epc_vis_ignore(options.visibility_ignore_mode);
        const bool surface_group_ignore =
            visibility_ignore_mode == ReflEpcVisibilityIgnoreMode::SurfaceGroup;
        const int slot_count = ray_count * max_bounces;
        const int expected_prim_count = static_cast<int>(slices(options.expected_prim_ids));
        const int surface_group_id_count = static_cast<int>(slices(options.surface_group_id));
        const int surface_group_count = static_cast<int>(slices(options.surface_group_size));
        const int surface_group_member_count =
            static_cast<int>(slices(options.surface_group_members));
        const int final_ignore_group_count =
            static_cast<int>(slices(options.final_ignore_group_ids));
        const bool has_surface_groups =
            surface_group_id_count > 0 ||
            surface_group_count > 0 ||
            surface_group_member_count > 0 ||
            options.surface_max_group_size > 0;
        require(expected_prim_count == 0 || expected_prim_count == slot_count,
                "Scene::trace_refl_epc(): expected_prim_ids width must be n_rays * max_bounces.");
        require(final_ignore_group_count == 0 ||
                    final_ignore_group_count == 1 ||
                    final_ignore_group_count == ray_count,
                "Scene::trace_refl_epc(): final_ignore_group_ids width must be 1 or match ray count.");
        if (has_surface_groups) {
            const int triangle_count =
                static_cast<int>(slices(triangle_info_detached_.p0));
            require(surface_group_id_count == triangle_count,
                    "Scene::trace_refl_epc(): surface_group_id width must match triangle count.");
            require(surface_group_count > 0,
                    "Scene::trace_refl_epc(): surface_group_size must be non-empty when surface groups are provided.");
            require(options.surface_max_group_size > 0,
                    "Scene::trace_refl_epc(): surface_max_group_size must be positive when surface groups are provided.");
            require(surface_group_member_count >=
                        surface_group_count * options.surface_max_group_size,
                    "Scene::trace_refl_epc(): surface_group_members must contain surface_group_size * surface_max_group_size entries.");
        }
        require(!surface_group_ignore || has_surface_groups,
                "Scene::trace_refl_epc(): visibility_ignore_mode='surface_group' requires surface group tables.");
        require(final_ignore_group_count == 0 || surface_group_ignore,
                "Scene::trace_refl_epc(): final_ignore_group_ids require visibility_ignore_mode='surface_group'.");

        const Mask active_detached =
            sanitize_reflection_active<Detached>(ray, active);
        if (drjit::none(active_detached)) {
            return result;
        }

        const OptixSceneSelection scenes = select_optix_scenes();
        const OptixScene *primary_scene = scenes.primary;
        const OptixScene *secondary_scene = scenes.secondary;
        int split_mode = scenes.split_mode;
        int hitgroup_record_count = scenes.hitgroup_record_count;

        require(primary_scene != nullptr && primary_scene->is_ready(),
                "Scene::trace_refl_epc(): OptiX scene is not ready.");
        require(hitgroup_record_count > 0,
                "Scene::trace_refl_epc(): invalid hitgroup record count.");

        drjit::eval(ray.o,
                    ray.d,
                    ray.tmax,
                    receiver,
                    active_detached);
        if (expected_prim_count > 0) {
            drjit::eval(options.expected_prim_ids);
        }
        if (has_surface_groups) {
            drjit::eval(options.surface_group_id,
                        options.surface_group_size,
                        options.surface_group_members);
        }
        if (final_ignore_group_count > 0) {
            drjit::eval(options.final_ignore_group_ids);
        }

        ensure_pipeline(segment_pair_visibility_pipeline_, primary_scene->context(),
                        hitgroup_record_count, segment_pair_visibility_pipeline_config());
        ensure_pipeline(reflection_epc_pipeline_, primary_scene->context(),
                        hitgroup_record_count, reflection_epc_pipeline_config());

        ensure_reflection_epc_geometry_ready();

        ReflEpcRaw raw = alloc_refl_epc_raw(ray_count, max_bounces);

        ReflEpcParams params = {};
        params.primary_handle = primary_scene->ias_handle();
        params.secondary_handle =
            secondary_scene != nullptr && secondary_scene->is_ready()
                ? secondary_scene->ias_handle()
                : 0ull;
        params.split_mode = split_mode;
        params.tri_p0_x = triangle_info_detached_.p0.x().data();
        params.tri_p0_y = triangle_info_detached_.p0.y().data();
        params.tri_p0_z = triangle_info_detached_.p0.z().data();
        params.tri_e1_x = triangle_info_detached_.e1.x().data();
        params.tri_e1_y = triangle_info_detached_.e1.y().data();
        params.tri_e1_z = triangle_info_detached_.e1.z().data();
        params.tri_e2_x = triangle_info_detached_.e2.x().data();
        params.tri_e2_y = triangle_info_detached_.e2.y().data();
        params.tri_e2_z = triangle_info_detached_.e2.z().data();
        params.tri_fn_x = triangle_info_detached_.face_normal.x().data();
        params.tri_fn_y = triangle_info_detached_.face_normal.y().data();
        params.tri_fn_z = triangle_info_detached_.face_normal.z().data();
        params.face_offsets = face_offsets_.data();
        params.n_meshes = mesh_count_;
        params.n_triangles = static_cast<int>(slices(triangle_info_detached_.p0));
        params.expected_prim_ids =
            expected_prim_count > 0 ? options.expected_prim_ids.data() : nullptr;
        params.expected_prim_count = expected_prim_count;
        params.surface_group_id =
            has_surface_groups ? options.surface_group_id.data() : nullptr;
        params.surface_group_id_count = surface_group_id_count;
        params.surface_group_size =
            has_surface_groups ? options.surface_group_size.data() : nullptr;
        params.surface_group_count = surface_group_count;
        params.surface_group_members =
            has_surface_groups ? options.surface_group_members.data() : nullptr;
        params.surface_max_group_size =
            has_surface_groups ? options.surface_max_group_size : 0;
        params.visibility_ignore_mode =
            surface_group_ignore ? ReflEpcVisibilityIgnoreSurfaceGroup
                                 : ReflEpcVisibilityIgnorePrimitive;
        params.final_ignore_group_ids =
            final_ignore_group_count > 0 ? options.final_ignore_group_ids.data() : nullptr;
        params.final_ignore_group_count = final_ignore_group_count;
        params.ray_ox = ray.o.x().data();
        params.ray_oy = ray.o.y().data();
        params.ray_oz = ray.o.z().data();
        params.ray_dx = ray.d.x().data();
        params.ray_dy = ray.d.y().data();
        params.ray_dz = ray.d.z().data();
        params.ray_tmax = ray.tmax.data();
        params.rx_x = receiver.x().data();
        params.rx_y = receiver.y().data();
        params.rx_z = receiver.z().data();
        params.rx_count = receiver_count;
        params.active_mask = reinterpret_cast<const uint8_t *>(active_detached.data());
        params.n_rays = ray_count;
        params.max_bounces = max_bounces;
        params.out_valid = reinterpret_cast<uint8_t *>(raw.valid.data());
        params.out_bounce_count = raw.bounce_count.data();
        params.out_path_length = raw.path_length.data();
        params.out_point_x = raw.point_x.data();
        params.out_point_y = raw.point_y.data();
        params.out_point_z = raw.point_z.data();
        params.out_trace_prim_ids = raw.trace_prim_ids.data();
        params.out_resolved_prim_ids = raw.resolved_prim_ids.data();
        params.out_surface_group_ids = raw.surface_group_ids.data();
        params.out_plane_normal_x = raw.plane_normal_x.data();
        params.out_plane_normal_y = raw.plane_normal_y.data();
        params.out_plane_normal_z = raw.plane_normal_z.data();
        params.out_first_blocked_segment = raw.first_blocked_segment.data();
        params.out_first_blocked_prim = raw.first_blocked_prim.data();
        params.out_first_blocked_group = raw.first_blocked_group.data();

        reflection_epc_pipeline_->launch(0, params);

        result.valid = raw.valid;
        result.bounce_count = raw.bounce_count;
        result.path_length = raw.path_length;
        result.reflection_points =
            Vector3f(raw.point_x, raw.point_y, raw.point_z);
        result.prim_ids = raw.trace_prim_ids;
        result.trace_prim_ids = raw.trace_prim_ids;
        result.resolved_prim_ids = raw.resolved_prim_ids;
        result.surface_group_ids = raw.surface_group_ids;
        result.plane_normals =
            Vector3f(raw.plane_normal_x,
                             raw.plane_normal_y,
                             raw.plane_normal_z);
        result.first_blocked_segment = raw.first_blocked_segment;
        result.first_blocked_prim = raw.first_blocked_prim;
        result.first_blocked_group = raw.first_blocked_group;
        return result;
    }
}

template <bool Detached>
ReflEpcFieldT<Detached> Scene::trace_refl_epc_field(
    const RayT<Detached> &ray,
    const Vector3fT<Detached> &receiver,
    int max_bounces,
    const ReflEpcFieldOptionsT<Detached> &options,
    MaskT<Detached> active) const {
    ScopedNativeLaunchStage native_launch_stage(NativeLaunchStage::TraceReflections);
    require(is_ready(), "Scene::trace_refl_epc_field(): scene is not built.");
    require(!pending_updates_,
            "Scene::trace_refl_epc_field(): scene has pending updates. Call Scene::sync() first.");
    require(max_bounces > 0,
            "Scene::trace_refl_epc_field(): max_bounces must be positive.");
    require(max_bounces <= ReflEpcMaxBounces,
            "Scene::trace_refl_epc_field(): max_bounces exceeds the native EPC limit.");
    require(options.omega > 0.f,
            "Scene::trace_refl_epc_field(): omega must be positive.");
    require(options.wavelength > 0.f,
            "Scene::trace_refl_epc_field(): wavelength must be positive.");

    const int ray_count = static_cast<int>(slices(ray.o));
    ReflEpcFieldT<Detached> result =
        init_refl_epc_field<Detached>(
            ray_count,
            max_bounces,
            options);
    if (ray_count == 0) {
        return result;
    }

    if constexpr (!Detached) {
        require(false,
                "Scene::trace_refl_epc_field(): native EPC field is a non-AD native fast path. "
                "Pass a non-AD Ray and detached receiver positions.");
        return result;
    } else {
        const int receiver_count = static_cast<int>(slices(receiver));
        require(receiver_count == 1 || receiver_count == ray_count,
                "Scene::trace_refl_epc_field(): receiver width must be 1 or match ray count.");
        const int slot_count = ray_count * max_bounces;
        require(static_cast<int>(slices(options.slot_plane_normal)) == slot_count,
                "Scene::trace_refl_epc_field(): slot_plane_normal width must be n_rays * max_bounces.");
        require(static_cast<int>(slices(options.slot_eta_r)) == slot_count,
                "Scene::trace_refl_epc_field(): slot_eta_r width must be n_rays * max_bounces.");
        require(static_cast<int>(slices(options.slot_mu_r)) == slot_count,
                "Scene::trace_refl_epc_field(): slot_mu_r width must be n_rays * max_bounces.");
        require(static_cast<int>(slices(options.slot_sigma)) == slot_count,
                "Scene::trace_refl_epc_field(): slot_sigma width must be n_rays * max_bounces.");
        require(static_cast<int>(slices(options.slot_gain)) == slot_count,
                "Scene::trace_refl_epc_field(): slot_gain width must be n_rays * max_bounces.");
        const int tx_pol_count = static_cast<int>(slices(options.tx_polarization));
        require(tx_pol_count == 1 || tx_pol_count == ray_count,
                "Scene::trace_refl_epc_field(): tx_polarization width must be 1 or match ray count.");

        drjit::eval(options.slot_plane_normal,
                    options.slot_eta_r,
                    options.slot_mu_r,
                    options.slot_sigma,
                    options.slot_gain,
                    options.tx_polarization);

        const ReflEpcOptions epc_options =
            epc_options_from_field_options(options);
        const ReflEpc epc =
            trace_refl_epc<true>(
                ray,
                receiver,
                max_bounces,
                epc_options,
                active);

        ReflEpcFieldParams params = {};
        params.n_rays = ray_count;
        params.max_bounces = max_bounces;
        params.epc_valid = reinterpret_cast<const uint8_t *>(epc.valid.data());
        params.epc_bounce_count = epc.bounce_count.data();
        params.epc_path_length = epc.path_length.data();
        params.ray_ox = ray.o.x().data();
        params.ray_oy = ray.o.y().data();
        params.ray_oz = ray.o.z().data();
        params.rx_x = receiver.x().data();
        params.rx_y = receiver.y().data();
        params.rx_z = receiver.z().data();
        params.rx_count = receiver_count;
        params.hit_x = epc.reflection_points.x().data();
        params.hit_y = epc.reflection_points.y().data();
        params.hit_z = epc.reflection_points.z().data();
        params.epc_normal_x = epc.plane_normals.x().data();
        params.epc_normal_y = epc.plane_normals.y().data();
        params.epc_normal_z = epc.plane_normals.z().data();
        const bool return_resolved_prim_ids =
            options.return_geom && options.return_resolved_prim_ids;
        const bool return_surface_group_ids =
            options.return_geom && options.return_surface_group_ids;
        params.resolved_prim_ids =
            return_resolved_prim_ids ? epc.resolved_prim_ids.data() : nullptr;
        params.surface_group_ids =
            return_surface_group_ids ? epc.surface_group_ids.data() : nullptr;
        params.slot_normal_x = options.slot_plane_normal.x().data();
        params.slot_normal_y = options.slot_plane_normal.y().data();
        params.slot_normal_z = options.slot_plane_normal.z().data();
        params.slot_eta_r = options.slot_eta_r.data();
        params.slot_mu_r = options.slot_mu_r.data();
        params.slot_sigma = options.slot_sigma.data();
        params.slot_gain = options.slot_gain.data();
        params.tx_pol_x = options.tx_polarization.x().data();
        params.tx_pol_y = options.tx_polarization.y().data();
        params.tx_pol_z = options.tx_polarization.z().data();
        params.tx_pol_count = tx_pol_count;
        params.omega = options.omega;
        params.wavelength = options.wavelength;
        params.out_valid = reinterpret_cast<uint8_t *>(result.valid.data());
        params.out_bounce_count = result.bounce_count.data();
        params.out_path_length = result.path_length.data();
        params.out_field_x_re = result.field_x_re.data();
        params.out_field_x_im = result.field_x_im.data();
        params.out_field_y_re = result.field_y_re.data();
        params.out_field_y_im = result.field_y_im.data();
        params.out_field_z_re = result.field_z_re.data();
        params.out_field_z_im = result.field_z_im.data();

        if (options.return_endpoints) {
            params.out_tx_x = result.tx_pos.x().data();
            params.out_tx_y = result.tx_pos.y().data();
            params.out_tx_z = result.tx_pos.z().data();
            params.out_first_hit_x = result.first_hit.x().data();
            params.out_first_hit_y = result.first_hit.y().data();
            params.out_first_hit_z = result.first_hit.z().data();
            params.out_last_hit_x = result.last_hit.x().data();
            params.out_last_hit_y = result.last_hit.y().data();
            params.out_last_hit_z = result.last_hit.z().data();
        }
        if (options.return_geom && options.return_hit_points) {
            params.out_hit_x = result.hit_points.x().data();
            params.out_hit_y = result.hit_points.y().data();
            params.out_hit_z = result.hit_points.z().data();
        }
        if (options.return_geom && options.return_normals) {
            params.out_normal_x = result.normals.x().data();
            params.out_normal_y = result.normals.y().data();
            params.out_normal_z = result.normals.z().data();
        }
        if (return_resolved_prim_ids) {
            params.out_resolved_prim_ids = result.resolved_prim_ids.data();
        }
        if (return_surface_group_ids) {
            params.out_surface_group_ids = result.surface_group_ids.data();
        }

        reflection_epc_field_gpu(params);
        return result;
    }
}

template <bool Detached>
ReflEpcFieldT<Detached> Scene::trace_refl_epc_field(
    const Vector3fT<Detached> &tx_position,
    const Vector3fT<Detached> &receiver,
    int max_bounces,
    const ReflEpcFieldOptionsT<Detached> &options,
    MaskT<Detached> active) const {
    ScopedNativeLaunchStage native_launch_stage(NativeLaunchStage::TraceReflections);
    require(is_ready(), "Scene::trace_refl_epc_field(): scene is not built.");
    require(!pending_updates_,
            "Scene::trace_refl_epc_field(): scene has pending updates. Call Scene::sync() first.");
    require(max_bounces > 0,
            "Scene::trace_refl_epc_field(): max_bounces must be positive.");
    require(max_bounces <= ReflEpcMaxBounces,
            "Scene::trace_refl_epc_field(): max_bounces exceeds the native EPC limit.");
    require(options.omega > 0.f,
            "Scene::trace_refl_epc_field(): omega must be positive.");
    require(options.wavelength > 0.f,
            "Scene::trace_refl_epc_field(): wavelength must be positive.");

    const int ray_count = static_cast<int>(slices(tx_position));
    ReflEpcFieldT<Detached> result =
        init_refl_epc_field<Detached>(
            ray_count,
            max_bounces,
            options);
    if (ray_count == 0) {
        return result;
    }

    if constexpr (!Detached) {
        const int receiver_count = static_cast<int>(slices(receiver));
        require(receiver_count == 1 || receiver_count == ray_count,
                "Scene::trace_refl_epc_field(): receiver width must be 1 or match transmitter count.");
        const int slot_count = ray_count * max_bounces;
        const int expected_prim_count = static_cast<int>(slices(options.expected_prim_ids));
        require(expected_prim_count == slot_count,
                "Scene::trace_refl_epc_field(): expected_prim_ids width must be n_rays * max_bounces.");
        require(static_cast<int>(slices(options.slot_plane_point)) == slot_count,
                "Scene::trace_refl_epc_field(): slot_plane_point width must be n_rays * max_bounces.");
        require(static_cast<int>(slices(options.slot_plane_normal)) == slot_count,
                "Scene::trace_refl_epc_field(): slot_plane_normal width must be n_rays * max_bounces.");
        require(static_cast<int>(slices(options.slot_eta_r)) == slot_count,
                "Scene::trace_refl_epc_field(): slot_eta_r width must be n_rays * max_bounces.");
        require(static_cast<int>(slices(options.slot_mu_r)) == slot_count,
                "Scene::trace_refl_epc_field(): slot_mu_r width must be n_rays * max_bounces.");
        require(static_cast<int>(slices(options.slot_sigma)) == slot_count,
                "Scene::trace_refl_epc_field(): slot_sigma width must be n_rays * max_bounces.");
        require(static_cast<int>(slices(options.slot_gain)) == slot_count,
                "Scene::trace_refl_epc_field(): slot_gain width must be n_rays * max_bounces.");
        const int tx_pol_count = static_cast<int>(slices(options.tx_polarization));
        require(tx_pol_count == 1 || tx_pol_count == ray_count,
                "Scene::trace_refl_epc_field(): tx_polarization width must be 1 or match transmitter count.");

        Mask active_detached = sanitize_segment_active<false>(
            tx_position,
            receiver,
            active);
        if (drjit::none(active_detached)) {
            result.valid = full<MaskAD>(false, ray_count);
            result.bounce_count = full<IntAD>(0, ray_count);
            result.path_length = full<FloatAD>(Infinity, ray_count);
            result.field_x_re = zeros<FloatAD>(ray_count);
            result.field_x_im = zeros<FloatAD>(ray_count);
            result.field_y_re = zeros<FloatAD>(ray_count);
            result.field_y_im = zeros<FloatAD>(ray_count);
            result.field_z_re = zeros<FloatAD>(ray_count);
            result.field_z_im = zeros<FloatAD>(ray_count);
            return result;
        }

        const Int slot_base = arange<Int>(ray_count) * Int(max_bounces);
        const MaskAD active_ad = MaskAD(active_detached);

        struct ComplexADValue {
            FloatAD re;
            FloatAD im;
        };
        struct ComplexVectorAD {
            Vector3fAD re;
            Vector3fAD im;
        };

        auto complex_add = [](const ComplexADValue &a,
                              const ComplexADValue &b) -> ComplexADValue {
            return {a.re + b.re, a.im + b.im};
        };
        auto complex_sub = [](const ComplexADValue &a,
                              const ComplexADValue &b) -> ComplexADValue {
            return {a.re - b.re, a.im - b.im};
        };
        auto complex_mul = [](const ComplexADValue &a,
                              const ComplexADValue &b) -> ComplexADValue {
            return {a.re * b.re - a.im * b.im,
                    a.re * b.im + a.im * b.re};
        };
        auto complex_scale = [](const ComplexADValue &a,
                                const FloatAD &scale) -> ComplexADValue {
            return {a.re * scale, a.im * scale};
        };
        auto complex_div = [](const ComplexADValue &a,
                              const ComplexADValue &b) -> ComplexADValue {
            const FloatAD denom =
                maximum(b.re * b.re + b.im * b.im, FloatAD(Epsilon));
            return {(a.re * b.re + a.im * b.im) / denom,
                    (a.im * b.re - a.re * b.im) / denom};
        };
        auto complex_sqrt = [](const ComplexADValue &a) -> ComplexADValue {
            const FloatAD mag =
                sqrt(maximum(a.re * a.re + a.im * a.im, FloatAD(0.f)));
            const MaskAD positive_real_axis =
                (abs(a.im) <= FloatAD(Epsilon)) && (a.re > FloatAD(Epsilon));
            const FloatAD real_part =
                sqrt(maximum(FloatAD(0.5f) * (mag + a.re), FloatAD(0.f)));
            const FloatAD imag_abs =
                sqrt(maximum(FloatAD(0.5f) * (mag - a.re), FloatAD(1e-20f)));
            const FloatAD imag_sign =
                select(a.im < FloatAD(0.f), FloatAD(-1.f), FloatAD(1.f));
            return {
                select(positive_real_axis, sqrt(a.re), real_part),
                select(positive_real_axis, FloatAD(0.f), imag_sign * imag_abs),
            };
        };
        auto normalize_safe = [](const Vector3fAD &value,
                                 const Vector3fAD &fallback) -> Vector3fAD {
            const FloatAD value_norm = norm(value);
            return select(value_norm > FloatAD(Epsilon),
                          value / maximum(value_norm, FloatAD(Epsilon)),
                          fallback);
        };
        auto stable_perpendicular = [&](const Vector3fAD &direction,
                                        const Vector3fAD &preferred) -> Vector3fAD {
            const Vector3fAD dir =
                normalize_safe(direction, Vector3fAD(FloatAD(1.f), FloatAD(0.f), FloatAD(0.f)));
            const Vector3fAD projected = preferred - dot(preferred, dir) * dir;
            const Vector3fAD axis =
                select(abs(dir.x()) < FloatAD(0.9f),
                       Vector3fAD(FloatAD(1.f), FloatAD(0.f), FloatAD(0.f)),
                       Vector3fAD(FloatAD(0.f), FloatAD(1.f), FloatAD(0.f)));
            const Vector3fAD fallback = axis - dot(axis, dir) * dir;
            return select(squared_norm(projected) > FloatAD(1e-12f),
                          normalize_safe(projected, axis),
                          normalize_safe(fallback,
                                         Vector3fAD(FloatAD(0.f), FloatAD(0.f), FloatAD(1.f))));
        };
        auto complex_dot_real = [](const ComplexVectorAD &field,
                                   const Vector3fAD &basis) -> ComplexADValue {
            return {dot(field.re, basis), dot(field.im, basis)};
        };
        auto slot_reflection_coefficients =
            [&](const IntAD &slot,
                const FloatAD &cos_theta,
                const MaskAD &slot_active) -> std::pair<ComplexADValue, ComplexADValue> {
            const FloatAD eta_r =
                maximum(gather<FloatAD>(options.slot_eta_r, slot, slot_active),
                        FloatAD(Epsilon));
            const FloatAD sigma =
                maximum(gather<FloatAD>(options.slot_sigma, slot, slot_active),
                        FloatAD(0.f));
            const FloatAD gain = gather<FloatAD>(options.slot_gain, slot, slot_active);
            const FloatAD mu_r =
                maximum(gather<FloatAD>(options.slot_mu_r, slot, slot_active),
                        FloatAD(Epsilon));
            const FloatAD omega = maximum(FloatAD(options.omega), FloatAD(Epsilon));
            const ComplexADValue eta = {
                eta_r,
                -sigma / (omega * FloatAD(8.854187817e-12f))
            };
            const ComplexADValue mu = {mu_r, FloatAD(0.f)};
            const FloatAD cos_clamped =
                minimum(maximum(abs(cos_theta), FloatAD(Epsilon)), FloatAD(1.f));
            const FloatAD sin2 =
                maximum(FloatAD(0.f), FloatAD(1.f) - cos_clamped * cos_clamped);
            const ComplexADValue a =
                complex_sqrt(complex_sub(complex_mul(mu, eta),
                                         ComplexADValue{sin2, FloatAD(0.f)}));
            const ComplexADValue mu_cos = {mu_r * cos_clamped, FloatAD(0.f)};
            const ComplexADValue eta_cos = {eta.re * cos_clamped,
                                            eta.im * cos_clamped};
            const ComplexADValue r_te =
                complex_scale(
                    complex_div(complex_sub(mu_cos, a),
                                complex_add(mu_cos, a)),
                    gain);
            const ComplexADValue r_tm =
                complex_scale(
                    complex_div(complex_sub(eta_cos, a),
                                complex_add(eta_cos, a)),
                    gain);
            return {r_te, r_tm};
        };
        auto reflect_field_vector =
            [&](const ComplexVectorAD &field,
                const Vector3fAD &incident_dir,
                const Vector3fAD &slot_normal,
                const IntAD &slot,
                const MaskAD &slot_active) -> ComplexVectorAD {
            const Vector3fAD incident_hat =
                normalize_safe(incident_dir,
                               Vector3fAD(FloatAD(1.f), FloatAD(0.f), FloatAD(0.f)));
            Vector3fAD normal_hat =
                normalize_safe(slot_normal,
                               Vector3fAD(FloatAD(0.f), FloatAD(0.f), FloatAD(1.f)));
            normal_hat = select(dot(incident_hat, normal_hat) > FloatAD(0.f),
                                -normal_hat,
                                normal_hat);
            const FloatAD dot_dn = dot(incident_hat, normal_hat);
            const Vector3fAD reflected_dir =
                normalize_safe(incident_hat - FloatAD(2.f) * dot_dn * normal_hat,
                               -incident_hat);
            Vector3fAD s_hat = cross(normal_hat, incident_hat);
            s_hat = select(squared_norm(s_hat) > FloatAD(1e-12f),
                           normalize_safe(s_hat, stable_perpendicular(incident_hat, normal_hat)),
                           stable_perpendicular(incident_hat, normal_hat));
            Vector3fAD p_in_hat = cross(s_hat, incident_hat);
            p_in_hat =
                select(squared_norm(p_in_hat) > FloatAD(1e-12f),
                       normalize_safe(p_in_hat, stable_perpendicular(incident_hat, normal_hat)),
                       stable_perpendicular(incident_hat, normal_hat));
            Vector3fAD p_out_hat = cross(s_hat, reflected_dir);
            p_out_hat =
                select(squared_norm(p_out_hat) > FloatAD(1e-12f),
                       normalize_safe(p_out_hat, stable_perpendicular(reflected_dir, normal_hat)),
                       stable_perpendicular(reflected_dir, normal_hat));

            const auto [r_te, r_tm] =
                slot_reflection_coefficients(slot, abs(dot(incident_hat, normal_hat)), slot_active);
            const ComplexADValue e_s = complex_dot_real(field, s_hat);
            const ComplexADValue e_p = complex_dot_real(field, p_in_hat);
            const ComplexADValue out_s = complex_mul(r_te, e_s);
            const ComplexADValue out_p = complex_mul(r_tm, e_p);
            return {
                s_hat * out_s.re + p_out_hat * out_p.re,
                s_hat * out_s.im + p_out_hat * out_p.im,
            };
        };

        std::vector<Vector3fAD> images;
        images.reserve(static_cast<size_t>(max_bounces) + 1);
        images.push_back(tx_position);
        MaskAD valid = active_ad;

        for (int bounce = 0; bounce < max_bounces; ++bounce) {
            const Int slot = slot_base + Int(bounce);
            const IntAD slot_ad = IntAD(slot);
            Vector3fAD plane_point =
                gather<Vector3fAD>(options.slot_plane_point, slot_ad, active_ad);
            Vector3fAD plane_normal =
                normalize_safe(gather<Vector3fAD>(options.slot_plane_normal, slot_ad, active_ad),
                               Vector3fAD(FloatAD(0.f), FloatAD(0.f), FloatAD(1.f)));
            const Int expected_prim =
                gather<Int>(options.expected_prim_ids, slot, active_detached);
            valid = valid && MaskAD(expected_prim >= Int(0)) &&
                    (squared_norm(plane_normal) > FloatAD(0.f));
            const FloatAD plane_distance =
                dot(images.back() - plane_point, plane_normal);
            images.push_back(
                select(valid,
                       images.back() - FloatAD(2.f) * plane_distance * plane_normal,
                       images.back()));
        }

        Vector3fAD rx = receiver;
        if (receiver_count == 1 && ray_count > 1) {
            rx = gather<Vector3fAD>(receiver, zeros<IntAD>(ray_count), full<MaskAD>(true, ray_count));
        }
        Vector3fAD target = rx;
        std::vector<Vector3fAD> hits(static_cast<size_t>(max_bounces));
        std::vector<Vector3fAD> normals(static_cast<size_t>(max_bounces));
        for (int bounce = max_bounces - 1; bounce >= 0; --bounce) {
            const Int slot = slot_base + Int(bounce);
            const IntAD slot_ad = IntAD(slot);
            const Vector3fAD plane_point =
                gather<Vector3fAD>(options.slot_plane_point, slot_ad, active_ad);
            const Vector3fAD plane_normal =
                normalize_safe(gather<Vector3fAD>(options.slot_plane_normal, slot_ad, active_ad),
                               Vector3fAD(FloatAD(0.f), FloatAD(0.f), FloatAD(1.f)));
            const Vector3fAD line = target - images[static_cast<size_t>(bounce + 1)];
            const FloatAD denom = dot(line, plane_normal);
            const FloatAD t =
                dot(plane_point - images[static_cast<size_t>(bounce + 1)], plane_normal) /
                denom;
            const MaskAD hit_valid =
                valid &&
                drjit::isfinite(t) &&
                (abs(denom) > FloatAD(Epsilon)) &&
                (t > FloatAD(0.f)) &&
                (t < FloatAD(1.f));
            const Vector3fAD hit =
                images[static_cast<size_t>(bounce + 1)] + t * line;
            hits[static_cast<size_t>(bounce)] = select(hit_valid, hit, zeros<Vector3fAD>(ray_count));
            normals[static_cast<size_t>(bounce)] =
                select(hit_valid, plane_normal, zeros<Vector3fAD>(ray_count));
            if (options.return_geom && options.return_hit_points) {
                scatter(result.hit_points, hits[static_cast<size_t>(bounce)], slot_ad, hit_valid);
            }
            if (options.return_geom && options.return_normals) {
                scatter(result.normals, normals[static_cast<size_t>(bounce)], slot_ad, hit_valid);
            }
            target = select(hit_valid, hit, target);
            valid = hit_valid;
        }

        FloatAD path_length = zeros<FloatAD>(ray_count);
        Vector3fAD previous = tx_position;
        for (int bounce = 0; bounce < max_bounces; ++bounce) {
            const Vector3fAD hit = hits[static_cast<size_t>(bounce)];
            path_length += norm(hit - previous);
            previous = hit;
        }
        path_length += norm(rx - previous);
        valid = valid && (path_length > FloatAD(Epsilon)) && drjit::isfinite(path_length);

        const Int pol_idx =
            tx_pol_count == 1 ? zeros<Int>(ray_count) : arange<Int>(ray_count);
        const Vector3fAD tx_pol =
            gather<Vector3fAD>(options.tx_polarization, IntAD(pol_idx), active_ad);
        const Vector3fAD first_dir =
            normalize_safe(hits.front() - tx_position,
                           Vector3fAD(FloatAD(1.f), FloatAD(0.f), FloatAD(0.f)));
        const Vector3fAD transverse_pol =
            stable_perpendicular(first_dir, tx_pol);
        ComplexVectorAD field = {
            transverse_pol,
            zeros<Vector3fAD>(ray_count),
        };
        Vector3fAD field_previous = tx_position;
        for (int bounce = 0; bounce < max_bounces; ++bounce) {
            const Int slot = slot_base + Int(bounce);
            const IntAD slot_ad = IntAD(slot);
            const Vector3fAD hit = hits[static_cast<size_t>(bounce)];
            const Vector3fAD incident_dir =
                normalize_safe(hit - field_previous,
                               Vector3fAD(FloatAD(1.f), FloatAD(0.f), FloatAD(0.f)));
            field = reflect_field_vector(
                field,
                incident_dir,
                normals[static_cast<size_t>(bounce)],
                slot_ad,
                active_ad);
            field_previous = hit;
        }
        const FloatAD wave_k =
            FloatAD(2.f * Pi) / maximum(FloatAD(options.wavelength), FloatAD(Epsilon));
        const FloatAD phase = -wave_k * path_length;
        const FloatAD amplitude =
            FloatAD(options.wavelength) /
            (FloatAD(4.f * Pi) * maximum(path_length, FloatAD(Epsilon)));
        const FloatAD phase_cos = cos(phase);
        const FloatAD phase_sin = sin(phase);
        const Vector3fAD out_re =
            amplitude * (field.re * phase_cos - field.im * phase_sin);
        const Vector3fAD out_im =
            amplitude * (field.re * phase_sin + field.im * phase_cos);
        valid = valid &&
                drjit::isfinite(out_re.x()) &&
                drjit::isfinite(out_re.y()) &&
                drjit::isfinite(out_re.z()) &&
                drjit::isfinite(out_im.x()) &&
                drjit::isfinite(out_im.y()) &&
                drjit::isfinite(out_im.z());

        result.valid = valid;
        result.bounce_count =
            select(valid, full<IntAD>(max_bounces, ray_count), full<IntAD>(0, ray_count));
        result.path_length =
            select(valid, path_length, full<FloatAD>(Infinity, ray_count));
        result.field_x_re = select(valid, out_re.x(), FloatAD(0.f));
        result.field_x_im = select(valid, out_im.x(), FloatAD(0.f));
        result.field_y_re = select(valid, out_re.y(), FloatAD(0.f));
        result.field_y_im = select(valid, out_im.y(), FloatAD(0.f));
        result.field_z_re = select(valid, out_re.z(), FloatAD(0.f));
        result.field_z_im = select(valid, out_im.z(), FloatAD(0.f));

        if (options.return_endpoints) {
            result.tx_pos = tx_position;
            result.first_hit = max_bounces > 0 ? hits.front() : zeros<Vector3fAD>(ray_count);
            result.last_hit = max_bounces > 0 ? hits.back() : zeros<Vector3fAD>(ray_count);
        }
        if (options.return_geom && options.return_resolved_prim_ids) {
            result.resolved_prim_ids = IntAD(options.expected_prim_ids);
        }
        return result;
    } else {
        const int receiver_count = static_cast<int>(slices(receiver));
        require(receiver_count == 1 || receiver_count == ray_count,
                "Scene::trace_refl_epc_field(): receiver width must be 1 or match transmitter count.");
        const int slot_count = ray_count * max_bounces;
        const int expected_prim_count = static_cast<int>(slices(options.expected_prim_ids));
        require(expected_prim_count == slot_count,
                "Scene::trace_refl_epc_field(): expected_prim_ids width must be n_rays * max_bounces.");
        require(static_cast<int>(slices(options.slot_plane_point)) == slot_count,
                "Scene::trace_refl_epc_field(): slot_plane_point width must be n_rays * max_bounces.");
        require(static_cast<int>(slices(options.slot_plane_normal)) == slot_count,
                "Scene::trace_refl_epc_field(): slot_plane_normal width must be n_rays * max_bounces.");
        require(static_cast<int>(slices(options.slot_eta_r)) == slot_count,
                "Scene::trace_refl_epc_field(): slot_eta_r width must be n_rays * max_bounces.");
        require(static_cast<int>(slices(options.slot_mu_r)) == slot_count,
                "Scene::trace_refl_epc_field(): slot_mu_r width must be n_rays * max_bounces.");
        require(static_cast<int>(slices(options.slot_sigma)) == slot_count,
                "Scene::trace_refl_epc_field(): slot_sigma width must be n_rays * max_bounces.");
        require(static_cast<int>(slices(options.slot_gain)) == slot_count,
                "Scene::trace_refl_epc_field(): slot_gain width must be n_rays * max_bounces.");
        const int tx_pol_count = static_cast<int>(slices(options.tx_polarization));
        require(tx_pol_count == 1 || tx_pol_count == ray_count,
                "Scene::trace_refl_epc_field(): tx_polarization width must be 1 or match transmitter count.");

        const ReflEpcVisibilityIgnoreMode visibility_ignore_mode =
            parse_refl_epc_vis_ignore(options.visibility_ignore_mode);
        const bool surface_group_ignore =
            visibility_ignore_mode == ReflEpcVisibilityIgnoreMode::SurfaceGroup;
        const int surface_group_id_count = static_cast<int>(slices(options.surface_group_id));
        const int surface_group_count = static_cast<int>(slices(options.surface_group_size));
        const int surface_group_member_count =
            static_cast<int>(slices(options.surface_group_members));
        const int final_ignore_group_count =
            static_cast<int>(slices(options.final_ignore_group_ids));
        const bool has_surface_groups =
            surface_group_id_count > 0 ||
            surface_group_count > 0 ||
            surface_group_member_count > 0 ||
            options.surface_max_group_size > 0;
        require(final_ignore_group_count == 0 ||
                    final_ignore_group_count == 1 ||
                    final_ignore_group_count == ray_count,
                "Scene::trace_refl_epc_field(): final_ignore_group_ids width must be 1 or match transmitter count.");
        if (has_surface_groups) {
            const int triangle_count =
                static_cast<int>(slices(triangle_info_detached_.p0));
            require(surface_group_id_count == triangle_count,
                    "Scene::trace_refl_epc_field(): surface_group_id width must match triangle count.");
            require(surface_group_count > 0,
                    "Scene::trace_refl_epc_field(): surface_group_size must be non-empty when surface groups are provided.");
            require(options.surface_max_group_size > 0,
                    "Scene::trace_refl_epc_field(): surface_max_group_size must be positive when surface groups are provided.");
            require(surface_group_member_count >=
                        surface_group_count * options.surface_max_group_size,
                    "Scene::trace_refl_epc_field(): surface_group_members must contain surface_group_size * surface_max_group_size entries.");
        }
        require(!surface_group_ignore || has_surface_groups,
                "Scene::trace_refl_epc_field(): visibility_ignore_mode='surface_group' requires surface group tables.");
        require(final_ignore_group_count == 0 || surface_group_ignore,
                "Scene::trace_refl_epc_field(): final_ignore_group_ids require visibility_ignore_mode='surface_group'.");

        Mask active_detached = sanitize_segment_active<Detached>(
            tx_position,
            receiver,
            active);
        if (drjit::none(active_detached)) {
            return result;
        }

        const OptixSceneSelection scenes = select_optix_scenes();
        const OptixScene *primary_scene = scenes.primary;
        const OptixScene *secondary_scene = scenes.secondary;
        int split_mode = scenes.split_mode;
        int hitgroup_record_count = scenes.hitgroup_record_count;
        require(primary_scene != nullptr && primary_scene->is_ready(),
                "Scene::trace_refl_epc_field(): OptiX scene is not ready.");
        require(hitgroup_record_count > 0,
                "Scene::trace_refl_epc_field(): invalid hitgroup record count.");

        drjit::eval(tx_position,
                    receiver,
                    active_detached,
                    options.expected_prim_ids,
                    options.slot_plane_point,
                    options.slot_plane_normal,
                    options.slot_eta_r,
                    options.slot_mu_r,
                    options.slot_sigma,
                    options.slot_gain,
                    options.tx_polarization);
        if (has_surface_groups) {
            drjit::eval(options.surface_group_id,
                        options.surface_group_size,
                        options.surface_group_members);
        }
        if (final_ignore_group_count > 0) {
            drjit::eval(options.final_ignore_group_ids);
        }

        ensure_pipeline(segment_pair_visibility_pipeline_, primary_scene->context(),
                        hitgroup_record_count, segment_pair_visibility_pipeline_config());
        ensure_pipeline(reflection_epc_pipeline_, primary_scene->context(),
                        hitgroup_record_count, reflection_epc_pipeline_config());

        ensure_reflection_epc_geometry_ready();

        ReflEpcRaw raw = alloc_refl_epc_raw(ray_count, max_bounces);
        ReflEpcParams epc_params = {};
        epc_params.primary_handle = primary_scene->ias_handle();
        epc_params.secondary_handle =
            secondary_scene != nullptr && secondary_scene->is_ready()
                ? secondary_scene->ias_handle()
                : 0ull;
        epc_params.split_mode = split_mode;
        epc_params.tri_p0_x = triangle_info_detached_.p0.x().data();
        epc_params.tri_p0_y = triangle_info_detached_.p0.y().data();
        epc_params.tri_p0_z = triangle_info_detached_.p0.z().data();
        epc_params.tri_e1_x = triangle_info_detached_.e1.x().data();
        epc_params.tri_e1_y = triangle_info_detached_.e1.y().data();
        epc_params.tri_e1_z = triangle_info_detached_.e1.z().data();
        epc_params.tri_e2_x = triangle_info_detached_.e2.x().data();
        epc_params.tri_e2_y = triangle_info_detached_.e2.y().data();
        epc_params.tri_e2_z = triangle_info_detached_.e2.z().data();
        epc_params.tri_fn_x = triangle_info_detached_.face_normal.x().data();
        epc_params.tri_fn_y = triangle_info_detached_.face_normal.y().data();
        epc_params.tri_fn_z = triangle_info_detached_.face_normal.z().data();
        epc_params.face_offsets = face_offsets_.data();
        epc_params.n_meshes = mesh_count_;
        epc_params.n_triangles = static_cast<int>(slices(triangle_info_detached_.p0));
        epc_params.expected_prim_ids = options.expected_prim_ids.data();
        epc_params.expected_prim_count = expected_prim_count;
        epc_params.surface_group_id =
            has_surface_groups ? options.surface_group_id.data() : nullptr;
        epc_params.surface_group_id_count = surface_group_id_count;
        epc_params.surface_group_size =
            has_surface_groups ? options.surface_group_size.data() : nullptr;
        epc_params.surface_group_count = surface_group_count;
        epc_params.surface_group_members =
            has_surface_groups ? options.surface_group_members.data() : nullptr;
        epc_params.surface_max_group_size =
            has_surface_groups ? options.surface_max_group_size : 0;
        epc_params.visibility_ignore_mode =
            surface_group_ignore ? ReflEpcVisibilityIgnoreSurfaceGroup
                                 : ReflEpcVisibilityIgnorePrimitive;
        epc_params.final_ignore_group_ids =
            final_ignore_group_count > 0 ? options.final_ignore_group_ids.data() : nullptr;
        epc_params.final_ignore_group_count = final_ignore_group_count;
        epc_params.ray_ox = tx_position.x().data();
        epc_params.ray_oy = tx_position.y().data();
        epc_params.ray_oz = tx_position.z().data();
        epc_params.ray_dx = nullptr;
        epc_params.ray_dy = nullptr;
        epc_params.ray_dz = nullptr;
        epc_params.ray_tmax = nullptr;
        epc_params.direct_plane_point_x = options.slot_plane_point.x().data();
        epc_params.direct_plane_point_y = options.slot_plane_point.y().data();
        epc_params.direct_plane_point_z = options.slot_plane_point.z().data();
        epc_params.direct_plane_normal_x = options.slot_plane_normal.x().data();
        epc_params.direct_plane_normal_y = options.slot_plane_normal.y().data();
        epc_params.direct_plane_normal_z = options.slot_plane_normal.z().data();
        epc_params.rx_x = receiver.x().data();
        epc_params.rx_y = receiver.y().data();
        epc_params.rx_z = receiver.z().data();
        epc_params.rx_count = receiver_count;
        epc_params.active_mask = reinterpret_cast<const uint8_t *>(active_detached.data());
        epc_params.n_rays = ray_count;
        epc_params.max_bounces = max_bounces;
        epc_params.out_valid = reinterpret_cast<uint8_t *>(raw.valid.data());
        epc_params.out_bounce_count = raw.bounce_count.data();
        epc_params.out_path_length = raw.path_length.data();
        epc_params.out_point_x = raw.point_x.data();
        epc_params.out_point_y = raw.point_y.data();
        epc_params.out_point_z = raw.point_z.data();
        epc_params.out_trace_prim_ids = raw.trace_prim_ids.data();
        epc_params.out_resolved_prim_ids = raw.resolved_prim_ids.data();
        epc_params.out_surface_group_ids = raw.surface_group_ids.data();
        epc_params.out_plane_normal_x = raw.plane_normal_x.data();
        epc_params.out_plane_normal_y = raw.plane_normal_y.data();
        epc_params.out_plane_normal_z = raw.plane_normal_z.data();
        epc_params.out_first_blocked_segment = raw.first_blocked_segment.data();
        epc_params.out_first_blocked_prim = raw.first_blocked_prim.data();
        epc_params.out_first_blocked_group = raw.first_blocked_group.data();
        reflection_epc_pipeline_->launch(0, epc_params);

        ReflEpcFieldParams field_params = {};
        field_params.n_rays = ray_count;
        field_params.max_bounces = max_bounces;
        field_params.epc_valid = reinterpret_cast<const uint8_t *>(raw.valid.data());
        field_params.epc_bounce_count = raw.bounce_count.data();
        field_params.epc_path_length = raw.path_length.data();
        field_params.ray_ox = tx_position.x().data();
        field_params.ray_oy = tx_position.y().data();
        field_params.ray_oz = tx_position.z().data();
        field_params.rx_x = receiver.x().data();
        field_params.rx_y = receiver.y().data();
        field_params.rx_z = receiver.z().data();
        field_params.rx_count = receiver_count;
        field_params.hit_x = raw.point_x.data();
        field_params.hit_y = raw.point_y.data();
        field_params.hit_z = raw.point_z.data();
        field_params.epc_normal_x = raw.plane_normal_x.data();
        field_params.epc_normal_y = raw.plane_normal_y.data();
        field_params.epc_normal_z = raw.plane_normal_z.data();
        const bool return_resolved_prim_ids =
            options.return_geom && options.return_resolved_prim_ids;
        const bool return_surface_group_ids =
            options.return_geom && options.return_surface_group_ids;
        field_params.resolved_prim_ids =
            return_resolved_prim_ids ? raw.resolved_prim_ids.data() : nullptr;
        field_params.surface_group_ids =
            return_surface_group_ids ? raw.surface_group_ids.data() : nullptr;
        field_params.slot_normal_x = options.slot_plane_normal.x().data();
        field_params.slot_normal_y = options.slot_plane_normal.y().data();
        field_params.slot_normal_z = options.slot_plane_normal.z().data();
        field_params.slot_eta_r = options.slot_eta_r.data();
        field_params.slot_mu_r = options.slot_mu_r.data();
        field_params.slot_sigma = options.slot_sigma.data();
        field_params.slot_gain = options.slot_gain.data();
        field_params.tx_pol_x = options.tx_polarization.x().data();
        field_params.tx_pol_y = options.tx_polarization.y().data();
        field_params.tx_pol_z = options.tx_polarization.z().data();
        field_params.tx_pol_count = tx_pol_count;
        field_params.omega = options.omega;
        field_params.wavelength = options.wavelength;
        field_params.out_valid = reinterpret_cast<uint8_t *>(result.valid.data());
        field_params.out_bounce_count = result.bounce_count.data();
        field_params.out_path_length = result.path_length.data();
        field_params.out_field_x_re = result.field_x_re.data();
        field_params.out_field_x_im = result.field_x_im.data();
        field_params.out_field_y_re = result.field_y_re.data();
        field_params.out_field_y_im = result.field_y_im.data();
        field_params.out_field_z_re = result.field_z_re.data();
        field_params.out_field_z_im = result.field_z_im.data();

        if (options.return_endpoints) {
            field_params.out_tx_x = result.tx_pos.x().data();
            field_params.out_tx_y = result.tx_pos.y().data();
            field_params.out_tx_z = result.tx_pos.z().data();
            field_params.out_first_hit_x = result.first_hit.x().data();
            field_params.out_first_hit_y = result.first_hit.y().data();
            field_params.out_first_hit_z = result.first_hit.z().data();
            field_params.out_last_hit_x = result.last_hit.x().data();
            field_params.out_last_hit_y = result.last_hit.y().data();
            field_params.out_last_hit_z = result.last_hit.z().data();
        }
        if (options.return_geom && options.return_hit_points) {
            field_params.out_hit_x = result.hit_points.x().data();
            field_params.out_hit_y = result.hit_points.y().data();
            field_params.out_hit_z = result.hit_points.z().data();
        }
        if (options.return_geom && options.return_normals) {
            field_params.out_normal_x = result.normals.x().data();
            field_params.out_normal_y = result.normals.y().data();
            field_params.out_normal_z = result.normals.z().data();
        }
        if (return_resolved_prim_ids) {
            field_params.out_resolved_prim_ids = result.resolved_prim_ids.data();
        }
        if (return_surface_group_ids) {
            field_params.out_surface_group_ids = result.surface_group_ids.data();
        }

        reflection_epc_field_gpu(field_params);
        return result;
    }
}

template <bool Detached>
AccumResultT<Detached> Scene::accumulate_reflections(
    const RayT<Detached> &ray,
    const Vector3fT<Detached> &tx_position,
    const AccumGrid &grid,
    const MaterialT<Detached> &material,
    int max_bounces,
    const AccumOptions &options,
    MaskT<Detached> active,
    const Vector3fT<Detached> &tx_polarization) const {
    require(is_ready(), "Scene::accumulate_reflections(): scene is not built.");
    require(!pending_updates_,
            "Scene::accumulate_reflections(): scene has pending updates. Call Scene::sync() first.");
    require(max_bounces > 0,
            "Scene::accumulate_reflections(): max_bounces must be positive.");
    require(grid.axis >= 0 && grid.axis <= 2,
            "Scene::accumulate_reflections(): grid.axis must be 0, 1, or 2.");
    require(grid.resolution0 > 0 && grid.resolution1 > 0,
            "Scene::accumulate_reflections(): grid resolution must be positive.");
    require(grid.coord0_min < grid.coord0_max && grid.coord1_min < grid.coord1_max,
            "Scene::accumulate_reflections(): grid bounds must be ordered.");
    require(options.wavelength > 0.f,
            "Scene::accumulate_reflections(): wavelength must be positive.");
    require(options.cell_area > 0.f,
            "Scene::accumulate_reflections(): cell_area must be positive.");
    require(options.solid_angle_per_ray >= 0.f,
            "Scene::accumulate_reflections(): solid_angle_per_ray must be non-negative.");
    require(options.wedge_capacity >= 0,
            "Scene::accumulate_reflections(): wedge_capacity must be non-negative.");
    require(options.wedge_sample_stride >= 1,
            "Scene::accumulate_reflections(): wedge_sample_stride must be >= 1.");

    const int ray_count = static_cast<int>(slices(ray.o));
    const int grid_cell_count = grid.resolution0 * grid.resolution1;
    AccumResultT<Detached> result;
    result.ray_count = ray_count;
    result.max_bounces = max_bounces;
    result.grid_cell_count = grid_cell_count;

    if constexpr (!Detached) {
        const ReflectionChainAD chain =
            this->template trace_reflections<false>(ray, max_bounces, active);

        result.reflection_power = zeros<FloatAD>(grid_cell_count);
        result.reflection_field_x =
            drjit::Complex<FloatAD>(zeros<FloatAD>(grid_cell_count),
                                    zeros<FloatAD>(grid_cell_count));
        result.reflection_field_y =
            drjit::Complex<FloatAD>(zeros<FloatAD>(grid_cell_count),
                                    zeros<FloatAD>(grid_cell_count));
        result.reflection_field_z =
            drjit::Complex<FloatAD>(zeros<FloatAD>(grid_cell_count),
                                    zeros<FloatAD>(grid_cell_count));
        result.reflection_count = full<IntAD>(0, 1);
        result.wedge_events.capacity = options.wedge_capacity;
        result.wedge_events.count = full<IntAD>(0, 1);
        const int event_count = std::max(1, options.wedge_capacity);
        result.wedge_events.ray_index = full<IntAD>(-1, event_count);
        result.wedge_events.hit_points = zeros<Vector3fAD>(event_count);
        result.wedge_events.normals = zeros<Vector3fAD>(event_count);
        result.wedge_events.prim_id = full<IntAD>(-1, event_count);
        result.wedge_events.directions = zeros<Vector3fAD>(event_count);
        result.wedge_events.source_points = zeros<Vector3fAD>(event_count);
        result.wedge_events.src_power = zeros<FloatAD>(event_count);
        result.wedge_events.initial_directions = zeros<Vector3fAD>(event_count);
        result.wedge_events.bounce_depth = full<IntAD>(-1, event_count);

        if (ray_count <= 0 || grid_cell_count <= 0) {
            return result;
        }
        const int material_count = static_cast<int>(slices(material.gain));
        require(material_count > 0,
                "Scene::accumulate_reflections(): material payload must not be empty.");
        require(static_cast<int>(slices(material.eta_r)) == material_count &&
                    static_cast<int>(slices(material.sigma)) == material_count &&
                    static_cast<int>(slices(material.mu_r)) == material_count &&
                    static_cast<int>(slices(material.valid)) == material_count,
                "Scene::accumulate_reflections(): material payload fields must have matching widths.");

        auto component = [](const Vector3fAD &value, int axis) -> FloatAD {
            if (axis == 0) {
                return value.x();
            }
            if (axis == 1) {
                return value.y();
            }
            return value.z();
        };
        auto plane_point = [](int axis,
                              const FloatAD &position,
                              const FloatAD &coord0,
                              const FloatAD &coord1) -> Vector3fAD {
            if (axis == 0) {
                return Vector3fAD(position, coord0, coord1);
            }
            if (axis == 1) {
                return Vector3fAD(coord0, position, coord1);
            }
            return Vector3fAD(coord0, coord1, position);
        };
        auto coords_from_point = [](const Vector3fAD &point,
                                    int axis,
                                    FloatAD &coord0,
                                    FloatAD &coord1) {
            if (axis == 0) {
                coord0 = point.y();
                coord1 = point.z();
            } else if (axis == 1) {
                coord0 = point.x();
                coord1 = point.z();
            } else {
                coord0 = point.x();
                coord1 = point.y();
            }
        };
        auto broadcast_vec = [](const Vector3fAD &value, int width) -> Vector3fAD {
            const int value_width = static_cast<int>(slices(value));
            if (value_width == width) {
                return value;
            }
            const UIntAD zero_index = zeros<UIntAD>(width);
            const MaskAD active = full<MaskAD>(true, width);
            return gather<Vector3fAD>(value, zero_index, active);
        };
        struct ComplexADValue {
            FloatAD re;
            FloatAD im;
        };
        struct ComplexVectorAD {
            Vector3fAD re;
            Vector3fAD im;
        };
        auto complex_add = [](const ComplexADValue &a,
                              const ComplexADValue &b) -> ComplexADValue {
            return {a.re + b.re, a.im + b.im};
        };
        auto complex_sub = [](const ComplexADValue &a,
                              const ComplexADValue &b) -> ComplexADValue {
            return {a.re - b.re, a.im - b.im};
        };
        auto complex_mul = [](const ComplexADValue &a,
                              const ComplexADValue &b) -> ComplexADValue {
            return {a.re * b.re - a.im * b.im,
                    a.re * b.im + a.im * b.re};
        };
        auto complex_scale = [](const ComplexADValue &a,
                                const FloatAD &scale) -> ComplexADValue {
            return {a.re * scale, a.im * scale};
        };
        auto complex_div = [](const ComplexADValue &a,
                              const ComplexADValue &b) -> ComplexADValue {
            const FloatAD denom =
                maximum(b.re * b.re + b.im * b.im, FloatAD(Epsilon));
            return {(a.re * b.re + a.im * b.im) / denom,
                    (a.im * b.re - a.re * b.im) / denom};
        };
        auto complex_sqrt = [](const ComplexADValue &a) -> ComplexADValue {
            const FloatAD mag =
                sqrt(maximum(a.re * a.re + a.im * a.im, FloatAD(0.f)));
            const MaskAD positive_real_axis =
                (abs(a.im) <= FloatAD(Epsilon)) && (a.re > FloatAD(Epsilon));
            const FloatAD real_part =
                sqrt(maximum(FloatAD(0.5f) * (mag + a.re), FloatAD(0.f)));
            const FloatAD imag_abs =
                sqrt(maximum(FloatAD(0.5f) * (mag - a.re), FloatAD(1e-20f)));
            const FloatAD imag_sign =
                select(a.im < FloatAD(0.f), FloatAD(-1.f), FloatAD(1.f));
            return {
                select(positive_real_axis, sqrt(a.re), real_part),
                select(positive_real_axis, FloatAD(0.f), imag_sign * imag_abs),
            };
        };
        auto normalize_safe = [](const Vector3fAD &value,
                                 const Vector3fAD &fallback) -> Vector3fAD {
            const FloatAD value_norm = norm(value);
            return select(value_norm > FloatAD(Epsilon),
                          value / maximum(value_norm, FloatAD(Epsilon)),
                          fallback);
        };
        auto stable_perpendicular = [&](const Vector3fAD &direction,
                                        const Vector3fAD &preferred) -> Vector3fAD {
            const Vector3fAD dir =
                normalize_safe(direction, Vector3fAD(FloatAD(1.f), FloatAD(0.f), FloatAD(0.f)));
            const Vector3fAD projected = preferred - dot(preferred, dir) * dir;
            const Vector3fAD axis =
                select(abs(dir.x()) < FloatAD(0.9f),
                       Vector3fAD(FloatAD(1.f), FloatAD(0.f), FloatAD(0.f)),
                       Vector3fAD(FloatAD(0.f), FloatAD(1.f), FloatAD(0.f)));
            const Vector3fAD fallback = axis - dot(axis, dir) * dir;
            return select(squared_norm(projected) > FloatAD(1e-12f),
                          normalize_safe(projected, axis),
                          normalize_safe(fallback,
                                         Vector3fAD(FloatAD(0.f), FloatAD(0.f), FloatAD(1.f))));
        };
        auto complex_dot_real = [](const ComplexVectorAD &field,
                                   const Vector3fAD &basis) -> ComplexADValue {
            return {dot(field.re, basis), dot(field.im, basis)};
        };
        auto complex_vector_power = [](const ComplexVectorAD &field) -> FloatAD {
            return squared_norm(field.re) + squared_norm(field.im);
        };
        auto material_reflection_coefficients =
            [&](const IntAD &prim,
                const FloatAD &cos_theta,
                const MaskAD &slot_active) -> std::pair<ComplexADValue, ComplexADValue> {
            const MaskAD prim_in_range =
                slot_active && (prim >= IntAD(0)) && (prim < IntAD(material_count));
            const IntAD safe_prim = select(prim_in_range, prim, IntAD(0));
            const MaskAD prim_valid =
                prim_in_range && gather<MaskAD>(material.valid, safe_prim, prim_in_range);
            const FloatAD eta_r =
                maximum(gather<FloatAD>(material.eta_r, safe_prim, prim_valid),
                        FloatAD(Epsilon));
            const FloatAD sigma =
                maximum(gather<FloatAD>(material.sigma, safe_prim, prim_valid),
                        FloatAD(0.f));
            const FloatAD gain = gather<FloatAD>(material.gain, safe_prim, prim_valid);
            const FloatAD mu_r =
                maximum(gather<FloatAD>(material.mu_r, safe_prim, prim_valid),
                        FloatAD(Epsilon));
            const FloatAD omega = maximum(
                FloatAD(2.f * Pi) * FloatAD(299792458.f) /
                    maximum(FloatAD(options.wavelength), FloatAD(Epsilon)),
                FloatAD(Epsilon));
            const ComplexADValue eta = {
                eta_r,
                -sigma / (omega * FloatAD(8.854187817e-12f))
            };
            const ComplexADValue mu = {mu_r, FloatAD(0.f)};
            const FloatAD cos_clamped =
                minimum(maximum(abs(cos_theta), FloatAD(Epsilon)), FloatAD(1.f));
            const FloatAD sin2 =
                maximum(FloatAD(0.f), FloatAD(1.f) - cos_clamped * cos_clamped);
            const ComplexADValue a =
                complex_sqrt(complex_sub(complex_mul(mu, eta),
                                         ComplexADValue{sin2, FloatAD(0.f)}));
            const ComplexADValue mu_cos = {mu_r * cos_clamped, FloatAD(0.f)};
            const ComplexADValue eta_cos = {eta.re * cos_clamped,
                                            eta.im * cos_clamped};
            const ComplexADValue zero = {FloatAD(0.f), FloatAD(0.f)};
            const ComplexADValue r_te_raw =
                complex_scale(
                    complex_div(complex_sub(mu_cos, a),
                                complex_add(mu_cos, a)),
                    gain);
            const ComplexADValue r_tm_raw =
                complex_scale(
                    complex_div(complex_sub(eta_cos, a),
                                complex_add(eta_cos, a)),
                    gain);
            const ComplexADValue r_te = {
                select(prim_valid, r_te_raw.re, zero.re),
                select(prim_valid, r_te_raw.im, zero.im),
            };
            const ComplexADValue r_tm = {
                select(prim_valid, r_tm_raw.re, zero.re),
                select(prim_valid, r_tm_raw.im, zero.im),
            };
            return {r_te, r_tm};
        };
        auto reflect_field_vector =
            [&](const ComplexVectorAD &field,
                const Vector3fAD &incident_dir,
                const Vector3fAD &slot_normal,
                const IntAD &prim,
                const MaskAD &slot_active) -> std::pair<ComplexVectorAD, Vector3fAD> {
            const Vector3fAD incident_hat =
                normalize_safe(incident_dir,
                               Vector3fAD(FloatAD(1.f), FloatAD(0.f), FloatAD(0.f)));
            Vector3fAD normal_hat =
                normalize_safe(slot_normal,
                               Vector3fAD(FloatAD(0.f), FloatAD(0.f), FloatAD(1.f)));
            normal_hat = select(dot(incident_hat, normal_hat) > FloatAD(0.f),
                                -normal_hat,
                                normal_hat);
            const FloatAD dot_dn = dot(incident_hat, normal_hat);
            const Vector3fAD reflected_dir =
                normalize_safe(incident_hat - FloatAD(2.f) * dot_dn * normal_hat,
                               -incident_hat);
            Vector3fAD s_hat = cross(normal_hat, incident_hat);
            s_hat = select(squared_norm(s_hat) > FloatAD(1e-12f),
                           normalize_safe(s_hat, stable_perpendicular(incident_hat, normal_hat)),
                           stable_perpendicular(incident_hat, normal_hat));
            Vector3fAD p_in_hat = cross(s_hat, incident_hat);
            p_in_hat =
                select(squared_norm(p_in_hat) > FloatAD(1e-12f),
                       normalize_safe(p_in_hat, stable_perpendicular(incident_hat, normal_hat)),
                       stable_perpendicular(incident_hat, normal_hat));
            Vector3fAD p_out_hat = cross(s_hat, reflected_dir);
            p_out_hat =
                select(squared_norm(p_out_hat) > FloatAD(1e-12f),
                       normalize_safe(p_out_hat, stable_perpendicular(reflected_dir, normal_hat)),
                       stable_perpendicular(reflected_dir, normal_hat));
            const auto [r_te, r_tm] =
                material_reflection_coefficients(prim, abs(dot(incident_hat, normal_hat)), slot_active);
            const ComplexADValue e_s = complex_dot_real(field, s_hat);
            const ComplexADValue e_p = complex_dot_real(field, p_in_hat);
            const ComplexADValue out_s = complex_mul(r_te, e_s);
            const ComplexADValue out_p = complex_mul(r_tm, e_p);
            return {{
                s_hat * out_s.re + p_out_hat * out_p.re,
                s_hat * out_s.im + p_out_hat * out_p.im,
            }, reflected_dir};
        };

        const UIntAD ray_index = arange<UIntAD>(ray_count);
        const IntAD ray_slot = IntAD(ray_index);
        const IntAD base_slot = ray_slot * IntAD(max_bounces);
        const MaskAD active_ad = active;
        Vector3fAD origin = ray.o;
        Vector3fAD direction = normalize(ray.d);
        Vector3fAD image_source = broadcast_vec(tx_position, ray_count);
        Vector3fAD tx_pol = broadcast_vec(tx_polarization, ray_count);
        Vector3fAD transverse_pol = tx_pol - dot(tx_pol, direction) * direction;
        transverse_pol = select(
            squared_norm(transverse_pol) > FloatAD(1e-12f),
            normalize_safe(transverse_pol,
                           stable_perpendicular(direction, tx_pol)),
            stable_perpendicular(direction, tx_pol));
        ComplexVectorAD field = {transverse_pol, zeros<Vector3fAD>(ray_count)};
        FloatAD path_length = zeros<FloatAD>(ray_count);
        MaskAD current_active = active_ad;

        const FloatAD span0 = FloatAD(grid.coord0_max - grid.coord0_min);
        const FloatAD span1 = FloatAD(grid.coord1_max - grid.coord1_min);
        const FloatAD wavelength = FloatAD(options.wavelength);
        const FloatAD wave_gain = wavelength / FloatAD(4.f * Pi);
        const FloatAD solid_angle = FloatAD(options.solid_angle_per_ray);
        const FloatAD cell_area = FloatAD(options.cell_area);
        const FloatAD wave_k =
            select(abs(FloatAD(options.k)) > FloatAD(Epsilon),
                   FloatAD(options.k),
                   FloatAD(2.f * Pi) / maximum(wavelength, FloatAD(Epsilon)));

        for (int bounce = 0; bounce < max_bounces; ++bounce) {
            const IntAD slot = base_slot + IntAD(bounce);
            const MaskAD bounce_active =
                current_active && (chain.bounce_count > IntAD(bounce));
            if (drjit::none(detach<false>(bounce_active))) {
                break;
            }

            const Vector3fAD hit_point =
                gather<Vector3fAD>(chain.hit_points, slot, bounce_active);
            Vector3fAD normal =
                gather<Vector3fAD>(chain.geo_normals, slot, bounce_active);
            normal = normalize(normal);
            normal = select(dot(direction, normal) > FloatAD(0.f), -normal, normal);
            const IntAD prim =
                gather<IntAD>(chain.global_prim_ids, slot, bounce_active);
            const MaskAD material_active = bounce_active;

            const Vector3fAD event_source = image_source;
            const Vector3fAD event_direction = direction;
            const FloatAD event_source_power = complex_vector_power(field) * solid_angle;
            const FloatAD image_distance = dot(image_source - hit_point, normal);
            image_source = select(
                material_active,
                image_source - FloatAD(2.f) * image_distance * normal,
                image_source);
            const auto reflected = reflect_field_vector(field, direction, normal, prim, material_active);
            direction = select(material_active, reflected.second, direction);
            origin = select(
                material_active,
                hit_point + FloatAD(Epsilon) * direction,
                origin);
            field = {
                select(material_active, reflected.first.re, zeros<Vector3fAD>(ray_count)),
                select(material_active, reflected.first.im, zeros<Vector3fAD>(ray_count)),
            };
            path_length = select(
                material_active,
                path_length + gather<FloatAD>(chain.t, slot, bounce_active),
                path_length);

            if (options.collect_wedges && options.wedge_capacity > 0) {
                const IntAD event_slot =
                    (ray_slot * IntAD(max_bounces) + IntAD(bounce)) /
                    IntAD(options.wedge_sample_stride);
                const MaskAD wedge_active =
                    material_active && (event_slot >= IntAD(0)) &&
                    (event_slot < IntAD(options.wedge_capacity));
                scatter(
                    result.wedge_events.ray_index,
                    ray_slot,
                    event_slot,
                    wedge_active);
                scatter(
                    result.wedge_events.hit_points,
                    hit_point,
                    event_slot,
                    wedge_active);
                scatter(
                    result.wedge_events.normals,
                    normal,
                    event_slot,
                    wedge_active);
                scatter(
                    result.wedge_events.prim_id,
                    prim,
                    event_slot,
                    wedge_active);
                scatter(
                    result.wedge_events.directions,
                    event_direction,
                    event_slot,
                    wedge_active);
                scatter(
                    result.wedge_events.source_points,
                    event_source,
                    event_slot,
                    wedge_active);
                scatter(
                    result.wedge_events.src_power,
                    event_source_power,
                    event_slot,
                    wedge_active);
                scatter(
                    result.wedge_events.initial_directions,
                    normalize(ray.d),
                    event_slot,
                    wedge_active);
                scatter(
                    result.wedge_events.bounce_depth,
                    IntAD(bounce),
                    event_slot,
                    wedge_active);
                scatter_reduce(
                    ReduceOp::Add,
                    result.wedge_events.count,
                    IntAD(1),
                    zeros<IntAD>(ray_count),
                    wedge_active);
            }

            FloatAD blocker_t = chain.trailing_t;
            if (bounce + 1 < max_bounces) {
                const IntAD next_slot = slot + IntAD(1);
                const MaskAD next_valid = chain.bounce_count > IntAD(bounce + 1);
                blocker_t = select(
                    next_valid,
                    gather<FloatAD>(chain.t, next_slot, next_valid),
                    blocker_t);
            }

            const FloatAD axis_dir = component(direction, grid.axis);
            const FloatAD safe_axis_dir =
                axis_dir + select(axis_dir >= FloatAD(0.f), FloatAD(Epsilon), FloatAD(-Epsilon));
            const FloatAD t_plane =
                (FloatAD(grid.position) - component(origin, grid.axis)) / safe_axis_dir;
            const Vector3fAD target = origin + direction * t_plane;
            FloatAD coord0 = zeros<FloatAD>(ray_count);
            FloatAD coord1 = zeros<FloatAD>(ray_count);
            coords_from_point(target, grid.axis, coord0, coord1);

            const MaskAD plane_active =
                material_active &&
                (complex_vector_power(field) > FloatAD(0.f)) &&
                (t_plane > FloatAD(RayEpsilon)) &&
                (t_plane < blocker_t) &&
                (coord0 >= FloatAD(grid.coord0_min)) &&
                (coord0 < FloatAD(grid.coord0_max)) &&
                (coord1 >= FloatAD(grid.coord1_min)) &&
                (coord1 < FloatAD(grid.coord1_max));
            const FloatAD u = (coord0 - FloatAD(grid.coord0_min)) / span0;
            const FloatAD v = (coord1 - FloatAD(grid.coord1_min)) / span1;
            IntAD ix = IntAD(u * FloatAD(grid.resolution0));
            IntAD iy = IntAD(v * FloatAD(grid.resolution1));
            ix = minimum(maximum(ix, IntAD(0)), IntAD(grid.resolution0 - 1));
            iy = minimum(maximum(iy, IntAD(0)), IntAD(grid.resolution1 - 1));
            const IntAD cell = iy * IntAD(grid.resolution0) + ix;

            const Vector3fAD target_plane =
                plane_point(grid.axis, FloatAD(grid.position), coord0, coord1);
            const FloatAD unfolded_distance =
                maximum(norm(target_plane - image_source), FloatAD(Epsilon));
            const FloatAD cos_theta = maximum(abs(axis_dir), FloatAD(Epsilon));
            const FloatAD geometry_power_scale =
                solid_angle / cell_area *
                unfolded_distance * unfolded_distance / cos_theta;
            const FloatAD amplitude_scale =
                wave_gain / unfolded_distance *
                sqrt(maximum(geometry_power_scale, FloatAD(0.f)));
            const ComplexADValue phase = {
                cos(wave_k * unfolded_distance),
                -sin(wave_k * unfolded_distance)
            };
            const ComplexADValue coeff = complex_scale(phase, amplitude_scale);
            const ComplexVectorAD contribution_field = {
                field.re * coeff.re - field.im * coeff.im,
                field.re * coeff.im + field.im * coeff.re,
            };
            const FloatAD contribution_power =
                complex_vector_power(contribution_field);
            const MaskAD contribution_active =
                plane_active && drjit::isfinite(contribution_power) && (contribution_power > FloatAD(0.f));
            scatter_reduce(
                ReduceOp::Add,
                result.reflection_field_x.x(),
                contribution_field.re.x(),
                cell,
                contribution_active);
            scatter_reduce(
                ReduceOp::Add,
                result.reflection_field_x.y(),
                contribution_field.im.x(),
                cell,
                contribution_active);
            scatter_reduce(
                ReduceOp::Add,
                result.reflection_field_y.x(),
                contribution_field.re.y(),
                cell,
                contribution_active);
            scatter_reduce(
                ReduceOp::Add,
                result.reflection_field_y.y(),
                contribution_field.im.y(),
                cell,
                contribution_active);
            scatter_reduce(
                ReduceOp::Add,
                result.reflection_field_z.x(),
                contribution_field.re.z(),
                cell,
                contribution_active);
            scatter_reduce(
                ReduceOp::Add,
                result.reflection_field_z.y(),
                contribution_field.im.z(),
                cell,
                contribution_active);
            scatter_reduce(
                ReduceOp::Add,
                result.reflection_power,
                contribution_power,
                cell,
                contribution_active);
            scatter_reduce(
                ReduceOp::Add,
                result.reflection_count,
                IntAD(1),
                zeros<IntAD>(ray_count),
                contribution_active);

            const int next_depth = bounce + 1;
            MaskAD continue_active = material_active;
            if (options.rr_depth > 0 && options.rr_prob < 1.f && next_depth >= options.rr_depth) {
                const FloatAD rr_field_power = complex_vector_power(field);
                const FloatAD continue_prob =
                    minimum(maximum(rr_field_power, FloatAD(1e-8f)),
                            maximum(FloatAD(options.rr_prob), FloatAD(1e-8f)));
                const FloatAD rr_scale =
                    FloatAD(1.f) / sqrt(maximum(continue_prob, FloatAD(1e-8f)));
                field = {
                    select(continue_active, field.re * rr_scale, field.re),
                    select(continue_active, field.im * rr_scale, field.im),
                };
            }
            if (options.stop_threshold > 0.f) {
                const FloatAD fspl =
                    wavelength / (FloatAD(4.f * Pi) * maximum(path_length, FloatAD(Epsilon)));
                continue_active =
                    continue_active &&
                    (complex_vector_power(field) * fspl * fspl > FloatAD(options.stop_threshold));
            }
            current_active = continue_active;
        }
        return result;
    } else {
        ScopedNativeLaunchStage native_launch_stage(
            NativeLaunchStage::AccumulateReflections);
        auto initialize_result_storage = [&]() {
            result.reflection_power = zeros<Float>(grid_cell_count);
            result.reflection_field_x =
                drjit::Complex<Float>(zeros<Float>(grid_cell_count),
                                      zeros<Float>(grid_cell_count));
            result.reflection_field_y =
                drjit::Complex<Float>(zeros<Float>(grid_cell_count),
                                      zeros<Float>(grid_cell_count));
            result.reflection_field_z =
                drjit::Complex<Float>(zeros<Float>(grid_cell_count),
                                      zeros<Float>(grid_cell_count));
            result.reflection_count = full<Int>(0, 1);
            result.wedge_events.capacity = options.wedge_capacity;
            result.wedge_events.count = full<Int>(0, 1);
            const int event_count = std::max(1, options.wedge_capacity);
            result.wedge_events.ray_index = full<Int>(-1, event_count);
            result.wedge_events.hit_points = zeros<Vector3f>(event_count);
            result.wedge_events.normals = zeros<Vector3f>(event_count);
            result.wedge_events.prim_id = full<Int>(-1, event_count);
            result.wedge_events.directions = zeros<Vector3f>(event_count);
            result.wedge_events.source_points = zeros<Vector3f>(event_count);
            result.wedge_events.src_power = zeros<Float>(event_count);
            result.wedge_events.initial_directions = zeros<Vector3f>(event_count);
            result.wedge_events.bounce_depth = full<Int>(-1, event_count);
        };
        if (ray_count == 0) {
            initialize_result_storage();
            return result;
        }

        require(static_cast<int>(slices(ray.d)) == ray_count &&
                    static_cast<int>(slices(ray.tmax)) == ray_count,
                "Scene::accumulate_reflections(): ray fields must have matching widths.");
        const int tx_count = static_cast<int>(slices(tx_position));
        require(tx_count == 1 || tx_count == ray_count,
                "Scene::accumulate_reflections(): tx_position width must be 1 or match ray count.");
        const int tx_pol_count = static_cast<int>(slices(tx_polarization));
        require(tx_pol_count == 1 || tx_pol_count == ray_count,
                "Scene::accumulate_reflections(): tx_polarization width must be 1 or match ray count.");

        const int material_count = static_cast<int>(slices(material.eta_r));
        require(material_count > 0,
                "Scene::accumulate_reflections(): material payload must not be empty.");
        require(static_cast<int>(slices(material.sigma)) == material_count &&
                    static_cast<int>(slices(material.gain)) == material_count &&
                    static_cast<int>(slices(material.mu_r)) == material_count &&
                    static_cast<int>(slices(material.valid)) == material_count,
                "Scene::accumulate_reflections(): material payload fields must have matching widths.");

        const int triangle_count = static_cast<int>(slices(triangle_info_detached_.p0));
        require(material_count >= triangle_count,
                "Scene::accumulate_reflections(): material payload must provide one entry per global primitive.");

        const OptixSceneSelection scenes = select_optix_scenes();
        const OptixScene *primary_scene = scenes.primary;
        const OptixScene *secondary_scene = scenes.secondary;
        int split_mode = scenes.split_mode;
        int hitgroup_record_count = scenes.hitgroup_record_count;

        require(primary_scene != nullptr && primary_scene->is_ready(),
                "Scene::accumulate_reflections(): OptiX scene is not ready.");
        require(hitgroup_record_count > 0,
                "Scene::accumulate_reflections(): invalid hitgroup record count.");

        ensure_pipeline(reflection_accumulation_pipeline_, primary_scene->context(),
                        hitgroup_record_count, reflection_accumulation_pipeline_config());

        initialize_result_storage();

        Vector3f tx_detached = tx_position;
        if (tx_count == 1 && ray_count > 1) {
            const Int zero_index = full<Int>(0, ray_count);
            tx_detached = Vector3f(
                gather<Float>(tx_position.x(), zero_index),
                gather<Float>(tx_position.y(), zero_index),
                gather<Float>(tx_position.z(), zero_index));
        }
        Vector3f tx_pol_detached = tx_polarization;
        if (tx_pol_count == 1 && ray_count > 1) {
            const Int zero_index = full<Int>(0, ray_count);
            tx_pol_detached = Vector3f(
                gather<Float>(tx_polarization.x(), zero_index),
                gather<Float>(tx_polarization.y(), zero_index),
                gather<Float>(tx_polarization.z(), zero_index));
        }

        Mask active_detached = sanitize_reflection_active<true>(ray, active);
        active_detached &= drjit::isfinite(tx_detached.x()) &&
                           drjit::isfinite(tx_detached.y()) &&
                           drjit::isfinite(tx_detached.z()) &&
                           drjit::isfinite(tx_pol_detached.x()) &&
                           drjit::isfinite(tx_pol_detached.y()) &&
                           drjit::isfinite(tx_pol_detached.z());
        if (drjit::none(active_detached)) {
            return result;
        }

        drjit::eval(ray.o,
                    ray.d,
                    ray.tmax,
                    tx_detached,
                    tx_pol_detached,
                    active_detached,
                    triangle_info_detached_.p0,
                    triangle_info_detached_.e1,
                    triangle_info_detached_.e2,
                    triangle_info_detached_.face_normal,
                    face_offsets_,
                    material.eta_r,
                    material.sigma,
                    material.gain,
                    material.mu_r,
                    material.valid);

        AccumRaw raw = allocate_reflection_accumulation_raw(
            ray_count, max_bounces, grid_cell_count, options.wedge_capacity);
        initialize_reflection_accumulation_raw(raw);

        AccumParams params = {};
        params.primary_handle = primary_scene->ias_handle();
        params.secondary_handle =
            secondary_scene != nullptr && secondary_scene->is_ready() ? secondary_scene->ias_handle() : 0ull;
        params.split_mode = split_mode;
        params.tri_p0_x = triangle_info_detached_.p0.x().data();
        params.tri_p0_y = triangle_info_detached_.p0.y().data();
        params.tri_p0_z = triangle_info_detached_.p0.z().data();
        params.tri_e1_x = triangle_info_detached_.e1.x().data();
        params.tri_e1_y = triangle_info_detached_.e1.y().data();
        params.tri_e1_z = triangle_info_detached_.e1.z().data();
        params.tri_e2_x = triangle_info_detached_.e2.x().data();
        params.tri_e2_y = triangle_info_detached_.e2.y().data();
        params.tri_e2_z = triangle_info_detached_.e2.z().data();
        params.tri_fn_x = triangle_info_detached_.face_normal.x().data();
        params.tri_fn_y = triangle_info_detached_.face_normal.y().data();
        params.tri_fn_z = triangle_info_detached_.face_normal.z().data();
        params.face_offsets = face_offsets_.data();
        params.n_meshes = mesh_count_;
        params.n_triangles = triangle_count;
        params.ray_ox = ray.o.x().data();
        params.ray_oy = ray.o.y().data();
        params.ray_oz = ray.o.z().data();
        params.ray_dx = ray.d.x().data();
        params.ray_dy = ray.d.y().data();
        params.ray_dz = ray.d.z().data();
        params.ray_tmax = ray.tmax.data();
        params.active_mask = reinterpret_cast<const uint8_t *>(active_detached.data());
        params.n_rays = ray_count;
        params.tx_x = tx_detached.x().data();
        params.tx_y = tx_detached.y().data();
        params.tx_z = tx_detached.z().data();
        params.tx_pol_x = tx_pol_detached.x().data();
        params.tx_pol_y = tx_pol_detached.y().data();
        params.tx_pol_z = tx_pol_detached.z().data();
        params.max_bounces = max_bounces;
        params.wavelength = options.wavelength;
        params.k = options.k;
        params.solid_angle_per_ray = options.solid_angle_per_ray;
        params.cell_area = options.cell_area;
        params.seed = options.seed;
        params.rr_depth = options.rr_depth;
        params.rr_prob = options.rr_prob;
        params.stop_threshold = options.stop_threshold;
        params.grid_axis = grid.axis;
        params.grid_position = grid.position;
        params.grid_coord0_min = grid.coord0_min;
        params.grid_coord0_max = grid.coord0_max;
        params.grid_coord1_min = grid.coord1_min;
        params.grid_coord1_max = grid.coord1_max;
        params.grid_resolution0 = grid.resolution0;
        params.grid_resolution1 = grid.resolution1;
        params.material_eta_r = material.eta_r.data();
        params.material_sigma = material.sigma.data();
        params.material_gain = material.gain.data();
        params.material_mu_r = material.mu_r.data();
        params.material_valid = reinterpret_cast<const uint8_t *>(material.valid.data());
        params.material_count = material_count;
        params.collect_wedges = options.collect_wedges ? 1 : 0;
        params.collect_wedge_prefixes = options.collect_wedge_prefixes ? 1 : 0;
        params.wedge_capacity = options.wedge_capacity;
        params.wedge_sample_stride = options.wedge_sample_stride;
        params.out_reflection_power = raw.reflection_power.data();
        params.out_field_x_re = raw.field_x_re.data();
        params.out_field_x_im = raw.field_x_im.data();
        params.out_field_y_re = raw.field_y_re.data();
        params.out_field_y_im = raw.field_y_im.data();
        params.out_field_z_re = raw.field_z_re.data();
        params.out_field_z_im = raw.field_z_im.data();
        params.out_reflection_count = raw.reflection_count.data();
        params.out_wedge_count = raw.wedge_count.data();
        params.out_wedge_ray_index = raw.wedge_ray_index.data();
        params.out_wedge_hit_x = raw.wedge_hit_x.data();
        params.out_wedge_hit_y = raw.wedge_hit_y.data();
        params.out_wedge_hit_z = raw.wedge_hit_z.data();
        params.out_wedge_normal_x = raw.wedge_normal_x.data();
        params.out_wedge_normal_y = raw.wedge_normal_y.data();
        params.out_wedge_normal_z = raw.wedge_normal_z.data();
        params.out_wedge_prim_id = raw.wedge_prim_id.data();
        params.out_wedge_dir_x = raw.wedge_dir_x.data();
        params.out_wedge_dir_y = raw.wedge_dir_y.data();
        params.out_wedge_dir_z = raw.wedge_dir_z.data();
        params.out_wedge_source_x = raw.wedge_source_x.data();
        params.out_wedge_source_y = raw.wedge_source_y.data();
        params.out_wedge_source_z = raw.wedge_source_z.data();
        params.out_wedge_source_power = raw.wedge_source_power.data();
        params.out_wedge_initial_dir_x = raw.wedge_initial_dir_x.data();
        params.out_wedge_initial_dir_y = raw.wedge_initial_dir_y.data();
        params.out_wedge_initial_dir_z = raw.wedge_initial_dir_z.data();
        params.out_wedge_bounce_depth = raw.wedge_bounce_depth.data();

        reflection_accumulation_pipeline_->launch(0, params);

        result.reflection_power = raw.reflection_power;
        result.reflection_field_x =
            drjit::Complex<Float>(raw.field_x_re, raw.field_x_im);
        result.reflection_field_y =
            drjit::Complex<Float>(raw.field_y_re, raw.field_y_im);
        result.reflection_field_z =
            drjit::Complex<Float>(raw.field_z_re, raw.field_z_im);
        result.reflection_count = raw.reflection_count;
        result.wedge_events.capacity = options.wedge_capacity;
        result.wedge_events.count = raw.wedge_count;
        result.wedge_events.ray_index = raw.wedge_ray_index;
        result.wedge_events.hit_points =
            Vector3f(raw.wedge_hit_x, raw.wedge_hit_y, raw.wedge_hit_z);
        result.wedge_events.normals =
            Vector3f(raw.wedge_normal_x, raw.wedge_normal_y, raw.wedge_normal_z);
        result.wedge_events.prim_id = raw.wedge_prim_id;
        result.wedge_events.directions =
            Vector3f(raw.wedge_dir_x, raw.wedge_dir_y, raw.wedge_dir_z);
        result.wedge_events.source_points =
            Vector3f(raw.wedge_source_x, raw.wedge_source_y, raw.wedge_source_z);
        result.wedge_events.src_power = raw.wedge_source_power;
        result.wedge_events.initial_directions = Vector3f(
            raw.wedge_initial_dir_x,
            raw.wedge_initial_dir_y,
            raw.wedge_initial_dir_z);
        result.wedge_events.bounce_depth = raw.wedge_bounce_depth;
        return result;
    }
}

template <bool Detached>
DfrAccumT<Detached> Scene::accum_dfr_direct(
    const DfrStatesT<Detached> &states,
    const DfrGrid &grid,
    const DfrMaterialT<Detached> &material,
    const DfrOptions &options,
    MaskT<Detached> active) const {
    ScopedNativeLaunchStage native_launch_stage(
        NativeLaunchStage::AccumDfr);
    require(is_ready(), "Scene::accum_dfr_direct(): scene is not built.");
    require(!pending_updates_,
            "Scene::accum_dfr_direct(): scene has pending updates. Call Scene::sync() first.");
    require(grid.axis >= 0 && grid.axis <= 2,
            "Scene::accum_dfr_direct(): grid.axis must be 0, 1, or 2.");
    require(grid.resolution0 > 0 && grid.resolution1 > 0,
            "Scene::accum_dfr_direct(): grid resolution must be positive.");
    require(grid.coord0_min < grid.coord0_max && grid.coord1_min < grid.coord1_max,
            "Scene::accum_dfr_direct(): grid bounds must be ordered.");
    require(grid.cell_area > 0.f,
            "Scene::accum_dfr_direct(): grid.cell_area must be positive.");
    require(options.wavelength > 0.f,
            "Scene::accum_dfr_direct(): wavelength must be positive.");
    require(options.max_order == 1,
            "Scene::accum_dfr_direct(): only max_order == 1 is supported.");

    DfrAccumT<Detached> result;
    const int grid_cell_count = grid.resolution0 * grid.resolution1;
    result.grid_cell_count = grid_cell_count;
    if constexpr (!Detached) {
        require_dfr_direct_custom_ad_supported(options);
        return dfr_direct_accum_custom_op(
            this,
            states,
            grid,
            material,
            options,
            triangle_info_.p0,
            triangle_info_.face_normal,
            global_geometry_.vertices,
            global_geometry_.faces,
            active);

        result.power = zeros<FloatAD>(grid_cell_count);
        result.field_x =
            drjit::Complex<FloatAD>(zeros<FloatAD>(grid_cell_count),
                                    zeros<FloatAD>(grid_cell_count));
        result.field_y =
            drjit::Complex<FloatAD>(zeros<FloatAD>(grid_cell_count),
                                    zeros<FloatAD>(grid_cell_count));
        result.field_z =
            drjit::Complex<FloatAD>(zeros<FloatAD>(grid_cell_count),
                                    zeros<FloatAD>(grid_cell_count));
        result.direct_count = full<IntAD>(0, 1);
        result.keller_count = full<IntAD>(0, 1);
        result.suffix_count = full<IntAD>(0, 1);
        result.vis_rejects = full<IntAD>(0, 1);
        result.edge_vis_rejects = full<IntAD>(0, 1);
        result.utd_rejects = full<IntAD>(0, 1);
        result.edge_uses = full<IntAD>(0, 1);

        const int state_width = static_cast<int>(slices(states.edge_index));
        const int state_count = states.count > 0 ? states.count : state_width;
        if (state_count == 0) {
            return result;
        }
        require(state_count > 0 && state_count <= state_width,
                "Scene::accum_dfr_direct(): invalid state count.");
        require(static_cast<int>(slices(states.edge_pos)) >= state_count &&
                    static_cast<int>(slices(states.edge_dir)) >= state_count &&
                    static_cast<int>(slices(states.edge_t_min)) >= state_count &&
                    static_cast<int>(slices(states.edge_t_max)) >= state_count &&
                    static_cast<int>(slices(states.n0)) >= state_count &&
                    static_cast<int>(slices(states.n1)) >= state_count &&
                    static_cast<int>(slices(states.prim0)) >= state_count &&
                    static_cast<int>(slices(states.prim1)) >= state_count &&
                    static_cast<int>(slices(states.exterior_angle)) >= state_count &&
                    static_cast<int>(slices(states.src)) >= state_count &&
                    static_cast<int>(slices(states.src_power)) >= state_count &&
                    static_cast<int>(slices(states.wi)) >= state_count &&
                    static_cast<int>(slices(states.d0)) >= state_count &&
                    static_cast<int>(slices(states.prefix_depth)) >= state_count,
                "Scene::accum_dfr_direct(): state fields must cover state count.");

        const int material_count = static_cast<int>(slices(material.eta_r));
        require(material_count > 0,
                "Scene::accum_dfr_direct(): material payload must not be empty.");
        require(static_cast<int>(slices(material.sigma)) == material_count &&
                    static_cast<int>(slices(material.mu_r)) == material_count &&
                    static_cast<int>(slices(material.gain)) == material_count &&
                    static_cast<int>(slices(material.valid)) == material_count,
                "Scene::accum_dfr_direct(): material payload fields must have matching widths.");
        {
            const OptixSceneSelection scenes = select_optix_scenes();
            const OptixScene *primary_scene = scenes.primary;
            require(primary_scene != nullptr && primary_scene->is_ready(),
                    "Scene::accum_dfr_direct(): OptiX scene is not ready.");
            require(scenes.hitgroup_record_count > 0,
                    "Scene::accum_dfr_direct(): invalid hitgroup record count.");
            ensure_pipeline(diffraction_accumulation_pipeline_,
                            primary_scene->context(),
                            scenes.hitgroup_record_count,
                            diffraction_accumulation_pipeline_config());
        }

        const int direct_samples =
            (options.strategy_mask & RAYD_DFR_DIRECT) != 0
                ? (options.direct_samples > 0 ? options.direct_samples : options.samples)
                : 0;
        const int keller_samples =
            (options.strategy_mask & RAYD_DFR_KELLER) != 0 ? options.keller_samples : 0;
        const int suffix_samples =
            (options.strategy_mask & RAYD_DFR_SUFFIX_REFL) != 0 ? options.suffix_samples : 0;
        const int launch_count = direct_samples + keller_samples + suffix_samples;
        if (launch_count <= 0) {
            return result;
        }

        MaskAD active_ad = active;
        const int active_width = static_cast<int>(slices(active_ad));
        if (active_width == 1 && state_count > 1) {
            active_ad = gather<MaskAD>(active_ad, zeros<IntAD>(state_count));
        } else {
            require(active_width == state_count,
                    "Scene::accum_dfr_direct(): active width must be 1 or match state count.");
        }

        auto grid_cell_center = [](const DfrGrid &grid_desc,
                                   const IntAD &cell) -> Vector3fAD {
            const IntAD ix = cell % IntAD(grid_desc.resolution0);
            const IntAD iy = cell / IntAD(grid_desc.resolution0);
            const FloatAD u =
                (FloatAD(ix) + FloatAD(0.5f)) /
                maximum(FloatAD(grid_desc.resolution0), FloatAD(1.f));
            const FloatAD v =
                (FloatAD(iy) + FloatAD(0.5f)) /
                maximum(FloatAD(grid_desc.resolution1), FloatAD(1.f));
            const FloatAD c0 =
                FloatAD(grid_desc.coord0_min) +
                u * FloatAD(grid_desc.coord0_max - grid_desc.coord0_min);
            const FloatAD c1 =
                FloatAD(grid_desc.coord1_min) +
                v * FloatAD(grid_desc.coord1_max - grid_desc.coord1_min);
            if (grid_desc.axis == 0) {
                return Vector3fAD(FloatAD(grid_desc.position), c0, c1);
            }
            if (grid_desc.axis == 1) {
                return Vector3fAD(c0, FloatAD(grid_desc.position), c1);
            }
            return Vector3fAD(c0, c1, FloatAD(grid_desc.position));
        };
        auto hash_u32 = [](UIntAD value) -> UIntAD {
            value ^= value >> 16u;
            value *= UIntAD(0x7feb352du);
            value ^= value >> 15u;
            value *= UIntAD(0x846ca68bu);
            value ^= value >> 16u;
            return value;
        };
        auto uniform01 = [&](const UIntAD &sample_lane, unsigned int stream) -> FloatAD {
            const UIntAD h =
                hash_u32(sample_lane ^ (UIntAD(stream) * UIntAD(0x9e3779b9u)) ^
                         UIntAD(static_cast<unsigned int>(options.seed)));
            return FloatAD(h & UIntAD(0x00ffffffu)) * FloatAD(1.f / 16777216.f);
        };
        auto material_gain_for_faces = [&](const IntAD &prim0,
                                           const IntAD &prim1,
                                           const MaskAD &sample_active) -> FloatAD {
            const MaskAD prim0_in_range =
                sample_active && (prim0 >= IntAD(0)) && (prim0 < IntAD(material_count));
            const IntAD safe0 = select(prim0_in_range, prim0, IntAD(0));
            const MaskAD prim0_valid =
                prim0_in_range && gather<MaskAD>(material.valid, safe0, prim0_in_range);
            const MaskAD prim1_in_range =
                sample_active && (prim1 >= IntAD(0)) && (prim1 < IntAD(material_count));
            const IntAD safe1 = select(prim1_in_range, prim1, IntAD(0));
            const MaskAD prim1_valid =
                prim1_in_range && gather<MaskAD>(material.valid, safe1, prim1_in_range);
            const IntAD chosen = select(prim0_valid, safe0, safe1);
            const MaskAD chosen_valid = prim0_valid || prim1_valid;
            const FloatAD gain =
                gather<FloatAD>(material.gain, chosen, chosen_valid);
            return select(chosen_valid, maximum(gain, FloatAD(0.f)), FloatAD(1.f));
        };

        const UIntAD lane = arange<UIntAD>(launch_count);
        const IntAD lane_i = IntAD(lane);
        const MaskAD is_direct = lane_i < IntAD(direct_samples);
        const MaskAD is_keller =
            !is_direct && (lane_i < IntAD(direct_samples + keller_samples));
        const MaskAD is_suffix =
            !is_direct && !is_keller && (lane_i < IntAD(launch_count));
        const IntAD state_idx = IntAD(lane % UIntAD(state_count));
        const IntAD cell =
            IntAD((lane / UIntAD(state_count)) % UIntAD(grid_cell_count));
        const MaskAD lane_active = full<MaskAD>(true, launch_count);
        const MaskAD state_active =
            gather<MaskAD>(active_ad, state_idx, lane_active);

        const Vector3fAD edge_pos =
            gather<Vector3fAD>(states.edge_pos, state_idx, state_active);
        const Vector3fAD edge_dir =
            normalize(gather<Vector3fAD>(states.edge_dir, state_idx, state_active));
        const FloatAD edge_t_min =
            gather<FloatAD>(states.edge_t_min, state_idx, state_active);
        const FloatAD edge_t_max =
            gather<FloatAD>(states.edge_t_max, state_idx, state_active);
        const FloatAD edge_t =
            edge_t_min + uniform01(lane, 0u) * (edge_t_max - edge_t_min);
        const Vector3fAD edge_point = edge_pos + edge_t * edge_dir;
        const Vector3fAD source =
            gather<Vector3fAD>(states.src, state_idx, state_active);
        const FloatAD src_power =
            gather<FloatAD>(states.src_power, state_idx, state_active);
        const IntAD prim0 = gather<IntAD>(states.prim0, state_idx, state_active);
        const IntAD prim1 = gather<IntAD>(states.prim1, state_idx, state_active);
        const FloatAD exterior_angle =
            gather<FloatAD>(states.exterior_angle, state_idx, state_active);
        const Vector3fAD target = grid_cell_center(grid, cell);

        const MaskAD finite_active =
            state_active &&
            drjit::isfinite(source.x()) &&
            drjit::isfinite(source.y()) &&
            drjit::isfinite(source.z()) &&
            drjit::isfinite(edge_point.x()) &&
            drjit::isfinite(edge_point.y()) &&
            drjit::isfinite(edge_point.z()) &&
            drjit::isfinite(src_power);
        const SegmentPairVisibilityAD visibility =
            this->template visible_pair<false>(
                edge_point,
                source,
                target,
                Int(),
                finite_active);
        const MaskAD visible = visibility.visible_a && visibility.visible_b;

        const FloatAD source_distance =
            maximum(norm(edge_point - source), FloatAD(Epsilon));
        const FloatAD target_distance =
            maximum(norm(target - edge_point), FloatAD(Epsilon));
        const FloatAD edge_length =
            maximum(edge_t_max - edge_t_min, FloatAD(0.f));
        const FloatAD wedge_scale =
            minimum(
                maximum(exterior_angle, FloatAD(0.25f * Pi)) / FloatAD(2.f * Pi),
                FloatAD(2.f));
        const FloatAD material_gain =
            material_gain_for_faces(prim0, prim1, finite_active);
        const IntAD strategy_samples = select(
            is_direct,
            IntAD(std::max(direct_samples, 1)),
            select(is_keller,
                   IntAD(std::max(keller_samples, 1)),
                   IntAD(std::max(suffix_samples, 1))));
        const FloatAD contribution =
            src_power *
            material_gain *
            edge_length *
            FloatAD(grid.cell_area) *
            wedge_scale /
            FloatAD(strategy_samples) /
            (source_distance * source_distance * target_distance * target_distance);
        const MaskAD contribution_active =
            visible && (contribution > FloatAD(0.f)) && drjit::isfinite(contribution);
        scatter_reduce(
            ReduceOp::Add,
            result.power,
            contribution,
            cell,
            contribution_active);
        const FloatAD amplitude =
            sqrt(maximum(contribution, FloatAD(0.f)));
        scatter_reduce(
            ReduceOp::Add,
            result.field_x.x(),
            amplitude,
            cell,
            contribution_active);
        scatter_reduce(
            ReduceOp::Add,
            result.direct_count,
            IntAD(1),
            zeros<IntAD>(launch_count),
            contribution_active && is_direct);
        scatter_reduce(
            ReduceOp::Add,
            result.keller_count,
            IntAD(1),
            zeros<IntAD>(launch_count),
            contribution_active && is_keller);
        scatter_reduce(
            ReduceOp::Add,
            result.suffix_count,
            IntAD(1),
            zeros<IntAD>(launch_count),
            contribution_active && is_suffix);
        scatter_reduce(
            ReduceOp::Add,
            result.edge_uses,
            IntAD(1),
            zeros<IntAD>(launch_count),
            contribution_active && options.collect_edge_use);
        scatter_reduce(
            ReduceOp::Add,
            result.vis_rejects,
            IntAD(1),
            zeros<IntAD>(launch_count),
            finite_active && !visible && options.collect_debug_counts);
        return result;
    } else {
        result.power = zeros<Float>(grid_cell_count);
        result.field_x =
            drjit::Complex<Float>(zeros<Float>(grid_cell_count),
                                  zeros<Float>(grid_cell_count));
        result.field_y =
            drjit::Complex<Float>(zeros<Float>(grid_cell_count),
                                  zeros<Float>(grid_cell_count));
        result.field_z =
            drjit::Complex<Float>(zeros<Float>(grid_cell_count),
                                  zeros<Float>(grid_cell_count));
        result.direct_count = full<Int>(0, 1);
        result.keller_count = full<Int>(0, 1);
        result.suffix_count = full<Int>(0, 1);
        result.vis_rejects = full<Int>(0, 1);
        result.edge_vis_rejects = full<Int>(0, 1);
        result.utd_rejects = full<Int>(0, 1);
        result.edge_uses = full<Int>(0, 1);

        const int state_width = static_cast<int>(slices(states.edge_index));
        const int state_count = states.count > 0 ? states.count : state_width;
        if (state_count == 0) {
            return result;
        }
        require(state_count > 0 && state_count <= state_width,
                "Scene::accum_dfr_direct(): invalid state count.");
        require(static_cast<int>(slices(states.edge_pos)) >= state_count &&
                    static_cast<int>(slices(states.edge_dir)) >= state_count &&
                    static_cast<int>(slices(states.edge_t_min)) >= state_count &&
                    static_cast<int>(slices(states.edge_t_max)) >= state_count &&
                    static_cast<int>(slices(states.n0)) >= state_count &&
                    static_cast<int>(slices(states.n1)) >= state_count &&
                    static_cast<int>(slices(states.prim0)) >= state_count &&
                    static_cast<int>(slices(states.prim1)) >= state_count &&
                    static_cast<int>(slices(states.exterior_angle)) >= state_count &&
                    static_cast<int>(slices(states.src)) >= state_count &&
                    static_cast<int>(slices(states.src_power)) >= state_count &&
                    static_cast<int>(slices(states.wi)) >= state_count &&
                    static_cast<int>(slices(states.d0)) >= state_count &&
                    static_cast<int>(slices(states.prefix_depth)) >= state_count,
                "Scene::accum_dfr_direct(): state fields must cover state count.");

        const int material_count = static_cast<int>(slices(material.eta_r));
        require(material_count > 0,
                "Scene::accum_dfr_direct(): material payload must not be empty.");
        require(static_cast<int>(slices(material.sigma)) == material_count &&
                    static_cast<int>(slices(material.mu_r)) == material_count &&
                    static_cast<int>(slices(material.gain)) == material_count &&
                    static_cast<int>(slices(material.valid)) == material_count,
                "Scene::accum_dfr_direct(): material payload fields must have matching widths.");

        const int direct_samples =
            (options.strategy_mask & RAYD_DFR_DIRECT) != 0
                ? (options.direct_samples > 0 ? options.direct_samples : options.samples)
                : 0;
        const int keller_samples =
            (options.strategy_mask & RAYD_DFR_KELLER) != 0 ? options.keller_samples : 0;
        const int suffix_samples =
            (options.strategy_mask & RAYD_DFR_SUFFIX_REFL) != 0 ? options.suffix_samples : 0;
        const int launch_count = direct_samples + keller_samples + suffix_samples;
        if (launch_count <= 0) {
            return result;
        }

        Mask active_detached = active;
        const int active_width = static_cast<int>(slices(active_detached));
        if (active_width == 1 && state_count > 1) {
            active_detached = gather<Mask>(active_detached, zeros<Int>(state_count));
        } else {
            require(active_width == state_count,
                    "Scene::accum_dfr_direct(): active width must be 1 or match state count.");
        }
        active_detached &= drjit::isfinite(states.src.x()) &&
                           drjit::isfinite(states.src.y()) &&
                           drjit::isfinite(states.src.z()) &&
                           drjit::isfinite(states.edge_pos.x()) &&
                           drjit::isfinite(states.edge_pos.y()) &&
                           drjit::isfinite(states.edge_pos.z()) &&
                           drjit::isfinite(states.src_power);

        const OptixSceneSelection scenes = select_optix_scenes();
        const OptixScene *primary_scene = scenes.primary;
        const OptixScene *secondary_scene = scenes.secondary;
        const int split_mode = scenes.split_mode;
        const int hitgroup_record_count = scenes.hitgroup_record_count;
        const int triangle_count =
            static_cast<int>(slices(triangle_info_detached_.p0));
        require(primary_scene != nullptr && primary_scene->is_ready(),
                "Scene::accum_dfr_direct(): OptiX scene is not ready.");
        require(hitgroup_record_count > 0,
                "Scene::accum_dfr_direct(): invalid hitgroup record count.");
        if (suffix_samples > 0) {
            require(triangle_count > 0,
                    "Scene::accum_dfr_direct(): suffix reflection requires scene triangles.");
            require(material_count >= triangle_count,
                    "Scene::accum_dfr_direct(): suffix reflection requires per-triangle materials.");
        }

        ensure_pipeline(diffraction_accumulation_pipeline_,
                        primary_scene->context(),
                        hitgroup_record_count,
                        diffraction_accumulation_pipeline_config());

        drjit::eval(states.edge_index,
                    states.edge_pos,
                    states.edge_dir,
                    states.edge_t_min,
                    states.edge_t_max,
                    states.n0,
                    states.n1,
                    states.prim0,
                    states.prim1,
                    states.exterior_angle,
                    states.src,
                    states.src_power,
                    states.wi,
                    states.d0,
                    states.prefix_depth,
                    active_detached,
                    material.eta_r,
                    material.sigma,
                    material.mu_r,
                    material.gain,
                    material.valid);
        if (suffix_samples > 0) {
            drjit::eval(triangle_info_detached_.p0,
                        triangle_info_detached_.e1,
                        triangle_info_detached_.e2,
                        triangle_info_detached_.face_normal,
                        face_offsets_);
        }

        DfrAccumRaw raw = alloc_dfr_accum_raw(grid_cell_count);
        init_dfr_accum_raw(raw);

        DfrAccumParams params = {};
        params.primary_handle = primary_scene->ias_handle();
        params.secondary_handle =
            secondary_scene != nullptr && secondary_scene->is_ready() ? secondary_scene->ias_handle() : 0ull;
        params.split_mode = split_mode;
        params.n_rays = launch_count;
        params.active_mask = reinterpret_cast<const uint8_t *>(active_detached.data());
        params.state_count = state_count;
        params.state_edge_index = states.edge_index.data();
        params.state_edge_pos_x = states.edge_pos.x().data();
        params.state_edge_pos_y = states.edge_pos.y().data();
        params.state_edge_pos_z = states.edge_pos.z().data();
        params.state_edge_dir_x = states.edge_dir.x().data();
        params.state_edge_dir_y = states.edge_dir.y().data();
        params.state_edge_dir_z = states.edge_dir.z().data();
        params.state_edge_t_min = states.edge_t_min.data();
        params.state_edge_t_max = states.edge_t_max.data();
        params.state_n0_x = states.n0.x().data();
        params.state_n0_y = states.n0.y().data();
        params.state_n0_z = states.n0.z().data();
        params.state_n1_x = states.n1.x().data();
        params.state_n1_y = states.n1.y().data();
        params.state_n1_z = states.n1.z().data();
        params.state_prim0 = states.prim0.data();
        params.state_prim1 = states.prim1.data();
        params.state_exterior_angle = states.exterior_angle.data();
        params.state_src_x = states.src.x().data();
        params.state_src_y = states.src.y().data();
        params.state_src_z = states.src.z().data();
        params.state_src_power = states.src_power.data();
        params.state_wi_x = states.wi.x().data();
        params.state_wi_y = states.wi.y().data();
        params.state_wi_z = states.wi.z().data();
        params.state_d0_x = states.d0.x().data();
        params.state_d0_y = states.d0.y().data();
        params.state_d0_z = states.d0.z().data();
        params.state_prefix_depth = states.prefix_depth.data();
        params.grid_axis = grid.axis;
        params.grid_position = grid.position;
        params.grid_coord0_min = grid.coord0_min;
        params.grid_coord0_max = grid.coord0_max;
        params.grid_coord1_min = grid.coord1_min;
        params.grid_coord1_max = grid.coord1_max;
        params.grid_resolution0 = grid.resolution0;
        params.grid_resolution1 = grid.resolution1;
        params.grid_cell_area = grid.cell_area;
        params.tri_p0_x = suffix_samples > 0 ? triangle_info_detached_.p0.x().data() : nullptr;
        params.tri_p0_y = suffix_samples > 0 ? triangle_info_detached_.p0.y().data() : nullptr;
        params.tri_p0_z = suffix_samples > 0 ? triangle_info_detached_.p0.z().data() : nullptr;
        params.tri_e1_x = suffix_samples > 0 ? triangle_info_detached_.e1.x().data() : nullptr;
        params.tri_e1_y = suffix_samples > 0 ? triangle_info_detached_.e1.y().data() : nullptr;
        params.tri_e1_z = suffix_samples > 0 ? triangle_info_detached_.e1.z().data() : nullptr;
        params.tri_e2_x = suffix_samples > 0 ? triangle_info_detached_.e2.x().data() : nullptr;
        params.tri_e2_y = suffix_samples > 0 ? triangle_info_detached_.e2.y().data() : nullptr;
        params.tri_e2_z = suffix_samples > 0 ? triangle_info_detached_.e2.z().data() : nullptr;
        params.tri_fn_x = suffix_samples > 0 ? triangle_info_detached_.face_normal.x().data() : nullptr;
        params.tri_fn_y = suffix_samples > 0 ? triangle_info_detached_.face_normal.y().data() : nullptr;
        params.tri_fn_z = suffix_samples > 0 ? triangle_info_detached_.face_normal.z().data() : nullptr;
        params.face_offsets = suffix_samples > 0 ? face_offsets_.data() : nullptr;
        params.n_meshes = mesh_count_;
        params.n_triangles = triangle_count;
        params.suffix_candidate_prim_id = nullptr;
        params.suffix_candidate_count = 0;
        params.material_eta_r = material.eta_r.data();
        params.material_sigma = material.sigma.data();
        params.material_mu_r = material.mu_r.data();
        params.material_gain = material.gain.data();
        params.material_valid = reinterpret_cast<const uint8_t *>(material.valid.data());
        params.material_count = material_count;
        params.wavelength = options.wavelength;
        params.k = options.k;
        params.seed = options.seed;
        params.samples = options.samples;
        params.max_order = options.max_order;
        params.direct_samples = direct_samples;
        params.keller_samples = keller_samples;
        params.suffix_samples = suffix_samples;
        params.strategy_mask = options.strategy_mask;
        params.sample_sequence = options.sample_sequence;
        params.receiver_model = options.receiver_model;
        params.collect_edge_use = options.collect_edge_use ? 1 : 0;
        params.collect_debug_counts = options.collect_debug_counts ? 1 : 0;
        params.out_power = raw.power.data();
        params.out_field_x_re = raw.field_x_re.data();
        params.out_field_x_im = raw.field_x_im.data();
        params.out_field_y_re = raw.field_y_re.data();
        params.out_field_y_im = raw.field_y_im.data();
        params.out_field_z_re = raw.field_z_re.data();
        params.out_field_z_im = raw.field_z_im.data();
        params.out_direct_count = raw.direct_count.data();
        params.out_keller_count = raw.keller_count.data();
        params.out_suffix_count = raw.suffix_count.data();
        params.out_vis_rejects = raw.vis_rejects.data();
        params.out_edge_vis_rejects =
            raw.edge_vis_rejects.data();
        params.out_utd_rejects = raw.utd_rejects.data();
        params.out_edge_uses = raw.edge_uses.data();
        if (active_dfr_direct_tape_capture != nullptr &&
            active_dfr_direct_tape_capture->launch_count == launch_count) {
            params.tape_active = reinterpret_cast<uint8_t *>(
                active_dfr_direct_tape_capture->active.data());
            params.tape_state_idx =
                active_dfr_direct_tape_capture->state_idx.data();
            params.tape_cell =
                active_dfr_direct_tape_capture->cell.data();
            params.tape_material_idx =
                active_dfr_direct_tape_capture->material_idx.data();
            params.tape_edge_u =
                active_dfr_direct_tape_capture->edge_u.data();
        }

        diffraction_accumulation_pipeline_->launch(0, params);

        result.power = raw.power;
        result.field_x =
            drjit::Complex<Float>(raw.field_x_re, raw.field_x_im);
        result.field_y =
            drjit::Complex<Float>(raw.field_y_re, raw.field_y_im);
        result.field_z =
            drjit::Complex<Float>(raw.field_z_re, raw.field_z_im);
        result.direct_count = raw.direct_count;
        result.keller_count = raw.keller_count;
        result.suffix_count = raw.suffix_count;
        result.vis_rejects = raw.vis_rejects;
        result.edge_vis_rejects =
            raw.edge_vis_rejects;
        result.utd_rejects = raw.utd_rejects;
        result.edge_uses = raw.edge_uses;
        return result;
    }
}

template <bool Detached>
DfrAccumT<Detached> Scene::accum_dfr(
    const DfrStatesT<Detached> &initial_states,
    const DfrStatesT<Detached> &recursive_states,
    const DfrGrid &grid,
    const DfrMaterialT<Detached> &material,
    const DfrOptions &options,
    MaskT<Detached> active) const {
    ScopedNativeLaunchStage native_launch_stage(
        NativeLaunchStage::AccumDfr);
    require(is_ready(), "Scene::accum_dfr(): scene is not built.");
    require(!pending_updates_,
            "Scene::accum_dfr(): scene has pending updates. Call Scene::sync() first.");
    require(grid.axis >= 0 && grid.axis <= 2,
            "Scene::accum_dfr(): grid.axis must be 0, 1, or 2.");
    require(grid.resolution0 > 0 && grid.resolution1 > 0,
            "Scene::accum_dfr(): grid resolution must be positive.");
    require(grid.coord0_min < grid.coord0_max && grid.coord1_min < grid.coord1_max,
            "Scene::accum_dfr(): grid bounds must be ordered.");
    require(grid.cell_area > 0.f,
            "Scene::accum_dfr(): grid.cell_area must be positive.");
    require(options.wavelength > 0.f,
            "Scene::accum_dfr(): wavelength must be positive.");
    require(options.max_order == 2 || options.max_order == 3,
            "Scene::accum_dfr(): only max_order 2 or 3 is supported.");

    DfrAccumT<Detached> result;
    const int grid_cell_count = grid.resolution0 * grid.resolution1;
    result.grid_cell_count = grid_cell_count;
    if constexpr (!Detached) {
        require_dfr_chain_custom_ad_supported(options);
        return dfr_chain_accum_custom_op(
            this,
            initial_states,
            recursive_states,
            grid,
            material,
            options,
            triangle_info_.p0,
            triangle_info_.face_normal,
            global_geometry_.vertices,
            global_geometry_.faces,
            active);

        result.power = zeros<FloatAD>(grid_cell_count);
        result.field_x =
            drjit::Complex<FloatAD>(zeros<FloatAD>(grid_cell_count),
                                    zeros<FloatAD>(grid_cell_count));
        result.field_y =
            drjit::Complex<FloatAD>(zeros<FloatAD>(grid_cell_count),
                                    zeros<FloatAD>(grid_cell_count));
        result.field_z =
            drjit::Complex<FloatAD>(zeros<FloatAD>(grid_cell_count),
                                    zeros<FloatAD>(grid_cell_count));
        result.direct_count = full<IntAD>(0, 1);
        result.keller_count = full<IntAD>(0, 1);
        result.suffix_count = full<IntAD>(0, 1);
        result.vis_rejects = full<IntAD>(0, 1);
        result.edge_vis_rejects = full<IntAD>(0, 1);
        result.utd_rejects = full<IntAD>(0, 1);
        result.edge_uses = full<IntAD>(0, 1);

        const int initial_width = static_cast<int>(slices(initial_states.edge_index));
        const int initial_count =
            initial_states.count > 0 ? initial_states.count : initial_width;
        const int recursive_width = static_cast<int>(slices(recursive_states.edge_index));
        const int recursive_count =
            recursive_states.count > 0 ? recursive_states.count : recursive_width;
        if (initial_count == 0 || recursive_count == 0) {
            return result;
        }
        require(initial_count > 0 && initial_count <= initial_width,
                "Scene::accum_dfr(): invalid initial state count.");
        require(recursive_count > 0 && recursive_count <= recursive_width,
                "Scene::accum_dfr(): invalid recursive state count.");
        require(static_cast<int>(slices(initial_states.edge_pos)) >= initial_count &&
                    static_cast<int>(slices(initial_states.edge_dir)) >= initial_count &&
                    static_cast<int>(slices(initial_states.edge_t_min)) >= initial_count &&
                    static_cast<int>(slices(initial_states.edge_t_max)) >= initial_count &&
                    static_cast<int>(slices(initial_states.prim0)) >= initial_count &&
                    static_cast<int>(slices(initial_states.prim1)) >= initial_count &&
                    static_cast<int>(slices(initial_states.exterior_angle)) >= initial_count &&
                    static_cast<int>(slices(initial_states.src)) >= initial_count &&
                    static_cast<int>(slices(initial_states.src_power)) >= initial_count,
                "Scene::accum_dfr(): initial state fields must cover state count.");
        require(static_cast<int>(slices(recursive_states.edge_index)) >= recursive_count &&
                    static_cast<int>(slices(recursive_states.edge_pos)) >= recursive_count &&
                    static_cast<int>(slices(recursive_states.edge_dir)) >= recursive_count &&
                    static_cast<int>(slices(recursive_states.edge_t_min)) >= recursive_count &&
                    static_cast<int>(slices(recursive_states.edge_t_max)) >= recursive_count &&
                    static_cast<int>(slices(recursive_states.prim0)) >= recursive_count &&
                    static_cast<int>(slices(recursive_states.prim1)) >= recursive_count &&
                    static_cast<int>(slices(recursive_states.exterior_angle)) >= recursive_count,
                "Scene::accum_dfr(): recursive state fields must cover state count.");

        const int material_count = static_cast<int>(slices(material.eta_r));
        require(material_count > 0,
                "Scene::accum_dfr(): material payload must not be empty.");
        require(static_cast<int>(slices(material.sigma)) == material_count &&
                    static_cast<int>(slices(material.mu_r)) == material_count &&
                    static_cast<int>(slices(material.gain)) == material_count &&
                    static_cast<int>(slices(material.valid)) == material_count,
                "Scene::accum_dfr(): material payload fields must have matching widths.");
        {
            const OptixSceneSelection scenes = select_optix_scenes();
            const OptixScene *primary_scene = scenes.primary;
            require(primary_scene != nullptr && primary_scene->is_ready(),
                    "Scene::accum_dfr(): OptiX scene is not ready.");
            require(scenes.hitgroup_record_count > 0,
                    "Scene::accum_dfr(): invalid hitgroup record count.");
            ensure_pipeline(diffraction_accumulation_pipeline_,
                            primary_scene->context(),
                            scenes.hitgroup_record_count,
                            diffraction_accumulation_pipeline_config());
        }

        const int direct_samples =
            (options.strategy_mask & RAYD_DFR_DIRECT) != 0
                ? (options.direct_samples > 0 ? options.direct_samples : options.samples)
                : 0;
        const int keller_samples =
            (options.strategy_mask & RAYD_DFR_KELLER) != 0 ? options.keller_samples : 0;
        const int suffix_samples =
            (options.strategy_mask & RAYD_DFR_SUFFIX_REFL) != 0 ? options.suffix_samples : 0;
        const int launch_count = direct_samples + keller_samples + suffix_samples;
        if (launch_count <= 0) {
            return result;
        }

        MaskAD active_ad = active;
        const int active_width = static_cast<int>(slices(active_ad));
        if (active_width == 1 && initial_count > 1) {
            active_ad = gather<MaskAD>(active_ad, zeros<IntAD>(initial_count));
        } else {
            require(active_width == initial_count,
                    "Scene::accum_dfr(): active width must be 1 or match initial state count.");
        }

        auto grid_cell_center = [](const DfrGrid &grid_desc,
                                   const IntAD &cell) -> Vector3fAD {
            const IntAD ix = cell % IntAD(grid_desc.resolution0);
            const IntAD iy = cell / IntAD(grid_desc.resolution0);
            const FloatAD u =
                (FloatAD(ix) + FloatAD(0.5f)) /
                maximum(FloatAD(grid_desc.resolution0), FloatAD(1.f));
            const FloatAD v =
                (FloatAD(iy) + FloatAD(0.5f)) /
                maximum(FloatAD(grid_desc.resolution1), FloatAD(1.f));
            const FloatAD c0 =
                FloatAD(grid_desc.coord0_min) +
                u * FloatAD(grid_desc.coord0_max - grid_desc.coord0_min);
            const FloatAD c1 =
                FloatAD(grid_desc.coord1_min) +
                v * FloatAD(grid_desc.coord1_max - grid_desc.coord1_min);
            if (grid_desc.axis == 0) {
                return Vector3fAD(FloatAD(grid_desc.position), c0, c1);
            }
            if (grid_desc.axis == 1) {
                return Vector3fAD(c0, FloatAD(grid_desc.position), c1);
            }
            return Vector3fAD(c0, c1, FloatAD(grid_desc.position));
        };
        auto hash_u32 = [](UIntAD value) -> UIntAD {
            value ^= value >> 16u;
            value *= UIntAD(0x7feb352du);
            value ^= value >> 15u;
            value *= UIntAD(0x846ca68bu);
            value ^= value >> 16u;
            return value;
        };
        auto uniform01 = [&](const UIntAD &sample_lane, unsigned int stream) -> FloatAD {
            const UIntAD h =
                hash_u32(sample_lane ^ (UIntAD(stream) * UIntAD(0x9e3779b9u)) ^
                         UIntAD(static_cast<unsigned int>(options.seed)));
            return FloatAD(h & UIntAD(0x00ffffffu)) * FloatAD(1.f / 16777216.f);
        };

        auto material_gain_for_faces = [&](const IntAD &prim0,
                                           const IntAD &prim1,
                                           const MaskAD &sample_active) -> FloatAD {
            const MaskAD prim0_in_range =
                sample_active && (prim0 >= IntAD(0)) && (prim0 < IntAD(material_count));
            const IntAD safe0 = select(prim0_in_range, prim0, IntAD(0));
            const MaskAD prim0_valid =
                prim0_in_range && gather<MaskAD>(material.valid, safe0, prim0_in_range);
            const MaskAD prim1_in_range =
                sample_active && (prim1 >= IntAD(0)) && (prim1 < IntAD(material_count));
            const IntAD safe1 = select(prim1_in_range, prim1, IntAD(0));
            const MaskAD prim1_valid =
                prim1_in_range && gather<MaskAD>(material.valid, safe1, prim1_in_range);
            const IntAD chosen = select(prim0_valid, safe0, safe1);
            const MaskAD chosen_valid = prim0_valid || prim1_valid;
            const FloatAD gain =
                gather<FloatAD>(material.gain, chosen, chosen_valid);
            return select(chosen_valid, maximum(gain, FloatAD(0.f)), FloatAD(1.f));
        };

        auto chain_event_weight = [&](const FloatAD &src_power,
                                      const IntAD &face0_prim,
                                      const IntAD &face1_prim,
                                      const FloatAD &edge_t_min,
                                      const FloatAD &edge_t_max,
                                      const FloatAD &exterior_angle,
                                      const Vector3fAD &source,
                                      const Vector3fAD &edge_point,
                                      const Vector3fAD &target,
                                      const MaskAD &sample_active) -> FloatAD {
            const FloatAD source_distance =
                maximum(norm(edge_point - source), FloatAD(Epsilon));
            const FloatAD target_distance =
                maximum(norm(target - edge_point), FloatAD(Epsilon));
            const FloatAD edge_length =
                maximum(edge_t_max - edge_t_min, FloatAD(0.f));
            const FloatAD wedge_scale =
                minimum(
                    maximum(exterior_angle, FloatAD(0.25f * Pi)) / FloatAD(2.f * Pi),
                    FloatAD(2.f));
            const FloatAD material_gain =
                material_gain_for_faces(face0_prim, face1_prim, sample_active);
            return src_power *
                   material_gain *
                   edge_length *
                   wedge_scale /
                   (source_distance * source_distance * target_distance * target_distance);
        };

        const UIntAD lane = arange<UIntAD>(launch_count);
        const IntAD lane_i = IntAD(lane);
        const MaskAD is_direct = lane_i < IntAD(direct_samples);
        const MaskAD is_keller =
            !is_direct && (lane_i < IntAD(direct_samples + keller_samples));
        const MaskAD is_suffix =
            !is_direct && !is_keller && (lane_i < IntAD(launch_count));
        const IntAD first_idx = IntAD(lane % UIntAD(initial_count));
        const UIntAD second_hash = hash_u32(
            lane ^ (UIntAD(static_cast<unsigned int>(options.seed)) * UIntAD(0x9e3779b9u)) ^
            UIntAD(0x51ed270bu));
        const IntAD second_idx = IntAD(second_hash % UIntAD(recursive_count));
        const UIntAD third_hash = hash_u32(
            lane ^ (UIntAD(static_cast<unsigned int>(options.seed)) * UIntAD(0x85ebca6bu)) ^
            UIntAD(0xc2b2ae35u));
        const IntAD third_idx =
            IntAD(third_hash % UIntAD(recursive_count));
        const IntAD cell =
            IntAD((lane / UIntAD(initial_count)) % UIntAD(grid_cell_count));
        const MaskAD lane_active = full<MaskAD>(true, launch_count);
        const MaskAD first_active =
            gather<MaskAD>(active_ad, first_idx, lane_active);
        const IntAD first_edge_index =
            gather<IntAD>(initial_states.edge_index, first_idx, first_active);
        const IntAD second_edge_index =
            gather<IntAD>(recursive_states.edge_index, second_idx, first_active);
        const IntAD third_edge_index =
            gather<IntAD>(recursive_states.edge_index, third_idx, first_active);
        const MaskAD distinct_edges =
            (first_edge_index != second_edge_index) &&
            ((IntAD(options.max_order) == IntAD(2)) ||
             ((first_edge_index != third_edge_index) &&
              (second_edge_index != third_edge_index)));

        const Vector3fAD first_edge_pos =
            gather<Vector3fAD>(initial_states.edge_pos, first_idx, first_active);
        const Vector3fAD first_edge_dir =
            normalize(gather<Vector3fAD>(initial_states.edge_dir, first_idx, first_active));
        const FloatAD first_t_min =
            gather<FloatAD>(initial_states.edge_t_min, first_idx, first_active);
        const FloatAD first_t_max =
            gather<FloatAD>(initial_states.edge_t_max, first_idx, first_active);
        const FloatAD first_t =
            first_t_min + uniform01(lane, 0u) * (first_t_max - first_t_min);
        const Vector3fAD first_point = first_edge_pos + first_t * first_edge_dir;

        const Vector3fAD second_edge_pos =
            gather<Vector3fAD>(recursive_states.edge_pos, second_idx, first_active);
        const Vector3fAD second_edge_dir =
            normalize(gather<Vector3fAD>(recursive_states.edge_dir, second_idx, first_active));
        const FloatAD second_t_min =
            gather<FloatAD>(recursive_states.edge_t_min, second_idx, first_active);
        const FloatAD second_t_max =
            gather<FloatAD>(recursive_states.edge_t_max, second_idx, first_active);
        const FloatAD second_t =
            second_t_min + uniform01(lane, 2u) * (second_t_max - second_t_min);
        const Vector3fAD second_point = second_edge_pos + second_t * second_edge_dir;

        const Vector3fAD third_edge_pos =
            gather<Vector3fAD>(recursive_states.edge_pos, third_idx, first_active);
        const Vector3fAD third_edge_dir =
            normalize(gather<Vector3fAD>(recursive_states.edge_dir, third_idx, first_active));
        const FloatAD third_t_min =
            gather<FloatAD>(recursive_states.edge_t_min, third_idx, first_active);
        const FloatAD third_t_max =
            gather<FloatAD>(recursive_states.edge_t_max, third_idx, first_active);
        const FloatAD third_t =
            third_t_min + uniform01(lane, 4u) * (third_t_max - third_t_min);
        const Vector3fAD third_point = third_edge_pos + third_t * third_edge_dir;

        const Vector3fAD source =
            gather<Vector3fAD>(initial_states.src, first_idx, first_active);
        const FloatAD src_power =
            gather<FloatAD>(initial_states.src_power, first_idx, first_active);
        const Vector3fAD target = grid_cell_center(grid, cell);
        const Vector3fAD terminal_point =
            select(IntAD(options.max_order) == IntAD(3), third_point, second_point);

        const MaskAD finite_active =
            first_active &&
            distinct_edges &&
            drjit::isfinite(source.x()) &&
            drjit::isfinite(source.y()) &&
            drjit::isfinite(source.z()) &&
            drjit::isfinite(first_point.x()) &&
            drjit::isfinite(first_point.y()) &&
            drjit::isfinite(first_point.z()) &&
            drjit::isfinite(second_point.x()) &&
            drjit::isfinite(second_point.y()) &&
            drjit::isfinite(second_point.z()) &&
            drjit::isfinite(src_power);

        const SegmentPairVisibilityAD first_visibility =
            this->template visible_pair<false>(
                first_point,
                source,
                second_point,
                Int(),
                finite_active);
        const SegmentPairVisibilityAD terminal_visibility =
            this->template visible_pair<false>(
                terminal_point,
                select(IntAD(options.max_order) == IntAD(3), second_point, target),
                target,
                Int(),
                finite_active);
        const MaskAD source_visible = first_visibility.visible_a;
        const MaskAD first_edge_visible = first_visibility.visible_b;
        const MaskAD second_edge_visible =
            select(IntAD(options.max_order) == IntAD(3),
                   terminal_visibility.visible_a,
                   full<MaskAD>(true, launch_count));
        const MaskAD target_visible =
            select(IntAD(options.max_order) == IntAD(3),
                   terminal_visibility.visible_b,
                   terminal_visibility.visible_a);
        const MaskAD visible =
            source_visible && first_edge_visible && second_edge_visible && target_visible;

        const IntAD first_prim0 =
            gather<IntAD>(initial_states.prim0, first_idx, finite_active);
        const IntAD first_prim1 =
            gather<IntAD>(initial_states.prim1, first_idx, finite_active);
        const FloatAD first_exterior =
            gather<FloatAD>(initial_states.exterior_angle, first_idx, finite_active);
        const FloatAD first_weight = chain_event_weight(
            src_power,
            first_prim0,
            first_prim1,
            first_t_min,
            first_t_max,
            first_exterior,
            source,
            first_point,
            second_point,
            finite_active);

        const IntAD second_prim0 =
            gather<IntAD>(recursive_states.prim0, second_idx, finite_active);
        const IntAD second_prim1 =
            gather<IntAD>(recursive_states.prim1, second_idx, finite_active);
        const FloatAD second_exterior =
            gather<FloatAD>(recursive_states.exterior_angle, second_idx, finite_active);
        const Vector3fAD second_target =
            select(IntAD(options.max_order) == IntAD(3), third_point, target);
        const FloatAD second_weight = chain_event_weight(
            FloatAD(1.f),
            second_prim0,
            second_prim1,
            second_t_min,
            second_t_max,
            second_exterior,
            first_point,
            second_point,
            second_target,
            finite_active);

        const IntAD third_prim0 =
            gather<IntAD>(recursive_states.prim0, third_idx, finite_active);
        const IntAD third_prim1 =
            gather<IntAD>(recursive_states.prim1, third_idx, finite_active);
        const FloatAD third_exterior =
            gather<FloatAD>(recursive_states.exterior_angle, third_idx, finite_active);
        const FloatAD third_weight = chain_event_weight(
            FloatAD(1.f),
            third_prim0,
            third_prim1,
            third_t_min,
            third_t_max,
            third_exterior,
            second_point,
            third_point,
            target,
            finite_active);
        FloatAD chain_weight = first_weight * second_weight;
        chain_weight = select(IntAD(options.max_order) == IntAD(3),
                              chain_weight * third_weight,
                              chain_weight);

        const FloatAD wave_gain_per_event =
            (FloatAD(options.wavelength) / FloatAD(4.f * Pi)) *
            (FloatAD(options.wavelength) / FloatAD(4.f * Pi));
        const FloatAD wave_gain =
            select(IntAD(options.max_order) == IntAD(3),
                   wave_gain_per_event * wave_gain_per_event,
                   wave_gain_per_event);
        const IntAD strategy_samples = select(
            is_direct,
            IntAD(std::max(direct_samples, 1)),
            select(is_keller,
                   IntAD(std::max(keller_samples, 1)),
                   IntAD(std::max(suffix_samples, 1))));
        const FloatAD contribution =
            chain_weight *
            wave_gain *
            FloatAD(grid.cell_area) /
            FloatAD(strategy_samples);
        const MaskAD contribution_active =
            visible && (contribution > FloatAD(0.f)) && drjit::isfinite(contribution);
        scatter_reduce(
            ReduceOp::Add,
            result.power,
            contribution,
            cell,
            contribution_active);
        const FloatAD amplitude =
            sqrt(maximum(contribution, FloatAD(0.f)));
        scatter_reduce(
            ReduceOp::Add,
            result.field_x.x(),
            amplitude,
            cell,
            contribution_active);
        scatter_reduce(
            ReduceOp::Add,
            result.direct_count,
            IntAD(1),
            zeros<IntAD>(launch_count),
            contribution_active && is_direct);
        scatter_reduce(
            ReduceOp::Add,
            result.keller_count,
            IntAD(1),
            zeros<IntAD>(launch_count),
            contribution_active && is_keller);
        scatter_reduce(
            ReduceOp::Add,
            result.suffix_count,
            IntAD(1),
            zeros<IntAD>(launch_count),
            contribution_active && is_suffix);
        scatter_reduce(
            ReduceOp::Add,
            result.edge_uses,
            IntAD(1),
            zeros<IntAD>(launch_count),
            contribution_active && options.collect_edge_use);
        scatter_reduce(
            ReduceOp::Add,
            result.vis_rejects,
            IntAD(1),
            zeros<IntAD>(launch_count),
            finite_active && !target_visible && options.collect_debug_counts);
        scatter_reduce(
            ReduceOp::Add,
            result.edge_vis_rejects,
            IntAD(1),
            zeros<IntAD>(launch_count),
            finite_active && target_visible &&
                (!source_visible || !first_edge_visible || !second_edge_visible) &&
                options.collect_debug_counts);
        scatter_reduce(
            ReduceOp::Add,
            result.utd_rejects,
            IntAD(1),
            zeros<IntAD>(launch_count),
            first_active && !distinct_edges && options.collect_debug_counts);
        return result;
    } else {
        result.power = zeros<Float>(grid_cell_count);
        result.field_x =
            drjit::Complex<Float>(zeros<Float>(grid_cell_count),
                                  zeros<Float>(grid_cell_count));
        result.field_y =
            drjit::Complex<Float>(zeros<Float>(grid_cell_count),
                                  zeros<Float>(grid_cell_count));
        result.field_z =
            drjit::Complex<Float>(zeros<Float>(grid_cell_count),
                                  zeros<Float>(grid_cell_count));
        result.direct_count = full<Int>(0, 1);
        result.keller_count = full<Int>(0, 1);
        result.suffix_count = full<Int>(0, 1);
        result.vis_rejects = full<Int>(0, 1);
        result.edge_vis_rejects = full<Int>(0, 1);
        result.utd_rejects = full<Int>(0, 1);
        result.edge_uses = full<Int>(0, 1);

        const int initial_width = static_cast<int>(slices(initial_states.edge_index));
        const int initial_count =
            initial_states.count > 0 ? initial_states.count : initial_width;
        const int recursive_width = static_cast<int>(slices(recursive_states.edge_index));
        const int recursive_count =
            recursive_states.count > 0 ? recursive_states.count : recursive_width;
        if (initial_count == 0 || recursive_count == 0) {
            return result;
        }
        require(initial_count > 0 && initial_count <= initial_width,
                "Scene::accum_dfr(): invalid initial state count.");
        require(recursive_count > 0 && recursive_count <= recursive_width,
                "Scene::accum_dfr(): invalid recursive state count.");
        require(static_cast<int>(slices(initial_states.edge_pos)) >= initial_count &&
                    static_cast<int>(slices(initial_states.edge_dir)) >= initial_count &&
                    static_cast<int>(slices(initial_states.edge_t_min)) >= initial_count &&
                    static_cast<int>(slices(initial_states.edge_t_max)) >= initial_count &&
                    static_cast<int>(slices(initial_states.prim0)) >= initial_count &&
                    static_cast<int>(slices(initial_states.prim1)) >= initial_count &&
                    static_cast<int>(slices(initial_states.exterior_angle)) >= initial_count &&
                    static_cast<int>(slices(initial_states.src)) >= initial_count &&
                    static_cast<int>(slices(initial_states.src_power)) >= initial_count,
                "Scene::accum_dfr(): initial state fields must cover state count.");
        require(static_cast<int>(slices(recursive_states.edge_pos)) >= recursive_count &&
                    static_cast<int>(slices(recursive_states.edge_dir)) >= recursive_count &&
                    static_cast<int>(slices(recursive_states.edge_t_min)) >= recursive_count &&
                    static_cast<int>(slices(recursive_states.edge_t_max)) >= recursive_count &&
                    static_cast<int>(slices(recursive_states.prim0)) >= recursive_count &&
                    static_cast<int>(slices(recursive_states.prim1)) >= recursive_count &&
                    static_cast<int>(slices(recursive_states.exterior_angle)) >= recursive_count,
                "Scene::accum_dfr(): recursive state fields must cover state count.");

        const int material_count = static_cast<int>(slices(material.eta_r));
        require(material_count > 0,
                "Scene::accum_dfr(): material payload must not be empty.");
        require(static_cast<int>(slices(material.sigma)) == material_count &&
                    static_cast<int>(slices(material.mu_r)) == material_count &&
                    static_cast<int>(slices(material.gain)) == material_count &&
                    static_cast<int>(slices(material.valid)) == material_count,
                "Scene::accum_dfr(): material payload fields must have matching widths.");

        const int direct_samples =
            (options.strategy_mask & RAYD_DFR_DIRECT) != 0
                ? (options.direct_samples > 0 ? options.direct_samples : options.samples)
                : 0;
        const int keller_samples =
            (options.strategy_mask & RAYD_DFR_KELLER) != 0 ? options.keller_samples : 0;
        const int suffix_samples =
            (options.strategy_mask & RAYD_DFR_SUFFIX_REFL) != 0 ? options.suffix_samples : 0;
        const int launch_count = direct_samples + keller_samples + suffix_samples;
        if (launch_count <= 0) {
            return result;
        }

        Mask active_detached = active;
        const int active_width = static_cast<int>(slices(active_detached));
        if (active_width == 1 && initial_count > 1) {
            active_detached = gather<Mask>(active_detached, zeros<Int>(initial_count));
        } else {
            require(active_width == initial_count,
                    "Scene::accum_dfr(): active width must be 1 or match initial state count.");
        }
        active_detached &= drjit::isfinite(initial_states.src.x()) &&
                           drjit::isfinite(initial_states.src.y()) &&
                           drjit::isfinite(initial_states.src.z()) &&
                           drjit::isfinite(initial_states.edge_pos.x()) &&
                           drjit::isfinite(initial_states.edge_pos.y()) &&
                           drjit::isfinite(initial_states.edge_pos.z()) &&
                           drjit::isfinite(initial_states.src_power);
        Mask recursive_active = drjit::isfinite(recursive_states.edge_pos.x()) &&
                                drjit::isfinite(recursive_states.edge_pos.y()) &&
                                drjit::isfinite(recursive_states.edge_pos.z()) &&
                                drjit::isfinite(recursive_states.edge_dir.x()) &&
                                drjit::isfinite(recursive_states.edge_dir.y()) &&
                                drjit::isfinite(recursive_states.edge_dir.z());

        const OptixSceneSelection scenes = select_optix_scenes();
        const OptixScene *primary_scene = scenes.primary;
        const OptixScene *secondary_scene = scenes.secondary;
        const int split_mode = scenes.split_mode;
        const int hitgroup_record_count = scenes.hitgroup_record_count;
        const int triangle_count =
            static_cast<int>(slices(triangle_info_detached_.p0));
        require(primary_scene != nullptr && primary_scene->is_ready(),
                "Scene::accum_dfr(): OptiX scene is not ready.");
        require(hitgroup_record_count > 0,
                "Scene::accum_dfr(): invalid hitgroup record count.");
        if (suffix_samples > 0) {
            require(triangle_count > 0,
                    "Scene::accum_dfr(): suffix reflection requires scene triangles.");
            require(material_count >= triangle_count,
                    "Scene::accum_dfr(): suffix reflection requires per-triangle materials.");
        }

        ensure_pipeline(diffraction_accumulation_pipeline_,
                        primary_scene->context(),
                        hitgroup_record_count,
                        diffraction_accumulation_pipeline_config());

        drjit::eval(initial_states.edge_index,
                    initial_states.edge_pos,
                    initial_states.edge_dir,
                    initial_states.edge_t_min,
                    initial_states.edge_t_max,
                    initial_states.prim0,
                    initial_states.prim1,
                    initial_states.exterior_angle,
                    initial_states.src,
                    initial_states.src_power,
                    recursive_states.edge_index,
                    recursive_states.edge_pos,
                    recursive_states.edge_dir,
                    recursive_states.edge_t_min,
                    recursive_states.edge_t_max,
                    recursive_states.prim0,
                    recursive_states.prim1,
                    recursive_states.exterior_angle,
                    active_detached,
                    recursive_active,
                    material.eta_r,
                    material.sigma,
                    material.mu_r,
                    material.gain,
                    material.valid);
        if (suffix_samples > 0) {
            drjit::eval(triangle_info_detached_.p0,
                        triangle_info_detached_.e1,
                        triangle_info_detached_.e2,
                        triangle_info_detached_.face_normal,
                        face_offsets_);
        }

        DfrAccumRaw raw = alloc_dfr_accum_raw(grid_cell_count);
        init_dfr_accum_raw(raw);

        DfrAccumParams params = {};
        params.primary_handle = primary_scene->ias_handle();
        params.secondary_handle =
            secondary_scene != nullptr && secondary_scene->is_ready() ? secondary_scene->ias_handle() : 0ull;
        params.split_mode = split_mode;
        params.n_rays = launch_count;
        params.active_mask = reinterpret_cast<const uint8_t *>(active_detached.data());
        params.state_count = initial_count;
        params.state_edge_index = initial_states.edge_index.data();
        params.state_edge_pos_x = initial_states.edge_pos.x().data();
        params.state_edge_pos_y = initial_states.edge_pos.y().data();
        params.state_edge_pos_z = initial_states.edge_pos.z().data();
        params.state_edge_dir_x = initial_states.edge_dir.x().data();
        params.state_edge_dir_y = initial_states.edge_dir.y().data();
        params.state_edge_dir_z = initial_states.edge_dir.z().data();
        params.state_edge_t_min = initial_states.edge_t_min.data();
        params.state_edge_t_max = initial_states.edge_t_max.data();
        params.state_prim0 = initial_states.prim0.data();
        params.state_prim1 = initial_states.prim1.data();
        params.state_exterior_angle = initial_states.exterior_angle.data();
        params.state_src_x = initial_states.src.x().data();
        params.state_src_y = initial_states.src.y().data();
        params.state_src_z = initial_states.src.z().data();
        params.state_src_power = initial_states.src_power.data();

        params.recursive_state_count = recursive_count;
        params.recursive_active_mask =
            reinterpret_cast<const uint8_t *>(recursive_active.data());
        params.recursive_state_edge_index = recursive_states.edge_index.data();
        params.recursive_state_edge_pos_x = recursive_states.edge_pos.x().data();
        params.recursive_state_edge_pos_y = recursive_states.edge_pos.y().data();
        params.recursive_state_edge_pos_z = recursive_states.edge_pos.z().data();
        params.recursive_state_edge_dir_x = recursive_states.edge_dir.x().data();
        params.recursive_state_edge_dir_y = recursive_states.edge_dir.y().data();
        params.recursive_state_edge_dir_z = recursive_states.edge_dir.z().data();
        params.recursive_state_edge_t_min = recursive_states.edge_t_min.data();
        params.recursive_state_edge_t_max = recursive_states.edge_t_max.data();
        params.recursive_state_prim0 = recursive_states.prim0.data();
        params.recursive_state_prim1 = recursive_states.prim1.data();
        params.recursive_state_exterior_angle = recursive_states.exterior_angle.data();

        params.grid_axis = grid.axis;
        params.grid_position = grid.position;
        params.grid_coord0_min = grid.coord0_min;
        params.grid_coord0_max = grid.coord0_max;
        params.grid_coord1_min = grid.coord1_min;
        params.grid_coord1_max = grid.coord1_max;
        params.grid_resolution0 = grid.resolution0;
        params.grid_resolution1 = grid.resolution1;
        params.grid_cell_area = grid.cell_area;
        params.tri_p0_x = suffix_samples > 0 ? triangle_info_detached_.p0.x().data() : nullptr;
        params.tri_p0_y = suffix_samples > 0 ? triangle_info_detached_.p0.y().data() : nullptr;
        params.tri_p0_z = suffix_samples > 0 ? triangle_info_detached_.p0.z().data() : nullptr;
        params.tri_e1_x = suffix_samples > 0 ? triangle_info_detached_.e1.x().data() : nullptr;
        params.tri_e1_y = suffix_samples > 0 ? triangle_info_detached_.e1.y().data() : nullptr;
        params.tri_e1_z = suffix_samples > 0 ? triangle_info_detached_.e1.z().data() : nullptr;
        params.tri_e2_x = suffix_samples > 0 ? triangle_info_detached_.e2.x().data() : nullptr;
        params.tri_e2_y = suffix_samples > 0 ? triangle_info_detached_.e2.y().data() : nullptr;
        params.tri_e2_z = suffix_samples > 0 ? triangle_info_detached_.e2.z().data() : nullptr;
        params.tri_fn_x = suffix_samples > 0 ? triangle_info_detached_.face_normal.x().data() : nullptr;
        params.tri_fn_y = suffix_samples > 0 ? triangle_info_detached_.face_normal.y().data() : nullptr;
        params.tri_fn_z = suffix_samples > 0 ? triangle_info_detached_.face_normal.z().data() : nullptr;
        params.face_offsets = suffix_samples > 0 ? face_offsets_.data() : nullptr;
        params.n_meshes = mesh_count_;
        params.n_triangles = triangle_count;
        params.suffix_candidate_prim_id = nullptr;
        params.suffix_candidate_count = 0;
        params.material_eta_r = material.eta_r.data();
        params.material_sigma = material.sigma.data();
        params.material_mu_r = material.mu_r.data();
        params.material_gain = material.gain.data();
        params.material_valid = reinterpret_cast<const uint8_t *>(material.valid.data());
        params.material_count = material_count;
        params.wavelength = options.wavelength;
        params.k = options.k;
        params.seed = options.seed;
        params.samples = options.samples;
        params.max_order = options.max_order;
        params.direct_samples = direct_samples;
        params.keller_samples = keller_samples;
        params.suffix_samples = suffix_samples;
        params.strategy_mask = options.strategy_mask;
        params.sample_sequence = options.sample_sequence;
        params.receiver_model = options.receiver_model;
        params.collect_edge_use = options.collect_edge_use ? 1 : 0;
        params.collect_debug_counts = options.collect_debug_counts ? 1 : 0;
        params.out_power = raw.power.data();
        params.out_field_x_re = raw.field_x_re.data();
        params.out_field_x_im = raw.field_x_im.data();
        params.out_field_y_re = raw.field_y_re.data();
        params.out_field_y_im = raw.field_y_im.data();
        params.out_field_z_re = raw.field_z_re.data();
        params.out_field_z_im = raw.field_z_im.data();
        params.out_direct_count = raw.direct_count.data();
        params.out_keller_count = raw.keller_count.data();
        params.out_suffix_count = raw.suffix_count.data();
        params.out_vis_rejects = raw.vis_rejects.data();
        params.out_edge_vis_rejects =
            raw.edge_vis_rejects.data();
        params.out_utd_rejects = raw.utd_rejects.data();
        params.out_edge_uses = raw.edge_uses.data();
        if (active_dfr_direct_tape_capture != nullptr &&
            active_dfr_direct_tape_capture->launch_count == launch_count) {
            params.tape_active = reinterpret_cast<uint8_t *>(
                active_dfr_direct_tape_capture->active.data());
            params.tape_state_idx =
                active_dfr_direct_tape_capture->state_idx.data();
            params.tape_cell =
                active_dfr_direct_tape_capture->cell.data();
            params.tape_material_idx =
                active_dfr_direct_tape_capture->material_idx.data();
            params.tape_edge_u =
                active_dfr_direct_tape_capture->edge_u.data();
        }

        diffraction_accumulation_pipeline_->launch(1, params);

        result.power = raw.power;
        result.field_x =
            drjit::Complex<Float>(raw.field_x_re, raw.field_x_im);
        result.field_y =
            drjit::Complex<Float>(raw.field_y_re, raw.field_y_im);
        result.field_z =
            drjit::Complex<Float>(raw.field_z_re, raw.field_z_im);
        result.direct_count = raw.direct_count;
        result.keller_count = raw.keller_count;
        result.suffix_count = raw.suffix_count;
        result.vis_rejects = raw.vis_rejects;
        result.edge_vis_rejects =
            raw.edge_vis_rejects;
        result.utd_rejects = raw.utd_rejects;
        result.edge_uses = raw.edge_uses;
        return result;
    }
}

template <bool Detached>
MaskT<Detached> Scene::shadow_test(const RayT<Detached> &ray, MaskT<Detached> active) const {
    require(is_ready(), "Scene::shadow_test(): scene is not built.");
    require(!pending_updates_, "Scene::shadow_test(): scene has pending updates. Call Scene::sync() first.");

    const bool symbolic_optix_query = optix_split_active_ && uses_symbolic_optix_query_path();
    if (!optix_split_active_ || symbolic_optix_query) {
        return optix_scene_->template shadow_test<Detached>(ray, active);
    }

    const MaskT<Detached> static_hit =
        optix_static_scene_->template shadow_test<Detached>(ray, active);
    const MaskT<Detached> dynamic_hit =
        optix_dynamic_scene_->template shadow_test<Detached>(ray, active);
    return static_hit || dynamic_hit;
}

template <bool Detached>
SegmentVisibilityT<Detached> Scene::visible(
    const Vector3fT<Detached> &start,
    const Vector3fT<Detached> &end,
    const Int &ignore_prim_ids,
    MaskT<Detached> active) const {
    require(is_ready(), "Scene::visible(): scene is not built.");
    require(!pending_updates_,
            "Scene::visible(): scene has pending updates. Call Scene::sync() first.");

    const int ray_count = static_cast<int>(slices(start));
    require(static_cast<int>(slices(end)) == ray_count,
            "Scene::visible(): start and end must have the same width.");

    if (ray_count == 0) {
        SegmentVisibilityT<Detached> result;
        result.ray_count = ray_count;
        result.visible = full<MaskT<Detached>>(false, ray_count);
        return result;
    }

    const int ignore_count = static_cast<int>(slices(ignore_prim_ids));
    int ignore_k = 0;
    if (ignore_count > 0) {
        require(ignore_count % ray_count == 0,
                "Scene::visible(): ignore_prim_ids width must be a multiple of ray count.");
        ignore_k = ignore_count / ray_count;
    }

    const bool use_jit_visibility = use_jit_trace_visibility_path(ignore_k);
    if (!use_jit_visibility) {
        const SegmentPairVisibilityT<Detached> pair =
            visible_pair<Detached>(start, end, end, ignore_prim_ids, active);
        SegmentVisibilityT<Detached> result;
        result.ray_count = pair.ray_count;
        result.visible = pair.visible_a;
        return result;
    }

    const Mask active_detached = sanitize_segment_active<Detached>(start, end, active);
    Vector3f start_detached;
    Vector3f end_detached;
    if constexpr (!Detached) {
        start_detached = detach<false>(start);
        end_detached = detach<false>(end);
    } else {
        start_detached = start;
        end_detached = end;
    }

    return trace_segment_visibility_jit_no_ignore<Detached>(
        *optix_scene_, start_detached, end_detached, active_detached);
}

template <bool Detached>
SegmentPairVisibilityT<Detached> Scene::visible_pair(
    const Vector3fT<Detached> &start,
    const Vector3fT<Detached> &end_a,
    const Vector3fT<Detached> &end_b,
    const Int &ignore_prim_ids,
    MaskT<Detached> active) const {
    require(is_ready(), "Scene::visible_pair(): scene is not built.");
    require(!pending_updates_,
            "Scene::visible_pair(): scene has pending updates. Call Scene::sync() first.");

    const int ray_count = static_cast<int>(slices(start));
    require(static_cast<int>(slices(end_a)) == ray_count &&
                static_cast<int>(slices(end_b)) == ray_count,
            "Scene::visible_pair(): start, end_a, and end_b must have the same width.");

    SegmentPairVisibilityT<Detached> result;
    result.ray_count = ray_count;
    result.visible_a = full<MaskT<Detached>>(false, ray_count);
    result.visible_b = full<MaskT<Detached>>(false, ray_count);
    if (ray_count == 0) {
        return result;
    }

    const int ignore_count = static_cast<int>(slices(ignore_prim_ids));
    int ignore_k = 0;
    if (ignore_count > 0) {
        require(ignore_count % ray_count == 0,
                "Scene::visible_pair(): ignore_prim_ids width must be a multiple of ray count.");
        ignore_k = ignore_count / ray_count;
    }

    const Mask active_detached =
        sanitize_segment_active<Detached>(start, end_a, active) &&
        sanitize_segment_active<Detached>(start, end_b, active);
    Vector3f start_detached;
    Vector3f end_a_detached;
    Vector3f end_b_detached;
    if constexpr (!Detached) {
        start_detached = detach<false>(start);
        end_a_detached = detach<false>(end_a);
        end_b_detached = detach<false>(end_b);
    } else {
        start_detached = start;
        end_a_detached = end_a;
        end_b_detached = end_b;
    }

    if (use_jit_trace_visibility_path(ignore_k)) {
        return trace_segment_pair_visibility_jit_no_ignore<Detached>(
            *optix_scene_,
            start_detached,
            end_a_detached,
            end_b_detached,
            active_detached);
    }

    eval_segment_visibility_common(
        start_detached, face_offsets_, ignore_prim_ids, ignore_k, active_detached);
    drjit::eval(end_a_detached, end_b_detached);

    ensure_pipeline(segment_pair_visibility_pipeline_, optix_scene_->context(),
                    mesh_count_, segment_pair_visibility_pipeline_config());
    return trace_segment_pair_visibility_native<Detached>(
        *optix_scene_,
        *segment_pair_visibility_pipeline_,
        face_offsets_,
        mesh_count_,
        start_detached,
        end_a_detached,
        end_b_detached,
        ignore_prim_ids,
        ignore_k,
        active_detached);
}

template <bool Detached>
AxialEdgeVisibilityT<Detached> Scene::visible_edge(
    const Vector3fT<Detached> &src,
    const Vector3fT<Detached> &edge_pos,
    const Vector3fT<Detached> &edge_dir,
    const FloatT<Detached> &edge_t_min,
    const FloatT<Detached> &edge_t_max,
    const std::vector<float> &sample_fractions,
    MaskT<Detached> active) const {
    require(!sample_fractions.empty(),
            "Scene::visible_edge(): sample_fractions must not be empty.");
    require(sample_fractions.size() <= SegmentVisibilityMaxSamples,
            "Scene::visible_edge(): at most 16 sample fractions are supported.");
    require(is_ready(), "Scene::visible_edge(): scene is not built.");
    require(!pending_updates_,
            "Scene::visible_edge(): scene has pending updates. Call Scene::sync() first.");

    const int state_count = static_cast<int>(slices(src));
    require(static_cast<int>(slices(edge_pos)) == state_count &&
                static_cast<int>(slices(edge_dir)) == state_count &&
                static_cast<int>(slices(edge_t_min)) == state_count &&
                static_cast<int>(slices(edge_t_max)) == state_count,
            "Scene::visible_edge(): all inputs must have the same width.");

    AxialEdgeVisibilityT<Detached> result;
    result.state_count = state_count;
    result.any_visible = full<MaskT<Detached>>(false, state_count);
    if (state_count == 0) {
        return result;
    }

    Mask active_detached;
    Vector3f source_detached;
    Vector3f edge_pos_detached;
    Vector3f edge_dir_detached;
    Float edge_t_min_detached;
    Float edge_t_max_detached;
    if constexpr (!Detached) {
        active_detached = detach<false>(active);
        source_detached = detach<false>(src);
        edge_pos_detached = detach<false>(edge_pos);
        edge_dir_detached = detach<false>(edge_dir);
        edge_t_min_detached = detach<false>(edge_t_min);
        edge_t_max_detached = detach<false>(edge_t_max);
    } else {
        active_detached = active;
        source_detached = src;
        edge_pos_detached = edge_pos;
        edge_dir_detached = edge_dir;
        edge_t_min_detached = edge_t_min;
        edge_t_max_detached = edge_t_max;
    }

    active_detached &= drjit::isfinite(source_detached.x()) &&
                       drjit::isfinite(source_detached.y()) &&
                       drjit::isfinite(source_detached.z()) &&
                       drjit::isfinite(edge_pos_detached.x()) &&
                       drjit::isfinite(edge_pos_detached.y()) &&
                       drjit::isfinite(edge_pos_detached.z()) &&
                       drjit::isfinite(edge_dir_detached.x()) &&
                       drjit::isfinite(edge_dir_detached.y()) &&
                       drjit::isfinite(edge_dir_detached.z()) &&
                       drjit::isfinite(edge_t_min_detached) &&
                       drjit::isfinite(edge_t_max_detached);

    if (active_trace_visibility_backend() != TraceVisibilityBackend::Native) {
        return trace_axial_edge_visibility_jit<Detached>(
            *optix_scene_,
            source_detached,
            edge_pos_detached,
            edge_dir_detached,
            edge_t_min_detached,
            edge_t_max_detached,
            sample_fractions,
            active_detached);
    }

    ensure_pipeline(axial_edge_visibility_pipeline_, optix_scene_->context(),
                    mesh_count_, axial_edge_visibility_pipeline_config());
    return trace_axial_edge_visibility_native<Detached>(
        *optix_scene_,
        *axial_edge_visibility_pipeline_,
        face_offsets_,
        mesh_count_,
        source_detached,
        edge_pos_detached,
        edge_dir_detached,
        edge_t_min_detached,
        edge_t_max_detached,
        sample_fractions,
        active_detached);
}

template <bool Detached>
SegmentChainVisibilityT<Detached> Scene::visible_chain(
    const Vector3fT<Detached> &points,
    const Int &chain_length,
    const Int &ignore_prim_per_segment,
    MaskT<Detached> active) const {
    require(is_ready(), "Scene::visible_chain(): scene is not built.");
    require(!pending_updates_,
            "Scene::visible_chain(): scene has pending updates. Call Scene::sync() first.");

    const int chain_count = static_cast<int>(slices(chain_length));
    const int point_count = static_cast<int>(slices(points));

    SegmentChainVisibilityT<Detached> result;
    result.chain_count = chain_count;
    result.max_segments = 0;
    result.all_visible = full<MaskT<Detached>>(false, chain_count);
    result.first_blocked_segment = full<IntT<Detached>>(-1, chain_count);
    result.first_blocked_prim = full<IntT<Detached>>(-1, chain_count);
    if (chain_count == 0) {
        return result;
    }

    require(point_count % chain_count == 0,
            "Scene::visible_chain(): points width must be a multiple of chain count.");
    const int max_points = point_count / chain_count;
    require(max_points >= 2,
            "Scene::visible_chain(): each chain must contain at least two points.");
    const int max_segments = max_points - 1;
    result.max_segments = max_segments;

    const int ignore_count = static_cast<int>(slices(ignore_prim_per_segment));
    int ignore_k = 0;
    if (ignore_count > 0) {
        const int ignore_slots = chain_count * max_segments;
        require(ignore_count % ignore_slots == 0,
                "Scene::visible_chain(): ignore_prim_per_segment width must be a multiple of chain_count * max_segments.");
        ignore_k = ignore_count / ignore_slots;
    }

    Mask active_detached;
    Vector3f points_detached;
    if constexpr (!Detached) {
        active_detached = detach<false>(active);
        points_detached = detach<false>(points);
    } else {
        active_detached = active;
        points_detached = points;
    }
    active_detached &= chain_length >= 0;

    if (use_jit_trace_visibility_path(ignore_k)) {
        return trace_segment_chain_visibility_jit_no_ignore<Detached>(
            *optix_scene_,
            points_detached,
            chain_length,
            chain_count,
            max_points,
            max_segments,
            active_detached);
    }

    eval_segment_visibility_common(
        points_detached, face_offsets_, ignore_prim_per_segment, ignore_k, active_detached);
    drjit::eval(chain_length);

    ensure_pipeline(segment_chain_visibility_pipeline_, optix_scene_->context(),
                    mesh_count_, segment_chain_visibility_pipeline_config());
    return trace_segment_chain_visibility_native<Detached>(
        *optix_scene_,
        *segment_chain_visibility_pipeline_,
        face_offsets_,
        mesh_count_,
        points_detached,
        chain_length,
        ignore_prim_per_segment,
        ignore_k,
        chain_count,
        max_points,
        max_segments,
        active_detached);
}

template <bool Detached>
NearestPointEdgeT<Detached> Scene::nearest_edge(const Vector3fT<Detached> &point, MaskT<Detached> active) const {
    require(is_ready(), "Scene::nearest_edge(point): scene is not built.");
    require(!pending_updates_, "Scene::nearest_edge(point): scene has pending updates. Call Scene::sync() first.");

    const int query_count = static_cast<int>(slices(point));
    NearestPointEdgeT<Detached> result = initialize_nearest_point_edge_result<Detached>(query_count);
    if (edge_count_ == 0) {
        return result;
    }

    ensure_scene_edge_data_ready();

    Mask active_detached;
    if constexpr (!Detached) {
        active_detached = detach<false>(active);
        active_detached &= drjit::isfinite(detach<false>(point.x()));
        active_detached &= drjit::isfinite(detach<false>(point.y()));
        active_detached &= drjit::isfinite(detach<false>(point.z()));
        active &= MaskAD(active_detached);
    } else {
        active_detached = active;
        active_detached &= drjit::isfinite(point.x()) && drjit::isfinite(point.y()) && drjit::isfinite(point.z());
        active = active_detached;
    }

    if (drjit::none(active_detached)) {
        return result;
    }

    MaskT<Detached> query_mask = active;
    const bool use_optix_candidate = edge_backend_uses_optix_point(edge_bvh_backend_);
    ClosestEdgeCandidate candidate =
        use_optix_candidate
            ? edge_optix_->template nearest_edge<Detached>(point, query_mask)
            : edge_bvh_->template nearest_edge<Detached>(point, query_mask);
    const Mask valid_detached = detach<false>(query_mask) && (candidate.global_edge_id >= 0);
    if (drjit::none(valid_detached)) {
        return result;
    }

    const Int global_edge_id_detached =
        use_optix_candidate
            ? candidate.global_edge_id
            : edge_bvh_->map_to_global(candidate.global_edge_id, valid_detached);
    const Int shape_id_detached =
        gather<Int>(edge_shape_ids_, global_edge_id_detached, valid_detached);
    const Int edge_id_detached =
        gather<Int>(edge_local_ids_, global_edge_id_detached, valid_detached);

    if constexpr (!Detached) {
        const MaskAD valid = MaskAD(valid_detached);
        const IntAD global_edge_id = IntAD(global_edge_id_detached);
        const Vector3fAD p0 = gather<Vector3fAD>(edge_info_.start, global_edge_id, valid);
        const Vector3fAD e1 = gather<Vector3fAD>(edge_info_.edge, global_edge_id, valid);
        const MaskAD is_boundary = gather<MaskAD>(edge_info_.is_boundary, global_edge_id, valid);

        FloatAD edge_t;
        Vector3fAD edge_point;
        FloatAD distance_sq;
        std::tie(edge_t, edge_point, distance_sq) = closest_point_on_segment<false>(point, p0, e1);

        result.distance = select(valid, sqrt(distance_sq), result.distance);
        result.point = select(valid, point, result.point);
        result.edge_t = select(valid, edge_t, result.edge_t);
        result.edge_point = select(valid, edge_point, result.edge_point);
        result.shape_id = select(valid, IntAD(shape_id_detached), result.shape_id);
        result.edge_id = select(valid, IntAD(edge_id_detached), result.edge_id);
        result.global_edge_id = select(valid, global_edge_id, result.global_edge_id);
        result.is_boundary = select(valid, is_boundary, result.is_boundary);
    } else {
        const Vector3f p0 =
            gather<Vector3f>(detach<false>(edge_info_.start), global_edge_id_detached, valid_detached);
        const Vector3f e1 =
            gather<Vector3f>(detach<false>(edge_info_.edge), global_edge_id_detached, valid_detached);
        const Mask is_boundary =
            gather<Mask>(detach<false>(edge_info_.is_boundary), global_edge_id_detached, valid_detached);

        Float edge_t;
        Vector3f edge_point;
        Float distance_sq;
        std::tie(edge_t, edge_point, distance_sq) = closest_point_on_segment<true>(point, p0, e1);

        result.distance = select(valid_detached, sqrt(distance_sq), result.distance);
        result.point = select(valid_detached, point, result.point);
        result.edge_t = select(valid_detached, edge_t, result.edge_t);
        result.edge_point = select(valid_detached, edge_point, result.edge_point);
        result.shape_id = select(valid_detached, shape_id_detached, result.shape_id);
        result.edge_id = select(valid_detached, edge_id_detached, result.edge_id);
        result.global_edge_id = select(valid_detached, global_edge_id_detached, result.global_edge_id);
        result.is_boundary = select(valid_detached, is_boundary, result.is_boundary);
    }

    return result;
}

template <bool Detached>
NearestRayEdgeT<Detached> Scene::nearest_edge(const RayT<Detached> &ray, MaskT<Detached> active) const {
    require(is_ready(), "Scene::nearest_edge(ray): scene is not built.");
    require(!pending_updates_, "Scene::nearest_edge(ray): scene has pending updates. Call Scene::sync() first.");

    const int query_count = static_cast<int>(slices(ray.o));
    NearestRayEdgeT<Detached> result = initialize_nearest_ray_edge_result<Detached>(query_count);
    if (edge_count_ == 0) {
        return result;
    }

    ensure_scene_edge_data_ready();

    Float t_max_input;
    Mask active_detached;
    if constexpr (!Detached) {
        t_max_input = detach<false>(ray.tmax);
        active_detached = detach<false>(active);
        active_detached &= drjit::isfinite(detach<false>(ray.o.x())) &&
                           drjit::isfinite(detach<false>(ray.o.y())) &&
                           drjit::isfinite(detach<false>(ray.o.z()));
        active_detached &= drjit::isfinite(detach<false>(ray.d.x())) &&
                           drjit::isfinite(detach<false>(ray.d.y())) &&
                           drjit::isfinite(detach<false>(ray.d.z()));
        active_detached &= squared_norm(Vector3f(detach<false>(ray.d.x()),
                                                        detach<false>(ray.d.y()),
                                                        detach<false>(ray.d.z()))) > 0.f;
        active_detached &= ~drjit::isfinite(t_max_input) || (t_max_input > 0.f);
        active &= MaskAD(active_detached);
    } else {
        t_max_input = ray.tmax;
        active_detached = active;
        active_detached &= drjit::isfinite(ray.o.x()) && drjit::isfinite(ray.o.y()) && drjit::isfinite(ray.o.z());
        active_detached &= drjit::isfinite(ray.d.x()) && drjit::isfinite(ray.d.y()) && drjit::isfinite(ray.d.z());
        active_detached &= squared_norm(ray.d) > 0.f;
        active_detached &= ~drjit::isfinite(t_max_input) || (t_max_input > 0.f);
        active = active_detached;
    }

    if (drjit::none(active_detached)) {
        return result;
    }

    MaskT<Detached> query_mask = active;
    const bool use_optix_candidate = edge_backend_uses_optix_ray(edge_bvh_backend_);
    ClosestEdgeCandidate candidate =
        use_optix_candidate
            ? edge_optix_->template nearest_edge<Detached>(ray, query_mask)
            : edge_bvh_->template nearest_edge<Detached>(ray, query_mask);
    const Mask valid_detached = detach<false>(query_mask) && (candidate.global_edge_id >= 0);
    if (drjit::none(valid_detached)) {
        return result;
    }

    const Mask finite_tmax = drjit::isfinite(t_max_input);
    const Int global_edge_id_detached =
        use_optix_candidate
            ? candidate.global_edge_id
            : edge_bvh_->map_to_global(candidate.global_edge_id, valid_detached);
    const Int shape_id_detached =
        gather<Int>(edge_shape_ids_, global_edge_id_detached, valid_detached);
    const Int edge_id_detached =
        gather<Int>(edge_local_ids_, global_edge_id_detached, valid_detached);

    if constexpr (!Detached) {
        const MaskAD valid = MaskAD(valid_detached);
        const IntAD global_edge_id = IntAD(global_edge_id_detached);
        const Vector3fAD p0 = gather<Vector3fAD>(edge_info_.start, global_edge_id, valid);
        const Vector3fAD e1 = gather<Vector3fAD>(edge_info_.edge, global_edge_id, valid);
        const MaskAD is_boundary = gather<MaskAD>(edge_info_.is_boundary, global_edge_id, valid);

        const MaskAD finite_mask = valid && MaskAD(finite_tmax);
        const MaskAD infinite_mask = valid && !MaskAD(finite_tmax);
        const FloatAD safe_tmax = select(finite_mask, FloatAD(t_max_input), zeros<FloatAD>(query_count));

        FloatAD query_t = zeros<FloatAD>(query_count);
        Vector3fAD query_point = zeros<Vector3fAD>(query_count);
        FloatAD edge_t = zeros<FloatAD>(query_count);
        Vector3fAD edge_point = zeros<Vector3fAD>(query_count);
        FloatAD distance_sq = full<FloatAD>(Infinity, query_count);

        if (drjit::any(finite_mask)) {
            FloatAD segment_query_t;
            Vector3fAD segment_query_point;
            FloatAD segment_edge_t;
            Vector3fAD segment_edge_point;
            FloatAD segment_distance_sq;
            std::tie(segment_query_t, segment_query_point, segment_edge_t, segment_edge_point, segment_distance_sq) =
                closest_segment_segment<false>(ray.o, ray.d * safe_tmax, p0, e1);

            query_t = select(finite_mask, segment_query_t * safe_tmax, query_t);
            query_point = select(finite_mask, segment_query_point, query_point);
            edge_t = select(finite_mask, segment_edge_t, edge_t);
            edge_point = select(finite_mask, segment_edge_point, edge_point);
            distance_sq = select(finite_mask, segment_distance_sq, distance_sq);
        }

        if (drjit::any(infinite_mask)) {
            FloatAD ray_query_t;
            Vector3fAD ray_query_point;
            FloatAD ray_edge_t;
            Vector3fAD ray_edge_point;
            FloatAD ray_distance_sq;
            std::tie(ray_query_t, ray_query_point, ray_edge_t, ray_edge_point, ray_distance_sq) =
                closest_ray_segment<false>(ray.o, ray.d, p0, e1);

            query_t = select(infinite_mask, ray_query_t, query_t);
            query_point = select(infinite_mask, ray_query_point, query_point);
            edge_t = select(infinite_mask, ray_edge_t, edge_t);
            edge_point = select(infinite_mask, ray_edge_point, edge_point);
            distance_sq = select(infinite_mask, ray_distance_sq, distance_sq);
        }

        result.distance = select(valid, sqrt(distance_sq), result.distance);
        result.ray_t = select(valid, query_t, result.ray_t);
        result.point = select(valid, query_point, result.point);
        result.edge_t = select(valid, edge_t, result.edge_t);
        result.edge_point = select(valid, edge_point, result.edge_point);
        result.shape_id = select(valid, IntAD(shape_id_detached), result.shape_id);
        result.edge_id = select(valid, IntAD(edge_id_detached), result.edge_id);
        result.global_edge_id = select(valid, global_edge_id, result.global_edge_id);
        result.is_boundary = select(valid, is_boundary, result.is_boundary);
    } else {
        const Vector3f p0 =
            gather<Vector3f>(detach<false>(edge_info_.start), global_edge_id_detached, valid_detached);
        const Vector3f e1 =
            gather<Vector3f>(detach<false>(edge_info_.edge), global_edge_id_detached, valid_detached);
        const Mask is_boundary =
            gather<Mask>(detach<false>(edge_info_.is_boundary), global_edge_id_detached, valid_detached);

        const Mask finite_mask = valid_detached && finite_tmax;
        const Mask infinite_mask = valid_detached && !finite_tmax;
        const Float safe_tmax = select(finite_mask, t_max_input, zeros<Float>(query_count));

        Float query_t = zeros<Float>(query_count);
        Vector3f query_point = zeros<Vector3f>(query_count);
        Float edge_t = zeros<Float>(query_count);
        Vector3f edge_point = zeros<Vector3f>(query_count);
        Float distance_sq = full<Float>(Infinity, query_count);

        if (drjit::any(finite_mask)) {
            Float segment_query_t;
            Vector3f segment_query_point;
            Float segment_edge_t;
            Vector3f segment_edge_point;
            Float segment_distance_sq;
            std::tie(segment_query_t, segment_query_point, segment_edge_t, segment_edge_point, segment_distance_sq) =
                closest_segment_segment<true>(ray.o, ray.d * safe_tmax, p0, e1);

            query_t = select(finite_mask, segment_query_t * safe_tmax, query_t);
            query_point = select(finite_mask, segment_query_point, query_point);
            edge_t = select(finite_mask, segment_edge_t, edge_t);
            edge_point = select(finite_mask, segment_edge_point, edge_point);
            distance_sq = select(finite_mask, segment_distance_sq, distance_sq);
        }

        if (drjit::any(infinite_mask)) {
            Float ray_query_t;
            Vector3f ray_query_point;
            Float ray_edge_t;
            Vector3f ray_edge_point;
            Float ray_distance_sq;
            std::tie(ray_query_t, ray_query_point, ray_edge_t, ray_edge_point, ray_distance_sq) =
                closest_ray_segment<true>(ray.o, ray.d, p0, e1);

            query_t = select(infinite_mask, ray_query_t, query_t);
            query_point = select(infinite_mask, ray_query_point, query_point);
            edge_t = select(infinite_mask, ray_edge_t, edge_t);
            edge_point = select(infinite_mask, ray_edge_point, edge_point);
            distance_sq = select(infinite_mask, ray_distance_sq, distance_sq);
        }

        result.distance = select(valid_detached, sqrt(distance_sq), result.distance);
        result.ray_t = select(valid_detached, query_t, result.ray_t);
        result.point = select(valid_detached, query_point, result.point);
        result.edge_t = select(valid_detached, edge_t, result.edge_t);
        result.edge_point = select(valid_detached, edge_point, result.edge_point);
        result.shape_id = select(valid_detached, shape_id_detached, result.shape_id);
        result.edge_id = select(valid_detached, edge_id_detached, result.edge_id);
        result.global_edge_id = select(valid_detached, global_edge_id_detached, result.global_edge_id);
        result.is_boundary = select(valid_detached, is_boundary, result.is_boundary);
    }

    return result;
}

template <bool Detached>
NearestEdgesTopKT<Detached> Scene::nearest_edges(const Vector3fT<Detached> &point,
                                                       int k,
                                                       MaskT<Detached> active) const {
    require(is_ready(), "Scene::nearest_edges(point): scene is not built.");
    require(!pending_updates_,
            "Scene::nearest_edges(point): scene has pending updates. Call Scene::sync() first.");
    require(k > 0, "Scene::nearest_edges(point): k must be positive.");
    require(k <= 16, "Scene::nearest_edges(point): k must be <= 16.");

    const int query_count = static_cast<int>(slices(point));
    const int output_count = query_count * k;
    NearestEdgesTopKT<Detached> result;
    result.query_count = query_count;
    result.k = k;
    result.is_valid = full<MaskT<Detached>>(false, output_count);
    result.distances = full<FloatT<Detached>>(Infinity, output_count);
    result.points = zeros<Vector3fT<Detached>>(output_count);
    result.edge_t = zeros<FloatT<Detached>>(output_count);
    result.edge_points = zeros<Vector3fT<Detached>>(output_count);
    result.shape_ids = full<IntT<Detached>>(-1, output_count);
    result.edge_ids = full<IntT<Detached>>(-1, output_count);
    result.global_edge_ids = full<IntT<Detached>>(-1, output_count);
    result.is_boundary = full<MaskT<Detached>>(false, output_count);
    if (query_count == 0 || edge_count_ == 0) {
        return result;
    }

    ensure_scene_edge_data_ready();

    Mask active_detached;
    if constexpr (!Detached) {
        active_detached = detach<false>(active);
        active_detached &= drjit::isfinite(detach<false>(point.x())) &&
                           drjit::isfinite(detach<false>(point.y())) &&
                           drjit::isfinite(detach<false>(point.z()));
        active &= MaskAD(active_detached);
    } else {
        active_detached = active;
        active_detached &= drjit::isfinite(point.x()) &&
                           drjit::isfinite(point.y()) &&
                           drjit::isfinite(point.z());
        active = active_detached;
    }

    if (drjit::none(active_detached)) {
        return result;
    }

    MaskT<Detached> query_mask = active;
    const bool use_optix_candidate = edge_backend_uses_optix_topk(edge_bvh_backend_);
    const ClosestEdgeTopKCandidate candidate =
        use_optix_candidate
            ? edge_optix_->template nearest_edges<Detached>(point, k, query_mask)
            : edge_bvh_->template nearest_edges<Detached>(point, k, query_mask);
    const Mask valid_detached = candidate.is_valid;
    if (drjit::none(valid_detached)) {
        return result;
    }

    const Int output_index = arange<Int>(output_count);
    const Int output_query = output_index / k;
    const Int global_edge_id_detached =
        use_optix_candidate
            ? candidate.global_edge_ids
            : edge_bvh_->map_to_global(candidate.global_edge_ids, valid_detached);
    const Int shape_id_detached =
        gather<Int>(edge_shape_ids_, global_edge_id_detached, valid_detached);
    const Int edge_id_detached =
        gather<Int>(edge_local_ids_, global_edge_id_detached, valid_detached);

    if constexpr (!Detached) {
        const MaskAD valid = MaskAD(valid_detached);
        const IntAD global_edge_id = IntAD(global_edge_id_detached);
        const IntAD query_id = IntAD(output_query);
        const Vector3fAD output_point = gather<Vector3fAD>(point, query_id, valid);
        const Vector3fAD edge_start =
            gather<Vector3fAD>(edge_info_.start, global_edge_id, valid);
        const Vector3fAD edge_vector =
            gather<Vector3fAD>(edge_info_.edge, global_edge_id, valid);
        const MaskAD boundary =
            gather<MaskAD>(edge_info_.is_boundary, global_edge_id, valid);

        FloatAD edge_t;
        Vector3fAD edge_point;
        FloatAD distance_sq;
        std::tie(edge_t, edge_point, distance_sq) =
            closest_point_on_segment<false>(output_point, edge_start, edge_vector);

        result.is_valid = valid;
        result.distances = select(valid, sqrt(distance_sq), result.distances);
        result.points = select(valid, output_point, result.points);
        result.edge_t = select(valid, edge_t, result.edge_t);
        result.edge_points = select(valid, edge_point, result.edge_points);
        result.shape_ids = select(valid, IntAD(shape_id_detached), result.shape_ids);
        result.edge_ids = select(valid, IntAD(edge_id_detached), result.edge_ids);
        result.global_edge_ids = select(valid, global_edge_id, result.global_edge_ids);
        result.is_boundary = select(valid, boundary, result.is_boundary);
    } else {
        const Vector3f output_point =
            gather<Vector3f>(point, output_query, valid_detached);
        const Vector3f edge_start =
            gather<Vector3f>(detach<false>(edge_info_.start), global_edge_id_detached, valid_detached);
        const Vector3f edge_vector =
            gather<Vector3f>(detach<false>(edge_info_.edge), global_edge_id_detached, valid_detached);
        const Mask boundary =
            gather<Mask>(detach<false>(edge_info_.is_boundary), global_edge_id_detached, valid_detached);

        Float edge_t;
        Vector3f edge_point;
        Float distance_sq;
        std::tie(edge_t, edge_point, distance_sq) =
            closest_point_on_segment<true>(output_point, edge_start, edge_vector);

        result.is_valid = valid_detached;
        result.distances = select(valid_detached, sqrt(distance_sq), result.distances);
        result.points = select(valid_detached, output_point, result.points);
        result.edge_t = select(valid_detached, edge_t, result.edge_t);
        result.edge_points = select(valid_detached, edge_point, result.edge_points);
        result.shape_ids = select(valid_detached, shape_id_detached, result.shape_ids);
        result.edge_ids = select(valid_detached, edge_id_detached, result.edge_ids);
        result.global_edge_ids = select(valid_detached, global_edge_id_detached, result.global_edge_ids);
        result.is_boundary = select(valid_detached, boundary, result.is_boundary);
    }
    return result;
}

template Intersection Scene::intersect<true>(const Ray &ray, Mask active, RayFlags flags) const;
template IntersectionAD Scene::intersect<false>(const RayAD &ray, MaskAD active, RayFlags flags) const;
template ReflectionChain Scene::trace_reflections<true>(const Ray &ray,
                                                                int max_bounces,
                                                                const ReflectionTraceOptions &options,
                                                                Mask active) const;
template ReflectionChainAD Scene::trace_reflections<false>(const RayAD &ray,
                                                         int max_bounces,
                                                         const ReflectionTraceOptions &options,
                                                         MaskAD active) const;
template ReflectionChain Scene::trace_reflections<true>(const Ray &ray,
                                                                int max_bounces,
                                                                Mask active) const;
template ReflectionChainAD Scene::trace_reflections<false>(const RayAD &ray,
                                                         int max_bounces,
                                                         MaskAD active) const;
template AccumResult Scene::accumulate_reflections<true>(
    const Ray &ray,
    const Vector3f &tx_position,
    const AccumGrid &grid,
    const Material &material,
    int max_bounces,
    const AccumOptions &options,
    Mask active,
    const Vector3f &tx_polarization) const;
template AccumResultAD Scene::accumulate_reflections<false>(
    const RayAD &ray,
    const Vector3fAD &tx_position,
    const AccumGrid &grid,
    const MaterialAD &material,
    int max_bounces,
    const AccumOptions &options,
    MaskAD active,
    const Vector3fAD &tx_polarization) const;
template DfrAccum Scene::accum_dfr_direct<true>(
    const DfrStates &states,
    const DfrGrid &grid,
    const DfrMaterial &material,
    const DfrOptions &options,
    Mask active) const;
template DfrAccumAD Scene::accum_dfr_direct<false>(
    const DfrStatesAD &states,
    const DfrGrid &grid,
    const DfrMaterialAD &material,
    const DfrOptions &options,
    MaskAD active) const;
template DfrAccum Scene::accum_dfr<true>(
    const DfrStates &initial_states,
    const DfrStates &recursive_states,
    const DfrGrid &grid,
    const DfrMaterial &material,
    const DfrOptions &options,
    Mask active) const;
template DfrAccumAD Scene::accum_dfr<false>(
    const DfrStatesAD &initial_states,
    const DfrStatesAD &recursive_states,
    const DfrGrid &grid,
    const DfrMaterialAD &material,
    const DfrOptions &options,
    MaskAD active) const;
template DfrPaths Scene::trace_dfr_paths<true>(
    const Vector3f &tx_positions,
    const Vector3f &rx_positions,
    const DfrStates &states,
    const DfrMaterial &material,
    const DfrPathOptions &options,
    Mask active) const;
template DfrPathsAD Scene::trace_dfr_paths<false>(
    const Vector3fAD &tx_positions,
    const Vector3fAD &rx_positions,
    const DfrStatesAD &states,
    const DfrMaterialAD &material,
    const DfrPathOptions &options,
    MaskAD active) const;
template ReflEpc Scene::trace_refl_epc<true>(
    const Ray &ray,
    const Vector3f &receiver,
    int max_bounces,
    const ReflEpcOptions &options,
    Mask active) const;
template ReflEpcAD Scene::trace_refl_epc<false>(
    const RayAD &ray,
    const Vector3fAD &receiver,
    int max_bounces,
    const ReflEpcOptions &options,
    MaskAD active) const;
template ReflEpcField Scene::trace_refl_epc_field<true>(
    const Ray &ray,
    const Vector3f &receiver,
    int max_bounces,
    const ReflEpcFieldOptions &options,
    Mask active) const;
template ReflEpcFieldAD Scene::trace_refl_epc_field<false>(
    const RayAD &ray,
    const Vector3fAD &receiver,
    int max_bounces,
    const ReflEpcFieldOptionsAD &options,
    MaskAD active) const;
template ReflEpcField Scene::trace_refl_epc_field<true>(
    const Vector3f &tx_position,
    const Vector3f &receiver,
    int max_bounces,
    const ReflEpcFieldOptions &options,
    Mask active) const;
template ReflEpcFieldAD Scene::trace_refl_epc_field<false>(
    const Vector3fAD &tx_position,
    const Vector3fAD &receiver,
    int max_bounces,
    const ReflEpcFieldOptionsAD &options,
    MaskAD active) const;
template ReflectionTrace Scene::trace_bounces<true>(
    const Ray &ray,
    int max_bounces,
    const ReflectionTraceOptions &options,
    Mask active) const;
template ReflectionTraceAD Scene::trace_bounces<false>(
    const RayAD &ray,
    int max_bounces,
    const ReflectionTraceOptions &options,
    MaskAD active) const;
template ReflectionTrace Scene::trace_bounces<true>(
    const Ray &ray,
    int max_bounces,
    Mask active) const;
template ReflectionTraceAD Scene::trace_bounces<false>(
    const RayAD &ray,
    int max_bounces,
    MaskAD active) const;
template Mask Scene::shadow_test<true>(const Ray &ray, Mask active) const;
template MaskAD Scene::shadow_test<false>(const RayAD &ray, MaskAD active) const;
template SegmentVisibility Scene::visible<true>(
    const Vector3f &start,
    const Vector3f &end,
    const Int &ignore_prim_ids,
    Mask active) const;
template SegmentVisibilityAD Scene::visible<false>(
    const Vector3fAD &start,
    const Vector3fAD &end,
    const Int &ignore_prim_ids,
    MaskAD active) const;
template SegmentPairVisibility Scene::visible_pair<true>(
    const Vector3f &start,
    const Vector3f &end_a,
    const Vector3f &end_b,
    const Int &ignore_prim_ids,
    Mask active) const;
template SegmentPairVisibilityAD Scene::visible_pair<false>(
    const Vector3fAD &start,
    const Vector3fAD &end_a,
    const Vector3fAD &end_b,
    const Int &ignore_prim_ids,
    MaskAD active) const;
template AxialEdgeVisibility Scene::visible_edge<true>(
    const Vector3f &src,
    const Vector3f &edge_pos,
    const Vector3f &edge_dir,
    const Float &edge_t_min,
    const Float &edge_t_max,
    const std::vector<float> &sample_fractions,
    Mask active) const;
template AxialEdgeVisibilityAD Scene::visible_edge<false>(
    const Vector3fAD &src,
    const Vector3fAD &edge_pos,
    const Vector3fAD &edge_dir,
    const FloatAD &edge_t_min,
    const FloatAD &edge_t_max,
    const std::vector<float> &sample_fractions,
    MaskAD active) const;
template SegmentChainVisibility Scene::visible_chain<true>(
    const Vector3f &points,
    const Int &chain_length,
    const Int &ignore_prim_per_segment,
    Mask active) const;
template SegmentChainVisibilityAD Scene::visible_chain<false>(
    const Vector3fAD &points,
    const Int &chain_length,
    const Int &ignore_prim_per_segment,
    MaskAD active) const;
template NearestPointEdge Scene::nearest_edge<true>(const Vector3f &point, Mask active) const;
template NearestPointEdgeAD Scene::nearest_edge<false>(const Vector3fAD &point, MaskAD active) const;
template NearestRayEdge Scene::nearest_edge<true>(const Ray &ray, Mask active) const;
template NearestRayEdgeAD Scene::nearest_edge<false>(const RayAD &ray, MaskAD active) const;
template NearestEdgesTopK Scene::nearest_edges<true>(
    const Vector3f &point,
    int k,
    Mask active) const;
template NearestEdgesTopKAD Scene::nearest_edges<false>(
    const Vector3fAD &point,
    int k,
    MaskAD active) const;

} // namespace rayd

