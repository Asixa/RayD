#include <utility>

#include <drjit/custom.h>
#include <nanobind/nanobind.h>

#include "scene_internal.h"
#include <rayd/multipath/diffraction_accumulation_ad.h>

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

thread_local DfrDirectTapeCapture *active_dfr_direct_tape_capture = nullptr;

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

void require_dfr_direct_custom_ad_supported_impl(const DfrOptions &options) {
    require(options.max_order == 1,
            "Scene::accum_dfr_direct(): native AD currently supports max_order == 1.");
}

void require_dfr_chain_custom_ad_supported_impl(const DfrOptions &options) {
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

DfrAccumAD dfr_direct_accum_custom_op_impl(const Scene *scene,
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

DfrAccumAD dfr_chain_accum_custom_op_impl(const Scene *scene,
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

} // namespace

void require_dfr_direct_custom_ad_supported(const DfrOptions &options) {
    require_dfr_direct_custom_ad_supported_impl(options);
}

void require_dfr_chain_custom_ad_supported(const DfrOptions &options) {
    require_dfr_chain_custom_ad_supported_impl(options);
}

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
    return dfr_direct_accum_custom_op_impl(scene,
                                          states,
                                          grid,
                                          material,
                                          options,
                                          suffix_tri_p0,
                                          suffix_tri_face_normal,
                                          suffix_vertices,
                                          suffix_faces,
                                          active);
}

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
    return dfr_chain_accum_custom_op_impl(scene,
                                         initial_states,
                                         recursive_states,
                                         grid,
                                         material,
                                         options,
                                         suffix_tri_p0,
                                         suffix_tri_face_normal,
                                         suffix_vertices,
                                         suffix_faces,
                                         active);
}

} // namespace rayd
