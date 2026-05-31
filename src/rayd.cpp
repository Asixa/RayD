#include <rayd/rayd.h>

#include <stdexcept>

#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <drjit/python.h>

#include <rayd/ray.h>
#include <rayd/transform.h>
#include <rayd/edge/edge.h>
#include <rayd/multipath/segment_visibility.h>
#include <rayd/mesh.h>
#include <rayd/optix.h>
#include <rayd/scene/scene.h>

#include <rayd/native_launch_audit.h>

namespace nb = nanobind;
using namespace nb::literals;
using namespace rayd;

namespace {

/// Number of Dr.Jit-compatible CUDA devices; throws if none are available.
int checked_cuda_device_count() {
    const int count = jit_cuda_device_count();
    if (count <= 0)
        throw std::runtime_error("No Dr.Jit-compatible CUDA devices are available.");
    return count;
}

/// Make \p device the active Dr.Jit CUDA device, optionally initializing OptiX on it.
int set_rayd_device(int device, bool initialize_optix) {
    const int count = checked_cuda_device_count();
    if (device < 0 || device >= count) {
        throw std::runtime_error(
            "set_device(): device index " + std::to_string(device) +
            " is out of range for " + std::to_string(count) +
            " Dr.Jit-compatible CUDA device(s).");
    }

    // RayD scenes, BVHs, and OptiX pipelines are bound to the current thread's
    // CUDA/OptiX state. Synchronize before switching and eagerly initialize the
    // target OptiX context so subsequent Scene::build() calls use that device.
    jit_sync_thread();
    drjit::set_device(device);

    if (initialize_optix) {
        jit_optix_context();
        init_optix_api();
    }

    return jit_cuda_device();
}

/// The Python type object for Dr.Jit array type \p T, bound once and cached.
template <typename T>
nb::object drjit_python_type() {
    static nb::object type = []() {
        drjit::ArrayBinding b;
        return drjit::bind_array<T>(b);
    }();
    return type;
}

/// Load a Dr.Jit array \p T from a Python handle, optionally coercing other types; returns success.
template <typename T>
bool drjit_try_load(nb::handle src, T &value, bool convert) {
    if (!src.is_valid())
        return false;

    if (nb::inst_check(src)) {
        try {
            if (nb::type_info(src.type()) == typeid(T)) {
                value = *nb::inst_ptr<T>(src);
                return true;
            }
        } catch (...) {
            PyErr_Clear();
        }
    }

    if (!convert)
        return false;

    try {
        nb::object converted = nb::steal<nb::object>(PyObject_CallOneArg(
            drjit_python_type<T>().ptr(), src.ptr()));
        if (!converted.is_valid()) {
            PyErr_Clear();
            return false;
        }
        value = *nb::inst_ptr<T>(converted);
        return true;
    } catch (...) {
        PyErr_Clear();
        return false;
    }
}

ReflEpcFieldOptionsAD refl_epc_field_options_ad_from_detached(
        const ReflEpcFieldOptions &options) {
    ReflEpcFieldOptionsAD out;
    out.expected_prim_ids = options.expected_prim_ids;
    out.surface_group_id = options.surface_group_id;
    out.surface_group_size = options.surface_group_size;
    out.surface_group_members = options.surface_group_members;
    out.surface_max_group_size = options.surface_max_group_size;
    out.visibility_ignore_mode = options.visibility_ignore_mode;
    out.final_ignore_group_ids = options.final_ignore_group_ids;
    out.slot_plane_point = Vector3fAD(options.slot_plane_point);
    out.slot_plane_normal = Vector3fAD(options.slot_plane_normal);
    out.slot_eta_r = FloatAD(options.slot_eta_r);
    out.slot_mu_r = FloatAD(options.slot_mu_r);
    out.slot_sigma = FloatAD(options.slot_sigma);
    out.slot_gain = FloatAD(options.slot_gain);
    out.tx_polarization = Vector3fAD(options.tx_polarization);
    out.omega = options.omega;
    out.wavelength = options.wavelength;
    out.return_geom = options.return_geom;
    out.return_endpoints = options.return_endpoints;
    out.return_hit_points = options.return_hit_points;
    out.return_normals = options.return_normals;
    out.return_resolved_prim_ids = options.return_resolved_prim_ids;
    out.return_surface_group_ids = options.return_surface_group_ids;
    return out;
}

/// Wrap a C++ Dr.Jit array \p value in a new Python object of the matching type.
template <typename T>
nb::handle drjit_from_cpp(const T &value) {
    nb::object obj = nb::steal<nb::object>(PyObject_CallNoArgs(drjit_python_type<T>().ptr()));
    if (!obj.is_valid())
        throw nb::python_error();
    *nb::inst_ptr<T>(obj) = value;
    return obj.release();
}

} // namespace

namespace nanobind::detail {

/// nanobind type caster bridging any Dr.Jit array type to/from its Python counterpart.
template <typename T>
struct type_caster<T, std::enable_if_t<drjit::is_array_v<T>, int>> {
    using Value = T;
    static constexpr auto Name = const_name<T>();
    template <typename U> using Cast = movable_cast_t<U>;
    template <typename U> static constexpr bool can_cast() { return true; }

    bool from_python(handle src, uint8_t flags, cleanup_list *) noexcept {
        return drjit_try_load<T>(src, value,
                                 (flags & (uint8_t) cast_flags::convert) != 0);
    }

    static handle from_cpp(const T &src, rv_policy, cleanup_list *) {
        return drjit_from_cpp(src);
    }

    static handle from_cpp(T *src, rv_policy policy, cleanup_list *cleanup) {
        if (!src)
            return none().release();
        return from_cpp(*src, policy, cleanup);
    }

    explicit operator T*() { return &value; }
    explicit operator T&() { return value; }
    explicit operator T&&() { return std::move(value); }

    Value value;
};

} // namespace nanobind::detail

NB_MODULE(rayd, m) {
    auto native_kernel_dict = [](const NativeKernelLaunchStat &entry) {
        nb::dict result;
        result["label"] = entry.label;
        result["launches"] = entry.launches;
        result["total_threads"] = entry.total_threads;
        result["max_threads"] = entry.max_threads;
        result["total_items"] = entry.total_items;
        result["max_items"] = entry.max_items;
        return result;
    };

    auto native_stage_dict = [&](const NativeLaunchStageStats &stats) {
        nb::list kernels;
        for (const NativeKernelLaunchStat &entry : stats.kernels) {
            kernels.append(native_kernel_dict(entry));
        }

        nb::dict result;
        result["cuda_kernel_launches"] = stats.cuda_kernel_launches;
        result["cuda_kernel_total_threads"] = stats.cuda_kernel_total_threads;
        result["cuda_memcpy"] = stats.cuda_memcpy;
        result["cuda_memcpy_async"] = stats.cuda_memcpy_async;
        result["cuda_memset_async"] = stats.cuda_memset_async;
        result["cuda_stream_synchronize"] = stats.cuda_stream_synchronize;
        result["cuda_event_record"] = stats.cuda_event_record;
        result["cuda_stream_wait_event"] = stats.cuda_stream_wait_event;
        result["cub_reduce"] = stats.cub_reduce;
        result["cub_sort"] = stats.cub_sort;
        result["cub_scan"] = stats.cub_scan;
        result["jit_memcpy"] = stats.jit_memcpy;
        result["jit_memcpy_async"] = stats.jit_memcpy_async;
        result["optix_accel_build"] = stats.optix_accel_build;
        result["optix_accel_compact"] = stats.optix_accel_compact;
        result["optix_launch"] = stats.optix_launch;
        result["optix_launch_time_ms"] = stats.optix_launch_time_ms;
        result["optix_launch_time_min_ms"] = stats.optix_launch_time_min_ms;
        result["optix_launch_time_max_ms"] = stats.optix_launch_time_max_ms;
        result["kernels"] = kernels;
        return result;
    };

    auto bind_section = [](const char *name, auto &&fn) {
        try {
            fn();
        } catch (const std::exception &e) {
            nb::raise("%s: %s", name, e.what());
        }
    };

    nb::module_::import_("drjit");
    nb::module_::import_("drjit.cuda");
    nb::module_::import_("drjit.cuda.ad");
    {
        drjit::ArrayBinding b;
        drjit::bind_all<drjit::CUDAArray<float>>(b);
        drjit::bind_all<drjit::CUDADiffArray<float>>(b);
    }

    m.doc() = "Differentiable geometry queries built on Dr.Jit and OptiX.";
    m.attr("__name__") = "rayd";
    m.def("device_count",
          &checked_cuda_device_count,
          "Return the number of Dr.Jit-compatible CUDA devices visible to RayD.");
    m.def("current_device",
          []() { return checked_cuda_device_count() > 0 ? jit_cuda_device() : 0; },
          "Return the current thread's active Dr.Jit CUDA device index.");
    m.attr("REFLECTION_EXPORT_FULL") =
        nb::int_(static_cast<int>(RAYD_REFLECTION_EXPORT_FULL));
    m.attr("REFLECTION_EXPORT_MINIMAL") =
        nb::int_(static_cast<int>(RAYD_REFLECTION_EXPORT_MINIMAL));
    m.attr("REFLECTION_EXPORT_COUNT_ONLY") =
        nb::int_(static_cast<int>(RAYD_REFLECTION_EXPORT_COUNT_ONLY));
    m.def("native_launch_audit_clear",
          &native_launch_audit_clear,
          "Clear grouped native launch audit counters.");
    m.def("native_launch_audit",
          [native_stage_dict]() {
              const NativeLaunchAuditSnapshot snapshot = native_launch_audit_snapshot();
              nb::dict result;
              result["unknown"] = native_stage_dict(snapshot.unknown);
              result["build"] = native_stage_dict(snapshot.build);
              result["sync"] = native_stage_dict(snapshot.sync);
              result["trace_reflections"] = native_stage_dict(snapshot.trace_reflections);
              result["accumulate_reflections"] =
                  native_stage_dict(snapshot.accumulate_reflections);
              result["accum_dfr"] =
                  native_stage_dict(snapshot.accum_dfr);
              return result;
          },
          "Return grouped native launch audit counters.");
    m.def("set_device",
          &set_rayd_device,
          "device"_a,
          "initialize_optix"_a = true,
          "Set the current thread's active CUDA device for RayD.\n\n"
          "Call this before constructing RayD meshes, scenes, cameras, or "
          "Dr.Jit arrays that you intend to use with them. When "
          "initialize_optix=True, RayD also initializes the OptiX device "
          "context for the selected device.");
    // Naming convention: the bare class name is the non-AD variant, which is the
    // common case; the autodiff variant carries an "AD" suffix (e.g. Ray / RayAD,
    // Intersection / IntersectionAD). The C++ aliases follow the same convention,
    // so each Python class name matches its underlying C++ type.
    bind_section("core types", [&]() {
        nb::class_<Ray>(m, "Ray")
            .def(nb::init<>())
            .def(nb::init<const Vector3f &, const Vector3f &>())
            .def("reversed", &Ray::reversed)
            .def_rw("o", &Ray::o)
            .def_rw("d", &Ray::d)
            .def_rw("tmax", &Ray::tmax);

        nb::class_<RayAD>(m, "RayAD")
            .def(nb::init<>())
            .def(nb::init<const Vector3fAD &, const Vector3fAD &>())
            .def("reversed", &RayAD::reversed)
            .def_rw("o", &RayAD::o)
            .def_rw("d", &RayAD::d)
            .def_rw("tmax", &RayAD::tmax);

        nb::class_<SecondaryEdgeInfoAD>(m, "SecondaryEdgeInfo")
            .def(nb::init<>())
            .def("size", &SecondaryEdgeInfoAD::size)
            .def_ro("start", &SecondaryEdgeInfoAD::start)
            .def_ro("edge", &SecondaryEdgeInfoAD::edge)
            .def_ro("normal0", &SecondaryEdgeInfoAD::normal0)
            .def_ro("normal1", &SecondaryEdgeInfoAD::normal1)
            .def_ro("opposite", &SecondaryEdgeInfoAD::opposite)
            .def_ro("is_boundary", &SecondaryEdgeInfoAD::is_boundary);

        nb::class_<SceneEdgeInfo>(m, "SceneEdgeInfo")
            .def(nb::init<>())
            .def("size", &SceneEdgeInfo::size)
            .def_ro("start", &SceneEdgeInfo::start)
            .def_ro("edge", &SceneEdgeInfo::edge)
            .def_ro("end", &SceneEdgeInfo::end)
            .def_ro("length", &SceneEdgeInfo::length)
            .def_ro("normal0", &SceneEdgeInfo::normal0)
            .def_ro("normal1", &SceneEdgeInfo::normal1)
            .def_ro("is_boundary", &SceneEdgeInfo::is_boundary)
            .def_ro("shape_id", &SceneEdgeInfo::shape_id)
            .def_ro("local_edge_id", &SceneEdgeInfo::local_edge_id)
            .def_ro("global_edge_id", &SceneEdgeInfo::global_edge_id);

        nb::class_<SceneEdgeTopology>(m, "SceneEdgeTopology")
            .def(nb::init<>())
            .def("size", &SceneEdgeTopology::size)
            .def_ro("v0", &SceneEdgeTopology::v0)
            .def_ro("v1", &SceneEdgeTopology::v1)
            .def_ro("v0_global", &SceneEdgeTopology::v0_global)
            .def_ro("v1_global", &SceneEdgeTopology::v1_global)
            .def_ro("face0_local", &SceneEdgeTopology::face0_local)
            .def_ro("face1_local", &SceneEdgeTopology::face1_local)
            .def_ro("face0_global", &SceneEdgeTopology::face0_global)
            .def_ro("face1_global", &SceneEdgeTopology::face1_global)
            .def_ro("opposite_vertex0", &SceneEdgeTopology::opposite_vertex0)
            .def_ro("opposite_vertex1", &SceneEdgeTopology::opposite_vertex1)
            .def_ro("opposite_vertex0_global", &SceneEdgeTopology::opposite_vertex0_global)
            .def_ro("opposite_vertex1_global", &SceneEdgeTopology::opposite_vertex1_global);

        nb::class_<SceneGeometry>(m, "SceneGeometry")
            .def(nb::init<>())
            .def("vertex_count", &SceneGeometry::vertex_count)
            .def("face_count", &SceneGeometry::face_count)
            .def_ro("vertices", &SceneGeometry::vertices)
            .def_ro("faces", &SceneGeometry::faces)
            .def_ro("face_normal", &SceneGeometry::face_normal)
            .def_ro("shape_id", &SceneGeometry::shape_id)
            .def_ro("local_prim_id", &SceneGeometry::local_prim_id)
            .def_ro("global_prim_id", &SceneGeometry::global_prim_id);

        nb::enum_<RayFlags>(m, "RayFlags", nb::is_arithmetic())
            .value("None", RayFlags::None)
            .value("Geometric", RayFlags::Geometric)
            .value("ShadingN", RayFlags::ShadingN)
            .value("UV", RayFlags::UV)
            .value("All", RayFlags::All);

        nb::class_<ReflectionTraceOptions>(m, "ReflectionTraceOptions")
            .def(nb::init<>())
            .def_rw("deduplicate", &ReflectionTraceOptions::deduplicate)
            .def_rw("canonical_prim_table", &ReflectionTraceOptions::canonical_prim_table)
            .def_rw("image_source_tolerance", &ReflectionTraceOptions::image_source_tolerance)
            .def_rw("export_mode", &ReflectionTraceOptions::export_mode)
            .def_rw("return_trailing", &ReflectionTraceOptions::return_trailing);

        nb::class_<ReflEpcOptions>(m, "ReflEpcOptions")
            .def(nb::init<>())
            .def_rw("expected_prim_ids", &ReflEpcOptions::expected_prim_ids)
            .def_rw("surface_group_id", &ReflEpcOptions::surface_group_id)
            .def_rw("surface_group_size", &ReflEpcOptions::surface_group_size)
            .def_rw("surface_group_members", &ReflEpcOptions::surface_group_members)
            .def_rw("surface_max_group_size", &ReflEpcOptions::surface_max_group_size)
            .def_rw("visibility_ignore_mode", &ReflEpcOptions::visibility_ignore_mode)
            .def_rw("final_ignore_group_ids", &ReflEpcOptions::final_ignore_group_ids);

        nb::class_<ReflEpcFieldOptions, ReflEpcOptions>(
                m, "ReflEpcFieldOptions")
            .def(nb::init<>())
            .def_rw("slot_plane_point", &ReflEpcFieldOptions::slot_plane_point)
            .def_rw("slot_plane_normal", &ReflEpcFieldOptions::slot_plane_normal)
            .def_rw("slot_eta_r", &ReflEpcFieldOptions::slot_eta_r)
            .def_rw("slot_mu_r", &ReflEpcFieldOptions::slot_mu_r)
            .def_rw("slot_sigma", &ReflEpcFieldOptions::slot_sigma)
            .def_rw("slot_gain", &ReflEpcFieldOptions::slot_gain)
            .def_rw("tx_polarization", &ReflEpcFieldOptions::tx_polarization)
            .def_rw("omega", &ReflEpcFieldOptions::omega)
            .def_rw("wavelength", &ReflEpcFieldOptions::wavelength)
            .def_rw("return_geom", &ReflEpcFieldOptions::return_geom)
            .def_rw("return_endpoints", &ReflEpcFieldOptions::return_endpoints)
            .def_rw("return_hit_points", &ReflEpcFieldOptions::return_hit_points)
            .def_rw("return_normals", &ReflEpcFieldOptions::return_normals)
            .def_rw("return_resolved_prim_ids",
                    &ReflEpcFieldOptions::return_resolved_prim_ids)
            .def_rw("return_surface_group_ids",
                    &ReflEpcFieldOptions::return_surface_group_ids);

        nb::class_<ReflEpcFieldOptionsAD, ReflEpcOptions>(
                m, "ReflEpcFieldOptionsAD")
            .def(nb::init<>())
            .def_rw("slot_plane_point", &ReflEpcFieldOptionsAD::slot_plane_point)
            .def_rw("slot_plane_normal", &ReflEpcFieldOptionsAD::slot_plane_normal)
            .def_rw("slot_eta_r", &ReflEpcFieldOptionsAD::slot_eta_r)
            .def_rw("slot_mu_r", &ReflEpcFieldOptionsAD::slot_mu_r)
            .def_rw("slot_sigma", &ReflEpcFieldOptionsAD::slot_sigma)
            .def_rw("slot_gain", &ReflEpcFieldOptionsAD::slot_gain)
            .def_rw("tx_polarization", &ReflEpcFieldOptionsAD::tx_polarization)
            .def_rw("omega", &ReflEpcFieldOptionsAD::omega)
            .def_rw("wavelength", &ReflEpcFieldOptionsAD::wavelength)
            .def_rw("return_geom", &ReflEpcFieldOptionsAD::return_geom)
            .def_rw("return_endpoints", &ReflEpcFieldOptionsAD::return_endpoints)
            .def_rw("return_hit_points", &ReflEpcFieldOptionsAD::return_hit_points)
            .def_rw("return_normals", &ReflEpcFieldOptionsAD::return_normals)
            .def_rw("return_resolved_prim_ids",
                    &ReflEpcFieldOptionsAD::return_resolved_prim_ids)
            .def_rw("return_surface_group_ids",
                    &ReflEpcFieldOptionsAD::return_surface_group_ids);

        nb::class_<Intersection>(m, "Intersection")
            .def("is_valid", &Intersection::is_valid)
            .def_ro("t", &Intersection::t)
            .def_ro("p", &Intersection::p)
            .def_ro("n", &Intersection::n)
            .def_ro("geo_n", &Intersection::geo_n)
            .def_ro("uv", &Intersection::uv)
            .def_ro("barycentric", &Intersection::barycentric)
            .def_ro("shape_id", &Intersection::shape_id)
            .def_ro("prim_id", &Intersection::prim_id)
            .def_ro("local_prim_id", &Intersection::local_prim_id)
            .def_ro("global_prim_id", &Intersection::global_prim_id);

        nb::class_<IntersectionAD>(m, "IntersectionAD")
            .def("is_valid", &IntersectionAD::is_valid)
            .def_ro("t", &IntersectionAD::t)
            .def_ro("p", &IntersectionAD::p)
            .def_ro("n", &IntersectionAD::n)
            .def_ro("geo_n", &IntersectionAD::geo_n)
            .def_ro("uv", &IntersectionAD::uv)
            .def_ro("barycentric", &IntersectionAD::barycentric)
            .def_ro("shape_id", &IntersectionAD::shape_id)
            .def_ro("prim_id", &IntersectionAD::prim_id)
            .def_ro("local_prim_id", &IntersectionAD::local_prim_id)
            .def_ro("global_prim_id", &IntersectionAD::global_prim_id);

        nb::class_<ReflectionChain>(m, "ReflectionChain")
            .def("is_valid", &ReflectionChain::is_valid)
            .def_ro("max_bounces", &ReflectionChain::max_bounces)
            .def_ro("ray_count", &ReflectionChain::ray_count)
            .def_ro("bounce_count", &ReflectionChain::bounce_count)
            .def_ro("discovery_count", &ReflectionChain::discovery_count)
            .def_ro("representative_ray_index", &ReflectionChain::representative_ray_index)
            .def_ro("t", &ReflectionChain::t)
            .def_ro("hit_points", &ReflectionChain::hit_points)
            .def_ro("geo_normals", &ReflectionChain::geo_normals)
            .def_ro("image_sources", &ReflectionChain::image_sources)
            .def_ro("plane_points", &ReflectionChain::plane_points)
            .def_ro("plane_normals", &ReflectionChain::plane_normals)
            .def_ro("shape_ids", &ReflectionChain::shape_ids)
            .def_ro("prim_ids", &ReflectionChain::prim_ids)
            .def_ro("local_prim_ids", &ReflectionChain::local_prim_ids)
            .def_ro("global_prim_ids", &ReflectionChain::global_prim_ids)
            .def_ro("trailing_t", &ReflectionChain::trailing_t)
            .def_ro("trailing_prim", &ReflectionChain::trailing_prim)
            .def_ro("trailing_dir", &ReflectionChain::trailing_dir)
            .def_ro("trailing_origin", &ReflectionChain::trailing_origin);

        nb::class_<ReflectionChainAD>(m, "ReflectionChainAD")
            .def("is_valid", &ReflectionChainAD::is_valid)
            .def_ro("max_bounces", &ReflectionChainAD::max_bounces)
            .def_ro("ray_count", &ReflectionChainAD::ray_count)
            .def_ro("bounce_count", &ReflectionChainAD::bounce_count)
            .def_ro("discovery_count", &ReflectionChainAD::discovery_count)
            .def_ro("representative_ray_index", &ReflectionChainAD::representative_ray_index)
            .def_ro("t", &ReflectionChainAD::t)
            .def_ro("hit_points", &ReflectionChainAD::hit_points)
            .def_ro("geo_normals", &ReflectionChainAD::geo_normals)
            .def_ro("image_sources", &ReflectionChainAD::image_sources)
            .def_ro("plane_points", &ReflectionChainAD::plane_points)
            .def_ro("plane_normals", &ReflectionChainAD::plane_normals)
            .def_ro("shape_ids", &ReflectionChainAD::shape_ids)
            .def_ro("prim_ids", &ReflectionChainAD::prim_ids)
            .def_ro("local_prim_ids", &ReflectionChainAD::local_prim_ids)
            .def_ro("global_prim_ids", &ReflectionChainAD::global_prim_ids)
            .def_ro("trailing_t", &ReflectionChainAD::trailing_t)
            .def_ro("trailing_prim", &ReflectionChainAD::trailing_prim)
            .def_ro("trailing_dir", &ReflectionChainAD::trailing_dir)
            .def_ro("trailing_origin", &ReflectionChainAD::trailing_origin);

        nb::class_<AccumGrid>(m, "AccumGrid")
            .def(nb::init<>())
            .def_rw("axis", &AccumGrid::axis)
            .def_rw("position", &AccumGrid::position)
            .def_rw("coord0_min", &AccumGrid::coord0_min)
            .def_rw("coord0_max", &AccumGrid::coord0_max)
            .def_rw("coord1_min", &AccumGrid::coord1_min)
            .def_rw("coord1_max", &AccumGrid::coord1_max)
            .def_rw("resolution0", &AccumGrid::resolution0)
            .def_rw("resolution1", &AccumGrid::resolution1);

        nb::class_<AccumOptions>(m, "AccumOptions")
            .def(nb::init<>())
            .def_rw("wavelength", &AccumOptions::wavelength)
            .def_rw("k", &AccumOptions::k)
            .def_rw("solid_angle_per_ray", &AccumOptions::solid_angle_per_ray)
            .def_rw("cell_area", &AccumOptions::cell_area)
            .def_rw("seed", &AccumOptions::seed)
            .def_rw("rr_depth", &AccumOptions::rr_depth)
            .def_rw("rr_prob", &AccumOptions::rr_prob)
            .def_rw("stop_threshold", &AccumOptions::stop_threshold)
            .def_rw("collect_wedges", &AccumOptions::collect_wedges)
            .def_rw("collect_wedge_prefixes", &AccumOptions::collect_wedge_prefixes)
            .def_rw("wedge_capacity", &AccumOptions::wedge_capacity)
            .def_rw("wedge_sample_stride", &AccumOptions::wedge_sample_stride);

        nb::class_<Material>(m, "Material")
            .def(nb::init<>())
            .def_rw("eta_r", &Material::eta_r)
            .def_rw("sigma", &Material::sigma)
            .def_rw("gain", &Material::gain)
            .def_rw("mu_r", &Material::mu_r)
            .def_rw("valid", &Material::valid);

        nb::class_<MaterialAD>(m, "MaterialAD")
            .def(nb::init<>())
            .def_rw("eta_r", &MaterialAD::eta_r)
            .def_rw("sigma", &MaterialAD::sigma)
            .def_rw("gain", &MaterialAD::gain)
            .def_rw("mu_r", &MaterialAD::mu_r)
            .def_rw("valid", &MaterialAD::valid);

        nb::class_<WedgeEvents>(m, "WedgeEvents")
            .def_ro("capacity", &WedgeEvents::capacity)
            .def_ro("count", &WedgeEvents::count)
            .def_ro("ray_index", &WedgeEvents::ray_index)
            .def_ro("hit_points", &WedgeEvents::hit_points)
            .def_ro("normals", &WedgeEvents::normals)
            .def_ro("prim_id", &WedgeEvents::prim_id)
            .def_ro("directions", &WedgeEvents::directions)
            .def_ro("source_points", &WedgeEvents::source_points)
            .def_ro("src_power", &WedgeEvents::src_power)
            .def_ro("initial_directions", &WedgeEvents::initial_directions)
            .def_ro("bounce_depth", &WedgeEvents::bounce_depth);

        nb::class_<WedgeEventsAD>(m, "WedgeEventsAD")
            .def_ro("capacity", &WedgeEventsAD::capacity)
            .def_ro("count", &WedgeEventsAD::count)
            .def_ro("ray_index", &WedgeEventsAD::ray_index)
            .def_ro("hit_points", &WedgeEventsAD::hit_points)
            .def_ro("normals", &WedgeEventsAD::normals)
            .def_ro("prim_id", &WedgeEventsAD::prim_id)
            .def_ro("directions", &WedgeEventsAD::directions)
            .def_ro("source_points", &WedgeEventsAD::source_points)
            .def_ro("src_power", &WedgeEventsAD::src_power)
            .def_ro("initial_directions", &WedgeEventsAD::initial_directions)
            .def_ro("bounce_depth", &WedgeEventsAD::bounce_depth);

        nb::class_<AccumResult>(m, "AccumResult")
            .def_ro("ray_count", &AccumResult::ray_count)
            .def_ro("max_bounces", &AccumResult::max_bounces)
            .def_ro("grid_cell_count", &AccumResult::grid_cell_count)
            .def_ro("reflection_power", &AccumResult::reflection_power)
            .def_ro("reflection_field_x", &AccumResult::reflection_field_x)
            .def_ro("reflection_field_y", &AccumResult::reflection_field_y)
            .def_ro("reflection_field_z", &AccumResult::reflection_field_z)
            .def_ro("reflection_count", &AccumResult::reflection_count)
            .def_ro("wedge_events", &AccumResult::wedge_events);

        nb::class_<AccumResultAD>(m, "AccumResultAD")
            .def_ro("ray_count", &AccumResultAD::ray_count)
            .def_ro("max_bounces", &AccumResultAD::max_bounces)
            .def_ro("grid_cell_count", &AccumResultAD::grid_cell_count)
            .def_ro("reflection_power", &AccumResultAD::reflection_power)
            .def_ro("reflection_field_x", &AccumResultAD::reflection_field_x)
            .def_ro("reflection_field_y", &AccumResultAD::reflection_field_y)
            .def_ro("reflection_field_z", &AccumResultAD::reflection_field_z)
            .def_ro("reflection_count", &AccumResultAD::reflection_count)
            .def_ro("wedge_events", &AccumResultAD::wedge_events);

        m.attr("RAYD_DFR_DIRECT") =
            nb::int_(static_cast<int>(RAYD_DFR_DIRECT));
        m.attr("RAYD_DFR_KELLER") =
            nb::int_(static_cast<int>(RAYD_DFR_KELLER));
        m.attr("RAYD_DFR_SUFFIX_REFL") =
            nb::int_(static_cast<int>(RAYD_DFR_SUFFIX_REFL));
        m.attr("RAYD_DFR_HASH") = nb::int_(static_cast<int>(RAYD_DFR_HASH));
        m.attr("RAYD_DFR_SOBOL") = nb::int_(static_cast<int>(RAYD_DFR_SOBOL));
        m.attr("RAYD_DFR_MATCHED_ISO") =
            nb::int_(static_cast<int>(RAYD_DFR_MATCHED_ISO));

        nb::class_<DfrGrid>(m, "DfrGrid")
            .def(nb::init<>())
            .def_rw("axis", &DfrGrid::axis)
            .def_rw("position", &DfrGrid::position)
            .def_rw("coord0_min", &DfrGrid::coord0_min)
            .def_rw("coord0_max", &DfrGrid::coord0_max)
            .def_rw("coord1_min", &DfrGrid::coord1_min)
            .def_rw("coord1_max", &DfrGrid::coord1_max)
            .def_rw("resolution0", &DfrGrid::resolution0)
            .def_rw("resolution1", &DfrGrid::resolution1)
            .def_rw("cell_area", &DfrGrid::cell_area);

        nb::class_<DfrOptions>(m, "DfrOptions")
            .def(nb::init<>())
            .def_rw("wavelength", &DfrOptions::wavelength)
            .def_rw("k", &DfrOptions::k)
            .def_rw("seed", &DfrOptions::seed)
            .def_rw("samples", &DfrOptions::samples)
            .def_rw("max_order", &DfrOptions::max_order)
            .def_rw("direct_samples", &DfrOptions::direct_samples)
            .def_rw("keller_samples", &DfrOptions::keller_samples)
            .def_rw("suffix_samples", &DfrOptions::suffix_samples)
            .def_rw("strategy_mask", &DfrOptions::strategy_mask)
            .def_rw("sample_sequence", &DfrOptions::sample_sequence)
            .def_rw("receiver_model", &DfrOptions::receiver_model)
            .def_rw("collect_edge_use", &DfrOptions::collect_edge_use)
            .def_rw("collect_debug_counts",
                    &DfrOptions::collect_debug_counts);

        nb::class_<DfrCoherentOptions>(m, "DfrCoherentOptions")
            .def(nb::init<>())
            .def_rw("wavelength", &DfrCoherentOptions::wavelength)
            .def_rw("k", &DfrCoherentOptions::k)
            .def_rw("max_order", &DfrCoherentOptions::max_order)
            .def_rw("receiver_model", &DfrCoherentOptions::receiver_model)
            .def_rw("select_diffraction_point",
                    &DfrCoherentOptions::select_diffraction_point)
            .def_rw("prefilter_visibility",
                    &DfrCoherentOptions::prefilter_visibility)
            .def_rw("collect_debug_counts",
                    &DfrCoherentOptions::collect_debug_counts)
            .def_rw("omega", &DfrCoherentOptions::omega)
            .def_rw("tx_pol_x", &DfrCoherentOptions::tx_pol_x)
            .def_rw("tx_pol_y", &DfrCoherentOptions::tx_pol_y)
            .def_rw("tx_pol_z", &DfrCoherentOptions::tx_pol_z)
            .def_rw("higher_probe_radius_scale",
                    &DfrCoherentOptions::higher_probe_radius_scale)
            .def_rw("higher_probe_radius_min",
                    &DfrCoherentOptions::higher_probe_radius_min)
            .def_rw("higher_probe_radius_max",
                    &DfrCoherentOptions::higher_probe_radius_max)
            .def_rw("higher_filter_visibility",
                    &DfrCoherentOptions::higher_filter_visibility);

        nb::class_<DfrMaterial>(m, "DfrMaterial")
            .def(nb::init<>())
            .def_rw("eta_r", &DfrMaterial::eta_r)
            .def_rw("sigma", &DfrMaterial::sigma)
            .def_rw("mu_r", &DfrMaterial::mu_r)
            .def_rw("gain", &DfrMaterial::gain)
            .def_rw("valid", &DfrMaterial::valid);

        nb::class_<DfrMaterialAD>(m, "DfrMaterialAD")
            .def(nb::init<>())
            .def_rw("eta_r", &DfrMaterialAD::eta_r)
            .def_rw("sigma", &DfrMaterialAD::sigma)
            .def_rw("mu_r", &DfrMaterialAD::mu_r)
            .def_rw("gain", &DfrMaterialAD::gain)
            .def_rw("valid", &DfrMaterialAD::valid);

        nb::class_<DfrCoherentUtdStates>(m, "DfrCoherentUtdStates")
            .def(nb::init<>())
            .def_rw("count", &DfrCoherentUtdStates::count)
            .def_rw("edge_index", &DfrCoherentUtdStates::edge_index)
            .def_rw("edge_pos", &DfrCoherentUtdStates::edge_pos)
            .def_rw("edge_dir", &DfrCoherentUtdStates::edge_dir)
            .def_rw("n0", &DfrCoherentUtdStates::n0)
            .def_rw("n_face_n", &DfrCoherentUtdStates::n_face_n)
            .def_rw("wedge_n", &DfrCoherentUtdStates::wedge_n)
            .def_rw("edge_line_min", &DfrCoherentUtdStates::edge_line_min)
            .def_rw("edge_line_max", &DfrCoherentUtdStates::edge_line_max)
            .def_rw("source_pos", &DfrCoherentUtdStates::source_pos)
            .def_rw("incident_field", &DfrCoherentUtdStates::incident_field)
            .def_rw("incident_normal_derivative", &DfrCoherentUtdStates::incident_normal_derivative)
            .def_rw("r_face0", &DfrCoherentUtdStates::r_face0)
            .def_rw("r_face_n", &DfrCoherentUtdStates::r_face_n)
            .def_rw("incident_vector_x", &DfrCoherentUtdStates::incident_vector_x)
            .def_rw("incident_vector_y", &DfrCoherentUtdStates::incident_vector_y)
            .def_rw("incident_vector_z", &DfrCoherentUtdStates::incident_vector_z)
            .def_rw("incident_normal_derivative_vector_x", &DfrCoherentUtdStates::incident_normal_derivative_vector_x)
            .def_rw("incident_normal_derivative_vector_y", &DfrCoherentUtdStates::incident_normal_derivative_vector_y)
            .def_rw("incident_normal_derivative_vector_z", &DfrCoherentUtdStates::incident_normal_derivative_vector_z)
            .def_rw("incident_jones_u", &DfrCoherentUtdStates::incident_jones_u)
            .def_rw("incident_jones_v", &DfrCoherentUtdStates::incident_jones_v)
            .def_rw("incident_derivative_jones_u", &DfrCoherentUtdStates::incident_derivative_jones_u)
            .def_rw("incident_derivative_jones_v", &DfrCoherentUtdStates::incident_derivative_jones_v)
            .def_rw("incident_basis_u", &DfrCoherentUtdStates::incident_basis_u)
            .def_rw("incident_basis_v", &DfrCoherentUtdStates::incident_basis_v)
            .def_rw("incident_basis_k", &DfrCoherentUtdStates::incident_basis_k)
            .def_rw("face0_operator_m00", &DfrCoherentUtdStates::face0_operator_m00)
            .def_rw("face0_operator_m01", &DfrCoherentUtdStates::face0_operator_m01)
            .def_rw("face0_operator_m10", &DfrCoherentUtdStates::face0_operator_m10)
            .def_rw("face0_operator_m11", &DfrCoherentUtdStates::face0_operator_m11)
            .def_rw("face1_operator_m00", &DfrCoherentUtdStates::face1_operator_m00)
            .def_rw("face1_operator_m01", &DfrCoherentUtdStates::face1_operator_m01)
            .def_rw("face1_operator_m10", &DfrCoherentUtdStates::face1_operator_m10)
            .def_rw("face1_operator_m11", &DfrCoherentUtdStates::face1_operator_m11)
            .def_rw("face0_eta_r", &DfrCoherentUtdStates::face0_eta_r)
            .def_rw("face0_mu_r", &DfrCoherentUtdStates::face0_mu_r)
            .def_rw("face0_sigma", &DfrCoherentUtdStates::face0_sigma)
            .def_rw("face0_gain", &DfrCoherentUtdStates::face0_gain)
            .def_rw("face0_use_fresnel", &DfrCoherentUtdStates::face0_use_fresnel)
            .def_rw("face1_eta_r", &DfrCoherentUtdStates::face1_eta_r)
            .def_rw("face1_mu_r", &DfrCoherentUtdStates::face1_mu_r)
            .def_rw("face1_sigma", &DfrCoherentUtdStates::face1_sigma)
            .def_rw("face1_gain", &DfrCoherentUtdStates::face1_gain)
            .def_rw("face1_use_fresnel", &DfrCoherentUtdStates::face1_use_fresnel)
            .def_rw("select_stationary_point", &DfrCoherentUtdStates::select_stationary_point)
            .def_rw("owner_code", &DfrCoherentUtdStates::owner_code)
            .def_rw("adjacent_face0", &DfrCoherentUtdStates::adjacent_face0)
            .def_rw("adjacent_face1", &DfrCoherentUtdStates::adjacent_face1)
            .def_rw("path_length_prefix", &DfrCoherentUtdStates::path_length_prefix)
            .def_rw("first_interaction_pos", &DfrCoherentUtdStates::first_interaction_pos)
            .def_rw("source_type_code", &DfrCoherentUtdStates::source_type_code)
            .def_rw("prefix_reflection_depth", &DfrCoherentUtdStates::prefix_reflection_depth)
            .def_rw("intermediate_reflection_depth", &DfrCoherentUtdStates::intermediate_reflection_depth)
            .def_rw("suffix_reflection_depth", &DfrCoherentUtdStates::suffix_reflection_depth)
            .def_rw("approximation_mode_code", &DfrCoherentUtdStates::approximation_mode_code)
            .def_rw("order", &DfrCoherentUtdStates::order);

        nb::class_<DfrCoherentUtdStatesAD>(m, "DfrCoherentUtdStatesAD")
            .def(nb::init<>())
            .def_rw("count", &DfrCoherentUtdStatesAD::count)
            .def_rw("edge_index", &DfrCoherentUtdStatesAD::edge_index)
            .def_rw("edge_pos", &DfrCoherentUtdStatesAD::edge_pos)
            .def_rw("edge_dir", &DfrCoherentUtdStatesAD::edge_dir)
            .def_rw("n0", &DfrCoherentUtdStatesAD::n0)
            .def_rw("n_face_n", &DfrCoherentUtdStatesAD::n_face_n)
            .def_rw("wedge_n", &DfrCoherentUtdStatesAD::wedge_n)
            .def_rw("edge_line_min", &DfrCoherentUtdStatesAD::edge_line_min)
            .def_rw("edge_line_max", &DfrCoherentUtdStatesAD::edge_line_max)
            .def_rw("source_pos", &DfrCoherentUtdStatesAD::source_pos)
            .def_rw("incident_field", &DfrCoherentUtdStatesAD::incident_field)
            .def_rw("incident_normal_derivative", &DfrCoherentUtdStatesAD::incident_normal_derivative)
            .def_rw("r_face0", &DfrCoherentUtdStatesAD::r_face0)
            .def_rw("r_face_n", &DfrCoherentUtdStatesAD::r_face_n)
            .def_rw("incident_vector_x", &DfrCoherentUtdStatesAD::incident_vector_x)
            .def_rw("incident_vector_y", &DfrCoherentUtdStatesAD::incident_vector_y)
            .def_rw("incident_vector_z", &DfrCoherentUtdStatesAD::incident_vector_z)
            .def_rw("incident_normal_derivative_vector_x", &DfrCoherentUtdStatesAD::incident_normal_derivative_vector_x)
            .def_rw("incident_normal_derivative_vector_y", &DfrCoherentUtdStatesAD::incident_normal_derivative_vector_y)
            .def_rw("incident_normal_derivative_vector_z", &DfrCoherentUtdStatesAD::incident_normal_derivative_vector_z)
            .def_rw("incident_jones_u", &DfrCoherentUtdStatesAD::incident_jones_u)
            .def_rw("incident_jones_v", &DfrCoherentUtdStatesAD::incident_jones_v)
            .def_rw("incident_derivative_jones_u", &DfrCoherentUtdStatesAD::incident_derivative_jones_u)
            .def_rw("incident_derivative_jones_v", &DfrCoherentUtdStatesAD::incident_derivative_jones_v)
            .def_rw("incident_basis_u", &DfrCoherentUtdStatesAD::incident_basis_u)
            .def_rw("incident_basis_v", &DfrCoherentUtdStatesAD::incident_basis_v)
            .def_rw("incident_basis_k", &DfrCoherentUtdStatesAD::incident_basis_k)
            .def_rw("face0_operator_m00", &DfrCoherentUtdStatesAD::face0_operator_m00)
            .def_rw("face0_operator_m01", &DfrCoherentUtdStatesAD::face0_operator_m01)
            .def_rw("face0_operator_m10", &DfrCoherentUtdStatesAD::face0_operator_m10)
            .def_rw("face0_operator_m11", &DfrCoherentUtdStatesAD::face0_operator_m11)
            .def_rw("face1_operator_m00", &DfrCoherentUtdStatesAD::face1_operator_m00)
            .def_rw("face1_operator_m01", &DfrCoherentUtdStatesAD::face1_operator_m01)
            .def_rw("face1_operator_m10", &DfrCoherentUtdStatesAD::face1_operator_m10)
            .def_rw("face1_operator_m11", &DfrCoherentUtdStatesAD::face1_operator_m11)
            .def_rw("face0_eta_r", &DfrCoherentUtdStatesAD::face0_eta_r)
            .def_rw("face0_mu_r", &DfrCoherentUtdStatesAD::face0_mu_r)
            .def_rw("face0_sigma", &DfrCoherentUtdStatesAD::face0_sigma)
            .def_rw("face0_gain", &DfrCoherentUtdStatesAD::face0_gain)
            .def_rw("face0_use_fresnel", &DfrCoherentUtdStatesAD::face0_use_fresnel)
            .def_rw("face1_eta_r", &DfrCoherentUtdStatesAD::face1_eta_r)
            .def_rw("face1_mu_r", &DfrCoherentUtdStatesAD::face1_mu_r)
            .def_rw("face1_sigma", &DfrCoherentUtdStatesAD::face1_sigma)
            .def_rw("face1_gain", &DfrCoherentUtdStatesAD::face1_gain)
            .def_rw("face1_use_fresnel", &DfrCoherentUtdStatesAD::face1_use_fresnel)
            .def_rw("select_stationary_point", &DfrCoherentUtdStatesAD::select_stationary_point)
            .def_rw("owner_code", &DfrCoherentUtdStatesAD::owner_code)
            .def_rw("adjacent_face0", &DfrCoherentUtdStatesAD::adjacent_face0)
            .def_rw("adjacent_face1", &DfrCoherentUtdStatesAD::adjacent_face1)
            .def_rw("path_length_prefix", &DfrCoherentUtdStatesAD::path_length_prefix)
            .def_rw("first_interaction_pos", &DfrCoherentUtdStatesAD::first_interaction_pos)
            .def_rw("source_type_code", &DfrCoherentUtdStatesAD::source_type_code)
            .def_rw("prefix_reflection_depth", &DfrCoherentUtdStatesAD::prefix_reflection_depth)
            .def_rw("intermediate_reflection_depth", &DfrCoherentUtdStatesAD::intermediate_reflection_depth)
            .def_rw("suffix_reflection_depth", &DfrCoherentUtdStatesAD::suffix_reflection_depth)
            .def_rw("approximation_mode_code", &DfrCoherentUtdStatesAD::approximation_mode_code)
            .def_rw("order", &DfrCoherentUtdStatesAD::order);

        nb::class_<DfrCoherentEdge>(m, "DfrCoherentEdge")
            .def(nb::init<>())
            .def_rw("count", &DfrCoherentEdge::count)
            .def_rw("edge_index", &DfrCoherentEdge::edge_index)
            .def_rw("edge_pos", &DfrCoherentEdge::edge_pos)
            .def_rw("edge_dir", &DfrCoherentEdge::edge_dir)
            .def_rw("n0", &DfrCoherentEdge::n0)
            .def_rw("n_face_n", &DfrCoherentEdge::n_face_n)
            .def_rw("wedge_n", &DfrCoherentEdge::wedge_n)
            .def_rw("edge_line_min", &DfrCoherentEdge::edge_line_min)
            .def_rw("edge_line_max", &DfrCoherentEdge::edge_line_max)
            .def_rw("adjacent_face0", &DfrCoherentEdge::adjacent_face0)
            .def_rw("adjacent_face1", &DfrCoherentEdge::adjacent_face1)
            .def_rw("ignore_prim_ids", &DfrCoherentEdge::ignore_prim_ids)
            .def_rw("ignore_k", &DfrCoherentEdge::ignore_k);

        nb::class_<DfrCoherentEdgeAD>(m, "DfrCoherentEdgeAD")
            .def(nb::init<>())
            .def_rw("count", &DfrCoherentEdgeAD::count)
            .def_rw("edge_index", &DfrCoherentEdgeAD::edge_index)
            .def_rw("edge_pos", &DfrCoherentEdgeAD::edge_pos)
            .def_rw("edge_dir", &DfrCoherentEdgeAD::edge_dir)
            .def_rw("n0", &DfrCoherentEdgeAD::n0)
            .def_rw("n_face_n", &DfrCoherentEdgeAD::n_face_n)
            .def_rw("wedge_n", &DfrCoherentEdgeAD::wedge_n)
            .def_rw("edge_line_min", &DfrCoherentEdgeAD::edge_line_min)
            .def_rw("edge_line_max", &DfrCoherentEdgeAD::edge_line_max)
            .def_rw("adjacent_face0", &DfrCoherentEdgeAD::adjacent_face0)
            .def_rw("adjacent_face1", &DfrCoherentEdgeAD::adjacent_face1)
            .def_rw("ignore_prim_ids", &DfrCoherentEdgeAD::ignore_prim_ids)
            .def_rw("ignore_k", &DfrCoherentEdgeAD::ignore_k);

        nb::class_<DfrCoherentCandidatePairs>(m, "DfrCoherentCandidatePairs")
            .def(nb::init<>())
            .def_rw("count", &DfrCoherentCandidatePairs::count)
            .def_rw("visibility_filtered", &DfrCoherentCandidatePairs::visibility_filtered)
            .def_rw("prev_index", &DfrCoherentCandidatePairs::prev_index)
            .def_rw("edge_index", &DfrCoherentCandidatePairs::edge_index);

        nb::class_<DfrCoherentCandidatePairsAD>(m, "DfrCoherentCandidatePairsAD")
            .def(nb::init<>())
            .def_rw("count", &DfrCoherentCandidatePairsAD::count)
            .def_rw("visibility_filtered", &DfrCoherentCandidatePairsAD::visibility_filtered)
            .def_rw("prev_index", &DfrCoherentCandidatePairsAD::prev_index)
            .def_rw("edge_index", &DfrCoherentCandidatePairsAD::edge_index);

        nb::class_<DfrStates>(m, "DfrStates")
            .def(nb::init<>())
            .def_rw("count", &DfrStates::count)
            .def_rw("edge_index", &DfrStates::edge_index)
            .def_rw("edge_pos", &DfrStates::edge_pos)
            .def_rw("edge_dir", &DfrStates::edge_dir)
            .def_rw("edge_t_min", &DfrStates::edge_t_min)
            .def_rw("edge_t_max", &DfrStates::edge_t_max)
            .def_rw("n0", &DfrStates::n0)
            .def_rw("n1", &DfrStates::n1)
            .def_rw("prim0", &DfrStates::prim0)
            .def_rw("prim1", &DfrStates::prim1)
            .def_rw("exterior_angle", &DfrStates::exterior_angle)
            .def_rw("src", &DfrStates::src)
            .def_rw("src_power", &DfrStates::src_power)
            .def_rw("wi",
                    &DfrStates::wi)
            .def_rw("d0",
                    &DfrStates::d0)
            .def_rw("prefix_depth",
                    &DfrStates::prefix_depth);

        nb::class_<DfrStatesAD>(m, "DfrStatesAD")
            .def(nb::init<>())
            .def_rw("count", &DfrStatesAD::count)
            .def_rw("edge_index", &DfrStatesAD::edge_index)
            .def_rw("edge_pos", &DfrStatesAD::edge_pos)
            .def_rw("edge_dir", &DfrStatesAD::edge_dir)
            .def_rw("edge_t_min", &DfrStatesAD::edge_t_min)
            .def_rw("edge_t_max", &DfrStatesAD::edge_t_max)
            .def_rw("n0", &DfrStatesAD::n0)
            .def_rw("n1", &DfrStatesAD::n1)
            .def_rw("prim0", &DfrStatesAD::prim0)
            .def_rw("prim1", &DfrStatesAD::prim1)
            .def_rw("exterior_angle", &DfrStatesAD::exterior_angle)
            .def_rw("src", &DfrStatesAD::src)
            .def_rw("src_power", &DfrStatesAD::src_power)
            .def_rw("wi",
                    &DfrStatesAD::wi)
            .def_rw("d0",
                    &DfrStatesAD::d0)
            .def_rw("prefix_depth",
                    &DfrStatesAD::prefix_depth);

        nb::class_<DfrAccum>(m, "DfrAccum")
            .def(nb::init<>())
            .def_rw("grid_cell_count", &DfrAccum::grid_cell_count)
            .def_rw("power", &DfrAccum::power)
            .def_rw("field_x",
                    &DfrAccum::field_x)
            .def_rw("field_y",
                    &DfrAccum::field_y)
            .def_rw("field_z",
                    &DfrAccum::field_z)
            .def_rw("direct_count", &DfrAccum::direct_count)
            .def_rw("keller_count", &DfrAccum::keller_count)
            .def_rw("suffix_count", &DfrAccum::suffix_count)
            .def_rw("vis_rejects",
                    &DfrAccum::vis_rejects)
            .def_rw("edge_vis_rejects",
                    &DfrAccum::edge_vis_rejects)
            .def_rw("utd_rejects", &DfrAccum::utd_rejects)
            .def_rw("edge_uses", &DfrAccum::edge_uses);

        nb::class_<DfrAccumAD>(m, "DfrAccumAD")
            .def(nb::init<>())
            .def_rw("grid_cell_count", &DfrAccumAD::grid_cell_count)
            .def_rw("power",
                    &DfrAccumAD::power)
            .def_rw("field_x",
                    &DfrAccumAD::field_x)
            .def_rw("field_y",
                    &DfrAccumAD::field_y)
            .def_rw("field_z",
                    &DfrAccumAD::field_z)
            .def_rw("direct_count", &DfrAccumAD::direct_count)
            .def_rw("keller_count", &DfrAccumAD::keller_count)
            .def_rw("suffix_count", &DfrAccumAD::suffix_count)
            .def_rw("vis_rejects",
                    &DfrAccumAD::vis_rejects)
            .def_rw("edge_vis_rejects",
                    &DfrAccumAD::edge_vis_rejects)
            .def_rw("utd_rejects", &DfrAccumAD::utd_rejects)
            .def_rw("edge_uses", &DfrAccumAD::edge_uses);

        nb::class_<DfrCoherentAccum>(m, "DfrCoherentAccum")
            .def(nb::init<>())
            .def_rw("grid_cell_count", &DfrCoherentAccum::grid_cell_count)
            .def_rw("direct_field_x",
                    &DfrCoherentAccum::direct_field_x)
            .def_rw("direct_field_y",
                    &DfrCoherentAccum::direct_field_y)
            .def_rw("direct_field_z",
                    &DfrCoherentAccum::direct_field_z)
            .def_rw("multi_field_x",
                    &DfrCoherentAccum::multi_field_x)
            .def_rw("multi_field_y",
                    &DfrCoherentAccum::multi_field_y)
            .def_rw("multi_field_z",
                    &DfrCoherentAccum::multi_field_z)
            .def_rw("direct_count", &DfrCoherentAccum::direct_count)
            .def_rw("multi_count", &DfrCoherentAccum::multi_count)
            .def_rw("visibility_reject_count",
                    &DfrCoherentAccum::visibility_reject_count)
            .def_rw("utd_reject_count",
                    &DfrCoherentAccum::utd_reject_count);

        nb::class_<DfrCoherentAccumAD>(m, "DfrCoherentAccumAD")
            .def(nb::init<>())
            .def_rw("grid_cell_count", &DfrCoherentAccumAD::grid_cell_count)
            .def_rw("direct_field_x",
                    &DfrCoherentAccumAD::direct_field_x)
            .def_rw("direct_field_y",
                    &DfrCoherentAccumAD::direct_field_y)
            .def_rw("direct_field_z",
                    &DfrCoherentAccumAD::direct_field_z)
            .def_rw("multi_field_x",
                    &DfrCoherentAccumAD::multi_field_x)
            .def_rw("multi_field_y",
                    &DfrCoherentAccumAD::multi_field_y)
            .def_rw("multi_field_z",
                    &DfrCoherentAccumAD::multi_field_z)
            .def_rw("direct_count", &DfrCoherentAccumAD::direct_count)
            .def_rw("multi_count", &DfrCoherentAccumAD::multi_count)
            .def_rw("visibility_reject_count",
                    &DfrCoherentAccumAD::visibility_reject_count)
            .def_rw("utd_reject_count",
                    &DfrCoherentAccumAD::utd_reject_count);

        nb::class_<DfrPathOptions>(m, "DfrPathOptions")
            .def(nb::init<>())
            .def_rw("wavelength", &DfrPathOptions::wavelength)
            .def_rw("k", &DfrPathOptions::k)
            .def_rw("seed", &DfrPathOptions::seed)
            .def_rw("max_order", &DfrPathOptions::max_order)
            .def_rw("max_paths", &DfrPathOptions::max_paths)
            .def_rw("max_rx", &DfrPathOptions::max_rx)
            .def_rw("strategy_mask", &DfrPathOptions::strategy_mask)
            .def_rw("sample_count", &DfrPathOptions::sample_count)
            .def_rw("return_geom", &DfrPathOptions::return_geom)
            .def_rw("receiver_model", &DfrPathOptions::receiver_model);

        nb::class_<DfrPaths>(m, "DfrPaths")
            .def(nb::init<>())
            .def_rw("capacity", &DfrPaths::capacity)
            .def_rw("count", &DfrPaths::count)
            .def_rw("valid", &DfrPaths::valid)
            .def_rw("tx_id", &DfrPaths::tx_id)
            .def_rw("rx_id", &DfrPaths::rx_id)
            .def_rw("order", &DfrPaths::order)
            .def_rw("edge0", &DfrPaths::edge0)
            .def_rw("edge1", &DfrPaths::edge1)
            .def_rw("edge2", &DfrPaths::edge2)
            .def_rw("delay", &DfrPaths::delay)
            .def_rw("field_x", &DfrPaths::field_x)
            .def_rw("field_y", &DfrPaths::field_y)
            .def_rw("field_z", &DfrPaths::field_z)
            .def_rw("p0", &DfrPaths::p0)
            .def_rw("p1", &DfrPaths::p1)
            .def_rw("p2", &DfrPaths::p2);

        nb::class_<DfrPathsAD>(m, "DfrPathsAD")
            .def(nb::init<>())
            .def_rw("capacity", &DfrPathsAD::capacity)
            .def_rw("count", &DfrPathsAD::count)
            .def_rw("valid", &DfrPathsAD::valid)
            .def_rw("tx_id", &DfrPathsAD::tx_id)
            .def_rw("rx_id", &DfrPathsAD::rx_id)
            .def_rw("order", &DfrPathsAD::order)
            .def_rw("edge0", &DfrPathsAD::edge0)
            .def_rw("edge1", &DfrPathsAD::edge1)
            .def_rw("edge2", &DfrPathsAD::edge2)
            .def_rw("delay", &DfrPathsAD::delay)
            .def_rw("field_x", &DfrPathsAD::field_x)
            .def_rw("field_y", &DfrPathsAD::field_y)
            .def_rw("field_z", &DfrPathsAD::field_z)
            .def_rw("p0", &DfrPathsAD::p0)
            .def_rw("p1", &DfrPathsAD::p1)
            .def_rw("p2", &DfrPathsAD::p2);

        nb::class_<ReflEpc>(m, "ReflEpc")
            .def_ro("ray_count", &ReflEpc::ray_count)
            .def_ro("max_bounces", &ReflEpc::max_bounces)
            .def_ro("valid", &ReflEpc::valid)
            .def_ro("bounce_count", &ReflEpc::bounce_count)
            .def_ro("path_length", &ReflEpc::path_length)
            .def_ro("reflection_points", &ReflEpc::reflection_points)
            .def_ro("prim_ids", &ReflEpc::prim_ids)
            .def_ro("trace_prim_ids", &ReflEpc::trace_prim_ids)
            .def_ro("resolved_prim_ids", &ReflEpc::resolved_prim_ids)
            .def_ro("surface_group_ids", &ReflEpc::surface_group_ids)
            .def_ro("plane_normals", &ReflEpc::plane_normals)
            .def_ro("first_blocked_segment", &ReflEpc::first_blocked_segment)
            .def_ro("first_blocked_prim", &ReflEpc::first_blocked_prim)
            .def_ro("first_blocked_group", &ReflEpc::first_blocked_group);

        nb::class_<ReflEpcAD>(m, "ReflEpcAD")
            .def_ro("ray_count", &ReflEpcAD::ray_count)
            .def_ro("max_bounces", &ReflEpcAD::max_bounces)
            .def_ro("valid", &ReflEpcAD::valid)
            .def_ro("bounce_count", &ReflEpcAD::bounce_count)
            .def_ro("path_length", &ReflEpcAD::path_length)
            .def_ro("reflection_points", &ReflEpcAD::reflection_points)
            .def_ro("prim_ids", &ReflEpcAD::prim_ids)
            .def_ro("trace_prim_ids", &ReflEpcAD::trace_prim_ids)
            .def_ro("resolved_prim_ids", &ReflEpcAD::resolved_prim_ids)
            .def_ro("surface_group_ids", &ReflEpcAD::surface_group_ids)
            .def_ro("plane_normals", &ReflEpcAD::plane_normals)
            .def_ro("first_blocked_segment", &ReflEpcAD::first_blocked_segment)
            .def_ro("first_blocked_prim", &ReflEpcAD::first_blocked_prim)
            .def_ro("first_blocked_group", &ReflEpcAD::first_blocked_group);

        nb::class_<ReflEpcField>(
                m, "ReflEpcField")
            .def_ro("ray_count", &ReflEpcField::ray_count)
            .def_ro("max_bounces", &ReflEpcField::max_bounces)
            .def_ro("valid", &ReflEpcField::valid)
            .def_ro("bounce_count", &ReflEpcField::bounce_count)
            .def_ro("path_length", &ReflEpcField::path_length)
            .def_ro("field_x_re", &ReflEpcField::field_x_re)
            .def_ro("field_x_im", &ReflEpcField::field_x_im)
            .def_ro("field_y_re", &ReflEpcField::field_y_re)
            .def_ro("field_y_im", &ReflEpcField::field_y_im)
            .def_ro("field_z_re", &ReflEpcField::field_z_re)
            .def_ro("field_z_im", &ReflEpcField::field_z_im)
            .def_ro("tx_pos", &ReflEpcField::tx_pos)
            .def_ro("first_hit", &ReflEpcField::first_hit)
            .def_ro("last_hit", &ReflEpcField::last_hit)
            .def_ro("hit_points", &ReflEpcField::hit_points)
            .def_ro("normals", &ReflEpcField::normals)
            .def_ro("resolved_prim_ids",
                    &ReflEpcField::resolved_prim_ids)
            .def_ro("surface_group_ids",
                    &ReflEpcField::surface_group_ids);

        nb::class_<ReflEpcFieldAD>(m, "ReflEpcFieldAD")
            .def_ro("ray_count", &ReflEpcFieldAD::ray_count)
            .def_ro("max_bounces", &ReflEpcFieldAD::max_bounces)
            .def_ro("valid", &ReflEpcFieldAD::valid)
            .def_ro("bounce_count", &ReflEpcFieldAD::bounce_count)
            .def_ro("path_length", &ReflEpcFieldAD::path_length)
            .def_ro("field_x_re", &ReflEpcFieldAD::field_x_re)
            .def_ro("field_x_im", &ReflEpcFieldAD::field_x_im)
            .def_ro("field_y_re", &ReflEpcFieldAD::field_y_re)
            .def_ro("field_y_im", &ReflEpcFieldAD::field_y_im)
            .def_ro("field_z_re", &ReflEpcFieldAD::field_z_re)
            .def_ro("field_z_im", &ReflEpcFieldAD::field_z_im)
            .def_ro("tx_pos", &ReflEpcFieldAD::tx_pos)
            .def_ro("first_hit", &ReflEpcFieldAD::first_hit)
            .def_ro("last_hit", &ReflEpcFieldAD::last_hit)
            .def_ro("hit_points", &ReflEpcFieldAD::hit_points)
            .def_ro("normals", &ReflEpcFieldAD::normals)
            .def_ro("resolved_prim_ids", &ReflEpcFieldAD::resolved_prim_ids)
            .def_ro("surface_group_ids", &ReflEpcFieldAD::surface_group_ids);

        nb::class_<ReflectionBounce>(m, "ReflectionBounce")
            .def("is_valid", &ReflectionBounce::is_valid)
            .def_ro("t", &ReflectionBounce::t)
            .def_ro("hit_points", &ReflectionBounce::hit_points)
            .def_ro("geo_normals", &ReflectionBounce::geo_normals)
            .def_ro("image_sources", &ReflectionBounce::image_sources)
            .def_ro("plane_points", &ReflectionBounce::plane_points)
            .def_ro("plane_normals", &ReflectionBounce::plane_normals)
            .def_ro("shape_ids", &ReflectionBounce::shape_ids)
            .def_ro("prim_ids", &ReflectionBounce::prim_ids)
            .def_ro("local_prim_ids", &ReflectionBounce::local_prim_ids)
            .def_ro("global_prim_ids", &ReflectionBounce::global_prim_ids);

        nb::class_<ReflectionBounceAD>(m, "ReflectionBounceAD")
            .def("is_valid", &ReflectionBounceAD::is_valid)
            .def_ro("t", &ReflectionBounceAD::t)
            .def_ro("hit_points", &ReflectionBounceAD::hit_points)
            .def_ro("geo_normals", &ReflectionBounceAD::geo_normals)
            .def_ro("image_sources", &ReflectionBounceAD::image_sources)
            .def_ro("plane_points", &ReflectionBounceAD::plane_points)
            .def_ro("plane_normals", &ReflectionBounceAD::plane_normals)
            .def_ro("shape_ids", &ReflectionBounceAD::shape_ids)
            .def_ro("prim_ids", &ReflectionBounceAD::prim_ids)
            .def_ro("local_prim_ids", &ReflectionBounceAD::local_prim_ids)
            .def_ro("global_prim_ids", &ReflectionBounceAD::global_prim_ids);

        nb::class_<ReflectionTrace>(m, "ReflectionTrace")
            .def("is_valid", &ReflectionTrace::is_valid)
            .def("bounce",
                 [](const ReflectionTrace &trace, int index) {
                     if (index < 0 || index >= trace.max_bounces) {
                         throw std::out_of_range("ReflectionTrace.bounce(): index out of range.");
                     }
                     return trace.bounces[static_cast<size_t>(index)];
                 },
                 "index"_a)
            .def_ro("max_bounces", &ReflectionTrace::max_bounces)
            .def_ro("ray_count", &ReflectionTrace::ray_count)
            .def_ro("deduplicate_requested", &ReflectionTrace::deduplicate_requested)
            .def_ro("deduplicate_applied", &ReflectionTrace::deduplicate_applied)
            .def_ro("bounce_count", &ReflectionTrace::bounce_count)
            .def_ro("discovery_count", &ReflectionTrace::discovery_count)
            .def_ro("representative_ray_index", &ReflectionTrace::representative_ray_index)
            .def_ro("dedup_keep_mask", &ReflectionTrace::dedup_keep_mask)
            .def_ro("bounces", &ReflectionTrace::bounces);

        nb::class_<ReflectionTraceAD>(m, "ReflectionTraceAD")
            .def("is_valid", &ReflectionTraceAD::is_valid)
            .def("bounce",
                 [](const ReflectionTraceAD &trace, int index) {
                     if (index < 0 || index >= trace.max_bounces) {
                         throw std::out_of_range("ReflectionTraceAD.bounce(): index out of range.");
                     }
                     return trace.bounces[static_cast<size_t>(index)];
                 },
                 "index"_a)
            .def_ro("max_bounces", &ReflectionTraceAD::max_bounces)
            .def_ro("ray_count", &ReflectionTraceAD::ray_count)
            .def_ro("deduplicate_requested", &ReflectionTraceAD::deduplicate_requested)
            .def_ro("deduplicate_applied", &ReflectionTraceAD::deduplicate_applied)
            .def_ro("bounce_count", &ReflectionTraceAD::bounce_count)
            .def_ro("discovery_count", &ReflectionTraceAD::discovery_count)
            .def_ro("representative_ray_index", &ReflectionTraceAD::representative_ray_index)
            .def_ro("dedup_keep_mask", &ReflectionTraceAD::dedup_keep_mask)
            .def_ro("bounces", &ReflectionTraceAD::bounces);

        nb::class_<NearestPointEdge>(m, "NearestPointEdge")
            .def("is_valid", &NearestPointEdge::is_valid)
            .def_ro("distance", &NearestPointEdge::distance)
            .def_ro("point", &NearestPointEdge::point)
            .def_ro("edge_t", &NearestPointEdge::edge_t)
            .def_ro("edge_point", &NearestPointEdge::edge_point)
            .def_ro("shape_id", &NearestPointEdge::shape_id)
            .def_ro("edge_id", &NearestPointEdge::edge_id)
            .def_ro("global_edge_id", &NearestPointEdge::global_edge_id)
            .def_ro("is_boundary", &NearestPointEdge::is_boundary);

        nb::class_<NearestPointEdgeAD>(m, "NearestPointEdgeAD")
            .def("is_valid", &NearestPointEdgeAD::is_valid)
            .def_ro("distance", &NearestPointEdgeAD::distance)
            .def_ro("point", &NearestPointEdgeAD::point)
            .def_ro("edge_t", &NearestPointEdgeAD::edge_t)
            .def_ro("edge_point", &NearestPointEdgeAD::edge_point)
            .def_ro("shape_id", &NearestPointEdgeAD::shape_id)
            .def_ro("edge_id", &NearestPointEdgeAD::edge_id)
            .def_ro("global_edge_id", &NearestPointEdgeAD::global_edge_id)
            .def_ro("is_boundary", &NearestPointEdgeAD::is_boundary);

        nb::class_<NearestRayEdge>(m, "NearestRayEdge")
            .def("is_valid", &NearestRayEdge::is_valid)
            .def_ro("distance", &NearestRayEdge::distance)
            .def_ro("ray_t", &NearestRayEdge::ray_t)
            .def_ro("point", &NearestRayEdge::point)
            .def_ro("edge_t", &NearestRayEdge::edge_t)
            .def_ro("edge_point", &NearestRayEdge::edge_point)
            .def_ro("shape_id", &NearestRayEdge::shape_id)
            .def_ro("edge_id", &NearestRayEdge::edge_id)
            .def_ro("global_edge_id", &NearestRayEdge::global_edge_id)
            .def_ro("is_boundary", &NearestRayEdge::is_boundary);

        nb::class_<NearestRayEdgeAD>(m, "NearestRayEdgeAD")
            .def("is_valid", &NearestRayEdgeAD::is_valid)
            .def_ro("distance", &NearestRayEdgeAD::distance)
            .def_ro("ray_t", &NearestRayEdgeAD::ray_t)
            .def_ro("point", &NearestRayEdgeAD::point)
            .def_ro("edge_t", &NearestRayEdgeAD::edge_t)
            .def_ro("edge_point", &NearestRayEdgeAD::edge_point)
            .def_ro("shape_id", &NearestRayEdgeAD::shape_id)
            .def_ro("edge_id", &NearestRayEdgeAD::edge_id)
            .def_ro("global_edge_id", &NearestRayEdgeAD::global_edge_id)
            .def_ro("is_boundary", &NearestRayEdgeAD::is_boundary);

        nb::class_<NearestEdgesTopK>(m, "NearestEdgesTopK")
            .def_ro("query_count", &NearestEdgesTopK::query_count)
            .def_ro("k", &NearestEdgesTopK::k)
            .def_ro("is_valid", &NearestEdgesTopK::is_valid)
            .def_ro("distances", &NearestEdgesTopK::distances)
            .def_ro("points", &NearestEdgesTopK::points)
            .def_ro("edge_t", &NearestEdgesTopK::edge_t)
            .def_ro("edge_points", &NearestEdgesTopK::edge_points)
            .def_ro("shape_ids", &NearestEdgesTopK::shape_ids)
            .def_ro("edge_ids", &NearestEdgesTopK::edge_ids)
            .def_ro("global_edge_ids", &NearestEdgesTopK::global_edge_ids)
            .def_ro("is_boundary", &NearestEdgesTopK::is_boundary);

        nb::class_<NearestEdgesTopKAD>(m, "NearestEdgesTopKAD")
            .def_ro("query_count", &NearestEdgesTopKAD::query_count)
            .def_ro("k", &NearestEdgesTopKAD::k)
            .def_ro("is_valid", &NearestEdgesTopKAD::is_valid)
            .def_ro("distances", &NearestEdgesTopKAD::distances)
            .def_ro("points", &NearestEdgesTopKAD::points)
            .def_ro("edge_t", &NearestEdgesTopKAD::edge_t)
            .def_ro("edge_points", &NearestEdgesTopKAD::edge_points)
            .def_ro("shape_ids", &NearestEdgesTopKAD::shape_ids)
            .def_ro("edge_ids", &NearestEdgesTopKAD::edge_ids)
            .def_ro("global_edge_ids", &NearestEdgesTopKAD::global_edge_ids)
            .def_ro("is_boundary", &NearestEdgesTopKAD::is_boundary);

        nb::class_<SegmentVisibility>(m, "SegmentVisibility")
            .def_ro("ray_count", &SegmentVisibility::ray_count)
            .def_ro("visible", &SegmentVisibility::visible);

        nb::class_<SegmentVisibilityAD>(m, "SegmentVisibilityAD")
            .def_ro("ray_count", &SegmentVisibilityAD::ray_count)
            .def_ro("visible", &SegmentVisibilityAD::visible);

        nb::class_<SegmentPairVisibility>(m, "SegmentPairVisibility")
            .def_ro("ray_count", &SegmentPairVisibility::ray_count)
            .def_ro("visible_a", &SegmentPairVisibility::visible_a)
            .def_ro("visible_b", &SegmentPairVisibility::visible_b);

        nb::class_<SegmentPairVisibilityAD>(m, "SegmentPairVisibilityAD")
            .def_ro("ray_count", &SegmentPairVisibilityAD::ray_count)
            .def_ro("visible_a", &SegmentPairVisibilityAD::visible_a)
            .def_ro("visible_b", &SegmentPairVisibilityAD::visible_b);

        nb::class_<AxialEdgeVisibility>(m, "AxialEdgeVisibility")
            .def_ro("state_count", &AxialEdgeVisibility::state_count)
            .def_ro("any_visible", &AxialEdgeVisibility::any_visible);

        nb::class_<AxialEdgeVisibilityAD>(m, "AxialEdgeVisibilityAD")
            .def_ro("state_count", &AxialEdgeVisibilityAD::state_count)
            .def_ro("any_visible", &AxialEdgeVisibilityAD::any_visible);

        nb::class_<SegmentChainVisibility>(m, "SegmentChainVisibility")
            .def_ro("chain_count", &SegmentChainVisibility::chain_count)
            .def_ro("max_segments", &SegmentChainVisibility::max_segments)
            .def_ro("all_visible", &SegmentChainVisibility::all_visible)
            .def_ro("first_blocked_segment",
                    &SegmentChainVisibility::first_blocked_segment)
            .def_ro("first_blocked_prim",
                    &SegmentChainVisibility::first_blocked_prim);

        nb::class_<SegmentChainVisibilityAD>(m, "SegmentChainVisibilityAD")
            .def_ro("chain_count", &SegmentChainVisibilityAD::chain_count)
            .def_ro("max_segments", &SegmentChainVisibilityAD::max_segments)
            .def_ro("all_visible", &SegmentChainVisibilityAD::all_visible)
            .def_ro("first_blocked_segment",
                    &SegmentChainVisibilityAD::first_blocked_segment)
            .def_ro("first_blocked_prim",
                    &SegmentChainVisibilityAD::first_blocked_prim);

        nb::class_<SceneSyncProfile>(m, "SceneSyncProfile")
            .def_ro("mesh_update_ms", &SceneSyncProfile::mesh_update_ms)
            .def_ro("triangle_scatter_ms", &SceneSyncProfile::triangle_scatter_ms)
            .def_ro("triangle_eval_ms", &SceneSyncProfile::triangle_eval_ms)
            .def_ro("edge_scatter_ms", &SceneSyncProfile::edge_scatter_ms)
            .def_ro("edge_refit_ms", &SceneSyncProfile::edge_refit_ms)
            .def_ro("optix_sync_ms", &SceneSyncProfile::optix_sync_ms)
            .def_ro("total_ms", &SceneSyncProfile::total_ms)
            .def_ro("optix_gas_update_ms", &SceneSyncProfile::optix_gas_update_ms)
            .def_ro("optix_ias_update_ms", &SceneSyncProfile::optix_ias_update_ms)
            .def_ro("updated_meshes", &SceneSyncProfile::updated_meshes)
            .def_ro("updated_vertex_meshes", &SceneSyncProfile::updated_vertex_meshes)
            .def_ro("updated_transform_meshes", &SceneSyncProfile::updated_transform_meshes)
            .def_ro("updated_edge_meshes", &SceneSyncProfile::updated_edge_meshes)
            .def_ro("updated_edges", &SceneSyncProfile::updated_edges);

        nb::class_<SceneEdgeBVHStats>(m, "SceneEdgeBVHStats")
            .def_ro("primitive_count", &SceneEdgeBVHStats::primitive_count)
            .def_ro("node_count", &SceneEdgeBVHStats::node_count)
            .def_ro("internal_node_count", &SceneEdgeBVHStats::internal_node_count)
            .def_ro("leaf_node_count", &SceneEdgeBVHStats::leaf_node_count)
            .def_ro("max_height", &SceneEdgeBVHStats::max_height)
            .def_ro("refit_level_count", &SceneEdgeBVHStats::refit_level_count)
            .def_ro("min_leaf_size", &SceneEdgeBVHStats::min_leaf_size)
            .def_ro("max_leaf_size", &SceneEdgeBVHStats::max_leaf_size)
            .def_ro("avg_leaf_size", &SceneEdgeBVHStats::avg_leaf_size)
            .def_ro("root_surface_area", &SceneEdgeBVHStats::root_surface_area)
            .def_ro("internal_surface_area_sum", &SceneEdgeBVHStats::internal_surface_area_sum)
            .def_ro("sibling_overlap_surface_area_sum",
                    &SceneEdgeBVHStats::sibling_overlap_surface_area_sum)
            .def_ro("sibling_overlap_surface_area_avg",
                    &SceneEdgeBVHStats::sibling_overlap_surface_area_avg)
            .def_ro("normalized_sibling_overlap",
                    &SceneEdgeBVHStats::normalized_sibling_overlap)
            .def_ro("leaf_size_histogram", &SceneEdgeBVHStats::leaf_size_histogram);
    });

    bind_section("mesh", [&]() {
        nb::class_<Mesh>(m, "Mesh")
            .def(nb::init<>())
            .def("__init__",
                 [](Mesh *mesh,
                    nb::handle v_obj,
                    const Vector3i &f,
                    nb::handle uv_obj,
                    const Vector3i &f_uv,
                    bool verbose) {
                     const std::string v_module_name =
                         nb::cast<std::string>(v_obj.type().attr("__module__"));
                     const std::string v_type_name =
                         nb::cast<std::string>(v_obj.type().attr("__name__"));
                     const std::string uv_module_name =
                         nb::cast<std::string>(uv_obj.type().attr("__module__"));
                     const std::string uv_type_name =
                         nb::cast<std::string>(uv_obj.type().attr("__name__"));

                     const bool has_live_v =
                         v_module_name == "drjit.cuda.ad" && v_type_name == "Array3f";
                     const bool has_live_uv =
                         uv_module_name == "drjit.cuda.ad" && uv_type_name == "Array2f";
                     if (!has_live_v && !has_live_uv) {
                         throw nb::next_overload();
                     }

                     Vector3f v_detached;
                     if (has_live_v) {
                         const Vector3fAD v_live = nb::cast<Vector3fAD>(v_obj);
                         v_detached = detach<false>(v_live);
                     } else if (!drjit_try_load<Vector3f>(v_obj, v_detached, true)) {
                         throw nb::next_overload();
                     }

                     Vector2f uv_detached;
                     if (has_live_uv) {
                         const Vector2fAD uv_live = nb::cast<Vector2fAD>(uv_obj);
                         uv_detached = detach<false>(uv_live);
                     } else if (!drjit_try_load<Vector2f>(uv_obj, uv_detached, true)) {
                         throw nb::next_overload();
                     }

                     new (mesh) Mesh(v_detached, f, uv_detached, f_uv, verbose);
                     if (has_live_v) {
                         const Vector3fAD v_live = nb::cast<Vector3fAD>(v_obj);
                         mesh->set_vertex_positions(v_live);
                     }
                     if (has_live_uv) {
                         const Vector2fAD uv_live = nb::cast<Vector2fAD>(uv_obj);
                         mesh->set_vertex_uv(uv_live);
                     }
                 },
                 "v"_a,
                 "f"_a,
                 "uv"_a = Vector2f(),
                 "f_uv"_a = Vector3i(),
                 "verbose"_a = false)
            .def("__init__",
                 [](Mesh *mesh,
                    const Vector3f &v,
                    const Vector3i &f,
                    const Vector2f &uv,
                    const Vector3i &f_uv,
                    bool verbose) {
                     new (mesh) Mesh(v, f, uv, f_uv, verbose);
                 },
                 "v"_a,
                 "f"_a,
                 "uv"_a = Vector2f(),
                 "f_uv"_a = Vector3i(),
                 "verbose"_a = false)
            .def("build", &Mesh::build)
            .def("set_transform", &Mesh::set_transform, "mat"_a, "set_left"_a = true)
            .def("append_transform", &Mesh::append_transform, "mat"_a, "append_left"_a = true)
            .def("edge_indices", [](const Mesh &mesh) {
                const auto &edge_indices = mesh.edge_indices();
                return nb::make_tuple(edge_indices[0],
                                      edge_indices[1],
                                      edge_indices[2],
                                      edge_indices[3],
                                      edge_indices[4]);
            })
            .def("secondary_edges", [](const Mesh &mesh) {
                return mesh.secondary_edge_info() != nullptr ? *mesh.secondary_edge_info() : SecondaryEdgeInfoAD();
            })
            .def_prop_ro("num_vertices", &Mesh::vertex_count)
            .def_prop_ro("num_faces", &Mesh::face_count)
            .def_prop_rw("to_world", &Mesh::to_world, &Mesh::set_to_world)
            .def_prop_rw("to_world_left", &Mesh::to_world_left, &Mesh::set_to_world_left)
            .def_prop_rw("to_world_right", &Mesh::to_world_right, &Mesh::set_to_world_right)
            .def_prop_rw("vertex_positions", &Mesh::vertex_positions, &Mesh::set_vertex_positions)
            .def_prop_ro("vertex_positions_world", &Mesh::vertex_positions_world)
            .def_prop_ro("vertex_normals", &Mesh::vertex_normals)
            .def_prop_rw("vertex_uv", &Mesh::vertex_uv, &Mesh::set_vertex_uv)
            .def_prop_rw("face_indices", &Mesh::face_indices, &Mesh::set_face_indices)
            .def_prop_rw("face_uv_indices", &Mesh::face_uv_indices, &Mesh::set_face_uv_indices)
            .def_prop_rw("use_face_normals", &Mesh::use_face_normals, &Mesh::set_use_face_normals)
            .def_prop_rw("edges_enabled", &Mesh::edges_enabled, &Mesh::set_edges_enabled)
            .def("__repr__", &Mesh::to_string);
    });

    bind_section("scene", [&]() {
        nb::class_<Scene>(m, "Scene")
            .def(nb::init<const std::string &>(), "edge_bvh_backend"_a = "optix")
            .def("add_mesh", &Scene::add_mesh, "mesh"_a, "dynamic"_a = false)
            .def("build", &Scene::build)
            .def("update_mesh_vertices", &Scene::update_mesh_vertices, "mesh_id"_a, "positions"_a)
            .def("set_mesh_transform", &Scene::set_mesh_transform, "mesh_id"_a, "mat"_a, "set_left"_a = true)
            .def("append_mesh_transform", &Scene::append_mesh_transform, "mesh_id"_a, "mat"_a, "append_left"_a = true)
            .def("set_edge_mask",
                 nb::overload_cast<const rayd::Mask &>(&Scene::set_edge_mask),
                 nb::arg("mask"))
            .def("set_edge_mask",
                 nb::overload_cast<const rayd::MaskAD &>(&Scene::set_edge_mask),
                 nb::arg("mask"))
            .def("sync", &Scene::sync)
            .def("is_ready", &Scene::is_ready)
            .def("has_pending_updates", &Scene::has_pending_updates)
            .def_prop_ro("last_sync_profile", &Scene::last_sync_profile)
            .def("edge_info", &Scene::edge_info)
            .def_prop_ro("edge_bvh_backend", &Scene::edge_bvh_backend)
            .def("edge_bvh_stats", &Scene::edge_bvh_stats)
            .def("edge_topology", &Scene::edge_topology)
            .def("edge_mask", &Scene::edge_mask)
            .def("mesh_face_offsets", &Scene::mesh_face_offsets)
            .def("mesh_edge_offsets", &Scene::mesh_edge_offsets)
            .def("mesh_vertex_offsets", &Scene::mesh_vertex_offsets)
            .def("global_geometry", &Scene::global_geometry)
            .def("triangle_edge_indices",
                 [](const Scene &scene, const Int &prim_id, bool global) {
                     const auto edge_ids = scene.triangle_edge_indices(prim_id, global);
                     return nb::make_tuple(edge_ids[0], edge_ids[1], edge_ids[2]);
                 },
                 "prim_id"_a,
                 "global_"_a = true)
            .def("edge_adjacent_faces",
                 [](const Scene &scene, const Int &edge_id, bool global) {
                     const auto face_ids = scene.edge_adjacent_faces(edge_id, global);
                     return nb::make_tuple(face_ids[0], face_ids[1]);
                 },
                 "edge_id"_a,
                 "global_"_a = true)
            .def("intersect",
                 [](const Scene &scene, const Ray &ray, rayd::Mask active, RayFlags flags) {
                     return scene.intersect<true>(ray, active, flags);
                 },
                 nb::arg("ray").noconvert(), "active"_a = true, "flags"_a = RayFlags::All)
            .def("intersect",
                 [](const Scene &scene, const RayAD &ray, rayd::MaskAD active, RayFlags flags) {
                     return scene.intersect<false>(ray, active, flags);
                 },
                 nb::arg("ray").noconvert(), "active"_a = true, "flags"_a = RayFlags::All)
            .def("trace_reflections",
                 [](const Scene &scene,
                    const Ray &ray,
                    int max_bounces,
                    rayd::Mask active,
                    bool symbolic) -> nb::object {
                     if (symbolic) {
                         return nb::cast(scene.trace_bounces<true>(ray, max_bounces, active));
                     }
                     return nb::cast(scene.trace_reflections<true>(ray, max_bounces, active));
                 },
                 nb::arg("ray").noconvert(), "max_bounces"_a, "active"_a = true, "symbolic"_a = true)
            .def("trace_reflections",
                 [](const Scene &scene,
                    const RayAD &ray,
                    int max_bounces,
                    rayd::MaskAD active,
                    bool symbolic) -> nb::object {
                     if (symbolic) {
                         return nb::cast(scene.trace_bounces<false>(ray, max_bounces, active));
                     }
                     return nb::cast(scene.trace_reflections<false>(ray, max_bounces, active));
                 },
                 nb::arg("ray").noconvert(), "max_bounces"_a, "active"_a = true, "symbolic"_a = true)
            .def("trace_reflections",
                 [](const Scene &scene,
                    const Ray &ray,
                    int max_bounces,
                    const ReflectionTraceOptions &options,
                    rayd::Mask active,
                    bool symbolic) -> nb::object {
                     if (symbolic) {
                         return nb::cast(scene.trace_bounces<true>(ray, max_bounces, options, active));
                     }
                     return nb::cast(scene.trace_reflections<true>(ray, max_bounces, options, active));
                 },
                 nb::arg("ray").noconvert(),
                 "max_bounces"_a,
                 "options"_a,
                 "active"_a = true,
                 "symbolic"_a = true)
            .def("trace_reflections",
                 [](const Scene &scene,
                    const RayAD &ray,
                    int max_bounces,
                    const ReflectionTraceOptions &options,
                    rayd::MaskAD active,
                    bool symbolic) -> nb::object {
                     if (symbolic) {
                         return nb::cast(scene.trace_bounces<false>(ray, max_bounces, options, active));
                     }
                     return nb::cast(scene.trace_reflections<false>(ray, max_bounces, options, active));
                 },
                 nb::arg("ray").noconvert(),
                 "max_bounces"_a,
                 "options"_a,
                 "active"_a = true,
                 "symbolic"_a = true)
            .def("trace_reflections",
                 [](const Scene &scene,
                    const Ray &ray,
                    int max_bounces,
                    bool deduplicate,
                    const Int &canonical_prim_table,
                    float image_source_tolerance,
                    rayd::Mask active,
                    bool symbolic) -> nb::object {
                     ReflectionTraceOptions options;
                     options.deduplicate = deduplicate;
                     options.canonical_prim_table = canonical_prim_table;
                     options.image_source_tolerance = image_source_tolerance;
                     if (symbolic) {
                         return nb::cast(scene.trace_bounces<true>(ray, max_bounces, options, active));
                     }
                     return nb::cast(scene.trace_reflections<true>(ray, max_bounces, options, active));
                 },
                 nb::arg("ray").noconvert(),
                 "max_bounces"_a,
                 "deduplicate"_a = false,
                 "canonical_prim_table"_a = Int(),
                 "image_source_tolerance"_a = 1e-5f,
                 "active"_a = true,
                 "symbolic"_a = true)
            .def("trace_reflections",
                 [](const Scene &scene,
                    const RayAD &ray,
                    int max_bounces,
                    bool deduplicate,
                    const Int &canonical_prim_table,
                    float image_source_tolerance,
                    rayd::MaskAD active,
                    bool symbolic) -> nb::object {
                     ReflectionTraceOptions options;
                     options.deduplicate = deduplicate;
                     options.canonical_prim_table = canonical_prim_table;
                     options.image_source_tolerance = image_source_tolerance;
                     if (symbolic) {
                         return nb::cast(scene.trace_bounces<false>(ray, max_bounces, options, active));
                     }
                     return nb::cast(scene.trace_reflections<false>(ray, max_bounces, options, active));
                 },
                 nb::arg("ray").noconvert(),
                 "max_bounces"_a,
                 "deduplicate"_a = false,
                 "canonical_prim_table"_a = Int(),
                  "image_source_tolerance"_a = 1e-5f,
                  "active"_a = true,
                  "symbolic"_a = true)
            .def("trace_refl_epc",
                 [](const Scene &scene,
                    const Ray &ray,
                    const Vector3f &receiver,
                    int max_bounces,
                    nb::object options_obj,
                    rayd::Mask active) {
                     ReflEpcOptions options;
                     if (!options_obj.is_none()) {
                         bool parsed_options = false;
                         if (nb::inst_check(options_obj)) {
                             try {
                                 if (nb::type_info(options_obj.type()) ==
                                     typeid(ReflEpcOptions)) {
                                     options =
                                         *nb::inst_ptr<ReflEpcOptions>(options_obj);
                                     parsed_options = true;
                                 }
                             } catch (...) {
                                 PyErr_Clear();
                             }
                         }
                         if (!parsed_options) {
                             active = nb::cast<rayd::Mask>(options_obj);
                         }
                     }
                     return scene.trace_refl_epc<true>(
                          ray, receiver, max_bounces, options, active);
                  },
                  nb::arg("ray").noconvert(),
                  nb::arg("receiver"),
                  "max_bounces"_a,
                  "options"_a = nb::none(),
                  "active"_a = rayd::Mask(true))
            .def("trace_refl_epc_field",
                 [](const Scene &scene,
                    nb::handle source_obj,
                    nb::handle receiver_obj,
                    int max_bounces,
                    const ReflEpcFieldOptions &options,
                    nb::handle active_obj) -> nb::object {
                     auto is_ad_drjit = [](nb::handle obj) {
                         const std::string type_name =
                             nb::cast<std::string>(nb::str(obj.type()));
                         return type_name.find("drjit.cuda.ad.") != std::string::npos;
                     };
                      if (nb::isinstance<RayAD>(source_obj)) {
                          const RayAD ray = nb::cast<RayAD>(source_obj);
                          const Vector3fAD receiver =
                              nb::cast<Vector3fAD>(receiver_obj);
                          const rayd::MaskAD active =
                              nb::cast<rayd::MaskAD>(active_obj);
                          const ReflEpcFieldOptionsAD options_ad =
                              refl_epc_field_options_ad_from_detached(options);
                          return nb::cast(scene.trace_refl_epc_field<false>(
                              ray, receiver, max_bounces, options_ad, active));
                      }
                      if (is_ad_drjit(source_obj) ||
                          is_ad_drjit(receiver_obj) ||
                          is_ad_drjit(active_obj)) {
                         const Vector3fAD tx_position =
                             nb::cast<Vector3fAD>(source_obj);
                          const Vector3fAD receiver =
                              nb::cast<Vector3fAD>(receiver_obj);
                          const rayd::MaskAD active =
                              nb::cast<rayd::MaskAD>(active_obj);
                          const ReflEpcFieldOptionsAD options_ad =
                              refl_epc_field_options_ad_from_detached(options);
                          return nb::cast(scene.trace_refl_epc_field<false>(
                              tx_position, receiver, max_bounces, options_ad, active));
                      }
                     throw nb::next_overload();
                 },
                 nb::arg("source"),
                  nb::arg("receiver"),
                  "max_bounces"_a,
                  "options"_a,
                  "active"_a = true)
            .def("trace_refl_epc_field",
                 [](const Scene &scene,
                    nb::handle source_obj,
                    nb::handle receiver_obj,
                    int max_bounces,
                    const ReflEpcFieldOptionsAD &options,
                    nb::handle active_obj) -> nb::object {
                     auto is_ad_drjit = [](nb::handle obj) {
                         const std::string type_name =
                             nb::cast<std::string>(nb::str(obj.type()));
                         return type_name.find("drjit.cuda.ad.") != std::string::npos;
                     };
                     if (nb::isinstance<RayAD>(source_obj)) {
                         const RayAD ray = nb::cast<RayAD>(source_obj);
                         const Vector3fAD receiver =
                             nb::cast<Vector3fAD>(receiver_obj);
                         const rayd::MaskAD active =
                             nb::cast<rayd::MaskAD>(active_obj);
                         return nb::cast(scene.trace_refl_epc_field<false>(
                             ray, receiver, max_bounces, options, active));
                     }
                     if (is_ad_drjit(source_obj) ||
                         is_ad_drjit(receiver_obj) ||
                         is_ad_drjit(active_obj)) {
                         const Vector3fAD tx_position =
                             nb::cast<Vector3fAD>(source_obj);
                         const Vector3fAD receiver =
                             nb::cast<Vector3fAD>(receiver_obj);
                         const rayd::MaskAD active =
                             nb::cast<rayd::MaskAD>(active_obj);
                         return nb::cast(scene.trace_refl_epc_field<false>(
                             tx_position, receiver, max_bounces, options, active));
                     }
                     throw nb::next_overload();
                 },
                 nb::arg("source"),
                 nb::arg("receiver"),
                 "max_bounces"_a,
                 "options"_a,
                 "active"_a = true)
            .def("trace_refl_epc_field",
                 [](const Scene &scene,
                    const RayAD &ray,
                     const Vector3fAD &receiver,
                     int max_bounces,
                     const ReflEpcFieldOptionsAD &options,
                    rayd::MaskAD active) {
                     return scene.trace_refl_epc_field<false>(
                         ray, receiver, max_bounces, options, active);
                 },
                 nb::arg("ray").noconvert(),
                 nb::arg("receiver").noconvert(),
                  "max_bounces"_a,
                  "options"_a,
                  "active"_a = rayd::MaskAD(true))
            .def("trace_refl_epc_field",
                 [](const Scene &scene,
                     const Vector3fAD &tx_position,
                     const Vector3fAD &receiver,
                     int max_bounces,
                     const ReflEpcFieldOptionsAD &options,
                    rayd::MaskAD active) {
                     return scene.trace_refl_epc_field<false>(
                         tx_position, receiver, max_bounces, options, active);
                 },
                 nb::arg("tx_position").noconvert(),
                 nb::arg("receiver").noconvert(),
                 "max_bounces"_a,
                 "options"_a,
                 "active"_a = rayd::MaskAD(true))
            .def("trace_refl_epc_field",
                 [](const Scene &scene,
                    const Ray &ray,
                    const Vector3f &receiver,
                    int max_bounces,
                    const ReflEpcFieldOptions &options,
                    rayd::Mask active) {
                     return scene.trace_refl_epc_field<true>(
                         ray, receiver, max_bounces, options, active);
                 },
                 nb::arg("ray").noconvert(),
                 nb::arg("receiver"),
                  "max_bounces"_a,
                  "options"_a,
                  "active"_a = rayd::Mask(true))
            .def("trace_refl_epc_field",
                 [](const Scene &scene,
                    const Vector3f &tx_position,
                    const Vector3f &receiver,
                    int max_bounces,
                    const ReflEpcFieldOptions &options,
                    rayd::Mask active) {
                     return scene.trace_refl_epc_field<true>(
                         tx_position, receiver, max_bounces, options, active);
                 },
                 nb::arg("tx_position"),
                 nb::arg("receiver"),
                 "max_bounces"_a,
                 "options"_a,
                 "active"_a = rayd::Mask(true))
            .def("accumulate_reflections",
                 [](const Scene &scene,
                    nb::handle ray_obj,
                    nb::handle tx_position_obj,
                     const AccumGrid &grid,
                     nb::handle material_obj,
                     int max_bounces,
                     const AccumOptions &options,
                     nb::handle active_obj,
                     nb::handle tx_polarization_obj) -> nb::object {
                      if (nb::isinstance<Ray>(ray_obj)) {
                          const Ray ray = nb::cast<Ray>(ray_obj);
                          const Vector3f tx_position =
                              nb::cast<Vector3f>(tx_position_obj);
                          const Vector3f tx_polarization =
                              nb::cast<Vector3f>(tx_polarization_obj);
                          const Material material =
                              nb::cast<Material>(material_obj);
                          const rayd::Mask active =
                              nb::cast<rayd::Mask>(active_obj);
                          return nb::cast(scene.accumulate_reflections<true>(
                              ray, tx_position, grid, material, max_bounces, options, active,
                              tx_polarization));
                      }
                      if (nb::isinstance<RayAD>(ray_obj)) {
                          const RayAD ray = nb::cast<RayAD>(ray_obj);
                          const Vector3fAD tx_position =
                              nb::cast<Vector3fAD>(tx_position_obj);
                          const Vector3fAD tx_polarization =
                              nb::cast<Vector3fAD>(tx_polarization_obj);
                          const MaterialAD material =
                              nb::cast<MaterialAD>(material_obj);
                          const rayd::MaskAD active = nb::cast<rayd::MaskAD>(active_obj);
                          return nb::cast(scene.accumulate_reflections<false>(
                              ray, tx_position, grid, material, max_bounces, options, active,
                              tx_polarization));
                      }
                      throw nb::next_overload();
                  },
                 nb::arg("ray"),
                 nb::arg("tx_position"),
                 "grid"_a,
                  "material"_a,
                  "max_bounces"_a,
                  "options"_a = AccumOptions(),
                  "active"_a = true,
                  "tx_polarization"_a = Vector3f(1.f, 0.f, 0.f))
            .def("accum_dfr_direct",
                 [](const Scene &scene,
                    nb::handle states_obj,
                    const DfrGrid &grid,
                    nb::handle material_obj,
                    const DfrOptions &options,
                    nb::handle active_obj) -> nb::object {
                     if (nb::isinstance<DfrStates>(states_obj)) {
                         const DfrStates states =
                             nb::cast<DfrStates>(states_obj);
                         const DfrMaterial material =
                             nb::cast<DfrMaterial>(material_obj);
                         const rayd::Mask active = nb::cast<rayd::Mask>(active_obj);
                         return nb::cast(scene.accum_dfr_direct<true>(
                             states, grid, material, options, active));
                     }
                     if (nb::isinstance<DfrStatesAD>(states_obj)) {
                         const DfrStatesAD states =
                             nb::cast<DfrStatesAD>(states_obj);
                         const DfrMaterialAD material =
                             nb::cast<DfrMaterialAD>(material_obj);
                         const rayd::MaskAD active =
                             nb::cast<rayd::MaskAD>(active_obj);
                         return nb::cast(scene.accum_dfr_direct<false>(
                             states, grid, material, options, active));
                     }
                     throw nb::next_overload();
                 },
                 "states"_a,
                 "grid"_a,
                 "material"_a,
                  "options"_a = DfrOptions(),
                  "active"_a = true)
            .def("accum_dfr_coherent_direct",
                 [](const Scene &scene,
                    nb::handle states_obj,
                    const DfrGrid &grid,
                    nb::handle material_obj,
                    const DfrCoherentOptions &options,
                    nb::handle active_obj) -> nb::object {
                     if (nb::isinstance<DfrCoherentUtdStates>(states_obj)) {
                         const DfrCoherentUtdStates states =
                             nb::cast<DfrCoherentUtdStates>(states_obj);
                         const rayd::Mask active = nb::cast<rayd::Mask>(active_obj);
                         return nb::cast(scene.accum_dfr_coherent_direct<true>(
                             states, grid, options, active));
                     }
                     if (nb::isinstance<DfrCoherentUtdStatesAD>(states_obj)) {
                         const DfrCoherentUtdStatesAD states =
                             nb::cast<DfrCoherentUtdStatesAD>(states_obj);
                         const rayd::MaskAD active =
                             nb::cast<rayd::MaskAD>(active_obj);
                         return nb::cast(scene.accum_dfr_coherent_direct<false>(
                             states, grid, options, active));
                     }
                     if (nb::isinstance<DfrStates>(states_obj)) {
                         const DfrStates states =
                             nb::cast<DfrStates>(states_obj);
                         const DfrMaterial material =
                             nb::cast<DfrMaterial>(material_obj);
                         const rayd::Mask active = nb::cast<rayd::Mask>(active_obj);
                         return nb::cast(scene.accum_dfr_coherent_direct<true>(
                             states, grid, material, options, active));
                     }
                     if (nb::isinstance<DfrStatesAD>(states_obj)) {
                         const DfrStatesAD states =
                             nb::cast<DfrStatesAD>(states_obj);
                         const DfrMaterialAD material =
                             nb::cast<DfrMaterialAD>(material_obj);
                         const rayd::MaskAD active =
                             nb::cast<rayd::MaskAD>(active_obj);
                         return nb::cast(scene.accum_dfr_coherent_direct<false>(
                             states, grid, material, options, active));
                     }
                     throw nb::next_overload();
                 },
                 "states"_a,
                  "grid"_a,
                  "material"_a,
                  "options"_a = DfrCoherentOptions(),
                  "active"_a = true)
            .def("build_dfr_coherent_tx_states",
                 [](const Scene &scene,
                    nb::handle edges_obj,
                    nb::handle tx_position_obj,
                    nb::handle material_obj,
                    const DfrCoherentOptions &options,
                    nb::handle active_obj) -> nb::object {
                     if (nb::isinstance<DfrCoherentEdge>(edges_obj)) {
                         const DfrCoherentEdge edges =
                             nb::cast<DfrCoherentEdge>(edges_obj);
                         const Vector3f tx_position =
                             nb::cast<Vector3f>(tx_position_obj);
                         const DfrMaterial material =
                             nb::cast<DfrMaterial>(material_obj);
                         const rayd::Mask active = nb::cast<rayd::Mask>(active_obj);
                         return nb::cast(scene.build_dfr_coherent_tx_states<true>(
                             edges, tx_position, material, options, active));
                     }
                     if (nb::isinstance<DfrCoherentEdgeAD>(edges_obj)) {
                         const DfrCoherentEdgeAD edges =
                             nb::cast<DfrCoherentEdgeAD>(edges_obj);
                         const Vector3fAD tx_position =
                             nb::cast<Vector3fAD>(tx_position_obj);
                         const DfrMaterialAD material =
                             nb::cast<DfrMaterialAD>(material_obj);
                         const rayd::MaskAD active =
                             nb::cast<rayd::MaskAD>(active_obj);
                         return nb::cast(scene.build_dfr_coherent_tx_states<false>(
                             edges, tx_position, material, options, active));
                     }
                     throw nb::next_overload();
                 },
                 "edges"_a,
                 "tx_position"_a,
                 "material"_a,
                 "options"_a = DfrCoherentOptions(),
                  "active"_a = true)
            .def("build_dfr_coherent_higher_candidates",
                 [](const Scene &scene,
                    nb::handle prev_states_obj,
                    nb::handle edges_obj,
                    nb::handle global_to_local_edge_index_obj,
                    const DfrCoherentOptions &options,
                    nb::handle active_obj) -> nb::object {
                     if (nb::isinstance<DfrCoherentUtdStates>(prev_states_obj) &&
                         nb::isinstance<DfrCoherentEdge>(edges_obj)) {
                         const DfrCoherentUtdStates prev_states =
                             nb::cast<DfrCoherentUtdStates>(prev_states_obj);
                         const DfrCoherentEdge edges =
                             nb::cast<DfrCoherentEdge>(edges_obj);
                         const Int global_to_local_edge_index =
                             nb::cast<Int>(global_to_local_edge_index_obj);
                         const rayd::Mask active = nb::cast<rayd::Mask>(active_obj);
                         return nb::cast(
                             scene.build_dfr_coherent_higher_candidates<true>(
                                 prev_states,
                                 edges,
                                 global_to_local_edge_index,
                                 options,
                                 active));
                     }
                     if (nb::isinstance<DfrCoherentUtdStatesAD>(prev_states_obj) &&
                         nb::isinstance<DfrCoherentEdgeAD>(edges_obj)) {
                         const DfrCoherentUtdStatesAD prev_states =
                             nb::cast<DfrCoherentUtdStatesAD>(prev_states_obj);
                         const DfrCoherentEdgeAD edges =
                             nb::cast<DfrCoherentEdgeAD>(edges_obj);
                         const IntAD global_to_local_edge_index =
                             nb::cast<IntAD>(global_to_local_edge_index_obj);
                         const rayd::MaskAD active =
                             nb::cast<rayd::MaskAD>(active_obj);
                         return nb::cast(
                             scene.build_dfr_coherent_higher_candidates<false>(
                                 prev_states,
                                 edges,
                                 global_to_local_edge_index,
                                 options,
                                 active));
                     }
                     throw nb::next_overload();
                 },
                 "prev_states"_a,
                 "edges"_a,
                 "global_to_local_edge_index"_a,
                 "options"_a = DfrCoherentOptions(),
                 "active"_a = true)
            .def("accum_dfr",
                 [](const Scene &scene,
                    nb::handle initial_states_obj,
                    nb::handle recursive_states_obj,
                    const DfrGrid &grid,
                    nb::handle material_obj,
                    const DfrOptions &options,
                    nb::handle active_obj) -> nb::object {
                     if (nb::isinstance<DfrStates>(initial_states_obj)) {
                         const DfrStates initial_states =
                             nb::cast<DfrStates>(initial_states_obj);
                         const DfrStates recursive_states =
                             nb::cast<DfrStates>(recursive_states_obj);
                         const DfrMaterial material =
                             nb::cast<DfrMaterial>(material_obj);
                         const rayd::Mask active = nb::cast<rayd::Mask>(active_obj);
                         return nb::cast(scene.accum_dfr<true>(
                             initial_states, recursive_states, grid, material, options, active));
                     }
                     if (nb::isinstance<DfrStatesAD>(initial_states_obj)) {
                         const DfrStatesAD initial_states =
                             nb::cast<DfrStatesAD>(initial_states_obj);
                         const DfrStatesAD recursive_states =
                             nb::cast<DfrStatesAD>(recursive_states_obj);
                         const DfrMaterialAD material =
                             nb::cast<DfrMaterialAD>(material_obj);
                         const rayd::MaskAD active =
                             nb::cast<rayd::MaskAD>(active_obj);
                         return nb::cast(scene.accum_dfr<false>(
                             initial_states, recursive_states, grid, material, options, active));
                     }
                     throw nb::next_overload();
                 },
                 "initial_states"_a,
                 "recursive_states"_a,
                 "grid"_a,
                 "material"_a,
                 "options"_a = DfrOptions(),
                 "active"_a = true)
            .def("trace_dfr_paths",
                 [](const Scene &scene,
                    const Vector3f &tx_positions,
                    const Vector3f &rx_positions,
                    const DfrStates &states,
                    const DfrMaterial &material,
                    const DfrPathOptions &options,
                    rayd::Mask active) {
                     return scene.trace_dfr_paths<true>(
                         tx_positions, rx_positions, states, material, options, active);
                 },
                 "tx_positions"_a,
                 "rx_positions"_a,
                 "states"_a,
                 "material"_a,
                 "options"_a = DfrPathOptions(),
                 "active"_a = true)
            .def("trace_dfr_paths",
                 [](const Scene &scene,
                    const Vector3fAD &tx_positions,
                    const Vector3fAD &rx_positions,
                    const DfrStatesAD &states,
                    const DfrMaterialAD &material,
                    const DfrPathOptions &options,
                    rayd::MaskAD active) {
                     return scene.trace_dfr_paths<false>(
                         tx_positions, rx_positions, states, material, options, active);
                 },
                 "tx_positions"_a,
                 "rx_positions"_a,
                 "states"_a,
                 "material"_a,
                 "options"_a = DfrPathOptions(),
                 "active"_a = true)
            .def("shadow_test",
                 [](const Scene &scene, const Ray &ray, rayd::Mask active) {
                     return scene.shadow_test<true>(ray, active);
                 },
                 nb::arg("ray").noconvert(), "active"_a = true)
            .def("shadow_test",
                 [](const Scene &scene, const RayAD &ray, rayd::MaskAD active) {
                     return scene.shadow_test<false>(ray, active);
                 },
                 nb::arg("ray").noconvert(), "active"_a = true)
            .def("visible",
                 [](const Scene &scene,
                    nb::handle start_obj,
                    nb::handle end_obj,
                    const Int &ignore_prim_ids,
                    nb::handle active_obj) -> nb::object {
                     const std::string module_name =
                         nb::cast<std::string>(start_obj.type().attr("__module__"));
                     const std::string type_name =
                         nb::cast<std::string>(start_obj.type().attr("__name__"));

                     if (module_name == "drjit.cuda.ad" && type_name == "Array3f") {
                         Vector3fAD start = nb::cast<Vector3fAD>(start_obj);
                         Vector3fAD end = nb::cast<Vector3fAD>(end_obj);
                         rayd::MaskAD active = nb::cast<rayd::MaskAD>(active_obj);
                         return nb::cast(scene.visible<false>(
                             start, end, ignore_prim_ids, active));
                     }

                     if (module_name == "drjit.cuda" && type_name == "Array3f") {
                         Vector3f start = nb::cast<Vector3f>(start_obj);
                         Vector3f end = nb::cast<Vector3f>(end_obj);
                         rayd::Mask active = nb::cast<rayd::Mask>(active_obj);
                         return nb::cast(scene.visible<true>(
                             start, end, ignore_prim_ids, active));
                     }
                     throw nb::next_overload();
                 },
                 nb::arg("start"),
                 nb::arg("end"),
                 "ignore_prim_ids"_a = Int(),
                 "active"_a = true)
            .def("visible_pair",
                 [](const Scene &scene,
                    nb::handle start_obj,
                    nb::handle end_a_obj,
                    nb::handle end_b_obj,
                    const Int &ignore_prim_ids,
                    nb::handle active_obj) -> nb::object {
                     const std::string module_name =
                         nb::cast<std::string>(start_obj.type().attr("__module__"));
                     const std::string type_name =
                         nb::cast<std::string>(start_obj.type().attr("__name__"));

                     if (module_name == "drjit.cuda.ad" && type_name == "Array3f") {
                         Vector3fAD start = nb::cast<Vector3fAD>(start_obj);
                         Vector3fAD end_a = nb::cast<Vector3fAD>(end_a_obj);
                         Vector3fAD end_b = nb::cast<Vector3fAD>(end_b_obj);
                         rayd::MaskAD active = nb::cast<rayd::MaskAD>(active_obj);
                         return nb::cast(scene.visible_pair<false>(
                             start, end_a, end_b, ignore_prim_ids, active));
                     }

                     if (module_name == "drjit.cuda" && type_name == "Array3f") {
                         Vector3f start = nb::cast<Vector3f>(start_obj);
                         Vector3f end_a = nb::cast<Vector3f>(end_a_obj);
                         Vector3f end_b = nb::cast<Vector3f>(end_b_obj);
                         rayd::Mask active = nb::cast<rayd::Mask>(active_obj);
                         return nb::cast(scene.visible_pair<true>(
                             start, end_a, end_b, ignore_prim_ids, active));
                     }
                     throw nb::next_overload();
                 },
                 nb::arg("start"),
                 nb::arg("end_a"),
                 nb::arg("end_b"),
                 "ignore_prim_ids"_a = Int(),
                 "active"_a = true)
            .def("visible_edge",
                 [](const Scene &scene,
                    nb::handle src_obj,
                    nb::handle edge_pos_obj,
                    nb::handle edge_dir_obj,
                    nb::handle edge_t_min_obj,
                    nb::handle edge_t_max_obj,
                    const std::vector<float> &sample_fractions,
                    nb::handle active_obj) -> nb::object {
                     const std::string module_name =
                         nb::cast<std::string>(src_obj.type().attr("__module__"));
                     const std::string type_name =
                         nb::cast<std::string>(src_obj.type().attr("__name__"));

                     if (module_name == "drjit.cuda.ad" && type_name == "Array3f") {
                         Vector3fAD src = nb::cast<Vector3fAD>(src_obj);
                         Vector3fAD edge_pos = nb::cast<Vector3fAD>(edge_pos_obj);
                         Vector3fAD edge_dir = nb::cast<Vector3fAD>(edge_dir_obj);
                         FloatAD edge_t_min = nb::cast<FloatAD>(edge_t_min_obj);
                         FloatAD edge_t_max = nb::cast<FloatAD>(edge_t_max_obj);
                         rayd::MaskAD active = nb::cast<rayd::MaskAD>(active_obj);
                         return nb::cast(scene.visible_edge<false>(
                             src,
                             edge_pos,
                             edge_dir,
                             edge_t_min,
                             edge_t_max,
                             sample_fractions,
                             active));
                     }

                     if (module_name == "drjit.cuda" && type_name == "Array3f") {
                         Vector3f src =
                             nb::cast<Vector3f>(src_obj);
                         Vector3f edge_pos =
                             nb::cast<Vector3f>(edge_pos_obj);
                         Vector3f edge_dir =
                             nb::cast<Vector3f>(edge_dir_obj);
                         Float edge_t_min =
                             nb::cast<Float>(edge_t_min_obj);
                         Float edge_t_max =
                             nb::cast<Float>(edge_t_max_obj);
                         rayd::Mask active =
                             nb::cast<rayd::Mask>(active_obj);
                         return nb::cast(scene.visible_edge<true>(
                             src,
                             edge_pos,
                             edge_dir,
                             edge_t_min,
                             edge_t_max,
                             sample_fractions,
                             active));
                     }
                     throw nb::next_overload();
                 },
                 nb::arg("src"),
                 nb::arg("edge_pos"),
                 nb::arg("edge_dir"),
                 "edge_t_min"_a,
                 "edge_t_max"_a,
                 "sample_fractions"_a = std::vector<float>{ 0.f, 0.25f, 0.5f, 0.75f, 1.f },
                 "active"_a = true)
            .def("visible_chain",
                 [](const Scene &scene,
                    nb::handle points_obj,
                    const Int &chain_length,
                    const Int &ignore_prim_per_segment,
                    nb::handle active_obj) -> nb::object {
                     const std::string module_name =
                         nb::cast<std::string>(points_obj.type().attr("__module__"));
                     const std::string type_name =
                         nb::cast<std::string>(points_obj.type().attr("__name__"));

                     if (module_name == "drjit.cuda.ad" && type_name == "Array3f") {
                         Vector3fAD points = nb::cast<Vector3fAD>(points_obj);
                         rayd::MaskAD active = nb::cast<rayd::MaskAD>(active_obj);
                         return nb::cast(scene.visible_chain<false>(
                             points, chain_length, ignore_prim_per_segment, active));
                     }

                     if (module_name == "drjit.cuda" && type_name == "Array3f") {
                         Vector3f points = nb::cast<Vector3f>(points_obj);
                         rayd::Mask active =
                             nb::cast<rayd::Mask>(active_obj);
                         return nb::cast(scene.visible_chain<true>(
                             points, chain_length, ignore_prim_per_segment, active));
                     }
                     throw nb::next_overload();
                 },
                 nb::arg("points"),
                 "chain_length"_a,
                 "ignore_prim_per_segment"_a = Int(),
                 "active"_a = true)
            .def("nearest_edge",
                 [](const Scene &scene, nb::handle point_obj, nb::handle active_obj) -> nb::object {
                     const std::string module_name =
                         nb::cast<std::string>(point_obj.type().attr("__module__"));
                     const std::string type_name =
                         nb::cast<std::string>(point_obj.type().attr("__name__"));

                     if (module_name == "drjit.cuda.ad" && type_name == "Array3f") {
                         Vector3fAD point = nb::cast<Vector3fAD>(point_obj);
                         rayd::MaskAD active = nb::cast<rayd::MaskAD>(active_obj);
                         return nb::cast(scene.nearest_edge<false>(point, active));
                     }

                     if (module_name == "drjit.cuda" && type_name == "Array3f") {
                         Vector3f point_detached = nb::cast<Vector3f>(point_obj);
                         rayd::Mask active = nb::cast<rayd::Mask>(active_obj);
                         return nb::cast(scene.nearest_edge<true>(point_detached, active));
                     }
                     throw nb::next_overload();
                 },
                 nb::arg("point"), "active"_a = true)
            .def("nearest_edge",
                 [](const Scene &scene, const Ray &ray, rayd::Mask active) {
                     return scene.nearest_edge<true>(ray, active);
                 },
                 nb::arg("ray").noconvert(), "active"_a = true)
            .def("nearest_edge",
                 [](const Scene &scene, const RayAD &ray, rayd::MaskAD active) {
                     return scene.nearest_edge<false>(ray, active);
                 },
                 nb::arg("ray").noconvert(), "active"_a = true)
            .def("nearest_edges",
                 [](const Scene &scene, nb::handle point_obj, int k, nb::handle active_obj) -> nb::object {
                     const std::string module_name =
                         nb::cast<std::string>(point_obj.type().attr("__module__"));
                     const std::string type_name =
                         nb::cast<std::string>(point_obj.type().attr("__name__"));

                     if (module_name == "drjit.cuda.ad" && type_name == "Array3f") {
                         Vector3fAD point = nb::cast<Vector3fAD>(point_obj);
                         rayd::MaskAD active = nb::cast<rayd::MaskAD>(active_obj);
                         return nb::cast(scene.nearest_edges<false>(point, k, active));
                     }

                     if (module_name == "drjit.cuda" && type_name == "Array3f") {
                         Vector3f point_detached = nb::cast<Vector3f>(point_obj);
                         rayd::Mask active = nb::cast<rayd::Mask>(active_obj);
                         return nb::cast(scene.nearest_edges<true>(point_detached, k, active));
                     }
                     throw nb::next_overload();
                 },
                 nb::arg("point"), "k"_a, "active"_a = true)
            .def_prop_ro("num_meshes", &Scene::num_meshes)
            .def_prop_ro("version", &Scene::version)
            .def_prop_ro("edge_version", &Scene::edge_version)
            .def("__repr__", &Scene::to_string);
    });
}
