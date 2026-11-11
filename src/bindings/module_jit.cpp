// Copyright Xingyu Chen.
// Implements bindings support for module Dr.Jit.

#include <rayd/jit/core.h>

#include <stdexcept>

#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <drjit/python.h>

#include <rayd/jit/core.h>
#include <rayd/jit/edge.h>
#include <rayd/jit/visibility.h>
#include <rayd/jit/mesh.h>
#include <rayd/jit/mixed_scene.h>
#include <rayd/jit/optix.h>
#include <rayd/jit/sdf.h>
#include <rayd/jit/surfel.h>
#include <rayd/jit/scene.h>

#include <rayd/jit/native_launch_audit.h>

namespace nb = nanobind;
using namespace nb::literals;
using namespace rayd;

namespace {

/// Canonical name of a resolved triangle trace backend.
const char* trace_backend_name(TraceBackendKind kind) {
    switch (kind) {
    case TraceBackendKind::Optix:
        return "optix";
    case TraceBackendKind::Cuda:
        return "cuda";
    default:
        return "none";
    }
}

/// Reject a non-positive Scene.trace_reflections() bounce count as a Python ValueError.
void require_positive_max_bounces(int max_bounces) {
    if (max_bounces < 1)
        throw nb::value_error("Scene.trace_reflections(): max_bounces must be positive.");
}

/// Property getter exposing the int option field \p field as the Python enum \p Enum.
template <typename Enum, typename Class> auto enum_field_getter(int Class::* field) {
    return [field](const Class& self) { return static_cast<Enum>(self.*field); };
}

/// Property setter storing the Python enum \p Enum into the int option field \p field.
template <typename Enum, typename Class> auto enum_field_setter(int Class::* field) {
    return [field](Class& self, Enum value) { self.*field = static_cast<int>(value); };
}

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
        throw std::runtime_error("set_device(): device index " + std::to_string(device) + " is out of range for " +
                                 std::to_string(count) + " Dr.Jit-compatible CUDA device(s).");
    }

    // RayD scenes, BVHs, and OptiX pipelines are bound to the current thread's
    // CUDA/OptiX state. Synchronize before switching and eagerly initialize the
    // target OptiX context so subsequent Scene::build() calls use that device.
    jit_sync_thread();
    drjit::set_device(device);

    if (initialize_optix && optix_available()) {
        jit_optix_context();
        init_optix_api();
    }

    return jit_cuda_device();
}

/// Borrow the registered Python type object for Dr.Jit array type \p T.
template <typename T> nb::handle drjit_python_type() {
    nb::handle type = nb::type<T>();
    if (!type.is_valid())
        throw std::runtime_error("The requested Dr.Jit array type is not registered.");
    return type;
}

/// True when \p src carries derivatives right now: drjit.grad_enabled() also honours drjit.suspend_grad().
/// A failing query counts as grad-enabled so that a conversion is refused rather than silently detaching.
bool drjit_grad_enabled(nb::handle src) noexcept {
    try {
        return nb::cast<bool>(nb::module_::import_("drjit").attr("grad_enabled")(src));
    } catch (...) {
        PyErr_Clear();
        return true;
    }
}

/// Load a Dr.Jit array \p T from a Python handle, optionally coercing other types; returns success.
/// Converting a grad-enabled AD array into a detached \p T would drop its derivative without a trace, so that
/// conversion fails: an AD overload is chosen instead when one exists, and the call is refused otherwise.
template <typename T> bool drjit_try_load(nb::handle src, T& value, bool convert) {
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
    if constexpr (!drjit::is_diff_v<T>) {
        if (drjit_grad_enabled(src))
            return false;
    }

    try {
        nb::object converted = nb::steal<nb::object>(PyObject_CallOneArg(drjit_python_type<T>().ptr(), src.ptr()));
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

/// Refuse a grad-enabled input of a forward-only operation instead of tracing it detached.
void require_no_derivatives(nb::handle value, const char* operation, const char* input) {
    if (drjit_grad_enabled(value)) {
        throw std::runtime_error(std::string(operation) +
                                 " is forward-only and does not support reverse-mode AD or "
                                 "JVP: its " +
                                 input +
                                 " carries derivatives. Pass a detached value (drjit.detach) "
                                 "or call it under drjit.suspend_grad().");
    }
}

/// Refuse a forward-only operation on a scene whose geometry carries derivatives.
void require_detached_scene_geometry(const Scene& scene, const char* operation) {
    // Query through Python so that drjit.suspend_grad() scopes are honoured like for every other input.
    if (drjit_grad_enabled(nb::cast(scene.global_geometry().vertices))) {
        throw std::runtime_error(std::string(operation) +
                                 " is forward-only and does not support reverse-mode AD or "
                                 "JVP: the scene geometry carries derivatives. Build the scene from detached "
                                 "vertices or call it under drjit.suspend_grad().");
    }
}

/// True when \p obj is an instance of a Dr.Jit CUDA AD array type.
bool is_ad_drjit_array(nb::handle obj) {
    const std::string type_name = nb::cast<std::string>(nb::str(obj.type()));
    return type_name.find("drjit.cuda.ad.") != std::string::npos;
}

/// Load an input of a query whose result is discrete. Such a query has no derivative, so a grad-enabled value is
/// detached explicitly here rather than refused by the detached-array conversion.
template <typename T> T cast_detached_query_input(nb::handle value) {
    if (drjit_grad_enabled(value))
        return nb::cast<T>(nb::module_::import_("drjit").attr("detach")(value));
    return nb::cast<T>(value);
}

ReflEpcFieldOptionsAD refl_epc_field_options_ad_from_detached(const ReflEpcFieldOptions& options) {
    ReflEpcFieldOptionsAD out;
    out.expected_prim_ids = options.expected_prim_ids;
    out.surface_group_id = options.surface_group_id;
    out.surface_group_size = options.surface_group_size;
    out.surface_group_members = options.surface_group_members;
    out.surface_max_group_size = options.surface_max_group_size;
    out.visibility_ignore_mode = options.visibility_ignore_mode;
    out.plane_tolerance = options.plane_tolerance;
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
template <typename T> nb::handle drjit_from_cpp(const T& value) {
    nb::object obj = nb::steal<nb::object>(PyObject_CallNoArgs(drjit_python_type<T>().ptr()));
    if (!obj.is_valid())
        throw nb::python_error();
    *nb::inst_ptr<T>(obj) = value;
    return obj.release();
}

} // namespace

namespace nanobind::detail {

/// nanobind type caster bridging any Dr.Jit array type to/from its Python counterpart.
template <typename T> struct type_caster<T, std::enable_if_t<drjit::is_array_v<T>, int>> {
    using Value = T;
    static constexpr auto Name = const_name<T>();
    template <typename U> using Cast = movable_cast_t<U>;
    template <typename U> static constexpr bool can_cast() { return true; }

    bool from_python(handle src, uint8_t flags, cleanup_list*) noexcept {
        return drjit_try_load<T>(src, value, (flags & (uint8_t)cast_flags::convert) != 0);
    }

    static handle from_cpp(const T& src, rv_policy, cleanup_list*) { return drjit_from_cpp(src); }

    static handle from_cpp(T* src, rv_policy policy, cleanup_list* cleanup) {
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

NB_MODULE(_C, m) {
    auto native_kernel_dict = [](const NativeKernelLaunchStat& entry) {
        nb::dict result;
        result["label"] = entry.label;
        result["launches"] = entry.launches;
        result["total_threads"] = entry.total_threads;
        result["max_threads"] = entry.max_threads;
        result["total_items"] = entry.total_items;
        result["max_items"] = entry.max_items;
        return result;
    };

    auto native_stage_dict = [&](const NativeLaunchStageStats& stats) {
        nb::list kernels;
        for (const NativeKernelLaunchStat& entry : stats.kernels) {
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

    auto bind_section = [](const char* name, auto&& fn) {
        try {
            fn();
        } catch (const std::exception& e) {
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
    m.def("device_count", &checked_cuda_device_count,
          "Return the number of Dr.Jit-compatible CUDA devices visible to RayD.\n\n"
          "Raises RuntimeError when there is none.");
    m.def(
        "current_device", []() { return checked_cuda_device_count() > 0 ? jit_cuda_device() : 0; },
        "Return the current thread's active Dr.Jit CUDA device index.\n\n"
        "Raises RuntimeError when no Dr.Jit-compatible CUDA device is available.");
    m.def("native_launch_audit_clear", &native_launch_audit_clear, "Reset every native launch audit counter to zero.");
    m.def(
        "native_launch_audit",
        [native_stage_dict]() {
            const NativeLaunchAuditSnapshot snapshot = native_launch_audit_snapshot();
            nb::dict result;
            result["unknown"] = native_stage_dict(snapshot.unknown);
            result["build"] = native_stage_dict(snapshot.build);
            result["sync"] = native_stage_dict(snapshot.sync);
            result["intersect"] = native_stage_dict(snapshot.intersect);
            result["trace_reflections"] = native_stage_dict(snapshot.trace_reflections);
            result["accum_refl"] = native_stage_dict(snapshot.accumulate_reflections);
            result["accum_dfr"] = native_stage_dict(snapshot.accum_dfr);
            result["surfel_trace"] = native_stage_dict(snapshot.surfel_trace);
            return result;
        },
        "Return the native launch audit counters grouped by operation stage.\n\n"
        "The result maps each stage ('unknown', 'build', 'sync', 'intersect', 'trace_reflections', 'accum_refl', "
        "'accum_dfr', 'surfel_trace') to a dict of CUDA, CUB, Dr.Jit, and OptiX launch and copy counts plus a "
        "'kernels' list with per-kernel launch, thread, and item totals. OptiX launch times are recorded only when "
        "the environment variable RAYD_NATIVE_LAUNCH_AUDIT_TIMING is set to a value other than 0. Counters "
        "accumulate until native_launch_audit_clear().");
    m.def("set_device", &set_rayd_device, "device"_a, "initialize_optix"_a = true,
          "Set the current thread's active CUDA device for RayD.\n\n"
          "Call this before constructing RayD meshes, scenes, or "
          "Dr.Jit arrays that you intend to use with them. When "
          "initialize_optix=True, RayD also initializes the OptiX device "
          "context when OptiX is available; otherwise CUDA-only operation "
          "remains available. Returns the new active device index. Raises "
          "RuntimeError when no device is available or device is out of range.");
    m.def(
        "optix_available", []() { return optix_available(); },
        "Return True when a usable OptiX driver is present on this system.\n\n"
        "This is a non-throwing capability probe (safe on machines without "
        "OptiX). Set the environment variable RAYD_DISABLE_OPTIX=1 to force a "
        "False result, which lets an OptiX-capable machine exercise the "
        "OptiX-less code paths.");
    // Naming convention: the bare class name is the non-AD variant, which is the
    // common case; the autodiff variant carries an "AD" suffix (e.g. Ray / RayAD,
    // Intersection / IntersectionAD). The C++ aliases follow the same convention,
    // so each Python class name matches its underlying C++ type.
    bind_section("core types", [&]() {
        nb::class_<Ray>(m, "Ray",
                        "Batch of rays with origins o, directions d, and maximum distances tmax.\n"
                        "\n"
                        "Input record. Ray(o, d) sets tmax to infinity. Directions need not be normalized. Use RayAD "
                        "for differentiable queries.")
            .def(nb::init<>(), "Create a ray batch. With o and d, tmax is infinity.")
            .def(nb::init<const Vector3f&, const Vector3f&>())
            .def("reversed", &Ray::reversed, "Return the same rays with flipped directions.")
            .def_rw("o", &Ray::o, "Ray origins.")
            .def_rw("d", &Ray::d, "Ray directions; they need not be normalized.")
            .def_rw("tmax", &Ray::tmax, "Maximum ray distance; hits beyond it are ignored.");

        nb::class_<RayAD>(m, "RayAD",
                          "Batch of rays with Dr.Jit AD arrays for o, d, and tmax.\n"
                          "\n"
                          "Input record. Passing a RayAD selects the differentiable overload of a query.")
            .def(nb::init<>(), "Create an AD ray batch. With o and d, tmax is infinity.")
            .def(nb::init<const Vector3fAD&, const Vector3fAD&>())
            .def("reversed", &RayAD::reversed, "Return the same rays with flipped directions.")
            .def_rw("o", &RayAD::o, "Ray origins.")
            .def_rw("d", &RayAD::d, "Ray directions; they need not be normalized.")
            .def_rw("tmax", &RayAD::tmax, "Maximum ray distance; hits beyond it are ignored.");

        nb::class_<SecondaryEdgeInfoAD>(m, "SecondaryEdgeInfo",
                                        "World-space per-edge data of one mesh, as used by the edge queries.\n"
                                        "\n"
                                        "Result record returned by Mesh.secondary_edges().")
            .def(nb::init<>())
            .def("size", &SecondaryEdgeInfoAD::size, "Return the number of edges.")
            .def_ro("start", &SecondaryEdgeInfoAD::start)
            .def_ro("edge", &SecondaryEdgeInfoAD::edge)
            .def_ro("normal0", &SecondaryEdgeInfoAD::normal0)
            .def_ro("normal1", &SecondaryEdgeInfoAD::normal1)
            .def_ro("opposite", &SecondaryEdgeInfoAD::opposite)
            .def_ro("is_boundary", &SecondaryEdgeInfoAD::is_boundary);

        nb::class_<SceneEdgeInfo>(m, "SceneEdgeInfo",
                                  "Scene-global per-edge geometry and ids.\n"
                                  "\n"
                                  "Result record returned by Scene.edge_info().")
            .def(nb::init<>())
            .def("size", &SceneEdgeInfo::size, "Return the number of edges.")
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

        nb::class_<SceneEdgeTopology>(m, "SceneEdgeTopology",
                                      "Scene-global edge connectivity, one entry per edge.\n"
                                      "\n"
                                      "Result record returned by Scene.edge_topology(). Fields ending in _global index "
                                      "the scene-global vertex and face buffers; the others are per-mesh. face1, "
                                      "opposite_vertex1, and their global forms are -1 for boundary edges.")
            .def(nb::init<>())
            .def("size", &SceneEdgeTopology::size, "Return the number of edges.")
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

        nb::class_<SceneGeometry>(
            m, "SceneGeometry",
            "All meshes' world-space vertices and faces flattened into one scene-global index space.\n"
            "\n"
            "Result record returned by Scene.global_geometry().")
            .def(nb::init<>())
            .def("vertex_count", &SceneGeometry::vertex_count, "Return the number of scene-global vertices.")
            .def("face_count", &SceneGeometry::face_count, "Return the number of scene-global faces.")
            .def_ro("vertices", &SceneGeometry::vertices)
            .def_ro("faces", &SceneGeometry::faces)
            .def_ro("face_normal", &SceneGeometry::face_normal)
            .def_ro("shape_id", &SceneGeometry::shape_id)
            .def_ro("local_prim_id", &SceneGeometry::local_prim_id)
            .def_ro("global_prim_id", &SceneGeometry::global_prim_id);

        nb::enum_<RayFlags>(
            m, "RayFlags", nb::is_arithmetic(),
            "Flags that select which Intersection fields Scene.intersect() computes.\n"
            "\n"
            "Geometric selects the geometric normal geo_n, ShadingN the interpolated shading normal n, and UV the "
            "texture coordinates uv. All sets every flag and Empty none. t and the ids are always filled.")
            .value("Empty", RayFlags::None)
            .value("Geometric", RayFlags::Geometric)
            .value("ShadingN", RayFlags::ShadingN)
            .value("UV", RayFlags::UV)
            .value("All", RayFlags::All);

        nb::enum_<SurfelPrimitiveMode>(m, "SurfelPrimitiveMode",
                                       "Proxy triangle geometry used to find surfel candidates.\n"
                                       "\n"
                                       "Icosahedron20 builds a thin 20-triangle shell per surfel, QuadTriangles two "
                                       "triangles, and SingleTriangle one.")
            .value("Icosahedron20", SurfelPrimitiveMode::Icosahedron20)
            .value("QuadTriangles", SurfelPrimitiveMode::QuadTriangles)
            .value("SingleTriangle", SurfelPrimitiveMode::SingleTriangle);

        nb::enum_<SurfelColorModel>(m, "SurfelColorModel",
                                    "How SurfelAppearance values are interpreted.\n"
                                    "\n"
                                    "ConstantRGB stores three values per surfel, SH stores spherical-harmonic RGB "
                                    "coefficients, and FeatureChannels stores channel_count arbitrary values.")
            .value("ConstantRGB", SurfelColorModel::ConstantRGB)
            .value("SH", SurfelColorModel::SH)
            .value("FeatureChannels", SurfelColorModel::FeatureChannels);

        nb::enum_<SurfelRenderMode>(m, "SurfelRenderMode",
                                    "Output of SurfelScene.render(): Alpha, Depth, RGB, RGBDepth, or Feature channels.")
            .value("Alpha", SurfelRenderMode::Alpha)
            .value("Depth", SurfelRenderMode::Depth)
            .value("RGB", SurfelRenderMode::RGB)
            .value("RGBDepth", SurfelRenderMode::RGBDepth)
            .value("Feature", SurfelRenderMode::Feature);

        nb::class_<SurfelTraceOptions>(
            m, "SurfelTraceOptions",
            "Options for surfel tracing and alpha compositing.\n"
            "\n"
            "Options record for SurfelScene and MixedScene.add_surfel(). Values are checked when the SurfelScene is "
            "created or built, and an invalid value raises RuntimeError.")
            .def(nb::init<>())
            .def_rw("alpha_min", &SurfelTraceOptions::alpha_min,
                    "Minimum Gaussian alpha a surfel contributes; must lie in (0, 1).")
            .def_rw("cutoff", &SurfelTraceOptions::cutoff,
                    "Cap on the proxy radius in local Gaussian units; inf disables it.")
            .def_rw("alpha_cap", &SurfelTraceOptions::alpha_cap, "Upper bound on a surfel's alpha; must lie in (0, 1].")
            .def_rw(
                "proxy_epsilon", &SurfelTraceOptions::proxy_epsilon,
                "Normal-axis thickness of the Icosahedron20 proxy relative to the Gaussian radius; must be positive.")
            .def_rw("max_candidate_hits", &SurfelTraceOptions::max_candidate_hits,
                    "Per-ray candidate buffer size for alpha compositing.")
            .def_rw("primitive_mode", &SurfelTraceOptions::primitive_mode, "Proxy geometry per surfel.")
            .def_rw("face_forward", &SurfelTraceOptions::face_forward,
                    "Flip each surfel normal to face the incoming ray.")
            .def_rw("single_launch", &SurfelTraceOptions::single_launch,
                    "Use the native OptiX pipeline for detached intersect and composite calls.")
            .def_rw("collect_candidate_stats", &SurfelTraceOptions::collect_candidate_stats,
                    "Collect per-ray candidate-buffer occupancy diagnostics.")
            .def_rw("opacity_aware_proxy_bounds", &SurfelTraceOptions::opacity_aware_proxy_bounds,
                    "Size each proxy from the surfel's opacity.")
            .def_rw("continue_after_full_buffer", &SurfelTraceOptions::continue_after_full_buffer,
                    "Keep tracing after the candidate buffer fills, until transmittance drops below transmittance_min.")
            .def_rw("transmittance_min", &SurfelTraceOptions::transmittance_min,
                    "Transmittance at which compositing stops; must lie in [0, 1).")
            .def_rw("max_trace_segments", &SurfelTraceOptions::max_trace_segments,
                    "Maximum trace segments per ray when continue_after_full_buffer is set; must be positive.");

        nb::class_<SdfTraceOptions>(
            m, "SdfTraceOptions",
            "Sphere-tracing options for SdfGrid queries.\n"
            "\n"
            "Options record for SdfGrid.intersect(), visible(), trace_reflections(), and MixedScene.add_sdf().")
            .def(nb::init<>())
            .def_rw("max_steps", &SdfTraceOptions::max_steps, "Maximum sphere-tracing steps; must be at least 1.")
            .def_rw("relaxation", &SdfTraceOptions::relaxation, "Step relaxation factor in (0, 1].")
            .def_rw("eps_hit", &SdfTraceOptions::eps_hit,
                    "Hit threshold in world units. A non-positive value selects 1e-3 of the smallest voxel size.");

        nb::class_<SurfelRenderOptions>(
            m, "SurfelRenderOptions",
            "Options for SurfelScene.render().\n"
            "\n"
            "Options record. Use the rgb() or feature() factory for the common configurations.")
            .def(nb::init<>())
            .def_static(
                "rgb",
                [](int sh_degree, const ScalarVector3f& background_rgb, bool normal) {
                    SurfelRenderOptions options;
                    options.mode = SurfelRenderMode::RGB;
                    options.color_model = sh_degree > 0 ? SurfelColorModel::SH : SurfelColorModel::ConstantRGB;
                    options.sh_degree = sh_degree;
                    options.channel_count = 3;
                    options.background_rgb = background_rgb;
                    options.normal = normal;
                    return options;
                },
                "sh_degree"_a = 0, "background_rgb"_a = ScalarVector3f(0.f, 0.f, 0.f), "normal"_a = false,
                "Return options for an RGB render.\n"
                "\n"
                "sh_degree > 0 selects the SH color model, otherwise ConstantRGB. background_rgb is the background "
                "color and normal requests the alpha-weighted normal.")
            .def_static(
                "feature",
                [](int channel_count) {
                    SurfelRenderOptions options;
                    options.mode = SurfelRenderMode::Feature;
                    options.color_model = SurfelColorModel::FeatureChannels;
                    options.channel_count = channel_count;
                    return options;
                },
                "channel_count"_a, "Return options for a render of channel_count feature channels.")
            .def_rw("mode", &SurfelRenderOptions::mode, "What to render.")
            .def_rw("color_model", &SurfelRenderOptions::color_model,
                    "Color model; it must match the scene's SurfelAppearance.")
            .def_rw("normal", &SurfelRenderOptions::normal, "Also output the alpha-weighted surfel normal.")
            .def_rw("sh_degree", &SurfelRenderOptions::sh_degree, "Spherical-harmonic degree to evaluate, in [0, 3].")
            .def_rw("channel_count", &SurfelRenderOptions::channel_count,
                    "Number of feature channels for the Feature mode.")
            .def_rw("channel_chunk", &SurfelRenderOptions::channel_chunk, "Not used by the current renderer.")
            .def_rw("background_rgb", &SurfelRenderOptions::background_rgb, "Background color for the RGB modes.");

        nb::enum_<ReflectionExportMode>(m, "ReflectionExportMode", nb::is_arithmetic(),
                                        "Payload exported by Scene.trace_reflections() with symbolic=False.\n"
                                        "\n"
                                        "FULL exports every field, MINIMAL exports bounce counts plus per-slot ids and "
                                        "t, and COUNT_ONLY exports bounce counts only.")
            .value("FULL", RAYD_REFLECTION_EXPORT_FULL)
            .value("MINIMAL", RAYD_REFLECTION_EXPORT_MINIMAL)
            .value("COUNT_ONLY", RAYD_REFLECTION_EXPORT_COUNT_ONLY);

        nb::class_<ReflectionTraceOptions>(m, "ReflectionTraceOptions",
                                           "Options for Scene.trace_reflections().\n"
                                           "\n"
                                           "Options record.")
            .def(nb::init<>())
            .def_rw("deduplicate", &ReflectionTraceOptions::deduplicate,
                    "Merge paths with the same reflector sequence. Needs symbolic=False and the FULL export mode.")
            .def_rw("canonical_prim_table", &ReflectionTraceOptions::canonical_prim_table,
                    "Optional map from primitive id to a canonical id used when deduplicating.")
            .def_rw("image_source_tolerance", &ReflectionTraceOptions::image_source_tolerance,
                    "Distance tolerance when comparing image sources during deduplication.")
            .def_prop_rw("export_mode", enum_field_getter<ReflectionExportMode>(&ReflectionTraceOptions::export_mode),
                         enum_field_setter<ReflectionExportMode>(&ReflectionTraceOptions::export_mode),
                         "Exported payload; AD traces always export FULL.")
            .def_rw("return_trailing", &ReflectionTraceOptions::return_trailing,
                    "Trace and export the segment after the last reflection.");

        nb::class_<ReflEpcOptions>(m, "ReflEpcOptions",
                                   "Options for Scene.trace_refl_epc(): the expected reflector sequence and the "
                                   "surface-group tables used for visibility.\n"
                                   "\n"
                                   "Options record.")
            .def(nb::init<>())
            .def_rw("expected_prim_ids", &ReflEpcOptions::expected_prim_ids,
                    "Expected reflector per slot, n_rays * max_bounces entries. trace_refl_epc() also accepts an empty "
                    "array.")
            .def_rw("surface_group_id", &ReflEpcOptions::surface_group_id, "Surface-group id per scene triangle.")
            .def_rw("surface_group_size", &ReflEpcOptions::surface_group_size, "Member count of each surface group.")
            .def_rw("surface_group_members", &ReflEpcOptions::surface_group_members,
                    "Flattened member table with at least surface_group_count * surface_max_group_size entries.")
            .def_rw("surface_max_group_size", &ReflEpcOptions::surface_max_group_size,
                    "Row width of surface_group_members.")
            .def_rw("visibility_ignore_mode", &ReflEpcOptions::visibility_ignore_mode,
                    "How visibility checks ignore the reflecting surfaces: 'primitive' or 'surface_group'. "
                    "'surface_group' needs the surface-group tables.")
            .def_rw("plane_tolerance", &ReflEpcOptions::plane_tolerance,
                    "Relative tolerance for a reflection point to lie on its plane.")
            .def_rw("final_ignore_group_ids", &ReflEpcOptions::final_ignore_group_ids,
                    "Surface groups ignored on the last segment to the receiver, one or n_rays entries. Needs "
                    "visibility_ignore_mode='surface_group'.");

        nb::class_<ReflEpcFieldOptions, ReflEpcOptions>(
            m, "ReflEpcFieldOptions",
            "Options for Scene.trace_refl_epc_field() with detached arrays.\n"
            "\n"
            "Options record. Extends ReflEpcOptions with the per-slot reflecting planes and materials, the source "
            "polarization and frequency, and the output selectors. Per-slot arrays have n_rays * max_bounces entries.")
            .def(nb::init<>())
            .def_rw("slot_plane_point", &ReflEpcFieldOptions::slot_plane_point,
                    "Point on the reflecting plane of each slot.")
            .def_rw("slot_plane_normal", &ReflEpcFieldOptions::slot_plane_normal,
                    "Normal of the reflecting plane of each slot.")
            .def_rw("slot_eta_r", &ReflEpcFieldOptions::slot_eta_r, "Relative permittivity of each slot.")
            .def_rw("slot_mu_r", &ReflEpcFieldOptions::slot_mu_r, "Relative permeability of each slot.")
            .def_rw("slot_sigma", &ReflEpcFieldOptions::slot_sigma, "Conductivity of each slot.")
            .def_rw("slot_gain", &ReflEpcFieldOptions::slot_gain, "Extra gain factor of each slot.")
            .def_rw("tx_polarization", &ReflEpcFieldOptions::tx_polarization,
                    "Transmitter polarization vector, one or n_rays entries.")
            .def_rw("omega", &ReflEpcFieldOptions::omega, "Angular frequency in rad/s; must be positive.")
            .def_rw("wavelength", &ReflEpcFieldOptions::wavelength, "Wavelength in world units; must be positive.")
            .def_rw("return_geom", &ReflEpcFieldOptions::return_geom,
                    "Master switch for the per-slot geometry outputs.")
            .def_rw("return_endpoints", &ReflEpcFieldOptions::return_endpoints,
                    "Output tx_pos, first_hit, and last_hit.")
            .def_rw("return_hit_points", &ReflEpcFieldOptions::return_hit_points,
                    "Output per-slot hit points when return_geom is set.")
            .def_rw("return_normals", &ReflEpcFieldOptions::return_normals,
                    "Output per-slot normals when return_geom is set.")
            .def_rw("return_resolved_prim_ids", &ReflEpcFieldOptions::return_resolved_prim_ids,
                    "Output resolved primitive ids when return_geom is set.")
            .def_rw("return_surface_group_ids", &ReflEpcFieldOptions::return_surface_group_ids,
                    "Output surface-group ids when return_geom is set.");

        nb::class_<ReflEpcFieldOptionsAD, ReflEpcOptions>(
            m, "ReflEpcFieldOptionsAD",
            "Options for Scene.trace_refl_epc_field() whose per-slot planes, materials, and polarization are AD "
            "arrays.\n"
            "\n"
            "Options record. Same fields as ReflEpcFieldOptions; use it to differentiate with respect to those values.")
            .def(nb::init<>())
            .def_rw("slot_plane_point", &ReflEpcFieldOptionsAD::slot_plane_point,
                    "Point on the reflecting plane of each slot.")
            .def_rw("slot_plane_normal", &ReflEpcFieldOptionsAD::slot_plane_normal,
                    "Normal of the reflecting plane of each slot.")
            .def_rw("slot_eta_r", &ReflEpcFieldOptionsAD::slot_eta_r, "Relative permittivity of each slot.")
            .def_rw("slot_mu_r", &ReflEpcFieldOptionsAD::slot_mu_r, "Relative permeability of each slot.")
            .def_rw("slot_sigma", &ReflEpcFieldOptionsAD::slot_sigma, "Conductivity of each slot.")
            .def_rw("slot_gain", &ReflEpcFieldOptionsAD::slot_gain, "Extra gain factor of each slot.")
            .def_rw("tx_polarization", &ReflEpcFieldOptionsAD::tx_polarization,
                    "Transmitter polarization vector, one or n_rays entries.")
            .def_rw("omega", &ReflEpcFieldOptionsAD::omega, "Angular frequency in rad/s; must be positive.")
            .def_rw("wavelength", &ReflEpcFieldOptionsAD::wavelength, "Wavelength in world units; must be positive.")
            .def_rw("return_geom", &ReflEpcFieldOptionsAD::return_geom,
                    "Master switch for the per-slot geometry outputs.")
            .def_rw("return_endpoints", &ReflEpcFieldOptionsAD::return_endpoints,
                    "Output tx_pos, first_hit, and last_hit.")
            .def_rw("return_hit_points", &ReflEpcFieldOptionsAD::return_hit_points,
                    "Output per-slot hit points when return_geom is set.")
            .def_rw("return_normals", &ReflEpcFieldOptionsAD::return_normals,
                    "Output per-slot normals when return_geom is set.")
            .def_rw("return_resolved_prim_ids", &ReflEpcFieldOptionsAD::return_resolved_prim_ids,
                    "Output resolved primitive ids when return_geom is set.")
            .def_rw("return_surface_group_ids", &ReflEpcFieldOptionsAD::return_surface_group_ids,
                    "Output surface-group ids when return_geom is set.");

        nb::class_<Intersection>(m, "Intersection",
                                 "Closest ray-triangle hit per ray.\n"
                                 "\n"
                                 "Result record of Scene.intersect() and MixedScene.intersect(). Missed rays have t = "
                                 "inf and ids = -1; check is_valid() before reading other fields.")
            .def("is_valid", &Intersection::is_valid, "Return the per-row validity mask.")
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

        nb::class_<IntersectionAD>(m, "IntersectionAD",
                                   "AD variant of Intersection, returned when intersect() gets a RayAD.")
            .def("is_valid", &IntersectionAD::is_valid, "Return the per-row validity mask.")
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

        nb::class_<SurfelIntersection>(m, "SurfelIntersection",
                                       "Closest surfel hit per ray.\n"
                                       "\n"
                                       "Result record of SurfelScene.intersect(). Missed rays have surfel_id = -1.")
            .def("is_valid", &SurfelIntersection::is_valid, "Return the per-row validity mask.")
            .def_ro("t", &SurfelIntersection::t)
            .def_ro("p", &SurfelIntersection::p)
            .def_ro("n", &SurfelIntersection::n)
            .def_ro("local_uv", &SurfelIntersection::local_uv)
            .def_ro("gaussian_weight", &SurfelIntersection::gaussian_weight)
            .def_ro("opacity", &SurfelIntersection::opacity)
            .def_ro("alpha", &SurfelIntersection::alpha)
            .def_ro("value", &SurfelIntersection::value)
            .def_ro("surfel_id", &SurfelIntersection::surfel_id)
            .def_ro("triangle_id", &SurfelIntersection::triangle_id);

        nb::class_<SurfelIntersectionAD>(m, "SurfelIntersectionAD", "AD variant of SurfelIntersection.")
            .def("is_valid", &SurfelIntersectionAD::is_valid, "Return the per-row validity mask.")
            .def_ro("t", &SurfelIntersectionAD::t)
            .def_ro("p", &SurfelIntersectionAD::p)
            .def_ro("n", &SurfelIntersectionAD::n)
            .def_ro("local_uv", &SurfelIntersectionAD::local_uv)
            .def_ro("gaussian_weight", &SurfelIntersectionAD::gaussian_weight)
            .def_ro("opacity", &SurfelIntersectionAD::opacity)
            .def_ro("alpha", &SurfelIntersectionAD::alpha)
            .def_ro("value", &SurfelIntersectionAD::value)
            .def_ro("surfel_id", &SurfelIntersectionAD::surfel_id)
            .def_ro("triangle_id", &SurfelIntersectionAD::triangle_id);

        nb::class_<SdfIntersection>(m, "SdfIntersection",
                                    "Closest SDF surface hit per ray.\n"
                                    "\n"
                                    "Result record of SdfGrid.intersect(). valid marks hits.")
            .def("is_valid", &SdfIntersection::is_valid, "Return the per-row validity mask.")
            .def_ro("t", &SdfIntersection::t)
            .def_ro("valid", &SdfIntersection::hit_mask)
            .def_ro("position", &SdfIntersection::position)
            .def_ro("normal", &SdfIntersection::normal)
            .def_ro("steps", &SdfIntersection::steps);

        nb::class_<SdfIntersectionAD>(m, "SdfIntersectionAD",
                                      "AD variant of SdfIntersection, returned when SdfGrid.intersect() gets a RayAD.")
            .def("is_valid", &SdfIntersectionAD::is_valid, "Return the per-row validity mask.")
            .def_ro("t", &SdfIntersectionAD::t)
            .def_ro("valid", &SdfIntersectionAD::hit_mask)
            .def_ro("position", &SdfIntersectionAD::position)
            .def_ro("normal", &SdfIntersectionAD::normal)
            .def_ro("steps", &SdfIntersectionAD::steps);

        nb::class_<SurfelComposite>(m, "SurfelComposite",
                                    "Front-to-back alpha composite of the surfels along each ray.\n"
                                    "\n"
                                    "Result record of SurfelScene.composite_alpha() and composite_alpha_reference().")
            .def("is_valid", &SurfelComposite::is_valid, "Return the per-row validity mask.")
            .def_ro("intensity", &SurfelComposite::intensity)
            .def_ro("alpha", &SurfelComposite::alpha)
            .def_ro("transmittance", &SurfelComposite::transmittance)
            .def_ro("depth", &SurfelComposite::depth)
            .def_ro("candidate_count", &SurfelComposite::candidate_count)
            .def_ro("candidate_buffer_full", &SurfelComposite::candidate_buffer_full);

        nb::class_<SurfelCompositeAD>(m, "SurfelCompositeAD", "AD variant of SurfelComposite.")
            .def("is_valid", &SurfelCompositeAD::is_valid, "Return the per-row validity mask.")
            .def_ro("intensity", &SurfelCompositeAD::intensity)
            .def_ro("alpha", &SurfelCompositeAD::alpha)
            .def_ro("transmittance", &SurfelCompositeAD::transmittance)
            .def_ro("depth", &SurfelCompositeAD::depth)
            .def_ro("candidate_count", &SurfelCompositeAD::candidate_count)
            .def_ro("candidate_buffer_full", &SurfelCompositeAD::candidate_buffer_full);

        nb::class_<SurfelRender>(
            m, "SurfelRender",
            "Alpha-composited surfel render per ray.\n"
            "\n"
            "Result record of SurfelScene.render(). channels is a flat [ray_count, channel_count] buffer.")
            .def("is_valid", &SurfelRender::is_valid, "Return the per-row validity mask.")
            .def_ro("channels", &SurfelRender::channels)
            .def_ro("rgb", &SurfelRender::rgb)
            .def_ro("normal", &SurfelRender::normal)
            .def_ro("alpha", &SurfelRender::alpha)
            .def_ro("transmittance", &SurfelRender::transmittance)
            .def_ro("depth", &SurfelRender::depth)
            .def_ro("candidate_count", &SurfelRender::candidate_count)
            .def_ro("candidate_buffer_full", &SurfelRender::candidate_buffer_full)
            .def_ro("channel_count", &SurfelRender::channel_count);

        nb::class_<SurfelRenderAD>(m, "SurfelRenderAD", "AD variant of SurfelRender.")
            .def("is_valid", &SurfelRenderAD::is_valid, "Return the per-row validity mask.")
            .def_ro("channels", &SurfelRenderAD::channels)
            .def_ro("rgb", &SurfelRenderAD::rgb)
            .def_ro("normal", &SurfelRenderAD::normal)
            .def_ro("alpha", &SurfelRenderAD::alpha)
            .def_ro("transmittance", &SurfelRenderAD::transmittance)
            .def_ro("depth", &SurfelRenderAD::depth)
            .def_ro("candidate_count", &SurfelRenderAD::candidate_count)
            .def_ro("candidate_buffer_full", &SurfelRenderAD::candidate_buffer_full)
            .def_ro("channel_count", &SurfelRenderAD::channel_count);

        nb::class_<ReflectionChain>(
            m, "ReflectionChain",
            "Reflection paths with per-bounce arrays laid out ray-major as ray_count * max_bounces slots.\n"
            "\n"
            "Result record of Scene.trace_reflections() with symbolic=False, and of the SdfGrid, SurfelScene, and "
            "MixedScene reflection traces. bounce_count gives the valid slots per ray.")
            .def("is_valid", &ReflectionChain::is_valid, "Return the per-row validity mask.")
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

        nb::class_<ReflectionChainAD>(m, "ReflectionChainAD", "AD variant of ReflectionChain.")
            .def("is_valid", &ReflectionChainAD::is_valid, "Return the per-row validity mask.")
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

        nb::class_<AccumGrid>(
            m, "AccumGrid",
            "Axis-aligned accumulation plane for Scene.accum_refl().\n"
            "\n"
            "Input record. The plane is normal to axis at offset position. Its in-plane coordinates are (y, z) for "
            "axis 0, (x, z) for axis 1, and (x, y) for axis 2, and cell (i, j) has the flat index i + j * resolution0.")
            .def(nb::init<>())
            .def_rw("axis", &AccumGrid::axis, "Plane normal axis: 0 = x, 1 = y, 2 = z.")
            .def_rw("position", &AccumGrid::position, "Plane offset along axis.")
            .def_rw("coord0_min", &AccumGrid::coord0_min, "Lower bound of the first in-plane coordinate.")
            .def_rw("coord0_max", &AccumGrid::coord0_max,
                    "Upper bound of the first in-plane coordinate; must exceed coord0_min.")
            .def_rw("coord1_min", &AccumGrid::coord1_min, "Lower bound of the second in-plane coordinate.")
            .def_rw("coord1_max", &AccumGrid::coord1_max,
                    "Upper bound of the second in-plane coordinate; must exceed coord1_min.")
            .def_rw("resolution0", &AccumGrid::resolution0,
                    "Cell count along the first in-plane coordinate; must be positive.")
            .def_rw("resolution1", &AccumGrid::resolution1,
                    "Cell count along the second in-plane coordinate; must be positive.");

        nb::class_<AccumOptions>(m, "AccumOptions",
                                 "Options for Scene.accum_refl().\n"
                                 "\n"
                                 "Options record.")
            .def(nb::init<>())
            .def_rw("wavelength", &AccumOptions::wavelength, "Wavelength in world units; must be positive.")
            .def_rw("k", &AccumOptions::k, "Wavenumber; 0 uses 2 * pi / wavelength.")
            .def_rw("solid_angle_per_ray", &AccumOptions::solid_angle_per_ray,
                    "Solid angle each ray represents; must be non-negative.")
            .def_rw("cell_area", &AccumOptions::cell_area, "Area of one grid cell; must be positive.")
            .def_rw("seed", &AccumOptions::seed, "Random seed.")
            .def_rw("rr_depth", &AccumOptions::rr_depth, "Bounce depth at which Russian roulette starts.")
            .def_rw("rr_prob", &AccumOptions::rr_prob, "Survival probability of each roulette test.")
            .def_rw("stop_threshold", &AccumOptions::stop_threshold,
                    "Stop a path once its contribution falls below this value.")
            .def_rw("collect_wedges", &AccumOptions::collect_wedges,
                    "Record diffraction-wedge events in AccumResult.wedge_events.")
            .def_rw("collect_wedge_prefixes", &AccumOptions::collect_wedge_prefixes,
                    "Also record the reflection prefix of each wedge event.")
            .def_rw("wedge_capacity", &AccumOptions::wedge_capacity,
                    "Maximum number of wedge events to store; must be non-negative.")
            .def_rw("wedge_sample_stride", &AccumOptions::wedge_sample_stride,
                    "Sampling stride for prefix wedge events; must be at least 1.");

        nb::class_<Material>(m, "ReflMaterial",
                             "Electromagnetic surface parameters for Scene.accum_refl().\n"
                             "\n"
                             "Input record with one entry per scene-global primitive. Use ReflMaterialAD to "
                             "differentiate with respect to the material.")
            .def(nb::init<>())
            .def_rw("eta_r", &Material::eta_r, "Relative permittivity.")
            .def_rw("sigma", &Material::sigma, "Conductivity.")
            .def_rw("gain", &Material::gain, "Extra gain factor.")
            .def_rw("mu_r", &Material::mu_r, "Relative permeability.")
            .def_rw("valid", &Material::valid, "Whether the entry holds a real material.");

        nb::class_<MaterialAD>(m, "ReflMaterialAD",
                               "ReflMaterial with AD arrays.\n"
                               "\n"
                               "Input record for the differentiable Scene.accum_refl() overload.")
            .def(nb::init<>())
            .def_rw("eta_r", &MaterialAD::eta_r, "Relative permittivity.")
            .def_rw("sigma", &MaterialAD::sigma, "Conductivity.")
            .def_rw("gain", &MaterialAD::gain, "Extra gain factor.")
            .def_rw("mu_r", &MaterialAD::mu_r, "Relative permeability.")
            .def_rw("valid", &MaterialAD::valid, "Whether the entry holds a real material.");

        nb::class_<WedgeEvents>(
            m, "WedgeEvents",
            "Diffraction-wedge events recorded by Scene.accum_refl().\n"
            "\n"
            "Result record. The first count entries of the per-event arrays are valid, up to capacity.")
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

        nb::class_<WedgeEventsAD>(m, "WedgeEventsAD", "AD variant of WedgeEvents.")
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

        nb::class_<AccumResult>(m, "AccumResult",
                                "Reflected power and complex field accumulated per grid cell.\n"
                                "\n"
                                "Result record of Scene.accum_refl(). Grid arrays have grid_cell_count entries.")
            .def_ro("ray_count", &AccumResult::ray_count)
            .def_ro("max_bounces", &AccumResult::max_bounces)
            .def_ro("grid_cell_count", &AccumResult::grid_cell_count)
            .def_ro("reflection_power", &AccumResult::reflection_power)
            .def_ro("reflection_field_x", &AccumResult::reflection_field_x)
            .def_ro("reflection_field_y", &AccumResult::reflection_field_y)
            .def_ro("reflection_field_z", &AccumResult::reflection_field_z)
            .def_ro("reflection_count", &AccumResult::reflection_count)
            .def_ro("wedge_events", &AccumResult::wedge_events);

        nb::class_<AccumResultAD>(m, "AccumResultAD",
                                  "AD variant of AccumResult, returned when Scene.accum_refl() gets a RayAD.")
            .def_ro("ray_count", &AccumResultAD::ray_count)
            .def_ro("max_bounces", &AccumResultAD::max_bounces)
            .def_ro("grid_cell_count", &AccumResultAD::grid_cell_count)
            .def_ro("reflection_power", &AccumResultAD::reflection_power)
            .def_ro("reflection_field_x", &AccumResultAD::reflection_field_x)
            .def_ro("reflection_field_y", &AccumResultAD::reflection_field_y)
            .def_ro("reflection_field_z", &AccumResultAD::reflection_field_z)
            .def_ro("reflection_count", &AccumResultAD::reflection_count)
            .def_ro("wedge_events", &AccumResultAD::wedge_events);

        nb::enum_<DfrStrategyMask>(m, "DfrStrategy", nb::is_flag(), nb::is_arithmetic(),
                                   "Flags that select the Monte-Carlo strategies of the diffraction accumulators.\n"
                                   "\n"
                                   "DIRECT samples direct edge diffraction, KELLER samples the Keller cone, and "
                                   "SUFFIX_REFL adds a reflection after the diffraction event.")
            .value("DIRECT", RAYD_DFR_DIRECT)
            .value("KELLER", RAYD_DFR_KELLER)
            .value("SUFFIX_REFL", RAYD_DFR_SUFFIX_REFL);

        nb::enum_<DfrSampleSequence>(m, "DfrSampleSequence", nb::is_arithmetic(),
                                     "Sample sequence of the diffraction accumulators: HASH or SOBOL.")
            .value("HASH", RAYD_DFR_HASH)
            .value("SOBOL", RAYD_DFR_SOBOL);

        nb::enum_<DfrReceiverModel>(
            m, "DfrReceiverModel", nb::is_arithmetic(),
            "Receiver model of the diffraction kernels. MATCHED_ISO, a matched isotropic receiver, is the only model.")
            .value("MATCHED_ISO", RAYD_DFR_MATCHED_ISO);

        nb::class_<DfrGrid>(
            m, "DfrGrid",
            "Axis-aligned accumulation plane for the diffraction accumulators.\n"
            "\n"
            "Input record. The plane is normal to axis at offset position. Its in-plane coordinates are (y, z) for "
            "axis 0, (x, z) for axis 1, and (x, y) for axis 2, and cell (i, j) has the flat index i + j * resolution0.")
            .def(nb::init<>())
            .def_rw("axis", &DfrGrid::axis, "Plane normal axis: 0 = x, 1 = y, 2 = z.")
            .def_rw("position", &DfrGrid::position, "Plane offset along axis.")
            .def_rw("coord0_min", &DfrGrid::coord0_min, "Lower bound of the first in-plane coordinate.")
            .def_rw("coord0_max", &DfrGrid::coord0_max,
                    "Upper bound of the first in-plane coordinate; must exceed coord0_min.")
            .def_rw("coord1_min", &DfrGrid::coord1_min, "Lower bound of the second in-plane coordinate.")
            .def_rw("coord1_max", &DfrGrid::coord1_max,
                    "Upper bound of the second in-plane coordinate; must exceed coord1_min.")
            .def_rw("resolution0", &DfrGrid::resolution0,
                    "Cell count along the first in-plane coordinate; must be positive.")
            .def_rw("resolution1", &DfrGrid::resolution1,
                    "Cell count along the second in-plane coordinate; must be positive.")
            .def_rw("cell_area", &DfrGrid::cell_area, "Area of one grid cell; must be positive.");

        nb::class_<DfrOptions>(m, "DfrOptions",
                               "Options for Scene.accum_dfr_direct() and Scene.accum_dfr().\n"
                               "\n"
                               "Options record.")
            .def(nb::init<>())
            .def_rw("wavelength", &DfrOptions::wavelength, "Wavelength in world units; must be positive.")
            .def_rw("k", &DfrOptions::k,
                    "Wavenumber used for phases. It is used as given, not derived from wavelength.")
            .def_rw("seed", &DfrOptions::seed, "Seed of the sample sequence.")
            .def_rw("samples", &DfrOptions::samples, "Direct-strategy sample count used when direct_samples is 0.")
            .def_rw("max_order", &DfrOptions::max_order,
                    "Diffraction order: 1 for accum_dfr_direct(), 2 or 3 for accum_dfr().")
            .def_rw("direct_samples", &DfrOptions::direct_samples,
                    "Samples for the DIRECT strategy; 0 falls back to samples.")
            .def_rw("keller_samples", &DfrOptions::keller_samples, "Samples for the KELLER strategy.")
            .def_rw("suffix_samples", &DfrOptions::suffix_samples, "Samples for the SUFFIX_REFL strategy.")
            .def_prop_rw("strategy_mask", enum_field_getter<DfrStrategyMask>(&DfrOptions::strategy_mask),
                         enum_field_setter<DfrStrategyMask>(&DfrOptions::strategy_mask),
                         "Enabled strategies; a strategy runs only when its flag is set.")
            .def_prop_rw("sample_sequence", enum_field_getter<DfrSampleSequence>(&DfrOptions::sample_sequence),
                         enum_field_setter<DfrSampleSequence>(&DfrOptions::sample_sequence), "Sample sequence.")
            .def_prop_rw("receiver_model", enum_field_getter<DfrReceiverModel>(&DfrOptions::receiver_model),
                         enum_field_setter<DfrReceiverModel>(&DfrOptions::receiver_model), "Receiver model.")
            .def_rw("collect_edge_use", &DfrOptions::collect_edge_use, "Fill DfrAccum.edge_uses.")
            .def_rw("collect_debug_counts", &DfrOptions::collect_debug_counts,
                    "Fill the rejection counters of DfrAccum.");

        nb::class_<DfrCoherentOptions>(
            m, "DfrCoherentOptions",
            "Options for deterministic coherent diffraction: Scene.accum_dfr_coherent_direct(), "
            "build_dfr_coherent_tx_states(), and build_dfr_coherent_higher_candidates().\n"
            "\n"
            "Options record.")
            .def(nb::init<>())
            .def_rw("wavelength", &DfrCoherentOptions::wavelength, "Wavelength in world units; must be positive.")
            .def_rw("k", &DfrCoherentOptions::k,
                    "Wavenumber used for phases. It is used as given; build_dfr_coherent_tx_states() requires k > 0.")
            .def_rw("max_order", &DfrCoherentOptions::max_order,
                    "Diffraction order; accum_dfr_coherent_direct() supports only 1.")
            .def_prop_rw("receiver_model", enum_field_getter<DfrReceiverModel>(&DfrCoherentOptions::receiver_model),
                         enum_field_setter<DfrReceiverModel>(&DfrCoherentOptions::receiver_model),
                         "Receiver model; only MATCHED_ISO is supported.")
            .def_rw("select_diffraction_point", &DfrCoherentOptions::select_diffraction_point,
                    "For DfrStates input, diffract at the first-order stationary point for each cell instead of the "
                    "edge midpoint.")
            .def_rw("prefilter_visibility", &DfrCoherentOptions::prefilter_visibility,
                    "Skip lanes whose source-to-edge or edge-to-cell segment is blocked.")
            .def_rw("collect_debug_counts", &DfrCoherentOptions::collect_debug_counts,
                    "Fill vis_rejects and utd_rejects.")
            .def_rw("omega", &DfrCoherentOptions::omega,
                    "Angular frequency in rad/s for the face Fresnel coefficients. A value <= 0 uses the face "
                    "operators stored in the UTD states.")
            .def_rw("tx_pol_x", &DfrCoherentOptions::tx_pol_x, "x component of the transmitter polarization.")
            .def_rw("tx_pol_y", &DfrCoherentOptions::tx_pol_y, "y component of the transmitter polarization.")
            .def_rw("tx_pol_z", &DfrCoherentOptions::tx_pol_z, "z component of the transmitter polarization.")
            .def_rw("higher_probe_radius_scale", &DfrCoherentOptions::higher_probe_radius_scale,
                    "Probe radius as a fraction of the source-to-edge distance; must be positive.")
            .def_rw("higher_probe_radius_min", &DfrCoherentOptions::higher_probe_radius_min,
                    "Lower clamp of the probe radius.")
            .def_rw("higher_probe_radius_max", &DfrCoherentOptions::higher_probe_radius_max,
                    "Upper clamp of the probe radius.")
            .def_rw("higher_filter_visibility", &DfrCoherentOptions::higher_filter_visibility,
                    "Drop candidate pairs whose edge-to-edge segment is blocked.");

        nb::class_<DfrMaterial>(m, "DfrMaterial",
                                "Electromagnetic surface parameters for the diffraction kernels.\n"
                                "\n"
                                "Input record. Use DfrMaterialAD to differentiate with respect to the material.")
            .def(nb::init<>())
            .def_rw("eta_r", &DfrMaterial::eta_r, "Relative permittivity.")
            .def_rw("sigma", &DfrMaterial::sigma, "Conductivity.")
            .def_rw("mu_r", &DfrMaterial::mu_r, "Relative permeability.")
            .def_rw("gain", &DfrMaterial::gain, "Extra gain factor.")
            .def_rw("valid", &DfrMaterial::valid, "Whether the entry holds a real material.");

        nb::class_<DfrMaterialAD>(m, "DfrMaterialAD",
                                  "DfrMaterial with AD arrays.\n"
                                  "\n"
                                  "Input record for the AD diffraction overloads.")
            .def(nb::init<>())
            .def_rw("eta_r", &DfrMaterialAD::eta_r, "Relative permittivity.")
            .def_rw("sigma", &DfrMaterialAD::sigma, "Conductivity.")
            .def_rw("mu_r", &DfrMaterialAD::mu_r, "Relative permeability.")
            .def_rw("gain", &DfrMaterialAD::gain, "Extra gain factor.")
            .def_rw("valid", &DfrMaterialAD::valid, "Whether the entry holds a real material.");

        nb::class_<DfrCoherentUtdStates>(
            m, "DfrCoherentUtdStates",
            "Deterministic UTD states for coherent diffraction accumulation, one per (source, edge) pair.\n"
            "\n"
            "Input and result record. Produced by Scene.build_dfr_coherent_tx_states() and consumed by "
            "Scene.accum_dfr_coherent_direct(). count is the number of valid states.")
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

        nb::class_<DfrCoherentUtdStatesAD>(
            m, "DfrCoherentUtdStatesAD",
            "DfrCoherentUtdStates with AD arrays. The coherent diffraction operations reject AD inputs.")
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

        nb::class_<DfrCoherentEdge>(m, "DfrCoherentEdge",
                                    "Selected diffraction edges used to build coherent UTD states.\n"
                                    "\n"
                                    "Input record for Scene.build_dfr_coherent_tx_states() and "
                                    "build_dfr_coherent_higher_candidates(). count is the number of valid edges.")
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

        nb::class_<DfrCoherentEdgeAD>(
            m, "DfrCoherentEdgeAD",
            "DfrCoherentEdge with AD arrays. The coherent diffraction operations reject AD inputs.")
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

        nb::class_<DfrCoherentCandidatePairs>(
            m, "DfrCoherentCandidatePairs",
            "Candidate (previous state, next edge) pairs for higher-order coherent diffraction.\n"
            "\n"
            "Result record of Scene.build_dfr_coherent_higher_candidates(). The first count entries of prev_index and "
            "edge_index are valid; visibility_filtered is 1 when the pairs were visibility filtered.")
            .def(nb::init<>())
            .def_rw("count", &DfrCoherentCandidatePairs::count)
            .def_rw("visibility_filtered", &DfrCoherentCandidatePairs::visibility_filtered)
            .def_rw("prev_index", &DfrCoherentCandidatePairs::prev_index)
            .def_rw("edge_index", &DfrCoherentCandidatePairs::edge_index);

        nb::class_<DfrCoherentCandidatePairsAD>(m, "DfrCoherentCandidatePairsAD",
                                                "AD variant of DfrCoherentCandidatePairs.")
            .def(nb::init<>())
            .def_rw("count", &DfrCoherentCandidatePairsAD::count)
            .def_rw("visibility_filtered", &DfrCoherentCandidatePairsAD::visibility_filtered)
            .def_rw("prev_index", &DfrCoherentCandidatePairsAD::prev_index)
            .def_rw("edge_index", &DfrCoherentCandidatePairsAD::edge_index);

        nb::class_<DfrStates>(m, "DfrStates",
                              "Sampled diffraction states: an edge segment plus the source that illuminates it.\n"
                              "\n"
                              "Input record for the diffraction accumulators and Scene.trace_dfr_paths().")
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
            .def_rw("wi", &DfrStates::wi)
            .def_rw("d0", &DfrStates::d0)
            .def_rw("prefix_depth", &DfrStates::prefix_depth);

        nb::class_<DfrStatesAD>(m, "DfrStatesAD",
                                "DfrStates with AD arrays.\n"
                                "\n"
                                "Input record for the differentiable diffraction overloads.")
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
            .def_rw("wi", &DfrStatesAD::wi)
            .def_rw("d0", &DfrStatesAD::d0)
            .def_rw("prefix_depth", &DfrStatesAD::prefix_depth);

        nb::class_<DfrAccum>(
            m, "DfrAccum",
            "Diffraction power and complex field accumulated per grid cell, plus strategy and rejection counters.\n"
            "\n"
            "Result record of Scene.accum_dfr_direct() and Scene.accum_dfr(). Grid arrays have grid_cell_count "
            "entries.")
            .def(nb::init<>())
            .def_rw("grid_cell_count", &DfrAccum::grid_cell_count)
            .def_rw("power", &DfrAccum::power)
            .def_rw("field_x", &DfrAccum::field_x)
            .def_rw("field_y", &DfrAccum::field_y)
            .def_rw("field_z", &DfrAccum::field_z)
            .def_rw("direct_count", &DfrAccum::direct_count)
            .def_rw("keller_count", &DfrAccum::keller_count)
            .def_rw("suffix_count", &DfrAccum::suffix_count)
            .def_rw("vis_rejects", &DfrAccum::vis_rejects)
            .def_rw("edge_vis_rejects", &DfrAccum::edge_vis_rejects)
            .def_rw("utd_rejects", &DfrAccum::utd_rejects)
            .def_rw("edge_uses", &DfrAccum::edge_uses);

        nb::class_<DfrAccumAD>(m, "DfrAccumAD", "AD variant of DfrAccum, returned by the AD overloads.")
            .def(nb::init<>())
            .def_rw("grid_cell_count", &DfrAccumAD::grid_cell_count)
            .def_rw("power", &DfrAccumAD::power)
            .def_rw("field_x", &DfrAccumAD::field_x)
            .def_rw("field_y", &DfrAccumAD::field_y)
            .def_rw("field_z", &DfrAccumAD::field_z)
            .def_rw("direct_count", &DfrAccumAD::direct_count)
            .def_rw("keller_count", &DfrAccumAD::keller_count)
            .def_rw("suffix_count", &DfrAccumAD::suffix_count)
            .def_rw("vis_rejects", &DfrAccumAD::vis_rejects)
            .def_rw("edge_vis_rejects", &DfrAccumAD::edge_vis_rejects)
            .def_rw("utd_rejects", &DfrAccumAD::utd_rejects)
            .def_rw("edge_uses", &DfrAccumAD::edge_uses);

        nb::class_<DfrCoherentAccum>(
            m, "DfrCoherentAccum",
            "Coherent diffraction field accumulated per grid cell.\n"
            "\n"
            "Result record of Scene.accum_dfr_coherent_direct(). Grid arrays have grid_cell_count entries.")
            .def(nb::init<>())
            .def_rw("grid_cell_count", &DfrCoherentAccum::grid_cell_count)
            .def_rw("direct_field_x", &DfrCoherentAccum::direct_field_x)
            .def_rw("direct_field_y", &DfrCoherentAccum::direct_field_y)
            .def_rw("direct_field_z", &DfrCoherentAccum::direct_field_z)
            .def_rw("multi_field_x", &DfrCoherentAccum::multi_field_x)
            .def_rw("multi_field_y", &DfrCoherentAccum::multi_field_y)
            .def_rw("multi_field_z", &DfrCoherentAccum::multi_field_z)
            .def_rw("direct_count", &DfrCoherentAccum::direct_count)
            .def_rw("multi_count", &DfrCoherentAccum::multi_count)
            .def_rw("vis_rejects", &DfrCoherentAccum::visibility_reject_count)
            .def_rw("utd_rejects", &DfrCoherentAccum::utd_reject_count);

        nb::class_<DfrCoherentAccumAD>(m, "DfrCoherentAccumAD", "AD variant of DfrCoherentAccum.")
            .def(nb::init<>())
            .def_rw("grid_cell_count", &DfrCoherentAccumAD::grid_cell_count)
            .def_rw("direct_field_x", &DfrCoherentAccumAD::direct_field_x)
            .def_rw("direct_field_y", &DfrCoherentAccumAD::direct_field_y)
            .def_rw("direct_field_z", &DfrCoherentAccumAD::direct_field_z)
            .def_rw("multi_field_x", &DfrCoherentAccumAD::multi_field_x)
            .def_rw("multi_field_y", &DfrCoherentAccumAD::multi_field_y)
            .def_rw("multi_field_z", &DfrCoherentAccumAD::multi_field_z)
            .def_rw("direct_count", &DfrCoherentAccumAD::direct_count)
            .def_rw("multi_count", &DfrCoherentAccumAD::multi_count)
            .def_rw("vis_rejects", &DfrCoherentAccumAD::visibility_reject_count)
            .def_rw("utd_rejects", &DfrCoherentAccumAD::utd_reject_count);

        nb::class_<DfrPathOptions>(m, "DfrPathOptions",
                                   "Options for Scene.trace_dfr_paths().\n"
                                   "\n"
                                   "Options record. max_paths must be set to a positive value.")
            .def(nb::init<>())
            .def_rw("wavelength", &DfrPathOptions::wavelength, "Wavelength in world units; must be positive.")
            .def_rw("k", &DfrPathOptions::k,
                    "Wavenumber used for path phases. It is used as given, not derived from wavelength.")
            .def_rw("seed", &DfrPathOptions::seed, "Not read by the first-order path export.")
            .def_rw("max_order", &DfrPathOptions::max_order, "Diffraction order; only 1 is supported.")
            .def_rw("max_paths", &DfrPathOptions::max_paths,
                    "Maximum states exported per (transmitter, receiver) pair; must be positive.")
            .def_rw("max_rx", &DfrPathOptions::max_rx, "Use at most this many receivers; 0 uses all.")
            .def_prop_rw("strategy_mask", enum_field_getter<DfrStrategyMask>(&DfrPathOptions::strategy_mask),
                         enum_field_setter<DfrStrategyMask>(&DfrPathOptions::strategy_mask),
                         "Enabled strategies; must include DIRECT.")
            .def_rw("sample_count", &DfrPathOptions::sample_count, "Not read by the first-order path export.")
            .def_rw("return_geom", &DfrPathOptions::return_geom,
                    "Not read by the first-order path export; p0 is always written.")
            .def_prop_rw("receiver_model", enum_field_getter<DfrReceiverModel>(&DfrPathOptions::receiver_model),
                         enum_field_setter<DfrReceiverModel>(&DfrPathOptions::receiver_model),
                         "Receiver model; only MATCHED_ISO is supported.");

        nb::class_<DfrPaths>(m, "DfrPaths",
                             "Exported first-order diffraction paths.\n"
                             "\n"
                             "Result record of Scene.trace_dfr_paths(). Arrays are allocated to capacity, and the "
                             "first count[0] entries are valid paths.")
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

        nb::class_<DfrPathsAD>(m, "DfrPathsAD",
                               "AD variant of DfrPaths, returned by the AD overload of Scene.trace_dfr_paths().")
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

        nb::class_<ReflEpc>(m, "ReflEpc",
                            "Reflection paths corrected to end at the receiver.\n"
                            "\n"
                            "Result record of Scene.trace_refl_epc(). Per-ray arrays have ray_count entries; per-slot "
                            "arrays have ray_count * max_bounces entries.")
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

        nb::class_<ReflEpcAD>(
            m, "ReflEpcAD",
            "AD variant of ReflEpc. Scene.trace_refl_epc() never returns it because it rejects AD inputs.")
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
            m, "ReflEpcField",
            "Reflection paths corrected to end at the receiver, with the complex field at the receiver.\n"
            "\n"
            "Result record of Scene.trace_refl_epc_field(). The field components are split into _re and _im arrays. "
            "Geometry and endpoint arrays are filled only when the matching ReflEpcFieldOptions flags are set.")
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
            .def_ro("resolved_prim_ids", &ReflEpcField::resolved_prim_ids)
            .def_ro("surface_group_ids", &ReflEpcField::surface_group_ids);

        nb::class_<ReflEpcFieldAD>(
            m, "ReflEpcFieldAD",
            "AD variant of ReflEpcField, returned by the differentiable Scene.trace_refl_epc_field() overloads.")
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

        nb::class_<ReflectionBounce>(m, "ReflectionBounce",
                                     "One reflection bounce for a batch of rays.\n"
                                     "\n"
                                     "Result record returned by ReflectionTrace.bounce().")
            .def("is_valid", &ReflectionBounce::is_valid, "Return the per-row validity mask.")
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

        nb::class_<ReflectionBounceAD>(m, "ReflectionBounceAD", "AD variant of ReflectionBounce.")
            .def("is_valid", &ReflectionBounceAD::is_valid, "Return the per-row validity mask.")
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

        nb::class_<ReflectionTrace>(m, "ReflectionTrace",
                                    "Reflection paths stored as one ReflectionBounce per bounce level.\n"
                                    "\n"
                                    "Result record of Scene.trace_reflections() with symbolic=True.")
            .def("is_valid", &ReflectionTrace::is_valid, "Return the per-row validity mask.")
            .def(
                "bounce",
                [](const ReflectionTrace& trace, int index) {
                    if (index < 0 || index >= trace.max_bounces) {
                        throw std::out_of_range("ReflectionTrace.bounce(): index out of range.");
                    }
                    return trace.bounces[static_cast<size_t>(index)];
                },
                "index"_a,
                "Return the ReflectionBounce at bounce level index. An index outside [0, max_bounces) raises "
                "IndexError.")
            .def_ro("max_bounces", &ReflectionTrace::max_bounces)
            .def_ro("ray_count", &ReflectionTrace::ray_count)
            .def_ro("deduplicate_requested", &ReflectionTrace::deduplicate_requested)
            .def_ro("deduplicate_applied", &ReflectionTrace::deduplicate_applied)
            .def_ro("bounce_count", &ReflectionTrace::bounce_count)
            .def_ro("discovery_count", &ReflectionTrace::discovery_count)
            .def_ro("representative_ray_index", &ReflectionTrace::representative_ray_index)
            .def_ro("dedup_keep_mask", &ReflectionTrace::dedup_keep_mask)
            .def_ro("bounces", &ReflectionTrace::bounces);

        nb::class_<ReflectionTraceAD>(m, "ReflectionTraceAD", "AD variant of ReflectionTrace.")
            .def("is_valid", &ReflectionTraceAD::is_valid, "Return the per-row validity mask.")
            .def(
                "bounce",
                [](const ReflectionTraceAD& trace, int index) {
                    if (index < 0 || index >= trace.max_bounces) {
                        throw std::out_of_range("ReflectionTraceAD.bounce(): index out of range.");
                    }
                    return trace.bounces[static_cast<size_t>(index)];
                },
                "index"_a,
                "Return the ReflectionBounce at bounce level index. An index outside [0, max_bounces) raises "
                "IndexError.")
            .def_ro("max_bounces", &ReflectionTraceAD::max_bounces)
            .def_ro("ray_count", &ReflectionTraceAD::ray_count)
            .def_ro("deduplicate_requested", &ReflectionTraceAD::deduplicate_requested)
            .def_ro("deduplicate_applied", &ReflectionTraceAD::deduplicate_applied)
            .def_ro("bounce_count", &ReflectionTraceAD::bounce_count)
            .def_ro("discovery_count", &ReflectionTraceAD::discovery_count)
            .def_ro("representative_ray_index", &ReflectionTraceAD::representative_ray_index)
            .def_ro("dedup_keep_mask", &ReflectionTraceAD::dedup_keep_mask)
            .def_ro("bounces", &ReflectionTraceAD::bounces);

        nb::class_<NearestPointEdge>(m, "NearestPointEdge",
                                     "Nearest scene edge to each query point.\n"
                                     "\n"
                                     "Result record of Scene.nearest_edge() with a point batch. Rows with no edge hold "
                                     "distance = inf and ids = -1.")
            .def("is_valid", &NearestPointEdge::is_valid, "Return the per-row validity mask.")
            .def_ro("distance", &NearestPointEdge::distance)
            .def_ro("point", &NearestPointEdge::point)
            .def_ro("edge_t", &NearestPointEdge::edge_t)
            .def_ro("edge_point", &NearestPointEdge::edge_point)
            .def_ro("shape_id", &NearestPointEdge::shape_id)
            .def_ro("edge_id", &NearestPointEdge::edge_id)
            .def_ro("global_edge_id", &NearestPointEdge::global_edge_id)
            .def_ro("is_boundary", &NearestPointEdge::is_boundary);

        nb::class_<NearestPointEdgeAD>(m, "NearestPointEdgeAD", "AD variant of NearestPointEdge.")
            .def("is_valid", &NearestPointEdgeAD::is_valid, "Return the per-row validity mask.")
            .def_ro("distance", &NearestPointEdgeAD::distance)
            .def_ro("point", &NearestPointEdgeAD::point)
            .def_ro("edge_t", &NearestPointEdgeAD::edge_t)
            .def_ro("edge_point", &NearestPointEdgeAD::edge_point)
            .def_ro("shape_id", &NearestPointEdgeAD::shape_id)
            .def_ro("edge_id", &NearestPointEdgeAD::edge_id)
            .def_ro("global_edge_id", &NearestPointEdgeAD::global_edge_id)
            .def_ro("is_boundary", &NearestPointEdgeAD::is_boundary);

        nb::class_<NearestRayEdge>(
            m, "NearestRayEdge",
            "Nearest scene edge to each query ray or segment.\n"
            "\n"
            "Result record of Scene.nearest_edge() with a Ray. Rows with no edge hold distance = inf and ids = -1.")
            .def("is_valid", &NearestRayEdge::is_valid, "Return the per-row validity mask.")
            .def_ro("distance", &NearestRayEdge::distance)
            .def_ro("ray_t", &NearestRayEdge::ray_t)
            .def_ro("point", &NearestRayEdge::point)
            .def_ro("edge_t", &NearestRayEdge::edge_t)
            .def_ro("edge_point", &NearestRayEdge::edge_point)
            .def_ro("shape_id", &NearestRayEdge::shape_id)
            .def_ro("edge_id", &NearestRayEdge::edge_id)
            .def_ro("global_edge_id", &NearestRayEdge::global_edge_id)
            .def_ro("is_boundary", &NearestRayEdge::is_boundary);

        nb::class_<NearestRayEdgeAD>(m, "NearestRayEdgeAD", "AD variant of NearestRayEdge.")
            .def("is_valid", &NearestRayEdgeAD::is_valid, "Return the per-row validity mask.")
            .def_ro("distance", &NearestRayEdgeAD::distance)
            .def_ro("ray_t", &NearestRayEdgeAD::ray_t)
            .def_ro("point", &NearestRayEdgeAD::point)
            .def_ro("edge_t", &NearestRayEdgeAD::edge_t)
            .def_ro("edge_point", &NearestRayEdgeAD::edge_point)
            .def_ro("shape_id", &NearestRayEdgeAD::shape_id)
            .def_ro("edge_id", &NearestRayEdgeAD::edge_id)
            .def_ro("global_edge_id", &NearestRayEdgeAD::global_edge_id)
            .def_ro("is_boundary", &NearestRayEdgeAD::is_boundary);

        nb::class_<NearestEdgesTopK>(
            m, "NearestEdgesTopK",
            "The k nearest scene edges to each query point.\n"
            "\n"
            "Result record of Scene.nearest_edges(). Arrays have query_count * k slots, each query's k results "
            "contiguous and ordered nearest first; valid marks filled slots.")
            .def_ro("query_count", &NearestEdgesTopK::query_count)
            .def_ro("k", &NearestEdgesTopK::k)
            .def_ro("valid", &NearestEdgesTopK::is_valid)
            .def(
                "is_valid", [](const NearestEdgesTopK& self) { return self.is_valid; },
                "Return the per-row validity mask.")
            .def_ro("distances", &NearestEdgesTopK::distances)
            .def_ro("points", &NearestEdgesTopK::points)
            .def_ro("edge_t", &NearestEdgesTopK::edge_t)
            .def_ro("edge_points", &NearestEdgesTopK::edge_points)
            .def_ro("shape_ids", &NearestEdgesTopK::shape_ids)
            .def_ro("edge_ids", &NearestEdgesTopK::edge_ids)
            .def_ro("global_edge_ids", &NearestEdgesTopK::global_edge_ids)
            .def_ro("is_boundary", &NearestEdgesTopK::is_boundary);

        nb::class_<NearestEdgesTopKAD>(m, "NearestEdgesTopKAD", "AD variant of NearestEdgesTopK.")
            .def_ro("query_count", &NearestEdgesTopKAD::query_count)
            .def_ro("k", &NearestEdgesTopKAD::k)
            .def_ro("valid", &NearestEdgesTopKAD::is_valid)
            .def(
                "is_valid", [](const NearestEdgesTopKAD& self) { return self.is_valid; },
                "Return the per-row validity mask.")
            .def_ro("distances", &NearestEdgesTopKAD::distances)
            .def_ro("points", &NearestEdgesTopKAD::points)
            .def_ro("edge_t", &NearestEdgesTopKAD::edge_t)
            .def_ro("edge_points", &NearestEdgesTopKAD::edge_points)
            .def_ro("shape_ids", &NearestEdgesTopKAD::shape_ids)
            .def_ro("edge_ids", &NearestEdgesTopKAD::edge_ids)
            .def_ro("global_edge_ids", &NearestEdgesTopKAD::global_edge_ids)
            .def_ro("is_boundary", &NearestEdgesTopKAD::is_boundary);

        nb::class_<SegmentVisibility>(m, "SegmentVisibility",
                                      "Per-segment visibility mask.\n"
                                      "\n"
                                      "Result record of Scene.visible() and MixedScene.visible(). visible is True "
                                      "where the endpoints see each other.")
            .def_ro("ray_count", &SegmentVisibility::ray_count)
            .def_ro("visible", &SegmentVisibility::visible);

        nb::class_<SegmentVisibilityAD>(
            m, "SegmentVisibilityAD", "AD variant of SegmentVisibility. Visibility is discrete and has no derivative.")
            .def_ro("ray_count", &SegmentVisibilityAD::ray_count)
            .def_ro("visible", &SegmentVisibilityAD::visible);

        nb::class_<SegmentPairVisibility>(m, "SegmentPairVisibility",
                                          "Visibility of two endpoints from a shared start point.\n"
                                          "\n"
                                          "Result record of Scene.visible_pair().")
            .def_ro("ray_count", &SegmentPairVisibility::ray_count)
            .def_ro("visible_a", &SegmentPairVisibility::visible_a)
            .def_ro("visible_b", &SegmentPairVisibility::visible_b);

        nb::class_<SegmentPairVisibilityAD>(
            m, "SegmentPairVisibilityAD",
            "AD variant of SegmentPairVisibility. Visibility is discrete and has no derivative.")
            .def_ro("ray_count", &SegmentPairVisibilityAD::ray_count)
            .def_ro("visible_a", &SegmentPairVisibilityAD::visible_a)
            .def_ro("visible_b", &SegmentPairVisibilityAD::visible_b);

        nb::class_<AxialEdgeVisibility>(m, "AxialEdgeVisibility",
                                        "Whether the source sees any sampled point of each edge.\n"
                                        "\n"
                                        "Result record of Scene.visible_edge().")
            .def_ro("state_count", &AxialEdgeVisibility::state_count)
            .def_ro("any_visible", &AxialEdgeVisibility::any_visible);

        nb::class_<AxialEdgeVisibilityAD>(
            m, "AxialEdgeVisibilityAD",
            "AD variant of AxialEdgeVisibility. Visibility is discrete and has no derivative.")
            .def_ro("state_count", &AxialEdgeVisibilityAD::state_count)
            .def_ro("any_visible", &AxialEdgeVisibilityAD::any_visible);

        nb::class_<SegmentChainVisibility>(
            m, "SegmentChainVisibility",
            "Visibility of every segment of each polyline, with the first blocker.\n"
            "\n"
            "Result record of Scene.visible_chain(). Blocker ids are scene-global primitive ids.")
            .def_ro("chain_count", &SegmentChainVisibility::chain_count)
            .def_ro("max_segments", &SegmentChainVisibility::max_segments)
            .def_ro("all_visible", &SegmentChainVisibility::all_visible)
            .def_ro("first_blocked_segment", &SegmentChainVisibility::first_blocked_segment)
            .def_ro("first_blocked_prim", &SegmentChainVisibility::first_blocked_prim);

        nb::class_<SegmentChainVisibilityAD>(
            m, "SegmentChainVisibilityAD",
            "AD variant of SegmentChainVisibility. Visibility is discrete and has no derivative.")
            .def_ro("chain_count", &SegmentChainVisibilityAD::chain_count)
            .def_ro("max_segments", &SegmentChainVisibilityAD::max_segments)
            .def_ro("all_visible", &SegmentChainVisibilityAD::all_visible)
            .def_ro("first_blocked_segment", &SegmentChainVisibilityAD::first_blocked_segment)
            .def_ro("first_blocked_prim", &SegmentChainVisibilityAD::first_blocked_prim);

        nb::class_<SceneSyncProfile>(m, "SceneSyncProfile",
                                     "Timings (milliseconds) and update counts of the last Scene.sync().\n"
                                     "\n"
                                     "Result record returned by Scene.last_sync_profile.")
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

        nb::class_<SceneEdgeBVHStats>(m, "SceneEdgeBVHStats",
                                      "Structure and quality metrics of the built edge BVH, for diagnostics.\n"
                                      "\n"
                                      "Result record returned by Scene.edge_bvh_stats().")
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
            .def_ro("sibling_overlap_surface_area_sum", &SceneEdgeBVHStats::sibling_overlap_surface_area_sum)
            .def_ro("sibling_overlap_surface_area_avg", &SceneEdgeBVHStats::sibling_overlap_surface_area_avg)
            .def_ro("normalized_sibling_overlap", &SceneEdgeBVHStats::normalized_sibling_overlap)
            .def_ro("leaf_size_histogram", &SceneEdgeBVHStats::leaf_size_histogram);
    });

    bind_section("surfel", [&]() {
        nb::class_<SurfelGeometry>(
            m, "SurfelGeometry",
            "Surfel positions and tangent axes.\n"
            "\n"
            "Input record for SurfelScene. Each surfel is a 2D Gaussian centered at center whose local axes are "
            "tangent_u and tangent_v. All three arrays have one entry per surfel.")
            .def(nb::init<>())
            .def(nb::init<const Vector3f&, const Vector3f&, const Vector3f&>(), "center"_a, "tangent_u"_a,
                 "tangent_v"_a, "Create surfel geometry from per-surfel center, tangent_u, and tangent_v arrays.")
            .def(nb::init<const Vector3fAD&, const Vector3fAD&, const Vector3fAD&>(), "center"_a, "tangent_u"_a,
                 "tangent_v"_a)
            .def_prop_ro("surfel_count", &SurfelGeometry::surfel_count, "Number of surfels.")
            .def_prop_ro("center", &SurfelGeometry::center, "Surfel centers.")
            .def_prop_ro("tangent_u", &SurfelGeometry::tangent_u,
                         "First tangent axis of each surfel; its length is the Gaussian scale along it.")
            .def_prop_ro("tangent_v", &SurfelGeometry::tangent_v,
                         "Second tangent axis of each surfel; its length is the Gaussian scale along it.");

        nb::class_<SurfelAppearance>(
            m, "SurfelAppearance",
            "Per-surfel opacity and color values.\n"
            "\n"
            "Input record for SurfelScene.update_appearance(). values holds surfel_count * values_per_surfel entries, "
            "where values_per_surfel is channel_count, or 3 * (sh_degree + 1)^2 for the SH model.")
            .def(nb::init<>())
            .def(nb::init<const Float&, const Float&, SurfelColorModel, int, int>(), "opacity"_a, "values"_a,
                 "color_model"_a, "channel_count"_a, "sh_degree"_a = 0,
                 "Create an appearance from opacity (one per surfel) and values laid out for color_model.")
            .def(nb::init<const FloatAD&, const FloatAD&, SurfelColorModel, int, int>(), "opacity"_a, "values"_a,
                 "color_model"_a, "channel_count"_a, "sh_degree"_a = 0)
            .def_static("rgb", nb::overload_cast<const Float&, const Vector3f&>(&SurfelAppearance::rgb), "opacity"_a,
                        "rgb"_a, "Return a ConstantRGB appearance from per-surfel opacity and RGB color.")
            .def_static("rgb", nb::overload_cast<const FloatAD&, const Vector3fAD&>(&SurfelAppearance::rgb),
                        "opacity"_a, "rgb"_a)
            .def_static("features", nb::overload_cast<const Float&, const Float&, int>(&SurfelAppearance::features),
                        "opacity"_a, "values"_a, "channel_count"_a,
                        "Return a FeatureChannels appearance with channel_count values per surfel.")
            .def_static("features", nb::overload_cast<const FloatAD&, const FloatAD&, int>(&SurfelAppearance::features),
                        "opacity"_a, "values"_a, "channel_count"_a)
            .def_static(
                "sh", nb::overload_cast<const Float&, const Float&, int>(&SurfelAppearance::sh), "opacity"_a,
                "coeffs"_a, "sh_degree"_a,
                "Return an SH appearance from RGB spherical-harmonic coefficients of degree sh_degree in [0, 3].")
            .def_static("sh", nb::overload_cast<const FloatAD&, const FloatAD&, int>(&SurfelAppearance::sh),
                        "opacity"_a, "coeffs"_a, "sh_degree"_a)
            .def_prop_ro("surfel_count", &SurfelAppearance::surfel_count, "Number of surfels.")
            .def_prop_ro("channel_count", &SurfelAppearance::channel_count, "Number of color channels.")
            .def_prop_ro("sh_degree", &SurfelAppearance::sh_degree,
                         "Spherical-harmonic degree; 0 for the non-SH models.")
            .def_prop_ro("color_model", &SurfelAppearance::color_model, "Color model of values.")
            .def_prop_ro("opacity", &SurfelAppearance::opacity, "Per-surfel opacity.")
            .def_prop_ro("values", &SurfelAppearance::values, "Flat per-surfel color values.");

        nb::class_<SurfelCloud>(m, "SurfelCloud",
                                "Surfel geometry with one opacity and one scalar value per surfel.\n"
                                "\n"
                                "Input record for SurfelScene and MixedScene.add_surfel().")
            .def(nb::init<>())
            .def(
                "__init__",
                [](SurfelCloud* cloud, const Vector3f& center, const Vector3f& tangent_u, const Vector3f& tangent_v,
                   const Float& opacity,
                   const Float& value) { new (cloud) SurfelCloud(center, tangent_u, tangent_v, opacity, value); },
                "center"_a, "tangent_u"_a, "tangent_v"_a, "opacity"_a = Float(), "value"_a = Float(),
                "Create a surfel cloud. opacity and value default to 1 for every surfel; when given they need one "
                "entry per surfel.")
            .def(
                "__init__",
                [](SurfelCloud* cloud, const Vector3fAD& center, const Vector3fAD& tangent_u,
                   const Vector3fAD& tangent_v, const FloatAD& opacity,
                   const FloatAD& value) { new (cloud) SurfelCloud(center, tangent_u, tangent_v, opacity, value); },
                "center"_a, "tangent_u"_a, "tangent_v"_a, "opacity"_a = FloatAD(), "value"_a = FloatAD())
            .def_prop_ro("surfel_count", &SurfelCloud::surfel_count, "Number of surfels.")
            .def_prop_ro("center", &SurfelCloud::center, "Surfel centers.")
            .def_prop_ro("tangent_u", &SurfelCloud::tangent_u, "First tangent axis of each surfel.")
            .def_prop_ro("tangent_v", &SurfelCloud::tangent_v, "Second tangent axis of each surfel.")
            .def_prop_ro("opacity", &SurfelCloud::opacity, "Per-surfel opacity.")
            .def_prop_ro("value", &SurfelCloud::value, "Per-surfel scalar value.");

        nb::class_<SurfelScene>(
            m, "SurfelScene",
            "Ray-tracing scene of 2D Gaussian surfels, traced through proxy triangles with OptiX.\n"
            "\n"
            "Build it with build() before querying. Queries must run on the CUDA device it was built on.")
            .def(nb::init<const SurfelCloud&, const SurfelTraceOptions&>(), "cloud"_a,
                 "options"_a = SurfelTraceOptions(),
                 "Create a surfel scene from a SurfelCloud or a SurfelGeometry. A SurfelGeometry gets opacity and "
                 "value 1. Invalid options raise RuntimeError.")
            .def(nb::init<const SurfelGeometry&, const SurfelTraceOptions&>(), "geometry"_a,
                 "options"_a = SurfelTraceOptions())
            .def("build", &SurfelScene::build,
                 "Build the proxy triangles and the OptiX acceleration structure on the current CUDA device.")
            .def("is_ready", &SurfelScene::is_ready, "Return True once build() has run.")
            .def("update_geometry", &SurfelScene::update_geometry, "geometry"_a,
                 "Replace the surfel geometry and rebuild the acceleration structure. The scene must be built and the "
                 "surfel count must not change.")
            .def("update_appearance", &SurfelScene::update_appearance, "appearance"_a,
                 "Replace the surfel appearance; the surfel count must not change.")
            .def_prop_ro("surfel_count", &SurfelScene::surfel_count, "Number of surfels.")
            .def_prop_ro("triangle_count", &SurfelScene::triangle_count, "Number of proxy triangles.")
            .def_prop_ro("build_count", &SurfelScene::build_count,
                         "Number of builds so far, update_geometry() included.")
            .def(
                "intersect",
                [](const SurfelScene& scene, const Ray& ray, rayd::Mask active) {
                    return scene.intersect<true>(ray, active);
                },
                nb::arg("ray").noconvert(), "active"_a = true,
                "Return the closest surfel hit per ray as a SurfelIntersection.\n"
                "\n"
                "A RayAD returns SurfelIntersectionAD with the hit attributes recomputed from the AD surfel "
                "parameters.")
            .def(
                "intersect",
                [](const SurfelScene& scene, const RayAD& ray, rayd::MaskAD active) {
                    return scene.intersect<false>(ray, active);
                },
                nb::arg("ray").noconvert(), "active"_a = true)
            .def(
                "composite_alpha",
                [](const SurfelScene& scene, const Ray& ray, rayd::Mask active) {
                    return scene.composite_alpha<true>(ray, active);
                },
                nb::arg("ray").noconvert(), "active"_a = true,
                "Alpha-composite the surfels along each ray, front to back, and return a SurfelComposite.")
            .def(
                "composite_alpha",
                [](const SurfelScene& scene, const RayAD& ray, rayd::MaskAD active) {
                    return scene.composite_alpha<false>(ray, active);
                },
                nb::arg("ray").noconvert(), "active"_a = true)
            .def(
                "composite_alpha_reference",
                [](const SurfelScene& scene, const Ray& ray, rayd::Mask active) {
                    return scene.composite_alpha_reference<true>(ray, active);
                },
                nb::arg("ray").noconvert(), "active"_a = true,
                "Compute the same composite as composite_alpha() with the reference implementation instead of the "
                "native pipeline.")
            .def(
                "composite_alpha_reference",
                [](const SurfelScene& scene, const RayAD& ray, rayd::MaskAD active) {
                    return scene.composite_alpha_reference<false>(ray, active);
                },
                nb::arg("ray").noconvert(), "active"_a = true)
            .def(
                "render",
                [](const SurfelScene& scene, const Ray& ray, const SurfelRenderOptions& render_options,
                   rayd::Mask active) { return scene.render<true>(ray, render_options, active); },
                nb::arg("ray").noconvert(), "render_options"_a = SurfelRenderOptions(), "active"_a = true,
                "Render each ray with render_options and return a SurfelRender.\n"
                "\n"
                "render_options must match the appearance: its color model, an SH degree no higher than the "
                "appearance's, and no more feature channels than it has. A mismatch raises RuntimeError.")
            .def(
                "render",
                [](const SurfelScene& scene, const RayAD& ray, const SurfelRenderOptions& render_options,
                   rayd::MaskAD active) { return scene.render<false>(ray, render_options, active); },
                nb::arg("ray").noconvert(), "render_options"_a = SurfelRenderOptions(), "active"_a = true)
            .def(
                "shadow_test",
                [](const SurfelScene& scene, const Ray& ray, rayd::Mask active) {
                    return scene.shadow_test<true>(ray, active);
                },
                nb::arg("ray").noconvert(), "active"_a = true,
                "Return a mask that is True where the ray hits any surfel.")
            .def(
                "shadow_test",
                [](const SurfelScene& scene, const RayAD& ray, rayd::MaskAD active) {
                    return scene.shadow_test<false>(ray, active);
                },
                nb::arg("ray").noconvert(), "active"_a = true)
            .def(
                "visible",
                [](const SurfelScene& scene, const Vector3f& start, const Vector3f& end, rayd::Mask active) {
                    return scene.visible<true>(start, end, active);
                },
                "start"_a, "end"_a, "active"_a = true,
                "Return a mask that is True where no surfel blocks the segment from start to end. Degenerate segments "
                "are not visible.")
            .def(
                "visible",
                [](const SurfelScene& scene, const Vector3fAD& start, const Vector3fAD& end, rayd::MaskAD active) {
                    return scene.visible<false>(start, end, active);
                },
                "start"_a, "end"_a, "active"_a = true)
            .def(
                "trace_reflections",
                [](const SurfelScene& scene, const Ray& ray, int max_bounces, rayd::Mask active) {
                    return scene.trace_reflections<true>(ray, max_bounces, active);
                },
                nb::arg("ray").noconvert(), "max_bounces"_a, "active"_a = true,
                "Trace specular reflections off the surfels for up to max_bounces bounces and return a "
                "ReflectionChain. max_bounces must be non-negative.")
            .def(
                "trace_reflections",
                [](const SurfelScene& scene, const RayAD& ray, int max_bounces, rayd::MaskAD active) {
                    return scene.trace_reflections<false>(ray, max_bounces, active);
                },
                nb::arg("ray").noconvert(), "max_bounces"_a, "active"_a = true);
    });

    bind_section("sdf", [&]() {
        nb::class_<SdfGrid>(m, "SdfGrid",
                            "Signed distance field sampled on a regular grid inside an oriented box.\n"
                            "\n"
                            "Standalone geometry for ray queries; it can also be added to a MixedScene.")
            .def(nb::init<const Float&, int, int, int, const Vector3f&, const Float&, const Vector3f&>(), "values"_a,
                 "nx"_a, "ny"_a, "nz"_a, "position"_a, "rotation"_a, "scale"_a,
                 "Create a grid from vertex-centered samples.\n"
                 "\n"
                 "Args:\n"
                 "    values: nx * ny * nz signed distances in world units, negative inside, flattened with z fastest: "
                 "index (i * ny + j) * nz + k.\n"
                 "    nx, ny, nz: Sample counts per axis, each at least 2.\n"
                 "    position: World-space center of the box (one vector).\n"
                 "    rotation: Box rotation as a unit quaternion of four entries in (w, x, y, z) order.\n"
                 "    scale: Full side lengths of the box (one vector).\n"
                 "\n"
                 "AD arrays make the grid differentiable with respect to values and placement. Invalid shapes raise "
                 "RuntimeError.")
            .def(nb::init<const FloatAD&, int, int, int, const Vector3fAD&, const FloatAD&, const Vector3fAD&>(),
                 "values"_a, "nx"_a, "ny"_a, "nz"_a, "position"_a, "rotation"_a, "scale"_a)
            .def_prop_ro("nx", &SdfGrid::nx, "Sample count along x.")
            .def_prop_ro("ny", &SdfGrid::ny, "Sample count along y.")
            .def_prop_ro("nz", &SdfGrid::nz, "Sample count along z.")
            .def_prop_ro("value_count", &SdfGrid::value_count, "Total sample count, nx * ny * nz.")
            .def_prop_ro("values", &SdfGrid::values, "Flat signed-distance samples.")
            .def_prop_ro("position", &SdfGrid::position, "World-space box center.")
            .def_prop_ro("rotation", &SdfGrid::rotation, "Box rotation quaternion in (w, x, y, z) order.")
            .def_prop_ro("scale", &SdfGrid::scale, "Box side lengths.")
            .def(
                "intersect",
                [](const SdfGrid& grid, const Ray& ray, const SdfTraceOptions& options, rayd::Mask active) {
                    return grid.intersect<true>(ray, options, active);
                },
                nb::arg("ray").noconvert(), "options"_a = SdfTraceOptions(), "active"_a = true,
                "Sphere-trace each ray against the zero level set and return an SdfIntersection.\n"
                "\n"
                "Tracing is limited to the oriented box and ray.tmax. With a RayAD the result is differentiable with "
                "respect to the ray, the values, and the placement (position, rotation, scale); the hit is fixed by "
                "the forward pass.")
            .def(
                "intersect",
                [](const SdfGrid& grid, const RayAD& ray, const SdfTraceOptions& options, rayd::MaskAD active) {
                    return grid.intersect<false>(ray, options, active);
                },
                nb::arg("ray").noconvert(), "options"_a = SdfTraceOptions(), "active"_a = true)
            .def(
                "visible",
                [](const SdfGrid& grid, const Vector3f& start, const Vector3f& end, const SdfTraceOptions& options,
                   rayd::Mask active) { return grid.visible<true>(start, end, options, active); },
                "start"_a, "end"_a, "options"_a = SdfTraceOptions(), "active"_a = true,
                "Return a mask that is True where the SDF surface does not block the segment from start to end.")
            .def(
                "visible",
                [](const SdfGrid& grid, const Vector3fAD& start, const Vector3fAD& end, const SdfTraceOptions& options,
                   rayd::MaskAD active) { return grid.visible<false>(start, end, options, active); },
                "start"_a, "end"_a, "options"_a = SdfTraceOptions(), "active"_a = true)
            .def(
                "trace_reflections",
                [](const SdfGrid& grid, const Ray& ray, int max_bounces, const SdfTraceOptions& options,
                   rayd::Mask active) { return grid.trace_reflections<true>(ray, max_bounces, options, active); },
                nb::arg("ray").noconvert(), "max_bounces"_a, "options"_a = SdfTraceOptions(), "active"_a = true,
                "Trace specular reflections off the SDF surface for up to max_bounces bounces and return a "
                "ReflectionChain. max_bounces must be non-negative.")
            .def(
                "trace_reflections",
                [](const SdfGrid& grid, const RayAD& ray, int max_bounces, const SdfTraceOptions& options,
                   rayd::MaskAD active) { return grid.trace_reflections<false>(ray, max_bounces, options, active); },
                nb::arg("ray").noconvert(), "max_bounces"_a, "options"_a = SdfTraceOptions(), "active"_a = true);
    });

    bind_section("mesh", [&]() {
        nb::class_<Mesh>(m, "Mesh",
                         "Triangle mesh with optional UVs and a world transform.\n"
                         "\n"
                         "Input geometry for Scene.add_mesh() and MixedScene.add_mesh(). The world transform is "
                         "to_world_left @ to_world @ to_world_right.")
            .def(nb::init<>())
            .def(
                "__init__",
                [](Mesh* mesh, nb::handle v_obj, const Vector3i& f, nb::handle uv_obj, const Vector3i& f_uv,
                   bool verbose) {
                    const std::string v_module_name = nb::cast<std::string>(v_obj.type().attr("__module__"));
                    const std::string v_type_name = nb::cast<std::string>(v_obj.type().attr("__name__"));
                    const std::string uv_module_name = nb::cast<std::string>(uv_obj.type().attr("__module__"));
                    const std::string uv_type_name = nb::cast<std::string>(uv_obj.type().attr("__name__"));

                    const bool has_live_v = v_module_name == "drjit.cuda.ad" && v_type_name == "Array3f";
                    const bool has_live_uv = uv_module_name == "drjit.cuda.ad" && uv_type_name == "Array2f";
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
                "v"_a, "f"_a, "uv"_a = Vector2f(), "f_uv"_a = Vector3i(), "verbose"_a = false,
                "Create a mesh.\n"
                "\n"
                "Args:\n"
                "    v: Object-space vertex positions, one entry per vertex.\n"
                "    f: Vertex indices of each triangle.\n"
                "    uv: Optional texture coordinates, indexed by f_uv when given and by vertex otherwise.\n"
                "    f_uv: Optional UV indices of each triangle; when omitted, uv needs one entry per vertex.\n"
                "    verbose: Print the vertex, face, and edge counts after loading.\n"
                "\n"
                "Grad-enabled drjit.cuda.ad positions or UVs are kept as AD arrays so gradients reach them.")
            .def(
                "__init__",
                [](Mesh* mesh, const Vector3f& v, const Vector3i& f, const Vector2f& uv, const Vector3i& f_uv,
                   bool verbose) { new (mesh) Mesh(v, f, uv, f_uv, verbose); },
                "v"_a, "f"_a, "uv"_a = Vector2f(), "f_uv"_a = Vector3i(), "verbose"_a = false)
            .def("build", &Mesh::build, "Build the mesh's derived geometry and GPU buffers from its current state.")
            .def("set_transform", &Mesh::set_transform, "mat"_a, "set_left"_a = true,
                 "Replace the left (set_left=True) or right factor of the world transform with mat.")
            .def(
                "append_transform", &Mesh::append_transform, "mat"_a, "append_left"_a = true,
                "Compose mat onto the left (append_left=True, pre-multiply) or right (post-multiply) transform factor.")
            .def(
                "edge_indices",
                [](const Mesh& mesh) {
                    const auto& edge_indices = mesh.edge_indices();
                    return nb::make_tuple(edge_indices[0], edge_indices[1], edge_indices[2], edge_indices[3],
                                          edge_indices[4]);
                },
                "Return the per-edge connectivity as five Int arrays: (vertex0, vertex1, face0, face1, "
                "opposite_vertex).\n"
                "\n"
                "face1 is -1 for boundary edges. Raises RuntimeError if the faces changed since build() or admit no "
                "supported edge topology.")
            .def(
                "secondary_edges",
                [](const Mesh& mesh) {
                    return mesh.secondary_edge_info() != nullptr ? *mesh.secondary_edge_info() : SecondaryEdgeInfoAD();
                },
                "Return the world-space per-edge data as SecondaryEdgeInfo, or an empty record before it is computed.")
            .def_prop_ro("num_vertices", &Mesh::vertex_count, "Number of vertices.")
            .def_prop_ro("num_faces", &Mesh::face_count, "Number of triangles.")
            .def_prop_rw("to_world", &Mesh::to_world, &Mesh::set_to_world, "Middle factor of the world transform.")
            .def_prop_rw("to_world_left", &Mesh::to_world_left, &Mesh::set_to_world_left,
                         "Left factor of the world transform.")
            .def_prop_rw("to_world_right", &Mesh::to_world_right, &Mesh::set_to_world_right,
                         "Right factor of the world transform.")
            .def_prop_rw("vertex_positions", &Mesh::vertex_positions, &Mesh::set_vertex_positions,
                         "Object-space vertex positions.")
            .def_prop_ro("vertex_positions_world", &Mesh::vertex_positions_world, "World-space vertex positions.")
            .def_prop_ro("vertex_normals", &Mesh::vertex_normals,
                         "Object-space area-weighted vertex normals, filled by build().")
            .def_prop_rw("vertex_uv", &Mesh::vertex_uv, &Mesh::set_vertex_uv, "Texture coordinates.")
            .def_prop_rw("face_indices", &Mesh::face_indices, &Mesh::set_face_indices,
                         "Vertex indices of each triangle.")
            .def_prop_rw("face_uv_indices", &Mesh::face_uv_indices, &Mesh::set_face_uv_indices,
                         "UV indices of each triangle.")
            .def_prop_rw("use_face_normals", &Mesh::use_face_normals, &Mesh::set_use_face_normals,
                         "Use the geometric face normal as the shading normal.")
            .def_prop_rw("edges_enabled", &Mesh::edges_enabled, &Mesh::set_edges_enabled,
                         "Include this mesh's edges in the scene edge set used by edge queries.")
            .def("__repr__", &Mesh::to_string);
    });

    bind_section("scene", [&]() {
        nb::class_<Scene>(m, "Scene",
                          "Triangle-mesh scene with its ray-tracing and edge acceleration structures.\n"
                          "\n"
                          "Add meshes with add_mesh() or add_instance(), call build(), then run queries. Dynamic "
                          "meshes can be edited and refitted with sync().")
            .def(
                "__init__",
                [](Scene* scene, const std::string& trace_backend, const std::string& edge_bvh_backend) {
                    new (scene) Scene(edge_bvh_backend, trace_backend);
                },
                nb::kw_only(), "trace_backend"_a = "auto", "edge_bvh_backend"_a = "auto",
                "Create an empty scene.\n"
                "\n"
                "Args:\n"
                "    trace_backend: Triangle trace backend. 'auto' selects 'optix' when OptiX is available and 'cuda' "
                "otherwise; 'optix' requires OptiX at build(); 'cuda' uses a CUDA BVH; 'none' builds no triangle "
                "backend, which leaves only the edge queries.\n"
                "    edge_bvh_backend: Nearest-edge backend: 'auto' (same as 'drjit'), 'drjit', 'optix', or "
                "'optix_drjit'.\n"
                "\n"
                "Backend strings match exactly and case-sensitively; any other value raises ValueError.")
            .def("add_mesh", &Scene::add_mesh, "mesh"_a, "dynamic"_a = false,
                 "Add a copy of mesh and return its mesh id.\n"
                 "\n"
                 "Adding a mesh invalidates a previous build(). Pass dynamic=True to allow later "
                 "update_mesh_vertices() and transform edits applied by sync().")
            .def("add_instance", &Scene::add_instance, "geometry_id"_a, "transform"_a, "dynamic"_a = true,
                 "Add a transformed instance of an existing static mesh and return its instance id.\n"
                 "\n"
                 "The instance shares the acceleration structure of mesh geometry_id but has its own 4x4 transform and "
                 "its own scene-global primitive range. A dynamic instance can be moved with set_mesh_transform() or "
                 "append_mesh_transform() followed by sync(). Instancing a dynamic mesh raises RuntimeError.")
            .def("build", &Scene::build,
                 "Build the triangle and edge acceleration structures. Call it before any query.\n"
                 "\n"
                 "Raises RuntimeError when the scene has no meshes or no triangles, or when a world-space vertex "
                 "coordinate is NaN or infinite.")
            .def("update_mesh_vertices", &Scene::update_mesh_vertices, "mesh_id"_a, "positions"_a,
                 "Queue new object-space vertex positions for dynamic mesh mesh_id; sync() applies them.\n"
                 "\n"
                 "positions needs one entry per mesh vertex. Raises RuntimeError if the scene is not built or the mesh "
                 "is not dynamic.")
            .def("set_mesh_transform", &Scene::set_mesh_transform, "mesh_id"_a, "mat"_a, "set_left"_a = true,
                 "Queue a new left (set_left=True) or right transform factor for dynamic mesh mesh_id; sync() applies "
                 "it.\n"
                 "\n"
                 "Raises RuntimeError if the scene is not built or the mesh is not dynamic.")
            .def("append_mesh_transform", &Scene::append_mesh_transform, "mesh_id"_a, "mat"_a, "append_left"_a = true,
                 "Queue mat composed onto the left (append_left=True) or right transform factor of dynamic mesh "
                 "mesh_id; sync() applies it.\n"
                 "\n"
                 "Raises RuntimeError if the scene is not built or the mesh is not dynamic.")
            .def("set_edge_mask", nb::overload_cast<const rayd::Mask&>(&Scene::set_edge_mask), nb::arg("mask"),
                 "Restrict the edge queries to the edges where mask is True.\n"
                 "\n"
                 "mask has one entry per scene-global edge. Raises RuntimeError if the scene is not built or the size "
                 "does not match.")
            .def("set_edge_mask", nb::overload_cast<const rayd::MaskAD&>(&Scene::set_edge_mask), nb::arg("mask"))
            .def("sync", &Scene::sync,
                 "Apply the queued dynamic-mesh edits and refit the acceleration structures in place.\n"
                 "\n"
                 "Raises RuntimeError if the scene is not built or an updated vertex is NaN or infinite; a failed sync "
                 "keeps its updates pending.")
            .def("is_ready", &Scene::is_ready, "Return True once the scene is built and needs no rebuild.")
            .def("has_pending_updates", &Scene::has_pending_updates,
                 "Return True when edits are queued that sync() has not applied yet.")
            .def_prop_ro("last_sync_profile", &Scene::last_sync_profile,
                         "Timings and update counts of the last sync().")
            .def("edge_info", &Scene::edge_info,
                 "Return the scene-global edge geometry and ids as SceneEdgeInfo. Raises RuntimeError if the scene is "
                 "not built or has pending updates (call sync() first).")
            .def_prop_ro("edge_bvh_backend", &Scene::edge_bvh_backend,
                         "Resolved edge backend name: 'drjit', 'optix', or 'optix_drjit'.")
            .def_prop_ro(
                "trace_backend", [](const Scene& scene) { return trace_backend_name(scene.trace_backend_kind()); },
                "Canonical name of the resolved triangle trace backend "
                "('optix', 'cuda', or 'none').")
            .def(
                "capabilities",
                [](const Scene& scene) {
                    const TraceBackendKind kind = scene.trace_backend_kind();
                    const bool has_optix = kind == TraceBackendKind::Optix;
                    const bool has_cuda = kind == TraceBackendKind::Cuda;
                    const bool has_trace = has_optix || has_cuda;
                    // Both trace backends serve the full multipath surface: OptiX
                    // through its pipelines, CUDA through the P4 fused executor
                    // (visibility / reflection trace + accumulation / diffraction
                    // paths + accumulation / EPC).
                    const bool multipath = has_trace;
                    nb::dict caps;
                    caps["trace_backend"] = trace_backend_name(kind);
                    caps["optix_available"] = optix_available();
                    caps["edge_backend"] = scene.edge_bvh_backend();
                    nb::list integration;
                    if (has_optix) {
                        integration.append("jit_symbolic");
                        integration.append("eager_native");
                    } else if (has_cuda) {
                        integration.append("eager_native");
                    }
                    caps["integration"] = integration;
                    caps["intersect"] = has_trace;
                    caps["shadow_test"] = has_trace;
                    caps["visibility"] = multipath;
                    caps["reflection_trace"] = multipath;
                    caps["reflection_accumulation"] = multipath;
                    caps["diffraction"] = multipath;
                    caps["epc"] = multipath;
                    caps["nearest_edge"] = true;
                    return caps;
                },
                "Machine-readable capability map for this scene's resolved "
                "backend plan (trace backend, edge backend, integration modes, "
                "and per-operation availability).")
            .def("edge_bvh_stats", &Scene::edge_bvh_stats,
                 "Return structure and quality metrics of the edge BVH. Raises RuntimeError if the scene is not built "
                 "or has pending updates (call sync() first).")
            .def("edge_topology", &Scene::edge_topology,
                 "Return the scene-global edge connectivity as SceneEdgeTopology.")
            .def("edge_mask", &Scene::edge_mask, "Return the per-edge mask set by set_edge_mask().")
            .def("mesh_face_offsets", &Scene::mesh_face_offsets,
                 "Return the prefix sum of face counts: mesh m owns faces [offset[m], offset[m + 1]).")
            .def("mesh_edge_offsets", &Scene::mesh_edge_offsets, "Return the prefix sum of per-mesh edge counts.")
            .def("mesh_vertex_offsets", &Scene::mesh_vertex_offsets, "Return the prefix sum of per-mesh vertex counts.")
            .def("global_geometry", &Scene::global_geometry,
                 "Return the vertices, faces, normals, and ids of all meshes in one scene-global index space. Raises "
                 "RuntimeError if the scene is not built or has pending updates (call sync() first).")
            .def(
                "triangle_edge_indices",
                [](const Scene& scene, const Int& prim_id, bool global) {
                    const auto edge_ids = scene.triangle_edge_indices(prim_id, global);
                    return nb::make_tuple(edge_ids[0], edge_ids[1], edge_ids[2]);
                },
                "prim_id"_a, "global_"_a = true,
                "Return the three edge ids of each scene-global triangle prim_id.\n"
                "\n"
                "global_ selects scene-global edge ids (True) or per-mesh edge ids. Out-of-range triangles return -1.")
            .def(
                "edge_adjacent_faces",
                [](const Scene& scene, const Int& edge_id, bool global) {
                    const auto face_ids = scene.edge_adjacent_faces(edge_id, global);
                    return nb::make_tuple(face_ids[0], face_ids[1]);
                },
                "edge_id"_a, "global_"_a = true,
                "Return the two faces adjacent to each scene-global edge edge_id.\n"
                "\n"
                "global_ selects scene-global face ids (True) or per-mesh face ids. The second face is -1 on a "
                "boundary edge, and out-of-range edges return -1.")
            .def(
                "intersect",
                [](const Scene& scene, const Ray& ray, rayd::Mask active, RayFlags flags) {
                    return scene.intersect<true>(ray, active, flags);
                },
                nb::arg("ray").noconvert(), "active"_a = true, "flags"_a = RayFlags::All,
                "Find the closest triangle hit of each ray.\n"
                "\n"
                "Args:\n"
                "    ray: Ray or RayAD batch.\n"
                "    active: Per-ray mask; inactive rays return a miss.\n"
                "    flags: RayFlags selecting which hit fields to compute.\n"
                "\n"
                "Returns:\n"
                "    Intersection, or IntersectionAD for a RayAD.\n"
                "\n"
                "Differentiable: with a RayAD, the hit fields carry derivatives (VJP and JVP) with respect to the ray "
                "and the scene geometry, with the hit triangle fixed by the forward pass. Raises RuntimeError if the "
                "scene is not built or has pending updates (call sync() first). An empty ray batch also raises "
                "RuntimeError.")
            .def(
                "intersect",
                [](const Scene& scene, const RayAD& ray, rayd::MaskAD active, RayFlags flags) {
                    return scene.intersect<false>(ray, active, flags);
                },
                nb::arg("ray").noconvert(), "active"_a = true, "flags"_a = RayFlags::All)
            .def(
                "_cuda_first_blocker_selftest",
                [](const Scene& scene, const Vector3f& origin, const Vector3f& direction, const rayd::Float& tmax,
                   const std::vector<int>& ignore) {
                    return scene.cuda_first_blocker_selftest(origin, direction, tmax, ignore);
                },
                nb::arg("origin"), nb::arg("direction"), nb::arg("tmax"), nb::arg("ignore") = std::vector<int>(),
                "P3 CUDA-backend test hook: closest blocker global primitive id per "
                "ray with an optional ignore list. Requires trace_backend='cuda'.")
            .def(
                "trace_reflections",
                [](const Scene& scene, const Ray& ray, int max_bounces, rayd::Mask active,
                   bool symbolic) -> nb::object {
                    require_positive_max_bounces(max_bounces);
                    if (symbolic) {
                        return nb::cast(scene.trace_bounces<true>(ray, max_bounces, active));
                    }
                    return nb::cast(scene.trace_reflections<true>(ray, max_bounces, active));
                },
                nb::arg("ray").noconvert(), "max_bounces"_a, "active"_a = true, "symbolic"_a = true,
                "Trace specular reflection paths for up to max_bounces bounces per ray.\n"
                "\n"
                "Args:\n"
                "    ray: Ray or RayAD batch.\n"
                "    max_bounces: Maximum number of reflections; must be positive.\n"
                "    options: Optional ReflectionTraceOptions. The deduplicate, canonical_prim_table, and "
                "image_source_tolerance keywords are a shorthand for it.\n"
                "    active: Per-ray mask.\n"
                "    symbolic: True returns a ReflectionTrace with one record per bounce; False returns a flat "
                "ReflectionChain.\n"
                "\n"
                "Returns:\n"
                "    ReflectionTrace or ReflectionChain, or their AD variants for a RayAD.\n"
                "\n"
                "Differentiable: with a RayAD, the path geometry carries derivatives (VJP and JVP) with respect to the "
                "ray and the scene geometry, with the reflector sequence fixed by the forward pass. Deduplication "
                "requires symbolic=False. A non-positive max_bounces raises ValueError; raises RuntimeError if the "
                "scene is not built or has pending updates (call sync() first).")
            .def(
                "trace_reflections",
                [](const Scene& scene, const RayAD& ray, int max_bounces, rayd::MaskAD active,
                   bool symbolic) -> nb::object {
                    require_positive_max_bounces(max_bounces);
                    if (symbolic) {
                        return nb::cast(scene.trace_bounces<false>(ray, max_bounces, active));
                    }
                    return nb::cast(scene.trace_reflections<false>(ray, max_bounces, active));
                },
                nb::arg("ray").noconvert(), "max_bounces"_a, "active"_a = true, "symbolic"_a = true)
            .def(
                "trace_reflections",
                [](const Scene& scene, const Ray& ray, int max_bounces, const ReflectionTraceOptions& options,
                   rayd::Mask active, bool symbolic) -> nb::object {
                    require_positive_max_bounces(max_bounces);
                    if (symbolic) {
                        return nb::cast(scene.trace_bounces<true>(ray, max_bounces, options, active));
                    }
                    return nb::cast(scene.trace_reflections<true>(ray, max_bounces, options, active));
                },
                nb::arg("ray").noconvert(), "max_bounces"_a, "options"_a, "active"_a = true, "symbolic"_a = true)
            .def(
                "trace_reflections",
                [](const Scene& scene, const RayAD& ray, int max_bounces, const ReflectionTraceOptions& options,
                   rayd::MaskAD active, bool symbolic) -> nb::object {
                    require_positive_max_bounces(max_bounces);
                    if (symbolic) {
                        return nb::cast(scene.trace_bounces<false>(ray, max_bounces, options, active));
                    }
                    return nb::cast(scene.trace_reflections<false>(ray, max_bounces, options, active));
                },
                nb::arg("ray").noconvert(), "max_bounces"_a, "options"_a, "active"_a = true, "symbolic"_a = true)
            .def(
                "trace_reflections",
                [](const Scene& scene, const Ray& ray, int max_bounces, bool deduplicate,
                   const Int& canonical_prim_table, float image_source_tolerance, rayd::Mask active,
                   bool symbolic) -> nb::object {
                    require_positive_max_bounces(max_bounces);
                    ReflectionTraceOptions options;
                    options.deduplicate = deduplicate;
                    options.canonical_prim_table = canonical_prim_table;
                    options.image_source_tolerance = image_source_tolerance;
                    if (symbolic) {
                        return nb::cast(scene.trace_bounces<true>(ray, max_bounces, options, active));
                    }
                    return nb::cast(scene.trace_reflections<true>(ray, max_bounces, options, active));
                },
                nb::arg("ray").noconvert(), "max_bounces"_a, "deduplicate"_a = false, "canonical_prim_table"_a = Int(),
                "image_source_tolerance"_a = 1e-5f, "active"_a = true, "symbolic"_a = true)
            .def(
                "trace_reflections",
                [](const Scene& scene, const RayAD& ray, int max_bounces, bool deduplicate,
                   const Int& canonical_prim_table, float image_source_tolerance, rayd::MaskAD active,
                   bool symbolic) -> nb::object {
                    require_positive_max_bounces(max_bounces);
                    ReflectionTraceOptions options;
                    options.deduplicate = deduplicate;
                    options.canonical_prim_table = canonical_prim_table;
                    options.image_source_tolerance = image_source_tolerance;
                    if (symbolic) {
                        return nb::cast(scene.trace_bounces<false>(ray, max_bounces, options, active));
                    }
                    return nb::cast(scene.trace_reflections<false>(ray, max_bounces, options, active));
                },
                nb::arg("ray").noconvert(), "max_bounces"_a, "deduplicate"_a = false, "canonical_prim_table"_a = Int(),
                "image_source_tolerance"_a = 1e-5f, "active"_a = true, "symbolic"_a = true)
            .def(
                "trace_refl_epc",
                [](const Scene& scene, const Ray& ray, nb::handle receiver_obj, int max_bounces,
                   const ReflEpcOptions* options, rayd::Mask active) {
                    require_no_derivatives(receiver_obj, "Scene.trace_refl_epc()", "receiver");
                    require_detached_scene_geometry(scene, "Scene.trace_refl_epc()");
                    const Vector3f receiver = nb::cast<Vector3f>(receiver_obj);
                    return scene.trace_refl_epc<true>(ray, receiver, max_bounces,
                                                      options != nullptr ? *options : ReflEpcOptions(), active);
                },
                nb::arg("ray").noconvert(), nb::arg("receiver"), "max_bounces"_a,
                nb::arg("options").none() = nb::none(), "active"_a = true,
                "Trace reflection paths from each ray and correct them into paths that end exactly at the receiver.\n"
                "\n"
                "Args:\n"
                "    ray: Detached Ray batch.\n"
                "    receiver: Receiver positions, one or one per ray.\n"
                "    max_bounces: Maximum number of reflections, from 1 to 8.\n"
                "    options: Optional ReflEpcOptions.\n"
                "    active: Per-ray mask.\n"
                "\n"
                "Returns:\n"
                "    ReflEpc.\n"
                "\n"
                "Not differentiable: a grad-enabled receiver or scene geometry raises RuntimeError. Raises "
                "RuntimeError if the scene is not built or has pending updates (call sync() first).")
            .def(
                "trace_refl_epc_field",
                [](const Scene& scene, nb::handle ray_obj, nb::handle receiver_obj, int max_bounces,
                   const ReflEpcFieldOptions& options, nb::handle active_obj) -> nb::object {
                    // A RayAD with detached field options: promote the options so the AD trace runs.
                    if (!nb::isinstance<RayAD>(ray_obj))
                        throw nb::next_overload();
                    const RayAD ray = nb::cast<RayAD>(ray_obj);
                    const Vector3fAD receiver = nb::cast<Vector3fAD>(receiver_obj);
                    const rayd::MaskAD active = nb::cast<rayd::MaskAD>(active_obj);
                    const ReflEpcFieldOptionsAD options_ad = refl_epc_field_options_ad_from_detached(options);
                    return nb::cast(scene.trace_refl_epc_field<false>(ray, receiver, max_bounces, options_ad, active));
                },
                nb::arg("ray"), nb::arg("receiver"), "max_bounces"_a, "options"_a, "active"_a = true,
                "Trace corrected reflection paths to the receiver and evaluate the complex field there.\n"
                "\n"
                "Args:\n"
                "    ray or source: A Ray, or transmitter positions (one per path).\n"
                "    receiver: Receiver positions, one or one per path.\n"
                "    max_bounces: Maximum number of reflections, from 1 to 8.\n"
                "    options: ReflEpcFieldOptions or ReflEpcFieldOptionsAD; per-slot arrays need n_paths * max_bounces "
                "entries.\n"
                "    active: Per-path mask.\n"
                "\n"
                "Returns:\n"
                "    ReflEpcField, or ReflEpcFieldAD when any input is an AD array.\n"
                "\n"
                "Differentiable through the source overload: with AD arrays, the result carries derivatives (VJP and "
                "JVP) with respect to the source, the receiver, the scene geometry, and the per-slot planes and "
                "materials. The ray overload is forward-only, and a RayAD raises RuntimeError. Raises RuntimeError if "
                "the scene is not built or has pending updates (call sync() first).")
            .def(
                "trace_refl_epc_field",
                [](const Scene& scene, nb::handle source_obj, nb::handle receiver_obj, int max_bounces,
                   const ReflEpcFieldOptions& options, nb::handle active_obj) -> nb::object {
                    // Any AD point input with detached field options selects the AD transmitter-position trace.
                    if (!is_ad_drjit_array(source_obj) && !is_ad_drjit_array(receiver_obj) &&
                        !is_ad_drjit_array(active_obj))
                        throw nb::next_overload();
                    const Vector3fAD source = nb::cast<Vector3fAD>(source_obj);
                    const Vector3fAD receiver = nb::cast<Vector3fAD>(receiver_obj);
                    const rayd::MaskAD active = nb::cast<rayd::MaskAD>(active_obj);
                    const ReflEpcFieldOptionsAD options_ad = refl_epc_field_options_ad_from_detached(options);
                    return nb::cast(
                        scene.trace_refl_epc_field<false>(source, receiver, max_bounces, options_ad, active));
                },
                nb::arg("source"), nb::arg("receiver"), "max_bounces"_a, "options"_a, "active"_a = true)
            .def(
                "trace_refl_epc_field",
                [](const Scene& scene, nb::handle ray_obj, nb::handle receiver_obj, int max_bounces,
                   const ReflEpcFieldOptionsAD& options, nb::handle active_obj) -> nb::object {
                    // A RayAD whose receiver or mask still needs conversion to its AD type.
                    if (!nb::isinstance<RayAD>(ray_obj))
                        throw nb::next_overload();
                    const RayAD ray = nb::cast<RayAD>(ray_obj);
                    const Vector3fAD receiver = nb::cast<Vector3fAD>(receiver_obj);
                    const rayd::MaskAD active = nb::cast<rayd::MaskAD>(active_obj);
                    return nb::cast(scene.trace_refl_epc_field<false>(ray, receiver, max_bounces, options, active));
                },
                nb::arg("ray"), nb::arg("receiver"), "max_bounces"_a, "options"_a, "active"_a = true)
            .def(
                "trace_refl_epc_field",
                [](const Scene& scene, nb::handle source_obj, nb::handle receiver_obj, int max_bounces,
                   const ReflEpcFieldOptionsAD& options, nb::handle active_obj) -> nb::object {
                    // AD transmitter-position trace whose point inputs still need conversion to their AD types.
                    if (!is_ad_drjit_array(source_obj) && !is_ad_drjit_array(receiver_obj) &&
                        !is_ad_drjit_array(active_obj))
                        throw nb::next_overload();
                    const Vector3fAD source = nb::cast<Vector3fAD>(source_obj);
                    const Vector3fAD receiver = nb::cast<Vector3fAD>(receiver_obj);
                    const rayd::MaskAD active = nb::cast<rayd::MaskAD>(active_obj);
                    return nb::cast(scene.trace_refl_epc_field<false>(source, receiver, max_bounces, options, active));
                },
                nb::arg("source"), nb::arg("receiver"), "max_bounces"_a, "options"_a, "active"_a = true)
            .def(
                "trace_refl_epc_field",
                [](const Scene& scene, const RayAD& ray, const Vector3fAD& receiver, int max_bounces,
                   const ReflEpcFieldOptionsAD& options, rayd::MaskAD active) {
                    return scene.trace_refl_epc_field<false>(ray, receiver, max_bounces, options, active);
                },
                nb::arg("ray").noconvert(), nb::arg("receiver").noconvert(), "max_bounces"_a, "options"_a,
                "active"_a = true)
            .def(
                "trace_refl_epc_field",
                [](const Scene& scene, const Vector3fAD& source, const Vector3fAD& receiver, int max_bounces,
                   const ReflEpcFieldOptionsAD& options, rayd::MaskAD active) {
                    return scene.trace_refl_epc_field<false>(source, receiver, max_bounces, options, active);
                },
                nb::arg("source").noconvert(), nb::arg("receiver").noconvert(), "max_bounces"_a, "options"_a,
                "active"_a = true)
            .def(
                "trace_refl_epc_field",
                [](const Scene& scene, const Ray& ray, const Vector3f& receiver, int max_bounces,
                   const ReflEpcFieldOptions& options, rayd::Mask active) {
                    return scene.trace_refl_epc_field<true>(ray, receiver, max_bounces, options, active);
                },
                nb::arg("ray").noconvert(), nb::arg("receiver"), "max_bounces"_a, "options"_a, "active"_a = true)
            .def(
                "trace_refl_epc_field",
                [](const Scene& scene, const Vector3f& source, const Vector3f& receiver, int max_bounces,
                   const ReflEpcFieldOptions& options, rayd::Mask active) {
                    return scene.trace_refl_epc_field<true>(source, receiver, max_bounces, options, active);
                },
                nb::arg("source"), nb::arg("receiver"), "max_bounces"_a, "options"_a, "active"_a = true)
            .def(
                "accum_refl",
                [](const Scene& scene, nb::handle ray_obj, nb::handle tx_position_obj, const AccumGrid& grid,
                   nb::handle material_obj, int max_bounces, const AccumOptions& options, nb::handle active_obj,
                   nb::handle tx_polarization_obj) -> nb::object {
                    if (nb::isinstance<Ray>(ray_obj)) {
                        // A detached Ray selects the forward-only evaluation; differentiation needs a RayAD.
                        require_no_derivatives(tx_position_obj, "Scene.accum_refl() with a detached Ray",
                                               "tx_position");
                        require_no_derivatives(tx_polarization_obj, "Scene.accum_refl() with a detached Ray",
                                               "tx_polarization");
                        const Ray ray = nb::cast<Ray>(ray_obj);
                        const Vector3f tx_position = nb::cast<Vector3f>(tx_position_obj);
                        const Vector3f tx_polarization = nb::cast<Vector3f>(tx_polarization_obj);
                        const Material material = nb::cast<Material>(material_obj);
                        const rayd::Mask active = nb::cast<rayd::Mask>(active_obj);
                        return nb::cast(scene.accumulate_reflections<true>(ray, tx_position, grid, material,
                                                                           max_bounces, options, active,
                                                                           tx_polarization));
                    }
                    if (nb::isinstance<RayAD>(ray_obj)) {
                        const RayAD ray = nb::cast<RayAD>(ray_obj);
                        const Vector3fAD tx_position = nb::cast<Vector3fAD>(tx_position_obj);
                        const Vector3fAD tx_polarization = nb::cast<Vector3fAD>(tx_polarization_obj);
                        const MaterialAD material = nb::cast<MaterialAD>(material_obj);
                        const rayd::MaskAD active = nb::cast<rayd::MaskAD>(active_obj);
                        return nb::cast(scene.accumulate_reflections<false>(ray, tx_position, grid, material,
                                                                            max_bounces, options, active,
                                                                            tx_polarization));
                    }
                    throw nb::next_overload();
                },
                nb::arg("ray"), nb::arg("tx_position"), "grid"_a, "material"_a, "max_bounces"_a,
                "options"_a = AccumOptions(), "active"_a = true, "tx_polarization"_a = nb::make_tuple(1.f, 0.f, 0.f),
                "Trace reflection paths and accumulate the reflected field and power on a grid.\n"
                "\n"
                "Args:\n"
                "    ray: Ray or RayAD batch leaving the transmitter.\n"
                "    tx_position: Transmitter position, one or one per ray.\n"
                "    grid: AccumGrid receiving plane.\n"
                "    material: ReflMaterial (ReflMaterialAD with a RayAD) with one entry per scene-global primitive.\n"
                "    max_bounces: Maximum number of reflections; must be positive.\n"
                "    options: AccumOptions.\n"
                "    active: Per-ray mask.\n"
                "    tx_polarization: Transmitter polarization, one or one per ray.\n"
                "\n"
                "Returns:\n"
                "    AccumResult, or AccumResultAD for a RayAD.\n"
                "\n"
                "Differentiable with a RayAD: VJP and JVP with respect to the scene geometry, the transmitter, and the "
                "material, with the paths fixed by the forward pass. With a detached Ray, a grad-enabled tx_position "
                "or tx_polarization raises RuntimeError. Invalid grid or options values raise RuntimeError, as does a "
                "scene that is not built or has pending updates.")
            .def(
                "accum_dfr_direct",
                [](const Scene& scene, nb::handle states_obj, const DfrGrid& grid, nb::handle material_obj,
                   const DfrOptions& options, nb::handle active_obj) -> nb::object {
                    if (nb::isinstance<DfrStates>(states_obj)) {
                        const DfrStates states = nb::cast<DfrStates>(states_obj);
                        const DfrMaterial material = nb::cast<DfrMaterial>(material_obj);
                        const rayd::Mask active = nb::cast<rayd::Mask>(active_obj);
                        return nb::cast(scene.accum_dfr_direct<true>(states, grid, material, options, active));
                    }
                    if (nb::isinstance<DfrStatesAD>(states_obj)) {
                        const DfrStatesAD states = nb::cast<DfrStatesAD>(states_obj);
                        const DfrMaterialAD material = nb::cast<DfrMaterialAD>(material_obj);
                        const rayd::MaskAD active = nb::cast<rayd::MaskAD>(active_obj);
                        return nb::cast(scene.accum_dfr_direct<false>(states, grid, material, options, active));
                    }
                    throw nb::next_overload();
                },
                "states"_a, "grid"_a, "material"_a, "options"_a = DfrOptions(), "active"_a = true,
                "Accumulate first-order edge diffraction from sampled states on a grid by Monte-Carlo sampling.\n"
                "\n"
                "Args:\n"
                "    states: DfrStates, or DfrStatesAD for the AD overload.\n"
                "    grid: DfrGrid receiving plane.\n"
                "    material: DfrMaterial, or DfrMaterialAD for the AD overload.\n"
                "    options: DfrOptions; max_order must be 1.\n"
                "    active: Per-state mask, one or one per state.\n"
                "\n"
                "Returns:\n"
                "    DfrAccum, or DfrAccumAD for the AD overload.\n"
                "\n"
                "Differentiable with AD states: VJP and JVP with respect to the scene geometry and the material. "
                "Invalid grid or options values raise RuntimeError, as does a scene that is not built or has pending "
                "updates.")
            .def(
                "accum_dfr_coherent_direct",
                [](const Scene& scene, nb::handle states_obj, const DfrGrid& grid, nb::handle material_obj,
                   const DfrCoherentOptions& options, nb::handle active_obj) -> nb::object {
                    if (nb::isinstance<DfrCoherentUtdStates>(states_obj)) {
                        const DfrCoherentUtdStates states = nb::cast<DfrCoherentUtdStates>(states_obj);
                        const rayd::Mask active = nb::cast<rayd::Mask>(active_obj);
                        return nb::cast(scene.accum_dfr_coherent_direct<true>(states, grid, options, active));
                    }
                    if (nb::isinstance<DfrCoherentUtdStatesAD>(states_obj)) {
                        const DfrCoherentUtdStatesAD states = nb::cast<DfrCoherentUtdStatesAD>(states_obj);
                        const rayd::MaskAD active = nb::cast<rayd::MaskAD>(active_obj);
                        return nb::cast(scene.accum_dfr_coherent_direct<false>(states, grid, options, active));
                    }
                    if (nb::isinstance<DfrStates>(states_obj)) {
                        const DfrStates states = nb::cast<DfrStates>(states_obj);
                        const DfrMaterial material = nb::cast<DfrMaterial>(material_obj);
                        const rayd::Mask active = nb::cast<rayd::Mask>(active_obj);
                        return nb::cast(scene.accum_dfr_coherent_direct<true>(states, grid, material, options, active));
                    }
                    if (nb::isinstance<DfrStatesAD>(states_obj)) {
                        const DfrStatesAD states = nb::cast<DfrStatesAD>(states_obj);
                        const DfrMaterialAD material = nb::cast<DfrMaterialAD>(material_obj);
                        const rayd::MaskAD active = nb::cast<rayd::MaskAD>(active_obj);
                        return nb::cast(
                            scene.accum_dfr_coherent_direct<false>(states, grid, material, options, active));
                    }
                    throw nb::next_overload();
                },
                "states"_a, "grid"_a, "material"_a, "options"_a = DfrCoherentOptions(), "active"_a = true,
                "Accumulate the first-order diffracted field on a grid deterministically, one lane per (state, cell) "
                "pair.\n"
                "\n"
                "Args:\n"
                "    states: DfrCoherentUtdStates (material is then ignored and may be None) or DfrStates.\n"
                "    grid: DfrGrid receiving plane.\n"
                "    material: DfrMaterial for DfrStates input.\n"
                "    options: DfrCoherentOptions; max_order must be 1 and receiver_model MATCHED_ISO.\n"
                "    active: Per-state mask, one or one per state.\n"
                "\n"
                "Returns:\n"
                "    DfrCoherentAccum.\n"
                "\n"
                "Forward-only: the AD overloads raise RuntimeError. Invalid grid or options values raise RuntimeError, "
                "as does a scene that is not built or has pending updates.")
            .def(
                "build_dfr_coherent_tx_states",
                [](const Scene& scene, nb::handle edges_obj, nb::handle tx_position_obj, nb::handle material_obj,
                   const DfrCoherentOptions& options, nb::handle active_obj) -> nb::object {
                    if (nb::isinstance<DfrCoherentEdge>(edges_obj)) {
                        require_no_derivatives(tx_position_obj, "Scene.build_dfr_coherent_tx_states()", "tx_position");
                        const DfrCoherentEdge edges = nb::cast<DfrCoherentEdge>(edges_obj);
                        const Vector3f tx_position = nb::cast<Vector3f>(tx_position_obj);
                        const DfrMaterial material = nb::cast<DfrMaterial>(material_obj);
                        const rayd::Mask active = nb::cast<rayd::Mask>(active_obj);
                        return nb::cast(
                            scene.build_dfr_coherent_tx_states<true>(edges, tx_position, material, options, active));
                    }
                    if (nb::isinstance<DfrCoherentEdgeAD>(edges_obj)) {
                        const DfrCoherentEdgeAD edges = nb::cast<DfrCoherentEdgeAD>(edges_obj);
                        const Vector3fAD tx_position = nb::cast<Vector3fAD>(tx_position_obj);
                        const DfrMaterialAD material = nb::cast<DfrMaterialAD>(material_obj);
                        const rayd::MaskAD active = nb::cast<rayd::MaskAD>(active_obj);
                        return nb::cast(
                            scene.build_dfr_coherent_tx_states<false>(edges, tx_position, material, options, active));
                    }
                    throw nb::next_overload();
                },
                "edges"_a, "tx_position"_a, "material"_a, "options"_a = DfrCoherentOptions(), "active"_a = true,
                "Build first-order coherent UTD states that illuminate the given edges from a transmitter.\n"
                "\n"
                "Args:\n"
                "    edges: DfrCoherentEdge selection.\n"
                "    tx_position: Transmitter position, one or one per edge.\n"
                "    material: DfrMaterial.\n"
                "    options: DfrCoherentOptions; wavelength and k must be positive.\n"
                "    active: Per-edge mask, one or one per edge.\n"
                "\n"
                "Returns:\n"
                "    DfrCoherentUtdStates for Scene.accum_dfr_coherent_direct().\n"
                "\n"
                "Forward-only: the AD overload and a grad-enabled tx_position raise RuntimeError. Raises RuntimeError "
                "if the scene is not built or has pending updates (call sync() first).")
            .def(
                "build_dfr_coherent_higher_candidates",
                [](const Scene& scene, nb::handle prev_states_obj, nb::handle edges_obj,
                   nb::handle global_to_local_edge_index_obj, const DfrCoherentOptions& options,
                   nb::handle active_obj) -> nb::object {
                    if (nb::isinstance<DfrCoherentUtdStates>(prev_states_obj) &&
                        nb::isinstance<DfrCoherentEdge>(edges_obj)) {
                        const DfrCoherentUtdStates prev_states = nb::cast<DfrCoherentUtdStates>(prev_states_obj);
                        const DfrCoherentEdge edges = nb::cast<DfrCoherentEdge>(edges_obj);
                        const Int global_to_local_edge_index = nb::cast<Int>(global_to_local_edge_index_obj);
                        const rayd::Mask active = nb::cast<rayd::Mask>(active_obj);
                        return nb::cast(scene.build_dfr_coherent_higher_candidates<true>(prev_states, edges,
                                                                                         global_to_local_edge_index,
                                                                                         options, active));
                    }
                    if (nb::isinstance<DfrCoherentUtdStatesAD>(prev_states_obj) &&
                        nb::isinstance<DfrCoherentEdgeAD>(edges_obj)) {
                        const DfrCoherentUtdStatesAD prev_states = nb::cast<DfrCoherentUtdStatesAD>(prev_states_obj);
                        const DfrCoherentEdgeAD edges = nb::cast<DfrCoherentEdgeAD>(edges_obj);
                        const IntAD global_to_local_edge_index = nb::cast<IntAD>(global_to_local_edge_index_obj);
                        const rayd::MaskAD active = nb::cast<rayd::MaskAD>(active_obj);
                        return nb::cast(scene.build_dfr_coherent_higher_candidates<false>(prev_states, edges,
                                                                                          global_to_local_edge_index,
                                                                                          options, active));
                    }
                    throw nb::next_overload();
                },
                "prev_states"_a, "edges"_a, "global_to_local_edge_index"_a, "options"_a = DfrCoherentOptions(),
                "active"_a = true,
                "Find candidate next edges for higher-order coherent diffraction from a set of UTD states.\n"
                "\n"
                "Probe rays around each previous state's edge find nearby edges; the probe radius is the "
                "source-to-edge distance times options.higher_probe_radius_scale, clamped to [higher_probe_radius_min, "
                "higher_probe_radius_max].\n"
                "\n"
                "Args:\n"
                "    prev_states: DfrCoherentUtdStates to expand.\n"
                "    edges: DfrCoherentEdge candidates.\n"
                "    global_to_local_edge_index: Map from scene-global edge id to an index into edges, or -1.\n"
                "    options: DfrCoherentOptions.\n"
                "    active: Per-state mask, one or one per previous state.\n"
                "\n"
                "Returns:\n"
                "    DfrCoherentCandidatePairs.\n"
                "\n"
                "Forward-only: the AD overload raises RuntimeError. Raises RuntimeError if the scene is not built or "
                "has pending updates (call sync() first).")
            .def(
                "accum_dfr",
                [](const Scene& scene, nb::handle initial_states_obj, nb::handle recursive_states_obj,
                   const DfrGrid& grid, nb::handle material_obj, const DfrOptions& options,
                   nb::handle active_obj) -> nb::object {
                    if (nb::isinstance<DfrStates>(initial_states_obj)) {
                        const DfrStates initial_states = nb::cast<DfrStates>(initial_states_obj);
                        const DfrStates recursive_states = nb::cast<DfrStates>(recursive_states_obj);
                        const DfrMaterial material = nb::cast<DfrMaterial>(material_obj);
                        const rayd::Mask active = nb::cast<rayd::Mask>(active_obj);
                        return nb::cast(
                            scene.accum_dfr<true>(initial_states, recursive_states, grid, material, options, active));
                    }
                    if (nb::isinstance<DfrStatesAD>(initial_states_obj)) {
                        const DfrStatesAD initial_states = nb::cast<DfrStatesAD>(initial_states_obj);
                        const DfrStatesAD recursive_states = nb::cast<DfrStatesAD>(recursive_states_obj);
                        const DfrMaterialAD material = nb::cast<DfrMaterialAD>(material_obj);
                        const rayd::MaskAD active = nb::cast<rayd::MaskAD>(active_obj);
                        return nb::cast(
                            scene.accum_dfr<false>(initial_states, recursive_states, grid, material, options, active));
                    }
                    throw nb::next_overload();
                },
                "initial_states"_a, "recursive_states"_a, "grid"_a, "material"_a, "options"_a = DfrOptions(),
                "active"_a = true,
                "Accumulate second- or third-order edge-diffraction chains on a grid by Monte-Carlo sampling.\n"
                "\n"
                "Args:\n"
                "    initial_states: DfrStates of the first diffraction event.\n"
                "    recursive_states: DfrStates drawn for each further event.\n"
                "    grid: DfrGrid receiving plane.\n"
                "    material: DfrMaterial.\n"
                "    options: DfrOptions; max_order must be 2 or 3.\n"
                "    active: Per-state mask, one or one per initial state.\n"
                "\n"
                "Returns:\n"
                "    DfrAccum, or DfrAccumAD for AD states.\n"
                "\n"
                "Differentiable with AD states: VJP and JVP with respect to the scene geometry and the material. "
                "Invalid grid or options values raise RuntimeError, as does a scene that is not built or has pending "
                "updates.")
            .def(
                "trace_dfr_paths",
                [](const Scene& scene, const Vector3f& tx_positions, const Vector3f& rx_positions,
                   const DfrStates& states, const DfrMaterial& material, const DfrPathOptions& options,
                   rayd::Mask active) {
                    return scene.trace_dfr_paths<true>(tx_positions, rx_positions, states, material, options, active);
                },
                "tx_positions"_a, "rx_positions"_a, "states"_a, "material"_a, "options"_a = DfrPathOptions(),
                "active"_a = true,
                "Export first-order diffraction paths for every (transmitter, receiver, state) triple.\n"
                "\n"
                "Args:\n"
                "    tx_positions: Transmitter positions.\n"
                "    rx_positions: Receiver positions.\n"
                "    states: DfrStates, or DfrStatesAD for the AD overload.\n"
                "    material: DfrMaterial, or DfrMaterialAD for the AD overload.\n"
                "    options: DfrPathOptions; max_paths must be positive.\n"
                "    active: Per-state mask, one or one per state.\n"
                "\n"
                "Returns:\n"
                "    DfrPaths with capacity n_tx * n_rx * min(n_states, max_paths), or DfrPathsAD.\n"
                "\n"
                "Differentiable with AD inputs: VJP and JVP with respect to the continuous inputs. Raises RuntimeError "
                "if the scene is not built or has pending updates (call sync() first).")
            .def(
                "trace_dfr_paths",
                [](const Scene& scene, const Vector3fAD& tx_positions, const Vector3fAD& rx_positions,
                   const DfrStatesAD& states, const DfrMaterialAD& material, const DfrPathOptions& options,
                   rayd::MaskAD active) {
                    return scene.trace_dfr_paths<false>(tx_positions, rx_positions, states, material, options, active);
                },
                "tx_positions"_a, "rx_positions"_a, "states"_a, "material"_a, "options"_a = DfrPathOptions(),
                "active"_a = true)
            .def(
                "shadow_test",
                [](const Scene& scene, const Ray& ray, rayd::Mask active) {
                    return scene.shadow_test<true>(ray, active);
                },
                nb::arg("ray").noconvert(), "active"_a = true,
                "Return a mask that is True where the ray hits any triangle within tmax.\n"
                "\n"
                "The result is discrete and has no derivative. Raises RuntimeError if the scene is not built or has "
                "pending updates (call sync() first).")
            .def(
                "shadow_test",
                [](const Scene& scene, const RayAD& ray, rayd::MaskAD active) {
                    return scene.shadow_test<false>(ray, active);
                },
                nb::arg("ray").noconvert(), "active"_a = true)
            .def(
                "visible",
                [](const Scene& scene, nb::handle start_obj, nb::handle end_obj, nb::handle active_obj,
                   const Int& ignore_prim_ids) -> nb::object {
                    const std::string module_name = nb::cast<std::string>(start_obj.type().attr("__module__"));
                    const std::string type_name = nb::cast<std::string>(start_obj.type().attr("__name__"));

                    if (module_name == "drjit.cuda.ad" && type_name == "Array3f") {
                        Vector3fAD start = nb::cast<Vector3fAD>(start_obj);
                        Vector3fAD end = nb::cast<Vector3fAD>(end_obj);
                        rayd::MaskAD active = nb::cast<rayd::MaskAD>(active_obj);
                        return nb::cast(scene.visible<false>(start, end, ignore_prim_ids, active));
                    }

                    if (module_name == "drjit.cuda" && type_name == "Array3f") {
                        const Vector3f start = cast_detached_query_input<Vector3f>(start_obj);
                        const Vector3f end = cast_detached_query_input<Vector3f>(end_obj);
                        rayd::Mask active = nb::cast<rayd::Mask>(active_obj);
                        return nb::cast(scene.visible<true>(start, end, ignore_prim_ids, active));
                    }
                    throw nb::next_overload();
                },
                nb::arg("start"), nb::arg("end"), "active"_a = true, nb::kw_only(), "ignore_prim_ids"_a = Int(),
                "Test whether the segment from start to end is unoccluded.\n"
                "\n"
                "Args:\n"
                "    start: Segment start points.\n"
                "    end: Segment end points, one per start point.\n"
                "    active: Per-segment mask.\n"
                "    ignore_prim_ids: Optional scene-global primitive ids to treat as transparent, K per segment laid "
                "out segment-major.\n"
                "\n"
                "Returns:\n"
                "    SegmentVisibility, or SegmentVisibilityAD for AD points.\n"
                "\n"
                "The result is discrete and has no derivative. Segments with a non-finite endpoint are not visible. "
                "Raises RuntimeError if the scene is not built or has pending updates (call sync() first).")
            .def(
                "visible_pair",
                [](const Scene& scene, nb::handle start_obj, nb::handle end_a_obj, nb::handle end_b_obj,
                   const Int& ignore_prim_ids, nb::handle active_obj) -> nb::object {
                    const std::string module_name = nb::cast<std::string>(start_obj.type().attr("__module__"));
                    const std::string type_name = nb::cast<std::string>(start_obj.type().attr("__name__"));

                    if (module_name == "drjit.cuda.ad" && type_name == "Array3f") {
                        Vector3fAD start = nb::cast<Vector3fAD>(start_obj);
                        Vector3fAD end_a = nb::cast<Vector3fAD>(end_a_obj);
                        Vector3fAD end_b = nb::cast<Vector3fAD>(end_b_obj);
                        rayd::MaskAD active = nb::cast<rayd::MaskAD>(active_obj);
                        return nb::cast(scene.visible_pair<false>(start, end_a, end_b, ignore_prim_ids, active));
                    }

                    if (module_name == "drjit.cuda" && type_name == "Array3f") {
                        const Vector3f start = cast_detached_query_input<Vector3f>(start_obj);
                        const Vector3f end_a = cast_detached_query_input<Vector3f>(end_a_obj);
                        const Vector3f end_b = cast_detached_query_input<Vector3f>(end_b_obj);
                        rayd::Mask active = nb::cast<rayd::Mask>(active_obj);
                        return nb::cast(scene.visible_pair<true>(start, end_a, end_b, ignore_prim_ids, active));
                    }
                    throw nb::next_overload();
                },
                nb::arg("start"), nb::arg("end_a"), nb::arg("end_b"), "ignore_prim_ids"_a = Int(), "active"_a = true,
                "Test the visibility of two endpoints from each start point in one pass.\n"
                "\n"
                "Returns SegmentPairVisibility with visible_a for end_a and visible_b for end_b. ignore_prim_ids works "
                "as in visible(). The result is discrete and has no derivative. Raises RuntimeError if the scene is "
                "not built or has pending updates (call sync() first).")
            .def(
                "visible_edge",
                [](const Scene& scene, nb::handle source_obj, nb::handle edge_position_obj,
                   nb::handle edge_direction_obj, nb::handle edge_t_min_obj, nb::handle edge_t_max_obj,
                   const std::vector<float>& sample_fractions, nb::handle active_obj) -> nb::object {
                    const std::string module_name = nb::cast<std::string>(source_obj.type().attr("__module__"));
                    const std::string type_name = nb::cast<std::string>(source_obj.type().attr("__name__"));

                    if (module_name == "drjit.cuda.ad" && type_name == "Array3f") {
                        Vector3fAD source = nb::cast<Vector3fAD>(source_obj);
                        Vector3fAD edge_position = nb::cast<Vector3fAD>(edge_position_obj);
                        Vector3fAD edge_direction = nb::cast<Vector3fAD>(edge_direction_obj);
                        FloatAD edge_t_min = nb::cast<FloatAD>(edge_t_min_obj);
                        FloatAD edge_t_max = nb::cast<FloatAD>(edge_t_max_obj);
                        rayd::MaskAD active = nb::cast<rayd::MaskAD>(active_obj);
                        return nb::cast(scene.visible_edge<false>(source, edge_position, edge_direction, edge_t_min,
                                                                  edge_t_max, sample_fractions, active));
                    }

                    if (module_name == "drjit.cuda" && type_name == "Array3f") {
                        const Vector3f source = cast_detached_query_input<Vector3f>(source_obj);
                        const Vector3f edge_position = cast_detached_query_input<Vector3f>(edge_position_obj);
                        const Vector3f edge_direction = cast_detached_query_input<Vector3f>(edge_direction_obj);
                        const Float edge_t_min = cast_detached_query_input<Float>(edge_t_min_obj);
                        const Float edge_t_max = cast_detached_query_input<Float>(edge_t_max_obj);
                        rayd::Mask active = nb::cast<rayd::Mask>(active_obj);
                        return nb::cast(scene.visible_edge<true>(source, edge_position, edge_direction, edge_t_min,
                                                                 edge_t_max, sample_fractions, active));
                    }
                    throw nb::next_overload();
                },
                nb::arg("source"), nb::arg("edge_position"), nb::arg("edge_direction"), "edge_t_min"_a, "edge_t_max"_a,
                "sample_fractions"_a = std::vector<float>{0.f, 0.25f, 0.5f, 0.75f, 1.f}, "active"_a = true,
                "Test whether each source sees any sampled point of an edge segment.\n"
                "\n"
                "Args:\n"
                "    source: Source points.\n"
                "    edge_position: Edge origins.\n"
                "    edge_direction: Edge directions.\n"
                "    edge_t_min, edge_t_max: Edge extent along edge_direction from edge_position.\n"
                "    sample_fractions: 1 to 16 fractions of [edge_t_min, edge_t_max] to sample.\n"
                "    active: Per-edge mask.\n"
                "\n"
                "Returns:\n"
                "    AxialEdgeVisibility, or its AD variant for AD points.\n"
                "\n"
                "The result is discrete and has no derivative. Raises RuntimeError if the scene is not built or has "
                "pending updates (call sync() first).")
            .def(
                "visible_chain",
                [](const Scene& scene, nb::handle points_obj, const Int& chain_length,
                   const Int& ignore_prim_per_segment, nb::handle active_obj) -> nb::object {
                    const std::string module_name = nb::cast<std::string>(points_obj.type().attr("__module__"));
                    const std::string type_name = nb::cast<std::string>(points_obj.type().attr("__name__"));

                    if (module_name == "drjit.cuda.ad" && type_name == "Array3f") {
                        Vector3fAD points = nb::cast<Vector3fAD>(points_obj);
                        rayd::MaskAD active = nb::cast<rayd::MaskAD>(active_obj);
                        return nb::cast(
                            scene.visible_chain<false>(points, chain_length, ignore_prim_per_segment, active));
                    }

                    if (module_name == "drjit.cuda" && type_name == "Array3f") {
                        const Vector3f points = cast_detached_query_input<Vector3f>(points_obj);
                        rayd::Mask active = nb::cast<rayd::Mask>(active_obj);
                        return nb::cast(
                            scene.visible_chain<true>(points, chain_length, ignore_prim_per_segment, active));
                    }
                    throw nb::next_overload();
                },
                nb::arg("points"), "chain_length"_a, "ignore_prim_per_segment"_a = Int(), "active"_a = true,
                "Test every segment of each polyline chain for occlusion.\n"
                "\n"
                "Args:\n"
                "    points: Chain vertices, chain-major, with the same number of points (at least 2) per chain.\n"
                "    chain_length: Number of used points in each chain; its width is the chain count.\n"
                "    ignore_prim_per_segment: Optional scene-global primitive ids to ignore, K per segment.\n"
                "    active: Per-chain mask.\n"
                "\n"
                "Returns:\n"
                "    SegmentChainVisibility with the first blocked segment and primitive of each chain.\n"
                "\n"
                "The result is discrete and has no derivative. Raises RuntimeError if the scene is not built or has "
                "pending updates (call sync() first).")
            .def(
                "nearest_edge",
                [](const Scene& scene, nb::handle query_obj, nb::handle active_obj) -> nb::object {
                    const std::string module_name = nb::cast<std::string>(query_obj.type().attr("__module__"));
                    const std::string type_name = nb::cast<std::string>(query_obj.type().attr("__name__"));

                    if (module_name == "drjit.cuda.ad" && type_name == "Array3f") {
                        Vector3fAD point = nb::cast<Vector3fAD>(query_obj);
                        rayd::MaskAD active = nb::cast<rayd::MaskAD>(active_obj);
                        return nb::cast(scene.nearest_edge<false>(point, active));
                    }

                    if (module_name == "drjit.cuda" && type_name == "Array3f") {
                        Vector3f point_detached = nb::cast<Vector3f>(query_obj);
                        rayd::Mask active = nb::cast<rayd::Mask>(active_obj);
                        return nb::cast(scene.nearest_edge<true>(point_detached, active));
                    }
                    throw nb::next_overload();
                },
                nb::arg("query"), "active"_a = true,
                "Find the nearest scene edge to each query point or ray.\n"
                "\n"
                "query is an Array3f point batch, which returns NearestPointEdge, or a Ray, which returns "
                "NearestRayEdge. A ray searches the segment [0, tmax] when tmax is finite and the half-line otherwise; "
                "rows with a non-finite origin, a zero or non-finite direction, or a non-positive tmax return the miss "
                "values. Edges disabled by set_edge_mask() are skipped.\n"
                "\n"
                "Differentiable: with AD inputs, distance and closest points carry derivatives (VJP and JVP) with "
                "respect to the query and the scene geometry, with the winning edge fixed by the forward pass. Raises "
                "RuntimeError if the scene is not built or has pending updates (call sync() first).")
            .def(
                "nearest_edge",
                [](const Scene& scene, const Ray& ray, rayd::Mask active) {
                    return scene.nearest_edge<true>(ray, active);
                },
                nb::arg("query").noconvert(), "active"_a = true)
            .def(
                "nearest_edge",
                [](const Scene& scene, const RayAD& ray, rayd::MaskAD active) {
                    return scene.nearest_edge<false>(ray, active);
                },
                nb::arg("query").noconvert(), "active"_a = true)
            .def(
                "nearest_edges",
                [](const Scene& scene, nb::handle point_obj, int k, nb::handle active_obj) -> nb::object {
                    const std::string module_name = nb::cast<std::string>(point_obj.type().attr("__module__"));
                    const std::string type_name = nb::cast<std::string>(point_obj.type().attr("__name__"));

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
                nb::arg("point"), "k"_a, "active"_a = true,
                "Find the k nearest scene edges to each query point, nearest first, for 1 <= k <= 16.\n"
                "\n"
                "Returns NearestEdgesTopK. Ties are broken toward the lower global edge id.\n"
                "\n"
                "Differentiable: with an AD point batch, distances and closest points carry derivatives (VJP and JVP) "
                "with respect to the points and the scene geometry, with the selected edges fixed by the forward pass. "
                "A k outside [1, 16] raises RuntimeError; raises RuntimeError if the scene is not built or has pending "
                "updates (call sync() first).")
            .def_prop_ro("num_meshes", &Scene::num_meshes, "Number of meshes and instances.")
            .def_prop_ro("version", &Scene::version, "Counter that increases whenever the scene geometry changes.")
            .def_prop_ro("edge_version", &Scene::edge_version, "Counter that increases whenever the edge set changes.")
            .def("__repr__", &Scene::to_string);
    });

    bind_section("mixed scene", [&]() {
        nb::class_<MixedScene>(m, "MixedScene",
                               "Scene that combines triangle meshes, SDF grids, and surfel clouds.\n"
                               "\n"
                               "Supports intersect(), visible(), trace_reflections(), and transmittance(); diffraction "
                               "stays on Scene. Ids number meshes first, then SDF grids, then surfel clouds.")
            .def(
                "__init__",
                [](MixedScene* scene, const std::string& trace_backend, const std::string& edge_bvh_backend) {
                    new (scene) MixedScene(edge_bvh_backend, trace_backend);
                },
                nb::kw_only(), "trace_backend"_a = "auto", "edge_bvh_backend"_a = "auto",
                "Create an empty mixed scene. trace_backend and edge_bvh_backend configure its triangle Scene and take "
                "the same values as in Scene(); an unknown value raises ValueError.")
            .def("add_mesh", &MixedScene::add_mesh, "mesh"_a, "dynamic"_a = false,
                 "Add a copy of mesh and return its mesh id.")
            .def("add_sdf", &MixedScene::add_sdf, "grid"_a, "options"_a = SdfTraceOptions(),
                 "Add an SdfGrid traced with options and return its index among the SDF grids.")
            .def("add_surfel", &MixedScene::add_surfel, "surfel"_a, "options"_a = SurfelTraceOptions(),
                 "Add a surfel cloud traced with options and return its index among the surfel clouds.")
            .def("build", &MixedScene::build,
                 "Build every mesh, SDF, and surfel acceleration structure. Call it before any query.")
            .def("is_ready", &MixedScene::is_ready, "Return True once build() has run and nothing was added since.")
            .def_prop_ro("num_meshes", &MixedScene::num_meshes, "Number of meshes.")
            .def_prop_ro("num_sdfs", &MixedScene::num_sdfs, "Number of SDF grids.")
            .def_prop_ro("num_surfel_scenes", &MixedScene::num_surfel_scenes, "Number of surfel clouds.")
            .def(
                "intersect",
                [](const MixedScene& scene, const Ray& ray, rayd::Mask active, RayFlags flags) {
                    return scene.intersect<true>(ray, active, flags);
                },
                nb::arg("ray").noconvert(), "active"_a = true, "flags"_a = RayFlags::All,
                "Find the closest mesh, SDF, or surfel hit of each ray.\n"
                "\n"
                "Returns an Intersection (IntersectionAD for a RayAD). shape_id counts meshes, then SDF grids, then "
                "surfel clouds; global_prim_id counts mesh faces, then one id per SDF grid, then surfels.\n"
                "\n"
                "Differentiable: with a RayAD, the selected hit's continuous fields carry derivatives (VJP and JVP) "
                "with respect to the ray and the selected geometry, with the winner fixed by the forward pass. Raises "
                "RuntimeError if the scene is not built.")
            .def(
                "intersect",
                [](const MixedScene& scene, const RayAD& ray, rayd::MaskAD active, RayFlags flags) {
                    return scene.intersect<false>(ray, active, flags);
                },
                nb::arg("ray").noconvert(), "active"_a = true, "flags"_a = RayFlags::All)
            .def(
                "visible",
                [](const MixedScene& scene, const Vector3f& start, const Vector3f& end, rayd::Mask active) {
                    return scene.visible<true>(start, end, active);
                },
                "start"_a, "end"_a, "active"_a = true,
                "Test whether the segment from start to end is unoccluded by any mesh, SDF, or surfel.\n"
                "\n"
                "Returns SegmentVisibility. The result is discrete and has no derivative. Raises RuntimeError if the "
                "scene is not built.")
            .def(
                "visible",
                [](const MixedScene& scene, const Vector3fAD& start, const Vector3fAD& end, rayd::MaskAD active) {
                    return scene.visible<false>(start, end, active);
                },
                "start"_a, "end"_a, "active"_a = true)
            .def(
                "trace_reflections",
                [](const MixedScene& scene, const Ray& ray, int max_bounces, rayd::Mask active) {
                    return scene.trace_reflections<true>(ray, max_bounces, active);
                },
                nb::arg("ray").noconvert(), "max_bounces"_a, "active"_a = true,
                "Trace specular reflections off meshes, SDFs, and surfels for up to max_bounces bounces.\n"
                "\n"
                "Returns a ReflectionChain. max_bounces must be non-negative. Raises RuntimeError if the scene is not "
                "built.")
            .def(
                "trace_reflections",
                [](const MixedScene& scene, const RayAD& ray, int max_bounces, rayd::MaskAD active) {
                    return scene.trace_reflections<false>(ray, max_bounces, active);
                },
                nb::arg("ray").noconvert(), "max_bounces"_a, "active"_a = true)
            .def(
                "transmittance",
                [](const MixedScene& scene, const Ray& ray, rayd::Mask active) {
                    return scene.transmittance<true>(ray, active);
                },
                nb::arg("ray").noconvert(), "active"_a = true,
                "Return the transmittance along each ray.\n"
                "\n"
                "A mesh hit makes the ray opaque (0), surfel alpha attenuates it multiplicatively, and SDF grids are "
                "ignored. Inactive rays return 1.\n"
                "\n"
                "Differentiable: with a RayAD, the result carries derivatives (VJP and JVP) with respect to the ray "
                "and the surfel fields. Raises RuntimeError if the scene is not built.")
            .def(
                "transmittance",
                [](const MixedScene& scene, const RayAD& ray, rayd::MaskAD active) {
                    return scene.transmittance<false>(ray, active);
                },
                nb::arg("ray").noconvert(), "active"_a = true);
    });
}
