#include <rayd/rayd.h>

#include <stdexcept>

#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <drjit/python.h>

#include <rayd/intersection.h>
#include <rayd/ray.h>
#include <rayd/transform.h>
#include <rayd/edge.h>
#include <rayd/mesh.h>
#include <rayd/optix.h>
#include <rayd/camera.h>
#include <rayd/scene/scene.h>

namespace nb = nanobind;
using namespace nb::literals;
using namespace rayd;

namespace {

int checked_cuda_device_count() {
    const int count = jit_cuda_device_count();
    if (count <= 0)
        throw std::runtime_error("No Dr.Jit-compatible CUDA devices are available.");
    return count;
}

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

template <typename T>
nb::object drjit_python_type() {
    static nb::object type = []() {
        drjit::ArrayBinding b;
        return drjit::bind_array<T>(b);
    }();
    return type;
}

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

    jit_set_flag(JitFlag::SymbolicLoops, false);

    m.doc() = "Differentiable geometry queries built on Dr.Jit and OptiX.";
    m.attr("__name__") = "rayd";
    m.def("device_count",
          &checked_cuda_device_count,
          "Return the number of Dr.Jit-compatible CUDA devices visible to RayD.");
    m.def("current_device",
          []() { return checked_cuda_device_count() > 0 ? jit_cuda_device() : 0; },
          "Return the current thread's active Dr.Jit CUDA device index.");
    m.def("set_device",
          &set_rayd_device,
          "device"_a,
          "initialize_optix"_a = true,
          "Set the current thread's active CUDA device for RayD.\n\n"
          "Call this before constructing RayD meshes, scenes, cameras, or "
          "torch/Dr.Jit arrays that you intend to use with them. When "
          "initialize_optix=True, RayD also initializes the OptiX device "
          "context for the selected device.");
    bind_section("core types", [&]() {
        nb::class_<RayDetached>(m, "RayDetached")
            .def(nb::init<>())
            .def(nb::init<const Vector3fDetached &, const Vector3fDetached &>())
            .def("reversed", &RayDetached::reversed)
            .def_rw("o", &RayDetached::o)
            .def_rw("d", &RayDetached::d)
            .def_rw("tmax", &RayDetached::tmax);

        nb::class_<Ray>(m, "Ray")
            .def(nb::init<>())
            .def(nb::init<const Vector3f &, const Vector3f &>())
            .def("reversed", &Ray::reversed)
            .def_rw("o", &Ray::o)
            .def_rw("d", &Ray::d)
            .def_rw("tmax", &Ray::tmax);

        nb::class_<PrimaryEdgeSample>(m, "PrimaryEdgeSample")
            .def_ro("x_dot_n", &PrimaryEdgeSample::x_dot_n)
            .def_ro("idx", &PrimaryEdgeSample::idx)
            .def_ro("ray_n", &PrimaryEdgeSample::ray_n)
            .def_ro("ray_p", &PrimaryEdgeSample::ray_p)
            .def_ro("pdf", &PrimaryEdgeSample::pdf);

        nb::class_<SecondaryEdgeInfo>(m, "SecondaryEdgeInfo")
            .def(nb::init<>())
            .def("size", &SecondaryEdgeInfo::size)
            .def_ro("start", &SecondaryEdgeInfo::start)
            .def_ro("edge", &SecondaryEdgeInfo::edge)
            .def_ro("normal0", &SecondaryEdgeInfo::normal0)
            .def_ro("normal1", &SecondaryEdgeInfo::normal1)
            .def_ro("opposite", &SecondaryEdgeInfo::opposite)
            .def_ro("is_boundary", &SecondaryEdgeInfo::is_boundary);

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
            .def_ro("face0_local", &SceneEdgeTopology::face0_local)
            .def_ro("face1_local", &SceneEdgeTopology::face1_local)
            .def_ro("face0_global", &SceneEdgeTopology::face0_global)
            .def_ro("face1_global", &SceneEdgeTopology::face1_global)
            .def_ro("opposite_vertex0", &SceneEdgeTopology::opposite_vertex0)
            .def_ro("opposite_vertex1", &SceneEdgeTopology::opposite_vertex1);

        nb::enum_<RayFlags>(m, "RayFlags", nb::is_arithmetic())
            .value("None", RayFlags::None)
            .value("Geometric", RayFlags::Geometric)
            .value("ShadingN", RayFlags::ShadingN)
            .value("UV", RayFlags::UV)
            .value("All", RayFlags::All);

        nb::class_<IntersectionDetached>(m, "IntersectionDetached")
            .def("is_valid", &IntersectionDetached::is_valid)
            .def_ro("t", &IntersectionDetached::t)
            .def_ro("p", &IntersectionDetached::p)
            .def_ro("n", &IntersectionDetached::n)
            .def_ro("geo_n", &IntersectionDetached::geo_n)
            .def_ro("uv", &IntersectionDetached::uv)
            .def_ro("barycentric", &IntersectionDetached::barycentric)
            .def_ro("shape_id", &IntersectionDetached::shape_id)
            .def_ro("prim_id", &IntersectionDetached::prim_id);

        nb::class_<Intersection>(m, "Intersection")
            .def("is_valid", &Intersection::is_valid)
            .def_ro("t", &Intersection::t)
            .def_ro("p", &Intersection::p)
            .def_ro("n", &Intersection::n)
            .def_ro("geo_n", &Intersection::geo_n)
            .def_ro("uv", &Intersection::uv)
            .def_ro("barycentric", &Intersection::barycentric)
            .def_ro("shape_id", &Intersection::shape_id)
            .def_ro("prim_id", &Intersection::prim_id);

        nb::class_<NearestPointEdgeDetached>(m, "NearestPointEdgeDetached")
            .def("is_valid", &NearestPointEdgeDetached::is_valid)
            .def_ro("distance", &NearestPointEdgeDetached::distance)
            .def_ro("point", &NearestPointEdgeDetached::point)
            .def_ro("edge_t", &NearestPointEdgeDetached::edge_t)
            .def_ro("edge_point", &NearestPointEdgeDetached::edge_point)
            .def_ro("shape_id", &NearestPointEdgeDetached::shape_id)
            .def_ro("edge_id", &NearestPointEdgeDetached::edge_id)
            .def_ro("global_edge_id", &NearestPointEdgeDetached::global_edge_id)
            .def_ro("is_boundary", &NearestPointEdgeDetached::is_boundary);

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

        nb::class_<NearestRayEdgeDetached>(m, "NearestRayEdgeDetached")
            .def("is_valid", &NearestRayEdgeDetached::is_valid)
            .def_ro("distance", &NearestRayEdgeDetached::distance)
            .def_ro("ray_t", &NearestRayEdgeDetached::ray_t)
            .def_ro("point", &NearestRayEdgeDetached::point)
            .def_ro("edge_t", &NearestRayEdgeDetached::edge_t)
            .def_ro("edge_point", &NearestRayEdgeDetached::edge_point)
            .def_ro("shape_id", &NearestRayEdgeDetached::shape_id)
            .def_ro("edge_id", &NearestRayEdgeDetached::edge_id)
            .def_ro("global_edge_id", &NearestRayEdgeDetached::global_edge_id)
            .def_ro("is_boundary", &NearestRayEdgeDetached::is_boundary);

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
                    const Vector3fDetached &v,
                    const Vector3iDetached &f,
                    const Vector2fDetached &uv,
                    const Vector3iDetached &f_uv,
                    bool verbose) {
                     new (mesh) Mesh(v, f, uv, f_uv, verbose);
                 },
                 "v"_a,
                 "f"_a,
                 "uv"_a = Vector2fDetached(),
                 "f_uv"_a = Vector3iDetached(),
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
                return mesh.secondary_edge_info() != nullptr ? *mesh.secondary_edge_info() : SecondaryEdgeInfo();
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

    bind_section("camera", [&]() {
        nb::class_<Camera>(m, "Camera")
            .def(nb::init<float, float, float>(), "fov_x"_a = 45.f, "near_clip"_a = 1e-4f, "far_clip"_a = 1e4f)
            .def(nb::init<float, float, float, float, float, float>(),
                 "fx"_a, "fy"_a, "cx"_a, "cy"_a, "near_clip"_a = 1e-4f, "far_clip"_a = 1e4f)
            .def_static("perspective",
                 [](float fov_x, float near_clip, float far_clip) {
                     return Camera(fov_x, near_clip, far_clip);
                 },
                 "fov_x"_a = 45.f, "near_clip"_a = 1e-4f, "far_clip"_a = 1e4f)
            .def_static("from_intrinsics",
                 [](float fx, float fy, float cx, float cy, float near_clip, float far_clip) {
                     return Camera(fx, fy, cx, cy, near_clip, far_clip);
                 },
                 "fx"_a, "fy"_a, "cx"_a, "cy"_a, "near_clip"_a = 1e-4f, "far_clip"_a = 1e4f)
            .def("build", &Camera::build, "cache"_a = true)
            .def("render", &Camera::render, "scene"_a, "background"_a = 0.f)
            .def("render_grad", &Camera::render_grad, "scene"_a, "spp"_a = 4, "background"_a = 0.f)
            .def("prepare_edges", &Camera::prepare_primary_edges, "scene"_a)
            .def("sample_ray",
                 nb::overload_cast<const Vector2fDetached &>(&Camera::sample_primary_ray, nb::const_),
                 "sample"_a)
            .def("sample_ray",
                 nb::overload_cast<const Vector2f &>(&Camera::sample_primary_ray, nb::const_),
                 "sample"_a)
            .def("sample_edge", &Camera::sample_primary_edge, "sample1"_a)
            .def("set_transform", &Camera::set_transform, "mat"_a, "set_left"_a = true)
            .def("append_transform", &Camera::append_transform, "mat"_a, "append_left"_a = true)
            .def_prop_rw("width", &Camera::width, &Camera::set_width)
            .def_prop_rw("height", &Camera::height, &Camera::set_height)
            .def_prop_rw("to_world", &Camera::to_world, &Camera::set_to_world)
            .def_prop_rw("to_world_left", &Camera::to_world_left, &Camera::set_to_world_left)
            .def_prop_rw("to_world_right", &Camera::to_world_right, &Camera::set_to_world_right)
            .def_prop_ro("camera_to_sample", &Camera::camera_to_sample)
            .def_prop_ro("sample_to_camera", &Camera::sample_to_camera)
            .def_prop_ro("world_to_sample", &Camera::world_to_sample)
            .def_prop_ro("sample_to_world", &Camera::sample_to_world)
            .def_prop_ro("slang_handle", [](Camera &c) -> uint64_t {
                return static_cast<uint64_t>(reinterpret_cast<std::uintptr_t>(&c));
            })
            .def("__repr__", &Camera::to_string);
    });

    bind_section("scene", [&]() {
        nb::class_<Scene>(m, "Scene")
            .def(nb::init<>())
            .def("add_mesh", &Scene::add_mesh, "mesh"_a, "dynamic"_a = false)
            .def("build", &Scene::build)
            .def("update_mesh_vertices", &Scene::update_mesh_vertices, "mesh_id"_a, "positions"_a)
            .def("set_mesh_transform", &Scene::set_mesh_transform, "mesh_id"_a, "mat"_a, "set_left"_a = true)
            .def("append_mesh_transform", &Scene::append_mesh_transform, "mesh_id"_a, "mat"_a, "append_left"_a = true)
            .def("set_edge_mask",
                 nb::overload_cast<const rayd::MaskDetached &>(&Scene::set_edge_mask),
                 nb::arg("mask"))
            .def("set_edge_mask",
                 nb::overload_cast<const rayd::Mask &>(&Scene::set_edge_mask),
                 nb::arg("mask"))
            .def("sync", &Scene::sync)
            .def("is_ready", &Scene::is_ready)
            .def("has_pending_updates", &Scene::has_pending_updates)
            .def_prop_ro("last_sync_profile", &Scene::last_sync_profile)
            .def("edge_info", &Scene::edge_info)
            .def("edge_bvh_stats", &Scene::edge_bvh_stats)
            .def("edge_topology", &Scene::edge_topology)
            .def("edge_mask", &Scene::edge_mask)
            .def("mesh_face_offsets", &Scene::mesh_face_offsets)
            .def("mesh_edge_offsets", &Scene::mesh_edge_offsets)
            .def("triangle_edge_indices",
                 [](const Scene &scene, const IntDetached &prim_id, bool global) {
                     const auto edge_ids = scene.triangle_edge_indices(prim_id, global);
                     return nb::make_tuple(edge_ids[0], edge_ids[1], edge_ids[2]);
                 },
                 "prim_id"_a,
                 "global_"_a = true)
            .def("edge_adjacent_faces",
                 [](const Scene &scene, const IntDetached &edge_id, bool global) {
                     const auto face_ids = scene.edge_adjacent_faces(edge_id, global);
                     return nb::make_tuple(face_ids[0], face_ids[1]);
                 },
                 "edge_id"_a,
                 "global_"_a = true)
            .def("intersect",
                 [](const Scene &scene, const RayDetached &ray, rayd::MaskDetached active, RayFlags flags) {
                     return scene.intersect<true>(ray, active, flags);
                 },
                 nb::arg("ray").noconvert(), "active"_a = true, "flags"_a = RayFlags::All)
            .def("intersect",
                 [](const Scene &scene, const Ray &ray, rayd::Mask active, RayFlags flags) {
                     return scene.intersect<false>(ray, active, flags);
                 },
                 nb::arg("ray").noconvert(), "active"_a = true, "flags"_a = RayFlags::All)
            .def("shadow_test",
                 [](const Scene &scene, const RayDetached &ray, rayd::MaskDetached active) {
                     return scene.shadow_test<true>(ray, active);
                 },
                 nb::arg("ray").noconvert(), "active"_a = true)
            .def("shadow_test",
                 [](const Scene &scene, const Ray &ray, rayd::Mask active) {
                     return scene.shadow_test<false>(ray, active);
                 },
                 nb::arg("ray").noconvert(), "active"_a = true)
            .def("nearest_edge",
                 [](const Scene &scene, nb::handle point_obj, nb::handle active_obj) -> nb::object {
                     const std::string module_name =
                         nb::cast<std::string>(point_obj.type().attr("__module__"));
                     const std::string type_name =
                         nb::cast<std::string>(point_obj.type().attr("__name__"));

                     if (module_name == "drjit.cuda.ad" && type_name == "Array3f") {
                         Vector3f point = nb::cast<Vector3f>(point_obj);
                         rayd::Mask active = nb::cast<rayd::Mask>(active_obj);
                         return nb::cast(scene.nearest_edge<false>(point, active));
                     }

                     if (module_name == "drjit.cuda" && type_name == "Array3f") {
                         Vector3fDetached point_detached = nb::cast<Vector3fDetached>(point_obj);
                         rayd::MaskDetached active = nb::cast<rayd::MaskDetached>(active_obj);
                         return nb::cast(scene.nearest_edge<true>(point_detached, active));
                     }
                     throw nb::next_overload();
                 },
                 nb::arg("point"), "active"_a = true)
            .def("nearest_edge",
                 [](const Scene &scene, const RayDetached &ray, rayd::MaskDetached active) {
                     return scene.nearest_edge<true>(ray, active);
                 },
                 nb::arg("ray").noconvert(), "active"_a = true)
            .def("nearest_edge",
                 [](const Scene &scene, const Ray &ray, rayd::Mask active) {
                     return scene.nearest_edge<false>(ray, active);
                 },
                 nb::arg("ray").noconvert(), "active"_a = true)
            .def_prop_ro("num_meshes", &Scene::num_meshes)
            .def_prop_ro("version", &Scene::version)
            .def_prop_ro("edge_version", &Scene::edge_version)
            .def_prop_ro("slang_handle", [](Scene &s) -> uint64_t {
                return static_cast<uint64_t>(reinterpret_cast<std::uintptr_t>(&s));
            })
            .def("__repr__", &Scene::to_string);
    });
}
