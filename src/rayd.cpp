#include <rayd/rayd.h>

#include <nanobind/stl/string.h>
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
            .def_ro("is_boundary", &NearestPointEdgeDetached::is_boundary);

        nb::class_<NearestPointEdge>(m, "NearestPointEdge")
            .def("is_valid", &NearestPointEdge::is_valid)
            .def_ro("distance", &NearestPointEdge::distance)
            .def_ro("point", &NearestPointEdge::point)
            .def_ro("edge_t", &NearestPointEdge::edge_t)
            .def_ro("edge_point", &NearestPointEdge::edge_point)
            .def_ro("shape_id", &NearestPointEdge::shape_id)
            .def_ro("edge_id", &NearestPointEdge::edge_id)
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
            .def_ro("is_boundary", &NearestRayEdge::is_boundary);

        nb::class_<SceneCommitProfile>(m, "SceneCommitProfile")
            .def_ro("mesh_update_ms", &SceneCommitProfile::mesh_update_ms)
            .def_ro("triangle_scatter_ms", &SceneCommitProfile::triangle_scatter_ms)
            .def_ro("triangle_eval_ms", &SceneCommitProfile::triangle_eval_ms)
            .def_ro("optix_commit_ms", &SceneCommitProfile::optix_commit_ms)
            .def_ro("total_ms", &SceneCommitProfile::total_ms)
            .def_ro("optix_gas_update_ms", &SceneCommitProfile::optix_gas_update_ms)
            .def_ro("optix_ias_update_ms", &SceneCommitProfile::optix_ias_update_ms)
            .def_ro("updated_meshes", &SceneCommitProfile::updated_meshes)
            .def_ro("updated_vertex_meshes", &SceneCommitProfile::updated_vertex_meshes)
            .def_ro("updated_transform_meshes", &SceneCommitProfile::updated_transform_meshes);
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
            .def("configure", &Mesh::configure)
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
        nb::class_<PerspectiveCamera>(m, "Camera")
            .def(nb::init<float, float, float>(), "fov_x"_a = 45.f, "near_clip"_a = 1e-4f, "far_clip"_a = 1e4f)
            .def(nb::init<float, float, float, float, float, float>(),
                 "fx"_a, "fy"_a, "cx"_a, "cy"_a, "near_clip"_a = 1e-4f, "far_clip"_a = 1e4f)
            .def_static("perspective",
                 [](float fov_x, float near_clip, float far_clip) {
                     return PerspectiveCamera(fov_x, near_clip, far_clip);
                 },
                 "fov_x"_a = 45.f, "near_clip"_a = 1e-4f, "far_clip"_a = 1e4f)
            .def_static("from_intrinsics",
                 [](float fx, float fy, float cx, float cy, float near_clip, float far_clip) {
                     return PerspectiveCamera(fx, fy, cx, cy, near_clip, far_clip);
                 },
                 "fx"_a, "fy"_a, "cx"_a, "cy"_a, "near_clip"_a = 1e-4f, "far_clip"_a = 1e4f)
            .def("configure", &PerspectiveCamera::configure, "cache"_a = true)
            .def("render", &PerspectiveCamera::render, "scene"_a, "background"_a = 0.f)
            .def("render_grad", &PerspectiveCamera::render_grad, "scene"_a, "spp"_a = 4, "background"_a = 0.f)
            .def("prepare_edges", &PerspectiveCamera::prepare_primary_edges, "scene"_a)
            .def("sample_ray",
                 nb::overload_cast<const Vector2fDetached &>(&PerspectiveCamera::sample_primary_ray, nb::const_),
                 "sample"_a)
            .def("sample_ray",
                 nb::overload_cast<const Vector2f &>(&PerspectiveCamera::sample_primary_ray, nb::const_),
                 "sample"_a)
            .def("sample_edge", &PerspectiveCamera::sample_primary_edge, "sample1"_a)
            .def("set_transform", &PerspectiveCamera::set_transform, "mat"_a, "set_left"_a = true)
            .def("append_transform", &PerspectiveCamera::append_transform, "mat"_a, "append_left"_a = true)
            .def_prop_rw("width", &PerspectiveCamera::width, &PerspectiveCamera::set_width)
            .def_prop_rw("height", &PerspectiveCamera::height, &PerspectiveCamera::set_height)
            .def_prop_rw("to_world", &PerspectiveCamera::to_world, &PerspectiveCamera::set_to_world)
            .def_prop_rw("to_world_left", &PerspectiveCamera::to_world_left, &PerspectiveCamera::set_to_world_left)
            .def_prop_rw("to_world_right", &PerspectiveCamera::to_world_right, &PerspectiveCamera::set_to_world_right)
            .def_prop_ro("camera_to_sample", &PerspectiveCamera::camera_to_sample)
            .def_prop_ro("sample_to_camera", &PerspectiveCamera::sample_to_camera)
            .def_prop_ro("world_to_sample", &PerspectiveCamera::world_to_sample)
            .def_prop_ro("sample_to_world", &PerspectiveCamera::sample_to_world)
            .def_prop_ro("slang_handle", [](PerspectiveCamera &c) -> uint64_t {
                return static_cast<uint64_t>(reinterpret_cast<std::uintptr_t>(&c));
            })
            .def("__repr__", &PerspectiveCamera::to_string);
    });

    bind_section("scene", [&]() {
        nb::class_<Scene>(m, "Scene")
            .def(nb::init<>())
            .def("add_mesh", &Scene::add_mesh, "mesh"_a, "dynamic"_a = false)
            .def("configure", &Scene::configure)
            .def("update_mesh_vertices", &Scene::update_mesh_vertices, "mesh_id"_a, "positions"_a)
            .def("set_mesh_transform", &Scene::set_mesh_transform, "mesh_id"_a, "mat"_a, "set_left"_a = true)
            .def("append_mesh_transform", &Scene::append_mesh_transform, "mesh_id"_a, "mat"_a, "append_left"_a = true)
            .def("commit_updates", &Scene::commit_updates)
            .def("is_ready", &Scene::is_ready)
            .def("has_pending_updates", &Scene::has_pending_updates)
            .def_prop_ro("last_commit_profile", &Scene::last_commit_profile)
            .def("intersect",
                 [](const Scene &scene, const RayDetached &ray, rayd::MaskDetached active) {
                     return scene.intersect<true>(ray, active);
                 },
                 nb::arg("ray").noconvert(), "active"_a = true)
            .def("intersect",
                 [](const Scene &scene, const Ray &ray, rayd::Mask active) {
                     return scene.intersect<false>(ray, active);
                 },
                 nb::arg("ray").noconvert(), "active"_a = true)
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
                 [](const Scene &scene, const Vector3f &point, rayd::Mask active) {
                     return scene.nearest_edge<false>(point, active);
                 },
                 nb::arg("point").noconvert(), "active"_a = true)
            .def("nearest_edge",
                 [](const Scene &scene, const Vector3fDetached &point, rayd::MaskDetached active) {
                     return scene.nearest_edge<true>(point, active);
                 },
                 nb::arg("point").noconvert(), "active"_a = true)
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
            .def_prop_ro("slang_handle", [](Scene &s) -> uint64_t {
                return static_cast<uint64_t>(reinterpret_cast<std::uintptr_t>(&s));
            })
            .def("__repr__", &Scene::to_string);
    });
}
