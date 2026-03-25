#pragma once

#include <cstdint>
#include <limits>
#include <type_traits>

#include <rayd/camera.h>
#include <rayd/scene/scene.h>

namespace rayd::slang {

struct Float2 {
    float x = 0.f;
    float y = 0.f;

    constexpr Float2() = default;
    constexpr Float2(float x_, float y_) : x(x_), y(y_) {}
};

struct Float3 {
    float x = 0.f;
    float y = 0.f;
    float z = 0.f;

    constexpr Float3() = default;
    constexpr Float3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
};

struct Ray {
    Float3 o;
    Float3 d = Float3(0.f, 0.f, 1.f);
    float tmax = std::numeric_limits<float>::infinity();

    constexpr Ray() = default;
    constexpr Ray(const Float3 &origin, const Float3 &direction,
                  float tmax_ = std::numeric_limits<float>::infinity())
        : o(origin), d(direction), tmax(tmax_) {}
};

struct Intersection {
    bool valid = false;
    float t = std::numeric_limits<float>::infinity();
    Float3 p;
    Float3 n;
    Float3 geo_n;
    Float2 uv;
    Float3 barycentric;
    int shape_id = -1;
    int prim_id = -1;
};

struct NearestPointEdge {
    bool valid = false;
    float distance = std::numeric_limits<float>::infinity();
    Float3 point;
    float edge_t = 0.f;
    Float3 edge_point;
    int shape_id = -1;
    int edge_id = -1;
    bool is_boundary = false;
};

struct NearestRayEdge {
    bool valid = false;
    float distance = std::numeric_limits<float>::infinity();
    float ray_t = 0.f;
    Float3 point;
    float edge_t = 0.f;
    Float3 edge_point;
    int shape_id = -1;
    int edge_id = -1;
    bool is_boundary = false;
};

struct PrimaryEdgeSample {
    bool valid = false;
    float x_dot_n = 0.f;
    int idx = -1;
    Ray ray_n;
    Ray ray_p;
    float pdf = 0.f;
};

struct SceneHandle {
    uint64_t value = 0;

    constexpr SceneHandle() = default;
    constexpr explicit SceneHandle(uint64_t value_) : value(value_) {}
    constexpr explicit operator bool() const { return value != 0; }
};

struct CameraHandle {
    uint64_t value = 0;

    constexpr CameraHandle() = default;
    constexpr explicit CameraHandle(uint64_t value_) : value(value_) {}
    constexpr explicit operator bool() const { return value != 0; }
};

namespace detail {

template <typename T>
inline uint64_t encode_handle(T *value) noexcept {
    return static_cast<uint64_t>(reinterpret_cast<std::uintptr_t>(value));
}

template <typename T>
inline T *decode_handle(uint64_t value) noexcept {
    return reinterpret_cast<T *>(static_cast<std::uintptr_t>(value));
}

template <typename T>
inline T &handle_ref(uint64_t value, const char *message) {
    T *ptr = decode_handle<T>(value);
    rayd::require(ptr != nullptr, message);
    return *ptr;
}

template <typename T, typename Value,
          std::enable_if_t<std::is_convertible_v<Value, T>, int> = 0>
inline T lane0(const Value &value) {
    return static_cast<T>(value);
}

template <typename T, typename ArrayT,
          std::enable_if_t<!std::is_convertible_v<ArrayT, T>, int> = 0>
inline T lane0(const ArrayT &value) {
    T result{};
    drjit::store(&result, value);
    return result;
}

template <typename DrVector>
inline Float2 to_float2(const DrVector &value) {
    drjit::eval(value);
    drjit::sync_thread();
    return Float2(lane0<float>(value.x()), lane0<float>(value.y()));
}

template <typename DrVector>
inline Float3 to_float3(const DrVector &value) {
    drjit::eval(value);
    drjit::sync_thread();
    return Float3(lane0<float>(value.x()),
                  lane0<float>(value.y()),
                  lane0<float>(value.z()));
}

inline FloatDetached scalar_float(float value) {
    return drjit::full<FloatDetached>(value, 1);
}

inline IntDetached scalar_int(int value) {
    return drjit::full<IntDetached>(value, 1);
}

inline MaskDetached scalar_mask(bool value) {
    return drjit::full<MaskDetached>(value, 1);
}

inline Vector2fDetached to_cuda(const Float2 &value) {
    return Vector2fDetached(scalar_float(value.x), scalar_float(value.y));
}

inline Vector3fDetached to_cuda(const Float3 &value) {
    return Vector3fDetached(scalar_float(value.x),
                            scalar_float(value.y),
                            scalar_float(value.z));
}

inline RayDetached to_cuda(const Ray &value) {
    return RayDetached(to_cuda(value.o), to_cuda(value.d), scalar_float(value.tmax));
}

inline Ray to_scalar(const RayDetached &value) {
    drjit::eval(value.o, value.d, value.tmax);
    drjit::sync_thread();
    return Ray(to_float3(value.o), to_float3(value.d), lane0<float>(value.tmax));
}

} // namespace detail

inline SceneHandle make_scene_handle(rayd::Scene &scene) noexcept {
    return SceneHandle(detail::encode_handle(&scene));
}

inline CameraHandle make_camera_handle(rayd::PerspectiveCamera &camera) noexcept {
    return CameraHandle(detail::encode_handle(&camera));
}

inline bool scene_is_ready(SceneHandle handle) {
    return detail::handle_ref<rayd::Scene>(handle.value,
                                           "rayd::slang::scene_is_ready(): null scene handle.")
        .is_ready();
}

inline bool scene_has_pending_updates(SceneHandle handle) {
    return detail::handle_ref<rayd::Scene>(
               handle.value,
               "rayd::slang::scene_has_pending_updates(): null scene handle.")
        .has_pending_updates();
}

inline void scene_configure(SceneHandle handle) {
    detail::handle_ref<rayd::Scene>(handle.value,
                                    "rayd::slang::scene_configure(): null scene handle.")
        .configure();
}

inline void scene_commit_updates(SceneHandle handle) {
    detail::handle_ref<rayd::Scene>(
        handle.value,
        "rayd::slang::scene_commit_updates(): null scene handle.")
        .commit_updates();
}

inline Intersection scene_intersect(SceneHandle handle,
                                    const Ray &ray,
                                    bool active = true) {
    const rayd::Scene &scene =
        detail::handle_ref<rayd::Scene>(handle.value,
                                        "rayd::slang::scene_intersect(): null scene handle.");

    const IntersectionDetached hit =
        scene.ray_intersect<true>(detail::to_cuda(ray), detail::scalar_mask(active));
    drjit::eval(hit.t,
                hit.p,
                hit.n,
                hit.geo_n,
                hit.uv,
                hit.barycentric,
                hit.shape_id,
                hit.prim_id);
    drjit::sync_thread();

    Intersection result;
    result.shape_id = detail::lane0<int>(hit.shape_id);
    result.prim_id = detail::lane0<int>(hit.prim_id);
    result.valid = result.prim_id >= 0;
    result.t = detail::lane0<float>(hit.t);
    result.p = detail::to_float3(hit.p);
    result.n = detail::to_float3(hit.n);
    result.geo_n = detail::to_float3(hit.geo_n);
    result.uv = detail::to_float2(hit.uv);
    result.barycentric = detail::to_float3(hit.barycentric);
    return result;
}

inline bool scene_shadow_test(SceneHandle handle,
                              const Ray &ray,
                              bool active = true) {
    const rayd::Scene &scene =
        detail::handle_ref<rayd::Scene>(handle.value,
                                        "rayd::slang::scene_shadow_test(): null scene handle.");
    const MaskDetached shadow =
        scene.shadow_test<true>(detail::to_cuda(ray), detail::scalar_mask(active));
    drjit::eval(shadow);
    drjit::sync_thread();
    return detail::lane0<bool>(shadow);
}

inline NearestPointEdge scene_closest_edge_point(SceneHandle handle,
                                                 const Float3 &point,
                                                 bool active = true) {
    const rayd::Scene &scene = detail::handle_ref<rayd::Scene>(
        handle.value,
        "rayd::slang::scene_closest_edge_point(): null scene handle.");
    const NearestPointEdgeDetached hit =
        scene.closest_edge<true>(detail::to_cuda(point), detail::scalar_mask(active));
    drjit::eval(hit.distance,
                hit.point,
                hit.edge_t,
                hit.edge_point,
                hit.shape_id,
                hit.edge_id,
                hit.is_boundary);
    drjit::sync_thread();

    NearestPointEdge result;
    result.shape_id = detail::lane0<int>(hit.shape_id);
    result.edge_id = detail::lane0<int>(hit.edge_id);
    result.valid = result.edge_id >= 0;
    result.distance = detail::lane0<float>(hit.distance);
    result.point = detail::to_float3(hit.point);
    result.edge_t = detail::lane0<float>(hit.edge_t);
    result.edge_point = detail::to_float3(hit.edge_point);
    result.is_boundary = detail::lane0<bool>(hit.is_boundary);
    return result;
}

inline NearestRayEdge scene_closest_edge_ray(SceneHandle handle,
                                             const Ray &ray,
                                             bool active = true) {
    const rayd::Scene &scene = detail::handle_ref<rayd::Scene>(
        handle.value,
        "rayd::slang::scene_closest_edge_ray(): null scene handle.");
    const NearestRayEdgeDetached hit =
        scene.closest_edge<true>(detail::to_cuda(ray), detail::scalar_mask(active));
    drjit::eval(hit.distance,
                hit.ray_t,
                hit.point,
                hit.edge_t,
                hit.edge_point,
                hit.shape_id,
                hit.edge_id,
                hit.is_boundary);
    drjit::sync_thread();

    NearestRayEdge result;
    result.shape_id = detail::lane0<int>(hit.shape_id);
    result.edge_id = detail::lane0<int>(hit.edge_id);
    result.valid = result.edge_id >= 0;
    result.distance = detail::lane0<float>(hit.distance);
    result.ray_t = detail::lane0<float>(hit.ray_t);
    result.point = detail::to_float3(hit.point);
    result.edge_t = detail::lane0<float>(hit.edge_t);
    result.edge_point = detail::to_float3(hit.edge_point);
    result.is_boundary = detail::lane0<bool>(hit.is_boundary);
    return result;
}

inline bool camera_is_ready(CameraHandle handle) {
    return detail::handle_ref<rayd::PerspectiveCamera>(
               handle.value,
               "rayd::slang::camera_is_ready(): null camera handle.")
        .is_ready();
}

inline bool camera_primary_edges_ready(CameraHandle handle) {
    return detail::handle_ref<rayd::PerspectiveCamera>(
               handle.value,
               "rayd::slang::camera_primary_edges_ready(): null camera handle.")
        .primary_edges_ready();
}

inline void camera_set_resolution(CameraHandle handle, int width, int height) {
    rayd::PerspectiveCamera &camera =
        detail::handle_ref<rayd::PerspectiveCamera>(
            handle.value,
            "rayd::slang::camera_set_resolution(): null camera handle.");
    camera.set_width(width);
    camera.set_height(height);
}

inline void camera_configure(CameraHandle handle, bool cache = true) {
    detail::handle_ref<rayd::PerspectiveCamera>(
        handle.value,
        "rayd::slang::camera_configure(): null camera handle.")
        .configure(cache);
}

inline void camera_prepare_edges(CameraHandle camera_handle,
                                 SceneHandle scene_handle) {
    rayd::PerspectiveCamera &camera =
        detail::handle_ref<rayd::PerspectiveCamera>(
            camera_handle.value,
            "rayd::slang::camera_prepare_edges(): null camera handle.");
    const rayd::Scene &scene =
        detail::handle_ref<rayd::Scene>(
            scene_handle.value,
            "rayd::slang::camera_prepare_edges(): null scene handle.");
    camera.prepare_primary_edges(scene);
}

inline Ray camera_sample_ray(CameraHandle handle, const Float2 &sample) {
    const rayd::PerspectiveCamera &camera =
        detail::handle_ref<rayd::PerspectiveCamera>(
            handle.value,
            "rayd::slang::camera_sample_ray(): null camera handle.");
    return detail::to_scalar(camera.sample_primary_ray(detail::to_cuda(sample)));
}

inline PrimaryEdgeSample camera_sample_primary_edge(CameraHandle handle,
                                                    float sample) {
    const rayd::PerspectiveCamera &camera =
        detail::handle_ref<rayd::PerspectiveCamera>(
            handle.value,
            "rayd::slang::camera_sample_primary_edge(): null camera handle.");
    const rayd::PrimaryEdgeSample sample_result =
        camera.sample_primary_edge(detail::scalar_float(sample));
    drjit::eval(sample_result.x_dot_n,
                sample_result.idx,
                sample_result.ray_n.o,
                sample_result.ray_n.d,
                sample_result.ray_n.tmax,
                sample_result.ray_p.o,
                sample_result.ray_p.d,
                sample_result.ray_p.tmax,
                sample_result.pdf);
    drjit::sync_thread();

    PrimaryEdgeSample result;
    result.idx = detail::lane0<int>(sample_result.idx);
    result.valid = result.idx >= 0;
    result.x_dot_n = detail::lane0<float>(sample_result.x_dot_n);
    result.ray_n = detail::to_scalar(sample_result.ray_n);
    result.ray_p = detail::to_scalar(sample_result.ray_p);
    result.pdf = detail::lane0<float>(sample_result.pdf);
    return result;
}

} // namespace rayd::slang
