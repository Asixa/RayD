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

struct IntersectionAD {
    bool valid = false;
    float t = std::numeric_limits<float>::infinity();
    Float3 p, n, geo_n, barycentric;
    Float2 uv;
    int shape_id = -1, prim_id = -1;
    Float3 dt_do;
    Float3 dt_dd;
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

// --- Constructors and field accessors callable from Slang __intrinsic_asm ---
// These exist so that slangc-generated C++ can construct / read POD fields
// without hitting the field-name mangling that __target_intrinsic imposes.

inline Float2 make_float2(float x, float y) { return Float2(x, y); }
inline Float3 make_float3(float x, float y, float z) { return Float3(x, y, z); }
inline Ray    make_ray(Float3 o, Float3 d, float tmax = std::numeric_limits<float>::infinity()) { return Ray(o, d, tmax); }

inline SceneHandle  make_scene_handle_raw(uint64_t v) { return SceneHandle(v); }
inline CameraHandle make_camera_handle_raw(uint64_t v) { return CameraHandle(v); }

// Intersection field accessors
inline bool   its_valid(const Intersection &h) { return h.valid; }
inline float  its_t(const Intersection &h) { return h.t; }
inline Float3 its_p(const Intersection &h) { return h.p; }
inline Float3 its_n(const Intersection &h) { return h.n; }
inline Float3 its_geo_n(const Intersection &h) { return h.geo_n; }
inline Float2 its_uv(const Intersection &h) { return h.uv; }
inline Float3 its_barycentric(const Intersection &h) { return h.barycentric; }
inline int    its_shape_id(const Intersection &h) { return h.shape_id; }
inline int    its_prim_id(const Intersection &h) { return h.prim_id; }

// IntersectionAD field accessors
inline bool   its_ad_valid(const IntersectionAD &h) { return h.valid; }
inline float  its_ad_t(const IntersectionAD &h) { return h.t; }
inline Float3 its_ad_p(const IntersectionAD &h) { return h.p; }
inline Float3 its_ad_n(const IntersectionAD &h) { return h.n; }
inline Float3 its_ad_geo_n(const IntersectionAD &h) { return h.geo_n; }
inline Float2 its_ad_uv(const IntersectionAD &h) { return h.uv; }
inline Float3 its_ad_barycentric(const IntersectionAD &h) { return h.barycentric; }
inline int    its_ad_shape_id(const IntersectionAD &h) { return h.shape_id; }
inline int    its_ad_prim_id(const IntersectionAD &h) { return h.prim_id; }
inline Float3 its_ad_dt_do(const IntersectionAD &h) { return h.dt_do; }
inline Float3 its_ad_dt_dd(const IntersectionAD &h) { return h.dt_dd; }

// Float2/3 field accessors
inline float f2_x(const Float2 &v) { return v.x; }
inline float f2_y(const Float2 &v) { return v.y; }
inline float f3_x(const Float3 &v) { return v.x; }
inline float f3_y(const Float3 &v) { return v.y; }
inline float f3_z(const Float3 &v) { return v.z; }

// Ray field accessors
inline Float3 ray_o(const Ray &r) { return r.o; }
inline Float3 ray_d(const Ray &r) { return r.d; }
inline float  ray_tmax(const Ray &r) { return r.tmax; }

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

// AD-enabled helpers: create 1-lane gradient-tracked arrays.
inline rayd::Float scalar_ad_float(float value) {
    rayd::Float arr = drjit::full<rayd::Float>(value, 1);
    drjit::enable_grad(arr);
    return arr;
}
inline rayd::Vector3f to_cuda_ad(const Float3 &value) {
    return rayd::Vector3f(scalar_ad_float(value.x),
                          scalar_ad_float(value.y),
                          scalar_ad_float(value.z));
}

} // namespace detail

inline SceneHandle make_scene_handle(rayd::Scene &scene) noexcept {
    return SceneHandle(detail::encode_handle(&scene));
}

inline CameraHandle make_camera_handle(rayd::Camera &camera) noexcept {
    return CameraHandle(detail::encode_handle(&camera));
}

// Scene query functions — implemented in src/slang_interop.cpp, exported from rayd_core.
bool scene_is_ready(SceneHandle handle);
bool scene_has_pending_updates(SceneHandle handle);
void scene_configure(SceneHandle handle);
void scene_commit_updates(SceneHandle handle);
Intersection scene_intersect(SceneHandle handle, const Ray &ray, bool active = true);
IntersectionAD scene_intersect_ad(SceneHandle handle, const Ray &ray, bool active = true);
bool scene_shadow_test(SceneHandle handle, const Ray &ray, bool active = true);
NearestPointEdge scene_nearest_edge_point(SceneHandle handle, const Float3 &point, bool active = true);
NearestRayEdge scene_nearest_edge_ray(SceneHandle handle, const Ray &ray, bool active = true);

// Camera query functions — implemented in src/slang_interop.cpp.
bool camera_is_ready(CameraHandle handle);
bool camera_primary_edges_ready(CameraHandle handle);
void camera_set_resolution(CameraHandle handle, int width, int height);
void camera_configure(CameraHandle handle, bool cache = true);
void camera_prepare_edges(CameraHandle camera_handle, SceneHandle scene_handle);
Ray camera_sample_ray(CameraHandle handle, const Float2 &sample);
PrimaryEdgeSample camera_sample_primary_edge(CameraHandle handle, float sample);

} // namespace rayd::slang
