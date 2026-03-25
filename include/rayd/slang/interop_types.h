#pragma once
// Lightweight POD types and helpers for Slang-generated C++ code.
// This header is self-contained — it does NOT include rayd internals or drjit.
// Use this when compiling slangc-generated C++ via torch.utils.cpp_extension.

#include <cstdint>
#include <limits>

namespace rayd::slang {

struct Float2 {
    float x = 0.f, y = 0.f;
    constexpr Float2() = default;
    constexpr Float2(float x_, float y_) : x(x_), y(y_) {}
};

struct Float3 {
    float x = 0.f, y = 0.f, z = 0.f;
    constexpr Float3() = default;
    constexpr Float3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
};

struct Ray {
    Float3 o;
    Float3 d = Float3(0.f, 0.f, 1.f);
    float tmax = std::numeric_limits<float>::infinity();
    constexpr Ray() = default;
    constexpr Ray(Float3 o_, Float3 d_, float t_ = std::numeric_limits<float>::infinity())
        : o(o_), d(d_), tmax(t_) {}
};

struct Intersection {
    bool valid = false;
    float t = std::numeric_limits<float>::infinity();
    Float3 p, n, geo_n;
    Float2 uv;
    Float3 barycentric;
    int shape_id = -1, prim_id = -1;
};

struct IntersectionAD {
    // forward result
    bool valid = false;
    float t = std::numeric_limits<float>::infinity();
    Float3 p, n, geo_n, barycentric;
    Float2 uv;
    int shape_id = -1, prim_id = -1;
    // backward: d(t)/d(ray)
    Float3 dt_do;  // d(t)/d(ray_origin)
    Float3 dt_dd;  // d(t)/d(ray_direction)
};

struct NearestPointEdge {
    bool valid = false;
    float distance = std::numeric_limits<float>::infinity();
    Float3 point;
    float edge_t = 0.f;
    Float3 edge_point;
    int shape_id = -1, edge_id = -1;
    bool is_boundary = false;
};

struct NearestRayEdge {
    bool valid = false;
    float distance = std::numeric_limits<float>::infinity();
    float ray_t = 0.f;
    Float3 point;
    float edge_t = 0.f;
    Float3 edge_point;
    int shape_id = -1, edge_id = -1;
    bool is_boundary = false;
};

struct PrimaryEdgeSample {
    bool valid = false;
    float x_dot_n = 0.f;
    int idx = -1;
    Ray ray_n, ray_p;
    float pdf = 0.f;
};

struct SceneHandle {
    uint64_t value = 0;
    constexpr SceneHandle() = default;
    constexpr explicit SceneHandle(uint64_t v) : value(v) {}
    constexpr explicit operator bool() const { return value != 0; }
};

struct CameraHandle {
    uint64_t value = 0;
    constexpr CameraHandle() = default;
    constexpr explicit CameraHandle(uint64_t v) : value(v) {}
    constexpr explicit operator bool() const { return value != 0; }
};

// --- Constructors ---
inline Float2 make_float2(float x, float y) { return {x, y}; }
inline Float3 make_float3(float x, float y, float z) { return {x, y, z}; }
inline Ray    make_ray(Float3 o, Float3 d, float t = std::numeric_limits<float>::infinity()) { return {o, d, t}; }
inline SceneHandle  make_scene_handle_raw(uint64_t v) { return SceneHandle(v); }
inline CameraHandle make_camera_handle_raw(uint64_t v) { return CameraHandle(v); }

// --- Float accessors ---
inline float f2_x(const Float2 &v) { return v.x; }
inline float f2_y(const Float2 &v) { return v.y; }
inline float f3_x(const Float3 &v) { return v.x; }
inline float f3_y(const Float3 &v) { return v.y; }
inline float f3_z(const Float3 &v) { return v.z; }

// --- Ray accessors ---
inline Float3 ray_o(const Ray &r) { return r.o; }
inline Float3 ray_d(const Ray &r) { return r.d; }
inline float  ray_tmax(const Ray &r) { return r.tmax; }

// --- Intersection accessors ---
inline bool   its_valid(const Intersection &h) { return h.valid; }
inline float  its_t(const Intersection &h) { return h.t; }
inline Float3 its_p(const Intersection &h) { return h.p; }
inline Float3 its_n(const Intersection &h) { return h.n; }
inline Float3 its_geo_n(const Intersection &h) { return h.geo_n; }
inline Float2 its_uv(const Intersection &h) { return h.uv; }
inline Float3 its_barycentric(const Intersection &h) { return h.barycentric; }
inline int    its_shape_id(const Intersection &h) { return h.shape_id; }
inline int    its_prim_id(const Intersection &h) { return h.prim_id; }

// --- IntersectionAD accessors ---
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

// --- Scene/Camera query stubs (linked at runtime from rayd_core) ---
// These are declared here and defined in rayd_core (interop.h).
// When link_rayd=True, load_module links rayd_core which provides the implementations.

Intersection scene_intersect(SceneHandle handle, const Ray &ray, bool active = true);
IntersectionAD scene_intersect_ad(SceneHandle handle, const Ray &ray, bool active = true);
bool scene_shadow_test(SceneHandle handle, const Ray &ray, bool active = true);
NearestPointEdge scene_nearest_edge_point(SceneHandle handle, const Float3 &point, bool active = true);
NearestRayEdge scene_nearest_edge_ray(SceneHandle handle, const Ray &ray, bool active = true);
bool scene_is_ready(SceneHandle handle);
bool scene_has_pending_updates(SceneHandle handle);
void scene_configure(SceneHandle handle);
void scene_commit_updates(SceneHandle handle);

Ray camera_sample_ray(CameraHandle handle, const Float2 &sample);
PrimaryEdgeSample camera_sample_primary_edge(CameraHandle handle, float sample);
bool camera_is_ready(CameraHandle handle);
bool camera_primary_edges_ready(CameraHandle handle);
void camera_set_resolution(CameraHandle handle, int width, int height);
void camera_configure(CameraHandle handle, bool cache = true);
void camera_prepare_edges(CameraHandle camera_handle, SceneHandle scene_handle);

} // namespace rayd::slang
