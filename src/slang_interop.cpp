// Implementations of rayd::slang:: interop functions.
// Moved out of the header so they become proper symbols in rayd_core,
// allowing external code (slangc-generated C++) to link against them.

#include <rayd/intersection.h>
#include <rayd/ray.h>
#include <rayd/slang/interop.h>

namespace rayd::slang {

bool scene_is_ready(SceneHandle handle) {
    return detail::handle_ref<rayd::Scene>(handle.value,
        "rayd::slang::scene_is_ready(): null scene handle.").is_ready();
}

bool scene_has_pending_updates(SceneHandle handle) {
    return detail::handle_ref<rayd::Scene>(handle.value,
        "rayd::slang::scene_has_pending_updates(): null scene handle.").has_pending_updates();
}

void scene_configure(SceneHandle handle) {
    detail::handle_ref<rayd::Scene>(handle.value,
        "rayd::slang::scene_configure(): null scene handle.").configure();
}

void scene_commit_updates(SceneHandle handle) {
    detail::handle_ref<rayd::Scene>(handle.value,
        "rayd::slang::scene_commit_updates(): null scene handle.").commit_updates();
}

Intersection scene_intersect(SceneHandle handle, const Ray &ray, bool active) {
    const rayd::Scene &scene = detail::handle_ref<rayd::Scene>(handle.value,
        "rayd::slang::scene_intersect(): null scene handle.");
    const IntersectionDetached hit =
        scene.ray_intersect<true>(detail::to_cuda(ray), detail::scalar_mask(active));
    drjit::eval(hit.t, hit.p, hit.n, hit.geo_n, hit.uv, hit.barycentric,
                hit.shape_id, hit.prim_id);
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

bool scene_shadow_test(SceneHandle handle, const Ray &ray, bool active) {
    const rayd::Scene &scene = detail::handle_ref<rayd::Scene>(handle.value,
        "rayd::slang::scene_shadow_test(): null scene handle.");
    const MaskDetached shadow =
        scene.shadow_test<true>(detail::to_cuda(ray), detail::scalar_mask(active));
    drjit::eval(shadow);
    drjit::sync_thread();
    return detail::lane0<bool>(shadow);
}

NearestPointEdge scene_closest_edge_point(SceneHandle handle, const Float3 &point, bool active) {
    const rayd::Scene &scene = detail::handle_ref<rayd::Scene>(handle.value,
        "rayd::slang::scene_closest_edge_point(): null scene handle.");
    const NearestPointEdgeDetached hit =
        scene.closest_edge<true>(detail::to_cuda(point), detail::scalar_mask(active));
    drjit::eval(hit.distance, hit.point, hit.edge_t, hit.edge_point,
                hit.shape_id, hit.edge_id, hit.is_boundary);
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

NearestRayEdge scene_closest_edge_ray(SceneHandle handle, const Ray &ray, bool active) {
    const rayd::Scene &scene = detail::handle_ref<rayd::Scene>(handle.value,
        "rayd::slang::scene_closest_edge_ray(): null scene handle.");
    const NearestRayEdgeDetached hit =
        scene.closest_edge<true>(detail::to_cuda(ray), detail::scalar_mask(active));
    drjit::eval(hit.distance, hit.ray_t, hit.point, hit.edge_t, hit.edge_point,
                hit.shape_id, hit.edge_id, hit.is_boundary);
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

bool camera_is_ready(CameraHandle handle) {
    return detail::handle_ref<rayd::PerspectiveCamera>(handle.value,
        "rayd::slang::camera_is_ready(): null camera handle.").is_ready();
}

bool camera_primary_edges_ready(CameraHandle handle) {
    return detail::handle_ref<rayd::PerspectiveCamera>(handle.value,
        "rayd::slang::camera_primary_edges_ready(): null camera handle.").primary_edges_ready();
}

void camera_set_resolution(CameraHandle handle, int width, int height) {
    rayd::PerspectiveCamera &camera = detail::handle_ref<rayd::PerspectiveCamera>(handle.value,
        "rayd::slang::camera_set_resolution(): null camera handle.");
    camera.set_width(width);
    camera.set_height(height);
}

void camera_configure(CameraHandle handle, bool cache) {
    detail::handle_ref<rayd::PerspectiveCamera>(handle.value,
        "rayd::slang::camera_configure(): null camera handle.").configure(cache);
}

void camera_prepare_edges(CameraHandle camera_handle, SceneHandle scene_handle) {
    rayd::PerspectiveCamera &camera = detail::handle_ref<rayd::PerspectiveCamera>(camera_handle.value,
        "rayd::slang::camera_prepare_edges(): null camera handle.");
    const rayd::Scene &scene = detail::handle_ref<rayd::Scene>(scene_handle.value,
        "rayd::slang::camera_prepare_edges(): null scene handle.");
    camera.prepare_primary_edges(scene);
}

Ray camera_sample_ray(CameraHandle handle, const Float2 &sample) {
    const rayd::PerspectiveCamera &camera = detail::handle_ref<rayd::PerspectiveCamera>(handle.value,
        "rayd::slang::camera_sample_ray(): null camera handle.");
    return detail::to_scalar(camera.sample_primary_ray(detail::to_cuda(sample)));
}

PrimaryEdgeSample camera_sample_primary_edge(CameraHandle handle, float sample) {
    const rayd::PerspectiveCamera &camera = detail::handle_ref<rayd::PerspectiveCamera>(handle.value,
        "rayd::slang::camera_sample_primary_edge(): null camera handle.");
    const rayd::PrimaryEdgeSample sample_result =
        camera.sample_primary_edge(detail::scalar_float(sample));
    drjit::eval(sample_result.x_dot_n, sample_result.idx,
                sample_result.ray_n.o, sample_result.ray_n.d, sample_result.ray_n.tmax,
                sample_result.ray_p.o, sample_result.ray_p.d, sample_result.ray_p.tmax,
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
