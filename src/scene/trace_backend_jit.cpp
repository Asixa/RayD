#include <rayd/trace/drjit/cuda_trace_backend.h>

#include <algorithm>
#include <vector>

#include <drjit-core/jit.h>

#include <rayd/diagnostics/drjit/native_launch_audit.h>
#include <src/scene/cuda_multipath_gpu_jit.h>
#include <rayd/trace/drjit/triangle_bvh_gpu.h>
#include <rayd/core/drjit/utils.h>

#include <rayd/shared/bvh/host_topology.h>
#include <rayd/shared/bvh/topology.h>

namespace rayd {

namespace {

constexpr int kTraversalStackDepth = shared::bvh::kBvhTraversalStackDepth;
constexpr int kLeafSize = shared::bvh::kBvhLeafSize;

Int load_ints(const std::vector<int> &values) {
    if (values.empty()) {
        return Int();
    }
    return load<Int>(values.data(), values.size());
}

std::vector<int> copy_ints_to_host(const Int &values) {
    const size_t count = values.size();
    if (count == 0) {
        return {};
    }
    std::vector<int> result(count);
    drjit::store(result.data(), values);
    return result;
}

Vector3f empty_vector3(int count) {
    return Vector3f(empty<Float>(count), empty<Float>(count), empty<Float>(count));
}

Vector3f load_vector3(const std::vector<ScalarVector3f> &values) {
    const size_t count = values.size();
    if (count == 0) {
        return Vector3f();
    }
    std::vector<float> x(count), y(count), z(count);
    for (size_t index = 0; index < count; ++index) {
        x[index] = values[index].x();
        y[index] = values[index].y();
        z[index] = values[index].z();
    }
    return Vector3f(load<Float>(x.data(), count), load<Float>(y.data(), count),
                    load<Float>(z.data(), count));
}

std::vector<ScalarVector3f> copy_vector3_to_host(const Vector3f &values) {
    const size_t count = static_cast<size_t>(slices(values));
    if (count == 0) {
        return {};
    }
    std::vector<float> x(count), y(count), z(count);
    drjit::store(x.data(), values.x());
    drjit::store(y.data(), values.y());
    drjit::store(z.data(), values.z());
    std::vector<ScalarVector3f> result(count);
    for (size_t index = 0; index < count; ++index) {
        result[index] = ScalarVector3f(x[index], y[index], z[index]);
    }
    return result;
}

TriBvhTrianglePtrs triangle_ptrs(const Vector3f &p0, const Vector3f &e1, const Vector3f &e2) {
    return {p0[0].data(), p0[1].data(), p0[2].data(),
            e1[0].data(), e1[1].data(), e1[2].data(),
            e2[0].data(), e2[1].data(), e2[2].data()};
}

TriBvhBoundsPtrs bounds_ptrs(Vector3f &min_bounds, Vector3f &max_bounds) {
    return {min_bounds[0].data(), min_bounds[1].data(), min_bounds[2].data(),
            max_bounds[0].data(), max_bounds[1].data(), max_bounds[2].data()};
}

} // namespace

CudaTraceBackend::CudaTraceBackend() = default;
CudaTraceBackend::~CudaTraceBackend() = default;

TraceCapabilities CudaTraceBackend::capabilities() const {
    TraceCapabilities caps;
    caps.closest_hit = true;
    caps.any_hit = true;
    caps.first_blocker = true;
    caps.ignore_primitives = true;
    caps.instancing = false;
    caps.refit = true;
    caps.compaction = true;
    caps.device_callable = false;   // eager native only; not a device-callable megakernel
    caps.jit_symbolic = false;      // eager native only; cannot fold into a megakernel
    // P4 Stage D: the CUDA fused multipath executor serves every multipath
    // pipeline -- reflection trace, segment visibility, reflection EPC, reflection
    // accumulation, diffraction path export, and diffraction accumulation
    // (accum_dfr / accum_dfr_direct / accum_dfr_coherent_direct).
    caps.fused_multipath = true;
    caps.cpu = false;
    return caps;
}

void CudaTraceBackend::build(const TriangleInfo &triangles, const Int &shape_id, const Int &local_prim_id) {
    build_or_refit(triangles, shape_id, local_prim_id, false);
}

void CudaTraceBackend::sync(const TriangleInfo &triangles, const Int &shape_id, const Int &local_prim_id) {
    build_or_refit(triangles, shape_id, local_prim_id, true);
}

void CudaTraceBackend::build_or_refit(const TriangleInfo &triangles,
                                      const Int &shape_id,
                                      const Int &local_prim_id,
                                      bool refit) {
    tri_p0_ = triangles.p0;
    tri_e1_ = triangles.e1;
    tri_e2_ = triangles.e2;
    shape_id_ = shape_id;
    local_prim_id_ = local_prim_id;
    primitive_count_ = static_cast<int>(slices(triangles.p0));
    require(primitive_count_ > 0, "CudaTraceBackend: scene has no triangles.");

    // The native builder uses its own non-blocking streams; finish the Dr.Jit
    // producers before exposing their device pointers.
    drjit::eval(tri_p0_, tri_e1_, tri_e2_);
    drjit::sync_thread();

    if (refit) {
        require(node_count_ > 0, "CudaTraceBackend::sync(): backend was never built.");
        drjit::eval(node_bbox_min_, node_bbox_max_, left_child_, right_child_, leaf_primitives_,
                    leaf_nodes_, refit_level_nodes_);
        drjit::sync_thread();

        const int level_count = static_cast<int>(refit_level_offsets_.empty()
                                                     ? 0
                                                     : refit_level_offsets_.size() - 1);
        refit_triangle_bvh_gpu(node_count_, triangle_ptrs(tri_p0_, tri_e1_, tri_e2_),
                               left_child_.data(), right_child_.data(), leaf_primitives_.data(),
                               leaf_nodes_.data(), static_cast<int>(leaf_nodes_.size()),
                               refit_level_nodes_.data(), refit_level_offsets_.data(), level_count,
                               bounds_ptrs(node_bbox_min_, node_bbox_max_));
        ready_ = true;
        return;
    }

    const int raw_node_count = std::max(2 * primitive_count_ - 1, 1);
    Vector3f primitive_bbox_min = empty_vector3(primitive_count_);
    Vector3f primitive_bbox_max = empty_vector3(primitive_count_);
    Vector3f raw_node_min = empty_vector3(raw_node_count);
    Vector3f raw_node_max = empty_vector3(raw_node_count);
    Int raw_left = empty<Int>(raw_node_count);
    Int raw_right = empty<Int>(raw_node_count);
    Int raw_leaf_primitive = empty<Int>(raw_node_count);
    Int raw_is_leaf = empty<Int>(raw_node_count);
    Int raw_primitive_leaf_node = empty<Int>(primitive_count_);

    build_triangle_bvh_gpu(primitive_count_, triangle_ptrs(tri_p0_, tri_e1_, tri_e2_),
                           bounds_ptrs(primitive_bbox_min, primitive_bbox_max),
                           bounds_ptrs(raw_node_min, raw_node_max), raw_left.data(),
                           raw_right.data(), raw_leaf_primitive.data(), raw_is_leaf.data(),
                           raw_primitive_leaf_node.data());

    const std::vector<int> host_left = copy_ints_to_host(raw_left);
    const std::vector<int> host_right = copy_ints_to_host(raw_right);
    const std::vector<int> host_is_leaf = copy_ints_to_host(raw_is_leaf);
    const std::vector<int> host_leaf_primitive = copy_ints_to_host(raw_leaf_primitive);
    const std::vector<ScalarVector3f> host_node_min = copy_vector3_to_host(raw_node_min);
    const std::vector<ScalarVector3f> host_node_max = copy_vector3_to_host(raw_node_max);

    std::vector<int> subtree_leaf_counts(static_cast<size_t>(raw_node_count), -1);
    shared::bvh::compute_subtree_leaf_count(0, host_left, host_right, host_is_leaf,
                                            subtree_leaf_counts);

    shared::bvh::HostCompactedBvh<ScalarVector3f> compacted;
    compacted.primitive_leaf_nodes.assign(static_cast<size_t>(primitive_count_), -1);
    shared::bvh::emit_compacted_preorder(0, host_left, host_right, host_leaf_primitive,
                                         host_is_leaf, subtree_leaf_counts, host_node_min,
                                         host_node_max, kLeafSize, compacted);
    require(compacted.leaf_primitives.size() == static_cast<size_t>(primitive_count_),
            "CudaTraceBackend::build(): compaction lost triangle primitives.");

    node_count_ = static_cast<int>(compacted.left_child.size());
    node_bbox_min_ = load_vector3(compacted.node_bbox_min);
    node_bbox_max_ = load_vector3(compacted.node_bbox_max);
    left_child_ = load_ints(compacted.left_child);
    right_child_ = load_ints(compacted.right_child);
    leaf_primitives_ = load_ints(compacted.leaf_primitives);
    primitive_leaf_node_ = load_ints(compacted.primitive_leaf_nodes);

    const std::vector<int> &final_left = compacted.left_child;
    const std::vector<int> &final_right = compacted.right_child;
    const std::vector<int> &final_is_leaf = compacted.is_leaf;

    std::vector<int> heights(static_cast<size_t>(node_count_), -1);
    const int max_height =
        shared::bvh::compute_node_height(0, final_left, final_right, final_is_leaf, heights);
    require(max_height + 1 <= kTraversalStackDepth,
            "CudaTraceBackend::build(): BVH depth exceeds the traversal stack capacity.");

    std::vector<std::vector<int>> levels(static_cast<size_t>(max_height + 1));
    std::vector<int> leaf_nodes;
    leaf_nodes.reserve(static_cast<size_t>(primitive_count_));
    for (int node = 0; node < node_count_; ++node) {
        if (final_is_leaf[static_cast<size_t>(node)] != 0) {
            leaf_nodes.push_back(node);
            continue;
        }
        const int height = heights[static_cast<size_t>(node)];
        if (height >= 1) {
            levels[static_cast<size_t>(height)].push_back(node);
        }
    }

    // Concatenate internal levels in ascending height so a refit updates every
    // child before its parent.
    std::vector<int> refit_level_nodes;
    refit_level_offsets_.clear();
    refit_level_offsets_.push_back(0);
    for (int height = 1; height <= max_height; ++height) {
        const std::vector<int> &level = levels[static_cast<size_t>(height)];
        if (level.empty()) {
            continue;
        }
        refit_level_nodes.insert(refit_level_nodes.end(), level.begin(), level.end());
        refit_level_offsets_.push_back(static_cast<int>(refit_level_nodes.size()));
    }
    refit_level_nodes_ = load_ints(refit_level_nodes);
    leaf_nodes_ = load_ints(leaf_nodes);

    drjit::eval(tri_p0_, tri_e1_, tri_e2_, shape_id_, local_prim_id_, node_bbox_min_,
                node_bbox_max_, left_child_, right_child_, leaf_primitives_, primitive_leaf_node_,
                leaf_nodes_, refit_level_nodes_);
    drjit::sync_thread();
    jit_flush_malloc_cache();
    ready_ = true;
}

template <bool Detached>
OptixIntersection CudaTraceBackend::intersect_impl(const RayT<Detached> &ray,
                                                   MaskT<Detached> &active) const {
    ScopedNativeLaunchStage stage(NativeLaunchStage::Intersect);
    require(ready_, "CudaTraceBackend::intersect(): backend is not built.");
    const int ray_count = static_cast<int>(slices(ray.o));

    OptixIntersection intersection;
    intersection.reserve(ray_count);
    if (ray_count == 0) {
        // Match the OptiX empty-batch RuntimeError so the public contract is
        // backend-independent.
        require(false, "CudaTraceBackend::intersect(): empty ray batch.");
    }

    Float ox, oy, oz, dx, dy, dz, t_max_input;
    if constexpr (!Detached) {
        ox = detach<false>(ray.o.x());
        oy = detach<false>(ray.o.y());
        oz = detach<false>(ray.o.z());
        dx = detach<false>(ray.d.x());
        dy = detach<false>(ray.d.y());
        dz = detach<false>(ray.d.z());
        t_max_input = detach<false>(ray.tmax);
    } else {
        ox = ray.o.x();
        oy = ray.o.y();
        oz = ray.o.z();
        dx = ray.d.x();
        dy = ray.d.y();
        dz = ray.d.z();
        t_max_input = ray.tmax;
    }

    const Mask active_detached = detach<false>(active);
    const Float t_max = select(drjit::isfinite(t_max_input), t_max_input, full<Float>(1e8f, ray_count));
    Int active_flags = select(active_detached, full<Int>(1, ray_count), zeros<Int>(ray_count));

    Int stack_nodes = empty<Int>(kTraversalStackDepth * ray_count);
    Int overflow = empty<Int>(ray_count);

    drjit::eval(ox, oy, oz, dx, dy, dz, t_max, active_flags, tri_p0_, tri_e1_, tri_e2_,
                node_bbox_min_, node_bbox_max_, left_child_, right_child_, leaf_primitives_,
                shape_id_, local_prim_id_, intersection.t, intersection.barycentric,
                intersection.shape_id, intersection.local_prim_id, stack_nodes, overflow);
    // eval() is a no-op for literal-backed arrays (e.g. the select()-folded
    // t_max or an all-ones active_flags); their fill kernels are only enqueued
    // by .data(). Touch every pointer the native launch consumes BEFORE the
    // stream sync, or the fill races the query kernel on the backend stream.
    (void) ox.data(); (void) oy.data(); (void) oz.data();
    (void) dx.data(); (void) dy.data(); (void) dz.data();
    (void) t_max.data(); (void) active_flags.data();
    (void) intersection.t.data();
    (void) intersection.barycentric[0].data(); (void) intersection.barycentric[1].data();
    (void) intersection.shape_id.data(); (void) intersection.local_prim_id.data();
    (void) stack_nodes.data(); (void) overflow.data();
    drjit::sync_thread();

    query_triangle_closest_hit_gpu(
        ray_count, primitive_count_, node_count_, static_cast<int>(leaf_primitives_.size()),
        RayEpsilon, triangle_ptrs(tri_p0_, tri_e1_, tri_e2_),
        {node_bbox_min_[0].data(), node_bbox_min_[1].data(), node_bbox_min_[2].data(),
         node_bbox_max_[0].data(), node_bbox_max_[1].data(), node_bbox_max_[2].data()},
        left_child_.data(), right_child_.data(), leaf_primitives_.data(),
        {ox.data(), oy.data(), oz.data(), dx.data(), dy.data(), dz.data(), t_max.data(),
         active_flags.data()},
        shape_id_.data(), local_prim_id_.data(), intersection.t.data(),
        intersection.barycentric[0].data(), intersection.barycentric[1].data(),
        intersection.shape_id.data(), intersection.local_prim_id.data(), stack_nodes.data(),
        overflow.data());

    const Mask is_hit = neq(intersection.shape_id, -1);
    if constexpr (!Detached) {
        active = MaskAD(is_hit);
    } else {
        active = is_hit;
    }
    return intersection;
}

template <bool Detached>
OptixIntersection CudaTraceBackend::intersect(const RayT<Detached> &ray, MaskT<Detached> &active) const {
    return intersect_impl<Detached>(ray, active);
}

template <bool Detached>
MaskT<Detached> CudaTraceBackend::shadow_test_impl(const RayT<Detached> &ray, MaskT<Detached> active) const {
    ScopedNativeLaunchStage stage(NativeLaunchStage::Intersect);
    require(ready_, "CudaTraceBackend::shadow_test(): backend is not built.");
    const int ray_count = static_cast<int>(slices(ray.o));

    MaskT<Detached> hit = full<MaskT<Detached>>(false, ray_count);
    if (ray_count == 0) {
        return hit;
    }

    Float ox, oy, oz, dx, dy, dz, t_max_input;
    if constexpr (!Detached) {
        ox = detach<false>(ray.o.x());
        oy = detach<false>(ray.o.y());
        oz = detach<false>(ray.o.z());
        dx = detach<false>(ray.d.x());
        dy = detach<false>(ray.d.y());
        dz = detach<false>(ray.d.z());
        t_max_input = detach<false>(ray.tmax);
    } else {
        ox = ray.o.x();
        oy = ray.o.y();
        oz = ray.o.z();
        dx = ray.d.x();
        dy = ray.d.y();
        dz = ray.d.z();
        t_max_input = ray.tmax;
    }

    const Mask active_detached = detach<false>(active);
    const Float t_max = select(drjit::isfinite(t_max_input), t_max_input, full<Float>(1e8f, ray_count));
    Int active_flags = select(active_detached, full<Int>(1, ray_count), zeros<Int>(ray_count));

    Int out_hit = empty<Int>(ray_count);
    Int stack_nodes = empty<Int>(kTraversalStackDepth * ray_count);
    Int overflow = empty<Int>(ray_count);

    drjit::eval(ox, oy, oz, dx, dy, dz, t_max, active_flags, tri_p0_, tri_e1_, tri_e2_,
                node_bbox_min_, node_bbox_max_, left_child_, right_child_, leaf_primitives_,
                out_hit, stack_nodes, overflow);
    // See intersect_impl: literal-backed arrays materialize on .data(), which
    // must happen before the stream sync to keep the native launch ordered.
    (void) ox.data(); (void) oy.data(); (void) oz.data();
    (void) dx.data(); (void) dy.data(); (void) dz.data();
    (void) t_max.data(); (void) active_flags.data();
    (void) out_hit.data(); (void) stack_nodes.data(); (void) overflow.data();
    drjit::sync_thread();

    query_triangle_occluded_gpu(
        ray_count, primitive_count_, node_count_, static_cast<int>(leaf_primitives_.size()),
        RayEpsilon, triangle_ptrs(tri_p0_, tri_e1_, tri_e2_),
        {node_bbox_min_[0].data(), node_bbox_min_[1].data(), node_bbox_min_[2].data(),
         node_bbox_max_[0].data(), node_bbox_max_[1].data(), node_bbox_max_[2].data()},
        left_child_.data(), right_child_.data(), leaf_primitives_.data(),
        {ox.data(), oy.data(), oz.data(), dx.data(), dy.data(), dz.data(), t_max.data(),
         active_flags.data()},
        out_hit.data(), stack_nodes.data(), overflow.data());

    const Mask hit_detached = neq(out_hit, 0);
    if constexpr (!Detached) {
        hit = MaskAD(hit_detached);
    } else {
        hit = hit_detached;
    }
    return hit;
}

template <bool Detached>
MaskT<Detached> CudaTraceBackend::shadow_test(const RayT<Detached> &ray, MaskT<Detached> active) const {
    return shadow_test_impl<Detached>(ray, active);
}

std::vector<int> CudaTraceBackend::first_blocker_selftest(const Vector3f &origin,
                                                          const Vector3f &direction,
                                                          const Float &tmax,
                                                          const std::vector<int> &ignore_prim_ids) const {
    ScopedNativeLaunchStage stage(NativeLaunchStage::Intersect);
    require(ready_, "CudaTraceBackend::first_blocker_selftest(): backend is not built.");
    const int ray_count = static_cast<int>(slices(origin));
    if (ray_count == 0) {
        return {};
    }

    Float ox = origin.x(), oy = origin.y(), oz = origin.z();
    Float dx = direction.x(), dy = direction.y(), dz = direction.z();
    const Float t_max = select(drjit::isfinite(tmax), tmax, full<Float>(1e8f, ray_count));

    // Broadcast the single ignore row across all rays (row-major, stride entries per ray).
    const int ignore_stride = static_cast<int>(ignore_prim_ids.size());
    Int ignore;
    if (ignore_stride > 0) {
        std::vector<int> tiled(static_cast<size_t>(ignore_stride) * ray_count);
        for (int r = 0; r < ray_count; ++r) {
            for (int i = 0; i < ignore_stride; ++i) {
                tiled[static_cast<size_t>(r) * ignore_stride + i] =
                    ignore_prim_ids[static_cast<size_t>(i)];
            }
        }
        ignore = load<Int>(tiled.data(), tiled.size());
    }

    Int out_global = empty<Int>(ray_count);
    Int stack_nodes = empty<Int>(kTraversalStackDepth * ray_count);
    Int overflow = empty<Int>(ray_count);

    drjit::eval(ox, oy, oz, dx, dy, dz, t_max, tri_p0_, tri_e1_, tri_e2_, node_bbox_min_,
                node_bbox_max_, left_child_, right_child_, leaf_primitives_, out_global,
                stack_nodes, overflow);
    // See intersect_impl: literal-backed arrays materialize on .data(), which
    // must happen before the stream sync to keep the native launch ordered.
    (void) ox.data(); (void) oy.data(); (void) oz.data();
    (void) dx.data(); (void) dy.data(); (void) dz.data();
    (void) t_max.data();
    (void) out_global.data(); (void) stack_nodes.data(); (void) overflow.data();
    if (ignore_stride > 0) {
        drjit::eval(ignore);
        (void) ignore.data();
    }
    drjit::sync_thread();

    query_triangle_first_blocker_gpu(
        ray_count, primitive_count_, node_count_, static_cast<int>(leaf_primitives_.size()),
        RayEpsilon, triangle_ptrs(tri_p0_, tri_e1_, tri_e2_),
        {node_bbox_min_[0].data(), node_bbox_min_[1].data(), node_bbox_min_[2].data(),
         node_bbox_max_[0].data(), node_bbox_max_[1].data(), node_bbox_max_[2].data()},
        left_child_.data(), right_child_.data(), leaf_primitives_.data(),
        {ox.data(), oy.data(), oz.data(), dx.data(), dy.data(), dz.data(), t_max.data(), nullptr},
        ignore_stride > 0 ? ignore.data() : nullptr, ignore_stride, out_global.data(),
        stack_nodes.data(), overflow.data());

    return copy_ints_to_host(out_global);
}

CudaMultipathBvh CudaTraceBackend::multipath_bvh() const {
    CudaMultipathBvh bvh;
    bvh.p0_x = tri_p0_[0].data();
    bvh.p0_y = tri_p0_[1].data();
    bvh.p0_z = tri_p0_[2].data();
    bvh.e1_x = tri_e1_[0].data();
    bvh.e1_y = tri_e1_[1].data();
    bvh.e1_z = tri_e1_[2].data();
    bvh.e2_x = tri_e2_[0].data();
    bvh.e2_y = tri_e2_[1].data();
    bvh.e2_z = tri_e2_[2].data();
    bvh.node_min_x = node_bbox_min_[0].data();
    bvh.node_min_y = node_bbox_min_[1].data();
    bvh.node_min_z = node_bbox_min_[2].data();
    bvh.node_max_x = node_bbox_max_[0].data();
    bvh.node_max_y = node_bbox_max_[1].data();
    bvh.node_max_z = node_bbox_max_[2].data();
    bvh.left_child = left_child_.data();
    bvh.right_child = right_child_.data();
    bvh.leaf_primitives = leaf_primitives_.data();
    bvh.shape_id = shape_id_.data();
    bvh.local_prim_id = local_prim_id_.data();
    bvh.primitive_count = primitive_count_;
    bvh.node_count = node_count_;
    bvh.leaf_primitive_count = static_cast<int>(leaf_primitives_.size());
    return bvh;
}

void CudaTraceBackend::materialize_for_fused_launch() const {
    // The fused kernel runs on its own non-blocking stream; finish the Dr.Jit
    // producers of the BVH buffers and drain the thread stream (which also drains
    // the param fills the caller enqueued via .data()) before exposing pointers.
    // eval() is a no-op for literal-backed arrays; the .data() touches below are
    // the real enqueue points and must precede sync_thread (commit b7f7226).
    drjit::eval(tri_p0_, tri_e1_, tri_e2_, node_bbox_min_, node_bbox_max_, left_child_,
                right_child_, leaf_primitives_, shape_id_, local_prim_id_);
    (void) tri_p0_[0].data(); (void) tri_p0_[1].data(); (void) tri_p0_[2].data();
    (void) tri_e1_[0].data(); (void) tri_e1_[1].data(); (void) tri_e1_[2].data();
    (void) tri_e2_[0].data(); (void) tri_e2_[1].data(); (void) tri_e2_[2].data();
    (void) node_bbox_min_[0].data(); (void) node_bbox_min_[1].data(); (void) node_bbox_min_[2].data();
    (void) node_bbox_max_[0].data(); (void) node_bbox_max_[1].data(); (void) node_bbox_max_[2].data();
    (void) left_child_.data(); (void) right_child_.data(); (void) leaf_primitives_.data();
    (void) shape_id_.data(); (void) local_prim_id_.data();
    drjit::sync_thread();
}

void CudaTraceBackend::run_reflection_trace(shared::optix::ReflectionTraceParams params,
                                            int lane_count) const {
    ScopedNativeLaunchStage stage(NativeLaunchStage::TraceReflections);
    require(ready_, "CudaTraceBackend::run_reflection_trace(): backend is not built.");
    params.split_mode = 0;
    params.primary_handle = 0;
    params.secondary_handle = 0;
    materialize_for_fused_launch();
    launch_reflection_trace_cuda(params, multipath_bvh(), lane_count);
}

void CudaTraceBackend::run_segment_visibility(shared::optix::SegmentVisibilityParams params,
                                              CudaSegmentVisibilityVariant variant,
                                              int lane_count) const {
    ScopedNativeLaunchStage stage(NativeLaunchStage::TraceReflections);
    require(ready_, "CudaTraceBackend::run_segment_visibility(): backend is not built.");
    // The CUDA triangle BVH is the scene; a non-zero handle sentinel satisfies the
    // algorithm's null-scene guard (segment_visibility_algo trace_segment).
    params.handle = 1ull;
    materialize_for_fused_launch();
    launch_segment_visibility_cuda(params, multipath_bvh(), variant, lane_count);
}

void CudaTraceBackend::run_reflection_accumulation(AccumParams params, int lane_count) const {
    ScopedNativeLaunchStage stage(NativeLaunchStage::AccumulateReflections);
    require(ready_, "CudaTraceBackend::run_reflection_accumulation(): backend is not built.");
    params.split_mode = 0;
    params.primary_handle = 0;
    params.secondary_handle = 0;
    materialize_for_fused_launch();
    launch_reflection_accumulation_cuda(params, multipath_bvh(), lane_count);
}

void CudaTraceBackend::run_reflection_epc(shared::optix::ReflEpcParams params, bool direct_only,
                                          bool primary_visibility_only, int lane_count) const {
    ScopedNativeLaunchStage stage(NativeLaunchStage::TraceReflections);
    require(ready_, "CudaTraceBackend::run_reflection_epc(): backend is not built.");
    params.split_mode = 0;
    params.primary_handle = 0;
    params.secondary_handle = 0;
    materialize_for_fused_launch();
    launch_reflection_epc_cuda(params, multipath_bvh(), direct_only, primary_visibility_only,
                               lane_count);
}

void CudaTraceBackend::run_dfr_paths(DfrPathParams params, int lane_count) const {
    ScopedNativeLaunchStage stage(NativeLaunchStage::AccumDfr);
    require(ready_, "CudaTraceBackend::run_dfr_paths(): backend is not built.");
    params.split_mode = 0;
    params.primary_handle = 0;
    params.secondary_handle = 0;
    materialize_for_fused_launch();
    launch_dfr_paths_cuda(params, multipath_bvh(), lane_count);
}

void CudaTraceBackend::run_dfr_accum_direct(DfrAccumParams params, bool has_non_suffix_strategy,
                                            bool has_suffix_strategy, int lane_count) const {
    ScopedNativeLaunchStage stage(NativeLaunchStage::AccumDfr);
    require(ready_, "CudaTraceBackend::run_dfr_accum_direct(): backend is not built.");
    params.split_mode = 0;
    params.primary_handle = 0;
    params.secondary_handle = 0;
    materialize_for_fused_launch();
    launch_dfr_accum_direct_cuda(params, multipath_bvh(), has_non_suffix_strategy,
                                 has_suffix_strategy, lane_count);
}

void CudaTraceBackend::run_dfr_accum_coherent(DfrAccumParams params, int lane_count) const {
    ScopedNativeLaunchStage stage(NativeLaunchStage::AccumDfr);
    require(ready_, "CudaTraceBackend::run_dfr_accum_coherent(): backend is not built.");
    params.split_mode = 0;
    params.primary_handle = 0;
    params.secondary_handle = 0;
    materialize_for_fused_launch();
    launch_dfr_accum_coherent_cuda(params, multipath_bvh(), lane_count);
}

void CudaTraceBackend::run_dfr_accum_chain(DfrAccumParams params, int lane_count) const {
    ScopedNativeLaunchStage stage(NativeLaunchStage::AccumDfr);
    require(ready_, "CudaTraceBackend::run_dfr_accum_chain(): backend is not built.");
    params.split_mode = 0;
    params.primary_handle = 0;
    params.secondary_handle = 0;
    materialize_for_fused_launch();
    launch_dfr_accum_chain_cuda(params, multipath_bvh(), lane_count);
}

void CudaTraceBackend::run_dfr_accum_combined(DfrAccumParams params, bool has_non_suffix_strategy,
                                              bool has_suffix_strategy, int lane_count) const {
    ScopedNativeLaunchStage stage(NativeLaunchStage::AccumDfr);
    require(ready_, "CudaTraceBackend::run_dfr_accum_combined(): backend is not built.");
    params.split_mode = 0;
    params.primary_handle = 0;
    params.secondary_handle = 0;
    materialize_for_fused_launch();
    launch_dfr_accum_combined_cuda(params, multipath_bvh(), has_non_suffix_strategy,
                                   has_suffix_strategy, lane_count);
}

template OptixIntersection CudaTraceBackend::intersect<true>(const Ray &, Mask &) const;
template OptixIntersection CudaTraceBackend::intersect<false>(const RayAD &, MaskAD &) const;
template Mask CudaTraceBackend::shadow_test<true>(const Ray &, Mask) const;
template MaskAD CudaTraceBackend::shadow_test<false>(const RayAD &, MaskAD) const;

} // namespace rayd

// Consolidated OptiX trace backend implementation.
#include <rayd/trace/drjit/optix_trace_backend.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace rayd {

namespace {

/// Whether to split static and dynamic meshes into separate OptiX scenes (env RAYD_OPTIX_SPLIT_MODE).
enum class OptixSplitMode {
    Auto,
    Off,
    On
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

} // namespace

OptixTraceBackend::OptixTraceBackend() = default;
OptixTraceBackend::~OptixTraceBackend() = default;

TraceCapabilities OptixTraceBackend::capabilities() const {
    TraceCapabilities caps;
    caps.closest_hit = true;
    caps.any_hit = true;
    caps.first_blocker = true;
    caps.ignore_primitives = true;
    caps.instancing = false;
    caps.refit = true;
    caps.compaction = true;
    caps.device_callable = false;
    caps.jit_symbolic = true;
    caps.fused_multipath = true;
    caps.cpu = false;
    return caps;
}

bool OptixTraceBackend::is_ready() const {
    if (split_active_) {
        return scene_ != nullptr && static_scene_ != nullptr &&
               dynamic_scene_ != nullptr && scene_->is_ready() &&
               static_scene_->is_ready() && dynamic_scene_->is_ready();
    }
    return scene_ != nullptr && scene_->is_ready();
}

void OptixTraceBackend::build(const std::vector<OptixSceneMeshDesc> &mesh_descs,
                              const std::vector<bool> &dynamic_flags) {
    hitgroup_record_count_ = static_cast<int>(mesh_descs.size());

    int static_mesh_count = 0;
    int dynamic_mesh_count = 0;
    for (bool dynamic : dynamic_flags) {
        if (dynamic) {
            ++dynamic_mesh_count;
        } else {
            ++static_mesh_count;
        }
    }

    split_active_ =
        should_split_optix_scene(active_optix_split_mode(), static_mesh_count, dynamic_mesh_count);
    static_mesh_indices_.clear();
    dynamic_mesh_indices_.clear();
    dynamic_mesh_local_index_.assign(dynamic_flags.size(), -1);

    scene_ = std::make_unique<OptixScene>();
    static_scene_ = std::make_unique<OptixScene>();
    dynamic_scene_ = std::make_unique<OptixScene>();

    if (split_active_) {
        std::vector<OptixSceneMeshDesc> static_mesh_descs;
        std::vector<OptixSceneMeshDesc> dynamic_mesh_descs;
        static_mesh_descs.reserve(static_mesh_count);
        dynamic_mesh_descs.reserve(dynamic_mesh_count);

        for (size_t mesh_index = 0; mesh_index < dynamic_flags.size(); ++mesh_index) {
            if (dynamic_flags[mesh_index]) {
                dynamic_mesh_local_index_[mesh_index] =
                    static_cast<int>(dynamic_mesh_descs.size());
                dynamic_mesh_indices_.push_back(static_cast<int>(mesh_index));
                dynamic_mesh_descs.push_back(mesh_descs[mesh_index]);
            } else {
                static_mesh_indices_.push_back(static_cast<int>(mesh_index));
                static_mesh_descs.push_back(mesh_descs[mesh_index]);
            }
        }

        scene_->build(mesh_descs);
        static_scene_->build(static_mesh_descs, scene_.get());
        dynamic_scene_->build(dynamic_mesh_descs, scene_.get());
    } else {
        scene_->build(mesh_descs);
    }
}

OptixTraceSyncResult OptixTraceBackend::sync(
    const std::vector<OptixSceneMeshDesc> &mesh_descs,
    const std::vector<OptixSceneMeshUpdate> &updates) {
    OptixTraceSyncResult result;

    if (split_active_) {
        if (!updates.empty()) {
            scene_->sync(mesh_descs, updates);
        }

        std::vector<OptixSceneMeshDesc> dynamic_mesh_descs;
        dynamic_mesh_descs.reserve(dynamic_mesh_indices_.size());
        for (int mesh_index : dynamic_mesh_indices_) {
            dynamic_mesh_descs.push_back(mesh_descs[static_cast<size_t>(mesh_index)]);
        }

        std::vector<OptixSceneMeshUpdate> dynamic_updates;
        dynamic_updates.reserve(updates.size());
        for (const OptixSceneMeshUpdate &update : updates) {
            const int dynamic_local_index =
                dynamic_mesh_local_index_[static_cast<size_t>(update.mesh_id)];
            if (dynamic_local_index < 0) {
                continue;
            }
            dynamic_updates.push_back(
                { dynamic_local_index, update.vertices_dirty, update.transform_dirty });
        }

        if (!dynamic_updates.empty()) {
            dynamic_scene_->sync(dynamic_mesh_descs, dynamic_updates);
        }
        if (!updates.empty()) {
            const OptixSyncProfile &optix_profile = scene_->last_sync_profile();
            result.gas_update_ms += optix_profile.gas_update_ms;
            result.ias_update_ms += optix_profile.ias_update_ms;
        }
        if (!dynamic_updates.empty()) {
            const OptixSyncProfile &optix_profile = dynamic_scene_->last_sync_profile();
            result.gas_update_ms += optix_profile.gas_update_ms;
            result.ias_update_ms += optix_profile.ias_update_ms;
        }
    } else {
        scene_->sync(mesh_descs, updates);
        const OptixSyncProfile &optix_profile = scene_->last_sync_profile();
        result.gas_update_ms = optix_profile.gas_update_ms;
        result.ias_update_ms = optix_profile.ias_update_ms;
    }

    return result;
}

OptixSceneSelection OptixTraceBackend::select_scenes() const {
    OptixSceneSelection selection;
    selection.hitgroup_record_count = hitgroup_record_count_;
    if (split_active_) {
        selection.primary = static_scene_.get();
        selection.secondary = dynamic_scene_.get();
        selection.split_mode = 1;
    } else {
        selection.primary = scene_.get();
    }
    return selection;
}

} // namespace rayd
