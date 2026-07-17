#include <rayd/trace/cuda_trace_backend.h>

#include <algorithm>
#include <vector>

#include <drjit-core/jit.h>

#include <rayd/native_launch_audit.h>
#include <rayd/trace/triangle_bvh_gpu.h>
#include <rayd/utils.h>

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
    caps.device_callable = false;   // arrives in P4 (device-callable fused executor)
    caps.jit_symbolic = false;      // eager native only; cannot fold into a megakernel
    caps.fused_multipath = false;   // P4
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

template OptixIntersection CudaTraceBackend::intersect<true>(const Ray &, Mask &) const;
template OptixIntersection CudaTraceBackend::intersect<false>(const RayAD &, MaskAD &) const;
template Mask CudaTraceBackend::shadow_test<true>(const Ray &, Mask) const;
template MaskAD CudaTraceBackend::shadow_test<false>(const RayAD &, MaskAD) const;

} // namespace rayd
