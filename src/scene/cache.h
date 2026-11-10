// Copyright Xingyu Chen.
// Declares internal scene support for cache.

#pragma once

#include <ATen/ATen.h>
#include <torch/custom_class.h>
#include <cuda_runtime_api.h>
#include <optix.h>
#include <src/bvh_topology.h>
#include <src/bvh_triangle_query.h>
#include <src/edge/bvh_types.h>

#include <condition_variable>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace rayd::torch_backend {

enum class TraceBackend : int64_t {
    Auto = 0,
    Optix = 1,
    Cuda = 2,
};

enum class EdgeBackend : int64_t {
    Auto = 0,
    Optix = 1,
    Cuda = 2,
};

struct MeshRecord {
    at::Tensor vertices;
    at::Tensor faces;
    at::Tensor uv;
    at::Tensor face_uv;
    at::Tensor to_world_left;
    at::Tensor to_world_right;
    bool use_face_normals = false;
    bool edges_enabled = true;
    bool dynamic = false;
    bool pending_update = false;
    int64_t geometry_owner_id = -1;
    at::Tensor instance_transform;
    bool pending_transform = false;
};

struct OptixTriangleAccel {
    at::Tensor vertex_buffer;
    at::Tensor index_buffer;
    at::Tensor gas_buffer;
    at::Tensor gas_temp_buffer;
    OptixTraversableHandle traversable = 0;
    // Refits of a dynamic GAS since it was last built from scratch; see update_triangle_accel().
    int64_t refits_since_build = 0;
};

struct OptixEdgeAccel {
    at::Tensor aabb_buffer;
    at::Tensor gas_buffer;
    at::Tensor gas_temp_buffer;
    OptixTraversableHandle traversable = 0;
    float search_radius = 0.0f;
};

struct OptixInstanceAccel {
    at::Tensor instance_buffer;
    at::Tensor ias_buffer;
    at::Tensor ias_temp_buffer;
    OptixTraversableHandle traversable = 0;
    // The GAS handles the IAS was last built over; an update is valid only while they are unchanged.
    std::vector<OptixTraversableHandle> child_handles;
};

struct CompactEdgeBvh {
    int64_t geometry_version = 0;
    int64_t node_count = 0;
    bool valid = false;
    at::Tensor primitive_min_x, primitive_min_y, primitive_min_z;
    at::Tensor primitive_max_x, primitive_max_y, primitive_max_z;
    at::Tensor node_min_x, node_min_y, node_min_z;
    at::Tensor node_max_x, node_max_y, node_max_z;
    at::Tensor left_child, right_child, leaf_primitives;
    at::Tensor parent, leaf_primitive, is_leaf, primitive_leaf_node;
    at::Tensor packed_bounds, reduced_bound, morton_in, morton_out;
    at::Tensor primitive_ids_in, primitive_ids_out, merge_counters, scratch;
    at::Tensor node_cost, internal_cost_arrivals, treelet_nodes;
};

struct CompactTriangleBvh {
    int64_t geometry_version = 0;
    int64_t node_count = 0;
    bool valid = false;
    at::Tensor primitive_min_x, primitive_min_y, primitive_min_z;
    at::Tensor primitive_max_x, primitive_max_y, primitive_max_z;
    at::Tensor node_min_x, node_min_y, node_min_z;
    at::Tensor node_max_x, node_max_y, node_max_z;
    at::Tensor left_child, right_child, leaf_primitives;
    at::Tensor parent, leaf_primitive, is_leaf, primitive_leaf_node;
    at::Tensor packed_bounds, reduced_bound, morton_in, morton_out;
    at::Tensor primitive_ids_in, primitive_ids_out, merge_counters, scratch;
};

// Reader/writer lock over one registered scene's state. Queries and backward passes share it for the whole op;
// sync(), vertex/transform updates and edge-mask changes take it exclusively. It is reader-preferring: a reader waits
// only while a writer holds the lock, never behind a queued writer, so a thread may nest reads safely.
class SceneStateLock {
  public:
    void lock_shared();
    bool try_lock_shared();
    void unlock_shared();
    void lock();
    bool try_lock();
    void unlock();

  private:
    std::mutex mutex_;
    std::condition_variable released_;
    int64_t readers_ = 0;
    bool writer_ = false;
};

// Orders one scene's device work across the CUDA streams that use it, without blocking the host. The state lock
// orders host access only: a query may still run on its stream when a writer on another stream refits or frees the
// memory it reads, and a query or a writer on another stream may start before a commit's refit or rebuild finished.
// Every reader therefore records an event on its stream when it releases the scene, and waits for the latest commit
// while that may still run; a writer makes its stream wait for the latest commit and every recorded reader before it
// touches anything, and records its own commit after it enqueued its work. Streams are identified by their handles,
// so a stream must not be destroyed while scene reads it issued are still running.
class SceneStreamOrder {
  public:
    SceneStreamOrder() = default;
    SceneStreamOrder(const SceneStreamOrder&) = delete;
    SceneStreamOrder& operator=(const SceneStreamOrder&) = delete;
    ~SceneStreamOrder();

    // Called under the shared state lock before a reader launches anything on `stream`.
    void begin_read(int device_index, cudaStream_t stream);
    // Called under the shared state lock after a reader enqueued its work on `stream`. It never throws: a failure
    // makes every later read and write of the scene throw instead of racing this reader.
    void end_read(int device_index, cudaStream_t stream) noexcept;
    // Whether a reader recorded since the last begin_write() may still run.
    bool reads_pending();
    // Called under the exclusive state lock (or with no reader left) before a writer mutates or frees scene memory:
    // `stream` waits for the previous commit and every recorded reader. Returns whether any reader was outstanding,
    // in which case blocks the writer frees must also be marked as used by `stream` so the caching allocator does not
    // hand them out early.
    bool begin_write(int device_index, cudaStream_t stream);
    // Called after a writer enqueued its work on `stream`: later readers and writers wait for it. Failures are
    // reported like end_read's.
    void end_write(int device_index, cudaStream_t stream) noexcept;
    // Makes every later read and write throw `error` instead of racing work that could not be ordered.
    void fail(const std::exception& error) noexcept;

  private:
    struct StreamState {
        cudaStream_t stream = nullptr;
        cudaEvent_t read_done = nullptr;
        bool read_pending = false;
    };
    // Streams tracked before entries whose reads finished are dropped.
    static constexpr size_t kMaxTrackedStreams = 16;
    StreamState& state_for(int device_index, cudaStream_t stream);
    void prune_finished_streams();
    // Whether the latest commit may still run on the device; caches its completion.
    bool commit_running();
    void require_ordered() const;
    void record_failure(const std::exception& error) noexcept;

    std::mutex mutex_;
    std::vector<StreamState> streams_;
    cudaEvent_t commit_ = nullptr;
    bool commit_pending_ = false;
    // Why an event could not be recorded; every later begin_read() and begin_write() reports it.
    std::string failure_;
};

// Every device tensor member is also listed in for_each_scene_tensor() (scene.cpp), which orders the release of scene
// memory after the queries that may still read it; a new member must be added there.
struct SceneCache {
    SceneCache() = default;
    SceneCache(const SceneCache&) = delete;
    SceneCache& operator=(const SceneCache&) = delete;
    // Orders the release of the scene's memory after every reader still running on another stream.
    ~SceneCache();

    // Guards the scene state against concurrent mutation; see SceneStateLock and get_scene().
    SceneStateLock state_lock;
    // Orders the scene's device work across streams: ScenePin drives it for every read, and each commit that
    // refits, replaces or frees device memory brackets itself with it.
    SceneStreamOrder stream_order;
    // Tensors an update replaced while a query might still read them; the next commit releases them in order.
    std::vector<at::Tensor> retired;
    // Serializes the lazily built custom BVHs, which readers construct on first use under the shared lock.
    std::mutex lazy_build_mutex;
    int64_t handle = 0;
    int64_t version = 1;
    int64_t last_sync_gas_updates = 0;
    int64_t last_sync_ias_updates = 0;
    int64_t edge_version = 1;
    // The `edge_version` the lazily derived edge SoA and OptiX edge GAS tiers were built from (0: never built).
    int64_t edge_soa_version = 0;
    int64_t edge_accel_version = 0;
    int64_t edge_mask_version = 1;
    // Advances whenever a committed sync() changes vertex positions or instance transforms. Differentiable ops
    // record it with their tape, because backward/JVP re-read the scene geometry instead of retracing.
    int64_t geometry_version = 1;
    int64_t device_index = 0;
    TraceBackend trace_backend = TraceBackend::Optix;
    EdgeBackend edge_backend = EdgeBackend::Optix;
    std::vector<MeshRecord> meshes;
    std::vector<OptixTriangleAccel> triangle_accels;
    OptixInstanceAccel triangle_ias;
    at::Tensor global_vertices;
    at::Tensor global_faces;
    // (F, 6) texture coordinates of each global face's three corners, defined only when some mesh supplies UVs;
    // faces of a mesh without UVs carry (0, 0), (1, 0), (0, 1), which reproduces the barycentric (u, v).
    at::Tensor triangle_uv;
    at::Tensor face_offsets;
    at::Tensor face_shape_id;
    at::Tensor face_local_id;
    at::Tensor primitive_identity;
    at::Tensor tri_p0_x;
    at::Tensor tri_p0_y;
    at::Tensor tri_p0_z;
    at::Tensor tri_e1_x;
    at::Tensor tri_e1_y;
    at::Tensor tri_e1_z;
    at::Tensor tri_e2_x;
    at::Tensor tri_e2_y;
    at::Tensor tri_e2_z;
    at::Tensor tri_fn_x;
    at::Tensor tri_fn_y;
    at::Tensor tri_fn_z;
    CompactTriangleBvh custom_triangle_bvh;
    at::Tensor edge_v0;
    at::Tensor edge_v1;
    at::Tensor edge_face0;
    at::Tensor edge_face1;
    at::Tensor edge_shape_id;
    at::Tensor edge_local_id;
    at::Tensor edge_p0_x;
    at::Tensor edge_p0_y;
    at::Tensor edge_p0_z;
    at::Tensor edge_e1_x;
    at::Tensor edge_e1_y;
    at::Tensor edge_e1_z;
    at::Tensor edge_mask;
    at::Tensor edge_opposite;
    CompactEdgeBvh custom_edge_bvh;
    OptixEdgeAccel edge_accel;
    std::vector<OptixEdgeAccel> edge_accels;
    at::Tensor tri_p0_packed;
    at::Tensor tri_e1_packed;
    at::Tensor tri_e2_packed;
    at::Tensor tri_fn_packed;
};

struct SceneHandle : torch::CustomClassHolder {
    explicit SceneHandle(int64_t handle_, bool owns_handle_ = true) : handle(handle_), owns_handle(owns_handle_) {}
    ~SceneHandle();
    int64_t handle = 0;
    bool owns_handle = true;
};

std::unique_ptr<SceneCache> create_scene_cache(std::vector<MeshRecord> meshes);
std::unique_ptr<SceneCache> create_scene_cache(std::vector<MeshRecord> meshes, TraceBackend trace_backend,
                                               EdgeBackend edge_backend);
int64_t create_scene(std::vector<MeshRecord> meshes);
// A registered scene kept alive and locked for the duration of one op. Registered scenes are shared-owned, so
// destroying the Python scene while an op (or a backward pass on an autograd thread) still uses it only drops the
// registry's reference. The lock is released before the reference, and a thread that holds the GIL releases it while
// it waits for the lock, so a writer that must re-enter Python while holding the lock cannot deadlock against it.
class ScenePin {
  public:
    // `device_read` orders a shared pin's stream against scene commits (SceneStreamOrder); pins that only read host
    // metadata skip it.
    ScenePin(std::shared_ptr<SceneCache> scene, bool exclusive, bool device_read = true);
    ScenePin(const ScenePin&) = delete;
    ScenePin& operator=(const ScenePin&) = delete;
    ScenePin(ScenePin&& other) noexcept;
    ScenePin& operator=(ScenePin&&) = delete;
    ~ScenePin();

    SceneCache& operator*() const noexcept { return *scene_; }
    SceneCache* operator->() const noexcept { return scene_.get(); }

  private:
    std::shared_ptr<SceneCache> scene_;
    bool exclusive_ = false;
    bool device_read_ = true;
};

void register_scene_cache(std::unique_ptr<SceneCache> scene);
void destroy_scene(int64_t handle);
// Shared access for queries and backward/JVP passes.
ScenePin get_scene(int64_t handle);
// Shared access for host-side metadata (versions, counts, backend names) that launches no device work.
ScenePin get_scene_metadata(int64_t handle);
// Exclusive access for operations that mutate the scene state.
ScenePin get_scene_for_update(int64_t handle);
c10::intrusive_ptr<SceneHandle> create_scene_cache_from_flat(
    std::vector<at::Tensor> vertices, std::vector<at::Tensor> faces, std::vector<at::Tensor> uv,
    std::vector<at::Tensor> face_uv, std::vector<at::Tensor> to_world_left, std::vector<at::Tensor> to_world_right,
    std::vector<at::Tensor> instance_transforms, std::vector<int64_t> mesh_flags,
    std::vector<int64_t> geometry_owner_ids);
int64_t scene_version(int64_t handle);
int64_t scene_geometry_version(int64_t handle);
// Throws when `recorded_version` (captured with a tape at forward time) no longer matches the scene geometry.
void require_scene_geometry_version(const SceneCache& scene, int64_t recorded_version, const char* operation);
int64_t scene_num_meshes(int64_t handle);
int64_t scene_num_geometries(int64_t handle);
int64_t scene_edge_count(int64_t handle);
std::string scene_trace_backend(int64_t handle);
std::string scene_edge_backend(int64_t handle);
void update_mesh_vertices(int64_t handle, int64_t mesh_id, at::Tensor vertices);
void update_instance_transform(int64_t handle, int64_t mesh_id, at::Tensor vertices, at::Tensor transform);
void sync_scene(int64_t handle);
int64_t scene_version(c10::intrusive_ptr<SceneHandle> scene);
int64_t scene_num_meshes(c10::intrusive_ptr<SceneHandle> scene);
int64_t scene_num_geometries(c10::intrusive_ptr<SceneHandle> scene);
int64_t scene_last_sync_gas_updates(c10::intrusive_ptr<SceneHandle> scene);
int64_t scene_last_sync_ias_updates(c10::intrusive_ptr<SceneHandle> scene);
int64_t scene_edge_count(c10::intrusive_ptr<SceneHandle> scene);
std::vector<at::Tensor> scene_edge_records(c10::intrusive_ptr<SceneHandle> scene);
std::vector<at::Tensor> scene_edge_records(SceneCache& scene);
std::vector<at::Tensor> scene_global_geometry(c10::intrusive_ptr<SceneHandle> scene);
at::Tensor get_scene_edge_mask(c10::intrusive_ptr<SceneHandle> scene);
void set_scene_edge_mask(c10::intrusive_ptr<SceneHandle> scene, at::Tensor mask);
// Lazily derive the edge query structures of the committed geometry; safe to call from concurrent readers.
void ensure_edge_soa(SceneCache& scene);
void ensure_optix_edge_accel(SceneCache& scene);
void ensure_custom_edge_bvh(SceneCache& scene);
void ensure_custom_triangle_bvh(SceneCache& scene);
rayd::shared::bvh::TriangleSoAView scene_triangle_view(const SceneCache& scene);
rayd::shared::bvh::AabbSoAView scene_triangle_bvh_bounds_view(const SceneCache& scene);
rayd::shared::bvh::CompactBvhTopologyView scene_triangle_bvh_topology_view(const SceneCache& scene);
rayd::shared::edge::EdgeSoAView scene_edge_view(const SceneCache& scene);
rayd::shared::edge::AabbSoAView scene_edge_bvh_bounds_view(const SceneCache& scene);
rayd::shared::edge::CompactBvhTopologyView scene_edge_bvh_topology_view(const SceneCache& scene);
void update_mesh_vertices(c10::intrusive_ptr<SceneHandle> scene, int64_t mesh_id, at::Tensor vertices);
void update_instance_transform(c10::intrusive_ptr<SceneHandle> scene, int64_t mesh_id, at::Tensor vertices,
                               at::Tensor transform);
void sync_scene(c10::intrusive_ptr<SceneHandle> scene);
std::vector<at::Tensor> split_scene_vertex_grad(c10::intrusive_ptr<SceneHandle> scene, at::Tensor grad_vertices);
at::Tensor pack_scene_vertex_tangents(c10::intrusive_ptr<SceneHandle> scene,
                                      std::vector<c10::optional<at::Tensor>> tangents);

} // namespace rayd::torch_backend
