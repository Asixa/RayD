#pragma once

#include <rayd/rayd.h>
#include <rayd/optix.h>
#include <vector>

namespace rayd {

struct OptixState;
class Mesh;

struct OptixIntersection {
    void reserve(int64_t size);

    int64_t m_size = 0;
    IntDetached shape_id;
    IntDetached global_prim_id;
    Vector2fDetached barycentric;
    FloatDetached t;
};

struct OptixSceneMeshDesc {
    const Mesh *mesh = nullptr;
    bool dynamic = false;
    int face_offset = 0;
    int mesh_id = -1;
};

struct OptixSceneMeshUpdate {
    int mesh_id = -1;
    bool vertices_dirty = false;
    bool transform_dirty = false;
};

struct OptixCommitProfile {
    double gas_update_ms = 0.0;
    double ias_update_ms = 0.0;
    double total_ms = 0.0;
    int updated_vertex_meshes = 0;
    int updated_transform_meshes = 0;
};

class OptixScene {
public:
    OptixScene();
    ~OptixScene();

    void configure(const std::vector<OptixSceneMeshDesc> &meshes);
    void commit_updates(const std::vector<OptixSceneMeshDesc> &meshes,
                        const std::vector<OptixSceneMeshUpdate> &updates);
    bool is_ready() const;
    const OptixCommitProfile &last_commit_profile() const { return last_commit_profile_; }

    template <bool Detached>
    OptixIntersection ray_intersect(const RayT<Detached> &ray,
                                     MaskT<Detached> &active) const;
    template <bool Detached>
    MaskT<Detached> shadow_test(const RayT<Detached> &ray,
                                MaskT<Detached> active) const;

private:
    OptixState *m_accel = nullptr;
    OptixCommitProfile last_commit_profile_;
};

} // namespace rayd
