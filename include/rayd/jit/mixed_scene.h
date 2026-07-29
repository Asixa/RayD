// Copyright Xingyu Chen.
// Declares the Dr.Jit unified mesh, SDF, and surfel scene API.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <rayd/jit/scene.h>
#include <rayd/jit/sdf.h>
#include <rayd/jit/surfel.h>
#include <rayd/jit/visibility.h>

namespace rayd {

/// GPU-resident composition of triangle meshes, bounded SDF grids, and surfels.
class MixedScene final {
  public:
    explicit MixedScene(const std::string& edge_bvh_backend = "auto", const std::string& trace_backend = "auto");
    ~MixedScene();

    MixedScene(const MixedScene&) = delete;
    MixedScene& operator=(const MixedScene&) = delete;

    int add_mesh(const Mesh& mesh, bool dynamic = false);
    int add_sdf(const SdfGrid& grid, const SdfTraceOptions& options = {});
    int add_surfel(const SurfelCloud& cloud, const SurfelTraceOptions& options = {});
    void build();
    bool is_ready() const;

    int num_meshes() const { return mesh_count_; }
    int num_sdfs() const { return static_cast<int>(sdfs_.size()); }
    int num_surfel_scenes() const { return static_cast<int>(surfels_.size()); }

    template <bool Detached>
    IntersectionT<Detached> intersect(const RayT<Detached>& ray, MaskT<Detached> active = true,
                                      RayFlags flags = RayFlags::All) const;

    template <bool Detached>
    SegmentVisibilityT<Detached> visible(const Vector3fT<Detached>& start, const Vector3fT<Detached>& end,
                                         MaskT<Detached> active = true) const;

    template <bool Detached>
    ReflectionChainT<Detached> trace_reflections(const RayT<Detached>& ray, int max_bounces,
                                                 MaskT<Detached> active = true) const;

    template <bool Detached>
    FloatT<Detached> transmittance(const RayT<Detached>& ray, MaskT<Detached> active = true) const;

  private:
    struct SdfEntry {
        SdfGrid grid;
        SdfTraceOptions options;
    };

    Scene mesh_scene_;
    std::vector<SdfEntry> sdfs_;
    std::vector<std::unique_ptr<SurfelScene>> surfels_;
    std::vector<int> surfel_prefix_;
    int mesh_count_ = 0;
    int mesh_face_count_ = 0;
    bool ready_ = false;
};

} // namespace rayd
