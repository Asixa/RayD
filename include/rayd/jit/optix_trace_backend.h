#pragma once

#include <memory>
#include <vector>

#include <rayd/jit/scene_optix.h>
#include <rayd/jit/trace_backend.h>

namespace rayd {

/// Which OptiX scene(s) a query should traverse, plus the SBT hit-group count.
/// In split mode a query traces the static and dynamic scenes separately and
/// merges the winners; otherwise it traces the single unified scene.
struct OptixSceneSelection {
    const OptixScene *primary = nullptr;    ///< Unified scene, or the static scene in split mode.
    const OptixScene *secondary = nullptr;  ///< Dynamic scene in split mode; null otherwise.
    int split_mode = 0;                     ///< 1 when the static/dynamic split is active.
    int hitgroup_record_count = 0;          ///< SBT hit-group record count (scene mesh count).
};

/// Aggregate OptiX GAS/IAS timing returned from a sync(), summed across scenes.
struct OptixTraceSyncResult {
    double gas_update_ms = 0.0;  ///< Total per-mesh GAS refit time.
    double ias_update_ms = 0.0;  ///< Total top-level IAS rebuild time.
};

/// \brief OptiX triangle trace backend: owns the OptiX GAS/IAS scene(s) and the
/// static/dynamic split bookkeeping formerly held directly by Scene.
///
/// The build/sync logic is a verbatim transplant of the previous Scene::build()
/// and Scene::sync() OptiX blocks, so results remain bit-identical.
class OptixTraceBackend final : public TraceBackend {
public:
    OptixTraceBackend();
    ~OptixTraceBackend() override;

    TraceBackendKind kind() const override { return TraceBackendKind::Optix; }
    TraceCapabilities capabilities() const override;
    bool is_ready() const override;

    /// \brief Build the OptiX acceleration structure(s) for \p meshes.
    /// \param meshes         Full ordered mesh descriptor list (SBT order).
    /// \param dynamic_flags  Per-mesh dynamic flag, parallel to \p meshes; drives
    ///                       the static/dynamic split decision.
    void build(const std::vector<OptixSceneMeshDesc> &meshes,
               const std::vector<bool> &dynamic_flags);

    /// Apply incremental \p updates; returns the aggregate GAS/IAS timing.
    OptixTraceSyncResult sync(const std::vector<OptixSceneMeshDesc> &meshes,
                              const std::vector<OptixSceneMeshUpdate> &updates);

    /// True when the static/dynamic split is active for this build.
    bool split_active() const { return split_active_; }
    /// The scene(s) a query should traverse (see OptixSceneSelection).
    OptixSceneSelection select_scenes() const;

    /// Unified scene (also the split-mode trace source that owns the pipeline/SBT).
    OptixScene &primary() const { return *scene_; }
    /// Static-only scene (split mode).
    OptixScene &static_scene() const { return *static_scene_; }
    /// Dynamic-only scene (split mode).
    OptixScene &dynamic_scene() const { return *dynamic_scene_; }

private:
    std::unique_ptr<OptixScene> scene_;
    std::unique_ptr<OptixScene> static_scene_;
    std::unique_ptr<OptixScene> dynamic_scene_;
    bool split_active_ = false;
    int hitgroup_record_count_ = 0;
    std::vector<int> static_mesh_indices_;
    std::vector<int> dynamic_mesh_indices_;
    std::vector<int> dynamic_mesh_local_index_;
};

} // namespace rayd
