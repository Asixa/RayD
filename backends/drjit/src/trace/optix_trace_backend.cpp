#include <rayd/trace/optix_trace_backend.h>

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
