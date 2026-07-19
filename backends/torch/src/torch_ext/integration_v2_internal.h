#pragma once

#include <rayd/torch/integration_v2.h>
#include <rayd/torch/scene/cache.h>

namespace rayd::torch::detail {

struct IntegrationAccess {
    static rayd::torch_backend::SceneCache &scene_cache(const SceneResource &scene);
};

} // namespace rayd::torch::detail
