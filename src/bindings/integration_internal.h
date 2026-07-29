// Copyright Xingyu Chen.
// Declares internal bindings support for integration internal.

#pragma once

#include <rayd/scene.h>
#include <src/scene/cache.h>

namespace rayd::torch::detail {

struct IntegrationAccess {
    static rayd::torch_backend::SceneCache &scene_cache(const SceneResource &scene);
};

} // namespace rayd::torch::detail
