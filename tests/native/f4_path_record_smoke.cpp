#include <rayd/path_exchange.h>

using namespace rayd::shared::multipath;

int main() {
    PathInteractionRecord interaction = {
        PathInteractionKind::Reflection,
        7,
        InvalidPathId,
        0,
        {1.0f, 2.0f, 3.0f},
        {0.0f, 1.0f, 0.0f},
    };
    PathRecord path = {};
    path.valid = 1u;
    path.fixed_winner = 1u;
    path.order = 1;
    path.source_index = 2;
    path.receiver_index = 3;
    path.provenance = PathProvenance::ReflectionTrace;
    path.available_fields = PathDerivativeInteractionPosition |
                            PathDerivativeInteractionNormal |
                            PathDerivativeTotalLength;
    path.differentiable_fields = PathDerivativeInteractionPosition |
                                 PathDerivativeTotalLength;
    path.interaction_count = 1u;
    path.total_length = 4.0f;
    PathRecordBatchView view = {
        &path,
        &interaction,
        nullptr,
        nullptr,
        1u,
        1u,
        PathDerivativeMode::None,
        0u,
    };
    return view.paths[0].order == 1 &&
                   view.interactions[0].global_primitive_id == 7
        ? 0
        : 1;
}
