#pragma once

#include <rayd/ray.h>
#include <rayd/optix.h>

namespace rayd {

struct SurfelOptixState;

/// Raw closest-hit result from the surfel triangle GAS. Detached by design:
/// SurfelScene re-gathers AD surfel parameters and recomputes hit attributes.
struct SurfelOptixIntersection {
    void reserve(int64_t size);

    int64_t m_size = 0;
    Int triangle_id;        ///< GAS primitive id; -1 when no hit.
    Vector2f barycentric;   ///< Built-in triangle barycentric (u, v).
    Float t;                ///< Hit distance; Infinity when no hit.
};

/// Standalone OptiX triangle GAS used by the surfel module.
class SurfelOptixScene {
public:
    SurfelOptixScene();
    ~SurfelOptixScene();

    SurfelOptixScene(const SurfelOptixScene &) = delete;
    SurfelOptixScene &operator=(const SurfelOptixScene &) = delete;

    void build(const Float &vertex_buffer,
               const Int &face_buffer,
               int vertex_count,
               int triangle_count);
    bool is_ready() const;

    template <bool Detached>
    SurfelOptixIntersection intersect(const RayT<Detached> &ray,
                                      MaskT<Detached> &active) const;

private:
    SurfelOptixState *m_accel = nullptr;
};

} // namespace rayd
