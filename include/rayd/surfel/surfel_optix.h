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

/// Detached alpha-composite result computed by the native surfel pipeline.
struct SurfelOptixComposite {
    void reserve(int64_t size);

    int64_t m_size = 0;
    int hit_capacity = 0;
    Float intensity;
    Float channels;
    Vector3f normal;
    Float alpha;
    Float transmittance;
    Float depth;
    Int surfel_id;  ///< Flat [ray_count, hit_capacity] sorted candidate ids.
    Float hit_t;    ///< Flat [ray_count, hit_capacity] analytic candidate depth.
    Float hit_alpha;
    Float hit_value;
    Int candidate_count;
    Mask candidate_buffer_full;
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
               int triangle_count,
               bool build_hitobject_pipeline = true);
    bool is_ready() const;

    template <bool Detached>
    SurfelOptixIntersection intersect(const RayT<Detached> &ray,
                                       MaskT<Detached> &active) const;

    template <bool Detached>
    SurfelOptixIntersection intersect(const RayT<Detached> &ray,
                                      const FloatT<Detached> &t_min,
                                      MaskT<Detached> &active) const;

    template <bool Detached>
    SurfelOptixIntersection trace_analytic_candidates(const RayT<Detached> &ray,
                                                      const Int &triangle_to_surfel_id,
                                                      const Vector3f &center,
                                                      const Vector3f &tangent_u,
                                                      const Vector3f &tangent_v,
                                                      const Float &opacity,
                                                      float alpha_min,
                                                      float alpha_cap,
                                                      int max_candidate_hits,
                                                      bool face_forward,
                                                      MaskT<Detached> &active) const;

    template <bool Detached>
    SurfelOptixComposite composite_alpha(const RayT<Detached> &ray,
                                         const Int &triangle_to_surfel_id,
                                         const Vector3f &center,
                                         const Vector3f &tangent_u,
                                         const Vector3f &tangent_v,
                                         const Float &opacity,
                                         const Float &value,
                                         float alpha_min,
                                         float alpha_cap,
                                         int max_candidate_hits,
                                         bool collect_candidate_stats,
                                         bool continue_after_full_buffer,
                                         float transmittance_min,
                                         int max_trace_segments,
                                         bool face_forward,
                                         MaskT<Detached> active) const;

    template <bool Detached>
    SurfelOptixComposite render(const RayT<Detached> &ray,
                                const Int &triangle_to_surfel_id,
                                const Vector3f &center,
                                const Vector3f &tangent_u,
                                const Vector3f &tangent_v,
                                const Float &opacity,
                                const Float &appearance_values,
                                int appearance_channel_count,
                                int color_model,
                                int appearance_sh_degree,
                                int sh_degree,
                                int render_channel_count,
                                bool output_normal,
                                const ScalarVector3f &background_rgb,
                                float alpha_min,
                                float alpha_cap,
                                int max_candidate_hits,
                                bool collect_candidate_stats,
                                bool continue_after_full_buffer,
                                float transmittance_min,
                                int max_trace_segments,
                                bool face_forward,
                                MaskT<Detached> active) const;

private:
    SurfelOptixState *m_accel = nullptr;
};

} // namespace rayd
