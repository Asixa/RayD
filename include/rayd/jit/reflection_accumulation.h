// Copyright Xingyu Chen.
// Declares the Dr.Jit reflection accumulation API.

#pragma once

#include <type_traits>

#include <drjit/complex.h>

#include <rayd/jit/core.h>

namespace rayd {

/// Axis-aligned 2D accumulation grid: a plane at `position` along `axis`, spanning
/// [coord0_min, coord0_max] x [coord1_min, coord1_max] at resolution0 x resolution1 cells.
struct AccumGrid {
    int axis = 2;          ///< Plane normal axis (0 = x, 1 = y, 2 = z).
    float position = 0.f;  ///< Plane offset along `axis`.
    float coord0_min = 0.f;
    float coord0_max = 0.f;
    float coord1_min = 0.f;
    float coord1_max = 0.f;
    int resolution0 = 0;   ///< Cell count along the first in-plane axis.
    int resolution1 = 0;   ///< Cell count along the second in-plane axis.
};

/// Parameters for native reflection accumulation (field synthesis and Russian-roulette termination).
struct AccumOptions {
    float wavelength = 1.f;          ///< Wavelength in world units.
    float k = 0.f;                   ///< Wavenumber (2*pi / wavelength).
    float solid_angle_per_ray = 1.f; ///< Solid angle each ray represents.
    float cell_area = 1.f;           ///< Area of one grid cell.
    int seed = 0;                    ///< RNG seed.
    int rr_depth = 0;                ///< Bounce depth at which Russian roulette starts.
    float rr_prob = 1.f;             ///< Survival probability per roulette test.
    float stop_threshold = 0.f;      ///< Stop a path once its contribution falls below this.
    bool collect_wedges = false;         ///< Record diffraction-wedge events.
    bool collect_wedge_prefixes = false; ///< Also record the reflection prefix of each wedge event.
    int wedge_capacity = 0;          ///< Maximum wedge events to store.
    int wedge_sample_stride = 1;     ///< Prefix wedge event sampling stride.
};

/// Per-primitive material used by reflection accumulation (electromagnetic surface parameters).
template <typename Float_>
struct MaterialData {
    static constexpr bool IsDetached = std::is_same_v<Float_, Float>;

    using Mask_ = std::conditional_t<IsDetached, Mask, MaskAD>;

    Float_ eta_r = full<Float_>(1.f, 1);  ///< Relative permittivity.
    Float_ sigma = full<Float_>(0.f, 1);  ///< Conductivity.
    Float_ gain = full<Float_>(1.f, 1);   ///< Extra gain factor.
    Float_ mu_r = full<Float_>(1.f, 1);   ///< Relative permeability.
    Mask_ valid = full<Mask_>(false, 1);  ///< Whether this entry holds a real material.

    DRJIT_STRUCT(MaterialData,
                 eta_r,
                 sigma,
                 gain,
                 mu_r,
                 valid)
};

/// Compacted list of diffraction-wedge events recorded during accumulation; `count`
/// entries (up to `capacity`) of the per-event arrays are valid.
template <typename Float_>
struct WedgeEventsData {
    static constexpr bool IsDetached = std::is_same_v<Float_, Float>;

    using Vec3f = std::conditional_t<IsDetached, Vector3f, Vector3fAD>;
    using Int_ = std::conditional_t<IsDetached, Int, IntAD>;

    int capacity = 0;                       ///< Allocated event slots.
    Int_ count = full<Int_>(0, 1);          ///< Number of recorded events.
    Int_ ray_index = full<Int_>(-1, 1);     ///< Source ray of each event.
    Vec3f hit_points = zeros<Vec3f>(1);     ///< Wedge location.
    Vec3f normals = zeros<Vec3f>(1);        ///< Surface normal at the wedge.
    Int_ prim_id = full<Int_>(-1, 1);       ///< Primitive forming the wedge.
    Vec3f directions = zeros<Vec3f>(1);          ///< Incident direction at the wedge.
    Vec3f source_points = zeros<Vec3f>(1);       ///< Virtual source used before the wedge event.
    Float_ src_power = zeros<Float_>(1);      ///< Source field power carried into the wedge.
    Vec3f initial_directions = zeros<Vec3f>(1);  ///< Primary ray direction that produced the event.
    Int_ bounce_depth = full<Int_>(-1, 1);       ///< Reflection depth at which the event occurred.

    DRJIT_STRUCT(WedgeEventsData,
                 count,
                 ray_index,
                 hit_points,
                 normals,
                 prim_id,
                 directions,
                 source_points,
                 src_power,
                 initial_directions,
                 bounce_depth)
};

/// Result of reflection accumulation: power and complex field accumulated per grid cell,
/// plus optional diffraction-wedge events. Grid arrays have grid_cell_count entries.
template <typename Float_>
struct AccumResultData {
    static constexpr bool IsDetached = std::is_same_v<Float_, Float>;

    using FloatArray = Float_;
    using ComplexArray = drjit::Complex<Float_>;
    using Int_ = std::conditional_t<IsDetached, Int, IntAD>;
    using WedgeBuffer = WedgeEventsData<Float_>;

    int ray_count = 0;
    int max_bounces = 0;
    int grid_cell_count = 0;
    FloatArray reflection_power = zeros<FloatArray>(1); ///< Accumulated power per grid cell.
    ComplexArray reflection_field_x =                   ///< Accumulated complex x field per cell.
        ComplexArray(zeros<FloatArray>(1), zeros<FloatArray>(1));
    ComplexArray reflection_field_y =                   ///< Accumulated complex y field per cell.
        ComplexArray(zeros<FloatArray>(1), zeros<FloatArray>(1));
    ComplexArray reflection_field_z =                   ///< Accumulated complex z field per cell.
        ComplexArray(zeros<FloatArray>(1), zeros<FloatArray>(1));
    Int_ reflection_count = full<Int_>(0, 1);           ///< Total reflection contributions accumulated.
    WedgeBuffer wedge_events;                            ///< Recorded wedge events (if collection was enabled).

    DRJIT_STRUCT(AccumResultData,
                 reflection_power,
                 reflection_field_x,
                 reflection_field_y,
                 reflection_field_z,
                 reflection_count,
                 wedge_events)
};

} // namespace rayd
