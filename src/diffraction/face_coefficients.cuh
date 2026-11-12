// Copyright Xingyu Chen.
// Defines the slab and layer-stack reflection coefficients of the diffraction face terms.

#pragma once

#include <rayd/field_transport.cuh>
#include <rayd/utd.h>
#include <src/transmission_device.cuh>

#include <cstdint>

// A diffraction face term reflects off its face. These providers give it the coefficient the specular reflection of
// that face uses: a finite slab of the face's (eta, sigma, mu) and thickness, or the whole layer stack when the face's
// layer-store row holds more than one layer (the stack evaluation leaves the scalar view unused). Without a thickness
// table the half-space coefficient of utd::HalfSpaceFaceCoefficients applies.

namespace rayd::shared::diffraction_faces {

namespace utd = ::rayd::shared::diffraction;
namespace em = ::rayd::shared::transmission;
namespace transport = ::rayd::shared::field_transport;

// The material layer store (CSR over layer-store rows) shared by the faces.
struct FaceLayerStore {
    const int* offset = nullptr;
    const int* count = nullptr;
    const float* thickness_m = nullptr;
    const float* eps_r = nullptr;
    const float* sigma_e = nullptr;
    const float* mu_r = nullptr;
    int row_count = 0;
    int64_t layer_total = 0;
};

// Slab thickness and layer-store row of one face; row < 0 or a row of fewer than two usable layers is a slab.
struct FaceLayers {
    float thickness_m = 0.f;
    int row = -1;
};

RAYD_DEVICE bool face_stack(const FaceLayerStore& store, const FaceLayers& face, em::LayerView& view) {
    if (store.count == nullptr || face.row < 0 || face.row >= store.row_count || store.count[face.row] < 2)
        return false;
    view = {store.offset, store.count, store.thickness_m, store.eps_r, store.sigma_e, store.mu_r, face.row};
    return em::layer_material_usable(view, store.layer_total);
}

// The wall a transmission path crosses a face with: layer-store row `row` (one or more usable layers).
RAYD_DEVICE bool face_wall(const FaceLayerStore& store, int row, em::LayerView& view) {
    if (store.count == nullptr || row < 0 || row >= store.row_count || store.count[row] < 1)
        return false;
    view = {store.offset, store.count, store.thickness_m, store.eps_r, store.sigma_e, store.mu_r, row};
    return em::layer_material_usable(view, store.layer_total);
}

// Total thickness of a wall, as the transmission paths sum it for their carrier correction.
RAYD_DEVICE float wall_thickness(const em::LayerView& view) {
    float thickness = 0.f;
    const int first = view.layer_offset[view.material];
    for (int layer = 0; layer < view.layer_count[view.material]; ++layer)
        thickness += fmaxf(view.layer_thickness_m[first + layer], 0.f);
    return thickness;
}

// Reflection coefficients (s / p = s x k convention, gain included) of one face at an incidence cosine.
RAYD_DEVICE utd::FaceFresnelT<float> face_fresnel(const FaceLayerStore& store, const FaceLayers& face,
                                                  const utd::FaceMaterialParams& material, float cos_theta,
                                                  float frequency_hz) {
    utd::FaceFresnelT<float> out;
    em::LayerView view;
    if (face_stack(store, face, view)) {
        // The stack recursion reports the TM coefficient as a tangential-E ratio, the negative of this convention.
        out.rTE = utd::cplx_mul_real(em::stack_rt(cos_theta, view, frequency_hz, em::kPolTE).r, material.gain);
        out.rTM = utd::cplx_mul_real(em::stack_rt(cos_theta, view, frequency_hz, em::kPolTM).r, -material.gain);
        return out;
    }
    transport::slab_fresnel(cos_theta, material.etaR, material.sigma, material.muR, material.gain, face.thickness_m,
                            frequency_hz, out.rTE, out.rTM);
    return out;
}

// Provider of a float pair (the order-1 export): faces with a material, at a positive angular frequency. Without a
// face-layer table (`layered` false) it is the half-space provider.
struct LayeredFaceCoefficients {
    bool layered = false;
    FaceLayerStore store;
    FaceLayers face0;
    FaceLayers face1;
    float omega = 0.f;
    float frequency_hz = 0.f;

    RAYD_DEVICE bool evaluates(int, const utd::FaceMaterialParams& material) const {
        return material.present > 0.5f && omega > 0.f;
    }
    RAYD_DEVICE utd::FaceFresnelT<float> operator()(int face, const utd::FaceMaterialParams& material,
                                                    float cos_theta) const {
        if (!layered)
            return utd::HalfSpaceFaceCoefficients<float>{omega}(face, material, cos_theta);
        return face_fresnel(store, face == 0 ? face0 : face1, material, cos_theta, frequency_hz);
    }
};

// The layered provider of a caller whose solve also carries straight-through transmission paths (with at most
// `transmission_crossings` wall crossings): each face also names the layer-store row those paths cross it with
// (-1 for a face they cannot cross), so the pair's incident terms compensate only the GO step the paths leave (see
// utd::wedge_incident_transmission).
struct TransmittingFaceCoefficients : LayeredFaceCoefficients {
    int transmission_crossings = 0;
    int transmission_row0 = -1;
    int transmission_row1 = -1;

    RAYD_DEVICE int transmission_depth() const { return layered ? transmission_crossings : 0; }
    // The face's wall transmission as the transmission paths evaluate it (TE and TM of its layer-store row).
    RAYD_DEVICE bool transmission(int face, const utd::FaceMaterialParams&, float cos_theta,
                                  utd::FaceTransmissionT<float>& out) const {
        em::LayerView view;
        if (!face_wall(store, face == 0 ? transmission_row0 : transmission_row1, view))
            return false;
        out.tTE = em::stack_rt(cos_theta, view, frequency_hz, em::kPolTE).t;
        out.tTM = em::stack_rt(cos_theta, view, frequency_hz, em::kPolTM).t;
        out.thicknessM = wall_thickness(view);
        return true;
    }
};

} // namespace rayd::shared::diffraction_faces
