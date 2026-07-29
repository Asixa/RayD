// Copyright Xingyu Chen.
// Implements bindings support for module.

#include <pybind11/pybind11.h>

#include <rayd/contracts.h>

namespace rayd::torch_backend {

namespace py = pybind11;

py::dict build_info() {
    py::dict info;
    info["backend"] = "torch";
    info["uses_dr_jit"] = false;
    info["role"] = "compatibility_metadata_shim";
    return info;
}

py::dict contract_values() {
    py::dict values;
    values["invalid_signed_id"] = shared::InvalidSignedId;
    values["invalid_unsigned_id"] = shared::InvalidUnsignedId;
    values["general_epsilon"] = shared::GeneralEpsilon;
    values["ray_epsilon"] = shared::RayEpsilon;
    values["shadow_epsilon"] = shared::ShadowEpsilon;
    values["edge_epsilon"] = shared::EdgeEpsilon;
    values["small_epsilon"] = shared::SmallEpsilon;
    values["vacuum_permittivity"] = shared::VacuumPermittivity;
    values["speed_of_light"] = shared::SpeedOfLight;
    values["ray_flags_none"] = static_cast<std::uint32_t>(shared::RayFlagBits::None);
    values["ray_flags_geometric"] = static_cast<std::uint32_t>(shared::RayFlagBits::Geometric);
    values["ray_flags_shading_n"] = static_cast<std::uint32_t>(shared::RayFlagBits::ShadingN);
    values["ray_flags_uv"] = static_cast<std::uint32_t>(shared::RayFlagBits::UV);
    values["ray_flags_all"] = static_cast<std::uint32_t>(shared::RayFlagBits::All);
    values["intersection_field_count"] = static_cast<std::uint32_t>(shared::IntersectionField::Count);
    values["nearest_point_edge_field_count"] = static_cast<std::uint32_t>(shared::NearestPointEdgeField::Count);
    values["nearest_ray_edge_field_count"] = static_cast<std::uint32_t>(shared::NearestRayEdgeField::Count);
    return values;
}

PYBIND11_MODULE(_C, m) {
    m.doc() = "RayD Torch compatibility metadata shim.";
    m.def("build_info", &build_info);
    m.def("contract_values", &contract_values);
}

} // namespace rayd::torch_backend
