"""Pure-Python fallback for the optional ``_C`` metadata compatibility shim."""


def build_info() -> dict[str, object]:
    return {
        "backend": "torch",
        "uses_dr_jit": False,
        "role": "compatibility_metadata_shim",
    }

def contract_values() -> dict[str, int | float]:
    return {
        "invalid_signed_id": -1,
        "invalid_unsigned_id": 0xFFFFFFFF,
        "general_epsilon": 1.0e-5,
        "ray_epsilon": 1.0e-3,
        "shadow_epsilon": 1.0e-3,
        "edge_epsilon": 1.0e-5,
        "small_epsilon": 1.0e-6,
        "vacuum_permittivity": 8.854187817e-12,
        "speed_of_light": 299792458.0,
        "ray_flags_none": 0x00,
        "ray_flags_geometric": 0x01,
        "ray_flags_shading_n": 0x02,
        "ray_flags_uv": 0x04,
        "ray_flags_all": 0x07,
        "intersection_field_count": 10,
        "nearest_point_edge_field_count": 8,
        "nearest_ray_edge_field_count": 9,
    }
