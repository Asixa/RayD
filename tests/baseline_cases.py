import argparse
import importlib.util
import os
import platform
import subprocess
import sys
import types
from pathlib import Path

from tests.baseline_utils import DEFAULT_TOLERANCE_POLICY, ROOT, write_baseline_tree


PROBE_POINTS = [
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 2.0, 3.0),
]

PROBE_DIRECTIONS = [
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 1.0, 0.0),
]


def _git_output(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def _optional_git_output(*args):
    try:
        return subprocess.check_output(
            args,
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _nvcc_version():
    try:
        return subprocess.check_output(["nvcc", "--version"], cwd=ROOT, text=True).strip().splitlines()[-1]
    except Exception:
        return None


def _lane_scalar(array, lane=0):
    if isinstance(array, (bool, int, float)):
        return array
    return list(array)[lane]


def _float_lane(array, lane=0):
    return float(_lane_scalar(array, lane))


def _int_lane(array, lane=0):
    return int(_lane_scalar(array, lane))


def _bool_lane(array, lane=0):
    return bool(_lane_scalar(array, lane))


def _vec(values, array_type):
    x, y, z = values
    return array_type([x], [y], [z])


def _vec2(values, array_type):
    x, y = values
    return array_type([x], [y])


def _vector_to_list(vec, size, lane=0):
    return [_float_lane(vec[i], lane) for i in range(size)]


def _vector_array_to_rows(vec, size):
    lanes = len(list(vec[0])) if size > 0 else 0
    return [[_float_lane(vec[column], lane) for column in range(size)] for lane in range(lanes)]


def _int_vector_array_to_rows(vec, size):
    lanes = len(list(vec[0])) if size > 0 else 0
    return [[_int_lane(vec[column], lane) for column in range(size)] for lane in range(lanes)]


def _bool_array_to_list(mask):
    return [bool(v) for v in list(mask)]


def _float_array_to_list(array):
    return [float(v) for v in list(array)]


def _int_array_to_list(array):
    return [int(v) for v in list(array)]


def _matrix_rows(pj, mat):
    rows = pj._test_matrix_rows(mat)
    return [_vector_to_list(row, 4) for row in rows]


def _transform_case(pj, cuda, mat):
    return {
        "matrix": _matrix_rows(pj, mat),
        "points": [
            {
                "input": list(point),
                "output": _vector_to_list(pj._test_transform_pos(mat, _vec(point, cuda.Array3f)), 3),
            }
            for point in PROBE_POINTS
        ],
        "directions": [
            {
                "input": list(direction),
                "output": _vector_to_list(pj._test_transform_dir(mat, _vec(direction, cuda.Array3f)), 3),
            }
            for direction in PROBE_DIRECTIONS
        ],
    }


def _ray_to_dict(ray):
    return {
        "o": _vector_to_list(ray.o, 3),
        "d": _vector_to_list(ray.d, 3),
        "tmax": _float_lane(ray.tmax),
    }


def _load_raydi():
    package_root = os.environ.get("RAYDI_PACKAGE_ROOT")
    if not package_root:
        import raydi as pj

        return pj

    package_dir = Path(package_root) / "raydi"
    ext_candidates = sorted(package_dir.glob("raydi*.pyd"))
    if not ext_candidates:
        raise RuntimeError(f"RAYDI_PACKAGE_ROOT does not contain raydi/*.pyd: {package_root}")

    sys.modules.pop("raydi.raydi", None)
    sys.modules.pop("raydi", None)
    spec = importlib.util.spec_from_file_location("raydi.raydi", ext_candidates[0])
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create import spec for {ext_candidates[0]}")

    ext_module = importlib.util.module_from_spec(spec)
    sys.modules["raydi.raydi"] = ext_module
    spec.loader.exec_module(ext_module)

    package = types.ModuleType("raydi")
    package.__file__ = str(package_dir / "__init__.py")
    package.__path__ = [str(package_dir)]
    package.__package__ = "raydi"
    package.raydi = ext_module

    for name, value in ext_module.__dict__.items():
        if name.startswith("__") and name not in {"__doc__", "__name__"}:
            continue
        setattr(package, name, value)

    sys.modules["raydi"] = package
    return package


def collect_baseline_data():
    pj = _load_raydi()
    import drjit as dr
    import drjit.cuda as cuda
    import drjit.cuda.ad as ad

    geometry = {}

    mesh = pj.Mesh(
        cuda.Array3f([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]),
        cuda.Array3i([0], [1], [2]),
    )
    scene = pj.Scene()
    scene.add_mesh(mesh)
    scene.configure()
    hit_ray = pj.RayDetached(cuda.Array3f([0.25], [0.25], [-1.0]), cuda.Array3f([0.0], [0.0], [1.0]))
    its = scene.intersect(hit_ray)
    geometry["constant_hit"] = {
        "valid": _bool_lane(its.is_valid()),
        "shape_id": _int_lane(its.shape_id),
        "prim_id": _int_lane(its.prim_id),
        "t": _float_lane(its.t),
        "p": _vector_to_list(its.p, 3),
        "n": _vector_to_list(its.n, 3),
        "geo_n": _vector_to_list(its.geo_n, 3),
        "uv": _vector_to_list(its.uv, 2),
        "barycentric": _vector_to_list(its.barycentric, 3),
    }

    miss_ray = pj.RayDetached(cuda.Array3f([2.0], [2.0], [-1.0]), cuda.Array3f([0.0], [0.0], [1.0]))
    its = scene.intersect(miss_ray)
    geometry["miss"] = {
        "valid": _bool_lane(its.is_valid()),
        "shape_id": _int_lane(its.shape_id),
        "prim_id": _int_lane(its.prim_id),
        "t_is_inf": abs(_float_lane(its.t)) == float("inf"),
    }

    mesh_a = pj.Mesh(
        cuda.Array3f([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]),
        cuda.Array3i([0], [1], [2]),
    )
    mesh_b = pj.Mesh(
        cuda.Array3f([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]),
        cuda.Array3i([0], [1], [2]),
    )
    mesh_b.vertex_positions = ad.Array3f([2.0, 3.0, 2.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0])
    multi_scene = pj.Scene()
    multi_scene.add_mesh(mesh_a)
    multi_scene.add_mesh(mesh_b)
    multi_scene.configure()
    its = multi_scene.intersect(
        pj.RayDetached(cuda.Array3f([2.25], [0.25], [-1.0]), cuda.Array3f([0.0], [0.0], [1.0]))
    )
    geometry["multi_mesh"] = {
        "valid": _bool_lane(its.is_valid()),
        "shape_id": _int_lane(its.shape_id),
        "prim_id": _int_lane(its.prim_id),
        "t": _float_lane(its.t),
        "p": _vector_to_list(its.p, 3),
    }

    uv_ray = pj.RayDetached(cuda.Array3f([0.25], [0.25], [-1.0]), cuda.Array3f([0.0], [0.0], [1.0]))
    mesh_no_uv = pj.Mesh(
        cuda.Array3f([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]),
        cuda.Array3i([0], [1], [2]),
    )
    scene_no_uv = pj.Scene()
    scene_no_uv.add_mesh(mesh_no_uv)
    scene_no_uv.configure()
    its_no_uv = scene_no_uv.intersect(uv_ray)
    mesh_with_uv = pj.Mesh(
        cuda.Array3f([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]),
        cuda.Array3i([0], [1], [2]),
        cuda.Array2f([0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
    )
    scene_with_uv = pj.Scene()
    scene_with_uv.add_mesh(mesh_with_uv)
    scene_with_uv.configure()
    its_with_uv = scene_with_uv.intersect(uv_ray)
    geometry["uv"] = {
        "missing": _vector_to_list(its_no_uv.uv, 2),
        "present": _vector_to_list(its_with_uv.uv, 2),
    }

    xs, ys, zs = [], [], []
    for iy in range(12):
        for ix in range(12):
            xs.append(-0.2 + ix * 0.12)
            ys.append(-0.2 + iy * 0.12)
            zs.append(-1.0)
    square_mesh = pj.Mesh(
        cuda.Array3f([0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]),
        cuda.Array3i([0, 0], [1, 2], [2, 3]),
    )
    batched_scene = pj.Scene()
    batched_scene.add_mesh(square_mesh)
    batched_scene.configure()
    rays = pj.RayDetached(cuda.Array3f(xs, ys, zs), cuda.Array3f([0.0] * len(xs), [0.0] * len(xs), [1.0] * len(xs)))
    its = batched_scene.intersect(rays)
    valid_flags = _bool_array_to_list(its.is_valid())
    geometry["batched_hits"] = {
        "num_rays": len(xs),
        "num_hits": sum(valid_flags),
        "valid_prefix": valid_flags[:16],
        "shape_prefix": _int_array_to_list(its.shape_id)[:16],
        "prim_prefix": _int_array_to_list(its.prim_id)[:16],
        "t_prefix": _float_array_to_list(its.t)[:16],
    }

    degenerate_mesh = pj.Mesh(
        cuda.Array3f([0.0, 1.0, 2.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        cuda.Array3i([0], [1], [2]),
    )
    degenerate_mesh.configure()
    secondary_edges = degenerate_mesh.secondary_edges()
    degenerate_scene = pj.Scene()
    degenerate_scene.add_mesh(degenerate_mesh)
    degenerate_scene.configure()
    its = degenerate_scene.intersect(
        pj.RayDetached(cuda.Array3f([0.5], [0.0], [-1.0]), cuda.Array3f([0.0], [0.0], [1.0]))
    )
    geometry["degenerate"] = {
        "valid": _bool_lane(its.is_valid()),
        "shape_id": _int_lane(its.shape_id),
        "prim_id": _int_lane(its.prim_id),
        "t_is_inf": abs(_float_lane(its.t)) == float("inf"),
        "edge_count": secondary_edges.size(),
        "boundary_mask": _bool_array_to_list(secondary_edges.is_boundary),
        "start": _vector_array_to_rows(secondary_edges.start, 3),
    }

    gradients = {}

    grad_mesh = pj.Mesh(
        cuda.Array3f([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]),
        cuda.Array3i([0], [1], [2]),
    )
    verts = ad.Array3f([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0])
    dr.enable_grad(verts)
    grad_mesh.vertex_positions = verts
    grad_scene = pj.Scene()
    grad_scene.add_mesh(grad_mesh)
    grad_scene.configure()
    ray = pj.Ray(ad.Array3f([0.25], [0.25], [-1.0]), ad.Array3f([0.0], [0.0], [1.0]))
    its = grad_scene.intersect(ray)
    dr.backward(its.t)
    grad = dr.grad(verts)
    gradients["vertex_gradients"] = {
        "valid": _bool_lane(its.is_valid()),
        "gradient_vectors": _vector_array_to_rows(grad, 3),
        "t": _float_lane(its.t),
    }

    transform_mesh = pj.Mesh(
        cuda.Array3f([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]),
        cuda.Array3i([0], [1], [2]),
    )
    tz = ad.Float([0.0])
    dr.enable_grad(tz)
    transform_mesh.to_world_left = ad.Matrix4f(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, tz],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    transform_scene = pj.Scene()
    transform_scene.add_mesh(transform_mesh)
    transform_scene.configure()
    ray = pj.Ray(ad.Array3f([0.25], [0.25], [-1.0]), ad.Array3f([0.0], [0.0], [1.0]))
    its = transform_scene.intersect(ray)
    dr.backward(its.t)
    gradients["transform_gradients"] = {
        "valid": _bool_lane(its.is_valid()),
        "grad_tz": _float_lane(dr.grad(tz)),
        "t": _float_lane(its.t),
        "matrix_rows": _matrix_rows(pj, transform_mesh.to_world_left),
    }

    identity = cuda.Matrix4f(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    translation = cuda.Matrix4f(
        [
            [1.0, 0.0, 0.0, 2.0],
            [0.0, 1.0, 0.0, -3.0],
            [0.0, 0.0, 1.0, 4.5],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    scale = cuda.Matrix4f(
        [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    rotation_z = cuda.Matrix4f(
        [
            [0.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    look_at = pj._test_look_at(
        cuda.Array3f([1.0], [2.0], [5.0]),
        cuda.Array3f([0.0], [0.0], [0.0]),
        cuda.Array3f([0.0], [1.0], [0.0]),
    )

    transform_cases = {
        "identity": _transform_case(pj, cuda, identity),
        "translation": _transform_case(pj, cuda, translation),
        "scale": _transform_case(pj, cuda, scale),
        "rotation_z_90": _transform_case(pj, cuda, rotation_z),
        "look_at": _transform_case(pj, cuda, look_at),
    }

    transforms = {
        "matrix_entry_checks": {name: case["matrix"] for name, case in transform_cases.items()},
        "point_transform_checks": {name: case["points"] for name, case in transform_cases.items()},
        "direction_transform_checks": {name: case["directions"] for name, case in transform_cases.items()},
    }

    left = cuda.Matrix4f(
        [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.5],
            [0.0, 0.0, 1.0, -2.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    raw = rotation_z
    right = scale
    composed = left @ raw @ right
    composition_case = _transform_case(pj, cuda, composed)
    transforms["composition_left_right"] = {
        "left": _matrix_rows(pj, left),
        "raw": _matrix_rows(pj, raw),
        "right": _matrix_rows(pj, right),
        "composed": composition_case["matrix"],
        "points": composition_case["points"],
        "directions": composition_case["directions"],
    }

    camera = {}

    fov_camera = pj.Camera(60.0, 1e-4, 1e3)
    fov_camera.width = 64
    fov_camera.height = 32
    fov_camera.configure(cache=False)
    camera["perspective_fov"] = {
        "helper_matrix": _matrix_rows(pj, pj._test_perspective(60.0, 1e-4, 1e3)),
        "camera_to_sample": _matrix_rows(pj, fov_camera.camera_to_sample),
        "sample_to_camera": _matrix_rows(pj, fov_camera.sample_to_camera),
    }

    intrinsic_camera = pj.Camera(0.75, 0.9, 0.45, 0.55, 1e-4, 1e3)
    intrinsic_camera.width = 80
    intrinsic_camera.height = 60
    intrinsic_camera.configure(cache=False)
    camera["perspective_intrinsic"] = {
        "helper_matrix": _matrix_rows(pj, pj._test_perspective_intrinsic(0.75, 0.9, 0.45, 0.55, 1e-4, 1e3)),
        "camera_to_sample": _matrix_rows(pj, intrinsic_camera.camera_to_sample),
        "sample_to_camera": _matrix_rows(pj, intrinsic_camera.sample_to_camera),
    }

    samples = [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0), (0.25, 0.75)]
    camera["sample_ray"] = {
        "default_camera": [
            {
                "sample": list(sample),
                "ray": _ray_to_dict(fov_camera.sample_ray(_vec2(sample, cuda.Array2f))),
            }
            for sample in samples
        ]
    }

    transformed_camera = pj.Camera(45.0, 1e-4, 1e4)
    transformed_camera.width = 40
    transformed_camera.height = 20
    transformed_camera.to_world_left = ad.Matrix4f(
        [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, -2.0],
            [0.0, 0.0, 1.0, 0.5],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    transformed_camera.to_world = ad.Matrix4f(
        [
            [1.0, 0.0, 0.0, 0.25],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    transformed_camera.to_world_right = ad.Matrix4f(
        [
            [0.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    transformed_camera.configure(cache=False)
    camera["world_to_sample"] = {
        "matrix": _matrix_rows(pj, transformed_camera.world_to_sample),
        "probe_points": [
            {
                "input": list(point),
                "output": _vector_to_list(
                    pj._test_transform_pos(transformed_camera.world_to_sample, _vec(point, ad.Array3f)),
                    3,
                ),
            }
            for point in PROBE_POINTS
        ],
    }
    camera["sample_to_world"] = {
        "matrix": _matrix_rows(pj, transformed_camera.sample_to_world),
        "probe_points": [
            {
                "input": list(point),
                "output": _vector_to_list(
                    pj._test_transform_pos(transformed_camera.sample_to_world, _vec(point, ad.Array3f)),
                    3,
                ),
            }
            for point in PROBE_POINTS
        ],
    }
    camera["sample_ray"]["transformed_camera"] = [
        {
            "sample": list(sample),
            "ray": _ray_to_dict(transformed_camera.sample_ray(_vec2(sample, cuda.Array2f))),
        }
        for sample in samples
    ]

    edge_mesh = pj.Mesh(
        cuda.Array3f([0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]),
        cuda.Array3i([0, 0], [1, 2], [2, 3]),
    )
    edge_mesh.configure()
    edge_indices = edge_mesh.edge_indices()
    secondary_edges = edge_mesh.secondary_edges()

    front_mesh = pj.Mesh(
        cuda.Array3f([-0.5, 0.5, 0.0], [-0.5, -0.5, 0.5], [3.0, 3.0, 3.0]),
        cuda.Array3i([0], [1], [2]),
    )
    edge_scene = pj.Scene()
    edge_scene.add_mesh(front_mesh)
    edge_scene.configure()

    edge_camera = pj.Camera(45.0, 1e-4, 1e4)
    edge_camera.width = 32
    edge_camera.height = 32
    edge_camera.configure(cache=False)
    edge_camera.prepare_edges(edge_scene)
    edge_sample = edge_camera.sample_edge(cuda.Float([0.25]))

    edges = {
        "secondary_edges": {
            "edge_indices": [_int_array_to_list(edge_index) for edge_index in edge_indices],
            "edge_count": secondary_edges.size(),
            "boundary_mask": _bool_array_to_list(secondary_edges.is_boundary),
            "start": _vector_array_to_rows(secondary_edges.start, 3),
            "edge": _vector_array_to_rows(secondary_edges.edge, 3),
            "normal0": _vector_array_to_rows(secondary_edges.normal0, 3),
            "normal1": _vector_array_to_rows(secondary_edges.normal1, 3),
            "opposite": _vector_array_to_rows(secondary_edges.opposite, 3),
        },
        "primary_edge_sampling": {
            "idx": _int_lane(edge_sample.idx),
            "pdf": _float_lane(edge_sample.pdf),
            "x_dot_n": _float_lane(edge_sample.x_dot_n),
            "ray_p": _ray_to_dict(edge_sample.ray_p),
            "ray_n": _ray_to_dict(edge_sample.ray_n),
        },
    }

    total_hits = 0
    total_samples = 0
    max_abs_grad = 0.0
    for _ in range(20):
        loop_mesh = pj.Mesh(
            cuda.Array3f([0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]),
            cuda.Array3i([0, 0], [1, 2], [2, 3]),
        )
        loop_scene = pj.Scene()
        loop_scene.add_mesh(loop_mesh)
        loop_scene.configure()

        loop_camera = pj.Camera(45.0, 1e-4, 1e4)
        loop_camera.width = 32
        loop_camera.height = 32
        loop_camera.configure()
        loop_camera.prepare_edges(loop_scene)
        sample = loop_camera.sample_edge(cuda.Float([0.25]))
        total_samples += int(_int_lane(sample.idx) >= 0)

        xs = [0.1 + 0.02 * (i % 8) for i in range(64)]
        ys = [0.1 + 0.02 * (i // 8) for i in range(64)]
        rays = pj.RayDetached(
            cuda.Array3f(xs, ys, [-1.0] * 64),
            cuda.Array3f([0.0] * 64, [0.0] * 64, [1.0] * 64),
        )
        its = loop_scene.intersect(rays)
        total_hits += sum(_bool_array_to_list(its.is_valid()))

        grad_mesh = pj.Mesh(
            cuda.Array3f([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]),
            cuda.Array3i([0], [1], [2]),
        )
        tz = ad.Float([0.0])
        dr.enable_grad(tz)
        grad_mesh.to_world_left = ad.Matrix4f(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, tz],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        grad_scene = pj.Scene()
        grad_scene.add_mesh(grad_mesh)
        grad_scene.configure()
        hit = grad_scene.intersect(
            pj.Ray(ad.Array3f([0.25], [0.25], [-1.0]), ad.Array3f([0.0], [0.0], [1.0]))
        )
        dr.backward(hit.t)
        max_abs_grad = max(max_abs_grad, abs(_float_lane(dr.grad(tz))))

    stress = {
        "repeated_run_summary": {
            "total_hits": total_hits,
            "total_samples": total_samples,
            "max_abs_grad": max_abs_grad,
        }
    }

    return {
        "geometry": geometry,
        "gradients": gradients,
        "transforms": transforms,
        "camera": camera,
        "edges": edges,
        "stress": stress,
    }


def collect_manifest():
    import drjit

    return {
        "baseline_version": "drjit_v0_4_6",
        "drjit_version": getattr(drjit, "__version__", None),
        "raydi_commit": _git_output("git", "rev-parse", "HEAD"),
        "drjit_commit": _optional_git_output(
            "git",
            "-C",
            str(Path(drjit.__file__).resolve().parent),
            "rev-parse",
            "HEAD",
        ),
        "python_version": sys.version,
        "platform": platform.platform(),
        "cuda_version": _nvcc_version(),
        "tolerance_policy": DEFAULT_TOLERANCE_POLICY,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--out", type=Path, default=ROOT / "tests" / "baselines" / "drjit_v0_4_6")
    args = parser.parse_args()

    manifest = collect_manifest()
    data = collect_baseline_data()
    if args.write:
        write_baseline_tree(manifest, data, args.out)
    else:
        import json

        print(json.dumps({"manifest": manifest, "data": data}, sort_keys=True))


if __name__ == "__main__":
    main()



