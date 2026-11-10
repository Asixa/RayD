# Copyright Xingyu Chen.
# Builds identical meshes, rays and points for the Torch and Dr.Jit backends from host-side row lists.

"""One host-side description per test scene, materialised in either backend.

Every geometry is a list of `MeshSpec` rows (vertices as `[x, y, z]`, faces as `[a, b, c]`), so the Torch and
Dr.Jit scenes of one parity case are built from the same Python floats and differ only in the backend. The
Dr.Jit converters take the `SimpleNamespace` returned by `tests.support.backends.drjit_modules()`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import numpy as np


try:
    import torch
except ImportError:  # A Dr.Jit-only environment; the parity tests skip before building anything.
    torch = None


Rows = list[list[float]]


@dataclass(frozen=True)
class MeshSpec:
    vertices: Rows
    faces: list[list[int]]
    uv: Rows | None = None
    dynamic: bool = False


@dataclass(frozen=True)
class SceneSpec:
    name: str
    meshes: tuple[MeshSpec, ...]


def _rows(tensor: torch.Tensor) -> list:
    return tensor.detach().cpu().tolist()


def single_triangle() -> SceneSpec:
    return SceneSpec("single_triangle", (MeshSpec([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [[0, 1, 2]]),))


def closed_box(half: float = 1.0) -> SceneSpec:
    from tests.support.geometry import box_mesh

    vertices, faces = box_mesh(torch.device("cpu"), half)
    return SceneSpec("closed_box", (MeshSpec(_rows(vertices), _rows(faces)),))


def grid_sheet(cells: int = 6, span: float = 2.0) -> SceneSpec:
    from tests.support.geometry import grid_mesh

    vertices, faces = grid_mesh(torch.device("cpu"), cells, span)
    return SceneSpec("grid_sheet", (MeshSpec(_rows(vertices), _rows(faces)),))


def overlapping_meshes() -> SceneSpec:
    """Two quads at z = 0 and z = 0.5 whose xy footprints overlap on [0, 1] x [0, 1]."""
    quad = [[0, 1, 2], [0, 2, 3]]
    lower = MeshSpec([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]], quad)
    upper = MeshSpec([[0.0, 0.0, 0.5], [2.0, 0.0, 0.5], [2.0, 2.0, 0.5], [0.0, 2.0, 0.5]], quad)
    return SceneSpec("overlapping_meshes", (lower, upper))


def non_manifold_fan() -> SceneSpec:
    """Three fins sharing the edge (0, 0, 0)-(1, 0, 0) at 0, 120 and 240 degrees around the x axis."""
    vertices = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    faces = []
    for index, angle in enumerate((0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0)):
        vertices.append([0.5, math.cos(angle), math.sin(angle)])
        faces.append([0, 1, 2 + index])
    return SceneSpec("non_manifold_fan", (MeshSpec(vertices, faces),))


def degenerate_faces() -> SceneSpec:
    """A unit quad plus repeated-index, collinear and duplicated faces that must not change any query."""
    vertices = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [2.0, 0.0, 0.0]]
    faces = [[0, 1, 2], [0, 2, 3], [0, 0, 0], [1, 1, 2], [0, 1, 4], [3, 3, 3]]
    return SceneSpec("degenerate_faces", (MeshSpec(vertices, faces),))


def all_scenes() -> tuple[SceneSpec, ...]:
    return (single_triangle(), closed_box(), grid_sheet(), overlapping_meshes(), non_manifold_fan(), degenerate_faces())


def flat_vertices(spec: SceneSpec) -> Rows:
    """Scene-global vertex rows in mesh order, matching both backends' global vertex numbering."""
    return [row for mesh in spec.meshes for row in mesh.vertices]


def bounds(spec: SceneSpec) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(flat_vertices(spec), dtype=np.float64)
    return vertices.min(axis=0), vertices.max(axis=0)


def scene_rays(spec: SceneSpec, count: int, seed: int) -> tuple[Rows, Rows]:
    """Rays from a shell well outside the scene aimed at points around its bounding box.

    Origins sit at least one unit outside the bounds, so every hit distance is far above both backends'
    `ray_tmin`, and about half of the rays miss.
    """
    low, high = bounds(spec)
    centre = 0.5 * (low + high)
    radius = 0.5 * float(np.linalg.norm(high - low)) + 1.0
    generator = np.random.default_rng(seed)
    origins = generator.normal(size=(count, 3))
    origins = centre + 1.5 * radius * origins / np.linalg.norm(origins, axis=1, keepdims=True)
    targets = low + (high - low) * generator.uniform(-0.1, 1.1, size=(count, 3))
    directions = targets - origins
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return origins.tolist(), directions.tolist()


def scene_points(spec: SceneSpec, count: int, seed: int, margin: float = 0.3) -> Rows:
    low, high = bounds(spec)
    generator = np.random.default_rng(seed)
    return (low - margin + (high - low + 2.0 * margin) * generator.uniform(size=(count, 3))).tolist()


def unit_vectors(count: int, seed: int) -> Rows:
    generator = np.random.default_rng(seed)
    values = generator.normal(size=(count, 3))
    return (values / np.linalg.norm(values, axis=1, keepdims=True)).tolist()


# Torch side.


def torch_rows(rows: Sequence[Sequence[float]], device: str = "cuda", dtype=None) -> torch.Tensor:
    width = len(rows[0]) if len(rows) else 3
    dtype = torch.float32 if dtype is None else dtype
    return torch.tensor(list(rows), device=device, dtype=dtype).reshape(len(rows), width).contiguous()


def torch_mesh(rt, mesh: MeshSpec, device: str = "cuda"):
    uv = None if mesh.uv is None else torch_rows(mesh.uv, device)
    return rt.Mesh(torch_rows(mesh.vertices, device), torch_rows(mesh.faces, device, torch.int32), uv=uv)


def torch_scene(rt, spec: SceneSpec, *, build: bool = True, device: str = "cuda", **scene_kwargs):
    scene = rt.Scene(**scene_kwargs)
    for mesh in spec.meshes:
        scene.add_mesh(torch_mesh(rt, mesh, device), dynamic=mesh.dynamic)
    if build:
        scene.build()
    return scene


def torch_ray(rt, origins: Rows, directions: Rows, tmax: Sequence[float] | None = None, device: str = "cuda"):
    o = torch_rows(origins, device)
    d = torch_rows(directions, device)
    if tmax is None:
        return rt.Ray(o, d)
    return rt.Ray(o, d, torch.tensor(list(tmax), device=device, dtype=torch.float32))


# Dr.Jit side.


def drjit_vec3(mods, rows: Sequence[Sequence[float]], ad: bool = False):
    array = mods.ad.Array3f if ad else mods.cuda.Array3f
    return array(*([float(row[axis]) for row in rows] for axis in range(3)))


def drjit_vec2(mods, rows: Sequence[Sequence[float]]):
    return mods.cuda.Array2f(*([float(row[axis]) for row in rows] for axis in range(2)))


def drjit_ivec3(mods, rows: Sequence[Sequence[int]]):
    return mods.cuda.Array3i(*([int(row[axis]) for row in rows] for axis in range(3)))


def drjit_mesh(mods, mesh: MeshSpec):
    if mesh.uv is None:
        return mods.pj.Mesh(drjit_vec3(mods, mesh.vertices), drjit_ivec3(mods, mesh.faces))
    return mods.pj.Mesh(drjit_vec3(mods, mesh.vertices), drjit_ivec3(mods, mesh.faces), drjit_vec2(mods, mesh.uv))


def drjit_scene(mods, spec: SceneSpec, *, build: bool = True, **scene_kwargs):
    scene = mods.pj.Scene(**scene_kwargs)
    for mesh in spec.meshes:
        scene.add_mesh(drjit_mesh(mods, mesh), dynamic=mesh.dynamic)
    if build:
        scene.build()
    return scene


def drjit_ray(mods, origins: Rows, directions: Rows, tmax: Sequence[float] | None = None):
    ray = mods.pj.Ray(drjit_vec3(mods, origins), drjit_vec3(mods, directions))
    if tmax is not None:
        ray.tmax = mods.cuda.Float([float(value) for value in tmax])
    return ray


# Conversions to host lists for comparison.


def host(value):
    """Return a Torch tensor or a Dr.Jit array as a NumPy array with one row per query.

    Dr.Jit vectors are stored component-major (`[3, N]`); they are transposed so both backends produce
    `[N, 3]` rows. Unsigned Dr.Jit ids are reinterpreted as int64 so `0xFFFFFFFF` stays distinguishable.
    """
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, (bool, int, float, np.ndarray)):
        return np.asarray(value)
    array = value.numpy()
    if array.ndim == 2:
        array = array.T
    if array.dtype == np.uint32:
        array = array.astype(np.int64)
    return np.ascontiguousarray(array)


class ParityRecorder:
    """Compares Torch and Dr.Jit outputs and records the largest deviation seen per operation.

    Floats must satisfy `|torch - drjit| <= atol + rtol * |drjit|` elementwise, and both backends must agree
    on which entries are non-finite (a miss distance of `inf` on one side and a number on the other is a
    mismatch, not a large deviation). Integers and booleans must be identical.
    """

    def __init__(self) -> None:
        self.max_deviation: dict[str, float] = {}

    def _note(self, operation: str, deviation: float) -> None:
        self.max_deviation[operation] = max(self.max_deviation.get(operation, 0.0), float(deviation))

    def close(
        self, testcase, operation: str, name: str, torch_value, drjit_value, atol: float, rtol: float, rows=None
    ) -> None:
        lhs = np.asarray(host(torch_value), dtype=np.float64)
        rhs = np.asarray(host(drjit_value), dtype=np.float64)
        testcase.assertEqual(lhs.size, rhs.size, f"{operation}.{name}: Torch {lhs.shape} vs Dr.Jit {rhs.shape}")
        lhs = lhs.reshape(rhs.shape) if lhs.shape != rhs.shape else lhs
        if rows is not None:
            lhs, rhs = lhs[rows], rhs[rows]
        finite_l, finite_r = np.isfinite(lhs), np.isfinite(rhs)
        mismatch = np.argwhere(finite_l != finite_r)
        testcase.assertEqual(
            mismatch.size,
            0,
            f"{operation}.{name}: finiteness differs at {mismatch[:4].tolist()} "
            f"(Torch {lhs[tuple(mismatch[0])] if mismatch.size else ''}, "
            f"Dr.Jit {rhs[tuple(mismatch[0])] if mismatch.size else ''})",
        )
        same_inf = ~finite_l & ~np.isnan(lhs)
        testcase.assertTrue(np.array_equal(lhs[same_inf], rhs[same_inf]), f"{operation}.{name}: infinities differ")
        testcase.assertFalse(np.isnan(lhs).any() or np.isnan(rhs).any(), f"{operation}.{name}: NaN in output")
        if not finite_l.any():
            self._note(operation, 0.0)
            return
        deviation = np.abs(lhs[finite_l] - rhs[finite_l])
        bound = atol + rtol * np.abs(rhs[finite_l])
        worst = int(np.argmax(deviation - bound))
        self._note(operation, float(deviation.max()))
        testcase.assertLessEqual(
            float(deviation[worst]),
            float(bound[worst]),
            f"{operation}.{name}: |Torch - Dr.Jit| = {deviation[worst]:.3e} > {bound[worst]:.3e} "
            f"(Torch {lhs[finite_l][worst]!r}, Dr.Jit {rhs[finite_l][worst]!r})",
        )

    def equal(self, testcase, operation: str, name: str, torch_value, drjit_value, rows=None) -> None:
        lhs = np.asarray(host(torch_value))
        rhs = np.asarray(host(drjit_value))
        testcase.assertEqual(lhs.size, rhs.size, f"{operation}.{name}: Torch {lhs.shape} vs Dr.Jit {rhs.shape}")
        lhs = lhs.reshape(rhs.shape) if lhs.shape != rhs.shape else lhs
        if rows is not None:
            lhs, rhs = lhs[rows], rhs[rows]
        different = np.argwhere(lhs != rhs)
        testcase.assertEqual(
            different.size,
            0,
            f"{operation}.{name}: {len(different)} entries differ, first at "
            f"{different[:1].tolist()} (Torch {lhs[tuple(different[0])] if different.size else ''}, "
            f"Dr.Jit {rhs[tuple(different[0])] if different.size else ''})",
        )
        self._note(operation, 0.0)

    def report(self) -> str:
        return "\n".join(f"{name:44s} {value:.3e}" for name, value in sorted(self.max_deviation.items()))
