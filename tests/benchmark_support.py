import argparse
import gc
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Any

import raydi as pj
import drjit as dr
import drjit.cuda as cuda
import drjit.cuda.ad as ad


def _try_import_mitsuba(variant: str):
    try:
        import mitsuba as mi  # type: ignore
    except ImportError:
        return None

    mi.set_variant(variant)
    return mi


def _make_grid_mesh_data(resolution: int, x_offset: float = 0.0, z_offset: float = 0.0) -> dict[str, list[float] | list[int]]:
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for y in range(resolution + 1):
        fy = y / resolution
        for x in range(resolution + 1):
            fx = x / resolution
            xs.append(x_offset + fx)
            ys.append(fy)
            zs.append(z_offset)

    i0: list[int] = []
    i1: list[int] = []
    i2: list[int] = []
    stride = resolution + 1
    for y in range(resolution):
        for x in range(resolution):
            v00 = y * stride + x
            v10 = v00 + 1
            v01 = v00 + stride
            v11 = v01 + 1
            i0.extend([v00, v00])
            i1.extend([v10, v11])
            i2.extend([v11, v01])

    return {
        "x": xs,
        "y": ys,
        "z": zs,
        "i0": i0,
        "i1": i1,
        "i2": i2,
    }


def _make_ray_data(side: int, x_offset: float = 0.0, z_origin: float = -1.0) -> dict[str, list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for iy in range(side):
        for ix in range(side):
            xs.append(x_offset + (ix + 0.5) / side)
            ys.append((iy + 0.5) / side)
            zs.append(z_origin)
    return {
        "ox": xs,
        "oy": ys,
        "oz": zs,
        "dx": [0.0] * len(xs),
        "dy": [0.0] * len(xs),
        "dz": [1.0] * len(xs),
    }


def _flatten_vec3_soa(vec: Any) -> list[float]:
    xs = list(vec[0])
    ys = list(vec[1])
    zs = list(vec[2])
    return [value for xyz in zip(xs, ys, zs) for value in xyz]


def _flatten_vec2_soa(vec: Any) -> list[float]:
    xs = list(vec[0])
    ys = list(vec[1])
    return [value for uv in zip(xs, ys) for value in uv]


def _vector_max_abs_diff(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"Length mismatch: {len(a)} != {len(b)}")
    if not a:
        return 0.0
    return max(abs(x - y) for x, y in zip(a, b))


def _vector_mean_abs_diff(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"Length mismatch: {len(a)} != {len(b)}")
    if not a:
        return 0.0
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


def _summarize_timings(times_s: list[float], query_count: int) -> dict[str, float]:
    avg_s = statistics.fmean(times_s)
    min_s = min(times_s)
    return {
        "min_ms": min_s * 1000.0,
        "avg_ms": avg_s * 1000.0,
        "qps_m": query_count / avg_s / 1e6,
    }


def _measure(fn, repeats: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        fn()
        dr.sync_thread()

    times_s: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        dr.sync_thread()
        times_s.append(time.perf_counter() - start)
    return times_s


def _scalar_to_float(value: Any) -> float:
    if isinstance(value, (float, int)):
        return float(value)
    try:
        return float(value)
    except TypeError:
        return float(value[0])


def _cleanup_drjit() -> None:
    gc.collect()
    dr.sync_thread()
    dr.flush_malloc_cache()
    dr.flush_kernel_cache()
    dr.sync_thread()


@dataclass
class IntersectionSummary:
    valid: list[bool]
    t: list[float]
    p: list[float]
    bary_uv: list[float]
    prim_index: list[int]

    def stats(self) -> dict[str, Any]:
        return {
            "num_rays": len(self.valid),
            "num_hits": sum(1 for v in self.valid if v),
            "t_min": min(self.t) if self.t else math.inf,
            "t_max": max(self.t) if self.t else -math.inf,
        }


class RayDiBackend:
    name = "raydi"

    def environment(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "drjit_version": getattr(dr, "__version__", "unknown"),
                    }

    def _mesh(self, mesh_data: dict[str, list[float] | list[int]]) -> Any:
        mesh = pj.Mesh(
            cuda.Array3f(mesh_data["x"], mesh_data["y"], mesh_data["z"]),
            cuda.Array3i(mesh_data["i0"], mesh_data["i1"], mesh_data["i2"]),
        )
        return mesh

    def _scene(self, mesh_data: dict[str, list[float] | list[int]], dynamic: bool) -> tuple[Any, int]:
        mesh = self._mesh(mesh_data)
        scene = pj.Scene()
        mesh_id = scene.add_mesh(mesh, dynamic=dynamic)
        scene.configure()
        return scene, mesh_id

    def _ray_detached(self, ray_data: dict[str, list[float]]) -> Any:
        return pj.RayDetached(
            cuda.Array3f(ray_data["ox"], ray_data["oy"], ray_data["oz"]),
            cuda.Array3f(ray_data["dx"], ray_data["dy"], ray_data["dz"]),
        )

    def _ray_ad(self, ray_data: dict[str, list[float]]) -> Any:
        return pj.Ray(
            ad.Array3f(ray_data["ox"], ray_data["oy"], ray_data["oz"]),
            ad.Array3f(ray_data["dx"], ray_data["dy"], ray_data["dz"]),
        )

    def forward_correctness(
        self,
        mesh_data: dict[str, list[float] | list[int]],
        ray_data: dict[str, list[float]],
    ) -> IntersectionSummary:
        scene, _ = self._scene(mesh_data, dynamic=False)
        its = scene.intersect(self._ray_detached(ray_data))
        dr.eval(its.t, its.p, its.barycentric, its.prim_id)
        dr.sync_thread()
        return IntersectionSummary(
            valid=[bool(v) for v in list(its.is_valid())],
            t=[float(v) for v in list(its.t)],
            p=_flatten_vec3_soa(its.p),
            bary_uv=_flatten_vec2_soa((its.barycentric[1], its.barycentric[2])),
            prim_index=[int(v) for v in list(its.prim_id)],
        )

    def forward_performance(
        self,
        mesh_data: dict[str, list[float] | list[int]],
        ray_data: dict[str, list[float]],
        repeats: int,
        warmup: int,
    ) -> dict[str, Any]:
        scene, _ = self._scene(mesh_data, dynamic=False)
        rays = self._ray_detached(ray_data)

        def run():
            its = scene.intersect(rays)
            dr.eval(its.t)

        return _summarize_timings(_measure(run, repeats, warmup), len(ray_data["ox"]))

    def dynamic_forward_correctness(
        self,
        mesh_data: dict[str, list[float] | list[int]],
        updated_mesh_data: dict[str, list[float] | list[int]],
        updated_ray_data: dict[str, list[float]],
    ) -> IntersectionSummary:
        scene, mesh_id = self._scene(mesh_data, dynamic=True)
        scene.update_mesh_vertices(
            mesh_id,
            cuda.Array3f(updated_mesh_data["x"], updated_mesh_data["y"], updated_mesh_data["z"]),
        )
        scene.commit_updates()

        its = scene.intersect(self._ray_detached(updated_ray_data))
        dr.eval(its.t, its.p, its.barycentric, its.prim_id)
        dr.sync_thread()
        return IntersectionSummary(
            valid=[bool(v) for v in list(its.is_valid())],
            t=[float(v) for v in list(its.t)],
            p=_flatten_vec3_soa(its.p),
            bary_uv=_flatten_vec2_soa((its.barycentric[1], its.barycentric[2])),
            prim_index=[int(v) for v in list(its.prim_id)],
        )

    def dynamic_forward_performance(
        self,
        mesh_data: dict[str, list[float] | list[int]],
        updated_mesh_data: dict[str, list[float] | list[int]],
        ray_data: dict[str, list[float]],
        updated_ray_data: dict[str, list[float]],
        repeats: int,
        warmup: int,
    ) -> dict[str, Any]:
        scene, mesh_id = self._scene(mesh_data, dynamic=True)
        base_positions = cuda.Array3f(mesh_data["x"], mesh_data["y"], mesh_data["z"])
        updated_positions = cuda.Array3f(updated_mesh_data["x"], updated_mesh_data["y"], updated_mesh_data["z"])
        base_rays = self._ray_detached(ray_data)
        rays = self._ray_detached(updated_ray_data)
        use_updated = False

        def run():
            nonlocal use_updated
            use_updated = not use_updated
            current_positions = updated_positions if use_updated else base_positions
            current_rays = rays if use_updated else base_rays
            scene.update_mesh_vertices(mesh_id, current_positions)
            scene.commit_updates()
            its = scene.intersect(current_rays)
            dr.eval(its.t)

        return _summarize_timings(_measure(run, repeats, warmup), len(updated_ray_data["ox"]))

    def gradient_correctness(
        self,
        mesh_data: dict[str, list[float] | list[int]],
        ray_data: dict[str, list[float]],
        dynamic_update: bool,
        updated_mesh_data: dict[str, list[float] | list[int]] | None = None,
    ) -> dict[str, Any]:
        mesh = pj.Mesh(
            cuda.Array3f(mesh_data["x"], mesh_data["y"], mesh_data["z"]),
            cuda.Array3i(mesh_data["i0"], mesh_data["i1"], mesh_data["i2"]),
        )

        target_mesh_data = updated_mesh_data if dynamic_update and updated_mesh_data is not None else mesh_data
        verts = ad.Array3f(target_mesh_data["x"], target_mesh_data["y"], target_mesh_data["z"])
        dr.enable_grad(verts)
        if not dynamic_update:
            mesh.vertex_positions = verts

        scene = pj.Scene()
        if dynamic_update:
            mesh_id = scene.add_mesh(mesh, dynamic=True)
        else:
            mesh_id = scene.add_mesh(mesh, dynamic=False)
        scene.configure()

        if dynamic_update:
            scene.update_mesh_vertices(mesh_id, verts)
            scene.commit_updates()

        its = scene.intersect(self._ray_ad(ray_data))
        loss = dr.sum(its.t)
        dr.backward(loss)
        grad = dr.grad(verts)
        dr.eval(grad)
        dr.sync_thread()

        return {
            "loss": _scalar_to_float(loss),
            "grad": _flatten_vec3_soa(grad),
            "num_vertices": len(mesh_data["x"]),
        }

    def gradient_performance(
        self,
        mesh_data: dict[str, list[float] | list[int]],
        ray_data: dict[str, list[float]],
        repeats: int,
        warmup: int,
        dynamic_update: bool,
        updated_mesh_data: dict[str, list[float] | list[int]] | None = None,
    ) -> dict[str, Any]:
        mesh = pj.Mesh(
            cuda.Array3f(mesh_data["x"], mesh_data["y"], mesh_data["z"]),
            cuda.Array3i(mesh_data["i0"], mesh_data["i1"], mesh_data["i2"]),
        )

        target_mesh_data = updated_mesh_data if dynamic_update and updated_mesh_data is not None else mesh_data
        verts = ad.Array3f(target_mesh_data["x"], target_mesh_data["y"], target_mesh_data["z"])
        dr.enable_grad(verts)
        if not dynamic_update:
            mesh.vertex_positions = verts

        scene = pj.Scene()
        if dynamic_update:
            mesh_id = scene.add_mesh(mesh, dynamic=True)
        else:
            mesh_id = scene.add_mesh(mesh, dynamic=False)
        scene.configure()
        rays = self._ray_ad(ray_data)

        def run():
            dr.set_grad(verts, 0)
            if dynamic_update:
                scene.update_mesh_vertices(mesh_id, verts)
                scene.commit_updates()

            its = scene.intersect(rays)
            loss = dr.sum(its.t)
            dr.backward(loss)
            dr.eval(dr.grad(verts))

        return _summarize_timings(_measure(run, repeats, warmup), len(ray_data["ox"]))


class MitsubaBackend:
    name = "mitsuba"

    def __init__(self, mi: Any, variant: str):
        self.mi = mi
        self.variant = variant

    def environment(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "variant": self.variant,
            "version": getattr(self.mi, "__version__", "unknown"),
        }

    def _mesh(self, mesh_data: dict[str, list[float] | list[int]]) -> Any:
        mi = self.mi
        mesh = mi.Mesh(
            "plane",
            vertex_count=len(mesh_data["x"]),
            face_count=len(mesh_data["i0"]),
            has_vertex_normals=False,
            has_vertex_texcoords=False,
        )
        params = mi.traverse(mesh)
        params["vertex_positions"] = dr.ravel(mi.Point3f(mesh_data["x"], mesh_data["y"], mesh_data["z"]))
        params["faces"] = dr.ravel(mi.Vector3u(mesh_data["i0"], mesh_data["i1"], mesh_data["i2"]))
        params.update()
        return mesh

    def _scene(self, mesh_data: dict[str, list[float] | list[int]]) -> tuple[Any, Any]:
        mi = self.mi
        mesh = self._mesh(mesh_data)
        scene = mi.load_dict(
            {
                "type": "scene",
                "mesh": mesh,
            }
        )
        return scene, mi.traverse(scene)

    def _ray(self, ray_data: dict[str, list[float]], ad_mode: bool) -> Any:
        mi = self.mi
        point_type = mi.Point3f if not ad_mode else mi.Point3f
        vector_type = mi.Vector3f if not ad_mode else mi.Vector3f
        return mi.Ray3f(
            point_type(ray_data["ox"], ray_data["oy"], ray_data["oz"]),
            vector_type(ray_data["dx"], ray_data["dy"], ray_data["dz"]),
        )

    def forward_correctness(
        self,
        mesh_data: dict[str, list[float] | list[int]],
        ray_data: dict[str, list[float]],
    ) -> IntersectionSummary:
        scene, _ = self._scene(mesh_data)
        its = scene.intersect(self._ray(ray_data, ad_mode=False))
        dr.eval(its.t, its.p, its.uv, its.prim_index)
        dr.sync_thread()
        return IntersectionSummary(
            valid=[bool(v) for v in list(its.is_valid())],
            t=[float(v) for v in list(its.t)],
            p=_flatten_vec3_soa(its.p),
            bary_uv=_flatten_vec2_soa(its.uv),
            prim_index=[int(v) for v in list(its.prim_index)],
        )

    def forward_performance(
        self,
        mesh_data: dict[str, list[float] | list[int]],
        ray_data: dict[str, list[float]],
        repeats: int,
        warmup: int,
    ) -> dict[str, Any]:
        scene, _ = self._scene(mesh_data)
        rays = self._ray(ray_data, ad_mode=False)

        def run():
            its = scene.intersect(rays)
            dr.eval(its.t)

        return _summarize_timings(_measure(run, repeats, warmup), len(ray_data["ox"]))

    def dynamic_forward_correctness(
        self,
        mesh_data: dict[str, list[float] | list[int]],
        updated_mesh_data: dict[str, list[float] | list[int]],
        updated_ray_data: dict[str, list[float]],
    ) -> IntersectionSummary:
        scene, params = self._scene(mesh_data)
        params["mesh.vertex_positions"] = dr.ravel(
            self.mi.Point3f(updated_mesh_data["x"], updated_mesh_data["y"], updated_mesh_data["z"])
        )
        params.update()

        its = scene.intersect(self._ray(updated_ray_data, ad_mode=False))
        dr.eval(its.t, its.p, its.uv, its.prim_index)
        dr.sync_thread()
        return IntersectionSummary(
            valid=[bool(v) for v in list(its.is_valid())],
            t=[float(v) for v in list(its.t)],
            p=_flatten_vec3_soa(its.p),
            bary_uv=_flatten_vec2_soa(its.uv),
            prim_index=[int(v) for v in list(its.prim_index)],
        )

    def dynamic_forward_performance(
        self,
        mesh_data: dict[str, list[float] | list[int]],
        updated_mesh_data: dict[str, list[float] | list[int]],
        ray_data: dict[str, list[float]],
        updated_ray_data: dict[str, list[float]],
        repeats: int,
        warmup: int,
    ) -> dict[str, Any]:
        scene, params = self._scene(mesh_data)
        base_positions = dr.ravel(
            self.mi.Point3f(mesh_data["x"], mesh_data["y"], mesh_data["z"])
        )
        updated_positions = dr.ravel(
            self.mi.Point3f(updated_mesh_data["x"], updated_mesh_data["y"], updated_mesh_data["z"])
        )
        base_rays = self._ray(ray_data, ad_mode=False)
        rays = self._ray(updated_ray_data, ad_mode=False)
        use_updated = False

        def run():
            nonlocal use_updated
            use_updated = not use_updated
            params["mesh.vertex_positions"] = updated_positions if use_updated else base_positions
            params.update()
            its = scene.intersect(rays if use_updated else base_rays)
            dr.eval(its.t)

        return _summarize_timings(_measure(run, repeats, warmup), len(updated_ray_data["ox"]))

    def gradient_correctness(
        self,
        mesh_data: dict[str, list[float] | list[int]],
        ray_data: dict[str, list[float]],
        dynamic_update: bool,
        updated_mesh_data: dict[str, list[float] | list[int]] | None = None,
    ) -> dict[str, Any]:
        scene, params = self._scene(mesh_data)
        target_mesh_data = updated_mesh_data if dynamic_update and updated_mesh_data is not None else mesh_data
        verts = dr.ravel(self.mi.Point3f(target_mesh_data["x"], target_mesh_data["y"], target_mesh_data["z"]))
        dr.enable_grad(verts)
        params["mesh.vertex_positions"] = verts
        params.update()

        its = scene.intersect(self._ray(ray_data, ad_mode=True))
        loss = dr.sum(its.t)
        dr.backward(loss)
        grad = dr.grad(verts)
        dr.eval(grad)
        dr.sync_thread()

        return {
            "loss": _scalar_to_float(loss),
            "grad": [float(v) for v in list(grad)],
            "num_vertices": len(mesh_data["x"]),
            "dynamic_update": dynamic_update,
        }

    def gradient_performance(
        self,
        mesh_data: dict[str, list[float] | list[int]],
        ray_data: dict[str, list[float]],
        repeats: int,
        warmup: int,
        dynamic_update: bool,
        updated_mesh_data: dict[str, list[float] | list[int]] | None = None,
    ) -> dict[str, Any]:
        # Keep scene/params/rays outside the timed loop to match the RayDi benchmark.
        # For dynamic updates, explicitly mark the reused parameter dirty so Mitsuba
        # refreshes internal state even when the same AD buffer is submitted repeatedly.
        scene, params = self._scene(mesh_data)
        target_mesh_data = updated_mesh_data if dynamic_update and updated_mesh_data is not None else mesh_data
        verts = dr.ravel(self.mi.Point3f(target_mesh_data["x"], target_mesh_data["y"], target_mesh_data["z"]))
        dr.enable_grad(verts)
        rays = self._ray(ray_data, ad_mode=True)

        if not dynamic_update:
            params["mesh.vertex_positions"] = verts
            params.set_dirty("mesh.vertex_positions")
            params.update()

        def run():
            dr.set_grad(verts, 0)
            if dynamic_update:
                params["mesh.vertex_positions"] = verts
                params.set_dirty("mesh.vertex_positions")
                params.update()
            its = scene.intersect(rays)
            loss = dr.sum(its.t)
            dr.backward(loss)
            dr.eval(dr.grad(verts))

        return _summarize_timings(_measure(run, repeats, warmup), len(ray_data["ox"]))


def _compare_intersections(a: IntersectionSummary, b: IntersectionSummary) -> dict[str, Any]:
    valid_mismatch = sum(1 for av, bv in zip(a.valid, b.valid) if av != bv)
    prim_mismatch = sum(1 for ap, bp in zip(a.prim_index, b.prim_index) if ap != bp)
    return {
        "valid_mismatch_count": valid_mismatch,
        "prim_index_mismatch_count": prim_mismatch,
        "max_abs_t": _vector_max_abs_diff(a.t, b.t),
        "mean_abs_t": _vector_mean_abs_diff(a.t, b.t),
        "max_abs_p": _vector_max_abs_diff(a.p, b.p),
        "mean_abs_p": _vector_mean_abs_diff(a.p, b.p),
        "max_abs_bary_uv": _vector_max_abs_diff(a.bary_uv, b.bary_uv),
        "mean_abs_bary_uv": _vector_mean_abs_diff(a.bary_uv, b.bary_uv),
    }


def _compare_gradients(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    grad_a = a["grad"]
    grad_b = b["grad"]
    return {
        "loss_abs_diff": abs(a["loss"] - b["loss"]),
        "max_abs_grad_diff": _vector_max_abs_diff(grad_a, grad_b),
        "mean_abs_grad_diff": _vector_mean_abs_diff(grad_a, grad_b),
    }


def _build_backends(names: list[str], mitsuba_variant: str) -> list[Any]:
    backends: list[Any] = []
    for name in names:
        if name == "raydi":
            backends.append(RayDiBackend())
        elif name == "mitsuba":
            mi = _try_import_mitsuba(mitsuba_variant)
            if mi is None:
                raise RuntimeError(
                    "Mitsuba is not installed in the current environment. "
                    "Install it first, e.g. `python -m pip install mitsuba`, "
                    "then rerun this benchmark."
                )
            backends.append(MitsubaBackend(mi, mitsuba_variant))
        else:
            raise ValueError(f"Unsupported backend: {name}")
    return backends


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark RayDi ray intersection and gradients against Mitsuba."
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["raydi", "mitsuba"],
        choices=["raydi", "mitsuba"],
        help="Backends to run.",
    )
    parser.add_argument("--mitsuba-variant", default="cuda_ad_rgb")
    parser.add_argument("--mesh-resolution", type=int, default=64)
    parser.add_argument("--ray-grid-side", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--dynamic-x-offset", type=float, default=2.0)
    args = parser.parse_args()

    base_mesh = _make_grid_mesh_data(args.mesh_resolution)
    updated_mesh = _make_grid_mesh_data(args.mesh_resolution, x_offset=args.dynamic_x_offset)
    base_rays = _make_ray_data(args.ray_grid_side)
    updated_rays = _make_ray_data(args.ray_grid_side, x_offset=args.dynamic_x_offset)

    backends = _build_backends(args.backends, args.mitsuba_variant)

    results: dict[str, Any] = {
        "config": {
            "mesh_resolution": args.mesh_resolution,
            "triangle_count": len(base_mesh["i0"]),
            "vertex_count": len(base_mesh["x"]),
            "ray_count": len(base_rays["ox"]),
            "repeats": args.repeats,
            "warmup": args.warmup,
            "dynamic_x_offset": args.dynamic_x_offset,
        },
        "environment": {backend.name: backend.environment() for backend in backends},
        "backends": {},
        "comparisons": {},
    }

    forward_static: dict[str, IntersectionSummary] = {}
    forward_dynamic: dict[str, IntersectionSummary] = {}
    grad_static: dict[str, dict[str, Any]] = {}
    grad_dynamic: dict[str, dict[str, Any]] = {}

    for backend in backends:
        backend_result = {
            "forward_static_correctness": None,
            "forward_static_performance": None,
            "forward_dynamic_correctness": None,
            "forward_dynamic_performance": None,
            "gradient_static_correctness": None,
            "gradient_static_performance": None,
            "gradient_dynamic_correctness": None,
            "gradient_dynamic_performance": None,
        }

        static_its = backend.forward_correctness(base_mesh, base_rays)
        _cleanup_drjit()
        dynamic_its = backend.dynamic_forward_correctness(base_mesh, updated_mesh, updated_rays)
        _cleanup_drjit()
        static_grad = backend.gradient_correctness(base_mesh, base_rays, dynamic_update=False)
        _cleanup_drjit()
        dynamic_grad = backend.gradient_correctness(
            base_mesh,
            updated_rays,
            dynamic_update=True,
            updated_mesh_data=updated_mesh,
        )
        _cleanup_drjit()

        forward_static[backend.name] = static_its
        forward_dynamic[backend.name] = dynamic_its
        grad_static[backend.name] = static_grad
        grad_dynamic[backend.name] = dynamic_grad

        backend_result["forward_static_correctness"] = static_its.stats()
        backend_result["forward_static_performance"] = backend.forward_performance(
            base_mesh, base_rays, args.repeats, args.warmup
        )
        _cleanup_drjit()
        backend_result["forward_dynamic_correctness"] = dynamic_its.stats()
        backend_result["forward_dynamic_performance"] = backend.dynamic_forward_performance(
            base_mesh, updated_mesh, base_rays, updated_rays, args.repeats, args.warmup
        )
        _cleanup_drjit()
        backend_result["gradient_static_correctness"] = {
            "loss": static_grad["loss"],
            "grad_abs_max": max(abs(v) for v in static_grad["grad"]) if static_grad["grad"] else 0.0,
            "num_vertices": static_grad["num_vertices"],
        }
        backend_result["gradient_static_performance"] = backend.gradient_performance(
            base_mesh, base_rays, args.repeats, args.warmup, dynamic_update=False
        )
        _cleanup_drjit()
        backend_result["gradient_dynamic_correctness"] = {
            "loss": dynamic_grad["loss"],
            "grad_abs_max": max(abs(v) for v in dynamic_grad["grad"]) if dynamic_grad["grad"] else 0.0,
            "num_vertices": dynamic_grad["num_vertices"],
        }
        backend_result["gradient_dynamic_performance"] = backend.gradient_performance(
            base_mesh,
            updated_rays,
            args.repeats,
            args.warmup,
            dynamic_update=True,
            updated_mesh_data=updated_mesh,
        )
        _cleanup_drjit()
        results["backends"][backend.name] = backend_result

    if "raydi" in forward_static and "mitsuba" in forward_static:
        results["comparisons"]["forward_static"] = _compare_intersections(
            forward_static["raydi"], forward_static["mitsuba"]
        )
        results["comparisons"]["forward_dynamic"] = _compare_intersections(
            forward_dynamic["raydi"], forward_dynamic["mitsuba"]
        )
        results["comparisons"]["gradient_static"] = _compare_gradients(
            grad_static["raydi"], grad_static["mitsuba"]
        )
        results["comparisons"]["gradient_dynamic"] = _compare_gradients(
            grad_dynamic["raydi"], grad_dynamic["mitsuba"]
        )

    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())



