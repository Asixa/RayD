"""Collect golden query results for the declarative scenes in ``scenes.py``.

``collect_golden(backend="drjit")`` builds every scene once and returns a
JSON-serializable dict keyed by scene name. Discrete outputs (hit/miss, ids,
counts, topology) are emitted as ints; continuous outputs (t, barycentric,
positions, normals, distances) as floats through the JSON repr round-trip.
Output ordering is deterministic (fixed query order, ``sort_keys`` on write).

Run standalone to regenerate the checked-in baselines::

    python -m tests.golden.runner --write

Only the drjit backend is wired today; ``backend`` is an explicit argument so
the torch backend can reuse the exact same scene definitions later.
"""

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

from tests.golden import scenes as scene_defs

ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = Path(__file__).resolve().parent / "baselines"

COMPARISON_POLICY = {
    "discrete": "exact bitwise equality (ints, bools-as-ints, strings)",
    "continuous": "contracts/operations.json tolerances "
    "(default_abs, default_rel)",
    "informative": "recorded but not compared (traversal-order or tie dependent)",
}


def _load_backend(backend):
    if backend != "drjit":
        raise ValueError(f"golden runner only supports backend='drjit', got {backend!r}")
    import drjit as dr
    import drjit.cuda as cuda
    import rayd.drjit as rd

    return dr, cuda, rd


def _ints(array):
    return [int(v) for v in list(array)]


def _bools_as_ints(array):
    return [int(bool(v)) for v in list(array)]


def _floats(array):
    return [float(v) for v in list(array)]


def _rows3(vec, width):
    return [[float(vec[0][i]), float(vec[1][i]), float(vec[2][i])] for i in range(width)]


def _rows2(vec, width):
    return [[float(vec[0][i]), float(vec[1][i])] for i in range(width)]


def _any_nan_scalar(array):
    return int(any(math.isnan(float(v)) for v in list(array)))


def _any_nan_vec(vec, width, comps):
    return int(any(math.isnan(float(vec[c][i])) for i in range(width) for c in range(comps)))


def _apply_informative(record, informative):
    if not informative:
        return record
    moved = {}
    if informative is True:
        names = list(record["discrete"].keys()) + list(record["continuous"].keys())
    else:
        names = list(informative)
    for name in names:
        if name in record["discrete"]:
            moved[name] = record["discrete"].pop(name)
        elif name in record["continuous"]:
            moved[name] = record["continuous"].pop(name)
    record["informative"] = moved
    return record


def _make_vec3(cuda, points):
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    zs = [float(p[2]) for p in points]
    return cuda.Array3f(xs, ys, zs)


def _make_ray(rd, cuda, query):
    ray = rd.Ray(_make_vec3(cuda, query["origins"]), _make_vec3(cuda, query["dirs"]))
    if "tmax" in query:
        ray.tmax = cuda.Float([float(v) for v in query["tmax"]])
    return ray


def _active(cuda, query):
    if "active" not in query:
        return None
    return cuda.Bool([bool(v) for v in query["active"]])


def _run_intersect(dr, cuda, rd, scene, query, offsets):
    if query.get("expect_raises"):
        try:
            ray = _make_ray(rd, cuda, query)
            its = scene.intersect(ray)
            dr.eval(its.t)
            width = int(dr.width(its.t))
        except Exception as exc:  # noqa: BLE001 - freezing the observed failure mode
            return {"kind": "intersect", "discrete": {"raises": 1, "exc_type": type(exc).__name__},
                    "continuous": {}}
        return {"kind": "intersect", "discrete": {"raises": 0, "width": width}, "continuous": {}}

    ray = _make_ray(rd, cuda, query)
    active = _active(cuda, query)
    its = scene.intersect(ray, active=active) if active is not None else scene.intersect(ray)
    width = int(dr.width(its.t))
    valid = _bools_as_ints(its.is_valid())
    shape_id = _ints(its.shape_id)
    local_prim_id = _ints(its.local_prim_id)
    global_prim_id = _ints(its.global_prim_id)
    discrete = {
        "valid": valid,
        "shape_id": shape_id,
        "prim_id": _ints(its.prim_id),
        "local_prim_id": local_prim_id,
        "global_prim_id": global_prim_id,
    }
    continuous = {
        "t": _floats(its.t),
        "p": _rows3(its.p, width),
        "n": _rows3(its.n, width),
        "geo_n": _rows3(its.geo_n, width),
        "uv": _rows2(its.uv, width),
        "bary": _rows3(its.barycentric, width),
    }
    for field in query.get("nan_flags", []):
        if field == "t":
            discrete["t_nan"] = _any_nan_scalar(its.t)
        elif field == "uv":
            discrete["uv_nan"] = _any_nan_vec(its.uv, width, 2)
        else:
            vec = {"p": its.p, "n": its.n, "geo_n": its.geo_n, "bary": its.barycentric}[field]
            discrete[f"{field}_nan"] = _any_nan_vec(vec, width, 3)
    if query.get("check_id_mapping"):
        ok = []
        for lane in range(width):
            if valid[lane]:
                ok.append(int(global_prim_id[lane] == offsets[shape_id[lane]] + local_prim_id[lane]))
            else:
                ok.append(int(global_prim_id[lane] == -1))
        discrete["id_mapping_ok"] = ok
    return _apply_informative(
        {"kind": "intersect", "discrete": discrete, "continuous": continuous},
        query.get("informative"),
    )


def _run_intersect_grid(dr, cuda, rd, scene, query):
    res = int(query["res"])
    x_min, x_max = float(query["x_min"]), float(query["x_max"])
    y_min, y_max = float(query["y_min"]), float(query["y_max"])
    z = float(query["z"])
    dir_z = float(query["dir_z"])
    xs, ys = [], []
    for j in range(res):
        for i in range(res):
            xs.append(x_min + (i + 0.5) * (x_max - x_min) / res)
            ys.append(y_min + (j + 0.5) * (y_max - y_min) / res)
    count = res * res
    ray = rd.Ray(cuda.Array3f(xs, ys, [z] * count), cuda.Array3f([0.0] * count, [0.0] * count, [dir_z] * count))
    its = scene.intersect(ray)
    valid = _bools_as_ints(its.is_valid())
    ts = _floats(its.t)
    hit_ts = [t for t, v in zip(ts, valid) if v]
    discrete = {"width": int(dr.width(its.t)), "hit_count": sum(valid)}
    continuous = {"t_hit_min": min(hit_ts), "t_hit_max": max(hit_ts)}
    return {"kind": "intersect_grid", "discrete": discrete, "continuous": continuous}


def _run_shadow(cuda, rd, scene, query):
    ray = _make_ray(rd, cuda, query)
    active = _active(cuda, query)
    shadow = scene.shadow_test(ray, active) if active is not None else scene.shadow_test(ray)
    return {"kind": "shadow_test", "discrete": {"hit": _bools_as_ints(shadow)}, "continuous": {}}


def _run_visible(cuda, scene, query):
    start = _make_vec3(cuda, query["start"])
    end = _make_vec3(cuda, query["end"])
    kwargs = {}
    if "ignore" in query:
        kwargs["ignore_prim_ids"] = cuda.Int([int(v) for v in query["ignore"]])
    active = _active(cuda, query)
    if active is not None:
        kwargs["active"] = active
    result = scene.visible(start, end, **kwargs)
    return {"kind": "visible", "discrete": {"visible": _bools_as_ints(result.visible)}, "continuous": {}}


def _run_visible_pair(cuda, scene, query):
    kwargs = {}
    if "ignore" in query:
        kwargs["ignore_prim_ids"] = cuda.Int([int(v) for v in query["ignore"]])
    active = _active(cuda, query)
    if active is not None:
        kwargs["active"] = active
    result = scene.visible_pair(
        _make_vec3(cuda, query["start"]),
        _make_vec3(cuda, query["end_a"]),
        _make_vec3(cuda, query["end_b"]),
        **kwargs,
    )
    return {
        "kind": "visible_pair",
        "discrete": {
            "visible_a": _bools_as_ints(result.visible_a),
            "visible_b": _bools_as_ints(result.visible_b),
        },
        "continuous": {},
    }


def _run_nearest_edge_point(dr, cuda, scene, query):
    pts = _make_vec3(cuda, query["points"])
    active = _active(cuda, query)
    res = scene.nearest_edge(pts, active) if active is not None else scene.nearest_edge(pts)
    width = int(dr.width(res.distance))
    discrete = {
        "valid": _bools_as_ints(res.is_valid()),
        "shape_id": _ints(res.shape_id),
        "edge_id": _ints(res.edge_id),
        "global_edge_id": _ints(res.global_edge_id),
        "is_boundary": _bools_as_ints(res.is_boundary),
    }
    continuous = {
        "distance": _floats(res.distance),
        "edge_t": _floats(res.edge_t),
        "point": _rows3(res.point, width),
        "edge_point": _rows3(res.edge_point, width),
    }
    return _apply_informative(
        {"kind": "nearest_edge_point", "discrete": discrete, "continuous": continuous},
        query.get("informative"),
    )


def _run_nearest_edge_ray(dr, cuda, rd, scene, query):
    ray = _make_ray(rd, cuda, query)
    active = _active(cuda, query)
    res = scene.nearest_edge(ray, active) if active is not None else scene.nearest_edge(ray)
    width = int(dr.width(res.distance))
    discrete = {
        "valid": _bools_as_ints(res.is_valid()),
        "shape_id": _ints(res.shape_id),
        "edge_id": _ints(res.edge_id),
        "global_edge_id": _ints(res.global_edge_id),
        "is_boundary": _bools_as_ints(res.is_boundary),
    }
    continuous = {
        "distance": _floats(res.distance),
        "ray_t": _floats(res.ray_t),
        "edge_t": _floats(res.edge_t),
        "point": _rows3(res.point, width),
        "edge_point": _rows3(res.edge_point, width),
    }
    return _apply_informative(
        {"kind": "nearest_edge_ray", "discrete": discrete, "continuous": continuous},
        query.get("informative"),
    )


def _run_nearest_edges(dr, cuda, scene, query):
    pts = _make_vec3(cuda, query["points"])
    k = int(query["k"])
    active = _active(cuda, query)
    res = scene.nearest_edges(pts, k, active) if active is not None else scene.nearest_edges(pts, k)
    width = int(dr.width(res.distances))
    discrete = {
        "query_count": int(res.query_count),
        "k": int(res.k),
        "is_valid": _bools_as_ints(res.is_valid),
        "shape_ids": _ints(res.shape_ids),
        "edge_ids": _ints(res.edge_ids),
        "global_edge_ids": _ints(res.global_edge_ids),
        "is_boundary": _bools_as_ints(res.is_boundary),
    }
    continuous = {
        "distances": _floats(res.distances),
        "edge_t": _floats(res.edge_t),
        "points": _rows3(res.points, width),
        "edge_points": _rows3(res.edge_points, width),
    }
    return _apply_informative(
        {"kind": "nearest_edges", "discrete": discrete, "continuous": continuous},
        query.get("informative"),
    )


# Query kinds that cannot run under the eager CUDA trace backend. Since P4 Stage
# D the CUDA fused executor serves segment visibility (visible / visible_pair),
# so nothing in the golden set is CUDA-unsupported.
_CUDA_UNSUPPORTED_KINDS = set()


def _collect_scene(dr, cuda, rd, scene_def, trace_backend=None):
    meshes = []
    scene = rd.Scene() if trace_backend is None else rd.Scene(trace_backend=trace_backend)
    skip_kinds = _CUDA_UNSUPPORTED_KINDS if trace_backend == "cuda" else set()
    for mesh_def in scene_def["meshes"]:
        mesh = rd.Mesh(
            _make_vec3(cuda, mesh_def["vertices"]),
            cuda.Array3i(
                [int(f[0]) for f in mesh_def["faces"]],
                [int(f[1]) for f in mesh_def["faces"]],
                [int(f[2]) for f in mesh_def["faces"]],
            ),
        )
        mesh_id = scene.add_mesh(mesh, dynamic=bool(mesh_def.get("dynamic", False)))
        meshes.append(mesh_id)
    scene.build()
    offsets = _ints(scene.mesh_face_offsets())

    queries = {}
    if scene_def.get("record_face_offsets"):
        queries["mesh_face_offsets"] = {
            "kind": "meta",
            "discrete": {"offsets": offsets},
            "continuous": {},
        }

    for query in scene_def["queries"]:
        kind = query["kind"]
        if kind in skip_kinds:
            continue
        if kind == "update_vertices":
            scene.update_mesh_vertices(meshes[int(query["mesh"])], _make_vec3(cuda, query["vertices"]))
            scene.sync()
            continue
        if kind == "intersect":
            record = _run_intersect(dr, cuda, rd, scene, query, offsets)
        elif kind == "intersect_grid":
            record = _run_intersect_grid(dr, cuda, rd, scene, query)
        elif kind == "shadow_test":
            record = _run_shadow(cuda, rd, scene, query)
        elif kind == "visible":
            record = _run_visible(cuda, scene, query)
        elif kind == "visible_pair":
            record = _run_visible_pair(cuda, scene, query)
        elif kind == "nearest_edge_point":
            record = _run_nearest_edge_point(dr, cuda, scene, query)
        elif kind == "nearest_edge_ray":
            record = _run_nearest_edge_ray(dr, cuda, rd, scene, query)
        elif kind == "nearest_edges":
            record = _run_nearest_edges(dr, cuda, scene, query)
        else:
            raise ValueError(f"unknown query kind {kind!r} in scene {scene_def['name']!r}")
        queries[query["name"]] = record

    return {"queries": queries}


def collect_golden(backend="drjit", trace_backend=None):
    """Collect golden results. ``trace_backend`` selects the triangle backend
    passed to ``rd.Scene`` (None keeps the default OptiX backend); under
    ``"cuda"`` the OptiX-only multipath queries are skipped."""
    dr, cuda, rd = _load_backend(backend)
    result = {}
    for scene_def in scene_defs.SCENES:
        result[scene_def["name"]] = _collect_scene(dr, cuda, rd, scene_def, trace_backend)
    return result


def _git_head():
    out = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        text=True, capture_output=True, check=False,
    )
    return out.stdout.strip() or "unknown"


def _gpu_manifest_fields():
    fields = {"gpu": "unknown", "driver": "unknown", "cuda": "unknown"}
    try:
        query = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            text=True, capture_output=True, check=False,
        )
        first = query.stdout.strip().splitlines()
        if first:
            name, _, driver = first[0].partition(",")
            fields["gpu"] = name.strip()
            fields["driver"] = driver.strip()
        banner = subprocess.run(["nvidia-smi"], text=True, capture_output=True, check=False)
        match = re.search(r"CUDA Version:\s*([0-9.]+)", banner.stdout)
        if match:
            fields["cuda"] = match.group(1)
    except FileNotFoundError:
        pass
    return fields


def write_baselines(backend="drjit"):
    data = collect_golden(backend)
    out_dir = BASELINE_DIR / ("optix" if backend == "drjit" else backend)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, scene_data in data.items():
        path = out_dir / f"{name}.json"
        path.write_text(
            json.dumps(scene_data, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
        )
    manifest = {
        "backend": backend,
        "rayd_commit": _git_head(),
        "python": sys.version.split()[0],
        "generation_command": "python -m tests.golden.runner --write",
        "comparison_policy": COMPARISON_POLICY,
        "scenes": sorted(data.keys()),
        **_gpu_manifest_fields(),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return out_dir


def main():
    parser = argparse.ArgumentParser(description="RayD golden scene runner")
    parser.add_argument("--backend", default="drjit")
    parser.add_argument("--write", action="store_true", help="write baseline JSON files")
    args = parser.parse_args()
    if args.write:
        out_dir = write_baselines(args.backend)
        print(f"wrote baselines to {out_dir}")
    else:
        print(json.dumps(collect_golden(args.backend), sort_keys=True))


if __name__ == "__main__":
    main()
