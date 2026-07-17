"""P4 Stage D gate: the CUDA fused multipath executor matches OptiX.

Every multipath pipeline the CUDA fused executor serves is run under both
``trace_backend='optix'`` and ``trace_backend='cuda'`` in fresh subprocesses
(so Dr.Jit/CUDA state never leaks between the two), and the results are compared:

* Discrete fields (visibility bools, blocker / prim / group ids, bounce and path
  counts, path topology) are BIT-IDENTICAL cuda vs optix.
* Continuous fields (t, hit points, path lengths, fields, grid power) match
  within the ``shared/contracts/operations.json`` tolerances.

Pipelines covered: reflection trace (``trace_reflections`` symbolic=False),
segment visibility (``visible`` / ``visible_pair`` / ``visible_edge`` /
``visible_chain``), reflection EPC (``trace_refl_epc`` primitive + surface-group,
``trace_refl_epc_field``), reflection accumulation (``accumulate_reflections``),
and first-order diffraction path export (``trace_dfr_paths``).
"""

import json
import math
import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]

# operations.json tolerances (default 1e-5, field 5e-5).
DEFAULT_ABS = 1e-5
DEFAULT_REL = 1e-5
FIELD_ABS = 5e-5
FIELD_REL = 5e-5

SCENE_SCRIPT = r'''
import json, math, sys
import numpy as np
import drjit as dr
import drjit.cuda as cuda
import rayd.drjit as rd

TB = sys.argv[1]

def build():
    floor = rd.Mesh(
        cuda.Array3f([-5.0, 5.0, 5.0, -5.0], [-5.0, -5.0, 5.0, 5.0], [0.0, 0.0, 0.0, 0.0]),
        cuda.Array3i([0, 0], [1, 2], [2, 3]))
    wall = rd.Mesh(
        cuda.Array3f([-1.0, 1.0, 0.0], [-1.0, -1.0, 1.0], [1.0, 1.0, 1.5]),
        cuda.Array3i([0], [1], [2]))
    scene = rd.Scene(trace_backend=TB)
    scene.add_mesh(floor); scene.add_mesh(wall); scene.build()
    return scene

def flt(a):
    return [float(x) for x in a.numpy().tolist()]
def i(a):
    return [int(x) for x in a.numpy().tolist()]
def b(a):
    return [bool(x) for x in a.numpy().tolist()]

out = {}
scene = build()

# reflection trace (symbolic=False -> native pipeline / CUDA fused arm)
o = cuda.Array3f([-3.0, 0.0, 2.0], [0.0, 1.0, -2.0], [3.0, 3.0, 3.0])
d = cuda.Array3f([1.0, 0.0, 0.5], [0.0, 0.2, 0.6], [-1.0, -1.0, -1.0])
d = d / dr.norm(d)
refl = scene.trace_reflections(rd.Ray(o, d), 3, symbolic=False)
out["refl_bounce_count"] = i(refl.bounce_count)
out["refl_shape_ids"] = i(refl.shape_ids)
out["refl_global_prim_ids"] = i(refl.global_prim_ids)
out["refl_t"] = flt(refl.t)
out["refl_hit"] = flt(dr.ravel(refl.hit_points))
out["refl_img"] = flt(dr.ravel(refl.image_sources))

# segment visibility family
start = cuda.Array3f([0.0, 0.0, -2.0], [0.0, 2.0, 0.0], [3.0, 3.0, 3.0])
end = cuda.Array3f([0.0, 0.0, 2.0], [0.0, 2.0, 0.0], [-1.0, -1.0, -1.0])
ignore = cuda.Int([-1, -1, -1]); active = cuda.Bool([True, True, True])
out["visible"] = b(scene.visible(start, end, ignore, active).visible)
pair = scene.visible_pair(start, end,
    cuda.Array3f([3.0, 3.0, 3.0], [3.0, 3.0, 3.0], [-3.0, -3.0, -3.0]), ignore, active)
out["visible_a"] = b(pair.visible_a); out["visible_b"] = b(pair.visible_b)
edge = scene.visible_edge(cuda.Array3f([0.0], [0.0], [3.0]), cuda.Array3f([-2.0], [0.0], [0.0]),
    cuda.Array3f([1.0], [0.0], [0.0]), cuda.Float([0.0]), cuda.Float([4.0]),
    [0.0, 0.25, 0.5, 0.75, 1.0], cuda.Bool([True]))
out["edge_visible"] = b(edge.any_visible)
pts = cuda.Array3f([0.0, 0.0, 0.0, 2.0, 2.0, 2.0], [0.0, 0.5, 1.0, 0.0, 0.5, 1.0],
                   [3.0, 1.0, -1.0, 3.0, 3.0, 3.0])
chain = scene.visible_chain(pts, cuda.Int([2, 2]), cuda.Int([]), cuda.Bool([True, True]))
out["chain_all_visible"] = b(chain.all_visible)
out["chain_first_blocked_segment"] = i(chain.first_blocked_segment)
out["chain_first_blocked_prim"] = i(chain.first_blocked_prim)

# accumulate reflections
n = 64
rng = np.random.default_rng(7)
th = rng.uniform(0.2, 0.9, n); ph = rng.uniform(0, 2 * math.pi, n)
aray = rd.Ray(
    cuda.Array3f([0.0] * n, [0.0] * n, [3.0] * n),
    cuda.Array3f((np.sin(th) * np.cos(ph)).astype(np.float32).tolist(),
                 (np.sin(th) * np.sin(ph)).astype(np.float32).tolist(),
                 (-np.cos(th)).astype(np.float32).tolist()))
mat = rd.Material()
mat.eta_r = cuda.Float([5.0, 5.0, 5.0]); mat.sigma = cuda.Float([0.01, 0.01, 0.01])
mat.gain = cuda.Float([1.0, 1.0, 1.0]); mat.mu_r = cuda.Float([1.0, 1.0, 1.0])
mat.valid = cuda.Bool([True, True, True])
grid = rd.AccumGrid()
grid.axis = 2; grid.position = 2.0
grid.coord0_min = -5.0; grid.coord0_max = 5.0; grid.coord1_min = -5.0; grid.coord1_max = 5.0
grid.resolution0 = 8; grid.resolution1 = 8
opts = rd.AccumOptions()
opts.wavelength = 0.1; opts.k = 0.5; opts.solid_angle_per_ray = 1.0; opts.cell_area = 1.0
opts.seed = 17; opts.rr_depth = 0; opts.rr_prob = 1.0; opts.stop_threshold = 0.0
acc = scene.accumulate_reflections(aray, cuda.Array3f([0.0], [0.0], [3.0]), grid, mat, 2, opts,
                                   cuda.Bool([True] * n), cuda.Array3f([1.0], [0.0], [0.0]))
out["acc_count"] = int(acc.reflection_count.numpy().tolist()[0])
out["acc_power"] = flt(acc.reflection_power)
out["acc_field_x_re"] = flt(acc.reflection_field_x.real)
out["acc_field_x_im"] = flt(acc.reflection_field_x.imag)

print(json.dumps(out, sort_keys=True))
'''

EPC_SCRIPT = r'''
import json, math, sys
import drjit as dr
import drjit.cuda as cuda
import rayd.drjit as rd
TB = sys.argv[1]
mirror = rd.Mesh(cuda.Array3f([-1.0, 1.0, 1.0, -1.0], [-1.0, -1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]),
                 cuda.Array3i([0, 0], [1, 2], [2, 3]))
scene = rd.Scene(trace_backend=TB); scene.add_mesh(mirror); scene.build()
ray = rd.Ray(cuda.Array3f([0.0], [-0.5], [-1.0]), cuda.Array3f([0.0], [0.0], [1.0]))
rx = cuda.Array3f([0.0], [1.5], [-1.0])
out = {}
opt = rd.ReflEpcOptions()
opt.expected_prim_ids = cuda.Int([1]); opt.surface_group_id = cuda.Int([0, 0])
opt.surface_group_size = cuda.Int([2]); opt.surface_group_members = cuda.Int([0, 1])
opt.surface_max_group_size = 2; opt.visibility_ignore_mode = "surface_group"
g = scene.trace_refl_epc(ray, rx, max_bounces=1, options=opt)
out["g_valid"] = [bool(x) for x in g.valid.numpy().tolist()]
out["g_path"] = [float(x) for x in g.path_length.numpy().tolist()]
out["g_point"] = [float(x) for x in dr.ravel(g.reflection_points).numpy().tolist()]
out["g_group"] = [int(x) for x in g.surface_group_ids.numpy().tolist()]
out["g_resolved"] = [int(x) for x in g.resolved_prim_ids.numpy().tolist()]
# EPC field (direct-primary discovery + backend-agnostic field kernel)
fo = rd.ReflEpcFieldOptions()
fo.slot_plane_normal = cuda.Array3f([0.0], [0.0], [1.0]); fo.slot_eta_r = cuda.Float([4.0])
fo.slot_mu_r = cuda.Float([1.0]); fo.slot_sigma = cuda.Float([0.0]); fo.slot_gain = cuda.Float([1.0])
fo.tx_polarization = cuda.Array3f([1.0], [0.0], [0.0])
fo.omega = 2.0 * math.pi * 3e8 / 0.1; fo.wavelength = 0.1
fr = scene.trace_refl_epc_field(ray, rx, max_bounces=1, options=fo)
out["f_valid"] = [bool(x) for x in fr.valid.numpy().tolist()]
out["f_path"] = [float(x) for x in fr.path_length.numpy().tolist()]
out["f_fx_re"] = [float(x) for x in fr.field_x_re.numpy().tolist()]
out["f_fx_im"] = [float(x) for x in fr.field_x_im.numpy().tolist()]
print(json.dumps(out, sort_keys=True))
'''

DFR_PATHS_SCRIPT = r'''
import json, math, sys
import numpy as np
import drjit.cuda as cuda
import rayd.drjit as pj
TB = sys.argv[1]
scene = pj.Scene(trace_backend=TB)
scene.add_mesh(pj.Mesh(cuda.Array3f([-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0], [10.0, 10.0, 10.0]),
                       cuda.Array3i([0], [1], [2])))
scene.build()
states = pj.DfrStates()
states.count = 1; states.edge_index = cuda.Int([0])
states.edge_pos = cuda.Array3f([0.0], [0.0], [0.0]); states.edge_dir = cuda.Array3f([1.0], [0.0], [0.0])
states.edge_t_min = cuda.Float([-0.5]); states.edge_t_max = cuda.Float([0.5])
states.n0 = cuda.Array3f([0.0], [1.0], [0.0]); states.n1 = cuda.Array3f([0.0], [-1.0], [0.0])
states.prim0 = cuda.Int([-1]); states.prim1 = cuda.Int([-1])
states.exterior_angle = cuda.Float([1.5 * math.pi])
states.src = cuda.Array3f([0.0], [0.0], [1.0]); states.src_power = cuda.Float([1.0])
states.wi = cuda.Array3f([0.0], [0.0], [-1.0]); states.d0 = cuda.Array3f([0.0], [0.0], [-1.0])
states.prefix_depth = cuda.Int([0])
mat = pj.DfrMaterial()
mat.eta_r = cuda.Float([4.0]); mat.sigma = cuda.Float([0.0]); mat.mu_r = cuda.Float([1.0])
mat.gain = cuda.Float([1.0]); mat.valid = cuda.Bool([True])
opt = pj.DfrPathOptions()
opt.wavelength = 0.125; opt.k = 50.26548245743669; opt.seed = 17; opt.max_order = 1
opt.max_paths = 4; opt.max_rx = 1; opt.strategy_mask = pj.RAYD_DFR_DIRECT
opt.sample_count = 4; opt.return_geom = 1; opt.receiver_model = pj.RAYD_DFR_MATCHED_ISO
r = scene.trace_dfr_paths(cuda.Array3f([0.0], [0.0], [1.0]), cuda.Array3f([0.0], [0.0], [-1.0]),
                          states, mat, opt, cuda.Bool([True]))
print(json.dumps({
    "capacity": int(r.capacity),
    "count": int(np.asarray(r.count, dtype=np.int32)[0]),
    "valid0": bool(np.asarray(r.valid, dtype=np.bool_)[0]),
    "rx0": int(np.asarray(r.rx_id, dtype=np.int32)[0]),
    "edge0": int(np.asarray(r.edge0, dtype=np.int32)[0]),
    "delay0": float(np.asarray(r.delay, dtype=np.float32)[0]),
    "fx_re": float(np.asarray(r.field_x.real, dtype=np.float32)[0]),
    "fx_im": float(np.asarray(r.field_x.imag, dtype=np.float32)[0]),
    "p0_x": float(np.asarray(r.p0.x, dtype=np.float32)[0]),
}, sort_keys=True))
'''


def _run(script, backend, timeout=240):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script), backend],
        cwd=str(ROOT), env=env, text=True, capture_output=True, timeout=timeout, check=False)
    if result.returncode != 0:
        raise AssertionError(
            f"Subprocess ({backend}) failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
    lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    return json.loads(lines[-1])


class CudaMultipathParityTests(unittest.TestCase):
    DISCRETE_PREFIXES = ("refl_bounce_count", "refl_shape_ids", "refl_global_prim_ids",
                         "visible", "edge_visible", "chain_", "acc_count", "g_valid", "g_group",
                         "g_resolved", "f_valid", "capacity", "count", "valid0", "rx0", "edge0")
    FIELD_KEYS = ("acc_field", "acc_power", "f_fx", "fx_re", "fx_im", "g_path", "f_path", "delay0")

    def _compare(self, optix, cuda):
        for key in sorted(optix):
            with self.subTest(key=key):
                ov, cv = optix[key], cuda[key]
                is_discrete = any(key.startswith(p) or key == p for p in self.DISCRETE_PREFIXES)
                is_field = any(key.startswith(p) for p in self.FIELD_KEYS)
                if isinstance(ov, list) and ov and isinstance(ov[0], (bool, int)) or is_discrete:
                    self.assertEqual(ov, cv, f"discrete mismatch for {key}")
                else:
                    abs_tol = FIELD_ABS if is_field else DEFAULT_ABS
                    rel_tol = FIELD_REL if is_field else DEFAULT_REL
                    ovl = ov if isinstance(ov, list) else [ov]
                    cvl = cv if isinstance(cv, list) else [cv]
                    for a, b in zip(ovl, cvl):
                        if math.isinf(a) or math.isnan(a):
                            self.assertEqual(repr(a), repr(b), f"sentinel mismatch {key}")
                            continue
                        ad = abs(a - b)
                        self.assertTrue(ad <= abs_tol or ad <= rel_tol * abs(a),
                                        f"{key}: optix={a} cuda={b} abs={ad}")

    def test_reflection_visibility_accumulation_parity(self):
        self._compare(_run(SCENE_SCRIPT, "optix"), _run(SCENE_SCRIPT, "cuda"))

    def test_reflection_epc_and_field_parity(self):
        optix = _run(EPC_SCRIPT, "optix")
        cuda = _run(EPC_SCRIPT, "cuda")
        # Surface-group EPC path must be valid and bit-identical.
        self.assertTrue(optix["g_valid"][0])
        self._compare(optix, cuda)

    def test_diffraction_paths_parity(self):
        optix = _run(DFR_PATHS_SCRIPT, "optix")
        cuda = _run(DFR_PATHS_SCRIPT, "cuda")
        self.assertTrue(optix["valid0"])
        self._compare(optix, cuda)


if __name__ == "__main__":
    unittest.main()
