"""Differentiable Cornell box — path tracing + edge-AD in ~200 lines."""

import argparse
import time
from pathlib import Path

import drjit as dr
import drjit.cuda as cuda
import drjit.cuda.ad as ad
import matplotlib.pyplot as plt
import numpy as np
import raydi as rd

PI = 3.14159265358979323846

# ---------------------------------------------------------------------------
# Scene: Cornell box with two blocks and an area light
# ---------------------------------------------------------------------------

LIGHT_P0 = np.float32([-0.25, 0.99, 2.65])
LIGHT_U = np.float32([0.50, 0.0, 0.0])
LIGHT_V = np.float32([0.0, 0.0, 0.60])
LIGHT_N = np.float32([0.0, -1.0, 0.0])
LIGHT_AREA = 0.50 * 0.60
LIGHT_E = np.float32([18.0, 16.0, 14.0])


def arr3(verts):
    v = np.asarray(verts, dtype=np.float32)
    return cuda.Array3f(v[:, 0].tolist(), v[:, 1].tolist(), v[:, 2].tolist())


def quad(a, b, c, d):
    return rd.Mesh(arr3([a, b, c, d]), cuda.Array3i([0, 0], [1, 2], [2, 3]))


def box(lo, hi):
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    v = arr3([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ])
    f = cuda.Array3i(
        [0, 0, 4, 4, 0, 0, 1, 1, 0, 0, 3, 3],
        [2, 3, 5, 6, 7, 4, 2, 6, 1, 5, 7, 6],
        [1, 2, 6, 7, 3, 7, 6, 5, 5, 4, 6, 2],
    )
    m = rd.Mesh(v, f)
    m.use_face_normals = True
    return m


def build_scene(tx):
    scene = rd.Scene()
    albedos, emissions = [], []

    def add(mesh, albedo, emission=(0, 0, 0)):
        scene.add_mesh(mesh)
        albedos.append(albedo)
        emissions.append(emission)

    add(quad([-1, -1, 1], [-1, -1, 5], [1, -1, 5], [1, -1, 1]), [.78, .78, .78])              # floor
    add(quad([-1, 1, 1], [1, 1, 1], [1, 1, 5], [-1, 1, 5]),     [.78, .78, .78])              # ceiling
    add(quad([-1, -1, 5], [-1, 1, 5], [1, 1, 5], [1, -1, 5]),   [.75, .75, .75])              # back
    add(quad([-1, -1, 1], [-1, 1, 1], [-1, 1, 5], [-1, -1, 5]), [.74, .14, .12])              # left (red)
    add(quad([1, -1, 1], [1, -1, 5], [1, 1, 5], [1, 1, 1]),     [.12, .55, .16])              # right (green)
    lq = np.array([[-0.25, 0.99, 2.65], [0.25, 0.99, 2.65],
                    [-0.25, 0.99, 3.25], [0.25, 0.99, 3.25]])[np.array([0, 1, 3, 2])]
    light_id = len(albedos)
    add(quad(*lq), [1, 1, 1], LIGHT_E)                                                        # light
    add(box([-0.72, -1.0, 2.70], [-0.18, 0.25, 3.62]), [.82, .82, .82])                       # tall block

    moving = box([0.18, -1.0, 1.82], [0.68, -0.34, 2.48])                                     # short block
    moving.to_world_left = ad.Matrix4f([[1, 0, 0, tx], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    add(moving, [.86, .84, .80])

    scene.configure()
    albedo_t = arr3(np.array(albedos, dtype=np.float32))
    emission_t = arr3(np.array(emissions, dtype=np.float32))
    return scene, albedo_t, emission_t, light_id

# ---------------------------------------------------------------------------
# Path tracer with next-event estimation
# ---------------------------------------------------------------------------


def ortho_frame(n):
    sign = dr.select(n[2] >= 0.0, 1.0, -1.0)
    a = -dr.rcp(sign + n[2])
    b = n[0] * n[1] * a
    return (cuda.Array3f(1.0 + sign * n[0] * n[0] * a, sign * b, -sign * n[0]),
            cuda.Array3f(b, sign + n[1] * n[1] * a, -n[1]))


def cosine_hemisphere(n, rng):
    u1, u2 = rng.next_float32(), rng.next_float32()
    r = dr.sqrt(u1)
    phi = 2.0 * PI * u2
    t, s = ortho_frame(n)
    return dr.normalize(t * (r * dr.cos(phi)) + s * (r * dr.sin(phi)) + n * dr.sqrt(dr.maximum(0.0, 1.0 - u1)))


def trace(scene, albedo_t, emission_t, light_id, primary_ray, rng, max_depth=4):
    n = dr.width(primary_ray.o[0])
    beta = dr.full(cuda.Array3f, 1.0, n)
    L = dr.zeros(cuda.Array3f, n)
    ray = primary_ray
    active = dr.full(cuda.Bool, True, n)

    for depth in range(max_depth):
        its = scene.intersect(ray, active)
        hit = active & its.is_valid()

        sn = dr.select(dr.dot(its.n, ray.d) > 0, -its.n, its.n)
        gn = dr.select(dr.dot(its.geo_n, ray.d) > 0, -its.geo_n, its.geo_n)
        sid = dr.select(hit, its.shape_id, 0)
        kd = dr.gather(cuda.Array3f, albedo_t, sid, hit)
        Le = dr.gather(cuda.Array3f, emission_t, sid, hit)

        # emission — first bounce only
        if depth == 0:
            L = dr.select(hit & (sid == light_id), L + beta * Le, L)

        surf = hit & ~(sid == light_id)

        # NEE: sample the area light
        lp = cuda.Array3f(*(LIGHT_P0[i] + rng.next_float32() * LIGHT_U[i] + rng.next_float32() * LIGHT_V[i] for i in range(3)))
        to_l = lp - its.p
        d2 = dr.maximum(dr.squared_norm(to_l), 1e-8)
        wi = to_l * dr.rsqrt(d2)
        cos_s = dr.maximum(dr.dot(sn, wi), 0.0)
        cos_l = dr.maximum(dr.dot(cuda.Array3f(*(-LIGHT_N).tolist()), wi), 0.0)
        can_shade = surf & (cos_s > 0) & (cos_l > 0)
        shadow_ray = rd.RayDetached(its.p + gn * 1e-3, wi)
        shadow_ray.tmax = dr.maximum(dr.sqrt(d2) - 2e-3, 1e-3)
        vis = dr.select(~scene.intersect(shadow_ray, can_shade).is_valid(), 1.0, 0.0)
        L = dr.select(surf, L + beta * kd * (cuda.Array3f(*LIGHT_E.tolist()) * (LIGHT_AREA * cos_s * cos_l * dr.rcp(PI * dr.maximum(d2, 1e-5)) * vis)), L)

        # bounce
        new_d = cosine_hemisphere(sn, rng)
        ray = rd.RayDetached(dr.select(surf, its.p + gn * 1e-3, ray.o), dr.select(surf, new_d, ray.d))
        beta = dr.select(surf, beta * kd, beta)
        active = surf

        # Russian roulette after depth 2
        if depth >= 2:
            q = dr.clip(dr.maximum(beta[0], dr.maximum(beta[1], beta[2])), 0.05, 0.95)
            survive = rng.next_float32() < q
            beta = dr.select(active & survive, beta * dr.rcp(q), beta)
            active &= survive

    return L

# ---------------------------------------------------------------------------
# Edge-AD: primary-edge visibility gradient
# ---------------------------------------------------------------------------


def render_edge_ad(scene, camera, albedo_t, emission_t, light_id, tx, seed=7, spp=16, max_depth=4):
    n = camera.width * camera.height
    sample_count = n * spp
    samples = (dr.arange(cuda.Float, sample_count) + 0.5) / float(sample_count)
    edge = camera.sample_edge(samples)
    valid = ad.Bool(edge.idx >= 0)
    safe_pdf = dr.maximum(edge.pdf, 1e-8)

    def trace_lum(ray_field):
        rng = cuda.PCG32(size=sample_count)
        rng.seed(initstate=dr.arange(cuda.UInt64, sample_count) + seed,
                 initseq=dr.full(cuda.UInt64, seed + 1, sample_count))
        rgb = trace(scene, albedo_t, emission_t, light_id, ray_field, rng, max_depth)
        return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]

    contribution = edge.x_dot_n * ad.Float((trace_lum(edge.ray_n) - trace_lum(edge.ray_p)) / safe_pdf)
    contribution = dr.select(valid, contribution / float(spp), 0.0)
    contribution -= ad.Float(dr.detach(contribution))

    image = dr.zeros(ad.Float, n)
    dr.scatter_reduce(dr.ReduceOp.Add, image, contribution, ad.UInt(edge.idx), valid)
    dr.set_grad(tx, 0.0)
    dr.set_grad(tx, 1.0)
    dr.forward_to(image)
    grad = dr.grad(image)
    dr.eval(grad)
    return grad

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--spp", type=int, default=256)
    ap.add_argument("--edge-spp", type=int, default=16)
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    w, h = args.width, args.height
    n = w * h

    # setup
    t0 = time.perf_counter()
    tx = ad.Float([0.0])
    dr.enable_grad(tx)
    scene, albedo_t, emission_t, light_id = build_scene(tx)

    camera = rd.Camera(55.0, 1e-4, 1e4)
    camera.width, camera.height = w, h
    camera.configure()
    camera.prepare_edges(scene)
    dr.sync_thread()
    t1 = time.perf_counter()

    # render image
    rng = cuda.PCG32(size=n * args.spp)
    rng.seed(initstate=dr.arange(cuda.UInt64, n * args.spp) + args.seed,
             initseq=dr.full(cuda.UInt64, args.seed + 1, n * args.spp))
    px = dr.repeat(cuda.Float(dr.arange(cuda.UInt, n) % w), args.spp)
    py = dr.repeat(cuda.Float(dr.arange(cuda.UInt, n) // w), args.spp)
    uv = cuda.Array2f((px + rng.next_float32()) / w, (py + rng.next_float32()) / h)
    primary = camera.sample_ray(uv)
    rgb = trace(scene, albedo_t, emission_t, light_id, primary, rng, args.max_depth)
    rgb = dr.block_sum(rgb, args.spp) / float(args.spp)
    rgb = dr.power(dr.minimum(rgb * dr.rcp(1.0 + rgb), 1.0), 1.0 / 2.2)
    dr.eval(rgb)
    dr.sync_thread()
    t2 = time.perf_counter()

    # edge-AD gradient
    grad = render_edge_ad(scene, camera, albedo_t, emission_t, light_id, tx,
                          seed=args.seed, spp=args.edge_spp, max_depth=args.max_depth)
    dr.sync_thread()
    t3 = time.perf_counter()

    # show results
    rgb_np = np.stack([np.asarray(dr.detach(rgb[i])) for i in range(3)], axis=1).reshape(h, w, 3)
    grad_np = np.asarray(dr.detach(grad)).reshape(h, w)

    print(f"resolution = {w}x{h}  spp = {args.spp}  edge_spp = {args.edge_spp}  max_depth = {args.max_depth}")
    print(f"timing:  setup {(t1-t0)*1e3:.0f}ms  render {(t2-t1)*1e3:.0f}ms  edge_ad {(t3-t2)*1e3:.0f}ms")

    out = Path(__file__).resolve().parent / "optical.png"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(np.clip(rgb_np, 0, 1))
    ax1.set_title("render")
    ax1.axis("off")
    lim = max(float(np.max(np.abs(grad_np))), 1e-6)
    ax2.imshow(grad_np, cmap="coolwarm", vmin=-lim, vmax=lim)
    ax2.set_title("edge-AD gradient")
    ax2.axis("off")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"saved = {out}")


if __name__ == "__main__":
    main()
