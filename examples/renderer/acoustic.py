"""Acoustic impulse response via GPU ray tracing — differentiable w.r.t. source position."""

import argparse
import time
from pathlib import Path

import drjit as dr
import drjit.cuda as cuda
import drjit.cuda.ad as ad
import matplotlib.pyplot as plt
import numpy as np
import raydi as rd

SPEED_OF_SOUND = 343.0  # m/s

# ---------------------------------------------------------------------------
# Scene: shoebox room
# ---------------------------------------------------------------------------


def arr3(verts):
    v = np.asarray(verts, dtype=np.float32)
    return cuda.Array3f(v[:, 0].tolist(), v[:, 1].tolist(), v[:, 2].tolist())


def quad(a, b, c, d):
    return rd.Mesh(arr3([a, b, c, d]), cuda.Array3i([0, 0], [1, 2], [2, 3]))


def build_room(room_size, absorption):
    """6-wall shoebox room.  Returns (scene, per-wall absorption coefficients)."""
    x, y, z = room_size
    scene = rd.Scene()
    absorb = []

    def wall(a, b, c, d, alpha):
        m = quad(a, b, c, d)
        m.use_face_normals = True
        scene.add_mesh(m)
        absorb.append(alpha)

    wall([0, 0, 0], [x, 0, 0], [x, 0, z], [0, 0, z], absorption)  # floor
    wall([0, y, 0], [0, y, z], [x, y, z], [x, y, 0], absorption)  # ceiling
    wall([0, 0, 0], [0, 0, z], [0, y, z], [0, y, 0], absorption)  # left
    wall([x, 0, 0], [x, y, 0], [x, y, z], [x, 0, z], absorption)  # right
    wall([0, 0, z], [x, 0, z], [x, y, z], [0, y, z], absorption)  # back
    wall([0, 0, 0], [0, y, 0], [x, y, 0], [x, 0, 0], absorption)  # front

    scene.configure()
    alpha_table = cuda.Float(absorb)
    return scene, alpha_table

# ---------------------------------------------------------------------------
# Acoustic ray tracer
# ---------------------------------------------------------------------------


def uniform_sphere(n, rng):
    """Sample directions uniformly on the unit sphere."""
    u = rng.next_float32()
    v = rng.next_float32()
    z = 1.0 - 2.0 * u
    r = dr.sqrt(dr.maximum(1.0 - z * z, 0.0))
    phi = 2.0 * 3.14159265 * v
    return cuda.Array3f(r * dr.cos(phi), r * dr.sin(phi), z)


def trace_acoustic(scene, alpha_table, source, receiver, receiver_radius,
                   n_rays, max_bounces, rng):
    """Trace rays from source, collect hits near receiver.

    Returns (arrival_time, energy) for each ray that reaches the receiver.
    Rays that never reach the receiver get time=0, energy=0.
    """
    ray_d = uniform_sphere(n_rays, rng)
    ones = dr.full(cuda.Float, 1.0, n_rays)
    ray_o = cuda.Array3f(source[0] * ones, source[1] * ones, source[2] * ones)
    energy = dr.full(cuda.Float, 1.0, n_rays)
    path_length = dr.zeros(cuda.Float, n_rays)
    active = dr.full(cuda.Bool, True, n_rays)

    # accumulators for receiver hits
    hit_time = dr.zeros(cuda.Float, n_rays)
    hit_energy = dr.zeros(cuda.Float, n_rays)
    hit_found = dr.full(cuda.Bool, False, n_rays)

    recv = cuda.Array3f(float(receiver[0]) * ones, float(receiver[1]) * ones, float(receiver[2]) * ones)
    r_sq = receiver_radius * receiver_radius

    for bounce in range(max_bounces):
        its = scene.intersect(rd.RayDetached(ray_o, ray_d), active)
        hit_wall = active & its.is_valid()

        # check if the ray segment passes near the receiver
        # project receiver onto ray: t_recv = dot(recv - ray_o, ray_d)
        to_recv = recv - ray_o
        t_recv = dr.dot(to_recv, ray_d)
        closest = ray_o + ray_d * dr.maximum(t_recv, 0.0)
        dist_sq = dr.squared_norm(closest - recv)
        # receiver hit: close enough AND before the wall hit
        recv_hit = hit_wall & ~hit_found & (dist_sq < r_sq) & (t_recv > 0) & (t_recv < its.t)
        recv_dist = path_length + dr.maximum(t_recv, 0.0)
        hit_time = dr.select(recv_hit, recv_dist / SPEED_OF_SOUND, hit_time)
        hit_energy = dr.select(recv_hit, energy / dr.maximum(recv_dist * recv_dist, 1e-4), hit_energy)
        hit_found = hit_found | recv_hit

        # update path length and reflect
        path_length = dr.select(hit_wall, path_length + its.t, path_length)
        wall_id = dr.select(hit_wall, its.shape_id, 0)
        alpha = dr.gather(cuda.Float, alpha_table, wall_id, hit_wall)
        energy = dr.select(hit_wall, energy * (1.0 - alpha), energy)

        # specular reflection
        n = its.geo_n
        n = dr.select(dr.dot(n, ray_d) > 0, -n, n)
        reflected = ray_d - 2.0 * dr.dot(ray_d, n) * n
        ray_o = dr.select(hit_wall, its.p + n * 1e-3, ray_o)
        ray_d = dr.select(hit_wall, dr.normalize(reflected), ray_d)
        active = hit_wall & ~hit_found

    return hit_time, hit_energy

# ---------------------------------------------------------------------------
# Differentiable wrapper: gradient of impulse response w.r.t. source x
# ---------------------------------------------------------------------------


def compute_ir(scene, alpha_table, source, receiver, receiver_radius,
               n_rays, max_bounces, seed, time_bins, t_max):
    rng = cuda.PCG32(size=n_rays)
    rng.seed(initstate=dr.arange(cuda.UInt64, n_rays) + seed,
             initseq=dr.full(cuda.UInt64, seed + 1, n_rays))

    arrival, energy = trace_acoustic(
        scene, alpha_table, source, receiver, receiver_radius,
        n_rays, max_bounces, rng)

    # bin into histogram
    valid = energy > 0
    bin_idx = cuda.UInt(dr.clip(arrival / t_max * time_bins, 0, time_bins - 1))
    ir = dr.zeros(cuda.Float, time_bins)
    dr.scatter_reduce(dr.ReduceOp.Add, ir, energy, bin_idx, valid)
    return ir

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-rays", type=int, default=1_000_000)
    ap.add_argument("--max-bounces", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--time-bins", type=int, default=200)
    ap.add_argument("--t-max", type=float, default=0.15)
    args = ap.parse_args()

    room_size = [8.0, 4.0, 6.0]
    absorption = 0.08
    source = [2.0, 1.8, 2.0]
    receiver = [6.0, 1.8, 4.0]
    receiver_radius = 0.3

    # build scene
    t0 = time.perf_counter()
    scene, alpha_table = build_room(room_size, absorption)
    dr.sync_thread()
    t1 = time.perf_counter()

    # forward IR
    ir = compute_ir(scene, alpha_table, source, receiver, receiver_radius,
                    args.n_rays, args.max_bounces, args.seed,
                    args.time_bins, args.t_max)
    dr.eval(ir)
    dr.sync_thread()
    t2 = time.perf_counter()

    # gradient of IR w.r.t. source x-position
    src_x = ad.Float([float(source[0])])
    dr.enable_grad(src_x)
    source_ad = [src_x, source[1], source[2]]
    ir_ad = compute_ir(scene, alpha_table, source_ad, receiver, receiver_radius,
                       args.n_rays, args.max_bounces, args.seed + 1,
                       args.time_bins, args.t_max)
    dr.forward(src_x)
    grad_ir = dr.grad(ir_ad)
    dr.eval(grad_ir)
    dr.sync_thread()
    t3 = time.perf_counter()

    # plot
    ir_np = np.asarray(dr.detach(ir))
    grad_np = np.asarray(dr.detach(grad_ir))
    t_axis = np.linspace(0, args.t_max * 1000, args.time_bins)  # ms

    out = Path(__file__).resolve().parent / "acoustic.png"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    ax1.bar(t_axis, ir_np, width=t_axis[1] - t_axis[0], color="steelblue")
    ax1.set_ylabel("energy")
    ax1.set_title(f"impulse response  ({args.n_rays:,} rays, {args.max_bounces} bounces)")
    lim = max(float(np.max(np.abs(grad_np))), 1e-8)
    ax2.bar(t_axis, grad_np, width=t_axis[1] - t_axis[0], color="indianred")
    ax2.set_xlabel("time (ms)")
    ax2.set_ylabel("d(energy)/d(source_x)")
    ax2.set_title("gradient w.r.t. source x-position")
    plt.tight_layout()
    plt.savefig(out, dpi=150)

    print(f"saved = {out}")
    print(f"room = {room_size}  absorption = {absorption}")
    print(f"source = {source}  receiver = {receiver}")
    print(f"rays = {args.n_rays:,}  bounces = {args.max_bounces}  bins = {args.time_bins}")
    print(f"timing:  setup {(t1-t0)*1e3:.0f}ms  ir {(t2-t1)*1e3:.0f}ms  grad {(t3-t2)*1e3:.0f}ms")
    print(f"peak energy = {float(np.max(ir_np)):.6f}  peak |grad| = {float(np.max(np.abs(grad_np))):.6f}")


if __name__ == "__main__":
    main()
