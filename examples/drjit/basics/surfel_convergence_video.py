# Copyright Xingyu Chen.
# Demonstrates surfel convergence video.

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
CWD = os.path.normcase(os.path.abspath(os.getcwd()))
sys.path = [
    entry
    for entry in sys.path
    if os.path.normcase(os.path.abspath(entry or CWD)) != os.path.normcase(str(THIS_FILE.parent))
]
sys.path.insert(0, str(REPO_ROOT))

import drjit as dr
import drjit.cuda as cuda
import drjit.cuda.ad as ad
import rayd.drjit as rd


def make_rays(size: int, extent: float, ad_mode: bool):
    xs = []
    ys = []
    for iy in range(size):
        y = -extent + 2.0 * extent * ((iy + 0.5) / size)
        for ix in range(size):
            x = -extent + 2.0 * extent * ((ix + 0.5) / size)
            xs.append(x)
            ys.append(y)

    count = size * size
    if ad_mode:
        return rd.RayAD(
            ad.Array3f(ad.Float(xs), ad.Float(ys), ad.Float([1.0] * count)),
            ad.Array3f(ad.Float([0.0] * count), ad.Float([0.0] * count), ad.Float([-1.0] * count)),
        )
    return rd.Ray(cuda.Array3f(xs, ys, [1.0] * count), cuda.Array3f([0.0] * count, [0.0] * count, [-1.0] * count))


def make_options() -> rd.SurfelTraceOptions:
    opts = rd.SurfelTraceOptions()
    opts.alpha_min = 1.0 / 255.0
    opts.primitive_mode = rd.SurfelPrimitiveMode.Icosahedron20
    return opts


def render_detached(cx, cy, su, sv, opacity, rays, opts, size: int) -> np.ndarray:
    count = len(cx)
    scene = rd.SurfelScene(
        rd.SurfelCloud(
            cuda.Array3f(cx, cy, [0.0] * count),
            cuda.Array3f(su, [0.0] * count, [0.0] * count),
            cuda.Array3f([0.0] * count, sv, [0.0] * count),
            cuda.Float(opacity),
        ),
        opts,
    )
    scene.build()
    image = scene.composite_alpha(rays).intensity
    dr.eval(image)
    return np.array([float(image[i]) for i in range(size * size)], dtype=np.float32).reshape(size, size)


def fit_step(params, target: np.ndarray, rays, opts, size: int):
    cx = ad.Float(params["cx"])
    cy = ad.Float(params["cy"])
    su = ad.Float(params["su"])
    sv = ad.Float(params["sv"])
    opacity = ad.Float(params["opacity"])
    z = ad.Float([0.0] * len(params["cx"]))
    zeros = ad.Float([0.0] * len(params["cx"]))

    dr.enable_grad(cx)
    dr.enable_grad(cy)
    dr.enable_grad(su)
    dr.enable_grad(sv)
    dr.enable_grad(opacity)

    scene = rd.SurfelScene(
        rd.SurfelCloud(ad.Array3f(cx, cy, z), ad.Array3f(su, zeros, zeros), ad.Array3f(zeros, sv, zeros), opacity), opts
    )
    scene.build()
    pred = scene.composite_alpha(rays).intensity
    residual = pred - ad.Float(target.reshape(-1).tolist())
    loss = dr.sum(residual * residual) / (size * size)
    dr.backward(loss)

    pred_np = np.array([float(pred[i]) for i in range(size * size)], dtype=np.float32).reshape(size, size)
    loss_value = float(loss[0])

    def grad_list(value):
        grad = dr.grad(value)
        dr.eval(grad)
        return [float(grad[i]) for i in range(len(params["cx"]))]

    return (
        pred_np,
        loss_value,
        {
            "cx": grad_list(cx),
            "cy": grad_list(cy),
            "su": grad_list(su),
            "sv": grad_list(sv),
            "opacity": grad_list(opacity),
        },
    )


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def apply_update(params, grads, lr_pos: float, lr_scale: float, lr_opacity: float):
    for i in range(len(params["cx"])):
        params["cx"][i] -= lr_pos * grads["cx"][i]
        params["cy"][i] -= lr_pos * grads["cy"][i]
        params["su"][i] = clamp(params["su"][i] - lr_scale * grads["su"][i], 0.04, 0.4)
        params["sv"][i] = clamp(params["sv"][i] - lr_scale * grads["sv"][i], 0.04, 0.4)
        params["opacity"][i] = clamp(params["opacity"][i] - lr_opacity * grads["opacity"][i], 0.05, 1.5)


def grayscale_panel(values: np.ndarray, vmax: float = 1.0) -> Image.Image:
    normalized = np.clip(values / vmax, 0.0, 1.0)
    rgb = (normalized[..., None] * np.array([255, 255, 255], dtype=np.float32)).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def error_panel(error: np.ndarray, vmax: float = 0.6) -> Image.Image:
    normalized = np.clip(np.abs(error) / vmax, 0.0, 1.0)
    rgb = np.zeros((*error.shape, 3), dtype=np.uint8)
    rgb[..., 0] = (255 * normalized).astype(np.uint8)
    rgb[..., 1] = (40 * (1.0 - normalized)).astype(np.uint8)
    rgb[..., 2] = (40 * (1.0 - normalized)).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def make_frame(target: np.ndarray, pred: np.ndarray, iteration: int, loss: float, total_iterations: int) -> Image.Image:
    panel_size = 220
    header_h = 58
    label_h = 28
    pad = 16
    width = pad * 4 + panel_size * 3
    height = header_h + label_h + panel_size + pad
    frame = Image.new("RGB", (width, height), (20, 22, 26))
    draw = ImageDraw.Draw(frame)

    draw.text((pad, 14), "Ray-traced 2DGS surfel image fitting", fill=(245, 245, 245))
    draw.text((pad, 34), f"iteration {iteration:03d}/{total_iterations:03d}    mse={loss:.6f}", fill=(180, 190, 205))

    panels = [
        ("target", grayscale_panel(target)),
        ("prediction", grayscale_panel(pred)),
        ("absolute error", error_panel(pred - target)),
    ]
    for index, (label, panel) in enumerate(panels):
        x = pad + index * (panel_size + pad)
        y = header_h
        draw.text((x, y), label, fill=(220, 225, 235))
        panel = panel.resize((panel_size, panel_size), Image.Resampling.BILINEAR)
        frame.paste(panel, (x, y + label_h))

    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a GIF showing 2DGS-style surfel image fitting convergence.")
    parser.add_argument("--size", type=int, default=80)
    parser.add_argument("--iterations", type=int, default=72)
    parser.add_argument("--frame-step", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/surfel_convergence"))
    args = parser.parse_args()

    opts = make_options()
    target_rays = make_rays(args.size, 1.0, ad_mode=False)
    fit_rays = make_rays(args.size, 1.0, ad_mode=True)

    target_params = {
        "cx": [-0.58, 0.58, -0.58, 0.58, 0.0],
        "cy": [-0.58, -0.58, 0.58, 0.58, 0.0],
        "su": [0.10, 0.12, 0.11, 0.10, 0.16],
        "sv": [0.13, 0.10, 0.12, 0.14, 0.16],
        "opacity": [0.95, 0.78, 0.86, 0.74, 1.0],
    }
    params = {
        "cx": [-0.70, 0.45, -0.44, 0.69, 0.10],
        "cy": [-0.42, -0.70, 0.72, 0.42, -0.10],
        "su": [0.15, 0.09, 0.16, 0.13, 0.12],
        "sv": [0.09, 0.15, 0.09, 0.12, 0.21],
        "opacity": [0.55, 0.50, 0.52, 0.48, 0.62],
    }

    target = render_detached(
        target_params["cx"],
        target_params["cy"],
        target_params["su"],
        target_params["sv"],
        target_params["opacity"],
        target_rays,
        opts,
        args.size,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    log = []
    pred = render_detached(
        params["cx"], params["cy"], params["su"], params["sv"], params["opacity"], target_rays, opts, args.size
    )
    initial_loss = float(np.mean((pred - target) ** 2))
    frames.append(make_frame(target, pred, 0, initial_loss, args.iterations))
    log.append({"iteration": 0, "loss": initial_loss, "params": json.loads(json.dumps(params))})

    for iteration in range(1, args.iterations + 1):
        pred, loss, grads = fit_step(params, target, fit_rays, opts, args.size)
        apply_update(params, grads, lr_pos=0.35, lr_scale=0.18, lr_opacity=0.5)
        log.append({"iteration": iteration, "loss": loss, "params": json.loads(json.dumps(params))})
        if iteration % args.frame_step == 0 or iteration == args.iterations:
            frames.append(make_frame(target, pred, iteration, loss, args.iterations))

    gif_path = output_dir / "2dgs_surfel_convergence.gif"
    final_png = output_dir / "2dgs_surfel_convergence_final.png"
    metrics_path = output_dir / "2dgs_surfel_convergence_metrics.json"

    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=90, loop=0, optimize=False)
    frames[-1].save(final_png)
    metrics_path.write_text(json.dumps(log, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "gif": str(gif_path),
                "final_png": str(final_png),
                "metrics": str(metrics_path),
                "initial_loss": log[0]["loss"],
                "final_loss": log[-1]["loss"],
                "frames": len(frames),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
