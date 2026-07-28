import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

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
import rayd.drjit as rd


def load_square_grayscale(path: Path, size: int, autocontrast: bool) -> Image.Image:
    image = Image.open(path).convert("L")
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    image = image.crop((left, top, left + side, top + side))
    if autocontrast:
        image = ImageOps.autocontrast(image)
    return image.resize((size, size), Image.Resampling.LANCZOS)


def make_grid(size: int) -> tuple[list[float], list[float], float]:
    spacing = 2.0 / size
    xs: list[float] = []
    ys: list[float] = []
    for iy in range(size):
        y = 1.0 - spacing * (iy + 0.5)
        for ix in range(size):
            x = -1.0 + spacing * (ix + 0.5)
            xs.append(x)
            ys.append(y)
    return xs, ys, spacing


def save_montage(target: Image.Image, reconstruction: Image.Image, error: Image.Image, path: Path) -> None:
    size = target.width
    panel = Image.new("RGB", (size * 3 + 32, size + 40), (20, 22, 26))
    draw = ImageDraw.Draw(panel)
    labels = ["target", "ray-traced surfel fit", "abs error x12"]
    images = [target.convert("RGB"), reconstruction.convert("RGB"), error.convert("RGB")]
    for index, (label, image) in enumerate(zip(labels, images)):
        x = index * (size + 16)
        draw.text((x, 6), label, fill=(235, 238, 245))
        panel.paste(image, (x, 32))
    panel.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit a grayscale image with one ray-traced 2DGS surfel per output pixel."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/surfel_web_fit"))
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--source-url", type=str, default="")
    parser.add_argument("--no-autocontrast", action="store_true")
    args = parser.parse_args()

    if args.size <= 0:
        raise ValueError("--size must be positive.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    target_image = load_square_grayscale(args.input, args.size, not args.no_autocontrast)
    target = np.asarray(target_image, dtype=np.float32) / 255.0

    xs, ys, spacing = make_grid(args.size)
    count = args.size * args.size
    alpha_min = 1.0 / 255.0
    opacity = target.reshape(-1).astype(np.float32)
    opacity_for_scene = np.where(opacity >= alpha_min, opacity, 0.0).astype(np.float32)

    opts = rd.SurfelTraceOptions()
    opts.alpha_min = alpha_min
    opts.cutoff = 0.49
    opts.proxy_epsilon = 0.02
    opts.primitive_mode = rd.SurfelPrimitiveMode.Icosahedron20

    cloud = rd.SurfelCloud(
        cuda.Array3f(xs, ys, [0.0] * count),
        cuda.Array3f([spacing] * count, [0.0] * count, [0.0] * count),
        cuda.Array3f([0.0] * count, [spacing] * count, [0.0] * count),
        cuda.Float(opacity_for_scene.tolist()),
    )
    scene = rd.SurfelScene(cloud, opts)
    scene.build()

    rays = rd.Ray(
        cuda.Array3f(xs, ys, [1.0] * count),
        cuda.Array3f([0.0] * count, [0.0] * count, [-1.0] * count),
    )
    its = scene.intersect(rays)
    valid = its.is_valid()
    intensity = dr.select(valid, its.opacity * its.gaussian_weight, cuda.Float([0.0] * count))
    dr.eval(valid, intensity, its.gaussian_weight)

    reconstruction = np.array([float(intensity[i]) for i in range(count)], dtype=np.float32)
    reconstruction = reconstruction.reshape(args.size, args.size)
    valid_count = sum(1 for i in range(count) if bool(valid[i]))
    error = reconstruction - target
    mse = float(np.mean(error * error))
    psnr = float("inf") if mse == 0.0 else 10.0 * math.log10(1.0 / mse)

    target_path = args.output_dir / "target_gray.png"
    reconstruction_path = args.output_dir / f"rayd_2dgs_surfel_fit_{count}.png"
    error_path = args.output_dir / "rayd_2dgs_surfel_fit_error.png"
    montage_path = args.output_dir / "rayd_2dgs_surfel_fit_montage.png"
    metrics_path = args.output_dir / "rayd_2dgs_surfel_fit_metrics.json"

    target_image.save(target_path)
    reconstruction_image = Image.fromarray(
        np.clip(reconstruction * 255.0, 0.0, 255.0).astype(np.uint8),
        mode="L",
    )
    reconstruction_image.save(reconstruction_path)
    error_image = Image.fromarray(
        np.clip(np.abs(error) * 12.0 * 255.0, 0.0, 255.0).astype(np.uint8),
        mode="L",
    )
    error_image.save(error_path)
    save_montage(target_image, reconstruction_image, error_image, montage_path)

    metrics = {
        "source_url": args.source_url,
        "source_file": str(args.input),
        "target_size": [args.size, args.size],
        "surfel_count": scene.surfel_count,
        "proxy_mode": str(rd.SurfelPrimitiveMode.Icosahedron20),
        "triangle_count": scene.triangle_count,
        "ray_count": count,
        "valid_hit_count": valid_count,
        "alpha_min": alpha_min,
        "proxy_cutoff": opts.cutoff,
        "mse": mse,
        "psnr_db": psnr,
        "outputs": {
            "target": str(target_path),
            "reconstruction": str(reconstruction_path),
            "error": str(error_path),
            "montage": str(montage_path),
        },
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
