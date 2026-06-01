import argparse
import json
import math
import os
import shutil
import sys
import urllib.request
import zipfile
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
import rayd as rd


DXGL_APPLE_URL = "https://dx.gl/api/v/EJbs8npt2RVM/vCHDLxjWG65d/dataset"
SH_Y00 = 0.28209479177387814
SH_Y1 = 0.4886025119029199


def evaluate_sh_rgb(coeffs: np.ndarray, view_dirs: np.ndarray, degree: int) -> np.ndarray:
    """Evaluate per-surfel RGB SH-like coefficients in Python.

    The basis is intentionally simple for this example:
    degree 0: [1]
    degree 1: [1, x, y, z]
    """
    if degree not in (0, 1):
        raise ValueError("Only --sh-degree 0 or 1 is supported by this example.")
    coeffs = np.asarray(coeffs, dtype=np.float32)
    view_dirs = np.asarray(view_dirs, dtype=np.float32)
    if coeffs.ndim != 3 or coeffs.shape[1] != 3:
        raise ValueError("coeffs must have shape [surfel_count, 3, basis_count].")
    if view_dirs.shape != (coeffs.shape[0], 3):
        raise ValueError("view_dirs must have shape [surfel_count, 3].")

    if degree == 0:
        basis = np.ones((coeffs.shape[0], 1), dtype=np.float32)
    else:
        basis = np.concatenate(
            [np.ones((coeffs.shape[0], 1), dtype=np.float32), view_dirs],
            axis=1,
        )
    if coeffs.shape[2] != basis.shape[1]:
        raise ValueError("SH coefficient count does not match degree.")
    return np.einsum("ncb,nb->nc", coeffs, basis, optimize=True)


def maybe_download_dxgl_apple(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_dir = output_dir / "dxgl_apple"
    transforms = scene_dir / "transforms.json"
    if transforms.is_file():
        return scene_dir

    zip_path = output_dir / "dxgl_apple.zip"
    print(f"Downloading DX.GL Apple dataset listed on Hugging Face to {zip_path} ...")
    urllib.request.urlretrieve(DXGL_APPLE_URL, zip_path)
    tmp_dir = output_dir / "dxgl_apple_extract"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(tmp_dir)

    candidates = [p for p in tmp_dir.rglob("transforms.json")]
    if not candidates:
        raise RuntimeError("Downloaded archive did not contain transforms.json.")
    extracted_scene = candidates[0].parent
    if scene_dir.exists():
        shutil.rmtree(scene_dir)
    shutil.move(str(extracted_scene), str(scene_dir))
    shutil.rmtree(tmp_dir)
    return scene_dir


def load_transforms(scene_dir: Path, max_views: int) -> dict:
    path = scene_dir / "transforms.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing transforms.json under {scene_dir}.")
    data = json.loads(path.read_text(encoding="utf-8"))
    frames = data.get("frames", [])
    if not frames:
        raise ValueError("transforms.json contains no frames.")
    normalized_frames = []
    for frame in frames[:max_views]:
        normalized = dict(frame)
        if "file_path" not in normalized and "color_path" in normalized:
            normalized["file_path"] = normalized["color_path"]
        if "transform_matrix" not in normalized and "transform" in normalized:
            # sonic-nerf stores column-major affine matrices with translation in the last row.
            normalized["transform_matrix"] = np.asarray(normalized["transform"], dtype=np.float32).T.tolist()
        normalized_frames.append(normalized)
    data["frames"] = normalized_frames
    if "camera_angle_x" not in data and "fov_x" in data:
        data["camera_angle_x"] = float(data["fov_x"])
    if "camera_angle_y" not in data and "fov_y" in data:
        data["camera_angle_y"] = float(data["fov_y"])
    return data


def resolve_frame_image(scene_dir: Path, frame: dict) -> Path:
    rel = frame.get("file_path") or frame.get("image_path")
    if not rel:
        raise ValueError("Frame is missing file_path/image_path.")
    path = scene_dir / rel
    if path.suffix == "":
        for suffix in (".png", ".jpg", ".jpeg", ".JPG"):
            candidate = path.with_suffix(suffix)
            if candidate.is_file():
                return candidate
    if not path.is_file():
        raise FileNotFoundError(f"Missing frame image: {path}")
    return path


def load_target_image(path: Path, size: int, background: float) -> tuple[np.ndarray, np.ndarray]:
    image = Image.open(path).convert("RGBA").resize((size, size), Image.Resampling.LANCZOS)
    rgba = np.asarray(image, dtype=np.float32) / 255.0
    alpha = rgba[..., 3:4]
    rgb = rgba[..., :3] * alpha + background * (1.0 - alpha)
    if np.all(alpha[..., 0] > 0.99):
        mask = np.linalg.norm(rgb - background, axis=2) > 0.03
    else:
        mask = alpha[..., 0] > 0.01
    return rgb.reshape(-1, 3).astype(np.float32), mask.reshape(-1).astype(np.float32)


def camera_intrinsics(transforms: dict, size: int) -> tuple[float, float, float, float]:
    source_w = float(transforms.get("w", size))
    source_h = float(transforms.get("h", size))
    scale_x = size / source_w
    scale_y = size / source_h
    if "fl_x" in transforms and "fl_y" in transforms:
        fx = float(transforms["fl_x"]) * scale_x
        fy = float(transforms["fl_y"]) * scale_y
    else:
        angle_x = float(transforms.get("camera_angle_x", math.radians(50.0)))
        angle_y = float(transforms.get("camera_angle_y", angle_x))
        fx = 0.5 * size / math.tan(0.5 * angle_x)
        fy = 0.5 * size / math.tan(0.5 * angle_y)
    cx = float(transforms.get("cx", source_w * 0.5)) * scale_x
    cy = float(transforms.get("cy", source_h * 0.5)) * scale_y
    return fx, fy, cx, cy


def make_camera_rays(transform: np.ndarray,
                     fx: float,
                     fy: float,
                     cx: float,
                     cy: float,
                     size: int,
                     ad_mode: bool):
    origins = np.repeat(transform[:3, 3][None, :], size * size, axis=0)
    dirs = []
    rotation = transform[:3, :3]
    for iy in range(size):
        for ix in range(size):
            cam_dir = np.array([(ix + 0.5 - cx) / fx, -(iy + 0.5 - cy) / fy, -1.0], dtype=np.float32)
            world_dir = rotation @ cam_dir
            world_dir /= max(np.linalg.norm(world_dir), 1e-8)
            dirs.append(world_dir)
    directions = np.asarray(dirs, dtype=np.float32)

    array3 = ad.Array3f if ad_mode else cuda.Array3f
    scalar = ad.Float if ad_mode else cuda.Float
    return (rd.RayAD if ad_mode else rd.Ray)(
        array3(
            scalar(origins[:, 0].tolist()),
            scalar(origins[:, 1].tolist()),
            scalar(origins[:, 2].tolist()),
        ),
        array3(
            scalar(directions[:, 0].tolist()),
            scalar(directions[:, 1].tolist()),
            scalar(directions[:, 2].tolist()),
        ),
    )


def ply_dtype(properties: list[tuple[str, str]]) -> np.dtype:
    dtype_map = {
        "char": "i1",
        "uchar": "u1",
        "uint8": "u1",
        "short": "<i2",
        "ushort": "<u2",
        "int": "<i4",
        "uint": "<u4",
        "float": "<f4",
        "float32": "<f4",
        "double": "<f8",
        "float64": "<f8",
    }
    return np.dtype([(name, dtype_map[kind]) for kind, name in properties])


def read_ply_xyz_rgb(path: Path, max_points: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    data = path.read_bytes()
    marker = b"end_header\n"
    header_end = data.find(marker)
    if header_end < 0:
        marker = b"end_header\r\n"
        header_end = data.find(marker)
    if header_end < 0:
        raise ValueError(f"Invalid PLY header: {path}")
    header_end += len(marker)
    header = data[:header_end].decode("ascii", errors="replace").splitlines()
    fmt = ""
    vertex_count = 0
    properties: list[tuple[str, str]] = []
    in_vertex = False
    for line in header:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[:1] == ["element"]:
            in_vertex = len(parts) >= 3 and parts[1] == "vertex"
            if in_vertex:
                vertex_count = int(parts[2])
        elif in_vertex and parts[0] == "property" and len(parts) >= 3 and parts[1] != "list":
            properties.append((parts[1], parts[2]))

    if vertex_count <= 0:
        raise ValueError(f"PLY contains no vertices: {path}")
    if fmt == "ascii":
        rows = np.loadtxt(path, skiprows=len(header), max_rows=vertex_count)
        prop_names = [name for _, name in properties]
        table = {name: rows[:, index] for index, name in enumerate(prop_names)}
    elif fmt == "binary_little_endian":
        arr = np.frombuffer(data, dtype=ply_dtype(properties), count=vertex_count, offset=header_end)
        table = {name: arr[name] for _, name in properties}
    else:
        raise ValueError(f"Unsupported PLY format '{fmt}' in {path}.")

    xyz = np.stack([table["x"], table["y"], table["z"]], axis=1).astype(np.float32)
    if all(name in table for name in ("red", "green", "blue")):
        rgb = np.stack([table["red"], table["green"], table["blue"]], axis=1).astype(np.float32)
        if rgb.max(initial=0.0) > 1.0:
            rgb /= 255.0
    else:
        rgb = np.full((xyz.shape[0], 3), 0.5, dtype=np.float32)

    if xyz.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        keep = rng.choice(xyz.shape[0], size=max_points, replace=False)
        xyz = xyz[keep]
        rgb = rgb[keep]
    return xyz, np.clip(rgb, 0.0, 1.0).astype(np.float32)


def fallback_sphere(max_points: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(max_points, dtype=np.float32)
    phi = math.pi * (3.0 - math.sqrt(5.0))
    y = 1.0 - 2.0 * (indices + 0.5) / max_points
    radius = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    theta = phi * indices
    xyz = np.stack([np.cos(theta) * radius, y, np.sin(theta) * radius], axis=1).astype(np.float32)
    rgb = np.stack([0.5 + 0.5 * xyz[:, 0], 0.5 + 0.5 * xyz[:, 1], 0.5 + 0.5 * xyz[:, 2]], axis=1)
    return xyz, np.clip(rgb, 0.0, 1.0).astype(np.float32)


def tangent_basis(centers: np.ndarray, scale_multiplier: float) -> tuple[np.ndarray, np.ndarray]:
    centroid = centers.mean(axis=0, keepdims=True)
    normals = centers - centroid
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.where(norms > 1e-6, normals / np.maximum(norms, 1e-6), np.array([[0.0, 0.0, 1.0]], dtype=np.float32))
    helper = np.tile(np.array([[0.0, 1.0, 0.0]], dtype=np.float32), (centers.shape[0], 1))
    near_parallel = np.abs(np.sum(helper * normals, axis=1)) > 0.9
    helper[near_parallel] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    tangent_u = np.cross(helper, normals)
    tangent_u /= np.maximum(np.linalg.norm(tangent_u, axis=1, keepdims=True), 1e-6)
    tangent_v = np.cross(normals, tangent_u)

    bbox = centers.max(axis=0) - centers.min(axis=0)
    base_scale = max(float(np.linalg.norm(bbox)) / math.sqrt(max(1, centers.shape[0])), 1e-3)
    scale = base_scale * scale_multiplier
    return (tangent_u * scale).astype(np.float32), (tangent_v * scale).astype(np.float32)


def array3_cuda(values: np.ndarray):
    return cuda.Array3f(
        cuda.Float(values[:, 0].tolist()),
        cuda.Float(values[:, 1].tolist()),
        cuda.Float(values[:, 2].tolist()),
    )


def array3_ad(values: np.ndarray):
    return ad.Array3f(
        ad.Float(values[:, 0].tolist()),
        ad.Float(values[:, 1].tolist()),
        ad.Float(values[:, 2].tolist()),
    )


def native_sh_flat_from_coeffs(coeffs: np.ndarray) -> np.ndarray:
    coeffs = np.asarray(coeffs, dtype=np.float32)
    if coeffs.ndim != 3 or coeffs.shape[1] != 3:
        raise ValueError("coeffs must have shape [surfel_count, 3, basis_count].")
    if coeffs.shape[2] == 1:
        native = np.zeros((coeffs.shape[0], 1, 3), dtype=np.float32)
        native[:, 0, :] = coeffs[:, :, 0] / SH_Y00
    elif coeffs.shape[2] == 4:
        native = np.zeros((coeffs.shape[0], 4, 3), dtype=np.float32)
        native[:, 0, :] = coeffs[:, :, 0] / SH_Y00
        native[:, 1, :] = coeffs[:, :, 2] / SH_Y1
        native[:, 2, :] = coeffs[:, :, 3] / SH_Y1
        native[:, 3, :] = coeffs[:, :, 1] / SH_Y1
    else:
        raise ValueError("Only degree-0 and degree-1 SH coefficients are supported.")
    return native.reshape(-1)


def coeff_grad_from_native_flat(grad_flat: np.ndarray, surfel_count: int, basis_count: int) -> np.ndarray:
    native = grad_flat.reshape(surfel_count, basis_count, 3)
    grad = np.zeros((surfel_count, 3, basis_count), dtype=np.float32)
    grad[:, :, 0] = native[:, 0, :] / SH_Y00
    if basis_count == 4:
        grad[:, :, 2] = native[:, 1, :] / SH_Y1
        grad[:, :, 3] = native[:, 2, :] / SH_Y1
        grad[:, :, 1] = native[:, 3, :] / SH_Y1
    return grad


def make_geometry(centers: np.ndarray, tangent_u: np.ndarray, tangent_v: np.ndarray, ad_mode: bool = False):
    array3 = array3_ad if ad_mode else array3_cuda
    return rd.SurfelGeometry(array3(centers), array3(tangent_u), array3(tangent_v))


def make_appearance(opacity, coeffs: np.ndarray, degree: int, ad_mode: bool = False):
    if degree == 0:
        rgb = coeffs[:, :, 0]
        return rd.SurfelAppearance.rgb(opacity, array3_ad(rgb) if ad_mode else array3_cuda(rgb))
    flat = native_sh_flat_from_coeffs(coeffs)
    values = ad.Float(flat.tolist()) if ad_mode else cuda.Float(flat.tolist())
    return rd.SurfelAppearance.sh(opacity, values, degree)


def make_options(max_candidate_hits: int) -> rd.SurfelTraceOptions:
    opts = rd.SurfelTraceOptions()
    opts.alpha_min = 1.0 / 255.0
    opts.alpha_cap = 0.99
    opts.max_candidate_hits = max_candidate_hits
    opts.primitive_mode = rd.SurfelPrimitiveMode.Icosahedron20
    opts.single_launch = True
    return opts


def view_dirs_to_camera(centers: np.ndarray, camera_origin: np.ndarray) -> np.ndarray:
    dirs = camera_origin[None, :] - centers
    dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-6)
    return dirs.astype(np.float32)


def render_rgb(centers: np.ndarray,
               tangent_u: np.ndarray,
               tangent_v: np.ndarray,
               opacity: np.ndarray,
               coeffs: np.ndarray,
               degree: int,
               rays,
               camera_origin: np.ndarray,
               opts: rd.SurfelTraceOptions,
               size: int) -> np.ndarray:
    del camera_origin
    scene = rd.SurfelScene(make_geometry(centers, tangent_u, tangent_v), opts)
    scene.build()
    scene.update_appearance(make_appearance(cuda.Float(opacity.tolist()), coeffs, degree))
    out = scene.render(rays, rd.SurfelRenderOptions.rgb(sh_degree=degree))
    dr.eval(out.rgb)
    channels = [
        np.array([float(out.rgb[channel][i]) for i in range(size * size)], dtype=np.float32)
        for channel in range(3)
    ]
    return np.stack(channels, axis=1).reshape(size, size, 3)


def fit_color_coefficients(centers: np.ndarray,
                           tangent_u: np.ndarray,
                           tangent_v: np.ndarray,
                           opacity: np.ndarray,
                           coeffs: np.ndarray,
                           views: list[dict],
                           opts: rd.SurfelTraceOptions,
                           iterations: int,
                           lr: float,
                           frame_callback=None,
                           video_every: int = 1) -> list[dict]:
    center_ad = array3_ad(centers)
    tangent_u_ad = array3_ad(tangent_u)
    tangent_v_ad = array3_ad(tangent_v)
    opacity_ad = ad.Float(opacity.tolist())
    scene = rd.SurfelScene(rd.SurfelGeometry(center_ad, tangent_u_ad, tangent_v_ad), opts)
    scene.build()
    log = []
    for iteration in range(1, iterations + 1):
        grad_coeffs = np.zeros_like(coeffs)
        losses = []
        for view in views:
            target = view["target"]
            mask = view["mask"]
            denom = max(float(mask.sum()), 1.0)
            if coeffs.shape[2] == 1:
                rgb_ad = array3_ad(coeffs[:, :, 0])
                dr.enable_grad(rgb_ad)
                scene.update_appearance(rd.SurfelAppearance.rgb(opacity_ad, rgb_ad))
                active_coeff = rgb_ad
            else:
                flat_values = ad.Float(native_sh_flat_from_coeffs(coeffs).tolist())
                dr.enable_grad(flat_values)
                scene.update_appearance(rd.SurfelAppearance.sh(opacity_ad, flat_values, 1))
                active_coeff = flat_values

            pred = scene.render(view["rays_ad"], rd.SurfelRenderOptions.rgb(sh_degree=0 if coeffs.shape[2] == 1 else 1)).rgb
            residual = (
                (pred[0] - ad.Float(target[:, 0].tolist())) * (pred[0] - ad.Float(target[:, 0].tolist())) +
                (pred[1] - ad.Float(target[:, 1].tolist())) * (pred[1] - ad.Float(target[:, 1].tolist())) +
                (pred[2] - ad.Float(target[:, 2].tolist())) * (pred[2] - ad.Float(target[:, 2].tolist()))
            ) * ad.Float(mask.tolist())
            loss = dr.sum(residual) / denom
            dr.backward(loss)
            losses.append(float(loss[0]))

            if coeffs.shape[2] == 1:
                grad_rgb = dr.grad(active_coeff)
                dr.eval(grad_rgb)
                grad_coeffs[:, 0, 0] += np.array([float(grad_rgb[0][i]) for i in range(centers.shape[0])], dtype=np.float32)
                grad_coeffs[:, 1, 0] += np.array([float(grad_rgb[1][i]) for i in range(centers.shape[0])], dtype=np.float32)
                grad_coeffs[:, 2, 0] += np.array([float(grad_rgb[2][i]) for i in range(centers.shape[0])], dtype=np.float32)
            else:
                grad_flat = dr.grad(active_coeff)
                dr.eval(grad_flat)
                grad_np = np.array([float(grad_flat[i]) for i in range(len(grad_flat))], dtype=np.float32)
                grad_coeffs += coeff_grad_from_native_flat(grad_np, centers.shape[0], coeffs.shape[2])

        coeffs -= lr * grad_coeffs / max(1, len(views))
        if coeffs.shape[2] == 1:
            coeffs[:, :, 0] = np.clip(coeffs[:, :, 0], -0.25, 1.25)
        log.append({"iteration": iteration, "loss": float(np.mean(losses))})
        print(f"iteration {iteration:04d}: loss={log[-1]['loss']:.6f}")
        if frame_callback is not None and (iteration % video_every == 0 or iteration == iterations):
            frame_callback(iteration, log[-1]["loss"])
    return log


def make_convergence_frame(target: np.ndarray, pred: np.ndarray, iteration: int, loss: float) -> Image.Image:
    size = target.shape[0]
    error = np.abs(pred - target)
    panels = [
        ("target", target),
        ("prediction", np.clip(pred, 0.0, 1.0)),
        ("abs error x4", np.clip(error * 4.0, 0.0, 1.0)),
    ]
    canvas = Image.new("RGB", (size * 3 + 32, size + 60), (20, 22, 26))
    draw = ImageDraw.Draw(canvas)
    draw.text((0, 6), f"RayD surfel multiview color fit   iter={iteration:04d}   loss={loss:.6f}", fill=(235, 238, 245))
    for index, (label, image) in enumerate(panels):
        x = index * (size + 16)
        draw.text((x, 27), label, fill=(200, 210, 225))
        panel = Image.fromarray((image * 255.0).astype(np.uint8), mode="RGB")
        canvas.paste(panel, (x, 56))
    return canvas


def save_montage(target: np.ndarray, pred: np.ndarray, path: Path) -> None:
    size = target.shape[0]
    error = np.abs(pred - target)
    panels = [
        ("target", target),
        ("prediction", np.clip(pred, 0.0, 1.0)),
        ("abs error x4", np.clip(error * 4.0, 0.0, 1.0)),
    ]
    canvas = Image.new("RGB", (size * 3 + 32, size + 40), (20, 22, 26))
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(panels):
        x = index * (size + 16)
        draw.text((x, 7), label, fill=(235, 238, 245))
        panel = Image.fromarray((image * 255.0).astype(np.uint8), mode="RGB")
        canvas.paste(panel, (x, 32))
    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit per-surfel RGB/degree-1 SH color from posed multi-view images using RayD surfel ray tracing."
    )
    parser.add_argument("--scene-dir", type=Path, default=None,
                        help="Dataset directory containing transforms.json, images/, and optionally points3D.ply.")
    parser.add_argument("--download-dxgl-apple", action="store_true",
                        help="Download the DX.GL Apple sample referenced by Hugging Face dxgl/multiview-datasets.")
    parser.add_argument("--download-dir", type=Path, default=Path("artifacts/datasets"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/surfel_multiview_color_fit"))
    parser.add_argument("--size", type=int, default=48)
    parser.add_argument("--views", type=int, default=3)
    parser.add_argument("--surfels", type=int, default=2048)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.25)
    parser.add_argument("--sh-degree", type=int, choices=[0, 1], default=0)
    parser.add_argument("--scale-multiplier", type=float, default=1.8)
    parser.add_argument("--opacity", type=float, default=0.65)
    parser.add_argument("--background", type=float, default=1.0)
    parser.add_argument("--max-candidate-hits", type=int, default=8)
    parser.add_argument("--video-every", type=int, default=1,
                        help="Save one convergence GIF frame every N iterations.")
    parser.add_argument("--gif-duration-ms", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if args.download_dxgl_apple:
        scene_dir = maybe_download_dxgl_apple(args.download_dir)
    elif args.scene_dir is not None:
        scene_dir = args.scene_dir
    else:
        raise SystemExit("Pass --scene-dir or --download-dxgl-apple.")

    transforms = load_transforms(scene_dir, args.views)
    fx, fy, cx, cy = camera_intrinsics(transforms, args.size)
    ply_path = scene_dir / transforms.get("ply_file_path", "points3D.ply")
    if ply_path.is_file():
        centers, initial_rgb = read_ply_xyz_rgb(ply_path, args.surfels, args.seed)
    else:
        print(f"No point cloud found at {ply_path}; using a synthetic sphere initializer.")
        centers, initial_rgb = fallback_sphere(args.surfels)

    tangent_u, tangent_v = tangent_basis(centers, args.scale_multiplier)
    opacity = np.full((centers.shape[0],), args.opacity, dtype=np.float32)
    basis_count = 1 if args.sh_degree == 0 else 4
    coeffs = np.zeros((centers.shape[0], 3, basis_count), dtype=np.float32)
    coeffs[:, :, 0] = initial_rgb

    views = []
    for frame in transforms["frames"]:
        transform = np.asarray(frame["transform_matrix"], dtype=np.float32)
        image_path = resolve_frame_image(scene_dir, frame)
        target, mask = load_target_image(image_path, args.size, args.background)
        rays_ad = make_camera_rays(transform, fx, fy, cx, cy, args.size, ad_mode=True)
        rays_detached = make_camera_rays(transform, fx, fy, cx, cy, args.size, ad_mode=False)
        camera_origin = transform[:3, 3].astype(np.float32)
        views.append({
            "image_path": str(image_path),
            "target": target,
            "target_image": target.reshape(args.size, args.size, 3),
            "mask": mask,
            "rays_ad": rays_ad,
            "rays_detached": rays_detached,
            "camera_origin": camera_origin,
            "view_dirs": view_dirs_to_camera(centers, camera_origin),
        })

    opts = make_options(args.max_candidate_hits)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    first_view = views[0]
    frames: list[Image.Image] = []

    def append_frame(iteration: int, loss: float) -> None:
        pred_frame = render_rgb(
            centers,
            tangent_u,
            tangent_v,
            opacity,
            coeffs,
            args.sh_degree,
            first_view["rays_detached"],
            first_view["camera_origin"],
            opts,
            args.size,
        )
        frames.append(make_convergence_frame(first_view["target_image"], pred_frame, iteration, loss))

    initial_pred = render_rgb(
        centers,
        tangent_u,
        tangent_v,
        opacity,
        coeffs,
        args.sh_degree,
        first_view["rays_detached"],
        first_view["camera_origin"],
        opts,
        args.size,
    )
    initial_loss = float(np.mean((initial_pred - first_view["target_image"]) ** 2))
    frames.append(make_convergence_frame(first_view["target_image"], initial_pred, 0, initial_loss))
    log = fit_color_coefficients(
        centers,
        tangent_u,
        tangent_v,
        opacity,
        coeffs,
        views,
        opts,
        args.iterations,
        args.lr,
        append_frame,
        max(1, args.video_every),
    )

    pred = render_rgb(
        centers,
        tangent_u,
        tangent_v,
        opacity,
        coeffs,
        args.sh_degree,
        first_view["rays_detached"],
        first_view["camera_origin"],
        opts,
        args.size,
    )
    montage_path = args.output_dir / "surfel_multiview_color_fit_montage.png"
    video_path = args.output_dir / "surfel_multiview_color_fit_convergence.gif"
    coeff_path = args.output_dir / "surfel_color_sh_coefficients.npz"
    metrics_path = args.output_dir / "surfel_multiview_color_fit_metrics.json"
    save_montage(first_view["target_image"], pred, montage_path)
    if frames:
        frames[0].save(
            video_path,
            save_all=True,
            append_images=frames[1:],
            duration=args.gif_duration_ms,
            loop=0,
            optimize=False,
        )
    np.savez_compressed(coeff_path, centers=centers, tangent_u=tangent_u, tangent_v=tangent_v, opacity=opacity, coeffs=coeffs)

    metrics = {
        "dataset": str(scene_dir),
        "dataset_reference": "https://huggingface.co/datasets/dxgl/multiview-datasets",
        "views": [view["image_path"] for view in views],
        "surfel_count": int(centers.shape[0]),
        "image_size": args.size,
        "sh_degree": args.sh_degree,
        "iterations": args.iterations,
        "final_loss": log[-1]["loss"] if log else None,
        "initial_loss": initial_loss,
        "outputs": {
            "montage": str(montage_path),
            "video": str(video_path),
            "coefficients": str(coeff_path),
            "metrics": str(metrics_path),
        },
        "notes": [
            "The scene/GAS is built once; color changes use SurfelScene.update_appearance().",
            "RGB and degree-1 SH color are rendered through RayD's native surfel render() path.",
        ],
    }
    metrics_path.write_text(json.dumps({"metrics": metrics, "log": log}, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
