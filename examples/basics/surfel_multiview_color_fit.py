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
SONIC_NERF_BASE_URL = (
    "https://huggingface.co/datasets/hayden-donnelly/sonic-nerf/resolve/main/multi_view_renders"
)
SONIC_NERF_REFERENCE = "https://huggingface.co/datasets/hayden-donnelly/sonic-nerf"
DXGL_REFERENCE = "https://huggingface.co/datasets/dxgl/multiview-datasets"
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


def download_url_if_missing(url: str, path: Path) -> None:
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {path}")
    urllib.request.urlretrieve(url, path)


def maybe_download_sonic_nerf(output_dir: Path, max_views: int) -> Path:
    scene_dir = output_dir / "sonic_nerf" / "multi_view_renders"
    scene_dir.mkdir(parents=True, exist_ok=True)
    transforms_path = scene_dir / "transforms.json"
    download_url_if_missing(f"{SONIC_NERF_BASE_URL}/transforms.json", transforms_path)

    data = json.loads(transforms_path.read_text(encoding="utf-8"))
    for frame in data.get("frames", [])[:max_views]:
        rel = frame.get("color_path")
        if rel:
            download_url_if_missing(f"{SONIC_NERF_BASE_URL}/{rel}", scene_dir / rel)
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


def select_loss_mask(foreground_mask: np.ndarray, mode: str) -> np.ndarray:
    if mode == "full":
        return np.ones_like(foreground_mask, dtype=np.float32)
    if mode == "foreground":
        return foreground_mask.astype(np.float32)
    raise ValueError("--loss-mask must be 'full' or 'foreground'.")


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
    origins, directions = camera_ray_arrays(transform, fx, fy, cx, cy, size)

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


def camera_ray_arrays(transform: np.ndarray,
                      fx: float,
                      fy: float,
                      cx: float,
                      cy: float,
                      size: int) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = np.meshgrid(np.arange(size, dtype=np.float32) + 0.5,
                         np.arange(size, dtype=np.float32) + 0.5)
    cam_dirs = np.stack([(xs - cx) / fx,
                         -(ys - cy) / fy,
                         -np.ones_like(xs)], axis=-1).reshape(-1, 3)
    directions = cam_dirs @ np.asarray(transform[:3, :3], dtype=np.float32).T
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-8)
    origins = np.repeat(np.asarray(transform[:3, 3], dtype=np.float32)[None, :],
                        size * size,
                        axis=0)
    return origins.astype(np.float32), directions.astype(np.float32)


def normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-6)


def random_tangent_basis(rng: np.random.Generator,
                         count: int,
                         scale: float) -> tuple[np.ndarray, np.ndarray]:
    normals = normalize_rows(rng.normal(size=(count, 3)).astype(np.float32))
    helper = np.tile(np.array([[0.0, 1.0, 0.0]], dtype=np.float32), (count, 1))
    near_parallel = np.abs(np.sum(helper * normals, axis=1)) > 0.9
    helper[near_parallel] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    tangent_u = normalize_rows(np.cross(helper, normals).astype(np.float32))
    tangent_v = normalize_rows(np.cross(normals, tangent_u).astype(np.float32))
    scale_jitter = rng.uniform(0.75, 1.25, size=(count, 1)).astype(np.float32)
    return (tangent_u * scale * scale_jitter).astype(np.float32), (tangent_v * scale * scale_jitter).astype(np.float32)


def initialize_random_surfel_field(args) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    count = int(args.surfels)
    rng = np.random.default_rng(int(args.seed))
    radius = float(args.random_radius)
    scale = float(args.initial_scale)
    centers = rng.uniform(-radius, radius, size=(count, 3)).astype(np.float32)
    tangent_u, tangent_v = random_tangent_basis(rng, count, scale)
    colors = rng.uniform(0.2, 0.8, size=(count, 3)).astype(np.float32)
    info = {
        "source": "random",
        "sample_count": count,
        "random_radius": radius,
        "initial_scale": scale,
    }
    return centers, colors, tangent_u, tangent_v, info


def values_per_surfel(degree: int) -> int:
    if degree == 0:
        return 3
    if degree == 1:
        return 12
    raise ValueError("Only --sh-degree 0 or 1 is supported by this example.")


def surfel_scales(tangent_u: np.ndarray, tangent_v: np.ndarray) -> np.ndarray:
    scale_u = np.linalg.norm(np.asarray(tangent_u, dtype=np.float32), axis=1)
    scale_v = np.linalg.norm(np.asarray(tangent_v, dtype=np.float32), axis=1)
    return ((scale_u + scale_v) * 0.5).astype(np.float32)


def scheduled_learning_rate(iteration: int,
                            base_lr: float,
                            final_lr: float,
                            warmup_iters: int,
                            total_iters: int) -> float:
    if total_iters <= 1:
        return float(final_lr)
    if warmup_iters > 0 and iteration <= warmup_iters:
        start_lr = float(base_lr) * 0.1
        alpha = max(0.0, min(1.0, iteration / float(warmup_iters)))
        return float(start_lr + (float(base_lr) - start_lr) * alpha)
    decay_start = max(1, int(warmup_iters))
    alpha = (iteration - decay_start) / max(1.0, float(total_iters - decay_start))
    alpha = max(0.0, min(1.0, alpha))
    if base_lr <= 0.0 or final_lr <= 0.0:
        return float(base_lr + (final_lr - base_lr) * alpha)
    return float(base_lr * ((final_lr / base_lr) ** alpha))


def effective_densify_until_iter(args) -> int:
    if int(args.densify_until_iter) > 0:
        return int(args.densify_until_iter)
    return max(int(args.densify_from_iter), int(round(float(args.iterations) * 0.8)))


def densify_and_prune_surfel_arrays(centers: np.ndarray,
                                    tangent_u: np.ndarray,
                                    tangent_v: np.ndarray,
                                    opacity: np.ndarray,
                                    values: np.ndarray,
                                    degree: int,
                                    grad_norm: np.ndarray,
                                    args,
                                    seed: int) -> dict:
    centers = np.asarray(centers, dtype=np.float32).reshape(-1, 3)
    tangent_u = np.asarray(tangent_u, dtype=np.float32).reshape(-1, 3)
    tangent_v = np.asarray(tangent_v, dtype=np.float32).reshape(-1, 3)
    opacity = np.asarray(opacity, dtype=np.float32).reshape(-1)
    grad_norm = np.asarray(grad_norm, dtype=np.float32).reshape(-1)
    per_surfel = values_per_surfel(degree)
    values_2d = np.asarray(values, dtype=np.float32).reshape(centers.shape[0], per_surfel)

    scale = surfel_scales(tangent_u, tangent_v)
    keep = (
        (opacity >= float(args.prune_opacity_threshold))
        & (scale >= float(args.min_scale))
        & (scale <= float(args.max_scale))
    )
    pruned = int(np.count_nonzero(~keep))
    if not np.any(keep):
        strongest = int(np.argmax(np.maximum(grad_norm, opacity)))
        keep[strongest] = True
        pruned -= 1

    centers = centers[keep]
    tangent_u = tangent_u[keep]
    tangent_v = tangent_v[keep]
    opacity = opacity[keep]
    values_2d = values_2d[keep]
    grad_norm = grad_norm[keep]
    scale = scale[keep]

    max_surfels = int(args.max_surfels)
    if max_surfels <= 0:
        max_surfels = centers.shape[0]
    budget = max(0, max_surfels - centers.shape[0])
    budget = min(budget, max(0, int(args.max_new_surfels_per_refine)))

    clone_centers = []
    clone_u = []
    clone_v = []
    clone_opacity = []
    clone_values = []
    cloned = 0
    split = 0
    if budget > 0:
        rng = np.random.default_rng(int(seed))
        candidates = np.flatnonzero(grad_norm >= float(args.densify_grad_threshold))
        if candidates.size:
            order = candidates[np.argsort(-grad_norm[candidates])]
            for index in order:
                if len(clone_centers) >= budget:
                    break
                offset_u = rng.normal(0.0, 0.35)
                offset_v = rng.normal(0.0, 0.35)
                jitter = (tangent_u[index] * offset_u + tangent_v[index] * offset_v).astype(np.float32)
                if scale[index] >= float(args.split_scale_threshold):
                    shrink = float(args.split_scale_shrink)
                    tangent_u[index] *= shrink
                    tangent_v[index] *= shrink
                    child_u = tangent_u[index].copy()
                    child_v = tangent_v[index].copy()
                    child_center = centers[index] + jitter
                    split += 1
                else:
                    child_u = tangent_u[index].copy()
                    child_v = tangent_v[index].copy()
                    child_center = centers[index] + 0.5 * jitter
                    cloned += 1
                parent_opacity = float(opacity[index])
                opacity[index] = np.float32(parent_opacity * 0.5)
                clone_centers.append(child_center.astype(np.float32))
                clone_u.append(child_u.astype(np.float32))
                clone_v.append(child_v.astype(np.float32))
                clone_opacity.append(parent_opacity * 0.5)
                clone_values.append(values_2d[index].copy())

    if clone_centers:
        centers = np.concatenate([centers, np.stack(clone_centers, axis=0)], axis=0)
        tangent_u = np.concatenate([tangent_u, np.stack(clone_u, axis=0)], axis=0)
        tangent_v = np.concatenate([tangent_v, np.stack(clone_v, axis=0)], axis=0)
        opacity = np.concatenate([opacity, np.asarray(clone_opacity, dtype=np.float32)], axis=0)
        values_2d = np.concatenate([values_2d, np.stack(clone_values, axis=0)], axis=0)

    stats = {
        "pruned": int(pruned),
        "cloned": int(cloned),
        "split": int(split),
        "surfel_count": int(centers.shape[0]),
    }
    return {
        "centers": centers.astype(np.float32),
        "tangent_u": tangent_u.astype(np.float32),
        "tangent_v": tangent_v.astype(np.float32),
        "opacity": opacity.astype(np.float32),
        "values": values_2d.reshape(-1).astype(np.float32),
        "stats": stats,
    }


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


def native_flat_from_initial_rgb(initial_rgb: np.ndarray, degree: int) -> np.ndarray:
    rgb = np.asarray(initial_rgb, dtype=np.float32)
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError("initial_rgb must have shape [surfel_count, 3].")
    if degree == 0:
        return rgb.reshape(-1)
    coeffs = np.zeros((rgb.shape[0], 3, 4), dtype=np.float32)
    coeffs[:, :, 0] = rgb
    return native_sh_flat_from_coeffs(coeffs)


def native_flat_to_coeffs(values: np.ndarray, surfel_count: int, degree: int) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float32)
    if degree == 0:
        return flat.reshape(surfel_count, 3, 1)
    native = flat.reshape(surfel_count, 4, 3)
    coeffs = np.zeros((surfel_count, 3, 4), dtype=np.float32)
    coeffs[:, :, 0] = native[:, 0, :] * SH_Y00
    coeffs[:, :, 2] = native[:, 1, :] * SH_Y1
    coeffs[:, :, 3] = native[:, 2, :] * SH_Y1
    coeffs[:, :, 1] = native[:, 3, :] * SH_Y1
    return coeffs


def make_appearance_from_native_values(opacity, values, degree: int):
    if degree == 0:
        return rd.SurfelAppearance.features(opacity, values, 3)
    return rd.SurfelAppearance.sh(opacity, values, degree)


def make_options(max_candidate_hits: int) -> rd.SurfelTraceOptions:
    opts = rd.SurfelTraceOptions()
    opts.alpha_min = 1.0 / 255.0
    opts.alpha_cap = 0.99
    opts.max_candidate_hits = max_candidate_hits
    opts.primitive_mode = rd.SurfelPrimitiveMode.Icosahedron20
    opts.single_launch = True
    return opts


def render_rgb(centers: np.ndarray,
               tangent_u: np.ndarray,
               tangent_v: np.ndarray,
               opacity: np.ndarray,
               coeffs: np.ndarray,
               degree: int,
               rays,
               camera_origin: np.ndarray,
               opts: rd.SurfelTraceOptions,
               size: int,
               background: float = 0.0) -> np.ndarray:
    del camera_origin
    state = GpuSurfelFitState(
        centers,
        tangent_u,
        tangent_v,
        opacity,
        native_sh_flat_from_coeffs(coeffs) if degree > 0 else coeffs[:, :, 0].reshape(-1),
        degree,
        opts,
        optimizer="sgd",
        background=background,
        fit_opacity=False,
        fit_geometry=False,
        geometry_lr_scale=0.0,
        center_bound=1.0,
        tangent_min=1e-4,
        tangent_max=1.0,
    )
    return state.render_preview(rays, size)


def target_ad_arrays(target: np.ndarray, mask: np.ndarray) -> tuple[ad.Array3f, ad.Float, float]:
    target_ad = ad.Array3f(
        ad.Float(target[:, 0].tolist()),
        ad.Float(target[:, 1].tolist()),
        ad.Float(target[:, 2].tolist()),
    )
    mask_ad = ad.Float(mask.tolist())
    return target_ad, mask_ad, max(float(mask.sum()), 1.0)


class GpuSurfelFitState:
    def __init__(self,
                 centers: np.ndarray,
                 tangent_u: np.ndarray,
                 tangent_v: np.ndarray,
                 opacity: np.ndarray,
                 initial_values: np.ndarray,
                 degree: int,
                 opts: rd.SurfelTraceOptions,
                 optimizer: str,
                 background: float,
                 fit_opacity: bool,
                 fit_geometry: bool,
                 geometry_lr_scale: float,
                 center_bound: float,
                 tangent_min: float,
                 tangent_max: float,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 eps: float = 1e-8,
                 train_build_count_base: int = 0,
                 preview_build_count_base: int = 0):
        self.surfel_count = int(centers.shape[0])
        self.degree = degree
        self.opts = opts
        self.optimizer = optimizer
        self.background = float(background)
        self.fit_opacity = fit_opacity
        self.fit_geometry = fit_geometry
        self.geometry_lr_scale = float(geometry_lr_scale)
        self.center_bound = float(center_bound)
        self.tangent_min = float(tangent_min)
        self.tangent_max = float(tangent_max)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.step_index = 0
        self.train_build_count = int(train_build_count_base)
        self.preview_build_count = int(preview_build_count_base)
        self.render_options = rd.SurfelRenderOptions.rgb(
            sh_degree=self.degree,
            background_rgb=[self.background, self.background, self.background],
        )

        self.center_values = array3_ad(centers)
        self.tangent_u_values = array3_ad(tangent_u)
        self.tangent_v_values = array3_ad(tangent_v)
        self.center_momentum = dr.detach(self.center_values * 0.0)
        self.center_velocity = dr.detach(self.center_values * 0.0)
        self.tangent_u_momentum = dr.detach(self.tangent_u_values * 0.0)
        self.tangent_u_velocity = dr.detach(self.tangent_u_values * 0.0)
        self.tangent_v_momentum = dr.detach(self.tangent_v_values * 0.0)
        self.tangent_v_velocity = dr.detach(self.tangent_v_values * 0.0)
        self.opacity_values = ad.Float(opacity.tolist())
        self.opacity_momentum = dr.detach(self.opacity_values * 0.0)
        self.opacity_velocity = dr.detach(self.opacity_values * 0.0)
        self.values = ad.Float(np.asarray(initial_values, dtype=np.float32).reshape(-1).tolist())
        self.momentum = dr.detach(self.values * 0.0)
        self.velocity = dr.detach(self.values * 0.0)
        self.rebuild_train_scene()
        self.rebuild_preview_scene()

    def rebuild_train_scene(self) -> None:
        self.train_scene = rd.SurfelScene(
            rd.SurfelGeometry(self.center_values, self.tangent_u_values, self.tangent_v_values),
            self.opts,
        )
        self.train_scene.build()
        self.train_build_count += 1

    def rebuild_preview_scene(self) -> None:
        self.preview_scene = rd.SurfelScene(
            rd.SurfelGeometry(
                dr.detach(self.center_values),
                dr.detach(self.tangent_u_values),
                dr.detach(self.tangent_v_values),
            ),
            self.opts,
        )
        self.preview_scene.build()
        self.preview_build_count += 1

    def prepare_values(self):
        self.values = dr.detach(self.values)
        dr.enable_grad(self.values)
        self.opacity_values = dr.detach(self.opacity_values)
        if self.fit_opacity:
            dr.enable_grad(self.opacity_values)
        self.center_values = dr.detach(self.center_values)
        self.tangent_u_values = dr.detach(self.tangent_u_values)
        self.tangent_v_values = dr.detach(self.tangent_v_values)
        if self.fit_geometry:
            dr.enable_grad(self.center_values)
            dr.enable_grad(self.tangent_u_values)
            dr.enable_grad(self.tangent_v_values)
        self.rebuild_train_scene()
        return self.values

    def update_train_appearance(self) -> None:
        self.train_scene.update_appearance(
            make_appearance_from_native_values(self.opacity_values, self.values, self.degree)
        )

    def render_loss(self, view: dict, render_options: rd.SurfelRenderOptions):
        pred = self.train_scene.render(view["rays_ad"], render_options).rgb
        delta = pred - view["target_ad"]
        residual = (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]) * view["mask_ad"]
        return dr.sum(residual) / view["denom"]

    def optimizer_step_param(self,
                             values,
                             momentum,
                             velocity,
                             lr: float,
                             minimum: float,
                             maximum: float):
        grad = dr.detach(dr.grad(values))
        if self.optimizer == "adam":
            momentum = dr.detach(self.beta1 * momentum + (1.0 - self.beta1) * grad)
            velocity = dr.detach(self.beta2 * velocity + (1.0 - self.beta2) * grad * grad)
            m_hat = momentum / (1.0 - self.beta1 ** self.step_index)
            v_hat = velocity / (1.0 - self.beta2 ** self.step_index)
            updated = values - lr * m_hat / (dr.sqrt(v_hat) + self.eps)
        else:
            updated = values - lr * grad
        updated = dr.minimum(dr.maximum(updated, minimum), maximum)
        return dr.detach(updated), momentum, velocity

    def orthogonalized_tangents(self):
        u = self.tangent_u_values
        v = self.tangent_v_values
        uu = u[0] * u[0] + u[1] * u[1] + u[2] * u[2]
        uv = u[0] * v[0] + u[1] * v[1] + u[2] * v[2]
        v = v - u * (uv / dr.maximum(uu, 1e-8))
        return self.clamp_vector_length(u), self.clamp_vector_length(v)

    def clamp_vector_length(self, values):
        length = dr.sqrt(values[0] * values[0] + values[1] * values[1] + values[2] * values[2])
        target = dr.minimum(dr.maximum(length, self.tangent_min), self.tangent_max)
        return dr.detach(values * (target / dr.maximum(length, 1e-8)))

    def optimizer_step(self, lr: float) -> None:
        if self.optimizer == "adam":
            self.step_index += 1
        self.values, self.momentum, self.velocity = self.optimizer_step_param(
            self.values,
            self.momentum,
            self.velocity,
            lr,
            -0.25 if self.degree == 0 else -2.0,
            1.25 if self.degree == 0 else 2.0,
        )
        if self.fit_opacity:
            self.opacity_values, self.opacity_momentum, self.opacity_velocity = self.optimizer_step_param(
                self.opacity_values,
                self.opacity_momentum,
                self.opacity_velocity,
                lr,
                0.0,
                0.99,
            )
        if self.fit_geometry:
            geometry_lr = lr * self.geometry_lr_scale
            self.center_values, self.center_momentum, self.center_velocity = self.optimizer_step_param(
                self.center_values,
                self.center_momentum,
                self.center_velocity,
                geometry_lr,
                -self.center_bound,
                self.center_bound,
            )
            self.tangent_u_values, self.tangent_u_momentum, self.tangent_u_velocity = self.optimizer_step_param(
                self.tangent_u_values,
                self.tangent_u_momentum,
                self.tangent_u_velocity,
                geometry_lr,
                -self.tangent_max,
                self.tangent_max,
            )
            self.tangent_v_values, self.tangent_v_momentum, self.tangent_v_velocity = self.optimizer_step_param(
                self.tangent_v_values,
                self.tangent_v_momentum,
                self.tangent_v_velocity,
                geometry_lr,
                -self.tangent_max,
                self.tangent_max,
            )
            self.tangent_u_values, self.tangent_v_values = self.orthogonalized_tangents()
        dr.eval(self.values, self.opacity_values, self.center_values, self.tangent_u_values, self.tangent_v_values)

    def render_preview(self, rays, size: int) -> np.ndarray:
        self.rebuild_preview_scene()
        preview_values = dr.detach(self.values)
        preview_opacity = dr.detach(self.opacity_values)
        self.preview_scene.update_appearance(
            make_appearance_from_native_values(preview_opacity, preview_values, self.degree)
        )
        out = self.preview_scene.render(rays, self.render_options)
        dr.eval(out.rgb)
        channels = [np.asarray(dr.detach(out.rgb[channel]), dtype=np.float32) for channel in range(3)]
        return np.stack(channels, axis=1).reshape(size, size, 3)

    def coeffs_numpy(self) -> np.ndarray:
        values_np = np.asarray(dr.detach(self.values), dtype=np.float32)
        return native_flat_to_coeffs(values_np, self.surfel_count, self.degree)

    def native_values_numpy(self) -> np.ndarray:
        return np.asarray(dr.detach(self.values), dtype=np.float32)

    def opacity_numpy(self) -> np.ndarray:
        return np.asarray(dr.detach(self.opacity_values), dtype=np.float32)

    def geometry_numpy(self) -> dict[str, np.ndarray]:
        return {
            "centers": np.stack([np.asarray(dr.detach(self.center_values[i]), dtype=np.float32) for i in range(3)], axis=1),
            "tangent_u": np.stack([np.asarray(dr.detach(self.tangent_u_values[i]), dtype=np.float32) for i in range(3)], axis=1),
            "tangent_v": np.stack([np.asarray(dr.detach(self.tangent_v_values[i]), dtype=np.float32) for i in range(3)], axis=1),
        }

    def center_grad_norm_numpy(self) -> np.ndarray:
        if not self.fit_geometry:
            return np.zeros((self.surfel_count,), dtype=np.float32)
        grad = dr.detach(dr.grad(self.center_values))
        channels = [np.asarray(grad[i], dtype=np.float32) for i in range(3)]
        return np.sqrt(channels[0] * channels[0] + channels[1] * channels[1] + channels[2] * channels[2]).astype(np.float32)

    def snapshot_numpy(self) -> dict[str, np.ndarray]:
        geometry = self.geometry_numpy()
        return {
            "centers": geometry["centers"],
            "tangent_u": geometry["tangent_u"],
            "tangent_v": geometry["tangent_v"],
            "opacity": self.opacity_numpy(),
            "values": self.native_values_numpy(),
        }

    def build_counts(self) -> dict[str, int]:
        return {
            "train_scene": int(self.train_build_count),
            "preview_scene": int(self.preview_build_count),
        }


def make_fit_state(centers: np.ndarray,
                   tangent_u: np.ndarray,
                   tangent_v: np.ndarray,
                   opacity: np.ndarray,
                   initial_values: np.ndarray,
                   args,
                   opts: rd.SurfelTraceOptions,
                   build_count_base: dict[str, int] | None = None) -> GpuSurfelFitState:
    build_count_base = build_count_base or {"train_scene": 0, "preview_scene": 0}
    return GpuSurfelFitState(
        centers,
        tangent_u,
        tangent_v,
        opacity,
        initial_values,
        args.sh_degree,
        opts,
        optimizer=args.optimizer,
        background=args.background,
        fit_opacity=args.fit_opacity,
        fit_geometry=args.fit_geometry,
        geometry_lr_scale=args.geometry_lr_scale,
        center_bound=max(args.random_radius * 2.0, 1.0),
        tangent_min=max(args.initial_scale * 0.05, 1e-4),
        tangent_max=max(args.initial_scale * 8.0, args.initial_scale + 1e-4),
        train_build_count_base=build_count_base["train_scene"],
        preview_build_count_base=build_count_base["preview_scene"],
    )


def fit_color_coefficients(state: GpuSurfelFitState,
                           views: list[dict],
                           args,
                           opts: rd.SurfelTraceOptions,
                           frame_callback=None,
                           video_every: int = 1) -> tuple[GpuSurfelFitState, list[dict], list[dict]]:
    state.optimizer = args.optimizer
    log = []
    refinement_log = []
    densify_until = effective_densify_until_iter(args)
    for iteration in range(1, args.iterations + 1):
        lr = scheduled_learning_rate(
            iteration,
            args.lr,
            args.lr_final,
            args.lr_warmup_iters,
            args.iterations,
        )
        state.prepare_values()
        state.update_train_appearance()
        loss_sum = ad.Float([0.0])
        for view in views:
            loss_sum += state.render_loss(view, state.render_options)
        loss = loss_sum / max(1, len(views))
        dr.backward(loss)
        loss_value = float(loss[0])
        center_grad_norm = state.center_grad_norm_numpy()
        state.optimizer_step(lr)
        entry = {
            "iteration": iteration,
            "loss": loss_value,
            "lr": lr,
            "surfel_count": int(state.surfel_count),
        }

        should_refine = (
            args.fit_geometry
            and int(args.densify_interval) > 0
            and iteration >= int(args.densify_from_iter)
            and iteration <= densify_until
            and iteration % int(args.densify_interval) == 0
        )
        if should_refine:
            snapshot = state.snapshot_numpy()
            refined = densify_and_prune_surfel_arrays(
                snapshot["centers"],
                snapshot["tangent_u"],
                snapshot["tangent_v"],
                snapshot["opacity"],
                snapshot["values"],
                state.degree,
                center_grad_norm,
                args,
                args.seed + iteration,
            )
            opacity = refined["opacity"]
            if int(args.opacity_reset_interval) > 0 and iteration % int(args.opacity_reset_interval) == 0:
                opacity = np.minimum(opacity, float(args.opacity_reset_value)).astype(np.float32)
            build_count_base = state.build_counts()
            state = make_fit_state(
                refined["centers"],
                refined["tangent_u"],
                refined["tangent_v"],
                opacity,
                refined["values"],
                args,
                opts,
                build_count_base,
            )
            refine_entry = dict(refined["stats"])
            refine_entry["iteration"] = iteration
            refinement_log.append(refine_entry)
            entry.update({
                "surfel_count": int(refine_entry["surfel_count"]),
                "densified": True,
                "pruned": int(refine_entry["pruned"]),
                "cloned": int(refine_entry["cloned"]),
                "split": int(refine_entry["split"]),
            })
        else:
            entry["densified"] = False

        log.append(entry)
        print(f"iteration {iteration:04d}: loss={loss_value:.6f} lr={lr:.6g} surfels={state.surfel_count}")
        if frame_callback is not None and (iteration % video_every == 0 or iteration == args.iterations):
            frame_callback(state, iteration, log[-1]["loss"])
    return state, log, refinement_log


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
                        help="Dataset directory containing transforms.json and posed images.")
    parser.add_argument("--download-dxgl-apple", action="store_true",
                        help="Download the DX.GL Apple sample referenced by Hugging Face dxgl/multiview-datasets.")
    parser.add_argument("--download-sonic-nerf", action="store_true",
                        help="Download the requested number of full-resolution views from Hugging Face hayden-donnelly/sonic-nerf.")
    parser.add_argument("--download-dir", type=Path, default=Path("artifacts/datasets"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/surfel_multiview_color_fit"))
    parser.add_argument("--size", type=int, default=48)
    parser.add_argument("--views", type=int, default=3)
    parser.add_argument("--surfels", type=int, default=2048)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--lr-final", type=float, default=0.0005,
                        help="Final learning rate for exponential decay after warmup.")
    parser.add_argument("--lr-warmup-iters", type=int, default=16,
                        help="Linearly warm up from 10%% of --lr over this many iterations.")
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--sh-degree", type=int, choices=[0, 1], default=1)
    parser.add_argument("--random-radius", type=float, default=1.25)
    parser.add_argument("--initial-scale", type=float, default=0.16)
    parser.add_argument("--opacity", type=float, default=0.25)
    parser.add_argument("--background", type=float, default=1.0)
    parser.add_argument("--fit-opacity", action=argparse.BooleanOptionalAction, default=True,
                        help="Optimize per-surfel opacity along with color; use --no-fit-opacity for color-only fitting.")
    parser.add_argument("--fit-geometry", action=argparse.BooleanOptionalAction, default=True,
                        help="Optimize surfel centers and tangent vectors; enabled for random-field reconstruction.")
    parser.add_argument("--geometry-lr-scale", type=float, default=0.05)
    parser.add_argument("--loss-mask", choices=["full", "foreground"], default="full",
                        help="Use full-image loss to constrain background, or only foreground pixels.")
    parser.add_argument("--max-candidate-hits", type=int, default=32)
    parser.add_argument("--densify-from-iter", type=int, default=32)
    parser.add_argument("--densify-until-iter", type=int, default=0,
                        help="Last iteration that can refine surfels; 0 means 80%% of training.")
    parser.add_argument("--densify-interval", type=int, default=16)
    parser.add_argument("--densify-grad-threshold", type=float, default=1e-4)
    parser.add_argument("--split-scale-threshold", type=float, default=0.18)
    parser.add_argument("--split-scale-shrink", type=float, default=0.7)
    parser.add_argument("--prune-opacity-threshold", type=float, default=0.01)
    parser.add_argument("--min-scale", type=float, default=0.002)
    parser.add_argument("--max-scale", type=float, default=0.75)
    parser.add_argument("--max-surfels", type=int, default=0,
                        help="Maximum surfel count after densification; 0 uses 4x the initial count.")
    parser.add_argument("--max-new-surfels-per-refine", type=int, default=512)
    parser.add_argument("--opacity-reset-interval", type=int, default=64)
    parser.add_argument("--opacity-reset-value", type=float, default=0.05)
    parser.add_argument("--video-every", type=int, default=1,
                        help="Save one convergence GIF frame every N iterations.")
    parser.add_argument("--gif-duration-ms", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.max_surfels <= 0:
        args.max_surfels = max(args.surfels, args.surfels * 4)

    dataset_reference = DXGL_REFERENCE
    if args.download_sonic_nerf:
        scene_dir = maybe_download_sonic_nerf(args.download_dir, args.views)
        dataset_reference = SONIC_NERF_REFERENCE
    elif args.download_dxgl_apple:
        scene_dir = maybe_download_dxgl_apple(args.download_dir)
    elif args.scene_dir is not None:
        scene_dir = args.scene_dir
        if "sonic" in str(scene_dir).lower():
            dataset_reference = SONIC_NERF_REFERENCE
    else:
        raise SystemExit("Pass --scene-dir, --download-sonic-nerf, or --download-dxgl-apple.")

    transforms = load_transforms(scene_dir, args.views)
    fx, fy, cx, cy = camera_intrinsics(transforms, args.size)
    centers, initial_rgb, tangent_u, tangent_v, geometry_info = initialize_random_surfel_field(args)

    opacity = np.full((centers.shape[0],), args.opacity, dtype=np.float32)
    initial_values = native_flat_from_initial_rgb(initial_rgb, args.sh_degree)

    views = []
    for frame in transforms["frames"]:
        transform = np.asarray(frame["transform_matrix"], dtype=np.float32)
        image_path = resolve_frame_image(scene_dir, frame)
        target, foreground_mask = load_target_image(image_path, args.size, args.background)
        loss_mask = select_loss_mask(foreground_mask, args.loss_mask)
        target_ad, mask_ad, denom = target_ad_arrays(target, loss_mask)
        rays_ad = make_camera_rays(transform, fx, fy, cx, cy, args.size, ad_mode=True)
        rays_detached = make_camera_rays(transform, fx, fy, cx, cy, args.size, ad_mode=False)
        camera_origin = transform[:3, 3].astype(np.float32)
        views.append({
            "image_path": str(image_path),
            "target": target,
            "target_ad": target_ad,
            "target_image": target.reshape(args.size, args.size, 3),
            "foreground_mask": foreground_mask,
            "mask": loss_mask,
            "mask_ad": mask_ad,
            "denom": denom,
            "rays_ad": rays_ad,
            "rays_detached": rays_detached,
            "camera_origin": camera_origin,
        })

    opts = make_options(args.max_candidate_hits)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    first_view = views[0]
    frames: list[Image.Image] = []
    state = make_fit_state(
        centers,
        tangent_u,
        tangent_v,
        opacity,
        initial_values,
        args,
        opts,
    )

    def append_frame(current_state: GpuSurfelFitState, iteration: int, loss: float) -> None:
        pred_frame = current_state.render_preview(first_view["rays_detached"], args.size)
        frames.append(make_convergence_frame(first_view["target_image"], pred_frame, iteration, loss))

    initial_pred = state.render_preview(first_view["rays_detached"], args.size)
    initial_loss = float(np.mean((initial_pred - first_view["target_image"]) ** 2))
    frames.append(make_convergence_frame(first_view["target_image"], initial_pred, 0, initial_loss))
    state, log, refinement_log = fit_color_coefficients(
        state,
        views,
        args,
        opts,
        append_frame,
        max(1, args.video_every),
    )

    pred = state.render_preview(first_view["rays_detached"], args.size)
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
    coeffs = state.coeffs_numpy()
    geometry_np = state.geometry_numpy()
    np.savez_compressed(
        coeff_path,
        centers=geometry_np["centers"],
        tangent_u=geometry_np["tangent_u"],
        tangent_v=geometry_np["tangent_v"],
        initial_centers=centers,
        initial_tangent_u=tangent_u,
        initial_tangent_v=tangent_v,
        opacity=state.opacity_numpy(),
        initial_opacity=opacity,
        coeffs=coeffs,
        native_values=state.native_values_numpy(),
    )

    metrics = {
        "dataset": str(scene_dir),
        "dataset_reference": dataset_reference,
        "views": [view["image_path"] for view in views],
        "initial_surfel_count": int(centers.shape[0]),
        "surfel_count": int(state.surfel_count),
        "image_size": args.size,
        "sh_degree": args.sh_degree,
        "iterations": args.iterations,
        "optimizer": args.optimizer,
        "fit_opacity": args.fit_opacity,
        "fit_geometry": args.fit_geometry,
        "geometry_lr_scale": args.geometry_lr_scale,
        "lr": args.lr,
        "lr_final": args.lr_final,
        "lr_warmup_iters": args.lr_warmup_iters,
        "background": args.background,
        "loss_mask": args.loss_mask,
        "densification": {
            "from_iter": args.densify_from_iter,
            "until_iter": effective_densify_until_iter(args),
            "interval": args.densify_interval,
            "grad_threshold": args.densify_grad_threshold,
            "split_scale_threshold": args.split_scale_threshold,
            "split_scale_shrink": args.split_scale_shrink,
            "prune_opacity_threshold": args.prune_opacity_threshold,
            "min_scale": args.min_scale,
            "max_scale": args.max_scale,
            "max_surfels": args.max_surfels,
            "max_new_surfels_per_refine": args.max_new_surfels_per_refine,
            "opacity_reset_interval": args.opacity_reset_interval,
            "opacity_reset_value": args.opacity_reset_value,
            "events": refinement_log,
        },
        "geometry": geometry_info,
        "final_loss": log[-1]["loss"] if log else None,
        "initial_loss": initial_loss,
        "scene_build_count": state.build_counts(),
        "outputs": {
            "montage": str(montage_path),
            "video": str(video_path),
            "coefficients": str(coeff_path),
            "metrics": str(metrics_path),
        },
        "notes": [
            "The surfel field starts from random centers, random tangent frames, random color, and uniform opacity.",
            "Geometry changes rebuild the surfel GAS so OptiX candidates track the current proxy meshes.",
            "RGB and degree-1 SH color are rendered through RayD's native surfel render() path.",
            "The default full-image loss constrains transparent/background pixels as well as the foreground.",
            "Per-surfel opacity and geometry are optimized by default for random-field reconstruction.",
            "Densification clones small high-gradient surfels and splits larger high-gradient surfels, then prunes low-opacity or out-of-range scales.",
        ],
    }
    metrics_path.write_text(json.dumps({"metrics": metrics, "log": log}, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
