from __future__ import annotations

from typing import Any

from ._env import _torch
from ._util import (
    _identity_matrix,
    _normalize_matrix_tensor,
    _normalize_scalar_tensor,
    _normalize_vector_tensor,
)
from ._convert import _matrix4_to_tensor
from .types import PrimaryEdgeSample, Ray
from ._state import _CameraState
from ._native import (
    _build_native_camera,
    _build_native_scene,
    _camera_render_grad_impl,
    _camera_render_impl,
    _camera_sample_edge_impl,
    _camera_sample_ray_impl,
)
from .scene import Scene


class Camera:
    def __init__(self, *args: float, **kwargs: float):
        # Support both positional (legacy) and keyword-arg styles.
        # Prefer the classmethods Camera.perspective() / Camera.from_intrinsics().
        if kwargs and not args:
            # Pure keyword construction
            if "fx" in kwargs:
                self._state = _CameraState(
                    mode="intrinsics",
                    fx=float(kwargs["fx"]),
                    fy=float(kwargs["fy"]),
                    cx=float(kwargs["cx"]),
                    cy=float(kwargs["cy"]),
                    near_clip=float(kwargs.get("near_clip", 1e-4)),
                    far_clip=float(kwargs.get("far_clip", 1e4)),
                )
            else:
                self._state = _CameraState(
                    mode="fov",
                    fov_x=float(kwargs.get("fov_x", 45.0)),
                    near_clip=float(kwargs.get("near_clip", 1e-4)),
                    far_clip=float(kwargs.get("far_clip", 1e4)),
                )
        elif len(args) in (0, 3):
            fov_x = 45.0 if len(args) == 0 else float(args[0])
            near_clip = 1e-4 if len(args) == 0 else float(args[1])
            far_clip = 1e4 if len(args) == 0 else float(args[2])
            self._state = _CameraState(mode="fov", fov_x=fov_x, near_clip=near_clip, far_clip=far_clip)
        elif len(args) in (4, 6):
            fx = float(args[0])
            fy = float(args[1])
            cx = float(args[2])
            cy = float(args[3])
            near_clip = 1e-4 if len(args) == 4 else float(args[4])
            far_clip = 1e4 if len(args) == 4 else float(args[5])
            self._state = _CameraState(
                mode="intrinsics",
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                near_clip=near_clip,
                far_clip=far_clip,
            )
        else:
            raise TypeError(
                "Camera() expects keyword args, (fov_x, near_clip, far_clip), "
                "or (fx, fy, cx, cy, near_clip, far_clip). "
                "Prefer Camera.perspective() or Camera.from_intrinsics()."
            )

        self._built = False
        self._version = 0
        self._prepared = False
        self._prepared_scene_ref: Scene | None = None
        self._prepared_scene_version: int | None = None
        self._prepared_camera_version: int | None = None

    @classmethod
    def perspective(cls, fov_x: float = 45.0, near_clip: float = 1e-4, far_clip: float = 1e4) -> "Camera":
        return cls(fov_x=fov_x, near_clip=near_clip, far_clip=far_clip)

    @classmethod
    def from_intrinsics(cls, fx: float, fy: float, cx: float, cy: float,
                        near_clip: float = 1e-4, far_clip: float = 1e4) -> "Camera":
        return cls(fx=fx, fy=fy, cx=cx, cy=cy, near_clip=near_clip, far_clip=far_clip)

    def _invalidate(self) -> None:
        self._built = False
        self._version += 1
        self._prepared = False
        self._prepared_scene_ref = None
        self._prepared_scene_version = None
        self._prepared_camera_version = None

    def _native_detached(self) -> Any:
        return _build_native_camera(self._state, preserve_gradients=False)

    def _require_built(self) -> None:
        if not self._built:
            raise RuntimeError("Camera is not built. Call build() before querying.")

    def _default_transform(self) -> _torch.Tensor:
        return _identity_matrix()

    def build(self, cache: bool = True) -> None:
        self._state.cache = bool(cache)
        self._native_detached()
        self._built = True
        self._version += 1
        self._prepared = False
        self._prepared_scene_ref = None
        self._prepared_scene_version = None
        self._prepared_camera_version = None

    def render(self, scene: Scene, background: float = 0.0) -> _torch.Tensor:
        self._require_built()
        if not isinstance(scene, Scene):
            raise TypeError("Camera.render() expects a rayd.torch.Scene.")
        scene._require_query_ready()
        return _camera_render_impl(self._state, scene._mesh_states(), float(background))

    def render_grad(self, scene: Scene, spp: int = 4, background: float = 0.0) -> _torch.Tensor:
        self._require_built()
        if not isinstance(scene, Scene):
            raise TypeError("Camera.render_grad() expects a rayd.torch.Scene.")
        scene._require_query_ready()
        return _camera_render_grad_impl(self._state, scene._mesh_states(), int(spp), float(background))

    def prepare_edges(self, scene: Scene) -> None:
        self._require_built()
        if not isinstance(scene, Scene):
            raise TypeError("Camera.prepare_edges() expects a rayd.torch.Scene.")
        scene._require_query_ready()
        native_scene = _build_native_scene(scene._mesh_states(), preserve_gradients=False)
        native_camera = self._native_detached()
        native_camera.prepare_edges(native_scene)
        self._prepared = True
        self._prepared_scene_ref = scene
        self._prepared_scene_version = scene._version
        self._prepared_camera_version = self._version

    def sample_ray(self, sample: Any) -> Ray:
        self._require_built()
        return _camera_sample_ray_impl(self._state, _normalize_vector_tensor(sample, "sample", 2, _torch.float32))

    def sample_edge(self, sample1: Any) -> PrimaryEdgeSample:
        self._require_built()
        if (
            not self._prepared
            or self._prepared_scene_ref is None
            or self._prepared_scene_version != self._prepared_scene_ref._version
            or self._prepared_camera_version != self._version
            or not self._prepared_scene_ref.is_ready()
            or self._prepared_scene_ref.has_pending_updates()
        ):
            raise RuntimeError("Camera.sample_edge(): camera is not prepared for the current scene state.")
        return _camera_sample_edge_impl(
            self._state,
            self._prepared_scene_ref._mesh_states(),
            _normalize_scalar_tensor(sample1, "sample1", _torch.float32),
        )

    def set_transform(self, mat: Any, set_left: bool = True) -> None:
        if set_left:
            self.to_world_left = mat
        else:
            self.to_world_right = mat

    def append_transform(self, mat: Any, append_left: bool = True) -> None:
        matrix = _normalize_matrix_tensor(mat, "mat")
        if append_left:
            self.to_world_left = matrix @ self.to_world_left
        else:
            self.to_world_right = self.to_world_right @ matrix

    @property
    def width(self) -> int:
        return int(self._state.width)

    @width.setter
    def width(self, value: int) -> None:
        self._state.width = int(value)
        self._invalidate()

    @property
    def height(self) -> int:
        return int(self._state.height)

    @height.setter
    def height(self, value: int) -> None:
        self._state.height = int(value)
        self._invalidate()

    @property
    def to_world(self) -> _torch.Tensor:
        return self._state.to_world if self._state.to_world is not None else self._default_transform()

    @to_world.setter
    def to_world(self, value: Any) -> None:
        self._state.to_world = _normalize_matrix_tensor(value, "to_world")
        self._invalidate()

    @property
    def to_world_left(self) -> _torch.Tensor:
        return self._state.to_world_left if self._state.to_world_left is not None else self._default_transform()

    @to_world_left.setter
    def to_world_left(self, value: Any) -> None:
        self._state.to_world_left = _normalize_matrix_tensor(value, "to_world_left")
        self._invalidate()

    @property
    def to_world_right(self) -> _torch.Tensor:
        return self._state.to_world_right if self._state.to_world_right is not None else self._default_transform()

    @to_world_right.setter
    def to_world_right(self, value: Any) -> None:
        self._state.to_world_right = _normalize_matrix_tensor(value, "to_world_right")
        self._invalidate()

    @property
    def camera_to_sample(self) -> _torch.Tensor:
        self._require_built()
        return _matrix4_to_tensor(self._native_detached().camera_to_sample).torch()

    @property
    def sample_to_camera(self) -> _torch.Tensor:
        self._require_built()
        return _matrix4_to_tensor(self._native_detached().sample_to_camera).torch()

    @property
    def world_to_sample(self) -> _torch.Tensor:
        self._require_built()
        return _matrix4_to_tensor(self._native_detached().world_to_sample).torch()

    @property
    def sample_to_world(self) -> _torch.Tensor:
        self._require_built()
        return _matrix4_to_tensor(self._native_detached().sample_to_world).torch()

    def __repr__(self) -> str:
        return f"Camera(width={self.width}, height={self.height}, built={self._built})"
