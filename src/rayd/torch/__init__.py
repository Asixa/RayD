from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

import drjit as dr

try:
    import torch as _torch
except ImportError as exc:  # pragma: no cover - exercised in subprocess tests
    raise ImportError(
        "rayd.torch requires the optional 'torch' dependency. "
        "Install it with `pip install rayd[torch]` or `pip install torch`."
    ) from exc


_native = importlib.import_module("rayd.rayd")
_cuda = importlib.import_module("drjit.cuda")
_cuda_ad = importlib.import_module("drjit.cuda.ad")

_PROFILE_FIELDS = (
    "mesh_update_ms",
    "triangle_scatter_ms",
    "triangle_eval_ms",
    "optix_commit_ms",
    "total_ms",
    "optix_gas_update_ms",
    "optix_ias_update_ms",
    "updated_meshes",
    "updated_vertex_meshes",
    "updated_transform_meshes",
)


def _default_cuda_device() -> _torch.device:
    if not _torch.cuda.is_available():
        raise RuntimeError("rayd.torch requires CUDA tensors, but torch.cuda.is_available() is False.")
    return _torch.device("cuda")


def _identity_matrix(device: _torch.device | None = None) -> _torch.Tensor:
    return _torch.eye(4, device=device or _default_cuda_device(), dtype=_torch.float32)


def _empty_vec2(device: _torch.device | None = None) -> _torch.Tensor:
    return _torch.empty((0, 2), device=device or _default_cuda_device(), dtype=_torch.float32)


def _empty_vec3(device: _torch.device | None = None) -> _torch.Tensor:
    return _torch.empty((0, 3), device=device or _default_cuda_device(), dtype=_torch.float32)


def _empty_idx3(device: _torch.device | None = None) -> _torch.Tensor:
    return _torch.empty((0, 3), device=device or _default_cuda_device(), dtype=_torch.int32)


def _is_torch_tensor(value: Any) -> bool:
    return isinstance(value, _torch.Tensor)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        return ()
    return tuple(int(v) for v in shape)


def _require_cuda_tensor(value: Any, name: str) -> _torch.Tensor:
    if not _is_torch_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if value.device.type != "cuda":
        raise TypeError(f"{name} must be a CUDA torch.Tensor. CPU tensors are not supported by rayd.torch.")
    return value


def _normalize_vector_tensor(value: Any, name: str, dims: int, dtype: _torch.dtype) -> _torch.Tensor:
    tensor = _require_cuda_tensor(value, name)
    if tensor.ndim == 1:
        if tensor.shape[0] != dims:
            raise ValueError(f"{name} must have shape ({dims},) or (N, {dims}).")
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim != 2 or tensor.shape[1] != dims:
        raise ValueError(f"{name} must have shape ({dims},) or (N, {dims}).")
    return tensor.to(dtype=dtype).contiguous()


def _normalize_matrix_tensor(value: Any, name: str) -> _torch.Tensor:
    tensor = _require_cuda_tensor(value, name)
    if tensor.ndim != 2 or tensor.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4).")
    return tensor.to(dtype=_torch.float32).contiguous()


def _normalize_scalar_tensor(value: Any, name: str, dtype: _torch.dtype) -> _torch.Tensor:
    tensor = _require_cuda_tensor(value, name)
    if tensor.ndim == 0:
        tensor = tensor.reshape(1)
    elif tensor.ndim != 1:
        raise ValueError(f"{name} must have shape () or (N,).")
    return tensor.to(dtype=dtype).contiguous()


def _expand_1d_tensor(tensor: _torch.Tensor, size: int, name: str) -> _torch.Tensor:
    if tensor.shape[0] == size:
        return tensor
    if tensor.shape[0] == 1 and size > 1:
        return tensor.expand(size).contiguous()
    raise ValueError(f"{name} must have length 1 or {size}, got {tensor.shape[0]}.")


def _normalize_active_tensor(active: Any, batch_size: int | None, name: str = "active") -> bool | _torch.Tensor:
    if isinstance(active, bool):
        return active
    tensor = _normalize_scalar_tensor(active, name, _torch.bool)
    if batch_size is not None:
        tensor = _expand_1d_tensor(tensor, batch_size, name)
    return tensor


def _batch_size_from_vector(value: Any, dims: int, name: str) -> int:
    shape = _shape_tuple(value)
    if len(shape) == 1 and shape[0] == dims:
        return 1
    if len(shape) == 2 and shape[1] == dims:
        return shape[0]
    raise ValueError(f"{name} must have shape ({dims},) or (N, {dims}).")


def _infer_diff(value: Any) -> bool:
    if value is None:
        return False
    if _is_torch_tensor(value):
        return bool(value.dtype.is_floating_point and value.requires_grad)
    tp = type(value)
    if dr.is_array_v(tp):
        return bool(dr.is_diff_v(tp))
    return False


def _as_drjit_value(value: Any) -> Any:
    if value is None:
        return None
    if dr.is_array_v(type(value)):
        return value
    if _is_torch_tensor(value):
        return dr.detail.import_tensor(value, True)
    return value


def _float_scalar_type(diff: bool):
    return _cuda_ad.Float if diff else _cuda.Float


def _bool_scalar_type(diff: bool):
    return _cuda_ad.Bool if diff else _cuda.Bool


def _vec2_type(diff: bool):
    return _cuda_ad.Array2f if diff else _cuda.Array2f


def _vec3_type(diff: bool):
    return _cuda_ad.Array3f if diff else _cuda.Array3f


def _vec4_type(diff: bool):
    return _cuda_ad.Array4f if diff else _cuda.Array4f


def _mat4_type(diff: bool):
    return _cuda_ad.Matrix4f if diff else _cuda.Matrix4f


def _tensor_to_scalar_array(value: Any, *, diff: bool | None = None, default: Any | None = None, name: str) -> Any:
    if value is None:
        return default

    diff = _infer_diff(value) if diff is None else diff
    target = _float_scalar_type(diff)
    arr = _as_drjit_value(value)
    if type(arr) is target:
        return arr
    if dr.is_array_v(type(arr)):
        if dr.is_tensor_v(arr):
            arr = dr.ravel(arr)
        return target(arr)
    return target(arr)


def _tensor_to_mask(value: Any, *, diff: bool) -> Any:
    if isinstance(value, bool):
        return value
    target = _bool_scalar_type(diff)
    arr = _as_drjit_value(value)
    if type(arr) is target:
        return arr
    if dr.is_array_v(type(arr)) and dr.is_tensor_v(arr):
        arr = dr.ravel(arr)
    return target(arr)


def _tensor_to_vec2(value: Any, *, diff: bool | None = None, allow_none: bool = False, name: str) -> Any:
    if value is None:
        if allow_none:
            return _cuda.Array2f()
        raise ValueError(f"{name} is required.")

    diff = _infer_diff(value) if diff is None else diff
    target = _vec2_type(diff)
    arr = _as_drjit_value(value)
    if type(arr) is target:
        return arr
    shape = _shape_tuple(arr)
    if len(shape) != 2 or shape[1] != 2:
        raise ValueError(f"{name} must have shape (N, 2).")
    return target(arr[:, 0], arr[:, 1])


def _tensor_to_vec3(value: Any, *, diff: bool | None = None, allow_none: bool = False, name: str) -> Any:
    if value is None:
        if allow_none:
            return _cuda.Array3f()
        raise ValueError(f"{name} is required.")

    diff = _infer_diff(value) if diff is None else diff
    target = _vec3_type(diff)
    arr = _as_drjit_value(value)
    if type(arr) is target:
        return arr
    shape = _shape_tuple(arr)
    if len(shape) != 2 or shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3).")
    return target(arr[:, 0], arr[:, 1], arr[:, 2])


def _tensor_to_vec3i(value: Any, *, allow_none: bool = False, name: str) -> Any:
    if value is None:
        if allow_none:
            return _cuda.Array3i()
        raise ValueError(f"{name} is required.")

    arr = _as_drjit_value(value)
    if type(arr) is _cuda.Array3i:
        return arr
    shape = _shape_tuple(arr)
    if len(shape) != 2 or shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3).")
    return _cuda.Array3i(arr[:, 0], arr[:, 1], arr[:, 2])


def _tensor_to_matrix4(value: Any, *, diff: bool | None = None, name: str, allow_none: bool = False) -> Any:
    if value is None:
        if allow_none:
            target = _mat4_type(False if diff is None else diff)
            return dr.identity(target)
        raise ValueError(f"{name} is required.")

    diff = _infer_diff(value) if diff is None else diff
    target = _mat4_type(diff)
    arr = _as_drjit_value(value)
    if type(arr) is target:
        return arr
    shape = _shape_tuple(arr)
    if shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4).")
    row_type = _vec4_type(diff)
    rows = []
    for i in range(4):
        rows.append(
            row_type(
                arr[i, 0].array,
                arr[i, 1].array,
                arr[i, 2].array,
                arr[i, 3].array,
            )
        )
    return target(rows[0], rows[1], rows[2], rows[3])


def _scalar_array_to_tensor(value: Any) -> Any:
    if value is None:
        return None
    if not dr.is_array_v(type(value)):
        return value
    if not dr.is_tensor_v(value):
        value = dr.tensor_t(type(value))(value)
    return value


def _vec2_to_tensor(value: Any) -> Any:
    scalar = type(value[0])
    tensor_type = dr.tensor_t(scalar)
    size = dr.width(value[0])
    out = dr.zeros(tensor_type, shape=(size, 2))
    out[:, 0] = value[0]
    out[:, 1] = value[1]
    return out


def _vec3_to_tensor(value: Any) -> Any:
    scalar = type(value[0])
    tensor_type = dr.tensor_t(scalar)
    size = dr.width(value[0])
    out = dr.zeros(tensor_type, shape=(size, 3))
    out[:, 0] = value[0]
    out[:, 1] = value[1]
    out[:, 2] = value[2]
    return out


def _matrix4_to_tensor(value: Any) -> Any:
    scalar = type(value[0][0])
    tensor_type = dr.tensor_t(scalar)
    out = dr.zeros(tensor_type, shape=(4, 4))
    for i in range(4):
        for j in range(4):
            out[i, j] = value[i][j]
    return out


def _torch_or_default(value: _torch.Tensor | None, default: _torch.Tensor) -> _torch.Tensor:
    return default if value is None else value


def _device_from_values(*values: Any) -> _torch.device:
    for value in values:
        if _is_torch_tensor(value):
            return value.device
    return _default_cuda_device()


def _normalize_public_ray_fields(o: Any, d: Any, tmax: Any) -> tuple[Any, Any, Any]:
    if o is None and d is None and tmax is None:
        return None, None, None
    if _is_torch_tensor(o) and _is_torch_tensor(d):
        origin = _normalize_vector_tensor(o, "o", 3, _torch.float32)
        direction = _normalize_vector_tensor(d, "d", 3, _torch.float32)
        if origin.shape[0] != direction.shape[0]:
            raise ValueError("o and d must have the same batch size.")
        limit = None
        if tmax is not None:
            limit = _normalize_scalar_tensor(tmax, "tmax", _torch.float32)
            limit = _expand_1d_tensor(limit, origin.shape[0], "tmax")
        return origin, direction, limit
    return o, d, tmax


class _StructRepr:
    DRJIT_STRUCT: dict[str, object] = {}

    def __repr__(self) -> str:
        parts = ", ".join(f"{name}={getattr(self, name)!r}" for name in self.DRJIT_STRUCT)
        return f"{type(self).__name__}({parts})"


class RayDetached(_StructRepr):
    DRJIT_STRUCT = {"o": object, "d": object, "tmax": object}

    def __init__(self, o: Any = None, d: Any = None, tmax: Any = None):
        self.o, self.d, self.tmax = _normalize_public_ray_fields(o, d, tmax)

    def reversed(self) -> "RayDetached":
        return RayDetached(self.o, -self.d, self.tmax)


class Ray(_StructRepr):
    DRJIT_STRUCT = {"o": object, "d": object, "tmax": object}

    def __init__(self, o: Any = None, d: Any = None, tmax: Any = None):
        self.o, self.d, self.tmax = _normalize_public_ray_fields(o, d, tmax)

    def reversed(self) -> "Ray":
        return Ray(self.o, -self.d, self.tmax)


class IntersectionDetached(_StructRepr):
    DRJIT_STRUCT = {
        "t": object,
        "p": object,
        "n": object,
        "geo_n": object,
        "uv": object,
        "barycentric": object,
        "shape_id": object,
        "prim_id": object,
    }

    def __init__(
        self,
        t: Any = None,
        p: Any = None,
        n: Any = None,
        geo_n: Any = None,
        uv: Any = None,
        barycentric: Any = None,
        shape_id: Any = None,
        prim_id: Any = None,
    ):
        self.t = t
        self.p = p
        self.n = n
        self.geo_n = geo_n
        self.uv = uv
        self.barycentric = barycentric
        self.shape_id = shape_id
        self.prim_id = prim_id

    def is_valid(self) -> Any:
        return self.prim_id >= 0


class Intersection(_StructRepr):
    DRJIT_STRUCT = IntersectionDetached.DRJIT_STRUCT

    def __init__(
        self,
        t: Any = None,
        p: Any = None,
        n: Any = None,
        geo_n: Any = None,
        uv: Any = None,
        barycentric: Any = None,
        shape_id: Any = None,
        prim_id: Any = None,
    ):
        self.t = t
        self.p = p
        self.n = n
        self.geo_n = geo_n
        self.uv = uv
        self.barycentric = barycentric
        self.shape_id = shape_id
        self.prim_id = prim_id

    def is_valid(self) -> Any:
        return self.prim_id >= 0


class NearestPointEdgeDetached(_StructRepr):
    DRJIT_STRUCT = {
        "distance": object,
        "point": object,
        "edge_t": object,
        "edge_point": object,
        "shape_id": object,
        "edge_id": object,
        "is_boundary": object,
    }

    def __init__(
        self,
        distance: Any = None,
        point: Any = None,
        edge_t: Any = None,
        edge_point: Any = None,
        shape_id: Any = None,
        edge_id: Any = None,
        is_boundary: Any = None,
    ):
        self.distance = distance
        self.point = point
        self.edge_t = edge_t
        self.edge_point = edge_point
        self.shape_id = shape_id
        self.edge_id = edge_id
        self.is_boundary = is_boundary

    def is_valid(self) -> Any:
        return self.edge_id >= 0


class NearestPointEdge(_StructRepr):
    DRJIT_STRUCT = NearestPointEdgeDetached.DRJIT_STRUCT

    def __init__(
        self,
        distance: Any = None,
        point: Any = None,
        edge_t: Any = None,
        edge_point: Any = None,
        shape_id: Any = None,
        edge_id: Any = None,
        is_boundary: Any = None,
    ):
        self.distance = distance
        self.point = point
        self.edge_t = edge_t
        self.edge_point = edge_point
        self.shape_id = shape_id
        self.edge_id = edge_id
        self.is_boundary = is_boundary

    def is_valid(self) -> Any:
        return self.edge_id >= 0


class NearestRayEdgeDetached(_StructRepr):
    DRJIT_STRUCT = {
        "distance": object,
        "ray_t": object,
        "point": object,
        "edge_t": object,
        "edge_point": object,
        "shape_id": object,
        "edge_id": object,
        "is_boundary": object,
    }

    def __init__(
        self,
        distance: Any = None,
        ray_t: Any = None,
        point: Any = None,
        edge_t: Any = None,
        edge_point: Any = None,
        shape_id: Any = None,
        edge_id: Any = None,
        is_boundary: Any = None,
    ):
        self.distance = distance
        self.ray_t = ray_t
        self.point = point
        self.edge_t = edge_t
        self.edge_point = edge_point
        self.shape_id = shape_id
        self.edge_id = edge_id
        self.is_boundary = is_boundary

    def is_valid(self) -> Any:
        return self.edge_id >= 0


class NearestRayEdge(_StructRepr):
    DRJIT_STRUCT = NearestRayEdgeDetached.DRJIT_STRUCT

    def __init__(
        self,
        distance: Any = None,
        ray_t: Any = None,
        point: Any = None,
        edge_t: Any = None,
        edge_point: Any = None,
        shape_id: Any = None,
        edge_id: Any = None,
        is_boundary: Any = None,
    ):
        self.distance = distance
        self.ray_t = ray_t
        self.point = point
        self.edge_t = edge_t
        self.edge_point = edge_point
        self.shape_id = shape_id
        self.edge_id = edge_id
        self.is_boundary = is_boundary

    def is_valid(self) -> Any:
        return self.edge_id >= 0


class PrimaryEdgeSample(_StructRepr):
    DRJIT_STRUCT = {
        "x_dot_n": object,
        "idx": object,
        "ray_n": object,
        "ray_p": object,
        "pdf": object,
    }

    def __init__(
        self,
        x_dot_n: Any = None,
        idx: Any = None,
        ray_n: Any = None,
        ray_p: Any = None,
        pdf: Any = None,
    ):
        self.x_dot_n = x_dot_n
        self.idx = idx
        self.ray_n = ray_n
        self.ray_p = ray_p
        self.pdf = pdf


class SecondaryEdgeInfo(_StructRepr):
    DRJIT_STRUCT = {
        "start": object,
        "edge": object,
        "normal0": object,
        "normal1": object,
        "opposite": object,
        "is_boundary": object,
    }

    def __init__(
        self,
        start: Any = None,
        edge: Any = None,
        normal0: Any = None,
        normal1: Any = None,
        opposite: Any = None,
        is_boundary: Any = None,
    ):
        self.start = start
        self.edge = edge
        self.normal0 = normal0
        self.normal1 = normal1
        self.opposite = opposite
        self.is_boundary = is_boundary

    def size(self) -> int:
        if self.is_boundary is None:
            return 0
        return int(_shape_tuple(self.is_boundary)[0])


class SceneCommitProfile:
    def __init__(self, native_profile: Any | None = None):
        for field in _PROFILE_FIELDS:
            setattr(self, field, getattr(native_profile, field, 0))

    def __repr__(self) -> str:
        parts = ", ".join(f"{field}={getattr(self, field)!r}" for field in _PROFILE_FIELDS)
        return f"SceneCommitProfile({parts})"


class _MeshState(_StructRepr):
    DRJIT_STRUCT = {
        "vertex_positions": object,
        "face_indices": object,
        "vertex_uv": object,
        "face_uv_indices": object,
        "to_world": object,
        "to_world_left": object,
        "to_world_right": object,
        "use_face_normals": object,
        "edges_enabled": object,
        "verbose": object,
    }

    def __init__(
        self,
        vertex_positions: Any = None,
        face_indices: Any = None,
        vertex_uv: Any = None,
        face_uv_indices: Any = None,
        to_world: Any = None,
        to_world_left: Any = None,
        to_world_right: Any = None,
        use_face_normals: bool = False,
        edges_enabled: bool = True,
        verbose: bool = False,
    ):
        self.vertex_positions = vertex_positions
        self.face_indices = face_indices
        self.vertex_uv = vertex_uv
        self.face_uv_indices = face_uv_indices
        self.to_world = to_world
        self.to_world_left = to_world_left
        self.to_world_right = to_world_right
        self.use_face_normals = use_face_normals
        self.edges_enabled = edges_enabled
        self.verbose = verbose

    def copy(self) -> "_MeshState":
        return _MeshState(
            vertex_positions=self.vertex_positions,
            face_indices=self.face_indices,
            vertex_uv=self.vertex_uv,
            face_uv_indices=self.face_uv_indices,
            to_world=self.to_world,
            to_world_left=self.to_world_left,
            to_world_right=self.to_world_right,
            use_face_normals=self.use_face_normals,
            edges_enabled=self.edges_enabled,
            verbose=self.verbose,
        )


class _CameraState(_StructRepr):
    DRJIT_STRUCT = {
        "mode": object,
        "fov_x": object,
        "fx": object,
        "fy": object,
        "cx": object,
        "cy": object,
        "near_clip": object,
        "far_clip": object,
        "width": object,
        "height": object,
        "cache": object,
        "to_world": object,
        "to_world_left": object,
        "to_world_right": object,
    }

    def __init__(
        self,
        mode: str = "fov",
        fov_x: float = 45.0,
        fx: float = 0.0,
        fy: float = 0.0,
        cx: float = 0.0,
        cy: float = 0.0,
        near_clip: float = 1e-4,
        far_clip: float = 1e4,
        width: int = 1,
        height: int = 1,
        cache: bool = True,
        to_world: Any = None,
        to_world_left: Any = None,
        to_world_right: Any = None,
    ):
        self.mode = mode
        self.fov_x = fov_x
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.near_clip = near_clip
        self.far_clip = far_clip
        self.width = width
        self.height = height
        self.cache = cache
        self.to_world = to_world
        self.to_world_left = to_world_left
        self.to_world_right = to_world_right


@dataclass
class _SceneMeshRecord:
    state: _MeshState
    dynamic: bool


def _ray_batch_size(ray: Ray | RayDetached) -> int:
    return _batch_size_from_vector(ray.o, 3, "ray.o")


def _public_ray_from_native(ray: Any, *, detached: bool) -> Ray | RayDetached:
    cls = RayDetached if detached else Ray
    return cls(
        o=_vec3_to_tensor(ray.o),
        d=_vec3_to_tensor(ray.d),
        tmax=_scalar_array_to_tensor(ray.tmax),
    )


def _intersection_from_native(its: Any, *, detached: bool) -> Intersection | IntersectionDetached:
    cls = IntersectionDetached if detached else Intersection
    return cls(
        t=_scalar_array_to_tensor(its.t),
        p=_vec3_to_tensor(its.p),
        n=_vec3_to_tensor(its.n),
        geo_n=_vec3_to_tensor(its.geo_n),
        uv=_vec2_to_tensor(its.uv),
        barycentric=_vec3_to_tensor(its.barycentric),
        shape_id=_scalar_array_to_tensor(its.shape_id),
        prim_id=_scalar_array_to_tensor(its.prim_id),
    )


def _nearest_point_from_native(result: Any, *, detached: bool) -> NearestPointEdge | NearestPointEdgeDetached:
    cls = NearestPointEdgeDetached if detached else NearestPointEdge
    return cls(
        distance=_scalar_array_to_tensor(result.distance),
        point=_vec3_to_tensor(result.point),
        edge_t=_scalar_array_to_tensor(result.edge_t),
        edge_point=_vec3_to_tensor(result.edge_point),
        shape_id=_scalar_array_to_tensor(result.shape_id),
        edge_id=_scalar_array_to_tensor(result.edge_id),
        is_boundary=_scalar_array_to_tensor(result.is_boundary),
    )


def _nearest_ray_from_native(result: Any, *, detached: bool) -> NearestRayEdge | NearestRayEdgeDetached:
    cls = NearestRayEdgeDetached if detached else NearestRayEdge
    return cls(
        distance=_scalar_array_to_tensor(result.distance),
        ray_t=_scalar_array_to_tensor(result.ray_t),
        point=_vec3_to_tensor(result.point),
        edge_t=_scalar_array_to_tensor(result.edge_t),
        edge_point=_vec3_to_tensor(result.edge_point),
        shape_id=_scalar_array_to_tensor(result.shape_id),
        edge_id=_scalar_array_to_tensor(result.edge_id),
        is_boundary=_scalar_array_to_tensor(result.is_boundary),
    )


def _secondary_edges_from_native(info: Any) -> SecondaryEdgeInfo:
    return SecondaryEdgeInfo(
        start=_vec3_to_tensor(info.start),
        edge=_vec3_to_tensor(info.edge),
        normal0=_vec3_to_tensor(info.normal0),
        normal1=_vec3_to_tensor(info.normal1),
        opposite=_vec3_to_tensor(info.opposite),
        is_boundary=_scalar_array_to_tensor(info.is_boundary),
    )


def _primary_edge_sample_from_native(sample: Any) -> PrimaryEdgeSample:
    return PrimaryEdgeSample(
        x_dot_n=_scalar_array_to_tensor(sample.x_dot_n),
        idx=_scalar_array_to_tensor(sample.idx),
        ray_n=_public_ray_from_native(sample.ray_n, detached=True),
        ray_p=_public_ray_from_native(sample.ray_p, detached=True),
        pdf=_scalar_array_to_tensor(sample.pdf),
    )


def _to_torch_struct(value: Any) -> Any:
    if value is None:
        return None
    if dr.is_array_v(type(value)):
        if not dr.is_tensor_v(value):
            value = dr.tensor_t(type(value))(value)
        return value.torch()
    if isinstance(value, tuple):
        return tuple(_to_torch_struct(v) for v in value)
    if isinstance(value, list):
        return [_to_torch_struct(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_torch_struct(v) for k, v in value.items()}
    desc = getattr(type(value), "DRJIT_STRUCT", None)
    if isinstance(desc, dict):
        out = type(value)()
        for name in desc:
            setattr(out, name, _to_torch_struct(getattr(value, name)))
        return out
    return value


def _mesh_device(state: _MeshState) -> _torch.device:
    return _device_from_values(
        state.vertex_positions,
        state.face_indices,
        state.vertex_uv,
        state.face_uv_indices,
        state.to_world,
        state.to_world_left,
        state.to_world_right,
    )


def _mesh_to_world_tensor(state: _MeshState, attr: str) -> _torch.Tensor:
    value = getattr(state, attr)
    if value is not None:
        return value
    return _identity_matrix(_mesh_device(state))


def _native_ray_from_public(ray: Ray | RayDetached) -> Any:
    detached = isinstance(ray, RayDetached)
    diff = not detached
    batch = _ray_batch_size(ray)
    default_tmax = dr.full(_float_scalar_type(diff), dr.inf, batch)
    native_ray = _native.RayDetached() if detached else _native.Ray()
    native_ray.o = _tensor_to_vec3(ray.o, diff=diff, name="ray.o")
    native_ray.d = _tensor_to_vec3(ray.d, diff=diff, name="ray.d")
    native_ray.tmax = _tensor_to_scalar_array(ray.tmax, diff=diff, default=default_tmax, name="ray.tmax")
    return native_ray


def _build_native_mesh(state: _MeshState, *, preserve_gradients: bool) -> Any:
    mesh = _native.Mesh(
        _tensor_to_vec3(state.vertex_positions, diff=False, name="mesh.vertex_positions"),
        _tensor_to_vec3i(state.face_indices, name="mesh.face_indices"),
        _tensor_to_vec2(state.vertex_uv, diff=False, allow_none=True, name="mesh.vertex_uv"),
        _tensor_to_vec3i(state.face_uv_indices, allow_none=True, name="mesh.face_uv_indices"),
        bool(state.verbose),
    )
    mesh.use_face_normals = bool(state.use_face_normals)
    mesh.edges_enabled = bool(state.edges_enabled)

    if state.to_world is not None:
        mesh.to_world = _tensor_to_matrix4(state.to_world, diff=False, name="mesh.to_world")
    if state.to_world_left is not None:
        mesh.to_world_left = _tensor_to_matrix4(state.to_world_left, diff=False, name="mesh.to_world_left")
    if state.to_world_right is not None:
        mesh.to_world_right = _tensor_to_matrix4(state.to_world_right, diff=False, name="mesh.to_world_right")

    if preserve_gradients:
        if _infer_diff(state.vertex_positions):
            mesh.vertex_positions = _tensor_to_vec3(state.vertex_positions, diff=True, name="mesh.vertex_positions")
        if state.vertex_uv is not None and _infer_diff(state.vertex_uv):
            mesh.vertex_uv = _tensor_to_vec2(state.vertex_uv, diff=True, name="mesh.vertex_uv")
        if state.to_world is not None and _infer_diff(state.to_world):
            mesh.to_world = _tensor_to_matrix4(state.to_world, diff=True, name="mesh.to_world")
        if state.to_world_left is not None and _infer_diff(state.to_world_left):
            mesh.to_world_left = _tensor_to_matrix4(state.to_world_left, diff=True, name="mesh.to_world_left")
        if state.to_world_right is not None and _infer_diff(state.to_world_right):
            mesh.to_world_right = _tensor_to_matrix4(state.to_world_right, diff=True, name="mesh.to_world_right")

    return mesh


def _build_native_scene(mesh_states: list[_MeshState], *, preserve_gradients: bool) -> Any:
    scene = _native.Scene()
    for state in mesh_states:
        scene.add_mesh(_build_native_mesh(state, preserve_gradients=preserve_gradients))
    scene.configure()
    return scene


def _build_native_camera(state: _CameraState, *, preserve_gradients: bool) -> Any:
    if state.mode == "intrinsics":
        camera = _native.Camera(state.fx, state.fy, state.cx, state.cy, state.near_clip, state.far_clip)
    else:
        camera = _native.Camera(state.fov_x, state.near_clip, state.far_clip)
    camera.width = int(state.width)
    camera.height = int(state.height)

    if state.to_world is not None:
        camera.to_world = _tensor_to_matrix4(state.to_world, diff=False, name="camera.to_world")
    if state.to_world_left is not None:
        camera.to_world_left = _tensor_to_matrix4(state.to_world_left, diff=False, name="camera.to_world_left")
    if state.to_world_right is not None:
        camera.to_world_right = _tensor_to_matrix4(state.to_world_right, diff=False, name="camera.to_world_right")

    if preserve_gradients:
        if state.to_world is not None and _infer_diff(state.to_world):
            camera.to_world = _tensor_to_matrix4(state.to_world, diff=True, name="camera.to_world")
        if state.to_world_left is not None and _infer_diff(state.to_world_left):
            camera.to_world_left = _tensor_to_matrix4(state.to_world_left, diff=True, name="camera.to_world_left")
        if state.to_world_right is not None and _infer_diff(state.to_world_right):
            camera.to_world_right = _tensor_to_matrix4(state.to_world_right, diff=True, name="camera.to_world_right")

    camera.configure(bool(state.cache))
    return camera


@dr.wrap(source="torch", target="drjit")
def _scene_intersect_impl(mesh_states: list[_MeshState], ray: Ray | RayDetached, active: Any) -> Any:
    detached = isinstance(ray, RayDetached)
    scene = _build_native_scene(mesh_states, preserve_gradients=True)
    its = scene.intersect(_native_ray_from_public(ray), _tensor_to_mask(active, diff=not detached))
    return _intersection_from_native(its, detached=detached)


@dr.wrap(source="torch", target="drjit")
def _scene_shadow_test_impl(mesh_states: list[_MeshState], ray: Ray | RayDetached, active: Any) -> Any:
    detached = isinstance(ray, RayDetached)
    scene = _build_native_scene(mesh_states, preserve_gradients=True)
    return _scalar_array_to_tensor(scene.shadow_test(_native_ray_from_public(ray), _tensor_to_mask(active, diff=not detached)))


@dr.wrap(source="torch", target="drjit")
def _scene_nearest_point_impl(mesh_states: list[_MeshState], point: Any, active: Any) -> Any:
    detached = not _infer_diff(point)
    scene = _build_native_scene(mesh_states, preserve_gradients=True)
    result = scene.nearest_edge(_tensor_to_vec3(point, diff=not detached, name="point"), _tensor_to_mask(active, diff=not detached))
    return _nearest_point_from_native(result, detached=detached)


@dr.wrap(source="torch", target="drjit")
def _scene_nearest_ray_impl(mesh_states: list[_MeshState], ray: Ray | RayDetached, active: Any) -> Any:
    detached = isinstance(ray, RayDetached)
    scene = _build_native_scene(mesh_states, preserve_gradients=True)
    result = scene.nearest_edge(_native_ray_from_public(ray), _tensor_to_mask(active, diff=not detached))
    return _nearest_ray_from_native(result, detached=detached)


@dr.wrap(source="torch", target="drjit")
def _camera_sample_ray_impl(state: _CameraState, sample: Any) -> Any:
    detached = not _infer_diff(sample)
    camera = _build_native_camera(state, preserve_gradients=True)
    ray = camera.sample_ray(_tensor_to_vec2(sample, diff=not detached, name="sample"))
    return _public_ray_from_native(ray, detached=detached)


@dr.wrap(source="torch", target="drjit")
def _camera_sample_edge_impl(state: _CameraState, mesh_states: list[_MeshState], sample1: Any) -> Any:
    scene = _build_native_scene(mesh_states, preserve_gradients=True)
    camera = _build_native_camera(state, preserve_gradients=True)
    camera.prepare_edges(scene)
    return _primary_edge_sample_from_native(camera.sample_edge(_tensor_to_scalar_array(sample1, diff=False, name="sample1")))


@dr.wrap(source="torch", target="drjit")
def _camera_render_impl(state: _CameraState, mesh_states: list[_MeshState], background: float) -> Any:
    scene = _build_native_scene(mesh_states, preserve_gradients=True)
    camera = _build_native_camera(state, preserve_gradients=True)
    return camera.render(scene, background)


@dr.wrap(source="torch", target="drjit")
def _camera_render_grad_impl(state: _CameraState, mesh_states: list[_MeshState], spp: int, background: float) -> Any:
    scene = _build_native_scene(mesh_states, preserve_gradients=True)
    camera = _build_native_camera(state, preserve_gradients=True)
    return camera.render_grad(scene, spp, background)


class Mesh:
    def __init__(
        self,
        v: Any = None,
        f: Any = None,
        uv: Any = None,
        f_uv: Any = None,
        verbose: bool = False,
    ):
        self._configured = False
        if v is None and f is None:
            self._state = _MeshState(verbose=bool(verbose))
            return
        if v is None or f is None:
            raise TypeError("Mesh() expects both v and f, or neither.")
        vertices = _normalize_vector_tensor(v, "v", 3, _torch.float32)
        faces = _normalize_vector_tensor(f, "f", 3, _torch.int32)
        device = vertices.device
        self._state = _MeshState(
            vertex_positions=vertices,
            face_indices=faces,
            vertex_uv=_normalize_vector_tensor(uv, "uv", 2, _torch.float32) if uv is not None else _empty_vec2(device),
            face_uv_indices=_normalize_vector_tensor(f_uv, "f_uv", 3, _torch.int32) if f_uv is not None else _empty_idx3(device),
            verbose=bool(verbose),
        )

    def _invalidate(self) -> None:
        self._configured = False

    def _native_detached(self) -> Any:
        mesh = _build_native_mesh(self._state, preserve_gradients=False)
        mesh.configure()
        return mesh

    def configure(self) -> None:
        self._native_detached()
        self._configured = True

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

    def edge_indices(self) -> tuple[_torch.Tensor, _torch.Tensor, _torch.Tensor, _torch.Tensor, _torch.Tensor]:
        edge_indices = self._native_detached().edge_indices()
        return tuple(_scalar_array_to_tensor(value).torch() for value in edge_indices)

    def secondary_edges(self) -> SecondaryEdgeInfo:
        return _to_torch_struct(_secondary_edges_from_native(self._native_detached().secondary_edges()))

    @property
    def num_vertices(self) -> int:
        return 0 if self._state.vertex_positions is None else int(self._state.vertex_positions.shape[0])

    @property
    def num_faces(self) -> int:
        return 0 if self._state.face_indices is None else int(self._state.face_indices.shape[0])

    @property
    def to_world(self) -> _torch.Tensor:
        return _mesh_to_world_tensor(self._state, "to_world")

    @to_world.setter
    def to_world(self, value: Any) -> None:
        self._state.to_world = _normalize_matrix_tensor(value, "to_world")
        self._invalidate()

    @property
    def to_world_left(self) -> _torch.Tensor:
        return _mesh_to_world_tensor(self._state, "to_world_left")

    @to_world_left.setter
    def to_world_left(self, value: Any) -> None:
        self._state.to_world_left = _normalize_matrix_tensor(value, "to_world_left")
        self._invalidate()

    @property
    def to_world_right(self) -> _torch.Tensor:
        return _mesh_to_world_tensor(self._state, "to_world_right")

    @to_world_right.setter
    def to_world_right(self, value: Any) -> None:
        self._state.to_world_right = _normalize_matrix_tensor(value, "to_world_right")
        self._invalidate()

    @property
    def vertex_positions(self) -> _torch.Tensor:
        return _torch_or_default(self._state.vertex_positions, _empty_vec3(_mesh_device(self._state)))

    @vertex_positions.setter
    def vertex_positions(self, value: Any) -> None:
        self._state.vertex_positions = _normalize_vector_tensor(value, "vertex_positions", 3, _torch.float32)
        self._invalidate()

    @property
    def vertex_positions_world(self) -> _torch.Tensor:
        return _vec3_to_tensor(self._native_detached().vertex_positions_world).torch()

    @property
    def vertex_normals(self) -> _torch.Tensor:
        return _vec3_to_tensor(self._native_detached().vertex_normals).torch()

    @property
    def vertex_uv(self) -> _torch.Tensor:
        return _torch_or_default(self._state.vertex_uv, _empty_vec2(_mesh_device(self._state)))

    @vertex_uv.setter
    def vertex_uv(self, value: Any) -> None:
        self._state.vertex_uv = _normalize_vector_tensor(value, "vertex_uv", 2, _torch.float32)
        self._invalidate()

    @property
    def face_indices(self) -> _torch.Tensor:
        return _torch_or_default(self._state.face_indices, _empty_idx3(_mesh_device(self._state)))

    @face_indices.setter
    def face_indices(self, value: Any) -> None:
        self._state.face_indices = _normalize_vector_tensor(value, "face_indices", 3, _torch.int32)
        self._invalidate()

    @property
    def face_uv_indices(self) -> _torch.Tensor:
        return _torch_or_default(self._state.face_uv_indices, _empty_idx3(_mesh_device(self._state)))

    @face_uv_indices.setter
    def face_uv_indices(self, value: Any) -> None:
        self._state.face_uv_indices = _normalize_vector_tensor(value, "face_uv_indices", 3, _torch.int32)
        self._invalidate()

    @property
    def use_face_normals(self) -> bool:
        return bool(self._state.use_face_normals)

    @use_face_normals.setter
    def use_face_normals(self, value: bool) -> None:
        self._state.use_face_normals = bool(value)
        self._invalidate()

    @property
    def edges_enabled(self) -> bool:
        return bool(self._state.edges_enabled)

    @edges_enabled.setter
    def edges_enabled(self, value: bool) -> None:
        self._state.edges_enabled = bool(value)
        self._invalidate()

    def __repr__(self) -> str:
        return (
            f"Mesh(num_vertices={self.num_vertices}, num_faces={self.num_faces}, "
            f"use_face_normals={self.use_face_normals}, edges_enabled={self.edges_enabled})"
        )


class Scene:
    def __init__(self):
        self._records: list[_SceneMeshRecord] = []
        self._ready = False
        self._pending_updates = False
        self._version = 0
        self._native_scene: Any | None = None
        self._last_commit_profile = SceneCommitProfile()

    def _mesh_states(self) -> list[_MeshState]:
        return [record.state for record in self._records]

    def _require_ready(self) -> None:
        if not self._ready:
            raise RuntimeError("Scene is not configured. Call configure() before querying.")

    def _require_query_ready(self) -> None:
        self._require_ready()
        if self._pending_updates:
            raise RuntimeError("Scene has pending updates. Call commit_updates() before querying.")

    def _validate_mesh_id(self, mesh_id: int) -> _SceneMeshRecord:
        if mesh_id < 0 or mesh_id >= len(self._records):
            raise IndexError(f"Invalid mesh_id: {mesh_id}")
        return self._records[mesh_id]

    def add_mesh(self, mesh: Mesh, dynamic: bool = False) -> int:
        if not isinstance(mesh, Mesh):
            raise TypeError("Scene.add_mesh() expects a rayd.torch.Mesh.")
        self._records.append(_SceneMeshRecord(mesh._state.copy(), bool(dynamic)))
        self._ready = False
        self._pending_updates = False
        self._native_scene = None
        return len(self._records) - 1

    def configure(self) -> None:
        native_scene = _native.Scene()
        for record in self._records:
            native_scene.add_mesh(_build_native_mesh(record.state, preserve_gradients=False), record.dynamic)
        native_scene.configure()
        self._native_scene = native_scene
        self._ready = True
        self._pending_updates = False
        self._version += 1

    def update_mesh_vertices(self, mesh_id: int, positions: Any) -> None:
        record = self._validate_mesh_id(mesh_id)
        if not record.dynamic:
            raise RuntimeError("Scene.update_mesh_vertices(): target mesh is not dynamic.")
        self._require_ready()
        new_positions = _normalize_vector_tensor(positions, "positions", 3, _torch.float32)
        if record.state.vertex_positions is not None and new_positions.shape[0] != record.state.vertex_positions.shape[0]:
            raise RuntimeError("Scene.update_mesh_vertices(): vertex count must remain unchanged.")
        record.state.vertex_positions = new_positions
        if self._native_scene is not None:
            self._native_scene.update_mesh_vertices(mesh_id, _tensor_to_vec3(new_positions, diff=False, name="positions"))
        self._pending_updates = True

    def set_mesh_transform(self, mesh_id: int, mat: Any, set_left: bool = True) -> None:
        record = self._validate_mesh_id(mesh_id)
        if not record.dynamic:
            raise RuntimeError("Scene.set_mesh_transform(): target mesh is not dynamic.")
        self._require_ready()
        matrix = _normalize_matrix_tensor(mat, "mat")
        if set_left:
            record.state.to_world_left = matrix
        else:
            record.state.to_world_right = matrix
        if self._native_scene is not None:
            self._native_scene.set_mesh_transform(mesh_id, _tensor_to_matrix4(matrix, diff=False, name="mat"), set_left)
        self._pending_updates = True

    def append_mesh_transform(self, mesh_id: int, mat: Any, append_left: bool = True) -> None:
        record = self._validate_mesh_id(mesh_id)
        if not record.dynamic:
            raise RuntimeError("Scene.append_mesh_transform(): target mesh is not dynamic.")
        self._require_ready()
        matrix = _normalize_matrix_tensor(mat, "mat")
        current = _mesh_to_world_tensor(record.state, "to_world_left" if append_left else "to_world_right")
        if append_left:
            record.state.to_world_left = matrix @ current
        else:
            record.state.to_world_right = current @ matrix
        if self._native_scene is not None:
            self._native_scene.append_mesh_transform(mesh_id, _tensor_to_matrix4(matrix, diff=False, name="mat"), append_left)
        self._pending_updates = True

    def commit_updates(self) -> None:
        self._require_ready()
        if self._native_scene is None:
            raise RuntimeError("Scene.commit_updates(): internal detached scene is unavailable.")
        self._native_scene.commit_updates()
        self._last_commit_profile = SceneCommitProfile(self._native_scene.last_commit_profile)
        self._pending_updates = False
        self._version += 1

    def is_ready(self) -> bool:
        return self._ready

    def has_pending_updates(self) -> bool:
        return self._pending_updates

    @property
    def last_commit_profile(self) -> SceneCommitProfile:
        return SceneCommitProfile(self._last_commit_profile)

    @property
    def num_meshes(self) -> int:
        return len(self._records)

    def intersect(self, ray: Ray | RayDetached, active: Any = True) -> Intersection | IntersectionDetached:
        self._require_query_ready()
        if not isinstance(ray, (Ray, RayDetached)):
            raise TypeError("Scene.intersect() expects a rayd.torch.Ray or rayd.torch.RayDetached.")
        return _scene_intersect_impl(self._mesh_states(), ray, _normalize_active_tensor(active, _ray_batch_size(ray)))

    def shadow_test(self, ray: Ray | RayDetached, active: Any = True) -> _torch.Tensor:
        self._require_query_ready()
        if not isinstance(ray, (Ray, RayDetached)):
            raise TypeError("Scene.shadow_test() expects a rayd.torch.Ray or rayd.torch.RayDetached.")
        return _scene_shadow_test_impl(self._mesh_states(), ray, _normalize_active_tensor(active, _ray_batch_size(ray)))

    def nearest_edge(self, query: Any, active: Any = True) -> Any:
        self._require_query_ready()
        if isinstance(query, (Ray, RayDetached)):
            return _scene_nearest_ray_impl(self._mesh_states(), query, _normalize_active_tensor(active, _ray_batch_size(query)))
        point = _normalize_vector_tensor(query, "point", 3, _torch.float32)
        return _scene_nearest_point_impl(self._mesh_states(), point, _normalize_active_tensor(active, point.shape[0]))

    def __repr__(self) -> str:
        return f"Scene(num_meshes={self.num_meshes}, ready={self._ready}, pending_updates={self._pending_updates})"


class Camera:
    def __init__(self, *args: float):
        if len(args) in (0, 3):
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
            raise TypeError("Camera() expects either (fov_x, near_clip, far_clip) or (fx, fy, cx, cy, near_clip, far_clip).")

        self._configured = False
        self._version = 0
        self._prepared = False
        self._prepared_scene_ref: Scene | None = None
        self._prepared_scene_version: int | None = None
        self._prepared_camera_version: int | None = None

    def _invalidate(self) -> None:
        self._configured = False
        self._version += 1
        self._prepared = False
        self._prepared_scene_ref = None
        self._prepared_scene_version = None
        self._prepared_camera_version = None

    def _native_detached(self) -> Any:
        return _build_native_camera(self._state, preserve_gradients=False)

    def _require_configured(self) -> None:
        if not self._configured:
            raise RuntimeError("Camera is not configured. Call configure() before querying.")

    def _default_transform(self) -> _torch.Tensor:
        return _identity_matrix()

    def configure(self, cache: bool = True) -> None:
        self._state.cache = bool(cache)
        self._native_detached()
        self._configured = True
        self._version += 1
        self._prepared = False
        self._prepared_scene_ref = None
        self._prepared_scene_version = None
        self._prepared_camera_version = None

    def render(self, scene: Scene, background: float = 0.0) -> _torch.Tensor:
        self._require_configured()
        if not isinstance(scene, Scene):
            raise TypeError("Camera.render() expects a rayd.torch.Scene.")
        scene._require_query_ready()
        return _camera_render_impl(self._state, scene._mesh_states(), float(background))

    def render_grad(self, scene: Scene, spp: int = 4, background: float = 0.0) -> _torch.Tensor:
        self._require_configured()
        if not isinstance(scene, Scene):
            raise TypeError("Camera.render_grad() expects a rayd.torch.Scene.")
        scene._require_query_ready()
        return _camera_render_grad_impl(self._state, scene._mesh_states(), int(spp), float(background))

    def prepare_edges(self, scene: Scene) -> None:
        self._require_configured()
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

    def sample_ray(self, sample: Any) -> Ray | RayDetached:
        self._require_configured()
        return _camera_sample_ray_impl(self._state, _normalize_vector_tensor(sample, "sample", 2, _torch.float32))

    def sample_edge(self, sample1: Any) -> PrimaryEdgeSample:
        self._require_configured()
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
        self._require_configured()
        return _matrix4_to_tensor(self._native_detached().camera_to_sample).torch()

    @property
    def sample_to_camera(self) -> _torch.Tensor:
        self._require_configured()
        return _matrix4_to_tensor(self._native_detached().sample_to_camera).torch()

    @property
    def world_to_sample(self) -> _torch.Tensor:
        self._require_configured()
        return _matrix4_to_tensor(self._native_detached().world_to_sample).torch()

    @property
    def sample_to_world(self) -> _torch.Tensor:
        self._require_configured()
        return _matrix4_to_tensor(self._native_detached().sample_to_world).torch()

    def __repr__(self) -> str:
        return f"Camera(width={self.width}, height={self.height}, configured={self._configured})"


__all__ = [
    "Camera",
    "Intersection",
    "IntersectionDetached",
    "Mesh",
    "NearestPointEdge",
    "NearestPointEdgeDetached",
    "NearestRayEdge",
    "NearestRayEdgeDetached",
    "PrimaryEdgeSample",
    "Ray",
    "RayDetached",
    "Scene",
    "SceneCommitProfile",
    "SecondaryEdgeInfo",
]
