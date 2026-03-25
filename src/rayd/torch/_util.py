from __future__ import annotations

from typing import Any

from ._env import _torch, dr


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
        return bool(dr.is_diff_v(tp) and dr.grad_enabled(value))
    return False


def _has_diff_fields(obj: Any) -> bool:
    for name in getattr(type(obj), 'DRJIT_STRUCT', {}):
        if _infer_diff(getattr(obj, name)):
            return True
    return False


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
