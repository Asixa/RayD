from __future__ import annotations

from typing import Any

from ._env import dr, _cuda, _cuda_ad
from ._util import _is_torch_tensor, _shape_tuple, _infer_diff


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
    return target(arr[:, 0].array, arr[:, 1].array)


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
    return target(arr[:, 0].array, arr[:, 1].array, arr[:, 2].array)


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
    return _cuda.Array3i(arr[:, 0].array, arr[:, 1].array, arr[:, 2].array)


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
