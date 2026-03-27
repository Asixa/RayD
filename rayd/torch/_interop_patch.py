from __future__ import annotations

from typing import Any


def _flatten_fixed(a: Any, flat: list[Any], desc: list[Any], /) -> None:
    tp = type(a)
    desc.append(tp)
    if tp is list or tp is tuple:
        desc.append(len(a))
        for value in a:
            _flatten_fixed(value, flat, desc)
    elif tp is dict:
        desc.append(tuple(a.keys()))
        for value in a.values():
            _flatten_fixed(value, flat, desc)
    else:
        struct_desc = getattr(tp, "DRJIT_STRUCT", None)
        if type(struct_desc) is dict:
            for key in struct_desc:
                _flatten_fixed(getattr(a, key), flat, desc)
        else:
            flat.append(a)


def _unflatten_fixed(flat: list[Any], desc: list[Any], /) -> Any:
    tp = desc.pop()
    if tp is list or tp is tuple:
        n = desc.pop()
        return tp(_unflatten_fixed(flat, desc) for _ in range(n))
    if tp is dict:
        keys = desc.pop()
        return {key: _unflatten_fixed(flat, desc) for key in keys}

    struct_desc = getattr(tp, "DRJIT_STRUCT", None)
    if type(struct_desc) is dict:
        result = tp()
        for key in struct_desc:
            setattr(result, key, _unflatten_fixed(flat, desc))
        return result
    return flat.pop()


_flatten_fixed.__rayd_patched__ = True
_unflatten_fixed.__rayd_patched__ = True


def _needs_interop_patch(interop: Any) -> bool:
    flatten = getattr(interop, "_flatten", None)
    unflatten = getattr(interop, "_unflatten", None)
    if flatten is None or unflatten is None:
        return False
    if getattr(flatten, "__rayd_patched__", False) and getattr(unflatten, "__rayd_patched__", False):
        return False

    class _Probe:
        DRJIT_STRUCT = {"value": int}

        def __init__(self):
            self.value = 1

    try:
        flat: list[Any] = []
        desc: list[Any] = []
        flatten(_Probe(), flat, desc)
    except AttributeError as exc:
        return "append" in str(exc)
    except Exception:
        return False
    return False


def install_drjit_interop_patch(interop: Any | None = None) -> bool:
    if interop is None:
        import drjit.interop as interop

    if not _needs_interop_patch(interop):
        return False

    interop._flatten = _flatten_fixed
    interop._unflatten = _unflatten_fixed
    return True


__all__ = [
    "_flatten_fixed",
    "_needs_interop_patch",
    "_unflatten_fixed",
    "install_drjit_interop_patch",
]
