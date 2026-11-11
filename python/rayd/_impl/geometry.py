# Copyright Xingyu Chen.
# Implements shared Python support for geometry.

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from enum import Enum, IntEnum, IntFlag
import math
from typing import TYPE_CHECKING, Any, TypeVar
import torch

from .runtime import C as _C
from .runtime import core_ops

if TYPE_CHECKING:
    from collections.abc import Callable

try:
    from enum import StrEnum as _StrEnum
except ImportError:  # Python 3.10 has no StrEnum; a str-mixin Enum compares equal to its value the same way.

    class _StrEnum(str, Enum):  # type: ignore[no-redef]
        def __str__(self) -> str:
            return str(self.value)

        def __hash__(self) -> int:
            return hash(self.value)


_CONTRACT_VALUES = {
    "invalid_signed_id": -1,
    "invalid_unsigned_id": 0xFFFFFFFF,
    "general_epsilon": 1.0e-5,
    "ray_epsilon": 1.0e-3,
    "shadow_epsilon": 1.0e-3,
    "edge_epsilon": 1.0e-5,
    "small_epsilon": 1.0e-6,
    "vacuum_permittivity": 8.854187817e-12,
    "speed_of_light": 299792458.0,
    "ray_flags_none": 0x00,
    "ray_flags_geometric": 0x01,
    "ray_flags_shading_n": 0x02,
    "ray_flags_uv": 0x04,
    "ray_flags_all": 0x07,
    "intersection_field_count": 10,
    "nearest_point_edge_field_count": 8,
    "nearest_ray_edge_field_count": 9,
}


def _validate_native_contract_values() -> None:
    if _C is None or not hasattr(_C, "contract_values"):
        return
    native = _C.contract_values()
    for key, expected in _CONTRACT_VALUES.items():
        actual = native[key]
        if isinstance(expected, float):
            if not math.isclose(actual, expected, rel_tol=1.0e-7, abs_tol=0.0):
                raise RuntimeError(f"RayD Torch native contract mismatch for {key}.")
        elif actual != expected:
            raise RuntimeError(f"RayD Torch native contract mismatch for {key}.")


_validate_native_contract_values()


RayFlags = IntFlag(
    "RayFlags",
    {
        "Empty": _CONTRACT_VALUES["ray_flags_none"],
        "Geometric": _CONTRACT_VALUES["ray_flags_geometric"],
        "ShadingN": _CONTRACT_VALUES["ray_flags_shading_n"],
        "UV": _CONTRACT_VALUES["ray_flags_uv"],
        "All": _CONTRACT_VALUES["ray_flags_all"],
    },
)


class TraceBackend(_StrEnum):
    """Triangle-intersection backend accepted by ``Scene(trace_backend=...)``."""

    AUTO = "auto"
    OPTIX = "optix"
    CUDA = "cuda"


class EdgeBVHBackend(_StrEnum):
    """Edge-query acceleration backend accepted by ``Scene(edge_bvh_backend=...)``."""

    AUTO = "auto"
    OPTIX = "optix"
    CUDA = "cuda"


class AccumStrategy(IntEnum):
    """Grid write strategy of ``Scene.accum_refl()``.

    ``AUTO`` stages per-sample writes and reduces them when the batch is large
    enough for the grid, and otherwise writes with atomics; ``ATOMIC`` and
    ``STAGED`` force one of the two.
    """

    AUTO = 0
    ATOMIC = 1
    STAGED = 2


def _describe_value(value: object) -> str:
    """Describe one record field without reading tensor data back to the host."""

    if isinstance(value, torch.Tensor):
        dtype = str(value.dtype).removeprefix("torch.")
        shape = ", ".join(str(int(size)) for size in value.shape)
        grad = ", requires_grad" if value.requires_grad else ""
        return f"{dtype}[{shape}] {value.device}{grad}"
    return repr(value)


_RecordT = TypeVar("_RecordT", bound="_Record")


class _Record:
    """Shared compact ``repr``, ``to(device)``, and ``detach()`` for Torch result records.

    A record lists its fields through ``_record_items`` (dataclass fields by
    default) and rebuilds itself from a field mapping through
    ``_record_rebuild``. Tensor fields are mapped, nested records recurse, and
    every other value is carried over unchanged.
    """

    __slots__ = ()

    def _record_items(self) -> tuple[tuple[str, Any], ...]:
        return tuple((field.name, getattr(self, field.name)) for field in dataclasses.fields(self))  # type: ignore[arg-type]

    def _record_rebuild(self: _RecordT, values: dict[str, Any]) -> _RecordT:
        return type(self)(**values)

    def _record_name(self) -> str:
        return type(self).__name__

    def __repr__(self) -> str:
        body = ", ".join(f"{name}={_describe_value(value)}" for name, value in self._record_items())
        return f"{self._record_name()}({body})"

    def _map_tensors(self: _RecordT, function: Callable[[torch.Tensor], torch.Tensor]) -> _RecordT:
        def convert(value: Any) -> Any:
            if isinstance(value, torch.Tensor):
                return function(value)
            if isinstance(value, _Record):
                return value._map_tensors(function)
            return value

        return self._record_rebuild({name: convert(value) for name, value in self._record_items()})

    def to(self: _RecordT, device: torch.device | str | int) -> _RecordT:
        """Return a new record with every tensor field moved to ``device``."""

        return self._map_tensors(lambda value: value.to(device))

    def detach(self: _RecordT) -> _RecordT:
        """Return a new record with every tensor field detached from autograd."""

        return self._map_tensors(lambda value: value.detach())


class _ValidRecord(_Record):
    """A result record whose per-row validity is the boolean tensor ``valid``."""

    __slots__ = ()

    valid: torch.Tensor

    def is_valid(self) -> torch.Tensor:
        return self.valid


def _complex_field(name: str) -> property:
    """Expose the split ``<name>_re``/``<name>_im`` tensors as one complex tensor."""

    def read(self: Any) -> torch.Tensor:
        return torch.complex(getattr(self, f"{name}_re"), getattr(self, f"{name}_im"))

    read.__doc__ = f"Complex ``{name}`` assembled from ``{name}_re`` and ``{name}_im``."
    return property(read)


def _require_float_cuda_tensor(value: torch.Tensor, name: str, shape_last: int | None) -> None:
    if value.device.type != "cuda":
        raise TypeError(f"{name} must be CUDA.")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} must be torch.float32.")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous.")
    if shape_last is not None and (value.ndim != 2 or value.shape[1] != shape_last):
        raise ValueError(f"{name} must have shape (N, {shape_last}).")


@dataclass(frozen=True, slots=True)
class Ray:
    o: torch.Tensor
    d: torch.Tensor
    tmax: torch.Tensor | None = None

    def __post_init__(self) -> None:
        _require_float_cuda_tensor(self.o, "Ray.o", 3)
        _require_float_cuda_tensor(self.d, "Ray.d", 3)
        if self.o.shape[0] != self.d.shape[0]:
            raise ValueError("Ray.o and Ray.d must have the same batch size.")
        if self.tmax is None:
            object.__setattr__(self, "tmax", torch.empty((0,), device=self.o.device, dtype=self.o.dtype))
        else:
            _require_float_cuda_tensor(self.tmax, "Ray.tmax", None)
            if self.tmax.ndim != 1 or (self.tmax.numel() != 0 and self.tmax.shape[0] != self.o.shape[0]):
                raise ValueError("Ray.tmax must be empty or have shape (N,).")


@dataclass(frozen=True, slots=True, repr=False)
class Intersection(_Record):
    t: torch.Tensor
    p: torch.Tensor
    n: torch.Tensor
    geo_n: torch.Tensor
    uv: torch.Tensor
    barycentric: torch.Tensor
    shape_id: torch.Tensor
    prim_id: torch.Tensor
    local_prim_id: torch.Tensor
    global_prim_id: torch.Tensor

    @property
    def instance_id(self) -> torch.Tensor:
        """Scene instance id; an alias of ``shape_id`` for instanced scenes."""

        return self.shape_id

    @property
    def valid(self) -> torch.Tensor:
        """Per-ray hit mask; the same tensor ``is_valid()`` returns."""

        return self.is_valid()

    def is_valid(self) -> torch.Tensor:
        if self.t.device.type == "cuda":
            return core_ops().intersection_valid(self.t, self.shape_id)
        if self.shape_id.numel() != self.t.numel():
            return torch.isfinite(self.t)
        return self.shape_id >= 0

    def _record_rebuild(self, values: dict[str, Any]) -> Intersection:
        # The lazy subclasses materialize into the plain record.
        return Intersection(**values)

    def _record_name(self) -> str:
        return "Intersection"


class _DeferredIntersection(Intersection):
    """Base of the intersection results that compute their fields on first access.

    They are ``Intersection`` instances, so ``isinstance`` and the field surface
    hold, but their dataclass slots stay unset: every field is a property that
    loads on demand. ``repr`` shows only what is already loaded.
    """

    __slots__ = ()

    def __setattr__(self, name: str, value: Any) -> None:
        # The frozen dataclass `__setattr__` refuses every subclass assignment
        # under Python 3.11 slots; the private load state stays assignable
        # while the public fields stay read-only.
        if not name.startswith("_"):
            raise dataclasses.FrozenInstanceError(f"cannot assign to field {name!r}")
        object.__setattr__(self, name, value)

    def _loaded_t(self) -> torch.Tensor | None:
        raise NotImplementedError

    def _loaded_full(self) -> Intersection | None:
        raise NotImplementedError

    def __repr__(self) -> str:
        full = self._loaded_full()
        if full is not None:
            return repr(full)
        t = self._loaded_t()
        shown = "<not loaded>" if t is None else _describe_value(t)
        return f"Intersection(t={shown}, <remaining fields load on access>)"


class _LazyIntersection(_DeferredIntersection):
    __slots__ = ("_load_t", "_load_full", "_t", "_full")

    def __init__(self, load_t: Callable[[], torch.Tensor], load_full: Callable[[], Intersection]) -> None:
        self._load_t = load_t
        self._load_full = load_full
        self._t: torch.Tensor | None = None
        self._full: Intersection | None = None

    def _loaded_t(self) -> torch.Tensor | None:
        return self._t

    def _loaded_full(self) -> Intersection | None:
        return self._full

    def _ensure_full(self) -> Intersection:
        if self._full is None:
            self._full = self._load_full()
        return self._full

    @property
    def t(self) -> torch.Tensor:  # type: ignore[override]
        if self._t is not None:
            return self._t
        if self._full is not None:
            return self._full.t
        self._t = self._load_t()
        return self._t

    @property
    def p(self) -> torch.Tensor:  # type: ignore[override]
        return self._ensure_full().p

    @property
    def n(self) -> torch.Tensor:  # type: ignore[override]
        return self._ensure_full().n

    @property
    def geo_n(self) -> torch.Tensor:  # type: ignore[override]
        return self._ensure_full().geo_n

    @property
    def uv(self) -> torch.Tensor:  # type: ignore[override]
        return self._ensure_full().uv

    @property
    def barycentric(self) -> torch.Tensor:  # type: ignore[override]
        return self._ensure_full().barycentric

    @property
    def shape_id(self) -> torch.Tensor:  # type: ignore[override]
        return self._ensure_full().shape_id

    @property
    def prim_id(self) -> torch.Tensor:  # type: ignore[override]
        return self._ensure_full().prim_id

    @property
    def local_prim_id(self) -> torch.Tensor:  # type: ignore[override]
        return self._ensure_full().local_prim_id

    @property
    def global_prim_id(self) -> torch.Tensor:  # type: ignore[override]
        return self._ensure_full().global_prim_id

    def is_valid(self) -> torch.Tensor:
        return self._ensure_full().is_valid()


class _ReducedIntersection(_DeferredIntersection):
    __slots__ = ("_scene", "_fields")

    def __init__(self, scene_handle: int, t: torch.Tensor) -> None:
        self._scene = scene_handle
        # `t` is the inherited dataclass slot; the frozen dataclass refuses a
        # plain assignment to a field name.
        object.__setattr__(self, "t", t)
        self._fields: tuple[torch.Tensor, ...] | None = None

    def _loaded_t(self) -> torch.Tensor | None:
        return self.t

    def _loaded_full(self) -> Intersection | None:
        return None

    def _empty_fields(self) -> tuple[torch.Tensor, ...]:
        if self._fields is None:
            self._fields = torch.ops.rayd_torch.intersection_empty_fields(self._scene, self.t)
        return self._fields

    @property
    def p(self) -> torch.Tensor:  # type: ignore[override]
        return self._empty_fields()[0]

    @property
    def n(self) -> torch.Tensor:  # type: ignore[override]
        return self._empty_fields()[1]

    @property
    def geo_n(self) -> torch.Tensor:  # type: ignore[override]
        return self._empty_fields()[2]

    @property
    def uv(self) -> torch.Tensor:  # type: ignore[override]
        return self._empty_fields()[3]

    @property
    def barycentric(self) -> torch.Tensor:  # type: ignore[override]
        return self._empty_fields()[4]

    @property
    def shape_id(self) -> torch.Tensor:  # type: ignore[override]
        return self._empty_fields()[5]

    @property
    def prim_id(self) -> torch.Tensor:  # type: ignore[override]
        return self._empty_fields()[6]

    @property
    def local_prim_id(self) -> torch.Tensor:  # type: ignore[override]
        return self._empty_fields()[7]

    @property
    def global_prim_id(self) -> torch.Tensor:  # type: ignore[override]
        return self._empty_fields()[8]

    def is_valid(self) -> torch.Tensor:
        return core_ops().intersection_valid(self.t, self.shape_id)


@dataclass(frozen=True, repr=False)
class NearestPointEdge(_Record):
    distance: torch.Tensor
    edge_point: torch.Tensor
    edge_t: torch.Tensor
    shape_id: torch.Tensor
    edge_id: torch.Tensor
    global_edge_id: torch.Tensor


@dataclass(frozen=True, repr=False)
class NearestEdgesTopK(_ValidRecord):
    query_count: int
    k: int
    valid: torch.Tensor
    distances: torch.Tensor
    points: torch.Tensor
    edge_t: torch.Tensor
    edge_points: torch.Tensor
    shape_ids: torch.Tensor
    edge_ids: torch.Tensor
    global_edge_ids: torch.Tensor
    is_boundary: torch.Tensor


@dataclass(frozen=True, repr=False)
class SegmentPairVisibility(_Record):
    ray_count: int
    visible_a: torch.Tensor
    visible_b: torch.Tensor


@dataclass(frozen=True, repr=False)
class AxialEdgeVisibility(_Record):
    state_count: int
    any_visible: torch.Tensor


@dataclass(frozen=True, repr=False)
class SegmentChainVisibility(_Record):
    chain_count: int
    max_segments: int
    all_visible: torch.Tensor
    first_blocked_segment: torch.Tensor
    first_blocked_prim: torch.Tensor


@dataclass(frozen=True, repr=False)
class NearestRayEdge(_Record):
    distance: torch.Tensor
    ray_t: torch.Tensor
    point: torch.Tensor
    edge_t: torch.Tensor
    edge_point: torch.Tensor
    shape_id: torch.Tensor
    edge_id: torch.Tensor
    global_edge_id: torch.Tensor


class ReflectionChain(_ValidRecord):
    __slots__ = ("_valid", "_t", "_image_sources", "_prim_ids", "_loader")

    def __init__(
        self,
        valid: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
        image_sources: torch.Tensor | None = None,
        prim_ids: torch.Tensor | None = None,
        *,
        loader: (Callable[[bool], tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]] | None) = None,
    ) -> None:
        self._valid = valid
        self._t = t
        self._image_sources = image_sources
        self._prim_ids = prim_ids
        self._loader = loader

    def _ensure_reduced(self) -> None:
        if self._valid is not None and self._t is not None and self._prim_ids is not None:
            return
        if self._loader is None:
            raise RuntimeError("ReflectionChain has no trace data loader.")
        valid, t, image_sources, prim_ids = self._loader(False)
        self._valid = valid
        self._t = t
        self._prim_ids = prim_ids
        if image_sources is not None:
            self._image_sources = image_sources

    def _ensure_full(self) -> None:
        if self._image_sources is not None:
            return
        if self._loader is None:
            raise RuntimeError("ReflectionChain has no image-source data.")
        valid, t, image_sources, prim_ids = self._loader(True)
        self._valid = valid
        self._t = t
        self._image_sources = image_sources
        self._prim_ids = prim_ids

    @property
    def valid(self) -> torch.Tensor:
        self._ensure_reduced()
        return self._valid

    @property
    def t(self) -> torch.Tensor:
        self._ensure_reduced()
        return self._t

    @property
    def image_sources(self) -> torch.Tensor:
        self._ensure_full()
        return self._image_sources

    @property
    def prim_ids(self) -> torch.Tensor:
        self._ensure_reduced()
        return self._prim_ids

    def _record_items(self) -> tuple[tuple[str, Any], ...]:
        # Moving or detaching needs every field, so it runs the full trace once.
        image_sources = self.image_sources
        return (("valid", self.valid), ("t", self.t), ("image_sources", image_sources), ("prim_ids", self.prim_ids))

    def __repr__(self) -> str:
        # Show only the fields already traced: a repr must not launch the trace.
        loaded = (
            ("valid", self._valid),
            ("t", self._t),
            ("image_sources", self._image_sources),
            ("prim_ids", self._prim_ids),
        )
        body = ", ".join(
            f"{name}={'<not loaded>' if value is None else _describe_value(value)}" for name, value in loaded
        )
        return f"ReflectionChain({body})"


@dataclass(frozen=True, repr=False)
class ReflEpcField(_ValidRecord):
    field_re: torch.Tensor
    field_im: torch.Tensor
    path_length: torch.Tensor
    valid: torch.Tensor
    resolved_prim_ids: torch.Tensor

    field = _complex_field("field")


@dataclass(frozen=True)
class ReflEpcOptions:
    expected_prim_ids: torch.Tensor
    direct_plane_points: torch.Tensor
    direct_plane_normals: torch.Tensor
    surface_group_id: torch.Tensor
    surface_group_size: torch.Tensor
    surface_group_members: torch.Tensor
    visibility_ignore_mode: str = "primitive"
    plane_tolerance: float = 1.0e-5


@dataclass(frozen=True, repr=False)
class ReflEpc(_ValidRecord):
    ray_count: int
    max_bounces: int
    valid: torch.Tensor
    path_length: torch.Tensor
    resolved_prim_ids: torch.Tensor
    surface_group_ids: torch.Tensor
    hit_points: torch.Tensor
    normals: torch.Tensor


@dataclass(frozen=True)
class AccumGrid:
    axis: int = 2
    position: float = 0.0
    coord0_min: float = 0.0
    coord0_max: float = 0.0
    coord1_min: float = 0.0
    coord1_max: float = 0.0
    resolution0: int = 0
    resolution1: int = 0


@dataclass(frozen=True)
class AccumOptions:
    wavelength: float = 1.0
    solid_angle_per_ray: float = 1.0
    collect_wedges: bool = False
    collect_wedge_prefixes: bool = False
    wedge_capacity: int = 0
    wedge_sample_stride: int = 1
    accumulation_strategy: AccumStrategy = AccumStrategy.AUTO
    compact_min_samples: int = 0
    staged_min_samples_per_cell: int = 0
    procedural_sample_count: int = 0
    include_los: bool = False

    def __post_init__(self) -> None:
        try:
            strategy = AccumStrategy(self.accumulation_strategy)
        except ValueError:
            raise ValueError(
                f"AccumOptions.accumulation_strategy must be an AccumStrategy value, got {self.accumulation_strategy!r}."
            ) from None
        object.__setattr__(self, "accumulation_strategy", strategy)


@dataclass(frozen=True)
class ReflMaterial:
    eta_r: torch.Tensor
    sigma: torch.Tensor
    mu_r: torch.Tensor
    gain: torch.Tensor
    valid: torch.Tensor

    @staticmethod
    def default(count: int, *, device: torch.device, dtype: torch.dtype = torch.float32) -> "ReflMaterial":
        return ReflMaterial(
            eta_r=torch.ones((count,), device=device, dtype=dtype),
            sigma=torch.zeros((count,), device=device, dtype=dtype),
            mu_r=torch.ones((count,), device=device, dtype=dtype),
            gain=torch.ones((count,), device=device, dtype=dtype),
            valid=torch.ones((count,), device=device, dtype=torch.bool),
        )


@dataclass(frozen=True, repr=False)
class WedgeEvents(_Record):
    capacity: int
    count: torch.Tensor
    ray_index: torch.Tensor
    hit_points: torch.Tensor
    normals: torch.Tensor
    prim_id: torch.Tensor
    directions: torch.Tensor
    source_points: torch.Tensor
    src_power: torch.Tensor
    initial_directions: torch.Tensor
    bounce_depth: torch.Tensor


@dataclass(frozen=True, repr=False)
class AccumResult(_Record):
    ray_count: int
    max_bounces: int
    grid_cell_count: int
    reflection_power: torch.Tensor
    reflection_field_x_re: torch.Tensor
    reflection_field_x_im: torch.Tensor
    reflection_field_y_re: torch.Tensor
    reflection_field_y_im: torch.Tensor
    reflection_field_z_re: torch.Tensor
    reflection_field_z_im: torch.Tensor
    reflection_count: torch.Tensor
    wedge_events: WedgeEvents

    reflection_field_x = _complex_field("reflection_field_x")
    reflection_field_y = _complex_field("reflection_field_y")
    reflection_field_z = _complex_field("reflection_field_z")


@dataclass(frozen=True)
class DfrGrid:
    axis: int = 2
    position: float = 0.0
    coord0_min: float = -1.0
    coord0_max: float = 1.0
    coord1_min: float = -1.0
    coord1_max: float = 1.0
    resolution0: int = 1
    resolution1: int = 1
    cell_area: float | None = None

    def resolved_cell_area(self) -> float:
        if self.cell_area is not None:
            return float(self.cell_area)
        span0 = float(self.coord0_max) - float(self.coord0_min)
        span1 = float(self.coord1_max) - float(self.coord1_min)
        return abs(span0 * span1) / float(int(self.resolution0) * int(self.resolution1))


@dataclass(frozen=True)
class DfrMaterial:
    eta_r: torch.Tensor
    sigma: torch.Tensor
    mu_r: torch.Tensor
    gain: torch.Tensor
    valid: torch.Tensor

    @staticmethod
    def default(count: int, *, device: torch.device, dtype: torch.dtype = torch.float32) -> "DfrMaterial":
        return DfrMaterial(
            eta_r=torch.ones((count,), device=device, dtype=dtype),
            sigma=torch.zeros((count,), device=device, dtype=dtype),
            mu_r=torch.ones((count,), device=device, dtype=dtype),
            gain=torch.ones((count,), device=device, dtype=dtype),
            valid=torch.ones((count,), device=device, dtype=torch.bool),
        )


class DfrPathLayout(IntEnum):
    """Storage order for first-order diffraction path export rows."""

    Compact = 0
    SourceLane = 1


@dataclass(frozen=True)
class DfrOptions:
    """Options of ``Scene.accum_dfr_direct()`` and ``Scene.accum_dfr()``.

    Field names and defaults follow the Dr.Jit ``DfrOptions``. ``max_order`` is
    read only by ``accum_dfr``. ``lane_offset``/``lane_count`` select the
    Monte-Carlo lane window (Torch only); the default runs every lane.
    """

    wavelength: float = 1.0
    seed: int = 0
    max_order: int = 1
    direct_samples: int = 0
    keller_samples: int = 0
    suffix_samples: int = 0
    lane_offset: int = 0
    lane_count: int = -1


@dataclass(frozen=True)
class DfrCoherentOptions:
    """Options of ``Scene.accum_dfr_coherent_direct()``.

    Field names and defaults follow the Dr.Jit ``DfrCoherentOptions``;
    ``lane_offset``/``lane_count`` select the ``(state, grid-cell)`` lane window
    (Torch only).
    """

    wavelength: float = 1.0
    select_diffraction_point: bool = True
    prefilter_visibility: bool = True
    lane_offset: int = 0
    lane_count: int = -1


@dataclass(frozen=True)
class DfrPathOptions:
    """Options of ``Scene.trace_dfr_paths()``.

    ``wavelength`` follows the Dr.Jit ``DfrPathOptions``. ``max_paths=None``
    exports every state (Dr.Jit requires a positive value). ``layout=None``
    picks ``Compact`` on one device and ``SourceLane`` on a multi-device scene.
    """

    wavelength: float = 1.0
    max_paths: int | None = None
    layout: DfrPathLayout | None = None


@dataclass(frozen=True)
class DfrStates:
    edge_index: torch.Tensor
    edge_pos: torch.Tensor
    edge_dir: torch.Tensor
    edge_t_min: torch.Tensor
    edge_t_max: torch.Tensor
    n0: torch.Tensor
    n1: torch.Tensor
    prim0: torch.Tensor
    prim1: torch.Tensor
    exterior_angle: torch.Tensor
    src: torch.Tensor
    src_power: torch.Tensor
    wi: torch.Tensor | None = None
    d0: torch.Tensor | None = None
    count: int | None = None

    @property
    def state_count(self) -> int:
        return int(self.edge_index.shape[0] if self.count is None else self.count)

    def with_default_vectors(self) -> "DfrStates":
        wi = self.wi
        d0 = self.d0
        if wi is None:
            wi = torch.zeros_like(self.edge_pos)
        if d0 is None:
            d0 = torch.zeros_like(self.edge_pos)
        return DfrStates(
            self.edge_index,
            self.edge_pos,
            self.edge_dir,
            self.edge_t_min,
            self.edge_t_max,
            self.n0,
            self.n1,
            self.prim0,
            self.prim1,
            self.exterior_angle,
            self.src,
            self.src_power,
            wi,
            d0,
            self.count,
        )


@dataclass(frozen=True, repr=False)
class DfrAccum(_Record):
    grid_cell_count: int
    power: torch.Tensor
    field_x_re: torch.Tensor
    field_x_im: torch.Tensor
    field_y_re: torch.Tensor
    field_y_im: torch.Tensor
    field_z_re: torch.Tensor
    field_z_im: torch.Tensor
    direct_count: torch.Tensor
    keller_count: torch.Tensor
    suffix_count: torch.Tensor
    vis_rejects: torch.Tensor
    edge_vis_rejects: torch.Tensor
    utd_rejects: torch.Tensor
    edge_uses: torch.Tensor

    field_x = _complex_field("field_x")
    field_y = _complex_field("field_y")
    field_z = _complex_field("field_z")


@dataclass(frozen=True, repr=False)
class DfrCoherentAccum(_Record):
    grid_cell_count: int
    direct_field_x_re: torch.Tensor
    direct_field_x_im: torch.Tensor
    direct_field_y_re: torch.Tensor
    direct_field_y_im: torch.Tensor
    direct_field_z_re: torch.Tensor
    direct_field_z_im: torch.Tensor
    multi_field_x_re: torch.Tensor
    multi_field_x_im: torch.Tensor
    multi_field_y_re: torch.Tensor
    multi_field_y_im: torch.Tensor
    multi_field_z_re: torch.Tensor
    multi_field_z_im: torch.Tensor
    direct_count: torch.Tensor
    multi_count: torch.Tensor
    vis_rejects: torch.Tensor
    utd_rejects: torch.Tensor

    direct_field_x = _complex_field("direct_field_x")
    direct_field_y = _complex_field("direct_field_y")
    direct_field_z = _complex_field("direct_field_z")
    multi_field_x = _complex_field("multi_field_x")
    multi_field_y = _complex_field("multi_field_y")
    multi_field_z = _complex_field("multi_field_z")


@dataclass(frozen=True, repr=False)
class DfrPaths(_ValidRecord):
    capacity: int
    count: torch.Tensor
    valid: torch.Tensor
    tx_id: torch.Tensor
    rx_id: torch.Tensor
    order: torch.Tensor
    edge0: torch.Tensor
    edge1: torch.Tensor
    edge2: torch.Tensor
    delay: torch.Tensor
    field_x_re: torch.Tensor
    field_x_im: torch.Tensor
    field_y_re: torch.Tensor
    field_y_im: torch.Tensor
    field_z_re: torch.Tensor
    field_z_im: torch.Tensor
    p0: torch.Tensor
    p1: torch.Tensor
    p2: torch.Tensor
    layout: DfrPathLayout = DfrPathLayout.Compact

    field_x = _complex_field("field_x")
    field_y = _complex_field("field_y")
    field_z = _complex_field("field_z")


@dataclass(frozen=True, repr=False)
class SdfIntersection(_ValidRecord):
    """ADR-0037 section 5 result of one SDF sphere trace.

    `t`, `position` and `normal` are differentiable; `valid` and `steps`
    carry no derivative. A missed lane reports `t = +inf` and exact positive
    zero in `position` and `normal`.
    """

    t: torch.Tensor
    valid: torch.Tensor
    position: torch.Tensor
    normal: torch.Tensor
    steps: torch.Tensor


@dataclass(frozen=True, repr=False)
class SceneGeometry(_Record):
    vertices: torch.Tensor
    faces: torch.Tensor
    face_normal: torch.Tensor
    shape_id: torch.Tensor
    local_prim_id: torch.Tensor
    global_prim_id: torch.Tensor
