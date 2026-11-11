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
    """Batch of rays with origins ``o``, directions ``d``, and maximum distances ``tmax``.

    Input record. ``o`` and ``d`` are contiguous CUDA ``float32`` ``(N, 3)``; directions need not be normalized.
    ``tmax`` is ``float32`` ``(N,)`` or omitted, which means infinity. Wrong devices or dtypes raise ``TypeError`` and
    wrong shapes ``ValueError``. Tensors that require grad or carry a forward-mode tangent select the differentiable
    path of a query.

    Attributes:
        o: Ray origins.
        d: Ray directions.
        tmax: Maximum ray distance; hits beyond it are ignored. An empty tensor means infinity.
    """

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
    """Closest ray-triangle hit per ray.

    Result record of ``Scene.intersect()`` and ``MixedScene.intersect()``. Missed rays have ``t = inf``, ids ``-1``, and
    zero vectors; check ``is_valid()`` before reading other fields. A reduced (``RayFlags.Empty``) or differentiable
    query returns a lazy subclass that computes the other fields on first access.

    Attributes:
        t: Hit distance along the ray, ``float32`` ``(N,)``.
        p: Hit point, ``float32`` ``(N, 3)``.
        n: Interpolated shading normal, ``float32`` ``(N, 3)``.
        geo_n: Geometric face normal, ``float32`` ``(N, 3)``.
        uv: Texture coordinates, ``float32`` ``(N, 2)``; the barycentric (u, v) when the mesh has no UVs.
        barycentric: Barycentric weights of the three triangle vertices, ``float32`` ``(N, 3)``.
        shape_id: Mesh or instance id, ``int32`` ``(N,)``.
        prim_id: Scene-global triangle id, ``int32`` ``(N,)``; equal to ``global_prim_id``.
        local_prim_id: Triangle id within its mesh, ``int32`` ``(N,)``.
        global_prim_id: Scene-global triangle id, ``int32`` ``(N,)``.
    """

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
        """Return the per-ray hit mask, ``bool`` ``(N,)``."""
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
    """Nearest scene edge to each query point.

    Result record of ``Scene.nearest_edge()`` with a point batch. Rows with no edge hold ``distance = inf`` and ids
    ``-1``.

    Attributes:
        distance: Distance to the nearest edge, ``float32`` ``(N,)``.
        edge_point: Closest point on that edge, ``float32`` ``(N, 3)``.
        edge_t: Position of ``edge_point`` along the edge, ``float32`` ``(N,)``.
        shape_id: Mesh id of the edge, ``int32`` ``(N,)``.
        edge_id: Edge id within its mesh, ``int32`` ``(N,)``.
        global_edge_id: Scene-global edge id, ``int32`` ``(N,)``.
    """

    distance: torch.Tensor
    edge_point: torch.Tensor
    edge_t: torch.Tensor
    shape_id: torch.Tensor
    edge_id: torch.Tensor
    global_edge_id: torch.Tensor


@dataclass(frozen=True, repr=False)
class NearestEdgesTopK(_ValidRecord):
    """The ``k`` nearest scene edges to each query point.

    Result record of ``Scene.nearest_edges()``. Per-slot fields have shape ``(N, k)`` (``(N, k, 3)`` for points), each
    query's results ordered nearest first; ``valid`` marks filled slots.

    Attributes:
        query_count: Number of query points ``N``.
        k: Slots per query.
        valid: Filled-slot mask, ``bool``.
        distances: Distance to each edge, ``float32``.
        points: The query point of each filled slot, ``float32``.
        edge_t: Position of the closest point along each edge, ``float32``.
        edge_points: Closest point on each edge, ``float32``.
        shape_ids: Mesh id of each edge, ``int32``.
        edge_ids: Edge id within its mesh, ``int32``.
        global_edge_ids: Scene-global edge id, ``int32``.
        is_boundary: Whether each edge has only one adjacent face, ``bool``.
    """

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
    """Visibility of two endpoints from a shared start point.

    Result record of ``Scene.visible_pair()``. The result is discrete and has no derivative.

    Attributes:
        ray_count: Number of start points ``N``.
        visible_a: Whether ``end_a`` is visible, ``bool`` ``(N,)``.
        visible_b: Whether ``end_b`` is visible, ``bool`` ``(N,)``.
    """

    ray_count: int
    visible_a: torch.Tensor
    visible_b: torch.Tensor


@dataclass(frozen=True, repr=False)
class AxialEdgeVisibility(_Record):
    """Whether each source sees any sampled point of its edge.

    Result record of ``Scene.visible_edge()``. The result is discrete and has no derivative.

    Attributes:
        state_count: Number of rows ``N``.
        any_visible: ``bool`` ``(N,)``.
    """

    state_count: int
    any_visible: torch.Tensor


@dataclass(frozen=True, repr=False)
class SegmentChainVisibility(_Record):
    """Visibility of every segment of each polyline, with the first blocker.

    Result record of ``Scene.visible_chain()``. The result is discrete and has no derivative.

    Attributes:
        chain_count: Number of chains ``N``.
        max_segments: Segments per chain, ``P - 1``.
        all_visible: Whether every used segment is unblocked, ``bool`` ``(N,)``.
        first_blocked_segment: Index of the first blocked segment, or ``-1``, ``int32`` ``(N,)``.
        first_blocked_prim: Scene-global primitive that blocks it, or ``-1``, ``int32`` ``(N,)``.
    """

    chain_count: int
    max_segments: int
    all_visible: torch.Tensor
    first_blocked_segment: torch.Tensor
    first_blocked_prim: torch.Tensor


@dataclass(frozen=True, repr=False)
class NearestRayEdge(_Record):
    """Nearest scene edge to each query ray or segment.

    Result record of ``Scene.nearest_edge()`` with a ``Ray``. Rows with no edge hold ``distance = inf`` and ids ``-1``.

    Attributes:
        distance: Distance between the ray and the nearest edge, ``float32`` ``(N,)``.
        ray_t: Distance along the ray of its closest point, ``float32`` ``(N,)``.
        point: Closest point on the ray, ``float32`` ``(N, 3)``.
        edge_t: Position of ``edge_point`` along the edge, ``float32`` ``(N,)``.
        edge_point: Closest point on the edge, ``float32`` ``(N, 3)``.
        shape_id: Mesh id of the edge, ``int32`` ``(N,)``.
        edge_id: Edge id within its mesh, ``int32`` ``(N,)``.
        global_edge_id: Scene-global edge id, ``int32`` ``(N,)``.
    """

    distance: torch.Tensor
    ray_t: torch.Tensor
    point: torch.Tensor
    edge_t: torch.Tensor
    edge_point: torch.Tensor
    shape_id: torch.Tensor
    edge_id: torch.Tensor
    global_edge_id: torch.Tensor


class ReflectionChain(_ValidRecord):
    """Reflection paths with per-bounce fields of shape ``(N, max_bounces)``.

    Result record of ``Scene.trace_reflections()`` and the SDF, surfel, and mixed reflection traces. It is lazy:
    ``valid``, ``t``, and ``prim_ids`` come from one trace, and ``image_sources`` from a second one that runs only when
    it is first read. ``to()`` and ``detach()`` run both.

    Attributes:
        valid: Whether each bounce slot holds a reflection, ``bool``.
        t: Hit distance of each bounce segment, ``float32``.
        image_sources: Image source of each bounce, ``float32`` ``(N, max_bounces, 3)``.
        prim_ids: Scene-global reflecting triangle, or ``-1``, ``int32``.
    """

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
        """Create a chain from its four fields, or from ``loader(full)`` that returns them on first access.

        ``loader`` returns ``(valid, t, image_sources, prim_ids)``; with ``full=False`` it may return ``None`` for
        ``image_sources``.
        """
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
        """Per-slot reflection mask, ``bool`` ``(N, max_bounces)``."""
        self._ensure_reduced()
        return self._valid

    @property
    def t(self) -> torch.Tensor:
        """Hit distance of each bounce segment, ``float32`` ``(N, max_bounces)``."""
        self._ensure_reduced()
        return self._t

    @property
    def image_sources(self) -> torch.Tensor:
        """Image source of each bounce, ``float32`` ``(N, max_bounces, 3)``.

        Reading it the first time runs the image-source trace.
        """
        self._ensure_full()
        return self._image_sources

    @property
    def prim_ids(self) -> torch.Tensor:
        """Scene-global reflecting triangle per slot, or ``-1``, ``int32`` ``(N, max_bounces)``."""
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
    """Reflection paths corrected to end at the receiver, with the field there.

    Result record of ``Scene.trace_refl_epc_field()``. The field is the x component, split into ``_re`` and ``_im``
    tensors and also exposed as the complex ``field``.

    Attributes:
        field_re: Real part of the field, ``float32`` ``(N,)``.
        field_im: Imaginary part of the field, ``float32`` ``(N,)``.
        path_length: Total path length, ``float32`` ``(N,)``.
        valid: Whether a corrected path exists, ``bool`` ``(N,)``.
        resolved_prim_ids: Scene-global reflecting triangle of the path, ``int32`` ``(N,)``.
    """

    field_re: torch.Tensor
    field_im: torch.Tensor
    path_length: torch.Tensor
    valid: torch.Tensor
    resolved_prim_ids: torch.Tensor

    field = _complex_field("field")


@dataclass(frozen=True)
class ReflEpcOptions:
    """Options for ``Scene.trace_refl_epc()``: the expected reflector sequence and the surface-group tables used for
    visibility.

    With source positions, the trace reflects each source across the given planes, so ``expected_prim_ids``,
    ``direct_plane_points``, ``direct_plane_normals``, and the three surface-group tables are required. With a ``Ray``,
    the direct planes must stay ``None``, ``expected_prim_ids`` optionally constrains the hit of each slot (``-1``
    leaves a slot free), and the surface-group tables are optional but given together.

    Attributes:
        expected_prim_ids: Expected reflector per slot, ``int32`` ``(N, max_bounces)``.
        direct_plane_points: Point on the reflecting plane of each slot, ``float32`` ``(N, max_bounces, 3)``.
        direct_plane_normals: Normal of the reflecting plane of each slot, ``float32`` ``(N, max_bounces, 3)``.
        surface_group_id: Surface-group id per scene triangle, ``int32``.
        surface_group_size: Member count of each surface group, ``int32``.
        surface_group_members: Flattened member table of the surface groups, ``int32``.
        visibility_ignore_mode: How visibility checks ignore the reflecting surfaces: ``"primitive"`` or
            ``"surface_group"``.
        plane_tolerance: Relative tolerance for a reflection point to lie on its plane.
    """

    expected_prim_ids: torch.Tensor | None = None
    direct_plane_points: torch.Tensor | None = None
    direct_plane_normals: torch.Tensor | None = None
    surface_group_id: torch.Tensor | None = None
    surface_group_size: torch.Tensor | None = None
    surface_group_members: torch.Tensor | None = None
    visibility_ignore_mode: str = "primitive"
    plane_tolerance: float = 1.0e-5


@dataclass(frozen=True)
class ReflEpcFieldOptions:
    """Options for ``Scene.trace_refl_epc_field()``, with the Dr.Jit field names and defaults.

    A tensor left at ``None`` selects the fixed default: ``eta_r = 4``, ``mu_r = 1``, ``sigma = 0``, ``gain = 1``, and
    an x-polarized source. Every tensor is forward-only: one that requires grad or carries a tangent is rejected.

    Attributes:
        slot_eta_r: Relative permittivity of each slot, ``float32`` ``(N, max_bounces)``.
        slot_mu_r: Relative permeability of each slot, ``float32`` ``(N, max_bounces)``.
        slot_sigma: Conductivity of each slot, ``float32`` ``(N, max_bounces)``.
        slot_gain: Extra gain factor of each slot, ``float32`` ``(N, max_bounces)``.
        tx_polarization: Transmitter polarization vector, ``float32`` ``(1, 3)`` or ``(N, 3)``.
        omega: Angular frequency in rad/s; it enters only the conductivity term.
        wavelength: Wavelength in world units; it sets the phase and the free-space amplitude.
    """

    slot_eta_r: torch.Tensor | None = None
    slot_mu_r: torch.Tensor | None = None
    slot_sigma: torch.Tensor | None = None
    slot_gain: torch.Tensor | None = None
    tx_polarization: torch.Tensor | None = None
    omega: float = 2.0 * math.pi * 299792458.0
    wavelength: float = 1.0


@dataclass(frozen=True, repr=False)
class ReflEpc(_ValidRecord):
    """Reflection paths corrected to end at the receiver.

    Result record of ``Scene.trace_refl_epc()``. Per-slot fields have shape ``(N, max_bounces)``.

    Attributes:
        ray_count: Number of paths ``N``.
        max_bounces: Slots per path.
        valid: Whether a corrected path exists, ``bool`` ``(N,)``.
        path_length: Total path length, ``float32`` ``(N,)``.
        resolved_prim_ids: Scene-global reflecting triangle per slot, ``int32``.
        surface_group_ids: Surface group per slot, ``int32``.
        hit_points: Reflection point per slot, ``float32`` ``(N, max_bounces, 3)``.
        normals: Surface normal per slot, ``float32`` ``(N, max_bounces, 3)``.
    """

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
    """Axis-aligned accumulation plane for ``Scene.accum_refl()``.

    Input record. The plane is normal to ``axis`` at offset ``position``. Its in-plane coordinates are (y, z) for axis
    0, (x, z) for axis 1, and (x, y) for axis 2. Result grids have shape ``(resolution1, resolution0)``.

    Attributes:
        axis: Plane normal axis: 0 = x, 1 = y, 2 = z.
        position: Plane offset along ``axis``.
        coord0_min: Lower bound of the first in-plane coordinate.
        coord0_max: Upper bound of the first in-plane coordinate; must exceed ``coord0_min``.
        coord1_min: Lower bound of the second in-plane coordinate.
        coord1_max: Upper bound of the second in-plane coordinate; must exceed ``coord1_min``.
        resolution0: Cell count along the first in-plane coordinate; must be positive.
        resolution1: Cell count along the second in-plane coordinate; must be positive.
    """

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
    """Options for ``Scene.accum_refl()``.

    An ``accumulation_strategy`` that is not an ``AccumStrategy`` value raises ``ValueError``.

    Attributes:
        wavelength: Wavelength in world units; must be positive.
        solid_angle_per_ray: Solid angle each ray represents; must be non-negative.
        collect_wedges: Record diffraction-wedge events in ``AccumResult.wedge_events``.
        collect_wedge_prefixes: Also record the reflection prefix of each wedge event.
        wedge_capacity: Maximum number of wedge events to store; must be non-negative.
        wedge_sample_stride: Sampling stride for prefix wedge events; must be at least 1.
        accumulation_strategy: Grid write strategy, an ``AccumStrategy``.
        compact_min_samples: Minimum sample count for ``AUTO`` to stage; 0 keeps the built-in threshold.
        staged_min_samples_per_cell: Minimum samples per grid cell for ``AUTO`` to stage; 0 keeps the built-in
            threshold.
        procedural_sample_count: Validated to be non-negative; the current kernels do not read it.
        include_los: Also accumulate the direct (zero-bounce) hit of the plane.
    """

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
    """Electromagnetic surface parameters for ``Scene.accum_refl()``.

    Input record with one entry per scene-global triangle; every field is a CUDA tensor of shape ``(F,)``.
    ``Scene.accum_refl()`` is forward-only, so no field may require grad.

    Attributes:
        eta_r: Relative permittivity, ``float32``.
        sigma: Conductivity, ``float32``.
        mu_r: Relative permeability, ``float32``.
        gain: Extra gain factor, ``float32``.
        valid: Whether the entry holds a real material, ``bool``.
    """

    eta_r: torch.Tensor
    sigma: torch.Tensor
    mu_r: torch.Tensor
    gain: torch.Tensor
    valid: torch.Tensor

    @staticmethod
    def default(count: int, *, device: torch.device, dtype: torch.dtype = torch.float32) -> "ReflMaterial":
        """Return ``count`` entries of a lossless unit material (``eta_r = mu_r = gain = 1``)."""
        return ReflMaterial(
            eta_r=torch.ones((count,), device=device, dtype=dtype),
            sigma=torch.zeros((count,), device=device, dtype=dtype),
            mu_r=torch.ones((count,), device=device, dtype=dtype),
            gain=torch.ones((count,), device=device, dtype=dtype),
            valid=torch.ones((count,), device=device, dtype=torch.bool),
        )


@dataclass(frozen=True, repr=False)
class WedgeEvents(_Record):
    """Diffraction-wedge events recorded by ``Scene.accum_refl()``.

    Result record. The first ``count[0]`` entries of the per-event fields are valid, up to ``capacity``.

    Attributes:
        capacity: Event slots allocated, from ``AccumOptions.wedge_capacity``.
        count: Events recorded, ``int32`` ``(1,)``.
        ray_index: Ray that produced each event, ``int32``.
        hit_points: Hit point, ``float32`` ``(capacity, 3)``.
        normals: Surface normal at the hit, ``float32`` ``(capacity, 3)``.
        prim_id: Scene-global triangle hit, ``int32``.
        directions: Incoming direction, ``float32`` ``(capacity, 3)``.
        source_points: Source point of the event's path segment, ``float32`` ``(capacity, 3)``.
        src_power: Source power of the event, ``float32``.
        initial_directions: Launch direction of the ray, ``float32`` ``(capacity, 3)``.
        bounce_depth: Reflections before the event, ``int32``.
    """

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
    """Reflected power and complex field accumulated per grid cell.

    Result record of ``Scene.accum_refl()``. Grid fields have shape ``(resolution1, resolution0)``; each field component
    is split into ``_re`` and ``_im`` tensors and also exposed as a complex property (``reflection_field_x`` and so on).

    Attributes:
        ray_count: Number of rays traced.
        max_bounces: Maximum reflections per path.
        grid_cell_count: ``resolution0 * resolution1``.
        reflection_power: Accumulated power per cell, ``float32``.
        reflection_field_x_re, reflection_field_x_im, reflection_field_y_re, reflection_field_y_im,
            reflection_field_z_re, reflection_field_z_im: Field components per cell, ``float32``.
        reflection_count: Number of accumulated contributions, ``int32`` ``(1,)``.
        wedge_events: ``WedgeEvents``; empty unless ``AccumOptions.collect_wedges`` is set.
    """

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
    """Axis-aligned accumulation plane for the diffraction accumulators.

    Input record. The plane is normal to ``axis`` at offset ``position``. Its in-plane coordinates are (y, z) for axis
    0, (x, z) for axis 1, and (x, y) for axis 2. Result grids have shape ``(resolution1, resolution0)``.

    Attributes:
        axis: Plane normal axis: 0 = x, 1 = y, 2 = z.
        position: Plane offset along ``axis``.
        coord0_min: Lower bound of the first in-plane coordinate.
        coord0_max: Upper bound of the first in-plane coordinate; must exceed ``coord0_min``.
        coord1_min: Lower bound of the second in-plane coordinate.
        coord1_max: Upper bound of the second in-plane coordinate; must exceed ``coord1_min``.
        resolution0: Cell count along the first in-plane coordinate; must be positive.
        resolution1: Cell count along the second in-plane coordinate; must be positive.
        cell_area: Area of one grid cell; ``None`` derives it from the bounds and resolution.
    """

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
        """Return ``cell_area``, or the plane area divided by the cell count when it is ``None``."""
        if self.cell_area is not None:
            return float(self.cell_area)
        span0 = float(self.coord0_max) - float(self.coord0_min)
        span1 = float(self.coord1_max) - float(self.coord1_min)
        return abs(span0 * span1) / float(int(self.resolution0) * int(self.resolution1))


@dataclass(frozen=True)
class DfrMaterial:
    """Electromagnetic surface parameters for the diffraction kernels.

    Input record with one entry per scene-global triangle; every field is a CUDA tensor of shape ``(F,)``. In the
    differentiable accumulators only ``gain`` carries a derivative.

    Attributes:
        eta_r: Relative permittivity, ``float32``.
        sigma: Conductivity, ``float32``.
        mu_r: Relative permeability, ``float32``.
        gain: Extra gain factor, ``float32``.
        valid: Whether the entry holds a real material, ``bool``.
    """

    eta_r: torch.Tensor
    sigma: torch.Tensor
    mu_r: torch.Tensor
    gain: torch.Tensor
    valid: torch.Tensor

    @staticmethod
    def default(count: int, *, device: torch.device, dtype: torch.dtype = torch.float32) -> "DfrMaterial":
        """Return ``count`` entries of a lossless unit material (``eta_r = mu_r = gain = 1``)."""
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
    """Options for ``Scene.accum_dfr_direct()`` and ``Scene.accum_dfr()``.

    Field names and defaults follow the Dr.Jit ``DfrOptions``. Torch runs the strategies whose sample count is non-zero,
    always uses the hash sample sequence and the matched isotropic receiver, and always fills its counters.

    Attributes:
        wavelength: Wavelength in world units; must be positive.
        seed: Seed of the sample sequence.
        max_order: Diffraction order: 2 or 3 for ``accum_dfr()``; ``accum_dfr_direct()`` ignores it.
        direct_samples: Samples for the direct strategy.
        keller_samples: Samples for the Keller-cone strategy.
        suffix_samples: Samples for the reflection-after-diffraction strategy.
        lane_offset: First global Monte-Carlo lane of this launch; a non-zero offset needs the OptiX trace backend.
        lane_count: Number of lanes to run; ``-1`` runs every remaining lane.
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
    """Options for ``Scene.accum_dfr_coherent_direct()``.

    Field names and defaults follow the Dr.Jit ``DfrCoherentOptions``. The lanes are one per (state, grid cell) pair.

    Attributes:
        wavelength: Wavelength in world units; must be positive.
        select_diffraction_point: Diffract at the first-order stationary point for each cell instead of the edge
            midpoint.
        prefilter_visibility: Skip lanes whose source-to-edge or edge-to-cell segment is blocked.
        lane_offset: First global lane of this launch.
        lane_count: Number of lanes to run; ``-1`` runs every remaining lane.
    """

    wavelength: float = 1.0
    select_diffraction_point: bool = True
    prefilter_visibility: bool = True
    lane_offset: int = 0
    lane_count: int = -1


@dataclass(frozen=True)
class DfrPathOptions:
    """Options for ``Scene.trace_dfr_paths()``.

    ``wavelength`` follows the Dr.Jit ``DfrPathOptions``.

    Attributes:
        wavelength: Wavelength in world units; must be positive.
        max_paths: States exported per (transmitter, receiver) pair. ``None`` or ``0`` exports every state; a negative
            value raises ``ValueError``.
        layout: ``DfrPathLayout``. ``None`` picks ``Compact`` on one device and ``SourceLane`` on a multi-device scene.
    """

    wavelength: float = 1.0
    max_paths: int | None = None
    layout: DfrPathLayout | None = None


def _resolve_dfr_max_paths(max_paths: int | None, state_count: int) -> int:
    """States exported per (transmitter, receiver) pair: ``None`` or ``0`` means every state."""
    if max_paths is None:
        return int(state_count)
    value = int(max_paths)
    if value < 0:
        raise ValueError(f"DfrPathOptions.max_paths must be None or non-negative, got {max_paths}.")
    return int(state_count) if value == 0 else value


@dataclass(frozen=True)
class DfrStates:
    """Sampled diffraction states: an edge segment plus the source that illuminates it.

    Input record for the diffraction accumulators and ``Scene.trace_dfr_paths()``. Per-state fields are CUDA tensors
    with ``S`` rows.

    Attributes:
        edge_index: Scene-global edge id, ``int32`` ``(S,)``.
        edge_pos: Edge origin, ``float32`` ``(S, 3)``.
        edge_dir: Edge direction, ``float32`` ``(S, 3)``.
        edge_t_min: Start of the edge segment along ``edge_dir``, ``float32`` ``(S,)``.
        edge_t_max: End of the edge segment along ``edge_dir``, ``float32`` ``(S,)``.
        n0: Normal of the first adjacent face, ``float32`` ``(S, 3)``.
        n1: Normal of the second adjacent face, ``float32`` ``(S, 3)``.
        prim0: Scene-global first adjacent face, or ``-1``, ``int32`` ``(S,)``.
        prim1: Scene-global second adjacent face, or ``-1``, ``int32`` ``(S,)``.
        exterior_angle: Exterior wedge angle in radians, ``float32`` ``(S,)``.
        src: Source position, ``float32`` ``(S, 3)``.
        src_power: Source power, ``float32`` ``(S,)``.
        wi: Optional per-state direction vector read by the accumulators, ``float32`` ``(S, 3)``; zero when omitted.
        d0: Optional second per-state direction vector, ``float32`` ``(S, 3)``; zero when omitted.
        count: Optional logical state count; ``None`` uses the row count.
    """

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
        """Number of states: ``count`` when set, else the row count of ``edge_index``."""
        return int(self.edge_index.shape[0] if self.count is None else self.count)

    def with_default_vectors(self) -> "DfrStates":
        """Return a copy whose missing ``wi`` and ``d0`` are zero tensors."""
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
    """Diffraction power and complex field accumulated per grid cell, plus strategy and rejection counters.

    Result record of ``Scene.accum_dfr_direct()`` and ``Scene.accum_dfr()``. Grid fields have shape ``(resolution1,
    resolution0)``; counters are ``int32`` ``(1,)``. Each field component is split into ``_re`` and ``_im`` tensors and
    also exposed as a complex property (``field_x`` and so on).

    Attributes:
        grid_cell_count: ``resolution0 * resolution1``.
        power: Accumulated power per cell, ``float32``.
        field_x_re, field_x_im, field_y_re, field_y_im, field_z_re, field_z_im: Field components per cell, ``float32``.
        direct_count: Direct-strategy samples taken.
        keller_count: Keller-cone samples taken.
        suffix_count: Reflection-after-diffraction samples taken.
        vis_rejects: Samples rejected by visibility.
        edge_vis_rejects: Samples rejected by edge visibility.
        utd_rejects: Samples rejected by the UTD evaluation.
        edge_uses: Direct plus Keller samples that used an edge.
    """

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
    """Coherent diffraction field accumulated per grid cell.

    Result record of ``Scene.accum_dfr_coherent_direct()``. Every field has shape ``(resolution1, resolution0)``. Each
    field component is split into ``_re`` and ``_im`` tensors and also exposed as a complex property (``direct_field_x``
    and so on).

    Attributes:
        grid_cell_count: ``resolution0 * resolution1``.
        direct_field_x_re, direct_field_x_im, direct_field_y_re, direct_field_y_im, direct_field_z_re,
            direct_field_z_im: First-order diffracted field per cell, ``float32``.
        multi_field_x_re, multi_field_x_im, multi_field_y_re, multi_field_y_im, multi_field_z_re, multi_field_z_im:
            Higher-order field per cell, ``float32``.
        direct_count: Contributing first-order lanes per cell, ``int32``.
        multi_count: Contributing higher-order lanes per cell, ``int32``.
        vis_rejects: Lanes rejected by visibility per cell, ``int32``.
        utd_rejects: Lanes rejected by the UTD evaluation per cell, ``int32``.
    """

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
    """Exported first-order diffraction paths.

    Result record of ``Scene.trace_dfr_paths()``. Per-path fields have ``capacity`` rows. With ``Compact`` layout the
    first ``count[0]`` rows are the valid paths; with ``SourceLane`` the row of a (tx, rx, state) lane is ``((tx *
    rx_count + rx) * state_limit) + state`` and ``valid`` marks the filled rows. Each field component is split into
    ``_re`` and ``_im`` tensors and also exposed as a complex property (``field_x`` and so on).

    Attributes:
        capacity: Allocated rows, ``T * R * min(state_count, max_paths)``.
        count: Number of exported paths, ``int32`` ``(1,)``.
        valid: Filled-row mask, ``bool``.
        tx_id: Transmitter index, ``int32``.
        rx_id: Receiver index, ``int32``.
        order: Diffraction order, ``int32``; always 1.
        edge0: Scene-global diffracting edge, ``int32``.
        edge1, edge2: Further edges of higher-order paths, ``int32``; unused at order 1.
        delay: Path length divided by the speed of light, ``float32``.
        field_x_re, field_x_im, field_y_re, field_y_im, field_z_re, field_z_im: Field components at the receiver,
            ``float32``.
        p0: Diffraction point on ``edge0``, ``float32`` ``(capacity, 3)``.
        p1, p2: Diffraction points on ``edge1`` and ``edge2``, ``float32`` ``(capacity, 3)``; unused at order 1.
        layout: The ``DfrPathLayout`` the rows use.
    """

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
    """Closest SDF surface hit per ray.

    Result record of ``SdfGrid.intersect()`` and ``sdf_intersect()``. ``t``, ``position``, and ``normal`` are
    differentiable; ``valid`` and ``steps`` carry no derivative. A missed ray has ``t = inf`` and zero ``position`` and
    ``normal``.

    Attributes:
        t: Hit distance, ``float32`` ``(N,)``.
        valid: Hit mask, ``bool`` ``(N,)``.
        position: Hit point, ``float32`` ``(N, 3)``.
        normal: Surface normal at the hit, ``float32`` ``(N, 3)``.
        steps: Sphere-tracing steps taken, ``int32`` ``(N,)``.
    """

    t: torch.Tensor
    valid: torch.Tensor
    position: torch.Tensor
    normal: torch.Tensor
    steps: torch.Tensor


@dataclass(frozen=True, repr=False)
class SceneGeometry(_Record):
    """All meshes' world-space vertices and faces flattened into one scene-global index space.

    Result record of ``Scene.global_geometry()``. Per-face fields have one row per scene-global triangle.

    Attributes:
        vertices: Scene-global vertex positions, ``float32`` ``(V, 3)``.
        faces: Scene-global vertex indices of each triangle, ``int32`` ``(F, 3)``.
        face_normal: Face normal, ``float32`` ``(F, 3)``.
        shape_id: Mesh or instance id of each triangle, ``int32`` ``(F,)``.
        local_prim_id: Triangle id within its mesh, ``int32`` ``(F,)``.
        global_prim_id: Scene-global triangle id, ``int32`` ``(F,)``.
    """

    vertices: torch.Tensor
    faces: torch.Tensor
    face_normal: torch.Tensor
    shape_id: torch.Tensor
    local_prim_id: torch.Tensor
    global_prim_id: torch.Tensor
