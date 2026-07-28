from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, IntFlag
import math
from typing import TYPE_CHECKING, Iterable, Sequence


class PathInteractionKind(IntEnum):
    NONE = 0
    REFLECTION = 1
    DIFFRACTION = 2


class PathProvenance(IntEnum):
    UNKNOWN = 0
    REFLECTION_TRACE = 1
    REFLECTION_EPC = 2
    DIFFRACTION_DIRECT = 3
    DIFFRACTION_CHAIN = 4
    IMPORTED = 5


class PathDerivativeMode(IntEnum):
    NONE = 0
    TANGENT = 1
    ADJOINT = 2


class PathDerivativeField(IntFlag):
    NONE = 0
    INTERACTION_POSITION = 1 << 0
    INTERACTION_NORMAL = 1 << 1
    TOTAL_LENGTH = 1 << 2
    DELAY = 1 << 3
    AOD = 1 << 4
    AOA = 1 << 5
    COMPLEX_FIELD = 1 << 6
    POWER = 1 << 7


Vec3 = tuple[float, float, float]
Complex3 = tuple[complex, complex, complex]


def _vec3(value: Sequence[float] | None, name: str) -> Vec3 | None:
    if value is None:
        return None
    if len(value) != 3:
        raise ValueError(f"{name} must contain three components.")
    result = tuple(float(component) for component in value)
    if not all(math.isfinite(component) for component in result):
        raise ValueError(f"{name} must be finite.")
    return result  # type: ignore[return-value]


def _complex3(value: Sequence[complex] | None) -> Complex3 | None:
    if value is None:
        return None
    if len(value) != 3:
        raise ValueError("field must contain three complex components.")
    result = tuple(complex(component) for component in value)
    if not all(math.isfinite(v.real) and math.isfinite(v.imag) for v in result):
        raise ValueError("field must be finite.")
    return result  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class PathInteraction:
    kind: PathInteractionKind
    global_primitive_id: int = -1
    global_edge_id: int = -1
    position: Vec3 | None = None
    normal: Vec3 | None = None

    if TYPE_CHECKING:
        # `__post_init__` normalizes any three-component sequence, so the
        # constructor accepts more than the normalized attribute type.
        def __init__(
            self,
            kind: PathInteractionKind,
            global_primitive_id: int = ...,
            global_edge_id: int = ...,
            position: Sequence[float] | None = ...,
            normal: Sequence[float] | None = ...,
        ) -> None: ...

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", PathInteractionKind(self.kind))
        object.__setattr__(self, "global_primitive_id", int(self.global_primitive_id))
        object.__setattr__(self, "global_edge_id", int(self.global_edge_id))
        object.__setattr__(self, "position", _vec3(self.position, "position"))
        object.__setattr__(self, "normal", _vec3(self.normal, "normal"))
        if self.kind is PathInteractionKind.REFLECTION:
            if self.global_primitive_id < 0 or self.global_edge_id != -1:
                raise ValueError("reflection requires one scene-global primitive ID.")
        elif self.kind is PathInteractionKind.DIFFRACTION:
            if self.global_edge_id < 0 or self.global_primitive_id != -1:
                raise ValueError("diffraction requires one scene-global edge ID.")
        elif self.global_primitive_id != -1 or self.global_edge_id != -1:
            raise ValueError("interaction kind NONE cannot carry a winner ID.")


@dataclass(frozen=True, slots=True)
class PathDerivative:
    total_length: float | None = None
    delay: float | None = None
    aod: Vec3 | None = None
    aoa: Vec3 | None = None
    field: Complex3 | None = None
    power: float | None = None

    if TYPE_CHECKING:
        # `__post_init__` normalizes any three-component sequence, so the
        # constructor accepts more than the normalized attribute type.
        def __init__(
            self,
            total_length: float | None = ...,
            delay: float | None = ...,
            aod: Sequence[float] | None = ...,
            aoa: Sequence[float] | None = ...,
            field: Sequence[complex] | None = ...,
            power: float | None = ...,
        ) -> None: ...

    def __post_init__(self) -> None:
        object.__setattr__(self, "aod", _vec3(self.aod, "derivative aod"))
        object.__setattr__(self, "aoa", _vec3(self.aoa, "derivative aoa"))
        object.__setattr__(self, "field", _complex3(self.field))
        for name in ("total_length", "delay", "power"):
            value = getattr(self, name)
            if value is not None:
                value = float(value)
                if not math.isfinite(value):
                    raise ValueError(f"derivative {name} must be finite.")
                object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class PathInteractionDerivative:
    position: Vec3 | None = None
    normal: Vec3 | None = None

    if TYPE_CHECKING:
        # `__post_init__` normalizes any three-component sequence, so the
        # constructor accepts more than the normalized attribute type.
        def __init__(
            self,
            position: Sequence[float] | None = ...,
            normal: Sequence[float] | None = ...,
        ) -> None: ...

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", _vec3(self.position, "derivative position"))
        object.__setattr__(self, "normal", _vec3(self.normal, "derivative normal"))


@dataclass(frozen=True, slots=True)
class PathRecord:
    valid: bool
    order: int
    source_index: int
    receiver_index: int
    provenance: PathProvenance
    interactions: tuple[PathInteraction, ...] = ()
    total_length: float | None = None
    delay: float | None = None
    aod: Vec3 | None = None
    aoa: Vec3 | None = None
    field: Complex3 | None = None
    power: float | None = None
    fixed_winner: bool = True
    differentiable_fields: PathDerivativeField = PathDerivativeField.NONE
    derivative_mode: PathDerivativeMode = PathDerivativeMode.NONE
    derivative: PathDerivative | None = None
    interaction_derivatives: tuple[PathInteractionDerivative, ...] = ()

    if TYPE_CHECKING:
        # `__post_init__` normalizes any three-component sequence, so the
        # constructor accepts more than the normalized attribute type.
        def __init__(
            self,
            valid: bool,
            order: int,
            source_index: int,
            receiver_index: int,
            provenance: PathProvenance,
            interactions: tuple[PathInteraction, ...] = ...,
            total_length: float | None = ...,
            delay: float | None = ...,
            aod: Sequence[float] | None = ...,
            aoa: Sequence[float] | None = ...,
            field: Sequence[complex] | None = ...,
            power: float | None = ...,
            fixed_winner: bool = ...,
            differentiable_fields: PathDerivativeField = ...,
            derivative_mode: PathDerivativeMode = ...,
            derivative: PathDerivative | None = ...,
            interaction_derivatives: tuple[PathInteractionDerivative, ...] = ...,
        ) -> None: ...

    def __post_init__(self) -> None:
        object.__setattr__(self, "valid", bool(self.valid))
        object.__setattr__(self, "order", int(self.order))
        object.__setattr__(self, "source_index", int(self.source_index))
        object.__setattr__(self, "receiver_index", int(self.receiver_index))
        object.__setattr__(self, "provenance", PathProvenance(self.provenance))
        object.__setattr__(self, "interactions", tuple(self.interactions))
        if not all(isinstance(item, PathInteraction) for item in self.interactions):
            raise TypeError("interactions must contain PathInteraction records.")
        object.__setattr__(self, "aod", _vec3(self.aod, "aod"))
        object.__setattr__(self, "aoa", _vec3(self.aoa, "aoa"))
        object.__setattr__(self, "field", _complex3(self.field))
        object.__setattr__(
            self, "differentiable_fields", PathDerivativeField(self.differentiable_fields)
        )
        object.__setattr__(self, "derivative_mode", PathDerivativeMode(self.derivative_mode))
        object.__setattr__(self, "interaction_derivatives", tuple(self.interaction_derivatives))
        if self.derivative is not None and not isinstance(self.derivative, PathDerivative):
            raise TypeError("derivative must be a PathDerivative record.")
        if not all(
            isinstance(item, PathInteractionDerivative)
            for item in self.interaction_derivatives
        ):
            raise TypeError(
                "interaction_derivatives must contain PathInteractionDerivative records."
            )
        if self.order < 0 or self.order != len(self.interactions):
            raise ValueError("order must equal the number of interactions.")
        if self.source_index < -1 or self.receiver_index < -1:
            raise ValueError("source/receiver indices use -1 as the only invalid value.")
        if not self.fixed_winner:
            raise ValueError("path exchange derivatives require fixed_winner=True.")
        if self.valid and self.provenance is PathProvenance.UNKNOWN:
            raise ValueError("valid paths require explicit provenance.")
        for name in ("total_length", "delay", "power"):
            value = getattr(self, name)
            if value is not None:
                value = float(value)
                if not math.isfinite(value) or value < 0.0:
                    raise ValueError(f"{name} must be finite and non-negative.")
                object.__setattr__(self, name, value)
        if self.differentiable_fields & ~self.available_fields:
            raise ValueError("differentiable_fields must be a subset of available_fields.")
        self._validate_derivative_payload()

    def _validate_derivative_payload(self) -> None:
        if self.derivative_mode is PathDerivativeMode.NONE:
            if self.derivative is not None or self.interaction_derivatives:
                raise ValueError("derivative payload requires tangent or adjoint mode.")
            return
        if self.differentiable_fields == PathDerivativeField.NONE:
            raise ValueError("derivative mode requires differentiable_fields.")
        path_fields = {
            PathDerivativeField.TOTAL_LENGTH: "total_length",
            PathDerivativeField.DELAY: "delay",
            PathDerivativeField.AOD: "aod",
            PathDerivativeField.AOA: "aoa",
            PathDerivativeField.COMPLEX_FIELD: "field",
            PathDerivativeField.POWER: "power",
        }
        for bit, name in path_fields.items():
            if self.differentiable_fields & bit:
                if self.derivative is None or getattr(self.derivative, name) is None:
                    raise ValueError(f"missing derivative value for {name}.")
        interaction_bits = self.differentiable_fields & (
            PathDerivativeField.INTERACTION_POSITION
            | PathDerivativeField.INTERACTION_NORMAL
        )
        if interaction_bits:
            if len(self.interaction_derivatives) != len(self.interactions):
                raise ValueError("interaction derivative indexing must match the primal path.")
            for derivative in self.interaction_derivatives:
                if (
                    interaction_bits & PathDerivativeField.INTERACTION_POSITION
                    and derivative.position is None
                ):
                    raise ValueError("missing interaction position derivative.")
                if (
                    interaction_bits & PathDerivativeField.INTERACTION_NORMAL
                    and derivative.normal is None
                ):
                    raise ValueError("missing interaction normal derivative.")

    @property
    def available_fields(self) -> PathDerivativeField:
        fields = PathDerivativeField.NONE
        if self.interactions and all(item.position is not None for item in self.interactions):
            fields |= PathDerivativeField.INTERACTION_POSITION
        if self.interactions and all(item.normal is not None for item in self.interactions):
            fields |= PathDerivativeField.INTERACTION_NORMAL
        if self.total_length is not None:
            fields |= PathDerivativeField.TOTAL_LENGTH
        if self.delay is not None:
            fields |= PathDerivativeField.DELAY
        if self.aod is not None:
            fields |= PathDerivativeField.AOD
        if self.aoa is not None:
            fields |= PathDerivativeField.AOA
        if self.field is not None:
            fields |= PathDerivativeField.COMPLEX_FIELD
        if self.power is not None:
            fields |= PathDerivativeField.POWER
        return fields

    def as_exchange_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "fixed_winner": self.fixed_winner,
            "order": self.order,
            "source_index": self.source_index,
            "receiver_index": self.receiver_index,
            "provenance": int(self.provenance),
            "available_fields": int(self.available_fields),
            "differentiable_fields": int(self.differentiable_fields),
            "derivative_mode": int(self.derivative_mode),
            "interactions": [
                {
                    "kind": int(item.kind),
                    "global_primitive_id": item.global_primitive_id,
                    "global_edge_id": item.global_edge_id,
                    "position": item.position,
                    "normal": item.normal,
                }
                for item in self.interactions
            ],
            "total_length": self.total_length,
            "delay": self.delay,
            "aod": self.aod,
            "aoa": self.aoa,
            "field": self.field,
            "power": self.power,
            "derivative": None if self.derivative is None else {
                "total_length": self.derivative.total_length,
                "delay": self.derivative.delay,
                "aod": self.derivative.aod,
                "aoa": self.derivative.aoa,
                "field": self.derivative.field,
                "power": self.derivative.power,
            },
            "interaction_derivatives": [
                {"position": item.position, "normal": item.normal}
                for item in self.interaction_derivatives
            ],
        }


def reflection_path_record(
    global_primitive_ids: Iterable[int],
    *,
    segment_lengths: Iterable[float] | None = None,
    positions: Iterable[Sequence[float]] | None = None,
    normals: Iterable[Sequence[float]] | None = None,
    source_index: int = -1,
    receiver_index: int = -1,
    provenance: PathProvenance = PathProvenance.REFLECTION_TRACE,
    total_length: float | None = None,
    delay: float | None = None,
    aod: Sequence[float] | None = None,
    aoa: Sequence[float] | None = None,
    field: Sequence[complex] | None = None,
    power: float | None = None,
    differentiable_fields: PathDerivativeField = PathDerivativeField.NONE,
    derivative_mode: PathDerivativeMode = PathDerivativeMode.NONE,
    derivative: PathDerivative | None = None,
    interaction_derivatives: Iterable[PathInteractionDerivative] = (),
) -> PathRecord:
    primitive_ids = tuple(int(value) for value in global_primitive_ids)
    position_values = tuple(positions) if positions is not None else (None,) * len(primitive_ids)
    normal_values = tuple(normals) if normals is not None else (None,) * len(primitive_ids)
    if len(position_values) != len(primitive_ids) or len(normal_values) != len(primitive_ids):
        raise ValueError("reflection geometry must match the primitive sequence.")
    lengths = tuple(float(value) for value in segment_lengths) if segment_lengths is not None else ()
    if lengths and len(lengths) != len(primitive_ids):
        raise ValueError("segment_lengths must match the reflection order.")
    if total_length is None and lengths:
        total_length = sum(lengths)
    interactions = tuple(
        PathInteraction(PathInteractionKind.REFLECTION, primitive, -1, position, normal)
        for primitive, position, normal in zip(primitive_ids, position_values, normal_values)
    )
    return PathRecord(
        valid=True,
        order=len(interactions),
        source_index=source_index,
        receiver_index=receiver_index,
        provenance=provenance,
        interactions=interactions,
        total_length=total_length,
        delay=delay,
        aod=aod,
        aoa=aoa,
        field=field,
        power=power,
        differentiable_fields=differentiable_fields,
        derivative_mode=derivative_mode,
        derivative=derivative,
        interaction_derivatives=tuple(interaction_derivatives),
    )


def diffraction_path_record(
    global_edge_ids: Iterable[int],
    *,
    positions: Iterable[Sequence[float]] | None = None,
    normals: Iterable[Sequence[float]] | None = None,
    source_index: int = -1,
    receiver_index: int = -1,
    provenance: PathProvenance = PathProvenance.DIFFRACTION_DIRECT,
    total_length: float | None = None,
    delay: float | None = None,
    aod: Sequence[float] | None = None,
    aoa: Sequence[float] | None = None,
    field: Sequence[complex] | None = None,
    power: float | None = None,
    differentiable_fields: PathDerivativeField = PathDerivativeField.NONE,
    derivative_mode: PathDerivativeMode = PathDerivativeMode.NONE,
    derivative: PathDerivative | None = None,
    interaction_derivatives: Iterable[PathInteractionDerivative] = (),
) -> PathRecord:
    edge_ids = tuple(int(value) for value in global_edge_ids)
    position_values = tuple(positions) if positions is not None else (None,) * len(edge_ids)
    normal_values = tuple(normals) if normals is not None else (None,) * len(edge_ids)
    if len(position_values) != len(edge_ids) or len(normal_values) != len(edge_ids):
        raise ValueError("diffraction geometry must match the edge sequence.")
    complex_field = _complex3(field)
    if power is None and complex_field is not None:
        power = sum(abs(value) ** 2 for value in complex_field)
    interactions = tuple(
        PathInteraction(PathInteractionKind.DIFFRACTION, -1, edge, position, normal)
        for edge, position, normal in zip(edge_ids, position_values, normal_values)
    )
    return PathRecord(
        valid=True,
        order=len(interactions),
        source_index=source_index,
        receiver_index=receiver_index,
        provenance=provenance,
        interactions=interactions,
        total_length=total_length,
        delay=delay,
        aod=aod,
        aoa=aoa,
        field=complex_field,
        power=power,
        differentiable_fields=differentiable_fields,
        derivative_mode=derivative_mode,
        derivative=derivative,
        interaction_derivatives=tuple(interaction_derivatives),
    )
