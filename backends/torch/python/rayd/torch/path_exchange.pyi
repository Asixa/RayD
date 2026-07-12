from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import IntEnum, IntFlag
from typing import TypeAlias

Vec3: TypeAlias = tuple[float, float, float]
Complex3: TypeAlias = tuple[complex, complex, complex]

class PathInteractionKind(IntEnum):
    NONE: PathInteractionKind
    REFLECTION: PathInteractionKind
    DIFFRACTION: PathInteractionKind

class PathProvenance(IntEnum):
    UNKNOWN: PathProvenance
    REFLECTION_TRACE: PathProvenance
    REFLECTION_EPC: PathProvenance
    DIFFRACTION_DIRECT: PathProvenance
    DIFFRACTION_CHAIN: PathProvenance
    IMPORTED: PathProvenance

class PathDerivativeMode(IntEnum):
    NONE: PathDerivativeMode
    TANGENT: PathDerivativeMode
    ADJOINT: PathDerivativeMode

class PathDerivativeField(IntFlag):
    NONE: PathDerivativeField
    INTERACTION_POSITION: PathDerivativeField
    INTERACTION_NORMAL: PathDerivativeField
    TOTAL_LENGTH: PathDerivativeField
    DELAY: PathDerivativeField
    AOD: PathDerivativeField
    AOA: PathDerivativeField
    COMPLEX_FIELD: PathDerivativeField
    POWER: PathDerivativeField

class PathInteraction:
    kind: PathInteractionKind
    global_primitive_id: int
    global_edge_id: int
    position: Vec3 | None
    normal: Vec3 | None
    def __init__(self, kind: PathInteractionKind, global_primitive_id: int = ..., global_edge_id: int = ..., position: Sequence[float] | None = ..., normal: Sequence[float] | None = ...) -> None: ...

class PathDerivative:
    total_length: float | None
    delay: float | None
    aod: Vec3 | None
    aoa: Vec3 | None
    field: Complex3 | None
    power: float | None
    def __init__(self, total_length: float | None = ..., delay: float | None = ..., aod: Sequence[float] | None = ..., aoa: Sequence[float] | None = ..., field: Sequence[complex] | None = ..., power: float | None = ...) -> None: ...

class PathInteractionDerivative:
    position: Vec3 | None
    normal: Vec3 | None
    def __init__(self, position: Sequence[float] | None = ..., normal: Sequence[float] | None = ...) -> None: ...

@dataclass(frozen=True)
class PathRecord:
    valid: bool
    order: int
    source_index: int
    receiver_index: int
    provenance: PathProvenance
    interactions: tuple[PathInteraction, ...]
    total_length: float | None
    delay: float | None
    aod: Vec3 | None
    aoa: Vec3 | None
    field: Complex3 | None
    power: float | None
    fixed_winner: bool
    differentiable_fields: PathDerivativeField
    derivative_mode: PathDerivativeMode
    derivative: PathDerivative | None
    interaction_derivatives: tuple[PathInteractionDerivative, ...]
    @property
    def available_fields(self) -> PathDerivativeField: ...
    def as_exchange_dict(self) -> dict[str, object]: ...

def reflection_path_record(global_primitive_ids: Iterable[int], *, segment_lengths: Iterable[float] | None = ..., positions: Iterable[Sequence[float]] | None = ..., normals: Iterable[Sequence[float]] | None = ..., source_index: int = ..., receiver_index: int = ..., provenance: PathProvenance = ..., total_length: float | None = ..., delay: float | None = ..., aod: Sequence[float] | None = ..., aoa: Sequence[float] | None = ..., field: Sequence[complex] | None = ..., power: float | None = ..., differentiable_fields: PathDerivativeField = ..., derivative_mode: PathDerivativeMode = ..., derivative: PathDerivative | None = ..., interaction_derivatives: Iterable[PathInteractionDerivative] = ...) -> PathRecord: ...
def diffraction_path_record(global_edge_ids: Iterable[int], *, positions: Iterable[Sequence[float]] | None = ..., normals: Iterable[Sequence[float]] | None = ..., source_index: int = ..., receiver_index: int = ..., provenance: PathProvenance = ..., total_length: float | None = ..., delay: float | None = ..., aod: Sequence[float] | None = ..., aoa: Sequence[float] | None = ..., field: Sequence[complex] | None = ..., power: float | None = ..., differentiable_fields: PathDerivativeField = ..., derivative_mode: PathDerivativeMode = ..., derivative: PathDerivative | None = ..., interaction_derivatives: Iterable[PathInteractionDerivative] = ...) -> PathRecord: ...
