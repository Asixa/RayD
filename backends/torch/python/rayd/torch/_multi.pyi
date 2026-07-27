from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Callable

import torch

@dataclass(frozen=True)
class MultiDeviceOptions:
    weights: Sequence[float] | None = ...
    warm_up: bool = ...
    chunk_rays: int | None = ...
    offload: Callable[[int, Any], None] | None = ...
    tape_memory_budget_bytes: int | None = ...

@dataclass
class ChunkPlan:
    operation: str
    total_rows: int
    chunk_rays: int
    source: str
    row_bytes: int
    budget_bytes: int | None = ...
    chunk_count: int = ...
    measured_row_bytes: float | None = ...

def calibrate_chunk_size(
    operation: str,
    total_rows: int,
    *,
    row_bytes: int,
    chunk_rays: int | None = ...,
    budget_bytes: int | None = ...,
) -> ChunkPlan: ...
def plan(
    devices: Sequence[int | str | torch.device] | None,
    options: MultiDeviceOptions | None,
    *,
    trace_backend: str,
    edge_bvh_backend: str,
) -> Any: ...
