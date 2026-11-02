# Copyright Xingyu Chen.
# Builds backend-neutral adversarial ray batches and scenes for the opt-in stress tests.

from __future__ import annotations

import math

import numpy as np

BOX_VERTICES = np.array(
    [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]],
    dtype=np.float32,
)
BOX_FACES = np.array(
    [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4]]
    + [[1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]],
    dtype=np.int32,
)
# A collinear sliver, a zero-area point triangle, a tilted ordinary triangle, and a repeated-index triangle.
DEGENERATE_VERTICES = np.array(
    [[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1], [-1, -1, 0.5], [1, -1, 0.5], [0, 1, 0.5]],
    dtype=np.float32,
)
DEGENERATE_FACES = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 0, 1]], dtype=np.int32)
SCENES = {"box": (BOX_VERTICES, BOX_FACES), "degenerate": (DEGENERATE_VERTICES, DEGENERATE_FACES)}


def adversarial_rays(random_count: int = 4096, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (origins, directions, well_formed) with grazing, coplanar, vertex/edge-aimed, and invalid rays.

    ``well_formed`` is false for rows with a NaN/Inf/zero/denormal/overflowing direction or a non-finite or
    astronomically distant origin; operations must survive those rows but need not return finite values for them.
    """
    rng = np.random.default_rng(seed)
    origins = [(rng.random((random_count, 3)) - 0.5).astype(np.float32)]
    directions = [rng.standard_normal((random_count, 3)).astype(np.float32)]
    # Rays lying exactly in, or parallel to, the box faces and the degenerate triangles' planes.
    for z in (0.0, 1.0, -1.0, 0.5):
        for direction in ([1, 0, 0], [0, 1, 0], [1, 1, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1]):
            origins.append(np.array([[0, 0, z], [0.3, 0.2, z], [1, 1, z], [-1, 0, z]], np.float32))
            directions.append(np.array([direction] * 4, np.float32))
    # Rays aimed exactly at vertices, edge midpoints, and face centres.
    targets = np.array(
        [[1, 1, 1], [1, -1, 1], [1, 0, 1], [0, 0, 1], [1, 1, 0], [0, 0, 0], [0.5, 0.5, 0], [0, 1, 0]], np.float32
    )
    for start in ([0.1, 0.2, -0.3], [0.0, 0.0, 5.0]):
        start = np.array([start], np.float32)
        origins.append(np.repeat(start, len(targets), 0))
        directions.append(targets - start)
    # Grazing directions down to exactly parallel.
    for eps in (1e-3, 1e-5, 1e-7, 1e-9, 1e-20, 0.0):
        origins.append(np.array([[-0.9, 0, 0.2], [0, -0.9, -0.2]], np.float32))
        directions.append(np.array([[1, 0, -eps], [0, 1, eps]], np.float32))
    well_formed_count = sum(len(chunk) for chunk in origins)
    bad_directions = np.array(
        [
            [0, 0, 0],
            [math.nan, 0, 1],
            [math.inf, 0, 0],
            [-math.inf, 1, 0],
            [1e-30, 0, 0],
            [1e30, 1e30, 1e30],
            [1e-45, 1e-45, 1e-45],
            [3.4e38, 3.4e38, 3.4e38],
        ],
        np.float32,
    )
    bad_origins = np.array(
        [[math.nan, 0, 0], [math.inf, 0, 0], [1e30, 0, 0], [0, 0, -1e20], [-math.inf] * 3], np.float32
    )
    origins += [np.zeros_like(bad_directions), bad_origins]
    directions += [bad_directions, np.repeat(np.array([[0, 0, 1]], np.float32), len(bad_origins), 0)]
    origin = np.concatenate(origins)
    direction = np.concatenate(directions)
    well_formed = np.zeros(len(origin), dtype=bool)
    well_formed[:well_formed_count] = True
    return origin, direction, well_formed
