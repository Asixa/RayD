"""Backend-neutral declarative golden scenes for RayD geometry and edge queries.

Each scene is pure Python data: mesh vertices/faces as lists of ``[x, y, z]``
points and index triples, plus an ordered list of query dicts that
``tests.golden.runner`` interprets against a backend. The definitions freeze the
current RayD public query contract (see ``RAY_TRACING_BACKEND_ARCHITECTURE.md``
section 16); they are not tuned to any "correct" value.

Query kinds understood by the runner:
  intersect, intersect_grid, shadow_test, visible, visible_pair,
  nearest_edge_point, nearest_edge_ray, nearest_edges, update_vertices.

Optional per-query keys:
  tmax, active, nan_flags, check_id_mapping, expect_raises, informative.
``informative`` is either ``True`` (whole query recorded-not-compared) or a list
of output field names to move into the informative bucket. Rays/points whose
winner is traversal-order or tie dependent are tagged informative so the strict
cross-backend comparison skips them.
"""

# vertices of a unit right triangle in the z = 0 plane, geometric normal +z
SINGLE_TRI_VERTS = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
SINGLE_TRI_FACES = [[0, 1, 2]]

# unit square as two triangles sharing the v0-v2 diagonal (5 edges, 4 boundary)
QUAD_VERTS = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
QUAD_FACES = [[0, 1, 2], [0, 2, 3]]


SCENES = [
    {
        "name": "single_tri",
        "cases": ["miss", "front_back_face", "finite_tmax"],
        "meshes": [{"vertices": SINGLE_TRI_VERTS, "faces": SINGLE_TRI_FACES}],
        "queries": [
            {
                "name": "front_face_hit",
                "kind": "intersect",
                "origins": [[0.25, 0.25, -1.0]],
                "dirs": [[0.0, 0.0, 1.0]],
            },
            {
                "name": "back_face_hit",
                "kind": "intersect",
                "origins": [[0.25, 0.25, 1.0]],
                "dirs": [[0.0, 0.0, -1.0]],
            },
            {
                "name": "spatial_miss",
                "kind": "intersect",
                "origins": [[2.0, 2.0, -1.0]],
                "dirs": [[0.0, 0.0, 1.0]],
            },
            {
                "name": "tmax_miss",
                "kind": "intersect",
                "origins": [[0.25, 0.25, -1.0]],
                "dirs": [[0.0, 0.0, 1.0]],
                "tmax": [0.5],
            },
        ],
    },
    {
        "name": "shared_edge_quad",
        "cases": ["shared_edge_vertex"],
        "meshes": [{"vertices": QUAD_VERTS, "faces": QUAD_FACES}],
        "queries": [
            {
                "name": "strict_tri0_interior",
                "kind": "intersect",
                "origins": [[0.75, 0.25, -1.0]],
                "dirs": [[0.0, 0.0, 1.0]],
            },
            {
                "name": "strict_tri1_interior",
                "kind": "intersect",
                "origins": [[0.25, 0.75, -1.0]],
                "dirs": [[0.0, 0.0, 1.0]],
            },
            {
                "name": "exact_shared_edge_midpoint",
                "kind": "intersect",
                "origins": [[0.5, 0.5, -1.0]],
                "dirs": [[0.0, 0.0, 1.0]],
                "informative": True,
            },
            {
                "name": "exact_shared_vertex",
                "kind": "intersect",
                "origins": [[0.0, 0.0, -1.0]],
                "dirs": [[0.0, 0.0, 1.0]],
                "informative": True,
            },
        ],
    },
    {
        "name": "degenerate_tri",
        "cases": ["degenerate_triangle", "miss"],
        "meshes": [
            {
                # tri 0 is collinear (zero area) on the x axis; tri 1 is valid
                "vertices": [
                    [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0],
                    [0.0, 3.0, 0.0], [1.0, 3.0, 0.0], [0.0, 4.0, 0.0],
                ],
                "faces": [[0, 1, 2], [3, 4, 5]],
            }
        ],
        "queries": [
            {
                "name": "aim_at_degenerate",
                "kind": "intersect",
                "origins": [[1.0, 0.0, -1.0]],
                "dirs": [[0.0, 0.0, 1.0]],
                "nan_flags": ["t", "p", "n", "geo_n", "uv", "bary"],
            },
            {
                "name": "aim_at_valid",
                "kind": "intersect",
                "origins": [[0.4, 3.3, -1.0]],
                "dirs": [[0.0, 0.0, 1.0]],
                "nan_flags": ["t", "p", "n", "geo_n", "uv", "bary"],
            },
        ],
    },
    {
        "name": "large_coordinates",
        "cases": ["large_coordinates"],
        "meshes": [
            {
                "vertices": [
                    [1000000.0, 1000000.0, 1000000.0],
                    [1000001.0, 1000000.0, 1000000.0],
                    [1000000.0, 1000001.0, 1000000.0],
                ],
                "faces": SINGLE_TRI_FACES,
            }
        ],
        "queries": [
            {
                "name": "translated_hit",
                "kind": "intersect",
                "origins": [[1000000.25, 1000000.25, 999999.0]],
                "dirs": [[0.0, 0.0, 1.0]],
            }
        ],
    },
    {
        "name": "self_intersection",
        "cases": ["self_intersection"],
        "meshes": [
            {
                # tri 0 at z = 0, tri 1 parallel at z = 0.5
                "vertices": [
                    [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.5], [1.0, 0.0, 0.5], [0.0, 1.0, 0.5],
                ],
                "faces": [[0, 1, 2], [3, 4, 5]],
            }
        ],
        "queries": [
            {
                # origin exactly on tri 0; +z ray does not self-hit (tmin swallow)
                "name": "on_surface_along_normal",
                "kind": "intersect",
                "origins": [[0.25, 0.25, 0.0]],
                "dirs": [[0.0, 0.0, 1.0]],
            },
            {
                "name": "on_surface_against_normal",
                "kind": "intersect",
                "origins": [[0.25, 0.25, 0.0]],
                "dirs": [[0.0, 0.0, -1.0]],
            },
        ],
    },
    {
        "name": "multi_mesh_ids",
        "cases": ["multi_mesh", "id_mapping"],
        "record_face_offsets": True,
        "meshes": [
            {"vertices": SINGLE_TRI_VERTS, "faces": SINGLE_TRI_FACES},
            {
                "vertices": [
                    [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [2.0, 1.0, 0.0],
                    [3.0, 0.0, 0.0], [4.0, 0.0, 0.0], [3.0, 1.0, 0.0],
                ],
                "faces": [[0, 1, 2], [3, 4, 5]],
            },
            {
                "vertices": [[5.0, 0.0, 0.0], [6.0, 0.0, 0.0], [5.0, 1.0, 0.0]],
                "faces": SINGLE_TRI_FACES,
            },
        ],
        "queries": [
            {
                "name": "hit_each_mesh",
                "kind": "intersect",
                "origins": [
                    [0.25, 0.25, -1.0],
                    [3.25, 0.25, -1.0],
                    [5.25, 0.25, -1.0],
                ],
                "dirs": [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
                "check_id_mapping": True,
            }
        ],
    },
    {
        "name": "finite_tmax_visibility",
        "cases": ["finite_tmax"],
        "meshes": [{"vertices": SINGLE_TRI_VERTS, "faces": SINGLE_TRI_FACES}],
        "queries": [
            {
                "name": "segment_through_surface",
                "kind": "visible",
                "start": [[0.25, 0.25, -1.0]],
                "end": [[0.25, 0.25, 1.0]],
            },
            {
                "name": "segment_ends_before_surface",
                "kind": "visible",
                "start": [[0.25, 0.25, -1.0]],
                "end": [[0.25, 0.25, -0.5]],
            },
            {
                "name": "pair_blocked_and_clear",
                "kind": "visible_pair",
                "start": [[0.25, 0.25, -1.0]],
                "end_a": [[0.25, 0.25, 1.0]],
                "end_b": [[2.0, 2.0, 1.0]],
            },
            {
                "name": "shadow_tmax_before_surface",
                "kind": "shadow_test",
                "origins": [[0.25, 0.25, -1.0]],
                "dirs": [[0.0, 0.0, 1.0]],
                "tmax": [0.5],
            },
            {
                "name": "shadow_tmax_past_surface",
                "kind": "shadow_test",
                "origins": [[0.25, 0.25, -1.0]],
                "dirs": [[0.0, 0.0, 1.0]],
                "tmax": [2.0],
            },
        ],
    },
    {
        "name": "ignore_primitive",
        "cases": ["ignore_primitive"],
        "meshes": [{"vertices": SINGLE_TRI_VERTS, "faces": SINGLE_TRI_FACES}],
        "queries": [
            {
                "name": "no_ignore",
                "kind": "visible",
                "start": [[0.25, 0.25, -1.0]],
                "end": [[0.25, 0.25, 1.0]],
            },
            {
                "name": "ignore_pair_prim0",
                "kind": "visible",
                "start": [[0.25, 0.25, -1.0]],
                "end": [[0.25, 0.25, 1.0]],
                "ignore": [0],
            },
            {
                "name": "ignore_list_nine_entries",
                "kind": "visible",
                "start": [[0.25, 0.25, -1.0]],
                "end": [[0.25, 0.25, 1.0]],
                "ignore": [-1, -1, -1, -1, -1, -1, -1, -1, 0],
            },
        ],
    },
    {
        "name": "dynamic_refit",
        "cases": ["dynamic_refit"],
        "meshes": [
            {"vertices": SINGLE_TRI_VERTS, "faces": SINGLE_TRI_FACES, "dynamic": True}
        ],
        "queries": [
            {
                "name": "pre_update_hit",
                "kind": "intersect",
                "origins": [[0.25, 0.25, -1.0]],
                "dirs": [[0.0, 0.0, 1.0]],
            },
            {
                "name": "shift_mesh_plus_two_x",
                "kind": "update_vertices",
                "mesh": 0,
                "vertices": [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [2.0, 1.0, 0.0]],
            },
            {
                "name": "post_update_same_ray_miss",
                "kind": "intersect",
                "origins": [[0.25, 0.25, -1.0]],
                "dirs": [[0.0, 0.0, 1.0]],
            },
            {
                "name": "post_update_shifted_ray_hit",
                "kind": "intersect",
                "origins": [[2.25, 0.25, -1.0]],
                "dirs": [[0.0, 0.0, 1.0]],
            },
        ],
    },
    {
        "name": "inactive_lanes",
        "cases": ["inactive_lane"],
        "meshes": [{"vertices": SINGLE_TRI_VERTS, "faces": SINGLE_TRI_FACES}],
        "queries": [
            {
                "name": "intersect_mixed_active",
                "kind": "intersect",
                "origins": [[0.25, 0.25, -1.0], [0.25, 0.25, -1.0]],
                "dirs": [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
                "active": [1, 0],
            },
            {
                "name": "visible_mixed_active",
                "kind": "visible",
                "start": [[0.25, 0.25, -1.0], [0.25, 0.25, -1.0]],
                "end": [[0.25, 0.25, 1.0], [0.25, 0.25, 1.0]],
                "active": [1, 0],
            },
        ],
    },
    {
        "name": "batch_sizes",
        "cases": ["empty_or_large_batch"],
        "meshes": [{"vertices": QUAD_VERTS, "faces": QUAD_FACES}],
        "queries": [
            {
                "name": "batch_of_one",
                "kind": "intersect",
                "origins": [[0.5, 0.5, -1.0]],
                "dirs": [[0.0, 0.0, 1.0]],
                "informative": ["prim_id", "local_prim_id", "global_prim_id", "bary"],
            },
            {
                "name": "batch_of_4096_grid",
                "kind": "intersect_grid",
                "x_min": -0.5,
                "x_max": 1.5,
                "y_min": -0.5,
                "y_max": 1.5,
                "res": 64,
                "z": -1.0,
                "dir_z": 1.0,
            },
            {
                "name": "batch_of_zero",
                "kind": "intersect",
                "origins": [],
                "dirs": [],
                "expect_raises": True,
            },
        ],
    },
    {
        "name": "edge_queries",
        "cases": ["point_ray_nearest", "finite_infinite_ray", "boundary_edge", "topk"],
        "meshes": [{"vertices": QUAD_VERTS, "faces": QUAD_FACES}],
        "queries": [
            {
                "name": "point_near_boundary_edge",
                "kind": "nearest_edge_point",
                "points": [[0.5, -0.2, 0.0]],
            },
            {
                "name": "point_near_internal_edge",
                "kind": "nearest_edge_point",
                "points": [[0.52, 0.48, 0.0]],
            },
            {
                "name": "ray_finite_segment",
                "kind": "nearest_edge_ray",
                "origins": [[0.5, 0.0, 1.0]],
                "dirs": [[0.0, 0.0, -1.0]],
                "tmax": [2.0],
            },
            {
                "name": "ray_infinite",
                "kind": "nearest_edge_ray",
                "origins": [[0.5, 0.0, 1.0]],
                "dirs": [[0.0, 0.0, -1.0]],
            },
            {
                "name": "topk_k4",
                "kind": "nearest_edges",
                "points": [[0.35, 0.2, 0.0]],
                "k": 4,
            },
        ],
    },
    {
        "name": "edge_tie",
        "cases": ["equal_distance_tie"],
        "meshes": [
            {
                # two triangles giving edges symmetric about y = 0 through origin
                "vertices": [
                    [-1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 2.0, 0.0],
                    [-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, -2.0, 0.0],
                ],
                "faces": [[0, 1, 2], [3, 4, 5]],
            }
        ],
        "queries": [
            {
                "name": "point_equidistant_two_edges",
                "kind": "nearest_edge_point",
                "points": [[0.0, 0.0, 0.0]],
                "informative": [
                    "shape_id", "edge_id", "global_edge_id",
                    "is_boundary", "edge_t", "edge_point",
                ],
            },
            {
                "name": "topk_equidistant",
                "kind": "nearest_edges",
                "points": [[0.0, 0.0, 0.0]],
                "k": 4,
                "informative": [
                    "shape_ids", "edge_ids", "global_edge_ids",
                    "is_boundary", "edge_t", "points", "edge_points",
                ],
            },
        ],
    },
]
