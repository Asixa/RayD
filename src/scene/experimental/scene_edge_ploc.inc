// Experimental PLOC host-side reference implementation and helper types.
// This include unit is intentionally pulled into scene_edge.cpp's anonymous namespace.

constexpr int EdgeBVHPLOCSearchRadius = 8;
constexpr int EdgeBVHPLOCSearchRadiusMid = 16;
constexpr int EdgeBVHPLOCSearchRadiusWide = 32;

struct PLOCCluster {
    int node_index = -1;
    ScalarVector3f bbox_min;
    ScalarVector3f bbox_max;
    ScalarVector3f centroid;
};

struct PLOCCandidatePair {
    int left_index = -1;
    int right_index = -1;
    float cost = 0.f;
};

struct HostRawEdgeBVH {
    std::vector<int> left_child;
    std::vector<int> right_child;
    std::vector<int> is_leaf;
    std::vector<int> leaf_primitive;
    std::vector<int> primitive_leaf_node;
    std::vector<ScalarVector3f> node_bbox_min;
    std::vector<ScalarVector3f> node_bbox_max;
};

float ploc_merge_cost(const PLOCCluster &a, const PLOCCluster &b) {
    return bbox_surface_area(scalar_min(a.bbox_min, b.bbox_min),
                             scalar_max(a.bbox_max, b.bbox_max));
}

PLOCCluster merge_ploc_clusters(const PLOCCluster &left,
                                const PLOCCluster &right,
                                int node_index,
                                HostRawEdgeBVH &raw_bvh) {
    raw_bvh.left_child[static_cast<size_t>(node_index)] = left.node_index;
    raw_bvh.right_child[static_cast<size_t>(node_index)] = right.node_index;
    raw_bvh.leaf_primitive[static_cast<size_t>(node_index)] = -1;
    raw_bvh.is_leaf[static_cast<size_t>(node_index)] = 0;
    raw_bvh.node_bbox_min[static_cast<size_t>(node_index)] =
        scalar_min(left.bbox_min, right.bbox_min);
    raw_bvh.node_bbox_max[static_cast<size_t>(node_index)] =
        scalar_max(left.bbox_max, right.bbox_max);
    return PLOCCluster{
        node_index,
        raw_bvh.node_bbox_min[static_cast<size_t>(node_index)],
        raw_bvh.node_bbox_max[static_cast<size_t>(node_index)],
        (raw_bvh.node_bbox_min[static_cast<size_t>(node_index)] +
         raw_bvh.node_bbox_max[static_cast<size_t>(node_index)]) * 0.5f
    };
}

int ploc_search_radius(int cluster_count) {
    if (cluster_count <= 1) {
        return 0;
    }

    int search_radius = EdgeBVHPLOCSearchRadius;
    if (cluster_count <= 2048) {
        search_radius = EdgeBVHPLOCSearchRadiusWide;
    } else if (cluster_count <= 8192) {
        search_radius = EdgeBVHPLOCSearchRadiusMid;
    }
    return std::min(search_radius, cluster_count - 1);
}

float ploc_pairing_cost_sum(const std::vector<PLOCCluster> &clusters, int begin, int end) {
    float cost = 0.f;
    for (int index = begin; index + 1 <= end; index += 2) {
        cost += ploc_merge_cost(clusters[static_cast<size_t>(index)],
                                clusters[static_cast<size_t>(index + 1)]);
    }
    return cost;
}

HostRawEdgeBVH build_ploc_bvh_host(const std::vector<ScalarVector3f> &primitive_bbox_min,
                                   const std::vector<ScalarVector3f> &primitive_bbox_max) {
    const int primitive_count = static_cast<int>(primitive_bbox_min.size());
    require(primitive_count == static_cast<int>(primitive_bbox_max.size()),
            "SceneEdge::build(): invalid primitive bbox arrays for PLOC.");

    const int node_count = std::max(2 * primitive_count - 1, 1);
    const int internal_count = std::max(primitive_count - 1, 0);
    HostRawEdgeBVH raw_bvh;
    raw_bvh.left_child.assign(static_cast<size_t>(node_count), -1);
    raw_bvh.right_child.assign(static_cast<size_t>(node_count), -1);
    raw_bvh.is_leaf.assign(static_cast<size_t>(node_count), 0);
    raw_bvh.leaf_primitive.assign(static_cast<size_t>(node_count), -1);
    raw_bvh.primitive_leaf_node.assign(static_cast<size_t>(primitive_count), -1);
    raw_bvh.node_bbox_min.assign(static_cast<size_t>(node_count), empty_bbox_min());
    raw_bvh.node_bbox_max.assign(static_cast<size_t>(node_count), empty_bbox_max());

    if (primitive_count == 0) {
        return raw_bvh;
    }

    if (primitive_count == 1) {
        raw_bvh.is_leaf[0] = 1;
        raw_bvh.leaf_primitive[0] = 0;
        raw_bvh.primitive_leaf_node[0] = 0;
        raw_bvh.node_bbox_min[0] = primitive_bbox_min[0];
        raw_bvh.node_bbox_max[0] = primitive_bbox_max[0];
        return raw_bvh;
    }

    ScalarVector3f scene_bbox_min = empty_bbox_min();
    ScalarVector3f scene_bbox_max = empty_bbox_max();
    std::vector<ScalarVector3f> primitive_centroids(static_cast<size_t>(primitive_count));
    for (int primitive = 0; primitive < primitive_count; ++primitive) {
        scene_bbox_min = scalar_min(scene_bbox_min, primitive_bbox_min[static_cast<size_t>(primitive)]);
        scene_bbox_max = scalar_max(scene_bbox_max, primitive_bbox_max[static_cast<size_t>(primitive)]);
        primitive_centroids[static_cast<size_t>(primitive)] =
            (primitive_bbox_min[static_cast<size_t>(primitive)] +
             primitive_bbox_max[static_cast<size_t>(primitive)]) * 0.5f;
    }

    std::vector<uint32_t> morton_codes(static_cast<size_t>(primitive_count), 0u);
    std::vector<int> primitive_order(static_cast<size_t>(primitive_count), 0);
    for (int primitive = 0; primitive < primitive_count; ++primitive) {
        primitive_order[static_cast<size_t>(primitive)] = primitive;
        morton_codes[static_cast<size_t>(primitive)] =
            morton_code_3d_host(primitive_centroids[static_cast<size_t>(primitive)],
                                scene_bbox_min,
                                scene_bbox_max);
    }
    std::stable_sort(primitive_order.begin(),
                     primitive_order.end(),
                     [&morton_codes](int a, int b) {
                         const uint32_t code_a = morton_codes[static_cast<size_t>(a)];
                         const uint32_t code_b = morton_codes[static_cast<size_t>(b)];
                         if (code_a != code_b) {
                             return code_a < code_b;
                         }
                         return a < b;
                     });

    std::vector<PLOCCluster> current_clusters;
    current_clusters.reserve(static_cast<size_t>(primitive_count));
    for (int rank = 0; rank < primitive_count; ++rank) {
        const int primitive = primitive_order[static_cast<size_t>(rank)];
        const int node_index = internal_count + rank;
        raw_bvh.is_leaf[static_cast<size_t>(node_index)] = 1;
        raw_bvh.leaf_primitive[static_cast<size_t>(node_index)] = primitive;
        raw_bvh.primitive_leaf_node[static_cast<size_t>(primitive)] = node_index;
        raw_bvh.node_bbox_min[static_cast<size_t>(node_index)] =
            primitive_bbox_min[static_cast<size_t>(primitive)];
        raw_bvh.node_bbox_max[static_cast<size_t>(node_index)] =
            primitive_bbox_max[static_cast<size_t>(primitive)];
        current_clusters.push_back(PLOCCluster{
            node_index,
            raw_bvh.node_bbox_min[static_cast<size_t>(node_index)],
            raw_bvh.node_bbox_max[static_cast<size_t>(node_index)],
            primitive_centroids[static_cast<size_t>(primitive)]
        });
    }

    int next_internal_node = internal_count - 1;
    while (current_clusters.size() > 1) {
        const int cluster_count = static_cast<int>(current_clusters.size());
        const int search_radius = ploc_search_radius(cluster_count);
        std::vector<int> best_neighbors(static_cast<size_t>(cluster_count), -1);
        std::vector<float> best_costs(static_cast<size_t>(cluster_count),
                                      std::numeric_limits<float>::infinity());

        for (int cluster_index = 0; cluster_index < cluster_count; ++cluster_index) {
            const int begin = std::max(0, cluster_index - search_radius);
            const int end = std::min(cluster_count - 1, cluster_index + search_radius);
            for (int neighbor_index = begin; neighbor_index <= end; ++neighbor_index) {
                if (neighbor_index == cluster_index) {
                    continue;
                }
                const float candidate_cost =
                    ploc_merge_cost(current_clusters[static_cast<size_t>(cluster_index)],
                                    current_clusters[static_cast<size_t>(neighbor_index)]);
                if (candidate_cost < best_costs[static_cast<size_t>(cluster_index)] ||
                    (candidate_cost == best_costs[static_cast<size_t>(cluster_index)] &&
                     neighbor_index < best_neighbors[static_cast<size_t>(cluster_index)])) {
                    best_costs[static_cast<size_t>(cluster_index)] = candidate_cost;
                    best_neighbors[static_cast<size_t>(cluster_index)] = neighbor_index;
                }
            }
        }

        std::vector<PLOCCandidatePair> candidate_pairs;
        candidate_pairs.reserve(static_cast<size_t>(cluster_count));
        for (int cluster_index = 0; cluster_index < cluster_count; ++cluster_index) {
            const int neighbor_index = best_neighbors[static_cast<size_t>(cluster_index)];
            if (neighbor_index >= 0 && neighbor_index < cluster_count &&
                neighbor_index != cluster_index) {
                const int left_index = std::min(cluster_index, neighbor_index);
                const int right_index = std::max(cluster_index, neighbor_index);
                candidate_pairs.push_back(PLOCCandidatePair{
                    left_index,
                    right_index,
                    ploc_merge_cost(current_clusters[static_cast<size_t>(left_index)],
                                    current_clusters[static_cast<size_t>(right_index)])
                });
            }
        }

        std::sort(candidate_pairs.begin(),
                  candidate_pairs.end(),
                  [](const PLOCCandidatePair &a, const PLOCCandidatePair &b) {
                      if (a.left_index != b.left_index) {
                          return a.left_index < b.left_index;
                      }
                      if (a.right_index != b.right_index) {
                          return a.right_index < b.right_index;
                      }
                      return a.cost < b.cost;
                  });
        candidate_pairs.erase(
            std::unique(candidate_pairs.begin(),
                        candidate_pairs.end(),
                        [](const PLOCCandidatePair &a, const PLOCCandidatePair &b) {
                            return a.left_index == b.left_index &&
                                   a.right_index == b.right_index;
                        }),
            candidate_pairs.end());
        std::sort(candidate_pairs.begin(),
                  candidate_pairs.end(),
                  [](const PLOCCandidatePair &a, const PLOCCandidatePair &b) {
                      if (a.cost != b.cost) {
                          return a.cost < b.cost;
                      }
                      const int span_a = a.right_index - a.left_index;
                      const int span_b = b.right_index - b.left_index;
                      if (span_a != span_b) {
                          return span_a < span_b;
                      }
                      return a.left_index < b.left_index;
                  });

        std::vector<int> pair_with(static_cast<size_t>(cluster_count), -1);
        for (const PLOCCandidatePair &candidate : candidate_pairs) {
            if (pair_with[static_cast<size_t>(candidate.left_index)] >= 0 ||
                pair_with[static_cast<size_t>(candidate.right_index)] >= 0) {
                continue;
            }
            pair_with[static_cast<size_t>(candidate.left_index)] = candidate.right_index;
            pair_with[static_cast<size_t>(candidate.right_index)] = candidate.left_index;
        }

        std::vector<PLOCCluster> next_clusters;
        next_clusters.reserve(static_cast<size_t>((cluster_count + 1) / 2));
        const auto emit_merge = [&](int left_index, int right_index) {
            require(next_internal_node >= 0,
                    "SceneEdge::build(): PLOC ran out of internal nodes.");
            next_clusters.push_back(
                merge_ploc_clusters(current_clusters[static_cast<size_t>(left_index)],
                                    current_clusters[static_cast<size_t>(right_index)],
                                    next_internal_node--,
                                    raw_bvh));
        };

        int cluster_index = 0;
        while (cluster_index < cluster_count) {
            const int partner_index = pair_with[static_cast<size_t>(cluster_index)];
            if (partner_index > cluster_index) {
                emit_merge(cluster_index, partner_index);
                ++cluster_index;
                continue;
            }

            if (partner_index >= 0) {
                ++cluster_index;
                continue;
            }

            const int run_begin = cluster_index;
            while (cluster_index < cluster_count &&
                   pair_with[static_cast<size_t>(cluster_index)] < 0) {
                ++cluster_index;
            }
            const int run_end = cluster_index - 1;
            const int run_length = run_end - run_begin + 1;

            if (run_length == 1) {
                next_clusters.push_back(current_clusters[static_cast<size_t>(run_begin)]);
                continue;
            }

            int pair_begin = run_begin;
            if ((run_length & 1) != 0) {
                const float leave_last_cost =
                    ploc_pairing_cost_sum(current_clusters, run_begin, run_end - 1);
                const float leave_first_cost =
                    ploc_pairing_cost_sum(current_clusters, run_begin + 1, run_end);
                if (leave_first_cost < leave_last_cost) {
                    next_clusters.push_back(current_clusters[static_cast<size_t>(run_begin)]);
                    pair_begin = run_begin + 1;
                }
            }

            for (int pair_index = pair_begin; pair_index + 1 <= run_end; pair_index += 2) {
                emit_merge(pair_index, pair_index + 1);
            }

            if (((run_end - pair_begin + 1) & 1) != 0) {
                next_clusters.push_back(current_clusters[static_cast<size_t>(run_end)]);
            }
        }

        require(next_clusters.size() < current_clusters.size(),
                "SceneEdge::build(): PLOC did not make clustering progress.");
        current_clusters = std::move(next_clusters);
    }

    require(current_clusters.size() == 1 && current_clusters[0].node_index == 0,
            "SceneEdge::build(): PLOC did not produce a valid root.");
    require(next_internal_node == -1,
            "SceneEdge::build(): PLOC did not consume every internal node.");
    return raw_bvh;
}
