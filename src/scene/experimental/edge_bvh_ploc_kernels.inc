// Experimental PLOC GPU kernels. This include unit is intentionally pulled into edge_bvh.cu.

__device__ inline float ploc_merge_cost_bounds(const Bounds3 &a, const Bounds3 &b) {
    return bbox_surface_area_bounds(merge_bounds(a, b));
}

__global__ void initialize_ploc_leaves_kernel(
    int primitive_count,
    int internal_count,
    const int *sorted_primitives,
    const float *primitive_bbox_min_x,
    const float *primitive_bbox_min_y,
    const float *primitive_bbox_min_z,
    const float *primitive_bbox_max_x,
    const float *primitive_bbox_max_y,
    const float *primitive_bbox_max_z,
    float *node_bbox_min_x,
    float *node_bbox_min_y,
    float *node_bbox_min_z,
    float *node_bbox_max_x,
    float *node_bbox_max_y,
    float *node_bbox_max_z,
    int *leaf_primitive,
    int *is_leaf,
    int *primitive_leaf_node,
    int *current_nodes,
    Bounds3 *current_bounds) {
    const int leaf_index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (leaf_index >= primitive_count) {
        return;
    }

    const int primitive = sorted_primitives[leaf_index];
    const int node_index = internal_count + leaf_index;
    const Bounds3 bounds{
        Float3(primitive_bbox_min_x[primitive],
               primitive_bbox_min_y[primitive],
               primitive_bbox_min_z[primitive]),
        Float3(primitive_bbox_max_x[primitive],
               primitive_bbox_max_y[primitive],
               primitive_bbox_max_z[primitive])
    };

    store_bounds(node_index,
                 bounds,
                 node_bbox_min_x,
                 node_bbox_min_y,
                 node_bbox_min_z,
                 node_bbox_max_x,
                 node_bbox_max_y,
                 node_bbox_max_z);
    leaf_primitive[node_index] = primitive;
    is_leaf[node_index] = 1;
    primitive_leaf_node[primitive] = node_index;
    current_nodes[leaf_index] = node_index;
    current_bounds[leaf_index] = bounds;
}

__global__ void compute_ploc_best_neighbors_kernel(int cluster_count,
                                                   int search_radius,
                                                   const Bounds3 *current_bounds,
                                                   int *best_neighbor) {
    const int cluster_index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (cluster_index >= cluster_count) {
        return;
    }

    const Bounds3 self_bounds = current_bounds[cluster_index];
    const int begin = max(0, cluster_index - search_radius);
    const int end = min(cluster_count - 1, cluster_index + search_radius);
    float best_cost = 1e30f;
    int best_index = -1;
    for (int neighbor_index = begin; neighbor_index <= end; ++neighbor_index) {
        if (neighbor_index == cluster_index) {
            continue;
        }
        const float candidate_cost =
            ploc_merge_cost_bounds(self_bounds, current_bounds[neighbor_index]);
        if (candidate_cost < best_cost ||
            (candidate_cost == best_cost &&
             (best_index < 0 || neighbor_index < best_index))) {
            best_cost = candidate_cost;
            best_index = neighbor_index;
        }
    }
    best_neighbor[cluster_index] = best_index;
}

__device__ inline uint64_t pack_ploc_candidate_sort_key(float cost, uint32_t tie_breaker) {
    return (static_cast<uint64_t>(__float_as_uint(cost)) << 32) |
           static_cast<uint64_t>(tie_breaker);
}

__global__ void generate_ploc_candidate_pairs_kernel(int cluster_count,
                                                     const Bounds3 *current_bounds,
                                                     const int *best_neighbor,
                                                     uint64_t *candidate_sort_keys,
                                                     int *candidate_ids,
                                                     int *candidate_left,
                                                     int *candidate_right) {
    const int cluster_index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (cluster_index >= cluster_count) {
        return;
    }

    candidate_ids[cluster_index] = cluster_index;
    const int neighbor_index = best_neighbor[cluster_index];
    if (neighbor_index < 0 || neighbor_index >= cluster_count || neighbor_index == cluster_index) {
        candidate_left[cluster_index] = -1;
        candidate_right[cluster_index] = -1;
        candidate_sort_keys[cluster_index] = 0xffffffffffffffffull;
        return;
    }

    const int left_index = min(cluster_index, neighbor_index);
    const int right_index = max(cluster_index, neighbor_index);
    candidate_left[cluster_index] = left_index;
    candidate_right[cluster_index] = right_index;
    candidate_sort_keys[cluster_index] = pack_ploc_candidate_sort_key(
        ploc_merge_cost_bounds(current_bounds[left_index], current_bounds[right_index]),
        static_cast<uint32_t>(cluster_index));
}

__global__ void greedy_select_ploc_sorted_pairs_kernel(int candidate_count,
                                                       const int *sorted_candidate_ids,
                                                       const int *candidate_left,
                                                       const int *candidate_right,
                                                       int *pair_role,
                                                       int *pair_partner,
                                                       int *node_used) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    for (int candidate_rank = 0; candidate_rank < candidate_count; ++candidate_rank) {
        const int candidate_id = sorted_candidate_ids[candidate_rank];
        const int left_index = candidate_left[candidate_id];
        const int right_index = candidate_right[candidate_id];
        if (left_index < 0 || right_index < 0) {
            continue;
        }
        if (node_used[left_index] != 0 || node_used[right_index] != 0) {
            continue;
        }

        node_used[left_index] = 1;
        node_used[right_index] = 1;
        pair_role[left_index] = 1;
        pair_role[right_index] = -1;
        pair_partner[left_index] = right_index;
        pair_partner[right_index] = left_index;
    }
}

__global__ void mark_ploc_unmatched_runs_kernel(int cluster_count,
                                                const int *pair_role,
                                                int *run_start_flag,
                                                int *run_end_flag) {
    const int cluster_index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (cluster_index >= cluster_count) {
        return;
    }

    const bool unmatched = pair_role[cluster_index] == 0;
    run_start_flag[cluster_index] =
        unmatched && (cluster_index == 0 || pair_role[cluster_index - 1] != 0) ? 1 : 0;
    run_end_flag[cluster_index] =
        unmatched && (cluster_index + 1 == cluster_count || pair_role[cluster_index + 1] != 0) ? 1 : 0;
}

__global__ void scatter_ploc_run_bounds_kernel(int cluster_count,
                                               const int *pair_role,
                                               const int *run_start_flag,
                                               const int *run_end_flag,
                                               const int *run_prefix,
                                               int *run_start_indices,
                                               int *run_end_indices) {
    const int cluster_index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (cluster_index >= cluster_count || pair_role[cluster_index] != 0) {
        return;
    }

    const int run_id = run_prefix[cluster_index] - 1;
    if (run_id < 0) {
        return;
    }

    if (run_start_flag[cluster_index] != 0) {
        run_start_indices[run_id] = cluster_index;
    }
    if (run_end_flag[cluster_index] != 0) {
        run_end_indices[run_id] = cluster_index;
    }
}

__global__ void compute_ploc_run_pairing_mode_from_starts_kernel(
    int cluster_count,
    const Bounds3 *current_bounds,
    const int *run_start_flag,
    const int *run_prefix,
    const int *run_end_indices,
    int *run_leave_first) {
    const int cluster_index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (cluster_index >= cluster_count || run_start_flag[cluster_index] == 0) {
        return;
    }

    const int run_index = run_prefix[cluster_index] - 1;
    const int start = cluster_index;
    const int end = run_end_indices[run_index];
    const int run_length = end - start + 1;
    int leave_first = 0;
    if (run_length > 1 && (run_length & 1) != 0) {
        float leave_last_cost = 0.f;
        for (int index = start; index + 1 <= end - 1; index += 2) {
            leave_last_cost +=
                ploc_merge_cost_bounds(current_bounds[index], current_bounds[index + 1]);
        }

        float leave_first_cost = 0.f;
        for (int index = start + 1; index + 1 <= end; index += 2) {
            leave_first_cost +=
                ploc_merge_cost_bounds(current_bounds[index], current_bounds[index + 1]);
        }
        leave_first = leave_first_cost < leave_last_cost ? 1 : 0;
    }

    run_leave_first[run_index] = leave_first;
}

__global__ void build_ploc_emission_plan_kernel(int cluster_count,
                                                const int *pair_role,
                                                const int *pair_partner,
                                                const int *run_prefix,
                                                const int *run_start_indices,
                                                const int *run_end_indices,
                                                const int *run_leave_first,
                                                int *emit_flag,
                                                int *merge_flag,
                                                int *emit_partner) {
    const int cluster_index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (cluster_index >= cluster_count) {
        return;
    }

    emit_flag[cluster_index] = 0;
    merge_flag[cluster_index] = 0;
    emit_partner[cluster_index] = -1;

    if (pair_role[cluster_index] == 1) {
        emit_flag[cluster_index] = 1;
        merge_flag[cluster_index] = 1;
        emit_partner[cluster_index] = pair_partner[cluster_index];
        return;
    }

    if (pair_role[cluster_index] != 0) {
        return;
    }

    const int run_id = run_prefix[cluster_index] - 1;
    if (run_id < 0) {
        return;
    }

    const int start = run_start_indices[run_id];
    const int end = run_end_indices[run_id];
    const int run_length = end - start + 1;
    if (run_length == 1) {
        emit_flag[cluster_index] = 1;
        return;
    }

    const int leave_first = run_leave_first[run_id];
    if (leave_first != 0 && cluster_index == start) {
        emit_flag[cluster_index] = 1;
        return;
    }
    if (leave_first == 0 && (run_length & 1) != 0 && cluster_index == end) {
        emit_flag[cluster_index] = 1;
        return;
    }

    const int pair_begin = start + leave_first;
    const int pair_end = end - ((leave_first == 0 && (run_length & 1) != 0) ? 1 : 0);
    if (cluster_index < pair_begin || cluster_index > pair_end) {
        return;
    }

    const int offset = cluster_index - pair_begin;
    if ((offset & 1) == 0 && cluster_index + 1 <= pair_end) {
        emit_flag[cluster_index] = 1;
        merge_flag[cluster_index] = 1;
        emit_partner[cluster_index] = cluster_index + 1;
    }
}

__global__ void emit_ploc_next_clusters_kernel(
    int cluster_count,
    int next_internal_node,
    const int *current_nodes_in,
    const Bounds3 *current_bounds_in,
    const int *emit_flag,
    const int *emit_offsets,
    const int *merge_flag,
    const int *merge_offsets,
    const int *emit_partner,
    int *current_nodes_out,
    Bounds3 *current_bounds_out,
    float *node_bbox_min_x,
    float *node_bbox_min_y,
    float *node_bbox_min_z,
    float *node_bbox_max_x,
    float *node_bbox_max_y,
    float *node_bbox_max_z,
    int *left_child,
    int *right_child,
    int *leaf_primitive,
    int *is_leaf,
    int *parent) {
    const int cluster_index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (cluster_index >= cluster_count || emit_flag[cluster_index] == 0) {
        return;
    }

    const int out_index = emit_offsets[cluster_index];
    if (merge_flag[cluster_index] != 0) {
        const int partner_index = emit_partner[cluster_index];
        const int node_index = next_internal_node - merge_offsets[cluster_index];
        const Bounds3 merged =
            merge_bounds(current_bounds_in[cluster_index], current_bounds_in[partner_index]);
        current_nodes_out[out_index] = node_index;
        current_bounds_out[out_index] = merged;
        left_child[node_index] = current_nodes_in[cluster_index];
        right_child[node_index] = current_nodes_in[partner_index];
        parent[current_nodes_in[cluster_index]] = node_index;
        parent[current_nodes_in[partner_index]] = node_index;
        leaf_primitive[node_index] = -1;
        is_leaf[node_index] = 0;
        store_bounds(node_index,
                     merged,
                     node_bbox_min_x,
                     node_bbox_min_y,
                     node_bbox_min_z,
                     node_bbox_max_x,
                     node_bbox_max_y,
                     node_bbox_max_z);
        return;
    }

    current_nodes_out[out_index] = current_nodes_in[cluster_index];
    current_bounds_out[out_index] = current_bounds_in[cluster_index];
}

__global__ void finalize_ploc_round_counts_kernel(int cluster_count,
                                                  const int *run_prefix,
                                                  const int *emit_flag,
                                                  const int *emit_offsets,
                                                  const int *merge_flag,
                                                  const int *merge_offsets,
                                                  int *round_counts) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    if (cluster_count <= 0) {
        round_counts[0] = 0;
        round_counts[1] = 0;
        round_counts[2] = 0;
        return;
    }

    const int last_index = cluster_count - 1;
    round_counts[0] = run_prefix[last_index];
    round_counts[1] = emit_offsets[last_index] + emit_flag[last_index];
    round_counts[2] = merge_offsets[last_index] + merge_flag[last_index];
}
