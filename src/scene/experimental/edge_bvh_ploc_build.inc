// Experimental PLOC GPU build path. This include unit is intentionally pulled into edge_bvh.cu.

int ploc_search_radius_gpu(int cluster_count) {
    if (cluster_count <= 1) {
        return 0;
    }

    int search_radius = 8;
    if (cluster_count <= 2048) {
        search_radius = 32;
    } else if (cluster_count <= 8192) {
        search_radius = 16;
    }
    return std::min(search_radius, cluster_count - 1);
}

void build_edge_ploc_bvh_gpu(
    int primitive_count,
    const float *edge_p0_x,
    const float *edge_p0_y,
    const float *edge_p0_z,
    const float *edge_e1_x,
    const float *edge_e1_y,
    const float *edge_e1_z,
    float *primitive_bbox_min_x,
    float *primitive_bbox_min_y,
    float *primitive_bbox_min_z,
    float *primitive_bbox_max_x,
    float *primitive_bbox_max_y,
    float *primitive_bbox_max_z,
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
    int *primitive_leaf_node) {
    require_local(primitive_count > 0, "build_edge_ploc_bvh_gpu(): primitive_count must be positive.");

    try {
        const int node_count = std::max(2 * primitive_count - 1, 1);
        const int internal_count = std::max(primitive_count - 1, 0);
        const int block_size = 256;
        const int primitive_blocks = (primitive_count + block_size - 1) / block_size;
        const EdgeBVHPostBuildStrategy post_build_strategy = active_edge_bvh_post_build_strategy();
        const EdgeBVHTreeletScheduleMode treelet_schedule_mode =
            active_edge_bvh_treelet_schedule_mode();
        const bool treelet_enabled =
            post_build_strategy == EdgeBVHPostBuildStrategy::GpuTreelet &&
            primitive_count >= EdgeBVHTreeletMinPrimitives &&
            internal_count > 0;
        CudaStreamHandle stream_handle;
        const cudaStream_t stream = stream_handle.get();

        CudaBuffer<Bounds3> primitive_bounds(static_cast<size_t>(primitive_count));
        CudaBuffer<Bounds3> reduced_bounds(1);
        CudaBuffer<uint32_t> morton_codes_in(static_cast<size_t>(primitive_count));
        CudaBuffer<uint32_t> morton_codes_out(static_cast<size_t>(primitive_count));
        CudaBuffer<int> primitive_indices_in(static_cast<size_t>(primitive_count));
        CudaBuffer<int> primitive_indices_out(static_cast<size_t>(primitive_count));
        CudaBuffer<uint64_t> candidate_sort_keys_in(static_cast<size_t>(primitive_count));
        CudaBuffer<uint64_t> candidate_sort_keys_out(static_cast<size_t>(primitive_count));
        CudaBuffer<int> candidate_ids_in(static_cast<size_t>(primitive_count));
        CudaBuffer<int> candidate_ids_out(static_cast<size_t>(primitive_count));
        CudaBuffer<int> candidate_left(static_cast<size_t>(primitive_count));
        CudaBuffer<int> candidate_right(static_cast<size_t>(primitive_count));
        CudaBuffer<int> current_nodes_a(static_cast<size_t>(primitive_count));
        CudaBuffer<int> current_nodes_b(static_cast<size_t>(primitive_count));
        CudaBuffer<Bounds3> current_bounds_a(static_cast<size_t>(primitive_count));
        CudaBuffer<Bounds3> current_bounds_b(static_cast<size_t>(primitive_count));
        CudaBuffer<int> best_neighbor(static_cast<size_t>(primitive_count));
        CudaBuffer<int> node_used(static_cast<size_t>(primitive_count));
        CudaBuffer<int> pair_role(static_cast<size_t>(primitive_count));
        CudaBuffer<int> pair_partner(static_cast<size_t>(primitive_count));
        CudaBuffer<int> run_start_flag(static_cast<size_t>(primitive_count));
        CudaBuffer<int> run_end_flag(static_cast<size_t>(primitive_count));
        CudaBuffer<int> run_prefix(static_cast<size_t>(primitive_count));
        CudaBuffer<int> run_start_indices(static_cast<size_t>(primitive_count));
        CudaBuffer<int> run_end_indices(static_cast<size_t>(primitive_count));
        CudaBuffer<int> run_leave_first(static_cast<size_t>(primitive_count));
        CudaBuffer<int> emit_flag(static_cast<size_t>(primitive_count));
        CudaBuffer<int> emit_offsets(static_cast<size_t>(primitive_count));
        CudaBuffer<int> merge_flag(static_cast<size_t>(primitive_count));
        CudaBuffer<int> merge_offsets(static_cast<size_t>(primitive_count));
        CudaBuffer<int> emit_partner(static_cast<size_t>(primitive_count));
        CudaBuffer<int> round_counts(3);
        CudaBuffer<int> parent(static_cast<size_t>(node_count));
        CudaBuffer<float> node_costs(static_cast<size_t>(node_count));
        std::vector<int> host_left_child;
        std::vector<int> host_right_child;
        std::vector<int> host_is_leaf;
        std::vector<int> subtree_leaf_counts;
        std::vector<int> node_heights;
        std::vector<std::vector<int>> host_level_groups;
        FlatNodeLevels internal_level_schedule;
        int max_height = 0;

        compute_primitive_bounds_kernel<<<primitive_blocks, block_size, 0, stream>>>(
            primitive_count,
            edge_p0_x,
            edge_p0_y,
            edge_p0_z,
            edge_e1_x,
            edge_e1_y,
            edge_e1_z,
            primitive_bbox_min_x,
            primitive_bbox_min_y,
            primitive_bbox_min_z,
            primitive_bbox_max_x,
            primitive_bbox_max_y,
            primitive_bbox_max_z,
            primitive_bounds.get());
        check_cuda_last_error("build_edge_ploc_bvh_gpu(): failed to launch primitive-bounds kernel");

        size_t reduce_temp_size = 0;
        check_cuda_call(
            cub::DeviceReduce::Reduce(nullptr,
                                      reduce_temp_size,
                                      primitive_bounds.get(),
                                      reduced_bounds.get(),
                                      primitive_count,
                                      BoundsUnion(),
                                      Bounds3::empty(),
                                      stream),
            "build_edge_ploc_bvh_gpu(): failed to size scene-bound reduction");
        CudaBuffer<char> reduce_temp(reduce_temp_size);
        check_cuda_call(
            cub::DeviceReduce::Reduce(reduce_temp.get(),
                                      reduce_temp_size,
                                      primitive_bounds.get(),
                                      reduced_bounds.get(),
                                      primitive_count,
                                      BoundsUnion(),
                                      Bounds3::empty(),
                                      stream),
            "build_edge_ploc_bvh_gpu(): failed to reduce scene bounds");
        check_cuda_last_error("build_edge_ploc_bvh_gpu(): failed to launch scene-bound reduction");

        Bounds3 scene_bounds = Bounds3::empty();
        check_cuda_call(cudaMemcpyAsync(&scene_bounds,
                                        reduced_bounds.get(),
                                        sizeof(Bounds3),
                                        cudaMemcpyDeviceToHost,
                                        stream),
                        "build_edge_ploc_bvh_gpu(): failed to copy scene bounds");
        init_sequence_kernel<<<primitive_blocks, block_size, 0, stream>>>(
            primitive_count, primitive_indices_in.get());
        check_cuda_last_error("build_edge_ploc_bvh_gpu(): failed to launch primitive-index initialization");
        check_cuda_call(cudaStreamSynchronize(stream),
                        "build_edge_ploc_bvh_gpu(): failed to prepare Morton inputs");

        compute_morton_codes_kernel<<<primitive_blocks, block_size, 0, stream>>>(
            primitive_count,
            scene_bounds,
            primitive_bbox_min_x,
            primitive_bbox_min_y,
            primitive_bbox_min_z,
            primitive_bbox_max_x,
            primitive_bbox_max_y,
            primitive_bbox_max_z,
            morton_codes_in.get());
        check_cuda_last_error("build_edge_ploc_bvh_gpu(): failed to launch Morton-code kernel");

        size_t sort_temp_size = 0;
        check_cuda_call(
            cub::DeviceRadixSort::SortPairs(nullptr,
                                            sort_temp_size,
                                            morton_codes_in.get(),
                                            morton_codes_out.get(),
                                            primitive_indices_in.get(),
                                            primitive_indices_out.get(),
                                            primitive_count,
                                            0,
                                            32,
                                            stream),
            "build_edge_ploc_bvh_gpu(): failed to size radix sort");
        CudaBuffer<char> sort_temp(sort_temp_size);
        size_t pair_sort_temp_size = 0;
        check_cuda_call(
            cub::DeviceRadixSort::SortPairs(nullptr,
                                            pair_sort_temp_size,
                                            candidate_sort_keys_in.get(),
                                            candidate_sort_keys_out.get(),
                                            candidate_ids_in.get(),
                                            candidate_ids_out.get(),
                                            primitive_count,
                                            0,
                                            64,
                                            stream),
            "build_edge_ploc_bvh_gpu(): failed to size candidate sort");
        CudaBuffer<char> pair_sort_temp(pair_sort_temp_size);
        check_cuda_call(
            cub::DeviceRadixSort::SortPairs(sort_temp.get(),
                                            sort_temp_size,
                                            morton_codes_in.get(),
                                            morton_codes_out.get(),
                                            primitive_indices_in.get(),
                                            primitive_indices_out.get(),
                                            primitive_count,
                                            0,
                                            32,
                                            stream),
            "build_edge_ploc_bvh_gpu(): failed to sort Morton codes");
        check_cuda_last_error("build_edge_ploc_bvh_gpu(): failed to launch Morton sort");

        memset_int_async(left_child,
                         -1,
                         static_cast<size_t>(node_count),
                         stream,
                         "build_edge_ploc_bvh_gpu(): failed to init left_child");
        memset_int_async(right_child,
                         -1,
                         static_cast<size_t>(node_count),
                         stream,
                         "build_edge_ploc_bvh_gpu(): failed to init right_child");
        memset_int_async(leaf_primitive,
                         -1,
                         static_cast<size_t>(node_count),
                         stream,
                         "build_edge_ploc_bvh_gpu(): failed to init leaf_primitive");
        memset_int_async(is_leaf,
                         0,
                         static_cast<size_t>(node_count),
                         stream,
                         "build_edge_ploc_bvh_gpu(): failed to init is_leaf");
        memset_int_async(primitive_leaf_node,
                         -1,
                         static_cast<size_t>(primitive_count),
                         stream,
                         "build_edge_ploc_bvh_gpu(): failed to init primitive_leaf_node");
        memset_int_async(parent.get(),
                         -1,
                         static_cast<size_t>(node_count),
                         stream,
                         "build_edge_ploc_bvh_gpu(): failed to init parent");

        initialize_ploc_leaves_kernel<<<primitive_blocks, block_size, 0, stream>>>(
            primitive_count,
            internal_count,
            primitive_indices_out.get(),
            primitive_bbox_min_x,
            primitive_bbox_min_y,
            primitive_bbox_min_z,
            primitive_bbox_max_x,
            primitive_bbox_max_y,
            primitive_bbox_max_z,
            node_bbox_min_x,
            node_bbox_min_y,
            node_bbox_min_z,
            node_bbox_max_x,
            node_bbox_max_y,
            node_bbox_max_z,
            leaf_primitive,
            is_leaf,
            primitive_leaf_node,
            current_nodes_a.get(),
            current_bounds_a.get());
        check_cuda_last_error("build_edge_ploc_bvh_gpu(): failed to initialize leaf clusters");

        size_t inclusive_scan_temp_size = 0;
        size_t exclusive_emit_temp_size = 0;
        size_t exclusive_merge_temp_size = 0;
        check_cuda_call(cub::DeviceScan::InclusiveSum(nullptr,
                                                      inclusive_scan_temp_size,
                                                      run_start_flag.get(),
                                                      run_prefix.get(),
                                                      primitive_count,
                                                      stream),
                        "build_edge_ploc_bvh_gpu(): failed to size run scan");
        check_cuda_call(cub::DeviceScan::ExclusiveSum(nullptr,
                                                      exclusive_emit_temp_size,
                                                      emit_flag.get(),
                                                      emit_offsets.get(),
                                                      primitive_count,
                                                      stream),
                        "build_edge_ploc_bvh_gpu(): failed to size emit scan");
        check_cuda_call(cub::DeviceScan::ExclusiveSum(nullptr,
                                                      exclusive_merge_temp_size,
                                                      merge_flag.get(),
                                                      merge_offsets.get(),
                                                      primitive_count,
                                                      stream),
                        "build_edge_ploc_bvh_gpu(): failed to size merge scan");
        CudaBuffer<char> scan_temp(std::max(inclusive_scan_temp_size,
                                            std::max(exclusive_emit_temp_size,
                                                     exclusive_merge_temp_size)));

        int current_cluster_count = primitive_count;
        int next_internal_node = internal_count - 1;
        int host_round_counts[3] = { 0, 0, 0 };
        CudaBuffer<int> *current_nodes_in = &current_nodes_a;
        CudaBuffer<int> *current_nodes_out = &current_nodes_b;
        CudaBuffer<Bounds3> *current_bounds_in = &current_bounds_a;
        CudaBuffer<Bounds3> *current_bounds_out = &current_bounds_b;

        while (current_cluster_count > 1) {
            const int cluster_blocks = (current_cluster_count + block_size - 1) / block_size;
            const int search_radius = ploc_search_radius_gpu(current_cluster_count);

            compute_ploc_best_neighbors_kernel<<<cluster_blocks, block_size, 0, stream>>>(
                current_cluster_count,
                search_radius,
                current_bounds_in->get(),
                best_neighbor.get());
            check_cuda_last_error("build_edge_ploc_bvh_gpu(): failed to launch neighbor search");

            generate_ploc_candidate_pairs_kernel<<<cluster_blocks, block_size, 0, stream>>>(
                current_cluster_count,
                current_bounds_in->get(),
                best_neighbor.get(),
                candidate_sort_keys_in.get(),
                candidate_ids_in.get(),
                candidate_left.get(),
                candidate_right.get());
            check_cuda_last_error("build_edge_ploc_bvh_gpu(): failed to generate candidate pairs");

            check_cuda_call(
                cub::DeviceRadixSort::SortPairs(pair_sort_temp.get(),
                                                pair_sort_temp_size,
                                                candidate_sort_keys_in.get(),
                                                candidate_sort_keys_out.get(),
                                                candidate_ids_in.get(),
                                                candidate_ids_out.get(),
                                                current_cluster_count,
                                                0,
                                                64,
                                                stream),
                "build_edge_ploc_bvh_gpu(): failed to sort candidate pairs");
            check_cuda_last_error("build_edge_ploc_bvh_gpu(): failed to launch candidate sort");

            memset_int_async(node_used.get(),
                             0,
                             static_cast<size_t>(current_cluster_count),
                             stream,
                             "build_edge_ploc_bvh_gpu(): failed to reset node-used flags");
            memset_int_async(pair_role.get(),
                             0,
                             static_cast<size_t>(current_cluster_count),
                             stream,
                             "build_edge_ploc_bvh_gpu(): failed to reset pair roles");
            memset_int_async(pair_partner.get(),
                             -1,
                             static_cast<size_t>(current_cluster_count),
                             stream,
                             "build_edge_ploc_bvh_gpu(): failed to reset pair partners");

            greedy_select_ploc_sorted_pairs_kernel<<<1, 1, 0, stream>>>(
                current_cluster_count,
                candidate_ids_out.get(),
                candidate_left.get(),
                candidate_right.get(),
                pair_role.get(),
                pair_partner.get(),
                node_used.get());
            check_cuda_last_error("build_edge_ploc_bvh_gpu(): failed to greedily select candidate pairs");

            mark_ploc_unmatched_runs_kernel<<<cluster_blocks, block_size, 0, stream>>>(
                current_cluster_count,
                pair_role.get(),
                run_start_flag.get(),
                run_end_flag.get());
            check_cuda_last_error("build_edge_ploc_bvh_gpu(): failed to mark unmatched runs");

            check_cuda_call(cub::DeviceScan::InclusiveSum(scan_temp.get(),
                                                          inclusive_scan_temp_size,
                                                          run_start_flag.get(),
                                                          run_prefix.get(),
                                                          current_cluster_count,
                                                          stream),
                            "build_edge_ploc_bvh_gpu(): failed to scan unmatched runs");

            scatter_ploc_run_bounds_kernel<<<cluster_blocks, block_size, 0, stream>>>(
                current_cluster_count,
                pair_role.get(),
                run_start_flag.get(),
                run_end_flag.get(),
                run_prefix.get(),
                run_start_indices.get(),
                run_end_indices.get());
            check_cuda_last_error("build_edge_ploc_bvh_gpu(): failed to scatter run bounds");

            check_cuda_call(cudaMemsetAsync(run_leave_first.get(),
                                            0,
                                            sizeof(int) * static_cast<size_t>(primitive_count),
                                            stream),
                            "build_edge_ploc_bvh_gpu(): failed to reset run pairing modes");

            compute_ploc_run_pairing_mode_from_starts_kernel<<<cluster_blocks, block_size, 0, stream>>>(
                current_cluster_count,
                current_bounds_in->get(),
                run_start_flag.get(),
                run_prefix.get(),
                run_end_indices.get(),
                run_leave_first.get());
            check_cuda_last_error("build_edge_ploc_bvh_gpu(): failed to evaluate unmatched run pairing");

            build_ploc_emission_plan_kernel<<<cluster_blocks, block_size, 0, stream>>>(
                current_cluster_count,
                pair_role.get(),
                pair_partner.get(),
                run_prefix.get(),
                run_start_indices.get(),
                run_end_indices.get(),
                run_leave_first.get(),
                emit_flag.get(),
                merge_flag.get(),
                emit_partner.get());
            check_cuda_last_error("build_edge_ploc_bvh_gpu(): failed to build emission plan");

            check_cuda_call(cub::DeviceScan::ExclusiveSum(scan_temp.get(),
                                                          exclusive_emit_temp_size,
                                                          emit_flag.get(),
                                                          emit_offsets.get(),
                                                          current_cluster_count,
                                                          stream),
                            "build_edge_ploc_bvh_gpu(): failed to scan emit flags");
            check_cuda_call(cub::DeviceScan::ExclusiveSum(scan_temp.get(),
                                                          exclusive_merge_temp_size,
                                                          merge_flag.get(),
                                                          merge_offsets.get(),
                                                          current_cluster_count,
                                                          stream),
                            "build_edge_ploc_bvh_gpu(): failed to scan merge flags");

            emit_ploc_next_clusters_kernel<<<cluster_blocks, block_size, 0, stream>>>(
                current_cluster_count,
                next_internal_node,
                current_nodes_in->get(),
                current_bounds_in->get(),
                emit_flag.get(),
                emit_offsets.get(),
                merge_flag.get(),
                merge_offsets.get(),
                emit_partner.get(),
                current_nodes_out->get(),
                current_bounds_out->get(),
                node_bbox_min_x,
                node_bbox_min_y,
                node_bbox_min_z,
                node_bbox_max_x,
                node_bbox_max_y,
                node_bbox_max_z,
                left_child,
                right_child,
                leaf_primitive,
                is_leaf,
                parent.get());
            check_cuda_last_error("build_edge_ploc_bvh_gpu(): failed to emit next cluster level");

            finalize_ploc_round_counts_kernel<<<1, 1, 0, stream>>>(
                current_cluster_count,
                run_prefix.get(),
                emit_flag.get(),
                emit_offsets.get(),
                merge_flag.get(),
                merge_offsets.get(),
                round_counts.get());
            check_cuda_last_error("build_edge_ploc_bvh_gpu(): failed to finalize round counts");
            check_cuda_call(cudaMemcpyAsync(host_round_counts,
                                            round_counts.get(),
                                            sizeof(host_round_counts),
                                            cudaMemcpyDeviceToHost,
                                            stream),
                            "build_edge_ploc_bvh_gpu(): failed to copy round counts");
            check_cuda_call(cudaStreamSynchronize(stream),
                            "build_edge_ploc_bvh_gpu(): failed to complete clustering round");

            const int next_cluster_count = host_round_counts[1];
            const int merge_count = host_round_counts[2];
            require_local(merge_count > 0,
                          "build_edge_ploc_bvh_gpu(): clustering round did not make progress.");
            require_local(next_cluster_count > 0 && next_cluster_count < current_cluster_count,
                          "build_edge_ploc_bvh_gpu(): invalid next cluster count.");

            current_cluster_count = next_cluster_count;
            next_internal_node -= merge_count;
            std::swap(current_nodes_in, current_nodes_out);
            std::swap(current_bounds_in, current_bounds_out);
        }

        require_local(next_internal_node == -1,
                      "build_edge_ploc_bvh_gpu(): internal-node allocator did not terminate at the root.");

        if (treelet_enabled) {
            check_cuda_call(cudaStreamSynchronize(stream),
                            "build_edge_ploc_bvh_gpu(): failed to finalize raw topology");
            host_left_child = copy_int_buffer_to_host(left_child,
                                                      static_cast<size_t>(node_count),
                                                      "build_edge_ploc_bvh_gpu(): failed to copy left_child");
            host_right_child = copy_int_buffer_to_host(right_child,
                                                       static_cast<size_t>(node_count),
                                                       "build_edge_ploc_bvh_gpu(): failed to copy right_child");
            host_is_leaf = copy_int_buffer_to_host(is_leaf,
                                                   static_cast<size_t>(node_count),
                                                   "build_edge_ploc_bvh_gpu(): failed to copy is_leaf");
            const HostTreeLevels host_levels = build_host_tree_levels_from_topology(host_left_child,
                                                                                    host_right_child,
                                                                                    host_is_leaf);
            node_heights = host_levels.node_heights;
            host_level_groups = host_levels.levels;
            max_height = host_levels.max_height;
            size_t scheduled_internal_count = 0;
            for (size_t level_index = 1; level_index < host_level_groups.size(); ++level_index) {
                scheduled_internal_count += host_level_groups[level_index].size();
            }
            require_local(static_cast<int>(scheduled_internal_count) == internal_count,
                          "build_edge_ploc_bvh_gpu(): failed to levelize every internal node.");
            internal_level_schedule = flatten_node_levels(host_level_groups);
            subtree_leaf_counts.assign(static_cast<size_t>(node_count), 1);
            for (int height = 1; height <= max_height; ++height) {
                for (int node_index : host_level_groups[static_cast<size_t>(height)]) {
                    subtree_leaf_counts[static_cast<size_t>(node_index)] =
                        subtree_leaf_counts[static_cast<size_t>(
                            host_left_child[static_cast<size_t>(node_index)])] +
                        subtree_leaf_counts[static_cast<size_t>(
                            host_right_child[static_cast<size_t>(node_index)])];
                }
            }

            std::vector<std::vector<int>> recompute_levels(static_cast<size_t>(max_height + 1));
            std::vector<std::vector<int>> optimize_levels(static_cast<size_t>(max_height + 1));
            for (int node_index = 0; node_index < internal_count; ++node_index) {
                const int height = node_heights[static_cast<size_t>(node_index)];
                if (subtree_leaf_counts[static_cast<size_t>(node_index)] >=
                    EdgeBVHTreeletMinSubtreeLeaves) {
                    optimize_levels[static_cast<size_t>(height)].push_back(node_index);
                } else {
                    recompute_levels[static_cast<size_t>(height)].push_back(node_index);
                }
            }
            const FlatNodeLevels recompute_schedule = flatten_node_levels(recompute_levels);
            const FlatNodeLevels optimize_schedule = flatten_node_levels(optimize_levels);
            CudaBuffer<int> recompute_nodes_device(recompute_schedule.nodes.size());
            CudaBuffer<int> optimize_nodes_device(optimize_schedule.nodes.size());
            if (treelet_schedule_mode == EdgeBVHTreeletScheduleMode::FlatLevels) {
                if (!recompute_schedule.nodes.empty()) {
                    check_cuda_call(cudaMemcpy(recompute_nodes_device.get(),
                                               recompute_schedule.nodes.data(),
                                               recompute_schedule.nodes.size() * sizeof(int),
                                               cudaMemcpyHostToDevice),
                                    "build_edge_ploc_bvh_gpu(): failed to upload recompute schedule");
                }
                if (!optimize_schedule.nodes.empty()) {
                    check_cuda_call(cudaMemcpy(optimize_nodes_device.get(),
                                               optimize_schedule.nodes.data(),
                                               optimize_schedule.nodes.size() * sizeof(int),
                                               cudaMemcpyHostToDevice),
                                    "build_edge_ploc_bvh_gpu(): failed to upload treelet schedule");
                }
            }

            const float scene_scale =
                fmaxf(scene_bounds.max.x - scene_bounds.min.x,
                      fmaxf(scene_bounds.max.y - scene_bounds.min.y,
                            scene_bounds.max.z - scene_bounds.min.z));
            const float inflation =
                fmaxf(scene_scale * EdgeBVHTreeletCostInflationRatio, 1e-6f);
            initialize_leaf_costs_kernel<<<primitive_blocks, block_size, 0, stream>>>(
                primitive_count,
                node_bbox_min_x,
                node_bbox_min_y,
                node_bbox_min_z,
                node_bbox_max_x,
                node_bbox_max_y,
                node_bbox_max_z,
                node_costs.get(),
                inflation);
            check_cuda_last_error(
                "build_edge_ploc_bvh_gpu(): failed to launch leaf-cost initialization");

            for (int height = 1; height <= max_height; ++height) {
                const int recompute_start =
                    recompute_schedule.level_offsets[static_cast<size_t>(height)];
                const int recompute_end =
                    recompute_schedule.level_offsets[static_cast<size_t>(height + 1)];
                const int recompute_count = recompute_end - recompute_start;
                if (recompute_count > 0) {
                    const int level_blocks = (recompute_count + block_size - 1) / block_size;
                    if (treelet_schedule_mode == EdgeBVHTreeletScheduleMode::FlatLevels) {
                        update_internal_nodes_kernel<<<level_blocks, block_size, 0, stream>>>(
                            recompute_count,
                            recompute_nodes_device.get() + recompute_start,
                            left_child,
                            right_child,
                            node_bbox_min_x,
                            node_bbox_min_y,
                            node_bbox_min_z,
                            node_bbox_max_x,
                            node_bbox_max_y,
                            node_bbox_max_z,
                            node_costs.get(),
                            inflation);
                    } else {
                        const std::vector<int> &recompute_nodes =
                            recompute_levels[static_cast<size_t>(height)];
                        CudaBuffer<int> device_nodes(recompute_nodes.size());
                        check_cuda_call(cudaMemcpy(device_nodes.get(),
                                                   recompute_nodes.data(),
                                                   recompute_nodes.size() * sizeof(int),
                                                   cudaMemcpyHostToDevice),
                                        "build_edge_ploc_bvh_gpu(): failed to upload recompute nodes");
                        update_internal_nodes_kernel<<<level_blocks, block_size, 0, stream>>>(
                            recompute_count,
                            device_nodes.get(),
                            left_child,
                            right_child,
                            node_bbox_min_x,
                            node_bbox_min_y,
                            node_bbox_min_z,
                            node_bbox_max_x,
                            node_bbox_max_y,
                            node_bbox_max_z,
                            node_costs.get(),
                            inflation);
                    }
                    check_cuda_last_error(
                        "build_edge_ploc_bvh_gpu(): failed to launch internal-node recompute");
                }

                const int optimize_start =
                    optimize_schedule.level_offsets[static_cast<size_t>(height)];
                const int optimize_end =
                    optimize_schedule.level_offsets[static_cast<size_t>(height + 1)];
                const int optimize_count = optimize_end - optimize_start;
                if (optimize_count > 0) {
                    const int level_blocks = (optimize_count + block_size - 1) / block_size;
                    if (treelet_schedule_mode == EdgeBVHTreeletScheduleMode::FlatLevels) {
                        optimize_selected_treelets_kernel<<<level_blocks, block_size, 0, stream>>>(
                            optimize_count,
                            optimize_nodes_device.get() + optimize_start,
                            is_leaf,
                            left_child,
                            right_child,
                            parent.get(),
                            node_bbox_min_x,
                            node_bbox_min_y,
                            node_bbox_min_z,
                            node_bbox_max_x,
                            node_bbox_max_y,
                            node_bbox_max_z,
                            leaf_primitive,
                            node_costs.get(),
                            inflation);
                    } else {
                        const std::vector<int> &optimize_nodes =
                            optimize_levels[static_cast<size_t>(height)];
                        CudaBuffer<int> device_nodes(optimize_nodes.size());
                        check_cuda_call(cudaMemcpy(device_nodes.get(),
                                                   optimize_nodes.data(),
                                                   optimize_nodes.size() * sizeof(int),
                                                   cudaMemcpyHostToDevice),
                                        "build_edge_ploc_bvh_gpu(): failed to upload treelet roots");
                        optimize_selected_treelets_kernel<<<level_blocks, block_size, 0, stream>>>(
                            optimize_count,
                            device_nodes.get(),
                            is_leaf,
                            left_child,
                            right_child,
                            parent.get(),
                            node_bbox_min_x,
                            node_bbox_min_y,
                            node_bbox_min_z,
                            node_bbox_max_x,
                            node_bbox_max_y,
                            node_bbox_max_z,
                            leaf_primitive,
                            node_costs.get(),
                            inflation);
                    }
                    check_cuda_last_error(
                        "build_edge_ploc_bvh_gpu(): failed to launch GPU treelet optimization");
                }
            }
        }

        check_cuda_call(cudaStreamSynchronize(stream),
                        "build_edge_ploc_bvh_gpu(): failed to complete build");
    } catch (const std::exception &e) {
        throw_runtime_error_local(std::string("build_edge_ploc_bvh_gpu(): ") + e.what());
    }
}
