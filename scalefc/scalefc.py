# -*- coding: utf-8 -*-
# Author: chenhuan
# Email: chen_huan@whu.edu.cn
# Date: 2025-08-21
# Description: ScaleFC algorithm for heterogeneous geographical flow.
# The paper link: https://doi.org/10.1016/j.compenvurbsys.2025.102338

from typing import Callable, Literal, Optional, Union, List

import numpy as np
from joblib import Parallel, delayed
from scipy.spatial import KDTree
from collections import deque

import time

from .util import (
    ODArray,
    Label,
    flow_attribute_number,
    flow_attribute_length,
    flow_attribute_centroid_OD,
    flow_distance_other_flow_matrix_max_euclidean,
    flow_indicator_rmse_OD_distance,
    flow_neighbor_knn_indices,
    log_message,
)

__all__ = ["flow_cluster_scalefc"]


def _is_strongly_connected_flow_group(OD: np.ndarray, **kwargs) -> bool:
    """Return True when flow group is strongly connected."""
    fixed_eps = kwargs.get("fixed_eps")
    scale_factor = kwargs.get("scale_factor")
    scale_factor_func = kwargs.get("scale_factor_func")

    if fixed_eps is None:
        cf = flow_attribute_centroid_OD(OD)
        eps = scale_factor_func(cf, scale_factor)
    else:
        eps = fixed_eps
    return flow_indicator_rmse_OD_distance(OD) <= eps


# 是否舍弃流簇, true则舍弃，false则保留
# k是条数
def _can_discard_flow_group(OD, **kwargs):
    k = kwargs["min_flows"]
    return flow_attribute_number(OD) < k + 1


def _scale_factor_func_linear(OD: np.ndarray, scale_factor: float) -> float:
    """
    The linear scale factor function for a given OD flow and scale factor, which is used to calculate the
    neighborhood search range of each flow. Here, r = 0.5 * α * l
    """
    assert scale_factor > 0 and scale_factor <= 1, f"Invalid scale_factor: {scale_factor}"
    return 0.5 * scale_factor * flow_attribute_length(OD)


def _scale_factor_func_sqrt(OD: np.ndarray, scale_factor: float) -> float:
    """
    The non-linear scale factor function for a given OD flow and scale factor, which is used to calculate the
    neighborhood search range of each flow. Here, r = 0.5 * (α^0.5) * l
    """
    assert scale_factor > 0 and scale_factor <= 1, f"Invalid scale_factor: {scale_factor}"
    return 0.5 * np.sqrt(scale_factor) * flow_attribute_length(OD)


def _scale_factor_func_square(OD: np.ndarray, scale_factor: float) -> float:
    """
    The non-linear scale factor function for a given OD flow and scale factor, which is used to calculate the
    neighborhood search range of each flow. Here, r = 0.5 * (α^2) * l
    """
    assert scale_factor > 0 and scale_factor <= 1, f"Invalid scale_factor: {scale_factor}"
    return 0.5 * np.square(scale_factor) * flow_attribute_length(OD)


def _scale_factor_func_tanh(OD: np.ndarray, scale_factor: float) -> float:
    """
    The non-linear scale factor function for a given OD flow and scale factor. r = 0.5(1-e^(-α))/(1+e^(-α)) * l
    """
    assert scale_factor > 0, f"Invalid scale_factor: {scale_factor}"
    return 0.5 * flow_attribute_length(OD) * (1 - np.exp(-scale_factor)) / (1 + np.exp(-scale_factor))


def _check_scale_factor_func(
    scale_factor_func: Union[
        Literal["linear", "square", "sqrt", "tanh"],
        Callable[[np.ndarray, float], np.ndarray],
    ],
) -> Callable[[np.ndarray, float], np.ndarray]:
    if isinstance(scale_factor_func, Callable):
        return scale_factor_func

    elif isinstance(scale_factor_func, str):
        assert scale_factor_func in [
            "linear",
            "square",
            "sqrt",
            "tanh",
        ], f"Invalid scale_factor_func: {scale_factor_func}. Options: linear, square, sqrt, tanh, Please use one of them."
        if scale_factor_func == "linear":
            return _scale_factor_func_linear
        elif scale_factor_func == "square":
            return _scale_factor_func_square
        elif scale_factor_func == "sqrt":
            return _scale_factor_func_sqrt
        elif scale_factor_func == "tanh":
            return _scale_factor_func_tanh

    else:
        raise RuntimeError(
            "Invalid scale_factor_func: {scale_factor_func}, must be callable or str, and str options: linear, square, sqrt, tanh."
        )


def _rearrange_flow_group_indices(OD: np.ndarray) -> List[int]:
    """Optimized rearrange the flow in OD matrix, and return a new OD matrix's indices.

    Args:
        OD (np.ndarray): OD matrix has 4 columns:  ox, oy, dx, dy.

    Returns:
        List[int]: rearrange indices
    """
    if np.ndim(OD) == 1:
        return [0]

    # Use ptp for efficient range calculation
    ranges = np.ptp(OD, axis=0)

    # Vectorized max operations
    o_max = np.max(ranges[:2])
    d_max = np.max(ranges[2:])

    # Conditional copy and swap - only when needed
    if o_max > d_max:
        # Use advanced indexing for efficient column swapping
        OD_work = OD[:, [2, 3, 0, 1]]
    else:
        OD_work = OD

    # 1. get centroid flow - inline calculation to avoid function call overhead
    cf = np.mean(OD_work, axis=0)

    # 2. calculate centroid flow angle - inline to avoid function overhead
    cf_vec = cf[2:] - cf[:2]
    cf_ang = np.arctan2(cf_vec[1], cf_vec[0])
    cf_ang = np.mod(cf_ang, 2 * np.pi)

    # 3. assign new O point and calculate angles in one operation
    # Create modified OD matrix with new origin points
    modified_OD = OD_work.copy()
    modified_OD[:, :2] = cf[:2]

    # Calculate vectors and angles vectorized
    vecs = modified_OD[:, 2:] - modified_OD[:, :2]
    ang = np.arctan2(vecs[:, 1], vecs[:, 0])
    ang = np.mod(ang, 2 * np.pi)

    # 4. get reverse flow angle
    cf_rev_ang = np.mod(cf_ang + np.pi, 2 * np.pi)

    # 5. calculate the sequence - use concatenate for better performance
    all_ang = np.concatenate([ang, [cf_rev_ang]])
    res = np.argsort(all_ang)

    # Find cf_index using numpy operations - more efficient than list.index()
    cf_index = np.where(res == len(all_ang) - 1)[0][0]

    # Use numpy slicing and concatenation, then convert to list
    result = np.concatenate([res[cf_index + 1 :], res[:cf_index]])

    return result.tolist()


# 计算每条流的局部密度
def _calculate_flow_group_local_density(OD: np.ndarray, k: int, n_jobs: Optional[int] = None) -> list[float]:
    knn_idx = flow_neighbor_knn_indices(OD, k)
    # add the flow self-index
    knn_idx = np.column_stack([range(len(OD)), knn_idx])
    
    if not n_jobs:
        local_densities = []
        for i in range(flow_attribute_number(OD)):
            # get cluster
            sub_od = OD[knn_idx[i], :]
            # calculate
            cur_indicator = flow_indicator_rmse_OD_distance(sub_od)
            local_densities.append(cur_indicator)

        return local_densities
    else:

        def process_batch(batch_indices):
            batch_results = []
            for i in batch_indices:
                sub_od = OD[knn_idx[i], :]
                cur_indicator = flow_indicator_rmse_OD_distance(sub_od)
                batch_results.append(cur_indicator)
            return batch_results

        num_flows = flow_attribute_number(OD)
        indices = np.arange(num_flows)
        batch_size = 1
        batches = [indices[i : i + batch_size] for i in range(0, num_flows, batch_size)]

        results = Parallel(n_jobs=n_jobs, backend="threading")(delayed(process_batch)(batch) for batch in batches)

        # Flatten the list of results
        local_densities = [item for sublist in results for item in sublist]

        return local_densities


def _calculate_flow_group_local_density_gradient(local_denisty: list) -> list[float]:
    local_denisty = np.asarray(local_denisty)
    if len(local_denisty) < 2:
        return []

    derivative = np.abs((local_denisty[1:] - local_denisty[:-1]) / local_denisty[:-1])

    return derivative


def _find_flow_group_partitioningflow_arg(rearrange_arg: list, local_density: list) -> int:
    if len(local_density) < 3:
        pf_idx = rearrange_arg[0]
    else:
        local_density = np.take(local_density, rearrange_arg)
        derivative = _calculate_flow_group_local_density_gradient(local_density)
        if len(derivative) + 1 == len(local_density):
            pf_idx = rearrange_arg[np.argmax(derivative) + 1]
        elif len(derivative) == len(local_density):
            pf_idx = rearrange_arg[np.argmax(derivative)]
        else:
            raise RuntimeError(
                f"Check your _calculate_flow_group_local_density_gradient len(local_density): {len(local_density)}, len(local_density_gradient): {len(derivative)}"
            )
    return pf_idx


def _get_left_right_flow_sub_group_indices_by_pf(rearrange_arg, pf_idx: int) -> tuple[list[int], list[int]]:
    assert pf_idx in rearrange_arg
    if not isinstance(rearrange_arg, list):
        rearrange_arg = list(rearrange_arg)
    idx = rearrange_arg.index(pf_idx)
    return rearrange_arg[:idx], rearrange_arg[idx + 1 :]


def _partition_flow_group_and_return_arg(
    OD: np.ndarray,
    k: int,
    n_jobs: Optional[int] = None,
) -> tuple[list[int], list[int], int]:
    new_idx = _rearrange_flow_group_indices(OD)
    local_density = _calculate_flow_group_local_density(OD, k, n_jobs=n_jobs)
    if 0 in local_density:
        local_density = [x + 1 for x in local_density]
    pf_idx = _find_flow_group_partitioningflow_arg(new_idx, local_density)
    l, r = _get_left_right_flow_sub_group_indices_by_pf(new_idx, pf_idx)
    return l, r, pf_idx


# Optimized function
def _get_spatially_connected_flow_groups(OD: ODArray, min_flows: int, eps: Union[float, list, tuple, np.ndarray]) -> Label:
    """get spatially connected flow groups function.

    Args:
        OD (ODArray): OD matrix with 4 columns: ox, oy, dx, dy
        min_flows (int): Minimum number of flows to form a group
        eps (Union[float, list, tuple, np.ndarray]): Distance threshold(s)

    Returns:
        Label: Array of cluster labels
    """
    n = len(OD)

    # Handle eps parameter efficiently
    if not hasattr(eps, "__len__"):
        eps_array = eps  # Keep as scalar if possible
        use_scalar_eps = True
    else:
        eps_array = np.asarray(eps, dtype=np.float64)
        use_scalar_eps = False

    # Extract coordinates
    O = OD[:, :2]  # Origin coordinates (ox, oy)
    D = OD[:, 2:]  # Destination coordinates (dx, dy)

    # Build KDTrees with optimized leaf size for better performance
    tree_O = KDTree(O, leafsize=16)  # Smaller leaf size for better performance
    tree_D = KDTree(D, leafsize=16)

    # Initialize labels array
    labels = np.full(n, -2, dtype=np.int32)
    cluster_id = 0

    # Pre-allocate queue for better memory management
    # Use dynamic queue that can grow as needed
    queue = []

    # Inline neighbor query function for better performance
    def region_query_fast(point_idx: int, radius: float):
        """Fast neighbor query with optimized intersection."""
        # Get neighbors from both trees
        indices_O = tree_O.query_ball_point(O[point_idx], radius)
        indices_D = tree_D.query_ball_point(D[point_idx], radius)

        # Fast intersection using sorted arrays
        if len(indices_O) < len(indices_D):
            # Convert smaller list to set for faster lookup
            set_O = set(indices_O)
            return [idx for idx in indices_D if idx in set_O]
        else:
            set_D = set(indices_D)
            return [idx for idx in indices_O if idx in set_D]

    # Main clustering loop
    for i in range(n):
        if labels[i] != -2:  # Skip already visited points
            continue

        # Get current eps value
        current_eps = eps_array if use_scalar_eps else eps_array[i]

        # Query neighbors for current point
        neighbors = region_query_fast(i, current_eps)

        if len(neighbors) < min_flows:
            labels[i] = -1  # Mark as noise
        else:
            # Start new cluster
            labels[i] = cluster_id

            # Initialize queue with neighbors (excluding self)
            queue.clear()  # Clear queue for reuse
            for j in neighbors:
                if j != i:
                    if labels[j] == -2:  # Unvisited point
                        labels[j] = cluster_id
                        queue.append(j)
                    elif labels[j] == -1:  # Reclassify noise point
                        labels[j] = cluster_id

            # BFS expansion
            queue_idx = 0
            while queue_idx < len(queue):
                j = queue[queue_idx]
                queue_idx += 1

                # Get eps for current point
                j_eps = eps_array if use_scalar_eps else eps_array[j]
                j_neighbors = region_query_fast(j, j_eps)

                if len(j_neighbors) >= min_flows:  # j is core point
                    for k in j_neighbors:
                        if k != j:  # Skip self
                            if labels[k] == -2:  # Unvisited point
                                labels[k] = cluster_id
                                queue.append(k)
                            elif labels[k] == -1:  # Reclassify noise point
                                labels[k] = cluster_id

            cluster_id += 1  # Increment cluster ID

    return labels


def flow_cluster_scalefc(
    OD: ODArray,
    scale_factor: float | None = 0.1,
    min_flows: int = 5,
    scale_factor_func: Union[
        Literal["linear", "square", "sqrt", "tanh"],
        Callable[[np.ndarray, float], np.ndarray],
    ] = "linear",
    fixed_eps: float | None = None,
    n_jobs: int | None = None,
    debug: bool | Literal["simple", "full"] = False,
    show_time_usage: bool = False,
    **kwargs,
) -> Label:
    """Performs ScaleFC algorithm.

        This function implements the flow clustering algorithm proposed in the paper:
        'ScaleFC: A scale-aware geographical flow clustering algorithm for heterogeneous origin-destination data'.
        The algorithm operates by:
    
            Step 1: Identifying flow groups via spatial connectivity
            Step 2: Recognizing strongly-connected flow groups
            Step 3: Handling weakly-connected flow groups
            Step 4: Reallocating partitioning flows and outputting cluster results

        The ScaleFC algorithm is particularly effective for clustering flow data where flows
        exhibit uneven length, heterogeneous density, and weak connectivity. The details of the algorithm can be found in the paper.

        Args:
            OD (ODArray): Origin-Destination flow matrix as a numpy array with shape (N, 4)
                containing [origin_x, origin_y, destination_x, destination_y] coordinates.
                Type alias for np.ndarray[np.float32].
            scale_factor (float | None, optional): Scale factor in range (0, 1] used to
                calculate the neighborhood range (epsilon) for each flow.
                Smaller values create tighter clusters. Defaults to 0.1.
            min_flows (int, optional): Minimum number of flows required to form a valid
                cluster. Flows in groups smaller than this threshold are considered noise.
                Must be a positive integer. Defaults to 5.
            scale_factor_func (Union[Literal["linear", "square", "sqrt", "tanh"],
                Callable[[np.ndarray, float], np.ndarray]], optional): Function or string
                identifier specifying how to calculate epsilon from the scale factor.
                Custom callable should accept (flow_data, scale_factor) and return epsilon.
                Defaults to "linear".
            fixed_eps (float | None, optional): Fixed epsilon value for neighborhood queries.
                When provided, overrides the scale_factor-based calculation. Must be positive.
                Defaults to None.
            n_jobs (int | None, optional): Number of parallel jobs for computation.
                - None: Sequential execution
                - -1: Use all available CPU cores
                - Positive integer: Use specified number of cores
                Defaults to None.
            debug (bool | Literal["simple", "full"], optional): Whether to print detailed debug information during
                algorithm execution, including intermediate clustering results and timing.
                Defaults to False.
            show_time_usage (bool, optional): Whether to show time usage of each step.
                Defaults to False.
            **kwargs: Additional keyword arguments for advanced customization:
                - spatially_connected_flow_groups_label (np.ndarray, optional):
                  Pre-computed cluster labels for spatially connected groups. Must have
                  same length as OD array.
                - is_strongly_connected_flow_group_func (Callable, optional):
                  Custom function to determine if a flow group is strongly connected.
                  Should accept (OD_subset, **params) and return bool.
                - can_discard_flow_group_func (Callable, optional):
                  Custom function to determine if a flow group should be discarded.
                  Should accept (OD_subset, **params) and return bool.

        Returns:
            Label: Cluster labels as numpy array of integers with shape (N,) where N is
                the number of input flows. Type alias for np.ndarray[np.int32].
                - Non-negative integers: Cluster IDs (0, 1, 2, ...)
                - -1: Noise flows (flows not belonging to any cluster)

        Raises:
            AssertionError: If input validation fails:
                - OD is not a 2D array with exactly 4 columns
                - min_flows is not a positive integer
                - fixed_eps is not positive when provided
                - n_jobs is less than -1
            ValueError: If invalid keyword arguments are provided or if
                spatially_connected_flow_groups_label has incorrect length.
    """
    assert np.ndim(OD) == 2 and OD.shape[1] == 4, f"OD matrix must be a 2D array with 4 columns, got {OD.shape}"
    assert min_flows > 0 and isinstance(min_flows, int), f"min_flows must be a positive integer, got {min_flows}"
    assert fixed_eps is None or fixed_eps > 0, f"fixed_eps must be a positive number, got {fixed_eps}"
    assert n_jobs is None or n_jobs > -2, f"n_jobs must be a positive number, got {n_jobs}"
    kwargs_valid = [
        "spatially_connected_flow_groups_label",
        "is_strongly_connected_flow_group_func",
        "can_discard_flow_group_func",
    ]
    for keys in kwargs.keys():
        if keys not in kwargs_valid:
            raise ValueError(f"Invalid kwargs: {keys}")
    debug_full = False
    debug_simple = False
    if isinstance(debug, str):
        if debug == "simple":
            debug_simple = True
            debug_full = False
        elif debug == "full":
            debug_simple = True
            debug_full = True
        else:
            raise ValueError(f"Invalid debug: {debug}")
    elif isinstance(debug, bool):
        debug_simple = debug
    else:
        raise ValueError(f"Invalid debug: {debug}")
    del debug  # avoid debug being used in the following code

    OD_len = flow_attribute_number(OD)

    # if it is given, use it directly and check its length
    spatially_connected_flow_groups_label = kwargs.get("spatially_connected_flow_groups_label", None)

    scale_factor_func = _check_scale_factor_func(scale_factor_func)
    is_strongly_connected_flow_group_func = kwargs.get(
        "is_strongly_connected_flow_group_func", _is_strongly_connected_flow_group
    )
    can_discard_flow_group_func = kwargs.get("can_discard_flow_group_func", _can_discard_flow_group)
    # get algorithm parameters for is_strongly_connected_flow_group_func and can_discard_flow_group_func
    algorithm_params = dict(
        scale_factor=scale_factor,
        min_flows=min_flows,
        scale_factor_func=scale_factor_func,
        fixed_eps=fixed_eps,
    )
    # process indices all the way
    # result, like: [[1, 3], [2, 5]], that means [1, 3] and [2, 5] are two clusetr respectively, [0, 4] are noises
    result_subclusters_indices = []
    all_pf_indices = []
    four_steps_time_usage = [0, 0, 0, 0]

    log_message(
        debug_simple, f"Start ScaleFC algorithm on {OD_len:_} flows, scale factor: {scale_factor}, min flows: {min_flows}."
    )

    if spatially_connected_flow_groups_label is not None:
        assert len(spatially_connected_flow_groups_label) == OD_len, (
            f"Invalid spatially_connected_flow_groups_label, length: {len(spatially_connected_flow_groups_label)}; expected: {OD_len}"
        )

        log_message(debug_simple, "The label of spatially connected flow groups is given, skip the first step.")

    else:
        time_11 = time.time()
        if fixed_eps is None:
            spatially_connected_flow_groups_label = _get_spatially_connected_flow_groups(
                OD, min_flows=min_flows, eps=scale_factor_func(OD, scale_factor)
            )

        else:
            spatially_connected_flow_groups_label = _get_spatially_connected_flow_groups(OD, min_flows=min_flows, eps=fixed_eps)
            log_message(debug_simple, f"Use fixed eps to idenfity spatially connected flow groups instead of scale factor.")
        time_12 = time.time()
        four_steps_time_usage[0] = time_12 - time_11
        log_message(
            show_time_usage,
            f"[TIME] Time usage of `Step 1: Identifying flow groups via spatial connectivity' is {time_12 - time_11:.3f}s",
        )

    _labels, _counts = np.unique(spatially_connected_flow_groups_label, return_counts=True)

    waited_subgroups_indices_queue = deque()
    for label in _labels:
        if label < 0:  # noise
            continue
        current_indices = np.where(spatially_connected_flow_groups_label == label)[0]
        waited_subgroups_indices_queue.append(current_indices)

    log_message(debug_simple, f"Initially, there are {len(_labels)} spatially-connected flow groups.")
    log_message(debug_full, f"The spatially_connected_flow_groups_label, label: {_labels}, count: {_counts}")


    if n_jobs is None or n_jobs == 1 or n_jobs == 0:
        if n_jobs == 0: n_jobs = None
        parallel_backend = "sequential"
        log_message(debug_simple, "Process flow groups sequentially.")
    else:
        parallel_backend = "loky"
        log_message(debug_simple, "Process flow groups in parallel.")
        
    def process_subgroup(current_indices):
        current_flow_group = OD[current_indices, :]

        time_21 = time.time()
        if can_discard_flow_group_func(current_flow_group, **algorithm_params):
            log_message(debug_full, f"Discard current flow group - indices: {current_indices}")
            
            time_22 = time.time()
            return None, None, time_22 - time_21, 0

        if is_strongly_connected_flow_group_func(current_flow_group, **algorithm_params):
            log_message(debug_full, f"Save current strongly-connected flow group - indices: {current_indices}")
            time_22 = time.time()
            return current_indices, None, time_22 - time_21, 0
        
        time_22 = time.time()
        
        time_31 = time_22
        # find pf and partition current flow group
        l, r, pf_idx = _partition_flow_group_and_return_arg(current_flow_group, min_flows, -1)
        time_32 = time.time()

        # if False:
        #     curlist = [pf_idx] + l + r
        #     curlist.sort()
        #     assert np.array_equal(curlist, range(len(current_indices))), "ops!"
        left_indices = current_indices[np.asarray(l, dtype=np.int32)] if l else None
        right_indices = current_indices[np.asarray(r, dtype=np.int32)] if r else None
        pf_index = current_indices[pf_idx] if pf_idx else None

        log_message(debug_full, f"Find partitioning flow's index of current OD: {pf_index}")
        log_message(debug_full, f"There are {len(l)} flows in left flow sub-group.")
        log_message(debug_full, f"There are {len(r)} flows in right flow sub-group.")

        return None, (left_indices, right_indices, pf_index), time_22 - time_21, time_32 - time_31

    

    # from dask.distributed import LocalCluster
    # dask_cluster = LocalCluster(n_workers=6, threads_per_worker=2, processes=True)
    # client = dask_cluster.get_client()
    with Parallel(n_jobs=n_jobs, ackend=parallel_backend) as parallel:
        while waited_subgroups_indices_queue:

            results = parallel(
                delayed(process_subgroup)(waited_subgroups_indices_queue.popleft())
                for _ in range(len(waited_subgroups_indices_queue))
            )
            
            max_s2 = 0
            max_s3 = 0
            
            for result in results:
                if result[0] is not None:
                    result_subclusters_indices.append(result[0])
                if result[1] is not None:
                    left_indices, right_indices, pf_index = result[1]
                    if left_indices is not None:
                        waited_subgroups_indices_queue.append(left_indices)
                    if right_indices is not None:
                        waited_subgroups_indices_queue.append(right_indices)
                    if pf_index is not None:
                        all_pf_indices.append(pf_index)
                
                max_s2 = max(max_s2, result[2])
                max_s3 = max(max_s3, result[3])
                
            # 
            four_steps_time_usage[1] += max_s2 # Step 2
            four_steps_time_usage[2] += max_s3 # Step 3
        # client.close()
        # dask_cluster.close()

    log_message(show_time_usage, f"[TIME] Time usage of `Step 2: Recognizing strongly-connected flow groups' is {four_steps_time_usage[1]:.3f}s")
    log_message(show_time_usage, f"[TIME] Time usage of `Step 3: Handling weakly-connected flow groups' is {four_steps_time_usage[2]:.3f}s")

    labels = np.full(OD_len, fill_value=-1, dtype=int)
    if not result_subclusters_indices:
        log_message(debug_simple, "All flows are identified as noise.")

        total_t = sum(four_steps_time_usage)
        log_message(show_time_usage, f"[TIME] Total time usage of the ScaleFC algorithm is {total_t:.3f}s, {total_t / 60:.3f}min")
        return labels

    for i, x in enumerate(result_subclusters_indices):
        labels[x] = i

    if not all_pf_indices:
        log_message(debug_simple, "There are no partitioning flows.")
        
        total_t = sum(four_steps_time_usage)
        log_message(show_time_usage, f"[TIME] Total time usage of the ScaleFC algorithm is {total_t:.3f}s, {total_t / 60:.3f}min")
        return labels

    log_message(debug_simple, f"There are {len(all_pf_indices)} partitioning flows.")

    
    # process pd_index here
    time_41 = time.time()
    all_pf_indices = np.asarray(all_pf_indices, dtype=np.int32)
    # find cf for each group and calculate cf and pf's distance
    groups_centroid_flow = []
    groups_eps_threshold = []
    groups_current_rmse = []

    for x in result_subclusters_indices:
        group_data = OD[x, :]
        centroid = flow_attribute_centroid_OD(group_data)
        groups_centroid_flow.append(centroid)


        if fixed_eps is None:
            eps = scale_factor_func(centroid, scale_factor)
        else:
            eps = fixed_eps
        groups_eps_threshold.append(eps)

        current_rmse = flow_indicator_rmse_OD_distance(group_data)
        groups_current_rmse.append(current_rmse)

    all_pf = OD[all_pf_indices, :]
    groups_centroid_flow = np.array(groups_centroid_flow)
    groups_eps_threshold = np.array(groups_eps_threshold)
    groups_current_rmse = np.array(groups_current_rmse)

    pf_cf_dis = flow_distance_other_flow_matrix_max_euclidean(all_pf, groups_centroid_flow)

    pf_cf_dis_min_arg = np.argmin(pf_cf_dis, axis=1)
    assert len(all_pf_indices) == len(pf_cf_dis_min_arg), "ops!"

    pf_to_nearest_centroid_dist = pf_cf_dis[np.arange(len(all_pf)), pf_cf_dis_min_arg]

    for i, x in enumerate(pf_cf_dis_min_arg):

        if pf_to_nearest_centroid_dist[i] > groups_eps_threshold[x]:
            continue
        cur_group = OD[result_subclusters_indices[x], :]
        group_size = len(cur_group)

        pf_to_centroid_dist = pf_to_nearest_centroid_dist[i]
        new_rmse_squared = (group_size * groups_current_rmse[x] ** 2 + pf_to_centroid_dist**2) / (group_size + 1)
        new_rmse = np.sqrt(new_rmse_squared)

        if new_rmse <= groups_eps_threshold[x]:
            labels[all_pf_indices[i]] = x

    label_indices = np.unique(labels)
    label_indices = label_indices[label_indices != -1]
    log_message(debug_simple, f"There are {len(label_indices)} clusters in total.")

    time_42 = time.time()
    four_steps_time_usage[3] += time_42 - time_41
    total_t = sum(four_steps_time_usage)
    log_message(show_time_usage, f"[TIME] Time usage of `Step 4: Reallocating PFs and outputting cluster results' is {four_steps_time_usage[3]:.3f}s")
    log_message(show_time_usage, f"[TIME] Total time usage of the ScaleFC algorithm is {total_t:.3f}s, {total_t / 60:.3f}min")

    return labels
