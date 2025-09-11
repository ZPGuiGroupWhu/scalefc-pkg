from typing import TypeAlias, Union
import numpy as np
from sklearn.metrics import pairwise_distances
import logging
from datetime import datetime


# Shape: N x 4, means OX,OY, DX,DY
ODArray: TypeAlias = np.ndarray[np.float32]

Label: TypeAlias = np.ndarray[np.int32]


def flow_attribute_number(OD_any: ODArray) -> int:
    """get the number of OD flow.
    """
    if np.ndim(OD_any) == 1:
        return 1
    elif np.ndim(OD_any) == 2:
        return np.shape(OD_any)[0]
    else:
        raise ValueError(f"Invalid OD flow data, data's shape: {np.shape(OD_any)}; expected 2-dimension array.")
    

def flow_attribute_length(OD: ODArray) -> np.ndarray | float:
    """calaulate flow length."""
    newOD = np.reshape(OD, (-1, 4))
    ox, oy, dx, dy = newOD[:, 0], newOD[:, 1], newOD[:, 2], newOD[:, 3]
    res = np.sqrt(np.square(ox - dx) + np.square(oy - dy))
    if np.ndim(OD) == 1:
        return res[0]
    else:
        return res
    
def flow_attribute_centroid_OD(OD: ODArray) -> np.ndarray:
    """get flow centroid OD
    """
    if np.ndim(OD) == 1:
        return np.asarray(OD)
    else:
        return np.mean(OD, axis=0)


def flow_distance_other_flow_matrix_max_euclidean(OD1: ODArray, OD2: ODArray) -> np.ndarray:
    OD1 = np.reshape(OD1, (-1, 4))
    n_jobs = -1

    op = OD1[:, 0:2]
    dp = OD1[:, 2:4]
    if OD2 is None:
        op_dis = pairwise_distances(op, n_jobs=n_jobs)
        dp_dis = pairwise_distances(dp, n_jobs=n_jobs)
    else:
        OD2 = np.reshape(OD2, (-1, 4))
        op2 = OD2[:, 0:2]
        dp2 = OD2[:, 2:4]
        op_dis = pairwise_distances(op, op2, n_jobs=n_jobs)
        dp_dis = pairwise_distances(dp, dp2, n_jobs=n_jobs)
    return np.maximum(op_dis, dp_dis)


def log_message(flag: bool, message: str) -> None:
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if flag is True:
        print(f"{now} - \033[32mDEBUG\033[0m - {message}")


def flow_indicator_rmse_OD_distance(OD: np.ndarray) -> float:
    """calculate flow group compactness indicator
    """
    centroid_flow = flow_attribute_centroid_OD(OD)

    number = flow_attribute_number(OD)
    max_dist = flow_distance_other_flow_matrix_max_euclidean(centroid_flow, OD)
    res = np.sqrt(np.sum(np.square(max_dist)) / number)
    return res


def flow_attribute_OD2Vec(OD: ODArray) -> np.ndarray:
    """convert flow to vector, shape: N x 4 -> N x 2."""
    if np.ndim(OD) == 1:
        ox, oy, dx, dy = OD[0], OD[1], OD[2], OD[3]
        return np.array([dx - ox, dy - oy])
    else:
        ox, oy, dx, dy = OD[:, 0], OD[:, 1], OD[:, 2], OD[:, 3]
    res = np.vstack((dx - ox, dy - oy)).T
    return res

    
def flow_neighbor_knn_indices(OD: np.ndarray, k: int):
    """
    Find KNN neighbor indices for OD flows using custom distance metric.
    """
    n_flows = OD.shape[0]
    
    if k >= n_flows:
        raise ValueError(f"k ({k}) must be less than the number of flows ({n_flows})")
    
    # Extract origin and destination coordinates
    origins = OD[:, :2]  # (N, 2)
    destinations = OD[:, 2:]  # (N, 2)
    
    knn_indices = np.zeros((n_flows, k), dtype=int)
    
    for i in range(n_flows):
        # Vectorized distance computation
        # Origin distances
        o_diffs = origins - origins[i]  # (N, 2)
        o_distances = np.sqrt(np.sum(o_diffs**2, axis=1))  # (N,)
        
        # Destination distances
        d_diffs = destinations - destinations[i]  # (N, 2)
        d_distances = np.sqrt(np.sum(d_diffs**2, axis=1))  # (N,)
        
        # Combined distance (max of origin and destination distances)
        combined_distances = np.maximum(o_distances, d_distances)
        
        # Exclude self by setting distance to infinity
        combined_distances[i] = np.inf
        
        # Find k nearest neighbors
        nearest_indices = np.argpartition(combined_distances, k)[:k]
        # Sort by distance
        nearest_indices = nearest_indices[np.argsort(combined_distances[nearest_indices])]
        
        knn_indices[i] = nearest_indices
    
    return knn_indices