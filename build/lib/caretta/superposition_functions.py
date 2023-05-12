import numba as nb
import numpy as np
from caretta import helper


@nb.njit
def paired_svd_superpose(coords_1: np.ndarray, coords_2: np.ndarray):
    """
    Superpose paired coordinates on each other using Kabsch superposition (SVD)

    Parameters
    ----------
    coords_1
        numpy array of coordinate data for the first protein; shape = (n, 3)
    coords_2
        numpy array of corresponding coordinate data for the second protein; shape = (n, 3)

    Returns
    -------
    rotation matrix, translation matrix for optimal superposition
    """
    centroid_1, centroid_2 = (
        helper.nb_mean_axis_0(coords_1),
        helper.nb_mean_axis_0(coords_2),
    )
    coords_1_c, coords_2_c = coords_1 - centroid_1, coords_2 - centroid_2
    correlation_matrix = np.dot(coords_2_c.T, coords_1_c)
    u, s, v = np.linalg.svd(correlation_matrix)
    reflect = np.linalg.det(u) * np.linalg.det(v) < 0
    if reflect:
        s[-1] = -s[-1]
        u[:, -1] = -u[:, -1]
    rotation_matrix = np.dot(u, v)
    translation_matrix = centroid_1 - np.dot(centroid_2, rotation_matrix)
    return rotation_matrix.astype(np.float64), translation_matrix.astype(np.float64)


@nb.njit
def paired_svd_superpose_with_subset(
        coords_1, coords_2, common_coords_1, common_coords_2
):
    """
    Superpose two sets of un-aligned coordinates using smaller subsets of aligned coordinates

    Parameters
    ----------
    coords_1
    coords_2
    common_coords_1
    common_coords_2

    Returns
    -------
    superposed coord_1, superposed coords_2, superposed common_coords_2
    """
    rot, tran = paired_svd_superpose(common_coords_1, common_coords_2)
    coords_1 = coords_1 - helper.nb_mean_axis_0(common_coords_1)
    coords_2 = np.dot(coords_2 - helper.nb_mean_axis_0(common_coords_2), rot)
    common_coords_2_rot = apply_rotran(common_coords_2, rot, tran)
    return coords_1, coords_2, common_coords_2_rot


@nb.njit
def apply_rotran(
        coords: np.ndarray, rotation_matrix: np.ndarray, translation_matrix: np.ndarray
) -> np.ndarray:
    """
    Applies a rotation and translation matrix onto coordinates

    Parameters
    ----------
    coords
    rotation_matrix
    translation_matrix

    Returns
    -------
    transformed coordinates
    """
    return np.dot(coords, rotation_matrix) + translation_matrix
