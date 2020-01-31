import numba as nb
import numpy as np


@nb.njit
# @numba_cc.export('normalize', 'f64[:](f64[:])')
def normalize(numbers):
    minv, maxv = np.min(numbers), np.max(numbers)
    return (numbers - minv) / (maxv - minv)


@nb.njit
# @numba_cc.export('nb_mean_axis_0', 'f64[:](f64[:])')
def nb_mean_axis_0(array: np.ndarray) -> np.ndarray:
    """
    Same as np.mean(array, axis=0) but njitted
    """
    mean_array = np.zeros(array.shape[1])
    for i in range(array.shape[1]):
        mean_array[i] = np.mean(array[:, i])
    return mean_array


@nb.njit
# @numba_cc.export('svd_superimpose', '(f64[:], f64[:])')
def svd_superimpose(coords_1: np.ndarray, coords_2: np.ndarray):
    """
    Superimpose paired coordinates on each other using svd

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
    centroid_1, centroid_2 = nb_mean_axis_0(coords_1), nb_mean_axis_0(coords_2)
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
# @numba_cc.export('apply_rotran', '(f64[:], f64[:], f64[:])')
def apply_rotran(coords: np.ndarray, rotation_matrix: np.ndarray, translation_matrix: np.ndarray) -> np.ndarray:
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


# @numba_cc.export('superpose_with_pos', '(f64[:], f64[:], f64[:], f64[:])')
@nb.njit
def superpose_with_pos(coords_1, coords_2, common_coords_1, common_coords_2):
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
    rot, tran = svd_superimpose(common_coords_1, common_coords_2)
    coords_1 = coords_1 - nb_mean_axis_0(common_coords_1)
    coords_2 = np.dot(coords_2 - nb_mean_axis_0(common_coords_2), rot)
    common_coords_2_rot = apply_rotran(common_coords_2, rot, tran)
    return coords_1, coords_2, common_coords_2_rot


@nb.njit
# @numba_cc.export('make_distance_matrix', '(f64[:], f64[:], f64, b1)')
def make_distance_matrix(coords_1: np.ndarray, coords_2: np.ndarray, gamma, normalized=False) -> np.ndarray:
    """
    Makes matrix of euclidean distances of each coordinate in coords_1 to each coordinate in coords_2
    TODO: probably faster to do upper triangle += transpose
    Parameters
    ----------
    coords_1
        shape = (n, 3)
    coords_2
        shape = (m, 3)
    gamma
    normalized
    Returns
    -------
    matrix; shape = (n, m)
    """
    distance_matrix = np.zeros((coords_1.shape[0], coords_2.shape[0]))
    for i in range(coords_1.shape[0]):
        for j in range(coords_2.shape[0]):
            distance_matrix[i, j] = np.exp(-gamma * np.sum((coords_1[i] - coords_2[j]) ** 2, axis=-1))
    if normalized:
        return normalize(distance_matrix)
    else:
        return distance_matrix


@nb.njit
# @numba_cc.export('get_rmsd', '(f64[:], f64[:])')
def get_rmsd(coords_1: np.ndarray, coords_2: np.ndarray) -> float:
    """
    RMSD of paired coordinates = normalized square-root of sum of squares of euclidean distances
    """
    return np.sqrt(np.sum((coords_1 - coords_2) ** 2) / coords_1.shape[0])


@nb.njit
# @numba_cc.export('get_caretta_score', '(f64[:], f64[:], f64, b1)')
def get_caretta_score(coords_1: np.ndarray, coords_2: np.ndarray, gamma, normalized) -> float:
    """
    Get caretta score for a a set of paired coordinates

    Parameters
    ----------
    coords_1
    coords_2
    gamma
    normalized

    Returns
    -------
    Caretta score
    """
    score = 0
    for i in range(coords_1.shape[0]):
        score += np.exp(
            -gamma * np.sum((coords_1[i] - coords_2[i]) ** 2, axis=-1))
    if normalized:
        return score / coords_1.shape[0]
    else:
        return score



