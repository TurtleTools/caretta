import numba as nb
import numpy as np

from caretta import helper


@nb.njit
def _nb_mean_axis_0(array: np.ndarray) -> np.ndarray:
    """
    Same as np.mean(array, axis=0) but njitted
    """
    mean_array = np.zeros(array.shape[1])
    for i in range(array.shape[1]):
        mean_array[i] = np.mean(array[:, i])
    return mean_array


@nb.njit
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
    centroid_1, centroid_2 = _nb_mean_axis_0(coords_1), _nb_mean_axis_0(coords_2)
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


@nb.njit
def make_distance_matrix(coords_1: np.ndarray, coords_2: np.ndarray, tm_score=False, normalize=False) -> np.ndarray:
    """
    Makes matrix of euclidean distances of each coordinate in coords_1 to each coordinate in coords_2
    TODO: probably faster to do upper triangle += transpose
    Parameters
    ----------
    coords_1
        shape = (n, 3)
    coords_2
        shape = (m, 3)
    tm_score
    normalize
    Returns
    -------
    matrix; shape = (n, m)
    """
    distance_matrix = np.zeros((coords_1.shape[0], coords_2.shape[0]))
    gamma = 0.3
    for i in range(coords_1.shape[0]):
        for j in range(coords_2.shape[0]):
            if tm_score:
                d0 = 1.24 * (min(coords_1.shape[0], coords_2.shape[0]) - 15) ** (1 / 3) - 1.8
                distance_matrix[i, j] = 1 / (1 + np.sum((coords_1[i] - coords_2[j]) ** 2, axis=-1) / d0 ** 2)
                # distance_matrix[i, j] = np.sqrt(np.sum((coords_1[i] - coords_2[j]) ** 2, axis=-1))
            else:
                # distance_matrix[i, j] = 1 / (1 + np.sqrt(np.sum((coords_1[i] - coords_2[j]) ** 2, axis=-1)))
                distance_matrix[i, j] = np.exp(
                    -gamma * np.sqrt(np.sum((coords_1[i] - coords_2[j]) ** 2, axis=-1)))  # / (0.01*max(coords_1.shape[0], coords_2.shape[0])))
    if normalize:
        return helper.normalize(distance_matrix)
    else:
        return distance_matrix


@nb.njit
def get_rmsd(coords_1: np.ndarray, coords_2: np.ndarray) -> float:
    """
    RMSD of paired coordinates = normalized square-root of sum of squares of euclidean distances
    """
    return np.sqrt(np.sum((coords_1 - coords_2) ** 2) / coords_1.shape[0])


@nb.njit
def get_exp_distances(coords_1: np.ndarray, coords_2: np.ndarray, gamma=0.3) -> float:
    """
    """
    return np.sum(np.exp(-gamma * (coords_1 - coords_2) ** 2))


@nb.njit
def get_rmsd_superimposed(coords_1: np.ndarray, coords_2: np.ndarray) -> float:
    """
    RMSD of paired coordinates after SVD superimposing

    Parameters
    ----------
    coords_1
        numpy array of coordinate data for the first protein; shape = (n, 3)
    coords_2
        numpy array of corresponding coordinate data for the second protein; shape = (n, 3)

    Returns
    -------
    rmsd
    """
    rot, tran = svd_superimpose(coords_1, coords_2)
    coords_2 = apply_rotran(coords_2, rot, tran)
    return get_rmsd(coords_1, coords_2)
