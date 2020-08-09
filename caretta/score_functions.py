import numba as nb
import numpy as np
from caretta import helper


@nb.njit
def get_caretta_score(coord_1: np.ndarray, coord_2: np.ndarray, gamma=0.03):
    """
    Gaussian (RBF) score of similarity between two coordinates
    """
    return np.exp(-gamma * np.sum((coord_1 - coord_2) ** 2, axis=-1))


@nb.njit
def get_signal_score(signal_1, signal_2):
    """
    Used in superposition_functions.signal_superpose_function
    """
    return np.median(np.exp(-0.1 * (signal_1 - signal_2) ** 2))


@nb.njit
def get_rmsd(coords_1: np.ndarray, coords_2: np.ndarray) -> float:
    """
    RMSD of paired coordinates = normalized square-root of sum of squares of euclidean distances
    """
    return np.sqrt(np.sum((coords_1 - coords_2) ** 2) / coords_1.shape[0])


def make_score_matrix_python(
    coords_1: np.ndarray, coords_2: np.ndarray, score_function, normalized=False
) -> np.ndarray:
    """
    Makes matrix of scores of each coordinate in coords_1 to each coordinate in coords_2
    Parameters
    ----------
    coords_1
        shape = (n, 3)
    coords_2
        shape = (m, 3)
    score_function
    normalized
    Returns
    -------
    matrix; shape = (n, m)
    """
    score_matrix = np.zeros((coords_1.shape[0], coords_2.shape[0]))
    for i in range(coords_1.shape[0]):
        for j in range(coords_2.shape[0]):
            score_matrix[i, j] = score_function(coords_1[i], coords_2[j])
    if normalized:
        return helper.normalize(score_matrix)
    else:
        return score_matrix


@nb.njit
def make_score_matrix(
    coords_1: np.ndarray, coords_2: np.ndarray, score_function, normalized=False
) -> np.ndarray:
    """
    Makes matrix of scores of each coordinate in coords_1 to each coordinate in coords_2
    Parameters
    ----------
    coords_1
        shape = (n, 3)
    coords_2
        shape = (m, 3)
    score_function
    normalized
    Returns
    -------
    matrix; shape = (n, m)
    """
    score_matrix = np.zeros((coords_1.shape[0], coords_2.shape[0]))
    for i in range(coords_1.shape[0]):
        for j in range(coords_2.shape[0]):
            score_matrix[i, j] = score_function(coords_1[i], coords_2[j])
    if normalized:
        return helper.normalize(score_matrix)
    else:
        return score_matrix


@nb.njit
def get_total_score(
    coords_1: np.ndarray, coords_2: np.ndarray, score_function, normalized=False
) -> float:
    """
    Get total score for a set of paired coordinates

    Parameters
    ----------
    coords_1
    coords_2
    score_function
    normalized
        if True, divides by length of first coordinate set

    Returns
    -------
    Total score
    """
    score = 0
    for i in range(coords_1.shape[0]):
        score += score_function(coords_1[i], coords_2[i])
    if normalized:
        return score / coords_1.shape[0]
    else:
        return score


@nb.njit
def get_common_positions(aln_array_1, aln_array_2, gap=-1):
    """
    Return positions where neither alignment has a gap

    Parameters
    ----------
    aln_array_1
    aln_array_2
    gap

    Returns
    -------
    common_positions_1, common_positions_2
    """
    pos_1 = np.array(
        [
            aln_array_1[i]
            for i in range(len(aln_array_1))
            if aln_array_1[i] != gap and aln_array_2[i] != gap
        ],
        dtype=np.int64,
    )
    pos_2 = np.array(
        [
            aln_array_2[i]
            for i in range(len(aln_array_2))
            if aln_array_1[i] != gap and aln_array_2[i] != gap
        ],
        dtype=np.int64,
    )
    return pos_1, pos_2
