import numba as nb
import numpy as np


@nb.njit
def _make_dtw_matrix(distance_matrix: np.ndarray,
                     gap_open_penalty: float = 0.,
                     gap_extend_penalty: float = 0.):
    """
    Make cost matrix using dynamic time warping

    Parameters
    ----------
    distance_matrix
        matrix of distances between corresponding vectors of the two vector sets; shape = (n, m)
    gap_open_penalty
        penalty for opening a (series of) gap(s)
    gap_extend_penalty
        penalty for extending an existing series of gaps

    Returns
    -------
    accumulated cost matrix; shape = (n, m)
    """
    n, m = distance_matrix.shape
    matrix = np.zeros((n + 1, m + 1, 3), dtype=np.float64)
    matrix[:, 0, :] = np.inf
    matrix[0, :, :] = np.inf
    matrix[0, 0] = 0
    backtrack = np.zeros((n + 1, m + 1, 3), dtype=np.int32)
    for i in range(1, n + 1):
        matrix[i, 0, 0] = gap_open_penalty + ((i - 1) * gap_extend_penalty)
        matrix[i, 0, 1] = gap_open_penalty + ((i - 1) * gap_extend_penalty)
        matrix[i, 0, 2] = np.inf - gap_open_penalty
        backtrack[i, 0] = 0

    for j in range(1, m + 1):
        matrix[0, j, 0] = np.inf - gap_open_penalty
        matrix[0, j, 1] = gap_open_penalty + ((j - 1) * gap_extend_penalty)
        matrix[0, j, 2] = gap_open_penalty + ((j - 1) * gap_extend_penalty)
        backtrack[0, j] = 1

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            scores_lower = np.array([matrix[i - 1, j, 0] + gap_extend_penalty, matrix[i - 1, j, 1] + gap_open_penalty]) + distance_matrix[
                i - 1, j - 1]
            min_index_lower = np.argmin(scores_lower)
            min_value_lower = scores_lower[min_index_lower]
            matrix[i, j, 0] = min_value_lower
            backtrack[i, j, 0] = min_index_lower

            scores_upper = np.array([matrix[i, j - 1, 1] + gap_open_penalty, matrix[i, j - 1, 2] + gap_extend_penalty]) + distance_matrix[
                i - 1, j - 1]
            min_index_upper = np.argmin(scores_upper)
            min_value_upper = scores_upper[min_index_upper]
            matrix[i, j, 2] = min_value_upper
            backtrack[i, j, 2] = min_index_upper + 1

            scores = np.array([matrix[i, j, 0], matrix[i - 1, j - 1, 1], matrix[i, j, 2]]) + distance_matrix[i - 1, j - 1]
            min_index = np.argmin(scores)
            min_value = scores[min_index]
            matrix[i, j, 1] = min_value
            backtrack[i, j, 1] = min_index
    return matrix, backtrack


@nb.njit
def _get_dtw_alignment(start_direction, backtrack: np.ndarray, n1, m1):
    """
    Finds optimal warping path from a backtrack matrix

    Parameters
    ----------
    start_direction
    backtrack
    n1
        length of first sequence
    m1
        length of second sequence

    Returns
    -------
    aligned_indices_1, aligned_indices_2
    """
    indices_1 = np.zeros(n1 + m1 + 1, dtype=np.int64)
    indices_2 = np.zeros(n1 + m1 + 1, dtype=np.int64)
    index = 0
    n, m = n1, m1
    direction = start_direction
    while not (n == 0 and m == 0):
        if m == 0:
            n -= 1
            indices_1[index] = n
            indices_2[index] = -1
        elif n == 0:
            m -= 1
            indices_1[index] = -1
            indices_2[index] = m
        else:
            if direction == 0:
                direction = backtrack[n, m, 0]
                n -= 1
                indices_1[index] = n
                indices_2[index] = -1
            elif direction == 1:
                direction = backtrack[n, m, 1]
                n -= 1
                m -= 1
                indices_1[index] = n
                indices_2[index] = m
            elif direction == 2:
                direction = backtrack[n, m, 2]
                m -= 1
                indices_1[index] = -1
                indices_2[index] = m
        index += 1
    return indices_1[:index][::-1], indices_2[:index][::-1]


@nb.njit
def dtw_align(distance_matrix: np.ndarray, gap_open_penalty: float = 0., gap_extend_penalty: float = 0.):
    """
    Align two objects using dynamic time warping

    Parameters
    ----------
    distance_matrix
        (n x m) matrix of distances between all points of both objects
    gap_open_penalty
        penalty for opening a (series of) gap(s)
    gap_extend_penalty
        penalty for extending an existing series of gaps
    Returns
    -------
    aligned_indices_1, aligned_indices_2
    """
    matrix, backtrack = _make_dtw_matrix(distance_matrix, gap_open_penalty, gap_extend_penalty)
    n = distance_matrix.shape[0]
    m = distance_matrix.shape[1]
    scores = np.array([matrix[n, m, 0], matrix[n, m, 1], matrix[n, m, 2]])
    index = np.argmin(scores)
    aln_1, aln_2 = _get_dtw_alignment(index, backtrack, n, m)
    return aln_1, aln_2
