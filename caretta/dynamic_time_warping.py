import numba as nb
import numpy as np

MIN_FLOAT64 = np.finfo(np.float64).min


@nb.njit(cache=False)
def _make_dtw_matrix(
    score_matrix: np.ndarray,
    gap_open_penalty: float = 0.0,
    gap_extend_penalty: float = 0.0,
):
    """
    Make cost matrix using dynamic time warping

    Parameters
    ----------
    score_matrix
        matrix of scores between corresponding vectors of the two vector sets; shape = (n, m)
    gap_open_penalty
        penalty for opening a (series of) gap(s)
    gap_extend_penalty
        penalty for extending an existing series of gaps

    Returns
    -------
    accumulated cost matrix; shape = (n, m)
    """
    gap_open_penalty *= -1
    gap_extend_penalty *= -1
    n, m = score_matrix.shape
    matrix = np.zeros((n + 1, m + 1, 3), dtype=np.float64)
    matrix[:, 0, :] = MIN_FLOAT64
    matrix[0, :, :] = MIN_FLOAT64
    matrix[0, 0] = 0
    backtrack = np.zeros((n + 1, m + 1, 3), dtype=np.int64)
    for i in range(1, n + 1):
        matrix[i, 0, 0] = 0
        matrix[i, 0, 1] = 0
        matrix[i, 0, 2] = MIN_FLOAT64 - gap_open_penalty
        backtrack[i, 0] = 0

    for j in range(1, m + 1):
        matrix[0, j, 0] = MIN_FLOAT64 - gap_open_penalty
        matrix[0, j, 1] = 0
        matrix[0, j, 2] = 0
        backtrack[0, j] = 1

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            scores_lower = np.array(
                [
                    matrix[i - 1, j, 0] + gap_extend_penalty,
                    matrix[i - 1, j, 1] + gap_open_penalty,
                ]
            )
            index_lower = np.argmax(scores_lower)
            matrix[i, j, 0] = scores_lower[index_lower]
            backtrack[i, j, 0] = index_lower

            scores_upper = np.array(
                [
                    matrix[i, j - 1, 1] + gap_open_penalty,
                    matrix[i, j - 1, 2] + gap_extend_penalty,
                ]
            )
            index_upper = np.argmax(scores_upper)
            matrix[i, j, 2] = scores_upper[index_upper]
            backtrack[i, j, 2] = index_upper + 1

            scores = np.array(
                [
                    matrix[i, j, 0],
                    matrix[i - 1, j - 1, 1] + score_matrix[i - 1, j - 1],
                    matrix[i, j, 2],
                ]
            )
            index = np.argmax(scores)
            matrix[i, j, 1] = scores[index]
            backtrack[i, j, 1] = index
    return matrix, backtrack


@nb.njit(cache=False)
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
            index += 1
        elif n == 0:
            m -= 1
            indices_1[index] = -1
            indices_2[index] = m
            index += 1
        else:
            if direction == 0:
                direction = backtrack[n, m, 0]
                n -= 1
                indices_1[index] = n
                indices_2[index] = -1
                index += 1
            elif direction == 1:
                direction = backtrack[n, m, 1]
                if direction == 1:
                    n -= 1
                    m -= 1
                    indices_1[index] = n
                    indices_2[index] = m
                    index += 1
            elif direction == 2:
                direction = backtrack[n, m, 2]
                m -= 1
                indices_1[index] = -1
                indices_2[index] = m
                index += 1
    return indices_1[:index][::-1], indices_2[:index][::-1]


@nb.njit
def dtw_align(
    score_matrix: np.ndarray,
    gap_open_penalty: float = 0.0,
    gap_extend_penalty: float = 0.0,
):
    """
    Align two objects using dynamic time warping

    Parameters
    ----------
    score_matrix
        (n x m) matrix of scores between all points of both objects
    gap_open_penalty
        penalty for opening a (series of) gap(s)
    gap_extend_penalty
        penalty for extending an existing series of gaps
    Returns
    -------
    aligned_indices_1, aligned_indices_2
    """
    matrix, backtrack = _make_dtw_matrix(
        score_matrix, gap_open_penalty, gap_extend_penalty
    )
    n = score_matrix.shape[0]
    m = score_matrix.shape[1]
    scores = np.array([matrix[n, m, 0], matrix[n, m, 1], matrix[n, m, 2]])
    index = np.argmax(scores)
    aln_1, aln_2 = _get_dtw_alignment(index, backtrack, n, m)
    return aln_1, aln_2, scores[index]
