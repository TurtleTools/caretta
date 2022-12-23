import numba as nb
import numpy as np

MIN_FLOAT64 = np.finfo(np.float64).min


@nb.njit(cache=False)
def _make_dtw_matrix(
        seq1: np.ndarray, seq2: np.ndarray,
        score_matrix: np.ndarray,
        gap_open_penalty: float = 0.0,
        gap_extend_penalty: float = 0.0,
):
    """
    Make cost matrix using dynamic time warping

    Parameters
    ----------
    seq1
        array of size n of indices into score matrix for first sequence
    seq2
        array of size n indices into score matrix for second sequence
    score_matrix
        matrix of scores either
            of size (nxm) between all points of both objects meaning seq1 and seq2 are just np.arange(n) and np.arange(m)
            or of size (axa) between the alphabet of both objects meaning seq1 and seq2 are the indices of the alphabet
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
    n, m = len(seq1), len(seq2)
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
                    matrix[i - 1, j - 1, 1] + score_matrix[seq1[i - 1], seq2[j - 1]],
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
        seq1: np.ndarray,
        seq2: np.ndarray,
        score_matrix: np.ndarray,
        gap_open_penalty: float = 0.0,
        gap_extend_penalty: float = 0.0,
):
    """
    Align two objects using dynamic time warping

    Parameters
    ----------
    seq1
        array of size n of indices into score matrix for first sequence
    seq2
        array of size n indices into score matrix for second sequence
    score_matrix
        matrix of scores either
            of size (nxm) between all points of both objects meaning seq1 and seq2 are just np.arange(n) and np.arange(m)
            or of size (axa) between the alphabet of both objects meaning seq1 and seq2 are the indices of the alphabet
    gap_open_penalty
        penalty for opening a (series of) gap(s)
    gap_extend_penalty
        penalty for extending an existing series of gaps
    Returns
    -------
    aligned_indices_1, aligned_indices_2
    """
    matrix, backtrack = _make_dtw_matrix(seq1, seq2,
                                         score_matrix, gap_open_penalty, gap_extend_penalty
                                         )
    n = len(seq1)
    m = len(seq2)
    scores = np.array([matrix[n, m, 0], matrix[n, m, 1], matrix[n, m, 2]])
    index = np.argmax(scores)
    aln_1, aln_2 = _get_dtw_alignment(index, backtrack, n, m)
    return aln_1, aln_2, scores[index]


@nb.njit
def smith_waterman_score(seq1, seq2, matrix, gap=-1.):
    # Initialize the alignment score matrix with zeros
    rows = len(seq1) + 1
    cols = len(seq2) + 1
    score_matrix = np.zeros((rows, cols))

    # Fill in the score matrix
    for i in range(1, rows):
        for j in range(1, cols):
            if seq2[j - 1] == -1:
                break
            # Calculate the score for each possible alignment
            diagonal_score = score_matrix[i - 1][j - 1] + matrix[seq1[i - 1], seq2[j - 1]]
            left_score = score_matrix[i][j - 1] + gap
            up_score = score_matrix[i - 1][j] + gap
            # Take the maximum of the three possible scores
            score_matrix[i][j] = max(0., diagonal_score, left_score, up_score)

    # Find the position of the highest score in the matrix
    max_score = 0.
    for i in range(1, rows):
        for j in range(1, cols):
            if score_matrix[i][j] > max_score:
                max_score = score_matrix[i][j]
    return max_score


@nb.njit
def smith_waterman(seq1, seq2, score_matrix, gap=-1.):
    rows = len(seq1) + 1
    cols = len(seq2) + 1
    matrix = np.zeros((rows, cols))

    for i in range(1, rows):
        for j in range(1, cols):
            # Calculate the score for each possible alignment
            diagonal_score = matrix[i - 1][j - 1] + score_matrix[seq1[i - 1], seq2[j - 1]]
            left_score = matrix[i][j - 1] + gap
            up_score = matrix[i - 1][j] + gap
            # Take the maximum of the three possible scores
            matrix[i][j] = max(0, diagonal_score, left_score, up_score)

    # Find the position of the highest score in the matrix
    max_score = 0
    max_pos = None
    for i in range(1, rows):
        for j in range(1, cols):
            if matrix[i][j] > max_score:
                max_score = matrix[i][j]
                max_pos = (i, j)

    # Traceback to find the optimal alignment
    i, j = max_pos
    max_aln_length = i + j + 1
    align1 = np.zeros(max_aln_length, dtype=np.int64)
    align2 = np.zeros(max_aln_length, dtype=np.int64)
    index = 0
    while i > 0 and j > 0:
        score = matrix[i][j]
        diagonal_score = matrix[i - 1][j - 1]
        left_score = matrix[i][j - 1]
        up_score = matrix[i - 1][j]
        if score == 0:
            break
        elif score == diagonal_score + score_matrix[seq1[i - 1], seq2[j - 1]]:
            i -= 1
            j -= 1
            align1[index] = i
            align2[index] = j
            index += 1
        elif score == left_score + gap:
            j -= 1
            align1[index] = -1
            align2[index] = j
            index += 1
        elif score == up_score + gap:
            i -= 1
            align1[index] = i
            align2[index] = -1
            index += 1
    return align1[:index][::-1], align2[:index][::-1], max_score
