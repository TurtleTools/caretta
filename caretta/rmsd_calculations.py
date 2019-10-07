import matplotlib.pyplot as plt
import numba as nb
import numpy as np

from caretta import dynamic_time_warping as dtw
from caretta import helper


@nb.njit
def nb_mean_axis_0(array: np.ndarray) -> np.ndarray:
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
def svd_superimpose_rot(coords_1: np.ndarray, coords_2: np.ndarray):
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
    correlation_matrix = np.dot(coords_2.T, coords_1)
    u, s, v = np.linalg.svd(correlation_matrix)
    reflect = np.linalg.det(u) * np.linalg.det(v) < 0
    if reflect:
        s[-1] = -s[-1]
        u[:, -1] = -u[:, -1]
    rotation_matrix = np.dot(u, v)
    return rotation_matrix.astype(np.float64)


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
def apply_rot(coords: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
    """
    Applies a rotation and translation matrix onto coordinates

    Parameters
    ----------
    coords
    rotation_matrix

    Returns
    -------
    transformed coordinates
    """
    return np.dot(coords, rotation_matrix)


@nb.njit
def superpose_with_pos(coords_1, coords_2, common_coords_1, common_coords_2):
    rot, tran = svd_superimpose(common_coords_1, common_coords_2)
    coords_1 = coords_1 - nb_mean_axis_0(common_coords_1)
    coords_2 = apply_rot(coords_2 - nb_mean_axis_0(common_coords_2), rot)
    common_coords_2_rot = apply_rotran(common_coords_2, rot, tran)
    return coords_1, coords_2, common_coords_2_rot


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
    gamma = 0.15
    for i in range(coords_1.shape[0]):
        for j in range(coords_2.shape[0]):
            if tm_score:
                d0 = 1.24 * (min(coords_1.shape[0], coords_2.shape[0]) - 15) ** (1 / 3) - 1.8
                distance_matrix[i, j] = 1 / (1 + np.sum((coords_1[i] - coords_2[j]) ** 2, axis=-1) / d0 ** 2)
            else:
                distance_matrix[i, j] = np.exp(
                    -gamma * np.sqrt(np.sum((coords_1[i] - coords_2[j]) ** 2, axis=-1)))
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
def get_exp_distances(coords_1: np.ndarray, coords_2: np.ndarray, normalize, gamma=0.15) -> float:
    """
    """
    score = 0
    for i in range(coords_1.shape[0]):
        score += np.exp(
            -gamma * np.sqrt(np.sum((coords_1[i] - coords_2[i]) ** 2, axis=-1)))
    if normalize:
        return score / coords_1.shape[0]
    else:
        return score


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


@nb.njit
def make_signal(coords):
    centroid = coords[int(np.ceil(coords.shape[0] / 2))]
    distances = np.zeros(coords.shape[0])
    for i in range(coords.shape[0]):
        distances[i] = np.sqrt(np.sum((coords[i] - centroid) ** 2, axis=-1))
    return distances


@nb.njit
def make_signal_index(coords, index):
    centroid = coords[index]
    distances = np.zeros(coords.shape[0])
    for i in range(coords.shape[0]):
        distances[i] = np.sqrt(np.sum((coords[i] - centroid) ** 2, axis=-1))
    return distances


@nb.njit
def make_signal_other(coords, index, length):
    coords_sub = coords[index: index + length]
    return make_signal(coords_sub)


@nb.njit
def slide(coords_1, coords_2):
    flip = False
    if coords_1.shape[0] > coords_2.shape[0]:
        coords_1, coords_2 = coords_2, coords_1
        flip = True
    signal_1 = make_signal(coords_1)
    length = signal_1.shape[0]
    dots = np.zeros(coords_2.shape[0] - length)
    for i in range(coords_2.shape[0] - length):
        signal_2 = make_signal_other(coords_2, i, length)
        dots[i] = np.sum(np.abs(signal_1 - signal_2))
    if dots.shape[0] > 0:
        max_index = np.argmin(dots)
    else:
        max_index = 0
    coords_1, coords_2, _ = superpose_with_pos(coords_1, coords_2, coords_1, coords_2[max_index: max_index + length])
    if flip:
        return coords_2, coords_1
    else:
        return coords_1, coords_2


@nb.njit
def slide1(coords_1, coords_2, limit=20):
    flip = False
    if coords_1.shape[0] > coords_2.shape[0]:
        coords_1, coords_2 = coords_2, coords_1
        flip = True
    if coords_1.shape[0] < limit:
        limit = coords_1.shape[0]
    signal_1 = make_signal(coords_1)
    length = signal_1.shape[0]
    pad = np.zeros((length, 3))
    padded_long = np.concatenate((pad, coords_2, pad))
    dots = np.zeros(padded_long.shape[0] - length)
    dots[:] = np.inf
    for i in range(limit, padded_long.shape[0] - length - limit):
        signal_2 = make_signal_other(padded_long, i, length)
        dots[i] = np.sum(np.abs(signal_1 - signal_2))
    best_index = np.argmin(dots)
    if best_index < length:
        coords_1_i = coords_1[-best_index:]
        coords_2_i = coords_2[:best_index]
    elif best_index <= coords_2.shape[0]:
        coords_2_i = coords_2[best_index - length:best_index]
        coords_1_i = coords_1
    elif best_index == coords_1.shape[0] == coords_2.shape[0]:
        coords_1_i, coords_2_i = coords_1, coords_2
    else:
        coords_1_i = coords_1[:-(best_index - coords_2.shape[0])]
        coords_2_i = coords_2[best_index - length:]
    coords_1, coords_2, _ = superpose_with_pos(coords_1, coords_2, coords_1_i, coords_2_i)
    if flip:
        return coords_2, coords_1
    else:
        return coords_1, coords_2


# @nb.njit
def slide_middle(coords_1, coords_2, size=40):
    sub_coords = coords_1[max(0, coords_1.shape[0] // 2 - size // 2): min(coords_1.shape[0], coords_1.shape[0] // 2 + size // 2)]
    signal_1 = make_signal(sub_coords)
    length = signal_1.shape[0]
    dots = np.zeros(coords_2.shape[0] - length)
    for i in range(coords_2.shape[0] - length):
        signal_2 = make_signal(coords_2[i: i + length])
        dots[i] = np.sum(np.abs(signal_1 - signal_2))
    if dots.shape[0] > 0:
        max_index = np.argmin(dots)
    else:
        max_index = 0
    coords_1, coords_2, _ = superpose_with_pos(coords_1, coords_2, sub_coords, coords_2[max_index: max_index + length])
    return coords_1, coords_2


# @nb.njit
def dtw_signals(coords_1, coords_2, gap_open_penalty, gap_extend_penalty, size=50, overlap=1, gamma=0.15):
    signals_1 = np.zeros(((coords_1.shape[0] - size) // overlap, size))
    signals_2 = np.zeros(((coords_2.shape[0] - size) // overlap, size))
    middles_1 = np.zeros((signals_1.shape[0], coords_1.shape[1]))
    middles_2 = np.zeros((signals_2.shape[0], coords_2.shape[1]))
    for x, i in enumerate(range(0, signals_1.shape[0] * overlap, overlap)):
        signals_1[x] = make_signal(coords_1[i: i + size])
        middles_1[x] = coords_1[int(np.ceil((i + size) / 2))]
    for x, i in enumerate(range(0, signals_2.shape[0] * overlap, overlap)):
        signals_2[x] = make_signal(coords_2[i: i + size])
        middles_2[x] = coords_2[int(np.ceil((i + size) / 2))]
    distance_matrix = np.zeros((signals_1.shape[0], signals_2.shape[0]))
    for i in range(signals_1.shape[0]):
        for j in range(signals_2.shape[0]):
            distance_matrix[i, j] = np.median(np.exp(-gamma * np.sqrt((signals_1[i] - signals_2[j]) ** 2)))
    dtw_1, dtw_2, _ = dtw.dtw_align(distance_matrix, 0, 0)
    pos_1, pos_2 = helper.get_common_positions(dtw_1, dtw_2)
    print("Signal inner")
    plt.imshow(distance_matrix)
    plt.colorbar()
    plt.plot(pos_2, pos_1, c="red")
    plt.pause(0.1)
    aln_coords_1 = np.zeros((len(pos_1), coords_1.shape[1]))
    aln_coords_2 = np.zeros((len(pos_2), coords_2.shape[1]))
    for i, (p1, p2) in enumerate(zip(pos_1, pos_2)):
        aln_coords_1[i] = middles_1[p1]
        aln_coords_2[i] = middles_2[p2]
    coords_1, coords_2, _ = superpose_with_pos(coords_1, coords_2, aln_coords_1, aln_coords_2)
    return coords_1, coords_2


@nb.njit
def dtw_signals_index(coords_1, coords_2, index, gap_open_penalty, gap_extend_penalty, size=30, overlap=1, gamma=0.15, plot=False):
    signals_1 = np.zeros(((coords_1.shape[0] - size) // overlap, size))
    signals_2 = np.zeros(((coords_2.shape[0] - size) // overlap, size))
    middles_1 = np.zeros((signals_1.shape[0], coords_1.shape[1]))
    middles_2 = np.zeros((signals_2.shape[0], coords_2.shape[1]))
    if index == -1:
        index = size - 1
    for x, i in enumerate(range(0, signals_1.shape[0] * overlap, overlap)):
        signals_1[x] = make_signal_index(coords_1[i: i + size], index)
        middles_1[x] = coords_1[i + index]
    for x, i in enumerate(range(0, signals_2.shape[0] * overlap, overlap)):
        signals_2[x] = make_signal_index(coords_2[i: i + size], index)
        middles_2[x] = coords_2[i + index]
    distance_matrix = np.zeros((signals_1.shape[0], signals_2.shape[0]))
    for i in range(signals_1.shape[0]):
        for j in range(signals_2.shape[0]):
            distance_matrix[i, j] = np.median(np.exp(-gamma * np.sqrt((signals_1[i] - signals_2[j]) ** 2)))
    dtw_1, dtw_2, _ = dtw.dtw_align(distance_matrix, 0., 0.)
    pos_1, pos_2 = helper.get_common_positions(dtw_1, dtw_2)
    # if plot:
    #     print("Signal inner")
    #     plt.imshow(distance_matrix)
    #     plt.colorbar()
    #     plt.plot(pos_2, pos_1, c="red")
    #     plt.show()
    aln_coords_1 = np.zeros((len(pos_1), coords_1.shape[1]))
    aln_coords_2 = np.zeros((len(pos_2), coords_2.shape[1]))
    for i, (p1, p2) in enumerate(zip(pos_1, pos_2)):
        aln_coords_1[i] = middles_1[p1]
        aln_coords_2[i] = middles_2[p2]
    coords_1, coords_2, _ = superpose_with_pos(coords_1, coords_2, aln_coords_1, aln_coords_2)
    return coords_1, coords_2
