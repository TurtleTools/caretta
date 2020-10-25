import numba as nb
import numpy as np
from geometricus import MomentInvariants, SplitType, GeometricusEmbedding, MomentType
from caretta import dynamic_time_warping as dtw, score_functions, helper

"""
Provides pairwise superposition functions to use in Caretta
Each takes coords_1, coords_2, parameters as input
    parameters is a dict with gap_open_penalty, gap_extend_penalty, gamma, and other function-specific parameters as keys

returns score, superposed_coords_1, superposed_coords_2
"""


def dtw_svd_superpose_function(
    coords_1, coords_2, parameters: dict,
):
    """
    Assumes coords_1 and coords_2 are already in a well-superposed state,
    runs DTW alignment and then superposes with Kabsch on the aligning positions
    """
    score_matrix = score_functions.make_score_matrix(
        coords_1,
        coords_2,
        score_functions.get_caretta_score,
        parameters["gamma"],
        normalized=False,
    )
    _, coords_1, coords_2, common_coords_1, common_coords_2 = _align_and_superpose(
        coords_1,
        coords_2,
        score_matrix,
        parameters["gap_open_penalty"],
        parameters["gap_extend_penalty"],
    )
    return (
        score_functions.get_total_score(
            common_coords_1,
            common_coords_2,
            score_functions.get_caretta_score,
            parameters["gamma"],
            False,
        ),
        coords_1,
        coords_2,
    )


def signal_superpose_function(coords_1, coords_2, parameters):
    """
    Makes initial superposition of coordinates using DTW alignment of overlapping signals
    A signal is a vector of euclidean distances of first (or last) coordinate to all others in a "size"-residue stretch
    """
    score_first, c1_first, c2_first = _signal_superpose_index(
        0,
        coords_1,
        coords_2,
        parameters["gap_open_penalty"],
        parameters["gap_extend_penalty"],
        size=parameters["size"],
    )
    score_last, c1_last, c2_last = _signal_superpose_index(
        -1,
        coords_1,
        coords_2,
        parameters["gap_open_penalty"],
        parameters["gap_extend_penalty"],
        size=parameters["size"],
    )
    if score_first > score_last:
        return score_first, c1_first, c2_first
    else:
        return score_last, c1_last, c2_last


def signal_svd_superpose_function(coords_1, coords_2, parameters):
    """
    Uses signal_superpose followed by dtw_svd_superpose
    """
    _, coords_1, coords_2 = signal_superpose_function(coords_1, coords_2, parameters)
    return dtw_svd_superpose_function(coords_1, coords_2, parameters)


def moment_superpose_function(coords_1, coords_2, parameters):
    """
    Uses 4 rotation/translation invariant moments for each "split_size"-mer to run DTW
    """
    if "upsample_rate" not in parameters:
        parameters["upsample_rate"] = 10
    if "moment_types" not in parameters:
        parameters["moment_types"] = ["O_3", "O_4", "O_5", "F"]
    if "scale" not in parameters:
        parameters["scale"] = True
    if "gamma_moment" not in parameters:
        parameters["gamma_moment"] = 0.6
    if "gamma" not in parameters:
        parameters["gamma"] = 0.03
    if "gap_open_penalty" not in parameters:
        parameters["gap_open_penalty"] = 0.0
    if "gap_extend_penalty" not in parameters:
        parameters["gap_extend_penalty"] = 0.0

    moment_types = [MomentType[x] for x in parameters["moment_types"]]
    moments_1 = MomentInvariants.from_coordinates(
        "name",
        coords_1,
        split_type=SplitType[parameters["split_type"]],
        split_size=parameters["split_size"],
        upsample_rate=parameters["upsample_rate"],
        moment_types=moment_types,
    ).moments
    moments_2 = MomentInvariants.from_coordinates(
        "name",
        coords_2,
        split_type=SplitType[parameters["split_type"]],
        split_size=parameters["split_size"],
        upsample_rate=parameters["upsample_rate"],
        moment_types=moment_types,
    ).moments
    if parameters["scale"]:
        moments_1 = np.log1p(moments_1)
        moments_2 = np.log1p(moments_2)
    score_matrix = score_functions.make_score_matrix(
        moments_1,
        moments_2,
        score_functions.get_caretta_score,
        parameters["gamma_moment"],
        normalized=True,
    )
    score, coords_1, coords_2, _, _ = _align_and_superpose(
        coords_1,
        coords_2,
        score_matrix,
        parameters["gap_open_penalty"],
        parameters["gap_extend_penalty"],
    )
    return score, coords_1, coords_2


def geometricus_superpose_function(coords_1, coords_2, parameters):
    if "upsample_rate" not in parameters:
        parameters["upsample_rate"] = 10
    invariants = [
        MomentInvariants.from_coordinates(
            "name1",
            coords_1,
            split_type=SplitType[parameters[f"split_type"]],
            split_size=parameters[f"split_size"],
            upsample_rate=parameters["upsample_rate"],
        ),
        MomentInvariants.from_coordinates(
            "name2",
            coords_2,
            split_type=SplitType[parameters[f"split_type"]],
            split_size=parameters[f"split_size"],
            upsample_rate=parameters["upsample_rate"],
        ),
    ]
    embedder = GeometricusEmbedding.from_invariants(
        invariants, resolution=parameters["resolution"], protein_keys=["name1", "name2"]
    )
    score_matrix = score_functions.make_score_matrix(
        np.array(embedder.proteins_to_shapemers["name1"]),
        np.array(embedder.proteins_to_shapemers["name2"]),
        score_functions.get_caretta_score,
        gamma=parameters["gamma"],
        normalized=False,
    )
    score, coords_1, coords_2, _, _ = _align_and_superpose(
        coords_1,
        coords_2,
        score_matrix,
        parameters["gap_open_penalty"],
        parameters["gap_extend_penalty"],
    )
    return score, coords_1, coords_2


def geometricus_svd_superpose_function(coords_1, coords_2, parameters):
    """
    Uses moment_multiple_superpose followed by dtw_svd_superpose
    """
    _, coords_1, coords_2 = geometricus_superpose_function(
        coords_1, coords_2, parameters
    )
    return dtw_svd_superpose_function(coords_1, coords_2, parameters)


def moment_multiple_superpose_function(coords_1, coords_2, parameters):
    """
    Uses 4 rotation/translation invariant moments for each "split_size"-mer with different fragmentation approaches to run DTW
    """
    moments_1 = []
    moments_2 = []
    if "upsample_rate" not in parameters:
        parameters["upsample_rate"] = 10
    for i in range(parameters["num_split_types"]):
        if f"moment_types_{i}" not in parameters:
            parameters[f"moment_types_{i}"] = ["O_3", "O_4", "O_5", "F"]
    if "scale" not in parameters:
        parameters["scale"] = True
    if "gamma_moment" not in parameters:
        parameters["gamma_moment"] = 0.6
    if "gamma" not in parameters:
        parameters["gamma"] = 0.03
    if "gap_open_penalty" not in parameters:
        parameters["gap_open_penalty"] = 0.0
    if "gap_extend_penalty" not in parameters:
        parameters["gap_extend_penalty"] = 0.0

    for i in range(parameters["num_split_types"]):
        moment_types = [MomentType[x] for x in parameters[f"moment_types_{i}"]]
        moments_1_1 = MomentInvariants.from_coordinates(
            "name",
            coords_1,
            split_type=SplitType[parameters[f"split_type_{i}"]],
            split_size=parameters[f"split_size_{i}"],
            upsample_rate=parameters["upsample_rate"],
            moment_types=moment_types,
        ).moments
        moments_2_1 = MomentInvariants.from_coordinates(
            "name",
            coords_2,
            split_type=SplitType[parameters[f"split_type_{i}"]],
            split_size=parameters[f"split_size_{i}"],
            upsample_rate=parameters["upsample_rate"],
            moment_types=moment_types,
        ).moments
        if parameters["scale"]:
            moments_1_1 = np.log1p(moments_1_1)
            moments_2_1 = np.log1p(moments_2_1)
        moments_1.append(moments_1_1)
        moments_2.append(moments_2_1)
    score_matrix = np.zeros((moments_1[0].shape[0], moments_2[0].shape[0]))
    for (m_1, m_2) in zip(moments_1, moments_2):
        score_matrix += score_functions.make_score_matrix(
            m_1,
            m_2,
            score_functions.get_caretta_score,
            gamma=parameters["gamma_moment"],
            normalized=True,
        )
    score, coords_1, coords_2, _, _ = _align_and_superpose(
        coords_1,
        coords_2,
        score_matrix,
        parameters["gap_open_penalty"],
        parameters["gap_extend_penalty"],
    )
    return score, coords_1, coords_2


def moment_multiple_svd_superpose_function(coords_1, coords_2, parameters):
    """
    Uses moment_multiple_superpose followed by dtw_svd_superpose
    """
    _, coords_1, coords_2 = moment_multiple_superpose_function(
        coords_1, coords_2, parameters
    )
    return dtw_svd_superpose_function(coords_1, coords_2, parameters)


def moment_svd_superpose_function(coords_1, coords_2, parameters):
    """
    Uses moment_superpose followed by dtw_svd_superpose
    """
    _, coords_1, coords_2 = moment_superpose_function(coords_1, coords_2, parameters)
    return dtw_svd_superpose_function(coords_1, coords_2, parameters)


@nb.njit
def _align_and_superpose(
    coords_1, coords_2, score_matrix, gap_open_penalty, gap_extend_penalty
):
    """
    Runs DTW on a score matrix and Kabsch superposition on resulting alignment
    """
    dtw_aln_array_1, dtw_aln_array_2, score = dtw.dtw_align(
        score_matrix, gap_open_penalty, gap_extend_penalty
    )
    pos_1, pos_2 = helper.get_common_positions(dtw_aln_array_1, dtw_aln_array_2)
    common_coords_1, common_coords_2 = coords_1[pos_1], coords_2[pos_2]
    coords_1, coords_2, common_coords_2 = paired_svd_superpose_with_subset(
        coords_1, coords_2, common_coords_1, common_coords_2
    )
    return score, coords_1, coords_2, common_coords_1, common_coords_2


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


def _signal_superpose_index(
    index,
    coords_1,
    coords_2,
    gap_open_penalty=0.0,
    gap_extend_penalty=0.0,
    size=30,
    overlap=1,
):
    """
    Makes initial superposition using DTW alignment of overlapping signals
    A signal is a vector of euclidean distances of first (or last) coordinate to all others in a 30-residue stretch
    """

    def _make_signal_index(coords, idx):
        centroid = coords[idx]
        distances = np.zeros(coords.shape[0])
        for c in range(coords.shape[0]):
            distances[c] = np.sqrt(np.sum((coords[c] - centroid) ** 2, axis=-1))
        return distances

    signals_1 = np.zeros(((coords_1.shape[0] - size) // overlap, size))
    signals_2 = np.zeros(((coords_2.shape[0] - size) // overlap, size))
    middles_1 = np.zeros((signals_1.shape[0], coords_1.shape[1]))
    middles_2 = np.zeros((signals_2.shape[0], coords_2.shape[1]))
    if index == -1:
        index = size - 1
    for x, i in enumerate(range(0, signals_1.shape[0] * overlap, overlap)):
        signals_1[x] = _make_signal_index(coords_1[i : i + size], index)
        middles_1[x] = coords_1[i + index]
    for x, i in enumerate(range(0, signals_2.shape[0] * overlap, overlap)):
        signals_2[x] = _make_signal_index(coords_2[i : i + size], index)
        middles_2[x] = coords_2[i + index]
    score_matrix = score_functions.make_score_matrix(
        signals_1,
        signals_2,
        score_functions.get_signal_score,
        gamma=0.1,
        normalized=False,
    )
    dtw_1, dtw_2, score = dtw.dtw_align(
        score_matrix, gap_open_penalty, gap_extend_penalty
    )
    pos_1, pos_2 = helper.get_common_positions(dtw_1, dtw_2)
    aln_coords_1 = np.zeros((len(pos_1), coords_1.shape[1]))
    aln_coords_2 = np.zeros((len(pos_2), coords_2.shape[1]))
    for i, (p1, p2) in enumerate(zip(pos_1, pos_2)):
        aln_coords_1[i] = middles_1[p1]
        aln_coords_2[i] = middles_2[p2]
    coords_1, coords_2, _ = paired_svd_superpose_with_subset(
        coords_1, coords_2, aln_coords_1, aln_coords_2
    )
    return score, coords_1, coords_2


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
    coords_1 = coords_1 - helper.nb_mean_axis_0(common_coords_1)
    coords_2 = np.dot(coords_2 - helper.nb_mean_axis_0(common_coords_2), rot)
    common_coords_2_rot = apply_rotran(common_coords_2, rot, tran)
    return coords_1, coords_2, common_coords_2_rot
