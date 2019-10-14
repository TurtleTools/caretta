import numba as nb
import numpy as np

from caretta import dynamic_time_warping as dtw
from caretta import rmsd_calculations, helper


@nb.njit
def get_common_coordinates(coords_1, coords_2, aln_1, aln_2, gap=-1):
    assert aln_1.shape == aln_2.shape
    pos_1, pos_2 = helper.get_common_positions(aln_1, aln_2, gap)
    return coords_1[pos_1], coords_2[pos_2]


@nb.njit
def make_signal_index(coords, index):
    centroid = coords[index]
    distances = np.zeros(coords.shape[0])
    for i in range(coords.shape[0]):
        distances[i] = np.sqrt(np.sum((coords[i] - centroid) ** 2, axis=-1))
    return distances


@nb.njit
def dtw_signals_index(coords_1, coords_2, gamma, index, gap_open_penalty, gap_extend_penalty, size=30, overlap=1):
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
            distance_matrix[i, j] = np.median(np.exp(
                -0.1 * (signals_1[i] - signals_2[j]) ** 2))
    dtw_1, dtw_2, _ = dtw.dtw_align(distance_matrix, 0., 0.)
    pos_1, pos_2 = helper.get_common_positions(dtw_1, dtw_2)
    aln_coords_1 = np.zeros((len(pos_1), coords_1.shape[1]))
    aln_coords_2 = np.zeros((len(pos_2), coords_2.shape[1]))
    for i, (p1, p2) in enumerate(zip(pos_1, pos_2)):
        aln_coords_1[i] = middles_1[p1]
        aln_coords_2[i] = middles_2[p2]
    coords_1, coords_2, _ = rmsd_calculations.superpose_with_pos(coords_1, coords_2, aln_coords_1, aln_coords_2)
    return coords_1, coords_2


@nb.njit
def get_dtw_signal_rmsd_pos(coords_1, coords_2, gamma, index, gap_open_penalty, gap_extend_penalty):
    coords_1[:, :3], coords_2[:, :3] = dtw_signals_index(coords_1[:, :3], coords_2[:, :3], gamma, index, gap_open_penalty, gap_extend_penalty)
    distance_matrix = rmsd_calculations.make_distance_matrix(coords_1[:, :3], coords_2[:, :3],
                                                             gamma,
                                                             tm_score=False,
                                                             normalize=False)
    dtw_aln_array_1, dtw_aln_array_2, _ = dtw.dtw_align(distance_matrix, gap_open_penalty, gap_extend_penalty)
    pos_1, pos_2 = helper.get_common_positions(dtw_aln_array_1, dtw_aln_array_2)
    common_coords_1, common_coords_2 = coords_1[pos_1], coords_2[pos_2]
    rot, tran = rmsd_calculations.svd_superimpose(common_coords_1[:, :3], common_coords_2[:, :3])
    common_coords_2[:, :3] = rmsd_calculations.apply_rotran(common_coords_2[:, :3], rot, tran)
    return rmsd_calculations.get_exp_distances(common_coords_1, common_coords_2, gamma, False), pos_1, pos_2


@nb.njit
def get_secondary_distance_matrix(secondary_1, secondary_2, gap=0):
    score_matrix = np.zeros((secondary_1.shape[0], secondary_2.shape[0]))
    for i in range(secondary_1.shape[0]):
        for j in range(secondary_2.shape[0]):
            if secondary_1[i] == secondary_2[j]:
                if secondary_1[i] != gap:
                    score_matrix[i, j] = 1
            else:
                score_matrix[i, j] = -1
    return score_matrix


@nb.njit
def get_secondary_rmsd_pos(secondary_1, secondary_2, coords_1, coords_2, gamma, gap_open_sec, gap_extend_sec):
    distance_matrix = get_secondary_distance_matrix(secondary_1, secondary_2)
    dtw_aln_array_1, dtw_aln_array_2, _ = dtw.dtw_align(distance_matrix, gap_open_sec, gap_extend_sec)
    pos_1, pos_2 = helper.get_common_positions(dtw_aln_array_1, dtw_aln_array_2)
    common_coords_1, common_coords_2 = coords_1[pos_1][:, :3], coords_2[pos_2][:, :3]
    rot, tran = rmsd_calculations.svd_superimpose(common_coords_1, common_coords_2)
    common_coords_2 = rmsd_calculations.apply_rotran(common_coords_2, rot, tran)
    return rmsd_calculations.get_exp_distances(common_coords_1, common_coords_2, gamma, False), pos_1, pos_2


@nb.njit
def get_pairwise_alignment(coords_1, coords_2,
                           secondary_1, secondary_2,
                           gamma,
                           gap_open_sec, gap_extend_sec,
                           gap_open_penalty,
                           gap_extend_penalty, max_iter=100):
    rmsd_1, pos_1_1, pos_2_1 = get_dtw_signal_rmsd_pos(coords_1, coords_2, gamma, 0, gap_open_penalty, gap_extend_penalty)
    rmsd_2, pos_1_2, pos_2_2 = get_secondary_rmsd_pos(secondary_1, secondary_2, coords_1[:, :3], coords_2[:, :3], gamma, gap_open_sec, gap_extend_sec)
    rmsd_3, pos_1_3, pos_2_3 = get_dtw_signal_rmsd_pos(coords_1, coords_2, gamma, -1, gap_open_penalty, gap_extend_penalty)
    if rmsd_1 > rmsd_2:
        if rmsd_3 > rmsd_1:
            pos_1, pos_2 = pos_1_3, pos_2_3
        else:
            pos_1, pos_2 = pos_1_1, pos_2_1
    else:
        if rmsd_3 > rmsd_2:
            pos_1, pos_2 = pos_1_3, pos_2_3
        else:
            pos_1, pos_2 = pos_1_2, pos_2_2
    common_coords_1, common_coords_2 = coords_1[pos_1][:, :3], coords_2[pos_2][:, :3]
    coords_1[:, :3], coords_2[:, :3], _ = rmsd_calculations.superpose_with_pos(coords_1[:, :3], coords_2[:, :3],
                                                                               common_coords_1, common_coords_2)
    distance_matrix = rmsd_calculations.make_distance_matrix(coords_1, coords_2, gamma, tm_score=False, normalize=False)
    dtw_aln_array_1, dtw_aln_array_2, score = dtw.dtw_align(distance_matrix, gap_open_penalty, gap_extend_penalty)
    for i in range(max_iter):
        pos_1, pos_2 = helper.get_common_positions(dtw_aln_array_1, dtw_aln_array_2)
        common_coords_1, common_coords_2 = coords_1[pos_1][:, :3], coords_2[pos_2][:, :3]
        coords_1[:, :3], coords_2[:, :3], _ = rmsd_calculations.superpose_with_pos(coords_1[:, :3], coords_2[:, :3], common_coords_1, common_coords_2)

        distance_matrix = rmsd_calculations.make_distance_matrix(coords_1, coords_2,
                                                                 gamma,
                                                                 tm_score=False,
                                                                 normalize=False)
        dtw_1, dtw_2, new_score = dtw.dtw_align(distance_matrix, gap_open_penalty, gap_extend_penalty)
        if int(new_score) > int(score):
            dtw_aln_array_1, dtw_aln_array_2, score = dtw_1, dtw_2, new_score
        else:
            break
    return dtw_aln_array_1, dtw_aln_array_2, score
