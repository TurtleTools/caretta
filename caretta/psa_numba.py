import numba as nb
import numpy as np

from caretta import dynamic_time_warping as dtw
from caretta import rmsd_calculations, helper


def string_to_array(secondary):
    return np.array(secondary, dtype='S1').view(np.int8)


@nb.njit
def get_common_coordinates(coords_1, coords_2, aln_1, aln_2, gap=-1):
    assert aln_1.shape == aln_2.shape
    pos_1, pos_2 = helper.get_common_positions(aln_1, aln_2, gap)
    return coords_1[pos_1], coords_2[pos_2]


@nb.njit
def get_slide_rmsd_pos(coords_1, coords_2, gap_open_penalty, gap_extend_penalty):
    coords_1_1, coords_2_1 = rmsd_calculations.slide(coords_1[:, :3],
                                                     coords_2[:, :3])
    distance_matrix = rmsd_calculations.make_distance_matrix(coords_1_1, coords_2_1,
                                                             tm_score=False,
                                                             normalize=False)
    dtw_aln_array_1, dtw_aln_array_2, _ = dtw.dtw_align(distance_matrix, gap_open_penalty, gap_extend_penalty)
    pos_1, pos_2 = helper.get_common_positions(dtw_aln_array_1, dtw_aln_array_2)
    common_coords_1, common_coords_2 = coords_1_1[pos_1], coords_2_1[pos_2]
    rot, tran = rmsd_calculations.svd_superimpose(common_coords_1, common_coords_2)
    common_coords_2 = rmsd_calculations.apply_rotran(common_coords_2, rot, tran)
    return rmsd_calculations.get_exp_distances(common_coords_1, common_coords_2, False), pos_1, pos_2


@nb.njit
def get_slide_rmsd_pos_2(coords_1, coords_2, gap_open_penalty, gap_extend_penalty):
    coords_1_1, coords_2_1 = rmsd_calculations.slide1(coords_1[:, :3],
                                                      coords_2[:, :3])
    distance_matrix = rmsd_calculations.make_distance_matrix(coords_1_1, coords_2_1,
                                                             tm_score=False,
                                                             normalize=False)
    dtw_aln_array_1, dtw_aln_array_2, _ = dtw.dtw_align(distance_matrix, gap_open_penalty, gap_extend_penalty)
    pos_1, pos_2 = helper.get_common_positions(dtw_aln_array_1, dtw_aln_array_2)
    common_coords_1, common_coords_2 = coords_1_1[pos_1], coords_2_1[pos_2]
    rot, tran = rmsd_calculations.svd_superimpose(common_coords_1, common_coords_2)
    common_coords_2 = rmsd_calculations.apply_rotran(common_coords_2, rot, tran)
    return rmsd_calculations.get_exp_distances(common_coords_1, common_coords_2, False), pos_1, pos_2


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
def get_secondary_rmsd_pos(secondary_1, secondary_2, coords_1, coords_2, gap_open_sec, gap_extend_sec):
    distance_matrix = get_secondary_distance_matrix(secondary_1, secondary_2)
    dtw_aln_array_1, dtw_aln_array_2, _ = dtw.dtw_align(distance_matrix, gap_open_sec, gap_extend_sec)
    pos_1, pos_2 = helper.get_common_positions(dtw_aln_array_1, dtw_aln_array_2)
    common_coords_1, common_coords_2 = coords_1[pos_1][:, :3], coords_2[pos_2][:, :3]
    rot, tran = rmsd_calculations.svd_superimpose(common_coords_1, common_coords_2)
    common_coords_2 = rmsd_calculations.apply_rotran(common_coords_2, rot, tran)
    return rmsd_calculations.get_exp_distances(common_coords_1, common_coords_2, False), pos_1, pos_2


@nb.njit
def get_pairwise_alignment(coords_1, coords_2,
                           secondary_1, secondary_2,
                           gap_open_sec, gap_extend_sec,
                           gap_open_penalty,
                           gap_extend_penalty):
    rmsd_1, pos_1_1, pos_2_1 = get_slide_rmsd_pos(coords_1, coords_2, gap_open_penalty, gap_extend_penalty)
    rmsd_2, pos_1_2, pos_2_2 = get_secondary_rmsd_pos(secondary_1, secondary_2, coords_1, coords_2, gap_open_sec, gap_extend_sec)
    rmsd_3, pos_1_3, pos_2_3 = get_slide_rmsd_pos_2(coords_1, coords_2, gap_open_penalty, gap_extend_penalty)
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
    common_coords_1, common_coords_2 = coords_1[pos_1], coords_2[pos_2]
    rotation_matrix, translation_matrix = rmsd_calculations.svd_superimpose(common_coords_1[:, :3], common_coords_2[:, :3])
    coords_2[:, :3] = rmsd_calculations.apply_rotran(coords_2[:, :3], rotation_matrix, translation_matrix)
    distance_matrix = rmsd_calculations.make_distance_matrix(coords_1, coords_2, tm_score=False, normalize=False)
    dtw_aln_array_1, dtw_aln_array_2, score = dtw.dtw_align(distance_matrix, gap_open_penalty, gap_extend_penalty)
    for i in range(2):
        pos_1, pos_2 = helper.get_common_positions(dtw_aln_array_1, dtw_aln_array_2)
        common_coords_1, common_coords_2 = coords_1[pos_1], coords_2[pos_2]
        rotation_matrix, translation_matrix = rmsd_calculations.svd_superimpose(common_coords_1[:, :3], common_coords_2[:, :3])
        coords_2[:, :3] = rmsd_calculations.apply_rotran(coords_2[:, :3],
                                                         rotation_matrix, translation_matrix)
        distance_matrix = rmsd_calculations.make_distance_matrix(coords_1, coords_2,
                                                                 tm_score=False,
                                                                 normalize=False)
        dtw_aln_array_1, dtw_aln_array_2, score = dtw.dtw_align(distance_matrix, gap_open_penalty, gap_extend_penalty)
    return dtw_aln_array_1, dtw_aln_array_2, score
