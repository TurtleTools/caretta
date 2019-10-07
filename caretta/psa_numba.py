import matplotlib.pyplot as plt
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


# @nb.njit
def get_best_match_rmsd_pos(coords_1, coords_2, gap_open_penalty, gap_extend_penalty, split_size=40):
    split_size = np.min((split_size, coords_1.shape[0], coords_2.shape[0]))
    min_i, min_j = 0, 0
    min_rmsd = np.inf
    for i in range(coords_1.shape[0] - split_size + 1):
        for j in range(coords_2.shape[0] - split_size + 1):
            rmsd = rmsd_calculations.get_rmsd_superimposed(coords_1[i: i + split_size], coords_2[j: j + split_size])
            if rmsd < min_rmsd:
                min_rmsd = rmsd
                min_i, min_j = i, j
    coords_1_1, coords_2_1, _ = rmsd_calculations.superpose_with_pos(coords_1, coords_2, coords_1[min_i: min_i + split_size],
                                                                     coords_2[min_j: min_j + split_size])
    # rot, tran = rmsd_calculations.svd_superimpose(coords_1[min_i: min_i + split_size], coords_2[min_j: min_j + split_size])
    # coords_1_1 = rmsd_calculations.apply_rotran(coords_1, rot, tran)
    # coords_2_1 = rmsd_calculations.apply_rotran(coords_2, rot, tran)
    distance_matrix = rmsd_calculations.make_distance_matrix(coords_1_1, coords_2_1,
                                                             tm_score=False,
                                                             normalize=False)
    dtw_aln_array_1, dtw_aln_array_2, _ = dtw.dtw_align(distance_matrix, gap_open_penalty, gap_extend_penalty)
    pos_1, pos_2 = helper.get_common_positions(dtw_aln_array_1, dtw_aln_array_2)
    print("Best match")
    plt.imshow(distance_matrix)
    plt.colorbar()
    plt.plot(pos_2, pos_1, c="red")
    plt.pause(0.1)
    common_coords_1, common_coords_2 = coords_1_1[pos_1], coords_2_1[pos_2]
    rot, tran = rmsd_calculations.svd_superimpose(common_coords_1, common_coords_2)
    common_coords_2 = rmsd_calculations.apply_rotran(common_coords_2, rot, tran)
    return rmsd_calculations.get_exp_distances(common_coords_1, common_coords_2, False), pos_1, pos_2


def get_split_and_compare_rmsd_pos1(coords_1: np.ndarray, coords_2: np.ndarray, gap_open_penalty, gap_extend_penalty, split_size=10, gamma=0.15):
    # print(coords_1.shape, coords_2.shape)
    splits_1 = np.zeros((coords_1.shape[0] - split_size + 1, split_size, coords_1.shape[1]))
    splits_2 = np.zeros((coords_2.shape[0] - split_size + 1, split_size, coords_2.shape[1]))
    for x, i in enumerate(range(0, coords_1.shape[0] - split_size + 1)):
        splits_1[x] = coords_1[i: i + split_size]
    for x, i in enumerate(range(0, coords_2.shape[0] - split_size + 1)):
        splits_2[x] = coords_2[i: i + split_size]
    pw_matrix = np.zeros((splits_1.shape[0], splits_2.shape[0]))
    for i in range(pw_matrix.shape[0]):
        for j in range(pw_matrix.shape[1]):
            pw_matrix[i, j] = np.exp(-gamma * rmsd_calculations.get_rmsd_superimposed(splits_1[i], splits_2[j]))
    dtw_1, dtw_2, _ = dtw.dtw_align(pw_matrix, 0, 0)
    pos_1, pos_2 = helper.get_common_positions(dtw_1, dtw_2)

    # plt.imshow(pw_matrix)
    # plt.colorbar()
    # plt.plot(pos_2, pos_1, c="red")
    # plt.pause(0.1)
    aln_coords_1 = np.zeros((len(pos_1) * split_size, coords_1.shape[1]))
    aln_coords_2 = np.zeros((len(pos_2) * split_size, coords_2.shape[1]))
    shift = 0
    cont = True

    for i, (p1, p2) in enumerate(zip(pos_1, pos_2)):
        if (i > 1) and (p1 - pos_1[i - 1]) == 1 and (p2 - pos_2[i - 1]) == 1 and cont == True:
            aln_coords_1[shift + split_size] = splits_1[p1][-1]
            aln_coords_2[shift + split_size] = splits_2[p2][-1]
            shift += 1
            # print(shift)
        elif (i > 1) and (p1 - pos_1[i - 1]) == 1 and (p2 - pos_2[i - 1]) == 1 and cont == False:
            aln_coords_1[shift: shift + split_size] = splits_1[p1]
            aln_coords_2[shift:  shift + split_size] = splits_2[p2]
            shift += split_size
            cont = True
        elif cont == True:
            shift = max(shift - split_size + 1, 0)
            aln_coords_1[shift:] = 0
            aln_coords_2[shift:] = 0
            cont = False
        else:
            continue
    aln_coords_1 = aln_coords_1[:shift]
    aln_coords_2 = aln_coords_2[:shift]
    # print(aln_coords_1.shape, aln_coords_2.shape)
    coords_1_1, coords_2_1, _ = rmsd_calculations.superpose_with_pos(coords_1, coords_2, aln_coords_1, aln_coords_2)
    # rot, tran = rmsd_calculations.svd_superimpose(aln_coords_1, aln_coords_2)
    # coords_1_1 = rmsd_calculations.apply_rotran(coords_1, rot, tran)
    # coords_2_1 = rmsd_calculations.apply_rotran(coords_2, rot, tran)
    distance_matrix = rmsd_calculations.make_distance_matrix(coords_1_1, coords_2_1,
                                                             tm_score=False,
                                                             normalize=False)
    dtw_aln_array_1, dtw_aln_array_2, _ = dtw.dtw_align(distance_matrix, 0, 0)
    pos_1, pos_2 = helper.get_common_positions(dtw_aln_array_1, dtw_aln_array_2)
    print("split and compare")
    plt.imshow(distance_matrix)
    plt.colorbar()
    plt.plot(pos_2, pos_1, c="red")
    plt.pause(0.1)

    common_coords_1, common_coords_2 = coords_1_1[pos_1], coords_2_1[pos_2]
    rot, tran = rmsd_calculations.svd_superimpose(common_coords_1, common_coords_2)
    common_coords_2 = rmsd_calculations.apply_rotran(common_coords_2, rot, tran)
    return rmsd_calculations.get_exp_distances(common_coords_1, common_coords_2, False), pos_1, pos_2


@nb.njit
def get_split_and_compare_rmsd_pos(coords_1: np.ndarray, coords_2: np.ndarray, gap_open_penalty, gap_extend_penalty, split_size=50, gamma=0.15):
    splits_1 = np.zeros((coords_1.shape[0] // split_size, split_size, coords_1.shape[1]))
    splits_2 = np.zeros((coords_2.shape[0] // split_size, split_size, coords_2.shape[1]))
    for x, i in enumerate(range(0, coords_1.shape[0] - split_size, split_size)):
        splits_1[x] = coords_1[i: i + split_size]
    for x, i in enumerate(range(0, coords_2.shape[0] - split_size, split_size)):
        splits_2[x] = coords_2[i: i + split_size]
    pw_matrix = np.zeros((splits_1.shape[0], splits_2.shape[0]))
    for i in range(pw_matrix.shape[0]):
        for j in range(pw_matrix.shape[1]):
            pw_matrix[i, j] = np.exp(-gamma * rmsd_calculations.get_rmsd_superimposed(splits_1[i], splits_2[j]))
    dtw_1, dtw_2, _ = dtw.dtw_align(pw_matrix, 0, 0)
    pos_1, pos_2 = helper.get_common_positions(dtw_1, dtw_2)
    aln_coords_1 = np.zeros((len(pos_1) * split_size, coords_1.shape[1]))
    aln_coords_2 = np.zeros((len(pos_2) * split_size, coords_2.shape[1]))
    for i, (p1, p2) in enumerate(zip(pos_1, pos_2)):
        aln_coords_1[i * split_size: (i + 1) * split_size] = splits_1[p1]
        aln_coords_2[i * split_size: (i + 1) * split_size] = splits_2[p2]
    rot, tran = rmsd_calculations.svd_superimpose(aln_coords_1, aln_coords_2)
    coords_1_1 = rmsd_calculations.apply_rotran(coords_1, rot, tran)
    coords_2_1 = rmsd_calculations.apply_rotran(coords_2, rot, tran)
    distance_matrix = rmsd_calculations.make_distance_matrix(coords_1_1, coords_2_1,
                                                             tm_score=False,
                                                             normalize=False)
    dtw_aln_array_1, dtw_aln_array_2, _ = dtw.dtw_align(distance_matrix, gap_open_penalty, gap_extend_penalty)
    pos_1, pos_2 = helper.get_common_positions(dtw_aln_array_1, dtw_aln_array_2)
    common_coords_1, common_coords_2 = coords_1_1[pos_1], coords_2_1[pos_2]
    rot, tran = rmsd_calculations.svd_superimpose(common_coords_1, common_coords_2)
    common_coords_2 = rmsd_calculations.apply_rotran(common_coords_2, rot, tran)
    return rmsd_calculations.get_exp_distances(common_coords_1, common_coords_2, False), pos_1, pos_2


# @nb.njit
def get_slide_middle_rmsd_pos(coords_1, coords_2, gap_open_penalty, gap_extend_penalty):
    coords_1_1, coords_2_1 = rmsd_calculations.slide_middle(coords_1[:, :3],
                                                            coords_2[:, :3])
    distance_matrix = rmsd_calculations.make_distance_matrix(coords_1_1, coords_2_1,
                                                             tm_score=False,
                                                             normalize=False)
    dtw_aln_array_1, dtw_aln_array_2, _ = dtw.dtw_align(distance_matrix, gap_open_penalty, gap_extend_penalty)
    pos_1, pos_2 = helper.get_common_positions(dtw_aln_array_1, dtw_aln_array_2)
    print("Slide middle")
    plt.imshow(distance_matrix)
    plt.colorbar()
    plt.plot(pos_2, pos_1, c="red")
    plt.pause(0.1)
    common_coords_1, common_coords_2 = coords_1_1[pos_1], coords_2_1[pos_2]
    rot, tran = rmsd_calculations.svd_superimpose(common_coords_1, common_coords_2)
    common_coords_2 = rmsd_calculations.apply_rotran(common_coords_2, rot, tran)
    return rmsd_calculations.get_exp_distances(common_coords_1, common_coords_2, False), pos_1, pos_2


# @nb.njit
def get_slide_best_rmsd_pos(coords_1, coords_2, gap_open_penalty, gap_extend_penalty):
    coords_1_1, coords_2_1 = slide_best(coords_1[:, :3], coords_2[:, :3])
    distance_matrix = rmsd_calculations.make_distance_matrix(coords_1_1, coords_2_1,
                                                             tm_score=False,
                                                             normalize=False)
    dtw_aln_array_1, dtw_aln_array_2, _ = dtw.dtw_align(distance_matrix, gap_open_penalty, gap_extend_penalty)
    pos_1, pos_2 = helper.get_common_positions(dtw_aln_array_1, dtw_aln_array_2)
    print("Slide best")
    plt.imshow(distance_matrix)
    plt.colorbar()
    plt.plot(pos_2, pos_1, c="red")
    plt.pause(0.1)
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
    # print("Slide Padded")
    # plt.imshow(distance_matrix)
    # plt.colorbar()
    # plt.plot(pos_2, pos_1, c="red")
    # plt.pause(0.1)
    common_coords_1, common_coords_2 = coords_1_1[pos_1], coords_2_1[pos_2]
    rot, tran = rmsd_calculations.svd_superimpose(common_coords_1, common_coords_2)
    common_coords_2 = rmsd_calculations.apply_rotran(common_coords_2, rot, tran)
    return rmsd_calculations.get_exp_distances(common_coords_1, common_coords_2, False), pos_1, pos_2


# @nb.njit
def get_slide_rmsd_pos(coords_1, coords_2, gap_open_penalty, gap_extend_penalty):
    coords_1_1, coords_2_1 = rmsd_calculations.slide(coords_1[:, :3],
                                                     coords_2[:, :3])
    distance_matrix = rmsd_calculations.make_distance_matrix(coords_1_1, coords_2_1,
                                                             tm_score=False,
                                                             normalize=False)
    dtw_aln_array_1, dtw_aln_array_2, _ = dtw.dtw_align(distance_matrix, gap_open_penalty, gap_extend_penalty)
    pos_1, pos_2 = helper.get_common_positions(dtw_aln_array_1, dtw_aln_array_2)
    print("Slide")
    plt.imshow(distance_matrix)
    plt.colorbar()
    plt.plot(pos_2, pos_1, c="red")
    plt.pause(0.1)
    common_coords_1, common_coords_2 = coords_1_1[pos_1], coords_2_1[pos_2]
    rot, tran = rmsd_calculations.svd_superimpose(common_coords_1, common_coords_2)
    common_coords_2 = rmsd_calculations.apply_rotran(common_coords_2, rot, tran)
    return rmsd_calculations.get_exp_distances(common_coords_1, common_coords_2, False), pos_1, pos_2


@nb.njit
def get_dtw_signal_rmsd_pos(coords_1, coords_2, index, gap_open_penalty, gap_extend_penalty):
    coords_1_1, coords_2_1 = rmsd_calculations.dtw_signals_index(coords_1[:, :3],
                                                                 coords_2[:, :3], index, gap_open_penalty, gap_extend_penalty)
    distance_matrix = rmsd_calculations.make_distance_matrix(coords_1_1, coords_2_1,
                                                             tm_score=False,
                                                             normalize=False)
    dtw_aln_array_1, dtw_aln_array_2, _ = dtw.dtw_align(distance_matrix, gap_open_penalty, gap_extend_penalty)
    pos_1, pos_2 = helper.get_common_positions(dtw_aln_array_1, dtw_aln_array_2)
    # print("Signal")
    # plt.imshow(distance_matrix)
    # plt.colorbar()
    # plt.plot(pos_2, pos_1, c="red")
    # plt.pause(0.1)
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


# @nb.njit
def slide_best(coords_1, coords_2, size=20, num_best=5):
    signal_1s = np.zeros((coords_1.shape[0] // size, size))
    for x, i in enumerate(range(0, coords_1.shape[0] - size, size)):
        signal_1s[x] = rmsd_calculations.make_signal(coords_1[i: i + size])
    signal_2s = np.zeros((coords_2.shape[0] // size, size))
    for x, i in enumerate(range(0, coords_2.shape[0] - size, size)):
        signal_2s[x] = rmsd_calculations.make_signal(coords_2[i: i + size])

    pw_matrix = np.zeros((signal_1s.shape[0], signal_2s.shape[0]))
    for i in range(pw_matrix.shape[0]):
        for j in range(pw_matrix.shape[1]):
            pw_matrix[i, j] = np.dot(signal_1s[i], signal_2s[j])
    dtw_aln_array_1, dtw_aln_array_2, _ = dtw.dtw_align(pw_matrix, 0, 0)
    pos_1, pos_2 = helper.get_common_positions(dtw_aln_array_1, dtw_aln_array_2)
    plt.imshow(pw_matrix)
    plt.colorbar()
    plt.plot(pos_2, pos_1, c="red")
    plt.pause(0.1)
    aln_coords_1 = np.zeros((len(pos_1) * size, coords_1.shape[1]))
    aln_coords_2 = np.zeros((len(pos_2) * size, coords_2.shape[1]))
    for i, (p1, p2) in enumerate(zip(pos_1, pos_2)):
        aln_coords_1[i * size: (i + 1) * size] = coords_1[p1 * size: (p1 + 1) * size]
        aln_coords_2[i * size: (i + 1) * size] = coords_2[p2 * size: (p2 + 1) * size]
    rot, tran = rmsd_calculations.svd_superimpose(aln_coords_1, aln_coords_2)
    # best_c_1 = np.zeros((num_best * size, coords_1.shape[1]))
    # best_c_2 = np.zeros((num_best * size, coords_2.shape[1]))
    # inds = np.argsort(-pw_matrix.flatten())
    # for i in range(num_best):
    #     max_i, max_j = np.unravel_index(inds[i], pw_matrix.shape)
    #     best_c_1[i*size: (i+1)*size] = coords_1[max_i: max_i+size]
    #     best_c_2[i * size: (i + 1) * size] = coords_2[max_j: max_j + size]
    # rot, tran = rmsd_calculations.svd_superimpose(best_c_1, best_c_2)
    coords_1 = rmsd_calculations.apply_rotran(coords_1, rot, tran)
    coords_2 = rmsd_calculations.apply_rotran(coords_2, rot, tran)
    return coords_1, coords_2


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
    rmsd_1, pos_1_1, pos_2_1 = get_dtw_signal_rmsd_pos(coords_1[:, :3], coords_2[:, :3], 0, gap_open_penalty, gap_extend_penalty)
    rmsd_2, pos_1_2, pos_2_2 = get_secondary_rmsd_pos(secondary_1, secondary_2, coords_1[:, :3], coords_2[:, :3], gap_open_sec, gap_extend_sec)
    rmsd_3, pos_1_3, pos_2_3 = get_dtw_signal_rmsd_pos(coords_1[:, :3], coords_2[:, :3], -1, gap_open_penalty, gap_extend_penalty)
    # rmsd_3, pos_1_3, pos_2_3 = get_split_and_compare_rmsd_pos1(coords_1[:, :3], coords_2[:, :3], gap_open_penalty, gap_extend_penalty)
    # rmsd_4, pos_1_4, pos_2_4 = get_best_match_rmsd_pos(coords_1[:, :3], coords_2[:, :3], gap_open_penalty, gap_extend_penalty)
    # print(rmsd_1, rmsd_2, rmsd_3)
    # rmsds = np.array([rmsd_1, rmsd_2, rmsd_3, rmsd_4])
    # poses = np.array([(pos_1_1, pos_2_1), (pos_1_2, pos_2_2), (pos_1_3, pos_2_3), (pos_1_4, pos_2_4)])
    # index = np.argmax(rmsds)
    # pos_1, pos_2 = poses[index]
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
    # pos_1, pos_2 = pos_1_3, pos_2_3
    common_coords_1, common_coords_2 = coords_1[pos_1][:, :3], coords_2[pos_2][:, :3]
    coords_1[:, :3], coords_2[:, :3], common_coords_2 = rmsd_calculations.superpose_with_pos(coords_1[:, :3], coords_2[:, :3],
                                                                                             common_coords_1, common_coords_2)
    distance_matrix = rmsd_calculations.make_distance_matrix(coords_1, coords_2, tm_score=False, normalize=False)
    dtw_aln_array_1, dtw_aln_array_2, score = dtw.dtw_align(distance_matrix, gap_open_penalty, gap_extend_penalty)
    for i in range(3):
        pos_1, pos_2 = helper.get_common_positions(dtw_aln_array_1, dtw_aln_array_2)
        # print("final")
        # plt.imshow(distance_matrix)
        # plt.colorbar()
        # plt.plot(pos_2, pos_1, c="red")
        # plt.show()
        common_coords_1, common_coords_2 = coords_1[pos_1][:, :3], coords_2[pos_2][:, :3]
        coords_1[:, :3], coords_2[:, :3], common_coords_2 = rmsd_calculations.superpose_with_pos(coords_1[:, :3], coords_2[:, :3],
                                                                                                 common_coords_1, common_coords_2)

        distance_matrix = rmsd_calculations.make_distance_matrix(coords_1, coords_2,
                                                                 tm_score=False,
                                                                 normalize=False)
        dtw_aln_array_1, dtw_aln_array_2, score = dtw.dtw_align(distance_matrix, gap_open_penalty, gap_extend_penalty)
    return dtw_aln_array_1, dtw_aln_array_2, score
