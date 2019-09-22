import typing

import matplotlib.pyplot as plt
import numba as nb
import numpy as np
from scipy import ndimage

from caretta import dynamic_time_warping as dtw
from caretta import rmsd_calculations, helper


class RMSD:
    def __init__(self, rmsd, num_common_coords, aln_sequence_1, aln_sequence_2, gap=-1):
        self.rmsd = rmsd
        self.num_common_coords = num_common_coords
        self.coverage_1 = num_common_coords / len([a for a in aln_sequence_1 if a != gap])
        self.coverage_2 = num_common_coords / len([a for a in aln_sequence_2 if a != gap])
        self.coverage_aln = num_common_coords / len(aln_sequence_1)
        self.aln_sequence_1 = aln_sequence_1
        self.aln_sequence_2 = aln_sequence_2


def string_to_array(secondary):
    return np.array(secondary, dtype='S1').view(np.int8)


class Structure:
    def __init__(self, name: str, sequence: typing.Union[str, None], coords: np.ndarray, secondary, features, make_feature_matrix=False,
                 feature_names=(),
                 add_column=True):
        self.name = name
        self.sequence = sequence
        self.secondary = secondary
        if add_column:
            self.coords = np.hstack((coords, np.zeros((coords.shape[0], 1))))

        else:
            self.coords = coords
        if make_feature_matrix:
            self.features = self.get_feature_matrix(features, feature_names)
            if add_column:
                self.features = np.hstack((self.features, np.zeros((self.features.shape[0], 1))))
        else:
            self.features = features
            if add_column:
                self.features = np.hstack((self.features, np.zeros((self.features.shape[0], 1))))

    def get_feature_matrix(self, dssp_features, feature_names):
        feature_names = [f for f in feature_names if f != "secondary"]
        feature_matrix = np.zeros((self.coords.shape[0], len(feature_names)))
        pos = [x for x, s in enumerate(self.sequence) if s.upper() != 'X']
        for i, feature in enumerate(feature_names):
            feature_values = dssp_features[feature].astype(np.float64)[pos]
            feature_matrix[:, i] = ndimage.gaussian_filter1d(helper.normalize(feature_values), sigma=2)
            # feature_matrix[:, i] = helper.normalize(feature_values)
        return feature_matrix


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


class StructurePair:
    def __init__(self, structure_1: Structure, structure_2: Structure):
        self.structure_1 = structure_1
        self.structure_2 = structure_2

    def get_common_coordinates(self, aln_array_1: np.ndarray, aln_array_2: np.ndarray, gap=-1, get_coords=True):
        """
        Gets coordinates according to an alignment where neither position is a gap

        Parameters
        ----------
        aln_array_1
        aln_array_2
        gap
            character representing gaps (-1 for array, '-' for string)

        Returns
        -------
        common_coords_1, common_coords_2
        """
        assert aln_array_1.shape == aln_array_2.shape
        pos_1, pos_2 = helper.get_common_positions(aln_array_1, aln_array_2, gap)
        if get_coords:
            return self.structure_1.coords[[p for p in pos_1 if p < self.structure_1.coords.shape[0]]], \
                   self.structure_2.coords[[p for p in pos_2 if p < self.structure_2.coords.shape[0]]]
        else:
            return self.structure_1.features[[p for p in pos_1 if p < self.structure_1.features.shape[0]]], \
                   self.structure_2.features[[p for p in pos_2 if p < self.structure_2.features.shape[0]]]

    def get_rmsd_coverage(self, aln_array_1, aln_array_2, gap=-1) -> RMSD:
        """
        Finds SVD superimposed RMSD of two sets of coordinates
        aligned according to aln_sequence_1 and aln_sequence_2

        RMSD is calculated only on positions where neither sequence has a gap

        Parameters
        ----------
        aln_array_1
            aligned indices of first protein
        aln_array_2
            aligned indices of second protein
        gap
            gap character (-1 if array, '-' if string)

        Returns
        ------- 
        RMSD of paired coordinates,
        coverage of first protein (number of positions used in RMSD calculation / total number of positions),
        coverage of second protein,
        coverage of alignment (number of positions used in RMSD calculation / length of alignment)
        """
        common_coords_1, common_coords_2 = self.get_common_coordinates(aln_array_1, aln_array_2)
        try:
            rmsd = rmsd_calculations.get_rmsd_superimposed(common_coords_1, common_coords_2)
        except ZeroDivisionError:
            rmsd = 999
        return RMSD(rmsd, common_coords_1.shape[0], aln_array_1, aln_array_2, gap=gap)

    def get_exp_distances(self, aln_array_1, aln_array_2, normalize=True):
        common_coords_1, common_coords_2 = self.get_common_coordinates(aln_array_1, aln_array_2)
        return rmsd_calculations.get_exp_distances(common_coords_1[:, :3], common_coords_2[:, :3], normalize)

    def get_exp_feature_distances(self, aln_array_1, aln_array_2):
        pos_1, pos_2 = helper.get_common_positions(aln_array_1, aln_array_2, -1)
        return rmsd_calculations.get_exp_distances(self.structure_1.features[pos_1], self.structure_2.features[pos_2])

    def get_tm_score(self, aln_array_1, aln_array_2, superimpose=True):
        common_coords_1, common_coords_2 = self.get_common_coordinates(aln_array_1, aln_array_2)
        if superimpose:
            rotation_matrix, translation_matrix = rmsd_calculations.svd_superimpose(common_coords_1, common_coords_2)
            common_coords_2 = rmsd_calculations.apply_rotran(common_coords_2, rotation_matrix, translation_matrix)
        l_min = min(self.structure_1.coords.shape[0], self.structure_2.coords.shape[0])
        if l_min > 21:
            d0 = 1.24 * (l_min - 15) ** (1 / 3) - 1.8
        else:
            d0 = 0.5
        return (1 / l_min) * sum(
            1 / (1 + (np.sum((common_coords_1[i] - common_coords_2[i]) ** 2, axis=-1) / d0) ** 2) for i in range(common_coords_1.shape[0]))

    def get_dtw_secondary_alignment(self, gap_open=1, gap_extend=1):
        distance_matrix = get_secondary_distance_matrix(self.structure_1.secondary, self.structure_2.secondary)
        dtw_aln_array_1, dtw_aln_array_2, score = dtw.dtw_align(distance_matrix, gap_open, gap_extend)
        return dtw_aln_array_1, dtw_aln_array_2, score

    def get_dtw_feature_alignment(self, gap_open_feature: float = 1., gap_extend_feature: float = 1):
        distance_matrix = rmsd_calculations.make_distance_matrix(self.structure_1.features, self.structure_2.features, normalize=False)
        dtw_aln_array_1, dtw_aln_array_2, score = dtw.dtw_align(distance_matrix, gap_open_feature, gap_extend_feature)
        # pos_1, pos_2 = helper.get_common_positions(dtw_aln_array_1, dtw_aln_array_2)
        # plt.figure(figsize=(10, 10))
        # plt.imshow(distance_matrix, interpolation="nearest")
        # plt.colorbar()
        # plt.scatter(pos_2, pos_1, c='red', s=10, alpha=0.5)
        #
        # plt.pause(0.0001)
        return dtw_aln_array_1, dtw_aln_array_2, score

    def get_dtw_feature_coord_alignment(self, gap_open_feature: float = 1., gap_extend_feature: float = 1., gap_open_coord: float = 10.,
                                        gap_extend_coord: float = 2.):
        feature_aln_1, feature_aln_2, score = self.get_dtw_feature_alignment(gap_open_feature, gap_extend_feature)
        return self.get_dtw_coord_alignment(feature_aln_1, feature_aln_2, gap_open_coord, gap_extend_coord)

    def get_slide_rmsd_pos(self, gap_open_penalty, gap_extend_penalty):
        coords_1, coords_2 = rmsd_calculations.slide(self.structure_1.coords[:, :3], self.structure_2.coords[:, :3])
        distance_matrix = rmsd_calculations.make_distance_matrix(coords_1, coords_2,
                                                                 tm_score=False,
                                                                 normalize=False)
        dtw_aln_array_1, dtw_aln_array_2, score = dtw.dtw_align(distance_matrix, gap_open_penalty, gap_extend_penalty)
        pos_1, pos_2 = helper.get_common_positions(dtw_aln_array_1, dtw_aln_array_2)
        common_coords_1, common_coords_2 = coords_1[pos_1], coords_2[pos_2]
        return rmsd_calculations.get_rmsd_superimposed(common_coords_1, common_coords_2), pos_1, pos_2

    def get_secondary_rmsd_pos(self, gap_open_sec, gap_extend_sec):
        distance_matrix = get_secondary_distance_matrix(self.structure_1.secondary, self.structure_2.secondary)
        aln_array_1, aln_array_2, _ = dtw.dtw_align(distance_matrix, gap_open_sec, gap_extend_sec)
        pos_1, pos_2 = helper.get_common_positions(aln_array_1, aln_array_2)
        common_coords_1, common_coords_2 = self.structure_1.coords[pos_1], self.structure_2.coords[pos_2]
        return rmsd_calculations.get_rmsd_superimposed(common_coords_1, common_coords_2), pos_1, pos_2

    def get_dtw_coord_secondary_alignment(self, gap_open_sec=3, gap_extend_sec=1,
                                          gap_open_penalty: float = 0.,
                                          gap_extend_penalty=0.,
                                          superimpose: bool = True, plot=False):

        rmsd_1, pos_1_1, pos_2_1 = self.get_slide_rmsd_pos(gap_open_penalty, gap_extend_penalty)
        rmsd_2, pos_1_2, pos_2_2 = self.get_secondary_rmsd_pos(gap_open_sec, gap_extend_sec)
        print(rmsd_1, rmsd_2)
        if rmsd_1 < rmsd_2:
            pos_1, pos_2 = pos_1_1, pos_2_1
        else:
            pos_1, pos_2 = pos_1_2, pos_2_2
        # distance_matrix = get_secondary_distance_matrix(self.structure_1.secondary, self.structure_2.secondary)
        # aln_array_1, aln_array_2, _ = dtw.dtw_align(distance_matrix, gap_open_sec, gap_extend_sec)
        # coords_2 = self.structure_2.coords.copy()
        # if superimpose:
        #     pos_1, pos_2 = helper.get_common_positions(aln_array_1, aln_array_2)
        #     common_coords_1, common_coords_2 = self.structure_1.coords[pos_1], self.structure_2.coords[pos_2]
        #     print("Secondary", rmsd_calculations.get_rmsd_superimposed(common_coords_1, common_coords_2))
        #     rotation_matrix, translation_matrix = rmsd_calculations.svd_superimpose(common_coords_1[:, :3], common_coords_2[:, :3])
        #     coords_2[:, :3] = rmsd_calculations.apply_rotran(self.structure_2.coords[:, :3], rotation_matrix, translation_matrix)
        common_coords_1, common_coords_2 = self.structure_1.coords[pos_1], self.structure_2.coords[pos_2]
        rotation_matrix, translation_matrix = rmsd_calculations.svd_superimpose(common_coords_1[:, :3], common_coords_2[:, :3])
        self.structure_2.coords[:, :3] = rmsd_calculations.apply_rotran(self.structure_2.coords[:, :3], rotation_matrix, translation_matrix)
        distance_matrix = rmsd_calculations.make_distance_matrix(self.structure_1.coords, self.structure_2.coords, tm_score=False, normalize=False)
        dtw_aln_array_1, dtw_aln_array_2, score = dtw.dtw_align(distance_matrix, gap_open_penalty, gap_extend_penalty)
        if superimpose:
            for i in range(2):
                pos_1, pos_2 = helper.get_common_positions(dtw_aln_array_1, dtw_aln_array_2)
                common_coords_1, common_coords_2 = self.structure_1.coords[pos_1], self.structure_2.coords[pos_2]
                rotation_matrix, translation_matrix = rmsd_calculations.svd_superimpose(common_coords_1[:, :3], common_coords_2[:, :3])
                self.structure_2.coords[:, :3] = rmsd_calculations.apply_rotran(self.structure_2.coords[:, :3],
                                                                                rotation_matrix, translation_matrix)
                distance_matrix = rmsd_calculations.make_distance_matrix(self.structure_1.coords, self.structure_2.coords, tm_score=False,
                                                                         normalize=False)
                dtw_aln_array_1, dtw_aln_array_2, score = dtw.dtw_align(distance_matrix, gap_open_penalty, gap_extend_penalty)
        if plot:
            pos_1, pos_2 = helper.get_common_positions(dtw_aln_array_1, dtw_aln_array_2)
            plt.figure(figsize=(10, 10))
            plt.imshow(distance_matrix, interpolation="nearest")
            plt.colorbar()
            plt.scatter(pos_2, pos_1, c='red', s=10, alpha=0.5)

            plt.pause(0.0001)
        return dtw_aln_array_1, dtw_aln_array_2, score

    def get_dtw_coord_alignment(self, aln_array_1: np.ndarray, aln_array_2: np.ndarray, gap_open_penalty: float = 0., gap_extend_penalty=0.,
                                superimpose: bool = True, plot=False):
        """
        Aligns two sets of coordinates using dynamic time warping
        aln_array_1 and aln_array_2 are used to find the initial rotation/translation

        Parameters
        ----------
        aln_array_1
            initial alignment of first set
        aln_array_2
            initial alignment of second set
        gap_open_penalty
            penalty for opening a (series of) gap(s)
        gap_extend_penalty
            penalty for extending an existing series of gaps
        superimpose
            if True, superimposes coords based on aln_array_1 and aln_array_2 before running dtw
        Returns
        -------
        aligned_indices_1, aligned_indices_2
        """
        coords_2 = self.structure_2.coords.copy()
        if superimpose:
            pos_1, pos_2 = helper.get_common_positions(aln_array_1, aln_array_2)
            common_coords_1, common_coords_2 = self.structure_1.coords[pos_1], self.structure_2.coords[pos_2]
            rotation_matrix, translation_matrix = rmsd_calculations.svd_superimpose(common_coords_1[:, :3], common_coords_2[:, :3])
            coords_2[:, :3] = rmsd_calculations.apply_rotran(self.structure_2.coords[:, :3], rotation_matrix, translation_matrix)
        # feature_distance_matrix = 0. * rmsd_calculations.make_distance_matrix(self.structure_1.features, self.structure_2.features, normalize=False)
        distance_matrix = rmsd_calculations.make_distance_matrix(self.structure_1.coords, coords_2, tm_score=False, normalize=False)
        # distance_matrix += feature_distance_matrix
        dtw_aln_array_1, dtw_aln_array_2, score = dtw.dtw_align(distance_matrix, gap_open_penalty, gap_extend_penalty)
        if superimpose:
            for i in range(2):
                pos_1, pos_2 = helper.get_common_positions(dtw_aln_array_1, dtw_aln_array_2)
                common_coords_1, common_coords_2 = self.structure_1.coords[pos_1], self.structure_2.coords[pos_2]
                rotation_matrix, translation_matrix = rmsd_calculations.svd_superimpose(common_coords_1[:, :3], common_coords_2[:, :3])
                self.structure_2.coords[:, :3] = rmsd_calculations.apply_rotran(self.structure_2.coords[:, :3],
                                                                                rotation_matrix, translation_matrix)
                distance_matrix = rmsd_calculations.make_distance_matrix(self.structure_1.coords, self.structure_2.coords, tm_score=False,
                                                                         normalize=False)
                # distance_matrix += feature_distance_matrix
                dtw_aln_array_1, dtw_aln_array_2, score = dtw.dtw_align(distance_matrix, gap_open_penalty, gap_extend_penalty)
        if plot:
            pos_1, pos_2 = helper.get_common_positions(dtw_aln_array_1, dtw_aln_array_2)
            plt.figure(figsize=(10, 10))
            plt.imshow(distance_matrix, interpolation="nearest")
            plt.colorbar()
            plt.scatter(pos_2, pos_1, c='red', s=10, alpha=0.5)

            plt.pause(0.0001)
        return dtw_aln_array_1, dtw_aln_array_2, score
