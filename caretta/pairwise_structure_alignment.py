import typing

import numpy as np

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


class Structure:
    def __init__(self, name: str, sequence: typing.Union[str, None], coords: np.ndarray, features, make_feature_matrix=False, feature_names=(),
                 add_column=True):
        self.name = name
        self.sequence = sequence
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
        feature_matrix = np.zeros((self.coords.shape[0], len(feature_names)))
        pos = [x for x, s in enumerate(self.sequence) if s.upper() != 'X']
        for i, feature in enumerate(feature_names):
            feature_values = dssp_features[feature].astype(np.float64)[pos]
            # feature_matrix[:, i] = ndimage.gaussian_filter1d(helper.normalize(feature_values), sigma=2)
            feature_matrix[:, i] = helper.normalize(feature_values)
        return feature_matrix


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

    def get_dtw_coord_alignment(self, aln_array_1: np.ndarray, aln_array_2: np.ndarray, gap_open_penalty: float = 0., gap_extend_penalty=0.,
                                superimpose: bool = True):
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
        distance_matrix = rmsd_calculations.make_distance_matrix(self.structure_1.coords, coords_2, tm_score=False, normalize=False)
        dtw_aln_array_1, dtw_aln_array_2, score = dtw.dtw_align(distance_matrix, gap_open_penalty, gap_extend_penalty)
        if superimpose:
            for i in range(3):
                pos_1, pos_2 = helper.get_common_positions(dtw_aln_array_1, dtw_aln_array_2)
                common_coords_1, common_coords_2 = self.structure_1.coords[pos_1], self.structure_2.coords[pos_2]
                rotation_matrix, translation_matrix = rmsd_calculations.svd_superimpose(common_coords_1[:, :3], common_coords_2[:, :3])
                self.structure_2.coords[:, :3] = rmsd_calculations.apply_rotran(self.structure_2.coords[:, :3],
                                                                                rotation_matrix, translation_matrix)
                distance_matrix = rmsd_calculations.make_distance_matrix(self.structure_1.coords, self.structure_2.coords, tm_score=False,
                                                                         normalize=False)
                dtw_aln_array_1, dtw_aln_array_2, score = dtw.dtw_align(distance_matrix, gap_open_penalty, gap_extend_penalty)

        # pos_1, pos_2 = helper.get_common_positions(dtw_aln_array_1, dtw_aln_array_2)
        # plt.figure(figsize=(10, 10))
        # plt.imshow(distance_matrix, interpolation="nearest")
        # plt.colorbar()
        # plt.scatter(pos_2, pos_1, c='red', s=10, alpha=0.5)
        #
        # plt.pause(0.0001)
        return dtw_aln_array_1, dtw_aln_array_2, score
