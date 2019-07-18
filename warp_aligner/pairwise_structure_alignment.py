import numpy as np
import numba as nb
import typing
from warp_aligner import dynamic_time_warping as dtw
from warp_aligner import rmsd_calculations, helper


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
    def __init__(self, name: str, sequence: typing.Union[str, None], coords: np.ndarray):
        self.name = name
        self.sequence = sequence
        self.coords = coords


class StructurePair:
    def __init__(self, structure_1: Structure, structure_2: Structure):
        self.structure_1 = structure_1
        self.structure_2 = structure_2

    def get_common_coordinates(self, aln_sequence_1: typing.Union[str, np.ndarray], aln_sequence_2: typing.Union[str, np.ndarray], gap=-1):
        """
        Gets coordinates according to an alignment where neither position is a gap

        Parameters
        ----------
        aln_sequence_1
        aln_sequence_2
        gap
            character representing gaps (-1 for array, '-' for string)

        Returns
        -------
        common_coords_1, common_coords_2
        """
        assert len(aln_sequence_1) == len(aln_sequence_2)
        pos_1, pos_2 = helper.get_common_positions(aln_sequence_1, aln_sequence_2, gap)
        return self.structure_1.coords[pos_1], self.structure_2.coords[pos_2]

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
        rmsd = rmsd_calculations.get_rmsd_superimposed(common_coords_1, common_coords_2)
        return RMSD(rmsd, common_coords_1.shape[0], aln_array_1, aln_array_2, gap=gap)

    def get_dtw_alignment(self, aln_array_1: np.ndarray, aln_array_2: np.ndarray, gap_open_penalty: float = 0., gap_extend_penalty=0.):
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

        Returns
        -------
        aligned_indices_1, aligned_indices_2
        """
        pos_1, pos_2 = helper.get_common_positions(aln_array_1, aln_array_2)
        common_coords_1, common_coords_2 = self.structure_1.coords[pos_1], self.structure_2.coords[pos_2]
        rotation_matrix, translation_matrix = rmsd_calculations.svd_superimpose(common_coords_1, common_coords_2)
        coords_2 = rmsd_calculations.apply_rotran(self.structure_2.coords[pos_2], rotation_matrix, translation_matrix)
        distance_matrix = rmsd_calculations.make_euclidean_matrix(self.structure_1.coords[pos_1], coords_2)
        dtw_aln_array_1, dtw_aln_array_2 = dtw.dtw_align(distance_matrix, gap_open_penalty, gap_extend_penalty)
        return dtw_aln_array_1, dtw_aln_array_2
