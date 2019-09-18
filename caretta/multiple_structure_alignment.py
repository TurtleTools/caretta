import typing

import numba as nb
import numpy as np

from caretta import neighbor_joining as nj
from caretta import pairwise_structure_alignment as psa
from caretta import rmsd_calculations, helper


@nb.njit
def get_mean_coords(aln_coords_1: np.ndarray, aln_coords_2: np.ndarray) -> np.ndarray:
    """
    Mean of two coordinate sets (of the same shape)

    Parameters
    ----------
    aln_coords_1
    aln_coords_2

    Returns
    -------
    mean_coords
    """
    mean_coords = np.zeros(aln_coords_1.shape)
    for i in range(aln_coords_1.shape[0]):
        mean_coords[i] = np.array([np.nanmean(np.array([aln_coords_1[i, x], aln_coords_2[i, x]])) for x in range(aln_coords_1.shape[1])])
    return mean_coords


class StructureMultiple:
    def __init__(self, structures: typing.List[psa.Structure]):
        self.structures = [s for s in structures]
        self.final_structures = []
        self.num_structures = len(structures)
        self.tree = None
        self.branch_lengths = None

    def make_pairwise_dtw_score_matrix(self, gap_open_penalty: float = 1., gap_extend_penalty: float = 1.):
        num = self.num_structures
        pairwise_matrix = np.zeros((num, num))
        pairwise_alns = {}
        for i in range(num):
            for j in range(i, num):
                name_1, name_2 = self.structures[i].name, self.structures[j].name
                structure_pair = psa.StructurePair(self.structures[i], self.structures[j])
                dtw_aln_1, dtw_aln_2, score = structure_pair.get_dtw_feature_alignment(gap_open_feature=gap_open_penalty,
                                                                                       gap_extend_feature=gap_extend_penalty)
                pairwise_matrix[i, j] = score
                pairwise_matrix[j, i] = score
                pairwise_alns[(name_1, name_2)] = (dtw_aln_1, dtw_aln_2)
                pairwise_alns[(name_2, name_1)] = (dtw_aln_2, dtw_aln_1)
        return pairwise_matrix, pairwise_alns

    def make_pairwise_rmsd_matrix(self, alignments: dict, superimpose_first: bool = True):
        """
        Find RMSDs of pairwise alignment of each pair of sequences

        Parameters
        ----------
        alignments
        superimpose_first
            if True then superimposes all structures to first structure first

        Returns
        -------
        RMSD matrix, coverage matrix
        """
        num = self.num_structures
        pairwise_rmsd_matrix = np.zeros((num, num))
        pairwise_coverage = np.zeros((num, num))
        if superimpose_first:
            structures = [psa.Structure(self.structures[0].name,
                                        self.structures[0].sequence,
                                        self.structures[0].coords,
                                        self.structures[0].features)]
            for s in self.structures[1:]:
                pos_1, pos_2 = helper.get_common_positions(helper.aligned_string_to_array(alignments[structures[0].name]),
                                                           helper.aligned_string_to_array(alignments[s.name]))
                common_coords_1, common_coords_2 = structures[0].coords[pos_1], s.coords[pos_2]
                rotation_matrix, translation_matrix = rmsd_calculations.svd_superimpose(common_coords_1, common_coords_2)
                coords_2 = rmsd_calculations.apply_rotran(s.coords, rotation_matrix, translation_matrix)
                structures.append(psa.Structure(s.name, s.sequence, coords_2, s.features))
        else:
            structures = [s for s in self.structures]
        for i in range(num):
            for j in range(i, num):
                name_1, name_2 = structures[i].name, structures[j].name
                structure_pair = psa.StructurePair(structures[i], structures[j])
                if isinstance(alignments[name_1], str):
                    aln_1 = helper.aligned_string_to_array(alignments[name_1])
                    aln_2 = helper.aligned_string_to_array(alignments[name_2])
                else:
                    aln_1 = alignments[name_1]
                    aln_2 = alignments[name_2]
                common_coords_1, common_coords_2 = structure_pair.get_common_coordinates(aln_1, aln_2)
                if superimpose_first:
                    rmsd = rmsd_calculations.get_rmsd(common_coords_1, common_coords_2)
                else:
                    try:
                        rmsd = rmsd_calculations.get_rmsd_superimposed(common_coords_1, common_coords_2)
                    except ZeroDivisionError:
                        rmsd = 999
                pairwise_rmsd_matrix[i, j] = rmsd
                pairwise_coverage[i, j] = len(aln_1) / common_coords_1.shape[0]
        return pairwise_rmsd_matrix, pairwise_coverage

    def make_pairwise_dtw_rmsd_matrix(self, alignments: dict, superimpose: bool = True, gap_open_penalty: float = 0.,
                                      gap_extend_penalty: float = 0.):
        """
        Find RMSDs of pairwise alignment of each pair of sequences

        Parameters
        ----------
        alignments
            initial alignments
        superimpose
            if True then superimposes data using alignments before running DTW
        gap_open_penalty
            penalty for opening a (series of) gap(s)
        gap_extend_penalty
            penalty for extending an existing series of gaps

        Returns
        -------
        RMSD matrix, coverage matrix
        """
        num = self.num_structures
        pairwise_rmsd_matrix = np.zeros((num, num))
        pairwise_coverage = np.zeros((num, num))
        for i in range(num):
            for j in range(num):
                name_1, name_2 = self.structures[i].name, self.structures[j].name
                structure_pair = psa.StructurePair(self.structures[i], self.structures[j])
                dtw_aln_1, dtw_aln_2 = structure_pair.get_dtw_coord_alignment(alignments[name_1], alignments[name_2],
                                                                              superimpose=superimpose,
                                                                              gap_open_penalty=gap_open_penalty,
                                                                              gap_extend_penalty=gap_extend_penalty)
                rmsd_class = structure_pair.get_rmsd_coverage(dtw_aln_1, dtw_aln_2)
                pairwise_rmsd_matrix[i, j] = rmsd_class.rmsd
                pairwise_coverage[i, j] = rmsd_class.coverage_aln
        return pairwise_rmsd_matrix, pairwise_coverage

    def _get_i_j_alignment(self, i: int, j: int, aln_array_1: np.ndarray, aln_array_2: np.ndarray,
                           gap_open_penalty: float,
                           gap_extend_penalty: float,
                           superimpose: bool = True):
        """
        Get DTW-alignment of two structures

        Parameters
        ----------
        i
            index of structure_1
        j
            index of structure_2
        aln_array_1
            aligned sequence of structure_1
        aln_array_2
            aligned_sequence of structure_2
        gap_open_penalty
            penalty for opening a (series of) gap(s)
        gap_extend_penalty
            penalty for extending an existing series of gaps

        Returns
        -------
        aligned_coordinates of structure_1,
        aligned_coordinates of structure_2,
        DTW alignment of structure_1 coords,
        DTW alignment of structure_2 coords
        """
        structure_pair = psa.StructurePair(self.final_structures[i], self.final_structures[j])
        dtw_aln_1, dtw_aln_2 = structure_pair.get_dtw_coord_alignment(aln_array_1, aln_array_2,
                                                                      gap_open_penalty=gap_open_penalty,
                                                                      gap_extend_penalty=gap_extend_penalty)
        if superimpose:
            common_coords_1, common_coords_2 = structure_pair.get_common_coordinates(dtw_aln_1, dtw_aln_2)
            rot, tran = rmsd_calculations.svd_superimpose(common_coords_1, common_coords_2)
            coords_2 = rmsd_calculations.apply_rotran(structure_pair.structure_2.coords, rot, tran)
        else:
            coords_2 = structure_pair.structure_2.coords
        aln_coords_1 = helper.get_aligned_data(dtw_aln_1, structure_pair.structure_1.coords, -1)
        aln_coords_2 = helper.get_aligned_data(dtw_aln_2, coords_2, -1)
        return aln_coords_1, aln_coords_2, dtw_aln_1, dtw_aln_2

    def _get_i_j_feature_alignment(self, i: int, j: int, aln_1: np.ndarray = None, aln_2: np.ndarray = None, gap_open_penalty: float = 1.,
                                   gap_extend_penalty: float = 1.):
        """
        Get DTW-alignment of two structures

        Parameters
        ----------
        i
            index of structure_1
        j
            index of structure_2
        aln_1
            aligned sequence of structure_1
        aln_2
            aligned_sequence of structure_2
        gap_open_penalty
            penalty for opening a (series of) gap(s)
        gap_extend_penalty
            penalty for extending an existing series of gaps

        Returns
        -------
        aligned_coordinates of structure_1,
        aligned_coordinates of structure_2,
        DTW alignment of structure_1 coords,
        DTW alignment of structure_2 coords
        """
        structure_pair = psa.StructurePair(self.final_structures[i], self.final_structures[j])
        if aln_1 is None or aln_2 is None:
            dtw_aln_1, dtw_aln_2, _ = structure_pair.get_dtw_feature_alignment(
                gap_open_feature=gap_open_penalty,
                gap_extend_feature=gap_extend_penalty)
        else:
            dtw_aln_1, dtw_aln_2 = aln_1, aln_2
        aln_features_1 = helper.get_aligned_data(dtw_aln_1, structure_pair.structure_1.features, -1)
        aln_features_2 = helper.get_aligned_data(dtw_aln_2, structure_pair.structure_2.features, -1)
        return aln_features_1, aln_features_2, dtw_aln_1, dtw_aln_2

    def align_features(self, gap_open_penalty: float = 1., gap_extend_penalty: float = 1.) -> dict:
        self.final_structures = [s for s in self.structures]
        pw_matrix, pw_alns = self.make_pairwise_dtw_score_matrix(gap_open_penalty=gap_open_penalty, gap_extend_penalty=gap_extend_penalty)
        tree, branch_lengths = nj.neighbor_joining(pw_matrix)
        msa_alignments = {s.name: {s.name: s.sequence} for s in self.final_structures}

        def make_intermediate_node(n1, n2, n_int):
            name_1, name_2 = self.final_structures[n1].name, self.final_structures[n2].name
            aln_1, aln_2 = pw_alns.get((name_1, name_2), (None, None))
            name_int = f"int-{n_int}"
            aln_coords_1, aln_coords_2, dtw_aln_1, dtw_aln_2 = self._get_i_j_feature_alignment(n1, n2, aln_1, aln_2,
                                                                                               gap_open_penalty=gap_open_penalty,
                                                                                               gap_extend_penalty=gap_extend_penalty)
            msa_alignments[name_1] = {name: "".join([sequence[i] if i != -1 else '-' for i in dtw_aln_1]) for name, sequence in
                                      msa_alignments[name_1].items()}
            msa_alignments[name_2] = {name: "".join([sequence[i] if i != -1 else '-' for i in dtw_aln_2]) for name, sequence in
                                      msa_alignments[name_2].items()}
            msa_alignments[name_int] = {**msa_alignments[name_1], **msa_alignments[name_2]}

            mean_features = get_mean_coords(aln_coords_1, aln_coords_2)
            self.final_structures.append(psa.Structure(name_int, None, self.final_structures[n1].coords, mean_features))

        for x in range(0, tree.shape[0] - 1, 2):
            node_1, node_2, node_int = tree[x, 0], tree[x + 1, 0], tree[x, 1]
            assert tree[x + 1, 1] == node_int
            make_intermediate_node(node_1, node_2, node_int)

        node_1, node_2 = tree[-1, 0], tree[-1, 1]
        make_intermediate_node(node_1, node_2, "final")
        return {**msa_alignments[self.final_structures[node_1].name], **msa_alignments[self.final_structures[node_2].name]}

    def align(self, alignments: dict, gap_open_penalty: float = 10., gap_extend_penalty: float = 5, superimpose: bool = True) -> dict:
        """
        Makes a multiple structure alignment

        Parameters
        ----------
        alignments
            initial sequence alignment
        gap_open_penalty
            penalty for opening a (series of) gap(s)
        gap_extend_penalty
            penalty for extending an existing series of gaps
        superimpose
            if True, uses alignments to superimpose data before running DTW
        Returns
        -------
        DTW-based multiple sequence alignment
        final coordinates stored in self.structures[-1]
        intermediate nodes of neighbor-joining tree stored in self.structures[self.num_structures:]
        """
        self.final_structures = [s for s in self.structures]
        alignments = {k: helper.aligned_string_to_array(alignments[k]) for k in alignments}
        pw_rmsd_matrix, pw_cov_matrix = self.make_pairwise_dtw_rmsd_matrix(alignments,
                                                                           gap_open_penalty=gap_open_penalty,
                                                                           gap_extend_penalty=gap_extend_penalty,
                                                                           superimpose=superimpose)
        pw_matrix = (pw_rmsd_matrix - np.min(pw_rmsd_matrix)) / (np.max(pw_rmsd_matrix) - np.min(pw_rmsd_matrix))
        pw_matrix *= (1 - pw_cov_matrix)
        tree, branch_lengths = nj.neighbor_joining(pw_matrix)
        self.tree = tree
        self.branch_lengths = branch_lengths
        msa_alignments = {s.name: {s.name: s.sequence} for s in self.final_structures}

        def make_intermediate_node(n1, n2, n_int):
            name_1, name_2 = self.final_structures[n1].name, self.final_structures[n2].name
            name_int = f"int-{n_int}"
            aln_coords_1, aln_coords_2, dtw_aln_1, dtw_aln_2 = self._get_i_j_alignment(n1, n2, alignments[name_1], alignments[name_2],
                                                                                       gap_open_penalty=gap_open_penalty,
                                                                                       gap_extend_penalty=gap_extend_penalty,
                                                                                       superimpose=superimpose)
            msa_alignments[name_1] = {name: "".join([sequence[i] if i != -1 else '-' for i in dtw_aln_1]) for name, sequence in
                                      msa_alignments[name_1].items()}
            msa_alignments[name_2] = {name: "".join([sequence[i] if i != -1 else '-' for i in dtw_aln_2]) for name, sequence in
                                      msa_alignments[name_2].items()}
            msa_alignments[name_int] = {**msa_alignments[name_1], **msa_alignments[name_2]}

            mean_coords = get_mean_coords(aln_coords_1, aln_coords_2)
            self.final_structures.append(psa.Structure(name_int, None, mean_coords, self.final_structures[n1].features))
            alignments[name_int] = alignments[name_1]

        for x in range(0, self.tree.shape[0] - 1, 2):
            node_1, node_2, node_int = self.tree[x, 0], self.tree[x + 1, 0], self.tree[x, 1]
            assert self.tree[x + 1, 1] == node_int
            make_intermediate_node(node_1, node_2, node_int)

        node_1, node_2 = self.tree[-1, 0], self.tree[-1, 1]
        make_intermediate_node(node_1, node_2, "final")
        return {**msa_alignments[self.final_structures[node_1].name], **msa_alignments[self.final_structures[node_2].name]}
