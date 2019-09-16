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


def make_structures(names: list, sequences: list, coords: list) -> list:
    return [psa.Structure(names[i], sequences[i], coords[i]) for i in range(len(names))]


class StructureMultiple:
    def __init__(self, structures: typing.List[psa.Structure]):
        self.structures = [s for s in structures]
        self.num_structures = len(structures)
        self.tree = None
        self.branch_lengths = None

    def make_pairwise_rmsd_matrix(self, alignments: dict, run_dtw: bool = False, superimpose: bool = True, gap_open_penalty: float = 0.,
                                  gap_extend_penalty: float = 0.):
        """
        Find RMSDs of pairwise alignment of each pair of sequences

        Parameters
        ----------
        alignments
            initial alignments
        run_dtw
            if True then re-aligns using DTW and gap penalties
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
        pairwise_matrix = np.zeros((num, num))
        pairwise_coverage = np.zeros((num, num))
        for i in range(num):
            for j in range(num):
                name_1, name_2 = self.structures[i].name, self.structures[j].name
                structure_pair = psa.StructurePair(self.structures[i], self.structures[j])
                if run_dtw:
                    dtw_aln_1, dtw_aln_2 = structure_pair.get_dtw_alignment(alignments[name_1], alignments[name_2],
                                                                            superimpose=superimpose,
                                                                            gap_open_penalty=gap_open_penalty,
                                                                            gap_extend_penalty=gap_extend_penalty)
                    rmsd_class = structure_pair.get_rmsd_coverage(dtw_aln_1, dtw_aln_2)
                else:
                    if isinstance(alignments[name_1], str):
                        aln_1 = helper.aligned_string_to_array(alignments[name_1])
                        aln_2 = helper.aligned_string_to_array(alignments[name_2])
                    else:
                        aln_1 = alignments[name_1]
                        aln_2 = alignments[name_2]
                    rmsd_class = structure_pair.get_rmsd_coverage(aln_1, aln_2)
                pairwise_matrix[i, j] = rmsd_class.rmsd
                pairwise_coverage[i, j] = rmsd_class.coverage_aln
        return pairwise_matrix, pairwise_coverage

    def _get_i_j_alignment(self, i: int, j: int, aln_array_1: np.ndarray, aln_array_2: np.ndarray, gap_open_penalty: float,
                           gap_extend_penalty: float):
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
        structure_pair = psa.StructurePair(self.structures[i], self.structures[j])
        dtw_aln_1, dtw_aln_2 = structure_pair.get_dtw_alignment(aln_array_1, aln_array_2,
                                                                gap_open_penalty=gap_open_penalty,
                                                                gap_extend_penalty=gap_extend_penalty)
        common_coords_1, common_coords_2 = structure_pair.get_common_coordinates(dtw_aln_1, dtw_aln_2)
        rot, tran = rmsd_calculations.svd_superimpose(common_coords_1, common_coords_2)
        coords_2 = rmsd_calculations.apply_rotran(structure_pair.structure_2.coords, rot, tran)
        aln_coords_1 = helper.get_aligned_data(dtw_aln_1, structure_pair.structure_1.coords, -1)
        aln_coords_2 = helper.get_aligned_data(dtw_aln_2, coords_2, -1)
        return aln_coords_1, aln_coords_2, dtw_aln_1, dtw_aln_2

    def align(self, alignments: dict, gap_open_penalty: float = 0., gap_extend_penalty: float = 0, superimpose: bool = True) -> dict:
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
        alignments = {k: helper.aligned_string_to_array(alignments[k]) for k in alignments}
        pw_matrix, _ = self.make_pairwise_rmsd_matrix(alignments,
                                                      gap_open_penalty=gap_open_penalty,
                                                      gap_extend_penalty=gap_extend_penalty,
                                                      run_dtw=True,
                                                      superimpose=superimpose)
        tree, branch_lengths = nj.neighbor_joining(pw_matrix)
        self.tree = tree
        self.branch_lengths = branch_lengths
        msa_alignments = {s.name: {s.name: s.sequence} for s in self.structures}
        for x in range(0, tree.shape[0] - 1, 2):
            node_1, node_2, node_int = tree[x, 0], tree[x + 1, 0], tree[x, 1]
            assert tree[x + 1, 1] == node_int
            name_1, name_2 = self.structures[node_1].name, self.structures[node_2].name
            name_int = f"int-{node_int}"
            aln_coords_1, aln_coords_2, dtw_aln_1, dtw_aln_2 = self._get_i_j_alignment(node_1, node_2,
                                                                                       alignments[name_1], alignments[name_2],
                                                                                       gap_open_penalty=gap_open_penalty,
                                                                                       gap_extend_penalty=gap_extend_penalty)
            msa_alignments[name_1] = {name: "".join([sequence[i] if i != -1 else '-' for i in dtw_aln_1]) for name, sequence in
                                      msa_alignments[name_1].items()}
            msa_alignments[name_2] = {name: "".join([sequence[i] if i != -1 else '-' for i in dtw_aln_2]) for name, sequence in
                                      msa_alignments[name_2].items()}
            msa_alignments[name_int] = {**msa_alignments[name_1], **msa_alignments[name_2]}

            mean_coords = get_mean_coords(aln_coords_1, aln_coords_2)
            self.structures.append(psa.Structure(name_int, None, mean_coords))
            alignments[name_int] = alignments[name_1]

        node_1, node_2 = tree[-1, 0], tree[-1, 1]
        name_1, name_2 = self.structures[node_1].name, self.structures[node_2].name
        aln_coords_1, aln_coords_2, dtw_aln_1, dtw_aln_2 = self._get_i_j_alignment(node_1, node_2, alignments[name_1], alignments[name_2],
                                                                                   gap_open_penalty=gap_open_penalty,
                                                                                   gap_extend_penalty=gap_extend_penalty)
        msa_alignments[name_1] = {name: "".join([sequence[i] if i != -1 else '-' for i in dtw_aln_1]) for name, sequence in
                                  msa_alignments[name_1].items()}
        msa_alignments[name_2] = {name: "".join([sequence[i] if i != -1 else '-' for i in dtw_aln_2]) for name, sequence in
                                  msa_alignments[name_2].items()}
        mean_coords = get_mean_coords(aln_coords_1, aln_coords_2)
        self.structures.append(psa.Structure(f"int-final", None, mean_coords))
        return {**msa_alignments[name_1], **msa_alignments[name_2]}
