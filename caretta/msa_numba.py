import numba as nb
import numpy as np

from caretta import neighbor_joining as nj
from caretta import psa_numba as psa
from caretta import rmsd_calculations, helper


@nb.njit(parallel=True)
def make_pairwise_dtw_score_matrix(coords_array, secondary_array, lengths_array, gamma,
                                   gap_open_penalty: float, gap_extend_penalty: float,
                                   gap_open_sec, gap_extend_sec):
    pairwise_matrix = np.zeros((coords_array.shape[0], coords_array.shape[0]))
    for i in range(pairwise_matrix.shape[0] - 1):
        for j in range(i + 1, pairwise_matrix.shape[1]):
            dtw_aln_1, dtw_aln_2, score = psa.get_pairwise_alignment(coords_array[i, :lengths_array[i]], coords_array[j, :lengths_array[j]],
                                                                     secondary_array[i, :lengths_array[i]], secondary_array[j, :lengths_array[j]],
                                                                     gamma,
                                                                     gap_open_sec=gap_open_sec,
                                                                     gap_extend_sec=gap_extend_sec,
                                                                     gap_open_penalty=gap_open_penalty,
                                                                     gap_extend_penalty=gap_extend_penalty)
            common_coords_1, common_coords_2 = psa.get_common_coordinates(coords_array[i, :lengths_array[i]],
                                                                          coords_array[j, :lengths_array[j]],
                                                                          dtw_aln_1, dtw_aln_2)
            rotation_matrix, translation_matrix = rmsd_calculations.svd_superimpose(common_coords_1[:, :3], common_coords_2[:, :3])
            common_coords_2[:, :3] = rmsd_calculations.apply_rotran(common_coords_2[:, :3], rotation_matrix, translation_matrix)
            score = rmsd_calculations.get_exp_distances(common_coords_1, common_coords_2, gamma, True)
            pairwise_matrix[i, j] = -score
    pairwise_matrix += pairwise_matrix.T
    return pairwise_matrix


@nb.njit
def _get_alignment_data(coords_1, coords_2, secondary_1, secondary_2, gamma,
                        gap_open_sec, gap_extend_sec,
                        gap_open_penalty: float, gap_extend_penalty: float):
    dtw_aln_1, dtw_aln_2, _ = psa.get_pairwise_alignment(
        coords_1, coords_2,
        secondary_1, secondary_2,
        gamma,
        gap_open_sec=gap_open_sec,
        gap_extend_sec=gap_extend_sec,
        gap_open_penalty=gap_open_penalty,
        gap_extend_penalty=gap_extend_penalty)
    pos_1, pos_2 = helper.get_common_positions(dtw_aln_1, dtw_aln_2, -1)
    coords_1[:, :3], coords_2[:, :3], _ = rmsd_calculations.superpose_with_pos(coords_1[:, :3], coords_2[:, :3],
                                                                               coords_1[pos_1][:, :3], coords_2[pos_2][:, :3])
    aln_coords_1 = helper.get_aligned_data(dtw_aln_1, coords_1, -1)
    aln_coords_2 = helper.get_aligned_data(dtw_aln_2, coords_2, -1)
    aln_sec_1 = helper.get_aligned_string_data(dtw_aln_1, secondary_1, -1)
    aln_sec_2 = helper.get_aligned_string_data(dtw_aln_2, secondary_2, -1)
    return aln_coords_1, aln_coords_2, aln_sec_1, aln_sec_2, dtw_aln_1, dtw_aln_2


@nb.njit
def get_mean_coords_extra(aln_coords_1: np.ndarray, aln_coords_2: np.ndarray) -> np.ndarray:
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
        mean_coords[i, :-1] = np.array([np.nanmean(np.array([aln_coords_1[i, x], aln_coords_2[i, x]])) for x in range(aln_coords_1.shape[1] - 1)])
        if not np.isnan(aln_coords_1[i, 0]):
            mean_coords[i, -1] += aln_coords_1[i, -1]
        if not np.isnan(aln_coords_2[i, 0]):
            mean_coords[i, -1] += aln_coords_2[i, -1]
    return mean_coords


@nb.njit
def get_mean_secondary(aln_sec_1: np.ndarray, aln_sec_2: np.ndarray, gap=0) -> np.ndarray:
    """
    Mean of two coordinate sets (of the same shape)

    Parameters
    ----------
    aln_sec_1
    aln_sec_2
    gap

    Returns
    -------
    mean_sec
    """
    mean_sec = np.zeros(aln_sec_1.shape, dtype=aln_sec_1.dtype)
    for i in range(aln_sec_1.shape[0]):
        if aln_sec_1[i] == aln_sec_2[i]:
            mean_sec[i] = aln_sec_1[i]
        else:
            if aln_sec_1[i] != gap:
                mean_sec[i] = aln_sec_1[i]
            elif aln_sec_2[i] != gap:
                mean_sec[i] = aln_sec_2[i]
    return mean_sec


class Structure:
    def __init__(self, name, sequence, secondary, coords, add_column=True):
        consensus_weight = 1.
        self.name = name
        self.sequence = sequence
        self.secondary = secondary
        if add_column:
            add = np.zeros((coords.shape[0], 1)) + consensus_weight
            self.coords = np.hstack((coords, add))
        else:
            self.coords = coords


class StructureMultiple:
    def __init__(self, structures):
        self.structures = structures
        self.lengths_array = np.array([len(s.sequence) for s in self.structures])
        self.max_length = np.max(self.lengths_array)
        self.coords_array = np.zeros((len(self.structures), self.max_length, self.structures[0].coords.shape[1]))
        self.secondary_array = np.zeros((len(self.structures), self.max_length))
        for i in range(len(self.structures)):
            self.coords_array[i, :self.lengths_array[i]] = self.structures[i].coords
            self.secondary_array[i, :self.lengths_array[i]] = self.structures[i].secondary
        self.final_structures = []
        self.tree = None
        self.branch_lengths = None

    def align(self, gamma, gap_open_sec, gap_extend_sec, gap_open_penalty, gap_extend_penalty, pw_matrix=None,
              plot=True) -> dict:
        print("Aligning...")
        if pw_matrix is None:
            pw_matrix = make_pairwise_dtw_score_matrix(self.coords_array,
                                                       self.secondary_array,
                                                       self.lengths_array,
                                                       gamma,
                                                       gap_open_sec, gap_extend_sec,
                                                       gap_open_penalty, gap_extend_penalty)
        tree, branch_lengths = nj.neighbor_joining(pw_matrix)
        self.tree = tree
        self.branch_lengths = branch_lengths
        self.final_structures = [s for s in self.structures]
        msa_alignments = {s.name: {s.name: s.sequence} for s in self.structures}

        def make_intermediate_node(n1, n2, n_int):
            name_1, name_2 = self.final_structures[n1].name, self.final_structures[n2].name
            name_int = f"int-{n_int}"
            n1_coords = self.final_structures[n1].coords
            n1_coords[:, -1] *= len(msa_alignments[name_2])
            n1_coords[:, -1] /= 2.
            n2_coords = self.final_structures[n2].coords
            n2_coords[:, -1] *= len(msa_alignments[name_1])
            n2_coords[:, -1] /= 2.
            aln_coords_1, aln_coords_2, aln_sec_1, aln_sec_2, dtw_aln_1, dtw_aln_2 = _get_alignment_data(n1_coords,
                                                                                                         n2_coords,
                                                                                                         self.final_structures[
                                                                                                             n1].secondary,
                                                                                                         self.final_structures[
                                                                                                             n2].secondary,
                                                                                                         gamma,
                                                                                                         gap_open_sec=gap_open_sec,
                                                                                                         gap_extend_sec=gap_extend_sec,
                                                                                                         gap_open_penalty=gap_open_penalty,
                                                                                                         gap_extend_penalty=gap_extend_penalty,
                                                                                                         plot=plot)
            aln_coords_1[:, -1] *= 2. / len(msa_alignments[name_2])
            aln_coords_2[:, -1] *= 2. / len(msa_alignments[name_1])
            msa_alignments[name_1] = {name: "".join([sequence[i] if i != -1 else '-' for i in dtw_aln_1]) for name, sequence in
                                      msa_alignments[name_1].items()}
            msa_alignments[name_2] = {name: "".join([sequence[i] if i != -1 else '-' for i in dtw_aln_2]) for name, sequence in
                                      msa_alignments[name_2].items()}
            msa_alignments[name_int] = {**msa_alignments[name_1], **msa_alignments[name_2]}

            mean_coords = get_mean_coords_extra(aln_coords_1, aln_coords_2)
            mean_sec = get_mean_secondary(aln_sec_1, aln_sec_2, 0)
            self.final_structures.append(Structure(name_int, None, mean_sec, mean_coords, add_column=False))

        for x in range(0, self.tree.shape[0] - 1, 2):
            node_1, node_2, node_int = self.tree[x, 0], self.tree[x + 1, 0], self.tree[x, 1]
            assert self.tree[x + 1, 1] == node_int
            make_intermediate_node(node_1, node_2, node_int)

        node_1, node_2 = self.tree[-1, 0], self.tree[-1, 1]
        make_intermediate_node(node_1, node_2, "final")
        alignment = {**msa_alignments[self.final_structures[node_1].name], **msa_alignments[self.final_structures[node_2].name]}
        return alignment

    def superpose(self, alignments: dict):
        reference_index = 0
        reference_key = self.structures[reference_index].name
        core_indices = np.array([i for i in range(len(alignments[reference_key])) if '-' not in [alignments[n][i] for n in alignments]])
        aln_ref = helper.aligned_string_to_array(alignments[reference_key])
        ref_coords = self.structures[reference_index].coords[np.array([aln_ref[c] for c in core_indices])][:, :3]
        ref_centroid = rmsd_calculations.nb_mean_axis_0(ref_coords)
        ref_coords -= ref_centroid
        for i in range(len(self.structures)):
            if i == reference_index:
                self.structures[i].coords[:, :3] -= ref_centroid
            else:
                aln_c = helper.aligned_string_to_array(alignments[self.structures[i].name])
                common_coords_2 = self.structures[i].coords[np.array([aln_c[c] for c in core_indices])][:, :3]
                rotation_matrix, translation_matrix = rmsd_calculations.svd_superimpose(ref_coords, common_coords_2)
                self.structures[i].coords[:, :3] = rmsd_calculations.apply_rotran(self.structures[i].coords[:, :3], rotation_matrix,
                                                                                  translation_matrix)

    def make_pairwise_rmsd_matrix(self, alignments: dict, superpose_first: bool = True):
        """
        Find RMSDs and coverages of the alignment of each pair of sequences

        Parameters
        ----------
        alignments
        superpose_first
            if True then superposes all structures to first structure first

        Returns
        -------
        RMSD matrix, coverage matrix
        """
        num = len(self.structures)
        pairwise_rmsd_matrix = np.zeros((num, num))
        pairwise_rmsd_matrix[:] = np.nan
        pairwise_coverage = np.zeros((num, num))
        pairwise_coverage[:] = np.nan
        if superpose_first:
            self.superpose(alignments)
        for i in range(num):
            for j in range(i + 1, num):
                name_1, name_2 = self.structures[i].name, self.structures[j].name
                if isinstance(alignments[name_1], str):
                    aln_1 = helper.aligned_string_to_array(alignments[name_1])
                    aln_2 = helper.aligned_string_to_array(alignments[name_2])
                else:
                    aln_1 = alignments[name_1]
                    aln_2 = alignments[name_2]
                common_coords_1, common_coords_2 = psa.get_common_coordinates(self.structures[i].coords[:, :3],
                                                                              self.structures[j].coords[:, :3], aln_1, aln_2)
                assert common_coords_1.shape[0] > 0
                if not superpose_first:
                    rot, tran = rmsd_calculations.svd_superimpose(common_coords_1, common_coords_2)
                    common_coords_2 = rmsd_calculations.apply_rotran(common_coords_2, rot, tran)
                pairwise_rmsd_matrix[i, j] = pairwise_rmsd_matrix[j, i] = rmsd_calculations.get_rmsd(common_coords_1, common_coords_2)
                pairwise_coverage[i, j] = pairwise_coverage[j, i] = common_coords_1.shape[0] / len(aln_1)
        return pairwise_rmsd_matrix, pairwise_coverage
