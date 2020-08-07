import pickle
import typing
from dataclasses import dataclass
from pathlib import Path

import numba as nb
import numpy as np
import prody as pd
from scipy.spatial.distance import pdist, squareform
from geometricus import protein_utility, geometricus, utility

from caretta import dynamic_time_warping as dtw, neighbor_joining as nj, score_functions, superposition_functions, feature_extraction


@nb.njit(cache=False)
def get_common_coordinates(coords_1, coords_2, aln_1, aln_2, gap=-1):
    """
    Return coordinate positions aligned in both coords_1 and coords_2
    """
    assert aln_1.shape == aln_2.shape
    pos_1, pos_2 = score_functions.get_common_positions(aln_1, aln_2, gap)
    return coords_1[pos_1], coords_2[pos_2]


@nb.njit
def get_mean_coords(aln_1, coords_1: np.ndarray, aln_2, coords_2: np.ndarray) -> np.ndarray:
    """
    Mean of two coordinate sets (of the same shape)
    """
    mean_coords = np.zeros((aln_1.shape[0], coords_1.shape[1]))
    for i, (x, y) in enumerate(zip(aln_1, aln_2)):
        if x == -1:
            mean_coords[i] = coords_2[y]
        elif y == -1:
            mean_coords[i] = coords_1[x]
        else:
            mean_coords[i] = np.array([np.nanmean(np.array([coords_1[x, d],
                                                            coords_2[y, d]])) for d in range(coords_1.shape[1])])
    return mean_coords


@nb.njit
def get_pairwise_braycurtis(fingerprints):
    res = np.zeros((fingerprints.shape[0], fingerprints.shape[0]), dtype=np.float64)
    for i in range(fingerprints.shape[0]):
        for j in range(fingerprints.shape[0]):
            res[i, j] = np.sum(np.abs(fingerprints[i] - fingerprints[j]))/np.sum(np.abs(fingerprints[i] + fingerprints[j]))
    return res


def get_mean_weights(weights_1, weights_2, aln_1, aln_2, gap=-1):
    mean_weights = np.zeros(aln_1.shape[0])
    for i, (x, y) in enumerate(zip(aln_1, aln_2)):
        if not x == gap:
            mean_weights[i] += weights_1[x]
        if not y == gap:
            mean_weights[i] += weights_2[y]
    return mean_weights


@dataclass
class OutputFiles:
    fasta_file: Path = Path("./result.fasta")
    pdb_folder: Path = Path("./result_pdb/")
    cleaned_pdb_folder: Path = Path("./cleaned_pdb")
    feature_file: Path = Path("./result_features.pkl")
    class_file: Path = Path("./result_class.pkl")


@dataclass
class StructureMultiple:
    """
    Class for multiple structure alignment

    Constructor Arguments
    ---------------------
    structures
        list of chelonia_utilities.protein_utility.Structure objects
    superposition_function
        a function that takes two coordinate sets as input and superposes them
        returns a score, superposed_coords_1, superposed_coords_2
    score_function
        a function that takes two paired coordinate sets (same shape) and returns a score
    consensus_weight
        weights the effect of well-aligned columns on the progressive alignment
    final_structures
        makes the progressive alignment tree of structures, last one is the consensus structure of the multiple alignment
    tree
        neighbor joining tree of indices controlling alignment - indices beyond len(structures) refer to intermediate nodes
    branch_lengths
    alignment
        indices of aligning residues from each structure, gaps are -1s
    """
    structures: typing.List[protein_utility.Structure]
    sequences: typing.Dict[str, str]
    superposition_function: typing.Callable[[np.ndarray, np.ndarray, dict, typing.Callable[[np.ndarray, np.ndarray], float]],
                                            typing.Tuple[float, np.ndarray, np.ndarray]] = lambda x, y: (0, x, y)
    score_function: typing.Callable[[np.ndarray, np.ndarray], float] = score_functions.get_caretta_score
    gamma: float = 0.3
    mean_function: typing.Callable[
        [np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray] = get_mean_coords
    consensus_weight: float = 1.
    final_structures: typing.Union[None, typing.List[protein_utility.Structure]] = None
    final_consensus_weights: typing.Union[None, typing.List[np.ndarray]] = None
    tree: typing.Union[None, np.ndarray] = None
    branch_lengths: typing.Union[None, np.ndarray] = None
    alignment: typing.Union[None, dict] = None
    output_folder: Path = Path("./caretta_results")

    @staticmethod
    def align_from_pdb_files(input_pdb, gap_open_penalty=1., gap_extend_penalty=0.01, consensus_weight=1.,
                             output_folder=Path("../caretta_results"),
                             num_threads=20,
                             write_fasta=False,
                             write_pdb=False,
                             write_features=False,
                             write_class=False,
                             overwrite_dssp=False):
        """
        Caretta aligns protein structures and returns a sequence alignment, a set of aligned feature matrices, superposed PDB files, and
        a class with intermediate structures made during progressive alignment.
        Parameters
        ----------
        input_pdb
            Can be \n
            A list of PDB files
            A list of PDB IDs
            A folder with input protein files
            A file which lists PDB filenames on each line
            A file which lists PDB IDs on each line
        gap_open_penalty
            default 1
        gap_extend_penalty
            default 0.01
        consensus_weight
            default 1
        output_folder
            default "caretta_results"
        num_threads
            Number of threads to use for feature extraction
        write_fasta
            True => writes alignment as fasta file (default True)
            writes to output_folder / result.fasta
        write_pdb
            True => writes all protein PDB files superposed by alignment (default True)
             writes to output_folder / superposed_pdb
        write_features
            True => extracts and writes aligned features as a dictionary of numpy arrays into a pickle file (default True)
            writes to output_folder / result_features.pkl
        write_class
            True => writes StructureMultiple class with intermediate structures and tree to pickle file (default True)
            writes to output_folder / result_class.pkl
        overwrite_dssp
            Forces DSSP to rerun even if files are already present (default False)

        Returns
        -------
        StructureMultiple class
        """
        msa_class = StructureMultiple.from_pdb_files(input_pdb,
                                                     superposition_function=superposition_functions.moment_svd_superpose_function,
                                                     consensus_weight=consensus_weight,
                                                     output_folder=output_folder
                                                     )
        pw_matrix = msa_class.make_pairwise_dtw_matrix()
        msa_class.align(pw_matrix, gap_open_penalty, gap_extend_penalty)
        msa_class.write_files(write_fasta,
                              write_pdb,
                              write_features,
                              write_class, num_threads, overwrite_dssp)
        return msa_class

    @classmethod
    def from_pdb_files(cls, input_pdb, superposition_function, score_function=score_functions.get_caretta_score, consensus_weight=1.,
                       output_folder=Path("./caretta_results")):
        """
        Makes a StructureMultiple object from a list of pdb files/names or a folder of pdb files

        Parameters
        ----------
        input_pdb
            list of pdb files/names or a folder containing pdb files
        superposition_function
            a function that takes two coordinate sets as input and superposes them
            returns a score, superposed_coords_1, superposed_coords_2
        score_function
            a function that takes two paired coordinate sets (same shape) and returns a score
        consensus_weight
            weights the effect of well-aligned columns on the progressive alignment
        output_folder

        Returns
        -------
        StructureMultiple object (unaligned)
        """
        output_folder = Path(output_folder)
        if not output_folder.exists():
            output_folder.mkdir()

        cleaned_pdb_folder = output_folder / "cleaned_pdb"
        if not cleaned_pdb_folder.exists():
            cleaned_pdb_folder.mkdir()
        pdb_files = protein_utility.parse_pdb_files_and_clean(input_pdb, cleaned_pdb_folder)

        structures = []
        sequences = {}
        for pdb_file in pdb_files:
            pdb_name = utility.get_file_parts(pdb_file)[1]
            protein = pd.parsePDB(str(pdb_file)).select("protein")
            indices = [i for i, a in enumerate(protein.iterAtoms()) if a.getName() == 'CA']
            protein = protein[indices]
            coordinates = protein.getCoords()
            structures.append(protein_utility.Structure(pdb_name, coordinates.shape[0], coordinates))
            sequences[pdb_name] = protein.getSequence()
        msa_class = StructureMultiple(structures, sequences, superposition_function, score_function=score_function,
                                      consensus_weight=consensus_weight, output_folder=output_folder)
        return msa_class

    def get_pairwise_alignment(self, coords_1, coords_2, parameters,
                               gap_open_penalty: float, gap_extend_penalty: float,
                               weight=False, weights_1=None, weights_2=None, n_iter=10):
        """
        Aligns coords_1 to coords_2 by first superposing and then running dtw on the score matrix of the superposed coordinate sets
        Parameters
        ----------
        coords_1
        coords_2
        parameters
        gap_open_penalty
        gap_extend_penalty
        weight
        weights_1
        weights_2
        n_iter

        Returns
        -------
        alignment_1, alignment_2, score, superposed_coords_1, superposed_coords_2
        """
        #  if exclude_last:
        #      _, coords_1[:, :-1], coords_2[:, :-1] = self.superposition_function(coords_1[:, :-1], coords_2[:, :-1])
        #  else:
        _, coords_1, coords_2 = self.superposition_function(coords_1, coords_2, parameters)
        if weight:
            assert weights_1 is not None
            assert weights_2 is not None
            weights_1 = weights_1.reshape(-1, 1)
            weights_2 = weights_2.reshape(-1, 1)
        else:
            weights_1 = np.zeros((coords_1.shape[0], 1))
            weights_2 = np.zeros((coords_2.shape[0], 1))

        score_matrix = score_functions.make_score_matrix(np.hstack((coords_1, weights_1)),
                                                         np.hstack((coords_2, weights_2)), self.score_function)
        dtw_aln_array_1, dtw_aln_array_2, dtw_score = dtw.dtw_align(score_matrix, gap_open_penalty, gap_extend_penalty)
        for i in range(n_iter):
            pos_1, pos_2 = score_functions.get_common_positions(dtw_aln_array_1, dtw_aln_array_2)
            common_coords_1, common_coords_2 = coords_1[pos_1], coords_2[pos_2]
            c1, c2, common_coords_2 = superposition_functions.paired_svd_superpose_with_subset(coords_1, coords_2, common_coords_1, common_coords_2)
            score_matrix = score_functions.make_score_matrix(np.hstack((c1, weights_1)), np.hstack((c2, weights_2)), self.score_function)
            aln_1, aln_2, score = dtw.dtw_align(score_matrix, gap_open_penalty, gap_extend_penalty)
            if score > dtw_score:
                coords_1 = c1
                coords_2 = c2
                dtw_score = score
                dtw_aln_array_1 = aln_1
                dtw_aln_array_2 = aln_2
            else:
                break
        return dtw_aln_array_1, dtw_aln_array_2, dtw_score, coords_1, coords_2

    def make_pairwise_shape_matrix(self, resolution: np.ndarray, kmer_size=30, radius=10, metric="braycurtis"):
        """
        Makes an all vs. all matrix of distance scores between all the structures.
        Distance is measured by the braycurtis distance of each structure's normalized shape counts.
        Parameters
        ----------
        resolution
        kmer_size
        radius
        metric
            distance metric (accepts any metric supported by scipy.spatial.distance

        Returns
        -------
        [n x n] similarity matrix
        """
        pairwise_matrix = np.zeros((len(self.structures), len(self.structures)))
        kmer_invariants = {self.structures[i].name: geometricus.MomentInvariants.from_coordinates(self.structures[i].name,
                                                                                                  self.structures[i].coordinates,
                                                                                                  None,
                                                                                                  split_size=kmer_size, split_type="kmer") for i in
                           range(len(self.structures))}
        radius_invariants = {self.structures[i].name: geometricus.MomentInvariants.from_coordinates(self.structures[i].name,
                                                                                                    self.structures[i].coordinates,
                                                                                                    None,
                                                                                                    split_size=radius, split_type="radius") for i in
                             range(pairwise_matrix.shape[0])}
        shapes = geometricus.GeometricusEmbedding(kmer_invariants, radius_invariants, resolution,
                                                  [self.structures[i].name for i in range(len(self.structures))])
        return squareform(pdist(shapes.embedding, metric=metric))

    def make_pairwise_dtw_matrix(self, parameters: dict, gap_open_penalty: float, gap_extend_penalty: float, invert=True):
        """
        Makes an all vs. all matrix of distance (or similarity) scores between all the structures using pairwise alignment.

        Parameters
        ----------
        gap_open_penalty
        gap_extend_penalty
        invert
            if True returns distance matrix
            if False returns similarity matrix

        Returns
        -------
        [n x n] matrix
        """
        pairwise_matrix = np.zeros((len(self.structures), len(self.structures)))
        for i in range(pairwise_matrix.shape[0] - 1):
            for j in range(i + 1, pairwise_matrix.shape[1]):
                coords_1, coords_2 = self.structures[i].coordinates, self.structures[j].coordinates,
                dtw_aln_1, dtw_aln_2, score, coords_1, coords_2 = self.get_pairwise_alignment(coords_1, coords_2,
                                                                                              parameters,
                                                                                              gap_open_penalty=gap_open_penalty,
                                                                                              gap_extend_penalty=gap_extend_penalty,
                                                                                              weight=False)
                common_coords_1, common_coords_2 = get_common_coordinates(coords_1,
                                                                          coords_2,
                                                                          dtw_aln_1, dtw_aln_2)
                pairwise_matrix[i, j] = score_functions.get_total_score(common_coords_1, common_coords_2, self.score_function, True)
                if invert:
                    pairwise_matrix[i, j] *= -1
        pairwise_matrix += pairwise_matrix.T
        return pairwise_matrix

    def align(self, pw_matrix, parameters, gap_open_penalty, gap_extend_penalty) -> dict:
        """
        Makes a multiple structure alignment

        Parameters
        ----------
        pw_matrix
            pairwise similarity matrix to base the neighbor joining tree on
        gap_open_penalty
        gap_extend_penalty

        Returns
        -------
        alignment = {name: indices of aligning residues with gaps as -1s}
        """
        print("Aligning...")
        if len(self.structures) == 2:
            coords_1, coords_2 = self.structures[0].coordinates, self.structures[1].coordinates,
            dtw_1, dtw_2, _, _, _ = self.get_pairwise_alignment(coords_1, coords_2, parameters,
                                                                gap_open_penalty=gap_open_penalty,
                                                                gap_extend_penalty=gap_extend_penalty, weight=False)
            self.alignment = {self.structures[0].name: dtw_1,
                              self.structures[1].name: dtw_2}
            return self.alignment
        assert pw_matrix is not None
        assert pw_matrix.shape[0] == len(self.structures)
        tree, branch_lengths = nj.neighbor_joining(pw_matrix)
        self.tree = tree
        self.branch_lengths = branch_lengths
        print("Neighbor joining tree constructed")
        self.final_structures = [s for s in self.structures]
        self.final_consensus_weights = [np.full((self.structures[i].coordinates.shape[0], 1),
                                                self.consensus_weight, dtype=np.float64) for i in range(len(self.structures))]
        msa_alignments = {s.name: {s.name: np.arange(s.length)} for s in self.final_structures}

        def make_intermediate_node(n1, n2, n_int):
            name_1, name_2 = self.final_structures[n1].name, self.final_structures[n2].name
            name_int = f"int-{n_int}"
            n1_coords = self.final_structures[n1].coordinates
            n1_weights = self.final_consensus_weights[n1]
            n1_weights *= len(msa_alignments[name_2])
            n1_weights /= 2
            n2_coords = self.final_structures[n2].coordinates
            n2_weights = self.final_consensus_weights[n2]
            n2_weights *= len(msa_alignments[name_1])
            n2_weights /= 2
            dtw_aln_1, dtw_aln_2, score, n1_coords, n2_coords = self.get_pairwise_alignment(n1_coords, n2_coords, parameters,
                                                                                            gap_open_penalty=gap_open_penalty,
                                                                                            gap_extend_penalty=gap_extend_penalty,
                                                                                            weight=True, weights_1=n1_weights, weights_2=n2_weights)
            n1_weights *= 2. / len(msa_alignments[name_2])
            n2_weights *= 2. / len(msa_alignments[name_1])
            msa_alignments[name_1] = {name: np.array([sequence[i] if i != -1 else -1 for i in dtw_aln_1]) for name, sequence in
                                      msa_alignments[name_1].items()}
            msa_alignments[name_2] = {name: np.array([sequence[i] if i != -1 else -1 for i in dtw_aln_2]) for name, sequence in
                                      msa_alignments[name_2].items()}
            msa_alignments[name_int] = {**msa_alignments[name_1], **msa_alignments[name_2]}

            mean_coords = self.mean_function(dtw_aln_1, n1_coords, dtw_aln_2, n2_coords)
            mean_weights = get_mean_weights(n1_weights, n2_weights, dtw_aln_1, dtw_aln_2)
            self.final_structures.append(protein_utility.Structure(name_int, mean_coords.shape[0], mean_coords))
            self.final_consensus_weights.append(mean_weights)

        for x in range(0, self.tree.shape[0] - 1, 2):
            node_1, node_2, node_int = self.tree[x, 0], self.tree[x + 1, 0], self.tree[x, 1]
            assert self.tree[x + 1, 1] == node_int
            make_intermediate_node(node_1, node_2, node_int)

        node_1, node_2 = self.tree[-1, 0], self.tree[-1, 1]
        make_intermediate_node(node_1, node_2, "final")
        alignment = {**msa_alignments[self.final_structures[node_1].name], **msa_alignments[self.final_structures[node_2].name]}
        self.alignment = alignment
        return alignment

    def write_files(self, write_fasta,
                    write_pdb,
                    write_features,
                    write_class, num_threads, overwrite_dssp):
        if any((write_fasta, write_pdb, write_pdb, write_class)):
            print("Writing files...")
        if write_fasta:
            self.write_alignment(self.output_folder / "result.fasta")
        if write_pdb:
            pdb_folder = self.output_folder / "superposed_pdb"
            if not pdb_folder.exists():
                pdb_folder.mkdir()
            self.write_superposed_pdbs(pdb_folder)
        if write_features:
            dssp_dir = self.output_folder / ".caretta_tmp"
            if not dssp_dir.exists():
                dssp_dir.mkdir()
            with open(str(self.output_folder / "result_features.pkl"), "wb") as f:
                pickle.dump(self.get_aligned_features(dssp_dir, num_threads, overwrite_dssp), f)
        if write_class:
            with open(str(self.output_folder / "result_class.pkl"), "wb") as f:
                pickle.dump(self, f)

    def write_alignment(self, filename, alignments: dict = None):
        """
        Writes alignment to a fasta file
        """
        if alignments is None:
            alignments = self.alignment
        with open(filename, "w") as f:
            for key in alignments:
                sequence = "".join(self.sequences[key][n] if n != -1 else '-' for n in alignments[key])
                f.write(f">{key}\n{sequence}\n")

    def write_superposed_pdbs(self, output_pdb_folder, alignments: dict = None):
        """
        Superposes PDBs according to alignment and writes transformed PDBs to files
        (View with Pymol)

        Parameters
        ----------
        alignments
        output_pdb_folder
        """
        if alignments is None:
            alignments = self.alignment
        output_pdb_folder = Path(output_pdb_folder)
        if not output_pdb_folder.exists():
            output_pdb_folder.mkdir()
        reference_name = self.structures[0].name
        reference_pdb = pd.parsePDB(str(self.output_folder / f"cleaned_pdb/{self.structures[0].name}.pdb"))
        core_indices = np.array([i for i in range(len(alignments[reference_name])) if -1 not in [alignments[n][i] for n in alignments]])
        aln_ref = alignments[reference_name]
        ref_coords_core = reference_pdb[utility.get_alpha_indices(reference_pdb)].getCoords().astype(np.float64)[
            np.array([aln_ref[c] for c in core_indices])]
        ref_centroid = utility.nb_mean_axis_0(ref_coords_core)
        ref_coords_core -= ref_centroid
        transformation = pd.Transformation(np.eye(3), -ref_centroid)
        reference_pdb = pd.applyTransformation(transformation, reference_pdb)
        pd.writePDB(str(output_pdb_folder / f"{reference_name}.pdb"), reference_pdb)
        for i in range(1, len(self.structures)):
            name = self.structures[i].name
            pdb = pd.parsePDB(str(self.output_folder / f"cleaned_pdb/{self.structures[i].name}.pdb"))
            aln_name = alignments[name]
            common_coords_2 = pdb[utility.get_alpha_indices(pdb)].getCoords().astype(np.float64)[np.array([aln_name[c] for c in core_indices])]
            rotation_matrix, translation_matrix = superposition_functions.svd_superimpose(ref_coords_core, common_coords_2)
            transformation = pd.Transformation(rotation_matrix.T, translation_matrix)
            pdb = pd.applyTransformation(transformation, pdb)
            pd.writePDB(str(output_pdb_folder / f"{name}.pdb"), pdb)

    def get_aligned_features(self, dssp_dir, num_threads, force_overwrite, alignments: dict = None):
        """
        Get dict of aligned features
        """
        if alignments is None:
            alignments = self.alignment
        features = feature_extraction.get_features_multiple(protein_utility.parse_pdb_files(self.output_folder / "cleaned_pdb"),
                                                            str(dssp_dir),
                                                            num_threads=num_threads, force_overwrite=force_overwrite)
        feature_names = list(features[0].keys())
        aligned_features = {}
        alignment_length = len(alignments[self.structures[0].name])
        for feature_name in feature_names:
            if feature_name == "secondary":
                continue
            aligned_features[feature_name] = np.zeros((len(self.structures), alignment_length))
            aligned_features[feature_name][:] = np.nan
            for p in range(len(self.structures)):
                farray = features[p][feature_name]
                if "gnm" in feature_name or "anm" in feature_name:
                    farray = farray / np.nansum(farray ** 2) ** 0.5
                indices = [i for i in range(alignment_length) if alignments[self.structures[p].name][i] != '-']
                aligned_features[feature_name][p, indices] = farray
        return aligned_features

    def superpose(self, alignments: dict = None):
        """
        Superposes structures to first structure according to core positions in alignment using Kabsch superposition
        """
        if alignments is None:
            alignments = self.alignment
        reference_index = np.argmax([s.length for s in self.structures])
        reference_key = self.structures[reference_index].name
        # core_indices = np.array([i for i in range(len(alignments[reference_key])) if '-' not in [alignments[n][i] for n in alignments]])
        aln_ref = alignments[reference_key]
        # ref_coords = self.structures[reference_index].coordinates[np.array([aln_ref[c] for c in core_indices])]
        # ref_centroid = general_utility.nb_mean_axis_0(ref_coords)
        # ref_coords -= ref_centroid
        for i in range(len(self.structures)):
            # if i == reference_index:
            #    self.structures[i].coordinates -= ref_centroid
            # else:
            aln_c = alignments[self.structures[i].name]
            common_coords_1, common_coords_2 = get_common_coordinates(self.structures[reference_index].coordinates,
                                                                      self.structures[i].coordinates, aln_ref, aln_c)
            assert common_coords_1.shape[0] > 0
            # common_coords_2 = self.structures[i].coordinates[np.array([aln_c[c] for c in core_indices])]
            rotation_matrix, translation_matrix = superposition_functions.paired_svd_superpose(common_coords_1, common_coords_2)
            self.structures[i].coordinates = superposition_functions.apply_rotran(self.structures[i].coordinates, rotation_matrix,
                                                                                  translation_matrix)

    def make_pairwise_rmsd_coverage_matrix(self, alignments: dict = None, superpose_first: bool = True):
        """
        Find RMSDs and coverages of the alignment of each pair of sequences

        Parameters
        ----------
        alignments
            if None uses self.alignment
        superpose_first
            if True then superposes all structures to first structure first

        Returns
        -------
        RMSD matrix, coverage matrix
        """
        if alignments is None:
            alignments = self.alignment
        num = len(self.structures)
        pairwise_rmsd_matrix = np.zeros((num, num))
        pairwise_rmsd_matrix[:] = np.nan
        pairwise_coverage = np.zeros((num, num))
        pairwise_coverage[:] = np.nan
        if superpose_first:
            self.superpose(alignments)
        for i in range(num - 1):
            for j in range(i + 1, num):
                name_1, name_2 = self.structures[i].name, self.structures[j].name
                aln_1 = alignments[name_1]
                aln_2 = alignments[name_2]
                common_coords_1, common_coords_2 = get_common_coordinates(self.structures[i].coordinates,
                                                                          self.structures[j].coordinates, aln_1, aln_2)
                assert common_coords_1.shape[0] > 0
                if not superpose_first:
                    rot, tran = superposition_functions.paired_svd_superpose(common_coords_1, common_coords_2)
                    common_coords_2 = superposition_functions.apply_rotran(common_coords_2, rot, tran)
                pairwise_rmsd_matrix[i, j] = pairwise_rmsd_matrix[j, i] = score_functions.get_rmsd(common_coords_1, common_coords_2)
                pairwise_coverage[i, j] = pairwise_coverage[j, i] = common_coords_1.shape[0] / len(aln_1)
        return pairwise_rmsd_matrix, pairwise_coverage
