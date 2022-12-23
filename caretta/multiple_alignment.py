from numpy import ndarray

import pickle
import typing
from dataclasses import dataclass
from pathlib import Path

import numba as nb
import numpy as np
import prody as pd
import typer
from numpy.linalg import LinAlgError

from caretta.score_functions import make_score_matrix
from caretta.superposition_functions import paired_svd_superpose_with_subset
from geometricus import moment_invariants
from scipy.spatial.distance import pdist, squareform
from copy import deepcopy
import arviz as az

from caretta import (
    neighbor_joining as nj,
    score_functions,
    superposition_functions,
    feature_extraction,
    helper,
)

from abc import ABC, abstractmethod

from geometricus.model_utility import ShapemerLearn
from geometricus.compare import align_pairwise


def alignment_to_numpy(alignment):
    aln_np = {}
    for n in alignment:
        aln_seq = []
        index = 0
        for a in alignment[n]:
            if a == "-":
                aln_seq.append(-1)
            else:
                aln_seq.append(index)
                index += 1
        aln_np[n] = np.array(aln_seq)
    return aln_np


@nb.njit
def make_coverage_gap_distance_matrix(alignment_array):
    distance_matrix = np.zeros((alignment_array.shape[0], alignment_array.shape[0]))
    matrix_aligning = np.zeros((alignment_array.shape[0], alignment_array.shape[0]), dtype=np.int32)
    for i in range(alignment_array.shape[0]):
        indices_i = np.argwhere(alignment_array[i] != -1)[:, 0]
        length_i = len(indices_i)
        for j in range(alignment_array.shape[0]):
            num_gaps = np.sum(alignment_array[j][indices_i] == -1)
            distance_matrix[i, j] = np.sum(alignment_array[j][indices_i] == -1) / length_i
            matrix_aligning[i, j] = length_i - num_gaps
    return distance_matrix, matrix_aligning


@nb.njit(parallel=True)
def tm_score(coords_1, coords_2, l1, l2):
    d1 = 1.24 * (l1 - 15) ** 1 / 3 - 1.8
    d2 = 1.24 * (l2 - 15) ** 1 / 3 - 1.8
    sum_1 = 0
    sum_2 = 0
    for i in range(coords_1.shape[0]):
        sum_1 += 1 / (1 + (np.sum(coords_1[i] - coords_2[i]) / d1) ** 2)
        sum_2 += 1 / (1 + (np.sum(coords_1[i] - coords_2[i]) / d2) ** 2)
    t1 = (1 / l1) * sum_1
    t2 = (1 / l2) * sum_2
    return max(t1, t2)


@nb.njit
def get_common_vectors(
        vectors_1: np.ndarray, vectors_2: np.ndarray, aln_1: np.ndarray, aln_2: np.ndarray
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    Return residue vectors of positions aligned in both vectors_1 and vectors_2
    """
    assert aln_1.shape == aln_2.shape
    pos_1, pos_2 = helper.get_common_positions(aln_1, aln_2)
    return vectors_1[pos_1], vectors_2[pos_2]


@nb.njit
def get_mean_coords(
        aln_1, coords_1: np.ndarray, aln_2, coords_2: np.ndarray
) -> np.ndarray:
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
            mean_coords[i] = np.array(
                [
                    np.nanmean(np.array([coords_1[x, d], coords_2[y, d]]))
                    for d in range(coords_1.shape[1])
                ]
            )
    return mean_coords


def get_mean_weights(
        weights_1: np.ndarray, weights_2: np.ndarray, aln_1: np.ndarray, aln_2: np.ndarray
) -> np.ndarray:
    mean_weights = np.zeros(aln_1.shape[0])
    for i, (x, y) in enumerate(zip(aln_1, aln_2)):
        if not x == -1:
            mean_weights[i] += weights_1[x]
        if not y == -1:
            mean_weights[i] += weights_2[y]
    return mean_weights


@nb.njit
def get_pairwise_alignment(
        vectors_1,
        vectors_2,
        gamma,
        gap_open_penalty: float,
        gap_extend_penalty: float,
        weights_1: np.ndarray,
        weights_2: np.ndarray,
        n_iter=3,
):
    score_matrix = score_functions.make_score_matrix(
        np.hstack((vectors_1, weights_1)),
        np.hstack((vectors_2, weights_2)),
        score_functions.get_caretta_score,
        gamma,
    )
    dtw_aln_array_1, dtw_aln_array_2, dtw_score = align_pairwise.dtw_align(
        np.arange(vectors_1.shape[0]), np.arange(vectors_2.shape[0]),
        score_matrix, gap_open_penalty, gap_extend_penalty
    )
    # for i in range(n_iter):
    #     pos_1, pos_2 = helper.get_common_positions(dtw_aln_array_1, dtw_aln_array_2)
    #     common_coords_1, common_coords_2 = coords_1[pos_1], coords_2[pos_2]
    #     (
    #         c1,
    #         c2,
    #         common_coords_2,
    #     ) = superposition_functions.paired_svd_superpose_with_subset(
    #         coords_1, coords_2, common_coords_1, common_coords_2
    #     )
    #     score_matrix = score_functions.make_score_matrix(
    #         np.hstack((c1, weights_1)),
    #         np.hstack((c2, weights_2)),
    #         score_functions.get_caretta_score,
    #         gamma,
    #     )
    #     aln_1, aln_2, score = dtw.dtw_align(
    #         score_matrix, gap_open_penalty, gap_extend_penalty
    #     )
    #     if score > dtw_score:
    #         coords_1 = c1
    #         coords_2 = c2
    #         dtw_score = score
    #         dtw_aln_array_1 = aln_1
    #         dtw_aln_array_2 = aln_2
    #     else:
    #         break
    return dtw_aln_array_1, dtw_aln_array_2, dtw_score, vectors_1, vectors_2


@dataclass
class OutputFiles:
    output_folder: Path = Path("./caretta_results")
    fasta_file: Path = Path("./result.fasta")
    pdb_folder: Path = Path("./result_pdb/")
    cleaned_pdb_folder: Path = Path("./cleaned_pdb")
    matrix_file: Path = Path("./result_matrix.pkl")
    feature_file: Path = Path("./result_features.pkl")
    class_file: Path = Path("./result_class.pkl")
    tmp_folder: Path = Path("./tmp/")

    @classmethod
    def from_folder(cls, output_folder):
        return cls(output_folder / "result.fasta", output_folder / "result_pdb/", output_folder / "cleaned_pdb",
                   output_folder / "result_matrix.pkl", output_folder / "result_features.pkl",
                   output_folder / "result_class.pkl", output_folder / "tmp/")


DEFAULT_SUPERPOSITION_PARAMETERS = {
    # must-have
    "gap_open_penalty": 0.0,
    "gap_extend_penalty": 0.0,
    "gamma": 0.03,
    # changes per superposition_function
    "split_type": "KMER",
    "split_size": 20,
    "scale": True,
    "gamma_moment": 0.6,
    "n_iter": 3,
}


# abstract class for Protein implementing score_function and mean_function
class SequenceBase(ABC):
    @abstractmethod
    def score_function(self, other: "SequenceBase", **kwargs) -> np.ndarray:
        pass

    @abstractmethod
    def mean_function(self, other: "SequenceBase", aln_1: np.ndarray, aln_2: np.ndarray, name_int: str,
                      **kwargs) -> "SequenceBase":
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass


class MultipleAlignmentBase(ABC):
    @abstractmethod
    def get_pairwise_distances_fast(self, **kwargs) -> np.ndarray:
        pass

    @abstractmethod
    def make_tree(self, pairwise_distance_matrix, **kwargs):
        # self.tree, self.branch_lengths = nj.neighbor_joining(self.pairwise_distance_matrix)
        pass

    @abstractmethod
    def progressive_align(self, tree, gap_open_penalty, gap_extend_penalty, **kwargs) -> typing.Tuple[
        typing.Dict[str, np.ndarray],
        typing.List[SequenceBase]]:
        pass

    @abstractmethod
    def multiple_align(self, **kwargs) -> typing.Dict[str, np.ndarray]:
        """
        pairwise_distance_matrix = self.get_pairwise_distances()
        tree = self.make_tree(pairwise_distance_matrix)
        alignment = self.progressive_align(tree)
        """
        pass


@dataclass
class MultipleAlignment(MultipleAlignmentBase, ABC):
    sequences: typing.List[SequenceBase]
    tree: typing.Optional[np.ndarray] = None
    branch_lengths: typing.Optional[np.ndarray] = None
    alignment: typing.Optional[typing.Dict[str, np.ndarray]] = None
    final_sequences: typing.Optional[typing.List[SequenceBase]] = None

    def get_pairwise_distances_fast(self, **kwargs) -> np.ndarray:
        pass

    def make_tree(self, pairwise_distance_matrix, **kwargs):
        tree, branch_lengths = nj.neighbor_joining(pairwise_distance_matrix)
        return tree, branch_lengths

    def progressive_align(self, tree,
                          gap_open_penalty,
                          gap_extend_penalty, **kwargs) -> typing.Tuple[typing.Dict[str, np.ndarray],
                                                                        typing.List[SequenceBase]]:
        final_sequences = [s for s in self.sequences]
        final_alignments = {
            s.name: {s.name: np.arange(len(s))} for s in final_sequences
        }

        def make_intermediate_node(n1, n2, n_int):
            name_1, name_2 = (
                final_sequences[n1].name,
                final_sequences[n2].name,
            )

            name_int = f"int-{n_int}"
            score_matrix = final_sequences[n1].score_function(
                final_sequences[n2]
            )
            aln_1, aln_2, score = align_pairwise.dtw_align(
                np.arange(score_matrix.shape[0]), np.arange(score_matrix.shape[0]),
                score_matrix, gap_open_penalty=gap_open_penalty, gap_extend_penalty=gap_extend_penalty
            )
            intermediate_sequence = final_sequences[n1].mean_function(final_sequences[n2], aln_1, aln_2, name_int)

            final_alignments[name_1] = {
                name: np.array([sequence[i] if i != -1 else -1 for i in aln_1])
                for name, sequence in final_alignments[name_1].items()
            }
            final_alignments[name_2] = {
                name: np.array([sequence[i] if i != -1 else -1 for i in aln_2])
                for name, sequence in final_alignments[name_2].items()
            }
            final_alignments[name_int] = {
                **final_alignments[name_1],
                **final_alignments[name_2],
            }

            final_sequences.append(
                intermediate_sequence
            )

        with typer.progressbar(
                range(0, tree.shape[0] - 1, 2), label="Aligning"
        ) as progress:
            for x in progress:
                node_1, node_2, node_int = (
                    tree[x, 0],
                    tree[x + 1, 0],
                    tree[x, 1],
                )
                assert tree[x + 1, 1] == node_int
                make_intermediate_node(node_1, node_2, node_int)
        node_1, node_2 = tree[-1, 0], tree[-1, 1]
        make_intermediate_node(node_1, node_2, "final")
        alignment = {
            **final_alignments[final_sequences[node_1].name],
            **final_alignments[final_sequences[node_2].name],
        }
        return alignment, final_sequences

    def multiple_align(self, gap_open_penalty, gap_extend_penalty, **kwargs) -> typing.Dict[str, np.ndarray]:
        if len(self.sequences) == 2:
            score_matrix = self.sequences[0].score_function(
                self.sequences[1]
            )
            aln_1, aln_2, score = align_pairwise.dtw_align(
                np.arange(score_matrix.shape[0]), np.arange(score_matrix.shape[0]),
                score_matrix, gap_open_penalty=gap_open_penalty, gap_extend_penalty=gap_extend_penalty
            )
            self.alignment = {
                self.sequences[0].name: aln_1,
                self.sequences[1].name: aln_2,
            }
            return self.alignment

        pairwise_distance_matrix = self.get_pairwise_distances_fast()
        self.tree, self.branch_lengths = self.make_tree(pairwise_distance_matrix)
        self.alignment, self.final_sequences = self.progressive_align(self.tree, gap_open_penalty, gap_extend_penalty)
        return self.alignment

    def to_sequence_alignment(self, alignment=None):
        """
        Convert the multiple alignment of residue indices into to amino acid strings
        """
        if alignment is None:
            alignment = self.alignment
        sequence_alignment = {}
        for p in self.sequences:
            sequence = str(p)
            sequence_alignment[p.name] = "".join(sequence[i] if i != -1 else "-" for i in alignment[p.name])
        return sequence_alignment

    def write_alignment(self, fasta_file, alignment=None):
        """
        Writes alignment to a fasta file
        """
        if alignment is None:
            alignment = self.alignment
        with open(fasta_file, "w") as f:
            for p in self.sequences:
                sequence = str(p)
                aligned_sequence = "".join(sequence[i] if i != -1 else "-" for i in alignment[p.name])
                f.write(f">{p.name}\n{aligned_sequence}\n")


@nb.njit
def get_gaussian_score(vector_1: np.ndarray, vector_2: np.ndarray, gamma=0.03):
    """
    Gaussian (RBF) score of similarity between two coordinates
    """
    return np.exp(-gamma * np.sum((vector_1 - vector_2) ** 2, axis=-1))


@nb.njit
def make_score_matrix(
        vector_1: np.ndarray, vector_2: np.ndarray, gamma, normalized=False
) -> np.ndarray:
    score_matrix = np.zeros((vector_1.shape[0], vector_2.shape[0]))

    if normalized:
        both = np.concatenate((vector_1, vector_2))
        mean, std = helper.nb_mean_axis_0(both), helper.nb_std_axis_0(both)
        vector_1 = (vector_1 - mean) / std
        vector_2 = (vector_2 - mean) / std
    for i in range(vector_1.shape[0]):
        for j in range(vector_2.shape[0]):
            score_matrix[i, j] = get_gaussian_score(vector_1[i], vector_2[j], gamma)
    return score_matrix


@dataclass(eq=False)
class Protein(SequenceBase, ABC):
    name: str
    tensors: np.ndarray
    """Geometricus tensor representations of the protein"""
    coordinates: np.ndarray = None
    """Coordinates of the protein"""
    sequence: str = ""
    """Amino acid sequence of the protein"""
    gamma_tensor: float = 0.03
    """Gamma parameter for the tensor score function"""
    gamma_coords: float = 0.03
    """Gamma parameter for the coordinate score function"""

    def score_function(self, other: "Protein", flexible=True) -> np.ndarray:
        if flexible:
            return make_score_matrix(self.tensors, other.tensors, self.gamma_tensor)
        else:
            score_matrix = make_score_matrix(self.tensors, other.tensors)
            aln_1, aln_2, score = align_pairwise.dtw_align(score_matrix)
            pos_1, pos_2 = helper.get_common_positions(aln_1, aln_2)
            common_coords_1, common_coords_2 = self.coordinates[pos_1], other.coordinates[pos_2]
            coords_1, coords_2, common_coords_2 = paired_svd_superpose_with_subset(
                self.coordinates[pos_1], other.coordinates[pos_2], common_coords_1, common_coords_2)
            return make_score_matrix(coords_1, coords_2, self.gamma_coords)

    def mean_function(self, other: "Protein", aln_1: np.ndarray, aln_2: np.ndarray, name_int: str,
                      flexible=True, **kwargs) -> "Protein":
        tensors_mean = np.zeros((len(aln_1), self.tensors.shape[1]))
        for i, (x, y) in enumerate(zip(aln_1, aln_2)):
            if x == -1:
                tensors_mean[i] = other.tensors[y]
            elif y == -1:
                tensors_mean[i] = self.tensors[x]
            else:
                tensors_mean[i] = (self.tensors[x] + other.tensors[y]) / 2
        if flexible:
            return Protein(name_int, tensors_mean)
        pos_1, pos_2 = helper.get_common_positions(aln_1, aln_2)
        common_coords_1, common_coords_2 = self.coordinates[pos_1], other.coordinates[pos_2]
        coords_1, coords_2, common_coords_2 = paired_svd_superpose_with_subset(
            self.coordinates[pos_1], other.coordinates[pos_2], common_coords_1, common_coords_2
        )
        coordinates_mean = np.zeros((len(aln_1), self.coordinates.shape[1]))
        for i, (x, y) in enumerate(zip(aln_1, aln_2)):
            if x == -1:
                coordinates_mean[i] = coords_2[y]
            elif y == -1:
                coordinates_mean[i] = coords_1[x]
            else:
                coordinates_mean[i] = (coords_1[x] + coords_2[y]) / 2
        return Protein(name_int, tensors_mean, coordinates_mean)

    def __len__(self) -> int:
        return self.tensors.shape[0]

    @property
    def name(self) -> str:
        return self.name

    def __str__(self):
        return self.sequence


def align_from_structure_files(
        input_files: typing.Union[typing.List[str], Path, str],
        model: ShapemerLearn,
        gap_open_penalty: float = 1.0,
        gap_extend_penalty: float = 0.01,
        consensus_weight: float = 1.0,
        full: bool = False,
        output_folder: typing.Union[str, Path] = Path("../caretta_results"),
        num_threads: int = 20,
        write_fasta: bool = False,
        write_pdb: bool = False,
        write_features: bool = False,
        only_dssp: bool = True,
        write_class: bool = False,
        write_matrix: bool = False,
        verbose: bool = True,
):
    """
    Caretta aligns protein structures and can output a sequence alignment, superposed PDB files,
    a set of aligned feature matrices and a class with intermediate structures made during progressive alignment.

        Parameters
        ----------
        input_files
            Can be \n
            A list of structure files (.pdb, .pdb.gz, .cif, .cif.gz),
            A list of (structure_file, chain)
            A list of PDBIDs or PDBID_chain or (PDB ID, chain)
            A folder with input structure files,
            A file which lists structure filenames or "structure_filename, chain" on each line,
            A file which lists PDBIDs or PDBID_chain or PDBID, chain on each line
        gap_open_penalty
            default 1
        gap_extend_penalty
            default 0.01
        consensus_weight
            default 1
        full
            True =>  Uses all-vs-all pairwise Caretta alignment to make the distance matrix (much slower)
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
        only_dssp
            True => only DSSP features extracted
            False => all features
        write_class
            True => writes StructureMultiple class with intermediate structures and tree to pickle file (default True)
            writes to output_folder / result_class.pkl
        write_matrix
            True => writes distance matrix to text file (default False)
            writes to output_folder / matrix.matscance
        verbose
            controls verbosity

    Returns
    -------
    MultipleAlignment class
    """
    if output_folder is None:
        output_files = OutputFiles()
    else:
        output_files = OutputFiles(output_folder=Path(output_folder))
    if not output_files.output_folder.exists():
        output_files.output_folder.mkdir()

    if not output_files.cleaned_pdb_folder.exists():
        output_files.cleaned_pdb_folder.mkdir()
    pdb_files = helper.parse_protein_files_and_clean(input_files, output_files.cleaned_pdb_folder)
    if verbose:
        typer.echo(f"Found {len(pdb_files)} structure files")
    protein_moments, errors = moment_invariants.get_invariants_for_files(pdb_files,
                                                                         n_threads=num_threads,
                                                                         verbose=verbose)
    if verbose:
        typer.echo(f"Found {len(protein_moments)} structures with valid invariants")

    proteins = [Protein(m.name, m.get_tensor_model(model), m.calpha_coordinates, m.sequence) for m in protein_moments]
    msa_class = MultipleAlignment(proteins)

    if len(msa_class.sequences) > 2:
        if full:
            msa_class.make_pairwise_dtw_matrix(
                gap_open_penalty, gap_extend_penalty, verbose=verbose
            )
        else:
            protein_to_shapemer_indices = [m.get_shapemer_indices_model(model) for m in protein_moments]

        alignment = msa_class.multiple_align(verbose=verbose)
    else:
        alignment = msa_class.multiple_align(
            gap_open_penalty=gap_open_penalty,
            gap_extend_penalty=gap_extend_penalty,
            verbose=verbose,
        )
    if verbose and any(
            (write_fasta, write_pdb, write_pdb, write_class, write_matrix)
    ):
        typer.echo("Writing files...")

    write_files(
        alignment,
        output_files,
        write_fasta=write_fasta,
        write_pdb=write_pdb,
        write_features=write_features,
        write_class=write_class,
        write_matrix=write_matrix,
        num_threads=num_threads,
        only_dssp=only_dssp,
        verbose=verbose,
    )
    return msa_class


def write_files(
        alignment,
        output_files,
        write_fasta,
        write_pdb,
        write_features,
        write_class,
        write_matrix,
        only_dssp=True,
        num_threads=4,
        verbose: bool = False,
):
    if verbose and any(
            (write_fasta, write_pdb, write_pdb, write_class, write_matrix)
    ):
        typer.echo("Writing files...")
    if write_fasta:
        self.write_alignment()
        if verbose:
            typer.echo(
                f"FASTA file: {typer.style(str(output_files.fasta_file), fg=typer.colors.GREEN)}",
            )
    if write_pdb:
        if not output_files.pdb_folder.exists():
            output_files.pdb_folder.mkdir()
        write_superposed_pdbs(output_files.cleaned_pdb_folder,
                              alignment,
                              output_files.pdb_folder, verbose=verbose)
        if verbose:
            typer.echo(
                f"Superposed PDB files: {typer.style(str(output_files.pdb_folder), fg=typer.colors.GREEN)}"
            )
    if write_features:
        if not output_files.tmp_folder.exists():
            output_files.tmp_folder.mkdir()
        names, features = get_aligned_features(
            alignment, output_files.cleaned_pdb_folder, str(output_files.tmp_folder),
            only_dssp=only_dssp, num_threads=num_threads
        )
        with open(output_files.feature_file, "wb") as f:
            pickle.dump((names, features), f)
        if verbose:
            typer.echo(
                f"Aligned features: {typer.style(str(output_files.feature_file), fg=typer.colors.GREEN)}"
            )
    if write_class:
        with open(output_files.class_file, "wb") as f:
            pickle.dump(self, f)
        if verbose:
            typer.echo(
                f"Class file: {typer.style(str(output_files.class_file), fg=typer.colors.GREEN)}"
            )
    if write_matrix:
        helper.write_distance_matrix(
            [s.name for s in self.structures],
            self.pairwise_distance_matrix,
            output_files.matrix_file,
        )
        if verbose:
            typer.echo(
                f"Distance matrix file: {typer.style(str(output_files.matrix_file), fg=typer.colors.GREEN)}"
            )


# @dataclass
# class ProteinMultiple:
#     """
#     Class for multiple protein alignment
#
#     Constructor Arguments
#     ---------------------
#     proteins: List[Protein]
#         list of Protein objects
#     superposition_parameters # TODO: remove
#         dictionary of parameters to pass to the superposition function
#     superposition_function # TODO: remove
#         a function that takes two coordinate sets as input and superposes them
#         returns a score, superposed_coords_1, superposed_coords_2
#     score_function # TODO: remove
#         a function that takes two paired coordinate sets (same shape) and returns a score
#     consensus_weight
#         weights the effect of well-aligned columns on the progressive alignment
#     final_structures # TODO: rename, final_proteins
#         makes the progressive alignment tree of structures, last one is the consensus structure of the multiple alignment
#     tree
#         neighbor joining tree of indices controlling alignment - indices beyond len(structures) refer to intermediate nodes
#     branch_lengths
#     alignment
#         indices of aligning residues from each structure, gaps are -1s
#     """
#
#     proteins: typing.List[SequenceBase]
#     sequences: typing.Dict[str, str]
#     superposition_parameters: typing.Dict[str, typing.Any] = field(
#         default_factory=lambda: DEFAULT_SUPERPOSITION_PARAMETERS)
#     superposition_function: typing.Callable[
#         [
#             np.ndarray,
#             np.ndarray,
#             dict,
#             typing.Callable[[np.ndarray, np.ndarray], float],
#         ],
#         typing.Tuple[float, np.ndarray, np.ndarray],
#     ] = lambda x, y: (0, x, y)
#     mean_function: typing.Callable[
#         [np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray
#     ] = get_mean_coords
#     consensus_weight: float = 1.0
#     pairwise_distance_matrix: typing.Union[None, np.ndarray] = None
#     reference_structure_index: typing.Union[None, int] = None
#     final_proteins: typing.Union[None, typing.List[Protein]] = None
#     final_consensus_weights: typing.Union[None, typing.List[np.ndarray]] = None
#     tree: typing.Union[None, np.ndarray] = None
#     branch_lengths: typing.Union[None, np.ndarray] = None
#     alignment: typing.Union[dict, None] = None
#     output_files: OutputFiles = OutputFiles()
#
#     @staticmethod
#     def align_from_structure_files(
#             input_files: typing.Union[typing.List[str], Path, str],
#             gap_open_penalty: float = 1.0,
#             gap_extend_penalty: float = 0.01,
#             consensus_weight: float = 1.0,
#             full: bool = False,
#             output_folder: typing.Union[str, Path] = Path("../caretta_results"),
#             num_threads: int = 20,
#             write_fasta: bool = False,
#             write_pdb: bool = False,
#             write_features: bool = False,
#             only_dssp: bool = True,
#             write_class: bool = False,
#             write_matrix: bool = False,
#             verbose: bool = True,
#     ):
#         """
#         Caretta aligns protein structures and can output a sequence alignment, superposed PDB files,
#         a set of aligned feature matrices and a class with intermediate structures made during progressive alignment.
#
#         Parameters
#         ----------
#         input_files
#             Can be \n
#             A list of structure files (.pdb, .pdb.gz, .cif, .cif.gz),
#             A list of (structure_file, chain)
#             A list of PDBIDs or PDBID_chain or (PDB ID, chain)
#             A folder with input structure files,
#             A file which lists structure filenames or "structure_filename, chain" on each line,
#             A file which lists PDBIDs or PDBID_chain or PDBID, chain on each line
#         gap_open_penalty
#             default 1
#         gap_extend_penalty
#             default 0.01
#         consensus_weight
#             default 1
#         full
#             True =>  Uses all-vs-all pairwise Caretta alignment to make the distance matrix (much slower)
#         output_folder
#             default "caretta_results"
#         num_threads
#             Number of threads to use for feature extraction
#         write_fasta
#             True => writes alignment as fasta file (default True)
#             writes to output_folder / result.fasta
#         write_pdb
#             True => writes all protein PDB files superposed by alignment (default True)
#              writes to output_folder / superposed_pdb
#         write_features
#             True => extracts and writes aligned features as a dictionary of numpy arrays into a pickle file (default True)
#             writes to output_folder / result_features.pkl
#         only_dssp
#             True => only DSSP features extracted
#             False => all features
#         write_class
#             True => writes StructureMultiple class with intermediate structures and tree to pickle file (default True)
#             writes to output_folder / result_class.pkl
#         write_matrix
#             True => writes distance matrix to text file (default False)
#             writes to output_folder / matrix.mat
#         verbose
#             controls verbosity
#
#         Returns
#         -------
#         StructureMultiple class
#         """
#
#         msa_class = ProteinMultiple.from_structure_files(
#             input_files,
#             superposition_parameters=DEFAULT_SUPERPOSITION_PARAMETERS,
#             superposition_function=superposition_functions.moment_svd_superpose_function,
#             consensus_weight=consensus_weight,
#             output_folder=output_folder,
#             verbose=verbose,
#         )
#         if len(msa_class.proteins) > 2:
#             if full:
#                 msa_class.make_pairwise_dtw_matrix(
#                     gap_open_penalty, gap_extend_penalty, verbose=verbose
#                 )
#             else:
#                 msa_class.make_pairwise_shape_matrix(
#                     verbose=verbose
#                 )
#             msa_class.align(gap_open_penalty, gap_extend_penalty, verbose=verbose)
#         else:
#             msa_class.reference_structure_index = 0
#             msa_class.align(
#                 gap_open_penalty=gap_open_penalty,
#                 gap_extend_penalty=gap_extend_penalty,
#                 verbose=verbose,
#             )
#
#         msa_class.write_files(
#             write_fasta=write_fasta,
#             write_pdb=write_pdb,
#             write_features=write_features,
#             write_class=write_class,
#             write_matrix=write_matrix,
#             num_threads=num_threads,
#             only_dssp=only_dssp,
#             verbose=verbose,
#         )
#         return msa_class
#
#     @classmethod
#     def from_structure_files(
#             cls,
#             input_files,
#             split_infos=moment_invariants.SPLIT_INFOS,
#             moment_types=moment_invariants.MOMENT_TYPES,
#             resolution=None,
#             model=None,
#             superposition_parameters=None,
#             superposition_function=superposition_functions.moment_svd_superpose_function,
#             consensus_weight=1.0,
#             output_folder=Path("./caretta_results"),
#             n_threads=1,
#             verbose: bool = False,
#     ):
#         """
#         Makes a StructureMultiple object from a list of pdb files/names or a folder of pdb files
#
#         Parameters
#         ----------
#         input_files
#             list of structure files/names or a folder containing pdb files
#         superposition_parameters
#             parameters to give to the superposition function
#         superposition_function
#             a function that takes two coordinate sets as input and superposes them
#             returns a score, superposed_coords_1, superposed_coords_2
#         consensus_weight
#             weights the effect of well-aligned columns on the progressive alignment
#         output_folder
#         n_threads
#         verbose
#
#         Returns
#         -------
#         ProteinMultiple object (unaligned)
#         """
#         if superposition_parameters is None:
#             superposition_parameters = DEFAULT_SUPERPOSITION_PARAMETERS
#         if output_folder is None:
#             output_files = OutputFiles()
#         else:
#             output_files = OutputFiles(output_folder=Path(output_folder))
#         if not output_files.output_folder.exists():
#             output_files.output_folder.mkdir()
#
#         if not output_files.cleaned_pdb_folder.exists():
#             output_files.cleaned_pdb_folder.mkdir()
#         pdb_files = helper.parse_protein_files_and_clean(input_files, output_files.cleaned_pdb_folder)
#         if verbose:
#             typer.echo(f"Found {len(pdb_files)} structure files")
#         protein_moments, errors = moment_invariants.get_invariants_for_files(pdb_files, split_infos=split_infos,
#                                                                              moment_types=moment_types,
#                                                                              n_threads=n_threads, verbose=verbose)
#         if verbose:
#             typer.echo(f"Found {len(protein_moments)} structures with valid invariants")
#
#         proteins = []
#         structures = []
#         sequences = {}
#         for m_class in protein_moments:
#             coordinates = m_class.calpha_coordinates
#             if model is None:
#                 proteins.append(Protein(m_class.name,
#                                         coordinates.shape[0],
#                                         m_class.get_shapemers_binned(resolution)))
#             else:
#                 proteins.append(Protein(m_class.name, coordinates.shape[0],
#                                         m_class.get_shapemers_model(model)))
#             structures.append(Structure(m_class.name, coordinates.shape[0], coordinates))
#             sequences[m_class.name] = m_class.sequence
#         msa_class = ProteinMultiple(
#             proteins,
#             sequences,
#             structures=structures,
#             superposition_parameters=superposition_parameters,
#             superposition_function=superposition_function,
#             consensus_weight=consensus_weight,
#             output_files=output_files,
#         )
#         return msa_class
#
#     @classmethod
#     def from_coordinates(
#             cls,
#             names: typing.List[str],
#             coordinates_list: typing.List[np.ndarray],
#             sequences: typing.List[str],
#             superposition_parameters: dict,
#             superposition_function=superposition_functions.moment_multiple_svd_superpose_function,
#             consensus_weight=1.0,
#             output_folder=Path("./caretta_results"),
#     ):
#         """
#         Makes a StructureMultiple object from a list of coordinates
#
#         Parameters
#         ----------
#         names
#         coordinates_list
#         sequences
#         superposition_parameters
#             parameters to give to the superposition function
#         superposition_function
#             a function that takes two coordinate sets as input and superposes them
#             returns a score, superposed_coords_1, superposed_coords_2
#         consensus_weight
#             weights the effect of well-aligned columns on the progressive alignment
#         output_folder
#
#         Returns
#         -------
#         ProteinMultiple object (unaligned)
#         """
#         if output_folder is None:
#             output_files = OutputFiles()
#         else:
#             output_files = OutputFiles(output_folder=Path(output_folder))
#         if not output_files.output_folder.exists():
#             output_files.output_folder.mkdir()
#         sequences = {n: s for n, s in zip(names, sequences)}
#         structures = []
#         for name, coordinates in zip(names, coordinates_list):
#             structures.append(Protein(name, coordinates.shape[0], coordinates))
#         msa_class = ProteinMultiple(
#             structures,
#             sequences,
#             superposition_parameters=superposition_parameters,
#             superposition_function=superposition_function,
#             consensus_weight=consensus_weight,
#             output_files=output_files,
#         )
#         return msa_class
#
#     def get_pairwise_alignment(
#             self,
#             vectors_1,
#             vectors_2,
#             gap_open_penalty: float,
#             gap_extend_penalty: float,
#             weight=False,
#             weights_1=None,
#             weights_2=None,
#             n_iter=3,
#             verbose: bool = False,
#     ):
#         """
#         Aligns vectors_1 to vectors_2 by first superposing and then running dtw on the score matrix of
#         the superposed vector sets
#         Parameters
#         ----------
#         vectors_1
#         vectors_2
#         gap_open_penalty
#         gap_extend_penalty
#         weight
#         weights_1
#         weights_2
#         n_iter
#         verbose
#
#         Returns
#         -------
#         alignment_1, alignment_2, score, superposed_coords_1, superposed_coords_2
#         """
#         _, vectors_1, vectors_2 = self.superposition_function(
#             vectors_1, vectors_2, self.superposition_parameters
#         )
#
#         if weight:
#             assert weights_1 is not None
#             assert weights_2 is not None
#             weights_1 = weights_1.reshape(-1, 1)
#             weights_2 = weights_2.reshape(-1, 1)
#         else:
#             weights_1 = np.zeros((vectors_1.shape[0], 1))
#             weights_2 = np.zeros((vectors_2.shape[0], 1))
#         return get_pairwise_alignment(
#             vectors_1,
#             vectors_2,
#             self.superposition_parameters["gamma"],
#             gap_open_penalty,
#             gap_extend_penalty,
#             weights_1,
#             weights_2,
#             n_iter,
#         )
#
#     def make_pairwise_shape_matrix(
#             self,
#             resolution: typing.Union[float, np.ndarray] = 2.0,
#             parameters: dict = None,
#             metric="braycurtis",
#             verbose: bool = False,
#     ):
#         """
#         Makes an all vs. all matrix of distance scores between all the structures.
#
#         Parameters
#         ----------
#         resolution
#         parameters
#             to use for making invariants
#             needs to have
#             num_split_types
#             split_type_i
#             split_size_i
#         metric
#             distance metric (accepts any metric supported by scipy.spatial.distance
#         verbose
#         Returns
#         -------
#         [n x n] distance matrix
#         """
#         if verbose:
#             typer.echo("Calculating pairwise distances...")
#         if parameters is None:
#             parameters = dict(num_split_types=1, split_type_0="KMER", split_size_0=20)
#         embedders = []
#         for i in range(parameters["num_split_types"]):
#             invariants = (
#                 MomentInvariants.from_coordinates(
#                     s.name,
#                     s.coordinates,
#                     None,
#                     split_size=parameters[f"split_size_{i}"],
#                     split_type=SplitType[parameters[f"split_type_{i}"]],
#                 )
#                 for s in self.proteins
#             )
#             embedders.append(
#                 Geometricus.from_invariants(
#                     invariants,
#                     resolution=resolution,
#                     protein_keys=[s.name for s in self.proteins],
#                 )
#             )
#         distance_matrix = squareform(
#             pdist(
#                 np.hstack([embedder.embedding for embedder in embedders]),
#                 metric=metric,
#             )
#         )
#         self.reference_structure_index = np.argmin(
#             np.median(distance_matrix, axis=0)
#         )
#         self.pairwise_distance_matrix = distance_matrix
#
#     def make_pairwise_dtw_matrix(
#             self,
#             gap_open_penalty: float,
#             gap_extend_penalty: float,
#             invert=True,
#             verbose: bool = False,
#     ):
#         """
#         Makes an all vs. all matrix of distance (or similarity) scores between all the structures using pairwise alignment.
#
#         Parameters
#         ----------
#         gap_open_penalty
#         gap_extend_penalty
#         invert
#             if True returns distance matrix
#             if False returns similarity matrix
#         verbose
#         Returns
#         -------
#         [n x n] matrix
#         """
#         if verbose:
#             typer.echo("Calculating pairwise distances...")
#         pairwise_matrix = np.zeros((len(self.proteins), len(self.proteins)))
#         for i in range(pairwise_matrix.shape[0] - 1):
#             for j in range(i + 1, pairwise_matrix.shape[1]):
#                 vectors_1, vectors_2 = (
#                     self.proteins[i].representation,
#                     self.proteins[j].representation,
#                 )
#                 (
#                     dtw_aln_1,
#                     dtw_aln_2,
#                     score,
#                     vectors_1,
#                     vectors_2,
#                 ) = self.get_pairwise_alignment(
#                     vectors_1,
#                     vectors_2,
#                     gap_open_penalty=gap_open_penalty,
#                     gap_extend_penalty=gap_extend_penalty,
#                     weight=False,
#                 )
#                 common_vectors_1, common_vectors_2 = get_common_vectors(
#                     vectors_1, vectors_2, dtw_aln_1, dtw_aln_2
#                 )
#                 pairwise_matrix[i, j] = score_functions.get_total_score(
#                     common_vectors_1,
#                     common_vectors_2,
#                     score_functions.get_caretta_score,
#                     self.superposition_parameters["gamma"],
#                     True,
#                 )
#                 if invert:
#                     pairwise_matrix[i, j] *= -1
#         pairwise_matrix += pairwise_matrix.T
#         self.reference_structure_index = np.argmin(
#             np.median(pairwise_matrix, axis=0)
#         )
#         self.pairwise_distance_matrix = pairwise_matrix
#
#     def align(
#             self,
#             gap_open_penalty,
#             gap_extend_penalty,
#             return_sequence: bool = True,
#             verbose: bool = False,
#     ) -> dict:
#         """
#         Makes a multiple protein alignment
#
#         Parameters
#         ----------
#         gap_open_penalty
#         gap_extend_penalty
#         return_sequence
#             if True returns sequence alignment
#             else indices of aligning residues with gaps as -1s
#         verbose
#         Returns
#         -------
#         alignment = {name: indices of aligning residues with gaps as -1s}
#         """
#         if len(self.proteins) == 2:
#             vector_1, vector_2 = (
#                 self.proteins[0].representation,
#                 self.proteins[1].representation,
#             )
#             dtw_1, dtw_2, _, _, _ = self.get_pairwise_alignment(
#                 vector_1,
#                 vector_2,
#                 gap_open_penalty=gap_open_penalty,
#                 gap_extend_penalty=gap_extend_penalty,
#                 weight=False,
#                 n_iter=self.superposition_parameters["n_iter"],
#                 verbose=verbose,
#             )
#             self.alignment = {
#                 self.proteins[0].name: dtw_1,
#                 self.proteins[1].name: dtw_2,
#             }
#             if return_sequence:
#                 return self.make_sequence_alignment()
#             else:
#                 return self.alignment
#         assert self.pairwise_distance_matrix is not None
#         assert self.pairwise_distance_matrix.shape[0] == len(self.proteins)
#         if verbose:
#             typer.echo("Constructing neighbor joining tree...")
#         self.tree, self.branch_lengths = nj.neighbor_joining(self.pairwise_distance_matrix)
#         self.final_proteins = [s for s in self.proteins]
#         self.final_consensus_weights = [
#             np.full(
#                 (self.proteins[i].representation.shape[0], 1),
#                 self.consensus_weight,
#                 dtype=np.float64,
#             )
#             for i in range(len(self.proteins))
#         ]
#         msa_alignments = {
#             s.name: {s.name: np.arange(s.length)} for s in self.final_proteins
#         }
#
#         def make_intermediate_node(n1, n2, n_int):
#             name_1, name_2 = (
#                 self.final_proteins[n1].name,
#                 self.final_proteins[n2].name,
#             )
#
#             name_int = f"int-{n_int}"
#             n1_vectors = self.final_proteins[n1].representation
#             n1_weights = self.final_consensus_weights[n1]
#             n1_weights *= len(msa_alignments[name_2])
#             n1_weights /= 2 * (
#                     len(msa_alignments[name_2]) + len(msa_alignments[name_1])
#             )
#             n2_vectors = self.final_proteins[n2].representation
#             n2_weights = self.final_consensus_weights[n2]
#             n2_weights *= len(msa_alignments[name_1])
#             n2_weights /= 2 * (
#                     len(msa_alignments[name_2]) + len(msa_alignments[name_1])
#             )
#             (
#                 dtw_aln_1,
#                 dtw_aln_2,
#                 score,
#                 n1_vectors,
#                 n2_vectors,
#             ) = self.get_pairwise_alignment(
#                 n1_vectors,
#                 n2_vectors,
#                 gap_open_penalty=gap_open_penalty,
#                 gap_extend_penalty=gap_extend_penalty,
#                 weight=True,
#                 weights_1=n1_weights,
#                 weights_2=n2_weights,
#                 n_iter=self.superposition_parameters["n_iter"],
#             )
#             n1_weights *= (
#                     2
#                     * (len(msa_alignments[name_2]) + len(msa_alignments[name_1]))
#                     / len(msa_alignments[name_2])
#             )
#             n2_weights *= (
#                     2
#                     * (len(msa_alignments[name_2]) + len(msa_alignments[name_1]))
#                     / len(msa_alignments[name_1])
#             )
#             msa_alignments[name_1] = {
#                 name: np.array([sequence[i] if i != -1 else -1 for i in dtw_aln_1])
#                 for name, sequence in msa_alignments[name_1].items()
#             }
#             msa_alignments[name_2] = {
#                 name: np.array([sequence[i] if i != -1 else -1 for i in dtw_aln_2])
#                 for name, sequence in msa_alignments[name_2].items()
#             }
#             msa_alignments[name_int] = {
#                 **msa_alignments[name_1],
#                 **msa_alignments[name_2],
#             }
#
#             mean_vectors = self.mean_function(dtw_aln_1, n1_vectors, dtw_aln_2, n2_vectors)
#             mean_weights = get_mean_weights(
#                 n1_weights, n2_weights, dtw_aln_1, dtw_aln_2
#             )
#             self.final_proteins.append(
#                 Protein(name_int, mean_vectors.shape[0], mean_vectors)
#             )
#             self.final_consensus_weights.append(mean_weights)
#
#         if verbose:
#             with typer.progressbar(
#                     range(0, self.tree.shape[0] - 1, 2), label="Aligning"
#             ) as progress:
#                 for x in progress:
#                     node_1, node_2, node_int = (
#                         self.tree[x, 0],
#                         self.tree[x + 1, 0],
#                         self.tree[x, 1],
#                     )
#                     assert self.tree[x + 1, 1] == node_int
#                     make_intermediate_node(node_1, node_2, node_int)
#         else:
#             for x in range(0, self.tree.shape[0] - 1, 2):
#                 node_1, node_2, node_int = (
#                     self.tree[x, 0],
#                     self.tree[x + 1, 0],
#                     self.tree[x, 1],
#                 )
#                 assert self.tree[x + 1, 1] == node_int
#                 make_intermediate_node(node_1, node_2, node_int)
#
#         node_1, node_2 = self.tree[-1, 0], self.tree[-1, 1]
#         make_intermediate_node(node_1, node_2, "final")
#         alignment = {
#             **msa_alignments[self.final_proteins[node_1].name],
#             **msa_alignments[self.final_proteins[node_2].name],
#         }
#         self.alignment = alignment
#         if return_sequence:
#             return self.make_sequence_alignment()
#         else:
#             return alignment
#
#     def make_sequence_alignment(self, alignment=None):
#         sequence_alignment = {}
#         if alignment is None:
#             alignment = self.alignment
#         for s in self.proteins:
#             sequence_alignment[s.name] = "".join(
#                 self.sequences[s.name][i] if i != -1 else "-" for i in alignment[s.name]
#             )
#         return sequence_alignment
#
#     def write_alignment(self):
#         """
#         Writes alignment to a fasta file
#         """
#         sequence_alignment = self.make_sequence_alignment()
#         with open(self.output_files.fasta_file, "w") as f:
#             for key in sequence_alignment:
#                 f.write(f">{key}\n{sequence_alignment[key]}\n")
#
#     def get_profile_alignment(
#             self, msa_class_new, gap_open_penalty: float, gap_extend_penalty: float,
#     ):
#         # Assumes self has already been aligned
#         msa_class_profile = deepcopy(self)
#         profile_sequence = np.arange(
#             len(msa_class_profile.alignment[msa_class_profile.proteins[0].name])
#         )
#         aligned_structures = msa_class_profile.get_aligned_structures()
#         profile_coords = np.nanmean(
#             np.array(
#                 [
#                     az.hdi(aligned_structures[:, i], skipna=True)
#                     for i in range(aligned_structures.shape[1])
#                 ]
#             ),
#             axis=-1,
#         )
#         profile_structure = Structure(
#             "caretta_profile", aligned_structures.shape[1], profile_coords
#         )
#         alignment = msa_class_profile.make_sequence_alignment()
#         for new_structure in msa_class_new.proteins:
#             msa_class = ProteinMultiple(
#                 [profile_structure, new_structure],
#                 {
#                     "caretta_profile": profile_sequence,
#                     new_structure.name: msa_class_new.sequences[new_structure.name],
#                 },
#                 msa_class_profile.superposition_parameters,
#                 msa_class_profile.superposition_function,
#             )
#             alignment_indices = msa_class.align(
#                 gap_open_penalty, gap_extend_penalty, False
#             )
#             alignment = {}
#             alignment[new_structure.name] = "".join(
#                 msa_class_new.sequences[new_structure.name][i] if i != -1 else "-"
#                 for i in alignment_indices[new_structure.name]
#             )
#             for n in msa_class_profile.sequences:
#                 if n != new_structure.name:
#                     alignment[n] = "".join(
#                         msa_class_profile.sequences[n][
#                             msa_class_profile.alignment[n][i]
#                         ]
#                         if i != -1 and msa_class_profile.alignment[n][i] != -1
#                         else "-"
#                         for i in alignment_indices["caretta_profile"]
#                     )
#             msa_class_profile.proteins.append(new_structure)
#             msa_class_profile.sequences[new_structure.name] = msa_class_new.sequences[
#                 new_structure.name
#             ]
#             msa_class_profile.alignment = alignment_to_numpy(alignment)
#             aligned_structures = msa_class_profile.get_aligned_structures()
#             profile_coords = np.nanmean(
#                 np.array(
#                     [
#                         az.hdi(aligned_structures[:, i], skipna=True)
#                         for i in range(aligned_structures.shape[1])
#                     ]
#                 ),
#                 axis=-1,
#             )
#             profile_structure = Structure(
#                 "caretta_profile", aligned_structures.shape[1], profile_coords
#             )
#         return alignment
#
#     def write_files(
#             self,
#             write_fasta,
#             write_pdb,
#             write_features,
#             write_class,
#             write_matrix,
#             only_dssp=True,
#             num_threads=4,
#             verbose: bool = False,
#     ):
#         if verbose and any(
#                 (write_fasta, write_pdb, write_pdb, write_class, write_matrix)
#         ):
#             typer.echo("Writing files...")
#         if write_fasta:
#             self.write_alignment()
#             if verbose:
#                 typer.echo(
#                     f"FASTA file: {typer.style(str(self.output_files.fasta_file), fg=typer.colors.GREEN)}",
#                 )
#         if write_pdb:
#             if not self.output_files.pdb_folder.exists():
#                 self.output_files.pdb_folder.mkdir()
#             write_superposed_pdbs(self.output_files.cleaned_pdb_folder,
#                                   self.alignment,
#                                   self.structures[self.reference_structure_index].name,
#                                   self.output_files.pdb_folder, verbose=verbose)
#             if verbose:
#                 typer.echo(
#                     f"Superposed PDB files: {typer.style(str(self.output_files.pdb_folder), fg=typer.colors.GREEN)}"
#                 )
#         if write_features:
#             if not self.output_files.tmp_folder.exists():
#                 self.output_files.tmp_folder.mkdir()
#             names, features = get_aligned_features(
#                 self.alignment, self.output_files.cleaned_pdb_folder, str(self.output_files.tmp_folder),
#                 only_dssp=only_dssp, num_threads=num_threads
#             )
#             with open(self.output_files.feature_file, "wb") as f:
#                 pickle.dump((names, features), f)
#             if verbose:
#                 typer.echo(
#                     f"Aligned features: {typer.style(str(self.output_files.feature_file), fg=typer.colors.GREEN)}"
#                 )
#         if write_class:
#             with open(self.output_files.class_file, "wb") as f:
#                 pickle.dump(self, f)
#             if verbose:
#                 typer.echo(
#                     f"Class file: {typer.style(str(self.output_files.class_file), fg=typer.colors.GREEN)}"
#                 )
#         if write_matrix:
#             helper.write_distance_matrix(
#                 [s.name for s in self.structures],
#                 self.pairwise_distance_matrix,
#                 self.output_files.matrix_file,
#             )
#             if verbose:
#                 typer.echo(
#                     f"Distance matrix file: {typer.style(str(self.output_files.matrix_file), fg=typer.colors.GREEN)}"
#                 )


def write_superposed_pdbs(
        cleaned_pdb_folder, alignment, reference_name, output_pdb_folder, verbose: bool = False
):
    """
    Superposes PDBs according to alignment and writes transformed PDBs to files
    (View with Pymol)

    Parameters
    ----------
    cleaned_pdb_folder : Path
    output_pdb_folder
    alignment
    verbose
    """
    output_pdb_folder = Path(output_pdb_folder)
    if not output_pdb_folder.exists():
        output_pdb_folder.mkdir()
    core_indices = np.array(
        [
            i
            for i in range(len(alignment[reference_name]))
            if -1 not in [alignment[n][i] for n in alignment]
        ]
    )
    if verbose:
        typer.echo(
            f"{len(core_indices)} core positions in alignment of length {len(alignment[reference_name])}"
        )
    if len(core_indices) < len(alignment[reference_name]) // 2:
        if verbose:
            typer.echo(
                typer.style(
                    "Core indices are < half of alignment length, superposing using reference structures "
                    "instead",
                    fg=typer.colors.RED,
                )
            )
            typer.echo(
                typer.style(
                    "Please inspect the distance matrix to split divergent protein groups",
                    fg=typer.colors.RED,
                )
            )
        write_superposed_pdbs_references(cleaned_pdb_folder, alignment, output_pdb_folder, verbose=verbose)
    else:
        write_superposed_pdbs_core(cleaned_pdb_folder, alignment, reference_name, output_pdb_folder)


def write_superposed_pdbs_core(cleaned_pdb_folder, alignment, reference_name, output_pdb_folder):
    """
    Superposes PDBs according to core indices in alignment and writes transformed PDBs to files
    (View with Pymol)

    Parameters
    ----------
    cleaned_pdb_folder
    alignment
    reference_name
    output_pdb_folder
    """
    core_indices = np.array(
        [
            i
            for i in range(len(alignment[reference_name]))
            if -1 not in [alignment[n][i] for n in alignment]
        ]
    )
    reference_pdb = pd.parsePDB(
        str(
            cleaned_pdb_folder / f"{reference_name}.pdb"
        )
    )
    aln_ref = alignment[reference_name]
    ref_coords_core = (
        reference_pdb[helper.get_alpha_indices(reference_pdb)]
        .getCoords()
        .astype(np.float64)[np.array([aln_ref[c] for c in core_indices])]
    )
    ref_centroid = helper.nb_mean_axis_0(ref_coords_core)
    ref_coords_core -= ref_centroid
    for name in alignment:
        pdb = pd.parsePDB(
            str(cleaned_pdb_folder / f"{name}.pdb")
        )
        aln_name = alignment[name]
        common_coords_2 = (
            pdb[helper.get_alpha_indices(pdb)]
            .getCoords()
            .astype(np.float64)[np.array([aln_name[c] for c in core_indices])]
        )
        (
            rotation_matrix,
            translation_matrix,
        ) = superposition_functions.paired_svd_superpose(
            ref_coords_core, common_coords_2
        )
        transformation = pd.Transformation(rotation_matrix.T, translation_matrix)
        pdb = pd.applyTransformation(transformation, pdb)
        pd.writePDB(str(output_pdb_folder / f"{name}.pdb"), pdb)


def write_superposed_pdbs_reference(cleaned_pdb_folder, alignment, reference_name, output_pdb_folder):
    """
    Superposes PDBs according to reference structure and writes transformed PDBs to files
    (View with Pymol)

    Parameters
    ----------
    cleaned_pdb_folder
    alignment
    reference_name
    output_pdb_folder
    """
    reference_pdb = pd.parsePDB(
        str(
            cleaned_pdb_folder
            / f"{reference_name}.pdb"
        )
    )
    aln_ref = alignment[reference_name]
    reference_coords = (
        reference_pdb[helper.get_alpha_indices(reference_pdb)]
        .getCoords()
        .astype(np.float64)
    )
    pd.writePDB(str(output_pdb_folder / f"{reference_name}.pdb"), reference_pdb)
    for name in alignment:
        if name == reference_name:
            continue
        pdb = pd.parsePDB(
            str(cleaned_pdb_folder / f"{name}.pdb")
        )
        aln_name = alignment[name]
        common_coords_1, common_coords_2 = get_common_vectors(
            reference_coords,
            pdb[helper.get_alpha_indices(pdb)].getCoords().astype(np.float64),
            aln_ref,
            aln_name,
        )
        (
            rotation_matrix,
            translation_matrix,
        ) = superposition_functions.paired_svd_superpose(
            common_coords_1, common_coords_2
        )
        transformation = pd.Transformation(rotation_matrix.T, translation_matrix)
        pdb = pd.applyTransformation(transformation, pdb)
        pd.writePDB(str(output_pdb_folder / f"{name}.pdb"), pdb)


def get_reference_structures(alignment, coverage_cutoff=20, gap=-1):
    names = list(alignment.keys())
    alignment_array = np.array([alignment[name] for name in names])
    distance_matrix, matrix_aligning = make_coverage_gap_distance_matrix(alignment_array)
    coverage_cutoff = np.array([min(coverage_cutoff, sum(1 for x in alignment[name] if x != gap)) for name in names])

    reference_structures = {}
    first_reference_structure = np.argmin(np.median(distance_matrix, axis=0))
    not_covered = np.where(matrix_aligning[:, first_reference_structure] < coverage_cutoff[:])[0]
    covered = list(np.where(matrix_aligning[:, first_reference_structure] >= coverage_cutoff[:])[0])
    reference_structures[first_reference_structure] = [names[c] for c in covered]
    problematic = []
    while len(not_covered) > 0:
        if len(not_covered) > 1:
            reference_structure = covered[
                np.argmin(np.median(distance_matrix[not_covered, :][:, covered], axis=0))]
        else:
            reference_structure = covered[np.argmin(distance_matrix[not_covered, :][:, covered])]
        covered_i = not_covered[
            np.where(matrix_aligning[not_covered, reference_structure] >= coverage_cutoff[not_covered])[0]]
        if len(covered_i) == 0:
            problematic += list(not_covered)
            break
        not_covered = not_covered[
            np.where(matrix_aligning[not_covered, reference_structure] < coverage_cutoff[not_covered])[0]]
        reference_structures[reference_structure] = [names[c] for c in covered_i]
        covered += list(covered_i)
    no_aligning = []
    for i in problematic:
        found = False
        for j in covered:
            if matrix_aligning[i, j] >= coverage_cutoff[i]:
                reference_structures[j].append(names[i])
                found = True
                break
        if not found:
            no_aligning.append(names[i])
    return names[first_reference_structure], {names[k]: v for k, v in reference_structures.items()}, no_aligning


def write_superposed_pdbs_references(cleaned_pdb_folder, alignment, output_pdb_folder,
                                     coverage_cutoff=50, verbose=False):
    """
    Superposes PDBs according to a set of reference structures covering the full alignment
    (View with Pymol)

    Parameters
    ----------
    cleaned_pdb_folder
    alignment
    output_pdb_folder
    coverage_cutoff
        minimum number of aligning residues to consider a pair of structures for superposition
    verbose
    """
    first_reference_structure, reference_structures, no_aligning = get_reference_structures(alignment, coverage_cutoff)
    if verbose:
        typer.echo(f"Reference structures: {reference_structures}")
    reference_pdb = pd.parsePDB(
        str(
            cleaned_pdb_folder
            / f"{first_reference_structure}.pdb"
        )
    )
    pd.writePDB(str(output_pdb_folder / f"{first_reference_structure}.pdb"), reference_pdb)
    for reference_name in reference_structures:
        reference_pdb = pd.parsePDB(
            str(output_pdb_folder / f"{reference_name}.pdb")
        )
        aln_ref = alignment[reference_name]
        reference_coords = reference_pdb[helper.get_alpha_indices(reference_pdb)].getCoords().astype(np.float64)
        for name in reference_structures[reference_name]:
            structure = pd.parsePDB(
                str(cleaned_pdb_folder / f"{name}.pdb")
            )
            pdb_coords = structure[helper.get_alpha_indices(structure)].getCoords().astype(np.float64)
            aln_name = alignment[name]
            common_coords_1, common_coords_2 = get_common_vectors(
                reference_coords,
                pdb_coords,
                aln_ref,
                aln_name,
            )
            try:
                (
                    rotation_matrix,
                    translation_matrix,
                ) = superposition_functions.paired_svd_superpose(
                    common_coords_1, common_coords_2
                )
            except LinAlgError as e:
                typer.echo(f"Could not superpose {name} to {reference_name}")
                no_aligning.append(name)
                continue
            transformation = pd.Transformation(rotation_matrix.T, translation_matrix)
            structure = pd.applyTransformation(transformation, structure)
            pd.writePDB(str(output_pdb_folder / f"{name}.pdb"), structure)
    if len(no_aligning):
        with open(output_pdb_folder / "missing.txt", "w") as f:
            f.write("\n".join(no_aligning))
        if verbose:
            typer.echo(
                f"Could not superpose {len(no_aligning)} structures as there are not enough aligning residues: {no_aligning}")


def get_aligned_features(
        alignment, cleaned_pdb_folder, dssp_dir, num_threads, only_dssp: bool = True
) -> typing.Tuple[typing.List[str], typing.Dict[str, ndarray]]:
    """
    Get list of protein names and corresponding dict of aligned features
    """
    names = list(alignment.keys())
    pdb_files = [
        cleaned_pdb_folder / f"{name}.pdb"
        for name in names
    ]
    features = feature_extraction.get_features_multiple(
        pdb_files,
        str(dssp_dir),
        num_threads=num_threads,
        only_dssp=only_dssp,
        force_overwrite=True,
    )
    feature_names = list(features[0].keys())
    aligned_features = {}
    alignment_length = len(alignment[names[0]])
    for feature_name in feature_names:
        if feature_name == "secondary":
            continue
        aligned_features[feature_name] = np.zeros(
            (len(names), alignment_length)
        )
        aligned_features[feature_name][:] = np.nan
        for p in range(len(names)):
            farray = features[p][feature_name]
            if "gnm" in feature_name or "anm" in feature_name:
                farray = farray / np.nansum(farray ** 2) ** 0.5
            indices = [
                i
                for i in range(alignment_length)
                if alignment[names[p]][i] != "-"
            ]
            aligned_features[feature_name][p, indices] = farray
    return names, aligned_features


def superpose(alignment, proteins, reference_name, gap=-1):
    core_indices = np.array(
        [
            i
            for i in range(len(alignment[reference_name]))
            if gap not in [alignment[n][i] for n in alignment]
        ]
    )
    if len(core_indices) < len(alignment[reference_name]) // 2:
        return superpose_reference(alignment, proteins, reference_name)
    else:
        return superpose_core(alignment, proteins, reference_name, core_indices)


def superpose_core(alignment, proteins, reference_name, core_indices: np.ndarray = None, gap=-1):
    """
    Superposes structures to first structure according to core positions in alignment using Kabsch superposition
    """
    if core_indices is None:
        core_indices = np.array(
            [
                i
                for i in range(len(alignment[reference_name]))
                if gap not in [alignment[n][i] for n in alignment]
            ]
        )
    aln_ref = alignment[reference_name]
    reference_structure_index = [i for i in range(len(proteins)) if proteins[i].name == reference_name][0]
    ref_coords = proteins[reference_structure_index].coordinates[
        np.array([aln_ref[c] for c in core_indices])
    ]
    ref_centroid = helper.nb_mean_axis_0(ref_coords)
    ref_coords -= ref_centroid
    for i in range(len(proteins)):
        if i == reference_structure_index:
            proteins[i].coordinates -= ref_centroid
        else:
            aln_c = alignment[proteins[i].name]
            common_coords_2 = proteins[i].coordinates[
                np.array([aln_c[c] for c in core_indices])
            ]
            (
                rotation_matrix,
                translation_matrix,
            ) = superposition_functions.paired_svd_superpose(
                ref_coords, common_coords_2
            )
            proteins[i].coordinates = superposition_functions.apply_rotran(
                proteins[i].coordinates, rotation_matrix, translation_matrix
            )
    return proteins


def superpose_reference(alignment, proteins, reference_name):
    """
    Superposes structures to first structure according to reference structure using Kabsch superposition
    """
    reference_structure_index = [i for i in range(len(proteins)) if proteins[i].name == reference_name][0]
    aln_ref = alignment[reference_name]
    for i in range(len(proteins)):
        aln_c = alignment[proteins[i].name]
        common_coords_1, common_coords_2 = get_common_vectors(
            proteins[reference_structure_index].coordinates,
            proteins[i].coordinates,
            aln_ref,
            aln_c,
        )
        assert common_coords_1.shape[0] > 0
        (
            rotation_matrix,
            translation_matrix,
        ) = superposition_functions.paired_svd_superpose(
            common_coords_1, common_coords_2
        )
        proteins[i].coordinates = superposition_functions.apply_rotran(
            proteins[i].coordinates, rotation_matrix, translation_matrix
        )
    return proteins


def make_rmsd_coverage_tm_matrix(
        alignment, proteins, reference_name, superpose_first: bool = True
):
    """
    Find RMSDs and coverages of the alignment of each pair of sequences

    Parameters
    ----------
    alignment
    proteins
    reference_name
    superpose_first
        if True then superposes all structures to first structure first

    Returns
    -------
    RMSD matrix, coverage matrix
    """
    num = len(alignment)
    pairwise_rmsd_matrix = np.zeros((num, num))
    pairwise_rmsd_matrix[:] = np.nan
    pairwise_coverage = np.zeros((num, num))
    pairwise_coverage[:] = np.nan
    pairwise_tm = np.zeros((num, num))
    pairwise_tm[:] = np.nan
    if superpose_first:
        superpose(alignment, proteins, reference_name)
    names = [p.name for p in proteins]
    for i in range(num - 1):
        for j in range(i + 1, num):
            name_1, name_2 = names[i], names[j]
            aln_1 = alignment[name_1]
            aln_2 = alignment[name_2]
            common_coords_1, common_coords_2 = get_common_vectors(
                proteins[i].coordinates,
                proteins[j].coordinates,
                aln_1,
                aln_2,
            )
            assert common_coords_1.shape[0] > 0
            if not superpose_first:
                rot, tran = superposition_functions.paired_svd_superpose(
                    common_coords_1, common_coords_2
                )
                common_coords_2 = superposition_functions.apply_rotran(
                    common_coords_2, rot, tran
                )
            pairwise_rmsd_matrix[i, j] = pairwise_rmsd_matrix[
                j, i
            ] = score_functions.get_rmsd(common_coords_1, common_coords_2)
            pairwise_coverage[i, j] = pairwise_coverage[
                j, i
            ] = common_coords_1.shape[0] / len(aln_1)
            pairwise_tm[i, j] = pairwise_tm[j, i] = tm_score(
                common_coords_1,
                common_coords_2,
                proteins[i].length,
                proteins[j].length,
            )
    return pairwise_rmsd_matrix, pairwise_coverage, pairwise_tm


def get_aligned_structures(alignment, proteins, reference_name):
    names = [p.name for p in proteins]
    if type(alignment[names[0]]) == str:
        alignment = alignment_to_numpy(alignment)
    superpose(alignment, proteins, reference_name)
    aligned_structures = np.zeros(
        (len(names), len(alignment[names[0]]), 3)
    )
    nan_3 = np.zeros(3)
    nan_3[:] = np.nan
    for i in range(len(proteins)):
        aligned_structures[i] = [
            proteins[i].coordinates[x] if x != -1 else nan_3
            for x in alignment[proteins[i].name]
        ]
    return aligned_structures


def trigger_numba_compilation():
    """
    Run this at the beginning of a Caretta run to compile Numba functions
    """
    parameters = {
        "size": 1,
        "gap_open_penalty": 0.0,
        "gap_extend_penalty": 0.0,
        "gamma": 0.03,
    }
    coords_1 = np.zeros((2, 3))
    coords_2 = np.zeros((2, 3))
    tm_score(coords_1, coords_2, 2, 2)
    weights_1 = np.zeros((coords_1.shape[0], 1))
    weights_2 = np.zeros((coords_2.shape[0], 1))
    get_pairwise_alignment(
        coords_1, coords_2, parameters["gamma"], 0, 0, weights_1, weights_2
    )
    superposition_functions.signal_svd_superpose_function(
        coords_1, coords_2, parameters
    )
    distance_matrix = np.random.random((5, 5))
    nj.neighbor_joining(distance_matrix)
    aln_1 = np.array([0, -1, 1])
    aln_2 = np.array([0, 1, -1])
    get_common_vectors(coords_1, coords_2, aln_1, aln_2)
    get_mean_coords(aln_1, coords_1, aln_2, coords_2)
    make_coverage_gap_distance_matrix(np.vstack((aln_1, aln_2)))
