import itertools

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
from tqdm import tqdm
from geometricus import moment_invariants, Geometricus, ShapemerLearn

from caretta import (
    dynamic_time_warping as dtw,
    neighbor_joining as nj,
    score_functions,
    superposition_functions,
    feature_extraction,
    helper,
)

from abc import ABC, abstractmethod


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


@nb.njit
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


def get_mean_weights(
        weights_1: np.ndarray, weights_2: np.ndarray, aln_1: np.ndarray, aln_2: np.ndarray
) -> np.ndarray:
    mean_weights = np.zeros((aln_1.shape[0], 1))
    for i, (x, y) in enumerate(zip(aln_1, aln_2)):
        if not x == -1:
            mean_weights[i] += weights_1[x]
        if not y == -1:
            mean_weights[i] += weights_2[y]
    return mean_weights


@dataclass
class OutputFiles:
    output_folder: Path = Path("./caretta_results")
    fasta_file: Path = Path("./caretta_results/result.fasta")
    pdb_folder: Path = Path("./caretta_results/result_pdb/")
    cleaned_pdb_folder: Path = Path("./caretta_results/cleaned_pdb")
    matrix_folder: Path = Path("./caretta_results/result_matrix")
    feature_file: Path = Path("./caretta_results/result_features.pkl")
    class_file: Path = Path("./caretta_results/result_class.pkl")
    tmp_folder: Path = Path("./caretta_results/tmp/")

    @classmethod
    def from_folder(cls, output_folder):
        return cls(output_folder,
                   fasta_file=output_folder / "result.fasta",
                   pdb_folder=output_folder / "result_pdb/",
                   cleaned_pdb_folder=output_folder / "cleaned_pdb",
                   matrix_folder=output_folder / "result_matrix",
                   feature_file=output_folder / "result_features.pkl",
                   class_file=output_folder / "result_class.pkl",
                   tmp_folder=output_folder / "tmp/")


# abstract class for Protein implementing score_function and mean_function
class SequenceBase(ABC):
    name: str

    @abstractmethod
    def score_function(self, other: "SequenceBase", **kwargs) -> np.ndarray:
        pass

    # @abstractmethod
    def mean_function(self, other: "SequenceBase", aln_1: np.ndarray, aln_2: np.ndarray, name_int: str,
                      **kwargs) -> "SequenceBase":
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass


@nb.njit(parallel=True)
def make_count_matrix(residues_list, alphabet_size: int):
    out = np.zeros((len(residues_list), alphabet_size))
    for i in nb.prange(len(residues_list)):
        for j in range(len(residues_list[i])):
            out[i, residues_list[i][j]] += 1.
    return out


@nb.njit(parallel=True)
def braycurtis(counts_1, counts_2):
    out = np.zeros((counts_1.shape[0], counts_2.shape[0]))
    for i in nb.prange(counts_1.shape[0]):
        for j in range(counts_2.shape[0]):
            out[i, j] = np.abs(counts_1[i] - counts_2[j]).sum() / np.abs(counts_1[i] + counts_2[j]).sum()
    return out


@dataclass
class MultipleAlignment:
    sequences: typing.List[SequenceBase]
    tree: typing.Optional[np.ndarray] = None
    branch_lengths: typing.Optional[np.ndarray] = None
    alignment: typing.Optional[typing.Dict[str, np.ndarray]] = None
    final_sequences: typing.Optional[typing.List[SequenceBase]] = None
    final_consensus_weights: typing.Optional[typing.List[np.ndarray]] = None
    final_alignments: typing.Optional[typing.Dict[str, typing.Dict[str, np.ndarray]]] = None

    def make_pairwise_matrix(self, score_function_params=None):
        if score_function_params is None:
            score_function_params = {}
        pairwise_score_matrix = np.zeros((len(self.sequences), len(self.sequences)))
        for i in tqdm(range(len(self.sequences) - 1)):
            for j in range(i + 1, len(self.sequences)):
                pairwise_score_matrix[i, j] = pairwise_score_matrix[j, i] = dtw.smith_waterman_score(
                    np.arange(len(self.sequences[i])),
                    np.arange(len(self.sequences[j])),
                    self.sequences[i].score_function(
                        self.sequences[j],
                        **score_function_params))
        return pairwise_score_matrix

    def progressive_align(self, tree,
                          gap_open_penalty,
                          gap_extend_penalty,
                          consensus_weight,
                          gamma_weight,
                          score_function_params=None,
                          mean_function_params=None) -> typing.Dict[str, np.ndarray]:
        if mean_function_params is None:
            mean_function_params = {}
        if score_function_params is None:
            score_function_params = {}
        final_sequences = [s for s in self.sequences]
        final_alignments = {
            s.name: {s.name: np.arange(len(s))} for s in final_sequences
        }
        final_consensus_weights = [np.full(
            (len(s), 1),
            consensus_weight,
            dtype=np.float64,
        ) for s in final_sequences]

        def make_intermediate_node(n1, n2, n_int):
            name_1, name_2 = (
                final_sequences[n1].name,
                final_sequences[n2].name,
            )
            n1_weights, n2_weights = final_consensus_weights[n1], final_consensus_weights[n2]
            multiplier_n1 = len(final_alignments[name_2]) / (
                    2 * (len(final_alignments[name_1]) + len(final_alignments[name_2])))
            multiplier_n2 = len(final_alignments[name_1]) / (
                    2 * (len(final_alignments[name_1]) + len(final_alignments[name_2])))
            name_int = f"int-{n_int}"
            score_matrix = final_sequences[n1].score_function(
                final_sequences[n2], **score_function_params
            )
            score_matrix += score_functions.make_score_matrix(n1_weights * multiplier_n1,
                                                              n2_weights * multiplier_n2,
                                                              score_functions.get_gaussian_score,
                                                              gamma_weight)
            aln_1, aln_2, score = dtw.dtw_align(
                np.arange(score_matrix.shape[0]), np.arange(score_matrix.shape[1]),
                score_matrix, gap_open_penalty=gap_open_penalty, gap_extend_penalty=gap_extend_penalty
            )
            intermediate_sequence = final_sequences[n1].mean_function(final_sequences[n2], aln_1, aln_2, name_int,
                                                                      **mean_function_params)
            intermediate_weights = get_mean_weights(n1_weights, n2_weights, aln_1, aln_2)
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
            final_consensus_weights.append(intermediate_weights)

        for x in tqdm(range(0, tree.shape[0] - 1, 2), desc="Aligning"):
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
        self.final_consensus_weights = final_consensus_weights
        self.final_alignments = final_alignments
        self.final_sequences = final_sequences
        return alignment

    def multiple_align(self, pairwise_distance_matrix,
                       gap_open_penalty, gap_extend_penalty,
                       consensus_weight, gamma_weight,
                       score_function_params=None, mean_function_params=None) -> typing.Dict[str, np.ndarray]:
        if mean_function_params is None:
            mean_function_params = {}
        if score_function_params is None:
            score_function_params = {}
        if len(self.sequences) == 2:
            score_matrix = self.sequences[0].score_function(
                self.sequences[1], **score_function_params
            )
            aln_1, aln_2, score = dtw.dtw_align(
                np.arange(score_matrix.shape[0]), np.arange(score_matrix.shape[1]),
                score_matrix, gap_open_penalty=gap_open_penalty, gap_extend_penalty=gap_extend_penalty
            )
            self.alignment = {
                self.sequences[0].name: aln_1,
                self.sequences[1].name: aln_2,
            }
            return self.alignment

        self.tree, self.branch_lengths = nj.neighbor_joining(pairwise_distance_matrix)
        self.alignment = self.progressive_align(self.tree,
                                                gap_open_penalty,
                                                gap_extend_penalty,
                                                consensus_weight,
                                                gamma_weight,
                                                score_function_params,
                                                mean_function_params)
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


@dataclass
class Protein(SequenceBase):
    name: str
    tensors: np.ndarray
    coordinates: np.ndarray = None
    """Coordinates of the protein"""
    sequence: str = ""
    """Amino acid sequence of the protein"""

    def score_function(self, other: "Protein", flexible=False,
                       gamma_tensor=0.03, gamma_coords=0.03, verbose=True) -> np.ndarray:
        if flexible:
            return score_functions.make_score_matrix(self.tensors, other.tensors,
                                                     score_functions.get_gaussian_score,
                                                     gamma_tensor)
        else:
            score_matrix = score_functions.make_score_matrix(self.tensors,
                                                             other.tensors,
                                                             score_functions.get_gaussian_score,
                                                             gamma=gamma_tensor)
            aln_1, aln_2, score = dtw.smith_waterman(np.arange(score_matrix.shape[0]),
                                                     np.arange(score_matrix.shape[1]),
                                                     score_matrix,
                                                     gap=0.)
            pos_1, pos_2 = helper.get_common_positions(aln_1, aln_2)
            if len(pos_1) <= 3:
                if verbose:
                    typer.echo(
                        f"Too few aligning positions for {self.name} and {other.name}, "
                        f"continuing without superposition")
                coords_1, coords_2 = np.array(self.coordinates), np.array(other.coordinates)
            else:
                coords_1, coords_2, _ = superposition_functions.paired_svd_superpose_with_subset(
                    self.coordinates, other.coordinates, self.coordinates[pos_1], other.coordinates[pos_2]
                )
            return score_functions.make_score_matrix(coords_1, coords_2,
                                                     score_functions.get_gaussian_score,
                                                     gamma=gamma_coords)

    def mean_function(self, other: "Protein", aln_1: np.ndarray, aln_2: np.ndarray, name_int: str,
                      flexible=False, verbose=True) -> "Protein":
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
        if len(pos_1) <= 3:
            if verbose:
                typer.echo(
                    f"Too few aligning positions for {self.name} and {other.name}, continuing without superposition")
            coords_1, coords_2 = np.array(self.coordinates), np.array(other.coordinates)
        else:
            coords_1, coords_2, _ = superposition_functions.paired_svd_superpose_with_subset(
                self.coordinates, other.coordinates, self.coordinates[pos_1], other.coordinates[pos_2]
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

    def __str__(self):
        return self.sequence


@nb.njit(parallel=True)
def make_count_matrix(residues_list, alphabet_size: int):
    out = np.zeros((len(residues_list), alphabet_size))
    for i in nb.prange(len(residues_list)):
        for j in range(len(residues_list[i])):
            out[i, residues_list[i][j]] += 1
    return out


def align_from_structure_files(
        input_files: typing.Union[typing.List[str], Path, str],
        gap_open_penalty: float = 1.0,
        gap_extend_penalty: float = 0.01,
        consensus_weight: bool = True,
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
) -> typing.Tuple[MultipleAlignment, OutputFiles]:
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
            use weighting to reduce gaps in well-aligned positions (default True)
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
            True => extracts and writes aligned features as a dictionary of numpy arrays into a pickle file
            (default True)
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
        output_files = OutputFiles.from_folder(output_folder=Path(output_folder))
    if not output_files.output_folder.exists():
        output_files.output_folder.mkdir()

    if not output_files.cleaned_pdb_folder.exists():
        output_files.cleaned_pdb_folder.mkdir()
    pdb_files = helper.parse_protein_files_and_clean(input_files, output_files.cleaned_pdb_folder)
    if verbose:
        typer.echo(f"Found {len(pdb_files)} structure files")
    model = ShapemerLearn.load()
    protein_moments, errors = moment_invariants.get_invariants_for_structures(pdb_files,
                                                                              n_threads=num_threads,
                                                                              verbose=verbose)
    if verbose:
        typer.echo(f"Found {len(protein_moments)} structures with valid invariants")

    proteins = [Protein(m.name, m.get_tensor_model(model).astype(np.float64),
                        m.calpha_coordinates, m.sequence) for m in
                protein_moments]
    msa_class = MultipleAlignment(proteins)
    score_function_params = dict(flexible=False,
                                 gamma_tensor=7.,
                                 gamma_coords=0.03)
    mean_function_params = dict(flexible=False)

    pairwise_distance_matrix = np.array([[0, 1], [1, 0]])
    if len(msa_class.sequences) > 2:
        if full:
            pairwise_distance_matrix = msa_class.make_pairwise_matrix(
                score_function_params=score_function_params
            )
            pairwise_distance_matrix = pairwise_distance_matrix.max() - pairwise_distance_matrix
        else:
            shapemer_keys = list(map(tuple, itertools.product([0, 1], repeat=model.output_dimension)))
            shapemers = Geometricus.from_invariants(protein_moments, model=model)
            proteins_to_shapemer_indices, _ = shapemers.map_protein_to_shapemer_indices(shapemer_keys=shapemer_keys)
            count_matrix = make_count_matrix([proteins_to_shapemer_indices[m.name] for m in protein_moments],
                                             len(shapemer_keys))
            pairwise_distance_matrix = braycurtis(count_matrix, count_matrix)
    if write_matrix:
        if verbose:
            typer.echo("Writing guide tree distance matrix...")
        Path(output_files.matrix_folder).mkdir(exist_ok=True)
        helper.write_distance_matrix(
            [s.name for s in msa_class.sequences],
            pairwise_distance_matrix,
            output_files.matrix_folder / "distance_matrix_guide_tree.txt"
        )
    msa_class.pairwise_distance_matrix = pairwise_distance_matrix
    alignment = msa_class.multiple_align(
        pairwise_distance_matrix,
        gap_open_penalty=gap_open_penalty,
        gap_extend_penalty=gap_extend_penalty,
        consensus_weight=float(consensus_weight),
        gamma_weight=1.,
        score_function_params=score_function_params,
        mean_function_params=mean_function_params

    )
    if verbose and any(
            (write_fasta, write_pdb, write_pdb, write_features, write_class, write_matrix)
    ):
        typer.echo("Writing files...")

    if write_fasta:
        msa_class.write_alignment(output_files.fasta_file)
        if verbose:
            typer.echo(
                f"FASTA file: {typer.style(str(output_files.fasta_file), fg=typer.colors.GREEN)}",
            )
    if write_pdb:
        Path(output_files.pdb_folder).mkdir(exist_ok=True)
        write_superposed_pdbs(output_files.cleaned_pdb_folder,
                              alignment,
                              output_files.pdb_folder, verbose=verbose)
        if verbose:
            typer.echo(
                f"Superposed PDB files: {typer.style(str(output_files.pdb_folder), fg=typer.colors.GREEN)}"
            )
    if write_features:
        Path(output_files.tmp_folder).mkdir(exist_ok=True)
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
            pickle.dump(msa_class, f)
        if verbose:
            typer.echo(
                f"Class file: {typer.style(str(output_files.class_file), fg=typer.colors.GREEN)}"
            )
    if write_matrix:
        Path(output_files.matrix_folder).mkdir(exist_ok=True)
        rmsd, coverage, tm = make_rmsd_coverage_tm_matrix(alignment, msa_class.sequences,
                                                          superpose_first=False)

        helper.write_distance_matrix(
            [s.name for s in msa_class.sequences],
            rmsd,
            output_files.matrix_folder / "rmsd.txt"
        )
        helper.write_distance_matrix(
            [s.name for s in msa_class.sequences],
            coverage,
            output_files.matrix_folder / "coverage.txt"
        )
        helper.write_distance_matrix(
            [s.name for s in msa_class.sequences],
            tm,
            output_files.matrix_folder / "tm.txt"
        )
        if verbose:
            typer.echo(
                f"Distance matrix files in: {typer.style(str(output_files.matrix_folder), fg=typer.colors.GREEN)}"
            )
    return msa_class, output_files


def write_superposed_pdbs(
        cleaned_pdb_folder, alignment, output_pdb_folder, verbose: bool = False
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
    reference_name = sorted(alignment.keys(), key=lambda x: sum(1 for a in alignment[x] if a != -1), reverse=True)[0]
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
            cleaned_pdb_folder / f"{reference_name}"
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
            str(cleaned_pdb_folder / f"{name}")
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
        pd.writePDB(str(output_pdb_folder / f"{name}"), pdb)


def write_superposed_pdbs_reference(cleaned_pdb_folder, alignment, reference_name, output_pdb_folder, verbose=True):
    """
    Superposes PDBs according to reference structure and writes transformed PDBs to files
    (View with Pymol)

    Parameters
    ----------
    cleaned_pdb_folder
    alignment
    reference_name
    output_pdb_folder
    verbose
    """
    reference_pdb = pd.parsePDB(
        str(
            cleaned_pdb_folder
            / f"{reference_name}"
        )
    )
    aln_ref = alignment[reference_name]
    reference_coords = (
        reference_pdb[helper.get_alpha_indices(reference_pdb)]
        .getCoords()
        .astype(np.float64)
    )
    pd.writePDB(str(output_pdb_folder / f"{reference_name}"), reference_pdb)
    for name in alignment:
        if name == reference_name:
            continue
        pdb = pd.parsePDB(
            str(cleaned_pdb_folder / f"{name}")
        )
        aln_name = alignment[name]
        pos_1, pos_2 = helper.get_common_positions(aln_ref, aln_name)
        if len(pos_1) <= 3:
            if verbose:
                typer.echo(f"Not enough common positions to superpose {reference_name} and {name}")
            continue
        (
            rotation_matrix,
            translation_matrix,
        ) = superposition_functions.paired_svd_superpose(
            reference_coords[pos_1], pdb[helper.get_alpha_indices(pdb)].getCoords().astype(np.float64)[pos_2]
        )
        transformation = pd.Transformation(rotation_matrix.T, translation_matrix)
        pdb = pd.applyTransformation(transformation, pdb)
        pd.writePDB(str(output_pdb_folder / f"{name}"), pdb)


def get_reference_structures(alignment, minimum_coverage=50, gap=-1):
    """
    Returns a list of reference structures such that each structure assigned to a particular reference
    is at least minimum_coverage% covered by the alignment
    """
    names = list(alignment.keys())
    alignment_array = np.array([alignment[name] for name in names])
    distance_matrix, matrix_aligning = make_coverage_gap_distance_matrix(alignment_array)
    minimum_coverage = np.array(
        [minimum_coverage * sum(1 for x in alignment[name] if x != gap) / 100 for name in names])

    reference_structures = {}
    first_reference_structure = np.argmin(np.median(distance_matrix, axis=0))
    not_covered = np.where(matrix_aligning[:, first_reference_structure] < minimum_coverage[:])[0]
    covered = list(np.where(matrix_aligning[:, first_reference_structure] >= minimum_coverage[:])[0])
    reference_structures[first_reference_structure] = [names[c] for c in covered]
    problematic = []
    while len(not_covered) > 0:
        if len(not_covered) > 1:
            reference_structure = covered[
                np.argmin(np.median(distance_matrix[not_covered, :][:, covered], axis=0))]
        else:
            reference_structure = covered[np.argmin(distance_matrix[not_covered, :][:, covered])]
        covered_i = not_covered[
            np.where(matrix_aligning[not_covered, reference_structure] >= minimum_coverage[not_covered])[0]]
        if len(covered_i) == 0:
            problematic += list(not_covered)
            break
        not_covered = not_covered[
            np.where(matrix_aligning[not_covered, reference_structure] < minimum_coverage[not_covered])[0]]
        reference_structures[reference_structure] = [names[c] for c in covered_i]
        covered += list(covered_i)
    no_aligning = []
    for i in problematic:
        found = False
        for j in covered:
            if matrix_aligning[i, j] >= minimum_coverage[i]:
                reference_structures[j].append(names[i])
                found = True
                break
        if not found:
            no_aligning.append(names[i])
    return names[first_reference_structure], {names[k]: v for k, v in reference_structures.items()}, no_aligning


def write_superposed_pdbs_references(cleaned_pdb_folder, alignment, output_pdb_folder,
                                     minimum_coverage=50, verbose=False):
    """
    Superposes PDBs according to a set of reference structures such that each structure assigned to a particular
    reference is at least minimum_coverage% covered by the alignment
    (View with Pymol)

    Parameters
    ----------
    cleaned_pdb_folder
    alignment
    output_pdb_folder
    minimum_coverage
        minimum % of aligning residues to consider a pair of structures for superposition
    verbose
    """
    first_reference_structure, reference_structures, no_aligning = get_reference_structures(alignment, minimum_coverage)
    if verbose:
        typer.echo(f"Reference structure(s): " + " ".join(reference_structures.keys()))
        if len(reference_structures) > 1:
            with open(output_pdb_folder / "references.txt", "w") as f:
                for r in reference_structures:
                    f.write(r)
                    for k in reference_structures[r]:
                        f.write(f"\t{k}")
    reference_pdb = pd.parsePDB(
        str(
            cleaned_pdb_folder
            / f"{first_reference_structure}"
        )
    )
    pd.writePDB(str(output_pdb_folder / f"{first_reference_structure}"), reference_pdb)
    for reference_name in reference_structures:
        reference_pdb = pd.parsePDB(
            str(output_pdb_folder / f"{reference_name}")
        )
        aln_ref = alignment[reference_name]
        reference_coords = reference_pdb[helper.get_alpha_indices(reference_pdb)].getCoords().astype(np.float64)
        for name in reference_structures[reference_name]:
            structure = pd.parsePDB(
                str(cleaned_pdb_folder / f"{name}")
            )
            pdb_coords = structure[helper.get_alpha_indices(structure)].getCoords().astype(np.float64)
            aln_name = alignment[name]
            pos_1, pos_2 = helper.get_common_positions(aln_ref, aln_name)
            assert len(pos_1) > 3
            try:
                (
                    rotation_matrix,
                    translation_matrix,
                ) = superposition_functions.paired_svd_superpose(
                    reference_coords[pos_1], pdb_coords[pos_2]
                )
            except LinAlgError:
                if verbose:
                    typer.echo(f"Could not superpose {name} to {reference_name}")
                no_aligning.append(name)
                continue
            transformation = pd.Transformation(rotation_matrix.T, translation_matrix)
            structure = pd.applyTransformation(transformation, structure)
            pd.writePDB(str(output_pdb_folder / f"{name}"), structure)
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
        cleaned_pdb_folder / f"{name}"
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


def superpose(alignment, proteins, gap=-1) -> typing.List[Protein]:
    reference_name = sorted(alignment.keys(), key=lambda x: sum(1 for a in alignment[x] if a != -1), reverse=True)[
        0]
    core_indices = np.array(
        [
            i
            for i in range(len(alignment[reference_name]))
            if gap not in [alignment[n][i] for n in alignment]
        ]
    )
    print("Core indices", len(core_indices))
    if len(core_indices) < len(alignment[reference_name]) // 2:
        return superpose_reference(alignment, proteins, reference_name)
    else:
        return superpose_core(alignment, proteins, reference_name, core_indices)


def superpose_core(alignment, proteins, reference_name, core_indices: np.ndarray = None, gap=-1):
    """
    Superposes structures to reference structure according to core positions in alignment using Kabsch superposition
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
    for i in tqdm(range(len(proteins))):
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
    Superposes structures to reference structure according to reference structure using Kabsch superposition
    """
    reference_structure_index = [i for i in range(len(proteins)) if proteins[i].name == reference_name][0]
    aln_ref = alignment[reference_name]
    for i in range(len(proteins)):
        aln_c = alignment[proteins[i].name]
        pos_1, pos_2 = helper.get_common_positions(aln_ref, aln_c)
        assert len(pos_1) > 3
        (
            rotation_matrix,
            translation_matrix,
        ) = superposition_functions.paired_svd_superpose(
            proteins[reference_structure_index].coordinates[pos_1], proteins[i].coordinates[pos_2]
        )
        proteins[i].coordinates = superposition_functions.apply_rotran(
            proteins[i].coordinates, rotation_matrix, translation_matrix
        )
    return proteins


def superpose_references(alignment, proteins, minimum_coverage=50):
    """
    Superposes proteins according to a set of reference structures such that each structure assigned to a particular
    reference is at least minimum_coverage% covered by the alignment
    """
    names = [p.name for p in proteins]
    proteins = {p.name: p for p in proteins}
    first_reference_structure, reference_structures, no_aligning = get_reference_structures(alignment, minimum_coverage)
    for reference_name in reference_structures:
        aln_ref = alignment[reference_name]
        for name in reference_structures[reference_name]:
            aln_name = alignment[name]
            pos_1, pos_2 = helper.get_common_positions(aln_ref, aln_name)
            assert len(pos_1) > 3
            (
                rotation_matrix,
                translation_matrix,
            ) = superposition_functions.paired_svd_superpose(
                proteins[reference_name].coordinates[pos_1], proteins[name].coordinates[pos_2]
            )
            proteins[name].coordinates = superposition_functions.apply_rotran(
                proteins[name].coordinates, rotation_matrix, translation_matrix
            )
    return [proteins[name] for name in names]


def make_rmsd_coverage_tm_matrix(
        alignment, proteins, superpose_first: bool = True
):
    """
    Find RMSDs and coverages of the alignment of each pair of sequences

    Parameters
    ----------
    alignment
    proteins
    superpose_first
        if True then superposes all structures to reference structure first
        otherwise, superposition of each pair in the matrix

    Returns
    -------
    RMSD matrix, coverage matrix, TM-score matrix
    """
    num = len(alignment)
    pairwise_rmsd_matrix = np.zeros((num, num))
    pairwise_rmsd_matrix[:] = 0.
    pairwise_coverage = np.zeros((num, num))
    pairwise_coverage[:] = 1.
    pairwise_tm = np.zeros((num, num))
    pairwise_tm[:] = 1.
    if superpose_first:
        proteins = superpose(alignment, proteins)
    names = [p.name for p in proteins]
    for i in tqdm(range(num - 1)):
        for j in range(i + 1, num):
            name_1, name_2 = names[i], names[j]
            aln_1 = alignment[name_1]
            aln_2 = alignment[name_2]
            pos_1, pos_2 = helper.get_common_positions(aln_1, aln_2)
            assert len(pos_1) >= 3
            common_coords_1, common_coords_2 = proteins[i].coordinates[pos_1], proteins[j].coordinates[pos_2]
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
                len(proteins[i]),
                len(proteins[j]),
            )
    return pairwise_rmsd_matrix, pairwise_coverage, pairwise_tm


def trigger_numba_compilation():
    """
    Run this at the beginning of a Caretta run to compile Numba functions
    """
    coords_1 = np.zeros((2, 3))
    coords_2 = np.zeros((2, 3))
    tm_score(coords_1, coords_2, 2, 2)
    score_functions.get_rmsd(coords_1, coords_2)
    score_functions.get_gaussian_score(coords_1, coords_2)
    score_functions.make_score_matrix(coords_1, coords_2, score_functions.get_gaussian_score, gamma=0.03)
    distance_matrix = np.random.random((5, 5))
    nj.neighbor_joining(distance_matrix)
    aln_1 = np.array([0, -1, 1])
    aln_2 = np.array([0, 1, -1])
    helper.get_common_positions(aln_1, aln_2)
    get_mean_weights(coords_1[:, :1], coords_2[:, :1], aln_1, aln_2)
    make_coverage_gap_distance_matrix(np.vstack((aln_1, aln_2)))
    count_matrix = make_count_matrix([np.random.randint(0, 5, 10) for _ in range(10)], 5)
    braycurtis(count_matrix, count_matrix)
