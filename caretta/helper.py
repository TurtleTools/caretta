import subprocess
import typing
from typing import List, Union, Tuple
from pathlib import Path, PosixPath
import Bio.PDB
import numba as nb
import numpy as np
import prody as pd


def secondary_to_array(secondary):
    return np.array(secondary, dtype="S1").view(np.int8)


@nb.njit
def get_common_positions(aln_array_1, aln_array_2):
    """
    Return positions where neither alignment has a gap (-1)

    Parameters
    ----------
    aln_array_1
    aln_array_2

    Returns
    -------
    common_positions_1, common_positions_2
    """
    pos_1 = np.array(
        [
            aln_array_1[i]
            for i in range(len(aln_array_1))
            if aln_array_1[i] != -1 and aln_array_2[i] != -1
        ],
        dtype=np.int64,
    )
    pos_2 = np.array(
        [
            aln_array_2[i]
            for i in range(len(aln_array_2))
            if aln_array_1[i] != -1 and aln_array_2[i] != -1
        ],
        dtype=np.int64,
    )
    return pos_1, pos_2


@nb.njit
def nb_mean_axis_0(array: np.ndarray) -> np.ndarray:
    """
    Same as np.mean(array, axis=0) but njitted
    """
    mean_array = np.zeros(array.shape[1])
    for i in range(array.shape[1]):
        mean_array[i] = np.mean(array[:, i])
    return mean_array


@nb.njit
def nb_std_axis_0(array: np.ndarray) -> np.ndarray:
    """
    Same as np.std(array, axis=0) but njitted
    """
    std_array = np.zeros(array.shape[1])
    for i in range(array.shape[1]):
        std_array[i] = np.std(array[:, i])
    return std_array


@nb.njit
def normalize(numbers):
    minv, maxv = np.min(numbers), np.max(numbers)
    return (numbers - minv) / (maxv - minv)


def get_file_parts(input_filename: Union[str, Path]) -> Tuple[str, str, str]:
    """
    Gets directory path, name, and extension from a filename
    Parameters
    ----------
    input_filename

    Returns
    -------
    (path, name, extension)
    """
    input_filename = Path(input_filename)
    path = str(input_filename.parent)
    extension = input_filename.suffix
    name = input_filename.stem
    return path, name, extension


def get_alpha_indices(protein: pd.AtomGroup) -> List[int]:
    """
    Get indices of alpha carbons of pd AtomGroup object
    """
    return [a.getIndex() for a in protein.iterAtoms() if a.getName() == "CA"]


def get_beta_indices(protein: pd.AtomGroup) -> List[int]:
    """
    Get indices of beta carbons of pd AtomGroup object
    (If beta carbon doesn't exist, alpha carbon index is returned)
    """
    residue_splits = group_indices(protein.getResindices())
    i = 0
    indices = []
    for split in residue_splits:
        ca = None
        cb = None
        for _ in split:
            if protein[i].getName() == "CB":
                cb = protein[i].getIndex()
            if protein[i].getName() == "CA":
                ca = protein[i].getIndex()
            i += 1
        if cb is not None:
            indices.append(cb)
        else:
            assert ca is not None
            indices.append(ca)
    return indices


def group_indices(input_list: List[int]) -> List[List[int]]:
    """
    [1, 1, 1, 2, 2, 3, 3, 3, 4] -> [[0, 1, 2], [3, 4], [5, 6, 7], [8]]
    Parameters
    ----------
    input_list

    Returns
    -------
    list of lists
    """
    output_list = []
    current_list = []
    current_index = None
    for i in range(len(input_list)):
        if current_index is None:
            current_index = input_list[i]
        if input_list[i] == current_index:
            current_list.append(i)
        else:
            output_list.append(current_list)
            current_list = [i]
        current_index = input_list[i]
    output_list.append(current_list)
    return output_list


def clustal_msa_from_sequences(
    sequence_file, alignment_file, hmm_file=None, distance_matrix_file=None
):
    """
    Align sequences optionally using hmm_file as a guide

    Parameters
    ----------
    sequence_file
    alignment_file
    hmm_file
    distance_matrix_file
        Writes pairwise distance matrix into file if not None

    Returns
    -------

    """
    if hmm_file is not None:
        if distance_matrix_file is None:
            subprocess.check_call(
                f"clustalo "
                f"--output-order=input-order "
                f"--log=log_clustal.out "
                f"-i {sequence_file} "
                f"--hmm-in={hmm_file} "
                f"-o {alignment_file} "
                f"--threads=10 --force -v",
                shell=True,
            )
        else:
            subprocess.check_call(
                f"clustalo "
                f"--output-order=input-order "
                f"--log=log_clustal.out --full "
                f"--distmat-out={distance_matrix_file} "
                f"-i {sequence_file} "
                f"--hmm-in={hmm_file} "
                f"-o {alignment_file} "
                f"--threads=10 --force -v",
                shell=True,
            )
    else:
        if distance_matrix_file is None:
            subprocess.check_call(
                f"clustalo "
                f"--output-order=input-order "
                f"--log=log_clustal.out "
                f"-i {sequence_file} "
                f"-o {alignment_file} "
                f"--threads=10 --force -v",
                shell=True,
            )
        else:
            subprocess.check_call(
                f"clustalo "
                f"--output-order=input-order "
                f"--log=log_clustal.out "
                f"--full "
                f"--distmat-out={distance_matrix_file} "
                f"-i {sequence_file} "
                f"-o {alignment_file} "
                f"--threads=10 --force -v",
                shell=True,
            )


def get_sequences_from_fasta(
    fasta_file: Union[str, Path], prune_headers: bool = True
) -> dict:
    """
    Returns dict of accession to sequence from fasta file
    Parameters
    ----------
    fasta_file
    prune_headers
        only keeps accession upto first /

    Returns
    -------
    {accession:sequence}
    """
    sequences = {}
    with open(fasta_file) as f:
        current_sequence = []
        current_key = None
        for line in f:
            if not len(line.strip()):
                continue
            if line.startswith(">"):
                if current_key is None:
                    if "/" in line and prune_headers:
                        current_key = line.split(">")[1].split("/")[0].strip()
                    else:
                        current_key = line.split(">")[1].strip()
                else:
                    sequences[current_key] = "".join(current_sequence)
                    current_sequence = []
                    if "/" in line and prune_headers:
                        current_key = line.split(">")[1].split("/")[0].strip()
                    else:
                        current_key = line.split(">")[1].strip()
            else:
                current_sequence.append(line.strip())
        sequences[current_key] = "".join(current_sequence)
    return sequences


def read_pdb(input_file, name: str = None, chain: str = None) -> tuple:
    """
    returns protein information from PDB file
    Parameters
    ----------
    input_file
    name
        None => takes from filename
    chain
        only for that chain

    Returns
    -------
    structure Object, list of residue Objects, list of peptide Objects, sequence, sequence to residue index
    """
    if name is None:
        (_, name, _) = get_file_parts(input_file)
    input_file = str(input_file)
    structure = Bio.PDB.PDBParser().get_structure(name, input_file)
    if chain is not None:
        structure = structure[0][chain]
    residues = Bio.PDB.Selection.unfold_entities(structure, "R")
    peptides = Bio.PDB.PPBuilder().build_peptides(structure)
    sequence = "".join([str(peptide.get_sequence()) for peptide in peptides])
    residue_dict = dict(zip(residues, range(len(residues))))
    seq_to_res_index = [residue_dict[r] for peptide in peptides for r in peptide]
    return structure, residues, peptides, sequence, seq_to_res_index


def get_alpha_coordinates(residue) -> np.ndarray:
    """
    Returns alpha coordinates of BioPython residue object
    """
    return np.array(residue["CA"].get_coord())


def get_beta_coordinates(residue) -> np.ndarray:
    """
    Returns beta coordinates of BioPython residue object
    (alpha if Gly)
    """
    if residue.get_resname() == "GLY" or "CB" not in residue:
        return get_alpha_coordinates(residue)
    return np.array(residue["CB"].get_coord())


def parse_pdb_files(input_pdb):
    if type(input_pdb) == str or type(input_pdb) == PosixPath:
        input_pdb = Path(input_pdb)
        if input_pdb.is_dir():
            pdb_files = list(input_pdb.glob("*.pdb"))
        elif input_pdb.is_file():
            with open(input_pdb) as f:
                pdb_files = f.read().strip().split("\n")
        else:
            pdb_files = str(input_pdb).split("\n")
    else:
        pdb_files = list(input_pdb)
        if not Path(pdb_files[0]).is_file():
            pdb_files = [pd.fetchPDB(pdb_name) for pdb_name in pdb_files]
    return pdb_files


def parse_pdb_files_and_clean(
    input_pdb: str, output_pdb: Union[str, Path] = "./cleaned_pdb",
) -> List[Union[str, Path]]:
    if not Path(output_pdb).exists():
        Path(output_pdb).mkdir()
    pdb_files = parse_pdb_files(input_pdb)
    output_pdb_files = []
    for pdb_file in pdb_files:
        pdb = pd.parsePDB(pdb_file).select("protein")
        chains = pdb.getChids()
        if len(chains) and len(chains[0].strip()):
            pdb = pdb.select(f"chain {chains[0]}")
        output_pdb_file = str(Path(output_pdb) / f"{Path(pdb_file).stem}.pdb")
        pd.writePDB(output_pdb_file, pdb)
        output_pdb_files.append(output_pdb_file)
    return output_pdb_files


def write_distance_matrix(
    names: typing.List[str],
    distance_matrix: np.ndarray,
    filename: typing.Union[Path, str],
):
    """
    Writes distance matrix to file in clustal format

    Parameters
    ----------
    names
        protein names in order
    distance_matrix
    filename
    """
    with open(filename, "w") as f:
        f.write(f"{len(names)}\n")
        for i in range(len(names)):
            row = " ".join(f"{x:.4f}" for x in distance_matrix[i])
            f.write(f"{names[i]} {row}\n")


def read_distance_matrix(filename: typing.Union[Path, str]):
    """
    Read distance matrix file

    Parameters
    ----------
    filename

    Returns
    -------
    protein names in order, distance matrix
    """
    names = []
    num_proteins = 0
    with open(filename) as f:
        for i, line in enumerate(f):
            if i == 0:
                num_proteins = int(line.strip())
                continue
            names.append(line.split()[0].strip().split("/")[0].strip())
    assert len(names) == num_proteins
    distance_matrix = np.loadtxt(
        filename, skiprows=1, usecols=range(1, num_proteins + 1)
    )
    return names, distance_matrix
