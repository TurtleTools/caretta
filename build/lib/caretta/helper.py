import typing
from typing import List, Union
from pathlib import Path
import Bio.PDB
import numba as nb
import numpy as np
import prody as pd

from geometricus.protein_utility import get_structure_files, parse_structure_file


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
        name = Path(input_file).stem
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


def parse_protein_files_and_clean(
        input_value: str, output_folder: Union[str, Path] = "./cleaned_pdb",
) -> List[Union[str, Path]]:
    if not Path(output_folder).exists():
        Path(output_folder).mkdir()
    protein_files = get_structure_files(input_value)
    output_pdb_files = []
    for protein_file in protein_files:
        protein = parse_structure_file(str(protein_file)).select("protein")
        chains = protein.getChids()
        if len(chains) and len(chains[0].strip()):
            protein = protein.select(f"chain {chains[0]}")
        output_pdb_file = str(Path(output_folder) / f"{Path(protein_file).stem}.pdb")
        pd.writePDB(output_pdb_file, protein)
        protein = pd.parsePDB(output_pdb_file)
        while len(protein.getCoordsets()) > 1:
            protein.delCoordset(1)
        pd.writePDB(output_pdb_file, protein)
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
