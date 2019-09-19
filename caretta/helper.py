import subprocess
import typing
from pathlib import Path

import numba as nb
import numpy as np
import prody as pd


@nb.njit
def normalize(numbers):
    minv, maxv = np.min(numbers), np.max(numbers)
    return (numbers - minv) / (maxv - minv)


def aligned_string_to_array(aln: str) -> np.ndarray:
    """
    Aligned sequence to array of indices with gaps as -1

    Parameters
    ----------
    aln

    Returns
    -------
    indices
    """
    aln_array = np.zeros(len(aln), dtype=np.int64)
    i = 0
    for j in range(len(aln)):
        if aln[j] != '-':
            aln_array[j] = i
            i += 1
        else:
            aln_array[j] = -1
    return aln_array


@nb.njit
def get_common_positions(aln_array_1, aln_array_2, gap=-1):
    """
    Return positions where neither alignment has a gap

    Parameters
    ----------
    aln_array_1
    aln_array_2
    gap

    Returns
    -------
    common_positions_1, common_positions_2
    """
    pos_1 = np.array([aln_array_1[i] for i in range(len(aln_array_1)) if aln_array_1[i] != gap and aln_array_2[i] != gap], dtype=np.int64)
    pos_2 = np.array([aln_array_2[i] for i in range(len(aln_array_2)) if aln_array_1[i] != gap and aln_array_2[i] != gap], dtype=np.int64)
    return pos_1, pos_2


def get_aligned_data(aln_array: np.ndarray, data: np.ndarray, gap=-1):
    """
    Fills coordinates according to an alignment
    gaps (-1) in the sequence correspond to NaNs in the aligned coordinates

    Parameters
    ----------
    aln_array
        sequence (with gaps)
    data
        data to align
    gap
        character that represents gaps
    Returns
    -------
    aligned coordinates
    """
    pos = [i for i in range(len(aln_array)) if aln_array[i] != gap]
    assert len(pos) == data.shape[0], f"{len(pos)}, {data.shape[0]}"
    aln_coords = np.zeros((len(aln_array), data.shape[1]))
    aln_coords[:] = np.nan
    aln_coords[pos] = data
    return aln_coords


def get_file_parts(input_filename: typing.Union[str, Path]) -> tuple:
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


def get_alpha_indices(protein):
    """
    Get indices of alpha carbons of pd AtomGroup object
    """
    return [a.getIndex() for a in protein.iterAtoms() if a.getName() == 'CA']


def get_beta_indices(protein: pd.AtomGroup) -> list:
    """
    Get indices of beta carbons of pd AtomGroup object
    (alpha if Gly)
    """
    return [a.getIndex() for a in protein.iterAtoms() if
            (a.getResname() != 'GLY' and a.getName() == 'CB') or (a.getResname() == 'GLY' and a.getName() == 'CA')]


def clustal_msa_from_sequences(sequence_file, alignment_file, hmm_file=None, distance_matrix_file=None):
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
            subprocess.check_call(f"clustalo "
                                  f"--output-order=input-order "
                                  f"--log=log_clustal.out "
                                  f"-i {sequence_file} "
                                  f"--hmm-in={hmm_file} "
                                  f"-o {alignment_file} "
                                  f"--threads=10 --force -v", shell=True)
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
                shell=True)
    else:
        if distance_matrix_file is None:
            subprocess.check_call(
                f"clustalo "
                f"--output-order=input-order "
                f"--log=log_clustal.out "
                f"-i {sequence_file} "
                f"-o {alignment_file} "
                f"--threads=10 --force -v",
                shell=True)
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
                shell=True)


def get_sequences_from_fasta(fasta_file: typing.Union[str, Path], prune_headers: bool = True) -> dict:
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
            if line.startswith('>'):
                if current_key is None:
                    if "/" in line and prune_headers:
                        current_key = line.split(">")[1].split("/")[0].strip()
                    else:
                        current_key = line.split(">")[1].strip()
                else:
                    sequences[current_key] = ''.join(current_sequence)
                    current_sequence = []
                    if "/" in line and prune_headers:
                        current_key = line.split(">")[1].split("/")[0].strip()
                    else:
                        current_key = line.split(">")[1].strip()
            else:
                current_sequence.append(line.strip())
        sequences[current_key] = ''.join(current_sequence)
    return sequences
