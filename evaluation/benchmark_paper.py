import typing
from pathlib import Path

import numpy as np
import prody as pd

from caretta import helper
from caretta import multiple_structure_alignment as msa
from caretta import pairwise_structure_alignment as psa


def get_pdbs(pdb_dir) -> dict:
    """
    Get all the proteins in a directory as pd AtomGroup objects
    (Assumes all non-dot files in the directory are PDB files)

    Parameters
    ----------
    pdb_dir

    Returns
    -------
    dictionary {name: pd.AtomGroup}
    """
    pdb_dir = Path(pdb_dir)
    pdbs = {}
    for pdb_file in pdb_dir.glob("[!.]*"):
        pdbs[pdb_file.stem] = pd.parsePDB(pdb_file)
    return pdbs


def get_sequences(pdbs: typing.Dict[str, pd.AtomGroup]) -> typing.Dict[str, str]:
    """
    Get pdb sequences
    Parameters
    ----------
    pdbs
        dict of {name: pd.AtomGroup}

    Returns
    -------
    dict of {name: sequence}
    """
    return {n: pdbs[n][helper.get_alpha_indices(pdbs[n])].getSequence() for n in pdbs}


def get_sequence_alignment(sequences: typing.Dict[str, str], directory, name="ref") -> typing.Dict[str, str]:
    """
    Use clustal-omega to make a multiple sequence alignment

    Parameters
    ----------
    sequences
        dict of {name: sequence}
    directory
        dir to save sequence and alignment file
    name
        name to give sequence/alignment file
    Returns
    -------
    dict of {name: aln_sequence}
    """
    directory = Path(directory)
    sequence_file = directory / f"{name}.fasta"
    with open(sequence_file, "w") as f:
        for n in sequences:
            f.write(f">{n}\n{sequences[n]}\n")
    aln_sequence_file = directory / f"{name}_aln.fasta"
    helper.clustal_msa_from_sequences(sequence_file, aln_sequence_file)
    return helper.get_sequences_from_fasta(aln_sequence_file)


def get_structures(pdb_dir) -> typing.List[psa.Structure]:
    """
    Get list of Structure objects from a directory of PDB files
    """
    pdbs = get_pdbs(pdb_dir)
    names = list(pdbs.keys())
    sequences = get_sequences(pdbs)
    coordinates = [pdbs[n][helper.get_beta_indices(pdbs[n])].getCoords().astype(np.float64) for n in names]
    structures = msa.make_structures(names, [sequences[n] for n in names], coordinates)
    return structures


def get_msa(pdb_dir, sequence_dir, name="ref", gap_open_penalty: float = 0., gap_extend_penalty: float = 0.):
    """
    Example usage of above functions to make a structure-guided multiple sequence alignment from a directory of PDB files
    """
    structures = get_structures(pdb_dir)
    msa_class = msa.StructureMultiple(structures)
    aln_sequences = get_sequence_alignment({s.name: s.sequence for s in msa_class.structures}, sequence_dir, name)
    return msa_class.align(aln_sequences, gap_open_penalty, gap_extend_penalty)
