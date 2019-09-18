import multiprocessing
from pathlib import Path

import numba as nb
import numpy as np
import prody as pd

from caretta import helper


def get_anm_fluctuations(protein: pd.AtomGroup, n_modes: int = 50):
    """
    Get atom fluctuations using an Anisotropic network model with n_modes modes.
    """
    protein_anm, _ = pd.calcANM(protein, n_modes=n_modes, selstr='all')
    return pd.calcSqFlucts(protein_anm)


def get_gnm_fluctuations(protein: pd.AtomGroup, n_modes: int = 50):
    """
    Get atom fluctuations using a Gaussian network model with n_modes modes.
    """
    protein_gnm, _ = pd.calcGNM(protein, n_modes=n_modes, selstr='all')
    return pd.calcSqFlucts(protein_gnm)


def get_dssp_features_multiple(pdb_files, dssp_dir, num_threads=20):
    num_threads = min(len(pdb_files), num_threads)
    with multiprocessing.Pool(processes=num_threads) as pool:
        return pool.starmap(get_dssp_features, [(pdb_file, dssp_dir) for pdb_file in pdb_files])


def get_dssp_features(pdb_file: str, dssp_dir: str):
    """
    Gets dssp features

    Parameters
    ----------
    pdb_file
    dssp_dir

    Returns
    -------
    dict of dssp_
    NH_O_1_index, NH_O_1_energy
        hydrogen bonds; e.g. -3,-1.4 means: if this residue is residue i then N-H of I is h-bonded to C=O of I-3 with an
        electrostatic H-bond energy of -1.4 kcal/mol. There are two columns for each type of H-bond, to allow for bifurcated H-bonds.
    NH_O_2_index, NH_O_2_energy
    O_NH_1_index, O_NH_1_energy
    O_NH_2_index, O_NH_2_energy
    acc
        number of water molecules in contact with this residue *10. or residue water exposed surface in Angstrom^2.
    alpha
        virtual torsion angle (dihedral angle) defined by the four Cα atoms of residues I-1,I,I+1,I+2. Used to define chirality.
    kappa
        virtual bond angle (bend angle) defined by the three Cα atoms of residues I-2,I,I+2. Used to define bend (structure code ‘S’).
    phi
        IUPAC peptide backbone torsion angles.
    psi
        IUPAC peptide backbone torsion angles.
    tco
        cosine of angle between C=O of residue I and C=O of residue I-1. For α-helices, TCO is near +1, for β-sheets TCO is near -1.

    Ignores:
    dssp_bp1, dssp_bp2, and dssp_sheet_label: residue number of first and second bridge partner followed by one letter sheet label
    """
    pdb_file = str(pdb_file)
    _, name, _ = helper.get_file_parts(pdb_file)
    protein = pd.parsePDB(pdb_file)
    pdb_file = str(Path(dssp_dir) / f"{name}.pdb")
    pd.writePDB(pdb_file, protein)
    dssp_file = pd.execDSSP(pdb_file, outputname=name, outputdir=str(dssp_dir))
    protein = pd.parseDSSP(dssp=dssp_file, ag=protein, parseall=True)
    dssp_ignore = ["dssp_bp1", "dssp_bp2", "dssp_sheet_label", "dssp_resnum"]
    dssp_labels = [label for label in protein.getDataLabels() if label.startswith("dssp") and label not in dssp_ignore]
    data = {}
    beta_indices = helper.get_beta_indices(protein)
    indices = [protein[x].getData("dssp_resnum") for x in beta_indices]
    for label in dssp_labels:
        label_to_index = {i - 1: protein[x].getData(label) for i, x in zip(indices, beta_indices)}
        data[f"{label}"] = np.array([label_to_index[i] if i in label_to_index else 0 for i in range(len(beta_indices))])
    data["secondary"] = protein.getData("secondary")[beta_indices]
    return data


@nb.njit
def get_distances(coordinates: np.ndarray, num_neighbors=3):
    distances = np.zeros((coordinates.shape[0], num_neighbors), dtype=np.float64)
    for i in range(coordinates.shape[0]):
        for j in range(1, num_neighbors+1):
            distances[i, j] = np.sum(np.abs(coordinates[i] - coordinates[i-j]), axis=-1)
    return distances
