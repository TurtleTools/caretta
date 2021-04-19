import multiprocessing
from pathlib import Path

import numpy as np
import prody as pd
import Bio.PDB
from Bio.PDB.ResidueDepth import get_surface, residue_depth, ca_depth, min_dist
from caretta import helper


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


def get_residue_depths(pdb_file):
    """
    Get residue depths

    Parameters
    ----------
    pdb_file

    Returns
    -------
    dict of depth _ ca/cb/mean
    """
    structure, residues, _, _, _ = read_pdb(pdb_file)
    surface = get_surface(structure)
    data = {
        "depth_mean": np.array(
            [residue_depth(residue, surface) for residue in residues]
        ),
        "depth_cb": np.array(
            [min_dist(get_beta_coordinates(residue), surface) for residue in residues]
        ),
        "depth_ca": np.array([ca_depth(residue, surface) for residue in residues]),
    }
    return data


def get_fluctuations(protein: pd.AtomGroup, n_modes: int = 50):
    """
    Get atom fluctuations using anisotropic and Gaussian network models with n_modes modes.

    Parameters
    ----------
    protein
    n_modes

    Returns
    -------
    dict of anm_ca, anm_cb, gnm_ca, gnm_cb
    """
    data = {}
    beta_indices = helper.get_beta_indices(protein)
    alpha_indices = helper.get_alpha_indices(protein)
    data["anm_cb"] = get_anm_fluctuations(protein[beta_indices], n_modes)
    data["gnm_cb"] = get_gnm_fluctuations(protein[beta_indices], n_modes)
    data["anm_ca"] = get_anm_fluctuations(protein[alpha_indices], n_modes)
    data["gnm_ca"] = get_gnm_fluctuations(protein[alpha_indices], n_modes)
    return data


def get_anm_fluctuations(protein: pd.AtomGroup, n_modes: int = 50):
    """
    Get atom fluctuations using an Anisotropic network model with n_modes modes.
    """
    protein_anm, _ = pd.calcANM(protein, n_modes=n_modes, selstr="all")
    return pd.calcSqFlucts(protein_anm)


def get_gnm_fluctuations(protein: pd.AtomGroup, n_modes: int = 50):
    """
    Get atom fluctuations using a Gaussian network model with n_modes modes.
    """
    protein_gnm, _ = pd.calcGNM(protein, n_modes=n_modes, selstr="all")
    return pd.calcSqFlucts(protein_gnm)


def get_features_multiple(
    pdb_files, dssp_dir, num_threads=20, only_dssp=True, force_overwrite=True, n_modes=50
):
    """
    Extract features for a list of pdb_files in parallel

    Parameters
    ----------
    pdb_files
    dssp_dir
        directory to store tmp dssp files
    num_threads
    only_dssp
        extract only dssp features
    force_overwrite
        force rerun DSSP

    Returns
    -------
    List of feature dicts (same order as pdb_files)
    """
    num_threads = min(len(pdb_files), num_threads)
    with multiprocessing.Pool(processes=num_threads) as pool:
        return pool.starmap(
            get_features,
            [
                (pdb_file, dssp_dir, only_dssp, force_overwrite, n_modes)
                for pdb_file in pdb_files
            ],
        )


def get_features(pdb_file: str, dssp_dir: str, only_dssp=True, force_overwrite=True, n_modes=50):
    """
    Extract features from a pdb_file

    Parameters
    ----------
    pdb_file
    dssp_dir
        directory to store tmp dssp files
    only_dssp
        extract only dssp features (use if not interested in features)
    force_overwrite
        force rerun DSSP

    Returns
    -------
    dict of features
    """
    pdb_file = str(pdb_file)
    name = Path(pdb_file).stem
    protein = pd.parsePDB(pdb_file).select("protein")
    pdb_file = str(Path(dssp_dir) / f"{name}.pdb")
    pd.writePDB(pdb_file, protein)
    protein = pd.parsePDB(pdb_file)
    dssp_file = Path(dssp_dir) / f"{name}.dssp"
    if force_overwrite or not dssp_file.exists():
        dssp_file = pd.execDSSP(str(pdb_file), outputname=name, outputdir=str(dssp_dir))
    protein = pd.parseDSSP(dssp=str(dssp_file), ag=protein, parseall=True)
    data = get_dssp_features(protein)
    if only_dssp:
        return data
    else:
        data = {**data, **get_fluctuations(protein, n_modes)}
        try:
            data = {**data, **get_residue_depths(pdb_file)}
        except RuntimeError as e:
            print(f"Failed to calculate residue depths: {e}")
        return data


def get_dssp_features(protein_dssp):
    """
    Extracts DSSP features (assumes DSSP is run already)

    Parameters
    ----------
    protein_dssp
        protein on which execDSSP has been called
    Returns
    -------
    dict of secondary,
    dssp_
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
    dssp_ignore = ["dssp_bp1", "dssp_bp2", "dssp_sheet_label", "dssp_resnum"]
    dssp_labels = [
        label
        for label in protein_dssp.getDataLabels()
        if label.startswith("dssp") and label not in dssp_ignore
    ]
    data = {}
    alpha_indices = helper.get_alpha_indices(protein_dssp)
    indices = [protein_dssp[x].getData("dssp_resnum") for x in alpha_indices]
    assert len(alpha_indices) == len(indices)
    for label in dssp_labels + ["secondary"]:
        label_to_index = {
            i - 1: protein_dssp[x].getData(label)
            for i, x in zip(indices, alpha_indices)
        }
        data[label] = np.array(
            [
                label_to_index[i] if i in label_to_index else 0
                for i in range(len(alpha_indices))
            ]
        )
    return data
