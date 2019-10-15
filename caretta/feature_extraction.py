import multiprocessing
import subprocess
from pathlib import Path

import numpy as np
import prody as pd
from Bio.PDB.ResidueDepth import get_surface, residue_depth, ca_depth, min_dist

from caretta import helper


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
    structure, residues, _, _, _ = helper.read_pdb(pdb_file)
    surface = get_surface(structure)
    data = {"depth_mean": np.array([residue_depth(residue, surface) for residue in residues]),
            "depth_cb": np.array([min_dist(helper.get_beta_coordinates(residue), surface) for residue in residues]),
            "depth_ca": np.array([ca_depth(residue, surface) for residue in residues])}
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
    protein_anm, _ = pd.calcANM(protein, n_modes=n_modes, selstr='all')
    return pd.calcSqFlucts(protein_anm)


def get_gnm_fluctuations(protein: pd.AtomGroup, n_modes: int = 50):
    """
    Get atom fluctuations using a Gaussian network model with n_modes modes.
    """
    protein_gnm, _ = pd.calcGNM(protein, n_modes=n_modes, selstr='all')
    return pd.calcSqFlucts(protein_gnm)


def get_features_multiple(pdb_files, dssp_dir, num_threads=20, only_dssp=True, force_overwrite=False):
    """
    Extract features for a list of pdb_files in parallel

    Parameters
    ----------
    pdb_files
    dssp_dir
        directory to store tmp dssp files
    num_threads
    only_dssp
        extract only dssp features (use if not interested in features)
    force_overwrite
        force rerun DSSP

    Returns
    -------
    List of feature dicts (same order as pdb_files)
    """
    num_threads = min(len(pdb_files), num_threads)
    with multiprocessing.Pool(processes=num_threads) as pool:
        return pool.starmap(get_features, [(pdb_file, dssp_dir, only_dssp, force_overwrite) for pdb_file in pdb_files])


def get_features(pdb_file: str, dssp_dir: str, only_dssp=True, force_overwrite=True):
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
    _, name, _ = helper.get_file_parts(pdb_file)
    protein = pd.parsePDB(pdb_file)
    if Path(pdb_file).suffix != ".pdb":
        pdb_file = str(Path(dssp_dir) / f"{name}.pdb")
        pd.writePDB(pdb_file, protein)
    dssp_file = Path(dssp_dir) / f"{name}.dssp"
    if force_overwrite or not dssp_file.exists():
        dssp_file = pd.execDSSP(str(pdb_file), outputname=name, outputdir=str(dssp_dir))
    protein = pd.parseDSSP(dssp=dssp_file, ag=protein, parseall=True)
    data = get_dssp_features(protein)
    if only_dssp:
        return data
    else:
        data = {**data, **get_fluctuations(protein)}
        data = {**data, **get_residue_depths(pdb_file)}
        # data = {**data, **get_electrostatics(protein, pdb_file, es_dir=dssp_dir)}
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
    dssp_labels = [label for label in protein_dssp.getDataLabels() if label.startswith("dssp") and label not in dssp_ignore]
    data = {}
    alpha_indices = helper.get_alpha_indices(protein_dssp)
    indices = [protein_dssp[x].getData("dssp_resnum") for x in alpha_indices]
    for label in dssp_labels:
        label_to_index = {i - 1: protein_dssp[x].getData(label) for i, x in zip(indices, alpha_indices)}
        data[f"{label}"] = np.array([label_to_index[i] if i in label_to_index else 0 for i in range(len(alpha_indices))])
    data["secondary"] = protein_dssp.getData("secondary")[alpha_indices]
    return data


def get_electrostatics(protein: pd.AtomGroup, pdb_file: str, es_dir: str, overwrite=False):
    """
    Gets born and coulomb electrostatics for given protein

    Parameters
    ----------
    protein
    pdb_file
    es_dir
    overwrite

    Returns
    -------
    dict of born/coulomb _ Energy/x-force/y-force/z-force _ ca/cb/mean/min/max
    """
    es_dir = str(es_dir)
    pdb_file = str(pdb_file)
    data_born = _get_electrostatics(protein, pdb_file, es_dir, es_type="born", overwrite=overwrite)
    data_coulomb = _get_electrostatics(protein, pdb_file, es_dir, es_type="coulomb", overwrite=overwrite)
    return {**data_born, **data_coulomb}


def _run_electrostatics(pdb_file, es_dir: str, es_type: str, overwrite=False) -> tuple:
    """
    Run apbs-born / apbs-coulomb on protein

    NOTE: born is 0-indexed and coulomb is 1-indexed

    Parameters
    ----------
    pdb_file
    es_dir
    es_type
    overwrite

    Returns
    -------
    (es_file, pqr_file, add)
    """
    assert es_type == "born" or es_type == "coulomb"
    _, name, _ = helper.get_file_parts(pdb_file)
    pqr_file = Path(es_dir) / f"{name}.pqr"
    if not pqr_file.exists():
        pdb2pqr(pdb_file, pqr_file)
    es_file = Path(es_dir) / f"{name}_{es_type}.txt"
    if es_type == "born":
        add = 0
        if overwrite or not es_file.exists():
            apbs_born(str(pqr_file), str(es_file))
    else:
        add = 1
        if overwrite or not es_file.exists():
            apbs_coulomb(str(pqr_file), str(es_file))
    return es_file, pqr_file, add


def _get_electrostatics(protein: pd.AtomGroup, pdb_file: str, es_dir: str, es_type: str = "born", overwrite=False):
    """
    Run apbs-born / apbs-coulomb on protein

    NOTE: born is 0-indexed and coulomb is 1-indexed

    Parameters
    ----------
    protein
    pdb_file
    es_dir
    es_type
    overwrite

    Returns
    -------
    dict of born/coulomb _ Energy/x-force/y-force/z-force _ ca/cb/mean/ #min/max#
    """
    if not Path(pdb_file).exists():
        pd.writePDB(pdb_file, protein)
    es_file, pqr_file, add = _run_electrostatics(pdb_file, es_dir, es_type, overwrite)
    pqr_protein = pd.parsePQR(str(pqr_file))
    residue_splits = helper.group_indices(pqr_protein.getResindices())
    values, value_types = parse_electrostatics_file(es_file)
    data = {}
    for value_type in value_types:
        data[f"{es_type}_{value_type}_ca"] = np.array(
            [values[index + add][value_type] if value_type in values[index + add] else 0 for index in
             helper.get_alpha_indices(pqr_protein)])
        data[f"{es_type}_{value_type}_cb"] = np.array(
            [values[index + add][value_type] if value_type in values[index + add] else 0 for index in
             helper.get_beta_indices(pqr_protein)])
        data[f"{es_type}_{value_type}_mean"] = np.array(
            [np.nanmean([values[x + add][value_type] for x in split if (x + add) in values and value_type in values[x + add]]) for split in
             residue_splits])
    return data


def pdb2pqr(pdb_file, pqr_file):
    """
    Convert pdb file to pqr file
    """
    exe = "/mnt/nexenta/durai001/programs/pdb2pqr-2.1.1/pdb2pqr"
    command = f"{exe} --ff=amber --apbs-input {pdb_file} {pqr_file}"
    subprocess.check_call(command, shell=True)


def apbs_born(pqr_file, output_file, epsilon: int = 80):
    """
    Runs born electrostatics
    """
    exe = "/mnt/nexenta/durai001/programs/APBS-1.5/share/apbs/tools/bin/born"
    command = f"{exe} -v -f {epsilon} {pqr_file} > {output_file}"
    subprocess.check_call(command, shell=True)


def apbs_coulomb(pqr_file, output_file):
    """
    Runs coulomb electrostatics
    """
    exe = "/mnt/nexenta/durai001/programs/APBS-1.5/share/apbs/tools/bin/coulomb"
    command = f"{exe} -e -f {pqr_file} > {output_file}"
    subprocess.check_call(command, shell=True)


def parse_electrostatics_file(filename) -> tuple:
    """
    Parse file returned by running apbs_born or apbs_coulomb

    Parameters
    ----------
    filename

    Returns
    -------
    (dict(atom_number: dict(value_type: value)), set(value_types))
    """
    values = {}
    value_types = set()
    with open(filename) as f:
        for i, line in enumerate(f):
            if i < 10:
                continue
            if "Atom" not in line:
                break
            line = line.lstrip().rstrip()
            atom_number = int(line.split(":")[0][5:])
            value = float(line.split("=")[1].split()[0])
            value_type = line.split(":")[1].split("=")[0].lstrip().rstrip()
            value_types.add(value_type)
            if atom_number not in values:
                values[atom_number] = {}
            values[atom_number][value_type] = value
    return values, value_types
