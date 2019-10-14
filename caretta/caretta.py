import numpy as np
import prody as pd

from caretta import helper, feature_extraction, msa_numba


def get_structures(pdb_files, dssp_dir, num_threads=20, extract_all_features=True, force_overwrite=False):
    """
    Extract features and get coordinates for a list of PDB files

    Parameters
    ----------
    pdb_files
    dssp_dir
        directory to store tmp dssp files
    num_threads
    extract_all_features
        set to False if not interested in features (faster)
    force_overwrite
        forces DSSP to recalculate

    Returns
    -------
    list of Structure objects
    """
    pdbs = [pd.parsePDB(filename) for filename in pdb_files]
    alpha_indices = [helper.get_alpha_indices(pdb) for pdb in pdbs]
    sequences = [pdbs[i][alpha_indices[i]].getSequence() for i in range(len(pdbs))]
    coordinates = [pdbs[i][alpha_indices[i]].getCoords().astype(np.float64) for i in range(len(pdbs))]
    features = feature_extraction.get_features_multiple(pdb_files, str(dssp_dir), num_threads=num_threads, only_dssp=not extract_all_features,
                                                        force_overwrite=force_overwrite)
    structures = []
    for i in range(len(pdbs)):
        pdb_name = helper.get_file_parts(pdb_files[i])[1]
        structures.append(msa_numba.Structure(pdb_name,
                                              sequences[i],
                                              helper.secondary_to_array(features[i]["secondary"]),
                                              features[i],
                                              coordinates[i]))
    return structures


def align(structures, gap_open_penalty=1, gap_extend_penalty=0.01, consensus_weight=1.):
    """
    Aligns a list of Structure objects

    Parameters
    ----------
    structures
        output of get_structures
    gap_open_penalty
    gap_extend_penalty
    consensus_weight

    Returns
    -------
    msa_class (with intermediate nodes), alignment
    """
    msa_class = msa_numba.StructureMultiple(structures)
    caretta_alignment = msa_class.align(gamma=0.03, gap_open_sec=1, gap_extend_sec=0.1, gap_open_penalty=gap_open_penalty,
                                        gap_extend_penalty=gap_extend_penalty)
    return msa_class, caretta_alignment
