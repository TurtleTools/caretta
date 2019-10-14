import pickle
from pathlib import Path

import fire
import numpy as np
import prody as pd

from caretta import helper, msa_numba
from caretta.feature_extraction import get_features_multiple


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
    only_dssp = (not extract_all_features)
    features = get_features_multiple(pdb_files, str(dssp_dir), num_threads=num_threads, only_dssp=only_dssp, force_overwrite=force_overwrite)
    structures = []
    for i in range(len(pdbs)):
        pdb_name = helper.get_file_parts(pdb_files[i])[1]
        structures.append(msa_numba.Structure(pdb_name,
                                              pdb_files[i],
                                              sequences[i],
                                              helper.secondary_to_array(features[i]["secondary"]),
                                              features[i],
                                              coordinates[i]))
    return structures


def align(input_pdb,
          dssp_dir="caretta_tmp", num_threads=20, extract_all_features=True,
          gap_open_penalty=1., gap_extend_penalty=0.01, consensus_weight=1.,
          write_fasta=True, output_fasta_filename=None,
          write_pdb=True, output_pdb_folder=None,
          write_features=True, output_feature_filename=None,
          write_class=True, output_class_filename=None,
          force=False):
    if not Path(dssp_dir).exists():
        Path(dssp_dir).mkdir()
    input_pdb = Path(input_pdb)
    if input_pdb.is_dir():
        pdb_files = list(Path(input_pdb).glob("*.pdb"))
    elif input_pdb.is_file():
        with open(input_pdb) as f:
            pdb_files = f.read().strip().split('\n')
    else:
        pdb_files = list(input_pdb)
    if not Path(pdb_files[0]).is_file():
        pdb_files = [pd.fetchPDB(pdb_name) for pdb_name in pdb_files]
    print(f"Found {len(pdb_files)} PDB files")
    structures = get_structures(pdb_files, dssp_dir, num_threads=num_threads, extract_all_features=extract_all_features, force_overwrite=force)
    msa_class = msa_numba.StructureMultiple(structures)
    msa_class.align(gamma=0.03, gap_open_sec=1, gap_extend_sec=0.1, gap_open_penalty=gap_open_penalty,
                    gap_extend_penalty=gap_extend_penalty)
    if write_fasta:
        if output_fasta_filename is None:
            output_fasta_filename = "result.fasta"
        msa_class.write_alignment(output_fasta_filename)
    if write_pdb:
        if output_pdb_folder is None:
            output_pdb_folder = Path("result_pdb")
            if not output_pdb_folder.exists():
                output_pdb_folder.mkdir()
        msa_class.write_superposed_pdbs(output_pdb_folder)
    if write_features:
        if output_feature_filename is None:
            output_feature_filename = "result_features.pkl"
        with open(output_feature_filename, "wb") as f:
            pickle.dump(msa_class.get_aligned_features(), f)
    if write_class:
        if output_class_filename is None:
            output_class_filename = "result_class.pkl"
        with open(output_class_filename, "wb") as f:
            pickle.dump(msa_class, f)
    return msa_class


def align_cli(input_pdb,
              dssp_dir="caretta_tmp", num_threads=20, extract_all_features=True,
              gap_open_penalty=1., gap_extend_penalty=0.01, consensus_weight=1.,
              write_fasta=True, output_fasta_filename=None,
              write_pdb=True, output_pdb_folder=None,
              write_features=True, output_feature_filename=None,
              write_class=True, output_class_filename=None,
              force=False):
    """
    Caretta aligns protein structures and returns a sequence alignment, a set of aligned feature matrices, superposed PDB files, and
    a class with intermediate structures made during progressive alignment.
    Parameters
    ----------
    input_pdb
        Can be \n
        A folder with input protein files \n
        A file which lists PDB filenames on each line \n
        A file which lists PDB IDs on each line \n
    dssp_dir
        Folder to store temp DSSP files (default caretta_tmp)
    num_threads
        Number of threads to use for feature extraction
    extract_all_features
        True => obtains all features (default True) \n
        False => only DSSP features (faster)
    gap_open_penalty
        default 1
    gap_extend_penalty
        default 0.01
    consensus_weight
        default 1
    write_fasta
        True => writes alignment as fasta file (default True)
    output_fasta_filename
        Fasta file of alignment (default result.fasta)
    write_pdb
        True => writes all protein PDB files superposed by alignment (default True)
    output_pdb_folder
        Folder to write superposed PDB files (default result_pdb)
    write_features
        True => writes aligned features a s a dictionary of numpy arrays into a pickle file (default True)
    output_feature_filename
        Pickle file to write aligned features (default result_features.pkl)
    write_class
        True => writes StructureMultiple class with intermediate structures and tree to pickle file (default True)
    output_class_filename
        Pickle file to write StructureMultiple class (default result_class.pkl)
    force
        Forces DSSP to rerun (default False)

    Returns
    -------
    StructureMultiple class
    """
    align(input_pdb, dssp_dir, num_threads, extract_all_features, gap_open_penalty, gap_extend_penalty, consensus_weight, write_fasta,
          output_fasta_filename, write_pdb, output_pdb_folder, write_features, output_feature_filename, write_class, output_class_filename, force)


if __name__ == '__main__':
    fire.Fire(align_cli)
