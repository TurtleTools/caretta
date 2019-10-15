import pickle
import typing
from dataclasses import dataclass, field
from pathlib import Path

import numba as nb
import numpy as np
import prody as pd

from caretta import feature_extraction
from caretta import neighbor_joining as nj
from caretta import psa_numba as psa
from caretta import rmsd_calculations, helper


@nb.njit
def get_common_coordinates(coords_1, coords_2, aln_1, aln_2, gap=-1):
    assert aln_1.shape == aln_2.shape
    pos_1, pos_2 = helper.get_common_positions(aln_1, aln_2, gap)
    return coords_1[pos_1], coords_2[pos_2]


@nb.njit(parallel=True)
def make_pairwise_dtw_score_matrix(coords_array, secondary_array, lengths_array, gamma,
                                   gap_open_penalty: float, gap_extend_penalty: float,
                                   gap_open_sec, gap_extend_sec):
    pairwise_matrix = np.zeros((coords_array.shape[0], coords_array.shape[0]))
    for i in nb.prange(pairwise_matrix.shape[0] - 1):
        for j in range(i + 1, pairwise_matrix.shape[1]):
            dtw_aln_1, dtw_aln_2, score = psa.get_pairwise_alignment(coords_array[i, :lengths_array[i]], coords_array[j, :lengths_array[j]],
                                                                     secondary_array[i, :lengths_array[i]], secondary_array[j, :lengths_array[j]],
                                                                     gamma,
                                                                     gap_open_sec=gap_open_sec,
                                                                     gap_extend_sec=gap_extend_sec,
                                                                     gap_open_penalty=gap_open_penalty,
                                                                     gap_extend_penalty=gap_extend_penalty)
            common_coords_1, common_coords_2 = get_common_coordinates(coords_array[i, :lengths_array[i]],
                                                                      coords_array[j, :lengths_array[j]],
                                                                      dtw_aln_1, dtw_aln_2)
            rotation_matrix, translation_matrix = rmsd_calculations.svd_superimpose(common_coords_1[:, :3], common_coords_2[:, :3])
            common_coords_2[:, :3] = rmsd_calculations.apply_rotran(common_coords_2[:, :3], rotation_matrix, translation_matrix)
            score = rmsd_calculations.get_caretta_score(common_coords_1, common_coords_2, gamma, True)
            pairwise_matrix[i, j] = -score
    pairwise_matrix += pairwise_matrix.T
    return pairwise_matrix


@nb.njit
def _get_alignment_data(coords_1, coords_2, secondary_1, secondary_2, gamma,
                        gap_open_sec, gap_extend_sec,
                        gap_open_penalty: float, gap_extend_penalty: float):
    dtw_aln_1, dtw_aln_2, _ = psa.get_pairwise_alignment(
        coords_1, coords_2,
        secondary_1, secondary_2,
        gamma,
        gap_open_sec=gap_open_sec,
        gap_extend_sec=gap_extend_sec,
        gap_open_penalty=gap_open_penalty,
        gap_extend_penalty=gap_extend_penalty)
    pos_1, pos_2 = helper.get_common_positions(dtw_aln_1, dtw_aln_2, -1)
    coords_1[:, :3], coords_2[:, :3], _ = rmsd_calculations.superpose_with_pos(coords_1[:, :3], coords_2[:, :3],
                                                                               coords_1[pos_1][:, :3], coords_2[pos_2][:, :3])
    aln_coords_1 = helper.get_aligned_data(dtw_aln_1, coords_1, -1)
    aln_coords_2 = helper.get_aligned_data(dtw_aln_2, coords_2, -1)
    aln_sec_1 = helper.get_aligned_string_data(dtw_aln_1, secondary_1, -1)
    aln_sec_2 = helper.get_aligned_string_data(dtw_aln_2, secondary_2, -1)
    return aln_coords_1, aln_coords_2, aln_sec_1, aln_sec_2, dtw_aln_1, dtw_aln_2


@nb.njit
def get_mean_coords_extra(aln_coords_1: np.ndarray, aln_coords_2: np.ndarray) -> np.ndarray:
    """
    Mean of two coordinate sets (of the same shape)

    Parameters
    ----------
    aln_coords_1
    aln_coords_2

    Returns
    -------
    mean_coords
    """
    mean_coords = np.zeros(aln_coords_1.shape)
    for i in range(aln_coords_1.shape[0]):
        mean_coords[i, :-1] = np.array([np.nanmean(np.array([aln_coords_1[i, x], aln_coords_2[i, x]])) for x in range(aln_coords_1.shape[1] - 1)])
        if not np.isnan(aln_coords_1[i, 0]):
            mean_coords[i, -1] += aln_coords_1[i, -1]
        if not np.isnan(aln_coords_2[i, 0]):
            mean_coords[i, -1] += aln_coords_2[i, -1]
    return mean_coords


@nb.njit
def get_mean_secondary(aln_sec_1: np.ndarray, aln_sec_2: np.ndarray, gap=0) -> np.ndarray:
    """
    Mean of two coordinate sets (of the same shape)

    Parameters
    ----------
    aln_sec_1
    aln_sec_2
    gap

    Returns
    -------
    mean_sec
    """
    mean_sec = np.zeros(aln_sec_1.shape, dtype=aln_sec_1.dtype)
    for i in range(aln_sec_1.shape[0]):
        if aln_sec_1[i] == aln_sec_2[i]:
            mean_sec[i] = aln_sec_1[i]
        else:
            if aln_sec_1[i] != gap:
                mean_sec[i] = aln_sec_1[i]
            elif aln_sec_2[i] != gap:
                mean_sec[i] = aln_sec_2[i]
    return mean_sec


@dataclass(eq=False)
class Structure:
    name: str
    pdb_file: typing.Union[str, Path, None]
    sequence: typing.Union[str, None]
    secondary: np.ndarray = field(repr=False)
    features: typing.Union[np.ndarray, None] = field(repr=False)
    coords: np.ndarray = field(repr=False)

    @classmethod
    def from_pdb_file(cls, pdb_file: typing.Union[str, Path], dssp_dir="caretta_tmp",
                      extract_all_features=True, force_overwrite=False):
        pdb_name = helper.get_file_parts(pdb_file)[1]
        pdb = pd.parsePDB(str(pdb_file))
        alpha_indices = helper.get_alpha_indices(pdb)
        sequence = pdb[alpha_indices].getSequence()
        coordinates = pdb[alpha_indices].getCoords().astype(np.float64)
        only_dssp = (not extract_all_features)
        features = feature_extraction.get_features(str(pdb_file), str(dssp_dir), only_dssp=only_dssp, force_overwrite=force_overwrite)
        return cls(pdb_name, pdb_file, sequence, helper.secondary_to_array(features["secondary"]), features, coordinates)


@dataclass
class OutputFiles:
    fasta_file: Path = Path("./result.fasta")
    pdb_folder: Path = Path("./result_pdb/")
    feature_file: Path = Path("./result_features.pkl")
    class_file: Path = Path("./result_class.pkl")


def parse_pdb_files(input_pdb: str) -> typing.List[typing.Union[str, Path]]:
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
    return pdb_files


@dataclass
class StructureMultiple:
    """
    Class for multiple structure alignment
    """
    structures: typing.Union[None, typing.List[Structure]] = None
    lengths_array: typing.Union[None, np.ndarray] = None
    max_length: int = 0
    coords_array: typing.Union[None, np.ndarray] = None
    secondary_array: typing.Union[None, np.ndarray] = None
    final_structures: typing.Union[None, typing.List[Structure]] = None
    tree: typing.Union[None, np.ndarray] = None
    branch_lengths: typing.Union[None, np.ndarray] = None
    alignment: typing.Union[None, dict] = None
    output_files: OutputFiles = OutputFiles()

    @classmethod
    def from_structures(cls, structures: typing.List[Structure], output_fasta_filename=Path("./result.fasta"),
                        output_pdb_folder=Path("./result_pdb/"),
                        output_feature_filename=Path("./result_features.pkl"),
                        output_class_filename=Path("./result_class.pkl")):
        cls.structures = structures
        cls.lengths_array = np.array([len(s.sequence) for s in cls.structures])
        cls.max_length = np.max(cls.lengths_array)
        cls.coords_array = np.zeros((len(cls.structures), cls.max_length, cls.structures[0].coords.shape[1]))
        cls.secondary_array = np.zeros((len(cls.structures), cls.max_length))
        for i in range(len(cls.structures)):
            cls.coords_array[i, :cls.lengths_array[i]] = cls.structures[i].coords
            cls.secondary_array[i, :cls.lengths_array[i]] = cls.structures[i].secondary
        cls.final_structures = []
        cls.tree = None
        cls.branch_lengths = None
        cls.alignment = None
        cls.output_files = OutputFiles(output_fasta_filename, output_pdb_folder, output_feature_filename, output_class_filename)
        return cls

    @classmethod
    def from_pdb_files(cls, input_pdb,
                       dssp_dir=Path("./caretta_tmp/"), num_threads=20, extract_all_features=True,
                       consensus_weight=1.,
                       output_fasta_filename=Path("./result.fasta"),
                       output_pdb_folder=Path("./result_pdb/"),
                       output_feature_filename=Path("./result_features.pkl"),
                       output_class_filename=Path("./result_class.pkl"),
                       force=False):
        pdb_files = parse_pdb_files(input_pdb)
        if not Path(dssp_dir).exists():
            Path(dssp_dir).mkdir()
        pdbs = [pd.parsePDB(filename) for filename in pdb_files]
        alpha_indices = [helper.get_alpha_indices(pdb) for pdb in pdbs]
        sequences = [pdbs[i][alpha_indices[i]].getSequence() for i in range(len(pdbs))]
        coordinates = [np.hstack((pdbs[i][alpha_indices[i]].getCoords().astype(np.float64), np.zeros((len(sequences[i]), 1)) + consensus_weight))
                       for i in range(len(pdbs))]
        only_dssp = (not extract_all_features)
        features = feature_extraction.get_features_multiple(pdb_files, str(dssp_dir),
                                                            num_threads=num_threads, only_dssp=only_dssp, force_overwrite=force)
        structures = []
        for i in range(len(pdbs)):
            pdb_name = helper.get_file_parts(pdb_files[i])[1]
            structures.append(Structure(pdb_name,
                                        pdb_files[i],
                                        sequences[i],
                                        helper.secondary_to_array(features[i]["secondary"]),
                                        features[i],
                                        coordinates[i]))
        return cls.from_structures(structures, output_fasta_filename, output_pdb_folder, output_feature_filename, output_class_filename)

    @classmethod
    def align_from_pdb_files(cls, input_pdb,
                             dssp_dir="caretta_tmp", num_threads=20, extract_all_features=True,
                             gap_open_penalty=1., gap_extend_penalty=0.01, consensus_weight=1.,
                             write_fasta=True, output_fasta_filename=None,
                             write_pdb=True, output_pdb_folder=None,
                             write_features=True, output_feature_filename=None,
                             write_class=True, output_class_filename=None,
                             overwrite_dssp=False):
        """
        Caretta aligns protein structures and returns a sequence alignment, a set of aligned feature matrices, superposed PDB files, and
        a class with intermediate structures made during progressive alignment.
        Parameters
        ----------
        input_pdb
            Can be \n
            A list of PDB files
            A list of PDB IDs
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
        overwrite_dssp
            Forces DSSP to rerun (default False)

        Returns
        -------
        StructureMultiple class
        """
        cls.from_pdb_files(input_pdb,
                           dssp_dir, num_threads, extract_all_features,
                           consensus_weight,
                           output_fasta_filename,
                           output_pdb_folder,
                           output_feature_filename,
                           output_class_filename,
                           overwrite_dssp)
        cls.align(cls, gap_open_penalty, gap_extend_penalty)
        cls.write_files(cls, write_fasta,
                        write_pdb,
                        write_features,
                        write_class)
        return cls

    def align(self, gap_open_penalty, gap_extend_penalty, pw_matrix=None, gamma=0.03, gap_open_sec=1., gap_extend_sec=0.1) -> dict:
        print("Aligning...")
        if len(self.structures) == 2:
            dtw_1, dtw_2, _ = psa.get_pairwise_alignment(self.coords_array[0, :self.lengths_array[0]],
                                                         self.coords_array[1, :self.lengths_array[1]],
                                                         self.secondary_array[0, :self.lengths_array[0]],
                                                         self.secondary_array[1, :self.lengths_array[1]],
                                                         gamma,
                                                         gap_open_sec=gap_open_sec,
                                                         gap_extend_sec=gap_extend_sec,
                                                         gap_open_penalty=gap_open_penalty,
                                                         gap_extend_penalty=gap_extend_penalty)
            self.alignment = {self.structures[0].name: "".join([self.structures[0].sequence[i] if i != -1 else '-' for i in dtw_1]),
                              self.structures[1].name: "".join([self.structures[1].sequence[i] if i != -1 else '-' for i in dtw_2])}
            return self.alignment

        if pw_matrix is None:
            pw_matrix = make_pairwise_dtw_score_matrix(self.coords_array,
                                                       self.secondary_array,
                                                       self.lengths_array,
                                                       gamma,
                                                       gap_open_sec, gap_extend_sec,
                                                       gap_open_penalty, gap_extend_penalty)

        print("Pairwise score matrix calculation done")

        tree, branch_lengths = nj.neighbor_joining(pw_matrix)
        self.tree = tree
        self.branch_lengths = branch_lengths
        self.final_structures = [s for s in self.structures]
        msa_alignments = {s.name: {s.name: s.sequence} for s in self.structures}

        print("Neighbor joining tree constructed")

        def make_intermediate_node(n1, n2, n_int):
            name_1, name_2 = self.final_structures[n1].name, self.final_structures[n2].name
            name_int = f"int-{n_int}"
            n1_coords = self.final_structures[n1].coords
            n1_coords[:, -1] *= len(msa_alignments[name_2])
            n1_coords[:, -1] /= 2.
            n2_coords = self.final_structures[n2].coords
            n2_coords[:, -1] *= len(msa_alignments[name_1])
            n2_coords[:, -1] /= 2.
            aln_coords_1, aln_coords_2, aln_sec_1, aln_sec_2, dtw_aln_1, dtw_aln_2 = _get_alignment_data(n1_coords,
                                                                                                         n2_coords,
                                                                                                         self.final_structures[
                                                                                                             n1].secondary,
                                                                                                         self.final_structures[
                                                                                                             n2].secondary,
                                                                                                         gamma,
                                                                                                         gap_open_sec=gap_open_sec,
                                                                                                         gap_extend_sec=gap_extend_sec,
                                                                                                         gap_open_penalty=gap_open_penalty,
                                                                                                         gap_extend_penalty=gap_extend_penalty)
            aln_coords_1[:, -1] *= 2. / len(msa_alignments[name_2])
            aln_coords_2[:, -1] *= 2. / len(msa_alignments[name_1])
            msa_alignments[name_1] = {name: "".join([sequence[i] if i != -1 else '-' for i in dtw_aln_1]) for name, sequence in
                                      msa_alignments[name_1].items()}
            msa_alignments[name_2] = {name: "".join([sequence[i] if i != -1 else '-' for i in dtw_aln_2]) for name, sequence in
                                      msa_alignments[name_2].items()}
            msa_alignments[name_int] = {**msa_alignments[name_1], **msa_alignments[name_2]}

            mean_coords = get_mean_coords_extra(aln_coords_1, aln_coords_2)
            mean_sec = get_mean_secondary(aln_sec_1, aln_sec_2, 0)
            self.final_structures.append(Structure(name_int, None, None, mean_sec, None, mean_coords))

        for x in range(0, self.tree.shape[0] - 1, 2):
            node_1, node_2, node_int = self.tree[x, 0], self.tree[x + 1, 0], self.tree[x, 1]
            assert self.tree[x + 1, 1] == node_int
            make_intermediate_node(node_1, node_2, node_int)

        node_1, node_2 = self.tree[-1, 0], self.tree[-1, 1]
        make_intermediate_node(node_1, node_2, "final")
        alignment = {**msa_alignments[self.final_structures[node_1].name], **msa_alignments[self.final_structures[node_2].name]}
        self.alignment = alignment
        return alignment

    def write_files(self, write_fasta=True,
                    write_pdb=True,
                    write_features=True,
                    write_class=True):
        if write_fasta:
            self.write_alignment(self.output_files.fasta_file)
        if write_pdb:
            if not self.output_files.pdb_folder.exists():
                self.output_files.pdb_folder.mkdir()
            self.write_superposed_pdbs(self.output_files.pdb_folder)
        if write_features:
            with open(str(self.output_files.feature_file), "wb") as f:
                pickle.dump(self.get_aligned_features(), f)
        if write_class:
            with open(str(self.output_files.class_file), "wb") as f:
                pickle.dump(self, f)

    def superpose(self, alignments: dict = None):
        """
        Superpose structures according to alignment
        """
        if alignments is None:
            alignments = self.alignment
        reference_index = 0
        reference_key = self.structures[reference_index].name
        core_indices = np.array([i for i in range(len(alignments[reference_key])) if '-' not in [alignments[n][i] for n in alignments]])
        aln_ref = helper.aligned_string_to_array(alignments[reference_key])
        ref_coords = self.structures[reference_index].coords[np.array([aln_ref[c] for c in core_indices])][:, :3]
        ref_centroid = rmsd_calculations.nb_mean_axis_0(ref_coords)
        ref_coords -= ref_centroid
        for i in range(len(self.structures)):
            if i == reference_index:
                self.structures[i].coords[:, :3] -= ref_centroid
            else:
                aln_c = helper.aligned_string_to_array(alignments[self.structures[i].name])
                common_coords_2 = self.structures[i].coords[np.array([aln_c[c] for c in core_indices])][:, :3]
                rotation_matrix, translation_matrix = rmsd_calculations.svd_superimpose(ref_coords, common_coords_2)
                self.structures[i].coords[:, :3] = rmsd_calculations.apply_rotran(self.structures[i].coords[:, :3], rotation_matrix,
                                                                                  translation_matrix)

    def make_pairwise_rmsd_coverage_matrix(self, alignments: dict = None, superpose_first: bool = True):
        """
        Find RMSDs and coverages of the alignment of each pair of sequences

        Parameters
        ----------
        alignments
        superpose_first
            if True then superposes all structures to first structure first

        Returns
        -------
        RMSD matrix, coverage matrix
        """
        if alignments is None:
            alignments = self.alignment
        num = len(self.structures)
        pairwise_rmsd_matrix = np.zeros((num, num))
        pairwise_rmsd_matrix[:] = np.nan
        pairwise_coverage = np.zeros((num, num))
        pairwise_coverage[:] = np.nan
        if superpose_first:
            self.superpose(alignments)
        for i in range(num):
            for j in range(i + 1, num):
                name_1, name_2 = self.structures[i].name, self.structures[j].name
                if isinstance(alignments[name_1], str):
                    aln_1 = helper.aligned_string_to_array(alignments[name_1])
                    aln_2 = helper.aligned_string_to_array(alignments[name_2])
                else:
                    aln_1 = alignments[name_1]
                    aln_2 = alignments[name_2]
                common_coords_1, common_coords_2 = get_common_coordinates(self.structures[i].coords[:, :3],
                                                                          self.structures[j].coords[:, :3], aln_1, aln_2)
                assert common_coords_1.shape[0] > 0
                if not superpose_first:
                    rot, tran = rmsd_calculations.svd_superimpose(common_coords_1, common_coords_2)
                    common_coords_2 = rmsd_calculations.apply_rotran(common_coords_2, rot, tran)
                pairwise_rmsd_matrix[i, j] = pairwise_rmsd_matrix[j, i] = rmsd_calculations.get_rmsd(common_coords_1, common_coords_2)
                pairwise_coverage[i, j] = pairwise_coverage[j, i] = common_coords_1.shape[0] / len(aln_1)
        return pairwise_rmsd_matrix, pairwise_coverage

    def get_aligned_features(self, alignments: dict = None):
        """
        Get dict of aligned features
        """
        if alignments is None:
            alignments = self.alignment
        feature_names = list(self.structures[0].features.keys())
        aligned_features = {}
        alignment_length = len(alignments[self.structures[0].name])
        for feature_name in feature_names:
            if feature_name == "secondary":
                continue
            aligned_features[feature_name] = np.zeros((len(self.structures), alignment_length))
            aligned_features[feature_name][:] = np.nan
            for p in range(len(self.structures)):
                farray = self.structures[p].features[feature_name]
                if "gnm" in feature_name or "anm" in feature_name:
                    farray = farray / np.nansum(farray ** 2) ** 0.5
                indices = [i for i in range(alignment_length) if alignments[self.structures[p].name][i] != '-']
                aligned_features[feature_name][p, indices] = farray
        return aligned_features

    def write_alignment(self, filename, alignments: dict = None):
        """
        Writes alignment to a fasta file
        """
        if alignments is None:
            alignments = self.alignment
        with open(filename, "w") as f:
            for key in alignments:
                f.write(f">{key}\n{alignments[key]}\n")

    def write_superposed_pdbs(self, output_pdb_folder, alignments: dict = None):
        """
        Superposes PDBs according to alignment and writes transformed PDBs to files
        (View with Pymol)

        Parameters
        ----------
        alignments
        output_pdb_folder
        """
        if alignments is None:
            alignments = self.alignment
        output_pdb_folder = Path(output_pdb_folder)
        if not output_pdb_folder.exists():
            output_pdb_folder.mkdir()
        reference_name = self.structures[0].name
        reference_pdb = pd.parsePDB(self.structures[0].pdb_file)
        core_indices = np.array([i for i in range(len(alignments[reference_name])) if '-' not in [alignments[n][i] for n in alignments]])
        aln_ref = helper.aligned_string_to_array(alignments[reference_name])
        ref_coords_core = reference_pdb[helper.get_alpha_indices(reference_pdb)].getCoords().astype(np.float64)[
            np.array([aln_ref[c] for c in core_indices])]
        ref_centroid = rmsd_calculations.nb_mean_axis_0(ref_coords_core)
        transformation = pd.Transformation(np.eye(3), -ref_centroid)
        reference_pdb = pd.applyTransformation(transformation, reference_pdb)
        pd.writePDB(str(output_pdb_folder / f"{reference_name}.pdb"), reference_pdb)
        for i in range(1, len(self.structures)):
            name = self.structures[i].name
            pdb = pd.parsePDB(self.structures[i].pdb_file)
            aln_name = helper.aligned_string_to_array(alignments[name])
            common_coords_2 = pdb[helper.get_alpha_indices(pdb)].getCoords().astype(np.float64)[np.array([aln_name[c] for c in core_indices])]
            rotation_matrix, translation_matrix = rmsd_calculations.svd_superimpose(ref_coords_core, common_coords_2)
            transformation = pd.Transformation(rotation_matrix, translation_matrix)
            pdb = pd.applyTransformation(transformation, pdb)
            pd.writePDB(str(output_pdb_folder / f"{name}.pdb"), pdb)
