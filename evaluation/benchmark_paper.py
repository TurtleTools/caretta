import typing
from collections import defaultdict
from itertools import groupby
from pathlib import Path

import numpy as np
import prody as pd
import requests as rq

from caretta import feature_extraction
from caretta import helper, rmsd_calculations
from caretta import msa_numba
from caretta.old import multiple_structure_alignment as msa, pairwise_structure_alignment as psa
from evaluation import plotting

pd.confProDy(verbosity='none')


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


HSCALE = {'ALA': 1.8,
          'ARG': -4.5,
          'ASN': -3.5,
          'ASP': -3.5,
          'CYS': 2.5,
          'GLN': -3.5,
          'GLU': -3.5,
          'GLY': -0.4,
          'HIS': -3.2,
          'ILE': 4.5,
          'LEU': 3.8,
          'LYS': -3.9,
          'MET': 1.9,
          'PHE': 2.8,
          'PRO': -1.6,
          'SER': -0.8,
          'THR': -0.7,
          'TRP': -0.9,
          'TYR': -1.3,
          'VAL': 4.2}


def get_structures(pdb_dir, dssp_dir, feature_names):
    """
    Get list of Structure objects from a directory of PDB files
    """
    pdbs = get_pdbs(pdb_dir)
    names = list(pdbs.keys())
    sequences = get_sequences(pdbs)
    coordinates = [pdbs[n][helper.get_alpha_indices(pdbs[n])].getCoords().astype(np.float64) for n in names]
    features = feature_extraction.get_features_multiple([str(pdb_dir / f"{name}.ent") for name in names], str(dssp_dir))
    structures = [psa.Structure(names[i], sequences[names[i]], coordinates[i], psa.string_to_array(features[i]["secondary"]), features[i],
                                make_feature_matrix=False, feature_names=feature_names) for i in range(len(names))]
    return structures


def get_structures_nb(pdb_files, pdb_names, dssp_dir):
    """
    Get list of Structure objects from a directory of PDB files
    """
    pdbs = [pd.parsePDB(filename) for filename in pdb_files]
    alpha_indices = [helper.get_alpha_indices(pdb) for pdb in pdbs]
    sequences = [pdbs[i][alpha_indices[i]].getSequence() for i in range(len(pdbs))]
    coordinates = [pdbs[i][alpha_indices[i]].getCoords().astype(np.float64) for i in range(len(pdbs))]
    # coordinates = [c - rmsd_calculations._nb_mean_axis_0(c) for c in coordinates]
    features = feature_extraction.get_features_multiple(pdb_files, str(dssp_dir))
    structures = []
    for i in range(len(pdbs)):
        # secondary = []
        # for s in features[i]["secondary"]:
        #     if s in 'HGI':
        #         secondary.append('H')
        #     elif s in 'BE':
        #         secondary.append('B')
        #     elif s in 'TS':
        #         secondary.append('T')
        #     else:
        #         secondary.append(s)
        structures.append(msa_numba.Structure(pdb_names[i], sequences[i],
                                              helper.secondary_to_array(features[i]["secondary"]),
                                              coordinates[i]))
    return structures


def get_structures_names(names, pdb_dir, dssp_dir):
    """
    Get list of Structure objects from a directory of PDB files
    """
    pdbs = {name: pd.parsePDB(str(pdb_dir / f"{name}.pdb")).select("protein") for name in names}
    names = list(pdbs.keys())
    sequences = get_sequences(pdbs)
    coordinates = [pdbs[n][helper.get_alpha_indices(pdbs[n])].getCoords().astype(np.float64) for n in names]
    features = feature_extraction.get_features_multiple([str(pdb_dir / f"{name}.pdb") for name in names], str(dssp_dir))
    structures = [psa.Structure(names[i], sequences[names[i]], coordinates[i], psa.string_to_array(features[i]["secondary"]), features[i],
                                make_feature_matrix=False, feature_names=()) for i in range(len(names))]
    return structures


def get_msa(pdb_dir, sequence_dir, dssp_dir, feature_names, name="ref", gap_open_penalty: float = 0., gap_extend_penalty: float = 0.):
    """
    Example usage of above functions to make a structure-guided multiple sequence alignment from a directory of PDB files
    """
    structures = get_structures(pdb_dir, dssp_dir, feature_names)
    msa_class = msa.StructureMultiple(structures)
    aln_sequences = get_sequence_alignment({s.name: s.sequence for s in msa_class.structures}, sequence_dir, name)
    return msa_class.align(aln_sequences, gap_open_penalty, gap_extend_penalty)


def get_aligned_features(features_dict):
    feature_names = list(features_dict.keys())
    proteins = list(features_dict[feature_names[0]].keys())
    num_indices = len(features_dict[feature_names[0]][proteins[0]])
    aligned_features = np.zeros((len(proteins), num_indices * len(features_dict)))
    aligned_features[:] = np.nan
    mask = np.arange(num_indices)
    for f, feature_name in enumerate(feature_names):
        mask_f = mask * len(feature_names) + f
        for p, protein in enumerate(proteins):
            farray = features_dict[feature_name][protein]
            if "gnm" in feature_name or "anm" in feature_name:
                farray = farray / np.nansum(farray ** 2) ** 0.5
            aligned_features[p, mask_f] = farray
    return aligned_features, feature_names, proteins


def get_pdb_file(pdb_id, chain_id, start, end, index, from_atm_file=None):
    if from_atm_file is not None:
        pdb_file = Path(f"{from_atm_file}{(pdb_id + chain_id).lower()}.atm")
        if pdb_file.exists():
            pdb_obj = pd.parsePDB(str(pdb_file))
        else:
            name = f"{pdb_id}{chain_id}{index}".lower()
            pdb_file = f"{from_atm_file}{name}.atm"
            if Path(pdb_file).exists():
                pdb_obj = pd.parsePDB(pdb_file)
            else:
                pdb_obj = pd.parsePDB(f"{from_atm_file}{pdb_id.lower()}.atm")
    else:
        if chain_id == "":
            chain_id = 'A'
        pdb_obj = pd.parsePDB(pdb_id, chain=chain_id)
        pdb_obj = pdb_obj.select(f"protein and resnum {start} : {end + 1}")
    filename = f"{pdb_id}_{chain_id}_{start}.pdb"
    pd.writePDB(filename, pdb_obj)
    return filename


class SABmark_matt:
    def __init__(self, directory_name, folder):
        self.folder = Path(folder)
        self.dssp_dir = self.folder / "dssp_files"
        self.directory_name = directory_name
        self.structure_dir = self.folder / "superfamily" / self.directory_name / "domains"
        pdb_files = list(self.structure_dir.glob("*.ent"))
        print("Num proteins", len(pdb_files))
        pdb_names = [helper.get_file_parts(p)[1] for p in pdb_files]
        structures = get_structures_nb(pdb_files, pdb_names, dssp_dir=self.dssp_dir)
        self.msa_class = msa_numba.StructureMultiple(structures)
        self.caretta_alignment = dict()
        self.matt_alignment = dict()
        self.sequence_alignment = dict()

    def compute(self, gamma, gap_open_sec=1, gap_extend_sec=0.1, gap_open_penalty=0.1, gap_extend_penalty=0.01):
        self.sequence_alignment = get_sequence_alignment({s.name: s.sequence.upper().replace('X', '') for s in self.msa_class.structures},
                                                         self.folder / "sequence_alignments", self.directory_name)
        self.matt_alignment = helper.get_sequences_from_fasta(
            self.folder / "superfamily" / self.directory_name / "matt_alignment" / f"{self.directory_name}.fasta")
        self.matt_alignment = {k.split('.')[0]: self.matt_alignment[k] for k in self.matt_alignment}
        self.caretta_alignment = self.msa_class.align(gamma=gamma, gap_open_sec=gap_open_sec, gap_extend_sec=gap_extend_sec,
                                                      gap_open_penalty=gap_open_penalty, gap_extend_penalty=gap_extend_penalty)
        self.write_alignments()
        self._get_rmsd_cov(gamma)

    def write_alignments(self):
        for name, aln in zip(["caretta", "sequence", "matt"],
                             [self.caretta_alignment, self.sequence_alignment,
                              self.matt_alignment]):
            with open(self.structure_dir / f"{self.directory_name}_{name}.fasta", "w") as f:
                for key in aln:
                    f.write(f">{key}\n{aln[key]}\n")

    def _get_rmsd_cov(self, gamma):
        for name, aln in zip(["caretta", "sequence", "matt"],
                             [self.caretta_alignment, self.sequence_alignment,
                              self.matt_alignment]):
            try:
                rmsd_matrix, cov_matrix, frac_matrix, _ = self.msa_class.make_pairwise_rmsd_matrix(gamma, aln)
                print(name, np.nanmean(rmsd_matrix), np.nanmean(cov_matrix), np.nanmean(frac_matrix),
                      np.nanmean(cov_matrix) * np.nanmean(frac_matrix))
            except Exception as e:
                if e == KeyboardInterrupt:
                    break
                print(name, "could not be done..")

    def pairwise_all_vs_all_compare(self, t="caretta"):
        if t == "caretta":
            query = self.caretta_alignment
        else:
            query = self.matt_alignment
        equi = np.zeros((len(query), len(query)))
        rmsd = np.zeros((len(query), len(query)))
        core_scores = np.zeros((len(query), len(query)))
        self.msa_class.superpose(query)
        try:
            for i in range(len(query)):
                for j in range(len(query)):
                    distances, k1, k2 = pairwise_compare(query, self.msa_class, i, j)
                    core_distance, k5, k6 = pairwise_compare_core(query, self.msa_class, i, j)
                    rmsd[i, j] = np.sum(np.exp(-0.03 * core_distance ** 2))
                    low_distances = np.where(distances <= 4)[0]
                    equi[i, j] = low_distances.shape[0]
                    core_scores[i, j] = core_distance[np.where(core_distance <= 4)].shape[0]

        except Exception as e:
            pass
        return equi, rmsd, core_scores


class SABmark_mtmalign:
    def __init__(self, directory_name, folder, dssp_dir):
        self.folder = Path(folder)
        self.dssp_dir = dssp_dir
        self.directory_name = directory_name
        self.mtm_url = f"http://yanglab.nankai.edu.cn/mTM-align/benchmark/SABmark-sup/{self.directory_name}/{self.directory_name}_result.fasta"
        self.structure_dir = self.folder / self.directory_name / "pdb"
        pdb_files = list(self.structure_dir.glob("*.ent"))
        print("Num proteins:", len(pdb_files))
        pdb_names = [helper.get_file_parts(p)[1] for p in pdb_files]
        structures = get_structures_nb(pdb_files, pdb_names, dssp_dir=self.dssp_dir)
        self.msa_class = msa_numba.StructureMultiple(structures)
        self.caretta_alignment = dict()
        self.mtm_alignment = dict()
        self.sequence_alignment = dict()

    def compute(self, gamma, gap_open_sec=1, gap_extend_sec=0.1, gap_open_penalty=0.1, gap_extend_penalty=0.01):
        self.parse_mtm_fasta()
        self.caretta_alignment = self.msa_class.align(gamma=gamma, gap_open_sec=gap_open_sec, gap_extend_sec=gap_extend_sec,
                                                      gap_open_penalty=gap_open_penalty, gap_extend_penalty=gap_extend_penalty)
        self.write_alignments()
        self._get_rmsd_cov(gamma)

    def write_alignments(self):
        for name, aln in zip(["caretta", "mtm"],
                             [self.caretta_alignment,
                              self.mtm_alignment]):
            with open(self.structure_dir / f"{self.directory_name}_{name}.fasta", "w") as f:
                for key in aln:
                    f.write(f">{key}\n{aln[key]}\n")

    def _get_rmsd_cov(self, gamma):
        for name, aln in zip(["caretta", "mtm"],
                             [self.caretta_alignment,
                              self.mtm_alignment]):
            try:
                rmsd_matrix, cov_matrix, frac_matrix, _ = self.msa_class.make_pairwise_rmsd_matrix(gamma, aln)
                print(name, np.nanmean(rmsd_matrix), np.nanmean(cov_matrix), np.nanmean(frac_matrix),
                      np.nanmean(cov_matrix) * np.nanmean(frac_matrix))
            except Exception as e:
                if e == KeyboardInterrupt:
                    break
                print(name, "could not be done..")

    def pairwise_all_vs_all_compare(self, t="caretta"):
        if t == "caretta":
            query = self.caretta_alignment
        else:
            query = self.mtm_alignment
        equi = np.zeros((len(query), len(query)))
        rmsd = np.zeros((len(query), len(query)))
        core_scores = np.zeros((len(query), len(query)))
        self.msa_class.superpose(query)
        try:
            for i in range(len(query)):
                for j in range(len(query)):
                    distances, k1, k2 = pairwise_compare(query, self.msa_class, i, j)
                    core_distance, k5, k6 = pairwise_compare_core(query, self.msa_class, i, j)
                    low_distances = np.where(distances <= 4)[0]
                    equi[i, j] = low_distances.shape[0]
                    rmsd[i, j] = np.sum(np.exp(-0.15 * core_distance))
                    core_scores[i, j] = core_distance[np.where(core_distance <= 4)].shape[0]
        except Exception as e:
            pass
        return equi, rmsd, core_scores

    def parse_mtm_fasta(self):
        fasta_lines = rq.get(self.mtm_url).text.split("\n")
        sequence = False
        fasta_header = ""
        for _, content in groupby(fasta_lines, lambda x: x.startswith(">")):
            if not sequence:
                fasta_header = list(content)[0][1:].strip().split('.')[0]
                sequence = True
            else:
                self.mtm_alignment[fasta_header] = "".join([x.strip() for x in list(content)])
                sequence = False


class HomstradBenchmark:
    def __init__(self, filename):
        self.homstrad_id = filename.split("/")[-1].split(".")[0]
        self.matt_url = f"http://cb.csail.mit.edu/cb/matt/homstrad/{self.homstrad_id}.fasta"
        self.mTMA_url = f"http://yanglab.nankai.edu.cn/mTM-align/benchmark/homstrad/{self.homstrad_id}/{self.homstrad_id}_result.fasta"
        self.homstrad_url = f"http://yanglab.nankai.edu.cn/mTM-align/benchmark/homstrad/{self.homstrad_id}/{self.homstrad_id}.ali"
        self.output_folder = "/".join(filename.split("/")[:-1]) + "/"
        self.homstrad_ali_file = filename
        self.pdb_entries = dict()
        self.homstrad_alignment = dict()
        self.caretta_alignment = dict()
        self.matt_alignment = dict()
        self.sequence_alignment = dict()
        self.mTMA_alignment = dict()
        self.valid = False
        self.msa_class = None
        self.reference_key = None
        self.core_blocks = dict()

    def compute_and_store(self, gamma, gap_open_sec=1, gap_extend_sec=0.1, gap_open_penalty=0.1, gap_extend_penalty=0.08):
        self.parse_homstrad()
        self.parse_matt_fasta()
        self.parse_mTMalign_fasta()
        msa_class = self.caretta_align(gamma=gamma, gap_open_sec=gap_open_sec, gap_extend_sec=gap_extend_sec,
                                       gap_extend_penalty=gap_extend_penalty, gap_open_penalty=gap_open_penalty)
        self.write_alignments()
        self.reference_key = list(self.homstrad_alignment.keys())[0]
        self.get_homstrad_cores()
        self._get_rmsd_cov(msa_class, gamma)
        self.msa_class = msa_class
        return msa_class

    def parse_homstrad(self):
        indices = defaultdict(int)
        with open(self.homstrad_ali_file, "r") as f:
            _key = True
            fasta_names = list()
            sequences = list()
            for (_, content) in groupby(filter(lambda x: not x.startswith("C;"), f.readlines()),
                                        lambda x: x.startswith(">") or x.startswith("structure")):
                content = list(content)
                if _key:
                    headers = content[1].split(":")
                    if headers[3] == " ":
                        chain_id = ""
                    else:
                        chain_id = headers[3]
                    pdb_id = headers[1]
                    indices[pdb_id] += 1
                    start = int(headers[2])
                    end = int(headers[4])
                    unique_name = f"{pdb_id}_{chain_id}_{start}"
                    fasta_names.append(unique_name)
                    self.pdb_entries[unique_name] = (pdb_id, chain_id, start, end, indices[pdb_id])
                    _key = False
                else:
                    sequences.append("".join(list(map(str.strip, content))).replace("*", ""))
                    self.homstrad_alignment[fasta_names[-1]] = sequences[-1]
                    _key = True

    def parse_homstrad_from_url(self):
        lines = rq.get(self.homstrad_url).text.split("\n")
        _key = True
        fasta_names = list()
        sequences = list()
        for (_, content) in groupby(filter(lambda x: not x.startswith("C;"), lines), lambda x: x.startswith(">") or x.startswith("structure")):
            content = list(content)
            if _key:
                headers = content[1].split(":")
                if headers[3] == " ":
                    chain_id = ""
                else:
                    chain_id = headers[3]
                pdb_id = headers[1]
                start = int(headers[2])
                end = int(headers[4])
                header = (pdb_id + chain_id).lower()
                fasta_names.append(header)
                self.pdb_entries[header] = (pdb_id, chain_id, start, end)
                _key = False
            else:
                sequences.append("".join(list(map(str.strip, content))).replace("*", "").replace("/", "-"))
                self.homstrad_alignment[fasta_names[-1]] = sequences[-1]
                _key = True

    def caretta_align(self, gamma, gap_open_sec=1, gap_extend_sec=0.1, gap_open_penalty=0.1, gap_extend_penalty=0.08, dssp_dir="./dssp_features/"):
        pdb_entries = list(self.pdb_entries.values())
        print("Num proteins:", len(pdb_entries))
        pdb_files = [get_pdb_file(x[0], x[1], x[2], x[3], x[4], from_atm_file=self.output_folder) for x in pdb_entries]
        pdb_names = [f"{pdb_id}_{chain_id}_{start}" for (pdb_id, chain_id, start, _, _) in pdb_entries]
        structures = get_structures_nb(pdb_files, pdb_names, dssp_dir)
        self.sequence_alignment = get_sequence_alignment({s.name: s.sequence.upper().replace('X', '') for s in structures},
                                                         self.output_folder, self.homstrad_id)
        self.valid = self.validate_sequence()
        msa_class = msa_numba.StructureMultiple(structures)
        self.caretta_alignment = msa_class.align(gamma=gamma, gap_open_sec=gap_open_sec, gap_extend_sec=gap_extend_sec,
                                                 gap_open_penalty=gap_open_penalty, gap_extend_penalty=gap_extend_penalty)
        return msa_class

    def validate_sequence(self):
        for k in self.sequence_alignment:
            if not (self.sequence_alignment[k].replace("-", "").upper() == self.homstrad_alignment[k].replace("-", "").upper()):
                return False
        return True

    def write_alignments(self):
        for name, aln in zip(["caretta", "sequence", "homstrad", "matt", "mTMalign"],
                             [self.caretta_alignment, self.sequence_alignment,
                              self.homstrad_alignment, self.matt_alignment, self.mTMA_alignment]):
            with open(Path(self.output_folder) / f"{self.homstrad_id}_{name}.fasta", "w") as f:
                for key in aln:
                    f.write(f">{key}\n{aln[key]}\n")

    def _get_rmsd_cov(self, aln_obj, gamma):
        for name, aln in zip(["caretta", "sequence", "homstrad", "matt", "mTMalign"],
                             [self.caretta_alignment, self.sequence_alignment,
                              self.homstrad_alignment, self.matt_alignment, self.mTMA_alignment]):
            try:
                rmsd_matrix, cov_matrix, frac_matrix, _ = aln_obj.make_pairwise_rmsd_matrix(gamma, aln)
                print(name, np.nanmean(rmsd_matrix), np.nanmean(cov_matrix), np.nanmean(frac_matrix),
                      np.nanmean(cov_matrix) * np.nanmean(frac_matrix))
            except Exception as e:
                if e == KeyboardInterrupt:
                    break
                print(name, "could not be done..")

    def score_matrix(self, aln_obj):
        for name, aln in zip(["caretta", "sequence", "homstrad", "matt", "mTMalign"],
                             [self.caretta_alignment, self.sequence_alignment,
                              self.homstrad_alignment, self.matt_alignment, self.mTMA_alignment]):
            score_matrix, _ = aln_obj.make_pairwise_dtw_rmsd_matrix({k: helper.aligned_string_to_array(aln[k]) for k in aln})
            print(f"{name}:")
            plotting.plot_distance_matrix(score_matrix, [str(len(aln[k.name].replace('-', ''))) for k in aln_obj.structures])

    def parse_matt_fasta(self):
        fasta_lines = rq.get(self.matt_url).text.split("\n")
        sequence = False
        fasta_header = ""
        for _, content in groupby(fasta_lines, lambda x: x.startswith(">")):
            if not sequence:
                fasta_header = list(content)[0][1:].strip()
                sequence = True
            else:
                self.matt_alignment[fasta_header] = "".join([x.strip() for x in list(content)])
                sequence = False
        self._update_matt_headers()

    def parse_mTMalign_fasta(self):
        fasta_lines = rq.get(self.mTMA_url).text.split("\n")
        sequence = False
        fasta_header = ""
        for _, content in groupby(fasta_lines, lambda x: x.startswith(">")):
            if not sequence:
                current_header = list(content)[0][1:].strip().split(".")[0].lower()[:4]
                fasta_header = [x for x in list(self.homstrad_alignment.keys()) if x.lower().startswith(current_header)][0]
                sequence = True
            else:
                self.mTMA_alignment[fasta_header] = "".join([x.strip() for x in list(content)])
                sequence = False

    def _update_matt_headers(self):
        matt_with_new_headers = dict()
        for old_header, sequence in self.matt_alignment.items():
            for good_header, good_sequence in self.homstrad_alignment.items():
                if good_sequence.replace("-", "") == sequence.replace("-", ""):
                    matt_with_new_headers[good_header] = sequence
                else:
                    continue
        self.matt_alignment = matt_with_new_headers

    def get_homstrad_cores(self):
        self.reference_key = list(self.homstrad_alignment.keys())[0]
        all_sequences = [x[1] for x in sorted(self.homstrad_alignment.items(), key=lambda x: x[0])]
        n = 0
        for i in range(len(self.homstrad_alignment[self.reference_key])):
            if self.homstrad_alignment[self.reference_key][i] == "-":
                continue
            else:
                block = tuple([x[i] for x in all_sequences])
            if "-" in block:
                self.core_blocks[n] = None
                n += 1
                continue
            else:
                self.core_blocks[n] = block
                n += 1

    def compare_alignment_to_core_blocks(self, alignment):
        all_sequences = [x[1] for x in sorted(alignment.items(), key=lambda x: x[0])]
        n = 0
        score = 0
        for i in range(len(all_sequences[0])):
            if alignment[self.reference_key][i] == "-":
                continue
            else:
                block = tuple([x[i] for x in all_sequences])
            if "-" in block:
                n += 1
                continue
            elif self.core_blocks[n] == block:
                score += 1
                n += 1
            else:
                n += 1
        return score


def superimpose(msa_class, alignment_np):
    reference_structure = msa_class.structures[max([(len(s.sequence), i) for i, s in enumerate(msa_class.structures)])[1]]
    for i in range(len(msa_class.structures)):
        pos_1, pos_2 = helper.get_common_positions(alignment_np[reference_structure.name],
                                                   alignment_np[msa_class.structures[i].name])
        common_coords_1, common_coords_2 = reference_structure.coords[pos_1][:, :3], msa_class.structures[i].coords[pos_2][:, :3]
        rot, tran = rmsd_calculations.svd_superimpose(common_coords_1, common_coords_2)
        msa_class.structures[i].coords[:, :3] = rmsd_calculations.apply_rotran(msa_class.structures[i].coords[:, :3], rot, tran)
    return msa_class


def pairwise_compare(alignment, msa_class, i, j):
    alignment_np = {k: helper.aligned_string_to_array(alignment[k]) for k in alignment}
    # msa_class = superimpose(msa_class, alignment_np)
    i1, i2 = i, j
    pos_1, pos_2 = helper.get_common_positions(alignment_np[msa_class.structures[i1].name],
                                               alignment_np[msa_class.structures[i2].name])
    common_coords_1, common_coords_2 = msa_class.structures[i1].coords[pos_1][:, :3], msa_class.structures[i2].coords[pos_2][:, :3]
    return np.sqrt(np.sum((common_coords_1 - common_coords_2) ** 2, axis=-1)), msa_class.structures[i1].name, msa_class.structures[i2].name


def pairwise_compare_core(alignment, msa_class, i, j):
    i1, i2 = i, j
    core_indices = np.array(get_core_indices(alignment), dtype=int)
    noncore_indices = [i for i in range(len(alignment[msa_class.structures[i1].name])) if i not in core_indices]
    aln_np = {k: helper.aligned_string_to_array(alignment[k]) for k in alignment}
    alignment_1 = aln_np[msa_class.structures[i1].name]
    alignment_2 = aln_np[msa_class.structures[i2].name]
    alignment_1[noncore_indices] = -1
    alignment_2[noncore_indices] = -1
    pos_1, pos_2 = helper.get_common_positions(alignment_1,
                                               alignment_2)
    common_coords_1, common_coords_2 = msa_class.structures[i1].coords[pos_1][:, :3], msa_class.structures[i2].coords[pos_2][:, :3]
    return np.sqrt(np.sum((common_coords_1 - common_coords_2) ** 2, axis=-1)), msa_class.structures[i1].name, msa_class.structures[i2].name


def get_core_indices(alignment):
    indices = list()
    for i in range(len(list(alignment.values())[0])):
        full = True
        for seq in alignment.values():
            if seq[i] == "-":
                full = False
                break
            else:
                pass
        if full:
            indices.append(i)
        else:
            continue
    return indices


def pairwise_all_vs_all_compare(homstrad_alignment, t="caretta", compute=True):
    if compute:
        msa_class = homstrad_alignment.compute_and_store()
    else:
        msa_class = homstrad_alignment.msa_class
    ref = homstrad_alignment.homstrad_alignment
    if t == "caretta":
        query = homstrad_alignment.caretta_alignment
    elif t == "matt":
        query = homstrad_alignment.matt_alignment
    else:
        query = homstrad_alignment.mTMA_alignment
    msa_class.superpose(query)
    equi = np.zeros((len(query), len(query)))
    rmsd = np.zeros((len(query), len(query)))
    homstrad_score = np.zeros((len(query), len(query)))
    core_scores = np.zeros((len(query), len(query)))
    try:
        for i in range(len(query)):
            for j in range(len(query)):
                distances, k1, k2 = pairwise_compare(query, msa_class, i, j)
                distances_homstrad, k3, k4 = pairwise_compare(ref, msa_class, i, j)
                core_distance, k5, k6 = pairwise_compare_core(query, msa_class, i, j)
                assert (k1 == k3) and (k2 == k4) and (k5 == k1) and (k6 == k4)
                ref_seqs = clean_sequences(ref[k1], ref[k2])
                seqs = clean_sequences(query[k1], query[k2])
                homstrad_score[i, j] = compare_seq_pairs(ref_seqs, seqs, distances) / compare_seq_pairs(ref_seqs, ref_seqs, distances_homstrad)
                rmsd[i, j] = np.mean(distances)
                low_distances = np.where(distances <= 4)[0]
                equi[i, j] = low_distances.shape[0]
                core_scores[i, j] = core_distance[core_distance <= 4].shape[0]
    except Exception as e:
        pass
    return equi, rmsd, core_scores, homstrad_score


def compare_seq_pairs(ref_pair, q_pair, distances):
    ref_alignment, _ = get_seq_dict_from_pair(ref_pair)
    q_alignment, doubles = get_seq_dict_from_pair(q_pair)
    s = 0
    for k in ref_alignment:
        try:
            if q_alignment[k] == ref_alignment[k]:
                if distances[doubles[k]] < 4:
                    s += 1
        except KeyError:
            continue
    return s


def get_seq_dict_from_pair(ref_pair):
    c = 0
    ref_alignment = dict()
    d = 0
    double_indices = dict()
    for i in range(len(ref_pair[0])):
        if (ref_pair[0][i] != "-") and (ref_pair[1][i] != "-"):
            ref_alignment[c] = (ref_pair[0][i], ref_pair[1][i])
            double_indices[c] = d
            d += 1
            c += 1
        elif ref_pair[0][i] != "-":
            c += 1
        else:
            continue
    return ref_alignment, double_indices


def clean_sequences(seq1, seq2):
    seq1_mod = ""
    seq2_mod = ""
    for i in range(len(seq1)):
        if (seq1[i] == "-") and (seq2[i] == "-"):
            continue
        else:
            seq1_mod += seq1[i]
            seq2_mod += seq2[i]
    return seq1_mod, seq2_mod
