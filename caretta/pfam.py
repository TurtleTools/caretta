import prody as pd
from dataclasses import dataclass

from caretta import msa_numba as msa
from caretta import psa_numba as psa
from caretta import feature_extraction, helper, rmsd_calculations
import numpy as np
import requests as rq
import glob

@dataclass
class PdbEntry:
    PDB_ID: str
    CHAIN_ID: str
    PdbResNumStart: int
    PdbResNumEnd: int
    PFAM_ACC: str
    PFAM_Name: str
    PFAM_desc: str
    eValue: float

    @classmethod
    def from_data_line(cls, data_line):
        return cls(data_line[0], data_line[1], int(data_line[2]), int(data_line[3]),
                   data_line[4], data_line[5], data_line[6], float(data_line[7]))

    @classmethod
    def from_ali_header(cls, ali_header):
        headers = ali_header.split(":")
        if headers[3] == " ":
            chain_id = ""
        else:
            chain_id = headers[3]
        return cls(headers[1], chain_id, int(headers[2]), int(headers[4]),
                   "none", "none", "none", 0.0)

    @classmethod
    def from_user_input(cls, pdb_path, chain_id="A"):
        return cls(pdb_path, chain_id, -1, -1, "none", "none", "none", 0.0)

    def get_pdb(self, from_atm_file=None):
        if from_atm_file is not None:
            pdb_obj = pd.parsePDB(f"{from_atm_file}{(self.PDB_ID+self.CHAIN_ID).lower()}.atm")
        else:
            if self.CHAIN_ID == "":
                chain_id = 'A'
            pdb_obj = pd.parsePDB(self.PDB_ID, chain=self.CHAIN_ID) # TODO : change mkdir and etc..
            if self.PdbResNumStart == -1 and self.PdbResNumEnd == -1:
                pdb_obj = pdb_obj.select(f"protein")
            else:
                pdb_obj = pdb_obj.select(f"protein and resnum {self.PdbResNumStart} : {self.PdbResNumEnd+1}")
        filename = f"{self.PDB_ID}_{self.CHAIN_ID}_{self.PdbResNumStart}.pdb"
        pd.writePDB(filename, pdb_obj)
        return pd.parsePDB(filename), filename

    def get_features(self, from_file=None):
        try:
            pdb_obj, filename = self.get_pdb(from_atm_file=from_file)
        except TypeError:
            return (None, None)
        return (pdb_obj, filename)

    @property
    def filename(self):
        return f"{self.PDB_ID}_{self.CHAIN_ID}_{self.PdbResNumStart}.pdb"

    @property
    def unique_name(self):
        return f"{self.PDB_ID}_{self.CHAIN_ID}_{self.PdbResNumStart}"



def get_pdbs_from_folder(path):
    file_names = glob.glob(f"{path}/*.pdb")
    res = []
    for f in file_names:
        res.append(PdbEntry.from_user_input(f))
    return res


class PfamToPDB:
    def __init__(self, uri="https://www.rcsb.org/pdb/rest/hmmer?file=hmmer_pdb_all.txt", from_file=False,
                 limit=None):

        self.uri = uri
        if from_file:
            f = open("hmmer_pdb_all.txt", "r")
            raw_text = f.read()
            f.close()
        else:
            raw_text = rq.get(self.uri).text
        data_lines = [x.split("\t") for x in raw_text.split("\n")]
        self.headers = data_lines[0]
        self.pdb_entries = list()
        data_lines = data_lines[1:]
        self.pfam_to_pdb_ids = dict()
        self._initiate_pfam_to_pdbids(data_lines, limit=limit)

    def _initiate_pfam_to_pdbids(self, data_lines, limit=None):
        for line in data_lines:
            try:
                current_data = PdbEntry.from_data_line(line)
            except:
                continue
            if current_data.PFAM_ACC in self.pfam_to_pdb_ids:
                self.pfam_to_pdb_ids[current_data.PFAM_ACC].append(current_data)
            else:
                self.pfam_to_pdb_ids[current_data.PFAM_ACC] = [current_data]

        if limit:
            n = 0
            new = {}
            for k,v in self.pfam_to_pdb_ids.items():
                if n == limit:
                    break
                if len(v) < 4:
                    continue
                else:
                    new[k] = v
                    n += 1
            self.pfam_to_pdb_ids = new

    def multiple_structure_alignment_from_pfam(self, pdb_entries,
                                               gap_open_penalty=0.1,
                                               gap_extend_penalty=0.001):
        self.pdb_entries = pdb_entries
        objs_features = [x.get_features() for x in pdb_entries]
        pdb_objects = [x[0] for x in objs_features]
        pdb_features = feature_extraction.get_features_multiple([x[1] for x in objs_features],
                                                                "/mnt/local_scratch/akdel001/caretta_pfam/tmp/dssp_features/",
                                                                num_threads=20, only_dssp=False)
        pdb_names = [x.unique_name for x in pdb_entries]
        alpha_indices = [helper.get_alpha_indices(pdb) for pdb in pdb_objects]
        coordinates = [pdb[alpha_indices[i]].getCoords().astype(np.float64) for i, pdb in enumerate(pdb_objects)]
        structures = [msa.Structure(pdb_names[i],
                                    pdb_objects[i][alpha_indices[i]].getSequence(),
                                    helper.secondary_to_array(pdb_features[i]["secondary"]),
                                    # pdb_features[i],
                                    coordinates[i]) for i in range(len(pdb_names))]
        self.msa = msa.StructureMultiple(structures)
        self.caretta_alignment = self.msa.align(gamma=0.03, gap_open_sec=1, gap_extend_sec=0.01,
                                                gap_open_penalty=gap_open_penalty,
                                                gap_extend_penalty=gap_extend_penalty)
        return self.caretta_alignment, {pdb_names[i]: pdb_objects[i] for i in range(len(pdb_names))}, {pdb_names[i]: pdb_features[i] for i in range(len(pdb_names))}


    def get_entries_for_pfam(self, pfam_id, limit_by_score=1., limit_by_protein_number=40, gross_limit=1000):
        pdb_entries = list(filter(lambda x: (x.eValue < limit_by_score), self.pfam_to_pdb_ids[pfam_id]))[:limit_by_protein_number]
        return pdb_entries

    def alignment_from_folder(self):
        pass

    def to_fasta_str(self, alignment):
        res = []
        for k,v in alignment.items():
            res.append(f">{k}")
            res.append(v)
        return "\n".join(res)


def get_beta_indices_clean(protein: pd.AtomGroup) -> list:
    """
    Get indices of beta carbons of pd AtomGroup object
    """
    residue_splits = group_indices(protein.getResindices())
    i = 0
    indices = []
    for split in residue_splits:
        ca = None
        cb = None
        for _ in split:
            if protein[i].getName() == 'CB':
                cb = protein[i].getIndex()
            if protein[i].getName() == 'CA':
                ca = protein[i].getIndex()
            i += 1
        if cb is not None:
            indices.append(cb)
        else:
            assert ca is not None
            indices.append(ca)
    return indices


def group_indices(input_list: list) -> list:
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
