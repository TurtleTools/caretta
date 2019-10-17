from dataclasses import dataclass
from pathlib import Path

import prody as pd
import requests as rq

from caretta import msa_numba


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
    PDB_file: str = None

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
        return cls(pdb_path, chain_id, -1, -1, "none", "none", "none", 0.0, pdb_path)

    def get_pdb(self, from_atm_file=None):
        if from_atm_file is not None:
            pdb_obj = pd.parsePDB(f"{from_atm_file}{(self.PDB_ID + self.CHAIN_ID).lower()}.atm")
        else:
            if self.PDB_file is not None:
                pdb_obj = pd.parsePDB(self.PDB_file)
            else:
                pdb_obj = pd.parsePDB(self.PDB_ID, chain=self.CHAIN_ID)  # TODO : change mkdir and etc..
            if self.PdbResNumStart == -1 and self.PdbResNumEnd == -1:
                pdb_obj = pdb_obj.select(f"protein")
            else:
                pdb_obj = pdb_obj.select(f"protein and resnum {self.PdbResNumStart} : {self.PdbResNumEnd + 1}")
        if self.PDB_file is None:
            filename = f"{self.PDB_ID}_{self.CHAIN_ID}_{self.PdbResNumStart}.pdb"
            pd.writePDB(filename, pdb_obj)
        else:
            filename = self.PDB_file
        return pd.parsePDB(filename), filename

    def get_features(self, from_file=None):
        try:
            pdb_obj, filename = self.get_pdb(from_atm_file=from_file)
        except TypeError:
            return None, None
        return pdb_obj, filename

    @property
    def filename(self):
        if self.PDB_file is None:
            return f"{self.PDB_ID}_{self.CHAIN_ID}_{self.PdbResNumStart}.pdb"
        else:
            return self.PDB_file

    @property
    def unique_name(self):
        return f"{self.PDB_ID}_{self.CHAIN_ID}_{self.PdbResNumStart}"


def get_pdbs_from_folder(path):
    file_names = Path(path).glob("*.pdb")
    res = []
    for f in file_names:
        res.append(PdbEntry.from_user_input(str(f)))
    return res


class PfamToPDB:
    def __init__(self, uri="https://www.rcsb.org/pdb/rest/hmmer?file=hmmer_pdb_all.txt",
                 from_file=False,
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
        data_lines = data_lines[1:]
        self.pfam_to_pdb_ids = dict()
        self._initiate_pfam_to_pdbids(data_lines, limit=limit)
        self.msa = None
        self.caretta_alignment = None

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
            for k, v in self.pfam_to_pdb_ids.items():
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
        self.msa = PfamStructures.from_pdb_files([p.get_pdb()[1] for p in pdb_entries])
        self.caretta_alignment = self.msa.align(gap_open_penalty=gap_open_penalty, gap_extend_penalty=gap_extend_penalty)
        return self.caretta_alignment, {s.name: pd.parsePDB(s.pdb_file) for s in self.msa.structures}, {s.name: s.features for s in
                                                                                                        self.msa.structures}

    def get_entries_for_pfam(self, pfam_id, limit_by_score=1., limit_by_protein_number=50, gross_limit=1000):
        pdb_entries = list(filter(lambda x: (x.eValue < limit_by_score), self.pfam_to_pdb_ids[pfam_id]))[:limit_by_protein_number]
        return pdb_entries

    def alignment_from_folder(self):
        pass

    def to_fasta_str(self, alignment):
        res = []
        for k, v in alignment.items():
            res.append(f">{k}")
            res.append(v)
        return "\n".join(res)


class PfamStructures(msa_numba.StructureMultiple):
    def __init__(self, pdb_entries, dssp_dir="caretta_tmp", num_threads=20, extract_all_features=True,
                 consensus_weight=1.,
                 write_fasta=True, output_fasta_filename=Path("./result.fasta"),
                 write_pdb=True, output_pdb_folder=Path("./result_pdb/"),
                 write_features=True, output_feature_filename=Path("./result_features.pkl"),
                 write_class=True, output_class_filename=Path("./result_class.pkl"),
                 overwrite_dssp=False):
        self.pdb_entries = pdb_entries
        super(PfamStructures, self).from_pdb_files([p.get_pdb()[1] for p in self.pdb_entries], dssp_dir,
                                                   num_threads, extract_all_features,
                                                   consensus_weight,
                                                   output_fasta_filename,
                                                   output_pdb_folder,
                                                   output_feature_filename,
                                                   output_class_filename,
                                                   overwrite_dssp)
