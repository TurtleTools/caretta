import base64
import datetime
import pickle
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import prody as pd
import requests as rq

import typing

from caretta import multiple_alignment


def heatmap(data):
    return dict(
        data=[dict(z=data, type="heatmap", showscale=False)],
        layout=dict(margin=dict(l=25, r=25, t=25, b=25)),
    )


def empty_dict():
    data = [dict(z=np.zeros((2, 2)), type="heatmap", showscale=False)]
    layout = dict(margin=dict(l=25, r=25, t=25, b=25))
    return dict(data=data, layout=layout)


def empty_object(suite):
    return compress_object(np.zeros(0), suite)


def get_estimated_time(msa_class: multiple_alignment.StructureMultiple):
    n = len(msa_class.structures)
    l = max(s.length for s in msa_class.structures)
    func = lambda x, r: (x[0] ** 2 * r * x[1] ** 2)
    return str(datetime.timedelta(seconds=int(func((l, n), 9.14726052e-06))))


def line(data):
    y = np.array([np.nanmean(data[:, x]) for x in range(data.shape[1])])
    y_se = np.array(
        [np.nanstd(data[:, x]) / np.sqrt(data.shape[1]) for x in range(data.shape[1])]
    )

    data = [
        dict(
            y=list(y + y_se) + list(y - y_se)[::-1],
            x=list(range(data.shape[1])) + list(range(data.shape[1]))[::-1],
            fillcolor="lightblue",
            fill="toself",
            type="scatter",
            mode="lines",
            name="Standard error",
            line=dict(color="lightblue"),
        ),
        dict(
            y=y,
            x=np.arange(data.shape[1]),
            type="scatter",
            mode="lines",
            name="Mean",
            line=dict(color="blue"),
        ),
    ]
    return dict(
        data=data,
        layout=dict(legend=dict(x=0.5, y=1.2), margin=dict(l=25, r=25, t=25, b=25)),
    )


def scatter3D(coordinates_dict):
    data = []
    for k, v in coordinates_dict.items():
        x, y, z = v[:, 0], v[:, 1], v[:, 2]
        data.append(
            dict(
                x=x,
                y=y,
                z=z,
                mode="lines",
                type="scatter3d",
                text=None,
                name=str(k),
                line=dict(width=3, opacity=0.8),
            )
        )
    layout = dict(
        margin=dict(l=20, r=20, t=20, b=20),
        clickmode="event+select",
        scene=dict(
            xaxis=dict(visible=False, showgrid=False, showline=False),
            yaxis=dict(visible=False, showgrid=False, showline=False),
            zaxis=dict(visible=False, showgrid=False, showline=False),
        ),
    )
    return dict(data=data, layout=layout)


def write_feature_as_tsv(
    feature_data: np.ndarray, keys: typing.List[str], file_name: typing.Union[Path, str]
):
    with open(file_name, "w") as f:
        for i in range(feature_data.shape[0]):
            f.write(
                "\t".join([keys[i]] + [str(x) for x in list(feature_data[i])]) + "\n"
            )


def compress_object(raw_object, suite):
    return base64.b64encode(suite.encrypt(pickle.dumps(raw_object, protocol=4))).decode(
        "utf-8"
    )


def decompress_object(compressed_object, suite):
    return pickle.loads(suite.decrypt(base64.b64decode(compressed_object)))


def protein_to_aln_index(protein_index, aln_seq):
    n = 0
    for i in range(len(aln_seq)):
        if protein_index == n:
            return i
        elif aln_seq[i] == "-":
            pass
        else:
            n += 1


def aln_index_to_protein(alignment_index, alignment):
    res = dict()
    for k, v in alignment.items():
        if v[alignment_index] == "-":
            res[k] = None
        else:
            res[k] = alignment_index - v[:alignment_index].count("-")
    return res


def to_fasta_str(alignment):
    res = []
    for k, v in alignment.items():
        res.append(f">{k}")
        res.append(v)
    return "\n".join(res)


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
        return cls(
            data_line[0],
            data_line[1],
            int(data_line[2]),
            int(data_line[3]),
            data_line[4],
            data_line[5],
            data_line[6],
            float(data_line[7]),
        )

    @classmethod
    def from_ali_header(cls, ali_header):
        headers = ali_header.split(":")
        if headers[3] == " ":
            chain_id = ""
        else:
            chain_id = headers[3]
        return cls(
            headers[1],
            chain_id,
            int(headers[2]),
            int(headers[4]),
            "none",
            "none",
            "none",
            0.0,
        )

    @classmethod
    def from_user_input(cls, pdb_path, chain_id="none"):
        return cls(
            Path(pdb_path).stem, chain_id, -1, -1, "none", "none", "none", 0.0, pdb_path
        )

    def get_pdb(self, from_atm_file=None):
        if from_atm_file is not None:
            pdb_obj = pd.parsePDB(
                f"{from_atm_file}{(self.PDB_ID + self.CHAIN_ID).lower()}.atm"
            )
        else:
            if self.PDB_file is not None:
                pdb_obj = pd.parsePDB(self.PDB_file)
            else:
                pdb_obj = pd.parsePDB(
                    self.PDB_ID, chain=self.CHAIN_ID
                )  # TODO : change mkdir and etc..
            if self.PdbResNumStart == -1 and self.PdbResNumEnd == -1:
                pdb_obj = pdb_obj.select(f"protein")
            else:
                pdb_obj = pdb_obj.select(
                    f"protein and resnum {self.PdbResNumStart} : {self.PdbResNumEnd + 1}"
                )
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


class PfamToPDB:
    def __init__(
        self,
        uri="https://www.rcsb.org/pdb/rest/hmmer?file=hmmer_pdb_all.txt",
        from_file=False,
        limit=None,
    ):
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
        self.pdb_entries = None
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

    def get_entries_for_pfam(
        self, pfam_id, limit_by_score=1.0, limit_by_protein_number=50
    ):
        pdb_entries = list(
            filter(lambda x: (x.eValue < limit_by_score), self.pfam_to_pdb_ids[pfam_id])
        )[:limit_by_protein_number]
        return pdb_entries
