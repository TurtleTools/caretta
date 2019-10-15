import base64
import json
import os
import pickle

import dash
import dash_bio as dashbio
import dash_core_components as dcc
import dash_html_components as html
import flask
import numpy as np
import requests as rq
from dash_bio_utils import pdb_parser as parser

from caretta import helper
# from caretta_pfam.pfam import PfamToPDB, PdbEntry, superimpose, get_pdbs_from_folder, run_chosen_entries
from caretta.pfam import PfamToPDB, get_pdbs_from_folder

external_stylesheets = ["https://cdnjs.cloudflare.com/ajax/libs/skeleton/2.0.4/skeleton.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

model_data = rq.get(
    'https://raw.githubusercontent.com/plotly/dash-bio-docs-files/master/' +
    'mol3d/model_data.js'
).json()
styles_data = rq.get(
    'https://raw.githubusercontent.com/plotly/dash-bio-docs-files/master/' +
    'mol3d/styles_data.js'
).json()

shitty_model = json.loads(parser.create_data("./1AAC.pdb"))


def heatmap(core_features, zeros=False):
    if zeros:
        length = 2
        z = np.zeros((length, length))
        data = [dict(z=z, type="heatmap", showscale=False)]
        return {"data": data, "layout": {"margin": dict(l=25, r=25, t=25, b=25)}}
    else:
        length = len(core_features[list(core_features.keys())[0]])
        z = np.zeros((len(core_features), length))
        for i in range(len(core_features)):
            for j in range(length):
                z[i, j] = core_features[list(core_features.keys())[i]][j]
        data = [dict(z=z, type="heatmap", showscale=False)]
        return {"data": data, "layout": {"margin": dict(l=25, r=25, t=25, b=25)}}


def feature_line(core_features, alignment):
    length = len(core_features[list(core_features.keys())[0]])
    z = np.zeros((len(core_features), length))
    core_keys = list(core_features.keys())
    for i in range(len(core_features)):
        for j in range(length):
            if alignment[core_keys[i]][j] is not "-":
                z[i, j] = core_features[core_keys[i]][j]
            else:
                z[i, j] = np.NaN
    print(z)
    y = np.array([np.nanmean(z[:, x]) for x in range(z.shape[1])])
    y_med = np.array([np.nanmedian(z[:, x]) for x in range(z.shape[1])])
    y_se = np.array([np.nanstd(z[:, x]) / np.sqrt(z.shape[1]) for x in range(z.shape[1])])

    data = [dict(y=list(y + y_se) + list(y - y_se)[::-1], x=list(range(length)) + list(range(length))[::-1],
                 fillcolor="lightblue", fill="toself", type="scatter", mode="lines", name="Standard error",
                 line=dict(color='lightblue')),
            dict(y=y, x=np.arange(length), type="scatter", mode="lines", name="Mean",
                 line=dict(color='blue'))]
    return {"data": data, "layout": {"legend": dict(x=.5, y=1.2),
                                     "margin": dict(l=25, r=25, t=25, b=25)}}


def scatter3D(coord_dict):
    data = []
    for k, v in coord_dict.items():
        x, y, z = v[:, 0], v[:, 1], v[:, 2]
        data.append(dict(
            x=x,
            y=y,
            z=z,
            mode='lines',
            type='scatter3d',
            text=None,
            name=str(k),
            line=dict(
                width=3,
                opacity=0.8)))
    layout = {"margin": dict(l=20, r=20, t=20, b=20),
              'clickmode': 'event+select',
              'scene': {'xaxis': {"visible": False, "showgrid": False, "showline": False},
                        "yaxis": {"visible": False, "showgrid": False, "showline": False},
                        "zaxis": {"visible": False, "showgrid": False, "showline": False}}}
    return {"data": data, "layout": layout}


def check_gap(sequences, i):
    for seq in sequences:
        if seq[i] == "-":
            return True
    return False


def get_feature_z(features, alignments):
    core_indices = []
    sequences = list(alignments.values())
    for i in range(len(sequences[0])):
        if not check_gap(sequences, i):
            core_indices.append(i)
        else:
            continue
    return {x: features[x][np.arange(len(sequences[0]))] for x in features}


def write_as_csv(feature_dict, file_name):
    f = open(file_name, "w")
    print(feature_dict)
    for protein_name, features in feature_dict.items():
        f.write(";".join([protein_name] + [str(x) for x in list(features)]) + "\n")
    f.close()


def write_as_csv_all_features(feature_dict, file_name):
    f = open(file_name, "w")
    print(feature_dict)
    for protein_name, feature_dict in feature_dict.items():
        for feature_name, feature_values in feature_dict.items():
            f.write(";".join([protein_name, feature_name] + [str(x) for x in list(feature_values)]) + "\n")
    f.close()


def model_combine(files):
    model_data = {"atoms": [], "bonds": []}
    residue_index = 0
    atom_index = len(model_data["atoms"])
    for f in files:
        model = json.loads(parser.create_data(f))
        for atom in model["atoms"]:
            atom["residue_index"] += residue_index
            model_data["atoms"].append(atom)
        for bond in model["bonds"]:
            bond["atom2_index"] += atom_index
            bond["atom1_index"] += atom_index
            model_data["bonds"].append(bond)
        residue_index = model_data["atoms"][-1]["residue_index"]
        atom_index = len(model_data["atoms"])
    return model_data


print(model_data.keys())
print(model_data["atoms"][0].keys())
print(model_data["atoms"][:3])
print(model_data["bonds"][0].keys())
print(model_data["bonds"][0:3])
print("parser stuff:")
print(shitty_model.keys())
print(shitty_model["atoms"][0].keys())
print(shitty_model["atoms"][:3])
print(shitty_model["bonds"][0].keys())
print(shitty_model["bonds"][0:3])

box_style = {"box-shadow": "1px 3px 20px -4px rgba(0,0,0,0.75)",
             "border-radius": "5px", "background-color": "#f9f7f7"}

box_style_lg = {"top-margin": 25,
                "border-style": "solid",
                "border-color": "rgb(187, 187, 187)",
                "border-width": "1px",
                "border-radius": "5px",
                "background-color": "#edfdff"}

box_style_lr = {"top-margin": 25,
                "border-style": "solid",
                "border-color": "rgb(187, 187, 187)",
                "border-width": "1px",
                "border-radius": "5px",
                "background-color": "#ffbaba"}


def molecule_loader(pdb_filename):
    return dashbio.Molecule3dViewer(id="protein", modelData=json.loads(parser.create_data(pdb_filename)),
                                    styles=styles_data)


def structural_alignment_loader(file_names):
    model_data = model_combine(file_names)
    return dashbio.Molecule3dViewer(id="proteins", modelData=model_data)


def compress_object(raw_object):
    return base64.b64encode(pickle.dumps(raw_object, protocol=4)).decode("utf-8")


def decompress_object(compressed_object):
    return pickle.loads(base64.b64decode(compressed_object))


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


pfam_start = PfamToPDB(from_file=True, limit=50)
pfam_start = list(pfam_start.pfam_to_pdb_ids.keys())
pfam_start = [{"label": x, "value": x} for x in pfam_start]

# app.layout = html.Div([
#     dashbio.Molecule3dViewer(
#         id='my-dashbio-molecule3d',
#         modelData=shitty_model,
#         styles=styles_data
#     ),
#     "Selection data",
#     html.Hr(),
#     html.Div(id='molecule3d-output')
# ])

intro = """This is a demo webserver for Caretta. It generates multiple structure alignments for proteins from a selected 
Pfam domain and displays the alignment, the superposed proteins, and aligned structural features."""

pfam_selection = """
While the server is restricted to  a maximum of 50 proteins and 100 pfam domains, you can download this GUI from x and run it locally to use
 it on as many proteins as you'd like. All the generated data can further be exported for downstream use.

Choose: Please select a Pfam ID and click on Load Structures. You can then use the dropdown box to select which PDB IDs to align.

"""

app.layout = html.Div(children=[html.Div(html.Div([html.H1("Caretta - a multiple protein structure alignment and feature extraction tool",
                                                           style={"text-align": "center"}),
                                                   html.P(intro, style={"text-align": "center"})], className="row"),
                                         className="container"),
                                html.Div([html.Br(), html.P(children=compress_object(PfamToPDB(from_file=True,
                                                                                               limit=100)), id="pfam-class",
                                                            style={"display": "none"}),
                                          html.P(children="", id="feature-data",
                                                 style={"display": "none"}),
                                          html.P(children=compress_object(0), id="button1",
                                                 style={"display": "none"}),
                                          html.P(children=compress_object(0), id="button2",
                                                 style={"display": "none"}),
                                          html.P(children="", id="alignment-data",
                                                 style={"display": "none"}),
                                          html.Div([html.H3("Choose Pfam and Structures", className="row", style={"text-align": "center"}),
                                                    html.P(pfam_selection, className="row"),
                                                    html.Div([html.Div(dcc.Dropdown(placeholder="Choose Pfam ID",
                                                                                    options=pfam_start, id="pfam-ids"), className="four columns"),
                                                              html.Button("Load Structures", className="four columns", id="load-button"),
                                                              html.Div(
                                                                  dcc.Input(placeholder="Custom folder", value="", type="text", id="custom-folder"),
                                                                  className="four columns")], className="row"),
                                                    html.Div([html.Div(dcc.Dropdown(placeholder="Gap open penalty (1.0)",
                                                                                    options=[{"label": np.round(x, decimals=2), "value": x} for x in
                                                                                             np.arange(0, 5, 0.1)],
                                                                                    id="gap-open"), className="four columns"),
                                                              html.Div(dcc.Dropdown(multi=True, id="structure-selection"),
                                                                       className="four columns"),
                                                              html.Div(dcc.Dropdown(placeholder="Gap extend penalty (0.01)",
                                                                                    options=[{"label": np.round(x, decimals=3), "value": x} for x in
                                                                                             np.arange(0, 1, 0.002)],
                                                                                    id="gap-extend"),
                                                                       className="four columns")], className="row"),
                                                    html.Br(),
                                                    html.Div(html.Button("Align Pfam Structures", className="twelve columns", id="align"),
                                                             className="row")],
                                                   className="container"),
                                          html.Br()], className="container", style=box_style),
                                html.Br(),
                                html.Div(children=[html.Br(),
                                                   html.H3("Sequence alignment", className="row", style={"text-align": "center"}),
                                                   html.Div(
                                                       html.P("Sequence alignment is shown according to the structural alignment.", className="row"),
                                                       className="container"),
                                                   html.Div([html.Button("Download alignment", className="row", id="alignment-download"),
                                                             html.Div(children="", className="row", id="fasta-download-link")],
                                                            className="container"),
                                                   html.Div(html.P(id="alignment", className="twelve columns"), className="row")],
                                         className="container", style=box_style),
                                html.Br(),
                                html.Div([html.Br(),
                                          html.H3("Structural alignment", className="row", style={"text-align": "center"}),
                                          html.Div(html.P(
                                              "Selecting a residue on a structure will indicate its location on a chosen feature alignment in the next section",
                                              className="row"), className="container"),
                                          html.Div(html.Button("Download PDB", className="row", id="pdb", style={"align": "center"}),
                                                   className="container"),
                                          html.Div(children=dcc.Graph(figure=heatmap([[0, 0], [0, 0]], zeros=True), id="scatter3d"),
                                                   className="row", id="aligned-proteins"), html.Br()],
                                         className="container", style=box_style),
                                html.Br(),
                                html.Div(
                                    [html.Br(), html.Div([html.Div([html.H3("Feature alignment", className="row", style={"text-align": "center"}),
                                                                    html.P(
                                                                        "Selecting a residue position on the feature alignment will highlight the corresponding residues in all structures in the above 3d plot.",
                                                                        className="row"),
                                                                    dcc.Dropdown(placeholder="Choose a feature", id="feature-selection",
                                                                                 className="six columns"),
                                                                    html.Button("Display feature alignment", id="feature-button",
                                                                                className="six columns")], className="row"),
                                                          html.Div([html.Div([html.Button("Export feature", id="export"),
                                                                              html.Button("Export all features", id="export-all")], id="exporter"),
                                                                    html.Div(html.P(""), id="link-field"),
                                                                    html.Br()])], className="container"),

                                     html.Div(
                                         html.Div(dcc.Graph(figure=heatmap([[0, 0], [0, 0]], zeros=True), id="feature-line"), id="feature-plot1"),
                                         className="row"),
                                     html.Div(html.Div(dcc.Graph(figure=heatmap([[0, 0], [0, 0]], zeros=True), id="heatmap"), id="feature-plot2"),
                                              className="row")],
                                    className="container", style=box_style),
                                html.Br(), html.Br(), html.Div(id="testi")])


@app.callback(dash.dependencies.Output('fasta-download-link', 'children'),
              [dash.dependencies.Input('alignment-download', 'n_clicks')],
              [dash.dependencies.State("alignment-data", "children"),
               dash.dependencies.State("pfam-class", "children")])
def download_alignment(clicked, data, pfam_data):
    if clicked and data and pfam_data:
        alignment = decompress_object(data)
        if not alignment:
            return ""
        pfam_class = decompress_object(pfam_data)
        fasta = pfam_class.to_fasta_str(alignment)
        fnum = np.random.randint(0, 1000000000)
        fname = f"static/{fnum}.fasta"
        f = open(fname, "w")
        f.write(fasta)
        f.close()
        return html.A(f"Download %s here" % ("alignment" + ".fasta"), href="/%s" % fname)
    else:
        return ""


@app.callback(dash.dependencies.Output('structure-selection', 'options'),
              [dash.dependencies.Input('load-button', 'n_clicks')],
              [dash.dependencies.State("pfam-class", "children"),
               dash.dependencies.State("pfam-ids", "value"),
               dash.dependencies.State("custom-folder", "value")])
def show_selected_atoms(clicked, pfam_class, pfam_id, custom_folder):
    if clicked and pfam_class and pfam_id:
        pfam_class = decompress_object(pfam_class)
        pfam_structures = pfam_class.get_entries_for_pfam(pfam_id)
        print(pfam_structures)
        return [{"label": x.PDB_ID, "value": compress_object(x)} for x in pfam_structures]
    elif clicked and pfam_class and custom_folder:
        pdb_files = get_pdbs_from_folder(custom_folder)
        print(pdb_files)
        return [{"label": x.PDB_ID.split("/")[-1], "value": compress_object(x)} for x in pdb_files]
    else:
        return [{"label": "no selection", "value": "None"}]


@app.callback([dash.dependencies.Output("alignment", "children"),
               dash.dependencies.Output("aligned-proteins", "children"),
               dash.dependencies.Output("feature-data", "children"),
               dash.dependencies.Output("feature-selection", "options"),
               dash.dependencies.Output("alignment-data", "children")],
              [dash.dependencies.Input("align", "n_clicks")],
              [dash.dependencies.State("structure-selection", "value"),
               dash.dependencies.State("pfam-class", "children"),
               dash.dependencies.State("gap-open", "value"),
               dash.dependencies.State("gap-extend", "value")])
def align_structures(clicked, pdb_entries, pfam_class, gap_open, gap_extend):
    if clicked and pdb_entries and pfam_class:
        pfam_class = decompress_object(pfam_class)
        pdb_entries = [decompress_object(x) for x in pdb_entries]
        print("starting alignment")
        if gap_open and gap_extend:
            alignment, pdbs, features = pfam_class.multiple_structure_alignment_from_pfam(pdb_entries,
                                                                                          gap_open_penalty=gap_open,
                                                                                          gap_extend_penalty=gap_extend)
        else:
            alignment, pdbs, features = pfam_class.multiple_structure_alignment_from_pfam(pdb_entries)
        # aln_np = {k: helper.aligned_string_to_array(alignment[k]) for k in alignment}
        pfam_class.msa.superpose(alignment)
        # pdbs_sup = superimpose(pdbs, aln_np)
        # names = list(pdbs_sup.keys())
        # files = []

        # for n in names:
        #     files.append(f"test_{n}.pdb")
        #     pd.writePDB(f"test_{n}.pdb", pdbs_sup[n])
        print("ending alignment")
        fasta = pfam_class.to_fasta_str(alignment)
        component = dashbio.AlignmentChart(
            id='my-dashbio-alignmentchart',
            data=fasta, showconsensus=False, showconservation=False,
            overview=None, height=300,
            colorscale="hydrophobicity"
        )
        return component, dcc.Graph(figure=scatter3D({s.name: s.coords for s in pfam_class.msa.structures}), id="scatter3d"), compress_object(
            features), [{"label": x, "value": x} for x in features[list(features.keys())[0]]], compress_object(alignment)
    else:
        return "", "", compress_object(np.zeros(0)), [{"label": "no alignment present", "value": "no alignment"}], compress_object(np.zeros(0))


@app.callback([dash.dependencies.Output("feature-plot1", "children"),
               dash.dependencies.Output("feature-plot2", "children")],
              [dash.dependencies.Input("feature-button", "n_clicks")],
              [dash.dependencies.State("feature-selection", "value"),
               dash.dependencies.State("feature-data", "children"),
               dash.dependencies.State("alignment-data", "children")])
def display_feature(clicked, chosen_feature, feature_dict, aln):
    if clicked and chosen_feature and feature_dict:
        alignment = decompress_object(aln)
        feature_dict = decompress_object(feature_dict)
        chosen_feature_dict = {x: feature_dict[x][chosen_feature] for x in feature_dict}
        aln_np = {k: helper.aligned_string_to_array(alignment[k]) for k in alignment}
        chosen_feature_aln_dict = {x: helper.get_aligned_string_data(aln_np[x], chosen_feature_dict[x]) for x in feature_dict.keys()}
        z = get_feature_z(chosen_feature_aln_dict, alignment)
        component1 = dcc.Graph(figure=heatmap(z), id="heatmap")
        component2 = dcc.Graph(figure=feature_line(z, alignment), id="feature-line")
        return component2, component1
    else:
        return dcc.Graph(figure=heatmap([[0, 0], [0, 0]], zeros=True),
                         id="feature-line", style={"display": "none"}), dcc.Graph(figure=heatmap([[0, 0], [0, 0]], zeros=True),
                                                                                  id="heatmap", style={"display": "none"})


@app.callback([dash.dependencies.Output("link-field", "children"),
               dash.dependencies.Output("exporter", "children")],
              [dash.dependencies.Input("export", "n_clicks"),
               dash.dependencies.Input("export-all", "n_clicks")],
              [dash.dependencies.State("feature-selection", "value"),
               dash.dependencies.State("feature-data", "children"),
               dash.dependencies.State("alignment-data", "children")])
def write_output(clicked, clicked_all, chosen_feature, feature_dict, aln):
    if (clicked and chosen_feature and feature_dict and aln) and not clicked_all:
        alignment = decompress_object(aln)
        feature_dict = decompress_object(feature_dict)
        chosen_feature_dict = {x: feature_dict[x][chosen_feature] for x in feature_dict}
        aln_np = {k: helper.aligned_string_to_array(alignment[k]) for k in alignment}
        chosen_feature_aln_dict = {x: helper.get_aligned_string_data(aln_np[x], chosen_feature_dict[x]) for x in
                                   feature_dict.keys()}
        z = get_feature_z(chosen_feature_aln_dict, alignment)
        fnum = np.random.randint(0, 1000000000)
        fname = f"static/{fnum}.csv"
        write_as_csv(z, fname)
        return html.A(f"Download %s here" % (str(fnum) + ".csv"), href="/%s" % fname), [html.Button("Export feature", id="export"),
                                                                                        html.Button("Export all features", id="export-all")]
    elif (clicked_all and feature_dict and aln) and not clicked:
        feature_dict = decompress_object(feature_dict)
        fnum = np.random.randint(0, 1000000000)
        fname = f"static/{fnum}.csv"
        write_as_csv_all_features(feature_dict, fname)
        return html.A(f"Download %s here" % (str(fnum) + ".csv"), href="/%s" % fname), [html.Button("Export feature", id="export"),
                                                                                        html.Button("Export all features", id="export-all")]
    else:
        return "", [html.Button("Export feature", id="export"),
                    html.Button("Export all features", id="export-all")]


@app.server.route('/static/<path:path>')
def downlad_file(path):
    root_dir = os.getcwd()
    return flask.send_from_directory(
        os.path.join(root_dir, 'static'), path)


@app.callback([dash.dependencies.Output("feature-line", "figure"),
               dash.dependencies.Output("scatter3d", "figure"),
               dash.dependencies.Output("button1", "children"),
               dash.dependencies.Output("button2", "children")],
              [dash.dependencies.Input("scatter3d", "clickData"),
               dash.dependencies.Input("feature-line", "clickData")],
              [dash.dependencies.State("feature-line", "figure"),
               dash.dependencies.State("scatter3d", "figure"),
               dash.dependencies.State("button1", "children"),
               dash.dependencies.State("button2", "children"),
               dash.dependencies.State("alignment-data", "children")])
def update_features(clickdata_3d, clickdata_feature, feature_data, scatter3d_data, button1, button2, alignment_data):
    if feature_data and scatter3d_data and clickdata_feature and compress_object(
            (clickdata_feature["points"][0]["pointNumber"], clickdata_feature["points"][0]["curveNumber"])) != button1:
        alignment = decompress_object(alignment_data)
        number_of_structures = len(alignment)
        clickdata = clickdata_feature
        idx = clickdata["points"][0]["pointNumber"]
        protein_index = clickdata["points"][0]["curveNumber"]
        aln_positions = aln_index_to_protein(idx, alignment)
        button1 = compress_object((idx, protein_index))
        x, y = clickdata["points"][0]["x"], clickdata["points"][0]["y"]
        # try
        try:
            maxim, minim = np.max(feature_data["data"][0]["y"]), np.min(feature_data["data"][0]["y"])
        except KeyError:
            return feature_data, scatter3d_data, button1, button2
        if len(feature_data["data"]) > 2:
            feature_data["data"] = feature_data["data"][:-1]
        feature_data["data"] += [dict(y=[minim, maxim], x=[idx, idx], type="scatter", mode="lines",
                                      name="selected residue")]
        if len(scatter3d_data["data"]) > number_of_structures:
            scatter3d_data["data"] = scatter3d_data["data"][:-1]
        to_add = []
        for i in range(len(scatter3d_data["data"])):
            d = scatter3d_data["data"][i]
            k = d["name"]
            p = aln_positions[k]
            if p is not None:
                x, y, z = d["x"][p], d["y"][p], d["z"][p]
                to_add.append((x, y, z))
            else:
                print("None")
                continue
        scatter3d_data["data"] += [dict(x=[x[0] for x in to_add],
                                        y=[y[1] for y in to_add],
                                        z=[z[2] for z in to_add], type="scatter3d", mode="markers",
                                        name="selected residues")]
        return feature_data, scatter3d_data, button1, button2
    if feature_data and scatter3d_data and clickdata_3d and compress_object(
            (clickdata_3d["points"][0]["pointNumber"], clickdata_3d["points"][0]["curveNumber"])) != button2:
        alignment = decompress_object(alignment_data)
        number_of_structures = len(alignment)
        clickdata = clickdata_3d
        print(clickdata)
        idx = clickdata["points"][0]["pointNumber"]
        protein_index = clickdata["points"][0]["curveNumber"]
        button2 = compress_object((idx, protein_index))
        gapped_sequence = list(alignment.values())[protein_index]
        aln_index = protein_to_aln_index(idx, gapped_sequence)
        x, y, z = clickdata["points"][0]["x"], clickdata["points"][0]["y"], clickdata["points"][0]["z"]
        try:
            maxim, minim = np.max(feature_data["data"][0]["y"]), np.min(feature_data["data"][0]["y"])
        except KeyError:
            return feature_data, scatter3d_data, button1, button2
        if len(feature_data["data"]) > 2:
            feature_data["data"] = feature_data["data"][:-1]
        print(number_of_structures)
        feature_data["data"] += [dict(y=[minim, maxim], x=[aln_index, aln_index], type="scatter", mode="lines",
                                      name="selected_residue")]
        if len(scatter3d_data["data"]) > number_of_structures:
            scatter3d_data["data"] = scatter3d_data["data"][:-1]
        scatter3d_data["data"] += [dict(y=[y], x=[x], z=[z], type="scatter3d", mode="markers",
                                        name="selected residue")]
        return feature_data, scatter3d_data, button1, button2

    elif feature_data and scatter3d_data:
        return feature_data, scatter3d_data, button1, button2


if __name__ == '__main__':
    app.run_server(debug=True, host="0.0.0.0", port=3002)