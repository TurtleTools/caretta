import dash_core_components as dcc
import dash_html_components as html
import numpy as np
from caretta.app import app_helper
from pathlib import Path

box_style = {
    "box-shadow": "1px 3px 20px -4px rgba(0,0,0,0.75)",
    "border-radius": "5px",
    "background-color": "#f9f7f7",
}


def get_layout(
    introduction_text,
    input_text,
    placeholder_text,
    selection_text,
    suite,
    pfam_class=None,
):
    return html.Div(
        children=[
            get_introduction_panel(introduction_text),
            html.Br(),
            get_input_panel_layout(
                input_text, placeholder_text, selection_text, pfam_class
            ),
            html.Br(),
            get_hidden_variables_layout(suite),
            get_sequence_alignment_layout(),
            html.Br(),
            get_structure_alignment_layout(),
            html.Br(),
            get_feature_alignment_panel(),
            html.Br(),
        ]
    )


def get_introduction_panel(introduction_text: str):
    return html.Div(
        html.Div(
            [
                html.H1("Caretta", style={"text-align": "center"}),
                html.H3(
                    "a multiple protein structure alignment and feature extraction suite",
                    style={"text-align": "center"},
                ),
                html.P(dcc.Markdown(introduction_text), style={"text-align": "left"}),
            ],
            className="row",
        ),
        className="container",
    )


def get_hidden_variables_layout(suite):
    return html.Div(
        [
            html.P(children="", id="proteins-list", style={"display": "none"}),
            html.P(children="", id="feature-alignment-data", style={"display": "none"}),
            html.P(
                children=app_helper.compress_object(0, suite),
                id="structure-alignment-selected-residue",
                style={"display": "none"},
            ),
            html.P(
                children=app_helper.compress_object(0, suite),
                id="feature-alignment-selected-residue",
                style={"display": "none"},
            ),
            html.P(
                children="", id="sequence-alignment-data", style={"display": "none"}
            ),
            html.P(children="", id="caretta-class", style={"display": "none"}),
            html.P(
                children=app_helper.compress_object(
                    np.random.randint(0, 1000000000), suite
                ),
                id="unique-id",
                style={"display": "none"},
            ),
        ]
    )


def get_input_panel_layout(
    input_text: str,
    placeholder_text: str,
    selection_text: str,
    pfam_class: app_helper.PfamToPDB = None,
):
    if pfam_class is not None:
        user_input = dcc.Dropdown(
            placeholder="Choose Pfam ID",
            options=[{"label": x, "value": x} for x in pfam_class.pfam_to_pdb_ids],
            id="user-input",
        )

    else:
        user_input = (
            dcc.Textarea(
                placeholder=placeholder_text, value="", id="user-input", required=True,
            ),
        )
    return html.Div(
        [
            html.Div(
                [
                    html.Br(),
                    html.H3(
                        "Choose Structures",
                        className="row",
                        style={"text-align": "center"},
                    ),
                    html.P(input_text, className="row",),
                    html.Div(
                        [
                            html.Div(user_input, className="four columns",),
                            html.P(
                                dcc.Markdown(selection_text), className="four columns"
                            ),
                            html.Button(
                                "Load Structures",
                                className="four columns",
                                id="load-structures-button",
                            ),
                        ],
                        className="row",
                    ),
                    html.Div(
                        [
                            html.Div(
                                dcc.Dropdown(
                                    placeholder="Gap open penalty (1.0)",
                                    options=[
                                        {"label": np.round(x, decimals=2), "value": x,}
                                        for x in np.arange(0, 5, 0.1)
                                    ],
                                    id="gap-open-dropdown",
                                ),
                                className="four columns",
                            ),
                            html.Div(
                                dcc.Dropdown(
                                    multi=True,
                                    id="proteins-selection-dropdown",
                                    placeholder="Select PDB IDs to align",
                                ),
                                className="four columns",
                            ),
                            html.Div(
                                dcc.Dropdown(
                                    placeholder="Gap extend penalty (0.01)",
                                    options=[
                                        {"label": np.round(x, decimals=3), "value": x,}
                                        for x in np.arange(0, 1, 0.002)
                                    ],
                                    id="gap-extend-dropdown",
                                ),
                                className="four columns",
                            ),
                        ],
                        className="row",
                    ),
                    html.Br(),
                    html.Div(
                        html.Button(
                            "Align Structures",
                            className="twelve columns",
                            id="align-button",
                        ),
                        className="row",
                    ),
                    dcc.Loading(
                        id="loading-indicator",
                        children=[
                            html.Div(
                                id="loading-indicator-output",
                                style={"text-align": "center"},
                            )
                        ],
                        type="default",
                    ),
                    html.P(
                        id="time-estimate",
                        style={"text-align": "center"},
                        children="",
                        className="row",
                    ),
                ],
                className="container",
            ),
            html.Br(),
        ],
        className="container",
        style=box_style,
    )


def get_sequence_alignment_layout():
    return html.Div(
        children=[
            html.Br(),
            html.H3(
                "Sequence alignment", className="row", style={"text-align": "center"},
            ),
            html.Div(html.P("", className="row"), className="container"),
            html.Div(
                [
                    html.Button(
                        "Download sequence alignment",
                        className="twelve columns",
                        id="fasta-download-button",
                    ),
                    html.Div(children="", className="row", id="fasta-download-link"),
                ],
                className="container",
            ),
            html.Div(
                html.P(id="sequence-alignment", className="twelve columns"),
                className="row",
            ),
        ],
        className="container",
        style=box_style,
    )


def get_structure_alignment_layout():
    return html.Div(
        [
            html.Br(),
            html.H3(
                "Structural alignment", className="row", style={"text-align": "center"},
            ),
            html.Div(
                html.P(
                    "Click on a residue to see its position on the feature alignment in the next section.",
                    className="row",
                ),
                className="container",
            ),
            html.Div(
                [
                    html.Button(
                        "Download superposed PDBs",
                        id="pdb-download-button",
                        className="twelve columns",
                    ),
                    html.Div(children="", className="row", id="pdb-download-link"),
                ],
                className="container",
            ),
            html.Div(
                children=dcc.Graph(figure=app_helper.empty_dict(), id="scatter-plot",),
                className="row",
                id="structure-alignment",
            ),
            html.Br(),
        ],
        className="container",
        style=box_style,
    )


def get_feature_alignment_panel():
    return html.Div(
        [
            html.Br(),
            html.Div(
                [
                    html.Div(
                        [
                            html.H3(
                                "Feature alignment",
                                className="row",
                                style={"text-align": "center"},
                            ),
                            html.P(
                                "Click on a position in the feature alignment to see the corresponding residues in the previous section.",
                                className="row",
                            ),
                            dcc.Dropdown(
                                placeholder="Choose a feature",
                                id="feature-selection-dropdown",
                                className="six columns",
                            ),
                            html.Button(
                                "Display feature alignment",
                                id="display-feature-button",
                                className="six columns",
                            ),
                        ],
                        className="row",
                    ),
                    html.Br(),
                    html.Div(
                        [
                            html.Div(
                                get_export_feature_buttons(), id="feature-exporter",
                            ),
                            html.Div(html.P(""), id="feature-download-link"),
                            html.Br(),
                        ]
                    ),
                ],
                className="container",
            ),
            html.Div(
                html.Div(
                    dcc.Graph(figure=app_helper.empty_dict(), id="feature-line-graph",),
                    id="feature-line",
                    className="twelve columns",
                ),
                className="row",
            ),
            html.Div(
                html.Div(
                    dcc.Graph(
                        figure=app_helper.empty_dict(), id="feature-heatmap-graph",
                    ),
                    id="feature-heatmap",
                    className="twelve columns",
                ),
                className="row",
            ),
        ],
        className="container",
        style=box_style,
    )


def get_export_feature_buttons():
    return [
        html.Div(
            html.Button(
                "Download feature as tab-separated file",
                id="export-feature-button",
                className="twelve columns",
            ),
            className="row",
        ),
        html.Br(),
        html.Div(
            html.Button(
                "Download all features",
                id="export-all-features-button",
                className="twelve columns",
            ),
            className="row",
        ),
        html.Br(),
    ]


def get_download_string(filename):
    return html.Div(
        html.A(
            f"Download {Path(filename).stem}{Path(filename).suffix} here",
            href=f"/caretta/{filename}",
            className="twelve columns",
        ),
        className="container",
    )
