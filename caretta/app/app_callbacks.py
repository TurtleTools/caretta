from pathlib import Path
from zipfile import ZipFile

import dash
import dash_bio
import dash_core_components as dcc
import numpy as np
import pyparsing

from caretta import multiple_alignment
from caretta.app import app_helper, app_layout
from flask import send_from_directory, abort


def register_callbacks(app, get_pdb_entries, suite):
    """
    Register callbacks to dash app

    Parameters
    ----------
    app
        Dash app
    get_pdb_entries
        function that takes a single user inputted string and returns a list of PdbEntry objects
    suite
        Fernet object
    """
    # Function called when user inputs folder / Pfam ID and clicks "Load Structures"
    @app.callback(
        # Output
        # PDB files / PDB IDs found in user-inputted folder / Pfam ID
        dash.dependencies.Output("proteins-selection-dropdown", "options"),
        # Input
        [
            # Whether the load-structures-button has been clicked
            dash.dependencies.Input("load-structures-button", "n_clicks")
        ],
        # State
        [
            # User input (PDB folder / Pfam ID)
            dash.dependencies.State("user-input", "value"),
        ],
    )
    def load_structures(load_structures_button, user_input):
        if load_structures_button and user_input:
            pdb_entries = get_pdb_entries(user_input)
            labels = []
            for x in pdb_entries:
                label = x.PDB_ID
                if x.CHAIN_ID != "none":
                    label += f".{x.CHAIN_ID}"
                if x.PdbResNumStart != -1 and x.PdbResNumEnd != -1:
                    label += f" {x.PdbResNumStart}-{x.PdbResNumEnd}"
                labels.append(label)
            return [
                {"label": label, "value": app_helper.compress_object(x, suite)}
                for label, x in zip(labels, pdb_entries)
            ]
        else:
            return [{"label": "no selection", "value": "None"}]

    # Function called when user selects IDs to align and clicks Align
    @app.callback(
        # Outputs
        [
            # hidden Divs
            # MSA class
            dash.dependencies.Output("caretta-class", "children"),
            # aligned sequences dict
            dash.dependencies.Output("sequence-alignment-data", "children"),
            # aligned feature dict
            dash.dependencies.Output("feature-alignment-data", "children"),
            # sequence alignment panel
            dash.dependencies.Output("sequence-alignment", "children"),
            # structure alignment panel
            dash.dependencies.Output("structure-alignment", "children"),
            # feature panel selection box
            dash.dependencies.Output("feature-selection-dropdown", "options"),
            # stop loading indicator
            dash.dependencies.Output("loading-indicator-output", "children"),
        ],
        # Inputs
        [
            # Whether the Align button was clicked
            dash.dependencies.Input("align-button", "n_clicks")
        ],
        # States
        [
            # User input (PDB folder / Pfam ID)
            dash.dependencies.State("user-input", "value"),
            # Selected PDB IDs
            dash.dependencies.State("proteins-selection-dropdown", "value"),
            # Penalties
            dash.dependencies.State("gap-open-dropdown", "value"),
            dash.dependencies.State("gap-extend-dropdown", "value"),
            # Unique ID for setting output folder
            dash.dependencies.State("unique-id", "children"),
        ],
    )
    def align_structures(
        align_button,
        user_input,
        proteins_selection_dropdown,
        gap_open_dropdown,
        gap_extend_dropdown,
        unique_id,
    ):
        if align_button and user_input and proteins_selection_dropdown:
            pdb_entries = [
                app_helper.decompress_object(x, suite)
                for x in proteins_selection_dropdown
            ]
            if not gap_open_dropdown:
                gap_open_dropdown = 1
            if not gap_extend_dropdown:
                gap_extend_dropdown = 0.01
            pdb_files = []
            for p in pdb_entries:
                try:
                    pdb_files.append(p.get_pdb()[1])
                except (OSError, AttributeError, pyparsing.ParseException):
                    continue
            msa_class = multiple_alignment.StructureMultiple.from_pdb_files(
                pdb_files,
                multiple_alignment.DEFAULT_SUPERPOSITION_PARAMETERS,
                output_folder=f"static/results_{app_helper.decompress_object(unique_id, suite)}",
            )
            if len(msa_class.structures) > 2:
                msa_class.make_pairwise_shape_matrix()
                sequence_alignment = msa_class.align(
                    gap_open_penalty=gap_open_dropdown,
                    gap_extend_penalty=gap_extend_dropdown,
                )
            else:
                sequence_alignment = msa_class.align(
                    gap_open_penalty=gap_open_dropdown,
                    gap_extend_penalty=gap_extend_dropdown,
                )
            msa_class.superpose()
            fasta = app_helper.to_fasta_str(sequence_alignment)
            dssp_dir = msa_class.output_folder / ".caretta_tmp"
            if not dssp_dir.exists():
                dssp_dir.mkdir()
            features = msa_class.get_aligned_features(
                dssp_dir, num_threads=4, only_dssp=False
            )
            caretta_class = app_helper.compress_object(msa_class, suite)
            sequence_alignment_data = app_helper.compress_object(
                sequence_alignment, suite
            )
            feature_alignment_data = app_helper.compress_object(features, suite)

            sequence_alignment_component = dash_bio.AlignmentChart(
                id="sequence-alignment-graph",
                data=fasta,
                showconsensus=False,
                showconservation=False,
                overview=None,
                height=300,
                colorscale="hydrophobicity",
            )
            structure_alignment_component = dcc.Graph(
                figure=app_helper.scatter3D(
                    {s.name: s.coordinates for s in msa_class.structures}
                ),
                id="scatter-plot",
            )
            feature_selection_dropdown = [{"label": x, "value": x} for x in features]
            loading_indicator_output = ""
            return (
                caretta_class,
                sequence_alignment_data,
                feature_alignment_data,
                sequence_alignment_component,
                structure_alignment_component,
                feature_selection_dropdown,
                loading_indicator_output,
            )
        else:
            return (
                app_helper.empty_object(suite),
                app_helper.empty_object(suite),
                app_helper.empty_object(suite),
                "",
                "",
                [{"label": "no alignment present", "value": "no alignment"}],
                "",
            )

    # Function that displays mean +/- stdev and heatmap of user-selected feature
    @app.callback(
        # Outputs
        [
            # Feature line panel
            dash.dependencies.Output("feature-line", "children"),
            # Feature heatmap panel
            dash.dependencies.Output("feature-heatmap", "children"),
        ],
        # Inputs
        [
            # Whether the display feature button has been clicked
            dash.dependencies.Input("display-feature-button", "n_clicks")
        ],
        # States
        [
            # Dropdown of feature names
            dash.dependencies.State("feature-selection-dropdown", "value"),
            # Aligned features dict
            dash.dependencies.State("feature-alignment-data", "children"),
        ],
    )
    def display_feature(
        display_feature_button_clicked,
        feature_selection_dropdown_value,
        feature_alignment_data,
    ):
        if (
            display_feature_button_clicked
            and feature_selection_dropdown_value
            and feature_alignment_data
        ):
            feature_alignment_dict = app_helper.decompress_object(
                feature_alignment_data, suite
            )
            chosen_feature_data = feature_alignment_dict[
                feature_selection_dropdown_value
            ]
            feature_line_component = dcc.Graph(
                figure=app_helper.line(chosen_feature_data), id="feature-line-graph"
            )
            feature_heatmap_component = dcc.Graph(
                figure=app_helper.heatmap(chosen_feature_data),
                id="feature-heatmap-graph",
            )
            return feature_line_component, feature_heatmap_component
        else:
            return (
                dcc.Graph(
                    figure=app_helper.empty_dict(),
                    id="feature-line-graph",
                    style={"display": "none"},
                ),
                dcc.Graph(
                    figure=app_helper.empty_dict(),
                    id="feature-heatmap-graph",
                    style={"display": "none"},
                ),
            )

    # Function that updated selected residues in structure alignment panel and feature alignment panel
    @app.callback(
        # Outputs
        [
            # hidden Divs
            # Residue position selected in structure alignment panel
            dash.dependencies.Output(
                "structure-alignment-selected-residue", "children"
            ),
            # Residue position selected in feature alignment panel
            dash.dependencies.Output("feature-alignment-selected-residue", "children"),
            # Feature line component
            dash.dependencies.Output("feature-line-graph", "figure"),
            # 3D scatter component
            dash.dependencies.Output("scatter-plot", "figure"),
        ],
        # Inputs
        [
            # Clicked indices in 3D scatter component
            dash.dependencies.Input("scatter-plot", "clickData"),
            # Clicked indices in feature line component
            dash.dependencies.Input("feature-line-graph", "clickData"),
        ],
        # States
        [
            # Feature line component
            dash.dependencies.State("feature-line-graph", "figure"),
            # 3D scatter component
            dash.dependencies.State("scatter-plot", "figure"),
            # Residue position selected in structure alignment panel
            dash.dependencies.State("structure-alignment-selected-residue", "children"),
            # Residue position selected in feature alignment panel
            dash.dependencies.State("feature-alignment-selected-residue", "children"),
            # Aligned sequences dict
            dash.dependencies.State("sequence-alignment-data", "children"),
        ],
    )
    def update_interactive_panels(
        scatter_plot_clickdata,
        feature_line_clickdata,
        feature_line_graph,
        scatter_plot,
        structure_alignment_selected_residue,
        feature_alignment_selected_residue,
        sequence_alignment_data,
    ):
        if feature_line_graph and scatter_plot:
            changed = None
            clickdata = None
            if (
                feature_line_clickdata
                and app_helper.compress_object(
                    (
                        feature_line_clickdata["points"][0]["pointNumber"],
                        feature_line_clickdata["points"][0]["curveNumber"],
                    ),
                    suite,
                )
                != feature_alignment_selected_residue
            ):
                clickdata = feature_line_clickdata
                changed = "feature-panel"
            elif (
                scatter_plot_clickdata
                and app_helper.compress_object(
                    (
                        scatter_plot_clickdata["points"][0]["pointNumber"],
                        scatter_plot_clickdata["points"][0]["curveNumber"],
                    ),
                    suite,
                )
                != structure_alignment_selected_residue
            ):
                clickdata = scatter_plot_clickdata
                changed = "structure-panel"
            if changed is not None and clickdata is not None:
                # Save new clicked index
                aln_index = clickdata["points"][0]["pointNumber"]
                protein_index = clickdata["points"][0]["curveNumber"]
                if changed == "feature-panel":
                    feature_alignment_selected_residue = app_helper.compress_object(
                        (aln_index, protein_index), suite
                    )
                elif changed == "structure-panel":
                    structure_alignment_selected_residue = app_helper.compress_object(
                        (aln_index, protein_index), suite
                    )

                sequence_alignment = app_helper.decompress_object(
                    sequence_alignment_data, suite
                )
                number_of_structures = len(sequence_alignment)

                try:
                    maxim, minim = (
                        np.max(feature_line_graph["data"][0]["y"]),
                        np.min(feature_line_graph["data"][0]["y"]),
                    )
                except KeyError:
                    return (
                        structure_alignment_selected_residue,
                        feature_alignment_selected_residue,
                        feature_line_graph,
                        scatter_plot,
                    )
                if len(feature_line_graph["data"]) > 2:
                    feature_line_graph["data"] = feature_line_graph["data"][:-1]
                if len(scatter_plot["data"]) > number_of_structures:
                    scatter_plot["data"] = scatter_plot["data"][:-1]

                if changed == "feature-panel":
                    aln_positions = app_helper.aln_index_to_protein(
                        aln_index, sequence_alignment
                    )
                    feature_line_graph["data"] += [
                        dict(
                            y=[minim, maxim],
                            x=[aln_index, aln_index],
                            type="scatter",
                            mode="lines",
                            name="selected residue",
                        )
                    ]

                    to_add = []
                    for i in range(len(scatter_plot["data"])):
                        p = aln_positions[scatter_plot["data"][i]["name"]]
                        if p is not None:
                            x, y, z = (
                                scatter_plot["data"][i]["x"][p],
                                scatter_plot["data"][i]["y"][p],
                                scatter_plot["data"][i]["z"][p],
                            )
                            to_add.append((x, y, z))
                        else:
                            continue
                    scatter_plot["data"] += [
                        dict(
                            x=[x[0] for x in to_add],
                            y=[y[1] for y in to_add],
                            z=[z[2] for z in to_add],
                            type="scatter3d",
                            mode="markers",
                            name="selected residues",
                        )
                    ]
                elif changed == "structure-panel":
                    aligned_sequence = list(sequence_alignment.values())[protein_index]
                    aln_index = app_helper.protein_to_aln_index(
                        aln_index, aligned_sequence
                    )
                    x, y, z = (
                        clickdata["points"][0]["x"],
                        clickdata["points"][0]["y"],
                        clickdata["points"][0]["z"],
                    )
                    feature_line_graph["data"] += [
                        dict(
                            y=[minim, maxim],
                            x=[aln_index, aln_index],
                            type="scatter",
                            mode="lines",
                            name="selected_residue",
                        )
                    ]
                    scatter_plot["data"] += [
                        dict(
                            y=[y],
                            x=[x],
                            z=[z],
                            type="scatter3d",
                            mode="markers",
                            name="selected residue",
                        )
                    ]
        return (
            structure_alignment_selected_residue,
            feature_alignment_selected_residue,
            feature_line_graph,
            scatter_plot,
        )

    # Function to export FASTA alignment file
    @app.callback(
        # Outputs
        # Link ot download file
        dash.dependencies.Output("fasta-download-link", "children"),
        # Inputs
        [
            # Whether the FASTA download button has been clicked
            dash.dependencies.Input("fasta-download-button", "n_clicks")
        ],
        # States
        [
            # Aligned sequences dict
            dash.dependencies.State("sequence-alignment-data", "children"),
            # MSA class
            dash.dependencies.State("caretta-class", "children"),
        ],
    )
    def download_alignment(
        fasta_download_button_clicked, sequence_alignment_data, caretta_class
    ):
        if fasta_download_button_clicked and sequence_alignment_data and caretta_class:
            sequence_alignment = app_helper.decompress_object(
                sequence_alignment_data, suite
            )
            if not sequence_alignment:
                return ""
            msa_class = app_helper.decompress_object(caretta_class, suite)
            msa_class.write_files(
                write_pdb=False,
                write_fasta=True,
                write_class=False,
                write_features=False,
                write_matrix=False,
            )
            return app_layout.get_download_string(
                str(Path(msa_class.output_folder) / "result.fasta")
            )
        else:
            return ""

    # Function to export superposed PDB files
    @app.callback(
        # Output
        # Link to download files
        dash.dependencies.Output("pdb-download-link", "children"),
        # Inputs
        [
            # Whether the PDB download button has been clicked
            dash.dependencies.Input("pdb-download-button", "n_clicks")
        ],
        # States
        [
            # Aligned sequences dict
            dash.dependencies.State("sequence-alignment-data", "children"),
            # MSA class
            dash.dependencies.State("caretta-class", "children"),
        ],
    )
    def download_pdb(
        pdb_download_button_clicked, sequence_alignment_data, caretta_class
    ):
        if pdb_download_button_clicked and sequence_alignment_data and caretta_class:
            sequence_alignment = app_helper.decompress_object(
                sequence_alignment_data, suite
            )
            if not sequence_alignment:
                return ""
            msa_class = app_helper.decompress_object(caretta_class, suite)

            msa_class.write_files(
                write_pdb=True,
                write_fasta=False,
                write_class=False,
                write_features=False,
                write_matrix=False,
            )
            output_filename = f"{msa_class.output_folder}/superposed_pdbs.zip"
            pdb_zip_file = ZipFile(output_filename, mode="w")
            for pdb_file in (Path(msa_class.output_folder) / "superposed_pdbs").glob(
                "*.pdb"
            ):
                pdb_zip_file.write(
                    str(pdb_file), arcname=f"{pdb_file.stem}{pdb_file.suffix}"
                )
            return app_layout.get_download_string(output_filename)
        else:
            return ""

    # Function to export feature files
    @app.callback(
        # Outputs
        [
            # Link to download feature files
            dash.dependencies.Output("feature-download-link", "children"),
            # # Div with export buttons
            dash.dependencies.Output("feature-exporter", "children"),
        ],
        # Inputs
        [
            # Whether the export feature button has been clicked
            dash.dependencies.Input("export-feature-button", "n_clicks"),
            # Whether the export all features button has been clicked
            dash.dependencies.Input("export-all-features-button", "n_clicks"),
        ],
        # States
        [
            # Selected value in the dropdown of feature names
            dash.dependencies.State("feature-selection-dropdown", "value"),
            # Aligned feature dict
            dash.dependencies.State("feature-alignment-data", "children"),
            # MSA class
            dash.dependencies.State("caretta-class", "children"),
        ],
    )
    def download_features(
        export_feature_button_clicked,
        export_all_features_button_clicked,
        feature_selection_dropdown_value,
        feature_alignment_data,
        caretta_class,
    ):
        output_string = ""
        if feature_alignment_data and caretta_class:
            feature_alignment_dict = app_helper.decompress_object(
                feature_alignment_data, suite
            )
            msa_class = app_helper.decompress_object(caretta_class, suite)
            protein_names = [s.name for s in msa_class.structures]
            if (
                export_feature_button_clicked and feature_selection_dropdown_value
            ) and not export_all_features_button_clicked:
                output_filename = f"{msa_class.output_folder}/{'-'.join(feature_selection_dropdown_value.split())}.csv"
                app_helper.write_feature_as_tsv(
                    feature_alignment_dict[feature_selection_dropdown_value],
                    protein_names,
                    output_filename,
                )
                output_string = app_layout.get_download_string(output_filename)
            elif (
                export_all_features_button_clicked and not export_feature_button_clicked
            ):
                output_filename = f"{msa_class.output_folder}/features.zip"
                features_zip_file = ZipFile(output_filename, mode="w")
                for feature in feature_alignment_dict:
                    feature_file = (
                        f"{msa_class.output_folder}/{'-'.join(feature.split())}.csv"
                    )
                    app_helper.write_feature_as_tsv(
                        feature_alignment_dict[feature], protein_names, feature_file
                    )
                    features_zip_file.write(
                        str(feature_file), arcname=f"{'-'.join(feature.split())}.csv"
                    )
                output_string = app_layout.get_download_string(output_filename)
        return output_string, app_layout.get_export_feature_buttons()

    @app.server.route("/caretta/static/<path:path>")
    def download(path):
        """Serve a file from the static directory."""
        try:
            return send_from_directory(str(Path.cwd() / "static"), path)
        except FileNotFoundError:
            abort(404)
