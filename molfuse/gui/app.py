"""Main Dash application for MolFuSE GUI."""

from __future__ import annotations

import base64
from io import BytesIO, StringIO
import logging
from pathlib import Path
from typing import Dict, List, Optional

import dash
from dash import dcc, html, Input, Output, State, ALL, ctx
from dash import dcc as dcc_module  # For send_data_frame
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np

from molfuse.gui.components import ModelLoader, DescriptorCalculator, CandidateProjector, Visualizer


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Initialize Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)
app.title = "MolFuSE Candidate Scorer"


# Global state (will be populated by callbacks)
WORKSPACE_DIR = None
MODEL_LOADER = None
CURRENT_MODEL = None
CURRENT_ARTIFACTS = None


def create_layout():
    """Create main GUI layout."""
    return dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H1("MolFuSE Candidate Scorer", className='text-center mb-3 mt-3'),
                html.P(
                    "Production tool for scoring candidate molecules against trained MolFuSE models",
                    className='text-center text-muted mb-4'
                )
            ])
        ]),
        
        # Workspace selection
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("1. Select Workspace", className='mb-3'),
                        dbc.InputGroup([
                            dbc.InputGroupText("Workspace Path:"),
                            dbc.Input(
                                id="workspace-path-input",
                                placeholder="/path/to/experiment_workspace_v4",
                                type="text",
                                value=""  # Will be set by callback
                            ),
                            dbc.Button("Scan", id="scan-workspace-button", color="primary")
                        ], className='mb-2'),
                        html.Div(id="workspace-scan-status", className='mt-2')
                    ])
                ])
            ], width=12)
        ], className='mb-3'),
        
        # Model selection
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("2. Select Model", className='mb-3'),
                        dbc.Row([
                            dbc.Col([
                                html.Label("Molecular Function (KW):"),
                                dcc.Dropdown(
                                    id="kw-category-dropdown",
                                    placeholder="Select KW category...",
                                    clearable=False
                                )
                            ], width=6),
                            dbc.Col([
                                html.Label("Model Run:"),
                                dcc.Dropdown(
                                    id="model-run-dropdown",
                                    placeholder="Select model run...",
                                    clearable=False
                                )
                            ], width=6)
                        ]),
                        html.Div(id="model-info-display", className='mt-3')
                    ])
                ])
            ], width=12)
        ], className='mb-3'),
        
        # Candidate input
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("3. Input Candidates", className='mb-3'),
                        dbc.Tabs([
                            dbc.Tab([
                                dbc.Row([
                                    dbc.Col([
                                        dcc.Upload(
                                            id='candidate-csv-upload',
                                            children=html.Div([
                                                'Drag and Drop or ',
                                                html.A('Select CSV File')
                                            ]),
                                            style={
                                                'width': '100%',
                                                'height': '80px',
                                                'lineHeight': '80px',
                                                'borderWidth': '2px',
                                                'borderStyle': 'dashed',
                                                'borderRadius': '5px',
                                                'textAlign': 'center',
                                                'margin': '10px'
                                            },
                                            multiple=False
                                        ),
                                        html.Div(id='upload-status', className='mt-2')
                                    ])
                                ])
                            ], label="Upload CSV"),
                            dbc.Tab([
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("Enter SMILES (one per line):"),
                                        dbc.Textarea(
                                            id="smiles-text-input",
                                            placeholder="CC(C)Cc1ccc(cc1)C(C)C(O)=O\nCCOc1ccc2nc(sc2c1)S(=O)(=O)N",
                                            style={'width': '100%', 'height': '150px'},
                                            className='mb-2'
                                        )
                                    ])
                                ])
                            ], label="Text Input")
                        ]),
                        html.Div([
                            html.Label("Performance Note:", className='mt-3 text-muted small'),
                            html.P(
                                "GUI is optimized for 1-1000 candidates with real-time scoring. "
                                "For >10k candidates, use CLI: python -m molfuse.cli.score --help",
                                className='text-muted small'
                            )
                        ])
                    ])
                ])
            ], width=12)
        ], className='mb-3'),
        
        # Score button
        dbc.Row([
            dbc.Col([
                dbc.Button(
                    "Score Candidates",
                    id="score-button",
                    color="success",
                    size="lg",
                    className='w-100',
                    disabled=True
                )
            ], width={"size": 6, "offset": 3})
        ], className='mb-3'),
        
        # Results section
        dbc.Row([
            dbc.Col([
                html.Div(id="results-section", children=[
                    # Will be populated after scoring
                ])
            ])
        ]),
        
        # Hidden stores
        dcc.Store(id='model-artifacts-store', storage_type='memory'),
        dcc.Store(id='candidate-descriptors-store', storage_type='memory'),
        dcc.Store(id='scored-candidates-store', storage_type='memory')
        
    ], fluid=True)


app.layout = create_layout()


# ==================== Callbacks ====================

@app.callback(
    Output("workspace-path-input", "value"),
    Input("workspace-path-input", "id"),  # Triggers on page load
    prevent_initial_call=False
)
def set_initial_workspace(_):
    """Set workspace path from command line argument on page load."""
    if WORKSPACE_DIR:
        return str(WORKSPACE_DIR)
    return ""


@app.callback(
    [Output("workspace-scan-status", "children"),
     Output("kw-category-dropdown", "options"),
     Output("kw-category-dropdown", "value")],
    Input("scan-workspace-button", "n_clicks"),
    State("workspace-path-input", "value"),
    prevent_initial_call=True
)
def scan_workspace(n_clicks, workspace_path):
    """Scan workspace and populate KW category dropdown."""
    global WORKSPACE_DIR, MODEL_LOADER
    
    if not workspace_path:
        return dbc.Alert("Please enter workspace path", color="warning"), [], None
    
    workspace_path = Path(workspace_path).expanduser().resolve()
    
    if not workspace_path.exists():
        return dbc.Alert(f"Workspace not found: {workspace_path}", color="danger"), [], None
    
    try:
        MODEL_LOADER = ModelLoader(workspace_path)
        models = MODEL_LOADER.scan_models(phases=["phase1", "phase4"])
        
        if not models:
            return dbc.Alert("No models found in workspace", color="warning"), [], None
        
        n_valid = sum(m['valid'] for m in models)
        if n_valid == 0:
            return dbc.Alert(f"Found {len(models)} models but none are valid (missing artifacts)", color="warning"), [], None
        
        # Get KW categories
        kw_categories = MODEL_LOADER.get_kw_categories()
        if not kw_categories:
            return dbc.Alert(f"Found {n_valid} valid models but couldn't extract KW categories", color="warning"), [], None
        
        kw_options = [{"label": kw, "value": kw} for kw in kw_categories]
        
        WORKSPACE_DIR = workspace_path
        
        return (
            dbc.Alert(f"✓ Found {n_valid} valid models across {len(kw_categories)} KW categories", color="success"),
            kw_options,
            kw_categories[0]  # Auto-select first category
        )
    
    except Exception as e:
        logger.error(f"Error scanning workspace: {e}", exc_info=True)
        return dbc.Alert(f"Error scanning workspace: {e}", color="danger"), [], None


@app.callback(
    [Output("model-run-dropdown", "options"),
     Output("model-run-dropdown", "value")],
    Input("kw-category-dropdown", "value"),
    prevent_initial_call=True
)
def update_model_dropdown(selected_kw):
    """Update model dropdown based on selected KW category."""
    global MODEL_LOADER
    
    if not selected_kw or MODEL_LOADER is None:
        return [], None
    
    models = MODEL_LOADER.get_models_by_kw(selected_kw)
    
    if not models:
        return [], None
    
    # Create options with metadata
    options = []
    for m in models:
        if not m['valid']:
            continue
        
        label = f"{m['run_name']} | {m['method'].upper()} {m['n_components']}D | EF@1%: {m['ef_at_1']:.2f}"
        options.append({"label": label, "value": m['run_name']})
    
    # Auto-select best model by EF@1%
    best_model = max(models, key=lambda m: m['ef_at_1'] if not np.isnan(m['ef_at_1']) else -np.inf)
    
    return options, best_model['run_name']


@app.callback(
    [Output("model-info-display", "children"),
     Output("model-artifacts-store", "data"),
     Output("score-button", "disabled")],
    Input("model-run-dropdown", "value"),
    prevent_initial_call=True
)
def load_model(run_name):
    """Load selected model artifacts."""
    global MODEL_LOADER, CURRENT_MODEL, CURRENT_ARTIFACTS
    
    if not run_name or MODEL_LOADER is None:
        return html.P("No model selected", className='text-muted'), None, True
    
    try:
        artifacts = MODEL_LOADER.load_model_artifacts(run_name)
        
        if artifacts is None:
            return dbc.Alert("Failed to load model artifacts", color="danger"), None, True
        
        CURRENT_MODEL = run_name
        CURRENT_ARTIFACTS = artifacts
        
        # Display model info
        meta = artifacts['metadata']
        metrics = artifacts['metrics']
        
        info_card = dbc.Card([
            dbc.CardBody([
                html.H6("Model Details:", className='mb-2'),
                html.P([
                    html.Strong("Target: "), meta['target_accession'], html.Br(),
                    html.Strong("Method: "), f"{meta['method'].upper()} ({meta['n_components']}D)", html.Br(),
                    html.Strong("Representation: "), meta['representation'].capitalize(), html.Br(),
                    html.Strong("EF@1%: "), f"{meta['ef_at_1']:.3f}" if not np.isnan(meta['ef_at_1']) else "N/A", html.Br(),
                    html.Strong("ROC-AUC: "), f"{metrics.get('roc_auc', np.nan):.3f}" if not np.isnan(metrics.get('roc_auc', np.nan)) else "N/A", html.Br(),
                    html.Strong("MF Cloud Size: "), f"{len(artifacts['embedding_mf'])} molecules"
                ], className='small mb-0')
            ])
        ], color="light")
        
        # Store serializable artifacts metadata (convert Path to string)
        meta_serializable = {k: str(v) if isinstance(v, Path) else v for k, v in meta.items()}
        artifacts_data = {
            "run_name": run_name,
            "metadata": meta_serializable
        }
        
        return info_card, artifacts_data, False  # Enable score button
    
    except Exception as e:
        logger.error(f"Error loading model: {e}", exc_info=True)
        return dbc.Alert(f"Error loading model: {e}", color="danger"), None, True


@app.callback(
    [Output("candidate-descriptors-store", "data"),
     Output("upload-status", "children")],
    [Input("candidate-csv-upload", "contents"),
     Input("smiles-text-input", "value")],
    [State("candidate-csv-upload", "filename"),
     State("model-artifacts-store", "data")],
    prevent_initial_call=True
)
def process_candidate_input(csv_contents, smiles_text, csv_filename, artifacts_data):
    """Process candidate input and compute descriptors."""
    global CURRENT_ARTIFACTS
    
    if CURRENT_ARTIFACTS is None:
        return None, dbc.Alert("Please select a model first", color="warning")
    
    triggered_id = ctx.triggered_id
    
    # Determine input source
    smiles_list = []
    
    if triggered_id == "candidate-csv-upload" and csv_contents:
        # Parse CSV
        try:
            content_type, content_string = csv_contents.split(',')
            decoded = base64.b64decode(content_string)
            df_input = pd.read_csv(StringIO(decoded.decode('utf-8')))
            
            if 'SMILES' not in df_input.columns:
                return None, dbc.Alert("CSV must contain 'SMILES' column", color="danger")
            
            # Check if CSV already has pre-computed features
            feature_names = MODEL_LOADER.get_feature_names_from_scaler(CURRENT_ARTIFACTS['scaler'])
            has_precomputed = all(feat in df_input.columns for feat in feature_names) if feature_names else False
            
            if has_precomputed:
                logger.info("CSV contains pre-computed features; skipping descriptor calculation")
                # Use pre-computed features directly
                required_cols = ['SMILES'] + list(feature_names)
                df_descriptors = df_input[required_cols].copy()
                
                status_msg = html.Div([
                    html.P(f"✓ Loaded {len(df_descriptors)} candidates with pre-computed features"),
                    html.P(f"✓ Features: {len(feature_names)} descriptors")
                ])
                
                descriptors_data = df_descriptors.to_dict('records')
                return descriptors_data, dbc.Alert(status_msg, color="success")
            
            # Otherwise compute descriptors from SMILES
            smiles_list = df_input['SMILES'].astype(str).tolist()
            
        except Exception as e:
            logger.error(f"Error parsing CSV: {e}")
            return None, dbc.Alert(f"Error parsing CSV: {e}", color="danger")
    
    elif triggered_id == "smiles-text-input" and smiles_text:
        # Parse text input
        smiles_list = [s.strip() for s in smiles_text.strip().split('\n') if s.strip()]
    
    else:
        return None, None
    
    if not smiles_list:
        return None, dbc.Alert("No valid SMILES provided", color="warning")
    
    if len(smiles_list) > 10000:
        return None, dbc.Alert(
            f"Too many candidates ({len(smiles_list)}). GUI supports up to 10,000. "
            "For larger batches, use CLI: python -m molfuse.cli.score",
            color="warning"
        )
    
    # Extract feature names from scaler
    feature_names = MODEL_LOADER.get_feature_names_from_scaler(CURRENT_ARTIFACTS['scaler'])
    
    # Compute descriptors
    try:
        desc_calc = DescriptorCalculator(feature_names=feature_names)
        df_descriptors = desc_calc.compute_for_smiles_list(smiles_list, include_smiles=True)
        
        # Check for failures
        n_failed = df_descriptors[desc_calc.feature_names].isna().all(axis=1).sum()
        n_partial_nan = df_descriptors[desc_calc.feature_names].isna().any(axis=1).sum()
        
        status_parts = [
            f"✓ Computed descriptors for {len(df_descriptors)} candidates"
        ]
        
        if n_failed > 0:
            status_parts.append(f"⚠ {n_failed} failed to parse")
        
        if n_partial_nan > n_failed:
            status_parts.append(f"⚠ {n_partial_nan - n_failed} have partial NaN values")
        
        status_msg = html.Div([html.P(part) for part in status_parts])
        
        # Store descriptors
        descriptors_data = df_descriptors.to_dict('records')
        
        return descriptors_data, dbc.Alert(status_msg, color="success" if n_failed == 0 else "warning")
    
    except Exception as e:
        logger.error(f"Error computing descriptors: {e}", exc_info=True)
        return None, dbc.Alert(f"Error computing descriptors: {e}", color="danger")


@app.callback(
    [Output("scored-candidates-store", "data"),
     Output("results-section", "children")],
    Input("score-button", "n_clicks"),
    [State("candidate-descriptors-store", "data"),
     State("model-artifacts-store", "data")],
    prevent_initial_call=True
)
def score_candidates(n_clicks, descriptors_data, artifacts_data):
    """Score candidates using loaded model."""
    global CURRENT_ARTIFACTS
    
    if not descriptors_data or CURRENT_ARTIFACTS is None:
        return None, dbc.Alert("No candidates to score or model not loaded", color="warning")
    
    try:
        # Reconstruct DataFrame
        df_descriptors = pd.DataFrame(descriptors_data)
        
        # Project and score
        projector = CandidateProjector(CURRENT_ARTIFACTS)
        df_scored = projector.project_and_score(df_descriptors, smiles_col="SMILES")
        
        if len(df_scored) == 0:
            return None, dbc.Alert("No valid candidates after projection", color="warning")
        
        # Create visualizations
        visualizer = Visualizer()
        
        # 2D scatter plot
        coord_cols = [col for col in df_scored.columns if col.startswith('z')]
        projection_method = "pca" if len(coord_cols) > 2 else "first2"
        
        fig_scatter = visualizer.create_embedding_scatter(
            df_mf=CURRENT_ARTIFACTS['embedding_mf'],
            df_zinc=CURRENT_ARTIFACTS.get('embedding_zinc'),
            df_actives=CURRENT_ARTIFACTS.get('embedding_actives'),
            df_candidates=df_scored,
            coord_cols=coord_cols,
            projection_method=projection_method,
            title=f"Candidates in {CURRENT_ARTIFACTS['metadata']['molecular_function']} Space"
        )
        
        # Score distribution
        fig_dist = visualizer.create_score_distribution(
            df_candidates=df_scored,
            title="Candidate Score Distribution"
        )
        
        # Top candidates table
        df_top = df_scored.head(20)
        table_rows = []
        
        for _, row in df_top.iterrows():
            img_src = visualizer.mol_to_image_base64(row['SMILES'], size=(150, 150))
            
            table_rows.append(
                html.Tr([
                    html.Td(int(row['rank'])),
                    html.Td(f"{float(row['score']):.4f}"),
                    html.Td(f"{float(row['distance']):.4f}"),
                    html.Td(html.Img(src=img_src, style={'height': '100px'}) if img_src else "N/A"),
                    html.Td(row['SMILES'], style={'font-size': '0.8em', 'max-width': '300px', 'word-break': 'break-all'}),
                    html.Td([
                        html.Div(f"SMILES: {row['nearest_mf_smiles']}", style={'font-size': '0.8em'}),
                        html.Div(f"Activity: {row['nearest_mf_activity_type']} = {float(row['nearest_mf_activity_value_nM']):.1f} nM" if pd.notna(row['nearest_mf_activity_value_nM']) else "N/A", style={'font-size': '0.8em'}),
                        html.Div(f"Target: {row['nearest_mf_accession']}", style={'font-size': '0.8em'})
                    ])
                ])
            )
        
        top_table = dbc.Table(
            [
                html.Thead(html.Tr([
                    html.Th("Rank"),
                    html.Th("Score"),
                    html.Th("Distance"),
                    html.Th("Structure"),
                    html.Th("SMILES"),
                    html.Th("Nearest MF Neighbor")
                ])),
                html.Tbody(table_rows)
            ],
            bordered=True,
            hover=True,
            striped=True,
            className='mt-3'
        )
        
        # Summary stats
        summary_card = dbc.Card([
            dbc.CardBody([
                html.H5("Scoring Summary", className='mb-3'),
                html.P([
                    html.Strong("Total Candidates Scored: "), f"{len(df_scored)}", html.Br(),
                    html.Strong("Score Range: "), f"{df_scored['score'].min():.4f} to {df_scored['score'].max():.4f}", html.Br(),
                    html.Strong("Mean Distance to MF: "), f"{df_scored['distance'].mean():.4f}", html.Br(),
                    html.Strong("Median Distance to MF: "), f"{df_scored['distance'].median():.4f}"
                ])
            ])
        ], color="light", className='mb-3')
        
        # Export button
        export_section = dbc.Row([
            dbc.Col([
                dbc.Button(
                    "Download Scored Candidates CSV",
                    id="download-button",
                    color="primary",
                    className='mt-3 mb-3'
                ),
                dcc.Download(id="download-csv")
            ])
        ])
        
        # Assemble results
        results = html.Div([
            html.Hr(),
            html.H3("Results", className='mb-3'),
            summary_card,
            dbc.Row([
                dbc.Col([dcc.Graph(figure=fig_scatter)], width=12)
            ]),
            dbc.Row([
                dbc.Col([dcc.Graph(figure=fig_dist)], width=12)
            ]),
            html.H4("Top 20 Candidates", className='mt-4 mb-3'),
            top_table,
            export_section
        ])
        
        # Store scored data
        scored_data = df_scored.to_dict('records')
        
        return scored_data, results
    
    except Exception as e:
        logger.error(f"Error scoring candidates: {e}", exc_info=True)
        return None, dbc.Alert(f"Error scoring candidates: {e}", color="danger")


@app.callback(
    Output("download-csv", "data"),
    Input("download-button", "n_clicks"),
    State("scored-candidates-store", "data"),
    prevent_initial_call=True
)
def download_results(n_clicks, scored_data):
    """Download scored candidates as CSV."""
    if not scored_data:
        return None
    
    df = pd.DataFrame(scored_data)
    return dcc_module.send_data_frame(df.to_csv, "molfuse_scored_candidates.csv", index=False)


# ==================== Main Entry Point ====================

def main(workspace_dir: Optional[str] = None, host: str = "127.0.0.1", port: int = 8050, debug: bool = True):
    """
    Launch MolFuSE GUI.
    
    Args:
        workspace_dir: Optional workspace directory path (can be set via GUI)
        host: Host IP address
        port: Port number
        debug: Enable debug mode
    """
    global WORKSPACE_DIR
    
    if workspace_dir:
        WORKSPACE_DIR = Path(workspace_dir).expanduser().resolve()
        logger.info(f"Pre-configured workspace: {WORKSPACE_DIR}")
    
    logger.info(f"Starting MolFuSE GUI on http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MolFuSE GUI: Production candidate scoring tool")
    parser.add_argument("--workspace", type=str, help="Workspace directory path (optional)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host IP (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8050, help="Port number (default: 8050)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    main(workspace_dir=args.workspace, host=args.host, port=args.port, debug=args.debug)
