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
import json as _json
import subprocess
import sys
import tempfile
import uuid

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

# Training process registry — Popen objects can't be serialised into dcc.Store
_TRAINING_PROCESSES: Dict[str, subprocess.Popen] = {}


# Flask endpoint: open native directory picker via a subprocess so tkinter
# runs in its own process (avoids main-thread requirement inside Flask).
@app.server.route('/api/browse-directory')
def _api_browse_directory():
    """Open a native directory-picker dialog and return the chosen path."""
    try:
        result = subprocess.run(
            [
                sys.executable, '-c',
                'import tkinter as tk; from tkinter import filedialog; '
                'root = tk.Tk(); root.withdraw(); '
                'root.attributes("-topmost", True); '
                'p = filedialog.askdirectory(title="Select workspace directory"); '
                'print(p, end="")',
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        return {'path': result.stdout.strip()}
    except Exception as exc:
        return {'path': '', 'error': str(exc)}


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
                                placeholder="experiment_workspace_v4",
                                type="text",
                                value=""  # Will be set by callback
                            ),
                            dbc.Button("Browse…", id="browse-workspace-button", color="secondary", outline=True),
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
                                                html.A('Select CSV or Parquet File')
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
                                            accept='.csv,.parquet',
                                            multiple=False
                                        ),
                                        html.Div(id='upload-status', className='mt-2')
                                    ])
                                ])
                            ], label="Upload CSV / Parquet"),
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
                                "GUI is optimized for 1–1 000 candidates with real-time scoring. "
                                "For >10 000 candidates, use the CLI: "
                                "python -m molfuse.cli.phase1 --config <config.json> --workspace <workspace>",
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
        
        # Train New Model section (collapsible)
        html.Hr(),
        dbc.Row([
            dbc.Col([
                dbc.Accordion([
                    dbc.AccordionItem([
                        html.P(
                            "Train a new MolFuSE model on your own data. "
                            "Provide paths to feature files — relative paths are "
                            "resolved from the directory where the GUI was started. "
                            "Training runs via the phase 1 CLI pipeline and the model "
                            "is loaded into the browser automatically when done.",
                            className='text-muted small mb-3',
                        ),
                        dbc.Row([
                            dbc.Col([
                                html.Label("Run Name *"),
                                dbc.Input(
                                    id="train-run-name",
                                    placeholder="my_ABL1_umap_5d",
                                    type="text",
                                ),
                            ], width=4),
                            dbc.Col([
                                html.Label("Target Accession *"),
                                dbc.Input(
                                    id="train-target-accession",
                                    placeholder="P00519",
                                    type="text",
                                ),
                            ], width=4),
                            dbc.Col([
                                html.Label("Workspace Output Path"),
                                dbc.InputGroup([
                                    dbc.Input(
                                        id="train-workspace-path",
                                        placeholder="my_workspace  (default: current workspace)",
                                        type="text",
                                    ),
                                    dbc.Button("Browse…", id="browse-train-workspace-button", color="secondary", outline=True),
                                ]),
                            ], width=4),
                        ], className='mb-2'),
                        dbc.Row([
                            dbc.Col([
                                html.Label("MF Features File * (CSV or Parquet)"),
                                dbc.Input(
                                    id="train-mf-path",
                                    placeholder="data/KW-0808_Transferase_affinity_extracted_features.parquet",
                                    type="text",
                                ),
                            ], width=12),
                        ], className='mb-2'),
                        dbc.Row([
                            dbc.Col([
                                html.Label("ZINC Features File * (CSV or Parquet)"),
                                dbc.Input(
                                    id="train-zinc-path",
                                    placeholder="data/zinc/zinc_acquirable_extracted_features.parquet",
                                    type="text",
                                ),
                            ], width=12),
                        ], className='mb-2'),
                        dbc.Row([
                            dbc.Col([
                                html.Label(
                                    "Actives File (CSV or Parquet, optional — "
                                    "enrichment metrics skipped if absent)"
                                ),
                                dbc.Input(
                                    id="train-actives-path",
                                    placeholder="data/chembl/P00519_actives_extracted_features.parquet",
                                    type="text",
                                ),
                            ], width=12),
                        ], className='mb-3'),
                        html.Hr(),
                        dbc.Row([
                            dbc.Col([
                                html.Label("Method"),
                                dbc.RadioItems(
                                    id="train-method",
                                    options=[
                                        {"label": "UMAP", "value": "umap"},
                                        {"label": "PCA", "value": "pca"},
                                    ],
                                    value="umap",
                                    inline=True,
                                ),
                            ], width=4),
                            dbc.Col([
                                html.Label("Dimensions"),
                                dbc.Input(
                                    id="train-dimensions",
                                    type="number",
                                    value=5,
                                    min=2,
                                    max=50,
                                ),
                            ], width=4),
                            dbc.Col([
                                html.Label("Affinity Cutoff (nM)"),
                                dbc.Input(
                                    id="train-affinity-cutoff",
                                    type="number",
                                    value=100000,
                                    min=1,
                                ),
                            ], width=4),
                        ], className='mb-3'),
                        dbc.Row([
                            dbc.Col([
                                dbc.Button(
                                    "Train Model",
                                    id="train-button",
                                    color="warning",
                                ),
                            ], width="auto"),
                            dbc.Col([
                                html.Div(id="train-status-message"),
                            ]),
                        ], className='mb-2'),
                        html.Div([
                            html.Label("Training Log:", className='mb-1 mt-2'),
                            dbc.Textarea(
                                id="training-log-output",
                                readOnly=True,
                                style={
                                    'height': '300px',
                                    'fontFamily': 'monospace',
                                    'fontSize': '0.8em',
                                    'whiteSpace': 'pre',
                                },
                                className='mb-2',
                            ),
                        ], id="training-log-section", style={'display': 'none'}),
                    ], title="Train New Model (on your own data)"),
                ], start_collapsed=True),
            ], width=12),
        ], className='mb-3'),

        # Hidden stores
        dcc.Store(id='model-artifacts-store', storage_type='memory'),
        dcc.Store(id='candidate-descriptors-store', storage_type='memory'),
        dcc.Store(id='scored-candidates-store', storage_type='memory'),
        dcc.Store(id='training-state-store', storage_type='memory'),
        dcc.Interval(
            id='training-poll-interval',
            interval=2000,
            n_intervals=0,
            disabled=True,
        ),

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
            return (
                dbc.Alert(
                    "No models found in this workspace. "
                    "Use the \u2018Train New Model\u2019 section below to train a model, "
                    "or download a pre-trained workspace from Zenodo.",
                    color="warning",
                ),
                [], None,
            )

        n_valid = sum(m['valid'] for m in models)
        if n_valid == 0:
            return (
                dbc.Alert(
                    f"Found {len(models)} runs but none have complete artifacts. "
                    "Use the \u2018Train New Model\u2019 section below to retrain.",
                    color="warning",
                ),
                [], None,
            )
        
        # Get KW categories
        kw_categories = MODEL_LOADER.get_kw_categories()
        if not kw_categories:
            return dbc.Alert("No valid models found (missing scaler or embedding files).", color="warning"), [], None
        
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
        # Parse CSV or Parquet
        try:
            content_type, content_string = csv_contents.split(',')
            decoded = base64.b64decode(content_string)
            fname = (csv_filename or "").lower()
            if fname.endswith('.parquet'):
                df_input = pd.read_parquet(BytesIO(decoded))
            else:
                df_input = pd.read_csv(StringIO(decoded.decode('utf-8')))
            
            if 'SMILES' not in df_input.columns:
                return None, dbc.Alert("File must contain a 'SMILES' column", color="danger")
            
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
            logger.error(f"Error parsing file: {e}")
            return None, dbc.Alert(f"Error parsing file: {e}", color="danger")
    
    elif triggered_id == "smiles-text-input" and smiles_text:
        # Parse text input
        smiles_list = [s.strip() for s in smiles_text.strip().split('\n') if s.strip()]
    
    else:
        return None, None
    
    if not smiles_list:
        return None, dbc.Alert("No valid SMILES provided", color="warning")
    
    if len(smiles_list) > 10000:
        return None, dbc.Alert(
            f"Too many candidates ({len(smiles_list)}). GUI supports up to 10 000. "
            "For larger batches use the CLI: "
            "python -m molfuse.cli.phase1 --config <config.json> --workspace <workspace>",
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


# ==================== Training Callbacks ====================

@app.callback(
    [Output("training-state-store", "data"),
     Output("training-poll-interval", "disabled"),
     Output("train-status-message", "children"),
     Output("training-log-section", "style"),
     Output("training-log-output", "value")],
    Input("train-button", "n_clicks"),
    [State("train-run-name", "value"),
     State("train-target-accession", "value"),
     State("train-mf-path", "value"),
     State("train-zinc-path", "value"),
     State("train-actives-path", "value"),
     State("train-workspace-path", "value"),
     State("train-method", "value"),
     State("train-dimensions", "value"),
     State("train-affinity-cutoff", "value"),
     State("workspace-path-input", "value")],
    prevent_initial_call=True,
)
def start_training(
    n_clicks,
    run_name, target_accession,
    mf_path, zinc_path, actives_path,
    workspace_out, method, dimensions, affinity_cutoff,
    current_workspace,
):
    """Validate inputs, write a phase1 config, and launch the training subprocess."""
    hidden = {'display': 'none'}
    visible = {'display': 'block'}

    # --- validation ---
    errors = []
    if not run_name or not run_name.strip():
        errors.append("Run name is required.")
    if not target_accession or not target_accession.strip():
        errors.append("Target accession is required.")
    # Resolve all file paths relative to the GUI's launch directory so that
    # relative paths work correctly when passed to the subprocess.
    cwd = Path.cwd()

    def _resolve(p: str) -> Path:
        """Expand ~ and resolve relative to the GUI launch directory."""
        return (cwd / Path(p).expanduser()).resolve()

    mf_resolved = _resolve(mf_path.strip()) if mf_path and mf_path.strip() else None
    zinc_resolved = _resolve(zinc_path.strip()) if zinc_path and zinc_path.strip() else None
    actives_resolved = (
        _resolve(actives_path.strip())
        if actives_path and actives_path.strip()
        else None
    )

    if not mf_path or not mf_path.strip():
        errors.append("MF features file path is required.")
    elif not mf_resolved.exists():
        errors.append(f"MF features file not found: {mf_resolved}")
    if not zinc_path or not zinc_path.strip():
        errors.append("ZINC features file path is required.")
    elif not zinc_resolved.exists():
        errors.append(f"ZINC features file not found: {zinc_resolved}")
    if actives_resolved is not None and not actives_resolved.exists():
        errors.append(f"Actives file not found: {actives_resolved}")

    if errors:
        return (
            None, True,
            dbc.Alert([html.P(e, className='mb-0') for e in errors], color="danger"),
            hidden, "",
        )

    # --- workspace (resolve relative to CWD) ---
    ws_raw = (workspace_out or "").strip() or (current_workspace or "").strip() or "./workspace"
    ws = str((cwd / Path(ws_raw).expanduser()).resolve())

    # --- config dict (always use resolved absolute paths so subprocess can find files) ---
    config: Dict = {
        "run_name": run_name.strip(),
        "target": target_accession.strip(),
        "method": method or "umap",
        "dim": int(dimensions) if dimensions else 5,
        "mf_features_csv": str(mf_resolved),
        "zinc_features_csv": str(zinc_resolved),
        "affinity_cutoff_nM": int(affinity_cutoff) if affinity_cutoff else 100000,
        "on_empty_cutoff": "fallback",
        "representation": "features",
    }
    if actives_resolved is not None:
        config["actives_features_csv"] = str(actives_resolved)
    if (method or "umap") == "umap":
        config["umap_params"] = {
            "n_neighbors": 15,
            "min_dist": 0.1,
            "metric": "euclidean",
            "random_state": 42,
        }

    # --- write config and launch ---
    process_key = str(uuid.uuid4())
    tmp_dir = Path(tempfile.gettempdir()) / "molfuse_gui_training"
    tmp_dir.mkdir(exist_ok=True)
    config_path = tmp_dir / f"{process_key}_config.json"
    log_path = tmp_dir / f"{process_key}.log"

    config_path.write_text(_json.dumps(config, indent=2))

    try:
        with open(log_path, 'w') as log_file:
            proc = subprocess.Popen(
                [sys.executable, '-m', 'molfuse.cli.phase1',
                 '--config', str(config_path),
                 '--workspace', ws],
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
        _TRAINING_PROCESSES[process_key] = proc
    except Exception as exc:
        logger.error(f"Failed to start training process: {exc}", exc_info=True)
        return (
            None, True,
            dbc.Alert(f"Failed to start training: {exc}", color="danger"),
            hidden, "",
        )

    state = {
        "process_key": process_key,
        "log_path": str(log_path),
        "workspace": ws,
        "status": "running",
    }
    return (
        state, False,
        dbc.Alert(
            f"Training started (PID {proc.pid}). Log updates every 2 s…",
            color="info",
        ),
        visible, "",
    )


@app.callback(
    [Output("training-log-output", "value", allow_duplicate=True),
     Output("training-poll-interval", "disabled", allow_duplicate=True),
     Output("train-status-message", "children", allow_duplicate=True),
     Output("workspace-scan-status", "children", allow_duplicate=True),
     Output("kw-category-dropdown", "options", allow_duplicate=True),
     Output("kw-category-dropdown", "value", allow_duplicate=True)],
    Input("training-poll-interval", "n_intervals"),
    State("training-state-store", "data"),
    prevent_initial_call=True,
)
def poll_training(n_intervals, state):
    """Stream log output and handle training completion."""
    no_update = dash.no_update

    if not state or state.get("status") != "running":
        return no_update, True, no_update, no_update, no_update, no_update

    process_key = state.get("process_key")
    log_path = state.get("log_path")
    proc = _TRAINING_PROCESSES.get(process_key)

    if proc is None:
        return (
            no_update, True,
            dbc.Alert("Training process not found.", color="danger"),
            no_update, no_update, no_update,
        )

    # Read current log content
    log_text = ""
    try:
        lp = Path(log_path)
        if lp.exists():
            log_text = lp.read_text(errors="replace")
    except Exception:
        pass

    exit_code = proc.poll()

    if exit_code is None:
        # Still running — update log, keep polling
        return log_text, False, no_update, no_update, no_update, no_update

    # Process finished — remove from registry
    del _TRAINING_PROCESSES[process_key]

    if exit_code == 0:
        # Rescan workspace so the new model appears in the model browser
        ws_path = state.get("workspace", "")
        kw_options, kw_value, scan_status = [], None, no_update
        try:
            global MODEL_LOADER, WORKSPACE_DIR
            ws_resolved = Path(ws_path).expanduser().resolve()
            MODEL_LOADER = ModelLoader(ws_resolved)
            models = MODEL_LOADER.scan_models(phases=["phase1", "phase4"])
            n_valid = sum(m["valid"] for m in models)
            kw_cats = MODEL_LOADER.get_kw_categories()
            kw_options = [{"label": kw, "value": kw} for kw in kw_cats]
            kw_value = kw_cats[0] if kw_cats else None
            WORKSPACE_DIR = ws_resolved
            scan_status = dbc.Alert(
                f"✓ Found {n_valid} valid models across {len(kw_cats)} KW categories",
                color="success",
            )
        except Exception as exc:
            scan_status = dbc.Alert(
                f"Training succeeded but workspace rescan failed: {exc}",
                color="warning",
            )

        return (
            log_text, True,
            dbc.Alert("✓ Training complete! Model loaded into browser.", color="success"),
            scan_status, kw_options, kw_value,
        )
    else:
        return (
            log_text, True,
            dbc.Alert(
                f"Training failed (exit code {exit_code}). See log above.",
                color="danger",
            ),
            no_update, no_update, no_update,
        )


# ==================== Browse Callbacks (client-side) ====================

app.clientside_callback(
    """
    async function(n_clicks) {
        if (!n_clicks) return window.dash_clientside.no_update;
        const resp = await fetch('/api/browse-directory');
        const data = await resp.json();
        return data.path || window.dash_clientside.no_update;
    }
    """,
    Output("workspace-path-input", "value", allow_duplicate=True),
    Input("browse-workspace-button", "n_clicks"),
    prevent_initial_call=True,
)

app.clientside_callback(
    """
    async function(n_clicks) {
        if (!n_clicks) return window.dash_clientside.no_update;
        const resp = await fetch('/api/browse-directory');
        const data = await resp.json();
        return data.path || window.dash_clientside.no_update;
    }
    """,
    Output("train-workspace-path", "value"),
    Input("browse-train-workspace-button", "n_clicks"),
    prevent_initial_call=True,
)


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
