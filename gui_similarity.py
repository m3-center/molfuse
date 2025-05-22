import os
import requests
import base64
from io import BytesIO
import glob
from compress_pickle import load
import time # For cache timing debug if needed

# Data science libraries
import numpy as np
from scipy.spatial import cKDTree
import pandas as pd
import plotly.graph_objects as go

# Chemistry libraries
from rdkit import Chem
from rdkit.Chem import Draw

# GUI libraries
import dash
from dash import dcc, html, Input, Output, State, ALL, ctx # Import ctx
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from dash_extensions.enrich import DashProxy, MultiplexerTransform
from flask_caching import Cache

# ------------------ Global Variables ------------------
similarity_dimensions = [
    'PCA-1', 'PCA-2',
    'UMAP-Euclidian-1', 'UMAP-Euclidian-2',
    'UMAP-Cosine-1', 'UMAP-Cosine-2',
    'UMAP-Manhattan-1', 'UMAP-Manhattan-2',
    'UMAP-Hamming-1', 'UMAP-Hamming-2'
]
all_dimensions = similarity_dimensions

activity_types = ['IC50', 'Ki', 'Kd', 'EC50']

# Predefined list of features for 'features' data type
# This list must match the features used to train the scaler and PCA models for 'features' data.
RDKIT_FEATURES_LIST = [
    'DipoleMoment','ABC','nAcid','nBase','nAromAtom','nAtom','nH','nC','nN','nO','nS',
    'nP','nX','nBonds','nBondsO','nBondsS','nBondsD','nBondsT','nBondsA','nBondsM',
    'nBondsKS','nBondsKD','EState_VSA7','nHBAcc','nHBDon','Lipinski','apol','bpol',
    'nRing','n3Ring','n4Ring','n5Ring','n6Ring','n7Ring','n8Ring','nRot','Diameter',
    'TopoShapeIndex','Vabc','MW'
]


# ------------------ Dash Application Setup ------------------
app = DashProxy(__name__,
                transforms=[MultiplexerTransform()],
                external_stylesheets=[dbc.themes.BOOTSTRAP],
                suppress_callback_exceptions=True)
app.title = "UMMBAS Molecular Similarity Explorer"

cache = Cache(app.server, config={
    'CACHE_TYPE': 'SimpleCache',
    'CACHE_DEFAULT_TIMEOUT': 3600 # Cache for 1 hour
})

app.layout = dbc.Container([
    
    dbc.Row([
        dbc.Col(
            html.H1("UMMBAS Molecular Similarity Explorer",
                    className='text-center mb-4'),
            width=12
        )
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Upload(
                id='custom-features-upload',
                children=html.Div(
                    ['Drag and Drop or ', html.A('Upload Custom Features CSV')]),
                style={'width': '100%', 'height': '60px', 'lineHeight': '60px',
                       'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                       'textAlign': 'center', 'margin': '10px'},
                multiple=False
            )
        ], width=6),
        dbc.Col([
            dcc.Upload(
                id='custom-fingerprints-upload',
                children=html.Div(['Drag and Drop or ', html.A(
                    'Upload Custom Fingerprints CSV')]),
                style={'width': '100%', 'height': '60px', 'lineHeight': '60px',
                       'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                       'textAlign': 'center', 'margin': '10px'},
                multiple=False
            )
        ], width=6)
    ]),
    dbc.Row([
        dbc.Col(
            dbc.Button("Project Custom Data", id="project-custom-data-button",
                       color="info", className="mt-2 mb-3 w-100"),
            width={"size": 6, "offset": 3} # Centered button
        )
    ]),

    dbc.Row([
        dbc.Col([
            html.Label("Enter Protein Uniprot ID:"),
            dbc.Input(id="uniprot-id-input",
                      placeholder="e.g., P35580", type="text"),
        ], width=4),
        dbc.Col([
            dbc.Button("Fetch Protein Data", id="fetch-protein-button",
                       color="primary", className="mt-2")
        ], width=4),
        dbc.Col([
            html.Label("Select Molecular Function:"),
            dcc.Dropdown(id="molecular-function-dropdown", options=[],
                         placeholder="Select molecular function", clearable=False)
        ], width=4)
    ], style={'marginBottom': '20px'}),
    html.Hr(),
    dbc.Row([
        dbc.Col([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Label("Select X-Axis:",
                                   className='font-weight-bold'),
                        dcc.Dropdown(id='x-axis-dropdown',
                                     options=[{'label': dim, 'value': dim}
                                              for dim in all_dimensions],
                                     value='PCA-1', clearable=False, style={'width': '100%'})
                    ]),
                ], md=3),
                dbc.Col([
                    html.Div([
                        html.Label("Select Y-Axis:",
                                   className='font-weight-bold'),
                        dcc.Dropdown(id='y-axis-dropdown',
                                     options=[{'label': dim, 'value': dim}
                                              for dim in all_dimensions],
                                     value='PCA-2', clearable=False, style={'width': '100%'})
                    ]),
                ], md=3),
                dbc.Col([
                    html.Label("Select Data Type:"),
                    dcc.RadioItems(
                        id='data-type-radio',
                        options=[{'label': 'Features', 'value': 'features'},
                                 {'label': 'Fingerprints', 'value': 'fingerprints'}],
                        value='features',
                        labelStyle={'display': 'inline-block',
                                    'margin-right': '10px'}
                    )
                ], md=3)
            ], className='mb-4'),
            dcc.Graph(id='similarity-space-graph', style={'height': '80vh'})
        ], width=8),
        dbc.Col([
            html.Div([
                html.H5("Selected Molecules"),
                html.Div(id='selected-molecules-container',
                         style={'overflowY': 'auto', 'maxHeight': '80vh'})
            ])
        ], width=2),
        dbc.Col([
            html.Div([
                html.H5("Activity Threshold Filters"),
                *[html.Div([
                    html.Label(f"{atype} [nM]:"),
                    dcc.RangeSlider(id={'type': 'slider', 'index': atype},
                                    min=1, max=5, step=0.25, value=[1, 3.5], # Log10 scale
                                    marks={0.1: '1', 1: '10', 2: '100', # Values are log10
                                           3: '1k', 4: '10k', 5: '100k'},
                                    tooltip={"placement": "bottom", "always_visible": False}),
                    html.Div(id={'type': 'slider-value', 'index': atype}, # Optional display
                             style={'textAlign': 'center', 'marginTop': '5px'}),
                    html.Br()
                ], style={'marginBottom': '20px'}) for atype in activity_types]
            ], style={'marginBottom': '20px'}),
            html.Div([
                dbc.Form([dbc.Checklist(options=[{"label": "Show ZINC Compounds", "value": "SHOW_ZINC"}],
                                        value=[], id="zinc-visibility-checkbox", switch=True)])
            ], style={'marginBottom': '20px'}),
            html.Div([
                dbc.Form([dbc.Checklist(options=[{"label": "Show non-halogenated compounds", "value": "SHOW_NON_HALOGENATED"}],
                                        value=[], id="nonhalogenated-checkbox", switch=True)])
            ], style={'marginBottom': '20px'}),
            html.Div([
                dbc.InputGroup([dbc.InputGroupText("Number of Best/Worst Fits:"), dbc.Input(id='num-bestfits', type='number', min=1, placeholder="20", value=20)],
               className='mb-3'),
                dbc.Button("CUSTOM 2 Any-Target", id='export-bestfits-button-any', color="success", className='mb-3 w-100'),
                dbc.Button("CUSTOM 2 Selected Target", id='export-bestfits-button-target', color="success", className='mb-3 w-100'),
                dbc.Button("ZINC 2 Any-Target", id='export-zinc-button-any', color="info", className='mb-3 w-100'),
                dbc.Button("ZINC 2 Selected Target", id='export-zinc-button-target', color="info", className='mb-3 w-100'),
                dbc.Button("CUSTOM Worst 2 Any-Target", id='export-worstfits-button-any', color="warning", className='mb-3 w-100'),
                dbc.Button("CUSTOM Worst 2 Selected Target", id='export-worstfits-button-target', color="warning", className='mb-3 w-100'),
                dbc.Button("ZINC Worst 2 Any-Target", id='export-worst-zinc-button-any', color="secondary", className='mb-3 w-100'),
                dbc.Button("ZINC Worst 2 Selected Target", id='export-worst-zinc-button-target', color="secondary", className='mb-3 w-100'),
                dbc.InputGroup([dbc.InputGroupText("Filename:"), dbc.Input(id='save-selection-filename', placeholder="Enter filename", value="selected_molecules.csv")],
                            className='mb-3'),
                dbc.Button("Download Selection", id='save-selection-button', color="primary", className='mb-3 w-100'),
                dcc.Download(id="download-selection")
            ], style={'marginBottom': '0px'}),
        ], width=2, style={'borderLeft': '1px solid #ccc', 'paddingLeft': '20px'})
    ]),
    dcc.Store(id='selected-molecule-ids', storage_type='memory'),
    dcc.Store(id='similar-protein-ids', storage_type='memory'),
    dcc.Store(id='target-organism-store', storage_type='memory'),
    dcc.Store(id='target-motor-protein-dropdown', storage_type='memory'),
    # New stores for custom data handling
    dcc.Store(id='uploaded-custom-features-df-store', storage_type='memory'),
    dcc.Store(id='uploaded-custom-fingerprints-df-store', storage_type='memory'),
    dcc.Store(id='projected-custom-data-store', storage_type='memory'),
], fluid=True)

# ------------------ Helper Functions ------------------

def mol_to_image_base64(smiles, size=(200, 200)):
    if not smiles or pd.isna(smiles): return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    try:
        Chem.rdDepictor.Compute2DCoords(mol)
        img = Draw.MolToImage(mol, size=size)
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{encoded}"
    except Exception as e:
        print(f"Error generating image for SMILES {smiles}: {e}")
        return None

# ------------------ Data Loading and Caching ------------------

def load_similarity_file(data_type, molecular_function):
    # ... (This function remains largely unchanged, responsible for loading base CSV)
    print(f"Attempting to load base file for: type={data_type}, function='{molecular_function}'")
    if not molecular_function:
        raise ValueError("Molecular function cannot be empty")
    safe_function_name = molecular_function.replace(' ', '_').replace('/', '_') # Sanitize

    if data_type == 'features':
        pattern = os.path.join("datasets", "molecular_function_similarity_spaces",
                               f"*_{safe_function_name}_affinity_extracted_features_similarity_space.csv")
    elif data_type == 'fingerprints':
        pattern = os.path.join("datasets", "molecular_function_similarity_spaces",
                               f"*_{safe_function_name}_affinity_extracted_fingerprints_ECFP4_similarity_space.csv")
    else:
        raise ValueError(f"Invalid data type: {data_type}")

    files = glob.glob(pattern)
    if not files:
        print(f"Search pattern: {pattern}")
        raise FileNotFoundError(
            f"No base similarity space file found for function '{molecular_function}' (searched as '{safe_function_name}') with data type '{data_type}'")

    try:
        file_path = files[0]
        print(f"Loading base data from: {file_path}")
        start_time = time.time()
        df = pd.read_csv(file_path, low_memory=False)
        load_end_time = time.time()
        print(f"Initial load of {len(df)} rows from {file_path} took {load_end_time - start_time:.2f} seconds.")
        original_length = len(df) 

        required_cols = ['MOLECULE ID', 'SMILES', 'Activity Type', 'Standard Value (nM)', 'accession']
        for col in required_cols:
             if col not in df.columns:
                 print(f"Warning: Required column '{col}' not found in {file_path}. Adding column with NAs.")
                 df[col] = pd.NA
        
        coordinate_cols = [d for d in all_dimensions if d in df.columns]
        if not coordinate_cols:
             print("Warning: No standard coordinate columns found in the file. Plotting might fail.")
        else:
             required_cols.extend(coordinate_cols)

        df['MOLECULE ID'] = df['MOLECULE ID'].astype(str).str.strip().str.upper()
        df['Is CUSTOM'] = False 
        df['Is ZINC'] = df['MOLECULE ID'].str.startswith('ZINC', na=False)

        if 'SMILES' in df.columns:
             df['SMILES'] = df['SMILES'].astype(str) 
             df['Is Halogenated'] = df['SMILES'].str.contains(r'Cl|Br|I(?![nr])|F(?![elrm])', regex=True, na=False)
        else:
            print(f"Warning: SMILES column missing in {file_path}. Cannot calculate 'Is Halogenated'.")
            df['Is Halogenated'] = False 

        activity_value_col = 'Standard Value (nM)'
        activity_type_col = 'Activity Type'
        if activity_value_col in df.columns:
             df[activity_value_col] = pd.to_numeric(df[activity_value_col], errors='coerce')
        if activity_type_col in df.columns:
             df[activity_type_col] = df[activity_type_col].astype(str).fillna('Unknown') 

        for dim in all_dimensions:
            if dim in df.columns:
                df[dim] = pd.to_numeric(df[dim], errors='coerce')

        print(f"Processing dropna. Initial rows: {len(df)}")
        if coordinate_cols:
             rows_before_coord_drop = len(df)
             df.dropna(subset=coordinate_cols, how='all', inplace=True)
             rows_after_coord_drop = len(df)
             if rows_after_coord_drop < rows_before_coord_drop:
                  print(f"Dropped {rows_before_coord_drop - rows_after_coord_drop} rows missing ALL coordinate data ({coordinate_cols}).")
        else:
             print("Skipping coordinate dropna as no coordinate columns were identified.")

        activity_cols_to_check = [col for col in [activity_value_col, activity_type_col] if col in df.columns]
        if activity_cols_to_check:
             non_zinc_non_custom_mask = ~df['Is ZINC'] & ~df['Is CUSTOM']
             drop_activity_mask = df[non_zinc_non_custom_mask][activity_cols_to_check].isna().any(axis=1)
             indices_to_drop = df[non_zinc_non_custom_mask][drop_activity_mask].index
             if not indices_to_drop.empty:
                 df.drop(indices_to_drop, inplace=True)
                 print(f"Dropped {len(indices_to_drop)} non-ZINC/non-CUSTOM rows missing activity data ({activity_cols_to_check}).")
             else:
                 print("No non-ZINC/non-CUSTOM rows dropped due to missing activity data.")
        else:
             print("Skipping activity dropna for non-ZINC/non-CUSTOM as activity columns are missing.")
        
        process_end_time = time.time()
        print(f"Finished processing base data. Final shape: {df.shape} (Original: {original_length})")
        print(f"Processing took {process_end_time - load_end_time:.2f} seconds.")
        return df

    except Exception as e:
        print(f"Error loading or processing file {files[0] if files else 'N/A'}: {e}")
        return pd.DataFrame()

@cache.memoize()
def get_base_similarity_dataframe(data_type, molecular_function):
    print(f"CACHE: Checking cache for base data: type={data_type}, function='{molecular_function}'")
    try:
        df = load_similarity_file(data_type, molecular_function)
        print(f"CACHE: Loaded/Computed base DataFrame for {molecular_function}/{data_type}. Size: {len(df)}")
        return df
    except FileNotFoundError as e:
        print(f"CACHE: File not found error: {e}")
        return pd.DataFrame() 
    except ValueError as e:
        print(f"CACHE: Value error: {e}")
        return pd.DataFrame()

def project_custom_data(raw_custom_df, data_type, molecular_function, simspace_dim=2):
    """Project a raw custom dataset into similarity space using saved scaler and models."""
    print(f"Projecting raw custom data: type={data_type}, function='{molecular_function}'")
    if not isinstance(raw_custom_df, pd.DataFrame) or raw_custom_df.empty:
        print("Raw custom data DataFrame is empty or invalid, skipping projection.")
        return pd.DataFrame()

    df_projected = raw_custom_df.copy() # Work on a copy

    # Define non-descriptor columns
    INFO_COLUMNS = ['MOLECULE ID', 'SMILES', 'ZINC_ID', 'LABEL', 'MANUFACTURER', 'TRANCHE', 'Is CUSTOM', 'Is ZINC', 'Is Halogenated']
    ACTIVITY_COLUMNS = ['Target ChEMBL ID', 'Target Name', 'Activity Type', 'Standard Value (nM)', 'target_chembl_id', 'accession']
    non_descriptor_set = set(INFO_COLUMNS + ACTIVITY_COLUMNS)

    # Identify features for scaling and projection
    if data_type == 'features':
        features_for_scaling_projection = RDKIT_FEATURES_LIST
        missing_defined_features = [f for f in features_for_scaling_projection if f not in df_projected.columns]
        if missing_defined_features:
            print(f"Warning: Custom 'features' data missing defined descriptor columns: {missing_defined_features}. Cannot project.")
            # Add empty projection columns so structure matches, but data will be NaN
            for dim in all_dimensions: df_projected[dim] = np.nan
            return df_projected
    elif data_type == 'fingerprints':
        features_for_scaling_projection = [col for col in df_projected.columns if col not in non_descriptor_set and not col.startswith(('PCA-', 'UMAP-'))]
        if not features_for_scaling_projection:
            print("Warning: No descriptor columns found in custom 'fingerprints' data. Cannot project.")
            for dim in all_dimensions: df_projected[dim] = np.nan
            return df_projected
    else:
        raise ValueError(f"Invalid data type for projection: {data_type}")

    print(f"Using features for scaling/projection: {features_for_scaling_projection[:5]}... (total {len(features_for_scaling_projection)})")

    # Ensure descriptor columns are numeric, coerce errors
    df_projected[features_for_scaling_projection] = df_projected[features_for_scaling_projection].apply(pd.to_numeric, errors='coerce')

    # --- Handle NaNs by Dropping Rows with any NaN in descriptor columns ---
    # But skip this for the fingerprints..
    original_len = len(df_projected)
    df_projected.dropna(subset=features_for_scaling_projection, how='any', inplace=True)

    if len(df_projected) < original_len:
        print(f"Dropped {original_len - len(df_projected)} custom rows with NaN descriptor values in columns: {features_for_scaling_projection}.")
    
    if df_projected.empty:
        print("Custom data DataFrame is empty after dropping NaN descriptors.")
        return pd.DataFrame()

    X_descriptors = df_projected[features_for_scaling_projection].values

    # --- Model Path Setup ---
    safe_function_name = molecular_function.replace(' ', '_').replace('/', '_')
    name_root_pattern_base = ""
    if data_type == 'features':
        name_root_pattern_base = f"*_{safe_function_name}_affinity_extracted_features"
    elif data_type == 'fingerprints':
        name_root_pattern_base = f"*_{safe_function_name}_affinity_extracted_fingerprints_ECFP4"
    
    sim_space_pattern = os.path.join("datasets", "molecular_function_similarity_spaces", f"{name_root_pattern_base}_similarity_space.csv")
    sim_files = glob.glob(sim_space_pattern)
    if not sim_files:
        raise FileNotFoundError(f"Cannot find base similarity space CSV to determine model names for function '{molecular_function}' (pattern: {sim_space_pattern})")
    
    base_name_csv = os.path.basename(sim_files[0])
    name_root = base_name_csv.replace("_similarity_space.csv", "")
    model_dir = os.path.join("datasets", "molecular_function_similarity_spaces", "models")
    print(f"Using model root name: {name_root} from dir: {model_dir}")

    X_scaled = X_descriptors # Initialize, will be updated by scaler

    # --- Load and Apply Scaler ---
    scaler_filename_part = "scaler.lzma" if data_type == 'features' else "scaler.lzma"
    scaler_model_path = os.path.join(model_dir, f"{name_root}_{scaler_filename_part}")
    
    if not os.path.exists(scaler_model_path):
        print(f"Warning: Scaler model not found at {scaler_model_path}. Proceeding without scaling.")
    else:
        try:
            with open(scaler_model_path, "rb") as f:
                scaler = load(f)
            print(f"Scaler model loaded from {scaler_model_path}")
            
            # Check if scaler expects specific feature names (optional, depends on scaler type)
            if hasattr(scaler, 'feature_names_in_') and data_type == 'features':
                scaler_expected_features = scaler.feature_names_in_
                if not all(f in features_for_scaling_projection for f in scaler_expected_features) or \
                   not all(f in scaler_expected_features for f in features_for_scaling_projection): # Check both ways if order matters
                    # This case should be handled by RDKIT_FEATURES_LIST check earlier for features
                    print(f"Warning: Mismatch between custom data features and scaler's expected features. This might lead to errors or incorrect scaling.")
            
            X_scaled = scaler.transform(X_descriptors)
            print(f"Applied scaler. Scaled data shape: {X_scaled.shape}")
        except Exception as e:
            print(f"Error applying scaler model: {e}. Proceeding with unscaled data for subsequent models.")
            # Fallback to X_descriptors if scaling fails
            X_scaled = X_descriptors


    # --- Load and Apply PCA Model ---
    pca_model_path = os.path.join(model_dir, f"{name_root}_PCA.lzma")
    if not os.path.exists(pca_model_path):
        print(f"Warning: PCA model not found at {pca_model_path}. PCA dimensions will be NaN.")
        for i in range(simspace_dim): df_projected[f'PCA-{i+1}'] = np.nan
    else:
        try:
            with open(pca_model_path, "rb") as f: pca = load(f)
            print(f"PCA model loaded from {pca_model_path}")

            # Validate PCA input features if model has feature_names_in_
            # if hasattr(pca, 'feature_names_in_'):
            #     pca_expected_features = pca.feature_names_in_
            #     if len(pca_expected_features) != X_scaled.shape[1]:
            #          raise ValueError(f"PCA model expects {len(pca_expected_features)} features, but scaled data has {X_scaled.shape[1]} features.")
            #     # If data_type == 'features', RDKIT_FEATURES_LIST should align with pca_expected_features.
            #     # For fingerprints, it's assumed the number of fingerprint bits matches.

            pca_res = pca.transform(X_scaled) # Apply PCA to SCALED data
            print(f"Transformed PCA data shape: {pca_res.shape}")
            for i in range(min(simspace_dim, pca_res.shape[1])):
                df_projected[f'PCA-{i+1}'] = pca_res[:, i]
        except Exception as e:
            print(f"Error applying PCA model: {e}")
            for i in range(simspace_dim): df_projected[f'PCA-{i+1}'] = np.nan

    # --- Load and Apply UMAP Models ---
    umap_metrics = [('euclidean', 'UMAP-Euclidian'), ('hamming', 'UMAP-Hamming')] # ('cosine', 'UMAP-Cosine'), ('manhattan', 'UMAP-Manhattan')
    for metric, label_prefix in umap_metrics:
        umap_model_path = os.path.join(model_dir, f"{name_root}_{metric}_UMAP.lzma")
        if not os.path.exists(umap_model_path):
            print(f"Warning: UMAP model for metric '{metric}' not found at {umap_model_path}. Skipping.")
            for i in range(simspace_dim): df_projected[f'{label_prefix}-{i+1}'] = np.nan
            continue
        try:
            with open(umap_model_path, "rb") as f: umap_model = load(f)
            
            # UMAP is also applied to SCALED data
            umap_res = umap_model.transform(X_scaled)
            for i in range(min(simspace_dim, umap_res.shape[1])):
                df_projected[f'{label_prefix}-{i+1}'] = umap_res[:, i]

        except Exception as e:
            print(f"Error applying UMAP model for metric '{metric}': {e}")
            for i in range(simspace_dim): df_projected[f'{label_prefix}-{i+1}'] = np.nan
    
    # Ensure all expected dimension columns are present, even if projection failed for some
    for dim_col in all_dimensions:
        if dim_col not in df_projected.columns:
            df_projected[dim_col] = np.nan
            print(f"Added missing dimension column '{dim_col}' with NaNs to projected custom data.")

    print(f"Finished projecting custom data. Final shape: {df_projected.shape}")
    return df_projected


def parse_uploaded_csv(contents, filename):
    """Decodes, parses an uploaded CSV, and performs basic cleaning."""
    if contents is None:
        return None

    print(f"Parsing uploaded CSV: {filename}")
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    try:
        df = pd.read_csv(BytesIO(decoded), low_memory=False)
        print(f"Parsed uploaded CSV '{filename}'. Shape: {df.shape}")
        if df.empty:
             print(f"Uploaded CSV '{filename}' is empty.")
             return pd.DataFrame()

        # --- Basic Cleaning and Standard Column Setup ---
        if 'MOLECULE ID' not in df.columns:
            # Try to find a common ID column or generate one
            if 'ID' in df.columns: df.rename(columns={'ID': 'MOLECULE ID'}, inplace=True)
            elif 'Name' in df.columns: df.rename(columns={'Name': 'MOLECULE ID'}, inplace=True)
            else: df['MOLECULE ID'] = "CUSTOM_" + df.index.astype(str)
        df['MOLECULE ID'] = "CUSTOM_" + df['MOLECULE ID'].astype(str).fillna('UnknownID').str.strip().str.upper()
        
        if 'SMILES' not in df.columns:
            if 'smiles' in df.columns: df.rename(columns={'smiles': 'SMILES'}, inplace=True)
            else: df['SMILES'] = '' # Add empty SMILES column
        df['SMILES'] = df['SMILES'].astype(str)

        df['Is CUSTOM'] = True
        df['Is ZINC'] = False # Custom is not ZINC by definition here
        df['Is Halogenated'] = df['SMILES'].str.contains(
            r'Cl|Br|I(?![nr])|F(?![elrm])', regex=True, na=False)

        # Ensure other potentially relevant columns exist for downstream consistency
        if 'Activity Type' not in df.columns: df['Activity Type'] = 'Unknown'
        if 'Standard Value (nM)' not in df.columns: df['Standard Value (nM)'] = np.nan
        else: df['Standard Value (nM)'] = pd.to_numeric(df['Standard Value (nM)'], errors='coerce')
        if 'accession' not in df.columns: df['accession'] = 'Custom Upload' # Or some placeholder
        
        # If Fingerprint column exists, interpret the containing strings as an array and add as many columns as needed, each containing the nth entree of the fingerprint array
        if 'Fingerprint' in df.columns:
            # Remove first row containing names of columns
            fingerprint_cols = df['Fingerprint'].apply(lambda x: x.split(',') if isinstance(x, str) else [])
            
            df.drop(columns=['Fingerprint'], inplace=True)
            
            fingerprint_cols = fingerprint_cols.apply(lambda x: [int(i) for i in x])
            fingerprint_cols = pd.DataFrame(fingerprint_cols.tolist(), index=df.index)
            fingerprint_cols.columns = [f'{i+1}' for i in range(fingerprint_cols.shape[1])]
            # Ensure all fingerprint columns are numeric
            fingerprint_cols = fingerprint_cols.apply(pd.to_numeric, errors='coerce')
            # Add fingerprint columns to the main DataFrame
            df = pd.concat([df, fingerprint_cols], axis=1)
            
        
        # Remove any pre-existing projection columns to avoid conflicts
        cols_to_drop = [col for col in df.columns if col.startswith(('PCA-', 'UMAP-'))]
        if cols_to_drop:
            df.drop(columns=cols_to_drop, inplace=True)
            print(f"Dropped pre-existing projection columns from custom data: {cols_to_drop}")

        return df

    except Exception as e:
        print(f"Error parsing uploaded CSV file '{filename}': {e}")
        return pd.DataFrame()


def filter_similarity_data_optimized(df, slider_values_list, x_axis, y_axis,
                                     zinc_visibility, nonhalogenated_visibility,
                                     target_motor_protein, selected_function, similar_protein_ids):
    # ... (This function remains largely unchanged)
    if df is None or df.empty:
        print("filter_similarity_data_optimized: Input DataFrame is empty.")
        return pd.DataFrame()

    print(f"Filtering data: {len(df)} rows initially.")
    df_filtered = df.copy() 

    if not x_axis or x_axis not in df_filtered.columns:
        print(f"Warning: X-axis '{x_axis}' not found in DataFrame. Using 'PCA-1'.")
        x_axis = 'PCA-1'
        if x_axis not in df_filtered.columns: 
            print("Error: Fallback 'PCA-1' also not found. Cannot filter.")
            return pd.DataFrame()
    if not y_axis or y_axis not in df_filtered.columns:
        print(f"Warning: Y-axis '{y_axis}' not found in DataFrame. Using 'PCA-2'.")
        y_axis = 'PCA-2'
        if y_axis not in df_filtered.columns: 
             print("Error: Fallback 'PCA-2' also not found. Cannot filter.")
             return pd.DataFrame()

    required_cols = ['MOLECULE ID', 'SMILES', 'Activity Type', 'Standard Value (nM)',
                     'accession', 'Is CUSTOM', 'Is ZINC', 'Is Halogenated', x_axis, y_axis]
    # Ensure all required_cols are present, add with NA if not for safety, though they should be.
    for r_col in required_cols:
        if r_col not in df_filtered.columns:
            print(f"Warning: Column '{r_col}' missing in df_filtered. Adding as NA.")
            df_filtered[r_col] = pd.NA
            if r_col in [x_axis, y_axis]: # If a dimension column is missing, this is critical
                df_filtered[r_col] = np.nan


    slider_values_dict = dict(zip(activity_types, slider_values_list))
    activity_mask = pd.Series(True, index=df_filtered.index)

    for atype, log_range in slider_values_dict.items():
        if log_range is None: continue 
        log_min, log_max = log_range
        min_nM = 10**log_min
        max_nM = 10**log_max
        is_current_type = (df_filtered['Activity Type'] == atype)
        value_in_range = (df_filtered['Standard Value (nM)'].notna()) & \
                         (df_filtered['Standard Value (nM)'] >= min_nM) & \
                         (df_filtered['Standard Value (nM)'] <= max_nM)
        activity_mask &= (~is_current_type | value_in_range)

    condlist = [
        df_filtered['Is ZINC'],                                   
        df_filtered['Is CUSTOM'],                                 
        ~activity_mask,                                           
        (df_filtered['accession'] == target_motor_protein) & activity_mask, 
        (df_filtered['accession'].isin(similar_protein_ids if similar_protein_ids else [])) & activity_mask, 
    ]
    choicelist = [
        'ZINC',
        'CUSTOM',
        'OUTSIDE ACTIVITY RANGE',
        f'ACTIVE WRT {target_motor_protein}'.upper() if target_motor_protein else 'ACTIVE WRT TARGET', 
        f'ACTIVE WRT {selected_function}'.upper() if selected_function else 'ACTIVE WRT FUNCTION', 
    ]
    default_category = 'IGNORED' 
    df_filtered['Color Category'] = np.select(condlist, choicelist, default=default_category)

    visibility_mask = pd.Series(True, index=df_filtered.index)
    if 'SHOW_ZINC' not in zinc_visibility: 
        visibility_mask &= ~df_filtered['Is ZINC']
        print("Filtering out ZINC compounds.")
    if 'SHOW_NON_HALOGENATED' not in nonhalogenated_visibility: 
        visibility_mask &= df_filtered['Is Halogenated'] # Keep True, filter out False
        print("Filtering out non-halogenated compounds (showing only halogenated).") # Corrected logic interpretation
    else: # If checked, it means show non-halogenated, so we *don't* filter based on Is Halogenated
        print("Showing all compounds regardless of halogenation OR only non-halogenated based on checkbox state")
        # If the checkbox means "Show ONLY non-halogenated", then the logic needs inversion:
        # visibility_mask &= ~df_filtered['Is Halogenated'] # Keep non-halogenated (Is Halogenated == False)
        # Assuming current logic: if "SHOW_NON_HALOGENATED" is checked, this filter does nothing. If unchecked, only halogenated are shown.
        # Let's clarify: If box is checked, show non-halogenated. If unchecked, show all (or only halogenated, depending on intent).
        # Let's assume: if checked "SHOW_NON_HALOGENATED", we show compounds where Is Halogenated is False.
        # If unchecked, we show all (no filter on halogenation from this checkbox).
        # The current code implies: if SHOW_NON_HALOGENATED is *not* in value (i.e. box unchecked), then we keep only Halogenated.
        # If SHOW_NON_HALOGENATED *is* in value (i.e. box checked), then this filter part is skipped, showing all.
        # This is counter-intuitive for "Show non-halogenated".
        # Let's re-evaluate: If "Show non-halogenated" is checked, we *want* rows where 'Is Halogenated' is False.
        # If it's NOT checked, we don't apply this specific filter (i.e., show all regarding halogenation).

        # Corrected logic for "Show non-halogenated compounds" checkbox:
        # The value "SHOW_NON_HALOGENATED" is in nonhalogenated_visibility if the switch is ON.
        if "SHOW_NON_HALOGENATED" in nonhalogenated_visibility: # Switch is ON
             visibility_mask &= ~df_filtered['Is Halogenated'] # Keep only non-halogenated
             print("Showing only non-halogenated compounds.")
        # else: switch is OFF, so don't filter based on halogenation (show all)


    final_df = df_filtered[visibility_mask].copy() 
    print(f"Filtering complete. {len(final_df)} rows remaining.")

    available_dims = [dim for dim in all_dimensions if dim in final_df.columns]
    columns_to_keep = list(set(required_cols + available_dims + ['Color Category']))
    ordered_cols = [c for c in required_cols + ['Color Category'] if c in columns_to_keep] \
                 + [d for d in available_dims if d not in required_cols and d in columns_to_keep]
    missing_final_cols = [c for c in ordered_cols if c not in final_df.columns]
    if missing_final_cols:
         print(f"Warning: Columns expected in final filtered data are missing: {missing_final_cols}")

    return final_df[[c for c in ordered_cols if c in final_df.columns]]


def perform_distance_based_search(df_filtered, x_axis, y_axis, trigger_id, target_motor_protein, num_returned, worst=False):
    # ... (This function remains largely unchanged)
    print(f"Performing distance search: trigger={trigger_id}, worst={worst}, num={num_returned}")
    if df_filtered is None or df_filtered.empty:
        print("Distance search: Input DataFrame is empty.")
        return pd.DataFrame()

    required_cols = [x_axis, y_axis, 'Is CUSTOM', 'Is ZINC', 'Color Category', 'accession', 'MOLECULE ID', 'SMILES']
    missing_cols = [col for col in required_cols if col not in df_filtered.columns]
    if missing_cols:
        print(f"Distance search: Missing required columns: {missing_cols}")
        # Ensure x_axis and y_axis are present before trying to use them.
        # Add them with NaN if missing, so subsetting doesn't fail immediately,
        # though dropna will then empty the sets.
        for mc in missing_cols:
            if mc not in df_filtered.columns: df_filtered[mc] = np.nan
            if mc in [x_axis,y_axis]: df_filtered[mc] = np.nan # Critical for coords

    set_a = pd.DataFrame()
    set_b = pd.DataFrame()
    base_a_mask = ~df_filtered['Is CUSTOM'] & ~df_filtered['Is ZINC'] & \
                  ~df_filtered['Color Category'].isin(['IGNORED', 'OUTSIDE ACTIVITY RANGE'])

    if trigger_id in ['export-bestfits-button-any', 'export-worstfits-button-any']:
        set_a = df_filtered[base_a_mask].copy()
        set_b = df_filtered[df_filtered['Is CUSTOM']].copy()
    elif trigger_id in ['export-bestfits-button-target', 'export-worstfits-button-target']:
        if target_motor_protein:
            target_mask = base_a_mask & (df_filtered['accession'] == target_motor_protein)
            set_a = df_filtered[target_mask].copy()
        else: set_a = pd.DataFrame()
        set_b = df_filtered[df_filtered['Is CUSTOM']].copy()
    elif trigger_id in ['export-zinc-button-any', 'export-worst-zinc-button-any']:
        set_a = df_filtered[base_a_mask].copy()
        set_b = df_filtered[df_filtered['Is ZINC']].copy()
    elif trigger_id in ['export-zinc-button-target', 'export-worst-zinc-button-target']:
        if target_motor_protein:
            target_mask = base_a_mask & (df_filtered['accession'] == target_motor_protein)
            set_a = df_filtered[target_mask].copy()
        else: set_a = pd.DataFrame()
        set_b = df_filtered[df_filtered['Is ZINC']].copy()
    else:
        print(f"Unhandled trigger ID for distance search: {trigger_id}")
        return pd.DataFrame()

    if set_a.empty or set_b.empty:
        print(f"Warning: Set A ({len(set_a)}) or Set B ({len(set_b)}) is empty for trigger {trigger_id}. Cannot perform search.")
        return pd.DataFrame()

    set_a.dropna(subset=[x_axis, y_axis], inplace=True)
    set_b.dropna(subset=[x_axis, y_axis], inplace=True)
    if set_a.empty or set_b.empty:
        print(f"Warning: Set A ({len(set_a)}) or Set B ({len(set_b)}) became empty after dropping NaN coordinates.")
        return pd.DataFrame()

    set_a_coords = set_a[[x_axis, y_axis]].values
    set_b_coords = set_b[[x_axis, y_axis]].values

    try:
        tree = cKDTree(set_a_coords)
        min_distances, nearest_indices = tree.query(set_b_coords, k=1)
        set_b['Min Distance'] = min_distances
        set_a_original_indices = set_a.index[nearest_indices]
        set_b['TARGET_MOLECULE ID'] = set_a.loc[set_a_original_indices, 'MOLECULE ID'].values
        set_b['TARGET_SMILES'] = set_a.loc[set_a_original_indices, 'SMILES'].values
        set_b['TARGET_Accession'] = set_a.loc[set_a_original_indices, 'accession'].values
        set_b_sorted = set_b.sort_values(by='Min Distance', ascending=(not worst))
        set_b_sorted['Rank'] = np.arange(1, len(set_b_sorted) + 1)
        print(f"Distance search completed. Returning top {num_returned} results.")
        return set_b_sorted.head(num_returned)
    except Exception as e:
        print(f"Error during KDTree search: {e}")
        return pd.DataFrame()


def get_combined_dataframe(data_type, molecular_function, projected_custom_data_records):
    """Retrieves base data and combines with already projected custom data (if any)."""
    print("--------------------")
    print("Retrieving combined DataFrame...")
    start_time = time.time()

    base_df = get_base_similarity_dataframe(data_type, molecular_function)
    if base_df is None: base_df = pd.DataFrame()
    print(f"Base DataFrame retrieved. Shape: {base_df.shape}")

    custom_df = pd.DataFrame() # Default to empty
    if projected_custom_data_records:
        try:
            custom_df = pd.DataFrame(projected_custom_data_records)
            print(f"Reconstructed projected custom DataFrame. Shape: {custom_df.shape}")
        except Exception as e:
            print(f"Error reconstructing projected custom DataFrame from records: {e}")
            custom_df = pd.DataFrame() # Ensure it's an empty DF on error
    
    if not custom_df.empty:
        base_cols = set(base_df.columns)
        custom_cols = set(custom_df.columns)
        all_cols_union = list(base_cols.union(custom_cols))

        # Align columns: Add missing columns with NaN to both DataFrames
        for col in all_cols_union:
            if col not in base_df.columns:
                base_df[col] = np.nan
            if col not in custom_df.columns:
                custom_df[col] = np.nan
        
        base_df = base_df[all_cols_union]
        custom_df = custom_df[all_cols_union]
        
        combined_df = pd.concat([base_df, custom_df], ignore_index=True)
        print(f"Combined DataFrame created by concatenation. Shape: {combined_df.shape}")
    else:
        combined_df = base_df
        print("No projected custom data provided, using base DataFrame.")

    combined_df = combined_df.copy()
    end_time = time.time()
    print(f"DataFrame retrieval/combination took {end_time - start_time:.2f} seconds.")
    print("--------------------")
    return combined_df

# ------------------ Callbacks ------------------

# Callback to store uploaded raw custom features data
@app.callback(
    Output('uploaded-custom-features-df-store', 'data'),
    Input('custom-features-upload', 'contents'),
    State('custom-features-upload', 'filename'),
    prevent_initial_call=True
)
def store_uploaded_features_data(contents, filename):
    if contents:
        print(f"Custom features file uploaded: {filename}")
        df_raw = parse_uploaded_csv(contents, filename)
        if df_raw is not None and not df_raw.empty:
            print(f"Storing raw features data. Shape: {df_raw.shape}")
            return df_raw.to_dict('records')
    return None

# Callback to store uploaded raw custom fingerprints data
@app.callback(
    Output('uploaded-custom-fingerprints-df-store', 'data'),
    Input('custom-fingerprints-upload', 'contents'),
    State('custom-fingerprints-upload', 'filename'),
    prevent_initial_call=True
)
def store_uploaded_fingerprints_data(contents, filename):
    if contents:
        print(f"Custom fingerprints file uploaded: {filename}")
        df_raw = parse_uploaded_csv(contents, filename)
        if df_raw is not None and not df_raw.empty:
            print(f"Storing raw fingerprints data. Shape: {df_raw.shape}")
            return df_raw.to_dict('records')
    return None

# Callback for "Project Custom Data" button
@app.callback(
    Output('projected-custom-data-store', 'data'),
    Input('project-custom-data-button', 'n_clicks'),
    [State('data-type-radio', 'value'),
     State('molecular-function-dropdown', 'value'),
     State('uploaded-custom-features-df-store', 'data'),
     State('uploaded-custom-fingerprints-df-store', 'data')],
    prevent_initial_call=True
)
def handle_project_custom_data(n_clicks, data_type, molecular_function,
                               raw_features_records, raw_fingerprints_records):
    if not n_clicks or not molecular_function:
        print("Projection skipped: No button click or molecular function not selected.")
        raise PreventUpdate

    print(f"Project Custom Data button clicked. Data type: {data_type}, Function: {molecular_function}")
    
    raw_custom_df_records = None
    if data_type == 'features':
        raw_custom_df_records = raw_features_records
    elif data_type == 'fingerprints':
        raw_custom_df_records = raw_fingerprints_records

    if not raw_custom_df_records:
        print("No raw custom data available for projection.")
        return None # Or empty list of records

    try:
        raw_custom_df = pd.DataFrame(raw_custom_df_records)
        if raw_custom_df.empty:
            print("Raw custom DataFrame is empty. Cannot project.")
            return None
        
        print(f"Retrieved raw custom data for projection. Shape: {raw_custom_df.shape}")
        
        projected_df = project_custom_data(raw_custom_df, data_type, molecular_function)
        
        if projected_df is not None and not projected_df.empty:
            print(f"Custom data projected successfully. Shape: {projected_df.shape}")
            # Check if projection columns were actually added
            if not any(col.startswith('PCA-') for col in projected_df.columns):
                 print("Warning: PCA columns seem to be missing after projection.")
            return projected_df.to_dict('records')
        else:
            print("Projection resulted in an empty DataFrame.")
            return None
            
    except Exception as e:
        print(f"Error during custom data projection: {e}")
        import traceback
        traceback.print_exc()
        return None


# --- Protein Info Callbacks (Unchanged) ---
@app.callback(
    [Output("target-motor-protein-dropdown", "value"), 
     Output("molecular-function-dropdown", "options"),
     Output("molecular-function-dropdown", "value"),
     Output("target-organism-store", "data")],
    [Input("fetch-protein-button", "n_clicks")],
    [State("uniprot-id-input", "value")]
)
def fetch_protein_data(n_clicks, uniprot_id):
    if not n_clicks or not uniprot_id: raise PreventUpdate
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}?format=json"
    protein_name = uniprot_id 
    organism = None
    function_options = []
    default_function = None
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status() 
        data = response.json()
        try: protein_name = data.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", uniprot_id)
        except: pass 
        try: organism = data.get("organism", {}).get("scientificName", None)
        except: pass
        molecular_functions = set()
        for kw in data.get("keywords", []):
            if kw.get("category") == "Molecular function":
                fn = kw.get("name") 
                if fn: molecular_functions.add(fn)
        function_options = [{"label": fn, "value": fn} for fn in sorted(list(molecular_functions))]
        if function_options: default_function = function_options[0]["value"]
    except requests.exceptions.RequestException as e:
        print(f"Error fetching Uniprot data for {uniprot_id}: {e}")
        return uniprot_id, [], None, None 
    except Exception as e:
        print(f"Error processing Uniprot data for {uniprot_id}: {e}")
        return uniprot_id, [], None, None
    print(f"Fetched data for {uniprot_id}: Name='{protein_name}', Organism='{organism}', Functions='{default_function}'")
    return protein_name, function_options, default_function, organism

@cache.memoize() 
def fetch_proteins_by_function(function_query, organism):
    print(f"CACHE MISS/Executing: Fetching proteins from UniProt for function='{function_query}', Organism='{organism}'")
    if not function_query: return []
    quoted_function = f'"{function_query}"'
    if organism: query = f'(keyword:{quoted_function}) AND (organism_name:"{organism}")'
    else: query = f'keyword:{quoted_function}'
    base_url = "https://rest.uniprot.org/uniprotkb/search"
    params = {"query": query, "format": "json", "fields": "accession", "size": 500}
    primary_ids = set() 
    req_count = 0
    max_reqs = 10 
    try:
        start_time = time.time()
        while base_url and req_count < max_reqs:
            req_count += 1
            current_params_log = params if params else 'from next link'
            print(f"  Fetching page {req_count} from {base_url} (params: {current_params_log})")
            response = requests.get(base_url, params=params, timeout=20) 
            response.raise_for_status()
            data = response.json()
            results = data.get("results", [])
            page_ids = set(entry.get("primaryAccession") for entry in results if entry.get("primaryAccession"))
            print(f"  Page {req_count} returned {len(page_ids)} IDs.")
            primary_ids.update(page_ids)
            next_link_header = response.headers.get("Link")
            base_url = None 
            if next_link_header:
                links = requests.utils.parse_header_links(next_link_header)
                for link in links:
                    if link.get("rel") == "next":
                        base_url = link.get("url")
                        params = {} 
                        print(f"  Found next link: {base_url}")
                        break
            if not base_url and req_count < max_reqs: print(f"  No more pages found after page {req_count}.")
        end_time = time.time()
        print(f"  UniProt query finished in {end_time - start_time:.2f} seconds.")
        print(f"Found {len(primary_ids)} unique protein IDs for function '{function_query}'. Storing in cache.")
        return list(primary_ids)
    except requests.exceptions.RequestException as e:
        print(f"Error during UniProt request for function '{function_query}': {e}")
        return [] 
    except Exception as e:
        print(f"Unexpected error fetching proteins by function: {e}")
        return []

@app.callback(
    Output('similar-protein-ids', 'data'),
    [Input('molecular-function-dropdown', 'value')],
    [State('target-organism-store', 'data')] 
)
def update_similar_proteins(selected_function, organism):
    if not selected_function:
        print("No molecular function selected, skipping protein fetch.")
        raise PreventUpdate
    print(f"update_similar_proteins: Requesting similar proteins for '{selected_function}' (Organism: {organism})...")
    start_time = time.time()
    similar_ids = fetch_proteins_by_function(selected_function, organism)
    end_time = time.time()
    print(f"update_similar_proteins: Retrieved {len(similar_ids)} similar IDs in {end_time - start_time:.4f} seconds (may be from cache).")
    return similar_ids

# --- Main Graph Update Callback ---
@app.callback(
    Output('similarity-space-graph', 'figure'),
    [Input({'type': 'slider', 'index': ALL}, 'value'),
     Input('x-axis-dropdown', 'value'),
     Input('y-axis-dropdown', 'value'),
     Input('zinc-visibility-checkbox', 'value'),
     Input('nonhalogenated-checkbox', 'value'),
     Input('selected-molecule-ids', 'data'),
     Input('data-type-radio', 'value'),
     Input('molecular-function-dropdown', 'value'),
     Input('similar-protein-ids', 'data'),
     Input('projected-custom-data-store', 'data')], # New input
    [State('target-motor-protein-dropdown', 'value')]
    # Removed custom_xxx_contents from State
)
def filter_and_update_graph(slider_values, x_axis, y_axis, zinc_visibility, nonhalogenated_visibility,
                            selected_ids_data, data_type, selected_function, similar_protein_ids,
                            projected_custom_data_records, # New argument
                            target_motor_protein):

    print("\n--- filter_and_update_graph triggered ---")
    trigger_info = ctx.triggered_id if ctx.triggered_id else "initial load"
    # Handle dict trigger_id for pattern matching callbacks
    if isinstance(trigger_info, dict): trigger_info = list(trigger_info.keys())[0]
    print(f"Triggered by: {trigger_info}")


    if not selected_function or not x_axis or not y_axis:
        print("Missing selected function or axes, preventing update.")
        return go.Figure() 

    try:
        df_combined = get_combined_dataframe(data_type, selected_function, projected_custom_data_records)
    except FileNotFoundError as e:
         print(f"Error: Base data file not found: {e}")
         fig = go.Figure()
         fig.update_layout(title=f"Error: Data file not found for {selected_function}/{data_type}",
                          xaxis={'visible': False}, yaxis={'visible': False})
         return fig
    except Exception as e:
         print(f"Error getting combined dataframe: {e}")
         fig = go.Figure()
         fig.update_layout(title="Error loading or processing data",
                          xaxis={'visible': False}, yaxis={'visible': False})
         return fig

    if df_combined is None or df_combined.empty:
        print("No data available (base or custom) after loading/processing.")
        fig = go.Figure()
        fig.update_layout(title=f"No data loaded for {selected_function}/{data_type}",
                          xaxis={'visible': False}, yaxis={'visible': False})
        return fig

    if x_axis not in df_combined.columns or y_axis not in df_combined.columns:
        print(f"Error: Selected axes '{x_axis}' or '{y_axis}' not found in combined data.")
        available_dims_in_data = [d for d in all_dimensions if d in df_combined.columns]
        if len(available_dims_in_data) >= 2:
            x_axis = available_dims_in_data[0]
            y_axis = available_dims_in_data[1]
            print(f"Falling back to axes: '{x_axis}', '{y_axis}'")
        else:
            fig = go.Figure()
            fig.update_layout(title=f"Error: Insufficient coordinate data loaded ({x_axis}, {y_axis} missing)",
                              xaxis={'visible': False}, yaxis={'visible': False})
            return fig

    df_filtered = filter_similarity_data_optimized(df_combined, slider_values, x_axis, y_axis,
                                                 zinc_visibility, nonhalogenated_visibility,
                                                 target_motor_protein, selected_function, similar_protein_ids or [])

    if df_filtered.empty:
        print("Data is empty after filtering.")
        fig = go.Figure()
        fig.update_layout(title="No data points match current filters",
                          xaxis={'visible': False}, yaxis={'visible': False})
        return fig

    selected_ids_list = []
    if selected_ids_data and isinstance(selected_ids_data, dict):
        selected_ids_list = selected_ids_data.get("selected_ids", [])

    df_filtered['Selected'] = df_filtered['MOLECULE ID'].isin(selected_ids_list)
    df_selected = df_filtered[df_filtered['Selected']].copy()
    df_non_selected = df_filtered[~df_filtered['Selected']].copy()

    color_discrete_map = {
        'IGNORED': 'lightgrey', 'OUTSIDE ACTIVITY RANGE': 'grey', 'ZINC': 'purple', 'CUSTOM': 'green',
        f'ACTIVE WRT {sf.upper()}' if (sf := selected_function) else 'ACTIVE WRT FUNCTION': 'blue',
        f'ACTIVE WRT {tp.upper()}' if (tp := target_motor_protein) else 'ACTIVE WRT TARGET': 'red'
    }
    plot_order = ['IGNORED', 'OUTSIDE ACTIVITY RANGE', 'ZINC'] + \
                 [k for k in color_discrete_map if k.startswith('ACTIVE')] + ['CUSTOM']

    fig = go.Figure()
    for category in plot_order:
        color = color_discrete_map.get(category, 'black') 
        df_cat = df_non_selected[df_non_selected['Color Category'] == category]
        if df_cat.empty: continue
        fig.add_trace(go.Scattergl(
            x=df_cat[x_axis], y=df_cat[y_axis], mode='markers',
            marker=dict(size=8, opacity=0.8, symbol='circle', color=color),
            hoverinfo='skip', customdata=df_cat[['SMILES', 'MOLECULE ID']].values, name=category
        ))

    hover_texts = []
    if not df_selected.empty:
        for _, row in df_selected.iterrows():
            hover_texts.append(f"SMILES: {row.get('SMILES', 'N/A')}<br>MOLECULE ID: {row.get('MOLECULE ID', 'N/A')}<br>Category: {row.get('Color Category', 'N/A')}")
        fig.add_trace(go.Scattergl(
            x=df_selected[x_axis], y=df_selected[y_axis], mode='markers',
            marker=dict(size=16, opacity=1.0, symbol='circle', color='orange', line=dict(color='black', width=2)),
            text=hover_texts, hoverinfo='text', name='Selected', showlegend=True
        ))

    x_range, y_range = [-10, 10], [-10, 10] 
    if not df_filtered.empty:
        x_min, x_max = df_filtered[x_axis].min(), df_filtered[x_axis].max()
        y_min, y_max = df_filtered[y_axis].min(), df_filtered[y_axis].max()
        if pd.notna(x_min) and pd.notna(x_max) and x_min != x_max : # Ensure valid range
             x_range = [x_min - (x_max-x_min)*0.05, x_max + (x_max-x_min)*0.05]
        if pd.notna(y_min) and pd.notna(y_max) and y_min != y_max: # Ensure valid range
             y_range = [y_min - (y_max-y_min)*0.05, y_max + (y_max-y_min)*0.05]

    fig.update_layout(
        hovermode='closest', clickmode='event+select', 
        title=f"{x_axis} vs {y_axis} Projection ({len(df_filtered)} points shown)",
        xaxis_title=x_axis, yaxis_title=y_axis,
        xaxis=dict(range=x_range, constrain='domain'), 
        yaxis=dict(range=y_range, scaleanchor="x", scaleratio=1), 
        template='plotly_white', legend=dict(itemsizing='constant', traceorder='normal'),
        margin=dict(l=40, r=40, t=60, b=40), uirevision='constant'
    )
    print("--- filter_and_update_graph finished ---")
    return fig


@app.callback(
    Output('selected-molecule-ids', 'data'),
    [Input('similarity-space-graph', 'selectedData'),
     Input('similarity-space-graph', 'clickData'),
     Input('export-bestfits-button-any', 'n_clicks'),
     Input('export-bestfits-button-target', 'n_clicks'),
     Input('export-zinc-button-any', 'n_clicks'),
     Input('export-zinc-button-target', 'n_clicks'),
     Input('export-worstfits-button-any', 'n_clicks'),
     Input('export-worstfits-button-target', 'n_clicks'),
     Input('export-worst-zinc-button-any', 'n_clicks'),
     Input('export-worst-zinc-button-target', 'n_clicks')],
    [State('num-bestfits', 'value'),
     State('x-axis-dropdown', 'value'),
     State('y-axis-dropdown', 'value'),
     State({'type': 'slider', 'index': ALL}, 'value'), 
     State('zinc-visibility-checkbox', 'value'),       
     State('nonhalogenated-checkbox', 'value'),    
     State('data-type-radio', 'value'),                
     State('target-motor-protein-dropdown', 'value'), 
     State('molecular-function-dropdown', 'value'),    
     State('similar-protein-ids', 'data'),             
     State('projected-custom-data-store', 'data'), # Changed from custom_xxx_contents
     State('selected-molecule-ids', 'data')], 
    prevent_initial_call=True
)
def determine_selected_molecules(
    selectedData, clickData,
    n_clicks_bf_any, n_clicks_bf_target, n_clicks_z_any, n_clicks_z_target,
    n_clicks_wf_any, n_clicks_wf_target, n_clicks_wz_any, n_clicks_wz_target,
    n_bestfits, x_axis, y_axis, slider_values,
    zinc_visibility, nonhalogenated_visibility, data_type, target_motor_protein,
    selected_function, similar_protein_ids,
    projected_custom_data_records, # Changed argument
    current_selection_data):

    print("\n--- determine_selected_molecules triggered ---")
    triggered_id_raw = ctx.triggered_id
    triggered_id = triggered_id_raw if isinstance(triggered_id_raw, str) else list(triggered_id_raw.keys())[0] if triggered_id_raw else "unknown"
    prop_id = ctx.triggered[0]['prop_id']
    print(f"Trigger ID: {triggered_id}, Property: {prop_id}")

    selected_ids = []
    if isinstance(current_selection_data, dict):
        selected_ids = current_selection_data.get("selected_ids", [])
    output_data = {"selected_ids": selected_ids, "export_df": None} 

    if prop_id == 'similarity-space-graph.selectedData':
        if selectedData and selectedData.get('points'):
            output_data["selected_ids"] = [point['customdata'][1] for point in selectedData['points'] if 'customdata' in point and len(point['customdata']) > 1]
            print(f"Graph selection updated: {len(output_data['selected_ids'])} points selected.")
        else:
             output_data["selected_ids"] = []
             print("Graph selection cleared.")
        return output_data
    elif prop_id == 'similarity-space-graph.clickData':
        if clickData and clickData.get('points'):
            clicked_id = clickData['points'][0]['customdata'][1]
            if clicked_id in selected_ids: selected_ids.remove(clicked_id)
            else: selected_ids.append(clicked_id)
            output_data["selected_ids"] = selected_ids
            print(f"Clicked point: {clicked_id}. Selection count: {len(selected_ids)}")
            return output_data
        else:
            raise PreventUpdate

    export_buttons = [
        'export-bestfits-button-any', 'export-bestfits-button-target',
        'export-zinc-button-any', 'export-zinc-button-target',
        'export-worstfits-button-any', 'export-worstfits-button-target',
        'export-worst-zinc-button-any', 'export-worst-zinc-button-target'
    ]
    if triggered_id in export_buttons:
        print(f"Export button clicked: {triggered_id}")
        if not selected_function:
             print("Cannot export: Molecular function not selected.")
             raise PreventUpdate 
        try:
             df_combined = get_combined_dataframe(data_type, selected_function, projected_custom_data_records)
        except Exception as e:
             print(f"Error getting combined data for export: {e}")
             raise PreventUpdate 
        if df_combined is None or df_combined.empty:
            print("Cannot export: Combined data is empty.")
            raise PreventUpdate
        
        df_filtered = filter_similarity_data_optimized(df_combined, slider_values, x_axis, y_axis,
                                                     zinc_visibility, nonhalogenated_visibility,
                                                     target_motor_protein, selected_function, similar_protein_ids or [])
        if df_filtered.empty:
            print("Warning: Filtered data is empty. Cannot perform export search.")
            output_data["selected_ids"] = [] 
            output_data["export_df"] = pd.DataFrame().to_dict(orient="records") 
            return output_data

        num_returned = int(n_bestfits) if n_bestfits else 20
        worst_flag = triggered_id.startswith("export-worst")
        export_df_sorted = perform_distance_based_search(
            df_filtered, x_axis, y_axis, triggered_id, target_motor_protein, num_returned, worst=worst_flag
        )
        if export_df_sorted.empty:
            print(f"Warning: Distance search returned no results for trigger {triggered_id}.")
            output_data["selected_ids"] = [] 
            output_data["export_df"] = pd.DataFrame().to_dict(orient="records")
        else:
            export_ids = export_df_sorted['MOLECULE ID'].tolist()
            output_data["selected_ids"] = export_ids 
            output_data["export_df"] = export_df_sorted.to_dict(orient="records")
            print(f"Export successful. {len(export_ids)} molecules selected.")
        return output_data 
    print(f"Unhandled trigger in determine_selected_molecules: {triggered_id}")
    raise PreventUpdate


@app.callback(
    Output('selected-molecules-container', 'children'),
    [Input('selected-molecule-ids', 'data')], 
    [State('data-type-radio', 'value'),
     State('molecular-function-dropdown', 'value'),
     State('projected-custom-data-store', 'data')] # Changed
)
def display_selected_molecules(selected_ids_data, data_type, selected_function,
                               projected_custom_data_records): # Changed
    print("\n--- display_selected_molecules triggered ---")
    if not selected_function: return html.P("Select a molecular function.")

    selected_ids = []
    if selected_ids_data and isinstance(selected_ids_data, dict):
        selected_ids = selected_ids_data.get("selected_ids", [])
    if not selected_ids: return html.P("No molecules selected.")
    print(f"Displaying details for {len(selected_ids)} selected molecules.")

    try:
        df_combined = get_combined_dataframe(data_type, selected_function, projected_custom_data_records)
    except Exception as e:
         print(f"Error getting combined data for display: {e}")
         return html.Div([html.P("Error loading data to display molecule details."),
                          html.P(f"Selected IDs: {', '.join(selected_ids)}")])
    if df_combined is None or df_combined.empty:
         return html.P("Data not available to display molecule details.")

    selected_molecules_df = df_combined[df_combined['MOLECULE ID'].isin(selected_ids)].copy()
    if selected_molecules_df.empty:
        print("Selected IDs not found in the current combined dataset.")
        return html.P("Selected molecules not found (data might have reloaded or they are filtered out).")

    molecules_display = []
    selected_molecules_df['sort_order'] = selected_molecules_df['MOLECULE ID'].apply(lambda x: selected_ids.index(x) if x in selected_ids else float('inf'))
    selected_molecules_df.sort_values('sort_order', inplace=True)

    for _, row in selected_molecules_df.iterrows():
        info = row.to_dict()
        smiles = info.get('SMILES', 'N/A')
        mol_id = info.get('MOLECULE ID', 'N/A')
        img_src = mol_to_image_base64(smiles)
        details = [
            html.P(f"{mol_id}", style={'wordBreak': 'break-all', 'textAlign': 'left', 'fontWeight': 'bold', 'marginBottom': '5px'}),
            html.P(f"SMILES: {smiles}", style={'wordBreak': 'break-all', 'textAlign': 'left', 'fontSize': 'small', 'marginBottom': '3px'})
        ]
        if 'Activity Type' in info and pd.notna(info['Activity Type']) and info['Activity Type'] != 'Unknown':
            activity_val = f"{info['Standard Value (nM)']:.2f}" if pd.notna(info['Standard Value (nM)']) else "N/A"
            details.append(html.P(f"Activity: {info['Activity Type']} = {activity_val} nM", style={'fontSize': 'small', 'marginBottom': '3px'}))
        if 'accession' in info and pd.notna(info['accession']) and info['accession'] != 'Unknown' and info['accession'] != 'Custom Upload':
             details.append(html.P(f"Target: {info['accession']}", style={'fontSize': 'small', 'marginBottom': '3px'}))
        if 'Color Category' in info and pd.notna(info['Color Category']):
             details.append(html.P(f"Category: {info['Color Category']}", style={'fontSize': 'small', 'marginBottom': '3px'}))


        card_content = []
        if img_src:
            card_content.append(html.Img(src=img_src, style={'width': '100%', 'height': 'auto', 'maxWidth': '150px', 'marginBottom': '10px'}))
        else:
            card_content.append(html.P("[No Image]", style={'textAlign': 'center', 'fontStyle': 'italic'}))
        card_content.extend(details)
        molecules_display.append(dbc.Card(dbc.CardBody(card_content), className="mb-3 shadow-sm", style={'border': '1px solid #ddd'}))
    return molecules_display


@app.callback(
    Output("download-selection", "data"),
    Input("save-selection-button", "n_clicks"),
    [State("selected-molecule-ids", "data"),      
     State("save-selection-filename", "value"), 
     State('data-type-radio', 'value'),
     State('molecular-function-dropdown', 'value'),
     State('projected-custom-data-store', 'data')], # Changed
    prevent_initial_call=True
)
def save_selection(n_clicks, selected_ids_data, filename,
                   data_type, selected_function,
                   projected_custom_data_records): # Changed
    print("\n--- save_selection triggered ---")
    if not n_clicks or not selected_ids_data or not isinstance(selected_ids_data, dict):
        raise PreventUpdate
    if not filename: filename = "selected_molecules.csv"
    if not filename.lower().endswith('.csv'): filename += '.csv'

    df_to_download = pd.DataFrame()
    if selected_ids_data.get("export_df") is not None:
        print("Downloading pre-computed export data.")
        export_records = selected_ids_data["export_df"]
        df_to_download = pd.DataFrame(export_records)
    else:
        selected_ids = selected_ids_data.get("selected_ids", [])
        if not selected_ids: raise PreventUpdate
        print(f"Downloading data for {len(selected_ids)} selected IDs from current view.")
        if not selected_function: raise PreventUpdate
        try:
            df_combined = get_combined_dataframe(data_type, selected_function, projected_custom_data_records)
        except Exception as e:
            print(f"Error getting combined data for download: {e}")
            raise PreventUpdate 
        if df_combined is None or df_combined.empty: raise PreventUpdate
        df_to_download = df_combined[df_combined['MOLECULE ID'].isin(selected_ids)].copy()

    if df_to_download.empty:
        print("Download prevented: No data to download after filtering/selection.")
        raise PreventUpdate
    print(f"Preparing download for {len(df_to_download)} rows to file: {filename}")
    return dcc.send_data_frame(df_to_download.to_csv, filename, index=False)


@app.callback(
    Output({'type': 'slider-value', 'index': ALL}, 'children'),
    Input({'type': 'slider', 'index': ALL}, 'value'),
    prevent_initial_call=True
)
def display_slider_values(slider_values_list):
     outputs = []
     for val_range in slider_values_list:
         if val_range:
             min_val, max_val = 10**val_range[0], 10**val_range[1]
             min_str = f"{min_val:.1f}" if min_val < 1000 else f"{min_val:.1e}"
             max_str = f"{max_val:.1f}" if max_val < 1000 else f"{max_val:.1e}"
             outputs.append(f"{min_str} - {max_str} nM")
         else: outputs.append("N/A")
     return outputs

# ------------------ Main Execution ------------------
if __name__ == '__main__':
    # Initialize the Dash app
    app.run(debug=True)