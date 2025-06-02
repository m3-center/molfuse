import os
import numpy as np
import pandas as pd
import logging
import gc
from compress_pickle import dump
from sklearn.preprocessing import StandardScaler
import argparse
import json # For parsing features list

# CUML / Sklearn imports
try:
    from cuml import PCA as cumlPCA, UMAP as cumlUMAP, TSNE as cumlTSNE
    from cuml.common.device_selection import using_device_type
    CUML_AVAILABLE = True
    logging.info("cuML found. Using GPU for PCA, UMAP, t-SNE where possible.")
except ImportError:
    from sklearn.decomposition import PCA as sklearnPCA
    from sklearn.manifold import TSNE as sklearnTSNE
    try:
        from umap import UMAP as umapUMAP # umap-learn for CPU UMAP
        SKLEARN_UMAP_AVAILABLE = True
    except ImportError:
        SKLEARN_UMAP_AVAILABLE = False
        logging.warning("umap-learn package not found. CPU UMAP will not be available.")
    CUML_AVAILABLE = False
    logging.info("cuML not found. Falling back to scikit-learn for PCA, t-SNE. UMAP on CPU if 'umap-learn' installed.")

# Setup basic logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
log_formatter = logging.Formatter('%(asctime)s - %(levelname)-8s - %(filename)-25s - %(funcName)-25s - %(lineno)-4d - %(message)s')
try:
    file_handler = logging.FileHandler("calculate_simspaces.log", mode='w')
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.INFO) # Or DEBUG for more verbosity
    logger.addHandler(file_handler)
except Exception as e:
    print(f"CRITICAL: Failed to initialize file logger for calculate_simspaces.log: {e}")

# Stream Handler (console output)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_formatter)
stream_handler.setLevel(logging.INFO)
logger.addHandler(stream_handler)

logging.info("--- calculate_similarityspaces_exp.py script started, logging configured. ---")
if CUML_AVAILABLE:
    logging.info("cuML found. Using GPU for PCA, UMAP, t-SNE where possible.")
else:
    logging.info("cuML not found. Falling back to scikit-learn for PCA, t-SNE.")
    if SKLEARN_UMAP_AVAILABLE:
        logging.info("umap-learn package found. CPU UMAP is available.")
    else:
        logging.warning("umap-learn package not found. CPU UMAP will NOT be available.")

FINGERPRINT_COLUMN_PREFIX = "fp_" # Used for output columns after parsing
NUM_FINGERPRINT_BITS = 2048
PRECALCULATED_FP_COLUMN_NAME = "Fingerprint" # Name of the column holding the string "1,0,1..."

def get_descriptor_columns(df, representation_type, target_rdkit_features_list, is_parsed_fp=False):
    """
    Identifies descriptor columns.
    'is_parsed_fp' flag indicates if fingerprint string has already been expanded.
    """
    if representation_type == "features":
        # ... (feature column logic as before) ...
        descriptor_cols = [col for col in target_rdkit_features_list if col in df.columns]
        missing_config_feats = [col for col in target_rdkit_features_list if col not in df.columns]
        if missing_config_feats:
            logging.warning(f"The following configured RDKit features are missing from the input data: {missing_config_feats}. They will be ignored.")
        if not descriptor_cols:
            logging.error(f"No features from RDKIT_FEATURES_LIST_TARGET found in input DataFrame columns: {df.columns.tolist()[:10]}")
            return None
        return descriptor_cols
    elif representation_type == "fingerprints":
        if is_parsed_fp: # Already expanded to fp_0, fp_1, ...
            descriptor_cols = [f"{FINGERPRINT_COLUMN_PREFIX}{i}" for i in range(NUM_FINGERPRINT_BITS)]
            # Check if these columns actually exist
            actual_cols = [col for col in descriptor_cols if col in df.columns]
            if not actual_cols:
                 logging.error(f"Fingerprint representation selected, but no '{FINGERPRINT_COLUMN_PREFIX}*' columns found after expected parsing.")
                 return None
            return actual_cols
        else: # String format in PRECALCULATED_FP_COLUMN_NAME, will be parsed later
            if PRECALCULATED_FP_COLUMN_NAME not in df.columns:
                logging.error(f"Fingerprint representation selected, but the expected string column '{PRECALCULATED_FP_COLUMN_NAME}' is not found.")
                return None
            return [PRECALCULATED_FP_COLUMN_NAME] # Temporarily, will be expanded
    else:
        logging.error(f"Invalid representation_type: {representation_type}")
        return None

def parse_fingerprint_string_column(df_input, fp_string_col_name=PRECALCULATED_FP_COLUMN_NAME, num_bits=NUM_FINGERPRINT_BITS):
    """
    Parses a column containing fingerprint strings "0,1,0,..." into separate bit columns.
    Returns a new DataFrame with expanded fingerprint columns.
    """
    logging.info(f"Parsing fingerprint string column: '{fp_string_col_name}'")

    if fp_string_col_name not in df_input.columns:
        logging.error(f"Fingerprint string column '{fp_string_col_name}' not found in DataFrame for parsing.")
        empty_fp_df = pd.DataFrame(columns=[f"{FINGERPRINT_COLUMN_PREFIX}{i}" for i in range(num_bits)], index=df_input.index)
        df_output = pd.concat([df_input, empty_fp_df], axis=1) # Keep original columns, add empty fp cols
        return df_output


    fp_matrix = np.full((len(df_input), num_bits), np.nan) # Pre-allocate with NaNs

    for idx, fp_str in enumerate(df_input[fp_string_col_name]):
        if pd.isna(fp_str) or not isinstance(fp_str, str) or not fp_str.strip():
            # Already NaN due to pre-allocation
            continue
        bits = fp_str.split(',')
        if len(bits) == num_bits:
            try:
                fp_matrix[idx, :] = [int(b) for b in bits]
            except ValueError:
                 logging.debug(f"ValueError parsing bits in string (row {df_input.index[idx]}): {fp_str[:30]}... Filling with NaNs.")
                 # Already NaN
        else:
            logging.debug(f"Fingerprint string (row {df_input.index[idx]}) '{fp_str[:30]}...' has incorrect length {len(bits)}, expected {num_bits}. Filling with NaNs.")
            # Already NaN

    df_fp_bits = pd.DataFrame(fp_matrix,
                              columns=[f"{FINGERPRINT_COLUMN_PREFIX}{i}" for i in range(num_bits)],
                              index=df_input.index)

    df_output = pd.concat([df_input.drop(columns=[fp_string_col_name]), df_fp_bits], axis=1)
    logging.info(f"Fingerprint string column parsed. Original shape: {df_input.shape}, New shape after FP expansion: {df_output.shape}")
    return df_output


def process_similarity_calculations(
    chembl_mf_data_path, zinc_data_path, target_ligands_unscaled_path_for_tsne,
    simspace_dim, representation_type, target_id_name,
    output_simspace_dir, output_model_dir, target_rdkit_features_list,
    dr_method_flags, umap_metric_flags, tsne_params):

    base_name_prefix = f"{target_id_name}_{representation_type}_dim{simspace_dim}"
    logging.info(f"Processing for: {base_name_prefix}")

    try:
        df_chembl_mf = pd.read_csv(chembl_mf_data_path, low_memory=False)
        df_zinc = pd.read_csv(zinc_data_path, low_memory=False) if zinc_data_path and os.path.exists(zinc_data_path) else pd.DataFrame()
    except Exception as e:
        logging.error(f"Failed to load ChEMBL MF or ZINC data: {e}")
        return

    # --- NEW: Parse fingerprint strings if representation_type is fingerprints ---
    if representation_type == "fingerprints":
        if PRECALCULATED_FP_COLUMN_NAME in df_chembl_mf.columns:
            df_chembl_mf = parse_fingerprint_string_column(df_chembl_mf)
        else:
            logging.warning(f"Precalculated ChEMBL MF fingerprint file {chembl_mf_data_path} does not contain '{PRECALCULATED_FP_COLUMN_NAME}' column. Assuming it's already parsed or of different format.")
        
        if not df_zinc.empty:
            if PRECALCULATED_FP_COLUMN_NAME in df_zinc.columns:
                df_zinc = parse_fingerprint_string_column(df_zinc)
            else:
                logging.warning(f"Precalculated ZINC fingerprint file {zinc_data_path} does not contain '{PRECALCULATED_FP_COLUMN_NAME}' column. Assuming it's already parsed or of different format.")
    # --- END NEW ---

    # (Align columns and concat df_main_combined as before)
    all_main_cols = list(set(df_chembl_mf.columns) | (set(df_zinc.columns) if not df_zinc.empty else set()))
    # Ensure key ID columns are considered for alignment if they exist
    for id_col_candidate in ['Compound ChEMBL ID', 'ZINC_ID', 'SMILES']: # Add SMILES for safety
        if id_col_candidate not in all_main_cols and \
           (id_col_candidate in df_chembl_mf.columns or (not df_zinc.empty and id_col_candidate in df_zinc.columns)):
            all_main_cols.append(id_col_candidate)
    all_main_cols = sorted(list(set(all_main_cols))) # Unique and sorted

    for col in all_main_cols: 
        if col not in df_chembl_mf.columns: df_chembl_mf[col] = pd.NA # Use pd.NA for missing
        if not df_zinc.empty and col not in df_zinc.columns: df_zinc[col] = pd.NA
    
    df_main_combined = pd.concat([df_chembl_mf, df_zinc], ignore_index=True) if not df_zinc.empty else df_chembl_mf.copy()
    logging.info(f"Combined Main (ChEMBL MF + ZINC) DataFrame shape: {df_main_combined.shape}")

    # ***** NEW: Create unified 'MOLECULE ID' column *****
    if 'Compound ChEMBL ID' in df_main_combined.columns and 'ZINC_ID' in df_main_combined.columns:
        df_main_combined['MOLECULE ID'] = df_main_combined['Compound ChEMBL ID'].fillna(df_main_combined['ZINC_ID'])
    elif 'Compound ChEMBL ID' in df_main_combined.columns:
        df_main_combined['MOLECULE ID'] = df_main_combined['Compound ChEMBL ID']
    elif 'ZINC_ID' in df_main_combined.columns:
        df_main_combined['MOLECULE ID'] = df_main_combined['ZINC_ID']
    else:
        # Fallback if neither specific ID column exists, try to use SMILES or generate a placeholder
        if 'SMILES' in df_main_combined.columns:
            logging.warning("Neither 'Compound ChEMBL ID' nor 'ZINC_ID' found. Using 'SMILES' as 'MOLECULE ID'. This might not be unique.")
            df_main_combined['MOLECULE ID'] = df_main_combined['SMILES']
        else:
            logging.warning("Critical ID columns ('Compound ChEMBL ID', 'ZINC_ID', 'SMILES') missing. Generating placeholder 'MOLECULE ID'.")
            df_main_combined['MOLECULE ID'] = 'UNKNOWN_ID_' + pd.Series(df_main_combined.index).astype(str)
    
    # Ensure 'MOLECULE ID' is string and handle any remaining NaNs in it
    df_main_combined['MOLECULE ID'] = df_main_combined['MOLECULE ID'].astype(str).fillna('MISSING_MOLECULE_ID')
    logging.info(f"Unified 'MOLECULE ID' column created. Example IDs: {df_main_combined['MOLECULE ID'].unique()[:5]}")

    if df_main_combined.empty: # ... (rest of empty check as before)
        logging.error("Combined Main DataFrame is empty. Cannot proceed.")
        return

    # Now get_descriptor_columns will expect parsed FPs if repr_type is fingerprints
    descriptor_columns = get_descriptor_columns(df_main_combined, representation_type, target_rdkit_features_list, 
                                                is_parsed_fp=(representation_type == "fingerprints"))
    if not descriptor_columns: 
        logging.error(f"Could not determine descriptor columns for {representation_type}. Aborting this run.")
        return

    # (Drop NaNs, Scaling, PCA, UMAP, t-SNE logic as before)
    # ... The crucial part was parsing the FPs *before* this section ...
    df_main_combined[descriptor_columns] = df_main_combined[descriptor_columns].apply(pd.to_numeric, errors='coerce')
    original_rows_main = len(df_main_combined)
    df_main_combined.dropna(subset=descriptor_columns, how='any', inplace=True) # Drop if ANY descriptor is NaN
    if len(df_main_combined) < original_rows_main:
        logging.info(f"Dropped {original_rows_main - len(df_main_combined)} rows from Main data due to NaNs in descriptors.")
    if df_main_combined.empty:
        logging.error("Main DataFrame is empty after NaN drop. Cannot proceed.")
        return
    
    X_original_main = df_main_combined[descriptor_columns].values.astype(np.float32) # Ensure float32 for cuML compatibility

    logging.info("Standardizing Main data...")
    scaler = StandardScaler()
    X_scaled_main = scaler.fit_transform(X_original_main)
    
    scaler_model_path = os.path.join(output_model_dir, f"{base_name_prefix}_scaler.lzma")
    try:
        with open(scaler_model_path, "wb") as f: dump(scaler, f)
        logging.info(f"Saved scaler model to {scaler_model_path}")
    except Exception as e: logging.error(f"Failed to save scaler model: {e}")

    df_results_main = df_main_combined.copy() 

    # --- PCA (on main data) ---
    if dr_method_flags.get('pca'):
        # ... (PCA logic as provided in the previous full version of this script) ...
        logging.info(f"Performing PCA to {simspace_dim}D on Main data...")
        pca_cols = [f'PCA-{i+1}' for i in range(simspace_dim)]
        try:
            pca_model_main = (cumlPCA(n_components=simspace_dim, random_state=42) if CUML_AVAILABLE
                              else sklearnPCA(n_components=simspace_dim, random_state=42))
            pca_res_main = pca_model_main.fit_transform(X_scaled_main)
            for i in range(simspace_dim): df_results_main[pca_cols[i]] = pca_res_main[:, i]
            
            pca_model_path = os.path.join(output_model_dir, f"{base_name_prefix}_PCA_model.lzma")
            with open(pca_model_path, "wb") as f: dump(pca_model_main, f)
            logging.info(f"Saved PCA model to {pca_model_path}")
        except Exception as e:
            logging.error(f"PCA failed: {e}")
            for col in pca_cols: df_results_main[col] = np.nan
        gc.collect()


    # --- UMAP (on main data) ---
    if dr_method_flags.get('umap'):
        # ... (UMAP logic as provided in the previous full version) ...
        for metric_name, run_metric_flag in umap_metric_flags.items():
            if not run_metric_flag: continue
            
            umap_label_prefix = f"UMAP-{metric_name.capitalize()}"
            umap_cols = [f'{umap_label_prefix}-{i+1}' for i in range(simspace_dim)]
            logging.info(f"Performing UMAP ({metric_name}) to {simspace_dim}D on Main data...")
            try:
                if CUML_AVAILABLE:
                    umap_model = cumlUMAP(n_neighbors=15, min_dist=0.1, n_components=simspace_dim, metric=metric_name, random_state=42, verbose=False)
                elif SKLEARN_UMAP_AVAILABLE:
                    umap_model = umapUMAP(n_neighbors=15, min_dist=0.1, n_components=simspace_dim, metric=metric_name, random_state=42, verbose=False)
                else:
                    logging.warning(f"No UMAP library for {metric_name}. Skipping.")
                    for col in umap_cols: df_results_main[col] = np.nan
                    continue
                
                umap_res = umap_model.fit_transform(X_scaled_main)
                for i in range(simspace_dim): df_results_main[umap_cols[i]] = umap_res[:, i]
                
                umap_model_path = os.path.join(output_model_dir, f"{base_name_prefix}_{metric_name}_UMAP_model.lzma")
                with open(umap_model_path, "wb") as f: dump(umap_model, f)
                logging.info(f"Saved UMAP model ({metric_name}) to {umap_model_path}")
            except Exception as e:
                logging.error(f"UMAP ({metric_name}) failed: {e}")
                for col in umap_cols: df_results_main[col] = np.nan
            gc.collect()


    # --- t-SNE (co-embedding Main data + Target Ligands) ---
    if dr_method_flags.get('tsne'):
        logging.info("t-SNE flag is True. Checking conditions...")
        if simspace_dim == 2:
            logging.info(f"Attempting t-SNE as simspace_dim is 2 (Perp: {tsne_params['perplexity']}, InitPCA: {tsne_params['pca_components']})...")
            X_for_tsne_combined = X_scaled_main.copy() # Start with main scaled data
            num_main_data_points = len(X_scaled_main)
            df_target_ligands_unscaled_for_tsne = None
            
            if not target_ligands_unscaled_path_for_tsne or not os.path.exists(target_ligands_unscaled_path_for_tsne):
                logging.warning(f"Target ligands file for t-SNE not provided or not found: '{target_ligands_unscaled_path_for_tsne}'. t-SNE will run on main data only.")
            else:
                try:
                    logging.info(f"Loading target ligands for t-SNE from: {target_ligands_unscaled_path_for_tsne}")
                    df_target_ligands_unscaled_for_tsne_raw = pd.read_csv(target_ligands_unscaled_path_for_tsne, low_memory=False)
                    
                    # Target FPs are already parsed by calc_feat_fp by the time they are written to this file
                    # So is_parsed_fp should be True if representation_type is "fingerprints"
                    target_desc_cols = get_descriptor_columns(df_target_ligands_unscaled_for_tsne_raw,
                                                              representation_type, target_rdkit_features_list,
                                                              is_parsed_fp=(representation_type == "fingerprints"))
                    
                    if target_desc_cols:
                        logging.info(f"Identified {len(target_desc_cols)} descriptor columns in target ligands for t-SNE: {target_desc_cols[:5]}...")
                        df_target_ligands_unscaled_for_tsne = df_target_ligands_unscaled_for_tsne_raw.copy()
                        df_target_ligands_unscaled_for_tsne[target_desc_cols] = df_target_ligands_unscaled_for_tsne[target_desc_cols].apply(pd.to_numeric, errors='coerce')
                        original_target_rows = len(df_target_ligands_unscaled_for_tsne)
                        df_target_ligands_unscaled_for_tsne.dropna(subset=target_desc_cols, how='any', inplace=True)
                        if len(df_target_ligands_unscaled_for_tsne) < original_target_rows:
                            logging.info(f"Dropped {original_target_rows - len(df_target_ligands_unscaled_for_tsne)} rows from target ligands (t-SNE) due to NaNs in descriptors.")
                        
                        if not df_target_ligands_unscaled_for_tsne.empty:
                            logging.info(f"Target ligands for t-SNE shape after NaN drop: {df_target_ligands_unscaled_for_tsne.shape}")
                            X_target_unscaled_values = df_target_ligands_unscaled_for_tsne[target_desc_cols].values.astype(np.float32)
                            X_target_scaled_values = scaler.transform(X_target_unscaled_values)
                            
                            X_for_tsne_combined = np.vstack((X_scaled_main, X_target_scaled_values))
                            logging.info(f"Co-embedding {num_main_data_points} main points and {len(X_target_scaled_values)} target ligands for t-SNE. Combined shape: {X_for_tsne_combined.shape}")
                        else:
                            logging.warning("Target ligands DataFrame empty after NaN drop for t-SNE. t-SNE will run on main data only.")
                            df_target_ligands_unscaled_for_tsne = None # Ensure it's None if empty
                    else:
                        logging.warning("Could not get descriptor columns for target ligands for t-SNE. t-SNE will run on main data only.")
                        df_target_ligands_unscaled_for_tsne = None
                except Exception as e:
                    logging.error(f"Error loading/processing target ligands for t-SNE: {e}. t-SNE will run on main data only.")
                    df_target_ligands_unscaled_for_tsne = None

            tsne_cols = [f't-SNE-{i+1}' for i in range(simspace_dim)] # simspace_dim is 2 here
            try:
                actual_pca_components_for_tsne = min(tsne_params['pca_components'], X_for_tsne_combined.shape[0]-1, X_for_tsne_combined.shape[1])
                if actual_pca_components_for_tsne < 2: # PCA needs at least 2 components to be meaningful, and t-SNE needs some variance
                     logging.error(f"Cannot run t-SNE: too few samples/features for initial PCA. Need at least 2 effective components, got {actual_pca_components_for_tsne} from data shape {X_for_tsne_combined.shape} and config {tsne_params['pca_components']}.")
                     for col in tsne_cols: df_results_main[col] = np.nan
                else:
                    logging.info(f"t-SNE Step 1: Initial PCA to {actual_pca_components_for_tsne} components (data shape: {X_for_tsne_combined.shape}).")
                    pca_for_tsne = (cumlPCA(n_components=actual_pca_components_for_tsne, random_state=42) if CUML_AVAILABLE
                                    else sklearnPCA(n_components=actual_pca_components_for_tsne, random_state=42))
                    X_tsne_pca_reduced = pca_for_tsne.fit_transform(X_for_tsne_combined)

                    pca_for_tsne_model_path = os.path.join(output_model_dir, f"{base_name_prefix}_tSNE_internal_PCA_model.lzma")
                    with open(pca_for_tsne_model_path, "wb") as f: dump(pca_for_tsne, f)
                    logging.info(f"Saved t-SNE's internal PCA model to {pca_for_tsne_model_path}")

                    logging.info(f"t-SNE Step 2: t-SNE to {simspace_dim}D (Perp: {tsne_params['perplexity']}). Input PCA-reduced shape: {X_tsne_pca_reduced.shape}")
                    # For sklearn TSNE, init='pca' is often good. method='exact' for small N, 'barnes_hut' for larger N.
                    tsne_model = (cumlTSNE(n_components=simspace_dim, perplexity=tsne_params['perplexity'], random_state=42, method='barnes_hut', verbose=logging.DEBUG) if CUML_AVAILABLE # Changed verbose level
                                else sklearnTSNE(n_components=simspace_dim, perplexity=tsne_params['perplexity'], random_state=42, init='pca', method='barnes_hut', n_jobs=-1, verbose=1 if logger.isEnabledFor(logging.DEBUG) else 0)) # Link verbose to logger
                    
                    tsne_embedding_combined = tsne_model.fit_transform(X_tsne_pca_reduced)
                    logging.info(f"t-SNE fit_transform completed. Output shape: {tsne_embedding_combined.shape}")

                    tsne_embedding_main = tsne_embedding_combined[:num_main_data_points]
                    for i in range(simspace_dim): df_results_main[tsne_cols[i]] = tsne_embedding_main[:, i]

                    if df_target_ligands_unscaled_for_tsne is not None and not df_target_ligands_unscaled_for_tsne.empty and \
                    len(tsne_embedding_combined) > num_main_data_points:
                        tsne_embedding_target = tsne_embedding_combined[num_main_data_points:]
                        
                        # Ensure indices match between target_ligands_unscaled_for_tsne and the target portion of embedding
                        if len(tsne_embedding_target) == len(df_target_ligands_unscaled_for_tsne):
                            df_tsne_target_coords = pd.DataFrame(tsne_embedding_target, columns=tsne_cols, index=df_target_ligands_unscaled_for_tsne.index)
                            
                            # Include key ID columns from the original target df
                            id_cols_to_include_tsne = [col for col in ['SMILES', 'Compound ChEMBL ID', 'Activity Type', 'Standard Value (nM)', 'accession'] if col in df_target_ligands_unscaled_for_tsne.columns]
                            df_target_ids_for_tsne_output = df_target_ligands_unscaled_for_tsne[id_cols_to_include_tsne]
                            
                            df_target_ligands_projected_tsne_with_ids = df_target_ids_for_tsne_output.join(df_tsne_target_coords, how="inner")

                            tsne_target_proj_path = os.path.join(output_simspace_dir, f"{base_name_prefix}_tSNE_TARGET_PROJECTIONS.csv")
                            df_target_ligands_projected_tsne_with_ids.to_csv(tsne_target_proj_path, index=False)
                            logging.info(f"Saved t-SNE projections for TARGET LIGANDS (with IDs) to {tsne_target_proj_path} ({len(df_target_ligands_projected_tsne_with_ids)} records)")
                        else:
                            logging.error(f"t-SNE Mismatch: Target embedding length {len(tsne_embedding_target)} != df_target_ligands length {len(df_target_ligands_unscaled_for_tsne)}. Cannot save target t-SNE projections.")
                    else:
                        logging.info("No target ligands were co-embedded or available for t-SNE output.")
                    logging.info(f"t-SNE completed successfully.")
            except Exception as e:
                logging.error(f"t-SNE processing FAILED: {e}", exc_info=True) # Add exc_info for traceback in log
                for col in tsne_cols: df_results_main[col] = np.nan # Ensure columns exist
            gc.collect()
        else:
            logging.info(f"Skipping t-SNE calculation because simspace_dim is {simspace_dim} (t-SNE is configured to run only for 2D).")
    else:
        logging.info("t-SNE flag is False. Skipping t-SNE.")

    # --- Save Comprehensive Similarity Space CSV (for main data) ---
    output_csv_name = f"{base_name_prefix}_similarity_space.csv"
    output_csv_path = os.path.join(output_simspace_dir, output_csv_name)
    logging.info(f"Saving comprehensive similarity space for Main data to {output_csv_path}...")
    
    # Reconstruct with original indices for df_main_combined before dropna, then select relevant rows
    info_cols = [col for col in df_main_combined.columns if col not in descriptor_columns] 
    dr_cols = [col for col in df_results_main.columns if col.startswith(('PCA-', 'UMAP-', 't-SNE-'))]
    final_output_cols = info_cols + dr_cols
    
    cols_to_save = [col for col in final_output_cols if col in df_results_main.columns]
    
    try:
        df_results_main[cols_to_save].to_csv(output_csv_path, index=False)
        logging.info(f"Saved main data similarity space: {output_csv_path} (shape: {df_results_main[cols_to_save].shape})")
    except Exception as e:
        logging.error(f"Failed to save similarity space file {output_csv_path}: {e}")
    
    del df_main_combined, df_results_main, X_scaled_main, X_original_main
    gc.collect()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate similarity spaces.")
    parser.add_argument("--chembl_mf_data_path", required=True)
    parser.add_argument("--zinc_data_path", default=None) 
    parser.add_argument("--target_ligands_unscaled_path_for_tsne", default=None)
    parser.add_argument("--simspace_dim", type=int, required=True)
    parser.add_argument("--representation_type", required=True, choices=["features", "fingerprints"])
    parser.add_argument("--target_id_name", required=True)
    parser.add_argument("--output_simspace_dir", required=True)
    parser.add_argument("--output_model_dir", required=True)
    parser.add_argument("--rdkit_features_list_target_str", required=True)

    parser.add_argument("--dr_method_pca", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--dr_method_umap", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--dr_method_tsne", type=lambda x: (str(x).lower() == 'true'), default=False)
    
    parser.add_argument("--umap_metric_to_run_euclidean", action='store_true', default=False)
    parser.add_argument("--umap_metric_to_run_cosine", action='store_true', default=False)
    parser.add_argument("--umap_metric_to_run_manhattan", action='store_true', default=False)
    parser.add_argument("--umap_metric_to_run_hamming", action='store_true', default=False)
    
    parser.add_argument("--tsne_perplexity", type=float, default=30.0)
    parser.add_argument("--tsne_pca_components", type=int, default=50)
    
    args = parser.parse_args()
    logging.info(f"Parsed arguments: {args}")

    try:
        target_rdkit_features_list = json.loads(args.rdkit_features_list_target_str)
    except Exception as e:
        logging.error(f"Error parsing --rdkit_features_list_target_str: {e}. Aborting.")
        exit(1)

    os.makedirs(args.output_simspace_dir, exist_ok=True)
    os.makedirs(args.output_model_dir, exist_ok=True)

    dr_method_flags_dict = {'pca': args.dr_method_pca, 'umap': args.dr_method_umap, 'tsne': args.dr_method_tsne}
    active_umap_metrics = {}
    if args.dr_method_umap: # Only populate if UMAP is actually run
        if args.umap_metric_to_run_euclidean: active_umap_metrics['euclidean'] = True
        if args.umap_metric_to_run_cosine: active_umap_metrics['cosine'] = True
        if args.umap_metric_to_run_manhattan: active_umap_metrics['manhattan'] = True
        if args.umap_metric_to_run_hamming: active_umap_metrics['hamming'] = True
        if not active_umap_metrics: # If --dr_method_umap is true but no metrics specified, maybe default to euclidean or error
            logging.warning("UMAP method requested but no specific UMAP metrics specified via --umap_metric_to_run_*. Defaulting to Euclidean if UMAP flag is true but no metrics given, or add specific flags.")
            # Forcing at least one if dr_method_umap is true but no metrics given by user
            # active_umap_metrics['euclidean'] = True # Example default
            # Better to let it proceed and fail if no metrics are truly active from flags.

    tsne_params_dict = {'perplexity': args.tsne_perplexity, 'pca_components': args.tsne_pca_components}

    process_similarity_calculations(
        args.chembl_mf_data_path, args.zinc_data_path, args.target_ligands_unscaled_path_for_tsne,
        args.simspace_dim, args.representation_type, args.target_id_name,
        args.output_simspace_dir, args.output_model_dir, target_rdkit_features_list,
        dr_method_flags_dict, active_umap_metrics, tsne_params_dict
    )
    logging.info("Similarity space calculation script finished.")