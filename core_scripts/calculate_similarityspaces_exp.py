import os
import numpy as np
import pandas as pd
import logging 
import gc
from compress_pickle import dump 
from sklearn.preprocessing import StandardScaler
import argparse
import json 
import time 

# CUML / Sklearn imports
CUML_AVAILABLE = False
SKLEARN_UMAP_AVAILABLE = False 

try:
    from cuml import PCA as cumlPCA, UMAP as cumlUMAP, TSNE as cumlTSNE
    CUML_AVAILABLE = True
except ImportError:
    pass 

from sklearn.decomposition import PCA as sklearnPCA
from sklearn.manifold import TSNE as sklearnTSNE
try:
    from umap import UMAP as umapUMAP 
    SKLEARN_UMAP_AVAILABLE = True
except ImportError:
    pass 

# --- Logging Setup ---
logger = logging.getLogger() 
if logger.hasHandlers(): 
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
# Root logger level must be the lowest level you want any handler to see.
logger.setLevel(logging.DEBUG) 

# Corrected formatter
log_formatter = logging.Formatter('%(asctime)s - %(levelname)-8s - %(filename)-25s - %(funcName)-25s - %(lineno)-4d - %(message)s')

try:
    cs_log_file_path = "calculate_simspaces_internal.log" 
    file_handler = logging.FileHandler(cs_log_file_path, mode='w') 
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.DEBUG) # File handler will log DEBUG and above
    logger.addHandler(file_handler)
except Exception as e:
    print(f"CRITICAL: Failed to initialize file logger for {cs_log_file_path}: {e}")

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_formatter)
stream_handler.setLevel(logging.INFO) # Console can be less verbose
logger.addHandler(stream_handler)
# --- End Logging Setup ---

FINGERPRINT_COLUMN_PREFIX = "fp_"
NUM_FINGERPRINT_BITS = 2048 
PRECALCULATED_FP_STRING_COLUMN_NAME = "Fingerprint"

def parse_fingerprint_string_column_in_df(df_input, 
                                          fp_string_col_name=PRECALCULATED_FP_STRING_COLUMN_NAME, 
                                          num_bits=NUM_FINGERPRINT_BITS):
    logger.info(f"Attempting to parse fingerprint string column: '{fp_string_col_name}' in df of shape {df_input.shape}")
    if fp_string_col_name not in df_input.columns:
        logger.warning(f"Fingerprint string column '{fp_string_col_name}' not found. No parsing done.")
        return df_input 
    fp_matrix = np.full((len(df_input), num_bits), np.nan) 
    num_parse_errors = 0; num_length_mismatches = 0
    for idx, fp_str in enumerate(df_input[fp_string_col_name]):
        if pd.isna(fp_str) or not isinstance(fp_str, str) or not fp_str.strip(): continue
        bits = fp_str.split(',')
        if len(bits) == num_bits:
            try: fp_matrix[idx, :] = [int(b) for b in bits]
            except ValueError: 
                if num_parse_errors < 5: logger.debug(f"ValueError parsing bits (row index {df_input.index[idx]}): {fp_str[:30]}...")
                num_parse_errors += 1
        else:
            if num_length_mismatches < 5: logger.debug(f"FP string (row index {df_input.index[idx]}) '{fp_str[:30]}...' length {len(bits)} != expected {num_bits}.")
            num_length_mismatches +=1
    if num_parse_errors > 0: logger.warning(f"Encountered {num_parse_errors} ValueErrors during FP string parsing.")
    if num_length_mismatches > 0: logger.warning(f"Encountered {num_length_mismatches} length mismatches during FP string parsing.")
    fp_col_names = [f"{FINGERPRINT_COLUMN_PREFIX}{i}" for i in range(num_bits)]
    df_fp_bits = pd.DataFrame(fp_matrix, columns=fp_col_names, index=df_input.index)
    df_output = df_input.drop(columns=[fp_string_col_name])
    df_output = pd.concat([df_output, df_fp_bits], axis=1)
    logger.info(f"FP string column '{fp_string_col_name}' parsed. Added {len(fp_col_names)} new columns. New shape: {df_output.shape}")
    return df_output

def get_descriptor_columns(df, representation_type, target_rdkit_features_list, is_parsed_fp=False):
    if representation_type == "features":
        if not target_rdkit_features_list: 
            logger.error("Target RDKit features list is empty or None.")
            return None
        descriptor_cols = [col for col in target_rdkit_features_list if col in df.columns]
        missing_config_feats = [col for col in target_rdkit_features_list if col not in df.columns]
        if missing_config_feats: logger.warning(f"Configured RDKit features missing from input: {missing_config_feats}.")
        if not descriptor_cols:
            logger.error(f"No features from RDKIT_FEATURES_LIST_TARGET found in DF columns (first 10): {df.columns.tolist()[:10]}")
            return None
        return descriptor_cols
    elif representation_type == "fingerprints":
        if is_parsed_fp: 
            descriptor_cols = [f"{FINGERPRINT_COLUMN_PREFIX}{i}" for i in range(NUM_FINGERPRINT_BITS)]
            actual_cols = [col for col in descriptor_cols if col in df.columns]
            if not actual_cols:
                 logger.error(f"FP repr selected, but no '{FINGERPRINT_COLUMN_PREFIX}*' columns found after expected parsing.")
                 return None
            if len(actual_cols) != NUM_FINGERPRINT_BITS:
                 logger.warning(f"Expected {NUM_FINGERPRINT_BITS} FP columns, found {len(actual_cols)}. Using found columns.")
            return actual_cols 
        else: 
            if PRECALCULATED_FP_STRING_COLUMN_NAME not in df.columns:
                logger.error(f"FP repr selected, but expected string column '{PRECALCULATED_FP_STRING_COLUMN_NAME}' not found.")
                return None
            return [PRECALCULATED_FP_STRING_COLUMN_NAME] 
    else:
        logger.error(f"Invalid representation_type: {representation_type}")
        return None

def load_and_prepare_target_ligands_for_coembedding(
    target_ligands_unscaled_path, representation_type, target_rdkit_features_list, scaler_main_data
):
    if not target_ligands_unscaled_path or target_ligands_unscaled_path.lower() == "none" or not os.path.exists(target_ligands_unscaled_path):
        logger.warning(f"Target ligands file for co-embedding not provided, 'None', or not found: '{target_ligands_unscaled_path}'.")
        return None, None, None 
    try:
        logging.info(f"Loading target ligands for co-embedding from: {target_ligands_unscaled_path}")
        df_target_ligands_raw = pd.read_csv(target_ligands_unscaled_path, low_memory=False)
        if df_target_ligands_raw.empty:
            logger.warning(f"Target ligands file '{target_ligands_unscaled_path}' is empty.")
            return None, None, None
        
        target_desc_cols = get_descriptor_columns(df_target_ligands_raw,
                                                  representation_type, target_rdkit_features_list,
                                                  is_parsed_fp=(representation_type == "fingerprints"))
        if not target_desc_cols:
            logger.error(f"Could not get descriptor columns from target ligands file: {target_ligands_unscaled_path}.")
            return None, None, None

        df_target_ligands_processed = df_target_ligands_raw.copy()
        df_target_ligands_processed[target_desc_cols] = df_target_ligands_processed[target_desc_cols].apply(pd.to_numeric, errors='coerce')
        original_target_rows = len(df_target_ligands_processed)
        df_target_ligands_processed.dropna(subset=target_desc_cols, how='any', inplace=True)

        if len(df_target_ligands_processed) < original_target_rows:
            logging.info(f"Dropped {original_target_rows - len(df_target_ligands_processed)} target ligands (co-embedding) due to NaNs.")
        if df_target_ligands_processed.empty:
            logger.warning("Target ligands DataFrame empty after NaN drop for co-embedding.")
            return None, None, None
            
        dtype_to_use = np.int8 if representation_type == "fingerprints" else np.float32
        X_target_unscaled_numeric = df_target_ligands_processed[target_desc_cols].values.astype(dtype_to_use)

        X_target_scaled_numeric = scaler_main_data.transform(X_target_unscaled_numeric)
        return df_target_ligands_processed, X_target_scaled_numeric, X_target_unscaled_numeric
    except Exception as e:
        logger.error(f"Error loading/processing target ligands from '{target_ligands_unscaled_path}': {e}", exc_info=True)
        return None, None, None

def process_similarity_calculations(
    chembl_mf_data_path, zinc_data_path, target_ligands_unscaled_path_for_tsne_and_coembed,
    simspace_dim, representation_type, target_id_name,
    output_simspace_dir, output_model_dir, target_rdkit_features_list,
    dr_method_flags, umap_metric_flags, tsne_params,
    run_coembedding_for_pca_umap, dr_method_configs,
    current_random_state 
    ):

    # This logging setup is now done inside the orchestrator's main()
    # For standalone, a basic config is fine. The orchestrator's log is primary.
    # Re-doing it here to be safe if run standalone.
    # The setup_script_logging function is not defined here, let's keep it simple.
    # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)-8s - %(message)s')

    logger.info(f"--- process_similarity_calculations: Random_State: {current_random_state} ---")
    if CUML_AVAILABLE: logger.info("cuML is available.")
    else: logger.info("cuML is NOT available.")
    if SKLEARN_UMAP_AVAILABLE: logger.info("sklearn UMAP (umap-learn) is available.")
    else: logger.warning("sklearn UMAP (umap-learn) is NOT available. CPU UMAP will fail if attempted.")

    base_name_prefix = f"{target_id_name}_{representation_type}_dim{simspace_dim}"
    logging.info(f"Processing for: {base_name_prefix}")
    
    try:
        df_chembl_mf = pd.read_csv(chembl_mf_data_path, low_memory=False)
        logging.info(f"Loaded ChEMBL MF data: {chembl_mf_data_path}, shape: {df_chembl_mf.shape}")
        df_zinc = pd.read_csv(zinc_data_path, low_memory=False) if zinc_data_path and zinc_data_path.lower() != 'none' and os.path.exists(zinc_data_path) else pd.DataFrame()
        if not df_zinc.empty: logging.info(f"Loaded ZINC data: {zinc_data_path}, shape: {df_zinc.shape}")
        else: logging.info("No ZINC data or path invalid. Proceeding without ZINC.")
    except Exception as e: 
        logging.error(f"Failed to load ChEMBL/ZINC data for {base_name_prefix}: {e}", exc_info=True); return

    if representation_type == "fingerprints":
        if PRECALCULATED_FP_STRING_COLUMN_NAME in df_chembl_mf.columns: 
            df_chembl_mf = parse_fingerprint_string_column_in_df(df_chembl_mf)
        if not df_zinc.empty and PRECALCULATED_FP_STRING_COLUMN_NAME in df_zinc.columns: 
            df_zinc = parse_fingerprint_string_column_in_df(df_zinc)
    
    # --- FIX: Standardize Columns BEFORE Concatenation ---
    temp_df_for_cols = df_chembl_mf if not df_chembl_mf.empty else df_zinc
    descriptor_columns = get_descriptor_columns(temp_df_for_cols, representation_type, 
                                                target_rdkit_features_list, 
                                                is_parsed_fp=(representation_type == "fingerprints"))
    if not descriptor_columns: 
        logging.error(f"No descriptor columns determined for {representation_type}. Aborting {base_name_prefix}."); return

    info_cols_to_keep = ['SMILES', 'Compound ChEMBL ID', 'ZINC_ID']
    
    df_chembl_mf_std = pd.DataFrame()
    if not df_chembl_mf.empty:
        cols_to_keep_chembl = [col for col in info_cols_to_keep if col in df_chembl_mf.columns] + \
                              [col for col in descriptor_columns if col in df_chembl_mf.columns]
        df_chembl_mf_std = df_chembl_mf[cols_to_keep_chembl].copy()

    df_zinc_std = pd.DataFrame()
    if not df_zinc.empty:
        cols_to_keep_zinc = [col for col in info_cols_to_keep if col in df_zinc.columns] + \
                            [col for col in descriptor_columns if col in df_zinc.columns]
        df_zinc_std = df_zinc[cols_to_keep_zinc].copy()

    logging.info(f"Standardized ChEMBL df shape: {df_chembl_mf_std.shape}")
    logging.info(f"Standardized ZINC df shape: {df_zinc_std.shape}")

    df_main_combined = pd.concat([df_chembl_mf_std, df_zinc_std], ignore_index=True)
    logging.info(f"Combined Main DataFrame shape after standardizing columns: {df_main_combined.shape}")
    # --- END OF FIX ---
    
    if 'Compound ChEMBL ID' in df_main_combined.columns and 'ZINC_ID' in df_main_combined.columns: 
        df_main_combined['MOLECULE ID'] = df_main_combined['Compound ChEMBL ID'].fillna(df_main_combined['ZINC_ID'])
    elif 'Compound ChEMBL ID' in df_main_combined.columns: df_main_combined['MOLECULE ID'] = df_main_combined['Compound ChEMBL ID']
    elif 'ZINC_ID' in df_main_combined.columns: df_main_combined['MOLECULE ID'] = df_main_combined['ZINC_ID']
    else: 
        if 'SMILES' in df_main_combined.columns: df_main_combined['MOLECULE ID'] = df_main_combined['SMILES']
        else: df_main_combined['MOLECULE ID'] = 'UNKNOWN_ID_' + pd.Series(df_main_combined.index).astype(str)
    
    logging.info(f"MOLECULE ID column created. Shape: {df_main_combined.shape}")
    df_main_combined['MOLECULE ID'] = df_main_combined['MOLECULE ID'].astype(str).fillna('MISSING_ID')
    logging.info(f"MOLECULE ID column filled NaNs with 'MISSING_ID'. Shape: {df_main_combined.shape}")

    # --- REVISED/OPTIMIZED DATA PREP FLOW ---
    logging.info("Starting final data preparation for DR...")
    if representation_type == "features":
        logging.info("Applying pd.to_numeric to feature columns...")
        # Iterating is more memory-friendly than a single massive .apply()
        for col in descriptor_columns:
            df_main_combined[col] = pd.to_numeric(df_main_combined[col], errors='coerce')
        logging.info("Completed converting feature columns to numeric.")
    else:
        logging.info("Skipping redundant to_numeric for fingerprints as parsing already handled it.")

    original_rows_main_before_dropna = len(df_main_combined)
    logging.info(f"Original Main DataFrame rows before NaN drop: {original_rows_main_before_dropna}")
    
    df_results_main = df_main_combined.dropna(subset=descriptor_columns, how='any').copy()
    logging.info(f"Main DataFrame after NaN drop on descriptors: {df_results_main.shape}")
    
    del df_main_combined
    gc.collect()
    logging.info("Released memory from intermediate combined DataFrame.")

    if len(df_results_main) < original_rows_main_before_dropna: 
        logging.info(f"Dropped {original_rows_main_before_dropna - len(df_results_main)} rows due to NaNs in descriptors.")
    if df_results_main.empty: 
        logging.error(f"Main DataFrame (df_results_main) empty after NaN drop for {base_name_prefix}. Cannot proceed."); return
    
    dtype_to_use = np.int8 if representation_type == "fingerprints" else np.float32
    logging.info(f"Converting descriptor columns to NumPy array with dtype: {dtype_to_use}...")
    X_original_main_valid = df_results_main[descriptor_columns].values.astype(dtype_to_use)
    logging.info(f"X_original_main_valid shape: {X_original_main_valid.shape}, dtype: {X_original_main_valid.dtype}")
    # --- END REVISED FLOW ---
    
    scaler = StandardScaler()
    logging.info(f"Fitting StandardScaler on main data of shape {X_original_main_valid.shape}") 
    X_scaled_main = scaler.fit_transform(X_original_main_valid)
    logging.info(f"StandardScaler fitted. X_scaled_main shape: {X_scaled_main.shape}")
    scaler_model_path = os.path.join(output_model_dir, f"{base_name_prefix}_scaler.lzma")
    try: 
        with open(scaler_model_path, "wb") as f: dump(scaler, f)
        logging.info(f"Saved scaler model: {scaler_model_path}")
    except Exception as e: logging.error(f"Failed to save scaler model {scaler_model_path}: {e}")

    df_target_ligands_for_coembed_info, X_target_scaled_for_coembed, X_target_original_for_coembed = None, None, None
    needs_coembed_data = dr_method_flags.get('tsne') or (run_coembedding_for_pca_umap and (dr_method_flags.get('pca') or dr_method_flags.get('umap')))
    if needs_coembed_data:
        df_target_ligands_for_coembed_info, X_target_scaled_for_coembed, X_target_original_for_coembed = \
            load_and_prepare_target_ligands_for_coembedding( target_ligands_unscaled_path_for_tsne_and_coembed,
                representation_type, target_rdkit_features_list, scaler )
        if not (X_target_scaled_for_coembed is not None and X_target_original_for_coembed is not None and df_target_ligands_for_coembed_info is not None):
            logging.warning("Failed to fully load/process target ligands for co-embedding. Some co-embedding steps will be skipped.")
        else: logging.info(f"Prepared {X_target_scaled_for_coembed.shape[0]} target ligands for co-embedding.")

    X_main_pre_reduced_for_manifold = None
    X_target_pre_reduced_for_manifold_coembed = None
    if representation_type == "fingerprints":
        n_pca_components_for_manifold = tsne_params.get('pca_components', 50)
        logging.info(f"Fingerprints detected. Applying initial PCA to {n_pca_components_for_manifold} components before UMAP/t-SNE.")
        n_effective_pca_components = min(n_pca_components_for_manifold, X_scaled_main.shape[0] - 1 if X_scaled_main.shape[0]>1 else 1, X_scaled_main.shape[1])
        if n_effective_pca_components < 2:
            logging.error(f"Cannot perform pre-reduction PCA, effective components ({n_effective_pca_components}) is too low. Using full-dimensional scaled data for UMAP/t-SNE.")
            X_main_pre_reduced_for_manifold = X_scaled_main
        else:
            try:
                pca_for_manifold_model = (cumlPCA(n_components=n_effective_pca_components, random_state=current_random_state) 
                                          if CUML_AVAILABLE 
                                          else sklearnPCA(n_components=n_effective_pca_components, random_state=current_random_state))
                X_main_pre_reduced_for_manifold = pca_for_manifold_model.fit_transform(X_scaled_main)
                logging.info(f"Pre-reduction PCA complete for main data. Shape: {X_main_pre_reduced_for_manifold.shape}")
                if run_coembedding_for_pca_umap and X_target_scaled_for_coembed is not None:
                    X_target_pre_reduced_for_manifold_coembed = pca_for_manifold_model.transform(X_target_scaled_for_coembed)
                    logging.info(f"Pre-reduction PCA complete for target co-embedding data. Shape: {X_target_pre_reduced_for_manifold_coembed.shape}")
            except Exception as e:
                logging.error(f"Failed to perform pre-reduction PCA for UMAP/t-SNE: {e}", exc_info=True)
                X_main_pre_reduced_for_manifold = X_scaled_main
                X_target_pre_reduced_for_manifold_coembed = X_target_scaled_for_coembed
    else:
        logging.info("Features detected. UMAP will run on original scaled feature space.")
        X_main_pre_reduced_for_manifold = X_scaled_main
        X_target_pre_reduced_for_manifold_coembed = X_target_scaled_for_coembed

    if dr_method_flags.get('pca'):
        pca_config = dr_method_configs.get('pca', {})
        pca_cols = [f'PCA-{i+1}' for i in range(simspace_dim)]
        try: 
            logging.info(f"NON-CO-EMBEDDED PCA on data shape {X_scaled_main.shape}")
            pca_model_main = (cumlPCA(n_components=simspace_dim, random_state=current_random_state) if CUML_AVAILABLE else sklearnPCA(n_components=simspace_dim, random_state=current_random_state))
            pca_res_main = pca_model_main.fit_transform(X_scaled_main)
            if pca_res_main.shape[0] == len(df_results_main):
                for i in range(simspace_dim): df_results_main[pca_cols[i]] = pca_res_main[:, i]
                pca_model_path = os.path.join(output_model_dir, f"{base_name_prefix}_PCA_model.lzma")
                with open(pca_model_path, "wb") as f: dump(pca_model_main, f)
                logging.info(f"Saved NON-CO PCA model: {pca_model_path}")
            else: logging.error(f"PCA projection length mismatch: {pca_res_main.shape[0]} vs {len(df_results_main)}")
        except Exception as e: 
            logging.error(f"NON-CO PCA failed: {e}", exc_info=True)
            for col in pca_cols: 
                if col not in df_results_main: df_results_main[col] = np.nan
        gc.collect()
        
        if run_coembedding_for_pca_umap and pca_config.get("allow_coembedding", False) and X_target_scaled_for_coembed is not None and df_target_ligands_for_coembed_info is not None:
            try: 
                X_for_pca_coembed = np.vstack((X_scaled_main, X_target_scaled_for_coembed))
                logging.info(f"CO-EMBEDDED PCA on data shape {X_for_pca_coembed.shape}")
                pca_model_co = (cumlPCA(n_components=simspace_dim, random_state=current_random_state) if CUML_AVAILABLE else sklearnPCA(n_components=simspace_dim, random_state=current_random_state))
                pca_res_co = pca_model_co.fit_transform(X_for_pca_coembed)
                pca_res_target_co = pca_res_co[X_scaled_main.shape[0]:]
                if len(pca_res_target_co) == len(df_target_ligands_for_coembed_info):
                    df_pca_target_co_coords = pd.DataFrame(pca_res_target_co, columns=pca_cols, index=df_target_ligands_for_coembed_info.index)
                    id_cols = [c for c in ['SMILES','Compound ChEMBL ID','Activity Type','Standard Value (nM)','accession'] if c in df_target_ligands_for_coembed_info]
                    df_target_co_ids = df_target_ligands_for_coembed_info[id_cols]
                    df_target_pca_co_full = df_target_co_ids.join(df_pca_target_co_coords, how="inner")
                    pca_co_path = os.path.join(output_simspace_dir, f"{base_name_prefix}_PCA_COEMBED_TARGET_PROJECTIONS.csv")
                    df_target_pca_co_full.to_csv(pca_co_path, index=False); logging.info(f"Saved CO-EMBEDDED PCA target projections: {pca_co_path}")
                else: logging.error("PCA co-embed length mismatch.")
            except Exception as e: logging.error(f"CO-EMBEDDED PCA failed: {e}", exc_info=True)
            gc.collect()

    if dr_method_flags.get('umap'):
        for metric_name, run_metric_flag in umap_metric_flags.items():
            if not run_metric_flag: continue
            
            umap_config = dr_method_configs.get(f"umap_{metric_name}", {})
            umap_cols = [f"UMAP-{metric_name.capitalize()}-{i+1}" for i in range(simspace_dim)]
            
            current_X_main_umap = X_scaled_main
            current_X_target_coembed_umap = X_target_scaled_for_coembed
            attempt_cuml_umap = CUML_AVAILABLE 

            if representation_type == "fingerprints":
                logging.info(f"For fingerprints, UMAP ({metric_name}) will run on the PCA pre-reduced data.")
                current_X_main_umap = X_main_pre_reduced_for_manifold
                current_X_target_coembed_umap = X_target_pre_reduced_for_manifold_coembed
                if metric_name.lower() in ["hamming", "manhattan"]:
                    attempt_cuml_umap = False 
                    logging.info(f"Forcing scikit-learn UMAP for '{metric_name}' on fingerprints (after PCA).")
            
            umap_model_main_for_projection = None 
            try: 
                if attempt_cuml_umap: 
                    logging.info(f"Attempting cuML UMAP for {metric_name} (projection).")
                    umap_model_main_for_projection = cumlUMAP(n_components=simspace_dim, metric=metric_name, random_state=current_random_state, n_neighbors=15, min_dist=0.1, verbose=False)
                elif SKLEARN_UMAP_AVAILABLE: 
                    logging.info(f"Using scikit-learn UMAP for {metric_name} (projection).")
                    umap_model_main_for_projection = umapUMAP(n_components=simspace_dim, metric=metric_name, random_state=current_random_state, n_neighbors=15, min_dist=0.1, verbose=False)
                else: 
                    logging.warning(f"No suitable UMAP library for NON-CO-EMBEDDED UMAP ({metric_name}). Skipping."); 
                    for col in umap_cols: df_results_main[col] = np.nan 
                    if run_coembedding_for_pca_umap and umap_config.get("allow_coembedding", False): logging.warning(f"Skipping CO-EMBEDDED UMAP for {metric_name} as projection model cannot be built.")
                    continue
                
                logging.info(f"Fitting projection UMAP ({metric_name}) on main data of shape {current_X_main_umap.shape}")
                umap_res_main = umap_model_main_for_projection.fit_transform(current_X_main_umap)
                if umap_res_main.shape[0] == len(df_results_main):
                    for i in range(simspace_dim): df_results_main[umap_cols[i]] = umap_res_main[:, i]
                    umap_model_path = os.path.join(output_model_dir, f"{base_name_prefix}_{metric_name}_UMAP_model.lzma")
                    with open(umap_model_path, "wb") as f: dump(umap_model_main_for_projection, f) 
                    logging.info(f"Saved NON-CO-EMBEDDED UMAP model ({metric_name}) to {umap_model_path}")
                else: 
                    logging.error(f"UMAP projection result length mismatch for {metric_name}: {umap_res_main.shape[0]} vs {len(df_results_main)}")
                    for col in umap_cols: df_results_main[col] = np.nan 
            except Exception as e: 
                logging.error(f"NON-CO-EMBEDDED UMAP ({metric_name}) failed: {e}", exc_info=True)
                if attempt_cuml_umap and SKLEARN_UMAP_AVAILABLE and ("metric is not supported" in str(e).lower() or "cuML Error" in str(e) or "libcuml. Persönlicher Fehler" in str(e)):
                    logging.info(f"cuML UMAP failed for {metric_name}, trying scikit-learn UMAP (projection) as fallback...")
                    try:
                        umap_model_main_for_projection = umapUMAP(n_components=simspace_dim, metric=metric_name, random_state=current_random_state, n_neighbors=15, min_dist=0.1, verbose=False)
                        umap_res_main_fb = umap_model_main_for_projection.fit_transform(current_X_main_umap)
                        if umap_res_main_fb.shape[0] == len(df_results_main):
                            for i in range(simspace_dim): df_results_main[umap_cols[i]] = umap_res_main_fb[:, i]
                            umap_model_path_fb = os.path.join(output_model_dir, f"{base_name_prefix}_{metric_name}_UMAP_model_sklearn_fallback.lzma")
                            with open(umap_model_path_fb, "wb") as f: dump(umap_model_main_for_projection, f) 
                            logging.info(f"Saved scikit-learn UMAP model (projection fallback for {metric_name}) to {umap_model_path_fb}")
                        else: logging.error(f"UMAP projection fallback length mismatch for {metric_name}.")
                    except Exception as e_fb:
                        logging.error(f"scikit-learn UMAP (projection fallback for {metric_name}) also failed: {e_fb}", exc_info=True)
                        for col in umap_cols: df_results_main[col] = np.nan
                else:
                     for col in umap_cols: df_results_main[col] = np.nan
                if run_coembedding_for_pca_umap and umap_config.get("allow_coembedding", False): 
                    logging.warning(f"Skipping CO-EMBEDDED UMAP for {metric_name} due to projection model failure.")
                gc.collect(); continue 
            gc.collect()

            if run_coembedding_for_pca_umap and umap_config.get("allow_coembedding", False) and \
               current_X_target_coembed_umap is not None and df_target_ligands_for_coembed_info is not None:
                try:
                    X_coembed_umap = np.vstack((current_X_main_umap, current_X_target_coembed_umap))
                    logging.info(f"CO-EMBEDDED UMAP ({metric_name}) on data shape {X_coembed_umap.shape}")
                    umap_model_co = None
                    if attempt_cuml_umap: umap_model_co = cumlUMAP(n_components=simspace_dim, metric=metric_name, random_state=current_random_state, n_neighbors=15, min_dist=0.1, verbose=False)
                    elif SKLEARN_UMAP_AVAILABLE: umap_model_co = umapUMAP(n_components=simspace_dim, metric=metric_name, random_state=current_random_state, n_neighbors=15, min_dist=0.1, verbose=False)
                    else: logging.warning(f"No UMAP lib for {metric_name}. Skip co-embedding."); continue
                    
                    umap_res_co = umap_model_co.fit_transform(X_coembed_umap)
                    umap_res_target_co = umap_res_co[current_X_main_umap.shape[0]:]
                    if len(umap_res_target_co) == len(df_target_ligands_for_coembed_info):
                        df_umap_target_co_coords = pd.DataFrame(umap_res_target_co, columns=umap_cols, index=df_target_ligands_for_coembed_info.index)
                        id_cols = [c for c in ['SMILES','Compound ChEMBL ID','Activity Type','Standard Value (nM)','accession'] if c in df_target_ligands_for_coembed_info]
                        df_target_co_ids = df_target_ligands_for_coembed_info[id_cols]
                        df_target_umap_co_full = df_target_co_ids.join(df_umap_target_co_coords, how="inner")
                        umap_co_path = os.path.join(output_simspace_dir, f"{base_name_prefix}_{metric_name}_UMAP_COEMBED_TARGET_PROJECTIONS.csv")
                        df_target_umap_co_full.to_csv(umap_co_path, index=False); logging.info(f"Saved CO-EMBEDDED UMAP ({metric_name}) target projections: {umap_co_path}")
                    else: logging.error(f"UMAP co-embed length mismatch for {metric_name}.")
                except Exception as e: 
                    logging.error(f"CO-EMBEDDED UMAP ({metric_name}) failed: {e}", exc_info=True)
                    # Fallback logic for co-embedded UMAP
                    if attempt_cuml_umap and SKLEARN_UMAP_AVAILABLE and ("metric is not supported" in str(e).lower() or "cuML Error" in str(e) or "libcuml. Persönlicher Fehler" in str(e)):
                        logging.info(f"cuML CO-EMBEDDED UMAP failed for {metric_name}, trying scikit-learn UMAP (co-embedding) as fallback...")
                        try:
                            umap_model_co_fb = umapUMAP(n_components=simspace_dim, metric=metric_name, random_state=current_random_state, n_neighbors=15, min_dist=0.1, verbose=False)
                            umap_res_co_fb = umap_model_co_fb.fit_transform(X_coembed_umap) 
                            umap_res_target_co_fb = umap_res_co_fb[current_X_main_umap.shape[0]:]
                            if len(umap_res_target_co_fb) == len(df_target_ligands_for_coembed_info):
                                df_umap_target_co_coords_fb = pd.DataFrame(umap_res_target_co_fb, columns=umap_cols, index=df_target_ligands_for_coembed_info.index)
                                df_target_ids_coembed_umap_fb = df_target_ligands_for_coembed_info[id_cols] 
                                df_target_projected_umap_coembed_with_ids_fb = df_target_ids_coembed_umap_fb.join(df_umap_target_co_coords_fb, how="inner")
                                umap_target_coembed_proj_path_fb = os.path.join(output_simspace_dir, f"{base_name_prefix}_{metric_name}_UMAP_COEMBED_TARGET_PROJECTIONS_sklearn_fallback.csv")
                                df_target_projected_umap_coembed_with_ids_fb.to_csv(umap_target_coembed_proj_path_fb, index=False)
                                logging.info(f"Saved scikit-learn CO-EMBEDDED UMAP ({metric_name}) target projections (fallback) to {umap_target_coembed_proj_path_fb}")
                            else: logging.error(f"scikit-learn UMAP Co-embedding (fallback) length mismatch for metric {metric_name}.")
                        except Exception as e_co_umap_fb:
                             logging.error(f"scikit-learn CO-EMBEDDED UMAP (fallback, {metric_name}) also failed: {e_co_umap_fb}", exc_info=True)
                gc.collect()

    # --- t-SNE ---
    if dr_method_flags.get('tsne'):
        if simspace_dim == 2: 
            if X_target_scaled_for_coembed is None or df_target_ligands_for_coembed_info is None:
                logging.warning("Skipping t-SNE: target co-embedding data not fully available.")
            else:
                tsne_cols = [f't-SNE-{i+1}' for i in range(simspace_dim)]
                try:
                    X_tsne_pca_reduced = None
                    if representation_type == "fingerprints":
                        logging.info("For fingerprints, t-SNE will use the already computed pre-reduced data.")
                        if X_main_pre_reduced_for_manifold is not None and X_target_pre_reduced_for_manifold_coembed is not None:
                            X_tsne_pca_reduced = np.vstack((X_main_pre_reduced_for_manifold, X_target_pre_reduced_for_manifold_coembed))
                        else:
                            logging.error("Pre-reduced data for t-SNE (fingerprints) is missing. Skipping.")
                    else: # For features, run the original t-SNE PCA step
                        X_for_tsne_combined = np.vstack((X_scaled_main, X_target_scaled_for_coembed))
                        actual_pca_comps = min(tsne_params['pca_components'], X_for_tsne_combined.shape[0]-1 if X_for_tsne_combined.shape[0]>1 else 1, X_for_tsne_combined.shape[1])
                        if actual_pca_comps < 2:
                             logging.error(f"Cannot run PCA for t-SNE, effective components too low ({actual_pca_comps}).")
                        else:
                            logging.info(f"t-SNE initial PCA to {actual_pca_comps} components on feature data shape {X_for_tsne_combined.shape}")
                            pca_tsne = (cumlPCA(n_components=actual_pca_comps, random_state=current_random_state) if CUML_AVAILABLE else sklearnPCA(n_components=actual_pca_comps, random_state=current_random_state))
                            X_tsne_pca_reduced = pca_tsne.fit_transform(X_for_tsne_combined)
                    
                    if X_tsne_pca_reduced is not None:
                        attempt_cuml_tsne_final = CUML_AVAILABLE
                        if representation_type == "fingerprints": 
                            attempt_cuml_tsne_final = False; logging.info("Forcing scikit-learn t-SNE for fingerprints.")
                        
                        perp_final = min(float(tsne_params['perplexity']), X_tsne_pca_reduced.shape[0]-2 if X_tsne_pca_reduced.shape[0]>1 else 0.0)
                        if X_tsne_pca_reduced.shape[0] <= 1 or perp_final < 1.0: 
                            logging.error(f"Too few samples ({X_tsne_pca_reduced.shape[0]}) or invalid perplexity ({perp_final}) for t-SNE after PCA. Skipping t-SNE."); 
                            for col in tsne_cols: df_results_main[col] = np.nan
                        else:
                            if perp_final != float(tsne_params['perplexity']): logging.info(f"Adjusted t-SNE perplexity to {perp_final}")
                            tsne_model = None
                            tsne_init_kwargs = {'n_components': simspace_dim, 'perplexity': perp_final, 'random_state': current_random_state}
                            cuml_verbose = 1 if logger.isEnabledFor(logging.INFO) else 0 
                            if logger.isEnabledFor(logging.DEBUG): cuml_verbose = 4
                            sklearn_verbose = 1 if logger.isEnabledFor(logging.DEBUG) else 0
                            if attempt_cuml_tsne_final:
                                tsne_init_kwargs.update({'method': 'barnes_hut', 'verbose': cuml_verbose})
                                if tsne_params.get('n_neighbors') is not None: tsne_init_kwargs['n_neighbors'] = tsne_params['n_neighbors']
                                tsne_model = cumlTSNE(**tsne_init_kwargs); logging.info("Using cuML t-SNE")
                            else:
                                tsne_init_kwargs.update({'init':'pca', 'method':'barnes_hut', 'n_jobs':-1, 'verbose': sklearn_verbose})
                                if 'n_iter' in tsne_params: tsne_init_kwargs['n_iter'] = tsne_params['n_iter']
                                if 'learning_rate' in tsne_params: tsne_init_kwargs['learning_rate'] = tsne_params['learning_rate']
                                tsne_model = sklearnTSNE(**tsne_init_kwargs); logging.info("Using sklearn t-SNE")
                            
                            tsne_embedding = tsne_model.fit_transform(X_tsne_pca_reduced)
                            tsne_emb_main = tsne_embedding[:X_scaled_main.shape[0]]
                            if tsne_emb_main.shape[0] == len(df_results_main):
                                for i in range(simspace_dim): df_results_main[tsne_cols[i]] = tsne_emb_main[:, i]
                            else: logging.error("t-SNE main result length mismatch.")
                            
                            tsne_emb_target = tsne_embedding[X_scaled_main.shape[0]:]
                            if len(tsne_emb_target) == len(df_target_ligands_for_coembed_info):
                                df_tsne_target_coords = pd.DataFrame(tsne_emb_target, columns=tsne_cols, index=df_target_ligands_for_coembed_info.index)
                                id_cols_tsne = [c for c in ['SMILES','Compound ChEMBL ID','Activity Type','Standard Value (nM)','accession'] if c in df_target_ligands_for_coembed_info]
                                df_target_ids_tsne = df_target_ligands_for_coembed_info[id_cols_tsne]
                                df_target_tsne_full = df_target_ids_tsne.join(df_tsne_target_coords, how="inner")
                                tsne_target_path = os.path.join(output_simspace_dir, f"{base_name_prefix}_tSNE_TARGET_PROJECTIONS.csv")
                                df_target_tsne_full.to_csv(tsne_target_path, index=False); logging.info(f"Saved t-SNE target projections: {tsne_target_path}")
                            else: logging.error("t-SNE target result length mismatch.")
                except Exception as e: 
                    logging.error(f"t-SNE failed: {e}", exc_info=True); 
                    for col in tsne_cols: df_results_main[col] = np.nan
        else: logging.info("Skipping t-SNE (dim != 2).")
        gc.collect()

    output_csv_path = os.path.join(output_simspace_dir, f"{base_name_prefix}_similarity_space.csv")
    try:
        info_cols = [c for c in df_results_main.columns if c not in descriptor_columns and not c.startswith(('PCA-','UMAP-','t-SNE-'))]
        dr_cols = [c for c in df_results_main.columns if c.startswith(('PCA-','UMAP-','t-SNE-'))]
        cols_to_save = [c for c in info_cols + dr_cols if c in df_results_main.columns] 
        if cols_to_save:
            df_results_main[cols_to_save].to_csv(output_csv_path, index=False)
            logging.info(f"Saved main simspace: {output_csv_path} (shape {df_results_main[cols_to_save].shape})")
        else: logging.error(f"No columns to save for main simspace CSV {base_name_prefix}")
    except Exception as e: logging.error(f"Failed to save simspace file {output_csv_path}: {e}", exc_info=True)
    
    del df_main_combined, df_results_main, X_original_main_valid, X_scaled_main
    if df_target_ligands_for_coembed_info is not None: del df_target_ligands_for_coembed_info
    if X_target_scaled_for_coembed is not None: del X_target_scaled_for_coembed
    if X_target_original_for_coembed is not None: del X_target_original_for_coembed
    gc.collect()
    logging.info(f"Finished processing for: {base_name_prefix}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate similarity spaces.")
    parser.add_argument("--chembl_mf_data_path", required=True)
    parser.add_argument("--zinc_data_path", default=None)
    parser.add_argument("--target_ligands_unscaled_path_for_tsne_and_coembed", default=None) 
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
    parser.add_argument("--n_neighbors", type=int, default=None, help="N_neighbors for cuML tSNE (passed from orchestrator).") 
    parser.add_argument("--run_coembedding_for_pca_umap", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--dr_method_configs_json_str", required=True)
    parser.add_argument("--random_state", type=int, default=42, help="Random state for DR algorithms.") # Added to __main__
    args_main = parser.parse_args()

    try:
        target_rdkit_features_list_main = json.loads(args_main.rdkit_features_list_target_str)
        dr_method_configs_dict_main = json.loads(args_main.dr_method_configs_json_str) 
    except Exception as e: logging.error(f"Error parsing JSON string arguments: {e}. Aborting."); exit(1)
    
    os.makedirs(args_main.output_simspace_dir, exist_ok=True)
    os.makedirs(args_main.output_model_dir, exist_ok=True)
    
    dr_method_flags_dict_main = {'pca': args_main.dr_method_pca, 'umap': args_main.dr_method_umap, 'tsne': args_main.dr_method_tsne}
    active_umap_metrics_main = {}
    if args_main.dr_method_umap:
        if args_main.umap_metric_to_run_euclidean: active_umap_metrics_main['euclidean'] = True
        if args_main.umap_metric_to_run_cosine: active_umap_metrics_main['cosine'] = True
        if args_main.umap_metric_to_run_manhattan: active_umap_metrics_main['manhattan'] = True
        if args_main.umap_metric_to_run_hamming: active_umap_metrics_main['hamming'] = True
        if not active_umap_metrics_main: logging.warning("UMAP requested but no UMAP metrics enabled.")
    else: logging.info("UMAP method not requested.")
    
    tsne_params_dict_main = {'perplexity': args_main.tsne_perplexity, 
                             'pca_components': args_main.tsne_pca_components}
    if args_main.n_neighbors is not None:
        tsne_params_dict_main['n_neighbors'] = args_main.n_neighbors

    process_similarity_calculations(
        args_main.chembl_mf_data_path, args_main.zinc_data_path, args_main.target_ligands_unscaled_path_for_tsne_and_coembed, 
        args_main.simspace_dim, args_main.representation_type, args_main.target_id_name,
        args_main.output_simspace_dir, args_main.output_model_dir, target_rdkit_features_list_main,
        dr_method_flags_dict_main, active_umap_metrics_main, tsne_params_dict_main,
        args_main.run_coembedding_for_pca_umap, dr_method_configs_dict_main,
        args_main.random_state # Pass the random_state
    )
    logging.info("Similarity space calculation script finished (direct run).")