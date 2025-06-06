import os
import numpy as np
import pandas as pd
import logging # Keep standard logging
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
except ImportError:
    from sklearn.decomposition import PCA as sklearnPCA
    from sklearn.manifold import TSNE as sklearnTSNE
    try:
        from umap import UMAP as umapUMAP # umap-learn for CPU UMAP
        SKLEARN_UMAP_AVAILABLE = True
    except ImportError:
        SKLEARN_UMAP_AVAILABLE = False
    CUML_AVAILABLE = False

# --- MODIFIED LOGGING SETUP ---
logger = logging.getLogger()
logger.setLevel(logging.DEBUG) 
if logger.hasHandlers():
    logger.handlers.clear()
log_formatter = logging.Formatter('%(asctime)s - %(levelname)-8s - %(filename)-25s - %(funcName)-25s - %(lineno)-4d - %(message)s')
try:
    file_handler = logging.FileHandler("calculate_simspaces.log", mode='a')
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.DEBUG) 
    logger.addHandler(file_handler)
except Exception as e:
    print(f"CRITICAL: Failed to initialize file logger for calculate_simspaces.log: {e}")
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_formatter)
stream_handler.setLevel(logging.DEBUG)
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
# --- END MODIFIED LOGGING SETUP ---


FINGERPRINT_COLUMN_PREFIX = "fp_"
NUM_FINGERPRINT_BITS = 2048
PRECALCULATED_FP_COLUMN_NAME = "Fingerprint"

def get_descriptor_columns(df, representation_type, target_rdkit_features_list, is_parsed_fp=False):
    if representation_type == "features":
        descriptor_cols = [col for col in target_rdkit_features_list if col in df.columns]
        missing_config_feats = [col for col in target_rdkit_features_list if col not in df.columns]
        if missing_config_feats:
            logging.warning(f"The following configured RDKit features are missing from the input data: {missing_config_feats}. They will be ignored.")
        if not descriptor_cols:
            logging.error(f"No features from RDKIT_FEATURES_LIST_TARGET found in input DataFrame columns: {df.columns.tolist()[:10]}")
            return None
        return descriptor_cols
    elif representation_type == "fingerprints":
        if is_parsed_fp: 
            descriptor_cols = [f"{FINGERPRINT_COLUMN_PREFIX}{i}" for i in range(NUM_FINGERPRINT_BITS)]
            actual_cols = [col for col in descriptor_cols if col in df.columns]
            if not actual_cols:
                 logging.error(f"Fingerprint representation selected, but no '{FINGERPRINT_COLUMN_PREFIX}*' columns found after expected parsing.")
                 return None
            if len(actual_cols) != NUM_FINGERPRINT_BITS:
                 logging.warning(f"Expected {NUM_FINGERPRINT_BITS} fingerprint columns ('{FINGERPRINT_COLUMN_PREFIX}*'), but found {len(actual_cols)}. Proceeding with found columns.")
            return actual_cols 
        else: 
            if PRECALCULATED_FP_COLUMN_NAME not in df.columns:
                logging.error(f"Fingerprint representation selected, but the expected string column '{PRECALCULATED_FP_COLUMN_NAME}' is not found.")
                return None
            return [PRECALCULATED_FP_COLUMN_NAME] 
    else:
        logging.error(f"Invalid representation_type: {representation_type}")
        return None

def parse_fingerprint_string_column(df_input, fp_string_col_name=PRECALCULATED_FP_COLUMN_NAME, num_bits=NUM_FINGERPRINT_BITS):
    logging.info(f"Parsing fingerprint string column: '{fp_string_col_name}'")
    if fp_string_col_name not in df_input.columns:
        logging.error(f"Fingerprint string column '{fp_string_col_name}' not found in DataFrame for parsing.")
        empty_fp_df = pd.DataFrame(columns=[f"{FINGERPRINT_COLUMN_PREFIX}{i}" for i in range(num_bits)], index=df_input.index)
        df_output = pd.concat([df_input, empty_fp_df], axis=1)
        return df_output

    fp_matrix = np.full((len(df_input), num_bits), np.nan) 
    for idx, fp_str in enumerate(df_input[fp_string_col_name]):
        if pd.isna(fp_str) or not isinstance(fp_str, str) or not fp_str.strip():
            continue
        bits = fp_str.split(',')
        if len(bits) == num_bits:
            try:
                fp_matrix[idx, :] = [int(b) for b in bits]
            except ValueError:
                 logging.debug(f"ValueError parsing bits in string (row {df_input.index[idx]}): {fp_str[:30]}... Filling with NaNs.")
        else:
            logging.debug(f"Fingerprint string (row {df_input.index[idx]}) '{fp_str[:30]}...' has incorrect length {len(bits)}, expected {num_bits}. Filling with NaNs.")
    df_fp_bits = pd.DataFrame(fp_matrix,
                              columns=[f"{FINGERPRINT_COLUMN_PREFIX}{i}" for i in range(num_bits)],
                              index=df_input.index)
    df_output = pd.concat([df_input.drop(columns=[fp_string_col_name]), df_fp_bits], axis=1)
    logging.info(f"Fingerprint string column parsed. Original shape: {df_input.shape}, New shape after FP expansion: {df_output.shape}")
    return df_output

def load_and_prepare_target_ligands_for_coembedding(
    target_ligands_unscaled_path, representation_type, target_rdkit_features_list, scaler_main_data
):
    if not target_ligands_unscaled_path or not os.path.exists(target_ligands_unscaled_path):
        logging.warning(f"Target ligands file for co-embedding not provided or not found: '{target_ligands_unscaled_path}'.")
        return None, None
    try:
        logging.info(f"Loading target ligands for co-embedding from: {target_ligands_unscaled_path}")
        df_target_ligands_raw = pd.read_csv(target_ligands_unscaled_path, low_memory=False)
        target_desc_cols = get_descriptor_columns(df_target_ligands_raw,
                                                  representation_type, target_rdkit_features_list,
                                                  is_parsed_fp=(representation_type == "fingerprints"))
        if not target_desc_cols:
            logging.warning("Could not get descriptor columns for target ligands for co-embedding.")
            return None, None
        df_target_ligands = df_target_ligands_raw.copy()
        df_target_ligands[target_desc_cols] = df_target_ligands[target_desc_cols].apply(pd.to_numeric, errors='coerce')
        original_target_rows = len(df_target_ligands)
        df_target_ligands.dropna(subset=target_desc_cols, how='any', inplace=True)
        if len(df_target_ligands) < original_target_rows:
            logging.info(f"Dropped {original_target_rows - len(df_target_ligands)} rows from target ligands (co-embedding) due to NaNs.")
        if df_target_ligands.empty:
            logging.warning("Target ligands DataFrame empty after NaN drop for co-embedding.")
            return None, None
        X_target_unscaled_values = df_target_ligands[target_desc_cols].values.astype(np.float32)
        X_target_scaled_values = scaler_main_data.transform(X_target_unscaled_values)
        return df_target_ligands, X_target_scaled_values
    except Exception as e:
        logging.error(f"Error loading/processing target ligands for co-embedding: {e}")
        return None, None

def process_similarity_calculations(
    chembl_mf_data_path, zinc_data_path, target_ligands_unscaled_path_for_tsne_and_coembed,
    simspace_dim, representation_type, target_id_name,
    output_simspace_dir, output_model_dir, target_rdkit_features_list,
    dr_method_flags, umap_metric_flags, tsne_params,
    run_coembedding_for_pca_umap, 
    dr_method_configs
    ):

    base_name_prefix = f"{target_id_name}_{representation_type}_dim{simspace_dim}"
    logging.info(f"Processing for: {base_name_prefix}")
    logging.info(f"Run co-embedding for PCA/UMAP: {run_coembedding_for_pca_umap}")

    try:
        df_chembl_mf = pd.read_csv(chembl_mf_data_path, low_memory=False)
        logging.info(f"Loaded ChEMBL MF data: {chembl_mf_data_path}, shape: {df_chembl_mf.shape}")
        df_zinc = pd.read_csv(zinc_data_path, low_memory=False) if zinc_data_path and zinc_data_path.lower() != 'none' and os.path.exists(zinc_data_path) else pd.DataFrame()
        if not df_zinc.empty:
            logging.info(f"Loaded ZINC data: {zinc_data_path}, shape: {df_zinc.shape}")
        else:
            logging.info("No ZINC data path provided or file not found. Proceeding without ZINC data.")
    except FileNotFoundError as fnf_error:
        logging.error(f"FATAL: Input data file not found: {fnf_error}. Aborting processing for {base_name_prefix}.")
        return 
    except Exception as e:
        logging.error(f"Failed to load ChEMBL MF or ZINC data: {e}. Aborting processing for {base_name_prefix}.")
        return

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

    all_main_cols = list(set(df_chembl_mf.columns) | (set(df_zinc.columns) if not df_zinc.empty else set()))
    for id_col_candidate in ['Compound ChEMBL ID', 'ZINC_ID', 'SMILES']:
        if id_col_candidate not in all_main_cols and \
           (id_col_candidate in df_chembl_mf.columns or (not df_zinc.empty and id_col_candidate in df_zinc.columns)):
            all_main_cols.append(id_col_candidate)
    all_main_cols = sorted(list(set(all_main_cols)))
    for col in all_main_cols:
        if col not in df_chembl_mf.columns: df_chembl_mf[col] = pd.NA
        if not df_zinc.empty and col not in df_zinc.columns: df_zinc[col] = pd.NA
    df_main_combined = pd.concat([df_chembl_mf, df_zinc], ignore_index=True) if not df_zinc.empty else df_chembl_mf.copy()
    logging.info(f"Combined Main (ChEMBL MF + ZINC) DataFrame shape: {df_main_combined.shape}")

    if 'Compound ChEMBL ID' in df_main_combined.columns and 'ZINC_ID' in df_main_combined.columns:
        df_main_combined['MOLECULE ID'] = df_main_combined['Compound ChEMBL ID'].fillna(df_main_combined['ZINC_ID'])
    elif 'Compound ChEMBL ID' in df_main_combined.columns:
        df_main_combined['MOLECULE ID'] = df_main_combined['Compound ChEMBL ID']
    elif 'ZINC_ID' in df_main_combined.columns:
        df_main_combined['MOLECULE ID'] = df_main_combined['ZINC_ID']
    else:
        if 'SMILES' in df_main_combined.columns:
            logging.warning("Neither 'Compound ChEMBL ID' nor 'ZINC_ID' found. Using 'SMILES' as 'MOLECULE ID'. This might not be unique.")
            df_main_combined['MOLECULE ID'] = df_main_combined['SMILES']
        else:
            logging.warning("Critical ID columns ('Compound ChEMBL ID', 'ZINC_ID', 'SMILES') missing. Generating placeholder 'MOLECULE ID'.")
            df_main_combined['MOLECULE ID'] = 'UNKNOWN_ID_' + pd.Series(df_main_combined.index).astype(str)
    df_main_combined['MOLECULE ID'] = df_main_combined['MOLECULE ID'].astype(str).fillna('MISSING_MOLECULE_ID')
    logging.info(f"Unified 'MOLECULE ID' column created. Example IDs (up to 5 unique): {df_main_combined['MOLECULE ID'].unique()[:5]}")

    if df_main_combined.empty:
        logging.error("Combined Main DataFrame is empty. Cannot proceed with DR. Aborting for this combination.")
        return

    descriptor_columns = get_descriptor_columns(df_main_combined, representation_type, target_rdkit_features_list,
                                                is_parsed_fp=(representation_type == "fingerprints"))
    if not descriptor_columns:
        logging.error(f"Could not determine descriptor columns for {representation_type}. Aborting DR for this run.")
        return
    logging.info(f"Identified {len(descriptor_columns)} descriptor columns for DR: {descriptor_columns[:5]}...")

    df_main_combined[descriptor_columns] = df_main_combined[descriptor_columns].apply(pd.to_numeric, errors='coerce')
    original_rows_main = len(df_main_combined)
    # Keep track of original indices that are valid
    main_valid_indices = df_main_combined.dropna(subset=descriptor_columns, how='any').index
    df_main_combined_valid = df_main_combined.loc[main_valid_indices].copy() # Use .loc to avoid SettingWithCopyWarning

    if len(df_main_combined_valid) < original_rows_main:
        logging.info(f"Dropped {original_rows_main - len(df_main_combined_valid)} rows from Main data due to NaNs in descriptors.")
    if df_main_combined_valid.empty:
        logging.error("Main DataFrame is empty after NaN drop from descriptor columns. Cannot proceed with DR.")
        return
    logging.info(f"Main DataFrame shape after NaN drop in descriptors: {df_main_combined.shape}")

    df_results_main = df_main_combined.dropna(subset=descriptor_columns, how='any').copy() # This will store DR results

    if len(df_results_main) < original_rows_main:
        logging.info(f"Dropped {original_rows_main - len(df_results_main)} rows from Main data due to NaNs in descriptors.")
    
    if df_results_main.empty:
        logging.error("Main DataFrame (df_results_main) is empty after NaN drop from descriptor columns. Cannot proceed.")
        return

    X_original_main_valid = df_results_main[descriptor_columns].values.astype(np.float32)
    
    logging.info("Standardizing Main data...")
    scaler = StandardScaler()
    X_scaled_main = scaler.fit_transform(X_original_main_valid) # This will have same length as df_results_main
    logging.info(f"StandardScaler fitted on main data. X_scaled_main shape: {X_scaled_main.shape}, df_results_main shape: {df_results_main.shape}")
    
    scaler_model_path = os.path.join(output_model_dir, f"{base_name_prefix}_scaler.lzma")
    try:
        with open(scaler_model_path, "wb") as f: dump(scaler, f)
        logging.info(f"Saved scaler model to {scaler_model_path}")
    except Exception as e: logging.error(f"Failed to save scaler model: {e}")

    df_results_main = df_main_combined.copy()

    df_target_ligands_for_coembed_raw_info = None # Stores original info + unscaled descriptors
    X_target_original_for_coembed = None         # Unscaled numerical descriptors for target
    X_target_scaled_for_coembed = None           # Scaled numerical descriptors for target
    needs_coembed_data = dr_method_flags.get('tsne') or \
                         (run_coembedding_for_pca_umap and (dr_method_flags.get('pca') or dr_method_flags.get('umap')))
    if needs_coembed_data:
        df_target_ligands_for_coembed_raw_info, temp_X_target_scaled = load_and_prepare_target_ligands_for_coembedding(
            target_ligands_unscaled_path_for_tsne_and_coembed,
            representation_type,
            target_rdkit_features_list,
            scaler # Pass the scaler fitted on main data
        )
        if df_target_ligands_for_coembed_raw_info is not None and not df_target_ligands_for_coembed_raw_info.empty:
            X_target_scaled_for_coembed = temp_X_target_scaled # This is already scaled
            # Extract unscaled original from the df_target_ligands_for_coembed_raw_info
            # This assumes target_desc_cols are the same as main descriptor_columns
            X_target_original_for_coembed = df_target_ligands_for_coembed_raw_info[descriptor_columns].values.astype(np.float32)
            logging.info(f"Prepared {X_target_scaled_for_coembed.shape[0]} target ligands for co-embedding (scaled and unscaled versions).")
        else:
            logging.warning("Failed to load/process target ligands for co-embedding. Co-embedding might fail or be skipped.")

    if dr_method_flags.get('pca'):
        # PCA always uses scaled data in this setup
        # ... (PCA logic for projection using X_scaled_main - same as before) ...
        # ... (PCA logic for co-embedding using vstack(X_scaled_main, X_target_scaled_for_coembed) - same as before) ...
        # For brevity, assuming PCA implementation from previous complete version is used here.
        # It should correctly use X_scaled_main for projection model,
        # and np.vstack((X_scaled_main, X_target_scaled_for_coembed)) for co-embedding.
        pca_config = dr_method_configs.get('pca', {})
        logging.info(f"Performing NON-CO-EMBEDDED PCA to {simspace_dim}D on Main data (using scaled)...")
        pca_cols = [f'PCA-{i+1}' for i in range(simspace_dim)]
        try:
            pca_model_main_for_projection = (cumlPCA(n_components=simspace_dim, random_state=42) if CUML_AVAILABLE else sklearnPCA(n_components=simspace_dim, random_state=42))
            pca_res_main_projection = pca_model_main_for_projection.fit_transform(X_scaled_main) # Uses scaled
            for i in range(simspace_dim): df_results_main[pca_cols[i]] = pca_res_main_projection[:, i]
            pca_model_path = os.path.join(output_model_dir, f"{base_name_prefix}_PCA_model.lzma") 
            with open(pca_model_path, "wb") as f: dump(pca_model_main_for_projection, f)
            logging.info(f"Saved NON-CO-EMBEDDED PCA model to {pca_model_path}")
        except Exception as e: logging.error(f"NON-CO-EMBEDDED PCA failed: {e}"); # ... fillna for pca_cols ...
        gc.collect()
        if run_coembedding_for_pca_umap and pca_config.get("allow_coembedding", False) and X_target_scaled_for_coembed is not None and df_target_ligands_for_coembed_raw_info is not None:
            logging.info(f"Performing CO-EMBEDDED PCA to {simspace_dim}D on Main data + Target Ligands (using scaled)...")
            try:
                X_for_pca_coembed = np.vstack((X_scaled_main, X_target_scaled_for_coembed)) # Uses scaled
                pca_model_coembed = (cumlPCA(n_components=simspace_dim, random_state=42) if CUML_AVAILABLE else sklearnPCA(n_components=simspace_dim, random_state=42))
                # ... (rest of PCA co-embedding and saving target projections - same as before) ...
                pca_res_coembed_combined = pca_model_coembed.fit_transform(X_for_pca_coembed)
                num_main_data_points = X_scaled_main.shape[0]
                pca_res_target_coembed = pca_res_coembed_combined[num_main_data_points:]
                if len(pca_res_target_coembed) == len(df_target_ligands_for_coembed_raw_info):
                    df_pca_target_coembed_coords = pd.DataFrame(pca_res_target_coembed, columns=pca_cols, index=df_target_ligands_for_coembed_raw_info.index)
                    id_cols_target = [col for col in ['SMILES', 'Compound ChEMBL ID', 'Activity Type', 'Standard Value (nM)', 'accession'] if col in df_target_ligands_for_coembed_raw_info.columns]
                    df_target_ids_coembed_pca = df_target_ligands_for_coembed_raw_info[id_cols_target]
                    df_target_projected_pca_coembed_with_ids = df_target_ids_coembed_pca.join(df_pca_target_coembed_coords, how="inner")
                    pca_target_coembed_proj_path = os.path.join(output_simspace_dir, f"{base_name_prefix}_PCA_COEMBED_TARGET_PROJECTIONS.csv")
                    df_target_projected_pca_coembed_with_ids.to_csv(pca_target_coembed_proj_path, index=False)
                    logging.info(f"Saved CO-EMBEDDED PCA projections for TARGET LIGANDS to {pca_target_coembed_proj_path}")
                else: logging.error(f"PCA Co-embedding length mismatch. Target co-embed projection not saved.")
            except Exception as e: logging.error(f"CO-EMBEDDED PCA failed: {e}")
            gc.collect()


    # --- UMAP ---
    if dr_method_flags.get('umap'):
        for metric_name, run_metric_flag in umap_metric_flags.items():
            if not run_metric_flag: continue
            # Cosine is already removed from config and orchestrator generation of flags

            umap_key_in_config = f"umap_{metric_name}" 
            umap_config = dr_method_configs.get(umap_key_in_config, {})
            umap_label_prefix = f"UMAP-{metric_name.capitalize()}"
            umap_cols = [f'{umap_label_prefix}-{i+1}' for i in range(simspace_dim)]

            X_input_main_for_this_umap = X_scaled_main 
            X_input_target_for_this_umap_coembed = X_target_scaled_for_coembed 
            use_unscaled_data_for_this_metric = False
            
            # Determine if cuML should be attempted for this UMAP run
            attempt_cuml_umap = CUML_AVAILABLE # Default to trying cuML if available

            if representation_type == "fingerprints":
                if metric_name.lower() in ["hamming", "manhattan"]:
                    use_unscaled_data_for_this_metric = True
                    X_input_main_for_this_umap = X_original_main_valid 
                    if X_target_original_for_coembed is not None:
                        X_input_target_for_this_umap_coembed = X_target_original_for_coembed
                    else: X_input_target_for_this_umap_coembed = None
                    logging.info(f"UMAP metric '{metric_name}' on fingerprints will use UNSCALED data.")
                    
                    # --- NEW: Force scikit-learn for Hamming/Manhattan on fingerprints ---
                    if metric_name.lower() in ["hamming", "manhattan"]:
                        attempt_cuml_umap = False
                        logging.info(f"Forcing scikit-learn UMAP for '{metric_name}' on fingerprints.")
                    # --- END NEW ---
                else: # e.g., Euclidean UMAP on fingerprints
                    logging.info(f"UMAP metric '{metric_name}' on fingerprints will use SCALED data.")
            else: # Features
                logging.info(f"UMAP metric '{metric_name}' on features will use SCALED data.")


            # Non-co-embedded (Projection) UMAP
            logging.info(f"Performing NON-CO-EMBEDDED UMAP ({metric_name}) to {simspace_dim}D on Main data...")
            try:
                umap_model_main_for_projection = None
                if attempt_cuml_umap: # Check our new flag
                    logging.info(f"Attempting cuML UMAP for {metric_name} (projection).")
                    umap_model_main_for_projection = cumlUMAP(n_neighbors=15, min_dist=0.1, n_components=simspace_dim, metric=metric_name, random_state=42, verbose=False)
                elif SKLEARN_UMAP_AVAILABLE:
                    logging.info(f"Using scikit-learn UMAP for {metric_name} (projection).")
                    umap_model_main_for_projection = umapUMAP(n_neighbors=15, min_dist=0.1, n_components=simspace_dim, metric=metric_name, random_state=42, verbose=False)
                else:
                    logging.warning(f"No suitable UMAP library for NON-CO-EMBEDDED UMAP ({metric_name}). Skipping.")
                    for col in umap_cols: df_results_main[col] = np.nan
                    if run_coembedding_for_pca_umap and umap_config.get("allow_coembedding", False): logging.warning(f"Skipping CO-EMBEDDED UMAP for {metric_name} as projection model failed.")
                    continue
                
                umap_res_main_projection = umap_model_main_for_projection.fit_transform(X_input_main_for_this_umap)
                for i in range(simspace_dim): df_results_main[umap_cols[i]] = umap_res_main_projection[:, i]
                umap_model_path = os.path.join(output_model_dir, f"{base_name_prefix}_{metric_name}_UMAP_model.lzma")
                with open(umap_model_path, "wb") as f: dump(umap_model_main_for_projection, f)
                logging.info(f"Saved NON-CO-EMBEDDED UMAP model ({metric_name}) to {umap_model_path}")
            except Exception as e:
                logging.error(f"NON-CO-EMBEDDED UMAP ({metric_name}) failed: {e}")
                # Fallback for cuML if it failed but sklearn is an option (and wasn't forced)
                if attempt_cuml_umap and SKLEARN_UMAP_AVAILABLE and "metric is not supported" in str(e).lower(): # or other specific cuML errors
                    logging.info(f"cuML UMAP failed for {metric_name}, trying scikit-learn UMAP (projection) as fallback...")
                    try:
                        umap_model_main_for_projection = umapUMAP(n_neighbors=15, min_dist=0.1, n_components=simspace_dim, metric=metric_name, random_state=42, verbose=False)
                        umap_res_main_projection = umap_model_main_for_projection.fit_transform(X_input_main_for_this_umap)
                        for i in range(simspace_dim): df_results_main[umap_cols[i]] = umap_res_main_projection[:, i]
                        umap_model_path_fb = os.path.join(output_model_dir, f"{base_name_prefix}_{metric_name}_UMAP_model_sklearn_fallback.lzma")
                        with open(umap_model_path_fb, "wb") as f: dump(umap_model_main_for_projection, f)
                        logging.info(f"Saved scikit-learn UMAP model (projection fallback for {metric_name}) to {umap_model_path_fb}")
                    except Exception as e_fb:
                        logging.error(f"scikit-learn UMAP (projection fallback for {metric_name}) also failed: {e_fb}")
                        for col in umap_cols: df_results_main[col] = np.nan
                else: # No fallback or other error
                     for col in umap_cols: df_results_main[col] = np.nan
                if run_coembedding_for_pca_umap and umap_config.get("allow_coembedding", False): logging.warning(f"Skipping CO-EMBEDDED UMAP for {metric_name} due to projection model failure.")
                gc.collect()
                continue 
            gc.collect()
            
            # Co-embedded UMAP
            if run_coembedding_for_pca_umap and umap_config.get("allow_coembedding", False) and \
               X_input_target_for_this_umap_coembed is not None and df_target_ligands_for_coembed_raw_info is not None:
                logging.info(f"Performing CO-EMBEDDED UMAP ({metric_name}) to {simspace_dim}D on Main data + Target Ligands...")
                try:
                    X_for_umap_coembed = np.vstack((X_input_main_for_this_umap, X_input_target_for_this_umap_coembed))
                    logging.info(f"UMAP Co-embedding data shape for metric {metric_name}: {X_for_umap_coembed.shape}")
                    
                    umap_model_coembed = None
                    if attempt_cuml_umap: # Check the flag again for co-embedding
                        logging.info(f"Attempting cuML UMAP for {metric_name} (co-embedding).")
                        umap_model_coembed = cumlUMAP(n_neighbors=15, min_dist=0.1, n_components=simspace_dim, metric=metric_name, random_state=42, verbose=False)
                    elif SKLEARN_UMAP_AVAILABLE:
                        logging.info(f"Using scikit-learn UMAP for {metric_name} (co-embedding).")
                        umap_model_coembed = umapUMAP(n_neighbors=15, min_dist=0.1, n_components=simspace_dim, metric=metric_name, random_state=42, verbose=False)
                    else: 
                        logging.warning(f"No suitable UMAP library for CO-EMBEDDED UMAP ({metric_name}). Skipping."); continue
                    
                    umap_res_coembed_combined = umap_model_coembed.fit_transform(X_for_umap_coembed)
                    num_main_data_points = X_input_main_for_this_umap.shape[0]
                    umap_res_target_coembed = umap_res_coembed_combined[num_main_data_points:]
                    if len(umap_res_target_coembed) == len(df_target_ligands_for_coembed_raw_info):
                        # ... (saving UMAP co-embedded target projections - same as before) ...
                        df_umap_target_coembed_coords = pd.DataFrame(umap_res_target_coembed, columns=umap_cols, index=df_target_ligands_for_coembed_raw_info.index)
                        id_cols_target = [col for col in ['SMILES', 'Compound ChEMBL ID', 'Activity Type', 'Standard Value (nM)', 'accession'] if col in df_target_ligands_for_coembed_raw_info.columns]
                        df_target_ids_coembed_umap = df_target_ligands_for_coembed_raw_info[id_cols_target]
                        df_target_projected_umap_coembed_with_ids = df_target_ids_coembed_umap.join(df_umap_target_coembed_coords, how="inner")
                        umap_target_coembed_proj_path = os.path.join(output_simspace_dir, f"{base_name_prefix}_{metric_name}_UMAP_COEMBED_TARGET_PROJECTIONS.csv")
                        df_target_projected_umap_coembed_with_ids.to_csv(umap_target_coembed_proj_path, index=False)
                        logging.info(f"Saved CO-EMBEDDED UMAP ({metric_name}) projections for TARGET LIGANDS to {umap_target_coembed_proj_path}")
                    else: logging.error(f"UMAP Co-embedding length mismatch for metric {metric_name}. Target co-embed projection not saved.")
                except Exception as e_co_umap: 
                    logging.error(f"CO-EMBEDDED UMAP ({metric_name}) failed: {e_co_umap}")
                    if attempt_cuml_umap and SKLEARN_UMAP_AVAILABLE and "metric is not supported" in str(e_co_umap).lower():
                        logging.info(f"cuML CO-EMBEDDED UMAP failed for {metric_name}, trying scikit-learn UMAP (co-embedding) as fallback...")
                        try:
                            umap_model_coembed_fb = umapUMAP(n_neighbors=15, min_dist=0.1, n_components=simspace_dim, metric=metric_name, random_state=42, verbose=False)
                            umap_res_coembed_combined_fb = umap_model_coembed_fb.fit_transform(X_for_umap_coembed) # X_for_umap_coembed is already prepared
                            num_main_data_points = X_input_main_for_this_umap.shape[0] # Re-affirm
                            umap_res_target_coembed_fb = umap_res_coembed_combined_fb[num_main_data_points:]
                            if len(umap_res_target_coembed_fb) == len(df_target_ligands_for_coembed_raw_info):
                                # ... (save sklearn fallback co-embed target projections) ...
                                df_umap_target_coembed_coords_fb = pd.DataFrame(umap_res_target_coembed_fb, columns=umap_cols, index=df_target_ligands_for_coembed_raw_info.index)
                                df_target_ids_coembed_umap_fb = df_target_ligands_for_coembed_raw_info[id_cols_target] # id_cols_target defined above
                                df_target_projected_umap_coembed_with_ids_fb = df_target_ids_coembed_umap_fb.join(df_umap_target_coembed_coords_fb, how="inner")
                                umap_target_coembed_proj_path_fb = os.path.join(output_simspace_dir, f"{base_name_prefix}_{metric_name}_UMAP_COEMBED_TARGET_PROJECTIONS_sklearn_fallback.csv")
                                df_target_projected_umap_coembed_with_ids_fb.to_csv(umap_target_coembed_proj_path_fb, index=False)
                                logging.info(f"Saved scikit-learn CO-EMBEDDED UMAP ({metric_name}) target projections (fallback) to {umap_target_coembed_proj_path_fb}")
                            else: logging.error(f"scikit-learn UMAP Co-embedding (fallback) length mismatch for metric {metric_name}.")
                        except Exception as e_co_umap_fb:
                            logging.error(f"scikit-learn CO-EMBEDDED UMAP (fallback, {metric_name}) also failed: {e_co_umap_fb}")
                gc.collect()
            elif run_coembedding_for_pca_umap and umap_config.get("allow_coembedding", False) and X_input_target_for_this_umap_coembed is None:
                logging.warning(f"Skipping CO-EMBEDDED UMAP for {metric_name} because target ligand data for co-embedding was not available.")

    # --- t-SNE ---
    if dr_method_flags.get('tsne'):
        tsne_config = dr_method_configs.get('tsne', {}) 
        if tsne_config.get("allow_coembedding", True): # tSNE is always co-embedded here
            if simspace_dim == 2:
                if X_target_scaled_for_coembed is None or df_target_ligands_for_coembed_raw_info is None:
                    logging.warning("Cannot run t-SNE co-embedding because target ligand data for co-embedding failed to load/process. Skipping t-SNE.")
                else:
                    logging.info(f"Attempting t-SNE (co-embedding by design) as simspace_dim is 2 ...")
                    X_for_tsne_combined = np.vstack((X_scaled_main, X_target_scaled_for_coembed)) # tSNE uses scaled data after PCA
                    num_main_data_points_tsne = X_scaled_main.shape[0]
                    tsne_cols = [f't-SNE-{i+1}' for i in range(simspace_dim)]
                    try:
                        actual_pca_components_for_tsne = min(tsne_params['pca_components'], X_for_tsne_combined.shape[0]-1 if X_for_tsne_combined.shape[0] > 1 else 1, X_for_tsne_combined.shape[1])
                        if actual_pca_components_for_tsne < 1: 
                            logging.error(f"Cannot run t-SNE: too few samples/features for initial PCA. Need at least 1 effective component, got {actual_pca_components_for_tsne}.")
                            for col in tsne_cols: df_results_main[col] = np.nan
                        else:
                            logging.info(f"t-SNE Step 1: Initial PCA to {actual_pca_components_for_tsne} components (data shape: {X_for_tsne_combined.shape}).")
                            # For t-SNE's internal PCA, cuML can be used if available
                            pca_for_tsne = (cumlPCA(n_components=actual_pca_components_for_tsne, random_state=42) if CUML_AVAILABLE
                                            else sklearnPCA(n_components=actual_pca_components_for_tsne, random_state=42))
                            X_tsne_pca_reduced = pca_for_tsne.fit_transform(X_for_tsne_combined)
                            # ... (save PCA model for tSNE) ...

                            logging.info(f"t-SNE Step 2: t-SNE to {simspace_dim}D. Input PCA-reduced shape: {X_tsne_pca_reduced.shape}")
                            
                            # --- NEW: Force scikit-learn t-SNE for fingerprints ---
                            attempt_cuml_tsne = CUML_AVAILABLE
                            if representation_type == "fingerprints":
                                attempt_cuml_tsne = False
                                logging.info("Forcing scikit-learn t-SNE for fingerprints.")
                            # --- END NEW ---

                            cuml_tsne_verbose_level = 0 
                            if logger.level == logging.DEBUG: cuml_tsne_verbose_level = 5 
                            elif logger.level == logging.INFO: cuml_tsne_verbose_level = 1
                            sklearn_tsne_verbose_level = 1 if logger.isEnabledFor(logging.DEBUG) else 0
                            
                            tsne_perplexity_to_use_final = min(tsne_params['perplexity'], X_tsne_pca_reduced.shape[0] - 2 if X_tsne_pca_reduced.shape[0] > 1 else 0)
                            if X_tsne_pca_reduced.shape[0] <= 1 or tsne_perplexity_to_use_final < 1:
                                logging.error(f"Too few samples ({X_tsne_pca_reduced.shape[0]}) or invalid perplexity ({tsne_perplexity_to_use_final}) for t-SNE after PCA. Skipping t-SNE.")
                                for col in tsne_cols: df_results_main[col] = np.nan
                            else:
                                if tsne_perplexity_to_use_final != tsne_params['perplexity']: logging.info(f"Adjusted t-SNE perplexity to: {tsne_perplexity_to_use_final}")

                                tsne_model = None
                                if attempt_cuml_tsne:
                                    logging.info("Attempting cuML t-SNE.")
                                    tsne_init_kwargs_tsne_cuml = {'n_components': simspace_dim, 'perplexity': tsne_perplexity_to_use_final, 'random_state': 42, 'method': 'barnes_hut', 'verbose': cuml_tsne_verbose_level}
                                    if args.n_neighbors is not None: tsne_init_kwargs_tsne_cuml['n_neighbors'] = args.n_neighbors # Use main script args if passed
                                    tsne_model = cumlTSNE(**tsne_init_kwargs_tsne_cuml)
                                else: # Use scikit-learn
                                    logging.info("Using scikit-learn t-SNE.")
                                    tsne_model = sklearnTSNE(n_components=simspace_dim, perplexity=tsne_perplexity_to_use_final, random_state=42, init='pca', method='barnes_hut', n_jobs=-1, verbose=sklearn_tsne_verbose_level)
                            
                                tsne_embedding_combined = tsne_model.fit_transform(X_tsne_pca_reduced)
                                # ... (rest of t-SNE: splitting main/target, saving target projections - same as before) ...
                                logging.info(f"t-SNE fit_transform completed. Output shape: {tsne_embedding_combined.shape}")
                                tsne_embedding_main = tsne_embedding_combined[:num_main_data_points_tsne]
                                for i in range(simspace_dim): df_results_main[tsne_cols[i]] = tsne_embedding_main[:, i]
                                if df_target_ligands_for_coembed_raw_info is not None and not df_target_ligands_for_coembed_raw_info.empty and \
                                len(tsne_embedding_combined) > num_main_data_points_tsne:
                                    tsne_embedding_target = tsne_embedding_combined[num_main_data_points_tsne:]
                                    if len(tsne_embedding_target) == len(df_target_ligands_for_coembed_raw_info):
                                        df_tsne_target_coords = pd.DataFrame(tsne_embedding_target, columns=tsne_cols, index=df_target_ligands_for_coembed_raw_info.index)
                                        id_cols_target = [col for col in ['SMILES', 'Compound ChEMBL ID', 'Activity Type', 'Standard Value (nM)', 'accession'] if col in df_target_ligands_for_coembed_raw_info.columns]
                                        df_target_ids_for_tsne_output = df_target_ligands_for_coembed_raw_info[id_cols_target]
                                        df_target_projected_tsne_with_ids = df_target_ids_for_tsne_output.join(df_tsne_target_coords, how="inner")
                                        tsne_target_proj_path = os.path.join(output_simspace_dir, f"{base_name_prefix}_tSNE_TARGET_PROJECTIONS.csv") 
                                        df_target_projected_tsne_with_ids.to_csv(tsne_target_proj_path, index=False)
                                        logging.info(f"Saved t-SNE projections for TARGET LIGANDS to {tsne_target_proj_path}")
                                    else: logging.error(f"t-SNE Mismatch: Target embedding length != df_target_ligands length.")
                                else: logging.info("No target ligands were co-embedded or available for t-SNE output.")
                                logging.info(f"t-SNE completed successfully.")
                    except Exception as e: 
                        logging.error(f"t-SNE processing FAILED: {e}", exc_info=True)
                        for col in tsne_cols: df_results_main[col] = np.nan
                    gc.collect()
            else: 
                logging.info(f"Skipping t-SNE calculation because simspace_dim is {simspace_dim} (t-SNE is configured to run only for 2D).")
        else: 
            logging.info("t-SNE co-embedding explicitly disallowed by config (should not happen for tSNE usually).")

    output_csv_name = f"{base_name_prefix}_similarity_space.csv"
    output_csv_path = os.path.join(output_simspace_dir, output_csv_name)
    logging.info(f"Preparing to save comprehensive similarity space for Main data to {output_csv_path}...")
    info_cols = [col for col in df_results_main.columns if col not in descriptor_columns and not col.startswith(('PCA-', 'UMAP-', 't-SNE-'))]
    dr_cols = [col for col in df_results_main.columns if col.startswith(('PCA-', 'UMAP-', 't-SNE-'))] 
    final_output_cols = info_cols + dr_cols
    cols_to_save = [col for col in final_output_cols if col in df_results_main.columns]
    if not cols_to_save:
        logging.error(f"No columns to save for the main similarity space CSV for {base_name_prefix}.")
    else:
        try:
            df_to_save_final = df_results_main[cols_to_save]
            df_to_save_final.to_csv(output_csv_path, index=False)
            logging.info(f"Saved main data similarity space (from non-coembedded PCA/UMAP): {output_csv_path} (shape: {df_to_save_final.shape})")
        except Exception as e:
            logging.error(f"Failed to save similarity space file {output_csv_path}: {e}")

    del df_main_combined, df_results_main, X_scaled_main, X_original_main
    if df_target_ligands_for_coembed is not None: del df_target_ligands_for_coembed
    if X_target_scaled_for_coembed is not None: del X_target_scaled_for_coembed
    gc.collect()
    logging.info(f"Finished processing for: {base_name_prefix}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate similarity spaces with optional co-embedding and conditional scaling/engine.")
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
    parser.add_argument("--umap_metric_to_run_manhattan", action='store_true', default=False)
    parser.add_argument("--umap_metric_to_run_hamming", action='store_true', default=False)
    parser.add_argument("--tsne_perplexity", type=float, default=30.0)
    parser.add_argument("--tsne_pca_components", type=int, default=50)
    parser.add_argument("--run_coembedding_for_pca_umap", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--dr_method_configs_json_str", required=True, help="JSON string of the 'dimensionality_reduction_methods' from main config.")
    args = parser.parse_args()
    logging.info(f"Parsed arguments: {args}")

    try:
        target_rdkit_features_list = json.loads(args.rdkit_features_list_target_str)
        dr_method_configs_dict = json.loads(args.dr_method_configs_json_str) 
    except Exception as e:
        logging.error(f"Error parsing JSON string arguments (RDKit features or DR configs): {e}. Aborting.")
        exit(1)

    os.makedirs(args.output_simspace_dir, exist_ok=True)
    os.makedirs(args.output_model_dir, exist_ok=True)
    dr_method_flags_dict = {'pca': args.dr_method_pca, 'umap': args.dr_method_umap, 'tsne': args.dr_method_tsne}
    active_umap_metrics = {}
    if args.dr_method_umap:
        if args.umap_metric_to_run_euclidean: active_umap_metrics['euclidean'] = True
        if args.umap_metric_to_run_manhattan: active_umap_metrics['manhattan'] = True
        if args.umap_metric_to_run_hamming: active_umap_metrics['hamming'] = True
        if not active_umap_metrics:
            logging.warning("UMAP method requested (--dr_method_umap=True) but no specific UMAP metrics enabled. UMAP will not run effectively.")
    else:
        logging.info("UMAP method is not requested (--dr_method_umap=False).")
    tsne_params_dict = {'perplexity': args.tsne_perplexity, 'pca_components': args.tsne_pca_components}

    logging.info(f"Starting process_similarity_calculations with DR flags: {dr_method_flags_dict}, UMAP metrics: {active_umap_metrics}, t-SNE params: {tsne_params_dict}, Run PCA/UMAP Co-embed: {args.run_coembedding_for_pca_umap}")
    process_similarity_calculations(
        args.chembl_mf_data_path, args.zinc_data_path, args.target_ligands_unscaled_path_for_tsne_and_coembed, 
        args.simspace_dim, args.representation_type, args.target_id_name,
        args.output_simspace_dir, args.output_model_dir, target_rdkit_features_list,
        dr_method_flags_dict, active_umap_metrics, tsne_params_dict,
        args.run_coembedding_for_pca_umap, 
        dr_method_configs_dict 
    )
    logging.info("Similarity space calculation script finished.")
