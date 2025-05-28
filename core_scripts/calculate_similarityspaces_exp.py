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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("calculate_simspaces.log"), logging.StreamHandler()])

FINGERPRINT_COLUMN_PREFIX = "fp_"
NUM_FINGERPRINT_BITS = 2048

def get_descriptor_columns(df, representation_type, target_rdkit_features_list):
    """Identifies descriptor columns based on representation type."""
    if representation_type == "features":
        descriptor_cols = [col for col in target_rdkit_features_list if col in df.columns]
        missing_config_feats = [col for col in target_rdkit_features_list if col not in df.columns]
        if missing_config_feats:
            logging.warning(f"The following configured RDKit features are missing from the input data: {missing_config_feats}. They will be ignored.")
        if not descriptor_cols:
            logging.error(f"No features from RDKIT_FEATURES_LIST_TARGET found in input DataFrame columns: {df.columns.tolist()[:10]}")
            return None
    elif representation_type == "fingerprints":
        available_fp_cols = [col for col in df.columns if col.startswith(FINGERPRINT_COLUMN_PREFIX)]
        if len(available_fp_cols) < NUM_FINGERPRINT_BITS * 0.8: # If significantly less than expected
             logging.warning(f"Found only {len(available_fp_cols)} fingerprint columns (expected around {NUM_FINGERPRINT_BITS}). Check input data.")
        if not available_fp_cols:
            logging.error(f"No fingerprint columns (starting with '{FINGERPRINT_COLUMN_PREFIX}') found in input DataFrame.")
            return None
        descriptor_cols = available_fp_cols
    else:
        logging.error(f"Invalid representation_type: {representation_type}")
        return None
    return descriptor_cols

def process_similarity_calculations(
    chembl_mf_data_path, zinc_data_path, target_ligands_unscaled_path_for_tsne,
    simspace_dim, representation_type, target_id_name,
    output_simspace_dir, output_model_dir, target_rdkit_features_list,
    dr_method_flags, umap_metric_flags, tsne_params):
    """
    Core processing function for calculating similarity spaces.
    dr_method_flags: dict like {'pca': True, 'umap': False, 'tsne': True}
    umap_metric_flags: dict like {'euclidean': True, 'cosine': False} for UMAP metrics
    tsne_params: dict like {'perplexity': 30, 'pca_components': 50}
    """
    base_name_prefix = f"{target_id_name}_{representation_type}_dim{simspace_dim}"
    logging.info(f"Processing for: {base_name_prefix}")

    try:
        df_chembl_mf = pd.read_csv(chembl_mf_data_path, low_memory=False)
        df_zinc = pd.read_csv(zinc_data_path, low_memory=False) if zinc_data_path and os.path.exists(zinc_data_path) else pd.DataFrame()
    except Exception as e:
        logging.error(f"Failed to load ChEMBL MF or ZINC data: {e}")
        return

    all_main_cols = list(set(df_chembl_mf.columns) | (set(df_zinc.columns) if not df_zinc.empty else set()))
    for col in all_main_cols: # Align columns
        if col not in df_chembl_mf.columns: df_chembl_mf[col] = np.nan
        if not df_zinc.empty and col not in df_zinc.columns: df_zinc[col] = np.nan
    
    df_main_combined = pd.concat([df_chembl_mf, df_zinc], ignore_index=True) if not df_zinc.empty else df_chembl_mf.copy()
    logging.info(f"Combined Main (ChEMBL MF + ZINC) DataFrame shape: {df_main_combined.shape}")
    if df_main_combined.empty:
        logging.error("Combined Main DataFrame is empty. Cannot proceed.")
        return

    descriptor_columns = get_descriptor_columns(df_main_combined, representation_type, target_rdkit_features_list)
    if not descriptor_columns: return

    df_main_combined[descriptor_columns] = df_main_combined[descriptor_columns].apply(pd.to_numeric, errors='coerce')
    original_rows_main = len(df_main_combined)
    df_main_combined.dropna(subset=descriptor_columns, how='any', inplace=True)
    if len(df_main_combined) < original_rows_main:
        logging.info(f"Dropped {original_rows_main - len(df_main_combined)} rows from Main data due to NaNs in descriptors.")
    if df_main_combined.empty:
        logging.error("Main DataFrame is empty after NaN drop. Cannot proceed.")
        return
    
    X_original_main = df_main_combined[descriptor_columns].values
    main_data_original_indices = df_main_combined.index # Keep track for df_results

    logging.info("Standardizing Main data...")
    scaler = StandardScaler()
    X_scaled_main = scaler.fit_transform(X_original_main)
    
    scaler_model_path = os.path.join(output_model_dir, f"{base_name_prefix}_scaler.lzma")
    try:
        with open(scaler_model_path, "wb") as f: dump(scaler, f)
        logging.info(f"Saved scaler model to {scaler_model_path}")
    except Exception as e: logging.error(f"Failed to save scaler model: {e}")

    df_results_main = df_main_combined.copy() # To store DR results for main data

    # --- PCA (on main data) ---
    if dr_method_flags.get('pca'):
        logging.info(f"Performing PCA to {simspace_dim}D on Main data...")
        pca_cols = [f'PCA-{i+1}' for i in range(simspace_dim)]
        try:
            pca_model_main = (cumlPCA(n_components=simspace_dim) if CUML_AVAILABLE
                              else sklearnPCA(n_components=simspace_dim))
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
        logging.info(f"Preparing for t-SNE (Perp: {tsne_params['perplexity']}, InitPCA: {tsne_params['pca_components']})...")
        X_for_tsne_combined = X_scaled_main
        num_main_data_points = len(X_scaled_main)
        df_target_ligands_unscaled = None
        
        # Load and scale target ligands for co-embedding
        try:
            df_target_ligands_unscaled = pd.read_csv(target_ligands_unscaled_path_for_tsne, low_memory=False)
            target_desc_cols = get_descriptor_columns(df_target_ligands_unscaled, representation_type, target_rdkit_features_list)
            if target_desc_cols:
                df_target_ligands_unscaled[target_desc_cols] = df_target_ligands_unscaled[target_desc_cols].apply(pd.to_numeric, errors='coerce')
                df_target_ligands_unscaled.dropna(subset=target_desc_cols, how='any', inplace=True) # Crucial
                
                if not df_target_ligands_unscaled.empty:
                    X_target_unscaled_values = df_target_ligands_unscaled[target_desc_cols].values
                    X_target_scaled_values = scaler.transform(X_target_unscaled_values) # Use scaler from main data
                    
                    X_for_tsne_combined = np.vstack((X_scaled_main, X_target_scaled_values))
                    logging.info(f"Co-embedding {num_main_data_points} main points and {len(X_target_scaled_values)} target ligands for t-SNE.")
                else:
                    logging.warning("Target ligands DataFrame empty after NaN drop for t-SNE. t-SNE will run on main data only.")
            else:
                logging.warning("Could not get descriptor columns for target ligands for t-SNE. t-SNE will run on main data only.")
        except Exception as e:
            logging.error(f"Error loading/scaling target ligands for t-SNE: {e}. t-SNE will run on main data only.")
            df_target_ligands_unscaled = None # Ensure it's None if loading failed

        tsne_cols = [f't-SNE-{i+1}' for i in range(simspace_dim)]
        try:
            # 1. Initial PCA for t-SNE
            logging.info(f"t-SNE Step 1: Initial PCA to {tsne_params['pca_components']} components.")
            pca_for_tsne = (cumlPCA(n_components=tsne_params['pca_components'], random_state=42) if CUML_AVAILABLE
                            else sklearnPCA(n_components=tsne_params['pca_components'], random_state=42))
            X_tsne_pca_reduced = pca_for_tsne.fit_transform(X_for_tsne_combined)

            pca_for_tsne_model_path = os.path.join(output_model_dir, f"{base_name_prefix}_tSNE_internal_PCA_model.lzma")
            with open(pca_for_tsne_model_path, "wb") as f: dump(pca_for_tsne, f)
            logging.info(f"Saved t-SNE's internal PCA model to {pca_for_tsne_model_path}")

            # 2. t-SNE
            logging.info(f"t-SNE Step 2: t-SNE to {simspace_dim}D (Perp: {tsne_params['perplexity']}).")
            tsne_model = (cumlTSNE(n_components=simspace_dim, perplexity=tsne_params['perplexity'], random_state=42, method='barnes_hut', verbose=False) if CUML_AVAILABLE
                          else sklearnTSNE(n_components=simspace_dim, perplexity=tsne_params['perplexity'], random_state=42, init='pca', method='barnes_hut', n_jobs=-1))
            tsne_embedding_combined = tsne_model.fit_transform(X_tsne_pca_reduced)

            # Separate main data and target ligand embeddings
            tsne_embedding_main = tsne_embedding_combined[:num_main_data_points]
            for i in range(simspace_dim): df_results_main[tsne_cols[i]] = tsne_embedding_main[:, i]

            if df_target_ligands_unscaled is not None and not df_target_ligands_unscaled.empty and \
               len(tsne_embedding_combined) > num_main_data_points:
                tsne_embedding_target = tsne_embedding_combined[num_main_data_points:]
                
                # Create DataFrame for target ligand t-SNE projections with their original IDs
                df_tsne_target_coords = pd.DataFrame(tsne_embedding_target, columns=tsne_cols, index=df_target_ligands_unscaled.index)
                
                id_cols_to_include = [col for col in ['SMILES', 'Compound ChEMBL ID'] if col in df_target_ligands_unscaled.columns]
                df_target_ids_for_tsne_output = df_target_ligands_unscaled[id_cols_to_include] # Has original index from before dropna
                
                # Align df_target_ids_for_tsne_output with df_tsne_target_coords using index
                df_target_ligands_projected_tsne_with_ids = df_target_ids_for_tsne_output.join(df_tsne_target_coords, how="inner")


                tsne_target_proj_path = os.path.join(output_simspace_dir, f"{base_name_prefix}_tSNE_TARGET_PROJECTIONS.csv")
                df_target_ligands_projected_tsne_with_ids.to_csv(tsne_target_proj_path, index=False)
                logging.info(f"Saved t-SNE projections for TARGET LIGANDS (with IDs) to {tsne_target_proj_path} ({len(df_target_ligands_projected_tsne_with_ids)} records)")
            
            logging.info(f"t-SNE completed.")
        except Exception as e:
            logging.error(f"t-SNE processing failed: {e}")
            for col in tsne_cols: df_results_main[col] = np.nan
        gc.collect()
        
    # --- Save Comprehensive Similarity Space CSV (for main data) ---
    output_csv_name = f"{base_name_prefix}_similarity_space.csv"
    output_csv_path = os.path.join(output_simspace_dir, output_csv_name)
    logging.info(f"Saving comprehensive similarity space for Main data to {output_csv_path}...")
    
    # Reconstruct with original indices for df_main_combined before dropna, then select relevant rows
    # This ensures that the output CSV matches the rows that actually had valid descriptors.
    # df_results_main currently has an index derived from df_main_combined *after* dropna.
    # We need to select the corresponding rows from the original df_main_combined (before dropna but after concat).
    # The `df_main_combined` used to create `df_results_main` already had NaNs dropped and indices reset.
    # So `df_results_main` contains the data for valid rows.

    info_cols = [col for col in df_main_combined.columns if col not in descriptor_columns] # Original info columns from valid rows
    dr_cols = [col for col in df_results_main.columns if col.startswith(('PCA-', 'UMAP-', 't-SNE-'))]
    final_output_cols = info_cols + dr_cols
    
    # Ensure all final_output_cols exist in df_results_main
    cols_to_save = [col for col in final_output_cols if col in df_results_main.columns]
    
    try:
        df_results_main[cols_to_save].to_csv(output_csv_path, index=False)
        logging.info(f"Saved main data similarity space: {output_csv_path} (shape: {df_results_main[cols_to_save].shape})")
    except Exception as e:
        logging.error(f"Failed to save similarity space file {output_csv_path}: {e}")
    
    del df_main_combined, df_results_main, X_scaled_main
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate similarity spaces.")
    parser.add_argument("--chembl_mf_data_path", required=True)
    parser.add_argument("--zinc_data_path", default=None) # Optional
    parser.add_argument("--target_ligands_unscaled_path_for_tsne", default=None, help="Path to UN SCALED target ligands (features/fp) for t-SNE co-embedding.")
    parser.add_argument("--simspace_dim", type=int, required=True)
    parser.add_argument("--representation_type", required=True, choices=["features", "fingerprints"])
    parser.add_argument("--target_id_name", required=True)
    parser.add_argument("--output_simspace_dir", required=True)
    parser.add_argument("--output_model_dir", required=True)
    parser.add_argument("--rdkit_features_list_target_str", required=True, help="JSON string of target RDKit feature names.")

    # DR method flags
    parser.add_argument("--dr_method_pca", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--dr_method_umap", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--dr_method_tsne", type=lambda x: (str(x).lower() == 'true'), default=False)
    
    # UMAP metrics (if --dr_method_umap is True)
    parser.add_argument("--umap_metric_to_run_euclidean", action='store_true')
    parser.add_argument("--umap_metric_to_run_cosine", action='store_true')
    parser.add_argument("--umap_metric_to_run_manhattan", action='store_true')
    parser.add_argument("--umap_metric_to_run_hamming", action='store_true')
    
    # t-SNE params (if --dr_method_tsne is True)
    parser.add_argument("--tsne_perplexity", type=float, default=30.0)
    parser.add_argument("--tsne_pca_components", type=int, default=50)
    
    args = parser.parse_args()

    try:
        target_rdkit_features_list = json.loads(args.rdkit_features_list_target_str)
    except Exception as e:
        logging.error(f"Error parsing --rdkit_features_list_target_str: {e}. Aborting.")
        exit(1)

    os.makedirs(args.output_simspace_dir, exist_ok=True)
    os.makedirs(args.output_model_dir, exist_ok=True)

    dr_method_flags_dict = {'pca': args.dr_method_pca, 'umap': args.dr_method_umap, 'tsne': args.dr_method_tsne}
    umap_metric_flags_dict = {
        'euclidean': args.umap_metric_to_run_euclidean,
        'cosine': args.umap_metric_to_run_cosine,
        'manhattan': args.umap_metric_to_run_manhattan,
        'hamming': args.umap_metric_to_run_hamming
    }
    # Filter only true flags for UMAP metrics to pass to function
    active_umap_metrics = {k:v for k,v in umap_metric_flags_dict.items() if v}

    tsne_params_dict = {'perplexity': args.tsne_perplexity, 'pca_components': args.tsne_pca_components}

    process_similarity_calculations(
        args.chembl_mf_data_path, args.zinc_data_path, args.target_ligands_unscaled_path_for_tsne,
        args.simspace_dim, args.representation_type, args.target_id_name,
        args.output_simspace_dir, args.output_model_dir, target_rdkit_features_list,
        dr_method_flags_dict, active_umap_metrics, tsne_params_dict
    )
    logging.info("Similarity space calculation script finished.")