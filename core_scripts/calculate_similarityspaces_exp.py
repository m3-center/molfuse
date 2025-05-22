import os
import glob
import numpy as np
import pandas as pd
import logging
import gc
from compress_pickle import dump # Using compress_pickle for models
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import argparse

# CUML for GPU acceleration if available, fallback to sklearn otherwise
try:
    from cuml import PCA as cumlPCA
    from cuml import UMAP as cumlUMAP
    from cuml.common.device_selection import using_device_type # For UMAP transform on CPU if needed
    CUML_AVAILABLE = True
    logging.info("cuML found. Using GPU for PCA and UMAP.")
except ImportError:
    from sklearn.decomposition import PCA as sklearnPCA
    # For UMAP, umap-learn is a common CPU alternative
    try:
        from umap import UMAP as umapUMAP 
        SKLEARN_UMAP_AVAILABLE = True
    except ImportError:
        SKLEARN_UMAP_AVAILABLE = False
    CUML_AVAILABLE = False
    logging.info("cuML not found. Falling back to scikit-learn for PCA. UMAP on CPU if 'umap-learn' is installed.")


# ------------------------ Configuration (for CLI) ------------------------
DEFAULT_OUTPUT_SIMSPACE_DIR = "temp/exp_simspaces"
DEFAULT_OUTPUT_MODEL_DIR = "temp/exp_models"

# Feature list (must match output of calculate_features_and_fingerprints_exp.py)
RDKIT_FEATURES_LIST_TARGET = [ # Copied from calculate_features_and_fingerprints_exp.py for consistency
    'DipoleMoment','ABC','nAcid','nBase','nAromAtom','nAtom','nH','nC','nN','nO','nS',
    'nP','nX','nBonds','nBondsO','nBondsS','nBondsD','nBondsT','nBondsA','nBondsM',
    'nBondsKS','nBondsKD','EState_VSA7','nHBAcc','nHBDon','Lipinski','apol','bpol',
    'nRing','n3Ring','n4Ring','n5Ring','n6Ring','n7Ring','n8Ring','nRot','Diameter',
    'TopoShapeIndex','Vabc','MW'
]
FINGERPRINT_COLUMN_PREFIX = "fp_"
NUM_FINGERPRINT_BITS = 2048


def process_file_for_simspace(
    chembl_mf_data_path, zinc_data_path,
    simspace_dim, representation_type, target_id_name,
    output_simspace_dir, output_model_dir,
    run_pca, run_umaps, umap_metrics_dict):
    """
    Core processing function.
    - Loads ChEMBL MF (excluded) and ZINC (excluded) data (already featurized/fingerprinted).
    - Combines them.
    - Performs scaling.
    - Performs selected dimensionality reductions (PCA, UMAPs).
    - Saves models and the final similarity space CSV.
    """
    base_name_prefix = f"{target_id_name}_{representation_type}_dim{simspace_dim}"
    logging.info(f"Processing for: {base_name_prefix}")

    try:
        df_chembl_mf = pd.read_csv(chembl_mf_data_path, low_memory=False)
        logging.info(f"Loaded ChEMBL MF data: {chembl_mf_data_path}, shape: {df_chembl_mf.shape}")
    except Exception as e:
        logging.error(f"Failed to load ChEMBL MF data from {chembl_mf_data_path}: {e}")
        return

    try:
        df_zinc = pd.read_csv(zinc_data_path, low_memory=False)
        logging.info(f"Loaded ZINC data: {zinc_data_path}, shape: {df_zinc.shape}")
    except Exception as e:
        logging.error(f"Failed to load ZINC data from {zinc_data_path}: {e}")
        df_zinc = pd.DataFrame() # Proceed without ZINC if it fails to load

    # Combine datasets
    # Ensure columns align before concat. Add missing columns with NaN.
    if not df_zinc.empty:
        all_cols = list(set(df_chembl_mf.columns) | set(df_zinc.columns))
        for col in all_cols:
            if col not in df_chembl_mf.columns: df_chembl_mf[col] = np.nan
            if col not in df_zinc.columns: df_zinc[col] = np.nan
        df_combined = pd.concat([df_chembl_mf[all_cols], df_zinc[all_cols]], ignore_index=True)
    else:
        df_combined = df_chembl_mf.copy()
    
    logging.info(f"Combined DataFrame shape: {df_combined.shape}")
    if df_combined.empty:
        logging.error("Combined DataFrame is empty. Cannot proceed.")
        return

    # Identify descriptor columns based on representation_type
    if representation_type == "features":
        descriptor_columns = RDKIT_FEATURES_LIST_TARGET
        missing_feats = [f for f in descriptor_columns if f not in df_combined.columns]
        if missing_feats:
            logging.error(f"Missing required feature columns in combined data: {missing_feats}. Cannot proceed.")
            return
    elif representation_type == "fingerprints":
        descriptor_columns = [f"{FINGERPRINT_COLUMN_PREFIX}{i}" for i in range(NUM_FINGERPRINT_BITS)]
        missing_fps = [fp for fp in descriptor_columns if fp not in df_combined.columns]
        if missing_fps:
            # Only log error if a substantial number are missing, could be some other fp type
            if len(missing_fps) > NUM_FINGERPRINT_BITS / 2 :
                 logging.error(f"Many fingerprint columns ({len(missing_fps)}) are missing (e.g., {missing_fps[:5]}). Cannot proceed.")
                 return
            else: # If only a few missing, proceed but warn
                 logging.warning(f"Some fingerprint columns ({len(missing_fps)}) are missing. Proceeding with available ones.")
                 descriptor_columns = [col for col in descriptor_columns if col in df_combined.columns]
                 if not descriptor_columns:
                     logging.error("No valid fingerprint columns found after checking. Cannot proceed.")
                     return
    else:
        logging.error(f"Invalid representation_type: {representation_type}")
        return

    # Ensure descriptor columns are numeric and handle NaNs by dropping rows
    # This is crucial for DR algorithms.
    df_combined[descriptor_columns] = df_combined[descriptor_columns].apply(pd.to_numeric, errors='coerce')
    
    original_rows = len(df_combined)
    df_combined.dropna(subset=descriptor_columns, how='any', inplace=True) # Drop if ANY descriptor is NaN
    rows_after_dropna = len(df_combined)
    if rows_after_dropna < original_rows:
        logging.info(f"Dropped {original_rows - rows_after_dropna} rows with NaN values in descriptor columns.")
    
    if df_combined.empty:
        logging.error("DataFrame is empty after NaN drop. Cannot proceed.")
        return

    X_original = df_combined[descriptor_columns].values
    if X_original.shape[1] == 0:
        logging.error(f"No descriptor columns selected for X_original. Columns were: {descriptor_columns}")
        return

    # --- Scaling ---
    logging.info("Standardizing input data...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_original)
    
    # Save scaler model
    scaler_model_path = os.path.join(output_model_dir, f"{base_name_prefix}_scaler.lzma") # Consistent naming
    try:
        with open(scaler_model_path, "wb") as f: dump(scaler, f)
        logging.info(f"Saved scaler model to {scaler_model_path}")
    except Exception as e:
        logging.error(f"Failed to save scaler model: {e}")

    del X_original # Free memory
    gc.collect()

    # --- Dimensionality Reduction ---
    df_results = df_combined.copy() # To store DR results alongside original data

    # PCA
    if run_pca:
        logging.info(f"Performing PCA to {simspace_dim} dimensions...")
        pca_cols = [f'PCA-{i+1}' for i in range(simspace_dim)]
        try:
            if CUML_AVAILABLE:
                pca_model = cumlPCA(n_components=simspace_dim, random_state=42)
            else:
                pca_model = sklearnPCA(n_components=simspace_dim, random_state=42)
            
            pca_res = pca_model.fit_transform(X_scaled)
            for i in range(simspace_dim):
                df_results[pca_cols[i]] = pca_res[:, i]

            pca_model_path = os.path.join(output_model_dir, f"{base_name_prefix}_PCA_model.lzma")
            with open(pca_model_path, "wb") as f: dump(pca_model, f)
            logging.info(f"Saved PCA model to {pca_model_path}")
            del pca_res
        except Exception as e:
            logging.error(f"PCA failed: {e}")
            for col in pca_cols: df_results[col] = np.nan # Add NaN columns if PCA fails


    # UMAP
    if run_umaps and umap_metrics_dict:
        for umap_key, metric_name in umap_metrics_dict.items(): # umap_key is like 'umap_euclidean'
            dr_label_for_file = umap_key.upper().replace("_", "-") # e.g., UMAP-EUCLIDEAN
            umap_model_file_label = metric_name # e.g. euclidean for UMAP_euclidean_model.lzma
            
            logging.info(f"Performing UMAP with metric: {metric_name} to {simspace_dim} dimensions...")
            umap_cols = [f'{dr_label_for_file}-{i+1}' for i in range(simspace_dim)]
            try:
                if CUML_AVAILABLE:
                    # Forcing UMAP to run on CPU for transform if needed and if cuml version supports it for transform consistency
                    # This is mainly for when transform is called separately, fit_transform usually handles device
                    umap_model = cumlUMAP(n_neighbors=15, min_dist=0.1, n_components=simspace_dim, 
                                          metric=metric_name, random_state=42, verbose=False)
                elif SKLEARN_UMAP_AVAILABLE:
                    umap_model = umapUMAP(n_neighbors=15, min_dist=0.1, n_components=simspace_dim,
                                          metric=metric_name, random_state=42, verbose=False)
                else:
                    logging.warning(f"UMAP library (cuML or umap-learn) not available. Skipping UMAP for {metric_name}.")
                    for col in umap_cols: df_results[col] = np.nan
                    continue
                
                umap_res = umap_model.fit_transform(X_scaled)
                for i in range(simspace_dim):
                    df_results[umap_cols[i]] = umap_res[:, i]

                umap_model_path = os.path.join(output_model_dir, f"{base_name_prefix}_{umap_model_file_label}_UMAP_model.lzma")
                with open(umap_model_path, "wb") as f: dump(umap_model, f)
                logging.info(f"Saved UMAP model ({metric_name}) to {umap_model_path}")
                del umap_res
            except Exception as e:
                logging.error(f"UMAP ({metric_name}) failed: {e}")
                for col in umap_cols: df_results[col] = np.nan
            gc.collect()

    # --- Save output CSV ---
    # The DR method label for the CSV should be generic if only one DR was run, or specific
    # For simplicity, we save one CSV per DR method if multiple are run, or one combined if that's desired.
    # Here, we save a single CSV that has ALL computed dimensions.
    # The orchestrator calls this script ONCE, and it computes all requested DR methods.
    # The file name here should reflect that it contains *all* computed dimensions for this base_name_prefix.
    # However, the original structure was one simspace CSV per DR method.
    # Let's stick to saving one comprehensive CSV from this script.
    # The project_and_analyze.py will then pick the relevant columns.
    
    output_csv_name = f"{base_name_prefix}_similarity_space.csv" # This will contain all computed DR dims
    output_csv_path = os.path.join(output_simspace_dir, output_csv_name)
    
    logging.info(f"Saving comprehensive similarity space file to {output_csv_path}...")
    try:
        # Select only relevant columns for final CSV to save space
        # Original info columns + all computed DR columns
        info_cols_to_keep = [col for col in df_combined.columns if col not in descriptor_columns]
        dr_cols_in_results = [col for col in df_results.columns if col.startswith(('PCA-', 'UMAP-'))]
        final_cols = info_cols_to_keep + dr_cols_in_results
        
        # Ensure all final_cols are actually in df_results, handle if some DR failed
        final_cols_present = [col for col in final_cols if col in df_results.columns]
        
        df_results[final_cols_present].to_csv(output_csv_path, index=False)
        logging.info(f"Saved {output_csv_path}")
    except Exception as e:
        logging.error(f"Failed to save similarity space file {output_csv_path}: {e}")
    
    del df_combined, df_results, X_scaled
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate similarity spaces from featurized/fingerprinted data.")
    parser.add_argument("--chembl_mf_data_path", required=True, help="Path to ChEMBL MF (excluded) data CSV (features/fingerprints).")
    parser.add_argument("--zinc_data_path", required=True, help="Path to ZINC (excluded) data CSV (features/fingerprints).")
    parser.add_argument("--simspace_dim", type=int, required=True, help="Number of dimensions for the similarity spaces.")
    parser.add_argument("--representation_type", required=True, choices=["features", "fingerprints"], help="Type of input data representation.")
    parser.add_argument("--target_id_name", required=True, help="Unique ID name for the target (for naming outputs).")
    parser.add_argument("--output_simspace_dir", default=DEFAULT_OUTPUT_SIMSPACE_DIR, help="Directory to save similarity space CSVs.")
    parser.add_argument("--output_model_dir", default=DEFAULT_OUTPUT_MODEL_DIR, help="Directory to save dimensionality reduction models.")

    # DR method selection flags
    parser.add_argument("--dr_method_pca", action="store_true", help="Run PCA.")
    # UMAP metrics are passed via a different mechanism if called from orchestrator
    # For CLI, we can simplify or add individual metric flags
    parser.add_argument("--dr_method_umap_euclidean", action="store_true", help="Run UMAP with Euclidean metric.")
    parser.add_argument("--dr_method_umap_cosine", action="store_true", help="Run UMAP with Cosine metric.")
    parser.add_argument("--dr_method_umap_manhattan", action="store_true", help="Run UMAP with Manhattan metric.")
    parser.add_argument("--dr_method_umap_hamming", action="store_true", help="Run UMAP with Hamming metric.")
    
    args = parser.parse_args()

    os.makedirs(args.output_simspace_dir, exist_ok=True)
    os.makedirs(args.output_model_dir, exist_ok=True)

    # Construct umap_metrics_dict from CLI args
    cli_umap_metrics = {}
    if args.dr_method_umap_euclidean: cli_umap_metrics['umap_euclidean'] = 'euclidean'
    if args.dr_method_umap_cosine: cli_umap_metrics['umap_cosine'] = 'cosine'
    if args.dr_method_umap_manhattan: cli_umap_metrics['umap_manhattan'] = 'manhattan'
    if args.dr_method_umap_hamming: cli_umap_metrics['umap_hamming'] = 'hamming'
    
    run_any_umap = bool(cli_umap_metrics)

    process_file_for_simspace(
        args.chembl_mf_data_path, args.zinc_data_path,
        args.simspace_dim, args.representation_type, args.target_id_name,
        args.output_simspace_dir, args.output_model_dir,
        run_pca=args.dr_method_pca,
        run_umaps=run_any_umap,
        umap_metrics_dict=cli_umap_metrics
    )
    logging.info("Similarity space calculation script finished.")