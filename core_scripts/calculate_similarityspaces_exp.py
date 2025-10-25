import os
import sys
import logging
import gc
import argparse
import json

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from compress_pickle import dump
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA as sklearnPCA
from sklearn.preprocessing import StandardScaler

from core_scripts.utils import PassthroughScaler

# cuML and UMAP imports
try:
    from cuml import PCA as cumlPCA, UMAP as cumlUMAP
    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False
try:
    from umap import UMAP as umapUMAP
    SKLEARN_UMAP_AVAILABLE = True
except ImportError:
    SKLEARN_UMAP_AVAILABLE = False
logger = logging.getLogger()
log_formatter = logging.Formatter(
    '%(asctime)s - %(levelname)-8s - %(filename)-25s - %(funcName)-25s - %(lineno)-4d - %(message)s')
FINGERPRINT_COLUMN_PREFIX = "fp_"
NUM_FINGERPRINT_BITS = 2048
PRECALCULATED_FP_STRING_COLUMN_NAME = "Fingerprint"


def is_gpu_available():
    if not CUML_AVAILABLE:
        return False
    try:
        import cupy
        if cupy.cuda.runtime.getDeviceCount() > 0:
            logger.info(
                "CUDA device found. cuML (GPU) will be used where possible.")
            return True
        return False
    except Exception:
        logger.warning(
            f"CUDA device not found or failed to initialize. Falling back to scikit-learn (CPU).")
        return False

# ... (All helper functions like setup_script_logging, get_descriptor_columns, save_model, etc. are unchanged) ...


def setup_script_logging(log_filename):
    if logger.hasHandlers():
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()
    file_handler = logging.FileHandler(log_filename, mode='a')
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_formatter)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.info(
        f"--- calculate_similarityspaces_exp.py logging attached to: {log_filename} ---")


def get_descriptor_columns(df, representation_type, target_rdkit_features_list):
    if representation_type == "features":
        if not target_rdkit_features_list:
            return []
        return [col for col in target_rdkit_features_list if col in df.columns]
    elif representation_type == "fingerprints":
        return [f"{FINGERPRINT_COLUMN_PREFIX}{i}" for i in range(NUM_FINGERPRINT_BITS)]
    return []


def save_model(model, path):
    try:
        with open(path, "wb") as f:
            dump(model, f)
        logger.info(f"Saved model to: {path}")
    except Exception as e:
        logger.error(f"Failed to save model to {path}: {e}", exc_info=True)


# construct_coembedded_dataframe() removed in v2.0: Co-embedding creates data leakage


def _parse_fingerprint_chunk(df_chunk):
    fp_matrix = np.full((len(df_chunk), NUM_FINGERPRINT_BITS), np.nan)
    fp_strings = df_chunk[PRECALCULATED_FP_STRING_COLUMN_NAME].values
    for idx, s in enumerate(fp_strings):
        if isinstance(s, str):
            try:
                bits = s.split(',')
                if len(bits) == NUM_FINGERPRINT_BITS:
                    fp_matrix[idx, :] = list(map(int, bits))
            except (ValueError, TypeError):
                continue
    fp_cols = [f"{FINGERPRINT_COLUMN_PREFIX}{i}" for i in range(
        NUM_FINGERPRINT_BITS)]
    df_parsed = pd.DataFrame(fp_matrix, columns=fp_cols, index=df_chunk.index)
    return pd.concat([df_chunk.drop(columns=[PRECALCULATED_FP_STRING_COLUMN_NAME]), df_parsed], axis=1)


def load_and_prepare_data(chembl_mf_path, zinc_path, repr_type, features_list):
    logger.info("STEP 1: Loading and preparing main dataset (MF Cloud + ZINC).")
    sample_df = pd.read_csv(chembl_mf_path, nrows=5)
    if repr_type == "fingerprints":
        sample_df = _parse_fingerprint_chunk(sample_df)
    descriptor_cols = get_descriptor_columns(
        sample_df, repr_type, features_list)
    if not descriptor_cols:
        raise ValueError("Could not determine descriptor columns.")
    logger.info(f"Using {len(descriptor_cols)} descriptor columns.")
    info_cols_to_keep = [
        'SMILES', 'Compound ChEMBL ID', 'ZINC_ID', 'DataSource']
    valid_info_chunks, valid_descriptor_chunks = [], []
    chunksize = 25000
    dtype_to_use = np.int8 if repr_type == "fingerprints" else np.float32
    input_files = {chembl_mf_path: "ChEMBL_MF"}
    if zinc_path and zinc_path.lower() != 'none' and os.path.exists(zinc_path):
        input_files[zinc_path] = "ZINC"
    for path, source in input_files.items():
        logger.info(f"Processing '{source}' from {path} in chunks...")
        for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
            chunk['DataSource'] = source
            if repr_type == "fingerprints":
                chunk = _parse_fingerprint_chunk(chunk)
            if repr_type == "features":
                chunk[descriptor_cols] = chunk[descriptor_cols].apply(
                    pd.to_numeric, errors='coerce')
            chunk.dropna(subset=descriptor_cols, how='any', inplace=True)
            if not chunk.empty:
                valid_info_chunks.append(
                    chunk[[c for c in info_cols_to_keep if c in chunk.columns]].copy())
                valid_descriptor_chunks.append(
                    chunk[descriptor_cols].values.astype(dtype_to_use))
            del chunk
            gc.collect()
    if not valid_descriptor_chunks:
        raise ValueError("No valid data rows found.")
    X_original = np.vstack(valid_descriptor_chunks)
    df_info = pd.concat(valid_info_chunks, ignore_index=True)
    logger.info(
        f"Loaded {len(df_info)} total valid records. Data matrix shape: {X_original.shape}")
    
    # DEDUPLICATE: Remove duplicate Compound ChEMBL IDs (safety check)
    if 'Compound ChEMBL ID' in df_info.columns:
        original_rows = len(df_info)
        unique_compounds_before = df_info['Compound ChEMBL ID'].nunique()
        
        dup_counts = df_info['Compound ChEMBL ID'].value_counts()
        duplicates = dup_counts[dup_counts > 1]
        
        if len(duplicates) > 0:
            logger.warning(f"⚠ Found {len(duplicates):,} duplicate Compound ChEMBL IDs in loaded data!")
            logger.warning(f"⚠ This should have been handled in prepare_data.py - deduplicating now as safety measure")
            
            # Keep first occurrence (arbitrary but consistent)
            df_info = df_info.drop_duplicates(subset='Compound ChEMBL ID', keep='first')
            X_original = X_original[df_info.index.values]
            df_info = df_info.reset_index(drop=True)
            
            logger.info(f"✓ Deduplicated: {original_rows:,} → {len(df_info):,} rows (removed {original_rows - len(df_info):,})")
        else:
            logger.info(f"✓ No duplicates detected ({unique_compounds_before:,} unique compounds)")
    
    if 'Compound ChEMBL ID' in df_info.columns and 'ZINC_ID' in df_info.columns:
        df_info['MOLECULE ID'] = df_info['Compound ChEMBL ID'].fillna(
            df_info['ZINC_ID'])
    else:
        df_info['MOLECULE ID'] = df_info.get('Compound ChEMBL ID', df_info.get(
            'ZINC_ID', 'ID_' + df_info.index.astype(str)))
    return df_info, X_original, descriptor_cols


def prepare_target_ligands(path, repr_type, features_list, scaler):
    """
    Load and prepare held-out target ligands for PROJECTION ONLY.
    These ligands will be transformed using a pre-fitted DR model.
    Co-embedding removed in v2.0 to prevent data leakage.
    """
    if not path or path.lower() == 'none' or not os.path.exists(path):
        logger.warning(
            f"Target ligands file not found or not provided. Projection will be skipped.")
        return None, None, None
    logger.info(f"Preparing held-out active ligands from {path} for projection.")
    df_target_raw = pd.read_csv(path, low_memory=False)
    if df_target_raw.empty:
        return None, None, None
    descriptor_cols = get_descriptor_columns(
        df_target_raw, repr_type, features_list)
    df_target_raw.dropna(subset=descriptor_cols, inplace=True)
    if df_target_raw.empty:
        return None, None, None
    dtype_to_use = np.int8 if repr_type == "fingerprints" else np.float32
    X_target_original = df_target_raw[descriptor_cols].values.astype(
        dtype_to_use)
    X_target_processed = scaler.transform(X_target_original)
    logger.info(
        f"Prepared {len(df_target_raw)} target ligands for projection.")
    return df_target_raw, X_target_original, X_target_processed


def run_pca(X_processed, df_info, config, out_paths, use_gpu):
    """
    Run PCA using PROJECTION-ONLY strategy (v2.0).
    Fits PCA model on training data (MF cloud + decoys) only.
    Held-out actives will be projected separately to prevent data leakage.
    """
    logger.info("--- Running PCA Projection ---")
    dr_cols = [f'PCA-{i+1}' for i in range(config['simspace_dim'])]
    logger.info(
        f"Fitting PCA projection model on main data ({X_processed.shape})...")
    model = cumlPCA(
        **config['cuml_params']) if use_gpu else sklearnPCA(**config['sklearn_params'])
    projection_coords = model.fit_transform(X_processed)
    save_model(model, out_paths['projection_model'])
    logger.info("PCA projection complete. Model saved.")
    return pd.DataFrame(projection_coords, columns=dr_cols, index=df_info.index)


# t-SNE removed in v2.0: Only supports co-embedding (data leakage), incompatible with projection-only strategy


def run_umap_for_metric(metric, X_dict, df_info, config, out_paths, use_gpu):
    """
    Run UMAP using PROJECTION-ONLY strategy (v2.0).
    Fits UMAP model on training data (MF cloud + decoys) only.
    Held-out actives will be projected separately to prevent data leakage.
    
    v2.0: Fingerprints ONLY support Jaccard metric (binary-native).
    """
    logger.info(f"--- Running UMAP Projection for metric: {metric} ---")
    
    # v2.0: Validate fingerprint + metric combinations
    if config['repr_type'] == "fingerprints" and metric.lower() != "jaccard":
        logger.error(f"ERROR: Fingerprints only support Jaccard metric in v2.0, not '{metric}'.")
        logger.error("Rationale: Non-Jaccard metrics require scaling, which destroys binary fingerprint meaning.")
        return pd.DataFrame(columns=out_paths['dr_cols'])
    
    if config['repr_type'] == "features":
        logger.info(f"UMAP ({metric}) on features: using SCALED data.")
        X_main = X_dict['scaled']
    elif config['repr_type'] == "fingerprints":
        logger.info(f"UMAP-Jaccard on fingerprints: using RAW BINARY 2048-bit data (no scaling, no PCA).")
        X_main = X_dict['original']
    
    if X_main is None:
        logger.error(f"Input data for UMAP ({metric}) is None. Skipping.")
        return pd.DataFrame(columns=out_paths['dr_cols'])
    
    logger.info(f"Fitting UMAP projection model ({metric}) on main data ({X_main.shape})...")
    
    # Use sklearn UMAP for Jaccard (cuML doesn't support it well)
    use_cuml_for_metric = use_gpu and metric.lower() not in ["jaccard", "hamming"]
    model = cumlUMAP(**config['cuml_params']) if use_cuml_for_metric else umapUMAP(
        metric=metric, **config['sklearn_params'])
    projection_coords = model.fit_transform(X_main)
    save_model(model, out_paths['projection_model'])
    logger.info(f"UMAP ({metric}) projection complete. Model saved.")
    return pd.DataFrame(projection_coords, columns=out_paths['dr_cols'], index=df_info.index)


def main():
    # --- Argument Parsing (Unchanged) ---
    parser = argparse.ArgumentParser(
        description="Calculate similarity spaces using various DR methods.")
    # ... all args ...
    parser.add_argument("--fingerprint_pca_components", type=int, default=50) 
    parser.add_argument("--log_file_path", required=True,
                        help="Path to the unified log file for appending.")
    parser.add_argument("--chembl_mf_data_path", required=True)
    parser.add_argument("--zinc_data_path", default=None)
    parser.add_argument(
        "--target_ligands_path_for_projection", default=None,
        help="Path to held-out target ligands for projection (v2.0 projection-only strategy)")
    parser.add_argument("--simspace_dim", type=int, required=True)
    parser.add_argument("--representation_type", required=True,
                        choices=["features", "fingerprints"])
    parser.add_argument("--target_id_name", required=True)
    parser.add_argument("--output_simspace_dir", required=True)
    parser.add_argument("--output_model_dir", required=True)
    parser.add_argument("--rdkit_features_list_target_str", required=True)
    parser.add_argument(
        "--dr_method_pca", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument(
        "--dr_method_umap", type=lambda x: (str(x).lower() == 'true'), default=False)
    # t-SNE removed in v2.0
    parser.add_argument("--umap_metric_to_run_euclidean",
                        action='store_true', default=False)
    parser.add_argument("--umap_metric_to_run_cosine",
                        action='store_true', default=False)
    parser.add_argument("--umap_metric_to_run_manhattan",
                        action='store_true', default=False)
    parser.add_argument("--umap_metric_to_run_hamming",
                        action='store_true', default=False)
    parser.add_argument("--umap_metric_to_run_jaccard",
                        action='store_true', default=False)
    # t-SNE arguments removed in v2.0
    parser.add_argument("--n_neighbors", type=int, default=None)
    # Co-embedding removed in v2.0 - projection only
    parser.add_argument("--dr_method_configs_json_str", required=True)
    parser.add_argument("--random_state", type=int, default=42,
                       help="Random seed for reproducibility (only used if --use_fixed_seed is True)")
    parser.add_argument("--use_fixed_seed", action='store_true', default=False,
                       help="Enable fixed random seed (disables UMAP multi-threading for exact reproducibility)")
    args = parser.parse_args()

    # --- Initial Setup ---
    setup_script_logging(args.log_file_path)
    GPU_ENABLED = is_gpu_available()
    
    # Log seed usage for clarity
    if args.use_fixed_seed:
        logger.info(f"🔒 FIXED SEED MODE: random_state={args.random_state} (reproducible, single-threaded UMAP)")
    else:
        logger.info(f"⚡ MULTI-THREADED MODE: NO fixed seed (10× faster UMAP, non-reproducible due to race conditions)")
        logger.info(f"   Note: Seed value {args.random_state} used for folder naming only, NOT for DR model initialization")

    features_list = json.loads(args.rdkit_features_list_target_str)
    dr_configs = json.loads(args.dr_method_configs_json_str)

    # 1. Load Data
    df_info, X_original, _ = load_and_prepare_data(
        args.chembl_mf_data_path, args.zinc_data_path, args.representation_type, features_list)

    # 2. Data Preparation (v2.0: Simplified)
    if args.representation_type == "features":
        logger.info("STEP 2: Scaling features data with StandardScaler.")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_original)
        save_model(scaler, os.path.join(args.output_model_dir,
                   f"{args.target_id_name}_features_dim{args.simspace_dim}_scaler.lzma"))
    else:  # fingerprints
        logger.info("STEP 2: Fingerprints detected - NO SCALING will be applied (v2.0: Jaccard only).")
        scaler = PassthroughScaler()
        X_scaled = X_original.copy()  # No transformation for fingerprints
        save_model(scaler, os.path.join(args.output_model_dir,
                   f"{args.target_id_name}_fingerprints_dim{args.simspace_dim}_scaler.lzma"))

    # 3. Prepare Target Ligands for projection (v2.0 projection-only strategy)
    df_target, X_target_original, X_target_scaled = prepare_target_ligands(
        args.target_ligands_path_for_projection, 
        args.representation_type, 
        features_list, 
        scaler
    )

    # 5. Run DR Methods
    logger.info("STEP 5: Running selected dimensionality reduction methods.")
    df_results = df_info.copy()
    base_name = f"{args.target_id_name}_{args.representation_type}_dim{args.simspace_dim}"

    if args.dr_method_pca:
        # v2.0: PCA works for both features and fingerprints (projection-only)
        # Note: For fingerprints, PCA treats binary vectors as numerical features
        # UMAP-Jaccard is preferred for fingerprints, but PCA is still valid for comparison
        
        # Conditionally add random_state based on use_fixed_seed flag
        cuml_params = {'n_components': args.simspace_dim}
        sklearn_params = {'n_components': args.simspace_dim}
        if args.use_fixed_seed:
            cuml_params['random_state'] = args.random_state
            sklearn_params['random_state'] = args.random_state
        
        pca_run_config = {
            'simspace_dim': args.simspace_dim,
            'cuml_params': cuml_params,
            'sklearn_params': sklearn_params
        }
        out_paths = {
            'projection_model': os.path.join(args.output_model_dir, f"{base_name}_PCA_model.lzma")
        }
        pca_coords = run_pca(X_scaled, df_info, pca_run_config, out_paths, GPU_ENABLED)
        df_results = df_results.join(pca_coords)

    # t-SNE execution removed in v2.0: Only supports co-embedding (data leakage)
    
    if args.dr_method_umap:
        # Projection-only in v2.0: only training data (no target data in dict)
        X_data_dict = {
            'original': X_original,  # Raw data (binary for fingerprints, unscaled for features)
            'scaled': X_scaled       # Scaled for features, unchanged for fingerprints (PassthroughScaler)
        }
        active_metrics = [m for m in ['euclidean', 'cosine', 'manhattan',
                                      'hamming', 'jaccard'] if getattr(args, f"umap_metric_to_run_{m}")]
        
        # v2.0: Warn if fingerprints used with non-Jaccard metrics
        if args.representation_type == "fingerprints":
            invalid_metrics = [m for m in active_metrics if m != 'jaccard']
            if invalid_metrics:
                logger.warning(f"WARNING: Fingerprints in v2.0 only support Jaccard metric.")
                logger.warning(f"Skipping invalid metrics for fingerprints: {invalid_metrics}")
                logger.warning("Rationale: Non-Jaccard metrics require scaling, destroying binary fingerprint meaning.")
        
        for metric in active_metrics:
            umap_params = dr_configs.get(f"umap_{metric}", {})
            n_neighbors = umap_params.get('n_neighbors', 15)
            min_dist = umap_params.get('min_dist', 0.1)
            logger.info(f"Preparing UMAP run for metric '{metric}' with n_neighbors={n_neighbors} and min_dist={min_dist}")

            if args.representation_type == "fingerprints":
                low_memory = True
            else:
                low_memory = False
            
            # Conditionally add random_state based on use_fixed_seed flag
            sklearn_params = {
                'n_components': args.simspace_dim,
                'n_neighbors': n_neighbors,
                'min_dist': min_dist,
                'low_memory': low_memory
            }
            cuml_params = {
                'n_components': args.simspace_dim,
                'n_neighbors': n_neighbors,
                'min_dist': min_dist,
                'metric': metric
            }
            
            if args.use_fixed_seed:
                sklearn_params['random_state'] = args.random_state
                cuml_params['random_state'] = args.random_state

            umap_config = {
                'repr_type': args.representation_type,
                'cuml_params': cuml_params,
                'sklearn_params': sklearn_params
            }

            out_paths = {
                'dr_cols': [f'UMAP-{metric.capitalize()}-{i+1}' for i in range(args.simspace_dim)],
                'projection_model': os.path.join(args.output_model_dir, f"{base_name}_{metric}_UMAP_model.lzma")
            }
            
            umap_coords = run_umap_for_metric(
                metric, X_data_dict, df_info, umap_config, out_paths, GPU_ENABLED)
            df_results = df_results.join(umap_coords)

    # 6. Save Final & 7. Cleanup
    logger.info("STEP 6: Saving final combined similarity space file.")
    df_results.to_csv(os.path.join(args.output_simspace_dir,
                      f"{base_name}_similarity_space.csv"), index=False)
    logger.info("STEP 7: Script finished.")
    gc.collect()


if __name__ == "__main__":
    main()
