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
from sklearn.manifold import TSNE as sklearnTSNE
from sklearn.preprocessing import StandardScaler

from core_scripts.utils import PassthroughScaler

# ... (Module imports and initial setup are unchanged) ...
try:
    from cuml import PCA as cumlPCA, UMAP as cumlUMAP, TSNE as cumlTSNE
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


def construct_coembedded_dataframe(df_main_info, df_target_info, coembed_coords, dr_col_names):
    df_coords = pd.DataFrame(coembed_coords, columns=dr_col_names)
    main_info = df_main_info[['MOLECULE ID', 'SMILES', 'DataSource']].copy()
    target_info = df_target_info[[
        'SMILES', 'Activity Type', 'Standard Value (nM)', 'accession']].copy()
    target_info['MOLECULE ID'] = df_target_info['Compound ChEMBL ID']
    target_info['DataSource'] = 'HELDOUT_ACTIVE'
    df_full_info = pd.concat([main_info, target_info], ignore_index=True)
    return pd.concat([df_full_info, df_coords], axis=1)


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
    if 'Compound ChEMBL ID' in df_info.columns and 'ZINC_ID' in df_info.columns:
        df_info['MOLECULE ID'] = df_info['Compound ChEMBL ID'].fillna(
            df_info['ZINC_ID'])
    else:
        df_info['MOLECULE ID'] = df_info.get('Compound ChEMBL ID', df_info.get(
            'ZINC_ID', 'ID_' + df_info.index.astype(str)))
    return df_info, X_original, descriptor_cols


def prepare_target_ligands(path, repr_type, features_list, scaler):
    if not path or path.lower() == 'none' or not os.path.exists(path):
        logger.warning(
            f"Target ligands file not found or not provided. Co-embedding will be skipped.")
        return None, None, None
    logger.info(f"Preparing held-out active ligands from {path}.")
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
        f"Prepared {len(df_target_raw)} target ligands for co-embedding.")
    return df_target_raw, X_target_original, X_target_processed
# (DR functions are unchanged)


def run_pca(X_processed, df_info, X_target_processed, df_target_info, config, out_paths, use_gpu):
    logger.info("--- Running PCA ---")
    dr_cols = [f'PCA-{i+1}' for i in range(config['simspace_dim'])]
    logger.info(
        f"Fitting PCA projection model on main data ({X_processed.shape})...")
    model = cumlPCA(
        **config['cuml_params']) if use_gpu else sklearnPCA(**config['sklearn_params'])
    projection_coords = model.fit_transform(X_processed)
    save_model(model, out_paths['projection_model'])
    if config['run_coembedding'] and X_target_processed is not None:
        logger.info("Running PCA co-embedding...")
        X_coembed = np.vstack((X_processed, X_target_processed))
        coembed_model = cumlPCA(
            **config['cuml_params']) if use_gpu else sklearnPCA(**config['sklearn_params'])
        coembed_coords = coembed_model.fit_transform(X_coembed)
        df_coembed = construct_coembedded_dataframe(
            df_info, df_target_info, coembed_coords, dr_cols)
        df_coembed.to_csv(out_paths['coembed_space'], index=False)
        logger.info(
            f"Saved co-embedded PCA space to {out_paths['coembed_space']}")
    return pd.DataFrame(projection_coords, columns=dr_cols, index=df_info.index)


def run_tsne(X_pre_reduced, df_info, X_target_pre_reduced, df_target_info, config, out_paths, use_gpu):
    logger.info("--- Running t-SNE ---")
    if X_target_pre_reduced is None:
        logger.warning("Cannot run t-SNE: held-out actives data is missing.")
        return pd.DataFrame(columns=out_paths['dr_cols'])
    X_coembed = np.vstack((X_pre_reduced, X_target_pre_reduced))
    sklearn_params = config['sklearn_params'].copy()
    adjusted_perplexity = min(sklearn_params.get(
        'perplexity', 30.0), X_coembed.shape[0] - 1)
    sklearn_params['perplexity'] = adjusted_perplexity
    model = cumlTSNE(**config['cuml_params']
                     ) if use_gpu else sklearnTSNE(**sklearn_params)
    logger.info(
        f"Fitting t-SNE on co-embedded data ({X_coembed.shape}) with perplexity={adjusted_perplexity}...")
    coembed_coords = model.fit_transform(X_coembed)
    df_coembed = construct_coembedded_dataframe(
        df_info, df_target_info, coembed_coords, out_paths['dr_cols'])
    df_coembed.to_csv(out_paths['coembed_space'], index=False)
    logger.info(
        f"Saved co-embedded t-SNE space to {out_paths['coembed_space']}")
    projection_coords = coembed_coords[:len(df_info), :]
    return pd.DataFrame(projection_coords, columns=out_paths['dr_cols'], index=df_info.index)


def run_umap_for_metric(metric, X_dict, df_info, df_target_info, config, out_paths, use_gpu):
    logger.info(f"--- Running UMAP for metric: {metric} with configuration: {config} ---")
    if config['repr_type'] == "features":
        logger.info(f"UMAP ({metric}) on features: using SCALED data.")
        X_main, X_target = X_dict['scaled'], X_dict['target_scaled']
    elif config['repr_type'] == "fingerprints":
        if metric.lower() in ["jaccard", "hamming"]:
            logger.info(
                f"UMAP ({metric}) on fingerprints: using RAW, UNPROCESSED, 2048-D data.")
            X_main, X_target = X_dict['original'], X_dict['target_original']
        else:
            logger.info(
                f"UMAP ({metric}) on fingerprints: using SCALED then PCA-REDUCED data.")
            X_main, X_target = X_dict['pca_reduced'], X_dict['target_pca_reduced']
    if X_main is None:
        logger.error(f"Input data for UMAP ({metric}) is None. Skipping.")
        return pd.DataFrame(columns=out_paths['dr_cols'])
    logger.info(
        f"Fitting UMAP projection model ({metric}) on main data ({X_main.shape})...")
    use_cuml_for_metric = use_gpu and metric.lower() not in [
        "jaccard", "hamming"]
    model = cumlUMAP(**config['cuml_params']) if use_cuml_for_metric else umapUMAP(
        metric=metric, **config['sklearn_params'])
    projection_coords = model.fit_transform(X_main)
    save_model(model, out_paths['projection_model'])
    if config['run_coembedding'] and X_target is not None:
        logger.info(f"Running UMAP co-embedding ({metric})...")
        X_coembed = np.vstack((X_main, X_target))
        coembed_model = cumlUMAP(**config['cuml_params']) if use_cuml_for_metric else umapUMAP(
            metric=metric, **config['sklearn_params'])
        coembed_coords = coembed_model.fit_transform(X_coembed)
        df_coembed = construct_coembedded_dataframe(
            df_info, df_target_info, coembed_coords, out_paths['dr_cols'])
        df_coembed.to_csv(out_paths['coembed_space'], index=False)
        logger.info(
            f"Saved co-embedded UMAP space ({metric}) to {out_paths['coembed_space']}")
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
        "--target_ligands_unscaled_path_for_tsne_and_coembed", default=None)
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
    parser.add_argument(
        "--dr_method_tsne", type=lambda x: (str(x).lower() == 'true'), default=False)
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
    parser.add_argument("--tsne_perplexity", type=float, default=30.0)
    parser.add_argument("--tsne_pca_components", type=int, default=50)
    parser.add_argument("--n_neighbors", type=int, default=None)
    parser.add_argument("--run_coembedding_for_pca_umap",
                        type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--dr_method_configs_json_str", required=True)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    # --- Initial Setup ---
    setup_script_logging(args.log_file_path)
    GPU_ENABLED = is_gpu_available()

    features_list = json.loads(args.rdkit_features_list_target_str)
    dr_configs = json.loads(args.dr_method_configs_json_str)

    # 1. Load Data
    df_info, X_original, _ = load_and_prepare_data(
        args.chembl_mf_data_path, args.zinc_data_path, args.representation_type, features_list)

    # --- FIX 1: HYBRID DATA PREPARATION ---
    # Prepare different data versions based on representation type.
    scaler_for_pca = StandardScaler()
    scaler_for_binary = PassthroughScaler()

    if args.representation_type == "features":
        logger.info("STEP 2: Scaling features data with StandardScaler.")
        X_scaled = scaler_for_pca.fit_transform(X_original)
        save_model(scaler_for_pca, os.path.join(args.output_model_dir,
                   f"{args.target_id_name}_features_dim{args.simspace_dim}_scaler.lzma"))
    else:  # fingerprints
        logger.info(
            "STEP 2: Preparing both scaled and unscaled versions of fingerprint data.")
        X_scaled = scaler_for_pca.fit_transform(X_original)
        # We save both potential "scalers" so the downstream script can load the correct one
        save_model(scaler_for_pca, os.path.join(args.output_model_dir,
                   f"{args.target_id_name}_fingerprints_dim{args.simspace_dim}_scaler_for_pca.lzma"))
        save_model(scaler_for_binary, os.path.join(args.output_model_dir,
                   f"{args.target_id_name}_fingerprints_dim{args.simspace_dim}_scaler_for_binary.lzma"))

    # 3. Prepare Target Ligands
    # This step is now more complex as we need to prepare targets with the correct scaler
    df_target, X_target_original, X_target_scaled = prepare_target_ligands(
        args.target_ligands_unscaled_path_for_tsne_and_coembed, args.representation_type, features_list, scaler_for_pca
    )
    _, _, X_target_binary = prepare_target_ligands(
        args.target_ligands_unscaled_path_for_tsne_and_coembed, args.representation_type, features_list, scaler_for_binary
    )

    # --- FIX 2: UNIVERSAL PCA PRE-REDUCTION FOR T-SNE ---
    # 4. Pre-reduce Data with PCA for Manifold Learning
    X_pca_reduced, X_target_pca_reduced = None, None
    if args.representation_type == "fingerprints":
        logger.info("STEP 4: Pre-reducing SCALED fingerprints with PCA for manifold learning.")
        n_comps = min(args.fingerprint_pca_components, X_scaled.shape[0] - 1, X_scaled.shape[1])
        pca_pre_model = cumlPCA(n_components=n_comps) if GPU_ENABLED else sklearnPCA(n_components=n_comps)
        X_pca_reduced = pca_pre_model.fit_transform(X_scaled)
        if X_target_scaled is not None:
            X_target_pca_reduced = pca_pre_model.transform(X_target_scaled)
        logger.info(f"Fingerprint PCA pre-reduction complete. New shape: {X_pca_reduced.shape}")
    
    elif args.representation_type == "features" and args.dr_method_tsne:
        # This block is now ONLY for t-SNE on features, restoring the old correct behavior
        logger.info("STEP 4: Pre-reducing SCALED features with PCA for t-SNE.")
        n_comps = min(args.tsne_pca_components, X_scaled.shape[0] - 1, X_scaled.shape[1])
        pca_pre_model = cumlPCA(n_components=n_comps) if GPU_ENABLED else sklearnPCA(n_components=n_comps)
        X_pca_reduced = pca_pre_model.fit_transform(X_scaled)
        if X_target_scaled is not None:
            X_target_pca_reduced = pca_pre_model.transform(X_target_scaled)
        logger.info(f"Feature PCA pre-reduction for t-SNE complete. New shape: {X_pca_reduced.shape}")

    # 5. Run DR Methods
    logger.info("STEP 5: Running selected dimensionality reduction methods.")
    df_results = df_info.copy()
    base_name = f"{args.target_id_name}_{args.representation_type}_dim{args.simspace_dim}"

    if args.dr_method_pca:
        # PCA always uses the scaled data
        X_main_pca = X_scaled
        X_target_pca = X_target_scaled
        pca_run_config = {'simspace_dim': args.simspace_dim, 'run_coembedding': args.run_coembedding_for_pca_umap, 'cuml_params': {
            'n_components': args.simspace_dim, 'random_state': args.random_state}, 'sklearn_params': {'n_components': args.simspace_dim, 'random_state': args.random_state}}
        out_paths = {'projection_model': os.path.join(args.output_model_dir, f"{base_name}_PCA_model.lzma"), 'coembed_space': os.path.join(
            args.output_simspace_dir, f"{base_name}_PCA_similarity_space_COEMBED.csv")}
        pca_coords = run_pca(X_scaled, df_info, X_target_scaled,
                             df_target, pca_run_config, out_paths, GPU_ENABLED)
        df_results = df_results.join(pca_coords)

    if args.dr_method_tsne:
        # t-SNE can now run in any dimensionality (2, 3, 5, 10, 20, etc.)
        tsne_run_config = {'cuml_params': {'n_components': args.simspace_dim, 'random_state': args.random_state}, 'sklearn_params': {
            'n_components': args.simspace_dim, 'perplexity': args.tsne_perplexity, 'random_state': args.random_state, 'n_jobs': -1}}
        out_paths = {'dr_cols': [f't-SNE-{i+1}' for i in range(args.simspace_dim)], 'coembed_space': os.path.join(
            args.output_simspace_dir, f"{base_name}_tSNE_similarity_space_COEMBED.csv")}
        # t-SNE now always uses the PCA-reduced data
        tsne_coords = run_tsne(X_pca_reduced, df_info, X_target_pca_reduced,
                               df_target, tsne_run_config, out_paths, GPU_ENABLED)
        df_results = df_results.join(tsne_coords)

    if args.dr_method_umap:
        X_data_dict = {
            'original': X_original, 'scaled': X_scaled, 'pca_reduced': X_pca_reduced,
            'target_original': X_target_original, 'target_scaled': X_target_scaled, 'target_pca_reduced': X_target_pca_reduced
        }
        active_metrics = [m for m in ['euclidean', 'cosine', 'manhattan',
                                      'hamming', 'jaccard'] if getattr(args, f"umap_metric_to_run_{m}")]
        for metric in active_metrics:
            umap_params = dr_configs.get(f"umap_{metric}", {})
            n_neighbors = umap_params.get('n_neighbors', 15)
            min_dist = umap_params.get('min_dist', 0.1) # Get min_dist, with a default
            logger.info(f"Preparing UMAP run for metric '{metric}' with n_neighbors={n_neighbors} and min_dist={min_dist}")

            sklearn_params = {'n_components': args.simspace_dim, 'random_state': args.random_state, 'n_neighbors': n_neighbors, 'min_dist': min_dist}
            cuml_params = {'n_components': args.simspace_dim, 'random_state': args.random_state, 'n_neighbors': n_neighbors, 'min_dist': min_dist, 'metric': metric}

            umap_config = {'repr_type': args.representation_type, 'run_coembedding': args.run_coembedding_for_pca_umap,
                           'cuml_params': cuml_params, 'sklearn_params': sklearn_params}


            out_paths = {'dr_cols': [f'UMAP-{metric.capitalize()}-{i+1}' for i in range(args.simspace_dim)], 'projection_model': os.path.join(args.output_model_dir,
                                                                                                                                              f"{base_name}_{metric}_UMAP_model.lzma"), 'coembed_space': os.path.join(args.output_simspace_dir, f"{base_name}_{metric}_UMAP_similarity_space_COEMBED.csv")}
            
            umap_coords = run_umap_for_metric(
                metric, X_data_dict, df_info, df_target, umap_config, out_paths, GPU_ENABLED)
            df_results = df_results.join(umap_coords)

    # 6. Save Final & 7. Cleanup
    logger.info("STEP 6: Saving final combined similarity space file.")
    df_results.to_csv(os.path.join(args.output_simspace_dir,
                      f"{base_name}_similarity_space.csv"), index=False)
    logger.info("STEP 7: Script finished.")
    gc.collect()


if __name__ == "__main__":
    main()
