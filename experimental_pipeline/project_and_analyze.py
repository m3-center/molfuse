import pandas as pd
import numpy as np
import os
import argparse
import logging
import json # For parsing features list
from compress_pickle import load as decompress_pickle_load
from scipy.spatial import distance
import matplotlib.pyplot as plt
import seaborn as sns

# CUML imports (only for using_device_type if UMAP model is cuML and transform is separate)
try:
    from cuml.common.device_selection import using_device_type
    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("project_analyze.log"), logging.StreamHandler()])

FINGERPRINT_COLUMN_PREFIX = "fp_"
NUM_FINGERPRINT_BITS = 2048

def get_descriptor_columns_for_proj(df, representation_type, target_rdkit_features_list):
    """Identifies descriptor columns for projection based on representation type."""
    if representation_type == "features":
        descriptor_cols = [col for col in target_rdkit_features_list if col in df.columns]
        if not descriptor_cols:
            logging.error("No target RDKit features found in target ligands data for projection.")
            return None
    elif representation_type == "fingerprints":
        descriptor_cols = [col for col in df.columns if col.startswith(FINGERPRINT_COLUMN_PREFIX)]
        if not descriptor_cols:
            logging.error("No fingerprint columns found in target ligands data for projection.")
            return None
    else:
        logging.error(f"Invalid representation_type for projection: {representation_type}")
        return None
    return descriptor_cols


def project_target_ligands_with_models(target_ligands_repr_df, scaler_model, dr_model, 
                                       representation_type, simspace_dim, dr_method_key, dr_short_name,
                                       target_rdkit_features_list):
    """
    Projects target ligands using loaded scaler and DR models (PCA/UMAP).
    """
    if target_ligands_repr_df.empty:
        logging.warning("Target ligands DataFrame for projection is empty.")
        return pd.DataFrame(columns=[f"{dr_short_name}-{i+1}" for i in range(simspace_dim)])


    descriptor_columns = get_descriptor_columns_for_proj(target_ligands_repr_df, representation_type, target_rdkit_features_list)
    if not descriptor_columns:
        return pd.DataFrame(columns=[f"{dr_short_name}-{i+1}" for i in range(simspace_dim)])

    # Ensure essential ID columns are present for merging back
    id_cols_present = [col for col in ['SMILES', 'Compound ChEMBL ID'] if col in target_ligands_repr_df.columns]
    original_ids_df = target_ligands_repr_df[id_cols_present].copy() if id_cols_present else pd.DataFrame(index=target_ligands_repr_df.index)
    
    # Convert to numeric and handle NaNs by dropping rows *before* scaling
    target_ligands_repr_df[descriptor_columns] = target_ligands_repr_df[descriptor_columns].apply(pd.to_numeric, errors='coerce')
    
    num_before_dropna = len(target_ligands_repr_df)
    # Keep track of original indices of rows that are kept
    valid_rows_mask = target_ligands_repr_df[descriptor_columns].notna().all(axis=1)
    target_ligands_valid_df = target_ligands_repr_df[valid_rows_mask].copy()
    
    if len(target_ligands_valid_df) < num_before_dropna:
        logging.info(f"Dropped {num_before_dropna - len(target_ligands_valid_df)} target ligands with NaN descriptor values before projection.")

    if target_ligands_valid_df.empty:
        logging.warning("Target ligands DataFrame is empty after NaN drop. No projection possible with models.")
        return pd.DataFrame(columns=[f"{dr_short_name}-{i+1}" for i in range(simspace_dim)])

    X_target = target_ligands_valid_df[descriptor_columns].values
    
    try:
        X_target_scaled = scaler_model.transform(X_target)
    except Exception as e:
        logging.error(f"Error scaling target ligands for {dr_short_name}: {e}")
        return pd.DataFrame(columns=[f"{dr_short_name}-{i+1}" for i in range(simspace_dim)])
    
    projected_coords = None
    try:
        if CUML_AVAILABLE and 'umap' in dr_method_key.lower() and hasattr(dr_model, 'transform'):
            with using_device_type('cpu'): # Ensure UMAP transform consistency if model trained on GPU
                projected_coords = dr_model.transform(X_target_scaled)
        else:
            projected_coords = dr_model.transform(X_target_scaled)
    except Exception as e:
        logging.error(f"Error transforming target ligands with DR model {dr_short_name}: {e}")
        return pd.DataFrame(columns=[f"{dr_short_name}-{i+1}" for i in range(simspace_dim)])

    projection_cols = [f"{dr_short_name}-{i+1}" for i in range(simspace_dim)]
    # Use index from target_ligands_valid_df to align with original_ids_df
    df_projected_coords = pd.DataFrame(projected_coords[:, :simspace_dim], columns=projection_cols, index=target_ligands_valid_df.index)
    
    # Merge back with original IDs using the preserved valid indices
    df_projected_ligands_with_ids = original_ids_df.loc[df_projected_coords.index].join(df_projected_coords)
    
    return df_projected_ligands_with_ids.reset_index(drop=True)


def calculate_distances_to_cloud(projected_ligands_df_with_ids, mf_cloud_coords_df, k_for_knn_list, dr_short_name, simspace_dim):
    """Calculates various distances from projected ligands to the MF cloud."""
    if projected_ligands_df_with_ids.empty:
        logging.warning("Projected target ligands DataFrame is empty. Cannot calculate distances.")
        return pd.DataFrame()
    if mf_cloud_coords_df.empty:
        logging.warning("MF cloud coordinates DataFrame is empty. Distances will be NaN.")
        # Create an empty df with expected distance columns
        dist_cols = ['min_dist_to_mf_cloud'] + \
                    [f'avg_dist_top_{k}_in_mf_cloud' for k in k_for_knn_list] + \
                    ['dist_to_mf_cloud_centroid']
        id_cols_present = [col for col in ['SMILES', 'Compound ChEMBL ID'] if col in projected_ligands_df_with_ids.columns]
        return pd.DataFrame(columns=id_cols_present + dist_cols)


    coord_cols = [f"{dr_short_name}-{i+1}" for i in range(simspace_dim)]
    if not all(col in projected_ligands_df_with_ids.columns for col in coord_cols) or \
       not all(col in mf_cloud_coords_df.columns for col in coord_cols):
        logging.error(f"Coordinate columns mismatch for distance calculation ({dr_short_name}). "
                      f"Projected: {projected_ligands_df_with_ids.columns.tolist()}, MF Cloud: {mf_cloud_coords_df.columns.tolist()}")
        return pd.DataFrame() # Or return projected_ligands_df_with_ids with NaN distances

    projected_points = projected_ligands_df_with_ids[coord_cols].values
    mf_cloud_points = mf_cloud_coords_df[coord_cols].values

    all_distances_data = []
    # Re-attach IDs after distance calculations
    id_df_part = projected_ligands_df_with_ids[[col for col in ['SMILES', 'Compound ChEMBL ID'] if col in projected_ligands_df_with_ids.columns]].copy()


    if mf_cloud_points.shape[0] > 0 :
        pairwise_distances = distance.cdist(projected_points, mf_cloud_points, 'euclidean')
        id_df_part['min_dist_to_mf_cloud'] = np.min(pairwise_distances, axis=1)
        
        for k_val in k_for_knn_list:
            if mf_cloud_points.shape[0] >= k_val:
                sorted_pairwise_dists = np.sort(pairwise_distances, axis=1)
                id_df_part[f'avg_dist_top_{k_val}_in_mf_cloud'] = np.mean(sorted_pairwise_dists[:, :k_val], axis=1)
            else:
                id_df_part[f'avg_dist_top_{k_val}_in_mf_cloud'] = np.nan
        
        mf_cloud_centroid = np.mean(mf_cloud_points, axis=0)
        dist_to_centroid_vals = distance.cdist(projected_points, mf_cloud_centroid.reshape(1, -1), 'euclidean')
        id_df_part['dist_to_mf_cloud_centroid'] = dist_to_centroid_vals.flatten()
    else: # MF cloud is empty
        id_df_part['min_dist_to_mf_cloud'] = np.nan
        for k_val in k_for_knn_list: id_df_part[f'avg_dist_top_{k_val}_in_mf_cloud'] = np.nan
        id_df_part['dist_to_mf_cloud_centroid'] = np.nan
        
    return id_df_part


def plot_projection_results(df_projected_target_ligands, df_simspace_main_data, mf_cloud_coords_for_plot, 
                            distances_df_with_ids, output_dir, target_id_name, dr_short_name, simspace_dim, 
                            representation_type):
    """Generates and saves plots if simspace_dim is 2."""
    # ... (Plotting logic from previous version of project_and_analyze.py - largely unchanged) ...
    # Ensure it uses df_projected_target_ligands for red 'x' markers,
    # mf_cloud_coords_for_plot for blue MF cloud markers,
    # and a sample from df_simspace_main_data for ZINC grey markers.
    # The column names for coordinates must be consistent (e.g., PCA-1, PCA-2 or UMAP-Metric-1, UMAP-Metric-2).
    if simspace_dim != 2:
        logging.debug(f"Skipping 2D plots as simspace_dim is {simspace_dim}.")
        return

    coord_cols_2d = [f"{dr_short_name}-1", f"{dr_short_name}-2"]
    
    # Check if necessary coordinate columns exist in all relevant dataframes
    if not all(col in df_projected_target_ligands.columns for col in coord_cols_2d):
        logging.warning(f"Projected target ligands missing 2D coordinate columns for plotting: {coord_cols_2d}")
        return
    if not mf_cloud_coords_for_plot.empty and not all(col in mf_cloud_coords_for_plot.columns for col in coord_cols_2d):
        logging.warning(f"MF cloud data missing 2D coordinate columns for plotting: {coord_cols_2d}")
        # Don't return, plot what we can
    if not df_simspace_main_data.empty and not all(col in df_simspace_main_data.columns for col in coord_cols_2d):
        logging.warning(f"Main simspace data missing 2D coordinate columns for ZINC sample plotting: {coord_cols_2d}")
        # Don't return, plot what we can

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 10))
    
    # Plot ZINC sample (if present in main_data)
    # Heuristic: MOLECULE ID starts with ZINC or ZINC_ID column exists
    df_zinc_sample = pd.DataFrame()
    if 'MOLECULE ID' in df_simspace_main_data.columns:
        zinc_rows = df_simspace_main_data[df_simspace_main_data['MOLECULE ID'].astype(str).str.startswith('ZINC', na=False)]
        if not zinc_rows.empty:
             df_zinc_sample = zinc_rows.sample(n=min(1000, len(zinc_rows)), random_state=42, replace=False)
    elif 'ZINC_ID' in df_simspace_main_data.columns:
        zinc_rows = df_simspace_main_data[df_simspace_main_data['ZINC_ID'].notna()]
        if not zinc_rows.empty:
            df_zinc_sample = zinc_rows.sample(n=min(1000, len(zinc_rows)), random_state=42, replace=False)

    if not df_zinc_sample.empty and all(col in df_zinc_sample.columns for col in coord_cols_2d):
        plt.scatter(df_zinc_sample[coord_cols_2d[0]], df_zinc_sample[coord_cols_2d[1]], 
                    label="ZINC Sample", alpha=0.2, s=15, color='darkgrey', marker='.')
    
    # Plot MF Cloud
    if not mf_cloud_coords_for_plot.empty and all(col in mf_cloud_coords_for_plot.columns for col in coord_cols_2d):
        plt.scatter(mf_cloud_coords_for_plot[coord_cols_2d[0]], mf_cloud_coords_for_plot[coord_cols_2d[1]], 
                    label="Molecular Function Cloud (ChEMBL)", alpha=0.4, s=25, color='dodgerblue', marker='o')
    
    # Plot Projected Target Ligands
    plt.scatter(df_projected_target_ligands[coord_cols_2d[0]], df_projected_target_ligands[coord_cols_2d[1]], 
                label=f"Projected Target Ligands ({target_id_name})", alpha=0.9, s=60, color='red', marker='x', edgecolor='black')

    plt.title(f"Similarity Space ({representation_type}, {dr_short_name}, Dim={simspace_dim})\nTarget: {target_id_name}", fontsize=14)
    plt.xlabel(coord_cols_2d[0], fontsize=12)
    plt.ylabel(coord_cols_2d[1], fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    scatter_path = os.path.join(output_dir, f"{target_id_name}_{representation_type}_{dr_short_name.replace('-', '_')}_dim{simspace_dim}_scatter.png")
    try:
        plt.savefig(scatter_path, dpi=200, bbox_inches='tight')
        logging.info(f"Saved scatter plot to {scatter_path}")
    except Exception as e:
        logging.error(f"Failed to save scatter plot {scatter_path}: {e}")
    plt.close()

    # Histogram of Minimum Distances
    if 'min_dist_to_mf_cloud' in distances_df_with_ids.columns and distances_df_with_ids['min_dist_to_mf_cloud'].notna().any():
        plt.figure(figsize=(9, 7))
        sns.histplot(distances_df_with_ids['min_dist_to_mf_cloud'].dropna(), kde=True, bins=20, color='skyblue', edgecolor='black')
        mean_dist = distances_df_with_ids['min_dist_to_mf_cloud'].mean()
        median_dist = distances_df_with_ids['min_dist_to_mf_cloud'].median()
        plt.title(f"Min. Distances of Target Ligands to MF Cloud\n{target_id_name} ({representation_type}, {dr_short_name}, Dim={simspace_dim})", fontsize=14)
        plt.xlabel("Minimum Euclidean Distance to MF Cloud", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        if pd.notna(mean_dist): plt.axvline(mean_dist, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {mean_dist:.3f}')
        if pd.notna(median_dist): plt.axvline(median_dist, color='green', linestyle='dashed', linewidth=1.5, label=f'Median: {median_dist:.3f}')
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.5)
        hist_path = os.path.join(output_dir, f"{target_id_name}_{representation_type}_{dr_short_name.replace('-', '_')}_dim{simspace_dim}_min_distances_hist.png")
        try:
            plt.savefig(hist_path, dpi=200, bbox_inches='tight')
            logging.info(f"Saved min distance histogram to {hist_path}")
        except Exception as e:
            logging.error(f"Failed to save histogram {hist_path}: {e}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Project target ligands and analyze distances.")
    parser.add_argument("--target_ligands_repr_path", default=None, help="Path to CSV of target ligands (featurized/fingerprinted) - for PCA/UMAP.")
    parser.add_argument("--precomputed_target_projections_path", default=None, help="Path to CSV of pre-calculated target ligand projections (used for t-SNE).")
    parser.add_argument("--simspace_csv_path", required=True, help="Path to the comprehensive similarity space CSV (main data).")
    parser.add_argument("--model_dir_for_projection", default=None, help="Directory containing scaler and DR models (for PCA/UMAP).")
    parser.add_argument("--model_name_root_for_projection", default=None, help="Base name for finding models (e.g., target_repr_dimX).")
    parser.add_argument("--dr_method_key", required=True, help="Key for DR method from config (e.g., 'pca', 'umap_euclidean', 'tsne').")
    parser.add_argument("--dr_short_name", required=True, help="Short name for DR method (e.g., 'PCA', 'UMAP-Euclidean', 't-SNE').")
    parser.add_argument("--simspace_dim", type=int, required=True, help="Dimensionality of the similarity space.")
    parser.add_argument("--k_for_knn", type=str, required=True, help="Comma-separated list of k values for k-NN distance (e.g., '3,5').")
    parser.add_argument("--output_dir", required=True, help="Directory to save analysis results.")
    parser.add_argument("--target_id_name", required=True, help="Target ID name for file naming.")
    parser.add_argument("--representation_type", required=True, choices=["features", "fingerprints"])
    parser.add_argument("--rdkit_features_list_target_str", required=True, help="JSON string of target RDKit feature names.")
    args = parser.parse_args()

    try:
        target_rdkit_features_list = json.loads(args.rdkit_features_list_target_str)
    except Exception as e:
        logging.error(f"Error parsing --rdkit_features_list_target_str: {e}. Aborting project_and_analyze.")
        return

    k_for_knn_list = [int(k.strip()) for k in args.k_for_knn.split(',')]
    df_projected_target_ligands_with_ids = pd.DataFrame()

    if args.precomputed_target_projections_path and os.path.exists(args.precomputed_target_projections_path):
        logging.info(f"Loading pre-computed target projections for {args.dr_short_name} from: {args.precomputed_target_projections_path}")
        try:
            df_projected_target_ligands_with_ids = pd.read_csv(args.precomputed_target_projections_path)
            # Ensure coordinate columns match dr_short_name convention
            expected_coord_cols = [f"{args.dr_short_name}-{i+1}" for i in range(args.simspace_dim)]
            if not all(col in df_projected_target_ligands_with_ids.columns for col in expected_coord_cols):
                # Attempt to rename if cols are just '0', '1', ... or 't-SNE-1' but short_name is 'tSNE'
                # This part might need to be more robust depending on how t-SNE projections are saved
                rename_map = {}
                for i in range(args.simspace_dim):
                    if str(i) in df_projected_target_ligands_with_ids.columns and expected_coord_cols[i] not in df_projected_target_ligands_with_ids.columns:
                        rename_map[str(i)] = expected_coord_cols[i]
                    elif f"t-SNE-{i+1}" in df_projected_target_ligands_with_ids.columns and args.dr_short_name == "t-SNE" and expected_coord_cols[i] not in df_projected_target_ligands_with_ids.columns:
                         # This case should already be fine if calc_simspace saves with "t-SNE-1" etc.
                         pass 
                if rename_map:
                    logging.info(f"Renaming columns in precomputed projections: {rename_map}")
                    df_projected_target_ligands_with_ids.rename(columns=rename_map, inplace=True)
        except Exception as e:
            logging.error(f"Error loading pre-computed target projections: {e}")
            # Fall through to model-based projection if possible, or fail if that's not an option.
            # For t-SNE, this is the only way, so if it fails, we can't proceed for t-SNE.
            if args.dr_method_key == "tsne": return
    
    if df_projected_target_ligands_with_ids.empty and args.dr_method_key != "tsne": # Not t-SNE or t-SNE precomputed failed
        if not (args.target_ligands_repr_path and args.model_dir_for_projection and args.model_name_root_for_projection):
            logging.error("Required paths for model-based projection (PCA/UMAP) not provided and no precomputed t-SNE found. Aborting.")
            return
        logging.info(f"Projecting target ligands for {args.target_id_name} ({args.representation_type}, {args.dr_short_name}, dim={args.simspace_dim}) using models...")
        try:
            scaler_path = os.path.join(args.model_dir_for_projection, f"{args.model_name_root_for_projection}_scaler.lzma")
            dr_model_filename_part = ""
            if args.dr_method_key == "pca": dr_model_filename_part = "PCA_model.lzma"
            elif "umap" in args.dr_method_key:
                metric_for_filename = args.dr_method_key.split("_")[1]
                dr_model_filename_part = f"{metric_for_filename}_UMAP_model.lzma"
            
            if not dr_model_filename_part: 
                logging.error(f"Could not determine DR model filename for key: {args.dr_method_key}"); return
            dr_model_path = os.path.join(args.model_dir_for_projection, f"{args.model_name_root_for_projection}_{dr_model_filename_part}")

            with open(scaler_path, "rb") as f: scaler_model = decompress_pickle_load(f)
            with open(dr_model_path, "rb") as f: dr_model = decompress_pickle_load(f)
            
            df_target_ligands_repr = pd.read_csv(args.target_ligands_repr_path, low_memory=False)
            df_projected_target_ligands_with_ids = project_target_ligands_with_models(
                df_target_ligands_repr, scaler_model, dr_model,
                args.representation_type, args.simspace_dim, args.dr_method_key, args.dr_short_name,
                target_rdkit_features_list
            )
        except Exception as e:
            logging.error(f"Error during model loading or projection for {args.dr_short_name}: {e}")
            # Create empty df with ID columns if projection fails, so distance saving doesn't break
            df_projected_target_ligands_with_ids = pd.DataFrame(columns=['SMILES', 'Compound ChEMBL ID'] + [f"{args.dr_short_name}-{i+1}" for i in range(args.simspace_dim)])


    if df_projected_target_ligands_with_ids.empty:
        logging.warning(f"No target ligands projected for {args.dr_short_name}. Skipping distance analysis and plotting.")
    else:
        logging.info(f"Successfully projected/loaded {len(df_projected_target_ligands_with_ids)} target ligands for {args.dr_short_name}.")

    # --- Load comprehensive simspace for MF cloud and ZINC sample ---
    try:
        df_simspace_main_data = pd.read_csv(args.simspace_csv_path, low_memory=False)
    except Exception as e:
        logging.error(f"Error loading main similarity space file {args.simspace_csv_path}: {e}")
        df_simspace_main_data = pd.DataFrame() # Allow proceeding to save empty distances if this fails

    # --- Prepare MF Cloud Coordinates from the main simspace data ---
    mf_cloud_coords_for_dist_calc = pd.DataFrame()
    if not df_simspace_main_data.empty:
        # ChEMBL MF cloud = non-ZINC entries from the main simspace data
        # (assuming main simspace was built from chembl_mf_excluded and zinc_excluded)
        # A robust way to identify ChEMBL entries: presence of 'Compound ChEMBL ID' or 'accession'
        # and absence of 'ZINC_ID' (if ZINC_ID is a reliable marker for ZINC only)
        chembl_mask = pd.Series(False, index=df_simspace_main_data.index)
        if 'accession' in df_simspace_main_data.columns:
            chembl_mask = df_simspace_main_data['accession'].notna()
        elif 'Compound ChEMBL ID' in df_simspace_main_data.columns:
            chembl_mask = df_simspace_main_data['Compound ChEMBL ID'].notna()
        
        # Further ensure they are not ZINC if ZINC_ID is present
        if 'ZINC_ID' in df_simspace_main_data.columns:
            chembl_mask &= df_simspace_main_data['ZINC_ID'].isna()

        mf_cloud_df_full = df_simspace_main_data[chembl_mask].copy()
        
        mf_cloud_coord_cols = [f"{args.dr_short_name}-{i+1}" for i in range(args.simspace_dim)]
        if not mf_cloud_df_full.empty and all(col in mf_cloud_df_full.columns for col in mf_cloud_coord_cols):
            mf_cloud_coords_for_dist_calc = mf_cloud_df_full[mf_cloud_coord_cols].dropna()
        else:
            logging.warning(f"MF Cloud data from main simspace is empty or missing coordinate columns for {args.dr_short_name}.")
    else:
        logging.warning("Main similarity space CSV is empty. MF Cloud will be empty.")


    # --- Calculate Distances ---
    logging.info(f"Calculating distances to MF cloud for {args.dr_short_name}...")
    distances_df = calculate_distances_to_cloud(
        df_projected_target_ligands_with_ids, 
        mf_cloud_coords_for_dist_calc, 
        k_for_knn_list, 
        args.dr_short_name,
        args.simspace_dim
    )
    
    distances_csv_path = os.path.join(args.output_dir, f"{args.target_id_name}_{args.representation_type}_{args.dr_short_name.replace('-', '_')}_dim{args.simspace_dim}_distances.csv")
    try:
        distances_df.to_csv(distances_csv_path, index=False)
        logging.info(f"Saved distances analysis to: {distances_csv_path} ({len(distances_df)} ligand distance entries)")
    except Exception as e:
        logging.error(f"Failed to save distances CSV {distances_csv_path}: {e}")

    # --- Generate Plots (if 2D) ---
    if args.simspace_dim == 2 and not df_projected_target_ligands_with_ids.empty:
        logging.info("Generating 2D plots...")
        plot_projection_results(df_projected_target_ligands_with_ids, df_simspace_main_data, 
                                mf_cloud_coords_for_dist_calc, # Pass the coordinate-only df for plotting MF cloud
                                distances_df, args.output_dir, args.target_id_name, 
                                args.dr_short_name, args.simspace_dim, args.representation_type)

    logging.info(f"Projection and analysis for {args.target_id_name} ({args.representation_type}, {args.dr_short_name}, dim={args.simspace_dim}) completed.")

if __name__ == "__main__":
    main()