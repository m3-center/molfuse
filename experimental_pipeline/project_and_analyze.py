import pandas as pd
import numpy as np
import os
import argparse
import logging
import json
from compress_pickle import load as decompress_pickle_load
from scipy.spatial import distance # For cdist, distance calculations
import matplotlib.pyplot as plt
import seaborn as sns

# CUML for GPU acceleration if available (for UMAP transform if model is cuML)
try:
    from cuml.common.device_selection import using_device_type
    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Feature list (must match how features were generated and how models were trained)
RDKIT_FEATURES_LIST_TARGET = [
    'DipoleMoment','ABC','nAcid','nBase','nAromAtom','nAtom','nH','nC','nN','nO','nS',
    'nP','nX','nBonds','nBondsO','nBondsS','nBondsD','nBondsT','nBondsA','nBondsM',
    'nBondsKS','nBondsKD','EState_VSA7','nHBAcc','nHBDon','Lipinski','apol','bpol',
    'nRing','n3Ring','n4Ring','n5Ring','n6Ring','n7Ring','n8Ring','nRot','Diameter',
    'TopoShapeIndex','Vabc','MW'
]
FINGERPRINT_COLUMN_PREFIX = "fp_"
NUM_FINGERPRINT_BITS = 2048


def project_target_ligands(target_ligands_repr_df, scaler_model, dr_model, 
                           representation_type, simspace_dim, dr_method_key, dr_short_name):
    """
    Projects target ligands into the similarity space.
    - Loads target ligands (already featurized/fingerprinted).
    - Applies scaling and dimensionality reduction.
    """
    if target_ligands_repr_df.empty:
        logging.warning("Target ligands DataFrame is empty. No projection possible.")
        return pd.DataFrame()

    if representation_type == "features":
        descriptor_columns = RDKIT_FEATURES_LIST_TARGET
    elif representation_type == "fingerprints":
        descriptor_columns = [f"{FINGERPRINT_COLUMN_PREFIX}{i}" for i in range(NUM_FINGERPRINT_BITS)]
    else:
        raise ValueError(f"Invalid representation_type: {representation_type}")

    # Ensure all descriptor columns are present in the target ligands df
    missing_cols = [col for col in descriptor_columns if col not in target_ligands_repr_df.columns]
    if missing_cols:
        logging.error(f"Missing descriptor columns in target ligands data for projection: {missing_cols}")
        # Add missing columns with NaN to allow processing, though results will be NaN for these rows
        for col in missing_cols: target_ligands_repr_df[col] = np.nan
        # return pd.DataFrame() # Or allow projection with NaNs if models can handle (they usually can't for fit, but transform might work differently)

    target_ligands_repr_df[descriptor_columns] = target_ligands_repr_df[descriptor_columns].apply(pd.to_numeric, errors='coerce')
    
    # Store original indices and SMILES/IDs before dropping NaNs
    original_indices = target_ligands_repr_df.index
    id_cols = [col for col in ['SMILES', 'Compound ChEMBL ID', 'MOLECULE ID'] if col in target_ligands_repr_df.columns]
    original_ids_df = target_ligands_repr_df[id_cols].copy()

    # Drop rows with any NaN in descriptor columns (important before scaling/transform)
    num_before_dropna = len(target_ligands_repr_df)
    target_ligands_repr_df.dropna(subset=descriptor_columns, how='any', inplace=True)
    if len(target_ligands_repr_df) < num_before_dropna:
        logging.info(f"Dropped {num_before_dropna - len(target_ligands_repr_df)} target ligands with NaN descriptor values before projection.")

    if target_ligands_repr_df.empty:
        logging.warning("Target ligands DataFrame is empty after NaN drop. No projection possible.")
        return pd.DataFrame()

    X_target = target_ligands_repr_df[descriptor_columns].values
    
    try:
        X_target_scaled = scaler_model.transform(X_target)
    except Exception as e:
        logging.error(f"Error scaling target ligands: {e}")
        return pd.DataFrame()
    
    projected_coords = None
    try:
        if CUML_AVAILABLE and hasattr(dr_model, 'transform') and 'umap' in dr_method_key.lower() :
             # cuML UMAP transform might need device context if model was trained on GPU
             # and transform is happening on CPU or vice-versa.
             # Often, if data is numpy, it assumes CPU.
            with using_device_type('cpu'): # To ensure consistency if transform is on CPU
                projected_coords = dr_model.transform(X_target_scaled)
        else: # sklearn PCA/UMAP or cuML PCA (transform is usually fine)
            projected_coords = dr_model.transform(X_target_scaled)

    except Exception as e:
        logging.error(f"Error transforming target ligands with DR model {dr_short_name}: {e}")
        return pd.DataFrame()

    # Create DataFrame for projected ligands
    projection_cols = [f"{dr_short_name}-{i+1}" for i in range(simspace_dim)]
    df_projected_ligands = pd.DataFrame(projected_coords[:, :simspace_dim], columns=projection_cols, index=target_ligands_repr_df.index)
    
    # Merge back with original IDs
    df_projected_ligands = original_ids_df.loc[df_projected_ligands.index].join(df_projected_ligands)
    
    return df_projected_ligands


def calculate_distances(projected_ligands_df, mf_cloud_coords_df, k_for_knn_list, dr_short_name):
    """
    Calculates distances from projected ligands to the MF cloud.
    - projected_ligands_df: DataFrame with columns like 'DR_METHOD-1', 'DR_METHOD-2', ...
    - mf_cloud_coords_df: DataFrame with the same coordinate columns for the MF cloud points.
    - k_for_knn_list: list of k values for k-NN distance.
    """
    if projected_ligands_df.empty or mf_cloud_coords_df.empty:
        logging.warning("Cannot calculate distances: projected ligands or MF cloud is empty.")
        return pd.DataFrame()

    coord_cols = [col for col in projected_ligands_df.columns if col.startswith(dr_short_name + "-")]
    if not coord_cols:
        logging.error(f"No coordinate columns found starting with '{dr_short_name}-' in projected_ligands_df.")
        return pd.DataFrame()
        
    projected_points = projected_ligands_df[coord_cols].values
    mf_cloud_points = mf_cloud_coords_df[coord_cols].values

    # Calculate all pairwise distances
    # (n_projected_points, n_mf_cloud_points)
    pairwise_distances = distance.cdist(projected_points, mf_cloud_points, 'euclidean')

    results = []
    for i in range(len(projected_ligands_df)):
        ligand_distances = pairwise_distances[i, :]
        res = {
            'SMILES': projected_ligands_df.iloc[i].get('SMILES', 'N/A'),
            'Compound ChEMBL ID': projected_ligands_df.iloc[i].get('Compound ChEMBL ID', 'N/A'),
            'min_dist_to_mf_cloud': np.min(ligand_distances) if ligand_distances.size > 0 else np.nan,
        }
        
        for k in k_for_knn_list:
            if len(ligand_distances) >= k:
                sorted_dists = np.sort(ligand_distances)
                res[f'avg_dist_top_{k}_in_mf_cloud'] = np.mean(sorted_dists[:k])
            else:
                res[f'avg_dist_top_{k}_in_mf_cloud'] = np.nan # Not enough points in cloud

        results.append(res)
    
    df_distances = pd.DataFrame(results)

    # Calculate distance to centroid of MF cloud
    if mf_cloud_points.shape[0] > 0:
        mf_cloud_centroid = np.mean(mf_cloud_points, axis=0)
        dist_to_centroid = distance.cdist(projected_points, mf_cloud_centroid.reshape(1, -1), 'euclidean')
        df_distances['dist_to_mf_cloud_centroid'] = dist_to_centroid.flatten()
    else:
        df_distances['dist_to_mf_cloud_centroid'] = np.nan
        
    return df_distances

def plot_results(df_projected_ligands, df_simspace_full, mf_cloud_coords_df, 
                 distances_df, output_dir, target_id_name, dr_short_name, simspace_dim, 
                 representation_type):
    """Generates and saves plots if simspace_dim is 2."""
    if simspace_dim != 2:
        logging.info(f"Skipping scatter/histogram plots as simspace_dim is {simspace_dim} (not 2).")
        return

    coord_cols_2d = [f"{dr_short_name}-1", f"{dr_short_name}-2"]
    if not all(col in df_projected_ligands.columns for col in coord_cols_2d) or \
       not all(col in mf_cloud_coords_df.columns for col in coord_cols_2d):
        logging.warning(f"Cannot generate 2D plots: Coordinate columns {coord_cols_2d} not found in projected data or MF cloud.")
        return

    plt.style.use('seaborn-v0_8-whitegrid')

    # --- Scatter Plot ---
    plt.figure(figsize=(10, 8))
    
    # Plot ZINC sample from the full simspace (if ZINC data was included)
    # Heuristic: Check for ZINC_ID or if 'MOLECULE ID' starts with ZINC
    is_zinc_col = None
    if 'ZINC_ID' in df_simspace_full.columns and df_simspace_full['ZINC_ID'].notna().any():
        is_zinc_col = 'ZINC_ID'
    elif 'MOLECULE ID' in df_simspace_full.columns and df_simspace_full['MOLECULE ID'].str.startswith('ZINC', na=False).any():
         # This is a bit more complex: need to identify ZINC rows correctly based on how 'MOLECULE ID' was formed
         pass # For now, let's assume 'ZINC_ID' is the primary way to identify ZINC for plotting

    if is_zinc_col:
        df_zinc_sample = df_simspace_full[df_simspace_full[is_zinc_col].notna()].sample(n=min(500, len(df_simspace_full[df_simspace_full[is_zinc_col].notna()])), random_state=42, replace=False)
        if all(col in df_zinc_sample.columns for col in coord_cols_2d):
            plt.scatter(df_zinc_sample[coord_cols_2d[0]], df_zinc_sample[coord_cols_2d[1]], 
                        label="ZINC Sample", alpha=0.3, s=20, color='grey', marker='.')
    
    # Plot MF Cloud
    plt.scatter(mf_cloud_coords_df[coord_cols_2d[0]], mf_cloud_coords_df[coord_cols_2d[1]], 
                label="Molecular Function Cloud (ChEMBL)", alpha=0.5, s=30, color='cornflowerblue', marker='o')
    
    # Plot Projected Target Ligands
    plt.scatter(df_projected_ligands[coord_cols_2d[0]], df_projected_ligands[coord_cols_2d[1]], 
                label=f"Projected Target Ligands ({target_id_name})", alpha=0.9, s=70, color='red', marker='x', edgecolor='black')

    plt.title(f"Similarity Space ({representation_type}, {dr_short_name}, Dim={simspace_dim})\nTarget: {target_id_name}")
    plt.xlabel(coord_cols_2d[0])
    plt.ylabel(coord_cols_2d[1])
    plt.legend()
    plt.grid(True)
    scatter_path = os.path.join(output_dir, f"{target_id_name}_{representation_type}_{dr_short_name.replace('-', '_')}_dim{simspace_dim}_scatter.png")
    plt.savefig(scatter_path, dpi=300)
    plt.close()
    logging.info(f"Saved scatter plot to {scatter_path}")

    # --- Histogram of Minimum Distances ---
    if 'min_dist_to_mf_cloud' in distances_df.columns and distances_df['min_dist_to_mf_cloud'].notna().any():
        plt.figure(figsize=(8, 6))
        sns.histplot(distances_df['min_dist_to_mf_cloud'], kde=True, bins=20)
        mean_dist = distances_df['min_dist_to_mf_cloud'].mean()
        median_dist = distances_df['min_dist_to_mf_cloud'].median()
        plt.title(f"Distribution of Min Distances to MF Cloud\n{target_id_name} ({representation_type}, {dr_short_name}, Dim={simspace_dim})")
        plt.xlabel("Minimum Euclidean Distance to MF Cloud")
        plt.ylabel("Frequency")
        plt.axvline(mean_dist, color='r', linestyle='dashed', linewidth=1, label=f'Mean: {mean_dist:.2f}')
        plt.axvline(median_dist, color='g', linestyle='dashed', linewidth=1, label=f'Median: {median_dist:.2f}')
        plt.legend()
        hist_path = os.path.join(output_dir, f"{target_id_name}_{representation_type}_{dr_short_name.replace('-', '_')}_dim{simspace_dim}_min_distances_hist.png")
        plt.savefig(hist_path, dpi=300)
        plt.close()
        logging.info(f"Saved min distance histogram to {hist_path}")


def main():
    parser = argparse.ArgumentParser(description="Project target ligands and analyze distances.")
    parser.add_argument("--target_ligands_repr_path", required=True, help="Path to CSV of target ligands (featurized/fingerprinted).")
    parser.add_argument("--simspace_csv_path", required=True, help="Path to the comprehensive similarity space CSV.")
    parser.add_argument("--model_dir_for_projection", required=True, help="Directory containing scaler and DR models.")
    parser.add_argument("--model_name_root_for_projection", required=True, help="Base name used for saving models (e.g., target_repr_dimX_DR).")
    parser.add_argument("--dr_method_key", required=True, help="Key for DR method from config (e.g., 'pca', 'umap_euclidean').")
    parser.add_argument("--dr_short_name", required=True, help="Short name for DR method (e.g., 'PCA', 'UMAP-Euclidean').")
    parser.add_argument("--simspace_dim", type=int, required=True, help="Dimensionality of the similarity space.")
    parser.add_argument("--k_for_knn", type=str, required=True, help="Comma-separated list of k values for k-NN distance (e.g., '3,5').")
    parser.add_argument("--output_dir", required=True, help="Directory to save analysis results (distances CSV, plots).")
    parser.add_argument("--target_id_name", required=True, help="Target ID name for file naming.")
    
    args = parser.parse_args()

    k_for_knn_list = [int(k.strip()) for k in args.k_for_knn.split(',')]
    
    # Determine representation type from model_name_root or pass explicitly
    # Assuming model_name_root contains 'features' or 'fingerprints'
    representation_type = "features" if "features" in args.model_name_root_for_projection.lower() else "fingerprints"

    # --- Load Models ---
    base_model_filename_prefix = os.path.join(args.model_dir_for_projection, args.model_name_root_for_projection)
    
    scaler_path = f"{base_model_filename_prefix}_scaler.lzma" # From calc_simspace_exp
    
    # Construct DR model path based on dr_method_key
    dr_model_filename_part = ""
    if args.dr_method_key == "pca":
        dr_model_filename_part = "PCA_model.lzma"
    elif "umap" in args.dr_method_key:
        # e.g., umap_euclidean -> euclidean_UMAP_model.lzma
        metric_for_filename = args.dr_method_key.split("_")[1] 
        dr_model_filename_part = f"{metric_for_filename}_UMAP_model.lzma"
    
    if not dr_model_filename_part:
        logging.error(f"Could not determine DR model filename part for key: {args.dr_method_key}")
        return
        
    dr_model_path = os.path.join(args.model_dir_for_projection, f"{args.model_name_root_for_projection}_{dr_model_filename_part}")

    try:
        with open(scaler_path, "rb") as f: scaler_model = decompress_pickle_load(f)
        with open(dr_model_path, "rb") as f: dr_model = decompress_pickle_load(f)
        logging.info(f"Loaded scaler from: {scaler_path}")
        logging.info(f"Loaded DR model ({args.dr_short_name}) from: {dr_model_path}")
    except FileNotFoundError as e:
        logging.error(f"Model file not found: {e}. Cannot proceed with projection.")
        return
    except Exception as e:
        logging.error(f"Error loading models: {e}")
        return

    # --- Load Data ---
    try:
        df_target_ligands_repr = pd.read_csv(args.target_ligands_repr_path, low_memory=False)
        df_simspace_full = pd.read_csv(args.simspace_csv_path, low_memory=False)
    except FileNotFoundError as e:
        logging.error(f"Data file not found: {e}. Cannot proceed.")
        return
    except Exception as e:
        logging.error(f"Error loading data files: {e}")
        return

    # --- Project Target Ligands ---
    logging.info(f"Projecting target ligands for {args.target_id_name} ({representation_type}, {args.dr_short_name}, dim={args.simspace_dim})...")
    df_projected_ligands = project_target_ligands(
        df_target_ligands_repr, scaler_model, dr_model,
        representation_type, args.simspace_dim, args.dr_method_key, args.dr_short_name
    )

    if df_projected_ligands.empty:
        logging.warning("Projection resulted in an empty DataFrame. No analysis will be performed.")
        # Create an empty distances file for consistency
        empty_distances_df = pd.DataFrame(columns=['SMILES', 'Compound ChEMBL ID', 'min_dist_to_mf_cloud'] + 
                                                  [f'avg_dist_top_{k}_in_mf_cloud' for k in k_for_knn_list] + 
                                                  ['dist_to_mf_cloud_centroid'])
        distances_csv_path = os.path.join(args.output_dir, f"{args.target_id_name}_{representation_type}_{args.dr_short_name.replace('-', '_')}_dim{args.simspace_dim}_distances.csv")
        empty_distances_df.to_csv(distances_csv_path, index=False)
        logging.info(f"Saved empty distances file: {distances_csv_path}")
        return

    # --- Prepare MF Cloud Coordinates ---
    # The MF cloud consists of ChEMBL compounds from the simspace that are NOT the target protein itself.
    # 'accession' column should be in df_simspace_full from the ChEMBL data.
    # We need the target_uniprot_id to exclude it. This needs to be fetched from config or passed.
    # For now, assume all non-ZINC, non-CUSTOM in df_simspace_full are part of MF cloud for simplicity,
    # as df_simspace_full was built from 'chembl_mf_excluded' and 'zinc_excluded'.
    # A more precise filter would be `(df_simspace_full['accession'].notna()) & (~df_simspace_full['ZINC_ID'].notna())` etc.
    # For now, let's assume the input df_simspace_full's ChEMBL portion *is* the MF cloud.
    # Need to select the correct coordinate columns for the MF cloud.
    
    mf_cloud_coord_cols = [f"{args.dr_short_name}-{i+1}" for i in range(args.simspace_dim)]
    
    # Identify ChEMBL compounds in the full similarity space to form the MF cloud
    # Heuristic: if 'Compound ChEMBL ID' is present and notna, it's likely ChEMBL
    # Or if 'accession' is present (as ZINC/Custom won't have target accessions from ChEMBL)
    if 'accession' in df_simspace_full.columns and df_simspace_full['accession'].notna().any():
         mf_cloud_df_full = df_simspace_full[df_simspace_full['accession'].notna()].copy()
    elif 'Compound ChEMBL ID' in df_simspace_full.columns and df_simspace_full['Compound ChEMBL ID'].notna().any():
         mf_cloud_df_full = df_simspace_full[df_simspace_full['Compound ChEMBL ID'].notna()].copy()
    else:
         logging.warning("Cannot reliably identify ChEMBL MF cloud points in the similarity space. Distance calculations might be inaccurate.")
         mf_cloud_df_full = pd.DataFrame() # Empty

    if mf_cloud_df_full.empty or not all(col in mf_cloud_df_full.columns for col in mf_cloud_coord_cols):
        logging.warning(f"MF Cloud data is empty or missing coordinate columns ({mf_cloud_coord_cols}). Distance calculations might fail or be empty.")
        mf_cloud_coords_for_dist_calc = pd.DataFrame(columns=mf_cloud_coord_cols) # Empty df with correct cols
    else:
        mf_cloud_coords_for_dist_calc = mf_cloud_df_full[mf_cloud_coord_cols].dropna()


    # --- Calculate Distances ---
    logging.info("Calculating distances to MF cloud...")
    distances_df = calculate_distances(df_projected_ligands, mf_cloud_coords_for_dist_calc, k_for_knn_list, args.dr_short_name)
    
    distances_csv_path = os.path.join(args.output_dir, f"{args.target_id_name}_{representation_type}_{args.dr_short_name.replace('-', '_')}_dim{args.simspace_dim}_distances.csv")
    distances_df.to_csv(distances_csv_path, index=False)
    logging.info(f"Saved distances analysis to: {distances_csv_path}")

    # --- Generate Plots (if 2D) ---
    if args.simspace_dim == 2:
        logging.info("Generating 2D plots...")
        plot_results(df_projected_ligands, df_simspace_full, mf_cloud_coords_for_dist_calc, 
                     distances_df, args.output_dir, args.target_id_name, 
                     args.dr_short_name, args.simspace_dim, representation_type)

    logging.info(f"Projection and analysis for {args.target_id_name} ({representation_type}, {args.dr_short_name}, dim={args.simspace_dim}) completed.")


if __name__ == "__main__":
    main()