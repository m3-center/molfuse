import pandas as pd
import numpy as np
import os
import argparse
import logging
import json
from compress_pickle import load as decompress_pickle_load
from scipy.spatial import distance
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc as sklearn_auc # For enrichment

# CUML imports (only for using_device_type if UMAP model is cuML and transform is separate)
try:
    from cuml.common.device_selection import using_device_type
    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)-8s - %(filename)-25s - %(funcName)-25s - %(message)s',
                    handlers=[logging.FileHandler("project_analyze.log"), logging.StreamHandler()])

FINGERPRINT_COLUMN_PREFIX = "fp_"
NUM_FINGERPRINT_BITS = 2048

# (get_descriptor_columns_for_proj and project_target_ligands_with_models remain largely the same as the last full version)
# Minor adjustments might be needed in project_target_ligands_with_models to ensure it robustly returns
# a DataFrame with consistent ID columns ('SMILES', 'Compound ChEMBL ID') and the projection columns.

def get_descriptor_columns_for_proj(df, representation_type, target_rdkit_features_list):
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
    if target_ligands_repr_df.empty:
        logging.warning("Target ligands DataFrame for projection is empty.")
        return pd.DataFrame()

    descriptor_columns = get_descriptor_columns_for_proj(target_ligands_repr_df, representation_type, target_rdkit_features_list)
    if not descriptor_columns:
        return pd.DataFrame()

    id_cols_to_preserve = [col for col in ['SMILES', 'Compound ChEMBL ID', 'Standard Value (nM)', 'Activity Type', 'accession'] if col in target_ligands_repr_df.columns]
    original_ids_and_activity_df = target_ligands_repr_df[id_cols_to_preserve].copy()

    target_ligands_repr_df[descriptor_columns] = target_ligands_repr_df[descriptor_columns].apply(pd.to_numeric, errors='coerce')
    valid_rows_mask = target_ligands_repr_df[descriptor_columns].notna().all(axis=1)
    target_ligands_valid_df = target_ligands_repr_df[valid_rows_mask].copy()

    if len(target_ligands_valid_df) < len(target_ligands_repr_df):
        logging.info(f"Dropped {len(target_ligands_repr_df) - len(target_ligands_valid_df)} target ligands with NaN descriptor values before projection.")

    if target_ligands_valid_df.empty:
        logging.warning("Target ligands DataFrame is empty after NaN drop. No projection possible with models.")
        return pd.DataFrame()

    X_target = target_ligands_valid_df[descriptor_columns].values
    try:
        X_target_scaled = scaler_model.transform(X_target)
    except Exception as e:
        logging.error(f"Error scaling target ligands for {dr_short_name}: {e}"); return pd.DataFrame()
    
    projected_coords = None
    try:
        if CUML_AVAILABLE and 'umap' in dr_method_key.lower() and hasattr(dr_model, 'transform'):
            with using_device_type('cpu'): projected_coords = dr_model.transform(X_target_scaled)
        else: projected_coords = dr_model.transform(X_target_scaled)
    except Exception as e:
        logging.error(f"Error transforming target ligands with DR model {dr_short_name}: {e}"); return pd.DataFrame()

    projection_cols = [f"{dr_short_name}-{i+1}" for i in range(simspace_dim)]
    df_projected_coords = pd.DataFrame(projected_coords[:, :simspace_dim], columns=projection_cols, index=target_ligands_valid_df.index)
    
    df_projected_ligands_with_ids = original_ids_and_activity_df.loc[df_projected_coords.index].join(df_projected_coords)
    return df_projected_ligands_with_ids.reset_index(drop=True)


def calculate_compound_scores(compounds_projected_df, mf_cloud_coords_df, dr_short_name, simspace_dim):
    """
    Calculates proximity scores for a set of projected compounds to the MF cloud.
    Score is -min_dist_to_mf_cloud. Also returns min_dist for reference.
    """
    scores_df = pd.DataFrame(index=compounds_projected_df.index)
    id_cols = [col for col in ['SMILES', 'Compound ChEMBL ID', 'MOLECULE ID', 'ZINC_ID'] if col in compounds_projected_df.columns] # Preserve any available ID
    for id_col in id_cols:
        if id_col in compounds_projected_df:
             scores_df[id_col] = compounds_projected_df[id_col]


    if compounds_projected_df.empty or mf_cloud_coords_df.empty:
        logging.warning("Cannot calculate scores: projected compounds or MF cloud is empty.")
        scores_df['min_dist_to_mf_cloud'] = np.nan
        scores_df['score'] = np.nan
        return scores_df

    coord_cols = [f"{dr_short_name}-{i+1}" for i in range(simspace_dim)]
    if not all(col in compounds_projected_df.columns for col in coord_cols) or \
       not all(col in mf_cloud_coords_df.columns for col in coord_cols):
        logging.error(f"Coordinate columns mismatch for score calculation ({dr_short_name}).")
        scores_df['min_dist_to_mf_cloud'] = np.nan
        scores_df['score'] = np.nan
        return scores_df

    projected_points = compounds_projected_df[coord_cols].values
    mf_cloud_points = mf_cloud_coords_df[coord_cols].values

    if mf_cloud_points.shape[0] > 0 and projected_points.shape[0] > 0:
        pairwise_distances = distance.cdist(projected_points, mf_cloud_points, 'euclidean')
        min_distances = np.min(pairwise_distances, axis=1)
        scores_df['min_dist_to_mf_cloud'] = min_distances
        scores_df['score'] = -min_distances  # Higher score for smaller distance
    else:
        scores_df['min_dist_to_mf_cloud'] = np.nan
        scores_df['score'] = np.nan
        
    return scores_df

def calculate_enrichment_metrics(ranked_df, active_label_col='is_active'):
    """Calculates ROC-AUC, PR-AUC, and EFs."""
    metrics = {}
    y_true = ranked_df[active_label_col].values
    y_scores = ranked_df['score'].values # Assumes higher score is better

    if len(np.unique(y_true)) < 2: # Needs both actives and inactives for AUC/EF
        logging.warning("Not enough class diversity to calculate ROC/PR AUC or EFs.")
        metrics['roc_auc'] = np.nan
        metrics['pr_auc'] = np.nan
        for x_percent in [0.01, 0.05, 0.10]: # 1%, 5%, 10%
            metrics[f'ef_{int(x_percent*100)}%'] = np.nan
        return metrics

    metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    metrics['pr_auc'] = sklearn_auc(recall, precision)

    total_actives = np.sum(y_true)
    total_compounds = len(y_true)
    
    # Sort by score descending for EF calculation
    df_sorted_for_ef = ranked_df.sort_values(by='score', ascending=False)

    for x_percent in [0.01, 0.05, 0.10]: # 1%, 5%, 10%
        num_top_x = int(np.ceil(x_percent * total_compounds))
        if num_top_x == 0: # Handle very small datasets
            metrics[f'ef_{int(x_percent*100)}%'] = np.nan if total_actives > 0 else 1.0 # Or 0.0?
            continue

        actives_in_top_x = np.sum(df_sorted_for_ef[active_label_col].iloc[:num_top_x])
        
        ef_denominator = (total_actives / total_compounds) * num_top_x
        if ef_denominator > 0 :
            metrics[f'ef_{int(x_percent*100)}%'] = actives_in_top_x / ef_denominator
        else: # No actives overall or num_top_x is 0 due to small N
            metrics[f'ef_{int(x_percent*100)}%'] = 0.0 if total_actives > 0 else 1.0 # If no actives, EF is undefined or 1

    return metrics

def calculate_affinity_correlation(ranked_actives_df, activity_col='Standard Value (nM)'):
    """Calculates Spearman correlation if activity data is present and numeric."""
    if activity_col not in ranked_actives_df.columns or ranked_actives_df[activity_col].isna().all():
        return np.nan
    
    # Convert activity to numeric; smaller nM is better, so for correlation with score (higher=better),
    # we might need to transform activity (e.g., pActivity = -log10(Activity_Molar))
    # Or, correlate -score with nM_value.
    # For simplicity, let's correlate score with -log10(nM_value * 1e-9) if nM
    
    # Ensure activity is numeric and positive for log
    numeric_activity = pd.to_numeric(ranked_actives_df[activity_col], errors='coerce')
    valid_activity_mask = numeric_activity.notna() & (numeric_activity > 0)
    
    if valid_activity_mask.sum() < 2: # Need at least 2 points for correlation
        return np.nan
        
    # Calculate pActivity (e.g., pIC50 from IC50 in nM)
    p_activity = -np.log10(numeric_activity[valid_activity_mask] * 1e-9) # Convert nM to M for pActivity
    scores_for_corr = ranked_actives_df.loc[valid_activity_mask, 'score']

    if len(p_activity) < 2 or len(scores_for_corr) < 2 or p_activity.nunique() < 2 or scores_for_corr.nunique() < 2:
        return np.nan # Not enough variance or data points

    correlation, _ = pd.Series(p_activity).corr(pd.Series(scores_for_corr), method='spearman'), # Get only rho
    return correlation

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


# (plot_projection_results function as in previous full version)
def plot_projection_results(df_projected_target_ligands, df_simspace_main_data, mf_cloud_coords_for_plot, 
                            distances_df_with_ids, output_dir, target_id_name, dr_short_name, simspace_dim, 
                            representation_type):
    # ... (Full plotting logic from before) ...
    # This function now receives distances_df_with_ids which contains min_dist for histogram
    if simspace_dim != 2:
        logging.debug(f"Skipping 2D plots as simspace_dim is {simspace_dim}.")
        return

    coord_cols_2d = [f"{dr_short_name}-1", f"{dr_short_name}-2"]
    
    if not all(col in df_projected_target_ligands.columns for col in coord_cols_2d):
        logging.warning(f"Projected target ligands missing 2D coordinate columns for plotting: {coord_cols_2d}")
        return
    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 10))
    
    df_zinc_sample = pd.DataFrame()
    if 'MOLECULE ID' in df_simspace_main_data.columns:
        zinc_rows = df_simspace_main_data[df_simspace_main_data['MOLECULE ID'].astype(str).str.startswith('ZINC', na=False)]
        if not zinc_rows.empty and len(zinc_rows) >=1 : # Ensure there's at least one ZINC row to sample
             df_zinc_sample = zinc_rows.sample(n=min(1000, len(zinc_rows)), random_state=42, replace=False)
    elif 'ZINC_ID' in df_simspace_main_data.columns:
        zinc_rows = df_simspace_main_data[df_simspace_main_data['ZINC_ID'].notna()]
        if not zinc_rows.empty and len(zinc_rows) >=1 :
            df_zinc_sample = zinc_rows.sample(n=min(1000, len(zinc_rows)), random_state=42, replace=False)

    if not df_zinc_sample.empty and all(col in df_zinc_sample.columns for col in coord_cols_2d):
        plt.scatter(df_zinc_sample[coord_cols_2d[0]], df_zinc_sample[coord_cols_2d[1]], 
                    label="ZINC Sample (Decoys)", alpha=0.2, s=15, color='darkgrey', marker='.')
    
    if not mf_cloud_coords_for_plot.empty and all(col in mf_cloud_coords_for_plot.columns for col in coord_cols_2d):
        plt.scatter(mf_cloud_coords_for_plot[coord_cols_2d[0]], mf_cloud_coords_for_plot[coord_cols_2d[1]], 
                    label="Molecular Function Cloud (ChEMBL)", alpha=0.4, s=25, color='dodgerblue', marker='o')
    
    plt.scatter(df_projected_target_ligands[coord_cols_2d[0]], df_projected_target_ligands[coord_cols_2d[1]], 
                label=f"Projected Target Actives ({target_id_name})", alpha=0.9, s=60, color='red', marker='P', edgecolor='black') # Changed marker

    plt.title(f"Similarity Space ({representation_type}, {dr_short_name}, Dim={simspace_dim})\nTarget: {target_id_name}", fontsize=14)
    plt.xlabel(coord_cols_2d[0], fontsize=12)
    plt.ylabel(coord_cols_2d[1], fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    scatter_path = os.path.join(output_dir, f"{target_id_name}_{representation_type}_{dr_short_name.replace('-', '_')}_dim{simspace_dim}_scatter.png")
    try:
        plt.savefig(scatter_path, dpi=200, bbox_inches='tight')
        logging.info(f"Saved scatter plot to {scatter_path}")
    except Exception as e: logging.error(f"Failed to save scatter plot {scatter_path}: {e}")
    plt.close()

    # Histogram of Minimum Distances FOR ACTIVES ONLY
    # The distances_df_with_ids now refers to the combined df for ranking if we follow that logic strictly.
    # For this histogram, we usually want to see the distribution for the *true actives*.
    # Let's assume distances_df_with_ids is passed *after* being filtered for actives OR it contains an 'is_active' column
    if 'min_dist_to_mf_cloud' in df_projected_target_ligands.columns and df_projected_target_ligands['min_dist_to_mf_cloud'].notna().any(): # Use the df_projected_target_ligands which also has dists
        plt.figure(figsize=(9, 7))
        sns.histplot(df_projected_target_ligands['min_dist_to_mf_cloud'].dropna(), kde=True, bins=20, color='skyblue', edgecolor='black')
        mean_dist = df_projected_target_ligands['min_dist_to_mf_cloud'].mean()
        median_dist = df_projected_target_ligands['min_dist_to_mf_cloud'].median()
        plt.title(f"Min. Distances of Target ACTIVES to MF Cloud\n{target_id_name} ({representation_type}, {dr_short_name}, Dim={simspace_dim})", fontsize=14)
        # ... (rest of histogram plotting as before) ...
        if pd.notna(mean_dist): plt.axvline(mean_dist, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {mean_dist:.3f}')
        if pd.notna(median_dist): plt.axvline(median_dist, color='green', linestyle='dashed', linewidth=1.5, label=f'Median: {median_dist:.3f}')
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.5)
        hist_path = os.path.join(output_dir, f"{target_id_name}_{representation_type}_{dr_short_name.replace('-', '_')}_dim{simspace_dim}_min_distances_hist_ACTIVES.png")
        try:
            plt.savefig(hist_path, dpi=200, bbox_inches='tight')
            logging.info(f"Saved min distance histogram for ACTIVES to {hist_path}")
        except Exception as e: logging.error(f"Failed to save histogram {hist_path}: {e}")
        plt.close()


def main():
    # ... (argparse as in the previous full version - ensure it has all necessary args) ...
    parser = argparse.ArgumentParser(description="Project target ligands and analyze distances.")
    parser.add_argument("--target_ligands_repr_path", default=None)
    parser.add_argument("--precomputed_target_projections_path", default=None)
    parser.add_argument("--simspace_csv_path", required=True)
    parser.add_argument("--model_dir_for_projection", default=None)
    parser.add_argument("--model_name_root_for_projection", default=None)
    parser.add_argument("--dr_method_key", required=True)
    parser.add_argument("--dr_short_name", required=True)
    parser.add_argument("--simspace_dim", type=int, required=True)
    parser.add_argument("--k_for_knn", type=str, required=True) # No longer used for primary ranking
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_id_name", required=True)
    parser.add_argument("--representation_type", required=True, choices=["features", "fingerprints"])
    parser.add_argument("--rdkit_features_list_target_str", required=True)
    args = parser.parse_args()

    try:
        target_rdkit_features_list = json.loads(args.rdkit_features_list_target_str)
    except Exception as e:
        logging.error(f"Error parsing --rdkit_features_list_target_str: {e}. Aborting."); return

    df_projected_target_actives = pd.DataFrame() # Renamed for clarity

    # --- 1. Get Projected Coordinates for Target Actives ---
    if args.precomputed_target_projections_path and os.path.exists(args.precomputed_target_projections_path):
        logging.info(f"Loading pre-computed target active projections for {args.dr_short_name} from: {args.precomputed_target_projections_path}")
        try:
            df_projected_target_actives = pd.read_csv(args.precomputed_target_projections_path)
            # (Ensure coord column names are correct as in previous version)
        except Exception as e: logging.error(f"Error loading pre-computed target projections: {e}")
    
    if df_projected_target_actives.empty and args.dr_method_key != "tsne":
        # (Model loading and projection logic for PCA/UMAP as in previous version)
        # ...
        # This section populates df_projected_target_actives using models
        # ...
        logging.info(f"Projecting target actives for {args.target_id_name} ({args.representation_type}, {args.dr_short_name}, dim={args.simspace_dim}) using models...")
        try:
            # ... (model loading from previous version) ...
            scaler_path = os.path.join(args.model_dir_for_projection, f"{args.model_name_root_for_projection}_scaler.lzma")
            # ... (construct dr_model_path) ...
            dr_model_filename_part = ""
            if args.dr_method_key == "pca": dr_model_filename_part = "PCA_model.lzma"
            elif "umap" in args.dr_method_key:
                metric_for_filename = args.dr_method_key.split("_")[1]
                dr_model_filename_part = f"{metric_for_filename}_UMAP_model.lzma"
            
            if not dr_model_filename_part: logging.error(f"Could not determine DR model filename for key: {args.dr_method_key}"); return
            dr_model_path = os.path.join(args.model_dir_for_projection, f"{args.model_name_root_for_projection}_{dr_model_filename_part}")

            with open(scaler_path, "rb") as f: scaler_model = decompress_pickle_load(f)
            with open(dr_model_path, "rb") as f: dr_model = decompress_pickle_load(f)
            
            df_target_ligands_repr_raw = pd.read_csv(args.target_ligands_repr_path, low_memory=False) # raw features/fp for targets
            df_projected_target_actives = project_target_ligands_with_models(
                df_target_ligands_repr_raw, scaler_model, dr_model,
                args.representation_type, args.simspace_dim, args.dr_method_key, args.dr_short_name,
                target_rdkit_features_list
            )
        except Exception as e: logging.error(f"Error during model loading or projection for {args.dr_short_name}: {e}")


    if df_projected_target_actives.empty:
        logging.warning(f"No target actives projected for {args.dr_short_name}. Cannot perform ranking analysis.")
        # Save an empty metrics file for consistency
        pd.DataFrame().to_csv(os.path.join(args.output_dir, f"{args.target_id_name}_{args.representation_type}_{args.dr_short_name.replace('-', '_')}_dim{args.simspace_dim}_ranking_metrics.csv"), index=False)
        return

    logging.info(f"Successfully projected/loaded {len(df_projected_target_actives)} target actives for {args.dr_short_name}.")

    # --- 2. Load Main Simspace & Identify MF Cloud and ZINC Decoys ---
    try:
        df_simspace_main_data = pd.read_csv(args.simspace_csv_path, low_memory=False)
    except Exception as e:
        logging.error(f"Error loading main similarity space file {args.simspace_csv_path}: {e}")
        df_simspace_main_data = pd.DataFrame()

    mf_cloud_coords = pd.DataFrame()
    df_zinc_decoys_projected = pd.DataFrame()
    coord_cols_for_analysis = [f"{args.dr_short_name}-{i+1}" for i in range(args.simspace_dim)]

    if not df_simspace_main_data.empty and all(c in df_simspace_main_data.columns for c in coord_cols_for_analysis) and \
       'MOLECULE ID' in df_simspace_main_data.columns: # Crucial check
        
        # Identify ChEMBL MF Cloud (non-ZINC entries)
        chembl_mask = df_simspace_main_data['MOLECULE ID'].astype(str).str.startswith('CHEMBL', na=False) # Assuming ChEMBL IDs
        if 'ZINC_ID' in df_simspace_main_data.columns: # More robust: not ZINC
             chembl_mask = chembl_mask & df_simspace_main_data['ZINC_ID'].isna()
        elif 'MOLECULE ID' in df_simspace_main_data.columns: # Alternative if ZINC_ID not present but MOLECULE ID is
             chembl_mask = chembl_mask & ~df_simspace_main_data['MOLECULE ID'].astype(str).str.startswith('ZINC', na=False)
        
        mf_cloud_df_full = df_simspace_main_data[chembl_mask].copy()
        if not mf_cloud_df_full.empty:
            mf_cloud_coords = mf_cloud_df_full[coord_cols_for_analysis].dropna()
        
        # Identify ZINC Decoys
        zinc_mask = df_simspace_main_data['MOLECULE ID'].astype(str).str.startswith('ZINC', na=False)
        if 'ZINC_ID' in df_simspace_main_data.columns: # More robust
            zinc_mask = df_simspace_main_data['ZINC_ID'].notna()
        
        df_zinc_decoys_projected_all_info = df_simspace_main_data[zinc_mask].copy()
        if not df_zinc_decoys_projected_all_info.empty:
            # Keep IDs and projected coordinates for ZINC decoys
            id_cols_zinc = [col for col in ['SMILES', 'MOLECULE ID', 'ZINC_ID'] if col in df_zinc_decoys_projected_all_info.columns]
            df_zinc_decoys_projected = df_zinc_decoys_projected_all_info[id_cols_zinc + coord_cols_for_analysis].dropna(subset=coord_cols_for_analysis)
    else:
        logging.warning(f"Main similarity space CSV is empty or missing coordinate columns for {args.dr_short_name}. Ranking analysis will be limited.")


    # --- 3. Calculate Scores for Actives and Decoys ---
    logging.info(f"Calculating scores for target actives and ZINC decoys relative to MF cloud ({args.dr_short_name})...")
    df_target_actives_scores = calculate_compound_scores(df_projected_target_actives, mf_cloud_coords, args.dr_short_name, args.simspace_dim)
    df_zinc_decoys_scores = calculate_compound_scores(df_zinc_decoys_projected, mf_cloud_coords, args.dr_short_name, args.simspace_dim)

    df_target_actives_scores['is_active'] = 1
    df_zinc_decoys_scores['is_active'] = 0

    # Combine for ranking
    # Ensure consistent columns before concat, especially ID columns. Pick one primary ID if multiple exist.
    # For simplicity, let's assume 'SMILES' is present and sufficient for joining later if needed.
    # The calculate_compound_scores function should try to preserve available IDs.
    
    # Align columns before concat, prioritizing those in actives_scores
    common_cols = list(set(df_target_actives_scores.columns) & set(df_zinc_decoys_scores.columns))
    df_for_ranking = pd.concat([
        df_target_actives_scores[common_cols], 
        df_zinc_decoys_scores[common_cols]
    ], ignore_index=True).dropna(subset=['score']) # Drop rows where score couldn't be calculated (e.g. empty MF cloud)


    # --- 4. Calculate Enrichment Metrics ---
    ranking_metrics = {}
    if not df_for_ranking.empty and df_for_ranking['is_active'].nunique() > 1: # Need both actives and decoys
        logging.info(f"Calculating enrichment metrics (Total ranked: {len(df_for_ranking)}, Actives: {df_target_actives_scores['is_active'].sum()})...")
        ranking_metrics = calculate_enrichment_metrics(df_for_ranking, active_label_col='is_active')
    else:
        logging.warning("Not enough data or class diversity in df_for_ranking to calculate enrichment metrics.")
        ranking_metrics = {'roc_auc': np.nan, 'pr_auc': np.nan, 'ef_1%': np.nan, 'ef_5%': np.nan, 'ef_10%': np.nan}

    # --- 5. Calculate Affinity Correlation (for actives only) ---
    # The df_projected_target_actives should have 'Standard Value (nM)' if it was in the input
    # and preserved by project_target_ligands_with_models
    # We need to merge scores back to it, or pass the activity column through df_target_actives_scores
    
    # Ensure df_target_actives_scores has the activity column if df_projected_target_actives did
    if 'Standard Value (nM)' in df_projected_target_actives.columns and 'SMILES' in df_projected_target_actives.columns: # Assuming SMILES as a key
        # Merge score into the original projected actives df that has activity data
        df_actives_for_corr = pd.merge(df_projected_target_actives, 
                                       df_target_actives_scores[['SMILES', 'score']], # Use SMILES to merge score
                                       on='SMILES', how='left')
        ranking_metrics['spearman_rho_affinity_vs_score'] = calculate_affinity_correlation(df_actives_for_corr, activity_col='Standard Value (nM)')
    else:
        ranking_metrics['spearman_rho_affinity_vs_score'] = np.nan
        logging.debug("Skipping affinity correlation: 'Standard Value (nM)' or 'SMILES' not found in projected target actives for merging scores.")


    # --- Save Ranking Metrics ---
    df_ranking_metrics = pd.DataFrame([ranking_metrics]) # Convert dict to DataFrame for saving
    metrics_csv_path = os.path.join(args.output_dir, f"{args.target_id_name}_{args.representation_type}_{args.dr_short_name.replace('-', '_')}_dim{args.simspace_dim}_ranking_metrics.csv")
    try:
        df_ranking_metrics.to_csv(metrics_csv_path, index=False)
        logging.info(f"Saved ranking metrics to: {metrics_csv_path}")
    except Exception as e:
        logging.error(f"Failed to save ranking metrics CSV {metrics_csv_path}: {e}")
        
    # Save individual distances for actives (min_dist, knn_dist, centroid_dist) for supplementary info
    # Re-calculate these for actives only, as calculate_compound_scores only returns min_dist and score
    # This part requires the original k_for_knn list and more detailed distance calculation for actives
    k_for_knn_list_int = [int(k.strip()) for k in args.k_for_knn.split(',')]
    if not df_projected_target_actives.empty: # Ensure we have projected actives
        df_detailed_active_distances = calculate_distances_to_cloud( # The old function
            df_projected_target_actives, # This should have the coordinate columns
            mf_cloud_coords,
            k_for_knn_list_int,
            args.dr_short_name,
            args.simspace_dim
        )
        detailed_distances_path = os.path.join(args.output_dir, f"{args.target_id_name}_{args.representation_type}_{args.dr_short_name.replace('-', '_')}_dim{args.simspace_dim}_detailed_active_distances.csv")
        try:
            df_detailed_active_distances.to_csv(detailed_distances_path, index=False)
            logging.info(f"Saved detailed distances for ACTIVES to: {detailed_distances_path}")
        except Exception as e:
            logging.error(f"Failed to save detailed active distances CSV: {e}")


    # --- Generate Plots (if 2D) ---
    # Pass df_projected_target_actives (which now needs min_dist if hist relies on it)
    # or pass df_target_actives_scores for the histogram.
    # Let's make sure df_projected_target_actives gets the 'min_dist_to_mf_cloud' for plotting
    if not df_projected_target_actives.empty and not df_target_actives_scores.empty and 'SMILES' in df_projected_target_actives and 'SMILES' in df_target_actives_scores:
         df_plot_actives = pd.merge(df_projected_target_actives, df_target_actives_scores[['SMILES', 'min_dist_to_mf_cloud']], on='SMILES', how='left')
    else:
         df_plot_actives = df_projected_target_actives # Fallback if merge fails

    if args.simspace_dim == 2 and not df_plot_actives.empty:
        logging.info("Generating 2D plots...")
        plot_projection_results(df_plot_actives, df_simspace_main_data, 
                                mf_cloud_coords, # Pass the coordinate-only df for plotting MF cloud
                                df_plot_actives, # Pass this again for histogram of its min_dist
                                args.output_dir, args.target_id_name, 
                                args.dr_short_name, args.simspace_dim, args.representation_type)

    logging.info(f"Projection and analysis for {args.target_id_name} ({args.representation_type}, {args.dr_short_name}, dim={args.simspace_dim}) completed.")

if __name__ == "__main__":
    main()