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
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc as sklearn_auc

# CUML imports
try:
    from cuml.common.device_selection import using_device_type
    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)-8s - %(filename)-25s - %(funcName)-25s - %(message)s',
                    handlers=[logging.FileHandler("project_analyze.log", mode='a'), logging.StreamHandler()])

FINGERPRINT_COLUMN_PREFIX = "fp_"
NUM_FINGERPRINT_BITS = 2048
CDIST_BATCH_SIZE = 5000 # Configurable batch size for cdist to manage memory

# (get_descriptor_columns_for_proj, project_target_ligands_with_models - same as last complete version)
def get_descriptor_columns_for_proj(df, representation_type, target_rdkit_features_list):
    if representation_type == "features":
        descriptor_cols = [col for col in target_rdkit_features_list if col in df.columns]
        if not descriptor_cols: logging.error("No target RDKit features found in target ligands data for projection."); return None
        missing_in_df = [col for col in target_rdkit_features_list if col not in df.columns]
        if missing_in_df: logging.warning(f"Target ligand data missing configured features: {missing_in_df}")
    elif representation_type == "fingerprints":
        descriptor_cols = [col for col in df.columns if col.startswith(FINGERPRINT_COLUMN_PREFIX)]
        if not descriptor_cols: logging.error("No fingerprint columns found in target ligands data for projection."); return None
    else:
        logging.error(f"Invalid representation_type for projection: {representation_type}"); return None
    return descriptor_cols

def project_target_ligands_with_models(target_ligands_repr_df_input, scaler_model, dr_model,
                                       representation_type, simspace_dim, dr_method_key, dr_short_name,
                                       target_rdkit_features_list):
    if target_ligands_repr_df_input.empty: logging.warning("Target ligands DataFrame for projection is empty."); return pd.DataFrame()
    target_ligands_repr_df = target_ligands_repr_df_input.copy()
    descriptor_columns = get_descriptor_columns_for_proj(target_ligands_repr_df, representation_type, target_rdkit_features_list)
    if not descriptor_columns: return pd.DataFrame()
    original_cols_df = target_ligands_repr_df.copy()
    target_ligands_repr_df[descriptor_columns] = target_ligands_repr_df[descriptor_columns].apply(pd.to_numeric, errors='coerce')
    valid_rows_mask = target_ligands_repr_df[descriptor_columns].notna().all(axis=1)
    target_ligands_valid_for_projection_df = target_ligands_repr_df[valid_rows_mask]
    if len(target_ligands_valid_for_projection_df) < len(target_ligands_repr_df):
        logging.info(f"Dropped {len(target_ligands_repr_df) - len(target_ligands_valid_for_projection_df)} target ligands with NaN descriptor values before projection.")
    if target_ligands_valid_for_projection_df.empty: logging.warning("Target ligands DataFrame is empty after NaN drop."); return pd.DataFrame()
    X_target = target_ligands_valid_for_projection_df[descriptor_columns].values
    try: X_target_scaled = scaler_model.transform(X_target)
    except Exception as e: logging.error(f"Error scaling target ligands for {dr_short_name}: {e}"); return pd.DataFrame()
    projected_coords_values = None
    try:
        if CUML_AVAILABLE and 'umap' in dr_method_key.lower() and hasattr(dr_model, 'transform'):
            with using_device_type('gpu'): projected_coords_values = dr_model.transform(X_target_scaled)
        else: projected_coords_values = dr_model.transform(X_target_scaled)
    except Exception as e: logging.error(f"Error transforming target ligands with DR model {dr_short_name}: {e}"); return pd.DataFrame()
    projection_cols_names = [f"{dr_short_name}-{i+1}" for i in range(simspace_dim)]
    df_projected_coords_only = pd.DataFrame(projected_coords_values[:, :simspace_dim], columns=projection_cols_names, index=target_ligands_valid_for_projection_df.index)
    df_projected_ligands_full_info = original_cols_df.loc[df_projected_coords_only.index].join(df_projected_coords_only)
    return df_projected_ligands_full_info.reset_index(drop=True)

def calculate_compound_scores(compounds_df_with_coords, mf_cloud_coords_df, dr_short_name, simspace_dim):
    output_df = compounds_df_with_coords.copy()
    output_df['min_dist_to_mf_cloud'] = np.nan
    output_df['score'] = np.nan

    if compounds_df_with_coords.empty or mf_cloud_coords_df.empty:
        logging.debug("Cannot calculate scores: projected compounds or MF cloud is empty.")
        return output_df

    coord_cols = [f"{dr_short_name}-{i+1}" for i in range(simspace_dim)]
    if not all(col in compounds_df_with_coords.columns for col in coord_cols):
        logging.error(f"Input compounds missing coordinate columns for score calculation: {coord_cols}. Available: {compounds_df_with_coords.columns.tolist()}")
        return output_df
    if not all(col in mf_cloud_coords_df.columns for col in coord_cols):
        logging.error(f"MF cloud missing coordinate columns for score calculation: {coord_cols}. Available: {mf_cloud_coords_df.columns.tolist()}")
        return output_df

    try:
        projected_points_all = compounds_df_with_coords[coord_cols].astype(float).values
        mf_cloud_points = mf_cloud_coords_df[coord_cols].astype(float).values
    except ValueError as ve:
        logging.error(f"Could not convert coordinate columns to float for distance calculation: {ve}")
        return output_df

    if mf_cloud_points.shape[0] > 0 and projected_points_all.shape[0] > 0:
        all_min_distances = []
        num_projected_points = projected_points_all.shape[0]

        for i in range(0, num_projected_points, CDIST_BATCH_SIZE):
            batch_projected_points = projected_points_all[i:i+CDIST_BATCH_SIZE]
            logging.debug(f"Processing cdist batch {i//CDIST_BATCH_SIZE + 1}/{(num_projected_points -1)//CDIST_BATCH_SIZE + 1}, size: {batch_projected_points.shape[0]}")
            try:
                pairwise_distances_batch = distance.cdist(batch_projected_points, mf_cloud_points, 'euclidean')
                min_distances_batch = np.min(pairwise_distances_batch, axis=1)
                all_min_distances.append(min_distances_batch)
            except MemoryError as me:
                logging.error(f"MemoryError during cdist batch processing (batch size {CDIST_BATCH_SIZE}): {me}. Try reducing CDIST_BATCH_SIZE.")
                # Fill remaining with NaNs or re-raise, for now, just log and continue, NaNs will propagate
                # Calculate how many NaNs are needed for the rest of this batch and subsequent points
                num_failed_in_batch = batch_projected_points.shape[0]
                all_min_distances.append(np.full(num_failed_in_batch, np.nan))
                # If we want to stop entirely on memory error for a batch:
                # raise me
            except Exception as e:
                logging.error(f"Exception during cdist batch processing: {e}")
                num_failed_in_batch = batch_projected_points.shape[0]
                all_min_distances.append(np.full(num_failed_in_batch, np.nan))


        if all_min_distances:
            min_distances_combined = np.concatenate(all_min_distances)
            # Ensure the length matches the original dataframe to assign back correctly
            if len(min_distances_combined) == len(output_df):
                output_df['min_dist_to_mf_cloud'] = min_distances_combined
                output_df['score'] = -min_distances_combined # Score is negative distance
            else:
                logging.error(f"Length mismatch after cdist batching: expected {len(output_df)}, got {len(min_distances_combined)}. Scores will be NaN.")
    else:
        logging.debug("No points in MF cloud or projected points after type conversion/filtering.")
        # Already initialized to NaN, so no action needed

    return output_df


# (calculate_detailed_distances_for_actives - same as last complete version)
def calculate_detailed_distances_for_actives(projected_actives_df, mf_cloud_coords_df, k_for_knn_list, dr_short_name, simspace_dim):
    output_df = projected_actives_df[[col for col in ['SMILES', 'Compound ChEMBL ID'] if col in projected_actives_df.columns]].copy() 
    if projected_actives_df.empty or mf_cloud_coords_df.empty:
        output_df['min_dist_to_mf_cloud'] = np.nan
        for k_val in k_for_knn_list: output_df[f'avg_dist_top_{k_val}_in_mf_cloud'] = np.nan
        output_df['dist_to_mf_cloud_centroid'] = np.nan
        return output_df
    coord_cols = [f"{dr_short_name}-{i+1}" for i in range(simspace_dim)]
    if not all(col in projected_actives_df.columns for col in coord_cols) or \
       not all(col in mf_cloud_coords_df.columns for col in coord_cols):
        logging.error(f"Coordinate columns mismatch for detailed distance calculation ({dr_short_name}).")
        output_df['min_dist_to_mf_cloud'] = np.nan
        for k_val in k_for_knn_list: output_df[f'avg_dist_top_{k_val}_in_mf_cloud'] = np.nan
        output_df['dist_to_mf_cloud_centroid'] = np.nan
        return output_df
    projected_points = projected_actives_df[coord_cols].values
    mf_cloud_points = mf_cloud_coords_df[coord_cols].values
    if mf_cloud_points.shape[0] > 0 and projected_points.shape[0] > 0:
        pairwise_distances = distance.cdist(projected_points, mf_cloud_points, 'euclidean')
        output_df['min_dist_to_mf_cloud'] = np.min(pairwise_distances, axis=1)
        for k_val in k_for_knn_list:
            if mf_cloud_points.shape[0] >= k_val:
                sorted_pairwise_dists = np.sort(pairwise_distances, axis=1)
                output_df[f'avg_dist_top_{k_val}_in_mf_cloud'] = np.mean(sorted_pairwise_dists[:, :k_val], axis=1)
            else: output_df[f'avg_dist_top_{k_val}_in_mf_cloud'] = np.nan
        mf_cloud_centroid = np.mean(mf_cloud_points, axis=0)
        dist_to_centroid_vals = distance.cdist(projected_points, mf_cloud_centroid.reshape(1, -1), 'euclidean')
        output_df['dist_to_mf_cloud_centroid'] = dist_to_centroid_vals.flatten()
    else:
        output_df['min_dist_to_mf_cloud'] = np.nan
        for k_val in k_for_knn_list: output_df[f'avg_dist_top_{k_val}_in_mf_cloud'] = np.nan
        output_df['dist_to_mf_cloud_centroid'] = np.nan
    return output_df

# (calculate_enrichment_metrics - same as last complete version)
def calculate_enrichment_metrics(ranked_df, active_label_col='is_active'):
    metrics = {}
    y_true = ranked_df[active_label_col].values
    valid_scores_mask = ranked_df['score'].notna()
    if valid_scores_mask.sum() < len(ranked_df): logging.warning(f"Found {len(ranked_df) - valid_scores_mask.sum()} NaN scores. Excluding them from enrichment calculation.")
    y_true_valid = y_true[valid_scores_mask]; y_scores_valid = ranked_df.loc[valid_scores_mask, 'score'].values
    if len(np.unique(y_true_valid)) < 2 or len(y_true_valid) < 2 : 
        logging.warning(f"Not enough class diversity or data points ({len(y_true_valid)}) after filtering NaN scores to calculate ROC/PR AUC or EFs.")
        metrics['roc_auc'] = np.nan; metrics['pr_auc'] = np.nan
        for x_percent in [0.01, 0.05, 0.10]: metrics[f'ef_{int(x_percent*100)}%'] = np.nan
        return metrics
    try:
        metrics['roc_auc'] = roc_auc_score(y_true_valid, y_scores_valid)
        precision, recall, _ = precision_recall_curve(y_true_valid, y_scores_valid)
        metrics['pr_auc'] = sklearn_auc(recall, precision)
    except ValueError as ve:
        logging.warning(f"ValueError during AUC calculation: {ve}"); metrics['roc_auc'] = np.nan; metrics['pr_auc'] = np.nan
    total_actives = np.sum(y_true_valid); total_compounds = len(y_true_valid)
    df_sorted_for_ef = ranked_df[valid_scores_mask].sort_values(by='score', ascending=False)
    for x_percent in [0.01, 0.05, 0.10]: 
        ef_key = f'ef_{int(x_percent*100)}%'
        if total_compounds == 0: metrics[ef_key] = np.nan; continue
        num_top_x = int(np.ceil(x_percent * total_compounds))
        if num_top_x == 0 and total_actives > 0: metrics[ef_key] = np.nan; continue
        elif num_top_x == 0 and total_actives == 0: metrics[ef_key] = 1.0; continue
        actives_in_top_x = np.sum(df_sorted_for_ef[active_label_col].iloc[:num_top_x])
        ef_denominator_ideal = (total_actives / total_compounds) 
        if ef_denominator_ideal > 0 : ef_observed = actives_in_top_x / num_top_x; metrics[ef_key] = ef_observed / ef_denominator_ideal
        else: metrics[ef_key] = 0.0 if actives_in_top_x > 0 else (1.0 if total_actives == 0 else np.nan)
    return metrics

# (calculate_affinity_correlation - same as last complete version with detailed logging)
def calculate_affinity_correlation(df_with_score_and_activity, activity_col='Standard Value (nM)'):
    logging.info(f"Attempting to calculate affinity correlation using activity column: '{activity_col}'.")
    if activity_col not in df_with_score_and_activity.columns: logging.warning(f"Activity column '{activity_col}' not found. Cannot calculate Spearman's Rho."); return np.nan
    if 'score' not in df_with_score_and_activity.columns: logging.warning(f"'score' column not found. Cannot calculate Spearman's Rho."); return np.nan
    logging.debug(f"Data for correlation (first 5 rows of relevant columns):\n{df_with_score_and_activity[['SMILES', activity_col, 'score']].head() if 'SMILES' in df_with_score_and_activity else df_with_score_and_activity[[activity_col, 'score']].head()}" )
    numeric_activity = pd.to_numeric(df_with_score_and_activity[activity_col], errors='coerce')
    valid_activity_mask = numeric_activity.notna() & (numeric_activity > 0) 
    logging.info(f"Total entries for correlation: {len(df_with_score_and_activity)}. Entries with valid numeric activity (>0): {valid_activity_mask.sum()}.")
    if valid_activity_mask.sum() < 2: logging.warning(f"Not enough valid activity data points ({valid_activity_mask.sum()}) to calculate Spearman's Rho."); return np.nan
    activity_molar = numeric_activity[valid_activity_mask] * 1e-9
    p_activity = -np.log10(activity_molar)
    scores_for_corr = df_with_score_and_activity.loc[valid_activity_mask, 'score']
    logging.debug(f"Number of pActivity values for corr: {len(p_activity)}, unique: {p_activity.nunique()}")
    logging.debug(f"Number of scores for corr: {len(scores_for_corr)}, unique: {scores_for_corr.nunique()}")
    if p_activity.nunique() < 2 or scores_for_corr.nunique() < 2 or len(p_activity) < 2 : logging.warning("Not enough unique value pairs in pActivity/scores or too few data points to calculate meaningful Spearman's Rho."); return np.nan
    try:
        correlation = pd.Series(p_activity, name='pActivity').corr(pd.Series(scores_for_corr, name='score'), method='spearman')
        logging.info(f"Calculated Spearman's Rho: {correlation:.4f}")
        if pd.isna(correlation): logging.warning("Spearman correlation resulted in NaN by pandas.Series.corr()."); return np.nan
        return correlation
    except Exception as e: logging.error(f"Error during Spearman correlation calculation: {e}"); return np.nan

# (plot_projection_results - same as last complete version)
def plot_projection_results(df_projected_target_actives_with_min_dist, df_simspace_main_data, mf_cloud_coords_for_plot, 
                            output_dir, target_id_name, dr_short_name, simspace_dim, 
                            representation_type):
    if simspace_dim != 2: logging.debug(f"Skipping 2D plots as simspace_dim is {simspace_dim}."); return
    coord_cols_2d = [f"{dr_short_name}-1", f"{dr_short_name}-2"]
    if not all(col in df_projected_target_actives_with_min_dist.columns for col in coord_cols_2d):
        logging.warning(f"Projected target actives missing 2D coordinate columns for plotting: {coord_cols_2d}"); return
    plt.style.use('seaborn-v0_8-whitegrid'); plt.figure(figsize=(12, 10))
    df_zinc_sample = pd.DataFrame()
    if 'MOLECULE ID' in df_simspace_main_data.columns:
        df_simspace_main_data['MOLECULE ID'] = df_simspace_main_data['MOLECULE ID'].astype(str)
        zinc_rows = df_simspace_main_data[df_simspace_main_data['MOLECULE ID'].str.startswith('ZINC', na=False)]
        if not zinc_rows.empty and len(zinc_rows) >=1 : df_zinc_sample = zinc_rows.sample(n=min(1000, len(zinc_rows)), random_state=42, replace=False)
    elif 'ZINC_ID' in df_simspace_main_data.columns: 
        zinc_rows = df_simspace_main_data[df_simspace_main_data['ZINC_ID'].notna()]
        if not zinc_rows.empty and len(zinc_rows) >=1 : df_zinc_sample = zinc_rows.sample(n=min(1000, len(zinc_rows)), random_state=42, replace=False)
    if not df_zinc_sample.empty and all(col in df_zinc_sample.columns for col in coord_cols_2d):
        plt.scatter(df_zinc_sample[coord_cols_2d[0]], df_zinc_sample[coord_cols_2d[1]], label="ZINC Sample (Decoys)", alpha=0.2, s=15, color='darkgrey', marker='.')
    if not mf_cloud_coords_for_plot.empty and all(col in mf_cloud_coords_for_plot.columns for col in coord_cols_2d):
        plt.scatter(mf_cloud_coords_for_plot[coord_cols_2d[0]], mf_cloud_coords_for_plot[coord_cols_2d[1]], label="Molecular Function Cloud (ChEMBL)", alpha=0.4, s=25, color='dodgerblue', marker='o')
    plt.scatter(df_projected_target_actives_with_min_dist[coord_cols_2d[0]], df_projected_target_actives_with_min_dist[coord_cols_2d[1]], label=f"Projected Target Actives ({target_id_name})", alpha=0.9, s=60, color='red', marker='P', edgecolor='black')
    plt.title(f"Similarity Space ({representation_type}, {dr_short_name}, Dim={simspace_dim})\nTarget: {target_id_name}", fontsize=14)
    plt.xlabel(coord_cols_2d[0], fontsize=12); plt.ylabel(coord_cols_2d[1], fontsize=12)
    plt.legend(fontsize=10); plt.grid(True, linestyle='--', alpha=0.7)
    scatter_path = os.path.join(output_dir, f"{target_id_name}_{representation_type}_{dr_short_name.replace('-', '_')}_dim{simspace_dim}_scatter.png")
    try: plt.savefig(scatter_path, dpi=200, bbox_inches='tight'); logging.info(f"Saved scatter plot to {scatter_path}")
    except Exception as e: logging.error(f"Failed to save scatter plot {scatter_path}: {e}")
    plt.close()
    if 'min_dist_to_mf_cloud' in df_projected_target_actives_with_min_dist.columns and df_projected_target_actives_with_min_dist['min_dist_to_mf_cloud'].notna().any():
        plt.figure(figsize=(9, 7))
        sns.histplot(df_projected_target_actives_with_min_dist['min_dist_to_mf_cloud'].dropna(), kde=True, bins=20, color='skyblue', edgecolor='black')
        mean_dist = df_projected_target_actives_with_min_dist['min_dist_to_mf_cloud'].mean(); median_dist = df_projected_target_actives_with_min_dist['min_dist_to_mf_cloud'].median()
        plt.title(f"Min. Distances of Target ACTIVES to MF Cloud\n{target_id_name} ({representation_type}, {dr_short_name}, Dim={simspace_dim})", fontsize=14)
        plt.xlabel("Minimum Euclidean Distance to MF Cloud", fontsize=12); plt.ylabel("Frequency", fontsize=12)
        if pd.notna(mean_dist): plt.axvline(mean_dist, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {mean_dist:.3f}')
        if pd.notna(median_dist): plt.axvline(median_dist, color='green', linestyle='dashed', linewidth=1.5, label=f'Median: {median_dist:.3f}')
        plt.legend(fontsize=10); plt.grid(True, linestyle='--', alpha=0.5)
        hist_path = os.path.join(output_dir, f"{target_id_name}_{representation_type}_{dr_short_name.replace('-', '_')}_dim{simspace_dim}_min_distances_hist_ACTIVES.png")
        try: plt.savefig(hist_path, dpi=200, bbox_inches='tight'); logging.info(f"Saved min distance histogram for ACTIVES to {hist_path}")
        except Exception as e: logging.error(f"Failed to save histogram {hist_path}: {e}")
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Project target ligands, analyze distances, calculate ranking metrics, and save full ranking for docking.")
    parser.add_argument("--target_ligands_repr_path", default=None, help="Path to CSV of target ligands (featurized/fingerprinted) - for PCA/UMAP.")
    parser.add_argument("--precomputed_target_projections_path", default=None, help="Path to CSV of pre-calculated target ligand projections (used for t-SNE).")
    parser.add_argument("--simspace_csv_path", required=True, help="Path to the comprehensive similarity space CSV (main data).")
    parser.add_argument("--model_dir_for_projection", default=None, help="Directory containing scaler and DR models (for PCA/UMAP).")
    parser.add_argument("--model_name_root_for_projection", default=None, help="Base name for finding models (e.g., target_repr_dimX).")
    parser.add_argument("--dr_method_key", required=True, help="Key for DR method from config (e.g., 'pca', 'umap_euclidean', 'tsne').")
    parser.add_argument("--dr_short_name", required=True, help="Short name for DR method (e.g., 'PCA', 'UMAP-Euclidean', 't-SNE').")
    parser.add_argument("--simspace_dim", type=int, required=True, help="Dimensionality of the similarity space.")
    parser.add_argument("--k_for_knn", type=str, required=True, help="Comma-separated list of k values for k-NN distance (for detailed_active_distances).")
    parser.add_argument("--output_dir", required=True, help="Directory to save analysis results.")
    parser.add_argument("--target_id_name", required=True, help="Target ID name for file naming.")
    parser.add_argument("--representation_type", required=True, choices=["features", "fingerprints"])
    parser.add_argument("--rdkit_features_list_target_str", required=True, help="JSON string of target RDKit feature names.")
    args = parser.parse_args()

    try:
        target_rdkit_features_list = json.loads(args.rdkit_features_list_target_str)
    except Exception as e:
        logging.error(f"Error parsing --rdkit_features_list_target_str: {e}. Aborting."); return

    df_projected_target_actives = pd.DataFrame()
    # --- 1. Get Projected Coordinates for Target Actives ---
    # (Same logic as previous full version for loading/projecting df_projected_target_actives)
    if args.precomputed_target_projections_path and os.path.exists(args.precomputed_target_projections_path):
        logging.info(f"Loading pre-computed target active projections for {args.dr_short_name} from: {args.precomputed_target_projections_path}")
        try:
            df_projected_target_actives = pd.read_csv(args.precomputed_target_projections_path)
            expected_coord_cols = [f"{args.dr_short_name}-{i+1}" for i in range(args.simspace_dim)]
            if not all(col in df_projected_target_actives.columns for col in expected_coord_cols): # Check if renaming is needed
                # Simple renaming if columns are just 0, 1, 2...
                current_cols = df_projected_target_actives.columns.tolist()
                potential_numeric_cols = [c for c in current_cols if c.isdigit() or (c.startswith('t-SNE-') and c.split('-')[-1].isdigit())]

                rename_map = {}
                if len(potential_numeric_cols) >= args.simspace_dim :
                    # Attempt to map based on common patterns or simple numeric sequence
                    # This part might need to be more robust depending on how t-SNE output cols are named
                    # if they aren't already the dr_short_name-N format.
                    # Assuming for now they might be '0', '1' or 't-SNE-1', 't-SNE-2' from previous script
                    for i in range(args.simspace_dim):
                        # If precomputed has '0', '1', etc.
                        if str(i) in df_projected_target_actives.columns:
                            rename_map[str(i)] = expected_coord_cols[i]
                        # If precomputed has specific format like 't-SNE-1' but short name is just 't-SNE'
                        elif f"{args.dr_short_name}-{i+1}" in df_projected_target_actives.columns and f"{args.dr_short_name}-{i+1}" != expected_coord_cols[i] :
                             # This case shouldn't happen if short_name is consistent, but as a safeguard
                             pass # Already correct or handled by first check if column name IS expected_coord_cols[i]
                        elif f"t-SNE-{i+1}" in df_projected_target_actives.columns and expected_coord_cols[i].startswith("t-SNE"): # common for tSNE
                            rename_map[f"t-SNE-{i+1}"] = expected_coord_cols[i] # ensure consistency if short name was slightly different

                if rename_map :
                    logging.info(f"Renaming precomputed projection columns: {rename_map}")
                    df_projected_target_actives.rename(columns=rename_map, inplace=True)
                # Final check after trying to rename
                if not all(col in df_projected_target_actives.columns for col in expected_coord_cols):
                     logging.warning(f"Precomputed target projections still missing expected columns {expected_coord_cols} after rename attempt. Columns present: {df_projected_target_actives.columns.tolist()}")
                     df_projected_target_actives = pd.DataFrame() # Invalidate if still wrong

        except Exception as e: logging.error(f"Error loading pre-computed target projections: {e}")
    if df_projected_target_actives.empty and args.dr_method_key != "tsne": # tSNE relies on precomputed
        logging.info(f"Projecting target actives for {args.target_id_name} ({args.representation_type}, {args.dr_short_name}, dim={args.simspace_dim}) using models...")
        if not args.target_ligands_repr_path or not os.path.exists(args.target_ligands_repr_path):
            logging.error(f"Target ligands representation path not provided or not found: {args.target_ligands_repr_path}. Cannot project for {args.dr_short_name}")
        elif not args.model_dir_for_projection or not os.path.exists(args.model_dir_for_projection):
             logging.error(f"Model directory for projection not provided or not found: {args.model_dir_for_projection}. Cannot project for {args.dr_short_name}")
        else:
            try:
                scaler_path = os.path.join(args.model_dir_for_projection, f"{args.model_name_root_for_projection}_scaler.lzma")
                dr_model_filename_part = ""
                if args.dr_method_key == "pca": dr_model_filename_part = "PCA_model.lzma"
                elif "umap" in args.dr_method_key:
                    metric_for_filename = args.dr_method_key.split("_")[1] if "_" in args.dr_method_key else args.dr_method_key # e.g. "euclidean"
                    dr_model_filename_part = f"{metric_for_filename}_UMAP_model.lzma"
                if not dr_model_filename_part: logging.error(f"Could not determine DR model filename for key: {args.dr_method_key}"); return
                dr_model_path = os.path.join(args.model_dir_for_projection, f"{args.model_name_root_for_projection}_{dr_model_filename_part}")

                if not os.path.exists(scaler_path): logging.error(f"Scaler model not found: {scaler_path}"); return
                if not os.path.exists(dr_model_path): logging.error(f"DR model not found: {dr_model_path}"); return

                with open(scaler_path, "rb") as f: scaler_model = decompress_pickle_load(f)
                with open(dr_model_path, "rb") as f: dr_model = decompress_pickle_load(f)
                df_target_ligands_repr_raw = pd.read_csv(args.target_ligands_repr_path, low_memory=False)
                df_projected_target_actives = project_target_ligands_with_models(
                    df_target_ligands_repr_raw, scaler_model, dr_model,
                    args.representation_type, args.simspace_dim, args.dr_method_key, args.dr_short_name,
                    target_rdkit_features_list
                )
            except Exception as e: logging.error(f"Error during model loading or projection for {args.dr_short_name}: {e}")

    if df_projected_target_actives.empty:
        logging.warning(f"No target actives projected or loaded for {args.dr_short_name}. Cannot perform ranking analysis for this DR method.")
        # Save empty metrics file
        pd.DataFrame([{'roc_auc': np.nan, 'pr_auc': np.nan, 'ef_1%': np.nan, 'ef_5%': np.nan, 'ef_10%': np.nan, 'spearman_rho_affinity_vs_score': np.nan}]).to_csv(
            os.path.join(args.output_dir, f"{args.target_id_name}_{args.representation_type}_{args.dr_short_name.replace('-', '_')}_dim{args.simspace_dim}_ranking_metrics.csv"), index=False
        )
        # Also save empty docking file
        docking_filename_empty = f"{args.target_id_name.upper()}-{args.dr_short_name.upper().replace('-', '')}-{args.simspace_dim}D-{args.representation_type.upper()}.csv"
        pd.DataFrame().to_csv(os.path.join(args.output_dir, docking_filename_empty), index=False)
        return # Critical to stop if no actives
    logging.info(f"Projected/loaded {len(df_projected_target_actives)} target actives for {args.dr_short_name}.")

    # --- 2. Load Main Simspace & Identify MF Cloud and ZINC Decoys ---
    try:
        df_simspace_main_data = pd.read_csv(args.simspace_csv_path, low_memory=False)
    except Exception as e:
        logging.error(f"Error loading main simspace file {args.simspace_csv_path}: {e}")
        df_simspace_main_data = pd.DataFrame() # Ensure it's a DataFrame

    mf_cloud_coords = pd.DataFrame()
    df_zinc_decoys_with_coords = pd.DataFrame()
    coord_cols_for_analysis = [f"{args.dr_short_name}-{i+1}" for i in range(args.simspace_dim)]

    if not df_simspace_main_data.empty:
        if not all(c in df_simspace_main_data.columns for c in coord_cols_for_analysis):
            logging.warning(f"Main simspace CSV {args.simspace_csv_path} missing one or more expected coordinate columns for {args.dr_short_name}: {coord_cols_for_analysis}. Present: {df_simspace_main_data.columns.tolist()}")
        elif 'MOLECULE ID' not in df_simspace_main_data.columns:
            logging.warning(f"Main simspace CSV {args.simspace_csv_path} missing 'MOLECULE ID' column.")
        else:
            df_simspace_main_data['MOLECULE ID'] = df_simspace_main_data['MOLECULE ID'].astype(str)
            # MF Cloud: Non-ZINC entries with valid coordinates
            chembl_mask = ~df_simspace_main_data['MOLECULE ID'].str.startswith('ZINC', na=False)
            mf_cloud_df_full = df_simspace_main_data[chembl_mask].copy()
            mf_cloud_coords_temp = mf_cloud_df_full[coord_cols_for_analysis].dropna()
            if not mf_cloud_coords_temp.empty:
                mf_cloud_coords = mf_cloud_coords_temp
            else:
                logging.warning(f"MF cloud for {args.dr_short_name} is empty after coordinate selection/dropna from {args.simspace_csv_path}.")

            # ZINC Decoys: ZINC entries with valid coordinates
            zinc_mask = df_simspace_main_data['MOLECULE ID'].str.startswith('ZINC', na=False)
            df_zinc_decoys_all_info = df_simspace_main_data[zinc_mask].copy()
            id_cols_zinc = [col for col in ['SMILES', 'MOLECULE ID', 'ZINC_ID'] if col in df_zinc_decoys_all_info.columns]
            df_zinc_decoys_with_coords_temp = df_zinc_decoys_all_info[id_cols_zinc + coord_cols_for_analysis].dropna(subset=coord_cols_for_analysis)
            if not df_zinc_decoys_with_coords_temp.empty:
                 df_zinc_decoys_with_coords = df_zinc_decoys_with_coords_temp
            else:
                logging.warning(f"ZINC decoys for {args.dr_short_name} is empty after coordinate selection/dropna from {args.simspace_csv_path}.")
    else:
        logging.warning(f"Main simspace data from {args.simspace_csv_path} is empty. Cannot extract MF cloud or ZINC decoys.")


    # --- 3. Calculate Scores for Actives and Decoys ---
    # Ensure mf_cloud_coords is not empty before proceeding
    if mf_cloud_coords.empty:
        logging.warning(f"MF Cloud is empty for {args.dr_short_name}. Scores for actives and decoys will be NaN.")
        df_target_actives_scored = df_projected_target_actives.copy()
        df_target_actives_scored['min_dist_to_mf_cloud'] = np.nan
        df_target_actives_scored['score'] = np.nan

        df_zinc_decoys_scored = df_zinc_decoys_with_coords.copy() # df_zinc_decoys_with_coords might be empty too
        if not df_zinc_decoys_scored.empty:
            df_zinc_decoys_scored['min_dist_to_mf_cloud'] = np.nan
            df_zinc_decoys_scored['score'] = np.nan
    else:
        df_target_actives_scored = calculate_compound_scores(df_projected_target_actives, mf_cloud_coords, args.dr_short_name, args.simspace_dim)
        df_zinc_decoys_scored = calculate_compound_scores(df_zinc_decoys_with_coords, mf_cloud_coords, args.dr_short_name, args.simspace_dim)

    df_target_actives_scored['TYPE'] = 'HELDOUT_ACTIVE'
    if not df_zinc_decoys_scored.empty : df_zinc_decoys_scored['TYPE'] = 'DECOY'


    # --- 4. Combine, Rank, and Save for Docking ---
    df_for_ranking_and_docking = pd.DataFrame() # Initialize

    # Ensure MOLECULE ID is present in both, preferring specific IDs if available
    # For actives, MOLECULE ID might be Compound ChEMBL ID
    if 'Compound ChEMBL ID' in df_target_actives_scored.columns and 'MOLECULE ID' not in df_target_actives_scored.columns:
        df_target_actives_scored['MOLECULE ID'] = df_target_actives_scored['Compound ChEMBL ID']
    elif 'SMILES' in df_target_actives_scored.columns and 'MOLECULE ID' not in df_target_actives_scored.columns: # Fallback
        df_target_actives_scored['MOLECULE ID'] = "ACTIVE_" + df_target_actives_scored.index.astype(str) # Ensure unique ID if others fail
    df_target_actives_scored['MOLECULE ID'] = df_target_actives_scored['MOLECULE ID'].fillna("ACTIVE_UNKNOWN_" + df_target_actives_scored.index.astype(str))


    # For decoys, MOLECULE ID might be ZINC_ID
    if not df_zinc_decoys_scored.empty:
        if 'ZINC_ID' in df_zinc_decoys_scored.columns and 'MOLECULE ID' not in df_zinc_decoys_scored.columns:
            df_zinc_decoys_scored['MOLECULE ID'] = df_zinc_decoys_scored['ZINC_ID']
        elif 'SMILES' in df_zinc_decoys_scored.columns and 'MOLECULE ID' not in df_zinc_decoys_scored.columns: # Fallback
            df_zinc_decoys_scored['MOLECULE ID'] = "DECOY_" + df_zinc_decoys_scored.index.astype(str)
        df_zinc_decoys_scored['MOLECULE ID'] = df_zinc_decoys_scored['MOLECULE ID'].fillna("DECOY_UNKNOWN_" + df_zinc_decoys_scored.index.astype(str))


    # Define columns for docking output: MOLECULE ID, SMILES, TYPE, COORDINATES..., SCORE, RANK
    cols_for_docking_output = ['MOLECULE ID', 'SMILES', 'TYPE', 'score'] + coord_cols_for_analysis

    df_list_for_concat = []
    for df, df_type_label in [(df_target_actives_scored, "Actives"), (df_zinc_decoys_scored, "Decoys")]:
        if df is None or df.empty:
            logging.debug(f"{df_type_label} DataFrame is empty, skipping for docking concat.")
            continue
        temp_df = pd.DataFrame() # Start with empty df to selectively add columns
        for col in cols_for_docking_output:
            if col in df.columns:
                temp_df[col] = df[col]
            else: # Ensure column exists even if all NaNs, for consistent concatenation
                temp_df[col] = np.nan
        if not temp_df.empty:
            df_list_for_concat.append(temp_df)

    if not df_list_for_concat:
        logging.warning(f"No data from actives or decoys to save for docking output for {args.dr_short_name}.")
    else:
        df_for_ranking_and_docking = pd.concat(df_list_for_concat, ignore_index=True).dropna(subset=['score'])
        if not df_for_ranking_and_docking.empty:
            df_for_ranking_and_docking.sort_values(by='score', ascending=False, inplace=True)
            df_for_ranking_and_docking['RANKING'] = np.arange(1, len(df_for_ranking_and_docking) + 1)

            # Select final columns for docking output
            final_docking_cols = ['MOLECULE ID', 'SMILES', 'TYPE'] + \
                                 [col for col in coord_cols_for_analysis if col in df_for_ranking_and_docking.columns] + \
                                 ['score', 'RANKING']

            df_docking_output = df_for_ranking_and_docking[final_docking_cols].copy()

            # Rename coordinate columns for generic output if needed, e.g., COORD_1, COORD_2
            coord_rename_map = {old_col: f"COORD_{i+1}" for i, old_col in enumerate(coord_cols_for_analysis) if old_col in df_docking_output.columns}
            df_docking_output.rename(columns=coord_rename_map, inplace=True)

            docking_filename = f"{args.target_id_name.upper()}-{args.dr_short_name.upper().replace('-', '')}-{args.simspace_dim}D-{args.representation_type.upper()}.csv"
            docking_output_path = os.path.join(args.output_dir, docking_filename)
            try:
                df_docking_output.to_csv(docking_output_path, index=False)
                logging.info(f"Saved ranked data for docking to: {docking_output_path}")
            except Exception as e:
                logging.error(f"Failed to save docking output CSV {docking_output_path}: {e}")
        else:
            logging.warning(f"DataFrame for ranking/docking is empty after dropping NaN scores for {args.dr_short_name}.")


    # --- 5. Calculate Enrichment Metrics (using df_for_ranking_and_docking) ---
    ranking_metrics = {}
    if not df_for_ranking_and_docking.empty and 'TYPE' in df_for_ranking_and_docking.columns and df_for_ranking_and_docking['TYPE'].nunique() > 1:
        df_for_ranking_and_docking['is_active_enrichment'] = (df_for_ranking_and_docking['TYPE'] == 'HELDOUT_ACTIVE').astype(int)
        logging.info(f"Calculating enrichment metrics (Total ranked: {len(df_for_ranking_and_docking)}, Actives: {df_for_ranking_and_docking['is_active_enrichment'].sum()})...")
        ranking_metrics = calculate_enrichment_metrics(df_for_ranking_and_docking, active_label_col='is_active_enrichment')
    else:
        logging.warning(f"Not enough data or diversity in 'TYPE' column for enrichment metrics for {args.dr_short_name}. Populating with NaNs.")
        ranking_metrics = {'roc_auc': np.nan, 'pr_auc': np.nan, 'ef_1%': np.nan, 'ef_5%': np.nan, 'ef_10%': np.nan}

    # --- 6. Calculate Affinity Correlation (for actives only) ---
    if not df_target_actives_scored.empty and 'TYPE' in df_target_actives_scored.columns:
        actives_for_corr = df_target_actives_scored[df_target_actives_scored['TYPE']=='HELDOUT_ACTIVE']
        if not actives_for_corr.empty:
            logging.info(f"Preparing for affinity correlation using df_target_actives_scored. Actives with scores: {len(actives_for_corr)}")
            ranking_metrics['spearman_rho_affinity_vs_score'] = calculate_affinity_correlation(
                actives_for_corr,
                activity_col='Standard Value (nM)'
            )
        else:
            logging.info("No HELDOUT_ACTIVE type compounds in df_target_actives_scored for affinity correlation.")
            ranking_metrics['spearman_rho_affinity_vs_score'] = np.nan
    else:
        ranking_metrics['spearman_rho_affinity_vs_score'] = np.nan

    # --- Save Ranking Metrics ---
    df_ranking_metrics_output = pd.DataFrame([ranking_metrics])
    metrics_csv_path = os.path.join(args.output_dir, f"{args.target_id_name}_{args.representation_type}_{args.dr_short_name.replace('-', '_')}_dim{args.simspace_dim}_ranking_metrics.csv")
    try: df_ranking_metrics_output.to_csv(metrics_csv_path, index=False); logging.info(f"Saved ranking metrics to: {metrics_csv_path}")
    except Exception as e: logging.error(f"Failed to save ranking metrics CSV {metrics_csv_path}: {e}")


    # --- Save Detailed Distances for Actives (Supplementary) ---
    k_for_knn_list_int = [int(k.strip()) for k in args.k_for_knn.split(',')]
    # Use df_projected_target_actives as it's the input for detailed distances
    if not df_projected_target_actives.empty and not mf_cloud_coords.empty:
        df_detailed_active_distances = calculate_detailed_distances_for_actives(df_projected_target_actives, mf_cloud_coords, k_for_knn_list_int, args.dr_short_name, args.simspace_dim)
        detailed_distances_path = os.path.join(args.output_dir, f"{args.target_id_name}_{args.representation_type}_{args.dr_short_name.replace('-', '_')}_dim{args.simspace_dim}_detailed_active_distances.csv")
        try: df_detailed_active_distances.to_csv(detailed_distances_path, index=False); logging.info(f"Saved detailed distances for ACTIVES to: {detailed_distances_path}")
        except Exception as e: logging.error(f"Failed to save detailed active distances CSV: {e}")
    elif mf_cloud_coords.empty:
         logging.warning(f"Cannot calculate detailed active distances as MF cloud is empty for {args.dr_short_name}.")


    # --- Generate Plots (if 2D) ---
    # Use df_target_actives_scored for plotting as it contains scores/distances for coloring/analysis if needed in future
    if args.simspace_dim == 2 and not df_target_actives_scored.empty and 'TYPE' in df_target_actives_scored.columns and not mf_cloud_coords.empty:
        actives_for_plot = df_target_actives_scored[df_target_actives_scored['TYPE']=='HELDOUT_ACTIVE']
        if not actives_for_plot.empty:
            logging.info("Generating 2D plots...")
            plot_projection_results(actives_for_plot,
                                    df_simspace_main_data, mf_cloud_coords,
                                    args.output_dir, args.target_id_name,
                                    args.dr_short_name, args.simspace_dim, args.representation_type)
        else:
            logging.info(f"No HELDOUT_ACTIVE type compounds to plot for {args.dr_short_name}.")
    elif args.simspace_dim == 2 and mf_cloud_coords.empty:
        logging.warning(f"Cannot generate 2D plots as MF cloud is empty for {args.dr_short_name}.")


    logging.info(f"Projection and analysis for {args.target_id_name} ({args.representation_type}, {args.dr_short_name}, dim={args.simspace_dim}) completed.")

if __name__ == "__main__":
    main()