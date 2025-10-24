import pandas as pd
import numpy as np
import os
import argparse
import logging
import json
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from compress_pickle import load as decompress_pickle_load
from scipy.spatial import distance
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc as sklearn_auc
from core_scripts.utils import PassthroughScaler

# CUML imports
try:
    from cuml.common.device_selection import using_device_type
    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False

# Setup basic logging
# Ensure this file logger also uses mode 'a' if orchestrator runs it multiple times for same log file,
# or give it unique names based on args.output_dir or similar if that's desired.
# For now, mode='a' is fine.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)-8s - %(filename)-25s - %(funcName)-25s - %(lineno)-4d - %(message)s',
                    handlers=[logging.FileHandler("project_analyze.log", mode='a'), logging.StreamHandler()])

FINGERPRINT_COLUMN_PREFIX = "fp_"
NUM_FINGERPRINT_BITS = 2048
CDIST_BATCH_SIZE = 5000 

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
    target_ligands_valid_df = target_ligands_repr_df[valid_rows_mask]
    if len(target_ligands_valid_df) < len(target_ligands_repr_df):
        logging.info(f"Dropped {len(target_ligands_repr_df) - len(target_ligands_valid_df)} target ligands with NaN values before projection.")
    if target_ligands_valid_df.empty: logging.warning("Target ligands DataFrame is empty after NaN drop."); return pd.DataFrame()
    
    X_target = target_ligands_valid_df[descriptor_columns].values.astype(np.float32)
    try: X_target_scaled = scaler_model.transform(X_target)
    except Exception as e: logging.error(f"Error scaling target ligands for {dr_short_name}: {e}"); return pd.DataFrame()
    
    projected_coords_values = None
    try:
        projected_coords_values = dr_model.transform(X_target_scaled)
    except Exception as e: logging.error(f"Error transforming target ligands with DR model {dr_short_name}: {e}"); return pd.DataFrame()
    
    projection_cols_names = [f"{dr_short_name}-{i+1}" for i in range(simspace_dim)]
    df_projected_coords_only = pd.DataFrame(projected_coords_values[:, :simspace_dim], columns=projection_cols_names, index=target_ligands_valid_df.index)
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
                num_failed_in_batch = batch_projected_points.shape[0]
                all_min_distances.append(np.full(num_failed_in_batch, np.nan))
            except Exception as e:
                logging.error(f"Exception during cdist batch processing: {e}")
                num_failed_in_batch = batch_projected_points.shape[0]
                all_min_distances.append(np.full(num_failed_in_batch, np.nan))
        if all_min_distances:
            min_distances_combined = np.concatenate(all_min_distances)
            if len(min_distances_combined) == len(output_df):
                output_df['min_dist_to_mf_cloud'] = min_distances_combined
                output_df['score'] = -min_distances_combined 
            else:
                logging.error(f"Length mismatch after cdist batching: expected {len(output_df)}, got {len(min_distances_combined)}. Scores will be NaN.")
    else:
        logging.debug("No points in MF cloud or projected points after type conversion/filtering.")
    return output_df

def calculate_detailed_distances_for_actives(projected_actives_df, mf_cloud_coords_df, k_for_knn_list, dr_short_name, simspace_dim):
    # Ensure essential ID columns are preserved, even if they don't exist in input
    id_cols_to_try = ['SMILES', 'Compound ChEMBL ID']
    existing_id_cols = [col for col in id_cols_to_try if col in projected_actives_df.columns]
    output_df = projected_actives_df[existing_id_cols].copy() if existing_id_cols else pd.DataFrame(index=projected_actives_df.index)

    output_df['min_dist_to_mf_cloud'] = np.nan 
    for k_val in k_for_knn_list: output_df[f'avg_dist_top_{k_val}_in_mf_cloud'] = np.nan
    output_df['dist_to_mf_cloud_centroid'] = np.nan

    if projected_actives_df.empty or mf_cloud_coords_df.empty:
        logging.debug("Actives or MF cloud empty for detailed distances.")
        return output_df
    coord_cols = [f"{dr_short_name}-{i+1}" for i in range(simspace_dim)]
    if not all(col in projected_actives_df.columns for col in coord_cols) or \
       not all(col in mf_cloud_coords_df.columns for col in coord_cols):
        logging.error(f"Coordinate columns mismatch for detailed distance calculation ({dr_short_name}).")
        return output_df 
    projected_points = projected_actives_df[coord_cols].values
    mf_cloud_points = mf_cloud_coords_df[coord_cols].values
    if mf_cloud_points.shape[0] > 0 and projected_points.shape[0] > 0:
        try:
            pairwise_distances = distance.cdist(projected_points, mf_cloud_points, 'euclidean')
            output_df['min_dist_to_mf_cloud'] = np.min(pairwise_distances, axis=1)
            for k_val in k_for_knn_list:
                if mf_cloud_points.shape[0] >= k_val:
                    sorted_pairwise_dists = np.sort(pairwise_distances, axis=1)
                    output_df[f'avg_dist_top_{k_val}_in_mf_cloud'] = np.mean(sorted_pairwise_dists[:, :k_val], axis=1)
            mf_cloud_centroid = np.mean(mf_cloud_points, axis=0)
            dist_to_centroid_vals = distance.cdist(projected_points, mf_cloud_centroid.reshape(1, -1), 'euclidean')
            output_df['dist_to_mf_cloud_centroid'] = dist_to_centroid_vals.flatten()
        except MemoryError as me:
            logging.error(f"MemoryError in calculate_detailed_distances_for_actives: {me}. Results for this section will be NaN.")
        except Exception as e:
            logging.error(f"Exception in calculate_detailed_distances_for_actives: {e}")
    return output_df

def calculate_enrichment_metrics(ranked_df, active_label_col='is_active'):
    metrics = {}
    y_true = ranked_df[active_label_col].values
    valid_scores_mask = ranked_df['score'].notna()
    if valid_scores_mask.sum() < len(ranked_df): logging.warning(f"Found {len(ranked_df) - valid_scores_mask.sum()} NaN scores. Excluding them from enrichment calculation.")
    y_true_valid = y_true[valid_scores_mask]; y_scores_valid = ranked_df.loc[valid_scores_mask, 'score'].values
    if len(np.unique(y_true_valid)) < 2 or len(y_true_valid) < 2 : 
        logging.warning(f"Not enough class diversity or data points ({len(y_true_valid)}, unique labels: {np.unique(y_true_valid)}) after filtering NaN scores to calculate ROC/PR AUC or EFs.")
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
        if num_top_x == 0 : # If top x% is 0 compounds
            metrics[ef_key] = np.nan # Or 0, definition dependent. NaN if cannot be calculated.
            logging.debug(f"EF for {x_percent*100}%: num_top_x is 0, setting EF to NaN.")
            continue
        actives_in_top_x = np.sum(df_sorted_for_ef[active_label_col].iloc[:num_top_x])
        ef_denominator_ideal = (total_actives / total_compounds) 
        if ef_denominator_ideal > 0 : 
            ef_observed = actives_in_top_x / num_top_x
            metrics[ef_key] = ef_observed / ef_denominator_ideal
        elif total_actives == 0: # No actives in the entire dataset
             metrics[ef_key] = 1.0 # By some definitions, if no actives, EF is 1 (no enrichment possible, no dis-enrichment)
        else: # total_actives > 0 but ef_denominator_ideal is 0 (should not happen if total_compounds > 0)
             metrics[ef_key] = np.nan 
    return metrics

def calculate_affinity_correlation(df_with_score_and_activity, activity_col='Standard Value (nM)'):
    logging.info(f"Attempting to calculate affinity correlation using activity column: '{activity_col}'.")
    if activity_col not in df_with_score_and_activity.columns: logging.warning(f"Activity column '{activity_col}' not found. Cannot calculate Spearman's Rho."); return np.nan
    if 'score' not in df_with_score_and_activity.columns: logging.warning(f"'score' column not found. Cannot calculate Spearman's Rho."); return np.nan
    
    df_corr_subset = df_with_score_and_activity[[activity_col, 'score']].copy()
    df_corr_subset[activity_col] = pd.to_numeric(df_corr_subset[activity_col], errors='coerce')
    df_corr_subset['score'] = pd.to_numeric(df_corr_subset['score'], errors='coerce')

    valid_activity_mask = df_corr_subset[activity_col].notna() & (df_corr_subset[activity_col] > 0)
    valid_score_mask = df_corr_subset['score'].notna()
    valid_overall_mask = valid_activity_mask & valid_score_mask

    logging.info(f"Total entries for correlation: {len(df_with_score_and_activity)}. Entries with valid numeric activity (>0) and valid score: {valid_overall_mask.sum()}.")
    if valid_overall_mask.sum() < 2: logging.warning(f"Not enough valid activity & score data points ({valid_overall_mask.sum()}) to calculate Spearman's Rho."); return np.nan
    
    activity_molar = df_corr_subset.loc[valid_overall_mask, activity_col] * 1e-9
    p_activity = -np.log10(activity_molar)
    scores_for_corr = df_corr_subset.loc[valid_overall_mask, 'score']
    
    logging.debug(f"Number of pActivity values for corr: {len(p_activity)}, unique: {p_activity.nunique()}")
    logging.debug(f"Number of scores for corr: {len(scores_for_corr)}, unique: {scores_for_corr.nunique()}")
    if p_activity.nunique() < 2 or scores_for_corr.nunique() < 2 or len(p_activity) < 2 : logging.warning("Not enough unique value pairs in pActivity/scores or too few data points to calculate meaningful Spearman's Rho."); return np.nan
    try:
        correlation = pd.Series(p_activity.values, name='pActivity').corr(pd.Series(scores_for_corr.values, name='score'), method='spearman')
        logging.info(f"Calculated Spearman's Rho: {correlation:.4f}")
        if pd.isna(correlation): logging.warning("Spearman correlation resulted in NaN by pandas.Series.corr()."); return np.nan
        return correlation
    except Exception as e: logging.error(f"Error during Spearman correlation calculation: {e}"); return np.nan

def plot_projection_results(df_projected_target_actives_with_min_dist, df_simspace_main_data, mf_cloud_coords_for_plot, 
                            output_dir, target_id_name, 
                            dr_short_name_for_plot, # MODIFIED: Use a specific param for plotting
                            simspace_dim, 
                            representation_type):
    if simspace_dim != 2: logging.debug(f"Skipping 2D plots as simspace_dim is {simspace_dim}."); return
    
    # Use dr_short_name_for_plot for coordinate column lookup IF it's the name used in the df
    # The df_projected_target_actives_with_min_dist gets its coordinate columns named based on the 
    # dr_short_name passed to project_target_ligands_with_models or from precomputed file column names.
    # Let's assume the coord_cols in df_projected_target_actives_with_min_dist use the dr_short_name_for_plot.
    coord_cols_2d = [f"{dr_short_name_for_plot}-1", f"{dr_short_name_for_plot}-2"]

    if not all(col in df_projected_target_actives_with_min_dist.columns for col in coord_cols_2d):
        # Fallback: if df_projected_target_actives_with_min_dist has columns like "PCA-1", but dr_short_name_for_plot differs
        # This situation needs careful handling. The most robust way is to ensure that the
        # df_projected_target_actives_with_min_dist ALWAYS has columns named according to the method being plotted.
        # The current project_and_analyze.py structure should ensure this if orchestrator passes correct dr_short_name.
        logging.warning(f"Projected target actives missing 2D coordinate columns for plotting: {coord_cols_2d}. Columns available: {df_projected_target_actives_with_min_dist.columns.tolist()}. This might indicate a mismatch in dr_short_name used for processing vs. plotting.")
        # Try to find any 'SHORTNAME-1', 'SHORTNAME-2' pattern if the specific one isn't found. This is a bit of a hack.
        found_fallback_coords = False
        for col_prefix_candidate in df_projected_target_actives_with_min_dist.columns:
            if col_prefix_candidate.endswith("-1"):
                base_prefix = col_prefix_candidate[:-2]
                if f"{base_prefix}-2" in df_projected_target_actives_with_min_dist.columns:
                    coord_cols_2d = [f"{base_prefix}-1", f"{base_prefix}-2"]
                    logging.info(f"Using fallback coordinate columns for plot: {coord_cols_2d}")
                    found_fallback_coords = True
                    break
        if not found_fallback_coords:
            logging.error("Still cannot find valid coordinate columns for 2D plot.")
            return

    if mf_cloud_coords_for_plot.empty and (df_simspace_main_data is None or df_simspace_main_data.empty):
        logging.warning("Skipping plot: MF cloud and main simspace data are both empty.")
        return

    plt.style.use('seaborn-v0_8-whitegrid'); plt.figure(figsize=(12, 10))
    df_zinc_sample = pd.DataFrame()
    if df_simspace_main_data is not None and not df_simspace_main_data.empty:
        # Simspace main data coord columns should also match dr_short_name_for_plot (or the base name it was created with)
        # This implies the df_simspace_main_data passed here should have coordinates corresponding to the current DR method.
        if all(col in df_simspace_main_data.columns for col in coord_cols_2d):
            if 'MOLECULE ID' in df_simspace_main_data.columns:
                df_simspace_main_data['MOLECULE ID'] = df_simspace_main_data['MOLECULE ID'].astype(str)
                zinc_rows = df_simspace_main_data[df_simspace_main_data['MOLECULE ID'].str.startswith('ZINC', na=False)]
                if not zinc_rows.empty and len(zinc_rows) >=1 : df_zinc_sample = zinc_rows.sample(n=min(1000, len(zinc_rows)), random_state=42, replace=False)
            elif 'ZINC_ID' in df_simspace_main_data.columns: 
                zinc_rows = df_simspace_main_data[df_simspace_main_data['ZINC_ID'].notna()]
                if not zinc_rows.empty and len(zinc_rows) >=1 : df_zinc_sample = zinc_rows.sample(n=min(1000, len(zinc_rows)), random_state=42, replace=False)
        else:
            logging.warning(f"df_simspace_main_data does not have expected coord_cols {coord_cols_2d} for ZINC sampling in plot.")
    
    if not df_zinc_sample.empty and all(col in df_zinc_sample.columns for col in coord_cols_2d):
        plt.scatter(df_zinc_sample[coord_cols_2d[0]], df_zinc_sample[coord_cols_2d[1]], label="ZINC Sample (Decoys)", alpha=0.2, s=15, color='darkgrey', marker='.')
    
    if not mf_cloud_coords_for_plot.empty and all(col in mf_cloud_coords_for_plot.columns for col in coord_cols_2d):
        plt.scatter(mf_cloud_coords_for_plot[coord_cols_2d[0]], mf_cloud_coords_for_plot[coord_cols_2d[1]], label="Molecular Function Cloud (ChEMBL)", alpha=0.4, s=25, color='dodgerblue', marker='o')
    
    plt.scatter(df_projected_target_actives_with_min_dist[coord_cols_2d[0]], df_projected_target_actives_with_min_dist[coord_cols_2d[1]], label=f"Projected Target Actives ({target_id_name})", alpha=0.9, s=60, color='red', marker='P', edgecolor='black')
    
    # Use dr_short_name_for_plot in title and filenames
    plot_title_dr_name = dr_short_name_for_plot 
    plt.title(f"Similarity Space ({representation_type}, {plot_title_dr_name}, Dim={simspace_dim})\nTarget: {target_id_name}", fontsize=14)
    plt.xlabel(coord_cols_2d[0], fontsize=12); plt.ylabel(coord_cols_2d[1], fontsize=12)
    plt.legend(fontsize=10); plt.grid(True, linestyle='--', alpha=0.7)
    
    # Filename uses plot_title_dr_name (which is dr_short_name_for_plot)
    scatter_path = os.path.join(output_dir, f"{target_id_name}_{representation_type}_{plot_title_dr_name.replace('-', '_')}_dim{simspace_dim}_scatter.png")
    try: plt.savefig(scatter_path, dpi=200, bbox_inches='tight'); logging.info(f"Saved scatter plot to {scatter_path}")
    except Exception as e: logging.error(f"Failed to save scatter plot {scatter_path}: {e}")
    plt.close()

    if 'min_dist_to_mf_cloud' in df_projected_target_actives_with_min_dist.columns and df_projected_target_actives_with_min_dist['min_dist_to_mf_cloud'].notna().any():
        plt.figure(figsize=(9, 7))
        sns.histplot(df_projected_target_actives_with_min_dist['min_dist_to_mf_cloud'].dropna(), kde=True, bins=20, color='skyblue', edgecolor='black')
        mean_dist = df_projected_target_actives_with_min_dist['min_dist_to_mf_cloud'].mean(); median_dist = df_projected_target_actives_with_min_dist['min_dist_to_mf_cloud'].median()
        plt.title(f"Min. Distances of Target ACTIVES to MF Cloud\n{target_id_name} ({representation_type}, {plot_title_dr_name}, Dim={simspace_dim})", fontsize=14)
        plt.xlabel("Minimum Euclidean Distance to MF Cloud", fontsize=12); plt.ylabel("Frequency", fontsize=12)
        if pd.notna(mean_dist): plt.axvline(mean_dist, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {mean_dist:.3f}')
        if pd.notna(median_dist): plt.axvline(median_dist, color='green', linestyle='dashed', linewidth=1.5, label=f'Median: {median_dist:.3f}')
        plt.legend(fontsize=10); plt.grid(True, linestyle='--', alpha=0.5)
        hist_path = os.path.join(output_dir, f"{target_id_name}_{representation_type}_{plot_title_dr_name.replace('-', '_')}_dim{simspace_dim}_min_distances_hist_ACTIVES.png")
        try: plt.savefig(hist_path, dpi=200, bbox_inches='tight'); logging.info(f"Saved min distance histogram for ACTIVES to {hist_path}")
        except Exception as e: logging.error(f"Failed to save histogram {hist_path}: {e}")
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Project target ligands, analyze distances, and calculate ranking metrics.")
    parser.add_argument("--simspace_csv_path", required=True, help="Path to the similarity space CSV.")
    parser.add_argument("--dr_method_key", required=True, help="Key for DR method from config (e.g., 'pca', 'umap_euclidean').")
    parser.add_argument("--dr_short_name", required=True, help="Short name for DR method for coordinate column lookup.")
    parser.add_argument("--simspace_dim", type=int, required=True)
    parser.add_argument("--k_for_knn", type=str, required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_id_name", required=True)
    parser.add_argument("--representation_type", required=True, choices=["features", "fingerprints"])
    parser.add_argument("--rdkit_features_list_target_str", required=True)
    # Arguments for projection strategy
    parser.add_argument("--target_ligands_repr_path", default=None)
    parser.add_argument("--model_dir_for_projection", default=None)
    parser.add_argument("--model_name_root_for_projection", default=None)
    parser.add_argument("--affinity_cutoff", type=float, default=100000.0, 
                        help="Affinity cutoff in nM. Only held-out actives with 'Standard Value (nM)' at or below this value will be used for analysis. Default is 100,000 nM (100 uM).")
    
    args = parser.parse_args()
    
    # The name for output files and plot titles should be derived from the unique output directory
    dr_name_for_outputs = os.path.basename(os.path.normpath(args.output_dir)).replace('_', '-')
    
    logging.info(f"--- project_and_analyze.py started for Target: {args.target_id_name}, Repr: {args.representation_type}, DR: {dr_name_for_outputs}, Dim: {args.simspace_dim} ---")
    logging.info(f"Using affinity cutoff for actives: <= {args.affinity_cutoff} nM")

    try:
        target_rdkit_features_list = json.loads(args.rdkit_features_list_target_str)
    except Exception as e:
        logging.error(f"Error parsing --rdkit_features_list_target_str: {e}. Aborting."); return

    df_projected_target_actives = pd.DataFrame()
    df_simspace_main_data = pd.DataFrame()
    
    # CASE 1: PROJECTION mode - target actives must be loaded/projected separately.
    # The main simspace CSV contains only MF Cloud + ZINC.
    # Triggered when BOTH --target_ligands_repr_path AND --model_dir_for_projection are provided
    if (args.target_ligands_repr_path and args.target_ligands_repr_path.lower() != 'none' and
        args.model_dir_for_projection and args.model_dir_for_projection.lower() != 'none'):
        logging.info(f"PROJECTION mode: Projecting target actives through DR models...")
        try:
            df_simspace_main_data = pd.read_csv(args.simspace_csv_path, low_memory=False)
            
            # v2.0: Simplified scaler logic
            # Features always use StandardScaler, fingerprints always use PassthroughScaler (no scaling)
            scaler_suffix = "scaler.lzma"
            if args.representation_type == "fingerprints":
                logging.info("Using PassthroughScaler for fingerprints (v2.0: no scaling for binary vectors).")
            else:
                logging.info("Using StandardScaler for features.")
                    
            scaler_path = os.path.join(args.model_dir_for_projection, f"{args.model_name_root_for_projection}_{scaler_suffix}")
            metric = args.dr_method_key.split("_")[-1] if "umap" in args.dr_method_key else ""
            dr_model_filename = f"{args.model_name_root_for_projection}_{args.dr_short_name.replace('-','_')}_model.lzma" if args.dr_method_key != "pca" else f"{args.model_name_root_for_projection}_PCA_model.lzma"
            if "umap" in args.dr_method_key:
                dr_model_filename = f"{args.model_name_root_for_projection}_{metric}_UMAP_model.lzma"
            dr_model_path = os.path.join(args.model_dir_for_projection, dr_model_filename)
            
            with open(scaler_path, "rb") as f: scaler_model = decompress_pickle_load(f)
            with open(dr_model_path, "rb") as f: dr_model = decompress_pickle_load(f)
            df_target_ligands_repr_raw = pd.read_csv(args.target_ligands_repr_path, low_memory=False)
            
            df_projected_target_actives = project_target_ligands_with_models(
                df_target_ligands_repr_raw, scaler_model, dr_model,
                args.representation_type, args.simspace_dim, args.dr_method_key, args.dr_short_name,
                target_rdkit_features_list)
        except Exception as e:
            logging.error(f"Error during model loading or projection for {args.dr_short_name}: {e}", exc_info=True)

    # CASE 2: Phase 2 DATA REUSE mode - reuse MF cloud + ZINC, but still project actives
    # The similarity space CSV only contains MF + ZINC (not actives)
    # Phase 2 still needs to project actives, but can reuse the similarity space
    else:
        logging.info(f"Phase 2 DATA REUSE mode: Reusing MF cloud + ZINC from Phase 1, projecting actives with new cutoff")
        try:
            # Load the similarity space (MF + ZINC only)
            df_simspace_main_data = pd.read_csv(args.simspace_csv_path, low_memory=False)
            logging.info(f"  Loaded {len(df_simspace_main_data)} compounds from pre-computed similarity space (MF + ZINC)")
            
            # Check if we have target ligands file and models to project actives
            if not args.target_ligands_repr_path or args.target_ligands_repr_path.lower() == 'none':
                logging.error("Phase 2 requires --target_ligands_repr_path to project actives with new cutoff!")
                return
            
            if not args.model_dir_for_projection or args.model_dir_for_projection.lower() == 'none':
                logging.error("Phase 2 requires --model_dir_for_projection to project actives!")
                return
            
            # Load and project target actives (same as Phase 1, but with different cutoff)
            logging.info(f"  Loading target ligands from: {args.target_ligands_repr_path}")
            df_target_ligands_repr_raw = pd.read_csv(args.target_ligands_repr_path, low_memory=False)
            
            # Load scaler and DR model
            scaler_suffix = "scaler.lzma"
            if args.representation_type == "fingerprints":
                logging.info("Using PassthroughScaler for fingerprints.")
            else:
                logging.info("Using StandardScaler for features.")
            
            scaler_path = os.path.join(args.model_dir_for_projection, f"{args.model_name_root_for_projection}_{scaler_suffix}")
            
            metric = args.dr_method_key.split("_")[-1] if "umap" in args.dr_method_key else ""
            if args.dr_method_key == "pca":
                dr_model_filename = f"{args.model_name_root_for_projection}_PCA_model.lzma"
            elif "umap" in args.dr_method_key:
                dr_model_filename = f"{args.model_name_root_for_projection}_{metric}_UMAP_model.lzma"
            else:
                dr_model_filename = f"{args.model_name_root_for_projection}_{args.dr_short_name.replace('-','_')}_model.lzma"
            
            dr_model_path = os.path.join(args.model_dir_for_projection, dr_model_filename)
            
            with open(scaler_path, "rb") as f: scaler_model = decompress_pickle_load(f)
            with open(dr_model_path, "rb") as f: dr_model = decompress_pickle_load(f)
            
            # Project target actives
            df_projected_target_actives = project_target_ligands_with_models(
                df_target_ligands_repr_raw, scaler_model, dr_model,
                args.representation_type, args.simspace_dim, args.dr_method_key, args.dr_short_name,
                target_rdkit_features_list)
            
            logging.info(f"  Projected {len(df_projected_target_actives)} target actives")
            
        except Exception as e:
            logging.error(f"Error in Phase 2 DATA REUSE mode: {e}", exc_info=True)
            return
    
    # DO NOT FILTER TARGET LIGANDS BY AFFINITY CUTOFF
    # The affinity_cutoff parameter applies only to the MF cloud (for scoring),
    # not to target ligands being evaluated. All target ligands should be present
    # in the ranking regardless of their potency.
    # 
    # Historical note: This filtering was previously applied here, but it was incorrect
    # for Phase 2 experiments where we want to evaluate enrichment across all potency tiers.
    logging.info(f"Target ligands loaded: {len(df_projected_target_actives)} compounds (no affinity filtering applied)")
    
    if df_projected_target_actives.empty:
        logging.warning(f"No target actives projected or loaded for DR: {args.dr_short_name} (method key: {args.dr_method_key}). Cannot perform ranking analysis.")
        # Save empty metrics file so report generator doesn't fail finding it
        empty_metrics_df = pd.DataFrame([{'roc_auc': np.nan, 'pr_auc': np.nan, 'ef_1%': np.nan, 'ef_5%': np.nan, 'ef_10%': np.nan, 'spearman_rho_affinity_vs_score': np.nan}])
        empty_metrics_path = os.path.join(args.output_dir, f"{args.target_id_name}_{args.representation_type}_{args.dr_short_name.replace('-', '_')}_dim{args.simspace_dim}_ranking_metrics.csv")
        try: empty_metrics_df.to_csv(empty_metrics_path, index=False)
        except Exception as e_save: logging.error(f"Failed to save empty metrics CSV {empty_metrics_path}: {e_save}")
        
        docking_filename_empty = f"{args.target_id_name.upper()}-{args.dr_short_name.upper().replace('-', '')}-{args.simspace_dim}D-{args.representation_type.upper()}.csv"
        empty_docking_path = os.path.join(args.output_dir, docking_filename_empty)
        try: pd.DataFrame().to_csv(empty_docking_path, index=False)
        except Exception as e_save: logging.error(f"Failed to save empty docking CSV {empty_docking_path}: {e_save}")
        return 
    logging.info(f"Successfully obtained {len(df_projected_target_actives)} target actives for DR: {args.dr_short_name} (method key: {args.dr_method_key}).")

    # NOTE: df_simspace_main_data is already populated from either:
    #   - PROJECTION mode: loaded at line ~362 (MF cloud + ZINC only)
    #   - DATA REUSE mode: loaded and split at line ~395 (excludes actives)
    # No need to reload the CSV here - this was a redundant operation that doubled I/O time!
    # OLD CODE (removed for performance):
    # try: df_simspace_main_data = pd.read_csv(args.simspace_csv_path, low_memory=False)
    # except Exception as e: logging.error(f"Error loading main simspace file {args.simspace_csv_path}: {e}"); df_simspace_main_data = pd.DataFrame()
    
    if df_projected_target_actives.empty:
        logging.warning(f"No target actives remaining after affinity filtering. Cannot perform ranking.");
        # (save empty files and return logic here)
        return
    if df_simspace_main_data.empty:
        logging.error(f"Main similarity space data is empty. Cannot perform ranking."); return
    
    
    mf_cloud_coords = pd.DataFrame(); df_zinc_decoys_with_coords = pd.DataFrame()
    coord_cols_for_analysis = [f"{args.dr_short_name}-{i+1}" for i in range(args.simspace_dim)]

    if not df_simspace_main_data.empty:
        # Check for 'DataSource' first as it's more explicit
        if 'DataSource' in df_simspace_main_data.columns:
            mf_cloud_mask = df_simspace_main_data['DataSource'] == 'ChEMBL_MF'
            zinc_mask = df_simspace_main_data['DataSource'] == 'ZINC'
        # Fallback to MOLECULE ID if DataSource is missing
        elif 'MOLECULE ID' in df_simspace_main_data.columns:
            mf_cloud_mask = ~df_simspace_main_data['MOLECULE ID'].str.startswith('ZINC', na=False)
            zinc_mask = df_simspace_main_data['MOLECULE ID'].str.startswith('ZINC', na=False)
        else:
            logging.error("'DataSource' or 'MOLECULE ID' column not found in simspace CSV. Cannot distinguish ZINC/ChEMBL.")
            mf_cloud_mask = pd.Series(False, index=df_simspace_main_data.index)
            zinc_mask = pd.Series(False, index=df_simspace_main_data.index)

        mf_cloud_df_full = df_simspace_main_data[mf_cloud_mask].copy()
        
        # Apply affinity cutoff to MF cloud for Phase 2 experiments
        # This filters which MF molecules are used for distance-based scoring
        if args.affinity_cutoff is not None:
            logging.info(f"Attempting to filter MF cloud by affinity cutoff: <= {args.affinity_cutoff} nM")
            
            # Check if affinity data is already in the similarity space
            if 'Standard Value (nM)' not in mf_cloud_df_full.columns:
                logging.info("Affinity data not in similarity space CSV. Attempting to merge from MF cloud source file...")
                
                # The MF cloud source file is in the Phase 1 run directory structure
                # Path: {phase1_run_dir}/{target_id}/temp_data/{target_id}_chembl_mf_excluded_{representation}.csv
                # We need to construct this path from the current simspace_csv_path
                
                # Extract directory structure from simspace_csv_path
                # Example: .../run_seed42.../TyrosineProteinKinaseABL1_P00519/similarity_spaces/features/dim_5/...csv
                simspace_dir = os.path.dirname(args.simspace_csv_path)
                # Go up to target directory: .../TyrosineProteinKinaseABL1_P00519/
                target_dir = os.path.dirname(os.path.dirname(os.path.dirname(simspace_dir)))
                
                # Construct path to MF cloud source file
                mf_file_path = os.path.join(target_dir, "temp_data", f"{args.target_id_name}_chembl_mf_excluded_{args.representation_type}.csv")
                
                if os.path.exists(mf_file_path):
                    logging.info(f"  Loading MF affinity data from: {mf_file_path}")
                    try:
                        # Load only the two required columns to minimize memory
                        mf_affinity_data = pd.read_csv(
                            mf_file_path, 
                            usecols=['Compound ChEMBL ID', 'Standard Value (nM)'], 
                            dtype={'Compound ChEMBL ID': str, 'Standard Value (nM)': np.float32},
                            low_memory=False
                        )
                        logging.info(f"  Loaded {len(mf_affinity_data)} affinity records from source file")
                        
                        # Merge affinity data into MF cloud
                        original_mf_count = len(mf_cloud_df_full)
                        mf_cloud_df_full = mf_cloud_df_full.merge(
                            mf_affinity_data,
                            on='Compound ChEMBL ID',
                            how='left'
                        )
                        
                        # Free memory immediately after merge
                        del mf_affinity_data
                        import gc
                        gc.collect()
                        
                        logging.info(f"  Merged affinity data for {len(mf_cloud_df_full[mf_cloud_df_full['Standard Value (nM)'].notna()])} of {original_mf_count} MF molecules")
                    except Exception as e:
                        logging.warning(f"  Failed to load/merge MF affinity data: {e}")
                        logging.warning(f"  Proceeding without affinity filtering (will use full MF cloud)")
                else:
                    logging.warning(f"  MF cloud source file not found: {mf_file_path}")
                    logging.warning(f"  Proceeding without affinity filtering (will use full MF cloud)")
            
            # Now apply the cutoff if we have affinity data
            if 'Standard Value (nM)' in mf_cloud_df_full.columns:
                mf_cloud_df_full['Standard Value (nM)'] = pd.to_numeric(mf_cloud_df_full['Standard Value (nM)'], errors='coerce')
                
                original_mf_count = len(mf_cloud_df_full)
                mf_cloud_df_full = mf_cloud_df_full[
                    (mf_cloud_df_full['Standard Value (nM)'] <= args.affinity_cutoff) |
                    (mf_cloud_df_full['Standard Value (nM)'].isna())
                ].copy()
                filtered_mf_count = len(mf_cloud_df_full)
                logging.info(f"  MF cloud filtering complete. Kept {filtered_mf_count} of {original_mf_count} MF molecules (cutoff: <= {args.affinity_cutoff} nM)")
            else:
                logging.warning("  'Standard Value (nM)' column not available. Using full MF cloud for all cutoffs.")
        
        mf_cloud_coords = mf_cloud_df_full.dropna(subset=coord_cols_for_analysis)
        df_zinc_decoys_all_info = df_simspace_main_data[zinc_mask].copy()
        df_zinc_decoys_with_coords = df_zinc_decoys_all_info.dropna(subset=coord_cols_for_analysis)
    else:
        logging.warning("Could not split main simspace data. Ranking may be incorrect.")
        
    df_target_actives_scored = calculate_compound_scores(df_projected_target_actives, mf_cloud_coords, args.dr_short_name, args.simspace_dim)
    df_zinc_decoys_scored = calculate_compound_scores(df_zinc_decoys_with_coords, mf_cloud_coords, args.dr_short_name, args.simspace_dim)

    if not df_target_actives_scored.empty: df_target_actives_scored['TYPE'] = 'HELDOUT_ACTIVE'
    if not df_zinc_decoys_scored.empty : df_zinc_decoys_scored['TYPE'] = 'DECOY'       

    df_for_ranking_and_docking = pd.DataFrame() 

    if not df_target_actives_scored.empty:
        if 'Compound ChEMBL ID' in df_target_actives_scored.columns and 'MOLECULE ID' not in df_target_actives_scored.columns:
            df_target_actives_scored['MOLECULE ID'] = df_target_actives_scored['Compound ChEMBL ID']
        elif 'SMILES' in df_target_actives_scored.columns and 'MOLECULE ID' not in df_target_actives_scored.columns: 
            df_target_actives_scored['MOLECULE ID'] = "ACTIVE_SMILES_" + df_target_actives_scored.index.astype(str)
        elif 'MOLECULE ID' not in df_target_actives_scored.columns:
             df_target_actives_scored['MOLECULE ID'] = pd.Series("ACTIVE_IDX_" + df_target_actives_scored.index.astype(str), index=df_target_actives_scored.index)
        fill_values_actives = pd.Series("ACTIVE_UNKNOWN_" + df_target_actives_scored.index.astype(str), index=df_target_actives_scored.index)
        df_target_actives_scored['MOLECULE ID'] = df_target_actives_scored['MOLECULE ID'].fillna(fill_values_actives)

    if not df_zinc_decoys_scored.empty:
        if 'ZINC_ID' in df_zinc_decoys_scored.columns and 'MOLECULE ID' not in df_zinc_decoys_scored.columns:
            df_zinc_decoys_scored['MOLECULE ID'] = df_zinc_decoys_scored['ZINC_ID']
        elif 'SMILES' in df_zinc_decoys_scored.columns and 'MOLECULE ID' not in df_zinc_decoys_scored.columns: 
            df_zinc_decoys_scored['MOLECULE ID'] = "DECOY_SMILES_" + df_zinc_decoys_scored.index.astype(str)
        elif 'MOLECULE ID' not in df_zinc_decoys_scored.columns:
            df_zinc_decoys_scored['MOLECULE ID'] = pd.Series("DECOY_IDX_" + df_zinc_decoys_scored.index.astype(str), index=df_zinc_decoys_scored.index)
        fill_values_decoys = pd.Series("DECOY_UNKNOWN_" + df_zinc_decoys_scored.index.astype(str), index=df_zinc_decoys_scored.index)
        df_zinc_decoys_scored['MOLECULE ID'] = df_zinc_decoys_scored['MOLECULE ID'].fillna(fill_values_decoys)

    cols_for_docking_output = ['MOLECULE ID', 'SMILES', 'TYPE', 'score'] + coord_cols_for_analysis
    df_list_for_concat = []
    # Use .any() to check if df is not None and not empty
    if df_target_actives_scored is not None and not df_target_actives_scored.empty: 
        temp_df_actives = pd.DataFrame()
        for col in cols_for_docking_output:
            if col in df_target_actives_scored.columns: temp_df_actives[col] = df_target_actives_scored[col]
            else: temp_df_actives[col] = np.nan
        if not temp_df_actives.empty: df_list_for_concat.append(temp_df_actives)

    if df_zinc_decoys_scored is not None and not df_zinc_decoys_scored.empty:
        temp_df_decoys = pd.DataFrame()
        for col in cols_for_docking_output:
            if col in df_zinc_decoys_scored.columns: temp_df_decoys[col] = df_zinc_decoys_scored[col]
            else: temp_df_decoys[col] = np.nan
        if not temp_df_decoys.empty: df_list_for_concat.append(temp_df_decoys)

    if not df_list_for_concat:
        logging.warning(f"No data from actives or decoys to save for docking output for {args.dr_short_name}.")
    else:
        df_for_ranking_and_docking = pd.concat(df_list_for_concat, ignore_index=True)
        # Drop rows where score is NaN, as they cannot be ranked
        df_for_ranking_and_docking.dropna(subset=['score'], inplace=True) 
        if not df_for_ranking_and_docking.empty:
            df_for_ranking_and_docking.sort_values(by='score', ascending=False, inplace=True)
            df_for_ranking_and_docking['RANKING'] = np.arange(1, len(df_for_ranking_and_docking) + 1)
            final_docking_cols = ['MOLECULE ID', 'SMILES', 'TYPE'] + \
                                 [col for col in coord_cols_for_analysis if col in df_for_ranking_and_docking.columns] + \
                                 ['score', 'RANKING']
            df_docking_output = df_for_ranking_and_docking[final_docking_cols].copy()
            coord_rename_map = {old_col: f"COORD_{i+1}" for i, old_col in enumerate(coord_cols_for_analysis) if old_col in df_docking_output.columns}
            df_docking_output.rename(columns=coord_rename_map, inplace=True)

            docking_filename = f"{args.target_id_name.upper()}-{args.dr_short_name.upper().replace('-', '')}-{args.simspace_dim}D-{args.representation_type.upper()}.csv"
            docking_output_path = os.path.join(args.output_dir, docking_filename)
            try:
                df_docking_output.to_csv(docking_output_path, index=False)
                logging.info(f"Saved ranked data for docking to: {docking_output_path}")
            except Exception as e:
                logging.error(f"Failed to save docking output CSV {docking_output_path}: {e}")
            
            # Save comprehensive files for each molecule type (all info + coordinates)
            logging.info("Saving comprehensive data files for MF cloud, target actives, and ZINC decoys...")
            
            # 1. MF Cloud - complete info with coordinates
            if not mf_cloud_coords.empty:
                mf_output_path = os.path.join(args.output_dir, f"{args.target_id_name}_MF_cloud_complete_{args.representation_type}_{args.dr_short_name}_dim{args.simspace_dim}.csv")
                try:
                    mf_cloud_coords.to_csv(mf_output_path, index=False)
                    logging.info(f"Saved MF cloud complete data ({len(mf_cloud_coords)} compounds) to: {mf_output_path}")
                except Exception as e:
                    logging.error(f"Failed to save MF cloud complete CSV {mf_output_path}: {e}")
            
            # 2. Target Actives - complete info with coordinates and scores
            if not df_target_actives_scored.empty:
                actives_output_path = os.path.join(args.output_dir, f"{args.target_id_name}_actives_complete_{args.representation_type}_{args.dr_short_name}_dim{args.simspace_dim}.csv")
                try:
                    df_target_actives_scored.to_csv(actives_output_path, index=False)
                    logging.info(f"Saved target actives complete data ({len(df_target_actives_scored)} compounds) to: {actives_output_path}")
                except Exception as e:
                    logging.error(f"Failed to save actives complete CSV {actives_output_path}: {e}")
            
            # 3. ZINC Decoys - complete info with coordinates and scores
            if not df_zinc_decoys_scored.empty:
                zinc_output_path = os.path.join(args.output_dir, f"{args.target_id_name}_ZINC_decoys_complete_{args.representation_type}_{args.dr_short_name}_dim{args.simspace_dim}.csv")
                try:
                    df_zinc_decoys_scored.to_csv(zinc_output_path, index=False)
                    logging.info(f"Saved ZINC decoys complete data ({len(df_zinc_decoys_scored)} compounds) to: {zinc_output_path}")
                except Exception as e:
                    logging.error(f"Failed to save ZINC decoys complete CSV {zinc_output_path}: {e}")
        else:
            logging.warning(f"DataFrame for ranking/docking is empty after dropping NaN scores for {args.dr_short_name}.")

    ranking_metrics = {}
    if not df_for_ranking_and_docking.empty and 'TYPE' in df_for_ranking_and_docking.columns and df_for_ranking_and_docking['TYPE'].nunique() > 1:
        df_for_ranking_and_docking['is_active_enrichment'] = (df_for_ranking_and_docking['TYPE'] == 'HELDOUT_ACTIVE').astype(int)
        logging.info(f"Calculating enrichment metrics (Total ranked: {len(df_for_ranking_and_docking)}, Actives: {df_for_ranking_and_docking['is_active_enrichment'].sum()})...")
        ranking_metrics = calculate_enrichment_metrics(df_for_ranking_and_docking, active_label_col='is_active_enrichment')
    else:
        logging.warning(f"Not enough data or diversity in 'TYPE' column for enrichment metrics for {args.dr_short_name}. Populating with NaNs.")
        ranking_metrics = {'roc_auc': np.nan, 'pr_auc': np.nan, 'ef_1%': np.nan, 'ef_5%': np.nan, 'ef_10%': np.nan}

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

    df_ranking_metrics_output = pd.DataFrame([ranking_metrics])
    metrics_csv_path = os.path.join(args.output_dir, f"{args.target_id_name}_{args.representation_type}_{args.dr_short_name.replace('-', '_')}_dim{args.simspace_dim}_ranking_metrics.csv")
    try: df_ranking_metrics_output.to_csv(metrics_csv_path, index=False); logging.info(f"Saved ranking metrics to: {metrics_csv_path}")
    except Exception as e: logging.error(f"Failed to save ranking metrics CSV {metrics_csv_path}: {e}")

    k_for_knn_list_int = [int(k.strip()) for k in args.k_for_knn.split(',')]
    if not df_projected_target_actives.empty and not mf_cloud_coords.empty:
        df_detailed_active_distances = calculate_detailed_distances_for_actives(df_projected_target_actives, mf_cloud_coords, k_for_knn_list_int, args.dr_short_name, args.simspace_dim)
        detailed_distances_path = os.path.join(args.output_dir, f"{args.target_id_name}_{args.representation_type}_{args.dr_short_name.replace('-', '_')}_dim{args.simspace_dim}_detailed_active_distances.csv")
        try: df_detailed_active_distances.to_csv(detailed_distances_path, index=False); logging.info(f"Saved detailed distances for ACTIVES to: {detailed_distances_path}")
        except Exception as e: logging.error(f"Failed to save detailed active distances CSV: {e}")
    elif mf_cloud_coords.empty:
         logging.warning(f"Cannot calculate detailed active distances as MF cloud is empty for {args.dr_short_name}.")

    if args.simspace_dim == 2 and not df_target_actives_scored.empty and 'TYPE' in df_target_actives_scored.columns:
        actives_for_plot = df_target_actives_scored[df_target_actives_scored['TYPE']=='HELDOUT_ACTIVE']
        if not actives_for_plot.empty:
            logging.info("Generating 2D plots...")
            plot_projection_results(actives_for_plot, 
                                    df_simspace_main_data, mf_cloud_coords, 
                                    args.output_dir, args.target_id_name, 
                                    args.dr_short_name,
                                    args.simspace_dim, args.representation_type)
        else:
            logging.info(f"No HELDOUT_ACTIVE type compounds to plot for {args.dr_short_name}.")
    elif args.simspace_dim == 2 and df_target_actives_scored.empty:
        logging.warning(f"Cannot generate 2D plots as df_target_actives_scored is empty for {args.dr_short_name}.")

    logging.info(f"Projection and analysis for Target: {args.target_id_name}, Repr: {args.representation_type}, DR: {args.dr_short_name}, Dim: {args.simspace_dim} completed.")

if __name__ == "__main__":
    main()
