import pandas as pd
import numpy as np
import os
import argparse
import logging
from scipy.spatial import distance

# Basic Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)-8s - %(filename)-25s - %(funcName)-25s - %(lineno)-4d - %(message)s')

CDIST_BATCH_SIZE_RANK_ZINC = 10000 # Can be adjusted

def calculate_zinc_scores(zinc_df_with_coords, mf_cloud_coords_df, dr_short_name, simspace_dim):
    """Scores ZINC compounds based on min distance to MF cloud."""
    output_df = zinc_df_with_coords.copy()
    output_df['min_dist_to_mf_cloud'] = np.nan
    output_df['score'] = np.nan

    if zinc_df_with_coords.empty or mf_cloud_coords_df.empty:
        logging.warning("ZINC df or MF cloud df is empty. Cannot calculate scores.")
        return output_df

    coord_cols = [f"{dr_short_name}-{i+1}" for i in range(simspace_dim)]
    if not all(col in zinc_df_with_coords.columns for col in coord_cols):
        logging.error(f"ZINC df missing coordinate columns: {coord_cols}. Available: {zinc_df_with_coords.columns.tolist()}")
        return output_df
    if not all(col in mf_cloud_coords_df.columns for col in coord_cols):
        logging.error(f"MF cloud df missing coordinate columns: {coord_cols}. Available: {mf_cloud_coords_df.columns.tolist()}")
        return output_df

    try:
        zinc_points_all = zinc_df_with_coords[coord_cols].astype(float).values
        mf_cloud_points = mf_cloud_coords_df[coord_cols].astype(float).values
    except ValueError as ve:
        logging.error(f"Could not convert coordinate columns to float: {ve}")
        return output_df

    if mf_cloud_points.shape[0] > 0 and zinc_points_all.shape[0] > 0:
        all_min_distances = []
        num_zinc_points = zinc_points_all.shape[0]
        for i in range(0, num_zinc_points, CDIST_BATCH_SIZE_RANK_ZINC):
            batch_zinc_points = zinc_points_all[i:i+CDIST_BATCH_SIZE_RANK_ZINC]
            logging.debug(f"Processing ZINC cdist batch {i//CDIST_BATCH_SIZE_RANK_ZINC + 1}/{(num_zinc_points -1)//CDIST_BATCH_SIZE_RANK_ZINC + 1}, size: {batch_zinc_points.shape[0]}")
            try:
                pairwise_distances_batch = distance.cdist(batch_zinc_points, mf_cloud_points, 'euclidean')
                min_distances_batch = np.min(pairwise_distances_batch, axis=1)
                all_min_distances.append(min_distances_batch)
            except MemoryError as me:
                logging.error(f"MemoryError during ZINC cdist batch (size {CDIST_BATCH_SIZE_RANK_ZINC}): {me}.")
                all_min_distances.append(np.full(batch_zinc_points.shape[0], np.nan))
            except Exception as e:
                logging.error(f"Exception during ZINC cdist batch: {e}")
                all_min_distances.append(np.full(batch_zinc_points.shape[0], np.nan))
        
        if all_min_distances:
            min_distances_combined = np.concatenate(all_min_distances)
            if len(min_distances_combined) == len(output_df):
                output_df['min_dist_to_mf_cloud'] = min_distances_combined
                output_df['score'] = -min_distances_combined 
            else:
                logging.error("ZINC cdist batching length mismatch. Scores will be NaN.")
    return output_df


def main():
    parser = argparse.ArgumentParser(description="Rank ZINC decoys from a similarity space CSV based on proximity to MF cloud.")
    parser.add_argument("--simspace_csv_path", required=True, help="Path to the comprehensive similarity space CSV.")
    parser.add_argument("--dr_short_name", required=True, help="Short name of the DR method (e.g., PCA, UMAP-Euclidean) for coordinate columns.")
    parser.add_argument("--simspace_dim", type=int, required=True, help="Dimensionality of the similarity space.")
    parser.add_argument("--output_dir", required=True, help="Directory to save the ranked ZINC CSV file.")
    parser.add_argument("--target_id_name", required=True, help="Target ID name for output file naming.")
    parser.add_argument("--representation_type", required=True, help="Representation type (features/fingerprints) for output file naming.")
    args = parser.parse_args()

    logging.info(f"Starting ZINC decoy ranking for Target: {args.target_id_name}, Repr: {args.representation_type}, DR: {args.dr_short_name}, Dim: {args.simspace_dim}")
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        df_simspace_main = pd.read_csv(args.simspace_csv_path, low_memory=False)
    except FileNotFoundError:
        logging.error(f"Similarity space CSV not found: {args.simspace_csv_path}. Aborting.")
        return
    except Exception as e:
        logging.error(f"Error loading similarity space CSV {args.simspace_csv_path}: {e}. Aborting.")
        return

    if df_simspace_main.empty:
        logging.error(f"Similarity space CSV {args.simspace_csv_path} is empty. Aborting.")
        return

    if 'MOLECULE ID' not in df_simspace_main.columns:
        logging.error("'MOLECULE ID' column not found in simspace CSV. Cannot distinguish ZINC/ChEMBL. Aborting.")
        return
        
    df_simspace_main['MOLECULE ID'] = df_simspace_main['MOLECULE ID'].astype(str)

    coord_cols = [f"{args.dr_short_name}-{i+1}" for i in range(args.simspace_dim)]
    if not all(c in df_simspace_main.columns for c in coord_cols):
        logging.error(f"Simspace CSV missing one or more coordinate columns for {args.dr_short_name}: {coord_cols}. Present: {df_simspace_main.columns.tolist()}")
        # Attempt to proceed if some DR results might exist, but this DR method will be skipped for ranking.
        # For now, let's make it a hard stop for this specific DR method.
        return


    # Identify MF Cloud and ZINC compounds
    mf_cloud_mask = ~df_simspace_main['MOLECULE ID'].str.startswith('ZINC', na=False)
    df_mf_cloud_full = df_simspace_main[mf_cloud_mask].copy()
    df_zinc_full = df_simspace_main[~mf_cloud_mask].copy()

    mf_cloud_coords = df_mf_cloud_full[coord_cols].dropna()
    # ZINC compounds must have coordinates to be scored
    df_zinc_with_coords = df_zinc_full.dropna(subset=coord_cols).copy()


    if mf_cloud_coords.empty:
        logging.warning(f"MF Cloud has no valid coordinates for DR method {args.dr_short_name}. ZINC scores will be NaN.")
    if df_zinc_with_coords.empty:
        logging.warning(f"No ZINC compounds with valid coordinates found for DR method {args.dr_short_name}. No ranking will be performed.")
        # Save an empty file to indicate completion for this DR method
        output_filename = f"{args.target_id_name}_{args.representation_type}_{args.dr_short_name.replace('-', '_')}_dim{args.simspace_dim}_ranked_ZINC.csv"
        output_path = os.path.join(args.output_dir, output_filename)
        pd.DataFrame().to_csv(output_path, index=False)
        logging.info(f"Saved empty ranked ZINC file to: {output_path}")
        return

    logging.info(f"Ranking {len(df_zinc_with_coords)} ZINC compounds against {len(mf_cloud_coords)} MF Cloud points.")
    df_zinc_scored = calculate_zinc_scores(df_zinc_with_coords, mf_cloud_coords, args.dr_short_name, args.simspace_dim)

    if 'score' in df_zinc_scored.columns and df_zinc_scored['score'].notna().any():
        df_zinc_ranked = df_zinc_scored.sort_values(by='score', ascending=False).copy()
        df_zinc_ranked['RANKING'] = np.arange(1, len(df_zinc_ranked) + 1)
    else:
        logging.warning("No valid scores calculated for ZINC compounds. Ranking based on original order or index.")
        df_zinc_ranked = df_zinc_scored.copy()
        df_zinc_ranked['RANKING'] = np.nan # Indicate ranking was not possible based on score

    # Select columns for output: MOLECULE ID, SMILES (if exists), score, RANKING, coordinates
    output_cols = ['MOLECULE ID']
    if 'SMILES' in df_zinc_ranked.columns: output_cols.append('SMILES')
    output_cols.extend(['score', 'min_dist_to_mf_cloud', 'RANKING'])
    output_cols.extend(coord_cols)
    
    # Ensure all selected output columns exist in df_zinc_ranked
    final_output_cols = [col for col in output_cols if col in df_zinc_ranked.columns]
    df_zinc_output = df_zinc_ranked[final_output_cols]

    output_filename = f"{args.target_id_name}_{args.representation_type}_{args.dr_short_name.replace('-', '_')}_dim{args.simspace_dim}_ranked_ZINC.csv"
    output_path = os.path.join(args.output_dir, output_filename)
    try:
        df_zinc_output.to_csv(output_path, index=False)
        logging.info(f"Saved ranked ZINC compounds to: {output_path} (Shape: {df_zinc_output.shape})")
    except Exception as e:
        logging.error(f"Failed to save ranked ZINC CSV {output_path}: {e}")

if __name__ == "__main__":
    main()