#!/usr/bin/env python3
"""
Test Impact of MF Cloud Duplicates on Phase 1 Results

This script assesses whether deduplicating the MF cloud in existing Phase 1 
similarity spaces significantly affects enrichment metrics, or whether we need
to retrain DR models from scratch.

Strategy:
1. Load an existing Phase 1 similarity space (with duplicates)
2. Deduplicate the MF cloud coordinates using minimum affinity aggregation
3. Recalculate distance-based scores for target ligands
4. Recalculate enrichment metrics (EF@1%, ROC-AUC, etc.)
5. Compare original vs. deduplicated results

Decision criteria:
- If EF@1% changes < 5% → Minimal fix acceptable (just deduplicate coordinates)
- If EF@1% changes > 5% → Must retrain DR models (full Phase 1 rerun required)

Usage:
    python test_phase1_duplication_impact.py \\
        --phase1_run run_seed42_20241020_120000 \\
        --target TyrosineProteinKinaseABL1_P00519 \\
        --representation features \\
        --dr_method PCA \\
        --dimension 2
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
from scipy.spatial import distance
from sklearn.metrics import roc_auc_score, average_precision_score

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_similarity_space(simspace_path):
    """Load similarity space CSV with MF cloud, target actives, and ZINC decoys."""
    logging.info(f"Loading similarity space from: {simspace_path}")
    df = pd.read_csv(simspace_path, low_memory=False)
    logging.info(f"  Total compounds: {len(df):,}")
    
    # Identify compound types
    if 'DataSource' in df.columns:
        n_mf = len(df[df['DataSource'] == 'ChEMBL_MF'])
        n_zinc = len(df[df['DataSource'] == 'ZINC'])
        logging.info(f"  MF cloud: {n_mf:,}")
        logging.info(f"  ZINC decoys: {n_zinc:,}")
    
    return df


def load_target_actives(phase1_run, target_id, representation, dr_method, dimension):
    """Load target actives with coordinates (projected through DR model)."""
    # Try multiple possible file patterns
    patterns = [
        f"{target_id}/results/{representation}/dim_{dimension}/{dr_method}/{target_id}_actives_complete_{representation}_{dr_method}_dim{dimension}.csv",
        f"{target_id}/results/{representation}/dim_{dimension}/{dr_method.replace('-', '_')}/{target_id}_actives_complete_{representation}_{dr_method.replace('-', '_')}_dim{dimension}.csv",
    ]
    
    for pattern in patterns:
        actives_path = os.path.join(phase1_run, pattern)
        if os.path.exists(actives_path):
            logging.info(f"Loading target actives from: {actives_path}")
            df = pd.read_csv(actives_path, low_memory=False)
            logging.info(f"  Target actives: {len(df):,}")
            return df, actives_path
    
    logging.error(f"Could not find target actives file in {phase1_run}")
    return None, None


def load_affinity_data(phase1_run, target_id, representation):
    """Load affinity data for the MF cloud."""
    affinity_path = os.path.join(
        phase1_run, 
        target_id, 
        "temp_data", 
        f"{target_id}_chembl_mf_excluded_{representation}.csv"
    )
    
    if not os.path.exists(affinity_path):
        logging.warning(f"Affinity file not found: {affinity_path}")
        return None
    
    logging.info(f"Loading affinity data from: {affinity_path}")
    df = pd.read_csv(affinity_path, low_memory=False)
    
    if 'Standard Value (nM)' not in df.columns:
        logging.warning("No 'Standard Value (nM)' column in affinity file")
        return None
    
    logging.info(f"  Affinity records: {len(df):,}")
    logging.info(f"  Unique compounds: {df['Compound ChEMBL ID'].nunique():,}")
    
    # Check for duplicates
    dup_counts = df['Compound ChEMBL ID'].value_counts()
    dups = dup_counts[dup_counts > 1]
    if len(dups) > 0:
        logging.info(f"  ⚠ Duplicates found: {len(dups):,} compounds appear multiple times")
        logging.info(f"  ⚠ Duplication factor: {len(df) / df['Compound ChEMBL ID'].nunique():.2f}x")
    
    return df[['Compound ChEMBL ID', 'Standard Value (nM)']].copy()


def deduplicate_mf_cloud(df_simspace, affinity_df, coord_cols):
    """
    Deduplicate MF cloud using minimum affinity aggregation.
    
    Returns:
        df_mf_dedup: Deduplicated MF cloud with coordinates
        dedup_stats: Dictionary with deduplication statistics
    """
    logging.info("\n" + "="*80)
    logging.info("DEDUPLICATING MF CLOUD")
    logging.info("="*80)
    
    # Extract MF cloud
    if 'DataSource' in df_simspace.columns:
        df_mf = df_simspace[df_simspace['DataSource'] == 'ChEMBL_MF'].copy()
    else:
        # Assume non-ZINC entries are MF cloud
        df_mf = df_simspace[~df_simspace['MOLECULE ID'].str.startswith('ZINC', na=False)].copy()
    
    logging.info(f"MF cloud before deduplication: {len(df_mf):,} rows")
    logging.info(f"Unique Compound ChEMBL IDs: {df_mf['Compound ChEMBL ID'].nunique():,}")
    
    # Check for duplicates in coordinates
    coord_dup_counts = df_mf['Compound ChEMBL ID'].value_counts()
    coord_dups = coord_dup_counts[coord_dup_counts > 1]
    
    if len(coord_dups) == 0:
        logging.info("✓ No duplicates found in MF cloud coordinates")
        return df_mf, {'duplicates_found': False}
    
    logging.info(f"⚠ Found {len(coord_dups):,} compounds with duplicate coordinates")
    logging.info(f"⚠ Total duplicate rows: {coord_dup_counts[coord_dups].sum() - len(coord_dups):,}")
    
    # Merge with affinity data
    if affinity_df is not None:
        logging.info("\nMerging affinity data...")
        
        # Aggregate affinity to minimum per compound
        affinity_agg = affinity_df.groupby('Compound ChEMBL ID').agg({
            'Standard Value (nM)': 'min'
        }).reset_index()
        
        logging.info(f"  Aggregated {len(affinity_df):,} → {len(affinity_agg):,} unique compounds (using minimum affinity)")
        
        df_mf = df_mf.merge(affinity_agg, on='Compound ChEMBL ID', how='left')
        logging.info(f"  Merged affinity data for {df_mf['Standard Value (nM)'].notna().sum():,} of {len(df_mf):,} MF compounds")
    
    # Deduplicate by keeping row with minimum affinity (or first occurrence if no affinity)
    if 'Standard Value (nM)' in df_mf.columns and df_mf['Standard Value (nM)'].notna().any():
        # Sort by affinity (ascending) so minimum affinity comes first
        df_mf_sorted = df_mf.sort_values('Standard Value (nM)', ascending=True)
        df_mf_dedup = df_mf_sorted.drop_duplicates(subset='Compound ChEMBL ID', keep='first')
        logging.info(f"  Deduplicated using minimum affinity strategy")
    else:
        # No affinity data, just keep first occurrence
        df_mf_dedup = df_mf.drop_duplicates(subset='Compound ChEMBL ID', keep='first')
        logging.info(f"  Deduplicated using first-occurrence strategy (no affinity data)")
    
    logging.info(f"\nMF cloud after deduplication: {len(df_mf_dedup):,} rows")
    logging.info(f"Rows removed: {len(df_mf) - len(df_mf_dedup):,}")
    logging.info(f"Reduction: {(1 - len(df_mf_dedup)/len(df_mf)) * 100:.1f}%")
    
    dedup_stats = {
        'duplicates_found': True,
        'original_rows': len(df_mf),
        'deduplicated_rows': len(df_mf_dedup),
        'rows_removed': len(df_mf) - len(df_mf_dedup),
        'reduction_percent': (1 - len(df_mf_dedup)/len(df_mf)) * 100
    }
    
    return df_mf_dedup, dedup_stats


def calculate_distances_to_mf_cloud(df_actives, df_mf_cloud, coord_cols):
    """Calculate minimum distance from each active to MF cloud."""
    logging.info("\nCalculating distances to MF cloud...")
    
    actives_coords = df_actives[coord_cols].values
    mf_coords = df_mf_cloud[coord_cols].values
    
    logging.info(f"  Actives: {len(actives_coords):,} compounds")
    logging.info(f"  MF cloud: {len(mf_coords):,} compounds")
    
    # Calculate pairwise distances
    distances = distance.cdist(actives_coords, mf_coords, 'euclidean')
    
    # Get minimum distance for each active
    min_distances = distances.min(axis=1)
    
    logging.info(f"  Min distance range: [{min_distances.min():.4f}, {min_distances.max():.4f}]")
    logging.info(f"  Mean min distance: {min_distances.mean():.4f}")
    
    return min_distances


def calculate_enrichment_metrics(df_ranked, top_percent=0.01):
    """Calculate enrichment metrics from ranked compounds."""
    if 'is_active' not in df_ranked.columns:
        logging.error("No 'is_active' column in ranked data")
        return {}
    
    # Calculate top X%
    n_total = len(df_ranked)
    n_top = int(np.ceil(top_percent * n_total))
    
    # EF@X%
    n_actives_total = df_ranked['is_active'].sum()
    n_actives_in_top = df_ranked.head(n_top)['is_active'].sum()
    
    ef_observed = n_actives_in_top / n_top
    ef_random = n_actives_total / n_total
    ef = ef_observed / ef_random if ef_random > 0 else 0
    
    # ROC-AUC and PR-AUC
    if df_ranked['score'].notna().all() and df_ranked['is_active'].notna().all():
        try:
            roc_auc = roc_auc_score(df_ranked['is_active'], df_ranked['score'])
            pr_auc = average_precision_score(df_ranked['is_active'], df_ranked['score'])
        except:
            roc_auc = np.nan
            pr_auc = np.nan
    else:
        roc_auc = np.nan
        pr_auc = np.nan
    
    return {
        'ef_1_pct': ef,
        'n_actives_total': n_actives_total,
        'n_actives_in_top_1_pct': n_actives_in_top,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc
    }


def compare_metrics(original, deduplicated):
    """Compare original vs deduplicated metrics and determine if retraining is needed."""
    logging.info("\n" + "="*80)
    logging.info("COMPARISON: ORIGINAL vs DEDUPLICATED")
    logging.info("="*80)
    
    for metric in ['ef_1_pct', 'roc_auc', 'pr_auc']:
        orig_val = original.get(metric, np.nan)
        dedup_val = deduplicated.get(metric, np.nan)
        
        if not np.isnan(orig_val) and not np.isnan(dedup_val):
            diff = dedup_val - orig_val
            pct_change = (diff / orig_val * 100) if orig_val != 0 else np.nan
            
            logging.info(f"\n{metric.upper()}:")
            logging.info(f"  Original:      {orig_val:.4f}")
            logging.info(f"  Deduplicated:  {dedup_val:.4f}")
            logging.info(f"  Change:        {diff:+.4f} ({pct_change:+.2f}%)")
    
    # Decision logic
    ef_orig = original.get('ef_1_pct', np.nan)
    ef_dedup = deduplicated.get('ef_1_pct', np.nan)
    
    if not np.isnan(ef_orig) and not np.isnan(ef_dedup):
        pct_change = abs((ef_dedup - ef_orig) / ef_orig * 100) if ef_orig != 0 else 0
        
        logging.info("\n" + "="*80)
        logging.info("DECISION")
        logging.info("="*80)
        
        if pct_change < 5:
            logging.info(f"✓ EF@1% change is {pct_change:.2f}% (< 5% threshold)")
            logging.info("✓ MINIMAL FIX ACCEPTABLE:")
            logging.info("  - Deduplicate existing similarity space coordinates")
            logging.info("  - Recalculate distances and enrichment metrics")
            logging.info("  - NO need to retrain DR models")
            return 'minimal_fix'
        else:
            logging.info(f"✗ EF@1% change is {pct_change:.2f}% (> 5% threshold)")
            logging.info("✗ FULL RETRAINING REQUIRED:")
            logging.info("  - DR models likely learned distorted manifolds from duplicates")
            logging.info("  - Must deduplicate MF cloud BEFORE Phase 1")
            logging.info("  - Must retrain all DR models on clean data")
            logging.info("  - Must re-project all compounds and recalculate metrics")
            return 'full_retrain'
    
    return 'inconclusive'


def main():
    parser = argparse.ArgumentParser(
        description="Test impact of MF cloud deduplication on Phase 1 results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--phase1_run', required=True,
                       help='Path to Phase 1 run directory (e.g., experiment_workspace_v3_phase1/run_seed42_...)')
    parser.add_argument('--target', required=True,
                       help='Target ID (e.g., TyrosineProteinKinaseABL1_P00519)')
    parser.add_argument('--representation', required=True, choices=['features', 'fingerprints'],
                       help='Representation type')
    parser.add_argument('--dr_method', required=True,
                       help='DR method (e.g., PCA, UMAP-Euclidean)')
    parser.add_argument('--dimension', type=int, required=True,
                       help='Dimensionality (e.g., 2, 5, 10)')
    
    args = parser.parse_args()
    
    logging.info("="*80)
    logging.info("TESTING IMPACT OF MF CLOUD DUPLICATION ON PHASE 1 RESULTS")
    logging.info("="*80)
    logging.info(f"Phase 1 run: {args.phase1_run}")
    logging.info(f"Target: {args.target}")
    logging.info(f"Representation: {args.representation}")
    logging.info(f"DR method: {args.dr_method}")
    logging.info(f"Dimension: {args.dimension}")
    
    # Construct paths
    dr_method_safe = args.dr_method.replace('-', '_')
    simspace_path = os.path.join(
        args.phase1_run,
        args.target,
        "similarity_spaces",
        args.representation,
        f"dim_{args.dimension}",
        f"{args.target}_{args.representation}_dim{args.dimension}_similarity_space.csv"
    )
    
    if not os.path.exists(simspace_path):
        logging.error(f"Similarity space not found: {simspace_path}")
        sys.exit(1)
    
    # Define coordinate columns
    coord_cols = [f"{args.dr_method}-{i+1}" for i in range(args.dimension)]
    
    # 1. Load similarity space
    df_simspace = load_similarity_space(simspace_path)
    
    # 2. Load target actives
    df_actives, actives_path = load_target_actives(
        args.phase1_run, args.target, args.representation, dr_method_safe, args.dimension
    )
    
    if df_actives is None:
        logging.error("Could not load target actives - cannot proceed")
        sys.exit(1)
    
    # 3. Load affinity data
    affinity_df = load_affinity_data(args.phase1_run, args.target, args.representation)
    
    # 4. Get ORIGINAL MF cloud (with duplicates)
    if 'DataSource' in df_simspace.columns:
        df_mf_original = df_simspace[df_simspace['DataSource'] == 'ChEMBL_MF'].copy()
    else:
        df_mf_original = df_simspace[~df_simspace['MOLECULE ID'].str.startswith('ZINC', na=False)].copy()
    
    logging.info(f"\nOriginal MF cloud: {len(df_mf_original):,} compounds")
    
    # 5. Calculate ORIGINAL distances and scores
    logging.info("\n" + "="*80)
    logging.info("CALCULATING ORIGINAL METRICS (WITH DUPLICATES)")
    logging.info("="*80)
    
    min_distances_orig = calculate_distances_to_mf_cloud(df_actives, df_mf_original, coord_cols)
    df_actives_orig = df_actives.copy()
    df_actives_orig['min_dist_to_mf'] = min_distances_orig
    df_actives_orig['score'] = -min_distances_orig  # Higher score = closer to MF cloud
    df_actives_orig['is_active'] = 1  # All are actives
    
    # Add ZINC decoys for realistic enrichment calculation
    if 'DataSource' in df_simspace.columns:
        df_zinc = df_simspace[df_simspace['DataSource'] == 'ZINC'].copy()
        if not df_zinc.empty:
            logging.info(f"Adding {len(df_zinc):,} ZINC decoys for enrichment calculation")
            zinc_distances = calculate_distances_to_mf_cloud(df_zinc, df_mf_original, coord_cols)
            df_zinc['min_dist_to_mf'] = zinc_distances
            df_zinc['score'] = -zinc_distances
            df_zinc['is_active'] = 0
            
            # Combine for ranking
            df_ranked_orig = pd.concat([df_actives_orig, df_zinc], ignore_index=True)
        else:
            df_ranked_orig = df_actives_orig
    else:
        df_ranked_orig = df_actives_orig
    
    df_ranked_orig = df_ranked_orig.sort_values('score', ascending=False).reset_index(drop=True)
    df_ranked_orig['rank'] = df_ranked_orig.index + 1
    
    metrics_original = calculate_enrichment_metrics(df_ranked_orig)
    
    logging.info("\nOriginal metrics:")
    for k, v in metrics_original.items():
        logging.info(f"  {k}: {v}")
    
    # 6. Deduplicate MF cloud
    df_mf_dedup, dedup_stats = deduplicate_mf_cloud(df_simspace, affinity_df, coord_cols)
    
    if not dedup_stats.get('duplicates_found'):
        logging.info("\n✓ No duplicates found in MF cloud - no changes needed!")
        sys.exit(0)
    
    # 7. Calculate DEDUPLICATED distances and scores
    logging.info("\n" + "="*80)
    logging.info("CALCULATING DEDUPLICATED METRICS")
    logging.info("="*80)
    
    min_distances_dedup = calculate_distances_to_mf_cloud(df_actives, df_mf_dedup, coord_cols)
    df_actives_dedup = df_actives.copy()
    df_actives_dedup['min_dist_to_mf'] = min_distances_dedup
    df_actives_dedup['score'] = -min_distances_dedup
    df_actives_dedup['is_active'] = 1
    
    # Add ZINC decoys
    if 'DataSource' in df_simspace.columns:
        df_zinc = df_simspace[df_simspace['DataSource'] == 'ZINC'].copy()
        if not df_zinc.empty:
            zinc_distances_dedup = calculate_distances_to_mf_cloud(df_zinc, df_mf_dedup, coord_cols)
            df_zinc['min_dist_to_mf'] = zinc_distances_dedup
            df_zinc['score'] = -zinc_distances_dedup
            df_zinc['is_active'] = 0
            
            df_ranked_dedup = pd.concat([df_actives_dedup, df_zinc], ignore_index=True)
        else:
            df_ranked_dedup = df_actives_dedup
    else:
        df_ranked_dedup = df_actives_dedup
    
    df_ranked_dedup = df_ranked_dedup.sort_values('score', ascending=False).reset_index(drop=True)
    df_ranked_dedup['rank'] = df_ranked_dedup.index + 1
    
    metrics_deduplicated = calculate_enrichment_metrics(df_ranked_dedup)
    
    logging.info("\nDeduplicated metrics:")
    for k, v in metrics_deduplicated.items():
        logging.info(f"  {k}: {v}")
    
    # 8. Compare and make recommendation
    decision = compare_metrics(metrics_original, metrics_deduplicated)
    
    logging.info("\n" + "="*80)
    logging.info("TEST COMPLETE")
    logging.info("="*80)
    
    if decision == 'minimal_fix':
        sys.exit(0)
    elif decision == 'full_retrain':
        sys.exit(1)
    else:
        sys.exit(2)


if __name__ == '__main__':
    main()
