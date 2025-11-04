#!/usr/bin/env python3
"""
39D Feature Space Distance Baseline (v2.0)

This script tests whether dimensionality reduction adds value over using the raw 39D
physicochemical feature space. Compounds are ranked by minimum Euclidean distance to
the MF cloud in the original 39D feature space (after StandardScaler).

Tests: Does DR improve performance over raw features?

Usage:
    python baselines/baseline_feature_space_distance.py \
        --mf_cloud_features datasets/molecular_function_features_fingerprints/KW-0808_Transferase_affinity_extracted_features.csv \
        --target_actives_features datasets/protein_collection/TyrosineProteinKinaseABL1_P00519_affinity_extracted_features.csv \
        --zinc_decoys_features datasets/molecular_function_features_fingerprints/zinc/zinc_acquirable_extracted_features.csv \
        --output_dir baselines/results/feature_space_baseline/
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.spatial.distance import cdist
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def calculate_ef_at_1_percent(labels, top_percent=0.01):
    """Calculate Enrichment Factor at top X% of ranked list."""
    n_total = len(labels)
    n_actives = np.sum(labels)
    
    if n_actives == 0:
        return 0.0
    
    n_top = max(1, int(n_total * top_percent))
    n_actives_in_top = np.sum(labels[:n_top])
    
    random_hit_rate = n_actives / n_total
    actual_hit_rate = n_actives_in_top / n_top
    
    ef = actual_hit_rate / random_hit_rate if random_hit_rate > 0 else 0.0
    return ef


def load_features(csv_path, feature_cols):
    """Load and extract physicochemical features."""
    logger.info(f"Loading features from: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    
    # Filter for feature columns that exist
    available_cols = [col for col in feature_cols if col in df.columns]
    logger.info(f"Found {len(available_cols)}/{len(feature_cols)} feature columns")
    
    # Drop NaN rows
    df_clean = df[available_cols].dropna()
    logger.info(f"Loaded {len(df_clean)} compounds with complete features")
    
    X = df_clean[available_cols].values.astype(np.float32)
    return X, df_clean


def feature_space_baseline(mf_features, active_features, decoy_features):
    """
    Rank compounds by minimum Euclidean distance to MF cloud in 39D feature space.
    
    Args:
        mf_features: MF cloud features (already scaled)
        active_features: Held-out actives features (already scaled)
        decoy_features: ZINC decoys features (already scaled)
    
    Returns:
        Dictionary with performance metrics
    """
    logger.info(f"\nCalculating distances in 39D feature space...")
    logger.info(f"  MF cloud: {mf_features.shape}")
    logger.info(f"  Actives: {active_features.shape}")
    logger.info(f"  Decoys: {decoy_features.shape}")
    
    # Calculate minimum distance to MF cloud for actives
    logger.info("Computing distances for actives...")
    active_distances = np.min(cdist(active_features, mf_features, metric='euclidean'), axis=1)
    
    # Calculate minimum distance to MF cloud for decoys
    logger.info("Computing distances for decoys...")
    decoy_distances = np.min(cdist(decoy_features, mf_features, metric='euclidean'), axis=1)
    
    # Combine labels and distances
    labels = np.array([1] * len(active_distances) + [0] * len(decoy_distances))
    distances = np.concatenate([active_distances, decoy_distances])
    
    # Rank by distance (ascending = most similar first)
    sorted_indices = np.argsort(distances)
    sorted_labels = labels[sorted_indices]
    sorted_distances = distances[sorted_indices]
    
    # Calculate metrics
    ef1 = calculate_ef_at_1_percent(sorted_labels, top_percent=0.01)
    
    # For ROC-AUC and PR-AUC, we need "scores" where higher = more likely active
    # Since lower distance = more similar, we use negative distances
    scores = -distances
    roc_auc = roc_auc_score(labels, scores)
    pr_auc = average_precision_score(labels, scores)
    
    results = {
        'n_actives': len(active_distances),
        'n_decoys': len(decoy_distances),
        'n_total': len(labels),
        'ef1_percent': float(ef1),
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'mean_distance_actives': float(np.mean(active_distances)),
        'median_distance_actives': float(np.median(active_distances)),
        'mean_distance_decoys': float(np.mean(decoy_distances)),
        'median_distance_decoys': float(np.median(decoy_distances))
    }
    
    logger.info(f"\n39D Feature Space Baseline Results:")
    logger.info(f"  EF@1%: {results['ef1_percent']:.3f}")
    logger.info(f"  ROC-AUC: {results['roc_auc']:.3f}")
    logger.info(f"  PR-AUC: {results['pr_auc']:.3f}")
    logger.info(f"  Mean distance (actives): {results['mean_distance_actives']:.3f}")
    logger.info(f"  Mean distance (decoys): {results['mean_distance_decoys']:.3f}")
    
    return results, sorted_labels, sorted_distances


def main():
    parser = argparse.ArgumentParser(description="39D Feature Space Distance Baseline")
    parser.add_argument("--mf_cloud_features", required=True, help="MF cloud features CSV")
    parser.add_argument("--target_actives_features", required=True, help="Target actives features CSV")
    parser.add_argument("--zinc_decoys_features", required=True, help="ZINC decoys features CSV")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define 39 RDKit features (same as in config)
    feature_cols = [
        "DipoleMoment", "ABC", "nAcid", "nBase", "nAromAtom", "nAtom", "nH", "nC", "nN", "nO", "nS",
        "nP", "nX", "nBonds", "nBondsO", "nBondsS", "nBondsD", "nBondsT", "nBondsA", "nBondsM",
        "nBondsKS", "nBondsKD", "EState_VSA7", "nHBAcc", "nHBDon", "Lipinski", "apol", "bpol",
        "nRing", "n3Ring", "n4Ring", "n5Ring", "n6Ring", "n7Ring", "n8Ring", "nRot", "Diameter",
        "TopoShapeIndex", "Vabc", "MW"
    ]
    
    # Load features
    mf_features, _ = load_features(args.mf_cloud_features, feature_cols)
    active_features, _ = load_features(args.target_actives_features, feature_cols)
    decoy_features, _ = load_features(args.zinc_decoys_features, feature_cols)
    
    # Apply StandardScaler (same preprocessing as DR methods)
    logger.info("\nApplying StandardScaler to features...")
    scaler = StandardScaler()
    mf_features_scaled = scaler.fit_transform(mf_features)
    active_features_scaled = scaler.transform(active_features)
    decoy_features_scaled = scaler.transform(decoy_features)
    
    # Extract target name
    target_name = Path(args.target_actives_features).stem.replace("_affinity_extracted_features", "")
    
    # Run baseline
    results, sorted_labels, sorted_distances = feature_space_baseline(
        mf_features_scaled, active_features_scaled, decoy_features_scaled
    )
    
    # Save results
    results_file = output_dir / f"{target_name}_feature_space_baseline_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved results to: {results_file}")
    
    # Save ranked list
    rankings_df = pd.DataFrame({
        'rank': range(1, len(sorted_labels) + 1),
        'label': sorted_labels,
        'distance': sorted_distances
    })
    rankings_file = output_dir / f"{target_name}_feature_space_baseline_rankings.csv"
    rankings_df.to_csv(rankings_file, index=False)
    logger.info(f"Saved rankings to: {rankings_file}")
    
    # Save summary
    summary_df = pd.DataFrame([{
        'target': target_name,
        'method': '39D_Feature_Space',
        'n_actives': results['n_actives'],
        'n_decoys': results['n_decoys'],
        'ef1_percent': results['ef1_percent'],
        'roc_auc': results['roc_auc'],
        'pr_auc': results['pr_auc']
    }])
    summary_file = output_dir / f"{target_name}_feature_space_baseline_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    logger.info(f"Saved summary to: {summary_file}")
    
    logger.info("\n✅ 39D feature space baseline complete!")


if __name__ == "__main__":
    main()
