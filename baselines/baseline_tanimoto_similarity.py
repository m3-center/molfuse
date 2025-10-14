#!/usr/bin/env python3
"""
Tanimoto Similarity Baseline (v2.0)

This script tests fingerprint-based similarity searching using maximum Tanimoto
similarity to the MF cloud. This is a standard benchmark in virtual screening.

Tests: Do feature-based DR methods outperform simple fingerprint similarity?

Usage:
    python baselines/baseline_tanimoto_similarity.py \
        --mf_cloud_fingerprints datasets/molecular_function_features_fingerprints/KW-0808_Transferase_affinity_extracted_fingerprints_ECFP4.csv \
        --target_actives_fingerprints datasets/protein_collection/TyrosineProteinKinaseABL1_P00519_affinity_extracted_fingerprints_ECFP4.csv \
        --zinc_decoys_fingerprints datasets/molecular_function_features_fingerprints/zinc/zinc_acquirable_extracted_fingerprints_ECFP4.csv \
        --output_dir baselines/results/tanimoto_baseline/
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
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


def parse_fingerprint_csv(csv_path, fp_col='ECFP4_2048', n_bits=2048):
    """
    Parse fingerprint CSV with comma-separated bit strings.
    
    Args:
        csv_path: Path to CSV file
        fp_col: Column name containing fingerprint string
        n_bits: Number of bits in fingerprint
    
    Returns:
        Binary matrix (n_compounds × n_bits)
    """
    logger.info(f"Loading fingerprints from: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    
    fp_matrix = []
    valid_indices = []
    
    for idx, fp_str in enumerate(df[fp_col]):
        if isinstance(fp_str, str):
            try:
                bits = list(map(int, fp_str.split(',')))
                if len(bits) == n_bits:
                    fp_matrix.append(bits)
                    valid_indices.append(idx)
            except (ValueError, TypeError):
                continue
    
    fp_matrix = np.array(fp_matrix, dtype=np.int8)
    logger.info(f"Loaded {len(fp_matrix)} valid fingerprints ({len(fp_matrix)}/{len(df)} compounds)")
    
    return fp_matrix


def tanimoto_similarity_vectorized(fp1, fp2_matrix):
    """
    Calculate Tanimoto similarity between one fingerprint and a matrix of fingerprints.
    
    Tanimoto = (A ∩ B) / (A ∪ B) = (A ∩ B) / (|A| + |B| - |A ∩ B|)
    
    Args:
        fp1: Single fingerprint (n_bits,)
        fp2_matrix: Matrix of fingerprints (n_fps × n_bits)
    
    Returns:
        Array of Tanimoto similarities (n_fps,)
    """
    intersection = np.sum(fp1 & fp2_matrix, axis=1)
    union = np.sum(fp1 | fp2_matrix, axis=1)
    
    # Avoid division by zero
    tanimoto = np.where(union > 0, intersection / union, 0.0)
    return tanimoto


def max_tanimoto_to_cloud(query_fps, cloud_fps, batch_size=100):
    """
    Calculate maximum Tanimoto similarity from each query to any cloud member.
    
    Args:
        query_fps: Query fingerprints (n_query × n_bits)
        cloud_fps: Cloud fingerprints (n_cloud × n_bits)
        batch_size: Process queries in batches to manage memory
    
    Returns:
        Array of max Tanimoto similarities (n_query,)
    """
    n_query = len(query_fps)
    max_similarities = np.zeros(n_query)
    
    logger.info(f"Computing max Tanimoto similarities for {n_query} queries...")
    
    for i in range(0, n_query, batch_size):
        end_i = min(i + batch_size, n_query)
        batch = query_fps[i:end_i]
        
        for j, fp in enumerate(batch):
            similarities = tanimoto_similarity_vectorized(fp, cloud_fps)
            max_similarities[i + j] = np.max(similarities)
        
        if (end_i % 1000 == 0) or (end_i == n_query):
            logger.info(f"  Processed {end_i}/{n_query} queries")
    
    return max_similarities


def tanimoto_baseline(mf_fps, active_fps, decoy_fps):
    """
    Rank compounds by maximum Tanimoto similarity to MF cloud.
    
    Args:
        mf_fps: MF cloud fingerprints
        active_fps: Held-out actives fingerprints
        decoy_fps: ZINC decoys fingerprints
    
    Returns:
        Dictionary with performance metrics
    """
    logger.info(f"\nCalculating Tanimoto similarities...")
    logger.info(f"  MF cloud: {mf_fps.shape}")
    logger.info(f"  Actives: {active_fps.shape}")
    logger.info(f"  Decoys: {decoy_fps.shape}")
    
    # Calculate max Tanimoto similarity to MF cloud
    active_similarities = max_tanimoto_to_cloud(active_fps, mf_fps)
    decoy_similarities = max_tanimoto_to_cloud(decoy_fps, mf_fps)
    
    # Combine labels and similarities
    labels = np.array([1] * len(active_similarities) + [0] * len(decoy_similarities))
    similarities = np.concatenate([active_similarities, decoy_similarities])
    
    # Rank by similarity (descending = most similar first)
    sorted_indices = np.argsort(-similarities)  # Negative for descending
    sorted_labels = labels[sorted_indices]
    sorted_similarities = similarities[sorted_indices]
    
    # Calculate metrics
    ef1 = calculate_ef_at_1_percent(sorted_labels, top_percent=0.01)
    roc_auc = roc_auc_score(labels, similarities)
    pr_auc = average_precision_score(labels, similarities)
    
    results = {
        'n_actives': len(active_similarities),
        'n_decoys': len(decoy_similarities),
        'n_total': len(labels),
        'ef1_percent': float(ef1),
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'mean_tanimoto_actives': float(np.mean(active_similarities)),
        'median_tanimoto_actives': float(np.median(active_similarities)),
        'mean_tanimoto_decoys': float(np.mean(decoy_similarities)),
        'median_tanimoto_decoys': float(np.median(decoy_similarities)),
        'max_tanimoto_actives': float(np.max(active_similarities)),
        'max_tanimoto_decoys': float(np.max(decoy_similarities))
    }
    
    logger.info(f"\nTanimoto Similarity Baseline Results:")
    logger.info(f"  EF@1%: {results['ef1_percent']:.3f}")
    logger.info(f"  ROC-AUC: {results['roc_auc']:.3f}")
    logger.info(f"  PR-AUC: {results['pr_auc']:.3f}")
    logger.info(f"  Mean Tanimoto (actives): {results['mean_tanimoto_actives']:.3f}")
    logger.info(f"  Mean Tanimoto (decoys): {results['mean_tanimoto_decoys']:.3f}")
    
    return results, sorted_labels, sorted_similarities


def main():
    parser = argparse.ArgumentParser(description="Tanimoto Similarity Baseline")
    parser.add_argument("--mf_cloud_fingerprints", required=True, help="MF cloud fingerprints CSV")
    parser.add_argument("--target_actives_fingerprints", required=True, help="Target actives fingerprints CSV")
    parser.add_argument("--zinc_decoys_fingerprints", required=True, help="ZINC decoys fingerprints CSV")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--fp_column", default="ECFP4_2048", help="Fingerprint column name")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load fingerprints
    mf_fps = parse_fingerprint_csv(args.mf_cloud_fingerprints, args.fp_column)
    active_fps = parse_fingerprint_csv(args.target_actives_fingerprints, args.fp_column)
    decoy_fps = parse_fingerprint_csv(args.zinc_decoys_fingerprints, args.fp_column)
    
    # Extract target name
    target_name = Path(args.target_actives_fingerprints).stem.replace("_affinity_extracted_fingerprints_ECFP4", "")
    
    # Run baseline
    results, sorted_labels, sorted_similarities = tanimoto_baseline(mf_fps, active_fps, decoy_fps)
    
    # Save results
    results_file = output_dir / f"{target_name}_tanimoto_baseline_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved results to: {results_file}")
    
    # Save ranked list
    rankings_df = pd.DataFrame({
        'rank': range(1, len(sorted_labels) + 1),
        'label': sorted_labels,
        'tanimoto_similarity': sorted_similarities
    })
    rankings_file = output_dir / f"{target_name}_tanimoto_baseline_rankings.csv"
    rankings_df.to_csv(rankings_file, index=False)
    logger.info(f"Saved rankings to: {rankings_file}")
    
    # Save summary
    summary_df = pd.DataFrame([{
        'target': target_name,
        'method': 'Tanimoto_Similarity',
        'n_actives': results['n_actives'],
        'n_decoys': results['n_decoys'],
        'ef1_percent': results['ef1_percent'],
        'roc_auc': results['roc_auc'],
        'pr_auc': results['pr_auc']
    }])
    summary_file = output_dir / f"{target_name}_tanimoto_baseline_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    logger.info(f"Saved summary to: {summary_file}")
    
    logger.info("\n✅ Tanimoto similarity baseline complete!")


if __name__ == "__main__":
    main()
