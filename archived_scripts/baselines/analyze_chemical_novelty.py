#!/usr/bin/env python3
"""
Chemical Novelty Analysis (v2.0)

This script analyzes the chemical novelty of held-out actives relative to the MF cloud
by computing the distribution of maximum Tanimoto similarities. This helps contextualize
enrichment performance - finding actives that are chemically novel is more impressive
than finding actives similar to known ligands.

Interpretation:
    - High similarity (>0.7): Expected enrichment (similar to known actives)
    - Medium similarity (0.4-0.7): Moderate enrichment expected
    - Low similarity (<0.4): Impressive enrichment (finding novel chemotypes)

Usage:
    python baselines/analyze_chemical_novelty.py \
        --mf_cloud_fingerprints datasets/molecular_function_features_fingerprints/KW-0808_Transferase_affinity_extracted_fingerprints_ECFP4.csv \
        --target_actives_fingerprints datasets/protein_collection/TyrosineProteinKinaseABL1_P00519_affinity_extracted_fingerprints_ECFP4.csv \
        --output_dir baselines/results/chemical_novelty/
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_fingerprint_csv(csv_path, fp_col='ECFP4_2048', n_bits=2048):
    """Parse fingerprint CSV with comma-separated bit strings."""
    logger.info(f"Loading fingerprints from: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    
    fp_matrix = []
    smiles_list = []
    
    for idx, row in df.iterrows():
        fp_str = row[fp_col]
        if isinstance(fp_str, str):
            try:
                bits = list(map(int, fp_str.split(',')))
                if len(bits) == n_bits:
                    fp_matrix.append(bits)
                    smiles_list.append(row.get('SMILES', ''))
            except (ValueError, TypeError):
                continue
    
    fp_matrix = np.array(fp_matrix, dtype=np.int8)
    logger.info(f"Loaded {len(fp_matrix)} valid fingerprints")
    
    return fp_matrix, smiles_list


def tanimoto_similarity_vectorized(fp1, fp2_matrix):
    """Calculate Tanimoto similarity between one fingerprint and a matrix."""
    intersection = np.sum(fp1 & fp2_matrix, axis=1)
    union = np.sum(fp1 | fp2_matrix, axis=1)
    tanimoto = np.where(union > 0, intersection / union, 0.0)
    return tanimoto


def analyze_chemical_novelty(active_fps, cloud_fps, batch_size=100):
    """
    Analyze chemical novelty of actives relative to MF cloud.
    
    Args:
        active_fps: Active compound fingerprints
        cloud_fps: MF cloud fingerprints
        batch_size: Batch size for processing
    
    Returns:
        Array of max Tanimoto similarities for each active
    """
    n_actives = len(active_fps)
    max_similarities = np.zeros(n_actives)
    
    logger.info(f"\nAnalyzing chemical novelty for {n_actives} actives...")
    
    for i in range(0, n_actives, batch_size):
        end_i = min(i + batch_size, n_actives)
        batch = active_fps[i:end_i]
        
        for j, fp in enumerate(batch):
            similarities = tanimoto_similarity_vectorized(fp, cloud_fps)
            max_similarities[i + j] = np.max(similarities)
        
        if (end_i % 100 == 0) or (end_i == n_actives):
            logger.info(f"  Processed {end_i}/{n_actives} actives")
    
    return max_similarities


def calculate_novelty_statistics(similarities):
    """Calculate comprehensive statistics for novelty distribution."""
    stats = {
        'n_compounds': len(similarities),
        'mean_similarity': float(np.mean(similarities)),
        'median_similarity': float(np.median(similarities)),
        'std_similarity': float(np.std(similarities)),
        'min_similarity': float(np.min(similarities)),
        'max_similarity': float(np.max(similarities)),
        'q1_similarity': float(np.percentile(similarities, 25)),
        'q3_similarity': float(np.percentile(similarities, 75)),
        'iqr_similarity': float(np.percentile(similarities, 75) - np.percentile(similarities, 25)),
        
        # Novelty categories
        'n_high_similarity': int(np.sum(similarities > 0.7)),
        'n_medium_similarity': int(np.sum((similarities >= 0.4) & (similarities <= 0.7))),
        'n_low_similarity': int(np.sum(similarities < 0.4)),
        'pct_high_similarity': float(100 * np.sum(similarities > 0.7) / len(similarities)),
        'pct_medium_similarity': float(100 * np.sum((similarities >= 0.4) & (similarities <= 0.7)) / len(similarities)),
        'pct_low_similarity': float(100 * np.sum(similarities < 0.4) / len(similarities))
    }
    
    # Interpretation
    if stats['mean_similarity'] > 0.7:
        interpretation = "Low novelty - actives highly similar to MF cloud (enrichment expected)"
    elif stats['mean_similarity'] > 0.4:
        interpretation = "Moderate novelty - some chemical diversity from MF cloud"
    else:
        interpretation = "High novelty - actives chemically distinct from MF cloud (impressive if enriched)"
    
    stats['interpretation'] = interpretation
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Chemical Novelty Analysis")
    parser.add_argument("--mf_cloud_fingerprints", required=True, help="MF cloud fingerprints CSV")
    parser.add_argument("--target_actives_fingerprints", required=True, help="Target actives fingerprints CSV")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--fp_column", default="ECFP4_2048", help="Fingerprint column name")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load fingerprints
    cloud_fps, _ = parse_fingerprint_csv(args.mf_cloud_fingerprints, args.fp_column)
    active_fps, active_smiles = parse_fingerprint_csv(args.target_actives_fingerprints, args.fp_column)
    
    # Extract target name
    target_name = Path(args.target_actives_fingerprints).stem.replace("_affinity_extracted_fingerprints_ECFP4", "")
    
    # Analyze novelty
    max_similarities = analyze_chemical_novelty(active_fps, cloud_fps)
    
    # Calculate statistics
    stats = calculate_novelty_statistics(max_similarities)
    
    logger.info(f"\nChemical Novelty Analysis Results:")
    logger.info(f"  Number of actives: {stats['n_compounds']}")
    logger.info(f"  Mean Tanimoto: {stats['mean_similarity']:.3f} ± {stats['std_similarity']:.3f}")
    logger.info(f"  Median Tanimoto: {stats['median_similarity']:.3f}")
    logger.info(f"  Range: [{stats['min_similarity']:.3f}, {stats['max_similarity']:.3f}]")
    logger.info(f"  IQR: [{stats['q1_similarity']:.3f}, {stats['q3_similarity']:.3f}]")
    logger.info(f"\n  Novelty Distribution:")
    logger.info(f"    High similarity (>0.7): {stats['n_high_similarity']} ({stats['pct_high_similarity']:.1f}%)")
    logger.info(f"    Medium similarity (0.4-0.7): {stats['n_medium_similarity']} ({stats['pct_medium_similarity']:.1f}%)")
    logger.info(f"    Low similarity (<0.4): {stats['n_low_similarity']} ({stats['pct_low_similarity']:.1f}%)")
    logger.info(f"\n  Interpretation: {stats['interpretation']}")
    
    # Save statistics
    stats_file = output_dir / f"{target_name}_chemical_novelty_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"\nSaved statistics to: {stats_file}")
    
    # Save distribution
    distribution_df = pd.DataFrame({
        'compound_index': range(len(max_similarities)),
        'max_tanimoto_to_mf_cloud': max_similarities,
        'smiles': active_smiles if len(active_smiles) == len(max_similarities) else [''] * len(max_similarities)
    })
    distribution_file = output_dir / f"{target_name}_chemical_novelty_distribution.csv"
    distribution_df.to_csv(distribution_file, index=False)
    logger.info(f"Saved distribution to: {distribution_file}")
    
    # Save summary
    summary_df = pd.DataFrame([{
        'target': target_name,
        'n_actives': stats['n_compounds'],
        'mean_tanimoto': stats['mean_similarity'],
        'median_tanimoto': stats['median_similarity'],
        'std_tanimoto': stats['std_similarity'],
        'q1_tanimoto': stats['q1_similarity'],
        'q3_tanimoto': stats['q3_similarity'],
        'pct_high_similarity': stats['pct_high_similarity'],
        'pct_medium_similarity': stats['pct_medium_similarity'],
        'pct_low_similarity': stats['pct_low_similarity'],
        'interpretation': stats['interpretation']
    }])
    summary_file = output_dir / f"{target_name}_chemical_novelty_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    logger.info(f"Saved summary to: {summary_file}")
    
    logger.info("\n✅ Chemical novelty analysis complete!")


if __name__ == "__main__":
    main()
