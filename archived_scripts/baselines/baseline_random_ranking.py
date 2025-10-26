#!/usr/bin/env python3
"""
Random Shuffling Baseline for Virtual Screening (v2.0)

This script establishes the random performance baseline by shuffling the combined
list of held-out actives and decoys multiple times and calculating enrichment metrics.

Expected Result: Mean EF@1% ≈ 1.0 (confirms correct implementation)

Usage:
    python baselines/baseline_random_ranking.py \
        --target_actives datasets/protein_collection/TyrosineProteinKinaseABL1_P00519_affinity.csv \
        --zinc_decoys datasets/zinc_data.csv \
        --n_iterations 1000 \
        --output_dir baselines/results/random_baseline/
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def calculate_ef_at_1_percent(labels, top_percent=0.01):
    """
    Calculate Enrichment Factor at top X% of ranked list.
    
    Args:
        labels: Binary array (1=active, 0=decoy), already ranked
        top_percent: Fraction of top compounds to consider (default 1%)
    
    Returns:
        Enrichment Factor at top_percent
    """
    n_total = len(labels)
    n_actives = np.sum(labels)
    
    if n_actives == 0:
        return 0.0
    
    n_top = max(1, int(n_total * top_percent))
    n_actives_in_top = np.sum(labels[:n_top])
    
    # EF = (actives_found / total_in_top) / (total_actives / total_compounds)
    random_hit_rate = n_actives / n_total
    actual_hit_rate = n_actives_in_top / n_top
    
    ef = actual_hit_rate / random_hit_rate if random_hit_rate > 0 else 0.0
    return ef


def random_baseline_experiment(n_actives, n_decoys, n_iterations=1000, random_seed=42):
    """
    Perform random shuffling experiment.
    
    Args:
        n_actives: Number of active compounds
        n_decoys: Number of decoy compounds
        n_iterations: Number of random shuffles to perform
        random_seed: Random seed for reproducibility
    
    Returns:
        Dictionary with statistics
    """
    np.random.seed(random_seed)
    
    # Create labels array (1=active, 0=decoy)
    labels = np.array([1] * n_actives + [0] * n_decoys)
    n_total = len(labels)
    
    logger.info(f"Running random baseline: {n_actives} actives, {n_decoys} decoys, {n_iterations} iterations")
    
    ef1_scores = []
    
    for i in range(n_iterations):
        # Randomly shuffle indices
        shuffled_indices = np.random.permutation(n_total)
        shuffled_labels = labels[shuffled_indices]
        
        # Calculate EF@1%
        ef1 = calculate_ef_at_1_percent(shuffled_labels, top_percent=0.01)
        ef1_scores.append(ef1)
        
        if (i + 1) % 100 == 0:
            logger.info(f"Completed {i + 1}/{n_iterations} iterations")
    
    ef1_scores = np.array(ef1_scores)
    
    # Calculate statistics
    results = {
        'n_actives': n_actives,
        'n_decoys': n_decoys,
        'n_total': n_total,
        'n_iterations': n_iterations,
        'random_seed': random_seed,
        'ef1_mean': float(np.mean(ef1_scores)),
        'ef1_std': float(np.std(ef1_scores)),
        'ef1_median': float(np.median(ef1_scores)),
        'ef1_min': float(np.min(ef1_scores)),
        'ef1_max': float(np.max(ef1_scores)),
        'ef1_q1': float(np.percentile(ef1_scores, 25)),
        'ef1_q3': float(np.percentile(ef1_scores, 75)),
        'ef1_ci_95_lower': float(np.percentile(ef1_scores, 2.5)),
        'ef1_ci_95_upper': float(np.percentile(ef1_scores, 97.5)),
        'ef1_distribution': ef1_scores.tolist()
    }
    
    logger.info(f"\nRandom Baseline Results:")
    logger.info(f"  EF@1% Mean: {results['ef1_mean']:.3f} ± {results['ef1_std']:.3f}")
    logger.info(f"  EF@1% Median: {results['ef1_median']:.3f}")
    logger.info(f"  EF@1% 95% CI: [{results['ef1_ci_95_lower']:.3f}, {results['ef1_ci_95_upper']:.3f}]")
    logger.info(f"  EF@1% Range: [{results['ef1_min']:.3f}, {results['ef1_max']:.3f}]")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Random Shuffling Baseline for Virtual Screening")
    parser.add_argument("--target_actives", required=True, help="Path to target actives CSV")
    parser.add_argument("--zinc_decoys", required=True, help="Path to ZINC decoys CSV")
    parser.add_argument("--n_iterations", type=int, default=1000, help="Number of random shuffles")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", required=True, help="Output directory for results")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data to get counts
    logger.info(f"Loading target actives from: {args.target_actives}")
    df_actives = pd.read_csv(args.target_actives)
    n_actives = len(df_actives)
    logger.info(f"Found {n_actives} active compounds")
    
    logger.info(f"Loading ZINC decoys from: {args.zinc_decoys}")
    df_decoys = pd.read_csv(args.zinc_decoys)
    n_decoys = len(df_decoys)
    logger.info(f"Found {n_decoys} decoy compounds")
    
    # Extract target name from filename
    target_name = Path(args.target_actives).stem.replace("_affinity", "")
    
    # Run random baseline experiment
    results = random_baseline_experiment(n_actives, n_decoys, args.n_iterations, args.random_seed)
    
    # Save results
    results_file = output_dir / f"{target_name}_random_baseline_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved results to: {results_file}")
    
    # Save distribution to CSV for plotting
    distribution_df = pd.DataFrame({
        'iteration': range(args.n_iterations),
        'ef1_percent': results['ef1_distribution']
    })
    distribution_file = output_dir / f"{target_name}_random_baseline_ef1_distribution.csv"
    distribution_df.to_csv(distribution_file, index=False)
    logger.info(f"Saved EF@1% distribution to: {distribution_file}")
    
    # Save summary statistics
    summary_df = pd.DataFrame([{
        'target': target_name,
        'n_actives': results['n_actives'],
        'n_decoys': results['n_decoys'],
        'n_iterations': results['n_iterations'],
        'ef1_mean': results['ef1_mean'],
        'ef1_std': results['ef1_std'],
        'ef1_median': results['ef1_median'],
        'ef1_ci_95_lower': results['ef1_ci_95_lower'],
        'ef1_ci_95_upper': results['ef1_ci_95_upper']
    }])
    summary_file = output_dir / f"{target_name}_random_baseline_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    logger.info(f"Saved summary to: {summary_file}")
    
    logger.info("\n✅ Random baseline experiment complete!")


if __name__ == "__main__":
    main()
