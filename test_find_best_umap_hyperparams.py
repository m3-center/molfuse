#!/usr/bin/env python3
"""
Quick test script to find the best UMAP hyperparameters from Phase 1 results.

Analyzes stratified_enrichment_detailed.csv to identify:
1. Best UMAP config for HIGH potency enrichment (EF@1% for high-potent actives)
2. Best UMAP config for OVERALL enrichment (EF@1% across all actives)

Usage:
    python test_find_best_umap_hyperparams.py
"""

import pandas as pd
import numpy as np
import os

# Path to Phase 1 detailed results
RESULTS_FILE = "reporting/stratified_enrichment_detailed.csv"

def parse_config_details(df):
    """Extract DR method, hyperparameters from config names."""
    results = []
    
    for _, row in df.iterrows():
        config = row['config']
        
        # Skip non-UMAP configs
        if 'umap' not in config.lower():
            continue
        
        # Parse UMAP hyperparameters from config name
        # Format: tyro_features_umap_euclidean_dim5_nn5_md0.0_seed42
        parts = config.split('_')
        
        representation = None
        metric = None
        dimension = None
        n_neighbors = None
        min_dist = None
        seed = None
        
        for i, part in enumerate(parts):
            if part == 'features' or part == 'fingerprints':
                representation = part
            elif part == 'euclidean' or part == 'cosine':
                metric = part
            elif part.startswith('dim') and len(part) > 3:
                dimension = int(part[3:])
            elif part.startswith('nn') and len(part) > 2:
                n_neighbors = int(part[2:])
            elif part.startswith('md'):
                try:
                    min_dist = float(part[2:])
                except ValueError:
                    pass
            elif part.startswith('seed'):
                try:
                    seed = int(part[4:])
                except ValueError:
                    pass
        
        # Add parsed data to results
        result = {
            'config': config,
            'representation': representation,
            'metric': metric,
            'dimension': dimension,
            'n_neighbors': n_neighbors,
            'min_dist': min_dist,
            'seed': seed,
            'target': row['target'],
            'ef_1_overall': row.get('ef_1%_overall', np.nan),
            'ef_1_high': row.get('ef_1%_high-potent', np.nan),
            'ef_1_medium': row.get('ef_1%_medium-potent', np.nan),
            'ef_1_low': row.get('ef_1%_low-potent', np.nan),
        }
        
        results.append(result)
    
    return pd.DataFrame(results)


def aggregate_by_hyperparams(df, metric_col):
    """Aggregate results by hyperparameter combinations.
    
    Args:
        df: DataFrame with parsed configs
        metric_col: Column to aggregate (e.g., 'ef_1_overall' or 'ef_1_high')
    
    Returns:
        DataFrame sorted by mean performance
    """
    # Group by hyperparameters (excluding seed and target)
    groupby_cols = ['representation', 'metric', 'dimension', 'n_neighbors', 'min_dist']
    
    # Filter out rows with missing hyperparams or metric values
    df_valid = df.dropna(subset=groupby_cols + [metric_col])
    
    if df_valid.empty:
        print(f"Warning: No valid data for metric {metric_col}")
        return pd.DataFrame()
    
    # Aggregate
    summary = df_valid.groupby(groupby_cols).agg({
        metric_col: ['mean', 'std', 'count']
    }).reset_index()
    
    # Flatten column names
    summary.columns = groupby_cols + ['mean', 'std', 'count']
    
    # Sort by mean performance (descending)
    summary = summary.sort_values('mean', ascending=False)
    
    return summary


def main():
    print("="*80)
    print("PHASE 1 UMAP HYPERPARAMETER ANALYSIS")
    print("="*80)
    print()
    
    # Check if file exists
    if not os.path.exists(RESULTS_FILE):
        print(f"ERROR: Results file not found: {RESULTS_FILE}")
        print("Please run Phase 1 analysis first to generate this file.")
        return 1
    
    # Load results
    print(f"Loading results from: {RESULTS_FILE}")
    df = pd.read_csv(RESULTS_FILE)
    print(f"  Total configs: {len(df)}")
    print()
    
    # Parse config details
    print("Parsing UMAP configurations...")
    df_parsed = parse_config_details(df)
    print(f"  UMAP configs found: {len(df_parsed)}")
    print()
    
    if df_parsed.empty:
        print("ERROR: No UMAP configurations found in results!")
        return 1
    
    # Analyze for HIGH potency enrichment
    print("-"*80)
    print("BEST UMAP HYPERPARAMETERS FOR HIGH-POTENCY ENRICHMENT")
    print("-"*80)
    
    df_high = aggregate_by_hyperparams(df_parsed, 'ef_1_high')
    
    if not df_high.empty:
        print("\nTop 5 configurations:")
        print(df_high.head(10).to_string(index=False))
        print()
        
        # Best config
        best_high = df_high.iloc[0]
        print("WINNER (High-Potency EF@1%):")
        print(f"  Representation: {best_high['representation']}")
        print(f"  Metric: {best_high['metric']}")
        print(f"  Dimension: {int(best_high['dimension'])}")
        print(f"  n_neighbors: {int(best_high['n_neighbors'])}")
        print(f"  min_dist: {best_high['min_dist']}")
        print(f"  Mean EF@1% (high): {best_high['mean']:.2f} ± {best_high['std']:.2f} (n={int(best_high['count'])})")
    else:
        print("No valid data for high-potency enrichment")
    
    print()
    print()
    
    # Analyze for OVERALL enrichment
    print("-"*80)
    print("BEST UMAP HYPERPARAMETERS FOR OVERALL ENRICHMENT")
    print("-"*80)
    
    df_overall = aggregate_by_hyperparams(df_parsed, 'ef_1_overall')
    
    if not df_overall.empty:
        print("\nTop 5 configurations:")
        print(df_overall.head(10).to_string(index=False))
        print()
        
        # Best config
        best_overall = df_overall.iloc[0]
        print("WINNER (Overall EF@1%):")
        print(f"  Representation: {best_overall['representation']}")
        print(f"  Metric: {best_overall['metric']}")
        print(f"  Dimension: {int(best_overall['dimension'])}")
        print(f"  n_neighbors: {int(best_overall['n_neighbors'])}")
        print(f"  min_dist: {best_overall['min_dist']}")
        print(f"  Mean EF@1% (overall): {best_overall['mean']:.2f} ± {best_overall['std']:.2f} (n={int(best_overall['count'])})")
    else:
        print("No valid data for overall enrichment")
    
    print()
    print("="*80)
    print("COMPARISON")
    print("="*80)
    
    if not df_high.empty and not df_overall.empty:
        # Check if winners are the same
        best_high = df_high.iloc[0]
        best_overall = df_overall.iloc[0]
        
        high_config = (best_high['representation'], best_high['metric'], 
                      int(best_high['dimension']), int(best_high['n_neighbors']), 
                      best_high['min_dist'])
        overall_config = (best_overall['representation'], best_overall['metric'], 
                         int(best_overall['dimension']), int(best_overall['n_neighbors']), 
                         best_overall['min_dist'])
        
        if high_config == overall_config:
            print("✓ SAME configuration wins for both metrics!")
        else:
            print("✗ DIFFERENT configurations win for each metric")
            print()
            print("High-potency winner:")
            print(f"  {high_config[0]}/{high_config[1]}/dim{high_config[2]}/nn{high_config[3]}/md{high_config[4]}")
            print()
            print("Overall winner:")
            print(f"  {overall_config[0]}/{overall_config[1]}/dim{overall_config[2]}/nn{overall_config[3]}/md{overall_config[4]}")
    
    print()
    print("="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
