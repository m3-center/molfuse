#!/usr/bin/env python3
"""
Extract best configurations from Phase 1 results for use in Phase 2-4.

Analyzes Phase 1 results and identifies:
- Best PCA-features per dimension (2D, 5D, 10D)
- Best UMAP-Euclidean-features per dimension (2D, 5D, 10D)
- Best PCA-fingerprints at 2D
- Best UMAP-Jaccard-fingerprints at 2D
- Overall best PCA (any dimension)
- Overall best UMAP (any dimension)
"""

import json
import glob
import os
import pandas as pd
from collections import defaultdict

def extract_results_from_workspace(workspace_dir):
    """Extract EF@1% results from Phase 1 workspace."""
    results = []
    
    # Find all ranking metrics files
    pattern = os.path.join(workspace_dir, "run_seed*/*/results/*/dim_*/*/*_ranking_metrics.csv")
    metrics_files = glob.glob(pattern)
    
    print(f"Found {len(metrics_files)} metrics files")
    
    for metrics_file in metrics_files:
        try:
            # Parse path to extract run info
            path_parts = metrics_file.split(os.sep)
            run_dir = path_parts[-7]  # run_seedXX_...
            target = path_parts[-6]
            representation = path_parts[-4]
            dim_folder = path_parts[-3]
            dimension = int(dim_folder.split('_')[1])
            method_folder = path_parts[-2]
            
            # Load metrics
            df = pd.read_csv(metrics_file)
            if 'ef_1%' in df.columns:
                ef_1_pct = df['ef_1%'].values[0]
            elif 'EF@1%' in df.columns:
                ef_1_pct = df['EF@1%'].values[0]
            else:
                continue
            
            # Extract seed from run_dir
            seed = None
            for part in run_dir.split('_'):
                if part.startswith('seed'):
                    seed = int(part.replace('seed', ''))
                    break
            
            # Parse method and hyperparameters from run_dir
            method_type = None
            n_neighbors = None
            min_dist = None
            
            if 'pca' in run_dir.lower():
                method_type = 'PCA'
            elif 'umap' in run_dir.lower():
                if 'euclidean' in run_dir:
                    method_type = 'UMAP-Euclidean'
                elif 'jaccard' in run_dir:
                    method_type = 'UMAP-Jaccard'
                
                # Extract hyperparameters
                parts = run_dir.split('_')
                for part in parts:
                    if part.startswith('nn'):
                        n_neighbors = int(part[2:])
                    elif part.startswith('md'):
                        min_dist = float(part[2:])
            
            results.append({
                'representation': representation,
                'method': method_type,
                'dimension': dimension,
                'n_neighbors': n_neighbors,
                'min_dist': min_dist,
                'seed': seed,
                'ef_1_pct': ef_1_pct,
                'run_dir': run_dir
            })
            
        except Exception as e:
            print(f"Error processing {metrics_file}: {e}")
            continue
    
    return pd.DataFrame(results)


def find_best_configs(df):
    """Find best configurations for each method and dimension."""
    best_configs = {}
    
    # Group by representation, method, dimension, and hyperparameters
    # Average across seeds
    grouped = df.groupby(['representation', 'method', 'dimension', 'n_neighbors', 'min_dist'])
    aggregated = grouped['ef_1_pct'].agg(['mean', 'std', 'count']).reset_index()
    aggregated.columns = ['representation', 'method', 'dimension', 'n_neighbors', 'min_dist', 
                          'ef_1_pct_mean', 'ef_1_pct_std', 'n_seeds']
    
    # ========================================================================
    # Best PCA-features per dimension
    # ========================================================================
    for dim in [2, 5, 10]:
        mask = ((aggregated['representation'] == 'features') & 
                (aggregated['method'] == 'PCA') & 
                (aggregated['dimension'] == dim))
        if mask.any():
            best = aggregated[mask].nlargest(1, 'ef_1_pct_mean').iloc[0]
            best_configs[f'features_pca_dim{dim}'] = {
                'representation': 'features',
                'method': 'PCA',
                'dimension': int(dim),
                'ef_1_pct_mean': float(best['ef_1_pct_mean']),
                'ef_1_pct_std': float(best['ef_1_pct_std']),
                'n_seeds': int(best['n_seeds'])
            }
    
    # ========================================================================
    # Best UMAP-Euclidean-features per dimension
    # ========================================================================
    for dim in [2, 5, 10]:
        mask = ((aggregated['representation'] == 'features') & 
                (aggregated['method'] == 'UMAP-Euclidean') & 
                (aggregated['dimension'] == dim))
        if mask.any():
            best = aggregated[mask].nlargest(1, 'ef_1_pct_mean').iloc[0]
            best_configs[f'features_umap_euclidean_dim{dim}'] = {
                'representation': 'features',
                'method': 'UMAP-Euclidean',
                'dimension': int(dim),
                'n_neighbors': int(best['n_neighbors']),
                'min_dist': float(best['min_dist']),
                'ef_1_pct_mean': float(best['ef_1_pct_mean']),
                'ef_1_pct_std': float(best['ef_1_pct_std']),
                'n_seeds': int(best['n_seeds'])
            }
    
    # ========================================================================
    # Best PCA-fingerprints at 2D
    # ========================================================================
    mask = ((aggregated['representation'] == 'fingerprints') & 
            (aggregated['method'] == 'PCA') & 
            (aggregated['dimension'] == 2))
    if mask.any():
        best = aggregated[mask].nlargest(1, 'ef_1_pct_mean').iloc[0]
        best_configs['fingerprints_pca_2d'] = {
            'representation': 'fingerprints',
            'method': 'PCA',
            'dimension': int(best['dimension']),
            'ef_1_pct_mean': float(best['ef_1_pct_mean']),
            'ef_1_pct_std': float(best['ef_1_pct_std']),
            'n_seeds': int(best['n_seeds'])
        }
    
    # ========================================================================
    # Best UMAP-Jaccard-fingerprints at 2D
    # ========================================================================
    mask = ((aggregated['representation'] == 'fingerprints') & 
            (aggregated['method'] == 'UMAP-Jaccard') & 
            (aggregated['dimension'] == 2))
    if mask.any():
        best = aggregated[mask].nlargest(1, 'ef_1_pct_mean').iloc[0]
        best_configs['fingerprints_umap_jaccard_2d'] = {
            'representation': 'fingerprints',
            'method': 'UMAP-Jaccard',
            'dimension': int(best['dimension']),
            'n_neighbors': int(best['n_neighbors']),
            'min_dist': float(best['min_dist']),
            'ef_1_pct_mean': float(best['ef_1_pct_mean']),
            'ef_1_pct_std': float(best['ef_1_pct_std']),
            'n_seeds': int(best['n_seeds'])
        }
    
    # ========================================================================
    # Overall best PCA (any dimension, any representation)
    # ========================================================================
    mask = (aggregated['method'] == 'PCA')
    if mask.any():
        best = aggregated[mask].nlargest(1, 'ef_1_pct_mean').iloc[0]
        best_configs['best_pca_overall'] = {
            'representation': best['representation'],
            'method': 'PCA',
            'dimension': int(best['dimension']),
            'ef_1_pct_mean': float(best['ef_1_pct_mean']),
            'ef_1_pct_std': float(best['ef_1_pct_std']),
            'n_seeds': int(best['n_seeds'])
        }
    
    # ========================================================================
    # Overall best UMAP (any dimension, any representation)
    # ========================================================================
    mask = (aggregated['method'].str.contains('UMAP'))
    if mask.any():
        best = aggregated[mask].nlargest(1, 'ef_1_pct_mean').iloc[0]
        best_configs['best_umap_overall'] = {
            'representation': best['representation'],
            'method': best['method'],
            'dimension': int(best['dimension']),
            'n_neighbors': int(best['n_neighbors']) if pd.notna(best['n_neighbors']) else None,
            'min_dist': float(best['min_dist']) if pd.notna(best['min_dist']) else None,
            'ef_1_pct_mean': float(best['ef_1_pct_mean']),
            'ef_1_pct_std': float(best['ef_1_pct_std']),
            'n_seeds': int(best['n_seeds'])
        }
    
    return best_configs


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract best configs from Phase 1')
    parser.add_argument('--workspace', required=True, help='Phase 1 workspace directory')
    parser.add_argument('--output', default='phase1_best_configs.json', 
                       help='Output JSON file')
    args = parser.parse_args()
    
    print("="*80)
    print("Extracting Best Configurations from Phase 1")
    print("="*80)
    print(f"Workspace: {args.workspace}")
    print()
    
    # Extract results
    print("Extracting results...")
    df = extract_results_from_workspace(args.workspace)
    
    if df.empty:
        print("ERROR: No results found!")
        return
    
    print(f"Found {len(df)} result entries")
    print(f"Unique methods: {df['method'].unique()}")
    print(f"Unique dimensions: {sorted(df['dimension'].unique())}")
    print()
    
    # Find best configs
    print("Finding best configurations...")
    best_configs = find_best_configs(df)
    
    # Save to file
    with open(args.output, 'w') as f:
        json.dump(best_configs, f, indent=2)
    
    print(f"\n✅ Saved best configs to: {args.output}")
    print()
    
    # Display summary
    print("="*80)
    print("BEST CONFIGURATIONS")
    print("="*80)
    for key, config in best_configs.items():
        print(f"\n{key}:")
        for k, v in config.items():
            if k == 'ef_1_pct_mean':
                print(f"  {k}: {v:.2f}")
            elif k == 'ef_1_pct_std':
                print(f"  {k}: {v:.2f}")
            else:
                print(f"  {k}: {v}")
    print()


if __name__ == "__main__":
    main()
