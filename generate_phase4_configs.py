#!/usr/bin/env python3
"""
Generate Phase 4 configuration files for UMMBAS v3.0
Phase 4: Affinity Cutoff Analysis

Tests best overall PCA and best overall UMAP configurations
at their optimal dimensions with varying affinity cutoffs:
100,000 / 10,000 / 1,000 / 100 nM

Only on Tyrosine kinase (reference target).
"""

import json
import os

# Configuration
OUTPUT_DIR = "hyperparam_configs_v3_phase4_cutoff"
BASE_CONFIG_PATH = "experiment_config.json"
BEST_CONFIGS_PATH = "phase1_best_configs.json"
SEEDS = [42, 43, 44, 45, 46]

# Affinity cutoffs to test (in nM)
AFFINITY_CUTOFFS = [100000, 10000, 1000, 100]

# Target for Phase 4
PHASE4_TARGET = {
    "id_name": "TyrosineProteinKinaseABL1_P00519",
    "display_name": "Tyrosine-protein Kinase ABL1",
    "uniprot_id": "P00519",
    "molecular_function_canonical_name": "Transferase",
    "molecular_function_filename_segment": "Transferase",
    "molecular_function_display_name": "Transferase",
    "molecular_function_kw_code": "KW-0808"
}


def load_base_config():
    """Load base experiment configuration."""
    with open(BASE_CONFIG_PATH, 'r') as f:
        return json.load(f)


def load_best_configs():
    """Load best configurations from Phase 1."""
    with open(BEST_CONFIGS_PATH, 'r') as f:
        return json.load(f)


def create_cutoff_config(method_config, cutoff_nM, seed, base_config):
    """Create cutoff analysis configuration."""
    
    representation = method_config['representation']
    method = method_config['method']
    dimension = method_config['dimension']
    
    config = {
        "global_settings": base_config["global_settings"].copy(),
        "targets": [PHASE4_TARGET],
        "representations": [representation],
        "random_seed": seed,
        "phase": "phase4_cutoff_analysis",
        "affinity_cutoff_nM": cutoff_nM,
        "experiment_type": f"{representation}_{method.lower().replace('-', '_')}_dim{dimension}_cutoff{cutoff_nM}nM"
    }
    
    # Override dimensions, workspace, and affinity cutoff
    config["global_settings"]["simspace_dims_to_test"] = [dimension]
    config["global_settings"]["workspace_base_dir"] = "experiment_workspace_v3_phase4/"
    config["global_settings"]["affinity_cutoff_nM"] = cutoff_nM
    
    # Set DR method
    if method == 'PCA':
        config["dimensionality_reduction_methods"] = {
            "pca": {"short_name": "PCA"}
        }
    elif 'UMAP' in method:
        metric = 'euclidean' if 'Euclidean' in method else 'jaccard'
        config["dimensionality_reduction_methods"] = {
            f"umap_{metric}": {
                "short_name": method,
                "metric": metric,
                "n_neighbors": method_config['n_neighbors'],
                "min_dist": method_config['min_dist']
            }
        }
    
    return config


def main():
    """Generate all Phase 4 configuration files."""
    print("="*80)
    print("UMMBAS v3.0 - Phase 4 Affinity Cutoff Analysis Config Generator")
    print("="*80)
    print()
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    
    # Load configs
    base_config = load_base_config()
    
    print(f"Loading best configurations from: {BEST_CONFIGS_PATH}")
    best_configs = load_best_configs()
    
    # Extract overall best PCA and UMAP
    best_pca = best_configs.get('best_pca_overall')
    best_umap = best_configs.get('best_umap_overall')
    
    if not best_pca or not best_umap:
        print("ERROR: Could not find best overall PCA or UMAP configs from Phase 1!")
        print("Available keys:", list(best_configs.keys()))
        return
    
    print()
    print("Best overall configurations for cutoff analysis:")
    print(f"  Best PCA: {best_pca['representation']}-{best_pca['method']}-{best_pca['dimension']}D")
    print(f"    EF@1% = {best_pca['ef_1_pct_mean']:.2f} ± {best_pca['ef_1_pct_std']:.2f}")
    print()
    print(f"  Best UMAP: {best_umap['representation']}-{best_umap['method']}-{best_umap['dimension']}D")
    print(f"    nn={best_umap['n_neighbors']}, md={best_umap['min_dist']}")
    print(f"    EF@1% = {best_umap['ef_1_pct_mean']:.2f} ± {best_umap['ef_1_pct_std']:.2f}")
    print()
    
    config_count = 0
    
    # ========================================================================
    # Generate configs for each cutoff
    # ========================================================================
    print(f"Generating configs for affinity cutoffs: {AFFINITY_CUTOFFS} nM")
    print()
    
    for cutoff in AFFINITY_CUTOFFS:
        cutoff_label = f"{cutoff//1000}k" if cutoff >= 1000 else str(cutoff)
        
        # Best PCA configs
        for seed in SEEDS:
            config = create_cutoff_config(best_pca, cutoff, seed, base_config)
            
            filename = (f"config_tyro_{best_pca['representation']}_pca_"
                       f"dim{best_pca['dimension']}_cutoff{cutoff_label}nM_seed{seed}.json")
            filepath = os.path.join(OUTPUT_DIR, filename)
            
            with open(filepath, 'w') as f:
                json.dump(config, f, indent=2)
            
            config_count += 1
        
        # Best UMAP configs
        for seed in SEEDS:
            config = create_cutoff_config(best_umap, cutoff, seed, base_config)
            
            method_str = best_umap['method'].lower().replace('-', '_')
            filename = (f"config_tyro_{best_umap['representation']}_{method_str}_"
                       f"dim{best_umap['dimension']}_nn{best_umap['n_neighbors']}_"
                       f"md{best_umap['min_dist']}_cutoff{cutoff_label}nM_seed{seed}.json")
            filepath = os.path.join(OUTPUT_DIR, filename)
            
            with open(filepath, 'w') as f:
                json.dump(config, f, indent=2)
            
            config_count += 1
    
    # ========================================================================
    # Summary
    # ========================================================================
    print()
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total configs generated: {config_count}")
    print()
    print("Breakdown:")
    print(f"  Cutoffs: {len(AFFINITY_CUTOFFS)}")
    print(f"  Methods: 2 (best PCA, best UMAP)")
    print(f"  Seeds: {len(SEEDS)}")
    print(f"  Total: {len(AFFINITY_CUTOFFS)} × 2 × {len(SEEDS)} = {config_count}")
    print()
    print("Cutoff sensitivity analysis:")
    print("  100,000 nM = 100 µM (very permissive)")
    print("   10,000 nM =  10 µM (moderate)")
    print("    1,000 nM =   1 µM (stringent)")
    print("      100 nM = 100 nM (very stringent)")
    print()
    print("This will reveal optimal cutoff threshold per method.")
    print("="*80)


if __name__ == "__main__":
    main()
