#!/usr/bin/env python3
"""
Generate Phase 1 configuration files for UMMBAS v3.0
Phase 1: Tyro Dimensionality × Hyperparameter Sweep

Generates configs for:
- Features-PCA: 2D, 5D, 10D
- Features-UMAP-Euclidean: 2D, 5D, 10D with hyperparameters
- Fingerprints-PCA: 2D only
- Fingerprints-UMAP-Jaccard: 2D only with hyperparameters
"""

import json
import os
from itertools import product

# Configuration
OUTPUT_DIR = "hyperparam_configs_v3_phase1"
BASE_CONFIG_PATH = "experiment_config.json"
SEEDS = [42, 43, 44, 45, 46]

# UMAP hyperparameters (same for all dimensions)
N_NEIGHBORS = [10, 20, 100, 500]
MIN_DIST = [0.01, 0.1, 0.5]

# Dimensions
FEATURE_DIMS = [2, 5, 10]
FINGERPRINT_DIMS = [2]  # Only 2D for fingerprints

# Target for Phase 1
PHASE1_TARGET = {
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


def create_pca_config(representation, dimension, seed, base_config):
    """Create PCA configuration."""
    config = {
        "global_settings": base_config["global_settings"].copy(),
        "targets": [PHASE1_TARGET],
        "representations": [representation],
        "dimensionality_reduction_methods": {
            "pca": {"short_name": "PCA"}
        },
        "random_seed": seed,
        "phase": "phase1_hyperparam_sweep",
        "experiment_type": f"{representation}_pca_dim{dimension}"
    }
    
    # Override dimensions and workspace
    config["global_settings"]["simspace_dims_to_test"] = [dimension]
    config["global_settings"]["workspace_base_dir"] = "experiment_workspace_v3_phase1/"
    
    return config


def create_umap_config(representation, metric, dimension, n_neighbors, min_dist, seed, base_config):
    """Create UMAP configuration."""
    metric_name = "euclidean" if representation == "features" else "jaccard"
    short_name = f"UMAP-{'Euclidean' if metric_name == 'euclidean' else 'Jaccard'}"
    
    config = {
        "global_settings": base_config["global_settings"].copy(),
        "targets": [PHASE1_TARGET],
        "representations": [representation],
        "dimensionality_reduction_methods": {
            f"umap_{metric_name}": {
                "short_name": short_name,
                "metric": metric_name,
                "n_neighbors": n_neighbors,
                "min_dist": min_dist
            }
        },
        "random_seed": seed,
        "phase": "phase1_hyperparam_sweep",
        "experiment_type": f"{representation}_umap_{metric_name}_dim{dimension}_nn{n_neighbors}_md{min_dist}"
    }
    
    # Override dimensions and workspace
    config["global_settings"]["simspace_dims_to_test"] = [dimension]
    config["global_settings"]["workspace_base_dir"] = "experiment_workspace_v3_phase1/"
    
    return config


def main():
    """Generate all Phase 1 configuration files."""
    print("="*80)
    print("UMMBAS v3.0 - Phase 1 Config Generator")
    print("="*80)
    print()
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    
    # Load base config
    base_config = load_base_config()
    
    config_count = 0
    
    # ========================================================================
    # FEATURES - PCA (2D, 5D, 10D)
    # ========================================================================
    print("Generating Features-PCA configs...")
    for dim in FEATURE_DIMS:
        for seed in SEEDS:
            config = create_pca_config("features", dim, seed, base_config)
            filename = f"config_tyro_features_pca_dim{dim}_seed{seed}.json"
            filepath = os.path.join(OUTPUT_DIR, filename)
            
            with open(filepath, 'w') as f:
                json.dump(config, f, indent=2)
            
            config_count += 1
    
    print(f"  Created {len(FEATURE_DIMS) * len(SEEDS)} configs")
    
    # ========================================================================
    # FEATURES - UMAP-Euclidean (2D, 5D, 10D × hyperparameters)
    # ========================================================================
    print("Generating Features-UMAP-Euclidean configs...")
    for dim, nn, md, seed in product(FEATURE_DIMS, N_NEIGHBORS, MIN_DIST, SEEDS):
        config = create_umap_config("features", "euclidean", dim, nn, md, seed, base_config)
        filename = f"config_tyro_features_umap_euclidean_dim{dim}_nn{nn}_md{md}_seed{seed}.json"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        
        config_count += 1
    
    print(f"  Created {len(FEATURE_DIMS) * len(N_NEIGHBORS) * len(MIN_DIST) * len(SEEDS)} configs")
    
    # ========================================================================
    # FINGERPRINTS - PCA (2D only)
    # ========================================================================
    print("Generating Fingerprints-PCA configs...")
    for dim in FINGERPRINT_DIMS:
        for seed in SEEDS:
            config = create_pca_config("fingerprints", dim, seed, base_config)
            filename = f"config_tyro_fingerprints_pca_dim{dim}_seed{seed}.json"
            filepath = os.path.join(OUTPUT_DIR, filename)
            
            with open(filepath, 'w') as f:
                json.dump(config, f, indent=2)
            
            config_count += 1
    
    print(f"  Created {len(FINGERPRINT_DIMS) * len(SEEDS)} configs")
    
    # ========================================================================
    # FINGERPRINTS - UMAP-Jaccard (2D only × hyperparameters)
    # ========================================================================
    print("Generating Fingerprints-UMAP-Jaccard configs...")
    for dim, nn, md, seed in product(FINGERPRINT_DIMS, N_NEIGHBORS, MIN_DIST, SEEDS):
        config = create_umap_config("fingerprints", "jaccard", dim, nn, md, seed, base_config)
        filename = f"config_tyro_fingerprints_umap_jaccard_dim{dim}_nn{nn}_md{md}_seed{seed}.json"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        
        config_count += 1
    
    print(f"  Created {len(FINGERPRINT_DIMS) * len(N_NEIGHBORS) * len(MIN_DIST) * len(SEEDS)} configs")
    
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
    print(f"  Features-PCA:               {len(FEATURE_DIMS) * len(SEEDS)}")
    print(f"  Features-UMAP-Euclidean:    {len(FEATURE_DIMS) * len(N_NEIGHBORS) * len(MIN_DIST) * len(SEEDS)}")
    print(f"  Fingerprints-PCA:           {len(FINGERPRINT_DIMS) * len(SEEDS)}")
    print(f"  Fingerprints-UMAP-Jaccard:  {len(FINGERPRINT_DIMS) * len(N_NEIGHBORS) * len(MIN_DIST) * len(SEEDS)}")
    print()
    print(f"Expected total runs: {config_count}")
    print("="*80)


if __name__ == "__main__":
    main()
