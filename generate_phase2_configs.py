#!/usr/bin/env python3
"""
Generate Phase 2 configuration files for UMMBAS v3.0
Phase 2: MF Cloud Ablation Study

Tests PCA-features and UMAP-features at their best dimensionalities
with varying MF cloud sizes: 0, 1K, 10K, 50K, 100K, 420K molecules

This validates the phase transition hypothesis showing PCA overtaking
UMAP at critical MF cloud mass.
"""

import json
import os

# Configuration
OUTPUT_DIR = "hyperparam_configs_v3_phase2_ablation"
BASE_CONFIG_PATH = "experiment_config.json"
BEST_CONFIGS_PATH = "phase1_best_configs.json"
SEEDS = [42, 43, 44, 45, 46]

# MF cloud sizes for ablation
MF_CLOUD_SIZES = [0, 1000, 10000, 50000, 100000, 420000]

# Target for Phase 2
PHASE2_TARGET = {
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


def create_ablation_config(method_config, mf_size, seed, base_config):
    """Create ablation configuration for a specific MF cloud size."""
    
    representation = method_config['representation']
    method = method_config['method']
    dimension = method_config['dimension']
    
    config = {
        "global_settings": base_config["global_settings"].copy(),
        "targets": [PHASE2_TARGET],
        "representations": [representation],
        "random_seed": seed,
        "phase": "phase2_mf_ablation",
        "mf_cloud_size": mf_size,
        "experiment_type": f"{representation}_{method.lower().replace('-', '_')}_dim{dimension}_mf{mf_size}"
    }
    
    # Override dimensions
    config["global_settings"]["simspace_dims_to_test"] = [dimension]
    
    # Override MF cloud size
    config["global_settings"]["mf_cloud_max_molecules"] = mf_size
    
    # Set DR method
    if method == 'PCA':
        config["dimensionality_reduction_methods"] = {
            "pca": {"short_name": "PCA"}
        }
    elif method == 'UMAP-Euclidean':
        config["dimensionality_reduction_methods"] = {
            "umap_euclidean": {
                "short_name": "UMAP-Euclidean",
                "metric": "euclidean",
                "n_neighbors": method_config['n_neighbors'],
                "min_dist": method_config['min_dist']
            }
        }
    
    return config


def main():
    """Generate all Phase 2 configuration files."""
    print("="*80)
    print("UMMBAS v3.0 - Phase 2 MF Cloud Ablation Config Generator")
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
    
    # Extract best PCA-features (at best dimension from Phase 1)
    best_pca = None
    for key in ['features_pca_dim2', 'features_pca_dim5', 'features_pca_dim10']:
        if key in best_configs:
            if best_pca is None or best_configs[key]['ef_1_pct_mean'] > best_pca['ef_1_pct_mean']:
                best_pca = best_configs[key]
    
    # Extract best UMAP-features (at best dimension from Phase 1)
    best_umap = None
    for key in ['features_umap_euclidean_dim2', 'features_umap_euclidean_dim5', 'features_umap_euclidean_dim10']:
        if key in best_configs:
            if best_umap is None or best_configs[key]['ef_1_pct_mean'] > best_umap['ef_1_pct_mean']:
                best_umap = best_configs[key]
    
    if not best_pca or not best_umap:
        print("ERROR: Could not find best PCA or UMAP configs from Phase 1!")
        print("Available keys:", list(best_configs.keys()))
        return
    
    print()
    print("Best configurations for ablation:")
    print(f"  PCA-features: {best_pca['dimension']}D, EF@1% = {best_pca['ef_1_pct_mean']:.2f}")
    print(f"  UMAP-features: {best_umap['dimension']}D (nn={best_umap['n_neighbors']}, md={best_umap['min_dist']}), EF@1% = {best_umap['ef_1_pct_mean']:.2f}")
    print()
    
    config_count = 0
    
    # ========================================================================
    # Generate configs for each MF cloud size
    # ========================================================================
    print(f"Generating configs for MF cloud sizes: {MF_CLOUD_SIZES}")
    print()
    
    for mf_size in MF_CLOUD_SIZES:
        mf_label = f"{mf_size//1000}k" if mf_size >= 1000 else str(mf_size)
        
        # PCA configs
        for seed in SEEDS:
            config = create_ablation_config(best_pca, mf_size, seed, base_config)
            filename = f"config_tyro_features_pca_dim{best_pca['dimension']}_mf{mf_label}_seed{seed}.json"
            filepath = os.path.join(OUTPUT_DIR, filename)
            
            with open(filepath, 'w') as f:
                json.dump(config, f, indent=2)
            
            config_count += 1
        
        # UMAP configs
        for seed in SEEDS:
            config = create_ablation_config(best_umap, mf_size, seed, base_config)
            filename = f"config_tyro_features_umap_dim{best_umap['dimension']}_nn{best_umap['n_neighbors']}_md{best_umap['min_dist']}_mf{mf_label}_seed{seed}.json"
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
    print(f"  MF cloud sizes: {len(MF_CLOUD_SIZES)}")
    print(f"  Methods: 2 (PCA, UMAP)")
    print(f"  Seeds: {len(SEEDS)}")
    print(f"  Total: {len(MF_CLOUD_SIZES)} × 2 × {len(SEEDS)} = {config_count}")
    print()
    print("This will test the phase transition hypothesis:")
    print("  - Expected PCA advantage at high MF counts (50K-420K)")
    print("  - Expected UMAP advantage at low MF counts (0-10K)")
    print("  - Crossover point predicted at ~10K-50K molecules")
    print("="*80)


if __name__ == "__main__":
    main()
