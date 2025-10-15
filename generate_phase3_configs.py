#!/usr/bin/env python3
"""
Generate Phase 3 configuration files for UMMBAS v3.0
Phase 3: Cross-Protein Generalization

Tests best configurations from Phase 1 on two additional proteins:
- Pyruvate Kinase M2 (same function: Transferase)
- Isocitrate Dehydrogenase NADP (different function: Oxidoreductase)

Tests 8 best configs:
- PCA-features: 2D, 5D, 10D
- UMAP-Euclidean-features: 2D, 5D, 10D
- PCA-fingerprints: 2D
- UMAP-Jaccard-fingerprints: 2D
"""

import json
import os

# Configuration
OUTPUT_DIR = "hyperparam_configs_v3_phase3_generalization"
BASE_CONFIG_PATH = "experiment_config.json"
BEST_CONFIGS_PATH = "phase1_best_configs.json"
SEEDS = [42, 43, 44, 45, 46]

# Generalization targets
GENERALIZATION_TARGETS = [
    {
        "id_name": "PyruvateKinaseM2_P14618",
        "display_name": "Pyruvate Kinase M2",
        "uniprot_id": "P14618",
        "molecular_function_canonical_name": "Transferase",
        "molecular_function_filename_segment": "Transferase",
        "molecular_function_display_name": "Transferase",
        "molecular_function_kw_code": "KW-0808"
    },
    {
        "id_name": "IsocitrateDehydrogenaseNADP_O75874",
        "display_name": "Isocitrate Dehydrogenase NADP cytoplasmic",
        "uniprot_id": "O75874",
        "molecular_function_canonical_name": "Oxidoreductase",
        "molecular_function_filename_segment": "Oxidoreductase",
        "molecular_function_display_name": "Oxidoreductase",
        "molecular_function_kw_code": "KW-0560"
    }
]


def load_base_config():
    """Load base experiment configuration."""
    with open(BASE_CONFIG_PATH, 'r') as f:
        return json.load(f)


def load_best_configs():
    """Load best configurations from Phase 1."""
    with open(BEST_CONFIGS_PATH, 'r') as f:
        return json.load(f)


def create_generalization_config(method_config, target, seed, base_config):
    """Create generalization configuration."""
    
    representation = method_config['representation']
    method = method_config['method']
    dimension = method_config['dimension']
    
    config = {
        "global_settings": base_config["global_settings"].copy(),
        "targets": [target],
        "representations": [representation],
        "random_seed": seed,
        "phase": "phase3_generalization",
        "experiment_type": f"{target['id_name']}_{representation}_{method.lower().replace('-', '_')}_dim{dimension}"
    }
    
    # Override dimensions
    config["global_settings"]["simspace_dims_to_test"] = [dimension]
    
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
    elif method == 'UMAP-Jaccard':
        config["dimensionality_reduction_methods"] = {
            "umap_jaccard": {
                "short_name": "UMAP-Jaccard",
                "metric": "jaccard",
                "n_neighbors": method_config['n_neighbors'],
                "min_dist": method_config['min_dist']
            }
        }
    
    return config


def main():
    """Generate all Phase 3 configuration files."""
    print("="*80)
    print("UMMBAS v3.0 - Phase 3 Generalization Config Generator")
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
    print()
    
    # Select configs to test
    configs_to_test = []
    
    # PCA-features at each dimension
    for dim in [2, 5, 10]:
        key = f'features_pca_dim{dim}'
        if key in best_configs:
            configs_to_test.append((key, best_configs[key]))
            print(f"✓ {key}: EF@1% = {best_configs[key]['ef_1_pct_mean']:.2f}")
    
    # UMAP-Euclidean-features at each dimension
    for dim in [2, 5, 10]:
        key = f'features_umap_euclidean_dim{dim}'
        if key in best_configs:
            cfg = best_configs[key]
            configs_to_test.append((key, cfg))
            print(f"✓ {key}: EF@1% = {cfg['ef_1_pct_mean']:.2f} (nn={cfg['n_neighbors']}, md={cfg['min_dist']})")
    
    # PCA-fingerprints at 2D
    if 'fingerprints_pca_dim2' in best_configs:
        configs_to_test.append(('fingerprints_pca_dim2', best_configs['fingerprints_pca_dim2']))
        print(f"✓ fingerprints_pca_dim2: EF@1% = {best_configs['fingerprints_pca_dim2']['ef_1_pct_mean']:.2f}")
    
    # UMAP-Jaccard-fingerprints at 2D
    if 'fingerprints_umap_jaccard_dim2' in best_configs:
        cfg = best_configs['fingerprints_umap_jaccard_dim2']
        configs_to_test.append(('fingerprints_umap_jaccard_dim2', cfg))
        print(f"✓ fingerprints_umap_jaccard_dim2: EF@1% = {cfg['ef_1_pct_mean']:.2f} (nn={cfg['n_neighbors']}, md={cfg['min_dist']})")
    
    print()
    print(f"Total configurations to test: {len(configs_to_test)}")
    print(f"Generalization targets: {len(GENERALIZATION_TARGETS)}")
    print()
    
    for target in GENERALIZATION_TARGETS:
        print(f"  - {target['display_name']} ({target['molecular_function_display_name']})")
    print()
    
    config_count = 0
    
    # ========================================================================
    # Generate configs for each target and method combination
    # ========================================================================
    for target in GENERALIZATION_TARGETS:
        target_short = target['id_name'].split('_')[0].lower()
        
        print(f"Generating configs for {target['display_name']}...")
        
        for config_key, method_config in configs_to_test:
            representation = method_config['representation']
            method = method_config['method']
            dimension = method_config['dimension']
            
            for seed in SEEDS:
                config = create_generalization_config(method_config, target, seed, base_config)
                
                # Build filename
                method_str = method.lower().replace('-', '_')
                filename = f"config_{target_short}_{representation}_{method_str}_dim{dimension}"
                
                if 'UMAP' in method:
                    filename += f"_nn{method_config['n_neighbors']}_md{method_config['min_dist']}"
                
                filename += f"_seed{seed}.json"
                filepath = os.path.join(OUTPUT_DIR, filename)
                
                with open(filepath, 'w') as f:
                    json.dump(config, f, indent=2)
                
                config_count += 1
        
        print(f"  Created {len(configs_to_test) * len(SEEDS)} configs")
    
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
    print(f"  Targets: {len(GENERALIZATION_TARGETS)}")
    print(f"  Methods: {len(configs_to_test)}")
    print(f"  Seeds: {len(SEEDS)}")
    print(f"  Total: {len(GENERALIZATION_TARGETS)} × {len(configs_to_test)} × {len(SEEDS)} = {config_count}")
    print()
    print("This will test:")
    print("  - Same molecular function generalization (Tyro → Pyru: both Transferase)")
    print("  - Different molecular function generalization (Tyro → Iso: Oxidoreductase)")
    print("  - Dimension transferability across proteins")
    print("="*80)


if __name__ == "__main__":
    main()
