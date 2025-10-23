#!/usr/bin/env python3
"""
Generate Phase 2 configuration files for UMMBAS v3.0
Phase 2: Affinity Cutoff Sensitivity Analysis

Tests different affinity cutoffs to determine optimal threshold for classifying
molecules as "active" in the MF cloud. This is critical because:
- Cutoff determines which molecules contribute to MF cloud diversity
- Affects downstream Phase 3 (MF cloud ablation) and Phase 4 (generalization)
- Must precede Phase 3 logically (cutoff affects MF cloud composition)

**Potency Tiers (from stratified enrichment analysis):**
- High Potent: 0.1-100 nM (drug-like, clinically relevant)
- Medium Potent: 100-1,000 nM (moderate affinity)
- Weak Potent: 1,000-100,000 nM (marginal, likely promiscuous)

**Cutoffs to Test:**
- 100 nM: Only high-potent compounds (strictest, highest quality)
- 1,000 nM (1 μM): High + medium potent (balanced quality/quantity)
- 10,000 nM (10 μM): High + medium + some weak (permissive)
- 100,000 nM (100 μM): All potencies (most permissive, maximum diversity)

**Computational Strategy:**
Phase 2 REUSES Phase 1 similarity spaces and DR models to save time:
- Does NOT recalculate features/fingerprints
- Does NOT refit PCA/UMAP models
- ONLY reruns ranking/evaluation with different affinity cutoffs
- Expected runtime: ~10-15 min per run (vs hours for full pipeline)

**Phase 2 Specific Selection:**
This phase tests only 3 specific configurations:
1. PCA/features/5D (best overall PCA)
2. UMAP/features/5D with overall-EF-optimized hyperparameters (quantity-focused)
3. UMAP/features/5D with high-potency-EF-optimized hyperparameters (quality-focused)

Total: 4 cutoffs × 3 configs × 5 seeds = 60 runs
Estimated cost: ~10-15 hours total (vs ~60+ hours if recalculating similarity spaces)
"""

import json
import os

# Configuration
OUTPUT_DIR = "hyperparam_configs_v3_phase2_cutoff"
BASE_CONFIG_PATH = "experiment_config.json"
BEST_CONFIGS_OVERALL_PATH = "phase1_best_configs_by_overall_ef.json"
BEST_CONFIGS_HIGH_POTENCY_PATH = "phase1_best_configs_by_high_potency_ef.json"
SEEDS = [42, 43, 44, 45, 46]  # Same seeds as Phase 1

# Affinity cutoffs (nM) aligned with potency tiers
AFFINITY_CUTOFFS = [
    100,      # High-potent only
    1000,     # High + Medium
    10000,    # High + Medium + some Weak
    100000    # All (default, most permissive)
]

# Target for Phase 2 (same as Phase 1)
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
    """Load best configurations from Phase 1 (both overall and high-potency)."""
    configs = {}
    
    # Load overall EF best configs
    if not os.path.exists(BEST_CONFIGS_OVERALL_PATH):
        raise FileNotFoundError(
            f"Phase 1 overall best configs not found: {BEST_CONFIGS_OVERALL_PATH}\n"
            f"Please run: python scripts/analyze_potency_stratified_enrichment.py "
            f"--workspace_dir experiment_workspace_v3_phase1 --output_dir potency_analysis_results"
        )
    with open(BEST_CONFIGS_OVERALL_PATH, 'r') as f:
        configs['overall'] = json.load(f)
    
    # Load high-potency EF best configs
    if not os.path.exists(BEST_CONFIGS_HIGH_POTENCY_PATH):
        raise FileNotFoundError(
            f"Phase 1 high-potency best configs not found: {BEST_CONFIGS_HIGH_POTENCY_PATH}\n"
            f"Please run: python scripts/analyze_potency_stratified_enrichment.py "
            f"--workspace_dir experiment_workspace_v3_phase1 --output_dir potency_analysis_results"
        )
    with open(BEST_CONFIGS_HIGH_POTENCY_PATH, 'r') as f:
        configs['high_potency'] = json.load(f)
    
    return configs


def create_cutoff_config(method_config, cutoff_nm, seed, base_config, config_label):
    """Create cutoff analysis configuration.
    
    Args:
        method_config: Best config from Phase 1 (dict with representation, method, dimension, etc.)
        cutoff_nm: Affinity cutoff in nM
        seed: Random seed
        base_config: Base experiment configuration
        config_label: Label for this config (e.g., 'overall', 'high_potency')
    
    Returns:
        Configuration dict for this cutoff analysis run
    """
    
    representation = method_config['representation']
    method = method_config['method']
    dimension = method_config['dimension']
    
    config = {
        "global_settings": base_config["global_settings"].copy(),
        "targets": [PHASE2_TARGET],
        "representations": [representation],
        "random_seed": seed,
        "phase": "phase2_cutoff_sensitivity",
        "affinity_cutoff_nM": cutoff_nm,
        "reuse_phase1_data": True,  # Flag to indicate data reuse strategy
        "phase1_workspace": "experiment_workspace_v3_phase1/",
        "selection_criterion": config_label,  # Track which criterion was used
        "experiment_type": f"{representation}_{method.lower().replace('-', '_')}_dim{dimension}_cutoff{cutoff_nm}nM_{config_label}"
    }
    
    # Override settings for Phase 2
    config["global_settings"]["simspace_dims_to_test"] = [dimension]
    config["global_settings"]["workspace_base_dir"] = "experiment_workspace_v3_phase2/"
    config["global_settings"]["affinity_cutoff_nM"] = cutoff_nm
    
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
    print("UMMBAS v3.0 - Phase 2 Cutoff Sensitivity Config Generator")
    print("="*80)
    print()
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    
    # Load base config and best Phase 1 configs
    base_config = load_base_config()
    
    try:
        best_configs = load_best_configs()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("\nPlease run potency stratified enrichment analysis first:")
        print("  python scripts/analyze_potency_stratified_enrichment.py \\")
        print("    --workspace_dir experiment_workspace_v3_phase1 \\")
        print("    --output_dir potency_analysis_results")
        return
    
    # Select specific configurations for Phase 2 cutoff analysis
    methods_to_test = []
    
    # 1. PCA-features 5D (from overall best configs)
    if 'features_pca_dim5' in best_configs['overall']:
        pca_config = best_configs['overall']['features_pca_dim5']
        methods_to_test.append(('PCA-features-5D', pca_config, 'pca_overall'))
        print(f"✓ PCA-features-5D: EF@1%={pca_config.get('ef_1_pct_mean', 'N/A'):.2f}")
    else:
        print("WARNING: PCA-features-5D not found in overall configs!")
    
    # 2. UMAP-features 5D (overall EF optimized - quantity-focused)
    if 'features_umap_euclidean_dim5' in best_configs['overall']:
        umap_overall = best_configs['overall']['features_umap_euclidean_dim5']
        methods_to_test.append(('UMAP-features-5D (overall-EF)', umap_overall, 'umap_overall'))
        print(f"✓ UMAP-features-5D (overall-EF optimized):")
        print(f"    nn={umap_overall['n_neighbors']}, md={umap_overall['min_dist']}")
        print(f"    Overall EF@1%={umap_overall.get('ef_1_pct_mean', 'N/A'):.2f}")
    else:
        print("WARNING: UMAP-features-5D not found in overall configs!")
    
    # 3. UMAP-features 5D (high-potency EF optimized - quality-focused)
    if 'features_umap_euclidean_dim5' in best_configs['high_potency']:
        umap_high = best_configs['high_potency']['features_umap_euclidean_dim5']
        methods_to_test.append(('UMAP-features-5D (high-potency-EF)', umap_high, 'umap_high_potency'))
        print(f"✓ UMAP-features-5D (high-potency-EF optimized):")
        print(f"    nn={umap_high['n_neighbors']}, md={umap_high['min_dist']}")
        print(f"    High-potency EF@1%={umap_high.get('high_ef_1_pct_mean', 'N/A'):.2f}")
    else:
        print("WARNING: UMAP-features-5D not found in high-potency configs!")
    
    if not methods_to_test:
        print("ERROR: No suitable 5D configurations found in best_configs!")
        print("Available overall keys:", list(best_configs['overall'].keys()))
        print("Available high-potency keys:", list(best_configs['high_potency'].keys()))
        return
    
    print()
    print(f"Affinity cutoffs to test: {AFFINITY_CUTOFFS} nM")
    print(f"Seeds: {SEEDS} (same as Phase 1)")
    print()
    
    # Generate configurations
    config_count = 0
    for method_label, method_config, config_type in methods_to_test:
        for cutoff_nm in AFFINITY_CUTOFFS:
            for seed in SEEDS:
                config = create_cutoff_config(method_config, cutoff_nm, seed, base_config, config_type)
                
                # Generate filename
                method_name = method_config['method'].lower().replace('-', '_')
                dim = method_config['dimension']
                
                # Include hyperparams in filename for UMAP to distinguish configs
                if 'n_neighbors' in method_config:
                    hyperparam_str = f"_nn{method_config['n_neighbors']}_md{method_config['min_dist']}"
                else:
                    hyperparam_str = ""
                
                filename = f"config_seed{seed}_{method_config['representation']}_{method_name}_dim{dim}{hyperparam_str}_cutoff{cutoff_nm}nM_{config_type}.json"
                filepath = os.path.join(OUTPUT_DIR, filename)
                
                # Save configuration
                with open(filepath, 'w') as f:
                    json.dump(config, f, indent=2)
                
                config_count += 1
    
    print(f"Generated {config_count} configuration files")
    print()
    
    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total configurations: {config_count}")
    print(f"  Configs: {len(methods_to_test)} (PCA-5D + 2×UMAP-5D)")
    print(f"  Cutoffs: {len(AFFINITY_CUTOFFS)}")
    print(f"  Seeds: {len(SEEDS)} (same as Phase 1)")
    print(f"  Expected runs: {len(methods_to_test)} × {len(AFFINITY_CUTOFFS)} × {len(SEEDS)} = {len(methods_to_test) * len(AFFINITY_CUTOFFS) * len(SEEDS)}")
    print()
    print("Configurations:")
    print("  1. PCA/features/5D (overall best PCA)")
    print("  2. UMAP/features/5D - overall-EF optimized (quantity-focused)")
    print("  3. UMAP/features/5D - high-potency-EF optimized (quality-focused)")
    print()
    print("Data Reuse Strategy:")
    print("  ✓ Reuses Phase 1 similarity spaces (features/fingerprints)")
    print("  ✓ Reuses Phase 1 DR models (PCA/UMAP fitted models)")
    print("  ✓ Same seeds as Phase 1 (enables exact model matching)")
    print("  ✗ Only reruns ranking with different affinity cutoffs")
    print()
    print("Estimated runtime: ~10-15 hours total (reusing Phase 1 data)")
    print()
    print("Research Questions:")
    print("  1. Which cutoff gives best High-potent EF@1%?")
    print("  2. Do quantity-optimized vs quality-optimized UMAP params prefer different cutoffs?")
    print("  3. How does cutoff affect Medium vs Weak potency enrichment?")
    print("  4. What is optimal cutoff for Phase 3 (MF cloud ablation)?")
    print()
    print("Next steps:")
    print("  1. Review generated configs in:", OUTPUT_DIR)
    print("  2. Run Phase 2: python scripts/run_phase2_cutoff_analysis.py")
    print("  3. Or submit to HPC: sbatch hpc/submit_v3_phase2.sh")
    print("="*80)


if __name__ == "__main__":
    main()
