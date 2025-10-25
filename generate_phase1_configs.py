#!/usr/bin/env python3
"""
Generate Phase 1 configuration files for UMMBAS v3.0
Phase 1: Tyro Dimensionality × Hyperparameter Sweep (RERUN with Deduplicated Data)

Generates configs for:
- Features-PCA: 2D, 5D, 10D
- Features-UMAP-Euclidean: 2D, 5D, 10D with REVISED hyperparameters
  - n_neighbors: [10, 20, 50, 100, 500] (DROP 3, 5 due to 72hr timeout)
  - min_dist: [0.0, 0.001, 0.005, 0.01, 0.1]
  - NO FIXED SEED: Enable UMAP multi-threading for 10× speedup
- Fingerprints: UNCHANGED (not affected by MF cloud duplicates)

CRITICAL CHANGES FROM ORIGINAL:
1. Removed nn=3, 5 (computationally infeasible - 72hr timeout)
2. Added nn=50, 100, 500 (hypothesis: optimal nn shifted upward after deduplication)
3. Disabled fixed seed to enable UMAP multi-threading (10× faster)
4. Focus on medium-large nn range (10-500) where true optimal likely is

RATIONALE:
- Original Phase 1 trained on 2.23× duplicated MF cloud
- Comparison test: nn=10 dropped 58.3% (43.78 → 18.27) with clean data
- nn=500 was robust (19.57) despite duplicates → validates testing large nn
- Small nn (3, 5) exploited duplicate clusters (artifact, not real signal)
"""

import json
import os
import glob
from itertools import product

# Configuration
OUTPUT_DIR = "hyperparam_configs_v3_phase1_rerun"
BASE_CONFIG_PATH = "experiment_config.json"
WORKSPACE_DIR = "experiment_workspace_v3_phase1"
SEEDS = [42, 43, 44, 45, 46]

# REVISED UMAP hyperparameters for FEATURES (post-deduplication analysis)
# Removed nn=3, 5 (72hr timeout), Added nn=50, 100, 500 (hypothesis-driven)
FEATURES_N_NEIGHBORS = [10, 20, 50, 100, 500]
FEATURES_MIN_DIST = [0.0, 0.001, 0.005, 0.01, 0.1]

# DISABLE FIXED SEED for 10× speedup via multi-threading
USE_FIXED_SEED = False  # Enable UMAP multi-threading (race conditions acceptable)

# ORIGINAL UMAP hyperparameters for FINGERPRINTS (keep unchanged - still running)
FINGERPRINTS_N_NEIGHBORS = [20, 50, 100]
FINGERPRINTS_MIN_DIST = [0.0, 0.001, 0.01, 0.1]

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


def check_experiment_completed(representation, dr_method, dimension, seed, n_neighbors=None, min_dist=None):
    """
    Check if experiment already completed by looking for ranking CSV file.
    
    Returns:
        bool: True if experiment completed (ranking file exists), False otherwise
    """
    if not os.path.exists(WORKSPACE_DIR):
        return False
    
    # Build expected run directory name pattern
    if dr_method.lower() == 'pca':
        run_pattern = f"run_seed{seed}_*tyro*{representation}*pca*dim{dimension}*"
    else:  # UMAP
        metric = "euclidean" if representation == "features" else "jaccard"
        run_pattern = f"run_seed{seed}_*tyro*{representation}*umap*{metric}*dim{dimension}*nn{n_neighbors}*md{min_dist}*"
    
    # Find matching run directories
    run_dirs = glob.glob(os.path.join(WORKSPACE_DIR, run_pattern))
    
    if not run_dirs:
        return False
    
    # Check for ranking CSV in results directory
    # Pattern: TARGET/results/REPR/dim_N/METHOD/*-RANKED.csv
    for run_dir in run_dirs:
        ranking_pattern = os.path.join(run_dir, "*/results/*/dim_*/*/*-RANKED.csv")
        ranking_files = glob.glob(ranking_pattern)
        
        if ranking_files:
            return True  # Found completed experiment
    
    return False


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
    print("UMMBAS v3.0 - Phase 1 RERUN Config Generator (Deduplicated Data)")
    print("="*80)
    print()
    print("CRITICAL CHANGES FROM ORIGINAL:")
    print("  1. MF cloud now DEDUPLICATED (2.23× → 1.0× for ABL1)")
    print("  2. nn=3, 5 REMOVED (72-hour timeout - computationally infeasible)")
    print("  3. nn=50, 100, 500 ADDED (hypothesis: optimal shifted upward)")
    print("  4. Fixed seed DISABLED (enable multi-threading for 10× speedup)")
    print()
    print("FEATURES (REVISED):")
    print(f"  n_neighbors: {FEATURES_N_NEIGHBORS}")
    print(f"  min_dist:    {FEATURES_MIN_DIST}")
    print(f"  Fixed seed:  {USE_FIXED_SEED} (multi-threaded UMAP)")
    print()
    print("FINGERPRINTS: UNCHANGED (original hyperparameters)")
    print(f"  n_neighbors: {FINGERPRINTS_N_NEIGHBORS}")
    print(f"  min_dist:    {FINGERPRINTS_MIN_DIST}")
    print()
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Workspace directory (for skip check): {WORKSPACE_DIR}")
    print()
    
    # Load base config
    base_config = load_base_config()
    
    config_count = 0
    skipped_count = 0
    
    # ========================================================================
    # FEATURES - PCA (2D, 5D, 10D)
    # ========================================================================
    print("Generating Features-PCA configs...")
    features_pca_created = 0
    features_pca_skipped = 0
    
    for dim in FEATURE_DIMS:
        for seed in SEEDS:
            # Check if already completed
            if check_experiment_completed("features", "pca", dim, seed):
                features_pca_skipped += 1
                skipped_count += 1
                continue
            
            config = create_pca_config("features", dim, seed, base_config)
            filename = f"config_tyro_features_pca_dim{dim}_seed{seed}.json"
            filepath = os.path.join(OUTPUT_DIR, filename)
            
            with open(filepath, 'w') as f:
                json.dump(config, f, indent=2)
            
            config_count += 1
            features_pca_created += 1
    
    print(f"  Created: {features_pca_created}, Skipped (completed): {features_pca_skipped}")
    
    # ========================================================================
    # FEATURES - UMAP-Euclidean (2D, 5D, 10D × REFINED hyperparameters)
    # ========================================================================
    print("Generating Features-UMAP-Euclidean configs (REFINED)...")
    features_umap_created = 0
    features_umap_skipped = 0
    
    for dim, nn, md, seed in product(FEATURE_DIMS, FEATURES_N_NEIGHBORS, FEATURES_MIN_DIST, SEEDS):
        # Check if already completed
        if check_experiment_completed("features", "umap", dim, seed, nn, md):
            features_umap_skipped += 1
            skipped_count += 1
            continue
        
        config = create_umap_config("features", "euclidean", dim, nn, md, seed, base_config)
        filename = f"config_tyro_features_umap_euclidean_dim{dim}_nn{nn}_md{md}_seed{seed}.json"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        
        config_count += 1
        features_umap_created += 1
    
    print(f"  Created: {features_umap_created}, Skipped (completed): {features_umap_skipped}")
    
    # ========================================================================
    # FINGERPRINTS - PCA (2D only) - KEEP UNCHANGED
    # ========================================================================
    print("Generating Fingerprints-PCA configs (UNCHANGED)...")
    fingerprints_pca_created = 0
    fingerprints_pca_skipped = 0
    
    for dim in FINGERPRINT_DIMS:
        for seed in SEEDS:
            # Check if already completed
            if check_experiment_completed("fingerprints", "pca", dim, seed):
                fingerprints_pca_skipped += 1
                skipped_count += 1
                continue
            
            config = create_pca_config("fingerprints", dim, seed, base_config)
            filename = f"config_tyro_fingerprints_pca_dim{dim}_seed{seed}.json"
            filepath = os.path.join(OUTPUT_DIR, filename)
            
            with open(filepath, 'w') as f:
                json.dump(config, f, indent=2)
            
            config_count += 1
            fingerprints_pca_created += 1
    
    print(f"  Created: {fingerprints_pca_created}, Skipped (completed): {fingerprints_pca_skipped}")
    
    # ========================================================================
    # FINGERPRINTS - UMAP-Jaccard (2D only × hyperparameters) - KEEP UNCHANGED
    # ========================================================================
    print("Generating Fingerprints-UMAP-Jaccard configs (UNCHANGED)...")
    fingerprints_umap_created = 0
    fingerprints_umap_skipped = 0
    
    for dim, nn, md, seed in product(FINGERPRINT_DIMS, FINGERPRINTS_N_NEIGHBORS, FINGERPRINTS_MIN_DIST, SEEDS):
        # Check if already completed
        if check_experiment_completed("fingerprints", "umap", dim, seed, nn, md):
            fingerprints_umap_skipped += 1
            skipped_count += 1
            continue
        
        config = create_umap_config("fingerprints", "jaccard", dim, nn, md, seed, base_config)
        filename = f"config_tyro_fingerprints_umap_jaccard_dim{dim}_nn{nn}_md{md}_seed{seed}.json"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        
        config_count += 1
        fingerprints_umap_created += 1
    
    print(f"  Created: {fingerprints_umap_created}, Skipped (completed): {fingerprints_umap_skipped}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    total_expected_features = (
        len(FEATURE_DIMS) * len(SEEDS) +  # PCA
        len(FEATURE_DIMS) * len(FEATURES_N_NEIGHBORS) * len(FEATURES_MIN_DIST) * len(SEEDS)  # UMAP
    )
    total_expected_fingerprints = (
        len(FINGERPRINT_DIMS) * len(SEEDS) +  # PCA
        len(FINGERPRINT_DIMS) * len(FINGERPRINTS_N_NEIGHBORS) * len(FINGERPRINTS_MIN_DIST) * len(SEEDS)  # UMAP
    )
    total_expected = total_expected_features + total_expected_fingerprints
    
    print()
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"NEW configs generated:    {config_count}")
    print(f"SKIPPED (completed):      {skipped_count}")
    print(f"TOTAL expected:           {total_expected}")
    print()
    print("Breakdown (Created / Skipped):")
    print(f"  Features-PCA:               {features_pca_created:3d} / {features_pca_skipped:3d}")
    print(f"  Features-UMAP-Euclidean:    {features_umap_created:3d} / {features_umap_skipped:3d}")
    print(f"  Fingerprints-PCA:           {fingerprints_pca_created:3d} / {fingerprints_pca_skipped:3d}")
    print(f"  Fingerprints-UMAP-Jaccard:  {fingerprints_umap_created:3d} / {fingerprints_umap_skipped:3d}")
    print()
    print("Expected total experiments:")
    print(f"  Features:      {total_expected_features}")
    print(f"  Fingerprints:  {total_expected_fingerprints}")
    print(f"  GRAND TOTAL:   {total_expected}")
    print()
    print("Grid sizes:")
    print(f"  Features-UMAP:      {len(FEATURES_N_NEIGHBORS)} nn × {len(FEATURES_MIN_DIST)} md = {len(FEATURES_N_NEIGHBORS) * len(FEATURES_MIN_DIST)} combinations")
    print(f"  Fingerprints-UMAP:  {len(FINGERPRINTS_N_NEIGHBORS)} nn × {len(FINGERPRINTS_MIN_DIST)} md = {len(FINGERPRINTS_N_NEIGHBORS) * len(FINGERPRINTS_MIN_DIST)} combinations")
    print("="*80)


if __name__ == "__main__":
    main()
