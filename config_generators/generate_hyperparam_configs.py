#!/usr/bin/env python3
"""
Generate hyperparameter sweep configurations for UMMBAS v2.0.

This script creates configs for ABL1 (Transferase) to find optimal UMAP parameters.

v2.0 Changes:
- Corrected ABL1 molecular function: Transferase (KW-0808) not "Protein kinase inhibitor"
- Removed t-SNE completely (co-embedding only, data leakage)
- Removed co-embedding (projection-only strategy prevents data leakage)
- Comprehensive UMAP hyperparameter grid for both features and fingerprints
- Tests both Euclidean (features) and Jaccard (fingerprints) distances

Hyperparameters to test:
- UMAP n_neighbors: [15, 50, 100, 200, 500, 1000]
- UMAP min_dist: [0.0, 0.01, 0.1, 0.5]
- Total: ~50 configs × 5 seeds = 250 jobs

Usage:
    python config_generators/generate_hyperparam_configs.py
"""

import json
import os
import copy

# --- Configuration for v2.0 Hyperparameter Sweep ---

BASE_CONFIG_FILE = "experiment_config.json"
OUTPUT_DIR = "hyperparam_configs"
SWEEP_WORKSPACE_DIR = "experiment_workspace_hyperparam_sweep_v2/"
SWEEP_REPORT_DIR = "final_report_hyperparam_sweep_v2/"
TARGET_FOR_SWEEP = "TyrosineProteinKinaseABL1_P00519"

# --- v2.0 HYPERPARAMETER DEFINITIONS ---
# Fixed parameters
FIXED_SIMSPACE_DIM = 2  # Only 2D for hyperparameter sweep

# UMAP hyperparameter grid (comprehensive search)
UMAP_N_NEIGHBORS_VALUES = [10, 100, 500]  # Comprehensive range
UMAP_MIN_DIST_VALUES = [0.0, 0.01, 0.1, 0.5]  # From tight to loose embedding

def generate_configs():
    """
    Generates JSON configuration files for v2.0 hyperparameter sweep.
    
    v2.0 Strategy: PROJECTION-ONLY (no co-embedding, no data leakage)
    
    Creates configs for:
    - PCA baseline (projection)
    - UMAP-Euclidean (features) with hyperparameter grid
    - UMAP-Jaccard (fingerprints) with hyperparameter grid
    """
    print("=" * 70)
    print("UMMBAS v2.0 - Hyperparameter Sweep Config Generation")
    print("=" * 70)
    print()
    
    try:
        with open(BASE_CONFIG_FILE, 'r') as f:
            base_config = json.load(f)
    except Exception as e:
        print(f"ERROR: Could not load base config file '{BASE_CONFIG_FILE}'. Aborting. Error: {e}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    target_config = next((t for t in base_config.get("targets", []) if t.get("id_name") == TARGET_FOR_SWEEP), None)
    if not target_config:
        print(f"ERROR: Target '{TARGET_FOR_SWEEP}' not found. Aborting.")
        return
    
    # Verify and update target config for v2.0
    if target_config.get("molecular_function_canonical_name") != "Transferase":
        print(f"⚠️  WARNING: ABL1 MF in base config is '{target_config.get('molecular_function_canonical_name')}'")
        print(f"⚠️  v2.0 requires: 'Transferase' (KW-0808)")
        print(f"⚠️  Updating target config for this sweep...")
        target_config["molecular_function_canonical_name"] = "Transferase"
        target_config["molecular_function_filename_segment"] = "Transferase"
        target_config["molecular_function_display_name"] = "Transferase"
        target_config["molecular_function_kw_code"] = "KW-0808"
    
    config_count = 0

    for repr_type in ["features", "fingerprints"]:
        print(f"\n{'─' * 70}")
        print(f"Representation: {repr_type.upper()}")
        print(f"{'─' * 70}")

        # --- Generate PCA Baseline Config (Projection-Only) ---
        print(f"\n  PCA (Projection):")
        new_config = copy.deepcopy(base_config)
        new_config["global_settings"]["workspace_base_dir"] = SWEEP_WORKSPACE_DIR
        new_config["global_settings"]["final_report_dir"] = SWEEP_REPORT_DIR
        new_config["global_settings"]["simspace_dims_to_test"] = [FIXED_SIMSPACE_DIM]
        new_config["representations"] = [repr_type]
        new_config["targets"] = [target_config]
        new_config["dimensionality_reduction_methods"] = {"pca": base_config["dimensionality_reduction_methods"]["pca"]}

        filename = f"config_{repr_type}_pca_projection.json"
        filepath = os.path.join(OUTPUT_DIR, filename)
        with open(filepath, 'w') as f: 
            json.dump(new_config, f, indent=2)
        print(f"    ✓ {filename}")
        config_count += 1
            
        # --- Generate UMAP Configs (Grid Search, Projection-Only) ---
        # Features: UMAP-Euclidean only (continuous descriptors)
        # Fingerprints: UMAP-Jaccard only (binary vectors, proper distance metric)
        if repr_type == "features":
            umap_methods = ["umap_euclidean"]
        else:  # fingerprints
            umap_methods = ["umap_jaccard"]

        for umap_key in umap_methods:
            metric_name = umap_key.replace("umap_", "").upper()
            print(f"\n  UMAP-{metric_name} (Projection):")
            print(f"    Hyperparameter grid:")
            print(f"      n_neighbors: {UMAP_N_NEIGHBORS_VALUES}")
            print(f"      min_dist: {UMAP_MIN_DIST_VALUES}")
            
            for n_neighbors in UMAP_N_NEIGHBORS_VALUES:
                for min_dist in UMAP_MIN_DIST_VALUES:
                    new_config = copy.deepcopy(base_config)
                    new_config["global_settings"]["workspace_base_dir"] = SWEEP_WORKSPACE_DIR
                    new_config["global_settings"]["final_report_dir"] = SWEEP_REPORT_DIR
                    new_config["global_settings"]["simspace_dims_to_test"] = [FIXED_SIMSPACE_DIM]
                    new_config["representations"] = [repr_type]
                    new_config["targets"] = [target_config]
                    
                    umap_cfg = copy.deepcopy(base_config["dimensionality_reduction_methods"][umap_key])
                    umap_cfg["n_neighbors"] = n_neighbors
                    umap_cfg["min_dist"] = min_dist
                    new_config["dimensionality_reduction_methods"] = {umap_key: umap_cfg}

                    filename = f"config_{repr_type}_{umap_key}_projection_nn{n_neighbors}_md{min_dist}.json"
                    filepath = os.path.join(OUTPUT_DIR, filename)
                    with open(filepath, 'w') as f: 
                        json.dump(new_config, f, indent=2)
                    config_count += 1
            
            total_combos = len(UMAP_N_NEIGHBORS_VALUES) * len(UMAP_MIN_DIST_VALUES)
            print(f"    ✓ Created {total_combos} configs")

    print(f"\n{'=' * 70}")
    print(f"Configuration Generation Complete")
    print(f"{'=' * 70}")
    print(f"\nTotal configs created: {config_count}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    print(f"v2.0 Summary:")
    print(f"  ✓ Target: ABL1 (Transferase KW-0808) - CORRECTED")
    print(f"  ✓ Strategy: Projection-only (no co-embedding, no data leakage)")
    print(f"  ✗ t-SNE: REMOVED (co-embedding only)")
    print(f"  ✓ UMAP grid: {len(UMAP_N_NEIGHBORS_VALUES)} × {len(UMAP_MIN_DIST_VALUES)} = {len(UMAP_N_NEIGHBORS_VALUES) * len(UMAP_MIN_DIST_VALUES)} combinations")
    print()
    print(f"Methods tested:")
    print(f"  • PCA (baseline): 2 configs (features + fingerprints)")
    print(f"  • UMAP-Euclidean: {len(UMAP_N_NEIGHBORS_VALUES) * len(UMAP_MIN_DIST_VALUES)} configs (features only)")
    print(f"  • UMAP-Jaccard: {len(UMAP_N_NEIGHBORS_VALUES) * len(UMAP_MIN_DIST_VALUES)} configs (fingerprints only)")
    print()
    print(f"To run the sweep:")
    print(f"  1. Submit jobs: bash hpc/submit_all_replicates_hyperparameterization.sh")
    print(f"  2. Each config runs with 5 seeds (42-46)")
    print(f"  3. Total jobs: {config_count} configs × 5 seeds = {config_count * 5} jobs")
    print(f"  4. Analyze results: sbatch hpc/ummbas_hyperparameterization_analysis.sh")
    print(f"{'=' * 70}")

if __name__ == "__main__":
    generate_configs()