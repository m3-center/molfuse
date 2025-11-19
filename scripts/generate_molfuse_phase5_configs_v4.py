#!/usr/bin/env python3
"""
Generate Phase 5 configs for validation & baseline experiments.

Phase 5: Three Critical Controls for Reviewer Response
    1. Database Bias Negative Control: Non-transferase (GPCR) scored against kinase model
    2. Raw Descriptor Baseline: 1-NN in high-D space (no UMAP)
    3. Tanimoto Baseline: ECFP4 fingerprint similarity

Target: ABL1 (P00519) - Transferase (kinase baseline from Phase 1/3)
Reference Set: KW-0808_Transferase @ 100 nM cutoff (optimal from Phase 2)

Usage:
    python scripts/generate_molfuse_phase5_configs_v4.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


# ============================================================================
# Configuration
# ============================================================================

# ABL1 kinase baseline (from Phase 1)
TARGET_TRANSFERASE = ("KW-0808_Transferase", "P00519")  # ABL1

# Non-kinase control: Select a GPCR category with sufficient data
# KW-0297_G-protein_coupled_receptor would be ideal, but if not available:
# Fall back to a different functional class (Ion Channel, Protease, etc.)
CONTROL_NON_TRANSFERASE = ("KW-0297_GPCR", "CHEMBL217")  # Dopamine D2 receptor

# Experiments to run
EXPERIMENTS = [
    {
        "name": "negative_control",
        "description": "Database bias negative control: Non-kinase actives scored against kinase model",
        "control_target": CONTROL_NON_TRANSFERASE,
    },
    {
        "name": "raw_descriptors",
        "description": "Raw descriptor baseline: 1-NN in high-D space (no UMAP)",
        "method": "raw",  # No dimensionality reduction
    },
    {
        "name": "tanimoto",
        "description": "Tanimoto baseline: ECFP4 fingerprint similarity",
        "method": "tanimoto",
        "representation": "fingerprints",
    },
]

N_REPLICATES = 5
RANDOM_SEEDS = [42, 123, 456, 789, 1011]

# Optimal hyperparameters from Phase 1 (for UMAP baseline comparison)
BEST_UMAP_FEATURES = {
    "dim": 2,
    "n_neighbors": 10,
    "min_dist": 0.01,
}

# Optimal affinity cutoff from Phase 2
AFFINITY_CUTOFF_NM = 100  # 100 nM

# Dataset paths
BASE_DATA_DIR = Path("datasets/molecular_function_affinity_data")
BASE_FEATURES_DIR = Path("datasets/molecular_function_features_fingerprints")


# ============================================================================
# Config Generator
# ============================================================================

def generate_phase5_configs(output_dir: Path) -> List[Path]:
    """
    Generate Phase 5 config files for validation experiments.
    
    Returns:
        List of generated config file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generated_configs = []
    
    for exp in EXPERIMENTS:
        exp_name = exp["name"]
        
        for rep_idx, seed in enumerate(RANDOM_SEEDS[:N_REPLICATES], start=1):
            run_name = f"phase5_{exp_name}_rep{rep_idx}"
            
            config = {
                "run_name": run_name,
                "experiment_type": exp_name,
                "description": exp["description"],
                "target": TARGET_TRANSFERASE[1],  # P00519 (ABL1)
                "target_kw": TARGET_TRANSFERASE[0],  # KW-0808_Transferase
                "affinity_cutoff_nM": AFFINITY_CUTOFF_NM,
                "replicate": rep_idx,
                "random_seed": seed,
                "phase5_run_name": "validation",
            }
            
            # Experiment-specific config
            if exp_name == "negative_control":
                config["control_target"] = exp["control_target"][1]
                config["control_target_kw"] = exp["control_target"][0]
                config["method"] = "umap"
                config["representation"] = "features"
                config["dim"] = BEST_UMAP_FEATURES["dim"]
                config["umap_params"] = {
                    "n_neighbors": BEST_UMAP_FEATURES["n_neighbors"],
                    "min_dist": BEST_UMAP_FEATURES["min_dist"],
                }
                
            elif exp_name == "raw_descriptors":
                config["method"] = "raw"
                config["representation"] = "features"
                # No dimensionality reduction - use full feature space
                
            elif exp_name == "tanimoto":
                config["method"] = "tanimoto"
                config["representation"] = "fingerprints"
                config["fingerprint_type"] = "ECFP4"
                config["fingerprint_bits"] = 2048
            
            # Dataset paths (shared)
            config["mf_affinity_csv"] = str(BASE_DATA_DIR / f"{TARGET_TRANSFERASE[0]}_affinity.csv")
            config["mf_features_csv"] = str(BASE_FEATURES_DIR / f"{TARGET_TRANSFERASE[0]}_affinity_extracted_features.csv")
            config["mf_fingerprints_csv"] = str(BASE_FEATURES_DIR / f"{TARGET_TRANSFERASE[0]}_affinity_extracted_fingerprints_ECFP4.csv")
            config["zinc_features_csv"] = "datasets/zinc_data.csv"  # Placeholder
            config["zinc_fingerprints_csv"] = "datasets/zinc_data.csv"  # Placeholder
            
            # Save config
            config_path = output_dir / f"{run_name}.json"
            with config_path.open("w") as f:
                json.dump(config, f, indent=2)
            
            generated_configs.append(config_path)
            print(f"Generated: {config_path.name}")
    
    return generated_configs


def main():
    output_dir = Path("configs/molfuse_phase5_grid")
    
    print("="*80)
    print("PHASE 5 CONFIG GENERATOR: Validation & Baseline Experiments")
    print("="*80)
    print(f"Experiments: {len(EXPERIMENTS)}")
    print(f"Replicates per experiment: {N_REPLICATES}")
    print(f"Total configs: {len(EXPERIMENTS) * N_REPLICATES}")
    print(f"Target: {TARGET_TRANSFERASE[0]} ({TARGET_TRANSFERASE[1]})")
    print(f"Affinity cutoff: {AFFINITY_CUTOFF_NM} nM")
    print("="*80)
    
    print("\nExperiments:")
    for exp in EXPERIMENTS:
        print(f"  {exp['name']:20s}: {exp['description']}")
    
    generated = generate_phase5_configs(output_dir)
    
    print("\n" + "="*80)
    print(f"SUCCESS: Generated {len(generated)} configs")
    print(f"Output directory: {output_dir}")
    print("="*80)
    
    print("\nNext steps:")
    print("  1. Review configs in configs/molfuse_phase5_grid/")
    print("  2. Run Phase 5 experiments: python -m molfuse.cli.phase5 --config <config>.json")
    print("  3. Run post-analysis: python scripts/phase5_post_analysis.py")


if __name__ == "__main__":
    main()
