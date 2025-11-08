#!/usr/bin/env python3
"""
Generate Phase 4 configs for cross-target generalization study.

Creates configs for 8 target proteins × 5 replicates = 40 runs.
Uses UMAP/features (best method from Phase 3) with full MF cloud per target.

Target Selection (spanning 3 orders of magnitude):
    - KW-0049_Antioxidant: 83 compounds
    - KW-0929_Antimicrobial: 870 compounds
    - KW-0339_Growth_factor: 1,451 compounds
    - KW-0202_Cytokine: 5,812 compounds
    - KW-0358_Heparin-binding: 11,765 compounds
    - KW-0456_Lyase: 47,371 compounds
    - KW-0560_Oxidoreductase: 100,769 compounds
    - KW-0808_Transferase: 430,795 compounds (baseline)

Usage:
    python scripts/generate_molfuse_phase4_configs_v4.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple


# ============================================================================
# Configuration
# ============================================================================

# Targets with natural MF cloud sizes (from wc -l output)
TARGETS: List[Tuple[str, str, int]] = [
    ("KW-0049_Antioxidant", "ABL1_P00519", 83),  # Example accession, will need actual mapping
    ("KW-0929_Antimicrobial", "ABL1_P00519", 870),
    ("KW-0505_Motor_protein", "ABL1_P00519", 2445),
    ("KW-0202_Cytokine", "ABL1_P00519", 5812),
    ("KW-0358_Heparin-binding", "ABL1_P00519", 11765),
    ("KW-0456_Lyase", "ABL1_P00519", 47371),
    ("KW-0560_Oxidoreductase", "ABL1_P00519", 100769),
    ("KW-0808_Transferase", "ABL1_P00519", 430795),  # baseline
]

# NOTE: The accessions above are placeholders. In reality, you would need to:
# 1. Select specific proteins from each KW category
# 2. Map to actual UniProt accessions
# For now, we'll use target_kw as the identifier

N_REPLICATES = 5
RANDOM_SEEDS = [42, 123, 456, 789, 1011]

# Method: UMAP/features (best from Phase 3)
METHOD = "umap"
REPRESENTATION = "features"

# Best hyperparameters from Phase 1
BEST_UMAP_FEATURES = {
    "dim": 10,
    "n_neighbors": 5,
    "min_dist": 0.0,
}

# Optimal affinity cutoff from Phase 2 (assumed 100K nM for now)
AFFINITY_CUTOFF_NM = 100000

# Dataset paths (recalculated features)
BASE_DATA_DIR = Path("output_recalculated_full_datasets/datasets_2d_all")


# ============================================================================
# Config Generator
# ============================================================================

def generate_phase4_configs(output_dir: Path) -> List[Path]:
    """
    Generate Phase 4 config files for cross-target study.
    
    Returns:
        List of generated config file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generated_configs = []
    
    for target_kw, target_accession, mf_size in TARGETS:
        # Extract target short name (remove KW prefix)
        target_short = target_kw.split("_", 1)[1]  # e.g., "Antioxidant"
        
        for rep_idx, seed in enumerate(RANDOM_SEEDS[:N_REPLICATES], start=1):
            run_name = f"{METHOD}_{REPRESENTATION}_{target_short}_rep{rep_idx}"
            
            # Build dataset paths
            mf_features_csv = BASE_DATA_DIR / f"{target_kw}_affinity_extracted_features.csv"
            mf_fingerprints_csv = BASE_DATA_DIR / f"{target_kw}_affinity_extracted_fingerprints_ECFP4.csv"
            
            # ZINC datasets (shared across all targets)
            zinc_features_csv = BASE_DATA_DIR / "zinc_subset_features.csv"
            zinc_fingerprints_csv = BASE_DATA_DIR / "zinc_subset_fingerprints_ECFP4.csv"
            
            config = {
                "run_name": run_name,
                "method": METHOD,
                "representation": REPRESENTATION,
                "dim": BEST_UMAP_FEATURES["dim"],
                "target": target_accession,  # UniProt accession (placeholder)
                "target_kw": target_kw,  # KW identifier (primary)
                "target_short": target_short,
                "mf_size_natural": mf_size,
                "affinity_cutoff_nM": AFFINITY_CUTOFF_NM,
                "replicate": rep_idx,
                "random_seed": seed,
                "phase4_run_name": "cross_target",
                
                # UMAP hyperparameters
                "umap_params": {
                    "n_neighbors": BEST_UMAP_FEATURES["n_neighbors"],
                    "min_dist": BEST_UMAP_FEATURES["min_dist"],
                },
                
                # Dataset paths
                "mf_features_csv": str(mf_features_csv),
                "mf_fingerprints_csv": str(mf_fingerprints_csv),
                "zinc_features_csv": str(zinc_features_csv),
                "zinc_fingerprints_csv": str(zinc_fingerprints_csv),
            }
            
            # Save config
            config_path = output_dir / f"{run_name}.json"
            with config_path.open("w") as f:
                json.dump(config, f, indent=2)
            
            generated_configs.append(config_path)
            print(f"Generated: {config_path.name}")
    
    return generated_configs


def main():
    output_dir = Path("configs/molfuse_phase4_grid")
    
    print("="*80)
    print("PHASE 4 CONFIG GENERATOR: Cross-Target Generalization Study")
    print("="*80)
    print(f"Targets: {len(TARGETS)}")
    print(f"Replicates per target: {N_REPLICATES}")
    print(f"Total configs: {len(TARGETS) * N_REPLICATES}")
    print(f"Method: {METHOD}/{REPRESENTATION}")
    print(f"Hyperparameters: dim={BEST_UMAP_FEATURES['dim']}, n_neighbors={BEST_UMAP_FEATURES['n_neighbors']}, min_dist={BEST_UMAP_FEATURES['min_dist']}")
    print(f"Affinity cutoff: {AFFINITY_CUTOFF_NM} nM")
    print("="*80)
    
    generated = generate_phase4_configs(output_dir)
    
    print("\n" + "="*80)
    print(f"SUCCESS: Generated {len(generated)} configs")
    print(f"Output directory: {output_dir}")
    print("="*80)
    
    # Print summary by target
    print("\nConfigs per target:")
    for target_kw, _, mf_size in TARGETS:
        target_short = target_kw.split("_", 1)[1]
        print(f"  {target_short:20s} (MF={mf_size:>7,}): {N_REPLICATES} replicates")
    
    print("\nNext steps:")
    print("  1. Review configs in configs/molfuse_phase4_grid/")
    print("  2. Submit to HPC: bash hpc/submit_molfuse_phase4.sh configs/molfuse_phase4_grid experiment_workspace_v4")
    print("  3. Run post-analysis: python scripts/phase4_post_analysis.py")


if __name__ == "__main__":
    main()
