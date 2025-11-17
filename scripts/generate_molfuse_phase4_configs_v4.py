#!/usr/bin/env python3
"""
Generate Phase 4 configs for cross-target generalization study.

Creates configs for 8 target proteins × 5 replicates = 40 runs.
Uses UMAP/features (best method from Phase 3) with full MF cloud per target.

Target Selection (spanning 3 orders of magnitude):
    - KW-0049_Antioxidant: P00441 (SOD1, 39 compounds)
    - KW-0929_Antimicrobial: P14555 (PLA2G2A, 582 compounds)
    - KW-0505_Motor_protein: P52732 (KIF11, 1,158 compounds)
    - KW-0202_Cytokine: P43490 (NAMPT, 2,904 compounds)
    - KW-0358_Heparin-binding: P11362 (FGFR1, 4,150 compounds)
    - KW-0456_Lyase: P00918 (CA2, 9,685 compounds)
    - KW-0560_Oxidoreductase: P08684 (CYP3A4, 6,151 compounds)
    - KW-0808_Transferase: P00519 (ABL1, 5,505 compounds - baseline)

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
# Selected based on top compound count per KW category (from analyze_kw_targets.py)
TARGETS: List[Tuple[str, str]] = [
    ("KW-0049_Antioxidant", "P00441"),        # SOD1 - Superoxide dismutase [Cu-Zn]
    ("KW-0929_Antimicrobial", "P14555"),      # PLA2G2A - Phospholipase A2
    ("KW-0505_Motor_protein", "P52732"),      # KIF11 - Kinesin-like protein KIF11
    ("KW-0202_Cytokine", "P43490"),           # NAMPT - Nicotinamide phosphoribosyltransferase
    ("KW-0358_Heparin-binding", "P11362"),    # FGFR1 - Fibroblast growth factor receptor 1
    ("KW-0456_Lyase", "P00918"),              # CA2 - Carbonic anhydrase 2
    ("KW-0560_Oxidoreductase", "P08684"),     # CYP3A4 - Cytochrome P450 3A4
    ("KW-0808_Transferase", "P00519"),        # ABL1 - Tyrosine-protein kinase ABL1 (baseline)
]

N_REPLICATES = 5
RANDOM_SEEDS = [42, 123, 456, 789, 1011]

# Method: UMAP/features (best from Phase 3)
METHOD = "umap"
REPRESENTATION = "features"

# Best hyperparameters from Phase 1 (reporting/phase1_post_analysis/phase1_best_configs.json)
BEST_UMAP_FEATURES = {
    "dim": 2,
    "n_neighbors": 10,
    "min_dist": 0.01,
}

# Optimal affinity cutoff from Phase 2 (reporting/phase2_post_analysis/phase2_best_cutoffs.json)
# Use 100 nM for all targets (best performance across all methods)
# Note: This is used for SCORING only, not for filtering the training MF cloud
AFFINITY_CUTOFF_NM = 100  # 100 nM

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
    
    for target_kw, target_accession in TARGETS:
        # Extract target short name (remove KW prefix)
        target_short = target_kw.split("_", 1)[1]  # e.g., "Antioxidant"
        
        for rep_idx, seed in enumerate(RANDOM_SEEDS[:N_REPLICATES], start=1):
            run_name = f"{METHOD}_{REPRESENTATION}_{target_short}_rep{rep_idx}"
            
            # Build dataset paths
            mf_features_csv = BASE_DATA_DIR / f"{target_kw}_affinity_extracted_features.csv"
            mf_fingerprints_csv = BASE_DATA_DIR / f"{target_kw}_affinity_extracted_fingerprints_ECFP4.csv"
            
            # ZINC datasets (shared across all targets) - in zinc/ subdirectory
            zinc_features_csv = BASE_DATA_DIR / "zinc" / "zinc_acquirable_extracted_features.csv"
            zinc_fingerprints_csv = BASE_DATA_DIR / "zinc" / "zinc_acquirable_extracted_fingerprints_ECFP4.csv"
            
            config = {
                "run_name": run_name,
                "method": METHOD,
                "representation": REPRESENTATION,
                "dim": BEST_UMAP_FEATURES["dim"],
                "target": target_accession,  # UniProt accession (placeholder)
                "target_kw": target_kw,  # KW identifier (primary)
                "target_short": target_short,
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
    for target_kw, _ in TARGETS:
        target_short = target_kw.split("_", 1)[1]
        print(f"  {target_short:20s}: {N_REPLICATES} replicates")
    
    print("\nNext steps:")
    print("  1. Review configs in configs/molfuse_phase4_grid/")
    print("  2. Submit to HPC: bash hpc/submit_molfuse_phase4.sh configs/molfuse_phase4_grid experiment_workspace_v4")
    print("  3. Run post-analysis: python scripts/phase4_post_analysis.py")


if __name__ == "__main__":
    main()
