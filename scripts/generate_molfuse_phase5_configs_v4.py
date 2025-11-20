#!/usr/bin/env python3
"""
Generate Phase 5 configs for validation & baseline experiments.

Creates 17 configs total:
    1. Tanimoto (ECFP4 similarity baseline): 5 replicates
    2. Raw descriptors (no UMAP): 5 replicates
    3. Negative control (7 non-kinase KW sets): 1 per KW set

Phase 5 tests whether Phase 2's methodology (100 nM cutoff, UMAP preprocessing)
provides meaningful signal vs. simpler baselines.

Negative Control Rationale:
    Tests 7 non-kinase functional classes from Phase 4 as cross-validation:
    - If Phase 2 truly captures kinase-specific chemical space, non-kinase
      ligands should score ~randomly (EF@1% ≈ 1.0) against kinase MF cloud.
    - 7 independent tests provide robust evidence of specificity.
    
    KW sets (full datasets, no sampling):
        - KW-0049_Antioxidant (P00441, 39 compounds)
        - KW-0929_Antimicrobial (P14555, 582 compounds)
        - KW-0505_Motor_protein (P52732, 1,158 compounds)
        - KW-0202_Cytokine (P43490, 2,904 compounds)
        - KW-0358_Heparin-binding (P11362, 4,150 compounds)
        - KW-0456_Lyase (P00918, 9,685 compounds)
        - KW-0560_Oxidoreductase (P08684, 6,151 compounds)
    
    Excluded: KW-0808_Transferase (contains kinases)

Usage:
    python scripts/generate_molfuse_phase5_configs_v4.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List


# ============================================================================
# Configuration
# ============================================================================

# ABL1 target (kinase baseline from Phase 4)
TARGET_ACCESSION = "P00519"  # ABL1 - Tyrosine-protein kinase ABL1

# Negative control KW sets (non-kinase functional classes)
# Using 7 KW sets from Phase 4 (excluding KW-0808_Transferase which contains kinases)
NEGATIVE_CONTROL_KW_SETS = [
    ("KW-0049_Antioxidant", "P00441"),        # SOD1
    ("KW-0929_Antimicrobial", "P14555"),      # PLA2G2A
    ("KW-0505_Motor_protein", "P52732"),      # KIF11
    ("KW-0202_Cytokine", "P43490"),           # NAMPT
    ("KW-0358_Heparin-binding", "P11362"),    # FGFR1
    ("KW-0456_Lyase", "P00918"),              # CA2
    ("KW-0560_Oxidoreductase", "P08684"),     # CYP3A4
]

N_REPLICATES = 5
RANDOM_SEEDS = [42, 123, 456, 789, 1011]

# Dimension (2D for visualization, consistent with Phase 1-4)
DIM = 2

# Phase 2 methodology (100 nM affinity cutoff for scoring)
AFFINITY_CUTOFF_NM = 100

# Dataset paths (using Phase 4 recalculated datasets)
BASE_DATA_DIR = Path("output_recalculated_full_datasets/datasets_2d_all")

# Phase 2 best model directory (auto-detected in phase5.py, but specified here for clarity)
PHASE2_BEST_MODEL_DIR = "experiment_workspace_v4/phase2"


# ============================================================================
# Config Generator
# ============================================================================

def generate_tanimoto_configs(output_dir: Path) -> List[Path]:
    """Generate Tanimoto baseline configs (5 replicates)."""
    configs = []
    
    for rep_idx, seed in enumerate(RANDOM_SEEDS[:N_REPLICATES], start=1):
        run_name = f"tanimoto_rep{rep_idx}"
        
        # Build dataset paths (ABL1/kinase)
        # Need features CSV for actives/affinity filtering, fingerprints CSV for scoring
        mf_features_csv = BASE_DATA_DIR / "KW-0808_Transferase_affinity_extracted_features.csv"
        mf_fingerprints_csv = BASE_DATA_DIR / "KW-0808_Transferase_affinity_extracted_fingerprints_ECFP4.csv"
        zinc_features_csv = BASE_DATA_DIR / "zinc" / "zinc_acquirable_extracted_features.csv"
        zinc_fingerprints_csv = BASE_DATA_DIR / "zinc" / "zinc_acquirable_extracted_fingerprints_ECFP4.csv"
        
        config = {
            "run_name": run_name,
            "experiment_type": "tanimoto",
            "description": "Tanimoto ECFP4 similarity baseline (no UMAP, no feature selection)",
            "target": TARGET_ACCESSION,
            "affinity_cutoff_nM": AFFINITY_CUTOFF_NM,
            "replicate": rep_idx,
            "random_seed": seed,
            "dim": DIM,
            "phase5_run_name": "validation",
            
            # Dataset paths (features for filtering, fingerprints for scoring)
            "mf_features_csv": str(mf_features_csv),
            "mf_fingerprints_csv": str(mf_fingerprints_csv),
            "zinc_features_csv": str(zinc_features_csv),
            "zinc_fingerprints_csv": str(zinc_fingerprints_csv),
        }
        
        config_path = output_dir / f"{run_name}.json"
        with config_path.open("w") as f:
            json.dump(config, f, indent=2)
        
        configs.append(config_path)
        print(f"Generated: {config_path.name}")
    
    return configs


def generate_raw_descriptors_configs(output_dir: Path) -> List[Path]:
    """Generate raw descriptor configs (5 replicates, no UMAP)."""
    configs = []
    
    for rep_idx, seed in enumerate(RANDOM_SEEDS[:N_REPLICATES], start=1):
        run_name = f"raw_descriptors_rep{rep_idx}"
        
        # Build dataset paths (ABL1/kinase features)
        mf_features_csv = BASE_DATA_DIR / "KW-0808_Transferase_affinity_extracted_features.csv"
        zinc_features_csv = BASE_DATA_DIR / "zinc" / "zinc_acquirable_extracted_features.csv"
        
        config = {
            "run_name": run_name,
            "experiment_type": "raw_descriptors",
            "description": "High-D scaled features (no UMAP) - tests if dimensionality reduction is necessary",
            "target": TARGET_ACCESSION,
            "affinity_cutoff_nM": AFFINITY_CUTOFF_NM,
            "replicate": rep_idx,
            "random_seed": seed,
            "dim": DIM,
            "phase5_run_name": "validation",
            
            # Phase 2 model artifacts (auto-detected by highest EF@1%)
            "phase2_best_model_dir": PHASE2_BEST_MODEL_DIR,
            
            # Dataset paths (features only)
            "mf_features_csv": str(mf_features_csv),
            "zinc_features_csv": str(zinc_features_csv),
        }
        
        config_path = output_dir / f"{run_name}.json"
        with config_path.open("w") as f:
            json.dump(config, f, indent=2)
        
        configs.append(config_path)
        print(f"Generated: {config_path.name}")
    
    return configs


def generate_negative_control_configs(output_dir: Path) -> List[Path]:
    """Generate negative control configs (7 KW sets, 1 per set)."""
    configs = []
    
    for kw_name, kw_accession in NEGATIVE_CONTROL_KW_SETS:
        # Extract short name (e.g., "Antioxidant" from "KW-0049_Antioxidant")
        kw_short = kw_name.split("_", 1)[1]
        run_name = f"negative_control_{kw_short}"
        
        # Build dataset paths
        kw_features_csv = BASE_DATA_DIR / f"{kw_name}_affinity_extracted_features.csv"
        zinc_features_csv = BASE_DATA_DIR / "zinc" / "zinc_acquirable_extracted_features.csv"
        
        config = {
            "run_name": run_name,
            "experiment_type": "negative_control",
            "description": f"Negative control: {kw_short} ligands (non-kinase) scored against kinase MF cloud. Expected: EF@1% ≈ 1.0 (no enrichment)",
            "target": TARGET_ACCESSION,  # Kinase target (ABL1) for scoring reference
            "affinity_cutoff_nM": AFFINITY_CUTOFF_NM,
            "replicate": 1,  # No replicates for negative control
            "random_seed": RANDOM_SEEDS[0],  # Use first seed
            "dim": DIM,
            "phase5_run_name": "validation",
            
            # Phase 2 model artifacts (auto-detected by highest EF@1%)
            "phase2_best_model_dir": PHASE2_BEST_MODEL_DIR,
            
            # Negative control specific fields
            "negative_control_kw_csv": str(kw_features_csv),
            "negative_control_kw_name": kw_short,
            "negative_control_kw_accession": kw_accession,
            
            # Dataset paths
            "zinc_features_csv": str(zinc_features_csv),
        }
        
        config_path = output_dir / f"{run_name}.json"
        with config_path.open("w") as f:
            json.dump(config, f, indent=2)
        
        configs.append(config_path)
        print(f"Generated: {config_path.name}")
    
    return configs


def main():
    output_dir = Path("configs/molfuse_phase5_grid")
    
    print("="*80)
    print("PHASE 5 CONFIG GENERATOR: Validation & Baseline Experiments")
    print("="*80)
    print(f"Experiment 1: Tanimoto ECFP4 baseline ({N_REPLICATES} replicates)")
    print(f"Experiment 2: Raw descriptors (no UMAP) ({N_REPLICATES} replicates)")
    print(f"Experiment 3: Negative control ({len(NEGATIVE_CONTROL_KW_SETS)} KW sets)")
    print(f"Total configs: {N_REPLICATES * 2 + len(NEGATIVE_CONTROL_KW_SETS)}")
    print(f"Affinity cutoff: {AFFINITY_CUTOFF_NM} nM (Phase 2 methodology)")
    print("="*80)
    
    # Generate all configs
    tanimoto_configs = generate_tanimoto_configs(output_dir)
    raw_descriptor_configs = generate_raw_descriptors_configs(output_dir)
    negative_control_configs = generate_negative_control_configs(output_dir)
    
    all_configs = tanimoto_configs + raw_descriptor_configs + negative_control_configs
    
    print("\n" + "="*80)
    print(f"SUCCESS: Generated {len(all_configs)} configs")
    print(f"Output directory: {output_dir}")
    print("="*80)
    
    # Print summary
    print("\nExperiment breakdown:")
    print(f"  Tanimoto baseline:       {len(tanimoto_configs)} configs")
    print(f"  Raw descriptors (no UMAP): {len(raw_descriptor_configs)} configs")
    print(f"  Negative control (7 KW):   {len(negative_control_configs)} configs")
    
    print("\nNegative control KW sets:")
    for kw_name, kw_accession in NEGATIVE_CONTROL_KW_SETS:
        kw_short = kw_name.split("_", 1)[1]
        print(f"  {kw_short:20s} ({kw_accession})")
    
    print("\nNext steps:")
    print("  1. Review configs in configs/molfuse_phase5_grid/")
    print("  2. Local test: python -m molfuse.cli.phase5 --config configs/molfuse_phase5_grid/tanimoto_rep1.json --workspace experiment_workspace_v4")
    print("  3. Submit to HPC: bash hpc/submit_molfuse_phase5.sh configs/molfuse_phase5_grid experiment_workspace_v4")


if __name__ == "__main__":
    main()
