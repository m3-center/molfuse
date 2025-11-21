#!/usr/bin/env python3
"""
Generate Phase 5 Expansion configs for Cross-Target Validation.

Expands Phase 5 controls to all 8 targets from Phase 4:
    1. Tanimoto Baseline (5 reps x 8 targets = 40 runs)
    2. Raw Descriptors Baseline (5 reps x 8 targets = 40 runs)
    3. Negative Control Specificity Matrix (8 targets x 7 non-targets = 56 runs)

Total: 136 runs.

Usage:
    python scripts/generate_molfuse_phase5_expansion.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple


# ============================================================================
# Configuration
# ============================================================================

# Targets from Phase 4
TARGETS: List[Tuple[str, str]] = [
    ("KW-0049_Antioxidant", "P00441"),        # SOD1
    ("KW-0929_Antimicrobial", "P14555"),      # PLA2G2A
    ("KW-0505_Motor_protein", "P52732"),      # KIF11
    ("KW-0202_Cytokine", "P43490"),           # NAMPT
    ("KW-0358_Heparin-binding", "P11362"),    # FGFR1
    ("KW-0456_Lyase", "P00918"),              # CA2
    ("KW-0560_Oxidoreductase", "P08684"),     # CYP3A4
    ("KW-0808_Transferase", "P00519"),        # ABL1
]

N_REPLICATES = 5
RANDOM_SEEDS = [42, 123, 456, 789, 1011]

# Dimension (2D for visualization, consistent with Phase 1-4)
DIM = 2

# Phase 2 methodology (100 nM affinity cutoff for scoring)
AFFINITY_CUTOFF_NM = 100

# Dataset paths (using Phase 4 recalculated datasets)
BASE_DATA_DIR = Path("output_recalculated_full_datasets/datasets_2d_all")

# Phase 4 model directory (on HPC)
PHASE4_MODEL_DIR = "experiment_workspace_v4/phase4/cross_target"
MODEL_DIR_TEMPLATE = "umap_features_{target}_rep{replicate}"


# ============================================================================
# Config Generator
# ============================================================================

def generate_tanimoto_configs(output_dir: Path) -> List[Path]:
    """Generate Tanimoto baseline configs (5 replicates, ABL1 only)."""
    configs = []
    
    # Only run Tanimoto for ABL1 (Transferase) as that's the only target
    # where we have a Fingerprint UMAP model (from Phase 1) to compare against.
    abl1_target = [t for t in TARGETS if "Transferase" in t[0]]
    
    for target_kw, target_accession in abl1_target:
        target_short = target_kw.split("_", 1)[1]
        
        for rep_idx, seed in enumerate(RANDOM_SEEDS[:N_REPLICATES], start=1):
            run_name = f"tanimoto_{target_short}_rep{rep_idx}"
            
            # Build dataset paths
            mf_features_csv = BASE_DATA_DIR / f"{target_kw}_affinity_extracted_features.csv"
            mf_fingerprints_csv = BASE_DATA_DIR / f"{target_kw}_affinity_extracted_fingerprints_ECFP4.csv"
            zinc_features_csv = BASE_DATA_DIR / "zinc" / "zinc_acquirable_extracted_features.csv"
            zinc_fingerprints_csv = BASE_DATA_DIR / "zinc" / "zinc_acquirable_extracted_fingerprints_ECFP4.csv"
            
            config = {
                "run_name": run_name,
                "experiment_type": "tanimoto",
                "description": f"Tanimoto ECFP4 baseline for {target_short}",
                "target": target_accession,
                "target_short": target_short,
                "affinity_cutoff_nM": AFFINITY_CUTOFF_NM,
                "replicate": rep_idx,
                "random_seed": seed,
                "dim": DIM,
                "phase5_run_name": "expansion",
                
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
    """Generate raw descriptor configs (5 replicates x 8 targets)."""
    configs = []
    
    for target_kw, target_accession in TARGETS:
        target_short = target_kw.split("_", 1)[1]
        
        for rep_idx, seed in enumerate(RANDOM_SEEDS[:N_REPLICATES], start=1):
            run_name = f"raw_descriptors_{target_short}_rep{rep_idx}"
            
            # Build dataset paths
            mf_features_csv = BASE_DATA_DIR / f"{target_kw}_affinity_extracted_features.csv"
            zinc_features_csv = BASE_DATA_DIR / "zinc" / "zinc_acquirable_extracted_features.csv"
            
            config = {
                "run_name": run_name,
                "experiment_type": "raw_descriptors",
                "description": f"Raw descriptors baseline for {target_short}",
                "target": target_accession,
                "target_short": target_short,
                "affinity_cutoff_nM": AFFINITY_CUTOFF_NM,
                "replicate": rep_idx,
                "random_seed": seed,
                "dim": DIM,
                "phase5_run_name": "expansion",
                
                # Phase 4 model artifacts
                "phase2_best_model_dir": PHASE4_MODEL_DIR,
                "model_dir_template": MODEL_DIR_TEMPLATE,
                
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
    """Generate negative control configs (8 targets x 7 non-targets)."""
    configs = []
    
    for target_kw, target_accession in TARGETS:
        target_short = target_kw.split("_", 1)[1]
        
        # Loop through all OTHER targets as negative controls
        for neg_kw, neg_accession in TARGETS:
            if neg_kw == target_kw:
                continue
                
            neg_short = neg_kw.split("_", 1)[1]
            run_name = f"negative_control_{target_short}_vs_{neg_short}"
            
            # Build dataset paths
            # Target MF cloud (for model context)
            mf_features_csv = BASE_DATA_DIR / f"{target_kw}_affinity_extracted_features.csv"
            # Negative control KW set (to be scored)
            neg_features_csv = BASE_DATA_DIR / f"{neg_kw}_affinity_extracted_features.csv"
            # ZINC (decoys)
            zinc_features_csv = BASE_DATA_DIR / "zinc" / "zinc_acquirable_extracted_features.csv"
            
            config = {
                "run_name": run_name,
                "experiment_type": "negative_control",
                "description": f"Negative control: {neg_short} ligands scored against {target_short} model",
                "target": target_accession,
                "target_short": target_short,
                "affinity_cutoff_nM": AFFINITY_CUTOFF_NM,
                "replicate": 1,  # Use rep1 model for negative control
                "random_seed": RANDOM_SEEDS[0],
                "dim": DIM,
                "phase5_run_name": "expansion",
                
                # Phase 4 model artifacts
                "phase2_best_model_dir": PHASE4_MODEL_DIR,
                "model_dir_template": MODEL_DIR_TEMPLATE,
                
                # Negative control specific fields
                "negative_control_kw_csv": str(neg_features_csv),
                "negative_control_kw_name": neg_short,
                "negative_control_kw_accession": neg_accession,
                
                # Dataset paths
                "mf_features_csv": str(mf_features_csv), # Explicitly provide MF path to avoid fallback
                "zinc_features_csv": str(zinc_features_csv),
            }
            
            config_path = output_dir / f"{run_name}.json"
            with config_path.open("w") as f:
                json.dump(config, f, indent=2)
            
            configs.append(config_path)
            print(f"Generated: {config_path.name}")
    
    return configs


def main():
    output_dir = Path("configs/molfuse_phase5_expansion")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("PHASE 5 EXPANSION CONFIG GENERATOR")
    print("="*80)
    print(f"Targets: {len(TARGETS)}")
    print(f"Replicates: {N_REPLICATES}")
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
    print(f"  Raw descriptors:         {len(raw_descriptor_configs)} configs")
    print(f"  Negative control matrix: {len(negative_control_configs)} configs")
    
    print("\nNext steps:")
    print("  1. Review configs in configs/molfuse_phase5_expansion/")
    print("  2. Submit to HPC: bash hpc/submit_molfuse_phase5_expansion.sh configs/molfuse_phase5_expansion experiment_workspace_v4")


if __name__ == "__main__":
    main()
