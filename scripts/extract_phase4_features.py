#!/usr/bin/env python3
"""
Extract feature lists from Phase 4 runs.

This script reconstructs the exact feature preprocessing pipeline used in Phase 4
and saves the final feature list to `features_used.txt` in the artifacts directory.

**SAFETY**: This script is READ-ONLY except for creating `features_used.txt`.
It will NOT overwrite or modify any existing Phase 4 artifacts (embeddings, models, etc.).

Usage:
    # Single run
    python scripts/extract_phase4_features.py experiment_workspace_v4/phase4/cross_target/umap_features_Lyase_rep4
    
    # All Phase 4 runs (batch mode)
    for dir in experiment_workspace_v4/phase4/cross_target/umap_features_*; do
        python scripts/extract_phase4_features.py "$dir"
    done
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Import Phase 4's preprocessing functions
from molfuse.data.prep import (
    select_feature_columns,
    remove_zero_variance_features,
)


def extract_features_from_phase4_run(phase4_run_dir: Path, force: bool = False) -> None:
    """
    Extract and save the feature list used by a Phase 4 run.
    
    This reconstructs Phase 4's preprocessing pipeline using the exact same code path
    to guarantee feature list correctness.
    
    Args:
        phase4_run_dir: Path to Phase 4 run directory (e.g., experiment_workspace_v4/phase4/cross_target/umap_features_Lyase_rep4)
        force: If True, overwrite existing features_used.txt
    """
    print(f"\n{'='*80}")
    print(f"Extracting features from: {phase4_run_dir.name}")
    print(f"{'='*80}")
    
    # Validate directory structure
    artifacts_dir = phase4_run_dir / "artifacts"
    logs_dir = phase4_run_dir / "logs"
    
    if not artifacts_dir.exists():
        print(f"❌ ERROR: Artifacts directory not found: {artifacts_dir}")
        return
    
    if not logs_dir.exists():
        print(f"❌ ERROR: Logs directory not found: {logs_dir}")
        return
    
    # Check for Phase 4 summary (should exist)
    phase4_summary_path = logs_dir / "phase4_summary.json"
    if not phase4_summary_path.exists():
        print(f"❌ ERROR: Phase 4 summary not found: {phase4_summary_path}")
        return
    
    # Check if features_used.txt already exists
    features_file = artifacts_dir / "features_used.txt"
    if features_file.exists() and not force:
        print(f"✓ Features file already exists: {features_file}")
        print(f"  Use --force to overwrite")
        return
    
    # Load Phase 4 config
    print("Loading Phase 4 config...")
    with open(phase4_summary_path, "r") as f:
        phase4_summary = json.load(f)
    phase4_cfg = phase4_summary["config"]
    
    target = phase4_cfg["target"]
    print(f"  Target: {target}")
    
    # Load Phase 4's MF+ZINC data (same as Phase 4 preprocessing)
    print("Loading MF and ZINC data...")
    mf_csv = Path(phase4_cfg["mf_features_csv"])
    zinc_csv = Path(phase4_cfg["zinc_features_csv"])
    
    if not mf_csv.exists():
        print(f"❌ ERROR: MF CSV not found: {mf_csv}")
        return
    
    if not zinc_csv.exists():
        print(f"❌ ERROR: ZINC CSV not found: {zinc_csv}")
        return
    
    df_mf = pd.read_csv(mf_csv, low_memory=False)
    print(f"  MF: {len(df_mf):,} rows")
    
    df_zinc = pd.read_csv(zinc_csv, low_memory=False)
    print(f"  ZINC: {len(df_zinc):,} rows")
    
    # Apply Phase 4's preprocessing (EXACTLY as in phase4.py)
    print("\nApplying Phase 4 preprocessing...")
    
    # 1. Remove target from MF
    if "accession" in df_mf.columns:
        df_mf = df_mf[df_mf["accession"] != target].copy()
        print(f"  After removing target: {len(df_mf):,} rows")
    
    # 2. Deduplicate by SMILES
    smiles_col = "canonical_smiles" if "canonical_smiles" in df_mf.columns else "SMILES"
    df_mf = df_mf.drop_duplicates(subset=[smiles_col], keep="first").copy()
    df_zinc = df_zinc.drop_duplicates(subset=[smiles_col], keep="first").copy()
    print(f"  After deduplication: MF={len(df_mf):,}, ZINC={len(df_zinc):,}")
    
    # 3. Feature selection (select numeric columns)
    print("\nSelecting features...")
    feat_cols_mf = select_feature_columns(df_mf)
    feat_cols_zinc = select_feature_columns(df_zinc)
    common_feats = [c for c in feat_cols_mf if c in feat_cols_zinc]
    print(f"  Common features: {len(common_feats)}")
    
    # 4. Zero-variance filtering (CRITICAL STEP - this is where features are removed)
    print("\nApplying zero-variance filter...")
    df_train = pd.concat([df_mf[common_feats], df_zinc[common_feats]], axis=0, ignore_index=True)
    final_features = remove_zero_variance_features(df_train, common_feats, variance_threshold=1e-12)
    print(f"  Final features (after zero-variance filter): {len(final_features)}")
    
    # Verify against scaler
    scaler_path = artifacts_dir / "scaler.joblib"
    if scaler_path.exists():
        import joblib
        scaler = joblib.load(scaler_path)
        expected_n_features = scaler.n_features_in_
        
        if len(final_features) == expected_n_features:
            print(f"  ✓ Feature count matches scaler: {expected_n_features}")
        else:
            print(f"  ⚠ WARNING: Feature count mismatch!")
            print(f"    Extracted: {len(final_features)}")
            print(f"    Scaler expects: {expected_n_features}")
            print(f"    Difference: {len(final_features) - expected_n_features}")
            
            # Still save, but warn user
            if not force:
                response = input("Continue anyway? [y/N]: ")
                if response.lower() != 'y':
                    print("Aborted.")
                    return
    
    # Save feature list (SAFE: only creates new file, doesn't modify existing artifacts)
    print(f"\nSaving feature list to: {features_file}")
    with open(features_file, "w") as f:
        for feat in final_features:
            f.write(f"{feat}\n")
    
    print(f"✓ SUCCESS: Saved {len(final_features)} features")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Extract feature lists from Phase 4 runs (READ-ONLY operation, safe to run)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single run
    python scripts/extract_phase4_features.py experiment_workspace_v4/phase4/cross_target/umap_features_Lyase_rep4
    
    # Batch process all Phase 4 runs
    for dir in experiment_workspace_v4/phase4/cross_target/umap_features_*; do
        python scripts/extract_phase4_features.py "$dir"
    done
        """
    )
    parser.add_argument("phase4_run_dir", type=str,
                       help="Path to Phase 4 run directory")
    parser.add_argument("--force", action="store_true",
                       help="Overwrite existing features_used.txt")
    
    args = parser.parse_args()
    
    phase4_run_dir = Path(args.phase4_run_dir)
    
    if not phase4_run_dir.exists():
        print(f"❌ ERROR: Directory not found: {phase4_run_dir}")
        sys.exit(1)
    
    if not phase4_run_dir.is_dir():
        print(f"❌ ERROR: Not a directory: {phase4_run_dir}")
        sys.exit(1)
    
    extract_features_from_phase4_run(phase4_run_dir, force=args.force)


if __name__ == "__main__":
    main()
