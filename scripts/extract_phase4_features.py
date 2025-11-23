#!/usr/bin/env python3
"""
Extract feature lists and reconstruct imputers from Phase 4 runs.

This script reconstructs Phase 4's exact preprocessing pipeline and saves:
- features_used.txt: The exact features Phase 4 used after zero-variance filtering
- imputer.joblib: The imputer Phase 4 fitted (required by scaler, but Phase 4 didn't save it)

**SAFETY**: This script is READ-ONLY except for creating features_used.txt and imputer.joblib.
It will NOT overwrite or modify any existing Phase 4 artifacts (embeddings, scaler, UMAP model).

Usage:
    # Single run
    python scripts/extract_phase4_features.py experiment_workspace_v4/phase4/cross_target/umap_features_Lyase_rep4
    
    # All Phase 4 runs (batch mode)
    python scripts/extract_phase4_features.py --batch experiment_workspace_v4/phase4/cross_target
    
    # Force overwrite existing files
    python scripts/extract_phase4_features.py --batch --force experiment_workspace_v4/phase4/cross_target
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import List

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer

# Add parent directory to path to import molfuse
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import Phase 4's preprocessing functions
from molfuse.data.prep import (
    select_feature_columns,
    remove_zero_variance_features,
)


def extract_features_from_phase4_run(phase4_run_dir: Path, force: bool = False) -> bool:
    """
    Extract and save the feature list used by a Phase 4 run.
    
    This reconstructs Phase 4's preprocessing pipeline using the exact same code path
    to guarantee feature list correctness.
    
    Args:
        phase4_run_dir: Path to Phase 4 run directory
        force: If True, overwrite existing features_used.txt and imputer.joblib
        
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*80}")
    print(f"Processing: {phase4_run_dir.name}")
    print(f"{'='*80}")
    
    # Validate directory structure
    artifacts_dir = phase4_run_dir / "artifacts"
    logs_dir = phase4_run_dir / "logs"
    
    if not artifacts_dir.exists():
        print(f"❌ ERROR: Artifacts directory not found: {artifacts_dir}")
        return False
    
    if not logs_dir.exists():
        print(f"❌ ERROR: Logs directory not found: {logs_dir}")
        return False
    
    # Check for Phase 4 summary (should exist)
    phase4_summary_path = logs_dir / "phase4_summary.json"
    if not phase4_summary_path.exists():
        print(f"❌ ERROR: Phase 4 summary not found: {phase4_summary_path}")
        return False
    
    # Check if artifacts already exist
    features_file = artifacts_dir / "features_used.txt"
    imputer_file = artifacts_dir / "imputer.joblib"
    
    if not force and features_file.exists() and imputer_file.exists():
        print(f"✓ SKIP: Artifacts already exist (use --force to overwrite)")
        return True
    
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
        return False
    
    if not zinc_csv.exists():
        print(f"❌ ERROR: ZINC CSV not found: {zinc_csv}")
        return False
    
    # Load data in chunks to reduce memory
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
    print(f"  Final features: {len(final_features)}")
    
    # Verify against scaler
    scaler_path = artifacts_dir / "scaler.joblib"
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        expected_n_features = scaler.n_features_in_
        
        if len(final_features) == expected_n_features:
            print(f"  ✓ Feature count matches scaler: {expected_n_features}")
        else:
            print(f"  ⚠ WARNING: Feature count mismatch!")
            print(f"    Extracted: {len(final_features)}")
            print(f"    Scaler expects: {expected_n_features}")
            print(f"    Difference: {len(final_features) - expected_n_features}")
            return False
    
    # Reconstruct and save imputer (Phase 4 didn't save it, but scaler expects it)
    print("\nReconstructing imputer...")
    
    # Fit imputer on MF+ZINC with final features (EXACT same data Phase 4 used)
    X_train = df_train[final_features].to_numpy(dtype=float, copy=False)
    imputer = SimpleImputer(strategy='median', copy=True)
    imputer.fit(X_train)
    
    # Save imputer
    joblib.dump(imputer, imputer_file)
    print(f"  ✓ Saved imputer: {imputer_file.name}")
    
    # Save feature list
    print(f"\nSaving feature list...")
    with open(features_file, "w") as f:
        for feat in final_features:
            f.write(f"{feat}\n")
    print(f"  ✓ Saved {len(final_features)} features: {features_file.name}")
    
    # Clean up memory
    del df_mf, df_zinc, df_train, X_train, imputer
    gc.collect()
    
    print(f"✓ SUCCESS")
    return True


def find_phase4_runs(phase4_dir: Path) -> List[Path]:
    """
    Find all Phase 4 run directories in the given directory.
    
    Args:
        phase4_dir: Path to Phase 4 directory (e.g., experiment_workspace_v4/phase4/cross_target)
        
    Returns:
        List of Phase 4 run directories
    """
    if not phase4_dir.exists():
        return []
    
    # Look for directories with "umap_features_" prefix (Phase 4 naming convention)
    run_dirs = sorted([d for d in phase4_dir.iterdir() 
                      if d.is_dir() and d.name.startswith("umap_features_")])
    
    return run_dirs


def main():
    parser = argparse.ArgumentParser(
        description="Extract feature lists and reconstruct imputers from Phase 4 runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single run
    python scripts/extract_phase4_features.py experiment_workspace_v4/phase4/cross_target/umap_features_Lyase_rep4
    
    # Batch process all Phase 4 runs
    python scripts/extract_phase4_features.py --batch experiment_workspace_v4/phase4/cross_target
    
    # Force overwrite
    python scripts/extract_phase4_features.py --batch --force experiment_workspace_v4/phase4/cross_target
        """
    )
    parser.add_argument("path", type=str,
                       help="Path to Phase 4 run directory or parent directory (with --batch)")
    parser.add_argument("--batch", action="store_true",
                       help="Process all Phase 4 runs in the directory")
    parser.add_argument("--force", action="store_true",
                       help="Overwrite existing features_used.txt and imputer.joblib")
    
    args = parser.parse_args()
    
    path = Path(args.path)
    
    if not path.exists():
        print(f"❌ ERROR: Path not found: {path}")
        sys.exit(1)
    
    if not path.is_dir():
        print(f"❌ ERROR: Not a directory: {path}")
        sys.exit(1)
    
    # Determine which runs to process
    if args.batch:
        run_dirs = find_phase4_runs(path)
        
        if not run_dirs:
            print(f"❌ ERROR: No Phase 4 runs found in: {path}")
            print("  Expected directories with prefix: umap_features_")
            sys.exit(1)
        
        print(f"\nFound {len(run_dirs)} Phase 4 runs to process")
        print(f"{'='*80}")
        
        # Process all runs
        success_count = 0
        fail_count = 0
        skip_count = 0
        
        for i, run_dir in enumerate(run_dirs, 1):
            print(f"\n[{i}/{len(run_dirs)}] Processing: {run_dir.name}")
            
            success = extract_features_from_phase4_run(run_dir, force=args.force)
            
            if success:
                # Check if it was skipped or actually processed
                features_file = run_dir / "artifacts" / "features_used.txt"
                imputer_file = run_dir / "artifacts" / "imputer.joblib"
                if features_file.exists() and imputer_file.exists():
                    success_count += 1
                else:
                    skip_count += 1
            else:
                fail_count += 1
        
        # Summary
        print(f"\n{'='*80}")
        print(f"BATCH PROCESSING COMPLETE")
        print(f"{'='*80}")
        print(f"Total runs: {len(run_dirs)}")
        print(f"  ✓ Successful: {success_count}")
        if skip_count > 0:
            print(f"  ⊘ Skipped (already exist): {skip_count}")
        if fail_count > 0:
            print(f"  ✗ Failed: {fail_count}")
        
        if fail_count > 0:
            sys.exit(1)
    else:
        # Single run
        success = extract_features_from_phase4_run(path, force=args.force)
        
        if not success:
            sys.exit(1)


if __name__ == "__main__":
    main()
