#!/usr/bin/env python3
"""
Diagnose infinity/extreme values in Lyase (P00918) actives dataset.

Checks for:
1. Infinite values (inf, -inf)
2. Extremely large values (> 1e100)
3. Which features contain these problematic values
4. Which molecules (SMILES) have problematic values

Usage:
    python scripts/diagnose_lyase_infinity.py
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path


def diagnose_infinity_issues(
    mf_csv: Path,
    target_accession: str = "P00918",
    output_dir: Path = Path("reporting/lyase_infinity_diagnosis")
):
    """
    Diagnose infinity and extreme value issues in actives dataset.
    
    Args:
        mf_csv: Path to MF cloud CSV (contains actives too)
        target_accession: Target UniProt accession to filter actives
        output_dir: Where to save diagnostic reports
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("LYASE INFINITY DIAGNOSTIC")
    print("="*80)
    print(f"MF CSV: {mf_csv}")
    print(f"Target: {target_accession}")
    print()
    
    # Load data
    print("[1/6] Loading MF cloud CSV...")
    if mf_csv.suffix == ".parquet":
        df = pd.read_parquet(mf_csv)
    else:
        df = pd.read_csv(mf_csv, low_memory=False)
    print(f"  Loaded: {len(df):,} rows, {len(df.columns):,} columns")
    
    # Filter actives
    print(f"\n[2/6] Filtering actives (accession == {target_accession})...")
    df_actives = df[df['accession'] == target_accession].copy()
    print(f"  Actives: {len(df_actives):,} rows")
    
    if len(df_actives) == 0:
        print("ERROR: No actives found!")
        return
    
    # Identify numeric columns (potential features)
    print("\n[3/6] Identifying numeric feature columns...")
    metadata_cols = ['SMILES', 'Compound ChEMBL ID', 'accession', 'Standard Value (nM)', 'canonical_smiles']
    feature_cols = [c for c in df_actives.columns if c not in metadata_cols]
    
    # Try to convert to numeric (coerce errors to NaN)
    df_feat = df_actives[feature_cols].apply(pd.to_numeric, errors='coerce')
    numeric_cols = df_feat.columns[df_feat.notna().any()].tolist()
    print(f"  Numeric feature columns: {len(numeric_cols):,}")
    
    # Check for infinity
    print("\n[4/6] Checking for infinity values...")
    df_numeric = df_feat[numeric_cols]
    
    # Check for inf/-inf
    inf_mask = np.isinf(df_numeric.to_numpy())
    n_inf_values = inf_mask.sum()
    print(f"  Total infinite values: {n_inf_values:,}")
    
    if n_inf_values > 0:
        # Which columns have infinity?
        inf_per_col = np.isinf(df_numeric.to_numpy()).sum(axis=0)
        inf_cols = [col for col, count in zip(numeric_cols, inf_per_col) if count > 0]
        print(f"  Columns with infinity: {len(inf_cols):,}")
        
        # Save detailed report
        inf_report = pd.DataFrame({
            'feature': inf_cols,
            'n_inf_values': [inf_per_col[numeric_cols.index(col)] for col in inf_cols],
            'pct_inf': [100 * inf_per_col[numeric_cols.index(col)] / len(df_actives) for col in inf_cols]
        }).sort_values('n_inf_values', ascending=False)
        
        inf_report_path = output_dir / "infinity_features.csv"
        inf_report.to_csv(inf_report_path, index=False)
        print(f"  Saved: {inf_report_path}")
        print(f"\n  Top 10 features with infinity:")
        print(inf_report.head(10).to_string(index=False))
        
        # Which molecules have infinity?
        inf_per_row = np.isinf(df_numeric.to_numpy()).sum(axis=1)
        inf_row_mask = inf_per_row > 0
        n_molecules_with_inf = inf_row_mask.sum()
        print(f"\n  Molecules with ≥1 infinity: {n_molecules_with_inf:,} / {len(df_actives):,} ({100*n_molecules_with_inf/len(df_actives):.1f}%)")
        
        # Save molecules with infinity
        if 'SMILES' in df_actives.columns:
            df_inf_mols = df_actives.loc[inf_row_mask, ['SMILES', 'Compound ChEMBL ID']].copy()
            df_inf_mols['n_inf_features'] = inf_per_row[inf_row_mask]
            
            inf_mols_path = output_dir / "infinity_molecules.csv"
            df_inf_mols.to_csv(inf_mols_path, index=False)
            print(f"  Saved: {inf_mols_path}")
    
    # Check for extremely large values (not inf, but close)
    print("\n[5/6] Checking for extremely large values (> 1e100)...")
    extreme_mask = (np.abs(df_numeric.to_numpy()) > 1e100) & ~np.isinf(df_numeric.to_numpy()) & ~np.isnan(df_numeric.to_numpy())
    n_extreme_values = extreme_mask.sum()
    print(f"  Total extreme values: {n_extreme_values:,}")
    
    if n_extreme_values > 0:
        extreme_per_col = extreme_mask.sum(axis=0)
        extreme_cols = [col for col, count in zip(numeric_cols, extreme_per_col) if count > 0]
        print(f"  Columns with extreme values: {len(extreme_cols):,}")
        
        extreme_report = pd.DataFrame({
            'feature': extreme_cols,
            'n_extreme_values': [extreme_per_col[numeric_cols.index(col)] for col in extreme_cols],
            'max_value': [df_numeric[col].replace([np.inf, -np.inf], np.nan).max() for col in extreme_cols]
        }).sort_values('n_extreme_values', ascending=False)
        
        extreme_report_path = output_dir / "extreme_features.csv"
        extreme_report.to_csv(extreme_report_path, index=False)
        print(f"  Saved: {extreme_report_path}")
        print(f"\n  Top 10 features with extreme values:")
        print(extreme_report.head(10).to_string(index=False))
    
    # Check for NaN
    print("\n[6/6] Checking for NaN values...")
    nan_mask = np.isnan(df_numeric.to_numpy())
    n_nan_values = nan_mask.sum()
    print(f"  Total NaN values: {n_nan_values:,}")
    nan_per_col = nan_mask.sum(axis=0)
    nan_cols = [col for col, count in zip(numeric_cols, nan_per_col) if count > 0]
    print(f"  Columns with NaN: {len(nan_cols):,}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total actives: {len(df_actives):,}")
    print(f"Numeric features: {len(numeric_cols):,}")
    print(f"Infinite values: {n_inf_values:,} (in {len(inf_cols) if n_inf_values > 0 else 0} features)")
    print(f"Extreme values (>1e100): {n_extreme_values:,} (in {len(extreme_cols) if n_extreme_values > 0 else 0} features)")
    print(f"NaN values: {n_nan_values:,} (in {len(nan_cols)} features)")
    print()
    
    # Recommendation
    if n_inf_values > 0 or n_extreme_values > 0:
        print("RECOMMENDATION:")
        print("  1. Remove features with infinity/extreme values before training")
        print("  2. OR: Clip extreme values to a reasonable max (e.g., 1e10)")
        print("  3. OR: Remove molecules with problematic values")
        print()
        print(f"  If removing features: {len(inf_cols) + len(extreme_cols) if n_extreme_values > 0 else len(inf_cols)} features affected")
        print(f"  If removing molecules: {n_molecules_with_inf if n_inf_values > 0 else 0} molecules affected")
    else:
        print("No infinity or extreme values detected - issue may be in processing pipeline!")
    
    print("="*80)


def main():
    """Main entry point."""
    # Path to Lyase MF cloud (adjust if needed)
    base_dir = Path("output_recalculated_full_datasets/datasets_2d_all")
    mf_csv = base_dir / "KW-0456_Lyase_affinity_extracted_features.csv"
    
    # Check if Parquet exists (faster)
    mf_parquet = mf_csv.with_suffix('.parquet')
    if mf_parquet.exists():
        print(f"Using Parquet cache: {mf_parquet.name}")
        mf_csv = mf_parquet
    
    if not mf_csv.exists():
        print(f"ERROR: File not found: {mf_csv}")
        print("Please update the path in the script.")
        return
    
    diagnose_infinity_issues(mf_csv)


if __name__ == "__main__":
    main()
