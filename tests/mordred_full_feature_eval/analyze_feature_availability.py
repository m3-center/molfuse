#!/usr/bin/env python3
"""
Feature Availability Analysis for Mordred Descriptor Comparison

This script analyzes feature availability patterns across three representations:
1. 40-feature subset (baseline)
2. Full 2D Mordred (1613 features)
3. Full 2D+3D Mordred (1826 features)

For each representation, it compares four feature selection strategies:
- Strategy 1 (training_only): Select based on MF+ZINC variance only (current behavior)
- Strategy 2 (all_molecules): Only features with 100% coverage (actives + MF + ZINC)
- Strategy 3 (actives_first): Only features computable for ALL actives
- Strategy 4 (threshold_95): Features with ≥95% coverage + imputation

Outputs:
- summary.json: High-level comparison of all strategies
- feature_missingness_*.csv: Per-feature missingness rates for each representation
- molecule_nan_patterns_*.csv: Per-molecule NaN patterns for actives
- strategy_comparison_*.csv: Feature counts and molecule counts per strategy

Author: Feature availability diagnostic for Mordred comparison
"""

from __future__ import annotations
import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


def load_data_from_representation(
    feature_csv: Path,
    target_accession: str,
    n_mf: Optional[int] = None,
    n_target: Optional[int] = None,
    n_zinc: Optional[int] = None,
    zinc_orig_csv: Optional[Path] = None,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load actives, MF, and ZINC for a single representation.
    
    Args:
        feature_csv: Path to KW feature CSV
        target_accession: Target accession for splitting actives
        n_mf: Sample size for MF (None = all)
        n_target: Sample size for actives (None = all)
        n_zinc: Sample size for ZINC (None = all)
        zinc_orig_csv: Path to original ZINC CSV
        seed: Random seed
        
    Returns:
        df_actives, df_mf, df_zinc
    """
    # Check for Parquet version
    parquet_path = feature_csv.with_suffix('.parquet')
    
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    else:
        # Load with selective dtype
        dtype_dict = {
            'Compound ChEMBL ID': str,
            'SMILES': str,
            'Target ChEMBL ID': str,
            'Target Name': str,
            'Activity Type': str,
            'Standard Value (nM)': float,
            'target_chembl_id': str,
            'accession': str,
        }
        
        try:
            df = pd.read_csv(feature_csv, dtype=dtype_dict, engine='pyarrow', low_memory=False)
        except (ImportError, Exception):
            df = pd.read_csv(feature_csv, dtype=dtype_dict, low_memory=False)
    
    # Convert non-metadata columns to numeric
    metadata_cols = {'Compound ChEMBL ID', 'SMILES', 'Target ChEMBL ID', 'Target Name',
                     'Activity Type', 'target_chembl_id', 'accession'}
    for col in df.columns:
        if col not in metadata_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Split by accession
    mask_target = (df['accession'] == target_accession)
    df_actives = df[mask_target].copy()
    df_mf = df[~mask_target].copy()
    
    # Deduplicate by SMILES
    if 'Standard Value (nM)' in df_actives.columns:
        agg_dict = {'Standard Value (nM)': 'median'}
        for col in df_actives.columns:
            if col not in ['SMILES', 'Standard Value (nM)']:
                agg_dict[col] = 'first'
        df_actives = df_actives.groupby('SMILES', as_index=False).agg(agg_dict)
    else:
        df_actives = df_actives.drop_duplicates(subset=['SMILES'], keep='first')
    
    if 'Standard Value (nM)' in df_mf.columns:
        agg_dict = {'Standard Value (nM)': 'median'}
        for col in df_mf.columns:
            if col not in ['SMILES', 'Standard Value (nM)']:
                agg_dict[col] = 'first'
        df_mf = df_mf.groupby('SMILES', as_index=False).agg(agg_dict)
    else:
        df_mf = df_mf.drop_duplicates(subset=['SMILES'], keep='first')
    
    # Sample if requested
    if n_target and len(df_actives) > n_target:
        df_actives = df_actives.sample(n=n_target, random_state=seed)
    
    if n_mf and len(df_mf) > n_mf:
        df_mf = df_mf.sample(n=n_mf, random_state=seed)
    
    # Load ZINC
    if zinc_orig_csv:
        # Load original ZINC for SMILES
        try:
            df_zinc_orig = pd.read_csv(zinc_orig_csv, dtype={'SMILES': str}, engine='pyarrow')
        except (ImportError, Exception):
            df_zinc_orig = pd.read_csv(zinc_orig_csv, dtype={'SMILES': str}, low_memory=False)
        
        df_zinc_orig = df_zinc_orig.drop_duplicates(subset=['SMILES'], keep='first')
        
        if n_zinc and len(df_zinc_orig) > n_zinc:
            df_zinc_orig = df_zinc_orig.sample(n=n_zinc, random_state=seed)
        
        # Load ZINC features
        zinc_feature_csv = feature_csv.parent / "zinc" / "zinc_acquirable_extracted_features.csv"
        zinc_parquet = zinc_feature_csv.with_suffix('.parquet')
        
        if zinc_parquet.exists():
            df_zinc_feat = pd.read_parquet(zinc_parquet)
        else:
            dtype_dict = {'SMILES': str}
            try:
                df_zinc_feat = pd.read_csv(zinc_feature_csv, dtype=dtype_dict, engine='pyarrow', low_memory=False)
            except (ImportError, Exception):
                df_zinc_feat = pd.read_csv(zinc_feature_csv, dtype=dtype_dict, low_memory=False)
        
        # Convert non-SMILES columns to numeric
        for col in df_zinc_feat.columns:
            if col != 'SMILES':
                df_zinc_feat[col] = pd.to_numeric(df_zinc_feat[col], errors='coerce')
        
        # Merge
        df_zinc = df_zinc_orig[['SMILES']].merge(df_zinc_feat, on='SMILES', how='inner')
    else:
        df_zinc = pd.DataFrame()
    
    return df_actives, df_mf, df_zinc


def select_feature_columns(df: pd.DataFrame) -> List[str]:
    """Select numeric feature columns (exclude metadata)."""
    exclude = {
        'SMILES', 'smiles', 'canonical_smiles',
        'Compound ChEMBL ID', 'Target ChEMBL ID', 'Target Name',
        'Activity Type', 'Standard Value (nM)', 'target_chembl_id', 'accession',
        'ZINC_ID', 'LABEL', 'MANUFACTURER', 'TRANCHE',
        'source', 'label'
    }
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols 
                    if c not in exclude 
                    and not c.endswith('_aff') 
                    and not c.endswith('_feat')]
    
    return feature_cols


def compute_missingness_stats(
    df_actives: pd.DataFrame,
    df_mf: pd.DataFrame,
    df_zinc: pd.DataFrame,
    feature_cols: List[str]
) -> pd.DataFrame:
    """Compute per-feature missingness rates across all molecule sets.
    
    Returns:
        DataFrame with columns: feature, miss_rate_actives, miss_rate_mf, miss_rate_zinc, miss_rate_all
    """
    stats = []
    
    for feat in feature_cols:
        # Compute missingness for each set
        miss_actives = df_actives[feat].isna().mean() if feat in df_actives.columns else 1.0
        miss_mf = df_mf[feat].isna().mean() if feat in df_mf.columns else 1.0
        miss_zinc = df_zinc[feat].isna().mean() if feat in df_zinc.columns else 1.0
        
        # Combined missingness
        all_values = pd.concat([
            df_actives[feat] if feat in df_actives.columns else pd.Series(dtype=float),
            df_mf[feat] if feat in df_mf.columns else pd.Series(dtype=float),
            df_zinc[feat] if feat in df_zinc.columns else pd.Series(dtype=float)
        ])
        miss_all = all_values.isna().mean()
        
        stats.append({
            'feature': feat,
            'miss_rate_actives': miss_actives,
            'miss_rate_mf': miss_mf,
            'miss_rate_zinc': miss_zinc,
            'miss_rate_all': miss_all,
            'variance_train': pd.concat([
                df_mf[feat] if feat in df_mf.columns else pd.Series(dtype=float),
                df_zinc[feat] if feat in df_zinc.columns else pd.Series(dtype=float)
            ]).var()
        })
    
    return pd.DataFrame(stats)


def compute_molecule_nan_patterns(df: pd.DataFrame, feature_cols: List[str], label: str) -> pd.DataFrame:
    """Compute per-molecule NaN patterns.
    
    Returns:
        DataFrame with: SMILES, n_features_available, n_features_missing, pct_missing, label
    """
    X = df[feature_cols].copy()
    
    patterns = []
    for idx, row in df.iterrows():
        smiles = row['SMILES']
        feat_values = X.loc[idx]
        n_missing = feat_values.isna().sum()
        n_available = len(feature_cols) - n_missing
        pct_missing = 100.0 * n_missing / len(feature_cols)
        
        patterns.append({
            'SMILES': smiles,
            'n_features_total': len(feature_cols),
            'n_features_available': n_available,
            'n_features_missing': n_missing,
            'pct_missing': pct_missing,
            'source': label
        })
    
    return pd.DataFrame(patterns)


def apply_strategy_training_only(
    df_actives: pd.DataFrame,
    df_mf: pd.DataFrame,
    df_zinc: pd.DataFrame,
    feature_cols: List[str]
) -> Dict:
    """Strategy 1: Select based on MF+ZINC variance only (current behavior).
    
    Returns:
        Dict with: kept_features, n_actives_retained, n_mf_retained, n_zinc_retained
    """
    # Combine MF + ZINC
    X_train = pd.concat([df_mf[feature_cols], df_zinc[feature_cols]], axis=0)
    
    # Cast to numeric
    for c in X_train.columns:
        X_train[c] = pd.to_numeric(X_train[c], errors='coerce')
    
    # Remove zero-variance columns
    variances = X_train.var(axis=0)
    kept_features = variances[variances > 1e-12].index.tolist()
    
    # For actives: subset to kept features, impute (would happen in real pipeline)
    X_actives = df_actives[kept_features].copy()
    for c in X_actives.columns:
        X_actives[c] = pd.to_numeric(X_actives[c], errors='coerce')
    
    # Count how many actives would be retained after imputation (all of them)
    n_actives_retained = len(df_actives)
    
    return {
        'strategy': 'training_only',
        'kept_features': kept_features,
        'n_features': len(kept_features),
        'n_actives_retained': n_actives_retained,
        'n_mf_retained': len(df_mf),
        'n_zinc_retained': len(df_zinc),
    }


def apply_strategy_all_molecules(
    df_actives: pd.DataFrame,
    df_mf: pd.DataFrame,
    df_zinc: pd.DataFrame,
    feature_cols: List[str]
) -> Dict:
    """Strategy 2: Only features with 100% coverage (no NaNs anywhere).
    
    Returns:
        Dict with: kept_features, n_actives_retained, n_mf_retained, n_zinc_retained
    """
    # Find features with zero NaNs across all molecules
    kept_features = []
    
    for feat in feature_cols:
        all_values = pd.concat([
            df_actives[feat] if feat in df_actives.columns else pd.Series(dtype=float),
            df_mf[feat] if feat in df_mf.columns else pd.Series(dtype=float),
            df_zinc[feat] if feat in df_zinc.columns else pd.Series(dtype=float)
        ])
        
        if all_values.notna().all():
            kept_features.append(feat)
    
    # All molecules retained (no NaNs by definition)
    return {
        'strategy': 'all_molecules',
        'kept_features': kept_features,
        'n_features': len(kept_features),
        'n_actives_retained': len(df_actives),
        'n_mf_retained': len(df_mf),
        'n_zinc_retained': len(df_zinc),
    }


def apply_strategy_actives_first(
    df_actives: pd.DataFrame,
    df_mf: pd.DataFrame,
    df_zinc: pd.DataFrame,
    feature_cols: List[str]
) -> Dict:
    """Strategy 3: Only features computable for ALL actives.
    
    Returns:
        Dict with: kept_features, n_actives_retained, n_mf_retained, n_zinc_retained
    """
    # Find features with zero NaNs in actives
    kept_features = []
    
    for feat in feature_cols:
        if feat in df_actives.columns and df_actives[feat].notna().all():
            kept_features.append(feat)
    
    # Count how many MF/ZINC molecules have these features
    X_mf = df_mf[kept_features].copy()
    X_zinc = df_zinc[kept_features].copy()
    
    n_mf_retained = len(df_mf) - X_mf.isna().any(axis=1).sum()
    n_zinc_retained = len(df_zinc) - X_zinc.isna().any(axis=1).sum()
    
    return {
        'strategy': 'actives_first',
        'kept_features': kept_features,
        'n_features': len(kept_features),
        'n_actives_retained': len(df_actives),
        'n_mf_retained': n_mf_retained,
        'n_zinc_retained': n_zinc_retained,
    }


def apply_strategy_threshold_95(
    df_actives: pd.DataFrame,
    df_mf: pd.DataFrame,
    df_zinc: pd.DataFrame,
    feature_cols: List[str]
) -> Dict:
    """Strategy 4: Features with ≥95% coverage + imputation.
    
    Returns:
        Dict with: kept_features, n_actives_retained, n_mf_retained, n_zinc_retained
    """
    # Find features with ≤5% NaNs across all molecules
    kept_features = []
    
    for feat in feature_cols:
        all_values = pd.concat([
            df_actives[feat] if feat in df_actives.columns else pd.Series(dtype=float),
            df_mf[feat] if feat in df_mf.columns else pd.Series(dtype=float),
            df_zinc[feat] if feat in df_zinc.columns else pd.Series(dtype=float)
        ])
        
        miss_rate = all_values.isna().mean()
        if miss_rate <= 0.05:
            kept_features.append(feat)
    
    # With imputation, all molecules retained
    return {
        'strategy': 'threshold_95',
        'kept_features': kept_features,
        'n_features': len(kept_features),
        'n_actives_retained': len(df_actives),
        'n_mf_retained': len(df_mf),
        'n_zinc_retained': len(df_zinc),
    }


def analyze_representation(
    rep_name: str,
    feature_csv: Path,
    target_accession: str,
    zinc_orig_csv: Path,
    n_mf: Optional[int],
    n_target: Optional[int],
    n_zinc: Optional[int],
    output_dir: Path,
    seed: int
) -> Dict:
    """Analyze feature availability for a single representation."""
    
    print(f"\n{'='*80}")
    print(f"Analyzing: {rep_name}")
    print(f"{'='*80}")
    
    # Load data
    print(f"Loading data...")
    df_actives, df_mf, df_zinc = load_data_from_representation(
        feature_csv, target_accession, n_mf, n_target, n_zinc, zinc_orig_csv, seed
    )
    
    print(f"  Actives: {len(df_actives)}")
    print(f"  MF: {len(df_mf)}")
    print(f"  ZINC: {len(df_zinc)}")
    
    # Select feature columns
    feature_cols = select_feature_columns(df_actives)
    print(f"  Feature columns identified: {len(feature_cols)}")
    
    # Compute missingness stats
    print(f"Computing missingness statistics...")
    missingness_df = compute_missingness_stats(df_actives, df_mf, df_zinc, feature_cols)
    
    # Save missingness stats
    missingness_csv = output_dir / f"feature_missingness_{rep_name}.csv"
    missingness_df.to_csv(missingness_csv, index=False)
    print(f"  ✓ Saved: {missingness_csv.name}")
    
    # Compute per-molecule NaN patterns for actives
    print(f"Computing per-molecule NaN patterns...")
    patterns_df = compute_molecule_nan_patterns(df_actives, feature_cols, 'active')
    
    patterns_csv = output_dir / f"molecule_nan_patterns_{rep_name}.csv"
    patterns_df.to_csv(patterns_csv, index=False)
    print(f"  ✓ Saved: {patterns_csv.name}")
    
    # Apply all strategies
    print(f"Applying feature selection strategies...")
    
    strategies = {
        'training_only': apply_strategy_training_only(df_actives, df_mf, df_zinc, feature_cols),
        'all_molecules': apply_strategy_all_molecules(df_actives, df_mf, df_zinc, feature_cols),
        'actives_first': apply_strategy_actives_first(df_actives, df_mf, df_zinc, feature_cols),
        'threshold_95': apply_strategy_threshold_95(df_actives, df_mf, df_zinc, feature_cols),
    }
    
    # Save strategy comparison
    strategy_comparison = []
    for strat_name, strat_result in strategies.items():
        strategy_comparison.append({
            'strategy': strat_name,
            'n_features': strat_result['n_features'],
            'n_actives_retained': strat_result['n_actives_retained'],
            'n_mf_retained': strat_result['n_mf_retained'],
            'n_zinc_retained': strat_result['n_zinc_retained'],
            'actives_lost': len(df_actives) - strat_result['n_actives_retained'],
            'features_lost': len(feature_cols) - strat_result['n_features'],
        })
    
    strategy_df = pd.DataFrame(strategy_comparison)
    strategy_csv = output_dir / f"strategy_comparison_{rep_name}.csv"
    strategy_df.to_csv(strategy_csv, index=False)
    print(f"  ✓ Saved: {strategy_csv.name}")
    
    # Print summary
    print(f"\nStrategy Comparison for {rep_name}:")
    print(f"  {'Strategy':<20} {'Features':>10} {'Actives':>10} {'MF':>10} {'ZINC':>10}")
    print(f"  {'-'*60}")
    for _, row in strategy_df.iterrows():
        print(f"  {row['strategy']:<20} {row['n_features']:>10} {row['n_actives_retained']:>10} "
              f"{row['n_mf_retained']:>10} {row['n_zinc_retained']:>10}")
    
    # Summary stats for missingness
    print(f"\nMissingness Summary:")
    print(f"  Features with 0% missing (all molecules): {(missingness_df['miss_rate_all'] == 0).sum()}")
    print(f"  Features with ≤5% missing (all molecules): {(missingness_df['miss_rate_all'] <= 0.05).sum()}")
    print(f"  Features with ≤10% missing (all molecules): {(missingness_df['miss_rate_all'] <= 0.10).sum()}")
    print(f"  Features with >50% missing (actives): {(missingness_df['miss_rate_actives'] > 0.50).sum()}")
    
    # Return summary for cross-representation comparison
    return {
        'rep_name': rep_name,
        'n_features_total': len(feature_cols),
        'n_actives': len(df_actives),
        'n_mf': len(df_mf),
        'n_zinc': len(df_zinc),
        'strategies': strategies,
        'missingness_stats': {
            'features_0pct_missing': int((missingness_df['miss_rate_all'] == 0).sum()),
            'features_5pct_missing': int((missingness_df['miss_rate_all'] <= 0.05).sum()),
            'features_10pct_missing': int((missingness_df['miss_rate_all'] <= 0.10).sum()),
            'features_50pct_missing_actives': int((missingness_df['miss_rate_actives'] > 0.50).sum()),
        }
    }


def main(args: argparse.Namespace):
    """Main analysis workflow."""
    
    print("="*80)
    print("FEATURE AVAILABILITY ANALYSIS")
    print("="*80)
    
    # Setup paths
    base_dir = Path(args.base_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    features_40_dir = base_dir / "datasets" / "molecular_function_features_fingerprints"
    features_2d_dir = Path(args.full_2d_dir).resolve()
    features_2d3d_dir = Path(args.full_2d3d_dir).resolve()
    
    zinc_orig_csv = base_dir / "datasets" / "zinc_data.csv"
    
    kw_file = args.kw_file
    target_accession = args.target_accession
    
    print(f"\nConfiguration:")
    print(f"  KW file: {kw_file}")
    print(f"  Target accession: {target_accession}")
    print(f"  n_mf: {args.n_mf if args.n_mf else 'all'}")
    print(f"  n_target: {args.n_target if args.n_target else 'all'}")
    print(f"  n_zinc: {args.n_zinc if args.n_zinc else 'all'}")
    print(f"  Output: {output_dir}")
    
    # Analyze each representation
    results = {}
    
    # 40-feature
    results['40feature'] = analyze_representation(
        '40feature',
        features_40_dir / kw_file,
        target_accession,
        zinc_orig_csv,
        args.n_mf,
        args.n_target,
        args.n_zinc,
        output_dir,
        args.seed
    )
    
    # Full 2D
    results['full2d'] = analyze_representation(
        'full2d',
        features_2d_dir / kw_file,
        target_accession,
        zinc_orig_csv,
        args.n_mf,
        args.n_target,
        args.n_zinc,
        output_dir,
        args.seed
    )
    
    # Full 2D+3D
    results['full2d3d'] = analyze_representation(
        'full2d3d',
        features_2d3d_dir / kw_file,
        target_accession,
        zinc_orig_csv,
        args.n_mf,
        args.n_target,
        args.n_zinc,
        output_dir,
        args.seed
    )
    
    # Cross-representation summary
    print(f"\n{'='*80}")
    print("CROSS-REPRESENTATION SUMMARY")
    print(f"{'='*80}")
    
    summary = {
        'config': {
            'kw_file': kw_file,
            'target_accession': target_accession,
            'n_mf': args.n_mf,
            'n_target': args.n_target,
            'n_zinc': args.n_zinc,
            'seed': args.seed,
        },
        'representations': results
    }
    
    # Save summary JSON
    summary_json = output_dir / "summary.json"
    with open(summary_json, 'w') as f:
        json.dump(summary, f, indent=2, default=int)
    print(f"\n✓ Saved: {summary_json.name}")
    
    # Print comparison table
    print(f"\n{'Representation':<15} {'Total Features':>15} {'0% Missing':>12} {'≤5% Missing':>12} {'≤10% Missing':>13}")
    print(f"{'-'*70}")
    for rep_name, rep_data in results.items():
        ms = rep_data['missingness_stats']
        print(f"{rep_name:<15} {rep_data['n_features_total']:>15} {ms['features_0pct_missing']:>12} "
              f"{ms['features_5pct_missing']:>12} {ms['features_10pct_missing']:>13}")
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETED")
    print(f"{'='*80}")
    print(f"\nAll results saved to: {output_dir}")
    print(f"\nOutput files:")
    print(f"  - summary.json (high-level comparison)")
    print(f"  - feature_missingness_*.csv (per-feature missingness rates)")
    print(f"  - molecule_nan_patterns_*.csv (per-molecule NaN patterns)")
    print(f"  - strategy_comparison_*.csv (strategy performance)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Feature Availability Analysis for Mordred Comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python analyze_feature_availability.py \\
    --kw_file KW-0808_Transferase_affinity_extracted_features.csv \\
    --target_accession P00519 \\
    --output_dir tests/mordred_full_feature_eval/feature_availability_analysis

This will analyze feature availability patterns for all three representations
and compare four feature selection strategies.
        """
    )
    
    # Required
    p.add_argument("--kw_file", required=True,
                   help="KW file name (e.g., 'KW-0808_Transferase_affinity_extracted_features.csv')")
    p.add_argument("--target_accession", required=True,
                   help="Target accession (e.g., 'P00519')")
    
    # Sampling
    p.add_argument("--n_mf", type=int, default=None,
                   help="Sample size for MF (default: all)")
    p.add_argument("--n_target", type=int, default=None,
                   help="Sample size for actives (default: all)")
    p.add_argument("--n_zinc", type=int, default=None,
                   help="Sample size for ZINC (default: all)")
    
    # Paths
    p.add_argument("--base_dir", default=".",
                   help="Base directory (repo root)")
    p.add_argument("--full_2d_dir",
                   default="/home/ahagg2s/UMMBAS_screening_experiments/output_recalculated_full_datasets/datasets_2d_all",
                   help="Directory with full 2D features")
    p.add_argument("--full_2d3d_dir",
                   default="/home/ahagg2s/UMMBAS_screening_experiments/output_recalculated_full_datasets/datasets_2d3d_all",
                   help="Directory with full 2D+3D features")
    
    # Output
    p.add_argument("--output_dir",
                   default="tests/mordred_full_feature_eval/feature_availability_analysis",
                   help="Output directory")
    
    # Misc
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed")
    
    # Test mode
    p.add_argument("--test", action="store_true",
                   help="Test mode: use minimal samples (n_mf=50, n_target=20, n_zinc=50)")
    
    args = p.parse_args()
    
    if args.test:
        args.n_mf = 50
        args.n_target = 20
        args.n_zinc = 50
        args.output_dir = "tests/mordred_full_feature_eval/feature_availability_analysis_test"
    
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
