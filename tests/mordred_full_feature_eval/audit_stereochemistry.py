#!/usr/bin/env python3
"""
Audit stereochemistry prevalence in ZINC, MF, and ACTIVES molecular datasets.

Analyzes SMILES strings to detect:
1. Double bond stereochemistry: / and \\ characters
2. Tetrahedral stereochemistry: @ and @@ characters
3. Prevalence by molecular source (ZINC, MF cloud, target actives)

Usage:
    python audit_stereochemistry.py
    python audit_stereochemistry.py --output_dir ./audit_results
"""

import argparse
import re
from pathlib import Path
from typing import Dict, Tuple, List
import pandas as pd
import numpy as np
from tqdm import tqdm


def detect_stereochemistry(smiles: str) -> Tuple[bool, bool, int, int]:
    """
    Detect stereochemistry markers in a SMILES string.
    
    Returns:
        (has_double_bond_stereo, has_chiral_center, n_double_bond_stereo, n_chiral_centers)
    """
    if not isinstance(smiles, str):
        return False, False, 0, 0
    
    # Count double bond stereochemistry markers (/ and \)
    # These appear adjacent to double bonds: C/C=C\C or C\C=C/C
    n_slash = smiles.count('/')
    n_backslash = smiles.count('\\')
    n_double_bond_stereo = n_slash + n_backslash
    has_double_bond_stereo = n_double_bond_stereo > 0
    
    # Count chiral center markers (@ and @@)
    # Pattern: [C@H] or [C@@H] etc.
    chiral_pattern = re.compile(r'@+')
    chiral_matches = chiral_pattern.findall(smiles)
    n_chiral_centers = len(chiral_matches)
    has_chiral_center = n_chiral_centers > 0
    
    return has_double_bond_stereo, has_chiral_center, n_double_bond_stereo, n_chiral_centers


def infer_smiles_column(df: pd.DataFrame) -> str:
    """Infer which column contains SMILES strings."""
    possible_cols = ["smiles", "SMILES", "canonical_smiles", "canonicalSmiles", 
                     "mol_smiles", "can_smiles", "Smiles"]
    
    for col in possible_cols:
        if col in df.columns:
            return col
    
    # Fallback: first object dtype column that looks like SMILES
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().astype(str).head(50)
            if sample.apply(lambda s: any(ch in s for ch in ["C", "N", "O", "=", "#"]) and len(s) <= 300).mean() > 0.5:
                return col
    
    raise ValueError("Could not infer SMILES column")


def load_and_audit_file(file_path: Path, source_label: str, max_rows: int = None) -> Dict:
    """Load a CSV file and audit its stereochemistry content."""
    print(f"\n{'='*80}")
    print(f"Auditing: {source_label}")
    print(f"File: {file_path}")
    print(f"{'='*80}")
    
    if not file_path.exists():
        print(f"⚠ File not found: {file_path}")
        return None
    
    # Load file (potentially in chunks for large files)
    try:
        if max_rows:
            df = pd.read_csv(file_path, nrows=max_rows)
        else:
            # Try loading entire file, fall back to chunked reading if too large
            try:
                df = pd.read_csv(file_path)
            except Exception:
                print(f"  File too large, reading in chunks...")
                chunks = []
                for chunk in pd.read_csv(file_path, chunksize=50000):
                    chunks.append(chunk)
                    if max_rows and sum(len(c) for c in chunks) >= max_rows:
                        break
                df = pd.concat(chunks, ignore_index=True)
                if max_rows:
                    df = df.head(max_rows)
    except Exception as e:
        print(f"✗ Error loading file: {e}")
        return None
    
    # Infer SMILES column
    try:
        smiles_col = infer_smiles_column(df)
    except ValueError as e:
        print(f"✗ {e}")
        return None
    
    print(f"  SMILES column: '{smiles_col}'")
    print(f"  Total rows: {len(df):,}")
    
    # Clean SMILES
    smiles_series = df[smiles_col].dropna().astype(str)
    smiles_series = smiles_series[smiles_series.str.len() > 0]
    print(f"  Valid SMILES: {len(smiles_series):,}")
    
    # Audit stereochemistry
    results = []
    for smiles in tqdm(smiles_series, desc=f"  Analyzing {source_label}", leave=False):
        has_db, has_chiral, n_db, n_chiral = detect_stereochemistry(smiles)
        results.append({
            'smiles': smiles,
            'has_double_bond_stereo': has_db,
            'has_chiral_center': has_chiral,
            'n_double_bond_stereo': n_db,
            'n_chiral_centers': n_chiral,
            'has_any_stereo': has_db or has_chiral,
        })
    
    results_df = pd.DataFrame(results)
    
    # Compute summary statistics
    n_total = len(results_df)
    n_with_db_stereo = results_df['has_double_bond_stereo'].sum()
    n_with_chiral = results_df['has_chiral_center'].sum()
    n_with_any = results_df['has_any_stereo'].sum()
    
    pct_db_stereo = 100 * n_with_db_stereo / n_total if n_total > 0 else 0
    pct_chiral = 100 * n_with_chiral / n_total if n_total > 0 else 0
    pct_any = 100 * n_with_any / n_total if n_total > 0 else 0
    
    print(f"\n  SUMMARY:")
    print(f"  {'─'*76}")
    print(f"  Molecules with double bond stereochemistry:  {n_with_db_stereo:>8,} / {n_total:>8,}  ({pct_db_stereo:>5.2f}%)")
    print(f"  Molecules with chiral centers:               {n_with_chiral:>8,} / {n_total:>8,}  ({pct_chiral:>5.2f}%)")
    print(f"  Molecules with ANY stereochemistry:          {n_with_any:>8,} / {n_total:>8,}  ({pct_any:>5.2f}%)")
    
    # Additional statistics for molecules WITH stereochemistry
    if n_with_any > 0:
        stereo_subset = results_df[results_df['has_any_stereo']]
        avg_db = stereo_subset['n_double_bond_stereo'].mean()
        avg_chiral = stereo_subset['n_chiral_centers'].mean()
        max_db = stereo_subset['n_double_bond_stereo'].max()
        max_chiral = stereo_subset['n_chiral_centers'].max()
        
        print(f"\n  For molecules WITH stereochemistry:")
        print(f"    Avg double bond stereo markers: {avg_db:.2f}  (max: {max_db})")
        print(f"    Avg chiral centers: {avg_chiral:.2f}  (max: {max_chiral})")
    
    return {
        'source': source_label,
        'file_path': str(file_path),
        'n_total': n_total,
        'n_with_double_bond_stereo': n_with_db_stereo,
        'n_with_chiral_center': n_with_chiral,
        'n_with_any_stereo': n_with_any,
        'pct_double_bond_stereo': pct_db_stereo,
        'pct_chiral': pct_chiral,
        'pct_any_stereo': pct_any,
        'results_df': results_df,
    }


def load_mf_cloud(base_dir: Path, max_files: int = None, max_per_file: int = None) -> Dict:
    """Load and audit MF cloud from multiple KW files."""
    print(f"\n{'='*80}")
    print(f"Auditing: MF CLOUD (Molecular Function Affinity Data)")
    print(f"Directory: {base_dir}")
    print(f"{'='*80}")
    
    if not base_dir.exists():
        print(f"⚠ Directory not found: {base_dir}")
        return None
    
    kw_files = sorted([p for p in base_dir.glob("*.csv") if p.is_file()])
    if max_files:
        kw_files = kw_files[:max_files]
    
    print(f"  Found {len(kw_files)} KW files")
    
    all_results = []
    for i, kw_file in enumerate(kw_files, 1):
        print(f"\n  [{i}/{len(kw_files)}] Processing: {kw_file.name}")
        try:
            df = pd.read_csv(kw_file)
            if max_per_file:
                df = df.head(max_per_file)
            
            smiles_col = infer_smiles_column(df)
            smiles_series = df[smiles_col].dropna().astype(str)
            smiles_series = smiles_series[smiles_series.str.len() > 0]
            
            print(f"    Valid SMILES: {len(smiles_series):,}")
            
            for smiles in smiles_series:
                has_db, has_chiral, n_db, n_chiral = detect_stereochemistry(smiles)
                all_results.append({
                    'smiles': smiles,
                    'kw_file': kw_file.name,
                    'has_double_bond_stereo': has_db,
                    'has_chiral_center': has_chiral,
                    'n_double_bond_stereo': n_db,
                    'n_chiral_centers': n_chiral,
                    'has_any_stereo': has_db or has_chiral,
                })
        except Exception as e:
            print(f"    ✗ Error: {e}")
            continue
    
    if not all_results:
        print("  ✗ No valid results")
        return None
    
    results_df = pd.DataFrame(all_results)
    
    # Remove duplicates (same SMILES may appear in multiple KW files)
    n_before = len(results_df)
    results_df = results_df.drop_duplicates(subset=['smiles'], keep='first')
    n_after = len(results_df)
    
    print(f"\n  Removed {n_before - n_after:,} duplicate SMILES across files")
    
    # Compute summary statistics
    n_total = len(results_df)
    n_with_db_stereo = results_df['has_double_bond_stereo'].sum()
    n_with_chiral = results_df['has_chiral_center'].sum()
    n_with_any = results_df['has_any_stereo'].sum()
    
    pct_db_stereo = 100 * n_with_db_stereo / n_total if n_total > 0 else 0
    pct_chiral = 100 * n_with_chiral / n_total if n_total > 0 else 0
    pct_any = 100 * n_with_any / n_total if n_total > 0 else 0
    
    print(f"\n  SUMMARY (deduplicated):")
    print(f"  {'─'*76}")
    print(f"  Unique molecules: {n_total:,}")
    print(f"  Molecules with double bond stereochemistry:  {n_with_db_stereo:>8,} / {n_total:>8,}  ({pct_db_stereo:>5.2f}%)")
    print(f"  Molecules with chiral centers:               {n_with_chiral:>8,} / {n_total:>8,}  ({pct_chiral:>5.2f}%)")
    print(f"  Molecules with ANY stereochemistry:          {n_with_any:>8,} / {n_total:>8,}  ({pct_any:>5.2f}%)")
    
    return {
        'source': 'MF_CLOUD',
        'n_total': n_total,
        'n_with_double_bond_stereo': n_with_db_stereo,
        'n_with_chiral_center': n_with_chiral,
        'n_with_any_stereo': n_with_any,
        'pct_double_bond_stereo': pct_db_stereo,
        'pct_chiral': pct_chiral,
        'pct_any_stereo': pct_any,
        'results_df': results_df,
    }


def generate_report(results: List[Dict], output_dir: Path) -> None:
    """Generate comprehensive audit report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("GENERATING REPORT")
    print(f"{'='*80}")
    
    # 1) Summary table
    summary_rows = []
    for r in results:
        if r is None:
            continue
        summary_rows.append({
            'Source': r['source'],
            'Total Molecules': r['n_total'],
            'With DB Stereo': r['n_with_double_bond_stereo'],
            'With Chiral': r['n_with_chiral_center'],
            'With Any Stereo': r['n_with_any_stereo'],
            '% DB Stereo': r['pct_double_bond_stereo'],
            '% Chiral': r['pct_chiral'],
            '% Any Stereo': r['pct_any_stereo'],
        })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "stereochemistry_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"  ✓ Saved summary table: {summary_path}")
    
    # 2) Detailed results per source
    for r in results:
        if r is None or 'results_df' not in r:
            continue
        source_clean = r['source'].lower().replace(' ', '_')
        detail_path = output_dir / f"stereochemistry_details_{source_clean}.csv"
        r['results_df'].to_csv(detail_path, index=False)
        print(f"  ✓ Saved detailed results: {detail_path}")
    
    # 3) Text report
    report_path = output_dir / "stereochemistry_audit_report.txt"
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("STEREOCHEMISTRY AUDIT REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        for r in results:
            if r is None:
                continue
            f.write(f"\n{r['source']}\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Total molecules: {r['n_total']:,}\n")
            f.write(f"  With double bond stereochemistry: {r['n_with_double_bond_stereo']:,} ({r['pct_double_bond_stereo']:.2f}%)\n")
            f.write(f"  With chiral centers: {r['n_with_chiral_center']:,} ({r['pct_chiral']:.2f}%)\n")
            f.write(f"  With ANY stereochemistry: {r['n_with_any_stereo']:,} ({r['pct_any_stereo']:.2f}%)\n")
            f.write("\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("INTERPRETATION\n")
        f.write("=" * 80 + "\n\n")
        
        avg_stereo = np.mean([r['pct_any_stereo'] for r in results if r is not None])
        
        if avg_stereo < 5:
            f.write("⚠ LOW STEREOCHEMISTRY PREVALENCE (<5%)\n\n")
            f.write("Recommendation: 3D Mordred descriptors are NOT justified.\n")
            f.write("  • Current 2D approach is optimal for this chemical space\n")
            f.write("  • Stereochemistry is not a dominant factor\n")
            f.write("  • Focus on 2D feature engineering and representation optimization\n")
        elif avg_stereo < 20:
            f.write("⚠ MODERATE STEREOCHEMISTRY PREVALENCE (5-20%)\n\n")
            f.write("Recommendation: Consider lightweight stereochemistry features.\n")
            f.write("  • Add boolean flags: has_stereo, n_chiral_centers, n_db_stereo\n")
            f.write("  • These are cheap to compute (regex on SMILES)\n")
            f.write("  • Avoid expensive 3D conformer generation\n")
        else:
            f.write("✓ HIGH STEREOCHEMISTRY PREVALENCE (>20%)\n\n")
            f.write("Recommendation: Stereochemistry may be important.\n")
            f.write("  • Consider 3D Mordred descriptors for high-value screening\n")
            f.write("  • Use as refinement step after 2D pre-filtering\n")
            f.write("  • Test if stratified evaluation shows EF improvement on stereo-subset\n")
    
    print(f"  ✓ Saved text report: {report_path}")
    
    # 4) Print summary to console
    print(f"\n{'='*80}")
    print("AUDIT SUMMARY")
    print(f"{'='*80}\n")
    print(summary_df.to_string(index=False))
    print()


def main():
    parser = argparse.ArgumentParser(description="Audit stereochemistry prevalence in molecular datasets")
    parser.add_argument("--base_dir", type=str, default=None, 
                        help="Base directory (defaults to repo root)")
    parser.add_argument("--output_dir", type=str, default="tests/mordred_full_feature_eval/audit_results",
                        help="Output directory for audit results")
    parser.add_argument("--zinc_csv", type=str, default=None,
                        help="Path to ZINC CSV (default: datasets/zinc_data.csv)")
    parser.add_argument("--actives_csv", type=str, default=None,
                        help="Path to target actives CSV (default: first KW file)")
    parser.add_argument("--max_zinc", type=int, default=None,
                        help="Maximum ZINC molecules to analyze (default: all)")
    parser.add_argument("--max_mf_files", type=int, default=None,
                        help="Maximum MF KW files to process (default: all)")
    parser.add_argument("--max_per_mf_file", type=int, default=None,
                        help="Maximum molecules per MF file (default: all)")
    
    args = parser.parse_args()
    
    # Determine base directory
    if args.base_dir:
        base_dir = Path(args.base_dir).resolve()
    else:
        base_dir = Path(__file__).resolve().parents[2]
    
    datasets_dir = base_dir / "datasets"
    output_dir = Path(args.output_dir).resolve()
    
    print(f"\n{'='*80}")
    print("STEREOCHEMISTRY AUDIT")
    print(f"{'='*80}")
    print(f"Base directory: {base_dir}")
    print(f"Datasets directory: {datasets_dir}")
    print(f"Output directory: {output_dir}")
    
    results = []
    
    # 1) Audit ZINC
    zinc_path = Path(args.zinc_csv).resolve() if args.zinc_csv else (datasets_dir / "zinc_data.csv")
    zinc_result = load_and_audit_file(zinc_path, "ZINC", max_rows=args.max_zinc)
    results.append(zinc_result)
    
    # 2) Audit MF Cloud
    mf_dir = datasets_dir / "molecular_function_affinity_data"
    mf_result = load_mf_cloud(mf_dir, max_files=args.max_mf_files, max_per_file=args.max_per_mf_file)
    results.append(mf_result)
    
    # 3) Audit Target Actives
    if args.actives_csv:
        actives_path = Path(args.actives_csv).resolve()
    else:
        # Default: first KW file
        kw_files = sorted([p for p in mf_dir.glob("*.csv") if p.is_file()])
        actives_path = kw_files[0] if kw_files else None
    
    if actives_path:
        actives_result = load_and_audit_file(actives_path, "TARGET_ACTIVES", max_rows=None)
        results.append(actives_result)
    
    # 4) Generate report
    generate_report(results, output_dir)
    
    print(f"\n{'='*80}")
    print("AUDIT COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {output_dir}")
    print()


if __name__ == "__main__":
    main()
