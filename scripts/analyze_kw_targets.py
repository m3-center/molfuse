#!/usr/bin/env python3
"""
Analyze KW category files to identify unique target proteins.

For each KW category, this script:
1. Reads the MF affinity dataset
2. Extracts unique target proteins (via 'accession' column)
3. Counts compounds per target
4. Sorts by compound count (descending)

This helps select specific target proteins for Phase 4 cross-target study.

Usage:
    python scripts/analyze_kw_targets.py

Output:
    - Prints summary table for each KW category
    - Saves detailed report to: reporting/phase4_target_selection.txt
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


# KW categories of interest for Phase 4
KW_CATEGORIES = [
    ("KW-0049_Antioxidant", 83),
    ("KW-0929_Antimicrobial", 870),
    ("KW-0505_Motor_protein", 2445),
    ("KW-0202_Cytokine", 5812),
    ("KW-0358_Heparin-binding", 11765),
    ("KW-0456_Lyase", 47371),
    ("KW-0560_Oxidoreductase", 100769),
    ("KW-0808_Transferase", 430795),
]

BASE_DATA_DIR = Path("output_recalculated_full_datasets/datasets_2d_all")


def analyze_kw_category(kw_name: str, expected_total: int) -> pd.DataFrame:
    """
    Analyze a single KW category file and return target protein summary.
    
    Args:
        kw_name: KW category name (e.g., "KW-0049_Antioxidant")
        expected_total: Expected total compounds in category
    
    Returns:
        DataFrame with columns: [accession, gene_name, n_compounds, pct_of_category]
    """
    csv_path = BASE_DATA_DIR / f"{kw_name}_affinity_extracted_features.csv"
    
    if not csv_path.exists():
        print(f"  ⚠️  File not found: {csv_path}")
        return pd.DataFrame()
    
    # Read only metadata columns to save memory
    try:
        df = pd.read_csv(
            csv_path,
            usecols=lambda c: c in ["accession", "Compound ChEMBL ID", "Standard Value (nM)"],
            low_memory=False
        )
    except Exception as e:
        print(f"  ⚠️  Error reading {csv_path.name}: {e}")
        return pd.DataFrame()
    
    # Count compounds per target (accession)
    target_counts = df.groupby("accession").size().reset_index(name="n_compounds")
    target_counts = target_counts.sort_values("n_compounds", ascending=False)
    
    # Add percentage
    target_counts["pct_of_category"] = (target_counts["n_compounds"] / len(df)) * 100
    
    # Verify total
    actual_total = len(df)
    if actual_total != expected_total:
        print(f"  ⚠️  Total mismatch: expected {expected_total:,}, got {actual_total:,}")
    
    return target_counts


def format_target_summary(df: pd.DataFrame, top_n: int = 10) -> str:
    """Format target summary as readable table."""
    if len(df) == 0:
        return "  No data available"
    
    lines = []
    lines.append(f"  {'Rank':<6} {'Accession':<12} {'Compounds':>10} {'% of Category':>12}")
    lines.append("  " + "-" * 50)
    
    for idx, row in df.head(top_n).iterrows():
        rank = idx + 1 if isinstance(idx, int) else "?"
        lines.append(
            f"  {rank:<6} {row['accession']:<12} {row['n_compounds']:>10,} "
            f"{row['pct_of_category']:>11.1f}%"
        )
    
    if len(df) > top_n:
        lines.append(f"  ... and {len(df) - top_n} more targets")
    
    return "\n".join(lines)


def main():
    print("="*80)
    print("PHASE 4 TARGET SELECTION: KW Category Analysis")
    print("="*80)
    print(f"Base directory: {BASE_DATA_DIR}")
    print(f"Analyzing {len(KW_CATEGORIES)} KW categories...")
    print("="*80)
    print()
    
    results = {}
    report_lines = []
    
    for kw_name, expected_total in KW_CATEGORIES:
        kw_short = kw_name.split("_", 1)[1]
        print(f"📊 {kw_short} ({kw_name})")
        print(f"   Expected total: {expected_total:,} compounds")
        
        df_targets = analyze_kw_category(kw_name, expected_total)
        
        if len(df_targets) > 0:
            n_targets = len(df_targets)
            top_target = df_targets.iloc[0]
            print(f"   Unique targets: {n_targets}")
            print(f"   Top target: {top_target['accession']} ({top_target['n_compounds']:,} compounds, {top_target['pct_of_category']:.1f}%)")
            print()
            print(format_target_summary(df_targets, top_n=10))
            
            results[kw_name] = df_targets
            
            # Add to report
            report_lines.append("="*80)
            report_lines.append(f"{kw_short} ({kw_name})")
            report_lines.append(f"Total: {expected_total:,} | Unique targets: {n_targets}")
            report_lines.append("="*80)
            report_lines.append(format_target_summary(df_targets, top_n=20))
            report_lines.append("")
        else:
            print(f"   ⚠️  No data available")
            print()
        
        print()
    
    # Save detailed report
    output_dir = Path("reporting")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = output_dir / "phase4_target_selection.txt"
    with report_path.open("w") as f:
        f.write("PHASE 4 TARGET SELECTION REPORT\n")
        f.write(f"Generated: {pd.Timestamp.now()}\n")
        f.write("="*80 + "\n\n")
        f.write("\n".join(report_lines))
    
    print("="*80)
    print(f"✅ Report saved to: {report_path}")
    print("="*80)
    print()
    print("Next steps:")
    print("  1. Review top targets in each category")
    print("  2. Select one representative target per category")
    print("  3. Update TARGETS list in scripts/generate_molfuse_phase4_configs_v4.py")
    print("  4. Criteria for selection:")
    print("     - Sufficient compounds (ideally >50 for meaningful evaluation)")
    print("     - Well-studied targets (for biological interpretability)")
    print("     - Diversity across protein families")
    print()
    
    # Print quick summary for copy-paste
    print("="*80)
    print("SUGGESTED TARGETS (Top target from each category):")
    print("="*80)
    for kw_name, _ in KW_CATEGORIES:
        if kw_name in results:
            df = results[kw_name]
            top = df.iloc[0]
            kw_short = kw_name.split("_", 1)[1]
            print(f"  {kw_short:20s} → {top['accession']:12s} ({top['n_compounds']:>6,} compounds)")
    print()


if __name__ == "__main__":
    main()
