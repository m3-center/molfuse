#!/usr/bin/env python3
"""
Aggregate UMAP initialization test results.

Collects results from all test runs and generates a comparison summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict

import pandas as pd


def load_results(results_dir: Path) -> List[Dict]:
    """Load all result JSON files."""
    results = []
    for json_file in sorted(results_dir.glob("*.json")):
        with json_file.open("r") as f:
            results.append(json.load(f))
    return results


def create_summary(results: List[Dict]) -> Dict:
    """Create comparison summary."""
    # Convert to DataFrame for easy manipulation
    df = pd.DataFrame(results)
    
    # Sort by method (PCA first, then UMAP by init)
    df["sort_key"] = df.apply(
        lambda row: (0 if row["method"] == "pca" else 1, row.get("init", ""))
        , axis=1
    )
    df = df.sort_values("sort_key").drop(columns=["sort_key"])
    
    # Create summary table
    summary_table = []
    for _, row in df.iterrows():
        if row["method"] == "pca":
            label = "PCA (baseline)"
        else:
            label = f"UMAP (init={row['init']})"
        
        summary_table.append({
            "configuration": label,
            "ef_1%": f"{row['ef_1%']:.2f}",
            "ef_5%": f"{row['ef_5%']:.2f}",
            "roc_auc": f"{row['roc_auc']:.4f}",
            "pr_auc": f"{row['pr_auc']:.4f}",
            "dr_time_s": f"{row['time_dr_s']:.1f}",
            "total_time_s": f"{row['time_total_s']:.1f}",
            "peak_memory_mb": f"{row['memory_peak_mb']:.0f}",
        })
    
    # Find best by EF@1%
    best_ef1_idx = df["ef_1%"].idxmax()
    best_ef1_row = df.loc[best_ef1_idx]
    if best_ef1_row["method"] == "pca":
        best_ef1_config = "PCA (baseline)"
    else:
        best_ef1_config = f"UMAP (init={best_ef1_row['init']})"
    
    # Find fastest DR
    fastest_dr_idx = df["time_dr_s"].idxmin()
    fastest_dr_row = df.loc[fastest_dr_idx]
    if fastest_dr_row["method"] == "pca":
        fastest_dr_config = "PCA (baseline)"
    else:
        fastest_dr_config = f"UMAP (init={fastest_dr_row['init']})"
    
    summary = {
        "test_name": "UMAP Initialization Method Comparison",
        "num_configs": len(results),
        "table": summary_table,
        "highlights": {
            "best_ef1%": {
                "configuration": best_ef1_config,
                "value": float(best_ef1_row["ef_1%"]),
            },
            "fastest_dr": {
                "configuration": fastest_dr_config,
                "time_s": float(fastest_dr_row["time_dr_s"]),
            },
        },
        "raw_results": results,
    }
    
    return summary


def print_summary_table(summary: Dict) -> None:
    """Print formatted summary table."""
    print("\n" + "="*100)
    print("UMAP INITIALIZATION METHOD COMPARISON - RESULTS")
    print("="*100)
    print(f"\n{'Configuration':<30} {'EF@1%':>8} {'EF@5%':>8} {'ROC-AUC':>10} {'PR-AUC':>10} {'DR Time':>10} {'Total Time':>12} {'Peak Mem':>12}")
    print("-"*100)
    
    for row in summary["table"]:
        print(f"{row['configuration']:<30} {row['ef_1%']:>8} {row['ef_5%']:>8} "
              f"{row['roc_auc']:>10} {row['pr_auc']:>10} {row['dr_time_s']:>9}s "
              f"{row['total_time_s']:>11}s {row['peak_memory_mb']:>11}M")
    
    print("-"*100)
    print(f"\n✓ Best EF@1%: {summary['highlights']['best_ef1%']['configuration']} "
          f"({summary['highlights']['best_ef1%']['value']:.2f})")
    print(f"✓ Fastest DR: {summary['highlights']['fastest_dr']['configuration']} "
          f"({summary['highlights']['fastest_dr']['time_s']:.1f}s)")
    print("="*100 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Aggregate UMAP init test results")
    parser.add_argument("--results_dir", type=str, required=True,
                       help="Directory containing result JSON files")
    parser.add_argument("--output", type=str, required=True,
                       help="Output path for comparison summary JSON")
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_path = Path(args.output)
    
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return 1
    
    # Load results
    results = load_results(results_dir)
    
    if not results:
        print(f"Error: No result files found in {results_dir}")
        return 1
    
    print(f"Loaded {len(results)} result files")
    
    # Create summary
    summary = create_summary(results)
    
    # Print table
    print_summary_table(summary)
    
    # Save JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Summary saved to {output_path}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
