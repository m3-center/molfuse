#!/usr/bin/env python3
"""
Phase 2 Post-Analysis: Affinity Cutoff Sensitivity Visualization

Aggregates Phase 2 results and generates publication-quality plots:
- Cutoff-EF curves (EF@1/5/10% vs cutoff)
- Heatmaps (method × cutoff for each metric)
- Quality-quantity scatter (MF cloud size vs EF@1%)
- Potency-tier stratified analysis (if actives have potency data)

Usage:
    python scripts/phase2_post_analysis.py \
        --workspace_dir experiment_workspace_v4 \
        --phase2_run_name cutoff_sweep \
        --output_dir reporting/phase2_post_analysis
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Publication-quality style
matplotlib.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.titleweight": "bold",
    "legend.fontsize": 9,
    "font.family": "sans-serif",
})

try:
    import seaborn as sns
    _HAVE_SNS = True
except ImportError:
    sns = None
    _HAVE_SNS = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 2 Post-Analysis: Cutoff Sensitivity")
    p.add_argument("--workspace_dir", type=str, required=True, help="Base workspace directory")
    p.add_argument("--phase2_run_name", type=str, default="cutoff_sweep", help="Phase 2 run name")
    p.add_argument("--output_dir", type=str, required=True, help="Output directory for plots and CSVs")
    p.add_argument("--potency_tiers", type=str, default="0-100:High,100-1000:Medium,1000-100000:Weak",
                   help="Potency tiers as range:label pairs (comma-separated)")
    return p.parse_args()


def save_figure(fig: plt.Figure, output_dir: Path, basename: str) -> None:
    """Save figure in PNG and PDF formats."""
    fig.savefig(output_dir / f"{basename}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{basename}.pdf", bbox_inches="tight")
    plt.close(fig)


def aggregate_phase2_results(phase2_dir: Path) -> pd.DataFrame:
    """
    Aggregate Phase 2 metrics across models and cutoffs.

    Returns DataFrame with columns: model_key, method, representation, dim, cutoff_nM, ef1, ef5, ef10, roc_auc, pr_auc, n_mf
    """
    rows: List[Dict] = []

    # Scan for model subdirectories
    for model_dir in phase2_dir.iterdir():
        if not model_dir.is_dir() or model_dir.name in ("logs", "artifacts", "metrics"):
            continue

        # Scan for cutoff subdirectories
        for cutoff_dir in model_dir.iterdir():
            if not cutoff_dir.is_dir() or not cutoff_dir.name.startswith("cutoff_"):
                continue

            metrics_path = cutoff_dir / "metrics.json"
            if not metrics_path.exists():
                continue

            try:
                with metrics_path.open("r") as f:
                    metrics = json.load(f)
            except Exception:
                continue

            rows.append({
                "model_key": f"{metrics.get('method', 'unknown')}_{metrics.get('representation', 'unknown')}",
                "method": metrics.get("method", "unknown"),
                "representation": metrics.get("representation", "unknown"),
                "dim": metrics.get("dim", 0),
                "cutoff_nM": metrics.get("affinity_cutoff_nM", 0),
                "ef1": metrics.get("ef_1%", 0.0),
                "ef5": metrics.get("ef_5%", 0.0),
                "ef10": metrics.get("ef_10%", 0.0),
                "roc_auc": metrics.get("roc_auc", 0.0),
                "pr_auc": metrics.get("pr_auc", 0.0),
                "n_mf": metrics.get("n_mf_for_scoring", 0),
                "phase1_run": metrics.get("phase1_run", "unknown"),
            })

    df = pd.DataFrame(rows)
    return df


def plot_cutoff_curves(df: pd.DataFrame, metric_col: str, output_dir: Path, basename: str) -> None:
    """
    Plot metric vs cutoff curves for each model.

    Args:
        df: Aggregated metrics DataFrame
        metric_col: Metric column to plot (e.g., "ef1", "ef5", "ef10")
        output_dir: Output directory
        basename: Base filename for saved plots
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each model as a separate line
    for model_key in df["model_key"].unique():
        subset = df[df["model_key"] == model_key].sort_values("cutoff_nM")
        ax.plot(subset["cutoff_nM"], subset[metric_col], marker="o", label=model_key, linewidth=2, markersize=6)

    ax.set_xscale("log")
    ax.set_xlabel("Affinity Cutoff (nM)", fontsize=12, fontweight="bold")
    ax.set_ylabel(metric_col.upper().replace("_", " "), fontsize=12, fontweight="bold")
    ax.set_title(f"Cutoff Sensitivity: {metric_col.upper()}", fontsize=14, fontweight="bold")
    ax.legend(title="Model", fontsize=9, title_fontsize=10)
    ax.grid(True, alpha=0.3)

    save_figure(fig, output_dir, basename)
    print(f"Saved {basename}.png/pdf")


def plot_cutoff_heatmap(df: pd.DataFrame, metric_col: str, output_dir: Path, basename: str) -> None:
    """
    Plot heatmap of metric across models (rows) and cutoffs (columns).
    """
    # Pivot table: rows = model_key, columns = cutoff_nM
    pivot = df.pivot_table(index="model_key", columns="cutoff_nM", values=metric_col, aggfunc="mean")

    fig, ax = plt.subplots(figsize=(8, 5))

    if _HAVE_SNS:
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax, cbar_kws={"label": metric_col.upper()})
    else:
        im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{int(c)}" for c in pivot.columns], rotation=45)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, rotation=0)
        fig.colorbar(im, ax=ax, label=metric_col.upper())

        # Annotate cells
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.iloc[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black", fontsize=9)

    ax.set_xlabel("Cutoff (nM)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Model", fontsize=12, fontweight="bold")
    ax.set_title(f"Cutoff Heatmap: {metric_col.upper()}", fontsize=14, fontweight="bold")

    save_figure(fig, output_dir, basename)
    print(f"Saved {basename}.png/pdf")


def plot_quality_quantity(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Scatter plot: MF cloud size (x) vs EF@1% (y), colored by method/representation.
    Shows quality-quantity trade-off.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for model_key in df["model_key"].unique():
        subset = df[df["model_key"] == model_key]
        ax.scatter(subset["n_mf"], subset["ef1"], label=model_key, s=80, alpha=0.7)

    ax.set_xlabel("MF Cloud Size (compounds)", fontsize=12, fontweight="bold")
    ax.set_ylabel("EF@1%", fontsize=12, fontweight="bold")
    ax.set_title("Quality-Quantity Trade-off: MF Size vs EF@1%", fontsize=14, fontweight="bold")
    ax.legend(title="Model", fontsize=9, title_fontsize=10)
    ax.grid(True, alpha=0.3)

    save_figure(fig, output_dir, "quality_quantity_ef1")
    print("Saved quality_quantity_ef1.png/pdf")


def identify_best_cutoffs(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Identify best cutoff per model based on EF@1%.
    Save as JSON.
    """
    best_cutoffs: Dict[str, Dict] = {}

    for model_key in df["model_key"].unique():
        subset = df[df["model_key"] == model_key]
        best_row = subset.loc[subset["ef1"].idxmax()]

        best_cutoffs[model_key] = {
            "cutoff_nM": int(best_row["cutoff_nM"]),
            "ef1": float(best_row["ef1"]),
            "ef5": float(best_row["ef5"]),
            "ef10": float(best_row["ef10"]),
            "roc_auc": float(best_row["roc_auc"]),
            "pr_auc": float(best_row["pr_auc"]),
            "n_mf": int(best_row["n_mf"]),
        }

    best_cutoffs_path = output_dir / "phase2_best_cutoffs.json"
    with best_cutoffs_path.open("w") as f:
        json.dump(best_cutoffs, f, indent=2)

    print(f"\nBest cutoffs per model (saved to {best_cutoffs_path}):")
    for model_key, info in best_cutoffs.items():
        print(f"  {model_key}: {info['cutoff_nM']} nM (EF@1% = {info['ef1']:.2f})")


def main() -> None:
    args = parse_args()

    workspace_dir = Path(args.workspace_dir)
    phase2_run_name = args.phase2_run_name
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    phase2_dir = workspace_dir / "phase2" / phase2_run_name

    if not phase2_dir.exists():
        print(f"ERROR: Phase 2 directory not found: {phase2_dir}")
        return

    print("="*80)
    print("PHASE 2 POST-ANALYSIS: Affinity Cutoff Sensitivity")
    print("="*80)
    print(f"Workspace: {workspace_dir}")
    print(f"Phase 2 run: {phase2_run_name}")
    print(f"Output: {output_dir}")
    print()

    # Aggregate results
    print("Aggregating Phase 2 results...")
    df = aggregate_phase2_results(phase2_dir)

    if df.empty:
        print("ERROR: No Phase 2 results found. Check workspace directory.")
        return

    print(f"Found {len(df)} results across {df['model_key'].nunique()} models and {df['cutoff_nM'].nunique()} cutoffs")
    print()

    # Save aggregated CSV
    df.to_csv(output_dir / "phase2_aggregated_metrics.csv", index=False)
    print(f"Saved aggregated metrics: {output_dir / 'phase2_aggregated_metrics.csv'}")
    print()

    # Generate plots
    print("Generating plots...")

    # Cutoff curves
    plot_cutoff_curves(df, "ef1", output_dir, "cutoff_curves_ef1")
    plot_cutoff_curves(df, "ef5", output_dir, "cutoff_curves_ef5")
    plot_cutoff_curves(df, "ef10", output_dir, "cutoff_curves_ef10")

    # Heatmaps
    plot_cutoff_heatmap(df, "ef1", output_dir, "cutoff_heatmap_ef1")
    plot_cutoff_heatmap(df, "roc_auc", output_dir, "cutoff_heatmap_roc_auc")

    # Quality-quantity
    plot_quality_quantity(df, output_dir)

    # Identify best cutoffs
    identify_best_cutoffs(df, output_dir)

    print()
    print("="*80)
    print("PHASE 2 POST-ANALYSIS COMPLETED")
    print("="*80)
    print(f"All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
