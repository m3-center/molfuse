#!/usr/bin/env python3
"""
Active Rank Distribution Analysis

Diagnoses the EF1% vs BEDROC discrepancy by analyzing where active compounds
are ranked in the scored lists.

Scientific Question:
    If EF1% is high (20-50) but BEDROC is moderate (~0.55), we expect a
    bimodal distribution: some actives ranked very early (driving EF1%),
    others ranked poorly (dragging down BEDROC).

Usage:
    python scripts/analyze_active_rank_distribution.py \
        --workspace_dir experiment_workspace_v4 \
        --phase phase1 \
        --output_dir reporting/rank_analysis

Output:
    - active_rank_summary.csv: Per-run statistics
    - rank_distribution_aggregate.png: Histogram + CDF of all active ranks
    - bimodality_diagnostic.png: Scatter plot (top1% vs bottom50%)
    - ef1_vs_top1pct.png: Sanity check
    - distribution_by_dr_method.png: Breakdown by DR method
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Plotting (headless safe)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Publication-quality styling
matplotlib.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.titleweight": "bold",
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "font.family": "sans-serif",
})


# ============================================================================
# LOGGING
# ============================================================================

def setup_logger(output_dir: Path) -> logging.Logger:
    """Create logger with file and console handlers."""
    logger = logging.getLogger("active_rank_analysis")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    fh = logging.FileHandler(output_dir / "rank_analysis.log", mode="w")
    sh = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    
    return logger


# ============================================================================
# DATA LOADING
# ============================================================================

def find_runs(workspace: Path, phase: str) -> List[Path]:
    """Find all run directories with ranked_scores.csv."""
    phase_dir = workspace / phase
    if not phase_dir.exists():
        return []
    
    runs = []
    for run_dir in phase_dir.iterdir():
        if run_dir.is_dir():
            ranked_path = run_dir / "artifacts" / "ranked_scores.csv"
            if ranked_path.exists():
                runs.append(run_dir)
    
    return sorted(runs)


def load_ranked_scores(run_dir: Path) -> Optional[pd.DataFrame]:
    """Load and sort ranked_scores.csv."""
    ranked_path = run_dir / "artifacts" / "ranked_scores.csv"
    if not ranked_path.exists():
        return None
    
    try:
        df = pd.read_csv(ranked_path, low_memory=False)
        
        # Ensure proper sorting (best first)
        if "score" in df.columns:
            df = df.sort_values(by="score", ascending=False, kind="stable").reset_index(drop=True)
        elif "distance" in df.columns:
            df = df.sort_values(by="distance", ascending=True, kind="stable").reset_index(drop=True)
        else:
            return None
        
        return df
    except Exception:
        return None


def extract_run_config(run_dir: Path) -> Dict:
    """Extract run configuration from summary and metrics JSON."""
    summary_path = run_dir / "logs" / "phase1_summary.json"
    metrics_path = run_dir / "metrics" / "metrics.json"
    
    config: Dict = {}
    
    if summary_path.exists():
        try:
            with open(summary_path) as f:
                summary = json.load(f)
                cfg = summary.get("config", {})
                config["dr_method"] = cfg.get("dr_method", "unknown")
                config["representation"] = cfg.get("representation", "unknown")
                if cfg.get("dr_method") == "umap":
                    umap_cfg = cfg.get("umap", {})
                    config["n_neighbors"] = umap_cfg.get("n_neighbors")
                    config["min_dist"] = umap_cfg.get("min_dist")
                    config["metric"] = umap_cfg.get("metric")
        except Exception:
            pass
    
    if metrics_path.exists():
        try:
            with open(metrics_path) as f:
                metrics = json.load(f)
                config["target"] = metrics.get("target", "unknown")
                config["EF1"] = metrics.get("EF@1%")
                config["EF5"] = metrics.get("EF@5%")
                config["ROC_AUC"] = metrics.get("ROC-AUC")
        except Exception:
            pass
    
    return config


# ============================================================================
# RANK STATISTICS
# ============================================================================

def compute_rank_statistics(ranked_df: pd.DataFrame) -> Dict:
    """Compute rank distribution statistics for actives."""
    N_total = len(ranked_df)
    
    # Get active indices (ranks are 0-indexed after sorting)
    active_mask = ranked_df["source"] == "actives"
    active_ranks = np.where(active_mask)[0]
    
    N_actives = len(active_ranks)
    if N_actives == 0:
        return {"N_total": N_total, "N_actives": 0}
    
    # Convert to percentile (0-100)
    active_percentiles = (active_ranks / N_total) * 100
    
    stats = {
        "N_total": N_total,
        "N_actives": N_actives,
        "rank_min": int(active_ranks.min()),
        "rank_max": int(active_ranks.max()),
        "rank_mean": float(active_ranks.mean()),
        "rank_median": float(np.median(active_ranks)),
        "rank_std": float(active_ranks.std()),
        "percentile_min": float(active_percentiles.min()),
        "percentile_max": float(active_percentiles.max()),
        "percentile_mean": float(active_percentiles.mean()),
        "percentile_median": float(np.median(active_percentiles)),
        "percentile_std": float(active_percentiles.std()),
        # Key diagnostic: fraction of actives in top X%
        "frac_in_top1pct": float((active_percentiles <= 1.0).sum() / N_actives),
        "frac_in_top5pct": float((active_percentiles <= 5.0).sum() / N_actives),
        "frac_in_top10pct": float((active_percentiles <= 10.0).sum() / N_actives),
        # Bimodality indicator: fraction in bottom 50%
        "frac_in_bottom50pct": float((active_percentiles >= 50.0).sum() / N_actives),
        # Keep arrays for plotting
        "active_ranks": active_ranks,
        "active_percentiles": active_percentiles,
    }
    
    return stats


# ============================================================================
# PLOTTING
# ============================================================================

def plot_aggregate_distribution(all_percentiles: List[float], output_dir: Path, logger: logging.Logger) -> None:
    """Plot aggregate percentile distribution (histogram + CDF)."""
    if not all_percentiles:
        logger.warning("No percentiles to plot for aggregate distribution")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Histogram
    ax1 = axes[0]
    ax1.hist(all_percentiles, bins=100, edgecolor="black", alpha=0.7, color="steelblue")
    ax1.axvline(x=1.0, color="red", linestyle="--", linewidth=2, label="Top 1%")
    ax1.axvline(x=5.0, color="orange", linestyle="--", linewidth=2, label="Top 5%")
    ax1.axvline(x=50.0, color="green", linestyle="--", linewidth=2, label="Median")
    ax1.set_xlabel("Percentile Rank (%)")
    ax1.set_ylabel("Count (actives)")
    ax1.set_title("Distribution of Active Compound Ranks (All Runs)")
    ax1.legend()
    
    # Right: CDF
    ax2 = axes[1]
    sorted_pct = np.sort(all_percentiles)
    cdf = np.arange(1, len(sorted_pct) + 1) / len(sorted_pct)
    ax2.plot(sorted_pct, cdf, linewidth=2, color="steelblue")
    ax2.axvline(x=1.0, color="red", linestyle="--", linewidth=1.5, label="Top 1%")
    ax2.axvline(x=5.0, color="orange", linestyle="--", linewidth=1.5, label="Top 5%")
    ax2.axhline(y=0.5, color="gray", linestyle=":", linewidth=1, label="50% actives")
    ax2.set_xlabel("Percentile Rank (%)")
    ax2.set_ylabel("Cumulative Fraction of Actives")
    ax2.set_title("CDF: How Many Actives Are Found by Each Cutoff?")
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 1)
    ax2.legend(loc="lower right")
    
    plt.tight_layout()
    fig_path = output_dir / "rank_distribution_aggregate.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {fig_path}")


def plot_bimodality_diagnostic(df_results: pd.DataFrame, output_dir: Path, logger: logging.Logger) -> None:
    """Plot bimodality diagnostic (top1% vs bottom50%)."""
    if df_results.empty:
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x = df_results["frac_in_top1pct"] * 100
    y = df_results["frac_in_bottom50pct"] * 100
    
    ax.scatter(x, y, alpha=0.5, s=30, c="steelblue", edgecolors="black", linewidths=0.5)
    
    ax.set_xlabel("% of Actives in Top 1% (drives EF1%)")
    ax.set_ylabel("% of Actives in Bottom 50% (drags down BEDROC)")
    ax.set_title("Bimodality Diagnostic: Early Hits vs. Missed Actives")
    
    # Annotate quadrants
    ax.axhline(y=25, color="red", linestyle="--", alpha=0.5)
    ax.axvline(x=25, color="green", linestyle="--", alpha=0.5)
    ax.text(50, 5, "Good: Many early, few late", fontsize=9, color="green", ha="center")
    ax.text(50, 45, "Problem: Many early AND many late\n(bimodal = EF high, BEDROC moderate)", 
            fontsize=9, color="orange", ha="center")
    ax.text(10, 45, "Poor: Few early, many late", fontsize=9, color="red", ha="center")
    
    plt.tight_layout()
    fig_path = output_dir / "bimodality_diagnostic.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {fig_path}")


def plot_ef1_sanity_check(df_results: pd.DataFrame, output_dir: Path, logger: logging.Logger) -> None:
    """Plot EF1% vs fraction in top 1% (sanity check)."""
    if df_results.empty or "EF1" not in df_results.columns:
        return
    
    valid = df_results.dropna(subset=["EF1", "frac_in_top1pct"])
    if valid.empty:
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x = valid["frac_in_top1pct"] * 100
    y = valid["EF1"]
    
    ax.scatter(x, y, alpha=0.5, s=30, c="steelblue", edgecolors="black", linewidths=0.5)
    
    ax.set_xlabel("% of Actives Found in Top 1%")
    ax.set_ylabel("EF@1%")
    ax.set_title("Sanity Check: EF@1% vs Actual Top 1% Hit Rate")
    
    # Add diagonal reference
    max_val = max(x.max(), 60) if len(x) > 0 else 60
    ax.plot([0, max_val], [0, max_val], "r--", alpha=0.5, label="y = x (theoretical)")
    ax.legend()
    
    plt.tight_layout()
    fig_path = output_dir / "ef1_vs_top1pct.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {fig_path}")


def plot_by_dr_method(df_results: pd.DataFrame, output_dir: Path, logger: logging.Logger) -> None:
    """Plot distribution by DR method."""
    if df_results.empty or "dr_method" not in df_results.columns:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for ax, col, title in [
        (axes[0], "frac_in_top1pct", "Fraction of Actives in Top 1%"),
        (axes[1], "frac_in_bottom50pct", "Fraction of Actives in Bottom 50%"),
    ]:
        for dr in df_results["dr_method"].dropna().unique():
            subset = df_results[df_results["dr_method"] == dr][col].dropna() * 100
            if len(subset) > 0:
                ax.hist(subset, bins=20, alpha=0.5, label=f"{dr} (n={len(subset)})", edgecolor="black")
        ax.set_xlabel(title)
        ax.set_ylabel("Count (runs)")
        ax.legend()
    
    plt.suptitle("Performance by DR Method", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig_path = output_dir / "distribution_by_dr_method.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {fig_path}")


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze active compound rank distribution to diagnose EF vs BEDROC discrepancy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--workspace_dir", type=str, required=True,
                        help="Workspace directory (e.g., experiment_workspace_v4)")
    parser.add_argument("--phase", type=str, default="phase1",
                        help="Phase subdirectory (default: phase1)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results")
    
    args = parser.parse_args()
    
    workspace = Path(args.workspace_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger(output_dir)
    
    logger.info("=" * 60)
    logger.info("Active Rank Distribution Analysis")
    logger.info("=" * 60)
    logger.info(f"Workspace: {workspace}")
    logger.info(f"Phase: {args.phase}")
    logger.info(f"Output Dir: {output_dir}")
    
    # Find runs
    logger.info("")
    logger.info("Step 1: Finding runs")
    runs = find_runs(workspace, args.phase)
    logger.info(f"Found {len(runs)} runs with ranked_scores.csv")
    
    if len(runs) == 0:
        logger.error("No runs found. Exiting.")
        sys.exit(1)
    
    # Analyze runs
    logger.info("")
    logger.info("Step 2: Analyzing rank distributions")
    
    results = []
    all_percentiles: List[float] = []
    
    for i, run_dir in enumerate(runs):
        if (i + 1) % 50 == 0 or i == 0:
            logger.info(f"Processing run {i+1}/{len(runs)}: {run_dir.name}")
        
        ranked_df = load_ranked_scores(run_dir)
        if ranked_df is None:
            continue
        
        stats = compute_rank_statistics(ranked_df)
        config = extract_run_config(run_dir)
        
        # Build result row
        row = {
            "run_name": run_dir.name,
            **config,
            "N_total": stats.get("N_total"),
            "N_actives": stats.get("N_actives"),
            "rank_mean": stats.get("rank_mean"),
            "rank_median": stats.get("rank_median"),
            "rank_std": stats.get("rank_std"),
            "percentile_mean": stats.get("percentile_mean"),
            "percentile_median": stats.get("percentile_median"),
            "percentile_std": stats.get("percentile_std"),
            "frac_in_top1pct": stats.get("frac_in_top1pct"),
            "frac_in_top5pct": stats.get("frac_in_top5pct"),
            "frac_in_top10pct": stats.get("frac_in_top10pct"),
            "frac_in_bottom50pct": stats.get("frac_in_bottom50pct"),
        }
        results.append(row)
        
        # Collect percentiles for aggregate plot
        if "active_percentiles" in stats and stats["N_actives"] > 0:
            all_percentiles.extend(stats["active_percentiles"].tolist())
        
        # Free memory
        del ranked_df, stats
        gc.collect()
    
    # Save summary CSV
    df_results = pd.DataFrame(results)
    csv_path = output_dir / "active_rank_summary.csv"
    df_results.to_csv(csv_path, index=False)
    logger.info(f"Saved summary to: {csv_path} (rows={len(df_results)})")
    
    # Generate plots
    logger.info("")
    logger.info("Step 3: Generating diagnostic plots")
    
    plot_aggregate_distribution(all_percentiles, output_dir, logger)
    plot_bimodality_diagnostic(df_results, output_dir, logger)
    plot_ef1_sanity_check(df_results, output_dir, logger)
    plot_by_dr_method(df_results, output_dir, logger)
    
    # Summary statistics
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY STATISTICS")
    logger.info("=" * 60)
    
    if len(df_results) > 0:
        logger.info(f"Total runs analyzed: {len(df_results)}")
        
        for col in ["frac_in_top1pct", "frac_in_top5pct", "frac_in_bottom50pct", "percentile_mean", "percentile_median"]:
            if col in df_results.columns:
                vals = df_results[col].dropna()
                if col.startswith("frac_"):
                    vals = vals * 100
                    unit = "%"
                else:
                    unit = ""
                logger.info(f"  {col}: mean={vals.mean():.2f}{unit}, median={vals.median():.2f}{unit}, std={vals.std():.2f}{unit}")
        
        # Bimodality check
        bimodal_runs = df_results[
            (df_results["frac_in_top1pct"] > 0.20) & 
            (df_results["frac_in_bottom50pct"] > 0.25)
        ]
        logger.info(f"Bimodal runs (>20% in top1% AND >25% in bottom50%): {len(bimodal_runs)} / {len(df_results)}")
        
        if len(bimodal_runs) > 0:
            logger.info("")
            logger.info("INTERPRETATION: Bimodal distribution detected.")
            logger.info("The model finds SOME actives very well, but completely misses others.")
            logger.info("Likely cause: MF cloud covers one binding mode/scaffold well, but misses others.")
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("Analysis Complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
