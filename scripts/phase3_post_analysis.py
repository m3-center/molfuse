#!/usr/bin/env python3
"""
Phase 3 Post-Analysis: MF Cloud Ablation Study

Aggregates Phase 3 results and generates publication-quality plots:
- Degradation curves (EF@1% vs MF size) for each method
- Comparison grid (all methods overlaid)
- Multi-metric dashboard (EF@1/5/10%, ROC, PR)
- Phase transition analysis (degradation slope heatmap)
- Replicate variability analysis
- Correlation between MF size and performance variance

Usage:
    python scripts/phase3_post_analysis.py \
        --workspace_dir experiment_workspace_v4 \
        --phase3_run_name mf_ablation \
        --output_dir reporting/phase3_post_analysis
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

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
    sns.set_palette("colorblind")
except ImportError:
    sns = None
    _HAVE_SNS = False


# ============================================================================
# Data Aggregation
# ============================================================================

def collect_phase3_results(
    workspace_dir: Path,
    phase3_run_name: str,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Scan workspace for Phase 3 summary JSONs and build master DataFrame.
    
    Args:
        workspace_dir: Base workspace directory
        phase3_run_name: Name of Phase 3 run (e.g., "mf_ablation")
        logger: Logger instance
    
    Returns:
        DataFrame with all Phase 3 results
    """
    phase3_dir = workspace_dir / "phase3" / phase3_run_name
    
    if not phase3_dir.exists():
        raise FileNotFoundError(f"Phase 3 directory not found: {phase3_dir}")
    
    logger.info(f"Scanning {phase3_dir} for phase3_summary.json files...")
    
    # Find all summary files
    summary_files = list(phase3_dir.rglob("phase3_summary.json"))
    logger.info(f"Found {len(summary_files)} summary files")
    
    if len(summary_files) == 0:
        raise ValueError("No phase3_summary.json files found!")
    
    # Parse each summary
    records = []
    for summary_path in summary_files:
        try:
            with summary_path.open("r") as f:
                data = json.load(f)
            
            # Extract key fields
            record = {
                "run_name": data.get("run_name"),
                "method": data.get("method"),
                "representation": data.get("representation"),
                "dim": data.get("dim"),
                "mf_size_target": data.get("mf_size_target"),
                "mf_size_actual": data.get("mf_size_actual"),
                "affinity_cutoff_nM": data.get("affinity_cutoff_nM"),
                "replicate": data.get("config", {}).get("replicate", 1),
                "random_seed": data.get("random_seed"),
                "n_actives": data.get("n_actives"),
                "n_zinc": data.get("n_zinc"),
                "ef_1%": data.get("ef_1%"),
                "ef_5%": data.get("ef_5%"),
                "ef_10%": data.get("ef_10%"),
                "roc_auc": data.get("roc_auc"),
                "pr_auc": data.get("pr_auc"),
                "spearman_rho": data.get("spearman_rho"),
                "elapsed_time_s": data.get("elapsed_time_s"),
            }
            records.append(record)
        except Exception as e:
            logger.warning(f"Failed to parse {summary_path}: {e}")
            continue
    
    df = pd.DataFrame(records)
    logger.info(f"Loaded {len(df)} Phase 3 results")
    logger.info(f"  Methods: {df['method'].unique().tolist()}")
    logger.info(f"  Representations: {df['representation'].unique().tolist()}")
    logger.info(f"  MF sizes: {sorted(df['mf_size_actual'].unique().tolist())}")
    logger.info(f"  Replicates per condition: {df.groupby(['method', 'representation', 'mf_size_actual']).size().unique().tolist()}")
    
    return df


def aggregate_by_condition(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Aggregate results by (method, representation, mf_size_actual).
    
    Computes mean, SEM, std for all metrics across replicates.
    """
    logger.info("Aggregating results by condition...")
    
    group_keys = ["method", "representation", "dim", "mf_size_actual", "affinity_cutoff_nM"]
    
    agg_dict = {
        "ef_1%": ["mean", "sem", "std", "count"],
        "ef_5%": ["mean", "sem", "std"],
        "ef_10%": ["mean", "sem", "std"],
        "roc_auc": ["mean", "sem", "std"],
        "pr_auc": ["mean", "sem", "std"],
        "n_actives": "first",
        "n_zinc": "first",
    }
    
    df_agg = df.groupby(group_keys, dropna=False).agg(agg_dict).reset_index()
    
    # Flatten column names
    df_agg.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in df_agg.columns
    ]
    
    logger.info(f"Aggregated to {len(df_agg)} unique conditions")
    
    return df_agg


# ============================================================================
# Model Key Generation
# ============================================================================

def get_model_key(row: pd.Series) -> str:
    """Generate model_key from method and representation."""
    return f"{row['method']}_{row['representation']}"


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_degradation_curves(
    df: pd.DataFrame,
    df_agg: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger
) -> None:
    """
    Plot degradation curves: EF@1% vs MF size (4 panels, one per method×representation).
    
    Shows mean ± SEM across replicates, with log-scale X-axis.
    """
    logger.info("Generating degradation curves (4 panels)...")
    
    methods = sorted(df["method"].unique())
    representations = sorted(df["representation"].unique())
    
    # Create model_key
    df_agg["model_key"] = df_agg.apply(get_model_key, axis=1)
    model_keys = sorted(df_agg["model_key"].unique())
    
    n_models = len(model_keys)
    if n_models == 0:
        logger.warning("No models found, skipping degradation curves")
        return
    
    # Layout: 2×2 for 4 models
    nrows = 2 if n_models > 2 else 1
    ncols = 2 if n_models > 1 else 1
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 10), sharex=True, sharey=True)
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    colors = {"pca": "#1f77b4", "umap": "#ff7f0e"}
    
    for idx, model_key in enumerate(model_keys):
        ax = axes[idx]
        subset = df_agg[df_agg["model_key"] == model_key].copy()
        subset = subset.sort_values("mf_size_actual")
        
        method = subset["method"].iloc[0]
        representation = subset["representation"].iloc[0]
        
        # Plot mean ± SEM
        x = subset["mf_size_actual"].values
        y_mean = subset["ef_1%_mean"].values
        y_sem = subset["ef_1%_sem"].values
        
        ax.errorbar(
            x, y_mean, yerr=y_sem,
            marker="o", markersize=6, linewidth=2,
            label=f"{method.upper()}/{representation}",
            color=colors.get(method, "#333333"),
            capsize=4, capthick=1.5
        )
        
        ax.set_xscale("log")
        ax.set_xlabel("MF Cloud Size (compounds)", fontweight="bold")
        ax.set_ylabel("EF@1% (Mean ± SEM)", fontweight="bold")
        ax.set_title(f"{method.upper()} / {representation}", fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    
    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis("off")
    
    plt.tight_layout()
    
    output_path_png = output_dir / "mf_ablation_degradation_curves.png"
    output_path_pdf = output_dir / "mf_ablation_degradation_curves.pdf"
    fig.savefig(output_path_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_path_pdf, bbox_inches="tight")
    plt.close(fig)
    
    logger.info(f"Saved: {output_path_png.name}")


def plot_comparison_overlay(
    df_agg: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger
) -> None:
    """
    Plot all methods overlaid on single plot for comparison.
    """
    logger.info("Generating comparison overlay plot...")
    
    df_agg["model_key"] = df_agg.apply(get_model_key, axis=1)
    model_keys = sorted(df_agg["model_key"].unique())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {
        "pca_features": "#1f77b4",
        "pca_fingerprints": "#aec7e8",
        "umap_features": "#ff7f0e",
        "umap_fingerprints": "#ffbb78",
    }
    
    for model_key in model_keys:
        subset = df_agg[df_agg["model_key"] == model_key].copy()
        subset = subset.sort_values("mf_size_actual")
        
        x = subset["mf_size_actual"].values
        y_mean = subset["ef_1%_mean"].values
        y_sem = subset["ef_1%_sem"].values
        
        ax.errorbar(
            x, y_mean, yerr=y_sem,
            marker="o", markersize=6, linewidth=2,
            label=model_key.replace("_", "/"),
            color=colors.get(model_key, "#333333"),
            capsize=3, capthick=1.2
        )
    
    ax.set_xscale("log")
    ax.set_xlabel("MF Cloud Size (compounds)", fontweight="bold", fontsize=12)
    ax.set_ylabel("EF@1% (Mean ± SEM)", fontweight="bold", fontsize=12)
    ax.set_title("MF Cloud Ablation: Method Comparison", fontweight="bold", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)
    
    plt.tight_layout()
    
    output_path_png = output_dir / "mf_ablation_comparison.png"
    output_path_pdf = output_dir / "mf_ablation_comparison.pdf"
    fig.savefig(output_path_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_path_pdf, bbox_inches="tight")
    plt.close(fig)
    
    logger.info(f"Saved: {output_path_png.name}")


def plot_metrics_grid(
    df_agg: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger
) -> None:
    """
    Plot multi-metric dashboard: 4 methods × 3 metrics (EF@1%, ROC-AUC, PR-AUC).
    """
    logger.info("Generating multi-metric grid...")
    
    df_agg["model_key"] = df_agg.apply(get_model_key, axis=1)
    model_keys = sorted(df_agg["model_key"].unique())
    
    metrics = [("ef_1%_mean", "ef_1%_sem", "EF@1%"),
               ("roc_auc_mean", "roc_auc_sem", "ROC-AUC"),
               ("pr_auc_mean", "pr_auc_sem", "PR-AUC")]
    
    fig, axes = plt.subplots(len(model_keys), len(metrics), figsize=(15, 12), sharex=True)
    
    if len(model_keys) == 1:
        axes = axes.reshape(1, -1)
    
    colors = {
        "pca_features": "#1f77b4",
        "pca_fingerprints": "#aec7e8",
        "umap_features": "#ff7f0e",
        "umap_fingerprints": "#ffbb78",
    }
    
    for row_idx, model_key in enumerate(model_keys):
        subset = df_agg[df_agg["model_key"] == model_key].copy()
        subset = subset.sort_values("mf_size_actual")
        
        x = subset["mf_size_actual"].values
        
        for col_idx, (mean_col, sem_col, title) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            
            y_mean = subset[mean_col].values
            y_sem = subset[sem_col].values
            
            ax.errorbar(
                x, y_mean, yerr=y_sem,
                marker="o", markersize=5, linewidth=1.8,
                color=colors.get(model_key, "#333333"),
                capsize=3, capthick=1
            )
            
            ax.set_xscale("log")
            ax.grid(True, alpha=0.3)
            
            # Titles
            if row_idx == 0:
                ax.set_title(title, fontweight="bold")
            
            # Y-labels
            if col_idx == 0:
                ax.set_ylabel(model_key.replace("_", "/"), fontweight="bold")
            
            # X-labels
            if row_idx == len(model_keys) - 1:
                ax.set_xlabel("MF Size", fontweight="bold")
    
    plt.tight_layout()
    
    output_path_png = output_dir / "mf_ablation_metrics_grid.png"
    output_path_pdf = output_dir / "mf_ablation_metrics_grid.pdf"
    fig.savefig(output_path_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_path_pdf, bbox_inches="tight")
    plt.close(fig)
    
    logger.info(f"Saved: {output_path_png.name}")


def compute_degradation_slopes(df_agg: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Compute degradation slope between consecutive MF sizes (second derivative).
    
    Slope = (EF@1%[i+1] - EF@1%[i]) / (log10(MF[i+1]) - log10(MF[i]))
    """
    logger.info("Computing degradation slopes...")
    
    df_agg["model_key"] = df_agg.apply(get_model_key, axis=1)
    model_keys = sorted(df_agg["model_key"].unique())
    
    slope_records = []
    
    for model_key in model_keys:
        subset = df_agg[df_agg["model_key"] == model_key].copy()
        subset = subset.sort_values("mf_size_actual").reset_index(drop=True)
        
        for i in range(len(subset) - 1):
            mf_curr = subset.loc[i, "mf_size_actual"]
            mf_next = subset.loc[i + 1, "mf_size_actual"]
            ef_curr = subset.loc[i, "ef_1%_mean"]
            ef_next = subset.loc[i + 1, "ef_1%_mean"]
            
            # Log-scale slope
            if mf_curr > 0 and mf_next > 0:
                slope = (ef_next - ef_curr) / (np.log10(mf_next) - np.log10(mf_curr))
            else:
                slope = 0.0
            
            slope_records.append({
                "model_key": model_key,
                "method": subset.loc[i, "method"],
                "representation": subset.loc[i, "representation"],
                "mf_from": int(mf_curr),
                "mf_to": int(mf_next),
                "slope": slope,
            })
    
    df_slopes = pd.DataFrame(slope_records)
    logger.info(f"Computed {len(df_slopes)} slope values")
    
    return df_slopes


def plot_phase_transition_heatmap(
    df_slopes: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger
) -> None:
    """
    Plot heatmap of degradation slopes (identifies phase transitions).
    
    Steepest negative slope indicates critical MF size threshold.
    """
    logger.info("Generating phase transition heatmap...")
    
    if not _HAVE_SNS:
        logger.warning("Seaborn not available, skipping heatmap")
        return
    
    # Pivot for heatmap
    df_pivot = df_slopes.pivot_table(
        index="model_key",
        columns="mf_from",
        values="slope",
        aggfunc="mean"
    )
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sns.heatmap(
        df_pivot,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        cbar_kws={"label": "Slope (EF@1% / log10(MF size))"},
        ax=ax,
        linewidths=0.5,
        linecolor="gray"
    )
    
    ax.set_title("Phase Transition Analysis: Degradation Slopes", fontweight="bold", fontsize=14)
    ax.set_xlabel("MF Size (starting point)", fontweight="bold")
    ax.set_ylabel("Method / Representation", fontweight="bold")
    
    plt.tight_layout()
    
    output_path_png = output_dir / "mf_ablation_phase_transition.png"
    output_path_pdf = output_dir / "mf_ablation_phase_transition.pdf"
    fig.savefig(output_path_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_path_pdf, bbox_inches="tight")
    plt.close(fig)
    
    logger.info(f"Saved: {output_path_png.name}")


def plot_replicate_variance(
    df: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger
) -> None:
    """
    Plot box plots showing variance across replicates.
    """
    logger.info("Generating replicate variance analysis...")
    
    df["model_key"] = df.apply(get_model_key, axis=1)
    model_keys = sorted(df["model_key"].unique())
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    
    for idx, model_key in enumerate(model_keys):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        subset = df[df["model_key"] == model_key].copy()
        
        # Prepare data for box plot
        mf_sizes = sorted(subset["mf_size_actual"].unique())
        data_to_plot = []
        labels = []
        
        for mf_size in mf_sizes:
            ef_values = subset[subset["mf_size_actual"] == mf_size]["ef_1%"].values
            data_to_plot.append(ef_values)
            labels.append(f"{int(mf_size)}")
        
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
        
        # Color boxes
        for patch in bp['boxes']:
            patch.set_facecolor("#aec7e8")
        
        ax.set_xlabel("MF Cloud Size", fontweight="bold")
        ax.set_ylabel("EF@1% (individual replicates)", fontweight="bold")
        ax.set_title(f"{model_key.replace('_', '/')}", fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        ax.tick_params(axis='x', rotation=45)
    
    # Hide unused subplots
    for idx in range(len(model_keys), len(axes)):
        axes[idx].axis("off")
    
    plt.tight_layout()
    
    output_path_png = output_dir / "mf_ablation_replicate_variance.png"
    output_path_pdf = output_dir / "mf_ablation_replicate_variance.pdf"
    fig.savefig(output_path_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_path_pdf, bbox_inches="tight")
    plt.close(fig)
    
    logger.info(f"Saved: {output_path_png.name}")


def analyze_variance_correlation(
    df_agg: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger
) -> None:
    """
    Analyze correlation between MF size and performance variance.
    
    Question: Does smaller MF cloud lead to more unstable performance?
    """
    logger.info("Analyzing MF size vs variance correlation...")
    
    df_agg["model_key"] = df_agg.apply(get_model_key, axis=1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {
        "pca_features": "#1f77b4",
        "pca_fingerprints": "#aec7e8",
        "umap_features": "#ff7f0e",
        "umap_fingerprints": "#ffbb78",
    }
    
    for model_key in sorted(df_agg["model_key"].unique()):
        subset = df_agg[df_agg["model_key"] == model_key].copy()
        
        x = subset["mf_size_actual"].values
        y = subset["ef_1%_std"].values
        
        ax.scatter(
            x, y,
            label=model_key.replace("_", "/"),
            color=colors.get(model_key, "#333333"),
            s=80, alpha=0.7
        )
        
        # Fit trend line
        if len(x) > 2:
            z = np.polyfit(np.log10(x), y, 1)
            p = np.poly1d(z)
            x_fit = np.logspace(np.log10(x.min()), np.log10(x.max()), 100)
            ax.plot(x_fit, p(np.log10(x_fit)), "--", color=colors.get(model_key, "#333333"), alpha=0.5, linewidth=1.5)
    
    ax.set_xscale("log")
    ax.set_xlabel("MF Cloud Size (compounds)", fontweight="bold", fontsize=12)
    ax.set_ylabel("EF@1% Std Dev (across replicates)", fontweight="bold", fontsize=12)
    ax.set_title("MF Size vs Performance Variability", fontweight="bold", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)
    
    plt.tight_layout()
    
    output_path_png = output_dir / "mf_ablation_variance_correlation.png"
    output_path_pdf = output_dir / "mf_ablation_variance_correlation.pdf"
    fig.savefig(output_path_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_path_pdf, bbox_inches="tight")
    plt.close(fig)
    
    logger.info(f"Saved: {output_path_png.name}")


# ============================================================================
# Summary Statistics and Reports
# ============================================================================

def save_aggregated_summary(df_agg: pd.DataFrame, output_dir: Path, logger: logging.Logger) -> None:
    """Save aggregated summary CSV."""
    logger.info("Saving aggregated summary CSV...")
    
    output_path = output_dir / "phase3_summary_aggregated.csv"
    df_agg.to_csv(output_path, index=False)
    logger.info(f"Saved: {output_path.name} ({len(df_agg)} rows)")


def identify_minimum_viable_sizes(
    df_agg: pd.DataFrame,
    threshold_pct: float,
    output_dir: Path,
    logger: logging.Logger
) -> Dict[str, int]:
    """
    Identify minimum MF size achieving ≥threshold% of full-MF EF@1%.
    
    Args:
        df_agg: Aggregated results
        threshold_pct: Percentage threshold (e.g., 80 for 80%)
        output_dir: Output directory
        logger: Logger
    
    Returns:
        Dict mapping model_key → minimum viable MF size
    """
    logger.info(f"Identifying minimum viable MF sizes (≥{threshold_pct}% of full-MF)...")
    
    df_agg["model_key"] = df_agg.apply(get_model_key, axis=1)
    
    min_viable = {}
    
    for model_key in sorted(df_agg["model_key"].unique()):
        subset = df_agg[df_agg["model_key"] == model_key].copy()
        subset = subset.sort_values("mf_size_actual")
        
        # Get full-MF performance
        full_mf = subset[subset["mf_size_actual"] == subset["mf_size_actual"].max()]
        if len(full_mf) == 0:
            logger.warning(f"{model_key}: No full-MF data found")
            continue
        
        full_ef1 = full_mf["ef_1%_mean"].iloc[0]
        target_ef1 = full_ef1 * (threshold_pct / 100.0)
        
        # Find first MF size achieving target
        viable = subset[subset["ef_1%_mean"] >= target_ef1]
        
        if len(viable) > 0:
            min_size = int(viable["mf_size_actual"].min())
            min_viable[model_key] = min_size
            logger.info(f"  {model_key}: {min_size:,} compounds (EF@1% = {viable[viable['mf_size_actual'] == min_size]['ef_1%_mean'].iloc[0]:.2f}, target = {target_ef1:.2f})")
        else:
            min_viable[model_key] = None
            logger.warning(f"  {model_key}: No size achieves {threshold_pct}% threshold")
    
    # Save to JSON
    output_path = output_dir / "phase3_minimum_viable_sizes.json"
    with output_path.open("w") as f:
        json.dump(min_viable, f, indent=2)
    
    logger.info(f"Saved: {output_path.name}")
    
    return min_viable


def generate_analysis_report(
    df: pd.DataFrame,
    df_agg: pd.DataFrame,
    df_slopes: pd.DataFrame,
    min_viable: Dict[str, int],
    output_dir: Path,
    logger: logging.Logger
) -> None:
    """Generate markdown analysis report."""
    logger.info("Generating analysis report...")
    
    report_lines = [
        "# Phase 3 MF Cloud Ablation Study - Analysis Report",
        "",
        f"**Total Runs**: {len(df)}",
        f"**Unique Conditions**: {len(df_agg)}",
        f"**Methods**: {', '.join(sorted(df['method'].unique()))}",
        f"**Representations**: {', '.join(sorted(df['representation'].unique()))}",
        "",
        "## Summary Statistics",
        "",
        "### Performance by MF Size (EF@1%)",
        "",
    ]
    
    # Summary table
    df_agg["model_key"] = df_agg.apply(get_model_key, axis=1)
    
    report_lines.append("| Model | MF Size | EF@1% (Mean ± SEM) | n_replicates |")
    report_lines.append("|---|---|---|---|")
    
    for model_key in sorted(df_agg["model_key"].unique()):
        subset = df_agg[df_agg["model_key"] == model_key].sort_values("mf_size_actual")
        for _, row in subset.iterrows():
            report_lines.append(
                f"| {model_key} | {int(row['mf_size_actual']):,} | "
                f"{row['ef_1%_mean']:.2f} ± {row['ef_1%_sem']:.2f} | "
                f"{int(row['ef_1%_count'])} |"
            )
    
    report_lines.extend([
        "",
        "## Phase Transition Analysis",
        "",
        "Degradation slopes (EF@1% change per log10(MF size)):",
        "",
        "| Model | MF Transition | Slope |",
        "|---|---|---|",
    ])
    
    for _, row in df_slopes.iterrows():
        report_lines.append(
            f"| {row['model_key']} | {int(row['mf_from']):,} → {int(row['mf_to']):,} | "
            f"{row['slope']:.3f} |"
        )
    
    report_lines.extend([
        "",
        "## Minimum Viable MF Sizes (≥80% of Full-MF EF@1%)",
        "",
        "| Model | Min MF Size |",
        "|---|---|",
    ])
    
    for model_key, size in sorted(min_viable.items()):
        if size is not None:
            report_lines.append(f"| {model_key} | {size:,} |")
        else:
            report_lines.append(f"| {model_key} | N/A (threshold not met) |")
    
    report_lines.extend([
        "",
        "## Key Findings",
        "",
        "1. **Degradation Patterns**: [Describe how EF@1% changes with MF size]",
        "2. **Method Comparison**: [Compare PCA vs UMAP sensitivity to MF size]",
        "3. **Critical Thresholds**: [Identify phase transition points]",
        "4. **Recommendations**: [Practical guidance for MF cloud size selection]",
        "",
    ])
    
    # Write report
    output_path = output_dir / "phase3_analysis_report.md"
    with output_path.open("w") as f:
        f.write("\n".join(report_lines))
    
    logger.info(f"Saved: {output_path.name}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 3 Post-Analysis: MF Cloud Ablation")
    parser.add_argument("--workspace_dir", type=str, default="experiment_workspace_v4",
                       help="Workspace directory")
    parser.add_argument("--phase3_run_name", type=str, default="mf_ablation",
                       help="Phase 3 run name")
    parser.add_argument("--output_dir", type=str, default="reporting/phase3_post_analysis",
                       help="Output directory for analysis")
    parser.add_argument("--threshold_pct", type=float, default=80.0,
                       help="Threshold percentage for minimum viable MF size (default: 80)")
    args = parser.parse_args()
    
    workspace_dir = Path(args.workspace_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "analysis.log", mode="w"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info("="*80)
    logger.info("PHASE 3 POST-ANALYSIS: MF CLOUD ABLATION STUDY")
    logger.info("="*80)
    logger.info(f"Workspace: {workspace_dir}")
    logger.info(f"Phase 3 run: {args.phase3_run_name}")
    logger.info(f"Output: {output_dir}")
    logger.info("="*80)
    
    try:
        # 1. Collect results
        df = collect_phase3_results(workspace_dir, args.phase3_run_name, logger)
        
        # 2. Aggregate by condition
        df_agg = aggregate_by_condition(df, logger)
        
        # 3. Generate plots
        plot_degradation_curves(df, df_agg, output_dir, logger)
        plot_comparison_overlay(df_agg, output_dir, logger)
        plot_metrics_grid(df_agg, output_dir, logger)
        
        # 4. Phase transition analysis
        df_slopes = compute_degradation_slopes(df_agg, logger)
        plot_phase_transition_heatmap(df_slopes, output_dir, logger)
        
        # 5. Replicate variance
        plot_replicate_variance(df, output_dir, logger)
        
        # 6. Variance correlation
        analyze_variance_correlation(df_agg, output_dir, logger)
        
        # 7. Summary outputs
        save_aggregated_summary(df_agg, output_dir, logger)
        min_viable = identify_minimum_viable_sizes(df_agg, args.threshold_pct, output_dir, logger)
        
        # 8. Generate report
        generate_analysis_report(df, df_agg, df_slopes, min_viable, output_dir, logger)
        
        logger.info("="*80)
        logger.info("PHASE 3 POST-ANALYSIS COMPLETED")
        logger.info(f"Results saved to: {output_dir}")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
