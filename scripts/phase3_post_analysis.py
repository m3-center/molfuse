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
- Potency-stratified degradation curves (High/Medium/Weak tiers)

Potency Tiers (nM):
- High:   0.1 ≤ affinity ≤ 100
- Medium: 100 < affinity ≤ 1,000  
- Weak:   1,000 < affinity ≤ 100,000

Usage:
    python scripts/phase3_post_analysis.py \
        --workspace_dir experiment_workspace_v4 \
        --phase3_run_name mf_ablation \
        --output_dir reporting/phase3_post_analysis \
        --stratify  # Enable potency tier stratification
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# Add project root to path to import molfuse
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Import BEDROC and IEF metrics
from molfuse.metrics.metrics import bedroc, ief

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

def compute_bedroc_ief_from_artifacts_dir(artifacts_dir: Path) -> Dict[str, float]:
    """
    Compute BEDROC and IEF metrics from ranked_scores.csv retrospectively.
    
    Returns dict with keys: bedroc_20, bedroc_160, ief_20, ief_160
    """
    ranked_path = artifacts_dir / "ranked_scores.csv"
    if not ranked_path.exists():
        return {"bedroc_20": np.nan, "bedroc_160": np.nan, "ief_20": np.nan, "ief_160": np.nan}
    
    try:
        df = pd.read_csv(ranked_path)
        if "score" not in df.columns or "label" not in df.columns:
            return {"bedroc_20": np.nan, "bedroc_160": np.nan, "ief_20": np.nan, "ief_160": np.nan}
        
        labels = np.asarray(df["label"].values, dtype=int)
        scores = np.asarray(df["score"].values, dtype=float)
        
        bedroc_20 = bedroc(labels, scores, alpha=20.0)
        bedroc_160 = bedroc(labels, scores, alpha=160.9)
        ief_20 = ief(labels, scores, alpha=20.0)
        ief_160 = ief(labels, scores, alpha=160.9)
        
        return {
            "bedroc_20": bedroc_20,
            "bedroc_160": bedroc_160,
            "ief_20": ief_20,
            "ief_160": ief_160,
        }
    except Exception as e:
        print(f"WARNING: Error computing BEDROC/IEF from {ranked_path}: {e}")
        return {"bedroc_20": np.nan, "bedroc_160": np.nan, "ief_20": np.nan, "ief_160": np.nan}


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
            
            # Compute BEDROC and IEF retrospectively from ranked_scores.csv
            # summary_path is: workspace/phase3/mf_ablation/run_name/logs/phase3_summary.json
            # artifacts_dir is: workspace/phase3/mf_ablation/run_name/artifacts/
            run_dir = summary_path.parent.parent  # Go up from logs/ to run_name/
            artifacts_dir = run_dir / "artifacts"
            if artifacts_dir.exists():
                bedroc_ief_metrics = compute_bedroc_ief_from_artifacts_dir(artifacts_dir)
                record.update(bedroc_ief_metrics)
            else:
                record.update({"bedroc_20": np.nan, "bedroc_160": np.nan, "ief_20": np.nan, "ief_160": np.nan})
            
            records.append(record)
        except Exception as e:
            logger.warning(f"Failed to parse {summary_path}: {e}")
            continue
    
    df = pd.DataFrame(records)
    logger.info(f"Loaded {len(df)} Phase 3 results")
    logger.info(f"  Methods: {df['method'].unique().tolist()}")
    logger.info(f"  Representations: {df['representation'].unique().tolist()}")
    logger.info(f"  MF sizes: {sorted(df['mf_size_actual'].unique().tolist())}")
    
    # Add mf_size_target_numeric for consistent grouping
    df["mf_size_target_numeric"] = df["mf_size_target"].apply(
        lambda x: 999999 if str(x).lower() == "full" else int(x)
    )
    
    # DEBUGGING: Show detailed replicate counts per condition
    logger.info("\n" + "="*80)
    logger.info("DEBUGGING: Replicate counts per condition")
    logger.info("="*80)
    
    condition_counts = df.groupby(['method', 'representation', 'mf_size_target']).size().reset_index(name='n_replicates')
    condition_counts = condition_counts.sort_values(['method', 'representation', 'mf_size_target'])
    
    logger.info("\nExpected: 5 replicates per condition (4 methods × 6 MF sizes = 24 conditions)")
    logger.info(f"Actual: {len(condition_counts)} conditions found\n")
    
    for _, row in condition_counts.iterrows():
        status = "✓ OK" if row['n_replicates'] == 5 else "✗ IRREGULAR"
        logger.info(f"  {status:12s} | {row['method']:4s} / {row['representation']:12s} | MF={str(row['mf_size_target']):>6s} | n={row['n_replicates']}")
    
    # Summary of irregular conditions
    irregular = condition_counts[condition_counts['n_replicates'] != 5]
    if len(irregular) > 0:
        logger.warning(f"\n⚠️  Found {len(irregular)} conditions with irregular replicate counts:")
        for _, row in irregular.iterrows():
            logger.warning(f"    {row['method']}/{row['representation']}/MF={row['mf_size_target']}: {row['n_replicates']} replicates (expected 5)")
    else:
        logger.info("\n✓ All conditions have exactly 5 replicates")
    
    # Count per method-representation combo
    logger.info("\n" + "-"*80)
    logger.info("Runs per method-representation combo:")
    logger.info("-"*80)
    method_rep_counts = df.groupby(['method', 'representation']).size().reset_index(name='n_runs')
    for _, row in method_rep_counts.iterrows():
        expected = 30  # 6 MF sizes × 5 replicates
        status = "✓" if row['n_runs'] == expected else "✗"
        logger.info(f"  {status} {row['method']:4s} / {row['representation']:12s}: {row['n_runs']:3d} runs (expected {expected})")
    
    logger.info("="*80 + "\n")
    
    return df


def aggregate_by_condition(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Aggregate results by (method, representation, mf_size_target).
    
    Computes mean, SEM, std for all metrics across replicates.
    
    CRITICAL: Groups by mf_size_target (not mf_size_actual) because random subsampling
    causes slight variations in actual MF size across replicates (e.g., target=1000 gives
    actual sizes 993, 995, 996, 996, 999 due to affinity filtering + deduplication).
    """
    logger.info("Aggregating results by condition...")
    
    # Add mf_size_target_numeric for proper grouping
    df["mf_size_target_numeric"] = df["mf_size_target"].apply(
        lambda x: 999999 if str(x).lower() == "full" else int(x)
    )
    
    group_keys = ["method", "representation", "dim", "mf_size_target", "mf_size_target_numeric", "affinity_cutoff_nM"]
    
    agg_dict = {
        "ef_1%": ["mean", "sem", "std", "count"],
        "ef_5%": ["mean", "sem", "std"],
        "ef_10%": ["mean", "sem", "std"],
        "bedroc_20": ["mean", "sem", "std"],
        "bedroc_160": ["mean", "sem", "std"],
        "ief_20": ["mean", "sem", "std"],
        "ief_160": ["mean", "sem", "std"],
        "roc_auc": ["mean", "sem", "std"],
        "pr_auc": ["mean", "sem", "std"],
        "n_actives": "first",
        "n_zinc": "first",
        "mf_size_actual": "mean",  # Average actual size for reporting
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
# Potency Tier Stratification
# ============================================================================

def assign_potency_tier(affinity_nM: float) -> str:
    """
    Assign potency tier based on affinity (nM).
    
    Tiers:
    - High:   0.1 ≤ affinity ≤ 100
    - Medium: 100 < affinity ≤ 1,000  
    - Weak:   1,000 < affinity ≤ 100,000
    """
    if pd.isna(affinity_nM):
        return "unknown"
    elif 0.1 <= affinity_nM <= 100:
        return "high"
    elif 100 < affinity_nM <= 1000:
        return "medium"
    elif 1000 < affinity_nM <= 100000:
        return "weak"
    else:
        return "unknown"


def compute_tier_ef1(
    ranked_df: pd.DataFrame,
    tier: str,
    logger: logging.Logger
) -> Optional[float]:
    """
    Compute EF@1% for a specific potency tier.
    
    Args:
        ranked_df: DataFrame with columns ['score', 'label', 'tier']
        tier: Potency tier ('high', 'medium', 'weak')
        logger: Logger instance
    
    Returns:
        EF@1% for the tier, or None if tier has no actives
    """
    # Get tier-specific actives
    tier_actives = ranked_df[(ranked_df["label"] == 1) & (ranked_df["tier"] == tier)]
    n_tier_actives = len(tier_actives)
    
    if n_tier_actives == 0:
        return None
    
    # Total evaluation set size (actives + ZINC)
    n_total = len(ranked_df)
    
    # Top 1% cutoff
    cutoff_idx = int(np.ceil(n_total * 0.01))
    
    # Count tier actives in top 1%
    top_1pct = ranked_df.head(cutoff_idx)
    n_tier_found = len(top_1pct[(top_1pct["label"] == 1) & (top_1pct["tier"] == tier)])
    
    # EF@1% = (found / total_in_tier) / 0.01
    ef1 = (n_tier_found / n_tier_actives) / 0.01
    
    return ef1


def load_actives_with_tiers(
    workspace_dir: Path,
    run_name: str,
    logger: logging.Logger
) -> Optional[pd.DataFrame]:
    """
    Load actives embeddings with affinity values and assign potency tiers.
    
    Reads: workspace/phase3/mf_ablation/{run_name}/artifacts/embedding_actives.csv
    
    Returns:
        DataFrame with columns: Compound ChEMBL ID, Standard Value (nM), tier
    """
    artifacts_dir = workspace_dir / "phase3" / "mf_ablation" / run_name / "artifacts"
    actives_path = artifacts_dir / "embedding_actives.csv"
    
    if not actives_path.exists():
        logger.warning(f"Actives embedding not found: {actives_path}")
        return None
    
    try:
        # Load only ID and affinity columns
        df = pd.read_csv(actives_path, usecols=lambda c: c in ["Compound ChEMBL ID", "Standard Value (nM)"])
        
        # Assign tiers
        df["tier"] = df["Standard Value (nM)"].apply(assign_potency_tier)
        
        return df
    except Exception as e:
        logger.warning(f"Failed to load actives with tiers from {actives_path}: {e}")
        return None


def compute_stratified_metrics_for_run(
    workspace_dir: Path,
    run_name: str,
    logger: logging.Logger
) -> Optional[Dict[str, float]]:
    """
    Compute tier-specific EF@1% for a single Phase 3 run.
    
    Returns:
        Dict with keys: ef_1%_overall, ef_1%_high, ef_1%_medium, ef_1%_weak
    """
    artifacts_dir = workspace_dir / "phase3" / "mf_ablation" / run_name / "artifacts"
    
    # Load ranked scores
    ranked_path = artifacts_dir / "ranked_scores.csv"
    if not ranked_path.exists():
        logger.warning(f"Ranked scores not found: {ranked_path}")
        return None
    
    try:
        ranked_df = pd.read_csv(ranked_path)
    except Exception as e:
        logger.warning(f"Failed to load ranked scores from {ranked_path}: {e}")
        return None
    
    # Load actives with tiers
    actives_tiers = load_actives_with_tiers(workspace_dir, run_name, logger)
    if actives_tiers is None:
        return None
    
    # Merge tier info into ranked_df
    # Match on Compound ChEMBL ID (actives only, ZINC will have NaN tiers)
    ranked_df = ranked_df.merge(
        actives_tiers[["Compound ChEMBL ID", "tier"]],
        on="Compound ChEMBL ID",
        how="left"
    )
    
    # Fill NaN tiers (ZINC compounds) with "zinc"
    ranked_df["tier"] = ranked_df["tier"].fillna("zinc")
    
    # Compute overall EF@1%
    n_actives = (ranked_df["label"] == 1).sum()
    n_total = len(ranked_df)
    cutoff_idx = int(np.ceil(n_total * 0.01))
    n_found = (ranked_df.head(cutoff_idx)["label"] == 1).sum()
    ef1_overall = (n_found / n_actives) / 0.01 if n_actives > 0 else None
    
    # Compute tier-specific EF@1%
    ef1_high = compute_tier_ef1(ranked_df, "high", logger)
    ef1_medium = compute_tier_ef1(ranked_df, "medium", logger)
    ef1_weak = compute_tier_ef1(ranked_df, "weak", logger)
    
    return {
        "ef_1%_overall": ef1_overall,
        "ef_1%_high": ef1_high,
        "ef_1%_medium": ef1_medium,
        "ef_1%_weak": ef1_weak,
    }


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
        subset = subset.sort_values("mf_size_target_numeric")
        
        method = subset["method"].iloc[0]
        representation = subset["representation"].iloc[0]
        
        # Plot mean ± SEM
        x = subset["mf_size_target_numeric"].values
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
        subset = subset.sort_values("mf_size_target_numeric")
        
        x = subset["mf_size_target_numeric"].values
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


def plot_bedroc_ief_comparison_overlay(
    df_agg: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger
) -> None:
    """
    Plot BEDROC and IEF comparison overlays (similar to mf_ablation_comparison).
    Creates 4 separate plots: BEDROC(α=20), BEDROC(α=160), IEF(α=20), IEF(α=160).
    """
    logger.info("Generating BEDROC/IEF comparison overlay plots...")
    
    df_agg["model_key"] = df_agg.apply(get_model_key, axis=1)
    model_keys = sorted(df_agg["model_key"].unique())
    
    colors = {
        "pca_features": "#1f77b4",
        "pca_fingerprints": "#aec7e8",
        "umap_features": "#ff7f0e",
        "umap_fingerprints": "#ffbb78",
    }
    
    # Group metrics with same y-axis range
    metric_groups = [
        ([("bedroc_20_mean", "bedroc_20_sem", "BEDROC (α=20)"),
          ("bedroc_160_mean", "bedroc_160_sem", "BEDROC (α=160)")], "BEDROC"),
        ([("ief_20_mean", "ief_20_sem", "IEF (α=20)"),
          ("ief_160_mean", "ief_160_sem", "IEF (α=160)")], "IEF"),
    ]
    
    for metric_pairs, group_name in metric_groups:
        # Compute global y-axis range for this group
        all_values = []
        for mean_col, sem_col, _ in metric_pairs:
            if mean_col in df_agg.columns and sem_col in df_agg.columns:
                y_mean = df_agg[mean_col].dropna().values
                y_sem = df_agg[sem_col].dropna().values
                if len(y_mean) > 0 and len(y_sem) > 0:
                    all_values.extend(y_mean + y_sem)
                    all_values.extend(y_mean - y_sem)
        
        if len(all_values) == 0:
            logger.warning(f"No valid {group_name} values. Skipping {group_name} comparison plots.")
            continue
        
        y_min = max(0, np.min(all_values) * 0.95)
        y_max = np.max(all_values) * 1.05
        
        # Create plot for each metric in the group
        for mean_col, sem_col, title in metric_pairs:
            if mean_col not in df_agg.columns:
                logger.warning(f"{mean_col} not in dataframe. Skipping {title} plot.")
                continue
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            for model_key in model_keys:
                subset = df_agg[df_agg["model_key"] == model_key].copy()
                subset = subset.sort_values("mf_size_target_numeric")
                
                x = subset["mf_size_target_numeric"].values
                y_mean = subset[mean_col].values
                y_sem = subset[sem_col].values
                
                ax.errorbar(
                    x, y_mean, yerr=y_sem,
                    marker="o", markersize=6, linewidth=2,
                    label=model_key.replace("_", "/"),
                    color=colors.get(model_key, "#333333"),
                    capsize=3, capthick=1.2
                )
            
            ax.set_xscale("log")
            ax.set_ylim(y_min, y_max)  # Apply consistent y-axis range
            ax.set_xlabel("MF Cloud Size (compounds)", fontweight="bold", fontsize=12)
            ax.set_ylabel(f"{title} (Mean ± SEM)", fontweight="bold", fontsize=12)
            ax.set_title(f"MF Cloud Ablation: {title}", fontweight="bold", fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=10)
            
            plt.tight_layout()
            
            metric_name = mean_col.replace("_mean", "")
            output_path_png = output_dir / f"mf_ablation_{metric_name}_comparison.png"
            output_path_pdf = output_dir / f"mf_ablation_{metric_name}_comparison.pdf"
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
               ("bedroc_20_mean", "bedroc_20_sem", "BEDROC (α=20)"),
               ("bedroc_160_mean", "bedroc_160_sem", "BEDROC (α=160)"),
               ("ief_20_mean", "ief_20_sem", "IEF (α=20)"),
               ("ief_160_mean", "ief_160_sem", "IEF (α=160)"),
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
    
    # First pass: compute global y-axis ranges for each metric
    y_limits = {}
    for col_idx, (mean_col, sem_col, title) in enumerate(metrics):
        all_values = []
        for model_key in model_keys:
            subset = df_agg[df_agg["model_key"] == model_key].copy()
            y_mean = subset[mean_col].values
            y_sem = subset[sem_col].values
            valid_mask = ~np.isnan(y_mean) & ~np.isnan(y_sem)
            if valid_mask.any():
                all_values.extend(y_mean[valid_mask] + y_sem[valid_mask])
                all_values.extend(y_mean[valid_mask] - y_sem[valid_mask])
        
        if all_values:
            y_min = max(0, np.min(all_values) * 0.95)
            y_max = np.max(all_values) * 1.05
            y_limits[col_idx] = (y_min, y_max)
        else:
            y_limits[col_idx] = (0, 1)
    
    for row_idx, model_key in enumerate(model_keys):
        subset = df_agg[df_agg["model_key"] == model_key].copy()
        subset = subset.sort_values("mf_size_target_numeric")
        
        x = subset["mf_size_target_numeric"].values
        
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
            ax.set_ylim(y_limits[col_idx])
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
        subset = subset.sort_values("mf_size_target_numeric").reset_index(drop=True)
        
        for i in range(len(subset) - 1):
            mf_curr = subset.loc[i, "mf_size_target_numeric"]
            mf_next = subset.loc[i + 1, "mf_size_target_numeric"]
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
        mf_sizes = sorted(subset["mf_size_target_numeric"].unique())
        data_to_plot = []
        labels = []
        
        for mf_size in mf_sizes:
            ef_values = subset[subset["mf_size_target_numeric"] == mf_size]["ef_1%"].values
            data_to_plot.append(ef_values)
            if mf_size == 999999:
                labels.append("full")
            else:
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
        
        x = subset["mf_size_target_numeric"].values
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


def plot_stratified_degradation_curves(
    df: pd.DataFrame,
    df_stratified: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger
) -> None:
    """
    Plot potency-stratified degradation curves: 4 lines per method (Overall + High + Medium + Weak).
    
    Shows how MF cloud size affects enrichment of different potency tiers.
    """
    logger.info("Generating potency-stratified degradation curves...")
    
    # Aggregate stratified data by (method, representation, mf_size_target)
    df_stratified["mf_size_target_numeric"] = df_stratified["mf_size_target"].apply(
        lambda x: 999999 if str(x).lower() == "full" else int(x)
    )
    
    group_keys = ["method", "representation", "mf_size_target_numeric"]
    
    agg_dict = {
        "ef_1%_overall": ["mean", "sem"],
        "ef_1%_high": ["mean", "sem"],
        "ef_1%_medium": ["mean", "sem"],
        "ef_1%_weak": ["mean", "sem"],
    }
    
    df_strat_agg = df_stratified.groupby(group_keys, dropna=False).agg(agg_dict).reset_index()
    
    # Flatten column names
    df_strat_agg.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in df_strat_agg.columns
    ]
    
    # Create model_key
    df_strat_agg["model_key"] = df_strat_agg.apply(get_model_key, axis=1)
    model_keys = sorted(df_strat_agg["model_key"].unique())
    
    n_models = len(model_keys)
    if n_models == 0:
        logger.warning("No models found, skipping stratified curves")
        return
    
    # Layout: 2×2 for 4 models
    nrows = 2 if n_models > 2 else 1
    ncols = 2 if n_models > 1 else 1
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 12), sharex=True, sharey=False)
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    tier_colors = {
        "overall": "#333333",
        "high": "#1976D2",     # Blue (high potency)
        "medium": "#F57C00",   # Orange (medium)
        "weak": "#C62828",     # Red (weak)
    }
    
    tier_labels = {
        "overall": "Overall",
        "high": "High (≤100 nM)",
        "medium": "Medium (100-1K nM)",
        "weak": "Weak (1K-100K nM)",
    }
    
    # First pass: collect all y values to determine global y-axis range
    all_y_values = []
    for model_key in model_keys:
        subset = df_strat_agg[df_strat_agg["model_key"] == model_key].copy()
        for tier in ["overall", "high", "medium", "weak"]:
            mean_col = f"ef_1%_{tier}_mean"
            sem_col = f"ef_1%_{tier}_sem"
            if mean_col in subset.columns and sem_col in subset.columns:
                y_mean = subset[mean_col].values
                y_sem = subset[sem_col].values
                valid_mask = ~np.isnan(y_mean) & ~np.isnan(y_sem)
                if valid_mask.any():
                    all_y_values.extend(y_mean[valid_mask] + y_sem[valid_mask])
                    all_y_values.extend(y_mean[valid_mask] - y_sem[valid_mask])
    
    # Determine global y-axis limits with 10% padding
    if all_y_values:
        y_min = max(0, np.min(all_y_values) * 0.9)
        y_max = np.max(all_y_values) * 1.1
    else:
        y_min, y_max = 0, 10
    
    for idx, model_key in enumerate(model_keys):
        ax = axes[idx]
        subset = df_strat_agg[df_strat_agg["model_key"] == model_key].copy()
        subset = subset.sort_values("mf_size_target_numeric")
        
        method = subset["method"].iloc[0]
        representation = subset["representation"].iloc[0]
        
        x = subset["mf_size_target_numeric"].values
        
        # Plot each tier
        for tier in ["overall", "high", "medium", "weak"]:
            mean_col = f"ef_1%_{tier}_mean"
            sem_col = f"ef_1%_{tier}_sem"
            
            # Check if columns exist and have data
            if mean_col not in subset.columns or sem_col not in subset.columns:
                continue
            
            y_mean = subset[mean_col].values
            y_sem = subset[sem_col].values
            
            # Skip if all NaN
            if np.all(np.isnan(y_mean)):
                continue
            
            ax.errorbar(
                x, y_mean, yerr=y_sem,
                marker="o", markersize=5, linewidth=2,
                label=tier_labels[tier],
                color=tier_colors[tier],
                capsize=3, capthick=1.2,
                alpha=0.9 if tier == "overall" else 0.7
            )
        
        ax.set_xscale("log")
        ax.set_xlabel("MF Cloud Size (compounds)", fontweight="bold")
        ax.set_ylabel("EF@1% (Mean ± SEM)", fontweight="bold")
        ax.set_title(f"{method.upper()} / {representation}", fontweight="bold")
        ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    
    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis("off")
    
    plt.tight_layout()
    
    output_path_png = output_dir / "mf_ablation_stratified_curves.png"
    output_path_pdf = output_dir / "mf_ablation_stratified_curves.pdf"
    fig.savefig(output_path_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_path_pdf, bbox_inches="tight")
    plt.close(fig)
    
    logger.info(f"Saved: {output_path_png.name}")


def plot_bedroc_ief_stratified_curves(
    df: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger
) -> None:
    """
    Plot BEDROC and IEF stratified degradation curves (overall actives only).
    Creates 4 plots: BEDROC(α=20), BEDROC(α=160), IEF(α=20), IEF(α=160).
    
    Note: Shows overall metrics only (not tier-specific) since BEDROC/IEF 
    per tier would require filtering which changes denominator.
    
    Args:
        df: Main Phase 3 results dataframe (with BEDROC/IEF columns)
        output_dir: Output directory for plots
        logger: Logger instance
    """
    logger.info("Generating BEDROC/IEF stratified degradation curves...")
    
    # Check if BEDROC/IEF columns exist
    required_cols = ["bedroc_20", "bedroc_160", "ief_20", "ief_160"]
    if not all(col in df.columns for col in required_cols):
        logger.warning(f"Missing BEDROC/IEF columns. Skipping BEDROC/IEF stratified curves.")
        return
    
    # Aggregate data by (method, representation, mf_size_target)
    df["mf_size_target_numeric"] = df["mf_size_target"].apply(
        lambda x: 999999 if str(x).lower() == "full" else int(x)
    )
    
    group_keys = ["method", "representation", "mf_size_target_numeric"]
    
    agg_dict = {
        "bedroc_20": ["mean", "sem"],
        "bedroc_160": ["mean", "sem"],
        "ief_20": ["mean", "sem"],
        "ief_160": ["mean", "sem"],
    }
    
    df_agg = df.groupby(group_keys, dropna=False).agg(agg_dict).reset_index()
    
    # Flatten column names
    df_agg.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in df_agg.columns
    ]
    
    # Create model_key
    df_agg["model_key"] = df_agg.apply(get_model_key, axis=1)
    model_keys = sorted(df_agg["model_key"].unique())
    
    n_models = len(model_keys)
    if n_models == 0:
        logger.warning("No models found, skipping BEDROC/IEF stratified curves")
        return
    
    # Layout: 2×2 for 4 models
    nrows = 2 if n_models > 2 else 1
    ncols = 2 if n_models > 1 else 1
    
    # Group metrics with same y-axis range
    metric_groups = [
        ([("bedroc_20_mean", "bedroc_20_sem", "BEDROC (α=20)"),
          ("bedroc_160_mean", "bedroc_160_sem", "BEDROC (α=160)")], "bedroc", "BEDROC"),
        ([("ief_20_mean", "ief_20_sem", "IEF (α=20)"),
          ("ief_160_mean", "ief_160_sem", "IEF (α=160)")], "ief", "IEF"),
    ]
    
    for metric_pairs, file_prefix, group_name in metric_groups:
        # Compute global y-axis range for this group
        all_y_values = []
        for mean_col, sem_col, _ in metric_pairs:
            if mean_col in df_agg.columns and sem_col in df_agg.columns:
                for model_key in model_keys:
                    subset = df_agg[df_agg["model_key"] == model_key].copy()
                    y_mean = subset[mean_col].values
                    y_sem = subset[sem_col].values
                    valid_mask = ~np.isnan(y_mean) & ~np.isnan(y_sem)
                    if valid_mask.any():
                        all_y_values.extend(y_mean[valid_mask] + y_sem[valid_mask])
                        all_y_values.extend(y_mean[valid_mask] - y_sem[valid_mask])
        
        if len(all_y_values) == 0:
            logger.warning(f"No valid {group_name} values. Skipping {group_name} stratified curves.")
            continue
        
        # Determine global y-axis limits with padding
        y_min = max(0, np.min(all_y_values) * 0.9)
        y_max = np.max(all_y_values) * 1.1
        
        # Create separate plot for each metric in the group
        for mean_col, sem_col, title in metric_pairs:
            if mean_col not in df_agg.columns:
                logger.warning(f"{mean_col} not in dataframe. Skipping {title} plot.")
                continue
            
            fig, axes = plt.subplots(nrows, ncols, figsize=(14, 12), sharex=True, sharey=True)
            if n_models == 1:
                axes = np.array([axes])
            axes = axes.flatten()
            
            for idx, model_key in enumerate(model_keys):
                ax = axes[idx]
                subset = df_agg[df_agg["model_key"] == model_key].copy()
                subset = subset.sort_values("mf_size_target_numeric")
                
                method = subset["method"].iloc[0]
                representation = subset["representation"].iloc[0]
                
                x = subset["mf_size_target_numeric"].values
                y_mean = subset[mean_col].values
                y_sem = subset[sem_col].values
                
                mask = ~np.isnan(y_mean) & ~np.isnan(y_sem)
                if mask.any():
                    ax.errorbar(
                        x[mask], y_mean[mask], yerr=y_sem[mask],
                        marker="o", markersize=8, linewidth=2.5,
                        color="#1976D2", alpha=0.9,
                        capsize=4, capthick=1.5
                    )
                
                ax.set_xscale("log")
                ax.set_xlabel("MF Cloud Size (compounds)", fontweight="bold")
                ax.set_ylabel(f"{title} (Mean ± SEM)", fontweight="bold")
                ax.set_title(f"{method.upper()} / {representation}", fontweight="bold")
                ax.set_ylim(y_min, y_max)  # Apply consistent y-axis range
                ax.grid(True, alpha=0.3)
            
            # Hide unused subplots
            for idx in range(n_models, len(axes)):
                axes[idx].axis("off")
            
            plt.tight_layout()
            
            metric_name = mean_col.replace("_mean", "")
            output_path_png = output_dir / f"mf_ablation_{metric_name}_stratified_curves.png"
            output_path_pdf = output_dir / f"mf_ablation_{metric_name}_stratified_curves.pdf"
            fig.savefig(output_path_png, dpi=300, bbox_inches="tight")
            fig.savefig(output_path_pdf, bbox_inches="tight")
            plt.close(fig)
            
            logger.info(f"Saved: {output_path_png.name}")


def plot_tier_enrichment_ratio(
    df_stratified: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger
) -> None:
    """
    Plot enrichment ratio: EF@1%(tier) / EF@1%(overall) across MF sizes.
    
    Shows which tiers benefit most/least from large MF clouds.
    Ratio > 1 means tier performs better than overall average.
    """
    logger.info("Generating tier enrichment ratio plot...")
    
    # Aggregate
    df_stratified["mf_size_target_numeric"] = df_stratified["mf_size_target"].apply(
        lambda x: 999999 if str(x).lower() == "full" else int(x)
    )
    
    group_keys = ["method", "representation", "mf_size_target_numeric"]
    
    agg_dict = {
        "ef_1%_overall": "mean",
        "ef_1%_high": "mean",
        "ef_1%_medium": "mean",
        "ef_1%_weak": "mean",
    }
    
    df_agg = df_stratified.groupby(group_keys, dropna=False).agg(agg_dict).reset_index()
    
    # Compute ratios
    for tier in ["high", "medium", "weak"]:
        df_agg[f"ratio_{tier}"] = df_agg[f"ef_1%_{tier}"] / df_agg["ef_1%_overall"]
    
    # Create model_key
    df_agg["model_key"] = df_agg.apply(get_model_key, axis=1)
    model_keys = sorted(df_agg["model_key"].unique())
    
    n_models = len(model_keys)
    if n_models == 0:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    
    tier_colors = {
        "high": "#1976D2",     # Blue (high potency)
        "medium": "#F57C00",   # Orange (medium)
        "weak": "#C62828",     # Red (weak)
    }
    
    # First pass: collect all ratio values to determine global y-axis range
    all_ratio_values = []
    for model_key in model_keys:
        subset = df_agg[df_agg["model_key"] == model_key].copy()
        for tier in ["high", "medium", "weak"]:
            y = subset[f"ratio_{tier}"].values
            valid_mask = ~np.isnan(y)
            if valid_mask.any():
                all_ratio_values.extend(y[valid_mask])
    
    # Determine global y-axis limits with padding
    if all_ratio_values:
        y_min = min(0.5, np.min(all_ratio_values) * 0.9)
        y_max = max(1.5, np.max(all_ratio_values) * 1.1)
    else:
        y_min, y_max = 0.5, 1.5
    
    for idx, model_key in enumerate(model_keys):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        subset = df_agg[df_agg["model_key"] == model_key].copy()
        subset = subset.sort_values("mf_size_target_numeric")
        
        method = subset["method"].iloc[0]
        representation = subset["representation"].iloc[0]
        
        x = subset["mf_size_target_numeric"].values
        
        for tier in ["high", "medium", "weak"]:
            y = subset[f"ratio_{tier}"].values
            
            if not np.all(np.isnan(y)):
                ax.plot(
                    x, y,
                    marker="o", markersize=5, linewidth=2,
                    label=tier.capitalize(),
                    color=tier_colors[tier]
                )
        
        # Reference line at ratio=1 (tier = overall)
        ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="Overall")
        
        ax.set_xscale("log")
        ax.set_xlabel("MF Cloud Size", fontweight="bold")
        ax.set_ylabel("EF@1%(tier) / EF@1%(overall)", fontweight="bold")
        ax.set_title(f"{method.upper()} / {representation}", fontweight="bold")
        ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    
    # Hide unused
    for idx in range(n_models, len(axes)):
        axes[idx].axis("off")
    
    plt.tight_layout()
    
    output_path_png = output_dir / "mf_ablation_tier_enrichment_ratio.png"
    output_path_pdf = output_dir / "mf_ablation_tier_enrichment_ratio.pdf"
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
        subset = subset.sort_values("mf_size_target_numeric")
        
        # Get full-MF performance (largest mf_size_target_numeric)
        full_mf = subset[subset["mf_size_target_numeric"] == subset["mf_size_target_numeric"].max()]
        if len(full_mf) == 0:
            logger.warning(f"{model_key}: No full-MF data found")
            continue
        
        full_ef1 = full_mf["ef_1%_mean"].iloc[0]
        target_ef1 = full_ef1 * (threshold_pct / 100.0)
        
        # Find first MF size achieving target
        viable = subset[subset["ef_1%_mean"] >= target_ef1]
        
        if len(viable) > 0:
            min_size_numeric = int(viable["mf_size_target_numeric"].min())
            min_size_display = "full" if min_size_numeric == 999999 else min_size_numeric
            min_viable[model_key] = min_size_display
            logger.info(f"  {model_key}: {min_size_display} compounds (EF@1% = {viable[viable['mf_size_target_numeric'] == min_size_numeric]['ef_1%_mean'].iloc[0]:.2f}, target = {target_ef1:.2f})")
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
        subset = df_agg[df_agg["model_key"] == model_key].sort_values("mf_size_target_numeric")
        for _, row in subset.iterrows():
            mf_display = str(row['mf_size_target']) if row['mf_size_target'] != 999999 else "full"
            report_lines.append(
                f"| {model_key} | {mf_display} | "
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
    parser.add_argument("--stratify", action="store_true",
                       help="Enable potency tier stratification (High/Medium/Weak)")
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
    logger.info(f"Stratify by potency: {args.stratify}")
    logger.info("="*80)
    
    try:
        # 1. Collect results
        df = collect_phase3_results(workspace_dir, args.phase3_run_name, logger)
        
        # 2. Aggregate by condition
        df_agg = aggregate_by_condition(df, logger)
        
        # 3. Generate plots
        plot_degradation_curves(df, df_agg, output_dir, logger)
        plot_comparison_overlay(df_agg, output_dir, logger)
        plot_bedroc_ief_comparison_overlay(df_agg, output_dir, logger)
        plot_metrics_grid(df_agg, output_dir, logger)
        
        # 4. Phase transition analysis
        df_slopes = compute_degradation_slopes(df_agg, logger)
        plot_phase_transition_heatmap(df_slopes, output_dir, logger)
        
        # 5. Replicate variance
        plot_replicate_variance(df, output_dir, logger)
        
        # 6. Variance correlation
        analyze_variance_correlation(df_agg, output_dir, logger)
        
        # 7. Potency-stratified analysis (if enabled)
        if args.stratify:
            logger.info("\n" + "="*80)
            logger.info("POTENCY TIER STRATIFICATION")
            logger.info("="*80)
            
            # Compute tier-specific metrics for all runs
            logger.info("Computing tier-specific EF@1% for all runs...")
            stratified_records = []
            
            for _, row in df.iterrows():
                run_name = row["run_name"]
                tier_metrics = compute_stratified_metrics_for_run(workspace_dir, run_name, logger)
                
                if tier_metrics:
                    record = {
                        "run_name": run_name,
                        "method": row["method"],
                        "representation": row["representation"],
                        "mf_size_target": row["mf_size_target"],
                        "replicate": row["replicate"],
                        **tier_metrics
                    }
                    stratified_records.append(record)
            
            if stratified_records:
                df_stratified = pd.DataFrame(stratified_records)
                logger.info(f"Computed stratified metrics for {len(df_stratified)} runs")
                
                # Save stratified CSV
                strat_csv = output_dir / "phase3_summary_stratified.csv"
                df_stratified.to_csv(strat_csv, index=False)
                logger.info(f"Saved: {strat_csv.name}")
                
                # Generate stratified plots
                plot_stratified_degradation_curves(df, df_stratified, output_dir, logger)
                plot_bedroc_ief_stratified_curves(df, output_dir, logger)
                plot_tier_enrichment_ratio(df_stratified, output_dir, logger)
            else:
                logger.warning("No stratified metrics computed (missing artifacts?)")
        
        # 8. Summary outputs
        save_aggregated_summary(df_agg, output_dir, logger)
        min_viable = identify_minimum_viable_sizes(df_agg, args.threshold_pct, output_dir, logger)
        
        # 9. Generate report
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
