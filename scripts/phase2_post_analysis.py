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


# ============================================================================
# POTENCY TIER DEFINITIONS
# ============================================================================

POTENCY_TIERS = {
    "High": (0.1, 100.0),       # 0.1-100 nM (drug-like)
    "Medium": (100.0, 1000.0),  # 100-1000 nM (moderate)
    "Weak": (1000.0, 100000.0), # 1K-100K nM (marginal)
}


def assign_potency_tier(affinity_nM: float) -> Optional[str]:
    """
    Assign potency tier based on affinity value (nM).
    
    Returns:
        "High" if 0.1 ≤ affinity ≤ 100
        "Medium" if 100 < affinity ≤ 1000
        "Weak" if 1000 < affinity ≤ 100000
        None otherwise
    """
    try:
        val = float(affinity_nM)
    except (ValueError, TypeError):
        return None
    
    if val < 0.1 or val > 100000:
        return None
    
    for tier_name, (min_val, max_val) in POTENCY_TIERS.items():
        if min_val <= val <= max_val:
            return tier_name
    
    return None


def compute_ef_at_percent(
    ranked_df: pd.DataFrame,
    tier: Optional[str],
    top_pct: float
) -> Tuple[Optional[float], int]:
    """
    Compute Enrichment Factor at top X% for a specific tier.
    
    Args:
        ranked_df: Full ranked list (actives + ZINC), pre-sorted by score
        tier: "High", "Medium", "Weak", or None (all actives)
        top_pct: Fraction (e.g., 0.01 for 1%)
    
    Returns:
        Tuple of (EF value or None, N_actives_in_tier)
    """
    if ranked_df.empty:
        return None, 0
    
    N_total = len(ranked_df)
    k = max(1, int(np.ceil(top_pct * N_total)))
    top_k = ranked_df.head(k)
    
    if tier is None:
        # All actives
        mask_all = ranked_df["label"] == 1
        mask_top = top_k["label"] == 1
        N_actives = mask_all.sum()
        hits = mask_top.sum()
    else:
        # Specific tier
        mask_all = (ranked_df["label"] == 1) & (ranked_df["potency_tier"] == tier)
        mask_top = (top_k["label"] == 1) & (top_k["potency_tier"] == tier)
        N_actives = int(mask_all.sum())
        hits = mask_top.sum()
    
    if N_actives == 0:
        return None, 0
    
    # EF = (hits / N_actives) / (k / N_total)
    ef = (hits / N_actives) / (k / N_total)
    return float(ef), N_actives


def load_actives_with_affinity(phase1_workspace: Path, phase1_run: str) -> Optional[pd.DataFrame]:
    """
    Load actives with affinity from Phase 1 source data.
    
    Strategy:
    1. Load Phase 1 summary.json to get config
    2. Load actives CSV or MF CSV with affinity data
    
    Returns DataFrame with: Compound ChEMBL ID, SMILES, Standard Value (nM)
    """
    phase1_dir = phase1_workspace / "phase1" / phase1_run
    summary_path = phase1_dir / "logs" / "phase1_summary.json"
    
    if not summary_path.exists():
        return None
    
    try:
        with summary_path.open("r") as f:
            summary = json.load(f)
    except Exception:
        return None
    
    config = summary.get("config", {})
    
    # Try actives CSV first
    actives_csv = config.get("actives_features_csv")
    if actives_csv:
        p = Path(actives_csv)
        if p.exists():
            try:
                cols_needed = ["Compound ChEMBL ID", "SMILES", "Standard Value (nM)"]
                df = pd.read_csv(p, usecols=lambda c: c in cols_needed + ["accession"], low_memory=False)
                return df[cols_needed].copy()
            except Exception:
                pass
    
    # Fallback: load MF CSV and filter by target
    mf_csv = config.get("mf_features_csv")
    if mf_csv:
        p = Path(mf_csv)
        target = config.get("target", "")
        
        # Extract accession from target (e.g., "ABL1_P00519" -> "P00519")
        import re
        m = re.search(r"_([A-Z0-9]{6})$", target)
        if m and p.exists():
            accession = m.group(1)
            try:
                cols_needed = ["Compound ChEMBL ID", "SMILES", "Standard Value (nM)", "accession"]
                df_all = pd.read_csv(p, usecols=lambda c: c in cols_needed, low_memory=False)
                df = df_all[df_all["accession"] == accession].copy()
                return df[["Compound ChEMBL ID", "SMILES", "Standard Value (nM)"]].copy()
            except Exception:
                pass
    
    return None


def add_tier_metrics_to_cutoff(
    cutoff_dir: Path,
    phase1_workspace: Path,
    phase1_run: str
) -> Dict:
    """
    Compute tier-stratified metrics for one Phase 2 cutoff directory.
    
    Returns dict with tier metrics or empty dict if failed.
    """
    # Load ranked scores (now has compound IDs from Phase 2)
    ranked_path = cutoff_dir / "ranked_scores.csv"
    if not ranked_path.exists():
        return {}
    
    try:
        ranked_df = pd.read_csv(ranked_path, low_memory=False)
    except Exception:
        return {}
    
    # Load actives with affinity
    actives_df = load_actives_with_affinity(phase1_workspace, phase1_run)
    if actives_df is None or actives_df.empty:
        return {}
    
    # Join affinity to ranked list
    # Try ChEMBL ID first, fallback to SMILES
    if "Compound ChEMBL ID" in ranked_df.columns and "Compound ChEMBL ID" in actives_df.columns:
        join_key = "Compound ChEMBL ID"
        ranked_df["_join"] = ranked_df[join_key].astype(str).str.upper().str.strip()
        actives_df["_join"] = actives_df[join_key].astype(str).str.upper().str.strip()
    elif "SMILES" in ranked_df.columns and "SMILES" in actives_df.columns:
        join_key = "SMILES"
        ranked_df["_join"] = ranked_df[join_key].astype(str).str.strip()
        actives_df["_join"] = actives_df[join_key].astype(str).str.strip()
    else:
        return {}
    
    # Create affinity lookup
    affinity_lookup = actives_df[["_join", "Standard Value (nM)"]].dropna(subset=["_join"]).drop_duplicates(subset=["_join"])
    
    # Left join (only actives get affinity values)
    ranked_df = ranked_df.merge(affinity_lookup, on="_join", how="left")
    
    # Assign tiers
    ranked_df["potency_tier"] = ranked_df["Standard Value (nM)"].apply(assign_potency_tier)
    
    # Compute tier-specific EF@1%
    ef1_high, n_high = compute_ef_at_percent(ranked_df, "High", 0.01)
    ef1_medium, n_medium = compute_ef_at_percent(ranked_df, "Medium", 0.01)
    ef1_weak, n_weak = compute_ef_at_percent(ranked_df, "Weak", 0.01)
    
    return {
        "ef1_high": ef1_high,
        "ef1_medium": ef1_medium,
        "ef1_weak": ef1_weak,
        "n_actives_high": n_high,
        "n_actives_medium": n_medium,
        "n_actives_weak": n_weak,
    }


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


def aggregate_phase2_results(phase2_dir: Path, phase1_workspace: Path, compute_tiers: bool = True) -> pd.DataFrame:
    """
    Aggregate Phase 2 metrics across models and cutoffs.

    Args:
        phase2_dir: Phase 2 output directory
        phase1_workspace: Phase 1 workspace for loading affinity data
        compute_tiers: If True, compute tier-stratified metrics

    Returns DataFrame with columns: model_key, method, representation, dim, cutoff_nM, ef1, ef5, ef10, roc_auc, pr_auc, n_mf,
                                     ef1_high, ef1_medium, ef1_weak, n_actives_high, n_actives_medium, n_actives_weak
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

            row = {
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
            }
            
            # Compute tier-stratified metrics if requested
            if compute_tiers:
                phase1_run = metrics.get("phase1_run", "")
                if phase1_run:
                    tier_metrics = add_tier_metrics_to_cutoff(cutoff_dir, phase1_workspace, phase1_run)
                    row.update(tier_metrics)
                else:
                    # No Phase 1 run info, skip tier computation
                    row.update({
                        "ef1_high": None,
                        "ef1_medium": None,
                        "ef1_weak": None,
                        "n_actives_high": 0,
                        "n_actives_medium": 0,
                        "n_actives_weak": 0,
                    })
            
            rows.append(row)

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


def get_best_configs_per_method(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to best configuration per method × representation combination.
    
    Strategy:
    1. For each (method, representation) pair, find the model_key with highest mean EF@1% across all cutoffs
    2. This gives 4 best configs: PCA/features, PCA/fingerprints, UMAP/features, UMAP/fingerprints
    
    Args:
        df: Aggregated Phase 2 metrics DataFrame
    
    Returns:
        DataFrame filtered to best configs only (4 model_keys)
    """
    best_models = []
    
    for method in df["method"].unique():
        for rep in df["representation"].unique():
            subset = df[(df["method"] == method) & (df["representation"] == rep)].copy()
            
            if subset.empty:
                continue
            
            # Compute mean EF@1% across all cutoffs for each model_key
            avg_ef1 = subset.groupby("model_key")["ef1"].mean()
            if not avg_ef1.empty:
                best_model_key = avg_ef1.idxmax()
                best_models.append(best_model_key)
    
    filtered = df[df["model_key"].isin(best_models)].copy()
    return filtered


def plot_cutoff_tier_sensitivity(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Plot potency-tier stratified EF@1% vs affinity cutoff for best configs per method.
    
    Shows how different cutoffs affect enrichment of high/medium/weak potency actives.
    Only shows best configuration per method (PCA, UMAP) based on overall EF@1%.
    
    Research Questions:
    1. Strictness Hypothesis: Do stricter cutoffs preferentially enrich high-potency actives?
    2. Quality-Quantity Trade-off: Does reducing MF cloud hurt overall enrichment?
    3. Tier Inversion: Is there a cutoff where high-potency EF exceeds overall EF?
    4. Method Sensitivity: Do PCA and UMAP respond differently to cutoffs?
    
    Args:
        df: Aggregated Phase 2 metrics (must include ef1_high, ef1_medium, ef1_weak)
        output_dir: Output directory for plots
    """
    # Check if stratified metrics are available
    if df["ef1_high"].isna().all():
        print("WARNING: No stratified metrics found. Run phase2_add_stratified_metrics.py first.")
        print("Skipping tier-wise cutoff sensitivity plot.")
        return
    
    # Filter to best configs per method
    df_best = get_best_configs_per_method(df)
    
    if df_best.empty:
        print("WARNING: No best configs identified. Skipping tier-wise plot.")
        return
    
    print(f"\nBest configurations per method × representation (for tier-wise plot):")
    for model_key in sorted(df_best["model_key"].unique()):
        method = df_best[df_best["model_key"] == model_key]["method"].iloc[0]
        rep = df_best[df_best["model_key"] == model_key]["representation"].iloc[0]
        print(f"  {method}/{rep}: {model_key}")
    
    # Sort by cutoff for proper line plotting
    df_best = df_best.sort_values(by=["model_key", "cutoff_nM"]).reset_index(drop=True)
    
    # Create multi-panel plot (2×2 grid for 4 model_keys)
    n_models = df_best["model_key"].nunique()
    
    if n_models <= 2:
        # Horizontal layout for 1-2 models
        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), sharey=True, squeeze=False)
        axes = axes.flatten()
    else:
        # 2×2 grid for 3-4 models
        ncols = 2
        nrows = int(np.ceil(n_models / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 5 * nrows), sharey=True, squeeze=False)
        axes = axes.flatten()
    
    # Color scheme for tiers
    colors = {
        "All": "#2E7D32",      # Green (overall)
        "High": "#1976D2",     # Blue (high potency)
        "Medium": "#F57C00",   # Orange (medium)
        "Weak": "#C62828",     # Red (weak)
    }
    
    for idx, model_key in enumerate(sorted(df_best["model_key"].unique())):
        ax = axes[idx]
        subset = df_best[df_best["model_key"] == model_key].copy()
        
        # Get tier counts (use first row, should be same across cutoffs)
        n_high = subset["n_actives_high"].iloc[0] if "n_actives_high" in subset.columns else 0
        n_medium = subset["n_actives_medium"].iloc[0] if "n_actives_medium" in subset.columns else 0
        n_weak = subset["n_actives_weak"].iloc[0] if "n_actives_weak" in subset.columns else 0
        
        # Plot each tier
        x = subset["cutoff_nM"].values
        
        # All actives (baseline)
        y_all = subset["ef1"].values
        ax.plot(x, y_all, marker="o", label="All", color=colors["All"], 
                linewidth=2.5, markersize=8, alpha=0.9)
        
        # High potency (0.1-100 nM)
        y_high = subset["ef1_high"].values
        mask_high = ~np.isnan(y_high)
        if mask_high.any():
            ax.plot(x[mask_high], y_high[mask_high], marker="s", 
                   label=f"High (n={n_high})", 
                   color=colors["High"], linewidth=2, markersize=7, alpha=0.8)
        
        # Medium potency (100-1000 nM)
        y_medium = subset["ef1_medium"].values
        mask_medium = ~np.isnan(y_medium)
        if mask_medium.any():
            ax.plot(x[mask_medium], y_medium[mask_medium], marker="^", 
                   label=f"Medium (n={n_medium})", 
                   color=colors["Medium"], linewidth=2, markersize=7, alpha=0.8)
        
        # Weak potency (1K-100K nM)
        y_weak = subset["ef1_weak"].values
        mask_weak = ~np.isnan(y_weak)
        if mask_weak.any():
            ax.plot(x[mask_weak], y_weak[mask_weak], marker="D", 
                   label=f"Weak (n={n_weak})", 
                   color=colors["Weak"], linewidth=2, markersize=7, alpha=0.8)
        
        # Formatting
        ax.set_xscale("log")
        ax.set_xlabel("Affinity Cutoff (nM)", fontsize=11, fontweight="bold")
        if idx == 0:
            ax.set_ylabel("EF@1%", fontsize=11, fontweight="bold")
        
        # Extract method and representation for title
        method = subset["method"].iloc[0]
        rep = subset["representation"].iloc[0]
        dim = subset["dim"].iloc[0]
        
        # Add tier counts to title
        title = f"{method.upper()} ({rep}, dim={dim})\n"
        title += f"Tiers: High={n_high}, Medium={n_medium}, Weak={n_weak}"
        ax.set_title(title, fontsize=11, fontweight="bold")
        
        ax.legend(loc="best", fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        # Add vertical line at optimal cutoff (based on overall EF@1%)
        best_idx = subset["ef1"].idxmax()
        best_cutoff = subset.loc[best_idx, "cutoff_nM"]
        ax.axvline(best_cutoff, color="gray", linestyle="--", linewidth=1.5, alpha=0.6, 
                  label=f"Optimal: {int(best_cutoff)} nM")
    
    plt.suptitle("Potency-Tier Stratified Cutoff Sensitivity (Best Configs)", 
                fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    save_figure(fig, output_dir, "cutoff_tier_sensitivity_best_configs")
    print(f"Saved cutoff_tier_sensitivity_best_configs.png/pdf")
    
    # Save tier-wise metrics for best configs
    tier_cols = ["model_key", "method", "representation", "dim", "cutoff_nM", 
                 "ef1", "ef1_high", "ef1_medium", "ef1_weak",
                 "n_actives_high", "n_actives_medium", "n_actives_weak"]
    df_best[tier_cols].to_csv(output_dir / "phase2_tier_metrics_best_configs.csv", index=False)
    print(f"Saved tier metrics (best configs): {output_dir / 'phase2_tier_metrics_best_configs.csv'}")


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

    # Aggregate results (with tier-stratified metrics)
    print("Aggregating Phase 2 results and computing tier-stratified metrics...")
    print("(This may take a few minutes...)")
    df = aggregate_phase2_results(phase2_dir, workspace_dir, compute_tiers=True)

    if df.empty:
        print("ERROR: No Phase 2 results found. Check workspace directory.")
        return

    print(f"Found {len(df)} results across {df['model_key'].nunique()} models and {df['cutoff_nM'].nunique()} cutoffs")
    
    # Report tier counts
    if "n_actives_high" in df.columns and df["n_actives_high"].notna().any():
        # Get unique tier counts (should be same across cutoffs for a given target)
        tier_summary = df[["n_actives_high", "n_actives_medium", "n_actives_weak"]].drop_duplicates()
        if len(tier_summary) == 1:
            n_high = int(tier_summary["n_actives_high"].iloc[0])
            n_medium = int(tier_summary["n_actives_medium"].iloc[0])
            n_weak = int(tier_summary["n_actives_weak"].iloc[0])
            print(f"\nPotency tier counts:")
            print(f"  High (0.1-100 nM): {n_high} actives")
            print(f"  Medium (100-1K nM): {n_medium} actives")
            print(f"  Weak (1K-100K nM): {n_weak} actives")
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

    # Tier-wise cutoff sensitivity (best configs only)
    plot_cutoff_tier_sensitivity(df, output_dir)

    # Identify best cutoffs
    identify_best_cutoffs(df, output_dir)

    print()
    print("="*80)
    print("PHASE 2 POST-ANALYSIS COMPLETED")
    print("="*80)
    print(f"All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
