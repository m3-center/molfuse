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
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def load_actives_with_affinity(phase1_workspace: Path, phase1_run: str, cache: Dict = None) -> Optional[pd.DataFrame]:
    """
    Load actives with affinity from Phase 1 source data (with caching).
    
    Strategy:
    1. Load Phase 1 summary.json to get config
    2. Load actives CSV or MF CSV with affinity data
    3. Cache results to avoid re-loading same file
    
    Args:
        phase1_workspace: Base workspace directory
        phase1_run: Phase 1 run name
        cache: Optional dict to cache loaded data (keyed by phase1_run)
    
    Returns DataFrame with: Compound ChEMBL ID, SMILES, Standard Value (nM)
    """
    # Check cache first
    if cache is not None and phase1_run in cache:
        return cache[phase1_run]
    
    phase1_dir = phase1_workspace / "phase1" / phase1_run
    summary_path = phase1_dir / "logs" / "phase1_summary.json"
    
    if not summary_path.exists():
        result = None
    else:
        try:
            with summary_path.open("r") as f:
                summary = json.load(f)
        except Exception:
            result = None
        else:
            config = summary.get("config", {})
            result = None
            
            # Try actives CSV first
            actives_csv = config.get("actives_features_csv")
            if actives_csv:
                p = Path(actives_csv)
                if p.exists():
                    try:
                        cols_needed = ["Compound ChEMBL ID", "SMILES", "Standard Value (nM)"]
                        df = pd.read_csv(p, usecols=lambda c: c in cols_needed + ["accession"], low_memory=False)
                        result = df[cols_needed].copy()
                    except Exception:
                        pass
            
            # Fallback: load MF CSV and filter by target
            if result is None:
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
                            result = df[["Compound ChEMBL ID", "SMILES", "Standard Value (nM)"]].copy()
                        except Exception:
                            pass
    
    # Cache the result
    if cache is not None:
        cache[phase1_run] = result
    
    return result


def compute_all_metrics_for_cutoff(
    cutoff_dir: Path,
    phase1_workspace: Path,
    phase1_run: str,
    actives_cache: Dict,
    affinity_lookup_cache: Dict
) -> Dict:
    """
    Compute ALL metrics for one cutoff directory efficiently (single CSV load).
    
    Computes:
    - BEDROC and IEF (overall)
    - Tier-stratified EF@1%
    - Tier-stratified BEDROC and IEF
    
    Args:
        cutoff_dir: Path to cutoff directory
        phase1_workspace: Base workspace for Phase 1
        phase1_run: Phase 1 run name
        actives_cache: Cache for actives DataFrames (shared across calls)
        affinity_lookup_cache: Cache for affinity lookup dicts (shared across calls)
    
    Returns:
        Dict with all metrics, or empty dict if failed
    """
    ranked_path = cutoff_dir / "ranked_scores.csv"
    if not ranked_path.exists():
        return {}
    
    try:
        # Load ranked scores ONCE
        ranked_df = pd.read_csv(ranked_path, low_memory=False)
        
        if "score" not in ranked_df.columns or "label" not in ranked_df.columns:
            return {}
        
        # Extract arrays for BEDROC/IEF computation
        labels = np.asarray(ranked_df["label"].values, dtype=int)
        scores = np.asarray(ranked_df["score"].values, dtype=float)
        
        # Compute overall BEDROC and IEF
        bedroc_20 = bedroc(labels, scores, alpha=20.0)
        bedroc_160 = bedroc(labels, scores, alpha=160.9)
        ief_20 = ief(labels, scores, alpha=20.0)
        ief_160 = ief(labels, scores, alpha=160.9)
        
        metrics = {
            "bedroc_20": bedroc_20,
            "bedroc_160": bedroc_160,
            "ief_20": ief_20,
            "ief_160": ief_160,
        }
        
    except Exception as e:
        print(f"Error loading ranked scores for {cutoff_dir.name}: {e}")
        return {}
    
    # Compute tier-stratified metrics
    if phase1_run and phase1_run != "unknown":
        # Load actives with affinity (cached)
        actives_df = load_actives_with_affinity(phase1_workspace, phase1_run, cache=actives_cache)
        
        if actives_df is None or actives_df.empty:
            # No tier data available
            metrics.update({
                "ef1_high": None,
                "ef1_medium": None,
                "ef1_weak": None,
                "n_actives_high": 0,
                "n_actives_medium": 0,
                "n_actives_weak": 0,
                "bedroc_20_high": np.nan,
                "bedroc_20_medium": np.nan,
                "bedroc_20_weak": np.nan,
                "bedroc_160_high": np.nan,
                "bedroc_160_medium": np.nan,
                "bedroc_160_weak": np.nan,
                "ief_20_high": np.nan,
                "ief_20_medium": np.nan,
                "ief_20_weak": np.nan,
                "ief_160_high": np.nan,
                "ief_160_medium": np.nan,
                "ief_160_weak": np.nan,
            })
            return metrics
        
        # Create or retrieve affinity lookup (cached)
        lookup_key = phase1_run
        if lookup_key in affinity_lookup_cache:
            affinity_lookup = affinity_lookup_cache[lookup_key]
            join_key = affinity_lookup_cache[f"{lookup_key}_join_key"]
        else:
            # Determine join key and create lookup
            if "Compound ChEMBL ID" in ranked_df.columns and "Compound ChEMBL ID" in actives_df.columns:
                join_key = "Compound ChEMBL ID"
                # Normalize join keys
                actives_df["_join"] = actives_df[join_key].astype(str).str.upper().str.strip()
            elif "SMILES" in ranked_df.columns and "SMILES" in actives_df.columns:
                join_key = "SMILES"
                actives_df["_join"] = actives_df[join_key].astype(str).str.strip()
            else:
                # No valid join key
                metrics.update({
                    "ef1_high": None,
                    "ef1_medium": None,
                    "ef1_weak": None,
                    "n_actives_high": 0,
                    "n_actives_medium": 0,
                    "n_actives_weak": 0,
                    "bedroc_20_high": np.nan,
                    "bedroc_20_medium": np.nan,
                    "bedroc_20_weak": np.nan,
                    "bedroc_160_high": np.nan,
                    "bedroc_160_medium": np.nan,
                    "bedroc_160_weak": np.nan,
                    "ief_20_high": np.nan,
                    "ief_20_medium": np.nan,
                    "ief_20_weak": np.nan,
                    "ief_160_high": np.nan,
                    "ief_160_medium": np.nan,
                    "ief_160_weak": np.nan,
                })
                return metrics
            
            # Create affinity lookup dict
            affinity_lookup = actives_df[["_join", "Standard Value (nM)"]].dropna(subset=["_join"]).drop_duplicates(subset=["_join"])
            affinity_lookup_cache[lookup_key] = affinity_lookup
            affinity_lookup_cache[f"{lookup_key}_join_key"] = join_key
        
        # Normalize ranked_df join key (only once)
        if join_key == "Compound ChEMBL ID":
            ranked_df["_join"] = ranked_df[join_key].astype(str).str.upper().str.strip()
        else:  # SMILES
            ranked_df["_join"] = ranked_df[join_key].astype(str).str.strip()
        
        # Left join to add affinity
        ranked_df = ranked_df.merge(affinity_lookup, on="_join", how="left")
        
        # Assign potency tiers
        ranked_df["potency_tier"] = ranked_df["Standard Value (nM)"].apply(assign_potency_tier)
        
        # Compute tier-specific EF@1%
        ef1_high, n_high = compute_ef_at_percent(ranked_df, "High", 0.01)
        ef1_medium, n_medium = compute_ef_at_percent(ranked_df, "Medium", 0.01)
        ef1_weak, n_weak = compute_ef_at_percent(ranked_df, "Weak", 0.01)
        
        # Compute tier-specific BEDROC and IEF
        tier_bedroc_ief = {}
        
        for tier_name, tier_label in [("High", "high"), ("Medium", "medium"), ("Weak", "weak")]:
            # Create binary label: 1 if active AND in this tier, 0 otherwise
            tier_mask = ranked_df["potency_tier"] == tier_name
            tier_labels = tier_mask.astype(int).values
            
            # Only compute if we have positives in this tier
            if tier_labels.sum() > 0:
                tier_scores = ranked_df["score"].values
                
                tier_bedroc_ief[f"bedroc_20_{tier_label}"] = bedroc(tier_labels, tier_scores, alpha=20.0)
                tier_bedroc_ief[f"bedroc_160_{tier_label}"] = bedroc(tier_labels, tier_scores, alpha=160.9)
                tier_bedroc_ief[f"ief_20_{tier_label}"] = ief(tier_labels, tier_scores, alpha=20.0)
                tier_bedroc_ief[f"ief_160_{tier_label}"] = ief(tier_labels, tier_scores, alpha=160.9)
            else:
                tier_bedroc_ief[f"bedroc_20_{tier_label}"] = np.nan
                tier_bedroc_ief[f"bedroc_160_{tier_label}"] = np.nan
                tier_bedroc_ief[f"ief_20_{tier_label}"] = np.nan
                tier_bedroc_ief[f"ief_160_{tier_label}"] = np.nan
        
        # Add tier metrics to result
        metrics.update({
            "ef1_high": ef1_high,
            "ef1_medium": ef1_medium,
            "ef1_weak": ef1_weak,
            "n_actives_high": n_high,
            "n_actives_medium": n_medium,
            "n_actives_weak": n_weak,
            **tier_bedroc_ief,
        })
    else:
        # No phase1_run info, skip tier computation
        metrics.update({
            "ef1_high": None,
            "ef1_medium": None,
            "ef1_weak": None,
            "n_actives_high": 0,
            "n_actives_medium": 0,
            "n_actives_weak": 0,
            "bedroc_20_high": np.nan,
            "bedroc_20_medium": np.nan,
            "bedroc_20_weak": np.nan,
            "bedroc_160_high": np.nan,
            "bedroc_160_medium": np.nan,
            "bedroc_160_weak": np.nan,
            "ief_20_high": np.nan,
            "ief_20_medium": np.nan,
            "ief_20_weak": np.nan,
            "ief_160_high": np.nan,
            "ief_160_medium": np.nan,
            "ief_160_weak": np.nan,
        })
    
    return metrics


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


def compute_bedroc_ief_from_ranked_scores(cutoff_dir: Path) -> Dict[str, float]:
    """
    DEPRECATED: Use compute_all_metrics_for_cutoff() instead.
    
    This function is kept for backwards compatibility but should not be used.
    compute_all_metrics_for_cutoff() is much more efficient as it loads
    the CSV only once and computes all metrics together.
    """
    raise NotImplementedError("Use compute_all_metrics_for_cutoff() instead")


def aggregate_phase2_results(phase2_dir: Path, phase1_workspace: Path, compute_tiers: bool = True) -> pd.DataFrame:
    """
    Aggregate Phase 2 metrics across models, cutoffs, and replicates.

    Args:
        phase2_dir: Phase 2 output directory
        phase1_workspace: Phase 1 workspace for loading affinity data
        compute_tiers: If True, compute tier-stratified metrics

    Returns DataFrame with columns: model_key, method, representation, dim, cutoff_nM, replicate,
                                     ef1, ef5, ef10, roc_auc, pr_auc, n_mf,
                                     ef1_high, ef1_medium, ef1_weak, n_actives_high, n_actives_medium, n_actives_weak
                                     
    Note: Returns one row per (model_key, cutoff, replicate) combination.
          For aggregated statistics (mean ± SEM), call aggregate_replicates() on this output.
    """
    rows: List[Dict] = []
    
    # Create caches for expensive operations
    actives_cache: Dict = {}  # Cache for actives DataFrames (keyed by phase1_run)
    affinity_lookup_cache: Dict = {}  # Cache for affinity lookup dicts

    # Count total directories for progress tracking
    total_dirs = sum(1 for model_dir in phase2_dir.iterdir() 
                     if model_dir.is_dir() and model_dir.name not in ("logs", "artifacts", "metrics", "selected_models.json")
                     for cutoff_dir in model_dir.iterdir()
                     if cutoff_dir.is_dir() and cutoff_dir.name.startswith("cutoff_"))
    
    print(f"Processing {total_dirs} cutoff directories...")
    processed = 0

    # Scan for model subdirectories (now these are replicate-specific: e.g., ABL1_PCA_features_20d_rep1)
    for model_dir in phase2_dir.iterdir():
        if not model_dir.is_dir() or model_dir.name in ("logs", "artifacts", "metrics", "selected_models.json"):
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
            
            # Extract replicate number from phase1_run name (e.g., "ABL1_PCA_features_20d_rep4" -> 4)
            phase1_run = metrics.get("phase1_run", "unknown")
            import re
            rep_match = re.search(r'_rep(\d+)$', phase1_run)
            replicate = int(rep_match.group(1)) if rep_match else None

            row = {
                "model_key": f"{metrics.get('method', 'unknown')}_{metrics.get('representation', 'unknown')}",
                "method": metrics.get("method", "unknown"),
                "representation": metrics.get("representation", "unknown"),
                "dim": metrics.get("dim", 0),
                "cutoff_nM": metrics.get("affinity_cutoff_nM", 0),
                "replicate": replicate,
                "ef1": metrics.get("ef_1%", 0.0),
                "ef5": metrics.get("ef_5%", 0.0),
                "ef10": metrics.get("ef_10%", 0.0),
                "roc_auc": metrics.get("roc_auc", 0.0),
                "pr_auc": metrics.get("pr_auc", 0.0),
                "n_mf": metrics.get("n_mf_for_scoring", 0),
                "phase1_run": phase1_run,
            }
            
            # Compute ALL metrics efficiently (BEDROC/IEF + tiers) in single pass
            if compute_tiers and phase1_run and phase1_run != "unknown":
                all_metrics = compute_all_metrics_for_cutoff(
                    cutoff_dir, 
                    phase1_workspace, 
                    phase1_run,
                    actives_cache,
                    affinity_lookup_cache
                )
                row.update(all_metrics)
            else:
                # Just compute BEDROC/IEF without tiers
                all_metrics = compute_all_metrics_for_cutoff(
                    cutoff_dir, 
                    phase1_workspace, 
                    phase1_run,
                    actives_cache,
                    affinity_lookup_cache
                )
                row.update(all_metrics)
            
            rows.append(row)
            
            # Progress indicator
            processed += 1
            if processed % 10 == 0:
                print(f"  Processed {processed}/{total_dirs} cutoff directories...")

    print(f"  Completed: {processed}/{total_dirs} directories")
    df = pd.DataFrame(rows)
    return df


def aggregate_replicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate metrics across replicates, computing mean ± SEM.
    
    Groups by (model_key, method, representation, dim, cutoff_nM) and computes:
    - Mean, SEM, std, count for all numeric metrics
    - First value for categorical/ID columns
    
    Args:
        df: DataFrame with one row per (model_key, cutoff, replicate)
    
    Returns:
        DataFrame with one row per (model_key, cutoff), containing mean/SEM/std/count columns
    """
    # Identify metric columns to aggregate
    metric_cols = [
        "ef1", "ef5", "ef10", "roc_auc", "pr_auc", "n_mf",
        "bedroc_20", "bedroc_160", "ief_20", "ief_160",
        "ef1_high", "ef1_medium", "ef1_weak",
        "bedroc_20_high", "bedroc_20_medium", "bedroc_20_weak",
        "bedroc_160_high", "bedroc_160_medium", "bedroc_160_weak",
        "ief_20_high", "ief_20_medium", "ief_20_weak",
        "ief_160_high", "ief_160_medium", "ief_160_weak",
    ]
    
    # Filter to metrics that actually exist in df
    metric_cols = [c for c in metric_cols if c in df.columns]
    
    # Group by configuration
    group_keys = ["model_key", "method", "representation", "dim", "cutoff_nM"]
    
    # Build aggregation dictionary
    agg_dict = {}
    for col in metric_cols:
        agg_dict[col] = ["mean", "sem", "std", "count"]
    
    # Add categorical columns (take first value)
    for col in ["n_actives_high", "n_actives_medium", "n_actives_weak"]:
        if col in df.columns:
            agg_dict[col] = "first"
    
    # Aggregate
    df_agg = df.groupby(group_keys, as_index=False).agg(agg_dict)
    
    # Flatten multi-level column names
    df_agg.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in df_agg.columns.values]
    
    return df_agg


def plot_cutoff_curves(df: pd.DataFrame, metric_col: str, output_dir: Path, basename: str) -> None:
    """
    Plot metric vs cutoff curves for each model (aggregated across replicates).

    Args:
        df: Aggregated metrics DataFrame (with mean/SEM columns)
        metric_col: Metric column to plot (e.g., "ef1", "ef5", "ef10")
        output_dir: Output directory
        basename: Base filename for saved plots
    """
    # Check if we have aggregated data (mean column exists)
    mean_col = f"{metric_col}_mean"
    sem_col = f"{metric_col}_sem"
    
    if mean_col not in df.columns:
        print(f"WARNING: No aggregated data found for {metric_col}. Skipping plot.")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each model as a separate line with error bars
    for model_key in df["model_key"].unique():
        subset = df[df["model_key"] == model_key].sort_values("cutoff_nM")
        
        x = subset["cutoff_nM"].values
        y = subset[mean_col].values
        yerr = subset[sem_col].values if sem_col in subset.columns else None
        
        # Plot line
        ax.plot(x, y, marker="o", label=model_key, linewidth=2, markersize=6)
        
        # Add error bars if available
        if yerr is not None:
            ax.errorbar(x, y, yerr=yerr, fmt='none', capsize=3, alpha=0.5)

    ax.set_xscale("log")
    ax.set_xlabel("Affinity Cutoff (nM)", fontsize=12, fontweight="bold")
    ax.set_ylabel(metric_col.upper().replace("_", " ") + " (mean ± SEM)", fontsize=12, fontweight="bold")
    ax.set_title(f"Cutoff Sensitivity: {metric_col.upper()}", fontsize=14, fontweight="bold")
    ax.legend(title="Model", fontsize=9, title_fontsize=10)
    ax.grid(True, alpha=0.3)

    save_figure(fig, output_dir, basename)
    print(f"Saved {basename}.png/pdf")


def identify_best_cutoffs(df_agg: pd.DataFrame, output_dir: Path) -> None:
    """
    Identify best cutoff per model based on mean EF@1% across replicates.
    Save as JSON.
    
    Args:
        df_agg: Aggregated DataFrame with mean/SEM columns
    """
    best_cutoffs: Dict[str, Dict] = {}

    for model_key in df_agg["model_key"].unique():
        subset = df_agg[df_agg["model_key"] == model_key]
        best_row = subset.loc[subset["ef1_mean"].idxmax()]

        best_cutoffs[model_key] = {
            "cutoff_nM": int(best_row["cutoff_nM"]),
            "ef1_mean": float(best_row["ef1_mean"]),
            "ef1_sem": float(best_row["ef1_sem"]),
            "ef1_count": int(best_row["ef1_count"]),
            "ef5_mean": float(best_row["ef5_mean"]),
            "ef10_mean": float(best_row["ef10_mean"]),
            "roc_auc_mean": float(best_row["roc_auc_mean"]),
            "pr_auc_mean": float(best_row["pr_auc_mean"]),
            "n_mf_mean": float(best_row["n_mf_mean"]),
        }

    best_cutoffs_path = output_dir / "phase2_best_cutoffs.json"
    with best_cutoffs_path.open("w") as f:
        json.dump(best_cutoffs, f, indent=2)

    print(f"\nBest cutoffs per model (saved to {best_cutoffs_path}):")
    for model_key, info in best_cutoffs.items():
        print(f"  {model_key}: {info['cutoff_nM']} nM (EF@1% = {info['ef1_mean']:.2f} ± {info['ef1_sem']:.2f}, n={info['ef1_count']})")


def get_best_configs_per_method(df_agg: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to best configuration per method × representation combination.
    
    Strategy:
    1. For each (method, representation) pair, find the model_key with highest mean EF@1% across all cutoffs
    2. This gives 4 best configs: PCA/features, PCA/fingerprints, UMAP/features, UMAP/fingerprints
    
    Args:
        df_agg: Aggregated Phase 2 metrics DataFrame (with _mean columns)
    
    Returns:
        DataFrame filtered to best configs only (4 model_keys)
    """
    best_models = []
    
    # Use ef1_mean if available (aggregated data), otherwise ef1 (raw data)
    ef_col = "ef1_mean" if "ef1_mean" in df_agg.columns else "ef1"
    
    for method in df_agg["method"].unique():
        for rep in df_agg["representation"].unique():
            subset = df_agg[(df_agg["method"] == method) & (df_agg["representation"] == rep)].copy()
            
            if subset.empty:
                continue
            
            # Compute mean EF@1% across all cutoffs for each model_key
            avg_ef1 = subset.groupby("model_key")[ef_col].mean()
            if not avg_ef1.empty:
                best_model_key = avg_ef1.idxmax()
                best_models.append(best_model_key)
    
    filtered = df_agg[df_agg["model_key"].isin(best_models)].copy()
    return filtered


def plot_cutoff_tier_sensitivity(df_agg: pd.DataFrame, output_dir: Path) -> None:
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
        df_agg: Aggregated Phase 2 metrics with _mean columns (must include ef1_high_mean, ef1_medium_mean, ef1_weak_mean)
        output_dir: Output directory for plots
    """
    # Determine column names (aggregated vs raw)
    ef1_col = "ef1_mean" if "ef1_mean" in df_agg.columns else "ef1"
    ef1_high_col = "ef1_high_mean" if "ef1_high_mean" in df_agg.columns else "ef1_high"
    ef1_medium_col = "ef1_medium_mean" if "ef1_medium_mean" in df_agg.columns else "ef1_medium"
    ef1_weak_col = "ef1_weak_mean" if "ef1_weak_mean" in df_agg.columns else "ef1_weak"
    
    # Check if stratified metrics are available
    if ef1_high_col not in df_agg.columns or df_agg[ef1_high_col].isna().all():
        print("WARNING: No stratified metrics found. Run phase2_add_stratified_metrics.py first.")
        print("Skipping tier-wise cutoff sensitivity plot.")
        return
    
    # Filter to best configs per method
    df_best = get_best_configs_per_method(df_agg)
    
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
    
    # Compute global y-axis range across ALL models and tiers for consistency
    all_values = []
    for col in [ef1_col, ef1_high_col, ef1_medium_col, ef1_weak_col]:
        if col in df_best.columns:
            valid_vals = df_best[col].dropna().values
            if len(valid_vals) > 0:
                all_values.extend(valid_vals)
    
    if len(all_values) == 0:
        print("WARNING: No valid EF@1% values. Skipping plot.")
        return
    
    y_min = max(0, np.min(all_values) * 0.95)
    y_max = np.max(all_values) * 1.05
    
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
        y_all = subset[ef1_col].values
        ax.plot(x, y_all, marker="o", label="All", color=colors["All"], 
                linewidth=2.5, markersize=8, alpha=0.9)
        
        # High potency (0.1-100 nM)
        y_high = subset[ef1_high_col].values
        mask_high = ~np.isnan(y_high)
        if mask_high.any():
            ax.plot(x[mask_high], y_high[mask_high], marker="s", 
                   label=f"High (n={n_high})", 
                   color=colors["High"], linewidth=2, markersize=7, alpha=0.8)
        
        # Medium potency (100-1000 nM)
        y_medium = subset[ef1_medium_col].values
        mask_medium = ~np.isnan(y_medium)
        if mask_medium.any():
            ax.plot(x[mask_medium], y_medium[mask_medium], marker="^", 
                   label=f"Medium (n={n_medium})", 
                   color=colors["Medium"], linewidth=2, markersize=7, alpha=0.8)
        
        # Weak potency (1K-100K nM)
        y_weak = subset[ef1_weak_col].values
        mask_weak = ~np.isnan(y_weak)
        if mask_weak.any():
            ax.plot(x[mask_weak], y_weak[mask_weak], marker="D", 
                   label=f"Weak (n={n_weak})", 
                   color=colors["Weak"], linewidth=2, markersize=7, alpha=0.8)        # Formatting
        ax.set_xscale("log")
        ax.set_ylim(y_min, y_max)  # Apply consistent y-axis range
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
        best_idx = subset[ef1_col].idxmax()
        best_cutoff = subset.loc[best_idx, "cutoff_nM"]
        ax.axvline(best_cutoff, color="gray", linestyle="--", linewidth=1.5, alpha=0.6, 
                  label=f"Optimal: {int(best_cutoff)} nM")
    
    # Hide unused axes
    for idx in range(n_models, len(axes)):
        axes[idx].axis("off")
    
    plt.suptitle("Potency-Tier Stratified Cutoff Sensitivity (Best Configs)", 
                fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    save_figure(fig, output_dir, "cutoff_tier_sensitivity_best_configs")
    print(f"Saved cutoff_tier_sensitivity_best_configs.png/pdf")
    
    # Save tier-wise metrics for best configs
    tier_cols = ["model_key", "method", "representation", "dim", "cutoff_nM", 
                 ef1_col, ef1_high_col, ef1_medium_col, ef1_weak_col,
                 "n_actives_high", "n_actives_medium", "n_actives_weak"]
    # Only include columns that exist
    tier_cols = [c for c in tier_cols if c in df_best.columns]
    df_best[tier_cols].to_csv(output_dir / "phase2_tier_metrics_best_configs.csv", index=False)
    print(f"Saved tier metrics (best configs): {output_dir / 'phase2_tier_metrics_best_configs.csv'}")


def plot_cutoff_tier_bedroc_ief_sensitivity(df_agg: pd.DataFrame, output_dir: Path) -> None:
    """
    Plot tier-stratified BEDROC and IEF vs affinity cutoff for best configs per method.
    
    Creates 4 plots (BEDROC α=20, BEDROC α=160, IEF α=20, IEF α=160), each showing:
    - Overall metric (all actives)
    - High-potency metric (High tier actives only)
    - Medium-potency metric (Medium tier actives only)
    - Weak-potency metric (Weak tier actives only)
    
    Interpretation: "BEDROC_High" = How well does ranking enrich high-potency actives specifically?
    
    Args:
        df_agg: Aggregated Phase 2 metrics with tier-specific BEDROC/IEF _mean columns
        output_dir: Output directory for plots
    """
    # Check if tier-stratified BEDROC/IEF metrics are available (try _mean first, fallback to raw)
    required_cols_mean = ["bedroc_20_high_mean", "bedroc_20_medium_mean", "bedroc_20_weak_mean"]
    required_cols_raw = ["bedroc_20_high", "bedroc_20_medium", "bedroc_20_weak"]
    
    if all(col in df_agg.columns for col in required_cols_mean):
        use_mean = True
    elif all(col in df_agg.columns for col in required_cols_raw):
        use_mean = False
    else:
        print("WARNING: No tier-stratified BEDROC/IEF metrics found.")
        print("Skipping tier-stratified BEDROC/IEF sensitivity plots.")
        return
    
    # Filter to best configs per method
    df_best = get_best_configs_per_method(df_agg)
    
    if df_best.empty:
        print("WARNING: No best configs identified. Skipping tier-stratified BEDROC/IEF plot.")
        return
    
    print(f"\nBest configurations per method × representation (for tier-stratified BEDROC/IEF plot):")
    for model_key in sorted(df_best["model_key"].unique()):
        method = df_best[df_best["model_key"] == model_key]["method"].iloc[0]
        rep = df_best[df_best["model_key"] == model_key]["representation"].iloc[0]
        print(f"  {method}/{rep}: {model_key}")
    
    # Sort by cutoff for proper line plotting
    df_best = df_best.sort_values(by=["model_key", "cutoff_nM"]).reset_index(drop=True)
    
    # Define column names based on whether we're using aggregated or raw data
    suffix = "_mean" if use_mean else ""
    
    # Group metrics with same y-axis range
    metric_groups = [
        ([(f"bedroc_20{suffix}", f"bedroc_20_high{suffix}", f"bedroc_20_medium{suffix}", f"bedroc_20_weak{suffix}", "BEDROC (α=20)")],
         "bedroc_20", "BEDROC (α=20)"),
        ([(f"bedroc_160{suffix}", f"bedroc_160_high{suffix}", f"bedroc_160_medium{suffix}", f"bedroc_160_weak{suffix}", "BEDROC (α=160)")],
         "bedroc_160", "BEDROC (α=160)"),
        ([(f"ief_20{suffix}", f"ief_20_high{suffix}", f"ief_20_medium{suffix}", f"ief_20_weak{suffix}", "IEF (α=20)")],
         "ief_20", "IEF (α=20)"),
        ([(f"ief_160{suffix}", f"ief_160_high{suffix}", f"ief_160_medium{suffix}", f"ief_160_weak{suffix}", "IEF (α=160)")],
         "ief_160", "IEF (α=160)"),
    ]
    
    # Color scheme for tiers (same as EF plot)
    colors = {
        "All": "#2E7D32",      # Green (overall)
        "High": "#1976D2",     # Blue (high potency)
        "Medium": "#F57C00",   # Orange (medium)
        "Weak": "#C62828",     # Red (weak)
    }
    
    for metric_spec, file_prefix, group_title in metric_groups:
        all_col, high_col, medium_col, weak_col, _ = metric_spec[0]
        
        # Check if columns exist
        if all_col not in df_best.columns:
            print(f"WARNING: {all_col} not in dataframe. Skipping {group_title} plot.")
            continue
        
        # Compute global y-axis range for this metric group
        all_values = []
        for col in [all_col, high_col, medium_col, weak_col]:
            if col in df_best.columns:
                valid_vals = df_best[col].dropna().values
                if len(valid_vals) > 0:
                    all_values.extend(valid_vals)
        
        if len(all_values) == 0:
            print(f"WARNING: No valid {group_title} values. Skipping plot.")
            continue
        
        y_min = max(0, np.min(all_values) * 0.95)
        y_max = np.max(all_values) * 1.05
        
        # Create multi-panel plot (2×2 grid for 4 model_keys)
        n_models = df_best["model_key"].nunique()
        
        if n_models <= 2:
            fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), sharey=True, squeeze=False)
            axes = axes.flatten()
        else:
            ncols = 2
            nrows = int(np.ceil(n_models / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(12, 5 * nrows), sharey=True, squeeze=False)
            axes = axes.flatten()
        
        for idx, model_key in enumerate(sorted(df_best["model_key"].unique())):
            ax = axes[idx]
            subset = df_best[df_best["model_key"] == model_key].copy()
            
            # Get tier counts
            n_high = subset["n_actives_high"].iloc[0] if "n_actives_high" in subset.columns else 0
            n_medium = subset["n_actives_medium"].iloc[0] if "n_actives_medium" in subset.columns else 0
            n_weak = subset["n_actives_weak"].iloc[0] if "n_actives_weak" in subset.columns else 0
            
            x = subset["cutoff_nM"].values
            
            # All actives (baseline)
            y_all = subset[all_col].values
            mask_all = ~np.isnan(y_all)
            if mask_all.any():
                ax.plot(x[mask_all], y_all[mask_all], marker="o", label="All", 
                       color=colors["All"], linewidth=2.5, markersize=8, alpha=0.9)
            
            # High potency tier
            if high_col in subset.columns:
                y_high = subset[high_col].values
                mask_high = ~np.isnan(y_high)
                if mask_high.any():
                    ax.plot(x[mask_high], y_high[mask_high], marker="s", 
                           label=f"High (n={n_high})", 
                           color=colors["High"], linewidth=2, markersize=7, alpha=0.8)
            
            # Medium potency tier
            if medium_col in subset.columns:
                y_medium = subset[medium_col].values
                mask_medium = ~np.isnan(y_medium)
                if mask_medium.any():
                    ax.plot(x[mask_medium], y_medium[mask_medium], marker="^", 
                           label=f"Medium (n={n_medium})", 
                           color=colors["Medium"], linewidth=2, markersize=7, alpha=0.8)
            
            # Weak potency tier
            if weak_col in subset.columns:
                y_weak = subset[weak_col].values
                mask_weak = ~np.isnan(y_weak)
                if mask_weak.any():
                    ax.plot(x[mask_weak], y_weak[mask_weak], marker="D", 
                           label=f"Weak (n={n_weak})", 
                           color=colors["Weak"], linewidth=2, markersize=7, alpha=0.8)
            
            # Formatting
            ax.set_xscale("log")
            ax.set_ylim(y_min, y_max)  # Apply consistent y-axis range
            ax.set_xlabel("Affinity Cutoff (nM)", fontsize=11, fontweight="bold")
            if idx == 0:
                ax.set_ylabel(group_title, fontsize=11, fontweight="bold")
            
            # Extract method and representation for title
            method = subset["method"].iloc[0]
            rep = subset["representation"].iloc[0]
            dim = subset["dim"].iloc[0]
            
            title = f"{method.upper()} ({rep}, dim={dim})\n"
            title += f"Tiers: High={n_high}, Medium={n_medium}, Weak={n_weak}"
            ax.set_title(title, fontsize=11, fontweight="bold")
            
            ax.legend(loc="best", fontsize=9, framealpha=0.9)
            ax.grid(True, alpha=0.3)
            
            # Add vertical line at optimal cutoff (based on overall metric)
            if mask_all.any():
                best_idx = subset.loc[mask_all, all_col].idxmax()
                best_cutoff = subset.loc[best_idx, "cutoff_nM"]
                ax.axvline(best_cutoff, color="gray", linestyle="--", linewidth=1.5, alpha=0.6)
        
        # Hide unused axes
        for idx in range(n_models, len(axes)):
            axes[idx].axis("off")
        
        plt.suptitle(f"Potency-Tier Stratified Cutoff Sensitivity: {group_title} (Best Configs)", 
                    fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        
        save_figure(fig, output_dir, f"cutoff_tier_{file_prefix}_sensitivity_best_configs")
        print(f"Saved cutoff_tier_{file_prefix}_sensitivity_best_configs.png/pdf")


def plot_cutoff_bedroc_ief_sensitivity(df_agg: pd.DataFrame, output_dir: Path) -> None:
    """
    Plot BEDROC and IEF vs affinity cutoff for best configs per method.
    
    Shows how different cutoffs affect BEDROC and IEF metrics (overall actives only,
    not tier-specific since BEDROC/IEF per tier would require filtering which changes denominator).
    
    Args:
        df_agg: Aggregated Phase 2 metrics with _mean columns (must include bedroc_20_mean, bedroc_160_mean, ief_20_mean, ief_160_mean)
        output_dir: Output directory for plots
    """
    # Check if BEDROC/IEF metrics are available (try _mean first, fallback to raw)
    bedroc_col = "bedroc_20_mean" if "bedroc_20_mean" in df_agg.columns else "bedroc_20"
    
    if bedroc_col not in df_agg.columns or df_agg[bedroc_col].isna().all():
        print("WARNING: No BEDROC/IEF metrics found. Skipping BEDROC/IEF cutoff sensitivity plot.")
        return
    
    # Determine if we're using aggregated or raw data
    use_mean = "bedroc_20_mean" in df_agg.columns
    suffix = "_mean" if use_mean else ""
    
    # Filter to best configs per method
    df_best = get_best_configs_per_method(df_agg)
    
    if df_best.empty:
        print("WARNING: No best configs identified. Skipping BEDROC/IEF plot.")
        return
    
    print(f"\nBest configurations per method × representation (for BEDROC/IEF plot):")
    for model_key in sorted(df_best["model_key"].unique()):
        method = df_best[df_best["model_key"] == model_key]["method"].iloc[0]
        rep = df_best[df_best["model_key"] == model_key]["representation"].iloc[0]
        print(f"  {method}/{rep}: {model_key}")
    
    # Sort by cutoff for proper line plotting
    df_best = df_best.sort_values(by=["model_key", "cutoff_nM"]).reset_index(drop=True)
    
    # Create plots for BEDROC(α=20), BEDROC(α=160), IEF(α=20), IEF(α=160)
    # Group metrics with same y-axis range
    metric_groups = [
        ([(f"bedroc_20{suffix}", "BEDROC (α=20)"), (f"bedroc_160{suffix}", "BEDROC (α=160)")], "BEDROC"),
        ([(f"ief_20{suffix}", "IEF (α=20)"), (f"ief_160{suffix}", "IEF (α=160)")], "IEF"),
    ]
    
    for metric_pairs, group_name in metric_groups:
        # Compute global y-axis range for this group
        all_values = []
        for metric_col, _ in metric_pairs:
            if metric_col in df_best.columns:
                valid_vals = df_best[metric_col].dropna().values
                if len(valid_vals) > 0:
                    all_values.extend(valid_vals)
        
        if len(all_values) == 0:
            print(f"WARNING: No valid {group_name} values. Skipping {group_name} plots.")
            continue
        
        y_min = min(all_values)
        y_max = max(all_values)
        y_range = y_max - y_min
        y_limits = (y_min - 0.05 * y_range, y_max + 0.05 * y_range)
        
        # Create plot for each metric in the group
        for metric_col, metric_title in metric_pairs:
            if metric_col not in df_best.columns:
                print(f"WARNING: {metric_col} not in dataframe. Skipping {metric_title} plot.")
                continue
            
            # Create multi-panel plot (2×2 grid for 4 model_keys)
            n_models = df_best["model_key"].nunique()
            
            if n_models <= 2:
                fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), sharey=True, squeeze=False)
                axes = axes.flatten()
            else:
                ncols = 2
                nrows = int(np.ceil(n_models / ncols))
                fig, axes = plt.subplots(nrows, ncols, figsize=(12, 5 * nrows), sharey=True, squeeze=False)
                axes = axes.flatten()
            
            for idx, model_key in enumerate(sorted(df_best["model_key"].unique())):
                ax = axes[idx]
                subset = df_best[df_best["model_key"] == model_key].copy()
                
                # Plot metric vs cutoff
                x = subset["cutoff_nM"].values
                y = subset[metric_col].values
                
                mask = ~np.isnan(y)
                if mask.any():
                    ax.plot(x[mask], y[mask], marker="o", color="#1976D2",
                           linewidth=2.5, markersize=8, alpha=0.9)
                
                # Formatting
                ax.set_xscale("log")
                ax.set_xlabel("Affinity Cutoff (nM)", fontsize=11, fontweight="bold")
                ax.set_ylim(y_limits)  # Apply consistent y-axis range
                if idx == 0:
                    ax.set_ylabel(metric_title, fontsize=11, fontweight="bold")
                
                # Extract method and representation for title
                method = subset["method"].iloc[0]
                rep = subset["representation"].iloc[0]
                dim = subset["dim"].iloc[0]
                
                ax.set_title(f"{method.upper()} ({rep}, dim={dim})", fontsize=11, fontweight="bold")
                ax.grid(True, alpha=0.3)
                
                # Add vertical line at optimal cutoff (based on metric value)
                if mask.any():
                    best_idx = subset.loc[mask, metric_col].idxmax()
                    best_cutoff = subset.loc[best_idx, "cutoff_nM"]
                    ax.axvline(best_cutoff, color="gray", linestyle="--", linewidth=1.5, alpha=0.6)
            
            # Hide unused axes
            for idx in range(n_models, len(axes)):
                axes[idx].axis("off")
            
            plt.suptitle(f"Cutoff Sensitivity: {metric_title} (Best Configs)", 
                        fontsize=14, fontweight="bold", y=1.02)
            plt.tight_layout()
            
            filename = f"cutoff_{metric_col}_sensitivity_best_configs"
            save_figure(fig, output_dir, filename)
            print(f"Saved {filename}.png/pdf")


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

    # Aggregate results (with tier-stratified metrics, one row per replicate)
    print("Aggregating Phase 2 results and computing tier-stratified metrics...")
    print("(This may take a few minutes...)")
    df_raw = aggregate_phase2_results(phase2_dir, workspace_dir, compute_tiers=True)

    if df_raw.empty:
        print("ERROR: No Phase 2 results found. Check workspace directory.")
        return

    print(f"Found {len(df_raw)} individual results:")
    print(f"  {df_raw['model_key'].nunique()} model families")
    print(f"  {df_raw['cutoff_nM'].nunique()} cutoffs")
    print(f"  {df_raw['replicate'].nunique()} replicates per configuration")
    
    # Report tier counts
    if "n_actives_high" in df_raw.columns and df_raw["n_actives_high"].notna().any():
        # Get unique tier counts (should be same across cutoffs/replicates for a given target)
        tier_summary = df_raw[["n_actives_high", "n_actives_medium", "n_actives_weak"]].drop_duplicates()
        if len(tier_summary) == 1:
            n_high = int(tier_summary["n_actives_high"].iloc[0])
            n_medium = int(tier_summary["n_actives_medium"].iloc[0])
            n_weak = int(tier_summary["n_actives_weak"].iloc[0])
            print(f"\nPotency tier counts:")
            print(f"  High (0.1-100 nM): {n_high} actives")
            print(f"  Medium (100-1K nM): {n_medium} actives")
            print(f"  Weak (1K-100K nM): {n_weak} actives")
    print()

    # Save raw (per-replicate) metrics
    df_raw.to_csv(output_dir / "phase2_raw_metrics_per_replicate.csv", index=False)
    print(f"Saved raw metrics (per replicate): {output_dir / 'phase2_raw_metrics_per_replicate.csv'}")

    # Aggregate across replicates (compute mean ± SEM)
    print("\nAggregating metrics across replicates (mean ± SEM)...")
    df_agg = aggregate_replicates(df_raw)
    print(f"Aggregated to {len(df_agg)} configurations (model_key × cutoff)")
    
    # Save aggregated metrics
    df_agg.to_csv(output_dir / "phase2_aggregated_metrics.csv", index=False)
    print(f"Saved aggregated metrics: {output_dir / 'phase2_aggregated_metrics.csv'}")
    print()

    # Generate plots (using aggregated data with mean ± SEM)
    print("Generating plots...")

    # Cutoff curves for EF
    plot_cutoff_curves(df_agg, "ef1", output_dir, "cutoff_curves_ef1")
    plot_cutoff_curves(df_agg, "ef5", output_dir, "cutoff_curves_ef5")
    plot_cutoff_curves(df_agg, "ef10", output_dir, "cutoff_curves_ef10")
    
    # Cutoff curves for BEDROC and IEF
    plot_cutoff_curves(df_agg, "bedroc_20", output_dir, "cutoff_curves_bedroc_20")
    plot_cutoff_curves(df_agg, "bedroc_160", output_dir, "cutoff_curves_bedroc_160")
    plot_cutoff_curves(df_agg, "ief_20", output_dir, "cutoff_curves_ief_20")
    plot_cutoff_curves(df_agg, "ief_160", output_dir, "cutoff_curves_ief_160")

    # Tier-wise cutoff sensitivity (best configs only)
    plot_cutoff_tier_sensitivity(df_agg, output_dir)

    # Tier-wise BEDROC and IEF cutoff sensitivity (best configs only)
    plot_cutoff_tier_bedroc_ief_sensitivity(df_agg, output_dir)

    # BEDROC and IEF cutoff sensitivity (best configs only) - overall only, no tiers
    plot_cutoff_bedroc_ief_sensitivity(df_agg, output_dir)

    # Identify best cutoffs
    identify_best_cutoffs(df_agg, output_dir)

    print()
    print("="*80)
    print("PHASE 2 POST-ANALYSIS COMPLETED")
    print("="*80)
    print(f"All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
