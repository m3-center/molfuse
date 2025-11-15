#!/usr/bin/env python3
"""
Phase 4 Post-Analysis: Cross-Target Generalization Study

Aggregates Phase 4 results and generates publication-quality plots:
- Cross-target performance comparison (EF@1% across 9 kinases)
- Correlation: Natural MF cloud size vs High-potency EF@1%
- Potency-stratified enrichment across targets
- Target-specific degradation curves
- Statistical validation (Spearman correlation + significance)

Targets (Tyrosine Kinases):
- ABL1, ABL2, EGFR, ERBB2, SRC, LCK, YES1, FYN, LYN

Potency Tiers (nM):
- High:   0.1 ≤ affinity ≤ 100
- Medium: 100 < affinity ≤ 1,000  
- Weak:   1,000 < affinity ≤ 100,000

Usage (HPC-ready):
    python scripts/phase4_post_analysis.py \
        --workspace_dir experiment_workspace_v4 \
        --output_dir reporting/phase4_post_analysis \
        --stratify
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

# Import BEDROC, IEF, and EF metrics
from molfuse.metrics.metrics import bedroc, ief, ef_at_k_percent, roc_auc, pr_auc
from molfuse.scoring.nn import nn_min_distance_scores

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


def collect_phase4_results(
    workspace_dir: Path,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Scan workspace/phase4/ for all target summary JSONs and build master DataFrame.
    
    Expected structure:
        workspace_dir/phase4/{target}/run_{config}/logs/phase4_summary.json
    
    Returns:
        DataFrame with all Phase 4 cross-target results
    """
    phase4_dir = workspace_dir / "phase4"
    
    if not phase4_dir.exists():
        raise FileNotFoundError(f"Phase 4 directory not found: {phase4_dir}")
    
    logger.info(f"Scanning {phase4_dir} for phase4_summary.json files...")
    
    # Find all summary files
    summary_files = list(phase4_dir.rglob("phase4_summary.json"))
    logger.info(f"Found {len(summary_files)} summary files")
    
    if len(summary_files) == 0:
        raise ValueError("No phase4_summary.json files found!")
    
    # Parse each summary
    records = []
    for summary_path in summary_files:
        try:
            with summary_path.open("r") as f:
                data = json.load(f)
            
            # Extract key fields
            record = {
                "run_name": data.get("run_name"),
                "target": data.get("target"),
                "target_chembl_id": data.get("target_chembl_id"),
                "method": data.get("method"),
                "representation": data.get("representation"),
                "dim": data.get("dim"),
                "affinity_cutoff_nM": data.get("affinity_cutoff_nM"),
                "replicate": data.get("config", {}).get("replicate", 1),
                "random_seed": data.get("random_seed"),
                "natural_mf_size": data.get("natural_mf_size"),  # Pre-cutoff size
                "mf_size_actual": data.get("mf_size_actual"),      # Post-cutoff size
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
            artifacts_dir = summary_path.parent.parent / "artifacts"
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
    logger.info(f"Loaded {len(df)} Phase 4 results")
    logger.info(f"  Targets: {sorted(df['target'].unique().tolist())}")
    logger.info(f"  Methods: {df['method'].unique().tolist()}")
    logger.info(f"  Representations: {df['representation'].unique().tolist()}")
    
    # DEBUGGING: Show replicate counts per target
    logger.info("\n" + "="*80)
    logger.info("DEBUGGING: Replicate counts per target")
    logger.info("="*80)
    
    target_counts = df.groupby(['target', 'method', 'representation']).size().reset_index(name='n_replicates')
    target_counts = target_counts.sort_values(['target', 'method', 'representation'])
    
    logger.info("\nExpected: 5 replicates per target-method-representation combo")
    logger.info(f"Actual: {len(target_counts)} conditions found\n")
    
    for _, row in target_counts.iterrows():
        status = "✓ OK" if row['n_replicates'] == 5 else "✗ IRREGULAR"
        logger.info(f"  {status:12s} | {row['target']:8s} | {row['method']:4s} / {row['representation']:12s} | n={row['n_replicates']}")
    
    # Summary of irregular conditions
    irregular = target_counts[target_counts['n_replicates'] != 5]
    if len(irregular) > 0:
        logger.warning(f"\n⚠️  Found {len(irregular)} conditions with irregular replicate counts:")
        for _, row in irregular.iterrows():
            logger.warning(f"    {row['target']}/{row['method']}/{row['representation']}: {row['n_replicates']} replicates (expected 5)")
    else:
        logger.info("\n✓ All conditions have exactly 5 replicates")
    
    logger.info("="*80 + "\n")
    
    return df


def aggregate_by_target(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Aggregate results by (target, method, representation).
    
    Computes mean, SEM, std for all metrics across replicates.
    """
    logger.info("Aggregating results by target...")
    
    group_keys = ["target", "target_chembl_id", "method", "representation", "dim", 
                  "affinity_cutoff_nM", "natural_mf_size"]
    
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
        "spearman_rho": ["mean", "sem", "std"],
        "n_actives": "first",
        "n_zinc": "first",
        "mf_size_actual": "mean",  # Average actual size after cutoff
    }
    
    df_agg = df.groupby(group_keys, dropna=False).agg(agg_dict).reset_index()
    
    # Flatten column names
    df_agg.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in df_agg.columns
    ]
    
    logger.info(f"Aggregated to {len(df_agg)} unique target-method conditions")
    
    return df_agg


# ============================================================================
# Affinity Cutoff Sensitivity Analysis
# ============================================================================

def compute_metrics_for_cutoff(
    Z_mf: np.ndarray,
    Z_eval: np.ndarray,
    labels: np.ndarray,
    mf_affinity: np.ndarray,
    cutoff_nM: float,
    logger: logging.Logger
) -> Optional[Dict[str, float]]:
    """
    Re-score evaluation set using MF filtered by affinity cutoff.
    
    Args:
        Z_mf: MF embeddings (n_mf, dim)
        Z_eval: Evaluation embeddings (n_eval, dim) - actives + ZINC
        labels: Binary labels (1=active, 0=ZINC)
        mf_affinity: MF affinity values (nM)
        cutoff_nM: Affinity threshold
        logger: Logger
    
    Returns:
        Dict with metrics or None if MF cloud is empty after filtering
    """
    # Filter MF by cutoff
    mask = mf_affinity <= cutoff_nM
    Z_mf_filtered = Z_mf[mask]
    
    if len(Z_mf_filtered) == 0:
        logger.warning(f"MF cloud empty at cutoff {cutoff_nM} nM")
        return None
    
    # Re-score
    scores, distances = nn_min_distance_scores(Z_mf_filtered, Z_eval)
    
    # Compute metrics
    ef1 = ef_at_k_percent(scores, labels, 1.0)
    roc = roc_auc(labels, scores)
    pr = pr_auc(labels, scores)
    bedroc_20 = bedroc(labels, scores, alpha=20.0)
    bedroc_160 = bedroc(labels, scores, alpha=160.9)
    ief_20 = ief(labels, scores, alpha=20.0)
    ief_160 = ief(labels, scores, alpha=160.9)
    
    return {
        "cutoff_nM": float(cutoff_nM),
        "n_mf": int(len(Z_mf_filtered)),
        "ef_1%": float(ef1),
        "roc_auc": float(roc),
        "pr_auc": float(pr),
        "bedroc_20": float(bedroc_20),
        "bedroc_160": float(bedroc_160),
        "ief_20": float(ief_20),
        "ief_160": float(ief_160),
    }


def run_cutoff_sensitivity_for_run(
    workspace_dir: Path,
    run_name: str,
    cutoffs: List[float],
    logger: logging.Logger
) -> List[Dict[str, float]]:
    """
    Run affinity cutoff sensitivity analysis for a single Phase 4 run.
    
    Loads pre-computed embeddings and re-scores across cutoff range.
    """
    # Phase 4 structure: workspace/phase4/cross_target/{run_name}/artifacts
    artifacts_dir = workspace_dir / "phase4" / "cross_target" / run_name / "artifacts"
    
    # Load embeddings
    mf_path = artifacts_dir / "embedding_mf.csv"
    zinc_path = artifacts_dir / "embedding_zinc.csv"
    actives_path = artifacts_dir / "embedding_actives.csv"
    
    if not all(p.exists() for p in [mf_path, zinc_path, actives_path]):
        logger.warning(f"Missing embeddings for {run_name}")
        return []
    
    try:
        df_mf = pd.read_csv(mf_path)
        df_zinc = pd.read_csv(zinc_path)
        df_act = pd.read_csv(actives_path)
    except Exception as e:
        logger.warning(f"Failed to load embeddings for {run_name}: {e}")
        return []
    
    # Extract embedding dimensions
    dim_cols = [c for c in df_mf.columns if c.startswith("dim_")]
    
    Z_mf = df_mf[dim_cols].values
    Z_zinc = df_zinc[dim_cols].values
    Z_act = df_act[dim_cols].values
    
    # Build evaluation set
    Z_eval = np.vstack([Z_act, Z_zinc])
    labels = np.concatenate([
        np.ones(len(Z_act), dtype=int),
        np.zeros(len(Z_zinc), dtype=int)
    ])
    
    # Get MF affinity values
    if "Standard Value (nM)" not in df_mf.columns:
        logger.warning(f"No affinity column in {mf_path}")
        return []
    
    mf_affinity = pd.to_numeric(df_mf["Standard Value (nM)"], errors="coerce").values
    
    # Run cutoff sweep
    results = []
    for cutoff in cutoffs:
        metrics = compute_metrics_for_cutoff(Z_mf, Z_eval, labels, mf_affinity, cutoff, logger)
        if metrics is not None:
            results.append(metrics)
    
    return results


def aggregate_cutoff_sensitivity(
    workspace_dir: Path,
    df_runs: pd.DataFrame,
    cutoffs: List[float],
    output_dir: Path,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Run cutoff sensitivity analysis for all Phase 4 runs and aggregate.
    
    Returns:
        DataFrame with cutoff sensitivity metrics per target per cutoff
    """
    logger.info("\n" + "="*80)
    logger.info("AFFINITY CUTOFF SENSITIVITY ANALYSIS")
    logger.info("="*80)
    logger.info(f"Cutoffs: {cutoffs} nM")
    
    all_records = []
    
    for _, row in df_runs.iterrows():
        target = row["target"]
        run_name = row["run_name"]
        
        logger.info(f"Processing {run_name}...")
        
        cutoff_results = run_cutoff_sensitivity_for_run(
            workspace_dir, run_name, cutoffs, logger
        )
        
        for cutoff_metrics in cutoff_results:
            record = {
                "run_name": run_name,
                "target": row["target"],
                "method": row["method"],
                "representation": row["representation"],
                "replicate": row["replicate"],
                **cutoff_metrics
            }
            all_records.append(record)
    
    if not all_records:
        logger.error("No cutoff sensitivity results computed!")
        return pd.DataFrame()
    
    df_cutoff = pd.DataFrame(all_records)
    
    # Save full results
    cutoff_csv = output_dir / "phase4_cutoff_sensitivity_full.csv"
    df_cutoff.to_csv(cutoff_csv, index=False)
    logger.info(f"Saved: {cutoff_csv.name} ({len(df_cutoff)} rows)")
    
    # Aggregate by target and cutoff
    group_keys = ["target", "method", "representation", "cutoff_nM"]
    agg_dict = {
        "n_mf": "mean",
        "ef_1%": ["mean", "sem", "std"],
        "roc_auc": ["mean", "sem"],
        "pr_auc": ["mean", "sem"],
        "bedroc_20": ["mean", "sem"],
        "bedroc_160": ["mean", "sem"],
        "ief_20": ["mean", "sem"],
        "ief_160": ["mean", "sem"],
    }
    
    df_agg = df_cutoff.groupby(group_keys, dropna=False).agg(agg_dict).reset_index()
    df_agg.columns = ["_".join(col).strip("_") if isinstance(col, tuple) else col for col in df_agg.columns]
    
    # Save aggregated results
    agg_csv = output_dir / "phase4_cutoff_sensitivity_aggregated.csv"
    df_agg.to_csv(agg_csv, index=False)
    logger.info(f"Saved: {agg_csv.name} ({len(df_agg)} rows)")
    
    return df_agg


def plot_cutoff_sensitivity_per_target(
    df_cutoff_agg: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger
) -> None:
    """
    Plot cutoff sensitivity curves: EF@1% vs affinity cutoff, faceted by target.
    """
    logger.info("Generating cutoff sensitivity plots...")
    
    df_cutoff_agg["model_key"] = df_cutoff_agg["method"] + "_" + df_cutoff_agg["representation"]
    
    targets = sorted(df_cutoff_agg["target"].unique())
    n_targets = len(targets)
    
    # Create subplot grid
    n_cols = 3
    n_rows = int(np.ceil(n_targets / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows), sharex=True, sharey=False)
    axes = axes.flatten() if n_targets > 1 else [axes]
    
    colors = {
        "pca_features": "#1f77b4",
        "pca_fingerprints": "#aec7e8",
        "umap_features": "#ff7f0e",
        "umap_fingerprints": "#ffbb78",
    }
    
    for idx, target in enumerate(targets):
        ax = axes[idx]
        
        subset = df_cutoff_agg[df_cutoff_agg["target"] == target]
        
        for model_key in sorted(subset["model_key"].unique()):
            model_subset = subset[subset["model_key"] == model_key].sort_values("cutoff_nM")
            
            x = model_subset["cutoff_nM"].values
            y_mean = model_subset["ef_1%_mean"].values
            y_sem = model_subset["ef_1%_sem"].values
            
            ax.plot(x, y_mean, marker="o", label=model_key.replace("_", "/"),
                   color=colors.get(model_key, "#333333"), linewidth=2, markersize=6)
            ax.fill_between(x, y_mean - y_sem, y_mean + y_sem,
                            color=colors.get(model_key, "#333333"), alpha=0.2)
        
        ax.set_xscale("log")
        ax.set_xlabel("Affinity Cutoff (nM)", fontweight="bold")
        ax.set_ylabel("EF@1% (Mean ± SEM)", fontweight="bold")
        ax.set_title(f"{target}", fontweight="bold")
        ax.grid(True, alpha=0.3)
        
        if idx == 0:
            ax.legend(loc="best", fontsize=8)
    
    # Hide unused subplots
    for idx in range(n_targets, len(axes)):
        axes[idx].axis("off")
    
    plt.tight_layout()
    
    output_path_png = output_dir / "phase4_cutoff_sensitivity_per_target_ef1.png"
    output_path_pdf = output_dir / "phase4_cutoff_sensitivity_per_target_ef1.pdf"
    fig.savefig(output_path_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_path_pdf, bbox_inches="tight")
    plt.close(fig)
    
    logger.info(f"Saved: {output_path_png.name}")


def identify_best_cutoffs_per_target(
    df_cutoff_agg: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger
) -> Dict[str, Dict[str, float]]:
    """
    Identify optimal affinity cutoff for each target based on EF@1%.
    
    Returns:
        Dict[target][model_key] -> {"cutoff_nM": X, "ef_1%": Y, "n_mf": Z}
    """
    logger.info("Identifying best cutoffs per target...")
    
    df_cutoff_agg["model_key"] = df_cutoff_agg["method"] + "_" + df_cutoff_agg["representation"]
    
    best_cutoffs = {}
    
    for target in df_cutoff_agg["target"].unique():
        best_cutoffs[target] = {}
        
        for model_key in df_cutoff_agg["model_key"].unique():
            subset = df_cutoff_agg[
                (df_cutoff_agg["target"] == target) &
                (df_cutoff_agg["model_key"] == model_key)
            ].copy()
            
            if len(subset) == 0:
                continue
            
            # Find cutoff with max EF@1%
            best_row = subset.loc[subset["ef_1%_mean"].idxmax()]
            
            best_cutoffs[target][model_key] = {
                "cutoff_nM": float(best_row["cutoff_nM"]),
                "ef_1%": float(best_row["ef_1%_mean"]),
                "n_mf": int(best_row["n_mf_mean"]),
            }
    
    # Save to JSON
    output_path = output_dir / "phase4_best_cutoffs_per_target.json"
    with output_path.open("w") as f:
        json.dump(best_cutoffs, f, indent=2)
    
    logger.info(f"Saved: {output_path.name}")
    
    # Log results
    logger.info("\n" + "="*80)
    logger.info("OPTIMAL CUTOFFS PER TARGET")
    logger.info("="*80)
    
    for target in sorted(best_cutoffs.keys()):
        logger.info(f"\n{target}:")
        for model_key in sorted(best_cutoffs[target].keys()):
            info = best_cutoffs[target][model_key]
            logger.info(f"  {model_key:20s}: {info['cutoff_nM']:>7.0f} nM "
                       f"(EF@1%={info['ef_1%']:>6.2f}, MF={info['n_mf']:>6,})")
    
    logger.info("="*80 + "\n")
    
    return best_cutoffs


# ============================================================================
# Potency Tier Stratification
# ============================================================================

def assign_potency_tier(affinity_nM: float) -> str:
    """Assign potency tier based on affinity (nM)."""
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
    """Compute EF@1% for a specific potency tier."""
    tier_actives = ranked_df[(ranked_df["label"] == 1) & (ranked_df["tier"] == tier)]
    n_tier_actives = len(tier_actives)
    
    if n_tier_actives == 0:
        return None
    
    n_total = len(ranked_df)
    cutoff_idx = int(np.ceil(n_total * 0.01))
    
    top_1pct = ranked_df.head(cutoff_idx)
    n_tier_found = len(top_1pct[(top_1pct["label"] == 1) & (top_1pct["tier"] == tier)])
    
    ef1 = (n_tier_found / n_tier_actives) / 0.01
    
    return ef1


def load_actives_with_tiers(
    workspace_dir: Path,
    target: str,
    run_name: str,
    logger: logging.Logger
) -> Optional[pd.DataFrame]:
    """Load actives embeddings with affinity values and assign potency tiers."""
    artifacts_dir = workspace_dir / "phase4" / target / run_name / "artifacts"
    actives_path = artifacts_dir / "embedding_actives.csv"
    
    if not actives_path.exists():
        logger.warning(f"Actives embedding not found: {actives_path}")
        return None
    
    try:
        df = pd.read_csv(actives_path, usecols=lambda c: c in ["Compound ChEMBL ID", "Standard Value (nM)"])
        df["tier"] = df["Standard Value (nM)"].apply(assign_potency_tier)
        return df
    except Exception as e:
        logger.warning(f"Failed to load actives with tiers from {actives_path}: {e}")
        return None


def compute_stratified_metrics_for_run(
    workspace_dir: Path,
    target: str,
    run_name: str,
    logger: logging.Logger
) -> Optional[Dict[str, float]]:
    """Compute tier-specific EF@1% for a single Phase 4 run."""
    artifacts_dir = workspace_dir / "phase4" / target / run_name / "artifacts"
    
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
    actives_tiers = load_actives_with_tiers(workspace_dir, target, run_name, logger)
    if actives_tiers is None:
        return None
    
    # Merge tier info
    ranked_df = ranked_df.merge(
        actives_tiers[["Compound ChEMBL ID", "tier"]],
        on="Compound ChEMBL ID",
        how="left"
    )
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

def plot_cross_target_comparison(
    df_agg: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger
) -> None:
    """
    Plot cross-target enrichment comparison (bar chart).
    
    X-axis: Targets sorted by natural MF cloud size
    Y-axis: EF@1% (mean ± SEM)
    Colors: Method-representation combos
    """
    logger.info("Generating cross-target comparison bar chart...")
    
    # Create model_key
    df_agg["model_key"] = df_agg["method"] + "_" + df_agg["representation"]
    
    # Sort targets by actual MF size (post-cutoff), fallback to natural if not available
    if "mf_size_actual" in df_agg.columns:
        target_order = df_agg.groupby("target")["mf_size_actual"].mean().sort_values().index.tolist()
    else:
        target_order = df_agg.groupby("target")["natural_mf_size"].mean().sort_values().index.tolist()
    
    model_keys = sorted(df_agg["model_key"].unique())
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(target_order))
    width = 0.2  # Bar width
    
    colors = {
        "pca_features": "#1f77b4",
        "pca_fingerprints": "#aec7e8",
        "umap_features": "#ff7f0e",
        "umap_fingerprints": "#ffbb78",
    }
    
    for idx, model_key in enumerate(model_keys):
        subset = df_agg[df_agg["model_key"] == model_key].copy()
        subset = subset.set_index("target").reindex(target_order)
        
        y_mean = subset["ef_1%_mean"].values
        y_sem = subset["ef_1%_sem"].values
        
        offset = (idx - len(model_keys)/2 + 0.5) * width
        
        ax.bar(
            x + offset, y_mean, width,
            yerr=y_sem,
            label=model_key.replace("_", "/"),
            color=colors.get(model_key, "#333333"),
            capsize=3
        )
    
    ax.set_xlabel("Target (sorted by natural MF cloud size)", fontweight="bold", fontsize=12)
    ax.set_ylabel("EF@1% (Mean ± SEM)", fontweight="bold", fontsize=12)
    ax.set_title("Phase 4: Cross-Target Generalization", fontweight="bold", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(target_order, rotation=45, ha="right")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    
    output_path_png = output_dir / "phase4_cross_target_comparison.png"
    output_path_pdf = output_dir / "phase4_cross_target_comparison.pdf"
    fig.savefig(output_path_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_path_pdf, bbox_inches="tight")
    plt.close(fig)
    
    logger.info(f"Saved: {output_path_png.name}")


def plot_cross_target_bedroc_ief(
    df_agg: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger
) -> None:
    """
    Plot cross-target BEDROC and IEF comparison (bar charts).
    Creates separate plots for BEDROC(α=20), BEDROC(α=160), IEF(α=20), IEF(α=160).
    """
    logger.info("Generating cross-target BEDROC and IEF comparison bar charts...")
    
    df_agg["model_key"] = df_agg["method"] + "_" + df_agg["representation"]
    # Sort targets by actual MF size (post-cutoff)
    if "mf_size_actual" in df_agg.columns:
        target_order = df_agg.groupby("target")["mf_size_actual"].mean().sort_values().index.tolist()
    else:
        target_order = df_agg.groupby("target")["natural_mf_size"].mean().sort_values().index.tolist()
    model_keys = sorted(df_agg["model_key"].unique())
    
    colors = {
        "pca_features": "#1f77b4",
        "pca_fingerprints": "#aec7e8",
        "umap_features": "#ff7f0e",
        "umap_fingerprints": "#ffbb78",
    }
    
    metrics = [
        ("bedroc_20", "BEDROC (α=20)"),
        ("bedroc_160", "BEDROC (α=160)"),
        ("ief_20", "IEF (α=20)"),
        ("ief_160", "IEF (α=160)"),
    ]
    
    for metric_col, metric_title in metrics:
        fig, ax = plt.subplots(figsize=(14, 6))
        
        x = np.arange(len(target_order))
        width = 0.2
        
        for idx, model_key in enumerate(model_keys):
            subset = df_agg[df_agg["model_key"] == model_key].copy()
            subset = subset.set_index("target").reindex(target_order)
            
            y_mean = subset[f"{metric_col}_mean"].values
            y_sem = subset[f"{metric_col}_sem"].values
            
            offset = (idx - len(model_keys)/2 + 0.5) * width
            
            ax.bar(
                x + offset, y_mean, width,
                yerr=y_sem,
                label=model_key.replace("_", "/"),
                color=colors.get(model_key, "#333333"),
                capsize=3
            )
        
        ax.set_xlabel("Target (sorted by natural MF cloud size)", fontweight="bold", fontsize=12)
        ax.set_ylabel(f"{metric_title} (Mean ± SEM)", fontweight="bold", fontsize=12)
        ax.set_title(f"Phase 4: Cross-Target {metric_title}", fontweight="bold", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(target_order, rotation=45, ha="right")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        
        plt.tight_layout()
        
        output_path_png = output_dir / f"phase4_cross_target_{metric_col}.png"
        output_path_pdf = output_dir / f"phase4_cross_target_{metric_col}.pdf"
        fig.savefig(output_path_png, dpi=300, bbox_inches="tight")
        fig.savefig(output_path_pdf, bbox_inches="tight")
        plt.close(fig)
        
        logger.info(f"Saved: {output_path_png.name}")


def plot_mf_size_correlation(
    df_agg: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger
) -> Dict[str, Tuple[float, float]]:
    """
    Plot scatter: Natural MF cloud size vs High-potency EF@1%.
    
    Compute Spearman correlation + p-value for each method.
    
    Returns:
        Dict mapping model_key → (rho, p_value)
    """
    logger.info("Generating MF size vs High-potency EF@1% correlation plot...")
    
    # Need stratified data for this
    # For now, use overall EF@1% as proxy (will be replaced by high-potency in stratified version)
    
    df_agg["model_key"] = df_agg["method"] + "_" + df_agg["representation"]
    model_keys = sorted(df_agg["model_key"].unique())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {
        "pca_features": "#1f77b4",
        "pca_fingerprints": "#aec7e8",
        "umap_features": "#ff7f0e",
        "umap_fingerprints": "#ffbb78",
    }
    
    correlations = {}
    
    for model_key in model_keys:
        subset = df_agg[df_agg["model_key"] == model_key].copy()
        
        x = subset["natural_mf_size"].values
        y = subset["ef_1%_mean"].values
        
        # Compute Spearman correlation
        rho, p_value = stats.spearmanr(x, y)
        correlations[model_key] = (rho, p_value)
        
        # Scatter plot
        ax.scatter(
            x, y,
            label=f"{model_key.replace('_', '/')} (ρ={rho:.3f}, p={p_value:.3f})",
            color=colors.get(model_key, "#333333"),
            s=100, alpha=0.7, edgecolors="black"
        )
        
        # Fit trend line (log-scale)
        if len(x) > 2:
            try:
                z = np.polyfit(np.log10(x), y, 1)
                p = np.poly1d(z)
                x_fit = np.logspace(np.log10(x.min()), np.log10(x.max()), 100)
                ax.plot(x_fit, p(np.log10(x_fit)), "--", color=colors.get(model_key, "#333333"), alpha=0.6, linewidth=1.5)
            except np.linalg.LinAlgError:
                logger.warning(f"Could not fit trendline for {model_key} (SVD did not converge)")
    
    ax.set_xscale("log")
    ax.set_xlabel("Natural MF Cloud Size (pre-cutoff, compounds)", fontweight="bold", fontsize=12)
    ax.set_ylabel("Overall EF@1% (Mean)", fontweight="bold", fontsize=12)
    ax.set_title("MF Cloud Size vs Enrichment: Cross-Target Correlation", fontweight="bold", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    
    plt.tight_layout()
    
    output_path_png = output_dir / "phase4_mf_size_correlation.png"
    output_path_pdf = output_dir / "phase4_mf_size_correlation.pdf"
    fig.savefig(output_path_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_path_pdf, bbox_inches="tight")
    plt.close(fig)
    
    logger.info(f"Saved: {output_path_png.name}")
    
    # Log correlation results
    logger.info("\n" + "="*80)
    logger.info("SPEARMAN CORRELATION: Natural MF Size vs Overall EF@1%")
    logger.info("="*80)
    for model_key, (rho, p_value) in correlations.items():
        sig_marker = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "n.s."
        logger.info(f"  {model_key:20s}: ρ = {rho:+.3f}, p = {p_value:.4f} {sig_marker}")
    logger.info("="*80 + "\n")
    
    return correlations


def plot_stratified_cross_target(
    df_stratified: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger
) -> None:
    """
    Plot potency-stratified cross-target comparison.
    
    4 panels (one per tier: Overall, High, Medium, Weak)
    X-axis: Targets sorted by natural MF size
    Y-axis: Tier-specific EF@1%
    """
    logger.info("Generating stratified cross-target comparison...")
    
    # Aggregate by target
    df_stratified["model_key"] = df_stratified["method"] + "_" + df_stratified["representation"]
    
    group_keys = ["target", "method", "representation", "model_key"]
    agg_dict = {
        "ef_1%_overall": ["mean", "sem"],
        "ef_1%_high": ["mean", "sem"],
        "ef_1%_medium": ["mean", "sem"],
        "ef_1%_weak": ["mean", "sem"],
        "natural_mf_size": "first",
    }
    
    df_agg = df_stratified.groupby(group_keys, dropna=False).agg(agg_dict).reset_index()
    df_agg.columns = ["_".join(col).strip("_") if isinstance(col, tuple) else col for col in df_agg.columns]
    
    # Sort targets by natural MF size (use first value since all same per target)
    target_order = df_agg.groupby("target")["natural_mf_size_first"].first().sort_values().index.tolist()
    
    model_keys = sorted(df_agg["model_key"].unique())
    
    tiers = [
        ("ef_1%_overall", "Overall"),
        ("ef_1%_high", "High-Potency (≤100 nM)"),
        ("ef_1%_medium", "Medium (100-1K nM)"),
        ("ef_1%_weak", "Weak (1K-100K nM)"),
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True)
    axes = axes.flatten()
    
    colors = {
        "pca_features": "#1f77b4",
        "pca_fingerprints": "#aec7e8",
        "umap_features": "#ff7f0e",
        "umap_fingerprints": "#ffbb78",
    }
    
    x = np.arange(len(target_order))
    width = 0.2
    
    for tier_idx, (tier_col, tier_label) in enumerate(tiers):
        ax = axes[tier_idx]
        
        for model_idx, model_key in enumerate(model_keys):
            subset = df_agg[df_agg["model_key"] == model_key].copy()
            subset = subset.set_index("target").reindex(target_order)
            
            mean_col = f"{tier_col}_mean"
            sem_col = f"{tier_col}_sem"
            
            if mean_col not in subset.columns:
                continue
            
            y_mean = subset[mean_col].values
            y_sem = subset[sem_col].values
            
            offset = (model_idx - len(model_keys)/2 + 0.5) * width
            
            ax.bar(
                x + offset, y_mean, width,
                yerr=y_sem,
                label=model_key.replace("_", "/"),
                color=colors.get(model_key, "#333333"),
                capsize=2
            )
        
        ax.set_ylabel("EF@1% (Mean ± SEM)", fontweight="bold")
        ax.set_title(tier_label, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        
        if tier_idx >= 2:  # Bottom row
            ax.set_xticks(x)
            ax.set_xticklabels(target_order, rotation=45, ha="right")
            ax.set_xlabel("Target", fontweight="bold")
        
        if tier_idx == 0:
            ax.legend(loc="upper left", fontsize=8)
    
    plt.tight_layout()
    
    output_path_png = output_dir / "phase4_stratified_cross_target.png"
    output_path_pdf = output_dir / "phase4_stratified_cross_target.pdf"
    fig.savefig(output_path_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_path_pdf, bbox_inches="tight")
    plt.close(fig)
    
    logger.info(f"Saved: {output_path_png.name}")


def plot_high_potency_correlation(
    df_stratified: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger
) -> Dict[str, Tuple[float, float]]:
    """
    Plot scatter: Natural MF cloud size vs High-potency EF@1% (stratified).
    
    THIS IS THE KEY MANUSCRIPT FIGURE.
    
    Returns:
        Dict mapping model_key → (rho, p_value)
    """
    logger.info("Generating Natural MF Size vs High-Potency EF@1% correlation (STRATIFIED)...")
    
    # Aggregate by target
    df_stratified["model_key"] = df_stratified["method"] + "_" + df_stratified["representation"]
    
    group_keys = ["target", "method", "representation", "model_key", "natural_mf_size"]
    agg_dict = {
        "ef_1%_high": "mean",
    }
    
    df_agg = df_stratified.groupby(group_keys, dropna=False).agg(agg_dict).reset_index()
    
    model_keys = sorted(df_agg["model_key"].unique())
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = {
        "pca_features": "#1f77b4",
        "pca_fingerprints": "#aec7e8",
        "umap_features": "#ff7f0e",
        "umap_fingerprints": "#ffbb78",
    }
    
    correlations = {}
    
    for model_key in model_keys:
        subset = df_agg[df_agg["model_key"] == model_key].copy()
        
        x = subset["natural_mf_size"].values
        y = subset["ef_1%_high"].values
        
        # Remove NaN
        valid_mask = ~np.isnan(y)
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        
        if len(x_valid) < 3:
            logger.warning(f"{model_key}: Not enough valid data points for correlation")
            continue
        
        # Compute Spearman correlation
        rho, p_value = stats.spearmanr(x_valid, y_valid)
        correlations[model_key] = (rho, p_value)
        
        # Scatter plot
        ax.scatter(
            x_valid, y_valid,
            label=f"{model_key.replace('_', '/')} (ρ={rho:.3f}, p={p_value:.3f})",
            color=colors.get(model_key, "#333333"),
            s=120, alpha=0.7, edgecolors="black", linewidth=1.5
        )
        
        # Fit trend line (log-scale)
        if len(x_valid) > 2:
                try:
                    z = np.polyfit(np.log10(x_valid), y_valid, 1)
                    p = np.poly1d(z)
                    x_fit = np.logspace(np.log10(x_valid.min()), np.log10(x_valid.max()), 100)
                    ax.plot(x_fit, p(np.log10(x_fit)), "--", color=colors.get(model_key, "#333333"), alpha=0.7, linewidth=2)
                except np.linalg.LinAlgError:
                    logger.warning(f"Could not fit trendline for {model_key} (SVD did not converge)")
    
    ax.set_xscale("log")
    ax.set_xlabel("Natural MF Cloud Size (pre-cutoff, compounds)", fontweight="bold", fontsize=13)
    ax.set_ylabel("High-Potency EF@1% (≤100 nM, Mean)", fontweight="bold", fontsize=13)
    ax.set_title("Phase 4: MF Cloud Size vs High-Potency Enrichment", fontweight="bold", fontsize=15)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    
    output_path_png = output_dir / "phase4_high_potency_correlation.png"
    output_path_pdf = output_dir / "phase4_high_potency_correlation.pdf"
    fig.savefig(output_path_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_path_pdf, bbox_inches="tight")
    plt.close(fig)
    
    logger.info(f"Saved: {output_path_png.name}")
    
    # Log correlation results
    logger.info("\n" + "="*80)
    logger.info("SPEARMAN CORRELATION: Natural MF Size vs HIGH-POTENCY EF@1%")
    logger.info("="*80)
    for model_key, (rho, p_value) in correlations.items():
        sig_marker = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "n.s."
        logger.info(f"  {model_key:20s}: ρ = {rho:+.3f}, p = {p_value:.4f} {sig_marker}")
    logger.info("="*80 + "\n")
    
    return correlations


# ============================================================================
# Summary Outputs
# ============================================================================

def save_aggregated_summary(df_agg: pd.DataFrame, output_dir: Path, logger: logging.Logger) -> None:
    """Save aggregated summary CSV."""
    logger.info("Saving aggregated summary CSV...")
    
    output_path = output_dir / "phase4_summary_aggregated.csv"
    df_agg.to_csv(output_path, index=False)
    logger.info(f"Saved: {output_path.name} ({len(df_agg)} rows)")


def save_correlation_results(
    correlations: Dict[str, Tuple[float, float]],
    output_dir: Path,
    logger: logging.Logger
) -> None:
    """Save correlation statistics to JSON."""
    logger.info("Saving correlation statistics...")
    
    results = {}
    for model_key, (rho, p_value) in correlations.items():
        results[model_key] = {
            "spearman_rho": float(rho),
            "p_value": float(p_value),
            "significant_alpha_0.05": bool(p_value < 0.05),
            "significant_alpha_0.01": bool(p_value < 0.01),
        }
    
    output_path = output_dir / "phase4_correlation_stats.json"
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved: {output_path.name}")


def generate_analysis_report(
    df: pd.DataFrame,
    df_agg: pd.DataFrame,
    correlations: Dict[str, Tuple[float, float]],
    output_dir: Path,
    logger: logging.Logger
) -> None:
    """Generate markdown analysis report."""
    logger.info("Generating analysis report...")
    
    report_lines = [
        "# Phase 4 Cross-Target Generalization Study - Analysis Report",
        "",
        f"**Total Runs**: {len(df)}",
        f"**Targets**: {len(df['target'].unique())} ({', '.join(sorted(df['target'].unique()))})",
        f"**Methods**: {', '.join(sorted(df['method'].unique()))}",
        f"**Representations**: {', '.join(sorted(df['representation'].unique()))}",
        "",
        "## Cross-Target Performance Summary",
        "",
        "### Overall EF@1% by Target",
        "",
        "| Target | Natural MF Size | EF@1% (Mean ± SEM) | n_actives |",
        "|---|---|---|---|",
    ]
    
    # Summary table
    df_agg["model_key"] = df_agg["method"] + "_" + df_agg["representation"]
    
    # Get best-performing model for display (UMAP features)
    best_model = "umap_features"
    subset = df_agg[df_agg["model_key"] == best_model].sort_values("natural_mf_size")
    
    for _, row in subset.iterrows():
        report_lines.append(
            f"| {row['target']} | {int(row['natural_mf_size']):,} | "
            f"{row['ef_1%_mean']:.2f} ± {row['ef_1%_sem']:.2f} | "
            f"{int(row['n_actives'])} |"
        )
    
    report_lines.extend([
        "",
        "## Correlation Analysis: Natural MF Size vs Enrichment",
        "",
        "Spearman rank correlation between natural MF cloud size and overall EF@1%:",
        "",
        "| Method/Representation | Spearman ρ | p-value | Significant? |",
        "|---|---|---|---|",
    ])
    
    for model_key, (rho, p_value) in sorted(correlations.items()):
        sig = "Yes (p<0.05)" if p_value < 0.05 else "No (n.s.)"
        report_lines.append(f"| {model_key.replace('_', '/')} | {rho:+.3f} | {p_value:.4f} | {sig} |")
    
    report_lines.extend([
        "",
        "## Key Findings",
        "",
        "1. **Cross-Target Consistency**: [Describe consistency/variability across targets]",
        "2. **MF Cloud Size Effect**: [Interpret correlation results]",
        "3. **Method Robustness**: [Compare PCA vs UMAP across targets]",
        "4. **High-Potency Retrieval**: [Stratified tier analysis]",
        "",
        "## Recommendations",
        "",
        "- **Optimal Configuration**: [Best method-representation combo]",
        "- **MF Cloud Size Guidelines**: [Practical guidance based on correlation]",
        "- **Target-Specific Considerations**: [When to adjust strategy]",
        "",
    ])
    
    # Write report
    output_path = output_dir / "phase4_analysis_report.md"
    with output_path.open("w") as f:
        f.write("\n".join(report_lines))
    
    logger.info(f"Saved: {output_path.name}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 4 Post-Analysis: Cross-Target Generalization")
    parser.add_argument("--workspace_dir", type=str, default="experiment_workspace_v4",
                       help="Workspace directory")
    parser.add_argument("--output_dir", type=str, default="reporting/phase4_post_analysis",
                       help="Output directory for analysis")
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
    logger.info("PHASE 4 POST-ANALYSIS: CROSS-TARGET GENERALIZATION STUDY")
    logger.info("="*80)
    logger.info(f"Workspace: {workspace_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Stratify by potency: {args.stratify}")
    logger.info("="*80)
    
    try:
        # 1. Collect results
        df = collect_phase4_results(workspace_dir, logger)
        
        # 2. Aggregate by target
        df_agg = aggregate_by_target(df, logger)
        
        # 3. Affinity cutoff sensitivity analysis
        logger.info("\n" + "="*80)
        logger.info("AFFINITY CUTOFF SENSITIVITY ANALYSIS")
        logger.info("="*80)
        
        cutoffs = [100.0, 1000.0, 10000.0, 100000.0]
        df_cutoff_agg = aggregate_cutoff_sensitivity(workspace_dir, df, cutoffs, output_dir, logger)
        
        if not df_cutoff_agg.empty:
            plot_cutoff_sensitivity_per_target(df_cutoff_agg, output_dir, logger)
            best_cutoffs = identify_best_cutoffs_per_target(df_cutoff_agg, output_dir, logger)
        
        # 4. Generate plots
        plot_cross_target_comparison(df_agg, output_dir, logger)
        plot_cross_target_bedroc_ief(df_agg, output_dir, logger)
        correlations = plot_mf_size_correlation(df_agg, output_dir, logger)
        
        # 5. Potency-stratified analysis (if enabled)
        if args.stratify:
            logger.info("\n" + "="*80)
            logger.info("POTENCY TIER STRATIFICATION")
            logger.info("="*80)
            
            # Compute tier-specific metrics for all runs
            logger.info("Computing tier-specific EF@1% for all runs...")
            stratified_records = []
            
            for _, row in df.iterrows():
                target = row["target"]
                run_name = row["run_name"]
                
                tier_metrics = compute_stratified_metrics_for_run(workspace_dir, target, run_name, logger)
                
                if tier_metrics:
                    record = {
                        "run_name": run_name,
                        "target": target,
                        "method": row["method"],
                        "representation": row["representation"],
                        "replicate": row["replicate"],
                        "natural_mf_size": row["natural_mf_size"],
                        **tier_metrics
                    }
                    stratified_records.append(record)
            
            if stratified_records:
                df_stratified = pd.DataFrame(stratified_records)
                logger.info(f"Computed stratified metrics for {len(df_stratified)} runs")
                
                # Save stratified CSV
                strat_csv = output_dir / "phase4_summary_stratified.csv"
                df_stratified.to_csv(strat_csv, index=False)
                logger.info(f"Saved: {strat_csv.name}")
                
                # Generate stratified plots
                plot_stratified_cross_target(df_stratified, output_dir, logger)
                high_potency_correlations = plot_high_potency_correlation(df_stratified, output_dir, logger)
                
                # Save high-potency correlation results
                save_correlation_results(high_potency_correlations, output_dir, logger)
            else:
                logger.warning("No stratified metrics computed (missing artifacts?)")
        
        # 5. Summary outputs
        save_aggregated_summary(df_agg, output_dir, logger)
        save_correlation_results(correlations, output_dir, logger)
        
        # 6. Generate report
        generate_analysis_report(df, df_agg, correlations, output_dir, logger)
        
        logger.info("="*80)
        logger.info("PHASE 4 POST-ANALYSIS COMPLETED")
        logger.info(f"Results saved to: {output_dir}")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
