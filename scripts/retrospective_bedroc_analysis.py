#!/usr/bin/env python3
"""
Retrospective BEDROC Analysis for Completed molfuse Experiments

Calculates BEDROC metrics from existing ranked_scores.csv files across all phases.
Does NOT require re-running experiments—only reads archived artifacts.

BEDROC (Boltzmann-Enhanced Discrimination of ROC):
- Emphasizes early recognition with exponential weighting
- α=20: top ~8% emphasis (recommended default)
- α=160.9: top ~1% emphasis (aggressive screening)

Reference:
    Truchon & Bayly (2007). J. Chem. Inf. Model. 47(2):488-508

Usage:
    # Scan entire workspace (all phases)
    python scripts/retrospective_bedroc_analysis.py \
        --workspace_dir experiment_workspace_v4 \
        --output_dir reporting/bedroc_retrospective \
        --alpha 20.0 160.9

    # Phase-specific analysis
    python scripts/retrospective_bedroc_analysis.py \
        --workspace_dir experiment_workspace_v4 \
        --phases phase1 phase2 \
        --alpha 20.0
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from molfuse.metrics.metrics import bedroc, ief


# ============================================================================
# Logger Setup
# ============================================================================

def setup_logger(output_dir: Path) -> logging.Logger:
    """Setup logger with file and console handlers."""
    logger = logging.getLogger("bedroc_retrospective")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    log_path = output_dir / "bedroc_analysis.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # File handler
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(ch)
    
    return logger


# ============================================================================
# Data Loading
# ============================================================================

def load_ranked_scores(scores_path: Path) -> Optional[pd.DataFrame]:
    """Load ranked_scores.csv with robust error handling."""
    if not scores_path.exists():
        return None
    
    try:
        df = pd.read_csv(scores_path)
        required_cols = {"score", "label"}
        if not required_cols.issubset(df.columns):
            return None
        return df
    except Exception:
        return None


def load_run_metadata(run_dir: Path) -> Dict:
    """Extract run metadata from logs/phase*_summary.json."""
    metadata = {
        "run_name": run_dir.name,
        "phase": None,
        "method": None,
        "representation": None,
        "dim": None,
        "ef1": None,
        "roc_auc": None,
    }
    
    # Try to load summary JSON
    for summary_file in ["phase1_summary.json", "phase2_summary.json", 
                          "phase3_summary.json", "phase4_summary.json", "summary.json"]:
        summary_path = run_dir / "logs" / summary_file
        if summary_path.exists():
            try:
                with summary_path.open() as f:
                    data = json.load(f)
                
                # Extract phase
                if "phase1" in summary_file:
                    metadata["phase"] = "phase1"
                elif "phase2" in summary_file:
                    metadata["phase"] = "phase2"
                elif "phase3" in summary_file:
                    metadata["phase"] = "phase3"
                elif "phase4" in summary_file:
                    metadata["phase"] = "phase4"
                
                # Extract config
                cfg = data.get("config", {})
                metadata["method"] = cfg.get("method")
                metadata["representation"] = cfg.get("representation")
                metadata["dim"] = cfg.get("dim")
                
                break
            except Exception:
                continue
    
    # Try to load metrics.json for comparison
    metrics_path = run_dir / "metrics" / "metrics.json"
    if metrics_path.exists():
        try:
            with metrics_path.open() as f:
                metrics = json.load(f)
            metadata["ef1"] = metrics.get("ef_at_1_percent")
            metadata["roc_auc"] = metrics.get("roc_auc")
        except Exception:
            pass
    
    return metadata


# ============================================================================
# BEDROC Computation
# ============================================================================

def compute_bedroc_for_run(
    run_dir: Path,
    alpha_values: List[float],
    logger: logging.Logger
) -> Optional[Dict]:
    """
    Compute BEDROC for a single run across multiple α values.
    
    Returns:
        Dict with run metadata + BEDROC values, or None if ranked_scores.csv missing
    """
    scores_path = run_dir / "artifacts" / "ranked_scores.csv"
    
    df_scores = load_ranked_scores(scores_path)
    if df_scores is None:
        logger.debug(f"Skipping {run_dir.name}: no valid ranked_scores.csv")
        return None
    
    # Extract labels and scores
    labels = df_scores["label"].values
    scores_vals = df_scores["score"].values
    
    # Validate
    n_actives = int(np.sum(labels))
    n_total = len(labels)
    
    if n_actives == 0 or n_actives == n_total:
        logger.warning(f"Skipping {run_dir.name}: degenerate labels ({n_actives}/{n_total} actives)")
        return None
    
    # Compute BEDROC for each α
    result = load_run_metadata(run_dir)
    result["n_actives"] = n_actives
    result["n_decoys"] = n_total - n_actives
    result["n_total"] = n_total
    
    for alpha in alpha_values:
        bedroc_val = bedroc(labels, scores_vals, alpha=alpha)
        result[f"bedroc_alpha{alpha:.1f}"] = bedroc_val
    
    logger.info(f"✓ {run_dir.name}: BEDROC(α=20)={result.get('bedroc_alpha20.0', np.nan):.3f}")
    
    return result


# ============================================================================
# Workspace Scanning
# ============================================================================

def scan_workspace(
    workspace_dir: Path,
    phases: Optional[List[str]],
    alpha_values: List[float],
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Scan workspace for all runs and compute BEDROC retrospectively.
    
    Args:
        workspace_dir: Root workspace directory
        phases: List of phase names to scan (None = all phases)
        alpha_values: List of α values to compute
        logger: Logger instance
    
    Returns:
        DataFrame with one row per run
    """
    if phases is None:
        phases = ["phase1", "phase2", "phase3", "phase4"]
    
    results = []
    
    for phase in phases:
        phase_dir = workspace_dir / phase
        if not phase_dir.exists():
            logger.warning(f"Phase directory not found: {phase_dir}")
            continue
        
        logger.info(f"\nScanning {phase_dir.name}/...")
        
        # Scan all subdirectories
        for run_dir in sorted(phase_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            
            result = compute_bedroc_for_run(run_dir, alpha_values, logger)
            if result is not None:
                results.append(result)
    
    if not results:
        logger.error("No valid runs found with ranked_scores.csv")
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    logger.info(f"\n{'='*80}")
    logger.info(f"Processed {len(df)} runs across {len(phases)} phases")
    logger.info(f"{'='*80}")
    
    return df


# ============================================================================
# Summary Statistics
# ============================================================================

def compute_summary_statistics(
    df: pd.DataFrame,
    alpha_values: List[float],
    output_dir: Path,
    logger: logging.Logger
) -> None:
    """Generate summary statistics grouped by method/representation/dimension."""
    
    if df.empty:
        return
    
    # Group by configuration
    group_cols = []
    for col in ["phase", "method", "representation", "dim"]:
        if col in df.columns and df[col].notna().any():
            group_cols.append(col)
    
    if not group_cols:
        logger.warning("No grouping columns found; skipping aggregation")
        return
    
    bedroc_cols = [f"bedroc_alpha{alpha:.1f}" for alpha in alpha_values]
    
    # Aggregate
    agg_dict = {col: ["mean", "std", "min", "max", "count"] for col in bedroc_cols}
    df_agg = df.groupby(group_cols, dropna=False).agg(agg_dict).reset_index()
    
    # Flatten MultiIndex columns
    df_agg.columns = ["_".join(str(c) for c in col).strip("_") if isinstance(col, tuple) else col 
                      for col in df_agg.columns]
    
    # Save
    agg_path = output_dir / "bedroc_aggregated.csv"
    df_agg.to_csv(agg_path, index=False)
    logger.info(f"Saved aggregated summary: {agg_path}")
    
    # Print top performers
    for alpha in alpha_values:
        mean_col = f"bedroc_alpha{alpha:.1f}_mean"
        if mean_col in df_agg.columns:
            logger.info(f"\nTop 5 configs by BEDROC(α={alpha:.1f}):")
            top5 = df_agg.nlargest(5, mean_col)
            for _, row in top5.iterrows():
                config_str = " | ".join(f"{col}={row[col]}" for col in group_cols if col in row)
                logger.info(f"  {row[mean_col]:.3f} ± {row.get(f'bedroc_alpha{alpha:.1f}_std', 0):.3f} : {config_str}")


# ============================================================================
# Comparison with Existing Metrics
# ============================================================================

def compare_with_existing_metrics(
    df: pd.DataFrame,
    alpha_values: List[float],
    output_dir: Path,
    logger: logging.Logger
) -> None:
    """Correlate BEDROC with existing metrics (EF@1%, ROC-AUC)."""
    
    if df.empty:
        return
    
    logger.info(f"\n{'='*80}")
    logger.info("Correlation: BEDROC vs Existing Metrics")
    logger.info(f"{'='*80}")
    
    from scipy.stats import spearmanr, pearsonr
    
    for alpha in alpha_values:
        bedroc_col = f"bedroc_alpha{alpha:.1f}"
        if bedroc_col not in df.columns:
            continue
        
        logger.info(f"\nBEDROC(α={alpha:.1f}) correlations:")
        
        # vs EF@1%
        if "ef1" in df.columns:
            valid = df[[bedroc_col, "ef1"]].dropna()
            if len(valid) > 3:
                r_pearson, p_pearson = pearsonr(valid[bedroc_col], valid["ef1"])
                r_spearman, p_spearman = spearmanr(valid[bedroc_col], valid["ef1"])
                logger.info(f"  vs EF@1%:   Pearson r={r_pearson:.3f} (p={p_pearson:.2e}), Spearman ρ={r_spearman:.3f} (p={p_spearman:.2e})")
        
        # vs ROC-AUC
        if "roc_auc" in df.columns:
            valid = df[[bedroc_col, "roc_auc"]].dropna()
            if len(valid) > 3:
                r_pearson, p_pearson = pearsonr(valid[bedroc_col], valid["roc_auc"])
                r_spearman, p_spearman = spearmanr(valid[bedroc_col], valid["roc_auc"])
                logger.info(f"  vs ROC-AUC: Pearson r={r_pearson:.3f} (p={p_pearson:.2e}), Spearman ρ={r_spearman:.3f} (p={p_spearman:.2e})")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Retrospective BEDROC analysis for completed molfuse experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--workspace_dir",
        type=Path,
        required=True,
        help="Workspace directory containing phase1/, phase2/, etc."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory for results and plots"
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=["phase1", "phase2", "phase3", "phase4"],
        default=None,
        help="Phases to analyze (default: all phases)"
    )
    parser.add_argument(
        "--alpha",
        nargs="+",
        type=float,
        default=[20.0, 160.9],
        help="Alpha values for BEDROC (default: 20.0 160.9)"
    )
    
    args = parser.parse_args()
    
    # Setup
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(args.output_dir)
    
    logger.info("="*80)
    logger.info("RETROSPECTIVE BEDROC ANALYSIS")
    logger.info("="*80)
    logger.info(f"Workspace: {args.workspace_dir}")
    logger.info(f"Phases: {args.phases or 'all'}")
    logger.info(f"Alpha values: {args.alpha}")
    logger.info(f"Output: {args.output_dir}")
    
    # Scan workspace
    df = scan_workspace(
        workspace_dir=args.workspace_dir,
        phases=args.phases,
        alpha_values=args.alpha,
        logger=logger
    )
    
    if df.empty:
        logger.error("No valid runs found. Exiting.")
        return
    
    # Save full results
    results_path = args.output_dir / "bedroc_all_runs.csv"
    df.to_csv(results_path, index=False)
    logger.info(f"\nSaved full results: {results_path}")
    
    # Summary statistics
    compute_summary_statistics(df, args.alpha, args.output_dir, logger)
    
    # Correlation analysis
    compare_with_existing_metrics(df, args.alpha, args.output_dir, logger)
    
    logger.info(f"\n{'='*80}")
    logger.info("ANALYSIS COMPLETE")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()
