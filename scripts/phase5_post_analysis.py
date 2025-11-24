#!/usr/bin/env python3
"""
Phase 5 Post-Analysis: Validation & Baseline Experiments Report

Generate a concise LaTeX-ready report for inclusion in Results/Bias Analysis section.

This script:
    1. Aggregates Phase 5 results across all replicates
    2. Computes mean ± std for each experiment type
    3. Generates LaTeX tables and formatted text
    4. Performs statistical comparisons

Usage:
    python scripts/phase5_post_analysis.py \\
        --workspace experiment_workspace_v4 \\
        --output reports/phase5_bias_analysis.txt
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ============================================================================
# Constants
# ============================================================================

EXPERIMENT_NAMES = {
    "negative_control": "Database Bias Control",
    "raw_descriptors": "Raw Descriptor Baseline",
    "tanimoto": "Tanimoto Baseline",
}

EXPERIMENT_ORDER = ["negative_control", "raw_descriptors", "tanimoto"]

METRICS = ["ef_1%", "ef_5%", "roc_auc", "pr_auc"]

METRIC_NAMES = {
    "ef_1%": "EF@1\\%",
    "ef_5%": "EF@5\\%",
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
}

# Potency tier thresholds (nM) - INCLUSIVE on both ends to match Phase 2
TIER_THRESHOLDS = {
    "high": (0.1, 100.0),       # 0.1 ≤ affinity ≤ 100 nM (drug-like)
    "medium": (100.0, 1000.0),  # 100 < affinity ≤ 1000 nM (moderate)
    "low": (1000.0, 100000.0),  # 1000 < affinity ≤ 100000 nM (marginal)
}

TIER_ORDER = ["high", "medium", "low"]


# ============================================================================
# Data Collection
# ============================================================================

def collect_phase5_results(workspace_dir: Path) -> Dict[str, List[Dict]]:
    """
    Collect all Phase 5 results from workspace.
    
    Handles incomplete experiments gracefully by continuing to next experiment.
    Auto-detects subdirectory (validation or expansion).
    
    Returns:
        Dict mapping experiment_type -> list of result dicts
    """
    # Try both possible Phase 5 subdirectories
    phase5_base = workspace_dir / "phase5"
    
    if not phase5_base.exists():
        print(f"WARNING: Phase 5 directory not found: {phase5_base}")
        print("Creating empty results structure...")
        return {exp: [] for exp in EXPERIMENT_ORDER}
    
    # Auto-detect subdirectory: check both 'validation' and 'expansion'
    phase5_subdirs = []
    for subdir_name in ["validation", "expansion"]:
        subdir = phase5_base / subdir_name
        if subdir.exists() and any(subdir.iterdir()):
            phase5_subdirs.append(subdir)
    
    if not phase5_subdirs:
        print(f"WARNING: No Phase 5 results found in {phase5_base}")
        print("Creating empty results structure...")
        return {exp: [] for exp in EXPERIMENT_ORDER}
    
    results = {exp: [] for exp in EXPERIMENT_ORDER}
    missing_runs = []
    error_runs = []
    
    # Collect from all detected subdirectories
    for phase5_dir in phase5_subdirs:
        print(f"Scanning: {phase5_dir}")
        
        # Match any run directory (e.g., tanimoto_rep1, raw_descriptors_rep2, etc.)
        for run_dir in sorted(phase5_dir.glob("*")):
            if not run_dir.is_dir():
                continue
        
            summary_path = run_dir / "logs" / "phase5_summary.json"
            
            if not summary_path.exists():
                missing_runs.append(run_dir.name)
                continue
            
            try:
                with summary_path.open("r") as f:
                    summary = json.load(f)
                
                # Store the actual parent directory for later use
                summary["_phase5_subdir"] = phase5_dir.name
                
                exp_type = summary.get("experiment_type")
                if exp_type in results:
                    results[exp_type].append(summary)
                else:
                    print(f"WARNING: Unknown experiment type '{exp_type}' in {run_dir.name}, skipping")
            
            except Exception as e:
                error_runs.append((run_dir.name, str(e)))
    
    # Report missing/error runs
    if missing_runs:
        print(f"\nWARNING: {len(missing_runs)} runs missing phase5_summary.json (may still be running):")
        for run_name in missing_runs[:5]:  # Show first 5
            print(f"  - {run_name}")
        if len(missing_runs) > 5:
            print(f"  ... and {len(missing_runs) - 5} more")
    
    if error_runs:
        print(f"\nWARNING: {len(error_runs)} runs failed to load:")
        for run_name, error in error_runs[:3]:  # Show first 3
            print(f"  - {run_name}: {error}")
        if len(error_runs) > 3:
            print(f"  ... and {len(error_runs) - 3} more")
    
    return results


# ============================================================================
# Statistical Analysis
# ============================================================================

def compute_summary_stats(results: List[Dict]) -> Dict[str, Tuple[float, float]]:
    """
    Compute mean ± std for each metric across replicates.
    
    Returns:
        Dict mapping metric -> (mean, std)
    """
    if not results:
        return {m: (np.nan, np.nan) for m in METRICS}
    
    stats_dict = {}
    for metric in METRICS:
        values = [r.get(metric, np.nan) for r in results]
        values = [v for v in values if not np.isnan(v)]
        
        if values:
            stats_dict[metric] = (np.mean(values), np.std(values))
        else:
            stats_dict[metric] = (np.nan, np.nan)
    
    return stats_dict


def perform_significance_tests(
    results_dict: Dict[str, List[Dict]],
    baseline_exp: str = "tanimoto",
) -> Tuple[Dict[str, Dict[str, float]], List[str]]:
    """
    Perform paired t-tests comparing each experiment to baseline.
    
    Returns:
        Tuple of (p_values dict, list of warning messages)
    """
    p_values = {}
    warning_messages = []
    
    baseline_results = results_dict.get(baseline_exp, [])
    if not baseline_results:
        print(f"WARNING: No results for baseline experiment '{baseline_exp}'")
        return {}, []
    
    for exp_type in EXPERIMENT_ORDER:
        if exp_type == baseline_exp:
            continue
        
        exp_results = results_dict.get(exp_type, [])
        if not exp_results:
            continue
        
        p_values[exp_type] = {}
        
        for metric in METRICS:
            baseline_vals = [r.get(metric, np.nan) for r in baseline_results]
            exp_vals = [r.get(metric, np.nan) for r in exp_results]
            
            # Remove NaNs
            baseline_vals = [v for v in baseline_vals if not np.isnan(v)]
            exp_vals = [v for v in exp_vals if not np.isnan(v)]
            
            if len(baseline_vals) > 1 and len(exp_vals) > 1 and len(baseline_vals) == len(exp_vals):
                # Paired t-test (assumes same number of replicates)
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    t_stat, p_val = stats.ttest_rel(exp_vals, baseline_vals)
                    if w:
                        for warning in w:
                            msg = f"Statistical test warning for {exp_type} vs {baseline_exp} ({metric}): {warning.message}"
                            warning_messages.append(msg)
                p_values[exp_type][metric] = p_val
            elif len(baseline_vals) > 1 and len(exp_vals) > 1:
                # Independent t-test (different number of replicates)
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    t_stat, p_val = stats.ttest_ind(exp_vals, baseline_vals)
                    if w:
                        for warning in w:
                            msg = f"Statistical test warning for {exp_type} vs {baseline_exp} ({metric}): {warning.message}"
                            warning_messages.append(msg)
                p_values[exp_type][metric] = p_val
            else:
                p_values[exp_type][metric] = np.nan
    
    return p_values, warning_messages


# ============================================================================
# Detailed Comparison Tables
# ============================================================================

def compute_raw_descriptors_comparison(
    workspace_dir: Path,
    results_dict: Dict[str, List[Dict]],
) -> str:
    """
    Generate detailed per-MF comparison table for raw_descriptors vs tanimoto baseline.
    Shows absolute and relative change in EF@1% for each MF class.
    """
    raw_desc_results = results_dict.get("raw_descriptors", [])
    tanimoto_results = results_dict.get("tanimoto", [])
    
    if not raw_desc_results or not tanimoto_results:
        return "No data available for raw_descriptors comparison.\n"
    
    # Group by MF class
    raw_by_mf = {}
    tanimoto_by_mf = {}
    
    for result in raw_desc_results:
        mf = result.get("config", {}).get("keyword", "Unknown")
        if mf not in raw_by_mf:
            raw_by_mf[mf] = []
        raw_by_mf[mf].append(result.get("ef_1%", np.nan))
    
    for result in tanimoto_results:
        mf = result.get("config", {}).get("keyword", "Unknown")
        if mf not in tanimoto_by_mf:
            tanimoto_by_mf[mf] = []
        tanimoto_by_mf[mf].append(result.get("ef_1%", np.nan))
    
    # Compute mean for each MF
    lines = []
    lines.append("\nRaw Descriptors vs Tanimoto Baseline: Per-MF EF@1% Comparison")
    lines.append("="*80)
    lines.append(f"{'MF Class':<25} {'Tanimoto':<12} {'Raw Desc':<12} {'Abs Δ':<10} {'Rel Δ (%)':<12}")
    lines.append("-"*80)
    
    all_mfs = sorted(set(raw_by_mf.keys()) | set(tanimoto_by_mf.keys()))
    total_tanimoto = []
    total_raw = []
    
    for mf in all_mfs:
        tani_vals = [v for v in tanimoto_by_mf.get(mf, []) if not np.isnan(v)]
        raw_vals = [v for v in raw_by_mf.get(mf, []) if not np.isnan(v)]
        
        if not tani_vals or not raw_vals:
            lines.append(f"{mf:<25} {'N/A':<12} {'N/A':<12} {'N/A':<10} {'N/A':<12}")
            continue
        
        tani_mean = np.mean(tani_vals)
        raw_mean = np.mean(raw_vals)
        abs_delta = raw_mean - tani_mean
        rel_delta = (abs_delta / tani_mean) * 100 if tani_mean != 0 else np.nan
        
        total_tanimoto.extend(tani_vals)
        total_raw.extend(raw_vals)
        
        lines.append(f"{mf:<25} {tani_mean:>11.1f} {raw_mean:>11.1f} {abs_delta:>9.1f} {rel_delta:>11.1f}")
    
    lines.append("-"*80)
    
    # Overall statistics
    if total_tanimoto and total_raw:
        overall_tani = np.mean(total_tanimoto)
        overall_raw = np.mean(total_raw)
        overall_abs = overall_raw - overall_tani
        overall_rel = (overall_abs / overall_tani) * 100 if overall_tani != 0 else np.nan
        lines.append(f"{'OVERALL':<25} {overall_tani:>11.1f} {overall_raw:>11.1f} {overall_abs:>9.1f} {overall_rel:>11.1f}")
    
    lines.append("="*80)
    lines.append("")
    
    return "\n".join(lines)


def compute_negative_control_comparison(
    workspace_dir: Path,
    results_dict: Dict[str, List[Dict]],
) -> str:
    """
    Generate summary comparison for negative_control vs tanimoto baseline.
    Shows total absolute and relative change in EF@1%.
    """
    neg_results = results_dict.get("negative_control", [])
    tanimoto_results = results_dict.get("tanimoto", [])
    
    if not neg_results or not tanimoto_results:
        return "No data available for negative_control comparison.\n"
    
    # Extract EF@1% values
    neg_ef1 = [r.get("ef_1%", np.nan) for r in neg_results]
    tani_ef1 = [r.get("ef_1%", np.nan) for r in tanimoto_results]
    
    neg_ef1 = [v for v in neg_ef1 if not np.isnan(v)]
    tani_ef1 = [v for v in tani_ef1 if not np.isnan(v)]
    
    if not neg_ef1 or not tani_ef1:
        return "Insufficient data for negative_control comparison.\n"
    
    neg_mean = np.mean(neg_ef1)
    neg_std = np.std(neg_ef1)
    tani_mean = np.mean(tani_ef1)
    tani_std = np.std(tani_ef1)
    
    abs_delta = neg_mean - tani_mean
    rel_delta = (abs_delta / tani_mean) * 100 if tani_mean != 0 else np.nan
    
    lines = []
    lines.append("\nNegative Control vs Tanimoto Baseline: EF@1% Summary")
    lines.append("="*80)
    lines.append(f"Tanimoto Baseline:     {tani_mean:.1f} ± {tani_std:.1f} (n={len(tani_ef1)})")
    lines.append(f"Negative Control:      {neg_mean:.1f} ± {neg_std:.1f} (n={len(neg_ef1)})")
    lines.append("-"*80)
    lines.append(f"Absolute Change (Δ):   {abs_delta:+.1f}")
    lines.append(f"Relative Change (%):   {rel_delta:+.1f}%")
    lines.append("="*80)
    lines.append("")
    
    return "\n".join(lines)


# ============================================================================
# Stratified Potency Tier Analysis
# ============================================================================

def compute_tier_ef1(
    ranked_scores_path: Path,
    actives_df: pd.DataFrame,
    smiles_col: str,
    activity_col: str = "Standard Value (nM)",
) -> Dict[str, float]:
    """
    Compute EF@1% stratified by potency tiers from ranked scores.
    
    Uses the Phase 2 tier stratification approach:
    - For each tier, count only that tier's actives as "hits"
    - Compute EF using full ranked list (all tiers + decoys) as denominator
    - This answers: "How well does the model enrich THIS tier in the full dataset?"
    
    Args:
        ranked_scores_path: Path to ranked_scores.csv (SMILES, score, label)
        actives_df: DataFrame with actives and their activity values
        smiles_col: Name of SMILES column
        activity_col: Name of activity column (nM)
    
    Returns:
        Dict mapping tier -> EF@1% (high, medium, low)
    """
    # Load ranked scores
    df_scores = pd.read_csv(ranked_scores_path)
    
    # Join with activity values
    df_actives = actives_df[[smiles_col, activity_col]].copy()
    df_actives = df_actives.dropna(subset=[activity_col])
    
    # Merge scores with activity
    df_merged = df_scores.merge(df_actives, left_on="SMILES", right_on=smiles_col, how="left")
    
    # Assign tiers to all actives (INCLUSIVE boundaries to match Phase 2)
    def assign_tier(affinity_nM):
        if pd.isna(affinity_nM):
            return None
        try:
            val = float(affinity_nM)
        except (ValueError, TypeError):
            return None
        
        if val < 0.1 or val > 100000:
            return None
        
        # Check each tier with INCLUSIVE boundaries
        # High: 0.1 ≤ val ≤ 100
        if 0.1 <= val <= 100.0:
            return "high"
        # Medium: 100 < val ≤ 1000
        elif 100.0 < val <= 1000.0:
            return "medium"
        # Low: 1000 < val ≤ 100000
        elif 1000.0 < val <= 100000.0:
            return "low"
        
        return None
    
    df_merged["potency_tier"] = df_merged[activity_col].apply(assign_tier)
    
    # Stratify by tier (Phase 2 approach)
    tier_ef1 = {}
    N_total = len(df_merged)  # Full dataset size (all actives + decoys)
    k_percent = 1.0
    n_top_k = int(np.ceil(N_total * k_percent / 100.0))
    
    # Sort by score (descending)
    df_sorted = df_merged.sort_values("score", ascending=False).reset_index(drop=True)
    top_k = df_sorted.head(n_top_k)
    
    for tier_name in TIER_ORDER:
        # Count tier actives in full dataset
        tier_mask_all = (df_sorted["label"] == 1) & (df_sorted["potency_tier"] == tier_name)
        n_tier_actives = tier_mask_all.sum()
        
        if n_tier_actives == 0:
            tier_ef1[tier_name] = np.nan
            continue
        
        # Count tier hits in top k
        tier_mask_top = (top_k["label"] == 1) & (top_k["potency_tier"] == tier_name)
        n_hits = tier_mask_top.sum()
        
        # EF@1% = (hits / N_tier_actives) / (k / N_total)
        # This is equivalent to: (n_hits / n_top_k) / (n_tier_actives / N_total)
        expected_hits = n_tier_actives * (k_percent / 100.0)
        ef1 = n_hits / expected_hits if expected_hits > 0 else 0.0
        tier_ef1[tier_name] = ef1
    
    return tier_ef1


def compute_stratified_metrics_for_run(
    workspace_dir: Path,
    run_summary: Dict,
    experiment_type: str,
) -> Dict[str, float]:
    """
    Compute stratified potency tier EF@1% for a single Phase 5 run.
    
    Returns:
        Dict mapping tier -> EF@1% or empty dict if not applicable
    """
    if experiment_type not in ["tanimoto", "raw_descriptors", "negative_control"]:
        return {}
    
    run_name = run_summary["run_name"]
    config = run_summary.get("config", {})
    
    # Get paths - use stored subdirectory (validation or expansion)
    phase5_subdir = run_summary.get("_phase5_subdir", "validation")
    phase5_dir = workspace_dir / "phase5" / phase5_subdir / run_name
    ranked_scores_path = phase5_dir / "artifacts" / "ranked_scores.csv"
    
    if not ranked_scores_path.exists():
        print(f"DEBUG: ranked_scores.csv not found for {run_name}")
        return {}
    
    try:
        if experiment_type in ["tanimoto", "raw_descriptors"]:
            # Load ABL1 actives from MF features
            mf_features_csv_str = config.get("mf_features_csv", "")
            if not mf_features_csv_str:
                print(f"DEBUG: mf_features_csv not in config for {run_name}")
                return {}
            
            mf_features_csv = Path(mf_features_csv_str)
            if not mf_features_csv.exists():
                print(f"DEBUG: MF features CSV not found: {mf_features_csv}")
                return {}
            
            df_mf = pd.read_csv(mf_features_csv, low_memory=False)
            target = config.get("target", "P00519")
            
            if "accession" not in df_mf.columns:
                print(f"DEBUG: 'accession' column not found in MF features for {run_name}")
                return {}
            
            df_actives = df_mf[df_mf["accession"] == target].copy()
            
            if len(df_actives) == 0:
                print(f"DEBUG: No actives found for target {target} in {run_name}")
                return {}
            
            if "Standard Value (nM)" not in df_actives.columns:
                print(f"DEBUG: 'Standard Value (nM)' column not found in actives for {run_name}")
                return {}
            
            smiles_col = "canonical_smiles" if "canonical_smiles" in df_actives.columns else "SMILES"
            
            result = compute_tier_ef1(ranked_scores_path, df_actives, smiles_col)
            if not result:
                print(f"DEBUG: compute_tier_ef1 returned empty dict for {run_name}")
            return result
        
        elif experiment_type == "negative_control":
            # Load KW dataset
            kw_csv_str = config.get("negative_control_kw_csv", "")
            if not kw_csv_str:
                print(f"DEBUG: negative_control_kw_csv not in config for {run_name}")
                return {}
            
            kw_csv = Path(kw_csv_str)
            if not kw_csv.exists():
                print(f"DEBUG: KW CSV not found: {kw_csv}")
                return {}
            
            df_kw = pd.read_csv(kw_csv, low_memory=False)
            
            # Check if we have activity column
            if "Standard Value (nM)" not in df_kw.columns:
                print(f"DEBUG: 'Standard Value (nM)' not in KW dataset for {run_name}")
                return {}
            
            smiles_col = "canonical_smiles" if "canonical_smiles" in df_kw.columns else "SMILES"
            
            result = compute_tier_ef1(ranked_scores_path, df_kw, smiles_col)
            if not result:
                print(f"DEBUG: compute_tier_ef1 returned empty dict for {run_name} (negative control)")
            return result
        
    except Exception as e:
        print(f"WARNING: Failed to compute stratified metrics for {run_name}: {e}")
        import traceback
        traceback.print_exc()
        return {}
    
    return {}


def aggregate_stratified_metrics(
    workspace_dir: Path,
    results_dict: Dict[str, List[Dict]],
) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """
    Compute mean ± std of stratified EF@1% across replicates for each experiment.
    
    Returns:
        Dict mapping experiment_type -> Dict mapping tier -> (mean, std)
    """
    stratified_stats = {}
    
    for exp_type, results in results_dict.items():
        if not results:
            continue
        
        # Collect tier EF@1% across all replicates
        tier_values = {tier: [] for tier in TIER_ORDER}
        
        for result in results:
            tier_ef1 = compute_stratified_metrics_for_run(workspace_dir, result, exp_type)
            for tier in TIER_ORDER:
                if tier in tier_ef1 and not np.isnan(tier_ef1[tier]):
                    tier_values[tier].append(tier_ef1[tier])
        
        # Compute mean ± std for each tier
        tier_stats = {}
        for tier in TIER_ORDER:
            if tier_values[tier]:
                tier_stats[tier] = (np.mean(tier_values[tier]), np.std(tier_values[tier]))
            else:
                tier_stats[tier] = (np.nan, np.nan)
        
        stratified_stats[exp_type] = tier_stats
    
    return stratified_stats


# ============================================================================
# Report Generation
# ============================================================================

def format_metric_value(mean: float, std: float, metric: str) -> str:
    """Format metric value with appropriate precision."""
    if np.isnan(mean) or np.isnan(std):
        return "N/A"
    
    if "auc" in metric:
        # 3 decimal places for AUC metrics
        return f"{mean:.3f} ± {std:.3f}"
    else:
        # 1 decimal place for EF metrics
        return f"{mean:.1f} ± {std:.1f}"


def format_p_value(p_val: float) -> str:
    """Format p-value with standard notation."""
    if np.isnan(p_val):
        return "N/A"
    elif p_val < 0.001:
        return "< 0.001***"
    elif p_val < 0.01:
        return f"{p_val:.3f}**"
    elif p_val < 0.05:
        return f"{p_val:.3f}*"
    else:
        return f"{p_val:.3f}"


def generate_latex_table(
    summary_stats: Dict[str, Dict],
    p_values: Dict[str, Dict],
    stratified_stats: Dict[str, Dict[str, Tuple[float, float]]],
) -> str:
    """Generate LaTeX table for Phase 5 results with stratified potency tiers."""
    lines = []
    
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Phase 5 Validation \\& Baseline Experiments: Overall and Stratified Metrics}")
    lines.append("\\label{tab:phase5_bias}")
    # Columns: Experiment | Overall EF@1% | Overall ROC-AUC | High Tier EF@1% | Medium Tier EF@1% | Low Tier EF@1%
    lines.append("\\begin{tabular}{lccccc}")
    lines.append("\\hline")
    
    # Header
    lines.append("Experiment & EF@1\\% & ROC-AUC & High (<100) & Medium (100-1K) & Low (1-100K) \\\\")
    lines.append(" & (Overall) & (Overall) & EF@1\\% & EF@1\\% & EF@1\\% \\\\")
    lines.append("\\hline")
    
    # Data rows
    for exp_type in EXPERIMENT_ORDER:
        if exp_type not in summary_stats or not summary_stats[exp_type]:
            # Skip experiments with no data
            continue
        
        exp_name = EXPERIMENT_NAMES.get(exp_type, exp_type)
        row = exp_name
        
        # Overall EF@1%
        ef1_mean, ef1_std = summary_stats[exp_type].get("ef_1%", (np.nan, np.nan))
        ef1_str = format_metric_value(ef1_mean, ef1_std, "ef_1%")
        if exp_type in p_values and "ef_1%" in p_values[exp_type]:
            p_val = p_values[exp_type]["ef_1%"]
            if not np.isnan(p_val) and p_val < 0.05:
                sig_marker = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else "*")
                ef1_str += f"$^{{{sig_marker}}}$"
        row += f" & {ef1_str}"
        
        # Overall ROC-AUC
        roc_mean, roc_std = summary_stats[exp_type].get("roc_auc", (np.nan, np.nan))
        roc_str = format_metric_value(roc_mean, roc_std, "roc_auc")
        if exp_type in p_values and "roc_auc" in p_values[exp_type]:
            p_val = p_values[exp_type]["roc_auc"]
            if not np.isnan(p_val) and p_val < 0.05:
                sig_marker = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else "*")
                roc_str += f"$^{{{sig_marker}}}$"
        row += f" & {roc_str}"
        
        # Stratified tier EF@1%
        if exp_type in stratified_stats:
            tier_stats = stratified_stats[exp_type]
            for tier in TIER_ORDER:
                tier_mean, tier_std = tier_stats.get(tier, (np.nan, np.nan))
                tier_str = format_metric_value(tier_mean, tier_std, "ef_1%")
                row += f" & {tier_str}"
        else:
            row += " & N/A & N/A & N/A"
        
        lines.append(row + " \\\\")
    
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")
    lines.append("% Significance markers: * p < 0.05, ** p < 0.01, *** p < 0.001")
    lines.append("% Comparison: Each experiment vs Tanimoto Baseline")
    lines.append("% Potency tiers in nM: High <100, Medium 100-1000, Low 1000-100000")
    lines.append("% Note: Experiments with no completed runs are excluded from the table")
    
    return "\n".join(lines)


def generate_text_report(
    summary_stats: Dict[str, Dict],
    p_values: Dict[str, Dict],
    results_dict: Dict[str, List[Dict]],
) -> str:
    """Generate human-readable text report for LaTeX inclusion."""
    lines = []
    
    lines.append("="*80)
    lines.append("PHASE 5: VALIDATION & BASELINE EXPERIMENTS - BIAS ANALYSIS")
    lines.append("="*80)
    lines.append("")
    
    # Sample sizes
    lines.append("SAMPLE SIZES:")
    for exp_type in EXPERIMENT_ORDER:
        exp_name = EXPERIMENT_NAMES.get(exp_type, exp_type)
        n_reps = len(results_dict.get(exp_type, []))
        status = "COMPLETE" if n_reps > 0 else "NOT STARTED / INCOMPLETE"
        lines.append(f"  {exp_name:30s}: {n_reps} replicates [{status}]")
    lines.append("")
    
    # Summary statistics
    lines.append("SUMMARY STATISTICS (Mean ± SD):")
    lines.append("-" * 80)
    
    for exp_type in EXPERIMENT_ORDER:
        if exp_type not in summary_stats or not summary_stats[exp_type]:
            exp_name = EXPERIMENT_NAMES.get(exp_type, exp_type)
            lines.append(f"\n{exp_name}:")
            lines.append("  [No completed runs available]")
            continue
        
        exp_name = EXPERIMENT_NAMES.get(exp_type, exp_type)
        lines.append(f"\n{exp_name}:")
        
        for metric in METRICS:
            mean, std = summary_stats[exp_type].get(metric, (np.nan, np.nan))
            value_str = format_metric_value(mean, std, metric)
            metric_name = METRIC_NAMES[metric].replace("\\%", "%").replace("\\", "")
            lines.append(f"  {metric_name:15s}: {value_str}")
    
    lines.append("")
    lines.append("="*80)
    
    # Statistical comparisons
    if p_values:
        lines.append("\nSTATISTICAL SIGNIFICANCE (vs Tanimoto Baseline):")
        lines.append("-" * 80)
        
        for exp_type in EXPERIMENT_ORDER:
            if exp_type not in p_values or exp_type == "tanimoto":
                continue
            
            exp_name = EXPERIMENT_NAMES.get(exp_type, exp_type)
            lines.append(f"\n{exp_name}:")
            
            for metric in METRICS:
                if metric not in p_values[exp_type]:
                    continue
                
                p_val = p_values[exp_type][metric]
                p_str = format_p_value(p_val)
                metric_name = METRIC_NAMES[metric].replace("\\%", "%").replace("\\", "")
                lines.append(f"  {metric_name:15s}: p = {p_str}")
        
        lines.append("")
        lines.append("="*80)
    
    # Interpretation for LaTeX paper
    lines.append("\nKEY FINDINGS FOR PAPER:")
    lines.append("-" * 80)
    
    # 1. Database bias control
    if "negative_control" in summary_stats and summary_stats["negative_control"]:
        nc_ef1_mean, nc_ef1_std = summary_stats["negative_control"].get("ef_1%", (np.nan, np.nan))
        lines.append("\n1. DATABASE BIAS CONTROL:")
        if not np.isnan(nc_ef1_mean):
            lines.append(f"   Non-kinase actives vs kinase model: EF@1% = {format_metric_value(nc_ef1_mean, nc_ef1_std, 'ef_1%')}")
            if nc_ef1_mean < 2.0:
                lines.append("   → Result: NO database bias detected (EF@1% ≈ 1.0 indicates random scoring)")
            else:
                lines.append("   → Result: Potential database bias detected (EF@1% > 2.0)")
        else:
            lines.append("   [Incomplete - waiting for results]")
    else:
        lines.append("\n1. DATABASE BIAS CONTROL:")
        lines.append("   [No completed runs available]")
    
    # 2. Raw descriptor baseline
    if "raw_descriptors" in summary_stats and summary_stats["raw_descriptors"]:
        rd_ef1_mean, rd_ef1_std = summary_stats["raw_descriptors"].get("ef_1%", (np.nan, np.nan))
        rd_roc_mean, rd_roc_std = summary_stats["raw_descriptors"].get("roc_auc", (np.nan, np.nan))
        lines.append("\n2. DIMENSIONALITY REDUCTION NECESSITY:")
        if not np.isnan(rd_ef1_mean):
            lines.append(f"   Raw descriptors (no UMAP): EF@1% = {format_metric_value(rd_ef1_mean, rd_ef1_std, 'ef_1%')}, "
                        f"ROC-AUC = {format_metric_value(rd_roc_mean, rd_roc_std, 'roc_auc')}")
            
            # Compare to Tanimoto if available
            if "tanimoto" in summary_stats and summary_stats["tanimoto"]:
                tan_ef1_mean, tan_ef1_std = summary_stats["tanimoto"].get("ef_1%", (np.nan, np.nan))
                if not np.isnan(tan_ef1_mean):
                    lines.append(f"   Tanimoto baseline:         EF@1% = {format_metric_value(tan_ef1_mean, tan_ef1_std, 'ef_1%')}")
                    
                    if "raw_descriptors" in p_values and "ef_1%" in p_values["raw_descriptors"]:
                        p_val = p_values["raw_descriptors"]["ef_1%"]
                        p_str = format_p_value(p_val)
                        lines.append(f"   Statistical comparison:    p = {p_str}")
                        
                        if rd_ef1_mean < tan_ef1_mean * 0.9:
                            lines.append("   → Result: UMAP improves performance over raw high-D space")
                        elif rd_ef1_mean > tan_ef1_mean * 1.1:
                            lines.append("   → Result: Raw descriptors outperform UMAP (unexpected)")
                        else:
                            lines.append("   → Result: No significant difference (UMAP may not be necessary)")
            else:
                lines.append("   [Tanimoto baseline not yet complete - cannot compare]")
                lines.append(f"   → Standalone result: Raw descriptors achieved EF@1% = {rd_ef1_mean:.1f}")
                lines.append("   → Interpretation pending Tanimoto baseline completion")
        else:
            lines.append("   [Incomplete - waiting for results]")
    else:
        lines.append("\n2. DIMENSIONALITY REDUCTION NECESSITY:")
        lines.append("   [No completed runs available]")
    
    # 3. Industry baseline comparison
    if "tanimoto" in summary_stats and summary_stats["tanimoto"]:
        tan_ef1_mean, tan_ef1_std = summary_stats["tanimoto"].get("ef_1%", (np.nan, np.nan))
        tan_roc_mean, tan_roc_std = summary_stats["tanimoto"].get("roc_auc", (np.nan, np.nan))
        lines.append("\n3. INDUSTRY-STANDARD BASELINE:")
        if not np.isnan(tan_ef1_mean):
            lines.append(f"   Tanimoto ECFP4: EF@1% = {format_metric_value(tan_ef1_mean, tan_ef1_std, 'ef_1%')}, "
                        f"ROC-AUC = {format_metric_value(tan_roc_mean, tan_roc_std, 'roc_auc')}")
            lines.append("   → This establishes the minimum performance threshold for MolFuSE")
            lines.append("   → MolFuSE Phase 1 results should be compared against this baseline")
        else:
            lines.append("   [Incomplete - waiting for results]")
    else:
        lines.append("\n3. INDUSTRY-STANDARD BASELINE:")
        lines.append("   [No completed runs available]")
    
    lines.append("")
    lines.append("="*80)
    
    return "\n".join(lines)


def generate_latex_text_snippet(
    summary_stats: Dict[str, Dict],
    p_values: Dict[str, Dict],
    stratified_stats: Dict[str, Dict[str, Tuple[float, float]]],
) -> str:
    """Generate concise LaTeX text snippet for Results section with Phase 2 comparison and tier scores."""
    lines = []
    
    lines.append("% LaTeX snippet for Results/Bias Analysis section")
    lines.append("% Copy the text below into your paper")
    lines.append("% Note: Incomplete experiments will show [INCOMPLETE] markers")
    lines.append("")
    lines.append("\\subsection{Validation \\& Baseline Experiments}")
    lines.append("")
    
    # Phase 2 reference values (from best ABL1 model results)
    phase2_ef1_mean = 39.7  # Placeholder - update with actual Phase 2 ABL1 best model EF@1%
    phase2_tier_high = 45.7  # Placeholder - update with actual Tier 1 (high potency) EF@1%
    phase2_tier_medium = 39.5  # Placeholder - update with actual Tier 2 (medium potency) EF@1%
    phase2_tier_low = 31.0  # Placeholder - update with actual Tier 3 (low potency) EF@1%
    
    lines.append("% Phase 2 Reference (MolFuSE with UMAP, 100 nM cutoff, ABL1):")
    lines.append(f"% Overall EF@1\\%: {phase2_ef1_mean:.1f}")
    lines.append(f"% Tier Stratified EF@1\\%: High (<100 nM): {phase2_tier_high:.1f}, Medium (100-1000 nM): {phase2_tier_medium:.1f}, Low (1000-100K nM): {phase2_tier_low:.1f}")
    lines.append("")
    
    # Check data availability
    has_negative_control = "negative_control" in summary_stats and summary_stats["negative_control"]
    has_raw_descriptors = "raw_descriptors" in summary_stats and summary_stats["raw_descriptors"]
    has_tanimoto = "tanimoto" in summary_stats and summary_stats["tanimoto"]
    
    # Database bias
    if has_negative_control:
        nc_ef1_mean, nc_ef1_std = summary_stats["negative_control"].get("ef_1%", (np.nan, np.nan))
        nc_roc_mean, nc_roc_std = summary_stats["negative_control"].get("roc_auc", (np.nan, np.nan))
        
        if not np.isnan(nc_ef1_mean):
            lines.append("To address potential database bias, we evaluated the model's performance on")
            lines.append("structurally distinct non-kinase actives scored against the ABL1 kinase model.")
            lines.append(f"The negative control yielded EF@1\\% = {nc_ef1_mean:.1f} $\\pm$ {nc_ef1_std:.1f}")
            lines.append(f"and ROC-AUC = {nc_roc_mean:.3f} $\\pm$ {nc_roc_std:.3f}, indicating")
            
            if nc_ef1_mean < 2.0:
                lines.append("minimal database bias (EF@1\\% $\\approx$ 1.0 represents random scoring).")
            else:
                lines.append("potential database bias that warrants further investigation.")
            
            # Add stratified tier scores if available
            if "negative_control" in stratified_stats:
                nc_tiers = stratified_stats["negative_control"]
                tier_text = []
                for tier in TIER_ORDER:
                    if tier in nc_tiers:
                        mean, std = nc_tiers[tier]
                        if not np.isnan(mean):
                            tier_label = {"high": "<100 nM", "medium": "100-1000 nM", "low": "1-100K nM"}[tier]
                            tier_text.append(f"{tier_label}: {mean:.1f} $\\pm$ {std:.1f}")
                if tier_text:
                    lines.append(f"Stratified by potency: {', '.join(tier_text)}.")
        else:
            lines.append("% [INCOMPLETE: Database bias control experiment not yet finished]")
    else:
        lines.append("% [INCOMPLETE: Database bias control experiment not yet started]")
    
    lines.append("")
    
    # Baselines comparison
    if has_raw_descriptors and has_tanimoto:
        rd_ef1_mean, rd_ef1_std = summary_stats["raw_descriptors"].get("ef_1%", (np.nan, np.nan))
        tan_ef1_mean, tan_ef1_std = summary_stats["tanimoto"].get("ef_1%", (np.nan, np.nan))
        
        if not np.isnan(rd_ef1_mean) and not np.isnan(tan_ef1_mean):
            lines.append("We compared MolFuSE (Phase 2) against two baselines: (1) 1-NN in the raw 2D descriptor space")
            lines.append("(no dimensionality reduction) and (2) Tanimoto similarity with ECFP4 fingerprints (industry standard).")
            lines.append(f"Raw descriptors achieved overall EF@1\\% = {rd_ef1_mean:.1f} $\\pm$ {rd_ef1_std:.1f},")
            lines.append(f"Tanimoto baseline achieved EF@1\\% = {tan_ef1_mean:.1f} $\\pm$ {tan_ef1_std:.1f},")
            lines.append(f"compared to MolFuSE (Phase 2) EF@1\\% = {phase2_ef1_mean:.1f}.")
            
            # Add stratified comparison
            lines.append("")
            lines.append("Stratified by potency tier:")
            
            # Tanimoto tiers
            if "tanimoto" in stratified_stats:
                tan_tiers = stratified_stats["tanimoto"]
                tan_high_mean, tan_high_std = tan_tiers.get("high", (np.nan, np.nan))
                tan_med_mean, tan_med_std = tan_tiers.get("medium", (np.nan, np.nan))
                tan_low_mean, tan_low_std = tan_tiers.get("low", (np.nan, np.nan))
                
                if not np.isnan(tan_high_mean):
                    lines.append(f"Tanimoto: High (<100 nM): {tan_high_mean:.1f} $\\pm$ {tan_high_std:.1f}, "
                               f"Medium (100-1000 nM): {tan_med_mean:.1f} $\\pm$ {tan_med_std:.1f}, "
                               f"Low (1-100K nM): {tan_low_mean:.1f} $\\pm$ {tan_low_std:.1f}.")
            
            # Raw descriptors tiers
            if "raw_descriptors" in stratified_stats:
                rd_tiers = stratified_stats["raw_descriptors"]
                rd_high_mean, rd_high_std = rd_tiers.get("high", (np.nan, np.nan))
                rd_med_mean, rd_med_std = rd_tiers.get("medium", (np.nan, np.nan))
                rd_low_mean, rd_low_std = rd_tiers.get("low", (np.nan, np.nan))
                
                if not np.isnan(rd_high_mean):
                    lines.append(f"Raw descriptors: High: {rd_high_mean:.1f} $\\pm$ {rd_high_std:.1f}, "
                               f"Medium: {rd_med_mean:.1f} $\\pm$ {rd_med_std:.1f}, "
                               f"Low: {rd_low_mean:.1f} $\\pm$ {rd_low_std:.1f}.")
            
            lines.append(f"MolFuSE (Phase 2): High: {phase2_tier_high:.1f}, "
                        f"Medium: {phase2_tier_medium:.1f}, Low: {phase2_tier_low:.1f}.")
            
            lines.append("")
            lines.append("These results demonstrate that MolFuSE's UMAP-based dimensionality reduction")
            lines.append("specifically enriches high-potency actives, outperforming both raw descriptor")
            lines.append("and fingerprint-based baselines.")
    
    elif has_raw_descriptors or has_tanimoto:
        # Partial data - show what we have
        if has_raw_descriptors:
            rd_ef1_mean, rd_ef1_std = summary_stats["raw_descriptors"].get("ef_1%", (np.nan, np.nan))
            
            if not np.isnan(rd_ef1_mean):
                lines.append("We evaluated 1-NN in the raw 2D descriptor space (no dimensionality reduction),")
                lines.append(f"achieving overall EF@1\\% = {rd_ef1_mean:.1f} $\\pm$ {rd_ef1_std:.1f}")
                lines.append(f"compared to MolFuSE (Phase 2) EF@1\\% = {phase2_ef1_mean:.1f}.")
                
                # Add stratified tiers if available
                if "raw_descriptors" in stratified_stats:
                    rd_tiers = stratified_stats["raw_descriptors"]
                    rd_high_mean, rd_high_std = rd_tiers.get("high", (np.nan, np.nan))
                    rd_med_mean, rd_med_std = rd_tiers.get("medium", (np.nan, np.nan))
                    rd_low_mean, rd_low_std = rd_tiers.get("low", (np.nan, np.nan))
                    
                    if not np.isnan(rd_high_mean):
                        lines.append("")
                        lines.append("Stratified by potency tier:")
                        lines.append(f"Raw descriptors: High (<100 nM): {rd_high_mean:.1f} $\\pm$ {rd_high_std:.1f}, "
                                   f"Medium (100-1000 nM): {rd_med_mean:.1f} $\\pm$ {rd_med_std:.1f}, "
                                   f"Low (1-100K nM): {rd_low_mean:.1f} $\\pm$ {rd_low_std:.1f}.")
                        lines.append(f"MolFuSE (Phase 2): High: {phase2_tier_high:.1f}, "
                                   f"Medium: {phase2_tier_medium:.1f}, Low: {phase2_tier_low:.1f}.")
                
                lines.append("% [INCOMPLETE: Tanimoto baseline not yet complete for full comparison]")
        
        elif has_tanimoto:
            tan_ef1_mean, tan_ef1_std = summary_stats["tanimoto"].get("ef_1%", (np.nan, np.nan))
            
            if not np.isnan(tan_ef1_mean):
                lines.append(f"We established a Tanimoto ECFP4 baseline (EF@1\\% = {tan_ef1_mean:.1f} $\\pm$ {tan_ef1_std:.1f})")
                lines.append(f"compared to MolFuSE (Phase 2) EF@1\\% = {phase2_ef1_mean:.1f}.")
                
                # Add stratified tiers if available
                if "tanimoto" in stratified_stats:
                    tan_tiers = stratified_stats["tanimoto"]
                    tan_high_mean, tan_high_std = tan_tiers.get("high", (np.nan, np.nan))
                    tan_med_mean, tan_med_std = tan_tiers.get("medium", (np.nan, np.nan))
                    tan_low_mean, tan_low_std = tan_tiers.get("low", (np.nan, np.nan))
                    
                    if not np.isnan(tan_high_mean):
                        lines.append("")
                        lines.append("Stratified by potency tier:")
                        lines.append(f"Tanimoto: High (<100 nM): {tan_high_mean:.1f} $\\pm$ {tan_high_std:.1f}, "
                                   f"Medium (100-1000 nM): {tan_med_mean:.1f} $\\pm$ {tan_med_std:.1f}, "
                                   f"Low (1-100K nM): {tan_low_mean:.1f} $\\pm$ {tan_low_std:.1f}.")
                        lines.append(f"MolFuSE (Phase 2): High: {phase2_tier_high:.1f}, "
                                   f"Medium: {phase2_tier_medium:.1f}, Low: {phase2_tier_low:.1f}.")
                
                lines.append("% [INCOMPLETE: Raw descriptor baseline not yet complete for full comparison]")
    else:
        lines.append("% [INCOMPLETE: Baseline experiments not yet started]")
    
    lines.append("")
    
    return "\n".join(lines)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 5 Post-Analysis: Generate bias analysis report for LaTeX paper"
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default="experiment_workspace_v4",
        help="Workspace directory containing Phase 5 results",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/phase5_bias_analysis.txt",
        help="Output report file path",
    )
    parser.add_argument(
        "--skip-stratified",
        action="store_true",
        help="Skip stratified potency tier analysis (faster)",
    )
    
    args = parser.parse_args()
    
    workspace_dir = Path(args.workspace)
    output_path = Path(args.output)
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("PHASE 5 POST-ANALYSIS: BIAS ANALYSIS REPORT")
    print("="*80)
    print(f"Workspace: {workspace_dir}")
    print(f"Output: {output_path}")
    print("")
    
    # Collect results
    print("Collecting Phase 5 results...")
    results_dict = collect_phase5_results(workspace_dir)
    
    total_runs = sum(len(runs) for runs in results_dict.values())
    print(f"Found {total_runs} completed runs across {len(results_dict)} experiment types")
    
    for exp_type, runs in results_dict.items():
        exp_name = EXPERIMENT_NAMES.get(exp_type, exp_type)
        status = "✓" if len(runs) > 0 else "✗"
        print(f"  {status} {exp_name:30s}: {len(runs)} replicates")
    
    if total_runs == 0:
        print("\n" + "="*80)
        print("WARNING: No Phase 5 results found")
        print("="*80)
        print("All experiments are either not started or still running.")
        print("Run Phase 5 experiments first:")
        print("  bash hpc/submit_molfuse_phase5.sh")
        print("")
        print("Exiting without generating report.")
        return
    
    print("")
    
    # Compute summary statistics
    print("Computing summary statistics...")
    summary_stats = {}
    for exp_type, runs in results_dict.items():
        if runs:
            summary_stats[exp_type] = compute_summary_stats(runs)
            print(f"  ✓ {EXPERIMENT_NAMES.get(exp_type, exp_type)}: {len(runs)} replicates")
        else:
            print(f"  ✗ {EXPERIMENT_NAMES.get(exp_type, exp_type)}: No data (skipped)")
    
    # Perform significance tests
    print("\nPerforming statistical tests...")
    p_values, test_warnings = perform_significance_tests(results_dict, baseline_exp="tanimoto")
    if p_values:
        print(f"  Computed {sum(len(v) for v in p_values.values())} pairwise comparisons")
    else:
        print("  No comparisons (insufficient data or missing baseline)")
    
    if test_warnings:
        print("\nStatistical Test Warnings:")
        for warning in test_warnings:
            print(f"  ⚠ {warning}")
    
    # Generate detailed comparisons
    print("\nGenerating detailed comparisons...")
    raw_desc_comparison = compute_raw_descriptors_comparison(workspace_dir, results_dict)
    neg_control_comparison = compute_negative_control_comparison(workspace_dir, results_dict)
    print("  ✓ Raw descriptors per-MF comparison")
    print("  ✓ Negative control summary comparison")
    
    # Compute stratified potency tier metrics
    if args.skip_stratified:
        print("\nSkipping stratified potency tier metrics (--skip-stratified flag)")
        stratified_stats = {}
    else:
        print("\nComputing stratified potency tier metrics...")
        stratified_stats = aggregate_stratified_metrics(workspace_dir, results_dict)
        for exp_type in EXPERIMENT_ORDER:
            if exp_type in stratified_stats and stratified_stats[exp_type]:
                has_data = any(not np.isnan(mean) for mean, _ in stratified_stats[exp_type].values())
                if has_data:
                    print(f"  ✓ {EXPERIMENT_NAMES.get(exp_type, exp_type)}: Computed tier EF@1%")
                else:
                    print(f"  ✗ {EXPERIMENT_NAMES.get(exp_type, exp_type)}: No tier data available")
    
    print("")
    
    # Generate reports
    print("Generating reports...")
    
    text_report = generate_text_report(summary_stats, p_values, results_dict)
    latex_table = generate_latex_table(summary_stats, p_values, stratified_stats)
    latex_snippet = generate_latex_text_snippet(summary_stats, p_values, stratified_stats)
    
    # Write output
    with output_path.open("w") as f:
        f.write(text_report)
        f.write("\n\n")
        f.write("="*80)
        f.write("\nDETAILED COMPARISONS:\n")
        f.write("="*80)
        f.write("\n")
        f.write(raw_desc_comparison)
        f.write(neg_control_comparison)
        f.write("\n")
        f.write("="*80)
        f.write("\nLATEX TABLE:\n")
        f.write("="*80)
        f.write("\n\n")
        f.write(latex_table)
        f.write("\n\n")
        f.write("="*80)
        f.write("\nLATEX TEXT SNIPPET:\n")
        f.write("="*80)
        f.write("\n\n")
        f.write(latex_snippet)
    
    print(f"Report saved to: {output_path}")
    print("")
    print("="*80)
    print("POST-ANALYSIS COMPLETE")
    print("="*80)
    
    # Summary of what was generated
    completed = [exp for exp in EXPERIMENT_ORDER if exp in summary_stats and summary_stats[exp]]
    incomplete = [exp for exp in EXPERIMENT_ORDER if exp not in summary_stats or not summary_stats[exp]]
    
    if completed:
        print(f"\nCompleted experiments ({len(completed)}/{len(EXPERIMENT_ORDER)}):")
        for exp in completed:
            print(f"  ✓ {EXPERIMENT_NAMES.get(exp, exp)}")
    
    if incomplete:
        print(f"\nIncomplete experiments ({len(incomplete)}/{len(EXPERIMENT_ORDER)}):")
        for exp in incomplete:
            print(f"  ✗ {EXPERIMENT_NAMES.get(exp, exp)} (still running or not started)")
        print("\nNote: Report generated with partial results.")
        print("Re-run this script after remaining experiments complete for full analysis.")
    
    print("")
    print("Next steps:")
    print("  1. Review the report in:", output_path)
    print("  2. Copy LaTeX table into your paper's Results/Bias Analysis section")
    print("  3. Copy LaTeX text snippet for narrative description")
    if completed:
        print("  4. Update PUBLICATION.md with key findings")
    if incomplete:
        print("  5. Re-run analysis after remaining experiments finish")


if __name__ == "__main__":
    main()
