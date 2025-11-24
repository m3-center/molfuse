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

def load_phase1_reference_results(
    workspace_dir: Path,
    phase5_results: Dict[str, List[Dict]],
) -> Dict[str, List[Dict]]:
    """
    Load Phase 1/4 reference results that correspond to Phase 5 experiments.
    
    For each Phase 5 run, loads the Phase 1/4 baseline it's testing against:
    - tanimoto: Load Phase 1 ABL1_UMAP_fingerprints_20d_nn10_md0p0_rep[1-5]
    - raw_descriptors: Load Phase 4 cross_target features+UMAP runs (matching keyword/replicate)
    - negative_control: No baseline (testing database bias)
    
    Returns:
        Dict mapping experiment_type -> list of Phase 1/4 result dicts
    """
    phase1_dir = workspace_dir / "phase1"
    phase4_dir = workspace_dir / "phase4" / "cross_target"
    
    reference_results = {}
    
    # For each Phase 5 experiment type, find corresponding Phase 1/4 runs
    for exp_type, phase5_runs in phase5_results.items():
        if not phase5_runs:
            continue
        
        ref_runs = []
        
        for phase5_run in phase5_runs:
            config = phase5_run.get("config", {})
            # Phase 5 configs use 'target_short' not 'keyword'
            target_short = config.get("target_short", config.get("keyword", ""))
            replicate = config.get("replicate", 1)
            
            ref_summary = None
            run_dir = None
            
            # Determine which Phase 1/4 baseline to load
            if exp_type == "tanimoto":
                # Load Phase 1 best fingerprint model: ABL1_UMAP_fingerprints_20d_nn10_md0p0
                run_name = f"ABL1_UMAP_fingerprints_20d_nn10_md0p0_rep{replicate}"
                run_dir = phase1_dir / run_name
                
                if run_dir.exists():
                    summary_path = run_dir / "logs" / "phase1_summary.json"
                    if not summary_path.exists():
                        # Try metrics.json as fallback
                        summary_path = run_dir / "metrics" / "metrics.json"
                    
                    if summary_path.exists():
                        try:
                            with summary_path.open("r") as f:
                                ref_summary = json.load(f)
                        except Exception as e:
                            print(f"  Warning: Could not load {summary_path}: {e}")
                
            elif exp_type == "raw_descriptors":
                # Load Phase 4 features+UMAP run: umap_features_{target_short}_rep{replicate}
                if not target_short:
                    print(f"  Warning: No target_short found in config for raw_descriptors run")
                    continue
                
                run_name = f"umap_features_{target_short}_rep{replicate}"
                run_dir = phase4_dir / run_name
                
                if run_dir.exists():
                    summary_path = run_dir / "logs" / "phase4_summary.json"
                    if not summary_path.exists():
                        # Try metrics.json as fallback
                        summary_path = run_dir / "metrics" / "metrics.json"
                    
                    if summary_path.exists():
                        try:
                            with summary_path.open("r") as f:
                                ref_summary = json.load(f)
                        except Exception as e:
                            print(f"  Warning: Could not load {summary_path}: {e}")
                
            elif exp_type == "negative_control":
                # Negative control: no baseline comparison needed
                continue
            else:
                continue
            
            if ref_summary:
                ref_runs.append(ref_summary)
        
        if ref_runs:
            reference_results[exp_type] = ref_runs
            print(f"  Loaded {len(ref_runs)} Phase 1/4 reference runs for {exp_type}")
    
    return reference_results


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


def perform_baseline_comparison_tests(
    phase5_results: Dict[str, List[Dict]],
    phase1_reference: Dict[str, List[Dict]],
) -> Tuple[Dict[str, Dict[str, float]], List[str]]:
    """
    Compare each Phase 5 experiment against its Phase 1/4 baseline (with UMAP).
    
    Tests:
    - tanimoto (Phase 5 raw ECFP4) vs Phase 1 ECFP4+UMAP
    - raw_descriptors (Phase 5 raw features) vs Phase 1/4 features+UMAP
    - negative_control: not tested (different data - just report separately)
    
    Returns:
        Tuple of (p_values dict keyed by experiment, list of warning messages)
    """
    p_values = {}
    warning_messages = []
    
    for exp_type in ["tanimoto", "raw_descriptors"]:
        phase5_runs = phase5_results.get(exp_type, [])
        phase1_runs = phase1_reference.get(exp_type, [])
        
        if not phase5_runs or not phase1_runs:
            print(f"  Skipping {exp_type}: missing Phase 5 or Phase 1 data")
            continue
        
        p_values[exp_type] = {}
        
        for metric in METRICS:
            phase5_vals = [r.get(metric, np.nan) for r in phase5_runs]
            phase1_vals = [r.get(metric, np.nan) for r in phase1_runs]
            
            # Remove NaNs
            phase5_vals = [v for v in phase5_vals if not np.isnan(v)]
            phase1_vals = [v for v in phase1_vals if not np.isnan(v)]
            
            if len(phase5_vals) < 2 or len(phase1_vals) < 2:
                p_values[exp_type][metric] = np.nan
                continue
            
            # Independent t-test (Phase 5 no-UMAP vs Phase 1 with-UMAP)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                t_stat, p_val = stats.ttest_ind(phase5_vals, phase1_vals)
                if w:
                    for warning in w:
                        baseline_name = "Phase 1 ECFP4+UMAP" if exp_type == "tanimoto" else "Phase 1/4 features+UMAP"
                        msg = f"{exp_type} (no UMAP) vs {baseline_name} ({metric}): {warning.message}"
                        warning_messages.append(msg)
            
            p_values[exp_type][metric] = p_val
    
    return p_values, warning_messages


# ============================================================================
# Detailed Comparison: Each Phase 5 Experiment vs Its Phase 1/4 Baseline
# ============================================================================

def compute_baseline_comparisons(
    phase5_summary_stats: Dict[str, Dict],
    phase1_summary_stats: Dict[str, Dict],
) -> str:
    """
    Generate comparison table showing each Phase 5 experiment vs its Phase 1/4 baseline.
    
    Comparisons:
    - tanimoto (Phase 5 raw ECFP4) vs Phase 1 ECFP4+UMAP
    - raw_descriptors (Phase 5 raw features) vs Phase 1/4 features+UMAP
    - negative_control: Report separately (different data, no direct baseline)
    """
    lines = []
    lines.append("\nPhase 5 vs Phase 1/4 Baseline Comparisons")
    lines.append("="*80)
    
    # Tanimoto: Phase 5 raw ECFP4 vs Phase 1 ECFP4+UMAP
    if "tanimoto" in phase5_summary_stats and "tanimoto" in phase1_summary_stats:
        p5_ef1_mean, p5_ef1_std = phase5_summary_stats["tanimoto"].get("ef_1%", (np.nan, np.nan))
        p1_ef1_mean, p1_ef1_std = phase1_summary_stats["tanimoto"].get("ef_1%", (np.nan, np.nan))
        p5_roc_mean, p5_roc_std = phase5_summary_stats["tanimoto"].get("roc_auc", (np.nan, np.nan))
        p1_roc_mean, p1_roc_std = phase1_summary_stats["tanimoto"].get("roc_auc", (np.nan, np.nan))
        
        if not np.isnan(p5_ef1_mean) and not np.isnan(p1_ef1_mean):
            abs_delta = p5_ef1_mean - p1_ef1_mean
            rel_delta = (abs_delta / p1_ef1_mean) * 100 if p1_ef1_mean != 0 else np.nan
            
            lines.append("\n1. Tanimoto (Fingerprint Baseline):")
            lines.append(f"   Phase 5 (raw ECFP4, Jaccard, NO UMAP):     EF@1% = {p5_ef1_mean:.1f} ± {p5_ef1_std:.1f}  |  ROC-AUC = {p5_roc_mean:.3f} ± {p5_roc_std:.3f}")
            lines.append(f"   Phase 1 (ECFP4+UMAP, Euclidean):           EF@1% = {p1_ef1_mean:.1f} ± {p1_ef1_std:.1f}  |  ROC-AUC = {p1_roc_mean:.3f} ± {p1_roc_std:.3f}")
            lines.append("   " + "-"*70)
            lines.append(f"   Absolute Change (Δ):   {abs_delta:+.1f}")
            lines.append(f"   Relative Change (%):   {rel_delta:+.1f}%")
            
            if abs_delta > 0:
                lines.append(f"   → Phase 5 raw ECFP4 OUTPERFORMS Phase 1 ECFP4+UMAP by {abs(rel_delta):.1f}%")
                lines.append("   → UMAP is NOT necessary for fingerprints")
            else:
                lines.append(f"   → Phase 1 ECFP4+UMAP OUTPERFORMS Phase 5 raw ECFP4 by {abs(rel_delta):.1f}%")
                lines.append("   → UMAP improves fingerprint-based screening")
            
            # Add table format
            lines.append("")
            lines.append("   TABLE FORMAT:")
            lines.append(f"   {'Method':<35} {'EF@1%':<15} {'ROC-AUC':<15} {'Δ EF@1%':<12} {'Δ %':<10}")
            lines.append("   " + "-"*87)
            lines.append(f"   {'Phase 5 (raw ECFP4, NO UMAP)':<35} {p5_ef1_mean:>6.1f} ± {p5_ef1_std:<5.1f} {p5_roc_mean:>5.3f} ± {p5_roc_std:<5.3f} {'—':<12} {'—':<10}")
            lines.append(f"   {'Phase 1 (ECFP4+UMAP)':<35} {p1_ef1_mean:>6.1f} ± {p1_ef1_std:<5.1f} {p1_roc_mean:>5.3f} ± {p1_roc_std:<5.3f} {abs_delta:>+11.1f} {rel_delta:>+9.1f}%")
    
    # Raw Descriptors: Phase 5 raw features vs Phase 1/4 features+UMAP
    if "raw_descriptors" in phase5_summary_stats and "raw_descriptors" in phase1_summary_stats:
        p5_ef1_mean, p5_ef1_std = phase5_summary_stats["raw_descriptors"].get("ef_1%", (np.nan, np.nan))
        p1_ef1_mean, p1_ef1_std = phase1_summary_stats["raw_descriptors"].get("ef_1%", (np.nan, np.nan))
        p5_roc_mean, p5_roc_std = phase5_summary_stats["raw_descriptors"].get("roc_auc", (np.nan, np.nan))
        p1_roc_mean, p1_roc_std = phase1_summary_stats["raw_descriptors"].get("roc_auc", (np.nan, np.nan))
        
        if not np.isnan(p5_ef1_mean) and not np.isnan(p1_ef1_mean):
            abs_delta = p5_ef1_mean - p1_ef1_mean
            rel_delta = (abs_delta / p1_ef1_mean) * 100 if p1_ef1_mean != 0 else np.nan
            
            lines.append("\n2. Raw Descriptors (Feature Baseline):")
            lines.append(f"   Phase 5 (raw features, scaled, NO UMAP):   EF@1% = {p5_ef1_mean:.1f} ± {p5_ef1_std:.1f}  |  ROC-AUC = {p5_roc_mean:.3f} ± {p5_roc_std:.3f}")
            lines.append(f"   Phase 4 (features+UMAP, Euclidean):        EF@1% = {p1_ef1_mean:.1f} ± {p1_ef1_std:.1f}  |  ROC-AUC = {p1_roc_mean:.3f} ± {p1_roc_std:.3f}")
            lines.append("   " + "-"*70)
            lines.append(f"   Absolute Change (Δ):   {abs_delta:+.1f}")
            lines.append(f"   Relative Change (%):   {rel_delta:+.1f}%")
            
            if abs_delta > 0:
                lines.append(f"   → Phase 5 raw features OUTPERFORM Phase 4 features+UMAP by {abs(rel_delta):.1f}%")
                lines.append("   → UMAP is NOT necessary for descriptors")
            else:
                lines.append(f"   → Phase 4 features+UMAP OUTPERFORM Phase 5 raw features by {abs(rel_delta):.1f}%")
                lines.append("   → UMAP improves feature-based screening")
            
            # Add table format
            lines.append("")
            lines.append("   TABLE FORMAT:")
            lines.append(f"   {'Method':<35} {'EF@1%':<15} {'ROC-AUC':<15} {'Δ EF@1%':<12} {'Δ %':<10}")
            lines.append("   " + "-"*87)
            lines.append(f"   {'Phase 5 (raw features, NO UMAP)':<35} {p5_ef1_mean:>6.1f} ± {p5_ef1_std:<5.1f} {p5_roc_mean:>5.3f} ± {p5_roc_std:<5.3f} {'—':<12} {'—':<10}")
            lines.append(f"   {'Phase 4 (features+UMAP)':<35} {p1_ef1_mean:>6.1f} ± {p1_ef1_std:<5.1f} {p1_roc_mean:>5.3f} ± {p1_roc_std:<5.3f} {abs_delta:>+11.1f} {rel_delta:>+9.1f}%")
    
    # Negative Control: Just report (no baseline comparison)
    if "negative_control" in phase5_summary_stats:
        nc_ef1_mean, nc_ef1_std = phase5_summary_stats["negative_control"].get("ef_1%", (np.nan, np.nan))
        
        if not np.isnan(nc_ef1_mean):
            lines.append("\n3. Negative Control (Database Bias Test):")
            lines.append(f"   Non-kinase actives vs kinase model:        EF@1% = {nc_ef1_mean:.1f} ± {nc_ef1_std:.1f}")
            lines.append("   Expected random performance:               EF@1% ≈ 1.0")
            lines.append("   " + "-"*70)
            
            if nc_ef1_mean < 2.0:
                lines.append("   → EF@1% ≈ 1.0 indicates NO database bias (good)")
            else:
                lines.append(f"   → EF@1% = {nc_ef1_mean:.1f} suggests POTENTIAL database bias")
                lines.append("   → Model may be enriching ChEMBL compounds regardless of target")
    
    lines.append("\n" + "="*80)
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
    
    # Data rows (no significance markers - see pairwise comparisons separately)
    for exp_type in EXPERIMENT_ORDER:
        if exp_type not in summary_stats or not summary_stats[exp_type]:
            # Skip experiments with no data
            continue
        
        exp_name = EXPERIMENT_NAMES.get(exp_type, exp_type)
        row = exp_name
        
        # Overall EF@1%
        ef1_mean, ef1_std = summary_stats[exp_type].get("ef_1%", (np.nan, np.nan))
        ef1_str = format_metric_value(ef1_mean, ef1_std, "ef_1%")
        row += f" & {ef1_str}"
        
        # Overall ROC-AUC
        roc_mean, roc_std = summary_stats[exp_type].get("roc_auc", (np.nan, np.nan))
        roc_str = format_metric_value(roc_mean, roc_std, "roc_auc")
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
    lines.append("% Potency tiers in nM: High <100, Medium 100-1000, Low 1000-100000")
    lines.append("% For pairwise statistical significance, see text report")
    lines.append("% Note: Experiments with no completed runs are excluded from the table")
    
    return "\n".join(lines)


def generate_text_report(
    summary_stats: Dict[str, Dict],
    p_values: Dict[str, Dict],
    results_dict: Dict[str, List[Dict]],
    phase1_reference: Dict[str, Dict] = None,
) -> str:
    """Generate human-readable text report comparing Phase 5 to Phase 1/4 baselines."""
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
    
    # Phase 5 vs Phase 1/4 baseline comparisons
    if p_values:
        lines.append("\nSTATISTICAL SIGNIFICANCE (Phase 5 no-UMAP vs Phase 1/4 with-UMAP):")
        lines.append("-" * 80)
        
        for exp_type, metrics in p_values.items():
            exp_name = EXPERIMENT_NAMES.get(exp_type, exp_type)
            baseline_name = "Phase 1 ECFP4+UMAP" if exp_type == "tanimoto" else "Phase 1/4 features+UMAP"
            lines.append(f"\n{exp_name} (Phase 5 no-UMAP) vs {baseline_name}:")
            
            for metric in METRICS:
                if metric not in metrics:
                    continue
                
                p_val = metrics[metric]
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
    phase1_reference: Dict[str, Dict] = None,
) -> str:
    """Generate concise LaTeX text snippet showing Phase 5 vs Phase 1/4 baseline comparisons."""
    lines = []
    
    lines.append("% LaTeX snippet for Results/Bias Analysis section")
    lines.append("% Copy the text below into your paper")
    lines.append("% Note: Incomplete experiments will show [INCOMPLETE] markers")
    lines.append("")
    lines.append("\\subsection{Phase 5: Baseline Validation Experiments}")
    lines.append("")
    
    # Check data availability
    has_negative_control = "negative_control" in summary_stats and summary_stats["negative_control"]
    has_raw_descriptors = "raw_descriptors" in summary_stats and summary_stats["raw_descriptors"]
    has_tanimoto = "tanimoto" in summary_stats and summary_stats["tanimoto"]
    
    # Summary of all 3 experiments
    lines.append("We conducted three baseline experiments to validate MolFuSE's approach:")
    lines.append("")
    
    # Experiment 1: Tanimoto
    if has_tanimoto:
        tan_ef1_mean, tan_ef1_std = summary_stats["tanimoto"].get("ef_1%", (np.nan, np.nan))
        tan_roc_mean, tan_roc_std = summary_stats["tanimoto"].get("roc_auc", (np.nan, np.nan))
        
        if not np.isnan(tan_ef1_mean):
            lines.append("\\textbf{Tanimoto Baseline (Industry Standard):} ")
            lines.append(f"ECFP4 fingerprint similarity achieved EF@1\\% = {tan_ef1_mean:.1f} $\\pm$ {tan_ef1_std:.1f} ")
            lines.append(f"and ROC-AUC = {tan_roc_mean:.3f} $\\pm$ {tan_roc_std:.3f}.")
            
            if "tanimoto" in stratified_stats:
                tan_tiers = stratified_stats["tanimoto"]
                tier_strs = []
                for tier, label in [("high", "High (<100 nM)"), ("medium", "Med (100-1K nM)"), ("low", "Low (1-100K nM)")]:
                    mean, std = tan_tiers.get(tier, (np.nan, np.nan))
                    if not np.isnan(mean):
                        tier_strs.append(f"{label}: {mean:.1f}$\\pm${std:.1f}")
                if tier_strs:
                    lines.append(f"Stratified: {', '.join(tier_strs)}.")
            lines.append("")
        else:
            lines.append("\\textbf{Tanimoto Baseline:} [INCOMPLETE]")
            lines.append("")
    
    # Experiment 2: Raw Descriptors
    if has_raw_descriptors:
        rd_ef1_mean, rd_ef1_std = summary_stats["raw_descriptors"].get("ef_1%", (np.nan, np.nan))
        rd_roc_mean, rd_roc_std = summary_stats["raw_descriptors"].get("roc_auc", (np.nan, np.nan))
        
        if not np.isnan(rd_ef1_mean):
            lines.append("\\textbf{Raw Descriptors (No UMAP):} ")
            lines.append(f"1-NN in high-dimensional scaled feature space achieved EF@1\\% = {rd_ef1_mean:.1f} $\\pm$ {rd_ef1_std:.1f} ")
            lines.append(f"and ROC-AUC = {rd_roc_mean:.3f} $\\pm$ {rd_roc_std:.3f}.")
            
            if "raw_descriptors" in stratified_stats:
                rd_tiers = stratified_stats["raw_descriptors"]
                tier_strs = []
                for tier, label in [("high", "High"), ("medium", "Med"), ("low", "Low")]:
                    mean, std = rd_tiers.get(tier, (np.nan, np.nan))
                    if not np.isnan(mean):
                        tier_strs.append(f"{label}: {mean:.1f}$\\pm${std:.1f}")
                if tier_strs:
                    lines.append(f"Stratified: {', '.join(tier_strs)}.")
            lines.append("")
        else:
            lines.append("\\textbf{Raw Descriptors:} [INCOMPLETE]")
            lines.append("")
    
    # Experiment 3: Negative Control
    if has_negative_control:
        nc_ef1_mean, nc_ef1_std = summary_stats["negative_control"].get("ef_1%", (np.nan, np.nan))
        nc_roc_mean, nc_roc_std = summary_stats["negative_control"].get("roc_auc", (np.nan, np.nan))
        
        if not np.isnan(nc_ef1_mean):
            lines.append("\\textbf{Negative Control (Database Bias Test):} ")
            lines.append(f"Non-kinase actives scored against kinase model: EF@1\\% = {nc_ef1_mean:.1f} $\\pm$ {nc_ef1_std:.1f} ")
            lines.append(f"and ROC-AUC = {nc_roc_mean:.3f} $\\pm$ {nc_roc_std:.3f}.")
            
            if nc_ef1_mean < 2.0:
                lines.append("This near-random performance indicates minimal database bias.")
            else:
                lines.append("This elevated performance suggests potential database bias.")
            
            if "negative_control" in stratified_stats:
                nc_tiers = stratified_stats["negative_control"]
                tier_strs = []
                for tier, label in [("high", "High"), ("medium", "Med"), ("low", "Low")]:
                    mean, std = nc_tiers.get(tier, (np.nan, np.nan))
                    if not np.isnan(mean):
                        tier_strs.append(f"{label}: {mean:.1f}$\\pm${std:.1f}")
                if tier_strs:
                    lines.append(f"Stratified: {', '.join(tier_strs)}.")
            lines.append("")
        else:
            lines.append("\\textbf{Negative Control:} [INCOMPLETE]")
            lines.append("")
    
    # Interpretation note
    lines.append("\\textbf{Note:} Pairwise statistical comparisons between experiments are provided in the supplementary materials.")
    lines.append("All three experiments use the same evaluation set (ABL1 kinase actives + ZINC decoys).")
    
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
    
    # Load Phase 1/4 reference results (with UMAP baselines)
    print("\nLoading Phase 1/4 reference results (UMAP baselines)...")
    phase1_reference = load_phase1_reference_results(workspace_dir, results_dict)
    if phase1_reference:
        total_ref = sum(len(runs) for runs in phase1_reference.values())
        print(f"  Loaded {total_ref} Phase 1/4 baseline runs for comparison")
    else:
        print("  WARNING: No Phase 1/4 reference data found - cannot perform baseline comparisons")
    
    # Compute Phase 1/4 summary stats from reference results
    phase1_summary_stats = {}
    for exp_type in EXPERIMENT_ORDER:
        if exp_type in phase1_reference and phase1_reference[exp_type]:
            phase1_summary_stats[exp_type] = compute_summary_stats(phase1_reference[exp_type])
    
    # Perform statistical tests: Phase 5 (no UMAP) vs Phase 1/4 (with UMAP)
    print("\nPerforming statistical tests: Phase 5 (no UMAP) vs Phase 1/4 (with UMAP)...")
    p_values, test_warnings = perform_baseline_comparison_tests(results_dict, phase1_reference)
    if p_values:
        n_comparisons = sum(len(metrics) for metrics in p_values.values())
        print(f"  Computed {n_comparisons} metric comparisons across {len(p_values)} experiments")
    else:
        print("  No comparisons (insufficient data)")
    
    if test_warnings:
        print("\nStatistical Test Warnings:")
        for warning in test_warnings:
            print(f"  ⚠ {warning}")
    
    # Generate detailed comparisons (each Phase 5 experiment vs its Phase 1/4 baseline)
    print("\nGenerating baseline comparisons...")
    baseline_comparison = compute_baseline_comparisons(summary_stats, phase1_summary_stats)
    print("  ✓ Each Phase 5 experiment vs its Phase 1/4 baseline")
    
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
    
    # Compute Phase 1 reference summary stats for comparison
    phase1_summary_stats = {}
    if phase1_reference:
        for exp_type, runs in phase1_reference.items():
            if runs:
                phase1_summary_stats[exp_type] = compute_summary_stats(runs)
    
    # Generate reports
    print("Generating reports...")
    
    text_report = generate_text_report(summary_stats, p_values, results_dict, phase1_summary_stats)
    latex_table = generate_latex_table(summary_stats, p_values, stratified_stats)
    latex_snippet = generate_latex_text_snippet(summary_stats, p_values, stratified_stats, phase1_summary_stats)
    
    # Write output
    with output_path.open("w") as f:
        f.write(text_report)
        f.write("\n\n")
        f.write("="*80)
        f.write("\nBASELINE COMPARISONS (Phase 5 vs Phase 1/4):\n")
        f.write("="*80)
        f.write("\n")
        f.write(baseline_comparison)
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
