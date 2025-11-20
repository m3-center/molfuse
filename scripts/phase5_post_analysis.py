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


# ============================================================================
# Data Collection
# ============================================================================

def collect_phase5_results(workspace_dir: Path) -> Dict[str, List[Dict]]:
    """
    Collect all Phase 5 results from workspace.
    
    Returns:
        Dict mapping experiment_type -> list of result dicts
    """
    phase5_dir = workspace_dir / "phase5" / "validation"
    
    if not phase5_dir.exists():
        raise FileNotFoundError(f"Phase 5 validation directory not found: {phase5_dir}")
    
    results = {exp: [] for exp in EXPERIMENT_ORDER}
    
    for run_dir in sorted(phase5_dir.glob("phase5_*")):
        summary_path = run_dir / "logs" / "phase5_summary.json"
        
        if not summary_path.exists():
            print(f"WARNING: Missing summary for {run_dir.name}")
            continue
        
        try:
            with summary_path.open("r") as f:
                summary = json.load(f)
            
            exp_type = summary.get("experiment_type")
            if exp_type in results:
                results[exp_type].append(summary)
            else:
                print(f"WARNING: Unknown experiment type '{exp_type}' in {run_dir.name}")
        
        except Exception as e:
            print(f"ERROR: Failed to load {summary_path}: {e}")
    
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
) -> Dict[str, Dict[str, float]]:
    """
    Perform paired t-tests comparing each experiment to baseline.
    
    Returns:
        Dict mapping experiment -> Dict mapping metric -> p-value
    """
    p_values = {}
    
    baseline_results = results_dict.get(baseline_exp, [])
    if not baseline_results:
        print(f"WARNING: No results for baseline experiment '{baseline_exp}'")
        return {}
    
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
                t_stat, p_val = stats.ttest_rel(exp_vals, baseline_vals)
                p_values[exp_type][metric] = p_val
            elif len(baseline_vals) > 1 and len(exp_vals) > 1:
                # Independent t-test (different number of replicates)
                t_stat, p_val = stats.ttest_ind(exp_vals, baseline_vals)
                p_values[exp_type][metric] = p_val
            else:
                p_values[exp_type][metric] = np.nan
    
    return p_values


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


def generate_latex_table(summary_stats: Dict[str, Dict], p_values: Dict[str, Dict]) -> str:
    """Generate LaTeX table for Phase 5 results."""
    lines = []
    
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Phase 5 Validation \\& Baseline Experiments: Bias Analysis}")
    lines.append("\\label{tab:phase5_bias}")
    lines.append("\\begin{tabular}{l" + "c" * len(METRICS) + "}")
    lines.append("\\hline")
    
    # Header
    header = "Experiment"
    for metric in METRICS:
        header += f" & {METRIC_NAMES[metric]}"
    lines.append(header + " \\\\")
    lines.append("\\hline")
    
    # Data rows
    for exp_type in EXPERIMENT_ORDER:
        if exp_type not in summary_stats:
            continue
        
        exp_name = EXPERIMENT_NAMES.get(exp_type, exp_type)
        row = exp_name
        
        for metric in METRICS:
            mean, std = summary_stats[exp_type].get(metric, (np.nan, np.nan))
            value_str = format_metric_value(mean, std, metric)
            
            # Add significance marker if available
            if exp_type in p_values and metric in p_values[exp_type]:
                p_val = p_values[exp_type][metric]
                if not np.isnan(p_val) and p_val < 0.05:
                    sig_marker = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else "*")
                    value_str += f"$^{{{sig_marker}}}$"
            
            row += f" & {value_str}"
        
        lines.append(row + " \\\\")
    
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")
    lines.append("% Significance markers: * p < 0.05, ** p < 0.01, *** p < 0.001")
    lines.append("% Comparison: Each experiment vs Tanimoto Baseline")
    
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
        lines.append(f"  {exp_name:30s}: {n_reps} replicates")
    lines.append("")
    
    # Summary statistics
    lines.append("SUMMARY STATISTICS (Mean ± SD):")
    lines.append("-" * 80)
    
    for exp_type in EXPERIMENT_ORDER:
        if exp_type not in summary_stats:
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
    if "negative_control" in summary_stats:
        nc_ef1_mean, nc_ef1_std = summary_stats["negative_control"].get("ef_1%", (np.nan, np.nan))
        lines.append("\n1. DATABASE BIAS CONTROL:")
        lines.append(f"   Non-kinase actives vs kinase model: EF@1% = {format_metric_value(nc_ef1_mean, nc_ef1_std, 'ef_1%')}")
        if not np.isnan(nc_ef1_mean) and nc_ef1_mean < 2.0:
            lines.append("   → Result: NO database bias detected (EF@1% ≈ 1.0 indicates random scoring)")
        else:
            lines.append("   → Result: Potential database bias detected (EF@1% > 2.0)")
    
    # 2. Raw descriptor baseline
    if "raw_descriptors" in summary_stats and "tanimoto" in summary_stats:
        rd_ef1_mean, rd_ef1_std = summary_stats["raw_descriptors"].get("ef_1%", (np.nan, np.nan))
        tan_ef1_mean, tan_ef1_std = summary_stats["tanimoto"].get("ef_1%", (np.nan, np.nan))
        lines.append("\n2. DIMENSIONALITY REDUCTION NECESSITY:")
        lines.append(f"   Raw descriptors (no UMAP): EF@1% = {format_metric_value(rd_ef1_mean, rd_ef1_std, 'ef_1%')}")
        lines.append(f"   Tanimoto baseline:         EF@1% = {format_metric_value(tan_ef1_mean, tan_ef1_std, 'ef_1%')}")
        
        if "raw_descriptors" in p_values and "ef_1%" in p_values["raw_descriptors"]:
            p_val = p_values["raw_descriptors"]["ef_1%"]
            p_str = format_p_value(p_val)
            lines.append(f"   Statistical comparison:    p = {p_str}")
            
            if not np.isnan(rd_ef1_mean) and not np.isnan(tan_ef1_mean):
                if rd_ef1_mean < tan_ef1_mean * 0.9:
                    lines.append("   → Result: UMAP improves performance over raw high-D space")
                elif rd_ef1_mean > tan_ef1_mean * 1.1:
                    lines.append("   → Result: Raw descriptors outperform UMAP (unexpected)")
                else:
                    lines.append("   → Result: No significant difference (UMAP may not be necessary)")
    
    # 3. Industry baseline comparison
    if "tanimoto" in summary_stats:
        tan_ef1_mean, tan_ef1_std = summary_stats["tanimoto"].get("ef_1%", (np.nan, np.nan))
        tan_roc_mean, tan_roc_std = summary_stats["tanimoto"].get("roc_auc", (np.nan, np.nan))
        lines.append("\n3. INDUSTRY-STANDARD BASELINE:")
        lines.append(f"   Tanimoto ECFP4: EF@1% = {format_metric_value(tan_ef1_mean, tan_ef1_std, 'ef_1%')}, "
                    f"ROC-AUC = {format_metric_value(tan_roc_mean, tan_roc_std, 'roc_auc')}")
        lines.append("   → This establishes the minimum performance threshold for MolFuSE")
        lines.append("   → MolFuSE Phase 1 results should be compared against this baseline")
    
    lines.append("")
    lines.append("="*80)
    
    return "\n".join(lines)


def generate_latex_text_snippet(summary_stats: Dict[str, Dict], p_values: Dict[str, Dict]) -> str:
    """Generate concise LaTeX text snippet for Results section."""
    lines = []
    
    lines.append("% LaTeX snippet for Results/Bias Analysis section")
    lines.append("% Copy the text below into your paper")
    lines.append("")
    lines.append("\\subsection{Validation \\& Baseline Experiments}")
    lines.append("")
    
    # Database bias
    if "negative_control" in summary_stats:
        nc_ef1_mean, nc_ef1_std = summary_stats["negative_control"].get("ef_1%", (np.nan, np.nan))
        nc_roc_mean, nc_roc_std = summary_stats["negative_control"].get("roc_auc", (np.nan, np.nan))
        
        lines.append("To address potential database bias, we evaluated the model's performance on")
        lines.append("structurally distinct non-kinase actives (KW-0675\\_Receptor) scored against")
        lines.append(f"the ABL1 kinase model. The negative control yielded EF@1\\% = {nc_ef1_mean:.1f} $\\pm$ {nc_ef1_std:.1f}")
        lines.append(f"and ROC-AUC = {nc_roc_mean:.3f} $\\pm$ {nc_roc_std:.3f} (n=5 replicates), indicating")
        
        if not np.isnan(nc_ef1_mean) and nc_ef1_mean < 2.0:
            lines.append("minimal database bias (EF@1\\% $\\approx$ 1.0 represents random scoring).")
        else:
            lines.append("potential database bias that warrants further investigation.")
    
    lines.append("")
    
    # Baselines comparison
    if "raw_descriptors" in summary_stats and "tanimoto" in summary_stats:
        rd_ef1_mean, rd_ef1_std = summary_stats["raw_descriptors"].get("ef_1%", (np.nan, np.nan))
        tan_ef1_mean, tan_ef1_std = summary_stats["tanimoto"].get("ef_1%", (np.nan, np.nan))
        
        lines.append("We compared MolFuSE against two baselines: (1) 1-NN in the raw 2D descriptor space")
        lines.append("(no dimensionality reduction) and (2) Tanimoto similarity with ECFP4 fingerprints")
        lines.append(f"(industry standard). Raw descriptors achieved EF@1\\% = {rd_ef1_mean:.1f} $\\pm$ {rd_ef1_std:.1f},")
        lines.append(f"while Tanimoto baseline achieved EF@1\\% = {tan_ef1_mean:.1f} $\\pm$ {tan_ef1_std:.1f}.")
        
        if "raw_descriptors" in p_values and "ef_1%" in p_values["raw_descriptors"]:
            p_val = p_values["raw_descriptors"]["ef_1%"]
            if not np.isnan(p_val) and p_val < 0.05:
                lines.append(f"The difference was statistically significant (p = {format_p_value(p_val)}),")
                if rd_ef1_mean < tan_ef1_mean:
                    lines.append("demonstrating that dimensionality reduction with UMAP provides a meaningful")
                    lines.append("improvement over raw high-dimensional similarity scoring.")
            else:
                lines.append("The difference was not statistically significant (p > 0.05).")
    
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
        print(f"  {exp_name:30s}: {len(runs)} replicates")
    
    if total_runs == 0:
        print("\nERROR: No Phase 5 results found")
        print("Run Phase 5 experiments first:")
        print("  bash hpc/submit_molfuse_phase5.sh")
        return
    
    print("")
    
    # Compute summary statistics
    print("Computing summary statistics...")
    summary_stats = {}
    for exp_type, runs in results_dict.items():
        if runs:
            summary_stats[exp_type] = compute_summary_stats(runs)
    
    # Perform significance tests
    print("Performing statistical tests...")
    p_values = perform_significance_tests(results_dict, baseline_exp="tanimoto")
    
    print("")
    
    # Generate reports
    print("Generating reports...")
    
    text_report = generate_text_report(summary_stats, p_values, results_dict)
    latex_table = generate_latex_table(summary_stats, p_values)
    latex_snippet = generate_latex_text_snippet(summary_stats, p_values)
    
    # Write output
    with output_path.open("w") as f:
        f.write(text_report)
        f.write("\n\n")
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
    print("")
    print("Next steps:")
    print("  1. Review the report in:", output_path)
    print("  2. Copy LaTeX table into your paper's Results/Bias Analysis section")
    print("  3. Copy LaTeX text snippet for narrative description")
    print("  4. Update PUBLICATION.md with key findings")


if __name__ == "__main__":
    main()
