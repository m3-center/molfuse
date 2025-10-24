#!/usr/bin/env python3
"""
Phase 2 Cutoff Sensitivity Analysis and Visualization

This script analyzes the results of Phase 2 affinity cutoff sensitivity experiments
and generates publication-ready figures showing:

1. Cutoff sensitivity curves (EF@1% vs cutoff)
2. Potency-stratified enrichment heatmap
3. Active count vs performance trade-off

Usage:
    python analysis_scripts/analyze_phase2_cutoff_sensitivity.py \\
        --phase2_workspace experiment_workspace_v3_phase2 \\
        --output_dir phase2_analysis_results
"""

import os
import sys
import glob
import argparse
import logging
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)-8s - %(message)s'
)

# Plot styling
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Constants
CUTOFFS = [100, 1000, 10000, 100000]  # nM
CUTOFF_LABELS = ['100 nM', '1 μM', '10 μM', '100 μM']
CONFIG_TYPES = ['pca_overall', 'umap_overall', 'umap_high_potency']
CONFIG_LABELS = {
    'pca_overall': 'PCA/features/5D',
    'umap_overall': 'UMAP/features/5D (overall-EF)',
    'umap_high_potency': 'UMAP/features/5D (high-potency-EF)'
}
CONFIG_COLORS = {
    'pca_overall': '#1f77b4',  # blue
    'umap_overall': '#ff7f0e',  # orange
    'umap_high_potency': '#d62728'  # red
}
CONFIG_LINESTYLES = {
    'pca_overall': '-',
    'umap_overall': '--',
    'umap_high_potency': ':'
}


def find_phase2_runs(workspace_dir):
    """Find all Phase 2 run directories organized by config type and cutoff.
    
    Returns:
        dict: {config_type: {cutoff: [run_dirs]}}
    """
    runs = {cfg: {cutoff: [] for cutoff in CUTOFFS} for cfg in CONFIG_TYPES}
    
    pattern = os.path.join(workspace_dir, "run_seed*_features_*_dim5_cutoff*nM_*")
    all_runs = glob.glob(pattern)
    
    logging.info(f"Found {len(all_runs)} Phase 2 run directories")
    
    for run_dir in all_runs:
        basename = os.path.basename(run_dir)
        
        # Extract cutoff from directory name
        cutoff = None
        for c in CUTOFFS:
            if f"cutoff{c}nM" in basename:
                cutoff = c
                break
        
        if cutoff is None:
            logging.warning(f"Could not extract cutoff from: {basename}")
            continue
        
        # Extract config type
        config_type = None
        if '_pca_' in basename and basename.endswith('_pca_overall'):
            config_type = 'pca_overall'
        elif 'umap' in basename and basename.endswith('_umap_overall'):
            config_type = 'umap_overall'
        elif 'umap' in basename and basename.endswith('_umap_high_potency'):
            config_type = 'umap_high_potency'
        
        if config_type is None:
            logging.warning(f"Could not extract config type from: {basename}")
            continue
        
        runs[config_type][cutoff].append(run_dir)
    
    # Log summary
    for config_type in CONFIG_TYPES:
        for cutoff in CUTOFFS:
            count = len(runs[config_type][cutoff])
            logging.info(f"  {CONFIG_LABELS[config_type]} @ {cutoff}nM: {count} runs")
    
    return runs


def load_metrics(run_dir):
    """Load ranking metrics from a Phase 2 run directory.
    
    Returns:
        dict: Metrics including EF@1%, ROC AUC, etc., or None if not found
    """
    # Look for ranking metrics CSV (can be in PCA/ or UMAP_Euclidean/ subdirectory)
    pattern = os.path.join(
        run_dir,
        "TyrosineProteinKinaseABL1_P00519",
        "results",
        "features",
        "dim_5",
        "**",
        "*_ranking_metrics.csv"
    )
    
    csv_files = glob.glob(pattern, recursive=True)
    
    if not csv_files:
        logging.warning(f"No metrics found in: {run_dir}")
        return None
    
    if len(csv_files) > 1:
        logging.warning(f"Multiple metrics files found in {run_dir}, using first")
    
    try:
        df = pd.read_csv(csv_files[0])
        if df.empty:
            return None
        
        metrics = df.iloc[0].to_dict()
        return metrics
    
    except Exception as e:
        logging.error(f"Error loading metrics from {csv_files[0]}: {e}")
        return None


def load_ranked_data(run_dir):
    """Load ranked compound data to count actives.
    
    Returns:
        tuple: (num_actives, num_total) or (None, None) if not found
    """
    # Pattern to find ranked CSV files (can be in PCA/ or UMAP_Euclidean/)
    pattern = os.path.join(
        run_dir,
        "TyrosineProteinKinaseABL1_P00519",
        "results",
        "features",
        "dim_5",
        "**",
        "TYROSINEPROTEINKINASEABL1_P00519-*-5D-FEATURES.csv"
    )
    
    csv_files = glob.glob(pattern, recursive=True)
    
    if not csv_files:
        return None, None
    
    try:
        df = pd.read_csv(csv_files[0])
        if df.empty or 'TYPE' not in df.columns:
            return None, None
        num_actives = len(df[df['TYPE'] == 'HELDOUT_ACTIVE'])
        num_total = len(df)
        return num_actives, num_total
    
    except pd.errors.EmptyDataError:
        logging.debug(f"Empty CSV file: {csv_files[0]}")
        return None, None
    except Exception as e:
        logging.warning(f"Error loading ranked data from {csv_files[0]}: {e}")
        return None, None


def aggregate_results(phase2_runs):
    """Aggregate metrics across all Phase 2 runs.
    
    Returns:
        pd.DataFrame: Aggregated results with columns [config_type, cutoff, seed, ef_1%, roc_auc, etc.]
    """
    results = []
    
    for config_type in CONFIG_TYPES:
        for cutoff in CUTOFFS:
            run_dirs = phase2_runs[config_type][cutoff]
            
            for run_dir in run_dirs:
                # Extract seed from directory name
                basename = os.path.basename(run_dir)
                seed = None
                for s in [42, 43, 44, 45, 46]:
                    if f"seed{s}_" in basename:
                        seed = s
                        break
                
                if seed is None:
                    logging.warning(f"Could not extract seed from: {basename}")
                    continue
                
                # Load metrics
                metrics = load_metrics(run_dir)
                if metrics is None:
                    continue
                
                # Load active counts
                num_actives, num_total = load_ranked_data(run_dir)
                
                # Build result row
                row = {
                    'config_type': config_type,
                    'cutoff_nM': cutoff,
                    'seed': seed,
                    'ef_1_pct': metrics.get('ef_1%', np.nan),
                    'ef_5_pct': metrics.get('ef_5%', np.nan),
                    'ef_10_pct': metrics.get('ef_10%', np.nan),
                    'roc_auc': metrics.get('roc_auc', np.nan),
                    'pr_auc': metrics.get('pr_auc', np.nan),
                    'spearman_rho': metrics.get('spearman_rho_affinity_vs_score', np.nan),
                    'num_actives': num_actives,
                    'num_total': num_total
                }
                
                results.append(row)
    
    df = pd.DataFrame(results)
    logging.info(f"Aggregated {len(df)} result rows from Phase 2")
    
    return df


def plot_cutoff_sensitivity_curves(df_results, output_dir):
    """Plot 1: Cutoff sensitivity curves (main figure).
    
    Line plot showing EF@1% vs cutoff for each configuration.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Calculate mean and std for each config at each cutoff
    for config_type in CONFIG_TYPES:
        df_config = df_results[df_results['config_type'] == config_type]
        
        means = []
        stds = []
        cutoff_values = []
        
        for cutoff in CUTOFFS:
            df_cutoff = df_config[df_config['cutoff_nM'] == cutoff]
            
            if len(df_cutoff) > 0:
                mean_ef = df_cutoff['ef_1_pct'].mean()
                std_ef = df_cutoff['ef_1_pct'].std()
                
                means.append(mean_ef)
                stds.append(std_ef)
                cutoff_values.append(cutoff)
        
        if not means:
            continue
        
        # Plot line with error bars
        ax.errorbar(
            cutoff_values,
            means,
            yerr=stds,
            label=CONFIG_LABELS[config_type],
            color=CONFIG_COLORS[config_type],
            linestyle=CONFIG_LINESTYLES[config_type],
            linewidth=2.5,
            marker='o',
            markersize=8,
            capsize=5,
            capthick=2
        )
    
    # Formatting
    ax.set_xscale('log')
    ax.set_xlabel('Affinity Cutoff (nM)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Enrichment Factor @ 1%', fontsize=14, fontweight='bold')
    ax.set_title('Phase 2: Affinity Cutoff Sensitivity Analysis\nEF@1% vs Cutoff Threshold',
                 fontsize=16, fontweight='bold', pad=20)
    
    ax.set_xticks(CUTOFFS)
    ax.set_xticklabels(CUTOFF_LABELS)
    ax.tick_params(labelsize=12)
    
    ax.legend(fontsize=11, loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add annotations for potency tiers
    ax.axvspan(0, 100, alpha=0.1, color='green', label='High-potent\n(0.1-100 nM)')
    ax.axvspan(100, 1000, alpha=0.1, color='yellow', label='Medium-potent\n(100-1K nM)')
    ax.axvspan(1000, 10000, alpha=0.1, color='orange', label='Weak-potent\n(1K-10K nM)')
    ax.axvspan(10000, 100000, alpha=0.1, color='red', label='Very weak\n(>10K nM)')
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(output_dir, 'phase2_cutoff_sensitivity_curves.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logging.info(f"Saved cutoff sensitivity curves to: {output_path}")
    
    # Also save PDF for publication
    output_path_pdf = os.path.join(output_dir, 'phase2_cutoff_sensitivity_curves.pdf')
    plt.savefig(output_path_pdf, bbox_inches='tight')
    
    plt.close()


def plot_potency_stratified_heatmap(df_results, output_dir):
    """Plot 2: Potency-stratified enrichment heatmap.
    
    Heatmap showing EF@1% for different cutoffs and configurations.
    Each cell shows mean EF@1% across seeds.
    """
    # Prepare data for heatmap
    pivot_data = df_results.pivot_table(
        values='ef_1_pct',
        index='config_type',
        columns='cutoff_nM',
        aggfunc='mean'
    )
    
    # Reorder rows (only include configs that have data)
    available_configs = [ct for ct in CONFIG_TYPES if ct in pivot_data.index]
    if not available_configs:
        logging.warning("No data available for heatmap")
        return
    
    pivot_data = pivot_data.reindex(available_configs)
    
    # Reorder columns (only include cutoffs that have data)
    available_cutoffs = [c for c in CUTOFFS if c in pivot_data.columns]
    if not available_cutoffs:
        logging.warning("No cutoffs available for heatmap")
        return
    
    pivot_data = pivot_data[available_cutoffs]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create heatmap
    sns.heatmap(
        pivot_data,
        annot=True,
        fmt='.2f',
        cmap='YlOrRd',
        cbar_kws={'label': 'EF@1%'},
        linewidths=0.5,
        linecolor='gray',
        ax=ax,
        vmin=0,
        vmax=pivot_data.max().max() * 1.1
    )
    
    # Formatting
    ax.set_xlabel('Affinity Cutoff', fontsize=14, fontweight='bold')
    ax.set_ylabel('Configuration', fontsize=14, fontweight='bold')
    ax.set_title('Phase 2: Enrichment Performance Across Cutoffs\nMean EF@1% (n=5 seeds)',
                 fontsize=16, fontweight='bold', pad=20)
    
    # Update labels
    cutoff_labels_for_plot = [CUTOFF_LABELS[CUTOFFS.index(c)] for c in available_cutoffs]
    ax.set_xticklabels(cutoff_labels_for_plot, rotation=0, fontsize=11)
    ax.set_yticklabels([CONFIG_LABELS[ct] for ct in available_configs], rotation=0, fontsize=11)
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(output_dir, 'phase2_enrichment_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logging.info(f"Saved enrichment heatmap to: {output_path}")
    
    output_path_pdf = os.path.join(output_dir, 'phase2_enrichment_heatmap.pdf')
    plt.savefig(output_path_pdf, bbox_inches='tight')
    
    plt.close()


def plot_active_count_vs_performance(df_results, output_dir):
    """Plot 3: Active count vs performance trade-off.
    
    Scatter plot showing EF@1% vs number of actives retained at each cutoff.
    Points are sized by cutoff value, lines connect points for same config.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Calculate means for each config-cutoff combination
    summary = df_results.groupby(['config_type', 'cutoff_nM']).agg({
        'ef_1_pct': 'mean',
        'num_actives': 'mean'
    }).reset_index()
    
    # Plot for each configuration
    for config_type in CONFIG_TYPES:
        df_config = summary[summary['config_type'] == config_type].sort_values('cutoff_nM')
        
        if df_config.empty:
            continue
        
        # Plot line connecting points
        ax.plot(
            df_config['num_actives'],
            df_config['ef_1_pct'],
            color=CONFIG_COLORS[config_type],
            linestyle=CONFIG_LINESTYLES[config_type],
            linewidth=2,
            alpha=0.6,
            zorder=1
        )
        
        # Plot scatter points sized by cutoff
        for _, row in df_config.iterrows():
            # Size points by cutoff (larger cutoff = larger point)
            size = 100 + (row['cutoff_nM'] / 1000)  # Scale for visibility
            
            ax.scatter(
                row['num_actives'],
                row['ef_1_pct'],
                s=size,
                color=CONFIG_COLORS[config_type],
                edgecolors='black',
                linewidths=1.5,
                alpha=0.8,
                zorder=2,
                label=CONFIG_LABELS[config_type] if row['cutoff_nM'] == CUTOFFS[0] else ""
            )
            
            # Annotate with cutoff value
            cutoff_label = next(cl for c, cl in zip(CUTOFFS, CUTOFF_LABELS) if c == row['cutoff_nM'])
            ax.annotate(
                cutoff_label,
                (row['num_actives'], row['ef_1_pct']),
                xytext=(8, 8),
                textcoords='offset points',
                fontsize=9,
                alpha=0.7
            )
    
    # Formatting
    ax.set_xlabel('Number of Actives Retained', fontsize=14, fontweight='bold')
    ax.set_ylabel('Enrichment Factor @ 1%', fontsize=14, fontweight='bold')
    ax.set_title('Phase 2: Quality vs Quantity Trade-off\nEF@1% vs Active Count Across Cutoffs',
                 fontsize=16, fontweight='bold', pad=20)
    
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Legend
    handles, labels = ax.get_legend_handles_labels()
    # Remove duplicates
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=11, loc='best', frameon=True, shadow=True)
    
    # Add size legend for cutoff values
    size_legend_elements = [
        plt.scatter([], [], s=100 + (c / 1000), color='gray', edgecolors='black', 
                   linewidths=1.5, alpha=0.6, label=cl)
        for c, cl in zip(CUTOFFS, CUTOFF_LABELS)
    ]
    legend2 = ax.legend(
        handles=size_legend_elements,
        title='Cutoff',
        loc='lower right',
        fontsize=10,
        frameon=True,
        shadow=True
    )
    ax.add_artist(legend2)
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(output_dir, 'phase2_quality_vs_quantity.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logging.info(f"Saved quality vs quantity plot to: {output_path}")
    
    output_path_pdf = os.path.join(output_dir, 'phase2_quality_vs_quantity.pdf')
    plt.savefig(output_path_pdf, bbox_inches='tight')
    
    plt.close()


def generate_summary_report(df_results, output_dir):
    """Generate text summary of Phase 2 results."""
    
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("PHASE 2: AFFINITY CUTOFF SENSITIVITY ANALYSIS - SUMMARY REPORT")
    report_lines.append("="*80)
    report_lines.append("")
    
    # Overall statistics
    report_lines.append(f"Total runs analyzed: {len(df_results)}")
    report_lines.append(f"Configurations tested: {len(CONFIG_TYPES)}")
    report_lines.append(f"Cutoffs tested: {len(CUTOFFS)}")
    report_lines.append(f"Seeds per config-cutoff: {df_results.groupby(['config_type', 'cutoff_nM']).size().max()}")
    report_lines.append("")
    
    # Best performance by cutoff
    report_lines.append("-"*80)
    report_lines.append("BEST CONFIGURATION AT EACH CUTOFF (by mean EF@1%)")
    report_lines.append("-"*80)
    
    for cutoff, label in zip(CUTOFFS, CUTOFF_LABELS):
        df_cutoff = df_results[df_results['cutoff_nM'] == cutoff]
        if df_cutoff.empty:
            report_lines.append(f"{label:12s}: No data available")
            continue
        
        grouped = df_cutoff.groupby('config_type')['ef_1_pct'].mean()
        if grouped.isna().all():
            report_lines.append(f"{label:12s}: All EF values are NaN (jobs may have failed)")
            continue
        
        best_config = grouped.idxmax()
        best_ef = grouped.max()
        best_std = df_cutoff[df_cutoff['config_type'] == best_config]['ef_1_pct'].std()
        
        if pd.isna(best_config) or pd.isna(best_ef):
            report_lines.append(f"{label:12s}: Unable to determine best configuration")
            continue
        
        report_lines.append(f"{label:12s}: {CONFIG_LABELS[best_config]:40s} EF@1% = {best_ef:.2f} ± {best_std:.2f}")
    
    report_lines.append("")
    
    # Performance by configuration
    report_lines.append("-"*80)
    report_lines.append("PERFORMANCE BY CONFIGURATION (across all cutoffs)")
    report_lines.append("-"*80)
    
    for config_type in CONFIG_TYPES:
        df_config = df_results[df_results['config_type'] == config_type]
        mean_ef = df_config['ef_1_pct'].mean()
        std_ef = df_config['ef_1_pct'].std()
        
        report_lines.append(f"{CONFIG_LABELS[config_type]:40s}: Mean EF@1% = {mean_ef:.2f} ± {std_ef:.2f}")
    
    report_lines.append("")
    
    # Detailed table
    report_lines.append("-"*80)
    report_lines.append("DETAILED RESULTS (mean ± std across seeds)")
    report_lines.append("-"*80)
    
    summary = df_results.groupby(['config_type', 'cutoff_nM']).agg({
        'ef_1_pct': ['mean', 'std'],
        'roc_auc': ['mean', 'std'],
        'num_actives': 'mean'
    }).round(2)
    
    report_lines.append(summary.to_string())
    report_lines.append("")
    
    # Save report
    report_path = os.path.join(output_dir, 'phase2_summary_report.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    logging.info(f"Saved summary report to: {report_path}")
    
    # Also print to console
    print('\n'.join(report_lines))


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2 Cutoff Sensitivity Analysis and Visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--phase2_workspace',
        type=str,
        default='experiment_workspace_v3_phase2',
        help='Path to Phase 2 workspace directory'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='phase2_analysis_results',
        help='Output directory for plots and reports'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.phase2_workspace):
        logging.error(f"Phase 2 workspace not found: {args.phase2_workspace}")
        return 1
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    logging.info(f"Output directory: {args.output_dir}")
    
    # Find Phase 2 runs
    logging.info("Searching for Phase 2 run directories...")
    phase2_runs = find_phase2_runs(args.phase2_workspace)
    
    # Aggregate results
    logging.info("Aggregating metrics from Phase 2 runs...")
    df_results = aggregate_results(phase2_runs)
    
    if df_results.empty:
        logging.error("No results found! Make sure Phase 2 experiments have completed.")
        return 1
    
    # Save aggregated data
    results_csv = os.path.join(args.output_dir, 'phase2_aggregated_results.csv')
    df_results.to_csv(results_csv, index=False)
    logging.info(f"Saved aggregated results to: {results_csv}")
    
    # Generate plots
    logging.info("Generating Plot 1: Cutoff sensitivity curves...")
    plot_cutoff_sensitivity_curves(df_results, args.output_dir)
    
    logging.info("Generating Plot 2: Potency-stratified enrichment heatmap...")
    plot_potency_stratified_heatmap(df_results, args.output_dir)
    
    logging.info("Generating Plot 3: Active count vs performance trade-off...")
    plot_active_count_vs_performance(df_results, args.output_dir)
    
    # Generate summary report
    logging.info("Generating summary report...")
    generate_summary_report(df_results, args.output_dir)
    
    logging.info("="*80)
    logging.info("PHASE 2 ANALYSIS COMPLETE")
    logging.info("="*80)
    logging.info(f"Results saved to: {args.output_dir}")
    logging.info("Generated files:")
    logging.info("  - phase2_cutoff_sensitivity_curves.png/pdf")
    logging.info("  - phase2_enrichment_heatmap.png/pdf")
    logging.info("  - phase2_quality_vs_quantity.png/pdf")
    logging.info("  - phase2_aggregated_results.csv")
    logging.info("  - phase2_summary_report.txt")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
