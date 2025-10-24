#!/usr/bin/env python3
"""
Phase 2 Cutoff Sensitivity Analysis and Visualization

This script analyzes the results of Phase 2 affinity cutoff sensitivity experiments
and generates publication-ready figures showing:

1. Cutoff sensitivity curves (EF@1% vs cutoff)
2. Potency-stratified enrichment heatmap
3. Active count vs performance trade-off
4. Potency tier distribution by DR method

Configurations analyzed:
- PCA/features/5D (baseline)
- UMAP/features/5D (optimized for overall EF)

Note: Phase 1 analysis showed that UMAP configs optimized for overall EF and 
high-potency EF yield identical hyperparameters, so only one UMAP variant is tested.

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
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)-8s - %(message)s'
)

# Plot styling
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Constants
CUTOFFS = [100, 1000, 10000, 100000]  # nM
CUTOFF_LABELS = ['100 nM', '1 μM', '10 μM', '100 μM']
CONFIG_TYPES = ['pca_overall', 'umap_overall']
CONFIG_LABELS = {
    'pca_overall': 'PCA/features/5D',
    'umap_overall': 'UMAP/features/5D'
}
CONFIG_COLORS = {
    'pca_overall': '#1f77b4',  # blue
    'umap_overall': '#ff7f0e'  # orange
}
CONFIG_LINESTYLES = {
    'pca_overall': '-',
    'umap_overall': '--'
}


def find_phase2_runs(workspace_dir):
    """Find all Phase 2 run directories organized by config type and cutoff.
    
    Returns:
        dict: {config_type: {cutoff: [run_dirs]}}
    """
    runs = {cfg: {cutoff: [] for cutoff in CUTOFFS} for cfg in CONFIG_TYPES}
    
    # More flexible pattern to catch both PCA and UMAP directories
    # PCA: run_seed42_features_pca_dim5_cutoff100nM_pca_overall
    # UMAP: run_seed42_features_umap_euclidean_dim5_nn5_md0.0_cutoff100nM_umap_overall
    pattern = os.path.join(workspace_dir, "run_seed*cutoff*nM_*")
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
            logging.debug(f"  ✓ Matched PCA: {basename}")
        elif 'umap' in basename.lower():  # Case insensitive check
            if basename.endswith('_umap_overall'):
                config_type = 'umap_overall'
                logging.debug(f"  ✓ Matched UMAP overall: {basename}")
            elif basename.endswith('_umap_high_potency'):
                # Skip high_potency configs - not needed (same as overall)
                logging.debug(f"  ⊘ Skipping UMAP high-potency (same as overall): {basename}")
                continue
            else:
                logging.warning(f"UMAP directory doesn't match expected suffix: {basename}")
                logging.warning(f"  - Basename ends with: ...{basename[-30:]}")
        
        if config_type is None:
            logging.warning(f"Could not extract config type from: {basename}")
            logging.warning(f"  - Contains 'umap': {'umap' in basename}")
            logging.warning(f"  - Ends with '_umap_overall': {basename.endswith('_umap_overall')}")
            logging.warning(f"  - Ends with '_umap_high_potency': {basename.endswith('_umap_high_potency')}")
            logging.warning(f"  - Contains '_pca_': {'_pca_' in basename}")
            logging.warning(f"  - Ends with '_pca_overall': {basename.endswith('_pca_overall')}")
            continue
        
        runs[config_type][cutoff].append(run_dir)
        logging.debug(f"  → Added to {config_type} @ {cutoff}nM")
    
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
        logging.debug(f"  ✗ No metrics found using pattern: {pattern}")
        # Try to list what actually exists
        base_path = os.path.join(run_dir, "TyrosineProteinKinaseABL1_P00519", "results", "features", "dim_5")
        if os.path.exists(base_path):
            subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
            logging.debug(f"    Available subdirs in dim_5/: {subdirs}")
        else:
            logging.debug(f"    Path does not exist: {base_path}")
        return None
    
    if len(csv_files) > 1:
        logging.debug(f"  ! Multiple metrics files found in {run_dir}, using first")
    
    logging.debug(f"  ✓ Found metrics file: {csv_files[0]}")
    
    try:
        df = pd.read_csv(csv_files[0])
        if df.empty:
            logging.debug(f"  ✗ Metrics file is empty")
            return None
        
        metrics = df.iloc[0].to_dict()
        logging.debug(f"  ✓ Loaded metrics: EF@1% = {metrics.get('ef_1%', 'N/A')}")
        return metrics
    
    except Exception as e:
        logging.error(f"  ✗ Error loading metrics from {csv_files[0]}: {e}")
        return None


def load_actives_with_affinity(run_dir):
    """Load active compounds with their affinity data to enable potency stratification.
    
    Returns:
        pd.DataFrame: Actives with 'Standard Value (nM)' column, or None if not found
    """
    # Look for the complete actives file that includes affinity data
    pattern = os.path.join(
        run_dir,
        "TyrosineProteinKinaseABL1_P00519",
        "results",
        "features",
        "dim_5",
        "**",
        "*_actives_complete_*.csv"
    )
    
    csv_files = glob.glob(pattern, recursive=True)
    
    if not csv_files:
        logging.debug(f"  ✗ No actives complete file found in: {run_dir}")
        return None
    
    try:
        df = pd.read_csv(csv_files[0])
        if df.empty or 'Standard Value (nM)' not in df.columns:
            logging.debug(f"  ✗ Actives file missing affinity data")
            return None
        
        logging.debug(f"  ✓ Loaded {len(df)} actives with affinity data")
        return df
    
    except Exception as e:
        logging.warning(f"  ✗ Error loading actives from {csv_files[0]}: {e}")
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
                
                # Load actives with affinity for potency stratification
                df_actives = load_actives_with_affinity(run_dir)
                num_high_potency = num_medium_potency = num_low_potency = 0
                
                if df_actives is not None and 'Standard Value (nM)' in df_actives.columns:
                    affinity_vals = pd.to_numeric(df_actives['Standard Value (nM)'], errors='coerce')
                    num_high_potency = len(affinity_vals[(affinity_vals >= 0.1) & (affinity_vals <= 100)])
                    num_medium_potency = len(affinity_vals[(affinity_vals > 100) & (affinity_vals <= 1000)])
                    num_low_potency = len(affinity_vals[(affinity_vals > 1000) & (affinity_vals <= 10000)])
                
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
                    'num_total': num_total,
                    'num_high_potency': num_high_potency,
                    'num_medium_potency': num_medium_potency,
                    'num_low_potency': num_low_potency
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


def calculate_tier_specific_enrichment(run_dir):
    """Calculate EF@1% separately for each potency tier.
    
    Returns:
        dict: {tier: ef_1_pct} for 'high', 'medium', 'low' potency actives
    """
    # Load the ranked output file
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
        return {'high': np.nan, 'medium': np.nan, 'low': np.nan}
    
    try:
        df_ranked = pd.read_csv(csv_files[0])
        if df_ranked.empty or 'TYPE' not in df_ranked.columns:
            return {'high': np.nan, 'medium': np.nan, 'low': np.nan}
    except Exception as e:
        logging.debug(f"Error loading ranked data: {e}")
        return {'high': np.nan, 'medium': np.nan, 'low': np.nan}
    
    # Load actives with affinity data
    pattern_actives = os.path.join(
        run_dir,
        "TyrosineProteinKinaseABL1_P00519",
        "results",
        "features",
        "dim_5",
        "**",
        "*_actives_complete_*.csv"
    )
    
    csv_actives = glob.glob(pattern_actives, recursive=True)
    if not csv_actives:
        return {'high': np.nan, 'medium': np.nan, 'low': np.nan}
    
    try:
        df_actives = pd.read_csv(csv_actives[0])
        if 'Standard Value (nM)' not in df_actives.columns:
            return {'high': np.nan, 'medium': np.nan, 'low': np.nan}
    except Exception as e:
        logging.debug(f"Error loading actives: {e}")
        return {'high': np.nan, 'medium': np.nan, 'low': np.nan}
    
    # Merge affinity data into ranked data (use MOLECULE ID or SMILES)
    merge_col = None
    for col in ['MOLECULE ID', 'Compound ChEMBL ID', 'SMILES']:
        if col in df_ranked.columns and col in df_actives.columns:
            merge_col = col
            break
    
    if not merge_col:
        return {'high': np.nan, 'medium': np.nan, 'low': np.nan}
    
    df_ranked = df_ranked.merge(
        df_actives[[merge_col, 'Standard Value (nM)']],
        on=merge_col,
        how='left'
    )
    
    # Assign potency tiers
    affinity = pd.to_numeric(df_ranked['Standard Value (nM)'], errors='coerce')
    df_ranked['potency_tier'] = 'none'
    df_ranked.loc[(affinity >= 0.1) & (affinity <= 100), 'potency_tier'] = 'high'
    df_ranked.loc[(affinity > 100) & (affinity <= 1000), 'potency_tier'] = 'medium'
    df_ranked.loc[(affinity > 1000) & (affinity <= 10000), 'potency_tier'] = 'low'
    
    # Calculate EF@1% for each tier
    results = {}
    total_compounds = len(df_ranked)
    top_1_pct = max(1, int(0.01 * total_compounds))
    
    for tier in ['high', 'medium', 'low']:
        # Count tier actives in full set
        tier_actives_total = len(df_ranked[df_ranked['potency_tier'] == tier])
        
        if tier_actives_total == 0:
            results[tier] = 0.0
            continue
        
        # Count tier actives in top 1%
        df_top = df_ranked.head(top_1_pct)
        tier_actives_top = len(df_top[df_top['potency_tier'] == tier])
        
        # Calculate EF@1%
        # EF = (tier_actives_top / top_1_pct) / (tier_actives_total / total_compounds)
        enrichment = (tier_actives_top / top_1_pct) / (tier_actives_total / total_compounds)
        results[tier] = enrichment
    
    return results


def aggregate_tier_enrichments(phase2_runs):
    """Aggregate tier-specific enrichments across all Phase 2 runs."""
    results = []
    
    for config_type in CONFIG_TYPES:
        for cutoff in CUTOFFS:
            run_dirs = phase2_runs[config_type][cutoff]
            
            for run_dir in run_dirs:
                # Extract seed
                basename = os.path.basename(run_dir)
                seed = None
                for s in [42, 43, 44, 45, 46]:
                    if f"seed{s}_" in basename:
                        seed = s
                        break
                
                if seed is None:
                    continue
                
                # Calculate tier-specific enrichments
                tier_efs = calculate_tier_specific_enrichment(run_dir)
                
                # Build result row
                row = {
                    'config_type': config_type,
                    'cutoff_nM': cutoff,
                    'seed': seed,
                    'ef_1_pct_high': tier_efs['high'],
                    'ef_1_pct_medium': tier_efs['medium'],
                    'ef_1_pct_low': tier_efs['low']
                }
                
                results.append(row)
    
    df = pd.DataFrame(results)
    logging.info(f"Aggregated tier-specific enrichments for {len(df)} runs")
    
    return df


def plot_tier_specific_enrichment_curves(phase2_runs, output_dir):
    """Plot 4: Tier-specific enrichment across cutoffs.
    
    Multi-line plot showing how EF@1% for each potency tier changes
    as affinity cutoff increases. Separate subplots for PCA vs UMAP.
    """
    # Calculate tier-specific enrichments
    logging.info("Calculating tier-specific enrichments (this may take a moment)...")
    df_tier_efs = aggregate_tier_enrichments(phase2_runs)
    
    if df_tier_efs.empty:
        logging.warning("No tier-specific enrichment data available")
        return
    
    # Save tier enrichment data
    tier_csv = os.path.join(output_dir, 'phase2_tier_specific_enrichments.csv')
    df_tier_efs.to_csv(tier_csv, index=False)
    logging.info(f"Saved tier-specific enrichments to: {tier_csv}")
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    
    config_labels = {'pca_overall': 'PCA', 'umap_overall': 'UMAP'}
    tier_colors = {'high': '#2ecc71', 'medium': '#f39c12', 'low': '#e74c3c'}
    tier_labels = {'high': 'High (0.1-100 nM)', 'medium': 'Medium (100-1K nM)', 'low': 'Low (1K-10K nM)'}
    
    for idx, config_type in enumerate(CONFIG_TYPES):
        ax = axes[idx]
        df_config = df_tier_efs[df_tier_efs['config_type'] == config_type]
        
        if df_config.empty:
            ax.text(0.5, 0.5, f'No data for {config_labels[config_type]}', 
                   ha='center', va='center', fontsize=14, transform=ax.transAxes)
            continue
        
        # Plot lines for each potency tier
        for tier in ['high', 'medium', 'low']:
            means = []
            stds = []
            cutoff_values = []
            
            for cutoff in CUTOFFS:
                df_cutoff = df_config[df_config['cutoff_nM'] == cutoff]
                
                if len(df_cutoff) > 0:
                    ef_values = df_cutoff[f'ef_1_pct_{tier}']
                    # Filter out NaN and inf values
                    ef_values = ef_values[np.isfinite(ef_values)]
                    
                    if len(ef_values) > 0:
                        mean_ef = ef_values.mean()
                        std_ef = ef_values.std()
                        
                        means.append(mean_ef)
                        stds.append(std_ef)
                        cutoff_values.append(cutoff)
            
            if means:
                # Plot line with error bars
                ax.errorbar(
                    cutoff_values,
                    means,
                    yerr=stds,
                    label=tier_labels[tier],
                    color=tier_colors[tier],
                    linewidth=2.5,
                    marker='o',
                    markersize=8,
                    capsize=5,
                    capthick=2,
                    alpha=0.9
                )
        
        # Formatting
        ax.set_xscale('log')
        ax.set_xlabel('Affinity Cutoff (nM)', fontsize=13, fontweight='bold')
        if idx == 0:
            ax.set_ylabel('Enrichment Factor @ 1%\n(Tier-Specific)', fontsize=13, fontweight='bold')
        ax.set_title(f'{config_labels[config_type]} Method', fontsize=15, fontweight='bold', pad=15)
        
        ax.set_xticks(CUTOFFS)
        ax.set_xticklabels(CUTOFF_LABELS, fontsize=11)
        ax.tick_params(labelsize=11)
        
        ax.legend(fontsize=11, loc='best', frameon=True, shadow=True, fancybox=True)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add y=0 reference line
        ax.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    plt.suptitle('Phase 2: Tier-Specific Enrichment Across Cutoffs\nHow EF@1% Changes for Each Potency Tier (n=5 seeds)', 
                fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(output_dir, 'phase2_tier_specific_enrichment.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logging.info(f"Saved tier-specific enrichment plot to: {output_path}")
    
    output_path_pdf = os.path.join(output_dir, 'phase2_tier_specific_enrichment.pdf')
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
    
    logging.info("Generating Plot 4: Tier-specific enrichment curves...")
    plot_tier_specific_enrichment_curves(phase2_runs, args.output_dir)
    
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
    logging.info("  - phase2_tier_specific_enrichment.png/pdf")
    logging.info("  - phase2_aggregated_results.csv")
    logging.info("  - phase2_tier_specific_enrichments.csv")
    logging.info("  - phase2_summary_report.txt")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
