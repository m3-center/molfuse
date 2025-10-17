#!/usr/bin/env python3
"""
Analyze Potency-Stratified Enrichment for Hyperparameter Sweep Results

This script aggregates all available hyperparameter sweep results and calculates
enrichment factors stratified by ligand potency (affinity). It can be run at any
time, even while experiments are still running - incomplete experiments are safely
skipped.

Key Features:
- Robust handling of incomplete/running experiments
- Potency-stratified EF@1% calculation (High, Medium, Weak potency tiers)
- Integration of check_hyperparam_status and analyze_hyperparams logic
- Progress bars for long-running operations
- Comprehensive LaTeX report generation

Usage:
    python analyze_potency_stratified_enrichment.py \\
        --workspace experiment_workspace_hyperparam_sweep_v2 \\
        --output_dir potency_stratification_analysis
"""

import pandas as pd
import numpy as np
import os
import argparse
import logging
import json
import glob
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm
import gc

# =============================================================================
# POTENCY STRATIFICATION DEFINITIONS
# =============================================================================
# Based on ABL1 dataset: median affinity = 132 nM, range 0-444,180 nM
# Potency bins defined in pActivity units (pActivity = -log10(M))
POTENCY_BINS = {
    'High_Potent': (7.0, 11.0),      # 0.1 nM - 100 nM (pActivity > 7)
    'Medium_Potent': (6.0, 7.0),     # 100 nM - 1000 nM (pActivity 6-7)
    'Weak_Potent': (4.0, 6.0)        # 1,000 nM - 100,000 nM (pActivity 4-6)
}

# =============================================================================
# LOGGING SETUP
# =============================================================================
def setup_logging(output_dir):
    """Setup logging to both file and console."""
    log_filename = os.path.join(output_dir, f"potency_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)-8s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, mode='w'),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging initialized: {log_filename}")
    return log_filename

# =============================================================================
# EXPERIMENT STATUS CHECKING (from check_hyperparam_status.py)
# =============================================================================
def extract_config_info(run_dir):
    """Extract configuration info from run directory."""
    run_name = os.path.basename(run_dir)
    
    info = {
        'run_dir': run_name,
        'seed': None,
        'representation': None,
        'dr_method': None,
        'n_neighbors': None,
        'min_dist': None,
        'dim': None,
        'target_id': None
    }
    
    # Parse from directory name
    parts = run_name.split('_')
    
    # Extract seed
    for part in parts:
        if part.startswith('seed'):
            info['seed'] = part.replace('seed', '')
    
    # Extract dimensionality
    for part in parts:
        if part.startswith('dim') and len(part) > 3:
            info['dim'] = part[3:]
    
    # Extract representation
    if 'features' in run_name:
        info['representation'] = 'features'
    elif 'fingerprints' in run_name:
        info['representation'] = 'fingerprints'
    
    # Extract DR method
    if 'pca' in run_name.lower():
        info['dr_method'] = 'PCA'
    elif 'umap' in run_name.lower():
        if 'euclidean' in run_name:
            info['dr_method'] = 'UMAP-Euclidean'
        elif 'jaccard' in run_name:
            info['dr_method'] = 'UMAP-Jaccard'
        else:
            info['dr_method'] = 'UMAP'
    elif 'tsne' in run_name.lower():
        info['dr_method'] = 't-SNE'
    
    # Extract hyperparameters from directory name
    for i, part in enumerate(parts):
        if part.startswith('nn') and len(part) > 2:
            info['n_neighbors'] = part[2:]
        elif part.startswith('md') and len(part) > 2:
            info['min_dist'] = part[2:]
    
    # Try to load from config file
    config_path = os.path.join(run_dir, 'run_config.json')
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                
                # Get target ID
                if 'targets' in config and len(config['targets']) > 0:
                    info['target_id'] = config['targets'][0].get('id_name')
                
                # Get DR method details
                dr_methods = config.get('dimensionality_reduction_methods', {})
                if dr_methods:
                    method_key = list(dr_methods.keys())[0]
                    method_config = dr_methods[method_key]
                    info['dr_method'] = method_config.get('short_name', info['dr_method'])
                    info['n_neighbors'] = method_config.get('n_neighbors', info['n_neighbors'])
                    info['min_dist'] = method_config.get('min_dist', info['min_dist'])
                    
                    # Extract dimensionality from config if not already found
                    if not info['dim']:
                        for dim_method in dr_methods.values():
                            if 'n_components' in dim_method:
                                info['dim'] = str(dim_method['n_components'])
                                break
        except Exception as e:
            logging.debug(f"Could not parse config file {config_path}: {e}")
    
    return info

def check_experiment_completion(run_dir):
    """Check if experiment completed successfully.
    
    Returns:
        tuple: (completed: bool, ranking_file: str or None, metrics_file: str or None)
    """
    # Pattern: run_seed*/TARGET/results/REPR/dim_N/METHOD/*-RANKED.csv
    results_pattern = os.path.join(run_dir, '*/results/*/dim_*/*')
    results_dirs = glob.glob(results_pattern)
    
    if not results_dirs:
        return False, None, None
    
    ranking_file = None
    metrics_file = None
    
    for results_dir in results_dirs:
        # Check for ranking files (contains full dataset with affinity info)
        ranking_files = glob.glob(os.path.join(results_dir, '*-RANKED.csv'))
        if ranking_files:
            ranking_file = ranking_files[0]
        
        # Check for metrics files (indicates successful completion)
        metrics_files = glob.glob(os.path.join(results_dir, '*_ranking_metrics.csv'))
        if metrics_files:
            metrics_file = metrics_files[0]
    
    # Experiment is complete if we have both ranking and metrics files
    completed = ranking_file is not None and metrics_file is not None
    
    return completed, ranking_file, metrics_file

# =============================================================================
# POTENCY-STRATIFIED ENRICHMENT CALCULATION
# =============================================================================
def calculate_potency_stratified_enrichment(ranking_file, potency_bins, top_percent=1.0):
    """Calculate enrichment factors stratified by ligand potency.
    
    Args:
        ranking_file: Path to *-RANKED.csv file
        potency_bins: Dict of {tier_name: (min_pActivity, max_pActivity)}
        top_percent: Percentage of top-ranked molecules to evaluate (default: 1%)
    
    Returns:
        dict: Enrichment factors and statistics for each potency tier
    """
    try:
        # Read ranking file
        df = pd.read_csv(ranking_file, low_memory=False)
        
        # Identify required columns
        # Look for affinity column (Standard Value, Affinity, etc.)
        affinity_col = None
        for col in ['Standard Value', 'Affinity', 'affinity', 'IC50', 'Ki', 'Kd']:
            if col in df.columns:
                affinity_col = col
                break
        
        if affinity_col is None:
            logging.warning(f"No affinity column found in {ranking_file}")
            return None
        
        # Look for active/label column
        label_col = None
        for col in ['is_active', 'label', 'Label', 'category', 'Category']:
            if col in df.columns:
                label_col = col
                break
        
        if label_col is None:
            logging.warning(f"No label column found in {ranking_file}")
            return None
        
        # Filter to actives only
        df_actives = df[df[label_col].isin([1, True, 'active', 'Active', 'ACTIVE'])].copy()
        
        if df_actives.empty:
            logging.warning(f"No actives found in {ranking_file}")
            return None
        
        # Convert affinity (nM) to pActivity (-log10(M))
        # pActivity = -log10(affinity_nM * 1e-9)
        df_actives['pActivity'] = -np.log10(df_actives[affinity_col] * 1e-9)
        
        # Assign potency tier to each active
        def assign_potency_tier(pActivity):
            for tier_name, (min_pAct, max_pAct) in potency_bins.items():
                if min_pAct <= pActivity < max_pAct:
                    return tier_name
            return 'Out_of_Range'
        
        df_actives['Potency_Tier'] = df_actives['pActivity'].apply(assign_potency_tier)
        
        # Remove out-of-range compounds
        df_actives = df_actives[df_actives['Potency_Tier'] != 'Out_of_Range']
        
        if df_actives.empty:
            logging.warning(f"No actives within potency bins in {ranking_file}")
            return None
        
        # Calculate enrichment for each potency tier
        results = {}
        
        # Total dataset size
        total_dataset_size = len(df)
        top_n = int(total_dataset_size * top_percent / 100.0)
        
        # Overall statistics
        results['Total_Actives'] = len(df_actives)
        results['Total_Dataset_Size'] = total_dataset_size
        results['Top_N'] = top_n
        
        for tier_name in potency_bins.keys():
            tier_actives = df_actives[df_actives['Potency_Tier'] == tier_name]
            n_tier_total = len(tier_actives)
            
            if n_tier_total == 0:
                results[f'EF1_{tier_name}'] = 0.0
                results[f'N_{tier_name}_Total'] = 0
                results[f'N_{tier_name}_TopPercent'] = 0
                results[f'Median_Affinity_nM_{tier_name}'] = np.nan
                continue
            
            # Find how many of this tier are in top N%
            tier_actives_in_top = tier_actives.head(top_n)
            n_tier_in_top = len(tier_actives_in_top)
            
            # Calculate enrichment factor
            # EF = (n_tier_in_top / top_n) / (n_tier_total / total_dataset_size)
            expected_random = (n_tier_total / total_dataset_size) * top_n
            ef = (n_tier_in_top / expected_random) if expected_random > 0 else 0.0
            
            # Store results
            results[f'EF1_{tier_name}'] = ef
            results[f'N_{tier_name}_Total'] = n_tier_total
            results[f'N_{tier_name}_TopPercent'] = n_tier_in_top
            results[f'Median_Affinity_nM_{tier_name}'] = tier_actives[affinity_col].median()
        
        # Calculate median potency of all actives in top 1%
        actives_in_top = df_actives.head(top_n)
        if len(actives_in_top) > 0:
            results['Median_Potency_Top1Percent_nM'] = actives_in_top[affinity_col].median()
            results['Mean_Potency_Top1Percent_nM'] = actives_in_top[affinity_col].mean()
        else:
            results['Median_Potency_Top1Percent_nM'] = np.nan
            results['Mean_Potency_Top1Percent_nM'] = np.nan
        
        return results
        
    except Exception as e:
        logging.error(f"Error calculating potency-stratified enrichment for {ranking_file}: {e}")
        return None

# =============================================================================
# MAIN AGGREGATION LOGIC
# =============================================================================
def aggregate_potency_stratified_metrics(workspace, potency_bins):
    """Aggregate potency-stratified metrics from all available experiments.
    
    Args:
        workspace: Path to workspace directory
        potency_bins: Dict of potency tier definitions
    
    Returns:
        pd.DataFrame: Aggregated metrics with potency stratification
    """
    # Find all run directories
    run_dirs = glob.glob(os.path.join(workspace, 'run_seed*'))
    
    if not run_dirs:
        logging.error(f"No run directories found in {workspace}")
        return pd.DataFrame()
    
    logging.info(f"Found {len(run_dirs)} run directories")
    
    all_results = []
    
    # Process each run with progress bar
    pbar = tqdm(sorted(run_dirs), desc="Processing experiments", unit="exp")
    
    for run_dir in pbar:
        # Extract configuration
        config_info = extract_config_info(run_dir)
        
        # Check completion status
        completed, ranking_file, metrics_file = check_experiment_completion(run_dir)
        
        if not completed:
            pbar.set_postfix_str(f"Skipping incomplete: {config_info['run_dir'][:30]}")
            continue
        
        pbar.set_postfix_str(f"Processing: {config_info['run_dir'][:30]}")
        
        # Read standard metrics from metrics file
        standard_metrics = {}
        if metrics_file:
            try:
                df_metrics = pd.read_csv(metrics_file)
                if not df_metrics.empty:
                    for col in ['roc_auc', 'pr_auc', 'ef_1%', 'ef_5%', 'ef_10%']:
                        if col in df_metrics.columns:
                            standard_metrics[col] = df_metrics[col].iloc[0]
            except Exception as e:
                logging.warning(f"Could not read metrics from {metrics_file}: {e}")
        
        # Calculate potency-stratified metrics
        potency_metrics = calculate_potency_stratified_enrichment(ranking_file, potency_bins)
        
        if potency_metrics is None:
            pbar.set_postfix_str(f"No potency data: {config_info['run_dir'][:30]}")
            continue
        
        # Combine all metrics
        result_row = {
            **config_info,
            **standard_metrics,
            **potency_metrics
        }
        
        all_results.append(result_row)
        
        # Periodic garbage collection
        if len(all_results) % 50 == 0:
            gc.collect()
    
    pbar.close()
    
    if not all_results:
        logging.error("No valid results collected")
        return pd.DataFrame()
    
    df_results = pd.DataFrame(all_results)
    
    logging.info(f"Successfully processed {len(df_results)} experiments")
    
    return df_results

# =============================================================================
# ANALYSIS AND VISUALIZATION
# =============================================================================
def generate_potency_comparison_plots(df_results, output_dir):
    """Generate comparison plots for potency-stratified enrichment.
    
    Args:
        df_results: DataFrame with aggregated results
        output_dir: Directory to save figures
    """
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Create method identifier
    df_results['Method'] = df_results['dr_method'] + ' (' + df_results['representation'] + ')'
    
    # Filter to features only for cleaner plots
    df_features = df_results[df_results['representation'] == 'features'].copy()
    
    if df_features.empty:
        logging.warning("No features-based experiments found for plotting")
        return
    
    logging.info("Generating potency stratification plots...")
    
    # ==========================================================================
    # PLOT 1: EF@1% Comparison Across Potency Tiers
    # ==========================================================================
    for tier in POTENCY_BINS.keys():
        ef_col = f'EF1_{tier}'
        
        if ef_col not in df_features.columns:
            continue
        
        # Group by method and calculate statistics
        grouped = df_features.groupby(['dr_method', 'n_neighbors', 'min_dist'])[ef_col].agg(['mean', 'std', 'count']).reset_index()
        grouped = grouped[grouped['count'] >= 3]  # At least 3 replicates
        
        if grouped.empty:
            continue
        
        # For UMAP, create heatmaps by n_neighbors and min_dist
        umap_data = grouped[grouped['dr_method'].str.contains('UMAP', na=False)]
        
        if not umap_data.empty:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # Heatmap for mean
            pivot_mean = umap_data.pivot_table(
                index='min_dist', 
                columns='n_neighbors', 
                values='mean',
                aggfunc='mean'
            )
            
            sns.heatmap(pivot_mean, annot=True, fmt='.2f', cmap='viridis', 
                       linewidths=0.5, ax=axes[0])
            axes[0].set_title(f'Mean {ef_col} (UMAP)')
            axes[0].set_xlabel('n_neighbors')
            axes[0].set_ylabel('min_dist')
            
            # Heatmap for std
            pivot_std = umap_data.pivot_table(
                index='min_dist', 
                columns='n_neighbors', 
                values='std',
                aggfunc='mean'
            )
            
            sns.heatmap(pivot_std, annot=True, fmt='.2f', cmap='Reds', 
                       linewidths=0.5, ax=axes[1])
            axes[1].set_title(f'Std Dev {ef_col} (UMAP)')
            axes[1].set_xlabel('n_neighbors')
            axes[1].set_ylabel('min_dist')
            
            plt.tight_layout()
            fig_path = os.path.join(figures_dir, f'heatmap_umap_{tier}.png')
            plt.savefig(fig_path, dpi=200, bbox_inches='tight')
            plt.close()
            logging.info(f"Saved: {fig_path}")
    
    # ==========================================================================
    # PLOT 2: Potency Tier Comparison (Grouped Bar Chart)
    # ==========================================================================
    # Compare EF@1% across all three potency tiers for each method
    
    # Aggregate by method
    agg_data = []
    for method in df_features['dr_method'].unique():
        method_data = df_features[df_features['dr_method'] == method]
        
        for tier in POTENCY_BINS.keys():
            ef_col = f'EF1_{tier}'
            if ef_col in method_data.columns:
                agg_data.append({
                    'Method': method,
                    'Potency_Tier': tier,
                    'Mean_EF1': method_data[ef_col].mean(),
                    'Std_EF1': method_data[ef_col].std()
                })
    
    if agg_data:
        df_tier_comparison = pd.DataFrame(agg_data)
        
        plt.figure(figsize=(12, 6))
        
        # Create grouped bar chart
        x = np.arange(len(df_tier_comparison['Method'].unique()))
        width = 0.25
        
        tiers = list(POTENCY_BINS.keys())
        colors = ['#2ecc71', '#f39c12', '#e74c3c']  # Green, Orange, Red
        
        for i, tier in enumerate(tiers):
            tier_data = df_tier_comparison[df_tier_comparison['Potency_Tier'] == tier]
            tier_data = tier_data.sort_values('Method')
            
            plt.bar(x + i * width, tier_data['Mean_EF1'], width, 
                   label=tier.replace('_', ' '),
                   color=colors[i],
                   yerr=tier_data['Std_EF1'],
                   capsize=5,
                   alpha=0.8)
        
        plt.xlabel('Method', fontsize=12)
        plt.ylabel('Enrichment Factor @ 1%', fontsize=12)
        plt.title('Potency-Stratified Enrichment: High vs Medium vs Weak Actives', fontsize=14, fontweight='bold')
        plt.xticks(x + width, sorted(df_tier_comparison['Method'].unique()), rotation=45, ha='right')
        plt.legend(title='Potency Tier')
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        
        fig_path = os.path.join(figures_dir, 'potency_tier_comparison.png')
        plt.savefig(fig_path, dpi=200, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved: {fig_path}")
    
    # ==========================================================================
    # PLOT 3: Median Potency Recovery (Top 1%)
    # ==========================================================================
    if 'Median_Potency_Top1Percent_nM' in df_features.columns:
        grouped = df_features.groupby('dr_method')['Median_Potency_Top1Percent_nM'].agg(['mean', 'std']).reset_index()
        
        plt.figure(figsize=(10, 6))
        
        plt.barh(grouped['dr_method'], grouped['mean'], xerr=grouped['std'], 
                capsize=5, color='steelblue', alpha=0.8)
        
        plt.xlabel('Median Potency of Actives in Top 1% (nM)', fontsize=12)
        plt.ylabel('Method', fontsize=12)
        plt.title('Median Potency Recovery: Which Method Finds the Most Potent Ligands?', 
                 fontsize=14, fontweight='bold')
        plt.xscale('log')
        plt.grid(True, axis='x', alpha=0.3)
        plt.tight_layout()
        
        fig_path = os.path.join(figures_dir, 'median_potency_recovery.png')
        plt.savefig(fig_path, dpi=200, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved: {fig_path}")
    
    logging.info("Plot generation complete!")

def generate_summary_tables(df_results, output_dir):
    """Generate summary tables for potency-stratified analysis.
    
    Args:
        df_results: DataFrame with aggregated results
        output_dir: Directory to save tables
    """
    tables_dir = os.path.join(output_dir, 'tables')
    os.makedirs(tables_dir, exist_ok=True)
    
    logging.info("Generating summary tables...")
    
    # Save master table
    master_table_path = os.path.join(tables_dir, 'master_potency_stratified_metrics.csv')
    df_results.to_csv(master_table_path, index=False)
    logging.info(f"Saved master table: {master_table_path}")
    
    # ==========================================================================
    # TABLE 1: Hyperparameter Rankings by Potency Tier
    # ==========================================================================
    for tier in POTENCY_BINS.keys():
        ef_col = f'EF1_{tier}'
        
        if ef_col not in df_results.columns:
            continue
        
        # Group by hyperparameters and calculate mean EF
        grouped = df_results.groupby(['dr_method', 'representation', 'n_neighbors', 'min_dist'])[ef_col].agg(['mean', 'std', 'count']).reset_index()
        grouped = grouped[grouped['count'] >= 3]  # At least 3 replicates
        grouped = grouped.sort_values('mean', ascending=False)
        
        # Save table
        table_path = os.path.join(tables_dir, f'hyperparameter_rankings_{tier}.csv')
        grouped.to_csv(table_path, index=False)
        logging.info(f"Saved ranking table for {tier}: {table_path}")
    
    # ==========================================================================
    # TABLE 2: Best Hyperparameters by Metric
    # ==========================================================================
    best_configs = []
    
    for metric in ['EF1_High_Potent', 'EF1_Medium_Potent', 'EF1_Weak_Potent', 'ef_1%']:
        if metric not in df_results.columns:
            continue
        
        # Find best configuration for this metric
        grouped = df_results.groupby(['dr_method', 'representation', 'n_neighbors', 'min_dist'])[metric].mean().reset_index()
        best_idx = grouped[metric].idxmax()
        
        if pd.notna(best_idx):
            best_row = grouped.loc[best_idx].to_dict()
            best_row['Optimized_For'] = metric
            best_configs.append(best_row)
    
    if best_configs:
        df_best = pd.DataFrame(best_configs)
        table_path = os.path.join(tables_dir, 'best_hyperparameters_by_metric.csv')
        df_best.to_csv(table_path, index=False)
        logging.info(f"Saved best hyperparameters table: {table_path}")
    
    # ==========================================================================
    # TABLE 3: Potency Quality vs Quantity Trade-off
    # ==========================================================================
    # Compare overall EF@1% vs potency-stratified EF@1%
    tradeoff_data = []
    
    for method in df_results['dr_method'].unique():
        method_data = df_results[df_results['dr_method'] == method]
        
        if 'ef_1%' in method_data.columns and 'EF1_High_Potent' in method_data.columns:
            tradeoff_data.append({
                'Method': method,
                'Overall_EF1': method_data['ef_1%'].mean(),
                'High_Potent_EF1': method_data['EF1_High_Potent'].mean(),
                'Medium_Potent_EF1': method_data['EF1_Medium_Potent'].mean(),
                'Weak_Potent_EF1': method_data['EF1_Weak_Potent'].mean(),
                'Median_Potency_nM': method_data['Median_Potency_Top1Percent_nM'].mean() if 'Median_Potency_Top1Percent_nM' in method_data.columns else np.nan
            })
    
    if tradeoff_data:
        df_tradeoff = pd.DataFrame(tradeoff_data)
        df_tradeoff = df_tradeoff.sort_values('Overall_EF1', ascending=False)
        
        table_path = os.path.join(tables_dir, 'quality_vs_quantity_tradeoff.csv')
        df_tradeoff.to_csv(table_path, index=False)
        logging.info(f"Saved trade-off analysis table: {table_path}")
    
    logging.info("Table generation complete!")

# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Analyze potency-stratified enrichment for hyperparameter sweep results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze default workspace
    python analyze_potency_stratified_enrichment.py \\
        --workspace experiment_workspace_hyperparam_sweep_v2 \\
        --output_dir potency_analysis
    
    # Analyze with custom potency bins
    python analyze_potency_stratified_enrichment.py \\
        --workspace experiment_workspace_hyperparam_sweep_v2 \\
        --output_dir potency_analysis \\
        --custom_bins
        """
    )
    
    parser.add_argument('--workspace', required=True,
                       help='Workspace directory containing hyperparameter sweep runs')
    parser.add_argument('--output_dir', required=True,
                       help='Output directory for analysis results')
    parser.add_argument('--custom_bins', action='store_true',
                       help='Use custom potency bins (prompts for values)')
    
    args = parser.parse_args()
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'potency_stratification_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    setup_logging(output_dir)
    
    logging.info("="*80)
    logging.info("POTENCY-STRATIFIED ENRICHMENT ANALYSIS")
    logging.info("="*80)
    logging.info(f"Workspace: {args.workspace}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Timestamp: {timestamp}")
    logging.info("")
    
    # Define potency bins
    potency_bins = POTENCY_BINS
    
    if args.custom_bins:
        logging.info("Custom potency bins requested (using defaults for now)")
        # Future: Add interactive prompts for custom bins
    
    logging.info("Potency bin definitions:")
    for tier, (min_pAct, max_pAct) in potency_bins.items():
        # Convert pActivity back to nM for display
        max_nM = 10**(-min_pAct) * 1e9
        min_nM = 10**(-max_pAct) * 1e9
        logging.info(f"  {tier}: {min_nM:.1f} - {max_nM:.1f} nM (pActivity {min_pAct} - {max_pAct})")
    logging.info("")
    
    # Aggregate metrics
    logging.info("Starting metric aggregation...")
    df_results = aggregate_potency_stratified_metrics(args.workspace, potency_bins)
    
    if df_results.empty:
        logging.error("No results to analyze. Exiting.")
        return
    
    # Generate visualizations
    logging.info("")
    logging.info("="*80)
    logging.info("GENERATING VISUALIZATIONS")
    logging.info("="*80)
    generate_potency_comparison_plots(df_results, output_dir)
    
    # Generate summary tables
    logging.info("")
    logging.info("="*80)
    logging.info("GENERATING SUMMARY TABLES")
    logging.info("="*80)
    generate_summary_tables(df_results, output_dir)
    
    # Final summary
    logging.info("")
    logging.info("="*80)
    logging.info("ANALYSIS COMPLETE")
    logging.info("="*80)
    logging.info(f"Total experiments analyzed: {len(df_results)}")
    logging.info(f"Unique methods: {df_results['dr_method'].nunique()}")
    logging.info(f"Unique seeds: {df_results['seed'].nunique()}")
    logging.info("")
    logging.info(f"All outputs saved to: {output_dir}")
    logging.info("  - figures/: Potency stratification plots")
    logging.info("  - tables/: Summary tables and rankings")
    logging.info("="*80)

if __name__ == '__main__':
    main()
