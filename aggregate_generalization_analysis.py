#!/usr/bin/env python3
"""
Aggregate and analyze results from the generalization experiment.

This script:
1. Collects performance metrics from all target proteins (including ABL1 from hyperparameter sweep)
2. Compares method performance across proteins to assess generalization
3. Generates comparative visualizations and a LaTeX report
"""

import pandas as pd
import numpy as np
import os
import argparse
import logging
import json
import glob
import re
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# --- Logging Setup ---
log_file_name = f"generalization_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)-8s - %(funcName)-25s - %(lineno)-4d - %(message)s',
    handlers=[
        logging.FileHandler(log_file_name, mode='w'),
        logging.StreamHandler()
    ]
)

def collect_metrics_from_workspace(workspace_dir, workspace_label, is_cutoff_workspace=False):
    """
    Collect metrics from a workspace directory.
    
    Args:
        workspace_dir: Path to the workspace directory
        workspace_label: Label for this workspace (e.g., 'ABL1_hyperparam' or 'Generalization')
        is_cutoff_workspace: If True, expect results_cutoff_* subdirectories
    
    Returns:
        DataFrame with metrics
    """
    all_metrics = []
    
    # Pattern to find all ranking metrics files
    if is_cutoff_workspace:
        # Pattern includes results_cutoff_* directory
        pattern = os.path.join(workspace_dir, "run_seed*", "results_cutoff_*", "*", "results", "*", "dim_*", "*", "*_ranking_metrics.csv")
    else:
        pattern = os.path.join(workspace_dir, "run_seed*", "*", "results", "*", "dim_*", "*", "*_ranking_metrics.csv")
    
    logging.info(f"Searching for metrics in: {workspace_dir}")
    logging.info(f"Pattern: {pattern}")
    
    for metrics_file in glob.glob(pattern):
        try:
            path_parts = metrics_file.split(os.sep)
            
            # Extract run directory name
            run_dir_name = next(p for p in path_parts if p.startswith("run_seed"))
            
            # Extract affinity cutoff if this is cutoff workspace
            affinity_cutoff = 100000  # Default
            if is_cutoff_workspace:
                cutoff_dir = next((p for p in path_parts if p.startswith("results_cutoff_")), None)
                if cutoff_dir:
                    cutoff_match = re.search(r"results_cutoff_(\d+)", cutoff_dir)
                    if cutoff_match:
                        affinity_cutoff = int(cutoff_match.group(1))
            
            # Extract target ID (the directory after run_seed* or results_cutoff_*)
            if is_cutoff_workspace:
                cutoff_idx = path_parts.index(next(p for p in path_parts if p.startswith("results_cutoff_")))
                target_id = path_parts[cutoff_idx + 1]
            else:
                target_idx = path_parts.index(run_dir_name) + 1
                target_id = path_parts[target_idx]
            
            # Extract representation type
            repr_type = path_parts[path_parts.index("results") + 1]
            
            # Extract strategy directory
            dim_idx = path_parts.index(next(p for p in path_parts if p.startswith("dim_")))
            strategy_dir = path_parts[dim_idx + 1]
            
            # Extract seed
            seed_match = re.search(r"run_seed(\d+)", run_dir_name)
            if not seed_match:
                logging.warning(f"Could not extract seed from: {run_dir_name}")
                continue
            seed = int(seed_match.group(1))
            
            # Get config to extract method information
            run_config_path = os.path.join(workspace_dir, run_dir_name, "run_config.json")
            if not os.path.exists(run_config_path):
                logging.warning(f"Config not found: {run_config_path}")
                continue
            
            with open(run_config_path, 'r') as f:
                run_config = json.load(f)
            
            # Extract DR method info
            dr_method_key = list(run_config['dimensionality_reduction_methods'].keys())[0]
            dr_params = run_config['dimensionality_reduction_methods'][dr_method_key]
            
            # Extract hyperparameters
            hyperparam_str = "N/A"
            if 'n_neighbors' in dr_params and 'min_dist' in dr_params:
                hyperparam_str = f"nn={dr_params['n_neighbors']}, md={dr_params['min_dist']}"
            elif 'n_neighbors' in dr_params:
                hyperparam_str = f"nn={dr_params['n_neighbors']}"
            elif 'min_dist' in dr_params:
                hyperparam_str = f"md={dr_params['min_dist']}"
            elif 'perplexity' in dr_params:
                hyperparam_str = f"perplexity={dr_params['perplexity']}"
            
            # Determine embedding strategy
            embedding_strategy = "Projection"
            if "tsne" in dr_method_key:
                embedding_strategy = "Co-embedding (Native)"
            elif strategy_dir.endswith('_Coembed'):
                embedding_strategy = "Co-embedding"
            
            # Read metrics
            df_metrics = pd.read_csv(metrics_file)
            if df_metrics.empty:
                continue
            
            # Create metric row
            metric_row = {
                'Workspace': workspace_label,
                'Target_ID': target_id,
                'Seed': seed,
                'Affinity_Cutoff': affinity_cutoff,
                'Representation': repr_type.capitalize(),
                'DR_Method': dr_params['short_name'],
                'Embedding_Strategy': embedding_strategy,
                'Hyperparameters': hyperparam_str,
                'Method_Full': f"{dr_params['short_name']} ({embedding_strategy})"
            }
            
            # Extract performance metrics
            column_map = {'roc_auc': 'ROC_AUC', 'pr_auc': 'PR_AUC', 'ef_1%': 'EF_1Perc'}
            for csv_col, df_col in column_map.items():
                if csv_col in df_metrics.columns:
                    metric_row[df_col] = df_metrics[csv_col].iloc[0] if pd.notna(df_metrics[csv_col].iloc[0]) else np.nan
                else:
                    metric_row[df_col] = np.nan
            
            all_metrics.append(metric_row)
            
        except Exception as e:
            logging.warning(f"Failed to process {metrics_file}: {e}")
            continue
    
    df = pd.DataFrame(all_metrics)
    logging.info(f"Collected {len(df)} metric rows from {workspace_label}")
    
    return df

def select_optimal_cutoff_per_method(df):
    """
    Select the optimal affinity cutoff for each DR method based on mean EF@1% 
    across all target proteins.
    
    Returns:
        DataFrame with optimal cutoff selected for each method
        Dictionary mapping method to optimal cutoff
    """
    optimal_cutoffs = {}
    optimal_data_list = []
    
    # Group by DR_Method and Affinity_Cutoff, calculate mean EF@1% across all targets
    for dr_method in df['DR_Method'].unique():
        method_data = df[df['DR_Method'] == dr_method]
        
        # Calculate mean EF@1% for each cutoff across all targets and seeds
        cutoff_performance = method_data.groupby('Affinity_Cutoff')['EF_1Perc'].mean()
        
        if not cutoff_performance.empty and not cutoff_performance.isnull().all():
            optimal_cutoff = cutoff_performance.idxmax()
            optimal_cutoffs[dr_method] = optimal_cutoff
            
            # Filter data to only this optimal cutoff for this method
            method_optimal = method_data[method_data['Affinity_Cutoff'] == optimal_cutoff].copy()
            optimal_data_list.append(method_optimal)
            
            logging.info(f"Selected optimal cutoff for {dr_method}: {optimal_cutoff} nM "
                        f"(Mean EF@1%: {cutoff_performance[optimal_cutoff]:.2f})")
    
    df_optimal = pd.concat(optimal_data_list, ignore_index=True) if optimal_data_list else pd.DataFrame()
    
    return df_optimal, optimal_cutoffs

def create_cutoff_trend_plots(df, output_dir):
    """
    Create plots showing performance vs affinity cutoff for all target proteins.
    Similar to the cutoff analysis script but showing all proteins in one view.
    """
    os.makedirs(output_dir, exist_ok=True)
    plots_created = []
    
    # Use all cutoff data including baseline
    df_cutoff = df.copy()
    
    if df_cutoff.empty:
        logging.warning("No cutoff variation data available for trend plotting")
        return plots_created
    
    # Check if we have multiple cutoffs
    unique_cutoffs = df_cutoff['Affinity_Cutoff'].unique()
    if len(unique_cutoffs) < 2:
        logging.warning(f"Only one cutoff value found: {unique_cutoffs}. Need multiple cutoffs for trend analysis.")
        return plots_created
    
    # Map target IDs to friendly names
    target_name_map = {
        'TyrosineProteinKinaseABL1_P00519': 'ABL1',
        'PyruvateKinaseM2_P14618': 'Pyruvate Kinase M2',
        'IsocitrateDehydrogenaseNADP_O75874': 'Isocitrate Dehydrogenase'
    }
    df_cutoff['Target_Name'] = df_cutoff['Target_ID'].map(target_name_map)
    
    # Create combined method+strategy identifier
    df_cutoff['Method_Strategy'] = df_cutoff['DR_Method'] + ' (' + df_cutoff['Embedding_Strategy'] + ')'
    
    # Plot for each metric
    for metric in ['EF_1Perc', 'ROC_AUC', 'PR_AUC']:
        if metric not in df_cutoff.columns or df_cutoff[metric].isnull().all():
            continue
        
        # Create faceted plot by DR method
        unique_methods = sorted(df_cutoff['DR_Method'].unique())
        n_methods = len(unique_methods)
        
        if n_methods == 0:
            continue
        
        fig, axes = plt.subplots(1, n_methods, figsize=(6*n_methods, 5), sharey=True)
        if n_methods == 1:
            axes = [axes]
        
        fig.suptitle(f'{metric} vs. Affinity Cutoff Across Target Proteins', 
                    fontsize=16, fontweight='bold')
        
        for idx, dr_method in enumerate(unique_methods):
            ax = axes[idx]
            method_data = df_cutoff[df_cutoff['DR_Method'] == dr_method]
            
            # Plot each target with a different color
            for target in method_data['Target_Name'].unique():
                target_method_data = method_data[method_data['Target_Name'] == target]
                
                # For each embedding strategy
                for strategy in target_method_data['Embedding_Strategy'].unique():
                    strategy_data = target_method_data[target_method_data['Embedding_Strategy'] == strategy]
                    
                    # Calculate mean and std by cutoff
                    mean_by_cutoff = strategy_data.groupby('Affinity_Cutoff')[metric].mean()
                    std_by_cutoff = strategy_data.groupby('Affinity_Cutoff')[metric].std()
                    
                    label = f"{target} - {strategy}"
                    ax.errorbar(mean_by_cutoff.index, mean_by_cutoff.values,
                               yerr=std_by_cutoff.values, marker='o', label=label, 
                               capsize=5, alpha=0.7)
            
            ax.set_xscale('log')
            ax.set_xlabel('Affinity Cutoff (nM)', fontsize=11, fontweight='bold')
            if idx == 0:
                ax.set_ylabel(f'Mean {metric}', fontsize=11, fontweight='bold')
            ax.set_title(dr_method, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc='best')
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'{metric}_vs_cutoff_all_targets.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plots_created.append(plot_path)
        logging.info(f"Created cutoff trend plot: {plot_path}")
    
    return plots_created

def create_comparison_plots(df, output_dir):
    """Create comparative visualizations across proteins."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Map target IDs to friendly names
    target_name_map = {
        'TyrosineProteinKinaseABL1_P00519': 'ABL1\n(Hyperparameter Tuning)',
        'PyruvateKinaseM2_P14618': 'Pyruvate Kinase M2\n(Generalization)',
        'IsocitrateDehydrogenaseNADP_O75874': 'Isocitrate Dehydrogenase\n(Generalization)'
    }
    df['Target_Name'] = df['Target_ID'].map(target_name_map)
    
    plots_created = []
    
    # 1. Method comparison across proteins (grouped bar chart)
    for metric in ['EF_1Perc', 'ROC_AUC', 'PR_AUC']:
        if metric not in df.columns or df[metric].isnull().all():
            continue
        
        # Aggregate by method and target
        df_agg = df.groupby(['Target_Name', 'Method_Full'])[metric].agg(['mean', 'std']).reset_index()
        
        # Create plot
        fig, ax = plt.subplots(figsize=(14, 8))
        
        targets = df_agg['Target_Name'].unique()
        methods = df_agg['Method_Full'].unique()
        x = np.arange(len(methods))
        width = 0.25
        
        for i, target in enumerate(targets):
            target_data = df_agg[df_agg['Target_Name'] == target]
            means = [target_data[target_data['Method_Full'] == m]['mean'].values[0] 
                    if len(target_data[target_data['Method_Full'] == m]) > 0 else 0 
                    for m in methods]
            stds = [target_data[target_data['Method_Full'] == m]['std'].values[0] 
                   if len(target_data[target_data['Method_Full'] == m]) > 0 else 0 
                   for m in methods]
            
            ax.bar(x + i*width, means, width, label=target, yerr=stds, capsize=5)
        
        ax.set_xlabel('Method', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'Mean {metric}', fontsize=12, fontweight='bold')
        ax.set_title(f'{metric} Comparison Across Target Proteins', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.legend(title='Target Protein', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'{metric}_method_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plots_created.append(plot_path)
        logging.info(f"Created plot: {plot_path}")
    
    # 2. Heatmap of method performance across proteins
    for metric in ['EF_1Perc', 'ROC_AUC', 'PR_AUC']:
        if metric not in df.columns or df[metric].isnull().all():
            continue
        
        # Create pivot table
        pivot_data = df.groupby(['Target_Name', 'Method_Full'])[metric].mean().unstack(fill_value=0)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='viridis', ax=ax, 
                   cbar_kws={'label': f'Mean {metric}'})
        ax.set_title(f'{metric} Performance Heatmap', fontsize=14, fontweight='bold')
        ax.set_xlabel('Method', fontsize=12, fontweight='bold')
        ax.set_ylabel('Target Protein', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'{metric}_heatmap.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plots_created.append(plot_path)
        logging.info(f"Created plot: {plot_path}")
    
    # 3. Consistency plot - coefficient of variation across proteins
    fig, ax = plt.subplots(figsize=(10, 6))
    
    cv_data = []
    for method in df['Method_Full'].unique():
        method_data = df[df['Method_Full'] == method].groupby('Target_Name')['EF_1Perc'].mean()
        if len(method_data) > 1:
            cv = (method_data.std() / method_data.mean()) * 100 if method_data.mean() > 0 else 0
            mean_perf = method_data.mean()
            cv_data.append({'Method': method, 'CV_%': cv, 'Mean_EF_1Perc': mean_perf})
    
    df_cv = pd.DataFrame(cv_data)
    if not df_cv.empty:
        colors = plt.cm.viridis(np.linspace(0, 1, len(df_cv)))
        ax.scatter(df_cv['Mean_EF_1Perc'], df_cv['CV_%'], s=200, c=colors, alpha=0.6, edgecolors='black')
        
        for idx, row in df_cv.iterrows():
            ax.annotate(row['Method'], (row['Mean_EF_1Perc'], row['CV_%']), 
                       fontsize=8, ha='center', va='bottom')
        
        ax.set_xlabel('Mean EF@1% Across Proteins', fontsize=12, fontweight='bold')
        ax.set_ylabel('Coefficient of Variation (%)', fontsize=12, fontweight='bold')
        ax.set_title('Method Performance vs. Consistency', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'performance_consistency.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plots_created.append(plot_path)
        logging.info(f"Created plot: {plot_path}")
    
    return plots_created

def create_summary_tables(df, output_dir):
    """Create summary tables for the report."""
    
    os.makedirs(output_dir, exist_ok=True)
    tables_created = []
    
    # Map target IDs
    target_name_map = {
        'TyrosineProteinKinaseABL1_P00519': 'ABL1 (Tuning)',
        'PyruvateKinaseM2_P14618': 'Pyruvate Kinase M2',
        'IsocitrateDehydrogenaseNADP_O75874': 'Isocitrate Dehydrogenase'
    }
    df['Target_Name'] = df['Target_ID'].map(target_name_map)
    
    # 1. Overall performance summary
    summary_data = []
    for method in df['Method_Full'].unique():
        for target in df['Target_Name'].unique():
            subset = df[(df['Method_Full'] == method) & (df['Target_Name'] == target)]
            if not subset.empty:
                row = {
                    'Method': method,
                    'Target': target,
                    'Mean_EF_1Perc': subset['EF_1Perc'].mean(),
                    'Std_EF_1Perc': subset['EF_1Perc'].std(),
                    'Mean_ROC_AUC': subset['ROC_AUC'].mean(),
                    'Mean_PR_AUC': subset['PR_AUC'].mean(),
                    'N_Replicates': len(subset)
                }
                summary_data.append(row)
    
    df_summary = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, 'performance_summary.csv')
    df_summary.to_csv(summary_path, index=False)
    tables_created.append(summary_path)
    logging.info(f"Created table: {summary_path}")
    
    # 2. Method ranking by target
    ranking_data = []
    for target in df['Target_Name'].unique():
        target_data = df[df['Target_Name'] == target].groupby('Method_Full')['EF_1Perc'].mean().sort_values(ascending=False)
        for rank, (method, ef) in enumerate(target_data.items(), 1):
            ranking_data.append({'Target': target, 'Rank': rank, 'Method': method, 'Mean_EF_1Perc': ef})
    
    df_ranking = pd.DataFrame(ranking_data)
    ranking_path = os.path.join(output_dir, 'method_rankings.csv')
    df_ranking.to_csv(ranking_path, index=False)
    tables_created.append(ranking_path)
    logging.info(f"Created table: {ranking_path}")
    
    # 3. Generalization assessment - compare ABL1 vs other proteins
    gen_data = []
    abl1_methods = df[df['Target_Name'] == 'ABL1 (Tuning)'].groupby('Method_Full')['EF_1Perc'].mean()
    
    for method in abl1_methods.index:
        abl1_perf = abl1_methods[method]
        other_perfs = df[(df['Method_Full'] == method) & (df['Target_Name'] != 'ABL1 (Tuning)')].groupby('Target_Name')['EF_1Perc'].mean()
        
        if len(other_perfs) > 0:
            gen_data.append({
                'Method': method,
                'ABL1_EF_1Perc': abl1_perf,
                'Mean_Other_EF_1Perc': other_perfs.mean(),
                'Performance_Drop_%': ((abl1_perf - other_perfs.mean()) / abl1_perf * 100) if abl1_perf > 0 else 0
            })
    
    df_gen = pd.DataFrame(gen_data)
    gen_path = os.path.join(output_dir, 'generalization_assessment.csv')
    df_gen.to_csv(gen_path, index=False)
    tables_created.append(gen_path)
    logging.info(f"Created table: {gen_path}")
    
    return tables_created

def create_cutoff_analysis_plots(df, output_dir):
    """Create plots showing how performance varies with affinity cutoff across proteins."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter to only cutoff data
    df_cutoff = df[df['Workspace'] == 'Generalization_Cutoff'].copy()
    
    if df_cutoff.empty:
        logging.warning("No cutoff data available for plotting")
        return []
    
    # Map target IDs to friendly names
    target_name_map = {
        'TyrosineProteinKinaseABL1_P00519': 'ABL1',
        'PyruvateKinaseM2_P14618': 'Pyruvate Kinase M2',
        'IsocitrateDehydrogenaseNADP_O75874': 'Isocitrate Dehydrogenase'
    }
    df_cutoff['Target_Name'] = df_cutoff['Target_ID'].map(target_name_map)
    
    plots_created = []
    
    # Plot EF@1% vs cutoff for each method, faceted by target
    for metric in ['EF_1Perc']:
        if metric not in df_cutoff.columns or df_cutoff[metric].isnull().all():
            continue
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
        fig.suptitle(f'{metric} vs. Affinity Cutoff Across Proteins (Generalization Test)', 
                    fontsize=16, fontweight='bold')
        
        for idx, target in enumerate(df_cutoff['Target_Name'].unique()):
            ax = axes[idx]
            target_data = df_cutoff[df_cutoff['Target_Name'] == target]
            
            for method in target_data['Method_Full'].unique():
                method_data = target_data[target_data['Method_Full'] == method]
                mean_by_cutoff = method_data.groupby('Affinity_Cutoff')[metric].mean()
                std_by_cutoff = method_data.groupby('Affinity_Cutoff')[metric].std()
                
                ax.errorbar(mean_by_cutoff.index, mean_by_cutoff.values,
                           yerr=std_by_cutoff.values, marker='o', label=method, capsize=5)
            
            ax.set_xscale('log')
            ax.set_xlabel('Affinity Cutoff (nM)', fontsize=11, fontweight='bold')
            if idx == 0:
                ax.set_ylabel(f'Mean {metric}', fontsize=11, fontweight='bold')
            ax.set_title(target, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'{metric}_vs_cutoff_by_protein.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plots_created.append(plot_path)
        logging.info(f"Created plot: {plot_path}")
    
    return plots_created

def main():
    parser = argparse.ArgumentParser(description="Aggregate and analyze generalization experiment results.")
    parser.add_argument("--abl1_workspace", default="experiment_workspace_rerun_hyperparam_sweep/",
                       help="Workspace directory containing ABL1 hyperparameter results")
    parser.add_argument("--generalization_workspace", default="experiment_workspace_generalization/",
                       help="Workspace directory containing generalization results")
    parser.add_argument("--generalization_cutoff_workspace", default="experiment_workspace_generalization_cutoff/",
                       help="Optional: Workspace directory containing generalization cutoff analysis results")
    parser.add_argument("--output_dir", default="final_report_generalization/",
                       help="Output directory for report and visualizations")
    args = parser.parse_args()
    
    logging.info("=" * 80)
    logging.info("GENERALIZATION EXPERIMENT ANALYSIS")
    logging.info("=" * 80)
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_dir = os.path.join(args.output_dir, f"generalization_report_{timestamp}")
    os.makedirs(report_dir, exist_ok=True)
    
    figures_dir = os.path.join(report_dir, "figures")
    tables_dir = os.path.join(report_dir, "tables")
    
    # Collect metrics from ABL1 (use optimal hyperparameters only to match generalization)
    logging.info("\nCollecting ABL1 metrics (hyperparameter sweep)...")
    df_abl1 = collect_metrics_from_workspace(args.abl1_workspace, "ABL1_Hyperparam")
    
    # For fair comparison, filter ABL1 to only the default hyperparameters used in generalization
    # We'll keep all for now and let the analysis show the full picture
    
    # Collect metrics from generalization experiments
    logging.info("\nCollecting generalization metrics...")
    df_gen = collect_metrics_from_workspace(args.generalization_workspace, "Generalization")
    
    # Collect metrics from generalization cutoff analysis if provided
    df_gen_cutoff = pd.DataFrame()
    if args.generalization_cutoff_workspace and os.path.exists(args.generalization_cutoff_workspace):
        logging.info("\nCollecting generalization cutoff analysis metrics...")
        df_gen_cutoff = collect_metrics_from_workspace(
            args.generalization_cutoff_workspace, 
            "Generalization_Cutoff",
            is_cutoff_workspace=True
        )
    
    # Combine dataframes
    dfs_to_combine = [df_abl1, df_gen]
    if not df_gen_cutoff.empty:
        dfs_to_combine.append(df_gen_cutoff)
    df_all = pd.concat(dfs_to_combine, ignore_index=True)
    
    if df_all.empty:
        logging.error("No metrics collected. Aborting.")
        return
    
    # Save combined metrics
    master_csv = os.path.join(tables_dir, "master_generalization_metrics.csv")
    os.makedirs(tables_dir, exist_ok=True)
    df_all.to_csv(master_csv, index=False)
    logging.info(f"\nSaved master metrics to: {master_csv}")
    
    # STEP 1: Create cutoff trend plots for all target proteins (if cutoff data available)
    if not df_gen_cutoff.empty:
        logging.info("\n" + "=" * 80)
        logging.info("STEP 1: Creating cutoff trend analysis plots...")
        logging.info("=" * 80)
        cutoff_trend_plots = create_cutoff_trend_plots(df_gen_cutoff, figures_dir)
        logging.info(f"Created {len(cutoff_trend_plots)} cutoff trend plots")
        
        # STEP 2: Select optimal cutoff for each method based on mean EF@1% across all targets
        logging.info("\n" + "=" * 80)
        logging.info("STEP 2: Selecting optimal cutoff for each DR method...")
        logging.info("=" * 80)
        df_optimal_cutoff, optimal_cutoffs_map = select_optimal_cutoff_per_method(df_gen_cutoff)
        
        # Save optimal cutoff information
        optimal_cutoff_df = pd.DataFrame([
            {'DR_Method': method, 'Optimal_Cutoff_nM': cutoff}
            for method, cutoff in optimal_cutoffs_map.items()
        ])
        optimal_cutoff_path = os.path.join(tables_dir, 'optimal_cutoffs_by_method.csv')
        optimal_cutoff_df.to_csv(optimal_cutoff_path, index=False)
        logging.info(f"Saved optimal cutoff selections to: {optimal_cutoff_path}")
        
        # Use the optimal cutoff data for comparison plots
        df_for_comparison = df_optimal_cutoff
        logging.info(f"Using {len(df_for_comparison)} rows with optimal cutoffs for comparison plots")
    else:
        logging.info("\nNo cutoff data available. Using all data for comparison plots.")
        df_for_comparison = df_all
    
    # STEP 3: Create visualizations with optimal cutoff data
    logging.info("\n" + "=" * 80)
    logging.info("STEP 3: Creating comparison plots with optimal cutoff data...")
    logging.info("=" * 80)
    plots = create_comparison_plots(df_for_comparison, figures_dir)
    logging.info(f"Created {len(plots)} comparison plots")
    
    # Create summary tables
    logging.info("\nCreating summary tables...")
    tables = create_summary_tables(df_for_comparison, tables_dir)
    logging.info(f"Created {len(tables)} tables")
    
    # Print summary statistics
    logging.info("\n" + "=" * 80)
    logging.info("SUMMARY STATISTICS")
    logging.info("=" * 80)
    logging.info(f"\nTotal experiments collected: {len(df_all)}")
    logging.info(f"Unique targets: {df_all['Target_ID'].nunique()}")
    logging.info(f"Unique methods: {df_all['Method_Full'].nunique()}")
    logging.info(f"Seeds per config: {df_all['Seed'].nunique()}")
    
    if not df_gen_cutoff.empty:
        logging.info(f"\nCutoff analysis enabled:")
        logging.info(f"  Total cutoff experiments: {len(df_gen_cutoff)}")
        logging.info(f"  Unique affinity cutoffs tested: {sorted(df_gen_cutoff['Affinity_Cutoff'].unique())}")
        logging.info(f"  Experiments with optimal cutoffs: {len(df_for_comparison)}")
        logging.info(f"\nOptimal cutoffs selected:")
        for method, cutoff in optimal_cutoffs_map.items():
            logging.info(f"  {method}: {cutoff} nM")
    
    logging.info("\nTop performing methods (by mean EF@1% across all proteins):")
    top_methods = df_for_comparison.groupby('Method_Full')['EF_1Perc'].mean().sort_values(ascending=False).head(5)
    for method, ef in top_methods.items():
        logging.info(f"  {method}: {ef:.2f}")
    
    logging.info("\n" + "=" * 80)
    logging.info(f"Analysis complete. Results saved to: {report_dir}")
    logging.info("=" * 80)

if __name__ == "__main__":
    main()
