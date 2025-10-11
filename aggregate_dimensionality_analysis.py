#!/usr/bin/env python3
"""
Aggregate and analyze results from the dimensionality experiment.

This script:
1. Collects performance metrics from all dimensionality experiments (dimensions: 2, 3, 5, 10, 20)
2. Compares method performance across dimensions for ABL1
3. Generates line plots showing EF@1% vs dimensionality with error bars
4. Creates a comprehensive LaTeX report
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
from scipy import stats

# --- Logging Setup ---
log_file_name = f"dimensionality_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)-8s - %(funcName)-25s - %(lineno)-4d - %(message)s',
    handlers=[
        logging.FileHandler(log_file_name, mode='w'),
        logging.StreamHandler()
    ]
)

def collect_metrics_from_dimensionality_workspace(workspace_dir):
    """
    Collect metrics from the dimensionality experiment workspace.
    
    Args:
        workspace_dir: Path to the workspace directory (experiment_workspace_dimensionality)
    
    Returns:
        DataFrame with metrics across all dimensions
    """
    all_metrics = []
    
    # Pattern to find all ranking metrics files
    pattern = os.path.join(workspace_dir, "run_seed*", "*", "results", "*", "dim_*", "*", "*_ranking_metrics.csv")
    
    logging.info(f"Searching for metrics in: {workspace_dir}")
    logging.info(f"Pattern: {pattern}")
    
    for metrics_file in glob.glob(pattern):
        try:
            path_parts = metrics_file.split(os.sep)
            
            # Extract run directory name
            run_dir_name = next(p for p in path_parts if p.startswith("run_seed"))
            
            # Extract seed
            seed_match = re.search(r"run_seed(\d+)", run_dir_name)
            if not seed_match:
                logging.warning(f"Could not extract seed from: {run_dir_name}")
                continue
            seed = int(seed_match.group(1))
            
            # Extract target ID
            target_idx = path_parts.index(run_dir_name) + 1
            target_id = path_parts[target_idx]
            
            # Extract representation type
            repr_type = path_parts[path_parts.index("results") + 1]
            
            # Extract dimensionality
            dim_dir = next(p for p in path_parts if p.startswith("dim_"))
            dim_match = re.search(r"dim_(\d+)", dim_dir)
            if not dim_match:
                logging.warning(f"Could not extract dimension from: {dim_dir}")
                continue
            dimension = int(dim_match.group(1))
            
            # Extract strategy directory
            dim_idx = path_parts.index(dim_dir)
            strategy_dir = path_parts[dim_idx + 1]
            
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
            
            # Determine embedding strategy (should all be co-embedding)
            embedding_strategy = "Co-embedding"
            if "tsne" in dr_method_key and dimension == 2:
                embedding_strategy = "Co-embedding (Native)"
            
            # Read metrics
            df_metrics = pd.read_csv(metrics_file)
            if df_metrics.empty:
                continue
            
            # Create metric row
            metric_row = {
                'Target_ID': target_id,
                'Seed': seed,
                'Dimension': dimension,
                'Representation': repr_type.capitalize(),
                'DR_Method': dr_params['short_name'],
                'Embedding_Strategy': embedding_strategy,
                'Hyperparameters': hyperparam_str,
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
            logging.exception(e)
            continue
    
    df = pd.DataFrame(all_metrics)
    logging.info(f"Collected {len(df)} metric rows from dimensionality experiment")
    
    return df

def calculate_confidence_intervals(data, confidence=0.90):
    """
    Calculate confidence interval for a dataset.
    
    Args:
        data: Array-like of values
        confidence: Confidence level (default 0.90 for 90% CI)
    
    Returns:
        mean, lower_bound, upper_bound
    """
    data = np.array(data)
    data = data[~np.isnan(data)]
    
    if len(data) == 0:
        return np.nan, np.nan, np.nan
    
    mean = np.mean(data)
    
    if len(data) == 1:
        return mean, mean, mean
    
    # Calculate confidence interval
    sem = stats.sem(data)
    ci = sem * stats.t.ppf((1 + confidence) / 2., len(data) - 1)
    
    return mean, mean - ci, mean + ci

def create_dimensionality_line_plots(df, output_dir):
    """
    Create line plots showing performance metrics vs dimensionality for each method.
    
    Primary plot: EF@1% vs Dimension
    Secondary plots: ROC-AUC and PR-AUC vs Dimension
    """
    os.makedirs(output_dir, exist_ok=True)
    plots_created = []
    
    # Define dimensions in order
    dimensions = sorted(df['Dimension'].unique())
    
    # Define metrics to plot
    metrics_info = [
        {'metric': 'EF_1Perc', 'title': 'Enrichment Factor at 1% vs Dimensionality', 'ylabel': 'EF@1%', 'primary': True},
        {'metric': 'ROC_AUC', 'title': 'ROC-AUC vs Dimensionality', 'ylabel': 'ROC-AUC', 'primary': False},
        {'metric': 'PR_AUC', 'title': 'PR-AUC vs Dimensionality', 'ylabel': 'PR-AUC', 'primary': False}
    ]
    
    # Get unique DR methods
    dr_methods = df['DR_Method'].unique()
    
    # Color palette
    colors = {'PCA': '#1f77b4', 't-SNE': '#ff7f0e', 'UMAP-Euclidean': '#2ca02c'}
    markers = {'PCA': 'o', 't-SNE': 's', 'UMAP-Euclidean': '^'}
    
    for metric_info in metrics_info:
        metric = metric_info['metric']
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for method in dr_methods:
            method_data = df[df['DR_Method'] == method]
            
            means = []
            lower_bounds = []
            upper_bounds = []
            dims_with_data = []
            
            for dim in dimensions:
                dim_data = method_data[method_data['Dimension'] == dim][metric].dropna()
                
                if len(dim_data) > 0:
                    mean, lower, upper = calculate_confidence_intervals(dim_data)
                    means.append(mean)
                    lower_bounds.append(lower)
                    upper_bounds.append(upper)
                    dims_with_data.append(dim)
            
            if means:
                color = colors.get(method, None)
                marker = markers.get(method, 'o')
                
                # Plot line with error bars
                ax.errorbar(dims_with_data, means, 
                           yerr=[np.array(means) - np.array(lower_bounds), 
                                 np.array(upper_bounds) - np.array(means)],
                           label=method, marker=marker, markersize=8, linewidth=2,
                           color=color, capsize=5, capthick=2)
        
        # Formatting
        ax.set_xlabel('Dimensionality (d)', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric_info['ylabel'], fontsize=12, fontweight='bold')
        ax.set_title(metric_info['title'], fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, frameon=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xticks(dimensions)
        
        # Set y-axis limits based on metric
        if metric == 'EF_1Perc':
            ax.set_ylim(bottom=0)
        elif metric in ['ROC_AUC', 'PR_AUC']:
            ax.set_ylim(0, 1)
        
        plt.tight_layout()
        
        # Save figure
        output_file = os.path.join(output_dir, f"dimensionality_{metric.lower()}_lineplot.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        plots_created.append(output_file)
        logging.info(f"Created plot: {output_file}")
    
    return plots_created

def create_summary_tables(df, output_dir):
    """
    Create summary tables for each metric showing mean ± std across dimensions.
    """
    os.makedirs(output_dir, exist_ok=True)
    tables_created = []
    
    metrics = ['EF_1Perc', 'ROC_AUC', 'PR_AUC']
    
    for metric in metrics:
        # Group by method and dimension
        summary_data = []
        
        for method in sorted(df['DR_Method'].unique()):
            method_row = {'Method': method}
            
            for dim in sorted(df['Dimension'].unique()):
                dim_data = df[(df['DR_Method'] == method) & (df['Dimension'] == dim)][metric].dropna()
                
                if len(dim_data) > 0:
                    mean = dim_data.mean()
                    std = dim_data.std()
                    method_row[f'Dim_{dim}'] = f"{mean:.3f} ± {std:.3f}"
                else:
                    method_row[f'Dim_{dim}'] = "N/A"
            
            summary_data.append(method_row)
        
        # Create DataFrame
        summary_df = pd.DataFrame(summary_data)
        
        # Save to CSV
        output_file = os.path.join(output_dir, f"summary_table_{metric.lower()}.csv")
        summary_df.to_csv(output_file, index=False)
        tables_created.append(output_file)
        logging.info(f"Created table: {output_file}")
    
    return tables_created

def generate_latex_report(df, plots_created, tables_created, output_dir):
    """
    Generate a comprehensive LaTeX report for the dimensionality analysis.
    """
    report_dir = os.path.join(output_dir, f"dimensionality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(report_dir, exist_ok=True)
    
    # Copy plots and tables to report directory
    import shutil
    
    for plot_file in plots_created:
        shutil.copy(plot_file, report_dir)
    
    for table_file in tables_created:
        shutil.copy(table_file, report_dir)
    
    # Generate LaTeX document
    latex_content = []
    
    # Preamble
    latex_content.append(r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{float}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{amsmath}

\title{Impact of Similarity Space Dimensionality on Performance\\
\large UMMBAS Dimensionality Experiment Report}
\author{UMMBAS Project Team}
\date{\today}

\begin{document}
\maketitle

\begin{abstract}
This report presents the results of the dimensionality experiment, which investigates how 
the performance of different dimensionality reduction methods (PCA, UMAP, and t-SNE) is 
affected by varying the dimensionality of the similarity space. Using the ABL1 target 
protein and physicochemical features, we tested dimensions of 2, 3, 5, 10, and 20 using 
optimal hyperparameters identified in previous experiments. All analyses used the 
co-embedding strategy. Performance was evaluated using Enrichment Factor at 1\% (EF@1\%), 
ROC-AUC, and PR-AUC across 5 random seeds to ensure statistical robustness.
\end{abstract}

\clearpage
\tableofcontents
\clearpage

\section{Introduction}

The dimensionality of the similarity space is a critical parameter that can significantly 
impact the performance of virtual screening methods. While lower dimensions (2D, 3D) are 
easier to visualize and interpret, higher dimensions may capture more subtle molecular 
relationships that could improve screening performance.

This experiment addresses the following research question: \textbf{How does the 
dimensionality of the similarity space affect the ability to identify active compounds?}

\subsection{Experimental Design}

\begin{itemize}
    \item \textbf{Target Protein:} Tyrosine-protein Kinase ABL1 (P00519)
    \item \textbf{Molecular Representation:} Physicochemical features (39 features)
    \item \textbf{Dimensionality Reduction Methods:} PCA, UMAP (Euclidean), and t-SNE
    \item \textbf{Dimensions Tested:} 2, 3, 5, 10, and 20
    \item \textbf{Embedding Strategy:} Co-embedding only
    \item \textbf{Hyperparameters:} Optimal values from Experiment 1:
    \begin{itemize}
        \item PCA: No hyperparameters
        \item UMAP: n\_neighbors=500, min\_dist=0.01
        \item t-SNE: perplexity=1000
    \end{itemize}
    \item \textbf{Random Seeds:} 5 replicates (42-46)
    \item \textbf{Primary Metric:} Enrichment Factor at 1\% (EF@1\%)
    \item \textbf{Secondary Metrics:} ROC-AUC, PR-AUC
\end{itemize}

\section{Results}

\subsection{Primary Analysis: EF@1\% vs Dimensionality}

Figure~\ref{fig:ef1_lineplot} shows the enrichment factor at 1\% as a function of 
similarity space dimensionality for all three methods. Error bars represent 90\% 
confidence intervals across 5 random seeds.
""")
    
    # Add EF@1% plot
    ef_plot = next((p for p in plots_created if 'ef_1perc' in os.path.basename(p).lower()), None)
    if ef_plot:
        latex_content.append(r"""
\begin{figure}[H]
    \centering
    \includegraphics[width=0.95\textwidth]{""" + os.path.basename(ef_plot) + r"""}
    \caption{Enrichment Factor at 1\% vs Dimensionality. Error bars show 90\% confidence intervals across 5 random seeds.}
    \label{fig:ef1_lineplot}
\end{figure}
""")
    
    # Add interpretation section
    latex_content.append(r"""
\subsubsection{Key Observations}

\begin{itemize}
    \item \textbf{Performance trends:} Analyze how each method's performance changes with dimensionality.
    \item \textbf{Optimal dimensions:} Identify which dimension provides the best performance for each method.
    \item \textbf{Method comparison:} Compare the three methods across all dimensions.
    \item \textbf{Statistical significance:} Consider the error bars when interpreting differences.
\end{itemize}

\subsection{Secondary Analyses}

Figures~\ref{fig:rocauc_lineplot} and~\ref{fig:prauc_lineplot} show ROC-AUC and PR-AUC 
performance across dimensions, respectively.
""")
    
    # Add ROC-AUC plot
    roc_plot = next((p for p in plots_created if 'roc_auc' in os.path.basename(p).lower()), None)
    if roc_plot:
        latex_content.append(r"""
\begin{figure}[H]
    \centering
    \includegraphics[width=0.95\textwidth]{""" + os.path.basename(roc_plot) + r"""}
    \caption{ROC-AUC vs Dimensionality. Error bars show 90\% confidence intervals.}
    \label{fig:rocauc_lineplot}
\end{figure}
""")
    
    # Add PR-AUC plot
    pr_plot = next((p for p in plots_created if 'pr_auc' in os.path.basename(p).lower()), None)
    if pr_plot:
        latex_content.append(r"""
\begin{figure}[H]
    \centering
    \includegraphics[width=0.95\textwidth]{""" + os.path.basename(pr_plot) + r"""}
    \caption{PR-AUC vs Dimensionality. Error bars show 90\% confidence intervals.}
    \label{fig:prauc_lineplot}
\end{figure}
""")
    
    # Add summary tables
    latex_content.append(r"""
\section{Summary Tables}

The following tables provide numerical summaries of performance across all dimensions.
Each cell shows mean $\pm$ standard deviation across 5 random seeds.

""")
    
    # Add tables if available
    for table_file in tables_created:
        try:
            df_table = pd.read_csv(table_file)
            metric_name = os.path.basename(table_file).replace('summary_table_', '').replace('.csv', '').upper()
            
            latex_content.append(f"\\subsection{{{metric_name}}}\n\n")
            latex_content.append(r"\begin{table}[H]" + "\n")
            latex_content.append(r"    \centering" + "\n")
            latex_content.append(r"    \small" + "\n")
            
            # Convert DataFrame to LaTeX
            latex_table = df_table.to_latex(index=False, escape=False, column_format='l' + 'c' * (len(df_table.columns) - 1))
            latex_content.append(latex_table)
            
            latex_content.append(f"    \\caption{{{metric_name} across dimensions}}\n")
            latex_content.append(r"\end{table}" + "\n\n")
        except Exception as e:
            logging.warning(f"Could not include table {table_file}: {e}")
    
    # Conclusions
    latex_content.append(r"""
\section{Conclusions}

This experiment systematically investigated the impact of similarity space dimensionality 
on the performance of three dimensionality reduction methods (PCA, UMAP, and t-SNE) for 
virtual screening of active compounds.

\subsection{Main Findings}

\textit{[To be filled in based on results]}

\subsection{Recommendations}

Based on these results, we recommend:

\begin{itemize}
    \item \textit{[Optimal dimension for each method]}
    \item \textit{[Trade-offs between performance and interpretability]}
    \item \textit{[Method selection guidance based on use case]}
\end{itemize}

\section{Methods}

All experiments used the leave-one-target-out methodology with ABL1 as the target protein. 
The molecular function cloud (MF Cloud) was constructed from ChEMBL compounds with the same 
molecular function (Protein kinase inhibitor), excluding compounds for ABL1. ZINC decoys 
were used as negative examples. Performance was evaluated by ranking held-out active 
compounds against decoys based on minimum Euclidean distance to the MF Cloud in the 
dimensionality-reduced space.

\end{document}
""")
    
    # Write LaTeX file
    latex_file = os.path.join(report_dir, "dimensionality_analysis_report.tex")
    with open(latex_file, 'w') as f:
        f.writelines(latex_content)
    
    logging.info(f"LaTeX report written to: {latex_file}")
    
    # Try to compile LaTeX
    try:
        import subprocess
        logging.info("Attempting to compile LaTeX report...")
        result = subprocess.run(['pdflatex', '-interaction=nonstopmode', latex_file],
                              cwd=report_dir, capture_output=True, text=True)
        
        if result.returncode == 0:
            # Run twice for references
            subprocess.run(['pdflatex', '-interaction=nonstopmode', latex_file],
                         cwd=report_dir, capture_output=True, text=True)
            logging.info(f"PDF report generated: {os.path.join(report_dir, 'dimensionality_analysis_report.pdf')}")
        else:
            logging.warning("LaTeX compilation failed. Manual compilation may be needed.")
            logging.debug(f"LaTeX output: {result.stdout}")
    except FileNotFoundError:
        logging.warning("pdflatex not found. Please compile the .tex file manually.")
    except Exception as e:
        logging.warning(f"Could not compile LaTeX: {e}")
    
    return report_dir

def main():
    parser = argparse.ArgumentParser(description="Aggregate and analyze dimensionality experiment results.")
    parser.add_argument("--workspace", default="experiment_workspace_dimensionality/",
                       help="Path to the dimensionality experiment workspace directory")
    parser.add_argument("--output_dir", default="final_report_dimensionality/",
                       help="Path to output directory for reports and plots")
    args = parser.parse_args()
    
    logging.info("=" * 80)
    logging.info("DIMENSIONALITY EXPERIMENT ANALYSIS")
    logging.info("=" * 80)
    
    # Collect metrics
    logging.info(f"\nCollecting metrics from: {args.workspace}")
    df = collect_metrics_from_dimensionality_workspace(args.workspace)
    
    if df.empty:
        logging.error("No metrics collected. Exiting.")
        return
    
    logging.info(f"Total metrics collected: {len(df)}")
    logging.info(f"Dimensions found: {sorted(df['Dimension'].unique())}")
    logging.info(f"Methods found: {df['DR_Method'].unique()}")
    logging.info(f"Seeds found: {sorted(df['Seed'].unique())}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save collected metrics
    metrics_file = os.path.join(args.output_dir, "dimensionality_all_metrics.csv")
    df.to_csv(metrics_file, index=False)
    logging.info(f"Saved all metrics to: {metrics_file}")
    
    # Create line plots
    logging.info("\nGenerating dimensionality line plots...")
    plots_dir = os.path.join(args.output_dir, "plots")
    plots_created = create_dimensionality_line_plots(df, plots_dir)
    logging.info(f"Created {len(plots_created)} plots")
    
    # Create summary tables
    logging.info("\nGenerating summary tables...")
    tables_dir = os.path.join(args.output_dir, "tables")
    tables_created = create_summary_tables(df, tables_dir)
    logging.info(f"Created {len(tables_created)} tables")
    
    # Generate LaTeX report
    logging.info("\nGenerating LaTeX report...")
    report_dir = generate_latex_report(df, plots_created, tables_created, args.output_dir)
    
    logging.info("\n" + "=" * 80)
    logging.info("ANALYSIS COMPLETE")
    logging.info("=" * 80)
    logging.info(f"Output directory: {args.output_dir}")
    logging.info(f"Report directory: {report_dir}")
    logging.info(f"Log file: {log_file_name}")
    logging.info("=" * 80)

if __name__ == "__main__":
    main()
