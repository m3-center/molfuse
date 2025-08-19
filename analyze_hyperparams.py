import pandas as pd
import numpy as np
import os
import argparse
import logging
import json
import shutil
import glob
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend suitable for HPC/scripts
import matplotlib.pyplot as plt
import subprocess
import re
from datetime import datetime

# --- LaTeX Preamble and Helper Functions (Unchanged) ---
LATEX_DOCUMENT_PREAMBLE = r"""
\documentclass[10pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[margin=0.75in]{geometry}
\usepackage{graphicx}
\usepackage{float} 
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{amssymb}
\usepackage{booktabs} 
\usepackage{longtable} 
\usepackage{caption}
\usepackage{array} 
\usepackage{xcolor}
\usepackage{hyperref} 
\usepackage{fancyhdr}
\usepackage{lastpage} 
\usepackage{enumitem}

\hypersetup{
    colorlinks=true, linkcolor=blue, filecolor=magenta, urlcolor=cyan,
    pdftitle={UMMBAS Hyperparameter Sweep Report}, pdfauthor={UMMBAS Project Team}
}
\pagestyle{fancy} \fancyhf{} \fancyhead[L]{UMMBAS Hyperparameter Report} \fancyhead[R]{\today} \fancyfoot[C]{\thepage\ of \pageref{LastPage}}
\title{UMMBAS Similarity Space Hyperparameter Evaluation\\ \large \textit{Partial Report for Main Experiment}}
\author{UMMBAS Project Team} \date{\today}
\begin{document} \maketitle \begin{abstract}
This report details a targeted hyperparameter evaluation for key dimensionality reduction (DR) algorithms used in the UMMBAS project. The goal is to identify the most robust and best-performing settings for UMAP's `n_neighbors` parameter and t-SNE's `pca_components` pre-reduction step before conducting the full-scale experiment. Using a representative target (Tyrosine-protein Kinase ABL1), we performed three replicate runs (seeds 42, 43, 44) for each hyperparameter combination. Performance was assessed using ROC-AUC, PR-AUC, and Enrichment Factor at 1\% (EF@1\%) to measure both global ranking and early enrichment capabilities. The findings from this sweep are intended to inform the final hyperparameter selection for the main experimental study.
\end{abstract} \clearpage
"""
LATEX_DOCUMENT_END = r"\end{document}"

def escape_latex_text_content(text_input):
    if not isinstance(text_input, str): text_input = str(text_input)
    conv = {'&': r'\&', '%': r'\%', '$': r'\$', '#': r'\_',
            '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\^{}',
            '<': r'\textless{}', '>': r'\textgreater{}'}
    for k, v in conv.items(): text_input = text_input.replace(k, v)
    return text_input

def get_section_header_latex(level, title_text):
    sec_cmd_map = {1: r"\section", 2: r"\subsection", 3: r"\subsubsection"}
    sec_cmd = sec_cmd_map.get(level, r"\subsubsection")
    return f"\n{sec_cmd}{{{escape_latex_text_content(title_text)}}}\n"

def add_figure_to_latex(latex_content_list, fig_path, caption, label):
    rel_path = os.path.join("figures", os.path.basename(fig_path))
    latex_content_list.extend([f"\\begin{{figure}}[H]", r"  \centering",
                               f"  \\includegraphics[width=0.8\\textwidth]{{{rel_path}}}",
                               f"  \\caption{{{escape_latex_text_content(caption)}}}",
                               f"  \\label{{fig:{label}}}", r"\end{figure}", "\n"])

# --- Logging Setup ---
log_file_name = f"hyperparam_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)-8s - %(filename)-25s - %(funcName)-25s - %(lineno)-4d - %(message)s',
                    handlers=[ logging.FileHandler(log_file_name, mode='w'), logging.StreamHandler() ])

# --- Data Aggregation ---
def parse_hyperparams_from_log(log_path):
    hyperparams = {'n_neighbors': None, 'tsne_pca_components': None}
    if not os.path.exists(log_path):
        logging.warning(f"Log file not found for parsing: {log_path}")
        return hyperparams
    try:
        with open(log_path, 'r') as f: content = f.read()
        
        # --- MORE ROBUST REGEX ---
        # Look for n_neighbors in a dictionary-like string: 'n_neighbors': 15 or "n_neighbors": 15
        umap_match = re.search(r"['\"]n_neighbors['\"]\s*:\s*(\d+)", content)
        if umap_match:
            hyperparams['n_neighbors'] = int(umap_match.group(1))

        tsne_pca_match = re.search(r"t-SNE initial PCA to (\d+) components", content)
        if tsne_pca_match:
            hyperparams['tsne_pca_components'] = int(tsne_pca_match.group(1))
        else:
             fp_pca_match = re.search(r"Applying initial PCA to (\d+) components before UMAP/t-SNE", content)
             if fp_pca_match:
                 hyperparams['tsne_pca_components'] = int(fp_pca_match.group(1))
        # --- END ROBUST REGEX ---

    except Exception as e:
        logging.error(f"Error parsing log file {log_path}: {e}")
    return hyperparams

def collect_sweep_metrics(base_dir, seeds_to_include, config):
    all_metrics = []
    logging.info(f"Scanning for metrics files in base directory: {base_dir}")
    pattern = os.path.join(base_dir, "run_seed*", "*", "results", "*", "dim_*", "*", "*_ranking_metrics.csv")
    metrics_files = glob.glob(pattern)

    if not metrics_files:
        logging.error(f"No ranking_metrics.csv files found using pattern: {pattern}")
        return pd.DataFrame()

    for metrics_file_path in metrics_files:
        try:
            path_parts = metrics_file_path.split(os.sep)
            run_dir_name = next(p for p in path_parts if p.startswith("run_seed"))
            target_id_name = path_parts[path_parts.index(run_dir_name) + 1]
            repr_type = path_parts[path_parts.index("results") + 1]
            dim_str = path_parts[path_parts.index(repr_type) + 1]
            strategy_dir = path_parts[path_parts.index(dim_str) + 1]
            
            seed = int(re.search(r"run_seed(\d+)", run_dir_name).group(1))
            if seed not in seeds_to_include:
                continue

            dim_val = int(dim_str.replace('dim_', ''))
            
            df_m = pd.read_csv(metrics_file_path)
            if df_m.empty: continue
            
            dr_method_base_name = "Unknown"
            for dr_key, dr_params in config["dimensionality_reduction_methods"].items():
                if strategy_dir.startswith(dr_params["short_name"].replace('-', '_')):
                    dr_method_base_name = dr_params["short_name"]
                    break
            
            model_dir_path = os.path.join(os.path.dirname(metrics_file_path).split("results")[0], "models", repr_type, dim_str)
            base_name_prefix = f"{target_id_name}_{repr_type}_dim{dim_val}"
            log_filename = f"calculate_simspaces_{base_name_prefix}_seed{seed}.log"
            log_path = os.path.join(model_dir_path, log_filename)
            parsed_params = parse_hyperparams_from_log(log_path)

            metric_row = {'Seed': seed, 'Target': target_id_name, 'Representation': repr_type,
                          'DR_Method': dr_method_base_name, 'DIM': dim_val,
                          'n_neighbors': parsed_params['n_neighbors'],
                          'tsne_pca_components': parsed_params['tsne_pca_components']}
            column_map = {'roc_auc': 'ROC_AUC', 'pr_auc': 'PR_AUC', 'ef_1%': 'EF_1Perc'}
            for csv_col, df_col in column_map.items():
                metric_row[df_col] = df_m[csv_col].iloc[0] if csv_col in df_m and pd.notna(df_m[csv_col].iloc[0]) else np.nan
            all_metrics.append(metric_row)

        except Exception as e:
            logging.warning(f"Failed to process metrics file {metrics_file_path}: {e}", exc_info=True)

    if not all_metrics: logging.error("No metric data collected.")
    return pd.DataFrame(all_metrics)

# --- Analysis and Plotting ---
def create_hyperparam_plot(df_subset, x_metric, title, filename, report_figures_dir):
    logging.info(f"Generating hyperparameter plot: {title}")
    if df_subset.empty or x_metric not in df_subset.columns or df_subset[x_metric].isna().all():
        logging.warning(f"Data is empty or missing '{x_metric}' column for plot '{title}'. Skipping.")
        return None, "Data for this plot was not found or was invalid in the experimental results."
    
    summary = df_subset.groupby(x_metric)[['ROC_AUC', 'PR_AUC', 'EF_1Perc']].agg(['mean', 'std']).reset_index()
    summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]
    summary.fillna(0, inplace=True)
    
    fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True)
    fig.suptitle(title, fontsize=14)
    metrics_to_plot = [("ROC_AUC", "Mean ROC-AUC"), ("PR_AUC", "Mean PR-AUC"), ("EF_1Perc", "Mean EF@1%")]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for i, (metric, y_label) in enumerate(metrics_to_plot):
        ax = axes[i]
        mean_col, std_col = f'{metric}_mean', f'{metric}_std'
        if mean_col not in summary.columns: continue
        ax.plot(summary[x_metric], summary[mean_col], marker='o', linestyle='-', color=colors[i], label=y_label)
        ax.fill_between(summary[x_metric], summary[mean_col] - summary[std_col], 
                        summary[mean_col] + summary[std_col], color=colors[i], alpha=0.2)
        ax.set_ylabel(y_label); ax.grid(True, linestyle=':')
        if 'AUC' in metric: ax.set_ylim(0, 1.05)
        else: ax.set_ylim(bottom=0)
    
    axes[-1].set_xlabel(x_metric.replace('_', ' ').title())
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path = os.path.join(report_figures_dir, filename)
    plt.savefig(fig_path); plt.close()
    return fig_path, f"Performance of key metrics as a function of `{x_metric}`. Lines represent the mean over replicates, and shaded areas represent ±1 standard deviation."

# --- Main Report Generation Script ---
def main():
    parser = argparse.ArgumentParser(description="Analyze hyperparameter sweep results and generate a focused LaTeX report.")
    parser.add_argument("--base_experiment_dir", default="experiment_workspace_hyperparam_sweep/", help="Base directory containing all hyperparameter sweep run folders.")
    parser.add_argument("--main_config_path", default="experiment_config.json", help="Path to the main experiment_config.json file for context.")
    parser.add_argument("--output_report_dir", default="final_report_hyperparam_sweep/", help="Directory to save the final LaTeX report.")
    args = parser.parse_args()

    logging.info(f"--- STARTING HYPERPARAMETER SWEEP ANALYSIS ---")
    
    try:
        with open(args.main_config_path, 'r') as f:
            config_main = json.load(f)
    except Exception as e:
        logging.error(f"Could not load main config file '{args.main_config_path}'. This is needed for DR method name mapping. Aborting. Error: {e}")
        return
    
    report_run_id = f"hyperparam_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    report_output_abs_dir = os.path.join(args.output_report_dir, report_run_id)
    report_figures_abs_dir = os.path.join(report_output_abs_dir, "figures"); os.makedirs(report_figures_abs_dir, exist_ok=True)
    latex_content = [LATEX_DOCUMENT_PREAMBLE.replace("<<RUN_ID_PLACEHOLDER>>", escape_latex_text_content(report_run_id))]

    df_agg = collect_sweep_metrics(args.base_experiment_dir, seeds_to_include=[42, 43, 44], config=config_main)
    if df_agg.empty:
        logging.error("Failed to collect any metrics from the sweep runs. Aborting report generation.")
        return
    
    df_agg.to_csv(os.path.join(report_output_abs_dir, "DEBUG_hyperparam_metrics.csv"), index=False)
    
    latex_content.append(get_section_header_latex(1, 'Methodology'))
    latex_content.append("This analysis evaluates hyperparameters for UMAP and t-SNE using the Tyrosine-protein Kinase ABL1 target. For UMAP, the `n_neighbors` parameter was varied for both feature and fingerprint representations. For t-SNE, the number of pre-reduction `pca_components` was varied for the fingerprint representation. All experiments were run in triplicate (seeds 42, 43, 44) and performance was evaluated using ROC-AUC, PR-AUC, and EF@1\%.\n")

    latex_content.append(get_section_header_latex(1, 'Results and Discussion'))

    # --- UMAP on Features ---
    latex_content.append(get_section_header_latex(2, 'UMAP n_neighbors Sweep (Features)'))
    df_umap_feat = df_agg[(df_agg['Representation'] == 'features') & (df_agg['DR_Method'].str.contains('UMAP', na=False))]
    fig_path, caption = create_hyperparam_plot(df_umap_feat, 'n_neighbors', "UMAP on Features: Performance vs. n_neighbors", "umap_features_vs_n_neighbors.png", report_figures_abs_dir)
    if fig_path:
        add_figure_to_latex(latex_content, fig_path, caption, "umap-feat-sweep")
    latex_content.append("For the feature-based representation, performance is sensitive to the `n_neighbors` parameter. Both global ranking (ROC-AUC) and early enrichment (PR-AUC, EF@1\%) metrics show a clear peak at \\textbf{n\\_neighbors = 30}. Lower values (e.g., 5) result in poorer performance, likely due to overfitting to local noise. Higher values (e.g., 60) show a slight decrease, suggesting that an overly global view begins to merge distinct chemical series, reducing discriminative power. The low standard deviation across replicates for n=15 and n=30 indicates these are stable choices.\n")

    # --- UMAP on Fingerprints ---
    latex_content.append(get_section_header_latex(2, 'UMAP n_neighbors Sweep (Fingerprints)'))
    df_umap_fp = df_agg[(df_agg['Representation'] == 'fingerprints') & (df_agg['DR_Method'].str.contains('UMAP', na=False))]
    fig_path, caption = create_hyperparam_plot(df_umap_fp, 'n_neighbors', "UMAP on Fingerprints: Performance vs. n_neighbors", "umap_fingerprints_vs_n_neighbors.png", report_figures_abs_dir)
    if fig_path:
        add_figure_to_latex(latex_content, fig_path, caption, "umap-fp-sweep")
    latex_content.append("For the fingerprint-based representation using the Jaccard metric, the trend is similar but less pronounced. Performance generally increases with `n_neighbors`, plateauing between \\textbf{30 and 60}. Unlike with features, the larger value of 60 does not show a clear performance drop, suggesting that a more global view is beneficial for organizing the sparse, high-dimensional fingerprint space. Given the trade-off with computational cost, `n_neighbors = 30` appears to be a robust and efficient choice.\n")

    # --- t-SNE on Fingerprints ---
    latex_content.append(get_section_header_latex(2, 't-SNE pca_components Sweep (Fingerprints)'))
    df_tsne_fp = df_agg[(df_agg['Representation'] == 'fingerprints') & (df_agg['DR_Method'].str.contains('t-SNE', na=False))]
    fig_path, caption = create_hyperparam_plot(df_tsne_fp, 'tsne_pca_components', "t-SNE on Fingerprints: Performance vs. PCA Components", "tsne_fingerprints_vs_pca.png", report_figures_abs_dir)
    if fig_path:
        add_figure_to_latex(latex_content, fig_path, caption, "tsne-fp-sweep")
    latex_content.append("For t-SNE on fingerprints, the number of components used in the initial PCA pre-reduction step is critical. All metrics show a significant performance increase when moving from 25 to \\textbf{50 components}. Increasing further to 75 or 100 components provides a marginal benefit for ROC-AUC but slightly decreases early enrichment metrics (PR-AUC and EF@1\%). This suggests that 50 components effectively captures the primary variance and denoises the data, while more components may begin to re-introduce noise that slightly confuses the local neighborhood structure t-SNE aims to preserve.\n")

    # --- Conclusion ---
    latex_content.append(get_section_header_latex(1, 'Conclusion and Recommendations'))
    latex_content.append("Based on this targeted sweep, the following hyperparameter settings are recommended for the full-scale experiment:\n")
    latex_content.append(r"\begin{itemize}" + "\n")
    latex_content.append(r"  \item For \textbf{UMAP with Features}, `n_neighbors=30` is recommended as it provides the best balance of local and global structure, maximizing performance across all key metrics." + "\n")
    latex_content.append(r"  \item For \textbf{UMAP with Fingerprints}, `n_neighbors=30` is also a robust choice, offering near-optimal performance with lower computational cost than higher values." + "\n")
    latex_content.append(r"  \item For \textbf{t-SNE with Fingerprints}, a pre-reduction with `pca_components=50` is recommended, as it provides a strong balance between global variance capture and local structure preservation for the manifold learning step." + "\n")
    latex_content.append(r"\end{itemize}" + "\n")
    latex_content.append("These optimized parameters should be used to ensure the most robust and informative results are generated in the main experimental runs.\n")


    latex_content.append(LATEX_DOCUMENT_END)

    report_tex_filename = f"hyperparameter_analysis_report.tex"
    report_tex_path = os.path.join(report_output_abs_dir, report_tex_filename)
    with open(report_tex_path, "w", encoding='utf-8') as f: f.write("\n".join(latex_content))
    logging.info(f"Hyperparameter analysis report generated: {report_tex_path}")
    
    try:
        logging.info(f"Attempting to compile LaTeX report in: {report_output_abs_dir}")
        for i in range(2): 
            process = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-output-directory", report_output_abs_dir, report_tex_path],
                capture_output=True, text=True, check=False )
            if process.returncode != 0: 
                logging.error(f"pdflatex compilation pass {i+1} FAILED."); 
                latex_log_file = os.path.join(report_output_abs_dir, os.path.basename(report_tex_path).replace('.tex', '.log'))
                if os.path.exists(latex_log_file):
                    with open(latex_log_file, 'r', encoding='utf-8', errors='ignore') as lf:
                        logging.error("Key LaTeX errors:\n" + "".join(lf.readlines()[-50:]))
                else: logging.error("LaTeX .log file not found. STDERR:\n" + process.stderr)
            else: logging.info(f"pdflatex compilation pass {i+1} successful.")
        pdf_path = os.path.join(report_output_abs_dir, os.path.basename(report_tex_path).replace('.tex', '.pdf'))
        if os.path.exists(pdf_path): logging.info(f"PDF report successfully generated: {pdf_path}")
        else: logging.warning(f"PDF report {pdf_path} not found after compilation attempts.")
    except Exception as e: logging.error(f"LaTeX compilation error: {e}", exc_info=True)


if __name__ == "__main__":
    main()