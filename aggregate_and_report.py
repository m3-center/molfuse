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
import seaborn as sns 
import subprocess
import re
from datetime import datetime

# --- LaTeX Preamble and Helper Functions ---
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
\usepackage{subcaption} 
\usepackage{array} 
\usepackage{xcolor}
\usepackage{hyperref} 
\usepackage{fancyhdr}
\usepackage{lastpage} 
\usepackage{tocloft} 
\usepackage{enumitem}
\usepackage{rotating} 

\hypersetup{
    colorlinks=true, linkcolor=blue, filecolor=magenta, urlcolor=cyan,
    pdftitle={UMMBAS Aggregated Experiment Report}, pdfauthor={UMMBAS Project Team},
    bookmarksnumbered=true, pdfpagemode=UseOutlines
}
\renewcommand{\cftsecleader}{\cftdotfill{\cftdotsep}} 
\pagestyle{fancy} \fancyhf{} \fancyhead[L]{UMMBAS Aggregated Report} \fancyhead[R]{\today} \fancyfoot[C]{\thepage\ of \pageref{LastPage}}
\title{UMMBAS Molecular Similarity Experimental Evaluation Report (Aggregated Results)\\ \large \textit{Report Generated: \texttt{<<RUN_ID_PLACEHOLDER>>}}}
\author{UMMBAS Project Team} \date{\today}
\begin{document} \maketitle \begin{abstract}
This report presents aggregated results from multiple replicate experiments designed to evaluate the predictive capacity of molecular similarity spaces using a leave-one-target-out methodology. This study compares two molecular representations (physicochemical features and ECFP4 fingerprints), multiple dimensionality reduction (DR) techniques (PCA, UMAP, t-SNE), and two main embedding strategies (Projection vs. Co-embedding for PCA/UMAP). Performance is assessed based on the ability to rank known "held-out" active ligands for a specific target against a large set of decoy compounds. Key evaluation metrics include the Area Under the Receiver Operating Characteristic Curve (ROC-AUC), Precision-Recall Curve (PR-AUC), and Spearman's Rank Correlation ($\rho$) with experimental affinity. Results are presented as means and standard deviations over replicates to provide a statistically robust comparison of the different approaches and to investigate the interplay between molecular representation, DR method, embedding strategy, and target protein.
\end{abstract} \clearpage \tableofcontents \clearpage \listoffigures \clearpage \listoftables \clearpage
\section{Introduction}
\textit{Placeholder for Introduction text...}
\section{Methodology Overview}
\textit{Placeholder for Methodology Overview text...}
\subsection{Performance Evaluation Metrics}
\textit{Placeholder for Performance Metrics text...}
"""
LATEX_DOCUMENT_END = r"\end{document}"

def escape_latex_text_content(text_input):
    if not isinstance(text_input, str): text_input = str(text_input)
    conv = {'&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_',
            '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\^{}',
            '<': r'\textless{}', '>': r'\textgreater{}'}
    for k, v in conv.items(): text_input = text_input.replace(k, v)
    return text_input

def clean_for_label(text):
    if not isinstance(text, str): text = str(text)
    text = text.replace('_', '-').replace(' ', '-').replace('.', '-').replace('/', '-')
    text = re.sub(r'[^a-zA-Z0-9-]', '', text); text = re.sub(r'-+', '-', text) 
    return text.strip('-')[:50] 

def get_section_header_latex_standalone(level, title_text):
    sec_cmd_map = {1: r"\section", 2: r"\subsection", 3: r"\subsubsection", 4: r"\paragraph", 5: r"\subparagraph"}
    sec_cmd = sec_cmd_map.get(level, r"\paragraph")
    return f"\n{sec_cmd}{{{escape_latex_text_content(title_text)}}}\n"

def add_figure_to_latex_standalone(latex_content_list, relative_fig_path_in_tex, caption_text, label_text, placement="[H]", figure_width="0.9\\textwidth"):
    clean_label = clean_for_label(label_text)
    figure_path_for_latex = relative_fig_path_in_tex.replace(os.sep, '/')
    latex_content_list.extend([f"\\begin{{figure}}{placement}", r"  \centering",
                               f"  \\includegraphics[width={figure_width}]{{{figure_path_for_latex}}}",
                               f"  \\caption{{{escape_latex_text_content(caption_text)}}}",
                               f"  \\label{{fig:{clean_label}}}", r"\end{figure}", "\n"])

def add_dataframe_as_latex_table_standalone(latex_content_list, dataframe, caption_text, label_text, placement="[H]", col_format=None, font_size=r"\small"):
    clean_label = clean_for_label(label_text)
    if dataframe is not None and not dataframe.empty:
        latex_content_list.extend([f"\\begin{{table}}{placement}", r"  \centering"])
        if font_size: latex_content_list.append(font_size)
        latex_content_list.extend([f"  \\caption{{{escape_latex_text_content(caption_text)}}}",
                                   f"  \\label{{tab:{clean_label}}}"])
        df_for_latex = dataframe.copy()
        for col in df_for_latex.columns:
            if pd.api.types.is_numeric_dtype(df_for_latex[col]):
                df_for_latex[col] = df_for_latex[col].apply(
                    lambda x: f"{x:.3f}" if pd.notna(x) and isinstance(x, (float, np.floating)) and ( (abs(x) >= 0.001 and abs(x) < 1000) or x==0) else 
                              (f"{x:.2e}" if pd.notna(x) and isinstance(x, (float, np.floating)) else 
                              (str(int(x)) if pd.notna(x) and isinstance(x, (int, np.integer)) else ("N/A" if pd.isna(x) else str(x))) ) )
            else: 
                df_for_latex[col] = df_for_latex[col].astype(str).apply(lambda x: "N/A" if pd.isna(x) or (isinstance(x,str) and x.lower() == 'nan') else escape_latex_text_content(x))

        df_for_latex.columns = [escape_latex_text_content(str(c).replace('_', ' ').title()) for c in dataframe.columns]
        
        if col_format is None:
            # A more booktabs-friendly column format (no vertical lines)
            formats = ['l'] * len(df_for_latex.columns)
            for i, col_name_orig in enumerate(dataframe.columns):
                if pd.api.types.is_numeric_dtype(dataframe[col_name_orig]):
                    formats[i] = 'r'
            col_format = "".join(formats) # e.g., 'lrr'

        # --- THIS IS THE FIX ---
        # The 'booktabs=True' keyword argument has been removed to support older Pandas versions.
        # The default LaTeX output is already booktabs-friendly.
        latex_table_string = df_for_latex.to_latex(index=False, escape=False, column_format=col_format,
                                                   longtable=len(dataframe)>20, na_rep="N/A")
        # --- END OF FIX ---
                                                   
        latex_content_list.append(latex_table_string)
        latex_content_list.extend([r"\end{table}", "\n"])
    else: 
        latex_content_list.append(f"% Table '{escape_latex_text_content(label_text)}' is empty or None.\n")
# --- End LaTeX Helpers ---

agg_log_file_name = f"aggregation_report_generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)-8s - %(filename)-25s - %(funcName)-25s - %(lineno)-4d - %(message)s',
    handlers=[ logging.FileHandler(agg_log_file_name, mode='w'), logging.StreamHandler() ]
)

# --- Phase 1: Data Aggregation ---
def find_all_replicate_runs(base_experiment_dir):
    replicate_dirs = []
    pattern = os.path.join(base_experiment_dir, "run_seed*_repr*_*_*")
    logging.info(f"Scanning for all replicate run directories with pattern: {pattern}")
    dir_pattern_re = re.compile(r"run_seed(\d+)_repr(features|fingerprints)_(\d{8}_\d{6})")
    for dir_path in glob.glob(pattern):
        if os.path.isdir(dir_path):
            basename = os.path.basename(dir_path)
            match = dir_pattern_re.match(basename)
            if match:
                try:
                    seed_val = int(match.group(1)); repr_val = match.group(2)
                    replicate_dirs.append({"path": dir_path, "seed": seed_val, "repr_mode": repr_val})
                    logging.info(f"Found valid replicate run: {dir_path} (Seed: {seed_val}, Repr: {repr_val})")
                except (ValueError, IndexError) as e:
                    logging.warning(f"Could not parse details from directory name {basename}: {e}")
            else:
                logging.debug(f"Directory matched glob but not regex, skipping: {basename}")
    if not replicate_dirs: logging.warning(f"No valid replicate run directories found in '{base_experiment_dir}'")
    return sorted(replicate_dirs, key=lambda x: (x["seed"], x["repr_mode"]))

def collect_metrics_from_replicates(replicate_run_details, config_main):
    all_metrics_data = []
    gs = config_main['global_settings']
    for rep_info in replicate_run_details:
        rep_dir_path, replicate_seed, repr_type = rep_info["path"], rep_info["seed"], rep_info["repr_mode"]
        logging.info(f"Processing replicate: {os.path.basename(rep_dir_path)}")
        for target_info in config_main['targets']:
            if target_info.get("processing_mode", "full_analysis") == "similarity_space_only": continue
            target_results_base = os.path.join(rep_dir_path, target_info['id_name'], "results", repr_type)
            if not os.path.exists(target_results_base): continue
            for dr_key, dr_params in config_main["dimensionality_reduction_methods"].items():
                dr_short_name_base = dr_params["short_name"]
                strategies_to_check = [{"dir_leaf": dr_short_name_base.replace('-', '_'), "label": "Projection" if not dr_key == "tsne" else "Co-embedding (Native)"}]
                if gs.get("run_coembedding_for_pca_umap") and dr_params.get("allow_coembedding") and not dr_key == "tsne":
                    strategies_to_check.append({"dir_leaf": f"{dr_short_name_base}-Coembed".replace('-', '_'), "label": "Co-embedding"})
                for strategy in strategies_to_check:
                    for dim_val in gs['simspace_dims_to_test']:
                        if dr_key == "tsne" and dim_val != 2: continue
                        
                        # --- THIS IS THE CORRECTED LOGIC FOR FILENAME ---
                        # The directory is strategy-specific (e.g., PCA_Coembed)
                        strat_dir_path = os.path.join(target_results_base, f"dim_{dim_val}", strategy['dir_leaf'])
                        # The filename inside uses the BASE dr name (e.g., PCA)
                        base_dr_name_for_file = dr_short_name_base.replace('-', '_')
                        metrics_filename = f"{target_info['id_name']}_{repr_type}_{base_dr_name_for_file}_dim{dim_val}_ranking_metrics.csv"
                        metrics_file_path = os.path.join(strat_dir_path, metrics_filename)
                        # --- END CORRECTION ---

                        if os.path.exists(metrics_file_path):
                            try:
                                df_m = pd.read_csv(metrics_file_path)
                                if not df_m.empty:
                                    metric_row = {'Replicate_Seed': replicate_seed, 'Target': target_info['display_name'],
                                                  'Representation': repr_type.capitalize(), 'DR_Method': dr_short_name_base,
                                                  'Embedding_Strategy': strategy['label'], 'DIM': dim_val}
                                    column_map = {'roc_auc': 'ROC_AUC', 'pr_auc': 'PR_AUC', 'ef_1%': 'EF_1Perc',
                                                  'ef_5%': 'EF_5Perc', 'ef_10%': 'EF_10Perc', 
                                                  'spearman_rho_affinity_vs_score': 'spearman_rho_affinity_vs_score'}
                                    for csv_col, df_col in column_map.items():
                                        if csv_col in df_m.columns:
                                            value = df_m[csv_col].iloc[0]
                                            metric_row[df_col] = value if pd.notna(value) else np.nan
                                        else: metric_row[df_col] = np.nan
                                    all_metrics_data.append(metric_row)
                            except Exception as e: logging.error(f"Error reading {metrics_file_path}: {e}")
    if not all_metrics_data: logging.error("No metric data collected from any replicates.")
    return pd.DataFrame(all_metrics_data) if all_metrics_data else pd.DataFrame()


# --- Phase 2 & 3: Main Report Generation Logic ---
def main_report_generation():
    parser = argparse.ArgumentParser(description="Aggregate results and generate a unified LaTeX report.")
    parser.add_argument("--base_experiment_dir", required=True)
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--output_report_dir", required=True)
    args = parser.parse_args()

    logging.info(f"--- STARTING UNIFIED AGGREGATION AND REPORT GENERATION ---")
    with open(args.config_path, 'r') as f: config = json.load(f)
    gs = config['global_settings']

    report_run_id = f"unified_aggregated_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    report_output_abs_dir = os.path.join(args.output_report_dir, report_run_id)
    report_figures_abs_dir = os.path.join(report_output_abs_dir, "figures")
    os.makedirs(report_figures_abs_dir, exist_ok=True)
    latex_content = [LATEX_DOCUMENT_PREAMBLE.replace("<<RUN_ID_PLACEHOLDER>>", escape_latex_text_content(report_run_id))]
    replicate_details_list = find_all_replicate_runs(args.base_experiment_dir)
    if not replicate_details_list:
        logging.error("No replicate directories found. Aborting report."); return
    df_agg = collect_metrics_from_replicates(replicate_details_list, config)
    if df_agg.empty:
        logging.error("Aggregated metrics DataFrame is empty. Aborting report."); return
    df_agg.to_csv(os.path.join(report_output_abs_dir, "DEBUG_all_aggregated_metrics.csv"), index=False)
    
    # --- Overall Performance Summary ---
    latex_content.append(get_section_header_latex_standalone(1, 'Overall Comparative Summary of Ranking Performance'))
    
    # Figure 1: Main Comparison Bar Chart
    logging.info("Generating Figure 1: Main Comparison of Methods by Mean ROC-AUC...")
    try:
        df_fig1_data = df_agg.groupby(['Representation', 'DR_Method', 'Embedding_Strategy'])['ROC_AUC'].agg(['mean', 'std']).reset_index()
        df_fig1_pivot = df_fig1_data.pivot_table(index=['Representation', 'DR_Method'], columns='Embedding_Strategy', values=['mean', 'std'])
        df_fig1_pivot.columns = ['_'.join(col).strip() for col in df_fig1_pivot.columns.values]
        df_fig1_pivot.reset_index(inplace=True)
        for col in ['mean_Projection', 'std_Projection', 'mean_Co-embedding', 'std_Co-embedding', 'mean_Co-embedding (Native)', 'std_Co-embedding (Native)']:
            if col not in df_fig1_pivot.columns: df_fig1_pivot[col] = 0
        df_fig1_pivot.fillna(0, inplace=True)
        df_fig1_pivot['Plot_Label'] = df_fig1_pivot['DR_Method'] + "\n(" + df_fig1_pivot['Representation'] + ")"
        df_fig1_pivot.sort_values(by=['DR_Method', 'Representation'], inplace=True)
        
        plt.figure(figsize=(16, 8)); index = np.arange(len(df_fig1_pivot)); bar_width = 0.25
        plt.bar(index - bar_width, df_fig1_pivot['mean_Projection'], bar_width, yerr=df_fig1_pivot['std_Projection'], label='Projection', color='deepskyblue', capsize=4)
        plt.bar(index, df_fig1_pivot['mean_Co-embedding'], bar_width, yerr=df_fig1_pivot['std_Co-embedding'], label='Co-embedding', color='salmon', capsize=4)
        plt.bar(index + bar_width, df_fig1_pivot['mean_Co-embedding (Native)'], bar_width, yerr=df_fig1_pivot['std_Co-embedding (Native)'], label='Co-embedding (tSNE Native)', color='mediumseagreen', capsize=4)
        plt.xlabel("DR Method (Representation)"); plt.ylabel("Mean ROC-AUC (across Replicates, Targets, DIMs)")
        plt.title("Overall Comparison of Methods and Strategies by Mean ROC-AUC")
        plt.xticks(index, df_fig1_pivot['Plot_Label'], rotation=45, ha="right"); plt.ylim(bottom=0)
        plt.legend(title="Embedding Strategy"); plt.grid(True, linestyle='--', axis='y'); plt.tight_layout()
        
        fig1_path = os.path.join(report_figures_abs_dir, "fig1_overall_comparison.png")
        plt.savefig(fig1_path); plt.close()
        add_figure_to_latex_standalone(latex_content, os.path.join("figures", "fig1_overall_comparison.png"),
                                     "Overall comparison of methods. Bars show mean ROC-AUC averaged over all targets, dimensions, and replicates. Error bars represent standard deviation.",
                                     "fig-overall-comparison")
    except Exception as e:
        logging.error(f"Failed to generate Figure 1: {e}", exc_info=True); latex_content.append("Figure 1 could not be generated.\n")

    # Table 1: Main Summary Table
    logging.info("Generating Table 1: Main Summary of Aggregated Metrics...")
    try:
        df_table1_agg = df_agg.groupby(['Representation', 'DR_Method', 'Embedding_Strategy']).agg(
            Mean_ROC_AUC=('ROC_AUC', 'mean'), SD_ROC_AUC=('ROC_AUC', 'std'),
            Mean_PR_AUC=('PR_AUC', 'mean'), SD_PR_AUC=('PR_AUC', 'std'),
            Mean_Spearman_Rho=('spearman_rho_affinity_vs_score', 'mean'), SD_Spearman_Rho=('spearman_rho_affinity_vs_score', 'std')
        ).reset_index()
        df_table1_agg.fillna({'SD_ROC_AUC':0, 'SD_PR_AUC':0, 'SD_Spearman_Rho':0}, inplace=True)
        df_table1_agg['ROC_AUC'] = df_table1_agg.apply(lambda r: f"{r['Mean_ROC_AUC']:.3f} $\\pm$ {r['SD_ROC_AUC']:.3f}" if pd.notna(r['Mean_ROC_AUC']) else "N/A", axis=1)
        df_table1_agg['PR_AUC'] = df_table1_agg.apply(lambda r: f"{r['Mean_PR_AUC']:.3f} $\\pm$ {r['SD_PR_AUC']:.3f}" if pd.notna(r['Mean_PR_AUC']) else "N/A", axis=1)
        df_table1_agg['Spearman_Rho'] = df_table1_agg.apply(lambda r: f"{r['Mean_Spearman_Rho']:.3f} $\\pm$ {r['SD_Spearman_Rho']:.3f}" if pd.notna(r['Mean_Spearman_Rho']) else "N/A", axis=1)
        
        df_table1_display = df_table1_agg[['Representation', 'DR_Method', 'Embedding_Strategy', 'ROC_AUC', 'PR_AUC', 'Spearman_Rho']].copy()
        df_table1_display.sort_values(by=['Representation','DR_Method','Embedding_Strategy'], inplace=True)
        add_dataframe_as_latex_table_standalone(latex_content, df_table1_display,
            "Aggregated Performance Metrics (Mean $\\pm$ SD). Values are averaged over all replicates, targets, and dimensions.",
            "tab-main-summary", font_size=r"\\scriptsize")
    except Exception as e:
        logging.error(f"Failed to generate Table 1: {e}", exc_info=True); latex_content.append("Table 1 could not be generated.\n")
    
    # --- Detailed Analysis ---
    latex_content.append(f"\\clearpage\n{get_section_header_latex_standalone(1, 'Detailed Analysis')}")

    # Figure 2: Performance vs Dimension
    logging.info("Generating Figure 2: Performance vs. Dimension...")
    try:
        dr_methods_to_plot = sorted(df_agg['DR_Method'].unique())
        num_dr_methods = len(dr_methods_to_plot)
        fig2, axes2 = plt.subplots(num_dr_methods, 2, figsize=(14, 4 * num_dr_methods), sharex=True, sharey=True, squeeze=False)
        fig2.suptitle("Performance vs. Similarity Space Dimension", fontsize=14, y=1.0)

        for i, dr_method in enumerate(dr_methods_to_plot):
            for j, repr_type in enumerate(['Features', 'Fingerprints']):
                ax = axes2[i, j]
                df_subset = df_agg[(df_agg['DR_Method'] == dr_method) & (df_agg['Representation'] == repr_type)]
                if not df_subset.empty:
                    for strat_label in sorted(df_subset['Embedding_Strategy'].unique()):
                        strat_metrics = df_subset[df_subset['Embedding_Strategy'] == strat_label]
                        dim_summary = strat_metrics.groupby('DIM')['ROC_AUC'].agg(['mean', 'std']).reset_index().fillna(0)
                        ax.plot(dim_summary['DIM'], dim_summary['mean'], marker='o', linestyle='-', label=strat_label)
                        ax.fill_between(dim_summary['DIM'], dim_summary['mean'] - dim_summary['std'], dim_summary['mean'] + dim_summary['std'], alpha=0.2)
                ax.set_title(f"{dr_method} ({repr_type})"); ax.grid(True, linestyle=':'); ax.legend()
                if i == 0: ax.set_ylim(0, 1.05)
                if i == num_dr_methods - 1: ax.set_xlabel("SIMSPACE_DIM")
                if j == 0: ax.set_ylabel("Mean ROC-AUC")
        
        fig2.tight_layout(rect=[0, 0, 1, 0.98])
        fig2_path = os.path.join(report_figures_abs_dir, "fig2_perf_vs_dim.png")
        plt.savefig(fig2_path); plt.close()
        add_figure_to_latex_standalone(latex_content, os.path.join("figures", "fig2_perf_vs_dim.png"),
            "Mean ROC-AUC vs. Dimension, stratified by Representation and DR Method. Shaded areas represent $\\pm$1 SD across replicates and targets.",
            "fig-perf-vs-dim", figure_width="\\textwidth")
    except Exception as e:
        logging.error(f"Failed to generate Figure 2: {e}", exc_info=True); latex_content.append("Figure 2 could not be generated.\n")

    # Table 2: Best Config per Target
    logging.info("Generating Table 2: Best Config per Target...")
    try:
        mean_roc_per_config = df_agg.groupby(['Target', 'Representation', 'DR_Method', 'Embedding_Strategy', 'DIM'])['ROC_AUC'].mean().reset_index()
        idx = mean_roc_per_config.groupby(['Target'])['ROC_AUC'].idxmax()
        df_best_per_target = mean_roc_per_config.loc[idx]
        df_best_per_target_std = pd.merge(
            df_best_per_target,
            df_agg.groupby(['Target', 'Representation', 'DR_Method', 'Embedding_Strategy', 'DIM'])['ROC_AUC'].std().reset_index().rename(columns={'ROC_AUC': 'SD_ROC_AUC'}),
            on=['Target', 'Representation', 'DR_Method', 'Embedding_Strategy', 'DIM'], how='left'
        ).fillna(0)
        df_best_per_target_std['Mean_ROC_AUC'] = df_best_per_target_std.apply(lambda r: f"{r['ROC_AUC']:.3f} $\\pm$ {r['SD_ROC_AUC']:.3f}", axis=1)
        df_best_per_target_display = df_best_per_target_std[['Target','Representation','DR_Method','Embedding_Strategy','DIM','Mean_ROC_AUC']].copy()
        df_best_per_target_display.rename(columns={'DIM': 'Optimal_Dimension'}, inplace=True)
        add_dataframe_as_latex_table_standalone(latex_content, df_best_per_target_display,
            "Best performing configuration for each target, based on highest mean ROC-AUC across replicates.",
            "tab-best-per-target", font_size=r"\\tiny")
    except Exception as e:
        logging.error(f"Failed to generate Table 2: {e}", exc_info=True); latex_content.append("Table 2 could not be generated.\n")

    # Figure 3: Heatmap of Performance
    logging.info("Generating Figure 3: Heatmap of Performance...")
    try:
        heatmap_data = df_agg.groupby(['Target', 'Representation', 'DR_Method', 'Embedding_Strategy'])['ROC_AUC'].mean().reset_index()
        heatmap_pivot = heatmap_data.pivot_table(
            index=['Target', 'Representation'],
            columns=['DR_Method', 'Embedding_Strategy'],
            values='ROC_AUC'
        )
        plt.figure(figsize=(16, 10))
        sns.heatmap(heatmap_pivot, annot=True, fmt=".3f", cmap="viridis", linewidths=.5, annot_kws={"size": 8})
        plt.title("Heatmap of Best Mean ROC-AUC by Target, Representation, and Method")
        plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0); plt.tight_layout()
        fig3_path = os.path.join(report_figures_abs_dir, "fig3_heatmap.png")
        plt.savefig(fig3_path); plt.close()
        add_figure_to_latex_standalone(latex_content, os.path.join("figures", "fig3_heatmap.png"),
            "Heatmap of mean ROC-AUC performance. Each cell shows the highest mean ROC-AUC achieved for that configuration, maximized over all tested dimensions.",
            "fig-heatmap", figure_width="\\textwidth")
    except Exception as e:
        logging.error(f"Failed to generate Figure 3: {e}", exc_info=True); latex_content.append("Figure 3 could not be generated.\n")

    
    # --- Illustrative Results for Representative Targets ---
    latex_content.append(f"\\clearpage\n{get_section_header_latex_standalone(1, 'Illustrative Results for Representative Targets')}")
    representative_replicate_info = replicate_details_list[0]
    logging.info(f"Using replicate run '{os.path.basename(representative_replicate_info['path'])}' for illustrative 2D plots.")
    
    for target_info in config['targets']:
        if target_info.get("processing_mode", "full_analysis") == "similarity_space_only": continue
        target_id_name, target_display_name = target_info['id_name'], target_info['display_name']
        latex_content.append(f"\\clearpage\n{get_section_header_latex_standalone(2, f'Target: {target_display_name}')}")
        
        for repr_type in ["features", "fingerprints"]:
            latex_content.append(get_section_header_latex_standalone(3, f"Representation: {repr_type.capitalize()}"))
            rep_results_base = os.path.join(representative_replicate_info['path'], target_id_name, "results", repr_type, "dim_2")
            if not os.path.exists(rep_results_base):
                latex_content.append(f"No 2D results found in representative replicate for this target/representation.\n"); continue
            
            plots_added_for_repr = False
            for dr_key, dr_params in config["dimensionality_reduction_methods"].items():
                if dr_key == 'tsne' and '2' not in gs.get('simspace_dims_to_test', []): continue
                
                strategies_to_plot = [{"dir_leaf": dr_params["short_name"].replace('-', '_'), "title": dr_params["short_name"]}]
                if gs.get("run_coembedding_for_pca_umap") and dr_params.get("allow_coembedding") and not dr_key == "tsne":
                    strategies_to_plot.append({"dir_leaf": f"{dr_params['short_name']}-Coembed".replace('-', '_'), "title": f"{dr_params['short_name']}-Coembed"})
                
                for strategy in strategies_to_plot:
                    strat_dir_path = os.path.join(rep_results_base, strategy['dir_leaf'])
                    if os.path.exists(strat_dir_path):
                        # --- CORRECTED FILENAME LOGIC FOR PLOTS ---
                        # The filename inside uses the BASE dr name (e.g., PCA) even if directory is PCA_Coembed
                        base_dr_name_for_plot_file = dr_params["short_name"].replace('-', '_')
                        scatter_fname_orig = f"{target_id_name}_{repr_type}_{base_dr_name_for_plot_file}_dim2_scatter.png"
                        hist_fname_orig = f"{target_id_name}_{repr_type}_{base_dr_name_for_plot_file}_dim2_min_distances_hist_ACTIVES.png"
                        # --- END CORRECTION ---
                        
                        scatter_src = os.path.join(strat_dir_path, scatter_fname_orig)
                        hist_src = os.path.join(strat_dir_path, hist_fname_orig)

                        if os.path.exists(scatter_src) or os.path.exists(hist_src):
                            plots_added_for_repr = True
                            latex_content.append(get_section_header_latex_standalone(4, f"DR Method: {strategy['title']}"))
                            if os.path.exists(scatter_src):
                                clean_fname_s = f"{clean_for_label(target_id_name)}-{repr_type}-{clean_for_label(strategy['title'])}-scatter.png"
                                shutil.copy(scatter_src, os.path.join(report_figures_abs_dir, clean_fname_s))
                                add_figure_to_latex_standalone(latex_content, os.path.join("figures", clean_fname_s),
                                    f"2D Similarity Space for {target_display_name} ({strategy['title']}).",
                                    f"fig-scatter-{target_id_name}-{repr_type}-{strategy['title']}")
                            if os.path.exists(hist_src):
                                clean_fname_h = f"{clean_for_label(target_id_name)}-{repr_type}-{clean_for_label(strategy['title'])}-hist.png"
                                shutil.copy(hist_src, os.path.join(report_figures_abs_dir, clean_fname_h))
                                add_figure_to_latex_standalone(latex_content, os.path.join("figures", clean_fname_h),
                                    f"Histogram of min. distances for Actives of {target_display_name} ({strategy['title']}).",
                                    f"fig-hist-{target_id_name}-{repr_type}-{strategy['title']}", figure_width="0.6\\textwidth")
            if not plots_added_for_repr:
                latex_content.append("No 2D plots found for this representation in the representative replicate.\n")

    # --- Similarity Space Only Targets ---
    latex_content.append(f"\\clearpage\n{get_section_header_latex_standalone(1, 'Targets Processed for Docking Only')}")
    sim_space_only_targets = [t for t in config['targets'] if t.get("processing_mode") == "similarity_space_only"]
    if sim_space_only_targets:
        latex_content.append("The following targets were processed in 'similarity_space_only' mode due to limited known ligand data in ChEMBL. For these targets, similarity spaces and ranked lists of ZINC compounds were generated for subsequent docking studies, but comparative performance metrics are not applicable.\n\\begin{itemize}")
        for target_info in sim_space_only_targets:
            latex_content.append(f"  \\item {escape_latex_text_content(target_info['display_name'])}\n")
        latex_content.append("\\end{itemize}\n")
    else:
        latex_content.append("All targets were processed in 'full_analysis' mode.\n")

    latex_content.append(LATEX_DOCUMENT_END)

    # Save and compile LaTeX
    report_tex_filename = f"aggregated_unified_report.tex"
    report_tex_path = os.path.join(report_output_abs_dir, report_tex_filename)
    with open(report_tex_path, "w", encoding='utf-8') as f: f.write("\n".join(latex_content))
    logging.info(f"Aggregated LaTeX report structure generated: {report_tex_path}")
    
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
    main_report_generation()