import pandas as pd
import numpy as np
import os
import argparse
import logging
import json
import shutil
from collections import defaultdict
import matplotlib.pyplot as plt
import subprocess
import re # For cleaning labels

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)-8s - %(filename)-25s - %(funcName)-25s - %(message)s',
                    handlers=[logging.FileHandler("generate_standalone_report.log"), logging.StreamHandler()])

# --- Standalone LaTeX Document Shell ---
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
% \usepackage{booktabs} % REMOVED as per instruction
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

\hypersetup{
    colorlinks=true, linkcolor=blue, filecolor=magenta, urlcolor=cyan,
    pdftitle={UMMBAS Similarity Experiment Report}, pdfauthor={UMMBAS Project Team},
    pdfsubject={Molecular Similarity Analysis}, pdfkeywords={UMMBAS, Cheminformatics, Similarity},
    bookmarksnumbered=true, pdfpagemode=UseOutlines
}

\renewcommand{\cftsecleader}{\cftdotfill{\cftdotsep}} 

\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{UMMBAS Experimental Report}
\fancyhead[R]{\today}
\fancyfoot[C]{\thepage\ of \pageref{LastPage}}

\title{UMMBAS Molecular Similarity Experimental Evaluation Report\\ \large \textit{Experimental Run: \texttt{<<RUN_ID_PLACEHOLDER>>}}}
\author{UMMBAS Project Team}
\date{\today}

\begin{document}
\maketitle
\begin{abstract}
This report details experiments conducted to evaluate the predictive capacity of molecular similarity spaces using a leave-one-target-out inspired methodology. The central hypothesis posits that ligands active against a specific protein target will exhibit proximity to compounds known to interact with other proteins sharing the same broader molecular function, even when the specific target's known ligands are excluded during the construction of the similarity space. Experiments encompassed multiple target proteins, two molecular representations (physicochemical features and ECFP4 fingerprints), various dimensionality reduction (DR) techniques (PCA; UMAP with Euclidean, Cosine, Manhattan, and Hamming metrics; and t-SNE with pre-PCA), and a range of similarity space dimensionalities. Evaluation primarily focused on rank-based metrics (ROC-AUC, PR-AUC, Enrichment Factors) derived from the proximity of projected known target ligands to the molecular function (MF) cloud relative to decoy compounds, as well as correlation with experimental affinity data where available.
\end{abstract}
\clearpage
\tableofcontents
\clearpage
\listoffigures
\clearpage
\listoftables
\clearpage

\section{Introduction}
The exploration of chemical space for novel therapeutic agents is a cornerstone of drug discovery. Molecular similarity, a fundamental concept in cheminformatics, suggests that structurally similar molecules are likely to exhibit similar biological activities. This principle underpins many virtual screening and lead optimization strategies. This study investigates the utility of constructing and analyzing molecular similarity spaces to predict potential interactions between small molecules and protein targets, grouped by their shared molecular function.

The experimental design aims to rigorously test whether general molecular function similarity can guide the identification of ligands for a specific, "held-out" protein target. By systematically varying data representations, dimensionality reduction methods, and the dimensionality of the resulting spaces, we seek to identify optimal parameters and assess the overall robustness of this similarity-based approach. Key metrics involve measuring the proximity of known active ligands (for the held-out target) to the cloud of compounds associated with the broader molecular function after projection into these tailored similarity spaces, and evaluating their rank relative to decoy compounds.

\section{Methodology Overview}
The experimental methodology involved several key stages: dataset preparation (including segregation of held-out target actives and creation of ``blinded'' datasets), descriptor calculation for the held-out actives, construction of similarity spaces using various dimensionality reduction techniques and parameters, projection (or co-embedding) of held-out actives, and finally, performance evaluation using rank-based metrics. Each step was automated and configured via a central JSON file. The primary software components involved were Python scripts leveraging libraries such as RDKit, Mordred, Pandas, scikit-learn, Matplotlib, Seaborn, and optionally cuML for GPU acceleration. 

\subsection{Performance Evaluation Metrics}
\label{subsec:perf_metrics}
To assess the ability of each similarity space configuration to effectively ``rediscover'' the held-out target active ligands among a set of decoy compounds (derived from the ZINC database), several rank-based metrics were employed. Compounds were scored based on their proximity (primarily minimum Euclidean distance) to the Molecular Function (MF) Cloud, with smaller distances yielding higher scores. The held-out target ligands were labeled as ``actives'' and ZINC compounds as ``decoys.''

\begin{itemize}[leftmargin=*]
    \item \textbf{Receiver Operating Characteristic Area Under Curve (ROC-AUC):}
    The ROC curve plots the True Positive Rate (TPR, sensitivity) against the False Positive Rate (FPR, 1-specificity) at various ranking thresholds. The Area Under this Curve (AUC) provides a single measure of the model's ability to discriminate between active and decoy compounds across all thresholds. A ROC-AUC of 1.0 represents a perfect classifier, while 0.5 indicates random performance.
    Formally, $TPR = \frac{TP}{TP+FN}$ and $FPR = \frac{FP}{FP+TN}$, where TP, FP, TN, FN are True Positives, False Positives, True Negatives, and False Negatives, respectively.

    \item \textbf{Precision-Recall Area Under Curve (PR-AUC):}
    The Precision-Recall curve plots Precision against Recall (TPR) at various ranking thresholds. This metric is particularly informative for imbalanced datasets, common in virtual screening where decoys vastly outnumber actives. A higher PR-AUC indicates better performance.
    Formally, $Precision = \frac{TP}{TP+FP}$ and $Recall = TPR$.

    \item \textbf{Enrichment Factor (EF) at x\%:}
    The Enrichment Factor measures how many more active compounds are found in the top x\% of a ranked list compared to a random selection. It is defined as:
    $$ EF_{x\%} = \frac{\text{Actives found in top x\%} / \text{Compounds in top x\%}}{\text{Total Actives} / \text{Total Compounds}} $$
    EF values greater than 1 indicate enrichment. In this study, EF at 1\%, 5\%, and 10\% of the ranked list were calculated.

    \item \textbf{Spearman's Rank Correlation ($\rho$) with Affinity:}
    For held-out target actives where quantitative bioactivity data (e.g., pIC50, pKi, derived from 'Standard Value (nM)') was available, Spearman's rank correlation coefficient ($\rho$) was calculated. This assessed the monotonic relationship between the rank of an active compound (derived from its proximity score to the MF Cloud) and its experimentally determined affinity. A statistically significant positive correlation (given scores are based on `-distance`, and pActivity increases with potency) would suggest that compounds ranked higher by the similarity model also tend to be more potent.
\end{itemize}
The following sections detail the results obtained for each target protein based on these metrics.
"""

LATEX_DOCUMENT_END = r"""
\end{document}
"""

# --- LaTeX Helper Functions ---
def escape_latex_text_content(text_input):
    if not isinstance(text_input, str): text_input = str(text_input)
    conv = {'&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_',
            '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\^{}',
            '<': r'\textless{}', '>': r'\textgreater{}'}
    # Python's str.replace is not iterative by default, so this order is fine.
    # Backslash is intentionally not escaped here, as it's usually for LaTeX commands.
    # If a literal backslash character from input data needs to be printed, 
    # it would need specific handling (e.g. text_input.replace('\\', r'\textbackslash{}')).
    # For now, assume input text doesn't contain meaningful single backslashes to be printed as text.
    for k, v in conv.items():
        text_input = text_input.replace(k, v)
    return text_input

def clean_for_label(text):
    if not isinstance(text, str): text = str(text)
    text = text.replace('_', '-').replace(' ', '-').replace('.', '-').replace('/', '-')
    text = re.sub(r'[^a-zA-Z0-9-]', '', text) # Keep only alphanumeric and hyphen
    text = re.sub(r'-+', '-', text) # Replace multiple hyphens with single
    return text.strip('-')[:50] # Limit length for sanity

def get_section_header_latex_standalone(level, title_text):
    sec_cmd_map = {1: r"\section", 2: r"\subsection", 3: r"\subsubsection", 4: r"\paragraph"}
    sec_cmd = sec_cmd_map.get(level, r"\paragraph")
    return f"\n{sec_cmd}{{{escape_latex_text_content(title_text)}}}\n"

def add_figure_to_latex_standalone(latex_content_list, relative_fig_path_in_tex, caption_text, label_text, placement="[htbp]", figure_width="0.7\\linewidth"):
    clean_label = clean_for_label(label_text)
    # Ensure figure path uses forward slashes for LaTeX, even if OS uses backslashes
    figure_path_for_latex = relative_fig_path_in_tex.replace(os.sep, '/')
    latex_content_list.append(f"\\begin{{figure}}{placement}")
    latex_content_list.append(r"  \centering")
    latex_content_list.append(f"  \\includegraphics[width={figure_width}]{{{figure_path_for_latex}}}")
    latex_content_list.append(f"  \\caption{{{escape_latex_text_content(caption_text)}}}")
    latex_content_list.append(f"  \\label{{fig:{clean_label}}}")
    latex_content_list.append(r"\end{figure}")
    latex_content_list.append("\n")

def add_dataframe_as_latex_table_standalone(latex_content_list, dataframe, caption_text, label_text, placement="[htbp]", col_format=None, font_size=r"\small"):
    clean_label = clean_for_label(label_text)
    if dataframe is not None and not dataframe.empty:
        latex_content_list.append(f"\\begin{{table}}{placement}")
        latex_content_list.append(r"  \centering")
        if font_size: latex_content_list.append(font_size)
        latex_content_list.append(f"  \\caption{{{escape_latex_text_content(caption_text)}}}")
        latex_content_list.append(f"  \\label{{tab:{clean_label}}}")
        
        df_for_latex = dataframe.copy()
        for col in df_for_latex.select_dtypes(include=np.number).columns:
            df_for_latex[col] = df_for_latex[col].apply(
                lambda x: f"{x:.3f}" if pd.notna(x) and abs(x) >= 0.001 and abs(x) < 1000 else (f"{x:.2e}" if pd.notna(x) else "N/A")
            )
        
        df_for_latex.columns = [escape_latex_text_content(col.replace('_', ' ').title()) for col in df_for_latex.columns]
        
        if col_format is None:
            formats = ['l'] * len(df_for_latex.columns) 
            for i, col_name_orig in enumerate(dataframe.columns): 
                if pd.api.types.is_numeric_dtype(dataframe[col_name_orig]): formats[i] = 'r' 
            col_format = '|' + '|'.join(formats) + '|' # Default: left-aligned text, right-aligned numbers

        # REMOVED booktabs=True from here
        latex_content_list.append(df_for_latex.to_latex(index=False, escape=False, 
                                                        column_format=col_format,
                                                        longtable=isinstance(dataframe, pd.DataFrame) and len(dataframe)>20))
        latex_content_list.append(r"\end{table}")
        latex_content_list.append("\n")
    else:
        latex_content_list.append(f"% Table data for '{escape_latex_text_content(label_text)}' is empty or None.\n")

# --- Main Report Generation Logic ---
def main():
    parser = argparse.ArgumentParser(description="Generate Standalone LaTeX report from experiment results.")
    parser.add_argument("--experiment_run_dir", required=True, help="Path to the timestamped main experiment run directory.")
    parser.add_argument("--config_path", required=True, help="Path to the experiment_config.json file.")
    parser.add_argument("--output_dir", required=True, help="Directory to save the LaTeX report and figures subfolder.")
    args = parser.parse_args()

    with open(args.config_path, 'r') as f: config = json.load(f)
    gs = config['global_settings']
    
    run_id_for_display = escape_latex_text_content(os.path.basename(args.experiment_run_dir)) 
    preamble_filled = LATEX_DOCUMENT_PREAMBLE.replace("<<RUN_ID_PLACEHOLDER>>", run_id_for_display)
    latex_content = [preamble_filled]

    report_filename_base = clean_for_label(os.path.basename(args.experiment_run_dir)) + "-standalone-report"
    report_output_abs_dir = os.path.join(args.output_dir, report_filename_base) 
    report_figures_abs_dir = os.path.join(report_output_abs_dir, "figures") 
    os.makedirs(report_figures_abs_dir, exist_ok=True)
    
    latex_content.append(get_section_header_latex_standalone(1, "Experimental Results by Target Protein"))

    all_targets_ranking_metrics = [] 
    all_targets_dim_opt_data = [] # For storing (Target, Repr, DR, Optimal_DIM, Best_ROC_AUC)

    for target_info in config['targets']:
        target_id_name = target_info['id_name']
        target_display_name = target_info['display_name'] 
        mf_display_name = target_info['molecular_function_display_name']
        target_label_name = clean_for_label(target_id_name)
        
        latex_content.append(f"\\clearpage\n{get_section_header_latex_standalone(2, f'Target: {target_display_name} (MF: {mf_display_name})')}")
        target_results_base_dir = os.path.join(args.experiment_run_dir, target_id_name, "results")

        if not os.path.exists(target_results_base_dir):
            latex_content.append(f"Results data not found for target {escape_latex_text_content(target_id_name)}.\n")
            continue

        latex_content.append(get_section_header_latex_standalone(3, "Optimization of Similarity Space Dimensionality"))
        latex_content.append("The impact of similarity space dimensionality on ranking performance (ROC-AUC based on proximity to the MF Cloud) was assessed. "
                             f"Dimensionalities tested for PCA and UMAP were: {escape_latex_text_content(str(gs['simspace_dims_to_test']))}. t-SNE was evaluated only at 2 dimensions.\n")

        current_target_dim_opt_summary_rows = [] 

        for repr_type in config['representations']:
            latex_content.append(get_section_header_latex_standalone(4, f"Representation: {repr_type.capitalize()}"))
            repr_label = clean_for_label(repr_type)
            
            for dr_key, dr_params in config["dimensionality_reduction_methods"].items():
                dr_display_name = dr_params["short_name"]
                dr_short_name_fs = dr_display_name.replace('-', '_')
                dr_label = clean_for_label(dr_short_name_fs)
                dim_vs_roc_auc_data = [] 

                for simspace_dim_val in gs['simspace_dims_to_test']:
                    if dr_key == "tsne" and simspace_dim_val != 2: continue

                    results_path_for_dim_dr = os.path.join(target_results_base_dir, repr_type, f"dim_{simspace_dim_val}", dr_short_name_fs)
                    ranking_metrics_csv = os.path.join(results_path_for_dim_dr, f"{target_id_name}_{repr_type}_{dr_short_name_fs}_dim{simspace_dim_val}_ranking_metrics.csv")
                    
                    if os.path.exists(ranking_metrics_csv):
                        try:
                            df_metrics = pd.read_csv(ranking_metrics_csv)
                            if not df_metrics.empty and 'roc_auc' in df_metrics.columns:
                                current_roc_auc = df_metrics['roc_auc'].iloc[0]
                                if pd.notna(current_roc_auc): 
                                    dim_vs_roc_auc_data.append({'dim': simspace_dim_val, 'roc_auc': current_roc_auc})
                                
                                metrics_to_add = {
                                    'Target': target_display_name, 'Representation': repr_type.capitalize(), 
                                    'DR_Method': dr_display_name, 'DIM': simspace_dim_val,
                                    'ROC_AUC': current_roc_auc
                                }
                                # Use .get with a default Series of NaNs to avoid errors if column is missing
                                for metric_key_csv, metric_key_dict in [
                                    ('pr_auc', 'PR_AUC'), ('ef_1%', 'EF_1Perc'), 
                                    ('ef_5%', 'EF_5Perc'), ('ef_10%', 'EF_10Perc'), 
                                    ('spearman_rho_affinity_vs_score', 'Spearman_Rho')]:
                                    
                                    series_data = df_metrics.get(metric_key_csv, pd.Series([np.nan])) # Default to Series with NaN
                                    value_to_add = series_data.iloc[0] if not series_data.empty and not series_data.isna().all() else np.nan
                                    metrics_to_add[metric_key_dict] = value_to_add
                                all_targets_ranking_metrics.append(metrics_to_add)
                        except Exception as e: logging.warning(f"Could not process ranking metrics CSV {ranking_metrics_csv}: {e}")
                
                if dim_vs_roc_auc_data: 
                    df_plot = pd.DataFrame(dim_vs_roc_auc_data).sort_values(by='dim')
                    plt.figure(figsize=(8, 5))
                    plt.plot(df_plot['dim'], df_plot['roc_auc'], marker='o', linestyle='-')
                    plt.xlabel("SIMSPACE_DIM Value") 
                    plt.ylabel("ROC-AUC (vs. MF Cloud Proximity)")
                    plt.title(f"SIMSPACE_DIM vs. ROC-AUC for {target_display_name}\n({repr_type.capitalize()}, {dr_display_name})", fontsize=11) 
                    if dr_key != "tsne": plt.xticks(gs['simspace_dims_to_test'])
                    else: plt.xticks([2]) 
                    plt.ylim(0.0, 1.05) 
                    plt.grid(True, linestyle='--', alpha=0.6)
                    
                    plot_filename = f"{target_label_name}-{repr_label}-{dr_label}-dim-vs-rocauc.png"
                    plot_abs_path_dest = os.path.join(report_figures_abs_dir, plot_filename)
                    try:
                        plt.savefig(plot_abs_path_dest, dpi=150, bbox_inches='tight')
                        plot_relative_path_for_tex = os.path.join("figures", plot_filename).replace(os.sep, '/')
                        caption_text = f"ROC-AUC score versus SIMSPACE_DIM for target {target_display_name} ({repr_type.capitalize()}, DR: {dr_display_name})."
                        fig_label = f"{target_label_name}-{repr_label}-{dr_label}-dimopt-rocauc"
                        add_figure_to_latex_standalone(latex_content, plot_relative_path_for_tex, caption_text, fig_label)
                    except Exception as e: logging.error(f"Failed to save dim_opt plot {plot_abs_path_dest}: {e}")
                    plt.close()

                    if not df_plot.empty:
                        best_row = df_plot.loc[df_plot['roc_auc'].idxmax()]
                        current_target_dim_opt_summary_rows.append({
                            'Representation': repr_type.capitalize(), 'DR_Method': dr_display_name,
                            'Optimal_DIM_by_ROC_AUC': int(best_row['dim']), 
                            'Best_ROC_AUC': best_row['roc_auc']
                        })
            # Removed premature clearpage to group all DR method plots for a representation
            if dim_vs_roc_auc_data : latex_content.append("\\clearpage\n") # Add clearpage if plots were made for this representation's DR methods
        
        if current_target_dim_opt_summary_rows:
            df_summary = pd.DataFrame(current_target_dim_opt_summary_rows)
            latex_content.append(get_section_header_latex_standalone(3, f"Summary of Optimal Dimensionality (by ROC-AUC) for {target_display_name}"))
            add_dataframe_as_latex_table_standalone(latex_content, df_summary,
                                         f"Optimal SIMSPACE_DIM (maximizing ROC-AUC) and corresponding ROC-AUC for target {target_display_name}.",
                                         f"opt-dim-summary-{target_label_name}")
        latex_content.append("\\clearpage\n")

        latex_content.append(get_section_header_latex_standalone(2, "Illustrative 2D Projections (SIMSPACE_DIM = 2)"))
        dim_2_exists_for_target_report = False
        for repr_type_2d in config['representations']:
            repr_has_2d_plot_report = False
            repr_label_2d = clean_for_label(repr_type_2d)
            for dr_key_2d, dr_params_2d in config["dimensionality_reduction_methods"].items():
                dr_display_name_2d = dr_params_2d["short_name"]
                dr_short_name_fs_2d = dr_display_name_2d.replace('-', '_')
                dr_label_2d = clean_for_label(dr_short_name_fs_2d)
                
                dim2_results_path_report = os.path.join(target_results_base_dir, repr_type_2d, "dim_2", dr_short_name_fs_2d)
                if os.path.exists(dim2_results_path_report):
                    if not repr_has_2d_plot_report:
                        latex_content.append(get_section_header_latex_standalone(3, f"Representation: {repr_type_2d.capitalize()} (2D Projections)"))
                        repr_has_2d_plot_report = True
                        dim_2_exists_for_target_report = True

                    latex_content.append(get_section_header_latex_standalone(4, f"DR Method: {dr_display_name_2d} (2D)"))
                    
                    scatter_orig_fname = f"{target_id_name}_{repr_type_2d}_{dr_short_name_fs_2d}_dim2_scatter.png"
                    hist_orig_fname = f"{target_id_name}_{repr_type_2d}_{dr_short_name_fs_2d}_dim2_min_distances_hist_ACTIVES.png"
                    scatter_src_report = os.path.join(dim2_results_path_report, scatter_orig_fname)
                    hist_src_report = os.path.join(dim2_results_path_report, hist_orig_fname)
                    
                    scatter_clean_fname = f"{target_label_name}-{repr_label_2d}-{dr_label_2d}-dim2-scatter.png"
                    hist_clean_fname = f"{target_label_name}-{repr_label_2d}-{dr_label_2d}-dim2-hist-actives.png" # Be specific

                    if os.path.exists(scatter_src_report):
                        shutil.copy(scatter_src_report, os.path.join(report_figures_abs_dir, scatter_clean_fname))
                        fig_path_tex = os.path.join("figures", scatter_clean_fname).replace(os.sep, '/')
                        caption_s = f"2D Similarity space for {target_display_name} ({repr_type_2d.capitalize()}, {dr_display_name_2d})."
                        add_figure_to_latex_standalone(latex_content, fig_path_tex, caption_s, f"{target_label_name}-{repr_label_2d}-{dr_label_2d}-scatter2D")

                    if os.path.exists(hist_src_report):
                        shutil.copy(hist_src_report, os.path.join(report_figures_abs_dir, hist_clean_fname))
                        fig_path_tex = os.path.join("figures", hist_clean_fname).replace(os.sep, '/')
                        caption_h = f"Histogram of minimum distances for ACTIVES of {target_display_name} ({repr_type_2d.capitalize()}, {dr_display_name_2d}, 2D)."
                        add_figure_to_latex_standalone(latex_content, fig_path_tex, caption_h, f"{target_label_name}-{repr_label_2d}-{dr_label_2d}-hist2D-actives", figure_width="0.65\\textwidth")
            if repr_has_2d_plot_report: latex_content.append("\\clearpage\n")
        if not dim_2_exists_for_target_report:
            latex_content.append("No 2D projection results were found to display for this target.\n")


    latex_content.append(f"\\clearpage\n{get_section_header_latex_standalone(1, 'Overall Comparative Summary of Ranking Performance')}")
    if all_targets_ranking_metrics:
        df_all_metrics_summary = pd.DataFrame(all_targets_ranking_metrics)
        
        df_summary_dim2 = df_all_metrics_summary[df_all_metrics_summary['DIM'] == 2].copy()
        if not df_summary_dim2.empty:
            # Corrected column names to match what's in all_targets_ranking_metrics
            cols_for_groupby_dim2 = ['ROC_AUC', 'PR_AUC', 'EF_1Perc', 'EF_5Perc', 'EF_10Perc', 'Spearman_Rho'] 
            available_cols_for_groupby_dim2 = [col for col in cols_for_groupby_dim2 if col in df_summary_dim2.columns]
            
            if available_cols_for_groupby_dim2:
                summary_table_dim2 = df_summary_dim2.groupby(['Representation', 'DR_Method'])[available_cols_for_groupby_dim2].mean().reset_index()
                if 'ROC_AUC' in summary_table_dim2.columns:
                    summary_table_dim2.sort_values(by=['Representation', 'ROC_AUC'], ascending=[True, False], inplace=True)
                add_dataframe_as_latex_table_standalone(latex_content, summary_table_dim2,
                                             "Average Ranking Performance Metrics at 2 Dimensions across all targets. Spearman Rho is averaged.",
                                             "overall-summary-dim2-metrics", font_size=r"\scriptsize")
            else:
                latex_content.append("No standard ranking metric columns found for DIM=2 summary table.\n")

        if all_targets_dim_opt_data: 
             df_best_configs_from_dim_opt = pd.DataFrame(all_targets_dim_opt_data)
             if not df_best_configs_from_dim_opt.empty and 'Target' in df_best_configs_from_dim_opt.columns and 'Best_ROC_AUC' in df_best_configs_from_dim_opt.columns:
                 # Merge to get Spearman_Rho for these best ROC-AUC configs
                 df_best_configs_from_dim_opt.rename(columns={'Optimal_DIM_by_ROC_AUC': 'DIM', 'Best_ROC_AUC': 'Max_ROC_AUC'}, inplace=True)
                 
                 cols_to_merge = ['Target', 'Representation', 'DR_Method', 'DIM', 'Spearman_Rho']
                 available_cols_to_merge = [c for c in cols_to_merge if c in df_all_metrics_summary.columns]
                 
                 df_merged_best_rho = pd.merge(
                     df_best_configs_from_dim_opt,
                     df_all_metrics_summary[available_cols_to_merge], # Select only needed cols from potentially wide df
                     on=['Target', 'Representation', 'DR_Method', 'DIM'], # DIM is now Optimal_DIM
                     how='left'
                 )
                 
                 final_cols_best_configs = ['Target', 'Representation', 'DR_Method', 'DIM', 'Max_ROC_AUC']
                 if 'Spearman_Rho' in df_merged_best_rho.columns:
                     final_cols_best_configs.append('Spearman_Rho')
                 else: # Should not happen if merge is correct
                     df_merged_best_rho['Spearman_Rho'] = np.nan
                     final_cols_best_configs.append('Spearman_Rho')

                 df_best_per_target_display = df_merged_best_rho[final_cols_best_configs].copy()
                 df_best_per_target_display.rename(columns={'DIM': 'Optimal DIM (by ROCAUC)'}, inplace=True)
                 df_best_per_target_display.sort_values(by='Target', inplace=True)

                 add_dataframe_as_latex_table_standalone(latex_content, df_best_per_target_display,
                                              "Best performing configuration (Rep., DR, Opt. DIM) for each target by ROC-AUC, with corresponding Spearman $\\rho$.",
                                              "overall-best-config-per-target-rho", font_size=r"\scriptsize")
    else:
        latex_content.append("No ranking metrics data collected to generate overall summary tables.\n")

    latex_content.append(LATEX_DOCUMENT_END)

    report_tex_filename = "standalone_experiment_report.tex"
    report_tex_path = os.path.join(report_output_abs_dir, report_tex_filename)
    with open(report_tex_path, "w", encoding='utf-8') as f:
        f.write("\n".join(latex_content))
    
    logging.info(f"Standalone LaTeX report generated: {report_tex_path}")
    logging.info(f"Associated figures copied to: {report_figures_abs_dir}")

    try:
        logging.info(f"Attempting to compile LaTeX report in: {report_output_abs_dir}")
        for i in range(2): 
            process = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-output-directory", report_output_abs_dir, report_tex_path],
                capture_output=True, text=True, check=False 
            )
            if process.returncode != 0:
                logging.error(f"pdflatex compilation pass {i+1} failed.")
                log_file_path = os.path.join(report_output_abs_dir, "standalone_experiment_report.log")
                if os.path.exists(log_file_path):
                    with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as lf: 
                        logging.error("LaTeX Log:\n" + lf.read())
                else:
                    logging.error("STDOUT:\n" + process.stdout)
                    logging.error("STDERR:\n" + process.stderr)
                break 
        else: 
             logging.info(f"PDF report compilation attempt finished. Check {report_output_abs_dir} for {os.path.basename(report_tex_path).replace('.tex', '.pdf')}")
    except FileNotFoundError: logging.warning("pdflatex command not found. Please compile the .tex file manually.")
    except Exception as e: logging.error(f"An unexpected error occurred during pdflatex compilation: {e}")

if __name__ == "__main__":
    main()