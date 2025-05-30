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

# --- Standalone LaTeX Document Shell (same as last complete version) ---
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
    For held-out target actives where quantitative bioactivity data (e.g., pIC50, pKi, derived from 'Standard Value (nM)') was available, Spearman's rank correlation coefficient ($\rho$) was calculated. This assessed the monotonic relationship between the rank of an active compound (derived from its proximity score to the MF Cloud) and its experimentally determined affinity. A statistically significant positive correlation (given scores are `-distance`, and pActivity increases with potency) would suggest that compounds ranked higher by the similarity model also tend to be more potent.
\end{itemize}
The following sections detail the results obtained for each target protein based on these metrics.
"""

LATEX_DOCUMENT_END = r"""
\end{document}
"""

# --- LaTeX Helper Functions ---
def clean_for_label(text):
    """Cleans a string to be a safe LaTeX label (alphanumeric, hyphen)."""
    if not isinstance(text, str): text = str(text)
    text = text.replace('_', '-').replace(' ', '-').replace('.', '')
    return re.sub(r'[^a-zA-Z0-9-]', '', text) # Keep only alphanumeric and hyphen

def escape_latex_text_content(text_input):
    """Escapes special LaTeX characters in text content (captions, titles)."""
    if not isinstance(text_input, str): text_input = str(text_input)
    conv = {'&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_',
            '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\^{}',
            '<': r'\textless{}', '>': r'\textgreater{}'}
    # Do not escape backslash here unless you intend to print a literal backslash character.
    # LaTeX commands starting with \ should not be touched by this.
    for k, v in conv.items():
        text_input = text_input.replace(k, v)
    return text_input

def get_section_header_latex_standalone(level, title_text):
    sec_cmd_map = {1: r"\section", 2: r"\subsection", 3: r"\subsubsection", 4: r"\paragraph"}
    sec_cmd = sec_cmd_map.get(level, r"\paragraph")
    return f"\n{sec_cmd}{{{escape_latex_text_content(title_text)}}}\n"

def add_figure_to_latex_standalone(latex_content_list, relative_fig_path_in_tex, caption_text, label_text, placement="[htbp]", figure_width="0.7\\linewidth"):
    # Assumes relative_fig_path_in_tex is clean (e.g., "figures/my-plot.png")
    clean_label = clean_for_label(label_text)
    latex_content_list.append(f"\\begin{{figure}}{placement}") # Corrected placement syntax
    latex_content_list.append(r"  \centering")
    latex_content_list.append(f"  \\includegraphics[width={figure_width}]{{{relative_fig_path_in_tex}}}")
    latex_content_list.append(f"  \\caption{{{escape_latex_text_content(caption_text)}}}")
    latex_content_list.append(f"  \\label{{fig:{clean_label}}}")
    latex_content_list.append(r"\end{figure}")
    latex_content_list.append("\n")

def add_dataframe_as_latex_table_standalone(latex_content_list, dataframe, caption_text, label_text, placement="[htbp]", col_format=None, font_size=r"\small"):
    clean_label = clean_for_label(label_text)
    if dataframe is not None and not dataframe.empty:
        latex_content_list.append(f"\\begin{{table}}{placement}") # Corrected placement syntax
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
            col_format = '|' + '|'.join(formats) + '|'

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
    # ... (argparse setup as before) ...
    parser.add_argument("--experiment_run_dir", required=True)
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()


    with open(args.config_path, 'r') as f: config = json.load(f)
    gs = config['global_settings']
    
    run_id_cleaned = clean_for_label(os.path.basename(args.experiment_run_dir)) # For use in labels if needed
    run_id_for_display = escape_latex_text_content(os.path.basename(args.experiment_run_dir)) # For display in title
    
    preamble_filled = LATEX_DOCUMENT_PREAMBLE.replace("<<RUN_ID_PLACEHOLDER>>", run_id_for_display)
    latex_content = [preamble_filled]

    report_filename_base = clean_for_label(os.path.basename(args.experiment_run_dir)) + "-standalone-report" # Cleaned base for dir
    report_output_abs_dir = os.path.join(args.output_dir, report_filename_base) 
    report_figures_abs_dir = os.path.join(report_output_abs_dir, "figures") 
    os.makedirs(report_figures_abs_dir, exist_ok=True)
    
    all_targets_ranking_metrics = [] 
    all_targets_dim_opt_data = [] 

    for target_info in config['targets']:
        target_id_name = target_info['id_name']
        target_display_name = target_info['display_name'] 
        mf_display_name = target_info['molecular_function_display_name']
        
        # Cleaned names for labels
        target_label_name = clean_for_label(target_id_name)

        latex_content.append(f"\\clearpage\n{get_section_header_latex_standalone(1, f'Results for Target: {target_display_name} (MF: {mf_display_name})')}")
        target_results_base_dir = os.path.join(args.experiment_run_dir, target_id_name, "results")

        if not os.path.exists(target_results_base_dir):
            latex_content.append(f"Results data not found for target {escape_latex_text_content(target_id_name)}.\n")
            continue

        latex_content.append(get_section_header_latex_standalone(2, "Optimization of Similarity Space Dimensionality"))
        latex_content.append("The impact of similarity space dimensionality on ranking performance (ROC-AUC using minimum distance to MF Cloud for scoring) was assessed. "
                             f"Values tested for PCA and UMAP were: {escape_latex_text_content(str(gs['simspace_dims_to_test']))}. t-SNE was evaluated only at 2 dimensions.\n")

        current_target_dim_opt_summary_rows = [] 

        for repr_type in config['representations']:
            latex_content.append(get_section_header_latex_standalone(3, f"Representation: {repr_type.capitalize()}"))
            repr_label = clean_for_label(repr_type)
            
            for dr_key, dr_params in config["dimensionality_reduction_methods"].items():
                dr_display_name = dr_params["short_name"]
                dr_short_name_fs = dr_display_name.replace('-', '_') # Still used for finding files
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
                                
                                # Collect ALL metrics, ensuring keys are consistent
                                metrics_to_collect = {
                                    'Target': target_display_name, 'Representation': repr_type.capitalize(), 
                                    'DR_Method': dr_display_name, 'DIM': simspace_dim_val,
                                    'ROC_AUC': current_roc_auc
                                }
                                for metric_key in ['pr_auc', 'ef_1%', 'ef_5%', 'ef_10%', 'spearman_rho_affinity_vs_score']:
                                    metrics_to_collect[metric_key.upper().replace('%','Perc')] = df_metrics[metric_key].iloc[0] if metric_key in df_metrics.columns and not df_metrics[metric_key].empty and not df_metrics[metric_key].isna().all() else np.nan
                                all_targets_ranking_metrics.append(metrics_to_collect)

                        except Exception as e: logging.warning(f"Could not process ranking metrics CSV {ranking_metrics_csv}: {e}")
                
                if dim_vs_roc_auc_data: 
                    df_plot = pd.DataFrame(dim_vs_roc_auc_data).sort_values(by='dim')
                    plt.figure(figsize=(8, 5)) # Keep Matplotlib for plot generation
                    plt.plot(df_plot['dim'], df_plot['roc_auc'], marker='o', linestyle='-')
                    plt.xlabel("SIMSPACE_DIM Value") # Underscore is fine in Matplotlib axis label
                    plt.ylabel("ROC-AUC (vs. MF Cloud Proximity)")
                    plt.title(f"SIMSPACE_DIM vs. ROC-AUC for {target_display_name}\n({repr_type.capitalize()}, {dr_display_name})", fontsize=11) # No need to escape for Matplotlib title
                    if dr_key != "tsne": plt.xticks(gs['simspace_dims_to_test'])
                    else: plt.xticks([2]) 
                    plt.ylim(0, 1.05) 
                    plt.grid(True, linestyle='--', alpha=0.6)
                    
                    plot_filename = f"{target_label_name}-{repr_label}-{dr_label}-dim-vs-rocauc.png" # Cleaned filename
                    plot_abs_path_dest = os.path.join(report_figures_abs_dir, plot_filename)
                    try:
                        plt.savefig(plot_abs_path_dest, dpi=150, bbox_inches='tight')
                        plot_relative_path_for_tex = os.path.join("figures", plot_filename) # Path for \includegraphics
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
            if current_target_dim_opt_summary_rows and (len(current_target_dim_opt_summary_rows) % (len(config["dimensionality_reduction_methods"])*len(config['representations'])) == 0 or repr_type == config['representations'][-1]): # Add clearpage after each representation's DR methods
                latex_content.append("\\clearpage\n") 
        
        if current_target_dim_opt_summary_rows:
            df_summary = pd.DataFrame(current_target_dim_opt_summary_rows)
            latex_content.append(get_section_header_latex_standalone(3, f"Summary of Optimal Dimensionality (by ROC-AUC) for {target_display_name}"))
            add_dataframe_as_latex_table_standalone(latex_content, df_summary,
                                         f"Optimal SIMSPACE_DIM (maximizing ROC-AUC) and corresponding ROC-AUC for target {target_display_name}.",
                                         f"opt-dim-summary-{target_label_name}")
        latex_content.append("\\clearpage\n")

        latex_content.append(get_section_header_latex_standalone(2, "Illustrative 2D Projections (SIMSPACE_DIM = 2)"))
        # ... (2D plot logic - ensure labels are cleaned here too, similar to above)
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
                    hist_clean_fname = f"{target_label_name}-{repr_label_2d}-{dr_label_2d}-dim2-hist.png"

                    if os.path.exists(scatter_src_report):
                        shutil.copy(scatter_src_report, os.path.join(report_figures_abs_dir, scatter_clean_fname))
                        caption_s = f"2D Similarity space for {target_display_name} ({repr_type_2d.capitalize()}, {dr_display_name_2d})."
                        add_figure_to_latex_standalone(latex_content, os.path.join("figures", scatter_clean_fname), caption_s, f"{target_label_name}-{repr_label_2d}-{dr_label_2d}-scatter2D")

                    if os.path.exists(hist_src_report):
                        shutil.copy(hist_src_report, os.path.join(report_figures_abs_dir, hist_clean_fname))
                        caption_h = f"Histogram of minimum distances for ACTIVES of {target_display_name} ({repr_type_2d.capitalize()}, {dr_display_name_2d}, 2D)."
                        add_figure_to_latex_standalone(latex_content, os.path.join("figures", hist_clean_fname), caption_h, f"{target_label_name}-{repr_label_2d}-{dr_label_2d}-hist2D", figure_width="0.65\\textwidth")
            if repr_has_2d_plot_report: latex_content.append("\\clearpage\n")
        if not dim_2_exists_for_target_report:
            latex_content.append("No 2D projection results were found to display for this target.\n")


    latex_content.append(f"\\clearpage\n{get_section_header_latex_standalone(1, 'Overall Comparative Summary of Ranking Performance')}")
    if all_targets_ranking_metrics:
        df_all_metrics_summary = pd.DataFrame(all_targets_ranking_metrics)
        
        # Ensure column names are consistent with how they were added to all_targets_ranking_metrics
        # Keys used: 'Target', 'Representation', 'DR_Method', 'DIM', 'ROC_AUC', 'PR_AUC', 'EF_1Perc', 'EF_5Perc', 'EF_10Perc', 'SPEARMAN_RHO'
        
        df_summary_dim2 = df_all_metrics_summary[df_all_metrics_summary['DIM'] == 2].copy()
        if not df_summary_dim2.empty:
            cols_for_groupby = ['ROC_AUC', 'PR_AUC', 'EF_1Perc', 'EF_5Perc', 'EF_10Perc', 'SPEARMAN_RHO']
            available_cols_for_groupby = [col for col in cols_for_groupby if col in df_summary_dim2.columns]
            
            if available_cols_for_groupby:
                summary_table_dim2 = df_summary_dim2.groupby(['Representation', 'DR_Method'])[available_cols_for_groupby].mean().reset_index()
                if 'ROC_AUC' in summary_table_dim2: # Sort if ROC_AUC is present
                    summary_table_dim2.sort_values(by=['Representation', 'ROC_AUC'], ascending=[True, False], inplace=True)
                add_dataframe_as_latex_table_standalone(latex_content, summary_table_dim2,
                                             "Average Ranking Performance Metrics at 2 Dimensions across all targets.",
                                             "overall-summary-dim2-metrics", font_size=r"\scriptsize") # smaller font

        if all_targets_dim_opt_data: 
             df_best_overall_setups = pd.DataFrame(all_targets_dim_opt_data)
             if not df_best_overall_setups.empty and 'Target' in df_best_overall_setups.columns and 'Best_ROC_AUC' in df_best_overall_setups.columns:
                 # Check if groupby result is non-empty before idxmax
                 grouped_best = df_best_overall_setups.groupby('Target')['Best_ROC_AUC']
                 if not grouped_best.count().empty: # Check if there are groups with data
                     idx_best_overall = grouped_best.idxmax()
                     df_best_per_target_overall = df_best_overall_setups.loc[idx_best_overall].sort_values(by='Target')
                     df_best_per_target_overall = df_best_per_target_overall[['Target', 'Representation', 'DR_Method', 'Optimal_DIM_by_ROC_AUC', 'Best_ROC_AUC']]
                     df_best_per_target_overall.rename(columns={'Optimal_DIM_by_ROC_AUC': 'Optimal DIM', 'Best_ROC_AUC': 'Max ROC-AUC'}, inplace=True)
                     add_dataframe_as_latex_table_standalone(latex_content, df_best_per_target_overall,
                                                  "Best performing configuration (Rep., DR, Opt. DIM) for each target by ROC-AUC.",
                                                  "overall-best-config-per-target", font_size=r"\scriptsize")
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
        for i in range(2): # Run twice for TOC, references, etc.
            process = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-output-directory", report_output_abs_dir, report_tex_path],
                capture_output=True, text=True, check=False 
            )
            if process.returncode != 0:
                logging.error(f"pdflatex compilation pass {i+1} failed.")
                log_file_path = os.path.join(report_output_abs_dir, "standalone_experiment_report.log")
                if os.path.exists(log_file_path):
                    with open(log_file_path, 'r') as lf: logging.error("LaTeX Log:\n" + lf.read())
                else:
                    logging.error("STDOUT:\n" + process.stdout)
                    logging.error("STDERR:\n" + process.stderr)
                break 
        else: 
             logging.info(f"PDF report compilation attempt finished. Check {report_output_abs_dir} for standalone_experiment_report.pdf")
    except FileNotFoundError: logging.warning("pdflatex command not found. Please compile the .tex file manually.")
    except Exception as e: logging.error(f"An unexpected error occurred during pdflatex compilation: {e}")

if __name__ == "__main__":
    main()