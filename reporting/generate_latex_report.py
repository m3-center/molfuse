import pandas as pd
import numpy as np
import os
import argparse
import logging
import json
import shutil
from collections import defaultdict
import matplotlib
matplotlib.use('Agg') # Ensure non-interactive backend
import matplotlib.pyplot as plt
import subprocess
import re # For cleaning labels

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)-8s - %(filename)-25s - %(funcName)-25s - %(message)s',
                    handlers=[logging.FileHandler("generate_standalone_report.log", mode='w'), logging.StreamHandler()])

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
\usepackage{booktabs} % UNCOMMENTED
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
This report details experiments conducted to evaluate the predictive capacity of molecular similarity spaces
using a leave-one-target-out inspired methodology. The central hypothesis posits that ligands active against
a specific protein target will exhibit proximity to compounds known to interact with other proteins sharing
the same broader molecular function, even when the specific target's known ligands are excluded during the
construction of the similarity space. This study compares two main embedding strategies for PCA and UMAP:
(1) \textbf{Projection}, where models are trained on the main dataset (molecular function cloud and decoys) and
held-out target actives are subsequently projected; and (2) \textbf{Co-embedding}, where held-out target actives
are included with the main dataset during the dimensionality reduction model training. t-SNE is inherently
a co-embedding method in this setup. Experiments encompassed multiple target proteins, two molecular
representations (physicochemical features and ECFP4 fingerprints), various dimensionality reduction (DR)
techniques (PCA; UMAP with Euclidean, Manhattan, and Hamming metrics; and t-SNE with pre-
PCA), and a range of similarity space dimensionalities. Evaluation primarily focused on rank-based metrics
(ROC-AUC, PR-AUC, Enrichment Factors) derived from the proximity of projected/co-embedded known
target ligands to the molecular function (MF) cloud relative to decoy compounds, as well as correlation with
experimental affinity data where available.
\end{abstract}
\clearpage
\tableofcontents
\clearpage
\listoffigures
\clearpage
\listoftables
\clearpage

\section{Introduction}
The exploration of chemical space for novel therapeutic agents is a cornerstone of drug discovery. Molecular
similarity, a fundamental concept in cheminformatics, suggests that structurally similar molecules are likely
to exhibit similar biological activities. This principle underpins many virtual screening and lead optimization
strategies. This study investigates the utility of constructing and analyzing molecular similarity spaces to predict
potential interactions between small molecules and protein targets, grouped by their shared molecular function.

The experimental design aims to rigorously test whether general molecular function similarity can guide the
identification of ligands for a specific, "held-out" protein target. A key aspect of this investigation is the com-
parison of different embedding strategies: projecting held-out actives into pre-built spaces versus co-embedding
them with the reference data during space construction. By systematically varying data representations, di-
mensionality reduction methods, embedding strategies, and the dimensionality of the resulting spaces, we seek
to identify optimal parameters and assess the overall robustness of this similarity-based approach. Key metrics
involve measuring the proximity of known active ligands (for the held-out target) to the cloud of compounds as-
sociated with the broader molecular function after projection/co-embedding into these tailored similarity spaces,
and evaluating their rank relative to decoy compounds.

\section{Methodology Overview}
The experimental methodology involved several key stages: dataset preparation (including segregation of held-out
target actives and creation of "blinded" datasets), descriptor calculation for the held-out actives, construction of
similarity spaces using various dimensionality reduction techniques and parameters (including distinct handling
for projection and co-embedding strategies for PCA and UMAP), projection or co-embedding of held-out actives,
and finally, performance evaluation using rank-based metrics. Each step was automated and configured via a
central JSON file. The primary software components involved were Python scripts leveraging libraries such as
RDKit, Mordred, Pandas, scikit-learn, Matplotlib, Seaborn, and optionally cuML for GPU acceleration.

\subsection{Performance Evaluation Metrics}
\label{subsec:perf_metrics}
To assess the ability of each similarity space configuration and embedding strategy to effectively "rediscover" the
held-out target active ligands among a set of decoy compounds (derived from the ZINC database), several rank-
based metrics were employed. Compounds were scored based on their proximity (primarily minimum Euclidean
distance) to the Molecular Function (MF) Cloud, with smaller distances yielding higher scores. The held-out
target ligands were labeled as "actives" and ZINC compounds as "decoys."

\begin{itemize}[leftmargin=*]
    \item \textbf{Receiver Operating Characteristic Area Under Curve (ROC-AUC):}
    The ROC curve plots the True Positive Rate (TPR, sensitivity) against the False Positive Rate (FPR, 1-specificity) at various ranking thresh-
    olds. The Area Under this Curve (AUC) provides a single measure of the model's ability to discriminate
    between active and decoy compounds across all thresholds. A ROC-AUC of 1.0 represents a perfect classifier,
    while 0.5 indicates random performance. Formally, $TPR = \frac{TP}{TP+FN}$ and $FPR = \frac{FP}{FP+TN}$, where TP, FP, TN,
    FN are True Positives, False Positives, True Negatives, and False Negatives, respectively.

    \item \textbf{Precision-Recall Area Under Curve (PR-AUC):}
    The Precision-Recall curve plots Precision against
    Recall (TPR) at various ranking thresholds. This metric is particularly informative for imbalanced datasets,
    common in virtual screening where decoys vastly outnumber actives. A higher PR-AUC indicates better
    performance. Formally, $Precision = \frac{TP}{TP+FP}$ and $Recall = TPR$.

    \item \textbf{Enrichment Factor (EF) at x\%:}
    The Enrichment Factor measures how many more active compounds are
    found in the top x\% of a ranked list compared to a random selection. It is defined as:
    $$ EF_{x\%} = \frac{\text{Actives found in top x\%} / \text{Compounds in top x\%}}{\text{Total Actives} / \text{Total Compounds}} $$
    EF values greater than 1 indicate enrichment. In this study, EF at 1\%, 5\%, and 10\% of the ranked list were
    calculated.

    \item \textbf{Spearman's Rank Correlation ($\rho$) with Affinity:}
    For held-out target actives where quantitative bioac-
    tivity data (e.g., pIC50, pKi, derived from 'Standard Value (nM)') was available, Spearman's rank correlation
    coefficient ($\rho$) was calculated. This assessed the monotonic relationship between the rank of an active com-
    pound (derived from its proximity score to the MF Cloud) and its experimentally determined affinity. A
    statistically significant positive correlation (given scores are based on '-distance', and pActivity increases with
    potency) would suggest that compounds ranked higher by the similarity model also tend to be more potent.
\end{itemize}
The following sections detail the results obtained for each target protein based on these metrics, considering
different embedding strategies where applicable.
"""

LATEX_DOCUMENT_END = r"""
\end{document}
"""

def escape_latex_text_content(text_input):
    if not isinstance(text_input, str): text_input = str(text_input)
    conv = {'&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_',
            '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\^{}',
            '<': r'\textless{}', '>': r'\textgreater{}'}
    for k, v in conv.items():
        text_input = text_input.replace(k, v)
    return text_input

def clean_for_label(text):
    if not isinstance(text, str): text = str(text)
    text = text.replace('_', '-').replace(' ', '-').replace('.', '-').replace('/', '-')
    text = re.sub(r'[^a-zA-Z0-9-]', '', text) 
    text = re.sub(r'-+', '-', text) 
    return text.strip('-')[:50] 

def get_section_header_latex_standalone(level, title_text):
    sec_cmd_map = {1: r"\section", 2: r"\subsection", 3: r"\subsubsection", 4: r"\paragraph"}
    sec_cmd = sec_cmd_map.get(level, r"\paragraph")
    return f"\n{sec_cmd}{{{escape_latex_text_content(title_text)}}}\n"

def add_figure_to_latex_standalone(latex_content_list, relative_fig_path_in_tex, caption_text, label_text, placement="[H]", figure_width="0.7\\linewidth"):
    clean_label = clean_for_label(label_text)
    figure_path_for_latex = relative_fig_path_in_tex.replace(os.sep, '/')
    latex_content_list.append(f"\\begin{{figure}}{placement}")
    latex_content_list.append(r"  \centering")
    latex_content_list.append(f"  \\includegraphics[width={figure_width}]{{{figure_path_for_latex}}}")
    latex_content_list.append(f"  \\caption{{{escape_latex_text_content(caption_text)}}}")
    latex_content_list.append(f"  \\label{{fig:{clean_label}}}")
    latex_content_list.append(r"\end{figure}")
    latex_content_list.append("\n")

def add_dataframe_as_latex_table_standalone(latex_content_list, dataframe, caption_text, label_text, placement="[H]", col_format=None, font_size=r"\small"):
    clean_label = clean_for_label(label_text)
    if dataframe is not None and not dataframe.empty:
        latex_content_list.append(f"\\begin{{table}}{placement}")
        latex_content_list.append(r"  \centering")
        if font_size: latex_content_list.append(font_size)
        latex_content_list.append(f"  \\caption{{{escape_latex_text_content(caption_text)}}}")
        latex_content_list.append(f"  \\label{{tab:{clean_label}}}")
        
        df_for_latex = dataframe.copy()
        for col in df_for_latex.columns:
            if pd.api.types.is_numeric_dtype(df_for_latex[col]):
                df_for_latex[col] = df_for_latex[col].apply(
                    lambda x: f"{x:.3f}" if pd.notna(x) and isinstance(x, (float, np.floating)) and ( (abs(x) >= 0.001 and abs(x) < 1000) or x==0) else 
                              (f"{x:.2e}" if pd.notna(x) and isinstance(x, (float, np.floating)) else 
                              (str(int(x)) if pd.notna(x) and isinstance(x, (int, np.integer)) else ("N/A" if pd.isna(x) else str(x))) ) 
                )
            else: 
                df_for_latex[col] = df_for_latex[col].astype(str).apply(lambda x: "N/A" if pd.isna(x) or (isinstance(x,str) and x.lower() == 'nan') else escape_latex_text_content(x))

        df_for_latex.columns = [escape_latex_text_content(str(col).replace('_', ' ').title()) for col in dataframe.columns]
        
        if col_format is None:
            formats = ['l'] * len(df_for_latex.columns) 
            for i, col_name_orig in enumerate(dataframe.columns): 
                if pd.api.types.is_numeric_dtype(dataframe[col_name_orig]): formats[i] = 'r' 
            col_format = '|' + '|'.join(formats) + '|'

        latex_table_string = df_for_latex.to_latex(index=False, escape=False, 
                                                   column_format=col_format,
                                                   longtable=isinstance(dataframe, pd.DataFrame) and len(dataframe)>20,
                                                   na_rep="N/A") 
        latex_content_list.append(latex_table_string)
        latex_content_list.append(r"\end{table}")
        latex_content_list.append("\n")
    else:
        latex_content_list.append(f"% Table data for '{escape_latex_text_content(label_text)}' is empty or None.\n")

# --- Main Report Generation Logic ---
def main():
    parser = argparse.ArgumentParser(description="Generate Standalone LaTeX report from experiment results.")
    parser.add_argument("--experiment_run_dir", required=True)
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--output_dir", required=True)
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

    for target_info in config['targets']:
        target_id_name = target_info['id_name']
        target_display_name = target_info['display_name'] 
        logging.info(f"Processing Target: {target_display_name}")
        mf_display_name = target_info['molecular_function_display_name']
        target_label_name = clean_for_label(target_id_name)
        
        latex_content.append(f"\\clearpage\n{get_section_header_latex_standalone(2, f'Target: {target_display_name} (MF: {mf_display_name})')}")
        target_results_base_dir = os.path.join(args.experiment_run_dir, target_id_name, "results")
        if not os.path.exists(target_results_base_dir):
            logging.warning(f"Results base directory NOT FOUND for target {target_id_name}: {target_results_base_dir}")
            continue

        latex_content.append(get_section_header_latex_standalone(3, "Optimization of Similarity Space Dimensionality and Embedding Strategy"))
        latex_content.append("The impact of similarity space dimensionality on ranking performance (ROC-AUC based on proximity to the MF Cloud) was assessed for different embedding strategies. "
                             f"Dimensionalities tested for PCA and UMAP were: {escape_latex_text_content(str(gs['simspace_dims_to_test']))}. t-SNE was evaluated only at 2 dimensions (co-embedded by nature).\n")
        current_target_dim_opt_summary_rows = [] 

        for repr_type in config['representations']:
            logging.info(f"  Representation: {repr_type}")
            latex_content.append(get_section_header_latex_standalone(4, f"Representation: {repr_type.capitalize()}"))
            repr_label = clean_for_label(repr_type)
            
            for dr_key_config, dr_params_config in config["dimensionality_reduction_methods"].items():
                dr_short_name_base = dr_params_config["short_name"]
                logging.info(f"    DR Method (Base): {dr_short_name_base} (Key: {dr_key_config})")

                dr_label_base = clean_for_label(dr_short_name_base)
                
                strategies_to_find_results_for = []
                strategies_to_find_results_for.append({
                    "output_dir_leaf_name_fs": dr_short_name_base.replace('-', '_'), # Filesystem safe
                    "embedding_label_for_metrics": "Projection" if not dr_key_config == "tsne" else "Co-embedding (Native)",
                    "plot_legend_label": "Projection" if not dr_key_config == "tsne" else "Co-embedding (Native)"
                })
                # Co-embedding strategy for PCA/UMAP
                if gs.get("run_coembedding_for_pca_umap", False) and \
                   dr_params_config.get("allow_coembedding", False) and \
                   not dr_key_config == "tsne":
                    strategies_to_find_results_for.append({
                        "output_dir_leaf_name_fs": f"{dr_short_name_base}-Coembed".replace('-', '_'), # Filesystem safe
                        "embedding_label_for_metrics": "Co-embedding",
                        "plot_legend_label": "Co-embedding"
                    })
                
                logging.debug(f"      Strategies to check for {dr_short_name_base}: {strategies_to_find_results_for}")
                
                plt.figure(figsize=(8, 5))
                plot_created_for_dr_method = False

                for strategy_details in strategies_to_find_results_for:
                    results_dir_leaf_fs = strategy_details["output_dir_leaf_name_fs"] # e.g., PCA, PCA_Coembed (filesystem safe)
                    embedding_strat_label = strategy_details["embedding_label_for_metrics"]
                    # plot_legend_label = strategy_details["plot_legend_label"] # Not directly used for filename
                    logging.info(f"      Checking Strategy: '{embedding_strat_label}' (dir leaf: {results_dir_leaf_fs})")
                    dim_vs_roc_auc_data_strat = []

                    for simspace_dim_val in gs['simspace_dims_to_test']:
                        if dr_key_config == "tsne" and simspace_dim_val != 2:
                            continue
                        
                        results_path_for_strategy_dim = os.path.join(target_results_base_dir, repr_type, f"dim_{simspace_dim_val}", results_dir_leaf_fs)
                        base_dr_name_for_filename = dr_short_name_base.replace('-', '_') # e.g., PCA, UMAP_Euclidean
                        ranking_metrics_filename = f"{target_id_name}_{repr_type}_{base_dr_name_for_filename}_dim{simspace_dim_val}_ranking_metrics.csv"
                        ranking_metrics_csv_full_path = os.path.join(results_path_for_strategy_dim, ranking_metrics_filename)
                        
                        logging.debug(f"        DIM: {simspace_dim_val}, Attempting to load metrics file: {ranking_metrics_csv_full_path}")
                        # results_path_for_dim_dr_strat = os.path.join(target_results_base_dir, repr_type, f"dim_{simspace_dim_val}", results_dir_leaf)
                        # ranking_metrics_csv = os.path.join(results_path_for_dim_dr_strat, f"{target_id_name}_{repr_type}_{results_dir_leaf}_dim{simspace_dim_val}_ranking_metrics.csv")
                        
                        if os.path.exists(ranking_metrics_csv_full_path):
                            logging.info(f"        FOUND metrics file: {ranking_metrics_csv_full_path}")
                            try:
                                df_metrics = pd.read_csv(ranking_metrics_csv_full_path)
                                if not df_metrics.empty and 'roc_auc' in df_metrics.columns:
                                    current_roc_auc = df_metrics['roc_auc'].iloc[0]
                                    
                                    metrics_to_add = {
                                        'Target': target_display_name, 'Representation': repr_type.capitalize(), 
                                        'DR_Method': dr_short_name_base, 
                                        'Embedding_Strategy': embedding_strat_label, 
                                        'DIM': simspace_dim_val, 'ROC_AUC': current_roc_auc
                                    }
                                    for mk_csv, mk_dict in [('pr_auc', 'PR_AUC'), ('ef_1%', 'EF_1Perc'), ('ef_5%', 'EF_5Perc'), 
                                                            ('ef_10%', 'EF_10Perc'), ('spearman_rho_affinity_vs_score', 'Spearman_Rho')]:
                                        s_data = df_metrics.get(mk_csv, pd.Series([np.nan]))
                                        v_add = s_data.iloc[0] if not s_data.empty and pd.notna(s_data.iloc[0]) else np.nan
                                        metrics_to_add[mk_dict] = v_add
                                    all_targets_ranking_metrics.append(metrics_to_add)
                            except Exception as e: logging.warning(f"Could not process ranking metrics CSV {ranking_metrics_csv_full_path}: {e}")
                        else:
                            logging.error(f"        Did NOT find metrics file: {ranking_metrics_csv_full_path}")
                    
                    if dim_vs_roc_auc_data_strat:
                        df_plot_strat = pd.DataFrame(dim_vs_roc_auc_data_strat).sort_values(by='dim')
                        plt.plot(df_plot_strat['dim'], df_plot_strat['roc_auc'], marker='o', linestyle='-', label=plot_legend_label)
                        plot_created_for_dr_method = True
                        if not df_plot_strat.empty:
                            best_row_strat = df_plot_strat.loc[df_plot_strat['roc_auc'].idxmax()]
                            current_target_dim_opt_summary_rows.append({
                                'Representation': repr_type.capitalize(), 'DR_Method': dr_short_name_base,
                                'Embedding_Strategy': embedding_strat_label,
                                'Optimal_DIM_by_ROC_AUC': int(best_row_strat['dim']), 
                                'Best_ROC_AUC': best_row_strat['roc_auc']
                            })
                
                if plot_created_for_dr_method:
                    plt.xlabel("SIMSPACE_DIM Value") 
                    plt.ylabel("ROC-AUC (vs. MF Cloud Proximity)")
                    plt.title(f"SIMSPACE_DIM vs. ROC-AUC for {target_display_name}\n({repr_type.capitalize()}, {dr_short_name_base})", fontsize=11) 
                    if dr_key_config != "tsne": plt.xticks(gs['simspace_dims_to_test'])
                    else: plt.xticks([2]) 
                    plt.ylim(0.0, 1.05); plt.legend(title="Embedding Strategy")
                    plt.grid(True, linestyle='--', alpha=0.6)
                    plot_filename = f"{target_label_name}-{repr_label}-{dr_label_base}-dim-vs-rocauc-strat.png"
                    plot_abs_path_dest = os.path.join(report_figures_abs_dir, plot_filename)
                    try:
                        plt.savefig(plot_abs_path_dest, dpi=150, bbox_inches='tight')
                        plot_relative_path_for_tex = os.path.join("figures", plot_filename).replace(os.sep, '/')
                        caption_text = f"ROC-AUC score versus SIMSPACE_DIM for target {target_display_name} ({repr_type.capitalize()}, DR: {dr_short_name_base}), comparing different embedding strategies."
                        add_figure_to_latex_standalone(latex_content, plot_relative_path_for_tex, caption_text, f"{target_label_name}-{repr_label}-{dr_label_base}-dimopt-rocauc-strat")
                    except Exception as e: logging.error(f"Failed to save dim_opt_strat plot {plot_abs_path_dest}: {e}")
                plt.close() 
            if plot_created_for_dr_method : latex_content.append("\\clearpage\n") 
        
        if current_target_dim_opt_summary_rows:
            df_summary = pd.DataFrame(current_target_dim_opt_summary_rows)
            add_dataframe_as_latex_table_standalone(latex_content, df_summary,
                                         f"Optimal SIMSPACE_DIM (maximizing ROC-AUC) and corresponding ROC-AUC for target {target_display_name}, by embedding strategy.",
                                         f"opt-dim-summary-strat-{target_label_name}", font_size=r"\scriptsize")
        latex_content.append("\\clearpage\n")

        latex_content.append(get_section_header_latex_standalone(3, "Illustrative 2D Projections (SIMSPACE_DIM = 2)")) # Changed level to subsubsection
        dim_2_exists_for_target_report = False
        for repr_type_2d in config['representations']:
            repr_has_2d_plot_report = False
            repr_label_2d = clean_for_label(repr_type_2d)
            
            # Sub-subsection for representation under 2D projections
            # latex_content.append(get_section_header_latex_standalone(4, f"Representation: {repr_type_2d.capitalize()}"))


            for dr_key_2d, dr_params_2d_config in config["dimensionality_reduction_methods"].items():
                dr_short_name_base_2d = dr_params_2d_config["short_name"]
                
                strategies_2d_to_report = []
                if dr_key_2d == "tsne":
                    strategies_2d_to_report.append({"output_dir_leaf_name": dr_short_name_base_2d.replace('-', '_'), "plot_title_name": dr_short_name_base_2d, "embedding_label": "Co-embedding (Native)"})
                else: 
                    strategies_2d_to_report.append({"output_dir_leaf_name": dr_short_name_base_2d.replace('-', '_'), "plot_title_name": dr_short_name_base_2d, "embedding_label": "Projection"})
                    if gs.get("run_coembedding_for_pca_umap", False) and dr_params_2d_config.get("allow_coembedding", False):
                        strategies_2d_to_report.append({"output_dir_leaf_name": f"{dr_short_name_base_2d}-Coembed".replace('-', '_'), "plot_title_name": f"{dr_short_name_base_2d}-Coembed", "embedding_label": "Co-embedding"})
                
                # Check if any plots will be generated for this DR method before adding its header
                will_generate_plots_for_dr = False
                for strategy_2d_check in strategies_2d_to_report:
                    results_dir_leaf_2d_check = strategy_2d_check["output_dir_leaf_name"]
                    dim2_results_path_check = os.path.join(target_results_base_dir, repr_type_2d, "dim_2", results_dir_leaf_2d_check)
                    if os.path.exists(dim2_results_path_check):
                        will_generate_plots_for_dr = True
                        break
                
                if will_generate_plots_for_dr:
                    if not repr_has_2d_plot_report: # Add section header only once per representation
                        latex_content.append(get_section_header_latex_standalone(4, f"Representation: {repr_type_2d.capitalize()}"))
                        repr_has_2d_plot_report = True
                        dim_2_exists_for_target_report = True
                    # Sub-subsubsection for DR Method within Representation
                    latex_content.append(get_section_header_latex_standalone(5, f"DR Method: {dr_short_name_base_2d}"))


                for strategy_2d in strategies_2d_to_report:
                    results_dir_leaf_2d = strategy_2d["output_dir_leaf_name"]
                    plot_title_dr_name_2d = strategy_2d["plot_title_name"] 
                    embedding_strat_label_2d = strategy_2d["embedding_label"]
                    
                    dim2_results_path_report = os.path.join(target_results_base_dir, repr_type_2d, "dim_2", results_dir_leaf_2d)
                    if os.path.exists(dim2_results_path_report):
                        # Paragraph for specific strategy
                        latex_content.append(get_section_header_latex_standalone(6, f"Strategy: {embedding_strat_label_2d}"))
                        
                        scatter_orig_fname = f"{target_id_name}_{repr_type_2d}_{results_dir_leaf_2d}_dim2_scatter.png"
                        hist_orig_fname = f"{target_id_name}_{repr_type_2d}_{results_dir_leaf_2d}_dim2_min_distances_hist_ACTIVES.png"
                        
                        scatter_src_report = os.path.join(dim2_results_path_report, scatter_orig_fname)
                        hist_src_report = os.path.join(dim2_results_path_report, hist_orig_fname)
                        
                        scatter_clean_fname = f"{target_label_name}-{repr_label_2d}-{clean_for_label(results_dir_leaf_2d)}-dim2-scatter.png"
                        hist_clean_fname = f"{target_label_name}-{repr_label_2d}-{clean_for_label(results_dir_leaf_2d)}-dim2-hist-actives.png"

                        if os.path.exists(scatter_src_report):
                            shutil.copy(scatter_src_report, os.path.join(report_figures_abs_dir, scatter_clean_fname))
                            fig_path_tex = os.path.join("figures", scatter_clean_fname).replace(os.sep, '/')
                            caption_s = f"2D Similarity space for {target_display_name} ({repr_type_2d.capitalize()}, {plot_title_dr_name_2d})."
                            add_figure_to_latex_standalone(latex_content, fig_path_tex, caption_s, f"{target_label_name}-{repr_label_2d}-{clean_for_label(results_dir_leaf_2d)}-scatter2D")

                        if os.path.exists(hist_src_report):
                            shutil.copy(hist_src_report, os.path.join(report_figures_abs_dir, hist_clean_fname))
                            fig_path_tex = os.path.join("figures", hist_clean_fname).replace(os.sep, '/')
                            caption_h = f"Histogram of minimum distances for ACTIVES of {target_display_name} ({repr_type_2d.capitalize()}, {plot_title_dr_name_2d}, 2D)."
                            add_figure_to_latex_standalone(latex_content, fig_path_tex, caption_h, f"{target_label_name}-{repr_label_2d}-{clean_for_label(results_dir_leaf_2d)}-hist2D-actives", figure_width="0.65\\textwidth")
            if repr_has_2d_plot_report: latex_content.append("\\clearpage\n")
        if not dim_2_exists_for_target_report:
            latex_content.append("No 2D projection results were found to display for this target.\n")

    latex_content.append(f"\\clearpage\n{get_section_header_latex_standalone(1, 'Overall Comparative Summary of Ranking Performance')}")
    if all_targets_ranking_metrics:
        df_all_metrics_summary = pd.DataFrame(all_targets_ranking_metrics)
        logging.info("--- Collected Metrics Summary (df_all_metrics_summary) ---")
        logging.info(f"Shape: {df_all_metrics_summary.shape}")
        logging.info(f"Columns: {df_all_metrics_summary.columns.tolist()}")
        logging.info("\nUnique values for key columns:")
        for col in ['Target', 'Representation', 'DR_Method', 'Embedding_Strategy', 'DIM']:
            if col in df_all_metrics_summary.columns:
                logging.info(f"  {col}: {df_all_metrics_summary[col].unique().tolist()}")
        logging.info("\nHead of collected metrics:")
        try:
            logging.info(df_all_metrics_summary.head(20).to_string()) # Log more rows for debugging
        except Exception as e_log_df:
            logging.warning(f"Could not log df_all_metrics_summary head: {e_log_df}")
        logging.info(f"df_all_metrics_summary before DIM 2 filter (cols: {df_all_metrics_summary.columns.tolist()}):\n{df_all_metrics_summary.head()}")
        
        df_summary_dim2 = df_all_metrics_summary[df_all_metrics_summary['DIM'] == 2].copy()
        logging.info(f"df_summary_dim2 (cols: {df_summary_dim2.columns.tolist()}):\n{df_summary_dim2.head()}")

        if not df_summary_dim2.empty:
            cols_for_groupby_dim2 = ['ROC_AUC', 'PR_AUC', 'EF_1Perc', 'EF_5Perc', 'EF_10Perc', 'Spearman_Rho'] 
            available_cols_dim2 = [col for col in cols_for_groupby_dim2 if col in df_summary_dim2.columns]
            if available_cols_dim2:
                summary_table_dim2 = df_summary_dim2.groupby(['Representation', 'DR_Method', 'Embedding_Strategy'], as_index=False)[available_cols_dim2].mean()
                if 'ROC_AUC' in summary_table_dim2.columns:
                    summary_table_dim2.sort_values(by=['Representation', 'DR_Method', 'Embedding_Strategy', 'ROC_AUC'], ascending=[True, True, True, False], inplace=True)
                add_dataframe_as_latex_table_standalone(latex_content, summary_table_dim2,
                                             "Average Ranking Performance Metrics at 2 Dimensions across all targets, by DR Method and Embedding Strategy. Spearman Rho is averaged.",
                                             "overall-summary-dim2-metrics-by-strategy", font_size=r"\tiny")
            else: latex_content.append("No standard ranking metric columns found for DIM=2 summary table.\n")
        else: latex_content.append("No data for DIM=2 to generate summary table.\n")

        latex_content.append(f"\\clearpage\n{get_section_header_latex_standalone(2, 'Comparison of Embedding Strategies for PCA and UMAP')}")
        df_pca_umap_metrics = df_all_metrics_summary[
            df_all_metrics_summary['DR_Method'].str.contains("PCA|UMAP", case=False, na=False) &
            df_all_metrics_summary['Embedding_Strategy'].isin(["Projection", "Co-embedding"])
        ].copy()
        logging.info(f"df_pca_umap_metrics for strategy comparison plot (cols: {df_pca_umap_metrics.columns.tolist()}):\n{df_pca_umap_metrics.head()}")


        if not df_pca_umap_metrics.empty and 'ROC_AUC' in df_pca_umap_metrics.columns:
            avg_roc_strategies = df_pca_umap_metrics.groupby(
                ['Representation', 'DR_Method', 'Embedding_Strategy']
            )['ROC_AUC'].mean().unstack(level='Embedding_Strategy') 
            logging.info(f"avg_roc_strategies after unstack for plot (cols: {avg_roc_strategies.columns.tolist()}):\n{avg_roc_strategies.head()}")


            if not avg_roc_strategies.empty:
                avg_roc_strategies.reset_index(inplace=True)
                if "Projection" not in avg_roc_strategies.columns: avg_roc_strategies["Projection"] = np.nan
                if "Co-embedding" not in avg_roc_strategies.columns: avg_roc_strategies["Co-embedding"] = np.nan
                
                avg_roc_strategies['Plot_Label'] = avg_roc_strategies['DR_Method'] + " (" + avg_roc_strategies['Representation'] + ")"
                avg_roc_strategies_plot = avg_roc_strategies.dropna(subset=['Projection', 'Co-embedding'], how='all').copy()

                if not avg_roc_strategies_plot.empty:
                    plt.figure(figsize=(14, 8)) 
                    index = np.arange(len(avg_roc_strategies_plot))
                    bar_width = 0.35
                    bars1_data = avg_roc_strategies_plot['Projection'].fillna(0)
                    bars2_data = avg_roc_strategies_plot['Co-embedding'].fillna(0)
                    plt.bar(index - bar_width/2, bars1_data, bar_width, label='Projection', color='deepskyblue')
                    plt.bar(index + bar_width/2, bars2_data, bar_width, label='Co-embedding', color='salmon')
                    plt.xlabel("DR Method (Representation)", fontsize=13)
                    plt.ylabel("Average ROC-AUC (across Targets & DIMs)", fontsize=13)
                    plt.title("Comparison of Embedding Strategies (PCA/UMAP) by Average ROC-AUC", fontsize=15)
                    plt.xticks(index, avg_roc_strategies_plot['Plot_Label'], rotation=45, ha="right", fontsize=10)
                    plt.yticks(fontsize=10)
                    max_y_val_proj = avg_roc_strategies_plot['Projection'].max()
                    max_y_val_coem = avg_roc_strategies_plot['Co-embedding'].max()
                    overall_max_y = np.nanmax([max_y_val_proj, max_y_val_coem, 0.0])
                    plt.ylim(0, max(1.0, overall_max_y * 1.1) if pd.notna(overall_max_y) and overall_max_y > 0 else 1.0 )
                    plt.legend(fontsize=11)
                    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
                    plt.tight_layout()
                    plot_filename_strat = "embedding_strategy_comparison_roc_auc.png"
                    plot_abs_path_strat = os.path.join(report_figures_abs_dir, plot_filename_strat)
                    try:
                        plt.savefig(plot_abs_path_strat, dpi=150)
                        plot_relative_path_strat_tex = os.path.join("figures", plot_filename_strat).replace(os.sep, '/')
                        caption_strat = "Comparison of average ROC-AUC for Projection vs. Co-embedding strategies for PCA and UMAP methods. ROC-AUC values are averaged across all targets and tested dimensionalities for each specific DR method, representation, and embedding strategy."
                        add_figure_to_latex_standalone(latex_content, plot_relative_path_strat_tex, caption_strat, "fig-embed-strat-compare", figure_width="0.95\\linewidth")
                    except Exception as e: logging.error(f"Failed to save embedding strategy comparison plot {plot_abs_path_strat}: {e}")
                    plt.close()
                else:
                    latex_content.append("No data to plot after filtering for both Projection and Co-embedding ROC-AUC values for any DR method.\n")
                    logging.warning("No data to plot for embedding strategy comparison after filtering NaNs.")
            else:
                latex_content.append("The pivot operation for embedding strategy comparison resulted in an empty DataFrame or missing critical strategy columns.\n")
                logging.warning("Pivot for embedding strategy comparison resulted in empty DataFrame or missing columns.")
        else:
            latex_content.append("No PCA/UMAP metrics data for embedding strategy comparison or 'ROC_AUC' column missing.\n")
            logging.warning("No PCA/UMAP metrics or ROC_AUC column missing for strategy comparison plot.")

        if not df_all_metrics_summary.empty and 'ROC_AUC' in df_all_metrics_summary.columns:
            df_all_metrics_summary_no_nan_roc = df_all_metrics_summary.dropna(subset=['ROC_AUC']).copy()
            if not df_all_metrics_summary_no_nan_roc.empty:
                best_overall_idx = df_all_metrics_summary_no_nan_roc['ROC_AUC'].idxmax()
                best_overall_config = df_all_metrics_summary_no_nan_roc.loc[[best_overall_idx]]
                add_dataframe_as_latex_table_standalone(latex_content, best_overall_config,
                                                    "Overall Best Performing Configuration (by ROC-AUC across all Targets, Representations, DR Methods, Dimensions, and Embedding Strategies).",
                                                    "overall-best-config-global", font_size=r"\tiny")
            else: latex_content.append("No valid ROC-AUC data to determine overall best configuration.\n")
    else:
        latex_content.append("No ranking metrics data collected to generate overall summary tables.\n")

    latex_content.append(LATEX_DOCUMENT_END)
    # ... (rest of LaTeX compilation)
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
                logging.error(f"pdflatex compilation pass {i+1} FAILED with return code {process.returncode}.")
                latex_log_file = os.path.join(report_output_abs_dir, os.path.basename(report_tex_path).replace('.tex', '.log'))
                if os.path.exists(latex_log_file):
                    with open(latex_log_file, 'r', encoding='utf-8', errors='ignore') as lf:
                        log_content = lf.read()
                        error_lines = [line for line in log_content.splitlines() if line.startswith("! ") or " LaTeX Error:" in line or "Error:" in line or "Undefined control sequence" in line]
                        if error_lines: logging.error("Key LaTeX errors from log file:\n" + "\n".join(error_lines[:20]))
                        else: logging.error("No specific error lines found in .log, showing first 1KB of log:"); logging.error(log_content[:1024])
                else: 
                    logging.error("LaTeX .log file not found. STDOUT:\n" + process.stdout)
                    logging.error("STDERR:\n" + process.stderr)
            else: logging.info(f"pdflatex compilation pass {i+1} successful.")
        pdf_path = os.path.join(report_output_abs_dir, os.path.basename(report_tex_path).replace('.tex', '.pdf'))
        if os.path.exists(pdf_path): logging.info(f"PDF report successfully generated: {pdf_path}")
        else: logging.warning(f"PDF report {pdf_path} not found after compilation attempts, check logs.")
    except FileNotFoundError: logging.warning("pdflatex command not found. Please compile the .tex file manually.")
    except Exception as e: logging.error(f"An unexpected error occurred during pdflatex compilation: {e}")

if __name__ == "__main__":
    main()