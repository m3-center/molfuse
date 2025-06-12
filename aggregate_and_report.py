import pandas as pd
import numpy as np
import os
import argparse
import logging
import json
import shutil
import glob # For finding replicate directories
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt
import re
from datetime import datetime # For report run ID
import subprocess # For LaTeX compilation

# --- LaTeX Helper Functions & Preamble ---
# (These are assumed to be identical to the ones in your previous generate_latex_report.py)
# For brevity, I will not repeat them here, but they MUST be included in your actual file.
# This includes:
# - LATEX_DOCUMENT_PREAMBLE (with <<RUN_ID_PLACEHOLDER>>)
# - LATEX_DOCUMENT_END
# - escape_latex_text_content(text_input)
# - clean_for_label(text)
# - get_section_header_latex_standalone(level, title_text)
# - add_figure_to_latex_standalone(latex_content_list, ...)
# - add_dataframe_as_latex_table_standalone(latex_content_list, ...)
# Placeholder for required LaTeX preamble and helper functions
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
\title{UMMBAS Molecular Similarity Experimental Evaluation Report (Aggregated Results)\\ \large \textit{Report Generated: <<RUN_ID_PLACEHOLDER>>}}
\author{UMMBAS Project Team} \date{\today}
\begin{document} \maketitle \begin{abstract}This report presents aggregated results from multiple replicate experiments evaluating molecular similarity spaces. It compares embedding strategies (Projection vs. Co-embedding) for PCA and UMAP, alongside t-SNE. Metrics include ROC-AUC, PR-AUC, Enrichment Factors, and Spearman Correlation, averaged over replicates where applicable.\end{abstract} \clearpage \tableofcontents \clearpage \listoffigures \clearpage \listoftables \clearpage
\section{Introduction} \textit{Placeholder for Introduction text...}
\section{Methodology Overview} \textit{Placeholder for Methodology Overview text...}
\subsection{Performance Evaluation Metrics} \textit{Placeholder for Performance Metrics text...}
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
    sec_cmd_map = {1: r"\section", 2: r"\subsection", 3: r"\subsubsection", 4: r"\paragraph", 5: r"\subparagraph", 6: r"\textbf"} # Added 5,6
    sec_cmd = sec_cmd_map.get(level, r"\textbf")
    return f"\n{sec_cmd}{{{escape_latex_text_content(title_text)}}}\n"

def add_figure_to_latex_standalone(latex_content_list, relative_fig_path_in_tex, caption_text, label_text, placement="[H]", figure_width="0.7\\linewidth"):
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
            formats = ['l'] * len(df_for_latex.columns); 
            for i, cn_orig in enumerate(dataframe.columns): 
                if pd.api.types.is_numeric_dtype(dataframe[cn_orig]): formats[i] = 'r' 
            col_format = '|' + '|'.join(formats) + '|'
        latex_content_list.append(df_for_latex.to_latex(index=False, escape=False, column_format=col_format,
                                                        longtable=len(dataframe)>20, na_rep="N/A", booktabs=True))
        latex_content_list.extend([r"\end{table}", "\n"])
    else: latex_content_list.append(f"% Table '{escape_latex_text_content(label_text)}' empty.\n")
# --- End LaTeX Helpers ---


# Setup basic logging for this script
agg_log_file_name = f"aggregation_report_generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)-8s - %(filename)-25s - %(funcName)-25s - %(lineno)-4d - %(message)s',
    handlers=[
        logging.FileHandler(agg_log_file_name, mode='w'),
        logging.StreamHandler()
    ]
)

def find_replicate_runs(base_experiment_dir, expected_repr_mode_filter):
    replicate_dirs = []
    # Scan for directories matching the pattern run_seed<seed>_repr<repr_mode>_<timestamp>
    # The pattern should be robust to slight variations if needed.
    pattern = os.path.join(base_experiment_dir, f"run_seed*repr{expected_repr_mode_filter}_*")
    logging.info(f"Scanning for replicate run directories with pattern: {pattern}")
    
    for dir_path in glob.glob(pattern):
        if os.path.isdir(dir_path):
            # Extract seed and actual repr_mode from dirname to double check
            basename = os.path.basename(dir_path)
            try:
                parts = basename.split('_')
                seed_val = None
                repr_val = None
                for i, part in enumerate(parts):
                    if part == "seed" and i + 1 < len(parts):
                        seed_val = int(parts[i+1])
                    elif part == "repr" and i + 1 < len(parts):
                        repr_val = parts[i+1]
                
                if seed_val is not None and repr_val == expected_repr_mode_filter:
                    replicate_dirs.append({"path": dir_path, "seed": seed_val, "repr_mode": repr_val})
                    logging.info(f"Found valid replicate run: {dir_path} (Seed: {seed_val})")
                else:
                    logging.debug(f"Skipping directory (mismatch/parse error): {dir_path}")
            except ValueError:
                logging.warning(f"Could not parse seed from directory name: {basename}")
            except Exception as e:
                logging.warning(f"Error processing directory name {basename}: {e}")
                
    if not replicate_dirs:
        logging.warning(f"No replicate run directories found for representation '{expected_repr_mode_filter}' in '{base_experiment_dir}'")
    return sorted(replicate_dirs, key=lambda x: x["seed"]) # Sort by seed for consistency

def collect_metrics_from_replicates(replicate_run_details, config_main, report_for_representation_filter):
    all_metrics_data = []
    gs = config_main['global_settings']

    for rep_info in replicate_run_details:
        rep_dir_path = rep_info["path"]
        replicate_seed = rep_info["seed"]
        logging.info(f"Processing replicate run: {os.path.basename(rep_dir_path)} (Seed: {replicate_seed})")

        for target_info in config_main['targets']:
            target_id_name = target_info['id_name']
            target_display_name = target_info['display_name']
            target_processing_mode = target_info.get("processing_mode", "full_analysis")

            if target_processing_mode == "similarity_space_only":
                logging.debug(f"  Skipping metrics collection for target {target_id_name} (similarity_space_only mode).")
                continue

            target_results_base = os.path.join(rep_dir_path, target_id_name, "results")
            if not os.path.exists(target_results_base):
                logging.debug(f"  Results dir not found for target {target_id_name} in replicate {os.path.basename(rep_dir_path)}")
                continue
            
            # This script is run per representation, so repr_type loop should only run once
            repr_type = report_for_representation_filter 
            
            for dr_key_config, dr_params_config in config_main["dimensionality_reduction_methods"].items():
                dr_short_name_base = dr_params_config["short_name"]
                
                strategies_to_check = []
                # Strategy 1: Projection or Native tSNE
                strat1_dir_leaf_fs = dr_short_name_base.replace('-', '_')
                strat1_embedding_label = "Projection" if not dr_key_config == "tsne" else "Co-embedding (Native)"
                strategies_to_check.append({"output_dir_leaf_fs": strat1_dir_leaf_fs, "embedding_label": strat1_embedding_label})
                
                # Strategy 2: Co-embedding for PCA/UMAP
                if gs.get("run_coembedding_for_pca_umap", False) and \
                   dr_params_config.get("allow_coembedding", False) and \
                   not dr_key_config == "tsne":
                    strat2_dir_leaf_fs = f"{dr_short_name_base}-Coembed".replace('-', '_')
                    strat2_embedding_label = "Co-embedding"
                    strategies_to_check.append({"output_dir_leaf_fs": strat2_dir_leaf_fs, "embedding_label": strat2_embedding_label})

                for strategy_info in strategies_to_check:
                    strat_dir_leaf_fs = strategy_info["output_dir_leaf_fs"]
                    strat_label = strategy_info["embedding_label"]

                    for dim_val in gs['simspace_dims_to_test']:
                        if dr_key_config == "tsne" and dim_val != 2: continue
                        
                        # project_and_analyze.py saves metrics file named with its output_dir leaf name (strat_dir_leaf_fs)
                        metrics_filename = f"{target_id_name}_{repr_type}_{strat_dir_leaf_fs}_dim{dim_val}_ranking_metrics.csv"
                        metrics_file_path = os.path.join(target_results_base, repr_type, f"dim_{dim_val}", strat_dir_leaf_fs, metrics_filename)

                        if os.path.exists(metrics_file_path):
                            try:
                                df_m = pd.read_csv(metrics_file_path)
                                if not df_m.empty:
                                    metric_row = {
                                        'Replicate_Seed': replicate_seed, 'Target': target_display_name,
                                        'Representation': repr_type.capitalize(), 'DR_Method': dr_short_name_base,
                                        'Embedding_Strategy': strat_label, 'DIM': dim_val
                                    }
                                    for col_name in df_m.columns: # Add all metrics
                                        # Sanitize metric column names for DataFrame
                                        clean_col_name = col_name.replace('%', 'Perc').replace('-', '_').replace(' ', '_').upper()
                                        metric_row[clean_col_name] = df_m[col_name].iloc[0] if pd.notna(df_m[col_name].iloc[0]) else np.nan
                                    all_metrics_data.append(metric_row)
                            except Exception as e: logging.error(f"Error reading metrics {metrics_file_path}: {e}")
                        # else: logging.debug(f"Metrics file not found: {metrics_file_path}")
    
    if not all_metrics_data: logging.error("No metric data collected from any replicate runs.")
    return pd.DataFrame(all_metrics_data) if all_metrics_data else pd.DataFrame()


def main_report_generation():
    parser = argparse.ArgumentParser(description="Aggregate results and generate final LaTeX report.")
    parser.add_argument("--base_experiment_dir", required=True)
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--output_report_dir", required=True)
    parser.add_argument("--report_for_representation", required=True, choices=["features", "fingerprints"])
    args = parser.parse_args()

    logging.info(f"--- STARTING AGGREGATION AND REPORT GENERATION for Repr: {args.report_for_representation} ---")

    with open(args.config_path, 'r') as f: config = json.load(f)
    gs = config['global_settings']

    report_run_id = f"aggregated_report_repr_{args.report_for_representation}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    report_output_abs_dir = os.path.join(args.output_report_dir, report_run_id)
    report_figures_abs_dir = os.path.join(report_output_abs_dir, "figures")
    os.makedirs(report_figures_abs_dir, exist_ok=True)

    latex_content = [LATEX_DOCUMENT_PREAMBLE.replace("<<RUN_ID_PLACEHOLDER>>", escape_latex_text_content(report_run_id))]

    replicate_details_list = find_replicate_runs(args.base_experiment_dir, args.report_for_representation)
    if not replicate_details_list:
        logging.error("No replicate directories found. Cannot generate report."); latex_content.extend(["NO REPLICATE RUNS FOUND.", LATEX_DOCUMENT_END])
        with open(os.path.join(report_output_abs_dir, f"aggregated_report_repr_{args.report_for_representation}.tex"), "w") as f: f.write("\n".join(latex_content))
        return

    df_all_metrics_aggregated = collect_metrics_from_replicates(replicate_details_list, config, args.report_for_representation)
    if df_all_metrics_aggregated.empty:
        logging.error("Aggregated metrics DataFrame is empty. Report will be minimal."); latex_content.extend(["NO METRICS DATA COLLECTED.", LATEX_DOCUMENT_END])
        with open(os.path.join(report_output_abs_dir, f"aggregated_report_repr_{args.report_for_representation}.tex"), "w") as f: f.write("\n".join(latex_content))
        return
    
    logging.info(f"Aggregated metrics DataFrame shape: {df_all_metrics_aggregated.shape}")
    df_all_metrics_aggregated.to_csv(os.path.join(report_output_abs_dir, "DEBUG_all_aggregated_metrics.csv"), index=False) # For debugging

    # Report Structure
    latex_content.append(get_section_header_latex_standalone(1, f"Experimental Results by Target Protein (Representation: {args.report_for_representation.capitalize()})"))
    
    # Use first valid replicate for illustrative plots
    representative_replicate_dir = replicate_details_list[0]["path"] 
    representative_seed = replicate_details_list[0]["seed"]
    logging.info(f"Using replicate run '{os.path.basename(representative_replicate_dir)}' (Seed: {representative_seed}) for illustrative 2D plots.")

    for target_info in config['targets']:
        target_id_name = target_info['id_name']
        target_display_name = target_info['display_name']
        target_processing_mode = target_info.get("processing_mode", "full_analysis")
        target_label_name = clean_for_label(target_id_name)
        latex_content.append(f"\\clearpage\n{get_section_header_latex_standalone(2, f'Target: {target_display_name}')}")

        if target_processing_mode == "similarity_space_only":
            latex_content.append(f"Target {escape_latex_text_content(target_display_name)} was processed in 'similarity_space_only' mode. Ranked ZINC lists for docking were generated. No comparative metrics are presented here.\n")
            # Optionally, list found ranked_ZINC files from representative replicate
            ranked_zinc_base = os.path.join(representative_replicate_dir, target_id_name, "ranked_zinc_for_docking", args.report_for_representation)
            if os.path.exists(ranked_zinc_base):
                latex_content.append("\\textbf{Generated Ranked ZINC files (from representative replicate):}\n\\begin{itemize}\n")
                for root, _, files in os.walk(ranked_zinc_base):
                    for file in files:
                        if file.endswith("_ranked_ZINC.csv"):
                            latex_content.append(f"  \\item \\texttt{{{escape_latex_text_content(os.path.join(os.path.basename(root), file))}}}\n")
                latex_content.append("\\end{itemize}\n")
            continue

        # --- Dimensionality Optimization Plots (Mean +/- Std Dev) ---
        latex_content.append(get_section_header_latex_standalone(3, f"Optimization of Dimensionality & Strategy (Mean ROC-AUC $\\pm$ SD over {len(replicate_details_list)} Replicates)"))
        
        target_metrics = df_all_metrics_aggregated[
            (df_all_metrics_aggregated['Target'] == target_display_name) &
            (df_all_metrics_aggregated['Representation'] == args.report_for_representation.capitalize())
        ]

        if target_metrics.empty:
            latex_content.append("No metrics data found for this target and representation.\n")
        else:
            for dr_key_config, dr_params_config in config["dimensionality_reduction_methods"].items():
                dr_short_name_base = dr_params_config["short_name"]
                dr_label_base = clean_for_label(dr_short_name_base)
                
                plt.figure(figsize=(8, 5))
                plot_created_for_dr_method = False
                
                strategies_for_plot = target_metrics[target_metrics['DR_Method'] == dr_short_name_base]['Embedding_Strategy'].unique()

                for strat_label in strategies_for_plot:
                    strat_metrics = target_metrics[
                        (target_metrics['DR_Method'] == dr_short_name_base) &
                        (target_metrics['Embedding_Strategy'] == strat_label)
                    ]
                    if strat_metrics.empty: continue

                    # Group by DIM, calculate mean and std of ROC_AUC
                    dim_summary = strat_metrics.groupby('DIM')['ROC_AUC'].agg(['mean', 'std']).reset_index()
                    dim_summary.sort_values(by='DIM', inplace=True)
                    dim_summary['std'].fillna(0, inplace=True) # Std is NaN if only 1 replicate data point for a DIM

                    if not dim_summary.empty:
                        plt.plot(dim_summary['DIM'], dim_summary['mean'], marker='o', linestyle='-', label=strat_label)
                        plt.fill_between(dim_summary['DIM'], dim_summary['mean'] - dim_summary['std'], 
                                         dim_summary['mean'] + dim_summary['std'], alpha=0.2)
                        plot_created_for_dr_method = True
                
                if plot_created_for_dr_method:
                    plt.xlabel("SIMSPACE_DIM Value"); plt.ylabel("Mean ROC-AUC")
                    plt.title(f"DIM vs. Mean ROC-AUC for {target_display_name}\n({args.report_for_representation.capitalize()}, {dr_short_name_base})", fontsize=11)
                    if dr_key_config != "tsne": plt.xticks(gs['simspace_dims_to_test'])
                    else: plt.xticks([2])
                    plt.ylim(0.0, 1.05); plt.legend(title="Embedding Strategy")
                    plt.grid(True, linestyle='--', alpha=0.6)
                    plot_filename = f"{target_label_name}-{args.report_for_representation}-{dr_label_base}-dim-vs-mean-rocauc.png"
                    # ... (save and add figure to LaTeX) ...
                    plot_abs_path_dest = os.path.join(report_figures_abs_dir, plot_filename)
                    try:
                        plt.savefig(plot_abs_path_dest, dpi=150, bbox_inches='tight')
                        fig_path_tex = os.path.join("figures", plot_filename).replace(os.sep, '/')
                        add_figure_to_latex_standalone(latex_content, fig_path_tex, 
                            f"Mean ROC-AUC vs. SIMSPACE_DIM for {target_display_name} ({args.report_for_representation.capitalize()}, {dr_short_name_base}). Shaded area represents $\\pm$1 SD.",
                            f"fig-{target_label_name}-{dr_label_base}-dimopt-mean")
                    except Exception as e: logging.error(f"Failed to save dim_opt_mean plot: {e}")
                plt.close()
            if plot_created_for_dr_method: latex_content.append("\\clearpage\n")

        # --- Summary of Optimal Dimensionality (based on mean ROC_AUC) ---
        optimal_dim_rows = []
        if not target_metrics.empty:
            grouped_optimal = target_metrics.groupby(['Representation', 'DR_Method', 'Embedding_Strategy', 'DIM'])['ROC_AUC'].mean().reset_index()
            if not grouped_optimal.empty:
                idx = grouped_optimal.groupby(['Representation', 'DR_Method', 'Embedding_Strategy'])['ROC_AUC'].idxmax()
                df_optimal_dims = grouped_optimal.loc[idx][['Representation', 'DR_Method', 'Embedding_Strategy', 'DIM', 'ROC_AUC']]
                df_optimal_dims.rename(columns={'DIM': 'Optimal_DIM_Mean_ROC_AUC', 'ROC_AUC': 'Mean_ROC_AUC_at_Optimal_DIM'}, inplace=True)
                if not df_optimal_dims.empty:
                    add_dataframe_as_latex_table_standalone(latex_content, df_optimal_dims, 
                        f"Optimal SIMSPACE_DIM (by Mean ROC-AUC) for {target_display_name}, {args.report_for_representation.capitalize()}.",
                        f"tab-optimal-dim-{target_label_name}", font_size=r"\scriptsize")
        latex_content.append("\\clearpage\n")

        # --- Illustrative 2D Projections (from representative replicate) ---
        latex_content.append(get_section_header_latex_standalone(3, f"Illustrative 2D Projections (Seed: {representative_seed})"))
        # ... (Logic to find and copy 2D plots from representative_replicate_dir - similar to single run report, but paths include replicate dir)
        # ... (This loop structure is complex, ensure it correctly finds plot files based on strategy directory names)
        # ... Example:
        # for dr_key_2d, dr_params_2d_config in config["dimensionality_reduction_methods"].items():
        #    ... (determine strategies_2d_to_report based on config) ...
        #    for strategy_2d in strategies_2d_to_report:
        #        results_dir_leaf_2d = strategy_2d["output_dir_leaf_name_fs"]
        #        plot_title_name_2d = strategy_2d["plot_title_name"] # This should be like "PCA" or "PCA-Coembed"
        #        dim2_results_path_in_rep = os.path.join(representative_replicate_dir, target_id_name, "results", args.report_for_representation, "dim_2", results_dir_leaf_2d)
        #        scatter_fname = f"{target_id_name}_{args.report_for_representation}_{results_dir_leaf_2d}_dim2_scatter.png"
        #        # ... copy and add_figure_to_latex_standalone ...

    # --- Overall Comparative Summary (Aggregated) ---
    latex_content.append(f"\\clearpage\n{get_section_header_latex_standalone(1, 'Overall Comparative Summary of Ranking Performance (Aggregated)')}")
    
    # Table: Mean and Std Dev Metrics at 2D
    df_summary_dim2_agg = df_all_metrics_aggregated[df_all_metrics_aggregated['DIM'] == 2]
    if not df_summary_dim2_agg.empty:
        summary_table_dim2_agg = df_summary_dim2_agg.groupby(
            ['Representation', 'DR_Method', 'Embedding_Strategy']
        ).agg(
            Mean_ROC_AUC=('ROC_AUC', 'mean'), SD_ROC_AUC=('ROC_AUC', 'std'),
            Mean_PR_AUC=('PR_AUC', 'mean'), SD_PR_AUC=('PR_AUC', 'std'),
            Mean_EF_1Perc=('EF_1PERC', 'mean'), SD_EF_1Perc=('EF_1PERC', 'std'), # Ensure col name matches after upper()
            Mean_EF_5Perc=('EF_5PERC', 'mean'), SD_EF_5Perc=('EF_5PERC', 'std'),
            Mean_EF_10Perc=('EF_10PERC', 'mean'), SD_EF_10Perc=('EF_10PERC', 'std'),
            Mean_Spearman_Rho=('SPEARMAN_RHO', 'mean'), SD_Spearman_Rho=('SPEARMAN_RHO', 'std')
        ).reset_index()
        summary_table_dim2_agg.sort_values(by=['Representation', 'DR_Method', 'Embedding_Strategy', 'Mean_ROC_AUC'], 
                                           ascending=[True, True, True, False], inplace=True)
        add_dataframe_as_latex_table_standalone(latex_content, summary_table_dim2_agg,
            "Aggregated Ranking Metrics (Mean $\\pm$ SD over Replicates) at 2D, by DR Method and Embedding Strategy.",
            "tab-agg-summary-dim2", font_size=r"\tiny")
    else: latex_content.append("No DIM=2 data for aggregated summary table.\n")

    # Figure: Comparison of Embedding Strategies (Mean ROC_AUC +/- SD)
    latex_content.append(f"\\clearpage\n{get_section_header_latex_standalone(2, 'Comparison of Embedding Strategies for PCA and UMAP (Aggregated)')}")
    # ... (Filter df_all_metrics_aggregated for PCA/UMAP, "Projection" & "Co-embedding")
    # ... (Group by Repr, DR_Method, Embedding_Strategy, DIM -> mean ROC_AUC, std ROC_AUC)
    # ... (Unstack Embedding_Strategy)
    # ... (Plot grouped bar chart with error bars for std dev for each DIM, or average over DIMs and plot that)
    # This plot might become very busy if showing all DIMs. Averaging over DIMs is simpler for one plot:
    df_pca_umap_agg = df_all_metrics_aggregated[
        df_all_metrics_aggregated['DR_Method'].str.contains("PCA|UMAP", case=False, na=False) &
        df_all_metrics_aggregated['Embedding_Strategy'].isin(["Projection", "Co-embedding"])
    ]
    if not df_pca_umap_agg.empty:
        avg_roc_strat_agg = df_pca_umap_agg.groupby(
            ['Representation', 'DR_Method', 'Embedding_Strategy']
        )['ROC_AUC'].agg(['mean', 'std']).unstack(level='Embedding_Strategy')
        
        # avg_roc_strat_agg will have multi-level columns like ('mean', 'Projection'), ('std', 'Projection')
        if not avg_roc_strat_agg.empty and \
           (('mean', 'Projection') in avg_roc_strat_agg.columns and ('mean', 'Co-embedding') in avg_roc_strat_agg.columns):
            
            avg_roc_strat_agg.columns = ['_'.join(col).strip() for col in avg_roc_strat_agg.columns.values] # Flatten MultiIndex
            avg_roc_strat_agg.reset_index(inplace=True)
            avg_roc_strat_agg['Plot_Label'] = avg_roc_strat_agg['DR_Method'] + " (" + avg_roc_strat_agg['Representation'] + ")"
            
            # Ensure std columns exist and fillna for plotting
            for col in ['mean_Projection', 'std_Projection', 'mean_Co-embedding', 'std_Co-embedding']:
                if col not in avg_roc_strat_agg.columns: avg_roc_strat_agg[col] = 0 # or np.nan then .fillna(0)
            avg_roc_strat_agg.fillna(0, inplace=True) # Fill remaining NaNs with 0 for std if only 1 data point

            plt.figure(figsize=(14, 8))
            index = np.arange(len(avg_roc_strat_agg))
            bar_width = 0.35
            plt.bar(index - bar_width/2, avg_roc_strat_agg['mean_Projection'], bar_width, 
                    yerr=avg_roc_strat_agg['std_Projection'], label='Projection', color='deepskyblue', capsize=5, ecolor='darkgrey')
            plt.bar(index + bar_width/2, avg_roc_strat_agg['mean_Co-embedding'], bar_width, 
                    yerr=avg_roc_strat_agg['std_Co-embedding'], label='Co-embedding', color='salmon', capsize=5, ecolor='darkgrey')
            # ... (labels, title, xticks, legend, save - similar to single run report, but using mean/std) ...
            plt.xlabel("DR Method (Representation)", fontsize=13)
            plt.ylabel("Mean ROC-AUC (across Targets & DIMs $\\pm$ SD over Replicates)", fontsize=13)
            plt.title("Aggregated Comparison of Embedding Strategies by Mean ROC-AUC", fontsize=15)
            plt.xticks(index, avg_roc_strat_agg['Plot_Label'], rotation=45, ha="right", fontsize=10)
            max_y_val = avg_roc_strat_agg[['mean_Projection', 'mean_Co-embedding']].max().max()
            plt.ylim(0, max(1.0, max_y_val * 1.1 + avg_roc_strat_agg[['std_Projection', 'std_Co-embedding']].max().max() * 0.1) if pd.notna(max_y_val) else 1.0) # Adjust ylim for error bars
            plt.legend(fontsize=11); plt.grid(True, linestyle='--', alpha=0.7, axis='y'); plt.tight_layout()
            plot_filename_agg_strat = "aggregated_embedding_strategy_comparison_roc_auc.png"
            # ... (save and add figure to latex)
        else: latex_content.append("Insufficient data for aggregated embedding strategy comparison plot.\n")
    else: latex_content.append("No PCA/UMAP data for aggregated strategy comparison.\n")


    # Table: Overall Best Config (based on highest mean ROC_AUC across all replicates, targets, etc.)
    if not df_all_metrics_aggregated.empty and 'ROC_AUC' in df_all_metrics_aggregated.columns:
        mean_roc_per_config = df_all_metrics_aggregated.groupby(
            ['Target', 'Representation', 'DR_Method', 'Embedding_Strategy', 'DIM']
        )['ROC_AUC'].mean().reset_index()
        if not mean_roc_per_config.empty:
            best_overall_config_mean = mean_roc_per_config.loc[[mean_roc_per_config['ROC_AUC'].idxmax()]]
            add_dataframe_as_latex_table_standalone(latex_content, best_overall_config_mean,
                "Overall Best Performing Configuration (by Mean ROC-AUC across Replicates).",
                "tab-agg-best-overall", font_size=r"\tiny")
    
    latex_content.append(LATEX_DOCUMENT_END)

    # Save and compile LaTeX
    report_tex_filename = f"aggregated_report_repr_{args.report_for_representation}.tex"
    # ... (save and compile logic - same as before) ...
    report_tex_path = os.path.join(report_output_abs_dir, report_tex_filename)
    with open(report_tex_path, "w", encoding='utf-8') as f: f.write("\n".join(latex_content))
    logging.info(f"Aggregated LaTeX report structure generated: {report_tex_path}")
    try:
        logging.info(f"Attempting to compile LaTeX report in: {report_output_abs_dir}")
        for i in range(2): 
            process = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-output-directory", report_output_abs_dir, report_tex_path],
                capture_output=True, text=True, check=False )
            if process.returncode != 0: logging.error(f"pdflatex compilation pass {i+1} FAILED."); #... (log details)
            else: logging.info(f"pdflatex compilation pass {i+1} successful.")
        # ... (check PDF existence)
    except Exception as e: logging.error(f"LaTeX compilation error: {e}")


if __name__ == "__main__":
    main_report_generation()