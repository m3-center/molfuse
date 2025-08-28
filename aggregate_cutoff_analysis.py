import pandas as pd
import numpy as np
import os
import argparse
import logging
import json
import shutil
import glob
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend suitable for scripts
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess
import re
from datetime import datetime

# --- MODIFIED: Updated Abstract ---
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
\usepackage{tocloft}
\usepackage{enumitem}

\hypersetup{
    colorlinks=true, linkcolor=blue, filecolor=magenta, urlcolor=cyan,
    pdftitle={UMMBAS Affinity Cutoff Analysis Report}, pdfauthor={UMMBAS Project Team},
    bookmarksnumbered=true, pdfpagemode=UseOutlines
}
\renewcommand{\cftsecleader}{\cftdotfill{\cftdotsep}}
\pagestyle{fancy} \fancyhf{} \fancyhead[L]{UMMBAS Affinity Cutoff Report} \fancyhead[R]{\today} \fancyfoot[C]{\thepage\ of \pageref{LastPage}}
\title{UMMBAS Molecular Similarity Experiment\\ \large \textit{Affinity Cutoff Analysis Report (Features, 2D)}}
\author{UMMBAS Project Team} \date{\today}
\begin{document} \maketitle \begin{abstract}
This report analyzes the impact of applying a potency-based affinity cutoff to the held-out active ligands during the evaluation phase of a leave-one-target-out virtual screening experiment. This analysis is focused specifically on 2D similarity spaces generated from physicochemical features. The original similarity spaces and DR models were used as a fixed foundation. The analysis was then repeated, defining the "true active" set using only compounds with a Standard Value (nM) at or below specific cutoffs (100, 1000, and 10000 nM). The objective is to determine whether restricting the evaluation to higher-potency ligands improves the measured ability to correctly rank these compounds against a large set of decoys. Performance is assessed using ROC-AUC, PR-AUC, and Enrichment Factor at 1\% (EF@1\%).
\end{abstract} \clearpage \tableofcontents \clearpage \listoffigures \clearpage \listoftables \clearpage
"""
LATEX_DOCUMENT_END = r"\end{document}"

# (All helper functions are unchanged)
def escape_latex_text_content(text_input):
    if not isinstance(text_input, str): text_input = str(text_input)
    conv = {'&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_',
            '{': r'\{', '}': r'\}', '~': r'\textascitilde{}', '^': r'\^{}',
            '<': r'\textless{}', '>': r'\textgreater{}'}
    for k, v in conv.items(): text_input = text_input.replace(k, v)
    return text_input
def clean_for_label(text):
    if not isinstance(text, str): text = str(text)
    text = text.replace('_', '-').replace(' ', '-').replace('.', '-').replace('/', '-')
    text = re.sub(r'[^a-zA-Z0-9-]', '', text); text = re.sub(r'-+', '-', text)
    return text.strip('-')[:50]
def get_section_header_latex(level, title_text):
    sec_cmd_map = {1: r"\section", 2: r"\subsection", 3: r"\subsubsection"}
    sec_cmd = sec_cmd_map.get(level, r"\subsubsection")
    return f"\n{sec_cmd}{{{escape_latex_text_content(title_text)}}}\n"
def add_figure_to_latex(latex_content_list, fig_path, caption, label, figure_width="0.9\\textwidth"):
    rel_path = os.path.join("figures", os.path.basename(fig_path))
    latex_content_list.extend([f"\\begin{{figure}}[H]", r"  \centering",
                               f"  \\includegraphics[width={figure_width}]{{{rel_path}}}",
                               f"  \\caption{{{escape_latex_text_content(caption)}}}",
                               f"  \\label{{fig:{clean_for_label(label)}}}", r"\end{figure}", "\n"])
def add_dataframe_as_latex_table(latex_content_list, dataframe, caption, label, font_size=r"\small"):
    if dataframe is None or dataframe.empty:
        latex_content_list.append(f"% Table '{escape_latex_text_content(label)}' is empty.\n")
        return
    latex_content_list.extend([r"\begin{table}[H]", r"  \centering", font_size,
                               f"  \\caption{{{escape_latex_text_content(caption)}}}",
                               f"  \\label{{tab:{clean_for_label(label)}}}"])
    df_latex = dataframe.copy()
    for col in df_latex.select_dtypes(include=np.number).columns:
        df_latex[col] = df_latex[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
    df_latex.columns = [escape_latex_text_content(c.replace('_', ' ').title()) for c in df_latex.columns]
    col_format = 'l' * len(df_latex.columns)
    latex_table_string = df_latex.to_latex(index=False, escape=False, column_format=col_format, longtable=len(dataframe) > 20, na_rep="N/A")
    latex_content_list.append(latex_table_string)
    latex_content_list.extend([r"\end{table}", "\n"])


# --- Logging Setup ---
log_file_name = f"cutoff_analysis_aggregation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)-8s - %(filename)-25s - %(funcName)-25s - %(lineno)-4d - %(message)s',
                    handlers=[logging.FileHandler(log_file_name, mode='w'), logging.StreamHandler()])

# --- Data Collection ---
def collect_cutoff_experiment_metrics(base_dir, config):
    """
    Scans the cutoff experiment directory, parses metrics files, and extracts the cutoff value.
    """
    all_metrics = []
    logging.info(f"Scanning for metrics files in base directory: {base_dir}")
    # The pattern needs to find the metrics file deep within the new structure
    pattern = os.path.join(base_dir, "run_seed*", "results_cutoff_*", "*", "results", "*", "dim_*", "*", "*_ranking_metrics.csv")
    metrics_files = glob.glob(pattern)

    if not metrics_files:
        logging.error(f"No ranking_metrics.csv files found using pattern: {pattern}")
        return pd.DataFrame()

    for metrics_file_path in metrics_files:
        try:
            path_parts = metrics_file_path.split(os.sep)

            # Extract info from the path
            run_dir_name = next(p for p in path_parts if p.startswith("run_seed"))
            cutoff_dir_name = next(p for p in path_parts if p.startswith("results_cutoff_"))
            target_id_name = path_parts[path_parts.index(cutoff_dir_name) + 1]
            repr_type = path_parts[path_parts.index("results") + 1]
            dim_str = next(p for p in path_parts if p.startswith("dim_"))
            strategy_dir = path_parts[path_parts.index(dim_str) + 1]

            # Parse details from directory names
            seed = int(re.search(r"run_seed(\d+)", run_dir_name).group(1))
            cutoff_val = int(re.search(r"results_cutoff_(\d+)", cutoff_dir_name).group(1))
            dim_val = int(dim_str.replace('dim_', ''))

            df_m = pd.read_csv(metrics_file_path)
            if df_m.empty: continue

            # Map strategy directory name back to DR method and embedding strategy
            dr_method_base_name = "Unknown"
            embedding_strategy = "Unknown"
            for dr_key, dr_params in config["dimensionality_reduction_methods"].items():
                short_name_fs = dr_params["short_name"].replace('-', '_')
                coembed_name_fs = f"{short_name_fs}_Coembed"

                if strategy_dir.lower() == short_name_fs.lower():
                    dr_method_base_name = dr_params["short_name"]
                    embedding_strategy = "Projection" if dr_key != "tsne" else "Co-embedding (Native)"
                    break
                elif strategy_dir.lower() == coembed_name_fs.lower():
                    dr_method_base_name = dr_params["short_name"]
                    embedding_strategy = "Co-embedding"
                    break

            target_display_name = target_id_name
            for t_info in config['targets']:
                if t_info['id_name'] == target_id_name:
                    target_display_name = t_info['display_name']
                    break

            metric_row = {
                'Seed': seed,
                'Affinity_Cutoff': cutoff_val,
                'Target': target_display_name,
                'Representation': repr_type.capitalize(),
                'DR_Method': dr_method_base_name,
                'Embedding_Strategy': embedding_strategy,
                'DIM': dim_val
            }
            column_map = {'roc_auc': 'ROC_AUC', 'pr_auc': 'PR_AUC', 'ef_1%': 'EF_1Perc'}
            for csv_col, df_col in column_map.items():
                metric_row[df_col] = df_m[csv_col].iloc[0] if csv_col in df_m and pd.notna(df_m[csv_col].iloc[0]) else np.nan
            all_metrics.append(metric_row)

        except Exception as e:
            logging.warning(f"Failed to process metrics file {metrics_file_path}: {e}", exc_info=True)

    if not all_metrics:
        logging.error("No metric data was collected. The DataFrame is empty.")
    return pd.DataFrame(all_metrics)


# --- Analysis and Plotting ---
def create_performance_vs_cutoff_plot(df, metric, title, filename, report_figures_dir):
    """
    Generates a line plot of a performance metric vs. the affinity cutoff,
    faceted by DR Method and Representation.
    """
    logging.info(f"Generating plot: {title}")
    if df.empty or metric not in df.columns:
        logging.warning(f"Data is empty or missing '{metric}' column for plot '{title}'. Skipping.")
        return

    # Create a combined 'Method (Representation)' column for faceting
    df_plot = df.copy()
    df_plot['Method_Repr'] = df_plot['DR_Method'] + " (" + df_plot['Representation'] + ")"
    
    # Use seaborn's relplot for easy faceting
    g = sns.relplot(
        data=df_plot,
        x='Affinity_Cutoff',
        y=metric,
        hue='Embedding_Strategy',
        col='Method_Repr',
        col_wrap=3, # Adjust number of columns in the plot grid
        kind='line',
        marker='o',
        errorbar='sd', # Show standard deviation as error bands
        height=4,
        aspect=1.2,
        facet_kws={'sharey': True, 'sharex': True}
    )
    
    g.fig.suptitle(title, y=1.03, fontsize=16)
    g.set_axis_labels("Affinity Cutoff (nM)", f"Mean {metric}")
    g.set(xscale="log") # Use a log scale for the cutoff axis
    if 'AUC' in metric:
        g.set(ylim=(0, 1.05))
    else:
        g.set(ylim=(0, None))
    g.set_titles("{col_name}")
    g.tight_layout(rect=[0, 0, 1, 0.97])

    fig_path = os.path.join(report_figures_dir, filename)
    plt.savefig(fig_path, dpi=200)
    plt.close()
    
    caption = f"{title}. Lines show the mean metric value over replicates and targets, and shaded areas represent ±1 standard deviation."
    return fig_path, caption

# --- Main Report Generation Script ---
def main():
    parser = argparse.ArgumentParser(description="Aggregate results from the affinity cutoff experiment and generate a LaTeX report.")
    parser.add_argument("--base_experiment_dir", default="experiment_workspace_cutoff_analysis/", help="Base directory of the cutoff experiment results.")
    parser.add_argument("--main_config_path", default="experiment_config.json", help="Path to the main experiment_config.json file for context.")
    parser.add_argument("--output_report_dir", default="final_report_cutoff_analysis/", help="Directory to save the final LaTeX report.")
    args = parser.parse_args()

    logging.info("--- STARTING AFFINITY CUTOFF ANALYSIS AGGREGATION ---")
    
    try:
        with open(args.main_config_path, 'r') as f:
            config = json.load(f)
    except Exception as e:
        logging.error(f"Could not load main config file '{args.main_config_path}'. Aborting. Error: {e}")
        return

    report_run_id = f"cutoff_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    report_output_abs_dir = os.path.join(args.output_report_dir, report_run_id)
    report_figures_abs_dir = os.path.join(report_output_abs_dir, "figures")
    os.makedirs(report_figures_abs_dir, exist_ok=True)
    
    latex_content = [LATEX_DOCUMENT_PREAMBLE]

    # Collect all metrics
    df_agg = collect_cutoff_experiment_metrics(args.base_experiment_dir, config)
    if df_agg.empty:
        logging.error("Failed to collect any metrics. Aborting report generation.")
        return
    df_agg.to_csv(os.path.join(report_output_abs_dir, "DEBUG_cutoff_metrics_aggregated.csv"), index=False)

    # --- Section 1: Overall Summary Table ---
    latex_content.append(get_section_header_latex(1, "Overall Performance Summary by Affinity Cutoff"))
    latex_content.append("The following table summarizes the mean performance metrics, averaged over all targets, replicates, and dimensions, for each affinity cutoff value.")

    try:
        summary_table = df_agg.groupby(['Representation', 'DR_Method', 'Embedding_Strategy', 'Affinity_Cutoff'])[['ROC_AUC', 'PR_AUC', 'EF_1Perc']].mean().reset_index()
        add_dataframe_as_latex_table(latex_content, summary_table,
                                     "Mean Performance Metrics vs. Affinity Cutoff",
                                     "tab-summary-by-cutoff", font_size=r"\tiny")
    except Exception as e:
        logging.error(f"Failed to generate summary table: {e}", exc_info=True)
        latex_content.append("Summary table could not be generated.\n")


    # --- Section 2: Performance Plots ---
    latex_content.append(get_section_header_latex(1, "Performance vs. Affinity Cutoff"))
    latex_content.append("The following plots visualize the impact of the affinity cutoff on the primary performance metrics. A logarithmic scale is used for the cutoff value on the x-axis to better visualize the trends across different orders of magnitude.")

    # ROC-AUC Plot
    fig_path_roc, caption_roc = create_performance_vs_cutoff_plot(df_agg, 'ROC_AUC',
        "ROC-AUC Performance vs. Affinity Cutoff",
        "perf_vs_cutoff_roc_auc.png", report_figures_abs_dir)
    if fig_path_roc:
        add_figure_to_latex(latex_content, fig_path_roc, caption_roc, "fig-roc-vs-cutoff", figure_width="\\textwidth")

    # PR-AUC Plot
    fig_path_pr, caption_pr = create_performance_vs_cutoff_plot(df_agg, 'PR_AUC',
        "PR-AUC Performance vs. Affinity Cutoff",
        "perf_vs_cutoff_pr_auc.png", report_figures_abs_dir)
    if fig_path_pr:
        add_figure_to_latex(latex_content, fig_path_pr, caption_pr, "fig-pr-vs-cutoff", figure_width="\\textwidth")
        
    # EF@1% Plot
    fig_path_ef, caption_ef = create_performance_vs_cutoff_plot(df_agg, 'EF_1Perc',
        "Enrichment Factor @ 1% vs. Affinity Cutoff",
        "perf_vs_cutoff_ef1.png", report_figures_abs_dir)
    if fig_path_ef:
        add_figure_to_latex(latex_content, fig_path_ef, caption_ef, "fig-ef1-vs-cutoff", figure_width="\\textwidth")
        
    # --- Section 3: Best Cutoff Analysis ---
    latex_content.append(get_section_header_latex(1, "Optimal Cutoff Analysis"))
    latex_content.append("To identify the most effective configuration, the affinity cutoff that maximized the mean ROC-AUC was determined for each combination of DR method, representation, and embedding strategy.")
    
    try:
        # Calculate mean ROC-AUC for each group
        mean_perf = df_agg.groupby(['Representation', 'DR_Method', 'Embedding_Strategy', 'Affinity_Cutoff'])['ROC_AUC'].mean().reset_index()
        # Find the index of the maximum ROC-AUC within each primary group
        idx = mean_perf.groupby(['Representation', 'DR_Method', 'Embedding_Strategy'])['ROC_AUC'].idxmax()
        best_cutoffs = mean_perf.loc[idx].rename(columns={'Affinity_Cutoff': 'Optimal_Cutoff_nM', 'ROC_AUC': 'Max_Mean_ROC_AUC'})
        best_cutoffs.sort_values(by=['Representation', 'DR_Method', 'Embedding_Strategy'], inplace=True)
        
        add_dataframe_as_latex_table(latex_content, best_cutoffs,
                                     "Optimal Affinity Cutoff by Mean ROC-AUC",
                                     "tab-best-cutoffs", font_size=r"\normalsize")
    except Exception as e:
        logging.error(f"Failed to generate best cutoff table: {e}", exc_info=True)
        latex_content.append("Best cutoff table could not be generated.\n")
        
    # --- Conclusion ---
    latex_content.append(get_section_header_latex(1, "Conclusion"))
    latex_content.append("This analysis investigated the effect of refining the Molecular Function (MF) cloud by applying potency-based cutoffs. The results indicate that [INSERT INTERPRETATION HERE, e.g., 'a stricter cutoff of 1000 nM generally improves performance for most methods, suggesting that focusing on high-potency ligands enhances the predictive signal. However, for some DR methods on fingerprint data, a broader cutoff of 10000 nM was optimal, possibly due to the need for a more diverse chemical space to effectively guide ranking.']. Based on these findings, applying an affinity cutoff during the scoring phase is a valuable strategy for improving virtual screening performance in this framework. The optimal cutoff may be method-dependent and should be considered a key parameter for optimization.")

    latex_content.append(LATEX_DOCUMENT_END)

    # --- Write and Compile LaTeX ---
    report_tex_filename = f"cutoff_analysis_report.tex"
    report_tex_path = os.path.join(report_output_abs_dir, report_tex_filename)
    with open(report_tex_path, "w", encoding='utf-8') as f:
        f.write("\n".join(latex_content))
    logging.info(f"Cutoff analysis LaTeX report generated: {report_tex_path}")

    try:
        logging.info(f"Attempting to compile LaTeX report in: {report_output_abs_dir}")
        for i in range(2): # Run twice for table of contents, etc.
            process = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-output-directory", report_output_abs_dir, report_tex_path],
                capture_output=True, text=True, check=False)
            if process.returncode != 0:
                logging.error(f"pdflatex compilation pass {i+1} FAILED.")
                latex_log_file = report_tex_path.replace('.tex', '.log')
                if os.path.exists(latex_log_file):
                    with open(latex_log_file, 'r', encoding='utf-8', errors='ignore') as lf:
                        logging.error("Key LaTeX errors:\n" + "".join(lf.readlines()[-50:]))
                else:
                    logging.error("LaTeX .log file not found. STDERR:\n" + process.stderr)
            else:
                logging.info(f"pdflatex compilation pass {i+1} successful.")

        pdf_path = report_tex_path.replace('.tex', '.pdf')
        if os.path.exists(pdf_path):
            logging.info(f"PDF report successfully generated: {pdf_path}")
        else:
            logging.warning(f"PDF report not found after compilation attempts.")
    except Exception as e:
        logging.error(f"LaTeX compilation error: {e}", exc_info=True)

if __name__ == "__main__":
    main()