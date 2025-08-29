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
\usepackage{tocloft}
\usepackage{enumitem}

\hypersetup{
    colorlinks=true, linkcolor=blue, filecolor=magenta, urlcolor=cyan,
    pdftitle={UMMBAS Affinity Cutoff Analysis Report}, pdfauthor={UMMBAS Project Team},
    bookmarksnumbered=true, pdfpagemode=UseOutlines
}
\renewcommand{\cftsecleader}{\cftdotfill{\cftdotsep}}
\pagestyle{fancy} \fancyhf{} \fancyhead[L]{UMMBAS Affinity Cutoff Report} \fancyhead[R]{\today} \fancyfoot[C]{\thepage\ of \pageref{LastPage}}
\title{UMMBAS Molecular Similarity Experiment\\ \large \textit{Affinity Cutoff Analysis Report}}
\author{UMMBAS Project Team} \date{\today}
\begin{document} \maketitle \begin{abstract}
This report analyzes the impact of applying a potency-based affinity cutoff to the held-out active ligands during the evaluation phase of a leave-one-target-out virtual screening experiment. This analysis is focused specifically on 2D similarity spaces generated from physicochemical features. For each dimensionality reduction method and embedding strategy, the single best-performing hyperparameter configuration was first identified using a baseline affinity cutoff of 100,000 nM. The analysis was then repeated using only these optimal configurations, varying the affinity cutoff (100, 1000, 10000, and 100000 nM) to determine its effect on the peak performance of each method. Performance is assessed using ROC-AUC, PR-AUC, and Enrichment Factor at 1\% (EF@1\%).
\end{abstract} \clearpage \tableofcontents \clearpage \listoffigures \clearpage \listoftables \clearpage
"""
LATEX_DOCUMENT_END = r"\end{document}"
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
        if df_latex[col].dropna().apply(lambda x: x.is_integer()).all():
            df_latex[col] = df_latex[col].apply(lambda x: f"{int(x)}" if pd.notna(x) else "N/A")
        else:
            df_latex[col] = df_latex[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
    df_latex.columns = [escape_latex_text_content(c.replace('_', ' ').title()) for c in df_latex.columns]
    col_format = 'l' * len(df_latex.columns)
    latex_table_string = df_latex.to_latex(index=False, escape=False, column_format=col_format, longtable=len(dataframe) > 20, na_rep="N/A")
    latex_content_list.append(latex_table_string)
    latex_content_list.extend([r"\end{table}", "\n"])

# --- Logging and Data Collection ---
log_file_name = f"cutoff_analysis_aggregation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)-8s - %(filename)-25s - %(funcName)-25s - %(lineno)-4d - %(message)s',
                    handlers=[logging.FileHandler(log_file_name, mode='w'), logging.StreamHandler()])

def collect_cutoff_experiment_metrics(cutoff_results_dir, original_hyperparam_dir):
    # (This function is unchanged and correct)
    all_metrics = []
    pattern = os.path.join(cutoff_results_dir, "run_seed*", "results_cutoff_*", "*", "results", "*", "dim_*", "*", "*_ranking_metrics.csv")
    for metrics_file_path in glob.glob(pattern):
        try:
            path_parts = metrics_file_path.split(os.sep)
            run_dir_name = next(p for p in path_parts if p.startswith("run_seed"))
            cutoff_dir_name = next(p for p in path_parts if p.startswith("results_cutoff_"))
            repr_type = path_parts[path_parts.index("results") + 1]
            strategy_dir = path_parts[path_parts.index(next(p for p in path_parts if p.startswith("dim_"))) + 1]
            seed = int(re.search(r"run_seed(\d+)", run_dir_name).group(1))
            cutoff_val = int(re.search(r"results_cutoff_(\d+)", cutoff_dir_name).group(1))
            run_config_path = os.path.join(original_hyperparam_dir, run_dir_name, "run_config.json")
            if not os.path.exists(run_config_path): continue
            with open(run_config_path, 'r') as f: run_config = json.load(f)
            dr_method_key = list(run_config['dimensionality_reduction_methods'].keys())[0]
            dr_params = run_config['dimensionality_reduction_methods'][dr_method_key]
            hyperparam_name, hyperparam_value = "N/A", "N/A"
            if 'n_neighbors' in dr_params:
                hyperparam_name = 'n_neighbors'; hyperparam_value = dr_params['n_neighbors']
            elif 'perplexity' in dr_params:
                hyperparam_name = 'perplexity'; hyperparam_value = dr_params['perplexity']
            embedding_strategy = "Projection"
            if "tsne" in dr_method_key: embedding_strategy = "Co-embedding (Native)"
            elif strategy_dir.endswith('_Coembed'): embedding_strategy = "Co-embedding"
            df_m = pd.read_csv(metrics_file_path)
            if df_m.empty: continue
            metric_row = {'Seed': seed, 'Affinity_Cutoff': cutoff_val, 'Representation': repr_type.capitalize(),
                          'DR_Method': dr_params['short_name'], 'Embedding_Strategy': embedding_strategy,
                          'Hyperparameter': hyperparam_name, 'Hyperparameter_Value': hyperparam_value}
            for col in ['roc_auc', 'pr_auc', 'ef_1%']:
                metric_row[col.upper()] = df_m[col].iloc[0] if col in df_m and pd.notna(df_m[col].iloc[0]) else np.nan
            all_metrics.append(metric_row)
        except Exception as e:
            logging.warning(f"Failed to process metrics file {metrics_file_path}: {e}", exc_info=True)
    return pd.DataFrame(all_metrics)

def create_performance_vs_cutoff_plot(df, metric, title, filename, report_figures_dir):
    # (This function is unchanged and correct)
    if df.empty or metric not in df.columns:
        logging.warning(f"Data is empty or missing '{metric}' for plot '{title}'."); return None, None
    g = sns.relplot(data=df, x='Affinity_Cutoff', y=metric, hue='Embedding_Strategy',
                    col='Method_Repr', col_wrap=3, kind='line', marker='o', errorbar='sd',
                    height=4, aspect=1.2, facet_kws={'sharey': False, 'sharex': True})
    g.fig.suptitle(title, y=1.03, fontsize=16)
    g.set_axis_labels("Affinity Cutoff (nM)", f"Mean {metric}")
    g.set(xscale="log")
    g.set_titles("{col_name}")
    g.tight_layout(rect=[0, 0, 1, 0.97])
    fig_path = os.path.join(report_figures_dir, filename)
    plt.savefig(fig_path, dpi=200); plt.close()
    caption = f"{title}. Lines show mean metric value over replicates, shaded areas are ±1 SD."
    return fig_path, caption

def main():
    parser = argparse.ArgumentParser(description="Aggregate results from affinity cutoff experiment.")
    parser.add_argument("--base_experiment_dir", default="experiment_workspace_cutoff_analysis_features/", help="Base directory of the cutoff experiment results.")
    parser.add_argument("--original_workspace", default="experiment_workspace_hyperparam_sweep/", help="Directory containing the original hyperparameter sweep runs with their configs.")
    parser.add_argument("--output_report_dir", default="final_report_cutoff_analysis_features/", help="Directory to save the final LaTeX report.")
    args = parser.parse_args()

    logging.info("--- STARTING AFFINITY CUTOFF ANALYSIS AGGREGATION ---")
    report_run_id = f"cutoff_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    report_output_abs_dir = os.path.join(args.output_report_dir, report_run_id)
    report_figures_abs_dir = os.path.join(report_output_abs_dir, "figures"); os.makedirs(report_figures_abs_dir, exist_ok=True)
    latex_content = [LATEX_DOCUMENT_PREAMBLE]

    df_agg = collect_cutoff_experiment_metrics(args.base_experiment_dir, args.original_workspace)
    if df_agg.empty: logging.error("Failed to collect any metrics. Aborting."); return
    df_agg.to_csv(os.path.join(report_output_abs_dir, "DEBUG_cutoff_metrics_aggregated.csv"), index=False)
    
    df_agg['Method_Repr'] = df_agg['DR_Method'] + " (" + df_agg['Representation'] + ")"

    # --- SECTION 1: Overall Average Performance (All Hyperparameters) ---
    latex_content.append(get_section_header_latex(1, "Overall Performance vs. Affinity Cutoff (Averaged Over All Hyperparameters)"))
    latex_content.append("The following plots visualize the impact of the affinity cutoff, averaged across all tested hyperparameter configurations for each method.")
    
    for metric in ['ROC_AUC', 'PR_AUC', 'EF_1Perc']:
        fig_path, caption = create_performance_vs_cutoff_plot(df_agg, metric,
            f"Overall Mean {metric} vs. Affinity Cutoff",
            f"perf_vs_cutoff_{metric.lower()}_overall_avg.png", report_figures_abs_dir)
        if fig_path:
            add_figure_to_latex(latex_content, fig_path, caption, f"fig-{metric.lower()}-vs-cutoff-overall", figure_width="\\textwidth")

    # --- START OF CORRECTED LOGIC ---
    # --- SECTION 2: Identify and Report Best Hyperparameters ---
    latex_content.append(f"\\clearpage\n{get_section_header_latex(1, 'Optimal Hyperparameter Selection')}")
    latex_content.append("The single best hyperparameter value for each method and strategy was determined by identifying the setting that produced the highest mean ROC-AUC at the 100,000 nM affinity cutoff baseline. These optimal settings are used for all subsequent trend analysis.")
    
    df_baseline = df_agg[df_agg['Affinity_Cutoff'] == 100000].copy()
    optimal_replicate_dfs = []
    df_best_hyperparams_list = []
    
    unique_experiments = df_baseline[['Method_Repr', 'Embedding_Strategy', 'Hyperparameter']].drop_duplicates().to_records(index=False)

    for method, strategy, hyperparam_name in unique_experiments:
        df_exp = df_baseline[(df_baseline['Method_Repr'] == method) & (df_baseline['Embedding_Strategy'] == strategy)].copy()
        if df_exp.empty: continue
        
        optimal_value = "N/A"
        if hyperparam_name != 'N/A':
            mean_perf_table = df_exp.groupby('Hyperparameter_Value')['ROC_AUC'].mean().reset_index()
            if not mean_perf_table.empty:
                optimal_value = mean_perf_table.loc[mean_perf_table['ROC_AUC'].idxmax()]['Hyperparameter_Value']
        
        df_best_hyperparams_list.append({'Method_Repr': method, 'Embedding_Strategy': strategy, 'Hyperparameter': hyperparam_name, 'Optimal_Value': optimal_value})
        
        # Select all rows from the FULL dataset that match this best configuration
        df_optimal_replicates = df_agg[
            (df_agg['Method_Repr'] == method) &
            (df_agg['Embedding_Strategy'] == strategy) &
            (df_agg['Hyperparameter_Value'] == optimal_value)
        ].copy()
        optimal_replicate_dfs.append(df_optimal_replicates)

    df_best_hyperparams_table = pd.DataFrame(df_best_hyperparams_list)
    add_dataframe_as_latex_table(latex_content, df_best_hyperparams_table, "Selected Optimal Hyperparameters (by Mean ROC-AUC at 100,000 nM cutoff)", "tab-best-hyperparams", font_size=r"\normalsize")
    
    # --- SECTION 3: Performance Trends for Optimal Hyperparameters ---
    latex_content.append(f"\\clearpage\n{get_section_header_latex(1, 'Performance vs. Affinity Cutoff (Optimal Hyperparameters Only)')}")
    latex_content.append("The following plots visualize the impact of the affinity cutoff on only the peak performance configuration of each method, as determined in the previous section.")
    
    if optimal_replicate_dfs:
        df_filtered_for_trends = pd.concat(optimal_replicate_dfs, ignore_index=True)
        logging.info(f"Filtered dataset to {len(df_filtered_for_trends)} rows corresponding to optimal hyperparameters.")
        
        for metric in ['ROC_AUC', 'PR_AUC', 'EF_1Perc']:
            fig_path, caption = create_performance_vs_cutoff_plot(df_filtered_for_trends, metric,
                f"Optimal {metric} Performance vs. Affinity Cutoff",
                f"perf_vs_cutoff_{metric.lower()}_best_params.png", report_figures_abs_dir)
            if fig_path:
                add_figure_to_latex(latex_content, fig_path, caption, f"fig-{metric.lower()}-vs-cutoff-optimal", figure_width="\\textwidth")
    else:
        latex_content.append("Could not generate optimal performance plots because best hyperparameters were not identified.")
    # --- END OF CORRECTED LOGIC ---

    # --- Final LaTeX Generation ---
    latex_content.append(LATEX_DOCUMENT_END)
    report_tex_filename = f"cutoff_analysis_report.tex"
    report_tex_path = os.path.join(report_output_abs_dir, report_tex_filename)
    with open(report_tex_path, "w", encoding='utf-8') as f: f.write("\n".join(latex_content))
    logging.info(f"Cutoff analysis LaTeX report generated: {report_tex_path}")

    # (Compilation logic is unchanged)
    try:
        for i in range(2):
            subprocess.run(["pdflatex", "-interaction=nonstopmode", "-output-directory", report_output_abs_dir, report_tex_path], capture_output=True, text=True, check=False)
        pdf_path = report_tex_path.replace('.tex', '.pdf')
        if os.path.exists(pdf_path): logging.info(f"PDF report successfully generated: {pdf_path}")
        else: logging.warning(f"PDF report not found after compilation attempts.")
    except Exception as e:
        logging.error(f"LaTeX compilation error: {e}", exc_info=True)

if __name__ == "__main__":
    main()