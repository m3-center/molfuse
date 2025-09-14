import pandas as pd
import numpy as np
import os
import argparse
import logging
import json
import glob
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess
import re
from datetime import datetime

# --- LaTeX Preamble ---
LATEX_DOCUMENT_PREAMBLE = r"""
\documentclass[10pt,a4paper]{article}
\usepackage[utf8]{inputenc} \usepackage[T1]{fontenc} \usepackage[margin=0.75in]{geometry}
\usepackage{graphicx} \usepackage{float} \usepackage{amsmath} \usepackage{amsfonts} \usepackage{amssymb}
\usepackage{booktabs} \usepackage{longtable} \usepackage{caption} \usepackage{array} \usepackage{xcolor}
\usepackage{hyperref} \usepackage{fancyhdr} \usepackage{lastpage} \usepackage{tocloft} \usepackage{enumitem}

\hypersetup{ colorlinks=true, linkcolor=blue, filecolor=magenta, urlcolor=cyan }
\pagestyle{fancy} \fancyhf{} \fancyhead[L]{UMMBAS Generalization Report} \fancyhead[R]{\today} \fancyfoot[C]{\thepage\ of \pageref{LastPage}}
\title{UMMBAS Molecular Similarity Experiment\\ \large \textit{Hyperparameter Generalization and Cutoff Analysis Report}}
\author{UMMBAS Project Team} \date{\today}
\begin{document} \maketitle \begin{abstract}
This report evaluates the generalization of optimal hyperparameters across different protein targets and analyzes the impact of affinity cutoffs. Optimal hyperparameters for PCA, UMAP, and t-SNE were first determined on the ABL1 Kinase target. These fixed, optimal configurations were then used to generate similarity spaces for two new targets: PKM2 and IDH1. A subsequent affinity cutoff analysis (100, 1000, 10000, 100000 nM) was performed on all three targets. This report compares the performance trends (ROC-AUC, PR-AUC, EF@1\%) across the source target (ABL1) and the generalization targets (PKM2, IDH1) to assess the portability of the tuned models.
\end{abstract} \clearpage \tableofcontents \clearpage \listoffigures \clearpage \listoftables \clearpage
"""
LATEX_DOCUMENT_END = r"\end{document}"

# (All helper functions: escape_latex, clean_for_label, add_figure, add_table, etc. are correct and can be reused)
# ...

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

def add_dataframe_as_latex_table(latex_content_list, dataframe, caption, label, csv_save_path, font_size=r"\small"):
    if dataframe is None or dataframe.empty:
        latex_content_list.append(f"% Table '{escape_latex_text_content(label)}' is empty.\n")
        return
    
    df_to_save = dataframe.copy()
    for col in df_to_save.select_dtypes(include=np.number).columns:
        if df_to_save[col].dropna().apply(lambda x: x.is_integer()).all():
            df_to_save[col] = df_to_save[col].apply(lambda x: f"{int(x)}" if pd.notna(x) else "N/A")
        else:
            df_to_save[col] = df_to_save[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
    df_to_save.columns = [c.replace('_', ' ').title() for c in df_to_save.columns]
    
    try:
        csv_rel_path = os.path.join("tables", os.path.basename(csv_save_path))
        df_to_save.to_csv(csv_save_path, index=False)
        logging.info(f"Saved table data to: {csv_save_path}")
    except Exception as e:
        logging.error(f"Failed to save table data to {csv_save_path}: {e}")
        return

    latex_content_list.extend([
        r"\begin{table}[H]", r"  \centering", font_size,
        f"  \\caption{{{escape_latex_text_content(caption)}}}",
        f"  \\label{{tab:{clean_for_label(label)}}}",
        f"  \\pgfplotstabletypeset[mystyle]{{{csv_rel_path.replace(os.sep, '/')}}}",
        r"\end{table}", "\n"
    ])

    # --- Logging and Data Collection (Unchanged) ---
log_file_name = f"cutoff_analysis_aggregation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)-8s - %(filename)-25s - %(funcName)-25s - %(lineno)-4d - %(message)s',
                    handlers=[logging.FileHandler(log_file_name, mode='w'), logging.StreamHandler()])

def collect_cutoff_experiment_metrics(cutoff_results_dir, original_hyperparam_dir, main_config):
    all_metrics = []
    # --- FIX: Create a lookup map for target display names ---
    target_name_map = {t['id_name']: t.get('display_name', t['id_name']) for t in main_config.get('targets', [])}
    
    pattern = os.path.join(cutoff_results_dir, "run_seed*", "results_cutoff_*", "*", "results", "*", "dim_*", "*", "*_ranking_metrics.csv")
    for metrics_file_path in glob.glob(pattern):
        try:
            path_parts = metrics_file_path.split(os.sep)
            run_dir_name = next(p for p in path_parts if p.startswith("run_seed"))
            cutoff_dir_name = next(p for p in path_parts if p.startswith("results_cutoff_"))
            target_id_name = path_parts[path_parts.index(cutoff_dir_name) + 1]
            
            # --- FIX: Add Target_Name to the collected data ---
            target_display_name = target_name_map.get(target_id_name, target_id_name)
            
            # (The rest of the parsing is correct)
            repr_type = path_parts[path_parts.index("results") + 1]
            strategy_dir = path_parts[path_parts.index(next(p for p in path_parts if p.startswith("dim_"))) + 1]
            seed = int(re.search(r"run_seed(\d+)", run_dir_name).group(1))
            cutoff_val = int(re.search(r"results_cutoff_(\d+)", cutoff_dir_name).group(1))
            run_config_path = os.path.join(original_hyperparam_dir, run_dir_name, "run_config.json")
            if not os.path.exists(run_config_path): continue
            with open(run_config_path, 'r') as f: run_config = json.load(f)
            dr_method_key = list(run_config['dimensionality_reduction_methods'].keys())[0]
            dr_params = run_config['dimensionality_reduction_methods'][dr_method_key]
            embedding_strategy = "Projection"
            if "tsne" in dr_method_key: embedding_strategy = "Co-embedding (Native)"
            elif strategy_dir.endswith('_Coembed'): embedding_strategy = "Co-embedding"
            
            df_m = pd.read_csv(metrics_file_path)
            if df_m.empty: continue

            metric_row = {
                'Seed': seed, 'Affinity_Cutoff': cutoff_val,
                'Target_Name': target_display_name, # <-- ADDED
                'Representation': repr_type.capitalize(), 'DR_Method': dr_params['short_name'],
                'Embedding_Strategy': embedding_strategy
            }
            column_map = {'roc_auc': 'ROC_AUC', 'pr_auc': 'PR_AUC', 'ef_1%': 'EF_1Perc'}
            for csv_col, df_col in column_map.items():
                metric_row[df_col] = df_m[csv_col].iloc[0] if csv_col in df_m else np.nan
            all_metrics.append(metric_row)
        except Exception as e:
            logging.warning(f"Failed to process metrics file {metrics_file_path}: {e}")
    return pd.DataFrame(all_metrics)

def create_performance_vs_cutoff_plot(df, metric, title, filename, report_figures_dir, y_range=None):
    if df.empty or metric not in df.columns:
        logging.warning(f"Data is empty or missing '{metric}' for plot '{title}'."); return None, None
    g = sns.relplot(data=df, x='Affinity_Cutoff', y=metric, hue='Embedding_Strategy',
                    col='Method_Repr', col_wrap=3, kind='line', marker='o', errorbar='sd',
                    height=4, aspect=1.2, facet_kws={'sharey': False, 'sharex': True})
    g.fig.suptitle(title, y=1.0, fontsize=16)
    g.set_axis_labels("Affinity Cutoff (nM)", f"Mean {metric}")
    g.set(xscale="log")
    if y_range is not None: g.set(ylim=y_range)
    g.set_titles("{col_name}")
    g.fig.subplots_adjust(top=0.92)
    fig_path = os.path.join(report_figures_dir, filename)
    plt.savefig(fig_path, dpi=300); plt.close()
    caption = f"{title}. Lines show mean metric value over replicates, shaded areas are ±1 SD."
    return fig_path, caption

def main():
    parser = argparse.ArgumentParser(description="Aggregate and analyze hyperparameter generalization results.")
    parser.add_argument("--base_experiment_dir", default="generalization_analysis/", help="Base directory of the generalization cutoff experiment results.")
    parser.add_argument("--original_workspace", default="experiment_workspace_generalization/", help="Directory containing the original generalization runs with their configs.")
    parser.add_argument("--main_config_path", default="experiment_config.json", help="Path to the main experiment config file.")
    parser.add_argument("--output_report_dir", default="final_report_generalization/", help="Directory to save the final LaTeX report.")
    args = parser.parse_args()

    logging.info("--- STARTING HYPERPARAMETER GENERALIZATION ANALYSIS ---")
    try:
        with open(args.main_config_path, 'r') as f: main_config = json.load(f)
    except Exception as e:
        logging.error(f"Could not load main config file '{args.main_config_path}'. Aborting. Error: {e}")
        return
        
    report_run_id = f"generalization_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    report_output_abs_dir = os.path.join(args.output_report_dir, report_run_id)
    report_figures_abs_dir = os.path.join(report_output_abs_dir, "figures"); os.makedirs(report_figures_abs_dir, exist_ok=True)
    report_tables_abs_dir = os.path.join(report_output_abs_dir, "tables"); os.makedirs(report_tables_abs_dir, exist_ok=True)
    latex_content = [LATEX_DOCUMENT_PREAMBLE]

    df_agg = collect_cutoff_experiment_metrics(args.base_experiment_dir, args.original_workspace, main_config)
    if df_agg.empty:
        logging.error("Failed to collect any metrics. Aborting."); return
    df_agg.to_csv(os.path.join(report_tables_abs_dir, "master_generalization_metrics_aggregated.csv"), index=False)
    
    y_ranges = {}
    for metric in ['ROC_AUC', 'PR_AUC', 'EF_1Perc']:
        if metric in df_agg.columns and not df_agg[metric].isnull().all():
            min_val = df_agg[metric].min()
            max_val = df_agg[metric].max()
            if pd.isna(min_val) or pd.isna(max_val): continue
            padding = (max_val - min_val) * 0.1
            final_min = max(0.01, min_val - padding) if metric == 'EF_1Perc' else min_val - padding
            y_ranges[metric] = (final_min, max_val + padding)
            logging.info(f"Calculated Y-axis range for {metric}: {y_ranges[metric]}")
    
    # --- ANALYSIS SECTION ---
    latex_content.append(get_section_header_latex(1, "Analysis of Hyperparameter Generalization"))
    latex_content.append("The following plots compare the performance trends of models using hyperparameters optimized on ABL1 Kinase when applied to the original target and two new targets (PKM2, IDH1). Successful generalization is indicated if the performance curves for the new targets are similar to the ABL1 curve.")

    # Get a list of all unique Method + Strategy combinations
    unique_combinations = df_agg[['DR_Method', 'Embedding_Strategy']].drop_duplicates().to_records(index=False)

    for metric in ['ROC_AUC', 'PR_AUC', 'EF_1Perc']:
        latex_content.append(f"\\clearpage\n{get_section_header_latex(2, f'Generalization Performance by {metric}')}")
        
        for dr_method, strategy in unique_combinations:
            df_plot = df_agg[(df_agg['DR_Method'] == dr_method) & (df_agg['Embedding_Strategy'] == strategy)]
            if df_plot.empty: continue

            plt.figure(figsize=(8, 6))
            sns.lineplot(data=df_plot, x='Affinity_Cutoff', y=metric, hue='Target_Name', marker='o', errorbar='sd')
            
            title = f"{metric} vs. Cutoff for {dr_method} ({strategy})"
            plt.title(title)
            plt.xlabel("Affinity Cutoff (nM)")
            plt.ylabel(f"Mean {metric}")
            plt.xscale('log')
            if "EF" in metric: plt.yscale('log')
            
            if metric in y_ranges:
                plt.ylim(y_ranges[metric])
                
            plt.grid(True, linestyle='--')
            plt.legend(title="Target")
            plt.tight_layout()

            fig_filename = f"fig_generalization_{dr_method}_{strategy}_{metric}.pdf"
            fig_path = os.path.join(report_figures_abs_dir, fig_filename)
            plt.savefig(fig_path, dpi=300); plt.close()
            
            caption = f"Performance trend for {dr_method} ({strategy}) across three targets. The hyperparameters used were optimized on ABL1 Kinase."
            add_figure_to_latex(latex_content, fig_path, caption, f"fig-gen-{dr_method}-{strategy}-{metric}")

    # --- FINAL SUMMARY TABLE ---
    latex_content.append(f"\\clearpage\n{get_section_header_latex(1, 'Peak Performance Summary by Target')}")
    latex_content.append("This table summarizes the peak performance achieved for each method, strategy, and target after optimizing for the best affinity cutoff.")
    
    peak_perf_rows = []
    for (method, strategy, target), group_df in df_agg.groupby(['DR_Method', 'Embedding_Strategy', 'Target_Name']):
        peak_row = {'DR_Method': method, 'Embedding_Strategy': strategy, 'Target': target}
        for metric in ['ROC_AUC', 'PR_AUC', 'EF_1Perc']:
            if metric in group_df.columns:
                mean_perf_by_cutoff = group_df.groupby('Affinity_Cutoff')[metric].mean()
                if not mean_perf_by_cutoff.empty:
                    optimal_cutoff = mean_perf_by_cutoff.idxmax()
                    peak_performance = mean_perf_by_cutoff.max()
                    peak_row[f'Optimal_Cutoff_{metric}'] = optimal_cutoff
                    peak_row[f'Peak_Mean_{metric}'] = peak_performance
        peak_perf_rows.append(peak_row)

    if peak_perf_rows:
        df_peak_summary = pd.DataFrame(peak_perf_rows)
        table_label = "tab-peak-performance-generalization"
        csv_path = os.path.join(report_tables_abs_dir, f"{clean_for_label(table_label)}.csv")
        add_dataframe_as_latex_table(latex_content, df_peak_summary,
            "Peak performance after optimizing for affinity cutoff on each target.",
            table_label, csv_path, font_size=r"\scriptsize")

    # --- Final LaTeX Generation ---
    latex_content.append(LATEX_DOCUMENT_END)
    report_tex_filename = f"cutoff_analysis_report.tex"
    report_tex_path = os.path.join(report_output_abs_dir, report_tex_filename)
    with open(report_tex_path, "w", encoding='utf-8') as f: f.write("\n".join(latex_content))
    logging.info(f"Generalization/cutoff analysis LaTeX report generated: {report_tex_path}")

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
