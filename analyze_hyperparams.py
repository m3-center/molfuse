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

# --- LaTeX Preamble and Helper Functions (Unchanged) ---
# (The LATEX_DOCUMENT_PREAMBLE, LATEX_DOCUMENT_END, and helper functions
# like escape_latex_text_content, clean_for_label, etc., remain the same)
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
    pdftitle={UMMBAS Hyperparameter Analysis Report}, pdfauthor={UMMBAS Project Team},
    bookmarksnumbered=true, pdfpagemode=UseOutlines
}
\renewcommand{\cftsecleader}{\cftdotfill{\cftdotsep}} 
\pagestyle{fancy} \fancyhf{} \fancyhead[L]{UMMBAS Hyperparameter Report} \fancyhead[R]{\today} \fancyfoot[C]{\thepage\ of \pageref{LastPage}}
\title{UMMBAS Molecular Similarity Hyperparameter Evaluation Report\\ \large \textit{Report Generated: \texttt{<<RUN_ID_PLACEHOLDER>>}}}
\author{UMMBAS Project Team} \date{\today}
\begin{document} \maketitle \begin{abstract}
This report presents a comprehensive analysis of hyperparameter tuning experiments for the UMMBAS similarity space generation pipeline. The study focuses on systematically evaluating the impact of key hyperparameters for PCA, UMAP (\texttt{n\_neighbors}), and t-SNE (\texttt{perplexity} and \texttt{tsne\_pca\_components}) across different molecular representations (physicochemical features and ECFP4 fingerprints). Using a leave-one-target-out methodology, we assess virtual screening performance based on ROC-AUC, PR-AUC, and Enrichment Factors. The findings are used to identify optimal settings for these algorithms within the context of MF-guided chemical space exploration.
\end{abstract} \clearpage \tableofcontents \clearpage \listoffigures \clearpage \listoftables \clearpage
"""
LATEX_DOCUMENT_END = r"\end{document}"

def escape_latex_text_content(text_input):
    if not isinstance(text_input, str): text_input = str(text_input)
    conv = {'&': r'\&', '%': r'\%', '$': r'\$', '#': r'\"#', '_': r'\_',
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
    sec_cmd_map = {1: r"\section", 2: r"\subsection", 3: r"\subsubsection"}
    sec_cmd = sec_cmd_map.get(level, r"\subsubsection")
    return f"\n{sec_cmd}{{{escape_latex_text_content(title_text)}}}\n"

def add_figure_to_latex_standalone(latex_content_list, relative_fig_path_in_tex, caption_text, label_text, placement="[H]", figure_width="0.8\\textwidth"):
    clean_label = clean_for_label(label_text)
    figure_path_for_latex = relative_fig_path_in_tex.replace(os.sep, '/')
    latex_content_list.extend([f"\\begin{{figure}}{placement}", r"  \centering",
                               f"  \\includegraphics[width={figure_width}]{{{figure_path_for_latex}}}",
                               f"  \\caption{{{escape_latex_text_content(caption_text)}}}",
                               f"  \\label{{fig:{clean_label}}}", r"\end{figure}", "\n"])

def add_dataframe_as_latex_table_standalone(latex_content_list, dataframe, caption_text, label_text, placement="[H]", font_size=r"\small"):
    # (This function can remain largely the same, but ensure it handles numeric formatting well)
    clean_label = clean_for_label(label_text)
    if dataframe is None or dataframe.empty:
        latex_content_list.append(f"% Table '{escape_latex_text_content(label_text)}' is empty.\n")
        return
    latex_content_list.extend([f"\\begin{{table}}{placement}", r"  \centering", font_size,
                               f"  \\caption{{{escape_latex_text_content(caption_text)}}}",
                               f"  \\label{{tab:{clean_label}}}"])
    df_latex = dataframe.copy()
    for col in df_latex.select_dtypes(include=np.number).columns:
        df_latex[col] = df_latex[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
    df_latex.columns = [escape_latex_text_content(c.replace('_', ' ').title()) for c in df_latex.columns]
    col_format = 'l' * len(df_latex.columns)
    latex_table_string = df_latex.to_latex(index=False, escape=False, column_format=col_format, booktabs=True)
    latex_content_list.append(latex_table_string)
    latex_content_list.extend([r"\end{table}", "\n"])

# --- Logging and Data Collection ---
agg_log_file_name = f"hyperparam_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)-8s - %(filename)-25s - %(funcName)-25s - %(lineno)-4d - %(message)s',
                    handlers=[ logging.FileHandler(agg_log_file_name, mode='w'), logging.StreamHandler() ])

def find_all_replicate_runs(base_experiment_dir):
    # (This function remains unchanged)
    pattern = os.path.join(base_experiment_dir, "run_seed*_repr*_*_*")
    replicate_dirs = [d for d in glob.glob(pattern) if os.path.isdir(d)]
    logging.info(f"Found {len(replicate_dirs)} replicate run directories.")
    return sorted(replicate_dirs)

def collect_metrics_from_replicates(replicate_run_dirs):
    """
    Collects metrics from all replicate runs and intelligently extracts the
    hyperparameter that was swept for that specific run.
    """
    all_metrics_data = []
    for rep_dir_path in replicate_run_dirs:
        logging.info(f"Processing replicate: {os.path.basename(rep_dir_path)}")
        
        run_config_path = os.path.join(rep_dir_path, "run_config.json")
        if not os.path.exists(run_config_path):
            logging.warning(f"  run_config.json not found in {rep_dir_path}. Skipping.")
            continue
        with open(run_config_path, 'r') as f:
            run_config = json.load(f)

        gs = run_config['global_settings']
        # There should only be one representation and one DR method in each sweep config
        repr_type = run_config['representations'][0]
        dr_method_key = list(run_config['dimensionality_reduction_methods'].keys())[0]
        dr_params = run_config['dimensionality_reduction_methods'][dr_method_key]
        dr_short_name = dr_params['short_name']
        
        # --- Determine which hyperparameter was swept in this run ---
        hyperparam_name, hyperparam_value = "N/A", "N/A"
        if 'n_neighbors' in dr_params:
            hyperparam_name = 'n_neighbors'
            hyperparam_value = dr_params['n_neighbors']
        elif 'perplexity' in dr_params:
            hyperparam_name = 'perplexity'
            hyperparam_value = dr_params['perplexity']
        elif 'tsne_pca_components' in gs:
            hyperparam_name = 'tsne_pca_components'
            hyperparam_value = gs['tsne_pca_components']
        # --- End ---

        target_info = run_config['targets'][0]
        target_id_name = target_info['id_name']
        
        metrics_pattern = os.path.join(rep_dir_path, target_id_name, "results", repr_type, "dim_*", "*", "*_ranking_metrics.csv")
        metrics_files = glob.glob(metrics_pattern)

        for metrics_file_path in metrics_files:
            try:
                df_m = pd.read_csv(metrics_file_path)
                if df_m.empty: continue
                
                metric_row = {
                    'Representation': repr_type.capitalize(),
                    'DR_Method': dr_short_name,
                    'Hyperparameter_Name': hyperparam_name,
                    'Hyperparameter_Value': hyperparam_value
                }
                
                column_map = {'roc_auc': 'ROC_AUC', 'pr_auc': 'PR_AUC', 'ef_1%': 'EF_1Perc'}
                for csv_col, df_col in column_map.items():
                    metric_row[df_col] = df_m[csv_col].iloc[0] if csv_col in df_m else np.nan
                
                all_metrics_data.append(metric_row)
            except Exception as e:
                logging.error(f"Error reading metrics from {metrics_file_path}: {e}")

    if not all_metrics_data:
        logging.error("No metric data was collected.")
        return pd.DataFrame()
        
    return pd.DataFrame(all_metrics_data)

# --- Main Report Generation ---
def main_report_generation():
    parser = argparse.ArgumentParser(description="Analyze hyperparameter sweep results and generate a LaTeX report.")
    parser.add_argument("--base_experiment_dir", required=True, help="Base directory containing all hyperparameter sweep run folders.")
    parser.add_argument("--output_report_dir", required=True, help="Directory to save the final LaTeX report.")
    args = parser.parse_args()

    logging.info(f"--- STARTING HYPERPARAMETER ANALYSIS REPORT GENERATION ---")

    report_run_id = f"hyperparam_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    report_output_abs_dir = os.path.join(args.output_report_dir, report_run_id)
    report_figures_abs_dir = os.path.join(report_output_abs_dir, "figures"); os.makedirs(report_figures_abs_dir, exist_ok=True)
    latex_content = [LATEX_DOCUMENT_PREAMBLE.replace("<<RUN_ID_PLACEHOLDER>>", escape_latex_text_content(report_run_id))]
    
    replicate_dirs = find_all_replicate_runs(args.base_experiment_dir)
    if not replicate_dirs:
        logging.error("No replicate directories found. Aborting."); return
    
    df_agg = collect_metrics_from_replicates(replicate_dirs)
    if df_agg.empty:
        logging.error("Aggregated metrics DataFrame is empty. Aborting."); return
    df_agg.to_csv(os.path.join(report_output_abs_dir, "DEBUG_hyperparam_metrics_aggregated.csv"), index=False)
    
    # --- DYNAMICALLY GENERATE REPORT SECTIONS FOR EACH SWEEP ---
    latex_content.append(get_section_header_latex_standalone(1, 'Hyperparameter Sweep Analysis'))
    
    # Find all unique sweeps that were performed
    unique_sweeps = df_agg[['Representation', 'DR_Method', 'Hyperparameter_Name']].drop_duplicates().to_records(index=False)

    for repr_type, dr_method, hyperparam_name in unique_sweeps:
        if hyperparam_name == 'N/A': continue # Skip non-swept methods like PCA

        latex_content.append(f"\\clearpage\n{get_section_header_latex_standalone(2, f'Analysis for {dr_method} ({repr_type})')}")
        latex_content.append(get_section_header_latex_standalone(3, f'Sweeping: {hyperparam_name}'))

        # Filter data for the current sweep
        df_sweep = df_agg[
            (df_agg['Representation'] == repr_type) &
            (df_agg['DR_Method'] == dr_method) &
            (df_agg['Hyperparameter_Name'] == hyperparam_name)
        ].copy()

        if df_sweep.empty: continue

        # --- Generate Plots for this Sweep ---
        for metric_col in ['ROC_AUC', 'EF_1Perc']:
            plt.figure(figsize=(8, 6))
            summary = df_sweep.groupby('Hyperparameter_Value')[metric_col].agg(['mean', 'std']).reset_index().fillna(0)
            
            plt.plot(summary['Hyperparameter_Value'], summary['mean'], marker='o', linestyle='-')
            plt.fill_between(summary['Hyperparameter_Value'], summary['mean'] - summary['std'], summary['mean'] + summary['std'], alpha=0.2, label=f"$\\pm$1 SD")
            
            title = f'{metric_col} vs. {hyperparam_name} for {dr_method} ({repr_type})'
            plt.xlabel(hyperparam_name)
            plt.ylabel(f"Mean {metric_col}")
            plt.title(title)
            plt.grid(True, linestyle=':')
            plt.legend()
            plt.tight_layout()

            fig_filename = f"fig_{repr_type}_{dr_method}_{hyperparam_name}_{metric_col}.png"
            fig_path = os.path.join(report_figures_abs_dir, fig_filename)
            plt.savefig(fig_path); plt.close()
            
            add_figure_to_latex_standalone(latex_content, os.path.join("figures", fig_filename), title, clean_for_label(fig_filename))

        # --- Generate Summary Table for this Sweep ---
        summary_table = df_sweep.groupby('Hyperparameter_Value')[['ROC_AUC', 'PR_AUC', 'EF_1Perc']].agg(['mean', 'std']).reset_index()
        summary_table.columns = ['_'.join(col).strip() for col in summary_table.columns.values]
        summary_table.rename(columns={'Hyperparameter_Value_': hyperparam_name}, inplace=True)

        caption = f"Performance metrics for {dr_method} ({repr_type}) across different values of {hyperparam_name}."
        add_dataframe_as_latex_table_standalone(latex_content, summary_table, caption, clean_for_label(f"table_{repr_type}_{dr_method}_{hyperparam_name}"))

    # --- Final Report Generation ---
    latex_content.append(LATEX_DOCUMENT_END)
    report_tex_filename = "hyperparameter_analysis_report.tex"
    report_tex_path = os.path.join(report_output_abs_dir, report_tex_filename)
    with open(report_tex_path, "w", encoding='utf-8') as f:
        f.write("\n".join(latex_content))
    logging.info(f"Hyperparameter analysis report structure generated: {report_tex_path}")

    # (LaTeX compilation logic remains the same)
    try:
        for i in range(2): 
            subprocess.run(["pdflatex", "-interaction=nonstopmode", "-output-directory", report_output_abs_dir, report_tex_path], capture_output=True, text=True, check=False)
        pdf_path = os.path.join(report_output_abs_dir, os.path.basename(report_tex_path).replace('.tex', '.pdf'))
        if os.path.exists(pdf_path): logging.info(f"PDF report successfully generated: {pdf_path}")
        else: logging.warning(f"PDF report not found after compilation attempts.")
    except Exception as e:
        logging.error(f"LaTeX compilation error: {e}", exc_info=True)


if __name__ == "__main__":
    main_report_generation()