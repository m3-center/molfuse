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
    bookmarksnumbered=true, pdfdemode=UseOutlines
}
\renewcommand{\cftsecleader}{\cftdotfill{\cftdotsep}} 
\pagestyle{fancy} \fancyhf{} \fancyhead[L]{UMMBAS Hyperparameter Report} \fancyhead[R]{\today} \fancyfoot[C]{\thepage\ of \pageref{LastPage}}
\title{UMMBAS Molecular Similarity Hyperparameter Evaluation Report\\ \large \textit{Report Generated: \texttt{<<RUN_ID_PLACEHOLDER>>}}}
\author{UMMBAS Project Team} \date{\today}
\begin{document} \maketitle \begin{abstract}
This report presents a comprehensive analysis of hyperparameter tuning experiments for the UMMBAS similarity space generation pipeline. The study focuses on systematically evaluating the impact of key hyperparameters for PCA, UMAP (\texttt{n\_neighbors}), and t-SNE (\texttt{perplexity}) across different molecular representations (physicochemical features and ECFP4 fingerprints) and embedding strategies (Projection vs. Co-embedding). Using a leave-one-target-out methodology, we assess virtual screening performance based on ROC-AUC, PR-AUC, and Enrichment Factors. The findings are used to identify optimal settings for these algorithms within the context of MF-guided chemical space exploration.
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

def add_figure_to_latex_standalone(latex_content_list, relative_fig_path_in_tex, caption_text, label_text, placement="[H]", figure_width="0.9\\textwidth"):
    clean_label = clean_for_label(label_text)
    figure_path_for_latex = relative_fig_path_in_tex.replace(os.sep, '/')
    latex_content_list.extend([f"\\begin{{figure}}{placement}", r"  \centering",
                               f"  \\includegraphics[width={figure_width}]{{{figure_path_for_latex}}}",
                               f"  \\caption{{{escape_latex_text_content(caption_text)}}}",
                               f"  \\label{{fig:{clean_label}}}", r"\end{figure}", "\n"])

def add_dataframe_as_latex_table_standalone(latex_content_list, dataframe, caption_text, label_text, placement="[H]", font_size=r"\small"):
    clean_label = clean_for_label(label_text)
    if dataframe is None or dataframe.empty:
        latex_content_list.append(f"% Table '{escape_latex_text_content(label_text)}' is empty.\n")
        return
    latex_content_list.extend([f"\\begin{{table}}{placement}", r"  \centering", font_size,
                               f"  \\caption{{{escape_latex_text_content(caption_text)}}}",
                               f"  \\label{{tab:{clean_label}}}"])
    df_latex = dataframe.copy()
    for col in df_latex.select_dtypes(include=np.number).columns:
        if df_latex[col].dropna().apply(lambda x: x.is_integer()).all():
            df_latex[col] = df_latex[col].apply(lambda x: f"{int(x)}" if pd.notna(x) else "N/A")
        else:
            df_latex[col] = df_latex[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
    df_latex.columns = [escape_latex_text_content(c.replace('_', ' ').title()) for c in df_latex.columns]
    col_format = 'l' * len(df_latex.columns)
    latex_table_string = df_latex.to_latex(index=False, escape=False, column_format=col_format)
    latex_content_list.append(latex_table_string)
    latex_content_list.extend([r"\end{table}", "\n"])

# --- Logging and Data Collection ---
agg_log_file_name = f"hyperparam_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)-8s - %(filename)-25s - %(funcName)-25s - %(lineno)-4d - %(message)s',
                    handlers=[ logging.FileHandler(agg_log_file_name, mode='w'), logging.StreamHandler() ])

def find_all_replicate_runs(base_experiment_dir):
    pattern = os.path.join(base_experiment_dir, "run_seed*_repr*_*_*")
    replicate_dirs = [d for d in glob.glob(pattern) if os.path.isdir(d)]
    logging.info(f"Found {len(replicate_dirs)} replicate run directories.")
    return sorted(replicate_dirs)

def collect_metrics_from_replicates(replicate_run_dirs):
    all_metrics_data = []
    for rep_dir_path in replicate_run_dirs:
        run_config_path = os.path.join(rep_dir_path, "run_config.json")
        if not os.path.exists(run_config_path): continue
        try:
            with open(run_config_path, 'r') as f: run_config = json.load(f)
        except json.JSONDecodeError as e:
            logging.error(f"  CORRUPTED JSON in {run_config_path}. Skipping. Error: {e}")
            continue
        repr_type = run_config['representations'][0]
        dr_method_key = list(run_config['dimensionality_reduction_methods'].keys())[0]
        dr_params = run_config['dimensionality_reduction_methods'][dr_method_key]
        dr_short_name = dr_params['short_name']
        hyperparam_name, hyperparam_value = "N/A", "N/A"
        if 'n_neighbors' in dr_params:
            hyperparam_name = 'n_neighbors'; hyperparam_value = dr_params['n_neighbors']
        elif 'perplexity' in dr_params:
            hyperparam_name = 'perplexity'; hyperparam_value = dr_params['perplexity']
        target_id_name = run_config['targets'][0]['id_name']
        metrics_pattern = os.path.join(rep_dir_path, target_id_name, "results", repr_type, "dim_*", "*", "*_ranking_metrics.csv")
        for metrics_file_path in glob.glob(metrics_pattern):
            try:
                strategy_dir_name = os.path.basename(os.path.dirname(metrics_file_path))
                embedding_strategy = "Projection"
                if "tsne" in dr_method_key:
                    embedding_strategy = "Co-embedding (Native)"
                elif "-Coembed" in strategy_dir_name:
                    embedding_strategy = "Co-embedding"
                df_m = pd.read_csv(metrics_file_path)
                if df_m.empty: continue
                metric_row = {'Representation': repr_type.capitalize(), 'DR_Method': dr_short_name,
                              'Embedding_Strategy': embedding_strategy,
                              'Hyperparameter': hyperparam_name, 'Hyperparameter_Value': hyperparam_value}
                column_map = {'roc_auc': 'ROC_AUC', 'pr_auc': 'PR_AUC', 'ef_1%': 'EF_1Perc'}
                for csv_col, df_col in column_map.items():
                    metric_row[df_col] = df_m[csv_col].iloc[0] if csv_col in df_m else np.nan
                all_metrics_data.append(metric_row)
            except Exception as e:
                logging.error(f"Error reading metrics from {metrics_file_path}: {e}")
    if not all_metrics_data:
        logging.error("No metric data was collected."); return pd.DataFrame()
    return pd.DataFrame(all_metrics_data)

def main_report_generation():
    parser = argparse.ArgumentParser(description="Analyze hyperparameter sweep results.")
    parser.add_argument("--base_experiment_dir", required=True)
    parser.add_argument("--output_report_dir", required=True)
    args = parser.parse_args()

    logging.info(f"--- STARTING HYPERPARAMETER ANALYSIS REPORT GENERATION ---")
    report_run_id = f"hyperparam_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    report_output_abs_dir = os.path.join(args.output_report_dir, report_run_id)
    report_figures_abs_dir = os.path.join(report_output_abs_dir, "figures"); os.makedirs(report_figures_abs_dir, exist_ok=True)
    latex_content = [LATEX_DOCUMENT_PREAMBLE.replace("<<RUN_ID_PLACEHOLDER>>", escape_latex_text_content(report_run_id))]
    
    replicate_dirs = find_all_replicate_runs(args.base_experiment_dir)
    if not replicate_dirs: logging.error("No replicate directories found."); return
    df_agg = collect_metrics_from_replicates(replicate_dirs)
    if df_agg.empty: logging.error("Aggregated metrics DataFrame is empty."); return
    df_agg.to_csv(os.path.join(report_output_abs_dir, "DEBUG_hyperparam_metrics_aggregated.csv"), index=False)
    
    df_agg['Method'] = df_agg['DR_Method'] + ' (' + df_agg['Representation'] + ')'

    latex_content.append(get_section_header_latex_standalone(1, 'Detailed Hyperparameter Sweep Analysis'))
    
    all_best_configs = []
    unique_experiments = df_agg[['Method', 'Embedding_Strategy']].drop_duplicates().to_records(index=False)

    for method, strategy in unique_experiments:
        df_exp = df_agg[(df_agg['Method'] == method) & (df_agg['Embedding_Strategy'] == strategy)].copy()
        if df_exp.empty: continue
        hyperparam_name = df_exp['Hyperparameter'].iloc[0] # Using the corrected, consistent name
        
        latex_content.append(f"\\clearpage\n{get_section_header_latex_standalone(2, f'Analysis for {method} - Strategy: {strategy}')}")
        
        if hyperparam_name != 'N/A':
            latex_content.append(get_section_header_latex_standalone(3, f'Sweeping: {hyperparam_name}'))
            for metric_col in ['ROC_AUC', 'EF_1Perc']:
                plt.figure(figsize=(8, 6))
                summary = df_exp.groupby('Hyperparameter_Value')[metric_col].agg(['mean', 'std']).reset_index().fillna(0).sort_values(by='Hyperparameter_Value')
                plt.plot(summary['Hyperparameter_Value'], summary['mean'], marker='o', linestyle='-')
                plt.fill_between(summary['Hyperparameter_Value'], summary['mean'] - summary['std'], summary['mean'] + summary['std'], alpha=0.2, label=f"$\\pm$1 SD")
                title = f'{metric_col} vs. {hyperparam_name} for {method} ({strategy})'
                plt.xlabel(hyperparam_name); plt.ylabel(f"Mean {metric_col}")
                plt.title(title); plt.grid(True, linestyle=':'); plt.legend(); plt.tight_layout()
                fig_filename = f"fig_{clean_for_label(method)}_{clean_for_label(strategy)}_{hyperparam_name}_{metric_col}.png"
                fig_path = os.path.join(report_figures_abs_dir, fig_filename)
                plt.savefig(fig_path); plt.close()
                add_figure_to_latex_standalone(latex_content, os.path.join("figures", fig_filename), title, clean_for_label(fig_filename))

        summary_table_mean = df_exp.groupby('Hyperparameter_Value')[['ROC_AUC', 'PR_AUC', 'EF_1Perc']].agg(['mean', 'std']).reset_index()
        summary_table_mean.columns = ['_'.join(col).strip() for col in summary_table_mean.columns.values]
        summary_table_mean.rename(columns={'Hyperparameter_Value_': hyperparam_name}, inplace=True)
        add_dataframe_as_latex_table_standalone(latex_content, summary_table_mean, f"Performance metrics for {method} ({strategy}).", clean_for_label(f"table_{method}_{strategy}"))

        summary_table_median = df_exp.groupby('Hyperparameter_Value')[['ROC_AUC', 'PR_AUC', 'EF_1Perc']].agg('median').reset_index()
        if 'ROC_AUC' in summary_table_median.columns:
            best_row = summary_table_median.loc[summary_table_median['ROC_AUC'].idxmax()]
            optimal_value = 'N/A'
            if hyperparam_name != 'N/A':
                optimal_value = best_row[hyperparam_name]
            best_config_details = {
                'Method': method,
                'Embedding_Strategy': strategy,
                'Hyperparameter': hyperparam_name,
                'Optimal_Value': optimal_value, # Use the correctly determined value
                'ROC_AUC': best_row['ROC_AUC'],
                'PR_AUC': best_row['PR_AUC'],
                'EF_1Perc': best_row['EF_1Perc']
            }
            all_best_configs.append(best_config_details)

    latex_content.append(f"\\clearpage\n{get_section_header_latex_standalone(1, 'Overall Performance Summary')}")
    latex_content.append("This section synthesizes the results to compare the peak performance of each method and strategy, using the median as the central tendency and the 5th-95th percentiles as a 90% confidence interval.")

    if all_best_configs:
        best_configs_df = pd.DataFrame(all_best_configs)
        
        performance_data_at_best_params = pd.merge(df_agg, best_configs_df, on=['Method', 'Embedding_Strategy', 'Hyperparameter'], how='inner')
        performance_data_at_best_params = performance_data_at_best_params[performance_data_at_best_params['Hyperparameter_Value'] == performance_data_at_best_params['Optimal_Value']]

        p05 = lambda x: x.quantile(0.05)
        p95 = lambda x: x.quantile(0.95)
        agg_funcs = ['median', p05, p95]
        
        final_summary = performance_data_at_best_params.groupby(['Method', 'Embedding_Strategy'])[['ROC_AUC', 'PR_AUC', 'EF_1Perc']].agg(agg_funcs)
        final_summary.columns = ['_'.join(col).strip() for col in final_summary.columns.values]
        final_summary = final_summary.reset_index()
        
        for metric in ['ROC_AUC', 'PR_AUC', 'EF_1Perc']:
            latex_content.append(f"\\clearpage\n{get_section_header_latex_standalone(2, f'Peak Performance by Median {metric}')}")
            
            median_col, p05_col, p95_col = f'{metric}_median', f'{metric}_p05', f'{metric}_p95'
            if not all(c in final_summary.columns for c in [median_col, p05_col, p95_col]): continue

            plot_data = final_summary.sort_values(by=median_col, ascending=False)
            
            plt.figure(figsize=(12, 8))
            ax = sns.barplot(data=plot_data, x=median_col, y='Method', hue='Embedding_Strategy', palette='viridis', dodge=True)
            
            # Create a map to get error values for each bar
            error_map = {}
            for i, row in plot_data.iterrows():
                key = (row['Method'], row['Embedding_Strategy'])
                lower_err = row[median_col] - row[p05_col]
                upper_err = row[p95_col] - row[median_col]
                error_map[key] = (lower_err, upper_err)
            
            # Re-map error bars to the barplot's drawn patches
            # Get hue order from the legend
            try:
                hue_order = [t.get_text() for t in ax.get_legend().get_texts()]
            except AttributeError: # Legend might not be created if only one hue
                hue_order = plot_data['Embedding_Strategy'].unique().tolist()
                
            num_hue = len(hue_order)
            y_ticks = ax.get_yticks()
            
            for i, method in enumerate(plot_data['Method'].unique()):
                for j, strategy in enumerate(hue_order):
                    key = (method, strategy)
                    if key in error_map:
                        bar_y_pos = y_ticks[i] + (j - (num_hue - 1) / 2.) * (ax.patches[0].get_height())
                        median_val = plot_data[(plot_data['Method']==method) & (plot_data['Embedding_Strategy']==strategy)][median_col].values[0]
                        lower, upper = error_map[key]
                        ax.errorbar(x=[median_val], y=[bar_y_pos], xerr=[[lower], [upper]], fmt='none', c='black', capsize=5)

            plt.xlabel(f'Median {metric} (90% CI)'); plt.ylabel('Method (Representation)')
            plt.title(f'Peak Performance After Tuning (by Median {metric})')
            if "AUC" in metric: plt.xlim(0, max(1.0, plot_data[median_col].max() * 1.1) if plot_data[median_col].max() > 0 else 1.0)
            if "EF" in metric: plt.xscale('log')
            plt.grid(True, axis='x', linestyle='--'); plt.legend(title='Embedding Strategy'); plt.tight_layout()

            fig_filename = f"fig_peak_performance_{metric}_median_split.png"
            fig_path = os.path.join(report_figures_abs_dir, fig_filename)
            plt.savefig(fig_path); plt.close()
            
            caption = f"Comparison of the best median {metric} for each strategy. Error bars are the 90% confidence interval."
            add_figure_to_latex_standalone(latex_content, os.path.join("figures", fig_filename), caption, f"fig_peak_summary_{metric}_median_split")
            
            table_data = plot_data[['Method', 'Embedding_Strategy', median_col, p05_col, p95_col]]
            add_dataframe_as_latex_table_standalone(latex_content, table_data, f"Optimal performance ranked by median {metric}.", f"table_summary_{metric}_median_split")

    latex_content.append(LATEX_DOCUMENT_END)
    report_tex_filename = "hyperparameter_analysis_report.tex"
    report_tex_path = os.path.join(report_output_abs_dir, report_tex_filename)
    with open(report_tex_path, "w", encoding='utf-8') as f:
        f.write("\n".join(latex_content))
    logging.info(f"Hyperparameter analysis report structure generated: {report_tex_path}")
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