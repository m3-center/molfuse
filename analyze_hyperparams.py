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
\usepackage{pgfplotstable} % <-- Required package for reading CSVs

% PGFPlotstable configuration for booktabs-style tables
\pgfplotsset{compat=1.17}
\pgfplotstableset{
    mystyle/.style={
        col sep=comma,
        string type, % Treat all columns as strings, as formatting is done in Python
        header=true,
        every head row/.style={
            before row=\toprule,
            after row=\midrule
        },
        every last row/.style={
            after row=\bottomrule
        },
        display columns/0/.style={string type, column type={l}},
        fixed,
        read comma as period,
    }
}

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
            '{': r'\{', '}': r'\}', '~': r'\textascitilde{}', '^': r'\^{}',
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

def add_dataframe_as_latex_table_standalone(latex_content_list, dataframe, caption_text, label_text, csv_save_path, placement="[H]", font_size=r"\small"):
    clean_label = clean_for_label(label_text)
    if dataframe is None or dataframe.empty:
        latex_content_list.append(f"% Table '{escape_latex_text_content(label_text)}' is empty.\n")
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
        f"\\begin{{table}}{placement}", r"  \centering", font_size,
        f"  \\caption{{{escape_latex_text_content(caption_text)}}}",
        f"  \\label{{tab:{clean_label}}}",
        f"  \\pgfplotstabletypeset[mystyle]{{{csv_rel_path.replace(os.sep, '/')}}}",
        r"\end{table}", "\n"
    ])

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
                elif strategy_dir_name.endswith('_Coembed'):
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
    parser = argparse.ArgumentParser(description="Analyze hyperparameter sweep results and generate a LaTeX report.")
    parser.add_argument("--base_experiment_dir", required=True, help="Base directory containing all hyperparameter sweep run folders.")
    parser.add_argument("--output_report_dir", required=True, help="Directory to save the final LaTeX report.")
    args = parser.parse_args()

    logging.info(f"--- STARTING HYPERPARAMETER ANALYSIS REPORT GENERATION ---")
    report_run_id = f"hyperparam_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    report_output_abs_dir = os.path.join(args.output_report_dir, report_run_id)
    report_figures_abs_dir = os.path.join(report_output_abs_dir, "figures"); os.makedirs(report_figures_abs_dir, exist_ok=True)
    report_tables_abs_dir = os.path.join(report_output_abs_dir, "tables"); os.makedirs(report_tables_abs_dir, exist_ok=True)
    
    latex_content = [LATEX_DOCUMENT_PREAMBLE.replace("<<RUN_ID_PLACEHOLDER>>", escape_latex_text_content(report_run_id))]
    
    replicate_dirs = find_all_replicate_runs(args.base_experiment_dir)
    if not replicate_dirs:
        logging.error("No replicate directories found. Aborting.")
        return
    
    df_agg = collect_metrics_from_replicates(replicate_dirs)
    if df_agg.empty:
        logging.error("Aggregated metrics DataFrame is empty. Aborting.")
        return
        
    df_agg['Method'] = df_agg['DR_Method'] + ' (' + df_agg['Representation'] + ')'
    df_agg.to_csv(os.path.join(report_tables_abs_dir, "master_aggregated_metrics.csv"), index=False)
    
    # --- Calculate global y-axis ranges for fair plot comparisons ---
    y_ranges = {}
    for metric in ['ROC_AUC', 'PR_AUC', 'EF_1Perc']:
        if metric in df_agg.columns and not df_agg[metric].isnull().all():
            grouped = df_agg.groupby(['Method', 'Embedding_Strategy', 'Hyperparameter_Value'])[metric]
            means = grouped.mean()
            stds = grouped.std().fillna(0)
            min_val = (means - stds).min()
            max_val = (means + stds).max()
            padding = (max_val - min_val) * 0.1 if not np.isnan(max_val) else 0.1
            y_ranges[metric] = (min_val - padding, max_val + padding)
    
    # --- SECTION 1: DETAILED SWEEP ANALYSIS ---
    latex_content.append(get_section_header_latex_standalone(1, 'Detailed Hyperparameter Sweep Analysis'))
    
    unique_experiments = df_agg[['Method', 'Embedding_Strategy', 'Hyperparameter']].drop_duplicates().to_records(index=False)

    for method, strategy, hyperparam_name in unique_experiments:
        df_exp = df_agg[(df_agg['Method'] == method) & (df_agg['Embedding_Strategy'] == strategy)].copy()
        if df_exp.empty: continue
        
        latex_content.append(f"\\clearpage\n{get_section_header_latex_standalone(2, f'Analysis for {method} - Strategy: {strategy}')}")
        
        if hyperparam_name != 'N/A':
            latex_content.append(get_section_header_latex_standalone(3, f'Sweeping: {hyperparam_name}'))
            for metric_col in ['ROC_AUC', 'PR_AUC', 'EF_1Perc']:
                if metric_col not in df_exp.columns or df_exp[metric_col].isnull().all(): continue
                plt.figure(figsize=(8, 6))
                summary = df_exp.groupby('Hyperparameter_Value')[metric_col].agg(['mean', 'std']).reset_index().fillna(0).sort_values(by='Hyperparameter_Value')
                plt.plot(summary['Hyperparameter_Value'], summary['mean'], marker='o', linestyle='-')
                plt.fill_between(summary['Hyperparameter_Value'], summary['mean'] - summary['std'], summary['mean'] + summary['std'], alpha=0.2, label=f"$\\pm$1 SD")
                title = f'{metric_col} vs. {hyperparam_name} for {method} ({strategy})'
                plt.xlabel(hyperparam_name); plt.ylabel(f"Mean {metric_col}")
                plt.title(title); plt.grid(True, linestyle=':'); plt.legend()
                if y_ranges.get(metric_col): plt.ylim(y_ranges[metric_col])
                plt.tight_layout()
                fig_filename = f"fig_{clean_for_label(method)}_{clean_for_label(strategy)}_{hyperparam_name}_{metric_col}.png"
                fig_path = os.path.join(report_figures_abs_dir, fig_filename)
                plt.savefig(fig_path); plt.close()
                add_figure_to_latex_standalone(latex_content, os.path.join("figures", fig_filename), title, clean_for_label(fig_filename))
        
        summary_table = df_exp.groupby('Hyperparameter_Value')[['ROC_AUC', 'PR_AUC', 'EF_1Perc']].agg(['mean']).reset_index()
        summary_table.columns = [col[0] if col[1] == '' else '_'.join(col) for col in summary_table.columns]
        summary_table.rename(columns={'Hyperparameter_Value': hyperparam_name, 'ROC_AUC_mean': 'ROC_AUC', 'PR_AUC_mean': 'PR_AUC', 'EF_1Perc_mean': 'EF_1Perc'}, inplace=True)
        
        table_label = f"table_{method}_{strategy}"
        csv_path = os.path.join(report_tables_abs_dir, f"{clean_for_label(table_label)}.csv")
        add_dataframe_as_latex_table_standalone(latex_content, summary_table, f"Performance metrics for {method} ({strategy}).", table_label, csv_path)

    # --- SECTION 2: OVERALL PERFORMANCE SUMMARY ---
    latex_content.append(f"\\clearpage\n{get_section_header_latex_standalone(1, 'Overall Performance Summary')}")
    latex_content.append("This section synthesizes the results to compare the peak performance of each method and strategy. The optimal hyperparameter for each combination is determined independently for each performance metric (ROC-AUC, PR-AUC, and EF@1\%). Statistics are then calculated from all replicate runs corresponding to that metric-specific optimal setting.")

    for metric in ['ROC_AUC', 'PR_AUC', 'EF_1Perc']:
        latex_content.append(f"\\clearpage\n{get_section_header_latex_standalone(2, f'Peak Performance by Median {metric}')}")
        
        optimal_replicate_dfs_for_metric = []
        logging.info(f"--- Identifying optimal hyperparameters based on MEAN {metric} ---")
        
        unique_experiments = df_agg[['Method', 'Embedding_Strategy', 'Hyperparameter']].drop_duplicates().to_records(index=False)
        for method, strategy, hyperparam_name in unique_experiments:
            df_exp = df_agg[(df_agg['Method'] == method) & (df_agg['Embedding_Strategy'] == strategy)].copy()
            if df_exp.empty: continue
            
            optimal_value = "N/A"
            if hyperparam_name != 'N/A':
                mean_perf_table = df_exp.groupby('Hyperparameter_Value')[metric].mean().reset_index()
                if not mean_perf_table.empty and not mean_perf_table[metric].isnull().all():
                    optimal_value = mean_perf_table.loc[mean_perf_table[metric].idxmax()]['Hyperparameter_Value']
                df_optimal_replicates = df_exp[df_exp['Hyperparameter_Value'] == optimal_value].copy()
            else:
                df_optimal_replicates = df_exp.copy()
            
            df_optimal_replicates['Optimal_Value'] = optimal_value
            optimal_replicate_dfs_for_metric.append(df_optimal_replicates)

        if not optimal_replicate_dfs_for_metric:
            logging.warning(f"Could not find any optimal runs for metric {metric}. Skipping summary section.")
            continue

        performance_data_at_best_params = pd.concat(optimal_replicate_dfs_for_metric, ignore_index=True)
        
        def p05(x): return x.quantile(0.05)
        def p95(x): return x.quantile(0.95)
        agg_dict = {'ROC_AUC': ['median', p05, p95], 'PR_AUC': ['median', p05, p95], 'EF_1Perc': ['median', p05, p95]}
        
        final_summary = performance_data_at_best_params.groupby(['Method', 'Embedding_Strategy']).agg(agg_dict)
        final_summary.columns = ['_'.join(col).strip() for col in final_summary.columns.values]
        final_summary = final_summary.reset_index()
        
        optimal_values_df = performance_data_at_best_params[['Method', 'Embedding_Strategy', 'Optimal_Value']].drop_duplicates()
        final_summary = pd.merge(final_summary, optimal_values_df, on=['Method', 'Embedding_Strategy'])
        
        median_col, p05_col, p95_col = f'{metric}_median', f'{metric}_p05', f'{metric}_p95'
        if not all(c in final_summary.columns for c in [median_col, p05_col, p95_col]): continue

        plot_data = final_summary.sort_values(by=median_col, ascending=False)
        
        plt.figure(figsize=(12, 8))
        ax = sns.barplot(data=plot_data, x=median_col, y='Method', hue='Embedding_Strategy', palette='viridis', dodge=True)
        
        # --- START OF DEFINITIVE FIX for Error Bars ---
        error_map = { (row['Method'], row['Embedding_Strategy']): (row[median_col] - row[p05_col], row[p95_col] - row[median_col]) for _, row in plot_data.iterrows() }
        
        # This new logic iterates through the bars and their containers robustly
        for container in ax.containers:
            # Each container holds all bars for a single strategy (e.g., all "Projection" bars)
            strategy_name = container.get_label()
            
            # Iterate over the individual bars in this container
            for bar in container.patches:
                # Get the method name from the y-tick label that corresponds to the bar's position
                method_name = ax.get_yticklabels()[int(round(bar.get_y() + bar.get_height() / 2.0))].get_text()
                
                key = (method_name, strategy_name)
                if key in error_map:
                    err = error_map[key]
                    x_pos = bar.get_x() + bar.get_width()
                    y_pos = bar.get_y() + bar.get_height() / 2.0
                    ax.errorbar(x=[x_pos], y=[y_pos], xerr=[[err[0]], [err[1]]], fmt='none', c='black', capsize=4)
        # --- END OF DEFINITIVE FIX ---

        plt.xlabel(f'Median {metric} (90% CI)'); plt.ylabel('Method (Representation)')
        plt.title(f'Peak Performance After Tuning (Optimized for {metric})')
        if "AUC" in metric: plt.xlim(0, max(1.0, plot_data[median_col].max() * 1.1 if not plot_data.empty else 1.0))
        if "EF" in metric: plt.xscale('log')
        plt.grid(True, axis='x', linestyle='--'); plt.legend(title='Embedding Strategy'); plt.tight_layout()

        fig_filename = f"fig_peak_performance_{metric}_median_split.png"
        fig_path = os.path.join(report_figures_abs_dir, fig_filename)
        plt.savefig(fig_path); plt.close()
        
        caption = f"Comparison of the best median {metric}, optimized independently for each method. Error bars are the 90% CI."
        add_figure_to_latex_standalone(latex_content, os.path.join("figures", fig_filename), caption, f"fig_peak_summary_{metric}_split")
        
        table_data_final = final_summary[['Method', 'Embedding_Strategy', 'Optimal_Value', median_col, p05_col, p95_col]].sort_values(by=median_col, ascending=False)
        table_label = f"table_summary_{metric}_split"
        csv_path = os.path.join(report_tables_abs_dir, f"{clean_for_label(table_label)}.csv")
        add_dataframe_as_latex_table_standalone(latex_content, table_data_final, f"Optimal performance ranked by median {metric}.", table_label, csv_path)

    # --- Final Report Generation ---
    latex_content.append(LATEX_DOCUMENT_END)
    report_tex_filename = "hyperparameter_analysis_report.tex"
    report_tex_path = os.path.join(report_output_abs_dir, report_tex_filename)
    with open(report_tex_path, "w", encoding='utf-8') as f: f.write("\n".join(latex_content))
    logging.info(f"Report generated: {report_tex_path}")
    try:
        for i in range(2): 
            subprocess.run(["pdflatex", "-interaction=nonstopmode", "-output-directory", report_output_abs_dir, report_tex_path], capture_output=True, text=True, check=False)
        pdf_path = os.path.join(report_output_abs_dir, os.path.basename(report_tex_path).replace('.tex', '.pdf'))
        if os.path.exists(pdf_path): logging.info(f"PDF successfully generated: {pdf_path}")
        else: logging.warning(f"PDF report not found after compilation attempts.")
    except Exception as e:
        logging.error(f"LaTeX compilation error: {e}", exc_info=True)

if __name__ == "__main__":
    main_report_generation()