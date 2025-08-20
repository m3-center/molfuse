import pandas as pd
import numpy as np
import os
import argparse
import logging
import json
import shutil
import glob
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend suitable for HPC/scripts
import matplotlib.pyplot as plt
import seaborn as sns 
import subprocess
import re
from datetime import datetime

# --- LaTeX Preamble and Helper Functions (Assumed to be complete and correct from previous versions) ---
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
This report presents an analysis of hyperparameter tuning experiments for the UMMBAS similarity space generation pipeline. The study focuses on evaluating the impact of key hyperparameters for UMAP (\texttt{n\_neighbors}) and t-SNE (\texttt{tsne\_pca\_components}) on virtual screening performance. Using a leave-one-target-out methodology with multiple replicate runs, we assess performance based on ROC-AUC, PR-AUC, and Enrichment Factors. The findings are used to determine optimal default settings for these algorithms within the context of MF-guided chemical space exploration.
\end{abstract} \clearpage \tableofcontents \clearpage \listoffigures \clearpage \listoftables \clearpage
\section{Introduction}
\textit{Placeholder for Introduction text...}
\section{Methodology Overview}
\textit{Placeholder for Methodology Overview text...}
"""
LATEX_DOCUMENT_END = r"\end{document}"

def escape_latex_text_content(text_input):
    # ... (same helper function)
    if not isinstance(text_input, str): text_input = str(text_input)
    conv = {'&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_',
            '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\^{}',
            '<': r'\textless{}', '>': r'\textgreater{}'}
    for k, v in conv.items(): text_input = text_input.replace(k, v)
    return text_input

def clean_for_label(text):
    # ... (same helper function)
    if not isinstance(text, str): text = str(text)
    text = text.replace('_', '-').replace(' ', '-').replace('.', '-').replace('/', '-')
    text = re.sub(r'[^a-zA-Z0-9-]', '', text); text = re.sub(r'-+', '-', text) 
    return text.strip('-')[:50] 

def get_section_header_latex_standalone(level, title_text):
    # ... (same helper function)
    sec_cmd_map = {1: r"\section", 2: r"\subsection", 3: r"\subsubsection", 4: r"\paragraph", 5: r"\subparagraph"}
    sec_cmd = sec_cmd_map.get(level, r"\paragraph")
    return f"\n{sec_cmd}{{{escape_latex_text_content(title_text)}}}\n"

def add_figure_to_latex_standalone(latex_content_list, relative_fig_path_in_tex, caption_text, label_text, placement="[H]", figure_width="0.8\\textwidth"): # Adjusted default
    # ... (same helper function)
    clean_label = clean_for_label(label_text)
    figure_path_for_latex = relative_fig_path_in_tex.replace(os.sep, '/')
    latex_content_list.extend([f"\\begin{{figure}}{placement}", r"  \centering",
                               f"  \\includegraphics[width={figure_width}]{{{figure_path_for_latex}}}",
                               f"  \\caption{{{escape_latex_text_content(caption_text)}}}",
                               f"  \\label{{fig:{clean_label}}}", r"\end{figure}", "\n"])

def add_dataframe_as_latex_table_standalone(latex_content_list, dataframe, caption_text, label_text, placement="[H]", col_format=None, font_size=r"\small"):
    # ... (same helper function, with booktabs=True removed from .to_latex call)
    clean_label = clean_for_label(label_text)
    if dataframe is not None and not dataframe.empty:
        latex_content_list.extend([f"\\begin{{table}}{placement}", r"  \centering"])
        if font_size: latex_content_list.append(font_size)
        latex_content_list.extend([f"  \\caption{{{escape_latex_text_content(caption_text)}}}",
                                   f"  \\label{{tab:{clean_label}}}"])
        df_for_latex = dataframe.copy()
        for col in df_for_latex.columns:
            is_preformatted = False
            try:
                if isinstance(df_for_latex[col].iloc[0], str) and '$\\pm$' in df_for_latex[col].iloc[0]:
                    is_preformatted = True
            except IndexError: is_preformatted = False
            if is_preformatted:
                df_for_latex[col] = df_for_latex[col].apply(lambda x: x.replace('_', r'\_'))
            elif pd.api.types.is_numeric_dtype(df_for_latex[col]):
                df_for_latex[col] = df_for_latex[col].apply(
                    lambda x: f"{x:.3f}" if pd.notna(x) and isinstance(x, (float, np.floating)) and ( (abs(x) >= 0.001 and abs(x) < 1000) or x==0) else 
                              (f"{x:.2e}" if pd.notna(x) and isinstance(x, (float, np.floating)) else 
                              (str(int(x)) if pd.notna(x) and isinstance(x, (int, np.integer)) else ("N/A" if pd.isna(x) else str(x))) ) )
            else: 
                df_for_latex[col] = df_for_latex[col].astype(str).apply(lambda x: "N/A" if pd.isna(x) or (isinstance(x,str) and x.lower() == 'nan') else escape_latex_text_content(x))

        df_for_latex.columns = [escape_latex_text_content(str(c).replace('_', ' ').title()) for c in dataframe.columns]
        if col_format is None:
            col_format = 'l' * len(df_for_latex.columns)
        latex_table_string = df_for_latex.to_latex(index=False, escape=False, column_format=col_format,
                                                   longtable=len(dataframe)>20, na_rep="N/A", booktabs=True)
        latex_content_list.append(latex_table_string)
        latex_content_list.extend([r"\end{table}", "\n"])
    else: latex_content_list.append(f"% Table '{escape_latex_text_content(label_text)}' is empty or None.\n")
# --- End LaTeX Helpers ---

agg_log_file_name = f"hyperparam_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)-8s - %(filename)-25s - %(funcName)-25s - %(lineno)-4d - %(message)s',
                    handlers=[ logging.FileHandler(agg_log_file_name, mode='w'), logging.StreamHandler() ])

def find_all_replicate_runs(base_experiment_dir):
    # ... (same robust function from last version) ...
    replicate_dirs = []
    pattern = os.path.join(base_experiment_dir, "run_seed*_repr*_*_*")
    logging.info(f"Scanning for all replicate run directories with pattern: {pattern}")
    dir_pattern_re = re.compile(r"run_seed(\d+)_repr(features|fingerprints)_(\d{8}_\d{6})")
    for dir_path in glob.glob(pattern):
        if os.path.isdir(dir_path):
            basename = os.path.basename(dir_path)
            match = dir_pattern_re.match(basename)
            if match:
                try:
                    seed_val, repr_val = int(match.group(1)), match.group(2)
                    replicate_dirs.append({"path": dir_path, "seed": seed_val, "repr_mode": repr_val})
                    logging.info(f"Found valid replicate run: {dir_path} (Seed: {seed_val}, Repr: {repr_val})")
                except (ValueError, IndexError) as e: logging.warning(f"Could not parse details from {basename}: {e}")
            else: logging.debug(f"Directory matched glob but not regex, skipping: {basename}")
    if not replicate_dirs: logging.warning(f"No valid replicate run directories found in '{base_experiment_dir}'")
    return sorted(replicate_dirs, key=lambda x: (x["seed"], x["repr_mode"]))


def collect_metrics_from_replicates(replicate_run_details):
    all_metrics_data = []
    for rep_info in replicate_run_details:
        rep_dir_path, replicate_seed, repr_type = rep_info["path"], rep_info["seed"], rep_info["repr_mode"]
        logging.info(f"Processing replicate: {os.path.basename(rep_dir_path)}")
        
        # --- NEW: Load the run-specific config ---
        run_config_path = os.path.join(rep_dir_path, "run_config.json")
        if not os.path.exists(run_config_path):
            logging.warning(f"  run_config.json not found in {rep_dir_path}. Skipping replicate.")
            continue
        try:
            with open(run_config_path, 'r') as f:
                run_config = json.load(f)
        except Exception as e:
            logging.error(f"  Error reading {run_config_path}: {e}. Skipping replicate.")
            continue
        # --- END NEW ---

        gs = run_config['global_settings']
        
        for target_info in run_config['targets']:
            if target_info.get("processing_mode", "full_analysis") == "similarity_space_only": continue
            
            target_id_name = target_info['id_name']
            target_results_base = os.path.join(rep_dir_path, target_id_name, "results", repr_type)
            if not os.path.exists(target_results_base): continue

            for dr_key, dr_params in run_config["dimensionality_reduction_methods"].items():
                dr_short_name_base = dr_params["short_name"]
                
                strategies_to_check = [{"dir_leaf": dr_short_name_base.replace('-', '_'), "label": "Projection" if not dr_key == "tsne" else "Co-embedding (Native)"}]
                if gs.get("run_coembedding_for_pca_umap") and dr_params.get("allow_coembedding") and not dr_key == "tsne":
                    strategies_to_check.append({"dir_leaf": f"{dr_short_name_base}-Coembed".replace('-', '_'), "label": "Co-embedding"})
                
                for strategy in strategies_to_check:
                    for dim_val in gs['simspace_dims_to_test']:
                        if dr_key == "tsne" and dim_val != 2: continue
                        
                        strat_dir_path = os.path.join(target_results_base, f"dim_{dim_val}", strategy['dir_leaf'])
                        base_dr_name_for_metrics_file = dr_short_name_base.replace('-', '_')
                        metrics_filename = f"{target_id_name}_{repr_type}_{base_dr_name_for_metrics_file}_dim{dim_val}_ranking_metrics.csv"
                        metrics_file_path = os.path.join(strat_dir_path, metrics_filename)

                        if os.path.exists(metrics_file_path):
                            try:
                                df_m = pd.read_csv(metrics_file_path)
                                if not df_m.empty:
                                    metric_row = {'Replicate_Seed': replicate_seed, 'Target': target_info['display_name'],
                                                  'Representation': repr_type.capitalize(), 'DR_Method': dr_short_name_base,
                                                  'Embedding_Strategy': strategy['label'], 'DIM': dim_val,
                                                  # --- NEW: Capture hyperparameters from config ---
                                                  'UMAP_n_neighbors': dr_params.get('n_neighbors'),
                                                  'tSNE_PCA_Components': gs.get('tsne_pca_components')}
                                    
                                    # ... (baseline calculation and metric mapping - same as before) ...
                                    column_map = {'roc_auc': 'ROC_AUC', 'pr_auc': 'PR_AUC', 'ef_1%': 'EF_1Perc',
                                                  'ef_5%': 'EF_5Perc', 'ef_10%': 'EF_10Perc', 
                                                  'spearman_rho_affinity_vs_score': 'spearman_rho_affinity_vs_score'}
                                    for csv_col, df_col in column_map.items():
                                        if csv_col in df_m.columns:
                                            metric_row[df_col] = df_m[csv_col].iloc[0] if pd.notna(df_m[csv_col].iloc[0]) else np.nan
                                        else: metric_row[df_col] = np.nan
                                    all_metrics_data.append(metric_row)
                            except Exception as e: 
                                logging.error(f"Error reading metrics from {metrics_file_path}: {e}")
    
    if not all_metrics_data: 
        logging.error("No metric data collected.")
    return pd.DataFrame(all_metrics_data) if all_metrics_data else pd.DataFrame()

def main_report_generation():
    parser = argparse.ArgumentParser(description="Analyze hyperparameter sweep results and generate a LaTeX report.")
    parser.add_argument("--base_experiment_dir", required=True, help="Base directory containing all hyperparameter sweep run folders.")
    # No config path needed, as it's read from each run
    parser.add_argument("--output_report_dir", required=True, help="Directory to save the final LaTeX report.")
    args = parser.parse_args()

    logging.info(f"--- STARTING HYPERPARAMETER ANALYSIS REPORT GENERATION ---")

    report_run_id = f"hyperparam_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    report_output_abs_dir = os.path.join(args.output_report_dir, report_run_id)
    report_figures_abs_dir = os.path.join(report_output_abs_dir, "figures"); os.makedirs(report_figures_abs_dir, exist_ok=True)
    latex_content = [LATEX_DOCUMENT_PREAMBLE.replace("<<RUN_ID_PLACEHOLDER>>", escape_latex_text_content(report_run_id))]
    
    replicate_details_list = find_all_replicate_runs(args.base_experiment_dir)
    if not replicate_details_list: logging.error("No replicate directories found. Aborting."); return
    
    df_agg = collect_metrics_from_replicates(replicate_details_list)
    if df_agg.empty: logging.error("Aggregated metrics DataFrame is empty. Aborting."); return
    df_agg.to_csv(os.path.join(report_output_abs_dir, "DEBUG_hyperparam_metrics.csv"), index=False)
    
    # --- Analysis Section ---
    latex_content.append(get_section_header_latex_standalone(1, 'Hyperparameter Sweep Analysis'))
    
    # Generic plotting function for hyperparameter vs. metric
    def create_hyperparam_plot(df, hyperparam_col, metric_col, title, filename, latex_content_list, report_figures_dir):
        logging.info(f"Generating plot: {title}")
        try:
            plt.figure(figsize=(8, 6))
            # Group by the hyperparameter and calculate mean/std of the metric across replicates
            summary = df.groupby(hyperparam_col)[metric_col].agg(['mean', 'std']).reset_index().fillna(0)
            plt.plot(summary[hyperparam_col], summary['mean'], marker='o', linestyle='-', label=f"Mean {metric_col}")
            plt.fill_between(summary[hyperparam_col], summary['mean'] - summary['std'], summary['mean'] + summary['std'], alpha=0.2, label=f"$\\pm$1 SD")
            plt.xlabel(hyperparam_col.replace('_', ' ').title()); plt.ylabel(f"Mean {metric_col}")
            plt.title(title); plt.grid(True, linestyle=':'); plt.legend(); plt.tight_layout()
            
            fig_path = os.path.join(report_figures_dir, filename)
            plt.savefig(fig_path); plt.close()
            add_figure_to_latex_standalone(latex_content, os.path.join("figures", filename), title, f"fig-{clean_for_label(filename)}")
        except Exception as e:
            logging.error(f"Failed to generate plot '{title}': {e}", exc_info=True)

    # --- UMAP n_neighbors Analysis ---
    latex_content.append(get_section_header_latex_standalone(2, 'UMAP: n_neighbors Evaluation'))
    # Features
    df_umap_feat = df_agg[(df_agg['Representation'] == 'Features') & (df_agg['DR_Method'] == 'UMAP-Euclidean')]
    if not df_umap_feat.empty:
        create_hyperparam_plot(df_umap_feat, 'UMAP_n_neighbors', 'ROC_AUC', 
                               'UMAP n_neighbors vs. ROC-AUC (Features, Euclidean)', 
                               'fig_umap_n_roc_feat.png', latex_content, report_figures_abs_dir)
        create_hyperparam_plot(df_umap_feat, 'UMAP_n_neighbors', 'EF_1Perc', 
                               'UMAP n_neighbors vs. EF@1% (Features, Euclidean)', 
                               'fig_umap_n_ef1_feat.png', latex_content, report_figures_abs_dir)
    
    # Fingerprints
    df_umap_fp = df_agg[(df_agg['Representation'] == 'Fingerprints') & (df_agg['DR_Method'] == 'UMAP-Jaccard')]
    if not df_umap_fp.empty:
        create_hyperparam_plot(df_umap_fp, 'UMAP_n_neighbors', 'ROC_AUC', 
                               'UMAP n_neighbors vs. ROC-AUC (Fingerprints, Jaccard)', 
                               'fig_umap_n_roc_fp.png', latex_content, report_figures_abs_dir)
        create_hyperparam_plot(df_umap_fp, 'UMAP_n_neighbors', 'EF_1Perc', 
                               'UMAP n_neighbors vs. EF@1% (Fingerprints, Jaccard)', 
                               'fig_umap_n_ef1_fp.png', latex_content, report_figures_abs_dir)
    
    # --- t-SNE PCA Components Analysis ---
    latex_content.append(f"\\clearpage\n{get_section_header_latex_standalone(2, 't-SNE: PCA Components Evaluation (Fingerprints)')}")
    df_tsne_fp = df_agg[(df_agg['Representation'] == 'Fingerprints') & (df_agg['DR_Method'] == 't-SNE')]
    if not df_tsne_fp.empty:
        create_hyperparam_plot(df_tsne_fp, 'tSNE_PCA_Components', 'ROC_AUC', 
                               't-SNE PCA Components vs. ROC-AUC (Fingerprints)', 
                               'fig_tsne_pca_roc_fp.png', latex_content, report_figures_abs_dir)
        create_hyperparam_plot(df_tsne_fp, 'tSNE_PCA_Components', 'EF_1Perc', 
                               't-SNE PCA Components vs. EF@1% (Fingerprints)', 
                               'fig_tsne_pca_ef1_fp.png', latex_content, report_figures_abs_dir)
    
    latex_content.append(LATEX_DOCUMENT_END)

    # Save and compile LaTeX
    report_tex_filename = "hyperparameter_analysis_report.tex"
    report_tex_path = os.path.join(report_output_abs_dir, report_tex_filename)
    with open(report_tex_path, "w", encoding='utf-8') as f: f.write("\n".join(latex_content))
    logging.info(f"Hyperparameter analysis report structure generated: {report_tex_path}")
    
    try:
        logging.info(f"Attempting to compile LaTeX report in: {report_output_abs_dir}")
        for i in range(2): 
            process = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-output-directory", report_output_abs_dir, report_tex_path],
                capture_output=True, text=True, check=False )
            if process.returncode != 0: 
                logging.error(f"pdflatex compilation pass {i+1} FAILED."); 
                latex_log_file = os.path.join(report_output_abs_dir, os.path.basename(report_tex_path).replace('.tex', '.log'))
                if os.path.exists(latex_log_file):
                    with open(latex_log_file, 'r', encoding='utf-8', errors='ignore') as lf:
                        logging.error("Key LaTeX errors:\n" + "".join(lf.readlines()[-50:]))
                else: logging.error("LaTeX .log file not found. STDERR:\n" + process.stderr)
            else: logging.info(f"pdflatex compilation pass {i+1} successful.")
        pdf_path = os.path.join(report_output_abs_dir, os.path.basename(report_tex_path).replace('.tex', '.pdf'))
        if os.path.exists(pdf_path): logging.info(f"PDF report successfully generated: {pdf_path}")
        else: logging.warning(f"PDF report {pdf_path} not found after compilation attempts.")
    except Exception as e: logging.error(f"LaTeX compilation error: {e}", exc_info=True)


if __name__ == "__main__":
    main_report_generation()