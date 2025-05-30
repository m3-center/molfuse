import pandas as pd
import numpy as np
import os
import argparse
import logging
import json
import shutil # For copying figures
import subprocess # For pdflatex
from collections import defaultdict
import matplotlib.pyplot as plt # For generating dim_opt plots

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)-8s - %(filename)-25s - %(funcName)-25s - %(message)s',
                    handlers=[logging.FileHandler("generate_report.log"), logging.StreamHandler()])

LATEX_TEMPLATE_HEADER = r"""
\documentclass[10pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[margin=0.75in]{geometry}
\usepackage{graphicx}
\usepackage{float} 
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{amssymb}
\usepackage{booktabs} % Still include for general table quality, even if to_latex doesn't use the arg
\usepackage{longtable}
\usepackage{caption}
\usepackage{subcaption} 
\usepackage{array} 
\usepackage{xcolor}
\usepackage{hyperref} 
\usepackage{fancyhdr}
\usepackage{lastpage}
\usepackage{textcomp} % For \textunderscore

\hypersetup{
    colorlinks=true, linkcolor=blue, filecolor=magenta, urlcolor=cyan,
    pdftitle={UMMBAS Similarity Experiment Report}, pdfauthor={UMMBAS Project Team},
    pdfsubject={Molecular Similarity Analysis}, pdfkeywords={UMMBAS, Cheminformatics, Similarity},
    bookmarksnumbered=true, pdfpagemode=UseOutlines
}

\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{UMMBAS Experimental Report}
\fancyhead[R]{\today}
\fancyfoot[C]{\thepage\ of \pageref{LastPage}}

\title{UMMBAS Molecular Similarity Experimental Evaluation Report}
\author{UMMBAS Project Team}
\date{\today}


\begin{document}
\maketitle
\begin{abstract}
This report details experiments conducted to evaluate the predictive capacity of molecular similarity spaces using a leave-one-target-out inspired methodology. The central hypothesis posits that ligands active against a specific protein target will exhibit proximity to compounds known to interact with other proteins sharing the same broader molecular function, even when the specific target's known ligands are excluded during the construction of the similarity space. Experiments encompassed multiple target proteins, two molecular representations (physicochemical features and ECFP4 fingerprints), various dimensionality reduction (DR) techniques (PCA; UMAP with Euclidean, Cosine, Manhattan, and Hamming metrics; and t-SNE with pre-PCA), and a range of similarity space dimensionalities. Evaluation primarily focused on the distances of projected known target ligands to the molecular function (MF) cloud within these generated similarity spaces.
\end{abstract}
\clearpage
\tableofcontents
\clearpage

\section{Introduction}
The exploration of chemical space for novel therapeutic agents is a cornerstone of drug discovery. Molecular similarity, a fundamental concept in cheminformatics, suggests that structurally similar molecules are likely to exhibit similar biological activities. This principle underpins many virtual screening and lead optimization strategies. This study investigates the utility of constructing and analyzing molecular similarity spaces to predict potential interactions between small molecules and protein targets, grouped by their shared molecular function.

The experimental design aims to rigorously test whether general molecular function similarity can guide the identification of ligands for a specific, "held-out" protein target. By systematically varying data representations, dimensionality reduction methods, and the dimensionality of the resulting spaces, we seek to identify optimal parameters and assess the overall robustness of this similarity-based approach. Key metrics involve measuring the proximity of known active ligands (for the held-out target) to the cloud of compounds associated with the broader molecular function after projection into these tailored similarity spaces.
"""

LATEX_TEMPLATE_FOOTER = r"""
\clearpage
\phantomsection % Ensure correct link target for listoffigures
\addcontentsline{toc}{section}{List of Figures}
\listoffigures
\clearpage
\phantomsection % Ensure correct link target for listoftables
\addcontentsline{toc}{section}{List of Tables}
\listoftables
\end{document}
"""

def sanitize_for_latex(text_string):
    """Escapes special LaTeX characters in a string."""
    if not isinstance(text_string, str):
        text_string = str(text_string)
    replacements = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_', # Use our custom command for text underscores
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}',
        '\\_': r'_',
    }
    for char, escaped_char in replacements.items():
        text_string = text_string.replace(char, escaped_char)
    return text_string

def sanitize_for_pdf_bookmark(text_string):
    """Creates a plain text version of a string for PDF bookmarks."""
    if not isinstance(text_string, str):
        text_string = str(text_string)
    # Replace problematic LaTeX commands or characters with plain text equivalents
    text_string = text_string.replace('_', ' ') # Replace underscore with space
    text_string = text_string.replace(r'\&', 'and')
    text_string = text_string.replace(r'\%', ' percent')
    text_string = text_string.replace(r'\#', '')
    # Remove any other LaTeX specific markup or too complex characters
    # A more robust solution might involve a library or more extensive regex
    pdf_bookmark = ''.join(c for c in text_string if c.isalnum() or c.isspace() or c in ['.', ',', '(', ')', '-', ':', '[', ']'])
    return pdf_bookmark.strip()


def get_section_header_latex(level, title_str):
    sec_cmd_map = {1: r"\section", 2: r"\subsection", 3: r"\subsubsection", 4: r"\paragraph"}
    sec_cmd = sec_cmd_map.get(level, r"\paragraph")
    
    latex_display_title = sanitize_for_latex(title_str)
    pdf_bookmark_title = sanitize_for_pdf_bookmark(title_str)
    
    # Use \texorpdfstring to provide both versions to hyperref
    final_title_for_cmd = f"\\texorpdfstring{{{latex_display_title}}}{{{pdf_bookmark_title}}}"
    
    return f"\n{sec_cmd}{{{final_title_for_cmd}}}\n"


def add_figure_to_latex(latex_content_list, relative_fig_path_in_tex, caption_text, label_text, figure_width="0.75\\textwidth"):
    safe_caption = sanitize_for_latex(caption_text)
    safe_label = sanitize_for_latex(label_text).replace(r'\_', '_') # Labels can have underscores

    latex_content_list.append(r"\begin{figure}[H]")
    latex_content_list.append(r"  \centering")
    latex_content_list.append(f"  \\includegraphics[width={figure_width}]{{{relative_fig_path_in_tex}}}")
    latex_content_list.append(f"  \\caption{{{safe_caption}}}")
    latex_content_list.append(f"  \\label{{fig:{safe_label}}}")
    latex_content_list.append(r"\end{figure}")
    latex_content_list.append("\n")

def add_dataframe_as_latex_table(latex_content_list, dataframe, caption_text, label_text, col_format=None):
    if dataframe is not None and not dataframe.empty:
        safe_caption = sanitize_for_latex(caption_text)
        safe_label = sanitize_for_latex(label_text).replace(r'\_', '_')

        latex_content_list.append(r"\begin{table}[H]")
        latex_content_list.append(r"  \centering")
        latex_content_list.append(r"  \small") 
        latex_content_list.append(f"  \\caption{{{safe_caption}}}")
        latex_content_list.append(f"  \\label{{tab:{safe_label}}}")
        
        df_for_latex = dataframe.copy()
        for col in df_for_latex.select_dtypes(include=np.number).columns:
            df_for_latex[col] = df_for_latex[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) and isinstance(x, (int,float)) else sanitize_for_latex(str(x)))
        
        df_for_latex.columns = [sanitize_for_latex(col.replace('_', ' ').title()) for col in df_for_latex.columns]
        
        if col_format is None:
            col_format = '|' + 'l|' * len(df_for_latex.columns)

        # Remove booktabs=True if Pandas version is < 1.0.0
        # For current testing, assuming Pandas is modern enough or error will resurface if not.
        # If TypeError persists for 'booktabs', remove it from the to_latex call.
        try:
            latex_table_string = df_for_latex.to_latex(index=False, escape=False, column_format=col_format, booktabs=True)
        except TypeError:
            logging.warning("Pandas to_latex() does not support 'booktabs'. Generating table without it.")
            latex_table_string = df_for_latex.to_latex(index=False, escape=False, column_format=col_format)

        latex_content_list.append(latex_table_string)
        latex_content_list.append(r"\end{table}")
        latex_content_list.append("\n")
    else:
        latex_content_list.append(f"% Table data for '{sanitize_for_latex(label_text)}' is empty or None.\n")


def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX report from experiment results.")
    parser.add_argument("--experiment_run_dir", required=True, help="Path to the timestamped main experiment run directory.")
    parser.add_argument("--config_path", required=True, help="Path to the experiment_config.json file.")
    parser.add_argument("--output_dir", required=True, help="Directory to save the LaTeX report and figures.")
    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        config = json.load(f)
    gs = config['global_settings']
    
    report_filename_base = os.path.basename(args.experiment_run_dir) + "_report"
    report_output_abs_dir = os.path.join(args.output_dir, report_filename_base)
    report_figures_abs_dir = os.path.join(report_output_abs_dir, "figures")
    os.makedirs(report_figures_abs_dir, exist_ok=True)
    
    latex_content = [LATEX_TEMPLATE_HEADER]

    latex_content.append(get_section_header_latex(1, "Overall Summary and Key Findings"))
    all_targets_summary_metrics = [] 


    for target_info in config['targets']:
        target_id_name = target_info['id_name']
        target_display_name = target_info['display_name']
        mf_display_name = target_info['molecular_function_display_name']
        
        latex_content.append(f"\\clearpage\n{get_section_header_latex(1, f'Target: {target_display_name} (MF: {mf_display_name})')}")
        target_results_base_dir = os.path.join(args.experiment_run_dir, target_id_name, "results")

        if not os.path.exists(target_results_base_dir):
            latex_content.append(f"Results not found for target {sanitize_for_latex(target_id_name)}.\n")
            continue

        latex_content.append(get_section_header_latex(2, f"Optimization of Similarity Space Dimensionality (SIMSPACE\_DIM)"))
        simspace_dims_str = ', '.join(map(str, gs['simspace_dims_to_test']))
        latex_content.append(f"The optimal dimensionality for similarity projections was investigated by testing SIMSPACE\_DIM values of {simspace_dims_str}. "
                             "The mean of the minimum Euclidean distances from projected target ligands to the molecular function cloud served as the primary metric.\n")

        target_dim_opt_summary_rows = []

        for repr_type in config['representations']:
            latex_content.append(get_section_header_latex(3, f"Representation: {repr_type.capitalize()}"))
            
            for dr_key, dr_params in config["dimensionality_reduction_methods"].items():
                dr_display_name = dr_params["short_name"]
                dr_short_name_fs = dr_display_name.replace('-', '_').replace(' ', '_') # More robust fs name

                dim_vs_dist_data = []
                for simspace_dim_val in gs['simspace_dims_to_test']:
                    if dr_key == "tsne" and simspace_dim_val != 2:
                        continue 
                    
                    results_path_for_dim_dr = os.path.join(target_results_base_dir, repr_type, f"dim_{simspace_dim_val}", dr_short_name_fs)
                    distances_csv = os.path.join(results_path_for_dim_dr, f"{target_id_name}_{repr_type}_{dr_short_name_fs}_dim{simspace_dim_val}_distances.csv")
                    
                    if os.path.exists(distances_csv):
                        try:
                            df_dist = pd.read_csv(distances_csv)
                            if 'min_dist_to_mf_cloud' in df_dist and not df_dist['min_dist_to_mf_cloud'].dropna().empty:
                                mean_min_d = df_dist['min_dist_to_mf_cloud'].dropna().mean()
                                dim_vs_dist_data.append({'dim': simspace_dim_val, 'mean_min_dist': mean_min_d})
                        except Exception as e:
                            logging.warning(f"Could not process distances CSV {distances_csv}: {e}")
                
                if dim_vs_dist_data:
                    df_plot = pd.DataFrame(dim_vs_dist_data).sort_values(by='dim')
                    
                    plt.figure(figsize=(8, 5)) # Adjusted size
                    plt.plot(df_plot['dim'], df_plot['mean_min_dist'], marker='o', linestyle='-') # Removed label=dr_display_name for cleaner plot if many DRs
                    plt.xlabel(sanitize_for_latex("SIMSPACE\_DIM Value"))
                    plt.ylabel(sanitize_for_latex("Mean Min. Distance to MF Cloud"))
                    plt.title(sanitize_for_latex(f"SIMSPACE\_DIM Opt. for {target_display_name}\n({repr_type.capitalize()}, {dr_display_name})"), fontsize=11)
                    # Only show ticks that have data, esp. for t-SNE
                    actual_dims_plotted = df_plot['dim'].unique()
                    plt.xticks(actual_dims_plotted) 
                    plt.grid(True, linestyle='--', alpha=0.6)
                    
                    plot_filename = f"{target_id_name}_{repr_type}_{dr_short_name_fs}_dim_vs_mindist.png"
                    plot_abs_path_dest = os.path.join(report_figures_abs_dir, plot_filename)
                    try:
                        plt.savefig(plot_abs_path_dest, dpi=150, bbox_inches='tight')
                        plot_relative_path_for_tex = os.path.join("figures", plot_filename).replace('\\', '/') # Ensure fwd slashes
                        caption = f"Mean minimum distance to MF cloud vs. SIMSPACE\_DIM for target {target_display_name} ({repr_type.capitalize()}, DR: {dr_display_name})."
                        add_figure_to_latex(latex_content, plot_relative_path_for_tex, caption, f"{target_id_name}_{repr_type}_{dr_short_name_fs}_dimopt")
                    except Exception as e: logging.error(f"Failed to save dim_opt plot {plot_abs_path_dest}: {e}")
                    plt.close()

                    if not df_plot.empty:
                        best_row = df_plot.loc[df_plot['mean_min_dist'].idxmin()]
                        target_dim_opt_summary_rows.append({
                            'Representation': repr_type.capitalize(), 'DR_Method': dr_display_name, # Changed column name
                            'Optimal_SIMSPACE\_DIM': int(best_row['dim']), 
                            'Mean_Min_Dist_at_Optimal_DIM': best_row['mean_min_dist']
                        })
            latex_content.append("\\clearpage\n")
        
        if target_dim_opt_summary_rows:
            df_target_dim_opt_summary = pd.DataFrame(target_dim_opt_summary_rows)
            latex_content.append(get_section_header_latex(3, f"Summary of Optimal Dimensionality for {target_display_name}"))
            add_dataframe_as_latex_table(latex_content, df_target_dim_opt_summary,
                                         f"Optimal SIMSPACE\_DIM and corresponding Mean Minimum Distance for {target_display_name}.",
                                         f"opt_dim_summary_{target_id_name}")
            for _, row_data in df_target_dim_opt_summary.iterrows(): # Use iterrows for DataFrame
                all_targets_summary_metrics.append({'Target': target_display_name, **row_data.to_dict()})
        latex_content.append("\\clearpage\n")

        latex_content.append(get_section_header_latex(2, f"Detailed 2D Projections (SIMSPACE\_DIM = 2)"))
        dim_2_exists_for_target = False
        for repr_type in config['representations']:
            repr_has_2d_plot = False
            for dr_key, dr_params in config["dimensionality_reduction_methods"].items():
                dr_display_name = dr_params["short_name"]
                dr_short_name_fs = dr_display_name.replace('-', '_').replace(' ', '_')
                
                dim2_results_path = os.path.join(target_results_base_dir, repr_type, "dim_2", dr_short_name_fs)
                if os.path.exists(dim2_results_path):
                    if not repr_has_2d_plot:
                        latex_content.append(get_section_header_latex(3, f"Representation: {repr_type.capitalize()} (2D Projections)"))
                        repr_has_2d_plot = True
                        dim_2_exists_for_target = True

                    latex_content.append(get_section_header_latex(4, f"DR Method: {dr_display_name} (2D)"))
                    
                    scatter_filename_base = f"{target_id_name}_{repr_type}_{dr_short_name_fs}_dim2_scatter.png"
                    hist_filename_base = f"{target_id_name}_{repr_type}_{dr_short_name_fs}_dim2_min_distances_hist.png"
                    
                    scatter_src_path = os.path.join(dim2_results_path, scatter_filename_base)
                    hist_src_path = os.path.join(dim2_results_path, hist_filename_base)
                    
                    if os.path.exists(scatter_src_path):
                        shutil.copy(scatter_src_path, os.path.join(report_figures_abs_dir, scatter_filename_base))
                        caption_s = f"2D Similarity space for {target_display_name} ({repr_type.capitalize()}, {dr_display_name})."
                        add_figure_to_latex(latex_content, os.path.join("figures", scatter_filename_base).replace('\\','/'), caption_s, f"{target_id_name}_{repr_type}_{dr_short_name_fs}_scatter2D")

                    if os.path.exists(hist_src_path):
                        shutil.copy(hist_src_path, os.path.join(report_figures_abs_dir, hist_filename_base))
                        dist_csv = os.path.join(dim2_results_path, f"{target_id_name}_{repr_type}_{dr_short_name_fs}_dim2_distances.csv")
                        stats_str = ""
                        if os.path.exists(dist_csv):
                            try:
                                df_d = pd.read_csv(dist_csv)
                                if 'min_dist_to_mf_cloud' in df_d and not df_d['min_dist_to_mf_cloud'].dropna().empty:
                                    mean_d = df_d['min_dist_to_mf_cloud'].dropna().mean()
                                    median_d = df_d['min_dist_to_mf_cloud'].dropna().median()
                                    stats_str = f" Mean min dist: {mean_d:.3f}, Median: {median_d:.3f}."
                            except: pass
                        caption_h = f"Histogram of minimum distances for {target_display_name} ({repr_type.capitalize()}, {dr_display_name}, 2D).{stats_str}"
                        add_figure_to_latex(latex_content, os.path.join("figures", hist_filename_base).replace('\\','/'), caption_h, f"{target_id_name}_{repr_type}_{dr_short_name_fs}_hist2D", figure_width="0.7\\textwidth")
            if repr_has_2d_plot: latex_content.append("\\clearpage\n")
        if not dim_2_exists_for_target:
            latex_content.append("No 2D projection results were found for this target to display.\n")

    latex_content.append(f"\\clearpage\n{get_section_header_latex(1, 'Overall Comparative Summary')}")
    if all_targets_summary_metrics:
        df_all_summary = pd.DataFrame(all_targets_summary_metrics)
        df_all_summary = df_all_summary[['Target', 'Representation', 'DR_Method', 'Optimal_SIMSPACE\_DIM', 'Mean_Min_Dist_at_Optimal_DIM']]
        df_all_summary.sort_values(by=['Target', 'Representation', 'Mean_Min_Dist_at_Optimal_DIM'], inplace=True)
        
        latex_content.append("The table below summarizes the optimal SIMSPACE\_DIM and corresponding mean minimum distance to the MF cloud for each combination.\n")
        add_dataframe_as_latex_table(latex_content, df_all_summary,
                                     "Overall Summary: Optimal SIMSPACE\_DIM and Mean Minimum Distances.",
                                     "overall_summary_opt_dims", col_format='|p{3.2cm}|p{2cm}|p{2.8cm}|c|r|') # Adjusted col format
    else:
        latex_content.append("No summary data collected across targets.\n")

    latex_content.append(LATEX_TEMPLATE_FOOTER)

    report_tex_path = os.path.join(report_output_abs_dir, "experiment_report.tex")
    with open(report_tex_path, "w", encoding='utf-8') as f:
        f.write("\n".join(latex_content))
    
    logging.info(f"LaTeX report generated: {report_tex_path}")
    logging.info(f"Figures for report copied to: {report_figures_abs_dir}")

    try:
        logging.info(f"Attempting to compile LaTeX report in: {report_output_abs_dir}")
        for i in range(2): # Run pdflatex twice for TOC, references, etc.
            logging.info(f"pdflatex compilation pass {i+1}...")
            process = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-output-directory", report_output_abs_dir, report_tex_path],
                capture_output=True, text=True, encoding='utf-8', errors='replace', check=False
            )
            if process.returncode != 0:
                logging.error(f"pdflatex compilation failed on pass {i+1}. Log output below.")
                logging.error("STDOUT:\n" + process.stdout)
                logging.error("STDERR:\n" + process.stderr)
                log_file_path = os.path.join(report_output_abs_dir, "experiment_report.log")
                if os.path.exists(log_file_path): logging.error(f"See {log_file_path} for detailed LaTeX errors.")
                break 
        else: 
             logging.info(f"PDF report compilation attempt finished. Check {report_output_abs_dir} for experiment_report.pdf")
    except FileNotFoundError:
        logging.warning("pdflatex command not found. Please compile the .tex file manually.")
    except Exception as e:
        logging.error(f"An unexpected error occurred during pdflatex compilation: {e}")

if __name__ == "__main__":
    main()