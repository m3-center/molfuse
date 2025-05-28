import pandas as pd
import numpy as np
import os
import argparse
import logging
import json
import glob # Not strictly needed if paths are constructed directly, but can be useful
import shutil
from collections import defaultdict
import subprocess # For pdflatex
import matplotlib.pyplot as plt # For generating dim_opt plots within this script

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("generate_report.log"), logging.StreamHandler()])

LATEX_TEMPLATE_HEADER = r"""
\documentclass[10pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[margin=0.75in]{geometry} % Slightly wider margins
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
\usepackage{fancyhdr} % For headers/footers
\usepackage{lastpage} % For Page X of Y

\hypersetup{
    colorlinks=true, linkcolor=blue, filecolor=magenta, urlcolor=cyan,
    pdftitle={UMMBAS Similarity Experiment Report}, pdfauthor={UMMBAS Project Team},
    pdfsubject={Molecular Similarity Analysis}, pdfkeywords={UMMBAS, Cheminformatics, Similarity},
    bookmarksnumbered=true, pdfpagemode=UseOutlines % Show bookmarks panel
}

\pagestyle{fancy}
\fancyhf{} % Clear all header and footer fields
\fancyhead[L]{UMMBAS Experimental Report}
\fancyhead[R]{\today}
\fancyfoot[C]{\thepage\ of \pageref{LastPage}} % Page X of Y

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
\listoffigures
\clearpage
\listoftables
\end{document}
"""

def get_section_header_latex(level, title):
    sec_cmd_map = {1: r"\section", 2: r"\subsection", 3: r"\subsubsection", 4: r"\paragraph"}
    sec_cmd = sec_cmd_map.get(level, r"\paragraph") # Default to paragraph for deeper levels
    # Sanitize title for LaTeX: replace underscores, escape special chars if any more complex titles
    safe_title = title.replace('_', r'\_') 
    return f"\n{sec_cmd}{{{safe_title}}}\n"

def add_figure_to_latex(latex_content_list, relative_fig_path_in_tex, caption_text, label_text, figure_width="0.75\\textwidth"):
    # Assumes fig_path is relative to the .tex file, typically "figures/filename.png"
    latex_content_list.append(r"\begin{figure}[H]")
    latex_content_list.append(r"  \centering")
    latex_content_list.append(f"  \\includegraphics[width={figure_width}]{{{relative_fig_path_in_tex}}}")
    latex_content_list.append(f"  \\caption{{{caption_text.replace('_', r'\_')}}}")
    latex_content_list.append(f"  \\label{{fig:{label_text}}}")
    latex_content_list.append(r"\end{figure}")
    latex_content_list.append("\n")

def add_dataframe_as_latex_table(latex_content_list, dataframe, caption_text, label_text, col_format=None):
    if dataframe is not None and not dataframe.empty:
        latex_content_list.append(r"\begin{table}[H]")
        latex_content_list.append(r"  \centering")
        latex_content_list.append(r"  \small") # Make table text smaller if needed
        latex_content_list.append(f"  \\caption{{{caption_text.replace('_', r'\_')}}}")
        latex_content_list.append(f"  \\label{{tab:{label_text}}}")
        
        df_for_latex = dataframe.copy()
        # Format numeric columns to a few decimal places
        for col in df_for_latex.select_dtypes(include=np.number).columns:
            df_for_latex[col] = df_for_latex[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
        
        # Sanitize column names for LaTeX (replace underscores, title case)
        df_for_latex.columns = [col.replace('_', ' ').title() for col in df_for_latex.columns]
        
        if col_format is None:
            col_format = '|' + 'l|' * len(df_for_latex.columns) # Default left-aligned

        latex_content_list.append(df_for_latex.to_latex(index=False, escape=False, column_format=col_format, booktabs=True))
        latex_content_list.append(r"\end{table}")
        latex_content_list.append("\n")
    else:
        latex_content_list.append(f"% Table data for '{label_text}' is empty or None.\n")

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
    all_targets_summary_metrics = [] # To collect data for an overall summary table

    for target_info in config['targets']:
        target_id_name = target_info['id_name']
        target_display_name = target_info['display_name']
        mf_display_name = target_info['molecular_function_display_name']
        
        latex_content.append(f"\\clearpage\n{get_section_header_latex(1, f'Target: {target_display_name} (MF: {mf_display_name})')}")
        target_results_base_dir = os.path.join(args.experiment_run_dir, target_id_name, "results")

        if not os.path.exists(target_results_base_dir):
            latex_content.append(f"Results not found for target {target_id_name}.\n")
            continue

        # --- Section: Dimensionality Optimization ---
        latex_content.append(get_section_header_latex(2, "Optimization of Similarity Space Dimensionality (SIMSPACE\\_DIM)"))
        latex_content.append("The optimal dimensionality for similarity projections was investigated by testing SIMSPACE\\_DIM values of "
                             f"{', '.join(map(str, gs['simspace_dims_to_test']))}. "
                             "The mean of the minimum Euclidean distances from projected target ligands to the molecular function cloud served as the primary metric for this optimization.\n")

        target_dim_opt_summary_rows = []

        for repr_type in config['representations']:
            latex_content.append(get_section_header_latex(3, f"Representation: {repr_type.capitalize()}"))
            
            for dr_key, dr_params in config["dimensionality_reduction_methods"].items():
                dr_display_name = dr_params["short_name"]
                dr_short_name_fs = dr_display_name.replace('-', '_') # Filesystem friendly

                dim_vs_dist_data = []
                for simspace_dim_val in gs['simspace_dims_to_test']:
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
                    
                    plt.figure(figsize=(8, 5))
                    plt.plot(df_plot['dim'], df_plot['mean_min_dist'], marker='o', linestyle='-', label=dr_display_name)
                    plt.xlabel("SIMSPACE\\_DIM Value")
                    plt.ylabel("Mean Min. Distance to MF Cloud")
                    plt.title(f"SIMSPACE\\_DIM vs. Mean Min. Distance\nTarget: {target_display_name} ({repr_type.capitalize()}, {dr_display_name})", fontsize=11)
                    plt.xticks(gs['simspace_dims_to_test'])
                    plt.grid(True, linestyle='--', alpha=0.6)
                    plt.legend()
                    
                    plot_filename = f"{target_id_name}_{repr_type}_{dr_short_name_fs}_dim_vs_mindist.png"
                    plot_abs_path_dest = os.path.join(report_figures_abs_dir, plot_filename)
                    try:
                        plt.savefig(plot_abs_path_dest, dpi=150, bbox_inches='tight')
                        plot_relative_path_for_tex = os.path.join("figures", plot_filename) # Path relative to .tex file
                        caption = f"Mean minimum distance to MF cloud vs. SIMSPACE\\_DIM for {target_display_name} ({repr_type.capitalize()}, DR: {dr_display_name})."
                        add_figure_to_latex(latex_content, plot_relative_path_for_tex, caption, f"{target_id_name}_{repr_type}_{dr_short_name_fs}_dimopt")
                    except Exception as e: logging.error(f"Failed to save dim_opt plot {plot_abs_path_dest}: {e}")
                    plt.close()

                    # Find best dim for summary
                    if not df_plot.empty:
                        best_row = df_plot.loc[df_plot['mean_min_dist'].idxmin()]
                        target_dim_opt_summary_rows.append({
                            'Representation': repr_type.capitalize(), 'DR Method': dr_display_name,
                            'Optimal_SIMSPACE_DIM': int(best_row['dim']), 
                            'Mean_Min_Dist_at_Optimal_DIM': best_row['mean_min_dist']
                        })
            latex_content.append("\\clearpage\n") # After all DR methods for a representation

        if target_dim_opt_summary_rows:
            df_target_dim_opt_summary = pd.DataFrame(target_dim_opt_summary_rows)
            latex_content.append(get_section_header_latex(3, f"Summary of Optimal Dimensionality for {target_display_name}"))
            add_dataframe_as_latex_table(latex_content, df_target_dim_opt_summary,
                                         f"Optimal SIMSPACE\\_DIM and corresponding Mean Minimum Distance for {target_display_name}.",
                                         f"opt_dim_summary_{target_id_name}")
            
            # Add to overall summary
            for row in target_dim_opt_summary_rows:
                all_targets_summary_metrics.append({
                    'Target': target_display_name, **row
                })
        latex_content.append("\\clearpage\n")


        # --- Section: Detailed 2D Projections ---
        latex_content.append(get_section_header_latex(2, "Detailed 2D Projections (SIMSPACE\\_DIM = 2)"))
        dim_2_exists_for_target = False
        for repr_type in config['representations']:
            repr_has_2d_plot = False
            for dr_key, dr_params in config["dimensionality_reduction_methods"].items():
                dr_display_name = dr_params["short_name"]
                dr_short_name_fs = dr_display_name.replace('-', '_')
                
                dim2_results_path = os.path.join(target_results_base_dir, repr_type, "dim_2", dr_short_name_fs)
                if os.path.exists(dim2_results_path):
                    if not repr_has_2d_plot: # Add subsection for representation only if it has 2D plots
                        latex_content.append(get_section_header_latex(3, f"Representation: {repr_type.capitalize()} (2D Projections)"))
                        repr_has_2d_plot = True
                        dim_2_exists_for_target = True

                    latex_content.append(get_section_header_latex(4, f"DR Method: {dr_display_name} (2D)"))
                    
                    scatter_src = os.path.join(dim2_results_path, f"{target_id_name}_{repr_type}_{dr_short_name_fs}_dim2_scatter.png")
                    hist_src = os.path.join(dim2_results_path, f"{target_id_name}_{repr_type}_{dr_short_name_fs}_dim2_min_distances_hist.png")
                    
                    if os.path.exists(scatter_src):
                        scatter_dest_filename = os.path.basename(scatter_src)
                        shutil.copy(scatter_src, os.path.join(report_figures_abs_dir, scatter_dest_filename))
                        caption = f"2D Similarity space for {target_display_name} ({repr_type.capitalize()}, {dr_display_name})."
                        add_figure_to_latex(latex_content, os.path.join("figures", scatter_dest_filename), caption, f"{target_id_name}_{repr_type}_{dr_short_name_fs}_scatter2D")

                    if os.path.exists(hist_src):
                        hist_dest_filename = os.path.basename(hist_src)
                        shutil.copy(hist_src, os.path.join(report_figures_abs_dir, hist_dest_filename))
                        caption = f"Histogram of minimum distances for {target_display_name} ({repr_type.capitalize()}, {dr_display_name}, 2D)."
                        add_figure_to_latex(latex_content, os.path.join("figures", hist_dest_filename), caption, f"{target_id_name}_{repr_type}_{dr_short_name_fs}_hist2D", width="0.7\\textwidth")
            if repr_has_2d_plot: latex_content.append("\\clearpage\n")
        if not dim_2_exists_for_target:
            latex_content.append("No 2D projection results were found for this target.\n")


    # --- Overall Summary Table Section ---
    latex_content.append(f"\\clearpage\n{get_section_header_latex(1, 'Overall Comparative Summary')}")
    if all_targets_summary_metrics:
        df_all_summary = pd.DataFrame(all_targets_summary_metrics)
        # Sort or pivot for better readability
        df_all_summary = df_all_summary[['Target', 'Representation', 'DR Method', 'Optimal_SIMSPACE_DIM', 'Mean_Min_Dist_at_Optimal_DIM']]
        df_all_summary.sort_values(by=['Target', 'Representation', 'Mean_Min_Dist_at_Optimal_DIM'], inplace=True)
        
        latex_content.append("The following table summarizes the optimal SIMSPACE\\_DIM and corresponding mean minimum distance to the MF cloud achieved for each combination of target, representation, and DR method.\n")
        add_dataframe_as_latex_table(latex_content, df_all_summary,
                                     "Overall Summary: Optimal SIMSPACE\\_DIM and Mean Minimum Distances.",
                                     "overall_summary_opt_dims", col_format='|p{3.5cm}|p{2cm}|p{3cm}|c|r|') # Example column format
    else:
        latex_content.append("No summary data collected across targets.\n")

    latex_content.append(LATEX_TEMPLATE_FOOTER)

    report_tex_path = os.path.join(report_output_abs_dir, "experiment_report.tex")
    with open(report_tex_path, "w", encoding='utf-8') as f:
        f.write("\n".join(latex_content))
    
    logging.info(f"LaTeX report generated: {report_tex_path}")
    logging.info(f"Figures for report are in: {report_figures_abs_dir}")

    try:
        logging.info(f"Attempting to compile LaTeX report in: {report_output_abs_dir}")
        # Run pdflatex twice for table of contents, references, etc.
        for _ in range(2):
            process = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-output-directory", report_output_abs_dir, report_tex_path],
                capture_output=True, text=True, check=False # check=False to inspect output even on error
            )
            if process.returncode != 0:
                logging.error(f"pdflatex compilation failed on a pass. Log output below.")
                logging.error("STDOUT:\n" + process.stdout)
                logging.error("STDERR:\n" + process.stderr)
                # Try to find the .log file for more detailed errors
                log_file_path = os.path.join(report_output_abs_dir, "experiment_report.log")
                if os.path.exists(log_file_path):
                    logging.error(f"See {log_file_path} for detailed LaTeX errors.")
                break # Stop trying to compile if one pass fails
        else: # If loop completed without break
             logging.info(f"PDF report compilation attempt finished. Check {report_output_abs_dir} for experiment_report.pdf")

    except FileNotFoundError:
        logging.warning("pdflatex command not found. Please compile the .tex file manually.")
    except Exception as e: # Catch other potential errors during subprocess
        logging.error(f"An unexpected error occurred during pdflatex compilation: {e}")

if __name__ == "__main__":
    main()