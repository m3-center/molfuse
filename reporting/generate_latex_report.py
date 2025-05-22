import pandas as pd
import numpy as np
import os
import argparse
import logging
import json
import glob
import shutil
from collections import defaultdict
import matplotlib.pyplot as plt
import subprocess

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

LATEX_TEMPLATE_HEADER = r"""
\documentclass[10pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{float} % For [H] placement
\usepackage{amsmath}
\usepackage{booktabs} % For nice tables
\usepackage{longtable}
\usepackage{caption}
\usepackage{subcaption} % For subfigures
\usepackage{array} % For better column control in tables
\usepackage{hyperref} % For clickable links if needed
\usepackage{xcolor} % For colored text if desired

\hypersetup{
    colorlinks=true,
    linkcolor=blue,
    filecolor=magenta,      
    urlcolor=cyan,
    pdftitle={UMMBAS Similarity Experiment Report},
    pdfpagemode=FullScreen,
}

\title{UMMBAS Molecular Similarity Experimental Evaluation}
\author{UMMBAS Project Team}
\date{\today}

\begin{document}
\maketitle
\tableofcontents
\clearpage

\section{Introduction}
This report details experiments conducted to evaluate the predictive capacity of molecular similarity spaces using a leave-one-target-out cross-validation inspired approach. The core hypothesis is that ligands active against a specific protein target will cluster near compounds known to interact with other proteins sharing the same broader molecular function, even when the specific target's known ligands are excluded during space construction. 

Experiments were performed for multiple target proteins, utilizing two distinct molecular representations (physicochemical features and ECFP4 fingerprints), various dimensionality reduction (DR) techniques (PCA, UMAP with Euclidean, Cosine, Manhattan, and Hamming metrics), and a range of similarity space dimensionalities. The primary evaluation metric is the distance of projected known target ligands to the molecular function (MF) cloud in the generated similarity spaces.

"""

LATEX_TEMPLATE_FOOTER = r"""
\end{document}
"""

def get_section_header(level, title):
    sec_cmd = ""
    if level == 1: sec_cmd = r"\section"
    elif level == 2: sec_cmd = r"\subsection"
    elif level == 3: sec_cmd = r"\subsubsection"
    elif level == 4: sec_cmd = r"\paragraph"
    else: return f"% Unknown section level {level} for: {title}\n"
    return f"{sec_cmd}{{{title}}}\n"


def add_figure(latex_content, fig_path, caption, label, width="0.8\\textwidth"):
    if os.path.exists(fig_path):
        latex_content.append(r"\begin{figure}[H]")
        latex_content.append(r"  \centering")
        latex_content.append(f"  \\includegraphics[width={width}]{{{{{fig_path}}}}}") # Double {{}} for f-string
        latex_content.append(f"  \\caption{{{caption}}}")
        latex_content.append(f"  \\label{{fig:{label}}}")
        latex_content.append(r"\end{figure}")
        latex_content.append("\n")
    else:
        latex_content.append(f"% Figure not found: {fig_path}\n")

def add_table_from_df(latex_content, df, caption, label):
    if df is not None and not df.empty:
        latex_content.append(r"\begin{table}[H]")
        latex_content.append(r"  \centering")
        latex_content.append(f"  \\caption{{{caption}}}")
        latex_content.append(f"  \\label{{tab:{label}}}")
        # Format numbers in DataFrame for LaTeX output
        # Ensure column names are LaTeX-friendly (no underscores, etc.)
        df_latex = df.copy()
        for col in df_latex.select_dtypes(include=np.number).columns:
            df_latex[col] = df_latex[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
        
        df_latex.columns = [col.replace('_', ' ').title() for col in df_latex.columns]
        
        latex_content.append(df_latex.to_latex(index=False, escape=False, column_format='|' + 'l|'*len(df.columns)))
        latex_content.append(r"\end{table}")
        latex_content.append("\n")
    else:
        latex_content.append(f"% Table data not available for: {label}\n")


def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX report from experiment results.")
    parser.add_argument("--experiment_run_dir", required=True, help="Path to the timestamped main experiment run directory.")
    parser.add_argument("--config_path", required=True, help="Path to the experiment_config.json file.")
    parser.add_argument("--output_dir", required=True, help="Directory to save the LaTeX report and figures.")
    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        config = json.load(f)
    
    gs = config['global_settings']
    
    report_output_dir = os.path.join(args.output_dir, os.path.basename(args.experiment_run_dir) + "_report")
    report_figures_dir = os.path.join(report_output_dir, "figures")
    os.makedirs(report_figures_dir, exist_ok=True)
    
    latex_content = [LATEX_TEMPLATE_HEADER]

    # --- Overall Summary Section ---
    latex_content.append(get_section_header(1, "Overall Summary and Key Findings"))
    # Placeholder for overall summary table/plots later
    all_target_summary_data = []


    # --- Loop through each target ---
    for target_info in config['targets']:
        target_id_name = target_info['id_name']
        target_display_name = target_info['display_name']
        mf_display_name = target_info['molecular_function_display_name']
        
        latex_content.append(f"\\clearpage\n{get_section_header(1, f'Target: {target_display_name} (MF: {mf_display_name})')}")
        
        target_results_base = os.path.join(args.experiment_run_dir, target_id_name, "results")
        if not os.path.exists(target_results_base):
            latex_content.append(f"Results not found for target {target_id_name}.\n")
            continue

        # Store data for optimizing SIMSPACE_DIM
        dim_opt_data = defaultdict(lambda: defaultdict(list)) # repr -> dr_method -> list of (dim, mean_min_dist)

        # --- Loop through representations ---
        for repr_type in config['representations']:
            latex_content.append(get_section_header(2, f"Data Representation: {repr_type.capitalize()}"))
            
            # --- Loop through DR methods ---
            for dr_key, dr_params in config["dimensionality_reduction_methods"].items():
                dr_short_name_fs = dr_params["short_name"].replace('-', '_') # Filesystem friendly
                dr_display_name = dr_params["short_name"]

                # latex_content.append(get_section_header(3, f"DR Method: {dr_display_name}"))

                # Collect data for dimensionality optimization plot for this repr/DR
                current_dr_dim_data = []

                # --- Loop through SIMSPACE_DIM values ---
                for simspace_dim in gs['simspace_dims_to_test']:
                    results_path = os.path.join(target_results_base, repr_type, f"dim_{simspace_dim}", dr_short_name_fs)
                    distances_csv_path = os.path.join(results_path, f"{target_id_name}_{repr_type}_{dr_short_name_fs}_dim{simspace_dim}_distances.csv")
                    
                    if os.path.exists(distances_csv_path):
                        try:
                            df_distances = pd.read_csv(distances_csv_path)
                            if 'min_dist_to_mf_cloud' in df_distances.columns and not df_distances['min_dist_to_mf_cloud'].empty:
                                mean_min_dist = df_distances['min_dist_to_mf_cloud'].mean()
                                current_dr_dim_data.append({'dim': simspace_dim, 'mean_min_dist': mean_min_dist})
                                dim_opt_data[repr_type][dr_display_name].append((simspace_dim, mean_min_dist))
                        except Exception as e:
                            logging.warning(f"Could not read or process distances CSV: {distances_csv_path} - {e}")
                
                # Plot for SIMSPACE_DIM optimization for current repr/DR method
                if current_dr_dim_data:
                    df_plot_dim_opt = pd.DataFrame(current_dr_dim_data)
                    df_plot_dim_opt.sort_values(by='dim', inplace=True)
                    
                    plt.figure(figsize=(7, 4))
                    plt.plot(df_plot_dim_opt['dim'], df_plot_dim_opt['mean_min_dist'], marker='o', linestyle='-')
                    plt.xlabel("Similarity Space Dimensionality (SIMSPACE_DIM)")
                    plt.ylabel("Mean Min. Distance to MF Cloud")
                    plt.title(f"Optimizing SIMSPACE_DIM for {target_display_name}\n({repr_type.capitalize()}, {dr_display_name})")
                    plt.xticks(gs['simspace_dims_to_test'])
                    plt.grid(True, which="both", ls="-", alpha=0.5)
                    
                    dim_opt_plot_filename = f"{target_id_name}_{repr_type}_{dr_short_name_fs}_dim_vs_mindist.png"
                    dim_opt_plot_path_src = os.path.join(results_path, "..", dim_opt_plot_filename) # Save one level up from specific dim folder
                    dim_opt_plot_path_dest = os.path.join(report_figures_dir, dim_opt_plot_filename)
                    plt.savefig(dim_opt_plot_path_src) # Save in results dir too
                    plt.close()
                    
                    shutil.copy(dim_opt_plot_path_src, dim_opt_plot_path_dest)
                    caption = f"Mean minimum distance vs. SIMSPACE\\_DIM for target {target_display_name} ({repr_type.capitalize()}, DR: {dr_display_name})."
                    add_figure(latex_content, dim_opt_plot_path_dest, caption, f"{target_id_name}_{repr_type}_{dr_short_name_fs}_dimopt", width="0.7\\textwidth")

            latex_content.append("\\clearpage\n")
        
        # --- After iterating all DRs for a representation, find "optimal" dim ---
        # For simplicity, choose dim that gives lowest mean_min_dist across all DRs for that repr, or report per DR
        # This part needs more sophisticated logic to choose a single "optimal" or discuss variability.
        # For now, we will proceed to show 2D plots if they exist.

        latex_content.append(get_section_header(2, f"Detailed 2D Projections (if available)"))
        # Loop again to add 2D scatter and histograms
        for repr_type in config['representations']:
            latex_content.append(get_section_header(3, f"2D Plots for Representation: {repr_type.capitalize()}"))
            for dr_key, dr_params in config["dimensionality_reduction_methods"].items():
                dr_short_name_fs = dr_params["short_name"].replace('-', '_')
                dr_display_name = dr_params["short_name"]
                
                dim2_results_path = os.path.join(target_results_base, repr_type, "dim_2", dr_short_name_fs)
                if os.path.exists(dim2_results_path):
                    latex_content.append(get_section_header(4, f"DR Method: {dr_display_name} (2D)"))

                    scatter_plot_src = os.path.join(dim2_results_path, f"{target_id_name}_{repr_type}_{dr_short_name_fs}_dim2_scatter.png")
                    hist_plot_src = os.path.join(dim2_results_path, f"{target_id_name}_{repr_type}_{dr_short_name_fs}_dim2_min_distances_hist.png")
                    
                    scatter_plot_dest = os.path.join(report_figures_dir, os.path.basename(scatter_plot_src))
                    hist_plot_dest = os.path.join(report_figures_dir, os.path.basename(hist_plot_src))

                    if os.path.exists(scatter_plot_src):
                        shutil.copy(scatter_plot_src, scatter_plot_dest)
                        caption_scatter = f"2D Similarity space for {target_display_name} ({repr_type.capitalize()}, {dr_display_name}). Projected target ligands (red 'x')."
                        add_figure(latex_content, scatter_plot_dest, caption_scatter, f"{target_id_name}_{repr_type}_{dr_short_name_fs}_scatter2D")
                    
                    if os.path.exists(hist_plot_src):
                        shutil.copy(hist_plot_src, hist_plot_dest)
                        # Try to get mean/median from distances.csv for caption
                        distances_csv_path = os.path.join(dim2_results_path, f"{target_id_name}_{repr_type}_{dr_short_name_fs}_dim2_distances.csv")
                        dist_stats_caption = ""
                        if os.path.exists(distances_csv_path):
                            try:
                                df_dist = pd.read_csv(distances_csv_path)
                                if 'min_dist_to_mf_cloud' in df_dist:
                                    mean_d = df_dist['min_dist_to_mf_cloud'].mean()
                                    median_d = df_dist['min_dist_to_mf_cloud'].median()
                                    dist_stats_caption = f" Mean min dist: {mean_d:.2f}, Median min dist: {median_d:.2f}."
                            except: pass
                        caption_hist = f"Histogram of minimum distances for {target_display_name} ({repr_type.capitalize()}, {dr_display_name}, 2D).{dist_stats_caption}"
                        add_figure(latex_content, hist_plot_dest, caption_hist, f"{target_id_name}_{repr_type}_{dr_short_name_fs}_hist2D", width="0.7\\textwidth")
                    latex_content.append("\\clearpage\n")

        # --- Summary table for the current target ---
        target_summary_rows = []
        for repr_type in config['representations']:
            for dr_key, dr_params in config["dimensionality_reduction_methods"].items():
                dr_short_name_fs = dr_params["short_name"].replace('-', '_')
                dr_display_name = dr_params["short_name"]
                
                # Find the optimal dimension for this combo from dim_opt_data
                best_dim_for_combo = 'N/A'
                best_dist_for_combo = np.inf
                if dim_opt_data[repr_type][dr_display_name]:
                    sorted_dims = sorted(dim_opt_data[repr_type][dr_display_name], key=lambda x: x[1]) # Sort by mean_min_dist
                    if sorted_dims:
                        best_dim_for_combo = sorted_dims[0][0]
                        best_dist_for_combo = sorted_dims[0][1]

                # Now load the distances for this best_dim
                mean_min_dist_at_best_dim = best_dist_for_combo if best_dim_for_combo != 'N/A' else np.nan
                # Add other metrics if desired (kNN, centroid) by loading the specific distances.csv
                
                target_summary_rows.append({
                    'Representation': repr_type.capitalize(),
                    'DR Method': dr_display_name,
                    'Optimal DIM': best_dim_for_combo,
                    'Mean Min Dist @ Opt DIM': mean_min_dist_at_best_dim
                })
                all_target_summary_data.append({ # For overall summary
                    'Target': target_display_name,
                    'Representation': repr_type.capitalize(),
                    'DR Method': dr_display_name,
                    'Optimal DIM': best_dim_for_combo,
                    'Mean Min Dist @ Opt DIM': mean_min_dist_at_best_dim
                })


        if target_summary_rows:
            df_target_summary = pd.DataFrame(target_summary_rows)
            latex_content.append(get_section_header(2, f"Summary of Optimal Dimensionality for {target_display_name}"))
            add_table_from_df(latex_content, df_target_summary, 
                              f"Optimal SIMSPACE\\_DIM and corresponding Mean Minimum Distance to MF Cloud for target {target_display_name}.",
                              f"summary_opt_dim_{target_id_name}")
        latex_content.append("\\clearpage\n")


    # --- Add Overall Summary Section Content ---
    if all_target_summary_data:
        df_overall_summary = pd.DataFrame(all_target_summary_data)
        # This section in the LaTeX needs to be moved or populated here.
        # For now, just ensure the DataFrame is created.
        # It would be good to pivot this table or create comparative plots.
        logging.info("Overall summary table data collected. Needs formatting in LaTeX.")
        # Example: Table of best DR/Repr per target.
        # Example: Average performance of each DR method across targets.

    latex_content.append(LATEX_TEMPLATE_FOOTER)

    # Write to .tex file
    report_tex_path = os.path.join(report_output_dir, "experiment_report.tex")
    with open(report_tex_path, "w") as f:
        f.write("\n".join(latex_content))
    
    logging.info(f"LaTeX report generated: {report_tex_path}")
    logging.info(f"Figures copied to: {report_figures_dir}")
    logging.info("To compile: `pdflatex experiment_report.tex` (may need multiple runs)")

    # Attempt to compile LaTeX (optional, requires pdflatex in PATH)
    try:
        subprocess.run(["pdflatex", "-output-directory", report_output_dir, report_tex_path], check=True, capture_output=True, text=True)
        subprocess.run(["pdflatex", "-output-directory", report_output_dir, report_tex_path], check=True, capture_output=True, text=True) # Run twice for TOC/refs
        logging.info(f"PDF report compiled successfully in {report_output_dir}")
    except FileNotFoundError:
        logging.warning("pdflatex command not found. Please compile the .tex file manually.")
    except subprocess.CalledProcessError as e:
        logging.error(f"pdflatex compilation failed. Check experiment_report.log in {report_output_dir} for details.")
        logging.error(f"STDOUT: {e.stdout}")
        logging.error(f"STDERR: {e.stderr}")


if __name__ == "__main__":
    main()