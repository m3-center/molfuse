#!/usr/bin/env python3
"""
Analyze Dataset Counts from Actual Experimental Data
===================================================

This script analyzes the actual molecule counts used in UMMBAS experiments
by examining the temp_data files from experimental runs.

The UMMBAS pipeline uses leave-one-target-out virtual screening where:
1. MF Cloud: Compounds from other targets with same molecular function (excluding held-out target)
2. Held-out ligands: Target-specific compounds projected into similarity space for ranking
3. ZINC decoys: Background compounds for similarity space construction

Author: Analysis Script
Date: October 2024
"""

import os
import pandas as pd
import numpy as np
import glob
from pathlib import Path

def analyze_experimental_data():
    """Analyze actual experimental dataset counts from temp_data files."""
    
    print("="*80)
    print("UMMBAS Experimental Dataset Analysis")
    print("Analysis of Actual Experimental Data Files")
    print("="*80)
    
    # Base path to experimental workspace
    base_path = "/home/alex/Documents/_cloud/Funded_Projects/CompChem/UMMBAS/UMMBAS_screening_experiments/experiment_workspace_generalization"
    
    # Find all experimental run directories
    run_dirs = glob.glob(os.path.join(base_path, "run_seed46_config_*"))
    
    results = {}
    total_mf_compounds = 0
    total_target_compounds = 0
    total_zinc_compounds = 0
    
    print(f"Found {len(run_dirs)} experimental runs:")
    
    for run_dir in sorted(run_dirs):
        # Extract target info from directory name
        run_name = os.path.basename(run_dir)
        target_parts = run_name.replace("run_seed46_config_", "").replace("_features_pca_coembedding", "")
        
        # Find target subdirectory
        target_subdirs = glob.glob(os.path.join(run_dir, "*"))
        target_subdirs = [d for d in target_subdirs if os.path.isdir(d)]
        
        if not target_subdirs:
            print(f"  WARNING: No target subdirectory found in {run_name}")
            continue
            
        target_dir = target_subdirs[0]  # Should be only one
        target_name = os.path.basename(target_dir)
        temp_data_dir = os.path.join(target_dir, "temp_data")
        
        if not os.path.exists(temp_data_dir):
            print(f"  WARNING: No temp_data directory found for {target_name}")
            continue
        
        print(f"\n--- {target_name} ---")
        
        # Analyze each file type
        file_counts = {}
        
        # 1. MF Cloud (molecular function compounds excluding target)
        mf_file = os.path.join(temp_data_dir, f"{target_name}_chembl_mf_excluded_features.csv")
        if os.path.exists(mf_file):
            mf_count = sum(1 for line in open(mf_file)) - 1  # Subtract header
            file_counts['mf_cloud'] = mf_count
            total_mf_compounds += mf_count
            print(f"  MF Cloud (excluding {target_name}): {mf_count:,} compounds")
            
            # Check which targets are represented in MF cloud
            try:
                df_mf = pd.read_csv(mf_file, nrows=100)  # Sample to check targets
                if 'Target Name' in df_mf.columns:
                    unique_targets = df_mf['Target Name'].unique()
                    print(f"    MF Cloud targets: {', '.join(unique_targets[:5])}")
                    if len(unique_targets) > 5:
                        print(f"    ... and {len(unique_targets)-5} more")
            except Exception as e:
                print(f"    Could not analyze MF cloud targets: {e}")
        else:
            file_counts['mf_cloud'] = 0
            print(f"  MF Cloud: FILE NOT FOUND")
        
        # 2. Target ligands (held-out compounds)
        target_file = os.path.join(temp_data_dir, f"{target_name}_target_ligands_for_feature_calc_raw.csv")
        if os.path.exists(target_file):
            target_count = sum(1 for line in open(target_file)) - 1  # Subtract header
            file_counts['target_ligands'] = target_count
            total_target_compounds += target_count
            print(f"  Held-out {target_name} ligands: {target_count:,} compounds")
            
            # Check affinity distribution
            try:
                df_target = pd.read_csv(target_file, nrows=1000)  # Sample for analysis
                if 'Standard Value (nM)' in df_target.columns:
                    affinities = pd.to_numeric(df_target['Standard Value (nM)'], errors='coerce')
                    affinities = affinities.dropna()
                    if len(affinities) > 0:
                        print(f"    Affinity range: {affinities.min():.1f} - {affinities.max():.1f} nM")
                        print(f"    Median affinity: {affinities.median():.1f} nM")
                        # Count high-affinity compounds (≤ 100,000 nM)
                        high_affinity = (affinities <= 100000).sum()
                        print(f"    High affinity (≤100,000 nM): {high_affinity:,} compounds ({100*high_affinity/len(affinities):.1f}%)")
            except Exception as e:
                print(f"    Could not analyze target affinity: {e}")
        else:
            file_counts['target_ligands'] = 0
            print(f"  Target ligands: FILE NOT FOUND")
        
        # 3. ZINC decoys
        zinc_file = os.path.join(temp_data_dir, f"{target_name}_zinc_excluded_features.csv")
        if os.path.exists(zinc_file):
            zinc_count = sum(1 for line in open(zinc_file)) - 1  # Subtract header
            file_counts['zinc_decoys'] = zinc_count
            if target_name not in [r['target'] for r in results.values()]:  # Count ZINC only once
                total_zinc_compounds = zinc_count  # ZINC is shared across targets
            print(f"  ZINC decoys: {zinc_count:,} compounds")
        else:
            file_counts['zinc_decoys'] = 0
            print(f"  ZINC decoys: FILE NOT FOUND")
        
        # Calculate ratios
        mf_count = file_counts.get('mf_cloud', 0)
        target_count = file_counts.get('target_ligands', 0)
        
        if mf_count > 0 and target_count > 0:
            ratio = target_count / mf_count
            print(f"  Held-out/MF-Cloud ratio: {ratio:.1f}x")
            if ratio > 10:
                print(f"    ⚠️  WARNING: Held-out set is {ratio:.1f}x larger than MF cloud!")
        
        results[target_name] = {
            'target': target_name,
            'mf_cloud': file_counts.get('mf_cloud', 0),
            'target_ligands': file_counts.get('target_ligands', 0),
            'zinc_decoys': file_counts.get('zinc_decoys', 0),
            'ratio': target_count / mf_count if mf_count > 0 else float('inf')
        }
    
    # Summary statistics
    print("\n" + "="*80)
    print("EXPERIMENTAL DATASET SUMMARY")
    print("="*80)
    
    print(f"Total experimental targets analyzed: {len(results)}")
    print(f"Total unique MF cloud compounds: {total_mf_compounds:,}")
    print(f"Total held-out target ligands: {total_target_compounds:,}")
    print(f"ZINC decoy compounds (shared): {total_zinc_compounds:,}")
    print(f"Grand total experimental compounds: {total_mf_compounds + total_target_compounds + total_zinc_compounds:,}")
    
    # Detailed breakdown by target
    print(f"\nPer-target breakdown:")
    print(f"{'Target':<35} {'MF Cloud':<10} {'Held-out':<10} {'ZINC':<12} {'Ratio':<8}")
    print("-" * 80)
    
    for target_name, data in results.items():
        ratio_str = f"{data['ratio']:.1f}x" if data['ratio'] != float('inf') else "∞"
        print(f"{target_name:<35} {data['mf_cloud']:<10,} {data['target_ligands']:<10,} {data['zinc_decoys']:<12,} {ratio_str:<8}")
    
    # Identify problematic targets
    print(f"\nExperimental Design Analysis:")
    problematic_targets = [name for name, data in results.items() if data['ratio'] > 10]
    if problematic_targets:
        print(f"⚠️  Targets with sparse MF clouds (ratio > 10x):")
        for target in problematic_targets:
            data = results[target]
            print(f"   - {target}: {data['target_ligands']:,} held-out vs {data['mf_cloud']:,} MF cloud ({data['ratio']:.1f}x)")
        print(f"   This may indicate insufficient molecular function diversity for similarity space construction.")
    else:
        print(f"✅ All targets have reasonable MF cloud sizes relative to held-out sets.")
    
    # Generate LaTeX summary
    latex_code, csv_file = generate_latex_summary(results, total_mf_compounds, total_target_compounds, total_zinc_compounds)
    
    return results

def generate_latex_summary(results, total_mf, total_target, total_zinc):
    """Generate LaTeX table and CSV file summarizing the experimental datasets."""
    
    print("\n" + "="*80)
    print("LATEX SUMMARY FOR PUBLICATION")
    print("="*80)
    
    # Create CSV file programmatically
    csv_filename = 'experimental_dataset_summary.csv'
    
    # Prepare data for CSV
    csv_data = []
    for target_name, data in results.items():
        csv_data.append({
            'Target_Protein': target_name,
            'MF_Cloud_Compounds': data['mf_cloud'],
            'Held_out_Ligands': data['target_ligands'],
            'ZINC_Decoys': data['zinc_decoys'],
            'Ratio_H_MF': round(data['ratio'], 1) if data['ratio'] != float('inf') else 'inf'
        })
    
    # Add totals row
    csv_data.append({
        'Target_Protein': 'TOTAL',
        'MF_Cloud_Compounds': total_mf,
        'Held_out_Ligands': total_target,
        'ZINC_Decoys': total_zinc,
        'Ratio_H_MF': '--'
    })
    
    # Save CSV file
    df_summary = pd.DataFrame(csv_data)
    df_summary.to_csv(csv_filename, index=False)
    print(f"✅ CSV data saved to: {csv_filename}")
    
    # Generate LaTeX code that imports the CSV
    latex_code = r"""
% UMMBAS Experimental Dataset Summary
% Generated from actual experimental temp_data files
% Data imported from experimental_dataset_summary.csv

\usepackage{csvsimple}
\usepackage{booktabs}
\usepackage{threeparttable}

\begin{table}[ht]
\centering
\begin{threeparttable}
\caption{Experimental Dataset Composition for UMMBAS Virtual Screening}
\label{tab:ummbas_experimental_datasets}

% Import data from CSV file and format as table
\csvreader[
    tabular=lrrrc,
    table head=\toprule
    \textbf{Target Protein} & \textbf{MF Cloud} & \textbf{Held-out} & \textbf{ZINC} & \textbf{Ratio} \\
                            & \textbf{Compounds} & \textbf{Ligands} & \textbf{Decoys} & \textbf{(H/MF)} \\
    \midrule,
    late after line=\\,
    table foot=\bottomrule,
    before reading={\catcode`\_=12}, % Handle underscores in target names
    filter test={\ifnum\thecsvinputline<4}, % Only show first 3 data rows (excluding totals)
]{experimental_dataset_summary.csv}
{Target_Protein=\targetname, MF_Cloud_Compounds=\mfcloud, Held_out_Ligands=\heldout, ZINC_Decoys=\zinc, Ratio_H_MF=\ratio}
{\targetname & \num{\mfcloud} & \num{\heldout} & \num{\zinc} & \ratio}

% Add totals row manually with better formatting
\midrule
\csvreader[
    late after line=\\,
    filter test={\ifnum\thecsvinputline=4}, % Only show totals row (line 4)
]{experimental_dataset_summary.csv}
{Target_Protein=\targetname, MF_Cloud_Compounds=\mfcloud, Held_out_Ligands=\heldout, ZINC_Decoys=\zinc, Ratio_H_MF=\ratio}
{\textbf{\targetname} & \textbf{\num{\mfcloud}} & \textbf{\num{\heldout}} & \textbf{\num{\zinc}} & \textbf{\ratio}}

\begin{tablenotes}
\small
\item \textbf{MF Cloud}: Compounds binding to proteins with same molecular function as target, excluding target-specific compounds (leave-one-target-out design).
\item \textbf{Held-out Ligands}: Target-specific compounds projected into similarity space for virtual screening evaluation.  
\item \textbf{ZINC Decoys}: Background compounds from ZINC database used for similarity space construction.
\item \textbf{Ratio (H/MF)}: Ratio of held-out ligands to MF cloud size. Values $>10$ indicate sparse molecular function spaces.
\item All compounds filtered to $\leq 100{,}000$ nM binding affinity where applicable.
\item \textbf{Data Source}: Programmatically generated from experimental temp\_data files (""" + csv_filename + r""").
\end{tablenotes}
\end{threeparttable}
\end{table}

% Alternative simpler LaTeX table (if csvsimple is not available)
% Uncomment the following block if you prefer a manually formatted table:
%
% \begin{table}[ht]
% \centering
% \caption{Experimental Dataset Composition for UMMBAS Virtual Screening}
% \label{tab:ummbas_experimental_datasets_simple}
% \begin{tabular}{lrrrc}
% \toprule
% \textbf{Target Protein} & \textbf{MF Cloud} & \textbf{Held-out} & \textbf{ZINC} & \textbf{Ratio} \\
%                         & \textbf{Compounds} & \textbf{Ligands} & \textbf{Decoys} & \textbf{(H/MF)} \\
% \midrule
"""
    
    # Add manual table rows as backup
    for target_name, data in results.items():
        clean_name = target_name.replace('_', '\\_')
        ratio_str = f"{data['ratio']:.1f}" if data['ratio'] != float('inf') else "$\\infty$"
        latex_code += f"% {clean_name} & \\num{{{data['mf_cloud']}}} & \\num{{{data['target_ligands']}}} & \\num{{{data['zinc_decoys']}}} & {ratio_str} \\\\\n"
    
    latex_code += f"% \\midrule\n"
    latex_code += f"% \\textbf{{Total}} & \\textbf{{\\num{{{total_mf}}}}} & \\textbf{{\\num{{{total_target}}}}} & \\textbf{{\\num{{{total_zinc}}}}} & \\textbf{{--}} \\\\\n"
    latex_code += f"% \\bottomrule\n% \\end{{tabular}}\n% \\end{{table}}\n\n"
    
    # Add summary paragraph
    latex_code += f"""
% Summary paragraph for methods section
The experimental evaluation employed a leave-one-target-out virtual screening protocol across {len(results)} protein targets. 
Molecular function (MF) clouds containing {total_mf:,} compounds from functionally related proteins provided 
the similarity space foundation, while {total_target:,} target-specific ligands served as held-out compounds 
for screening evaluation. Additionally, {total_zinc:,} ZINC database decoys augmented the chemical space. 
The complete experimental dataset comprised {total_mf + total_target + total_zinc:,} unique molecular entities, 
with all bioactive compounds filtered to binding affinities $\\leq 100{{,}}000$ nM.
"""
    
    print(latex_code)
    
    # Save LaTeX to file
    with open('experimental_dataset_summary.tex', 'w') as f:
        f.write(latex_code)
    
    print(f"\n✅ LaTeX code saved to: experimental_dataset_summary.tex")
    print(f"✅ Table data saved to: {csv_filename}")
    print(f"\n📋 Usage Instructions:")
    print(f"   1. Include both files in your LaTeX project directory")
    print(f"   2. Add \\usepackage{{csvsimple}} and \\usepackage{{booktabs}} to your preamble")
    print(f"   3. Use \\input{{experimental_dataset_summary.tex}} in your document")
    print(f"   4. If csvsimple is unavailable, uncomment the simple table version")
    
    return latex_code, csv_filename

if __name__ == "__main__":
    try:
        results = analyze_experimental_data()
        print(f"\n🎯 Analysis complete! Found experimental data for {len(results)} targets.")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()