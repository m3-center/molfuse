#!/usr/bin/env python3
"""
Run affinity cutoff analysis on the GENERALIZATION experiments.

This script takes the completed generalization experiment results and re-analyzes
them with different affinity cutoffs to test if the cutoff-performance relationship
(observed in ABL1) generalizes to other proteins.

Scientific Question: Does the negative correlation between affinity cutoff and 
EF@1% (lower cutoff → higher EF@1%) also hold for PyruvateKinaseM2 and 
IsocitrateDehydrogenase?
"""

import os
import glob
import subprocess
import argparse
import logging
import re
import json
from datetime import datetime

# --- Logging Setup ---
log_file_name = f"cutoff_analysis_generalization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)-8s - %(filename)-25s - %(lineno)-4d - %(message)s',
    handlers=[
        logging.FileHandler(log_file_name, mode='w'),
        logging.StreamHandler()
    ]
)

def run_command(command_list, step_name="Command", cwd=None):
    """Execute a shell command and log the output."""
    logging.info(f"Running {step_name}: {' '.join(command_list)}")
    try:
        process = subprocess.run(
            command_list, capture_output=True, text=True, check=True, cwd=cwd
        )
        if process.stdout:
            logging.info(f"[{step_name} STDOUT]:\n{process.stdout.strip()}")
        if process.stderr:
            logging.warning(f"[{step_name} STDERR]:\n{process.stderr.strip()}")
        logging.info(f"{step_name} completed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"{step_name} FAILED with return code {e.returncode}.")
        logging.error(f"STDOUT:\n{e.stdout}")
        logging.error(f"STDERR:\n{e.stderr}")
        return False
    except Exception as e:
        logging.error(f"An unexpected exception occurred during {step_name}: {e}", exc_info=True)
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Run affinity cutoff analysis on generalization experiment results."
    )
    parser.add_argument(
        "--original_workspace",
        default="experiment_workspace_generalization/",
        help="Directory containing the completed generalization experiment runs."
    )
    parser.add_argument(
        "--output_workspace",
        default="experiment_workspace_generalization_cutoff/",
        help="Directory to save the cutoff analysis results."
    )
    parser.add_argument(
        "--config_path",
        default="experiment_config.json",
        help="Path to the main experiment config file for context."
    )
    parser.add_argument(
        "--cutoffs",
        default="100,1000,10000,100000",
        help="Comma-separated list of affinity cutoffs in nM."
    )
    args = parser.parse_args()

    logging.info("=" * 80)
    logging.info("STARTING AFFINITY CUTOFF ANALYSIS ON GENERALIZATION EXPERIMENTS")
    logging.info("=" * 80)

    try:
        with open(args.config_path, 'r') as f:
            main_config = json.load(f)
    except Exception as e:
        logging.error(f"Could not load main config file '{args.config_path}'. Error: {e}")
        logging.warning("Continuing without main config - will use defaults.")
        main_config = {
            'global_settings': {
                'k_for_knn_distance': [3, 5],
                'rdkit_features_list_target': []
            }
        }

    affinity_cutoffs = [int(c.strip()) for c in args.cutoffs.split(',')]
    logging.info(f"Will re-analyze using affinity cutoffs (nM): {affinity_cutoffs}")

    # Only process features (not fingerprints) and 2D (not other dimensions)
    REPR_TYPE_TO_PROCESS = "features"
    SIMSPACE_DIM_TO_PROCESS = 2

    # Find all generalization run directories
    # Pattern: run_seed*_config_TARGETNAME_features_METHOD
    original_run_dirs = glob.glob(
        os.path.join(args.original_workspace, f"run_seed*_config_*_features_*")
    )
    
    if not original_run_dirs:
        logging.error(
            f"No generalization runs found in '{args.original_workspace}'. "
            f"Pattern: run_seed*_config_*_features_*"
        )
        return

    logging.info(f"Found {len(original_run_dirs)} generalization run directories")

    for original_run_dir in original_run_dirs:
        run_basename = os.path.basename(original_run_dir)
        logging.info(f"\n{'='*80}")
        logging.info(f"Processing: {run_basename}")
        logging.info(f"{'='*80}")
        
        # Check if already processed
        output_run_dir = os.path.join(args.output_workspace, run_basename)
        if os.path.exists(output_run_dir):
            logging.info(f"  --> Output directory '{output_run_dir}' already exists. Skipping.")
            continue
        
        # Load run config
        run_config_path = os.path.join(original_run_dir, "run_config.json")
        if not os.path.exists(run_config_path):
            logging.warning(f"run_config.json not found in {original_run_dir}, skipping.")
            continue
        
        with open(run_config_path, 'r') as f:
            run_config = json.load(f)

        # Extract target information
        target_info = run_config['targets'][0]
        target_id_name = target_info['id_name']
        
        # Extract DR method information
        dr_method_key = list(run_config['dimensionality_reduction_methods'].keys())[0]
        dr_params = run_config['dimensionality_reduction_methods'][dr_method_key]
        
        logging.info(f"  Target: {target_info['display_name']}")
        logging.info(f"  DR Method: {dr_params['short_name']}")
        
        # Determine which strategies to run based on method and available results
        strategies_to_run = []
        
        if dr_method_key == 'tsne':
            # t-SNE is ONLY a co-embedding strategy
            strategies_to_run.append("Co-embedding (Native)")
        else:
            # For PCA and UMAP, check which result directories exist
            results_base = os.path.join(
                original_run_dir, target_id_name, "results",
                REPR_TYPE_TO_PROCESS, f"dim_{SIMSPACE_DIM_TO_PROCESS}"
            )
            
            # Check for projection results
            projection_dir = os.path.join(
                results_base, dr_params['short_name'].replace('-', '_')
            )
            if os.path.exists(projection_dir):
                strategies_to_run.append("Projection")
            
            # Check for co-embedding results
            coembed_dir = os.path.join(
                results_base, f"{dr_params['short_name'].replace('-', '_')}_Coembed"
            )
            if os.path.exists(coembed_dir):
                strategies_to_run.append("Co-embedding")
        
        if not strategies_to_run:
            logging.warning(f"  No valid strategies found for {run_basename}, skipping.")
            continue
        
        logging.info(f"  Strategies to process: {strategies_to_run}")

        # Process each strategy with each cutoff
        for strategy in strategies_to_run:
            for cutoff in affinity_cutoffs:
                logging.info(f"\n  ▶ Strategy: {strategy} | Cutoff: {cutoff} nM")

                # Build command
                cmd = ["python", "experimental_pipeline/project_and_analyze.py"]
                
                # Determine paths based on strategy
                if strategy == "Projection":
                    output_dir_leaf = dr_params['short_name'].replace('-', '_')
                    
                    simspace_path = os.path.join(
                        original_run_dir, target_id_name, "similarity_spaces",
                        REPR_TYPE_TO_PROCESS, f"dim_{SIMSPACE_DIM_TO_PROCESS}",
                        f"{target_id_name}_{REPR_TYPE_TO_PROCESS}_dim{SIMSPACE_DIM_TO_PROCESS}_similarity_space.csv"
                    )
                    
                    model_dir = os.path.join(
                        original_run_dir, target_id_name, "models",
                        REPR_TYPE_TO_PROCESS, f"dim_{SIMSPACE_DIM_TO_PROCESS}"
                    )
                    
                    model_name_root = f"{target_id_name}_{REPR_TYPE_TO_PROCESS}_dim{SIMSPACE_DIM_TO_PROCESS}"
                    
                    target_ligands_path = os.path.join(
                        original_run_dir, target_id_name, "target_ligands_calculated",
                        REPR_TYPE_TO_PROCESS,
                        f"{target_id_name}_target_ligands_for_calc_{REPR_TYPE_TO_PROCESS}.csv"
                    )
                    
                    cmd.extend([
                        "--simspace_csv_path", os.path.abspath(simspace_path),
                        "--target_ligands_repr_path", os.path.abspath(target_ligands_path),
                        "--model_dir_for_projection", os.path.abspath(model_dir),
                        "--model_name_root_for_projection", model_name_root
                    ])

                elif strategy in ["Co-embedding", "Co-embedding (Native)"]:
                    output_dir_leaf = f"{dr_params['short_name'].replace('-', '_')}_Coembed"
                    if strategy == "Co-embedding (Native)":
                        output_dir_leaf = dr_params['short_name'].replace('-', '_')
                    
                    # Determine co-embedding filename
                    metric = dr_params.get("metric", "")
                    
                    if dr_method_key.startswith("umap"):
                        coembed_filename = f"{target_id_name}_{REPR_TYPE_TO_PROCESS}_dim{SIMSPACE_DIM_TO_PROCESS}_{metric}_UMAP_similarity_space_COEMBED.csv"
                    elif dr_method_key == "tsne":
                        coembed_filename = f"{target_id_name}_{REPR_TYPE_TO_PROCESS}_dim{SIMSPACE_DIM_TO_PROCESS}_tSNE_similarity_space_COEMBED.csv"
                    else:  # PCA
                        coembed_filename = f"{target_id_name}_{REPR_TYPE_TO_PROCESS}_dim{SIMSPACE_DIM_TO_PROCESS}_{dr_params['short_name'].replace('-', '_')}_similarity_space_COEMBED.csv"
                    
                    simspace_path = os.path.join(
                        original_run_dir, target_id_name, "similarity_spaces",
                        REPR_TYPE_TO_PROCESS, f"dim_{SIMSPACE_DIM_TO_PROCESS}",
                        coembed_filename
                    )
                    
                    cmd.extend(["--simspace_csv_path", os.path.abspath(simspace_path)])

                # Check if required files exist
                if not os.path.exists(simspace_path):
                    logging.warning(f"    Required simspace file not found: {simspace_path}")
                    continue

                # Define output directory
                new_output_dir = os.path.join(
                    args.output_workspace, run_basename,
                    f"results_cutoff_{cutoff}", target_id_name, "results",
                    REPR_TYPE_TO_PROCESS, f"dim_{SIMSPACE_DIM_TO_PROCESS}",
                    output_dir_leaf
                )
                os.makedirs(new_output_dir, exist_ok=True)

                # Add common parameters
                cmd.extend([
                    "--dr_method_key", dr_method_key,
                    "--dr_short_name", dr_params["short_name"],
                    "--simspace_dim", str(SIMSPACE_DIM_TO_PROCESS),
                    "--k_for_knn", ",".join(map(str, main_config['global_settings']['k_for_knn_distance'])),
                    "--output_dir", os.path.abspath(new_output_dir),
                    "--target_id_name", target_id_name,
                    "--representation_type", REPR_TYPE_TO_PROCESS,
                    "--rdkit_features_list_target_str", json.dumps(main_config['global_settings'].get('rdkit_features_list_target', [])),
                    f"--affinity_cutoff={cutoff}"
                ])

                # Run the analysis
                success = run_command(
                    cmd,
                    f"Analyze-{target_id_name}-{strategy}-cutoff{cutoff}"
                )
                
                if not success:
                    logging.warning(f"    Analysis failed for cutoff {cutoff}")

    logging.info("\n" + "=" * 80)
    logging.info("AFFINITY CUTOFF ANALYSIS ON GENERALIZATION EXPERIMENTS COMPLETE")
    logging.info("=" * 80)
    logging.info(f"\nResults saved to: {args.output_workspace}")
    logging.info(f"Log file: {log_file_name}")

if __name__ == "__main__":
    main()
