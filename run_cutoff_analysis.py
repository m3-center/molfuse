import os
import glob
import subprocess
import argparse
import logging
import re
import json
from datetime import datetime

# --- Logging Setup ---
log_file_name = f"cutoff_analysis_orchestrator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)-8s - %(filename)-25s - %(lineno)-4d - %(message)s',
                    handlers=[logging.FileHandler(log_file_name, mode='w'), logging.StreamHandler()])

# --- Helper Function to Run Commands ---
def run_command(command_list, step_name="Command", cwd=None):
    """Runs a command, logs its output, and checks for errors."""
    logging.info(f"Running {step_name}: {' '.join(command_list)}")
    try:
        process = subprocess.run(
            command_list,
            capture_output=True,
            text=True,
            check=True, # Raise exception on non-zero return code
            cwd=cwd
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

# --- Main Orchestration Logic ---
def main():
    parser = argparse.ArgumentParser(description="Orchestrator for the affinity cutoff analysis experiment.")
    parser.add_argument("--original_workspace", default="experiment_workspace/", help="Directory containing the original experiment runs.")
    parser.add_argument("--output_workspace", default="experiment_workspace_cutoff_analysis/", help="Directory to save the new cutoff analysis results.")
    parser.add_argument("--config_path", default="experiment_config.json", help="Path to the main experiment config file.")
    parser.add_argument("--cutoffs", default="1000,10000,100000", help="Comma-separated list of affinity cutoffs in nM.")
    args = parser.parse_args()

    logging.info("--- STARTING AFFINITY CUTOFF RE-ANALYSIS ---")

    try:
        with open(args.config_path, 'r') as f:
            config = json.load(f)
    except Exception as e:
        logging.error(f"Could not load config file '{args.config_path}'. Aborting. Error: {e}")
        return

    affinity_cutoffs = [int(c.strip()) for c in args.cutoffs.split(',')]
    logging.info(f"Will re-analyze using affinity cutoffs (nM): {affinity_cutoffs}")

    # Find original experiment run directories
    original_run_dirs = glob.glob(os.path.join(args.original_workspace, "run_seed*"))
    if not original_run_dirs:
        logging.error(f"No original experiment runs found in '{args.original_workspace}'. Aborting.")
        return

    for original_run_dir in original_run_dirs:
        run_basename = os.path.basename(original_run_dir)
        logging.info(f"\n===== Processing Original Run: {run_basename} =====")

        for target_info in config['targets']:
            target_id_name = target_info['id_name']
            if target_info.get("processing_mode") == "similarity_space_only":
                logging.info(f"Skipping target '{target_id_name}' as it was run in 'similarity_space_only' mode.")
                continue

            logging.info(f"  --- Target: {target_info['display_name']} ---")

            for repr_type in config['representations']:
                logging.info(f"    -- Representation: {repr_type} --")

                for simspace_dim in config['global_settings']['simspace_dims_to_test']:
                    logging.info(f"      - Dimension: {simspace_dim} -")

                    # Path to the key input file from the original experiment
                    simspace_csv_path = os.path.join(original_run_dir, target_id_name, "similarity_spaces", repr_type, f"dim_{simspace_dim}", f"{target_id_name}_{repr_type}_dim{simspace_dim}_similarity_space.csv")

                    if not os.path.exists(simspace_csv_path):
                        logging.warning(f"Simspace file not found, skipping: {simspace_csv_path}")
                        continue

                    # Loop through all DR methods and strategies to re-analyze
                    for dr_key, dr_params in config["dimensionality_reduction_methods"].items():
                        strategies_to_run = [{"is_coembed": False, "short_name": dr_params["short_name"]}]
                        if config['global_settings'].get("run_coembedding_for_pca_umap") and dr_params.get("allow_coembedding") and dr_key != "tsne":
                            strategies_to_run.append({"is_coembed": True, "short_name": f"{dr_params['short_name']}-Coembed"})

                        for strategy in strategies_to_run:
                            # Define the directory leaf from the original experiment
                            original_dir_leaf = strategy['short_name'].replace('-', '_')
                            # Check if this strategy was actually run
                            original_results_dir = os.path.join(original_run_dir, target_id_name, "results", repr_type, f"dim_{simspace_dim}", original_dir_leaf)
                            if not os.path.exists(original_results_dir):
                                continue

                            logging.info(f"        * Re-analyzing DR Method: {strategy['short_name']} *")

                            # Loop through the cutoffs and run the analysis
                            for cutoff in affinity_cutoffs:
                                logging.info(f"          > Applying Cutoff: {cutoff} nM")

                                # Construct the NEW output directory
                                new_output_dir = os.path.join(args.output_workspace, run_basename, f"results_cutoff_{cutoff}", target_id_name, "results", repr_type, f"dim_{simspace_dim}", original_dir_leaf)
                                os.makedirs(new_output_dir, exist_ok=True)

                                # --- Construct the command for project_and_analyze.py ---
                                cmd = [
                                    "python", "experimental_pipeline/project_and_analyze.py",
                                    "--simspace_csv_path", os.path.abspath(simspace_csv_path),
                                    "--dr_method_key", dr_key,
                                    "--dr_short_name", dr_params["short_name"], # Use the BASE name for coord lookup
                                    "--simspace_dim", str(simspace_dim),
                                    "--k_for_knn", ",".join(map(str, config['global_settings']['k_for_knn_distance'])),
                                    "--output_dir", os.path.abspath(new_output_dir),
                                    "--target_id_name", target_id_name,
                                    "--representation_type", repr_type,
                                    "--rdkit_features_list_target_str", json.dumps(config['global_settings'].get('rdkit_features_list_target', [])),
                                    f"--affinity_cutoff={cutoff}" # THE NEW ARGUMENT
                                ]
                                
                                # Add paths to precomputed projections if it's a co-embedding run
                                precomputed_proj_path = "None"
                                if dr_key == "tsne":
                                    precomputed_proj_path = os.path.join(original_run_dir, target_id_name, "similarity_spaces", repr_type, f"dim_{simspace_dim}", f"{target_id_name}_{repr_type}_dim{simspace_dim}_tSNE_TARGET_PROJECTIONS.csv")
                                elif strategy["is_coembed"]:
                                    suffix = "PCA_COEMBED_TARGET_PROJECTIONS.csv" if dr_key == "pca" else f"{dr_params.get('metric')}_UMAP_COEMBED_TARGET_PROJECTIONS.csv"
                                    precomputed_proj_path = os.path.join(original_run_dir, target_id_name, "similarity_spaces", repr_type, f"dim_{simspace_dim}", f"{target_id_name}_{repr_type}_dim{simspace_dim}_{suffix}")

                                if os.path.exists(precomputed_proj_path):
                                     cmd.append(f"--precomputed_target_projections_path={os.path.abspath(precomputed_proj_path)}")

                                run_command(cmd, f"Analyze-{target_id_name}-{strategy['short_name']}-dim{simspace_dim}-cutoff{cutoff}")

    logging.info("--- AFFINITY CUTOFF RE-ANALYSIS ORCHESTRATION COMPLETE ---")

if __name__ == "__main__":
    main()