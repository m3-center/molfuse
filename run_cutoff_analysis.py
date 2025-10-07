import os
import glob
import subprocess
import argparse
import logging
import re
import json
from datetime import datetime

# --- Logging Setup and Helper Function (Unchanged) ---
log_file_name = f"cutoff_analysis_orchestrator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)-8s - %(filename)-25s - %(lineno)-4d - %(message)s',
                    handlers=[logging.FileHandler(log_file_name, mode='w'), logging.StreamHandler()])

def run_command(command_list, step_name="Command", cwd=None):
    logging.info(f"Running {step_name}: {' '.join(command_list)}")
    try:
        process = subprocess.run(
            command_list, capture_output=True, text=True, check=True, cwd=cwd
        )
        if process.stdout: logging.info(f"[{step_name} STDOUT]:\n{process.stdout.strip()}")
        if process.stderr: logging.warning(f"[{step_name} STDERR]:\n{process.stderr.strip()}")
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
    parser.add_argument("--original_workspace", default="experiment_workspace_rerun_hyperparam_sweep/", help="Directory containing the hyperparameter sweep runs.")
    parser.add_argument("--output_workspace", default="experiment_workspace_cutoff_analysis/", help="Directory to save the new cutoff analysis results.")
    parser.add_argument("--config_path", default="experiment_config.json", help="Path to the main experiment config file for context.")
    parser.add_argument("--cutoffs", default="100,1000,10000,100000", help="Comma-separated list of affinity cutoffs in nM.")
    args = parser.parse_args()

    logging.info("--- STARTING FOCUSED AFFINITY CUTOFF RE-ANALYSIS (FEATURES, 2D) ---")

    try:
        with open(args.config_path, 'r') as f:
            main_config = json.load(f)
    except Exception as e:
        logging.error(f"Could not load main config file '{args.config_path}'. Aborting. Error: {e}")
        return

    affinity_cutoffs = [int(c.strip()) for c in args.cutoffs.split(',')]
    logging.info(f"Will re-analyze using affinity cutoffs (nM): {affinity_cutoffs}")

    REPR_TYPE_TO_PROCESS = "features"
    SIMSPACE_DIM_TO_PROCESS = 2

    original_run_dirs = glob.glob(os.path.join(args.original_workspace, f"run_seed*_repr{REPR_TYPE_TO_PROCESS}_*"))
    if not original_run_dirs:
        logging.error(f"No original '{REPR_TYPE_TO_PROCESS}' runs found in '{args.original_workspace}'. Aborting.")
        return

    for original_run_dir in original_run_dirs:
        run_basename = os.path.basename(original_run_dir)
        logging.info(f"\n===== Processing Original Run: {run_basename} =====")
        
        output_run_dir = os.path.join(args.output_workspace, run_basename)
        if os.path.exists(output_run_dir):
            logging.info(f"  --> Output directory '{output_run_dir}' already exists. Skipping this run.")
            continue
        
        run_config_path = os.path.join(original_run_dir, "run_config.json")
        if not os.path.exists(run_config_path):
            logging.warning(f"run_config.json not found in {original_run_dir}, skipping.")
            continue
        with open(run_config_path, 'r') as f: run_config = json.load(f)

        target_info = run_config['targets'][0]
        target_id_name = target_info['id_name']
        dr_method_key = list(run_config['dimensionality_reduction_methods'].keys())[0]
        dr_params = run_config['dimensionality_reduction_methods'][dr_method_key]
        
        logging.info(f"  Target: {target_info['display_name']} | DR Method: {dr_params['short_name']}")
        
        # --- START OF FIX: Explicitly define and separate strategies ---
        strategies_to_run = []
        if dr_method_key == 'tsne':
            # t-SNE is ONLY a co-embedding strategy in this pipeline
            strategies_to_run.append("Co-embedding (Native)")
        else:
            # For PCA and UMAP, check which result directories exist
            if os.path.exists(os.path.join(original_run_dir, target_id_name, "results", REPR_TYPE_TO_PROCESS, f"dim_{SIMSPACE_DIM_TO_PROCESS}", dr_params['short_name'].replace('-', '_'))):
                strategies_to_run.append("Projection")
            if os.path.exists(os.path.join(original_run_dir, target_id_name, "results", REPR_TYPE_TO_PROCESS, f"dim_{SIMSPACE_DIM_TO_PROCESS}", f"{dr_params['short_name'].replace('-', '_')}_Coembed")):
                strategies_to_run.append("Co-embedding")
        # --- END OF FIX ---

        for strategy in strategies_to_run:
            for cutoff in affinity_cutoffs:
                logging.info(f"    * Re-analyzing Strategy: {strategy} with Cutoff: {cutoff} nM *")

                cmd = ["python", "experimental_pipeline/project_and_analyze.py"]
                simspace_path = ""
                
                if strategy == "Projection":
                    output_dir_leaf = dr_params['short_name'].replace('-', '_')
                    simspace_path = os.path.join(original_run_dir, target_id_name, "similarity_spaces", REPR_TYPE_TO_PROCESS, f"dim_{SIMSPACE_DIM_TO_PROCESS}", f"{target_id_name}_{REPR_TYPE_TO_PROCESS}_dim{SIMSPACE_DIM_TO_PROCESS}_similarity_space.csv")
                    model_dir = os.path.join(original_run_dir, target_id_name, "models", REPR_TYPE_TO_PROCESS, f"dim_{SIMSPACE_DIM_TO_PROCESS}")
                    model_name_root = f"{target_id_name}_{REPR_TYPE_TO_PROCESS}_dim{SIMSPACE_DIM_TO_PROCESS}"
                    target_ligands_path = os.path.join(original_run_dir, target_id_name, "target_ligands_calculated", REPR_TYPE_TO_PROCESS, f"{target_id_name}_target_ligands_for_calc_{REPR_TYPE_TO_PROCESS}.csv")
                    cmd.extend(["--simspace_csv_path", os.path.abspath(simspace_path), "--target_ligands_repr_path", os.path.abspath(target_ligands_path), "--model_dir_for_projection", os.path.abspath(model_dir), "--model_name_root_for_projection", model_name_root])

                elif strategy == "Co-embedding" or strategy == "Co-embedding (Native)":
                    output_dir_leaf = f"{dr_params['short_name'].replace('-', '_')}_Coembed"
                    if strategy == "Co-embedding (Native)": output_dir_leaf = dr_params['short_name'].replace('-', '_')
                    metric = dr_params.get("metric", "")
                    coembed_filename = f"{target_id_name}_{REPR_TYPE_TO_PROCESS}_dim{SIMSPACE_DIM_TO_PROCESS}_{dr_params['short_name'].replace('-', '_')}_similarity_space_COEMBED.csv"
                    if dr_method_key.startswith("umap"):
                        coembed_filename = f"{target_id_name}_{REPR_TYPE_TO_PROCESS}_dim{SIMSPACE_DIM_TO_PROCESS}_{metric}_UMAP_similarity_space_COEMBED.csv"
                    elif dr_method_key == "tsne":
                        coembed_filename = f"{target_id_name}_{REPR_TYPE_TO_PROCESS}_dim{SIMSPACE_DIM_TO_PROCESS}_tSNE_similarity_space_COEMBED.csv"
                    simspace_path = os.path.join(original_run_dir, target_id_name, "similarity_spaces", REPR_TYPE_TO_PROCESS, f"dim_{SIMSPACE_DIM_TO_PROCESS}", coembed_filename)
                    cmd.extend(["--simspace_csv_path", os.path.abspath(simspace_path)])

                if not os.path.exists(simspace_path):
                    logging.warning(f"Required simspace file not found, skipping: {simspace_path}")
                    continue

                new_output_dir = os.path.join(args.output_workspace, run_basename, f"results_cutoff_{cutoff}", target_id_name, "results", REPR_TYPE_TO_PROCESS, f"dim_{SIMSPACE_DIM_TO_PROCESS}", output_dir_leaf)
                os.makedirs(new_output_dir, exist_ok=True)

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

                run_command(cmd, f"Analyze-{target_id_name}-{strategy}-dim{SIMSPACE_DIM_TO_PROCESS}-cutoff{cutoff}")

    logging.info("--- AFFINITY CUTOFF RE-ANALYSIS ORCHESTRATION COMPLETE ---")

if __name__ == "__main__":
    main()
