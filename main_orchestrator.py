import json
import os
import shutil
import subprocess
import pandas as pd
import logging
from datetime import datetime
import argparse # Added for CLI flexibility for the orchestrator itself

# Setup basic logging to file and console
log_file_name = f"orchestrator_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)-8s - %(filename)-25s - %(funcName)-25s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_name),
        logging.StreamHandler()
    ]
)

def run_command(command_list, step_name="Command", cwd=None):
    """Runs a command, logs its output in real-time, and checks for errors."""
    logging.info(f"Running {step_name}: {' '.join(command_list)}")
    try:
        process = subprocess.Popen(
            command_list,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1, # Line buffered
            universal_newlines=True,
            cwd=cwd # Current working directory for the subprocess
        )

        # Log stdout and stderr line by line
        if process.stdout:
            for line in iter(process.stdout.readline, ''):
                logging.debug(f"[{step_name} STDOUT] {line.strip()}")
            process.stdout.close()
        
        stderr_output = ""
        if process.stderr:
            # Read all stderr at once after process finishes or if process fails quickly
            # This is a compromise; for very long stderr, it might delay logging it.
            # However, line-by-line stderr polling can be complex with Popen.
            for line in iter(process.stderr.readline, ''):
                logging.debug(f"[{step_name} STDERR_LINE] {line.strip()}") # Log line by line for debug
                stderr_output += line # Accumulate
            process.stderr.close()


        return_code = process.wait() # Wait for the process to complete

        if return_code == 0:
            logging.info(f"{step_name} completed successfully.")
            if stderr_output.strip(): # Log accumulated stderr even on success if it's not empty (for warnings)
                logging.debug(f"[{step_name} STDERR on success (accumulated)]\n{stderr_output.strip()}")
            return True
        else:
            logging.error(f"{step_name} FAILED with return code {return_code}.")
            if stderr_output.strip():
                logging.error(f"[{step_name} STDERR on failure (accumulated)]\n{stderr_output.strip()}")
            return False
    except FileNotFoundError:
        logging.error(f"{step_name} FAILED: Command not found (e.g., 'python' or script path incorrect).")
        return False
    except Exception as e:
        logging.error(f"Exception during {step_name}: {e}")
        return False

def get_mf_keyword_id_from_keywords_csv(mf_keywords_csv_path, canonical_mf_name):
    """Looks up the KW-ID for a given canonical molecular function name."""
    try:
        mf_keywords_df = pd.read_csv(mf_keywords_csv_path)
        row = mf_keywords_df[mf_keywords_df['Name'].str.lower() == canonical_mf_name.lower()]
        if not row.empty:
            return row.iloc[0]['ID']
        logging.warning(f"KW-ID not found for canonical MF name: '{canonical_mf_name}' in {mf_keywords_csv_path}")
    except Exception as e:
        logging.error(f"Error reading or searching {mf_keywords_csv_path} for {canonical_mf_name}: {e}")
    return None


def main(config_path="experiment_config.json"):
    logging.info(f"========== STARTING UMMBAS SIMILARITY EXPERIMENTAL PIPELINE (CONFIG: {config_path}) ==========")
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_path}. Aborting.")
        return
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON configuration file {config_path}: {e}. Aborting.")
        return

    gs = config['global_settings']
    workspace_base_dir = gs['workspace_base_dir']
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_experiment_run_dir = os.path.join(workspace_base_dir, f"run_{timestamp}")
    try:
        os.makedirs(current_experiment_run_dir, exist_ok=True)
        logging.info(f"Main experiment output directory: {current_experiment_run_dir}")
    except OSError as e:
        logging.error(f"Could not create experiment run directory {current_experiment_run_dir}: {e}. Aborting.")
        return
        
    rdkit_features_list_target_json_str = json.dumps(gs.get('rdkit_features_list_target', []))
    run_coembedding_pca_umap_flag = gs.get('run_coembedding_for_pca_umap', False)
    dr_method_configs_json_str = json.dumps(config.get('dimensionality_reduction_methods', {}))


    for target_info in config['targets']:
        target_id_name = target_info['id_name']
        # canonical_mf_name = target_info['molecular_function_canonical_name'] # Not directly used in orchestrator after prepare_data
        
        logging.info(f"\n========== PROCESSING TARGET: {target_info['display_name']} ({target_id_name}) ==========")

        target_workspace_dir = os.path.join(current_experiment_run_dir, target_id_name)
        temp_data_dir = os.path.join(target_workspace_dir, "temp_data")
        os.makedirs(temp_data_dir, exist_ok=True)

        logging.info("--- Step 1: Preparing Data (Filtering Pre-calculated & Extracting Raw Target Ligands) ---")
        cmd_prepare_data = [
            "python", "experimental_pipeline/prepare_data.py",
            "--config_path", os.path.abspath(config_path), 
            "--target_id_name", target_id_name,
            "--output_dir", os.path.abspath(temp_data_dir)
        ]
        if not run_command(cmd_prepare_data, f"Data Prep for {target_id_name}"):
            logging.error(f"Data preparation failed for {target_id_name}. Skipping this target.")
            continue
        
        raw_target_ligands_for_feature_calc_csv = os.path.join(temp_data_dir, f"{target_id_name}_target_ligands_for_feature_calc_raw.csv")
        if not os.path.exists(raw_target_ligands_for_feature_calc_csv):
            logging.error(f"Raw target ligands file not found after prepare_data: {raw_target_ligands_for_feature_calc_csv}. Skipping target {target_id_name}.")
            continue

        for repr_type in config['representations']:
            logging.info(f"  +++++ REPRESENTATION: {repr_type.upper()} for {target_id_name} +++++")
            
            target_ligands_repr_processed_dir = os.path.join(temp_data_dir, repr_type, "target_ligands_processed")
            os.makedirs(target_ligands_repr_processed_dir, exist_ok=True)

            logging.info(f"    --- Step 2a: Calculating {repr_type} for TARGET LIGANDS ---")
            cmd_molcalcs_target = [
                "python", "core_scripts/calculate_features_and_fingerprints_exp.py",
                "--input_csv", os.path.abspath(raw_target_ligands_for_feature_calc_csv),
                "--output_dir", os.path.abspath(target_ligands_repr_processed_dir),
                "--representation_type", repr_type,
                "--file_label", f"{target_id_name}_target_ligands_project", 
                "--n_jobs", str(gs.get('n_jobs_molcalcs', -1)),
            ]
            if not run_command(cmd_molcalcs_target, f"MolCalcs Target ({repr_type}) for {target_id_name}"):
                logging.error(f"Feature/Fingerprint calculation for TARGET LIGANDS ({repr_type}) failed. Skipping this representation for {target_id_name}.")
                continue
            
            processed_target_ligands_repr_file = os.path.join(target_ligands_repr_processed_dir, f"{target_id_name}_target_ligands_project_{repr_type}.csv")
            if not os.path.exists(processed_target_ligands_repr_file):
                 logging.error(f"Processed target ligand file not found: {processed_target_ligands_repr_file}. Skipping this representation for {target_id_name}.")
                 continue
            
            current_chembl_mf_filtered_path = os.path.join(temp_data_dir, f"{target_id_name}_chembl_mf_excluded_{repr_type}.csv")
            current_zinc_filtered_path = os.path.join(temp_data_dir, f"{target_id_name}_zinc_excluded_{repr_type}.csv")

            if not (os.path.exists(current_chembl_mf_filtered_path)): # ZINC is optional for calc_simspaces
                logging.error(f"Filtered pre-calculated ChEMBL MF file not found for {repr_type}. Skipping this representation.")
                logging.debug(f"Expected ChEMBL: {current_chembl_mf_filtered_path}")
                continue
            if not os.path.exists(current_zinc_filtered_path):
                 logging.warning(f"Filtered pre-calculated ZINC file not found ({current_zinc_filtered_path}). Proceeding without ZINC for this part if script handles it.")
                 # calc_similarityspaces can handle missing ZINC path by passing None


            for simspace_dim_val in gs['simspace_dims_to_test']:
                logging.info(f"      ##### SIMSPACE_DIM: {simspace_dim_val} for {repr_type}, {target_id_name} #####")

                models_output_dir_dim = os.path.join(target_workspace_dir, "models", repr_type, f"dim_{simspace_dim_val}")
                simspaces_output_dir_dim = os.path.join(target_workspace_dir, "similarity_spaces", repr_type, f"dim_{simspace_dim_val}")
                os.makedirs(models_output_dir_dim, exist_ok=True)
                os.makedirs(simspaces_output_dir_dim, exist_ok=True)

                logging.info(f"        --- Step 2c: Calculating Similarity Spaces (and Co-embeddings if enabled) ---")
                cmd_calc_simspace = [
                    "python", "core_scripts/calculate_similarityspaces_exp.py",
                    "--chembl_mf_data_path", os.path.abspath(current_chembl_mf_filtered_path),
                    # Pass ZINC path only if it exists
                    "--target_ligands_unscaled_path_for_tsne_and_coembed", os.path.abspath(processed_target_ligands_repr_file),
                    "--simspace_dim", str(simspace_dim_val),
                    "--representation_type", repr_type,
                    "--target_id_name", target_id_name,
                    "--output_simspace_dir", os.path.abspath(simspaces_output_dir_dim),
                    "--output_model_dir", os.path.abspath(models_output_dir_dim),
                    "--rdkit_features_list_target_str", rdkit_features_list_target_json_str,
                    f"--run_coembedding_for_pca_umap={str(run_coembedding_pca_umap_flag)}",
                    "--dr_method_configs_json_str", dr_method_configs_json_str
                ]
                if os.path.exists(current_zinc_filtered_path):
                    cmd_calc_simspace.extend(["--zinc_data_path", os.path.abspath(current_zinc_filtered_path)])
                else:
                    cmd_calc_simspace.extend(["--zinc_data_path", "None"]) # Explicitly pass None if not found

                
                active_dr_methods_config = config["dimensionality_reduction_methods"]
                cmd_calc_simspace.append(f"--dr_method_pca={str('pca' in active_dr_methods_config)}")
                
                has_umap_methods = any("umap" in k for k in active_dr_methods_config)
                cmd_calc_simspace.append(f"--dr_method_umap={str(has_umap_methods)}")
                if has_umap_methods:
                    for dr_key_conf, dr_params_conf in active_dr_methods_config.items():
                        if "umap" in dr_key_conf:
                            metric_name = dr_params_conf.get("metric", dr_key_conf.split('_')[-1])
                            # This creates flags like --umap_metric_to_run_euclidean (implicit True)
                            # The calculate_simspaces script uses action='store_true' for these.
                            cmd_calc_simspace.append(f"--umap_metric_to_run_{metric_name}")
                
                cmd_calc_simspace.append(f"--dr_method_tsne={str('tsne' in active_dr_methods_config)}")
                if 'tsne' in active_dr_methods_config:
                    cmd_calc_simspace.append(f"--tsne_perplexity={str(active_dr_methods_config['tsne']['perplexity'])}")
                    cmd_calc_simspace.append(f"--tsne_pca_components={str(gs['tsne_pca_components'])}")

                if not run_command(cmd_calc_simspace, f"Calc SimSpace/CoEmbed ({repr_type}, dim{simspace_dim_val}) for {target_id_name}"):
                    logging.error(f"Similarity space/co-embedding calculation failed for ({repr_type}, dim{simspace_dim_val}). Skipping analysis for this dimension.")
                    continue

                comprehensive_simspace_csv = os.path.join(simspaces_output_dir_dim, f"{target_id_name}_{repr_type}_dim{simspace_dim_val}_similarity_space.csv")
                if not os.path.exists(comprehensive_simspace_csv):
                    logging.error(f"Comprehensive similarity space CSV not found: {comprehensive_simspace_csv}. Skipping analysis for this dimension.")
                    continue

                # --- Loop for Projection and Analysis (Original + Co-embedded if applicable) ---
                for dr_key_analysis, dr_params_analysis_config in active_dr_methods_config.items():
                    dr_short_name_base = dr_params_analysis_config["short_name"] # e.g., "PCA", "UMAP-Euclidean"
                    
                    analysis_runs_for_dr_method = []
                    # Strategy 1: Projection (for PCA, UMAP) or Standard t-SNE (always co-embedded in its script)
                    analysis_runs_for_dr_method.append({
                        "strategy_label": "Projection" if not dr_key_analysis == "tsne" else "CoEmbedding (Native)", 
                        "dr_short_name_for_proj_analyze": dr_short_name_base, # Used for output dir and passed to project_and_analyze
                        "dr_short_name_for_coords_lookup": dr_short_name_base, # Coords in precomp file use base name
                        "is_coembed_strategy_run": False 
                    })

                    # Strategy 2: Co-embedding analysis for PCA, UMAP if enabled
                    allow_coembed_from_config = dr_params_analysis_config.get("allow_coembedding", False)
                    is_pca_or_umap = dr_key_analysis.startswith("pca") or dr_key_analysis.startswith("umap")

                    if run_coembedding_pca_umap_flag and is_pca_or_umap and allow_coembed_from_config:
                        analysis_runs_for_dr_method.append({
                            "strategy_label": "CoEmbedding",
                            "dr_short_name_for_proj_analyze": f"{dr_short_name_base}-Coembed".replace('-', '_'), # Distinct name for output_dir
                            "dr_short_name_for_coords_lookup": dr_short_name_base, # Coords in COEMBED_TARGET_PROJECTIONS file use base name
                            "is_coembed_strategy_run": True
                        })
                    
                    for run_details in analysis_runs_for_dr_method:
                        current_dr_short_name_for_output = run_details["dr_short_name_for_proj_analyze"]
                        current_strategy_label = run_details["strategy_label"]
                        is_coembed_run = run_details["is_coembed_strategy_run"]
                        dr_short_name_for_coords = run_details["dr_short_name_for_coords_lookup"]


                        logging.info(f"          --- Step 2d ({current_strategy_label}): Analyzing DR: {current_dr_short_name_for_output} ---")

                        if dr_key_analysis == "tsne" and simspace_dim_val != 2:
                            logging.info(f"Skipping Analyze for t-SNE ({current_dr_short_name_for_output}) as simspace_dim is {simspace_dim_val}.")
                            continue 
                        # If it's a t-SNE run, it's always the native co-embedding. Skip any additionally generated "CoEmbedding" strategy run for t-SNE.
                        if dr_key_analysis == "tsne" and is_coembed_run:
                            continue


                        results_output_dir_strategy = os.path.join(target_workspace_dir, "results", repr_type, f"dim_{simspace_dim_val}", current_dr_short_name_for_output.replace('-', '_'))
                        os.makedirs(results_output_dir_strategy, exist_ok=True)

                        cmd_project_analyze = [
                            "python", "experimental_pipeline/project_and_analyze.py",
                            "--simspace_csv_path", os.path.abspath(comprehensive_simspace_csv),
                            "--dr_method_key", dr_key_analysis, # Original key for model/file lookup logic
                            # "--dr_short_name", dr_short_name_for_coords, # Name used for coord columns
                            "--dr_short_name", run_details.get("dr_short_name_for_coords_lookup", current_dr_short_name_for_output), 
                            "--simspace_dim", str(simspace_dim_val),
                            "--k_for_knn", ','.join(map(str, gs['k_for_knn_distance'])),
                            "--output_dir", os.path.abspath(results_output_dir_strategy), # Unique output dir per strategy
                            "--target_id_name", target_id_name,
                            "--representation_type", repr_type,
                            "--rdkit_features_list_target_str", rdkit_features_list_target_json_str
                        ]
                        
                        model_name_root_calc_simspace = f"{target_id_name}_{repr_type}_dim{simspace_dim_val}"

                        if dr_key_analysis == "tsne": 
                            tsne_target_proj_file = os.path.join(simspaces_output_dir_dim, f"{model_name_root_calc_simspace}_tSNE_TARGET_PROJECTIONS.csv")
                            if os.path.exists(tsne_target_proj_file):
                                cmd_project_analyze.extend(["--precomputed_target_projections_path", os.path.abspath(tsne_target_proj_file)])
                            else:
                                logging.warning(f"t-SNE target projections file not found: {tsne_target_proj_file}. Cannot analyze t-SNE.")
                                continue 
                        
                        elif is_coembed_run: # Co-embedded PCA/UMAP uses precomputed target projections
                            coembed_suffix = ""
                            if dr_key_analysis == "pca": 
                                coembed_suffix = "PCA_COEMBED_TARGET_PROJECTIONS.csv"
                            elif "umap" in dr_key_analysis:
                                metric_umap = dr_params_analysis_config.get("metric", dr_key_analysis.split('_')[-1])
                                coembed_suffix = f"{metric_umap}_UMAP_COEMBED_TARGET_PROJECTIONS.csv"
                            
                            precomputed_coembed_file = os.path.join(simspaces_output_dir_dim, f"{model_name_root_calc_simspace}_{coembed_suffix}")
                            if os.path.exists(precomputed_coembed_file):
                                cmd_project_analyze.extend(["--precomputed_target_projections_path", os.path.abspath(precomputed_coembed_file)])
                            else:
                                logging.warning(f"Co-embedded target projection file not found for {current_dr_short_name_for_output}: {precomputed_coembed_file}. Skipping analysis.")
                                continue
                        else: # Standard Projection for PCA/UMAP (uses models from model_dir)
                            cmd_project_analyze.extend([
                                "--target_ligands_repr_path", os.path.abspath(processed_target_ligands_repr_file),
                                "--model_dir_for_projection", os.path.abspath(models_output_dir_dim),
                                "--model_name_root_for_projection", model_name_root_calc_simspace # Used to find scaler & DR model
                            ])
                        
                        if not run_command(cmd_project_analyze, f"Analyze ({current_dr_short_name_for_output}, {repr_type}, dim{simspace_dim_val}) for {target_id_name}"):
                           logging.error(f"Analysis failed for ({current_dr_short_name_for_output}, {repr_type}, dim{simspace_dim_val}).")
            # End simspace_dim_val loop
        # End repr_type loop
    # End target_info loop

    logging.info("\n--- Step 3: Generating Final Report ---")
    cmd_generate_report = [
        "python", "reporting/generate_latex_report.py",
        "--experiment_run_dir", os.path.abspath(current_experiment_run_dir),
        "--config_path", os.path.abspath(config_path),
        "--output_dir", os.path.abspath(gs['final_report_dir'])
    ]
    if not run_command(cmd_generate_report, "Report Generation"):
        logging.error("Report generation failed.")

    logging.info(f"========== UMMBAS SIMILARITY EXPERIMENTAL PIPELINE FINISHED (RUN: {timestamp}) ==========")
    logging.info(f"Main log file for this run: {log_file_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main orchestrator for UMMBAS similarity experiments.")
    parser.add_argument("--config", default="experiment_config.json", help="Path to the experiment configuration JSON file.")
    args = parser.parse_args()
    main(config_path=args.config)