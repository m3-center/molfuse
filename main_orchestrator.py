import json
import os
import shutil 
import subprocess
import pandas as pd 
import logging
from datetime import datetime
import argparse

# Global variable for log file name, will be set by setup_logging
log_file_name = "orchestrator_uninitialized.log" 
log_file_name_base = f"orchestrator_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def setup_logging(log_base_name, seed, repr_mode):
    """Sets up logging with seed and representation mode in filename."""
    global log_file_name 
    log_file_name = f"{log_base_name}_seed{seed}_repr{repr_mode}.log"
    
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            handler.close() 

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)-8s - %(filename)-25s - %(funcName)-25s - %(lineno)-4d - %(message)s',
        handlers=[
            logging.FileHandler(log_file_name, mode='w'), 
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging setup complete. Log file: {log_file_name}")


def run_command(command_list, step_name="Command", cwd=None):
    """Runs a command, logs its output in real-time, and checks for errors."""
    logging.info(f"Running {step_name}: {' '.join(command_list)}")
    try:
        process = subprocess.Popen(
            command_list,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1, 
            universal_newlines=True,
            cwd=cwd
        )

        stdout_output_lines = []
        if process.stdout:
            for line in iter(process.stdout.readline, ''):
                stripped_line = line.strip()
                if stripped_line:
                    logging.debug(f"[{step_name} STDOUT] {stripped_line}")
                stdout_output_lines.append(line)
            process.stdout.close()
        
        stderr_accumulated = ""
        if process.stderr:
            for line in iter(process.stderr.readline, ''):
                stripped_line = line.strip()
                if stripped_line:
                    logging.debug(f"[{step_name} STDERR_LINE] {stripped_line}")
                stderr_accumulated += line
            process.stderr.close()

        return_code = process.wait() 

        if return_code == 0:
            logging.info(f"{step_name} completed successfully.")
            if stderr_accumulated.strip():
                logging.info(f"[{step_name} STDERR on success (accumulated)]\n{stderr_accumulated.strip()}")
            return True
        else:
            logging.error(f"{step_name} FAILED with return code {return_code}.")
            if stdout_output_lines:
                logging.error(f"[{step_name} STDOUT (last 20 lines on failure)]:\n{''.join(stdout_output_lines[-20:])}")
            if stderr_accumulated.strip():
                logging.error(f"[{step_name} STDERR on failure (accumulated)]\n{stderr_accumulated.strip()}")
            return False
    except FileNotFoundError:
        logging.error(f"{step_name} FAILED: Command or script not found (check paths).")
        return False
    except Exception as e:
        logging.error(f"Exception during {step_name}: {e}", exc_info=True)
        return False

def get_mf_keyword_id_from_keywords_csv(mf_keywords_csv_path, canonical_mf_name):
    """Looks up the KW-ID for a given canonical molecular function name."""
    try:
        mf_keywords_df = pd.read_csv(mf_keywords_csv_path)
        row = mf_keywords_df[mf_keywords_df['Name'].str.lower() == str(canonical_mf_name).lower()]
        if not row.empty:
            return row.iloc[0]['ID']
        logging.warning(f"KW-ID not found for MF name: '{canonical_mf_name}' in {mf_keywords_csv_path}")
    except Exception as e:
        logging.error(f"Error reading or searching {mf_keywords_csv_path} for {canonical_mf_name}: {e}")
    return None


def main(config_path, representation_mode, random_seed_value):
    setup_logging(log_file_name_base, random_seed_value, representation_mode)

    logging.info(f"========== STARTING UMMBAS SIMILARITY EXPERIMENT REPLICATE RUN ==========")
    logging.info(f"Config: {config_path}, Representation Mode: {representation_mode}, Random Seed: {random_seed_value}")
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_path}. Aborting.")
        return
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON config {config_path}: {e}. Aborting.")
        return

    gs = config['global_settings']
    # The workspace_base_dir is now the parent directory for all runs from this config
    workspace_base_dir = gs['workspace_base_dir']
    
    run_specific_name = f"run_seed{random_seed_value}_repr{representation_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    current_replicate_run_dir = os.path.join(workspace_base_dir, run_specific_name)
    try:
        os.makedirs(current_replicate_run_dir, exist_ok=True)
        logging.info(f"Main output directory for this replicate run: {current_replicate_run_dir}")
    except OSError as e:
        logging.error(f"Could not create replicate run directory {current_replicate_run_dir}: {e}. Aborting.")
        return
        
    # --- NEW: Copy the configuration file to the run directory for traceability ---
    try:
        dest_config_path = os.path.join(current_replicate_run_dir, "run_config.json")
        shutil.copy2(config_path, dest_config_path)
        logging.info(f"Copied experiment configuration to: {dest_config_path}")
    except Exception as e:
        logging.error(f"FATAL: Failed to copy configuration file '{config_path}' to workspace. Aborting run. Error: {e}")
        return
    # --- END NEW ---

    rdkit_features_list_target_json_str = json.dumps(gs.get('rdkit_features_list_target', []))
    run_coembedding_pca_umap_flag = gs.get('run_coembedding_for_pca_umap', False)
    dr_method_configs_json_str = json.dumps(config.get('dimensionality_reduction_methods', {}))

    representations_to_process = []
    if representation_mode == "features":
        if "features" in config['representations']: representations_to_process.append("features")
        else: logging.error("'features' not in config['representations']."); return
    elif representation_mode == "fingerprints":
        if "fingerprints" in config['representations']: representations_to_process.append("fingerprints")
        else: logging.error("'fingerprints' not in config['representations']."); return
    else: logging.error(f"Invalid representation_mode: {representation_mode}"); return
    
    logging.info(f"This run will process representation(s): {representations_to_process}")

    for target_info in config['targets']:
        target_id_name = target_info['id_name']
        target_processing_mode = target_info.get("processing_mode", "full_analysis")
        
        logging.info(f"\n========== PROCESSING TARGET: {target_info['display_name']} ({target_id_name}) | Mode: {target_processing_mode} ==========")

        target_workspace_dir = os.path.join(current_replicate_run_dir, target_id_name)
        temp_data_dir = os.path.join(target_workspace_dir, "temp_data")
        os.makedirs(temp_data_dir, exist_ok=True)

        logging.info("--- Step 1: Preparing Data ---")
        cmd_prepare_data = [ "python", "experimental_pipeline/prepare_data.py",
            "--config_path", os.path.abspath(config_path), "--target_id_name", target_id_name,
            "--output_dir", os.path.abspath(temp_data_dir) ]
        if not run_command(cmd_prepare_data, f"Data Prep for {target_id_name}"):
            logging.error(f"Data preparation failed for {target_id_name}. Skipping target for this replicate."); continue
        
        raw_target_ligands_for_feature_calc_csv = os.path.join(temp_data_dir, f"{target_id_name}_target_ligands_for_feature_calc_raw.csv")
        
        for repr_type in representations_to_process:
            logging.info(f"  +++++ REPRESENTATION: {repr_type.upper()} for {target_id_name} +++++")
            
            target_ligands_repr_calc_output_dir = os.path.join(target_workspace_dir, "target_ligands_calculated", repr_type)
            os.makedirs(target_ligands_repr_calc_output_dir, exist_ok=True)
            processed_target_ligands_repr_file = os.path.join(target_ligands_repr_calc_output_dir, f"{target_id_name}_target_ligands_for_calc_{repr_type}.csv")

            if os.path.exists(raw_target_ligands_for_feature_calc_csv):
                logging.info(f"    --- Step 2a: Calculating {repr_type} for TARGET LIGANDS (if any) ---")
                cmd_molcalcs_target = [ "python", "core_scripts/calculate_features_and_fingerprints_exp.py",
                    "--input_csv", os.path.abspath(raw_target_ligands_for_feature_calc_csv),
                    "--output_dir", os.path.abspath(target_ligands_repr_calc_output_dir),
                    "--representation_type", repr_type,
                    "--file_label", f"{target_id_name}_target_ligands_for_calc", 
                    "--n_jobs", str(gs.get('n_jobs_molcalcs', -1))
                ]
                if not run_command(cmd_molcalcs_target, f"MolCalcs Target ({repr_type}) for {target_id_name}"):
                    logging.warning(f"Target ligand {repr_type} calculation failed. Co-embedding may be affected.")
            else:
                logging.warning(f"Raw target ligands file not found ({raw_target_ligands_for_feature_calc_csv}). Skipping target ligand calculation.")
                if not os.path.exists(processed_target_ligands_repr_file):
                    pd.DataFrame().to_csv(processed_target_ligands_repr_file, index=False)
                    logging.info(f"Created empty placeholder for: {processed_target_ligands_repr_file}")

            current_chembl_mf_filtered_path = os.path.join(temp_data_dir, f"{target_id_name}_chembl_mf_excluded_{repr_type}.csv")
            current_zinc_filtered_path = os.path.join(temp_data_dir, f"{target_id_name}_zinc_excluded_{repr_type}.csv")

            if not os.path.exists(current_chembl_mf_filtered_path):
                logging.error(f"ChEMBL MF file not found: {current_chembl_mf_filtered_path}. Skipping simspace for {repr_type}."); continue
            
            for simspace_dim_val in gs['simspace_dims_to_test']:
                logging.info(f"      ##### SIMSPACE_DIM: {simspace_dim_val} for {repr_type}, {target_id_name} #####")
                models_output_dir_dim = os.path.join(target_workspace_dir, "models", repr_type, f"dim_{simspace_dim_val}")
                simspaces_output_dir_dim = os.path.join(target_workspace_dir, "similarity_spaces", repr_type, f"dim_{simspace_dim_val}")
                os.makedirs(models_output_dir_dim, exist_ok=True); os.makedirs(simspaces_output_dir_dim, exist_ok=True)

                logging.info(f"        --- Step 2b: Calculating Similarity Spaces ---")
                cmd_calc_simspace = [ "python", "core_scripts/calculate_similarityspaces_exp.py",
                    "--chembl_mf_data_path", os.path.abspath(current_chembl_mf_filtered_path),
                    "--target_ligands_unscaled_path_for_tsne_and_coembed", os.path.abspath(processed_target_ligands_repr_file if os.path.exists(processed_target_ligands_repr_file) else "None"),
                    "--simspace_dim", str(simspace_dim_val), "--representation_type", repr_type,
                    "--target_id_name", target_id_name, "--output_simspace_dir", os.path.abspath(simspaces_output_dir_dim),
                    "--output_model_dir", os.path.abspath(models_output_dir_dim),
                    "--rdkit_features_list_target_str", rdkit_features_list_target_json_str,
                    f"--run_coembedding_for_pca_umap={str(run_coembedding_pca_umap_flag)}",
                    "--dr_method_configs_json_str", dr_method_configs_json_str,
                    "--random_state", str(random_seed_value) ]
                if os.path.exists(current_zinc_filtered_path): cmd_calc_simspace.extend(["--zinc_data_path", os.path.abspath(current_zinc_filtered_path)])
                else: cmd_calc_simspace.extend(["--zinc_data_path", "None"])
                
                active_dr_methods_cfg = config["dimensionality_reduction_methods"]
                cmd_calc_simspace.append(f"--dr_method_pca={str('pca' in active_dr_methods_cfg)}")
                has_umap = any("umap" in k for k in active_dr_methods_cfg); cmd_calc_simspace.append(f"--dr_method_umap={str(has_umap)}")
                if has_umap:
                    for drk, drp in active_dr_methods_cfg.items():
                        if "umap" in drk: cmd_calc_simspace.append(f"--umap_metric_to_run_{drp.get('metric')}")
                cmd_calc_simspace.append(f"--dr_method_tsne={str('tsne' in active_dr_methods_cfg)}")
                if 'tsne' in active_dr_methods_cfg:
                    cmd_calc_simspace.extend([f"--tsne_perplexity={str(active_dr_methods_cfg['tsne']['perplexity'])}", 
                                              f"--tsne_pca_components={str(gs['tsne_pca_components'])}"])
                    if 'tsne_n_neighbors' in gs: cmd_calc_simspace.append(f"--n_neighbors={str(gs['tsne_n_neighbors'])}")

                if not run_command(cmd_calc_simspace, f"Calc SimSpace ({repr_type}, dim{simspace_dim_val}) for {target_id_name}"):
                    logging.error(f"SimSpace calc failed. Skipping further analysis for this dim."); continue

                comprehensive_simspace_csv = os.path.join(simspaces_output_dir_dim, f"{target_id_name}_{repr_type}_dim{simspace_dim_val}_similarity_space.csv")
                if not os.path.exists(comprehensive_simspace_csv):
                    logging.error(f"Comprehensive simspace CSV not found: {comprehensive_simspace_csv}. Skipping."); continue

                if target_processing_mode == "full_analysis":
                    logging.info(f"        --- Step 2c: Projecting & Analyzing (Full Analysis Mode) ---")
                    for dr_key, dr_params_cfg in active_dr_methods_cfg.items():
                        base_dr_short_name = dr_params_cfg["short_name"]
                        analysis_runs = [{"strategy_label": "Projection" if not dr_key == "tsne" else "CoEmbedding (Native)", 
                                          "dr_name_for_output_files": base_dr_short_name, 
                                          "dr_name_for_coord_lookup": base_dr_short_name, "is_coembed_run": False }]
                        if run_coembedding_pca_umap_flag and dr_params_cfg.get("allow_coembedding", False) and (dr_key.startswith("pca") or dr_key.startswith("umap")):
                            analysis_runs.append({"strategy_label": "CoEmbedding", 
                                                  "dr_name_for_output_files": f"{base_dr_short_name}-Coembed", 
                                                  "dr_name_for_coord_lookup": base_dr_short_name, "is_coembed_run": True})
                        
                        for run_info in analysis_runs:
                            output_leaf_name = run_info["dr_name_for_output_files"]
                            coord_lookup_name = run_info["dr_name_for_coord_lookup"]
                            logging.info(f"          --- ({run_info['strategy_label']}) Analyzing DR: {output_leaf_name} ---")
                            if dr_key == "tsne" and simspace_dim_val != 2: logging.info("Skipping t-SNE analysis (dim != 2)."); continue
                            if dr_key == "tsne" and run_info["is_coembed_run"]: continue
                            results_out_dir = os.path.join(target_workspace_dir, "results", repr_type, f"dim_{simspace_dim_val}", output_leaf_name.replace('-', '_'))
                            os.makedirs(results_out_dir, exist_ok=True)
                            cmd_pa = [ "python", "experimental_pipeline/project_and_analyze.py",
                                "--simspace_csv_path", os.path.abspath(comprehensive_simspace_csv),
                                "--dr_method_key", dr_key, "--dr_short_name", coord_lookup_name,
                                "--simspace_dim", str(simspace_dim_val), "--k_for_knn", ','.join(map(str, gs['k_for_knn_distance'])),
                                "--output_dir", os.path.abspath(results_out_dir),
                                "--target_id_name", target_id_name, "--representation_type", repr_type,
                                "--rdkit_features_list_target_str", rdkit_features_list_target_json_str ]
                            
                            model_root_name = f"{target_id_name}_{repr_type}_dim{simspace_dim_val}"
                            if dr_key == "tsne":
                                tsne_proj_file = os.path.join(simspaces_output_dir_dim, f"{model_root_name}_tSNE_TARGET_PROJECTIONS.csv")
                                if os.path.exists(tsne_proj_file): cmd_pa.append(f"--precomputed_target_projections_path={os.path.abspath(tsne_proj_file)}")
                                else: logging.warning(f"t-SNE target proj file missing: {tsne_proj_file}. Skipping analysis."); continue
                            elif run_info["is_coembed_run"]:
                                co_suffix = "PCA_COEMBED_TARGET_PROJECTIONS.csv" if dr_key=="pca" else f"{dr_params_cfg.get('metric')}_UMAP_COEMBED_TARGET_PROJECTIONS.csv"
                                precomp_co_file = os.path.join(simspaces_output_dir_dim, f"{model_root_name}_{co_suffix}")
                                if os.path.exists(precomp_co_file): cmd_pa.append(f"--precomputed_target_projections_path={os.path.abspath(precomp_co_file)}")
                                else: logging.warning(f"Co-embed target proj file missing: {precomp_co_file}. Skipping analysis."); continue
                            else: # Projection for PCA/UMAP
                                cmd_pa.extend([f"--target_ligands_repr_path={os.path.abspath(processed_target_ligands_repr_file if os.path.exists(processed_target_ligands_repr_file) else 'None')}",
                                               f"--model_dir_for_projection={os.path.abspath(models_output_dir_dim)}",
                                               f"--model_name_root_for_projection={model_root_name}"])
                            if not run_command(cmd_pa, f"Analyze ({output_leaf_name}, {repr_type}, dim{simspace_dim_val})"):
                                logging.error(f"Analysis failed for {output_leaf_name}.")
                
                elif target_processing_mode == "similarity_space_only":
                    logging.info(f"        --- Step 2c: Ranking ZINC Decoys (Similarity Space Only Mode) ---")
                    for dr_key_rank, dr_params_rank_cfg in active_dr_methods_cfg.items():
                        dr_short_name_rank = dr_params_rank_cfg["short_name"]
                        ranked_zinc_out_dir = os.path.join(target_workspace_dir, "ranked_zinc_for_docking", repr_type, f"dim_{simspace_dim_val}", dr_short_name_rank.replace('-', '_'))
                        os.makedirs(ranked_zinc_out_dir, exist_ok=True)
                        cmd_rz = [ "python", "experimental_pipeline/rank_zinc_decoys.py",
                            "--simspace_csv_path", os.path.abspath(comprehensive_simspace_csv),
                            "--dr_short_name", dr_short_name_rank, "--simspace_dim", str(simspace_dim_val),
                            "--output_dir", os.path.abspath(ranked_zinc_out_dir),
                            "--target_id_name", target_id_name, "--representation_type", repr_type ]
                        if not run_command(cmd_rz, f"Rank ZINC ({dr_short_name_rank}, {repr_type}, dim{simspace_dim_val})"):
                           logging.error(f"ZINC Ranking failed for {dr_short_name_rank}.")
    logging.info(f"========== UMMBAS SIMILARITY EXPERIMENT REPLICATE RUN FINISHED (Seed: {random_seed_value}, Repr: {representation_mode}) ==========")
    logging.info(f"Log file for this replicate run: {log_file_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main orchestrator for a single replicate of UMMBAS experiments.")
    parser.add_argument("--config", default="experiment_config.json", help="Path to experiment config JSON.")
    parser.add_argument("--representation_mode", required=True, choices=["features", "fingerprints"], help="Process 'features' or 'fingerprints'.")
    parser.add_argument("--random_seed", required=True, type=int, help="Random seed for this replicate.")
    args = parser.parse_args()
    main(config_path=args.config, representation_mode=args.representation_mode, random_seed_value=args.random_seed)
