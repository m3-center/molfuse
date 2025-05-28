import json
import os
import shutil
import subprocess
import pandas as pd
import logging
from datetime import datetime
import argparse

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("orchestrator.log"), logging.StreamHandler()])

def run_command(command_list, step_name="Command"):
    """Runs a command and logs its output."""
    logging.info(f"Running {step_name}: {' '.join(command_list)}")
    try:
        process = subprocess.Popen(command_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True)
        for stdout_line in iter(process.stdout.readline, ""):
            logging.debug(f"[{step_name} STDOUT] {stdout_line.strip()}")
        process.stdout.close()
        stderr_output = process.stderr.read()
        process.stderr.close()
        return_code = process.wait()

        if return_code == 0:
            logging.info(f"{step_name} completed successfully.")
            if stderr_output: logging.debug(f"[{step_name} STDERR on success]\n{stderr_output.strip()}")
            return True
        else:
            logging.error(f"{step_name} failed with return code {return_code}.")
            if stderr_output: logging.error(f"[{step_name} STDERR on failure]\n{stderr_output.strip()}")
            return False
    except Exception as e:
        logging.error(f"Exception during {step_name}: {e}")
        return False

def get_mf_keyword_id(mf_keywords_df, canonical_mf_name):
    """Looks up the KW-ID for a given canonical molecular function name."""
    row = mf_keywords_df[mf_keywords_df['Name'].str.lower() == canonical_mf_name.lower()]
    if not row.empty:
        return row.iloc[0]['ID']
    logging.warning(f"KW-ID not found for canonical MF name: {canonical_mf_name}")
    return None

def main(config_path="experiment_config.json"):
    logging.info(f"Starting experimental pipeline with config: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)

    gs = config['global_settings']
    workspace_base_dir = gs['workspace_base_dir']
    
    try:
        mf_keywords_df = pd.read_csv(gs['molecular_function_keywords_csv_path'])
    except FileNotFoundError:
        logging.error(f"Molecular function keywords file not found: {gs['molecular_function_keywords_csv_path']}. Aborting.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_experiment_run_dir = os.path.join(workspace_base_dir, f"run_{timestamp}")
    os.makedirs(current_experiment_run_dir, exist_ok=True)
    logging.info(f"Main experiment run directory: {current_experiment_run_dir}")

    for target_info in config['targets']:
        target_id_name = target_info['id_name']
        canonical_mf_name = target_info['molecular_function_canonical_name']
        
        logging.info(f"\n========== Processing Target: {target_info['display_name']} ({target_id_name}) ==========")

        mf_kw_id = get_mf_keyword_id(mf_keywords_df, canonical_mf_name)
        if not mf_kw_id:
            logging.error(f"Skipping target {target_id_name} due to missing KW-ID for MF: '{canonical_mf_name}'.")
            continue
        
        mf_filename_segment = target_info.get('molecular_function_filename_segment', canonical_mf_name.replace(' ', '_').replace('/', '_'))
        
        target_workspace_dir = os.path.join(current_experiment_run_dir, target_id_name)
        temp_data_dir = os.path.join(target_workspace_dir, "temp_data")
        # Subdirectories for models, simspaces, results will be created per repr/dim/dr by sub-scripts or here
        os.makedirs(temp_data_dir, exist_ok=True) # ensure temp_data_dir exists

        logging.info("Step 1: Preparing data (filtering pre-calculated files & extracting raw target ligands)...")
        cmd_prepare_data = [
            "python", "experimental_pipeline/prepare_data.py",
            "--config_path", config_path,
            "--target_id_name", target_id_name,
            "--output_dir", temp_data_dir
        ]
        if not run_command(cmd_prepare_data, f"Data Prep for {target_id_name}"):
            logging.error(f"Data preparation failed for {target_id_name}. Skipping target.")
            continue
        
        raw_target_ligands_for_feature_calc_csv = os.path.join(temp_data_dir, f"{target_id_name}_target_ligands_for_feature_calc_raw.csv")

        for repr_type in config['representations']:
            logging.info(f"  +++++ Representation: {repr_type.upper()} for {target_id_name} +++++")
            
            target_ligands_repr_processed_dir = os.path.join(temp_data_dir, repr_type, "target_ligands_processed")
            os.makedirs(target_ligands_repr_processed_dir, exist_ok=True)

            logging.info(f"    Step 2a: Calculating {repr_type} for TARGET LIGANDS...")
            cmd_molcalcs_target = [
                "python", "core_scripts/calculate_features_and_fingerprints_exp.py",
                "--input_csv", raw_target_ligands_for_feature_calc_csv,
                "--output_dir", target_ligands_repr_processed_dir,
                "--representation_type", repr_type,
                "--file_label", f"{target_id_name}_target_ligands_project",
                "--n_jobs", str(gs['n_jobs_molcalcs']),
                "--rdkit_features_list_target_str", json.dumps(gs['rdkit_features_list_target']) # Pass as JSON string
            ]
            if not run_command(cmd_molcalcs_target, f"MolCalcs Target ({repr_type}) for {target_id_name}"):
                logging.error(f"MolCalcs for TARGET LIGANDS ({repr_type}) failed. Skipping this representation.")
                continue
            
            processed_target_ligands_repr_file = os.path.join(target_ligands_repr_processed_dir, f"{target_id_name}_target_ligands_project_{repr_type}.csv")
            
            # Paths to FILTERED PRE-CALCULATED files from prepare_data.py output
            current_chembl_mf_filtered_path = os.path.join(temp_data_dir, f"{target_id_name}_chembl_mf_excluded_{repr_type}.csv")
            current_zinc_filtered_path = os.path.join(temp_data_dir, f"{target_id_name}_zinc_excluded_{repr_type}.csv")

            for simspace_dim in gs['simspace_dims_to_test']:
                logging.info(f"      ##### SIMSPACE_DIM: {simspace_dim} for {repr_type}, {target_id_name} #####")

                models_dir = os.path.join(target_workspace_dir, "models", repr_type, f"dim_{simspace_dim}")
                simspaces_dir = os.path.join(target_workspace_dir, "similarity_spaces", repr_type, f"dim_{simspace_dim}")
                os.makedirs(models_dir, exist_ok=True)
                os.makedirs(simspaces_dir, exist_ok=True)

                logging.info(f"        Step 2c: Calculating Similarity Spaces...")
                cmd_calc_simspace = [
                    "python", "core_scripts/calculate_similarityspaces_exp.py",
                    "--chembl_mf_data_path", current_chembl_mf_filtered_path,
                    "--zinc_data_path", current_zinc_filtered_path,
                    "--target_ligands_unscaled_path_for_tsne", processed_target_ligands_repr_file, # For t-SNE co-embedding
                    "--simspace_dim", str(simspace_dim),
                    "--representation_type", repr_type,
                    "--target_id_name", target_id_name,
                    "--output_simspace_dir", simspaces_dir,
                    "--output_model_dir", models_dir,
                    "--rdkit_features_list_target_str", json.dumps(gs['rdkit_features_list_target']) # Pass as JSON string
                ]
                
                # Add DR method flags
                active_dr_methods = config["dimensionality_reduction_methods"]
                cmd_calc_simspace.append(f"--dr_method_pca={str('pca' in active_dr_methods)}")
                
                has_umap = any("umap" in k for k in active_dr_methods)
                cmd_calc_simspace.append(f"--dr_method_umap={str(has_umap)}")
                if has_umap:
                    for dr_key, dr_p in active_dr_methods.items():
                        if "umap" in dr_key:
                            cmd_calc_simspace.append(f"--umap_metric_to_run_{dr_p['metric']}=True")
                
                cmd_calc_simspace.append(f"--dr_method_tsne={str('tsne' in active_dr_methods)}")
                if 'tsne' in active_dr_methods:
                    cmd_calc_simspace.append(f"--tsne_perplexity={str(active_dr_methods['tsne']['perplexity'])}")
                    cmd_calc_simspace.append(f"--tsne_pca_components={str(gs['tsne_pca_components'])}")

                if not run_command(cmd_calc_simspace, f"Calc SimSpace ({repr_type}, dim{simspace_dim}) for {target_id_name}"):
                    logging.error(f"Similarity space calculation failed for ({repr_type}, dim{simspace_dim}). Skipping this dimension.")
                    continue

                # Comprehensive simspace CSV generated by the above script
                comprehensive_simspace_csv_path = os.path.join(simspaces_dir, f"{target_id_name}_{repr_type}_dim{simspace_dim}_similarity_space.csv")

                for dr_key, dr_params in active_dr_methods.items():
                    dr_short_name = dr_params["short_name"]
                    dr_short_name_fs = dr_short_name.replace('-', '_') # Filesystem friendly
                    
                    logging.info(f"          Step 2d: Projecting & Analyzing for DR: {dr_short_name}...")
                    
                    results_dir = os.path.join(target_workspace_dir, "results", repr_type, f"dim_{simspace_dim}", dr_short_name_fs)
                    os.makedirs(results_dir, exist_ok=True)

                    cmd_project_analyze = [
                        "python", "experimental_pipeline/project_and_analyze.py",
                        "--simspace_csv_path", comprehensive_simspace_csv_path,
                        "--dr_method_key", dr_key,
                        "--dr_short_name", dr_short_name,
                        "--simspace_dim", str(simspace_dim),
                        "--k_for_knn", ','.join(map(str, gs['k_for_knn_distance'])),
                        "--output_dir", results_dir,
                        "--target_id_name", target_id_name,
                        "--representation_type", repr_type, # Pass representation type
                        "--rdkit_features_list_target_str", json.dumps(gs['rdkit_features_list_target'])
                    ]
                    
                    model_name_root_for_proj = f"{target_id_name}_{repr_type}_dim{simspace_dim}" # Base for finding models

                    if dr_key == "tsne":
                        # t-SNE projections for target ligands were co-embedded and saved separately
                        tsne_target_proj_path = os.path.join(simspaces_dir, f"{model_name_root_for_proj}_tSNE_TARGET_PROJECTIONS.csv")
                        cmd_project_analyze.extend(["--precomputed_target_projections_path", tsne_target_proj_path])
                    else: # PCA or UMAP - need models and raw target ligand representations
                        cmd_project_analyze.extend([
                            "--target_ligands_repr_path", processed_target_ligands_repr_file,
                            "--model_dir_for_projection", models_dir,
                            "--model_name_root_for_projection", model_name_root_for_proj
                        ])
                    
                    if not run_command(cmd_project_analyze, f"Project/Analyze ({dr_short_name}, {repr_type}, dim{simspace_dim}) for {target_id_name}"):
                       logging.error(f"Projection & Analysis failed for ({dr_short_name}, {repr_type}, dim{simspace_dim}).")

    logging.info("\nStep 3: Generating final report...")
    cmd_generate_report = [
        "python", "reporting/generate_latex_report.py",
        "--experiment_run_dir", current_experiment_run_dir,
        "--config_path", config_path,
        "--output_dir", gs['final_report_dir']
    ]
    if not run_command(cmd_generate_report, "Report Generation"):
        logging.error("Report generation failed.")

    logging.info(f"===== Experimental pipeline finished for run: {timestamp} =====")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main orchestrator for UMMBAS similarity experiments.")
    parser.add_argument("--config", default="experiment_config.json", help="Path to the experiment configuration JSON file.")
    args = parser.parse_args()
    main(config_path=args.config)