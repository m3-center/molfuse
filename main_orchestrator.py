import json
import os
import shutil
import subprocess
import pandas as pd
import logging
from datetime import datetime

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_command(command_list, step_name="Command"):
    """Runs a command and logs its output."""
    logging.info(f"Running {step_name}: {' '.join(command_list)}")
    try:
        process = subprocess.Popen(command_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        if process.returncode == 0:
            logging.info(f"{step_name} completed successfully.")
            if stdout: logging.debug(f"STDOUT:\n{stdout}")
            if stderr: logging.debug(f"STDERR:\n{stderr}") # Log stderr even on success for warnings
            return True
        else:
            logging.error(f"{step_name} failed with return code {process.returncode}.")
            if stdout: logging.error(f"STDOUT:\n{stdout}")
            if stderr: logging.error(f"STDERR:\n{stderr}")
            return False
    except Exception as e:
        logging.error(f"Exception during {step_name}: {e}")
        return False

def get_mf_keyword_id(mf_keywords_df, canonical_mf_name):
    """Looks up the KW-ID for a given canonical molecular function name."""
    row = mf_keywords_df[mf_keywords_df['Name'].str.lower() == canonical_mf_name.lower()]
    if not row.empty:
        return row.iloc[0]['ID']
    return None

def main(config_path="experiment_config.json"):
    logging.info(f"Starting experimental pipeline with config: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)

    gs = config['global_settings']
    workspace_base_dir = gs['workspace_base_dir']
    
    # Load MF keywords for mapping
    try:
        mf_keywords_df = pd.read_csv(gs['molecular_function_keywords_csv_path'])
    except FileNotFoundError:
        logging.error(f"Molecular function keywords file not found at: {gs['molecular_function_keywords_csv_path']}")
        return

    # Create a timestamped main experiment run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_experiment_run_dir = os.path.join(workspace_base_dir, f"run_{timestamp}")
    os.makedirs(current_experiment_run_dir, exist_ok=True)
    logging.info(f"Main experiment run directory: {current_experiment_run_dir}")

    for target_info in config['targets']:
        target_id_name = target_info['id_name']
        target_uniprot_id = target_info['uniprot_id']
        canonical_mf_name = target_info['molecular_function_canonical_name']
        
        logging.info(f"\nProcessing Target: {target_info['display_name']} ({target_id_name})")

        # Get KW-ID for the molecular function
        mf_kw_id = get_mf_keyword_id(mf_keywords_df, canonical_mf_name)
        if not mf_kw_id:
            logging.error(f"Could not find KW-ID for MF: '{canonical_mf_name}' for target {target_id_name}. Skipping target.")
            continue
        
        # Use the new 'molecular_function_filename_segment' from config
        mf_filename_segment = target_info.get('molecular_function_filename_segment', canonical_mf_name.replace(' ', '_').replace('/', '_')) # Fallback if new field not present
        mf_affinity_filename = f"{mf_kw_id}_{mf_filename_segment}_affinity.csv" # Corrected construction
        mf_affinity_full_path = os.path.join(gs['molecular_function_affinity_data_dir'], mf_affinity_filename)

        if not os.path.exists(mf_affinity_full_path):
            logging.error(f"Molecular function affinity file not found: {mf_affinity_full_path} (using segment '{mf_filename_segment}'). Skipping target.")
            continue

        target_workspace_dir = os.path.join(current_experiment_run_dir, target_id_name)
        temp_data_dir = os.path.join(target_workspace_dir, "temp_data")
        models_base_dir = os.path.join(target_workspace_dir, "models") # Base for models per representation/dim/dr
        simspaces_base_dir = os.path.join(target_workspace_dir, "similarity_spaces")
        results_base_dir = os.path.join(target_workspace_dir, "results")

        os.makedirs(temp_data_dir, exist_ok=True)
        os.makedirs(models_base_dir, exist_ok=True)
        os.makedirs(simspaces_base_dir, exist_ok=True)
        os.makedirs(results_base_dir, exist_ok=True)

        # --- 1. Data Preparation ---
        logging.info("Step 1: Preparing data (filtering pre-calculated files and extracting raw target ligands)...")
        cmd_prepare_data = [
            "python", "experimental_pipeline/prepare_data.py",
            "--config_path", config_path,
            "--target_id_name", target_id_name,
            # No longer pass mf_affinity_path_resolved, prepare_data handles finding precalc files
            "--output_dir", temp_data_dir
        ]
        if not run_command(cmd_prepare_data, "Data Preparation"):
            logging.error(f"Data preparation failed for {target_id_name}. Skipping target.")
            continue
        
        # Path to the RAW target ligands (SMILES, IDs only) for separate feature calculation
        raw_target_ligands_for_feature_calc_csv = os.path.join(temp_data_dir, f"{target_id_name}_target_ligands_for_feature_calc_raw.csv")

        # Paths to the FILTERED PRE-CALCULATED files
        filtered_chembl_mf_features_csv = os.path.join(temp_data_dir, f"{target_id_name}_chembl_mf_excluded_features.csv")
        filtered_chembl_mf_fingerprints_csv = os.path.join(temp_data_dir, f"{target_id_name}_chembl_mf_excluded_fingerprints.csv")
        filtered_zinc_features_csv = os.path.join(temp_data_dir, f"{target_id_name}_zinc_excluded_features.csv")
        filtered_zinc_fingerprints_csv = os.path.join(temp_data_dir, f"{target_id_name}_zinc_excluded_fingerprints.csv")

        # --- 2. Loop through Representations (Features/Fingerprints) ---
        for repr_type in config['representations']:
            logging.info(f"  Processing Representation: {repr_type}")
            
            # --- 2a. Calculate Features/Fingerprints ONLY FOR TARGET LIGANDS ---
            logging.info(f"    Step 2a: Calculating {repr_type} for TARGET LIGANDS...")
            target_ligands_repr_output_dir = os.path.join(temp_data_dir, repr_type, "target_ligands_processed") # Specific subdir
            os.makedirs(target_ligands_repr_output_dir, exist_ok=True)

            cmd_molcalcs_target = [
                "python", "core_scripts/calculate_features_and_fingerprints_exp.py",
                "--input_csv", raw_target_ligands_for_feature_calc_csv, # Use the raw SMILES file
                "--output_dir", target_ligands_repr_output_dir,
                "--representation_type", repr_type,
                "--file_label", f"{target_id_name}_target_ligands_project", # Distinct label
                "--n_jobs", str(gs['n_jobs_molcalcs'])
            ]
            if not run_command(cmd_molcalcs_target, f"MolCalcs for TARGET LIGANDS ({repr_type})"):
                logging.error(f"Feature/Fingerprint calculation failed for TARGET LIGANDS ({repr_type}). Skipping this representation for {target_id_name}.")
                continue 

            # Path to the processed target ligands' features/fingerprints
            processed_target_ligands_repr_file = os.path.join(target_ligands_repr_output_dir, f"{target_id_name}_target_ligands_project_{repr_type}.csv")

            # Determine paths for pre-calculated, filtered ChEMBL MF and ZINC data for this repr_type
            current_chembl_mf_filtered_path = filtered_chembl_mf_features_csv if repr_type == "features" else filtered_chembl_mf_fingerprints_csv
            current_zinc_filtered_path = filtered_zinc_features_csv if repr_type == "features" else filtered_zinc_fingerprints_csv
            
            # --- 2b. Loop through SIMSPACE_DIM values ---
            for simspace_dim in gs['simspace_dims_to_test']:
                logging.info(f"      Processing SIMSPACE_DIM: {simspace_dim}")

                # --- 2c. Calculate Similarity Spaces (iterates DR methods internally) ---
                # `calculate_similarityspaces_exp.py` will handle different DR methods.
                # It needs the Chembl MF (excluded) and ZINC (excluded) files of current repr_type.
                
                current_models_dir = os.path.join(models_base_dir, repr_type, f"dim_{simspace_dim}")
                current_simspaces_dir = os.path.join(simspaces_base_dir, repr_type, f"dim_{simspace_dim}")
                os.makedirs(current_models_dir, exist_ok=True)
                os.makedirs(current_simspaces_dir, exist_ok=True)

                cmd_calc_simspace = [
                    "python", "core_scripts/calculate_similarityspaces_exp.py",
                    "--chembl_mf_data_path", current_chembl_mf_filtered_path, # Use filtered pre-calc
                    "--zinc_data_path", current_zinc_filtered_path,           # Use filtered pre-calc
                    "--simspace_dim", str(simspace_dim),
                    "--representation_type", repr_type,
                    "--target_id_name", target_id_name, # For naming models and output files
                    "--output_simspace_dir", current_simspaces_dir,
                    "--output_model_dir", current_models_dir
                ]
                # Add all DR methods from config to the command
                for dr_key, dr_params in config["dimensionality_reduction_methods"].items():
                    cmd_calc_simspace.extend([f"--dr_method_{dr_key}", "True"]) # e.g., --dr_method_pca True
                    if "metric" in dr_params: # For UMAP
                        cmd_calc_simspace.extend([f"--umap_metric_{dr_key.split('_')[-1]}", dr_params["metric"]])


                if not run_command(cmd_calc_simspace, f"Similarity Space Calculation (dim {simspace_dim}, {repr_type})"):
                    logging.error(f"Similarity space calculation failed for dim {simspace_dim}, {repr_type}. Skipping this dimension.")
                    continue

                # --- 2d. Projection & Analysis (Loop for each DR method's output) ---
                # After calculate_similarityspaces_exp.py runs, models and simspace CSVs are created.
                # We need to iterate through each DR method that was processed.
                for dr_key, dr_params in config["dimensionality_reduction_methods"].items():
                    dr_short_name = dr_params["short_name"] # e.g., PCA, UMAP-Euclidean
                    logging.info(f"        Step 2d: Projecting & Analyzing for DR: {dr_short_name} (dim {simspace_dim}, {repr_type})")
                    
                    # Construct paths to the specific model and simspace files
                    # Naming convention needs to be consistent with calculate_similarityspaces_exp.py
                    model_name_root = f"{target_id_name}_{repr_type}_dim{simspace_dim}_{dr_short_name.replace('-', '_')}"
                    
                    # Path to the specific DR model (e.g., .../PCA.lzma or .../euclidean_UMAP.lzma)
                    # This needs careful construction based on how calculate_similarityspaces_exp.py saves them.
                    # For now, assume a general model dir and project_and_analyze.py figures out the exact model file.
                    
                    simspace_csv_path = os.path.join(current_simspaces_dir, f"{model_name_root}_similarity_space.csv")
                    
                    if not os.path.exists(simspace_csv_path):
                        logging.warning(f"Simspace CSV not found: {simspace_csv_path}. Skipping projection for {dr_short_name}.")
                        continue

                    current_results_dir = os.path.join(results_base_dir, repr_type, f"dim_{simspace_dim}", dr_short_name.replace('-', '_'))
                    os.makedirs(current_results_dir, exist_ok=True)

                    cmd_project_analyze = [
                        "python", "experimental_pipeline/project_and_analyze.py",
                        "--target_ligands_repr_path", processed_target_ligands_repr_file, # Use the newly featurized target ligands
                        "--simspace_csv_path", simspace_csv_path,
                        "--model_dir_for_projection", current_models_dir, # Dir containing scaler and specific DR model
                        "--model_name_root_for_projection", model_name_root, # To help find the right scaler/DR model
                        "--dr_method_key", dr_key, # e.g. "pca", "umap_euclidean"
                        "--dr_short_name", dr_short_name, # e.g. "PCA", "UMAP-Euclidean"
                        "--simspace_dim", str(simspace_dim),
                        "--k_for_knn", ','.join(map(str,gs['k_for_knn_distance'])),
                        "--output_dir", current_results_dir,
                        "--target_id_name", target_id_name
                    ]
                    if not run_command(cmd_project_analyze, f"Projection & Analysis ({dr_short_name}, dim {simspace_dim}, {repr_type})"):
                        logging.error(f"Projection & Analysis failed for {dr_short_name}, dim {simspace_dim}, {repr_type}.")
                        # Continue to next DR method or dim

            # End of simspace_dim loop
        # End of repr_type loop
    # End of target_info loop

    # --- 3. Reporting ---
    logging.info("\nStep 3: Generating report...")
    cmd_generate_report = [
        "python", "reporting/generate_latex_report.py",
        "--experiment_run_dir", current_experiment_run_dir, # Pass the timestamped run directory
        "--config_path", config_path,
        "--output_dir", gs['final_report_dir']
    ]
    if not run_command(cmd_generate_report, "Report Generation"):
        logging.error("Report generation failed.")

    logging.info("Experimental pipeline finished.")

if __name__ == "__main__":
    # Example: python main_orchestrator.py --config_path my_custom_config.json
    # For now, using default config_path
    main()