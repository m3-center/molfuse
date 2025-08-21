import json
import os
import copy

# --- Configuration for the Sweep ---

# Base configuration file to use as a template
BASE_CONFIG_FILE = "experiment_config.json"

# Directory to save the generated config files
OUTPUT_DIR = "hyperparam_configs"

# Define dedicated output directories for the sweep
SWEEP_WORKSPACE_DIR = "experiment_workspace_hyperparam_sweep/"
SWEEP_REPORT_DIR = "final_report_hyperparam_sweep/"

# Target protein to use for the sweep (must match an 'id_name' in the base config)
TARGET_FOR_SWEEP = "TyrosineProteinKinaseABL1_P00519"

BIT_SELECTION_CSV_PATH = "fingerprint_variance_analysis/fingerprint_bit_variance_sorted.csv"
NUM_BITS_TO_USE = 2048

# --- MODIFIED: Hyperparameters and Dimensions to sweep ---
# Fixed similarity space dimension for the UMAP sweep
UMAP_SIMSPACE_DIM_FOR_SWEEP = [10]
# Fixed similarity space dimension for the t-SNE sweep
TSNE_SIMSPACE_DIM_FOR_SWEEP = [2]

# Reduced n_neighbors options for UMAP
UMAP_N_NEIGHBORS_OPTIONS = [5, 15, 30]
# Kept tsne_pca_components options the same
TSNE_PCA_COMPONENTS_OPTIONS = [25, 50, 75]
# --- END MODIFICATION ---

# DR methods to test for each representation
SWEEP_CONFIG = {
    "features": {
        "dr_methods": ["umap_euclidean"]
    },
    "fingerprints": {
        "dr_methods": ["umap_jaccard", "tsne"]
    }
}


def generate_configs():
    """
    Generates JSON configuration files for a targeted hyperparameter sweep.
    """
    print(f"--- Starting Hyperparameter Config Generation ---")
    
    try:
        with open(BASE_CONFIG_FILE, 'r') as f:
            base_config = json.load(f)
        print(f"Loaded base configuration from: {BASE_CONFIG_FILE}")
    except Exception as e:
        print(f"ERROR: Could not load base config file '{BASE_CONFIG_FILE}'. Aborting. Error: {e}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory for generated configs: {OUTPUT_DIR}")

    target_config_list = [t for t in base_config.get("targets", []) if t.get("id_name") == TARGET_FOR_SWEEP]
    if not target_config_list:
        print(f"ERROR: Target '{TARGET_FOR_SWEEP}' not found in base config's target list. Aborting.")
        return
    
    # --- Generate UMAP Sweep Configs ---
    print("\n--- Generating UMAP n_neighbors sweep configs ---")
    for repr_type, settings in SWEEP_CONFIG.items():
        for dr_method_key in settings["dr_methods"]:
            if not dr_method_key.startswith("umap"):
                continue
            
            for n_neighbors in UMAP_N_NEIGHBORS_OPTIONS:
                print(f"  Creating config for: {repr_type}, {dr_method_key}, n_neighbors={n_neighbors}")
                
                new_config = copy.deepcopy(base_config)
                
                # Modify config for this specific UMAP run
                new_config["global_settings"]["simspace_dims_to_test"] = UMAP_SIMSPACE_DIM_FOR_SWEEP
                new_config["global_settings"]["workspace_base_dir"] = SWEEP_WORKSPACE_DIR
                new_config["global_settings"]["final_report_dir"] = SWEEP_REPORT_DIR
                
                # --- NEW: Add bit selection settings if it's a fingerprints run ---
                if repr_type == "fingerprints":
                    new_config["global_settings"]["fingerprint_bit_selection_csv_path"] = BIT_SELECTION_CSV_PATH
                    new_config["global_settings"]["num_fingerprint_bits_to_use"] = NUM_BITS_TO_USE
                # --- END NEW ---

                new_config["targets"] = target_config_list
                new_config["representations"] = [repr_type]
                
                original_dr_method = copy.deepcopy(base_config["dimensionality_reduction_methods"][dr_method_key])
                original_dr_method["n_neighbors"] = n_neighbors
                new_config["dimensionality_reduction_methods"] = {
                    dr_method_key: original_dr_method
                }

                filename = f"config_target_{TARGET_FOR_SWEEP}_{repr_type}_{dr_method_key}_n{n_neighbors}.json"
                filepath = os.path.join(OUTPUT_DIR, filename)
                with open(filepath, 'w') as f:
                    json.dump(new_config, f, indent=2)
                print(f"    -> Saved to {filepath}")

    # --- Generate t-SNE Sweep Configs ---
    print("\n--- Generating t-SNE tsne_pca_components sweep configs ---")
    repr_type = "fingerprints"
    if "tsne" in SWEEP_CONFIG.get(repr_type, {}).get("dr_methods", []):
        for pca_comps in TSNE_PCA_COMPONENTS_OPTIONS:
            print(f"  Creating config for: {repr_type}, t-SNE, pca_components={pca_comps}")

            new_config = copy.deepcopy(base_config)

            # Modify config for this specific t-SNE run
            new_config["global_settings"]["simspace_dims_to_test"] = TSNE_SIMSPACE_DIM_FOR_SWEEP
            new_config["global_settings"]["tsne_pca_components"] = pca_comps
            new_config["global_settings"]["workspace_base_dir"] = SWEEP_WORKSPACE_DIR
            new_config["global_settings"]["final_report_dir"] = SWEEP_REPORT_DIR

            # --- NEW: Add bit selection settings for this fingerprints run ---
            new_config["global_settings"]["fingerprint_bit_selection_csv_path"] = BIT_SELECTION_CSV_PATH
            new_config["global_settings"]["num_fingerprint_bits_to_use"] = NUM_BITS_TO_USE
            # --- END NEW ---
            
            new_config["targets"] = target_config_list
            new_config["representations"] = [repr_type]
            
            original_tsne_method = copy.deepcopy(base_config["dimensionality_reduction_methods"]["tsne"])
            new_config["dimensionality_reduction_methods"] = {
                "tsne": original_tsne_method
            }

            filename = f"config_target_{TARGET_FOR_SWEEP}_{repr_type}_tsne_pca{pca_comps}.json"
            filepath = os.path.join(OUTPUT_DIR, filename)
            with open(filepath, 'w') as f:
                json.dump(new_config, f, indent=2)
            print(f"    -> Saved to {filepath}")

    print("\n--- Hyperparameter configuration generation complete! ---")
    print(f"All generated configs will output to base directory: '{SWEEP_WORKSPACE_DIR}'")
    print(f"Fingerprint-based configs will use the top {NUM_BITS_TO_USE} bits from '{BIT_SELECTION_CSV_PATH}'")


if __name__ == "__main__":
    generate_configs()