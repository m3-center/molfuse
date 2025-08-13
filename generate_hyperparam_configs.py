import json
import os
import copy

# --- Configuration for the Sweep ---

# Base configuration file to use as a template
BASE_CONFIG_FILE = "experiment_config.json"

# Directory to save the generated config files
OUTPUT_DIR = "hyperparam_configs"

# Target protein to use for the sweep (must match an 'id_name' in the base config)
TARGET_FOR_SWEEP = "TyrosineProteinKinaseABL1_P00519"

# Fixed similarity space dimension for the sweep
SIMSPACE_DIM_FOR_SWEEP = [10]

# Hyperparameters to sweep
UMAP_N_NEIGHBORS_OPTIONS = [5, 15, 30, 60]
TSNE_PCA_COMPONENTS_OPTIONS = [25, 50, 100]

# DR methods to test for each representation
# This allows us to only test UMAP-Euclidean for Features and UMAP-Jaccard for Fingerprints
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
    
    # 1. Load the base configuration
    try:
        with open(BASE_CONFIG_FILE, 'r') as f:
            base_config = json.load(f)
        print(f"Loaded base configuration from: {BASE_CONFIG_FILE}")
    except Exception as e:
        print(f"ERROR: Could not load base config file '{BASE_CONFIG_FILE}'. Aborting. Error: {e}")
        return

    # 2. Create the output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory for configs: {OUTPUT_DIR}")

    # 3. Filter for the single target for the sweep
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
                
                # Create a deep copy to avoid modifying the original
                new_config = copy.deepcopy(base_config)
                
                # --- Modify config for this specific run ---
                # Set global settings
                new_config["global_settings"]["simspace_dims_to_test"] = SIMSPACE_DIM_FOR_SWEEP
                
                # Isolate the single target
                new_config["targets"] = target_config_list
                
                # Isolate the single representation
                new_config["representations"] = [repr_type]
                
                # Isolate the single DR method and set the hyperparameter
                original_dr_method = new_config["dimensionality_reduction_methods"][dr_method_key]
                original_dr_method["n_neighbors"] = n_neighbors # Add/overwrite the n_neighbors value
                new_config["dimensionality_reduction_methods"] = {
                    dr_method_key: original_dr_method
                }

                # Save the new config file
                filename = f"config_target_{TARGET_FOR_SWEEP}_{repr_type}_{dr_method_key}_n{n_neighbors}.json"
                filepath = os.path.join(OUTPUT_DIR, filename)
                with open(filepath, 'w') as f:
                    json.dump(new_config, f, indent=2)
                print(f"    -> Saved to {filepath}")

    # --- Generate t-SNE Sweep Configs ---
    print("\n--- Generating t-SNE tsne_pca_components sweep configs ---")
    repr_type = "fingerprints" # t-SNE sweep is only for fingerprints
    if "tsne" in SWEEP_CONFIG.get(repr_type, {}).get("dr_methods", []):
        for pca_comps in TSNE_PCA_COMPONENTS_OPTIONS:
            print(f"  Creating config for: {repr_type}, t-SNE, pca_components={pca_comps}")

            new_config = copy.deepcopy(base_config)

            # --- Modify config for this specific run ---
            # Set global settings
            new_config["global_settings"]["simspace_dims_to_test"] = SIMSPACE_DIM_FOR_SWEEP
            new_config["global_settings"]["tsne_pca_components"] = pca_comps # Set the PCA components
            
            # Isolate the single target
            new_config["targets"] = target_config_list

            # Isolate the single representation
            new_config["representations"] = [repr_type]
            
            # Isolate the single DR method
            original_tsne_method = new_config["dimensionality_reduction_methods"]["tsne"]
            new_config["dimensionality_reduction_methods"] = {
                "tsne": original_tsne_method
            }

            # Save the new config file
            filename = f"config_target_{TARGET_FOR_SWEEP}_{repr_type}_tsne_pca{pca_comps}.json"
            filepath = os.path.join(OUTPUT_DIR, filename)
            with open(filepath, 'w') as f:
                json.dump(new_config, f, indent=2)
            print(f"    -> Saved to {filepath}")

    print("\n--- Hyperparameter configuration generation complete! ---")


if __name__ == "__main__":
    generate_configs()