import json
import os
import copy

# --- Configuration for the Sweep ---

BASE_CONFIG_FILE = "experiment_config.json"
OUTPUT_DIR = "hyperparam_configs"
SWEEP_WORKSPACE_DIR = "experiment_workspace_hyperparam_sweep/"
SWEEP_REPORT_DIR = "final_report_hyperparam_sweep/"
TARGET_FOR_SWEEP = "TyrosineProteinKinaseABL1_P00519"

# --- Hyperparameter Sweep Definitions ---

# Define the final, fixed dimensionality for the output similarity space
# This keeps the comparison consistent across hyperparameter settings.
FIXED_DIMS = {
    "features": [2],
    "fingerprints": [10]
}

# Define the parameters to sweep for each method and representation
HYPERPARAM_PLAN = {
    "features": {
        "pca": {
            "sweep": False # No hyperparams to sweep, will run once
        },
        "umap_euclidean": {
            "sweep": True,
            "param_name": "n_neighbors",
            "values": [15, 30, 50]
        },
        "tsne": {
            "sweep": True,
            "param_name": "perplexity",
            "values": [15, 30, 50]
        }
    },
    "fingerprints": {
        "pca": {
            "sweep": False
        },
        "umap_euclidean": { # This UMAP runs on PCA-reduced data
            "sweep": True,
            "param_name": "n_neighbors",
            "values": [15, 30, 50]
        },
        "umap_jaccard": { # This UMAP runs on raw binary data
            "sweep": True,
            "param_name": "n_neighbors",
            "values": [30, 50, 100, 150]
        },
        "umap_hamming": { # This UMAP runs on raw binary data
            "sweep": True,
            "param_name": "n_neighbors",
            "values": [30, 50, 100, 150]
        },
        "tsne": {
            "sweep": True,
            "param_name": "tsne_pca_components",
            "values": [25, 50, 100, 150]
        }
    }
}


def generate_configs():
    """
    Generates JSON configuration files for a comprehensive hyperparameter sweep
    based on the HYPERPARAM_PLAN.
    """
    print("--- Starting Hyperparameter Config Generation ---")
    
    try:
        with open(BASE_CONFIG_FILE, 'r') as f:
            base_config = json.load(f)
        print(f"Loaded base configuration from: {BASE_CONFIG_FILE}")
    except Exception as e:
        print(f"ERROR: Could not load base config file '{BASE_CONFIG_FILE}'. Aborting. Error: {e}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory for generated configs: {OUTPUT_DIR}")

    # Isolate the single target configuration for the sweep
    target_config = next((t for t in base_config.get("targets", []) if t.get("id_name") == TARGET_FOR_SWEEP), None)
    if not target_config:
        print(f"ERROR: Target '{TARGET_FOR_SWEEP}' not found in base config. Aborting.")
        return

    # --- Main Generation Loop ---
    for repr_type, methods in HYPERPARAM_PLAN.items():
        print(f"\n--- Generating configs for '{repr_type}' representation ---")
        
        for dr_method_key, plan in methods.items():
            
            # Use a list of values if sweeping, or a list with a single None if not.
            # This allows us to use a single loop structure for both cases.
            values_to_iterate = plan.get("values", [None]) if plan["sweep"] else [None]
            param_name = plan.get("param_name")

            for value in values_to_iterate:
                new_config = copy.deepcopy(base_config)

                # --- 1. Set Global Settings ---
                new_config["global_settings"]["workspace_base_dir"] = SWEEP_WORKSPACE_DIR
                new_config["global_settings"]["final_report_dir"] = SWEEP_REPORT_DIR
                new_config["global_settings"]["simspace_dims_to_test"] = FIXED_DIMS[repr_type]
                
                # --- 2. Set Representation and Target ---
                new_config["representations"] = [repr_type]
                new_config["targets"] = [target_config]

                # --- 3. Configure the specific DR method ---
                original_dr_method = copy.deepcopy(base_config["dimensionality_reduction_methods"][dr_method_key])
                new_config["dimensionality_reduction_methods"] = { dr_method_key: original_dr_method }
                
                # --- 4. Apply the hyperparameter being swept ---
                filename_suffix = ""
                if plan["sweep"]:
                    print(f"  Creating config for: {dr_method_key}, {param_name}={value}")
                    filename_suffix = f"_{param_name.replace('_', '')}{value}"
                    
                    if param_name == "n_neighbors":
                        new_config["dimensionality_reduction_methods"][dr_method_key]["n_neighbors"] = value
                    elif param_name == "perplexity":
                        new_config["dimensionality_reduction_methods"][dr_method_key]["perplexity"] = value
                    elif param_name == "tsne_pca_components":
                        new_config["global_settings"]["tsne_pca_components"] = value
                else:
                    print(f"  Creating single config for: {dr_method_key}")

                # --- 5. Generate Filename and Save ---
                filename = f"config_{repr_type}_{dr_method_key}{filename_suffix}.json"
                filepath = os.path.join(OUTPUT_DIR, filename)
                with open(filepath, 'w') as f:
                    json.dump(new_config, f, indent=2)
                print(f"    -> Saved to {filepath}")

    print("\n--- Hyperparameter configuration generation complete! ---")
    print(f"All generated configs will output to base directory: '{SWEEP_WORKSPACE_DIR}'")

if __name__ == "__main__":
    generate_configs()