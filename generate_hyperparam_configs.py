import json
import os
import copy

# --- Configuration for the Sweep ---

BASE_CONFIG_FILE = "experiment_config.json"
OUTPUT_DIR = "hyperparam_configs"
SWEEP_WORKSPACE_DIR = "experiment_workspace_hyperparam_sweep/"
SWEEP_REPORT_DIR = "final_report_hyperparam_sweep_isocitrate/"
TARGET_FOR_SWEEP = "IsocitrateDehydrogenaseNADP_O75874" # "PyruvateKinaseM2_P14618" # "TyrosineProteinKinaseABL1_P00519"

# Static, fixed number of PCA components for pre-processing fingerprints
# This is now a fixed part of the pipeline, not a hyperparameter to be swept.
FIXED_FINGERPRINT_PCA_COMPONENTS = 50

FIXED_DIMS = {
    "features": [2],
    "fingerprints": [2]
}

HYPERPARAM_PLAN = {
    "features": {
        # "pca": {"sweep": False},
        "umap_euclidean": {
            "sweep": True,
            "param_name": "n_neighbors",
            "values": [2, 3, 4]
            # "values": [5, 10, 20]
        } 
        # ,
        # "tsne": {
        #     "sweep": True,
        #     "param_name": "perplexity", # Now sweeping perplexity for features
        #     # "values": [5, 10, 15, 30, 50, 100, 150, 200, 250, 500, 750, 1000, 1250, 1500]
        #     "values": [5, 10]
        # }
    }
}

def generate_configs():
    print("--- Starting Hyperparameter Config Generation ---")
    
    try:
        with open(BASE_CONFIG_FILE, 'r') as f:
            base_config = json.load(f)
    except Exception as e:
        print(f"ERROR: Could not load base config file '{BASE_CONFIG_FILE}'. Aborting. Error: {e}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    target_config = next((t for t in base_config.get("targets", []) if t.get("id_name") == TARGET_FOR_SWEEP), None)
    if not target_config:
        print(f"ERROR: Target '{TARGET_FOR_SWEEP}' not found. Aborting.")
        return

    for repr_type, methods in HYPERPARAM_PLAN.items():
        print(f"\n--- Generating configs for '{repr_type}' representation ---")
        
        for dr_method_key, plan in methods.items():
            values_to_iterate = plan.get("values", [None]) if plan["sweep"] else [None]
            param_name = plan.get("param_name")

            for value in values_to_iterate:
                new_config = copy.deepcopy(base_config)

                # Set Global, Representation, and Target settings
                new_config["global_settings"]["workspace_base_dir"] = SWEEP_WORKSPACE_DIR
                new_config["global_settings"]["final_report_dir"] = SWEEP_REPORT_DIR
                new_config["global_settings"]["simspace_dims_to_test"] = FIXED_DIMS[repr_type]
                new_config["representations"] = [repr_type]
                new_config["targets"] = [target_config]
                
                # If it's a fingerprint run, add the fixed PCA components setting
                if repr_type == "fingerprints":
                    new_config["global_settings"]["fingerprint_pca_components"] = FIXED_FINGERPRINT_PCA_COMPONENTS

                # Configure the specific DR method
                original_dr_method = copy.deepcopy(base_config["dimensionality_reduction_methods"][dr_method_key])
                new_config["dimensionality_reduction_methods"] = { dr_method_key: original_dr_method }
                
                filename_suffix = ""
                if plan["sweep"]:
                    print(f"  Creating config for: {dr_method_key}, {param_name}={value}")
                    filename_suffix = f"_{param_name.replace('_', '')}{value}"
                    
                    # Apply the hyperparameter being swept to the correct location
                    if param_name in ["n_neighbors", "perplexity"]:
                        new_config["dimensionality_reduction_methods"][dr_method_key][param_name] = value
                else:
                    print(f"  Creating single config for: {dr_method_key}")

                filename = f"config_{repr_type}_{dr_method_key}{filename_suffix}.json"
                filepath = os.path.join(OUTPUT_DIR, filename)
                with open(filepath, 'w') as f:
                    json.dump(new_config, f, indent=2)
                print(f"    -> Saved to {filepath}")

    print("\n--- Hyperparameter configuration generation complete! ---")

if __name__ == "__main__":
    generate_configs()
