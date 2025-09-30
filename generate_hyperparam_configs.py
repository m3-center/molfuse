import json
import os
import copy

# --- Configuration for the NEW Hyperparameter Sweep ---

BASE_CONFIG_FILE = "experiment_config.json"
OUTPUT_DIR = "hyperparam_configs"
SWEEP_WORKSPACE_DIR = "experiment_workspace_rerun_hyperparam_sweep/"
SWEEP_REPORT_DIR = "final_report_rerun_hyperparam_sweep/"
TARGET_FOR_SWEEP = "TyrosineProteinKinaseABL1_P00519"

# --- NEW, COMPREHENSIVE HYPERPARAMETER DEFINITIONS ---
FIXED_SIMSPACE_DIM = 2
FIXED_FINGERPRINT_PCA_COMPONENTS = 325

TSNE_PERPLEXITY_VALUES = [15, 30, 50, 100, 500, 1000]
UMAP_N_NEIGHBORS_VALUES = [10, 100, 500]
UMAP_MIN_DIST_VALUES = [0.1, 0.25, 0.5]

def generate_configs():
    """
    Generates JSON configuration files for a comprehensive hyperparameter sweep,
    including a grid search for UMAP. Now controls strategy via a flag in the config.
    """
    print("--- Starting NEW Hyperparameter Config Generation ---")
    
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

    for repr_type in ["features", "fingerprints"]:
        print(f"\n--- Generating configs for '{repr_type}' representation ---")

        # --- Generate PCA Configs ---
        for strategy in ["projection", "coembedding"]:
            new_config = copy.deepcopy(base_config)
            new_config["global_settings"]["workspace_base_dir"] = SWEEP_WORKSPACE_DIR
            new_config["global_settings"]["final_report_dir"] = SWEEP_REPORT_DIR
            new_config["global_settings"]["simspace_dims_to_test"] = [FIXED_SIMSPACE_DIM]
            new_config["representations"] = [repr_type]
            new_config["targets"] = [target_config]
            new_config["dimensionality_reduction_methods"] = {"pca": base_config["dimensionality_reduction_methods"]["pca"]}
            if repr_type == "fingerprints":
                new_config["global_settings"]["fingerprint_pca_components"] = FIXED_FINGERPRINT_PCA_COMPONENTS
            
            new_config["global_settings"]["run_coembedding_for_pca_umap"] = (strategy == "coembedding")

            filename = f"config_{repr_type}_pca_{strategy}.json"
            filepath = os.path.join(OUTPUT_DIR, filename)
            with open(filepath, 'w') as f: json.dump(new_config, f, indent=2)
            print(f"  -> Saved {strategy} config to {filepath}")

        # --- Generate t-SNE Configs ---
        for perplexity in TSNE_PERPLEXITY_VALUES:
            new_config = copy.deepcopy(base_config)
            new_config["global_settings"]["workspace_base_dir"] = SWEEP_WORKSPACE_DIR
            new_config["global_settings"]["final_report_dir"] = SWEEP_REPORT_DIR
            new_config["global_settings"]["simspace_dims_to_test"] = [FIXED_SIMSPACE_DIM]
            new_config["representations"] = [repr_type]
            new_config["targets"] = [target_config]
            tsne_cfg = copy.deepcopy(base_config["dimensionality_reduction_methods"]["tsne"])
            tsne_cfg["perplexity"] = perplexity
            new_config["dimensionality_reduction_methods"] = {"tsne": tsne_cfg}
            if repr_type == "fingerprints":
                new_config["global_settings"]["fingerprint_pca_components"] = FIXED_FINGERPRINT_PCA_COMPONENTS

            filename = f"config_{repr_type}_tsne_perplexity{perplexity}.json"
            filepath = os.path.join(OUTPUT_DIR, filename)
            with open(filepath, 'w') as f: json.dump(new_config, f, indent=2)
            print(f"  -> Saved config to {filepath}")
            
        # --- Generate UMAP Configs (Grid Search) ---
        umap_methods = ["umap_euclidean"]
        # if repr_type == "fingerprints": umap_methods.extend(["umap_jaccard", "umap_hamming"])
        if repr_type == "fingerprints": umap_methods.extend(["umap_jaccard"])

        for umap_key in umap_methods:
            for n_neighbors in UMAP_N_NEIGHBORS_VALUES:
                for min_dist in UMAP_MIN_DIST_VALUES:
                    for strategy in ["projection", "coembedding"]:
                        new_config = copy.deepcopy(base_config)
                        new_config["global_settings"]["workspace_base_dir"] = SWEEP_WORKSPACE_DIR
                        new_config["global_settings"]["final_report_dir"] = SWEEP_REPORT_DIR
                        new_config["global_settings"]["simspace_dims_to_test"] = [FIXED_SIMSPACE_DIM]
                        new_config["representations"] = [repr_type]
                        new_config["targets"] = [target_config]
                        umap_cfg = copy.deepcopy(base_config["dimensionality_reduction_methods"][umap_key])
                        umap_cfg["n_neighbors"] = n_neighbors
                        umap_cfg["min_dist"] = min_dist
                        new_config["dimensionality_reduction_methods"] = {umap_key: umap_cfg}
                        if repr_type == "fingerprints":
                            new_config["global_settings"]["fingerprint_pca_components"] = FIXED_FINGERPRINT_PCA_COMPONENTS
                        
                        new_config["global_settings"]["run_coembedding_for_pca_umap"] = (strategy == "coembedding")

                        filename = f"config_{repr_type}_{umap_key}_{strategy}_nn{n_neighbors}_md{min_dist}.json"
                        filepath = os.path.join(OUTPUT_DIR, filename)
                        with open(filepath, 'w') as f: json.dump(new_config, f, indent=2)
                        print(f"  -> Saved {strategy} config to {filepath}")

    print("\n--- RERUN Hyperparameter configuration generation complete! ---")

if __name__ == "__main__":
    generate_configs()