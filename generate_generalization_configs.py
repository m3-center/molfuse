import json
import os
import copy

# --- Configuration for the Generalization Experiment ---

BASE_CONFIG_FILE = "experiment_config.json"
OUTPUT_DIR = "generalization_configs"
SWEEP_WORKSPACE_DIR = "experiment_workspace_generalization/"
SWEEP_REPORT_DIR = "final_report_generalization/"

TARGETS_TO_RUN = [
    {"id_name": "TyrosineProteinKinaseABL1_P00519", "display_name": "ABL1 Kinase"},
    {"id_name": "PyruvateKinaseM2_P14618", "display_name": "PKM2"},
    {"id_name": "IsocitrateDehydrogenaseNADP_O75874", "display_name": "IDH1"}
]

BEST_HYPERPARAMS = {
    "tsne": { "perplexity": 1250.0 },
    "umap_euclidean_projection": { "n_neighbors": 500.0 },
    "umap_euclidean_coembedding": { "n_neighbors": 200.0 }
}

METHODS_TO_RUN = {
    "tsne": BEST_HYPERPARAMS["tsne"],
    "umap_euclidean": {}
}

def generate_configs():
    print("--- Starting Generalization Config Generation ---")
    
    try:
        with open(BASE_CONFIG_FILE, 'r') as f:
            base_config = json.load(f)
    except Exception as e:
        print(f"ERROR: Could not load base config file '{BASE_CONFIG_FILE}'. Aborting. Error: {e}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # --- This experiment is only for features ---
    REPR_TYPE = "features"

    for target_info in TARGETS_TO_RUN:
        target_id = target_info["id_name"]
        print(f"\n--- Generating configs for Target: '{target_id}' ---")
        
        for dr_method_key, hyperparams in METHODS_TO_RUN.items():
            
            new_config = copy.deepcopy(base_config)

            new_config["global_settings"]["workspace_base_dir"] = SWEEP_WORKSPACE_DIR
            new_config["global_settings"]["final_report_dir"] = SWEEP_REPORT_DIR
            new_config["global_settings"]["simspace_dims_to_test"] = [2]
            
            new_config["representations"] = [REPR_TYPE]
            base_target_config = next((t for t in base_config.get("targets", []) if t.get("id_name") == target_id), None)
            if not base_target_config:
                print(f"  WARNING: Target '{target_id}' not found in base config. Skipping.")
                continue
            new_config["targets"] = [base_target_config]

            original_dr_method = copy.deepcopy(base_config["dimensionality_reduction_methods"][dr_method_key])
            
            if "perplexity" in hyperparams:
                original_dr_method["perplexity"] = hyperparams["perplexity"]
            
            if dr_method_key == "umap_euclidean":
                # Create Projection config
                proj_config = copy.deepcopy(new_config)
                proj_method_cfg = copy.deepcopy(original_dr_method)
                proj_method_cfg["n_neighbors"] = BEST_HYPERPARAMS["umap_euclidean_projection"]["n_neighbors"]
                proj_config["dimensionality_reduction_methods"] = {"umap_euclidean": proj_method_cfg}
                
                # --- FIX: Add representation type to filename ---
                filename_proj = f"config_{REPR_TYPE}_{target_id}_UMAP_projection.json"
                filepath_proj = os.path.join(OUTPUT_DIR, filename_proj)
                with open(filepath_proj, 'w') as f: json.dump(proj_config, f, indent=2)
                print(f"  -> Saved Projection config to {filepath_proj}")

                # Create Co-embedding config
                coembed_config = copy.deepcopy(new_config)
                coembed_method_cfg = copy.deepcopy(original_dr_method)
                coembed_method_cfg["n_neighbors"] = BEST_HYPERPARAMS["umap_euclidean_coembedding"]["n_neighbors"]
                coembed_config["dimensionality_reduction_methods"] = {"umap_euclidean": coembed_method_cfg}
                
                # --- FIX: Add representation type to filename ---
                filename_coembed = f"config_{REPR_TYPE}_{target_id}_UMAP_coembedding.json"
                filepath_coembed = os.path.join(OUTPUT_DIR, filename_coembed)
                with open(filepath_coembed, 'w') as f: json.dump(coembed_config, f, indent=2)
                print(f"  -> Saved Co-embedding config to {filepath_coembed}")
                
            else: # For PCA and t-SNE
                new_config["dimensionality_reduction_methods"] = {dr_method_key: original_dr_method}
                # --- FIX: Add representation type to filename ---
                filename = f"config_{REPR_TYPE}_{target_id}_{dr_method_key}.json"
                filepath = os.path.join(OUTPUT_DIR, filename)
                with open(filepath, 'w') as f: json.dump(new_config, f, indent=2)
                print(f"  -> Saved config to {filepath}")

    print("\n--- Generalization configuration generation complete! ---")

if __name__ == "__main__":
    generate_configs()