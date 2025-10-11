#!/usr/bin/env python3
"""
Generate configuration files for the dimensionality experiment.

This script creates configs for ABL1 using the best-performing methods and hyperparameters
from Experiment 1, testing different dimensionalities [2, 3, 5, 10, 20]. Only co-embedding
strategy is used as specified.

The goal is to understand how the dimensionality of the similarity space affects 
the performance of different dimensionality reduction methods.
"""

import json
import os
from pathlib import Path

# Output directory for dimensionality configs
OUTPUT_DIR = Path("dimensionality_configs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Global settings - same as in hyperparameter sweep
GLOBAL_SETTINGS = {
    "chembl_db_path": "datasets/chembl/chembl_35.db",
    "chembl_affinity_full_csv_path": "datasets/chembl/chembl_35_affinity_data.csv",
    "chembl_target_mapping_csv_path": "datasets/chembl/chembl_35_target_mapping.csv",
    "zinc_full_csv_path": "datasets/zinc_data.csv",
    "molecular_function_keywords_csv_path": "datasets/protein_collection/molecular_function_keywords.csv",
    "precalculated_chembl_mf_features_base_dir": "datasets/molecular_function_features_fingerprints/",
    "precalculated_chembl_mf_fingerprints_base_dir": "datasets/molecular_function_features_fingerprints/",
    "precalculated_zinc_features_path": "datasets/molecular_function_features_fingerprints/zinc/zinc_acquirable_extracted_features.csv",
    "precalculated_zinc_fingerprints_path": "datasets/molecular_function_features_fingerprints/zinc/zinc_acquirable_extracted_fingerprints_ECFP4.csv",
    "simspace_dims_to_test": [2, 3, 5, 10, 20],  # Testing multiple dimensions
    "k_for_knn_distance": [3, 5],
    "tsne_pca_components": 50,
    "affinity_cutoff_nM": 100000,
    "workspace_base_dir": "experiment_workspace_dimensionality/",  # UNIQUE WORKSPACE
    "final_report_dir": "final_report_dimensionality/",
    "n_jobs_molcalcs": -1,
    "rdkit_features_list_target": [
        "DipoleMoment", "ABC", "nAcid", "nBase", "nAromAtom", "nAtom", "nH", "nC", "nN", "nO", "nS",
        "nP", "nX", "nBonds", "nBondsO", "nBondsS", "nBondsD", "nBondsT", "nBondsA", "nBondsM",
        "nBondsKS", "nBondsKD", "EState_VSA7", "nHBAcc", "nHBDon", "Lipinski", "apol", "bpol",
        "nRing", "n3Ring", "n4Ring", "n5Ring", "n6Ring", "n7Ring", "n8Ring", "nRot", "Diameter",
        "TopoShapeIndex", "Vabc", "MW"
    ],
    "run_coembedding_for_pca_umap": True  # Only co-embedding for this experiment
}

# ABL1 target (same as hyperparameter sweep)
TARGET = {
    "id_name": "TyrosineProteinKinaseABL1_P00519",
    "display_name": "Tyrosine-protein Kinase ABL1",
    "uniprot_id": "P00519",
    "molecular_function_canonical_name": "Protein kinase inhibitor",
    "molecular_function_filename_segment": "Protein_kinase_inhibitor",
    "molecular_function_display_name": "Protein kinase inhibitor"
}

# Only features representation (best performing)
REPRESENTATIONS = ["features"]

# Dimensionality reduction methods with BEST hyperparameters from ABL1 analysis
# Only co-embedding strategy as specified
DR_METHODS = {
    "pca_coembedding": {
        "method_key": "pca",
        "config": {
            "short_name": "PCA",
            "allow_coembedding": True
        },
        "strategy": "coembedding"
    },
    "tsne_coembedding": {
        "method_key": "tsne",
        "config": {
            "short_name": "t-SNE",
            "perplexity": 1000,  # Best hyperparameter from Experiment 1
            "allow_coembedding": True
        },
        "strategy": "coembedding"
    },
    "umap_euclidean_coembedding": {
        "method_key": "umap_euclidean",
        "config": {
            "short_name": "UMAP-Euclidean",
            "metric": "euclidean",
            "n_neighbors": 500,  # Best hyperparameters from Experiment 1
            "min_dist": 0.01,
            "allow_coembedding": True
        },
        "strategy": "coembedding"
    }
}

def create_config(dr_method_name, dr_method_info):
    """Create a single configuration file."""
    
    config = {
        "global_settings": GLOBAL_SETTINGS.copy(),
        "targets": [TARGET],
        "representations": REPRESENTATIONS,
        "dimensionality_reduction_methods": {
            dr_method_info["method_key"]: dr_method_info["config"]
        }
    }
    
    # Generate filename
    filename = f"config_ABL1_features_{dr_method_name}.json"
    filepath = OUTPUT_DIR / filename
    
    # Write to file
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)
    
    return filepath

def main():
    print("=" * 70)
    print("Generating Dimensionality Experiment Configurations")
    print("=" * 70)
    print()
    print("This experiment will test the impact of similarity space dimensionality")
    print("on the performance of different dimensionality reduction methods.")
    print()
    print("Target: ABL1 (same as hyperparameter sweep)")
    print(f"Dimensions to test: {GLOBAL_SETTINGS['simspace_dims_to_test']}")
    print("Strategy: Co-embedding only")
    print("Representation: Features only")
    print()
    print("-" * 70)
    print()
    
    configs_created = []
    
    # Create one config per DR method (not per dimension - dimensions are handled internally)
    for dr_name, dr_info in DR_METHODS.items():
        print(f"Creating config for: {dr_info['config']['short_name']} ({dr_info['strategy']})")
        filepath = create_config(dr_name, dr_info)
        configs_created.append(str(filepath))
        print(f"  ✓ {filepath.name}")
    
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Total configs created: {len(configs_created)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    print("Each config will be run with 5 random seeds (42-46)")
    print(f"Total dimensions per config: {len(GLOBAL_SETTINGS['simspace_dims_to_test'])}")
    print(f"Total jobs: {len(configs_created)} configs × 5 seeds = {len(configs_created) * 5} jobs")
    print(f"Total analyses: {len(configs_created) * 5 * len(GLOBAL_SETTINGS['simspace_dims_to_test'])} (configs × seeds × dimensions)")
    print()
    print("Next steps:")
    print("  1. Review the generated configs in:", OUTPUT_DIR)
    print("  2. Submit jobs with: bash hpc/submit_dimensionality_jobs.sh")
    print("  3. After completion, aggregate results with:")
    print("     python aggregate_dimensionality_analysis.py")
    print()
    print("=" * 70)

if __name__ == "__main__":
    main()
