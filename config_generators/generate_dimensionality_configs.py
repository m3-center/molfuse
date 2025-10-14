#!/usr/bin/env python3
"""
Generate configuration files for the dimensionality experiment (v2.0).

This script creates configs for ABL1 using optimal hyperparameters from the 
hyperparameter sweep, testing different dimensionalities [2, 3, 5, 10, 20].

v2.0 Changes:
- Corrected ABL1 molecular function: Transferase (KW-0808) not "Protein kinase inhibitor"
- Removed t-SNE (co-embedding only, data leakage)
- Projection-only strategy (no co-embedding, no data leakage)
- Uses optimal UMAP parameters found in hyperparameter sweep

The goal is to understand how the dimensionality of the similarity space affects 
the performance of different dimensionality reduction methods.

Usage:
    python config_generators/generate_dimensionality_configs.py
"""

import json
import os
from pathlib import Path

# Output directory for dimensionality configs
OUTPUT_DIR = Path("dimensionality_configs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Global settings (v2.0 corrected)
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
    "affinity_cutoff_nM": 100000,
    "workspace_base_dir": "experiment_workspace_dimensionality_v2/",  # UNIQUE WORKSPACE
    "final_report_dir": "final_report_dimensionality_v2/",
    "n_jobs_molcalcs": -1,
    "rdkit_features_list_target": [
        "DipoleMoment", "ABC", "nAcid", "nBase", "nAromAtom", "nAtom", "nH", "nC", "nN", "nO", "nS",
        "nP", "nX", "nBonds", "nBondsO", "nBondsS", "nBondsD", "nBondsT", "nBondsA", "nBondsM",
        "nBondsKS", "nBondsKD", "EState_VSA7", "nHBAcc", "nHBDon", "Lipinski", "apol", "bpol",
        "nRing", "n3Ring", "n4Ring", "n5Ring", "n6Ring", "n7Ring", "n8Ring", "nRot", "Diameter",
        "TopoShapeIndex", "Vabc", "MW"
    ]
}

# ABL1 target (v2.0 CORRECTED)
TARGET = {
    "id_name": "TyrosineProteinKinaseABL1_P00519",
    "display_name": "Tyrosine-protein Kinase ABL1",
    "uniprot_id": "P00519",
    "molecular_function_canonical_name": "Transferase",  # CORRECTED from "Protein kinase inhibitor"
    "molecular_function_filename_segment": "Transferase",  # CORRECTED
    "molecular_function_display_name": "Transferase",  # CORRECTED
    "molecular_function_kw_code": "KW-0808"  # ADDED
}

# Only features representation (best performing)
REPRESENTATIONS = ["features"]

# v2.0: Dimensionality reduction methods with optimal hyperparameters
# PROJECTION-ONLY strategy (no co-embedding, no data leakage)
# NOTE: Update n_neighbors and min_dist after hyperparameter sweep results
DR_METHODS = {
    "pca_projection": {
        "method_key": "pca",
        "config": {
            "short_name": "PCA"
        },
        "strategy": "projection"
    },
    "umap_euclidean_projection": {
        "method_key": "umap_euclidean",
        "config": {
            "short_name": "UMAP-Euclidean",
            "metric": "euclidean",
            "n_neighbors": 500,  # Update from hyperparameter sweep results
            "min_dist": 0.01  # Update from hyperparameter sweep results
        },
        "strategy": "projection"
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
    print("UMMBAS v2.0 - Dimensionality Experiment Config Generation")
    print("=" * 70)
    print()
    print("This experiment tests the impact of similarity space dimensionality")
    print("on the performance of different dimensionality reduction methods.")
    print()
    print(f"Target: ABL1 (Transferase KW-0808) - v2.0 CORRECTED")
    print(f"Dimensions to test: {GLOBAL_SETTINGS['simspace_dims_to_test']}")
    print(f"Strategy: Projection-only (v2.0)")
    print(f"Representation: Features only")
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
    print("v2.0 Changes:")
    print("  ✓ ABL1 MF corrected: Transferase (KW-0808)")
    print("  ✗ t-SNE removed (co-embedding only)")
    print("  ✓ Projection-only strategy (no data leakage)")
    print()
    print("To run the experiment:")
    print("  1. Update UMAP hyperparameters from sweep results")
    print("  2. Submit jobs: bash hpc/submit_dimensionality_jobs.sh")
    print("  3. Each config runs with 5 seeds (42-46)")
    print(f"  4. Total dimensions per config: {len(GLOBAL_SETTINGS['simspace_dims_to_test'])}")
    print(f"  5. Aggregate: python analysis_scripts/aggregate_dimensionality_analysis.py")
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
