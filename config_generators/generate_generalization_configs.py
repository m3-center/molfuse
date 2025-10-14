#!/usr/bin/env python3
"""
Generate configuration files for the generalization experiment (v2.0).

This script creates configs for two target proteins (PyruvateKinaseM2 and 
IsocitrateDehydrogenaseNADP) using optimal hyperparameters from ABL1 analysis.

v2.0 Changes:
- Corrected molecular function assignments (kinases → Transferase KW-0808)
- Removed t-SNE (co-embedding only, data leakage)
- Removed co-embedding (data leakage in prospective screening)
- Projection-only strategy for all DR methods

We focus on features (not fingerprints) since fingerprints showed poor 
performance in the hyperparameter sweep.

The goal is to test whether the methods generalize across different target proteins
without expensive hyperparameter retuning.
"""

import json
import os
from pathlib import Path

# Output directory for generalization configs
OUTPUT_DIR = Path("generalization_configs")
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
    "simspace_dims_to_test": [2],  # Only 2D as in hyperparameter analysis
    "k_for_knn_distance": [3, 5],
    "affinity_cutoff_nM": 100000,
    "workspace_base_dir": "experiment_workspace_generalization/",  # UNIQUE WORKSPACE
    "final_report_dir": "final_report_generalization/",
    "n_jobs_molcalcs": -1,
    "rdkit_features_list_target": [
        "DipoleMoment", "ABC", "nAcid", "nBase", "nAromAtom", "nAtom", "nH", "nC", "nN", "nO", "nS",
        "nP", "nX", "nBonds", "nBondsO", "nBondsS", "nBondsD", "nBondsT", "nBondsA", "nBondsM",
        "nBondsKS", "nBondsKD", "EState_VSA7", "nHBAcc", "nHBDon", "Lipinski", "apol", "bpol",
        "nRing", "n3Ring", "n4Ring", "n5Ring", "n6Ring", "n7Ring", "n8Ring", "nRot", "Diameter",
        "TopoShapeIndex", "Vabc", "MW"
    ]
}

# Target proteins for generalization (excluding ABL1 which was used for hyperparameter tuning)
TARGETS = [
    {
        "id_name": "PyruvateKinaseM2_P14618",
        "display_name": "Pyruvate Kinase M2",
        "uniprot_id": "P14618",
        "molecular_function_canonical_name": "Transferase",  # CORRECTED from "Protein kinase inhibitor"
        "molecular_function_filename_segment": "Transferase",  # CORRECTED
        "molecular_function_display_name": "Transferase",  # CORRECTED
        "molecular_function_kw_code": "KW-0808"
    },
    {
        "id_name": "IsocitrateDehydrogenaseNADP_O75874",
        "display_name": "Isocitrate Dehydrogenase NADP cytoplasmic",
        "uniprot_id": "O75874",
        "molecular_function_canonical_name": "Oxidoreductase",
        "molecular_function_filename_segment": "Oxidoreductase",
        "molecular_function_display_name": "Oxidoreductase",
        "molecular_function_kw_code": "KW-0560"
    }
]

# Only features representation (fingerprints performed poorly)
REPRESENTATIONS = ["features"]

# v2.0: Dimensionality reduction methods (PROJECTION-ONLY)
# t-SNE removed (co-embedding only, data leakage)
# Co-embedding removed (data leakage)
# Using optimal hyperparameters from ABL1 analysis
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
            "n_neighbors": 500,
            "min_dist": 0.01
        },
        "strategy": "projection"
    }
}

def create_config(target, dr_method_name, dr_method_info):
    """Create a single configuration file."""
    
    config = {
        "global_settings": GLOBAL_SETTINGS.copy(),
        "targets": [target],
        "representations": REPRESENTATIONS,
        "dimensionality_reduction_methods": {
            dr_method_info["method_key"]: dr_method_info["config"]
        }
    }
    
    # Generate filename with target name to avoid overwriting
    target_short = target["id_name"]
    filename = f"config_{target_short}_features_{dr_method_name}.json"
    filepath = OUTPUT_DIR / filename
    
    # Write to file
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)
    
    return filepath

def main():
    print("=" * 70)
    print("Generating Generalization Experiment Configurations")
    print("=" * 70)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Number of targets: {len(TARGETS)}")
    print(f"Number of DR methods: {len(DR_METHODS)}")
    print(f"Total configs to generate: {len(TARGETS) * len(DR_METHODS)}")
    print()
    
    configs_created = []
    
    for target in TARGETS:
        print(f"\nTarget: {target['display_name']} ({target['id_name']})")
        
        for dr_name, dr_info in DR_METHODS.items():
            filepath = create_config(target, dr_name, dr_info)
            configs_created.append(filepath)
            print(f"  ✓ Created: {filepath.name}")
    
    print(f"\n{'=' * 70}")
    print(f"Successfully created {len(configs_created)} configuration files (v2.0)")
    print(f"{'=' * 70}")
    
    # Print summary
    print("\nConfiguration Summary (v2.0 - Projection-Only):")
    print("-" * 70)
    print("Methods included (with BEST hyperparameters from ABL1):")
    print("  • PCA (Projection-only)")
    print("  • UMAP-Euclidean (Projection-only, n_neighbors=500, min_dist=0.01)")
    print()
    print("v2.0 Changes:")
    print("  ✗ t-SNE removed (co-embedding only, data leakage)")
    print("  ✗ Co-embedding removed (data leakage)")
    print("  ✓ Projection-only strategy (prevents data leakage)")
    print()
    print("Targets (with corrected molecular functions):")
    for target in TARGETS:
        mf = target['molecular_function_canonical_name']
        kw = target['molecular_function_kw_code']
        print(f"  • {target['display_name']} ({mf}, {kw})")
    print()
    print("Workspace: experiment_workspace_generalization/")
    print("Reports: final_report_generalization/")
    print()
    print("To run the experiment:")
    print("  1. Submit jobs using: bash hpc/submit_generalization_jobs.sh")
    print("  2. Each config will be run with 5 random seeds (42-46)")
    print(f"  3. Total jobs: {len(configs_created)} configs × 5 seeds = {len(configs_created) * 5} jobs")
    print("=" * 70)

if __name__ == "__main__":
    main()
