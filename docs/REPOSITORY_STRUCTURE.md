# Repository Structure - v2.0

## Overview
This document describes the organizational structure of the UMMBAS screening experiments repository after v2.0 refactoring and reorganization.

## Root Directory Files

### Core Scripts
- **`main_orchestrator.py`**: Main pipeline orchestration script (coordinates all phases)
- **`run_generalization_experiment.sh`**: Shell script for running generalization experiments

### Configuration Files
- **`experiment_config.json`**: Main experiment configuration (v2.0 - projection-only, no t-SNE)
- **`debugging_config.json`**: Debugging configuration with smaller datasets

## Directory Structure

### `analysis_scripts/`
Scripts for data analysis, aggregation, and reporting:
- **`aggregate_and_report.py`**: Aggregates results and generates comprehensive reports
- **`aggregate_cutoff_analysis.py`**: Aggregates cutoff threshold analysis results
- **`aggregate_dimensionality_analysis.py`**: Aggregates dimensionality reduction analysis (v1.0)
- **`aggregate_generalization_analysis.py`**: Aggregates generalization experiment results
- **`analyze_experimental_dataset_counts.py`**: Analyzes experimental dataset statistics
- **`analyze_fingerprint_similarity.py`**: Analyzes bit-level fingerprint similarity patterns
- **`analyze_fingerprint_variance.py`**: Analyzes fingerprint bit variance across datasets
- **`analyze_hyperparams.py`**: Analyzes hyperparameter optimization results
- **`run_cutoff_analysis.py`**: Runs cutoff threshold sweep analysis
- **`run_cutoff_analysis_generalization.py`**: Runs cutoff analysis for generalization targets
- **`test_generalization_compatibility.py`**: Tests compatibility of generalization configurations

### `baselines/`
Baseline comparison methods (v2.0):
- **`baseline_random_ranking.py`**: Random shuffling baseline (EF@1% ≈ 1.0)
- **`baseline_feature_space_distance.py`**: 39D Euclidean distance baseline (no DR)
- **`baseline_tanimoto_similarity.py`**: ECFP4 Tanimoto similarity baseline
- **`analyze_chemical_novelty.py`**: Quantifies chemical novelty using Tanimoto distributions
- **`README.md`**: Comprehensive documentation with usage examples

### `config_generators/`
Scripts for generating experiment configuration files:
- **`generate_dimensionality_configs.py`**: Generates configs for dimensionality experiments (v1.0)
- **`generate_generalization_configs.py`**: Generates configs for generalization targets (v2.0)

### `core_scripts/`
Core computational pipeline scripts:
- **`calculate_features_and_fingerprints_exp.py`**: Calculates molecular features (RDKit descriptors) and fingerprints (ECFP4)
- **`calculate_similarityspaces_exp.py`**: Constructs similarity spaces using dimensionality reduction (v2.0 - projection-only, no t-SNE, no co-embedding)
- **`analyze_pca_variance.py`**: Analyzes PCA explained variance for optimal component selection
- **`utils.py`**: Utility functions used across the pipeline

### `datasets/`
Molecular function datasets and supporting files:
- **`molecular_function_affinity_data/`**: ChEMBL 35 bioactivity data filtered by molecular function
  - `KW-0808_Transferase_affinity.csv`: 425,088 records, 191,648 compounds, 831 targets (ABL1/PK excluded)
  - `KW-0560_Oxidoreductase_affinity.csv`: 97,729 records, 56,533 compounds, 270 targets (IDH excluded)
- **`molecular_function_features_fingerprints/`**: Pre-calculated features and fingerprints
  - `KW-0808_Transferase_features.pkl`: RDKit descriptors for Transferase compounds
  - `KW-0808_Transferase_fingerprints.pkl`: ECFP4 fingerprints for Transferase compounds
  - `KW-0560_Oxidoreductase_features.pkl`: RDKit descriptors for Oxidoreductase compounds
  - `KW-0560_Oxidoreductase_fingerprints.pkl`: ECFP4 fingerprints for Oxidoreductase compounds
- **`protein_collection/`**: Protein target information and UniProt verification results
- **`chembl/`**: Original ChEMBL data files
- **`zinc_data.csv`**: ZINC database decoy compounds
- **`README.md`**: Dataset documentation

### `docs/`
All documentation files (consolidated):
- **`ANALYSIS_PIPELINE_OVERVIEW.md`**: Overview of the analysis pipeline
- **`CUTOFF_ANALYSIS_UPDATES.md`**: Updates to cutoff threshold analysis
- **`DATASET_ANALYSIS_SUMMARY.md`**: Summary of dataset analysis
- **`DEBUGGING.md`**: Debugging guide
- **`DIMENSIONALITY_CHECKLIST.md`**: Dimensionality experiment checklist (v1.0)
- **`DIMENSIONALITY_EXPERIMENT_README.md`**: Dimensionality experiments documentation (v1.0)
- **`DIMENSIONALITY_IMPLEMENTATION_SUMMARY.md`**: Implementation summary (v1.0)
- **`DIMENSIONALITY_QUICKSTART.txt`**: Quick start guide for dimensionality experiments (v1.0)
- **`FINAL_DATASET_VERIFICATION.md`**: Final verification of v2.0 datasets
- **`GENERALIZATION_EXPERIMENT_README.md`**: Generalization experiments documentation
- **`GENERALIZATION_QUICKSTART.txt`**: Quick start guide for generalization experiments
- **`PHASE_1.1_COMPLETION_REPORT.md`**: Phase 1.1 completion report (UniProt verification)
- **`README_COMPLETE_PIPELINE.md`**: Complete pipeline documentation
- **`REFACTORING_PLAN_v2.0.md`**: Comprehensive v2.0 refactoring plan (75 pages)
- **`REPOSITORY_STRUCTURE.md`**: This file
- **`ummbas_dataset_analysis_experimental_approach.md`**: Experimental approach documentation
- **`ummbas_dataset_analysis_final.md`**: Final dataset analysis
- **`ummbas_dataset_analysis_report.md`**: Dataset analysis report

### `experimental_pipeline/`
Experimental workflow scripts (v1.0 framework):
- **`prepare_data.py`**: Prepares data for experiments
- **`project_and_analyze.py`**: Projects data and analyzes results
- **`rank_zinc_decoys.py`**: Ranks ZINC decoy compounds

### `generalization_configs/`
Per-target experiment configurations (v2.0):
- Configuration files for held-out targets (Pyruvate Kinase M2, Isocitrate Dehydrogenase)
- Format: `config_{TargetName}_{UniProtID}_{representation}_{method}_{strategy}.json`
- v2.0 contains 4 configs (projection-only, no t-SNE, no co-embedding)

### `hpc/`
HPC job submission and execution scripts (updated for v2.0 structure):
- **`submit_*.sh`**: Job submission scripts
- **`ummbas_*.sh`**: SLURM job definition scripts
- All scripts updated to use new paths (`analysis_scripts/`, `config_generators/`)

### `hyperparam_configs/`
Hyperparameter optimization configurations (v1.0):
- Contains hyperparameter sweep configurations for UMAP optimization

### `reporting/`
Legacy reporting scripts and templates

### `scripts/`
Utility and verification scripts:
- **`query_uniprot_molecular_functions.py`**: UniProt API verification script
- **`verify_transferase_data.py`**: Data filtering and validation script

### `logs/`, `final_report/`, `final_report_cutoff_analysis/`, `final_report_generalization/`, `hyperparameterization_report/`, `reports/`
Output directories for logs and generated reports

### `fingerprint_variance_analysis/`
Fingerprint bit variance analysis results (v1.0)

### `tests/`
Unit tests and integration tests

## Version History

### v2.0 (Current)
**Key Changes:**
- ✅ Fixed molecular function assignments (ABL1/PK → Transferase KW-0808)
- ✅ Removed t-SNE completely (architectural limitation: co-embedding only)
- ✅ Removed co-embedding (projection-only strategy)
- ✅ Fixed fingerprint processing (Jaccard-only, no scaling, no PCA)
- ✅ Implemented 4 baseline methods
- ✅ Reorganized repository structure (docs/, analysis_scripts/, config_generators/)

**Data:**
- Transferase MF cloud: 191,648 compounds (was 20 in v1.0)
- Oxidoreductase MF cloud: 56,533 compounds
- ABL1: 3,331 actives (Transferase) - held out
- Pyruvate Kinase M2: 1,019 actives (Transferase) - held out
- Isocitrate Dehydrogenase: 374 actives (Oxidoreductase) - held out

**Methodology:**
- Projection-only strategy (no data leakage)
- DR methods: PCA, UMAP-Euclidean (features only)
- Representations: Features only (39 RDKit descriptors)
- Fingerprints: ECFP4 with Jaccard distance (future work)

### v1.0 (Legacy)
**Issues:**
- ❌ ABL1/PK misclassified as "Protein kinase inhibitor" instead of "Transferase"
- ❌ Transferase MF cloud had only 20 compounds (3 non-kinase targets)
- ❌ Created 165:1 data leakage ratio (3,331 actives : 20 MF cloud)
- ❌ t-SNE only supported co-embedding (data leakage)
- ❌ Fingerprints were incorrectly scaled and PCA-reduced

## Usage

### Running Experiments (HPC)
```bash
# Submit generalization experiment
cd hpc/
sbatch submit_generalization_jobs.sh

# Submit hyperparameter optimization
sbatch submit_hyperparameter_jobs.sh
```

### Running Baselines (Local)
```bash
# Random baseline
python baselines/baseline_random_ranking.py <target_name> <uniprot_id> <mf_kw_code>

# Feature space distance baseline
python baselines/baseline_feature_space_distance.py <target_name> <uniprot_id> <mf_kw_code>

# Tanimoto similarity baseline
python baselines/baseline_tanimoto_similarity.py <target_name> <uniprot_id> <mf_kw_code>

# Chemical novelty analysis
python baselines/analyze_chemical_novelty.py <target_name> <uniprot_id> <mf_kw_code>
```

### Generating Configurations
```bash
# Generate generalization experiment configs (v2.0)
python config_generators/generate_generalization_configs.py

# Generate dimensionality experiment configs (v1.0)
python config_generators/generate_dimensionality_configs.py
```

### Aggregating Results
```bash
# Aggregate generalization results
python analysis_scripts/aggregate_generalization_analysis.py

# Aggregate cutoff analysis
python analysis_scripts/aggregate_cutoff_analysis.py

# Generate full report
python analysis_scripts/aggregate_and_report.py
```

## Next Steps (v2.0 Roadmap)

### Phase 4: Hyperparameter Re-optimization (HPC)
- Re-run UMAP hyperparameter sweep for ABL1 with corrected Transferase MF (191k compounds)
- Search grid: n_neighbors × min_dist × simspace_dims
- Expected runtime: ~48 hours

### Phase 5: Full Experimental Re-run (HPC)
- Run complete pipeline for all 3 targets × 5 seeds
- PCA + UMAP-Euclidean projection (features)
- Run all 4 baselines for each target
- Expected runtime: ~72 hours

## References
- **ChEMBL 35**: https://www.ebi.ac.uk/chembl/
- **UniProt Keywords**: Transferase (KW-0808), Oxidoreductase (KW-0560)
- **RDKit**: https://www.rdkit.org/
- **UMAP**: https://umap-learn.readthedocs.io/
