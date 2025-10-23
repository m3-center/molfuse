# UMMBAS Archive - Deprecated Features & Scripts

**Purpose**: This file documents deprecated features, scripts, and experiments that are no longer in active use in the v3.0 pipeline.

**Last Updated**: October 18, 2025

---

## Archived Versions

### v2.0 Pipeline (Deprecated September 2025)

The v2.0 pipeline used a different experimental design focused on single-target analysis without systematic hyperparameter sweeps.

**Key Differences from v3.0**:
- No multi-phase experimental design
- Manual hyperparameter selection
- Limited dimensionality testing
- No MF cloud ablation study
- Single-target focus (no generalization testing)
- No potency stratification

**Documentation Archived**:
- `docs/archive_v2.0/` - Contains all v2.0 documentation
- `docs/REFACTORING_PLAN_v2.0.md` - Migration plan from v2.0 to v3.0

### v1.0 Pipeline (Deprecated 2024)

Early prototype version with basic similarity space concept.

**Archived Location**: Historical commits only (no documentation preserved)

---

## Archived Scripts Directory

**Location**: `archived_scripts/`

This directory contains deprecated scripts that have been moved from active locations. All archived scripts include documentation in `archived_scripts/README.md` explaining:
- Original purpose
- Reason for deprecation
- Current replacement
- Restoration instructions if needed

**Contents**:
- `analyze_hyperparams.py` (old analysis script)
- `aggregate_and_report.py` (old reporting script)
- `quick_test_run.sh` (v2.0 test script)
- `extract_phase1_best_configs.py` (October 2025) - Functionality integrated into `analyze_potency_stratified_enrichment.py`

---

## Deprecated Scripts

### Configuration Management

#### `generate_hyperparam_configs.py` → REPLACED
**Deprecated**: October 2025  
**Replacement**: `generate_phase1_configs.py`  
**Reason**: v3.0 uses 4-phase design, old script only generated Phase 1 configs

#### `generate_dimensionality_configs.py` → REPLACED
**Deprecated**: October 2025  
**Replacement**: Integrated into `generate_phase1_configs.py`  
**Reason**: Dimensionality is now part of Phase 1 hyperparameter sweep

#### `generate_generalization_configs.py` → REPLACED
**Deprecated**: October 2025  
**Replacement**: `generate_phase3_configs.py`  
**Reason**: Renamed for consistency with 4-phase naming scheme

### Analysis Scripts (Old Versions)

#### `analyze_hyperparams.py` → REPLACED
**Deprecated**: October 2025  
**Archived Location**: `archived_scripts/analyze_hyperparams.py`  
**Replacement**: `scripts/analyze_potency_stratified_enrichment.py`  
**Reason**: New script adds:
- Potency tier stratification (High/Medium/Weak)
- Parallel processing (multiprocessing)
- Best hyperparameter identification
- PNG + PDF publication outputs
- Dimension-separated plots

**Migration Notes**: Old script still works for basic EF@1% analysis but lacks advanced features.

#### `aggregate_and_report.py` (Old Version) → DEPRECATED
**Deprecated**: October 2025  
**Archived Location**: `archived_scripts/aggregate_and_report.py`  
**Reason**: Replaced by phase-specific analysis scripts with enhanced features

### Experimental Scripts

#### `quick_test_run.sh` → DEPRECATED
**Deprecated**: October 2025  
**Archived Location**: `archived_scripts/quick_test_run.sh`  
**Reason**: Used for v2.0 testing, not compatible with v3.0 config structure

#### `setup_hpc_workspaces.sh` → MOVED (NOT DEPRECATED)
**Status**: Active, moved to correct location  
**Current Location**: `hpc/setup_hpc_workspaces.sh`  
**Reason**: Still in use for HPC workspace setup, moved from root to hpc/ directory for better organization

---

## Deprecated Configuration Formats

### Old Config Structure (v2.0)
```json
{
  "target_protein": "single_target_id",
  "dimensionality": 5,
  "method": "UMAP",
  "n_neighbors": 15,
  "min_dist": 0.1
}
```

**Issues**:
- Single dimension only
- No phase tracking
- No systematic seed management
- Limited hyperparameter grid

### New Config Structure (v3.0)
```json
{
  "global_settings": {
    "simspace_dims_to_test": [2, 5, 10],
    "workspace_base_dir": "experiment_workspace_v3_phase1/",
    "affinity_cutoff_nM": 100000,
    "mf_cloud_max_molecules": 420000
  },
  "targets": [{ ... }],
  "representations": ["features"],
  "dimensionality_reduction_methods": { ... },
  "random_seed": 42,
  "phase": "phase1_hyperparam_sweep",
  "experiment_type": "features_umap_euclidean_dim5_nn10_md0.01"
}
```

**Improvements**:
- Multi-dimensional testing
- Phase tracking
- Systematic seed management
- Comprehensive hyperparameter specification
- MF cloud size control
- Affinity cutoff specification

---

## Deprecated Analysis Methods

### Single-Metric Evaluation
**Deprecated**: October 2025  
**Replacement**: Multi-metric + potency stratification

**Old Approach**:
- Overall EF@1% only
- No distinction between high-potency and weak binders
- Single-seed evaluation

**New Approach (v3.0)**:
- Overall EF@1%
- Potency-stratified EF@1% (High/Medium/Weak tiers)
- ROC-AUC, PR-AUC
- 5-replicate statistical robustness
- Quality vs quantity trade-off analysis

### Visualization Methods

#### Bar plots only → ENHANCED
**Old**: Simple bar charts with overall EF@1%  
**New**: 
- Stratified bar plots by potency tier
- Dimension-separated plots
- Heatmaps (method × dimension × tier)
- Quality vs quantity scatter plots
- PNG (300 DPI) + PDF (vector) outputs

---

## Deprecated Datasets

### ChEMBL 33 Data → UPDATED
**Deprecated**: 2024  
**Current**: ChEMBL 35 (October 2025)  
**Location**: `datasets/molecular_function_affinity_data/`

**Changes**:
- Increased active compound counts
- Updated affinity measurements
- Improved data quality filters

### Old ZINC Subset → UPDATED
**Deprecated**: 2024  
**Current**: ZINC subset (1.29M compounds)  
**Location**: `datasets/zinc_data.csv`

**Changes**:
- Updated to more recent ZINC version
- Better property distribution
- Validated zero overlap with MF cloud

---

## Deprecated Documentation

### Moved to Archive

1. **`docs/archive_v2.0/`**
   - All v2.0-specific documentation
   - Migration guides
   - Old analysis reports

2. **`docs/archive_development/`**
   - Development notes from v1.0-v2.0
   - Prototype scripts
   - Early experimental results

### Superseded Documents

#### `DOCUMENTATION_CLEANUP_SUMMARY.md` → ARCHIVED
**Date**: October 2025  
**Reason**: Cleanup complete, archive for reference only

#### Old README versions → ARCHIVED
- `README_OLD.md` (if exists) → Removed
- `README_V2.md` (if exists) → Moved to archive

---

## Breaking Changes from v2.0 to v3.0

### Configuration Files
- **Change**: Config structure completely redesigned
- **Migration**: Cannot directly use v2.0 configs, must regenerate
- **Tool**: Config generators (`generate_phase*_configs.py`)

### Workspace Structure
- **Change**: New directory hierarchy with phase-specific workspaces
- **Old**: `experiment_workspace/`
- **New**: `experiment_workspace_v3_phase1/`, `_phase2/`, etc.

### Script Arguments
- **Change**: Main orchestrator now requires JSON config file
- **Old**: `python main_script.py --target XYZ --method UMAP ...`
- **New**: `python main_orchestrator.py --config config.json --seed 42`

### Output Files
- **Change**: New naming conventions and directory structure
- **Pattern**: `TARGET/results/REPR/dim_N/METHOD/ranking_files.csv`
- **Migration**: Old results must be manually mapped if reanalysis needed

---

## Deprecated HPC Submissions

### Old SLURM Scripts → REPLACED
**Deprecated**: September 2025  
**Replacement**: `hpc/submit_v3_phase*.sh`

**Old Scripts**:
- `submit_hyperparam_sweep.sh`
- `submit_dimensionality_test.sh`

**New Scripts**:
- `hpc/submit_v3_phase1.sh` - Hyperparameter sweep
- `hpc/submit_v3_phase2.sh` - MF cloud ablation
- `hpc/submit_v3_phase3.sh` - Generalization
- `hpc/submit_v3_phase4.sh` - Cutoff sensitivity

**Key Improvements**:
- Array job management
- Better resource allocation
- Checkpoint support
- Automatic error detection

---

## Experimental Features (Not Yet Production)

### Under Development

#### Checkpoint/Resume System
**Status**: Planned, not implemented  
**Purpose**: Resume interrupted HPC runs  
**ETA**: November 2025

#### Adaptive Hyperparameter Tuning
**Status**: Experimental idea  
**Purpose**: Use Bayesian optimization for hyperparameter search  
**ETA**: Unknown (research phase)

---

## Notes for Future Developers

### When to Update This File

1. **Major version changes** (v3.0 → v4.0)
2. **Script deprecations** (old script → new replacement)
3. **Breaking changes** to config formats or APIs
4. **Experimental design changes** affecting reproducibility

### Retention Policy

- Keep archived documentation for **2 major versions**
- Example: When v4.0 releases, remove v2.0 archives
- Exception: Keep v1.0 historical reference indefinitely

### Restoration Process

If you need to restore deprecated functionality:

1. Check git history: `git log --all --full-history -- path/to/file`
2. Checkout old version: `git checkout <commit> -- path/to/file`
3. Review compatibility issues before use
4. Update dependencies if needed

---

## Version History

- **v3.0** (October 2025): Current stable version
  - 4-phase experimental pipeline
  - Potency-stratified analysis
  - Parallel processing
  - Publication-quality outputs

- **v2.0** (Deprecated September 2025): Single-phase pipeline
  - Basic hyperparameter testing
  - Single-target analysis
  - Manual configuration

- **v1.0** (Deprecated 2024): Prototype
  - Proof of concept
  - Limited automation
  - Basic similarity space

---

**End of Archive Document**
