# Archived Scripts

This directory contains deprecated scripts from earlier versions of UMMBAS that are no longer actively maintained in v3.0.

**Archive Date**: October 18, 2025  
**Reason**: Replaced by enhanced v3.0 pipeline scripts

---

## Contents

### `analyze_hyperparams.py`
**Original Purpose**: Basic hyperparameter analysis for Phase 1  
**Status**: REPLACED  
**Replacement**: `scripts/analyze_potency_stratified_enrichment.py`  
**Reason for Deprecation**:
- Lacks potency tier stratification (High/Medium/Weak)
- No parallel processing
- No best hyperparameter identification
- No publication-quality outputs (PNG+PDF)
- No dimension-separated plots

**Can still be used for**: Basic EF@1% analysis without advanced features

---

### `aggregate_and_report.py`
**Original Purpose**: Compile metrics across multiple experiments  
**Status**: DEPRECATED  
**Replacement**: Phase-specific analysis scripts + potency stratification  
**Reason for Deprecation**:
- Replaced by more comprehensive analysis pipeline
- Enhanced scripts provide better reporting
- Phase-specific analyses more appropriate for v3.0

---

### `quick_test_run.sh`
**Original Purpose**: Quick testing script for v2.0 pipeline  
**Status**: NOT MAINTAINED  
**Reason for Deprecation**:
- Not compatible with v3.0 config structure
- Used for v2.0 testing only
- v3.0 uses JSON config files instead

**Alternative**: Use `main_orchestrator.py` with test configs

---

### `setup_hpc_workspaces.sh` → REMOVED FROM ARCHIVE
**Status**: File moved back to `hpc/setup_hpc_workspaces.sh`  
**Reason**: Still actively used for HPC workspace setup, was mistakenly archived

---

## Restoration

If you need to restore any of these scripts:

1. Copy from this directory back to original location
2. Review compatibility with v3.0 config format
3. Update dependencies if needed
4. Test thoroughly before using in production

## Version History

These scripts were part of:
- **v2.0 pipeline** (deprecated September 2025)
- Some date back to **v1.0** (deprecated 2024)

For current v3.0 scripts, see:
- `generate_phase1_configs.py` through `generate_phase4_configs.py`
- `main_orchestrator.py`
- `scripts/analyze_potency_stratified_enrichment.py`
- `extract_phase1_best_configs.py`

---

**See**: [`../ARCHIVE.md`](../ARCHIVE.md) for complete deprecation documentation
