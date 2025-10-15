# UMMBAS v3.0 - Comprehensive Refactoring Plan

**Version:** 3.0  
**Date:** October 15, 2025  
**Branch:** `3.0`  
**Status:** Planning Phase

---

## Executive Summary

UMMBAS v3.0 represents a major restructuring of the experimental pipeline to systematically investigate:
1. Optimal dimensionality for PCA vs UMAP methods
2. MF cloud size impact on algorithm performance (phase transition validation)
3. Cross-protein generalizability of optimal configurations
4. Affinity cutoff threshold sensitivity

**Total Experiments:** 440 runs (vs 650 in v2.0)  
**Key Insight:** Test dimensionality BEFORE generalization to identify true optimal configurations

---

## Phase Structure Overview

### **Phase 1: Tyro Dimensionality × Hyperparameter Sweep**
- **Target:** TyrosineProteinKinaseABL1_P00519 only
- **Purpose:** Identify optimal dimension and hyperparameters per method
- **Runs:** 260
- **Output:** Best config per method per dimension

### **Phase 2: MF Cloud Ablation Study**
- **Target:** Tyro only
- **Purpose:** Validate phase transition hypothesis (PCA overtaking UMAP)
- **Runs:** 60-120 (depends on best dimensions from Phase 1)
- **Output:** Crossover point, mechanistic understanding

### **Phase 3: Cross-Protein Generalization**
- **Targets:** PyruvateKinaseM2_P14618, IsocitrateDehydrogenaseNADP_O75874
- **Purpose:** Test transferability of optimal configs
- **Runs:** 80
- **Output:** Generalization performance metrics

### **Phase 4: Affinity Cutoff Analysis**
- **Target:** Tyro only
- **Purpose:** Sensitivity analysis for practical applications
- **Runs:** 40
- **Output:** Optimal cutoff recommendations

---

## Detailed Phase Specifications

### **PHASE 1: Tyro Dimensionality × Hyperparameter Sweep**

#### Representations & Methods
```
Features:
  - PCA: 2D, 5D, 10D
  - UMAP-Euclidean: 2D, 5D, 10D
  
Fingerprints:
  - PCA: 2D only
  - UMAP-Jaccard: 2D only
```

#### UMAP Hyperparameters (ALL dimensions)
```
n_neighbors: [10, 20, 100, 500]
min_dist: [0.01, 0.1, 0.5]
Total combinations: 12 per dimension
```

#### Run Breakdown
```
Features-PCA:              3 dims × 1 config × 5 seeds = 15 runs
Features-UMAP-Euclidean:   3 dims × 12 configs × 5 seeds = 180 runs
Fingerprints-PCA:          1 dim × 1 config × 5 seeds = 5 runs
Fingerprints-UMAP-Jaccard: 1 dim × 12 configs × 5 seeds = 60 runs
-----------------------------------------------------------
TOTAL:                                                260 runs
```

#### Expected Outputs
- `phase1_results/tyro_features_pca_dim{2,5,10}/`
  - Ranking metrics, ROC/AUC, PR/AUC
  - Similarity space coordinates
  - Distance distributions
- `phase1_results/tyro_features_umap_euclidean_dim{2,5,10}_nn{10,20,100,500}_md{0.01,0.1,0.5}/`
  - Same metrics as PCA
- `phase1_results/tyro_fingerprints_pca_dim2/`
- `phase1_results/tyro_fingerprints_umap_jaccard_dim2_nn{...}_md{...}/`

#### Analysis Outputs
- `phase1_analysis/best_configs_per_method.csv`
- `phase1_analysis/dimensionality_comparison_report.pdf`
- `phase1_analysis/hyperparameter_heatmaps.png/pdf`

---

### **PHASE 2: MF Cloud Ablation Study**

#### MF Cloud Sizes
```
0, 1K, 10K, 50K, 100K, 420K (6 levels)
```

#### Methods
**CORRECTED SPECIFICATION:**
- Best PCA-features at its best dimension (from Phase 1)
- Best UMAP-features at its best dimension (from Phase 1)

Example scenarios:
- If Phase 1 shows: PCA best at 5D, UMAP best at 10D
  - Test PCA-5D across all MF sizes
  - Test UMAP-10D across all MF sizes
- If Phase 1 shows: Both best at 2D
  - Test both at 2D (original plan)

#### Run Calculation
```
Scenario A (different best dims):
  6 MF sizes × 2 methods × 5 seeds = 60 runs

Scenario B (same best dim):
  6 MF sizes × 2 methods × 5 seeds = 60 runs

Maximum: 60 runs
```

#### Purpose
- Validate phase transition hypothesis
- Identify MF cloud size threshold where PCA overtakes UMAP
- Current observation: 20 MF (UMAP 48, PCA 2.5) → 420K MF (UMAP 39, PCA 57.6)
- Predicted crossover: 10K-50K molecules

#### Expected Outputs
- `phase2_results/tyro_mf{0,1k,10k,50k,100k,420k}_best_pca_dim{X}/`
- `phase2_results/tyro_mf{0,1k,10k,50k,100k,420k}_best_umap_dim{Y}/`
- `phase2_analysis/mf_cloud_crossover_plot.pdf`
- `phase2_analysis/phase_transition_report.pdf`

---

### **PHASE 3: Cross-Protein Generalization**

#### Target Proteins
```
Primary (Reference):
  - TyrosineProteinKinaseABL1_P00519 (Tyro) - Transferase

Generalization:
  - PyruvateKinaseM2_P14618 (Pyru) - Transferase (same function)
  - IsocitrateDehydrogenaseNADP_O75874 (Iso) - Oxidoreductase (different function)
```

**REMOVE from configs:**
- ActinCytoplasmic1_P60709
- ActinCytoplasmic2_P63261

#### Methods (Best Configs from Phase 1)
```
1. Best PCA-features-2D
2. Best PCA-features-5D
3. Best PCA-features-10D
4. Best UMAP-Euclidean-features-2D
5. Best UMAP-Euclidean-features-5D
6. Best UMAP-Euclidean-features-10D
7. Best PCA-fingerprints-2D
8. Best UMAP-Jaccard-fingerprints-2D

Total: 8 configs
```

#### Run Calculation
```
8 configs × 2 proteins × 5 seeds = 80 runs
```

#### Analysis Focus
- Same molecular function (Tyro→Pyru): High transferability expected
- Different molecular function (Tyro→Iso): Generalization test
- Dimension-wise transferability comparison

#### Expected Outputs
- `phase3_results/{pyru,iso}_{method}_dim{X}/`
- `phase3_analysis/generalization_comparison.pdf`
- `phase3_analysis/transferability_matrix.csv`
- `phase3_analysis/same_vs_different_function.pdf`

---

### **PHASE 4: Affinity Cutoff Analysis**

#### Cutoffs
```
100,000 nM, 10,000 nM, 1,000 nM, 100 nM
```

#### Methods
- Overall best PCA (best dimension + hyperparams from Phase 1)
- Overall best UMAP (best dimension + hyperparams from Phase 1)

#### Run Calculation
```
4 cutoffs × 2 methods × 5 seeds = 40 runs
```

#### Purpose
- Practical sensitivity analysis
- Optimal cutoff recommendations per method
- Compare cutoff robustness between PCA and UMAP

#### Expected Outputs
- `phase4_results/tyro_cutoff{100000,10000,1000,100}nM_{best_pca,best_umap}/`
- `phase4_analysis/cutoff_sensitivity_curves.pdf`
- `phase4_analysis/optimal_cutoff_recommendations.csv`

---

## New Analysis Module: PCA Dominance Investigation

### Module: `analysis_scripts/analyze_pca_dominance.py`

#### Analysis Components

**1. Variance Explained Analysis**
```python
- PCA eigenvalue spectrum (scree plots)
- Cumulative variance by dimension
- Compare features (40D→2/5/10D) vs fingerprints (2048D→2D)
- Outputs: variance_explained_comparison.pdf
```

**2. Manifold Quality Metrics**
```python
- UMAP trustworthiness score (neighborhood preservation)
- UMAP continuity score (smoothness)
- k-NN preservation (k=10,20,50)
- Compare across 2D/5D/10D
- Outputs: manifold_quality_comparison.csv, quality_heatmaps.pdf
```

**3. Gradient Linearity Analysis**
```python
- Measure linearity of MF cloud → TARGET gradient
- Compare PCA (linear projection) vs UMAP (non-linear manifold)
- Quantify "separability index"
- Outputs: gradient_linearity_analysis.pdf
```

**4. Distance Distribution Analysis**
```python
- Intra-class distances (ZINC, MF, TARGETS)
- Inter-class distances
- Silhouette scores per category
- Density estimation
- Outputs: distance_distributions.pdf, silhouette_scores.csv
```

**5. Compression Ratio Impact**
```python
Features compression:
  - 40D → 2D: 20:1
  - 40D → 5D: 8:1
  - 40D → 10D: 4:1

Fingerprints compression:
  - 2048D → 2D: 1024:1
  - (Not tested at higher dims in v3.0)

Quantify information loss per compression level
Outputs: compression_impact_analysis.pdf
```

**6. MF Cloud Structure Analysis**
```python
- Density estimation of MF cloud
- Convexity analysis (convex hull volume)
- Overlap quantification with ZINC/TARGETS regions
- "Bridge" vs "cluster" characterization
- Outputs: mf_cloud_structure.pdf, topology_metrics.csv
```

#### Final Deliverable
`reports/pca_dominance_comprehensive_report.pdf`
- Mechanistic explanation of PCA 2D superiority
- Dimension-dependent performance shifts
- Publish-ready figures (PNG + PDF)

---

## File Structure Changes

### New Directories
```
hyperparam_configs_v3/
  phase1_tyro_dimensionality/
    features/
      pca/
      umap_euclidean/
    fingerprints/
      pca/
      umap_jaccard/

ablation_configs_v3/
  phase2_mf_cloud/

generalization_configs_v3/
  phase3_cross_protein/
    pyru/
    iso/

cutoff_configs_v3/
  phase4_affinity_cutoff/

analysis_scripts/
  analyze_pca_dominance.py (NEW)
  
reports_v3/
  phase1_dimensionality/
  phase2_ablation/
  phase3_generalization/
  phase4_cutoff/
  final_comprehensive_report/
```

### Deprecated/Removed
```
- Remove ActinCytoplasmic proteins from experiment_config.json
- Archive v2.0 configs to hyperparam_configs_v2_archive/
- Archive generalization_configs/ to generalization_configs_v2_archive/
```

---

## Configuration File Updates

### `experiment_config.json` Changes

**Update `simspace_dims_to_test`:**
```json
"simspace_dims_to_test": [2, 5, 10]  // Remove 3, 20
```

**Update `representations`:**
```json
"representations": ["features", "fingerprints"]
```

**Remove Actin proteins from `targets` array:**
```json
// DELETE:
{
  "id_name": "ActinCytoplasmic1_P60709",
  ...
},
{
  "id_name": "ActinCytoplasmic2_P63261",
  ...
}
```

**Update `dimensionality_reduction_methods`:**
```json
"dimensionality_reduction_methods": {
  "pca": {"short_name": "PCA"},
  "umap_euclidean": {
    "short_name": "UMAP-Euclidean",
    "metric": "euclidean",
    "n_neighbors_grid": [10, 20, 100, 500],
    "min_dist_grid": [0.01, 0.1, 0.5]
  },
  "umap_jaccard": {
    "short_name": "UMAP-Jaccard",
    "metric": "jaccard",
    "n_neighbors_grid": [10, 20, 100, 500],
    "min_dist_grid": [0.01, 0.1, 0.5]
  }
}
```

### New Config Files

**Phase-specific config templates:**

1. `phase1_config_template.json` - Dimensionality sweep
2. `phase2_config_template.json` - MF ablation (with MF size parameter)
3. `phase3_config_template.json` - Cross-protein (copy best configs)
4. `phase4_config_template.json` - Cutoff analysis (with cutoff parameter)

---

## Script Modifications

### Scripts to Create

**1. `generate_phase1_configs.py`**
```python
Purpose: Generate 260 configs for Phase 1
- Features-PCA: 3 dims
- Features-UMAP: 3 dims × 12 hyperparams
- Fingerprints-PCA: 1 dim
- Fingerprints-UMAP-Jaccard: 1 dim × 12 hyperparams
```

**2. `generate_phase2_configs.py`**
```python
Purpose: Generate MF ablation configs
- Read best PCA and UMAP configs from phase1_results/
- Generate 6 MF size variants per best config
- Total: 60-120 configs (depends on Phase 1 results)
```

**3. `generate_phase3_configs.py`**
```python
Purpose: Generate generalization configs
- Read 8 best configs from Phase 1
- Replicate for Pyru and Iso targets
- Total: 80 configs
```

**4. `generate_phase4_configs.py`**
```python
Purpose: Generate cutoff analysis configs
- Read overall best PCA and UMAP from Phase 1
- Generate 4 cutoff variants each
- Total: 40 configs
```

**5. `extract_phase1_best_configs.py`**
```python
Purpose: Analyze Phase 1 results and extract best configs
- Parse all phase1_results/
- Rank by EF@1% within each method-dimension combo
- Output: phase1_analysis/best_configs_summary.json
```

**6. `orchestrate_v3_pipeline.py`**
```python
Purpose: Sequential phase orchestration
- Run Phase 1 → Extract best → Generate Phase 2 configs → Run Phase 2
- Run Phase 2 → Generate Phase 3 configs → Run Phase 3
- Run Phase 3 → Generate Phase 4 configs → Run Phase 4
- Run Phase 4 → Generate final report
```

**7. `analysis_scripts/analyze_pca_dominance.py`**
```python
Purpose: Comprehensive analysis of PCA 2D dominance
- 6 analysis components (see above)
- Generate comprehensive PDF report
```

### Scripts to Modify

**1. `main_orchestrator.py`**
```python
Changes:
- Add phase parameter (1, 2, 3, 4)
- Update config loading logic
- Add phase-specific result directories
- Add checkpointing between phases
```

**2. `core_scripts/calculate_similarityspaces_exp.py`**
```python
Changes:
- Ensure proper handling of 5D and 10D outputs
- Verify manifold quality metrics collection
- Add distance distribution calculations
```

**3. `experimental_pipeline/rank_zinc_decoys.py`**
```python
Changes:
- No major changes needed
- Verify works with all dimensions
```

**4. Existing analysis scripts**
```python
aggregate_and_report.py: Update for v3.0 structure
aggregate_cutoff_analysis.py: Use Phase 4 results
aggregate_generalization_analysis.py: Use Phase 3 results
```

---

## Orchestrator Logic Flow

### Phase 1 Execution
```
1. Load phase1_config_template.json
2. Generate 260 run configs
3. Execute all runs (parallel where possible)
4. Collect results
5. Run extract_phase1_best_configs.py
6. Generate phase1_analysis reports
7. CHECKPOINT: Wait for user verification
```

### Phase 2 Execution
```
1. Load best configs from phase1_analysis/best_configs_summary.json
2. Generate phase2 MF ablation configs
3. Execute 60 runs
4. Analyze MF cloud crossover
5. Generate phase2_analysis reports
6. CHECKPOINT: Wait for user verification
```

### Phase 3 Execution
```
1. Load best configs from phase1_analysis/
2. Generate phase3 generalization configs
3. Execute 80 runs
4. Analyze transferability
5. Generate phase3_analysis reports
6. CHECKPOINT: Wait for user verification
```

### Phase 4 Execution
```
1. Load overall best configs from phase1_analysis/
2. Generate phase4 cutoff configs
3. Execute 40 runs
4. Analyze cutoff sensitivity
5. Generate phase4_analysis reports
6. Generate FINAL comprehensive report
```

---

## Deliverables

### Per-Phase Deliverables

**Phase 1:**
- Best config JSON per method per dimension
- Dimensionality comparison report (PDF)
- Hyperparameter sensitivity heatmaps (PNG/PDF)
- Performance summary CSV

**Phase 2:**
- MF cloud crossover analysis (PDF)
- Phase transition validation report (PDF)
- Crossover point identification (CSV)

**Phase 3:**
- Generalization performance comparison (PDF)
- Transferability matrix (CSV)
- Same vs different function analysis (PDF)

**Phase 4:**
- Cutoff sensitivity curves (PDF)
- Optimal cutoff recommendations (CSV)
- Method-specific cutoff analysis (PDF)

### Final Comprehensive Report

**File:** `reports_v3/final_comprehensive_report/UMMBAS_v3_Final_Report.pdf`

**Sections:**
1. Executive Summary
2. Phase 1: Dimensionality Analysis
   - Best configurations per method
   - Compression ratio impact
   - Hyperparameter sensitivity
3. Phase 2: MF Cloud Phase Transition
   - Crossover point identification
   - Mechanistic explanation of PCA dominance
4. Phase 3: Cross-Protein Generalization
   - Transferability assessment
   - Same vs different molecular function
5. Phase 4: Affinity Cutoff Sensitivity
   - Optimal thresholds per method
   - Practical recommendations
6. PCA Dominance Analysis (NEW)
   - Variance explained
   - Manifold quality metrics
   - Gradient linearity
   - Distance distributions
   - Compression impact
   - MF cloud structure
7. Conclusions and Recommendations
8. Supplementary Figures (PNG/PDF)

---

## Implementation Checklist

### Configuration Phase
- [ ] Update `experiment_config.json` (remove Actin, update dims)
- [ ] Create phase1_config_template.json
- [ ] Create phase2_config_template.json
- [ ] Create phase3_config_template.json
- [ ] Create phase4_config_template.json

### Generation Scripts
- [ ] Create `generate_phase1_configs.py`
- [ ] Create `generate_phase2_configs.py`
- [ ] Create `generate_phase3_configs.py`
- [ ] Create `generate_phase4_configs.py`
- [ ] Create `extract_phase1_best_configs.py`

### Orchestration
- [ ] Create `orchestrate_v3_pipeline.py`
- [ ] Modify `main_orchestrator.py` for phase support
- [ ] Add checkpointing logic

### Analysis Scripts
- [ ] Create `analysis_scripts/analyze_pca_dominance.py`
- [ ] Update `aggregate_and_report.py` for v3.0
- [ ] Update `aggregate_cutoff_analysis.py` for Phase 4
- [ ] Update `aggregate_generalization_analysis.py` for Phase 3

### Testing
- [ ] Test Phase 1 config generation (verify 260 configs)
- [ ] Test single Phase 1 run end-to-end
- [ ] Test best config extraction logic
- [ ] Test Phase 2 config generation (depends on Phase 1 results)
- [ ] Test Phase 3 config generation
- [ ] Test Phase 4 config generation
- [ ] Test PCA dominance analysis module

### Documentation
- [ ] Create `README_V3_PIPELINE.md`
- [ ] Create `QUICKSTART_V3.md`
- [ ] Update main `README.md` with v3.0 section
- [ ] Document config file format changes
- [ ] Document analysis module usage

---

## Migration from v2.0 to v3.0

### Data Preservation
```bash
# Archive v2.0 configurations
mkdir -p archive_v2.0/
mv hyperparam_configs/ archive_v2.0/hyperparam_configs_v2/
mv generalization_configs/ archive_v2.0/generalization_configs_v2/

# Archive v2.0 results (if present locally)
mv experiment_workspace_hyperparam_sweep_v2/ archive_v2.0/ (if exists)
```

### Config Migration
```bash
# v2.0 configs had:
"simspace_dims_to_test": [2, 3, 5, 10, 20]
# v3.0 will have:
"simspace_dims_to_test": [2, 5, 10]

# v2.0 had 5 targets (including 2 Actins)
# v3.0 has 3 targets (removed Actins)
```

### Hyperparameter Changes
```bash
# v2.0 UMAP grid:
n_neighbors: [5, 10, 15, 20, 30]
min_dist: [0.01, 0.05, 0.1, 0.5, 1.0]
# 25 combinations

# v3.0 UMAP grid:
n_neighbors: [10, 20, 100, 500]
min_dist: [0.01, 0.1, 0.5]
# 12 combinations (52% reduction)
```

---

## Risk Assessment

### Technical Risks

**Risk 1: Phase dependencies**
- **Impact:** High - Each phase depends on previous results
- **Mitigation:** Robust checkpointing, manual verification gates

**Risk 2: Config generation errors**
- **Impact:** Medium - Could generate incorrect experiment configs
- **Mitigation:** Comprehensive testing, validation scripts, dry-run mode

**Risk 3: Best config extraction failure**
- **Impact:** High - Breaks subsequent phases
- **Mitigation:** Multiple ranking criteria, fallback logic, manual override option

**Risk 4: HPC job failures**
- **Impact:** Medium - Could lose partial results
- **Mitigation:** Per-run checkpointing, resume capability

### Scientific Risks

**Risk 1: Dimensionality doesn't help UMAP**
- **Impact:** Low - Still scientifically valuable negative result
- **Expected:** Higher dimensions likely help UMAP catch up to PCA

**Risk 2: Phase transition not observable**
- **Impact:** Low - Hypothesis may be wrong but still publishable
- **Expected:** Clear crossover in 10K-50K range based on v2.0 results

**Risk 3: Poor generalization**
- **Impact:** Medium - May indicate method overfitting to Tyro
- **Expected:** Good generalization to Pyru (same function), moderate to Iso

---

## Timeline Estimates

### Development Phase (Before HPC runs)
```
Week 1-2:
- [ ] Configuration updates
- [ ] Generation scripts
- [ ] Orchestration logic
- [ ] Testing with small test runs

Week 3:
- [ ] Analysis module development
- [ ] Documentation
- [ ] Validation and dry runs
```

### Execution Phase (On HPC)
```
Phase 1: ~1-2 weeks (260 runs, depends on parallelization)
Phase 2: ~1-3 days (60 runs)
Phase 3: ~3-5 days (80 runs)
Phase 4: ~1-2 days (40 runs)

Total HPC time: ~2-3 weeks
```

### Analysis Phase (After all runs)
```
Week 1:
- [ ] Run all analysis scripts
- [ ] Generate per-phase reports

Week 2:
- [ ] PCA dominance analysis
- [ ] Final comprehensive report
- [ ] Figure generation (PNG/PDF)
```

**Total Project Timeline: ~6-8 weeks**

---

## Success Criteria

### Phase 1 Success
- ✅ All 260 runs complete successfully
- ✅ Best configs identified per method per dimension
- ✅ Clear dimensionality trends observable

### Phase 2 Success
- ✅ MF cloud crossover point identified (expected 10K-50K)
- ✅ Phase transition validated
- ✅ Mechanistic understanding of PCA dominance

### Phase 3 Success
- ✅ Generalization performance quantified
- ✅ Same-function transferability confirmed
- ✅ Different-function transferability assessed

### Phase 4 Success
- ✅ Optimal cutoffs identified per method
- ✅ Sensitivity curves generated
- ✅ Practical recommendations provided

### Overall v3.0 Success
- ✅ 440 experiments completed with <5% failure rate
- ✅ Comprehensive understanding of dimensionality impact
- ✅ Publish-ready comparative analysis
- ✅ All figures in PNG + PDF format
- ✅ Reproducible pipeline with clear documentation

---

## Next Steps

**Immediate (This Session):**
1. ✅ Planning document complete
2. 🔄 Work on another task per user request
3. ⏳ Return to implementation later

**When Ready to Implement:**
1. Update `experiment_config.json`
2. Create Phase 1 config generation script
3. Test with small subset (1-2 runs per method)
4. Validate outputs before full HPC deployment
5. Proceed phase by phase with checkpoints

---

## Notes

- **Branch:** All v3.0 work on `3.0` branch
- **Backward Compatibility:** v2.0 results preserved in archive
- **Flexibility:** Manual override options at each phase boundary
- **Reproducibility:** All configs versioned, random seeds fixed
- **Publication Ready:** PNG + PDF for all figures, comprehensive documentation

---

**Document Status:** ✅ Complete and Ready for Review  
**Last Updated:** October 15, 2025  
**Author:** GitHub Copilot + User Collaboration
