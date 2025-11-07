# MolFuSE Experimental Program

**Framework**: Molecular Function-Guided Similarity Enhancement (MolFuSE)  
**Version**: 4.0  
**Status**: Phase 1 In Progress, Phase 2 Ready, Phase 3-4 Planned  
**Date**: November 2025

---

## Overview

The MolFuSE experimental program is designed to systematically validate the molecular function proximity hypothesis through four sequential phases. Each phase addresses specific research questions while building upon insights from previous phases.

**Experimental Dependency Chain**:
```
Phase 1 (Hyperparameters) → Phase 2 (Cutoff Sensitivity) → Phase 3 (MF Ablation) → Phase 4 (Generalization)
```

**Key Principle**: No retraining in Phase 2 (re-scoring only); full retraining in Phase 3 (ablation study).

---

## Phase 1: Hyperparameter Optimization

### Research Questions

1. **Primary**: What are optimal hyperparameters and dimensionality for PCA vs UMAP in molecular function-guided virtual screening?
2. **Secondary**: Does UMAP's nonlinear manifold learning provide advantage over PCA's linear projection?
3. **Mechanistic**: How does UMAP's neighborhood size (n_neighbors) affect enrichment performance?

### Hypotheses

**H1 (UMAP Superiority)**: UMAP will outperform PCA by capturing nonlinear manifold structure in molecular space.

**H2 (Dimensionality Effects)**: Higher dimensions (10D) will improve performance for both methods by preserving more information.

**H3 (Hyperparameter Sensitivity)**: UMAP performance will be highly sensitive to n_neighbors; small neighborhoods will create tighter, more discriminative clusters.

### Experimental Design

**Target**: ABL1 (Tyrosine-protein kinase; ChEMBL ID: P00519)

**Molecular Function**: Transferase (Kinase subfamily)

**Dataset Composition**:
- **MF Cloud**: ~191K unique compounds (ChEMBL "Transferase" keyword, excluding ABL1 actives)
- **Decoys**: 1,295,279 ZINC compounds (drug-like, clean-leads subset)
- **Target Actives**: 3,313 ABL1 actives (IC₅₀/Ki/Kd ≤ 100 μM)

**Molecular Representation**:
1. **Features**: Full 2D Mordred descriptors (~1,477 after cleaning)
2. **Fingerprints**: ECFP4 (2,048-bit, radius=2)

**Methods**:
1. **PCA**: Dimensions = {2, 5, 10}
2. **UMAP**: Dimensions = {2, 5, 10}
   - n_neighbors = {10, 20, 50, 100, 500}
   - min_dist = {0.0, 0.001, 0.005, 0.01, 0.1}
   - metric = Euclidean (features), Jaccard (fingerprints)

**Replicates**: 5 independent runs per configuration (for UMAP stochasticity assessment)

**Total Configurations**: 630
- PCA-features: 3 dims × 5 replicates = 15
- PCA-fingerprints: 3 dims × 5 replicates = 15
- UMAP-features: 3 dims × 5 nn × 5 md × 5 replicates = 375
- UMAP-fingerprints: 3 dims × 5 nn × 5 md × 5 replicates = 375 (marked for future investigation)

### Evaluation Metrics

**Primary**: Enrichment Factor at 1% (EF@1%)

**Secondary**:
- EF@5%, EF@10%
- ROC-AUC (global ranking quality)
- PR-AUC (precision-recall, class-imbalance aware)
- Spearman ρ (score vs pActivity correlation for actives)

**Variance**: Coefficient of variation across 5 replicates (UMAP only)

### Expected Outcomes

**Baseline Performance**: EF@1% = 57-58 for PCA (from v3 results)

**UMAP Performance**: Expected EF@1% = 40-50 (best hyperparameters)

**Fingerprints**: Expected to underperform features by 2-3× (binary vectors incompatible with PCA)

### Computational Resources

**Platform**: HPC cluster (64 cores, 64-200 GB RAM per node)

**Runtime per Configuration**:
- PCA: ~3-5 minutes
- UMAP (nn=10-50): ~10-15 minutes
- UMAP (nn=500): ~30-45 minutes

**Total Wall Clock**: ~30-40 hours sequential; ~2-3 hours parallel (32 concurrent jobs)

### Current Status (November 2025)

**Progress**: 
- ✅ Config generation: 630 configs created
- ✅ Pipeline adaptation: Full 2D Mordred integration complete
- ✅ Optimization: Memory reduced 10× (300GB → 30GB)
- ⏳ HPC execution: In progress
- ⏳ Analysis: Pending completion

**Deliverables**:
- `phase1_best_configs.json`: Best hyperparameters per method/representation/dimension
- Hyperparameter sensitivity plots (EF@1% heatmaps)
- Dimensionality impact analysis
- Seed variance quantification (UMAP)

---

## Phase 2: Affinity Cutoff Sensitivity Analysis

### Research Questions

1. **Primary**: Can we improve EF@1% by measuring distance only to high-potency ligands from the MF cloud?
2. **Secondary**: Do PCA and UMAP prefer different affinity cutoffs?
3. **Mechanistic**: How does cutoff affect enrichment across potency tiers (high/medium/weak)?
4. **Decision**: What is the optimal cutoff for Phase 3 (MF cloud ablation)?

### Hypotheses

**H1 (Strictness Improves Selectivity)**: Stricter affinity cutoffs (100 nM) will enrich high-potency actives because high-potency ligands share chemical features required for tight binding.

**H2 (Permissiveness Increases Diversity)**: Permissive cutoffs (100 μM) maximize diversity and may capture broader chemical space, but at the cost of including non-specific binders.

**H3 (Method-Specific Cutoffs)**: PCA may prefer permissive cutoffs (global variance), while UMAP may prefer stricter cutoffs (local structure).

### Experimental Design

**Critical Design Principle**: **RE-SCORING ONLY** (NO model retraining)

**Input**: Phase 1 best models and embeddings (reused from Phase 1 artifacts)

**Procedure**:
1. Load Phase 1 pre-computed embeddings: `embedding_mf.csv`, `embedding_zinc.csv`, `embedding_actives.csv`
2. Load MF source CSV to access `Standard Value (nM)` column
3. For each cutoff: filter MF embedding by affinity, re-score evaluation set via 1-NN, compute metrics
4. Save per-cutoff metrics and ranked scores

**Selected Models** (from Phase 1):
- Best PCA-features (by EF@1%)
- Best UMAP-Euclidean-features (by EF@1%)

**Cutoffs Tested**:
- **100 nM**: High-potent only (drug-like, clinically relevant)
- **1 μM**: High + Medium potency (balanced)
- **10 μM**: High + Medium + some Weak (permissive)
- **100 μM**: All potencies (maximum diversity, Phase 1 default)

**Potency Tier Definitions** (aligned with stratified enrichment analysis):
- **High**: 0.1-100 nM
- **Medium**: 100-1,000 nM
- **Weak**: 1,000-100,000 nM

**Replicates**: 5 per cutoff (reusing Phase 1 seeds/embeddings)

**Total Configurations**: 40
- 2 methods × 4 cutoffs × 5 replicates = 40

### Invariants

**No Retraining**: Phase 1 DR models and scalers are frozen; embeddings loaded from CSV

**No Re-projection**: MF, ZINC, and actives embeddings reused exactly as computed in Phase 1

**Only Scoring Changes**: MF cloud filtering affects only the distance measurement step

**Actives Never Filtered**: Cutoff applies to MF scoring set only; evaluation set unchanged

**Same Evaluation Set**: Actives + ZINC are identical across cutoffs (same Phase 1 sample)

### Evaluation Metrics

**Primary**: EF@1% (overall and stratified by potency tier)

**Secondary**:
- EF@5%, EF@10%
- ROC-AUC, PR-AUC
- Spearman ρ (if actives have affinity data)

**Stratified Analysis**:
- EF@1% for high-potent actives only
- EF@1% for medium-potent actives only
- EF@1% for weak-potent actives only

### Expected Outcomes

**Hypothesis A (Strictness Wins)**: 100 nM cutoff → highest EF@1% for high-potent actives

**Hypothesis B (Balance Wins)**: 1-10 μM cutoff → best overall EF@1% (balances quality and diversity)

**Hypothesis C (Method Divergence)**: PCA and UMAP show different optimal cutoffs

### Computational Resources

**Runtime**: ~10-15 minutes per cutoff (re-scoring only, no DR fitting)

**Total Wall Clock**: ~1 hour sequential; ~10 minutes parallel

**Speedup vs Full Pipeline**: 75% time savings (reusing Phase 1 data)

### Current Status (November 2025)

**Progress**:
- ✅ Config generator: `generate_molfuse_phase2_configs_v4.py` complete
- ✅ Orchestrator: `hpc/submit_molfuse_phase2.sh` ready
- ✅ CLI: `molfuse/cli/phase2.py` complete with median deduplication
- ✅ Compatibility verified: Phase 2 works with Phase 1 v4 outputs
- ⏳ Execution: Blocked pending Phase 1 completion

**Deliverables**:
- Cutoff sensitivity curves (EF@1% vs cutoff)
- Potency-stratified enrichment tables
- Optimal cutoff recommendation for Phase 3
- Method-specific cutoff preferences (if divergent)

---

## Phase 3: Molecular Function Cloud Ablation Study

### Research Questions

1. **Primary**: What happens to similarity space quality when MF cloud size decreases?
2. **Secondary**: Does performance degrade with smaller MF clouds due to reduced diversity?
3. **Mechanistic**: Do PCA and UMAP respond differently to MF cloud size changes?
4. **Phase Transition**: Is there a critical MF cloud size where UMAP begins to outperform PCA?

### Hypotheses

**H1 (Degradation with Size)**: Performance will degrade with smaller MF clouds due to reduced chemical diversity coverage.

**H2 (Overfitting Risk)**: Fewer training samples may lead to overfitting or unstable embeddings.

**H3 (Scoring Reliability)**: Distance-based scoring becomes less reliable with sparse reference sets.

**H4 (Method Reversal)**: Small MF clouds favor UMAP (local structure preservation); large MF clouds favor PCA (global variance preservation). Expected crossover at ~10K-50K molecules.

**H5 (Dimensionality Interaction)**: Effect of MF size may interact with dimensionality (low-D more sensitive).

### Experimental Design

**Critical Design Principle**: **FULL RETRAINING** for each MF cloud size

**Input**: Phase 1 best hyperparameters + Phase 2 optimal affinity cutoff

**Procedure**:
1. Subsample MF cloud to target size (stratified by target to preserve diversity)
2. Retrain scaler on subsampled MF + ZINC
3. Fit DR model (PCA or UMAP) on subsampled MF + ZINC
4. Project actives using new model
5. Score and compute metrics

**MF Cloud Sizes**:
- **0 molecules**: ZINC-only control (no molecular function guidance)
- **1,000 molecules**: Sparse MF cloud
- **10,000 molecules**: Small MF cloud
- **50,000 molecules**: Medium MF cloud
- **100,000 molecules**: Large MF cloud
- **Full (~191K)**: Phase 1 default (maximum diversity)

**Selected Models** (from Phase 1):
- Best PCA-features configuration
- Best UMAP-features configuration

**Affinity Cutoff**: Optimal cutoff from Phase 2 (applied consistently across all MF sizes)

**Replicates**: 5 per MF size (different random subsamples)

**Total Configurations**: 60
- 2 methods × 6 MF sizes × 5 replicates = 60

### Sampling Strategy

**Stratified Sampling**: For each MF size, sample proportionally from each target protein to preserve target diversity (not just top N most common).

**Seed Variation**: Each replicate uses different random seed for subsampling (tests robustness to sample selection).

### Evaluation Metrics

**Primary**: EF@1%

**Secondary**:
- EF@5%, EF@10%
- ROC-AUC, PR-AUC
- Silhouette score (cluster separation in embedded space)
- Explained variance (for PCA)
- Reconstruction error (for both methods)

**Diagnostic Metrics**:
- MF cloud density (mean pairwise distance)
- MF cloud radius (distance from centroid to farthest point)
- Embedding stability (correlation between replicate embeddings)

### Expected Outcomes

**MF = 0**: UMAP expected to outperform PCA (local structure in ZINC alone)

**MF = 1K-10K**: Performance degradation for both methods, but UMAP may be more robust

**MF = 50K-191K**: PCA expected to outperform UMAP (as observed in Phase 1)

**Crossover Point**: Expected at ~10K-50K molecules (to be determined empirically)

### Computational Resources

**Runtime per Configuration**:
- MF=0: ~5 minutes (small training set)
- MF=1K-10K: ~10-15 minutes
- MF=50K-191K: ~30-60 minutes (same as Phase 1)

**Total Wall Clock**: ~20-30 hours sequential; ~2-3 hours parallel

### Current Status (November 2025)

**Progress**:
- ✅ Config generator skeleton: `generate_molfuse_phase3_configs_v4.py` exists
- ⏳ Update needed: Incorporate Phase 2 optimal cutoff
- ⏳ CLI: `molfuse/cli/phase3.py` to be created
- ⏳ Execution: Blocked pending Phase 2 completion

**Deliverables**:
- MF size sensitivity curve (EF@1% vs MF size)
- PCA vs UMAP crossover point identification
- Embedding quality metrics vs MF size
- Subsampling robustness analysis (variance across replicates)

---

## Phase 4: Cross-Protein Generalization Study

### Research Questions

1. **Primary**: Do optimal configurations from Phase 1 transfer across different proteins?
2. **Secondary**: Does molecular function similarity affect transferability?
3. **Mechanistic**: Are hyperparameters target-specific or molecular function-specific?

### Hypotheses

**H1 (Same-MF Transfer)**: Configurations optimized on ABL1 will transfer to other transferases (PKM2) with minimal performance loss.

**H2 (Different-MF Degradation)**: Performance will degrade when transferring to proteins in different molecular functions (IDH1, oxidoreductase).

**H3 (Universal Hyperparameters)**: Some hyperparameters (e.g., PCA dimensionality) may be universal across targets, while others (e.g., UMAP n_neighbors) may be target-specific.

### Experimental Design

**Test Targets**:

1. **Pyruvate Kinase M2 (PKM2)**
   - ChEMBL ID: P14618
   - Molecular Function: **Transferase** (same as ABL1)
   - Actives: ~2,000 compounds (IC₅₀/Ki/Kd ≤ 100 μM)
   - Purpose: Test same-MF transfer

2. **Isocitrate Dehydrogenase 1 (IDH1)**
   - ChEMBL ID: O75874
   - Molecular Function: **Oxidoreductase** (different from ABL1)
   - Actives: ~1,500 compounds (IC₅₀/Ki/Kd ≤ 100 μM)
   - Purpose: Test different-MF transfer

**MF Clouds**:
- **PKM2**: Same as ABL1 (Transferase cloud, ~191K compounds, excluding PKM2 actives)
- **IDH1**: Oxidoreductase cloud (~500K compounds, excluding IDH1 actives)

**Transferred Configurations**:
- Best PCA-features from Phase 1
- Best UMAP-features from Phase 1
- Best PCA-fingerprints from Phase 1 (if available)
- Best UMAP-fingerprints from Phase 1 (if available)

**Affinity Cutoff**: Optimal cutoff from Phase 2 (applied consistently)

**MF Cloud Size**: Optimal size from Phase 3 (if different from full)

**Replicates**: 5 per target

**Total Configurations**: 40 (approximate)
- 4 best configs × 2 targets × 5 replicates = 40

### Evaluation Metrics

**Primary**: EF@1%

**Comparison**:
- Absolute performance on new target
- Relative performance vs Phase 1 (ABL1)
- Performance delta (new - ABL1)

**Statistical Testing**:
- Paired t-test (same config on ABL1 vs PKM2)
- Unpaired t-test (same config on ABL1 vs IDH1)
- Significance threshold: p < 0.05

### Expected Outcomes

**PKM2 (Same-MF)**:
- Expected: EF@1% within 80-120% of ABL1 performance
- Hypothesis: Transferase cloud generalizes across kinases

**IDH1 (Different-MF)**:
- Expected: EF@1% = 50-80% of ABL1 performance
- Hypothesis: Oxidoreductase chemistry differs from transferase chemistry
- Alternative: May perform well if optimal hyperparameters are universal

**Transferability Index**:
$$
\text{TI} = \frac{\text{EF@1%}_{\text{new target}}}{\text{EF@1%}_{\text{ABL1}}}
$$

- TI > 0.8: Good transfer
- TI = 0.5-0.8: Moderate transfer
- TI < 0.5: Poor transfer

### Computational Resources

**Runtime per Configuration**: ~30-60 minutes (same as Phase 1)

**Total Wall Clock**: ~20-40 hours sequential; ~2-3 hours parallel

### Current Status (November 2025)

**Progress**:
- ✅ Config generator skeleton: `generate_molfuse_phase4_configs_v4.py` exists
- ⏳ Update needed: Incorporate Phase 2/3 optimal parameters
- ⏳ CLI: Can reuse Phase 1 CLI with different target configs
- ⏳ Execution: Blocked pending Phase 3 completion

**Deliverables**:
- Transferability report (performance comparison table)
- Same-MF vs different-MF transfer analysis
- Hyperparameter universality assessment
- Target-specific optimization recommendations (if needed)

---

## Cross-Phase Analysis

### Integrated Research Questions

1. **Hyperparameter Universality**: Are Phase 1 optimal hyperparameters consistent across cutoffs (Phase 2), MF sizes (Phase 3), and targets (Phase 4)?

2. **Method Selection Criteria**: Under what conditions (cutoff, MF size, target) does UMAP outperform PCA?

3. **Practical Recommendations**: What is the recommended MolFuSE configuration for new targets with minimal optimization?

### Expected Synthesis

**Robust Configuration**: Combination of hyperparameters that performs well across all phases.

**Decision Tree**: Guidelines for choosing PCA vs UMAP based on:
- Molecular function annotation quality
- Available MF cloud size
- Computational budget
- Desired interpretability

### Final Deliverables

1. **Unified Results Table**: All phases, all metrics, all configurations
2. **Best Practices Guide**: Recommended workflow for applying MolFuSE to new targets
3. **Limitation Analysis**: Failure modes and when to use alternative methods
4. **Computational Cost Summary**: Runtime and memory requirements across phases

---

## Timeline

**Phase 1**: 2-3 weeks (HPC execution + analysis)  
**Phase 2**: 1 week (fast re-scoring + analysis)  
**Phase 3**: 2-3 weeks (full retraining + analysis)  
**Phase 4**: 2-3 weeks (new targets + analysis)  
**Integration**: 1-2 weeks (cross-phase analysis + manuscript prep)

**Total Estimated Duration**: 10-14 weeks

---

## Quality Control and Reproducibility

### Data Integrity Checks

**Overlap Verification**: Zero MF-ZINC, MF-actives, ZINC-actives overlap (SMILES-based)

**Deduplication Validation**: One row per unique SMILES (median affinity for duplicates)

**Target Exclusion**: Confirm target actives absent from MF cloud

**Affinity Filtering**: Verify cutoff application (MF only, actives never filtered)

### Computational Reproducibility

**Random Seeds**: Documented and consistent across phases

**Software Versions**: Pinned dependencies (requirements.txt)

**Configuration Manifest**: All hyperparameters saved in JSON configs

**Artifact Preservation**: All intermediate files saved (embeddings, models, rankings)

### Statistical Rigor

**Replicates**: 5 minimum for variance estimation

**Multiple Testing Correction**: Bonferroni or FDR correction for hyperparameter comparisons

**Confidence Intervals**: 95% CI reported for all primary metrics

**Significance Thresholds**: p < 0.05 (clearly stated in results)

---

## Conclusion

The MolFuSE experimental program represents a comprehensive investigation of molecular function-guided virtual screening across four complementary dimensions: hyperparameter optimization (Phase 1), affinity filtering (Phase 2), training set size (Phase 3), and target generalization (Phase 4). Together, these experiments will establish the method's capabilities, limitations, and optimal application strategies for ligand-based drug discovery.
