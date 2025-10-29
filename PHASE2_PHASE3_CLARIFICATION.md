# Phase 2 and Phase 3 Clarification Document

**Date**: October 29, 2025  
**Status**: Pre-implementation planning document  
**Purpose**: Clarify research questions and experimental designs for Phase 2 and Phase 3

---

## Critical Distinction: Retraining vs Re-scoring

### Phase 2: Affinity Cutoff Sensitivity (Re-scoring Only)
**NO RETRAINING** - Reuse Phase 1 models and embeddings

### Phase 3: MF Cloud Ablation (Full Retraining)
**FULL RETRAINING** - Build new models with reduced MF cloud sizes

---

## Phase 2: Affinity Cutoff Sensitivity Analysis

### Research Question
**Can we improve EF@1% by measuring distance only to more potent ligands from the MF cloud?**

### Hypothesis
Stricter affinity cutoffs (100 nM) will enrich high-potent actives because:
1. High-potency ligands share chemical features required for tight binding
2. Low-potency binders may be chemically diverse and add noise
3. Measuring distance to high-potency-only MF compounds creates a more selective scoring function

Counter-hypothesis: Permissive cutoffs (100 μM) maximize diversity and may capture broader chemical space.

### Experimental Design

#### Inputs (Reused from Phase 1)
- **Models**: Pre-trained PCA/UMAP models from Phase 1 best runs
- **Embeddings**: `embedding_mf.csv`, `embedding_zinc.csv`, `embedding_actives.csv`
- **Selection criteria**: Best 4 Phase 1 runs based on EF@1%:
  1. Best PCA/features
  2. Best PCA/fingerprints (if available; skip if none exist)
  3. Best UMAP/features
  4. Best UMAP/fingerprints

#### Procedure (Per Selected Model)
For each affinity cutoff in `[100, 1000, 10000, 100000]` nM:

1. **Load pre-computed embeddings** from Phase 1 artifacts (no re-projection)
2. **Re-read MF source CSV** to access `Standard Value (nM)` column
3. **Filter MF embedding** by affinity cutoff (keep only rows ≤ cutoff)
4. **Re-score evaluation set** (actives + ZINC) via exact 1-NN to filtered MF cloud
5. **Compute metrics**: EF@1/5/10%, ROC-AUC, PR-AUC, Spearman rho
6. **Save outputs**: Per-cutoff metrics and ranked scores

#### Key Invariants
- **No model retraining**: Phase 1 DR models and scalers are frozen
- **No re-projection**: Embeddings are loaded from Phase 1 CSVs
- **Only scoring changes**: MF cloud filtering affects only the distance measurement
- **Actives never filtered**: Cutoff applies to MF scoring set only
- **Same evaluation set**: Actives + ZINC are identical across cutoffs (same Phase 1 sample)

#### Expected Outputs (Per Selected Model × 4 Cutoffs = 16 runs)
```
workspace/phase2/cutoff_sweep/
├── logs/
│   ├── run.log
│   └── phase2_summary.json
├── selected_models.json  # Which Phase 1 runs were selected
├── <phase1_run_name_1>/
│   ├── cutoff_100nM/
│   │   ├── metrics.json       # EF@1/5/10%, AUCs, Spearman
│   │   └── ranked_scores.csv  # Re-scored actives+ZINC
│   ├── cutoff_1000nM/
│   ├── cutoff_10000nM/
│   └── cutoff_100000nM/
└── <phase1_run_name_2>/...
```

#### Analysis Questions
1. Does EF@1% improve with stricter cutoffs for high-potency actives?
2. Do PCA and UMAP show different cutoff sensitivities?
3. What is the quality-quantity trade-off (MF cloud size vs enrichment)?
4. Which cutoff optimizes performance for each method/representation combo?

---

## Phase 3: MF Cloud Ablation Study

### Research Question
**What happens to the similarity space when the MF cloud size becomes smaller? Does performance degrade?**

### Hypothesis
**Expected degradation**: Virtual screening performance will degrade as MF cloud size decreases because:
1. Smaller MF clouds reduce chemical diversity coverage
2. Fewer training samples may lead to overfitting or unstable embeddings
3. Distance-based scoring becomes less reliable with sparse reference sets

Secondary hypothesis (phase transition): UMAP may be more sensitive to MF cloud size than PCA due to its local manifold learning approach.

### Experimental Design

#### Inputs (Selected from Phase 2 Results)
- **Best cutoff**: Optimal affinity cutoff identified in Phase 2 (likely per method/representation)
- **Best hyperparameters**: Phase 1 best configs (n_neighbors, min_dist, dimensions)
- **Models to test**: Same 4 as Phase 2 (PCA/UMAP × features/fingerprints)

#### Procedure (Per Selected Model × 6 MF Sizes = 24 runs)
For each MF cloud size in `[0, 1000, 10000, 50000, 100000, full_size]`:

1. **Subsample MF cloud** by `Standard Value (nM)` ≤ optimal_cutoff (random sample to target size)
2. **Load ZINC and actives** from original Phase 1 CSVs (same held-out sets)
3. **Fit scaler** on subsampled MF + ZINC
4. **Train DR model** on subsampled MF + ZINC (NEW model with same hyperparams as Phase 1 best)
5. **Project actives** using the new scaler + model
6. **Score** via exact 1-NN to the subsampled MF cloud
7. **Compute metrics**: EF@1/5/10%, ROC-AUC, PR-AUC, Spearman rho
8. **Save models and embeddings**: Full artifacts as in Phase 1

#### Key Invariants
- **FULL RETRAINING**: New scaler and DR model fitted for each MF size
- **Same hyperparameters**: Use Phase 1 best configs (frozen n_neighbors, min_dist, dim)
- **Same held-out sets**: ZINC and actives are identical across MF sizes (no re-sampling)
- **Reproducibility**: Each MF subsample is seeded (or documented) for reproducibility
- **Size "0" control**: MF=0 means scoring against ZINC only (no MF cloud)

#### Expected Outputs (Per Model × 6 Sizes = 24 runs)
```
workspace/phase3/mf_ablation/
├── logs/
│   ├── run.log
│   └── phase3_summary.json
├── selected_models_and_cutoff.json  # Phase 2 best cutoff + Phase 1 best configs
├── <method>_<repr>_<dim>d_mf0/
│   ├── artifacts/
│   │   ├── scaler.joblib           # NEW scaler (ZINC-only for MF=0)
│   │   ├── {pca|umap}_model.joblib # NEW model
│   │   ├── embedding_mf.csv        # Empty or ZINC-only
│   │   ├── embedding_zinc.csv
│   │   └── embedding_actives.csv
│   ├── metrics.json
│   └── ranked_scores.csv
├── <method>_<repr>_<dim>d_mf1000/
├── <method>_<repr>_<dim>d_mf10000/
├── <method>_<repr>_<dim>d_mf50000/
├── <method>_<repr>_<dim>d_mf100000/
└── <method>_<repr>_<dim>d_mf_full/
```

#### Analysis Questions
1. At what MF cloud size does performance begin to degrade significantly?
2. Is there a critical "phase transition" point where PCA overtakes UMAP?
3. Does the degradation curve differ between features and fingerprints?
4. What is the minimum viable MF cloud size for useful enrichment?

---

## Experimental Dependency Chain

```
Phase 1: Hyperparameter Optimization
   ↓
   ├─→ Phase 2: Affinity Cutoff Sensitivity (Re-scoring)
   │      ↓
   │      └─→ Identify optimal cutoff per method/representation
   │             ↓
   └───────────→ Phase 3: MF Cloud Ablation (Retraining with optimal cutoff)
                    ↓
                    └─→ Phase 4: Cross-Protein Generalization
```

### Critical Path Logic
1. **Phase 1** identifies best hyperparameters (n_neighbors, min_dist, dimensions)
2. **Phase 2** identifies optimal affinity cutoff using Phase 1 models (no retraining)
3. **Phase 3** uses Phase 1 hyperparameters + Phase 2 cutoff to study MF cloud size effects (full retraining)
4. **Phase 4** uses Phase 1 hyperparameters + Phase 2 cutoff to test cross-protein generalization

---

## Implementation Priorities (Next Steps)

### Phase 2 Implementation (Current Focus)
1. Model selection utility (`molfuse/analysis/select_best_phase1.py`)
2. Phase 2 CLI (`molfuse/cli/phase2.py`) - re-scoring only
3. Config generator (`scripts/generate_molfuse_phase2_configs_v4.py`)
4. Post-analysis script (`scripts/phase2_post_analysis.py`)
5. HPC scripts (`hpc/molfuse_phase2_cpu.sh`, `hpc/submit_molfuse_phase2.sh`)

### Phase 3 Implementation (After Phase 2 Completion)
- Defer until Phase 2 results identify optimal cutoff
- Will reuse Phase 1 training pipeline with modified MF sampling logic
- Expected timeline: After Phase 2 analysis (1-2 weeks post-Phase 2 execution)

---

## Validation Checklist

### Phase 2 Validation
- [ ] Verify no model retraining occurs (confirm scaler/model loading only)
- [ ] Confirm embeddings are loaded from Phase 1 CSVs (no re-projection)
- [ ] Validate MF cloud filtering logic (cutoff application to pre-computed embeddings)
- [ ] Check metrics match expected ranges (EF@1% should vary with cutoff)
- [ ] Ensure actives are never filtered by affinity

### Phase 3 Validation
- [ ] Confirm NEW scalers and models are trained for each MF size
- [ ] Verify MF subsampling logic (reproducible random samples)
- [ ] Validate MF=0 control (ZINC-only scoring, no MF cloud)
- [ ] Check held-out sets remain constant across MF sizes
- [ ] Ensure hyperparameters match Phase 1 best configs

---

## Documentation Updates Required

### PLANNING.md
- [x] Clarify Phase 2 = re-scoring only (no retraining)
- [x] Clarify Phase 3 = full retraining with MF ablation
- [ ] Update task checklists to reflect correct experimental designs

### LAB_BOOK.md
- [ ] Add entry documenting Phase 2/3 clarification (Oct 29, 2025)
- [ ] Link to this clarification document
- [ ] Record research questions and hypotheses

### README_V4_MOLFUSE.md
- [ ] Add Phase 2 usage section (re-scoring workflow)
- [ ] Add Phase 3 usage section (retraining workflow)
- [ ] Document dependency chain (Phase 1 → 2 → 3 → 4)

---

## Key Takeaways

1. **Phase 2 is computationally cheap**: Only re-scoring (minutes per model × cutoff)
2. **Phase 3 is computationally expensive**: Full retraining (hours per model × MF size)
3. **Phase 2 informs Phase 3**: Optimal cutoff from Phase 2 is critical input to Phase 3
4. **Both test different hypotheses**:
   - Phase 2: Does scoring against high-potency-only MF improve enrichment?
   - Phase 3: Does reducing MF cloud size degrade performance (and by how much)?

---

**End of Clarification Document**
