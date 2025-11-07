# Phase 3: MF Cloud Ablation Study - Usage Guide

## Overview

**Research Question**: What happens to the similarity space when the MF cloud size becomes smaller? Does performance degrade?

**Hypothesis**: Performance will degrade as MF cloud size decreases due to reduced chemical diversity coverage.

## Experimental Design

- **MF Cloud Sizes**: `[10, 100, 1000, 10000, 100000, full]` (6 sizes)
- **Methods**: 4 best configs from Phase 1 (PCA/UMAP × features/fingerprints)
- **Replicates**: 5 replicates per (method × MF size) for statistical robustness
- **Total Runs**: **120** (4 methods × 6 sizes × 5 replicates)
- **Retraining**: FULL RETRAINING for each MF size (new scaler + DR model)
- **Hyperparameters**: Frozen (use Phase 1 best)
- **Affinity Cutoff**: Method-specific optimal from Phase 2
- **Subsampling**: Random (reproducible via seed: replicate × 42)
- **Held-out Sets**: ZINC + actives remain constant across all MF sizes

## Workflow

### Step 1: Generate Phase 3 Configs

**CRITICAL**: Requires Phase 1 and Phase 2 results. No hardcoded defaults.

```bash
# Default: reads from reporting/phase1_post_analysis/ and reporting/phase2_post_analysis/
python scripts/generate_molfuse_phase3_configs_v4.py \
    --output_dir configs/molfuse_phase3_grid

# Custom paths (if results are in different locations):
python scripts/generate_molfuse_phase3_configs_v4.py \
    --phase1_grouped reporting/phase1_post_analysis/phase1_summary_grouped.csv \
    --phase2_best_cutoffs reporting/phase2_post_analysis/phase2_best_cutoffs.json \
    --output_dir configs/molfuse_phase3_grid

# Custom MF sizes and replicates
python scripts/generate_molfuse_phase3_configs_v4.py \
    --mf_sizes "10,50,100,500,1000,full" \
    --replicates "1,2,3,4,5" \
    --output_dir configs/molfuse_phase3_grid
```

**What it does**:
1. Reads Phase 1 best hyperparameters (dim, n_neighbors, min_dist) from `phase1_summary_grouped.csv`
2. Reads Phase 2 optimal cutoffs from `phase2_best_cutoffs.json`
3. Generates **120 configs** (4 methods × 6 MF sizes × 5 replicates)
4. Each config uses **actual experimental results**, not assumptions

**Example Output**:
```
Phase 1 Best Configurations:
  pca_features: pca/features (dim=20)
  pca_fingerprints: pca/fingerprints (dim=20)
  umap_features: umap/features (dim=2) (n_neighbors=10, min_dist=0.01)
  umap_fingerprints: umap/fingerprints (dim=20) (n_neighbors=10, min_dist=0.0)

Phase 2 Optimal Cutoffs:
  pca_features: 1000 nM (EF@1% = 25.07)
  pca_fingerprints: 100 nM (EF@1% = 16.57)
  umap_features: 100 nM (EF@1% = 42.96)
  umap_fingerprints: 100 nM (EF@1% = 45.08)

✓ Generated 120 Phase 3 configs
```

### Step 2: Submit to HPC (Parallel Execution)

```bash
# Dry run (check what will be submitted)
bash hpc/submit_molfuse_phase3.sh --dry-run \
    configs/molfuse_phase3_grid \
    experiment_workspace_v4

# Actual submission
bash hpc/submit_molfuse_phase3.sh \
    configs/molfuse_phase3_grid \
    experiment_workspace_v4
```

**Features**:
- Idempotent: Skips runs with existing `phase3_summary.json`
- Parallel: All 120 jobs run simultaneously (subject to cluster limits)
- Robust: Each job is independent

### Step 3: Monitor Jobs

```bash
# Check job status
squeue -u $USER

# Watch jobs in real-time
watch -n 5 squeue -u $USER

# Check individual logs
tail -f logs/slurm/phase3_*.out
```

### Step 4: Analyze Results

```bash
# Aggregate results and generate plots
python scripts/phase3_post_analysis.py \
    --workspace_dir experiment_workspace_v4 \
    --phase3_run_name mf_ablation \
    --output_dir reporting/phase3_post_analysis
```

**Expected Outputs**:
- `phase3_aggregated_metrics.csv`: All results across MF sizes
- `mf_size_degradation_curves.png/pdf`: EF@1% vs MF size (4 methods)
- `phase_transition_analysis.json`: Critical MF size thresholds

## Local Testing

```bash
# Test single config locally
python -m molfuse.cli.phase3 \
    --config configs/molfuse_phase3_grid/pca_features_mf1000.json \
    --workspace test_workspace_phase3
```

## Output Structure

```
experiment_workspace_v4/phase3/mf_ablation/
├── logs/
│   ├── run.log
│   └── phase3_summary.json
├── pca_features_dim20_mf10_rep1/
│   ├── artifacts/
│   │   ├── scaler.joblib
│   │   ├── pca_model.joblib
│   │   ├── embedding_mf.csv
│   │   ├── embedding_zinc.csv
│   │   ├── embedding_actives.csv
│   │   └── ranked_scores.csv
│   ├── logs/
│   │   ├── run.log
│   │   └── phase3_summary.json
│   └── metrics/
│       └── metrics.json
├── pca_features_dim20_mf10_rep2/
├── ... (120 total runs: 4 methods × 6 sizes × 5 replicates)
└── umap_fingerprints_dim20_mf_full_rep5/
```

## Key Metrics

Each run computes:
- **EF@1%, EF@5%, EF@10%**: Enrichment factors
- **ROC-AUC, PR-AUC**: Global classification performance
- **Spearman ρ**: Correlation with experimental affinity (actives only)
- **MF size**: Actual size after subsampling

## Expected Results

1. **Degradation Curve**: EF@1% decreases as MF size decreases
2. **Phase Transition**: Critical MF size where performance drops sharply
3. **Method Comparison**: PCA vs UMAP sensitivity to MF cloud size
4. **Representation**: Features vs fingerprints degradation patterns

## Troubleshooting

### Issue: Empty MF cloud after cutoff
**Solution**: Check affinity cutoff is appropriate for your dataset. Most MF compounds should pass the cutoff.

### Issue: Out of memory
**Solution**: Increase `--mem` in `hpc/molfuse_phase3_cpu.sh` (default: 64G)

### Issue: Slow UMAP training
**Expected**: UMAP is slower than PCA. For large MF sizes (100K+), expect 1-4 hours per run.

### Issue: Different results from Phase 1
**Expected**: Phase 3 uses smaller MF clouds → different embeddings → different scores. This is the research question!

## Research Questions

1. **At what MF size does performance degrade significantly?**
   - Hypothesis: Below 1000 compounds, enrichment drops sharply

2. **Is there a phase transition point?**
   - Look for inflection point in degradation curves

3. **Do PCA and UMAP respond differently?**
   - UMAP may be more sensitive to small MF sizes (local manifold learning)

4. **What is the minimum viable MF cloud size?**
   - Practical threshold for future experiments

## Next Steps After Phase 3

1. **Document findings** in `LAB_BOOK.md`
2. **Update `PUBLICATION.md`** with key results
3. **Proceed to Phase 4**: Cross-protein generalization study
4. **Publication plots**: Export high-resolution figures for manuscript
