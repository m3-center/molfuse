# Phase 2 Usage Example

## Quick Start

### 1. Generate Phase 2 Config
```bash
python scripts/generate_molfuse_phase2_configs_v4.py
```

This creates `configs/molfuse_phase2_grid/phase2_cutoff_sweep.json` with default cutoffs [100, 1000, 10000, 100000] nM.

### 2. Run Phase 2 Locally
```bash
python -m molfuse.cli.phase2 \
    --config configs/molfuse_phase2_grid/phase2_cutoff_sweep.json \
    --workspace experiment_workspace_v4
```

**What happens**:
- Scans `experiment_workspace_v4/phase1/` for completed runs
- Selects best 4 models (PCA/features, PCA/fingerprints, UMAP/features, UMAP/fingerprints) by EF@1%
- For each selected model × 4 cutoffs = up to 16 re-scoring runs
- Loads pre-computed embeddings from Phase 1 (NO retraining)
- Filters MF embedding by affinity cutoff
- Re-scores actives+ZINC via 1-NN to filtered MF
- Saves metrics and ranked scores per cutoff

**Outputs**:
```
experiment_workspace_v4/phase2/cutoff_sweep/
├── logs/
│   ├── run.log
│   └── phase2_summary.json
├── selected_models.json
├── ABL1_PCA_features_5d_rep0/
│   ├── cutoff_100nM/
│   │   ├── metrics.json
│   │   └── ranked_scores.csv
│   ├── cutoff_1000nM/
│   ├── cutoff_10000nM/
│   └── cutoff_100000nM/
└── ABL1_UMAP_features_10d_nn10_md0p0_rep0/
    └── ...
```

### 3. Analyze Results
```bash
python scripts/phase2_post_analysis.py \
    --workspace_dir experiment_workspace_v4 \
    --phase2_run_name cutoff_sweep \
    --output_dir reporting/phase2_post_analysis
```

**What happens**:
- Aggregates metrics across all models and cutoffs
- Generates plots:
  - `cutoff_curves_ef1.png/pdf` (EF@1% vs cutoff for each model)
  - `cutoff_curves_ef5.png/pdf`
  - `cutoff_curves_ef10.png/pdf`
  - `cutoff_heatmap_ef1.png/pdf` (heatmap: models × cutoffs)
  - `cutoff_heatmap_roc_auc.png/pdf`
  - `quality_quantity_ef1.png/pdf` (MF size vs EF@1% scatter)
- Identifies best cutoff per model
- Saves `phase2_aggregated_metrics.csv` and `phase2_best_cutoffs.json`

**Outputs**:
```
reporting/phase2_post_analysis/
├── cutoff_curves_ef1.png + .pdf
├── cutoff_curves_ef5.png + .pdf
├── cutoff_curves_ef10.png + .pdf
├── cutoff_heatmap_ef1.png + .pdf
├── cutoff_heatmap_roc_auc.png + .pdf
├── quality_quantity_ef1.png + .pdf
├── phase2_aggregated_metrics.csv
└── phase2_best_cutoffs.json
```

---

## HPC Usage

### 1. Generate Config (same as local)
```bash
python scripts/generate_molfuse_phase2_configs_v4.py
```

### 2. Submit SLURM Job
```bash
bash hpc/submit_molfuse_phase2.sh \
    configs/molfuse_phase2_grid \
    experiment_workspace_v4 \
    ummbas_screening
```

**Options**:
- Add `--dry-run` to preview without submitting

**Idempotency**:
- Skips if `phase2_summary.json` already exists

### 3. Monitor Job
```bash
# Check SLURM queue
squeue -u $USER

# Tail log (replace <job_id> with actual SLURM job ID)
tail -f slurm_logs/phase2_<job_id>.out
```

### 4. Post-Analysis (after job completes)
```bash
python scripts/phase2_post_analysis.py \
    --workspace_dir experiment_workspace_v4 \
    --phase2_run_name cutoff_sweep \
    --output_dir reporting/phase2_post_analysis
```

---

## Research Questions Addressed

### Primary Question
**Can we improve EF@1% by measuring distance only to more potent ligands from the MF cloud?**

### Analysis Steps
1. **Cutoff curves**: Does EF@1% improve with stricter cutoffs (100 nM vs 100,000 nM)?
2. **Heatmaps**: Do different models (PCA vs UMAP, features vs fingerprints) show different cutoff sensitivities?
3. **Quality-quantity trade-off**: How does MF cloud size (after cutoff filtering) affect enrichment?
4. **Best cutoffs**: What is the optimal cutoff for each method/representation combo?

### Expected Outcomes
- **Hypothesis**: Stricter cutoffs (100 nM) will enrich high-potency actives by creating a more selective scoring function
- **Alternative**: Permissive cutoffs (100 μM) maximize diversity and may capture broader chemical space

---

## Troubleshooting

### Issue: "No completed runs found for X"
**Cause**: Phase 1 incomplete or missing required files  
**Solution**: Complete more Phase 1 runs or reduce `min_required` in Phase 2 CLI

### Issue: "Row count mismatch: MF source vs embedding"
**Cause**: Phase 1 data changed after embedding generation  
**Solution**: Re-run Phase 1 or use original MF CSV

### Issue: "No 'Standard Value (nM)' column in MF source"
**Cause**: MF CSV lacks affinity data  
**Solution**: Use MF CSV with affinity column or skip cutoff filtering

### Issue: "No MF compounds pass cutoff X nM"
**Cause**: Very strict cutoff with low-potency MF cloud  
**Solution**: Phase 2 automatically falls back to full MF cloud (logged as warning)

---

## Next Steps: Phase 3

After Phase 2 analysis identifies optimal cutoffs:
- Phase 3 will **retrain models** with subsampled MF clouds [0, 1K, 10K, 50K, 100K, full]
- Uses Phase 1 best hyperparameters + Phase 2 optimal cutoff
- Research question: Does reducing MF cloud size degrade performance?
