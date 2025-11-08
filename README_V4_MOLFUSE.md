# molfuse v4.0 (Scaffold)

This is the v4.0 refactor scaffold to align the HPC pipelines with the independently validated test harness.

Core invariants:
- StandardScaler fits on MF+ZINC only; actives are projected using this scaler (no leakage)
- UMAP runs without a fixed seed (random_state=None) to enable parallelism
- Exact 1-NN scoring backend in embedded space; score = -min_distance
- Affinity cutoff applies to MF cloud for scoring only; actives are not filtered
- Spearman's rho(pActivity vs score) is reported for actives
- Target-preserving exclusion enforced
 - If applying the affinity cutoff yields an empty MF set, the default policy is fail-fast (error). You can set `on_empty_cutoff: "fallback"` to use the full MF set for scoring instead.
 - Zero MF–ZINC overlap enforced by SMILES; any ZINC that exactly matches an MF compound is removed before training/scoring.
 - Deduplication policy: Median affinity aggregation per SMILES (robust to ChEMBL multi-assay measurement variance); one row per unique molecule.
 - Representations supported:
   - Features (scaled; PCA/UMAP-Euclidean)
   - Fingerprints (passthrough; PCA baseline, UMAP-Jaccard)

Outputs and logging:
- Each run writes a structured text log to `logs/run.log` with counts after each step (target exclusion, overlap removal, deduplication, NaN drops, cutoff, ZINC sampling) and model/training details.
- Artifacts now include embedded coordinate CSVs for all three sets:
  - `artifacts/embedding_mf.csv`
  - `artifacts/embedding_zinc.csv`
  - `artifacts/embedding_actives.csv`
  Columns: SMILES, optional Compound ChEMBL ID, and `z0..z{dim-1}`.
 - Models are saved:
   - `artifacts/scaler.joblib` (StandardScaler fitted on MF+ZINC)
   - `artifacts/pca_model.joblib` or `artifacts/umap_model.joblib` (DR model)

## Try it

- Phase 1 (end-to-end; derives actives from MF file by accession when `actives_features_csv` is missing or not found):

  ```console
  python -m molfuse.cli.phase1 --config configs/molfuse_phase1_example.json --workspace ./experiment_workspace_v4
  ```  

  Note: The example config is set to fingerprints + UMAP with Jaccard. To run a features baseline instead, set `representation: "features"`, switch `method: "pca"` or keep `method: "umap"` with `metric: "euclidean"`, and point CSVs to `*_extracted_features.csv`.

- Phase 2 (cutoff sensitivity; re-scoring only):

  1. Generate Phase 2 config:
  ```console
  python scripts/generate_molfuse_phase2_configs_v4.py
  ```  
  
  2. Run Phase 2 (auto-selects best Phase 1 models):
  ```console
  python -m molfuse.cli.phase2 --config configs/molfuse_phase2_grid/phase2_cutoff_sweep.json --workspace ./experiment_workspace_v4
  ```
  
  3. Analyze results:
  ```console
  python scripts/phase2_post_analysis.py --workspace_dir ./experiment_workspace_v4 --phase2_run_name cutoff_sweep --output_dir reporting/phase2_post_analysis
  ```

Notes:
- If your config omits `actives_features_csv` or points to a non-existent file, the CLI will split the MF file by target accession (e.g., P00519) to derive actives and exclude them from MF training.
- UMAP uses `random_state=None` for parallelism by default.
 - Control behavior when no MF pass the cutoff via `on_empty_cutoff`: `"error"` (default) or `"fallback"`.

## Phase 2: Affinity Cutoff Sensitivity

**Research Question**: Can we improve EF@1% by measuring distance only to more potent ligands from the MF cloud?

**Design**: Re-scoring only (NO retraining)
- Reuses Phase 1 best models and pre-computed embeddings
- Filters MF cloud by affinity cutoff [100, 1000, 10000, 100000] nM
- Re-scores actives+ZINC via 1-NN to filtered MF
- Computes metrics per cutoff

**Key Invariant**: No model retraining; only scoring changes with MF cloud filtering.

**Workflow**:
1. Complete Phase 1 (or have at least 1 PCA + 1 UMAP run per representation)
2. Generate Phase 2 config: `python scripts/generate_molfuse_phase2_configs_v4.py`
3. Run Phase 2: `python -m molfuse.cli.phase2 --config configs/molfuse_phase2_grid/phase2_cutoff_sweep.json --workspace experiment_workspace_v4`
4. Analyze: `python scripts/phase2_post_analysis.py --workspace_dir experiment_workspace_v4 --phase2_run_name cutoff_sweep --output_dir reporting/phase2_post_analysis`

**Outputs**:
- `workspace/phase2/cutoff_sweep/<phase1_run_name>/cutoff_<X>nM/metrics.json`
- `workspace/phase2/cutoff_sweep/<phase1_run_name>/cutoff_<X>nM/ranked_scores.csv`
- Post-analysis: cutoff curves, heatmaps, quality-quantity plots, best cutoffs JSON

## Phase 3: MF Cloud Ablation

**Research Question**: What happens to similarity space when MF cloud size decreases? Does performance degrade?

**Design**: FULL RETRAINING (unlike Phase 2)
- Subsamples MF cloud to sizes [0, 1K, 10K, 50K, 100K, full]
- Retrains scaler + DR model for each MF size
- Uses Phase 1 best hyperparameters + Phase 2 optimal cutoff
- Projects actives with each retrained model

**Key Difference from Phase 2**: Model retraining required; tests how training set size affects model quality.

**Expected Outcome**: Performance degradation with smaller MF clouds; possible UMAP/PCA crossover point.

## Phase 4: Cross-Target Generalization

**Research Question**: Does natural MF cloud size predict screening performance across different target proteins?

**Design**: Evaluate across 8 diverse targets (NO MF subsampling)
- Uses Phase 1 best method (UMAP/features, dim=10, n_neighbors=5, min_dist=0.0)
- Uses Phase 2 optimal cutoff (100,000 nM)
- Each target has natural MF cloud size (full KW category - target actives)
- 5 replicates per target (seeds: 42, 123, 456, 789, 1011)

**Targets** (spanning ~4 orders of magnitude in MF cloud size):
1. **KW-0049_Antioxidant**: P00441 (SOD1, 39 actives, 43 MF)
2. **KW-0929_Antimicrobial**: P14555 (PLA2G2A, 582 actives, 287 MF)
3. **KW-0505_Motor_protein**: P52732 (KIF11, 1,158 actives, 1,286 MF)
4. **KW-0202_Cytokine**: P43490 (NAMPT, 2,904 actives, 2,907 MF)
5. **KW-0358_Heparin-binding**: P11362 (FGFR1, 4,150 actives, 7,614 MF)
6. **KW-0456_Lyase**: P00918 (CA2, 9,685 actives, 37,685 MF)
7. **KW-0560_Oxidoreductase**: P08684 (CYP3A4, 6,151 actives, 94,617 MF)
8. **KW-0808_Transferase**: P00519 (ABL1, 5,505 actives, 425,289 MF)

**Key Difference from Phase 3**: Multiple targets with natural MF sizes (not artificial ablation).

**Workflow**:
1. Generate configs: `python scripts/generate_molfuse_phase4_configs_v4.py`
2. Submit to HPC: `bash hpc/submit_molfuse_phase4.sh configs/molfuse_phase4_grid experiment_workspace_v4`
3. Analyze: `python scripts/phase4_post_analysis.py --workspace_dir experiment_workspace_v4 --output_dir reporting/phase4_cross_target`

**Expected Outputs**:
- `workspace/phase4/cross_target/<run_name>/logs/phase4_summary.json`
- `workspace/phase4/cross_target/<run_name>/metrics/metrics.json`
- Post-analysis: MF size vs EF@1% scatter, per-target performance bars, correlation analysis

## HPC usage (SLURM)

### Phase 1

1) Generate Phase 1 configs (replicates and full grids):

  python scripts/generate_molfuse_phase1_configs_v4.py

  Outputs to `configs/molfuse_phase1_grid/`.
  - PCA: features and fingerprints, dims [2,5,10], 5 replicates
  - UMAP (features): Euclidean, dims [2,5,10], n_neighbors [5,10,50,100,500], min_dist [0.0,0.01,0.05,0.1], 5 replicates
  - UMAP (fingerprints): Jaccard, dims [2,5,10], n_neighbors [5,10,50,100,500], min_dist [0.0,0.01,0.05,0.1], 5 replicates
  Each config includes a `replicate` tag and a unique `run_name` with `_rep{n}`; runs write to separate folders.

2) Submit jobs:

  bash hpc/submit_molfuse_phase1.sh configs/molfuse_phase1_grid experiment_workspace_v4 ummbas_screening

  Internally runs `sbatch hpc/molfuse_phase1_cpu.sh <config> <workspace> <conda_env>`.

### Phase 2

1) Generate Phase 2 config:

  python scripts/generate_molfuse_phase2_configs_v4.py

  Outputs to `configs/molfuse_phase2_grid/phase2_cutoff_sweep.json`.

2) Submit Phase 2 job (after Phase 1 completes):

  bash hpc/submit_molfuse_phase2.sh configs/molfuse_phase2_grid experiment_workspace_v4 ummbas_screening

  Internally runs `sbatch hpc/molfuse_phase2_cpu.sh <config> <workspace> <conda_env>`.
  Idempotent: skips if `phase2_summary.json` exists.

3) Post-analysis (after Phase 2 completes):

  python scripts/phase2_post_analysis.py --workspace_dir experiment_workspace_v4 --phase2_run_name cutoff_sweep --output_dir reporting/phase2_post_analysis

### Phase 4

1) Generate Phase 4 configs (40 total: 8 targets × 5 replicates):

  python scripts/generate_molfuse_phase4_configs_v4.py

  Outputs to `configs/molfuse_phase4_grid/`.

2) Submit Phase 4 jobs (after Phase 1-2 complete):

  bash hpc/submit_molfuse_phase4.sh configs/molfuse_phase4_grid experiment_workspace_v4 ummbas_screening

  Internally runs `sbatch hpc/molfuse_phase4_cpu.sh <config> <workspace> <conda_env>`.
  Idempotent: skips if `phase4_summary.json` exists.

3) Post-analysis (after Phase 4 completes):

  python scripts/phase4_post_analysis.py --workspace_dir experiment_workspace_v4 --output_dir reporting/phase4_cross_target

Notes on fingerprints:
- Fingerprint CSV paths in the generator are derived by replacing `extracted_features.csv` with `extracted_fingerprints.csv`. Adjust the pattern if your filenames differ.
- UMAP uses `metric="jaccard"` automatically for fingerprint runs.