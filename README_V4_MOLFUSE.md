# molfuse v4.0 (Scaffold)

This is the v4.0 refactor scaffold to align the HPC pipelines with the independently validated test harness.

Core invariants:
- StandardScaler fits on MF+ZINC only; actives are projected using this scaler (no leakage)
- UMAP runs without a fixed seed (random_state=None) to enable parallelism
- Exact 1-NN scoring backend in embedded space; score = -min_distance
- Affinity cutoff applies to MF cloud for scoring only; actives are not filtered
- Spearman’s rho(pActivity vs score) is reported for actives
- Target-preserving exclusion enforced
 - If applying the affinity cutoff yields an empty MF set, the default policy is fail-fast (error). You can set `on_empty_cutoff: "fallback"` to use the full MF set for scoring instead.
 - Zero MF–ZINC overlap enforced by SMILES; any ZINC that exactly matches an MF compound is removed before training/scoring.
 - Deduplication policy: strictly by SMILES string (canonical_smiles/SMILES). One row per unique molecule.
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

- Phase 2 (logs cutoff config):
  ```console
  python -m molfuse.cli.phase2 --config configs/molfuse_phase1_example.json --workspace ./experiment_workspace_v4
  ```  

Notes:
- If your config omits `actives_features_csv` or points to a non-existent file, the CLI will split the MF file by target accession (e.g., P00519) to derive actives and exclude them from MF training.
- UMAP uses `random_state=None` for parallelism by default.
 - Control behavior when no MF pass the cutoff via `on_empty_cutoff`: `"error"` (default) or `"fallback"`.

## HPC usage (SLURM)

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

Notes on fingerprints:
- Fingerprint CSV paths in the generator are derived by replacing `extracted_features.csv` with `extracted_fingerprints.csv`. Adjust the pattern if your filenames differ.
- UMAP uses `metric="jaccard"` automatically for fingerprint runs.