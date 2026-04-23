# MolFuSE — Molecular Function-guided Similarity Explorer

**Molecular Function-guided Unsupervised Similarity-based Enrichment for Low-Data Virtual Screening**

Code repository for the MolFuSE method and experiments described in:

> *Molecular Function-guided Unsupervised Similarity-based Enrichment for Low-Data Virtual Screening* — Alexander Hagg, Dirk Reith, Matthias Preller, Karl N. Kirschner

MolFuSE leverages a protein's broader **Molecular Function (MF)** category to build a low-dimensional chemical similarity space for virtual screening — without requiring active compounds for the target of interest. Using a rigorous **leave-one-target-out** design across eight diverse targets, we show that MF cloud composition fundamentally shapes which dimensionality reduction method works best, and that this generalises across proteins spanning four orders of magnitude in MF cloud size.

---

## Installation

### 1. Conda environment

```bash
conda env create -f environment.yml
conda activate molfuse
```

### 2. Install the molfuse package

```bash
pip install -e .
```

### Dependencies

| Package | Use |
|---------|-----|
| `numpy`, `pandas`, `scipy` | Core data handling |
| `scikit-learn` | PCA, scaler, 1-NN, metrics |
| `umap-learn` | UMAP dimensionality reduction |
| `rdkit` | Canonical SMILES, BEDROC |
| `mordred` | 2D molecular descriptor computation |
| `joblib` | Model serialisation |
| `matplotlib`, `plotly` | Plots |
| `dash`, `dash-bootstrap-components` | GUI web app |
| `pyarrow` *(optional)* | 10–100× faster CSV loading + Parquet caching |
| `seaborn`, `polars` *(optional)* | Post-analysis scripts |

---

## Data

Datasets are not included in this repository (too large for git). The expected data layout is:

```
<data-dir>/                                         # pass to --data-dir
    KW-0049_Antioxidant_affinity_extracted_features.csv
    KW-0049_Antioxidant_affinity_extracted_fingerprints_ECFP4.csv
    KW-0808_Transferase_affinity_extracted_features.csv
    ...                                             # one CSV per KW category
    zinc/
        zinc_acquirable_extracted_features.csv
        zinc_acquirable_extracted_fingerprints_ECFP4.csv
    chembl/
        ABL1_P00519_actives_extracted_features.csv  # Phase 1 actives only
        ...
```

> **Data availability:** Datasets are deposited at [TODO: Zenodo/institutional repository link].  
> Features are full 2D Mordred descriptors (1 613 raw features); fingerprints are ECFP4 2048-bit.

---

## Experimental Design

### Five-phase pipeline

| Phase | Research question | Retrains? | Key output |
|-------|------------------|-----------|------------|
| **1** — Hyperparameter sweep | Which DR method/hyperparams work best? | ✅ Full | Scaler + DR model + embeddings + `metrics.json` |
| **2** — Affinity cutoff sensitivity | Does filtering MF by potency improve EF@1%? | ❌ Re-scoring only | Per-cutoff `metrics.json` + `ranked_scores.csv` |
| **3** — MF cloud ablation | What is the critical MF cloud size? | ✅ Full | Phase transition curve (UMAP→PCA crossover) |
| **4** — Cross-target generalization | Does performance vary across targets? | ✅ Full | Per-target EF@1%, MF size vs performance scatter |
| **5** — Validation baselines | Is UMAP/descriptor selection necessary? | varies | Tanimoto, raw-descriptor, and negative-control comparisons |

### Eight Phase 4 targets

| UniProt KW | Target | Actives | MF cloud |
|------------|--------|---------|----------|
| KW-0049 Antioxidant | P00441 SOD1 | 39 | 43 |
| KW-0929 Antimicrobial | P14555 PLA2G2A | 582 | 287 |
| KW-0505 Motor protein | P52732 KIF11 | 1 158 | 1 286 |
| KW-0202 Cytokine | P43490 NAMPT | 2 904 | 2 907 |
| KW-0358 Heparin-binding | P11362 FGFR1 | 4 150 | 7 614 |
| KW-0456 Lyase | P00918 CA2 | 9 685 | 37 685 |
| KW-0560 Oxidoreductase | P08684 CYP3A4 | 6 151 | 94 617 |
| KW-0808 Transferase | P00519 ABL1 | 5 505 | 425 289 |

### Core invariants

1. `StandardScaler` is fitted on MF + ZINC only; actives are projected with the frozen scaler (no leakage).
2. `UMAP random_state=None` to enable parallel execution (non-deterministic but faster on HPC).
3. Exact 1-NN scoring: `score = −min_{c ∈ MF} d(m, c)` (higher = closer to MF cloud).
4. Affinity cutoff filters the MF cloud for **scoring only**; actives are never filtered.
5. Spearman ρ(pActivity, score) is reported for actives.
6. Actives of the held-out target are excluded from the MF cloud before training.
7. Empty MF after cutoff → fail-fast by default (`on_empty_cutoff: "error"`); set `"fallback"` to use full MF.
8. Zero MF–ZINC overlap enforced via canonical SMILES.
9. Deduplication: median affinity per unique SMILES (robust to multi-assay variance).
10. Large CSVs are cached as Parquet on first load (10–100× speedup on subsequent runs).

### Scoring and metrics

$$\text{score}(m) = -\min_{c \in \text{MF cloud}} d(m, c)$$

Primary metric: **EF@1%** (enrichment factor at top 1%). Also reported: ROC-AUC, PR-AUC, BEDROC (α = 20), Spearman ρ.

---

## Running the Pipeline

### Step 1 — Generate configs

```bash
# Phase 1 (hyperparameter grid: ~400 configs)
python scripts/generate_molfuse_phase1_configs_v4.py --data-dir /path/to/datasets_2d_all

# Phase 2 (cutoff sweep config; references Phase 1 workspace)
python scripts/generate_molfuse_phase2_configs_v4.py

# Phase 3 (MF ablation; reads Phase 1 + Phase 2 results)
python scripts/generate_molfuse_phase3_configs_v4.py \
    --phase1_grouped reporting/phase1_post_analysis/phase1_summary_grouped.csv \
    --phase2_best_cutoffs reporting/phase2_post_analysis/phase2_best_cutoffs.json \
    --data_dir /path/to/datasets_2d_all

# Phase 4 (cross-target; 8 targets × 5 replicates = 40 configs)
python scripts/generate_molfuse_phase4_configs_v4.py --data-dir /path/to/datasets_2d_all

# Phase 5 (validation baselines)
python scripts/generate_molfuse_phase5_configs_v4.py --data-dir /path/to/datasets_2d_all
```

All generators write to `configs/molfuse_phase*_grid/` (gitignored; regenerate locally).

### Step 2 — Run a single experiment (local)

```bash
python -m molfuse.cli.phase1 --config configs/molfuse_phase1_example.json --workspace ./experiment_workspace_v4
python -m molfuse.cli.phase2 --config configs/molfuse_phase2_grid/phase2_cutoff_sweep.json --workspace ./experiment_workspace_v4
python -m molfuse.cli.phase3 --config configs/molfuse_phase3_grid/<run>.json --workspace ./experiment_workspace_v4
python -m molfuse.cli.phase4 --config configs/molfuse_phase4_grid/<run>.json --workspace ./experiment_workspace_v4
python -m molfuse.cli.phase5 --config configs/molfuse_phase5_grid/<run>.json --workspace ./experiment_workspace_v4
```

### Phase 5 sub-experiments

| `experiment_type` | Description |
|-------------------|-------------|
| `tanimoto` | Raw ECFP4 1-NN (no DR) — tests whether UMAP adds value over Tanimoto similarity |
| `raw_descriptors` | Scaled descriptors, no UMAP — tests whether dimensionality reduction is necessary |
| `negative_control` | Score non-kinase ligands against kinase model — tests database specificity |

### Expected outputs per run

```
experiment_workspace_v4/
└── phase1/<run_name>/
    ├── logs/run.log
    ├── artifacts/
    │   ├── scaler.joblib
    │   ├── pca_model.joblib  (or umap_model.joblib)
    │   ├── embedding_mf.csv
    │   ├── embedding_zinc.csv
    │   └── embedding_actives.csv
    └── metrics/
        ├── metrics.json
        └── ranked_scores.csv
```

---

## HPC Submission (SLURM)

### Environment setup

The SLURM job scripts use `$CONDA_ACTIVATE` and `$CONDA_ENV` environment variables with sensible defaults:

```bash
export CONDA_ACTIVATE=/path/to/miniforge3/bin/activate
export CONDA_ENV=molfuse
```

### Batch submission (idempotent)

```bash
# Submits all configs; skips runs that already have a summary JSON
bash hpc/submit_molfuse_phase1.sh configs/molfuse_phase1_grid experiment_workspace_v4
bash hpc/submit_molfuse_phase2.sh configs/molfuse_phase2_grid experiment_workspace_v4
bash hpc/submit_molfuse_phase3.sh configs/molfuse_phase3_grid experiment_workspace_v4
bash hpc/submit_molfuse_phase4.sh configs/molfuse_phase4_grid experiment_workspace_v4
bash hpc/submit_molfuse_phase5.sh configs/molfuse_phase5_grid experiment_workspace_v4

# Dry run (print jobs without submitting)
bash hpc/submit_molfuse_phase1.sh configs/molfuse_phase1_grid experiment_workspace_v4 --dry-run
```

SLURM resources per phase: 64 CPUs, 64–350 GB RAM, 2–72 h walltime.

---

## Post-analysis Scripts

| Script | Generates |
|--------|-----------|
| `scripts/phase1_post_analysis.py` | Method comparison plots, hyperparameter heatmaps, `phase1_summary_grouped.csv` |
| `scripts/phase1_stratified_scores.py` | Potency-tier EF@1% (High ≤100 nM / Medium / Weak) |
| `scripts/phase2_post_analysis.py` | Cutoff-EF curves, heatmaps, `phase2_best_cutoffs.json`, BEDROC/IEF |
| `scripts/phase3_post_analysis.py` | MF size degradation curves, phase transition analysis |
| `scripts/phase4_post_analysis.py` | Per-target EF bars, MF size vs EF scatter, BEDROC/IEF |
| `scripts/phase5_post_analysis.py` | LaTeX tables (mean ± SD), statistical comparison vs Phase 4 |
| `scripts/retrospective_bedroc_analysis.py` | Recompute BEDROC/IEF from existing `ranked_scores.csv` without re-running |
| `scripts/analyze_active_rank_distribution.py` | Cumulative rank distribution, bimodality diagnostics |
| `scripts/compare_phase3_phase4.py` | Overlay: Phase 3 ablation curve vs Phase 4 natural MF sizes |
| `scripts/verify_mf_cloud_sizes.py` | MF cloud size through each processing stage (data integrity check) |
| `scripts/analyze_phase4_mf_cloud_sizes.py` | MF cloud sizes for all Phase 4 targets (publication table) |
| `scripts/benchmark_computational_performance.py` | Wall-clock timing: Tanimoto vs features vs UMAP across molecule counts |
| `scripts/extract_phase4_features.py` | Compute Mordred 2D descriptors for Phase 4 KW CSVs |

---

## GUI — MolFuSE Interactive Scorer

A Dash web app for scoring candidate molecules against pre-trained models without re-running the pipeline.

```bash
python -m molfuse.gui.app --workspace /path/to/experiment_workspace_v4
# → http://127.0.0.1:8050
```

**Features:**
- Model browser (by KW category and function)
- CSV or pasted SMILES input
- On-the-fly Mordred 2D descriptor computation
- Real-time 1-NN scoring and ranking
- Interactive 2D embedding scatter, score histogram, ROC/PR curves
- RDKit structure images for nearest MF neighbours
- CSV export with columns: `SMILES, score, rank, distance, z0…zN, nearest_mf_smiles, nearest_mf_activity_type, nearest_mf_activity_value_nM, nearest_mf_accession`

**Capacity:** optimised for 1–1 000 candidates real-time; use the CLI for >10 000.

**Demo:**

```bash
python scripts/generate_gui_demo.py   # creates demo/workspace/ + demo/data/example_candidates.csv
python -m molfuse.gui.app --workspace demo/workspace
```

---

## Key Results

### Phase transition: MF cloud reverses PCA vs UMAP dominance

| MF cloud | UMAP EF@1% | PCA EF@1% | Winner |
|----------|-----------|----------|--------|
| MF = 0 (no cloud) | ~48 | ~2.5 | UMAP (19×) |
| MF = 191 K (full) | ~39 | ~57.6 | PCA (1.5×) |

**Interpretation:** small MF clouds produce isolated active clusters where UMAP's local structure excels; large MF clouds create smooth chemical gradients where PCA's global projection dominates. The crossover occurs at ~10 K–50 K molecules (validated in Phase 3).

---

## Repository Structure

```
.
├── environment.yml                     # Conda environment specification
├── pyproject.toml                      # pip-installable molfuse package
├── LICENSE
│
├── molfuse/                            # Python package (pip install -e .)
│   ├── cli/                            # Phase 1–5 CLI entry points
│   ├── data/prep.py                    # Feature selection, scaling, imputation
│   ├── dr/{pca,umap_}.py              # PCA and UMAP wrappers
│   ├── scoring/nn.py                   # Exact 1-NN scoring
│   ├── metrics/metrics.py             # EF@1%, ROC-AUC, PR-AUC, BEDROC, Spearman ρ
│   ├── io/paths.py                     # Workspace directory creation
│   ├── analysis/select_best_phase1.py  # Best Phase 1 model selector
│   └── gui/                            # Dash web application
│
├── scripts/
│   ├── generate_molfuse_phase{1-5}_configs_v4.py   # Config generators (--data-dir)
│   ├── generate_molfuse_phase5_expansion.py
│   ├── phase{1-5}_post_analysis.py                 # Per-phase result aggregation
│   ├── retrospective_bedroc_analysis.py
│   ├── analyze_active_rank_distribution.py
│   ├── compare_phase3_phase4.py
│   ├── verify_mf_cloud_sizes.py
│   ├── analyze_phase4_mf_cloud_sizes.py
│   ├── benchmark_computational_performance.py
│   ├── extract_phase4_features.py
│   └── generate_gui_demo.py
│
├── hpc/                                # SLURM job and submission scripts
│   ├── molfuse_phase{1-5}_cpu.sh       # Job scripts (use $CONDA_ACTIVATE / $CONDA_ENV)
│   ├── submit_molfuse_phase{1-5}.sh    # Idempotent batch submitters
│   └── phase{1-4}_analysis_suite.sh    # Post-analysis SLURM wrappers
│
└── configs/
    ├── molfuse_phase1_example.json     # Minimal worked example
    ├── molfuse_phase5_grid/            # Phase 5 validation configs (committed)
    └── molfuse_phase5_expansion/       # Phase 5 expansion configs (committed)
```

---

## Citation

If you use MolFuSE in your research, please cite:

> [TODO: authors, title, journal, year, DOI]

## License

[LICENSE](LICENSE)
