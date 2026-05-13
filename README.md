# MolFuSE — Molecular Function-guided Similarity Explorer

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.TODO.svg)](https://doi.org/10.5281/zenodo.TODO)

**Molecular Function-guided Unsupervised Similarity-based Enrichment for Low-Data Virtual Screening**

Code repository for the MolFuSE method and experiments described in:

> *Molecular Function-guided Unsupervised Similarity-based Enrichment for Low-Data Virtual Screening* — Alexander Hagg, Dirk Reith, Matthias Preller, Karl N. Kirschner

Given a SMILES library and a protein target, MolFuSE ranks candidates by chemical similarity to every protein sharing your target's biological function — no known actives for your specific target required.

MolFuSE leverages a protein's broader **Molecular Function (MF)** category to build a low-dimensional chemical similarity space for virtual screening — without requiring active compounds for the target of interest. Using a rigorous **leave-one-target-out** design across eight diverse targets, we show that MF cloud composition fundamentally shapes which dimensionality reduction method works best, and that this generalises across proteins spanning four orders of magnitude in MF cloud size.

---

## Table of Contents

- [Quick Start (5 minutes)](#quick-start-5-minutes)
- [Part A — MolFuSE GUI: Scoring Candidates Against Trained Models](#part-a--molfuse-gui-scoring-candidates-against-trained-models)
  - [A.1 Installation](#a1-installation)
  - [A.2 Data](#a2-data)
    - [A.2a Pre-trained model workspace](#a2a--pre-trained-model-workspace-gui-quick-start-x-gb)
    - [A.2b Full feature dataset](#a2b--full-feature-dataset-training--paper-experiments-x-gb)
  - [A.3 Launching the GUI](#a3-launching-the-gui)
  - [A.4 GUI Walkthrough](#a4-gui-walkthrough)
  - [A.5 Training a New Model (GUI)](#a5-training-a-new-model-gui)
  - [A.6 CLI Scoring (HPC / headless)](#a6-cli-scoring-hpc--headless)
- [Part B — Reproducing Paper Experiments](#part-b--reproducing-paper-experiments)
  - [B.1 Experimental Design](#b1-experimental-design)
  - [B.2 Running the Pipeline](#b2-running-the-pipeline)
  - [B.3 HPC Submission (SLURM)](#b3-hpc-submission-slurm)
  - [B.4 Post-analysis Scripts](#b4-post-analysis-scripts)
- [Key Results](#key-results)
- [Repository Structure](#repository-structure)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [License](#license)

---

## Quick Start (5 minutes)

> **Prerequisites:** conda or mamba, ≥ 48 GB RAM (or ≥ 16 GB if you load only PCA models).

```bash
# 1 — Install
git clone https://github.com/TODO/molfuse.git && cd molfuse
conda env create -f environment.yml && conda activate molfuse
pip install -e . --no-deps

# 2a — Try the GUI with the built-in demo (no download needed, < 1 min)
python scripts/generate_demo_workspace.py
python -m molfuse.gui --workspace demo_workspace
# Open http://127.0.0.1:8050 — upload demo_candidates.csv and click Score

# 2b — Or download the pre-trained model workspace from Zenodo (TODO: link)
#      Unpack so that experiment_workspace_v4/ is in the molfuse/ repo root.
#      Then: python -m molfuse.gui --workspace experiment_workspace_v4
```

In the browser: select a KW category → pick a model → upload your `candidates.csv` (or `demo_candidates.csv`) → click **Score Candidates**.

Need the full dataset or want to train on your own target? See [A.2 Data](#a2-data) and [A.5 Training](#a5-training-a-new-model-gui). Running on HPC without a display? See [A.6 CLI Scoring](#a6-cli-scoring-hpc--headless).

---

## Part A — MolFuSE GUI: Scoring Candidates Against Trained Models

The interactive GUI lets you score candidate molecules against any trained MolFuSE model — no Python scripting required. It also includes a training tab for building new models on your own data.

### A.1 Installation

**System requirements**

| | Minimum | Recommended |
|-|---------|-------------|
| OS | Linux, macOS (x86-64 or arm64), Windows (WSL2) | Linux |
| conda / mamba | 23.x | mamba (faster solver) |
| RAM | 16 GB *(PCA models only)* | 48 GB (all models) |
| Disk | ~3 GB (conda env) + ~15 GB (data archive) | SSD |
| Python | 3.12 | 3.12 |

**1. Create the conda environment**

```bash
conda env create -f environment.yml
conda activate molfuse
```

**2. Install the molfuse package**

```bash
# All deps are already managed by conda — skip pip's resolver
pip install -e . --no-deps
```

> **RAM requirement — minimum 48 GB.**  
> UMAP models trained on large MF clouds (e.g. KW-0808 Transferase, 425 K molecules) store the full kNN graph in the serialised joblib file and require ≥ 48 GB of RAM to load. PCA models for the same targets require < 1 GB and work on any workstation. If you are RAM-constrained, prefer PCA models or run scoring via the CLI on an HPC node (see §A.6).

| Package | Purpose |
|---------|---------|
| `numpy`, `pandas`, `scipy` | Core data handling |
| `scikit-learn` | PCA, scaler, 1-NN, metrics |
| `umap-learn` | UMAP dimensionality reduction |
| `rdkit` | Canonical SMILES, BEDROC |
| `mordred` | 2D molecular descriptor computation |
| `joblib` | Model serialisation |
| `pyarrow` | Parquet support + 10–100× faster CSV loading |
| `matplotlib`, `plotly` | Plots |
| `dash`, `dash-bootstrap-components` | GUI web app |
| `seaborn`, `polars` *(optional)* | Post-analysis scripts only |

---

### A.2 Data

All data are deposited at Zenodo:

> **[TODO: Zenodo link]**

The archive contains two separate downloads:

#### A.2a — Pre-trained model workspace (GUI quick start, ~X GB)

Download `experiment_workspace_v4.tar.gz` and unpack it in the repo root:

```bash
tar -xzf experiment_workspace_v4.tar.gz
# creates: experiment_workspace_v4/phase4/<run_name>/artifacts/...
```

This is all you need to run the GUI and score candidates. The directory name `experiment_workspace_v4` is what you pass to `--workspace`.

#### A.2b — Full feature dataset (training & paper experiments, ~X GB)

Download `molfuse_data.tar.gz` and unpack to a directory of your choice (`<data-dir>`):

```bash
tar -xzf molfuse_data.tar.gz -C /path/to/data
```

Expected layout:

```
<data-dir>/
    KW-0049_Antioxidant_affinity_extracted_features.parquet
    KW-0049_Antioxidant_affinity_extracted_fingerprints_ECFP4.parquet
    KW-0808_Transferase_affinity_extracted_features.parquet
    ...                                     # one file per UniProt KW category
    zinc/
        zinc_acquirable_extracted_features.parquet
        zinc_acquirable_extracted_fingerprints_ECFP4.parquet
    chembl/
        <UniProtID>_actives_extracted_features.parquet
        ...
```

CSV equivalents are also provided. Column schema:

| Column | Description |
|--------|-------------|
| `SMILES` | Canonical SMILES string |
| `accession` | UniProt accession of the source protein |
| `activity_type` | Assay type (e.g. `IC50`, `Ki`) |
| `activity_value_nM` | Activity value in nanomolar |

---

### A.3 Launching the GUI

```bash
python -m molfuse.gui --workspace experiment_workspace_v4
# → http://127.0.0.1:8050
```

The `--workspace` argument is optional; you can also type or paste the path directly in the browser once the GUI is open. Paths are resolved relative to the directory you launch from.

---

### A.4 GUI Walkthrough

**Step 1 — Select Workspace**  
Enter the path to your experiment workspace and click *Scan*. The GUI will find all valid trained models and report how many are available.

**Step 2 — Select Model**  
Pick a UniProt Molecular Function (KW) category, then choose a specific model run. The best model by EF@1% is pre-selected automatically.

**Step 3 — Input Candidates**  
Upload a CSV or Parquet file with a `SMILES` column, or paste SMILES directly (one per line). Files may optionally include pre-computed feature columns — if present, descriptor calculation is skipped. Optimised for 1–1 000 candidates; up to 10 000 are supported.

**Step 4 — Score**  
Click *Score Candidates*. The GUI computes Mordred 2D descriptors on the fly, projects candidates through the trained model, and ranks them by proximity to the MF cloud. Results include:
- Interactive 2D embedding scatter coloured by score
- Score distribution histogram
- Top 20 candidates table with RDKit structure images and nearest MF neighbour details
- CSV export: `SMILES, score, rank, distance, z0…zN, nearest_mf_smiles, nearest_mf_activity_type, nearest_mf_activity_value_nM, nearest_mf_accession`

---

### A.5 Training a New Model (GUI)

Expand the *Train New Model* accordion at the bottom of the GUI. Fill in:

- **Run name** and **target accession** (e.g. `P12345`)
- Paths to your **MF features file**, **ZINC features file**, and optionally an **actives file** (all relative to your launch directory; CSV or Parquet accepted)
- Hyperparameters: method (UMAP/PCA), dimensions, affinity cutoff

Click *Train Model*. Training runs the phase 1 CLI pipeline in the background with live log streaming. The new model is loaded into the model browser automatically when training completes.

**Training via CLI** (equivalent, for large datasets or HPC):

```bash
python -m molfuse.cli.phase1 --config configs/molfuse_phase1_example.json \
    --workspace experiment_workspace_v4
```

See `configs/molfuse_phase1_example.json` for the config format.

---

### A.6 CLI Scoring (HPC / headless)

For large candidate libraries or on HPC nodes without a display, use the standalone scoring script instead of the GUI.

**Discover available run names:**

```bash
python scripts/score_candidates.py --workspace experiment_workspace_v4 --list
```

This prints a table of all valid models and their run names — no model is loaded. Use any listed `run_name` with `--run-name`.

**Score candidates:**

```bash
python scripts/score_candidates.py \
    --workspace experiment_workspace_v4 \
    --run-name <run_name> \
    --candidates data/my_candidates.parquet \
    --output scored_candidates.csv
```

| Argument | Description |
|----------|-----------|
| `--workspace` | Path to the experiment workspace directory |
| `--run-name` | Run name from `--list` (subdirectory under `phase1/` or `phase4/`) |
| `--candidates` | CSV or Parquet file with a `SMILES` column (or pre-computed feature columns) |
| `--output` | Output CSV path (default: `scored_candidates.csv`) |
| `--phase` | `phase1` or `phase4` (default: auto-detect) |
| `--list` | List all valid run names and exit |

The script produces the same ranked CSV as the GUI export and has no RAM overhead beyond loading the model itself. On SLURM:

```bash
sbatch --mem=48G --cpus-per-task=8 --wrap \
    "conda run -n molfuse python scripts/score_candidates.py \
        --workspace experiment_workspace_v4 \
        --run-name <run_name> \
        --candidates candidates.parquet \
        --output scored.csv"
```

---

## Part B — Reproducing Paper Experiments

This section describes the full five-phase pipeline used to produce the results in the paper. Phases 1–5 are run via CLI; the GUI is not involved.

### B.1 Experimental Design

#### Five-phase pipeline

| Phase | Research question | Retrains? | Key output |
|-------|------------------|-----------|------------|
| **1** — Hyperparameter sweep | Which DR method/hyperparams work best? | Full | Scaler + DR model + embeddings + `metrics.json` |
| **2** — Affinity cutoff sensitivity | Does filtering MF by potency improve EF@1%? | Re-scoring only | Per-cutoff `metrics.json` + `ranked_scores.csv` |
| **3** — MF cloud ablation | What is the critical MF cloud size? | Full | Phase transition curve (UMAP→PCA crossover) |
| **4** — Cross-target generalization | Does performance vary across targets? | Full | Per-target EF@1%, MF size vs performance scatter |
| **5** — Validation baselines | Is UMAP/descriptor selection necessary? | varies | Tanimoto, raw-descriptor, and negative-control comparisons |

#### Eight Phase 4 targets

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

#### Core invariants

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

#### Scoring and metrics

$$\text{score}(m) = -\min_{c \in \text{MF cloud}} d(m, c)$$

Primary metric: **EF@1%** (enrichment factor at top 1%). Also reported: ROC-AUC, PR-AUC, BEDROC (α = 20), Spearman ρ.

---

### B.2 Running the Pipeline

All experiment configs are generated programmatically from the scripts below and are **not** committed to the repository.

**Step 1 — Generate configs**

```bash
# Phase 1 (hyperparameter grid: ~400 configs)
python scripts/generate_molfuse_phase1_configs_v4.py --data-dir <data-dir>

# Phase 2 (cutoff sweep; references Phase 1 workspace)
python scripts/generate_molfuse_phase2_configs_v4.py --workspace-dir experiment_workspace_v4

# Phase 3 (MF ablation; reads Phase 1 + Phase 2 results)
python scripts/generate_molfuse_phase3_configs_v4.py \
    --phase1_grouped reporting/phase1_post_analysis/phase1_summary_grouped.csv \
    --phase2_best_cutoffs reporting/phase2_post_analysis/phase2_best_cutoffs.json \
    --data-dir <data-dir>

# Phase 4 (cross-target; 8 targets × 5 replicates = 40 configs)
python scripts/generate_molfuse_phase4_configs_v4.py --data-dir <data-dir>

# Phase 5 (validation baselines)
python scripts/generate_molfuse_phase5_configs_v4.py \
    --data-dir <data-dir> --workspace-dir experiment_workspace_v4
```

All generators write to `configs/molfuse_phase*_grid/` (gitignored; regenerate locally).

**Step 2 — Run a single experiment (local)**

```bash
python -m molfuse.cli.phase1 --config configs/molfuse_phase1_example.json --workspace experiment_workspace_v4
python -m molfuse.cli.phase2 --config configs/molfuse_phase2_grid/phase2_cutoff_sweep.json --workspace experiment_workspace_v4
python -m molfuse.cli.phase3 --config configs/molfuse_phase3_grid/<run>.json --workspace experiment_workspace_v4
python -m molfuse.cli.phase4 --config configs/molfuse_phase4_grid/<run>.json --workspace experiment_workspace_v4
python -m molfuse.cli.phase5 --config configs/molfuse_phase5_grid/<run>.json --workspace experiment_workspace_v4
```

**Phase 5 sub-experiments**

| `experiment_type` | Description |
|-------------------|-------------|
| `tanimoto` | Raw ECFP4 1-NN (no DR) — tests whether UMAP adds value over Tanimoto similarity |
| `raw_descriptors` | Scaled descriptors, no UMAP — tests whether dimensionality reduction is necessary |
| `negative_control` | Score non-kinase ligands against kinase model — tests database specificity |

**Expected outputs per run**

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

### B.3 HPC Submission (SLURM)

The SLURM job scripts use `$CONDA_ACTIVATE` and `$CONDA_ENV` environment variables:

```bash
export CONDA_ACTIVATE=/path/to/miniforge3/bin/activate
export CONDA_ENV=molfuse
```

Batch submission (idempotent — skips runs that already have a summary JSON):

```bash
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

### B.4 Post-analysis Scripts

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

All scripts accept `--workspace_dir` (default: `experiment_workspace_v4`) and `--output_dir`:

```bash
python scripts/phase1_post_analysis.py --workspace_dir experiment_workspace_v4 --output_dir reporting/phase1_post_analysis
python scripts/phase2_post_analysis.py --workspace_dir experiment_workspace_v4 --output_dir reporting/phase2_post_analysis
python scripts/phase3_post_analysis.py --workspace_dir experiment_workspace_v4 --output_dir reporting/phase3_post_analysis
python scripts/phase4_post_analysis.py --workspace_dir experiment_workspace_v4 --output_dir reporting/phase4_post_analysis
python scripts/phase5_post_analysis.py --workspace_dir experiment_workspace_v4 --output_dir reporting/phase5_post_analysis
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
│   ├── gui/                            # Dash web application (Part A)
│   ├── cli/                            # Phase 1–5 CLI entry points (Part B)
│   ├── data/prep.py                    # Feature selection, scaling, imputation
│   ├── dr/{pca,umap_}.py              # PCA and UMAP wrappers
│   ├── scoring/nn.py                   # Exact 1-NN scoring
│   ├── metrics/metrics.py             # EF@1%, ROC-AUC, PR-AUC, BEDROC, Spearman ρ
│   ├── io/paths.py                     # Workspace directory creation
│   └── analysis/select_best_phase1.py  # Best Phase 1 model selector
│
├── scripts/
│   ├── score_candidates.py                          # CLI/HPC scorer (headless GUI equivalent)
│   ├── generate_molfuse_phase{1-5}_configs_v4.py   # Config generators (--data-dir)
│   ├── generate_molfuse_phase5_expansion.py
│   ├── phase{1-5}_post_analysis.py                 # Per-phase result aggregation
│   ├── retrospective_bedroc_analysis.py
│   ├── analyze_active_rank_distribution.py
│   ├── compare_phase3_phase4.py
│   ├── verify_mf_cloud_sizes.py
│   ├── analyze_phase4_mf_cloud_sizes.py
│   ├── benchmark_computational_performance.py
│   └── extract_phase4_features.py
│
├── hpc/                                # SLURM job and submission scripts
│   ├── score_candidates.sh             # CLI scorer job (48 GB, 8 CPUs)
│   ├── molfuse_phase{1-5}_cpu.sh       # Job scripts (use $CONDA_ACTIVATE / $CONDA_ENV)
│   ├── submit_molfuse_phase{1-5}.sh    # Idempotent batch submitters
│   └── phase{1-4}_analysis_suite.sh    # Post-analysis SLURM wrappers
│
└── configs/
    └── molfuse_phase1_example.json     # Minimal worked example for phase 1
```

---

## Troubleshooting

**`mordred` prints `SafetyError` warnings during `pip install` or import**  
These are non-fatal cache-corruption warnings from mordred's descriptor registry. They do not affect results. You can suppress them with `python -W ignore scripts/score_candidates.py ...`.

**`No models found` after pointing the GUI or CLI at the workspace**  
Verify the directory structure: the workspace must contain `phase1/<run_name>/artifacts/scaler.joblib` or `phase4/<run_name>/artifacts/scaler.joblib`. The run name is the subdirectory name directly under `phase1/` or `phase4/`, not any deeper path. Use `--list` to confirm:
```bash
python scripts/score_candidates.py --workspace experiment_workspace_v4 --list
```

**`MemoryError` or the process is killed when loading a UMAP model**  
The UMAP model for KW-0808 Transferase (425 K molecules) requires ≥ 48 GB RAM. On machines with less RAM, use the PCA variant of the same model (look for `PCA` in the run name). The PCA models require < 1 GB.

**GUI shows a blank page or `Address already in use`**  
Another process is using port 8050. Launch on a different port:
```bash
python -m molfuse.gui --workspace experiment_workspace_v4 --port 8051
```

**`pip install -e . --no-deps` fails with `mordred` version conflicts**  
This is expected — mordred 1.2.0 pins an old numpy. Use `--no-deps` exactly as shown; all dependencies are managed by conda.

**Descriptor calculation returns NaN for some SMILES**  
Invalid or disconnected SMILES are silently dropped. Sanitize your input with RDKit before passing to MolFuSE:
```python
from rdkit import Chem
valid = [smi for smi in smiles_list if Chem.MolFromSmiles(smi) is not None]
```

**`sklearn`/`numpy` pickle incompatibility when loading saved models**  
Models must be loaded with the same major version of scikit-learn used to save them. The pre-trained workspace was saved with scikit-learn 1.8 / numpy 2.4. Rebuild the conda environment from `environment.yml` to guarantee compatibility.

---

## Citation

If you use MolFuSE in your research, please cite:

> [TODO: authors, title, journal, year, DOI]

## License

[LICENSE](LICENSE)
