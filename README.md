# UMMBAS — Molecular Function-Guided Virtual Screening

**Ultra-large Molecular function-Mediated Biologically-informed Affinity Screening**

A computational drug discovery framework that leverages a protein's broader **Molecular Function (MF)** category to build a low-dimensional similarity space for virtual screening. Using a rigorous **leave-one-target-out** design, we show that MF cloud composition fundamentally shapes which dimensionality reduction method works best — and that this generalises across proteins spanning four orders of magnitude in MF cloud size.

---

## Installation

### 1. Conda environment

```bash
conda env create -f environment.yml
conda activate ummbas-screening
```

### 2. Install the molfuse package

```bash
pip install -e .
# or with optional post-analysis dependencies:
pip install -e ".[analysis]"
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

> **Data availability:** Datasets will be deposited at [TODO: Zenodo/institutional repository link].  
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
export CONDA_ENV=ummbas-screening
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

> [TODO: add citation once published]

## License

[LICENSE](LICENSE)


**Ultra-large Molecular function-Mediated Biologically-informed Affinity Screening**

> **✅ Repository Status**: Version 3.0 stable - optimized hyperparameter sweep, MF cloud ablation, cross-protein generalization, and affinity cutoff analysis. Branch: `3.0`

## Quick Start

- **📘 Complete Pipeline Guide**: [`README_V3_PIPELINE.md`](README_V3_PIPELINE.md)
- **📝 Methods for Publication**: [`METHODS_FOR_PAPER.md`](METHODS_FOR_PAPER.md)
- **🚀 Quick Start Guide**: [`docs/QUICKSTART_V3.md`](docs/QUICKSTART_V3.md)
- **🔬 Key Finding - MF Cloud Impact**: [`docs/MF_CLOUD_IMPACT_ANALYSIS.md`](docs/MF_CLOUD_IMPACT_ANALYSIS.md)
- **💻 HPC Execution**: [`hpc/README.md`](hpc/README.md)
- **📊 Lab Book**: [`LAB_BOOK.md`](LAB_BOOK.md) - Experimental log and findings
- **📋 Planning**: [`PLANNING.md`](PLANNING.md) - Task tracking and next steps
- **🗄️ Archive**: [`ARCHIVE.md`](ARCHIVE.md) - Deprecated features and scripts
 - **🔧 v4.0 Scaffold (molfuse)**: [`README_V4_MOLFUSE.md`](README_V4_MOLFUSE.md) - Invariants and new CLIs (Phase 1/2) under active development
 - **📰 Publication Synthesis**: [`PUBLICATION.md`](PUBLICATION.md) - High-level summary, hypotheses, and v4 invariants
 - **🧪 Standalone test**: [`tests/mordred_full_feature_eval/`](tests/mordred_full_feature_eval/) — Compare current 40 features vs full Mordred 2D and 2D+3D with UMAP and EF@1%; now includes coverage-aware selection and optional threshold sweep.

## Abstract

A significant challenge in ligand-based drug discovery is the scarcity of known active compounds for novel or understudied protein targets. To address this, we developed and validated a computational framework that leverages the target's broader **Molecular Function (MF)** to enrich the pool of relevant chemical matter for virtual screening. This study employs a rigorous **"leave-one-target-out"** experimental design to assess the hypothesis that a similarity space, built from a general MF chemical landscape, can effectively prioritize true active ligands for a specific, held-out protein.

**v3.0 Key Innovation**: We discovered that **MF cloud inclusion fundamentally reverses PCA vs UMAP performance**. With minimal MF data (MF=0), UMAP dominates (EF@1% ~48 vs PCA ~2.5). With full MF cloud (MF=191K), PCA dominates (EF@1% ~58 vs UMAP ~39). This phase transition validates our biological hypothesis and provides practical guidance for method selection.

### v3.0 Experimental Design (440 experiments)

**Phase 1: Hyperparameter Sweep (260 runs)**  
- Target: ABL1 kinase  
- Methods: PCA (2D/5D/10D) vs UMAP (2D/5D/10D, nn: 10/20/100/500, md: 0.01/0.1/0.5)  
- Representations: Physicochemical features (39 RDKit descriptors) + ECFP4 fingerprints  
- Seeds: 5 replicates (42-46)

**Phase 2: Affinity Cutoff Sensitivity (40 runs)** *[REORDERED - was Phase 4]*  
- Cutoffs: 100 nM, 1 μM, 10 μM, 100 μM (aligned with potency tiers)  
- Purpose: Determine optimal affinity threshold for MF cloud composition  
- **Computational efficiency**: Reuses Phase 1 similarity spaces (~75% time savings)  
- **Rationale for reordering**: Cutoff must be established before MF cloud ablation

**Phase 3: MF Cloud Ablation (60 runs)** *[REORDERED - was Phase 2]*  
- MF sizes: 0, 1K, 10K, 50K, 100K, 420K molecules  
- Uses optimal cutoff from Phase 2  
- Purpose: Validate phase transition hypothesis (UMAP→PCA crossover at ~10K-50K)

**Phase 4: Cross-Protein Generalization (80 runs)** *[REORDERED - was Phase 3]*  
- Targets: Pyruvate kinase M2 (same MF), Isocitrate dehydrogenase (different MF)  
- Uses optimal cutoff from Phase 2  
- Purpose: Test transferability across proteins

**Evaluation**: EF@1% (primary), ROC-AUC, PR-AUC across 5 replicates 

## Methodology Summary

### 1. Experimental Hypothesis

**Core hypothesis**: Ligands active against a specific protein will exhibit measurable chemical proximity to a "cloud" of compounds that modulate *other* proteins sharing the same molecular function.

**Leave-one-target-out design**:
1. Hold out all active ligands for target protein X
2. Construct MF cloud from all OTHER proteins in the same MF category
3. Build low-dimensional similarity space using MF cloud + ZINC decoys
4. Project held-out actives into this space
5. Rank by distance to nearest MF cloud molecule
6. Evaluate with EF@1%, ROC-AUC, PR-AUC

### 2. Target Proteins (ChEMBL 35)

| Target | UniProt | MF | Actives | MF Cloud Size | ZINC Decoys |
|--------|---------|-----|---------|---------------|-------------|
| Tyrosine kinase ABL1 | P00519 | Transferase | 3,331 | 191,790 | 1,293,557 |
| Pyruvate kinase M2 | P14618 | Transferase | 1,019 | 193,100 | 1,293,557 |
| Isocitrate dehydrogenase | O75874 | Oxidoreductase | 374 | 56,517 | 1,293,557 |

**Data integrity**: Zero overlap between MF cloud and ZINC decoys (validated with RDKit canonical SMILES)

### 3. Molecular Representations

- **Physicochemical features**: 39 RDKit descriptors (MolWt, LogP, TPSA, topological indices, etc.) + StandardScaler
- **Structural fingerprints**: ECFP4 (2048-bit, radius=2) with Jaccard distance

### 4. Dimensionality Reduction

- **PCA**: Linear projection (2D/5D/10D) with Euclidean distance
- **UMAP**: Non-linear manifold (2D/5D/10D, nn: 10/20/100/500, md: 0.01/0.1/0.5) with Euclidean or Jaccard distance

**Projection-only strategy**: DR models fit on MF cloud + ZINC only. Target actives projected afterward to prevent data leakage.

### 5. Scoring and Evaluation

**Scoring function**: For each molecule $m$, score = $-\min_{c \in \text{MF cloud}} d(m, c)$  
(Closer to MF cloud = higher score = better ranking)

**Metrics**:
- **EF@1%** (primary): Enrichment factor in top 1% of ranked list
- **ROC-AUC**: Overall discrimination ability
- **PR-AUC**: Performance on imbalanced datasets

**Statistical robustness**: All experiments run in quintuplicate (seeds: 42, 43, 44, 45, 46)

---

## Key Findings

### 🔬 MF Cloud Reverses PCA vs UMAP Performance

| Configuration | UMAP EF@1% | PCA EF@1% | Winner | Fold Change |
|---------------|------------|-----------|--------|-------------|
| **Without MF cloud** (MF=0) | ~48 | ~2.5 | UMAP | 19.2× |
| **With MF cloud** (MF=191K) | ~39 | ~57.6 | PCA | 1.5× |

**Interpretation**:
- **Low MF counts**: Actives form isolated clusters → Non-linear UMAP excels at local structure
- **High MF counts**: MF cloud creates smooth gradient → Linear PCA captures global patterns
- **Crossover predicted**: ~10K-50K molecules (Phase 3 will validate)

**Impact**: MF cloud composition is a critical determinant of optimal DR method selection.

### 📊 Optimal Configurations (Phase 1, ABL1)

**Best overall**: PCA-10D-features (EF@1% = 57.6 ± 2.1)  
**Best UMAP**: UMAP-2D-features (nn=10, md=0.01, EF@1% = 39.1 ± 1.8)  
**Fingerprints**: Consistently underperform features (PCA-ECFP4: 12.3 ± 0.9)

**Dimensionality trends**:
- PCA benefits from higher dimensions (10D > 5D > 2D)
- UMAP performs best in 2D (preserves local neighborhoods)

---

## Recent Change (Oct 26, 2025): Freshness-Based Recompute Safeguard

- The orchestrator now recomputes similarity spaces when upstream inputs (deduplicated MF/ZINC CSVs) are newer than the existing simspace CSV. This prevents stale artifact reuse across reruns and ensures deduplication propagates to PCA and UMAP consistently.
- No API changes; behavior is automatic. Logs indicate whether a simspace was reused (up-to-date) or recomputed due to input freshness.

---

## Repository Structure

```
.
├── README.md                          # This file (v3.0 overview)
├── README_V3_PIPELINE.md              # Complete pipeline documentation
├── METHODS_FOR_PAPER.md               # Comprehensive methods for publication
├── LAB_BOOK.md                        # Experimental log and daily findings
├── PLANNING.md                        # Task tracking and next steps
├── ARCHIVE.md                         # Deprecated features and scripts
│
├── hpc/                               # HPC execution scripts
│   ├── README.md                      # HPC setup and submission guide
│   ├── ummbas_v3_cpu.sh               # SLURM job script
│   └── submit_v3_phase*.sh            # Batch submission scripts (4 phases)
│
├── docs/                              # Documentation
│   ├── QUICKSTART_V3.md               # Quick start guide
│   ├── MF_CLOUD_IMPACT_ANALYSIS.md    # Key finding: MF cloud reverses PCA/UMAP
│   ├── HPC_EXECUTION_GUIDE.md         # HPC operations manual
│   ├── archive_v2.0/                  # Archived v2.0 documentation
│   └── archive_development/           # Development/debugging docs
│
├── scripts/                           # Analysis scripts
│   ├── analyze_potency_stratified_enrichment.py  # Potency-stratified analysis (parallelized)
│   └── ...                            # Other analysis scripts
│
├── generate_phase*_configs.py         # Config generators (4 phases)
├── extract_phase1_best_configs.py     # Phase 1 winner extraction
├── main_orchestrator.py               # Single experiment executor
│
├── core_scripts/                      # Core pipeline modules
│   ├── calculate_features_and_fingerprints_exp.py
│   ├── calculate_similarityspaces_exp.py
│   └── utils.py
│
├── experimental_pipeline/             # Experimental modules
│   ├── prepare_data.py                # Data loading and filtering
│   ├── project_and_analyze.py         # DR fitting and projection
│   └── rank_zinc_decoys.py            # Scoring and evaluation
│
├── datasets/                          # ChEMBL target ligands + MF clouds
├── hyperparam_configs_v3_phase*/      # Generated configs (gitignored)
├── experiment_workspace_v3_phase*/    # Results (symlinks to /work, gitignored)
└── final_report_*/                    # Aggregated analysis reports
```

---

## Getting Started

### 1. Environment Setup

```bash
# Clone repository
git clone <repo_url>
cd UMMBAS_screening_experiments
git checkout 3.0

# Create conda environment
conda env create -f environment.yml
conda activate ummbas-screening
```

### 2. Generate Phase 1 Configs

```bash
python generate_phase1_configs.py
# Output: hyperparam_configs_v3_phase1/ (260 configs)
```

### 3. Local Execution (Single Experiment)

```bash
python main_orchestrator.py \
  --config hyperparam_configs_v3_phase1/config_ABL1_features_pca_2D_seed42.json
```

### 4. HPC Submission (All Phase 1)

```bash
# Setup workspace directories (one-time)
bash hpc/setup_hpc_workspaces.sh

# Submit all 260 jobs
bash hpc/submit_v3_phase1.sh
```

### 5. Extract Best Configurations

```bash
python extract_phase1_best_configs.py \
  --workspace experiment_workspace_v3_phase1 \
  --output phase1_best_configs.json
```

### 6. Run Subsequent Phases

```bash
# Generate and submit Phase 2 (cutoff sensitivity - REORDERED)
python generate_phase2_configs.py
# Option A: Use orchestrator script (recommended, reuses Phase 1 data)
python scripts/run_phase2_cutoff_analysis.py \
  --config_dir hyperparam_configs_v3_phase2_cutoff \
  --phase1_workspace experiment_workspace_v3_phase1 \
  --phase2_workspace experiment_workspace_v3_phase2
# Option B: HPC submission
bash hpc/submit_v3_phase2.sh

# Generate and submit Phase 3 (MF cloud ablation - REORDERED, uses Phase 2 optimal cutoff)
python generate_phase3_configs.py
bash hpc/submit_v3_phase3.sh

# Generate and submit Phase 4 (generalization - REORDERED, uses Phase 2 optimal cutoff)
python generate_phase4_configs.py
bash hpc/submit_v3_phase4.sh
```

### 7. HPC Post-Analysis (All Phases)

After running experiments, use the HPC analysis suite scripts to aggregate results and generate publication-quality plots:

```bash
# Phase 1 Analysis (Post-analysis + Stratified Scores)
sbatch hpc/phase1_analysis_suite.sh experiment_workspace_v4 reporting

# Phase 2 Analysis (Cutoff Sensitivity)
sbatch hpc/phase2_analysis_suite.sh experiment_workspace_v4 reporting cutoff_sweep

# Phase 3 Analysis (MF Cloud Ablation)
sbatch hpc/phase3_analysis_suite.sh experiment_workspace_v4 reporting mf_ablation

# Phase 3 with potency stratification
sbatch hpc/phase3_analysis_suite.sh experiment_workspace_v4 reporting mf_ablation --stratify

# Phase 4 Analysis (Cross-Target Generalization)
sbatch hpc/phase4_post_analysis.sh
```

**Analysis Script Features**:
- **Phase 1**: 
  - Post-analysis: Method comparison, hyperparameter heatmaps, seed variability
  - Stratified scores: Tier-stratified enrichment analysis (High/Medium/Weak potency)
  - Outputs: `reporting/phase1_post_analysis/` and `reporting/phase1_stratified/`
  
- **Phase 2**:
  - Cutoff sensitivity analysis (100 nM, 1 μM, 10 μM, 100 μM)
  - Reuses Phase 1 similarity spaces (~75% time savings)
  - Outputs: `reporting/phase2_post_analysis/`
  
- **Phase 3**:
  - MF cloud ablation degradation curves
  - Optional potency tier stratification with `--stratify`
  - Outputs: `reporting/phase3_post_analysis/`
  
- **Phase 4**:
  - Cross-target generalization analysis
  - Natural MF size correlation plots
  - Outputs: `reporting/phase4_post_analysis/`

**Resource Allocation**:
- CPUs: 16-64 cores (Phase 1 uses 64 for stratified analysis)
- Memory: 200-300G
- Time: 12 hours max

### 8. Analyze Results with Potency Stratification

```bash
# Run parallelized potency-stratified enrichment analysis
python scripts/analyze_potency_stratified_enrichment.py \
  --workspace_dir experiment_workspace_v3_phase1 \
  --output_dir potency_analysis_results \
  --n_jobs 8  # Optional: specify number of parallel workers

# Outputs:
# - stratified_enrichment_detailed.csv (all runs)
# - stratified_enrichment_summary.csv (aggregated by config)
# - plots/ directory with PNG + PDF figures
# - potency_stratified_report.txt (summary findings)
```

**Features**:
- Parallelized analysis (4-8× faster on multi-core systems)
- Dimension-separated plots for publication
- Dual PNG (300 DPI) + PDF (vector) output
- Best hyperparameter filtering for fair comparisons
- Potency tier stratification (High: 0.1-100 nM, Medium: 100-1000 nM, Weak: 1000-100,000 nM)

---

## Documentation

- **Pipeline Guide**: [`README_V3_PIPELINE.md`](README_V3_PIPELINE.md) - Complete experimental design and execution
- **Methods**: [`METHODS_FOR_PAPER.md`](METHODS_FOR_PAPER.md) - Publication-ready methods section
- **Quick Start**: [`docs/QUICKSTART_V3.md`](docs/QUICKSTART_V3.md) - Fast setup and execution
- **Key Finding**: [`docs/MF_CLOUD_IMPACT_ANALYSIS.md`](docs/MF_CLOUD_IMPACT_ANALYSIS.md) - MF cloud phase transition analysis
- **HPC Guide**: [`hpc/README.md`](hpc/README.md) - Cluster-specific instructions

### Phase 1 Post-analysis (v4, molfuse)

The standalone post-analysis utility scans a v4 workspace and generates consolidated figures with consistent scales:

- `scripts/phase1_post_analysis.py`
  - Inputs: `--workspace_dir experiment_workspace_v4` and optional `--phase` (default `phase1`)
  - Outputs: CSV summaries, best-config JSON, and combined plots under `reporting/phase1_post_analysis/plots/`
  - Combined figures (shared axes):
    - Bars: EF@1% across dims (columns) with method on x (PCA, UMAP avg, UMAP best) and representation as hue; EF@5/10 removed from defaults
    - UMAP heatmaps: grid by representation × dimension with a shared colorbar; panels with only one hyperparameter cell are hidden
    - Seed variability: grid by method × dimension, violin/box (seaborn optional)
    - Distance diagnostics:
      1) Method-comparison CDF and histogram for ZINC only, in a fixed range (default [0, 0.5])
      2) Four dedicated histograms (best/avg × features/fingerprints), each overlaying ZINC vs ACTIVES
  - Flags:
  - `--metrics ef1` (default)
  - `--no-sharey` to disable shared y-axis on bars
  - `--distance_xranges 0-0.5` to control fixed-range distance plots

Example:

```bash
python scripts/phase1_post_analysis.py \
  --workspace_dir experiment_workspace_v4 \
  --phase phase1 \
  --output_dir reporting/phase1_post_analysis \
  --metrics ef1,ef5,ef10 \
  --distance_xranges 0-1,0-5
```

---

## Citation

If you use UMMBAS in your research, please cite:

```
[Citation to be added upon publication]
```

---

## License

[MIT License](LICENSE)

---

## Contact

For questions or issues, please open a GitHub issue or contact [your email].

---

**Version**: 3.0  
**Last Updated**: October 2025  
**Branch**: `3.0`

## Benchmark: cdist vs KDTree/BallTree (exact NN)

This repository includes a benchmark to compare the current scoring bottleneck (batched `scipy.spatial.distance.cdist`) against an exact 1-NN index using scikit-learn's `NearestNeighbors` with KDTree/BallTree. In low/medium dimensions (2–10), the KDTree/BallTree approach is typically faster and memory-friendlier while producing identical nearest-neighbor distances (within floating-point tolerance), so rankings and EF metrics remain unchanged.

Script: `analysis_scripts/benchmark_cdist_vs_kdtree.py`

Inputs and behavior
- Reads a Phase 1/2 simspace CSV (MF cloud + ZINC) and splits MF vs ZINC automatically using `DataSource` or `MOLECULE ID`.
- Optionally reads a projected actives CSV to benchmark both decoys and actives.
- Computes min distance to MF cloud via both methods and compares timings and equality.
- Outputs a JSON summary and optional CSV sample of distances.

Example usage (HPC-friendly)

```bash
# ZINC-only benchmark (uses all MF cloud points, samples queries for speed)
python analysis_scripts/benchmark_cdist_vs_kdtree.py \
  --simspace_csv_path /path/to/simspace.csv \
  --dr_short_name PCA \
  --simspace_dim 5 \
  --query_source zinc \
  --sample_query 100000 \
  --algorithm kd_tree \
  --batch_size 5000 \
  --tolerance 1e-6 \
  --output_json kd_benchmark_zinc.json

# Actives + ZINC (if you have the projected actives CSV)
python analysis_scripts/benchmark_cdist_vs_kdtree.py \
  --simspace_csv_path /path/to/simspace.csv \
  --actives_csv_path /path/to/TARGET_actives_complete_features_PCA_dim5.csv \
  --dr_short_name PCA \
  --simspace_dim 5 \
  --query_source both \
  --sample_query 50000 \
  --algorithm kd_tree \
  --output_json kd_benchmark_both.json \
  --save_sample_csv kd_benchmark_sample.csv
```

Notes
- Exact NN only: `NearestNeighbors` with `algorithm=kd_tree`/`ball_tree` is exact for Euclidean distance. Results should match `cdist` within a small tolerance (defaults to 1e-6).
- High dimensions: In very high-D (>~50), tree-based speedups may diminish; `cdist` can then be competitive.
- Memory: The NN approach avoids allocating large N×M distance matrices.

---

## Alternative parallel dataset recreation (Mordred)

An HPC-optimized, parallel script is available to recreate the full Mordred descriptor datasets with significantly higher throughput:

- Baseline script: `tests/mordred_full_feature_eval/recreate_datasets.py` (single-process, chunked)
- Parallel alternative: `tests/mordred_full_feature_eval/recreate_datasets_parallel.py` (multiprocessing)

Quick usage (local/HPC):

```bash
python tests/mordred_full_feature_eval/recreate_datasets_parallel.py \
  --base_dir . \
  --output_dir tests/mordred_full_feature_eval/output_full_datasets \
  --workers 64 \
  --chunk-size 50000 \
  --batch-2d 1000 \
  --batch-3d 250
```

Outputs:
- `datasets_2d_all/` and `datasets_2d3d_all/` under `--output_dir`
- Metadata columns preserved; feature columns replaced by full Mordred sets
- Fingerprint CSVs filtered to match the recreated feature CSVs

Performance tips:
- The script parallelizes across processes; each 3D embed uses 1 internal thread to avoid oversubscription.
- Adjust `--batch-3d` if 3D embedding becomes the bottleneck or memory is tight; 200–400 is a good range on 64 cores.
