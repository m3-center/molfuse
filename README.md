# UMMBAS v3.0 - Molecular Function-Guided Virtual Screening

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

### 7. Analyze Results with Potency Stratification

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