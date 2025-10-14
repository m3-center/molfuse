# UMMBAS Screening Experiments - v2.0

### **Experimental Design: Evaluating Molecular Function-Guided Similarity Spaces for Ligand-Based Virtual Screening**

> **⚠️ Repository Status**: This is version 2.0 of the UMMBAS pipeline. Major refactoring completed to fix critical v1.0 errors. See [`docs/REFACTORING_PLAN_v2.0.md`](docs/REFACTORING_PLAN_v2.0.md) for details.

#### **Quick Start**
- **Repository Structure**: See [`docs/REPOSITORY_STRUCTURE.md`](docs/REPOSITORY_STRUCTURE.md) for complete organization
- **Baselines**: See [`baselines/README.md`](baselines/README.md) for baseline implementations
- **Generalization Experiments**: See [`docs/GENERALIZATION_EXPERIMENT_README.md`](docs/GENERALIZATION_EXPERIMENT_README.md)
- **Complete Pipeline Guide**: See [`docs/README_COMPLETE_PIPELINE.md`](docs/README_COMPLETE_PIPELINE.md)

#### **Abstract**

A significant challenge in ligand-based drug discovery is the scarcity of known active compounds for novel or understudied protein targets. To address this, we developed and validated a computational framework that leverages the target's broader Molecular Function (MF) to enrich the pool of relevant chemical matter for virtual screening. This study employs a rigorous "leave-one-target-out" experimental design to assess the hypothesis that a similarity space, built from a general MF chemical landscape, can effectively prioritize true active ligands for a specific, held-out protein. 

**v2.0 Methodology**: We systematically evaluate physicochemical features (39 RDKit descriptors) using PCA and UMAP-Euclidean dimensionality reduction with a projection-only strategy. Performance is assessed over five replicate runs by ranking held-out actives against a large decoy set using ROC-AUC, PR-AUC, and Enrichment Factor at 1% (EF@1%), with comparisons to four baseline methods. 

#### **Methodology**

**1. Experimental Hypothesis and Design**

The central hypothesis of this study is that ligands active against a specific protein will exhibit measurable proximity to a "cloud" of compounds known to modulate other proteins that share the same molecular function. To test this, we implemented a leave-one-target-out experimental design. 

**v2.0 Targets (ChEMBL 35)**:
- **ABL1 Kinase** (P00519): 3,331 actives | MF: Transferase (KW-0808) | MF Cloud: 191,648 compounds
- **Pyruvate Kinase M2** (P14618): 1,019 actives | MF: Transferase (KW-0808) | MF Cloud: 191,648 compounds  
- **Isocitrate Dehydrogenase** (O75874): 374 actives | MF: Oxidoreductase (KW-0560) | MF Cloud: 56,533 compounds

For each target, its known ligands were sequestered as a "held-out active set". A reference chemical space was constructed using two populations: (1) an "MF Cloud," comprising all ligands from ChEMBL known to be active against any *other* human protein with the same UniProt MF keyword, explicitly excluding the held-out actives; and (2) a "Decoy Set," consisting of 1.2M+ presumed inactive compounds from ZINC, excluding any molecules identical to held-out actives.

The evaluation task: Can a scoring function based on proximity within this MF-guided space effectively rank held-out actives significantly higher than decoys? This process was repeated in quintuplicate (N=5) using different random seeds for statistical robustness.

**2. Molecular Representations (v2.0)**

*   **Physicochemical Features**: 39 RDKit descriptors (molecular weight, atom counts, topological indices, etc.) representing molecules in continuous property space. StandardScaler preprocessing applied.
*   **Structural Fingerprints (Future)**: 2048-bit ECFP4 fingerprints with Jaccard distance and UMAP. No scaling, no PCA pre-reduction.

**3. Similarity Space Construction (v2.0 - Projection-Only)**

Low-dimensional similarity spaces are generated using two dimensionality reduction algorithms:

*   **PCA (Principal Component Analysis)**: Linear dimensionality reduction for features
*   **UMAP (Uniform Manifold Approximation and Projection)**: Non-linear manifold learning with Euclidean distance for features

**Critical v2.0 Change**: **Projection-Only Strategy**
- DR models are fit using only the MF Cloud and Decoy Set
- Held-out actives are then transformed into this pre-existing space using the saved model
- This simulates true prospective screening (no data leakage)
- **Rationale**: t-SNE and co-embedding were removed due to architectural limitations (see [`docs/REFACTORING_PLAN_v2.0.md`](docs/REFACTORING_PLAN_v2.0.md))

**4. Performance Evaluation**

Performance is assessed by scoring all held-out actives and decoys based on their negative minimum Euclidean distance to any point in the MF Cloud within the generated similarity space. The ranked list is evaluated using:

*   **ROC-AUC**: Global discrimination ability across all thresholds
*   **PR-AUC**: Performance on imbalanced datasets (reported against random baseline)
*   **Enrichment Factor at 1% (EF@1%)**: Concentration of actives in top 1% vs random selection

**v2.0 Baselines**: Four baseline methods provide performance context:
- Random shuffling (1000 iterations, EF@1% ≈ 1.0 expected)
- 39D Euclidean distance (no dimensionality reduction)
- ECFP4 Tanimoto similarity (max similarity to MF cloud)
- Chemical novelty analysis (Tanimoto distribution quantification)

Results are aggregated across five replicate runs (mean ± SD), providing statistically sound comparisons.