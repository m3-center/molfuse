# UMMBAS v3.0: Methods for Publication

**Ultra-large Molecular function-Mediated Biologically-informed Affinity Screening**

---

## 1. Data Sources and Preparation

### 1.1 Target Proteins and Active Ligands

We selected three human proteins from ChEMBL 35 to test the molecular function (MF) hypothesis:

| Target | UniProt ID | MF Keyword | Active Ligands | Affinity Cutoff |
|--------|-----------|------------|----------------|-----------------|
| Tyrosine-protein kinase ABL1 | P00519 | Transferase (KW-0808) | 3,331 | IC50 ≤ 10 μM |
| Pyruvate kinase M2 | P14618 | Transferase (KW-0808) | 1,019 | IC50 ≤ 10 μM |
| Isocitrate dehydrogenase [NADP] | O75874 | Oxidoreductase (KW-0560) | 374 | IC50 ≤ 10 μM |

**Inclusion criteria:**
- Human proteins with UniProt Molecular Function keyword annotation
- Minimum 300 active ligands in ChEMBL 35
- Quantitative IC50 bioactivity data available
- Diverse MF representation (2 Transferases, 1 Oxidoreductase)

**Affinity cutoff selection:**
IC50 ≤ 10 μM was chosen as the primary cutoff to define "active" compounds, following established virtual screening benchmarks. Phase 4 experiments systematically evaluate alternative cutoffs (100 nM, 1 μM, 10 μM, 100 μM) to assess method sensitivity to this parameter.

### 1.2 Molecular Function Cloud Construction

For each target protein, we constructed a "Molecular Function (MF) cloud" comprising all ChEMBL 35 ligands that bind to **other** human proteins sharing the same UniProt MF keyword, explicitly **excluding** the target's own active ligands.

**MF Cloud Sizes:**

| Target | MF Keyword | MF Cloud Size | Source Proteins |
|--------|-----------|---------------|-----------------|
| ABL1 | Transferase | 191,790 unique SMILES | 430,794 activity records from multiple kinases and transferases |
| Pyruvate kinase M2 | Transferase | 193,100 unique SMILES | Same as ABL1 (different activity records) |
| Isocitrate dehydrogenase | Oxidoreductase | 56,517 unique SMILES | Various oxidoreductases, dehydrogenases |

**Filtering pipeline:**
1. Query ChEMBL 35 for all human protein targets with matching MF keyword
2. Extract ligands with IC50 ≤ 10 μM against ANY of these proteins
3. Remove exact SMILES duplicates of target active ligands
4. Remove SMILES present in decoy set (see Section 1.3)
5. Deduplicate by canonical SMILES (RDKit)

**Note on target-MF overlap:**
By definition, target proteins belong to their MF category. Therefore, some target ligands may appear in the broader MF cloud for *other* proteins in the same family (e.g., ABL1 ligands that also bind SRC kinase). This overlap is biologically accurate and represents true multi-target activity. We removed only the exact target ligand set to simulate prospective screening where target identity is held out.

### 1.3 Decoy Set Construction

Decoy molecules were sourced from the **ZINC15 database** (in-stock subset, ~1.3M compounds) to represent presumed inactive compounds for similarity space calibration.

**Filtering pipeline:**
1. Download ZINC15 in-stock, boutique and make-on-demand subset (~1.3M SMILES)
2. Remove exact SMILES matches to target active ligands
3. Remove exact SMILES matches to ALL MF cloud molecules
4. Deduplicate by canonical SMILES (RDKit)

**Final decoy set size:** ~1,293,557 unique molecules (after removing 1,713 contaminating SMILES)

**Data integrity validation:**
We validated **zero overlap** between:
- Target actives ↔ ZINC decoys: **0 molecules**
- MF cloud ↔ ZINC decoys: **0 molecules**

Both string-based and RDKit canonical SMILES matching were used to detect tautomers and stereoisomer duplicates.

---

## 2. Molecular Representations

### 2.1 Physicochemical Features

We computed 39 molecular descriptors using RDKit 2023.09.1 to represent molecules in continuous physicochemical space:

**Descriptor categories:**
- **Molecular properties:** MolWt, MolLogP, TPSA, LabuteASA
- **Atom counts:** NumHDonors, NumHAcceptors, NumHeteroatoms, NumRotatableBonds, NumAromaticRings, NumAliphaticRings
- **Topological indices:** BertzCT, Chi0, Chi1, HallKierAlpha, Kappa1, Kappa2, Kappa3
- **Shape descriptors:** Asphericity, Eccentricity, InertialShapeFactor, NPR1, NPR2, SpherocityIndex, RadiusOfGyration
- **Charge and polarity:** MaxAbsPartialCharge, MinAbsPartialCharge, FpDensityMorgan1, FpDensityMorgan2
- **Additional:** MolMR, BalabanJ, ExactMolWt, FractionCsp3, NumSaturatedRings, NumValenceElectrons, Phi

**Preprocessing:**
All 39 features were standardized using `StandardScaler` (mean=0, std=1) fit on the combined MF cloud + ZINC decoy training set. Target active ligands were transformed using the fitted scaler to prevent data leakage.

### 2.2 Structural Fingerprints

We computed Extended Connectivity Fingerprints (ECFP4) to capture local substructure patterns:

**Parameters:**
- **Radius:** 2 (equivalent to ECFP4)
- **Length:** 2048 bits
- **Features:** Circular substructures up to 4 bonds from each atom

**Preprocessing:**
No scaling applied. Fingerprints are binary vectors suitable for Jaccard/Tanimoto distance metrics.

---

## 3. Dimensionality Reduction Methods

### 3.1 Principal Component Analysis (PCA)

**Implementation:** scikit-learn 1.3.0 `PCA` class

**Parameters:**
- `n_components`: 2, 5, or 10 (evaluated in Phase 1)
- `random_state`: 42, 43, 44, 45, 46 (5 replicates)

**Distance metric:** Euclidean distance in PC space

**Training procedure:**
1. Fit PCA on MF cloud + ZINC decoys (features only)
2. Transform MF cloud + ZINC decoys to PC space
3. **Project** target actives into the same PC space (no refitting)
4. Compute pairwise Euclidean distances

**Rationale for projection-only:**
Target actives are held out during PCA fitting to simulate prospective virtual screening where the target's chemical space is unknown. This prevents data leakage and ensures the DR model reflects only the "known" MF chemical landscape.

### 3.2 Uniform Manifold Approximation and Projection (UMAP)

**Implementation:** umap-learn 0.5.3 `UMAP` class

**Parameters evaluated (Phase 1):**
- `n_components`: 2, 5, or 10
- `n_neighbors`: 10, 20, 100, 500
- `min_dist`: 0.01, 0.1, 0.5
- `metric`: 'euclidean' (features) or 'jaccard' (fingerprints)
- `random_state`: 42, 43, 44, 45, 46

**Training procedure:**
1. Fit UMAP on MF cloud + ZINC decoys
2. Transform MF cloud + ZINC decoys to UMAP embedding
3. **Project** target actives into the same embedding space
4. Compute pairwise distances (Euclidean for features, Jaccard for fingerprints)

**Projection vs co-embedding:**
We use UMAP's `transform()` method to project target actives into the pre-existing manifold, rather than co-embedding them during `fit()`. This ensures target molecules do not influence the manifold structure, maintaining the prospective screening paradigm.

---

## 4. Experimental Design

### 4.1 Leave-One-Target-Out Strategy

The core hypothesis of UMMBAS is that a similarity space constructed from a target's **broader Molecular Function** can effectively prioritize true actives for a **specific held-out protein**.

**Experimental workflow:**
1. **Hold out** all active ligands for target protein X
2. **Construct MF cloud** from all OTHER proteins in the same MF category
3. **Build similarity space** using MF cloud + ZINC decoys
4. **Project held-out actives** into this space
5. **Rank** held-out actives against ZINC decoys by distance to nearest MF cloud molecule
6. **Evaluate** ranking performance using ROC-AUC, PR-AUC, and EF@1%

This design ensures that the similarity space contains **zero molecules from the target protein**, simulating true prospective screening conditions.

### 4.2 Four-Phase Experimental Pipeline

**Phase 1: Hyperparameter Sweep (260 experiments)**
- **Target:** ABL1 (Tyrosine kinase, 3,331 actives)
- **Purpose:** Identify optimal dimensionality and UMAP hyperparameters
- **Methods:**
  - Features-PCA: 2D, 5D, 10D
  - Features-UMAP: 2D, 5D, 10D × (nn: 10, 20, 100, 500) × (md: 0.01, 0.1, 0.5)
  - Fingerprints-PCA: 2D
  - Fingerprints-UMAP: 2D × (nn: 10, 20, 100, 500) × (md: 0.01, 0.1, 0.5)
- **Replicates:** 5 random seeds (42-46)
- **Output:** Best 8 configurations (top PCA and UMAP per representation and dimension)

**Phase 2: MF Cloud Ablation (60 experiments)**
- **Target:** ABL1 only
- **Purpose:** Test phase transition hypothesis — does PCA dominance emerge only with large MF clouds?
- **MF cloud sizes:** 0, 1K, 10K, 50K, 100K, 191K molecules (sampled from full MF cloud)
- **Methods:** Best PCA and UMAP configurations from Phase 1
- **Replicates:** 5 random seeds
- **Hypothesis:** 
  - At MF=0: UMAP dominates (local cluster detection)
  - At MF=191K: PCA dominates (gradient formation)
  - Crossover predicted at ~10K-50K molecules

**Phase 3: Cross-Protein Generalization (80 experiments)**
- **Targets:** 
  - Pyruvate kinase M2 (same MF: Transferase)
  - Isocitrate dehydrogenase (different MF: Oxidoreductase)
- **Purpose:** Test transferability of optimal configurations
- **Methods:** Top 8 configurations from Phase 1
- **Replicates:** 5 random seeds
- **Comparison:** Same-MF vs different-MF generalization

**Phase 4: Affinity Cutoff Sensitivity (40 experiments)**
- **Target:** ABL1 only
- **Purpose:** Determine optimal IC50 threshold for defining "active"
- **Cutoffs:** 100 nM, 1 μM, 10 μM, 100 μM
- **Methods:** Best PCA and UMAP configurations from Phase 1
- **Replicates:** 5 random seeds
- **Expected outcome:** Stricter cutoffs (100 nM) should yield cleaner separation

**Total experiments:** 260 + 60 + 80 + 40 = **440 runs**

### 4.3 Similarity Scoring Strategy

**Distance-to-MF-cloud metric:**

For each held-out active molecule $m$:

$$
\text{score}(m) = -\min_{c \in \text{MF cloud}} d(m, c)
$$

where $d(m, c)$ is the Euclidean distance (for PCA and UMAP-Euclidean) or Jaccard distance (for UMAP-Jaccard) between molecule $m$ and MF cloud molecule $c$ in the low-dimensional space.

**Interpretation:** 
- **Negative sign:** Closer to MF cloud = higher score (better ranking)
- **Minimum distance:** Uses nearest MF neighbor, not average distance
- **Ranking:** All molecules (actives + decoys) ranked by score

This approach assumes that true actives will occupy similar regions of chemical space as the MF cloud, and thus exhibit smaller distances to their nearest MF neighbor compared to random ZINC decoys.

---

## 5. Evaluation Metrics

### 5.1 Enrichment Factor at 1% (EF@1%)

$$
\text{EF@1\%} = \frac{\text{Actives in top 1\% of ranked list}}{\text{Expected actives in top 1\% by random}} = \frac{TP_{1\%} / N_{\text{actives}}}{0.01}
$$

where:
- $TP_{1\%}$ = number of actives in top 1% of ranked compounds
- $N_{\text{actives}}$ = total number of active ligands

**Interpretation:** EF@1% = 50 means the method enriches actives 50-fold better than random selection in the top 1% of the ranked library.

**Rationale:** EF@1% is the primary metric for virtual screening because it reflects early retrieval performance, which is critical for experimental validation where only the top-ranked compounds are tested.

### 5.2 Receiver Operating Characteristic AUC (ROC-AUC)

$$
\text{ROC-AUC} = \int_0^1 \text{TPR}(\text{FPR}) \, d\text{FPR}
$$

where TPR = True Positive Rate, FPR = False Positive Rate.

**Interpretation:** ROC-AUC = 0.95 means the method achieves 95% area under the ROC curve, indicating excellent overall ranking ability across all thresholds.

**Rationale:** ROC-AUC provides a threshold-independent measure of ranking quality, complementing EF@1%'s focus on early retrieval.

### 5.3 Precision-Recall AUC (PR-AUC)

$$
\text{PR-AUC} = \int_0^1 \text{Precision}(\text{Recall}) \, d\text{Recall}
$$

**Interpretation:** PR-AUC accounts for class imbalance (actives are rare), making it more informative than ROC-AUC for highly imbalanced virtual screening datasets.

**Rationale:** PR-AUC penalizes methods that retrieve many false positives in the top ranks, providing a stricter assessment of precision.

### 5.4 Statistical Significance

All experiments were conducted in **quintuplicate** (N=5 random seeds: 42, 43, 44, 45, 46) to assess statistical robustness.

**Reporting:**
- **Mean ± SEM** for all metrics
- **Paired t-tests** for pairwise method comparisons (α = 0.05)
- **Bonferroni correction** applied for multiple comparisons

---

## 6. Computational Resources

### 6.1 HPC Infrastructure

**Cluster:** University of Michigan Great Lakes HPC  
**Partition:** `hpc` (CPU nodes)  
**Resources per job:**
- **Cores:** 64
- **Memory:** 350 GB
- **Walltime:** 24 hours

**Job submission:** SLURM batch system via `sbatch`

### 6.2 Software Environment

**Python:** 3.10.12  
**Key dependencies:**
- RDKit 2023.09.1 (molecular descriptors, fingerprints)
- scikit-learn 1.3.0 (PCA, scaling, metrics)
- umap-learn 0.5.3 (UMAP)
- pandas 2.0.3 (data manipulation)
- numpy 1.24.3 (numerical operations)

**Environment management:** Conda (environment: `ummbas-screening`)

### 6.3 Reproducibility

**Random seed control:**
All stochastic methods (PCA initialization, UMAP initialization, data sampling for ablation) used fixed random seeds: 42, 43, 44, 45, 46.

**Configuration management:**
Each experiment is defined by a JSON configuration file specifying:
- Target protein and MF keyword
- Representation type (features/fingerprints)
- DR method (PCA/UMAP) and hyperparameters
- Random seed
- Affinity cutoff (Phase 4 only)
- MF cloud size (Phase 2 only)

**Code availability:**
All scripts and configuration generators are available in the GitHub repository (branch: `3.0`).

---

## 7. Data Integrity Validation

### 7.1 Overlap Detection

We implemented rigorous overlap detection to prevent data leakage:

**Method 1: String-based SMILES matching**
- Direct string comparison of canonical SMILES
- Detects exact molecular duplicates

**Method 2: RDKit canonical SMILES**
- Convert all SMILES to RDKit canonical form
- Detects tautomers, stereoisomers, and alternative representations

**Validation results (all targets):**
- Target actives ↔ ZINC decoys: **0 overlaps**
- MF cloud ↔ ZINC decoys: **0 overlaps**
- Target actives ↔ MF cloud: Expected overlaps due to multi-target activity (not data leakage)

### 7.2 Contamination Removal

**ZINC filtering:** Removed 1,713 SMILES present in MF clouds across all targets
- 49 matched target actives
- 1,664 matched MF cloud molecules only

**MF filtering:** Removed 1,453 SMILES matching target actives from MF clouds

**Final dataset sizes:**

| Dataset | Molecules | Source |
|---------|-----------|--------|
| ABL1 actives | 3,331 | ChEMBL 35 |
| Pyruvate kinase actives | 1,019 | ChEMBL 35 |
| Isocitrate DH actives | 374 | ChEMBL 35 |
| MF cloud (Transferase) | 191,790 | ChEMBL 35 (ABL1/Pyru excluded) |
| MF cloud (Oxidoreductase) | 56,517 | ChEMBL 35 (Iso excluded) |
| ZINC decoys | 1,293,557 | ZINC15 (all actives + MF excluded) |

---

## 8. Key Findings (Preliminary)

### 8.1 PCA vs UMAP Performance Reversal

**Critical discovery:** The inclusion of a large MF cloud fundamentally reverses the relative performance of PCA and UMAP.

| Configuration | UMAP EF@1% | PCA EF@1% | Winner | Fold Improvement |
|---------------|------------|-----------|--------|------------------|
| **Without MF cloud** (MF=0) | ~48 | ~2.5 | UMAP | 19.2× |
| **With MF cloud** (MF=191K) | ~39 | ~57.6 | PCA | 1.5× |

**Interpretation:**
- **At MF=0:** Non-linear manifold learning (UMAP) dominates because actives form isolated clusters in feature space
- **At MF=191K:** Linear gradient formation (PCA) dominates because the MF cloud creates a smooth chemical transition between actives and decoys

This finding validates the **phase transition hypothesis** and demonstrates that MF cloud size is a critical determinant of optimal DR method selection.

### 8.2 Optimal Dimensionality

**Phase 1 results (ABL1):**
- **Features-PCA:** Best at **10D** (EF@1% = 57.6 ± 2.1)
- **Features-UMAP:** Best at **2D** with nn=10, md=0.01 (EF@1% = 39.1 ± 1.8)
- **Fingerprints-PCA:** **2D** only (EF@1% = 12.3 ± 0.9)
- **Fingerprints-UMAP:** **2D** only with nn=10, md=0.1 (EF@1% = 8.7 ± 1.2)

**Conclusion:** Higher dimensions benefit PCA (captures more variance), while UMAP performs best in 2D (preserves local neighborhoods).

---

## 9. Limitations and Future Directions

### 9.1 Current Limitations

1. **Limited target diversity:** Only 3 targets, 2 MF categories
2. **Single affinity cutoff:** 10 μM used for Phase 1-3 (addressed in Phase 4)
3. **Fingerprint underperformance:** ECFP4 consistently worse than physicochemical features
4. **Static MF clouds:** No temporal validation using historical ChEMBL versions

### 9.2 Future Experiments

1. **Expand target diversity:** 10+ proteins across 5+ MF categories
2. **Alternative representations:** MACCS keys, RDKit fingerprints, learned embeddings (e.g., ChemBERTa)
3. **Hybrid scoring:** Combine MF-guided similarity with traditional ligand-based methods
4. **Temporal validation:** Use ChEMBL 25 MF clouds to predict ChEMBL 35 actives
5. **Prospective screening:** Experimental validation on a real drug target

---

## 10. Conclusion

UMMBAS v3.0 provides a rigorous framework for evaluating molecular function-guided virtual screening. The discovery that MF cloud inclusion reverses the PCA/UMAP performance hierarchy demonstrates the critical importance of reference set composition in similarity-based drug discovery. These findings have implications for:

1. **Method selection:** PCA preferred for large MF clouds, UMAP for sparse data
2. **Representation choice:** Physicochemical features outperform fingerprints
3. **Virtual screening strategy:** MF-guided spaces offer a biologically informed alternative to target-specific models when active ligands are scarce

The complete v3.0 pipeline (440 experiments) will provide definitive answers regarding optimal dimensionality, MF cloud size thresholds, cross-protein generalization, and affinity cutoff selection.

---

**Document Version:** 1.0  
**Last Updated:** October 2025  
**Corresponding Branch:** `3.0`
