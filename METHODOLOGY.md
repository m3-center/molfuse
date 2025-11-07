# MolFuSE: Molecular Function-Guided Similarity Explorer for Virtual Screening

**Method Name**: MolFuSE (Molecular Function Similarity Explorer)  
**Classification**: Knowledge-based dimensionality reduction for ligand-based virtual screening  
**Version**: 4.0  
**Date**: November 2025

---

## Abstract

MolFuSE is a molecular function-guided virtual screening framework that leverages dimensionality reduction to project chemical space into low-dimensional similarity spaces optimized for target-specific ligand discovery. The method uses molecular function annotations from large-scale bioactivity databases to construct reference molecular clouds that anchor the similarity space, enabling enrichment of active compounds through distance-based scoring.

---

## 1. Conceptual Foundation

### 1.1 Core Hypothesis

**Molecular Function Proximity Principle**: Ligands active against a specific protein exhibit measurable chemical proximity to compounds that modulate other proteins sharing the same molecular function.

### 1.2 Theoretical Basis

Proteins within the same molecular function class (e.g., kinases, proteases, oxidoreductases) often:

1. **Share structural features**: Conserved active site architectures and binding pocket geometries
2. **Recognize similar chemistry**: Overlapping pharmacophoric requirements for substrate/inhibitor binding
3. **Form coherent chemical clouds**: Bioactive ligands cluster in chemical space regions defined by molecular function

### 1.3 Method Overview

MolFuSE exploits this principle by:

1. Constructing a **molecular function cloud**: Aggregating all known bioactive compounds for proteins in the target's molecular function class
2. **Dimensionality reduction**: Projecting high-dimensional molecular descriptors into low-dimensional similarity space (2D-10D)
3. **Anchor-based scoring**: Ranking candidate compounds by proximity to the molecular function cloud in the embedded space

This approach transforms virtual screening from pairwise similarity (typical ligand-based methods) to **population-based similarity**, where candidates are evaluated against the aggregate chemical knowledge of an entire protein family.

---

## 2. Data Components

### 2.1 Molecular Function Cloud

**Definition**: The molecular function (MF) cloud comprises all experimentally validated bioactive compounds for proteins sharing the target protein's molecular function annotation.

**Data Source**: ChEMBL bioactivity database, filtered by:
- Molecular function keyword (e.g., "Kinase" for tyrosine kinases)
- Target protein exclusion: compounds tested against the specific query target are removed to prevent data leakage

**Composition**:
- Typical size: 100,000-700,000 unique compounds (depending on molecular function)
- Affinity range: IC₅₀, Ki, or Kd ≤ 100 μM (default; adjustable via affinity cutoff parameter)
- Deduplication: One entry per unique compound (SMILES-based); for duplicates, median affinity across all targets is used

**Purpose**: Serves as a reference population that defines "molecular function-relevant" chemical space.

### 2.2 Decoy Library

**Definition**: Large library of presumably inactive compounds used as negative controls during ranking.

**Data Source**: ZINC database (clean-leads subset)

**Composition**:
- ~1.3 million drug-like compounds
- Properties: Lipinski's Rule of Five compliant, synthetically accessible
- Overlap removal: Any compound appearing in the MF cloud is excluded

**Purpose**: Provides background distribution against which active compound enrichment is measured.

### 2.3 Target Actives

**Definition**: Known active ligands for the specific query protein target.

**Data Source**: ChEMBL bioactivity database

**Composition**:
- Affinity threshold: IC₅₀, Ki, or Kd ≤ 100 μM (default)
- Typical size: hundreds to thousands of compounds per target
- Held-out during model training to prevent data leakage

**Purpose**: Evaluation set for measuring virtual screening performance (enrichment metrics).

---

## 3. Molecular Representation

### 3.1 Features Representation

**Type**: Continuous numerical descriptors

**Source**: Mordred 2D molecular descriptors

**Dimensionality**: 
- Raw: 1,613 2D descriptors
- After cleaning: ~1,477 descriptors (removal of zero-variance and all-missing features)

**Descriptor Categories**:
- Constitutional: atom counts, bond counts, molecular weight
- Topological: graph indices, shape descriptors, connectivity
- Geometric: 2D autocorrelations, distances
- Electronic: partial charges, electronegativity distributions
- Physicochemical: LogP, polar surface area, hydrogen bond donors/acceptors

**Preprocessing**:
1. **Imputation**: Missing values replaced with median (per-feature, calculated on MF cloud + decoys)
2. **Zero-variance filtering**: Features with variance < 10⁻¹² removed
3. **Standardization**: StandardScaler (zero mean, unit variance) fit on MF cloud + decoys only

**Distance Metric**: Euclidean distance (for dimensionality reduction and scoring)

### 3.2 Fingerprints Representation (Alternative)

**Type**: Binary molecular fingerprints

**Source**: Extended-Connectivity Fingerprints (ECFP4)

**Dimensionality**: 2,048 bits

**Parameters**:
- Radius: 2 (ECFP4)
- Folding: Hashed to 2,048 bits

**Preprocessing**: None (binary vectors used directly)

**Distance Metric**: Jaccard distance (for UMAP dimensionality reduction)

**Performance Note**: Features representation consistently outperforms fingerprints by 2-3× in enrichment metrics for our datasets.

---

## 4. Dimensionality Reduction

### 4.1 Overview

**Purpose**: Project high-dimensional molecular descriptors (~1,477D for features, 2,048D for fingerprints) into low-dimensional similarity space (2D-10D) suitable for visualization and distance-based scoring.

**Training Set**: MF cloud + decoy library (no target actives included to prevent data leakage)

**Projection Set**: Target actives are transformed into the learned space using the fitted model

### 4.2 Principal Component Analysis (PCA)

**Algorithm**: Linear orthogonal transformation that maximizes variance preservation

**Hyperparameters**:
- Dimensions: 2, 5, or 10 principal components

**Properties**:
- **Deterministic**: Identical results across runs
- **Global structure**: Preserves large-scale distances and variance patterns
- **Computational efficiency**: Fast fitting and transformation (~seconds for 1.5M compounds)

**Advantages**:
- Stable performance across dimensionality (57-58% enrichment at 1%)
- No hyperparameter tuning required
- Interpretable components (ordered by explained variance)

**Limitations**:
- Linear transformation only (cannot capture nonlinear manifold structure)
- Assumes Euclidean geometry in original space

### 4.3 Uniform Manifold Approximation and Projection (UMAP)

**Algorithm**: Nonlinear manifold learning via topological graph optimization

**Hyperparameters**:
- **Dimensions**: 2, 5, or 10
- **n_neighbors**: Number of nearest neighbors for local structure (10-500)
- **min_dist**: Minimum distance between points in embedded space (0.0-0.5)
- **metric**: Distance function in original space (Euclidean for features, Jaccard for fingerprints)

**Properties**:
- **Nonlinear**: Preserves local and global manifold structure
- **Stochastic**: Multiple runs with different random seeds yield slightly different embeddings
- **Hyperparameter-sensitive**: Performance varies significantly with n_neighbors choice

**Advantages**:
- Captures nonlinear relationships in molecular space
- Flexible topology preservation (tunable via min_dist)
- Potential for superior clustering of chemically similar compounds

**Limitations**:
- Hyperparameter tuning required (n_neighbors especially critical)
- Stochastic variability across runs
- Computationally expensive for large datasets (~minutes for 1.5M compounds)

---

## 5. Scoring and Ranking

### 5.1 Distance-Based Scoring

**Scoring Function**: Negative minimum Euclidean distance to the molecular function cloud

$$
\text{score}(x) = -\min_{m \in \text{MF}} \| x - m \|_2
$$

Where:
- $x$: Candidate compound in embedded space
- $\text{MF}$: Set of molecular function cloud compounds in embedded space
- $\| \cdot \|_2$: Euclidean distance (L2 norm)

**Interpretation**: Higher scores (closer to zero) indicate stronger proximity to molecular function-relevant chemistry.

**Implementation**: Exact 1-nearest neighbor search using KDTree (for Euclidean distances) or BallTree (for other metrics).

### 5.2 Affinity Cutoff Filtering

**Purpose**: Restrict molecular function cloud to high-potency ligands only, hypothesized to improve discriminative power.

**Cutoff Values**:
- 100 nM: High-potency only (drug-like, clinically relevant)
- 1 μM: High + moderate potency (balanced)
- 10 μM: Permissive (includes weaker binders)
- 100 μM: Maximum diversity (default, all bioactives)

**Application**: Cutoff filters the MF cloud **for scoring only**; actives are never filtered.

**Research Question**: Can stricter cutoffs improve enrichment by focusing on high-potency chemistry? (Tested in Phase 2 experiments)

### 5.3 Ranking

**Procedure**:
1. Compute score for each candidate compound (target actives + decoys)
2. Rank all compounds in descending order of score
3. Evaluate enrichment of known actives in top-ranked subsets

---

## 6. Evaluation Metrics

### 6.1 Enrichment Factor at k%

**Definition**: Ratio of active compound concentration in the top k% of ranked list vs random selection.

$$
\text{EF}_{k\%} = \frac{\text{Actives}_{\text{top } k\%} / N_{\text{top } k\%}}{\text{Actives}_{\text{total}} / N_{\text{total}}}
$$

**Interpretation**:
- EF@1% = 50: Top 1% of ranked compounds contains 50× more actives than random
- EF@1% = 1: Performance equivalent to random selection

**Clinical Relevance**: EF@1% reflects practical utility for experimental validation (reviewing top 1% of ranked library).

**Primary Metric**: EF@1% used for hyperparameter selection and method comparison.

### 6.2 Receiver Operating Characteristic - Area Under Curve (ROC-AUC)

**Definition**: Probability that a randomly selected active is ranked higher than a randomly selected decoy.

**Range**: 0.5 (random) to 1.0 (perfect separation)

**Interpretation**: Global ranking quality across all thresholds.

### 6.3 Precision-Recall - Area Under Curve (PR-AUC)

**Definition**: Average precision across all recall levels, weighted by class imbalance.

**Range**: Random baseline = fraction of actives; 1.0 = perfect

**Interpretation**: More sensitive to class imbalance than ROC-AUC; preferred for highly imbalanced datasets.

### 6.4 Spearman Rank Correlation

**Definition**: Correlation between predicted scores and experimental potency values (pActivity = 9 - log₁₀[IC₅₀(nM)]) for actives only.

**Range**: -1 to +1

**Interpretation**: Measures whether scoring function captures potency trends within active compounds.

---

## 7. Experimental Design Invariants

### 7.1 Data Leakage Prevention

**Target-Preserving Exclusion**: Any compound tested against the query target is excluded from the molecular function cloud.

**Rationale**: Prevents trivial enrichment from compounds already known to bind the target.

**Projection-Only Evaluation**: Target actives are never included in dimensionality reduction training; they are only transformed using the fitted model.

**Scaler Fitting**: StandardScaler fit on MF cloud + decoys only; actives transformed using this scaler (no leakage).

### 7.2 Deduplication

**SMILES-Based**: Canonical SMILES strings used for compound uniqueness.

**Policy**: One entry per unique SMILES; for duplicates, median affinity across all targets retained.

**Rationale**: Median aggregation is robust to outlier measurements and aligns with ChEMBL best practices.

### 7.3 Overlap Removal

**MF Cloud ∩ Decoys**: Any compound appearing in both is removed from decoys.

**MF Cloud ∩ Actives**: Any compound appearing in both is removed from MF cloud (target-preserving exclusion).

**Decoys ∩ Actives**: Any compound appearing in both is removed from decoys.

**Rationale**: Ensures clean separation between reference set, evaluation set, and background distribution.

### 7.4 Reproducibility

**Random Seeds**: Multiple independent runs (typically 5 replicates) for stochastic methods (UMAP).

**PCA**: Deterministic; single run sufficient.

**UMAP Parallelism**: `random_state=None` enables multi-threaded optimization (10× speedup); introduces minor stochastic variability across runs.

---

## 8. Computational Considerations

### 8.1 Scalability

**Dataset Size**: Method validated on datasets of ~1.5 million compounds.

**Memory Requirements**:
- Features representation: ~30-50 GB RAM for full pipeline
- Fingerprints representation: ~10-20 GB RAM
- Optimization: Selective dtype specification and Parquet caching reduce memory by 10×

**Runtime** (1.5M compounds on 64-core HPC node):
- PCA: ~5 minutes total (feature calculation + DR + scoring)
- UMAP: ~10-30 minutes total (depending on n_neighbors)

### 8.2 Optimization Strategies

**Parquet Caching**: CSV datasets converted to Parquet format for 10-100× faster loading on subsequent runs.

**Selective Data Types**: Metadata columns stored as strings, numerical features as float32 or float64.

**Imputation Before Scaling**: Missing values filled once; scaler operates on complete data.

**Exact 1-NN Backend**: KDTree/BallTree for efficient nearest neighbor search (vs brute-force distance matrix).

---

## 9. Hyperparameter Sensitivity

### 9.1 PCA

**Dimensions**: Minimal effect; 2D, 5D, and 10D yield nearly identical enrichment (EF@1% = 57-58).

**Interpretation**: First 2 principal components capture majority of task-relevant variance.

### 9.2 UMAP

**n_neighbors (CRITICAL)**:
- Small (10-20): Best enrichment performance (EF@1% = 40-46)
- Medium (50-100): Moderate performance (EF@1% = 25-35)
- Large (500): Poor performance (EF@1% = 20)
- **Interpretation**: Small neighborhoods create tight clusters with high discriminative power; large neighborhoods dilute signal by including too many decoy-decoy relationships.

**min_dist (WEAK EFFECT)**:
- Range tested: 0.0 to 0.5
- Variation: ±2% in EF@1% across 3 orders of magnitude
- **Interpretation**: Packing tightness is nearly irrelevant for ranking performance.

**Dimensions**:
- UMAP improves with dimension: 2D (EF@1% = 39) → 5D (44) → 10D (46)
- Gap to PCA narrows with dimension but never closes (PCA remains 1.3× better at 10D)

### 9.3 Affinity Cutoff

**Research Question**: Does restricting MF cloud to high-potency ligands improve enrichment?

**Cutoffs Tested**: 100 nM, 1 μM, 10 μM, 100 μM (default)

**Status**: Under investigation in Phase 2 experiments.

---

## 10. Method Comparison to Related Approaches

### 10.1 vs Traditional Ligand-Based Screening

**Traditional**: Pairwise Tanimoto similarity to known actives (e.g., using ECFP4 fingerprints).

**MolFuSE**: Population-based similarity to molecular function cloud via dimensionality reduction.

**Advantage**: Leverages collective knowledge of entire protein family; less sensitive to choice of query ligand.

### 10.2 vs Proteochemometric Modeling

**Proteochemometric**: Predicts activity using protein-ligand descriptor pairs.

**MolFuSE**: Uses only molecular descriptors and molecular function annotations; no protein structure required.

**Advantage**: Applicable to targets with limited or no structural information.

### 10.3 vs Deep Learning Embeddings

**Deep Learning**: Neural network-based molecular representations (e.g., graph neural networks).

**MolFuSE**: Classical dimensionality reduction (PCA/UMAP) on handcrafted descriptors.

**Advantage**: Interpretable features, deterministic results (PCA), minimal training data requirements.

---

## 11. Strengths and Limitations

### 11.1 Strengths

1. **Knowledge Leveraging**: Exploits molecular function annotations to guide similarity space construction
2. **Scalability**: Handles millions of compounds efficiently
3. **Simplicity**: Transparent methodology with interpretable components (PCA especially)
4. **Data Efficiency**: Requires only bioactivity data (no protein structures)
5. **Generalizability**: Applicable to any molecular function class with sufficient bioactivity data

### 11.2 Limitations

1. **Molecular Function Dependency**: Requires well-annotated molecular function databases (ChEMBL)
2. **Descriptor Choice**: Performance depends on molecular representation (features >> fingerprints)
3. **Target Coverage**: Limited to proteins with established molecular function annotations
4. **No Mechanism Insight**: Distance-based scoring does not explain binding modes
5. **Hyperparameter Sensitivity**: UMAP requires careful tuning of n_neighbors

---

## 12. Future Directions

### 12.1 Method Enhancements

1. **Multi-task Learning**: Jointly optimize similarity space for multiple targets within same molecular function
2. **Active Learning**: Iteratively refine molecular function cloud with experimental feedback
3. **Hybrid Descriptors**: Combine features and fingerprints via ensemble methods
4. **Protein Embeddings**: Incorporate protein sequence/structure embeddings for proteochemometric fusion

### 12.2 Application Extensions

1. **Polypharmacology**: Identify multi-target ligands by combining molecular function clouds
2. **Scaffold Hopping**: Explore chemically diverse compounds with molecular function proximity
3. **ADMET Filtering**: Integrate pharmacokinetic property predictions into ranking
4. **Target Repurposing**: Identify new indications for existing drugs via molecular function transfer

---

## 13. Conclusion

MolFuSE represents a knowledge-based approach to virtual screening that bridges ligand-based and systems-level perspectives. By anchoring molecular similarity to experimentally validated bioactivity patterns across entire protein families, the method achieves robust enrichment of active compounds while maintaining computational efficiency and interpretability. The framework is particularly well-suited for early-stage drug discovery where target structural information is limited but bioactivity data is abundant.
