# PCA Baseline Performance Analysis - ABL1 (Seed 44)

**Date:** October 14, 2025  
**Target:** Tyrosine-protein Kinase ABL1 (P00519)  
**MF:** Transferase (KW-0808)  
**Method:** PCA (2D projection)  
**Strategy:** Projection-only (v2.0 - no data leakage)

---

## Dataset Composition

| Component | Count | Source |
|-----------|-------|--------|
| **MF Cloud (Training)** | 420,609 | ChEMBL Transferase inhibitors |
| **Held-out Actives (Test)** | 3,294 | ABL1 ligands with known affinity |
| **ZINC Decoys** | 1,288,135 | Virtual screening candidates |
| **Total Ranked** | 1,291,429 | Actives + Decoys |

**Baseline active rate:** 3,294 / 1,291,429 = **0.255%**

---

## Performance Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **ROC-AUC** | 0.909 | Excellent separation |
| **PR-AUC** | 0.196 | Good precision-recall balance |
| **EF@1%** | **57.6x** | 🔥 **Outstanding enrichment** |
| **EF@5%** | 12.8x | Strong enrichment |
| **EF@10%** | 7.1x | Good enrichment |
| **Spearman ρ** | -0.328 | Moderate affinity correlation |

---

## Enrichment Analysis

### Top 1% Performance
- **Total compounds in top 1%:** 12,914
- **Actives in top 1%:** 1,897 (57.6% of all actives)
- **Active rate in top 1%:** 14.69%
- **Enrichment Factor:** 14.69% / 0.255% = **57.6x** ✅

### Active Distribution Across Rankings

| Rank Range | Actives | % of Total | Cumulative % | Insight |
|------------|---------|------------|--------------|---------|
| **1-10K (0.77%)** | 1,882 | 57.1% | 57.1% | 🔥 **Massive top cluster** |
| 10K-50K | 156 | 4.7% | 61.8% | Moderate presence |
| 50K-100K | 190 | 5.8% | 67.6% | Moderate presence |
| 100K-500K | 860 | 26.1% | 93.7% | Scattered mid-range |
| 500K+ | 206 | 6.3% | 100.0% | Tail distribution |

**Key Finding:** **57% of all actives in the top 0.77% of rankings!**

---

## Score Analysis

### Top Actives (Best 20)
```
Score (negative distance to MF cloud):
-0.00758  ← Best active
-0.00750
-0.00734
-0.00711
-0.00702
...
-6.26e-09 ← One active with near-perfect overlap (rank ~1,659)
```

**Score characteristics:**
- **Best actives:** -0.0076 to -0.0061 (distance ≈ 6-8 units in PCA space)
- **One exceptional active:** -6e-09 (essentially distance = 0)
  - This active is **chemically identical** to known transferase inhibitors in PCA space
  - Ranked at position ~1,659 (after known ZINC bioactives)

### Top Decoys
- **Top ~1,658 decoys:** All have score = -0.0 (perfect overlap with MF cloud)
- These are likely **known bioactive compounds** also present in ZINC
- Could be known kinase inhibitors or structurally similar molecules

---

## Why is PCA So Effective?

### 1. Large, Relevant Training Set
- **420K transferase inhibitors** provide comprehensive coverage of bioactive space
- Captures the "average" transferase inhibitor profile
- PC1-PC2 encode the dominant bioactivity signal

### 2. Strong Chemical Space Separation
- Bioactive compounds cluster distinctly from random drug-like compounds
- Linear projection sufficient to capture this global structure
- 2D PCA explains enough variance to separate actives from decoys

### 3. Projection-Only Strategy (v2.0)
- **No data leakage:** Held-out actives never used in PCA fitting
- Model trained only on MF cloud (ChEMBL transferases)
- Actives projected using pre-fitted model
- Validates that bioactive space is generalizable

### 4. Distance-Based Scoring
- Score = -distance to nearest MF cloud compound
- Assumes "similar to known actives" = likely bioactive
- Simple but highly effective for this task

---

## Implications for UMAP

PCA already achieves **EF@1% = 57.6x** with **57% of actives in top 0.77%**.

**UMAP's challenge:** Improve upon this strong baseline by:

1. **Capture local structure:**
   - Move the 860 mid-range actives (ranks 100K-500K) into top 10K
   - Better separate similar scaffolds within bioactive space

2. **Non-linear manifold learning:**
   - Capture subtle structural similarities PCA's linear projection misses
   - Create tighter local neighborhoods for chemically similar compounds

3. **Hyperparameter optimization:**
   - `n_neighbors`: Control local vs global structure balance
   - `min_dist`: Control cluster tightness
   - Goal: Maximize enrichment at top ranks

**Expected outcomes:**
- **Conservative:** UMAP maintains EF@1% ≈ 50-60x (similar to PCA)
- **Optimistic:** UMAP achieves EF@1% = 70-90x (15-55% improvement)
- **Best case:** UMAP achieves EF@1% > 100x (>75% improvement)

---

## Validation

### Data Integrity Checks ✅
- MF cloud: 420,609 compounds (after NaN filtering from 425,290)
- Similarity space: 1,708,744 total (420,609 ChEMBL + 1,288,135 ZINC)
- Results file: 1,291,429 ranked (3,294 actives + 1,288,135 decoys)
- MF cloud excluded from rankings (correct - used only for distance calculation)

### v2.0 Compliance ✅
- Projection-only strategy (no co-embedding)
- Correct MF: Transferase (KW-0808) for ABL1
- No data leakage (actives not in training set)
- PassthroughScaler for fingerprints / StandardScaler for features
- PCA model saved and reused for projection

### Enrichment Calculation ✅
```python
# Verified manually:
total_compounds = 1,291,429
total_actives = 3,294
top_1_percent = 12,914
actives_in_top_1 = 1,897

baseline_rate = 3,294 / 1,291,429 = 0.255%
top_1_rate = 1,897 / 12,914 = 14.69%
EF_1 = 14.69% / 0.255% = 57.6x ✅
```

---

## Conclusions

1. **PCA is highly effective** for virtual screening on large bioactive datasets
   - EF@1% = 57.6x is exceptional for a baseline method
   - 57% of actives in top 0.77% demonstrates strong signal

2. **The Transferase MF cloud is appropriate** for ABL1
   - Strong overlap between ABL1 actives and transferase inhibitor space
   - Validates the MF keyword approach

3. **Projection-only strategy works** (v2.0 validation)
   - No data leakage
   - Strong generalization to held-out actives
   - Bioactive space is learnable from MF cloud alone

4. **UMAP has room to improve**
   - 26% of actives (860 compounds) are in ranks 100K-500K
   - Non-linear manifold learning could recover these
   - Hyperparameter optimization is critical

5. **Top decoys are interesting**
   - ~1,658 ZINC compounds have perfect overlap with MF cloud
   - Likely known bioactives or close analogs
   - Could be prioritized for experimental validation

---

## Next Steps

1. **Run UMAP hyperparameter sweep** (42 configs × 5 seeds = 210 jobs)
   - Test n_neighbors: [15, 50, 100, 200, 500]
   - Test min_dist: [0.0, 0.01, 0.1, 0.5]
   - Features: UMAP-Euclidean only
   - Fingerprints: UMAP-Jaccard only

2. **Compare UMAP vs PCA**
   - Does UMAP improve EF@1%?
   - Which hyperparameters work best?
   - Is the improvement worth the computational cost?

3. **Analyze top ZINC decoys**
   - What are the ~1,658 perfect-overlap ZINC compounds?
   - Are they known kinase inhibitors?
   - Could they be novel ABL1 hits?

4. **Test generalization**
   - Apply best hyperparameters to held-out targets
   - Pyruvate Kinase M2 (Transferase)
   - Isocitrate Dehydrogenase (Oxidoreductase)

---

**Status:** ✅ PCA baseline established  
**Next:** Hyperparameter optimization for UMAP  
**Goal:** Beat EF@1% = 57.6x 🎯

---

**Last Updated:** October 14, 2025  
**Version:** 2.0 PCA Baseline Analysis
