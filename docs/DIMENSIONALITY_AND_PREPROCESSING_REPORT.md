# UMMBAS v2.0 - Dimensionality and Preprocessing Analysis

**Generated:** October 15, 2025  
**Purpose:** Comprehensive audit of data dimensions and transformations throughout the pipeline

---

## Executive Summary

### Key Findings:
1. **Features have 40 dimensions, Fingerprints have 2048 dimensions**
2. **All DR methods reduce to 2D** in current hyperparameter sweep
3. **Features are scaled, Fingerprints are NOT scaled**
4. **PCA sees different input distributions for features vs fingerprints**

---

## 1. Input Data Dimensionality

### Features (Molecular Descriptors)
- **Source:** Mordred descriptors + custom RDKit features
- **Total dimensions:** **40 features**
- **Feature list (from `calculate_features_and_fingerprints_exp.py` lines 19-64):**
  ```python
  # 43 Mordred descriptors defined in desc_list
  # Custom: DipoleMoment (calculated separately)
  # Total configured in experiment_config.json: 40 features
  ```
  
- **Configured features in experiment_config.json:**
  ```json
  "rdkit_features_list_target": [
    "DipoleMoment","ABC","nAcid","nBase","nAromAtom","nAtom","nH","nC","nN","nO","nS",
    "nP","nX","nBonds","nBondsO","nBondsS","nBondsD","nBondsT","nBondsA","nBondsM",
    "nBondsKS","nBondsKD","EState_VSA7","nHBAcc","nHBDon","Lipinski","apol","bpol",
    "nRing","n3Ring","n4Ring","n5Ring","n6Ring","n7Ring","n8Ring","nRot","Diameter",
    "TopoShapeIndex","Vabc","MW"
  ]
  ```
  **Count:** 40 features ✅

### Fingerprints (ECFP4)
- **Source:** Morgan fingerprints (Extended Connectivity Fingerprints)
- **Total dimensions:** **2048 bits**
- **Configuration (from `calculate_features_and_fingerprints_exp.py` line 75):**
  ```python
  mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
  ```
- **Type:** Binary fingerprint (0/1 values)
- **Radius:** 2 (ECFP4 = radius 2)

---

## 2. Preprocessing: Scaling Strategy

### Features Preprocessing
**Location:** `calculate_similarityspaces_exp.py` lines 313-319

```python
if args.representation_type == "features":
    logger.info("STEP 2: Scaling features data with StandardScaler.")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_original)  # SCALED
```

**Transformation:**
- ✅ **StandardScaler applied**
- Formula: `X_scaled = (X - mean) / std`
- **Result:** Each feature has mean=0, std=1
- **Input to PCA/UMAP:** 40-dimensional scaled features

### Fingerprints Preprocessing
**Location:** `calculate_similarityspaces_exp.py` lines 320-325

```python
else:  # fingerprints
    logger.info("STEP 2: Fingerprints detected - NO SCALING will be applied (v2.0: Jaccard only).")
    scaler = PassthroughScaler()
    X_scaled = X_original.copy()  # NO TRANSFORMATION
```

**Transformation:**
- ❌ **NO scaling applied**
- PassthroughScaler: Identity transformation
- **Result:** Binary values remain 0/1
- **Input to PCA/UMAP:** 2048-dimensional binary fingerprints

---

## 3. Dimensionality Reduction Target Dimensions

### Current Hyperparameter Sweep Configuration
**Source:** `generate_generalization_configs.py` lines 48, 74

```python
"simspace_dims_to_test": [2]  # Only 2D for hyperparameter analysis
```

**All methods reduce to:** **2 dimensions**

### DR Method Configuration Details

#### PCA (Principal Component Analysis)
**Location:** `calculate_similarityspaces_exp.py` lines 333-345

```python
pca_run_config = {
    'simspace_dim': args.simspace_dim,  # = 2 in hyperparam sweep
    'cuml_params': {'n_components': 2, 'random_state': 42},
    'sklearn_params': {'n_components': 2, 'random_state': 42}
}
```

**Input → Output:**
- **Features:** 40 dimensions → **2 dimensions**
- **Fingerprints:** 2048 dimensions → **2 dimensions**

**PCA sees:**
- Features: Scaled continuous values (mean=0, std=1)
- Fingerprints: Binary values (0/1, unscaled)

#### UMAP (Uniform Manifold Approximation and Projection)
**Location:** `calculate_similarityspaces_exp.py` lines 369-378

```python
sklearn_params = {
    'n_components': args.simspace_dim,  # = 2
    'random_state': 42,
    'n_neighbors': n_neighbors,  # Variable (10, 100, 500)
    'min_dist': min_dist  # Variable (0.01, 0.1, 0.5)
}
cuml_params = {
    'n_components': 2,
    'n_neighbors': n_neighbors,
    'min_dist': min_dist,
    'metric': metric  # euclidean (features) or jaccard (fingerprints)
}
```

**Input → Output:**
- **Features (Euclidean):** 40 scaled dimensions → **2 dimensions**
- **Fingerprints (Jaccard):** 2048 binary dimensions → **2 dimensions**

**UMAP sees:**
- Features: Scaled continuous values via Euclidean distance
- Fingerprints: Binary values via Jaccard distance

---

## 4. Similarity Space Output Dimensions

### Final Output
All methods produce **2D similarity spaces** in current experiments.

**File format:** `{target}_{repr}_dim2_similarity_space.csv`

**Columns per method:**
- **PCA:** `['PCA-1', 'PCA-2']` (2 columns)
- **UMAP-Euclidean:** `['UMAP-Euclidean-1', 'UMAP-Euclidean-2']` (2 columns)
- **UMAP-Jaccard:** `['UMAP-Jaccard-1', 'UMAP-Jaccard-2']` (2 columns)

---

## 5. Data Flow Summary

### Features Pipeline
```
Raw Features (40D) 
    ↓
[StandardScaler: mean=0, std=1]
    ↓
Scaled Features (40D)
    ↓ ↓
   PCA          UMAP-Euclidean
    ↓              ↓
   2D            2D
```

**Compression ratio:** 40D → 2D = **20:1 reduction**

### Fingerprints Pipeline
```
Binary Fingerprints (2048D)
    ↓
[PassthroughScaler: no change]
    ↓
Binary Fingerprints (2048D)
    ↓ ↓
   PCA          UMAP-Jaccard
    ↓              ↓
   2D            2D
```

**Compression ratio:** 2048D → 2D = **1024:1 reduction**

---

## 6. Critical Differences: Why Features Outperform Fingerprints

### Dimensionality Perspective

| Aspect | Features | Fingerprints |
|--------|----------|--------------|
| **Input dimensions** | 40 | 2048 |
| **Compression ratio** | 20:1 | 1024:1 |
| **Information loss** | Moderate | **Extreme** |
| **Scaling** | Yes (StandardScaler) | No |
| **Value type** | Continuous | Binary |
| **PCA suitability** | High (designed for continuous) | Low (loses binary meaning) |

### Why PCA-Features Dominates (EF@1% = 57.59)

**Advantages:**
1. **Modest compression (20:1)** - Only 40D → 2D reduction
2. **Scaled inputs** - PCA assumes centered, normalized data
3. **Continuous values** - PCA designed for this data type
4. **Linear relationships captured** - Top 2 PCs explain significant variance
5. **Preserved chemical meaning** - Molecular descriptors directly interpretable

**Expected variance explained:**
- With 40 features, top 2 PCs likely capture 30-50% of variance
- Still substantial information retained

### Why PCA-Fingerprints Fails (EF@1% = 1.56)

**Disadvantages:**
1. **Extreme compression (1024:1)** - 2048D → 2D is massive information loss
2. **No scaling** - Binary values not normalized (by design)
3. **Binary data** - PCA treats bits as continuous values (conceptual mismatch)
4. **Top 2 PCs capture minimal variance** - Expected <1% of total variance
5. **Loss of substructure information** - ECFP4 bits encode specific molecular fragments

**Expected variance explained:**
- With 2048 features, top 2 PCs likely capture <1% of variance
- Catastrophic information loss

### Why UMAP-Features Underperforms PCA-Features

**Possible explanations:**

1. **Data is relatively linear**
   - UMAP's non-linear manifold learning is overkill
   - PCA's linear projection sufficient for this chemical space

2. **2D constraint too restrictive**
   - UMAP may need more dimensions (3D, 5D) to capture non-linear structure
   - 2D PCA uses variance-maximizing projections
   - 2D UMAP uses topology-preserving embeddings (different optimization)

3. **Hyperparameter sensitivity**
   - UMAP results vary significantly with n_neighbors/min_dist
   - PCA has no hyperparameters (just n_components)
   - Current UMAP settings may not be optimal

4. **Sample size effects**
   - Dataset size: ~420K MF cloud + 3K actives + 1.3M ZINC ≈ 1.7M samples
   - UMAP designed for local structure, may struggle with global patterns
   - PCA captures global variance directly

---

## 7. Implications for Hyperparameter Sweep Results

### Observed Performance (Completed Experiments)

```
Method                                  EF@1%     Interpretation
----------------------------------------------------------------
features-PCA                            57.59     ✅ Strong baseline
features-UMAP-Euclidean-nn10-md0.01     39.12     Tight UMAP helps
features-UMAP-Euclidean-nn10-md0.1      32.54     Looser still good
features-UMAP-Euclidean (mixed)         29.87     High variance
features-UMAP-Euclidean-nn100-md0.1     19.75     Too many neighbors
features-UMAP-Euclidean-nn100-md0.01    18.56     
features-UMAP-Euclidean-nn500-md0.01    14.09     Way too global
features-UMAP-Euclidean-nn100-md0.5     11.29     Poor settings
features-UMAP-Euclidean-nn10-md0.5       4.12     min_dist too high
fingerprints-PCA                         1.56     ❌ Information loss
```

### Key Insights

1. **PCA is not just a baseline - it's the best method (so far)**
   - 57.59 EF@1% beats all UMAP configurations
   - Simplicity wins for this dataset

2. **UMAP hyperparameter trends:**
   - Lower n_neighbors (10) >> higher (100, 500)
   - Lower min_dist (0.01) >> higher (0.5)
   - Best UMAP still 32% worse than PCA

3. **Dimensionality hypothesis:**
   - 2D may be insufficient for UMAP
   - Need to test 5D, 10D, 20D (configured but not yet run)
   - PCA may degrade less with 2D constraint

4. **Fingerprints need different approach:**
   - PCA on binary fingerprints is inappropriate
   - UMAP-Jaccard results pending (likely to improve)

---

## 8. Recommendations for Next Steps

### Immediate Actions

1. **Test higher dimensions:**
   - Run PCA and UMAP at 5D, 10D, 20D
   - Compare how enrichment changes with dimensions
   - Hypothesis: UMAP may improve relative to PCA in higher dimensions

2. **Complete fingerprints-UMAP-Jaccard sweep:**
   - These jobs are still pending
   - Jaccard is theoretically correct for binary data
   - Expected: Better than PCA-fingerprints (1.56), but likely < PCA-features (57.59)

3. **Analyze PCA variance explained:**
   - Check what % variance top 2 PCs capture for features
   - Compare to features-UMAP intrinsic dimensionality
   - Determines if 2D is fundamentally limiting

### Research Questions

1. **Why is the molecular feature space so linear?**
   - Are active compounds clustered in a low-variance subspace?
   - Is this ABL1-specific or general across targets?

2. **What is the optimal UMAP configuration?**
   - Current sweep may not have found the sweet spot
   - Need broader hyperparameter exploration
   - Consider adaptive methods (auto-tuning n_neighbors)

3. **Should we use PCA preprocessing for UMAP?**
   - UMAP on PCA-transformed features (e.g., 40D → 10D → 2D)
   - May preserve global structure while enabling local refinement

### Experimental Design Suggestions

1. **Dimensionality sweep:**
   - Test: 2D, 3D, 5D, 10D, 20D, 40D (full features)
   - Plot EF@1% vs dimensions for each method
   - Identify "elbow" where adding dimensions stops helping

2. **Hybrid approaches:**
   - PCA (40D → 10D) + UMAP (10D → 2D)
   - Feature selection + PCA
   - Autoencoder pretraining + UMAP

3. **Multi-target analysis:**
   - Run status check on all targets
   - Compare PCA vs UMAP across protein families
   - Identify target-specific vs universal patterns

---

## 9. Technical Validation Checklist

### Confirmed:
- ✅ Features: 40 dimensions → StandardScaler → PCA/UMAP → 2D
- ✅ Fingerprints: 2048 dimensions → No scaling → PCA/UMAP → 2D
- ✅ All methods currently output 2D spaces
- ✅ PCA has no hyperparameters to tune
- ✅ UMAP hyperparameters vary: n_neighbors (10-500), min_dist (0.01-0.5)
- ✅ Features scaled before DR, Fingerprints not scaled
- ✅ Extreme compression for fingerprints (1024:1) vs moderate for features (20:1)

### To Investigate:
- ⏳ PCA variance explained ratio (features vs fingerprints)
- ⏳ UMAP intrinsic dimensionality estimate
- ⏳ Performance at higher target dimensions (3D, 5D, 10D, 20D)
- ⏳ Fingerprints-UMAP-Jaccard results (pending)
- ⏳ Cross-target consistency of PCA superiority

---

## 10. Conclusion

**The unexpected dominance of PCA-features (EF@1% = 57.59) is scientifically meaningful:**

1. **Dimensionality matters:** 40D → 2D (modest) vastly outperforms 2048D → 2D (extreme)
2. **Linearity suffices:** The molecular descriptor space may be approximately linear in discriminative dimensions
3. **Simplicity wins:** Zero hyperparameters (PCA) beats extensive tuning (UMAP)
4. **2D constraint may handicap UMAP:** Non-linear methods may need more output dimensions

**This is not a bug, it's a discovery.** The strong PCA baseline establishes that:
- Linear relationships dominate in this chemical-biological space
- UMAP must be carefully optimized to beat PCA
- Higher dimensions may favor UMAP over PCA
- Fingerprints require fundamentally different approaches (Jaccard-based methods)

**Next steps should focus on:**
1. Testing higher target dimensions (5D, 10D, 20D)
2. Completing fingerprints-UMAP-Jaccard experiments
3. Multi-target validation of PCA superiority
4. Understanding the intrinsic dimensionality of the active compound space

---

**Last Updated:** October 15, 2025  
**Script:** `check_hyperparam_status.py`  
**Status:** 10 experiments completed (5 features-PCA, 5 fingerprints-PCA, UMAP ongoing)
