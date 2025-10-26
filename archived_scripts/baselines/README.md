# Baseline Methods for Virtual Screening (v2.0)

This directory contains baseline methods for contextualizing the performance of dimensionality reduction-based virtual screening approaches.

## Baseline Scripts

### 1. Random Shuffling Baseline (`baseline_random_ranking.py`)

**Purpose**: Establish the random performance floor by shuffling held-out actives and decoys multiple times.

**Expected Result**: Mean EF@1% ≈ 1.0 (confirms correct implementation)

**Usage**:
```bash
python baselines/baseline_random_ranking.py \
    --target_actives datasets/protein_collection/TyrosineProteinKinaseABL1_P00519_affinity.csv \
    --zinc_decoys datasets/zinc_data.csv \
    --n_iterations 1000 \
    --output_dir baselines/results/random_baseline/
```

**Outputs**:
- `{target}_random_baseline_results.json` - Full statistics
- `{target}_random_baseline_ef1_distribution.csv` - EF@1% for each iteration
- `{target}_random_baseline_summary.csv` - Summary statistics

---

### 2. 39D Feature Space Baseline (`baseline_feature_space_distance.py`)

**Purpose**: Test whether dimensionality reduction adds value over using raw 39D physicochemical features.

**Method**: Rank compounds by minimum Euclidean distance to MF cloud in original 39D feature space (after StandardScaler).

**Usage**:
```bash
python baselines/baseline_feature_space_distance.py \
    --mf_cloud_features datasets/molecular_function_features_fingerprints/KW-0808_Transferase_affinity_extracted_features.csv \
    --target_actives_features datasets/protein_collection/TyrosineProteinKinaseABL1_P00519_affinity_extracted_features.csv \
    --zinc_decoys_features datasets/molecular_function_features_fingerprints/zinc/zinc_acquirable_extracted_features.csv \
    --output_dir baselines/results/feature_space_baseline/
```

**Outputs**:
- `{target}_feature_space_baseline_results.json` - Full metrics
- `{target}_feature_space_baseline_rankings.csv` - Ranked compound list
- `{target}_feature_space_baseline_summary.csv` - Summary metrics

**Key Metrics**: EF@1%, ROC-AUC, PR-AUC

---

### 3. Tanimoto Similarity Baseline (`baseline_tanimoto_similarity.py`)

**Purpose**: Test whether feature-based methods outperform simple fingerprint similarity searching.

**Method**: Rank compounds by maximum Tanimoto similarity to MF cloud using ECFP4 fingerprints.

**Usage**:
```bash
python baselines/baseline_tanimoto_similarity.py \
    --mf_cloud_fingerprints datasets/molecular_function_features_fingerprints/KW-0808_Transferase_affinity_extracted_fingerprints_ECFP4.csv \
    --target_actives_fingerprints datasets/protein_collection/TyrosineProteinKinaseABL1_P00519_affinity_extracted_fingerprints_ECFP4.csv \
    --zinc_decoys_fingerprints datasets/molecular_function_features_fingerprints/zinc/zinc_acquirable_extracted_fingerprints_ECFP4.csv \
    --output_dir baselines/results/tanimoto_baseline/
```

**Outputs**:
- `{target}_tanimoto_baseline_results.json` - Full metrics
- `{target}_tanimoto_baseline_rankings.csv` - Ranked compound list
- `{target}_tanimoto_baseline_summary.csv` - Summary metrics

**Key Metrics**: EF@1%, ROC-AUC, PR-AUC, Mean/Median Tanimoto

---

### 4. Chemical Novelty Analysis (`analyze_chemical_novelty.py`)

**Purpose**: Quantify the chemical novelty of held-out actives relative to the MF cloud.

**Method**: Calculate distribution of maximum Tanimoto similarity between held-out actives and MF cloud.

**Interpretation**:
- **High similarity (>0.7)**: Actives similar to known ligands → enrichment expected
- **Medium similarity (0.4-0.7)**: Moderate chemical diversity → moderate enrichment expected
- **Low similarity (<0.4)**: Actives chemically distinct → impressive if enriched

**Usage**:
```bash
python baselines/analyze_chemical_novelty.py \
    --mf_cloud_fingerprints datasets/molecular_function_features_fingerprints/KW-0808_Transferase_affinity_extracted_fingerprints_ECFP4.csv \
    --target_actives_fingerprints datasets/protein_collection/TyrosineProteinKinaseABL1_P00519_affinity_extracted_fingerprints_ECFP4.csv \
    --output_dir baselines/results/chemical_novelty/
```

**Outputs**:
- `{target}_chemical_novelty_stats.json` - Full statistics
- `{target}_chemical_novelty_distribution.csv` - Per-compound similarities
- `{target}_chemical_novelty_summary.csv` - Summary statistics

---

## Running All Baselines

Example for ABL1 target with Transferase MF cloud:

```bash
# 1. Random shuffling baseline
python baselines/baseline_random_ranking.py \
    --target_actives datasets/protein_collection/TyrosineProteinKinaseABL1_P00519_affinity.csv \
    --zinc_decoys datasets/zinc_data.csv \
    --n_iterations 1000 \
    --output_dir baselines/results/random_baseline/

# 2. 39D feature space baseline
python baselines/baseline_feature_space_distance.py \
    --mf_cloud_features datasets/molecular_function_features_fingerprints/KW-0808_Transferase_affinity_extracted_features.csv \
    --target_actives_features datasets/protein_collection/TyrosineProteinKinaseABL1_P00519_affinity_extracted_features.csv \
    --zinc_decoys_features datasets/molecular_function_features_fingerprints/zinc/zinc_acquirable_extracted_features.csv \
    --output_dir baselines/results/feature_space_baseline/

# 3. Tanimoto similarity baseline
python baselines/baseline_tanimoto_similarity.py \
    --mf_cloud_fingerprints datasets/molecular_function_features_fingerprints/KW-0808_Transferase_affinity_extracted_fingerprints_ECFP4.csv \
    --target_actives_fingerprints datasets/protein_collection/TyrosineProteinKinaseABL1_P00519_affinity_extracted_fingerprints_ECFP4.csv \
    --zinc_decoys_fingerprints datasets/molecular_function_features_fingerprints/zinc/zinc_acquirable_extracted_fingerprints_ECFP4.csv \
    --output_dir baselines/results/tanimoto_baseline/

# 4. Chemical novelty analysis
python baselines/analyze_chemical_novelty.py \
    --mf_cloud_fingerprints datasets/molecular_function_features_fingerprints/KW-0808_Transferase_affinity_extracted_fingerprints_ECFP4.csv \
    --target_actives_fingerprints datasets/protein_collection/TyrosineProteinKinaseABL1_P00519_affinity_extracted_fingerprints_ECFP4.csv \
    --output_dir baselines/results/chemical_novelty/
```

---

## Results Directory Structure

```
baselines/results/
├── random_baseline/
│   ├── {target}_random_baseline_results.json
│   ├── {target}_random_baseline_ef1_distribution.csv
│   └── {target}_random_baseline_summary.csv
├── feature_space_baseline/
│   ├── {target}_feature_space_baseline_results.json
│   ├── {target}_feature_space_baseline_rankings.csv
│   └── {target}_feature_space_baseline_summary.csv
├── tanimoto_baseline/
│   ├── {target}_tanimoto_baseline_results.json
│   ├── {target}_tanimoto_baseline_rankings.csv
│   └── {target}_tanimoto_baseline_summary.csv
└── chemical_novelty/
    ├── {target}_chemical_novelty_stats.json
    ├── {target}_chemical_novelty_distribution.csv
    └── {target}_chemical_novelty_summary.csv
```

---

## Dependencies

All baseline scripts use standard scientific Python libraries:
- `numpy` - Numerical operations
- `pandas` - Data manipulation
- `scikit-learn` - StandardScaler, metrics (ROC-AUC, PR-AUC)
- `scipy` - Statistical functions, distance calculations

No RDKit required (fingerprints pre-calculated).

---

## v2.0 Notes

These baselines are essential for:
1. **Validating implementation** - Random baseline should give EF@1% ≈ 1.0
2. **Assessing DR value** - Does PCA/UMAP improve over 39D features?
3. **Comparing to fingerprints** - Do features outperform Tanimoto?
4. **Contextualizing performance** - How novel are the actives we're finding?

All baselines use the same evaluation metrics (EF@1%, ROC-AUC, PR-AUC) as the main DR methods for fair comparison.
