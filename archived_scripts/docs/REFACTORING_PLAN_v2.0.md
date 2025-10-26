# UMMBAS v2.0 Refactoring Plan

**Date Created:** October 14, 2025  
**Branch:** `2.0`  
**Status:** Planning Phase

---

## Executive Summary

This document outlines a comprehensive refactoring plan for the MolFuSE (Molecular Functional Similarity Explorer) screening pipeline. Critical errors were discovered in v1.0 that fundamentally compromise experimental validity:

1. **Molecular Function Misclassification**: Kinase targets (ABL1, Pyruvate Kinase M2) were incorrectly assigned to "Protein kinase inhibitor" (KW-0649, compounds that INHIBIT kinases) instead of "Transferase" (KW-0808, what kinases biochemically ARE)
2. **Data Leakage via Co-embedding**: t-SNE and co-embedding strategies include held-out actives during dimensionality reduction, violating the prospective screening simulation
3. **Fingerprint Processing Errors**: Binary fingerprints were incorrectly scaled for non-binary-native UMAP metrics, destroying chemical meaning
4. **Abstract Overclaims**: Manuscript claims "orphan target simulation" but uses targets with thousands of known ligands

Version 2.0 will correct these fundamental issues, streamline the methodology to projection-only approaches, and re-run all experiments with corrected molecular function assignments.

---

## Original Abstract (From v1.0)

> **Abstract**
>
> A significant challenge in ligand-based drug discovery is the scarcity of known active compounds for novel or understudied protein targets. To address this, we developed and validated a computational framework that leverages the target's broader Molecular Function (MF) to enrich the pool of relevant chemical matter for virtual screening. This study employs a rigorous "leave-one-target-out" experimental design to assess the hypothesis that a similarity space, built from a general MF chemical landscape, can effectively prioritize true active ligands for a specific, held-out protein. We systematically compared two molecular representations (physicochemical features and ECFP4 fingerprints), three dimensionality reduction (DR) algorithms (PCA, UMAP, t-SNE), and two distinct embedding strategies (Projection vs. Co-embedding). Performance was evaluated over five replicate runs by ranking held-out actives against a large decoy set using rank-based metrics, including ROC-AUC, PR-AUC, and Enrichment Factor at 1% (EF@1%).

**Key Claims Requiring Revision:**
- ~~"three dimensionality reduction (DR) algorithms (PCA, UMAP, t-SNE)"~~ → Remove t-SNE (co-embedding only, data leakage)
- ~~"two distinct embedding strategies (Projection vs. Co-embedding)"~~ → Use projection only (co-embedding creates data leakage)
- Implied "orphan targets" → Targets have 202-5,505 ligands, not truly orphan

---

## Critical Errors Identified in v1.0

### 1. Molecular Function Misclassification

**Error:** ABL1 (P00519) and Pyruvate Kinase M2 (P14618) assigned to "Protein kinase inhibitor" (KW-0649)

**Impact:**
- MF cloud size: **20 compounds** (3 non-kinase targets: CDK-interacting protein, Tribbles 1/2)
- Expected size: ABL1 ~**5,505 compounds**, Pyruvate Kinase M2 ~**202 compounds** (from Transferase KW-0808)
- Data leakage ratio: ABL1 has 3,331 held-out actives vs 20 MF cloud (165:1 ratio)
- Experiment is testing "can we find kinase inhibitors using non-kinase ligands?" (scientifically invalid)

**Root Cause:**
- **Semantic confusion**: "Protein kinase inhibitor" refers to COMPOUNDS that inhibit kinases, not the kinase enzymes themselves
- **Correct biochemical classification**: Kinases are TRANSFERASES (phosphoryl transfer enzymes)

**Correct Assignments (UniProt Keywords):**
- ABL1 (P00519): **Transferase** (KW-0808) - catalyzes phosphoryl transfer
- Pyruvate Kinase M2 (P14618): **Transferase** (KW-0808) - catalyzes phosphoryl transfer
- Isocitrate Dehydrogenase (O75874): **Oxidoreductase** (KW-0560) - ALREADY CORRECT

### 2. t-SNE Architectural Limitation (Data Leakage)

**Error:** t-SNE implementation ONLY supports co-embedding, not projection

**Evidence from code (`calculate_similarityspaces_exp.py` line 226):**
```python
X_coembed = np.vstack((X_pre_reduced, X_target_pre_reduced))
coembed_coords = model.fit_transform(X_coembed)
```

**Impact:**
- t-SNE always includes held-out actives during fitting
- Creates data leakage - model "sees" the answer before ranking
- Cannot simulate true prospective screening
- sklearn's t-SNE has no `.transform()` method for out-of-sample extension

**Solution:** **REMOVE t-SNE entirely from v2.0**

### 3. Fingerprint Processing Errors

**Error:** Binary fingerprints scaled for Euclidean/Cosine/Manhattan UMAP metrics

**Evidence from code:**
- Line 363: `X_target_binary` created with PassthroughScaler but never added to `X_data_dict`
- Line 432: `X_data_dict` contains only `'original'`, `'scaled'`, `'pca_reduced'` keys
- Line 254-256: Non-Jaccard/Hamming metrics use `'pca_reduced'` (scaled then PCA'd) data

**Impact:**
- Scaling binary fingerprints destroys chemical meaning
- UMAP-Euclidean/Cosine/Manhattan on fingerprints: EF@1% < 6 (near-random performance)
- UMAP-Jaccard correctly uses raw binary: EF@1% ~6 (still poor vs features 20-35)

**Solution:** **Keep ONLY UMAP-Jaccard for fingerprints**, remove all other fingerprint-based methods. Make sure Jaccard receives the fingerprints' binary data.

### 4. Co-embedding Data Leakage

**Error:** Co-embedding includes held-out actives during DR fitting

**Impact:**
- For ABL1: 3,331 held-out actives vastly outnumber 20 MF cloud compounds
- Held-out molecules influence the similarity space structure
- Violates prospective screening simulation principles
- Performance may be inflated by data leakage

**Solution:** **Use projection ONLY** - fit DR models on MF cloud + decoys, then transform (project) held-out actives

---

## v2.0 Methodology: Corrected Approach

### Target Proteins & Molecular Functions

| Target | UniProt ID | **CORRECT** Molecular Function | KW Code | Expected MF Cloud Size |
|--------|------------|--------------------------------|---------|----------------------|
| Tyrosine-protein Kinase ABL1 | P00519 | **Transferase** | KW-0808 | ~5,505 compounds |
| Pyruvate Kinase M2 | P14618 | **Transferase** | KW-0808 | ~202 compounds |
| Isocitrate Dehydrogenase NADP | O75874 | **Oxidoreductase** | KW-0560 | ~97,729 compounds (correct) |

### Molecular Representations

1. **Physicochemical Features** (39 descriptors):
   - RDKit/Mordred: DipoleMoment, ABC, nAcid, nBase, MW, nRot, Lipinski, etc.
   - Preprocessing: StandardScaler (zero mean, unit variance)
   - Use for: PCA projection, UMAP-Euclidean projection

2. **ECFP4 Fingerprints** (2048-bit binary):
   - Extended-Connectivity Fingerprints, radius=2
   - Preprocessing: **NONE** (raw binary data)
   - Use for: **UMAP-Jaccard projection ONLY**

### Dimensionality Reduction Methods (Projection Only)

| Method | Representation | Metric/Distance | Hyperparameters | Rationale |
|--------|---------------|-----------------|-----------------|-----------|
| **PCA** | Features | Linear (variance maximization) | n_components (2, 3, 5, 10, 20) | Baseline, interpretable, no tuning |
| **UMAP-Euclidean** | Features | Euclidean distance | n_neighbors, min_dist | Non-linear, captures local structure |
| **UMAP-Jaccard** | Fingerprints | Jaccard similarity | n_neighbors, min_dist | Binary-native, chemical substructure |

**REMOVED from v2.0:**
- ❌ t-SNE (all representations) - co-embedding only, data leakage
- ❌ Co-embedding strategies (all methods) - data leakage
- ❌ UMAP-Euclidean/Cosine/Manhattan on fingerprints - incorrect scaling
- ❌ PCA on fingerprints - incorrect scaling

### Embedding Strategy: Projection Only

**Workflow:**
1. Construct training data: MF Cloud (ChEMBL, ligands that target proteins with same Molecular Function, target excluded) + ZINC decoys
2. Preprocess: Apply StandardScaler to features, NO scaling for fingerprints
3. Fit DR model on training data only
4. Save fitted model
5. Transform held-out actives using saved model (projection)
6. Rank by minimum distance to MF Cloud in similarity space
7. Calculate EF@1%, ROC-AUC, PR-AUC

**Key principle:** Held-out actives NEVER influence the similarity space construction

### Baselines (NEW in v2.0)

Essential for validating that DR adds value:

1. **Random Shuffling Baseline**
   - Randomly shuffle held-out actives 1000 times
   - Calculate EF@1% distribution
   - Expected: Mean ~1.0, establishes random performance floor

2. **39D Feature Space Baseline**
   - Calculate minimum Euclidean distance in original 39D feature space (after scaling, no DR)
   - Tests: Does DR add value over raw features?

3. **Tanimoto Similarity Baseline**
   - Calculate max Tanimoto similarity to MF cloud using raw ECFP4 fingerprints
   - Tests: Do feature-based methods outperform simple fingerprint similarity?

4. **Chemical Novelty Analysis**
   - Calculate distribution of max Tanimoto similarity between held-out actives and MF cloud
   - Quantifies chemical novelty (low similarity → more impressive enrichment)

---

## Step-by-Step Refactoring Plan

### Phase 1: Data Collection & Molecular Function Correction

**Priority:** CRITICAL  
**Location:** HPC (heavy computation)

#### 1.1. Query UniProt for Correct Molecular Functions

**Objective:** Verify that ABL1 and Pyruvate Kinase M2 are classified as "Transferase" in UniProt

**Tasks:**
- [ ] Query UniProt REST API for P00519 (ABL1) keywords
- [ ] Query UniProt REST API for P14618 (Pyruvate Kinase M2) keywords
- [ ] Verify KW-0808 (Transferase) is present in keyword list
- [ ] Document findings in `datasets/protein_collection/uniprot_verification_v2.0.csv`

**Expected Output:**
```csv
UniProt ID,Protein Name,Molecular Function,KW Code,Verified Date
P00519,Tyrosine-protein Kinase ABL1,Transferase,KW-0808,2025-10-14
P14618,Pyruvate Kinase M2,Transferase,KW-0808,2025-10-14
O75874,Isocitrate Dehydrogenase NADP,Oxidoreductase,KW-0560,2025-10-14
```

#### 1.2. Extract ChEMBL Transferase Bioactivity Data

**Objective:** Create corrected MF cloud datasets for ABL1 and Pyruvate Kinase M2

**Input:** `datasets/chembl/chembl_35_affinity_data.csv` (full ChEMBL 35 bioactivity)

**Query Logic:**
```sql
SELECT DISTINCT compound_chembl_id, canonical_smiles, standard_value, standard_units, accession
FROM chembl_35_affinity_data
WHERE accession IN (
  SELECT DISTINCT accession 
  FROM chembl_35_target_mapping 
  WHERE uniprot_keywords LIKE '%KW-0808%'  -- Transferase
  AND organism = 'Homo sapiens'
)
AND standard_value <= 100000  -- nM affinity cutoff
AND accession NOT IN ('P00519', 'P14618')  -- Exclude ABL1 and Pyruvate Kinase M2
```

**Expected Output:**
- `datasets/molecular_function_affinity_data/KW-0808_Transferase_affinity.csv`
- Expected compounds: ~5,505 unique for ABL1 MF cloud, ~202 for Pyruvate Kinase M2

**Tasks:**
- [ ] Write SQL query to extract Transferase bioactivity from ChEMBL 35
- [ ] Filter for human proteins only
- [ ] Apply affinity cutoff (≤100,000 nM)
- [ ] Exclude held-out targets (P00519, P14618)
- [ ] Save to CSV with columns: Compound ChEMBL ID, SMILES, Standard Value (nM), Accession, Activity Type

#### 1.3. Calculate Features and Fingerprints for Corrected MF Clouds

**Objective:** Recalculate physicochemical features and ECFP4 fingerprints for Transferase MF cloud

**Script:** `core_scripts/calculate_features_and_fingerprints_exp.py`

**Input:**
- `datasets/molecular_function_affinity_data/KW-0808_Transferase_affinity.csv`

**Output:**
- `datasets/molecular_function_features_fingerprints/KW-0808_Transferase_affinity_extracted_features.csv`
- `datasets/molecular_function_features_fingerprints/KW-0808_Transferase_affinity_extracted_fingerprints_ECFP4.csv`

**Tasks:**
- [ ] Run feature calculation on HPC for Transferase MF cloud (~5,505 compounds)
- [ ] Run fingerprint calculation on HPC for Transferase MF cloud
- [ ] Verify no NaN values in 39 RDKit features
- [ ] Verify 2048-bit ECFP4 fingerprints correctly formatted (comma-separated integers)

**HPC Command:**
```bash
python core_scripts/calculate_features_and_fingerprints_exp.py \
  --input datasets/molecular_function_affinity_data/KW-0808_Transferase_affinity.csv \
  --output_features datasets/molecular_function_features_fingerprints/KW-0808_Transferase_affinity_extracted_features.csv \
  --output_fingerprints datasets/molecular_function_features_fingerprints/KW-0808_Transferase_affinity_extracted_fingerprints_ECFP4.csv \
  --n_jobs 32
```

---

### Phase 2: Code Refactoring

**Priority:** HIGH  
**Location:** Local (development branch `2.0`)

#### 2.1. Update Configuration Files

**Files to modify:**
- `experiment_config.json`
- All `generalization_configs/config_*.json` files

**Changes for ABL1 and Pyruvate Kinase M2:**
```json
{
  "molecular_function_canonical_name": "Transferase",  // Changed from "Protein kinase inhibitor"
  "molecular_function_filename_segment": "Transferase",  // Changed from "Protein_kinase_inhibitor"
  "molecular_function_display_name": "Transferase",  // Changed from "Protein kinase inhibitor"
  "molecular_function_kw_code": "KW-0808"  // Add explicit KW code
}
```

**Tasks:**
- [ ] Update `experiment_config.json` targets array
- [ ] Generate new config files for generalization experiments
- [ ] Add validation: Check that MF data files exist before running

#### 2.2. Remove t-SNE from `calculate_similarityspaces_exp.py`

**File:** `core_scripts/calculate_similarityspaces_exp.py`

**Deletions:**
1. Remove `run_tsne()` function (lines 220-242)
2. Remove t-SNE imports:
   ```python
   from sklearn.manifold import TSNE as sklearnTSNE
   from cuml import TSNE as cumlTSNE  # if CUML_AVAILABLE
   ```
3. Remove t-SNE argument parsing:
   ```python
   parser.add_argument("--dr_method_tsne", ...)
   parser.add_argument("--tsne_perplexity", ...)
   parser.add_argument("--tsne_pca_components", ...)
   ```
4. Remove t-SNE execution block (lines 413-427)
5. Remove t-SNE from `dr_configs` in main()

**Tasks:**
- [ ] Delete `run_tsne()` function
- [ ] Remove t-SNE imports
- [ ] Remove t-SNE CLI arguments
- [ ] Remove t-SNE execution logic from main()
- [ ] Update logging to reflect t-SNE removal
- [ ] Test that PCA and UMAP still run correctly

#### 2.3. Remove Co-embedding from All DR Methods

**File:** `core_scripts/calculate_similarityspaces_exp.py`

**Changes:**

1. **Remove co-embedding argument:**
   ```python
   # DELETE this line
   parser.add_argument("--run_coembedding_for_pca_umap", type=lambda x: (str(x).lower() == 'true'), default=False)
   ```

2. **Simplify `run_pca()` function:**
   ```python
   def run_pca(X_processed, df_info, config, out_paths, use_gpu):
       logger.info("--- Running PCA Projection ---")
       dr_cols = [f'PCA-{i+1}' for i in range(config['simspace_dim'])]
       
       model = cumlPCA(**config['cuml_params']) if use_gpu else sklearnPCA(**config['sklearn_params'])
       projection_coords = model.fit_transform(X_processed)
       save_model(model, out_paths['projection_model'])
       
       return pd.DataFrame(projection_coords, columns=dr_cols, index=df_info.index)
   ```
   - Remove all `if config['run_coembedding']` blocks
   - Remove `X_target_processed`, `df_target_info` parameters
   - Remove `'coembed_space'` from `out_paths`

3. **Simplify `run_umap_for_metric()` function:**
   - Same changes as PCA: remove co-embedding logic
   - Keep projection-only workflow

4. **Delete `construct_coembedded_dataframe()` helper function:**
   - No longer needed

**Tasks:**
- [ ] Remove `--run_coembedding_for_pca_umap` argument
- [ ] Simplify `run_pca()` to projection only
- [ ] Simplify `run_umap_for_metric()` to projection only
- [ ] Delete `construct_coembedded_dataframe()` function
- [ ] Remove co-embedding CSV outputs
- [ ] Update function signatures (remove target ligand parameters)

#### 2.4. Fix Fingerprint Processing (UMAP-Jaccard Only)

**File:** `core_scripts/calculate_similarityspaces_exp.py`

**Changes:**

1. **Remove non-Jaccard UMAP arguments for fingerprints:**
   ```python
   # DELETE these for fingerprint configs
   parser.add_argument("--umap_metric_to_run_euclidean", ...)
   parser.add_argument("--umap_metric_to_run_cosine", ...)
   parser.add_argument("--umap_metric_to_run_manhattan", ...)
   parser.add_argument("--umap_metric_to_run_hamming", ...)
   # KEEP this one
   parser.add_argument("--umap_metric_to_run_jaccard", ...)
   ```

2. **Simplify data preparation for fingerprints:**
   ```python
   if args.representation_type == "fingerprints":
       logger.info("STEP 2: Fingerprints detected - NO SCALING will be applied.")
       scaler = PassthroughScaler()
       X_scaled = X_original.copy()  # No transformation
       save_model(scaler, os.path.join(args.output_model_dir,
                  f"{args.target_id_name}_fingerprints_scaler.lzma"))
   ```

3. **Simplify `run_umap_for_metric()` for fingerprints:**
   ```python
   def run_umap_for_metric(metric, X_original, df_info, config, out_paths, use_gpu):
       if config['repr_type'] == "fingerprints":
           if metric.lower() != "jaccard":
               logger.error(f"Fingerprints only support Jaccard metric, not {metric}. Skipping.")
               return pd.DataFrame(columns=out_paths['dr_cols'])
           logger.info(f"UMAP-Jaccard on fingerprints: using RAW 2048-bit binary data.")
           X_main = X_original  # No scaling, no PCA
   ```

4. **Remove PCA pre-reduction for fingerprints:**
   - Delete lines 365-374 (fingerprint PCA pre-reduction block)
   - Jaccard uses raw binary data directly

**Tasks:**
- [ ] Remove non-Jaccard UMAP metric arguments for fingerprints
- [ ] Simplify fingerprint data preparation (no scaling)
- [ ] Remove PCA pre-reduction for fingerprints
- [ ] Add validation: If fingerprints + non-Jaccard metric → error
- [ ] Update `X_data_dict` construction (remove 'pca_reduced' for fingerprints)

#### 2.5. Simplify Target Ligand Preparation

**File:** `core_scripts/calculate_similarityspaces_exp.py`

**Changes:**

Since we're removing co-embedding, target ligands are ONLY needed for projection (transform), not fitting.

**Simplify `prepare_target_ligands()` function:**
```python
def prepare_target_ligands(path, repr_type, features_list, scaler):
    """
    Load and prepare held-out target ligands for PROJECTION ONLY.
    These ligands will be transformed using a pre-fitted DR model.
    """
    if not path or path.lower() == 'none' or not os.path.exists(path):
        logger.warning(f"Target ligands file not found: {path}")
        return None, None
    
    logger.info(f"Loading held-out actives for projection: {path}")
    df_target = pd.read_csv(path, low_memory=False)
    
    descriptor_cols = get_descriptor_columns(df_target, repr_type, features_list)
    df_target.dropna(subset=descriptor_cols, inplace=True)
    
    if df_target.empty:
        return None, None
    
    dtype_to_use = np.int8 if repr_type == "fingerprints" else np.float32
    X_target_original = df_target[descriptor_cols].values.astype(dtype_to_use)
    
    # Apply same preprocessing as training data
    X_target_processed = scaler.transform(X_target_original)
    
    logger.info(f"Loaded {len(df_target)} held-out actives for projection.")
    return df_target, X_target_processed
```

**Remove from main():**
- Delete duplicate `prepare_target_ligands()` call with `scaler_for_binary`
- Keep single call with appropriate scaler for representation type

**Tasks:**
- [ ] Simplify `prepare_target_ligands()` function signature
- [ ] Remove duplicate preparation calls
- [ ] Update docstring to clarify projection-only usage

#### 2.6. Update Configuration Schema

**File:** `experiment_config.json`

**Remove from `global_settings`:**
```json
"run_coembedding_for_pca_umap": true  // DELETE this line
```

**Remove from `dimensionality_reduction_methods`:**
```json
"tsne": {"short_name": "t-SNE", "perplexity": 30, "allow_coembedding": true}  // DELETE
```

**Update UMAP methods:**
```json
"dimensionality_reduction_methods": {
  "pca": {"short_name": "PCA"},  // Remove "allow_coembedding"
  "umap_euclidean": {"short_name": "UMAP-Euclidean", "metric": "euclidean"},  // Remove "allow_coembedding"
  "umap_jaccard": {"short_name": "UMAP-Jaccard", "metric": "jaccard"}  // Only for fingerprints
}
```

**Tasks:**
- [ ] Remove t-SNE from DR methods
- [ ] Remove `run_coembedding_for_pca_umap` flag
- [ ] Remove `allow_coembedding` from all DR method definitions
- [ ] Update JSON schema documentation

---

### Phase 3: New Baseline Implementations

**Priority:** HIGH  
**Location:** Local → HPC

#### 3.1. Random Shuffling Baseline

**Create:** `baselines/baseline_random_ranking.py`

**Functionality:**
- Load held-out actives and decoys for each target
- Randomly shuffle combined list 1000 times
- Calculate EF@1% for each shuffle
- Report: Mean, Std, 95% CI, Min, Max
- Expected: Mean ~1.0 (random performance)

**Output:** `baselines/results/random_baseline_EF1pct_distribution.csv`

**Pseudocode:**
```python
def random_baseline(target_actives, decoys, n_iterations=1000):
    combined = np.concatenate([target_actives, decoys])
    labels = np.array([1]*len(target_actives) + [0]*len(decoys))
    
    ef1_scores = []
    for _ in range(n_iterations):
        shuffled_indices = np.random.permutation(len(combined))
        ef1 = calculate_ef_at_1_percent(labels[shuffled_indices])
        ef1_scores.append(ef1)
    
    return {
        'mean': np.mean(ef1_scores),
        'std': np.std(ef1_scores),
        'ci_95': np.percentile(ef1_scores, [2.5, 97.5])
    }
```

**Tasks:**
- [ ] Implement `baseline_random_ranking.py`
- [ ] Test on ABL1 dataset
- [ ] Run on HPC for all 3 targets × 5 seeds = 15 experiments
- [ ] Generate distribution plots (histogram of EF@1%)
- [ ] Save results to CSV

#### 3.2. 39D Feature Space Baseline

**Create:** `baselines/baseline_feature_space_distance.py`

**Functionality:**
- Load MF cloud features (StandardScaler applied)
- Load held-out actives features (StandardScaler.transform)
- Calculate minimum Euclidean distance from each active to any MF cloud member
- Rank actives + decoys by distance (ascending = most similar first)
- Calculate EF@1%, ROC-AUC, PR-AUC
- Tests: Does DR add value over raw 39D features?

**Output:** `baselines/results/feature_space_baseline_results.csv`

**Pseudocode:**
```python
def feature_space_baseline(mf_cloud_features, held_out_features, decoy_features):
    # All already scaled with StandardScaler
    
    # Calculate minimum distance to MF cloud for actives
    active_distances = np.min(cdist(held_out_features, mf_cloud_features, metric='euclidean'), axis=1)
    
    # Calculate minimum distance to MF cloud for decoys
    decoy_distances = np.min(cdist(decoy_features, mf_cloud_features, metric='euclidean'), axis=1)
    
    # Rank by distance (lower = more similar)
    labels = np.array([1]*len(active_distances) + [0]*len(decoy_distances))
    distances = np.concatenate([active_distances, decoy_distances])
    
    sorted_indices = np.argsort(distances)  # Ascending
    sorted_labels = labels[sorted_indices]
    
    return {
        'EF@1%': calculate_ef_at_1_percent(sorted_labels),
        'ROC-AUC': roc_auc_score(labels, -distances),  # Negative because lower is better
        'PR-AUC': average_precision_score(labels, -distances)
    }
```

**Tasks:**
- [ ] Implement `baseline_feature_space_distance.py`
- [ ] Test on ABL1 dataset
- [ ] Run on HPC for all 3 targets × 5 seeds
- [ ] Compare to PCA and UMAP-Euclidean projection results
- [ ] Document findings: Does DR improve over raw features?

#### 3.3. Tanimoto Similarity Baseline

**Create:** `baselines/baseline_tanimoto_similarity.py`

**Functionality:**
- Load MF cloud fingerprints (raw 2048-bit binary)
- Load held-out actives fingerprints (raw binary)
- Calculate maximum Tanimoto similarity from each active to any MF cloud member
- Rank actives + decoys by similarity (descending = most similar first)
- Calculate EF@1%, ROC-AUC, PR-AUC
- Tests: Do feature-based methods outperform simple fingerprint similarity?

**Output:** `baselines/results/tanimoto_baseline_results.csv`

**Pseudocode:**
```python
def tanimoto_baseline(mf_cloud_fps, held_out_fps, decoy_fps):
    from rdkit import DataStructs
    
    # Calculate max Tanimoto similarity to MF cloud for actives
    active_similarities = []
    for fp in held_out_fps:
        max_sim = max([DataStructs.TanimotoSimilarity(fp, mf_fp) for mf_fp in mf_cloud_fps])
        active_similarities.append(max_sim)
    
    # Calculate max Tanimoto similarity to MF cloud for decoys
    decoy_similarities = []
    for fp in decoy_fps:
        max_sim = max([DataStructs.TanimotoSimilarity(fp, mf_fp) for mf_fp in mf_cloud_fps])
        decoy_similarities.append(max_sim)
    
    # Rank by similarity (higher = more similar)
    labels = np.array([1]*len(active_similarities) + [0]*len(decoy_similarities))
    similarities = np.concatenate([active_similarities, decoy_similarities])
    
    sorted_indices = np.argsort(-similarities)  # Descending
    sorted_labels = labels[sorted_indices]
    
    return {
        'EF@1%': calculate_ef_at_1_percent(sorted_labels),
        'ROC-AUC': roc_auc_score(labels, similarities),
        'PR-AUC': average_precision_score(labels, similarities)
    }
```

**Tasks:**
- [ ] Implement `baseline_tanimoto_similarity.py`
- [ ] Test on ABL1 dataset
- [ ] Run on HPC for all 3 targets × 5 seeds
- [ ] Compare to UMAP-Jaccard projection results
- [ ] Assess: Is DR on fingerprints worth the computational cost?

#### 3.4. Chemical Novelty Analysis

**Create:** `baselines/analyze_chemical_novelty.py`

**Functionality:**
- Calculate distribution of max Tanimoto similarity between held-out actives and MF cloud
- Report statistics: Mean, Median, Q1, Q3, Min, Max
- Interpretation:
  - High similarity (>0.7): Expected enrichment (similar to known actives)
  - Medium similarity (0.4-0.7): Moderate enrichment expected
  - Low similarity (<0.4): Impressive enrichment (finding novel chemotypes)

**Output:** `baselines/results/chemical_novelty_analysis.csv`

**Pseudocode:**
```python
def chemical_novelty_analysis(mf_cloud_fps, held_out_fps):
    novelty_scores = []
    for active_fp in held_out_fps:
        max_similarity = max([DataStructs.TanimotoSimilarity(active_fp, mf_fp) 
                             for mf_fp in mf_cloud_fps])
        novelty_scores.append(max_similarity)
    
    return {
        'mean_similarity': np.mean(novelty_scores),
        'median_similarity': np.median(novelty_scores),
        'q1': np.percentile(novelty_scores, 25),
        'q3': np.percentile(novelty_scores, 75),
        'min': np.min(novelty_scores),
        'max': np.max(novelty_scores)
    }
```

**Tasks:**
- [ ] Implement `analyze_chemical_novelty.py`
- [ ] Run for all 3 targets
- [ ] Create violin plots of Tanimoto distributions
- [ ] Correlate chemical novelty with enrichment performance
- [ ] Add to manuscript: Contextualize enrichment difficulty

---

### Phase 4: Hyperparameter Re-optimization

**Priority:** CRITICAL  
**Location:** HPC

#### 4.1. Re-run Hyperparameter Sweep for ABL1 (Transferase)

**Objective:** Find optimal UMAP hyperparameters with corrected MF cloud (~5,505 compounds vs 20)

**Script:** `analyze_hyperparams.py` (existing)

**Search Grid:**
```python
umap_hyperparams = {
    'n_neighbors': [5, 10, 15, 20, 30, 50, 75, 100],
    'min_dist': [0.0, 0.001, 0.01, 0.05, 0.1, 0.3, 0.5]
}

simspace_dims = [2, 3, 5, 10, 20]
```

**Configurations to Test:**
- PCA + Features (no hyperparams, just test dimensions)
- UMAP-Euclidean + Features (grid search n_neighbors × min_dist)
- UMAP-Jaccard + Fingerprints (grid search n_neighbors × min_dist)

**Evaluation:**
- 5 random seeds per configuration
- Metric: Mean EF@1% across 5 seeds
- Select best hyperparameters based on mean EF@1%

**Expected Runtime:** ~48 hours on HPC (8 n_neighbors × 7 min_dist × 5 dims × 2 methods × 5 seeds)

**Tasks:**
- [ ] Update `analyze_hyperparams.py` to use Transferase MF
- [ ] Remove t-SNE and co-embedding from search space
- [ ] Run hyperparameter sweep on HPC for ABL1
- [ ] Analyze results, select best n_neighbors and min_dist per method
- [ ] Document optimal hyperparameters in `hyperparam_configs/optimal_params_v2.0.json`

#### 4.2. Transfer Hyperparameters to Pyruvate Kinase M2 and Isocitrate Dehydrogenase

**Objective:** Apply ABL1-optimized hyperparameters to remaining targets

**Rationale:**
- Transferable hyperparameters reduce computational cost
- Focus on generalization, not per-target optimization

**Tasks:**
- [ ] Apply ABL1-optimal UMAP hyperparameters to Pyruvate Kinase M2 (Transferase)
- [ ] Apply ABL1-optimal UMAP hyperparameters to Isocitrate Dehydrogenase (Oxidoreductase)
- [ ] Run 5 seeds per target × 3 methods (PCA, UMAP-Euclidean, UMAP-Jaccard)
- [ ] Calculate mean EF@1%, ROC-AUC, PR-AUC per target

---

### Phase 5: Full Experimental Re-run

**Priority:** CRITICAL  
**Location:** HPC

#### 5.1. Run Complete Pipeline for All Targets

**Workflow:**

1. **Data Preparation:**
   - Use corrected MF clouds (Transferase for kinases, Oxidoreductase for dehydrogenase)
   - Load pre-calculated features and fingerprints
   - Load ZINC decoys

2. **Similarity Space Construction:**
   - Run PCA projection (features)
   - Run UMAP-Euclidean projection (features, optimized hyperparameters)
   - Run UMAP-Jaccard projection (fingerprints, optimized hyperparameters)
   - Save fitted models

3. **Held-out Active Projection:**
   - Load held-out actives for target
   - Transform using saved DR models (projection)
   - NO co-embedding

4. **Ranking and Evaluation:**
   - Calculate minimum distance to MF cloud for actives and decoys
   - Rank by distance (ascending)
   - Calculate EF@1%, ROC-AUC, PR-AUC

5. **Baselines:**
   - Run random shuffling baseline
   - Run 39D feature space baseline
   - Run Tanimoto similarity baseline
   - Run chemical novelty analysis

**Replicates:** 5 random seeds per target (for UMAP stochasticity)

**Total Experiments:** 3 targets × 3 DR methods × 5 seeds + baselines = 45 main experiments + 15 baseline runs

**Expected Runtime:** ~72 hours on HPC

**Tasks:**
- [ ] Prepare HPC batch scripts for full pipeline
- [ ] Run PCA + UMAP-Euclidean + UMAP-Jaccard for ABL1 (5 seeds)
- [ ] Run PCA + UMAP-Euclidean + UMAP-Jaccard for Pyruvate Kinase M2 (5 seeds)
- [ ] Run PCA + UMAP-Euclidean + UMAP-Jaccard for Isocitrate Dehydrogenase (5 seeds)
- [ ] Run all baseline methods for each target
- [ ] Verify outputs: similarity space CSVs, fitted models, ranking CSVs
- [ ] Download results to local for analysis

---

### Phase 6: Results Aggregation & Analysis

**Priority:** HIGH  
**Location:** Local

#### 6.1. Update Aggregation Scripts

**Files to Modify:**
- `aggregate_generalization_analysis.py`
- `aggregate_hyperparams.py`
- `aggregate_and_report.py`

**Filtering Logic:**

1. **Remove t-SNE results:**
   ```python
   df = df[~df['DR Method'].str.contains('t-SNE', case=False)]
   ```

2. **Remove co-embedding results:**
   ```python
   df = df[df['Strategy'] == 'projection']  # Keep only projection
   ```

3. **Remove incorrect fingerprint methods:**
   ```python
   # Keep only UMAP-Jaccard for fingerprints
   df = df[~(
       (df['Representation'] == 'fingerprints') & 
       (df['DR Method'].isin(['PCA', 'UMAP-Euclidean', 'UMAP-Cosine', 'UMAP-Manhattan']))
   )]
   ```

**Tasks:**
- [ ] Update filtering logic in aggregation scripts
- [ ] Add baseline results to aggregated tables
- [ ] Generate summary statistics: Mean ± Std EF@1% per method per target
- [ ] Create comparison tables: DR methods vs baselines

#### 6.2. Generate Comprehensive Results Table

**Create:** `results/v2.0_comprehensive_results_table.csv`

**Columns:**
- Target
- Method (PCA, UMAP-Euclidean, UMAP-Jaccard, 39D Baseline, Tanimoto Baseline, Random)
- Representation (Features, Fingerprints, N/A)
- EF@1% (Mean ± Std)
- ROC-AUC (Mean ± Std)
- PR-AUC (Mean ± Std)
- Chemical Novelty (Mean Tanimoto to MF cloud)
- Statistical Significance (vs Random, p-value)

**Example Row:**
```
ABL1, UMAP-Euclidean, Features, 28.3 ± 4.2, 0.87 ± 0.03, 0.42 ± 0.05, 0.51, p < 0.001
ABL1, Random Shuffle, N/A, 1.02 ± 0.31, 0.50 ± 0.02, 0.003 ± 0.001, N/A, N/A
```

**Tasks:**
- [ ] Aggregate results from all experiments
- [ ] Calculate mean and std across 5 seeds
- [ ] Perform statistical tests (t-test vs random baseline)
- [ ] Create publication-ready LaTeX table
- [ ] Generate plots: EF@1% comparison (methods × targets)

#### 6.3. Create Visualization Dashboard

**Create:** `visualizations/v2.0_results_dashboard.py`

**Plots to Generate:**

1. **EF@1% Comparison (Bar Plot):**
   - X-axis: Methods (PCA, UMAP-Euclidean, UMAP-Jaccard, Baselines)
   - Y-axis: EF@1%
   - Grouped by: Target
   - Error bars: Std across 5 seeds
   - Horizontal line: Random baseline (1.0)

2. **ROC Curves:**
   - One subplot per target
   - Overlay: All DR methods + baselines
   - Include AUC in legend

3. **Precision-Recall Curves:**
   - One subplot per target
   - Overlay: All DR methods + baselines
   - Include PR-AUC in legend

4. **Chemical Novelty Distributions:**
   - Violin plots of Tanimoto similarity (held-out actives vs MF cloud)
   - One plot per target
   - Annotate: Mean, Median

5. **Hyperparameter Heatmaps:**
   - X-axis: n_neighbors
   - Y-axis: min_dist
   - Color: Mean EF@1%
   - One heatmap per method (UMAP-Euclidean, UMAP-Jaccard)

**Tasks:**
- [ ] Implement visualization dashboard script
- [ ] Generate all plots
- [ ] Export to PDF for manuscript
- [ ] Create supplementary figure PDF

---

### Phase 7: Manuscript Updates

**Priority:** MEDIUM  
**Location:** Local

#### 7.1. Update Abstract

**Revised Abstract (v2.0):**

> **Abstract**
>
> A significant challenge in ligand-based drug discovery is the scarcity of known active compounds for novel or understudied protein targets. To address this, we developed and validated a computational framework that leverages the target's broader Molecular Function (MF) to enrich the pool of relevant chemical matter for virtual screening. This study employs a rigorous "leave-one-target-out" experimental design to assess the hypothesis that a similarity space, built from a general MF chemical landscape, can effectively prioritize true active ligands for a specific, held-out protein. We systematically compared two molecular representations (physicochemical features and ECFP4 fingerprints) and two dimensionality reduction (DR) algorithms (PCA and UMAP) using a **projection-based strategy** that simulates prospective screening scenarios. Performance was evaluated over five replicate runs by ranking held-out actives against a large decoy set using rank-based metrics, including ROC-AUC, PR-AUC, and Enrichment Factor at 1% (EF@1%). To contextualize performance, we implemented three baseline methods: random shuffling, 39D feature space distance, and Tanimoto fingerprint similarity. Our results demonstrate that **feature-based UMAP projection** achieves substantial enrichment (EF@1% 20-35) compared to random baseline (EF@1% ~1), while fingerprint-based methods show modest performance. This framework provides a **data-efficient approach** for expanding chemical space exploration when target-specific training data is limited.

**Key Changes:**
- ~~"three dimensionality reduction (DR) algorithms (PCA, UMAP, t-SNE)"~~ → **"two dimensionality reduction (DR) algorithms (PCA and UMAP)"**
- ~~"two distinct embedding strategies (Projection vs. Co-embedding)"~~ → **"projection-based strategy"**
- Added: **"To contextualize performance, we implemented three baseline methods"**
- Changed tone: **"simulated orphan target scenarios"** instead of implying true orphan targets
- Added: **"data-efficient approach"** emphasizes practicality

**Tasks:**
- [ ] Update abstract in manuscript
- [ ] Update graphical abstract (remove t-SNE, co-embedding)
- [ ] Update keywords: Remove "t-SNE", add "projection-based virtual screening"

#### 7.2. Update Methods Section

**Key Updates:**

1. **Molecular Function Assignment:**
   - Add paragraph explaining UniProt keyword-based MF classification
   - Clarify: **"Kinases are classified as Transferases (KW-0808)"**
   - Add table: Target → UniProt ID → Molecular Function → KW Code

2. **Remove t-SNE:**
   - Delete entire t-SNE subsection
   - Update DR methods table to include only PCA and UMAP

3. **Remove Co-embedding:**
   - Delete co-embedding subsection
   - Emphasize: **"All experiments used projection-only strategy to simulate prospective screening"**

4. **Correct Fingerprint Processing:**
   - Add: **"Fingerprints were NOT scaled, maintaining binary integrity"**
   - Add: **"UMAP-Jaccard was the only DR method applied to fingerprints, as it natively supports binary data"**

5. **Add Baseline Methods:**
   - New subsection: **"Baseline Comparisons"**
   - Describe: Random shuffling, 39D feature space, Tanimoto similarity, chemical novelty

**Tasks:**
- [ ] Rewrite Molecular Function subsection
- [ ] Delete t-SNE and co-embedding methods
- [ ] Add baseline methods subsection
- [ ] Update methods flowchart (remove co-embedding branch)
- [ ] Update supplementary methods

#### 7.3. Update Results Section

**Key Updates:**

1. **Hyperparameter Optimization:**
   - Report optimal UMAP hyperparameters for ABL1 (features and fingerprints)
   - Show heatmaps of EF@1% vs n_neighbors × min_dist

2. **Main Results Table:**
   - Include baselines in all comparisons
   - Format: Method | Representation | EF@1% (Mean ± Std) | ROC-AUC | PR-AUC

3. **Performance Comparison:**
   - **Feature-based methods outperform fingerprint-based** (EF@1% 20-35 vs ~6)
   - **UMAP-Euclidean > PCA** for features (non-linear structure captured)
   - **All DR methods >> Random baseline** (statistically significant)
   - **DR adds value over raw 39D feature space** (compare to 39D baseline)

4. **Chemical Novelty Analysis:**
   - Report Tanimoto distributions per target
   - Correlate novelty with enrichment difficulty
   - Example: **"Despite mean Tanimoto similarity of 0.51 to MF cloud, UMAP-Euclidean achieved EF@1% of 28.3 for ABL1"**

**Tasks:**
- [ ] Create main results table with baselines
- [ ] Generate all results figures
- [ ] Write results narrative emphasizing projection-only approach
- [ ] Add statistical tests (t-tests vs random, Wilcoxon rank-sum)
- [ ] Update supplementary results tables

#### 7.4. Update Discussion Section

**Key Additions:**

1. **Limitations:**
   - **"Targets tested are NOT truly orphan - they have 202-5,505 known ligands"**
   - **"Validation as 'simulated low-data scenarios' rather than orphan target discovery"**
   - **"Future work: Test on truly orphan targets (<10 known ligands)"**

2. **Methodological Insights:**
   - **"Projection-only strategy ensures no data leakage"**
   - **"Feature-based DR outperforms fingerprint-based DR for this task"**
   - **"t-SNE removed due to architectural incompatibility with projection"**

3. **Practical Implications:**
   - **"Framework applicable when target has <100 known ligands and MF has >1000"**
   - **"UMAP-Euclidean on 39D features provides best balance of performance and interpretability"**

**Tasks:**
- [ ] Add limitations subsection
- [ ] Revise discussion to reflect corrected methodology
- [ ] Add future directions: Truly orphan targets, active learning integration
- [ ] Update conclusion

---

### Phase 8: Repository Cleanup

**Priority:** LOW  
**Location:** Local (branch `2.0`)

#### 8.1. Archive v1.0 Experiments

**Objective:** Move outdated experiments to archive, keep repository clean

**Tasks:**
- [ ] Create `archive_v1.0/` directory
- [ ] Move old experiment results:
  - `experiment_workspace_generalization/` → `archive_v1.0/experiment_workspace_generalization_OLD/`
  - `final_report_generalization/` → `archive_v1.0/final_report_generalization_OLD/`
  - `hyperparameterization_report/` → `archive_v1.0/hyperparameterization_report_OLD/`
- [ ] Add README: `archive_v1.0/README.md` explaining why archived (incorrect MF, t-SNE, co-embedding)

#### 8.2. Update Documentation

**Files to Update:**
- `README.md` - Add v2.0 overview, list key changes
- `ANALYSIS_PIPELINE_OVERVIEW.md` - Remove t-SNE, co-embedding
- `GENERALIZATION_EXPERIMENT_README.md` - Update with corrected MF
- Create `CHANGELOG_v2.0.md` - Document all changes from v1.0

**Tasks:**
- [ ] Write comprehensive v2.0 README
- [ ] Update pipeline documentation
- [ ] Create changelog
- [ ] Add usage examples for new baseline scripts

#### 8.3. Clean Up Config Files

**Objective:** Remove outdated configs, organize v2.0 configs

**Tasks:**
- [ ] Move old configs: `dimensionality_configs/` → `archive_v1.0/dimensionality_configs_OLD/`
- [ ] Move old generalization configs: `generalization_configs/` → `archive_v1.0/generalization_configs_OLD/`
- [ ] Generate clean v2.0 configs: `generalization_configs_v2/`
- [ ] Validate all v2.0 configs: Correct MF, no t-SNE, no co-embedding

#### 8.4. Add Unit Tests

**Create:** `tests/test_v2_pipeline.py`

**Test Cases:**
1. **Molecular Function Validation:**
   - Assert ABL1 uses Transferase (KW-0808)
   - Assert Pyruvate Kinase M2 uses Transferase (KW-0808)
   - Assert Isocitrate Dehydrogenase uses Oxidoreductase (KW-0560)

2. **No t-SNE:**
   - Assert 'tsne' not in DR methods
   - Assert `run_tsne()` function does not exist

3. **No Co-embedding:**
   - Assert `run_coembedding_for_pca_umap` not in config
   - Assert no `*_COEMBED.csv` files generated

4. **Fingerprint Processing:**
   - Assert fingerprints only use UMAP-Jaccard
   - Assert no scaling applied to fingerprints (PassthroughScaler)

**Tasks:**
- [ ] Implement unit tests
- [ ] Add CI/CD integration (GitHub Actions)
- [ ] Run tests before merging to main

---

## Validation Checklist

Before finalizing v2.0, validate:

### Data Integrity
- [ ] Transferase MF cloud has ~5,505 compounds (not 20)
- [ ] Pyruvate Kinase M2 Transferase MF cloud has ~202 compounds (not 20)
- [ ] Isocitrate Dehydrogenase Oxidoreductase MF cloud has ~97,729 compounds (unchanged)
- [ ] No NaN values in 39D features
- [ ] Fingerprints are 2048-bit binary (all 0 or 1)

### Code Correctness
- [ ] No `run_tsne()` function in codebase
- [ ] No co-embedding logic in `run_pca()` or `run_umap_for_metric()`
- [ ] Fingerprints NOT scaled for UMAP-Jaccard
- [ ] `X_data_dict` does not contain scaled fingerprints for Jaccard metric
- [ ] All DR models save correctly and can be reloaded

### Experiment Validity
- [ ] Held-out actives NEVER used during DR fitting (projection only)
- [ ] Random baseline EF@1% ≈ 1.0 (confirms correct implementation)
- [ ] Feature-based methods: EF@1% 20-35 (expected performance)
- [ ] Fingerprint UMAP-Jaccard: EF@1% 5-10 (expected performance)
- [ ] 5 seeds per experiment for robustness

### Documentation Completeness
- [ ] Abstract updated (no t-SNE, no co-embedding, baselines mentioned)
- [ ] Methods section updated (correct MF, projection only, baselines)
- [ ] Results section includes baseline comparisons
- [ ] Discussion acknowledges limitations (not truly orphan targets)
- [ ] README.md explains v2.0 changes
- [ ] CHANGELOG_v2.0.md documents all modifications

---

## Timeline Estimate

| Phase | Description | Duration | Dependencies |
|-------|-------------|----------|--------------|
| 1 | Data Collection & MF Correction | 1 week | HPC access, ChEMBL 35 |
| 2 | Code Refactoring | 3 days | Phase 1 complete |
| 3 | Baseline Implementations | 2 days | Phase 2 complete |
| 4 | Hyperparameter Re-optimization | 2 days (HPC) | Phase 1-3 complete |
| 5 | Full Experimental Re-run | 3 days (HPC) | Phase 4 complete |
| 6 | Results Aggregation & Analysis | 3 days | Phase 5 complete |
| 7 | Manuscript Updates | 1 week | Phase 6 complete |
| 8 | Repository Cleanup | 2 days | Phase 7 complete |

**Total Estimated Duration:** ~3-4 weeks (HPC jobs run in parallel)

---

## Success Criteria

v2.0 will be considered successful when:

1. ✅ ABL1 and Pyruvate Kinase M2 use Transferase MF (~5,505 and ~202 compounds)
2. ✅ t-SNE completely removed from codebase
3. ✅ Co-embedding completely removed from codebase
4. ✅ Fingerprints ONLY use UMAP-Jaccard with raw binary data
5. ✅ Random baseline confirms correct implementation (EF@1% ≈ 1.0)
6. ✅ Feature-based UMAP projection achieves EF@1% 20-35
7. ✅ All experiments use projection-only strategy (no data leakage)
8. ✅ Baselines provide context for performance evaluation
9. ✅ Manuscript accurately describes methodology and limitations
10. ✅ Repository is clean, documented, and reproducible

---

## Contact & Collaboration

**Primary Investigator:** Alex  
**Branch:** `2.0`  
**Repository:** `UMMBAS_screening_experiments`  
**HPC:** [Specify HPC system]

For questions or issues during refactoring, document in:
- GitHub Issues (tag with `v2.0-refactoring`)
- Lab notebook entries
- Progress meetings

---

## Appendix: Key File Paths

### Data Files (Input)
```
datasets/molecular_function_affinity_data/KW-0808_Transferase_affinity.csv  [TO BE CREATED]
datasets/molecular_function_features_fingerprints/KW-0808_Transferase_affinity_extracted_features.csv  [TO BE CREATED]
datasets/molecular_function_features_fingerprints/KW-0808_Transferase_affinity_extracted_fingerprints_ECFP4.csv  [TO BE CREATED]
```

### Configuration Files
```
experiment_config.json  [TO BE UPDATED]
generalization_configs_v2/config_ABL1_*.json  [TO BE GENERATED]
generalization_configs_v2/config_PyruvateKinaseM2_*.json  [TO BE GENERATED]
```

### Core Scripts (Modified)
```
core_scripts/calculate_similarityspaces_exp.py  [MAJOR REFACTOR]
aggregate_generalization_analysis.py  [UPDATE FILTERING]
analyze_hyperparams.py  [UPDATE FOR TRANSFERASE]
```

### New Scripts
```
baselines/baseline_random_ranking.py  [TO BE CREATED]
baselines/baseline_feature_space_distance.py  [TO BE CREATED]
baselines/baseline_tanimoto_similarity.py  [TO BE CREATED]
baselines/analyze_chemical_novelty.py  [TO BE CREATED]
```

### Results (Output)
```
experiment_workspace_v2/run_seed*_ABL1_Transferase/  [TO BE GENERATED]
final_report_v2/comprehensive_results_table.csv  [TO BE GENERATED]
visualizations/v2.0_results_dashboard.pdf  [TO BE GENERATED]
```

---

**END OF REFACTORING PLAN**

*Last Updated: October 14, 2025*
*Version: 2.0-DRAFT*
