# Sparsity Analysis: Features vs Fingerprints Representations

**Date:** October 27, 2025  
**Status:** Planning Phase  
**Related:** Phase 1 screening experiments (molfuse v4)

---

## Scientific Rationale

### Core Hypothesis

The two molecular representations employed in this study—RDKit/Mordred physicochemical descriptors (features) and ECFP4 circular fingerprints—exhibit fundamentally different sparsity profiles that may mechanistically explain their differential performance in dimensionality reduction and virtual screening tasks observed in Phase 1.

### Key Research Questions

1. **What is the absolute sparsity of each representation?**
   - Overall zero-fraction across entire dataset
   - Distribution of sparsity across molecules (row-wise)
   - Distribution of sparsity across dimensions (column-wise)

2. **Is sparsity uniform or structured?**
   - Are certain features/bits consistently zero? (uninformative dimensions)
   - Are certain features/bits consistently non-zero? (universal descriptors)
   - Does sparsity correlate with chemical diversity?

3. **How does sparsity relate to information content?**
   - Variance-sparsity relationship (dense features more informative?)
   - Entropy profiles for binary fingerprints
   - Effective dimensionality after thresholding low-information dimensions

4. **Are sparsity patterns molecular function-specific?**
   - Do different MF categories (Transferase, Kinase, etc.) show different sparsity?
   - Can sparsity signatures distinguish between MF classes?
   - Does chemical homogeneity within an MF correlate with lower sparsity?

5. **How does sparsity affect dimensionality reduction preprocessing?**
   - Does PCA struggle with sparse data? (covariance matrix singularity)
   - Does UMAP's metric learning handle sparsity better?
   - Optimal preprocessing (feature selection, imputation) per representation?

6. **Does sparsity predict Phase 1 performance?**
   - Correlation between per-MF sparsity and EF@1% scores
   - Interaction effects: sparsity × DR method × dimension
   - Threshold effects: performance cliff at certain sparsity levels?

---

## Data Structure

### Input Files

```
datasets/molecular_function_features_fingerprints/
├── KW-XXXX_<MF_name>_affinity_extracted_features.csv
│   Columns: Compound ChEMBL ID, SMILES, Target ChEMBL ID, Target Name,
│            Activity Type, Standard Value (nM), target_chembl_id, accession,
│            DipoleMoment, ABC, ABCGG, nAcid, nBase, SpAbs_A, ... (~200 descriptors)
│
├── KW-XXXX_<MF_name>_affinity_extracted_fingerprints_ECFP4.csv
│   Columns: Compound ChEMBL ID, SMILES, Target ChEMBL ID, Target Name,
│            Activity Type, Standard Value (nM), target_chembl_id, accession,
│            Fingerprint (comma-separated string of 2048 binary bits)
│
└── zinc/
    ├── zinc_acquirable_extracted_features.csv
    └── zinc_acquirable_extracted_fingerprints_ECFP4.csv
```

### Representation Specifications

**Features (RDKit + Mordred descriptors):**
- **Dimensionality:** ~200 continuous features
- **Value range:** Mixed (counts, ratios, energies, topological indices)
- **Expected sparsity:** 10-30% (some molecules lack certain structural motifs)
- **Preprocessing in Phase 1:** StandardScaler (mean=0, std=1)

**Fingerprints (ECFP4):**
- **Dimensionality:** 2048 binary bits
- **Value range:** {0, 1}
- **Expected sparsity:** 90-95% (most bits zero for any given molecule)
- **Preprocessing in Phase 1:** No scaling (binary preserved)

---

## Proposed Sparsity Metrics

### A. Global Sparsity Metrics

1. **Overall Sparsity (%):**
   ```
   sparsity = (# zero entries) / (# total entries) × 100
   ```

2. **Row-wise Sparsity (per molecule):**
   - Mean, median, IQR of % zeros per molecule
   - Distribution visualization (violin plot or histogram)
   - Identifies molecules with unusually sparse/dense profiles

3. **Column-wise Sparsity (per feature/bit):**
   - Mean, median, IQR of % zeros per dimension
   - Identifies universally sparse dimensions (candidates for removal)
   - Identifies universally dense dimensions (potential universal descriptors)

4. **Effective Dimensionality:**
   - Count of dimensions with >5% non-zero values (information threshold)
   - Count of dimensions with variance >0.01 (variability threshold)
   - Reduction ratio: effective_dims / original_dims

### B. Information Content Metrics

5. **Variance per Dimension:**
   - For features: standard variance calculation
   - For fingerprints: Bernoulli variance p(1-p) where p=activation frequency

6. **Entropy per Dimension (fingerprints):**
   ```
   H = -p·log₂(p) - (1-p)·log₂(1-p)
   ```
   - Maximum entropy (H=1.0) when p=0.5 (maximally informative)
   - Minimum entropy (H≈0) when p≈0 or p≈1 (uninformative)

7. **Gini Coefficient (feature usage inequality):**
   - Measures how evenly information is distributed across dimensions
   - Gini=0: all dimensions equally informative
   - Gini=1: one dimension contains all information

8. **Correlation with Sparsity:**
   - Pearson/Spearman r between sparsity and variance per dimension
   - Test hypothesis: sparse features have low variance (uninformative)

### C. Dataset Comparison Metrics

9. **Information Density:**
   ```
   info_density = (# non-zero entries per molecule) / (# dimensions)
   ```
   - Higher density = more information per molecule
   - Compare: MF cloud, ZINC, Actives, Inactives

10. **Sparsity Asymmetry:**
    - Compare ZINC vs MF sparsity distributions (Mann-Whitney U test)
    - Hypothesis: ZINC (drug-like space) is less sparse than bioactive MF cloud

11. **Compressibility Ratio:**
    ```
    compression = (sparse matrix size) / (dense matrix size)
    ```
    - Practical metric for storage/computation efficiency
    - Justifies sparse matrix representations in future implementations

### D. Molecular Function Stratification

12. **Per-MF Sparsity Profiles:**
    - Compute all above metrics separately for each MF category
    - ANOVA/Kruskal-Wallis test for heterogeneity across MFs
    - Hypothesis: chemically homogeneous MFs (e.g., kinases) have lower sparsity

13. **MF Signature Dimensions:**
    - Identify features/bits with high activation in one MF, low in others
    - Potential for MF-specific feature engineering
    - Relevance for understanding why certain targets screen better

### E. Dimensionality Reduction Implications

14. **PCA Compatibility Score:**
    - Condition number of covariance matrix (singularity risk)
    - Explained variance ratio in first K components vs sparsity
    - Hypothesis: PCA struggles with ultra-sparse fingerprints

15. **UMAP Preprocessing Requirements:**
    - Does distance distribution skew with high sparsity?
    - Nearest neighbor graph connectivity vs sparsity
    - Optimal k-NN parameter adjustments for sparse data

### F. Co-Activation Patterns (Fingerprints Only)

16. **Bit Co-occurrence Matrix:**
    - Compute pairwise bit activation frequency
    - Identify "always-together" bit pairs (substructure motifs)
    - Effective dimensionality from bit redundancy

17. **Bit Clustering:**
    - Hierarchical clustering of bits by co-activation patterns
    - Identifies groups of redundant bits
    - Suggests optimal dimensionality for fingerprint compression

---

## Proposed Visualizations

### Figure 1: Sparsity Overview (2×2 Grid)
**Purpose:** High-level comparison of representations

- **Panel A:** Global sparsity bar chart
  - X-axis: {Features, Fingerprints}
  - Y-axis: Sparsity (%)
  - Error bars: 95% CI from bootstrap

- **Panel B:** Row-wise sparsity distributions
  - Violin plots: Features vs Fingerprints
  - Y-axis: % zeros per molecule
  - Overlay: median line and quartiles

- **Panel C:** Column-wise sparsity distributions
  - Histograms (overlaid, semi-transparent)
  - X-axis: % zeros per dimension
  - Y-axis: Count of dimensions

- **Panel D:** Variance vs Sparsity scatterplot
  - X-axis: Column sparsity (%)
  - Y-axis: Variance (log scale)
  - Color: representation type
  - Fit lines: Pearson correlation

### Figure 2: Dimension-Wise Heatmap
**Purpose:** Identify informative vs uninformative dimensions

- **Layout:** Heatmap with dimensions as rows, metrics as columns
  - Columns: Sparsity, Variance, Entropy (FP only), Rank
  - Sorted by: information content (descending)
  - Color scale: normalized per metric

- **Annotations:**
  - Top-10 most informative features/bits labeled
  - Bottom-10 least informative (candidates for removal)

### Figure 3: MF-Stratified Comparison
**Purpose:** Test MF-specific sparsity patterns

- **Layout:** 2×N grid (N = number of MF categories analyzed)
  - Top row: Features
  - Bottom row: Fingerprints
  - Each cell: box plot of information density per molecule

- **Statistics:** 
  - ANOVA p-value annotated
  - Post-hoc pairwise comparisons (if significant)

### Figure 4: ZINC vs MF Cloud CDF
**Purpose:** Assess dataset distributional differences

- **Layout:** 2 panels (features | fingerprints)
- **Each panel:**
  - X-axis: Row-wise sparsity (% zeros)
  - Y-axis: Cumulative probability
  - Curves: ZINC (one line), Multiple MF categories (multiple lines)
  - Legend: dataset names
  - Shaded regions: IQR for ZINC

### Figure 5: Bit Co-Activation Network (Fingerprints Only)
**Purpose:** Visualize fingerprint structure and redundancy

- **Graph representation:**
  - Nodes: Top-100 most frequently activated bits
  - Node size: Activation frequency
  - Edges: Co-activation frequency (threshold: >10% co-occurrence)
  - Edge weight: Line thickness proportional to co-occurrence rate
  - Layout: Force-directed (networkx spring layout)

- **Interpretation:**
  - Dense clusters = redundant bits (same substructure detected multiple ways)
  - Isolated nodes = unique structural features

### Figure 6: Phase 1 Performance Correlation
**Purpose:** Link sparsity to screening performance

- **Layout:** Scatterplot matrix (3×3)
  - Rows: PCA, UMAP (best), UMAP (avg)
  - Columns: dim=2, dim=5, dim=10
  - Each cell: MF-level sparsity (X) vs EF@1% (Y)
  - Point color: representation (blue=features, orange=fingerprints)
  - Fit lines: Linear regression with 95% CI

---

## Statistical Testing Plan

### Primary Tests

1. **Representation Comparison (Global):**
   - **Test:** Mann-Whitney U (non-parametric)
   - **Null hypothesis:** Features and fingerprints have equal sparsity distributions
   - **Alternative:** Two-sided
   - **Significance:** α = 0.05

2. **Dataset Comparison (ZINC vs MF):**
   - **Test:** Mann-Whitney U per representation
   - **Correction:** Bonferroni (2 tests: features, fingerprints)
   - **Effect size:** Cohen's d

3. **MF Heterogeneity:**
   - **Test:** Kruskal-Wallis H (non-parametric ANOVA)
   - **Null hypothesis:** All MF categories have equal sparsity
   - **Post-hoc:** Dunn's test with Bonferroni correction
   - **Minimum group size:** N ≥ 100 molecules per MF

4. **Variance-Sparsity Correlation:**
   - **Test:** Spearman rank correlation (handles non-normality)
   - **Per-dimension analysis:** Across all features/bits
   - **Expected:** ρ < 0 (negative correlation)

5. **Performance Correlation (Phase 1):**
   - **Test:** Pearson correlation (if normality holds) or Spearman
   - **Variables:** MF-level sparsity vs EF@1%
   - **Stratified by:** DR method × dimension
   - **Control variables:** MF diversity (Tanimoto), dataset size

### Confidence Intervals

- **Bootstrap resampling:** 1,000 iterations for all sparsity estimates
- **CI level:** 95% (two-tailed)
- **Method:** Bias-corrected and accelerated (BCa) bootstrap

### Multiple Testing Correction

- **Method:** Benjamini-Hochberg FDR control
- **Applied to:** All pairwise MF comparisons, dimension-wise tests
- **Threshold:** q = 0.05

---

## Implementation Plan

### Script: `scripts/analyze_representation_sparsity.py`

#### Class Structure

```python
class SparsityAnalyzer:
    """
    Comprehensive sparsity analysis for molecular representations.
    
    Attributes:
        data_dir (Path): Root directory containing feature/fingerprint CSVs
        output_dir (Path): Directory for results and plots
        mf_categories (List[str]): MF categories to analyze (or "all")
        sample_size (int): Max molecules per dataset (for memory efficiency)
        bootstrap_n (int): Number of bootstrap iterations
        random_seed (int): For reproducibility
    """
```

#### Core Methods

1. **Data Loading:**
   - `load_features(csv_path)` → (metadata_df, feature_matrix)
   - `load_fingerprints(csv_path)` → (metadata_df, fingerprint_matrix)
   - Handle chunked loading for large files (>1M molecules)
   - Validate data integrity (no NaNs in feature space)

2. **Metric Computation:**
   - `compute_global_sparsity(X)` → Dict[str, float]
   - `compute_row_sparsity(X)` → np.ndarray (per-molecule)
   - `compute_col_sparsity(X)` → np.ndarray (per-dimension)
   - `compute_variance_profile(X)` → np.ndarray
   - `compute_entropy_profile(X)` → np.ndarray (fingerprints only)
   - `compute_information_density(X)` → Dict[str, float]
   - `compute_effective_dimensionality(X, threshold)` → int

3. **Comparative Analysis:**
   - `compare_representations(feat_data, fp_data)` → pd.DataFrame
   - `stratify_by_mf(datasets)` → pd.DataFrame
   - `compare_zinc_vs_mf(X_zinc, X_mf_dict)` → Dict
   - `compute_coactivation_matrix(X_fp)` → np.ndarray

4. **Statistical Testing:**
   - `test_sparsity_difference(X1, X2)` → Dict (p-value, effect size)
   - `test_mf_heterogeneity(mf_groups)` → Dict (ANOVA results)
   - `bootstrap_ci(X, metric_func, n_iter)` → Tuple[float, float]

5. **Visualization:**
   - `plot_sparsity_overview(results)` → List[Path]
   - `plot_dimension_heatmap(per_dim_stats)` → Path
   - `plot_mf_comparison(mf_results)` → Path
   - `plot_cdf_comparison(datasets)` → Path
   - `plot_coactivation_network(coact_matrix)` → Path
   - `plot_phase1_correlation(sparsity_df, phase1_df)` → Path

6. **Reporting:**
   - `generate_summary_table()` → pd.DataFrame
   - `generate_markdown_report()` → Path
   - Auto-generate interpretations based on statistical tests

#### CLI Interface

```bash
python scripts/analyze_representation_sparsity.py \
    --data_dir datasets/molecular_function_features_fingerprints \
    --output_dir reporting/sparsity_analysis \
    --mf_categories all \
    --sample_size 50000 \
    --bootstrap_n 1000 \
    --min_molecules 100 \
    --random_seed 42 \
    --phase1_summary reporting/phase1_post_analysis/phase1_summary_grouped.csv
```

**Arguments:**
- `--data_dir`: Root directory for feature/fingerprint CSVs
- `--output_dir`: Where to save results
- `--mf_categories`: Comma-separated MF names or "all"
- `--sample_size`: Subsample large datasets (0 = no subsampling)
- `--bootstrap_n`: Bootstrap iterations for CI estimation
- `--min_molecules`: Skip MFs with fewer molecules
- `--random_seed`: For reproducibility
- `--phase1_summary`: (Optional) Path to Phase 1 results for correlation analysis

---

## Output Structure

```
reporting/sparsity_analysis/
├── summary_statistics.csv
│   Columns: representation, dataset, n_molecules, n_dimensions,
│            global_sparsity, global_sparsity_ci_low, global_sparsity_ci_high,
│            row_sparsity_mean, row_sparsity_median, row_sparsity_std,
│            col_sparsity_mean, col_sparsity_median, col_sparsity_std,
│            effective_dims, info_density_mean, variance_mean
│
├── per_dimension_statistics.csv
│   Columns: representation, dimension_id, dimension_name,
│            sparsity, variance, entropy, rank_by_info,
│            mf_specificity_score
│
├── mf_comparison.csv
│   Columns: mf_category, representation, n_molecules,
│            global_sparsity, info_density_mean, variance_mean
│
├── statistical_tests.csv
│   Columns: test_name, comparison, test_statistic, p_value,
│            effect_size, significance, interpretation
│
├── coactivation_matrix.npz  # Sparse matrix (fingerprints only)
│
├── phase1_correlation.csv  # If --phase1_summary provided
│   Columns: mf_category, representation, dr_method, dim,
│            sparsity, ef1_mean, correlation_r, correlation_p
│
├── plots/
│   ├── sparsity_overview.{png,pdf}
│   ├── dimension_heatmap_features.{png,pdf}
│   ├── dimension_heatmap_fingerprints.{png,pdf}
│   ├── mf_stratified_comparison.{png,pdf}
│   ├── zinc_vs_mf_cdf_features.{png,pdf}
│   ├── zinc_vs_mf_cdf_fingerprints.{png,pdf}
│   ├── bit_coactivation_network.{png,pdf}
│   └── phase1_performance_correlation.{png,pdf}
│
├── report.md  # Auto-generated markdown summary
└── analysis.log  # Full execution log
```

---

## Expected Findings (Hypotheses)

### Hypothesis 1: Fingerprints Are Ultra-Sparse

**Prediction:**
- ECFP4 global sparsity: 90-95%
- Features global sparsity: 10-30%

**Implication:**
- Fingerprints suffer from curse of dimensionality in Euclidean space
- Tanimoto (Jaccard) similarity is more appropriate metric for fingerprints
- Explains why PCA on fingerprints performs poorly (covariance matrix near-singular)

**Test:**
- Mann-Whitney U comparing row-wise sparsity distributions
- Expected: p < 0.001, large effect size (Cohen's d > 1.0)

---

### Hypothesis 2: Sparse Dimensions Are Uninformative

**Prediction:**
- Negative correlation between sparsity and variance: ρ < -0.5
- Sparse features/bits have low entropy (approaching 0)
- Top 20% of dimensions by information content account for >80% of variance (Pareto principle)

**Implication:**
- Dimensionality reduction is essential (many dimensions are noise)
- Feature selection could improve performance
- Some dimensions could be removed with minimal information loss

**Test:**
- Spearman correlation per representation
- Identify "elbow" in sorted variance profile

---

### Hypothesis 3: MF Categories Have Distinct Sparsity Profiles

**Prediction:**
- Chemically homogeneous MFs (e.g., kinases) have lower sparsity
- Chemically diverse MFs (e.g., receptors) have higher sparsity
- Kruskal-Wallis p < 0.05 for heterogeneity across MFs

**Implication:**
- Some targets are inherently easier to screen (less sparse = more signal)
- MF-specific preprocessing may improve performance
- Explains variability in Phase 1 EF@1% across targets

**Test:**
- ANOVA/Kruskal-Wallis across 10+ MF categories
- Post-hoc Dunn's test to identify which pairs differ

---

### Hypothesis 4: ZINC Is Less Sparse Than MF Cloud

**Prediction:**
- ZINC mean sparsity < MF mean sparsity (for both representations)
- ZINC variance < MF variance (more consistent "drug-like" properties)

**Implication:**
- Distribution shift between training space (MF) and test space (ZINC)
- May contribute to difficulty in virtual screening
- Preprocessing should account for this shift

**Test:**
- Mann-Whitney U: ZINC vs MF
- CDF comparison: shift in central tendency

---

### Hypothesis 5: Sparsity Predicts Phase 1 DR Performance

**Prediction:**
- Negative correlation between MF sparsity and EF@1%: r < -0.3
- Stronger correlation for fingerprints than features
- Interaction: UMAP more robust to sparsity than PCA

**Implication:**
- Sparsity is a mechanistic driver of DR method performance
- Could guide method selection per target
- Justifies representation-specific DR hyperparameter tuning

**Test:**
- Pearson/Spearman correlation: sparsity vs EF@1%
- Stratified by: representation × DR method × dimension
- Multiple regression: EF@1% ~ sparsity + diversity + dataset_size

---

### Hypothesis 6: Fingerprint Bits Exhibit Strong Co-Activation

**Prediction:**
- Top 100 bits have >50% pairwise co-activation with ≥10 other bits
- Hierarchical clustering reveals 5-10 bit "modules" (substructure families)
- Effective fingerprint dimensionality < 500 (after redundancy removal)

**Implication:**
- ECFP4 contains massive redundancy
- Dimensionality reduction is not just useful but necessary
- Custom fingerprint design could reduce sparsity and improve performance

**Test:**
- Compute pairwise Jaccard similarity for bit vectors
- Apply community detection (Louvain algorithm)
- Count connected components in co-activation network

---

## Integration with Phase 1 Results

### Correlation Analysis Plan

**Objective:** Determine if sparsity explains variance in EF@1% scores observed in Phase 1

**Data Sources:**
1. **Sparsity metrics:** From this analysis (per-MF, per-representation)
2. **Performance metrics:** From `reporting/phase1_post_analysis/phase1_summary_grouped.csv`

**Variables:**

| Variable | Type | Source |
|----------|------|--------|
| `ef1_mean` | Continuous (dependent) | Phase 1 summary |
| `sparsity` | Continuous (predictor) | Sparsity analysis |
| `representation` | Categorical (moderator) | Both |
| `dr_method` | Categorical (moderator) | Phase 1 |
| `dimension` | Categorical (moderator) | Phase 1 |
| `mf_category` | Categorical (grouping) | Both |
| `dataset_size` | Continuous (covariate) | Sparsity analysis |
| `chemical_diversity` | Continuous (covariate) | Compute from Tanimoto |

**Statistical Model:**

```
EF@1% ~ sparsity + representation + dr_method + dimension +
        sparsity:representation + sparsity:dr_method +
        dataset_size + chemical_diversity
```

**Expected Outcomes:**
- Main effect of sparsity: β < 0 (p < 0.05)
- Interaction: sparsity × representation (fingerprints more affected)
- Interaction: sparsity × dr_method (UMAP more robust)

---

## Timeline and Prioritization

### Phase 1: Core Metrics (Week 1)
- [ ] Implement data loading infrastructure
- [ ] Compute global, row-wise, column-wise sparsity
- [ ] Generate Figure 1 (Sparsity Overview)
- [ ] Statistical tests: representation comparison, ZINC vs MF

### Phase 2: Detailed Analysis (Week 2)
- [ ] Variance-sparsity correlation
- [ ] Effective dimensionality analysis
- [ ] MF stratification
- [ ] Generate Figures 2-4

### Phase 3: Advanced Metrics (Week 3)
- [ ] Fingerprint co-activation analysis
- [ ] Bit clustering and network visualization
- [ ] Generate Figure 5
- [ ] Auto-generated markdown report

### Phase 4: Phase 1 Integration (Week 4)
- [ ] Load Phase 1 performance data
- [ ] Sparsity-performance correlation analysis
- [ ] Multiple regression modeling
- [ ] Generate Figure 6
- [ ] Final report with interpretations

---

## Success Criteria

### Quantitative Outcomes

1. **Complete sparsity characterization** of both representations across all datasets
2. **Statistical significance** in representation comparison (p < 0.05)
3. **Correlation coefficient** |r| > 0.3 between sparsity and Phase 1 performance
4. **Identification** of ≥10 uninformative dimensions (candidates for removal)
5. **Detection** of ≥5 MF-specific sparsity signatures

### Qualitative Outcomes

1. **Mechanistic understanding** of why certain representation-DR combinations succeed
2. **Actionable recommendations** for preprocessing and feature engineering
3. **Publication-ready figures** suitable for methods section of manuscript
4. **Hypothesis generation** for Phase 2 experiments
5. **Integration** with existing Phase 1 analysis narrative

---

## References and Related Work

### Relevant Literature

1. **Sparsity in Molecular Descriptors:**
   - Todeschini & Consonni (2009). *Molecular Descriptors for Chemoinformatics*
   - Discusses inherent sparsity in topological indices

2. **Fingerprint Sparsity:**
   - Rogers & Hahn (2010). "Extended-Connectivity Fingerprints"
   - Reports typical ECFP4 sparsity of 90-95%

3. **Curse of Dimensionality in Cheminformatics:**
   - Wale et al. (2008). "Comparison of Descriptor Spaces for Chemical Compound Retrieval"
   - Demonstrates performance degradation with high-dimensional sparse data

4. **Dimensionality Reduction on Sparse Data:**
   - McInnes & Healy (2018). "UMAP: Uniform Manifold Approximation and Projection"
   - Section on handling sparse input data

### Internal References

- **Phase 1 Results:** `PLANNING.md`, `LAB_BOOK.md`
- **Visualization Updates:** `PHASE1_VISUALIZATION_IMPROVEMENTS.md`
- **Pipeline Documentation:** `README_V4_MOLFUSE.md`

---

## Future Directions

### Immediate Next Steps

1. **Implement prototype** (global metrics only) to validate approach
2. **Test on single MF** (Transferase) before scaling to all categories
3. **Benchmark runtime** to ensure tractability for large datasets

### Long-Term Extensions

1. **Sparsity-Aware Preprocessing:**
   - Implement feature selection based on sparsity-variance trade-off
   - Custom imputation strategies for sparse features
   - Sparse matrix implementations for computational efficiency

2. **Representation Engineering:**
   - Design custom fingerprints with lower sparsity (e.g., reduced bit count)
   - Test alternative feature sets (e.g., E3FP, MACCS keys)
   - Explore learned representations (autoencoders)

3. **Phase 2 Integration:**
   - Use sparsity profiles to guide Phase 2 target selection
   - Stratify Phase 2 experiments by sparsity regime (low/medium/high)
   - Test sparsity-adaptive hyperparameter selection

4. **Mechanistic Modeling:**
   - Mathematical analysis: sparsity → distance distribution → DR performance
   - Simulation studies: synthetic sparse data with known ground truth
   - Theory: optimal sparsity for different DR methods

---

## Appendix: Technical Considerations

### Memory Management

- **Large datasets** (ZINC: ~2.5M molecules) require chunked processing
- **Strategy:** Process in 100k-molecule chunks, aggregate statistics
- **Sparse matrix storage:** Use `scipy.sparse.csr_matrix` for fingerprints

### Numerical Stability

- **Variance computation:** Use Welford's online algorithm to avoid catastrophic cancellation
- **Correlation:** Verify assumptions (linearity, normality) before Pearson; default to Spearman

### Reproducibility

- **Random seed:** Set for all sampling operations
- **Version control:** Document package versions (numpy, scipy, pandas)
- **Data provenance:** SHA256 checksums for input CSV files

### Computational Efficiency

- **Parallelization:** Use `joblib` for embarrassingly parallel operations (per-MF analysis)
- **Caching:** Store intermediate results (per-MF sparsity metrics) to avoid recomputation
- **Profiling:** Use `cProfile` to identify bottlenecks

---

**End of Planning Document**

*This document will be updated as the analysis progresses. Findings and interpretations will be added upon completion of each phase.*
