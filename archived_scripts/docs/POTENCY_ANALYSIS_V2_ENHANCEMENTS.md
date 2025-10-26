# Potency-Stratified Analysis - Enhanced Features (Oct 17, 2025)

## New Enhancements

Based on your request to compare PCA vs UMAP performance and analyze dimensionality impact across potency tiers, I've added comprehensive new analyses.

### ✨ NEW FEATURES

#### 1. PCA vs UMAP Comparison Across Potency Tiers

**New Plots:**
- `pca_vs_umap_by_tier.png` - Bar chart comparing mean EF@1% for PCA vs UMAP across High/Medium/Weak tiers
- `pca_vs_umap_distributions.png` - Box plots showing distribution of EF for each method×tier combination

**Key Questions Answered:**
- Is PCA or UMAP better at enriching **highly potent** compounds (0.1-100 nM)?
- Is PCA or UMAP better at enriching **medium potency** compounds (100-1,000 nM)?
- Is PCA or UMAP better at enriching **weak binders** (1,000-100,000 nM)?
- What is the performance variance for each method?

**Example Insights:**
```
High Potency (0.1-100 nM):
  PCA:  56.8 ± 12.3
  UMAP: 33.5 ± 8.7
  → PCA wins by 69.6%

Medium Potency (100-1,000 nM):
  PCA:  66.4 ± 15.2
  UMAP: 50.2 ± 11.4
  → PCA wins by 32.3%
```

#### 2. Dimensionality Impact Analysis

**New Plots:**
- `dimensionality_impact_by_tier.png` - Line plots showing EF vs dimensionality for each tier, comparing PCA and UMAP
- `method_dimension_tier_heatmap.png` - Heatmap showing performance of each Method×Dimension combination across tiers
- `optimal_dimensionality_by_tier.png` - Bar chart comparing performance across dimensions for each tier

**Key Questions Answered:**
- Does optimal dimensionality differ for High vs Medium vs Weak potency enrichment?
- Do PCA and UMAP have different optimal dimensionalities?
- Does 2D/5D/10D perform better for specific potency ranges?
- Is there a dimension×method interaction effect?

**Example Insights:**
```
High Potency:
  2D: 45.2 ± 8.1
  5D: 52.6 ± 10.3
  10D: 48.9 ± 9.7
  → Optimal: 5D (EF = 52.6)

Medium Potency:
  2D: 58.3 ± 12.4
  5D: 61.7 ± 14.8
  10D: 55.2 ± 11.9
  → Optimal: 5D (EF = 61.7)
```

#### 3. Enhanced Report

The text report now includes:
1. **PCA vs UMAP comparison** section with winner for each tier
2. **Dimensionality impact** section showing optimal dimension per tier
3. **Quality vs quantity trade-offs** (existing, enhanced)

### 📊 Complete Set of Plots

After running the enhanced script, you'll get **9 plots total**:

**NEW (PCA vs UMAP):**
1. `pca_vs_umap_by_tier.png` - Mean EF comparison
2. `pca_vs_umap_distributions.png` - Distribution box plots

**NEW (Dimensionality):**
3. `dimensionality_impact_by_tier.png` - EF vs dimension curves
4. `method_dimension_tier_heatmap.png` - Performance heatmap
5. `optimal_dimensionality_by_tier.png` - Best dimension per tier

**EXISTING (Enhanced):**
6. `stratified_ef_by_nn.png` - UMAP hyperparameter impact
7. `stratified_percent_found.png` - Recovery rates
8. `quality_vs_quantity_tradeoff.png` - Overall vs High-potent scatter
9. (Plus standard plots if min_dist variations included)

### 🚀 Run Updated Analysis

```bash
cd /home/ahagg2s/UMMBAS_screening_experiments

# Pull updated script
git pull

# Run full analysis with new plots
python scripts/analyze_potency_stratified_enrichment.py \
    --workspace_dir experiment_workspace_v3_phase1 \
    --output_dir potency_analysis_v2
```

### 📈 What to Look For

#### PCA vs UMAP Performance Pattern

**If PCA dominates across all tiers:**
- Suggests gradient-based approaches work better
- MF cloud creates smooth transitions
- Higher dimensionality helps preserve structure

**If UMAP dominates across all tiers:**
- Suggests local structure matters more
- Clustering/islands important
- Lower dimensionality (2D) might be optimal

**If performance varies by tier:**
- **PCA better for high-potent** = High-potent compounds form distinct gradient
- **UMAP better for high-potent** = High-potent compounds form isolated clusters
- **Similar performance** = Both capture relevant structure

#### Dimensionality Pattern

**If optimal dim is consistent across tiers:**
- Same dimensionality works for all potency ranges
- Simple recommendation: use that dimension

**If optimal dim varies by tier:**
- **High-potent prefers lower dim** = Strong signal, doesn't need high-D
- **Weak binders prefer higher dim** = Need more dimensions to separate noise
- **Medium optimal for all** = Sweet spot (likely 5D based on your data)

#### Method × Dimension Interaction

The heatmap will show if certain combinations excel:
- **PCA-5D best for high-potent** = Specific method-dim synergy
- **UMAP-2D best for weak** = Different optimal per use case
- **No clear pattern** = Main effects dominate (method OR dimension, not both)

### 💡 Interpretation Guide

#### Scenario 1: PCA-5D dominates all tiers
**Interpretation:** Phase transition hypothesis validated - MF cloud creates optimal gradient in 5D space

**Recommendation:** Use PCA-5D for all future work

#### Scenario 2: UMAP-2D best for high-potent, PCA-10D best for medium/weak
**Interpretation:** High-potent compounds cluster, others need dimensionality

**Recommendation:** Use UMAP-2D if prioritizing drug-like hits, PCA-10D for comprehensive screening

#### Scenario 3: Performance similar, but PCA more stable (lower variance)
**Interpretation:** Both methods capture structure, PCA more reliable

**Recommendation:** Use PCA for consistency, especially in production

### 📝 Updated Documentation

The report will now show:

```
===========================================================================
KEY FINDINGS
===========================================================================

1. PCA vs UMAP Performance by Potency Tier:

  High Potency (0.1-100 nM):
    PCA:  56.8 ± 12.3
    UMAP: 33.5 ± 8.7
    → PCA wins by 69.6%

  Medium Potency (100-1,000 nM):
    PCA:  66.4 ± 15.2
    UMAP: 50.2 ± 11.4
    → PCA wins by 32.3%

  Weak Potency (1,000-100,000 nM):
    PCA:  60.1 ± 14.1
    UMAP: 45.1 ± 10.8
    → PCA wins by 33.3%

2. Impact of Dimensionality:

  Dimensions tested: 2, 5, 10

  High Potency:
    2D:  48.2 ± 10.1
    5D:  56.8 ± 12.3
    10D: 52.1 ± 11.5
    → Optimal: 5D (EF = 56.8)

  Medium Potency:
    2D:  58.3 ± 13.2
    5D:  66.4 ± 15.2
    10D: 61.2 ± 14.1
    → Optimal: 5D (EF = 66.4)

  Weak Potency:
    2D:  55.1 ± 12.8
    5D:  60.1 ± 14.1
    10D: 57.3 ± 13.2
    → Optimal: 5D (EF = 60.1)

3. Quality vs Quantity Trade-offs:
  [existing analysis]
```

### 🎯 Key Scientific Questions Answered

1. **Does PCA or UMAP better enrich drug-like compounds?**
   - Answered by High Potency comparison
   - Critical for method selection

2. **Is optimal dimensionality consistent or potency-dependent?**
   - Answered by dimensionality analysis per tier
   - Informs future experimental design

3. **Do method advantages vary by compound quality?**
   - Answered by method×tier interaction
   - Reveals when to use which method

4. **Should we use same config for all potency ranges?**
   - Answered by comparing optimal configs per tier
   - Practical guidance for pipeline design

### 🔬 For Your Paper/Results Section

The new analyses provide publication-ready figures showing:

1. **Method comparison** - Clear PCA vs UMAP performance across clinically relevant potency ranges
2. **Dimensionality study** - Systematic evaluation of 2D/5D/10D impact
3. **Interaction effects** - Method×Dimension performance landscape
4. **Optimal configurations** - Data-driven recommendations per use case

All plots are 300 DPI, publication-quality, with clear labels and legends.

### 📌 Next Steps

1. Run updated analysis on your full dataset
2. Review the new plots for patterns
3. Check if results support phase transition hypothesis
4. Use findings to select optimal config for Phase 2+

---

**Updated**: October 17, 2025  
**New plots**: 5 additional (9 total)  
**New report sections**: 2 (PCA vs UMAP, Dimensionality)
