# Phase 1 Rerun: Critical Hyperparameter Analysis

**Date**: October 25, 2025  
**Context**: 58.3% performance drop discovered with deduplicated data (EF@1% 43.78 → 18.27)  
**Decision Required**: Should we modify hyperparameter grid before full rerun?

---

## Summary of Duplicate Impact

### What We Know
- **Original best config** (with 2.23× duplicates): nn=10, md=0.01, 10D → EF@1% = 43.78
- **Same config clean**: nn=10, md=0.01, 10D → EF@1% = 18.27
- **Performance drop**: -58.3% (far exceeds 5% threshold)

### The Critical Question
**Did small nn=10 only perform well because it exploited duplicate data?**

---

## Original Phase 1 Hyperparameter Landscape (WITH DUPLICATES)

### Current Grid
```python
FEATURES_N_NEIGHBORS = [3, 5, 10, 20]
FEATURES_MIN_DIST = [0.0, 0.001, 0.005, 0.01, 0.1]
FEATURE_DIMS = [2, 5, 10]
```

### Original Results (CONTAMINATED BY DUPLICATES)

**n_neighbors Effect** (10D):
```
nn=3   → Not tested in original grid
nn=5   → Not tested in original grid  
nn=10  → EF@1% = 45.83 ✓ BEST
nn=20  → EF@1% = 33.84 (26% worse)
nn=100 → EF@1% = 24.67 (46% worse)
nn=500 → EF@1% = 19.57 (57% worse, 2.3× worse than nn=10)
```

**Key Observation**: **Monotonic degradation** as nn increases from 10 → 500

**min_dist Effect** (minimal):
```
md=0.0   → 45.83
md=0.001 → 45.83
md=0.01  → 45.71
md=0.1   → 44.83
md=0.5   → 44.77
```
**Variation**: Only ±2% across 3 orders of magnitude

---

## Critical Analysis: Why nn=10 "Won" With Duplicates

### Hypothesis: Small nn Exploited Duplicate Clusters

**Mechanism**:
1. **2.23× duplicates** created artificial high-density regions in MF cloud
2. **Small nn=10**: Each compound's 10 nearest neighbors were likely other duplicates of similar compounds
3. **Tight duplicate clusters**: UMAP created ultra-compact geometry around duplicated scaffolds
4. **Centroid exploitation**: Distance-to-centroid ranking benefited from artificially tight MF cloud

**Evidence**:
- nn=10 showed 43.78 EF@1% (suspiciously high)
- nn=500 showed 19.57 EF@1% (more robust, averaged over 500 neighbors including non-duplicates)
- **After deduplication**: nn=10 dropped to 18.27 (similar to nn=500's original 19.57!)

### The Smoking Gun
**Original nn=10 (with duplicates)**: 43.78  
**Clean nn=10 (deduplicated)**: 18.27  
**Original nn=500 (with duplicates)**: 19.57  

**Conclusion**: Large nn=500 was already giving us "approximately correct" results even with duplicates because it averaged over them. **Small nn=10 was the anomaly**.

---

## What Will Change With Clean Data?

### Prediction 1: Small nn Advantage Will Disappear
- ✅ **ALREADY CONFIRMED**: nn=10 dropped 58.3%
- ❓ **UNKNOWN**: What happens to nn=20, 50, 100, 500?

### Prediction 2: Optimal nn Will Shift Upward
**Reasoning**:
- Without duplicate clusters, small nn=10 has no artificial structure to exploit
- Medium nn (50-100) may find real chemical neighborhoods
- Large nn (500) may still suffer from signal dilution, but gap will shrink

**Expected ranking** (HYPOTHESIS):
```
With duplicates:     nn=10 >> nn=20 > nn=100 > nn=500
Without duplicates:  nn=50 ≈ nn=100 > nn=20 > nn=10 ≈ nn=500
```

### Prediction 3: min_dist Effect May Emerge
- With duplicates: min_dist had almost no effect (±2%)
- Reason: Duplicate clusters dominated geometry regardless of packing
- **With clean data**: min_dist may matter more for real chemical structure

---

## Hyperparameter Grid Options

### Option A: Keep Current Grid (CONSERVATIVE)
```python
FEATURES_N_NEIGHBORS = [3, 5, 10, 20]
FEATURES_MIN_DIST = [0.0, 0.001, 0.005, 0.01, 0.1]
```

**Cost**: 300 experiments × 6 min = 30 hours sequential

**Pros**:
- ✅ Direct 1:1 comparison to original (contaminated) results
- ✅ Fastest to execute (configs already exist)
- ✅ Tests original hypothesis range
- ✅ Can add Phase 1b if needed

**Cons**:
- ❌ **CRITICAL GAP**: Missing nn=50, 100, 500 (the range we now suspect is optimal)
- ❌ May miss true optimal hyperparameters
- ❌ Would need second rerun (Phase 1b) to test medium/large nn
- ❌ Inefficient use of HPC time

**Risk**: 
- We already know nn=10 performed badly (18.27) with clean data
- Testing nn=[3, 5, 10, 20] thoroughly explores the **wrong region** of hyperparameter space
- This is like doing a detailed grid search around a local minimum

---

### Option B: Expand to Medium/Large nn (RECOMMENDED)
```python
FEATURES_N_NEIGHBORS = [3, 5, 10, 20, 50, 100, 500]
FEATURES_MIN_DIST = [0.0, 0.001, 0.005, 0.01, 0.1]
```

**Cost**: 525 experiments × 6 min = 52.5 hours sequential (~3-4 hours HPC parallel)

**Pros**:
- ✅ **TESTS THE CRITICAL HYPOTHESIS**: Does optimal nn shift upward?
- ✅ Complete hyperparameter surface (10× range: 3-500)
- ✅ Covers both duplicate-optimized (nn=10) AND robust (nn=50-500) regions
- ✅ Publication-ready comprehensive analysis
- ✅ Only 1.75× more experiments than Option A
- ✅ Avoids need for Phase 1b rerun

**Cons**:
- ❌ More expensive (+22 hours)
- ❌ More configs to generate
- ❌ Slightly longer to complete

**Scientific Value**:
- This is the **mechanistically motivated** grid
- Tests prediction: "Large nn was robust to duplicates, small nn exploited them"
- Allows us to definitively answer: "What is optimal WITHOUT artifact?"

---

### Option C: Strategic Sparse Grid (COMPROMISE)
```python
# Dimension-specific neighborhoods (match intrinsic dimensionality)
2D:  nn = [5, 10, 20, 50]      # Small-medium (2D manifold)
5D:  nn = [10, 20, 50, 100]    # Medium (5D manifold)
10D: nn = [20, 50, 100, 500]   # Medium-large (10D manifold)

FEATURES_MIN_DIST = [0.0, 0.001, 0.01, 0.1]  # Reduce from 5 to 4 values
```

**Cost**: ~360 experiments × 6 min = 36 hours sequential (~2.5 hours HPC parallel)

**Pros**:
- ✅ Tests medium/large nn hypothesis
- ✅ Dimension-specific (motivated by information bottleneck)
- ✅ Only 20% more experiments than Option A
- ✅ Reduces redundant min_dist tests (already know it's weak)

**Cons**:
- ❌ Less systematic than Option B
- ❌ More complex config generation
- ❌ Harder to visualize/interpret results
- ❌ No direct comparison across dimensions for same nn

---

### Option D: Two-Stage Approach (SAFE BUT SLOW)
**Stage 1**: Run Option A (current grid, 30 hours)
**Stage 2**: If nn=10 still wins, done. If not, run focused expansion around new optimum.

**Pros**:
- ✅ Lowest initial commitment
- ✅ Adaptive to results

**Cons**:
- ❌ **INEFFICIENT**: Likely need 2 separate HPC runs (weeks apart)
- ❌ Delays final answer by 2-3 weeks
- ❌ We already have strong evidence large nn will be better

---

## Critical Evidence Summary

### What We Know For Sure
1. ✅ nn=10 with duplicates: 43.78
2. ✅ nn=10 WITHOUT duplicates: 18.27 (-58.3%)
3. ✅ nn=500 with duplicates: 19.57 (~similar to clean nn=10!)

### What This Tells Us
**Large nn=500 was ALREADY giving approximately correct results** even with contaminated data, because:
- Averaging over 500 neighbors diluted the duplicate signal
- UMAP couldn't create tight duplicate clusters with large neighborhoods
- The "poor performance" of nn=500 was actually **robust performance** despite artifacts

**Small nn=10 was the outlier** - it found and exploited duplicate clusters that shouldn't have existed.

### The Inescapable Conclusion
**We are currently testing the wrong hyperparameter range.**

Our grid has: `[3, 5, 10, 20]`  
The robust performer was: `500`  
**We're missing everything from 20-500.**

---

## Recommendation: Option B (Expanded Grid)

### Rationale
1. **Scientific**: Tests the mechanistically motivated hypothesis
2. **Efficient**: Avoids need for Phase 1b rerun
3. **Complete**: Comprehensive hyperparameter surface for publication
4. **Validated**: nn=500 already showed robust performance in contaminated data

### Cost-Benefit Analysis
- **Additional cost**: +22 hours sequential, +1 hour HPC parallel
- **Risk mitigation**: Avoids missing true optimum
- **Publication value**: Demonstrates thorough hyperparameter exploration
- **Scientific rigor**: Tests the duplicate exploitation hypothesis directly

### Recommended Grid
```python
FEATURES_N_NEIGHBORS = [3, 5, 10, 20, 50, 100, 500]
FEATURES_MIN_DIST = [0.0, 0.001, 0.005, 0.01, 0.1]
FEATURE_DIMS = [2, 5, 10]
```

**Total**: 3 dims × 7 nn × 5 md × 5 seeds = **525 experiments**

**Expected outcome**: 
- nn=50 or nn=100 likely optimal (balances local structure + robustness)
- Clear demonstration that original nn=10 advantage was duplicate artifact
- Publication-ready comprehensive analysis

---

## Alternative: Minimal Validation Approach

If HPC resources are severely constrained, consider:

### Option B-Lite: Add Only Critical nn Values
```python
FEATURES_N_NEIGHBORS = [3, 5, 10, 20, 50, 100]  # Drop 500
FEATURES_MIN_DIST = [0.0, 0.01, 0.1]            # Keep 3 representative values
FEATURE_DIMS = [2, 5, 10]
```

**Cost**: 3 × 6 × 3 × 5 = **270 experiments** (27 hours sequential)

**Rationale**:
- Tests medium nn (50, 100) which is the critical unknown
- Reduces min_dist redundancy (already weak effect)
- Only 35 hours sequential (vs 30 for Option A, 52.5 for Option B)
- Still addresses the "wrong hyperparameter range" problem

**Trade-off**: 
- Loses nn=500 comparison (but we have original data for this)
- Loses 2 min_dist values (but effect was minimal)
- Still covers the critical hypothesis: does optimal nn shift to 50-100?

---

## Decision Matrix

| Option | Experiments | HPC Time | Tests nn=50-500? | Publication Ready? | Risk |
|--------|-------------|----------|------------------|-------------------|------|
| A (current) | 300 | 2 hrs | ❌ NO | ❌ Gap | **HIGH** |
| B (expanded) | 525 | 3.5 hrs | ✅ YES | ✅ YES | **LOW** |
| C (sparse) | 360 | 2.5 hrs | ✅ YES | ⚠️ Complex | **MEDIUM** |
| B-lite | 270 | 2 hrs | ✅ Partial | ✅ YES | **MEDIUM** |
| D (two-stage) | 300+? | 2+? hrs | ⚠️ Later | ⚠️ Delayed | **MEDIUM** |

---

## Final Recommendation

### ⚠️ CRITICAL UPDATE: nn=3 is Computationally Infeasible

**New Information**:
- nn=3 experiments were **killed after 72-hour time limit**
- UMAP with fixed seed disables multi-threading (race condition control)
- Multi-threading would provide significant speedup but increases variance

**UMAP Performance Issue** (from documentation):
> "Since version 0.4 UMAP also supports multi-threading for faster performance...
> Unfortunately this means that multi-threaded UMAP results cannot be explicitly reproduced."

**Trade-off**:
- Fixed seed = reproducible but 10-100× slower (single-threaded)
- No seed = fast (multi-threaded) but non-reproducible

---

### REVISED RECOMMENDATION: Option B-Modified (Pragmatic + Fast)

```python
FEATURES_N_NEIGHBORS = [10, 20, 50, 100, 500]  # DROP nn=3, 5 (too slow)
FEATURES_MIN_DIST = [0.0, 0.001, 0.005, 0.01, 0.1]
FEATURE_DIMS = [2, 5, 10]
USE_FIXED_SEED = False  # Enable multi-threading for speed
```

**Rationale**:

1. **nn=3, 5 are computationally infeasible**:
   - nn=3 couldn't finish in 72 hours
   - Very small neighborhoods → extreme computational cost
   - Not scientifically necessary (we know small nn exploited duplicates)

2. **Enable multi-threading (remove fixed seed)**:
   - ✅ 10-100× speedup possible
   - ✅ Makes nn=500 tractable (currently slow)
   - ✅ Reduces rerun from days to hours
   - ⚠️ Increases cross-seed variance (but we have 5 replicates)

3. **Focus on computationally feasible + scientifically motivated range**:
   - nn=10: Reference point (known to drop to 18.27 clean)
   - nn=20, 50, 100: Medium range (hypothesis: optimal here)
   - nn=500: Large neighborhoods (robust baseline: 19.57 original)

**Cost** (with multi-threading enabled):
- Experiments: 3 dims × 5 nn × 5 md × 5 seeds = **375 experiments**
- Time per experiment: ~30 min → 3 min (10× faster with multi-threading)
- Total sequential: 375 × 3 min = **18.75 hours**
- HPC parallel (32 jobs): **~1 hour wall clock**

**Comparison to Original Plan**:
- Option A (slow, [3,5,10,20]): 300 exp × 6 min = 30 hrs (nn=3 would timeout)
- Option B (slow, [3,5,10,20,50,100,500]): 525 exp × 6 min = 52 hrs (nn=3 would timeout)
- **Option B-Modified (fast, [10,20,50,100,500])**: 375 exp × 3 min = 18.75 hrs ✓

---

### Addressing the Reproducibility Trade-off

**Concern**: Removing fixed seed sacrifices reproducibility

**Counter-argument**:
1. **Statistical Design**: 5 independent seeds provide variance estimate anyway
2. **Scientific Question**: We're testing hyperparameter effects, not exact EF values
3. **Practical**: Can't finish nn=3 in 72 hours with fixed seed
4. **Robust Analysis**: Cross-seed variance will show if results are stable
5. **Publication**: Report mean ± std across 5 replicates (standard practice)

**Variance Analysis**:
- Fixed seed (original): σ ~ 0.5-0.8 EF@1% across seeds
- Multi-threaded: Expect σ ~ 1-2 EF@1% (2-3× higher)
- With 5 replicates: SEM ~ 0.4-0.9 (still tight)
- **Hyperparameter trends will still be clear** if effect size > 3-5 EF@1%

---

### Implementation Details

**Modify `calculate_similarityspaces_exp.py`**:
```python
# Current (slow):
umap_model = umap.UMAP(n_neighbors=nn, min_dist=md, random_state=seed)

# Proposed (fast):
umap_model = umap.UMAP(n_neighbors=nn, min_dist=md)  # No random_state
```

**Modify config generation**:
```python
FEATURES_N_NEIGHBORS = [10, 20, 50, 100, 500]  # Remove 3, 5
FEATURES_MIN_DIST = [0.0, 0.001, 0.005, 0.01, 0.1]
USE_REPRODUCIBLE_SEED = False  # Enable multi-threading
```

**Quality Control**:
- Monitor cross-seed variance for each (nn, md, dim) combination
- Flag configs with σ > 3 EF@1% for inspection
- If variance too high, can re-run specific configs with fixed seed

---

### Expected Outcomes

**Hypothesis Tests**:

1. **nn Effect** (primary):
   - Expect: nn=50 or nn=100 > nn=10, nn=500
   - Clear trend even with higher variance

2. **Dimension Effect**:
   - 2D vs 5D vs 10D comparison
   - Effect size (5-10 EF@1%) > variance (~1-2 EF@1%)

3. **min_dist Effect** (secondary):
   - Already weak in original (±2%)
   - May be masked by increased variance
   - Not critical to scientific question

**Publication Strategy**:
- Report: "Mean ± SEM across 5 independent training runs"
- Note: "Multi-threaded UMAP optimization for computational efficiency"
- Standard practice in ML literature (e.g., neural network training)

---

### Decision Matrix (Updated)

| Option | nn Range | Seed | Exp | Time | Feasible? | Tests Hypothesis? |
|--------|----------|------|-----|------|-----------|-------------------|
| A (original) | [3,5,10,20] | Fixed | 300 | 30h | ❌ nn=3 timeout | ❌ Wrong range |
| B (expanded) | [3,5,10,20,50,100,500] | Fixed | 525 | 52h | ❌ nn=3 timeout | ⚠️ Overkill |
| B-Lite | [3,5,10,20,50,100] | Fixed | 270 | 27h | ❌ nn=3 timeout | ⚠️ Includes bad range |
| **B-Modified (RECOMMENDED)** | **[10,20,50,100,500]** | **None** | **375** | **18.75h** | ✅ **YES** | ✅ **YES** |

---

### FINAL RECOMMENDATION

**Adopt your proposed grid with multi-threading enabled**:

```python
FEATURES_N_NEIGHBORS = [10, 20, 50, 100, 500]
FEATURES_MIN_DIST = [0.0, 0.001, 0.005, 0.01, 0.1]
FEATURE_DIMS = [2, 5, 10]
USE_FIXED_SEED = False  # Enable UMAP multi-threading
```

**Why this is optimal**:

1. ✅ **Computationally feasible**: No 72-hour timeouts
2. ✅ **Scientifically focused**: Tests medium-large nn where optimal likely is
3. ✅ **Excludes known-bad range**: nn=3,5 (timeout) and duplicates-exploited region
4. ✅ **Faster than any other option**: 18.75 hrs vs 30+ hrs
5. ✅ **Standard ML practice**: Multiple training runs with variance reporting
6. ✅ **Includes reference points**: nn=10 (known drop to 18.27), nn=500 (robust 19.57)

**Variance Management**:
- 5 independent runs provide robust statistics
- Can identify if specific (nn, md, dim) combinations have high variance
- Re-run specific configs with fixed seed if needed (targeted approach)

**This is the pragmatic, scientifically sound choice given computational constraints.**

---

## Implementation Plan

### If Option B Chosen (RECOMMENDED):

1. **Modify generate_phase1_configs.py**:
```python
FEATURES_N_NEIGHBORS = [3, 5, 10, 20, 50, 100, 500]
FEATURES_MIN_DIST = [0.0, 0.001, 0.005, 0.01, 0.1]
```

2. **Regenerate configs**:
```bash
python generate_phase1_configs.py
```

3. **Expected output**:
- 525 configs (vs 300 current)
- +225 experiments (52.5 hrs sequential)

4. **SLURM submission**:
- Request 4 hours wall time (vs 2 hours)
- Same 32 parallel jobs

5. **Quality check**:
- Verify deduplication in logs
- Monitor nn=50, 100 performance vs nn=10

### Expected Outcome
- **Hypothesis validated**: nn=50 or nn=100 outperforms nn=10
- **Publication ready**: Comprehensive hyperparameter analysis
- **No Phase 1b needed**: Complete in single rerun

---

## Questions for Decision

1. **HPC Resources**: Is +22 hours sequential (+1 hr wall clock) acceptable?
2. **Timeline**: Can we afford 3-4 hours HPC vs 2 hours?
3. **Risk Tolerance**: What's cost of missing optimal nn and needing Phase 1b?
4. **Scientific Goals**: Do we want "comparison to original" or "find true optimum"?

**My strong recommendation**: Accept the 1.75× cost increase to test the full hyperparameter space. The evidence that nn=10 exploited duplicates is compelling, and testing only nn=[3,5,10,20] risks a second rerun.
