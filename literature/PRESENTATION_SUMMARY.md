# Virtual Screening Benchmarks: Literature Summary for Presentation

## SLIDE 1: EF@1% Performance Benchmarks by Method Class

| Method Class | Typical EF@1% Range | Representative Values | Status | Key Citations |
|:------------|:-------------------|:---------------------|:-------|:--------------|
| **Traditional LBVS** | | | | |
| Tanimoto fingerprint | 3–8 | Baseline reference | Classical standard | [1,2,3] |
| 2D/3D similarity integration | 15–20 | EF@1% = 17.5–19.96 | Pre-ML baseline | [21] |
| **Structure-Based Docking** | | | | |
| AutoDock Vina (standard) | 2–9 | DUD-E: 8.8; Unbiased: 1.9 | Standard tool | [7,8] |
| Glide (commercial) | ~21 | Mean EF@1% = 21.3 (DUD-E) | Industry standard | [25] |
| With ML rescoring | 15–28 | OnionNet+Vina: 15.5; DHFR: 28–31 | Enhanced docking | [7,23] |
| **Machine Learning** | | | | |
| Random Forest (RF-Score-VS) | 7–40 | DUD-E: 39; DEKOIS: 9.8 | Dataset-dependent | [9] |
| Deep Learning (GNINA) | 8–19 | Target-dependent | CNN-based scoring | [7] |
| **Consensus Methods** | | | | |
| Exponential consensus | 14–18 | Multi-method ensemble | State-of-the-art | [11] |
| ENS-VS (SOTA) | **~53** | Mean EF@1% = 52.77 (DUD-E) | **Best reported** | [24] |

**Performance Classification:**
- EF@1% < 3: Poor (no enrichment)
- EF@1% 3–8: Fair (adequate for large libraries)
- **EF@1% 8–20: Good (recommended for campaigns)**
- **EF@1% 20–40: Excellent (resource-limited scenarios)**
- EF@1% > 40: Exceptional (often biased benchmarks)

---

## SLIDE 2: Critical Context & Your Method's Position

### Dataset Bias Warning
| Benchmark | Performance Inflation | Reason | Citation |
|:----------|:---------------------|:-------|:---------|
| DUD-E | **2–5× inflated EF** | Analogue bias + decoy bias | [29] |
| LIT-PCBA | **Unreliable** | Data leakage + duplication | [33] |
| **Recommendation** | Use unbiased splits (UMAP clustering) or prospective validation | | [17,18] |

### Interpreting Your Results

**If your PCA/UMAP + 1-NN achieves:**

| EF@1% | Scientific Interpretation | Competitive Position |
|:-----------|:-------------------------|:---------------------|
| **< 10** | Needs optimization (descriptors/DR params) | Below classical methods |
| **10–20** | **Solid performance** | Competitive with traditional docking |
| **20–30** | **Excellent performance** | **Matches commercial tools (Glide)** |
| **30–40** | **Outstanding** | **Approaches ML methods** |
| **> 40** | Exceptional (verify benchmark bias) | State-of-the-art territory |

### Key Advantages of Your Approach
- ✅ **Speed:** 5 orders of magnitude faster than docking (k-d tree search) [4]
- ✅ **Scalability:** Billion-molecule screening in seconds [4,12]
- ✅ **No structure required:** Pure ligand-based (unlike docking)
- ✅ **Theoretical foundation:** UMAP preserves chemical similarity manifolds [14,17,18]

### Critical Validation Steps
1. ✅ Benchmark on multiple targets (10–15 diverse proteins)
2. ✅ Use UMAP-based clustering splits to avoid analogue bias [17]
3. ✅ Report EF@1%, EF@5%, BEDROC (not just AUC) [6]
4. ✅ Compare to Morgan fingerprint + Tanimoto baseline

---

## Quick Reference: Key Citations

[1] LS-align ligand similarity (EF benchmarks)  
[2-3] Tanimoto coefficient benchmarks  
[4] Low-dimensional embeddings for rapid similarity search (arXiv 2402.07970)  
[6] Evaluating VS metrics (early recognition problem)  
[7] OnionNet-SFCT rescoring  
[8-9] RF-Score-VS on DUD-E/DEKOIS  
[11] Exponential consensus methods  
[12] SPRINT billion-molecule screening  
[14] Understanding UMAP  
[17-18] UMAP clustering splits for rigorous evaluation  
[21] 2D/3D similarity integration  
[23] DHFR ML rescoring  
[24] ENS-VS consensus (best reported EF@1% = 52.77)  
[25] Glide commercial docking baseline  
[29] DUD-E bias analysis  
[33] LIT-PCBA data leakage audit

**Full citations available in: `LITERATURE_REVIEW PERPLEXITY.md` and `LITERATURE_REVIEW GEMINI.md`**
