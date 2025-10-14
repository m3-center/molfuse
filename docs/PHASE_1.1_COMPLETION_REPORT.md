# Phase 1.1 Completion Report: UniProt Molecular Function Verification

**Date:** October 14, 2025  
**Status:** ✅ COMPLETED  
**Phase:** 1.1 - Query UniProt for Correct Molecular Functions

---

## Executive Summary

Successfully verified molecular function classifications for all three UMMBAS target proteins by querying the UniProt REST API. **All sanity checks passed.**

### Key Findings

✅ **ABL1 (P00519):** Confirmed as **Transferase (KW-0808)**  
✅ **Pyruvate Kinase M2 (P14618):** Confirmed as **Transferase (KW-0808)**  
✅ **Isocitrate Dehydrogenase (O75874):** Confirmed as **Oxidoreductase (KW-0560)**

---

## Detailed Results

### 1. Tyrosine-protein Kinase ABL1 (P00519)

**Expected:** Transferase (KW-0808)  
**Status:** ✅ VERIFIED

**All Molecular Function Keywords:**
- DNA-binding (KW-0238)
- Kinase (KW-0418)
- **Transferase (KW-0808)** ✓
- Tyrosine-protein kinase (KW-0829)

**Total Keywords:** 34  
**Rationale:** Kinases catalyze phosphoryl transfer reactions

---

### 2. Pyruvate Kinase M2 (P14618)

**Expected:** Transferase (KW-0808)  
**Status:** ✅ VERIFIED

**All Molecular Function Keywords:**
- Allosteric enzyme (KW-0021)
- Kinase (KW-0418)
- **Transferase (KW-0808)** ✓

**Total Keywords:** 25  
**Rationale:** Kinases catalyze phosphoryl transfer reactions

---

### 3. Isocitrate Dehydrogenase NADP cytoplasmic (O75874)

**Expected:** Oxidoreductase (KW-0560)  
**Status:** ✅ VERIFIED

**All Molecular Function Keywords:**
- **Oxidoreductase (KW-0560)** ✓

**Total Keywords:** 15  
**Rationale:** Dehydrogenases catalyze oxidation-reduction reactions

---

## Sanity Check Results

### ✅ CHECK 1: All targets have expected molecular functions?
**PASS** - All 3 targets verified!

### ✅ CHECK 2: ABL1 and Pyruvate Kinase M2 both have Transferase?
**PASS** - Both kinases verified as Transferases

### ✅ CHECK 3: Isocitrate Dehydrogenase has Oxidoreductase?
**PASS** - Isocitrate Dehydrogenase verified as Oxidoreductase

### ✅ CHECK 4: No kinases incorrectly classified as 'Protein kinase inhibitor'?
**PASS** - No targets have 'Protein kinase inhibitor' keyword

**Important Note:** The keyword "Protein kinase inhibitor" (KW-0649) refers to **compounds that inhibit kinases**, NOT the kinase enzymes themselves. This is why v1.0 was incorrect - it used KW-0649 for kinase targets, which gave a tiny MF cloud of only 20 non-kinase inhibitor compounds.

---

## Files Generated

1. **Verification Summary:**  
   `datasets/protein_collection/uniprot_verification_v2.0.csv`
   - Contains verification status for all 3 targets
   - Includes expected vs. found molecular functions
   - Documents all molecular function keywords per target

2. **Detailed Keywords:**  
   `datasets/protein_collection/uniprot_all_keywords_v2.0.csv`
   - All 74 keywords across 3 targets
   - Categorized by keyword type (Molecular function, Biological process, etc.)
   - Useful for understanding protein annotations

3. **JSON Data:**  
   `datasets/protein_collection/uniprot_verification_v2.0.json`
   - Machine-readable format for programmatic access
   - Includes timestamp and verification status

---

## Key Insights

### Why This Matters for v2.0

**v1.0 Error:**
- ABL1 and Pyruvate Kinase M2 were assigned to "Protein kinase inhibitor" (KW-0649)
- This gave a MF cloud of only **20 compounds** (non-kinase proteins: CDK-interacting protein, Tribbles 1/2)
- Created severe data imbalance: 3,331 held-out ABL1 actives vs 20 MF cloud (165:1 ratio)

**v2.0 Correction:**
- ABL1 and Pyruvate Kinase M2 correctly assigned to "Transferase" (KW-0808)
- Expected MF cloud: **~5,505 compounds** for ABL1, **~202 compounds** for Pyruvate Kinase M2
- Creates balanced, scientifically valid experiments

### Biochemical Justification

**Kinases are Transferases:**
- Catalyze phosphoryl group transfer from ATP to substrate
- EC classification: 2.7.x.x (Transferases - phosphotransferases)
- UniProt correctly classifies them under KW-0808 (Transferase)

**Dehydrogenases are Oxidoreductases:**
- Catalyze oxidation-reduction reactions involving NAD+/NADP+
- EC classification: 1.1.x.x (Oxidoreductases)
- UniProt correctly classifies them under KW-0560 (Oxidoreductase)

---

## Script Details

**Created Script:**  
`scripts/query_uniprot_molecular_functions.py`

**Functionality:**
- Queries UniProt REST API for each target protein
- Retrieves all keyword annotations
- Verifies expected molecular function is present
- Performs 4 sanity checks
- Exports results to CSV and JSON formats

**API Used:**  
`https://rest.uniprot.org/uniprotkb/{accession}.json`

**Rate Limiting:**  
0.5 second delay between requests (polite API usage)

---

## Comparison: v1.0 vs v2.0 Molecular Function Assignments

| Target | UniProt ID | v1.0 (INCORRECT) | v2.0 (CORRECT) | Impact |
|--------|------------|------------------|----------------|--------|
| ABL1 | P00519 | Protein kinase inhibitor (KW-0649) | **Transferase (KW-0808)** | MF cloud: 20 → ~5,505 compounds |
| Pyruvate Kinase M2 | P14618 | Protein kinase inhibitor (KW-0649) | **Transferase (KW-0808)** | MF cloud: 20 → ~202 compounds |
| Isocitrate Dehydrogenase | O75874 | Oxidoreductase (KW-0560) ✓ | **Oxidoreductase (KW-0560)** ✓ | No change (correct in v1.0) |

---

## Next Steps

✅ **Phase 1.1 Complete**

➡️ **Proceed to Phase 1.2:**  
Extract ChEMBL Transferase bioactivity data for KW-0808

**Expected Workflow:**
1. Query ChEMBL 35 database for human proteins with KW-0808 (Transferase)
2. Filter bioactivity data: ≤100,000 nM affinity
3. Exclude held-out targets: P00519 (ABL1), P14618 (Pyruvate Kinase M2)
4. Expected output: ~5,505 unique compounds for ABL1 MF cloud
5. Save to: `datasets/molecular_function_affinity_data/KW-0808_Transferase_affinity.csv`

---

## Conclusion

Phase 1.1 successfully validated the correct molecular function classifications from UniProt. All three targets are properly classified according to their biochemical activities:

- **Kinases → Transferases** (phosphoryl transfer)
- **Dehydrogenases → Oxidoreductases** (redox reactions)

This correction forms the foundation for valid v2.0 experiments. The next phase will extract the corrected MF clouds from ChEMBL, which should provide substantially larger and more appropriate chemical spaces for virtual screening.

---

**Verified by:** UMMBAS v2.0 Pipeline  
**Date:** 2025-10-14  
**Status:** Ready for Phase 1.2
