# UMMBAS Dataset Analysis Report
Generated: 2025-10-14 10:26:55

## 1. ZINC Database (Decoy Compounds)

- **Total compounds**: 1,295,279
- **Unique SMILES**: 1,295,270
- **Number of manufacturers**: 120
- **Number of tranches**: 98
- **Availability distribution**:
  - make_on_demand: 667,422
  - in_stock_for_immediate_delivery: 602,842
  - boutique: 25,015

## 2. ChEMBL Database Overview

- **Total bioactivity records**: 2,780,638
- **Unique compounds**: 1,113,469
- **Unique targets**: 9,808
- **Targets with UniProt mapping**: 2,074,979
- **Activity types**:
  - IC50: 1,896,276
  - Ki: 477,277
  - EC50: 338,479
  - Kd: 68,606

## 3. Target-Specific Known Ligands (Held-out Actives)

These are the compounds used as 'held-out actives' in the leave-one-target-out experiments.

### Pyruvate Kinase M2 (P14618)
- **Molecular Function**: Protein kinase inhibitor
- **Total bioactivity records**: 202
- **Records with affinity values**: 202
- **Unique compounds (all)**: 166
- **Unique compounds (with affinity)**: 166
- **Active compounds (≤100,000 nM)**: 159
- **Affinity range**: 0.5 - 1920000.0 nM
- **Median affinity**: 2110.0 nM

### Isocitrate Dehydrogenase NADP cytoplasmic (O75874)
- **Molecular Function**: Oxidoreductase
- **Total bioactivity records**: 3,039
- **Records with affinity values**: 3,039
- **Unique compounds (all)**: 1,838
- **Unique compounds (with affinity)**: 1,838
- **Active compounds (≤100,000 nM)**: 1,838
- **Affinity range**: 0.0 - 608000.0 nM
- **Median affinity**: 189.0 nM

### Tyrosine-protein Kinase ABL1 (P00519)
- **Molecular Function**: Protein kinase inhibitor
- **Total bioactivity records**: 5,505
- **Records with affinity values**: 5,505
- **Unique compounds (all)**: 3,331
- **Unique compounds (with affinity)**: 3,331
- **Active compounds (≤100,000 nM)**: 3,314
- **Affinity range**: 0.0 - 43800000.0 nM
- **Median affinity**: 120.0 nM

### Actin cytoplasmic 1 (P60709)
- **Molecular Function**: Hydrolase
- **Total bioactivity records**: 0
- **Records with affinity values**: 0
- **Unique compounds (all)**: 0
- **Unique compounds (with affinity)**: 0
- **Active compounds (≤100,000 nM)**: 0

### Actin cytoplasmic 2 (P63261)
- **Molecular Function**: Hydrolase
- **Total bioactivity records**: 0
- **Records with affinity values**: 0
- **Unique compounds (all)**: 0
- **Unique compounds (with affinity)**: 0
- **Active compounds (≤100,000 nM)**: 0

## 4. Molecular Function Clouds (MF Cloud)

These are the compounds used to create the 'similarity space' for each target.
Each MF Cloud contains compounds that bind to proteins with the same molecular function,
but excludes compounds for the specific target being tested (leave-one-target-out).

### Protein kinase inhibitor
- **Targets in experiment**: Pyruvate Kinase M2, Tyrosine-protein Kinase ABL1
- **MF Keyword ID**: KW-0649

**MF Clouds per target** (excluding target-specific compounds):
- **Pyruvate Kinase M2**:
  - Excluded target: P14618
  - MF Cloud active compounds: 813,825
  - Unique targets in cloud: 7,339
- **Tyrosine-protein Kinase ABL1**:
  - Excluded target: P00519
  - MF Cloud active compounds: 812,728
  - Unique targets in cloud: 7,339

### Oxidoreductase
- **Targets in experiment**: Isocitrate Dehydrogenase NADP cytoplasmic
- **MF Keyword ID**: KW-0560

**MF Clouds per target** (excluding target-specific compounds):
- **Isocitrate Dehydrogenase NADP cytoplasmic**:
  - Excluded target: O75874
  - MF Cloud active compounds: 812,194
  - Unique targets in cloud: 7,339

### Hydrolase
- **Targets in experiment**: Actin cytoplasmic 1, Actin cytoplasmic 2
- **MF Keyword ID**: KW-0378

**MF Clouds per target** (excluding target-specific compounds):
- **Actin cytoplasmic 1**:
  - Excluded target: P60709
  - MF Cloud active compounds: 813,950
  - Unique targets in cloud: 7,340
- **Actin cytoplasmic 2**:
  - Excluded target: P63261
  - MF Cloud active compounds: 813,950
  - Unique targets in cloud: 7,340

## 5. Summary Statistics

- **ZINC decoy compounds**: 1,295,279
- **Total held-out active compounds** (across all targets): 5,311
- **MF Cloud sizes** (active compounds, excluding target-specific):
  - Pyruvate Kinase M2: 813,825
  - Tyrosine-protein Kinase ABL1: 812,728
  - Isocitrate Dehydrogenase NADP cytoplasmic: 812,194
  - Actin cytoplasmic 1: 813,950
  - Actin cytoplasmic 2: 813,950

## 6. Experimental Context

In the UMMBAS experiments:
1. **ZINC compounds** serve as decoys (negative examples)
2. **Held-out actives** are the compounds we try to identify (positive examples)
3. **MF Cloud** compounds create the similarity space for ranking
4. The goal is to rank held-out actives higher than ZINC decoys
5. Performance is measured by how well the method separates actives from decoys
