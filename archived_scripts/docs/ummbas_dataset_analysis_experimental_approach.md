# UMMBAS Dataset Analysis Report
Generated: 2025-10-14 10:33:51

## 1. ZINC Database (Decoy Compounds)

- **Total compounds**: 1,295,279
- **Unique SMILES**: 1,295,270
- **Data source**: datasets/molecular_function_features_fingerprints/zinc/zinc_acquirable_extracted_features.csv
- **Number of manufacturers**: 120
- **Number of tranches**: 98
- **Availability distribution**:
  - make_on_demand: 667,422
  - in_stock_for_immediate_delivery: 602,842
  - boutique: 25,015

## 2. ChEMBL Database Overview

- ChEMBL data analysis skipped in favor of pre-calculated molecular function datasets
- This analysis uses the same data loading approach as the actual experiments
- Target-specific and MF cloud compounds are loaded from pre-calculated feature files

## 3. Target-Specific Known Ligands (Held-out Actives)

These are the compounds used as 'held-out actives' in the leave-one-target-out experiments.

### Pyruvate Kinase M2 (P14618)
- **Molecular Function**: Protein kinase inhibitor
- **Total bioactivity records**: 166
- **Records with affinity values**: 166
- **Unique compounds (all)**: 166
- **Unique compounds (with affinity)**: 166
- **Active compounds (≤100,000 nM)**: 159
- **Affinity range**: 0.5 - 1920000.0 nM
- **Median affinity**: 2055.0 nM

### Isocitrate Dehydrogenase NADP cytoplasmic (O75874)
- **Molecular Function**: Oxidoreductase
- **Total bioactivity records**: 1,838
- **Records with affinity values**: 1,838
- **Unique compounds (all)**: 1,838
- **Unique compounds (with affinity)**: 1,838
- **Active compounds (≤100,000 nM)**: 1,838
- **Affinity range**: 0.0 - 55500.0 nM
- **Median affinity**: 190.0 nM

### Tyrosine-protein Kinase ABL1 (P00519)
- **Molecular Function**: Protein kinase inhibitor
- **Total bioactivity records**: 3,331
- **Records with affinity values**: 3,331
- **Unique compounds (all)**: 3,331
- **Unique compounds (with affinity)**: 3,331
- **Active compounds (≤100,000 nM)**: 3,313
- **Affinity range**: 0.0 - 444180.0 nM
- **Median affinity**: 132.0 nM

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
  - MF Cloud active compounds: 16
  - Unique targets in cloud: 3
  - Data source: datasets/molecular_function_features_fingerprints/KW-0649_Protein_kinase_inhibitor_affinity_extracted_features.csv
- **Tyrosine-protein Kinase ABL1**:
  - Excluded target: P00519
  - MF Cloud active compounds: 16
  - Unique targets in cloud: 3
  - Data source: datasets/molecular_function_features_fingerprints/KW-0649_Protein_kinase_inhibitor_affinity_extracted_features.csv

### Oxidoreductase
- **Targets in experiment**: Isocitrate Dehydrogenase NADP cytoplasmic
- **MF Keyword ID**: KW-0560

**MF Clouds per target** (excluding target-specific compounds):
- **Isocitrate Dehydrogenase NADP cytoplasmic**:
  - Excluded target: O75874
  - MF Cloud active compounds: 55,191
  - Unique targets in cloud: 270
  - Data source: datasets/molecular_function_features_fingerprints/KW-0560_Oxidoreductase_affinity_extracted_features.csv

### Hydrolase
- **Targets in experiment**: Actin cytoplasmic 1, Actin cytoplasmic 2
- **MF Keyword ID**: KW-0378

**MF Clouds per target** (excluding target-specific compounds):
- **Actin cytoplasmic 1**:
  - Excluded target: P60709
  - MF Cloud active compounds: 139,841
  - Unique targets in cloud: 660
  - Data source: datasets/molecular_function_features_fingerprints/KW-0378_Hydrolase_affinity_extracted_features.csv
- **Actin cytoplasmic 2**:
  - Excluded target: P63261
  - MF Cloud active compounds: 139,841
  - Unique targets in cloud: 660
  - Data source: datasets/molecular_function_features_fingerprints/KW-0378_Hydrolase_affinity_extracted_features.csv

## 5. Summary Statistics

- **ZINC decoy compounds**: 1,295,279
- **Total held-out active compounds** (across all targets): 5,310
- **MF Cloud sizes** (active compounds, excluding target-specific):
  - Pyruvate Kinase M2: 16
  - Tyrosine-protein Kinase ABL1: 16
  - Isocitrate Dehydrogenase NADP cytoplasmic: 55,191
  - Actin cytoplasmic 1: 139,841
  - Actin cytoplasmic 2: 139,841

## 6. Experimental Context

In the UMMBAS experiments:
1. **ZINC compounds** serve as decoys (negative examples)
2. **Held-out actives** are the compounds we try to identify (positive examples)
3. **MF Cloud** compounds create the similarity space for ranking
4. The goal is to rank held-out actives higher than ZINC decoys
5. Performance is measured by how well the method separates actives from decoys
