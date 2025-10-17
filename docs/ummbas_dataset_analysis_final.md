# UMMBAS Dataset Analysis Report
Generated: 2025-10-14 12:18:31

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

## 5. Summary Statistics

- **ZINC decoy compounds**: 1,295,279
- **Total held-out active compounds** (across all targets): 5,310
- **MF Cloud sizes** (active compounds, excluding target-specific):
  - Pyruvate Kinase M2: 16
  - Tyrosine-protein Kinase ABL1: 16
  - Isocitrate Dehydrogenase NADP cytoplasmic: 55,191

## 6. Experimental Context

In the UMMBAS experiments:
1. **ZINC compounds** serve as decoys (negative examples)
2. **Held-out actives** are the compounds we try to identify (positive examples)
3. **MF Cloud** compounds create the similarity space for ranking
4. The goal is to rank held-out actives higher than ZINC decoys
5. Performance is measured by how well the method separates actives from decoys

**Note**: Only targets with `processing_mode='full_analysis'` are included in this analysis.
Targets with `processing_mode='similarity_space_only'` (e.g., Actin proteins) were used only for 
similarity space generation and are excluded from these counts.

## 7. LaTeX Dataset Description for Publication

```latex
\subsection{Dataset}
We employed a comprehensive molecular dataset comprising three protein targets 
for virtual screening evaluation. The dataset includes:\\[0.5em]

\textbf{Target Proteins:} Three protein targets were selected for leave-one-target-out 
virtual screening experiments: (1) Tyrosine-protein kinase ABL1 (UniProt: P00519), 
(2) Isocitrate dehydrogenase NADP cytoplasmic (UniProt: O75874), and 
(3) Pyruvate kinase M2 (UniProt: P14618). These targets represent diverse molecular 
functions including protein kinase inhibition and oxidoreductase activity.\\[0.5em]

\textbf{Held-out Active Compounds:} A total of 5,310 bioactive compounds 
with experimentally validated affinity (≤100,000 nM) were extracted from ChEMBL 35 
database. The distribution per target was: Pyruvate Kinase M2 (159), Isocitrate Dehydrogenase NADP cytoplasmic (1,838), Tyrosine-protein Kinase ABL1 (3,313).\\[0.5em]

\textbf{Decoy Compounds:} 1,295,279 purchasable small molecules from the ZINC 
database served as decoy compounds (presumed inactive) for virtual screening 
evaluation.\\[0.5em]

\textbf{Molecular Function Clouds:} For each target, similarity spaces were constructed 
using compounds that bind to proteins sharing the same molecular function, excluding 
the target-specific compounds (leave-one-target-out approach). The molecular function cloud sizes were: Pyruvate Kinase M2: 16, Tyrosine-protein Kinase ABL1: 16, Isocitrate Dehydrogenase NADP cytoplasmic: 55,191. These clouds provide the molecular context for dimensionality reduction and 
similarity-based virtual screening.\\[0.5em]

All molecular representations were computed using RDKit, including 39 physicochemical 
descriptors and 1024-bit Extended Connectivity Fingerprints (ECFP4). The dataset 
design ensures realistic virtual screening conditions where active compounds must be 
distinguished from a large background of presumed inactive molecules.
```
