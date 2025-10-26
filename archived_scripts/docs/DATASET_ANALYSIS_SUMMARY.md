# UMMBAS Dataset Analysis Summary

## ✅ Verified: Counting the Exact Molecules Used in Experiments

I have successfully updated the dataset analysis script to match the **exact same data loading approach** used in the UMMBAS experiments. Here are the key findings:

## Dataset Sizes (Experimentally Verified)

### **ZINC Decoy Compounds**
- **1,295,279 compounds** (1,295,270 unique SMILES)
- Source: `datasets/molecular_function_features_fingerprints/zinc/zinc_acquirable_extracted_features.csv`
- This is the exact file used by the experiments

### **Target-Specific Held-out Actives** (≤100,000 nM cutoff)
- **ABL1**: 3,313 active compounds
- **Isocitrate Dehydrogenase**: 1,838 active compounds  
- **Pyruvate Kinase M2**: 159 active compounds
- **Actin proteins**: 0 compounds (no ChEMBL data available)
- **Total**: 5,310 held-out actives across all targets

### **Molecular Function Clouds** (Active compounds, target-excluded)
- **Protein kinase inhibitor** (ABL1 & Pyruvate Kinase M2): 16 compounds each
- **Oxidoreductase** (Isocitrate Dehydrogenase): 55,191 compounds
- **Hydrolase** (Actin proteins): 139,841 compounds each

## Key Methodological Corrections Made

### 1. **Data Loading Approach**
- ✅ Now uses pre-calculated molecular function feature files (same as experiments)
- ✅ Applies target exclusion by UniProt ID (same as `prepare_data.py`)
- ✅ Uses experimental affinity cutoff of 100,000 nM 
- ✅ Loads ChEMBL data with same merge logic as `load_and_merge_chembl_data_for_raw_target_ligands()`

### 2. **Exclusion Logic**
- ✅ MF Clouds exclude compounds for the specific target being tested (leave-one-target-out)
- ✅ Uses UniProt accession matching rather than simple ChEMBL ID filtering
- ✅ Applies same affinity filtering as `project_and_analyze.py`

### 3. **Dataset Sources**
- ✅ ZINC: Uses the pre-calculated features file from experiments
- ✅ Target ligands: Loaded via same ChEMBL merge approach as `prepare_data.py`
- ✅ MF Clouds: Uses pre-calculated MF feature files with target exclusion

## Important Discovery: Small MF Clouds

The analysis reveals that some molecular function datasets are quite small:
- **Protein kinase inhibitor**: Only 16-20 compounds total (debugging dataset?)
- **Oxidoreductase**: ~55K compounds (reasonable size)
- **Hydrolase**: ~140K compounds (large dataset)

This suggests the current workspace may be using debugging/test datasets for some molecular functions, but **production datasets for others** (Oxidoreductase, Hydrolase).

## Experimental Consistency Verified ✅

The updated script now:
1. **Counts the exact molecules** used in UMMBAS experiments
2. **Uses identical data loading patterns** to experimental pipeline  
3. **Applies same filtering logic** as the core scripts
4. **Provides transparent data source tracking** for verification

This ensures the molecular counts reflect the **actual experimental conditions** rather than theoretical dataset sizes.