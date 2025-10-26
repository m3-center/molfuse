#!/usr/bin/env python3
"""
Extract Transferase (KW-0808) bioactivity data from ChEMBL 35 for UMMBAS v2.0.

Query ChEMBL for all human proteins with Transferase molecular function,
excluding held-out targets (ABL1, Pyruvate Kinase M2).

Author: UMMBAS v2.0
Date: 2025-10-14
"""

import sqlite3
import pandas as pd
import sys
from datetime import datetime

# Configuration
CHEMBL_DB_PATH = "datasets/chembl/chembl_35.db"
TARGET_MAPPING_PATH = "datasets/chembl/chembl_35_target_mapping.csv"
AFFINITY_DATA_PATH = "datasets/chembl/chembl_35_affinity_data.csv"
OUTPUT_PATH = "datasets/molecular_function_affinity_data/KW-0808_Transferase_affinity.csv"

TRANSFERASE_KW = "KW-0808"
AFFINITY_CUTOFF_NM = 100000
EXCLUDE_UNIPROT_IDS = ["P00519", "P14618"]  # ABL1, Pyruvate Kinase M2

print("="*80)
print("UMMBAS v2.0 - Extract ChEMBL Transferase Data")
print("="*80)
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Step 1: Load target mapping to find Transferase proteins
print("Step 1: Loading target mapping...")
try:
    df_mapping = pd.read_csv(TARGET_MAPPING_PATH)
    print(f"✓ Loaded {len(df_mapping)} target mappings")
except FileNotFoundError:
    print(f"✗ ERROR: Target mapping file not found: {TARGET_MAPPING_PATH}")
    sys.exit(1)

# Filter for Transferase (KW-0808) human proteins
transferase_targets = df_mapping[
    (df_mapping['uniprot_keywords'].str.contains(TRANSFERASE_KW, na=False)) &
    (df_mapping['organism'] == 'Homo sapiens')
]

# Exclude held-out targets
transferase_targets = transferase_targets[
    ~transferase_targets['accession'].isin(EXCLUDE_UNIPROT_IDS)
]

print(f"✓ Found {len(transferase_targets)} Transferase targets (excluding ABL1, Pyruvate Kinase M2)")
print(f"  UniProt IDs: {transferase_targets['accession'].nunique()} unique")

# Step 2: Load affinity data
print("\nStep 2: Loading affinity data...")
try:
    df_affinity = pd.read_csv(AFFINITY_DATA_PATH)
    print(f"✓ Loaded {len(df_affinity)} affinity records")
except FileNotFoundError:
    print(f"✗ ERROR: Affinity data file not found: {AFFINITY_DATA_PATH}")
    sys.exit(1)

# Step 3: Filter affinity data for Transferase targets
print("\nStep 3: Filtering affinity data...")
transferase_accessions = transferase_targets['accession'].unique()
df_filtered = df_affinity[
    (df_affinity['accession'].isin(transferase_accessions)) &
    (df_affinity['Standard Value (nM)'] <= AFFINITY_CUTOFF_NM)
]

print(f"✓ Filtered to {len(df_filtered)} records")
print(f"  Unique compounds: {df_filtered['Compound ChEMBL ID'].nunique()}")
print(f"  Unique targets: {df_filtered['accession'].nunique()}")

# Sanity checks
print("\n" + "="*80)
print("SANITY CHECKS")
print("="*80)

# Check 1: ABL1 and Pyruvate Kinase M2 excluded?
excluded_found = df_filtered['accession'].isin(EXCLUDE_UNIPROT_IDS).any()
print(f"\n✓ CHECK 1: ABL1 and Pyruvate Kinase M2 excluded?")
if not excluded_found:
    print(f"   ✅ PASS - Held-out targets not in dataset")
else:
    print(f"   ❌ FAIL - Held-out targets found in dataset!")
    sys.exit(1)

# Check 2: Expected compound count for ABL1 scenario
abl1_compound_count = df_filtered['Compound ChEMBL ID'].nunique()
print(f"\n✓ CHECK 2: Compound count in expected range?")
if 4000 <= abl1_compound_count <= 7000:
    print(f"   ✅ PASS - {abl1_compound_count} compounds (expected ~5,505)")
else:
    print(f"   ⚠️  WARNING - {abl1_compound_count} compounds (expected ~5,505)")
    print(f"      This may indicate data differences from original analysis")

# Check 3: All affinities <= cutoff?
max_affinity = df_filtered['Standard Value (nM)'].max()
print(f"\n✓ CHECK 3: All affinities <= {AFFINITY_CUTOFF_NM} nM?")
if max_affinity <= AFFINITY_CUTOFF_NM:
    print(f"   ✅ PASS - Max affinity: {max_affinity:.1f} nM")
else:
    print(f"   ❌ FAIL - Max affinity: {max_affinity:.1f} nM exceeds cutoff")
    sys.exit(1)

# Check 4: All human proteins?
if 'organism' in df_filtered.columns:
    non_human = df_filtered[df_filtered['organism'] != 'Homo sapiens']
    print(f"\n✓ CHECK 4: All human proteins?")
    if len(non_human) == 0:
        print(f"   ✅ PASS - All records from Homo sapiens")
    else:
        print(f"   ⚠️  WARNING - {len(non_human)} non-human records found")

# Step 4: Save output
print("\n" + "="*80)
print("SAVING OUTPUT")
print("="*80)

df_filtered.to_csv(OUTPUT_PATH, index=False)
print(f"\n✓ Saved Transferase affinity data to: {OUTPUT_PATH}")
print(f"  Total records: {len(df_filtered)}")
print(f"  Unique compounds: {df_filtered['Compound ChEMBL ID'].nunique()}")
print(f"  Unique targets: {df_filtered['accession'].nunique()}")

# Summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)
print(f"\nTop 10 targets by compound count:")
target_counts = df_filtered.groupby('accession')['Compound ChEMBL ID'].nunique().sort_values(ascending=False).head(10)
for acc, count in target_counts.items():
    target_name = transferase_targets[transferase_targets['accession']==acc]['target_name'].values
    name = target_name[0] if len(target_name) > 0 else 'Unknown'
    print(f"  {acc}: {count} compounds ({name})")

print("\n✅ Phase 1.2 Complete - Ready for Phase 1.3 (feature/fingerprint calculation)")
