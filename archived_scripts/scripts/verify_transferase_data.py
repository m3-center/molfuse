#!/usr/bin/env python3
"""Verify existing Transferase data and check if ABL1/Pyruvate Kinase M2 excluded."""

import pandas as pd

print("="*80)
print("Verifying KW-0808 Transferase Data")
print("="*80 + "\n")

df = pd.read_csv("datasets/molecular_function_affinity_data/KW-0808_Transferase_affinity.csv")

print(f"Total records: {len(df):,}")
print(f"Unique compounds: {df['Compound ChEMBL ID'].nunique():,}")
print(f"Unique targets: {df['accession'].nunique()}")

# Check if ABL1 (P00519) or Pyruvate Kinase M2 (P14618) are present
print("\n" + "="*80)
print("SANITY CHECKS")
print("="*80)

abl1_present = df['accession'].eq('P00519').any()
pk_present = df['accession'].eq('P14618').any()

print(f"\n✓ CHECK 1: ABL1 (P00519) excluded?")
if not abl1_present:
    print(f"   ✅ PASS - ABL1 not in dataset")
else:
    print(f"   ❌ FAIL - ABL1 found in dataset ({df[df['accession']=='P00519'].shape[0]} records)")

print(f"\n✓ CHECK 2: Pyruvate Kinase M2 (P14618) excluded?")
if not pk_present:
    print(f"   ✅ PASS - Pyruvate Kinase M2 not in dataset")
else:
    print(f"   ❌ FAIL - Pyruvate Kinase M2 found ({df[df['accession']=='P14618'].shape[0]} records)")

print(f"\n✓ CHECK 3: Compound count reasonable?")
n_compounds = df['Compound ChEMBL ID'].nunique()
if 4000 <= n_compounds <= 10000:
    print(f"   ✅ PASS - {n_compounds:,} compounds in expected range")
else:
    print(f"   ⚠️  INFO - {n_compounds:,} compounds (wider than expected ~5,505)")

# Check if we need to filter this data
if abl1_present or pk_present:
    print("\n⚠️  WARNING: Need to filter out held-out targets!")
    df_filtered = df[~df['accession'].isin(['P00519', 'P14618'])]
    df_filtered.to_csv("datasets/molecular_function_affinity_data/KW-0808_Transferase_affinity.csv", index=False)
    print(f"✓ Saved filtered data: {len(df_filtered):,} records, {df_filtered['Compound ChEMBL ID'].nunique():,} compounds")
else:
    print("\n✅ Data already clean - no filtering needed")
    print(f"\n➡️  Ready for Phase 1.3: Calculate features/fingerprints for {n_compounds:,} compounds")
