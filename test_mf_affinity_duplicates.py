#!/usr/bin/env python3
"""
Test script to investigate MF affinity data duplication issue.

This script checks if the MF cloud affinity source file has duplicate
Compound ChEMBL IDs, which would cause data explosion during merge.

Usage:
    python test_mf_affinity_duplicates.py
"""

import pandas as pd
import numpy as np
import os

# File paths
phase1_run = "experiment_workspace_v3_phase1/run_seed42_config_tyro_features_pca_dim5_seed42"
simspace_file = os.path.join(phase1_run, "TyrosineProteinKinaseABL1_P00519/similarity_spaces/features/dim_5/TyrosineProteinKinaseABL1_P00519_features_dim5_similarity_space.csv")
affinity_file = os.path.join(phase1_run, "TyrosineProteinKinaseABL1_P00519/temp_data/TyrosineProteinKinaseABL1_P00519_chembl_mf_excluded_features.csv")

print("="*80)
print("MF AFFINITY DATA DUPLICATION TEST")
print("="*80)

# 1. Load similarity space (MF cloud coordinates)
print("\n1. Loading similarity space MF cloud...")
df_simspace = pd.read_csv(simspace_file, low_memory=False)
print(f"   Total rows in similarity space: {len(df_simspace):,}")

# Filter to MF cloud only
if 'DataSource' in df_simspace.columns:
    df_mf_cloud = df_simspace[df_simspace['DataSource'] == 'ChEMBL_MF']
elif 'MOLECULE ID' in df_simspace.columns:
    df_mf_cloud = df_simspace[~df_simspace['MOLECULE ID'].str.startswith('ZINC', na=False)]
else:
    print("   ERROR: Cannot identify MF cloud in similarity space!")
    exit(1)

print(f"   MF cloud rows: {len(df_mf_cloud):,}")
print(f"   Unique Compound ChEMBL IDs in MF cloud: {df_mf_cloud['Compound ChEMBL ID'].nunique():,}")

# 2. Load affinity source file
print("\n2. Loading affinity source file...")
print(f"   File: {affinity_file}")
df_affinity = pd.read_csv(affinity_file, usecols=['Compound ChEMBL ID', 'Standard Value (nM)'], low_memory=False)
print(f"   Total rows in affinity file: {len(df_affinity):,}")
print(f"   Unique Compound ChEMBL IDs in affinity file: {df_affinity['Compound ChEMBL ID'].nunique():,}")

# 3. Check for duplicates
print("\n3. Checking for duplicate Compound ChEMBL IDs...")
duplicate_counts = df_affinity['Compound ChEMBL ID'].value_counts()
duplicates = duplicate_counts[duplicate_counts > 1]

if len(duplicates) > 0:
    print(f"   ✗ FOUND DUPLICATES!")
    print(f"   Number of compounds with duplicates: {len(duplicates):,}")
    print(f"   Total duplicate entries: {(duplicate_counts - 1).sum():,}")
    print(f"\n   Top 10 most duplicated compounds:")
    for chembl_id, count in duplicates.head(10).items():
        print(f"     {chembl_id}: {count} occurrences")
    
    # Show example of one duplicated compound
    example_id = duplicates.index[0]
    print(f"\n   Example: {example_id} appears {duplicates.iloc[0]} times with affinities:")
    example_rows = df_affinity[df_affinity['Compound ChEMBL ID'] == example_id]
    for idx, row in example_rows.head(5).iterrows():
        print(f"     - {row['Standard Value (nM)']} nM")
else:
    print(f"   ✓ No duplicates found - each Compound ChEMBL ID appears exactly once")

# 4. Simulate the merge to predict result size
print("\n4. Simulating merge to predict result size...")
merge_result = df_mf_cloud[['Compound ChEMBL ID']].merge(
    df_affinity,
    on='Compound ChEMBL ID',
    how='left'
)
print(f"   MF cloud before merge: {len(df_mf_cloud):,} rows")
print(f"   After merge: {len(merge_result):,} rows")
print(f"   Data explosion factor: {len(merge_result) / len(df_mf_cloud):.1f}x")

if len(merge_result) > len(df_mf_cloud):
    print(f"   ✗ MERGE CREATES {len(merge_result) - len(df_mf_cloud):,} EXTRA ROWS!")
else:
    print(f"   ✓ Merge does not create extra rows")

# 5. Propose solution
print("\n5. Proposed solution:")
if len(duplicates) > 0:
    print("   Aggregate duplicates BEFORE merge using one of:")
    print("   - Option A: Minimum affinity (most conservative - best binding)")
    print("   - Option B: Median affinity (middle ground)")
    print("   - Option C: Mean affinity (average)")
    
    # Test aggregation
    df_affinity_aggregated = df_affinity.groupby('Compound ChEMBL ID').agg({
        'Standard Value (nM)': ['min', 'median', 'mean', 'count']
    }).reset_index()
    df_affinity_aggregated.columns = ['Compound ChEMBL ID', 'min_affinity', 'median_affinity', 'mean_affinity', 'count']
    
    print(f"\n   After aggregation:")
    print(f"   - Unique compounds: {len(df_affinity_aggregated):,}")
    
    # Show statistics for compounds with multiple measurements
    multi_measurement = df_affinity_aggregated[df_affinity_aggregated['count'] > 1]
    if len(multi_measurement) > 0:
        print(f"   - Compounds with multiple measurements: {len(multi_measurement):,}")
        print(f"   - Average measurements per compound: {multi_measurement['count'].mean():.1f}")
        
        # Show an example
        example = multi_measurement.iloc[0]
        print(f"\n   Example aggregated compound ({example['Compound ChEMBL ID']}):")
        print(f"     - Minimum affinity: {example['min_affinity']:.2f} nM")
        print(f"     - Median affinity: {example['median_affinity']:.2f} nM")
        print(f"     - Mean affinity: {example['mean_affinity']:.2f} nM")
        print(f"     - Number of measurements: {int(example['count'])}")
    
    # Test merge with aggregated data
    merge_result_agg = df_mf_cloud[['Compound ChEMBL ID']].merge(
        df_affinity_aggregated[['Compound ChEMBL ID', 'min_affinity']],
        on='Compound ChEMBL ID',
        how='left'
    )
    print(f"\n   Merge with aggregated data:")
    print(f"   - Result size: {len(merge_result_agg):,} rows (same as MF cloud ✓)")
    print(f"   - Memory saved: {(len(merge_result) - len(merge_result_agg)) / len(merge_result) * 100:.1f}%")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
