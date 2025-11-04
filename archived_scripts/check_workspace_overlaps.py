#!/usr/bin/env python3
"""
Check for SMILES overlaps between MF, target ligands, and ZINC datasets
in a specific experiment workspace temp_data folder.

Usage:
    python check_workspace_overlaps.py --temp_data_dir <path>
    
Example:
    python check_workspace_overlaps.py --temp_data_dir \
        /home/ahagg2s/UMMBAS_screening_experiments/experiment_workspace_v3_phase1/run_seed44_config_tyro_features_pca_dim5_seed44/TyrosineProteinKinaseABL1_P00519/temp_data
"""

import os
import argparse
import pandas as pd
from pathlib import Path


def load_smiles_from_csv(csv_path, smiles_column='SMILES'):
    """Load unique SMILES from a CSV file."""
    if not os.path.exists(csv_path):
        return set(), 0
    
    try:
        df = pd.read_csv(csv_path)
        if smiles_column not in df.columns:
            print(f"  WARNING: '{smiles_column}' column not found in {os.path.basename(csv_path)}")
            return set(), 0
        
        smiles_set = set(df[smiles_column].dropna().unique())
        return smiles_set, len(df)
    except Exception as e:
        print(f"  ERROR reading {csv_path}: {e}")
        return set(), 0


def find_csv_files(temp_data_dir):
    """Find relevant CSV files in temp_data directory."""
    files = {
        'target_raw': None,
        'target_features': None,
        'target_fingerprints': None,
        'mf_features': None,
        'mf_fingerprints': None,
        'zinc_features': None,
        'zinc_fingerprints': None
    }
    
    # Pattern matching for files
    for file in os.listdir(temp_data_dir):
        if not file.endswith('.csv'):
            continue
        
        # Target ligands
        if 'target_ligands_for_feature_calc_raw' in file:
            files['target_raw'] = os.path.join(temp_data_dir, file)
        
        # Filtered datasets
        if 'chembl_mf_excluded' in file:
            if 'features' in file:
                files['mf_features'] = os.path.join(temp_data_dir, file)
            elif 'fingerprints' in file:
                files['mf_fingerprints'] = os.path.join(temp_data_dir, file)
        
        if 'zinc_excluded' in file:
            if 'features' in file:
                files['zinc_features'] = os.path.join(temp_data_dir, file)
            elif 'fingerprints' in file:
                files['zinc_fingerprints'] = os.path.join(temp_data_dir, file)
    
    return files


def check_overlaps(temp_data_dir):
    """Check for overlaps between datasets in temp_data directory."""
    
    print("="*80)
    print("UMMBAS v3.0 - Workspace Data Overlap Checker")
    print("="*80)
    print(f"\nTemp Data Directory: {temp_data_dir}\n")
    
    # Verify directory exists
    if not os.path.isdir(temp_data_dir):
        print(f"ERROR: Directory not found: {temp_data_dir}")
        return 1
    
    # Find CSV files
    print("Step 1: Locating CSV files...")
    print("-" * 80)
    files = find_csv_files(temp_data_dir)
    
    for key, path in files.items():
        if path:
            print(f"  ✓ Found {key}: {os.path.basename(path)}")
        else:
            print(f"  ✗ Not found: {key}")
    
    # Load SMILES from target ligands
    print("\n" + "="*80)
    print("Step 2: Loading SMILES from datasets...")
    print("-" * 80)
    
    target_smiles = set()
    target_count = 0
    if files['target_raw']:
        target_smiles, target_count = load_smiles_from_csv(files['target_raw'])
        print(f"  Target ligands (raw): {len(target_smiles)} unique SMILES ({target_count} total)")
    
    # Load MF SMILES (use features version as primary)
    mf_smiles = set()
    mf_count = 0
    mf_file = files['mf_features'] or files['mf_fingerprints']
    if mf_file:
        mf_smiles, mf_count = load_smiles_from_csv(mf_file)
        repr_type = 'features' if 'features' in mf_file else 'fingerprints'
        print(f"  MF cloud ({repr_type}): {len(mf_smiles)} unique SMILES ({mf_count} total)")
    
    # Load ZINC SMILES (use features version as primary)
    zinc_smiles = set()
    zinc_count = 0
    zinc_file = files['zinc_features'] or files['zinc_fingerprints']
    if zinc_file:
        zinc_smiles, zinc_count = load_smiles_from_csv(zinc_file)
        repr_type = 'features' if 'features' in zinc_file else 'fingerprints'
        print(f"  ZINC decoys ({repr_type}): {len(zinc_smiles)} unique SMILES ({zinc_count} total)")
    
    # Check for overlaps
    print("\n" + "="*80)
    print("Step 3: Checking for overlaps...")
    print("-" * 80)
    
    issues_found = False
    
    # Target ↔ MF overlap
    if target_smiles and mf_smiles:
        target_mf_overlap = target_smiles.intersection(mf_smiles)
        if target_mf_overlap:
            print(f"  ✗ Target ↔ MF overlap: {len(target_mf_overlap)} molecules")
            print(f"    This is EXPECTED (target ligands should be in MF cloud)")
            print(f"    Sample overlapping SMILES: {list(target_mf_overlap)[:3]}")
        else:
            print(f"  ⚠ Target ↔ MF overlap: 0 molecules")
            print(f"    WARNING: Target ligands should typically be in MF cloud!")
    
    # Target ↔ ZINC overlap (should be 0)
    if target_smiles and zinc_smiles:
        target_zinc_overlap = target_smiles.intersection(zinc_smiles)
        if target_zinc_overlap:
            print(f"  ✗ Target ↔ ZINC overlap: {len(target_zinc_overlap)} molecules")
            print(f"    ERROR: Target ligands contaminating ZINC decoys!")
            print(f"    Sample overlapping SMILES: {list(target_zinc_overlap)[:5]}")
            issues_found = True
        else:
            print(f"  ✓ Target ↔ ZINC overlap: 0 molecules (CORRECT)")
    
    # MF ↔ ZINC overlap (should be 0)
    if mf_smiles and zinc_smiles:
        mf_zinc_overlap = mf_smiles.intersection(zinc_smiles)
        if mf_zinc_overlap:
            print(f"  ✗ MF ↔ ZINC overlap: {len(mf_zinc_overlap)} molecules")
            print(f"    ERROR: MF cloud contaminating ZINC decoys!")
            print(f"    Sample overlapping SMILES: {list(mf_zinc_overlap)[:5]}")
            issues_found = True
        else:
            print(f"  ✓ MF ↔ ZINC overlap: 0 molecules (CORRECT)")
    
    # Summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    
    if not target_smiles and not mf_smiles and not zinc_smiles:
        print("  ⚠ No datasets found or loaded successfully")
        return 1
    
    print(f"  Datasets loaded: {sum([bool(target_smiles), bool(mf_smiles), bool(zinc_smiles)])}/3")
    print(f"  Target ligands: {len(target_smiles)} unique SMILES")
    print(f"  MF cloud: {len(mf_smiles)} unique SMILES")
    print(f"  ZINC decoys: {len(zinc_smiles)} unique SMILES")
    print()
    
    if issues_found:
        print("  ✗ DATA INTEGRITY ISSUES DETECTED!")
        print("  Target or MF molecules are contaminating ZINC decoys.")
        print("  This indicates prepare_data.py filtering may have failed.")
        return 1
    else:
        print("  ✓ NO CONTAMINATION DETECTED")
        print("  ZINC decoys are properly filtered (no target/MF overlap).")
        return 0


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Check for SMILES overlaps in UMMBAS workspace temp_data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python check_workspace_overlaps.py --temp_data_dir \\
        experiment_workspace_v3_phase1/run_seed42_config_tyro_features_pca_dim2_seed42/TyrosineProteinKinaseABL1_P00519/temp_data
        """
    )
    
    parser.add_argument(
        '--temp_data_dir',
        type=str,
        required=True,
        help='Path to the temp_data directory containing CSV files'
    )
    
    args = parser.parse_args()
    
    # Run overlap check
    exit_code = check_overlaps(args.temp_data_dir)
    
    print("\n" + "="*80)
    if exit_code == 0:
        print("✓ Overlap check completed successfully")
    else:
        print("✗ Overlap check found issues or errors")
    print("="*80)
    
    return exit_code


if __name__ == "__main__":
    exit(main())
