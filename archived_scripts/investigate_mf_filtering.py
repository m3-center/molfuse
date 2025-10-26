#!/usr/bin/env python3
"""
Thorough investigation of the MF cloud filtering discrepancy.

This script checks:
1. How many molecules are in the original MF cloud (before filtering)
2. How many target ligands we have
3. How many molecules are actually removed from MF cloud
4. What's in the removed molecules (are they really all ABL1 ligands?)
5. Cross-check with raw ChEMBL data

Usage:
    python investigate_mf_filtering.py --temp_data_dir <path>
"""

import os
import argparse
import pandas as pd
import json


def main():
    parser = argparse.ArgumentParser(
        description="Investigate MF cloud filtering discrepancy"
    )
    parser.add_argument(
        '--temp_data_dir',
        type=str,
        required=True,
        help='Path to temp_data directory'
    )
    parser.add_argument(
        '--config_path',
        type=str,
        default='experiment_config.json',
        help='Path to experiment config'
    )
    args = parser.parse_args()
    
    print("="*80)
    print("THOROUGH INVESTIGATION: MF Cloud Filtering")
    print("="*80)
    print()
    
    # Load config to get paths
    with open(args.config_path, 'r') as f:
        config = json.load(f)
    gs = config['global_settings']
    
    # 1. Load target ligands
    print("Step 1: Loading Target Ligands")
    print("-"*80)
    target_file = None
    for file in os.listdir(args.temp_data_dir):
        if 'target_ligands_for_feature_calc_raw' in file:
            target_file = os.path.join(args.temp_data_dir, file)
            break
    
    if not target_file or not os.path.exists(target_file):
        print("ERROR: Target ligands file not found!")
        return 1
    
    df_target = pd.read_csv(target_file)
    target_smiles = set(df_target['SMILES'].dropna().unique())
    target_chembl_ids = set(df_target['Compound ChEMBL ID'].dropna().unique())
    target_uniprot = df_target['accession'].iloc[0] if 'accession' in df_target.columns else 'Unknown'
    
    print(f"  Target UniProt ID: {target_uniprot}")
    print(f"  Target ligands file: {os.path.basename(target_file)}")
    print(f"  Target records: {len(df_target)}")
    print(f"  Unique SMILES: {len(target_smiles)}")
    print(f"  Unique ChEMBL IDs: {len(target_chembl_ids)}")
    
    # Check affinity distribution
    if 'Standard Value (nM)' in df_target.columns:
        affinities = df_target['Standard Value (nM)'].dropna()
        print(f"\n  Affinity Distribution:")
        print(f"    Count: {len(affinities)}")
        print(f"    Min: {affinities.min():.2f} nM")
        print(f"    Median: {affinities.median():.2f} nM")
        print(f"    Max: {affinities.max():.2f} nM")
        print(f"    < 100 nM: {(affinities < 100).sum()}")
        print(f"    < 1,000 nM: {(affinities < 1000).sum()}")
        print(f"    < 10,000 nM: {(affinities < 10000).sum()}")
        print(f"    < 100,000 nM: {(affinities < 100000).sum()}")
    
    # 2. Load ORIGINAL (unfiltered) MF cloud from precalculated files
    print("\n" + "="*80)
    print("Step 2: Loading ORIGINAL MF Cloud (before filtering)")
    print("-"*80)
    
    # Get MF info from temp_data filenames
    mf_filtered_file = None
    for file in os.listdir(args.temp_data_dir):
        if 'chembl_mf_excluded_features' in file:
            mf_filtered_file = os.path.join(args.temp_data_dir, file)
            break
    
    if not mf_filtered_file:
        print("ERROR: MF filtered file not found!")
        return 1
    
    # Try to find the original MF file
    precalc_mf_base_dir = gs.get('precalculated_chembl_mf_features_base_dir')
    if not precalc_mf_base_dir:
        print("ERROR: Cannot find precalculated_chembl_mf_features_base_dir in config!")
        return 1
    
    # List all files in MF directory
    print(f"\n  MF Cloud base directory: {precalc_mf_base_dir}")
    if os.path.exists(precalc_mf_base_dir):
        mf_files = [f for f in os.listdir(precalc_mf_base_dir) if 'Transferase' in f or 'KW-0808' in f]
        print(f"  Found {len(mf_files)} potential Transferase MF files:")
        for f in mf_files[:5]:  # Show first 5
            print(f"    - {f}")
        
        # Load the appropriate one
        mf_original_file = None
        for f in mf_files:
            if 'affinity_extracted_features.csv' in f:
                mf_original_file = os.path.join(precalc_mf_base_dir, f)
                break
        
        if mf_original_file and os.path.exists(mf_original_file):
            print(f"\n  Loading ORIGINAL MF cloud: {os.path.basename(mf_original_file)}")
            df_mf_original = pd.read_csv(mf_original_file, low_memory=False)
            print(f"    Total records: {len(df_mf_original)}")
            print(f"    Columns: {list(df_mf_original.columns)}")
            
            mf_original_smiles = set(df_mf_original['SMILES'].dropna().unique()) if 'SMILES' in df_mf_original.columns else set()
            print(f"    Unique SMILES: {len(mf_original_smiles)}")
            
            # Check accession column
            if 'accession' in df_mf_original.columns:
                unique_proteins = df_mf_original['accession'].nunique()
                print(f"    Unique protein targets (accession): {unique_proteins}")
                
                # Count molecules for our target
                df_target_in_mf = df_mf_original[df_mf_original['accession'] == target_uniprot]
                target_mols_in_mf = len(df_target_in_mf['SMILES'].dropna().unique()) if not df_target_in_mf.empty else 0
                print(f"\n    Molecules targeting {target_uniprot} in ORIGINAL MF cloud: {target_mols_in_mf}")
                print(f"    Records targeting {target_uniprot}: {len(df_target_in_mf)}")
                
                if not df_target_in_mf.empty and 'Standard Value (nM)' in df_target_in_mf.columns:
                    affinities_mf = df_target_in_mf['Standard Value (nM)'].dropna()
                    print(f"\n    Affinity distribution for {target_uniprot} in MF cloud:")
                    print(f"      Min: {affinities_mf.min():.2f} nM")
                    print(f"      Median: {affinities_mf.median():.2f} nM")
                    print(f"      Max: {affinities_mf.max():.2f} nM")
                
                # Sample some molecules
                if target_mols_in_mf > 0:
                    sample_smiles = df_target_in_mf['SMILES'].dropna().unique()[:5]
                    print(f"\n    Sample SMILES targeting {target_uniprot}:")
                    for s in sample_smiles:
                        print(f"      {s}")
            
            # 3. Load FILTERED MF cloud
            print("\n" + "="*80)
            print("Step 3: Loading FILTERED MF Cloud (after exclusion)")
            print("-"*80)
            
            df_mf_filtered = pd.read_csv(mf_filtered_file)
            mf_filtered_smiles = set(df_mf_filtered['SMILES'].dropna().unique()) if 'SMILES' in df_mf_filtered.columns else set()
            
            print(f"  Filtered MF file: {os.path.basename(mf_filtered_file)}")
            print(f"  Records after filtering: {len(df_mf_filtered)}")
            print(f"  Unique SMILES after filtering: {len(mf_filtered_smiles)}")
            
            # 4. Calculate what was removed
            print("\n" + "="*80)
            print("Step 4: Analyzing Removed Molecules")
            print("-"*80)
            
            removed_smiles = mf_original_smiles - mf_filtered_smiles
            print(f"  SMILES removed from MF cloud: {len(removed_smiles)}")
            print(f"  SMILES kept in MF cloud: {len(mf_filtered_smiles)}")
            
            # Check overlap with target ligands
            removed_overlap_with_target = removed_smiles.intersection(target_smiles)
            print(f"\n  Removed SMILES that match target ligands: {len(removed_overlap_with_target)}")
            print(f"  Removed SMILES that DON'T match target ligands: {len(removed_smiles) - len(removed_overlap_with_target)}")
            
            # This is the KEY question
            if len(removed_smiles) > len(target_smiles):
                discrepancy = len(removed_smiles) - len(target_smiles)
                print(f"\n  ⚠️  DISCREPANCY: {discrepancy} more molecules removed than target ligands!")
                print(f"      Expected to remove: {len(target_smiles)} (target ligands)")
                print(f"      Actually removed: {len(removed_smiles)}")
                
                # Where are these extra molecules coming from?
                extra_removed = removed_smiles - target_smiles
                print(f"\n  Investigating the {len(extra_removed)} extra removed molecules...")
                
                # Check if they're in the original MF cloud with target accession
                if 'accession' in df_mf_original.columns:
                    df_extra = df_mf_original[df_mf_original['SMILES'].isin(extra_removed)]
                    if not df_extra.empty:
                        print(f"    These molecules appear {len(df_extra)} times in original MF")
                        
                        # Count by accession
                        accession_counts = df_extra['accession'].value_counts().head(10)
                        print(f"\n    Top proteins these molecules target:")
                        for acc, count in accession_counts.items():
                            unique_smiles = df_extra[df_extra['accession'] == acc]['SMILES'].nunique()
                            is_target = " ← TARGET!" if acc == target_uniprot else ""
                            print(f"      {acc}: {count} records, {unique_smiles} unique SMILES{is_target}")
                        
                        # Check if target uniprot appears
                        target_in_extra = df_extra[df_extra['accession'] == target_uniprot]
                        if not target_in_extra.empty:
                            print(f"\n    ⚠️  {len(target_in_extra)} of these 'extra' molecules ARE labeled with target {target_uniprot}!")
                            print(f"        But they're not in our target ligands file!")
                            print(f"        This suggests the target ligands file is INCOMPLETE!")
                            
                            # Sample some
                            sample_extra_target = target_in_extra['SMILES'].unique()[:5]
                            print(f"\n        Sample SMILES in MF but not in target ligands:")
                            for s in sample_extra_target:
                                in_target = "IN TARGET SET" if s in target_smiles else "NOT IN TARGET SET"
                                print(f"          {s} - {in_target}")
            
            # 5. Check ZINC filtering
            print("\n" + "="*80)
            print("Step 5: Investigating ZINC Filtering")
            print("-"*80)
            
            # Load original ZINC
            precalc_zinc_path = gs.get('precalculated_zinc_features_path')
            if not precalc_zinc_path:
                print("  ERROR: Cannot find precalculated_zinc_features_path in config!")
            elif not os.path.exists(precalc_zinc_path):
                print(f"  ERROR: ZINC file doesn't exist: {precalc_zinc_path}")
            else:
                print(f"  Loading ORIGINAL ZINC: {os.path.basename(precalc_zinc_path)}")
                print("  (This may take a while - ZINC is large...)")
                
                # Load ZINC (this is big, so be patient)
                df_zinc_original = pd.read_csv(precalc_zinc_path, low_memory=False)
                zinc_original_smiles = set(df_zinc_original['SMILES'].dropna().unique()) if 'SMILES' in df_zinc_original.columns else set()
                
                print(f"    Total records: {len(df_zinc_original):,}")
                print(f"    Unique SMILES: {len(zinc_original_smiles):,}")
                
                # Load filtered ZINC
                zinc_filtered_file = None
                for file in os.listdir(args.temp_data_dir):
                    if 'zinc_excluded_features' in file:
                        zinc_filtered_file = os.path.join(args.temp_data_dir, file)
                        break
                
                if zinc_filtered_file and os.path.exists(zinc_filtered_file):
                    print(f"\n  Loading FILTERED ZINC: {os.path.basename(zinc_filtered_file)}")
                    df_zinc_filtered = pd.read_csv(zinc_filtered_file, low_memory=False)
                    zinc_filtered_smiles = set(df_zinc_filtered['SMILES'].dropna().unique()) if 'SMILES' in df_zinc_filtered.columns else set()
                    
                    print(f"    Records after filtering: {len(df_zinc_filtered):,}")
                    print(f"    Unique SMILES after filtering: {len(zinc_filtered_smiles):,}")
                    
                    # Calculate what was removed
                    zinc_removed_smiles = zinc_original_smiles - zinc_filtered_smiles
                    print(f"\n  SMILES removed from ZINC: {len(zinc_removed_smiles):,}")
                    
                    # Check overlap with target and MF
                    zinc_removed_target_overlap = zinc_removed_smiles.intersection(target_smiles)
                    zinc_removed_mf_overlap = zinc_removed_smiles.intersection(mf_original_smiles)
                    
                    print(f"    Removed SMILES that match target ligands: {len(zinc_removed_target_overlap):,}")
                    print(f"    Removed SMILES that match MF cloud: {len(zinc_removed_mf_overlap):,}")
                    print(f"    Removed SMILES from both: {len(zinc_removed_target_overlap.intersection(zinc_removed_mf_overlap)):,}")
                    
                    # This is the critical check
                    all_smiles_to_exclude = target_smiles.union(mf_original_smiles)
                    expected_to_remove = zinc_original_smiles.intersection(all_smiles_to_exclude)
                    
                    print(f"\n  Expected SMILES to remove (target + MF): {len(expected_to_remove):,}")
                    print(f"  Actually removed: {len(zinc_removed_smiles):,}")
                    
                    if len(zinc_removed_smiles) == len(expected_to_remove):
                        print("  ✓ ZINC filtering is CORRECT!")
                    else:
                        discrepancy = len(zinc_removed_smiles) - len(expected_to_remove)
                        print(f"  ⚠️  Discrepancy: {discrepancy:,} SMILES")
                        
                        if discrepancy > 0:
                            extra = zinc_removed_smiles - expected_to_remove
                            print(f"      {len(extra):,} extra SMILES removed (not in target or MF)")
                            print(f"      Sample extra removed: {list(extra)[:3]}")
                        else:
                            missing = expected_to_remove - zinc_removed_smiles
                            print(f"      {abs(discrepancy):,} SMILES should have been removed but weren't")
                            print(f"      Sample missing: {list(missing)[:3]}")
                    
                    # Check final integrity with FILTERED MF cloud
                    print(f"\n  Final ZINC Integrity Check (using FILTERED MF cloud):")
                    final_target_overlap = zinc_filtered_smiles.intersection(target_smiles)
                    final_mf_filtered_overlap = zinc_filtered_smiles.intersection(mf_filtered_smiles)
                    
                    print(f"    ZINC (filtered) ↔ Target overlap: {len(final_target_overlap):,}")
                    print(f"    ZINC (filtered) ↔ MF (filtered) overlap: {len(final_mf_filtered_overlap):,}")
                    
                    if len(final_target_overlap) == 0 and len(final_mf_filtered_overlap) == 0:
                        print(f"    ✓ ALL CHECKS PASS - No contamination!")
                    else:
                        print(f"    ✗ CONTAMINATION DETECTED!")
                        if final_target_overlap:
                            print(f"        Target contamination: {len(final_target_overlap):,} SMILES")
                            print(f"        Sample: {list(final_target_overlap)[:3]}")
                        if final_mf_filtered_overlap:
                            print(f"        MF contamination: {len(final_mf_filtered_overlap):,} SMILES")
                            print(f"        Sample: {list(final_mf_filtered_overlap)[:3]}")
                    
                    # Additional check: Use RDKit to canonicalize SMILES and recheck
                    print(f"\n  Deep Integrity Check (with RDKit canonicalization):")
                    try:
                        from rdkit import Chem
                        from rdkit import RDLogger
                        RDLogger.DisableLog('rdApp.*')  # Suppress RDKit warnings
                        
                        print("    Canonicalizing SMILES with RDKit...")
                        
                        # Canonicalize ZINC filtered SMILES
                        zinc_canonical = set()
                        zinc_failed = 0
                        for smi in zinc_filtered_smiles:
                            try:
                                mol = Chem.MolFromSmiles(smi)
                                if mol is not None:
                                    canonical = Chem.MolToSmiles(mol)
                                    zinc_canonical.add(canonical)
                                else:
                                    zinc_failed += 1
                            except:
                                zinc_failed += 1
                        
                        print(f"      ZINC: {len(zinc_canonical):,} canonical SMILES ({zinc_failed} failed)")
                        
                        # Canonicalize MF filtered SMILES
                        mf_canonical = set()
                        mf_failed = 0
                        for smi in mf_filtered_smiles:
                            try:
                                mol = Chem.MolFromSmiles(smi)
                                if mol is not None:
                                    canonical = Chem.MolToSmiles(mol)
                                    mf_canonical.add(canonical)
                                else:
                                    mf_failed += 1
                            except:
                                mf_failed += 1
                        
                        print(f"      MF (filtered): {len(mf_canonical):,} canonical SMILES ({mf_failed} failed)")
                        
                        # Canonicalize target SMILES
                        target_canonical = set()
                        target_failed = 0
                        for smi in target_smiles:
                            try:
                                mol = Chem.MolFromSmiles(smi)
                                if mol is not None:
                                    canonical = Chem.MolToSmiles(mol)
                                    target_canonical.add(canonical)
                                else:
                                    target_failed += 1
                            except:
                                target_failed += 1
                        
                        print(f"      Target: {len(target_canonical):,} canonical SMILES ({target_failed} failed)")
                        
                        # Check overlaps with canonical SMILES
                        canonical_zinc_target = zinc_canonical.intersection(target_canonical)
                        canonical_zinc_mf = zinc_canonical.intersection(mf_canonical)
                        
                        print(f"\n    Canonical overlap check:")
                        print(f"      ZINC ↔ Target: {len(canonical_zinc_target):,} molecules")
                        print(f"      ZINC ↔ MF (filtered): {len(canonical_zinc_mf):,} molecules")
                        
                        if len(canonical_zinc_target) == 0 and len(canonical_zinc_mf) == 0:
                            print(f"      ✓ PERFECT - Zero overlap after canonicalization!")
                        else:
                            print(f"      ✗ WARNING - Overlap detected even after canonicalization!")
                            if canonical_zinc_target:
                                # Find original SMILES for these canonical ones
                                print(f"\n      Canonical SMILES in both ZINC and Target:")
                                for canon_smi in list(canonical_zinc_target)[:3]:
                                    print(f"        {canon_smi}")
                            if canonical_zinc_mf:
                                print(f"\n      Canonical SMILES in both ZINC and MF (filtered):")
                                for canon_smi in list(canonical_zinc_mf)[:3]:
                                    print(f"        {canon_smi}")
                        
                    except ImportError:
                        print("    ⚠️  RDKit not available - skipping canonical SMILES check")
                        print("    (Install with: conda install -c conda-forge rdkit)")
                
                else:
                    print(f"  ERROR: Filtered ZINC file not found!")
            
            # 6. Summary
            print("\n" + "="*80)
            print("OVERALL SUMMARY")
            print("="*80)
            print(f"  Original MF cloud: {len(mf_original_smiles):,} unique SMILES")
            print(f"  Filtered MF cloud: {len(mf_filtered_smiles):,} unique SMILES")
            print(f"  Removed from MF: {len(removed_smiles):,} unique SMILES")
            print(f"  Target ligands: {len(target_smiles):,} unique SMILES")
            print(f"  MF discrepancy: {len(removed_smiles) - len(target_smiles):,} SMILES")
            
            if 'zinc_original_smiles' in locals() and 'zinc_filtered_smiles' in locals():
                print(f"\n  Original ZINC: {len(zinc_original_smiles):,} unique SMILES")
                print(f"  Filtered ZINC: {len(zinc_filtered_smiles):,} unique SMILES")
                print(f"  Removed from ZINC: {len(zinc_removed_smiles):,} unique SMILES")
            
            if len(removed_smiles) > len(target_smiles) * 1.1:
                print("\n  ⚠️  MAJOR ISSUE: Far more molecules removed than expected!")
                print("     Possible causes:")
                print("     1. Target ligands file doesn't include all ABL1 ligands from ChEMBL")
                print("     2. Different filtering criteria used for MF vs target ligands")
                print("     3. Bug in the filtering logic")
            
        else:
            print(f"  ERROR: Could not find original MF cloud file!")
            return 1
    else:
        print(f"  ERROR: MF base directory doesn't exist: {precalc_mf_base_dir}")
        return 1
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
