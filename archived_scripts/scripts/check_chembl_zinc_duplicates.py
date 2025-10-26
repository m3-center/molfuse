#!/usr/bin/env python3
"""
Check for duplicate molecules between ChEMBL MF cloud and ZINC decoys.

This script identifies:
1. Exact SMILES matches between ChEMBL MF molecules and ZINC decoys
2. Canonical SMILES matches (handling tautomers/stereoisomers)
3. InChI key matches (structural duplicates)
4. Summary statistics and duplicate molecule list

Usage:
    python check_chembl_zinc_duplicates.py [--config experiment_config.json] [--mf-filter Transferase]
"""

import os
import sys
import json
import argparse
import pandas as pd
from collections import defaultdict
from datetime import datetime

# Optional: RDKit for canonical SMILES and InChI keys
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("⚠️  RDKit not available. Will only check exact SMILES matches.")
    print("   Install RDKit for more comprehensive duplicate detection.")

def load_config(config_path):
    """Load experiment configuration."""
    with open(config_path, 'r') as f:
        return json.load(f)

def load_zinc_smiles(zinc_path):
    """Load ZINC decoy SMILES strings."""
    print(f"\n📂 Loading ZINC decoys from: {zinc_path}")
    
    if not os.path.exists(zinc_path):
        print(f"❌ ZINC file not found: {zinc_path}")
        return None
    
    try:
        df = pd.read_csv(zinc_path)
        print(f"   Loaded {len(df)} ZINC molecules")
        
        # Find SMILES column
        smiles_col = None
        for col in ['SMILES', 'smiles', 'Smiles', 'canonical_smiles']:
            if col in df.columns:
                smiles_col = col
                break
        
        if smiles_col is None:
            print(f"   Columns: {list(df.columns)}")
            print(f"❌ No SMILES column found. Expected: SMILES, smiles, or canonical_smiles")
            return None
        
        print(f"   Using SMILES column: '{smiles_col}'")
        return df[smiles_col].dropna().unique()
        
    except Exception as e:
        print(f"❌ Error loading ZINC file: {e}")
        return None

def load_chembl_mf_smiles(mf_base_dir, mf_filter=None):
    """Load ChEMBL molecular function SMILES from feature/fingerprint files.
    
    Args:
        mf_base_dir: Base directory containing MF feature/fingerprint files
        mf_filter: Optional MF category filter (e.g., 'Transferase'). If provided,
                   only load molecules from this MF category.
    """
    print(f"\n📂 Loading ChEMBL MF molecules from: {mf_base_dir}")
    if mf_filter:
        print(f"   🔍 Filtering for molecular function: {mf_filter}")
    
    if not os.path.exists(mf_base_dir):
        print(f"❌ ChEMBL MF directory not found: {mf_base_dir}")
        return None, None
    
    all_smiles = []
    molecule_sources = defaultdict(list)  # Track which MF categories each molecule appears in
    
    # Find all molecular function CSV files
    mf_files = []
    for root, dirs, files in os.walk(mf_base_dir):
        for file in files:
            if file.endswith('_features.csv') or file.endswith('_fingerprints_ECFP4.csv'):
                file_path = os.path.join(root, file)
                
                # If filter is set, only include matching files
                if mf_filter:
                    if mf_filter.lower() in file.lower():
                        mf_files.append(file_path)
                else:
                    mf_files.append(file_path)
    
    if not mf_files:
        if mf_filter:
            print(f"❌ No ChEMBL MF files found matching filter: {mf_filter}")
        else:
            print(f"❌ No ChEMBL MF feature/fingerprint files found")
        return None, None
    
    print(f"   Found {len(mf_files)} MF category file(s)")
    
    # Load SMILES from each file
    for mf_file in sorted(mf_files):
        try:
            df = pd.read_csv(mf_file, low_memory=False)
            
            # Find SMILES column
            smiles_col = None
            for col in ['SMILES', 'smiles', 'Smiles', 'canonical_smiles']:
                if col in df.columns:
                    smiles_col = col
                    break
            
            if smiles_col is None:
                continue
            
            # Extract MF category from filename
            mf_category = os.path.basename(mf_file).replace('_features.csv', '').replace('_fingerprints_ECFP4.csv', '')
            
            smiles_list = df[smiles_col].dropna().tolist()
            all_smiles.extend(smiles_list)
            
            # Track source categories
            for smi in smiles_list:
                molecule_sources[smi].append(mf_category)
            
            print(f"   ✓ {mf_category}: {len(smiles_list)} molecules")
            
        except Exception as e:
            print(f"   ⚠️  Error loading {os.path.basename(mf_file)}: {e}")
            continue
    
    if not all_smiles:
        print(f"❌ No SMILES found in ChEMBL MF files")
        return None, None
    
    print(f"\n   Total ChEMBL MF molecules loaded: {len(all_smiles)}")
    print(f"   Unique ChEMBL MF molecules: {len(set(all_smiles))}")
    
    return all_smiles, molecule_sources

def canonicalize_smiles(smiles_list):
    """Convert SMILES to canonical form using RDKit."""
    if not RDKIT_AVAILABLE:
        return None
    
    print(f"\n🔬 Canonicalizing SMILES with RDKit...")
    canonical_map = {}
    invalid_count = 0
    
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                canonical = Chem.MolToSmiles(mol, canonical=True)
                canonical_map[smi] = canonical
            else:
                invalid_count += 1
        except:
            invalid_count += 1
    
    print(f"   ✓ Canonicalized {len(canonical_map)} SMILES")
    if invalid_count > 0:
        print(f"   ⚠️  {invalid_count} invalid SMILES skipped")
    
    return canonical_map

def generate_inchi_keys(smiles_list):
    """Generate InChI keys from SMILES using RDKit."""
    if not RDKIT_AVAILABLE:
        return None
    
    print(f"\n🔬 Generating InChI keys with RDKit...")
    inchi_map = {}
    invalid_count = 0
    
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                inchi_key = Chem.MolToInchiKey(mol)
                if inchi_key:
                    inchi_map[smi] = inchi_key
                else:
                    invalid_count += 1
            else:
                invalid_count += 1
        except:
            invalid_count += 1
    
    print(f"   ✓ Generated {len(inchi_map)} InChI keys")
    if invalid_count > 0:
        print(f"   ⚠️  {invalid_count} molecules failed InChI generation")
    
    return inchi_map

def find_duplicates(chembl_smiles, zinc_smiles, chembl_sources):
    """Find duplicate molecules between ChEMBL MF and ZINC."""
    
    results = {
        'exact_matches': [],
        'canonical_matches': [],
        'inchi_matches': []
    }
    
    # 1. Exact SMILES matches
    print(f"\n{'='*80}")
    print(f"DUPLICATE DETECTION")
    print(f"{'='*80}")
    
    print(f"\n1️⃣  Checking exact SMILES matches...")
    chembl_set = set(chembl_smiles)
    zinc_set = set(zinc_smiles)
    exact_duplicates = chembl_set.intersection(zinc_set)
    
    for dup_smi in exact_duplicates:
        results['exact_matches'].append({
            'smiles': dup_smi,
            'chembl_sources': chembl_sources.get(dup_smi, [])
        })
    
    print(f"   Found {len(exact_duplicates)} exact SMILES matches")
    
    # 2. Canonical SMILES matches (if RDKit available)
    if RDKIT_AVAILABLE:
        print(f"\n2️⃣  Checking canonical SMILES matches...")
        
        chembl_canonical = canonicalize_smiles(chembl_smiles)
        zinc_canonical = canonicalize_smiles(zinc_smiles)
        
        if chembl_canonical and zinc_canonical:
            # Create reverse mapping: canonical -> original SMILES
            chembl_canon_to_orig = defaultdict(list)
            for orig, canon in chembl_canonical.items():
                chembl_canon_to_orig[canon].append(orig)
            
            zinc_canon_to_orig = defaultdict(list)
            for orig, canon in zinc_canonical.items():
                zinc_canon_to_orig[canon].append(orig)
            
            # Find canonical matches
            chembl_canon_set = set(chembl_canonical.values())
            zinc_canon_set = set(zinc_canonical.values())
            canon_duplicates = chembl_canon_set.intersection(zinc_canon_set)
            
            # Exclude exact matches already found
            canon_only_duplicates = []
            for canon_smi in canon_duplicates:
                chembl_origs = chembl_canon_to_orig[canon_smi]
                zinc_origs = zinc_canon_to_orig[canon_smi]
                
                # Check if any pair is NOT an exact match
                is_new = False
                for c_orig in chembl_origs:
                    for z_orig in zinc_origs:
                        if c_orig != z_orig:  # Different original SMILES
                            is_new = True
                            break
                    if is_new:
                        break
                
                if is_new:
                    canon_only_duplicates.append({
                        'canonical_smiles': canon_smi,
                        'chembl_smiles': chembl_origs,
                        'zinc_smiles': zinc_origs,
                        'chembl_sources': [chembl_sources.get(smi, []) for smi in chembl_origs]
                    })
            
            results['canonical_matches'] = canon_only_duplicates
            print(f"   Found {len(canon_only_duplicates)} canonical matches (not exact)")
        
        # 3. InChI key matches (if RDKit available)
        print(f"\n3️⃣  Checking InChI key matches...")
        
        chembl_inchi = generate_inchi_keys(chembl_smiles)
        zinc_inchi = generate_inchi_keys(zinc_smiles)
        
        if chembl_inchi and zinc_inchi:
            # Create reverse mapping: inchi -> original SMILES
            chembl_inchi_to_orig = defaultdict(list)
            for orig, inchi in chembl_inchi.items():
                chembl_inchi_to_orig[inchi].append(orig)
            
            zinc_inchi_to_orig = defaultdict(list)
            for orig, inchi in zinc_inchi.items():
                zinc_inchi_to_orig[inchi].append(orig)
            
            # Find InChI matches
            chembl_inchi_set = set(chembl_inchi.values())
            zinc_inchi_set = set(zinc_inchi.values())
            inchi_duplicates = chembl_inchi_set.intersection(zinc_inchi_set)
            
            # Exclude matches already found
            inchi_only_duplicates = []
            for inchi_key in inchi_duplicates:
                chembl_origs = chembl_inchi_to_orig[inchi_key]
                zinc_origs = zinc_inchi_to_orig[inchi_key]
                
                # Check if this is a new match (not exact or canonical)
                is_new = True
                for c_orig in chembl_origs:
                    if c_orig in zinc_set:  # Already found as exact match
                        is_new = False
                        break
                
                if is_new:
                    inchi_only_duplicates.append({
                        'inchi_key': inchi_key,
                        'chembl_smiles': chembl_origs,
                        'zinc_smiles': zinc_origs,
                        'chembl_sources': [chembl_sources.get(smi, []) for smi in chembl_origs]
                    })
            
            results['inchi_matches'] = inchi_only_duplicates
            print(f"   Found {len(inchi_only_duplicates)} InChI matches (not exact/canonical)")
    
    return results

def generate_report(results, output_dir):
    """Generate comprehensive duplicate report."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    
    # Summary statistics
    total_duplicates = (len(results['exact_matches']) + 
                       len(results['canonical_matches']) + 
                       len(results['inchi_matches']))
    
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"\nTotal duplicates found: {total_duplicates}")
    print(f"  - Exact SMILES matches: {len(results['exact_matches'])}")
    print(f"  - Canonical matches (non-exact): {len(results['canonical_matches'])}")
    print(f"  - InChI key matches (non-exact/canonical): {len(results['inchi_matches'])}")
    
    # Export exact matches
    if results['exact_matches']:
        exact_df = pd.DataFrame(results['exact_matches'])
        exact_df['sources_str'] = exact_df['chembl_sources'].apply(lambda x: '; '.join(x) if x else '')
        exact_path = os.path.join(output_dir, f'exact_duplicates_{timestamp}.csv')
        exact_df.to_csv(exact_path, index=False)
        print(f"\n✅ Exact matches saved to: {exact_path}")
        
        # Show examples
        print(f"\nExample exact duplicates (first 5):")
        for i, dup in enumerate(results['exact_matches'][:5]):
            print(f"  {i+1}. {dup['smiles']}")
            print(f"     Sources: {', '.join(dup['chembl_sources'][:3])}")
    
    # Export canonical matches
    if results['canonical_matches']:
        canon_records = []
        for match in results['canonical_matches']:
            canon_records.append({
                'canonical_smiles': match['canonical_smiles'],
                'chembl_smiles': '; '.join(match['chembl_smiles']),
                'zinc_smiles': '; '.join(match['zinc_smiles']),
                'chembl_sources': '; '.join(['; '.join(src) for src in match['chembl_sources']])
            })
        canon_df = pd.DataFrame(canon_records)
        canon_path = os.path.join(output_dir, f'canonical_duplicates_{timestamp}.csv')
        canon_df.to_csv(canon_path, index=False)
        print(f"\n✅ Canonical matches saved to: {canon_path}")
    
    # Export InChI matches
    if results['inchi_matches']:
        inchi_records = []
        for match in results['inchi_matches']:
            inchi_records.append({
                'inchi_key': match['inchi_key'],
                'chembl_smiles': '; '.join(match['chembl_smiles']),
                'zinc_smiles': '; '.join(match['zinc_smiles']),
                'chembl_sources': '; '.join(['; '.join(src) for src in match['chembl_sources']])
            })
        inchi_df = pd.DataFrame(inchi_records)
        inchi_path = os.path.join(output_dir, f'inchi_duplicates_{timestamp}.csv')
        inchi_df.to_csv(inchi_path, index=False)
        print(f"\n✅ InChI matches saved to: {inchi_path}")
    
    # Summary report
    summary_path = os.path.join(output_dir, f'duplicate_summary_{timestamp}.txt')
    with open(summary_path, 'w') as f:
        f.write(f"ChEMBL MF vs ZINC Duplicate Analysis\n")
        f.write(f"{'='*80}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\nSUMMARY\n")
        f.write(f"{'='*80}\n")
        f.write(f"Total duplicates found: {total_duplicates}\n")
        f.write(f"  - Exact SMILES matches: {len(results['exact_matches'])}\n")
        f.write(f"  - Canonical matches (non-exact): {len(results['canonical_matches'])}\n")
        f.write(f"  - InChI key matches: {len(results['inchi_matches'])}\n")
        f.write(f"\n")
        
        if results['exact_matches']:
            f.write(f"\nEXACT MATCHES (first 20)\n")
            f.write(f"{'-'*80}\n")
            for i, dup in enumerate(results['exact_matches'][:20]):
                f.write(f"{i+1}. {dup['smiles']}\n")
                f.write(f"   Sources: {', '.join(dup['chembl_sources'])}\n")
    
    print(f"\n✅ Summary report saved to: {summary_path}")
    print(f"\n{'='*80}")

def main():
    parser = argparse.ArgumentParser(description='Check for duplicates between ChEMBL MF and ZINC')
    parser.add_argument('--config', default='experiment_config.json',
                       help='Path to experiment config file')
    parser.add_argument('--output', default='duplicate_analysis',
                       help='Output directory for reports')
    parser.add_argument('--mf-filter', dest='mf_filter', default='Transferase',
                       help='Molecular function to filter (e.g., Transferase, Oxidoreductase). Default: Transferase')
    args = parser.parse_args()
    
    print("="*80)
    print("ChEMBL MF vs ZINC Duplicate Detection")
    print("="*80)
    print(f"Config: {args.config}")
    print(f"Output: {args.output}")
    if args.mf_filter:
        print(f"MF Filter: {args.mf_filter}")
    
    # Load config
    if not os.path.exists(args.config):
        print(f"\n❌ Config file not found: {args.config}")
        sys.exit(1)
    
    config = load_config(args.config)
    global_settings = config['global_settings']
    
    # Load ZINC SMILES
    zinc_smiles = load_zinc_smiles(global_settings['zinc_full_csv_path'])
    if zinc_smiles is None or len(zinc_smiles) == 0:
        print(f"\n❌ Failed to load ZINC SMILES. Exiting.")
        sys.exit(1)
    
    # Load ChEMBL MF SMILES (with filter)
    chembl_smiles, chembl_sources = load_chembl_mf_smiles(
        global_settings['precalculated_chembl_mf_features_base_dir'],
        mf_filter=args.mf_filter
    )
    if chembl_smiles is None or len(chembl_smiles) == 0:
        print(f"\n❌ Failed to load ChEMBL MF SMILES. Exiting.")
        sys.exit(1)
    
    # Find duplicates
    results = find_duplicates(chembl_smiles, zinc_smiles, chembl_sources)
    
    # Generate report
    generate_report(results, args.output)
    
    print(f"\n✅ Analysis complete!")

if __name__ == '__main__':
    main()
