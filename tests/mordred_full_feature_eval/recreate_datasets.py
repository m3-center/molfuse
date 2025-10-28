#!/usr/bin/env python3
"""
Recreate datasets structure with full Mordred descriptors (2D and 2D+3D).

This script processes ALL molecules from the original datasets/ directory and creates
two parallel structures with complete Mordred descriptor sets:
- datasets_2d_all/: Full 2D Mordred descriptors (1613 features)
- datasets_2d3d_all/: Full 2D+3D Mordred descriptors (1826 features)

The script:
1. Reads all molecules from datasets/molecular_function_features_fingerprints/
2. Computes full Mordred descriptors (2D and 2D+3D) for all molecules
3. Preserves exact folder structure (zinc/ subdirectory, all 58 KW files)
4. Validates recreated datasets and reports molecule losses

Note: This script computes descriptors on-the-fly and does NOT use a cache.

Author: Experimental script for dataset recreation
"""

from __future__ import annotations
import argparse
import gc
from pathlib import Path
from typing import Tuple, Optional, List, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

# RDKit imports
from rdkit import Chem
from rdkit.Chem import AllChem

# Mordred imports
# Mordred imports
from mordred import Calculator, descriptors


def smiles_to_rdkit_mol(s: str) -> Optional[Chem.Mol]:
    """Parse SMILES to RDKit molecule object."""
    try:
        m = Chem.MolFromSmiles(s)
        if m is None:
            return None
        Chem.SanitizeMol(m)
        return m
    except Exception:
        return None


def embed_3d(mol: Chem.Mol, seed: int = 42, max_attempts: int = 3) -> Optional[Chem.Mol]:
    """Generate 3D coordinates for molecule using RDKit ETKDG."""
    try:
        m = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        params.numThreads = 0
        for _ in range(max_attempts):
            if AllChem.EmbedMolecule(m, params) == 0:
                try:
                    # Optimize geometry
                    try:
                        AllChem.MMFFOptimizeMolecule(m)
                    except Exception:
                        AllChem.UFFOptimizeMolecule(m)
                    return m
                except Exception:
                    continue
        return None
    except Exception:
        return None


def compute_mordred_for_smiles(
    smiles_list: List[str],
    use_3d: bool,
    seed: int = 42
) -> Tuple[pd.DataFrame, List[str]]:
    """Compute Mordred descriptors for a list of SMILES.
    
    Returns:
        descriptor_df: DataFrame with 'smiles' column + descriptor columns
        failed_smiles: List of SMILES that failed computation
    """
    calc = Calculator(descriptors, ignore_3D=not use_3d)
    mols: List[Optional[Chem.Mol]] = []
    valid_smiles: List[str] = []
    failed_smiles: List[str] = []
    
    # Parse SMILES and generate 3D if needed
    for s in tqdm(smiles_list, desc=f"RDKit parse + {'3D' if use_3d else '2D'} prep", leave=False):
        m = smiles_to_rdkit_mol(s)
        if m is None:
            failed_smiles.append(s)
            continue
        
        if use_3d:
            m3d = embed_3d(m, seed=seed)
            if m3d is None:
                failed_smiles.append(s)
                continue
            mols.append(m3d)
            valid_smiles.append(s)
        else:
            mols.append(m)
            valid_smiles.append(s)
    
    # Compute descriptors
    if not mols:
        return pd.DataFrame(columns=["smiles"]), failed_smiles
    
    try:
        mordred_df = calc.pandas(mols)
    except Exception:
        # Fallback: compute per-molecule
        rows = []
        newly_failed = []
        for m, smi in tqdm(list(zip(mols, valid_smiles)), desc="Mordred per-mol (fallback)", leave=False):
            try:
                rows.append(calc(m))
            except Exception:
                rows.append({})
                newly_failed.append(smi)
        failed_smiles.extend(newly_failed)
        mordred_df = pd.DataFrame(rows)
    
    # Convert to numeric and filter
    for c in mordred_df.columns:
        mordred_df[c] = pd.to_numeric(mordred_df[c], errors="coerce")
    mordred_df = mordred_df.select_dtypes(include=[np.number])
    
    # Add SMILES column
    mordred_df.insert(0, "smiles", valid_smiles)
    
    # Remove duplicates
    mordred_df = mordred_df.drop_duplicates(subset=["smiles"], keep="last")
    
    return mordred_df, failed_smiles


def filter_fingerprints_by_features(
    fp_path: Path,
    feature_path: Path,
    output_path: Path,
    smiles_col: str = "SMILES"
) -> int:
    """Filter fingerprint file to only include molecules present in the feature file.
    
    Uses chunked processing for memory efficiency with large files.
    
    Args:
        fp_path: Path to original fingerprint file
        feature_path: Path to the created feature file (defines which molecules to keep)
        output_path: Path to save filtered fingerprint file
        smiles_col: Name of SMILES column (default: "SMILES")
    
    Returns:
        Number of molecules in filtered fingerprint file
    """
    if not fp_path.exists():
        print(f"      ⚠ Fingerprint file not found: {fp_path.name}")
        return 0
    
    if not feature_path.exists():
        print(f"      ⚠ Feature file not found: {feature_path.name}")
        return 0
    
    try:
        # Load SMILES from feature file (only SMILES column for memory efficiency)
        features_smiles = set(pd.read_csv(feature_path, usecols=[smiles_col], low_memory=False)[smiles_col].dropna().astype(str))
        
        # Process fingerprint file in chunks to avoid memory overflow
        chunk_size = 50_000
        total_kept = 0
        first_chunk = True
        
        for chunk in pd.read_csv(fp_path, chunksize=chunk_size, low_memory=False):
            if smiles_col not in chunk.columns:
                print(f"      ⚠ SMILES column '{smiles_col}' not found in {fp_path.name}")
                return 0
            
            # Filter chunk to only molecules present in feature file
            fp_filtered_chunk = chunk[chunk[smiles_col].isin(features_smiles)].copy()
            
            if not fp_filtered_chunk.empty:
                # Create output directory if needed
                if first_chunk:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Append to output file (write header only on first chunk)
                fp_filtered_chunk.to_csv(
                    output_path, 
                    mode='w' if first_chunk else 'a',
                    header=first_chunk,
                    index=False
                )
                total_kept += len(fp_filtered_chunk)
                first_chunk = False
            
            # Free memory
            del chunk, fp_filtered_chunk
            gc.collect()
        
        # If no molecules were kept, create empty file with header
        if total_kept == 0 and first_chunk:
            # Read just the header to create empty file
            header_df = pd.read_csv(fp_path, nrows=0)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            header_df.to_csv(output_path, index=False)
        
        return total_kept
    
    except Exception as e:
        print(f"      ✗ Error filtering fingerprints from {fp_path.name}: {e}")
        return 0


def verify_recreated_datasets(
    datasets_dir: Path,
    recreated_2d: Path,
    recreated_2d3d: Path,
    log_file: Path,
    initial_zinc_count: int = 0,
    initial_kw_count: int = 0
) -> bool:
    """Verify recreated datasets and generate comprehensive molecule count report.
    
    Returns True if all checks pass, False otherwise.
    """
    def log_print(msg: str):
        """Print to console and log file."""
        print(msg)
        with open(log_file, "a") as f:
            f.write(msg + "\n")
    
    log_print(f"\n{'='*80}")
    log_print("VERIFICATION: RECREATED DATASETS INTEGRITY CHECK")
    log_print(f"{'='*80}")
    
    all_passed = True
    
    # Check that directories exist
    if not recreated_2d.exists() or not recreated_2d3d.exists():
        log_print("✗ Recreated directories not found")
        return False
    
    # Verify ZINC file metadata (sample check)
    zinc_orig = datasets_dir / "zinc" / "zinc_acquirable_extracted_features.csv"
    zinc_2d = recreated_2d / "zinc" / "zinc_acquirable_extracted_features.csv"
    zinc_2d3d = recreated_2d3d / "zinc" / "zinc_acquirable_extracted_features.csv"
    
    if zinc_orig.exists() and zinc_2d.exists():
        df_orig = pd.read_csv(zinc_orig, nrows=100, low_memory=False)
        df_2d = pd.read_csv(zinc_2d, nrows=100)
        meta_cols = ['ZINC_ID', 'SMILES', 'LABEL', 'MANUFACTURER', 'TRANCHE']
        missing = [c for c in meta_cols if c not in df_2d.columns]
        if missing:
            log_print(f"✗ ZINC: Missing metadata columns: {missing}")
            all_passed = False
        else:
            log_print("✓ ZINC: Metadata columns preserved")
    
    # Verify sample KW file
    kw_files = sorted([f for f in datasets_dir.glob("KW-*.csv") 
                      if "_affinity_extracted_features.csv" in f.name])
    if kw_files:
        sample_kw = kw_files[0]
        kw_2d = recreated_2d / sample_kw.name
        if kw_2d.exists():
            df_orig = pd.read_csv(sample_kw, nrows=50, low_memory=False)
            df_2d = pd.read_csv(kw_2d, nrows=50)
            meta_cols = ['Compound ChEMBL ID', 'SMILES', 'accession']
            missing = [c for c in meta_cols if c not in df_2d.columns]
            if missing:
                log_print(f"✗ KW files: Missing metadata columns: {missing}")
                all_passed = False
            else:
                log_print("✓ KW files: Metadata columns preserved")
    
    # Comprehensive molecule count report
    log_print(f"\n{'='*80}")
    log_print("COMPREHENSIVE MOLECULE COUNT REPORT")
    log_print(f"{'='*80}")
    
    # Initial counts (what was requested for computation)
    log_print(f"\n[MOLECULES REQUESTED FOR COMPUTATION]")
    log_print(f"  ZINC molecules:  {initial_zinc_count:>10,}")
    log_print(f"  KW molecules:    {initial_kw_count:>10,}")
    log_print(f"  TOTAL requested: {initial_zinc_count + initial_kw_count:>10,}")
    
    # ZINC analysis
    log_print(f"\n[ZINC FILE - DETAILED ANALYSIS]")
    if zinc_orig.exists():
        df_orig = pd.read_csv(zinc_orig, usecols=['SMILES'], low_memory=False)
        smiles_orig = set(df_orig['SMILES'].dropna().astype(str).tolist())
        log_print(f"  Original:  {len(smiles_orig):>8,} unique SMILES")
        
        if zinc_2d.exists():
            df_2d = pd.read_csv(zinc_2d, usecols=['SMILES'])
            smiles_2d = set(df_2d['SMILES'].dropna().astype(str).tolist())
            pct_2d = len(smiles_2d)/len(smiles_orig)*100 if smiles_orig else 0
            log_print(f"  2D:        {len(smiles_2d):>8,} unique SMILES ({pct_2d:.2f}% of original)")
            
            if zinc_2d3d.exists():
                df_2d3d = pd.read_csv(zinc_2d3d, usecols=['SMILES'])
                smiles_2d3d = set(df_2d3d['SMILES'].dropna().astype(str).tolist())
                pct_2d3d = len(smiles_2d3d)/len(smiles_orig)*100 if smiles_orig else 0
                log_print(f"  2D+3D:     {len(smiles_2d3d):>8,} unique SMILES ({pct_2d3d:.2f}% of original)")
                
                overlap = smiles_2d & smiles_2d3d
                only_2d = smiles_2d - smiles_2d3d
                only_2d3d = smiles_2d3d - smiles_2d
                missing_both = len(smiles_orig) - len(smiles_2d | smiles_2d3d)
                
                log_print(f"\n  Overlap Analysis:")
                log_print(f"    Both 2D & 2D+3D:   {len(overlap):>8,} SMILES ({len(overlap)/len(smiles_orig)*100:.2f}% of original)")
                log_print(f"    Only in 2D:        {len(only_2d):>8,} SMILES (3D embedding failures)")
                log_print(f"    Only in 2D+3D:     {len(only_2d3d):>8,} SMILES (unexpected)")
                log_print(f"    Missing from both: {missing_both:>8,} SMILES")
                
                if only_2d3d:
                    log_print(f"    ⚠ WARNING: {len(only_2d3d)} molecules in 2D+3D but not 2D")
    
    # KW files analysis
    log_print(f"\n[KW MOLECULAR FUNCTION FILES]")
    kw_files = sorted([f for f in datasets_dir.glob("KW-*.csv") 
                      if "_affinity_extracted_features.csv" in f.name])
    
    total_orig = 0
    total_2d = 0
    total_2d3d = 0
    total_overlap = 0
    
    log_print(f"{'File':<60} {'Original':>10} {'2D':>10} {'2D+3D':>10} {'Overlap':>10}")
    log_print(f"{'-'*110}")
    
    for kw_file in kw_files:
        kw_name = kw_file.name
        kw_2d_path = recreated_2d / kw_name
        kw_2d3d_path = recreated_2d3d / kw_name
        
        df_orig = pd.read_csv(kw_file, usecols=['SMILES'], low_memory=False)
        smiles_orig = set(df_orig['SMILES'].dropna().astype(str).tolist())
        n_orig = len(smiles_orig)
        total_orig += n_orig
        
        n_2d = 0
        n_2d3d = 0
        n_overlap = 0
        
        if kw_2d_path.exists():
            df_2d = pd.read_csv(kw_2d_path, usecols=['SMILES'])
            smiles_2d = set(df_2d['SMILES'].dropna().astype(str).tolist())
            n_2d = len(smiles_2d)
            total_2d += n_2d
            
            if kw_2d3d_path.exists():
                df_2d3d = pd.read_csv(kw_2d3d_path, usecols=['SMILES'])
                smiles_2d3d = set(df_2d3d['SMILES'].dropna().astype(str).tolist())
                n_2d3d = len(smiles_2d3d)
                total_2d3d += n_2d3d
                
                overlap_set = smiles_2d & smiles_2d3d
                n_overlap = len(overlap_set)
                total_overlap += n_overlap
        
        display_name = kw_name[:57] + "..." if len(kw_name) > 60 else kw_name
        log_print(f"{display_name:<60} {n_orig:>10,} {n_2d:>10,} {n_2d3d:>10,} {n_overlap:>10,}")
    
    log_print(f"{'-'*110}")
    log_print(f"{'TOTAL':<60} {total_orig:>10,} {total_2d:>10,} {total_2d3d:>10,} {total_overlap:>10,}")
    
    # Summary statistics
    log_print(f"\n{'='*80}")
    log_print("SUMMARY STATISTICS: KW FILES")
    log_print(f"{'='*80}")
    
    if total_orig > 0:
        log_print(f"\nKW Files (Molecular Functions):")
        log_print(f"  Original total:        {total_orig:>10,} molecules")
        log_print(f"  2D total:              {total_2d:>10,} molecules ({total_2d/total_orig*100:.2f}%)")
        log_print(f"  2D+3D total:           {total_2d3d:>10,} molecules ({total_2d3d/total_orig*100:.2f}%)")
        log_print(f"  Both 2D & 2D+3D:       {total_overlap:>10,} molecules ({total_overlap/total_orig*100:.2f}%)")
        log_print(f"  Only in 2D:            {total_2d - total_overlap:>10,} molecules (3D failures)")
        only_2d3d = total_2d3d - total_overlap
        log_print(f"  Only in 2D+3D:         {only_2d3d:>10,} molecules (unexpected)")
        
        # Calculate molecules lost
        molecules_lost_2d = total_orig - total_2d
        molecules_lost_3d = total_2d - total_2d3d
        log_print(f"\n  MOLECULES LOST:")
        log_print(f"    Lost in 2D computation:  {molecules_lost_2d:>10,} molecules ({molecules_lost_2d/total_orig*100:.2f}%)")
        log_print(f"    Lost in 3D computation:  {molecules_lost_3d:>10,} molecules ({molecules_lost_3d/total_2d*100 if total_2d > 0 else 0:.2f}% of 2D)")
        
        if only_2d3d > 0:
            log_print(f"\n  ⚠ WARNING: {only_2d3d} molecules present in 2D+3D but NOT in 2D")
    
    # OVERALL SUCCESS RATES (requested vs delivered)
    log_print(f"\n{'='*80}")
    log_print("OVERALL SUCCESS RATES: REQUESTED vs DELIVERED")
    log_print(f"{'='*80}")
    
    # Get ZINC counts from recreated files
    zinc_2d_count = 0
    zinc_2d3d_count = 0
    if zinc_2d.exists():
        zinc_2d_count = sum(1 for _ in open(zinc_2d)) - 1
    if zinc_2d3d.exists():
        zinc_2d3d_count = sum(1 for _ in open(zinc_2d3d)) - 1
    
    # Calculate total delivered
    total_requested = initial_zinc_count + initial_kw_count
    total_delivered_2d = zinc_2d_count + total_2d
    total_delivered_2d3d = zinc_2d3d_count + total_2d3d
    
    log_print(f"\n2D DESCRIPTORS:")
    log_print(f"  Molecules requested:  {total_requested:>10,}")
    log_print(f"  Molecules delivered:  {total_delivered_2d:>10,}")
    log_print(f"  Success rate:         {total_delivered_2d/total_requested*100 if total_requested > 0 else 0:>10.2f}%")
    log_print(f"  Molecules lost:       {total_requested - total_delivered_2d:>10,} ({(total_requested - total_delivered_2d)/total_requested*100 if total_requested > 0 else 0:.2f}%)")
    
    log_print(f"\n  Breakdown by source:")
    log_print(f"    ZINC requested:     {initial_zinc_count:>10,}")
    log_print(f"    ZINC delivered:     {zinc_2d_count:>10,} ({zinc_2d_count/initial_zinc_count*100 if initial_zinc_count > 0 else 0:.2f}%)")
    log_print(f"    KW requested:       {initial_kw_count:>10,}")
    log_print(f"    KW delivered:       {total_2d:>10,} ({total_2d/initial_kw_count*100 if initial_kw_count > 0 else 0:.2f}%)")
    
    log_print(f"\n2D+3D DESCRIPTORS:")
    log_print(f"  Molecules requested:  {total_requested:>10,}")
    log_print(f"  Molecules delivered:  {total_delivered_2d3d:>10,}")
    log_print(f"  Success rate:         {total_delivered_2d3d/total_requested*100 if total_requested > 0 else 0:>10.2f}%")
    log_print(f"  Molecules lost:       {total_requested - total_delivered_2d3d:>10,} ({(total_requested - total_delivered_2d3d)/total_requested*100 if total_requested > 0 else 0:.2f}%)")
    
    log_print(f"\n  Breakdown by source:")
    log_print(f"    ZINC requested:     {initial_zinc_count:>10,}")
    log_print(f"    ZINC delivered:     {zinc_2d3d_count:>10,} ({zinc_2d3d_count/initial_zinc_count*100 if initial_zinc_count > 0 else 0:.2f}%)")
    log_print(f"    KW requested:       {initial_kw_count:>10,}")
    log_print(f"    KW delivered:       {total_2d3d:>10,} ({total_2d3d/initial_kw_count*100 if initial_kw_count > 0 else 0:.2f}%)")
    
    log_print(f"\n3D EMBEDDING FAILURE RATE:")
    if total_delivered_2d > 0:
        failures_3d = total_delivered_2d - total_delivered_2d3d
        log_print(f"  2D molecules:         {total_delivered_2d:>10,}")
        log_print(f"  2D+3D molecules:      {total_delivered_2d3d:>10,}")
        log_print(f"  3D failures:          {failures_3d:>10,} ({failures_3d/total_delivered_2d*100:.2f}% of 2D)")
    
    log_print(f"\n{'='*80}")
    if all_passed:
        log_print("✓ ALL VERIFICATION CHECKS PASSED")
        log_print("\n⚠ IMPORTANT: 2D and 2D+3D datasets contain DIFFERENT molecules")
        log_print("  • 2D+3D excludes molecules where 3D conformer generation failed")
        log_print("  • Use the overlap counts above for fair performance comparisons")
    else:
        log_print("✗ SOME VERIFICATION CHECKS FAILED")
    log_print(f"{'='*80}")
    
    return all_passed


def recreate_datasets_structure(
    base_dir: Path,
    output_dir: Path,
    seed: int = 42,
    limit_zinc: Optional[int] = None,
    limit_kw: Optional[int] = None
) -> tuple[int, int]:
    """Recreate datasets/molecular_function_features_fingerprints/ structure with full Mordred descriptors.
    
    Creates two parallel directory structures:
    - output_dir/datasets_2d_all/  (full 2D Mordred features)
    - output_dir/datasets_2d3d_all/  (full 2D+3D Mordred features)
    
    Each contains:
    - All KW-XXXX_*_affinity_extracted_features.csv files with metadata + new descriptors
    - zinc/zinc_acquirable_extracted_features.csv with metadata + new descriptors
    
    NOTE: Descriptors are computed on-the-fly for all molecules. Failed computations are excluded.
    
    Args:
        base_dir: Base directory containing datasets/
        output_dir: Output directory for recreated datasets
        seed: Random seed for 3D conformer generation
        limit_zinc: If set, only process first N molecules from ZINC (for testing)
        limit_kw: If set, only process first N molecules from each KW file (for testing)
    
    Returns:
        (initial_zinc_count, initial_kw_count): Number of molecules in original datasets
    """
    datasets_dir = base_dir / "datasets" / "molecular_function_features_fingerprints"
    
    print(f"\n{'='*80}")
    print("RECREATING DATASETS STRUCTURE WITH FULL MORDRED DESCRIPTORS")
    print(f"{'='*80}")
    
    # Create output directories
    out_2d = output_dir / "datasets_2d_all"
    out_2d3d = output_dir / "datasets_2d3d_all"
    (out_2d / "zinc").mkdir(parents=True, exist_ok=True)
    (out_2d3d / "zinc").mkdir(parents=True, exist_ok=True)
    
    # Get all KW feature files (exclude fingerprint files)
    kw_files = sorted([f for f in datasets_dir.glob("KW-*.csv") 
                      if "_affinity_extracted_features.csv" in f.name])
    
    # Track initial molecule counts
    initial_zinc_count = 0
    initial_kw_count = 0
    
    # Process ZINC first (in chunks to avoid memory issues)
    print("\n[1/2] Processing ZINC...")
    zinc_orig_path = datasets_dir / "zinc" / "zinc_acquirable_extracted_features.csv"
    if zinc_orig_path.exists():
        # First pass: count total molecules
        initial_zinc_count = sum(1 for _ in open(zinc_orig_path)) - 1  # subtract header
        
        if limit_zinc:
            print(f"  Total molecules in ZINC: {initial_zinc_count:,} (limiting to {limit_zinc:,} for testing)")
            molecules_to_process = min(limit_zinc, initial_zinc_count)
        else:
            print(f"  Total molecules in ZINC: {initial_zinc_count:,}")
            molecules_to_process = initial_zinc_count
        
        # Process in chunks to avoid memory overflow
        chunk_size = 100_000
        zinc_meta_cols = ["ZINC_ID", "SMILES", "LABEL", "MANUFACTURER", "TRANCHE"]
        
        out_path_2d = out_2d / "zinc" / "zinc_acquirable_extracted_features.csv"
        out_path_2d3d = out_2d3d / "zinc" / "zinc_acquirable_extracted_features.csv"
        
        total_2d = 0
        total_2d3d = 0
        total_processed = 0
        first_chunk = True
        
        print(f"  Processing in {chunk_size:,} molecule chunks...")
        
        for chunk_idx, zinc_chunk in enumerate(pd.read_csv(zinc_orig_path, chunksize=chunk_size, low_memory=False)):
            # Check if we've reached the limit
            if limit_zinc and total_processed >= molecules_to_process:
                break
            
            # Trim chunk if it would exceed limit
            if limit_zinc:
                remaining = molecules_to_process - total_processed
                if len(zinc_chunk) > remaining:
                    zinc_chunk = zinc_chunk.head(remaining)
            
            print(f"    Chunk {chunk_idx + 1}: {len(zinc_chunk):,} molecules", end=" → ", flush=True)
            total_processed += len(zinc_chunk)
            
            # Extract metadata and SMILES
            zinc_meta = zinc_chunk[[c for c in zinc_meta_cols if c in zinc_chunk.columns]].copy()
            smiles_list = zinc_chunk["SMILES"].dropna().astype(str).tolist()
            
            # Free the original chunk immediately
            del zinc_chunk
            gc.collect()
            
            # Compute 2D descriptors
            desc_2d, failed_2d = compute_mordred_for_smiles(smiles_list, use_3d=False, seed=seed)
            
            # Merge ALL metadata with descriptors (left merge keeps all original molecules)
            zinc_2d = zinc_meta.merge(desc_2d, left_on="SMILES", right_on="smiles", how="left")
            if "smiles" in zinc_2d.columns:
                zinc_2d = zinc_2d.drop(columns=["smiles"])
            
            # Only keep rows that have descriptors
            if not desc_2d.empty:
                first_desc_col = [c for c in desc_2d.columns if c != "smiles"][0]
                zinc_2d = zinc_2d.dropna(subset=[first_desc_col])
            else:
                # No descriptors computed - empty dataframe with columns
                zinc_2d = zinc_2d.iloc[:0]
            
            # Append to file (write header only on first chunk)
            if not zinc_2d.empty or first_chunk:
                zinc_2d.to_csv(out_path_2d, mode='w' if first_chunk else 'a', 
                              header=first_chunk, index=False)
            total_2d += len(zinc_2d)
            del zinc_2d, desc_2d
            gc.collect()
            
            # Compute 2D+3D descriptors
            desc_2d3d, failed_3d = compute_mordred_for_smiles(smiles_list, use_3d=True, seed=seed)
            
            # Merge ALL metadata with descriptors (left merge keeps all original molecules)
            zinc_2d3d = zinc_meta.merge(desc_2d3d, left_on="SMILES", right_on="smiles", how="left")
            if "smiles" in zinc_2d3d.columns:
                zinc_2d3d = zinc_2d3d.drop(columns=["smiles"])
            
            # Only keep rows that have descriptors
            if not desc_2d3d.empty:
                first_desc_col = [c for c in desc_2d3d.columns if c != "smiles"][0]
                zinc_2d3d = zinc_2d3d.dropna(subset=[first_desc_col])
            else:
                # No descriptors computed - empty dataframe with columns
                zinc_2d3d = zinc_2d3d.iloc[:0]
            
            if not zinc_2d3d.empty or first_chunk:
                zinc_2d3d.to_csv(out_path_2d3d, mode='w' if first_chunk else 'a',
                                header=first_chunk, index=False)
            total_2d3d += len(zinc_2d3d)
            del zinc_2d3d, desc_2d3d
            gc.collect()
            
            print(f"2D: {total_2d:,}, 2D+3D: {total_2d3d:,}")
            
            first_chunk = False
            del zinc_meta, smiles_list
            gc.collect()
        
        print(f"  ✓ ZINC complete: 2D={total_2d:,}, 2D+3D={total_2d3d:,}")
        
        # Now filter and recreate ZINC fingerprint files
        print(f"\n  Processing ZINC fingerprints...")
        zinc_fp_orig = datasets_dir / "zinc" / "zinc_acquirable_extracted_fingerprints_ECFP4.csv"
        
        # Filter fingerprints for 2D
        out_fp_2d = out_2d / "zinc" / "zinc_acquirable_extracted_fingerprints_ECFP4.csv"
        n_fp_2d = filter_fingerprints_by_features(zinc_fp_orig, out_path_2d, out_fp_2d, smiles_col="SMILES")
        print(f"    ✓ 2D fingerprints: {n_fp_2d:,} molecules")
        
        # Filter fingerprints for 2D+3D
        out_fp_2d3d = out_2d3d / "zinc" / "zinc_acquirable_extracted_fingerprints_ECFP4.csv"
        n_fp_2d3d = filter_fingerprints_by_features(zinc_fp_orig, out_path_2d3d, out_fp_2d3d, smiles_col="SMILES")
        print(f"    ✓ 2D+3D fingerprints: {n_fp_2d3d:,} molecules")
        
    else:
        print(f"  ⚠ ZINC file not found: {zinc_orig_path}")
    
    # Process all KW files
    print(f"\n[2/2] Processing {len(kw_files)} KW molecular function files...")
    for i, kw_path in enumerate(kw_files, 1):
        kw_name = kw_path.name
        print(f"\n  [{i}/{len(kw_files)}] {kw_name}")
        
        try:
            # Apply limit to KW files if specified
            if limit_kw:
                kw_orig = pd.read_csv(kw_path, nrows=limit_kw, low_memory=False)
                initial_kw_count += len(kw_orig)
                print(f"    Loaded: {len(kw_orig):,} molecules (limited to {limit_kw:,} for testing)", flush=True)
            else:
                kw_orig = pd.read_csv(kw_path, low_memory=False)
                initial_kw_count += len(kw_orig)
                print(f"    Loaded: {len(kw_orig):,} molecules", flush=True)
            
            # Metadata columns to preserve (ChEMBL/MF structure)
            meta_cols = [
                "Compound ChEMBL ID", "SMILES", "Target ChEMBL ID", "Target Name",
                "Activity Type", "Standard Value (nM)", "target_chembl_id", "accession"
            ]
            kw_meta = kw_orig[[c for c in meta_cols if c in kw_orig.columns]].copy()
            
            # Get SMILES list
            smiles_col = "SMILES" if "SMILES" in kw_orig.columns else "canonical_smiles"
            if smiles_col not in kw_orig.columns:
                print(f"    ⚠ No SMILES column found, skipping")
                del kw_orig
                gc.collect()
                continue
            
            smiles_list = kw_orig[smiles_col].dropna().astype(str).tolist()
            
            # Free original dataframe immediately
            del kw_orig
            gc.collect()
            
            # Compute 2D descriptors
            desc_2d, failed_2d = compute_mordred_for_smiles(smiles_list, use_3d=False, seed=seed)
            
            # Merge ALL metadata with descriptors (left merge keeps all original molecules)
            kw_2d = kw_meta.merge(desc_2d, left_on=smiles_col, right_on="smiles", how="left")
            if "smiles" in kw_2d.columns:
                kw_2d = kw_2d.drop(columns=["smiles"])
            
            # Only keep rows that have descriptors
            if not desc_2d.empty:
                first_desc_col = [c for c in desc_2d.columns if c != "smiles"][0]
                kw_2d = kw_2d.dropna(subset=[first_desc_col])
            else:
                kw_2d = kw_2d.iloc[:0]  # Empty dataframe with columns
            
            out_path_2d = out_2d / kw_name
            kw_2d.to_csv(out_path_2d, index=False)
            print(f"    ✓ 2D: {len(kw_2d):,} molecules")
            del kw_2d, desc_2d
            gc.collect()
            
            # Compute 2D+3D descriptors
            desc_2d3d, failed_3d = compute_mordred_for_smiles(smiles_list, use_3d=True, seed=seed)
            
            # Merge ALL metadata with descriptors (left merge keeps all original molecules)
            kw_2d3d = kw_meta.merge(desc_2d3d, left_on=smiles_col, right_on="smiles", how="left")
            if "smiles" in kw_2d3d.columns:
                kw_2d3d = kw_2d3d.drop(columns=["smiles"])
            
            # Only keep rows that have descriptors
            if not desc_2d3d.empty:
                first_desc_col = [c for c in desc_2d3d.columns if c != "smiles"][0]
                kw_2d3d = kw_2d3d.dropna(subset=[first_desc_col])
            else:
                kw_2d3d = kw_2d3d.iloc[:0]  # Empty dataframe with columns
            
            out_path_2d3d = out_2d3d / kw_name
            kw_2d3d.to_csv(out_path_2d3d, index=False)
            print(f"    ✓ 2D+3D: {len(kw_2d3d):,} molecules")
            del kw_2d3d, desc_2d3d
            gc.collect()
            
            # Process fingerprint file for this KW
            # Derive fingerprint filename from feature filename
            fp_name = kw_name.replace("_extracted_features.csv", "_extracted_fingerprints_ECFP4.csv")
            kw_fp_orig = datasets_dir / fp_name
            
            if kw_fp_orig.exists():
                # Filter fingerprints for 2D
                out_fp_2d = out_2d / fp_name
                n_fp_2d = filter_fingerprints_by_features(kw_fp_orig, out_path_2d, out_fp_2d, smiles_col=smiles_col)
                
                # Filter fingerprints for 2D+3D
                out_fp_2d3d = out_2d3d / fp_name
                n_fp_2d3d = filter_fingerprints_by_features(kw_fp_orig, out_path_2d3d, out_fp_2d3d, smiles_col=smiles_col)
                
                print(f"    ✓ Fingerprints: 2D={n_fp_2d:,}, 2D+3D={n_fp_2d3d:,}")
            else:
                print(f"    ⚠ Fingerprint file not found: {fp_name}")
            
            # Clean up all variables for this file
            del kw_meta, smiles_list
            gc.collect()
            
        except Exception as e:
            print(f"    ✗ Error processing {kw_name}: {e}")
            gc.collect()
            continue
    
    print(f"\n{'='*80}")
    print("DATASETS STRUCTURE RECREATION COMPLETED")
    print(f"{'='*80}")
    print(f"\n✓ Full 2D descriptors:    {out_2d}")
    print(f"✓ Full 2D+3D descriptors: {out_2d3d}")
    print(f"\nInitial molecule counts:")
    print(f"  • ZINC:     {initial_zinc_count:>10,} molecules")
    print(f"  • KW files: {initial_kw_count:>10,} molecules")
    print(f"\nKey properties of recreated files:")
    print(f"  • Metadata columns preserved from original datasets/")
    print(f"  • Row order preserved (same order as original for molecules with computed descriptors)")
    print(f"  • Feature columns replaced with full Mordred descriptor sets")
    print(f"  • Fingerprint files filtered to match feature files (ECFP4, no recalculation)")
    print(f"  • Only molecules with successfully computed descriptors are included")
    print(f"\nUsage: Update phase1.py configs to point to these directories")
    print(f"  Example (2D):")
    print(f"    mf_features_csv: {(out_2d / 'KW-0808_Transferase_affinity_extracted_features.csv').as_posix()}")
    print(f"    zinc_features_csv: {(out_2d / 'zinc' / 'zinc_acquirable_extracted_features.csv').as_posix()}")
    print(f"  Example (2D+3D):")
    print(f"    mf_features_csv: {(out_2d3d / 'KW-0808_Transferase_affinity_extracted_features.csv').as_posix()}")
    print(f"    zinc_features_csv: {(out_2d3d / 'zinc' / 'zinc_acquirable_extracted_features.csv').as_posix()}")
    
    return initial_zinc_count, initial_kw_count


def main():
    parser = argparse.ArgumentParser(
        description="Recreate dataset structure with full Mordred descriptors (computed on-the-fly)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Recreate datasets (full run)
  python recreate_datasets.py --output_dir output_full
  
  # Test with small subset
  python recreate_datasets.py --output_dir output_test --limit-zinc 1000 --limit-kw 100
  
  # With custom random seed for 3D conformer generation
  python recreate_datasets.py --output_dir output_full --seed 123
        """
    )
    
    parser.add_argument(
        "--base_dir",
        type=str,
        default=".",
        help="Base directory containing datasets/ (default: current directory)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for recreated datasets"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for 3D conformer generation (default: 42)"
    )
    
    parser.add_argument(
        "--limit-zinc",
        type=int,
        default=None,
        help="Limit number of ZINC molecules to process for testing (default: process all)"
    )
    
    parser.add_argument(
        "--limit-kw",
        type=int,
        default=None,
        help="Limit number of molecules per KW file for testing (default: process all)"
    )
    
    args = parser.parse_args()
    
    # Setup paths
    base_dir = Path(args.base_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Recreate datasets (compute descriptors on-the-fly)
    initial_zinc, initial_kw = recreate_datasets_structure(
        base_dir=base_dir,
        output_dir=out_dir,
        seed=args.seed,
        limit_zinc=args.limit_zinc,
        limit_kw=args.limit_kw
    )
    
    # Verify and report
    log_file = out_dir / "verification_report.log"
    log_file.write_text("")  # Clear log file
    
    datasets_dir = base_dir / "datasets" / "molecular_function_features_fingerprints"
    recreated_2d = out_dir / "datasets_2d_all"
    recreated_2d3d = out_dir / "datasets_2d3d_all"
    
    verification_passed = verify_recreated_datasets(
        datasets_dir=datasets_dir,
        recreated_2d=recreated_2d,
        recreated_2d3d=recreated_2d3d,
        log_file=log_file,
        initial_zinc_count=initial_zinc,
        initial_kw_count=initial_kw
    )
    
    print(f"\n✓ Verification report saved to: {log_file}")
    
    if not verification_passed:
        print(f"\n⚠ Warning: Some verification checks failed. Review the log file.")


if __name__ == "__main__":
    main()
