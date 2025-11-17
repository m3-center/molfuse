#!/usr/bin/env python3
"""
Verify MF Cloud Sizes: Phase 3 vs Phase 4

This script systematically counts MF cloud sizes to diagnose the discrepancy
between Phase 3 "full" (~96K) and Phase 4 Transferase (~187K).

Tests multiple hypotheses:
1. Raw CSV line counts (before any processing)
2. After deduplication by SMILES
3. After affinity cutoff filtering (100 nM vs 100,000 nM)
4. After target exclusion (P00519 actives removed)
5. After ZINC overlap removal
6. Final MF cloud size used for scoring

Usage:
    python scripts/verify_mf_cloud_sizes.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================================
# Helper Functions
# ============================================================================

def get_smiles_col(df: pd.DataFrame) -> str:
    """Get SMILES column name (prefer canonical_smiles)."""
    if "canonical_smiles" in df.columns:
        return "canonical_smiles"
    elif "SMILES" in df.columns:
        return "SMILES"
    else:
        raise ValueError("No SMILES column found")


def extract_accession(target: str) -> str:
    """Extract UniProt accession from target name (e.g., 'ABL1_P00519' -> 'P00519')."""
    m = re.search(r"_([A-Z0-9]{6})$", target)
    if m:
        return m.group(1)
    return target


def load_and_count(csv_path: Path, label: str) -> Dict:
    """Load CSV and return counts at various processing stages."""
    print(f"\n{'='*80}")
    print(f"Processing: {label}")
    print(f"File: {csv_path.name}")
    print(f"{'='*80}")
    
    results = {
        "label": label,
        "file": str(csv_path),
        "raw_lines": 0,
        "after_load": 0,
        "after_dedup": 0,
        "after_100nM_cutoff": 0,
        "after_100000nM_cutoff": 0,
        "unique_targets": 0,
        "unique_accessions": 0,
    }
    
    # Raw line count
    try:
        with open(csv_path, 'r') as f:
            results["raw_lines"] = sum(1 for _ in f) - 1  # Subtract header
        print(f"  Raw lines (excluding header): {results['raw_lines']:,}")
    except Exception as e:
        print(f"  ERROR reading file: {e}")
        return results
    
    # Load CSV
    try:
        # Selective dtype for metadata columns
        dtype_dict = {
            'Compound ChEMBL ID': str,
            'SMILES': str,
            'Target ChEMBL ID': str,
            'Target Name': str,
            'Activity Type': str,
            'Standard Value (nM)': float,
            'canonical_smiles': str,
            'target_chembl_id': str,
            'accession': str,
        }
        
        df = pd.read_csv(csv_path, dtype={k: v for k, v in dtype_dict.items() if k in pd.read_csv(csv_path, nrows=0).columns})
        results["after_load"] = len(df)
        print(f"  After load: {results['after_load']:,}")
        
        # Check for target/accession columns
        if "Target Name" in df.columns:
            unique_targets = df["Target Name"].nunique()
            results["unique_targets"] = int(unique_targets)
            print(f"  Unique targets: {unique_targets}")
            
            # Show sample targets
            sample_targets = df["Target Name"].value_counts().head(5)
            print(f"  Top 5 targets:")
            for target, count in sample_targets.items():
                print(f"    {target}: {count:,}")
        
        # Extract accessions
        if "Target Name" in df.columns:
            df["accession_extracted"] = df["Target Name"].apply(extract_accession)
            unique_accessions = df["accession_extracted"].nunique()
            results["unique_accessions"] = int(unique_accessions)
            print(f"  Unique accessions: {unique_accessions}")
            
            # Check if P00519 is present
            if "P00519" in df["accession_extracted"].values:
                p00519_count = (df["accession_extracted"] == "P00519").sum()
                print(f"  P00519 (ABL1) compounds: {p00519_count:,}")
        
        # Deduplication by SMILES
        smiles_col = get_smiles_col(df)
        df_dedup = df.drop_duplicates(subset=[smiles_col], keep="first")
        results["after_dedup"] = len(df_dedup)
        print(f"  After SMILES deduplication: {results['after_dedup']:,} ({results['after_load'] - results['after_dedup']:,} removed)")
        
        # Affinity cutoff: 100 nM (Phase 3 high-potency)
        if "Standard Value (nM)" in df_dedup.columns:
            affinity = pd.to_numeric(df_dedup["Standard Value (nM)"], errors="coerce")
            df_100nM = df_dedup[affinity <= 100.0]
            results["after_100nM_cutoff"] = len(df_100nM)
            print(f"  After 100 nM cutoff: {results['after_100nM_cutoff']:,}")
            
            # Affinity cutoff: 100,000 nM (Phase 4)
            df_100000nM = df_dedup[affinity <= 100000.0]
            results["after_100000nM_cutoff"] = len(df_100000nM)
            print(f"  After 100,000 nM cutoff: {results['after_100000nM_cutoff']:,}")
        
    except Exception as e:
        print(f"  ERROR processing file: {e}")
        import traceback
        traceback.print_exc()
    
    return results


# ============================================================================
# Main Analysis
# ============================================================================

def main():
    print("="*80)
    print("MF CLOUD SIZE VERIFICATION: Phase 3 vs Phase 4")
    print("="*80)
    
    base_dir = Path("output_recalculated_full_datasets/datasets_2d_all")
    
    # Test files
    test_files = [
        # Phase 3: Uses these files
        (base_dir / "KW-0808_Transferase_affinity_extracted_features.csv", 
         "Phase 3: Transferase Features"),
        (base_dir / "KW-0808_Transferase_affinity_extracted_fingerprints_ECFP4.csv", 
         "Phase 3: Transferase Fingerprints"),
        
        # Phase 4: Uses same files
        (base_dir / "KW-0808_Transferase_affinity_extracted_features.csv", 
         "Phase 4: Transferase Features"),
        (base_dir / "KW-0808_Transferase_affinity_extracted_fingerprints_ECFP4.csv", 
         "Phase 4: Transferase Fingerprints"),
        
        # Check other Phase 4 targets for comparison
        (base_dir / "KW-0049_Antioxidant_affinity_extracted_features.csv", 
         "Phase 4: Antioxidant Features"),
        (base_dir / "KW-0456_Lyase_affinity_extracted_features.csv", 
         "Phase 4: Lyase Features"),
        (base_dir / "KW-0560_Oxidoreductase_affinity_extracted_features.csv", 
         "Phase 4: Oxidoreductase Features"),
    ]
    
    all_results = []
    
    for csv_path, label in test_files:
        if not csv_path.exists():
            print(f"\n{'='*80}")
            print(f"SKIPPING: {label}")
            print(f"File not found: {csv_path}")
            print(f"{'='*80}")
            continue
        
        results = load_and_count(csv_path, label)
        all_results.append(results)
    
    # Summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Label':<40} {'Raw Lines':>12} {'After Dedup':>12} {'100nM':>12} {'100µM':>12} {'Targets':>8}")
    print("-"*80)
    
    for r in all_results:
        print(f"{r['label']:<40} {r['raw_lines']:>12,} {r['after_dedup']:>12,} "
              f"{r['after_100nM_cutoff']:>12,} {r['after_100000nM_cutoff']:>12,} {r['unique_targets']:>8}")
    
    # Key findings
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    # Find Phase 3 vs Phase 4 transferase entries
    phase3_features = [r for r in all_results if "Phase 3: Transferase Features" in r["label"]]
    phase4_features = [r for r in all_results if "Phase 4: Transferase Features" in r["label"]]
    
    if phase3_features and phase4_features:
        p3 = phase3_features[0]
        p4 = phase4_features[0]
        
        print(f"\nPhase 3 vs Phase 4 (Transferase/Features):")
        print(f"  Same source file: {Path(p3['file']).name == Path(p4['file']).name}")
        print(f"  Raw lines: {p3['raw_lines']:,} (Phase 3) vs {p4['raw_lines']:,} (Phase 4)")
        print(f"  After dedup: {p3['after_dedup']:,} (Phase 3) vs {p4['after_dedup']:,} (Phase 4)")
        print(f"  After 100 nM: {p3['after_100nM_cutoff']:,}")
        print(f"  After 100 µM: {p4['after_100000nM_cutoff']:,}")
        
        print(f"\nHypothesis Test:")
        print(f"  H1: Phase 3 used 100 nM cutoff → ~{p3['after_100nM_cutoff']:,} compounds")
        print(f"  H2: Phase 4 used 100 µM cutoff → ~{p4['after_100000nM_cutoff']:,} compounds")
        
        # Compare to reported values
        print(f"\nReported Values (from summary CSVs):")
        print(f"  Phase 3 'full' (umap/features): 96,657")
        print(f"  Phase 4 Transferase (umap/features): 187,034")
        
        print(f"\nDelta Analysis:")
        print(f"  Phase 3: Reported (96,657) vs Measured ({p3['after_100nM_cutoff']:,}) = "
              f"{96657 - p3['after_100nM_cutoff']:+,}")
        print(f"  Phase 4: Reported (187,034) vs Measured ({p4['after_100000nM_cutoff']:,}) = "
              f"{187034 - p4['after_100000nM_cutoff']:+,}")
    
    # Save results
    output_file = Path("reporting/mf_cloud_size_verification.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    print("="*80)


if __name__ == "__main__":
    main()
