#!/usr/bin/env python3
"""
Analyze MF Cloud Sizes for Phase 4 Targets

For each of the 8 Phase 4 target keyword collections, reports:
- MF cloud size at different affinity cutoffs: [100, 1000, 10000, 100000] nM
- This helps understand which targets have sufficient high-potency data

Usage:
    python scripts/analyze_phase4_mf_cloud_sizes.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def analyze_mf_cloud_by_cutoff(
    mf_csv_path: Path,
    target_protein: str,
    cutoffs: List[int] = [100, 1000, 10000, 100000]
) -> Dict[int, int]:
    """
    Load MF cloud CSV and count compounds at each affinity cutoff.
    
    Args:
        mf_csv_path: Path to MF cloud features/fingerprints CSV
        target_protein: Target protein ID to exclude from MF cloud
        cutoffs: List of affinity cutoffs in nM
    
    Returns:
        Dictionary mapping cutoff → count of compounds
    """
    # Load MF cloud
    df_mf = pd.read_csv(mf_csv_path)
    
    # Identify activity column
    activity_col = "Standard Value (nM)"
    if activity_col not in df_mf.columns:
        raise ValueError(f"Expected column '{activity_col}' not found in {mf_csv_path}")
    
    # Identify target column (could be "Target ChEMBL ID" or "Target Name")
    target_col = None
    for possible_col in ["Target ChEMBL ID", "Target Name", "Target_ChEMBL_ID", "Target_Name", "target", "Target"]:
        if possible_col in df_mf.columns:
            target_col = possible_col
            break
    
    if target_col is None:
        raise ValueError(f"No target column found in {mf_csv_path}. Available columns: {df_mf.columns.tolist()}")
    
    # Remove target protein from MF cloud (exclude actives)
    df_mf_clean = df_mf[df_mf[target_col] != target_protein].copy()
    
    # Convert affinity to numeric, drop NaN
    df_mf_clean[activity_col] = pd.to_numeric(df_mf_clean[activity_col], errors="coerce")
    df_mf_clean = df_mf_clean.dropna(subset=[activity_col])
    
    # Count compounds at each cutoff
    sizes = {}
    for cutoff in cutoffs:
        count = len(df_mf_clean[df_mf_clean[activity_col] <= cutoff])
        sizes[cutoff] = count
    
    return sizes


def main():
    print("="*80)
    print("PHASE 4 MF CLOUD SIZE ANALYSIS")
    print("="*80)
    print()
    
    # Load one config from each target to get metadata
    config_dir = Path("configs/molfuse_phase4_grid")
    
    if not config_dir.exists():
        raise FileNotFoundError(f"Config directory not found: {config_dir}")
    
    # Find all unique target_short values
    configs = list(config_dir.glob("*.json"))
    
    if len(configs) == 0:
        raise ValueError(f"No configs found in {config_dir}")
    
    # Parse configs and group by target_short
    target_info = {}
    for config_path in configs:
        with config_path.open("r") as f:
            data = json.load(f)
        
        target_short = data.get("target_short")
        if target_short and target_short not in target_info:
            target_info[target_short] = {
                "target": data.get("target"),
                "target_kw": data.get("target_kw"),
                "mf_features_csv": data.get("mf_features_csv"),
                "mf_fingerprints_csv": data.get("mf_fingerprints_csv"),
            }
    
    print(f"Found {len(target_info)} unique targets:")
    for target_short in sorted(target_info.keys()):
        print(f"  - {target_short} (target={target_info[target_short]['target']}, kw={target_info[target_short]['target_kw']})")
    print()
    
    # Affinity cutoffs to test
    cutoffs = [100, 1000, 10000, 100000]
    
    # Analyze each target
    results = []
    
    for target_short in sorted(target_info.keys()):
        info = target_info[target_short]
        target_protein = info["target"]
        mf_csv_path = Path(info["mf_features_csv"])
        
        print(f"Analyzing: {target_short} (target={target_protein})")
        print(f"  MF CSV: {mf_csv_path}")
        
        if not mf_csv_path.exists():
            print(f"  ⚠️  WARNING: MF CSV not found, skipping")
            print()
            continue
        
        try:
            sizes = analyze_mf_cloud_by_cutoff(mf_csv_path, target_protein, cutoffs)
            
            # Print results for this target
            print(f"  MF Cloud Sizes (excluding {target_protein}):")
            for cutoff in cutoffs:
                count = sizes[cutoff]
                status = "✓" if count > 0 else "✗ EMPTY"
                print(f"    {status:8s} ≤{cutoff:6d} nM: {count:8d} compounds")
            
            # Store for summary table
            results.append({
                "target_short": target_short,
                "target": target_protein,
                "target_kw": info["target_kw"],
                "mf_100nM": sizes[100],
                "mf_1000nM": sizes[1000],
                "mf_10000nM": sizes[10000],
                "mf_100000nM": sizes[100000],
            })
            
        except Exception as e:
            print(f"  ⚠️  ERROR: {e}")
        
        print()
    
    # Summary table
    print("="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print()
    
    df_summary = pd.DataFrame(results)
    
    # Reorder columns
    df_summary = df_summary[[
        "target_short", "target", "target_kw",
        "mf_100nM", "mf_1000nM", "mf_10000nM", "mf_100000nM"
    ]]
    
    # Sort by target_short
    df_summary = df_summary.sort_values("target_short")
    
    # Print as formatted table
    print(df_summary.to_string(index=False))
    print()
    
    # Save to CSV
    output_path = Path("reporting/phase4_mf_cloud_sizes_by_cutoff.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_summary.to_csv(output_path, index=False)
    print(f"Saved summary to: {output_path}")
    print()
    
    # Check for empty clouds at 100 nM cutoff
    empty_at_100 = df_summary[df_summary["mf_100nM"] == 0]
    if len(empty_at_100) > 0:
        print("⚠️  WARNING: The following targets have EMPTY MF clouds at ≤100 nM cutoff:")
        for _, row in empty_at_100.iterrows():
            print(f"    - {row['target_short']} ({row['target']})")
        print()
        print("This explains why Phase 4 used 100,000 nM cutoff instead of 100 nM!")
    else:
        print("✓ All targets have non-empty MF clouds at ≤100 nM cutoff")
    
    print()
    print("="*80)


if __name__ == "__main__":
    main()
