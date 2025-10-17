#!/usr/bin/env python3
"""
Debug script to test a single run directory and see what files are available.
"""
import os
import sys
import glob
import pandas as pd

def inspect_run_directory(run_dir):
    """Inspect a single run directory to understand its structure."""
    print(f"\n{'='*80}")
    print(f"Inspecting: {os.path.basename(run_dir)}")
    print(f"{'='*80}\n")
    
    # Check for config
    config_path = os.path.join(run_dir, 'run_config.json')
    print(f"Config file exists: {os.path.exists(config_path)}")
    
    # Look for ranked files
    ranked_patterns = [
        os.path.join(run_dir, "*/results/*/dim_*/*/*-RANKED.csv"),
        os.path.join(run_dir, "*/results/*/dim_*/*/*-FEATURES.csv"),
        os.path.join(run_dir, "*/results/*/dim_*/*/*-FINGERPRINTS.csv"),
    ]
    ranked_files = []
    for pattern in ranked_patterns:
        ranked_files.extend(glob.glob(pattern))
    # Filter out metrics files
    ranked_files = [f for f in ranked_files if 'metrics' not in f.lower() and 'distances' not in f.lower()]
    
    print(f"\nRanked files found: {len(ranked_files)}")
    if ranked_files:
        print(f"  First: {ranked_files[0]}")
        
        # Check columns
        try:
            df = pd.read_csv(ranked_files[0], nrows=5)
            print(f"  Columns: {list(df.columns)}")
            print(f"  Has 'Standard Value (nM)': {'Standard Value (nM)' in df.columns}")
            print(f"  Has 'RANKING': {'RANKING' in df.columns}")
            print(f"  Has 'TYPE': {'TYPE' in df.columns}")
            print(f"  Sample data:")
            print(df.head(2))
        except Exception as e:
            print(f"  Error reading: {e}")
    
    # Look for detailed distances files
    dist_pattern = os.path.join(run_dir, "*/results/*/dim_*/*/*detailed_active_distances.csv")
    dist_files = glob.glob(dist_pattern)
    print(f"\nDetailed distances files found: {len(dist_files)}")
    if dist_files:
        print(f"  First: {dist_files[0]}")
        try:
            df = pd.read_csv(dist_files[0], nrows=5)
            print(f"  Columns: {list(df.columns)}")
            print(f"  Has 'Standard Value (nM)': {'Standard Value (nM)' in df.columns}")
        except Exception as e:
            print(f"  Error reading: {e}")
    
    # Look for prepared data files
    prepared_pattern = os.path.join(run_dir, "*/prepared_data/*_heldout_target_actives.csv")
    prepared_files = glob.glob(prepared_pattern)
    print(f"\nPrepared data files found: {len(prepared_files)}")
    if prepared_files:
        print(f"  First: {prepared_files[0]}")
        try:
            df = pd.read_csv(prepared_files[0], nrows=5)
            print(f"  Columns: {list(df.columns)}")
            affinity_cols = [c for c in df.columns if 'value' in c.lower() or 'affinity' in c.lower() or 'pchembl' in c.lower()]
            print(f"  Affinity-related columns: {affinity_cols}")
        except Exception as e:
            print(f"  Error reading: {e}")
    
    # Look for raw temp data files
    raw_pattern = os.path.join(run_dir, "*/temp_data/*_target_ligands_for_feature_calc_raw.csv")
    raw_files = glob.glob(raw_pattern)
    print(f"\nRaw temp data files found: {len(raw_files)}")
    if raw_files:
        print(f"  First: {raw_files[0]}")
        try:
            df = pd.read_csv(raw_files[0], nrows=5)
            print(f"  Columns: {list(df.columns)}")
            affinity_cols = [c for c in df.columns if 'value' in c.lower() or 'affinity' in c.lower() or 'pchembl' in c.lower()]
            print(f"  Affinity-related columns: {affinity_cols}")
            print(f"  Has 'Standard Value (nM)': {'Standard Value (nM)' in df.columns}")
        except Exception as e:
            print(f"  Error reading: {e}")
    
    # Directory structure
    print(f"\nDirectory structure:")
    for root, dirs, files in os.walk(run_dir):
        level = root.replace(run_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        if level < 3:  # Only show first 3 levels
            subindent = ' ' * 2 * (level + 1)
            for file in files[:5]:  # Only show first 5 files
                print(f"{subindent}{file}")
            if len(files) > 5:
                print(f"{subindent}... and {len(files)-5} more files")
        if level >= 3:
            break

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python debug_run_structure.py <workspace_dir>")
        print("Will inspect the first run directory found")
        sys.exit(1)
    
    workspace_dir = sys.argv[1]
    run_pattern = os.path.join(workspace_dir, "run_seed*")
    run_dirs = glob.glob(run_pattern)
    
    if not run_dirs:
        print(f"No run directories found in {workspace_dir}")
        sys.exit(1)
    
    # Inspect first 2 runs
    for run_dir in run_dirs[:2]:
        inspect_run_directory(run_dir)
