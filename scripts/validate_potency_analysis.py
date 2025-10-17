#!/usr/bin/env python3
"""
Quick validation script to check if potency-stratified analysis can run.

Checks:
1. Workspace exists and has run directories
2. Run directories have ranked files
3. Affinity data is available (in ranked files or source files)
4. Reports how many runs are ready for analysis

Usage:
    python scripts/validate_potency_analysis.py --workspace_dir experiment_workspace_v3_phase1
"""

import os
import sys
import glob
import argparse
import pandas as pd
from pathlib import Path

def check_workspace(workspace_dir):
    """Validate workspace and check data availability."""
    
    if not os.path.exists(workspace_dir):
        print(f"❌ Workspace not found: {workspace_dir}")
        return False
    
    print(f"✓ Workspace found: {workspace_dir}")
    
    # Find run directories
    run_pattern = os.path.join(workspace_dir, "run_seed*")
    run_dirs = glob.glob(run_pattern)
    
    if not run_dirs:
        print(f"❌ No run directories found matching pattern: {run_pattern}")
        return False
    
    print(f"✓ Found {len(run_dirs)} run directories")
    
    # Check sample runs
    runs_with_ranked = 0
    runs_with_affinity_in_ranked = 0
    runs_with_affinity_in_source = 0
    
    sample_size = min(10, len(run_dirs))
    print(f"\nChecking sample of {sample_size} runs...")
    
    for run_dir in run_dirs[:sample_size]:
        run_name = os.path.basename(run_dir)
        
        # Check for ranked file
        ranked_pattern = os.path.join(run_dir, "*/results/*/dim_*/*/*-RANKED.csv")
        ranked_files = glob.glob(ranked_pattern)
        
        if not ranked_files:
            continue
        
        runs_with_ranked += 1
        
        # Check if affinity in ranked file
        try:
            df = pd.read_csv(ranked_files[0], nrows=5)
            if 'Standard Value (nM)' in df.columns:
                runs_with_affinity_in_ranked += 1
        except:
            pass
        
        # Check for affinity in source files
        distance_pattern = os.path.join(run_dir, "*/results/*/dim_*/*/*detailed_active_distances.csv")
        distance_files = glob.glob(distance_pattern)
        
        if distance_files:
            try:
                df = pd.read_csv(distance_files[0], nrows=5)
                if 'Standard Value (nM)' in df.columns:
                    runs_with_affinity_in_source += 1
            except:
                pass
        else:
            # Check prepared data
            active_pattern = os.path.join(run_dir, "*/prepared_data/*_heldout_target_actives.csv")
            active_files = glob.glob(active_pattern)
            
            if active_files:
                try:
                    df = pd.read_csv(active_files[0], nrows=5)
                    if 'Standard Value (nM)' in df.columns or 'standard_value' in df.columns:
                        runs_with_affinity_in_source += 1
                except:
                    pass
    
    print(f"\nSample Results ({sample_size} runs checked):")
    print(f"  Runs with ranked files: {runs_with_ranked}/{sample_size}")
    print(f"  Ranked files with affinity data: {runs_with_affinity_in_ranked}/{runs_with_ranked}")
    print(f"  Runs with affinity in source files: {runs_with_affinity_in_source}/{sample_size}")
    
    # Estimate total ready runs
    if runs_with_ranked > 0:
        completion_rate = runs_with_ranked / sample_size
        estimated_complete = int(len(run_dirs) * completion_rate)
        print(f"\n📊 Estimated ~{estimated_complete} runs are complete (have ranked files)")
    
    if runs_with_affinity_in_ranked > 0 or runs_with_affinity_in_source > 0:
        affinity_rate = max(runs_with_affinity_in_ranked, runs_with_affinity_in_source) / sample_size
        estimated_with_affinity = int(len(run_dirs) * affinity_rate)
        print(f"📊 Estimated ~{estimated_with_affinity} runs have affinity data available")
        
        if affinity_rate > 0.5:
            print(f"\n✅ Good! Potency-stratified analysis should work on most runs")
            print(f"\nReady to run:")
            print(f"  python scripts/analyze_potency_stratified_enrichment.py \\")
            print(f"    --workspace_dir {workspace_dir} \\")
            print(f"    --output_dir potency_analysis_results")
            return True
        else:
            print(f"\n⚠️  Warning: Affinity data availability is limited")
            print(f"   Analysis will only work on runs with affinity data")
            return True
    else:
        print(f"\n❌ No affinity data found in sampled runs")
        print(f"   Potency-stratified analysis requires 'Standard Value (nM)' column")
        print(f"\nPossible solutions:")
        print(f"  1. Ensure project_and_analyze.py preserves affinity columns")
        print(f"  2. Check that ChEMBL data includes Standard Value column")
        print(f"  3. Re-run experiments with affinity data preservation")
        return False

def main():
    parser = argparse.ArgumentParser(description="Validate workspace for potency-stratified analysis")
    parser.add_argument('--workspace_dir', required=True, help='Path to experiment workspace')
    args = parser.parse_args()
    
    print("=" * 70)
    print("Potency-Stratified Analysis - Data Validation")
    print("=" * 70)
    print()
    
    success = check_workspace(args.workspace_dir)
    
    print()
    print("=" * 70)
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == '__main__':
    main()
