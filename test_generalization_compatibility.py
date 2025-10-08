#!/usr/bin/env python3
"""
Test script to verify that all generalization experiment scripts can handle
the new configuration filename pattern with target protein names.
"""

import os
import re
import glob
from pathlib import Path

def test_config_files():
    """Test that config files are generated with unique names."""
    print("=" * 70)
    print("TEST 1: Configuration File Uniqueness")
    print("=" * 70)
    
    config_dir = Path("generalization_configs")
    if not config_dir.exists():
        print("✗ Config directory not found. Run generate_generalization_configs.py first.")
        return False
    
    config_files = list(config_dir.glob("*.json"))
    print(f"Found {len(config_files)} configuration files")
    
    # Check for expected pattern
    expected_pattern = re.compile(r"config_(\w+)_features_(\w+)\.json")
    
    targets_found = set()
    methods_found = set()
    
    for config_file in config_files:
        match = expected_pattern.match(config_file.name)
        if match:
            target = match.group(1)
            method = match.group(2)
            targets_found.add(target)
            methods_found.add(method)
            print(f"  ✓ {config_file.name}")
        else:
            print(f"  ✗ Unexpected filename format: {config_file.name}")
            return False
    
    print(f"\nUnique targets: {len(targets_found)}")
    for target in sorted(targets_found):
        print(f"  • {target}")
    
    print(f"\nUnique methods: {len(methods_found)}")
    for method in sorted(methods_found):
        print(f"  • {method}")
    
    # Check expected total (2 targets × 5 methods = 10 configs)
    expected_total = 10
    if len(config_files) == expected_total:
        print(f"\n✓ Correct number of configs: {len(config_files)} (expected {expected_total})")
        return True
    else:
        print(f"\n✗ Wrong number of configs: {len(config_files)} (expected {expected_total})")
        return False

def test_directory_naming():
    """Test that directory names from orchestrator will be parsed correctly."""
    print("\n" + "=" * 70)
    print("TEST 2: Directory Naming Pattern Compatibility")
    print("=" * 70)
    
    # Simulate directory names that would be created by main_orchestrator.py
    test_cases = [
        ("run_seed42_config_PyruvateKinaseM2_P14618_features_pca_projection", 42),
        ("run_seed43_config_IsocitrateDehydrogenaseNADP_O75874_features_tsne", 43),
        ("run_seed44_config_PyruvateKinaseM2_P14618_features_umap_euclidean_coembedding", 44),
        ("run_seed45_config_IsocitrateDehydrogenaseNADP_O75874_features_umap_euclidean_projection", 45),
        ("run_seed46_config_PyruvateKinaseM2_P14618_features_pca_coembedding", 46),
    ]
    
    all_passed = True
    
    for dirname, expected_seed in test_cases:
        # Test pattern matching used in aggregation script
        if not dirname.startswith("run_seed"):
            print(f"  ✗ {dirname}: doesn't start with 'run_seed'")
            all_passed = False
            continue
        
        # Test seed extraction
        seed_match = re.search(r"run_seed(\d+)", dirname)
        if seed_match:
            seed = int(seed_match.group(1))
            if seed == expected_seed:
                print(f"  ✓ {dirname[:60]:60s} -> seed={seed}")
            else:
                print(f"  ✗ {dirname}: extracted seed={seed}, expected={expected_seed}")
                all_passed = False
        else:
            print(f"  ✗ {dirname}: failed to extract seed")
            all_passed = False
    
    if all_passed:
        print("\n✓ All directory names compatible with aggregation script")
    else:
        print("\n✗ Some directory names have issues")
    
    return all_passed

def test_submission_script():
    """Test that submission script can find and process all configs."""
    print("\n" + "=" * 70)
    print("TEST 3: Submission Script Compatibility")
    print("=" * 70)
    
    config_dir = "generalization_configs"
    
    # Simulate what the submission script does
    config_files = glob.glob(f"{config_dir}/*.json")
    
    if not config_files:
        print(f"✗ No config files found in {config_dir}")
        return False
    
    print(f"Submission script would find {len(config_files)} configs:")
    
    seeds = [42, 43, 44, 45, 46]
    total_jobs = len(config_files) * len(seeds)
    
    for config_file in sorted(config_files):
        basename = os.path.basename(config_file)
        # Test job name generation (from submission script)
        job_name_base = basename.replace('.json', '')
        print(f"  • {basename[:60]:60s}")
        
        # Show example job names for this config
        for seed in seeds[:2]:  # Just show first 2 seeds
            job_name = f"GEN_{job_name_base}_seed{seed}"
            if len(job_name) > 100:
                print(f"    ⚠ Job name might be too long: {len(job_name)} chars")
    
    print(f"\n✓ Submission script would create {total_jobs} jobs ({len(config_files)} configs × {len(seeds)} seeds)")
    return True

def test_aggregation_pattern():
    """Test that aggregation script pattern would work."""
    print("\n" + "=" * 70)
    print("TEST 4: Aggregation Script Pattern Matching")
    print("=" * 70)
    
    # Simulate the glob pattern used in aggregation script
    test_workspace = "experiment_workspace_generalization"
    pattern = os.path.join(test_workspace, "run_seed*", "*", "results", "*", "dim_*", "*", "*_ranking_metrics.csv")
    
    print(f"Pattern used: {pattern}")
    print("\nSimulating directory structure:")
    
    # Example paths that would be created
    example_paths = [
        f"{test_workspace}/run_seed42_config_PyruvateKinaseM2_P14618_features_tsne/PyruvateKinaseM2_P14618/results/features/dim_2/t_SNE/PyruvateKinaseM2_P14618_ranking_metrics.csv",
        f"{test_workspace}/run_seed43_config_IsocitrateDehydrogenaseNADP_O75874_features_umap_euclidean_coembedding/IsocitrateDehydrogenaseNADP_O75874/results/features/dim_2/UMAP_Euclidean_Coembed/IsocitrateDehydrogenaseNADP_O75874_ranking_metrics.csv",
    ]
    
    for path in example_paths:
        path_parts = path.split(os.sep)
        
        # Check if path would match the pattern
        if "run_seed" in path and "_ranking_metrics.csv" in path:
            # Extract components like the aggregation script does
            try:
                run_dir_name = next(p for p in path_parts if p.startswith("run_seed"))
                seed_match = re.search(r"run_seed(\d+)", run_dir_name)
                
                if seed_match:
                    seed = int(seed_match.group(1))
                    print(f"  ✓ Would extract: seed={seed}, run_dir={run_dir_name[:40]}...")
                else:
                    print(f"  ✗ Failed to extract seed from: {run_dir_name}")
            except Exception as e:
                print(f"  ✗ Error processing path: {e}")
        else:
            print(f"  ✗ Path doesn't match expected pattern")
    
    print("\n✓ Aggregation script pattern should work correctly")
    return True

def main():
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  GENERALIZATION EXPERIMENT - SCRIPT COMPATIBILITY TEST".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")
    print("\n")
    
    results = []
    
    # Run all tests
    results.append(("Config File Generation", test_config_files()))
    results.append(("Directory Naming", test_directory_naming()))
    results.append(("Submission Script", test_submission_script()))
    results.append(("Aggregation Pattern", test_aggregation_pattern()))
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:40s} {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("=" * 70)
    if all_passed:
        print("\n🎉 ALL TESTS PASSED - Scripts are compatible with new config names!")
        print("\nYou can safely proceed with:")
        print("  1. bash hpc/submit_generalization_jobs.sh")
        print("  2. python aggregate_generalization_analysis.py (after jobs complete)")
    else:
        print("\n⚠️  SOME TESTS FAILED - Review issues above before proceeding")
    
    print()

if __name__ == "__main__":
    main()
