#!/usr/bin/env python3
"""
Phase 2 Orchestrator: Affinity Cutoff Sensitivity Analysis

This script orchestrates Phase 2 cutoff sensitivity experiments by REUSING
Phase 1 similarity spaces and DR models. This dramatically reduces runtime:
- Phase 1 full pipeline: ~2-3 hours per run (features + DR fitting + ranking)
- Phase 2 cutoff reanalysis: ~10-15 min per run (ranking only)

**Strategy:**
1. Locate Phase 1 run directories matching Phase 2 config (same seed/method/dim)
2. Find existing similarity spaces and DR models from Phase 1
3. Call project_and_analyze.py with:
   - Existing similarity space CSV
   - Existing DR model directory
   - NEW affinity cutoff parameter
4. Output to Phase 2 workspace with cutoff-specific subdirectory

**Computational Savings:**
- Does NOT recalculate features/fingerprints
- Does NOT refit PCA/UMAP models  
- ONLY reruns ranking/evaluation with different cutoff
- 75% time savings compared to full pipeline

Usage:
    python scripts/run_phase2_cutoff_analysis.py \\
        --config_dir hyperparam_configs_v3_phase2_cutoff \\
        --phase1_workspace experiment_workspace_v3_phase1 \\
        --phase2_workspace experiment_workspace_v3_phase2

    # Or run specific config
    python scripts/run_phase2_cutoff_analysis.py \\
        --config hyperparam_configs_v3_phase2_cutoff/config_seed42_features_pca_dim2_cutoff100nM.json \\
        --phase1_workspace experiment_workspace_v3_phase1 \\
        --phase2_workspace experiment_workspace_v3_phase2
"""

import os
import glob
import subprocess
import argparse
import logging
import json
from datetime import datetime
from pathlib import Path

# Logging setup
log_file = f"phase2_cutoff_orchestrator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)-8s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'),
        logging.StreamHandler()
    ]
)


def find_matching_phase1_run(phase2_config, phase1_workspace):
    """Find Phase 1 run directory matching Phase 2 config.
    
    Args:
        phase2_config: Phase 2 configuration dict
        phase1_workspace: Path to Phase 1 workspace
    
    Returns:
        Path to matching Phase 1 run directory, or None if not found
    """
    seed = phase2_config['random_seed']
    representation = phase2_config['representations'][0]
    dr_method_key = list(phase2_config['dimensionality_reduction_methods'].keys())[0]
    dr_config = phase2_config['dimensionality_reduction_methods'][dr_method_key]
    dimension = phase2_config['global_settings']['simspace_dims_to_test'][0]
    
    # Build search pattern based on representation and method
    # Phase 1 pattern: run_seed{seed}_config_tyro_{representation}_{method}_dim{dimension}_[params]_seed{seed}
    if 'PCA' in dr_config['short_name']:
        # PCA pattern: run_seed{seed}_config_tyro_{representation}_pca_dim{dimension}_seed{seed}
        pattern = f"run_seed{seed}_config_tyro_{representation}_pca_*dim{dimension}*seed{seed}"
    elif 'UMAP' in dr_config['short_name']:
        # UMAP pattern: run_seed{seed}_config_tyro_{representation}_umap_euclidean_dim{dimension}_nn{nn}_md{md}_seed{seed}
        nn = dr_config['n_neighbors']
        md = dr_config['min_dist']
        pattern = f"run_seed{seed}_config_tyro_{representation}_umap_euclidean_dim{dimension}_nn{nn}_md{md}_seed{seed}"
    else:
        logging.error(f"Unknown DR method: {dr_config['short_name']}")
        return None
    
    search_path = os.path.join(phase1_workspace, pattern)
    matches = glob.glob(search_path)
    
    if not matches:
        logging.warning(f"No Phase 1 run found matching pattern: {pattern}")
        return None
    
    if len(matches) > 1:
        logging.warning(f"Multiple Phase 1 runs found, using first: {matches[0]}")
    
    logging.info(f"  Found matching Phase 1 run: {os.path.basename(matches[0])}")
    return matches[0]


def find_similarity_space(phase1_run_dir, representation, dimension, target_id):
    """Find similarity space CSV file from Phase 1 run.
    
    Args:
        phase1_run_dir: Path to Phase 1 run directory
        representation: 'features' or 'fingerprints'
        dimension: Dimensionality (e.g., 2, 5, 10)
        target_id: Target ID (e.g., 'TyrosineProteinKinaseABL1_P00519')
    
    Returns:
        Path to similarity space CSV, or None if not found
    """
    pattern = os.path.join(
        phase1_run_dir,
        target_id,
        "similarity_spaces",
        representation,
        f"dim_{dimension}",
        f"{target_id}_{representation}_dim{dimension}_similarity_space.csv"
    )
    
    if os.path.exists(pattern):
        logging.info(f"  Found similarity space: {os.path.basename(pattern)}")
        return pattern
    
    logging.error(f"Similarity space not found: {pattern}")
    return None


def find_dr_model_dir(phase1_run_dir, representation, dimension, target_id):
    """Find DR model directory from Phase 1 run.
    
    Args:
        phase1_run_dir: Path to Phase 1 run directory
        representation: 'features' or 'fingerprints'
        dimension: Dimensionality
        target_id: Target ID
    
    Returns:
        Path to model directory, or None if not found
    """
    model_dir = os.path.join(
        phase1_run_dir,
        target_id,
        "models",
        representation,
        f"dim_{dimension}"
    )
    
    if os.path.exists(model_dir):
        logging.info(f"  Found DR model dir: {model_dir}")
        return model_dir
    
    logging.error(f"DR model directory not found: {model_dir}")
    return None


def find_target_ligands_path(phase1_run_dir, representation, target_id):
    """Find target ligands representation file from Phase 1 run.
    
    Args:
        phase1_run_dir: Path to Phase 1 run directory
        representation: 'features' or 'fingerprints'
        target_id: Target ID
    
    Returns:
        Path to target ligands file, or None if not found
    """
    pattern = os.path.join(
        phase1_run_dir,
        target_id,
        "target_ligands_calculated",
        representation,
        f"{target_id}_target_ligands_for_calc_{representation}.csv"
    )
    
    if os.path.exists(pattern):
        logging.info(f"  Found target ligands: {os.path.basename(pattern)}")
        return pattern
    
    logging.error(f"Target ligands file not found: {pattern}")
    return None


def run_cutoff_analysis(config_path, phase1_workspace, phase2_workspace, base_config_path="experiment_config.json"):
    """Run cutoff analysis for a single Phase 2 configuration.
    
    Args:
        config_path: Path to Phase 2 configuration JSON
        phase1_workspace: Path to Phase 1 workspace (source data)
        phase2_workspace: Path to Phase 2 workspace (output)
        base_config_path: Path to base experiment config for global settings
    
    Returns:
        True if successful, False otherwise
    """
    logging.info(f"\n{'='*80}")
    logging.info(f"Processing: {os.path.basename(config_path)}")
    logging.info(f"{'='*80}")
    
    # Load configuration
    with open(config_path, 'r') as f:
        phase2_config = json.load(f)
    
    with open(base_config_path, 'r') as f:
        base_config = json.load(f)
    
    # Extract config details
    target_info = phase2_config['targets'][0]
    target_id = target_info['id_name']
    representation = phase2_config['representations'][0]
    dimension = phase2_config['global_settings']['simspace_dims_to_test'][0]
    cutoff_nm = phase2_config['affinity_cutoff_nM']
    dr_method_key = list(phase2_config['dimensionality_reduction_methods'].keys())[0]
    dr_config = phase2_config['dimensionality_reduction_methods'][dr_method_key]
    
    logging.info(f"  Target: {target_info['display_name']}")
    logging.info(f"  Method: {dr_config['short_name']}")
    logging.info(f"  Representation: {representation}")
    logging.info(f"  Dimension: {dimension}D")
    logging.info(f"  Cutoff: {cutoff_nm} nM")
    
    # Find matching Phase 1 run
    phase1_run_dir = find_matching_phase1_run(phase2_config, phase1_workspace)
    if not phase1_run_dir:
        logging.error("Cannot proceed without Phase 1 data")
        return False
    
    # Find required Phase 1 files (only need similarity space for DATA REUSE mode)
    simspace_path = find_similarity_space(phase1_run_dir, representation, dimension, target_id)
    
    if not simspace_path:
        logging.error("Missing required similarity space file from Phase 1")
        return False
    
    # Build output directory for Phase 2
    config_basename = os.path.basename(config_path).replace('config_', '').replace('.json', '')
    phase2_run_dir = os.path.join(phase2_workspace, f"run_{config_basename}")
    
    output_dir = os.path.join(
        phase2_run_dir,
        target_id,
        "results",
        representation,
        f"dim_{dimension}",
        dr_config['short_name'].replace('-', '_')
    )
    
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"  Output directory: {output_dir}")
    
    # Build command for project_and_analyze.py
    # Phase 2 DATA REUSE mode: similarity space already contains projected actives
    # Do NOT pass --target_ligands_repr_path or --model_dir_for_projection
    # This triggers DATA REUSE mode instead of PROJECTION mode
    cmd = [
        "python", "experimental_pipeline/project_and_analyze.py",
        "--simspace_csv_path", os.path.abspath(simspace_path),
        # NOTE: Omit --target_ligands_repr_path to trigger DATA REUSE mode
        # NOTE: Omit --model_dir_for_projection to trigger DATA REUSE mode
        "--dr_method_key", dr_method_key,
        "--dr_short_name", dr_config["short_name"],
        "--simspace_dim", str(dimension),
        "--k_for_knn", ",".join(map(str, base_config['global_settings']['k_for_knn_distance'])),
        "--output_dir", os.path.abspath(output_dir),
        "--target_id_name", target_id,
        "--representation_type", representation,
        "--rdkit_features_list_target_str", json.dumps(base_config['global_settings'].get('rdkit_features_list_target', [])),
        "--affinity_cutoff", str(cutoff_nm)
    ]
    
    # Save Phase 2 run config
    run_config_path = os.path.join(phase2_run_dir, "run_config.json")
    with open(run_config_path, 'w') as f:
        json.dump(phase2_config, f, indent=2)
    
    # Run project_and_analyze.py
    logging.info(f"  Running project_and_analyze.py with cutoff={cutoff_nm} nM...")
    logging.debug(f"  Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        if result.stdout:
            logging.debug(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            logging.warning(f"STDERR:\n{result.stderr}")
        
        logging.info(f"  ✓ Analysis completed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        logging.error(f"  ✗ Analysis failed with return code {e.returncode}")
        logging.error(f"STDOUT:\n{e.stdout}")
        logging.error(f"STDERR:\n{e.stderr}")
        return False
    except Exception as e:
        logging.error(f"  ✗ Unexpected error: {e}", exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2 Cutoff Sensitivity Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--config', type=str,
                       help='Path to single Phase 2 config JSON')
    parser.add_argument('--config_dir', type=str,
                       help='Directory containing Phase 2 config JSONs')
    parser.add_argument('--phase1_workspace', type=str, 
                       default='experiment_workspace_v3_phase1',
                       help='Path to Phase 1 workspace (source data)')
    parser.add_argument('--phase2_workspace', type=str,
                       default='experiment_workspace_v3_phase2',
                       help='Path to Phase 2 workspace (output)')
    parser.add_argument('--base_config', type=str,
                       default='experiment_config.json',
                       help='Path to base experiment config')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.config and not args.config_dir:
        parser.error("Must provide either --config or --config_dir")
    
    if args.config and args.config_dir:
        parser.error("Cannot provide both --config and --config_dir")
    
    if not os.path.exists(args.phase1_workspace):
        parser.error(f"Phase 1 workspace not found: {args.phase1_workspace}")
    
    # Create Phase 2 workspace
    os.makedirs(args.phase2_workspace, exist_ok=True)
    
    # Get config file(s)
    if args.config:
        config_files = [args.config]
    else:
        config_files = sorted(glob.glob(os.path.join(args.config_dir, "config_*.json")))
    
    logging.info("="*80)
    logging.info("PHASE 2: AFFINITY CUTOFF SENSITIVITY ANALYSIS")
    logging.info("="*80)
    logging.info(f"Phase 1 workspace: {args.phase1_workspace}")
    logging.info(f"Phase 2 workspace: {args.phase2_workspace}")
    logging.info(f"Configurations to process: {len(config_files)}")
    logging.info(f"Log file: {log_file}")
    logging.info("="*80)
    
    # Process each configuration
    success_count = 0
    fail_count = 0
    
    for config_file in config_files:
        success = run_cutoff_analysis(
            config_file,
            args.phase1_workspace,
            args.phase2_workspace,
            args.base_config
        )
        
        if success:
            success_count += 1
        else:
            fail_count += 1
    
    # Summary
    logging.info("\n" + "="*80)
    logging.info("PHASE 2 ORCHESTRATION COMPLETE")
    logging.info("="*80)
    logging.info(f"Total configurations: {len(config_files)}")
    logging.info(f"  Successful: {success_count}")
    logging.info(f"  Failed: {fail_count}")
    logging.info(f"\nLog saved to: {log_file}")
    logging.info("="*80)
    
    if fail_count > 0:
        logging.warning(f"\n{fail_count} runs failed. Check log for details.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
