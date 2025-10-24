#!/usr/bin/env python3
"""
Rerun Single Phase 1 Configuration with Deduplicated Data and Compare Results

This script reruns ONE Phase 1 configuration after fixing the MF cloud duplication
issue, then compares enrichment metrics to assess the impact of duplicates on 
DR model training.

Strategy:
1. Parse original run configuration (seed, target, DR method, hyperparameters)
2. Run prepare_data.py with deduplication fix
3. Run calculate_similarityspaces_exp.py to train DR model on clean data
4. Run project_and_analyze.py to calculate enrichment metrics
5. Load original enrichment metrics from old run
6. Compare original vs clean results
7. Determine if full Phase 1 rerun is needed

Usage:
    python rerun_phase1_config_compare.py \\
        --original_run experiment_workspace_v3_phase1/run_seed44_config_tyro_features_umap_euclidean_dim5_nn10_md0.1_seed44 \\
        --output_workspace experiment_workspace_v3_phase1_clean_test \\
        --experiment_config experiment_config.json
"""

import os
import sys
import json
import argparse
import logging
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def parse_original_config(original_run_dir):
    """Extract configuration from original Phase 1 run."""
    config_path = os.path.join(original_run_dir, 'run_config.json')
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    logging.info(f"Loaded original configuration from: {config_path}")
    
    # Extract key parameters
    seed = config.get('random_seed')
    target = config['targets'][0]['id_name']
    representation = config['representations'][0]
    
    # Extract DR method details
    dr_methods = config.get('dimensionality_reduction_methods', {})
    dr_key = list(dr_methods.keys())[0]
    dr_config = dr_methods[dr_key]
    
    dr_method = dr_config.get('short_name')
    
    # Extract dimension from simspace_dims_to_test list (Phase 1 configs have single dimension)
    dims_list = config['global_settings'].get('simspace_dims_to_test', [])
    dimension = dims_list[0] if dims_list else None
    if dimension is None:
        # Fallback: try simspace_dim (used in some configs)
        dimension = config['global_settings'].get('simspace_dim')
    
    if dimension is None:
        raise ValueError("Could not extract dimension from config. Check 'simspace_dims_to_test' or 'simspace_dim' in global_settings")
    
    n_neighbors = dr_config.get('n_neighbors')
    min_dist = dr_config.get('min_dist')
    
    parsed = {
        'seed': seed,
        'target_id': target,
        'representation': representation,
        'dr_method': dr_method,
        'dimension': dimension,
        'n_neighbors': n_neighbors,
        'min_dist': min_dist,
        'full_config': config
    }
    
    logging.info(f"  Seed: {seed}")
    logging.info(f"  Target: {target}")
    logging.info(f"  Representation: {representation}")
    logging.info(f"  DR method: {dr_method}")
    logging.info(f"  Dimension: {dimension}D")
    logging.info(f"  n_neighbors: {n_neighbors}")
    logging.info(f"  min_dist: {min_dist}")
    
    return parsed


def run_prepare_data(config, output_workspace):
    """Run prepare_data.py with deduplication fix."""
    logging.info("\n" + "="*80)
    logging.info("STEP 1: PREPARE DATA (with deduplication)")
    logging.info("="*80)
    
    target_workspace = os.path.join(output_workspace, f"run_clean_seed{config['seed']}", config['target_id'])
    os.makedirs(target_workspace, exist_ok=True)
    
    temp_data_dir = os.path.join(target_workspace, "temp_data")
    os.makedirs(temp_data_dir, exist_ok=True)
    
    cmd = [
        'python', 'experimental_pipeline/prepare_data.py',
        '--config_path', args.experiment_config,
        '--target_id_name', config['target_id'],
        '--output_dir', temp_data_dir
    ]
    
    logging.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logging.error(f"prepare_data.py failed:")
        logging.error(result.stderr)
        raise RuntimeError("Data preparation failed")
    
    logging.info("✓ Data preparation complete")
    logging.info(f"  Output directory: {temp_data_dir}")
    
    # Verify deduplication happened
    mf_file = os.path.join(temp_data_dir, f"{config['target_id']}_chembl_mf_excluded_{config['representation']}.csv")
    if os.path.exists(mf_file):
        df_mf = pd.read_csv(mf_file, low_memory=False)
        if 'Compound ChEMBL ID' in df_mf.columns:
            n_rows = len(df_mf)
            n_unique = df_mf['Compound ChEMBL ID'].nunique()
            if n_rows == n_unique:
                logging.info(f"✓ MF cloud deduplicated: {n_rows:,} rows = {n_unique:,} unique compounds")
            else:
                logging.warning(f"⚠ MF cloud still has duplicates: {n_rows:,} rows vs {n_unique:,} unique!")
    
    return target_workspace, temp_data_dir


def run_calculate_similarityspaces(config, target_workspace, temp_data_dir):
    """Run calculate_similarityspaces_exp.py to train DR model."""
    logging.info("\n" + "="*80)
    logging.info("STEP 2: CALCULATE SIMILARITY SPACE (train DR model on clean data)")
    logging.info("="*80)
    
    # Prepare paths
    mf_data_path = os.path.join(temp_data_dir, f"{config['target_id']}_chembl_mf_excluded_{config['representation']}.csv")
    zinc_data_path = os.path.join(temp_data_dir, f"{config['target_id']}_zinc_excluded_{config['representation']}.csv")
    target_ligands_path = os.path.join(temp_data_dir, f"{config['target_id']}_target_ligands_for_feature_calc_raw.csv")
    
    output_simspace_dir = os.path.join(target_workspace, "similarity_spaces", config['representation'], f"dim_{config['dimension']}")
    output_model_dir = os.path.join(target_workspace, "models", config['representation'], f"dim_{config['dimension']}")
    os.makedirs(output_simspace_dir, exist_ok=True)
    os.makedirs(output_model_dir, exist_ok=True)
    
    log_file = os.path.join(target_workspace, "phase1_clean_rerun.log")
    
    # Build DR method config JSON
    dr_config = {
        f"umap_{config['dr_method'].split('-')[-1].lower()}": {
            "n_neighbors": config['n_neighbors'],
            "min_dist": config['min_dist']
        }
    }
    
    # Get feature list from original config (stored in global_settings)
    features_list = config['full_config']['global_settings'].get('rdkit_features_list_target', [])
    
    cmd = [
        'python', 'core_scripts/calculate_similarityspaces_exp.py',
        '--chembl_mf_data_path', mf_data_path,
        '--zinc_data_path', zinc_data_path,
        '--target_ligands_path_for_projection', target_ligands_path,
        '--representation_type', config['representation'],
        '--target_id_name', config['target_id'],
        '--output_simspace_dir', output_simspace_dir,
        '--output_model_dir', output_model_dir,
        '--log_file_path', log_file,
        '--simspace_dim', str(config['dimension']),
        '--dr_method_umap', 'true',  # This one takes a value
        '--n_neighbors', str(config['n_neighbors']),
        '--dr_method_configs_json_str', json.dumps(dr_config),
        '--rdkit_features_list_target_str', json.dumps(features_list),
        '--random_state', str(config['seed'])
    ]
    
    # Add metric flag (store_true flag - no value needed)
    metric_name = config['dr_method'].split('-')[-1].lower()
    cmd.append(f"--umap_metric_to_run_{metric_name}")
    
    logging.info(f"Running: {' '.join(cmd[:10])}...")  # Truncated for readability
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logging.error(f"calculate_similarityspaces_exp.py failed:")
        logging.error(result.stderr)
        raise RuntimeError("Similarity space calculation failed")
    
    logging.info("✓ Similarity space calculation complete")
    logging.info(f"  Output directory: {output_simspace_dir}")
    
    # Verify similarity space was created
    simspace_file = os.path.join(
        output_simspace_dir,
        f"{config['target_id']}_{config['representation']}_dim{config['dimension']}_similarity_space.csv"
    )
    
    if os.path.exists(simspace_file):
        df_simspace = pd.read_csv(simspace_file, low_memory=False, nrows=10)
        logging.info(f"✓ Similarity space created: {simspace_file}")
    else:
        raise FileNotFoundError(f"Similarity space not created: {simspace_file}")
    
    return output_simspace_dir, output_model_dir


def run_project_and_analyze(config, target_workspace, output_simspace_dir, output_model_dir, temp_data_dir):
    """Run project_and_analyze.py to calculate enrichment metrics."""
    logging.info("\n" + "="*80)
    logging.info("STEP 3: PROJECT AND ANALYZE (calculate enrichment on clean data)")
    logging.info("="*80)
    
    # Paths
    simspace_csv = os.path.join(
        output_simspace_dir,
        f"{config['target_id']}_{config['representation']}_dim{config['dimension']}_similarity_space.csv"
    )
    
    target_actives_path = os.path.join(
        target_workspace,
        "target_ligands_calculated",
        config['representation'],
        f"{config['target_id']}_target_ligands_for_calc_{config['representation']}.csv"
    )
    
    output_results_dir = os.path.join(
        target_workspace,
        "results",
        config['representation'],
        f"dim_{config['dimension']}",
        config['dr_method'].replace('-', '_')
    )
    os.makedirs(output_results_dir, exist_ok=True)
    
    cmd = [
        'python', 'experimental_pipeline/project_and_analyze.py',
        '--simspace_csv_path', simspace_csv,
        '--model_dir', output_model_dir,
        '--target_actives_path', target_actives_path,
        '--dr_short_name', config['dr_method'],
        '--simspace_dim', str(config['dimension']),
        '--output_dir', output_results_dir,
        '--target_id_name', config['target_id'],
        '--representation_type', config['representation'],
        '--k_for_knn', '5,10,20',
        '--affinity_cutoff_nM', '100000'
    ]
    
    logging.info(f"Running: {' '.join(cmd[:10])}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logging.error(f"project_and_analyze.py failed:")
        logging.error(result.stderr)
        raise RuntimeError("Projection and analysis failed")
    
    logging.info("✓ Projection and analysis complete")
    logging.info(f"  Output directory: {output_results_dir}")
    
    return output_results_dir


def load_metrics(results_dir, target_id, representation, dr_method, dimension):
    """Load enrichment metrics from results directory."""
    dr_method_safe = dr_method.replace('-', '_')
    metrics_file = os.path.join(
        results_dir,
        f"{target_id}_{representation}_{dr_method_safe}_dim{dimension}_ranking_metrics.csv"
    )
    
    if not os.path.exists(metrics_file):
        logging.warning(f"Metrics file not found: {metrics_file}")
        return None
    
    df_metrics = pd.read_csv(metrics_file)
    if len(df_metrics) > 0:
        return df_metrics.iloc[0].to_dict()
    
    return None


def compare_metrics(original_metrics, clean_metrics):
    """Compare original vs clean enrichment metrics."""
    logging.info("\n" + "="*80)
    logging.info("COMPARISON: ORIGINAL (with duplicates) vs CLEAN (deduplicated)")
    logging.info("="*80)
    
    if original_metrics is None:
        logging.error("Could not load original metrics")
        return None
    
    if clean_metrics is None:
        logging.error("Could not load clean metrics")
        return None
    
    metrics_to_compare = ['ef_1%', 'ef_5%', 'ef_10%', 'roc_auc', 'pr_auc']
    
    results = {}
    
    for metric in metrics_to_compare:
        orig_val = original_metrics.get(metric, np.nan)
        clean_val = clean_metrics.get(metric, np.nan)
        
        if not np.isnan(orig_val) and not np.isnan(clean_val):
            diff = clean_val - orig_val
            pct_change = (diff / orig_val * 100) if orig_val != 0 else np.nan
            
            logging.info(f"\n{metric.upper()}:")
            logging.info(f"  Original (duplicates):  {orig_val:.4f}")
            logging.info(f"  Clean (deduplicated):   {clean_val:.4f}")
            logging.info(f"  Change:                 {diff:+.4f} ({pct_change:+.2f}%)")
            
            results[metric] = {
                'original': orig_val,
                'clean': clean_val,
                'diff': diff,
                'pct_change': pct_change
            }
    
    # Decision logic
    ef1_pct_change = abs(results.get('ef_1%', {}).get('pct_change', 0))
    
    logging.info("\n" + "="*80)
    logging.info("DECISION")
    logging.info("="*80)
    
    if ef1_pct_change < 5:
        logging.info(f"✓ EF@1% change is {ef1_pct_change:.2f}% (< 5% threshold)")
        logging.info("✓ IMPACT IS MINIMAL:")
        logging.info("  - DR models appear robust to duplicate data in training")
        logging.info("  - Can proceed with existing Phase 1 results")
        logging.info("  - Consider documenting this limitation in methods")
        decision = 'minimal_impact'
    else:
        logging.info(f"✗ EF@1% change is {ef1_pct_change:.2f}% (≥ 5% threshold)")
        logging.info("✗ SIGNIFICANT IMPACT DETECTED:")
        logging.info("  - DR models learned distorted manifolds from duplicate data")
        logging.info("  - Enrichment metrics are significantly different")
        logging.info("  - Must rerun ALL Phase 1 experiments with deduplicated data")
        logging.info("  - Must update ALL downstream analyses (Phase 2, 3, 4)")
        decision = 'significant_impact'
    
    return {
        'decision': decision,
        'ef1_pct_change': ef1_pct_change,
        'metrics': results
    }


def main():
    global args
    
    parser = argparse.ArgumentParser(
        description="Rerun single Phase 1 config with clean data and compare",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--original_run', required=True,
                       help='Path to original Phase 1 run directory')
    parser.add_argument('--output_workspace', required=True,
                       help='Output workspace for clean rerun')
    parser.add_argument('--experiment_config', required=True,
                       help='Path to experiment_config.json')
    
    args = parser.parse_args()
    
    logging.info("="*80)
    logging.info("RERUN PHASE 1 CONFIG WITH DEDUPLICATED DATA")
    logging.info("="*80)
    logging.info(f"Original run: {args.original_run}")
    logging.info(f"Output workspace: {args.output_workspace}")
    logging.info(f"Experiment config: {args.experiment_config}")
    
    # Parse original configuration
    config = parse_original_config(args.original_run)
    
    # Create output workspace
    os.makedirs(args.output_workspace, exist_ok=True)
    
    try:
        # Run pipeline with deduplication
        target_workspace, temp_data_dir = run_prepare_data(config, args.output_workspace)
        output_simspace_dir, output_model_dir = run_calculate_similarityspaces(config, target_workspace, temp_data_dir)
        clean_results_dir = run_project_and_analyze(config, target_workspace, output_simspace_dir, output_model_dir, temp_data_dir)
        
        # Load metrics
        logging.info("\n" + "="*80)
        logging.info("LOADING METRICS FOR COMPARISON")
        logging.info("="*80)
        
        # Original metrics
        original_results_dir = os.path.join(
            args.original_run,
            "results",
            config['representation'],
            f"dim_{config['dimension']}",
            config['dr_method'].replace('-', '_')
        )
        
        logging.info(f"Loading original metrics from: {original_results_dir}")
        original_metrics = load_metrics(
            original_results_dir,
            config['target_id'],
            config['representation'],
            config['dr_method'],
            config['dimension']
        )
        
        logging.info(f"Loading clean metrics from: {clean_results_dir}")
        clean_metrics = load_metrics(
            clean_results_dir,
            config['target_id'],
            config['representation'],
            config['dr_method'],
            config['dimension']
        )
        
        # Compare
        comparison = compare_metrics(original_metrics, clean_metrics)
        
        # Save comparison results
        comparison_file = os.path.join(args.output_workspace, f"comparison_seed{config['seed']}.json")
        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, indent=2, default=str)
        
        logging.info(f"\n✓ Comparison results saved to: {comparison_file}")
        
        # Exit code based on decision
        if comparison['decision'] == 'minimal_impact':
            sys.exit(0)
        else:
            sys.exit(1)
    
    except Exception as e:
        logging.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(2)


if __name__ == '__main__':
    main()
