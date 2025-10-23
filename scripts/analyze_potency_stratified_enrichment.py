#!/usr/bin/env python3
"""
Potency-Stratified Enrichment Analysis for Phase 1 Experiments

This script analyzes enrichment performance broken down by ligand potency tiers to reveal
quantity vs. quality trade-offs in hyperparameter selection. Traditional EF@1% treats all
actives equally, but a 100 nM ligand (drug-like) is vastly more valuable than a 100,000 nM
ligand (weak binder).

**Potency Tiers:**
- High Potent: 0.1-100 nM (drug-like, clinically relevant)
- Medium Potent: 100-1,000 nM (moderate affinity)
- Weak Potent: 1,000-100,000 nM (marginal, likely promiscuous)

**Key Insights:**
- nn=10 might show high overall EF@1% by enriching many weak binders
- nn=20 might show lower overall EF@1% but enrich fewer, highly potent binders
- For drug discovery, nn=20 would be superior despite appearing worse by traditional metrics

Usage:
    # Analyze all completed runs in workspace
    python scripts/analyze_potency_stratified_enrichment.py \\
        --workspace_dir experiment_workspace_v3_phase1 \\
        --output_dir potency_analysis_results

    # Analyze specific seed or configuration
    python scripts/analyze_potency_stratified_enrichment.py \\
        --workspace_dir experiment_workspace_v3_phase1 \\
        --seed 42 \\
        --output_dir potency_analysis_results/seed42

    # Quick summary only (no detailed plots)
    python scripts/analyze_potency_stratified_enrichment.py \\
        --workspace_dir experiment_workspace_v3_phase1 \\
        --output_dir potency_analysis_results \\
        --summary_only
"""

import os
import sys
import json
import glob
import argparse
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for HPC
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)-8s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Potency tier definitions (in nM)
POTENCY_TIERS = {
    'High': (0.1, 100),          # Drug-like, clinically relevant
    'Medium': (100, 1000),       # Moderate affinity
    'Weak': (1000, 100000)       # Marginal, likely promiscuous
}

TIER_ORDER = ['High', 'Medium', 'Weak']
TIER_COLORS = {'High': '#2ecc71', 'Medium': '#f39c12', 'Weak': '#e74c3c'}


def parse_run_config(run_config_path):
    """Extract configuration details from run_config.json."""
    try:
        with open(run_config_path, 'r') as f:
            config = json.load(f)
        
        dr_methods = config.get('dimensionality_reduction_methods', {})
        if not dr_methods:
            return None
        
        dr_key = list(dr_methods.keys())[0]
        dr_config = dr_methods[dr_key]
        
        # Extract dimension - try both simspace_dim and simspace_dims_to_test
        global_settings = config.get('global_settings', {})
        dimension = global_settings.get('simspace_dim')
        
        # If simspace_dim not found, try simspace_dims_to_test (which is a list)
        if dimension is None:
            dims_to_test = global_settings.get('simspace_dims_to_test', [])
            if dims_to_test:
                dimension = dims_to_test[0]  # Take first dimension from list
        
        info = {
            'seed': config.get('random_seed'),
            'target': config.get('targets', [{}])[0].get('id_name'),
            'representation': config.get('representations', [None])[0],
            'dr_method': dr_config.get('short_name'),
            'dimension': dimension,
            'n_neighbors': dr_config.get('n_neighbors'),
            'min_dist': dr_config.get('min_dist'),
            'affinity_cutoff': global_settings.get('affinity_cutoff_nM', 100000)
        }
        return info
    except Exception as e:
        logging.debug(f"Could not parse config from {run_config_path}: {e}")
        return None


def find_ranked_file(run_dir):
    """Find the ranked CSV file in the run directory."""
    # Try multiple patterns - file naming has varied
    patterns = [
        os.path.join(run_dir, "*/results/*/dim_*/*/*-RANKED.csv"),
        os.path.join(run_dir, "*/results/*/dim_*/*/*-FEATURES.csv"),
        os.path.join(run_dir, "*/results/*/dim_*/*/*-FINGERPRINTS.csv"),
        os.path.join(run_dir, "*/results/*/dim_*/*/TYR*-*-*D-*.csv"),  # Tyro uppercase pattern
        os.path.join(run_dir, "*/results/*/dim_*/*/PYR*-*-*D-*.csv"),  # Pyru uppercase pattern
        os.path.join(run_dir, "*/results/*/dim_*/*/ISO*-*-*D-*.csv"),  # Iso uppercase pattern
    ]
    
    for pattern in patterns:
        ranked_files = glob.glob(pattern)
        if ranked_files:
            # Filter out metrics files
            for f in ranked_files:
                if 'metrics' not in f.lower() and 'distances' not in f.lower():
                    return f
    
    return None


def classify_by_potency(affinity_nM):
    """Classify a compound by its potency tier based on affinity."""
    if pd.isna(affinity_nM) or affinity_nM <= 0:
        return None
    
    for tier, (low, high) in POTENCY_TIERS.items():
        if low <= affinity_nM <= high:
            return tier
    
    # Outside all tiers
    if affinity_nM < POTENCY_TIERS['High'][0]:
        return 'High'  # Ultra-potent, group with High
    elif affinity_nM > POTENCY_TIERS['Weak'][1]:
        return None  # Too weak, exclude from analysis
    
    return None


def calculate_stratified_enrichment(ranked_df, top_percent=0.01):
    """
    Calculate enrichment metrics stratified by potency tier.
    
    Args:
        ranked_df: DataFrame with 'RANKING', 'TYPE', and 'Standard Value (nM)' columns
        top_percent: Fraction for enrichment calculation (default 1%)
    
    Returns:
        Dictionary with enrichment metrics per tier
    """
    # Filter to only actives
    actives_df = ranked_df[ranked_df['TYPE'] == 'HELDOUT_ACTIVE'].copy()
    
    if actives_df.empty:
        return None
    
    # Add potency tier classification
    actives_df['Potency_Tier'] = actives_df['Standard Value (nM)'].apply(classify_by_potency)
    
    # Count actives per tier
    tier_counts = actives_df['Potency_Tier'].value_counts()
    
    # Calculate enrichment for each tier
    total_compounds = len(ranked_df)
    top_n = int(np.ceil(top_percent * total_compounds))
    
    if top_n == 0:
        return None
    
    results = {}
    
    for tier in TIER_ORDER:
        tier_actives_total = tier_counts.get(tier, 0)
        
        if tier_actives_total == 0:
            results[tier] = {
                'total': 0,
                'in_top': 0,
                'ef': np.nan,
                'percent_found': np.nan
            }
            continue
        
        # Find how many of this tier are in top X%
        tier_actives_df = actives_df[actives_df['Potency_Tier'] == tier]
        tier_in_top = (tier_actives_df['RANKING'] <= top_n).sum()
        
        # Calculate enrichment factor for this tier
        ef_observed = tier_in_top / top_n
        ef_random = tier_actives_total / total_compounds
        ef = ef_observed / ef_random if ef_random > 0 else np.nan
        
        # Calculate percent of tier found
        percent_found = (tier_in_top / tier_actives_total) * 100
        
        results[tier] = {
            'total': tier_actives_total,
            'in_top': tier_in_top,
            'ef': ef,
            'percent_found': percent_found
        }
    
    # Calculate overall metrics for comparison
    total_actives = len(actives_df)
    actives_in_top = (actives_df['RANKING'] <= top_n).sum()
    overall_ef = (actives_in_top / top_n) / (total_actives / total_compounds)
    
    results['Overall'] = {
        'total': total_actives,
        'in_top': actives_in_top,
        'ef': overall_ef,
        'percent_found': (actives_in_top / total_actives) * 100
    }
    
    return results


def load_affinity_data(run_dir, target_id):
    """Load affinity data from source files."""
    
    # Strategy 1: Try temp_data raw file (most likely to have affinity)
    pattern0 = os.path.join(run_dir, "*/temp_data/*_target_ligands_for_feature_calc_raw.csv")
    raw_files = glob.glob(pattern0)
    
    if raw_files:
        try:
            df = pd.read_csv(raw_files[0])
            mol_id_col = None
            affinity_col = None
            
            # Check for molecule ID columns
            for col in ['Compound ChEMBL ID', 'MOLECULE ID', 'molecule_chembl_id']:
                if col in df.columns:
                    mol_id_col = col
                    break
            
            # Check for affinity columns
            if 'Standard Value (nM)' in df.columns:
                affinity_col = 'Standard Value (nM)'
            elif 'standard_value' in df.columns:
                # Assume it's in nM
                df['Standard Value (nM)'] = pd.to_numeric(df['standard_value'], errors='coerce')
                affinity_col = 'Standard Value (nM)'
            elif 'pchembl_value' in df.columns:
                # Convert pChEMBL to nM
                df['Standard Value (nM)'] = 10 ** (9 - pd.to_numeric(df['pchembl_value'], errors='coerce'))
                affinity_col = 'Standard Value (nM)'
            
            if mol_id_col and affinity_col:
                result_df = df[[mol_id_col, affinity_col]].copy()
                result_df.columns = ['MOLECULE ID', 'Standard Value (nM)']
                result_df['Standard Value (nM)'] = pd.to_numeric(
                    result_df['Standard Value (nM)'], errors='coerce'
                )
                return result_df
        except Exception as e:
            logging.debug(f"Error loading affinity from {raw_files[0]}: {e}")
    
    # Strategy 2: Try detailed active distances file
    pattern1 = os.path.join(run_dir, f"{target_id}/results/*/dim_*/*/*detailed_active_distances.csv")
    distance_files = glob.glob(pattern1)
    
    if not distance_files:
        # Try without target_id prefix
        pattern1 = os.path.join(run_dir, "*/results/*/dim_*/*/*detailed_active_distances.csv")
        distance_files = glob.glob(pattern1)
    
    if distance_files:
        try:
            df = pd.read_csv(distance_files[0])
            if 'Standard Value (nM)' in df.columns:
                mol_id_col = 'MOLECULE ID' if 'MOLECULE ID' in df.columns else 'Compound ChEMBL ID'
                if mol_id_col in df.columns:
                    result_df = df[[mol_id_col, 'Standard Value (nM)']].copy()
                    result_df.columns = ['MOLECULE ID', 'Standard Value (nM)']
                    return result_df
        except Exception as e:
            logging.debug(f"Error loading affinity from {distance_files[0]}: {e}")
    
    # Strategy 3: Try target_ligands_calculated file
    pattern2 = os.path.join(run_dir, "*/target_ligands_calculated/*/TyrosineProteinKinaseABL1_P00519_target_ligands_for_calc_*.csv")
    calc_files = glob.glob(pattern2)
    
    if not calc_files:
        # More general pattern
        pattern2 = os.path.join(run_dir, "*/target_ligands_calculated/*/*_target_ligands_for_calc_*.csv")
        calc_files = glob.glob(pattern2)
    
    if calc_files:
        try:
            df = pd.read_csv(calc_files[0])
            mol_id_col = None
            affinity_col = None
            
            for col in ['Compound ChEMBL ID', 'MOLECULE ID', 'molecule_chembl_id']:
                if col in df.columns:
                    mol_id_col = col
                    break
            
            if 'Standard Value (nM)' in df.columns:
                affinity_col = 'Standard Value (nM)'
            elif 'standard_value' in df.columns:
                df['Standard Value (nM)'] = pd.to_numeric(df['standard_value'], errors='coerce')
                affinity_col = 'Standard Value (nM)'
            elif 'pchembl_value' in df.columns:
                df['Standard Value (nM)'] = 10 ** (9 - pd.to_numeric(df['pchembl_value'], errors='coerce'))
                affinity_col = 'Standard Value (nM)'
            
            if mol_id_col and affinity_col:
                result_df = df[[mol_id_col, affinity_col]].copy()
                result_df.columns = ['MOLECULE ID', 'Standard Value (nM)']
                result_df['Standard Value (nM)'] = pd.to_numeric(
                    result_df['Standard Value (nM)'], errors='coerce'
                )
                return result_df
        except Exception as e:
            logging.debug(f"Error loading affinity from {calc_files[0]}: {e}")
    
    # Strategy 4: Try prepared data directory (without target_id prefix)
    pattern3 = os.path.join(run_dir, "*/prepared_data/*_heldout_target_actives.csv")
    active_files = glob.glob(pattern3)
    
    if active_files:
        try:
            df = pd.read_csv(active_files[0])
            # Map common column names
            mol_id_col = None
            affinity_col = None
            
            if 'Compound ChEMBL ID' in df.columns:
                mol_id_col = 'Compound ChEMBL ID'
            elif 'MOLECULE ID' in df.columns:
                mol_id_col = 'MOLECULE ID'
            elif 'molecule_chembl_id' in df.columns:
                mol_id_col = 'molecule_chembl_id'
            
            if 'Standard Value (nM)' in df.columns:
                affinity_col = 'Standard Value (nM)'
            elif 'standard_value' in df.columns:
                affinity_col = 'standard_value'
            elif 'pchembl_value' in df.columns:
                # Convert pChEMBL back to nM
                df_copy = df.copy()
                df_copy['Standard Value (nM)'] = 10 ** (9 - df_copy['pchembl_value'])
                affinity_col = 'Standard Value (nM)'
            
            if mol_id_col and affinity_col:
                result_df = df[[mol_id_col, affinity_col]].copy()
                result_df.columns = ['MOLECULE ID', 'Standard Value (nM)']
                # Convert to numeric, coercing errors
                result_df['Standard Value (nM)'] = pd.to_numeric(
                    result_df['Standard Value (nM)'], errors='coerce'
                )
                return result_df
        except Exception as e:
            logging.debug(f"Error loading affinity from {active_files[0]}: {e}")
    
    # Strategy 5: Try with target_id prefix
    pattern4 = os.path.join(run_dir, f"{target_id}/prepared_data/*_heldout_target_actives.csv")
    active_files = glob.glob(pattern4)
    
    if active_files:
        try:
            df = pd.read_csv(active_files[0])
            mol_id_col = None
            affinity_col = None
            
            if 'Compound ChEMBL ID' in df.columns:
                mol_id_col = 'Compound ChEMBL ID'
            elif 'MOLECULE ID' in df.columns:
                mol_id_col = 'MOLECULE ID'
            
            if 'Standard Value (nM)' in df.columns:
                affinity_col = 'Standard Value (nM)'
            elif 'standard_value' in df.columns:
                affinity_col = 'standard_value'
            
            if mol_id_col and affinity_col:
                result_df = df[[mol_id_col, affinity_col]].copy()
                result_df.columns = ['MOLECULE ID', 'Standard Value (nM)']
                result_df['Standard Value (nM)'] = pd.to_numeric(
                    result_df['Standard Value (nM)'], errors='coerce'
                )
                return result_df
        except Exception as e:
            logging.debug(f"Error loading affinity from {active_files[0]}: {e}")
    
    return None


def analyze_single_run(run_dir, verbose=False):
    """Analyze a single experimental run."""
    run_name = os.path.basename(run_dir)
    
    # Parse configuration
    config_path = os.path.join(run_dir, 'run_config.json')
    if not os.path.exists(config_path):
        if verbose:
            logging.warning(f"No config found for {run_name}")
        return None
    
    config_info = parse_run_config(config_path)
    if not config_info:
        if verbose:
            logging.warning(f"Could not parse config for {run_name}")
        return None
    
    # Find ranked file
    ranked_file = find_ranked_file(run_dir)
    if not ranked_file:
        if verbose:
            logging.warning(f"No ranked file found for {run_name}")
        return None
    
    # Load ranked data
    try:
        ranked_df = pd.read_csv(ranked_file)
    except Exception as e:
        logging.warning(f"Error loading {ranked_file}: {e}")
        return None
    
    # Check required columns
    required_cols = ['RANKING', 'TYPE', 'MOLECULE ID']
    missing_cols = [col for col in required_cols if col not in ranked_df.columns]
    if missing_cols:
        if verbose:
            logging.warning(f"Missing columns in {ranked_file}: {missing_cols}")
            logging.warning(f"Available columns: {list(ranked_df.columns)}")
        return None
    
    # Load affinity data if not already in ranked file
    if 'Standard Value (nM)' not in ranked_df.columns:
        if verbose:
            logging.info(f"No affinity data in ranked file, attempting to load from source...")
        affinity_df = load_affinity_data(run_dir, config_info['target'])
        
        if affinity_df is not None:
            # Merge affinity data with ranked data
            before_merge = len(ranked_df)
            ranked_df = ranked_df.merge(affinity_df, on='MOLECULE ID', how='left')
            if verbose:
                logging.info(f"Successfully merged affinity data for {run_name} ({len(ranked_df)} rows)")
        else:
            if verbose:
                logging.warning(f"Could not load affinity data for {run_name}")
            return None
    
    # Verify we have affinity data for actives
    actives_with_affinity = ranked_df[
        (ranked_df['TYPE'] == 'HELDOUT_ACTIVE') & 
        (ranked_df['Standard Value (nM)'].notna())
    ]
    
    if len(actives_with_affinity) == 0:
        if verbose:
            n_actives = (ranked_df['TYPE'] == 'HELDOUT_ACTIVE').sum()
            logging.warning(f"No actives with affinity data in {run_name} ({n_actives} total actives)")
        return None
    
    # Calculate stratified enrichment
    enrichment = calculate_stratified_enrichment(ranked_df, top_percent=0.01)
    if not enrichment:
        return None
    
    # Combine config info with enrichment results
    result = config_info.copy()
    result['run_dir'] = run_name
    result['ranked_file'] = ranked_file
    
    for tier in TIER_ORDER + ['Overall']:
        if tier in enrichment:
            result[f'{tier}_total'] = enrichment[tier]['total']
            result[f'{tier}_in_top'] = enrichment[tier]['in_top']
            result[f'{tier}_EF'] = enrichment[tier]['ef']
            result[f'{tier}_percent_found'] = enrichment[tier]['percent_found']
    
    return result


def scan_workspace(workspace_dir, seed_filter=None, verbose_sample=True, n_jobs=None):
    """Scan workspace directory for completed runs.
    
    Args:
        workspace_dir: Path to experiment workspace
        seed_filter: Optional seed number to filter by
        verbose_sample: Whether to analyze first run verbosely
        n_jobs: Number of parallel jobs (default: all available CPUs)
    """
    logging.info(f"Scanning workspace: {workspace_dir}")
    
    # Find all run directories
    run_pattern = os.path.join(workspace_dir, "run_seed*")
    run_dirs = glob.glob(run_pattern)
    
    if seed_filter is not None:
        run_dirs = [d for d in run_dirs if f"seed{seed_filter}" in os.path.basename(d)]
    
    logging.info(f"Found {len(run_dirs)} run directories")
    
    if not run_dirs:
        return pd.DataFrame()
    
    # Analyze first run with verbose output for debugging
    if verbose_sample:
        logging.info(f"Attempting to analyze first run with verbose output: {os.path.basename(run_dirs[0])}")
        first_result = analyze_single_run(run_dirs[0], verbose=True)
        if first_result:
            logging.info("✓ First run analyzed successfully!")
        else:
            logging.warning("✗ First run could not be analyzed. Check the warnings above.")
    
    # Determine number of parallel jobs
    if n_jobs is None:
        n_jobs = max(1, cpu_count() - 1)  # Leave one CPU free
    
    logging.info(f"Analyzing {len(run_dirs)} runs using {n_jobs} parallel workers...")
    
    # Parallelize the analysis using multiprocessing
    results = []
    with Pool(processes=n_jobs) as pool:
        # Use imap for progress bar compatibility
        for result in tqdm(pool.imap(analyze_single_run_wrapper, run_dirs), 
                          total=len(run_dirs), 
                          desc="Analyzing runs"):
            if result:
                results.append(result)
    
    logging.info(f"Successfully analyzed {len(results)} runs")
    return pd.DataFrame(results)


def analyze_single_run_wrapper(run_dir):
    """Wrapper for analyze_single_run to work with multiprocessing.
    
    This wrapper is needed because multiprocessing requires picklable functions.
    """
    return analyze_single_run(run_dir, verbose=False)


def create_summary_table(df_results):
    """Create summary table comparing configurations."""
    if df_results.empty:
        return pd.DataFrame()
    
    # Group by configuration (method, dimension, hyperparameters)
    group_cols = ['representation', 'dr_method', 'dimension', 'n_neighbors', 'min_dist']
    group_cols = [col for col in group_cols if col in df_results.columns]
    
    # Calculate mean and std for each metric
    agg_dict = {}
    for tier in TIER_ORDER + ['Overall']:
        ef_col = f'{tier}_EF'
        pct_col = f'{tier}_percent_found'
        total_col = f'{tier}_total'
        
        if ef_col in df_results.columns:
            agg_dict[f'{tier}_EF_mean'] = (ef_col, 'mean')
            agg_dict[f'{tier}_EF_std'] = (ef_col, 'std')
        if pct_col in df_results.columns:
            agg_dict[f'{tier}_pct_mean'] = (pct_col, 'mean')
        if total_col in df_results.columns:
            agg_dict[f'{tier}_total_mean'] = (total_col, 'mean')
    
    summary = df_results.groupby(group_cols).agg(**agg_dict).reset_index()
    
    # Sort by overall EF
    if 'Overall_EF_mean' in summary.columns:
        summary = summary.sort_values('Overall_EF_mean', ascending=False)
    
    return summary


# =============================================================================
# PLOTTING HELPER FUNCTIONS
# =============================================================================
def save_figure(fig, output_dir, basename, dpi=300):
    """Save figure in both PNG and PDF formats.
    
    Args:
        fig: matplotlib figure object
        output_dir: directory to save figures
        basename: base filename (without extension)
        dpi: resolution for PNG (default: 300)
    """
    png_path = os.path.join(output_dir, f'{basename}.png')
    pdf_path = os.path.join(output_dir, f'{basename}.pdf')
    
    fig.savefig(png_path, dpi=dpi, bbox_inches='tight')
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
    
    logging.info(f"  Saved: {basename}.png and {basename}.pdf")


def plot_pca_vs_umap_comparison(df_results, output_dir):
    """Compare PCA vs UMAP performance across potency tiers.
    
    For UMAP, selects only the best hyperparameters for each representation type.
    """
    logging.info("Generating PCA vs UMAP comparison plots...")
    
    # Create method-representation combinations
    df_results['method_repr'] = df_results.apply(
        lambda row: f"{row['dr_method']}-{row['representation']}" 
        if pd.notna(row.get('representation')) else row['dr_method'],
        axis=1
    )
    
    # For UMAP, identify best hyperparameters for each representation
    # Best = highest mean High-potent EF (most important for quality)
    best_umap_configs = {}
    
    # UMAP-Euclidean with features
    df_umap_euc = df_results[
        (df_results['dr_method'] == 'UMAP-Euclidean') &
        (df_results['representation'] == 'features')
    ]
    if not df_umap_euc.empty and 'n_neighbors' in df_umap_euc.columns:
        # Group by hyperparameters and find best config
        hyperparam_cols = [c for c in ['n_neighbors', 'min_dist'] if c in df_umap_euc.columns]
        if hyperparam_cols:
            grouped = df_umap_euc.groupby(hyperparam_cols)['High_EF'].mean()
            if not grouped.empty:
                best_params = grouped.idxmax()
                best_umap_configs['UMAP-Euclidean-features'] = best_params
                logging.info(f"Best UMAP-Euclidean (features): {best_params}")
    
    # UMAP-Jaccard with fingerprints
    df_umap_jac = df_results[
        (df_results['dr_method'] == 'UMAP-Jaccard') &
        (df_results['representation'] == 'fingerprints')
    ]
    if not df_umap_jac.empty and 'n_neighbors' in df_umap_jac.columns:
        # Group by hyperparameters and find best config
        hyperparam_cols = [c for c in ['n_neighbors', 'min_dist'] if c in df_umap_jac.columns]
        if hyperparam_cols:
            grouped = df_umap_jac.groupby(hyperparam_cols)['High_EF'].mean()
            if not grouped.empty:
                best_params = grouped.idxmax()
                best_umap_configs['UMAP-Jaccard-fingerprints'] = best_params
                logging.info(f"Best UMAP-Jaccard (fingerprints): {best_params}")
    
    # Define the configurations to compare
    configs_to_compare = {
        'PCA-features': {
            'method': 'PCA', 
            'repr': 'features', 
            'label': 'PCA (Features)',
            'short_label': 'PCA-feat',
            'filter_params': None
        },
        'PCA-fingerprints': {
            'method': 'PCA', 
            'repr': 'fingerprints', 
            'label': 'PCA (Fingerprints)',
            'short_label': 'PCA-fing',
            'filter_params': None
        },
        'UMAP-Euclidean-features': {
            'method': 'UMAP-Euclidean', 
            'repr': 'features', 
            'label': 'UMAP-Euclidean (Features)',
            'short_label': 'UMAP-Euc',
            'filter_params': best_umap_configs.get('UMAP-Euclidean-features')
        },
        'UMAP-Jaccard-fingerprints': {
            'method': 'UMAP-Jaccard', 
            'repr': 'fingerprints', 
            'label': 'UMAP-Jaccard (Fingerprints)',
            'short_label': 'UMAP-Jac',
            'filter_params': best_umap_configs.get('UMAP-Jaccard-fingerprints')
        }
    }
    
    # Filter to only the configurations we want
    df_methods_list = []
    for config_key, config_info in configs_to_compare.items():
        df_config = df_results[
            (df_results['dr_method'] == config_info['method']) &
            (df_results['representation'] == config_info['repr'])
        ]
        
        # Apply hyperparameter filtering for UMAP
        if config_info['filter_params'] is not None:
            if isinstance(config_info['filter_params'], tuple):
                # Multiple hyperparameters
                n_neighbors, min_dist = config_info['filter_params']
                df_config = df_config[
                    (df_config['n_neighbors'] == n_neighbors) &
                    (df_config['min_dist'] == min_dist)
                ]
            else:
                # Single hyperparameter (shouldn't happen but handle it)
                df_config = df_config[df_config['n_neighbors'] == config_info['filter_params']]
        
        if not df_config.empty:
            df_methods_list.append(df_config)
    
    if not df_methods_list:
        logging.warning("No matching method-representation combinations found for comparison")
        return
    
    df_methods = pd.concat(df_methods_list, ignore_index=True)
    
    # Plot 1: Mean EF by tier for all method-representation combinations
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(TIER_ORDER))
    n_configs = len(configs_to_compare)
    width = 0.8 / n_configs
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    
    config_idx = 0
    for config_key, config_info in configs_to_compare.items():
        df_config = df_methods[
            (df_methods['dr_method'] == config_info['method']) &
            (df_methods['representation'] == config_info['repr'])
        ]
        
        if df_config.empty:
            continue
        
        # Create label with hyperparameters for UMAP
        label = config_info['label']
        if config_info['filter_params'] is not None and isinstance(config_info['filter_params'], tuple):
            nn, md = config_info['filter_params']
            label += f"\n(nn={nn}, md={md})"
        
        means = [df_config[f'{tier}_EF'].mean() for tier in TIER_ORDER]
        stds = [df_config[f'{tier}_EF'].std() for tier in TIER_ORDER]
        
        offset = (config_idx - n_configs/2 + 0.5) * width
        bars = ax.bar(x + offset, means, width, yerr=stds, 
                     label=label, alpha=0.8, capsize=4,
                     color=colors[config_idx % len(colors)])
        
        # Add value labels
        for bar, mean_val in zip(bars, means):
            if not np.isnan(mean_val) and mean_val > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                       f'{mean_val:.1f}', ha='center', va='bottom', fontsize=8)
        
        config_idx += 1
    
    ax.set_xlabel('Potency Tier', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Enrichment Factor @ 1%', fontsize=12, fontweight='bold')
    ax.set_title('Method-Representation Comparison: Potency-Stratified Enrichment\n(UMAP: Best Hyperparameters Only)',
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    
    # Add potency ranges to x-axis labels
    tier_labels = [f"{tier}\n({POTENCY_TIERS[tier][0]}-{POTENCY_TIERS[tier][1]} nM)" 
                   for tier in TIER_ORDER]
    ax.set_xticklabels(tier_labels)
    
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, output_dir, 'pca_vs_umap_by_tier')
    plt.close()
    
    # =========================================================================
    # NEW: Generate dimension-separated versions of the same plot
    # =========================================================================
    dimensions = sorted(df_methods['dimension'].unique())
    for dim in dimensions:
        df_dim = df_methods[df_methods['dimension'] == dim]
        
        if df_dim.empty:
            continue
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(TIER_ORDER))
        n_configs = len(configs_to_compare)
        width = 0.8 / n_configs
        
        config_idx = 0
        for config_key, config_info in configs_to_compare.items():
            df_config = df_dim[
                (df_dim['dr_method'] == config_info['method']) &
                (df_dim['representation'] == config_info['repr'])
            ]
            
            if df_config.empty:
                continue
            
            label = config_info['label']
            if config_info['filter_params'] is not None and isinstance(config_info['filter_params'], tuple):
                nn, md = config_info['filter_params']
                label += f"\n(nn={nn}, md={md})"
            
            means = []
            stds = []
            for tier in TIER_ORDER:
                ef_col = f'{tier}_EF'
                if ef_col in df_config.columns:
                    means.append(df_config[ef_col].mean())
                    stds.append(df_config[ef_col].std())
                else:
                    means.append(0)
                    stds.append(0)
            
            offset = (config_idx - n_configs/2) * width + width/2
            bars = ax.bar(x + offset, means, width, yerr=stds,
                         label=label, alpha=0.8, capsize=4,
                         color=colors[config_idx % len(colors)])
            
            for bar, mean_val in zip(bars, means):
                if not np.isnan(mean_val) and mean_val > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                           f'{mean_val:.1f}', ha='center', va='bottom', fontsize=8)
            
            config_idx += 1
        
        ax.set_xlabel('Potency Tier', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean Enrichment Factor @ 1%', fontsize=12, fontweight='bold')
        ax.set_title(f'Method-Representation Comparison: Potency-Stratified Enrichment\n(Dimensionality: {dim}D, UMAP: Best Hyperparameters Only)',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        
        tier_labels = [f"{tier}\n({POTENCY_TIERS[tier][0]}-{POTENCY_TIERS[tier][1]} nM)" 
                       for tier in TIER_ORDER]
        ax.set_xticklabels(tier_labels)
        
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        save_figure(fig, output_dir, f'pca_vs_umap_by_tier_dim{dim}')
        plt.close()
    
    logging.info("PCA vs UMAP comparison plots saved (all dimensions + dimension-separated)")



def plot_dimensionality_impact(df_results, output_dir):
    """Analyze impact of dimensionality on potency-stratified enrichment.
    
    For UMAP, only uses the best hyperparameters for each representation type.
    """
    logging.info("Generating dimensionality impact plots...")
    
    if 'dimension' not in df_results.columns:
        logging.warning("No dimension data found")
        return
    
    # Get valid dimensions (excluding None/NaN)
    dimensions = sorted([d for d in df_results['dimension'].unique() if d is not None and not pd.isna(d)])
    
    if len(dimensions) == 0:
        logging.warning("No valid dimension data found")
        return
    
    # =========================================================================
    # BEST UMAP HYPERPARAMETER SELECTION (same as plot_pca_vs_umap_comparison)
    # =========================================================================
    best_umap_configs = {}
    
    # Find best UMAP-Euclidean (features) hyperparameters
    df_umap_euc = df_results[
        (df_results['dr_method'] == 'UMAP-Euclidean') & 
        (df_results['representation'] == 'features')
    ]
    if not df_umap_euc.empty and 'n_neighbors' in df_umap_euc.columns and 'High_EF' in df_umap_euc.columns:
        grouped = df_umap_euc.groupby(['n_neighbors', 'min_dist'])['High_EF'].mean()
        if not grouped.empty:
            best_params = grouped.idxmax()
            best_umap_configs['UMAP-Euclidean-features'] = best_params
            logging.info(f"Best UMAP-Euclidean (features) for dimensionality plot: {best_params}")
    
    # Find best UMAP-Jaccard (fingerprints) hyperparameters
    df_umap_jac = df_results[
        (df_results['dr_method'] == 'UMAP-Jaccard') & 
        (df_results['representation'] == 'fingerprints')
    ]
    if not df_umap_jac.empty and 'n_neighbors' in df_umap_jac.columns and 'High_EF' in df_umap_jac.columns:
        grouped = df_umap_jac.groupby(['n_neighbors', 'min_dist'])['High_EF'].mean()
        if not grouped.empty:
            best_params = grouped.idxmax()
            best_umap_configs['UMAP-Jaccard-fingerprints'] = best_params
            logging.info(f"Best UMAP-Jaccard (fingerprints) for dimensionality plot: {best_params}")
    
    # Filter df_results to include only best UMAP hyperparameters
    df_filtered_list = []
    
    # Add all PCA results
    df_pca = df_results[df_results['dr_method'] == 'PCA']
    if not df_pca.empty:
        df_filtered_list.append(df_pca)
    
    # Add best UMAP-Euclidean (features)
    if 'UMAP-Euclidean-features' in best_umap_configs:
        nn, md = best_umap_configs['UMAP-Euclidean-features']
        df_best_umap_euc = df_results[
            (df_results['dr_method'] == 'UMAP-Euclidean') &
            (df_results['representation'] == 'features') &
            (df_results['n_neighbors'] == nn) &
            (df_results['min_dist'] == md)
        ]
        if not df_best_umap_euc.empty:
            df_filtered_list.append(df_best_umap_euc)
    
    # Add best UMAP-Jaccard (fingerprints)
    if 'UMAP-Jaccard-fingerprints' in best_umap_configs:
        nn, md = best_umap_configs['UMAP-Jaccard-fingerprints']
        df_best_umap_jac = df_results[
            (df_results['dr_method'] == 'UMAP-Jaccard') &
            (df_results['representation'] == 'fingerprints') &
            (df_results['n_neighbors'] == nn) &
            (df_results['min_dist'] == md)
        ]
        if not df_best_umap_jac.empty:
            df_filtered_list.append(df_best_umap_jac)
    
    if not df_filtered_list:
        logging.warning("No data after filtering for best hyperparameters")
        return
    
    df_filtered = pd.concat(df_filtered_list, ignore_index=True)
    
    # Plot 1: EF by dimension for each potency tier (BEST HYPERPARAMETERS ONLY)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    
    for idx, tier in enumerate(TIER_ORDER):
        ax = axes[idx]
        ef_col = f'{tier}_EF'
        
        if ef_col not in df_filtered.columns:
            continue
        
        # Separate PCA and UMAP
        for method in ['PCA', 'UMAP']:
            if method == 'PCA':
                df_method = df_filtered[df_filtered['dr_method'] == 'PCA']
                marker, color, label = 'o', '#3498db', 'PCA'
            else:
                df_method = df_filtered[df_filtered['dr_method'].str.contains('UMAP', na=False)]
                marker, color, label = 's', '#e74c3c', 'UMAP (Best Params)'
            
            if df_method.empty:
                continue
            
            # Calculate mean and std for each dimension
            dim_stats = df_method.groupby('dimension')[ef_col].agg(['mean', 'std']).reset_index()
            
            ax.errorbar(dim_stats['dimension'], dim_stats['mean'], 
                       yerr=dim_stats['std'], marker=marker, 
                       color=color, label=label, linewidth=2, 
                       markersize=8, capsize=5, alpha=0.8)
        
        ax.set_title(f'{tier} Potency', fontweight='bold', fontsize=12)
        ax.set_xlabel('Dimensionality', fontsize=11, fontweight='bold')
        if idx == 0:
            ax.set_ylabel('Mean EF@1%', fontsize=11, fontweight='bold')
        ax.set_xticks(dimensions)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Impact of Dimensionality on Potency-Stratified Enrichment\n(UMAP: Best Hyperparameters Only)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, output_dir, 'dimensionality_impact_by_tier')
    plt.close()
    
    # Plot 2: Heatmap showing method×dimension×tier performance (BEST HYPERPARAMETERS ONLY)
    methods_for_heatmap = []
    for method in ['PCA', 'UMAP']:
        for dim in dimensions:
            if method == 'PCA':
                df_subset = df_filtered[(df_filtered['dr_method'] == 'PCA') & 
                                      (df_filtered['dimension'] == dim)]
            else:
                df_subset = df_filtered[(df_filtered['dr_method'].str.contains('UMAP', na=False)) & 
                                      (df_filtered['dimension'] == dim)]
            
            if not df_subset.empty:
                row_data = {'Method': method, 'Dim': dim}
                for tier in TIER_ORDER:
                    row_data[tier] = df_subset[f'{tier}_EF'].mean()
                methods_for_heatmap.append(row_data)
    
    if methods_for_heatmap:
        df_heatmap = pd.DataFrame(methods_for_heatmap)
        df_heatmap['Method_Dim'] = df_heatmap['Method'] + '-' + df_heatmap['Dim'].astype(str) + 'D'
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        
        heatmap_data = df_heatmap[TIER_ORDER].values
        
        im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto')
        
        # Set ticks
        ax.set_xticks(np.arange(len(TIER_ORDER)))
        ax.set_yticks(np.arange(len(df_heatmap)))
        ax.set_xticklabels(TIER_ORDER)
        ax.set_yticklabels(df_heatmap['Method_Dim'])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Mean EF@1%', rotation=270, labelpad=20, fontweight='bold')
        
        # Add values to cells
        for i in range(len(df_heatmap)):
            for j in range(len(TIER_ORDER)):
                val = heatmap_data[i, j]
                if not np.isnan(val):
                    text = ax.text(j, i, f'{val:.1f}',
                                 ha="center", va="center", color="black", fontsize=10)
        
        ax.set_title('Method × Dimensionality × Potency Tier Performance\n(UMAP: Best Hyperparameters Only)',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Potency Tier', fontsize=12, fontweight='bold')
        ax.set_ylabel('Method-Dimension', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        save_figure(fig, output_dir, 'method_dimension_tier_heatmap')
        plt.close()
    
    # Plot 3: Dimensionality preference by tier - using BEST HYPERPARAMETERS ONLY
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(dimensions))
    width = 0.25
    
    for idx, tier in enumerate(TIER_ORDER):
        ef_col = f'{tier}_EF'
        if ef_col not in df_filtered.columns:
            continue
        
        means = []
        for dim in dimensions:
            dim_data = df_filtered[df_filtered['dimension'] == dim][ef_col]
            means.append(dim_data.mean() if not dim_data.empty else 0)
        
        offset = (idx - 1) * width
        bars = ax.bar(x + offset, means, width, label=tier, 
                     color=TIER_COLORS[tier], alpha=0.8)
        
        # Add value labels
        for bar, mean_val in zip(bars, means):
            if not np.isnan(mean_val) and mean_val > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                       f'{mean_val:.1f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Dimensionality', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean EF@1% (PCA + Best UMAP)', fontsize=12, fontweight='bold')
    ax.set_title('Optimal Dimensionality by Potency Tier\n(UMAP: Best Hyperparameters Only)',
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{d}D' for d in dimensions])
    ax.legend(title='Potency Tier')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, output_dir, 'optimal_dimensionality_by_tier')
    plt.close()
    
    logging.info("Dimensionality impact plots saved")


def plot_stratified_comparison(df_results, output_dir):
    """Create comparison plots for stratified enrichment."""
    os.makedirs(output_dir, exist_ok=True)
    
    if df_results.empty:
        logging.warning("No results to plot")
        return
    
    logging.info("Generating stratified enrichment plots...")
    
    # NEW PLOT 1: PCA vs UMAP comparison across potency tiers
    plot_pca_vs_umap_comparison(df_results, output_dir)
    
    # NEW PLOT 2: Dimensionality impact analysis
    plot_dimensionality_impact(df_results, output_dir)
    
    # EXISTING PLOT 3: UMAP hyperparameter analysis
    if 'n_neighbors' in df_results.columns:
        df_umap = df_results[df_results['dr_method'].str.contains('UMAP', na=False)].copy()
        
        if not df_umap.empty:
            # Create plot data
            plot_data = []
            for tier in TIER_ORDER:
                ef_col = f'{tier}_EF'
                if ef_col in df_umap.columns:
                    for _, row in df_umap.iterrows():
                        plot_data.append({
                            'Tier': tier,
                            'EF@1%': row[ef_col],
                            'n_neighbors': row['n_neighbors'],
                            'min_dist': row['min_dist'],
                            'dimension': row['dimension'],
                            'config': f"nn={row['n_neighbors']}, md={row['min_dist']}"
                        })
            
            df_plot = pd.DataFrame(plot_data)
            
            # Plot 1: EF by tier, faceted by n_neighbors
            fig, axes = plt.subplots(1, len(df_umap['n_neighbors'].unique()), 
                                     figsize=(5*len(df_umap['n_neighbors'].unique()), 5),
                                     sharey=True)
            
            if len(df_umap['n_neighbors'].unique()) == 1:
                axes = [axes]
            
            for idx, nn in enumerate(sorted(df_umap['n_neighbors'].unique())):
                ax = axes[idx]
                df_subset = df_plot[df_plot['n_neighbors'] == nn]
                
                # Group by tier and calculate mean
                tier_means = df_subset.groupby('Tier')['EF@1%'].mean()
                
                bars = ax.bar(TIER_ORDER, [tier_means.get(t, 0) for t in TIER_ORDER],
                             color=[TIER_COLORS[t] for t in TIER_ORDER])
                
                ax.set_title(f'n_neighbors = {nn}', fontweight='bold')
                ax.set_xlabel('Potency Tier')
                if idx == 0:
                    ax.set_ylabel('Mean Enrichment Factor @ 1%')
                ax.grid(axis='y', alpha=0.3)
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    if not np.isnan(height):
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.1f}',
                               ha='center', va='bottom', fontsize=9)
            
            plt.suptitle('Potency-Stratified Enrichment: Impact of n_neighbors', 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            save_figure(fig, output_dir, 'stratified_ef_by_nn')
            plt.close()
            
            # Generate dimension-separated versions
            for dim in sorted(df_umap['dimension'].unique()):
                df_dim = df_plot[df_plot['dimension'] == dim]
                
                if df_dim.empty:
                    continue
                
                nn_values_dim = sorted(df_dim['n_neighbors'].unique())
                fig, axes = plt.subplots(1, len(nn_values_dim), 
                                         figsize=(5*len(nn_values_dim), 5),
                                         sharey=True)
                
                if len(nn_values_dim) == 1:
                    axes = [axes]
                
                for idx, nn in enumerate(nn_values_dim):
                    ax = axes[idx]
                    df_subset = df_dim[df_dim['n_neighbors'] == nn]
                    
                    tier_means = df_subset.groupby('Tier')['EF@1%'].mean()
                    
                    bars = ax.bar(TIER_ORDER, [tier_means.get(t, 0) for t in TIER_ORDER],
                                 color=[TIER_COLORS[t] for t in TIER_ORDER])
                    
                    ax.set_title(f'n_neighbors = {nn}', fontweight='bold')
                    ax.set_xlabel('Potency Tier')
                    if idx == 0:
                        ax.set_ylabel('Mean Enrichment Factor @ 1%')
                    ax.grid(axis='y', alpha=0.3)
                    
                    for bar in bars:
                        height = bar.get_height()
                        if not np.isnan(height):
                            ax.text(bar.get_x() + bar.get_width()/2., height,
                                   f'{height:.1f}',
                                   ha='center', va='bottom', fontsize=9)
                
                plt.suptitle(f'Potency-Stratified Enrichment: Impact of n_neighbors (Dim {dim}D)', 
                            fontsize=14, fontweight='bold')
                plt.tight_layout()
                save_figure(fig, output_dir, f'stratified_ef_by_nn_dim{dim}')
                plt.close()
            
            # Plot 2: Percent Found by Tier
            fig, ax = plt.subplots(figsize=(10, 6))
            
            x = np.arange(len(TIER_ORDER))
            width = 0.15
            
            nn_values = sorted(df_umap['n_neighbors'].unique())
            for idx, nn in enumerate(nn_values):
                df_subset = df_umap[df_umap['n_neighbors'] == nn]
                means = [df_subset[f'{tier}_percent_found'].mean() for tier in TIER_ORDER]
                
                offset = (idx - len(nn_values)/2) * width + width/2
                ax.bar(x + offset, means, width, label=f'nn={nn}')
            
            ax.set_xlabel('Potency Tier', fontweight='bold')
            ax.set_ylabel('% of Tier Found in Top 1%', fontweight='bold')
            ax.set_title('Potency-Stratified Recovery: Percent of Each Tier Found', 
                        fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(TIER_ORDER)
            ax.legend(title='n_neighbors', loc='upper right')
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            save_figure(fig, output_dir, 'stratified_percent_found_by_nn')
            plt.close()
            
            # Generate dimension-separated versions
            for dim in sorted(df_umap['dimension'].unique()):
                df_dim = df_umap[df_umap['dimension'] == dim]
                
                if df_dim.empty:
                    continue
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                x = np.arange(len(TIER_ORDER))
                width = 0.15
                
                nn_values_dim = sorted(df_dim['n_neighbors'].unique())
                for idx, nn in enumerate(nn_values_dim):
                    df_subset = df_dim[df_dim['n_neighbors'] == nn]
                    means = [df_subset[f'{tier}_percent_found'].mean() for tier in TIER_ORDER]
                    
                    offset = (idx - len(nn_values_dim)/2) * width + width/2
                    ax.bar(x + offset, means, width, label=f'nn={nn}')
                
                ax.set_xlabel('Potency Tier', fontweight='bold')
                ax.set_ylabel('% of Tier Found in Top 1%', fontweight='bold')
                ax.set_title(f'Potency-Stratified Recovery: Percent of Each Tier Found (Dim {dim}D)', 
                            fontsize=14, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(TIER_ORDER)
                ax.legend(title='n_neighbors', loc='upper right')
                ax.grid(axis='y', alpha=0.3)
                
                plt.tight_layout()
                save_figure(fig, output_dir, f'stratified_percent_found_by_nn_dim{dim}')
                plt.close()
            
            # =====================================================================
            # PLOT 3: EF by tier, faceted by min_dist (similar to n_neighbors)
            # =====================================================================
            if 'min_dist' in df_umap.columns:
                fig, axes = plt.subplots(1, len(df_umap['min_dist'].unique()), 
                                         figsize=(5*len(df_umap['min_dist'].unique()), 5),
                                         sharey=True)
                
                if len(df_umap['min_dist'].unique()) == 1:
                    axes = [axes]
                
                for idx, md in enumerate(sorted(df_umap['min_dist'].unique())):
                    ax = axes[idx]
                    df_subset = df_plot[df_plot['min_dist'] == md]
                    
                    # Group by tier and calculate mean
                    tier_means = df_subset.groupby('Tier')['EF@1%'].mean()
                    
                    bars = ax.bar(TIER_ORDER, [tier_means.get(t, 0) for t in TIER_ORDER],
                                 color=[TIER_COLORS[t] for t in TIER_ORDER])
                    
                    ax.set_title(f'min_dist = {md}', fontweight='bold')
                    ax.set_xlabel('Potency Tier')
                    if idx == 0:
                        ax.set_ylabel('Mean Enrichment Factor @ 1%')
                    ax.grid(axis='y', alpha=0.3)
                    
                    # Add value labels on bars
                    for bar in bars:
                        height = bar.get_height()
                        if not np.isnan(height):
                            ax.text(bar.get_x() + bar.get_width()/2., height,
                                   f'{height:.1f}',
                                   ha='center', va='bottom', fontsize=9)
                
                plt.suptitle('Potency-Stratified Enrichment: Impact of min_dist', 
                            fontsize=14, fontweight='bold')
                plt.tight_layout()
                save_figure(fig, output_dir, 'stratified_ef_by_min_dist')
                plt.close()
                
                # Generate dimension-separated versions
                for dim in sorted(df_umap['dimension'].unique()):
                    df_dim = df_plot[df_plot['dimension'] == dim]
                    
                    if df_dim.empty:
                        continue
                    
                    md_values_dim = sorted(df_dim['min_dist'].unique())
                    fig, axes = plt.subplots(1, len(md_values_dim), 
                                             figsize=(5*len(md_values_dim), 5),
                                             sharey=True)
                    
                    if len(md_values_dim) == 1:
                        axes = [axes]
                    
                    for idx, md in enumerate(md_values_dim):
                        ax = axes[idx]
                        df_subset = df_dim[df_dim['min_dist'] == md]
                        
                        tier_means = df_subset.groupby('Tier')['EF@1%'].mean()
                        
                        bars = ax.bar(TIER_ORDER, [tier_means.get(t, 0) for t in TIER_ORDER],
                                     color=[TIER_COLORS[t] for t in TIER_ORDER])
                        
                        ax.set_title(f'min_dist = {md}', fontweight='bold')
                        ax.set_xlabel('Potency Tier')
                        if idx == 0:
                            ax.set_ylabel('Mean Enrichment Factor @ 1%')
                        ax.grid(axis='y', alpha=0.3)
                        
                        for bar in bars:
                            height = bar.get_height()
                            if not np.isnan(height):
                                ax.text(bar.get_x() + bar.get_width()/2., height,
                                       f'{height:.1f}',
                                       ha='center', va='bottom', fontsize=9)
                    
                    plt.suptitle(f'Potency-Stratified Enrichment: Impact of min_dist (Dim {dim}D)', 
                                fontsize=14, fontweight='bold')
                    plt.tight_layout()
                    save_figure(fig, output_dir, f'stratified_ef_by_min_dist_dim{dim}')
                    plt.close()
                
                # =====================================================================
                # PLOT 4: Percent Found by Tier, faceted by min_dist
                # =====================================================================
                fig, ax = plt.subplots(figsize=(10, 6))
                
                x = np.arange(len(TIER_ORDER))
                width = 0.15
                
                md_values = sorted(df_umap['min_dist'].unique())
                for idx, md in enumerate(md_values):
                    df_subset = df_umap[df_umap['min_dist'] == md]
                    means = [df_subset[f'{tier}_percent_found'].mean() for tier in TIER_ORDER]
                    
                    offset = (idx - len(md_values)/2) * width + width/2
                    ax.bar(x + offset, means, width, label=f'md={md}')
                
                ax.set_xlabel('Potency Tier', fontweight='bold')
                ax.set_ylabel('% of Tier Found in Top 1%', fontweight='bold')
                ax.set_title('Potency-Stratified Recovery: Percent of Each Tier Found', 
                            fontsize=14, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(TIER_ORDER)
                ax.legend(title='min_dist', loc='upper right')
                ax.grid(axis='y', alpha=0.3)
                
                plt.tight_layout()
                save_figure(fig, output_dir, 'stratified_percent_found_by_min_dist')
                plt.close()
                
                # Generate dimension-separated versions
                for dim in sorted(df_umap['dimension'].unique()):
                    df_dim = df_umap[df_umap['dimension'] == dim]
                    
                    if df_dim.empty:
                        continue
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    x = np.arange(len(TIER_ORDER))
                    width = 0.15
                    
                    md_values_dim = sorted(df_dim['min_dist'].unique())
                    for idx, md in enumerate(md_values_dim):
                        df_subset = df_dim[df_dim['min_dist'] == md]
                        means = [df_subset[f'{tier}_percent_found'].mean() for tier in TIER_ORDER]
                        
                        offset = (idx - len(md_values_dim)/2) * width + width/2
                        ax.bar(x + offset, means, width, label=f'md={md}')
                    
                    ax.set_xlabel('Potency Tier', fontweight='bold')
                    ax.set_ylabel('% of Tier Found in Top 1%', fontweight='bold')
                    ax.set_title(f'Potency-Stratified Recovery: Percent of Each Tier Found (Dim {dim}D)', 
                                fontsize=14, fontweight='bold')
                    ax.set_xticks(x)
                    ax.set_xticklabels(TIER_ORDER)
                    ax.legend(title='min_dist', loc='upper right')
                    ax.grid(axis='y', alpha=0.3)
                    
                    plt.tight_layout()
                    save_figure(fig, output_dir, f'stratified_percent_found_by_min_dist_dim{dim}')
                    plt.close()
    
    # 3. Scatter plot: Overall EF vs High-Potent EF
    if 'Overall_EF' in df_results.columns and 'High_EF' in df_results.columns:
        fig, ax = plt.subplots(figsize=(8, 8))
        
        scatter_data = df_results.dropna(subset=['Overall_EF', 'High_EF'])
        
        if not scatter_data.empty:
            # Color by n_neighbors if available
            if 'n_neighbors' in scatter_data.columns:
                for nn in sorted(scatter_data['n_neighbors'].unique()):
                    df_nn = scatter_data[scatter_data['n_neighbors'] == nn]
                    ax.scatter(df_nn['Overall_EF'], df_nn['High_EF'], 
                              label=f'nn={nn}', alpha=0.6, s=100)
            else:
                ax.scatter(scatter_data['Overall_EF'], scatter_data['High_EF'], 
                          alpha=0.6, s=100)
            
            # Add diagonal line (where Overall = High-Potent)
            max_val = max(scatter_data['Overall_EF'].max(), scatter_data['High_EF'].max())
            ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='Equal')
            
            ax.set_xlabel('Overall EF@1%', fontweight='bold')
            ax.set_ylabel('High-Potent EF@1%', fontweight='bold')
            ax.set_title('Quality vs. Quantity Trade-off:\nHigh-Potent vs Overall Enrichment',
                        fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            save_figure(fig, output_dir, 'quality_vs_quantity_tradeoff')
            plt.close()
            
            # Generate dimension-separated versions
            for dim in sorted(scatter_data['dimension'].unique()):
                df_dim = scatter_data[scatter_data['dimension'] == dim]
                
                if df_dim.empty:
                    continue
                
                fig, ax = plt.subplots(figsize=(8, 8))
                
                # Color by n_neighbors if available
                if 'n_neighbors' in df_dim.columns:
                    for nn in sorted(df_dim['n_neighbors'].unique()):
                        df_nn = df_dim[df_dim['n_neighbors'] == nn]
                        ax.scatter(df_nn['Overall_EF'], df_nn['High_EF'], 
                                  label=f'nn={nn}', alpha=0.6, s=100)
                else:
                    ax.scatter(df_dim['Overall_EF'], df_dim['High_EF'], 
                              alpha=0.6, s=100)
                
                # Add diagonal line
                max_val = max(df_dim['Overall_EF'].max(), df_dim['High_EF'].max())
                ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='Equal')
                
                ax.set_xlabel('Overall EF@1%', fontweight='bold')
                ax.set_ylabel('High-Potent EF@1%', fontweight='bold')
                ax.set_title(f'Quality vs. Quantity Trade-off (Dim {dim}D):\nHigh-Potent vs Overall Enrichment',
                            fontsize=14, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                save_figure(fig, output_dir, f'quality_vs_quantity_tradeoff_dim{dim}')
                plt.close()
    
    logging.info(f"Plots saved to {output_dir}")


def generate_report(df_results, summary_df, output_dir):
    """Generate text report with key findings."""
    report_path = os.path.join(output_dir, 'potency_stratified_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("POTENCY-STRATIFIED ENRICHMENT ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total runs analyzed: {len(df_results)}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("POTENCY TIER DEFINITIONS\n")
        f.write("-" * 80 + "\n")
        for tier, (low, high) in POTENCY_TIERS.items():
            f.write(f"  {tier:12s}: {low:>8.1f} - {high:>8.1f} nM\n")
        f.write("\n")
        
        if not summary_df.empty:
            f.write("-" * 80 + "\n")
            f.write("TOP CONFIGURATIONS BY OVERALL ENRICHMENT\n")
            f.write("-" * 80 + "\n\n")
            
            # Show top 10 configs
            top_n = min(10, len(summary_df))
            for idx, row in summary_df.head(top_n).iterrows():
                f.write(f"Rank {idx+1}:\n")
                
                # Config details
                if 'dr_method' in row:
                    f.write(f"  Method: {row['dr_method']}")
                if 'dimension' in row:
                    f.write(f", Dim: {row['dimension']}")
                if 'n_neighbors' in row and pd.notna(row['n_neighbors']):
                    f.write(f", nn: {row['n_neighbors']}")
                if 'min_dist' in row and pd.notna(row['min_dist']):
                    f.write(f", md: {row['min_dist']}")
                f.write("\n")
                
                # Enrichment by tier
                f.write(f"  Enrichment Factors (mean ± std):\n")
                for tier in TIER_ORDER + ['Overall']:
                    ef_mean_col = f'{tier}_EF_mean'
                    ef_std_col = f'{tier}_EF_std'
                    if ef_mean_col in row and pd.notna(row[ef_mean_col]):
                        ef_mean = row[ef_mean_col]
                        ef_std = row.get(ef_std_col, np.nan)
                        if pd.notna(ef_std):
                            f.write(f"    {tier:12s}: {ef_mean:6.2f} ± {ef_std:5.2f}\n")
                        else:
                            f.write(f"    {tier:12s}: {ef_mean:6.2f}\n")
                f.write("\n")
            
            f.write("-" * 80 + "\n")
            f.write("KEY FINDINGS\n")
            f.write("-" * 80 + "\n\n")
            
            # Identify best UMAP hyperparameters first
            f.write("BEST UMAP HYPERPARAMETERS (by High-Potent EF):\n\n")
            
            # UMAP-Euclidean with features
            df_umap_euc = df_results[
                (df_results['dr_method'] == 'UMAP-Euclidean') &
                (df_results['representation'] == 'features')
            ]
            if not df_umap_euc.empty and 'n_neighbors' in df_umap_euc.columns:
                hyperparam_cols = [c for c in ['n_neighbors', 'min_dist'] if c in df_umap_euc.columns]
                if hyperparam_cols:
                    grouped = df_umap_euc.groupby(hyperparam_cols)['High_EF'].mean()
                    if not grouped.empty:
                        best_params = grouped.idxmax()
                        best_ef = grouped.max()
                        if isinstance(best_params, tuple):
                            nn, md = best_params
                            f.write(f"  UMAP-Euclidean (features):  nn={nn}, min_dist={md} → High-EF={best_ef:.2f}\n")
                        else:
                            f.write(f"  UMAP-Euclidean (features):  nn={best_params} → High-EF={best_ef:.2f}\n")
            
            # UMAP-Jaccard with fingerprints
            df_umap_jac = df_results[
                (df_results['dr_method'] == 'UMAP-Jaccard') &
                (df_results['representation'] == 'fingerprints')
            ]
            if not df_umap_jac.empty and 'n_neighbors' in df_umap_jac.columns:
                hyperparam_cols = [c for c in ['n_neighbors', 'min_dist'] if c in df_umap_jac.columns]
                if hyperparam_cols:
                    grouped = df_umap_jac.groupby(hyperparam_cols)['High_EF'].mean()
                    if not grouped.empty:
                        best_params = grouped.idxmax()
                        best_ef = grouped.max()
                        if isinstance(best_params, tuple):
                            nn, md = best_params
                            f.write(f"  UMAP-Jaccard (fingerprints): nn={nn}, min_dist={md} → High-EF={best_ef:.2f}\n")
                        else:
                            f.write(f"  UMAP-Jaccard (fingerprints): nn={best_params} → High-EF={best_ef:.2f}\n")
            f.write("\n")
            
            # Method-Representation comparison (using best UMAP hyperparameters)
            f.write("1. Method-Representation Performance by Potency Tier:\n")
            f.write("   (Note: UMAP results show BEST hyperparameters only)\n\n")
            
            # Define the configurations to compare
            configs_to_compare = {
                'pca_feat': {'method': 'PCA', 'repr': 'features', 'label': 'PCA-features'},
                'pca_fing': {'method': 'PCA', 'repr': 'fingerprints', 'label': 'PCA-fingerprints'},
                'umap_euc': {'method': 'UMAP-Euclidean', 'repr': 'features', 'label': 'UMAP-Euclidean-features'},
                'umap_jac': {'method': 'UMAP-Jaccard', 'repr': 'fingerprints', 'label': 'UMAP-Jaccard-fingerprints'}
            }
            
            # Get best UMAP configs for filtering
            best_umap_euc_params = None
            if not df_umap_euc.empty and 'n_neighbors' in df_umap_euc.columns:
                hyperparam_cols = [c for c in ['n_neighbors', 'min_dist'] if c in df_umap_euc.columns]
                if hyperparam_cols:
                    grouped = df_umap_euc.groupby(hyperparam_cols)['High_EF'].mean()
                    if not grouped.empty:
                        best_umap_euc_params = grouped.idxmax()
            
            best_umap_jac_params = None
            if not df_umap_jac.empty and 'n_neighbors' in df_umap_jac.columns:
                hyperparam_cols = [c for c in ['n_neighbors', 'min_dist'] if c in df_umap_jac.columns]
                if hyperparam_cols:
                    grouped = df_umap_jac.groupby(hyperparam_cols)['High_EF'].mean()
                    if not grouped.empty:
                        best_umap_jac_params = grouped.idxmax()
            
            for tier in TIER_ORDER:
                ef_col = f'{tier}_EF'
                f.write(f"  {tier} Potency ({POTENCY_TIERS[tier][0]}-{POTENCY_TIERS[tier][1]} nM):\n")
                
                config_means = {}
                for config_key, config_info in configs_to_compare.items():
                    config_data = df_results[
                        (df_results['dr_method'] == config_info['method']) &
                        (df_results['representation'] == config_info['repr'])
                    ]
                    
                    # Filter UMAP by best hyperparameters
                    if config_key == 'umap_euc' and best_umap_euc_params is not None:
                        if isinstance(best_umap_euc_params, tuple):
                            nn, md = best_umap_euc_params
                            config_data = config_data[
                                (config_data['n_neighbors'] == nn) &
                                (config_data['min_dist'] == md)
                            ]
                    elif config_key == 'umap_jac' and best_umap_jac_params is not None:
                        if isinstance(best_umap_jac_params, tuple):
                            nn, md = best_umap_jac_params
                            config_data = config_data[
                                (config_data['n_neighbors'] == nn) &
                                (config_data['min_dist'] == md)
                            ]
                    
                    config_data = config_data[ef_col]
                    
                    if not config_data.empty:
                        mean_ef = config_data.mean()
                        std_ef = config_data.std()
                        config_means[config_key] = mean_ef
                        f.write(f"    {config_info['label']:30s}: {mean_ef:6.2f} ± {std_ef:5.2f}\n")
                
                # Find best config for this tier
                if config_means:
                    best_config = max(config_means, key=config_means.get)
                    best_label = configs_to_compare[best_config]['label']
                    best_ef = config_means[best_config]
                    f.write(f"    → Winner: {best_label} (EF = {best_ef:.2f})\n\n")
                else:
                    f.write("    (No data available)\n\n")
            
            # Dimensionality impact
            f.write("2. Impact of Dimensionality:\n\n")
            if 'dimension' in df_results.columns:
                dimensions = sorted(df_results['dimension'].unique())
                f.write(f"  Dimensions tested: {', '.join(map(str, dimensions))}\n\n")
                
                for tier in TIER_ORDER:
                    ef_col = f'{tier}_EF'
                    f.write(f"  {tier} Potency:\n")
                    
                    best_dim = None
                    best_ef = -np.inf
                    
                    for dim in dimensions:
                        dim_data = df_results[df_results['dimension'] == dim][ef_col]
                        if not dim_data.empty:
                            mean_ef = dim_data.mean()
                            f.write(f"    {dim}D: {mean_ef:6.2f} ± {dim_data.std():5.2f}\n")
                            
                            if mean_ef > best_ef:
                                best_ef = mean_ef
                                best_dim = dim
                    
                    if best_dim:
                        f.write(f"    → Optimal: {best_dim}D (EF = {best_ef:.2f})\n\n")
            
            # Quality vs quantity trade-offs
            f.write("3. Quality vs Quantity Trade-offs:\n\n")
            if 'Overall_EF_mean' in summary_df.columns and 'High_EF_mean' in summary_df.columns:
                # Find config with best High-Potent EF
                best_high_idx = summary_df['High_EF_mean'].idxmax()
                best_high_row = summary_df.loc[best_high_idx]
                
                # Find config with best Overall EF
                best_overall_idx = summary_df['Overall_EF_mean'].idxmax()
                best_overall_row = summary_df.loc[best_overall_idx]
                
                if best_high_idx != best_overall_idx:
                    f.write("⚠️  QUALITY vs QUANTITY TRADE-OFF DETECTED:\n\n")
                    
                    f.write("Best for HIGH-POTENT enrichment:\n")
                    if 'n_neighbors' in best_high_row:
                        f.write(f"  Config: nn={best_high_row['n_neighbors']}, ")
                        f.write(f"md={best_high_row.get('min_dist', 'N/A')}\n")
                    f.write(f"  High-Potent EF: {best_high_row['High_EF_mean']:.2f}\n")
                    f.write(f"  Overall EF: {best_high_row['Overall_EF_mean']:.2f}\n\n")
                    
                    f.write("Best for OVERALL enrichment:\n")
                    if 'n_neighbors' in best_overall_row:
                        f.write(f"  Config: nn={best_overall_row['n_neighbors']}, ")
                        f.write(f"md={best_overall_row.get('min_dist', 'N/A')}\n")
                    f.write(f"  High-Potent EF: {best_overall_row['High_EF_mean']:.2f}\n")
                    f.write(f"  Overall EF: {best_overall_row['Overall_EF_mean']:.2f}\n\n")
                    
                    f.write("RECOMMENDATION: For drug discovery, prioritize the configuration\n")
                    f.write("with best High-Potent enrichment to find therapeutically relevant\n")
                    f.write("compounds, even if overall EF is slightly lower.\n\n")
                else:
                    f.write("✓ Best overall configuration also excels at High-Potent enrichment.\n\n")
        
        f.write("=" * 80 + "\n")
    
    logging.info(f"Report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Potency-Stratified Enrichment Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--workspace_dir', required=True,
                       help='Path to experiment workspace directory')
    parser.add_argument('--output_dir', required=True,
                       help='Output directory for analysis results')
    parser.add_argument('--seed', type=int,
                       help='Analyze only specific seed (optional)')
    parser.add_argument('--summary_only', action='store_true',
                       help='Generate summary only, skip detailed plots')
    parser.add_argument('--n_jobs', type=int, default=None,
                       help='Number of parallel workers (default: use all available CPUs - 1)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.workspace_dir):
        logging.error(f"Workspace directory not found: {args.workspace_dir}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup file logging
    log_file = os.path.join(args.output_dir, 'analysis.log')
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)-8s - %(message)s'))
    logging.getLogger().addHandler(file_handler)
    
    logging.info("=" * 80)
    logging.info("POTENCY-STRATIFIED ENRICHMENT ANALYSIS")
    logging.info("=" * 80)
    logging.info(f"Workspace: {args.workspace_dir}")
    logging.info(f"Output: {args.output_dir}")
    if args.seed:
        logging.info(f"Filtering by seed: {args.seed}")
    if args.n_jobs:
        logging.info(f"Parallel workers: {args.n_jobs}")
    
    # Scan workspace and analyze runs
    df_results = scan_workspace(args.workspace_dir, seed_filter=args.seed, n_jobs=args.n_jobs)
    
    if df_results.empty:
        logging.error("No results found. Check workspace path and data availability.")
        sys.exit(1)
    
    # Save detailed results
    results_file = os.path.join(args.output_dir, 'stratified_enrichment_detailed.csv')
    df_results.to_csv(results_file, index=False)
    logging.info(f"Detailed results saved to {results_file}")
    
    # Create summary table
    summary_df = create_summary_table(df_results)
    if not summary_df.empty:
        summary_file = os.path.join(args.output_dir, 'stratified_enrichment_summary.csv')
        summary_df.to_csv(summary_file, index=False)
        logging.info(f"Summary table saved to {summary_file}")
    
    # Generate plots (unless summary_only)
    if not args.summary_only:
        plot_dir = os.path.join(args.output_dir, 'plots')
        plot_stratified_comparison(df_results, plot_dir)
    
    # Generate report
    generate_report(df_results, summary_df, args.output_dir)
    
    logging.info("=" * 80)
    logging.info("Analysis complete!")
    logging.info(f"Results in: {args.output_dir}")
    logging.info("=" * 80)


if __name__ == '__main__':
    main()
