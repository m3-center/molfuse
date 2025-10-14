#!/usr/bin/env python3
"""
UMMBAS Dataset Analysis: Molecular Count Statistics

This script calculates and reports the number of molecules in different categories:
1. ZINC database compounds (decoys)
2. Known ligands for specific target proteins  
3. Known ligands for proteins with the same molecular function (MF Cloud)

The analysis follows the experimental design used in the UMMBAS screening experiments.
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import argparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_and_merge_chembl_data_for_raw_target_ligands(affinity_csv_path, target_mapping_csv_path, target_uniprot_id):
    """
    Loads ChEMBL affinity data and target mapping to identify compounds associated with a specific UniProt ID.
    This is the same function used in prepare_data.py for experimental consistency.
    """
    logger.info(f"Loading raw ChEMBL data to identify ligands for UniProt ID: {target_uniprot_id}")
    try:
        df_affinity = pd.read_csv(affinity_csv_path, low_memory=False)
        df_target_map = pd.read_csv(target_mapping_csv_path, low_memory=False)
    except FileNotFoundError as e:
        logger.error(f"Required ChEMBL base CSV file not found: {e}. Cannot identify target ligands.")
        raise

    # Merge affinity data with UniProt accession from target mapping
    df_merged = pd.merge(df_affinity, df_target_map,
                         left_on='Target ChEMBL ID', right_on='target_chembl_id',
                         how='left')
    
    if 'accession' not in df_merged.columns:
        logger.error("'accession' column not found after merging ChEMBL affinity and target mapping. Check base CSVs.")
        raise ValueError("'accession' column missing in merged base ChEMBL data.")
        
    df_merged.dropna(subset=['accession'], inplace=True) 

    # Identify ligands for the specific target UniProt ID
    target_ligands_df = df_merged[df_merged['accession'] == target_uniprot_id].copy()
    
    if target_ligands_df.empty:
        logger.warning(f"No ChEMBL ligands found for target UniProt ID: {target_uniprot_id} in raw ChEMBL data.")
    else:
        logger.info(f"Found {len(target_ligands_df)} activity records (may include duplicates by SMILES) for target {target_uniprot_id} from raw ChEMBL.")

    # Select essential columns, especially SMILES and Compound ChEMBL ID
    cols_to_keep = ['Compound ChEMBL ID', 'SMILES', 'Activity Type', 'Standard Value (nM)', 'accession'] 
    
    final_target_ligands_df = pd.DataFrame(columns=cols_to_keep) # Ensure all columns exist
    for col in cols_to_keep:
        if col in target_ligands_df.columns:
            final_target_ligands_df[col] = target_ligands_df[col]
        else:
            final_target_ligands_df[col] = pd.NA
            
    return final_target_ligands_df.drop_duplicates(subset=['Compound ChEMBL ID', 'SMILES'])

def load_config(config_path):
    """Load configuration file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def analyze_zinc_compounds(config):
    """Analyze ZINC compound database using the same data as experiments."""
    logger.info("Analyzing ZINC compound database...")
    
    gs = config['global_settings']
    
    # Use the pre-calculated ZINC features file that experiments actually use
    zinc_features_path = gs.get('precalculated_zinc_features_path')
    
    if not zinc_features_path or not os.path.exists(zinc_features_path):
        logger.error(f"Pre-calculated ZINC features file not found: {zinc_features_path}")
        # Fallback to original ZINC CSV path
        zinc_csv_path = gs.get('zinc_full_csv_path')
        if not zinc_csv_path or not os.path.exists(zinc_csv_path):
            logger.error(f"ZINC CSV file not found: {zinc_csv_path}")
            return None
        else:
            zinc_path_to_use = zinc_csv_path
    else:
        zinc_path_to_use = zinc_features_path
    
    try:
        # Read ZINC data
        df_zinc = pd.read_csv(zinc_path_to_use, low_memory=False)
        
        # Determine ID column (ZINC_ID vs ZINC ID)
        zinc_id_col = None
        if 'ZINC_ID' in df_zinc.columns:
            zinc_id_col = 'ZINC_ID'
        elif 'ZINC ID' in df_zinc.columns:
            zinc_id_col = 'ZINC ID'
        elif 'MOLECULE ID' in df_zinc.columns and df_zinc['MOLECULE ID'].str.startswith('ZINC', na=False).any():
            zinc_id_col = 'MOLECULE ID'
        
        stats = {
            'total_compounds': len(df_zinc),
            'unique_smiles': df_zinc['SMILES'].nunique() if 'SMILES' in df_zinc.columns else 0,
            'data_source': zinc_path_to_use
        }
        
        # Try to get additional metadata if available
        if 'LABEL' in df_zinc.columns:
            stats['availability_labels'] = df_zinc['LABEL'].value_counts().to_dict()
        else:
            stats['availability_labels'] = {}
            
        if 'MANUFACTURER' in df_zinc.columns:
            stats['manufacturers'] = df_zinc['MANUFACTURER'].nunique()
        else:
            stats['manufacturers'] = 0
            
        if 'TRANCHE' in df_zinc.columns:
            stats['tranches'] = df_zinc['TRANCHE'].nunique()
        else:
            stats['tranches'] = 0
        
        logger.info(f"ZINC database: {stats['total_compounds']:,} compounds, {stats['unique_smiles']:,} unique SMILES")
        logger.info(f"Data source: {zinc_path_to_use}")
        return stats
        
    except Exception as e:
        logger.error(f"Error analyzing ZINC data: {e}")
        return None

def analyze_chembl_affinity_data(chembl_csv_path, target_mapping_path):
    """Analyze ChEMBL affinity data for target proteins."""
    logger.info("Analyzing ChEMBL affinity data...")
    
    if not os.path.exists(chembl_csv_path):
        logger.error(f"ChEMBL affinity file not found: {chembl_csv_path}")
        return None
    
    if not os.path.exists(target_mapping_path):
        logger.error(f"Target mapping file not found: {target_mapping_path}")
        return None
    
    try:
        # Read ChEMBL affinity data
        df_affinity = pd.read_csv(chembl_csv_path, low_memory=False)
        df_target_mapping = pd.read_csv(target_mapping_path)
        
        # Merge with target mapping to get UniProt IDs
        df_merged = df_affinity.merge(
            df_target_mapping, 
            left_on='Target ChEMBL ID', 
            right_on='target_chembl_id', 
            how='left'
        )
        
        stats = {
            'total_bioactivity_records': len(df_affinity),
            'unique_compounds': df_affinity['Compound ChEMBL ID'].nunique(),
            'unique_targets': df_affinity['Target ChEMBL ID'].nunique(),
            'activity_types': df_affinity['Activity Type'].value_counts().to_dict(),
            'targets_with_uniprot': df_merged['accession'].notna().sum()
        }
        
        logger.info(f"ChEMBL: {stats['total_bioactivity_records']:,} bioactivity records")
        logger.info(f"ChEMBL: {stats['unique_compounds']:,} unique compounds")
        logger.info(f"ChEMBL: {stats['unique_targets']:,} unique targets")
        
        return stats, df_merged
        
    except Exception as e:
        logger.error(f"Error analyzing ChEMBL data: {e}")
        return None, None

def analyze_target_specific_ligands(config, target_configs, affinity_cutoff_nM=100000):
    """
    Analyze ligands for specific target proteins using the same approach as prepare_data.py.
    This matches the experimental pipeline's method of identifying held-out actives.
    """
    logger.info(f"Analyzing target-specific ligands using experimental approach (cutoff: {affinity_cutoff_nM:,} nM)...")
    
    gs = config['global_settings']
    target_stats = {}
    
    for target_info in target_configs:
        target_id = target_info['id_name']
        uniprot_id = target_info['uniprot_id']
        display_name = target_info['display_name']
        
        logger.info(f"Processing target: {display_name} ({uniprot_id})")
        
        try:
            # Use the same function as prepare_data.py to load target ligands
            df_target_ligands = load_and_merge_chembl_data_for_raw_target_ligands(
                gs['chembl_affinity_full_csv_path'],
                gs['chembl_target_mapping_csv_path'],
                uniprot_id
            )
            
            if df_target_ligands.empty:
                logger.warning(f"No data found for target {uniprot_id}")
                target_stats[target_id] = {
                    'display_name': display_name,
                    'uniprot_id': uniprot_id,
                    'molecular_function': target_info.get('molecular_function_canonical_name', 'Unknown'),
                    'total_records': 0,
                    'records_with_affinity': 0,
                    'unique_compounds_total': 0,
                    'unique_compounds_with_affinity': 0,
                    'active_compounds_cutoff': 0,
                    'activity_range_nM': None
                }
                continue
            
            # Convert affinity values to numeric, handling errors
            df_target_ligands['Standard Value (nM)'] = pd.to_numeric(df_target_ligands['Standard Value (nM)'], errors='coerce')
            
            # Filter out records without valid affinity values
            valid_affinity = df_target_ligands.dropna(subset=['Standard Value (nM)'])
            
            # Apply affinity cutoff (≤ cutoff = active) - same as project_and_analyze.py
            active_compounds = valid_affinity[valid_affinity['Standard Value (nM)'] <= affinity_cutoff_nM]
            
            target_stats[target_id] = {
                'display_name': display_name,
                'uniprot_id': uniprot_id,
                'molecular_function': target_info.get('molecular_function_canonical_name', 'Unknown'),
                'total_records': len(df_target_ligands),
                'records_with_affinity': len(valid_affinity),
                'unique_compounds_total': df_target_ligands['Compound ChEMBL ID'].nunique(),
                'unique_compounds_with_affinity': valid_affinity['Compound ChEMBL ID'].nunique(),
                'active_compounds_cutoff': active_compounds['Compound ChEMBL ID'].nunique(),
                'activity_range_nM': {
                    'min': float(valid_affinity['Standard Value (nM)'].min()) if len(valid_affinity) > 0 else None,
                    'max': float(valid_affinity['Standard Value (nM)'].max()) if len(valid_affinity) > 0 else None,
                    'median': float(valid_affinity['Standard Value (nM)'].median()) if len(valid_affinity) > 0 else None
                }
            }
            
            logger.info(f"  {display_name}: {target_stats[target_id]['active_compounds_cutoff']} active compounds")
            
        except Exception as e:
            logger.error(f"Error processing target {uniprot_id}: {e}")
            target_stats[target_id] = {
                'display_name': display_name,
                'uniprot_id': uniprot_id,
                'molecular_function': target_info.get('molecular_function_canonical_name', 'Unknown'),
                'total_records': 0,
                'records_with_affinity': 0,
                'unique_compounds_total': 0,
                'unique_compounds_with_affinity': 0,
                'active_compounds_cutoff': 0,
                'activity_range_nM': None
            }
    
    return target_stats

def analyze_molecular_function_clouds(config, target_configs, affinity_cutoff_nM=100000):
    """
    Analyze molecular function clouds using the same data loading approach as the experiments.
    This loads the pre-calculated MF datasets and applies the same exclusion logic as prepare_data.py
    """
    logger.info("Analyzing molecular function clouds using experimental data loading approach...")
    
    gs = config['global_settings']
    mf_stats = {}
    
    # Group targets by molecular function 
    mf_groups = {}
    for target_info in target_configs:
        mf_name = target_info['molecular_function_canonical_name']
        if mf_name not in mf_groups:
            mf_groups[mf_name] = []
        mf_groups[mf_name].append(target_info)
    
    for mf_name, targets_in_mf in mf_groups.items():
        logger.info(f"Processing molecular function: {mf_name}")
        
        # Load molecular function keywords
        mf_kw_id = None
        try:
            df_mf_keywords = pd.read_csv(gs['molecular_function_keywords_csv_path'])
            row = df_mf_keywords[df_mf_keywords['Name'].str.lower() == mf_name.lower()]
            if not row.empty:
                mf_kw_id = row.iloc[0]['ID']
            else:
                logger.warning(f"MF keyword ID not found for '{mf_name}'")
        except Exception as e:
            logger.warning(f"Could not load MF keywords: {e}")
        
        mf_cloud_stats = {}
        
        for target_info in targets_in_mf:
            target_uniprot = target_info['uniprot_id']
            target_name = target_info['display_name']
            target_id = target_info['id_name']
            mf_filename_segment = target_info['molecular_function_filename_segment']
            
            # Load the pre-calculated MF dataset (features version for counting)
            if mf_kw_id:
                mf_filename = f"{mf_kw_id}_{mf_filename_segment}_affinity_extracted_features.csv"
                mf_path = os.path.join(gs['precalculated_chembl_mf_features_base_dir'], mf_filename)
                
                try:
                    # Load the pre-calculated MF data
                    df_mf_full = pd.read_csv(mf_path, low_memory=False)
                    logger.info(f"Loaded MF dataset for {target_name}: {mf_path} (shape: {df_mf_full.shape})")
                    
                    # Apply the same exclusion logic as prepare_data.py
                    if 'accession' in df_mf_full.columns:
                        # Exclude compounds for this specific target by UniProt ID
                        df_mf_excluded = df_mf_full[df_mf_full['accession'] != target_uniprot].copy()
                        logger.info(f"Filtered by UniProt ID '{target_uniprot}'. Original: {len(df_mf_full)}, After exclusion: {len(df_mf_excluded)}")
                    else:
                        logger.warning(f"'accession' column not found in {mf_path}. Cannot apply target exclusion.")
                        df_mf_excluded = df_mf_full.copy()
                    
                    # Apply affinity cutoff
                    if 'Standard Value (nM)' in df_mf_excluded.columns:
                        df_mf_excluded['Standard Value (nM)'] = pd.to_numeric(df_mf_excluded['Standard Value (nM)'], errors='coerce')
                        valid_affinity = df_mf_excluded.dropna(subset=['Standard Value (nM)'])
                        active_mf_compounds = valid_affinity[valid_affinity['Standard Value (nM)'] <= affinity_cutoff_nM]
                        
                        unique_active_compounds = active_mf_compounds['Compound ChEMBL ID'].nunique()
                        unique_targets = df_mf_excluded['accession'].nunique() if 'accession' in df_mf_excluded.columns else 0
                    else:
                        logger.warning(f"'Standard Value (nM)' column not found in {mf_path}")
                        unique_active_compounds = 0
                        unique_targets = 0
                    
                    mf_cloud_stats[target_id] = {
                        'target_name': target_name,
                        'excluded_target': target_uniprot,
                        'mf_cloud_total_records': len(df_mf_excluded),
                        'mf_cloud_active_compounds': unique_active_compounds,
                        'mf_cloud_unique_targets': unique_targets,
                        'data_source': mf_path
                    }
                    
                    logger.info(f"  MF Cloud for {target_name}: {unique_active_compounds} active compounds")
                    
                except Exception as e:
                    logger.error(f"Error loading MF dataset {mf_path}: {e}")
                    mf_cloud_stats[target_id] = {
                        'target_name': target_name,
                        'excluded_target': target_uniprot,
                        'mf_cloud_total_records': 0,
                        'mf_cloud_active_compounds': 0,
                        'mf_cloud_unique_targets': 0,
                        'data_source': 'ERROR'
                    }
            else:
                logger.error(f"Cannot find MF keyword ID for '{mf_name}', skipping {target_name}")
                mf_cloud_stats[target_id] = {
                    'target_name': target_name,
                    'excluded_target': target_uniprot,
                    'mf_cloud_total_records': 0,
                    'mf_cloud_active_compounds': 0,
                    'mf_cloud_unique_targets': 0,
                    'data_source': 'NO_KW_ID'
                }
        
        mf_stats[mf_name] = {
            'targets_in_experiment': [t['display_name'] for t in targets_in_mf],
            'mf_keyword_id': mf_kw_id if mf_kw_id else 'Unknown',
            'clouds_per_target': mf_cloud_stats
        }
    
    return mf_stats

def generate_summary_report(zinc_stats, chembl_stats, target_stats, mf_stats, output_file):
    """Generate a comprehensive summary report."""
    logger.info(f"Generating summary report: {output_file}")
    
    with open(output_file, 'w') as f:
        f.write("# UMMBAS Dataset Analysis Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # ZINC Database Section
        f.write("## 1. ZINC Database (Decoy Compounds)\n\n")
        if zinc_stats:
            f.write(f"- **Total compounds**: {zinc_stats['total_compounds']:,}\n")
            f.write(f"- **Unique SMILES**: {zinc_stats['unique_smiles']:,}\n")
            f.write(f"- **Data source**: {zinc_stats['data_source']}\n")
            if zinc_stats['manufacturers'] > 0:
                f.write(f"- **Number of manufacturers**: {zinc_stats['manufacturers']:,}\n")
            if zinc_stats['tranches'] > 0:
                f.write(f"- **Number of tranches**: {zinc_stats['tranches']:,}\n")
            if zinc_stats['availability_labels']:
                f.write("- **Availability distribution**:\n")
                for label, count in zinc_stats['availability_labels'].items():
                    f.write(f"  - {label}: {count:,}\n")
        else:
            f.write("- No ZINC data available\n")
        f.write("\n")
        
        # ChEMBL Database Section
        f.write("## 2. ChEMBL Database Overview\n\n")
        if chembl_stats:
            f.write(f"- **Total bioactivity records**: {chembl_stats['total_bioactivity_records']:,}\n")
            f.write(f"- **Unique compounds**: {chembl_stats['unique_compounds']:,}\n")
            f.write(f"- **Unique targets**: {chembl_stats['unique_targets']:,}\n")
            f.write(f"- **Targets with UniProt mapping**: {chembl_stats['targets_with_uniprot']:,}\n")
            f.write("- **Activity types**:\n")
            for activity_type, count in list(chembl_stats['activity_types'].items())[:10]:  # Top 10
                f.write(f"  - {activity_type}: {count:,}\n")
        else:
            f.write("- ChEMBL data analysis skipped in favor of pre-calculated molecular function datasets\n")
            f.write("- This analysis uses the same data loading approach as the actual experiments\n")
            f.write("- Target-specific and MF cloud compounds are loaded from pre-calculated feature files\n")
        f.write("\n")
        
        # Target-Specific Ligands Section
        f.write("## 3. Target-Specific Known Ligands (Held-out Actives)\n\n")
        f.write("These are the compounds used as 'held-out actives' in the leave-one-target-out experiments.\n\n")
        
        for target_id, stats in target_stats.items():
            f.write(f"### {stats['display_name']} ({stats['uniprot_id']})\n")
            f.write(f"- **Molecular Function**: {stats['molecular_function']}\n")
            f.write(f"- **Total bioactivity records**: {stats['total_records']:,}\n")
            f.write(f"- **Records with affinity values**: {stats['records_with_affinity']:,}\n")
            f.write(f"- **Unique compounds (all)**: {stats['unique_compounds_total']:,}\n")
            f.write(f"- **Unique compounds (with affinity)**: {stats['unique_compounds_with_affinity']:,}\n")
            f.write(f"- **Active compounds (≤100,000 nM)**: {stats['active_compounds_cutoff']:,}\n")
            if stats['activity_range_nM'] and stats['activity_range_nM']['min'] is not None:
                f.write(f"- **Affinity range**: {stats['activity_range_nM']['min']:.1f} - {stats['activity_range_nM']['max']:.1f} nM\n")
                f.write(f"- **Median affinity**: {stats['activity_range_nM']['median']:.1f} nM\n")
            f.write("\n")
        
        # Molecular Function Clouds Section
        f.write("## 4. Molecular Function Clouds (MF Cloud)\n\n")
        f.write("These are the compounds used to create the 'similarity space' for each target.\n")
        f.write("Each MF Cloud contains compounds that bind to proteins with the same molecular function,\n")
        f.write("but excludes compounds for the specific target being tested (leave-one-target-out).\n\n")
        
        for mf_name, mf_data in mf_stats.items():
            f.write(f"### {mf_name}\n")
            f.write(f"- **Targets in experiment**: {', '.join(mf_data['targets_in_experiment'])}\n")
            f.write(f"- **MF Keyword ID**: {mf_data['mf_keyword_id']}\n")
            f.write("\n**MF Clouds per target** (excluding target-specific compounds):\n")
            
            for target_id, cloud_stats in mf_data['clouds_per_target'].items():
                f.write(f"- **{cloud_stats['target_name']}**:\n")
                f.write(f"  - Excluded target: {cloud_stats['excluded_target']}\n")
                f.write(f"  - MF Cloud active compounds: {cloud_stats['mf_cloud_active_compounds']:,}\n")
                f.write(f"  - Unique targets in cloud: {cloud_stats['mf_cloud_unique_targets']:,}\n")
                if 'data_source' in cloud_stats:
                    f.write(f"  - Data source: {cloud_stats['data_source']}\n")
            f.write("\n")
        
        # Summary Statistics
        f.write("## 5. Summary Statistics\n\n")
        total_zinc = zinc_stats['total_compounds'] if zinc_stats else 0
        total_held_out = sum([stats['active_compounds_cutoff'] for stats in target_stats.values()])
        
        f.write(f"- **ZINC decoy compounds**: {total_zinc:,}\n")
        f.write(f"- **Total held-out active compounds** (across all targets): {total_held_out:,}\n")
        
        # Calculate MF Cloud sizes
        f.write("- **MF Cloud sizes** (active compounds, excluding target-specific):\n")
        for mf_name, mf_data in mf_stats.items():
            for target_id, cloud_stats in mf_data['clouds_per_target'].items():
                f.write(f"  - {cloud_stats['target_name']}: {cloud_stats['mf_cloud_active_compounds']:,}\n")
        
        f.write("\n")
        f.write("## 6. Experimental Context\n\n")
        f.write("In the UMMBAS experiments:\n")
        f.write("1. **ZINC compounds** serve as decoys (negative examples)\n")
        f.write("2. **Held-out actives** are the compounds we try to identify (positive examples)\n")
        f.write("3. **MF Cloud** compounds create the similarity space for ranking\n")
        f.write("4. The goal is to rank held-out actives higher than ZINC decoys\n")
        f.write("5. Performance is measured by how well the method separates actives from decoys\n")
        f.write("\n")
        f.write("**Note**: Only targets with `processing_mode='full_analysis'` are included in this analysis.\n")
        f.write("Targets with `processing_mode='similarity_space_only'` (e.g., Actin proteins) were used only for \n")
        f.write("similarity space generation and are excluded from these counts.\n")
        
        # Add LaTeX dataset description
        f.write("\n")
        f.write("## 7. LaTeX Dataset Description for Publication\n\n")
        f.write("```latex\n")
        
        # Calculate totals for LaTeX
        total_zinc = zinc_stats['total_compounds'] if zinc_stats else 0
        total_held_out = sum([stats['active_compounds_cutoff'] for stats in target_stats.values()])
        
        # Get MF cloud info
        mf_cloud_info = []
        for mf_name, mf_data in mf_stats.items():
            for target_id, cloud_stats in mf_data['clouds_per_target'].items():
                mf_cloud_info.append(f"{cloud_stats['target_name']}: {cloud_stats['mf_cloud_active_compounds']:,}")
        
        f.write("\\subsection{Dataset}\n")
        f.write("We employed a comprehensive molecular dataset comprising three protein targets \n")
        f.write("for virtual screening evaluation. The dataset includes:\\\\[0.5em]\n\n")
        
        f.write("\\textbf{Target Proteins:} Three protein targets were selected for leave-one-target-out \n")
        f.write("virtual screening experiments: (1) Tyrosine-protein kinase ABL1 (UniProt: P00519), \n") 
        f.write("(2) Isocitrate dehydrogenase NADP cytoplasmic (UniProt: O75874), and \n")
        f.write("(3) Pyruvate kinase M2 (UniProt: P14618). These targets represent diverse molecular \n")
        f.write("functions including protein kinase inhibition and oxidoreductase activity.\\\\[0.5em]\n\n")
        
        f.write(f"\\textbf{{Held-out Active Compounds:}} A total of {total_held_out:,} bioactive compounds \n")
        f.write("with experimentally validated affinity (≤100,000 nM) were extracted from ChEMBL 35 \n")
        f.write("database. The distribution per target was: ")
        target_counts = []
        for target_id, stats in target_stats.items():
            target_counts.append(f"{stats['display_name']} ({stats['active_compounds_cutoff']:,})")
        f.write(", ".join(target_counts))
        f.write(".\\\\[0.5em]\n\n")
        
        f.write(f"\\textbf{{Decoy Compounds:}} {total_zinc:,} purchasable small molecules from the ZINC \n")
        f.write("database served as decoy compounds (presumed inactive) for virtual screening \n")
        f.write("evaluation.\\\\[0.5em]\n\n")
        
        f.write("\\textbf{Molecular Function Clouds:} For each target, similarity spaces were constructed \n")
        f.write("using compounds that bind to proteins sharing the same molecular function, excluding \n")
        f.write("the target-specific compounds (leave-one-target-out approach). ")
        f.write("The molecular function cloud sizes were: ")
        f.write(", ".join(mf_cloud_info))
        f.write(". These clouds provide the molecular context for dimensionality reduction and \n")
        f.write("similarity-based virtual screening.\\\\[0.5em]\n\n")
        
        f.write("All molecular representations were computed using RDKit, including 39 physicochemical \n")
        f.write("descriptors and 1024-bit Extended Connectivity Fingerprints (ECFP4). The dataset \n")
        f.write("design ensures realistic virtual screening conditions where active compounds must be \n")
        f.write("distinguished from a large background of presumed inactive molecules.\n")
        
        f.write("```\n")

def main():
    parser = argparse.ArgumentParser(description="Analyze UMMBAS dataset molecular counts")
    parser.add_argument("--config", default="experiment_config.json", 
                       help="Path to experiment configuration file")
    parser.add_argument("--output", default="ummbas_dataset_analysis.md",
                       help="Output file for the analysis report")
    parser.add_argument("--affinity_cutoff", type=float, default=100000,
                       help="Affinity cutoff in nM for defining active compounds")
    
    args = parser.parse_args()
    
    logger.info("Starting UMMBAS dataset analysis...")
    
    # Load configuration
    try:
        config = load_config(args.config)
        gs = config['global_settings']
        targets = config['targets']
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return
    
    # Filter targets to only include those used for full analysis (not similarity_space_only)
    analysis_targets = []
    for target_info in targets:
        processing_mode = target_info.get('processing_mode', 'full_analysis')
        if processing_mode == 'full_analysis':
            analysis_targets.append(target_info)
        else:
            logger.info(f"Skipping target {target_info['display_name']} (processing_mode: {processing_mode})")
    
    if not analysis_targets:
        logger.error("No targets found for full analysis. Check target configurations.")
        return
    
    logger.info(f"Found {len(analysis_targets)} targets for full analysis: {[t['display_name'] for t in analysis_targets]}")
    
    # Analyze ZINC database using experimental data paths
    zinc_stats = analyze_zinc_compounds(config)
    
    # Analyze target-specific ligands using experimental approach  
    target_stats = analyze_target_specific_ligands(config, analysis_targets, args.affinity_cutoff)
    
    # Analyze molecular function clouds using experimental data loading
    mf_stats = analyze_molecular_function_clouds(config, analysis_targets, args.affinity_cutoff)
    
    # Generate summary report
    generate_summary_report(
        zinc_stats, None, target_stats, mf_stats, args.output
    )
    
    logger.info(f"Analysis complete! Report saved to: {args.output}")
    
    # Print quick summary to console
    print("\n" + "="*60)
    print("UMMBAS DATASET SUMMARY")
    print("="*60)
    if zinc_stats:
        print(f"ZINC decoy compounds: {zinc_stats['total_compounds']:,}")
    
    print("\nTarget-specific active compounds (≤100,000 nM):")
    for target_id, stats in target_stats.items():
        print(f"  {stats['display_name']}: {stats['active_compounds_cutoff']:,}")
    
    print("\nMF Cloud active compounds (excluding target-specific):")
    for mf_name, mf_data in mf_stats.items():
        print(f"  {mf_name}:")
        for target_id, cloud_stats in mf_data['clouds_per_target'].items():
            print(f"    {cloud_stats['target_name']}: {cloud_stats['mf_cloud_active_compounds']:,}")
    
    print("="*60)

if __name__ == "__main__":
    main()