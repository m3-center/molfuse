import pandas as pd
import os
import argparse
import logging
import json
import sqlite3 # For direct ChEMBL DB query if needed for robustness

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_merge_chembl_data(db_path, affinity_csv_path, target_mapping_csv_path, target_uniprot_id):
    """
    Loads ChEMBL affinity data and target mapping.
    Prioritizes pre-extracted CSVs but can fall back to DB query if needed (simplified for now).
    Identifies SMILES for the specific target_uniprot_id.
    Filters out data points related to the target_uniprot_id for the excluded ChEMBL MF set.
    """
    logging.info("Loading ChEMBL affinity and target mapping data...")
    try:
        df_affinity = pd.read_csv(affinity_csv_path, low_memory=False)
        df_target_map = pd.read_csv(target_mapping_csv_path, low_memory=False)
    except FileNotFoundError as e:
        logging.error(f"Required ChEMBL CSV file not found: {e}. Cannot proceed with data preparation.")
        raise

    # Ensure standard column names from your collect_affinity_forall_functions.py
    # 'Compound ChEMBL ID', 'SMILES', 'Target ChEMBL ID', 'Target Name', 'Activity Type', 'Standard Value (nM)'
    # df_target_map should have 'target_chembl_id', 'accession'

    # Merge affinity data with UniProt accession from target mapping
    # Assuming 'Target ChEMBL ID' in df_affinity and 'target_chembl_id' in df_target_map
    df_merged = pd.merge(df_affinity, df_target_map,
                         left_on='Target ChEMBL ID', right_on='target_chembl_id',
                         how='left')
    
    if 'accession' not in df_merged.columns:
        logging.error("'accession' column not found after merging ChEMBL affinity and target mapping. Check CSVs.")
        raise ValueError("'accession' column missing.")
        
    df_merged.dropna(subset=['accession'], inplace=True) # Only keep records with mapped UniProt IDs

    # Identify ligands for the specific target UniProt ID (these will be projected)
    target_ligands_df = df_merged[df_merged['accession'] == target_uniprot_id].copy()
    
    # For the "excluded" ChEMBL MF set, we need all data *not* related to this specific target_uniprot_id
    # but still within the broader molecular function (this filtering happens later based on mf_affinity_path_resolved)
    chembl_all_others_df = df_merged[df_merged['accession'] != target_uniprot_id].copy()

    if target_ligands_df.empty:
        logging.warning(f"No ChEMBL ligands found for target UniProt ID: {target_uniprot_id}.")
    else:
        logging.info(f"Found {len(target_ligands_df['SMILES'].unique())} unique SMILES for target {target_uniprot_id}.")

    # Select and rename columns for consistency
    # For target_ligands_df, we mainly need SMILES, but keeping other info can be useful.
    # We need consistent column names if these DFs are fed into feature calculation.
    cols_to_keep = ['Compound ChEMBL ID', 'SMILES', 'Target ChEMBL ID', 'Target Name', 
                    'Activity Type', 'Standard Value (nM)', 'accession']
    
    target_ligands_df = target_ligands_df[cols_to_keep].drop_duplicates()
    chembl_all_others_df = chembl_all_others_df[cols_to_keep].drop_duplicates()
    
    return target_ligands_df, chembl_all_others_df


def main():
    parser = argparse.ArgumentParser(description="Prepare excluded datasets for similarity experiments.")
    parser.add_argument("--config_path", required=True, help="Path to the experiment_config.json file.")
    parser.add_argument("--target_id_name", required=True, help="Unique ID name for the target (e.g., PyruvateKinaseM2_P14618).")
    parser.add_argument("--mf_affinity_path_resolved", required=True, help="Full path to the specific ChEMBL Molecular Function affinity CSV file for this target's MF.")
    parser.add_argument("--output_dir", required=True, help="Directory to save the prepared CSV files.")
    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        config = json.load(f)

    gs = config['global_settings']
    
    # Find the specific target's info from config
    current_target_info = next((t for t in config['targets'] if t['id_name'] == args.target_id_name), None)
    if not current_target_info:
        logging.error(f"Target ID '{args.target_id_name}' not found in config file.")
        return
    
    target_uniprot_id = current_target_info['uniprot_id']
    
    # --- 1. Get RAW data for target ligands (to be featurized separately) ---
    # This uses the original ChEMBL affinity CSVs to find the specific target's compounds
    try:
        raw_target_ligands_df, _ = load_and_merge_chembl_data( # This is the function from previous prepare_data.py
            gs['chembl_db_path'],
            gs['chembl_affinity_full_csv_path'],
            gs['chembl_target_mapping_csv_path'],
            target_uniprot_id
        )
    except Exception as e:
        logging.error(f"Failed to load raw ChEMBL data to identify target ligands: {e}")
        return

    target_ligands_for_feature_calc_path = os.path.join(args.output_dir, f"{args.target_id_name}_target_ligands_for_feature_calc_raw.csv")
    if not raw_target_ligands_df.empty:
        # Save only essential columns like SMILES, Compound ChEMBL ID for feature calculation input
        raw_target_ligands_df[['SMILES', 'Compound ChEMBL ID']].to_csv(target_ligands_for_feature_calc_path, index=False)
        logging.info(f"Saved RAW target ligands (for feature calc) to: {target_ligands_for_feature_calc_path}")
    else:
        pd.DataFrame(columns=['SMILES', 'Compound ChEMBL ID']).to_csv(target_ligands_for_feature_calc_path, index=False)
        logging.warning(f"No RAW target ligands found for {target_uniprot_id}.")

    target_smiles_to_exclude = set(raw_target_ligands_df['SMILES'].dropna().unique()) if not raw_target_ligands_df.empty else set()

    # --- 2. Filter PRE-CALCULATED ChEMBL MF files ---
    for repr_type in ["features", "fingerprints"]: # Assuming config['representations']
        precalc_chembl_mf_base_dir = gs[f'precalculated_chembl_mf_{repr_type}_base_dir']
        
        # Construct pre-calculated ChEMBL MF filename
        mf_kw_id = get_mf_keyword_id_from_config(config, current_target_info['molecular_function_canonical_name']) # Helper needed
        if not mf_kw_id: # Error logged in helper
            continue 
        mf_filename_segment = current_target_info['molecular_function_filename_segment']
        
        # Example: KW-0418_Kinase_affinity_extracted_features.csv
        precalc_mf_filename = f"{mf_kw_id}_{mf_filename_segment}_affinity_extracted_{'features' if repr_type == 'features' else 'fingerprints_ECFP4'}.csv"
        precalc_mf_full_path = os.path.join(precalc_chembl_mf_base_dir, precalc_mf_filename)

        output_chembl_mf_excluded_path = os.path.join(args.output_dir, f"{args.target_id_name}_chembl_mf_excluded_{repr_type}.csv")

        if os.path.exists(precalc_mf_full_path):
            try:
                df_precalc_mf = pd.read_csv(precalc_mf_full_path, low_memory=False)
                # Filter out rows belonging to the target_uniprot_id
                # This assumes the pre-calculated files *might* still contain 'Compound ChEMBL ID'
                # that can be mapped back to the target_uniprot_id, or directly have 'accession'.
                # If 'accession' is not in pre-calc files, mapping is needed.
                
                # Simplest case: if 'Compound ChEMBL ID' is in precalc and we have target's ChEMBL IDs
                target_chembl_ids = set(raw_target_ligands_df['Compound ChEMBL ID'].dropna().unique()) if not raw_target_ligands_df.empty else set()

                if 'Compound ChEMBL ID' in df_precalc_mf.columns and target_chembl_ids:
                    # This is the key filtering step for pre-calculated ChEMBL MF data
                    df_filtered_precalc_mf = df_precalc_mf[~df_precalc_mf['Compound ChEMBL ID'].isin(target_chembl_ids)].copy()
                    df_filtered_precalc_mf.to_csv(output_chembl_mf_excluded_path, index=False)
                    logging.info(f"Saved filtered pre-calculated ChEMBL MF {repr_type} to: {output_chembl_mf_excluded_path}")
                elif 'accession' in df_precalc_mf.columns : # If accession is directly available
                     df_filtered_precalc_mf = df_precalc_mf[df_precalc_mf['accession'] != target_uniprot_id].copy()
                     df_filtered_precalc_mf.to_csv(output_chembl_mf_excluded_path, index=False)
                     logging.info(f"Saved filtered pre-calculated ChEMBL MF {repr_type} (by accession) to: {output_chembl_mf_excluded_path}")
                else:
                    logging.warning(f"Cannot filter pre-calculated ChEMBL MF file {precalc_mf_full_path} by Compound ChEMBL ID or accession. Copying as is.")
                    df_precalc_mf.to_csv(output_chembl_mf_excluded_path, index=False) # Copy if no filter possible
            except Exception as e:
                logging.error(f"Error processing pre-calculated ChEMBL MF file {precalc_mf_full_path}: {e}")
                pd.DataFrame().to_csv(output_chembl_mf_excluded_path, index=False) # Save empty on error
        else:
            logging.error(f"Pre-calculated ChEMBL MF file not found: {precalc_mf_full_path}")
            pd.DataFrame().to_csv(output_chembl_mf_excluded_path, index=False) # Save empty


    # --- 3. Filter PRE-CALCULATED ZINC files ---
    for repr_type in ["features", "fingerprints"]:
        precalc_zinc_path = gs[f'precalculated_zinc_{repr_type}_path']
        output_zinc_excluded_path = os.path.join(args.output_dir, f"{args.target_id_name}_zinc_excluded_{repr_type}.csv")

        if os.path.exists(precalc_zinc_path):
            try:
                df_precalc_zinc = pd.read_csv(precalc_zinc_path, low_memory=False)
                if 'SMILES' in df_precalc_zinc.columns and target_smiles_to_exclude:
                    df_filtered_precalc_zinc = df_precalc_zinc[~df_precalc_zinc['SMILES'].isin(target_smiles_to_exclude)].copy()
                    df_filtered_precalc_zinc.to_csv(output_zinc_excluded_path, index=False)
                    logging.info(f"Saved filtered pre-calculated ZINC {repr_type} to: {output_zinc_excluded_path}")
                else: # No SMILES to exclude or no SMILES column in ZINC precalc
                    logging.info(f"Copying pre-calculated ZINC {repr_type} as is (no SMILES for exclusion or missing SMILES col).")
                    df_precalc_zinc.to_csv(output_zinc_excluded_path, index=False)
            except Exception as e:
                logging.error(f"Error processing pre-calculated ZINC file {precalc_zinc_path}: {e}")
                pd.DataFrame().to_csv(output_zinc_excluded_path, index=False)
        else:
            logging.error(f"Pre-calculated ZINC file not found: {precalc_zinc_path}")
            pd.DataFrame().to_csv(output_zinc_excluded_path, index=False)

    logging.info(f"Data preparation for {args.target_id_name} (using pre-calculated files) completed.")

# Helper function for prepare_data.py to get KW-ID (needs access to config or keywords_df)
def get_mf_keyword_id_from_config(config, canonical_mf_name):
    # This should ideally load the keywords_df once and pass it around, or be part of a class.
    # For a standalone script, it might reload it.
    gs = config['global_settings']
    try:
        mf_keywords_df = pd.read_csv(gs['molecular_function_keywords_csv_path'])
        row = mf_keywords_df[mf_keywords_df['Name'].str.lower() == canonical_mf_name.lower()]
        if not row.empty:
            return row.iloc[0]['ID']
        logging.error(f"KW-ID not found for MF: {canonical_mf_name}")
    except Exception as e:
        logging.error(f"Error getting KW-ID for {canonical_mf_name}: {e}")
    return None

# (Need to define load_and_merge_chembl_data as in the previous version of prepare_data.py
# for step 1 to get raw_target_ligands_df for their SMILES and for separate feature calculation)
# ... [Insert load_and_merge_chembl_data function here] ...
# This function needs to be defined as it was in the previous version of prepare_data.py
# to correctly extract target_ligands_df from the base ChEMBL CSVs.
def load_and_merge_chembl_data(db_path, affinity_csv_path, target_mapping_csv_path, target_uniprot_id):
    logging.info("Loading ChEMBL affinity and target mapping data for raw target ligand identification...")
    try:
        df_affinity = pd.read_csv(affinity_csv_path, low_memory=False)
        df_target_map = pd.read_csv(target_mapping_csv_path, low_memory=False)
    except FileNotFoundError as e:
        logging.error(f"Required ChEMBL CSV file not found: {e}.")
        raise

    df_merged = pd.merge(df_affinity, df_target_map,
                         left_on='Target ChEMBL ID', right_on='target_chembl_id',
                         how='left')
    
    if 'accession' not in df_merged.columns:
        logging.error("'accession' column not found after merging. Check CSVs.")
        raise ValueError("'accession' column missing.")
        
    df_merged.dropna(subset=['accession'], inplace=True)
    target_ligands_df = df_merged[df_merged['accession'] == target_uniprot_id].copy()
    
    # No need for chembl_all_others_df from this function anymore in this revised workflow
    # We only need target_ligands_df to get their SMILES and ChEMBL IDs.
    
    if target_ligands_df.empty:
        logging.warning(f"No ChEMBL ligands found for target UniProt ID: {target_uniprot_id} in raw ChEMBL data.")
    
    cols_to_keep = ['Compound ChEMBL ID', 'SMILES', 'Target ChEMBL ID', 'Target Name', 
                    'Activity Type', 'Standard Value (nM)', 'accession']
    # Ensure all cols_to_keep are present, add with NA if not.
    for col in cols_to_keep:
        if col not in target_ligands_df.columns:
            target_ligands_df[col] = pd.NA
            
    return target_ligands_df[cols_to_keep].drop_duplicates(), pd.DataFrame() # Return empty df for the second output

if __name__ == "__main__":
    main()