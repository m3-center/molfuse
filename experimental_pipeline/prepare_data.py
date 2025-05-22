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

    logging.info(f"Preparing data for target: {current_target_info['display_name']} (UniProt: {target_uniprot_id})")
    os.makedirs(args.output_dir, exist_ok=True)

    # --- 1. Prepare ChEMBL datasets ---
    try:
        target_ligands_df, _ = load_and_merge_chembl_data(
            gs['chembl_db_path'], # Not directly used if CSVs exist, but passed for future potential
            gs['chembl_affinity_full_csv_path'],
            gs['chembl_target_mapping_csv_path'],
            target_uniprot_id
        )
    except Exception as e:
        logging.error(f"Failed to load and process ChEMBL base data: {e}")
        return

    # Save the target ligands that will be projected
    target_ligands_for_projection_path = os.path.join(args.output_dir, f"{args.target_id_name}_target_ligands_for_projection.csv")
    if not target_ligands_df.empty:
        target_ligands_df.to_csv(target_ligands_for_projection_path, index=False)
        logging.info(f"Saved target ligands for projection to: {target_ligands_for_projection_path} ({len(target_ligands_df)} records)")
    else:
        # Create empty file so downstream processes don't fail on file not found, they should handle empty DFs
        pd.DataFrame().to_csv(target_ligands_for_projection_path, index=False)
        logging.warning(f"No target ligands for projection for {args.target_id_name}. Empty file created: {target_ligands_for_projection_path}")


    # Create the "ChEMBL Molecular Function Excluded" dataset
    # This dataset should contain compounds active for the molecular function,
    # BUT NOT active against the specific target_uniprot_id.
    # The mf_affinity_path_resolved already contains compounds for the specific MF.
    # We need to filter out any remaining entries for the target_uniprot_id from this file.
    try:
        df_chembl_mf_specific = pd.read_csv(args.mf_affinity_path_resolved, low_memory=False)
    except FileNotFoundError:
        logging.error(f"Molecular function specific affinity file not found: {args.mf_affinity_path_resolved}")
        return
        
    # The df_chembl_mf_specific might or might not have an 'accession' column if it was generated
    # by collect_affinity_forall_functions.py before the merge step was fully integrated there.
    # For robustness, if 'accession' is not present, merge it.
    if 'accession' not in df_chembl_mf_specific.columns:
        logging.info(f"Merging 'accession' into {args.mf_affinity_path_resolved} as it's missing.")
        try:
            df_target_map = pd.read_csv(gs['chembl_target_mapping_csv_path'], low_memory=False)
            df_chembl_mf_specific = pd.merge(df_chembl_mf_specific, df_target_map,
                                     left_on='Target ChEMBL ID', right_on='target_chembl_id',
                                     how='left')
            if 'accession' not in df_chembl_mf_specific.columns: # Check again
                 logging.error(f"Still no 'accession' column after attempting merge for {args.mf_affinity_path_resolved}.")
                 return
            df_chembl_mf_specific.dropna(subset=['accession'], inplace=True)
        except Exception as e:
            logging.error(f"Error merging accession into MF specific file: {e}")
            return


    chembl_mf_excluded_df = df_chembl_mf_specific[df_chembl_mf_specific['accession'] != target_uniprot_id].copy()
    
    # Standardize columns for the output (matching what feature calculation might expect)
    # These are ChEMBL compounds, so they should have these columns.
    required_chembl_cols = ['Compound ChEMBL ID', 'SMILES', 'Target ChEMBL ID', 
                            'Target Name', 'Activity Type', 'Standard Value (nM)', 'accession']
    
    # Ensure all required columns are present, add NA if not (though they should be)
    for col in required_chembl_cols:
        if col not in chembl_mf_excluded_df.columns:
            chembl_mf_excluded_df[col] = pd.NA # Or handle as error depending on expectation

    # Filter to only necessary columns and drop duplicates again after filtering
    chembl_mf_excluded_df = chembl_mf_excluded_df[required_chembl_cols].drop_duplicates()
    
    chembl_mf_excluded_path = os.path.join(args.output_dir, f"{args.target_id_name}_chembl_mf_excluded.csv")
    chembl_mf_excluded_df.to_csv(chembl_mf_excluded_path, index=False)
    logging.info(f"Saved ChEMBL MF (excluded) data to: {chembl_mf_excluded_path} ({len(chembl_mf_excluded_df)} records)")

    # --- 2. Prepare ZINC dataset (exclude target ligands by SMILES) ---
    if not target_ligands_df.empty and 'SMILES' in target_ligands_df.columns:
        target_smiles_to_exclude = set(target_ligands_df['SMILES'].dropna().unique())
        logging.info(f"Excluding {len(target_smiles_to_exclude)} unique SMILES from ZINC dataset.")
        
        try:
            df_zinc_full = pd.read_csv(gs['zinc_full_csv_path'], low_memory=False)
        except FileNotFoundError:
            logging.error(f"Full ZINC dataset not found at: {gs['zinc_full_csv_path']}. Cannot prepare excluded ZINC.")
            return

        # Assuming ZINC CSV has 'SMILES' and 'ZINC_ID' and other columns from collect_ZINC.py
        # ('ZINC_ID', 'SMILES', 'LABEL', 'MANUFACTURER', 'TRANCHE')
        if 'SMILES' in df_zinc_full.columns:
            zinc_excluded_df = df_zinc_full[~df_zinc_full['SMILES'].isin(target_smiles_to_exclude)].copy()
            
            zinc_excluded_path = os.path.join(args.output_dir, f"{args.target_id_name}_zinc_excluded.csv")
            zinc_excluded_df.to_csv(zinc_excluded_path, index=False)
            logging.info(f"Saved ZINC (excluded) data to: {zinc_excluded_path} ({len(zinc_excluded_df)} records)")
        else:
            logging.error("SMILES column not found in ZINC dataset. Cannot exclude target ligands.")
    else:
        logging.warning("No target SMILES to exclude from ZINC or target_ligands_df is empty. Copying full ZINC dataset.")
        try:
            # If no SMILES to exclude, just copy the full ZINC file with the "excluded" name for consistency
            full_zinc_path = gs['zinc_full_csv_path']
            zinc_excluded_path = os.path.join(args.output_dir, f"{args.target_id_name}_zinc_excluded.csv")
            if os.path.exists(full_zinc_path):
                # For large files, consider a symlink or just noting that no exclusion happened
                # For now, copy for simplicity of downstream file existence checks
                df_zinc_full = pd.read_csv(full_zinc_path, low_memory=False)
                df_zinc_full.to_csv(zinc_excluded_path, index=False)
                logging.info(f"Copied full ZINC dataset to {zinc_excluded_path} as no exclusions were made.")
            else:
                logging.error(f"Full ZINC dataset not found at: {full_zinc_path} and no exclusions to make.")
        except Exception as e:
            logging.error(f"Error handling ZINC data when no exclusions: {e}")


    logging.info(f"Data preparation for {args.target_id_name} completed.")

if __name__ == "__main__":
    main()