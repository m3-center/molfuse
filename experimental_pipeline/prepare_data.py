import pandas as pd
import os
import argparse
import logging
import json

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("prepare_data.log"), logging.StreamHandler()])

def load_and_merge_chembl_data_for_raw_target_ligands(affinity_csv_path, target_mapping_csv_path, target_uniprot_id):
    """
    Loads ChEMBL affinity data and target mapping to identify compounds associated with a specific UniProt ID.
    This is used to get the "ground truth" target ligands that will be projected.
    """
    logging.info(f"Loading raw ChEMBL data to identify ligands for UniProt ID: {target_uniprot_id}")
    try:
        df_affinity = pd.read_csv(affinity_csv_path, low_memory=False)
        df_target_map = pd.read_csv(target_mapping_csv_path, low_memory=False)
    except FileNotFoundError as e:
        logging.error(f"Required ChEMBL base CSV file not found: {e}. Cannot identify target ligands.")
        raise

    # Merge affinity data with UniProt accession from target mapping
    df_merged = pd.merge(df_affinity, df_target_map,
                         left_on='Target ChEMBL ID', right_on='target_chembl_id',
                         how='left')
    
    if 'accession' not in df_merged.columns:
        logging.error("'accession' column not found after merging ChEMBL affinity and target mapping. Check base CSVs.")
        raise ValueError("'accession' column missing in merged base ChEMBL data.")
        
    df_merged.dropna(subset=['accession'], inplace=True) 

    # Identify ligands for the specific target UniProt ID
    target_ligands_df = df_merged[df_merged['accession'] == target_uniprot_id].copy()
    
    if target_ligands_df.empty:
        logging.warning(f"No ChEMBL ligands found for target UniProt ID: {target_uniprot_id} in raw ChEMBL data.")
    else:
        logging.info(f"Found {len(target_ligands_df)} activity records (may include duplicates by SMILES) for target {target_uniprot_id} from raw ChEMBL.")

    # Select essential columns, especially SMILES and Compound ChEMBL ID
    # These will be used for downstream feature calculation and exclusion.
    cols_to_keep = ['Compound ChEMBL ID', 'SMILES', 'Activity Type', 'Standard Value (nM)', 'accession'] 
    
    final_target_ligands_df = pd.DataFrame(columns=cols_to_keep) # Ensure all columns exist
    for col in cols_to_keep:
        if col in target_ligands_df.columns:
            final_target_ligands_df[col] = target_ligands_df[col]
        else:
            final_target_ligands_df[col] = pd.NA
            
    return final_target_ligands_df.drop_duplicates(subset=['Compound ChEMBL ID', 'SMILES'])


def get_mf_keyword_id_from_keywords_csv(mf_keywords_csv_path, canonical_mf_name):
    """Helper to lookup KW-ID from the molecular_function_keywords.csv file."""
    try:
        mf_keywords_df = pd.read_csv(mf_keywords_csv_path)
        row = mf_keywords_df[mf_keywords_df['Name'].str.lower() == canonical_mf_name.lower()]
        if not row.empty:
            return row.iloc[0]['ID']
        logging.error(f"KW-ID not found for MF: '{canonical_mf_name}' in {mf_keywords_csv_path}")
    except Exception as e:
        logging.error(f"Error reading or searching {mf_keywords_csv_path} for {canonical_mf_name}: {e}")
    return None


def main():
    parser = argparse.ArgumentParser(description="Prepare excluded datasets using pre-calculated feature/fingerprint files.")
    parser.add_argument("--config_path", required=True, help="Path to the experiment_config.json file.")
    parser.add_argument("--target_id_name", required=True, help="Unique ID name for the target.")
    parser.add_argument("--output_dir", required=True, help="Directory to save the prepared CSV files.")
    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        config = json.load(f)
    gs = config['global_settings']
    
    current_target_info = next((t for t in config['targets'] if t['id_name'] == args.target_id_name), None)
    if not current_target_info:
        logging.error(f"Target ID '{args.target_id_name}' not found in config file. Aborting prepare_data.")
        return
    
    target_uniprot_id = current_target_info['uniprot_id']
    canonical_mf_name = current_target_info['molecular_function_canonical_name']
    mf_filename_segment = current_target_info['molecular_function_filename_segment']

    logging.info(f"Preparing data for target: {current_target_info['display_name']} (UniProt: {target_uniprot_id})")
    os.makedirs(args.output_dir, exist_ok=True)

    # --- 1. Get RAW target ligands (SMILES, IDs) from base ChEMBL data ---
    # These are the ligands that will be "held out" and later projected.
    # Their features/fingerprints will be calculated separately.
    try:
        raw_target_ligands_df = load_and_merge_chembl_data_for_raw_target_ligands(
            gs['chembl_affinity_full_csv_path'],
            gs['chembl_target_mapping_csv_path'],
            target_uniprot_id
        )
    except Exception as e:
        logging.error(f"Failed to load raw ChEMBL data to identify target ligands: {e}. Aborting prepare_data.")
        return

    # Save this raw list (SMILES, IDs) for calculate_features_and_fingerprints_exp.py
    raw_target_ligands_output_path = os.path.join(args.output_dir, f"{args.target_id_name}_target_ligands_for_feature_calc_raw.csv")
    if not raw_target_ligands_df.empty:
        # Keep only necessary ID columns for feature calculation input and SMILES exclusion
        cols_for_raw_output = [
        'SMILES', 
        'Compound ChEMBL ID', 
        'Activity Type',          # Good to have for context
        'Standard Value (nM)',    # CRITICAL for Spearman correlation
        'accession'               # The UniProt ID of the target itself
        ]


        df_to_save = pd.DataFrame(columns=cols_for_raw_output)
        for col in cols_for_raw_output:
            if col in raw_target_ligands_df:
                df_to_save[col] = raw_target_ligands_df[col]
        df_to_save.drop_duplicates().to_csv(raw_target_ligands_output_path, index=False)
        logging.info(f"Saved RAW target ligands (for feature calc) to: {raw_target_ligands_output_path} ({len(df_to_save)} unique records)")
    else:
        pd.DataFrame(columns=['SMILES', 'Compound ChEMBL ID']).to_csv(raw_target_ligands_output_path, index=False)
        logging.warning(f"No RAW target ligands identified for {target_uniprot_id}. Empty raw file created.")

    target_smiles_to_exclude = set(raw_target_ligands_df['SMILES'].dropna().unique()) if not raw_target_ligands_df.empty else set()
    target_chembl_ids_to_exclude = set(raw_target_ligands_df['Compound ChEMBL ID'].dropna().unique()) if not raw_target_ligands_df.empty else set()

    # --- 2. Filter PRE-CALCULATED ChEMBL Molecular Function files AND collect MF SMILES for ZINC filtering ---
    # CRITICAL: We need to collect ALL MF cloud SMILES to remove them from ZINC decoys!
    mf_smiles_to_exclude_from_zinc = set()  # Will accumulate ALL MF cloud SMILES
    
    mf_kw_id = get_mf_keyword_id_from_keywords_csv(gs['molecular_function_keywords_csv_path'], canonical_mf_name)
    if not mf_kw_id:
        logging.error(f"Cannot proceed with ChEMBL MF file filtering for {args.target_id_name} due to missing KW-ID for '{canonical_mf_name}'.")
    else:
        for repr_type in config['representations']: # "features", "fingerprints"
            precalc_chembl_mf_base_dir = gs.get(f'precalculated_chembl_mf_{repr_type}_base_dir')
            if not precalc_chembl_mf_base_dir:
                logging.error(f"Config missing 'precalculated_chembl_mf_{repr_type}_base_dir'. Skipping ChEMBL MF {repr_type} filtering.")
                continue

            suffix = "features" if repr_type == "features" else "fingerprints_ECFP4"
            precalc_mf_filename = f"{mf_kw_id}_{mf_filename_segment}_affinity_extracted_{suffix}.csv"
            precalc_mf_full_path = os.path.join(precalc_chembl_mf_base_dir, precalc_mf_filename)
            
            output_chembl_mf_excluded_path = os.path.join(args.output_dir, f"{args.target_id_name}_chembl_mf_excluded_{repr_type}.csv")

            if os.path.exists(precalc_mf_full_path):
                try:
                    df_precalc_mf = pd.read_csv(precalc_mf_full_path, low_memory=False)
                    logging.info(f"Loaded pre-calculated ChEMBL MF {repr_type} file: {precalc_mf_full_path} (shape: {df_precalc_mf.shape})")
                    
                    # Exclusion logic:
                    # Prefer 'accession' column if present in pre-calculated file.
                    # Fallback to 'Compound ChEMBL ID' if 'accession' is not there.
                    if 'accession' in df_precalc_mf.columns:
                        df_filtered_precalc_mf = df_precalc_mf[df_precalc_mf['accession'] != target_uniprot_id].copy()
                        logging.info(f"Filtered ChEMBL MF {repr_type} by 'accession'. Original: {len(df_precalc_mf)}, Filtered: {len(df_filtered_precalc_mf)}")
                    elif 'Compound ChEMBL ID' in df_precalc_mf.columns and target_chembl_ids_to_exclude:
                        df_filtered_precalc_mf = df_precalc_mf[~df_precalc_mf['Compound ChEMBL ID'].isin(target_chembl_ids_to_exclude)].copy()
                        logging.info(f"Filtered ChEMBL MF {repr_type} by 'Compound ChEMBL ID'. Original: {len(df_precalc_mf)}, Filtered: {len(df_filtered_precalc_mf)}")
                    else:
                        logging.warning(f"Cannot filter pre-calculated ChEMBL MF file {precalc_mf_full_path} for target {target_uniprot_id}. "
                                        "Neither 'accession' nor ('Compound ChEMBL ID' and target ChEMBL IDs) available for filtering. Using file as is.")
                        df_filtered_precalc_mf = df_precalc_mf.copy()
                    
                    df_filtered_precalc_mf.to_csv(output_chembl_mf_excluded_path, index=False)
                    logging.info(f"Saved filtered pre-calculated ChEMBL MF {repr_type} to: {output_chembl_mf_excluded_path}")
                    
                    # COLLECT MF SMILES for ZINC filtering (from filtered MF cloud)
                    if 'SMILES' in df_filtered_precalc_mf.columns:
                        mf_smiles_from_this_file = set(df_filtered_precalc_mf['SMILES'].dropna().unique())
                        mf_smiles_to_exclude_from_zinc.update(mf_smiles_from_this_file)
                        logging.info(f"Collected {len(mf_smiles_from_this_file)} unique SMILES from MF {repr_type} file for ZINC filtering")

                except Exception as e:
                    logging.error(f"Error processing pre-calculated ChEMBL MF file {precalc_mf_full_path}: {e}")
                    pd.DataFrame().to_csv(output_chembl_mf_excluded_path, index=False) # Save empty on error
            else:
                logging.error(f"Pre-calculated ChEMBL MF file not found: {precalc_mf_full_path}")
                pd.DataFrame().to_csv(output_chembl_mf_excluded_path, index=False)

    # --- 3. Filter PRE-CALCULATED ZINC files ---
    # CRITICAL: Remove BOTH target ligands AND MF cloud molecules from ZINC decoys!
    logging.info(f"\n{'='*80}")
    logging.info(f"DATA INTEGRITY CHECK: Filtering ZINC decoys")
    logging.info(f"{'='*80}")
    logging.info(f"Target ligand SMILES to exclude: {len(target_smiles_to_exclude)}")
    logging.info(f"MF cloud SMILES to exclude: {len(mf_smiles_to_exclude_from_zinc)}")
    
    # Combine both exclusion sets
    all_smiles_to_exclude_from_zinc = target_smiles_to_exclude.union(mf_smiles_to_exclude_from_zinc)
    logging.info(f"TOTAL SMILES to exclude from ZINC: {len(all_smiles_to_exclude_from_zinc)}")
    
    for repr_type in config['representations']:
        precalc_zinc_path = gs.get(f'precalculated_zinc_{repr_type}_path')
        if not precalc_zinc_path:
            logging.error(f"Config missing 'precalculated_zinc_{repr_type}_path'. Skipping ZINC {repr_type} filtering.")
            continue
            
        output_zinc_excluded_path = os.path.join(args.output_dir, f"{args.target_id_name}_zinc_excluded_{repr_type}.csv")

        if os.path.exists(precalc_zinc_path):
            try:
                df_precalc_zinc = pd.read_csv(precalc_zinc_path, low_memory=False)
                logging.info(f"Loaded pre-calculated ZINC {repr_type} file: {precalc_zinc_path} (shape: {df_precalc_zinc.shape})")

                if 'SMILES' in df_precalc_zinc.columns and all_smiles_to_exclude_from_zinc:
                    # Filter out BOTH target ligands AND MF cloud molecules
                    df_filtered_precalc_zinc = df_precalc_zinc[~df_precalc_zinc['SMILES'].isin(all_smiles_to_exclude_from_zinc)].copy()
                    removed_count = len(df_precalc_zinc) - len(df_filtered_precalc_zinc)
                    logging.info(f"Filtered ZINC {repr_type} by SMILES (target + MF cloud exclusion).")
                    logging.info(f"  Original: {len(df_precalc_zinc)}, Filtered: {len(df_filtered_precalc_zinc)}, Removed: {removed_count}")
                    
                    # Report breakdown
                    if target_smiles_to_exclude:
                        zinc_target_overlap = df_precalc_zinc['SMILES'].isin(target_smiles_to_exclude).sum()
                        logging.info(f"  - Removed {zinc_target_overlap} ZINC molecules matching target ligands")
                    if mf_smiles_to_exclude_from_zinc:
                        zinc_mf_overlap = df_precalc_zinc['SMILES'].isin(mf_smiles_to_exclude_from_zinc).sum()
                        logging.info(f"  - Removed {zinc_mf_overlap} ZINC molecules matching MF cloud")
                else:
                    logging.warning(f"Cannot filter pre-calculated ZINC {repr_type} file by SMILES (column missing or no exclusions). Using file as is.")
                    df_filtered_precalc_zinc = df_precalc_zinc.copy()
                
                df_filtered_precalc_zinc.to_csv(output_zinc_excluded_path, index=False)
                logging.info(f"Saved filtered pre-calculated ZINC {repr_type} to: {output_zinc_excluded_path}")

            except Exception as e:
                logging.error(f"Error processing pre-calculated ZINC file {precalc_zinc_path}: {e}")
                pd.DataFrame().to_csv(output_zinc_excluded_path, index=False)
        else:
            logging.error(f"Pre-calculated ZINC file not found: {precalc_zinc_path}")
            pd.DataFrame().to_csv(output_zinc_excluded_path, index=False)

    logging.info(f"Data preparation for {args.target_id_name} (using pre-calculated files) completed.")

if __name__ == "__main__":
    main()