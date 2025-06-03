# standalone_tsne_test_combined.py
import os
import argparse
import logging
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Attempt cuML imports
CUML_AVAILABLE = False
SKLEARN_TSNE_AVAILABLE = True 

try:
    from cuml import TSNE as cumlTSNE, PCA as cumlPCA
    CUML_AVAILABLE = True
except ImportError:
    print("cuML not found. Will use scikit-learn for PCA and t-SNE.")

from sklearn.manifold import TSNE as sklearnTSNE
from sklearn.decomposition import PCA as sklearnPCA

# --- Robust Logging Setup ---
logger = logging.getLogger("StandaloneTSNETestCombined") 
logger.propagate = False 
logger.setLevel(logging.DEBUG) 

if logger.hasHandlers():
    logger.handlers.clear()

log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)-8s - %(filename)-25s - %(funcName)-25s - %(lineno)-4d - %(message)s')
log_file_path = f"standalone_tsne_test_combined_{time.strftime('%Y%m%d_%H%M%S')}.log"
try:
    file_handler = logging.FileHandler(log_file_path, mode='w')
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.DEBUG) 
    logger.addHandler(file_handler)
except Exception as e:
    print(f"CRITICAL: Failed to initialize file logger for {log_file_path}: {e}")
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_formatter)
stream_handler.setLevel(logging.INFO) 
logger.addHandler(stream_handler)
# --- End Logging Setup ---

# --- Fingerprint Parsing Logic (adapted from calculate_similarityspaces_exp.py) ---
FINGERPRINT_COLUMN_PREFIX = "fp_"
NUM_FINGERPRINT_BITS = 2048 # Assuming ECFP4 2048-bit, adjust if different
PRECALCULATED_FP_STRING_COLUMN_NAME = "Fingerprint" # The name of the string column

def parse_fingerprint_string_column_in_df(df_input, 
                                          fp_string_col_name=PRECALCULATED_FP_STRING_COLUMN_NAME, 
                                          num_bits=NUM_FINGERPRINT_BITS):
    """
    Parses a column containing fingerprint strings "0,1,0,..." into separate bit columns.
    Modifies the DataFrame in-place by adding new fp_ columns and dropping the original string column.
    Returns the modified DataFrame.
    """
    logger.info(f"Attempting to parse fingerprint string column: '{fp_string_col_name}'")
    
    if fp_string_col_name not in df_input.columns:
        logger.warning(f"Fingerprint string column '{fp_string_col_name}' not found in DataFrame. No parsing done.")
        return df_input # Return original df if column not found

    fp_matrix = np.full((len(df_input), num_bits), np.nan) 

    num_parse_errors = 0
    num_length_mismatches = 0

    for idx, fp_str in enumerate(df_input[fp_string_col_name]):
        if pd.isna(fp_str) or not isinstance(fp_str, str) or not fp_str.strip():
            continue # Will remain NaN
        bits = fp_str.split(',')
        if len(bits) == num_bits:
            try:
                fp_matrix[idx, :] = [int(b) for b in bits]
            except ValueError:
                 if num_parse_errors < 5: # Log only a few examples
                     logger.debug(f"ValueError parsing bits in string (row index {df_input.index[idx]}): {fp_str[:30]}... Filling with NaNs.")
                 num_parse_errors += 1
        else:
            if num_length_mismatches < 5: # Log only a few examples
                logger.debug(f"Fingerprint string (row index {df_input.index[idx]}) '{fp_str[:30]}...' has incorrect length {len(bits)}, expected {num_bits}. Filling with NaNs.")
            num_length_mismatches +=1
    
    if num_parse_errors > 0:
        logger.warning(f"Encountered {num_parse_errors} ValueError instances during fingerprint string parsing.")
    if num_length_mismatches > 0:
        logger.warning(f"Encountered {num_length_mismatches} length mismatches during fingerprint string parsing.")

    fp_col_names = [f"{FINGERPRINT_COLUMN_PREFIX}{i}" for i in range(num_bits)]
    df_fp_bits = pd.DataFrame(fp_matrix, columns=fp_col_names, index=df_input.index)
    
    # Drop the original string column and concatenate new bit columns
    df_output = df_input.drop(columns=[fp_string_col_name])
    df_output = pd.concat([df_output, df_fp_bits], axis=1)
    
    logger.info(f"Fingerprint string column '{fp_string_col_name}' parsed. Added {len(fp_col_names)} new columns. New shape: {df_output.shape}")
    return df_output
# --- End Fingerprint Parsing Logic ---


def select_descriptor_columns(df, column_prefix=None, column_list=None, is_fingerprint_run=False):
    if is_fingerprint_run: # For fingerprints, after parsing, columns will start with FINGERPRINT_COLUMN_PREFIX
        prefix_to_use = column_prefix if column_prefix else FINGERPRINT_COLUMN_PREFIX
        logger.info(f"Selecting fingerprint columns with prefix: '{prefix_to_use}'")
        actual_cols = [col for col in df.columns if col.startswith(prefix_to_use)]
        if not actual_cols:
            logger.error(f"No fingerprint columns found with prefix '{prefix_to_use}'. Available: {df.columns.tolist()[:10]}")
            return None
        logger.info(f"Found {len(actual_cols)} fingerprint columns with prefix '{prefix_to_use}'.")
        return actual_cols
    elif column_list: # For features
        logger.info(f"Attempting to use explicit feature column list: {column_list[:5]}...")
        actual_cols = [col for col in column_list if col in df.columns]
        if not actual_cols:
            logger.error(f"None of the specified feature columns found. Available: {df.columns.tolist()[:10]}")
            return None
        missing_cols = [col for col in column_list if col not in actual_cols]
        if missing_cols:
            logger.warning(f"Specified feature columns not found and will be ignored: {missing_cols}")
        logger.info(f"Using {len(actual_cols)} explicitly specified feature columns.")
        return actual_cols
    else:
        logger.error("For features, must specify column_list. For fingerprints, parsing will create prefixed columns.")
        return None


def load_and_combine_data(chembl_csv_path, zinc_csv_path, is_fingerprint_run):
    df_list = []
    
    if not os.path.exists(chembl_csv_path):
        logger.error(f"ChEMBL input CSV not found: {chembl_csv_path}")
        return None
    try:
        logger.info(f"Loading ChEMBL data from: {chembl_csv_path}")
        df_chembl = pd.read_csv(chembl_csv_path, low_memory=False)
        logger.info(f"ChEMBL data loaded. Shape: {df_chembl.shape}")
        if df_chembl.empty: logger.warning("ChEMBL DataFrame is empty.")
        
        if is_fingerprint_run:
            df_chembl = parse_fingerprint_string_column_in_df(df_chembl)
        
        if 'DataSource' not in df_chembl.columns: df_chembl['DataSource'] = 'ChEMBL_MF_Excluded'
        df_list.append(df_chembl)
    except Exception as e:
        logger.error(f"Failed to load ChEMBL CSV {chembl_csv_path}: {e}", exc_info=True)
        return None

    if zinc_csv_path and os.path.exists(zinc_csv_path):
        try:
            logger.info(f"Loading ZINC data from: {zinc_csv_path}")
            df_zinc = pd.read_csv(zinc_csv_path, low_memory=False)
            logger.info(f"ZINC data loaded. Shape: {df_zinc.shape}")
            if df_zinc.empty: logger.warning("ZINC DataFrame is empty.")

            if is_fingerprint_run:
                df_zinc = parse_fingerprint_string_column_in_df(df_zinc)

            if 'DataSource' not in df_zinc.columns: df_zinc['DataSource'] = 'ZINC_Excluded'
            df_list.append(df_zinc)
        except Exception as e:
            logger.error(f"Failed to load ZINC CSV {zinc_csv_path}: {e}", exc_info=True)
    elif zinc_csv_path:
        logger.warning(f"ZINC input CSV specified but not found: {zinc_csv_path}. Proceeding without it.")
    else:
        logger.info("No ZINC input CSV specified. Proceeding with ChEMBL data only.")

    if not df_list:
        logger.error("No dataframes were successfully loaded or processed.")
        return None
        
    logger.info("Combining loaded dataframes...")
    all_cols = set()
    for df_item in df_list: all_cols.update(df_item.columns)
    
    aligned_df_list = []
    for df_item in df_list:
        for col in all_cols:
            if col not in df_item.columns: df_item[col] = np.nan
        aligned_df_list.append(df_item[list(all_cols)])

    combined_df = pd.concat(aligned_df_list, ignore_index=True)
    logger.info(f"Combined DataFrame shape: {combined_df.shape}. Columns: {combined_df.columns.tolist()[:10]}...")
    return combined_df


def run_tsne_on_combined_data(args):
    logger.info(f"--- Starting t-SNE Test on Combined Data ---")
    logger.info(f"Arguments: {args}")

    is_fingerprint_run = args.run_type == "fingerprints"
    logger.info(f"Run type: {'Fingerprints' if is_fingerprint_run else 'Features'}")

    # 1. Load, Parse (if fingerprints), and Combine Data
    df_combined = load_and_combine_data(args.chembl_csv, args.zinc_csv, is_fingerprint_run)
    if df_combined is None or df_combined.empty:
        logger.error("Failed to load or combine data. Exiting.")
        return

    # 2. Select Descriptor Columns
    descriptor_columns = None
    if is_fingerprint_run:
        # For fingerprints, after parsing, columns will start with FINGERPRINT_COLUMN_PREFIX
        # The --descriptor_prefix argument can override this default if needed.
        descriptor_columns = select_descriptor_columns(df_combined, column_prefix=args.descriptor_prefix, is_fingerprint_run=True)
    elif args.descriptor_cols: # For features
        try:
            descriptor_list = [col.strip() for col in args.descriptor_cols.split(',')]
        except Exception as e:
            logger.error(f"Could not parse descriptor_cols string: {e}")
            descriptor_list = None
        descriptor_columns = select_descriptor_columns(df_combined, column_list=descriptor_list, is_fingerprint_run=False)
    else:
        logger.error("For features, --descriptor_cols must be specified. For fingerprints, parsing creates columns (use --descriptor_prefix if not default 'fp_').")
        return
        
    if not descriptor_columns:
        logger.error("Failed to identify descriptor columns in combined data. Exiting.")
        return
    logger.info(f"Selected {len(descriptor_columns)} descriptor columns. First 5: {descriptor_columns[:5]}")

    # 3. Prepare Data for DR (already numeric from parsing or explicit selection)
    try:
        data_for_dr = df_combined[descriptor_columns].copy() # Ensure it's a copy
        # Convert to numeric one last time just in case parsing left some non-numerics (shouldn't happen with np.nan fill)
        data_for_dr = data_for_dr.apply(pd.to_numeric, errors='coerce') 
        
        initial_rows = len(data_for_dr)
        original_valid_indices = data_for_dr.dropna(how='any').index 
        data_for_dr.dropna(how='any', inplace=True)
        rows_after_na_drop = len(data_for_dr)

        if rows_after_na_drop < initial_rows:
            logger.info(f"Dropped {initial_rows - rows_after_na_drop} rows due to NaNs in descriptor columns (post-parsing/selection).")
        if data_for_dr.empty:
            logger.error("Combined DataFrame is empty after dropping NaNs from descriptor columns.")
            return
        logger.info(f"Data ready for DR. Shape: {data_for_dr.shape}")
        X = data_for_dr.values.astype(np.float32)
    except Exception as e:
        logger.error(f"Error preparing data for DR: {e}", exc_info=True)
        return

    # ... (Rest of the script: Scaling, PCA, t-SNE, Saving, Plotting - remains the same as previous version) ...
    # ... The X variable now holds the combined, parsed, and cleaned data ...
    # 4. Scaling
    try:
        logger.info("Scaling data using StandardScaler...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        logger.info(f"Data scaled. Shape: {X_scaled.shape}")
    except Exception as e:
        logger.error(f"Error during scaling: {e}", exc_info=True)
        return

    # 5. Optional Initial PCA
    X_input_for_tsne = X_scaled
    if args.pca_components > 0:
        logger.info(f"Performing initial PCA to {args.pca_components} components...")
        n_pca_components = min(args.pca_components, X_scaled.shape[0] -1 if X_scaled.shape[0] > 1 else 1, X_scaled.shape[1])
        if n_pca_components < 1 : 
             logger.warning(f"Effective n_components for PCA ({n_pca_components}) is less than 1. Trying to proceed but t-SNE might fail. Data shape: {X_scaled.shape}")
             if n_pca_components == 0: 
                logger.error("Cannot run PCA with 0 components. Skipping PCA.")
                n_pca_components = -1 
        if n_pca_components >= 1:
            logger.info(f"Effective PCA components: {n_pca_components}")
            try:
                pca_engine_name = "cuML" if CUML_AVAILABLE and args.use_cuml else "scikit-learn"
                logger.info(f"Using {pca_engine_name} PCA...")
                pca_model = (cumlPCA(n_components=n_pca_components, random_state=42) if CUML_AVAILABLE and args.use_cuml
                              else sklearnPCA(n_components=n_pca_components, random_state=42))
                X_input_for_tsne = pca_model.fit_transform(X_scaled)
                logger.info(f"PCA completed. Shape after PCA: {X_input_for_tsne.shape}")
            except Exception as e:
                logger.error(f"Error during initial PCA: {e}", exc_info=True)
                logger.warning("Proceeding with un-PCA'd scaled data for t-SNE due to PCA error.")
                X_input_for_tsne = X_scaled 
        else:
            X_input_for_tsne = X_scaled 
    else:
        logger.info("Skipping initial PCA.")

    tsne_results = {}
    perplexity_to_use = min(args.perplexity, X_input_for_tsne.shape[0] - 2 if X_input_for_tsne.shape[0] >1 else 0) 
    if X_input_for_tsne.shape[0] <= 1 or perplexity_to_use < 1: # Check if perplexity is valid or samples are too few
        logger.error(f"Too few samples ({X_input_for_tsne.shape[0]}) or invalid perplexity ({perplexity_to_use}) for t-SNE. Exiting t-SNE step.")
        return
    if perplexity_to_use != args.perplexity:
        logger.info(f"Using adjusted perplexity: {perplexity_to_use} (original: {args.perplexity})")
    
    if X_input_for_tsne.shape[0] <= perplexity_to_use: # Should be caught above, but defensive
        logger.error(f"Number of samples ({X_input_for_tsne.shape[0]}) must be greater than perplexity ({perplexity_to_use}). Cannot run t-SNE.")
        return

    if CUML_AVAILABLE and args.use_cuml:
        logger.info(f"--- Attempting cuML t-SNE ---")
        cuml_tsne_verbose_level = 0 
        if logger.level == logging.DEBUG: cuml_tsne_verbose_level = 5 
        elif logger.level == logging.INFO: cuml_tsne_verbose_level = 1 
        try:
            start_time_cuml = time.time()
            cuml_tsne = cumlTSNE(n_components=2, perplexity=perplexity_to_use, random_state=42, 
                                 method='barnes_hut', verbose=cuml_tsne_verbose_level,
                                 n_neighbors=args.n_neighbors if args.n_neighbors else None) 
            logger.info(f"Initialized cuML TSNE. Input data shape for t-SNE: {X_input_for_tsne.shape}")
            embedding_cuml = cuml_tsne.fit_transform(X_input_for_tsne)
            end_time_cuml = time.time()
            logger.info(f"cuML t-SNE completed in {end_time_cuml - start_time_cuml:.2f} seconds. Embedding shape: {embedding_cuml.shape}")
            tsne_results['cuml'] = embedding_cuml
        except Exception as e:
            logger.error(f"cuML t-SNE FAILED: {e}", exc_info=True)
            tsne_results['cuml'] = None

    if not args.use_cuml_only or tsne_results.get('cuml') is None:
        logger.info(f"--- Attempting scikit-learn t-SNE ---")
        sklearn_tsne_verbose_level = 0
        if logger.level == logging.DEBUG : sklearn_tsne_verbose_level = 2
        elif logger.level == logging.INFO : sklearn_tsne_verbose_level = 1
        try:
            start_time_sklearn = time.time()
            X_input_for_sklearn_tsne = X_input_for_tsne
            if hasattr(X_input_for_tsne, 'get'): 
                X_input_for_sklearn_tsne = X_input_for_tsne.get()
            logger.info(f"Initialized scikit-learn TSNE. Input data shape for t-SNE: {X_input_for_sklearn_tsne.shape}")
            sklearn_tsne = sklearnTSNE(n_components=2, perplexity=perplexity_to_use, random_state=42, 
                                       method='barnes_hut', n_jobs=-1, verbose=sklearn_tsne_verbose_level,
                                       n_iter=args.n_iter, learning_rate=args.learning_rate)
            embedding_sklearn = sklearn_tsne.fit_transform(X_input_for_sklearn_tsne)
            end_time_sklearn = time.time()
            logger.info(f"scikit-learn t-SNE completed in {end_time_sklearn - start_time_sklearn:.2f} seconds. Embedding shape: {embedding_sklearn.shape}")
            tsne_results['sklearn'] = embedding_sklearn
        except Exception as e:
            logger.error(f"scikit-learn t-SNE FAILED: {e}", exc_info=True)
            tsne_results['sklearn'] = None

    chembl_basename = os.path.splitext(os.path.basename(args.chembl_csv))[0]
    output_file_prefix = f"{chembl_basename}_plus_zinc" if args.zinc_csv and os.path.exists(args.zinc_csv) else chembl_basename
    output_file_prefix += f"_{args.run_type}_pca{args.pca_components}_perp{int(perplexity_to_use)}" # Use actual perplexity

    for method_name, embedding in tsne_results.items():
        if embedding is not None:
            output_df_tsne_coords = pd.DataFrame(embedding, columns=['tSNE-1', 'tSNE-2'], index=original_valid_indices)
            if args.join_info_cols:
                info_cols_to_join = [col for col in df_combined.columns if col not in descriptor_columns]
                if info_cols_to_join:
                    output_df_final = df_combined.loc[original_valid_indices, info_cols_to_join].join(output_df_tsne_coords, how="inner")
                else: output_df_final = output_df_tsne_coords
            else: output_df_final = output_df_tsne_coords
            tsne_output_filename = f"{output_file_prefix}_tsne_results_{method_name}.csv"
            try:
                output_df_final.to_csv(tsne_output_filename, index=False)
                logger.info(f"Saved {method_name} t-SNE results to: {tsne_output_filename}")
                if args.plot:
                    import matplotlib.pyplot as plt 
                    plt.figure(figsize=(12, 10))
                    if 'DataSource' in output_df_final.columns:
                        for source_type, group_data in output_df_final.groupby('DataSource'):
                            plt.scatter(group_data['tSNE-1'], group_data['tSNE-2'], s=5, alpha=0.5, label=source_type)
                        plt.legend(title="Data Source")
                    else:
                        plt.scatter(output_df_final['tSNE-1'], output_df_final['tSNE-2'], s=5, alpha=0.5)
                    plt.title(f"t-SNE ({method_name}, PCA={args.pca_components}, Perp={perplexity_to_use}) for {os.path.basename(args.chembl_csv)}")
                    plt.xlabel("tSNE-1"); plt.ylabel("tSNE-2"); plt.grid(True)
                    plot_filename = f"{output_file_prefix}_tsne_plot_{method_name}.png"
                    plt.savefig(plot_filename, dpi=150); logger.info(f"Saved {method_name} t-SNE plot to: {plot_filename}")
                    plt.close()
            except Exception as e:
                logger.error(f"Error saving/plotting {method_name} t-SNE results: {e}", exc_info=True)
        else: logger.warning(f"No embedding result for {method_name} t-SNE to save/plot.")
    logger.info(f"--- t-SNE Test on Combined Data Finished (Log: {log_file_path}) ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone t-SNE Test Script for Combined ChEMBL and ZINC data.")
    parser.add_argument("--chembl_csv", required=True, help="Path to the ChEMBL (excluded) input CSV file.")
    parser.add_argument("--zinc_csv", type=str, default=None, help="Path to the ZINC (excluded) input CSV file (optional).")
    
    # New argument to distinguish run type
    parser.add_argument("--run_type", required=True, choices=['features', 'fingerprints'], help="Specify if running for features or fingerprints (for parsing).")
    
    parser.add_argument("--descriptor_prefix", type=str, default=FINGERPRINT_COLUMN_PREFIX, help=f"Prefix for fingerprint columns IF run_type is 'fingerprints' and they are already expanded (default: {FINGERPRINT_COLUMN_PREFIX}). If fingerprints are in a single string column, this script handles parsing from '{PRECALCULATED_FP_STRING_COLUMN_NAME}'.")
    parser.add_argument("--descriptor_cols", type=str, default=None, help="Comma-separated string of exact descriptor column names (used if run_type is 'features').")
    
    parser.add_argument("--pca_components", type=int, default=50, help="Number of components for initial PCA (0 to skip).")
    parser.add_argument("--perplexity", type=float, default=30.0, help="Perplexity for t-SNE.")
    parser.add_argument("--n_iter", type=int, default=1000, help="Number of iterations for t-SNE (sklearn).")
    parser.add_argument("--learning_rate", type=str, default='auto', help="Learning rate for t-SNE (sklearn). Can be float or 'auto'.")
    parser.add_argument("--n_neighbors", type=int, default=None, help="Number of neighbors for cuML t-SNE. If None, cuML uses default based on perplexity.")
    parser.add_argument("--use_cuml", action='store_true', help="Attempt to use cuML for PCA and t-SNE if available.")
    parser.add_argument("--use_cuml_only", action='store_true', help="Only run cuML versions, skip sklearn if cuML fails or not available.")
    parser.add_argument("--plot", action='store_true', help="Generate a scatter plot of the 2D t-SNE results.")
    parser.add_argument("--join_info_cols", action='store_true', help="Join original non-descriptor columns to the output t-SNE CSV.")
    
    cli_args = parser.parse_args()

    if cli_args.run_type == 'features' and not cli_args.descriptor_cols:
        logger.error("For --run_type features, you must specify --descriptor_cols.")
        exit(1)
    # For fingerprints, the script will try to parse "Fingerprint" column first, 
    # then use --descriptor_prefix if that column isn't found (assuming pre-expanded).

    if cli_args.learning_rate.lower() != 'auto':
        try:
            cli_args.learning_rate = float(cli_args.learning_rate)
        except ValueError:
            logger.error(f"Invalid learning_rate: {cli_args.learning_rate}. Must be a float or 'auto'. Defaulting to 'auto'.")
            cli_args.learning_rate = 'auto'

    run_tsne_on_combined_data(cli_args)