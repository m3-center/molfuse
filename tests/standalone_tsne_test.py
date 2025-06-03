# standalone_tsne_test.py
import os
import argparse
import logging
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Attempt cuML imports
CUML_AVAILABLE = False
SKLEARN_TSNE_AVAILABLE = True # Assume sklearn is always there for tSNE

try:
    from cuml import TSNE as cumlTSNE, PCA as cumlPCA
    from cuml.common.device_selection import using_device_type # For ensuring CPU fallback if needed
    CUML_AVAILABLE = True
except ImportError:
    print("cuML not found. Will use scikit-learn for PCA and t-SNE.")

# Always import sklearn versions for fallback/comparison
from sklearn.manifold import TSNE as sklearnTSNE
from sklearn.decomposition import PCA as sklearnPCA


# --- Robust Logging Setup ---
# Get the root logger
logger = logging.getLogger("StandaloneTSNETest") # Give it a unique name
logger.propagate = False # Prevent messages from going to the root logger if it's configured elsewhere
logger.setLevel(logging.DEBUG) # Set to DEBUG for maximum verbosity for this test

# Clear existing handlers from THIS logger to avoid duplicates if script is re-run
if logger.hasHandlers():
    logger.handlers.clear()

log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)-8s - %(filename)-25s - %(funcName)-25s - %(lineno)-4d - %(message)s')

# File Handler
log_file_path = f"standalone_tsne_test_{time.strftime('%Y%m%d_%H%M%S')}.log"
try:
    file_handler = logging.FileHandler(log_file_path, mode='w') # 'w' for a fresh log each test
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.DEBUG) 
    logger.addHandler(file_handler)
except Exception as e:
    print(f"CRITICAL: Failed to initialize file logger for {log_file_path}: {e}")

# Stream Handler (console output)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_formatter)
stream_handler.setLevel(logging.INFO) # Console can be less verbose
logger.addHandler(stream_handler)

# --- End Logging Setup ---


def select_descriptor_columns(df, column_prefix=None, column_list=None):
    """Selects descriptor columns based on prefix or explicit list."""
    if column_list:
        logger.info(f"Attempting to use explicit column list: {column_list[:5]}...")
        actual_cols = [col for col in column_list if col in df.columns]
        if not actual_cols:
            logger.error(f"None of the specified columns found in DataFrame. Available: {df.columns.tolist()[:10]}")
            return None
        missing_cols = [col for col in column_list if col not in actual_cols]
        if missing_cols:
            logger.warning(f"Specified columns not found and will be ignored: {missing_cols}")
        logger.info(f"Using {len(actual_cols)} explicitly specified columns.")
        return actual_cols
    elif column_prefix:
        logger.info(f"Attempting to select columns with prefix: '{column_prefix}'")
        actual_cols = [col for col in df.columns if col.startswith(column_prefix)]
        if not actual_cols:
            logger.error(f"No columns found with prefix '{column_prefix}'. Available: {df.columns.tolist()[:10]}")
            return None
        logger.info(f"Found {len(actual_cols)} columns with prefix '{column_prefix}'.")
        return actual_cols
    else:
        logger.error("Must specify either column_prefix or column_list.")
        return None


def run_tsne_test(args):
    logger.info(f"--- Starting t-SNE Test ---")
    logger.info(f"Arguments: {args}")

    if not os.path.exists(args.input_csv):
        logger.error(f"Input CSV not found: {args.input_csv}")
        return

    # 1. Load Data
    try:
        logger.info(f"Loading data from: {args.input_csv}")
        df = pd.read_csv(args.input_csv, low_memory=False)
        logger.info(f"Data loaded. Shape: {df.shape}")
        if df.empty:
            logger.error("Input DataFrame is empty.")
            return
    except Exception as e:
        logger.error(f"Failed to load CSV: {e}", exc_info=True)
        return

    # 2. Select Descriptor Columns
    if args.descriptor_cols:
        try:
            # If descriptor_cols is a string like "col1,col2,col3"
            descriptor_list = [col.strip() for col in args.descriptor_cols.split(',')]
            logger.debug(f"Parsed descriptor list: {descriptor_list}")
        except Exception as e:
            logger.error(f"Could not parse descriptor_cols string: {e}")
            descriptor_list = None
        descriptor_columns = select_descriptor_columns(df, column_list=descriptor_list)
    elif args.descriptor_prefix:
        descriptor_columns = select_descriptor_columns(df, column_prefix=args.descriptor_prefix)
    else:
        logger.error("No descriptor columns specified via --descriptor_cols or --descriptor_prefix.")
        return
        
    if not descriptor_columns:
        logger.error("Failed to identify descriptor columns. Exiting.")
        return
    logger.info(f"Selected {len(descriptor_columns)} descriptor columns. First 5: {descriptor_columns[:5]}")

    # 3. Prepare Data for DR (convert to numeric, drop NaNs)
    try:
        logger.debug(f"Converting descriptor columns to numeric...")
        data_for_dr = df[descriptor_columns].apply(pd.to_numeric, errors='coerce')
        initial_rows = len(data_for_dr)
        data_for_dr.dropna(how='any', inplace=True) # Drop rows with any NaN in descriptor columns
        rows_after_na_drop = len(data_for_dr)
        if rows_after_na_drop < initial_rows:
            logger.info(f"Dropped {initial_rows - rows_after_na_drop} rows due to NaNs in descriptor columns.")
        if data_for_dr.empty:
            logger.error("DataFrame is empty after dropping NaNs from descriptor columns.")
            return
        logger.info(f"Data ready for DR. Shape: {data_for_dr.shape}")
        X = data_for_dr.values.astype(np.float32) # Ensure float32 for potential cuML use
    except Exception as e:
        logger.error(f"Error preparing data for DR: {e}", exc_info=True)
        return

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
        # Ensure n_components for PCA is valid
        n_pca_components = min(args.pca_components, X_scaled.shape[0] -1, X_scaled.shape[1])
        if n_pca_components < 2 :
             logger.error(f"Cannot perform PCA: effective n_components ({n_pca_components}) is less than 2. Data shape: {X_scaled.shape}")
             return

        logger.info(f"Effective PCA components: {n_pca_components}")
        try:
            if CUML_AVAILABLE and args.use_cuml:
                logger.info("Using cuML PCA...")
                pca_model = cumlPCA(n_components=n_pca_components, random_state=42)
            else:
                logger.info("Using scikit-learn PCA...")
                pca_model = sklearnPCA(n_components=n_pca_components, random_state=42)
            
            X_input_for_tsne = pca_model.fit_transform(X_scaled)
            logger.info(f"PCA completed. Shape after PCA: {X_input_for_tsne.shape}")
        except Exception as e:
            logger.error(f"Error during initial PCA: {e}", exc_info=True)
            return
    else:
        logger.info("Skipping initial PCA.")

    # 6. Run t-SNE
    tsne_results = {}

    # Attempt cuML t-SNE if available and requested
    if CUML_AVAILABLE and args.use_cuml:
        logger.info(f"--- Attempting cuML t-SNE ---")
        logger.info(f"Parameters: n_components=2, perplexity={args.perplexity}, random_state=42, method='barnes_hut'")
        cuml_tsne_verbose_level = 0 
        if logger.level == logging.DEBUG: cuml_tsne_verbose_level = 5 # Higher verbosity for cuML if debugging
        elif logger.level == logging.INFO: cuml_tsne_verbose_level = 1 
        
        try:
            start_time_cuml = time.time()
            cuml_tsne = cumlTSNE(n_components=2, 
                                 perplexity=args.perplexity, 
                                 random_state=42, 
                                 method='barnes_hut', # or 'exact' for smaller datasets
                                 verbose=cuml_tsne_verbose_level,
                                 n_neighbors=args.n_neighbors if args.n_neighbors else None # Pass n_neighbors if specified
                                 ) 
            logger.info(f"Initialized cuML TSNE object. Input data shape for t-SNE: {X_input_for_tsne.shape}")
            
            # Ensure data is in a format cuML expects (cupy array or compatible numpy)
            # X_input_for_tsne is already float32 numpy array
            
            embedding_cuml = cuml_tsne.fit_transform(X_input_for_tsne)
            end_time_cuml = time.time()
            logger.info(f"cuML t-SNE completed in {end_time_cuml - start_time_cuml:.2f} seconds.")
            logger.info(f"cuML t-SNE embedding shape: {embedding_cuml.shape}")
            tsne_results['cuml'] = embedding_cuml
        except Exception as e:
            logger.error(f"cuML t-SNE FAILED: {e}", exc_info=True)
            if "Typically occurs when " in str(e): # Check for common cuML error messages
                 logger.error("This might be related to data size, perplexity, or n_neighbors. Try adjusting these.")
            tsne_results['cuml'] = None


    # Always attempt/run scikit-learn t-SNE (or if cuML failed/not used)
    if not args.use_cuml_only or tsne_results.get('cuml') is None:
        logger.info(f"--- Attempting scikit-learn t-SNE ---")
        logger.info(f"Parameters: n_components=2, perplexity={args.perplexity}, random_state=42, method='barnes_hut', n_jobs=-1")
        sklearn_tsne_verbose_level = 0
        if logger.level == logging.DEBUG : sklearn_tsne_verbose_level = 2 # Sklearn verbose levels are different
        elif logger.level == logging.INFO : sklearn_tsne_verbose_level = 1

        try:
            start_time_sklearn = time.time()
            # Make sure X_input_for_tsne is on CPU (numpy array) for sklearn
            X_input_for_sklearn_tsne = X_input_for_tsne
            if hasattr(X_input_for_tsne, 'get'): # Check if it's a CuPy array
                logger.debug("Converting data from GPU to CPU for scikit-learn t-SNE.")
                X_input_for_sklearn_tsne = X_input_for_tsne.get()
            
            logger.info(f"Initialized scikit-learn TSNE object. Input data shape for t-SNE: {X_input_for_sklearn_tsne.shape}")
            sklearn_tsne = sklearnTSNE(n_components=2, 
                                       perplexity=args.perplexity, 
                                       random_state=42, 
                                       method='barnes_hut', # 'exact' for N < 4000 might be better
                                       n_jobs=-1, # Use all cores
                                       verbose=sklearn_tsne_verbose_level,
                                       n_iter=args.n_iter,
                                       learning_rate=args.learning_rate
                                       )
            embedding_sklearn = sklearn_tsne.fit_transform(X_input_for_sklearn_tsne)
            end_time_sklearn = time.time()
            logger.info(f"scikit-learn t-SNE completed in {end_time_sklearn - start_time_sklearn:.2f} seconds.")
            logger.info(f"scikit-learn t-SNE embedding shape: {embedding_sklearn.shape}")
            tsne_results['sklearn'] = embedding_sklearn
        except Exception as e:
            logger.error(f"scikit-learn t-SNE FAILED: {e}", exc_info=True)
            tsne_results['sklearn'] = None


    # 7. Save results
    output_basename = os.path.splitext(os.path.basename(args.input_csv))[0]
    for method_name, embedding in tsne_results.items():
        if embedding is not None:
            output_df = pd.DataFrame(embedding, columns=['tSNE-1', 'tSNE-2'], index=data_for_dr.index) # Use original index of valid rows
            # Optionally, join back with original non-descriptor columns for context
            if args.join_info_cols:
                info_cols_to_join = [col for col in df.columns if col not in descriptor_columns]
                if info_cols_to_join:
                    output_df = df.loc[data_for_dr.index, info_cols_to_join].join(output_df, how="inner")

            tsne_output_filename = f"{output_basename}_tsne_results_{method_name}.csv"
            try:
                output_df.to_csv(tsne_output_filename, index=False)
                logger.info(f"Saved {method_name} t-SNE results to: {tsne_output_filename}")

                # 8. Optional Plotting
                if args.plot:
                    import matplotlib.pyplot as plt # Local import for plotting
                    plt.figure(figsize=(10, 8))
                    plt.scatter(output_df['tSNE-1'], output_df['tSNE-2'], s=5, alpha=0.5)
                    plt.title(f"t-SNE results ({method_name}) for {output_basename}")
                    plt.xlabel("tSNE-1")
                    plt.ylabel("tSNE-2")
                    plt.grid(True)
                    plot_filename = f"{output_basename}_tsne_plot_{method_name}.png"
                    plt.savefig(plot_filename)
                    logger.info(f"Saved {method_name} t-SNE plot to: {plot_filename}")
                    plt.close()

            except Exception as e:
                logger.error(f"Error saving/plotting {method_name} t-SNE results: {e}", exc_info=True)
        else:
            logger.warning(f"No embedding result for {method_name} t-SNE to save/plot.")
            
    logger.info("--- t-SNE Test Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone t-SNE Test Script.")
    parser.add_argument("--input_csv", required=True, help="Path to the input CSV file.")
    parser.add_argument("--descriptor_prefix", type=str, default=None, help="Prefix for descriptor columns (e.g., 'fp_').")
    parser.add_argument("--descriptor_cols", type=str, default=None, help="Comma-separated string of exact descriptor column names.")
    parser.add_argument("--pca_components", type=int, default=50, help="Number of components for initial PCA (0 to skip).")
    parser.add_argument("--perplexity", type=float, default=30.0, help="Perplexity for t-SNE.")
    parser.add_argument("--n_iter", type=int, default=1000, help="Number of iterations for t-SNE (sklearn).")
    parser.add_argument("--learning_rate", type=str, default='auto', help="Learning rate for t-SNE (sklearn). Can be float or 'auto'.") # Changed to str for 'auto'
    parser.add_argument("--n_neighbors", type=int, default=None, help="Number of neighbors for cuML t-SNE (perplexity method). If None, uses default based on perplexity.")

    parser.add_argument("--use_cuml", action='store_true', help="Attempt to use cuML for PCA and t-SNE if available.")
    parser.add_argument("--use_cuml_only", action='store_true', help="Only run cuML versions, skip sklearn if cuML fails or not available.")
    parser.add_argument("--plot", action='store_true', help="Generate a scatter plot of the 2D t-SNE results.")
    parser.add_argument("--join_info_cols", action='store_true', help="Join original non-descriptor columns to the output t-SNE CSV.")
    
    cli_args = parser.parse_args()

    # Validate descriptor specification
    if not cli_args.descriptor_prefix and not cli_args.descriptor_cols:
        logger.error("You must specify either --descriptor_prefix or --descriptor_cols.")
        exit(1)
    if cli_args.descriptor_prefix and cli_args.descriptor_cols:
        logger.warning("Both --descriptor_prefix and --descriptor_cols specified. --descriptor_cols will be used.")
        cli_args.descriptor_prefix = None # Prioritize explicit list

    # Convert learning_rate if it's not 'auto'
    if cli_args.learning_rate.lower() != 'auto':
        try:
            cli_args.learning_rate = float(cli_args.learning_rate)
        except ValueError:
            logger.error(f"Invalid learning_rate: {cli_args.learning_rate}. Must be a float or 'auto'. Defaulting to 'auto'.")
            cli_args.learning_rate = 'auto'


    run_tsne_test(cli_args)