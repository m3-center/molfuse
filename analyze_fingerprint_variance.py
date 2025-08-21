import pandas as pd
import numpy as np
import os
import argparse
import logging
import json
import glob
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)-8s - %(message)s')

FINGERPRINT_COLUMN_NAME = "Fingerprint"
NUM_FINGERPRINT_BITS = 2048

def calculate_bit_variance_from_files_in_chunks(file_paths, chunksize=50000):
    """
    Calculates the variance for each bit across large CSV files in a memory-efficient way.
    
    This function uses an online algorithm to compute variance by tracking the count,
    sum, and sum of squares for each bit across all chunks.
    
    Args:
        file_paths (list): A list of paths to the fingerprint CSV files.
        chunksize (int): The number of rows to process per chunk.

    Returns:
        numpy.ndarray: An array of shape (NUM_FINGERPRINT_BITS,) containing the variance for each bit.
    """
    logging.info("Starting memory-efficient variance calculation...")
    
    # Initialize accumulators for online variance calculation
    n_samples = 0
    sum_x = np.zeros(NUM_FINGERPRINT_BITS, dtype=np.float64)  # Sum of each bit's values (i.e., count of ON bits)
    sum_x2 = np.zeros(NUM_FINGERPRINT_BITS, dtype=np.float64) # Sum of squares of each bit's values (same as sum_x for binary)

    for file_path in file_paths:
        if not os.path.exists(file_path):
            logging.warning(f"File not found, skipping: {file_path}")
            continue
            
        logging.info(f"Processing file in chunks: {file_path}")
        try:
            # Use tqdm to show progress for large files
            with pd.read_csv(file_path, chunksize=chunksize, low_memory=False) as reader:
                for chunk in tqdm(reader, desc=f"Reading {os.path.basename(file_path)}"):
                    if FINGERPRINT_COLUMN_NAME not in chunk.columns:
                        logging.warning(f"'{FINGERPRINT_COLUMN_NAME}' not in chunk. Skipping chunk.")
                        continue

                    # Drop rows where the fingerprint string is missing
                    chunk.dropna(subset=[FINGERPRINT_COLUMN_NAME], inplace=True)
                    if chunk.empty:
                        continue
                    
                    # Efficiently parse the string column into a NumPy array
                    # The list comprehension is faster here than df.apply for this specific task
                    parsed_bits = np.array([list(map(int, s.split(','))) for s in chunk[FINGERPRINT_COLUMN_NAME]], dtype=np.int8)
                    
                    if parsed_bits.shape[1] != NUM_FINGERPRINT_BITS:
                        logging.error(f"Fingerprint length mismatch found. Expected {NUM_FINGERPRINT_BITS}, got {parsed_bits.shape[1]}. Skipping chunk.")
                        continue
                    
                    # Update statistics
                    n_samples += parsed_bits.shape[0]
                    sum_x += np.sum(parsed_bits, axis=0)
                    # For binary data (0 or 1), x^2 is the same as x
                    sum_x2 += np.sum(parsed_bits**2, axis=0)
                    
        except Exception as e:
            logging.error(f"Error processing file {file_path}: {e}", exc_info=True)

    if n_samples == 0:
        logging.error("No valid samples found to calculate variance.")
        return None

    # Calculate variance using the formula: Var(X) = E[X^2] - (E[X])^2
    mean_x = sum_x / n_samples
    mean_x2 = sum_x2 / n_samples
    variance = mean_x2 - (mean_x ** 2)
    
    logging.info(f"Variance calculation complete. Processed a total of {n_samples} molecules.")
    return variance

def plot_variances(variances, output_dir):
    """Generates and saves sorted and unsorted plots of bit variances."""
    if variances is None:
        logging.error("Cannot plot, variance array is None.")
        return

    logging.info("Generating variance plots...")
    os.makedirs(output_dir, exist_ok=True)
    
    # --- 1. Unsorted Variance Plot ---
    plt.figure(figsize=(12, 6))
    plt.bar(range(NUM_FINGERPRINT_BITS), variances, width=1.0)
    plt.title(f"Variance of Each Bit Across the Dataset (Unsorted)", fontsize=14)
    plt.xlabel("Fingerprint Bit Index", fontsize=12)
    plt.ylabel("Variance", fontsize=12)
    plt.xlim(0, NUM_FINGERPRINT_BITS)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    unsorted_path = os.path.join(output_dir, "fingerprint_bit_variance_unsorted.png")
    plt.savefig(unsorted_path)
    plt.close()
    logging.info(f"Saved unsorted variance plot to: {unsorted_path}")

    # --- 2. Sorted Variance Plot ---
    sorted_indices = np.argsort(variances)[::-1] # Sort descending
    sorted_vars = variances[sorted_indices]
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(NUM_FINGERPRINT_BITS), sorted_vars, width=1.0)
    plt.title(f"Variance of Each Bit Across the Dataset (Sorted Descending)", fontsize=14)
    plt.xlabel("Bit Rank (Sorted by Variance)", fontsize=12)
    plt.ylabel("Variance", fontsize=12)
    plt.xlim(0, NUM_FINGERPRINT_BITS)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add lines for top k bits
    top_k_values = [256, 512, 1024]
    for k in top_k_values:
        if k < NUM_FINGERPRINT_BITS:
            plt.axvline(x=k, color='red', linestyle='--', linewidth=1, label=f'Top {k} bits')

    plt.legend()
    sorted_path = os.path.join(output_dir, "fingerprint_bit_variance_sorted.png")
    plt.savefig(sorted_path)
    plt.close()
    logging.info(f"Saved sorted variance plot to: {sorted_path}")
    
    return sorted_indices

def main():
    parser = argparse.ArgumentParser(description="Analyze fingerprint bit variance and perform feature selection.")
    parser.add_argument("--base_experiment_dir", required=True, help="Base directory containing all hyperparameter sweep run folders (e.g., 'experiment_workspace_hyperparam_sweep/').")
    parser.add_argument("--target_id_name", required=True, help="The target_id_name used to find the specific input files (e.g., 'TyrosineProteinKinaseABL1_P00519').")
    parser.add_argument("--num_bits_to_select", type=int, default=512, help="Number of top variant bits to select and save.")
    parser.add_argument("--output_dir", default="feature_selection_results", help="Directory to save plots and selected bit lists.")
    args = parser.parse_args()

    # Create output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Find one representative run to get the data from (data is same for all replicates)
    pattern = os.path.join(args.base_experiment_dir, "run_seed*_reprfingerprints_*")
    replicate_dirs = glob.glob(pattern)
    if not replicate_dirs:
        logging.error(f"No fingerprint replicate runs found in '{args.base_experiment_dir}'. Cannot proceed.")
        return
    
    representative_run_dir = replicate_dirs[0]
    logging.info(f"Using data from representative run: {os.path.basename(representative_run_dir)}")

    temp_data_dir = os.path.join(representative_run_dir, args.target_id_name, "temp_data")
    
    chembl_file = os.path.join(temp_data_dir, f"{args.target_id_name}_chembl_mf_excluded_fingerprints.csv")
    zinc_file = os.path.join(temp_data_dir, f"{args.target_id_name}_zinc_excluded_fingerprints.csv")

    files_to_process = [chembl_file, zinc_file]
    
    # Calculate variance
    bit_variances = calculate_bit_variance_from_files_in_chunks(files_to_process)
    
    if bit_variances is not None:
        # Visualize and get sorted indices
        sorted_bit_indices = plot_variances(bit_variances, args.output_dir)
        
        # Select and save top K bits
        if sorted_bit_indices is not None and args.num_bits_to_select > 0:
            top_k_indices = sorted_bit_indices[:args.num_bits_to_select]
            logging.info(f"Top {args.num_bits_to_select} most variant bit indices (first 10): {top_k_indices[:10]}")
            
            # Save the list of indices to a file
            output_file_path = os.path.join(args.output_dir, f"top_{args.num_bits_to_select}_variant_fp_bits.json")
            try:
                with open(output_file_path, 'w') as f:
                    # Convert numpy array to list for JSON serialization
                    json.dump(top_k_indices.tolist(), f)
                logging.info(f"Saved list of top {args.num_bits_to_select} bit indices to: {output_file_path}")
            except Exception as e:
                logging.error(f"Failed to save selected bit indices: {e}")

if __name__ == "__main__":
    main()