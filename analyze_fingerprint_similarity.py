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
from scipy.spatial.distance import pdist, squareform

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)-8s - %(message)s')

FINGERPRINT_COLUMN_NAME = "Fingerprint"
NUM_FINGERPRINT_BITS = 2048

def analyze_similarity_and_duplicates(file_paths, sample_size=5000, chunksize=50000):
    """
    Analyzes fingerprint data for duplicates and calculates pairwise similarity on a sample.
    
    Args:
        file_paths (list): A list of paths to the fingerprint CSV files.
        sample_size (int): The number of fingerprints to sample for pairwise similarity calculation.
        chunksize (int): The number of rows to process per chunk.

    Returns:
        dict: A dictionary containing analysis results.
    """
    logging.info(f"Starting similarity and duplicate analysis on {len(file_paths)} file(s).")
    
    seen_fingerprints = set()
    all_fingerprints_list = [] # Will store fingerprints for sampling
    total_fingerprints = 0
    duplicate_count = 0

    for file_path in file_paths:
        if not os.path.exists(file_path):
            logging.warning(f"File not found, skipping: {file_path}")
            continue
            
        logging.info(f"Processing file in chunks: {file_path}")
        try:
            with pd.read_csv(file_path, chunksize=chunksize, low_memory=False) as reader:
                for chunk in tqdm(reader, desc=f"Reading {os.path.basename(file_path)}"):
                    if FINGERPRINT_COLUMN_NAME not in chunk.columns:
                        logging.warning(f"'{FINGERPRINT_COLUMN_NAME}' not in chunk. Skipping.")
                        continue

                    chunk.dropna(subset=[FINGERPRINT_COLUMN_NAME], inplace=True)
                    if chunk.empty:
                        continue
                    
                    total_fingerprints += len(chunk)
                    
                    for fp_string in chunk[FINGERPRINT_COLUMN_NAME]:
                        if fp_string in seen_fingerprints:
                            duplicate_count += 1
                        else:
                            seen_fingerprints.add(fp_string)
                            all_fingerprints_list.append(fp_string)

        except Exception as e:
            logging.error(f"Error processing file {file_path}: {e}", exc_info=True)

    if total_fingerprints == 0:
        logging.error("No valid fingerprints were processed.")
        return None

    num_unique = len(seen_fingerprints)
    analysis_results = {
        "total_fingerprints_processed": total_fingerprints,
        "unique_fingerprints": num_unique,
        "duplicate_fingerprints": duplicate_count,
        "uniqueness_ratio": num_unique / total_fingerprints if total_fingerprints > 0 else 0
    }
    
    logging.info("--- Duplicate Analysis Results ---")
    logging.info(f"Total Fingerprints Processed: {analysis_results['total_fingerprints_processed']:,}")
    logging.info(f"Unique Fingerprints Found:   {analysis_results['unique_fingerprints']:,}")
    logging.info(f"Duplicate Fingerprints Found: {analysis_results['duplicate_fingerprints']:,}")
    logging.info(f"Uniqueness Ratio:             {analysis_results['uniqueness_ratio']:.4f}")

    # --- Pairwise Tanimoto Similarity Calculation on a Sample ---
    if num_unique < 2:
        logging.warning("Not enough unique fingerprints to calculate pairwise similarity.")
        analysis_results['pairwise_similarities'] = None
        return analysis_results

    # Take a random sample from the unique fingerprints for pairwise calculation
    sample_size = min(sample_size, num_unique)
    logging.info(f"Taking a random sample of {sample_size} unique fingerprints for pairwise Tanimoto calculation...")
    
    # Use numpy for efficient random choice on the list of unique strings
    fp_sample_strings = np.random.choice(all_fingerprints_list, size=sample_size, replace=False)
    
    # Convert the sample of strings to a binary NumPy array
    fp_sample_array = np.array([list(map(int, s.split(','))) for s in fp_sample_strings], dtype=np.bool_)

    logging.info(f"Calculating pairwise Jaccard distances for {sample_size}x{sample_size} matrix...")
    # pdist calculates the condensed distance matrix (upper triangle)
    # For binary data, 'jaccard' distance is 1 - Tanimoto similarity
    try:
        # Use scipy's pdist for a fast, memory-efficient calculation
        jaccard_distances = pdist(fp_sample_array, 'jaccard')
        
        # Convert Jaccard distance to Tanimoto similarity
        tanimoto_similarities = 1 - jaccard_distances
        
        analysis_results['pairwise_similarities'] = tanimoto_similarities
        logging.info("Pairwise similarity calculation complete.")

    except Exception as e:
        logging.error(f"Failed to calculate pairwise similarities: {e}", exc_info=True)
        analysis_results['pairwise_similarities'] = None

    return analysis_results


def plot_similarity_histogram(similarities, output_dir):
    """Generates and saves a histogram of pairwise Tanimoto similarities."""
    if similarities is None or len(similarities) == 0:
        logging.error("Cannot plot, similarities array is None or empty.")
        return

    logging.info("Generating pairwise similarity histogram...")
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.hist(similarities, bins=50, range=(0, 1), density=True, edgecolor='black', alpha=0.7)
    plt.title(f"Distribution of Pairwise Tanimoto Similarities (Sample of {args.sample_size})", fontsize=14)
    plt.xlabel("Tanimoto Similarity", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.xlim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    mean_sim = np.mean(similarities)
    median_sim = np.median(similarities)
    plt.axvline(mean_sim, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {mean_sim:.3f}')
    plt.axvline(median_sim, color='green', linestyle='dashed', linewidth=1.5, label=f'Median: {median_sim:.3f}')
    plt.legend()
    
    plot_path = os.path.join(output_dir, "fingerprint_pairwise_similarity_histogram.png")
    plt.savefig(plot_path)
    plt.close()
    logging.info(f"Saved similarity histogram to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze fingerprint data for duplicates and pairwise similarity.")
    parser.add_argument("--base_experiment_dir", required=True, help="Base directory containing all hyperparameter sweep run folders (e.g., 'experiment_workspace_hyperparam_sweep/').")
    parser.add_argument("--target_id_name", required=True, help="The target_id_name used to find the specific input files (e.g., 'TyrosineProteinKinaseABL1_P00519').")
    parser.add_argument("--sample_size", type=int, default=5000, help="Number of unique fingerprints to sample for pairwise similarity calculation.")
    parser.add_argument("--output_dir", default="similarity_analysis_results", help="Directory to save plots and analysis results.")
    args.parser.parse_args()

    # Create output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Find one representative run to get the data from
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
    
    # Perform analysis
    analysis_results = analyze_similarity_and_duplicates(files_to_process, sample_size=args.sample_size)
    
    if analysis_results:
        # Save summary report to JSON
        report_path = os.path.join(args.output_dir, "similarity_analysis_summary.json")
        # Convert numpy array to list for JSON serialization
        if analysis_results.get('pairwise_similarities') is not None:
            analysis_results['pairwise_similarities_summary'] = {
                'count': len(analysis_results['pairwise_similarities']),
                'mean': np.mean(analysis_results['pairwise_similarities']),
                'std': np.std(analysis_results['pairwise_similarities']),
                'min': np.min(analysis_results['pairwise_similarities']),
                'max': np.max(analysis_results['pairwise_similarities'])
            }
            # Avoid saving the huge array to the JSON file
            del analysis_results['pairwise_similarities']
        
        try:
            with open(report_path, 'w') as f:
                json.dump(analysis_results, f, indent=4)
            logging.info(f"Saved analysis summary to: {report_path}")
        except Exception as e:
            logging.error(f"Failed to save analysis summary JSON: {e}")

        # Plot histogram if data is available
        similarities = analysis_results.get('pairwise_similarities_summary')
        if similarities: # Check if the summary was created
             # For plotting, we need the original array which was deleted. Let's recalculate or re-run.
             # Better: Pass the original array to the plotting function before deleting.
             # Let's adjust the flow.
             pass # See corrected main flow below

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze fingerprint data for duplicates and pairwise similarity.")
    parser.add_argument("--base_experiment_dir", required=True, help="Base directory containing all hyperparameter sweep run folders (e.g., 'experiment_workspace_hyperparam_sweep/').")
    parser.add_argument("--target_id_name", required=True, help="The target_id_name used to find the specific input files (e.g., 'TyrosineProteinKinaseABL1_P00519').")
    parser.add_argument("--sample_size", type=int, default=5000, help="Number of unique fingerprints to sample for pairwise similarity calculation.")
    parser.add_argument("--output_dir", default="similarity_analysis_results", help="Directory to save plots and analysis results.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    pattern = os.path.join(args.base_experiment_dir, "run_seed*_reprfingerprints_*")
    replicate_dirs = glob.glob(pattern)
    if not replicate_dirs:
        logging.error(f"No fingerprint replicate runs found in '{args.base_experiment_dir}'. Cannot proceed.")
        exit() # Use exit in __main__
    
    representative_run_dir = replicate_dirs[0]
    logging.info(f"Using data from representative run: {os.path.basename(representative_run_dir)}")
    temp_data_dir = os.path.join(representative_run_dir, args.target_id_name, "temp_data")
    chembl_file = os.path.join(temp_data_dir, f"{args.target_id_name}_chembl_mf_excluded_fingerprints.csv")
    zinc_file = os.path.join(temp_data_dir, f"{args.target_id_name}_zinc_excluded_fingerprints.csv")
    files_to_process = [chembl_file, zinc_file]
    
    analysis_results = analyze_similarity_and_duplicates(files_to_process, sample_size=args.sample_size)
    
    if analysis_results:
        # Plot histogram using the full similarity array
        similarities_array = analysis_results.get('pairwise_similarities')
        if similarities_array is not None:
            plot_similarity_histogram(similarities_array, args.output_dir)

        # Save summary report to JSON (after plotting)
        report_path = os.path.join(args.output_dir, "similarity_analysis_summary.json")
        # Create summary stats and remove the large array for the JSON file
        if analysis_results.get('pairwise_similarities') is not None:
            analysis_results['pairwise_similarities_summary'] = {
                'count': len(analysis_results['pairwise_similarities']),
                'mean': np.mean(analysis_results['pairwise_similarities']),
                'std': np.std(analysis_results['pairwise_similarities']),
                'min': np.min(analysis_results['pairwise_similarities']),
                'max': np.max(analysis_results['pairwise_similarities'])
            }
            del analysis_results['pairwise_similarities']
        
        try:
            with open(report_path, 'w') as f:
                json.dump(analysis_results, f, indent=4)
            logging.info(f"Saved analysis summary to: {report_path}")
        except Exception as e:
            logging.error(f"Failed to save analysis summary JSON: {e}")