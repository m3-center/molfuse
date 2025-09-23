import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import argparse
import logging
import os
import gc
from datetime import datetime

# --- Configuration ---
FINGERPRINT_COLUMN_PREFIX = "fp_"
NUM_FINGERPRINT_BITS = 2048
PRECALCULATED_FP_STRING_COLUMN_NAME = "Fingerprint"

# --- Logging Setup ---
log_file_name = f"pca_variance_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)-8s - %(message)s',
                    handlers=[logging.FileHandler(log_file_name, mode='w'), logging.StreamHandler()])

def find_elbow_point(cumulative_variance):
    """
    Finds the elbow of a cumulative variance curve.
    This is the point on the curve furthest from the line connecting the first and last points.
    """
    n_points = len(cumulative_variance)
    if n_points < 3:
        return n_points - 1
        
    all_coords = np.vstack((range(n_points), cumulative_variance)).T
    first_point, last_point = all_coords[0], all_coords[-1]
    line_vec = last_point - first_point
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
    vec_from_first = all_coords - first_point
    scalar_product = np.sum(vec_from_first * np.tile(line_vec_norm, (n_points, 1)), axis=1)
    vec_from_first_parallel = np.outer(scalar_product, line_vec_norm)
    vec_to_line = vec_from_first - vec_from_first_parallel
    dist_to_line = np.sqrt(np.sum(vec_to_line ** 2, axis=1))
    return np.argmax(dist_to_line)

def main():
    parser = argparse.ArgumentParser(description="Perform an Explained Variance analysis using PCA on a large fingerprint dataset.")
    parser.add_argument("--chembl_mf_data_path", required=True, help="Path to the ChEMBL Molecular Function fingerprint data CSV.")
    parser.add_argument("--zinc_data_path", required=True, help="Path to the ZINC decoys fingerprint data CSV.")
    parser.add_argument("--output_plot_path", default="explained_variance_plot.png", help="Path to save the output plot.")
    parser.add_argument("--max_components", type=int, default=250, help="The maximum number of principal components to calculate.")
    args = parser.parse_args()

    logging.info("--- STARTING EXPLAINED VARIANCE ANALYSIS ---")
    
    input_files = [args.chembl_mf_data_path, args.zinc_data_path]
    
    # --- Step 1: Load and parse all data in chunks ---
    logging.info("Step 1: Loading and parsing the complete fingerprint dataset in chunks...")
    fp_matrix_chunks = []
    total_rows = 0
    
    for file_path in input_files:
        logging.info(f"  Processing {file_path}...")
        try:
            chunk_iterator = pd.read_csv(file_path, chunksize=50000, low_memory=False, iterator=True)
            for chunk in chunk_iterator:
                if PRECALCULATED_FP_STRING_COLUMN_NAME not in chunk.columns:
                    logging.warning(f"'{PRECALCULATED_FP_STRING_COLUMN_NAME}' column not found in a chunk from {file_path}. Skipping chunk.")
                    continue
                
                fp_strings = chunk[PRECALCULATED_FP_STRING_COLUMN_NAME].dropna()
                if not fp_strings.empty:
                    parsed_chunk = np.array([list(map(int, s.split(','))) for s in fp_strings], dtype=np.int8)
                    fp_matrix_chunks.append(parsed_chunk)
                    total_rows += len(parsed_chunk)
                del chunk; gc.collect()
        except Exception as e:
            logging.error(f"Failed while processing {file_path}: {e}", exc_info=True)
            return

    if not fp_matrix_chunks:
        logging.error("No valid fingerprint data could be loaded from any input file. Aborting.")
        return

    logging.info(f"Combining chunks into a single data matrix...")
    fp_matrix = np.vstack(fp_matrix_chunks)
    del fp_matrix_chunks; gc.collect()
    
    logging.info(f"Successfully loaded data matrix with shape: {fp_matrix.shape}")

    # --- Step 2: Scale the data ---
    logging.info("Step 2: Scaling the full dataset...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(fp_matrix)
    del fp_matrix; gc.collect()

    # --- Step 3: Fit PCA ---
    logging.info("Step 3: Fitting PCA model...")
    n_components = min(args.max_components, X_scaled.shape[0], X_scaled.shape[1])
    logging.info(f"  Calculating up to {n_components} principal components.")
    
    pca = PCA(n_components=n_components)
    pca.fit(X_scaled)
    logging.info("PCA model has been successfully fitted.")
    
    # --- Step 4: Plot the results ---
    logging.info("Step 4: Generating the explained variance plot...")
    
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    elbow_point = find_elbow_point(cumulative_variance)
    variance_at_elbow = cumulative_variance[elbow_point]

    plt.figure(figsize=(12, 8))
    plt.plot(range(1, n_components + 1), cumulative_variance * 100, marker='.', linestyle='-', color='b')
    plt.xlabel("Number of Principal Components", fontsize=14)
    plt.ylabel("Cumulative Explained Variance (%)", fontsize=14)
    plt.title("Cumulative Explained Variance by Principal Components", fontsize=16)
    plt.grid(True, which="both", linestyle='--')
    plt.xticks(np.arange(0, n_components + 1, 25))
    plt.yticks(np.arange(0, 101, 10))
    
    plt.axvline(x=elbow_point + 1, color='r', linestyle='--', label=f'Elbow Point ({elbow_point + 1} components)')
    plt.axhline(y=variance_at_elbow * 100, color='r', linestyle='--')
    plt.annotate(f'Elbow at {elbow_point + 1} components\n({variance_at_elbow:.2%} variance explained)',
                 xy=(elbow_point + 1, variance_at_elbow * 100),
                 xytext=(elbow_point + 20, variance_at_elbow * 100 - 15),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 fontsize=12, backgroundcolor='w')
    
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(args.output_plot_path, dpi=300)
    
    logging.info(f"Plot saved to: {args.output_plot_path}")
    logging.info("---")
    logging.info(f"ANALYSIS COMPLETE: The identified elbow is at {elbow_point + 1} components.")
    logging.info(f"This captures {variance_at_elbow:.2%} of the total variance.")
    logging.info("Based on this, a value between 50 and 100 would be a robust choice for pre-processing.")
    logging.info("---")

if __name__ == "__main__":
    main()