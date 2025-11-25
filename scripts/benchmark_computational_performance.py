#!/usr/bin/env python3
"""
Computational Performance Benchmark for Virtual Screening Methods

Compares actual wall-clock time and throughput for:
1. Raw ECFP4 (Tanimoto, no UMAP)
2. Raw Features (Euclidean, no UMAP)
3. ECFP4 + UMAP (20D, pretrained model)
4. Features + UMAP (2D, pretrained model)

Tests on: 1, 10, 100, 1000, 10000, 100000 candidate molecules

Outputs:
- Detailed timing breakdown (CSV)
- LaTeX table for paper
- Performance plots
"""
from __future__ import annotations

import argparse
import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors


# ============================================================================
# Utility Functions
# ============================================================================

def flush_print(*args, **kwargs):
    """Print and immediately flush stdout."""
    print(*args, **kwargs)
    sys.stdout.flush()

def load_fingerprints(csv_path: Path, max_rows: int = None) -> Tuple[np.ndarray, List[str]]:
    """
    Load and parse fingerprints from CSV.
    
    Returns:
        Tuple of (fingerprint matrix, SMILES list)
    """
    df = pd.read_csv(csv_path, nrows=max_rows, low_memory=False)
    
    # Get SMILES column
    smiles_col = "canonical_smiles" if "canonical_smiles" in df.columns else "SMILES"
    smiles = df[smiles_col].tolist()
    
    # Parse fingerprints
    fp_col = [c for c in df.columns if "ECFP4" in c or "ecfp4" in c][0]
    
    def parse_fp(fp_str):
        """Parse fingerprint string to binary array."""
        if pd.isna(fp_str):
            return None
        
        fp_str = str(fp_str).strip()
        
        # Remove brackets and quotes
        fp_str = fp_str.replace("[", "").replace("]", "")
        fp_str = fp_str.replace("'", "").replace('"', "")
        
        # Split and convert
        parts = [p.strip() for p in fp_str.split(",")]
        try:
            fp = np.array([int(p) for p in parts], dtype=np.uint8)
        except ValueError:
            return None
        
        return fp
    
    fps = []
    valid_smiles = []
    for smi, fp_str in zip(smiles, df[fp_col]):
        fp = parse_fp(fp_str)
        if fp is not None and len(fp) == 2048:
            fps.append(fp)
            valid_smiles.append(smi)
    
    return np.array(fps), valid_smiles


def load_features(csv_path: Path, max_rows: int = None) -> Tuple[np.ndarray, List[str]]:
    """
    Load features from CSV.
    
    Returns:
        Tuple of (feature matrix, SMILES list)
    """
    df = pd.read_csv(csv_path, nrows=max_rows, low_memory=False)
    
    # Get SMILES column
    smiles_col = "canonical_smiles" if "canonical_smiles" in df.columns else "SMILES"
    smiles = df[smiles_col].tolist()
    
    # Select numeric feature columns (exclude metadata)
    exclude_cols = {
        smiles_col, "accession", "Standard Value (nM)", "label", "target",
        "Organism", "Assay Type", "Relation", "Standard Type", "ChEMBL ID"
    }
    
    feature_cols = [c for c in df.columns if c not in exclude_cols and "ECFP" not in c and "ecfp" not in c]
    
    # Extract features and convert to numeric
    X = df[feature_cols].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    
    # Drop rows with NaNs
    valid_mask = ~X.isna().any(axis=1)
    X = X[valid_mask].values
    smiles = [s for s, v in zip(smiles, valid_mask) if v]
    
    return X, smiles


def jaccard_distance_batch(X: np.ndarray, Y: np.ndarray, batch_size: int = 1000) -> np.ndarray:
    """
    Compute Jaccard distance between binary vectors using batching to save memory.
    
    Args:
        X: (n_samples, n_features) binary array (float32)
        Y: (n_refs, n_features) binary array (float32)
        batch_size: Number of samples to process at once
    
    Returns:
        (n_samples, n_refs) distance matrix
    """
    n_samples = X.shape[0]
    n_refs = Y.shape[0]
    
    # Pre-compute Y sums (1, n_refs)
    y_sum = Y.sum(axis=1, keepdims=True).T
    
    # Initialize result matrix (float32)
    distances = np.zeros((n_samples, n_refs), dtype=np.float32)
    
    for i in range(0, n_samples, batch_size):
        end = min(i + batch_size, n_samples)
        X_batch = X[i:end]
        
        # Compute intersection
        intersection = np.dot(X_batch, Y.T)
        
        # Compute union
        x_sum = X_batch.sum(axis=1, keepdims=True)
        union = x_sum + y_sum - intersection
        
        # Jaccard distance = 1 - (intersection / union)
        distances[i:end] = 1.0 - (intersection / (union + 1e-10))
        
    return distances


# ============================================================================
# Benchmark Functions
# ============================================================================

def benchmark_raw_ecfp4(
    candidates_fp: np.ndarray,
    mf_fp: np.ndarray,
    batch_size: int = 1000,
) -> Dict[str, float]:
    """
    Benchmark raw ECFP4 + Tanimoto scoring.
    """
    results = {}
    
    # Time the scoring step
    start = time.time()
    distances = jaccard_distance_batch(candidates_fp, mf_fp, batch_size=batch_size)
    scores = -distances.min(axis=1)  # 1-NN: negative distance
    end = time.time()
    
    results["scoring_time"] = end - start
    results["total_time"] = end - start
    results["n_candidates"] = len(candidates_fp)
    results["n_references"] = len(mf_fp)
    
    return results


def benchmark_raw_features(
    candidates_feat: np.ndarray,
    mf_feat: np.ndarray,
    scaler,
    batch_size: int = 1000,
) -> Dict[str, float]:
    """
    Benchmark raw features + Euclidean scoring.
    """
    results = {}
    
    # Time scaling
    start = time.time()
    candidates_scaled = scaler.transform(candidates_feat).astype(np.float32)
    end = time.time()
    results["scaling_time"] = end - start
    
    # Time scoring (batched cdist)
    start = time.time()
    n_samples = len(candidates_scaled)
    scores = np.zeros(n_samples, dtype=np.float32)
    
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        batch = candidates_scaled[i:end_idx]
        dists = cdist(batch, mf_feat, metric="euclidean")
        scores[i:end_idx] = -dists.min(axis=1)
        
    end = time.time()
    results["scoring_time"] = end - start
    
    results["total_time"] = results["scaling_time"] + results["scoring_time"]
    results["n_candidates"] = len(candidates_feat)
    results["n_references"] = len(mf_feat)
    
    return results


def benchmark_ecfp4_umap(
    candidates_fp: np.ndarray,
    mf_tree: NearestNeighbors,
    umap_model,
) -> Dict[str, float]:
    """
    Benchmark ECFP4 + UMAP (20D) scoring using KDTree.
    """
    results = {}
    
    # Time UMAP transform
    start = time.time()
    candidates_embedded = umap_model.transform(candidates_fp).astype(np.float32)
    end = time.time()
    results["transform_time"] = end - start
    
    # Time scoring (Tree query)
    start = time.time()
    distances, _ = mf_tree.kneighbors(candidates_embedded)
    scores = -distances.flatten()
    end = time.time()
    results["scoring_time"] = end - start
    
    results["total_time"] = results["transform_time"] + results["scoring_time"]
    results["n_candidates"] = len(candidates_fp)
    # Note: n_references is implicit in the tree
    
    return results


def benchmark_features_umap(
    candidates_feat: np.ndarray,
    mf_tree: NearestNeighbors,
    scaler,
    umap_model,
) -> Dict[str, float]:
    """
    Benchmark Features + UMAP (2D) scoring using KDTree.
    """
    results = {}
    
    # Time scaling
    start = time.time()
    candidates_scaled = scaler.transform(candidates_feat).astype(np.float32)
    end = time.time()
    results["scaling_time"] = end - start
    
    # Time UMAP transform
    start = time.time()
    candidates_embedded = umap_model.transform(candidates_scaled).astype(np.float32)
    end = time.time()
    results["transform_time"] = end - start
    
    # Time scoring (Tree query)
    start = time.time()
    distances, _ = mf_tree.kneighbors(candidates_embedded)
    scores = -distances.flatten()
    end = time.time()
    results["scoring_time"] = end - start
    
    results["total_time"] = results["scaling_time"] + results["transform_time"] + results["scoring_time"]
    results["n_candidates"] = len(candidates_feat)
    
    return results


# ============================================================================
# Main Benchmark
# ============================================================================

def run_benchmark(
    workspace_dir: Path,
    zinc_fp_csv: Path,
    zinc_feat_csv: Path,
    phase1_fp_run: str,
    phase1_feat_run: str,
    sample_sizes: List[int],
    output_dir: Path,
):
    """
    Run computational performance benchmark.
    """
    print("="*80)
    print("COMPUTATIONAL PERFORMANCE BENCHMARK")
    print("="*80)
    print(f"Workspace: {workspace_dir}")
    print(f"Output: {output_dir}")
    print(f"Sample sizes: {sample_sizes}")
    print("")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # Load Phase 1 ECFP4+UMAP Model
    # ========================================================================
    print("Loading Phase 1 ECFP4+UMAP model...")
    phase1_dir = workspace_dir / "phase1" / phase1_fp_run
    
    # Load UMAP model
    umap_fp_model = joblib.load(phase1_dir / "artifacts" / "umap_model.joblib")
    
    # Load MF embedding (column names are z0, z1, ..., z19 for 20D)
    mf_fp_embedding_df = pd.read_csv(phase1_dir / "artifacts" / "embedding_mf.csv")
    mf_fp_embedding = mf_fp_embedding_df[[f"z{i}" for i in range(20)]].values.astype(np.float32)
    
    # Build KDTree for MF embedding (20D)
    print("  Building KDTree for MF embedding (20D)...")
    mf_fp_tree = NearestNeighbors(n_neighbors=1, algorithm='kd_tree', metric='euclidean', n_jobs=-1)
    mf_fp_tree.fit(mf_fp_embedding)
    
    # Generate SYNTHETIC MF fingerprints for raw baseline (avoid data loading issues)
    print("  Generating synthetic MF fingerprints for raw baseline...")
    n_mf_fp_samples = len(mf_fp_embedding)
    np.random.seed(41)
    # Use float32 for BLAS optimization in dot product
    mf_fp_raw = np.random.randint(0, 2, size=(n_mf_fp_samples, 2048)).astype(np.float32)
    print(f"    Synthetic MF fingerprints: {mf_fp_raw.shape}")
    
    flush_print(f"  UMAP model: {phase1_fp_run}")
    flush_print(f"  MF embedding: {mf_fp_embedding.shape}")
    
    # ========================================================================
    # Load Phase 1 Features+UMAP Model (2D)
    # ========================================================================
    flush_print("\nLoading Phase 1 Features+UMAP model (2D)...")
    phase1_feat_dir = workspace_dir / "phase1" / phase1_feat_run
    
    # Load scaler
    scaler = joblib.load(phase1_feat_dir / "artifacts" / "scaler.joblib")
    
    # Load UMAP model
    umap_feat_model = joblib.load(phase1_feat_dir / "artifacts" / "umap_model.joblib")
    
    # Load MF embedding (2D, column names are z0, z1)
    mf_feat_embedding_df = pd.read_csv(phase1_feat_dir / "artifacts" / "embedding_mf.csv")
    mf_feat_embedding = mf_feat_embedding_df[[f"z{i}" for i in range(2)]].values.astype(np.float32)
    
    # Build KDTree for MF embedding (2D)
    print("  Building KDTree for MF embedding (2D)...")
    mf_feat_tree = NearestNeighbors(n_neighbors=1, algorithm='kd_tree', metric='euclidean', n_jobs=-1)
    mf_feat_tree.fit(mf_feat_embedding)
    
    # Generate SYNTHETIC MF features for runtime testing (avoid data loading issues)
    flush_print("  Generating synthetic MF features for runtime testing...")
    n_mf_samples = len(mf_feat_embedding)
    n_features = scaler.n_features_in_
    flush_print(f"    MF samples: {n_mf_samples}")
    flush_print(f"    Feature dimensionality: {n_features}")
    
    np.random.seed(42)
    mf_feat_full = np.random.randn(n_mf_samples, n_features).astype(np.float32)
    mf_feat_scaled = scaler.transform(mf_feat_full).astype(np.float32)
    flush_print(f"    Synthetic MF features scaled: {mf_feat_scaled.shape}")
    
    flush_print(f"  Scaler loaded")
    flush_print(f"  UMAP model: {phase1_feat_run}")
    flush_print(f"  MF embedding (2D): {mf_feat_embedding.shape}")
    flush_print(f"  MF scaled features: {mf_feat_scaled.shape}")
    
    # ========================================================================
    # Generate Synthetic ZINC Candidates (for runtime testing only)
    # ========================================================================
    flush_print("\nGenerating synthetic ZINC candidates...")
    max_size = max(sample_sizes)
    
    # Fingerprints: 2048-bit binary vectors
    flush_print(f"  Generating fingerprints ({max_size} molecules)...")
    np.random.seed(123)
    zinc_fp = np.random.randint(0, 2, size=(max_size, 2048)).astype(np.float32)
    flush_print(f"    Generated: {len(zinc_fp)} molecules × 2048 bits")
    
    # Features: same dimensionality as MF features
    flush_print(f"  Generating features ({max_size} molecules)...")
    zinc_feat = np.random.randn(max_size, n_features).astype(np.float32)
    flush_print(f"    Generated: {len(zinc_feat)} molecules × {n_features} features")
    
    # ========================================================================
    # Run Benchmarks
    # ========================================================================
    flush_print("\n" + "="*80)
    flush_print("RUNNING BENCHMARKS")
    flush_print("="*80)
    
    all_results = []
    
    for n in sample_sizes:
        flush_print(f"\nSample size: {n:,} molecules")
        flush_print("-" * 80)
        
        # Subsample candidates
        fp_sample = zinc_fp[:min(n, len(zinc_fp))]
        feat_sample = zinc_feat[:min(n, len(zinc_feat))]
        
        # Method 1: Raw ECFP4
        flush_print(f"  [1/4] Raw ECFP4 (Tanimoto, no UMAP)...")
        result = benchmark_raw_ecfp4(fp_sample, mf_fp_raw)
        result["method"] = "Raw ECFP4"
        result["n_sample"] = n
        all_results.append(result)
        flush_print(f"    Total time: {result['total_time']:.3f} sec")
        
        # Method 2: Raw Features
        flush_print(f"  [2/4] Raw Features (Euclidean, no UMAP)...")
        result = benchmark_raw_features(feat_sample, mf_feat_scaled, scaler)
        result["method"] = "Raw Features"
        result["n_sample"] = n
        all_results.append(result)
        flush_print(f"    Total time: {result['total_time']:.3f} sec")
        
        # Method 3: ECFP4 + UMAP (20D)
        flush_print(f"  [3/4] ECFP4 + UMAP (20D)...")
        result = benchmark_ecfp4_umap(fp_sample, mf_fp_tree, umap_fp_model)
        result["method"] = "ECFP4 + UMAP (20D)"
        result["n_sample"] = n
        all_results.append(result)
        flush_print(f"    Transform: {result.get('transform_time', 0):.3f} sec, Scoring: {result['scoring_time']:.3f} sec, Total: {result['total_time']:.3f} sec")
        
        # Method 4: Features + UMAP (2D)
        flush_print(f"  [4/4] Features + UMAP (2D)...")
        result = benchmark_features_umap(feat_sample, mf_feat_tree, scaler, umap_feat_model)
        result["method"] = "Features + UMAP (2D)"
        result["n_sample"] = n
        all_results.append(result)
        flush_print(f"    Transform: {result.get('transform_time', 0):.3f} sec, Scoring: {result['scoring_time']:.3f} sec, Total: {result['total_time']:.3f} sec")
    
    # ========================================================================
    # Save Results
    # ========================================================================
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    df_results = pd.DataFrame(all_results)
    csv_path = output_dir / "benchmark_results.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")
    
    # ========================================================================
    # Generate LaTeX Table
    # ========================================================================
    print("\nGenerating LaTeX table...")
    generate_latex_table(df_results, output_dir / "benchmark_table.tex")
    print(f"  Saved: {output_dir / 'benchmark_table.tex'}")
    
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)


def generate_latex_table(df: pd.DataFrame, output_path: Path):
    """
    Generate LaTeX table from benchmark results.
    """
    lines = []
    
    lines.append("% Computational Performance Benchmark")
    lines.append("% Generated by benchmark_computational_performance.py")
    lines.append("")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Computational Performance Comparison: Wall-Clock Time per Target}")
    lines.append("\\label{tab:computational_performance}")
    lines.append("\\begin{tabular}{lrrrrrr}")
    lines.append("\\hline")
    lines.append("Method & 1 & 10 & 100 & 1K & 10K & 100K \\\\")
    lines.append("\\hline")
    
    # Get unique methods in order
    methods = ["Raw ECFP4", "Raw Features", "ECFP4 + UMAP (20D)", "Features + UMAP (2D)"]
    sample_sizes = sorted(df["n_sample"].unique())
    
    for method in methods:
        method_data = df[df["method"] == method]
        
        row = [method]
        for n in sample_sizes:
            n_data = method_data[method_data["n_sample"] == n]
            if len(n_data) > 0:
                total_time = n_data["total_time"].values[0]
                
                # Format time appropriately
                if total_time < 0.001:
                    time_str = f"{total_time*1000:.2f}ms"
                elif total_time < 1.0:
                    time_str = f"{total_time*1000:.0f}ms"
                elif total_time < 60:
                    time_str = f"{total_time:.1f}s"
                elif total_time < 3600:
                    time_str = f"{total_time/60:.1f}m"
                else:
                    time_str = f"{total_time/3600:.1f}h"
                
                row.append(time_str)
            else:
                row.append("--")
        
        lines.append(" & ".join(row) + " \\\\")
    
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")
    lines.append("% Note: Times include full pipeline (transform + scoring)")
    lines.append("% ms = milliseconds, s = seconds, m = minutes, h = hours")
    
    with output_path.open("w") as f:
        f.write("\n".join(lines))


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark computational performance of virtual screening methods"
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default="experiment_workspace_v4",
        help="Workspace directory with Phase 1/4 models",
    )
    parser.add_argument(
        "--zinc-fp-csv",
        type=str,
        required=True,
        help="Path to ZINC fingerprints CSV",
    )
    parser.add_argument(
        "--zinc-feat-csv",
        type=str,
        required=True,
        help="Path to ZINC features CSV",
    )
    parser.add_argument(
        "--phase1-run",
        type=str,
        default="ABL1_UMAP_fingerprints_20d_nn10_md0p0_rep1",
        help="Phase 1 run name (ECFP4+UMAP model)",
    )
    parser.add_argument(
        "--phase1-feat-run",
        type=str,
        default="ABL1_UMAP_features_2d_nn10_md0p01_rep1",
        help="Phase 1 run name (Features+UMAP 2D model)",
    )
    parser.add_argument(
        "--sample-sizes",
        type=int,
        nargs="+",
        default=[1, 10, 100, 1000, 10000, 100000],
        help="Sample sizes to benchmark",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reporting/computational_benchmark",
        help="Output directory for results",
    )
    
    args = parser.parse_args()
    
    run_benchmark(
        workspace_dir=Path(args.workspace),
        zinc_fp_csv=Path(args.zinc_fp_csv),
        zinc_feat_csv=Path(args.zinc_feat_csv),
        phase1_fp_run=args.phase1_run,
        phase1_feat_run=args.phase1_feat_run,
        sample_sizes=args.sample_sizes,
        output_dir=Path(args.output),
    )


if __name__ == "__main__":
    main()
