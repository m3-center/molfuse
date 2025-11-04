#!/usr/bin/env python3
"""
Feature Comparison: 40-feature vs Full 2D vs Full 2D+3D Mordred Descriptors

This script evaluates whether using more Mordred descriptors improves virtual screening
performance by comparing three feature representations using the exact same evaluation
protocol as Phase 1.

The three representations:
1. Current 40-feature subset (baseline) - from datasets/molecular_function_features_fingerprints/
2. Full 2D Mordred (1613 features) - from output_recalculated_full_datasets/datasets_2d_all/
3. Full 2D+3D Mordred (1826 features) - from output_recalculated_full_datasets/datasets_2d3d_all/

Evaluation protocol (mimicking Phase 1):
- Select ONE KW file, split by accession (target actives vs MF cloud)
- Sample MF cloud and ZINC decoys
- Remove overlaps sequentially (actives → MF → ZINC)
- For each representation: feature selection, cleaning, scaling, UMAP, 1-NN scoring
- Compute metrics: EF@1%, ROC-AUC, PR-AUC, Spearman ρ
- Generate visualizations

Key differences from original feature_comparison.py:
- NO descriptor computation (loads pre-computed features)
- Loads from affinity CSVs to get potency values for ranking
- Looks up features in three separate directories
- Much faster execution (~minutes vs hours)

Performance optimizations:
- Selective dtype specification: metadata as str, numeric features inferred (~90% memory savings vs all-string)
- PyArrow CSV engine: 3-5x faster parsing (with graceful fallback)
- Automatic Parquet conversion: 10-100x faster subsequent loads
- Memory footprint: ~30GB for 10GB CSV (vs 300GB with dtype=str)

Author: Feature comparison script v2 (pre-computed features workflow)
"""

from __future__ import annotations
import argparse
import json
import math
import os
import gc
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Set
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ML/DR imports
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
import umap

# Viz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


# Current 40-feature names (baseline subset)
CURRENT_FEATURE_NAMES_40 = [
    "DipoleMoment", "ABC", "nAcid", "nBase", "nAromAtom", "nAtom", "nH", "nC", "nN", "nO", "nS", "nP", "nX",
    "nBonds", "nBondsO", "nBondsS", "nBondsD", "nBondsT", "nBondsA", "nBondsM", "nBondsKS", "nBondsKD",
    "EState_VSA7", "nHBAcc", "nHBDon", "Lipinski", "apol", "bpol", "nRing", "n3Ring", "n4Ring", "n5Ring",
    "n6Ring", "n7Ring", "n8Ring", "nRot", "Diameter", "TopoShapeIndex", "Vabc", "MW",
]


def load_kw_with_features(
    feature_csv: Path,
    target_accession: str,
    n_mf: Optional[int] = None,
    n_target: Optional[int] = None,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load KW feature file and split into target actives and MF cloud (Phase 1 style).
    
    The feature CSV already contains both affinity data AND features (all in one file).
    Automatically uses Parquet format if available (10-100x faster loading).
    
    Args:
        feature_csv: Path to feature CSV (has SMILES + Standard Value (nM) + accession + descriptors)
        target_accession: Accession to identify target actives
        n_mf: Number of MF molecules to sample (None = all)
        n_target: Number of target molecules to sample (None = all)
        seed: Random seed
        
    Returns:
        df_actives: DataFrame with SMILES + features + affinity (target molecules)
        df_mf: DataFrame with SMILES + features + affinity (MF cloud)
    """
    # Check for Parquet version and convert if needed
    parquet_path = feature_csv.with_suffix('.parquet')
    
    # NOTE: If you encounter dtype errors, delete existing .parquet files to force re-conversion
    if parquet_path.exists():
        # Load from Parquet (10-100x faster)
        print(f"  Loading from Parquet: {parquet_path.name}")
        df = pd.read_parquet(parquet_path)
    else:
        # Load from CSV with optimizations: selective dtype for metadata only (saves ~90% memory vs dtype=str)
        print(f"  Loading from CSV: {feature_csv.name}")
        
        # Only force string type for known text/metadata columns, let numeric columns be inferred efficiently
        dtype_dict = {
            'Compound ChEMBL ID': str,
            'SMILES': str,
            'Target ChEMBL ID': str,
            'Target Name': str,
            'Activity Type': str,
            'Standard Value (nM)': float,  # Explicitly numeric (inference can fail for unusual column names)
            'target_chembl_id': str,
            'accession': str,
            # All other numeric feature columns will be inferred (much more memory-efficient than str)
        }
        
        try:
            # Try PyArrow engine for 3-5x faster parsing
            df = pd.read_csv(feature_csv, dtype=dtype_dict, engine='pyarrow')
        except (ImportError, Exception):
            # Fallback to default engine if PyArrow not available
            df = pd.read_csv(feature_csv, dtype=dtype_dict, low_memory=False)
        
        # Convert to Parquet for future runs (one-time cost)
        try:
            print(f"  Converting to Parquet for faster future loads...")
            n_rows_csv = len(df)
            df.to_parquet(parquet_path, compression='snappy', engine='pyarrow', index=False)
            
            # Validate: check row count matches
            df_check = pd.read_parquet(parquet_path)
            n_rows_parquet = len(df_check)
            if n_rows_csv != n_rows_parquet:
                print(f"  WARNING: Row count mismatch! CSV={n_rows_csv}, Parquet={n_rows_parquet}")
                # Delete corrupted Parquet file
                parquet_path.unlink()
                print(f"  Deleted corrupted Parquet file")
            else:
                print(f"  ✓ Saved: {parquet_path.name} ({n_rows_parquet} rows)")
            del df_check
        except Exception as e:
            print(f"  Warning: Could not save Parquet file: {e}")
    
    # Check for required columns
    if 'SMILES' not in df.columns:
        raise ValueError(f"SMILES column not found in {feature_csv}")
    if 'accession' not in df.columns:
        raise ValueError(f"accession column not found in {feature_csv}")
    
    # Ensure Standard Value is numeric (defensive: dtype specification may not always work)
    if 'Standard Value (nM)' in df.columns:
        df['Standard Value (nM)'] = pd.to_numeric(df['Standard Value (nM)'], errors='coerce')
    
    # Convert all non-metadata columns to numeric immediately after loading
    # (prevents NaN explosion later when actives are processed)
    metadata_cols = {'Compound ChEMBL ID', 'SMILES', 'Target ChEMBL ID', 'Target Name', 
                     'Activity Type', 'target_chembl_id', 'accession'}
    for col in df.columns:
        if col not in metadata_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Split by accession (Phase 1 style)
    mask_target = (df['accession'] == target_accession)
    df_actives = df[mask_target].copy()
    df_mf = df[~mask_target].copy()
    
    # Deduplicate by SMILES (median affinity if duplicates)
    if 'Standard Value (nM)' in df_actives.columns:
        # Build aggregation dict for all columns (affinity already numeric from selective dtype loading)
        agg_dict = {'Standard Value (nM)': 'median'}
        for col in df_actives.columns:
            if col not in ['SMILES', 'Standard Value (nM)']:
                agg_dict[col] = 'first'
        df_actives = df_actives.groupby('SMILES', as_index=False).agg(agg_dict)
    else:
        df_actives = df_actives.drop_duplicates(subset=['SMILES'], keep='first')
    
    if 'Standard Value (nM)' in df_mf.columns:
        # Build aggregation dict (affinity already numeric from selective dtype loading)
        agg_dict = {'Standard Value (nM)': 'median'}
        for col in df_mf.columns:
            if col not in ['SMILES', 'Standard Value (nM)']:
                agg_dict[col] = 'first'
        df_mf = df_mf.groupby('SMILES', as_index=False).agg(agg_dict)
    else:
        df_mf = df_mf.drop_duplicates(subset=['SMILES'], keep='first')
    
    # Sample if requested
    if n_target and len(df_actives) > n_target:
        df_actives = df_actives.sample(n=n_target, random_state=seed)
    
    if n_mf and len(df_mf) > n_mf:
        df_mf = df_mf.sample(n=n_mf, random_state=seed)
    
    print(f"  Loaded actives: {len(df_actives)}")
    print(f"  Loaded MF: {len(df_mf)}")
    
    return df_actives, df_mf


def load_zinc_with_features(
    zinc_csv: Path,
    feature_csv: Path,
    n_zinc: Optional[int] = None,
    seed: int = 42
) -> pd.DataFrame:
    """Load ZINC molecules with features.
    
    Automatically uses Parquet format if available (10-100x faster loading).
    
    Args:
        zinc_csv: Path to ZINC CSV (original, for SMILES list)
        feature_csv: Path to ZINC feature CSV
        n_zinc: Number of ZINC molecules to sample (None = all)
        seed: Random seed
        
    Returns:
        df_zinc: DataFrame with SMILES + features
    """
    # Load original ZINC (just to get SMILES list for sampling)
    # ZINC CSV likely has SMILES + metadata columns
    dtype_dict = {'SMILES': str}
    
    try:
        # Try PyArrow engine for faster parsing
        df_zinc_orig = pd.read_csv(zinc_csv, dtype=dtype_dict, engine='pyarrow')
    except (ImportError, Exception):
        # Fallback to default engine
        df_zinc_orig = pd.read_csv(zinc_csv, dtype=dtype_dict, low_memory=False)
    
    if 'SMILES' not in df_zinc_orig.columns:
        raise ValueError(f"SMILES column not found in {zinc_csv}")
    
    # Deduplicate and sample
    df_zinc_orig = df_zinc_orig.drop_duplicates(subset=['SMILES'], keep='first')
    if n_zinc and len(df_zinc_orig) > n_zinc:
        df_zinc_orig = df_zinc_orig.sample(n=n_zinc, random_state=seed)
    
    # Load features with Parquet optimization
    parquet_path = feature_csv.with_suffix('.parquet')
    
    if parquet_path.exists():
        print(f"  Loading ZINC features from Parquet: {parquet_path.name}")
        df_feat = pd.read_parquet(parquet_path)
    else:
        print(f"  Loading ZINC features from CSV: {feature_csv.name}")
        
        # Only force string type for SMILES, let numeric columns be inferred efficiently
        dtype_dict = {'SMILES': str}
        
        try:
            # Try PyArrow engine for faster parsing
            df_feat = pd.read_csv(feature_csv, dtype=dtype_dict, engine='pyarrow')
        except (ImportError, Exception):
            # Fallback to default engine
            df_feat = pd.read_csv(feature_csv, dtype=dtype_dict, low_memory=False)
        
        # Convert to Parquet for future runs
        try:
            print(f"  Converting ZINC to Parquet for faster future loads...")
            n_rows_csv = len(df_feat)
            df_feat.to_parquet(parquet_path, compression='snappy', engine='pyarrow', index=False)
            
            # Validate: check row count matches
            df_check = pd.read_parquet(parquet_path)
            n_rows_parquet = len(df_check)
            if n_rows_csv != n_rows_parquet:
                print(f"  WARNING: Row count mismatch! CSV={n_rows_csv}, Parquet={n_rows_parquet}")
                # Delete corrupted Parquet file
                parquet_path.unlink()
                print(f"  Deleted corrupted Parquet file")
            else:
                print(f"  ✓ Saved: {parquet_path.name} ({n_rows_parquet} rows)")
            del df_check
        except Exception as e:
            print(f"  Warning: Could not save Parquet file: {e}")
    
    if 'SMILES' not in df_feat.columns:
        raise ValueError(f"SMILES column not found in {feature_csv}")
    
    # Convert all non-SMILES columns to numeric immediately after loading
    for col in df_feat.columns:
        if col != 'SMILES':
            df_feat[col] = pd.to_numeric(df_feat[col], errors='coerce')
    
    # Merge
    df_zinc = df_zinc_orig[['SMILES']].merge(df_feat, on='SMILES', how='inner')
    
    print(f"  Loaded ZINC: {len(df_zinc_orig)} (requested) → {len(df_zinc)} (with features)")
    
    return df_zinc


def remove_overlaps_sequential(
    df_actives: pd.DataFrame,
    df_mf: pd.DataFrame,
    df_zinc: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Remove overlaps sequentially: actives → MF → ZINC (Phase 1 style).
    
    Args:
        df_actives: Actives DataFrame
        df_mf: MF DataFrame
        df_zinc: ZINC DataFrame
        
    Returns:
        Filtered versions of all three DataFrames
    """
    actives_smiles = set(df_actives['SMILES'].dropna().astype(str))
    
    # Remove actives from MF
    mf_before = len(df_mf)
    df_mf = df_mf[~df_mf['SMILES'].isin(actives_smiles)].copy()
    print(f"  Removed actives from MF: {mf_before} → {len(df_mf)}")
    
    # Remove actives from ZINC
    zinc_before = len(df_zinc)
    df_zinc = df_zinc[~df_zinc['SMILES'].isin(actives_smiles)].copy()
    print(f"  Removed actives from ZINC: {zinc_before} → {len(df_zinc)}")
    
    # Remove MF from ZINC
    mf_smiles = set(df_mf['SMILES'].dropna().astype(str))
    zinc_before = len(df_zinc)
    df_zinc = df_zinc[~df_zinc['SMILES'].isin(mf_smiles)].copy()
    print(f"  Removed MF from ZINC: {zinc_before} → {len(df_zinc)}")
    
    return df_actives, df_mf, df_zinc


def select_feature_columns(df: pd.DataFrame) -> List[str]:
    """Select numeric feature columns (exclude metadata).
    
    Args:
        df: DataFrame with features
        
    Returns:
        List of feature column names
    """
    # Exclude known metadata columns (and any with _aff or _feat suffixes from merges)
    exclude = {
        'SMILES', 'smiles', 'canonical_smiles',
        'Compound ChEMBL ID', 'Target ChEMBL ID', 'Target Name',
        'Activity Type', 'Standard Value (nM)', 'target_chembl_id', 'accession',
        'ZINC_ID', 'LABEL', 'MANUFACTURER', 'TRANCHE',
        'source', 'label'
    }
    
    # Select numeric columns not in exclude list and not ending with _aff or _feat
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols 
                    if c not in exclude 
                    and not c.endswith('_aff') 
                    and not c.endswith('_feat')]
    
    return feature_cols


def extract_40_feature_subset(df: pd.DataFrame) -> List[str]:
    """Extract 40-feature subset by column name.
    
    Args:
        df: DataFrame with features
        
    Returns:
        List of column names present in both CURRENT_FEATURE_NAMES_40 and df
    """
    return [c for c in CURRENT_FEATURE_NAMES_40 if c in df.columns]


def clean_scale_features(X: pd.DataFrame) -> Tuple[np.ndarray, List[str], SimpleImputer, StandardScaler]:
    """Clean and scale features: cast numeric, remove zero-variance, impute, scale.
    
    Args:
        X: DataFrame with features
        
    Returns:
        X_scaled: Scaled feature matrix (numpy array)
        kept_cols: List of column names kept after cleaning
    """
    # Cast to numeric
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors='coerce')
    
    # Remove zero-variance columns
    variances = X.var(axis=0)
    nonzero_cols = variances[variances > 1e-12].index.tolist()
    X = X[nonzero_cols].copy()
    
    if X.shape[1] == 0:
        raise ValueError("No features remain after zero-variance removal")
    
    # Impute NaNs (median)
    imputer = SimpleImputer(strategy='median')
    X_imp = imputer.fit_transform(X)

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    # Return trained imputer and scaler so callers can transform other sets (e.g., actives)
    return X_scaled, nonzero_cols, imputer, scaler


def compute_umap_2d(
    X: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = 'euclidean',
    seed: Optional[int] = None
) -> np.ndarray:
    """Compute UMAP 2D embedding.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        metric: Distance metric
        seed: Random seed (None for UMAP parallelism)
        
    Returns:
        embedding: 2D coordinates (n_samples, 2)
    """
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
        verbose=False
    )
    embedding = reducer.fit_transform(X)
    
    return embedding.astype(np.float32)


def nn_min_distance_scores(Z_mf: np.ndarray, Z_eval: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Score molecules by 1-NN min distance to MF cloud (Phase 1 exact algorithm).
    
    Uses sklearn.neighbors.NearestNeighbors with 'auto' algorithm selection
    (KDTree or BallTree) for optimized exact 1-NN search.
    
    Args:
        Z_mf: MF embeddings (n_mf, dim)
        Z_eval: Evaluation set embeddings (n_eval, dim)
        
    Returns:
        scores: -distance (higher is better)
        distances: Raw distances
    """
    from sklearn.neighbors import NearestNeighbors
    
    # Fit 1-NN on MF cloud (Phase 1 exact algorithm)
    nn = NearestNeighbors(n_neighbors=1, metric='euclidean', algorithm='auto', n_jobs=-1)
    nn.fit(Z_mf)
    
    # Query distances for evaluation set
    distances, _ = nn.kneighbors(Z_eval, return_distance=True)
    distances = distances.reshape(-1)
    
    # Score = -distance (higher score = closer to MF)
    scores = -distances
    
    return scores, distances


def ef_at_percent(scores: np.ndarray, labels: np.ndarray, percent: float = 1.0) -> float:
    """Compute enrichment factor at top percent.
    
    Args:
        scores: Scores (higher is better)
        labels: Binary labels (1 = active, 0 = inactive)
        percent: Percentage of top molecules to consider
        
    Returns:
        EF@percent
    """
    N = len(scores)
    if N == 0:
        return float('nan')
    
    A = int(labels.sum())
    if A == 0:
        return float('nan')
    
    Nx = max(1, math.ceil((percent / 100.0) * N))
    order = np.argsort(-scores)  # descending
    top_idx = order[:Nx]
    TPx = int(labels[top_idx].sum())
    
    return (TPx / Nx) / (A / N)


def to_pactivity_from_nM(series: pd.Series) -> np.ndarray:
    """Convert nM affinity to pActivity = 9 - log10(nM)."""
    s = pd.to_numeric(series, errors='coerce')
    return (9.0 - np.log10(s)).values


def plot_umap_scatter(
    embedding: np.ndarray,
    labels: np.ndarray,
    title: str,
    output_path: Path
) -> None:
    """Plot UMAP scatter colored by source."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    palette = {'target': 'red', 'mf_cloud': 'blue', 'zinc': 'gray'}
    
    for source in ['zinc', 'mf_cloud', 'target']:  # Plot order (background to foreground)
        mask = (labels == source)
        if mask.any():
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                label=source,
                c=palette[source],
                s=10,
                alpha=0.6,
                rasterized=True
            )
    
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title(title)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    gc.collect()


def procrustes_align(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Align target to source via Procrustes (orthogonal rotation + scaling)."""
    if source.shape != target.shape:
        raise ValueError("Embeddings must have same shape")
    
    # Center
    Xc = source - source.mean(axis=0, keepdims=True)
    Yc = target - target.mean(axis=0, keepdims=True)
    
    # Scaling
    denom = float((Yc ** 2).sum())
    if denom == 0.0:
        return target.copy()
    
    # Orthogonal Procrustes
    U, S, Vt = np.linalg.svd(Yc.T @ Xc, full_matrices=False)
    R = U @ Vt
    s = float(S.sum()) / denom
    
    aligned = s * (Yc @ R) + source.mean(axis=0, keepdims=True)
    
    return aligned


def plot_movement(
    emb_from: np.ndarray,
    emb_to: np.ndarray,
    labels: np.ndarray,
    title: str,
    output_path: Path
) -> None:
    """Plot movement between embeddings (Procrustes aligned)."""
    # Align
    emb_to_aligned = procrustes_align(emb_from, emb_to)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    palette = {'target': 'red', 'mf_cloud': 'blue', 'zinc': 'gray'}
    
    for source in ['zinc', 'mf_cloud', 'target']:
        mask = (labels == source)
        if mask.any():
            # Plot arrows
            for i in np.where(mask)[0]:
                ax.arrow(
                    emb_from[i, 0], emb_from[i, 1],
                    emb_to_aligned[i, 0] - emb_from[i, 0],
                    emb_to_aligned[i, 1] - emb_from[i, 1],
                    color=palette[source],
                    alpha=0.3,
                    head_width=0.05,
                    head_length=0.05,
                    length_includes_head=True,
                    rasterized=True
                )
    
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    gc.collect()


def run_comparison(args: argparse.Namespace) -> None:
    """Main comparison workflow."""
    
    print("\n" + "="*80)
    print("FEATURE COMPARISON: 40-feature vs Full 2D vs Full 2D+3D")
    print("="*80)
    
    # Setup paths
    base_dir = Path(args.base_dir).resolve()
    features_40_dir = base_dir / "datasets" / "molecular_function_features_fingerprints"
    features_2d_dir = Path(args.full_2d_dir).resolve()
    features_2d3d_dir = Path(args.full_2d3d_dir).resolve()
    
    kw_file = args.kw_file
    target_accession = args.target_accession
    
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nConfiguration:")
    print(f"  KW file: {kw_file}")
    print(f"  Target accession: {target_accession}")
    print(f"  n_mf: {args.n_mf if args.n_mf else 'all'}")
    print(f"  n_target: {args.n_target if args.n_target else 'all'}")
    print(f"  n_zinc: {args.n_zinc}")
    print(f"  Affinity cutoff: {args.affinity_cutoff_nM} nM")
    
    # ========================================================================
    # STEP 1: Load data from all three representations
    # ========================================================================
    
    print(f"\n{'='*80}")
    print("STEP 1: Loading data from three representations")
    print("="*80)
    
    # 1A) Load 40-feature representation
    print("\n[1/3] Loading 40-feature representation...")
    
    features_40_csv = features_40_dir / kw_file
    
    print(f"  Features file: {features_40_csv.name}")
    
    if not features_40_csv.exists():
        raise FileNotFoundError(f"40-feature CSV not found: {features_40_csv}")
    
    df_actives_40, df_mf_40 = load_kw_with_features(
        features_40_csv, target_accession,
        n_mf=args.n_mf, n_target=args.n_target, seed=args.seed
    )
    
    # Load ZINC
    zinc_orig_csv = base_dir / "datasets" / "zinc_data.csv"
    zinc_40_csv = features_40_dir / "zinc" / "zinc_acquirable_extracted_features.csv"
    
    if not zinc_40_csv.exists():
        raise FileNotFoundError(f"ZINC 40-feature CSV not found: {zinc_40_csv}")
    
    df_zinc_40 = load_zinc_with_features(zinc_orig_csv, zinc_40_csv, args.n_zinc, seed=args.seed)
    
    # 1B) Load full 2D representation
    print("\n[2/3] Loading full 2D representation...")
    features_2d_csv = features_2d_dir / kw_file
    zinc_2d_csv = features_2d_dir / "zinc" / "zinc_acquirable_extracted_features.csv"
    
    if not features_2d_csv.exists():
        raise FileNotFoundError(f"Full 2D CSV not found: {features_2d_csv}")
    if not zinc_2d_csv.exists():
        raise FileNotFoundError(f"ZINC full 2D CSV not found: {zinc_2d_csv}")
    
    df_actives_2d, df_mf_2d = load_kw_with_features(
        features_2d_csv, target_accession,
        n_mf=args.n_mf, n_target=args.n_target, seed=args.seed
    )
    df_zinc_2d = load_zinc_with_features(zinc_orig_csv, zinc_2d_csv, args.n_zinc, seed=args.seed)
    
    # 1C) Load full 2D+3D representation
    print("\n[3/3] Loading full 2D+3D representation...")
    features_2d3d_csv = features_2d3d_dir / kw_file
    zinc_2d3d_csv = features_2d3d_dir / "zinc" / "zinc_acquirable_extracted_features.csv"
    
    if not features_2d3d_csv.exists():
        raise FileNotFoundError(f"Full 2D+3D CSV not found: {features_2d3d_csv}")
    if not zinc_2d3d_csv.exists():
        raise FileNotFoundError(f"ZINC full 2D+3D CSV not found: {zinc_2d3d_csv}")
    
    df_actives_2d3d, df_mf_2d3d = load_kw_with_features(
        features_2d3d_csv, target_accession,
        n_mf=args.n_mf, n_target=args.n_target, seed=args.seed
    )
    df_zinc_2d3d = load_zinc_with_features(zinc_orig_csv, zinc_2d3d_csv, args.n_zinc, seed=args.seed)
    
    # ========================================================================
    # STEP 2: Remove overlaps and find intersection
    # ========================================================================
    
    print(f"\n{'='*80}")
    print("STEP 2: Removing overlaps and finding intersection")
    print("="*80)
    
    # Remove overlaps for each representation
    print("\n40-feature representation:")
    df_actives_40, df_mf_40, df_zinc_40 = remove_overlaps_sequential(df_actives_40, df_mf_40, df_zinc_40)
    
    print("\nFull 2D representation:")
    df_actives_2d, df_mf_2d, df_zinc_2d = remove_overlaps_sequential(df_actives_2d, df_mf_2d, df_zinc_2d)
    
    print("\nFull 2D+3D representation:")
    df_actives_2d3d, df_mf_2d3d, df_zinc_2d3d = remove_overlaps_sequential(df_actives_2d3d, df_mf_2d3d, df_zinc_2d3d)
    
    # Find intersection (molecules present in ALL three representations)
    print("\nFinding intersection across representations...")
    
    smiles_40 = (
        set(df_actives_40['SMILES']) | 
        set(df_mf_40['SMILES']) | 
        set(df_zinc_40['SMILES'])
    )
    smiles_2d = (
        set(df_actives_2d['SMILES']) | 
        set(df_mf_2d['SMILES']) | 
        set(df_zinc_2d['SMILES'])
    )
    smiles_2d3d = (
        set(df_actives_2d3d['SMILES']) | 
        set(df_mf_2d3d['SMILES']) | 
        set(df_zinc_2d3d['SMILES'])
    )
    
    valid_smiles = smiles_40 & smiles_2d & smiles_2d3d
    
    print(f"  Molecules in 40-feature: {len(smiles_40)}")
    print(f"  Molecules in full 2D: {len(smiles_2d)}")
    print(f"  Molecules in full 2D+3D: {len(smiles_2d3d)}")
    print(f"  Intersection (valid for comparison): {len(valid_smiles)}")
    print(f"  Molecules lost: {len(smiles_40) - len(valid_smiles)}")
    
    if len(valid_smiles) < 10:
        raise ValueError("Too few molecules in intersection (<10). Check data sources.")
    
    # Filter to valid SMILES
    def filter_to_valid(df: pd.DataFrame, valid: Set[str]) -> pd.DataFrame:
        return df[df['SMILES'].isin(valid)].copy()
    
    df_actives_40 = filter_to_valid(df_actives_40, valid_smiles)
    df_mf_40 = filter_to_valid(df_mf_40, valid_smiles)
    df_zinc_40 = filter_to_valid(df_zinc_40, valid_smiles)
    
    df_actives_2d = filter_to_valid(df_actives_2d, valid_smiles)
    df_mf_2d = filter_to_valid(df_mf_2d, valid_smiles)
    df_zinc_2d = filter_to_valid(df_zinc_2d, valid_smiles)
    
    df_actives_2d3d = filter_to_valid(df_actives_2d3d, valid_smiles)
    df_mf_2d3d = filter_to_valid(df_mf_2d3d, valid_smiles)
    df_zinc_2d3d = filter_to_valid(df_zinc_2d3d, valid_smiles)
    
    print(f"\nAfter filtering to intersection:")
    print(f"  Actives: {len(df_actives_40)}")
    print(f"  MF cloud: {len(df_mf_40)}")
    print(f"  ZINC: {len(df_zinc_40)}")
    
    # ========================================================================
    # STEP 3: Prepare features for each representation
    # ========================================================================
    
    print(f"\n{'='*80}")
    print("STEP 3: Preparing features")
    print("="*80)
    
    # Store results for each representation
    results = {}
    
    for rep_name, df_act, df_mf, df_z in [
        ('40feature', df_actives_40, df_mf_40, df_zinc_40),
        ('full2d', df_actives_2d, df_mf_2d, df_zinc_2d),
        ('full2d3d', df_actives_2d3d, df_mf_2d3d, df_zinc_2d3d),
    ]:
        print(f"\n[{rep_name}] Processing features...")
        
        # Select feature columns
        if rep_name == '40feature':
            feat_cols = extract_40_feature_subset(df_mf)
            print(f"  40-feature subset: {len(feat_cols)} features found")
        else:
            feat_cols = select_feature_columns(df_mf)
            print(f"  Selected {len(feat_cols)} numeric features")
        
        # Extract features
        X_act = df_act[feat_cols].copy()
        X_mf = df_mf[feat_cols].copy()
        X_z = df_z[feat_cols].copy()
        
        # Concatenate for training (MF + ZINC)
        X_train = pd.concat([X_mf, X_z], axis=0, ignore_index=True)
        
        print(f"  Cleaning and scaling...")
        X_train_scaled, kept_cols, imputer, scaler = clean_scale_features(X_train)

        # Split back
        n_mf = len(X_mf)
        X_mf_scaled = X_train_scaled[:n_mf]
        X_z_scaled = X_train_scaled[n_mf:]

        # Prepare actives: keep same columns, ensure numeric types, and drop NaNs (Phase 1 style)
        X_act_kept = X_act[kept_cols].copy()
        for c in X_act_kept.columns:
            X_act_kept[c] = pd.to_numeric(X_act_kept[c], errors='coerce')
        
        # Drop rows with any NaN (matching Phase 1 behavior)
        X_act_kept = X_act_kept.dropna(axis=0, how='any')
        
        # Update df_act to match the kept rows (for consistency in later steps)
        df_act = df_act.loc[X_act_kept.index].copy()

        # Use the same imputer and scaler fit on training (MF+ZINC) to transform actives.
        if len(X_act_kept) == 0:
            X_act_scaled = np.empty((0, len(kept_cols)), dtype=float)
        else:
            X_act_imp = imputer.transform(X_act_kept)
            X_act_scaled = scaler.transform(X_act_imp)
        
        print(f"  Features after cleaning: {len(kept_cols)}")
        print(f"  Train set: MF={len(X_mf_scaled)}, ZINC={len(X_z_scaled)}")
        print(f"  Actives: {len(X_act_scaled)}") # type: ignore
        
        # Store
        results[rep_name] = {
            'X_mf': X_mf_scaled,
            'X_zinc': X_z_scaled,
            'X_actives': X_act_scaled,
            'df_mf': df_mf,
            'df_zinc': df_z,
            'df_actives': df_act,
            'kept_cols': kept_cols,
        }
        
        gc.collect()
    
    # ========================================================================
    # STEP 4: UMAP embeddings and scoring
    # ========================================================================
    
    print(f"\n{'='*80}")
    print("STEP 4: Computing UMAP embeddings and scoring")
    print("="*80)
    
    all_metrics = {}
    
    for rep_name, res in results.items():
        print(f"\n[{rep_name}] Computing UMAP...")
        
        # Concatenate for UMAP fit
        X_train_concat = np.vstack([res['X_mf'], res['X_zinc']])
        
        # Fit UMAP on MF+ZINC (Phase 1 workflow)
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=args.umap_n_neighbors,
            min_dist=args.umap_min_dist,
            metric='euclidean',
            random_state=None,  # UMAP parallelism
            verbose=False
        )
        emb_train = reducer.fit_transform(X_train_concat)
        
        # Split embeddings
        n_mf = len(res['X_mf'])
        Z_mf = emb_train[:n_mf]
        Z_zinc = emb_train[n_mf:]
        
        # Transform actives using the fitted UMAP (Phase 1 workflow)
        Z_actives = reducer.transform(res['X_actives'])
        
        print(f"  UMAP embeddings: MF={Z_mf.shape}, ZINC={Z_zinc.shape}, Actives={Z_actives.shape}")
        
        # Apply affinity cutoff to MF for scoring
        if 'Standard Value (nM)' in res['df_mf'].columns:
            affinity_vals = pd.to_numeric(res['df_mf']['Standard Value (nM)'], errors='coerce')
            mask_cutoff = (affinity_vals <= args.affinity_cutoff_nM).values
            Z_mf_scoring = Z_mf[mask_cutoff]
            print(f"  MF after affinity cutoff ({args.affinity_cutoff_nM} nM): {len(Z_mf_scoring)}")
        else:
            Z_mf_scoring = Z_mf
            print(f"  No affinity column; using all MF for scoring: {len(Z_mf_scoring)}")
        
        if len(Z_mf_scoring) == 0:
            print(f"  WARNING: Empty MF after cutoff; using all MF")
            Z_mf_scoring = Z_mf
        
        # Build evaluation set (actives + ZINC)
        Z_eval = np.vstack([Z_actives, Z_zinc])
        labels = np.concatenate([
            np.ones(len(Z_actives), dtype=int),
            np.zeros(len(Z_zinc), dtype=int)
        ])
        
        # Score by 1-NN distance to MF
        print(f"  Scoring by 1-NN distance to MF...")
        scores, distances = nn_min_distance_scores(Z_mf_scoring, Z_eval)
        
        # Compute metrics
        ef1 = ef_at_percent(scores, labels, 1.0)
        ef5 = ef_at_percent(scores, labels, 5.0)
        ef10 = ef_at_percent(scores, labels, 10.0)
        
        try:
            roc = float(roc_auc_score(labels, scores))
        except:
            roc = float('nan')
        
        try:
            pr = float(average_precision_score(labels, scores))
        except:
            pr = float('nan')
        
        # Spearman on actives only
        if 'Standard Value (nM)' in res['df_actives'].columns:
            pact = to_pactivity_from_nM(res['df_actives']['Standard Value (nM)'])
            scores_act = scores[:len(Z_actives)]
            
            # Remove NaNs
            valid_mask = ~np.isnan(pact) & ~np.isnan(scores_act)
            if valid_mask.sum() > 1:
                rho, rho_p = spearmanr(pact[valid_mask], scores_act[valid_mask])
            else:
                rho, rho_p = float('nan'), float('nan')
        else:
            rho, rho_p = float('nan'), float('nan')
        
        metrics = {
            'ef_1%': float(ef1),
            'ef_5%': float(ef5),
            'ef_10%': float(ef10),
            'roc_auc': roc,
            'pr_auc': pr,
            'spearman_rho': float(rho),
            'spearman_p': float(rho_p),
            'n_features': res['X_mf'].shape[1],
            'n_actives': len(Z_actives),
            'n_mf': len(Z_mf),
            'n_zinc': len(Z_zinc),
            'n_mf_scoring': len(Z_mf_scoring),
        }
        
        print(f"  Metrics: EF@1%={ef1:.2f}, ROC-AUC={roc:.3f}, PR-AUC={pr:.3f}, Spearman ρ={rho:.3f}")
        
        all_metrics[rep_name] = metrics
        
        # Store embeddings for visualization
        results[rep_name]['Z_mf'] = Z_mf
        results[rep_name]['Z_zinc'] = Z_zinc
        results[rep_name]['Z_actives'] = Z_actives
        results[rep_name]['scores'] = scores
        results[rep_name]['labels'] = labels
        
        gc.collect()
    
    # ========================================================================
    # STEP 5: Save results and visualizations
    # ========================================================================
    
    print(f"\n{'='*80}")
    print("STEP 5: Saving results and visualizations")
    print("="*80)
    
    # Save metrics summary
    summary = {
        'config': {
            'kw_file': kw_file,
            'target_accession': target_accession,
            'n_mf': args.n_mf,
            'n_target': args.n_target,
            'n_zinc': args.n_zinc,
            'affinity_cutoff_nM': args.affinity_cutoff_nM,
            'seed': args.seed,
        },
        'metrics': all_metrics,
        'molecule_counts': {
            'valid_smiles_intersection': len(valid_smiles),
            'molecules_lost': len(smiles_40) - len(valid_smiles),
        }
    }
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Saved summary: {output_dir / 'summary.json'}")
    
    # Save metrics table
    metrics_df = pd.DataFrame(all_metrics).T
    metrics_df.to_csv(output_dir / 'metrics_comparison.csv')
    print(f"✓ Saved metrics: {output_dir / 'metrics_comparison.csv'}")
    
    # Generate visualizations
    print(f"\nGenerating visualizations...")
    
    for rep_name, res in results.items():
        # Concatenate embeddings for plotting
        emb_all = np.vstack([res['Z_actives'], res['Z_mf'], res['Z_zinc']])
        labels_all = np.array(
            ['target'] * len(res['Z_actives']) +
            ['mf_cloud'] * len(res['Z_mf']) +
            ['zinc'] * len(res['Z_zinc'])
        )
        
        # UMAP scatter
        plot_umap_scatter(
            emb_all, labels_all,
            f"UMAP: {rep_name}",
            output_dir / f"umap_{rep_name}.png"
        )
        print(f"  ✓ {rep_name}: umap_{rep_name}.png")
        
        # Save scored molecules
        scored_df = pd.DataFrame({
            'SMILES': list(res['df_actives']['SMILES']) + list(res['df_zinc']['SMILES']),
            'source': ['target'] * len(res['df_actives']) + ['zinc'] * len(res['df_zinc']),
            'score': res['scores'],
            'label': res['labels'],
        })
        scored_df = scored_df.sort_values('score', ascending=False)
        scored_df.to_csv(output_dir / f"scored_molecules_{rep_name}.csv", index=False)
        print(f"  ✓ {rep_name}: scored_molecules_{rep_name}.csv")
    
    # Movement plots (Procrustes alignment)
    print(f"\nGenerating movement plots...")
    
    # Get common labels for movement
    emb_40_all = np.vstack([results['40feature']['Z_actives'], results['40feature']['Z_mf'], results['40feature']['Z_zinc']])
    emb_2d_all = np.vstack([results['full2d']['Z_actives'], results['full2d']['Z_mf'], results['full2d']['Z_zinc']])
    emb_2d3d_all = np.vstack([results['full2d3d']['Z_actives'], results['full2d3d']['Z_mf'], results['full2d3d']['Z_zinc']])
    
    n_act = len(results['40feature']['Z_actives'])
    n_mf = len(results['40feature']['Z_mf'])
    n_zinc = len(results['40feature']['Z_zinc'])
    
    labels_all = np.array(['target'] * n_act + ['mf_cloud'] * n_mf + ['zinc'] * n_zinc)
    
    plot_movement(emb_40_all, emb_2d_all, labels_all,
                  "Movement: 40-feature → Full 2D",
                  output_dir / "movement_40_to_2d.png")
    print(f"  ✓ movement_40_to_2d.png")
    
    plot_movement(emb_2d_all, emb_2d3d_all, labels_all,
                  "Movement: Full 2D → Full 2D+3D",
                  output_dir / "movement_2d_to_2d3d.png")
    print(f"  ✓ movement_2d_to_2d3d.png")
    
    # ========================================================================
    # DONE
    # ========================================================================
    
    print(f"\n{'='*80}")
    print("COMPARISON COMPLETED")
    print("="*80)
    
    print(f"\nSummary:")
    print(f"  {'Representation':<20} {'EF@1%':>8} {'ROC-AUC':>9} {'PR-AUC':>8} {'Spearman ρ':>12} {'Features':>10}")
    print(f"  {'-'*80}")
    for rep in ['40feature', 'full2d', 'full2d3d']:
        m = all_metrics[rep]
        print(f"  {rep:<20} {m['ef_1%']:>8.2f} {m['roc_auc']:>9.3f} {m['pr_auc']:>8.3f} {m['spearman_rho']:>12.3f} {m['n_features']:>10}")
    
    print(f"\n✓ All results saved to: {output_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Feature Comparison v2: Pre-computed features (40 vs Full 2D vs Full 2D+3D)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Example usage:
        python feature_comparison_v2.py \\
            --kw_file KW-0808_Transferase_affinity_extracted_features.csv \\
            --target_accession P00519 \\
            --n_mf 600 \\
            --n_zinc 600 \\
            --output_dir output/comparison_KW0808_P00519

        This will compare three feature representations using the same molecules and
        evaluation protocol as Phase 1.
        """
    )
    
    # Required arguments
    p.add_argument("--kw_file", required=True,
                   help="KW file name for features (e.g., 'KW-0808_Transferase_affinity_extracted_features.csv'). "
                        "Script will automatically find corresponding affinity file (without '_extracted_features' suffix)")
    p.add_argument("--target_accession", required=True,
                   help="Target accession to split actives from MF (e.g., 'P00519')")
    
    # Sampling parameters
    p.add_argument("--n_mf", type=int, default=None,
                   help="Number of MF molecules to sample (default: all)")
    p.add_argument("--n_target", type=int, default=None,
                   help="Number of target molecules to sample (default: all)")
    p.add_argument("--n_zinc", type=int, default=None,
                   help="Number of ZINC molecules to sample (default: all; was 600 for testing)")
    
    # Data directories
    p.add_argument("--base_dir", default=".",
                   help="Base directory (repo root, default: current directory)")
    p.add_argument("--full_2d_dir",
                   default="/home/ahagg2s/UMMBAS_screening_experiments/output_recalculated_full_datasets/datasets_2d_all",
                   help="Directory with pre-computed full 2D features")
    p.add_argument("--full_2d3d_dir",
                   default="/home/ahagg2s/UMMBAS_screening_experiments/output_recalculated_full_datasets/datasets_2d3d_all",
                   help="Directory with pre-computed full 2D+3D features")
    
    # Evaluation parameters
    p.add_argument("--affinity_cutoff_nM", type=float, default=100.0,
                   help="Affinity cutoff for MF cloud scoring (default: 100 nM)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42)")
    
    # UMAP parameters
    p.add_argument("--umap_n_neighbors", type=int, default=15,
                   help="UMAP n_neighbors (default: 15)")
    p.add_argument("--umap_min_dist", type=float, default=0.1,
                   help="UMAP min_dist (default: 0.1)")
    
    # Output
    p.add_argument("--output_dir", default="tests/mordred_full_feature_eval/output_v2",
                   help="Output directory for results")
    
    # Test mode
    p.add_argument("--test", action="store_true",
                   help="Test mode: use minimal sample sizes for quick validation (n_mf=50, n_target=20, n_zinc=50, completes in <1 min)")
    
    args = p.parse_args()
    
    # Override sampling parameters in test mode
    if args.test:
        print("\n[TEST MODE] Using minimal sample sizes for quick validation")
        args.n_mf = 50
        args.n_target = 20
        args.n_zinc = 50
        if args.output_dir == "tests/mordred_full_feature_eval/output_v2":
            args.output_dir = "tests/mordred_full_feature_eval/output_v2_test"
    
    return args


if __name__ == "__main__":
    args = parse_args()
    run_comparison(args)
