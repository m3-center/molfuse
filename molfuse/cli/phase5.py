#!/usr/bin/env python3
"""
Phase 5: Validation & Baseline Experiments

Three critical controls to address reviewer concerns:
    1. Database Bias Negative Control: Non-kinase actives vs kinase model
    2. Raw Descriptor Baseline: 1-NN in high-D space (no UMAP)
    3. Tanimoto Baseline: ECFP4 fingerprint similarity

Research Questions:
    - Does MolFuSE score any ChEMBL molecule highly (database bias)?
    - Is dimensionality reduction necessary (raw baseline)?
    - Does MolFuSE beat industry-standard fingerprint similarity (Tanimoto)?

Usage:
    python -m molfuse.cli.phase5 \\
        --config configs/phase5_grid/phase5_negative_control_rep1.json \\
        --workspace experiment_workspace_v4
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from sklearn.metrics.pairwise import cosine_similarity

from molfuse.data.prep import (
    select_feature_columns,
    remove_zero_variance_features,
    remove_rows_with_infinity,
    fit_scaler_on_mf_zinc,
)
from molfuse.dr.umap_ import fit_umap
from molfuse.metrics.metrics import ef_at_k_percent, roc_auc, pr_auc
from molfuse.scoring.nn import nn_min_distance_scores
from molfuse.io.paths import make_run_dirs


# ============================================================================
# Logger Setup
# ============================================================================

def setup_logger(log_path: Path) -> logging.Logger:
    """Setup logger with file and console handlers."""
    logger = logging.getLogger("phase5")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(ch)
    
    return logger


# ============================================================================
# Tanimoto Similarity Functions
# ============================================================================

def compute_ecfp4_fingerprints(smiles_list: List[str], radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """
    Compute ECFP4 fingerprints for a list of SMILES.
    
    Returns:
        Binary fingerprint matrix (n_molecules, n_bits)
    """
    fps = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # Invalid SMILES - use zero vector
            fps.append(np.zeros(n_bits, dtype=np.uint8))
        else:
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            fps.append(np.array(fp, dtype=np.uint8))
    return np.vstack(fps)


def tanimoto_similarity_max(query_fps: np.ndarray, ref_fps: np.ndarray) -> np.ndarray:
    """
    Compute max Tanimoto similarity for each query against reference set.
    
    Tanimoto = |A ∩ B| / |A ∪ B| = (A · B) / (|A| + |B| - A · B)
    
    Args:
        query_fps: Binary fingerprints (n_query, n_bits)
        ref_fps: Binary fingerprints (n_ref, n_bits)
    
    Returns:
        Max Tanimoto similarity for each query (n_query,)
    """
    # Compute dot product (intersection)
    intersect = query_fps @ ref_fps.T  # (n_query, n_ref)
    
    # Compute union
    query_popcount = query_fps.sum(axis=1, keepdims=True)  # (n_query, 1)
    ref_popcount = ref_fps.sum(axis=1, keepdims=True).T  # (1, n_ref)
    union = query_popcount + ref_popcount - intersect
    
    # Tanimoto = intersection / union
    tanimoto = intersect / (union + 1e-10)  # Add epsilon to avoid division by zero
    
    # Return max similarity for each query
    return tanimoto.max(axis=1)


# ============================================================================
# Data Loading
# ============================================================================

def load_and_filter_mf(csv_path: Path, target_accession: str, affinity_cutoff: float, logger: logging.Logger) -> pd.DataFrame:
    """Load MF cloud and filter by target and affinity cutoff."""
    df = pd.read_csv(csv_path, low_memory=False)
    logger.info(f"Loaded MF: {len(df):,} rows")
    
    # Remove target actives
    df = df[df["accession"] != target_accession].copy()
    logger.info(f"After removing target actives: {len(df):,} rows")
    
    # Apply affinity cutoff
    if "Standard Value (nM)" in df.columns:
        mask_cut = pd.to_numeric(df["Standard Value (nM)"], errors="coerce") <= affinity_cutoff
        df = df[mask_cut].copy()
        logger.info(f"After affinity cutoff (≤{affinity_cutoff} nM): {len(df):,} rows")
    
    return df


# ============================================================================
# Main Phase 5 Pipeline
# ============================================================================

def run_phase5(config_path: Path, workspace_dir: Path) -> None:
    """Execute Phase 5: Validation & Baseline Experiments."""
    
    # Load config
    with config_path.open("r") as f:
        cfg = json.load(f)
    
    run_name = cfg["run_name"]
    experiment_type = cfg["experiment_type"]
    target = cfg["target"]
    random_seed = cfg.get("random_seed", 42)
    
    # Create run directories
    phase5_run_name = cfg.get("phase5_run_name", "validation")
    ws = make_run_dirs(workspace_dir, phase="phase5", run_name=f"{phase5_run_name}/{run_name}")
    run_dir = ws["base"]
    logs_dir = ws["logs"]
    metrics_dir = ws["metrics"]
    artifacts_dir = ws["artifacts"]
    
    # Setup logger
    log_path = logs_dir / "run.log"
    logger = setup_logger(log_path)
    
    logger.info("="*80)
    logger.info("PHASE 5: VALIDATION & BASELINE EXPERIMENTS")
    logger.info("="*80)
    logger.info(f"Run: {run_name}")
    logger.info(f"Experiment: {experiment_type}")
    logger.info(f"Description: {cfg['description']}")
    logger.info(f"Target: {target}")
    logger.info(f"Random seed: {random_seed}")
    logger.info(f"Workspace: {run_dir}")
    logger.info("="*80)
    
    start_time = time.time()
    
    # ========================================================================
    # Load Datasets
    # ========================================================================
    logger.info("\n[1/6] Loading datasets...")
    
    # TODO: Implement actual data loading based on experiment type
    # This is a scaffold - you'll need to integrate with actual data paths
    
    logger.info(f"Experiment type: {experiment_type}")
    
    if experiment_type == "negative_control":
        logger.info("Database Bias Negative Control")
        logger.info("  - Load kinase actives (ABL1)")
        logger.info("  - Load non-kinase actives (GPCR control)")
        logger.info("  - Load ZINC decoys")
        logger.info("  - Project all through ABL1 UMAP model")
        logger.info("  - Score against kinase reference set")
        logger.info("  - Expected: Non-kinase should overlap with decoys")
        
    elif experiment_type == "raw_descriptors":
        logger.info("Raw Descriptor Baseline (No UMAP)")
        logger.info("  - Load kinase actives (ABL1)")
        logger.info("  - Load ZINC decoys")
        logger.info("  - Apply StandardScaler only (no UMAP)")
        logger.info("  - 1-NN scoring in high-D space (1,477 dimensions)")
        logger.info("  - Expected: EF@1% << 53% (UMAP baseline)")
        
    elif experiment_type == "tanimoto":
        logger.info("Tanimoto Baseline")
        logger.info("  - Load kinase actives (ABL1)")
        logger.info("  - Load ZINC decoys")
        logger.info("  - Generate ECFP4 fingerprints")
        logger.info("  - Max Tanimoto similarity vs reference set")
        logger.info("  - Expected: EF@1% ~15-25%")
    
    # Placeholder metrics (implement actual scoring logic)
    ef1 = 0.0
    roc = 0.0
    pr = 0.0
    
    elapsed = time.time() - start_time
    
    # ========================================================================
    # Save Summary
    # ========================================================================
    logger.info("\n[6/6] Saving summary...")
    
    summary = {
        "run_name": run_name,
        "experiment_type": experiment_type,
        "description": cfg["description"],
        "target": target,
        "random_seed": random_seed,
        "ef_1%": float(ef1),
        "roc_auc": float(roc),
        "pr_auc": float(pr),
        "elapsed_time_s": float(elapsed),
        "config": cfg,
    }
    
    with (metrics_dir / "metrics.json").open("w") as f:
        json.dump(summary, f, indent=2)
    
    with (logs_dir / "phase5_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Saved metrics and summary")
    logger.info("="*80)
    logger.info(f"PHASE 5 COMPLETED in {elapsed:.1f}s")
    logger.info(f"Experiment: {experiment_type} | EF@1%: {ef1:.2f} | ROC-AUC: {roc:.4f}")
    logger.info("="*80)


# ============================================================================
# CLI Entrypoint
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 5: Validation & Baseline Experiments")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to Phase 5 config JSON")
    parser.add_argument("--workspace", type=str, default="experiment_workspace_v4",
                       help="Workspace directory")
    
    args = parser.parse_args()
    
    config_path = Path(args.config)
    workspace_dir = Path(args.workspace)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    run_phase5(config_path, workspace_dir)


if __name__ == "__main__":
    main()
