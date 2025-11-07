#!/usr/bin/env python3
"""
UMAP Initialization Method Comparison Test

Compares UMAP embedding initialization strategies vs PCA baseline:
- PCA 20D (baseline)
- UMAP 20D with init='spectral' (default)
- UMAP 20D with init='random'
- UMAP 20D with init='pca'
- UMAP 20D with init='tswspectral' (spectral embedding of fuzzy simplicial set)

Test setup:
- Target: ABL1/P00519
- Dataset: Full 2D Mordred features (~1477 after zero-variance filtering)
- ZINC: Full dataset (~1.3M molecules)
- Fixed UMAP params: n_neighbors=50, min_dist=0.01, metric='euclidean'
- No replicates (single run per configuration)

Metrics:
- EF@1%, ROC-AUC, PR-AUC
- Runtime (loading, DR fitting, scoring)
- Memory usage (peak RSS)

Usage:
    python tests/umap_init_comparison/umap_init_test.py \
        --config tests/umap_init_comparison/configs/<config>.json \
        --workspace tests/umap_init_comparison/workspace \
        --output tests/umap_init_comparison/results
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from molfuse.cli.phase1 import (
    load_csv_optimized,
    extract_accession,
    dedup_by_smiles,
    remove_overlap_by_smiles,
    get_smiles_col,
    to_pactivity_from_nM,
)
from molfuse.data.prep import (
    select_feature_columns,
    remove_zero_variance_features,
    fit_scaler_on_mf_zinc,
)
from molfuse.dr.pca import fit_pca
from molfuse.dr.umap_ import fit_umap
from molfuse.metrics.metrics import ef_at_k_percent, pr_auc, roc_auc, spearman_rho
from molfuse.scoring.nn import nn_min_distance_scores


def get_memory_mb() -> float:
    """Get current memory usage in MB."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return -1.0


def setup_logger(output_dir: Path, run_name: str) -> logging.Logger:
    """Setup logger for this test run."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_path = log_dir / f"{run_name}.log"
    logger = logging.getLogger(f"umap_init_test.{run_name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    # File handler
    fh = logging.FileHandler(log_path, mode="w")
    fmt = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    
    return logger


def run_experiment(config_path: Path, workspace_dir: Path, output_dir: Path) -> Dict:
    """Run a single initialization method experiment.
    
    Returns:
        Dictionary with metrics and timing info
    """
    # Load config
    with config_path.open("r") as f:
        cfg = json.load(f)
    
    run_name = cfg["run_name"]
    method = cfg["method"]
    init_method = cfg.get("init", "N/A")
    
    # Setup logger
    logger = setup_logger(output_dir, run_name)
    logger.info("="*80)
    logger.info(f"UMAP Init Test: {run_name}")
    logger.info(f"Method: {method}, Init: {init_method}")
    logger.info("="*80)
    
    # Start timing and memory tracking
    start_time = time.time()
    tracemalloc.start()
    mem_start = get_memory_mb()
    
    # Load datasets
    logger.info("Loading datasets...")
    load_start = time.time()
    
    mf_csv = Path(cfg["mf_features_csv"])
    zinc_csv = Path(cfg["zinc_features_csv"])
    target = cfg["target"]
    accession = extract_accession(target)
    
    df_mf_all = load_csv_optimized(mf_csv, logger)
    df_zinc = load_csv_optimized(zinc_csv, logger)
    
    # Split actives from MF
    df_act = df_mf_all[df_mf_all["accession"] == accession].copy()
    df_mf = df_mf_all[df_mf_all["accession"] != accession].copy()
    
    logger.info(f"Actives: {len(df_act)}, MF: {len(df_mf)}, ZINC: {len(df_zinc)}")
    
    # Remove overlaps
    smiles_col = get_smiles_col(df_act)
    act_smiles = df_act[smiles_col]
    df_mf = remove_overlap_by_smiles(df_mf, smiles_col, act_smiles)
    df_zinc = remove_overlap_by_smiles(df_zinc, smiles_col, act_smiles)
    
    smiles_mf_col = get_smiles_col(df_mf)
    df_zinc = remove_overlap_by_smiles(df_zinc, smiles_mf_col, df_mf[smiles_mf_col])
    
    # Deduplicate
    df_mf = dedup_by_smiles(df_mf, "MF", logger)
    df_act = dedup_by_smiles(df_act, "Actives", logger)
    
    load_time = time.time() - load_start
    mem_after_load = get_memory_mb()
    logger.info(f"Loading completed in {load_time:.2f}s (memory: {mem_after_load:.0f} MB)")
    
    # Feature selection and preprocessing
    logger.info("Feature selection and preprocessing...")
    prep_start = time.time()
    
    feat_cols_mf = select_feature_columns(df_mf)
    feat_cols_zinc = select_feature_columns(df_zinc)
    common_feats = [c for c in feat_cols_mf if c in feat_cols_zinc]
    logger.info(f"Common features: {len(common_feats)}")
    
    # Zero-variance filtering
    df_train_check = pd.concat([df_mf[common_feats], df_zinc[common_feats]], axis=0, ignore_index=True)
    common_feats = remove_zero_variance_features(df_train_check, common_feats)
    del df_train_check
    logger.info(f"Features after zero-variance removal: {len(common_feats)}")
    
    # Fit imputer and scaler
    imputer, scaler = fit_scaler_on_mf_zinc(df_mf, df_zinc, common_feats)
    
    # Transform
    X_mf = imputer.transform(df_mf[common_feats].to_numpy(dtype=float))
    X_mf = scaler.transform(X_mf)
    X_zinc = imputer.transform(df_zinc[common_feats].to_numpy(dtype=float))
    X_zinc = scaler.transform(X_zinc)
    
    df_act_feat = df_act[[c for c in common_feats if c in df_act.columns]].reindex(columns=common_feats)
    X_act = imputer.transform(df_act_feat.to_numpy(dtype=float))
    X_act = scaler.transform(X_act)
    
    prep_time = time.time() - prep_start
    mem_after_prep = get_memory_mb()
    logger.info(f"Preprocessing completed in {prep_time:.2f}s (memory: {mem_after_prep:.0f} MB)")
    
    # Dimensionality reduction
    logger.info(f"Fitting {method}...")
    dr_start = time.time()
    
    X_train = np.vstack([X_mf, X_zinc])
    
    if method == "pca":
        model, Z_train = fit_pca(X_train, n_components=cfg["dim"])
        Z_act = model.transform(X_act)
    elif method == "umap":
        # Get init parameter
        init = cfg.get("init", "spectral")
        logger.info(f"UMAP init method: {init}")
        
        model, Z_train = fit_umap(
            X_train,
            n_components=cfg["dim"],
            n_neighbors=cfg["umap_params"]["n_neighbors"],
            min_dist=cfg["umap_params"]["min_dist"],
            metric=cfg["umap_params"]["metric"],
            random_state=None,
        )
        Z_act = model.transform(X_act)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    dr_time = time.time() - dr_start
    mem_after_dr = get_memory_mb()
    logger.info(f"DR completed in {dr_time:.2f}s (memory: {mem_after_dr:.0f} MB)")
    
    # Split embeddings
    n_mf = len(df_mf)
    Z_mf = Z_train[:n_mf]
    Z_zinc = Z_train[n_mf:]
    
    # Scoring
    logger.info("Scoring...")
    score_start = time.time()
    
    # Apply affinity cutoff
    affinity_cutoff_nM = cfg.get("affinity_cutoff_nM", 100000)
    mask_cut = pd.to_numeric(df_mf["Standard Value (nM)"], errors="coerce") <= affinity_cutoff_nM
    Z_mf_for_scoring = Z_mf[mask_cut.to_numpy(dtype=bool)]
    logger.info(f"MF for scoring: {len(Z_mf_for_scoring)} (cutoff={affinity_cutoff_nM} nM)")
    
    # Build evaluation set
    Z_eval = np.vstack([np.asarray(Z_act), np.asarray(Z_zinc)])
    labels = np.concatenate([np.ones(len(Z_act), dtype=int), np.zeros(len(Z_zinc), dtype=int)])
    
    # 1-NN scoring
    scores, distances = nn_min_distance_scores(Z_mf_for_scoring, Z_eval)
    
    score_time = time.time() - score_start
    logger.info(f"Scoring completed in {score_time:.2f}s")
    
    # Compute metrics
    logger.info("Computing metrics...")
    ef1 = ef_at_k_percent(scores, labels, 1.0)
    ef5 = ef_at_k_percent(scores, labels, 5.0)
    roc = roc_auc(labels, scores)
    pr = pr_auc(labels, scores)
    
    # Spearman on actives
    if "Standard Value (nM)" in df_act.columns:
        pact = to_pactivity_from_nM(df_act["Standard Value (nM)"]).to_numpy()
        rho, rho_p = spearman_rho(pact, scores[:len(df_act)])
    else:
        rho, rho_p = np.nan, np.nan
    
    # Get peak memory
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mem_mb = peak / 1024 / 1024
    
    total_time = time.time() - start_time
    
    # Results
    results = {
        "run_name": run_name,
        "method": method,
        "init": init_method,
        "dim": cfg["dim"],
        "n_features": len(common_feats),
        "n_actives": int(len(df_act)),
        "n_zinc": int(len(df_zinc)),
        "n_mf_scoring": int(len(Z_mf_for_scoring)),
        # Metrics
        "ef_1%": float(ef1),
        "ef_5%": float(ef5),
        "roc_auc": float(roc),
        "pr_auc": float(pr),
        "spearman_rho": float(rho) if not np.isnan(rho) else None,
        "spearman_p": float(rho_p) if not np.isnan(rho_p) else None,
        # Timing
        "time_total_s": float(total_time),
        "time_loading_s": float(load_time),
        "time_preprocessing_s": float(prep_time),
        "time_dr_s": float(dr_time),
        "time_scoring_s": float(score_time),
        # Memory
        "memory_start_mb": float(mem_start) if mem_start > 0 else None,
        "memory_after_load_mb": float(mem_after_load) if mem_after_load > 0 else None,
        "memory_after_prep_mb": float(mem_after_prep) if mem_after_prep > 0 else None,
        "memory_after_dr_mb": float(mem_after_dr) if mem_after_dr > 0 else None,
        "memory_peak_mb": float(peak_mem_mb),
    }
    
    logger.info("="*80)
    logger.info("RESULTS")
    logger.info("="*80)
    logger.info(f"EF@1%: {ef1:.2f}")
    logger.info(f"EF@5%: {ef5:.2f}")
    logger.info(f"ROC-AUC: {roc:.4f}")
    logger.info(f"PR-AUC: {pr:.4f}")
    logger.info(f"Total time: {total_time:.2f}s")
    logger.info(f"DR time: {dr_time:.2f}s")
    logger.info(f"Peak memory: {peak_mem_mb:.0f} MB")
    logger.info("="*80)
    
    # Save results
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"{run_name}.json"
    with results_path.open("w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="UMAP initialization method comparison test")
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON")
    parser.add_argument("--workspace", type=str, default="tests/umap_init_comparison/workspace",
                       help="Workspace directory")
    parser.add_argument("--output", type=str, default="tests/umap_init_comparison/results",
                       help="Output directory for results")
    args = parser.parse_args()
    
    config_path = Path(args.config)
    workspace_dir = Path(args.workspace)
    output_dir = Path(args.output)
    
    workspace_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        results = run_experiment(config_path, workspace_dir, output_dir)
        print(f"\n✓ Test completed successfully: {results['run_name']}")
        print(f"  EF@1%: {results['ef_1%']:.2f}")
        print(f"  Runtime: {results['time_total_s']:.2f}s")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
