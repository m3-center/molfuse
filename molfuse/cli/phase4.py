#!/usr/bin/env python3
"""
Phase 4: Cross-Target Generalization Study

Evaluates UMAP/features performance across different target proteins with varying
natural MF cloud sizes. Uses full MF cloud for each target (no subsampling).

Research Question:
    "Does performance vary across target proteins with different natural MF cloud
     sizes, and what is the minimum viable MF size for real-world screening?"

Key Differences from Phase 1:
    - Multiple target proteins (8 targets)
    - Natural MF cloud sizes (no artificial ablation)
    - Cross-target comparison focus

Usage:
    python -m molfuse.cli.phase4 \\
        --config configs/phase4_grid/phase4_antioxidant_rep1.json \\
        --workspace experiment_workspace_v4
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from molfuse.data.prep import (
    select_feature_columns,
    remove_zero_variance_features,
    remove_rows_with_infinity,
    fit_scaler_on_mf_zinc,
)
from molfuse.dr.pca import fit_pca
from molfuse.dr.umap_ import fit_umap
from molfuse.metrics.metrics import ef_at_k_percent, roc_auc, pr_auc, spearman_rho
from molfuse.scoring.nn import nn_min_distance_scores
from molfuse.io.paths import make_run_dirs


# ============================================================================
# Logger Setup
# ============================================================================

def setup_logger(log_path: Path) -> logging.Logger:
    """Setup logger with file and console handlers."""
    logger = logging.getLogger("phase4")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    # File handler
    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(ch)
    
    return logger


# ============================================================================
# Data Loading and Preprocessing
# ============================================================================

def load_csv_optimized(csv_path: Path, logger: logging.Logger) -> pd.DataFrame:
    """
    Load CSV with Parquet caching for faster subsequent loads.
    
    Strategy:
    1. Check for .parquet cache next to .csv
    2. If cache exists and is newer than CSV, load from Parquet
    3. Otherwise, load CSV and save Parquet cache
    """
    parquet_cache = csv_path.with_suffix(".parquet")
    
    # Check cache validity
    if parquet_cache.exists():
        csv_mtime = csv_path.stat().st_mtime
        cache_mtime = parquet_cache.stat().st_mtime
        
        if cache_mtime >= csv_mtime:
            logger.info(f"Loading from Parquet cache: {parquet_cache.name}")
            try:
                df = pd.read_parquet(parquet_cache)
                logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} cols")
                return df
            except Exception as e:
                logger.warning(f"Parquet cache corrupted, falling back to CSV: {e}")
    
    # Load CSV
    logger.info(f"Loading CSV: {csv_path.name}")
    df = pd.read_csv(csv_path, low_memory=False)
    logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} cols")
    
    # Defensive type conversion: convert all non-metadata columns to numeric
    # (prevents NaN explosion from string columns later in pipeline)
    # EXCEPT fingerprint columns which must remain as strings for parsing
    for col in df.columns:
        if col.lower() in ["fingerprint", "smiles", "canonical_smiles", "compound chembl id", "accession", "zinc_id", "target"]:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Save Parquet cache for future runs (one-time cost)
    try:
        logger.info(f"Saving Parquet cache: {parquet_cache.name}")
        df.to_parquet(parquet_cache, index=False, compression="snappy")
        logger.info(f"Parquet cache saved ({parquet_cache.stat().st_size / 1e6:.1f} MB)")
    except Exception as e:
        logger.warning(f"Failed to save Parquet cache: {e}")
    
    return df


def extract_accession(target: str) -> str:
    """Extract UniProt accession from target name (e.g., 'ABL1_P00519' -> 'P00519')."""
    # Match pattern: {GENE}_{ACCESSION}
    m = re.search(r"_([A-Z0-9]{6})$", target)
    if m:
        return m.group(1)
    return target


def get_smiles_col(df: pd.DataFrame) -> str:
    """Get SMILES column name (prefer canonical_smiles)."""
    if "canonical_smiles" in df.columns:
        return "canonical_smiles"
    elif "SMILES" in df.columns:
        return "SMILES"
    else:
        raise ValueError("No SMILES column found (expected 'canonical_smiles' or 'SMILES')")


def dedup_by_smiles(df: pd.DataFrame, label: str, logger: logging.Logger) -> pd.DataFrame:
    """Deduplicate by SMILES, keeping first occurrence."""
    smiles_col = get_smiles_col(df)
    n_before = len(df)
    df = df.drop_duplicates(subset=[smiles_col], keep="first").reset_index(drop=True)
    n_after = len(df)
    if n_before > n_after:
        logger.info(f"{label}: removed {n_before - n_after:,} duplicate SMILES ({n_after:,} remain)")
    return df


def remove_overlap_by_smiles(df: pd.DataFrame, smiles_col: str, exclude_smiles: pd.Series) -> pd.DataFrame:
    """Remove rows where SMILES appears in exclude_smiles set."""
    exclude_set = set(exclude_smiles.dropna().astype(str).str.strip().str.upper())
    n_before = len(df)
    
    df_smiles_col = get_smiles_col(df)
    mask = ~df[df_smiles_col].astype(str).str.strip().str.upper().isin(exclude_set)
    df = df[mask].reset_index(drop=True)
    
    n_after = len(df)
    removed = n_before - n_after
    if removed > 0:
        logging.getLogger("phase4").info(f"  Removed {removed:,} overlapping SMILES ({n_after:,} remain)")
    return df


def parse_fp_series(series: pd.Series, label: str, logger: logging.Logger) -> Tuple[np.ndarray, pd.Index]:
    """
    Parse fingerprint series from string format to numpy array.
    
    Handles various formats:
    - "[0,1,0,1,...]" (bracketed)
    - "0,1,0,1,..." (unbracketed)
    - With/without quotes and spaces
    
    Returns:
        X: numpy array (n_rows, fp_length) of 0/1 values
        idx: pandas Index of valid rows
    """
    before = len(series)
    ser = series.astype(str).str.strip().str.replace("\"", "", regex=False)
    # Drop NaNs or empty strings
    mask_nonempty = ser.notna() & (ser.str.len() > 0)
    ser = ser[mask_nonempty]

    parsed_list: List[np.ndarray] = []
    valid_idx: List[object] = []
    expected_len: Optional[int] = None
    
    for idx, s in ser.items():
        try:
            s_clean = s.strip("[]").replace(" ", "")
            arr = np.array([int(x) for x in s_clean.split(",") if x], dtype=np.int8)
            if expected_len is None:
                expected_len = len(arr)
            elif len(arr) != expected_len:
                continue  # skip mismatched lengths
            parsed_list.append(arr)
            valid_idx.append(idx)
        except Exception:
            continue
    
    if expected_len is None:
        raise RuntimeError(f"{label}: could not parse any fingerprint rows")
    
    X = np.vstack(parsed_list).astype(np.float32)
    after = X.shape[0]
    logger.info(f"{label}: parsed fingerprints: rows {before}->{after}, fp_len={expected_len}")
    return X, pd.Index(valid_idx)


def to_pactivity_from_nM(affinity_nM: pd.Series) -> pd.Series:
    """Convert affinity (nM) to pActivity: 9 - log10(nM)."""
    numeric = pd.to_numeric(affinity_nM, errors="coerce")
    return 9.0 - np.log10(numeric.clip(lower=1e-3))


# ============================================================================
# Main Phase 4 Pipeline
# ============================================================================

def run_phase4(config_path: Path, workspace_dir: Path) -> None:
    """Execute Phase 4: Cross-Target Generalization Study."""
    
    # Load config
    with config_path.open("r") as f:
        cfg = json.load(f)
    
    run_name = cfg["run_name"]
    method = cfg["method"]
    representation = cfg["representation"]
    target = cfg["target"]
    target_kw = cfg["target_kw"]  # e.g., "KW-0049_Antioxidant"
    random_seed = cfg.get("random_seed", 42)
    
    # Create run directories using make_run_dirs
    phase4_run_name = cfg.get("phase4_run_name", "cross_target")
    ws = make_run_dirs(workspace_dir, phase="phase4", run_name=f"{phase4_run_name}/{run_name}")
    run_dir = ws["base"]
    logs_dir = ws["logs"]
    metrics_dir = ws["metrics"]
    artifacts_dir = ws["artifacts"]
    
    # Setup logger
    log_path = logs_dir / "run.log"
    logger = setup_logger(log_path)
    
    logger.info("="*80)
    logger.info("PHASE 4: CROSS-TARGET GENERALIZATION STUDY")
    logger.info("="*80)
    logger.info(f"Run: {run_name}")
    logger.info(f"Method: {method}, Representation: {representation}")
    logger.info(f"Target: {target} ({target_kw})")
    logger.info(f"Random seed: {random_seed}")
    logger.info(f"Workspace: {run_dir}")
    logger.info("="*80)
    
    start_time = time.time()
    
    # ========================================================================
    # Step 1: Load Datasets
    # ========================================================================
    logger.info("\n[1/8] Loading datasets...")
    
    mf_csv = Path(cfg["mf_features_csv"] if representation == "features" else cfg["mf_fingerprints_csv"])
    zinc_csv = Path(cfg["zinc_features_csv"] if representation == "features" else cfg["zinc_fingerprints_csv"])
    
    df_mf_all = load_csv_optimized(mf_csv, logger)
    df_zinc = load_csv_optimized(zinc_csv, logger)
    
    # Split actives from MF
    accession = extract_accession(target)
    df_act = df_mf_all[df_mf_all["accession"] == accession].copy()
    df_mf = df_mf_all[df_mf_all["accession"] != accession].copy()
    
    logger.info(f"Actives: {len(df_act):,}, MF (full): {len(df_mf):,}, ZINC: {len(df_zinc):,}")
    
    # Deduplicate
    df_act = dedup_by_smiles(df_act, "Actives", logger)
    df_mf = dedup_by_smiles(df_mf, "MF", logger)
    df_zinc = dedup_by_smiles(df_zinc, "ZINC", logger)
    
    # Remove overlaps
    smiles_col_act = get_smiles_col(df_act)
    act_smiles = df_act[smiles_col_act]
    
    df_mf = remove_overlap_by_smiles(df_mf, get_smiles_col(df_mf), act_smiles)
    df_zinc = remove_overlap_by_smiles(df_zinc, get_smiles_col(df_zinc), act_smiles)
    
    smiles_col_mf = get_smiles_col(df_mf)
    df_zinc = remove_overlap_by_smiles(df_zinc, get_smiles_col(df_zinc), df_mf[smiles_col_mf])
    
    # ========================================================================
    # Step 2: Apply Affinity Cutoff (No Subsampling)
    # ========================================================================
    logger.info("\n[2/8] Applying affinity cutoff to MF cloud...")
    
    affinity_cutoff_nM = cfg.get("affinity_cutoff_nM", 100000)
    affinity_col = "Standard Value (nM)"
    
    if affinity_col in df_mf.columns:
        affinity = pd.to_numeric(df_mf[affinity_col], errors="coerce")
        mask = affinity <= affinity_cutoff_nM
        df_mf = df_mf[mask].copy()
        logger.info(f"MF after affinity cutoff (≤ {affinity_cutoff_nM} nM): {len(df_mf):,} compounds")
    else:
        logger.warning(f"No affinity column found, using full MF cloud")
    
    if len(df_mf) == 0:
        logger.error("MF cloud is empty after affinity filtering!")
        raise ValueError("Empty MF cloud after cutoff")
    
    logger.info(f"Final MF cloud size: {len(df_mf):,} compounds (full natural size)")
    
    # ========================================================================
    # Step 3: Feature Selection and Preprocessing
    # ========================================================================
    logger.info("\n[3/8] Feature selection and preprocessing...")
    
    if representation == "features":
        # Select numeric feature columns
        feat_cols_mf = select_feature_columns(df_mf)
        feat_cols_zinc = select_feature_columns(df_zinc)
        common_feats = [c for c in feat_cols_mf if c in feat_cols_zinc]
        logger.info(f"Common features: {len(common_feats)}")
        
        # Zero-variance filtering
        df_train_check = pd.concat([df_mf[common_feats], df_zinc[common_feats]], axis=0, ignore_index=True)
        common_feats = remove_zero_variance_features(df_train_check, common_feats)
        del df_train_check
        logger.info(f"Features after zero-variance removal: {len(common_feats)}")
        
        # Remove rows with infinity (prevents scikit-learn errors)
        n_act_before = len(df_act)
        df_act = remove_rows_with_infinity(df_act, [c for c in common_feats if c in df_act.columns])
        n_act_removed = n_act_before - len(df_act)
        if n_act_removed > 0:
            logger.info(f"Actives: removed {n_act_removed} rows with infinity ({len(df_act):,} remain)")
        
        n_mf_before = len(df_mf)
        df_mf = remove_rows_with_infinity(df_mf, common_feats)
        n_mf_removed = n_mf_before - len(df_mf)
        if n_mf_removed > 0:
            logger.info(f"MF: removed {n_mf_removed} rows with infinity ({len(df_mf):,} remain)")
        
        n_zinc_before = len(df_zinc)
        df_zinc = remove_rows_with_infinity(df_zinc, common_feats)
        n_zinc_removed = n_zinc_before - len(df_zinc)
        if n_zinc_removed > 0:
            logger.info(f"ZINC: removed {n_zinc_removed} rows with infinity ({len(df_zinc):,} remain)")
        
        # Fit imputer and scaler on MF + ZINC
        imputer, scaler = fit_scaler_on_mf_zinc(df_mf, df_zinc, common_feats)
        
        # Transform
        X_mf = imputer.transform(df_mf[common_feats].to_numpy(dtype=float))
        X_mf = scaler.transform(X_mf)
        
        X_zinc = imputer.transform(df_zinc[common_feats].to_numpy(dtype=float))
        X_zinc = scaler.transform(X_zinc)
        
        df_act_feat = df_act[[c for c in common_feats if c in df_act.columns]].reindex(columns=common_feats)
        X_act = imputer.transform(df_act_feat.to_numpy(dtype=float))
        X_act = scaler.transform(X_act)
        
        # Save scaler
        joblib.dump(scaler, artifacts_dir / "scaler.joblib")
        logger.info(f"Saved scaler (StandardScaler)")
        
    else:  # fingerprints
        # Parse fingerprints
        logger.info("Parsing fingerprints...")
        X_mf, idx_mf = parse_fp_series(df_mf["Fingerprint"], "MF", logger)
        df_mf = df_mf.loc[idx_mf].copy()
        
        X_zinc, idx_zinc = parse_fp_series(df_zinc["Fingerprint"], "ZINC", logger)
        df_zinc = df_zinc.loc[idx_zinc].copy()
        
        X_act, idx_act = parse_fp_series(df_act["Fingerprint"], "Actives", logger)
        df_act = df_act.loc[idx_act].copy()
        
        logger.info(f"Fingerprint dimensions: {X_mf.shape[1]}")
    
    # ========================================================================
    # Step 4: Dimensionality Reduction
    # ========================================================================
    logger.info(f"\n[4/8] Fitting {method.upper()} model...")
    
    X_train = np.vstack([X_mf, X_zinc])
    logger.info(f"Training data shape: {X_train.shape}")
    
    dim = cfg["dim"]
    
    if method == "pca":
        model, Z_train = fit_pca(X_train, n_components=dim)
        Z_act = model.transform(X_act)
        
        # Save model
        joblib.dump(model, artifacts_dir / "pca_model.joblib")
        logger.info(f"Saved PCA model ({dim} components)")
        
    elif method == "umap":
        umap_params = cfg["umap_params"]
        metric = "jaccard" if representation == "fingerprints" else "euclidean"
        
        model, Z_train = fit_umap(
            X_train,
            n_components=dim,
            n_neighbors=umap_params["n_neighbors"],
            min_dist=umap_params["min_dist"],
            metric=metric,
            random_state=None,  # Parallel execution
        )
        Z_act = model.transform(X_act)
        
        # Save model
        joblib.dump(model, artifacts_dir / "umap_model.joblib")
        logger.info(f"Saved UMAP model ({dim} components, metric={metric})")
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Split embeddings
    n_mf = len(df_mf)
    Z_mf = Z_train[:n_mf]
    Z_zinc = Z_train[n_mf:]
    
    logger.info(f"Embeddings: MF={Z_mf.shape}, ZINC={Z_zinc.shape}, Actives={Z_act.shape}")
    
    # ========================================================================
    # Step 5: Save Embeddings
    # ========================================================================
    logger.info("\n[5/8] Saving embeddings...")
    
    # MF embeddings
    emb_mf = pd.DataFrame(Z_mf, columns=[f"dim_{i}" for i in range(dim)])
    id_cols_mf = ["SMILES", "canonical_smiles", "Compound ChEMBL ID", "Standard Value (nM)", "accession"]
    for col in id_cols_mf:
        if col in df_mf.columns:
            emb_mf[col] = df_mf[col].values
    emb_mf.to_csv(artifacts_dir / "embedding_mf.csv", index=False)
    logger.info(f"Saved embedding_mf.csv ({len(emb_mf):,} rows)")
    
    # ZINC embeddings
    emb_zinc = pd.DataFrame(Z_zinc, columns=[f"dim_{i}" for i in range(dim)])
    id_cols_zinc = ["SMILES", "canonical_smiles", "Compound ChEMBL ID", "zinc_id"]
    for col in id_cols_zinc:
        if col in df_zinc.columns:
            emb_zinc[col] = df_zinc[col].values
    emb_zinc.to_csv(artifacts_dir / "embedding_zinc.csv", index=False)
    logger.info(f"Saved embedding_zinc.csv ({len(emb_zinc):,} rows)")
    
    # Actives embeddings
    emb_act = pd.DataFrame(Z_act, columns=[f"dim_{i}" for i in range(dim)])
    id_cols_act = ["SMILES", "canonical_smiles", "Compound ChEMBL ID", "Standard Value (nM)"]
    for col in id_cols_act:
        if col in df_act.columns:
            emb_act[col] = df_act[col].values
    emb_act.to_csv(artifacts_dir / "embedding_actives.csv", index=False)
    logger.info(f"Saved embedding_actives.csv ({len(emb_act):,} rows)")
    
    # ========================================================================
    # Step 6: Scoring
    # ========================================================================
    logger.info("\n[6/8] Scoring evaluation set...")
    
    # Build evaluation set: actives + ZINC
    Z_eval = np.vstack([np.asarray(Z_act), np.asarray(Z_zinc)])
    labels = np.concatenate([
        np.ones(len(Z_act), dtype=int),
        np.zeros(len(Z_zinc), dtype=int)
    ])
    
    logger.info(f"Evaluation set: {len(Z_eval):,} compounds ({len(Z_act):,} actives, {len(Z_zinc):,} ZINC)")
    
    # 1-NN scoring against MF cloud
    scores, distances = nn_min_distance_scores(Z_mf, Z_eval)
    
    logger.info(f"Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    logger.info(f"Distance range: [{distances.min():.4f}, {distances.max():.4f}]")
    
    # Save ranked scores
    ranked = pd.DataFrame({
        "score": scores,
        "distance": distances,
        "label": labels,
    })
    
    # Add compound IDs
    ids_combined = pd.concat([
        emb_act[[c for c in emb_act.columns if c not in [f"dim_{i}" for i in range(dim)]]],
        emb_zinc[[c for c in emb_zinc.columns if c not in [f"dim_{i}" for i in range(dim)]]]
    ], ignore_index=True)
    
    for col in ids_combined.columns:
        ranked[col] = ids_combined[col].values
    
    ranked = ranked.sort_values("score", ascending=False).reset_index(drop=True)
    ranked.to_csv(artifacts_dir / "ranked_scores.csv", index=False)
    logger.info(f"Saved ranked_scores.csv")
    
    # ========================================================================
    # Step 7: Compute Metrics
    # ========================================================================
    logger.info("\n[7/8] Computing metrics...")
    
    ef1 = ef_at_k_percent(scores, labels, 1.0)
    ef5 = ef_at_k_percent(scores, labels, 5.0)
    ef10 = ef_at_k_percent(scores, labels, 10.0)
    roc = roc_auc(labels, scores)
    pr = pr_auc(labels, scores)
    
    logger.info(f"EF@1%:  {ef1:.2f}")
    logger.info(f"EF@5%:  {ef5:.2f}")
    logger.info(f"EF@10%: {ef10:.2f}")
    logger.info(f"ROC-AUC: {roc:.4f}")
    logger.info(f"PR-AUC:  {pr:.4f}")
    
    # Spearman correlation (actives only)
    if "Standard Value (nM)" in df_act.columns:
        pact = to_pactivity_from_nM(df_act["Standard Value (nM)"]).to_numpy()
        rho, rho_p = spearman_rho(pact, scores[:len(df_act)])
        logger.info(f"Spearman ρ: {rho:.4f} (p={rho_p:.2e})")
    else:
        rho, rho_p = np.nan, np.nan
    
    # ========================================================================
    # Step 8: Save Summary
    # ========================================================================
    logger.info("\n[8/8] Saving summary...")
    
    elapsed = time.time() - start_time
    
    summary = {
        "run_name": run_name,
        "method": method,
        "representation": representation,
        "dim": dim,
        "target": target,
        "target_kw": target_kw,
        "mf_size_natural": int(len(df_mf)),
        "affinity_cutoff_nM": affinity_cutoff_nM,
        "random_seed": random_seed,
        "n_actives": int(len(df_act)),
        "n_zinc": int(len(df_zinc)),
        "ef_1%": float(ef1),
        "ef_5%": float(ef5),
        "ef_10%": float(ef10),
        "roc_auc": float(roc),
        "pr_auc": float(pr),
        "spearman_rho": float(rho) if not np.isnan(rho) else None,
        "spearman_p": float(rho_p) if not np.isnan(rho_p) else None,
        "elapsed_time_s": float(elapsed),
        "config": cfg,
    }
    
    # Save metrics
    with (metrics_dir / "metrics.json").open("w") as f:
        json.dump(summary, f, indent=2)
    
    # Save phase4_summary (for idempotent skip logic)
    with (logs_dir / "phase4_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Saved metrics and summary")
    logger.info("="*80)
    logger.info(f"PHASE 4 COMPLETED in {elapsed:.1f}s")
    logger.info(f"Target: {target_kw} | MF size: {len(df_mf):,} | EF@1%: {ef1:.2f} | ROC-AUC: {roc:.4f}")
    logger.info("="*80)


# ============================================================================
# CLI Entrypoint
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 4: Cross-Target Generalization Study")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to Phase 4 config JSON")
    parser.add_argument("--workspace", type=str, default="experiment_workspace_v4",
                       help="Workspace directory")
    
    args = parser.parse_args()
    
    config_path = Path(args.config)
    workspace_dir = Path(args.workspace)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    run_phase4(config_path, workspace_dir)


if __name__ == "__main__":
    main()
