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
    affinity_cutoff_nM = float(cfg.get("affinity_cutoff_nM", 100))
    
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
    logger.info(f"Affinity cutoff: {affinity_cutoff_nM} nM")
    logger.info(f"Workspace: {run_dir}")
    logger.info("="*80)
    
    start_time = time.time()
    np.random.seed(random_seed)
    
    # ========================================================================
    # Load Datasets
    # ========================================================================
    logger.info("\n[1/6] Loading datasets...")
    
    # Load MF features (contains affinity data embedded)
    mf_features_csv = Path(cfg["mf_features_csv"])
    logger.info(f"Loading MF features: {mf_features_csv}")
    df_mf_all = pd.read_csv(mf_features_csv, low_memory=False)
    logger.info(f"  Loaded: {len(df_mf_all):,} rows")
    
    # Filter MF by target and affinity cutoff
    df_mf = df_mf_all[df_mf_all["accession"] != target].copy()
    logger.info(f"  After removing target {target}: {len(df_mf):,} rows")
    
    if "Standard Value (nM)" in df_mf.columns:
        mask_cut = pd.to_numeric(df_mf["Standard Value (nM)"], errors="coerce") <= affinity_cutoff_nM
        df_mf = df_mf[mask_cut].copy()
        logger.info(f"  After affinity cutoff (≤{affinity_cutoff_nM} nM): {len(df_mf):,} rows")
    
    # Deduplicate MF by SMILES (median aggregation)
    smiles_col = "canonical_smiles" if "canonical_smiles" in df_mf.columns else "SMILES"
    before = len(df_mf)
    agg_dict = {}
    for col in df_mf.columns:
        if col == smiles_col:
            continue
        elif col == "Standard Value (nM)":
            agg_dict[col] = "median"
        else:
            agg_dict[col] = "first"
    df_mf = df_mf.groupby(smiles_col, as_index=False).agg(agg_dict)
    logger.info(f"  Deduplicated MF: {before:,} -> {len(df_mf):,} (removed {before-len(df_mf):,})")
    
    # Extract actives (target compounds)
    df_actives = df_mf_all[df_mf_all["accession"] == target].copy()
    logger.info(f"Actives for {target}: {len(df_actives):,} rows")
    
    # Deduplicate actives
    before_act = len(df_actives)
    df_actives = df_actives.groupby(smiles_col, as_index=False).agg(agg_dict)
    logger.info(f"  Deduplicated actives: {before_act:,} -> {len(df_actives):,} (removed {before_act-len(df_actives):,})")
    
    # Load ZINC decoys
    zinc_csv = Path(cfg["zinc_features_csv"])
    logger.info(f"Loading ZINC: {zinc_csv}")
    df_zinc = pd.read_csv(zinc_csv, low_memory=False)
    logger.info(f"  Loaded: {len(df_zinc):,} rows")
    
    # Deduplicate ZINC
    zinc_smiles_col = "canonical_smiles" if "canonical_smiles" in df_zinc.columns else "SMILES"
    before_zinc = len(df_zinc)
    df_zinc = df_zinc.drop_duplicates(subset=[zinc_smiles_col], keep="first").copy()
    logger.info(f"  Deduplicated ZINC: {before_zinc:,} -> {len(df_zinc):,} (removed {before_zinc-len(df_zinc):,})")
    
    # Remove MF-ZINC overlap
    mf_smiles = set(df_mf[smiles_col].values)
    before_overlap = len(df_zinc)
    df_zinc = df_zinc[~df_zinc[zinc_smiles_col].isin(mf_smiles)].copy()
    logger.info(f"  Removed MF-ZINC overlap: {before_overlap:,} -> {len(df_zinc):,} (removed {before_overlap-len(df_zinc):,})")
    
    # Remove actives from ZINC
    act_smiles = set(df_actives[smiles_col].values)
    before_act_overlap = len(df_zinc)
    df_zinc = df_zinc[~df_zinc[zinc_smiles_col].isin(act_smiles)].copy()
    logger.info(f"  Removed actives from ZINC: {before_act_overlap:,} -> {len(df_zinc):,} (removed {before_act_overlap-len(df_zinc):,})")
    
    # ========================================================================
    # Experiment-specific scoring
    # ========================================================================
    logger.info(f"\n[2/6] Running {experiment_type} experiment...")
    
    if experiment_type == "tanimoto":
        logger.info("Tanimoto Baseline: ECFP4 fingerprint similarity")
        
        # Generate ECFP4 fingerprints
        logger.info("  Computing ECFP4 fingerprints...")
        mf_fps = compute_ecfp4_fingerprints(df_mf[smiles_col].tolist(), radius=2, n_bits=2048)
        act_fps = compute_ecfp4_fingerprints(df_actives[smiles_col].tolist(), radius=2, n_bits=2048)
        zinc_fps = compute_ecfp4_fingerprints(df_zinc[zinc_smiles_col].tolist(), radius=2, n_bits=2048)
        logger.info(f"    MF: {mf_fps.shape}")
        logger.info(f"    Actives: {act_fps.shape}")
        logger.info(f"    ZINC: {zinc_fps.shape}")
        
        # Score via max Tanimoto similarity
        logger.info("  Computing Tanimoto scores...")
        act_scores = tanimoto_similarity_max(act_fps, mf_fps)
        zinc_scores = tanimoto_similarity_max(zinc_fps, mf_fps)
        logger.info(f"    Actives: min={act_scores.min():.4f}, max={act_scores.max():.4f}, mean={act_scores.mean():.4f}")
        logger.info(f"    ZINC: min={zinc_scores.min():.4f}, max={zinc_scores.max():.4f}, mean={zinc_scores.mean():.4f}")
        
    elif experiment_type == "raw_descriptors":
        logger.info("Raw Descriptor Baseline: No UMAP (high-D space)")
        
        # Use already-deduplicated DataFrames (df_mf, df_actives, df_zinc)
        # These were already deduplicated via median aggregation earlier
        logger.info(f"  Using deduplicated MF: {len(df_mf):,} rows")
        logger.info(f"  Using deduplicated actives: {len(df_actives):,} rows")
        logger.info(f"  Using deduplicated ZINC: {len(df_zinc):,} rows")
        
        # Select numeric features from MF
        feature_cols_mf = select_feature_columns(df_mf)
        feature_cols_zinc = select_feature_columns(df_zinc)
        feature_cols = [c for c in feature_cols_mf if c in feature_cols_zinc]
        logger.info(f"    Selected {len(feature_cols)} common numeric feature columns")
        
        if not feature_cols:
            raise RuntimeError("No common feature columns found between MF and ZINC")
        
        # Use imputer to handle NaNs (same as Phase 1)
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        
        logger.info("  Fitting imputer (median) + StandardScaler on MF+ZINC...")
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        
        # Extract feature matrices
        X_mf_raw = df_mf[feature_cols].to_numpy(dtype=float)
        X_zinc_raw = df_zinc[[c for c in feature_cols if c in df_zinc.columns]].reindex(columns=feature_cols).to_numpy(dtype=float)
        X_act_raw = df_actives[[c for c in feature_cols if c in df_actives.columns]].reindex(columns=feature_cols).to_numpy(dtype=float)
        
        # Remove infinity values (replace with NaN, then impute)
        logger.info("  Removing infinity values...")
        X_mf_raw[~np.isfinite(X_mf_raw)] = np.nan
        X_zinc_raw[~np.isfinite(X_zinc_raw)] = np.nan
        X_act_raw[~np.isfinite(X_act_raw)] = np.nan
        
        # Fit imputer + scaler on MF+ZINC
        X_combined = np.vstack([X_mf_raw, X_zinc_raw])
        imputer.fit(X_combined)
        X_combined_imputed = imputer.transform(X_combined)
        scaler.fit(X_combined_imputed)
        
        # Transform all datasets
        X_mf = scaler.transform(imputer.transform(X_mf_raw))
        X_act = scaler.transform(imputer.transform(X_act_raw))
        X_zinc = scaler.transform(imputer.transform(X_zinc_raw))
        
        logger.info(f"    MF features: {X_mf.shape}")
        logger.info(f"    Actives features: {X_act.shape}")
        logger.info(f"    ZINC features: {X_zinc.shape}")
        
        # Score via 1-NN in high-D space
        logger.info("  Computing 1-NN scores in high-D space...")
        act_scores, _ = nn_min_distance_scores(X_mf, X_act, metric="euclidean")
        zinc_scores, _ = nn_min_distance_scores(X_mf, X_zinc, metric="euclidean")
        logger.info(f"    Actives: min={act_scores.min():.4f}, max={act_scores.max():.4f}, mean={act_scores.mean():.4f}")
        logger.info(f"    ZINC: min={zinc_scores.min():.4f}, max={zinc_scores.max():.4f}, mean={zinc_scores.mean():.4f}")
        
    elif experiment_type == "negative_control":
        logger.info("Negative Control: Non-kinase (receptor) ligands vs kinase model")
        
        # Load receptor dataset (non-kinase control)
        receptor_csv = Path(cfg.get("receptor_features_csv", ""))
        if not receptor_csv or not receptor_csv.exists():
            logger.error(f"  Receptor dataset not found: {receptor_csv}")
            logger.info("  Creating empty results to mark as skipped")
            act_scores = np.array([])
            zinc_scores = np.array([])
        else:
            logger.info(f"  Loading receptor ligands: {receptor_csv}")
            df_receptors = pd.read_csv(receptor_csv, low_memory=False)
            logger.info(f"    Loaded: {len(df_receptors):,} rows")
            
            # Remove any overlap with target (unlikely but check)
            if "accession" in df_receptors.columns:
                df_receptors = df_receptors[df_receptors["accession"] != target].copy()
                logger.info(f"    After removing target overlap: {len(df_receptors):,} rows")
            
            # Deduplicate receptors
            receptor_smiles_col = "canonical_smiles" if "canonical_smiles" in df_receptors.columns else "SMILES"
            before_r = len(df_receptors)
            df_receptors = df_receptors.drop_duplicates(subset=[receptor_smiles_col], keep="first").copy()
            logger.info(f"    Deduplicated receptors: {before_r:,} -> {len(df_receptors):,} (removed {before_r-len(df_receptors):,})")
            
            # Remove MF overlap (unlikely)
            before_mf = len(df_receptors)
            df_receptors = df_receptors[~df_receptors[receptor_smiles_col].isin(mf_smiles)].copy()
            logger.info(f"    Removed MF overlap: {before_mf:,} -> {len(df_receptors):,} (removed {before_mf-len(df_receptors):,})")
            
            # Sample receptors to match active set size (for balanced metrics)
            n_sample = min(len(df_receptors), len(df_actives) * 10)  # Use up to 10x actives
            if len(df_receptors) > n_sample:
                df_receptors = df_receptors.sample(n=n_sample, random_state=random_seed).copy()
                logger.info(f"    Sampled receptors: {n_sample:,} rows")
            
            # Load Phase 1 best model (scaler + UMAP)
            logger.info("  Loading Phase 1 best features-based model artifacts...")
            phase1_model_dir = Path(cfg.get("phase1_best_model_dir", "experiment_workspace_v4/phase1"))
            
            # Find best features-based Phase 1 run (not fingerprints) by scanning for highest EF@1%
            best_ef1 = 0
            best_artifacts_dir = None
            for run_dir in phase1_model_dir.glob("*/"):
                metrics_file = run_dir / "metrics" / "metrics.json"
                scaler_file = run_dir / "artifacts" / "scaler.joblib"
                if metrics_file.exists() and scaler_file.exists():
                    # Check if it's a features-based model (not fingerprints)
                    try:
                        scaler_test = joblib.load(scaler_file)
                        is_features = not (isinstance(scaler_test, dict) and scaler_test.get("type") == "passthrough")
                        if is_features:
                            with metrics_file.open("r") as f:
                                metrics = json.load(f)
                            if metrics.get("ef_1%", 0) > best_ef1:
                                best_ef1 = metrics["ef_1%"]
                                best_artifacts_dir = run_dir / "artifacts"
                    except Exception:
                        continue
            
            if not best_artifacts_dir or not best_artifacts_dir.exists():
                logger.error("  No features-based Phase 1 model found - cannot run negative control")
                logger.info("  (Negative control requires features, not fingerprints)")
                logger.info("  Creating empty results to mark as skipped")
                act_scores = np.array([])
                zinc_scores = np.array([])
            else:
                logger.info(f"    Best features-based Phase 1 run: {best_artifacts_dir.parent.name} (EF@1%: {best_ef1:.2f})")
                
                # Load imputer, scaler and UMAP model
                imputer_path = best_artifacts_dir / "imputer.joblib"
                scaler_path = best_artifacts_dir / "scaler.joblib"
                umap_path = best_artifacts_dir / "umap_model.joblib"
                
                if not imputer_path.exists() or not scaler_path.exists() or not umap_path.exists():
                    logger.error(f"  Model artifacts not found: {imputer_path}, {scaler_path}, {umap_path}")
                    logger.info("  Creating empty results to mark as skipped")
                    act_scores = np.array([])
                    zinc_scores = np.array([])
                else:
                    phase1_imputer = joblib.load(imputer_path)
                    scaler = joblib.load(scaler_path)
                    umap_model = joblib.load(umap_path)
                    logger.info("    Loaded imputer + scaler + UMAP model")
                    
                    # Get feature names from imputer (exact features used in Phase 1)
                    phase1_features = None
                    if hasattr(phase1_imputer, 'feature_names_in_'):
                        phase1_features = list(phase1_imputer.feature_names_in_)
                        logger.info(f"    Phase 1 used {len(phase1_features)} features")
                    else:
                        logger.error("    Cannot determine Phase 1 feature names from imputer")
                        logger.info("    Creating empty results to mark as skipped")
                        act_scores = np.array([])
                        zinc_scores = np.array([])
                    
                    # Load Phase 1 MF embedding (to use as reference for scoring)
                    mf_embedding_path = best_artifacts_dir / "embedding_mf.csv"
                    if not mf_embedding_path.exists():
                        logger.error(f"  MF embedding not found: {mf_embedding_path}")
                        logger.info("  Creating empty results to mark as skipped")
                        act_scores = np.array([])
                        zinc_scores = np.array([])
                    else:
                        df_mf_embed = pd.read_csv(mf_embedding_path)
                        # Phase 1 embeddings use z0, z1, z2... column naming
                        embed_cols = [f"z{i}" for i in range(cfg["dim"])]
                        if not all(col in df_mf_embed.columns for col in embed_cols):
                            logger.error(f"  Expected columns {embed_cols} not found in embedding")
                            logger.error(f"  Available columns: {list(df_mf_embed.columns)}")
                            act_scores = np.array([])
                            zinc_scores = np.array([])
                        else:
                            X_mf_embed = df_mf_embed[embed_cols].values
                            logger.info(f"    Loaded MF embedding: {X_mf_embed.shape}")
                        
                            # Use exact Phase 1 features (from imputer.feature_names_in_)
                            if not phase1_features:
                                logger.error("  Phase 1 feature list not available")
                                act_scores = np.array([])
                                zinc_scores = np.array([])
                            else:
                                logger.info(f"    Using Phase 1 features: {len(phase1_features)} columns")
                                
                                # Check which Phase 1 features are missing in receptor data
                                missing_feats = [f for f in phase1_features if f not in df_receptors.columns]
                                if missing_feats:
                                    logger.warning(f"    {len(missing_feats)} Phase 1 features missing in receptor data (will be filled with NaN)")
                                
                                # Extract receptor features (missing features → NaN)
                                X_receptors_raw = df_receptors.reindex(columns=phase1_features).to_numpy(dtype=np.float64)
                                
                                # Remove infinity values (replace with NaN, then impute)
                                logger.info("    Removing infinity values from receptors...")
                                X_receptors_raw[~np.isfinite(X_receptors_raw)] = np.nan
                                logger.info(f"    Receptor features shape (pre-transform): {X_receptors_raw.shape}")
                                
                                # Transform through Phase 1 pipeline
                                X_receptors_imputed = phase1_imputer.transform(X_receptors_raw)
                                X_receptors_scaled = scaler.transform(X_receptors_imputed)
                                X_receptors_embed = umap_model.transform(X_receptors_scaled)
                                logger.info(f"    Transformed receptors: {X_receptors_embed.shape}")
                                
                                # Score receptors against kinase MF cloud
                                logger.info("  Scoring receptor ligands against kinase model...")
                                receptor_scores, _ = nn_min_distance_scores(X_mf_embed, X_receptors_embed, metric="euclidean")
                                logger.info(f"    Receptor scores: min={receptor_scores.min():.4f}, max={receptor_scores.max():.4f}, mean={receptor_scores.mean():.4f}")
                                
                                # Also score ZINC for comparison
                                zinc_scores, _ = nn_min_distance_scores(X_mf_embed, X_receptors_embed, metric="euclidean")  # Reuse receptor scores as "decoys"
                                
                                # For negative control: receptors are the "actives" (should NOT be enriched)
                                # ZINC decoys are replaced by a second sample of receptors
                                act_scores = receptor_scores[:len(receptor_scores)//2]
                                zinc_scores = receptor_scores[len(receptor_scores)//2:]
                                logger.info(f"    Split receptors: {len(act_scores)} test, {len(zinc_scores)} decoy")
                                logger.info("    Expected: EF@1% should be ~1.0 (no enrichment = random)")
    
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")
    
    # ========================================================================
    # Compute Metrics
    # ========================================================================
    logger.info("\n[3/6] Computing metrics...")
    
    # Handle empty results (e.g., negative_control not implemented)
    if len(act_scores) == 0 and len(zinc_scores) == 0:
        logger.warning("  No scores computed - skipping metrics")
        metrics_dict = {
            "experiment_type": experiment_type,
            "target": target,
            "run_name": run_name,
            "status": "skipped",
            "reason": "Experiment not implemented or no data"
        }
        metrics_path = metrics_dir / "metrics.json"
        with metrics_path.open("w") as f:
            json.dump(metrics_dict, f, indent=2)
        logger.info(f"  Saved skip status: {metrics_path}")
        
        elapsed = time.time() - start_time
        logger.info(f"\n[6/6] Phase 5 completed (skipped) in {elapsed:.1f}s")
        logger.info("="*80)
        return
    
    # Create labels (1 = active, 0 = decoy)
    labels = np.concatenate([np.ones(len(act_scores)), np.zeros(len(zinc_scores))])
    all_scores = np.concatenate([act_scores, zinc_scores])
    
    # Compute metrics
    ef1 = ef_at_k_percent(labels, all_scores, k_percent=1.0)
    ef5 = ef_at_k_percent(labels, all_scores, k_percent=5.0)
    roc = roc_auc(labels, all_scores)
    pr = pr_auc(labels, all_scores)
    
    logger.info(f"  EF@1%: {ef1:.2f}")
    logger.info(f"  EF@5%: {ef5:.2f}")
    logger.info(f"  ROC-AUC: {roc:.4f}")
    logger.info(f"  PR-AUC: {pr:.4f}")
    
    elapsed = time.time() - start_time
    
    # ========================================================================
    # Save Results
    # ========================================================================
    logger.info("\n[4/6] Saving ranked scores...")
    
    # Create ranked scores DataFrame
    all_smiles = list(df_actives[smiles_col].values) + list(df_zinc[zinc_smiles_col].values)
    ranked_df = pd.DataFrame({
        "SMILES": all_smiles,
        "score": all_scores,
        "label": labels,
    })
    ranked_df = ranked_df.sort_values("score", ascending=False).reset_index(drop=True)
    ranked_df.to_csv(artifacts_dir / "ranked_scores.csv", index=False)
    logger.info(f"  Saved: {artifacts_dir / 'ranked_scores.csv'}")
    
    logger.info("\n[5/6] Saving summary...")
    
    summary = {
        "run_name": run_name,
        "experiment_type": experiment_type,
        "description": cfg["description"],
        "target": target,
        "random_seed": random_seed,
        "affinity_cutoff_nM": affinity_cutoff_nM,
        "n_actives": int(len(act_scores)),
        "n_decoys": int(len(zinc_scores)),
        "n_mf_reference": int(len(df_mf)),
        "ef_1%": float(ef1),
        "ef_5%": float(ef5),
        "roc_auc": float(roc),
        "pr_auc": float(pr),
        "elapsed_time_s": float(elapsed),
        "config": cfg,
    }
    
    with (metrics_dir / "metrics.json").open("w") as f:
        json.dump(summary, f, indent=2)
    
    with (logs_dir / "phase5_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"  Saved metrics and summary")
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
