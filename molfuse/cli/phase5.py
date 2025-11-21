#!/usr/bin/env python3
"""
Phase 5: Validation & Baseline Experiments

Three critical controls to validate Phase 1/2 methodology:
    1. Raw Fingerprints Baseline: 1-NN in raw ECFP4 space (no UMAP, Jaccard distance)
    2. Raw Descriptors Baseline: 1-NN in high-D scaled space (no UMAP, Euclidean distance)
    3. Negative Control: Non-kinase actives vs kinase model (tests database bias)

Research Questions:
    1. Is dimensionality reduction (UMAP) necessary for fingerprints?
       - Phase 1 fingerprints: ECFP4 → UMAP(20D) → 1-NN(Euclidean)
       - Phase 5 raw_fingerprints: ECFP4 → 1-NN(Jaccard) in 2048D space
    2. Is dimensionality reduction (UMAP) necessary for molecular descriptors?
       - Phase 1 features: Mordred → UMAP(2D) → 1-NN(Euclidean)
       - Phase 5 raw_descriptors: Mordred → 1-NN(Euclidean) in ~1600D space
    3. Does the kinase model show database bias when tested on non-kinase ChEMBL compounds?
       - Expected: EF@1% ≈ 1.0 (random performance) for non-kinase compounds

Methodology:
    - All experiments use Phase 1/2's optimal affinity cutoff (100 nM) for MF cloud scoring
    - Cutoff applied AFTER transformations (for scoring only), not before
    - Raw fingerprints and raw descriptors load Phase 1's best ABL1 models and skip UMAP
    - Negative control tests kinase model on non-kinase KW sets (cross-validation approach)
    - All experiments use Phase 1's preprocessing (imputer/scaler for features, parser for fingerprints)

Usage:
    python -m molfuse.cli.phase5 \\
        --config configs/phase5_grid/phase5_tanimoto_rep1.json \\
        --workspace experiment_workspace_v4
"""
from __future__ import annotations

import argparse
import gc
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
from sklearn.impute import SimpleImputer
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
    from rdkit import DataStructs
    
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


def tanimoto_similarity_max(query_fps: np.ndarray, ref_fps: np.ndarray, chunk_size: int = 10000) -> np.ndarray:
    """
    Compute max Tanimoto similarity for each query against reference set.
    Uses chunking to avoid memory overflow for large query sets.
    
    Tanimoto = |A ∩ B| / |A ∪ B| = (A · B) / (|A| + |B| - A · B)
    
    Args:
        query_fps: Binary fingerprints (n_query, n_bits)
        ref_fps: Binary fingerprints (n_ref, n_bits)
        chunk_size: Number of queries to process at once (default: 10000)
    
    Returns:
        Max Tanimoto similarity for each query (n_query,)
    """
    n_query = query_fps.shape[0]
    max_similarities = np.zeros(n_query, dtype=np.float32)
    
    # Process in chunks to avoid memory overflow
    for i in range(0, n_query, chunk_size):
        end_idx = min(i + chunk_size, n_query)
        query_chunk = query_fps[i:end_idx]
        
        # Compute dot product (intersection)
        intersect = query_chunk @ ref_fps.T  # (chunk_size, n_ref)
        
        # Compute union
        query_popcount = query_chunk.sum(axis=1, keepdims=True)  # (chunk_size, 1)
        ref_popcount = ref_fps.sum(axis=1, keepdims=True).T  # (1, n_ref)
        union = query_popcount + ref_popcount - intersect
        
        # Tanimoto = intersection / union
        tanimoto = intersect / (union + 1e-10)  # Add epsilon to avoid division by zero
        
        # Store max similarity for this chunk
        max_similarities[i:end_idx] = tanimoto.max(axis=1)
    
    return max_similarities


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
    # Load Datasets (experiment-specific)
    # ========================================================================
    logger.info("\n[1/6] Loading datasets...")
    
    # Determine which datasets to load based on experiment type
    if experiment_type == "tanimoto":
        # Tanimoto needs features CSV for actives/affinity filtering
        # AND fingerprints CSV for scoring
        mf_features_csv = Path(cfg["mf_features_csv"])
        zinc_features_csv = Path(cfg["zinc_features_csv"])
        mf_fingerprints_csv = Path(cfg["mf_fingerprints_csv"])
        zinc_fingerprints_csv = Path(cfg["zinc_fingerprints_csv"])
        logger.info(f"  Experiment type: {experiment_type} (features for filtering + fingerprints for scoring)")
    elif experiment_type == "raw_descriptors":
        # Raw descriptors uses features only
        mf_features_csv = Path(cfg["mf_features_csv"])
        zinc_features_csv = Path(cfg["zinc_features_csv"])
        logger.info(f"  Experiment type: {experiment_type} (features)")
    elif experiment_type == "negative_control":
        # Negative control loads KW set (not MF), but needs MF for Phase 1 reconstruction
        # We'll load KW set in the experiment-specific section
        zinc_features_csv = Path(cfg["zinc_features_csv"])
        logger.info(f"  Experiment type: {experiment_type} (KW set + ZINC features)")
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")
    
    # Load MF features for actives extraction and affinity filtering
    # (not for negative_control, which loads KW set instead)
    if experiment_type != "negative_control":
        logger.info(f"Loading MF features for actives: {mf_features_csv}")
        df_mf_features = pd.read_csv(mf_features_csv, low_memory=False)
        logger.info(f"  Loaded: {len(df_mf_features):,} rows")
        
        # Extract actives FIRST (before any filtering) to get their SMILES
        df_actives = df_mf_features[df_mf_features["accession"] == target].copy()
        logger.info(f"Actives for {target}: {len(df_actives):,} rows")
        
        # Deduplicate actives by SMILES
        smiles_col = "canonical_smiles" if "canonical_smiles" in df_mf_features.columns else "SMILES"
        agg_dict = {}
        for col in df_actives.columns:
            if col == smiles_col:
                continue
            elif col == "Standard Value (nM)":
                agg_dict[col] = "median"
            else:
                agg_dict[col] = "first"
        before_act = len(df_actives)
        df_actives = df_actives.groupby(smiles_col, as_index=False).agg(agg_dict)
        logger.info(f"  Deduplicated actives: {before_act:,} -> {len(df_actives):,} (removed {before_act-len(df_actives):,})")
        
        # Get active SMILES set for exclusion
        active_smiles_set = set(df_actives[smiles_col].dropna())
        logger.info(f"  Active SMILES to exclude: {len(active_smiles_set):,}")
        
        # Filter MF: Remove target accession AND remove any molecule that appears in actives
        # This handles promiscuous binders (same molecule binding multiple kinases)
        df_mf_filtered = df_mf_features[df_mf_features["accession"] != target].copy()
        logger.info(f"  After removing target accession {target}: {len(df_mf_filtered):,} rows")
        
        # Remove molecules by SMILES (handles multi-target binders)
        before_smiles = len(df_mf_filtered)
        df_mf_filtered = df_mf_filtered[~df_mf_filtered[smiles_col].isin(active_smiles_set)].copy()
        logger.info(f"  After removing active SMILES: {len(df_mf_filtered):,} rows (removed {before_smiles - len(df_mf_filtered):,} promiscuous binders)")
        
        # DO NOT apply affinity cutoff here - wait until after transformations (like Phase 1/2)
        # Cutoff is for scoring only, not for data preprocessing
        
        # Deduplicate MF by SMILES (median aggregation)
        before = len(df_mf_filtered)
        df_mf_filtered = df_mf_filtered.groupby(smiles_col, as_index=False).agg(agg_dict)
        logger.info(f"  Deduplicated MF: {before:,} -> {len(df_mf_filtered):,} (removed {before-len(df_mf_filtered):,})")
        
        # CRITICAL: Verify actives were actually removed from MF cloud (data leakage check)
        mf_smiles_set = set(df_mf_filtered[smiles_col].dropna())
        overlap_mf_actives = active_smiles_set.intersection(mf_smiles_set)
        if len(overlap_mf_actives) > 0:
            logger.error(f"  CRITICAL ERROR: {len(overlap_mf_actives)} actives found in MF cloud (DATA LEAKAGE!)")
            logger.error(f"  This should never happen - actives must be excluded from MF")
            raise RuntimeError("Data leakage detected: actives found in MF reference set")
        logger.info(f"  ✓ Verified: Zero overlap between actives and MF cloud (no data leakage)")
        logger.info(f"  Note: Affinity cutoff (≤{affinity_cutoff_nM} nM) will be applied to MF for scoring only (after transformations)")
        
        # For Tanimoto: load fingerprints separately for scoring
        if experiment_type == "tanimoto":
            logger.info(f"Loading MF fingerprints for scoring: {mf_fingerprints_csv}")
            df_mf_fingerprints = pd.read_csv(mf_fingerprints_csv, low_memory=False)
            logger.info(f"  Loaded: {len(df_mf_fingerprints):,} rows")
            
            # Filter fingerprints to match filtered features (by SMILES)
            mf_smiles_keep = set(df_mf_filtered[smiles_col].values)
            fp_smiles_col = "canonical_smiles" if "canonical_smiles" in df_mf_fingerprints.columns else "SMILES"
            df_mf_fingerprints = df_mf_fingerprints[df_mf_fingerprints[fp_smiles_col].isin(mf_smiles_keep)].copy()
            logger.info(f"  After filtering to match MF features: {len(df_mf_fingerprints):,} rows")
            
            # Load actives fingerprints
            logger.info(f"Loading actives fingerprints for scoring: {mf_fingerprints_csv}")
            df_actives_fingerprints = pd.read_csv(mf_fingerprints_csv, low_memory=False)
            actives_smiles_keep = set(df_actives[smiles_col].values)
            df_actives_fingerprints = df_actives_fingerprints[df_actives_fingerprints[fp_smiles_col].isin(actives_smiles_keep)].copy()
            logger.info(f"  Actives fingerprints: {len(df_actives_fingerprints):,} rows")
            
            # Load ZINC fingerprints
            logger.info(f"Loading ZINC fingerprints for scoring: {zinc_fingerprints_csv}")
            df_zinc_fingerprints = pd.read_csv(zinc_fingerprints_csv, low_memory=False)
            logger.info(f"  Loaded: {len(df_zinc_fingerprints):,} rows")
            zinc_fp_smiles_col = "canonical_smiles" if "canonical_smiles" in df_zinc_fingerprints.columns else "SMILES"
            
            # Deduplicate ZINC fingerprints
            before_zinc_fp = len(df_zinc_fingerprints)
            df_zinc_fingerprints = df_zinc_fingerprints.drop_duplicates(subset=[zinc_fp_smiles_col], keep="first").copy()
            logger.info(f"  Deduplicated ZINC fingerprints: {before_zinc_fp:,} -> {len(df_zinc_fingerprints):,}")
            
            # Remove MF-ZINC overlap in fingerprints
            before_overlap = len(df_zinc_fingerprints)
            df_zinc_fingerprints = df_zinc_fingerprints[~df_zinc_fingerprints[zinc_fp_smiles_col].isin(mf_smiles_keep)].copy()
            logger.info(f"  Removed MF-ZINC overlap in fingerprints: {before_overlap:,} -> {len(df_zinc_fingerprints):,}")
            
            # Remove actives from ZINC fingerprints
            before_act_overlap = len(df_zinc_fingerprints)
            df_zinc_fingerprints = df_zinc_fingerprints[~df_zinc_fingerprints[zinc_fp_smiles_col].isin(actives_smiles_keep)].copy()
            logger.info(f"  Removed actives from ZINC fingerprints: {before_act_overlap:,} -> {len(df_zinc_fingerprints):,}")
            
            # CRITICAL: Verify ZINC fingerprints have zero overlap with actives
            zinc_fp_smiles_set = set(df_zinc_fingerprints[zinc_fp_smiles_col].dropna())
            overlap_zinc_actives = actives_smiles_keep.intersection(zinc_fp_smiles_set)
            if len(overlap_zinc_actives) > 0:
                logger.error(f"  CRITICAL ERROR: {len(overlap_zinc_actives)} actives found in ZINC fingerprints (DATA LEAKAGE!)")
                raise RuntimeError("Data leakage detected: actives found in ZINC decoy set")
            logger.info(f"  ✓ Verified: Zero overlap between actives and ZINC fingerprints (no data leakage)")
            
            # Assign to standard names for downstream processing
            df_mf = df_mf_fingerprints
            df_zinc = df_zinc_fingerprints
            df_actives = df_actives_fingerprints
            # Preserve SMILES column names for Tanimoto
            smiles_col = fp_smiles_col
            zinc_smiles_col = zinc_fp_smiles_col
        else:
            # For raw_descriptors: use filtered features directly
            df_mf = df_mf_filtered
            
            # Load ZINC features
            logger.info(f"Loading ZINC features: {zinc_features_csv}")
            df_zinc = pd.read_csv(zinc_features_csv, low_memory=False)
            logger.info(f"  Loaded: {len(df_zinc):,} rows")
            
            # Deduplicate ZINC
            zinc_smiles_col = "canonical_smiles" if "canonical_smiles" in df_zinc.columns else "SMILES"
            before_zinc = len(df_zinc)
            df_zinc = df_zinc.drop_duplicates(subset=[zinc_smiles_col], keep="first").copy()
            logger.info(f"  Deduplicated ZINC: {before_zinc:,} -> {len(df_zinc):,}")
            
            # Remove MF-ZINC overlap
            mf_smiles = set(df_mf[smiles_col].values)
            before_overlap = len(df_zinc)
            df_zinc = df_zinc[~df_zinc[zinc_smiles_col].isin(mf_smiles)].copy()
            logger.info(f"  Removed MF-ZINC overlap: {before_overlap:,} -> {len(df_zinc):,}")
            
            # Remove actives from ZINC
            act_smiles = set(df_actives[smiles_col].values)
            before_act_overlap = len(df_zinc)
            df_zinc = df_zinc[~df_zinc[zinc_smiles_col].isin(act_smiles)].copy()
            logger.info(f"  Removed actives from ZINC: {before_act_overlap:,} -> {len(df_zinc):,}")
            
            # CRITICAL: Verify ZINC has zero overlap with actives (data leakage check)
            zinc_smiles_set = set(df_zinc[zinc_smiles_col].dropna())
            overlap_zinc_actives = active_smiles_set.intersection(zinc_smiles_set)
            if len(overlap_zinc_actives) > 0:
                logger.error(f"  CRITICAL ERROR: {len(overlap_zinc_actives)} actives found in ZINC decoys (DATA LEAKAGE!)")
                raise RuntimeError("Data leakage detected: actives found in ZINC decoy set")
            logger.info(f"  ✓ Verified: Zero overlap between actives and ZINC (no data leakage)")
    
    # ========================================================================
    # Experiment-specific scoring
    # ========================================================================
    logger.info(f"\n[2/6] Running {experiment_type} experiment...")
    
    if experiment_type == "tanimoto":
        logger.info("Raw Fingerprint Baseline: 1-NN in raw ECFP4 space (no UMAP)")
        logger.info("  This tests whether dimensionality reduction (UMAP) is necessary for fingerprints")
        logger.info("  Scores in raw fingerprint space using Jaccard distance")
        
        # Use pre-loaded and filtered dataframes from Step 1
        # df_mf, df_zinc, df_actives are already assigned to the fingerprint dataframes
        logger.info(f"  Using loaded datasets: MF={len(df_mf):,}, Actives={len(df_actives):,}, ZINC={len(df_zinc):,}")
        
        # Parse ECFP4 fingerprints
        logger.info("  Parsing ECFP4 fingerprints...")
        fp_col = "ECFP4" if "ECFP4" in df_mf.columns else "Fingerprint"
        
        def parse_fp_series_local(series: pd.Series) -> Tuple[np.ndarray, pd.Index]:
            """Parse fingerprint strings to binary arrays."""
            import re
            ser = series.astype(str).str.strip().str.replace('"', '', regex=False)
            mask_nonempty = ser.notna() & (ser.str.len() > 0)
            ser = ser[mask_nonempty]
            
            parsed_list = []
            valid_idx = []
            expected_len = None
            for idx, s in ser.items():
                try:
                    ss = s.replace(" ", "").strip()
                    if ss.startswith("[") and ss.endswith("]"):
                        ss = ss[1:-1]
                    ss = re.sub(r"[^01,]", "", ss)
                    ss = ss.strip(",")
                    if not ss:
                        continue
                    arr = np.fromstring(ss, sep=",", dtype=np.uint8)
                    if expected_len is None:
                        expected_len = int(arr.shape[0])
                    if arr.shape[0] != expected_len or expected_len == 0:
                        continue
                    parsed_list.append(arr)
                    valid_idx.append(idx)
                except Exception:
                    continue
            if not parsed_list:
                raise RuntimeError("Could not parse any fingerprints")
            return np.vstack(parsed_list).astype(np.float32), pd.Index(valid_idx)
        
        # Parse MF
        FP_mf_full, idx_mf = parse_fp_series_local(df_mf[fp_col])
        # Align DataFrame with parsed fingerprints
        df_mf = df_mf.loc[idx_mf].copy()
        
        # Parse ZINC
        FP_zinc, idx_zinc = parse_fp_series_local(df_zinc[fp_col])
        df_zinc = df_zinc.loc[idx_zinc].copy()
        
        # Parse Actives
        if len(df_actives) == 0:
            FP_actives = np.zeros((0, FP_mf_full.shape[1]), dtype=np.float32)
        else:
            FP_actives, idx_act = parse_fp_series_local(df_actives[fp_col])
            df_actives = df_actives.loc[idx_act].copy()
            
        logger.info(f"  Parsed fingerprints: MF={FP_mf_full.shape}, Actives={FP_actives.shape}, ZINC={FP_zinc.shape}")
        
        # Apply affinity cutoff to MF for scoring only (like Phase 1/2)
        if "Standard Value (nM)" in df_mf.columns:
            mask_cut = pd.to_numeric(df_mf["Standard Value (nM)"], errors="coerce") <= affinity_cutoff_nM
            FP_mf = FP_mf_full[mask_cut.to_numpy(dtype=bool)]
            logger.info(f"  Applying affinity cutoff (≤{affinity_cutoff_nM} nM) to MF for scoring only")
            logger.info(f"    MF for scoring: {FP_mf.shape[0]:,} / {FP_mf_full.shape[0]:,} compounds")
        else:
            logger.warning("  No affinity column found; using full MF cloud for scoring")
            FP_mf = FP_mf_full
        
        # Score via 1-NN in raw fingerprint space using Jaccard distance
        logger.info("  Computing 1-NN scores in raw fingerprint space (Jaccard distance)...")
        if FP_mf.shape[0] == 0:
            logger.warning("  MF cloud empty after cutoff! Cannot score.")
            act_scores = np.array([])
            zinc_scores = np.array([])
        else:
            act_scores, _ = nn_min_distance_scores(FP_mf, FP_actives, metric="jaccard")
            zinc_scores, _ = nn_min_distance_scores(FP_mf, FP_zinc, metric="jaccard")
            logger.info(f"    Actives: min={act_scores.min():.4f}, max={act_scores.max():.4f}, mean={act_scores.mean():.4f}")
            logger.info(f"    ZINC: min={zinc_scores.min():.4f}, max={zinc_scores.max():.4f}, mean={zinc_scores.mean():.4f}")
        
    elif experiment_type == "raw_descriptors":
        logger.info("Raw Descriptor Baseline: 1-NN in high-D scaled space (no UMAP)")
        logger.info("  This tests whether dimensionality reduction (UMAP) is necessary")
        logger.info("  Uses Phase 1/4 best model preprocessing, scores in high-D feature space")
        
        # Load Phase 1/4 best model artifacts (imputer + scaler, NOT UMAP)
        logger.info("  Loading best model artifacts...")
        phase1_model_dir = Path(cfg.get("phase2_best_model_dir", "experiment_workspace_v4/phase1"))
        
        # Select replicate deterministically based on run-specific seed
        rng = np.random.RandomState(random_seed)
        replicate = rng.randint(1, 6)  # Random integer from 1 to 5
        
        # Support custom model directory templates (for Phase 4 expansion)
        model_dir_template = cfg.get("model_dir_template")
        if model_dir_template:
            # Use target_short from config if available, else try to derive from target string
            target_short = cfg.get("target_short")
            if not target_short:
                target_short = target.split("_")[-1] if "_" in target else target
            best_model_name = model_dir_template.format(target=target_short, replicate=replicate)
        else:
            # Legacy Phase 1 format
            best_model_name = f"ABL1_UMAP_features_2d_nn10_md0p01_rep{replicate}"
            
        best_phase1_dir = phase1_model_dir / best_model_name
        logger.info(f"    Selected replicate: {replicate} (seed: {random_seed})")
        
        if not best_phase1_dir.exists():
            logger.error(f"  Phase 1 model not found: {best_phase1_dir}")
            logger.info("  Creating empty results to mark as skipped")
            act_scores = np.array([])
            zinc_scores = np.array([])
        else:
            logger.info(f"    Using Phase 1 ABL1 model: {best_phase1_dir.name}")
            
            # Load Phase 1 artifacts directly (use different variable name to avoid shadowing Phase 5's artifacts_dir)
            phase1_artifacts_dir = best_phase1_dir / "artifacts"
            
            if not phase1_artifacts_dir.exists():
                logger.error(f"    Phase 1 artifacts not found: {phase1_artifacts_dir}")
                logger.info("    Creating empty results to mark as skipped")
                act_scores = np.array([])
                zinc_scores = np.array([])
            else:
                logger.info(f"    Loading Phase 1 artifacts from: {phase1_artifacts_dir.parent.name}")
                
                # Load scaler (required)
                scaler_path = phase1_artifacts_dir / "scaler.joblib"
                
                if not scaler_path.exists():
                    logger.error(f"    Scaler not found: {scaler_path}")
                    act_scores = np.array([])
                    zinc_scores = np.array([])
                else:
                    phase2_scaler = joblib.load(scaler_path)
                    
                    # Load or reconstruct imputer
                    imputer_path = phase1_artifacts_dir / "imputer.joblib"
                    if imputer_path.exists():
                        phase2_imputer = joblib.load(imputer_path)
                        logger.info("    Loaded imputer + scaler from Phase 1 pipeline")
                    else:
                        logger.warning(f"    Imputer not found at {imputer_path} - will reconstruct from Phase 1/4 data")
                        # Set to None; will reconstruct below after loading Phase 1 data
                        phase2_imputer = None
                    
                    # Reconstruct Phase 1's feature list (same preprocessing as Phase 1/2)
                    # Load Phase 1 config to get exact preprocessing
                    phase1_summary_path = phase1_artifacts_dir.parent / "logs" / "phase1_summary.json"
                    if not phase1_summary_path.exists():
                        logger.error(f"    Phase 1 summary not found: {phase1_summary_path}")
                        act_scores = np.array([])
                        zinc_scores = np.array([])
                    else:
                        with open(phase1_summary_path, "r") as f:
                            phase1_summary = json.load(f)
                        phase1_cfg = phase1_summary["config"]
                        
                        # Load Phase 1 MF+ZINC to reconstruct feature selection
                        logger.info("    Reconstructing Phase 1/2 feature selection...")
                        df_mf_p1 = pd.read_csv(phase1_cfg["mf_features_csv"], low_memory=False)
                        df_zinc_p1 = pd.read_csv(phase1_cfg["zinc_features_csv"], low_memory=False)
                        
                        # Apply Phase 1's preprocessing (target exclusion, dedup, zero-variance filter)
                        phase1_target = phase1_cfg.get("target", target)
                        if "accession" in df_mf_p1.columns:
                            df_mf_p1 = df_mf_p1[df_mf_p1["accession"] != phase1_target].copy()
                        
                        smiles_col_p1 = "canonical_smiles" if "canonical_smiles" in df_mf_p1.columns else "SMILES"
                        df_mf_p1 = df_mf_p1.drop_duplicates(subset=[smiles_col_p1], keep="first").copy()
                        df_zinc_p1 = df_zinc_p1.drop_duplicates(subset=[smiles_col_p1], keep="first").copy()
                        
                        # Feature selection + zero-variance filtering
                        feat_cols_mf = select_feature_columns(df_mf_p1)
                        feat_cols_zinc = select_feature_columns(df_zinc_p1)
                        common_feats = [c for c in feat_cols_mf if c in feat_cols_zinc]
                        
                        df_train_check = pd.concat([df_mf_p1[common_feats], df_zinc_p1[common_feats]], axis=0, ignore_index=True)
                        phase2_features = remove_zero_variance_features(df_train_check, common_feats, variance_threshold=1e-12)
                        
                        # Reconstruct imputer if it was missing (Phase 4 didn't save it)
                        if phase2_imputer is None:
                            logger.info("    Reconstructing imputer from Phase 1/4 MF+ZINC data (median strategy)...")
                            X_train_for_imputer = df_train_check[phase2_features].to_numpy(dtype=np.float64)
                            phase2_imputer = SimpleImputer(strategy='median', copy=True)
                            phase2_imputer.fit(X_train_for_imputer)
                            logger.info("    ✓ Imputer reconstructed successfully")
                        
                        del df_train_check, df_mf_p1, df_zinc_p1
                        gc.collect()
                        logger.info(f"    Phase 1/2 features (after zero-variance filter): {len(phase2_features)}")
                        
                        # Transform Phase 5's data through Phase 2's preprocessing
                        logger.info("    Transforming Phase 5 data through Phase 2 preprocessing...")
                        X_mf_raw = df_mf.reindex(columns=phase2_features).to_numpy(dtype=np.float64)
                        X_act_raw = df_actives.reindex(columns=phase2_features).to_numpy(dtype=np.float64)
                        X_zinc_raw = df_zinc.reindex(columns=phase2_features).to_numpy(dtype=np.float64)
                        
                        # Remove infinities
                        X_mf_raw[~np.isfinite(X_mf_raw)] = np.nan
                        X_act_raw[~np.isfinite(X_act_raw)] = np.nan
                        X_zinc_raw[~np.isfinite(X_zinc_raw)] = np.nan
                        
                        # Transform through Phase 2's pipeline (NO UMAP)
                        X_mf_scaled = phase2_scaler.transform(phase2_imputer.transform(X_mf_raw))
                        X_act_scaled = phase2_scaler.transform(phase2_imputer.transform(X_act_raw))
                        X_zinc_scaled = phase2_scaler.transform(phase2_imputer.transform(X_zinc_raw))
                        
                        logger.info(f"    Transformed shapes: MF={X_mf_scaled.shape}, actives={X_act_scaled.shape}, ZINC={X_zinc_scaled.shape}")
                        
                        # Apply affinity cutoff to MF for scoring only (like Phase 1/2)
                        if "Standard Value (nM)" in df_mf_filtered.columns:
                            mask_cut = pd.to_numeric(df_mf_filtered["Standard Value (nM)"], errors="coerce") <= affinity_cutoff_nM
                            X_mf_for_scoring = X_mf_scaled[mask_cut.to_numpy(dtype=bool)]
                            logger.info(f"  Applying affinity cutoff (≤{affinity_cutoff_nM} nM) to MF for scoring only")
                            logger.info(f"    MF for scoring: {X_mf_for_scoring.shape[0]:,} / {X_mf_scaled.shape[0]:,} compounds")
                        else:
                            logger.warning("  No affinity column found; using full MF cloud for scoring")
                            X_mf_for_scoring = X_mf_scaled
                        
                        # Score via 1-NN in high-D scaled space (NO UMAP transform)
                        logger.info("  Computing 1-NN scores in high-D scaled feature space...")
                        act_scores, _ = nn_min_distance_scores(X_mf_for_scoring, X_act_scaled, metric="euclidean")
                        zinc_scores, _ = nn_min_distance_scores(X_mf_for_scoring, X_zinc_scaled, metric="euclidean")
                        logger.info(f"    Actives: min={act_scores.min():.4f}, max={act_scores.max():.4f}, mean={act_scores.mean():.4f}")
                        logger.info(f"    ZINC: min={zinc_scores.min():.4f}, max={zinc_scores.max():.4f}, mean={zinc_scores.mean():.4f}")
        
    elif experiment_type == "negative_control":
        logger.info("Negative Control: Non-kinase KW set vs kinase model")
        logger.info("  This tests for database bias (kinase model scoring non-kinase compounds)")
        logger.info("  Expected result: EF@1% ≈ 1.0 (random performance, no enrichment)")
        
        # Load Phase 1/4 best model FIRST to get the correct MF source
        logger.info("  Loading best model...")
        phase1_model_dir = Path(cfg.get("phase2_best_model_dir", "experiment_workspace_v4/phase1"))
        
        # Select replicate deterministically based on run-specific seed
        rng = np.random.RandomState(random_seed)
        replicate = rng.randint(1, 6)  # Random integer from 1 to 5
        
        # Support custom model directory templates (for Phase 4 expansion)
        model_dir_template = cfg.get("model_dir_template")
        if model_dir_template:
            # Use target_short from config if available, else try to derive from target string
            target_short = cfg.get("target_short")
            if not target_short:
                target_short = target.split("_")[-1] if "_" in target else target
            best_model_name = model_dir_template.format(target=target_short, replicate=replicate)
        else:
            # Legacy Phase 1 format
            best_model_name = f"ABL1_UMAP_features_2d_nn10_md0p01_rep{replicate}"
            
        best_phase1_dir = phase1_model_dir / best_model_name
        logger.info(f"    Selected replicate: {replicate} (seed: {random_seed})")
        
        if not best_phase1_dir.exists():
            logger.error("  No Phase 1 ABL1 model found")
            act_scores = np.array([])
            zinc_scores = np.array([])
        else:
            logger.info(f"    Using Phase 1 ABL1 model: {best_phase1_dir.name}")
            
            # Load Phase 1 artifacts
            phase1_artifacts_dir = best_phase1_dir / "artifacts"
            phase1_summary_path = best_phase1_dir / "logs" / "phase1_summary.json"
            
            if not phase1_artifacts_dir.exists() or not phase1_summary_path.exists():
                logger.error(f"    Phase 1 artifacts or summary not found: {best_phase1_dir}")
                act_scores = np.array([])
                zinc_scores = np.array([])
            else:
                # Load Phase 1 config to get MF source
                with open(phase1_summary_path, "r") as f:
                    phase1_summary = json.load(f)
                phase1_cfg = phase1_summary["config"]
                
                # Load kinase MF cloud using path from Phase 1 config
                mf_features_csv = Path(phase1_cfg["mf_features_csv"])
                logger.info(f"  Loading kinase MF features from Phase 1 config: {mf_features_csv}")
                if not mf_features_csv.exists():
                    logger.warning(f"    MF file from Phase 1 config not found: {mf_features_csv}")
                    # Fallback to hardcoded path if original not found (e.g. different machine)
                    fallback_path = Path("output_recalculated_full_datasets/datasets_2d_all/KW-0808_Transferase_affinity_extracted_features.csv")
                    if fallback_path.exists():
                        logger.warning(f"    Falling back to default path: {fallback_path}")
                        mf_features_csv = fallback_path
                    else:
                        logger.error("    MF features CSV not found")
                        act_scores = np.array([])
                        zinc_scores = np.array([])
                        # Skip the rest of this block
                        mf_features_csv = None

                if mf_features_csv:
                    df_mf_kinase = pd.read_csv(mf_features_csv, low_memory=False)
                    logger.info(f"    Loaded kinase MF: {len(df_mf_kinase):,} rows")
                    
                    # Apply same filtering as Phase 1 (remove target, deduplicate)
                    df_mf_kinase = df_mf_kinase[df_mf_kinase["accession"] != target].copy()
                    mf_smiles_col = "canonical_smiles" if "canonical_smiles" in df_mf_kinase.columns else "SMILES"
                    df_mf_kinase = df_mf_kinase.drop_duplicates(subset=[mf_smiles_col], keep="first").copy()
                    mf_smiles = set(df_mf_kinase[mf_smiles_col].dropna())
                    logger.info(f"    Kinase MF cloud (FULL): {len(mf_smiles):,} unique SMILES")
                    
                    # Load non-kinase KW dataset from config
                    kw_csv = Path(cfg.get("negative_control_kw_csv", ""))
                    kw_name = cfg.get("negative_control_kw_name", "Unknown")
                    
                    if not kw_csv or not kw_csv.exists():
                        logger.error(f"  KW dataset not found: {kw_csv}")
                        act_scores = np.array([])
                        zinc_scores = np.array([])
                    else:
                        logger.info(f"  Loading KW set: {kw_name}")
                        df_kw = pd.read_csv(kw_csv, low_memory=False)
                        
                        # Remove any overlap with kinase target
                        if "accession" in df_kw.columns:
                            df_kw = df_kw[df_kw["accession"] != target].copy()
                        
                        # Deduplicate KW set
                        kw_smiles_col = "canonical_smiles" if "canonical_smiles" in df_kw.columns else "SMILES"
                        df_kw = df_kw.drop_duplicates(subset=[kw_smiles_col], keep="first").copy()
                        
                        # Remove MF overlap from KW set
                        before_mf = len(df_kw)
                        df_kw = df_kw[~df_kw[kw_smiles_col].isin(mf_smiles)].copy()
                        logger.info(f"    Removed MF overlap: {before_mf:,} -> {len(df_kw):,} (removed {before_mf-len(df_kw):,})")
                        
                        # Load ZINC features
                        logger.info("  Loading ZINC features...")
                        zinc_features_csv = Path(cfg["zinc_features_csv"])
                        df_zinc = pd.read_csv(zinc_features_csv, low_memory=False)
                        
                        # Deduplicate ZINC
                        zinc_smiles_col = "canonical_smiles" if "canonical_smiles" in df_zinc.columns else "SMILES"
                        df_zinc = df_zinc.drop_duplicates(subset=[zinc_smiles_col], keep="first").copy()
                        
                        # Remove MF overlap from ZINC
                        before_overlap = len(df_zinc)
                        df_zinc = df_zinc[~df_zinc[zinc_smiles_col].isin(mf_smiles)].copy()
                        logger.info(f"    Removed MF overlap from ZINC: {before_overlap:,} -> {len(df_zinc):,}")
                        
                        # Remove KW overlap from ZINC
                        kw_smiles = set(df_kw[kw_smiles_col].dropna())
                        before_kw_overlap = len(df_zinc)
                        df_zinc = df_zinc[~df_zinc[zinc_smiles_col].isin(kw_smiles)].copy()
                        logger.info(f"    Removed KW overlap from ZINC: {before_kw_overlap:,} -> {len(df_zinc):,}")
                        
                        # Load scaler and UMAP (required)
                        scaler_path = phase1_artifacts_dir / "scaler.joblib"
                        umap_path = phase1_artifacts_dir / "umap_model.joblib"
                        
                        if not all([p.exists() for p in [scaler_path, umap_path]]):
                            logger.error(f"    Missing artifacts in {phase1_artifacts_dir}")
                            act_scores = np.array([])
                            zinc_scores = np.array([])
                        else:
                            phase2_scaler = joblib.load(scaler_path)
                            phase2_umap = joblib.load(umap_path)
                            
                            # Load or reconstruct imputer
                            imputer_path = phase1_artifacts_dir / "imputer.joblib"
                            if imputer_path.exists():
                                phase2_imputer = joblib.load(imputer_path)
                                logger.info("    Loaded imputer + scaler + UMAP from Phase 1 pipeline")
                            else:
                                logger.warning(f"    Imputer not found at {imputer_path} - will reconstruct from Phase 1/4 data")
                                # Set to None; will reconstruct below after loading Phase 1 data
                                phase2_imputer = None
                            
                            # Reconstruct Phase 2 feature selection
                            logger.info("    Reconstructing Phase 2 feature selection...")
                            df_mf_p1 = pd.read_csv(phase1_cfg["mf_features_csv"], low_memory=False)
                            df_zinc_p1 = pd.read_csv(phase1_cfg["zinc_features_csv"], low_memory=False)
                            
                            phase1_target = phase1_cfg.get("target", target)
                            if "accession" in df_mf_p1.columns:
                                df_mf_p1 = df_mf_p1[df_mf_p1["accession"] != phase1_target].copy()
                            
                            smiles_col_p1 = "canonical_smiles" if "canonical_smiles" in df_mf_p1.columns else "SMILES"
                            df_mf_p1 = df_mf_p1.drop_duplicates(subset=[smiles_col_p1], keep="first").copy()
                            df_zinc_p1 = df_zinc_p1.drop_duplicates(subset=[smiles_col_p1], keep="first").copy()
                            
                            feat_cols_mf = select_feature_columns(df_mf_p1)
                            feat_cols_zinc = select_feature_columns(df_zinc_p1)
                            common_feats = [c for c in feat_cols_mf if c in feat_cols_zinc]
                            
                            df_train_check = pd.concat([df_mf_p1[common_feats], df_zinc_p1[common_feats]], axis=0, ignore_index=True)
                            phase2_features = remove_zero_variance_features(df_train_check, common_feats, variance_threshold=1e-12)
                            
                            # Reconstruct imputer if it was missing (Phase 4 didn't save it)
                            if phase2_imputer is None:
                                logger.info("    Reconstructing imputer from Phase 1/4 MF+ZINC data (median strategy)...")
                                X_train_for_imputer = df_train_check[phase2_features].to_numpy(dtype=np.float64)
                                phase2_imputer = SimpleImputer(strategy='median', copy=True)
                                phase2_imputer.fit(X_train_for_imputer)
                                logger.info("    ✓ Imputer reconstructed successfully")
                            
                            del df_train_check, df_mf_p1, df_zinc_p1
                            gc.collect()
                            logger.info(f"    Phase 2 features: {len(phase2_features)}")
                            
                            # Transform FULL kinase MF through Phase 2 pipeline
                            logger.info("    Transforming kinase MF through Phase 2 pipeline...")
                            X_mf_kinase_raw = df_mf_kinase.reindex(columns=phase2_features).to_numpy(dtype=np.float64)
                            X_mf_kinase_raw[~np.isfinite(X_mf_kinase_raw)] = np.nan
                            X_mf_kinase_scaled = phase2_scaler.transform(phase2_imputer.transform(X_mf_kinase_raw))
                            X_mf_kinase_embed = phase2_umap.transform(X_mf_kinase_scaled)
                            logger.info(f"    Kinase MF embedding (FULL): {X_mf_kinase_embed.shape}")
                            
                            # Apply affinity cutoff to kinase MF for scoring only (like Phase 1/2)
                            if "Standard Value (nM)" in df_mf_kinase.columns:
                                mask_cut = pd.to_numeric(df_mf_kinase["Standard Value (nM)"], errors="coerce") <= affinity_cutoff_nM
                                X_mf_embed = X_mf_kinase_embed[mask_cut.to_numpy(dtype=bool)]
                                logger.info(f"  Applying affinity cutoff (≤{affinity_cutoff_nM} nM) to kinase MF for scoring only")
                                logger.info(f"    Kinase MF for scoring: {X_mf_embed.shape[0]:,} / {X_mf_kinase_embed.shape[0]:,} compounds")
                            else:
                                logger.warning("  No affinity column found; using full kinase MF for scoring")
                                X_mf_embed = X_mf_kinase_embed
                            
                            # Transform KW set through Phase 2 pipeline
                            logger.info("    Transforming KW set through Phase 2 pipeline...")
                            X_kw_raw = df_kw.reindex(columns=phase2_features).to_numpy(dtype=np.float64)
                            X_kw_raw[~np.isfinite(X_kw_raw)] = np.nan
                            X_kw_scaled = phase2_scaler.transform(phase2_imputer.transform(X_kw_raw))
                            X_kw_embed = phase2_umap.transform(X_kw_scaled)
                            logger.info(f"    KW embedding: {X_kw_embed.shape}")
                            
                            # Transform ZINC through Phase 2 pipeline
                            logger.info("    Transforming ZINC through Phase 2 pipeline...")
                            X_zinc_raw = df_zinc.reindex(columns=phase2_features).to_numpy(dtype=np.float64)
                            X_zinc_raw[~np.isfinite(X_zinc_raw)] = np.nan
                            X_zinc_scaled = phase2_scaler.transform(phase2_imputer.transform(X_zinc_raw))
                            X_zinc_embed = phase2_umap.transform(X_zinc_scaled)
                            logger.info(f"    ZINC embedding: {X_zinc_embed.shape}")
                            
                            # Score KW (as "actives") and ZINC (as decoys) against kinase MF
                            logger.info("  Scoring KW set against kinase model...")
                            kw_scores, _ = nn_min_distance_scores(X_mf_embed, X_kw_embed, metric="euclidean")
                            zinc_scores, _ = nn_min_distance_scores(X_mf_embed, X_zinc_embed, metric="euclidean")
                            logger.info(f"    KW scores: min={kw_scores.min():.4f}, max={kw_scores.max():.4f}, mean={kw_scores.mean():.4f}")
                            logger.info(f"    ZINC scores: min={zinc_scores.min():.4f}, max={zinc_scores.max():.4f}, mean={zinc_scores.mean():.4f}")
                            
                            # KW set acts as "actives", ZINC as decoys
                            act_scores = kw_scores
                            df_actives = df_kw.copy()
                            smiles_col = kw_smiles_col
                            zinc_smiles_col = "canonical_smiles" if "canonical_smiles" in df_zinc.columns else "SMILES"
                            
                            logger.info(f"    {kw_name}: {len(act_scores):,} compounds")
                            logger.info(f"    ZINC decoys: {len(zinc_scores):,} compounds")
                            logger.info("    Expected: EF@1% ≈ 1.0 (random performance = no database bias)")
    
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
    
    # Compute metrics (NOTE: signature is ef_at_k_percent(scores, labels, k) - scores FIRST!)
    ef1 = ef_at_k_percent(all_scores, labels, k_percent=1.0)
    ef5 = ef_at_k_percent(all_scores, labels, k_percent=5.0)
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
        "n_mf_reference": int(len(df_mf)) if 'df_mf' in locals() else 0,
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
