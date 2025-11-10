#!/usr/bin/env python3
"""
Phase 1 Potency-Stratified Analysis (v4 molfuse) - REWRITTEN

Objective: Compare EF@1% for different potency tiers (High/Medium/Weak) vs overall.

Key Question: Do DR methods preferentially enrich high-potency actives over weak ones?

Potency Tiers (nM):
- High:   0.1 ≤ affinity ≤ 100
- Medium: 100 < affinity ≤ 1,000  
- Weak:   1,000 < affinity ≤ 100,000

Usage:
  python scripts/phase1_stratified_scores.py \
      --workspace_dir experiment_workspace_v4 \
      --phase phase1 \
      --output_dir reporting/phase1_stratified

Output:
  - stratified_summary.csv: One row per run with EF@1% for all tiers
  - stratified_grouped.csv: Aggregated by config (mean ± std across replicates)
  - plots/tier_comparison_*.png: Simple bar charts comparing tiers

Design Principles:
- Simple, clear logic (no overengineering)
- Correct math (use full ranked list including ZINC)
- No metadata pollution (PCA doesn't get UMAP params)
- Minimal output (2 CSVs, not 360)
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Plotting (headless safe)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add molfuse metrics
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from molfuse.metrics.metrics import bedroc, ief

# Global cache for MF data (avoid reloading same file multiple times)
_MF_CACHE: Dict[str, pd.DataFrame] = {}

# Publication-quality styling
matplotlib.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.titleweight": "bold",
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "font.family": "sans-serif",
})


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def setup_logger(out_dir: Path) -> logging.Logger:
    """Create logger with file and console handlers."""
    logger = logging.getLogger("phase1_stratified")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    fh = logging.FileHandler(out_dir / "stratified.log", mode="w")
    sh = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    
    return logger


def read_json(path: Path) -> Optional[dict]:
    """Safely read JSON file."""
    try:
        with path.open("r") as f:
            return json.load(f)
    except Exception:
        return None


def extract_accession(target: str) -> Optional[str]:
    """Extract UniProt accession from target name (e.g., TyrosineProteinKinaseABL1_P00519 → P00519)."""
    m = re.match(r".*_([A-Z0-9]+)$", target)
    return m.group(1) if m else None


# ============================================================================
# DATA LOADING
# ============================================================================

def load_actives_with_affinity(config: dict, target: str, logger: logging.Logger) -> Optional[pd.DataFrame]:
    """
    Load actives dataset with affinity values.
    
    Strategy:
    1. Try actives_features_csv if specified in config
    2. Fall back to filtering mf_features_csv by accession (with caching)
    
    Returns DataFrame with columns: Compound ChEMBL ID, SMILES, Standard Value (nM)
    """
    # Strategy 1: Explicit actives file
    actives_path = config.get("actives_features_csv")
    if actives_path:
        p = Path(actives_path)
        if p.exists():
            try:
                cols_needed = ["Compound ChEMBL ID", "SMILES", "Standard Value (nM)", "accession"]
                df = pd.read_csv(p, usecols=lambda c: c in cols_needed, low_memory=False)
                # Keep only essential columns to reduce memory
                df = df[["Compound ChEMBL ID", "SMILES", "Standard Value (nM)"]].copy()
                return df
            except Exception as e:
                logger.warning(f"Failed to load actives CSV {p}: {e}")
    
    # Strategy 2: Derive from MF file by accession (use cache to avoid reloading)
    mf_path = config.get("mf_features_csv")
    if not mf_path:
        logger.warning("No actives_features_csv or mf_features_csv in config")
        return None
    
    accession = extract_accession(target)
    if not accession:
        logger.warning(f"Cannot extract accession from target: {target}")
        return None
    
    p = Path(mf_path)
    if not p.exists():
        logger.warning(f"MF file not found: {p}")
        return None
    
    # Check cache first
    cache_key = str(p.resolve())
    if cache_key not in _MF_CACHE:
        try:
            cols_needed = ["Compound ChEMBL ID", "SMILES", "Standard Value (nM)", "accession"]
            df_all = pd.read_csv(p, usecols=lambda c: c in cols_needed, low_memory=False)
            _MF_CACHE[cache_key] = df_all
            logger.info(f"Cached MF data from {p} (rows={len(df_all)})")
        except Exception as e:
            logger.warning(f"Failed to load MF CSV {p}: {e}")
            return None
    else:
        df_all = _MF_CACHE[cache_key]
    
    if "accession" not in df_all.columns:
        logger.warning("MF file lacks accession column; cannot filter")
        return None
    
    df = df_all[df_all["accession"] == accession].copy()
    # Keep only essential columns
    df = df[["Compound ChEMBL ID", "SMILES", "Standard Value (nM)"]].copy()
    return df


def load_ranked_scores(run_dir: Path, logger: logging.Logger) -> Optional[pd.DataFrame]:
    """
    Load ranked_scores.csv and ensure proper sorting.
    
    Returns DataFrame sorted by ranking (best first):
    - If score column exists: descending (high score = better)
    - If distance column exists: ascending (low distance = better)
    """
    ranked_path = run_dir / "artifacts" / "ranked_scores.csv"
    if not ranked_path.exists():
        logger.warning(f"ranked_scores.csv not found: {ranked_path}")
        return None
    
    try:
        df = pd.read_csv(ranked_path, low_memory=False)
        
        # Ensure proper sorting
        if "score" in df.columns:
            df = df.sort_values(by="score", ascending=False, kind="stable").reset_index(drop=True)
        elif "distance" in df.columns:
            df = df.sort_values(by="distance", ascending=True, kind="stable").reset_index(drop=True)
        else:
            logger.warning(f"ranked_scores.csv has neither score nor distance column")
            return None
        
        return df
    except Exception as e:
        logger.error(f"Error loading ranked_scores.csv: {e}")
        return None


# ============================================================================
# AFFINITY JOINING & TIER ASSIGNMENT
# ============================================================================

def assign_potency_tier(affinity_nM: float) -> Optional[str]:
    """
    Assign potency tier based on affinity value (nM).
    
    Returns:
        "High" if 0.1 ≤ affinity ≤ 100
        "Medium" if 100 < affinity ≤ 1000
        "Weak" if 1000 < affinity ≤ 100000
        None otherwise (out of range or invalid)
    """
    try:
        val = float(affinity_nM)
    except (ValueError, TypeError):
        return None
    
    if val < 0.1 or val > 100000:
        return None
    
    if val <= 100.0:
        return "High"
    elif val <= 1000.0:
        return "Medium"
    else:
        return "Weak"


def find_column_ignorecase(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Find first matching column name (case-insensitive)."""
    col_map = {c.lower().strip(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in col_map:
            return col_map[candidate.lower()]
    return None


def join_affinity_to_ranked(ranked_df: pd.DataFrame, actives_df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Join affinity values to actives in ranked_df, then assign potency tiers.
    
    Join strategy:
    1. Try Compound ChEMBL ID (case-insensitive, normalized)
    2. Fallback to SMILES
    3. Left join: keep all rows from ranked_df
    4. Assign potency_tier column (NaN for ZINC or actives without affinity)
    
    CRITICAL: This operates on the FULL ranked_df (actives + ZINC), not just actives!
    """
    ranked_df = ranked_df.copy()
    actives_df = actives_df.copy()
    
    # Find join columns
    chembl_col_r = find_column_ignorecase(ranked_df, ["Compound ChEMBL ID", "compound_chembl_id", "molecule_chembl_id"])
    chembl_col_a = find_column_ignorecase(actives_df, ["Compound ChEMBL ID", "compound_chembl_id", "molecule_chembl_id"])
    
    if chembl_col_r and chembl_col_a:
        # ID-based join
        logger.info("Joining by Compound ChEMBL ID")
        ranked_df["_join_key"] = ranked_df[chembl_col_r].astype(str).str.upper().str.strip()
        actives_df["_join_key"] = actives_df[chembl_col_a].astype(str).str.upper().str.strip()
    else:
        # SMILES-based join
        logger.info("Joining by SMILES (ChEMBL ID not available)")
        smiles_col_r = find_column_ignorecase(ranked_df, ["SMILES"])
        smiles_col_a = find_column_ignorecase(actives_df, ["SMILES"])
        
        if not smiles_col_r or not smiles_col_a:
            logger.error("Cannot find join key (no ChEMBL ID or SMILES columns)")
            ranked_df["potency_tier"] = None
            return ranked_df
        
        ranked_df["_join_key"] = ranked_df[smiles_col_r].astype(str).str.strip()
        actives_df["_join_key"] = actives_df[smiles_col_a].astype(str).str.strip()
    
    # Create affinity lookup (one affinity per compound)
    if "Standard Value (nM)" not in actives_df.columns:
        logger.error("actives_df lacks Standard Value (nM) column")
        ranked_df["potency_tier"] = None
        return ranked_df
    
    affinity_lookup = actives_df[["_join_key", "Standard Value (nM)"]].dropna(subset=["_join_key"]).drop_duplicates(subset=["_join_key"])
    
    # Left join: keep all ranked rows
    merged = ranked_df.merge(affinity_lookup, on="_join_key", how="left")
    
    # Assign tiers
    merged["potency_tier"] = merged["Standard Value (nM)"].apply(assign_potency_tier)
    
    # Cleanup
    merged.drop(columns=["_join_key", "Standard Value (nM)"], inplace=True, errors="ignore")
    
    return merged


# ============================================================================
# ENRICHMENT FACTOR CALCULATION
# ============================================================================

def compute_ef_at_percent(ranked_df: pd.DataFrame, tier: Optional[str], top_pct: float) -> Optional[float]:
    """
    Compute Enrichment Factor at top X%.
    
    Args:
        ranked_df: Full ranked list (actives + ZINC), sorted by score/distance
        tier: "High", "Medium", "Weak", or None (for all actives)
        top_pct: Fraction of top molecules (e.g., 0.01 for top 1%)
    
    Returns:
        EF value or None if no actives in the specified tier
    
    Formula:
        EF@p% = (hits / N_actives) / (k / N_total)
        
    where:
        - N_total = total molecules (actives + ZINC)
        - k = ceil(p × N_total) = number of molecules in top p%
        - N_actives = number of actives in the specified tier (or all actives if tier=None)
        - hits = number of tier actives in top k
    """
    if ranked_df.empty:
        return None
    
    N_total = len(ranked_df)
    k = max(1, int(np.ceil(top_pct * N_total)))
    top_k = ranked_df.head(k)
    
    if tier is None:
        # All actives
        mask_all = ranked_df["source"] == "actives"
        mask_top = top_k["source"] == "actives"
        N_actives = mask_all.sum()
        hits = mask_top.sum()
    else:
        # Specific tier
        mask_all = (ranked_df["source"] == "actives") & (ranked_df["potency_tier"] == tier)
        mask_top = (top_k["source"] == "actives") & (top_k["potency_tier"] == tier)
        N_actives = mask_all.sum()
        hits = mask_top.sum()
    
    if N_actives == 0:
        return None
    
    # EF = (hits / N_actives) / (k / N_total)
    ef = (hits / N_actives) / (k / N_total)
    return float(ef)


# ============================================================================
# CONFIG EXTRACTION
# ============================================================================

def extract_run_config(summary_json: dict, metrics_json: dict) -> dict:
    """
    Extract run configuration WITHOUT polluting PCA with UMAP params.
    
    Returns dict with keys:
        - method: "pca" or "umap"
        - dim: dimensionality (int)
        - representation: "features" or "fingerprints"
        - umap_n_neighbors: only if method == "umap"
        - umap_min_dist: only if method == "umap"
    """
    config = summary_json.get("config", {})
    
    method = str(config.get("method", "unknown")).lower()
    
    try:
        dim = int(config.get("dim"))
    except (ValueError, TypeError):
        dim = None
    
    representation = str(config.get("representation", "unknown")).lower()
    
    # CRITICAL: Only extract UMAP params if method is actually UMAP
    if method == "umap":
        umap_params = config.get("umap_params", {})
        nn = umap_params.get("n_neighbors")
        md = umap_params.get("min_dist")
    else:
        nn = None
        md = None
    
    return {
        "method": method,
        "dim": dim,
        "representation": representation,
        "umap_n_neighbors": nn,
        "umap_min_dist": md,
    }


# ============================================================================
# PER-RUN ANALYSIS
# ============================================================================

def analyze_single_run_wrapper(run_dir: Path) -> Optional[dict]:
    """
    Wrapper for parallel execution - creates its own logger.
    
    Note: This function is designed for parallel execution and should not
    use the main logger to avoid conflicts between processes.
    """
    # Create a minimal logger for this process (only errors to stderr)
    logger = logging.getLogger(f"worker_{run_dir.name}")
    logger.setLevel(logging.WARNING)  # Only warnings and errors
    logger.handlers.clear()
    sh = logging.StreamHandler()
    fmt = logging.Formatter("%(levelname)s - %(message)s")
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    
    try:
        return analyze_single_run(run_dir, logger)
    except Exception as e:
        logger.error(f"Error processing {run_dir.name}: {e}")
        return None


def analyze_single_run(run_dir: Path, logger: logging.Logger) -> Optional[dict]:
    """
    Analyze a single Phase 1 run for stratified enrichment.
    
    Returns dict with EF@1% values for all tiers, or None if analysis fails.
    """
    logger.info(f"Processing run: {run_dir.name}")
    
    # Load metadata
    summary_json = read_json(run_dir / "logs" / "phase1_summary.json")
    metrics_json = read_json(run_dir / "metrics" / "metrics.json")
    
    if not summary_json or not metrics_json:
        logger.warning(f"Missing summary or metrics JSON for {run_dir.name}")
        return None
    
    config = summary_json.get("config", {})
    target = str(metrics_json.get("target", "UNKNOWN"))
    
    # Load ranked scores (actives + ZINC)
    ranked_df = load_ranked_scores(run_dir, logger)
    if ranked_df is None:
        return None
    
    # Load actives with affinity
    actives_df = load_actives_with_affinity(config, target, logger)
    if actives_df is None or actives_df.empty:
        logger.warning(f"No actives data available for {run_dir.name}")
        del ranked_df
        gc.collect()
        return None
    
    # Join affinity and assign tiers
    ranked_df = join_affinity_to_ranked(ranked_df, actives_df, logger)
    
    # Free actives_df immediately after join
    del actives_df
    gc.collect()
    
    # Diagnostic counts
    N_total = len(ranked_df)
    N_zinc = (ranked_df["source"] != "actives").sum()
    N_actives = (ranked_df["source"] == "actives").sum()
    N_high = ((ranked_df["source"] == "actives") & (ranked_df["potency_tier"] == "High")).sum()
    N_medium = ((ranked_df["source"] == "actives") & (ranked_df["potency_tier"] == "Medium")).sum()
    N_weak = ((ranked_df["source"] == "actives") & (ranked_df["potency_tier"] == "Weak")).sum()
    N_no_tier = ((ranked_df["source"] == "actives") & (ranked_df["potency_tier"].isna())).sum()
    
    logger.info(f"  N_total={N_total} | N_zinc={N_zinc} | N_actives={N_actives}")
    logger.info(f"  Tiers: High={N_high}, Medium={N_medium}, Weak={N_weak}, No_tier={N_no_tier}")
    
    # Compute EF@1% for all tiers
    ef1_all = compute_ef_at_percent(ranked_df, tier=None, top_pct=0.01)
    ef1_high = compute_ef_at_percent(ranked_df, tier="High", top_pct=0.01)
    ef1_medium = compute_ef_at_percent(ranked_df, tier="Medium", top_pct=0.01)
    ef1_weak = compute_ef_at_percent(ranked_df, tier="Weak", top_pct=0.01)
    
    # Compute BEDROC and IEF for overall AND tier-specific
    import numpy as np
    
    # Overall (all actives)
    labels_all = np.asarray(ranked_df["label"].values, dtype=int)
    scores_all = np.asarray(ranked_df["score"].values, dtype=float)
    
    bedroc_20 = bedroc(labels_all, scores_all, alpha=20.0)
    bedroc_160 = bedroc(labels_all, scores_all, alpha=160.9)
    ief_20 = ief(labels_all, scores_all, alpha=20.0)
    ief_160 = ief(labels_all, scores_all, alpha=160.9)
    
    # Tier-specific BEDROC and IEF
    # For each tier: create binary labels (1 if active in tier, 0 otherwise)
    def compute_tier_metrics(tier_name: str):
        """Compute BEDROC/IEF for a specific potency tier."""
        # Binary labels: 1 if active AND in this tier, 0 otherwise
        tier_labels = ((ranked_df["source"] == "actives") & (ranked_df["potency_tier"] == tier_name)).astype(int).values
        
        # Only compute if there are actives in this tier
        if tier_labels.sum() == 0:
            return None, None, None, None
        
        labels_arr = np.asarray(tier_labels, dtype=int)
        scores_arr = np.asarray(ranked_df["score"].values, dtype=float)
        
        b20 = bedroc(labels_arr, scores_arr, alpha=20.0)
        b160 = bedroc(labels_arr, scores_arr, alpha=160.9)
        i20 = ief(labels_arr, scores_arr, alpha=20.0)
        i160 = ief(labels_arr, scores_arr, alpha=160.9)
        
        return b20, b160, i20, i160
    
    bedroc_20_high, bedroc_160_high, ief_20_high, ief_160_high = compute_tier_metrics("High")
    bedroc_20_medium, bedroc_160_medium, ief_20_medium, ief_160_medium = compute_tier_metrics("Medium")
    bedroc_20_weak, bedroc_160_weak, ief_20_weak, ief_160_weak = compute_tier_metrics("Weak")
    
    # Format EF values for logging
    ef1_all_str = f"{ef1_all:.2f}" if ef1_all is not None else "N/A"
    ef1_high_str = f"{ef1_high:.2f}" if ef1_high is not None else "N/A"
    ef1_medium_str = f"{ef1_medium:.2f}" if ef1_medium is not None else "N/A"
    ef1_weak_str = f"{ef1_weak:.2f}" if ef1_weak is not None else "N/A"
    
    logger.info(f"  EF@1%: All={ef1_all_str}, High={ef1_high_str}, Medium={ef1_medium_str}, Weak={ef1_weak_str}")
    logger.info(f"  BEDROC(α=20)={bedroc_20:.3f}, BEDROC(α=160)={bedroc_160:.3f}")
    logger.info(f"  IEF(α=20)={ief_20:.3f}, IEF(α=160)={ief_160:.3f}")
    
    # Extract config
    run_config = extract_run_config(summary_json, metrics_json)
    
    # Prepare results
    result = {
        "run_name": run_dir.name,
        "target": target,
        **run_config,
        "EF1_all": ef1_all,
        "EF1_high": ef1_high,
        "EF1_medium": ef1_medium,
        "EF1_weak": ef1_weak,
        "BEDROC_20": bedroc_20,
        "BEDROC_160": bedroc_160,
        "IEF_20": ief_20,
        "IEF_160": ief_160,
        "BEDROC_20_high": bedroc_20_high,
        "BEDROC_20_medium": bedroc_20_medium,
        "BEDROC_20_weak": bedroc_20_weak,
        "BEDROC_160_high": bedroc_160_high,
        "BEDROC_160_medium": bedroc_160_medium,
        "BEDROC_160_weak": bedroc_160_weak,
        "IEF_20_high": ief_20_high,
        "IEF_20_medium": ief_20_medium,
        "IEF_20_weak": ief_20_weak,
        "IEF_160_high": ief_160_high,
        "IEF_160_medium": ief_160_medium,
        "IEF_160_weak": ief_160_weak,
        "N_total": N_total,
        "N_zinc": N_zinc,
        "N_actives": N_actives,
        "N_high": N_high,
        "N_medium": N_medium,
        "N_weak": N_weak,
    }
    
    # Free large DataFrames before returning
    del ranked_df
    gc.collect()
    
    return result


# ============================================================================
# AGGREGATION & GROUPING
# ============================================================================

def aggregate_by_config(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate runs by configuration (representation, method, dim, UMAP params).
    
    Returns grouped DataFrame with mean ± std for each tier's EF@1%, BEDROC, and IEF.
    """
    # Group keys
    group_keys = ["representation", "method", "dim", "umap_n_neighbors", "umap_min_dist"]
    
    # Aggregate
    grouped = df.groupby(group_keys, dropna=False).agg(
        EF1_all_mean=("EF1_all", "mean"),
        EF1_all_std=("EF1_all", "std"),
        EF1_high_mean=("EF1_high", "mean"),
        EF1_high_std=("EF1_high", "std"),
        EF1_medium_mean=("EF1_medium", "mean"),
        EF1_medium_std=("EF1_medium", "std"),
        EF1_weak_mean=("EF1_weak", "mean"),
        EF1_weak_std=("EF1_weak", "std"),
        BEDROC_20_mean=("BEDROC_20", "mean"),
        BEDROC_20_std=("BEDROC_20", "std"),
        BEDROC_160_mean=("BEDROC_160", "mean"),
        BEDROC_160_std=("BEDROC_160", "std"),
        IEF_20_mean=("IEF_20", "mean"),
        IEF_20_std=("IEF_20", "std"),
        IEF_160_mean=("IEF_160", "mean"),
        IEF_160_std=("IEF_160", "std"),
        BEDROC_20_high_mean=("BEDROC_20_high", "mean"),
        BEDROC_20_high_std=("BEDROC_20_high", "std"),
        BEDROC_20_medium_mean=("BEDROC_20_medium", "mean"),
        BEDROC_20_medium_std=("BEDROC_20_medium", "std"),
        BEDROC_20_weak_mean=("BEDROC_20_weak", "mean"),
        BEDROC_20_weak_std=("BEDROC_20_weak", "std"),
        BEDROC_160_high_mean=("BEDROC_160_high", "mean"),
        BEDROC_160_high_std=("BEDROC_160_high", "std"),
        BEDROC_160_medium_mean=("BEDROC_160_medium", "mean"),
        BEDROC_160_medium_std=("BEDROC_160_medium", "std"),
        BEDROC_160_weak_mean=("BEDROC_160_weak", "mean"),
        BEDROC_160_weak_std=("BEDROC_160_weak", "std"),
        IEF_20_high_mean=("IEF_20_high", "mean"),
        IEF_20_high_std=("IEF_20_high", "std"),
        IEF_20_medium_mean=("IEF_20_medium", "mean"),
        IEF_20_medium_std=("IEF_20_medium", "std"),
        IEF_20_weak_mean=("IEF_20_weak", "mean"),
        IEF_20_weak_std=("IEF_20_weak", "std"),
        IEF_160_high_mean=("IEF_160_high", "mean"),
        IEF_160_high_std=("IEF_160_high", "std"),
        IEF_160_medium_mean=("IEF_160_medium", "mean"),
        IEF_160_medium_std=("IEF_160_medium", "std"),
        IEF_160_weak_mean=("IEF_160_weak", "mean"),
        IEF_160_weak_std=("IEF_160_weak", "std"),
        n_runs=("run_name", "count"),
    ).reset_index()
    
    return grouped


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_figure1_manuscript(df_grouped: pd.DataFrame, output_dir: Path, logger: logging.Logger) -> None:
    """
    Create Figure 1 for manuscript: Phase 1 Stratified Baseline Results.
    
    4-panel bar chart layout (2x2 grid):
    - Row 1: UMAP features (best), UMAP fingerprints (best)
    - Row 2: PCA features (best), PCA fingerprints (best)
    
    Each panel shows EF@1% for High/Medium/Weak potency tiers (3 bars per panel).
    All panels share the same y-axis range for direct comparability.
    
    Publication-quality formatting with ACS JCIM style guidelines.
    """
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Define the 4 configurations to plot
    configs = [
        {
            "method": "umap",
            "representation": "features",
            "label": "UMAP + Mordred Features",
            "panel": "A",
            "position": (0, 0),
        },
        {
            "method": "umap",
            "representation": "fingerprints",
            "label": "UMAP + ECFP4 Fingerprints",
            "panel": "B",
            "position": (0, 1),
        },
        {
            "method": "pca",
            "representation": "features",
            "label": "PCA + Mordred Features",
            "panel": "C",
            "position": (1, 0),
        },
        {
            "method": "pca",
            "representation": "fingerprints",
            "label": "PCA + ECFP4 Fingerprints",
            "panel": "D",
            "position": (1, 1),
        },
    ]
    
    # First pass: find best configs and determine global y-axis range
    plot_data = []
    global_max = 0.0
    
    for config in configs:
        method = config["method"]
        rep = config["representation"]
        
        # Filter to this method + representation
        subset = df_grouped[
            (df_grouped["method"] == method) &
            (df_grouped["representation"] == rep)
        ].copy()
        
        if subset.empty:
            logger.warning(f"No data for {config['label']}")
            continue
        
        # Pick best config by high-potency EF@1%
        best_row = subset.sort_values("EF1_high_mean", ascending=False).iloc[0]
        
        # Extract values
        dim = int(best_row["dim"])
        high_mean = float(best_row.get("EF1_high_mean", 0))
        high_std = float(best_row.get("EF1_high_std", 0))
        medium_mean = float(best_row.get("EF1_medium_mean", 0))
        medium_std = float(best_row.get("EF1_medium_std", 0))
        weak_mean = float(best_row.get("EF1_weak_mean", 0))
        weak_std = float(best_row.get("EF1_weak_std", 0))
        
        # UMAP hyperparameters
        if method == "umap":
            nn = best_row.get("umap_n_neighbors")
            md = best_row.get("umap_min_dist")
            hp_str = f"dim={dim}, n_neighbors={nn}, min_dist={md}"
        else:
            hp_str = f"dim={dim}"
        
        # Store plot data
        plot_data.append({
            "config": config,
            "dim": dim,
            "hp_str": hp_str,
            "high_mean": high_mean,
            "high_std": high_std,
            "medium_mean": medium_mean,
            "medium_std": medium_std,
            "weak_mean": weak_mean,
            "weak_std": weak_std,
        })
        
        # Update global max
        global_max = max(global_max, 
                        high_mean + high_std, 
                        medium_mean + medium_std, 
                        weak_mean + weak_std)
    
    if not plot_data:
        logger.error("No valid data for Figure 1 manuscript plot")
        return
    
    # Set y-axis range with 10% headroom
    y_max = global_max * 1.1
    logger.info(f"Figure 1: Using shared y-axis range [0, {y_max:.1f}] (max={global_max:.1f})")
    
    # Create 2x2 subplot figure (publication size: 7" x 6.5")
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), sharey=True)
    
    # Plot each panel
    for data in plot_data:
        config = data["config"]
        row, col = config["position"]
        ax = axes[row, col]
        
        # Prepare bar data
        tiers = ["High", "Medium", "Weak"]
        means = [data["high_mean"], data["medium_mean"], data["weak_mean"]]
        stds = [data["high_std"], data["medium_std"], data["weak_std"]]
        
        # Color scheme: High (blue), Medium (orange), Weak (red)
        colors = ["#1f77b4", "#ff7f0e", "#d62728"]
        
        x = np.arange(len(tiers))
        bars = ax.bar(x, means, yerr=stds, capsize=4, width=0.6, 
                     color=colors, alpha=0.85, edgecolor="black", linewidth=0.8)
        
        # Annotate bars with numeric values
        for bar, mean_val, std_val in zip(bars, means, stds):
            if np.isfinite(mean_val):
                text_y = bar.get_height() + std_val + (y_max * 0.015)
                ax.text(bar.get_x() + bar.get_width()/2, text_y,
                       f"{mean_val:.1f}", ha="center", va="bottom", 
                       fontsize=9, fontweight="bold")
        
        # Panel label (A, B, C, D)
        ax.text(0.02, 0.98, config["panel"], transform=ax.transAxes,
               fontsize=14, fontweight="bold", verticalalignment="top",
               bbox=dict(boxstyle="square", facecolor="white", edgecolor="black", linewidth=1.5))
        
        # Title with method and hyperparameters
        ax.set_title(f"{config['label']}\n({data['hp_str']})", 
                    fontsize=10, fontweight="bold", pad=8)
        
        # Axis labels
        ax.set_xticks(x)
        ax.set_xticklabels(tiers, fontsize=10)
        ax.set_xlabel("Potency Tier", fontsize=10, fontweight="bold")
        
        # Y-axis label only on left column
        if col == 0:
            ax.set_ylabel("EF@1%", fontsize=11, fontweight="bold")
        
        # Set shared y-axis range
        ax.set_ylim(0, y_max)
        ax.grid(True, alpha=0.3, axis="y", linewidth=0.5)
    
    # Overall figure title
    fig.suptitle("Phase 1: Potency-Stratified Enrichment Baseline", 
                fontsize=13, fontweight="bold", y=0.995)
    
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    
    # Save as Figure 1 for manuscript
    fig.savefig(plots_dir / "figure1_phase1_stratified_baseline.png", dpi=300, bbox_inches="tight")
    fig.savefig(plots_dir / "figure1_phase1_stratified_baseline.pdf", bbox_inches="tight")
    plt.close(fig)
    
    logger.info("Saved Figure 1: figure1_phase1_stratified_baseline.png/.pdf")
    logger.info("="*60)
    logger.info("Figure 1 Details:")
    for data in plot_data:
        logger.info(f"  {data['config']['label']}:")
        logger.info(f"    Hyperparameters: {data['hp_str']}")
        logger.info(f"    High   EF@1% = {data['high_mean']:.2f} ± {data['high_std']:.2f}")
        logger.info(f"    Medium EF@1% = {data['medium_mean']:.2f} ± {data['medium_std']:.2f}")
        logger.info(f"    Weak   EF@1% = {data['weak_mean']:.2f} ± {data['weak_std']:.2f}")
    logger.info("="*60)


def plot_tier_comparison(df_grouped: pd.DataFrame, output_dir: Path, logger: logging.Logger) -> None:
    """
    Create simple bar charts comparing EF@1% across tiers.
    
    One figure per (representation, method, dim) showing:
    - X-axis: [All, High, Medium, Weak]
    - Y-axis: EF@1% (mean)
    - Error bars: std
    
    All plots share the same y-axis range for comparability.
    """
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # First pass: determine global y-axis range across all plots
    global_max = 0.0
    for (rep, method, dim), group in df_grouped.groupby(["representation", "method", "dim"], dropna=False):
        if pd.isna(rep) or pd.isna(method) or pd.isna(dim):
            continue
        
        if method == "umap":
            group = group.sort_values("EF1_all_mean", ascending=False).head(1)
            if group.empty:
                continue
        
        row = group.iloc[0]
        means = [
            row.get("EF1_all_mean", np.nan),
            row.get("EF1_high_mean", np.nan),
            row.get("EF1_medium_mean", np.nan),
            row.get("EF1_weak_mean", np.nan),
        ]
        
        # Find max EF value (no error bars for this calculation)
        for mean_val in means:
            if np.isfinite(mean_val):
                global_max = max(global_max, mean_val)
    
    # Set y-axis range: max EF value + 5
    y_max = global_max + 5.0
    
    logger.info(f"Using shared y-axis range: [0, {y_max:.1f}] (max EF={global_max:.1f})")
    
    # Second pass: create plots with shared y-axis
    for (rep, method, dim), group in df_grouped.groupby(["representation", "method", "dim"], dropna=False):
        if pd.isna(rep) or pd.isna(method) or pd.isna(dim):
            continue
        
        # For UMAP, pick the best hyperparams by EF1_all_mean
        if method == "umap":
            group = group.sort_values("EF1_all_mean", ascending=False).head(1)
            if group.empty:
                continue
            nn = group.iloc[0]["umap_n_neighbors"]
            md = group.iloc[0]["umap_min_dist"]
            title_suffix = f"(nn={nn}, md={md})"
        else:
            title_suffix = ""
        
        # Extract mean ± std for each tier
        row = group.iloc[0]
        
        tiers = ["All", "High", "Medium", "Weak"]
        means = [
            row.get("EF1_all_mean", np.nan),
            row.get("EF1_high_mean", np.nan),
            row.get("EF1_medium_mean", np.nan),
            row.get("EF1_weak_mean", np.nan),
        ]
        stds = [
            row.get("EF1_all_std", 0),
            row.get("EF1_high_std", 0),
            row.get("EF1_medium_std", 0),
            row.get("EF1_weak_std", 0),
        ]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(7, 5))
        
        x = np.arange(len(tiers))
        colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]  # green, blue, orange, red
        
        bars = ax.bar(x, means, yerr=stds, capsize=5, width=0.6, color=colors, alpha=0.8, edgecolor="black", linewidth=1.2)
        
        # Annotate bars with values (use fixed offset relative to y_max to avoid clipping)
        for bar, mean_val, std_val in zip(bars, means, stds):
            if np.isfinite(mean_val):
                height = bar.get_height()
                # Position text just above error bar, with fallback to bar height
                text_y = height + std_val + (y_max * 0.02)  # 2% of y_max as offset
                ax.text(bar.get_x() + bar.get_width()/2, text_y,
                       f"{mean_val:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        
        ax.set_xticks(x)
        ax.set_xticklabels(tiers, fontsize=10)
        ax.set_ylabel("EF@1%", fontsize=11, fontweight="bold")
        ax.set_xlabel("Potency Tier", fontsize=11, fontweight="bold")
        ax.set_title(f"{rep.upper()} | {method.upper()} | dim={dim} {title_suffix}", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        
        # Set shared y-axis range
        ax.set_ylim(bottom=0, top=y_max)
        
        fig.tight_layout()
        
        # Save
        filename = f"tier_comparison_{rep}_{method}_dim{dim}"
        fig.savefig(plots_dir / f"{filename}.png", dpi=300, bbox_inches="tight")
        fig.savefig(plots_dir / f"{filename}.pdf", bbox_inches="tight")
        plt.close(fig)
        
        logger.info(f"Saved: {filename}.png")


def plot_bedroc_manuscript_figure(df_grouped: pd.DataFrame, output_dir: Path, logger: logging.Logger, alpha: int = 20) -> None:
    """
    Create BEDROC manuscript figure in same format as EF@1% Figure 1.
    
    4-panel bar chart layout (2x2 grid):
    - Row 1: UMAP features (best), UMAP fingerprints (best)
    - Row 2: PCA features (best), PCA fingerprints (best)
    
    Each panel shows BEDROC for High/Medium/Weak potency tiers (3 bars per panel).
    All panels share the same y-axis range for direct comparability.
    
    Args:
        alpha: BEDROC alpha parameter (20 or 160)
    """
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    metric_prefix = f"BEDROC_{alpha}"
    
    # Define the 4 configurations to plot
    configs = [
        {"method": "umap", "representation": "features", "label": "UMAP + Mordred Features", "panel": "A", "position": (0, 0)},
        {"method": "umap", "representation": "fingerprints", "label": "UMAP + ECFP4 Fingerprints", "panel": "B", "position": (0, 1)},
        {"method": "pca", "representation": "features", "label": "PCA + Mordred Features", "panel": "C", "position": (1, 0)},
        {"method": "pca", "representation": "fingerprints", "label": "PCA + ECFP4 Fingerprints", "panel": "D", "position": (1, 1)},
    ]
    
    # First pass: find best configs and determine global y-axis range
    plot_data = []
    global_max = 0.0
    
    for config in configs:
        method = config["method"]
        rep = config["representation"]
        
        subset = df_grouped[(df_grouped["method"] == method) & (df_grouped["representation"] == rep)].copy()
        
        if subset.empty:
            logger.warning(f"No data for {config['label']}")
            continue
        
        # Pick best config by high-potency BEDROC
        best_row = subset.sort_values(f"{metric_prefix}_high_mean", ascending=False).iloc[0]
        
        # Extract values
        dim = int(best_row["dim"])
        high_mean = float(best_row.get(f"{metric_prefix}_high_mean", 0))
        high_std = float(best_row.get(f"{metric_prefix}_high_std", 0))
        medium_mean = float(best_row.get(f"{metric_prefix}_medium_mean", 0))
        medium_std = float(best_row.get(f"{metric_prefix}_medium_std", 0))
        weak_mean = float(best_row.get(f"{metric_prefix}_weak_mean", 0))
        weak_std = float(best_row.get(f"{metric_prefix}_weak_std", 0))
        
        if method == "umap":
            nn = best_row.get("umap_n_neighbors")
            md = best_row.get("umap_min_dist")
            hp_str = f"dim={dim}, n_neighbors={nn}, min_dist={md}"
        else:
            hp_str = f"dim={dim}"
        
        plot_data.append({
            "config": config, "dim": dim, "hp_str": hp_str,
            "high_mean": high_mean, "high_std": high_std,
            "medium_mean": medium_mean, "medium_std": medium_std,
            "weak_mean": weak_mean, "weak_std": weak_std,
        })
        
        global_max = max(global_max, high_mean + high_std, medium_mean + medium_std, weak_mean + weak_std)
    
    if not plot_data:
        logger.error(f"No valid data for BEDROC (α={alpha}) manuscript figure")
        return
    
    y_max = global_max * 1.1
    logger.info(f"BEDROC (α={alpha}): Using shared y-axis range [0, {y_max:.3f}]")
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), sharey=True)
    
    for data in plot_data:
        config = data["config"]
        row, col = config["position"]
        ax = axes[row, col]
        
        tiers = ["High", "Medium", "Weak"]
        means = [data["high_mean"], data["medium_mean"], data["weak_mean"]]
        stds = [data["high_std"], data["medium_std"], data["weak_std"]]
        colors = ["#1f77b4", "#ff7f0e", "#d62728"]
        
        x = np.arange(len(tiers))
        bars = ax.bar(x, means, yerr=stds, capsize=4, width=0.6, 
                     color=colors, alpha=0.85, edgecolor="black", linewidth=0.8)
        
        for bar, mean_val, std_val in zip(bars, means, stds):
            if np.isfinite(mean_val):
                text_y = bar.get_height() + std_val + (y_max * 0.015)
                ax.text(bar.get_x() + bar.get_width()/2, text_y,
                       f"{mean_val:.2f}", ha="center", va="bottom", 
                       fontsize=9, fontweight="bold")
        
        ax.text(0.02, 0.98, config["panel"], transform=ax.transAxes,
               fontsize=14, fontweight="bold", verticalalignment="top",
               bbox=dict(boxstyle="square", facecolor="white", edgecolor="black", linewidth=1.5))
        
        ax.set_title(f"{config['label']}\n({data['hp_str']})", fontsize=10, fontweight="bold", pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(tiers, fontsize=10)
        ax.set_xlabel("Potency Tier", fontsize=10, fontweight="bold")
        
        if col == 0:
            ax.set_ylabel(f"BEDROC (α={alpha})", fontsize=11, fontweight="bold")
        
        ax.set_ylim(0, y_max)
        ax.grid(True, alpha=0.3, axis="y", linewidth=0.5)
    
    fig.suptitle(f"Phase 1: BEDROC (α={alpha}) Across Methods", fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    
    filename = f"phase1_bedroc_{alpha}_across_methods"
    fig.savefig(plots_dir / f"{filename}.png", dpi=300, bbox_inches="tight")
    fig.savefig(plots_dir / f"{filename}.pdf", bbox_inches="tight")
    plt.close(fig)
    
    logger.info(f"Saved: {filename}.png/.pdf")


def plot_ief_manuscript_figure(df_grouped: pd.DataFrame, output_dir: Path, logger: logging.Logger, alpha: int = 20) -> None:
    """
    Create IEF manuscript figure in same format as EF@1% Figure 1.
    
    4-panel bar chart layout (2x2 grid):
    - Row 1: UMAP features (best), UMAP fingerprints (best)
    - Row 2: PCA features (best), PCA fingerprints (best)
    
    Each panel shows IEF for High/Medium/Weak potency tiers (3 bars per panel).
    All panels share the same y-axis range for direct comparability.
    
    Args:
        alpha: IEF alpha parameter (20 or 160)
    """
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    metric_prefix = f"IEF_{alpha}"
    
    # Define the 4 configurations to plot
    configs = [
        {"method": "umap", "representation": "features", "label": "UMAP + Mordred Features", "panel": "A", "position": (0, 0)},
        {"method": "umap", "representation": "fingerprints", "label": "UMAP + ECFP4 Fingerprints", "panel": "B", "position": (0, 1)},
        {"method": "pca", "representation": "features", "label": "PCA + Mordred Features", "panel": "C", "position": (1, 0)},
        {"method": "pca", "representation": "fingerprints", "label": "PCA + ECFP4 Fingerprints", "panel": "D", "position": (1, 1)},
    ]
    
    # First pass: find best configs and determine global y-axis range
    plot_data = []
    global_max = 0.0
    
    for config in configs:
        method = config["method"]
        rep = config["representation"]
        
        subset = df_grouped[(df_grouped["method"] == method) & (df_grouped["representation"] == rep)].copy()
        
        if subset.empty:
            logger.warning(f"No data for {config['label']}")
            continue
        
        # Pick best config by high-potency IEF
        best_row = subset.sort_values(f"{metric_prefix}_high_mean", ascending=False).iloc[0]
        
        # Extract values
        dim = int(best_row["dim"])
        high_mean = float(best_row.get(f"{metric_prefix}_high_mean", 0))
        high_std = float(best_row.get(f"{metric_prefix}_high_std", 0))
        medium_mean = float(best_row.get(f"{metric_prefix}_medium_mean", 0))
        medium_std = float(best_row.get(f"{metric_prefix}_medium_std", 0))
        weak_mean = float(best_row.get(f"{metric_prefix}_weak_mean", 0))
        weak_std = float(best_row.get(f"{metric_prefix}_weak_std", 0))
        
        if method == "umap":
            nn = best_row.get("umap_n_neighbors")
            md = best_row.get("umap_min_dist")
            hp_str = f"dim={dim}, n_neighbors={nn}, min_dist={md}"
        else:
            hp_str = f"dim={dim}"
        
        plot_data.append({
            "config": config, "dim": dim, "hp_str": hp_str,
            "high_mean": high_mean, "high_std": high_std,
            "medium_mean": medium_mean, "medium_std": medium_std,
            "weak_mean": weak_mean, "weak_std": weak_std,
        })
        
        global_max = max(global_max, high_mean + high_std, medium_mean + medium_std, weak_mean + weak_std)
    
    if not plot_data:
        logger.error(f"No valid data for IEF (α={alpha}) manuscript figure")
        return
    
    y_max = global_max * 1.1
    logger.info(f"IEF (α={alpha}): Using shared y-axis range [0, {y_max:.1f}]")
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), sharey=True)
    
    for data in plot_data:
        config = data["config"]
        row, col = config["position"]
        ax = axes[row, col]
        
        tiers = ["High", "Medium", "Weak"]
        means = [data["high_mean"], data["medium_mean"], data["weak_mean"]]
        stds = [data["high_std"], data["medium_std"], data["weak_std"]]
        colors = ["#1f77b4", "#ff7f0e", "#d62728"]
        
        x = np.arange(len(tiers))
        bars = ax.bar(x, means, yerr=stds, capsize=4, width=0.6, 
                     color=colors, alpha=0.85, edgecolor="black", linewidth=0.8)
        
        for bar, mean_val, std_val in zip(bars, means, stds):
            if np.isfinite(mean_val):
                text_y = bar.get_height() + std_val + (y_max * 0.015)
                ax.text(bar.get_x() + bar.get_width()/2, text_y,
                       f"{mean_val:.1f}", ha="center", va="bottom", 
                       fontsize=9, fontweight="bold")
        
        ax.text(0.02, 0.98, config["panel"], transform=ax.transAxes,
               fontsize=14, fontweight="bold", verticalalignment="top",
               bbox=dict(boxstyle="square", facecolor="white", edgecolor="black", linewidth=1.5))
        
        ax.set_title(f"{config['label']}\n({data['hp_str']})", fontsize=10, fontweight="bold", pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(tiers, fontsize=10)
        ax.set_xlabel("Potency Tier", fontsize=10, fontweight="bold")
        
        if col == 0:
            ax.set_ylabel(f"IEF (α={alpha})", fontsize=11, fontweight="bold")
        
        ax.set_ylim(0, y_max)
        ax.grid(True, alpha=0.3, axis="y", linewidth=0.5)
    
    fig.suptitle(f"Phase 1: IEF (α={alpha}) Across Methods", fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    
    filename = f"phase1_ief_{alpha}_across_methods"
    fig.savefig(plots_dir / f"{filename}.png", dpi=300, bbox_inches="tight")
    fig.savefig(plots_dir / f"{filename}.pdf", bbox_inches="tight")
    plt.close(fig)
    
    logger.info(f"Saved: {filename}.png/.pdf")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 1 Potency-Stratified Analysis (v4 molfuse)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python scripts/phase1_stratified_scores.py \
      --workspace_dir experiment_workspace_v4 \
      --phase phase1 \
      --output_dir reporting/phase1_stratified
        """
    )
    parser.add_argument("--workspace_dir", type=str, required=True, help="Path to experiment workspace (e.g., experiment_workspace_v4)")
    parser.add_argument("--phase", type=str, default="phase1", help="Phase subdirectory (default: phase1)")
    parser.add_argument("--output_dir", type=str, default="reporting/phase1_stratified", help="Output directory")
    parser.add_argument("--n_workers", type=int, default=None, help="Number of parallel workers (default: CPU count)")
    parser.add_argument("--plot_only", action="store_true", help="Skip data collection and only regenerate plots from existing CSV files")
    
    args = parser.parse_args()
    
    # Setup
    workspace_dir = Path(args.workspace_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger(output_dir)
    logger.info("="*80)
    logger.info("PHASE 1 POTENCY-STRATIFIED ANALYSIS")
    logger.info("="*80)
    logger.info(f"Workspace: {workspace_dir}")
    logger.info(f"Phase: {args.phase}")
    logger.info(f"Output: {output_dir}")
    
    # Check if we're in plot-only mode
    if args.plot_only:
        logger.info("="*80)
        logger.info("PLOT-ONLY MODE: Skipping data collection")
        logger.info("="*80)
        
        # Load existing data
        summary_csv = output_dir / "stratified_summary.csv"
        if not summary_csv.exists():
            logger.error(f"Cannot find existing summary file: {summary_csv}")
            logger.error("Run without --plot_only first to generate data")
            return
        
        logger.info(f"Loading existing data from: {summary_csv}")
        df_summary = pd.read_csv(summary_csv)
        logger.info(f"Loaded {len(df_summary)} runs")
        
        # Aggregate by config
        df_grouped = aggregate_by_config(df_summary)
        grouped_csv = output_dir / "stratified_grouped.csv"
        df_grouped.to_csv(grouped_csv, index=False)
        logger.info(f"Re-saved grouped data: {grouped_csv} (rows={len(df_grouped)})")
        
        # Create plots
        logger.info("Regenerating plots with shared y-axis...")
        plot_tier_comparison(df_grouped, output_dir, logger)
        
        # Create Figure 1 for manuscript (EF@1%)
        logger.info("Creating Figure 1 for manuscript (EF@1%)...")
        plot_figure1_manuscript(df_grouped, output_dir, logger)
        
        # Create BEDROC manuscript figures (α=20 and α=160)
        logger.info("Creating BEDROC manuscript figures...")
        plot_bedroc_manuscript_figure(df_grouped, output_dir, logger, alpha=20)
        plot_bedroc_manuscript_figure(df_grouped, output_dir, logger, alpha=160)
        
        # Create IEF manuscript figures (α=20 and α=160)
        logger.info("Creating IEF manuscript figures...")
        plot_ief_manuscript_figure(df_grouped, output_dir, logger, alpha=20)
        plot_ief_manuscript_figure(df_grouped, output_dir, logger, alpha=160)
        
        logger.info("="*80)
        logger.info("PLOT REGENERATION COMPLETE")
        logger.info("="*80)
        return
    
    # Normal mode: collect data
    logger.info("="*80)
    logger.info("DATA COLLECTION MODE")
    logger.info("="*80)
    
    # Find all runs
    phase_dir = workspace_dir / args.phase
    if not phase_dir.exists():
        logger.error(f"Phase directory not found: {phase_dir}")
        return
    
    run_dirs = sorted([d for d in phase_dir.iterdir() if d.is_dir()])
    logger.info(f"Found {len(run_dirs)} run directories")
    
    # Determine number of workers (cap at 8 to avoid memory pressure)
    if args.n_workers:
        n_workers = min(args.n_workers, 8)
        if args.n_workers > 8:
            logger.warning(f"Requested {args.n_workers} workers, capping at 8 to avoid OOM")
    else:
        n_workers = min(8, os.cpu_count() or 4)  # Conservative default
    
    logger.info(f"Using {n_workers} parallel workers (memory-conservative setting)")
    logger.info(f"MF data will be cached in memory to avoid reloading")
    
    # Analyze each run in parallel
    results: List[dict] = []
    total_runs = len(run_dirs)
    completed = 0
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all jobs
        future_to_run = {executor.submit(analyze_single_run_wrapper, run_dir): run_dir for run_dir in run_dirs}
        
        # Process results as they complete
        for future in as_completed(future_to_run):
            run_dir = future_to_run[future]
            completed += 1
            
            try:
                result = future.result()
                if result:
                    results.append(result)
                    logger.info(f"[{completed}/{total_runs}] ✓ {run_dir.name}")
                else:
                    logger.warning(f"[{completed}/{total_runs}] ✗ {run_dir.name} - No result")
            except Exception as e:
                logger.error(f"[{completed}/{total_runs}] ✗ {run_dir.name} - {e}")
    
    if not results:
        logger.warning("No results produced (all runs failed or incomplete)")
        return
    
    logger.info(f"Successfully analyzed {len(results)} runs")
    
    # Save per-run summary
    df_summary = pd.DataFrame(results)
    summary_csv = output_dir / "stratified_summary.csv"
    df_summary.to_csv(summary_csv, index=False)
    logger.info(f"Saved: {summary_csv} (rows={len(df_summary)})")
    
    # Aggregate by config
    df_grouped = aggregate_by_config(df_summary)
    grouped_csv = output_dir / "stratified_grouped.csv"
    df_grouped.to_csv(grouped_csv, index=False)
    logger.info(f"Saved: {grouped_csv} (rows={len(df_grouped)})")
    
    # Create plots
    logger.info("Creating tier comparison plots...")
    plot_tier_comparison(df_grouped, output_dir, logger)
    
    # Create Figure 1 for manuscript (EF@1%)
    logger.info("Creating Figure 1 for manuscript (EF@1%)...")
    plot_figure1_manuscript(df_grouped, output_dir, logger)
    
    # Create BEDROC manuscript figures (α=20 and α=160)
    logger.info("Creating BEDROC manuscript figures...")
    plot_bedroc_manuscript_figure(df_grouped, output_dir, logger, alpha=20)
    plot_bedroc_manuscript_figure(df_grouped, output_dir, logger, alpha=160)
    
    # Create IEF manuscript figures (α=20 and α=160)
    logger.info("Creating IEF manuscript figures...")
    plot_ief_manuscript_figure(df_grouped, output_dir, logger, alpha=20)
    plot_ief_manuscript_figure(df_grouped, output_dir, logger, alpha=160)
    
    logger.info("="*80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()
