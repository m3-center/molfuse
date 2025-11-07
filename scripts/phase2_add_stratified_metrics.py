#!/usr/bin/env python3
"""
Phase 2 Stratified Metrics Re-Analysis (Post-Hoc)

Re-analyzes existing Phase 2 results to add potency-tier stratified EF@1% metrics.

This script:
1. Loads Phase 2 ranked_scores.csv files
2. Joins actives with affinity data from source CSVs
3. Assigns potency tiers (High/Medium/Weak)
4. Computes tier-specific EF@1% for each cutoff
5. Saves extended metrics.json with stratified metrics

Usage:
    python scripts/phase2_add_stratified_metrics.py \
        --workspace_dir experiment_workspace_v4 \
        --phase2_run_name cutoff_sweep

Output:
    Updates each cutoff_*/metrics.json with new fields:
    - ef1_high, ef1_medium, ef1_weak
    - n_actives_high, n_actives_medium, n_actives_weak
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ============================================================================
# POTENCY TIER DEFINITIONS
# ============================================================================

POTENCY_TIERS = {
    "High": (0.1, 100.0),       # 0.1-100 nM (drug-like)
    "Medium": (100.0, 1000.0),  # 100-1000 nM (moderate)
    "Weak": (1000.0, 100000.0), # 1K-100K nM (marginal)
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def setup_logger() -> logging.Logger:
    """Create logger with console output."""
    logger = logging.getLogger("phase2_stratified")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    sh = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    
    return logger


def assign_potency_tier(affinity_nM: float) -> Optional[str]:
    """
    Assign potency tier based on affinity value (nM).
    
    Returns:
        "High" if 0.1 ≤ affinity ≤ 100
        "Medium" if 100 < affinity ≤ 1000
        "Weak" if 1000 < affinity ≤ 100000
        None otherwise
    """
    try:
        val = float(affinity_nM)
    except (ValueError, TypeError):
        return None
    
    if val < 0.1 or val > 100000:
        return None
    
    for tier_name, (min_val, max_val) in POTENCY_TIERS.items():
        if min_val <= val <= max_val:
            return tier_name
    
    return None


def find_column_ignorecase(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Find first matching column name (case-insensitive)."""
    col_map = {c.lower().strip(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in col_map:
            return col_map[candidate.lower()]
    return None


def extract_accession(phase1_run: str) -> Optional[str]:
    """
    Extract UniProt accession from Phase 1 run name.
    
    Example: "pca_features_dim2_rep1" -> look for target in workspace
    Fallback: extract from model directory name or config
    """
    # This is a simple pattern; may need adjustment based on actual naming
    m = re.search(r"_([A-Z0-9]{6})_", phase1_run)
    return m.group(1) if m else None


# ============================================================================
# DATA LOADING
# ============================================================================

def load_actives_with_affinity(
    phase1_workspace: Path,
    phase1_run_name: str,
    logger: logging.Logger
) -> Optional[pd.DataFrame]:
    """
    Load actives with affinity from Phase 1 source data.
    
    Strategy:
    1. Load Phase 1 summary.json to get config
    2. Load actives CSV from config (if specified)
    3. Fallback: load MF CSV and filter by target accession
    
    Returns DataFrame with: Compound ChEMBL ID, SMILES, Standard Value (nM)
    """
    # Load Phase 1 summary
    phase1_dir = phase1_workspace / "phase1" / phase1_run_name
    summary_path = phase1_dir / "logs" / "phase1_summary.json"
    
    if not summary_path.exists():
        logger.warning(f"Phase 1 summary not found: {summary_path}")
        return None
    
    try:
        with summary_path.open("r") as f:
            summary = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load Phase 1 summary: {e}")
        return None
    
    config = summary.get("config", {})
    
    # Strategy 1: Load actives CSV directly
    actives_csv = config.get("actives_features_csv")
    if actives_csv:
        p = Path(actives_csv)
        if p.exists():
            try:
                cols_needed = ["Compound ChEMBL ID", "SMILES", "Standard Value (nM)"]
                df = pd.read_csv(p, usecols=lambda c: c in cols_needed + ["accession"], low_memory=False)
                df = df[cols_needed].copy()
                logger.info(f"Loaded actives from {p.name}: {len(df)} rows")
                return df
            except Exception as e:
                logger.warning(f"Failed to load actives CSV: {e}")
    
    # Strategy 2: Load from MF CSV and filter by target
    mf_csv = config.get("mf_features_csv")
    if not mf_csv:
        logger.warning("No actives or MF CSV in Phase 1 config")
        return None
    
    # Get target accession from config
    target = config.get("target", "")
    accession = None
    
    # Try to extract accession from target string
    if target:
        m = re.search(r"_([A-Z0-9]{6})$", target)
        if m:
            accession = m.group(1)
    
    if not accession:
        logger.warning(f"Cannot extract accession from target: {target}")
        return None
    
    p = Path(mf_csv)
    if not p.exists():
        logger.warning(f"MF CSV not found: {p}")
        return None
    
    try:
        cols_needed = ["Compound ChEMBL ID", "SMILES", "Standard Value (nM)", "accession"]
        df_all = pd.read_csv(p, usecols=lambda c: c in cols_needed, low_memory=False)
        
        if "accession" not in df_all.columns:
            logger.warning("MF CSV lacks accession column")
            return None
        
        df = df_all[df_all["accession"] == accession].copy()
        df = df[["Compound ChEMBL ID", "SMILES", "Standard Value (nM)"]].copy()
        
        logger.info(f"Loaded actives from MF CSV (filtered by {accession}): {len(df)} rows")
        return df
    
    except Exception as e:
        logger.error(f"Failed to load MF CSV: {e}")
        return None


def load_ranked_scores(cutoff_dir: Path, logger: logging.Logger) -> Optional[pd.DataFrame]:
    """Load ranked_scores.csv from Phase 2 cutoff directory."""
    ranked_path = cutoff_dir / "ranked_scores.csv"
    
    if not ranked_path.exists():
        logger.warning(f"ranked_scores.csv not found: {ranked_path}")
        return None
    
    try:
        df = pd.read_csv(ranked_path, low_memory=False)
        
        # Ensure proper sorting (should already be sorted, but verify)
        if "score" in df.columns:
            df = df.sort_values(by="score", ascending=False, kind="stable").reset_index(drop=True)
        elif "distance" in df.columns:
            df = df.sort_values(by="distance", ascending=True, kind="stable").reset_index(drop=True)
        
        return df
    except Exception as e:
        logger.error(f"Error loading ranked_scores.csv: {e}")
        return None


# ============================================================================
# TIER ASSIGNMENT
# ============================================================================

def join_affinity_to_ranked(
    ranked_df: pd.DataFrame,
    actives_df: pd.DataFrame,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Join affinity values to ranked_df and assign potency tiers.
    
    This is a left join on the FULL ranked list (actives + ZINC).
    Only actives will get affinity values; ZINC will have NaN.
    """
    ranked_df = ranked_df.copy()
    actives_df = actives_df.copy()
    
    # Find join columns (prefer ChEMBL ID, fallback to SMILES)
    chembl_col_r = find_column_ignorecase(ranked_df, ["Compound ChEMBL ID", "compound_chembl_id"])
    chembl_col_a = find_column_ignorecase(actives_df, ["Compound ChEMBL ID", "compound_chembl_id"])
    
    if chembl_col_r and chembl_col_a:
        logger.info("Joining by Compound ChEMBL ID")
        ranked_df["_join_key"] = ranked_df[chembl_col_r].astype(str).str.upper().str.strip()
        actives_df["_join_key"] = actives_df[chembl_col_a].astype(str).str.upper().str.strip()
    else:
        logger.info("Joining by SMILES")
        smiles_col_r = find_column_ignorecase(ranked_df, ["SMILES", "canonical_smiles"])
        smiles_col_a = find_column_ignorecase(actives_df, ["SMILES", "canonical_smiles"])
        
        if not smiles_col_r or not smiles_col_a:
            logger.error("Cannot find join key (no ChEMBL ID or SMILES)")
            ranked_df["potency_tier"] = None
            return ranked_df
        
        ranked_df["_join_key"] = ranked_df[smiles_col_r].astype(str).str.strip()
        actives_df["_join_key"] = actives_df[smiles_col_a].astype(str).str.strip()
    
    # Create affinity lookup
    if "Standard Value (nM)" not in actives_df.columns:
        logger.error("actives_df lacks Standard Value (nM) column")
        ranked_df["potency_tier"] = None
        return ranked_df
    
    affinity_lookup = actives_df[["_join_key", "Standard Value (nM)"]].dropna(subset=["_join_key"]).drop_duplicates(subset=["_join_key"])
    
    # Left join
    merged = ranked_df.merge(affinity_lookup, on="_join_key", how="left")
    
    # Assign tiers
    merged["potency_tier"] = merged["Standard Value (nM)"].apply(assign_potency_tier)
    
    # Cleanup
    merged.drop(columns=["_join_key", "Standard Value (nM)"], inplace=True, errors="ignore")
    
    return merged


# ============================================================================
# ENRICHMENT CALCULATION
# ============================================================================

def compute_ef_at_percent(
    ranked_df: pd.DataFrame,
    tier: Optional[str],
    top_pct: float
) -> Optional[float]:
    """
    Compute Enrichment Factor at top X%.
    
    Args:
        ranked_df: Full ranked list (actives + ZINC), pre-sorted
        tier: "High", "Medium", "Weak", or None (all actives)
        top_pct: Fraction (e.g., 0.01 for 1%)
    
    Returns:
        EF value or None if no actives in tier
    """
    if ranked_df.empty:
        return None
    
    N_total = len(ranked_df)
    k = max(1, int(np.ceil(top_pct * N_total)))
    top_k = ranked_df.head(k)
    
    if tier is None:
        # All actives
        mask_all = ranked_df["label"] == 1  # Use label column (1=active, 0=decoy)
        mask_top = top_k["label"] == 1
        N_actives = mask_all.sum()
        hits = mask_top.sum()
    else:
        # Specific tier
        mask_all = (ranked_df["label"] == 1) & (ranked_df["potency_tier"] == tier)
        mask_top = (top_k["label"] == 1) & (top_k["potency_tier"] == tier)
        N_actives = mask_all.sum()
        hits = mask_top.sum()
    
    if N_actives == 0:
        return None
    
    # EF = (hits / N_actives) / (k / N_total)
    ef = (hits / N_actives) / (k / N_total)
    return float(ef)


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def process_cutoff_directory(
    cutoff_dir: Path,
    phase1_workspace: Path,
    logger: logging.Logger
) -> bool:
    """
    Re-analyze one Phase 2 cutoff directory to add stratified metrics.
    
    Returns True if successful, False otherwise.
    """
    logger.info(f"Processing: {cutoff_dir.relative_to(cutoff_dir.parent.parent.parent)}")
    
    # Load existing metrics
    metrics_path = cutoff_dir / "metrics.json"
    if not metrics_path.exists():
        logger.warning(f"  metrics.json not found, skipping")
        return False
    
    try:
        with metrics_path.open("r") as f:
            metrics = json.load(f)
    except Exception as e:
        logger.error(f"  Failed to load metrics.json: {e}")
        return False
    
    # Get Phase 1 run name from metrics
    phase1_run = metrics.get("phase1_run")
    if not phase1_run:
        logger.warning(f"  No phase1_run in metrics.json, skipping")
        return False
    
    # Load actives with affinity
    actives_df = load_actives_with_affinity(phase1_workspace, phase1_run, logger)
    if actives_df is None or actives_df.empty:
        logger.warning(f"  No actives data available, skipping")
        return False
    
    # Load ranked scores
    ranked_df = load_ranked_scores(cutoff_dir, logger)
    if ranked_df is None:
        return False
    
    # Join affinity and assign tiers
    ranked_df = join_affinity_to_ranked(ranked_df, actives_df, logger)
    
    # Diagnostic counts
    N_total = len(ranked_df)
    N_actives = (ranked_df["label"] == 1).sum()
    N_high = ((ranked_df["label"] == 1) & (ranked_df["potency_tier"] == "High")).sum()
    N_medium = ((ranked_df["label"] == 1) & (ranked_df["potency_tier"] == "Medium")).sum()
    N_weak = ((ranked_df["label"] == 1) & (ranked_df["potency_tier"] == "Weak")).sum()
    N_no_tier = ((ranked_df["label"] == 1) & (ranked_df["potency_tier"].isna())).sum()
    
    logger.info(f"  N_total={N_total} | N_actives={N_actives}")
    logger.info(f"  Tiers: High={N_high}, Medium={N_medium}, Weak={N_weak}, No_tier={N_no_tier}")
    
    # Compute stratified EF@1%
    ef1_high = compute_ef_at_percent(ranked_df, "High", 0.01)
    ef1_medium = compute_ef_at_percent(ranked_df, "Medium", 0.01)
    ef1_weak = compute_ef_at_percent(ranked_df, "Weak", 0.01)
    
    # Log results
    ef1_high_str = f"{ef1_high:.2f}" if ef1_high is not None else "N/A"
    ef1_medium_str = f"{ef1_medium:.2f}" if ef1_medium is not None else "N/A"
    ef1_weak_str = f"{ef1_weak:.2f}" if ef1_weak is not None else "N/A"
    
    logger.info(f"  EF@1%: High={ef1_high_str}, Medium={ef1_medium_str}, Weak={ef1_weak_str}")
    
    # Update metrics
    metrics.update({
        "ef1_high": ef1_high,
        "ef1_medium": ef1_medium,
        "ef1_weak": ef1_weak,
        "n_actives_high": int(N_high),
        "n_actives_medium": int(N_medium),
        "n_actives_weak": int(N_weak),
        "n_actives_no_tier": int(N_no_tier),
    })
    
    # Save updated metrics
    try:
        with metrics_path.open("w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"  ✓ Updated metrics.json with stratified EF@1%")
        return True
    except Exception as e:
        logger.error(f"  Failed to save metrics.json: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2 Stratified Metrics Re-Analysis (Post-Hoc)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--workspace_dir", type=str, required=True,
                       help="Base workspace directory (e.g., experiment_workspace_v4)")
    parser.add_argument("--phase2_run_name", type=str, default="cutoff_sweep",
                       help="Phase 2 run name (default: cutoff_sweep)")
    parser.add_argument("--phase1_workspace", type=str, default=None,
                       help="Phase 1 workspace (default: same as workspace_dir)")
    
    args = parser.parse_args()
    
    logger = setup_logger()
    
    workspace_dir = Path(args.workspace_dir).resolve()
    phase2_run_name = args.phase2_run_name
    phase1_workspace = Path(args.phase1_workspace).resolve() if args.phase1_workspace else workspace_dir
    
    phase2_dir = workspace_dir / "phase2" / phase2_run_name
    
    if not phase2_dir.exists():
        logger.error(f"Phase 2 directory not found: {phase2_dir}")
        return
    
    logger.info("="*80)
    logger.info("PHASE 2 STRATIFIED METRICS RE-ANALYSIS")
    logger.info("="*80)
    logger.info(f"Workspace: {workspace_dir}")
    logger.info(f"Phase 2 run: {phase2_run_name}")
    logger.info(f"Phase 1 workspace: {phase1_workspace}")
    logger.info("")
    
    # Find all model × cutoff directories
    processed = 0
    success = 0
    
    for model_dir in sorted(phase2_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name in ("logs", "artifacts", "metrics"):
            continue
        
        for cutoff_dir in sorted(model_dir.iterdir()):
            if not cutoff_dir.is_dir() or not cutoff_dir.name.startswith("cutoff_"):
                continue
            
            processed += 1
            if process_cutoff_directory(cutoff_dir, phase1_workspace, logger):
                success += 1
            logger.info("")
    
    logger.info("="*80)
    logger.info("RE-ANALYSIS COMPLETE")
    logger.info("="*80)
    logger.info(f"Processed: {processed} cutoff directories")
    logger.info(f"Success: {success}")
    logger.info(f"Failed: {processed - success}")
    
    if success > 0:
        logger.info("")
        logger.info("✓ Stratified metrics added to metrics.json files")
        logger.info("✓ Ready for visualization in phase2_post_analysis.py")


if __name__ == "__main__":
    main()
