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
import json
import logging
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
    2. Fall back to filtering mf_features_csv by accession
    
    Returns DataFrame with columns: Compound ChEMBL ID, SMILES, Standard Value (nM)
    """
    # Strategy 1: Explicit actives file
    actives_path = config.get("actives_features_csv")
    if actives_path:
        p = Path(actives_path)
        if p.exists():
            try:
                cols_needed = ["Compound ChEMBL ID", "canonical_smiles", "SMILES", "Standard Value (nM)", "accession"]
                df = pd.read_csv(p, usecols=lambda c: c in cols_needed, low_memory=False)
                logger.info(f"Loaded actives from: {p} (rows={len(df)})")
                return df
            except Exception as e:
                logger.warning(f"Failed to load actives CSV {p}: {e}")
    
    # Strategy 2: Derive from MF file by accession
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
    
    try:
        cols_needed = ["Compound ChEMBL ID", "canonical_smiles", "SMILES", "Standard Value (nM)", "accession"]
        df_all = pd.read_csv(p, usecols=lambda c: c in cols_needed, low_memory=False)
        
        if "accession" not in df_all.columns:
            logger.warning("MF file lacks accession column; cannot filter")
            return None
        
        df = df_all[df_all["accession"] == accession].copy()
        logger.info(f"Derived actives from MF by accession={accession}: rows={len(df)}")
        return df
    except Exception as e:
        logger.warning(f"Failed to derive actives from MF {p}: {e}")
        return None


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
    2. Fallback to canonical_smiles or SMILES
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
        smiles_col_r = find_column_ignorecase(ranked_df, ["canonical_smiles", "SMILES"])
        smiles_col_a = find_column_ignorecase(actives_df, ["canonical_smiles", "SMILES"])
        
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
        return None
    
    # Join affinity and assign tiers
    ranked_df = join_affinity_to_ranked(ranked_df, actives_df, logger)
    
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
    
    # Format EF values for logging
    ef1_all_str = f"{ef1_all:.2f}" if ef1_all is not None else "N/A"
    ef1_high_str = f"{ef1_high:.2f}" if ef1_high is not None else "N/A"
    ef1_medium_str = f"{ef1_medium:.2f}" if ef1_medium is not None else "N/A"
    ef1_weak_str = f"{ef1_weak:.2f}" if ef1_weak is not None else "N/A"
    
    logger.info(f"  EF@1%: All={ef1_all_str}, High={ef1_high_str}, Medium={ef1_medium_str}, Weak={ef1_weak_str}")
    
    # Extract config
    run_config = extract_run_config(summary_json, metrics_json)
    
    # Return results
    return {
        "run_name": run_dir.name,
        "target": target,
        **run_config,
        "EF1_all": ef1_all,
        "EF1_high": ef1_high,
        "EF1_medium": ef1_medium,
        "EF1_weak": ef1_weak,
        "N_total": N_total,
        "N_zinc": N_zinc,
        "N_actives": N_actives,
        "N_high": N_high,
        "N_medium": N_medium,
        "N_weak": N_weak,
    }


# ============================================================================
# AGGREGATION & GROUPING
# ============================================================================

def aggregate_by_config(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate runs by configuration (representation, method, dim, UMAP params).
    
    Returns grouped DataFrame with mean ± std for each tier's EF@1%.
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
        n_runs=("run_name", "count"),
    ).reset_index()
    
    return grouped


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_tier_comparison(df_grouped: pd.DataFrame, output_dir: Path, logger: logging.Logger) -> None:
    """
    Create simple bar charts comparing EF@1% across tiers.
    
    One figure per (representation, method, dim) showing:
    - X-axis: [All, High, Medium, Weak]
    - Y-axis: EF@1% (mean)
    - Error bars: std
    """
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # For each unique config, create a plot
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
        
        # Annotate bars with values
        for bar, mean_val in zip(bars, means):
            if np.isfinite(mean_val):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + max(stds)*0.1,
                       f"{mean_val:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        
        ax.set_xticks(x)
        ax.set_xticklabels(tiers, fontsize=10)
        ax.set_ylabel("EF@1%", fontsize=11, fontweight="bold")
        ax.set_xlabel("Potency Tier", fontsize=11, fontweight="bold")
        ax.set_title(f"{rep.upper()} | {method.upper()} | dim={dim} {title_suffix}", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        
        # Set y-axis to start at 0
        ax.set_ylim(bottom=0)
        
        fig.tight_layout()
        
        # Save
        filename = f"tier_comparison_{rep}_{method}_dim{dim}"
        fig.savefig(plots_dir / f"{filename}.png", dpi=300, bbox_inches="tight")
        fig.savefig(plots_dir / f"{filename}.pdf", bbox_inches="tight")
        plt.close(fig)
        
        logger.info(f"Saved: {filename}.png")


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
    
    # Find all runs
    phase_dir = workspace_dir / args.phase
    if not phase_dir.exists():
        logger.error(f"Phase directory not found: {phase_dir}")
        return
    
    run_dirs = sorted([d for d in phase_dir.iterdir() if d.is_dir()])
    logger.info(f"Found {len(run_dirs)} run directories")
    
    # Determine number of workers
    n_workers = args.n_workers if args.n_workers else None  # None = use all CPUs
    logger.info(f"Using {n_workers if n_workers else 'all available'} CPU cores for parallel processing")
    
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
    
    logger.info("="*80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()
