"""
Model selection utility for Phase 2.

Scans Phase 1 workspace and identifies best runs per method/representation based on EF@1%.
Robust to incomplete Phase 1 runs; skips missing or invalid runs.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional
import logging


logger = logging.getLogger(__name__)


def select_best_phase1_models(
    phase1_workspace: Path,
    phase1_phase_dir: str = "phase1",
    min_required: int = 1,
) -> Dict[str, List[Dict]]:
    """
    Select best Phase 1 runs for Phase 2 re-scoring.

    Scans phase1_workspace/phase1_phase_dir/ for completed runs.
    Identifies best hyperparameter configuration per (method, representation) combo based on EF@1%.
    Returns ALL replicates of the best configuration for statistical validity.

    Args:
        phase1_workspace: Base workspace directory
        phase1_phase_dir: Phase subdirectory name (default: "phase1")
        min_required: Minimum number of valid models required (default: 1)

    Returns:
        Dict with keys: "pca_features", "pca_fingerprints", "umap_features", "umap_fingerprints"
        Each value is a LIST of dicts (one per replicate) with keys: run_name, run_dir, ef1, method, representation, dim
        Missing combos have value [] (empty list).

    Raises:
        RuntimeError: If fewer than min_required valid model families found
    """
    phase1_dir = phase1_workspace / phase1_phase_dir
    if not phase1_dir.exists():
        raise FileNotFoundError(f"Phase 1 directory not found: {phase1_dir}")

    # Scan for completed runs
    candidates: List[Dict] = []
    for run_dir in phase1_dir.iterdir():
        if not run_dir.is_dir():
            continue

        summary_path = run_dir / "logs" / "phase1_summary.json"
        metrics_path = run_dir / "metrics" / "metrics.json"

        # Check for required files
        required_files = [
            summary_path,
            metrics_path,
            run_dir / "artifacts" / "scaler.joblib",
            run_dir / "artifacts" / "embedding_mf.csv",
            run_dir / "artifacts" / "embedding_zinc.csv",
            run_dir / "artifacts" / "embedding_actives.csv",
        ]
        if not all(f.exists() for f in required_files):
            logger.debug(f"Skipping incomplete run: {run_dir.name} (missing required files)")
            continue

        # Load summary and metrics
        try:
            with summary_path.open("r") as f:
                summary = json.load(f)
            with metrics_path.open("r") as f:
                metrics = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read JSON for {run_dir.name}: {e}")
            continue

        # Extract config and metrics
        cfg = summary.get("config", {})
        method = str(cfg.get("method", "unknown")).lower()
        representation = str(cfg.get("representation", "features")).lower()
        dim = int(cfg.get("dim", 0))
        ef1 = float(metrics.get("ef_1%", 0.0))

        # Check for model file
        if method == "pca":
            model_file = run_dir / "artifacts" / "pca_model.joblib"
        elif method == "umap":
            model_file = run_dir / "artifacts" / "umap_model.joblib"
        else:
            logger.debug(f"Unknown method {method} in {run_dir.name}")
            continue

        if not model_file.exists():
            logger.debug(f"Skipping {run_dir.name}: model file not found")
            continue

        candidates.append({
            "run_name": run_dir.name,
            "run_dir": str(run_dir),
            "method": method,
            "representation": representation,
            "dim": dim,
            "ef1": ef1,
            "mf_csv": str(cfg.get("mf_features_csv", "")),
        })

    logger.info(f"Found {len(candidates)} completed Phase 1 runs")

    # Group by (method, representation) and select best hyperparameter configuration by mean EF@1%
    combos = {
        "pca_features": ("pca", "features"),
        "pca_fingerprints": ("pca", "fingerprints"),
        "umap_features": ("umap", "features"),
        "umap_fingerprints": ("umap", "fingerprints"),
    }

    selected: Dict[str, List[Dict]] = {}
    for key, (method, representation) in combos.items():
        matching = [c for c in candidates if c["method"] == method and c["representation"] == representation]
        if not matching:
            selected[key] = []
            logger.warning(f"No completed runs found for {key}")
            continue

        # Group by hyperparameter configuration (dim + UMAP params if applicable)
        # For PCA: just dim
        # For UMAP: dim + n_neighbors + min_dist
        from collections import defaultdict
        config_groups = defaultdict(list)
        
        for c in matching:
            if c["method"] == "pca":
                config_key = (c["dim"],)
            else:  # umap
                # Extract UMAP params from run_name (format: ABL1_UMAP_features_2d_nn10_md0p01_rep4)
                run_name = c["run_name"]
                import re
                nn_match = re.search(r'nn(\d+)', run_name)
                md_match = re.search(r'md([\d.p]+)', run_name)
                n_neighbors = int(nn_match.group(1)) if nn_match else 50
                min_dist_str = md_match.group(1).replace('p', '.') if md_match else "0.01"
                min_dist = float(min_dist_str)
                config_key = (c["dim"], n_neighbors, min_dist)
            
            config_groups[config_key].append(c)
        
        # Find best configuration by mean EF@1% across replicates
        best_config_key = None
        best_mean_ef1 = -1
        for config_key, replicate_runs in config_groups.items():
            mean_ef1 = sum(r["ef1"] for r in replicate_runs) / len(replicate_runs)
            if mean_ef1 > best_mean_ef1:
                best_mean_ef1 = mean_ef1
                best_config_key = config_key
        
        # Return ALL replicates of the best configuration
        best_replicates = config_groups[best_config_key]
        selected[key] = best_replicates
        
        logger.info(
            f"Selected {key}: config={best_config_key}, mean_EF@1%={best_mean_ef1:.2f}, "
            f"n_replicates={len(best_replicates)}"
        )
        for rep in best_replicates:
            logger.info(f"  - {rep['run_name']} (EF@1%={rep['ef1']:.2f})")

    # Check minimum requirement
    valid_count = sum(1 for v in selected.values() if len(v) > 0)
    if valid_count < min_required:
        raise RuntimeError(
            f"Only {valid_count} valid Phase 1 model families found; need at least {min_required}. "
            f"Complete more Phase 1 runs or reduce min_required."
        )

    total_runs = sum(len(v) for v in selected.values())
    logger.info(f"Selected {valid_count}/4 model families ({total_runs} total runs) for Phase 2")
    return selected
