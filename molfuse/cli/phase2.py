"""
molfuse Phase 2: Affinity Cutoff Sensitivity Analysis

Research Question: Can we improve EF@1% by measuring distance only to more potent ligands?

Design: Re-scoring only (NO retraining)
- Load Phase 1 embeddings (MF, ZINC, actives)
- For each cutoff: filter MF by affinity, re-score via 1-NN, compute metrics
- No model retraining; only scoring changes with MF cloud filtering
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import joblib

from molfuse import __version__
from molfuse.analysis.select_best_phase1 import select_best_phase1_models
from molfuse.io.paths import make_run_dirs
from molfuse.metrics.metrics import ef_at_k_percent, pr_auc, roc_auc, spearman_rho
from molfuse.scoring.nn import nn_min_distance_scores


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="molfuse v4.0 Phase 2: Affinity Cutoff Sensitivity (Re-scoring Only)")
    p.add_argument("--config", type=str, required=True, help="Path to Phase 2 config JSON")
    p.add_argument("--workspace", type=str, required=True, help="Base workspace directory")
    return p.parse_args()


def to_pactivity_from_nM(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    vals = 9.0 - np.log10(s)
    return pd.Series(vals, index=series.index, name=getattr(series, "name", None))


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    with cfg_path.open("r") as f:
        cfg = json.load(f)

    # Create Phase 2 workspace
    run_name = cfg.get("run_name_prefix", "cutoff_sweep")
    ws = make_run_dirs(Path(args.workspace), phase="phase2", run_name=run_name)

    # Setup logging
    log_path = ws["logs"] / "run.log"
    logger = logging.getLogger(f"molfuse.phase2.{run_name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, mode="w")
    fmt = logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    logger.info("=" * 80)
    logger.info("PHASE 2 STARTED: Affinity Cutoff Sensitivity Analysis")
    logger.info("=" * 80)
    logger.info("Research Question: Can we improve EF@1% by scoring against high-potency-only MF?")
    logger.info("Design: Re-scoring only (NO model retraining)")

    # Select best Phase 1 models
    phase1_workspace = Path(cfg.get("phase1_workspace", args.workspace))
    phase1_phase_dir = cfg.get("phase1_phase_dir", "phase1")
    logger.info(f"Scanning Phase 1 workspace: {phase1_workspace / phase1_phase_dir}")

    try:
        selected_models = select_best_phase1_models(
            phase1_workspace=phase1_workspace,
            phase1_phase_dir=phase1_phase_dir,
            min_required=1,
        )
    except Exception as e:
        logger.error(f"Failed to select Phase 1 models: {e}")
        raise

    # Save selected models manifest
    selected_manifest = {
        k: v["run_name"] if v else None
        for k, v in selected_models.items()
    }
    (ws["base"] / "selected_models.json").write_text(json.dumps(selected_manifest, indent=2))
    logger.info(f"Selected models: {selected_manifest}")

    # Cutoff list
    cutoff_list = cfg.get("affinity_cutoff_nM_list", [100, 1000, 10000, 100000])
    logger.info(f"Cutoffs to test: {cutoff_list} nM")

    # Process each selected model
    for model_key, model_info in selected_models.items():
        if model_info is None:
            logger.warning(f"Skipping {model_key}: no valid Phase 1 run found")
            continue

        logger.info("=" * 80)
        logger.info(f"Processing {model_key}: {model_info['run_name']}")
        logger.info("=" * 80)

        phase1_run_dir = Path(model_info["run_dir"])

        # Load embeddings (NO re-projection)
        logger.info("Loading Phase 1 embeddings (NO re-projection)")
        emb_mf = pd.read_csv(phase1_run_dir / "artifacts" / "embedding_mf.csv")
        emb_zinc = pd.read_csv(phase1_run_dir / "artifacts" / "embedding_zinc.csv")
        emb_act = pd.read_csv(phase1_run_dir / "artifacts" / "embedding_actives.csv")

        # Extract coordinate columns
        z_cols = [c for c in emb_mf.columns if c.startswith("z")]
        Z_mf_full = emb_mf[z_cols].to_numpy(dtype=float)
        Z_zinc = emb_zinc[z_cols].to_numpy(dtype=float)
        Z_act = emb_act[z_cols].to_numpy(dtype=float)

        logger.info(f"Loaded embeddings: MF={len(Z_mf_full)}, ZINC={len(Z_zinc)}, Actives={len(Z_act)}")

        # Load MF source CSV to get affinity column
        mf_csv_path = Path(model_info["mf_csv"])
        if not mf_csv_path.exists():
            logger.error(f"MF source CSV not found: {mf_csv_path}")
            continue

        logger.info(f"Loading MF source CSV: {mf_csv_path}")
        df_mf_source = pd.read_csv(mf_csv_path, low_memory=False)

        # Check for affinity column
        if "Standard Value (nM)" not in df_mf_source.columns:
            logger.warning(f"No 'Standard Value (nM)' column in MF source; cannot apply cutoffs. Skipping {model_key}.")
            continue

        # Match MF source rows to embedding rows by SMILES (preferred) or Compound ID
        # Phase 1 has filtered/deduplicated the MF data, so we need to match rows
        smiles_col_emb = None
        for col in ["canonical_smiles", "SMILES"]:
            if col in emb_mf.columns:
                smiles_col_emb = col
                break
        
        id_col_emb = "Compound ChEMBL ID" if "Compound ChEMBL ID" in emb_mf.columns else None

        if not smiles_col_emb and not id_col_emb:
            logger.error(f"No SMILES or Compound ID column in embedding; cannot match to source. Skipping {model_key}.")
            continue

        # Try matching by SMILES first
        if smiles_col_emb:
            smiles_col_src = None
            for col in ["canonical_smiles", "SMILES"]:
                if col in df_mf_source.columns:
                    smiles_col_src = col
                    break
            
            if smiles_col_src:
                logger.info(f"Matching MF rows by SMILES ({smiles_col_emb} in embedding, {smiles_col_src} in source)")
                # Create a mapping: SMILES -> affinity
                affinity_map = df_mf_source.set_index(smiles_col_src)["Standard Value (nM)"]
                affinity_nM = emb_mf[smiles_col_emb].map(affinity_map)
                
                n_matched = affinity_nM.notna().sum()
                logger.info(f"Matched {n_matched}/{len(emb_mf)} MF rows by SMILES")
                
                if n_matched < len(emb_mf) * 0.95:  # Less than 95% matched
                    logger.warning(f"Only {n_matched}/{len(emb_mf)} rows matched; some affinity data may be missing")
            else:
                logger.error(f"SMILES column not found in MF source; cannot match. Skipping {model_key}.")
                continue
        elif id_col_emb and "Compound ChEMBL ID" in df_mf_source.columns:
            logger.info(f"Matching MF rows by Compound ChEMBL ID")
            affinity_map = df_mf_source.set_index("Compound ChEMBL ID")["Standard Value (nM)"]
            affinity_nM = emb_mf[id_col_emb].map(affinity_map)
            
            n_matched = affinity_nM.notna().sum()
            logger.info(f"Matched {n_matched}/{len(emb_mf)} MF rows by Compound ID")
            
            if n_matched < len(emb_mf) * 0.95:
                logger.warning(f"Only {n_matched}/{len(emb_mf)} rows matched; some affinity data may be missing")
        else:
            logger.error(f"Cannot match MF rows (no common key between embedding and source). Skipping {model_key}.")
            continue

        affinity_nM = pd.to_numeric(affinity_nM, errors="coerce")

        # Build evaluation set labels (same for all cutoffs)
        labels = np.concatenate([np.ones(len(Z_act), dtype=int), np.zeros(len(Z_zinc), dtype=int)])

        # Get actives' pActivity for Spearman (if available)
        if "Standard Value (nM)" in df_mf_source.columns:
            # Try to read actives source to get pActivity
            # For now, assume we can derive from ranked_scores.csv if it has labels
            # Alternatively, reload actives CSV if path is in config
            # Simplified: skip Spearman if actives lack affinity column
            pact_actives = None  # Placeholder; could be loaded from actives CSV if available
        else:
            pact_actives = None

        # Create model-specific output directory
        model_out_dir = ws["base"] / model_info["run_name"]
        model_out_dir.mkdir(exist_ok=True)

        # Process each cutoff
        for cutoff_nM in cutoff_list:
            logger.info("-" * 80)
            logger.info(f"Cutoff: {cutoff_nM} nM")

            # Filter MF by affinity
            mask_cutoff = affinity_nM <= cutoff_nM
            mask_cutoff = mask_cutoff.fillna(False).to_numpy(dtype=bool)
            Z_mf_filtered = Z_mf_full[mask_cutoff]

            n_mf_passing = int(mask_cutoff.sum())
            logger.info(f"MF after cutoff: {n_mf_passing}/{len(Z_mf_full)}")

            if n_mf_passing == 0:
                logger.warning(f"No MF compounds pass cutoff {cutoff_nM} nM; using full MF as fallback")
                Z_mf_filtered = Z_mf_full

            # Re-score evaluation set (actives + ZINC) via 1-NN to filtered MF
            Z_eval = np.vstack([Z_act, Z_zinc])
            scores, distances = nn_min_distance_scores(Z_mf_filtered, Z_eval)
            scores[np.isclose(scores, 0.0)] = 0.0
            distances[np.isclose(distances, 0.0)] = 0.0

            # Compute metrics
            ef1 = ef_at_k_percent(scores, labels, 1.0)
            ef5 = ef_at_k_percent(scores, labels, 5.0)
            ef10 = ef_at_k_percent(scores, labels, 10.0)
            roc = roc_auc(labels, scores)
            pr = pr_auc(labels, scores)

            # Spearman rho (placeholder; would need actives' affinity data)
            # For now, set to NaN
            rho, rho_p = np.nan, np.nan

            metrics = {
                "roc_auc": roc,
                "pr_auc": pr,
                "ef_1%": ef1,
                "ef_5%": ef5,
                "ef_10%": ef10,
                "spearman_rho": rho,
                "spearman_p": rho_p,
                "n_actives": int(len(Z_act)),
                "n_zinc_eval": int(len(Z_zinc)),
                "n_mf_for_scoring": int(n_mf_passing),
                "affinity_cutoff_nM": cutoff_nM,
                "method": model_info["method"],
                "representation": model_info["representation"],
                "dim": model_info["dim"],
                "phase1_run": model_info["run_name"],
            }

            logger.info(f"Metrics: EF@1%={ef1:.2f}, EF@5%={ef5:.2f}, EF@10%={ef10:.2f}, ROC-AUC={roc:.4f}, PR-AUC={pr:.4f}")

            # Save per-cutoff outputs
            cutoff_dir = model_out_dir / f"cutoff_{int(cutoff_nM)}nM"
            cutoff_dir.mkdir(exist_ok=True)

            (cutoff_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

            # Ranked scores
            ranked = pd.DataFrame({
                "score": scores,
                "distance": distances,
                "label": labels,
            })
            ranked.sort_values("score", ascending=False, inplace=True)
            ranked.to_csv(cutoff_dir / "ranked_scores.csv", index=False)

            logger.info(f"Saved outputs to {cutoff_dir}")

    # Summary
    summary = {
        "molfuse_version": __version__,
        "phase": "cutoff_sensitivity",
        "config": cfg,
        "selected_models": selected_manifest,
        "cutoffs_tested": cutoff_list,
        "invariants": {
            "no_retraining": True,
            "reuse_phase1_embeddings": True,
            "mf_filtering_by_cutoff": True,
            "actives_never_filtered": True,
        },
    }
    (ws["logs"] / "phase2_summary.json").write_text(json.dumps(summary, indent=2))

    logger.info("=" * 80)
    logger.info("PHASE 2 COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
