from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from molfuse import __version__
from molfuse.data.prep import fit_scaler_on_mf_zinc, select_feature_columns
from molfuse.dr.pca import fit_pca
from molfuse.dr.umap_ import fit_umap
from molfuse.io.paths import make_run_dirs
from molfuse.metrics.metrics import ef_at_k_percent, pr_auc, roc_auc, spearman_rho
from molfuse.scoring.nn import nn_min_distance_scores


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="molfuse v4.0 Phase 1 runner")
    p.add_argument("--config", type=str, required=True, help="Path to config JSON")
    p.add_argument("--workspace", type=str, required=True, help="Base workspace directory")
    return p.parse_args()


def extract_accession(target: str) -> Optional[str]:
    m = re.match(r".*_([A-Z0-9]+)$", target)
    return m.group(1) if m else None


def to_pactivity_from_nM(series: pd.Series) -> pd.Series:
    # pActivity = -log10(M) = 9 - log10(nM)
    s = pd.to_numeric(series, errors="coerce")
    return 9.0 - np.log10(s)


def remove_overlap_by_smiles(df: pd.DataFrame, smiles_col: str, to_exclude: pd.Series) -> pd.DataFrame:
    if smiles_col in df.columns:
        return df[~df[smiles_col].isin(set(to_exclude.dropna().astype(str)))]
    return df


def dedup_mf_by_compound(df: pd.DataFrame) -> pd.DataFrame:
    if "Compound ChEMBL ID" not in df.columns:
        return df.drop_duplicates()
    agg: Dict[str, str] = {}
    for c in df.columns:
        if c in ("Compound ChEMBL ID",):
            continue
        if c == "Standard Value (nM)":
            agg[c] = "median"  # robust to outliers
        else:
            agg[c] = "first"
    out = df.groupby("Compound ChEMBL ID", as_index=False).agg(agg)
    return out


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    with cfg_path.open("r") as f:
        cfg = json.load(f)

    run_name = cfg.get("run_name", f"{cfg.get('target','target')}_{cfg.get('method','pca')}_{cfg.get('dim',2)}d")
    ws = make_run_dirs(Path(args.workspace), phase="phase1", run_name=run_name)

    # Required inputs
    mf_csv = Path(cfg["mf_features_csv"]) if "mf_features_csv" in cfg else None
    zinc_csv = Path(cfg["zinc_features_csv"]) if "zinc_features_csv" in cfg else None
    actives_csv = Path(cfg["actives_features_csv"]) if "actives_features_csv" in cfg else None
    if not (mf_csv and zinc_csv and actives_csv and mf_csv.exists() and zinc_csv.exists() and actives_csv.exists()):
        err = {
            "error": "Missing required CSV inputs",
            "required_keys": ["mf_features_csv", "zinc_features_csv", "actives_features_csv"],
            "cfg": cfg,
        }
        (ws["logs"] / "phase1_error.json").write_text(json.dumps(err, indent=2))
        raise FileNotFoundError("Provide mf_features_csv, zinc_features_csv, and actives_features_csv in config.")

    target = cfg.get("target", "UNKNOWN")
    accession = extract_accession(target)
    method = cfg.get("method", "pca").lower()
    dim = int(cfg.get("dim", 2))
    umap_params = cfg.get("umap_params", {})
    affinity_cutoff_nM = float(cfg.get("affinity_cutoff_nM", 100000))
    sample_zinc = int(cfg.get("sample_zinc", 0))

    # Load
    df_mf = pd.read_csv(mf_csv)
    df_zinc = pd.read_csv(zinc_csv)
    df_act = pd.read_csv(actives_csv)

    # Target-preserving exclusion in MF (if column available)
    if accession is not None:
        col_acc = "accession" if "accession" in df_mf.columns else ("Target Accession" if "Target Accession" in df_mf.columns else None)
        if col_acc is not None:
            df_mf = df_mf[df_mf[col_acc] != accession]

    # Overlap removal by SMILES (actives vs MF+ZINC)
    smiles_col = "canonical_smiles" if "canonical_smiles" in df_act.columns else ("SMILES" if "SMILES" in df_act.columns else None)
    if smiles_col is not None:
        act_smiles = df_act[smiles_col]
        df_mf = remove_overlap_by_smiles(df_mf, smiles_col, act_smiles)
        df_zinc = remove_overlap_by_smiles(df_zinc, smiles_col, act_smiles)

    # Deduplicate MF by compound, robust median for affinity
    df_mf = dedup_mf_by_compound(df_mf)

    # Feature columns (numeric-only, exclude known non-features)
    feat_cols_mf = select_feature_columns(df_mf)
    feat_cols_zinc = select_feature_columns(df_zinc)
    common_feats = [c for c in feat_cols_mf if c in feat_cols_zinc]
    if not common_feats:
        raise RuntimeError("No common numeric feature columns found between MF and ZINC.")

    # Coerce numerics and drop NaNs
    for df in (df_mf, df_zinc, df_act):
        for c in common_feats:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df.dropna(subset=[c for c in common_feats if c in df.columns], inplace=True)

    # Fit scaler on MF+ZINC only
    scaler = fit_scaler_on_mf_zinc(df_mf, df_zinc, common_feats)
    X_mf = scaler.transform(df_mf[common_feats].to_numpy(dtype=float))
    X_zinc = scaler.transform(df_zinc[common_feats].to_numpy(dtype=float))
    # Actives projected using same scaler
    X_act = scaler.transform(df_act[[c for c in common_feats if c in df_act.columns]].reindex(columns=common_feats, fill_value=np.nan).to_numpy(dtype=float))

    # DR fit on MF+ZINC, projection for actives
    X_train = np.vstack([X_mf, X_zinc])
    if method == "pca":
        model, Z_train = fit_pca(X_train, n_components=dim)
        Z_act = model.transform(X_act)
    elif method == "umap":
        model, Z_train = fit_umap(
            X_train,
            n_components=dim,
            n_neighbors=int(umap_params.get("n_neighbors", 50)),
            min_dist=float(umap_params.get("min_dist", 0.01)),
            metric=umap_params.get("metric", "euclidean"),
            random_state=umap_params.get("random_state", None),
        )
        Z_act = model.transform(X_act)
    else:
        raise ValueError(f"Unsupported method: {method}")

    # Split Z_train back to Z_mf, Z_zinc
    n_mf = len(df_mf)
    Z_mf = Z_train[:n_mf]
    Z_zinc = Z_train[n_mf:]

    # Apply affinity cutoff to MF for scoring only (if column exists)
    if "Standard Value (nM)" in df_mf.columns:
        mask_cut = pd.to_numeric(df_mf["Standard Value (nM)"], errors="coerce") <= affinity_cutoff_nM
        Z_mf_for_scoring = Z_mf[mask_cut.to_numpy(dtype=bool)]
    else:
        Z_mf_for_scoring = Z_mf

    # Optional ZINC sampling
    if sample_zinc and sample_zinc > 0 and sample_zinc < len(df_zinc):
        idx = np.random.default_rng().choice(len(df_zinc), size=sample_zinc, replace=False)
        Z_zinc_eval = Z_zinc[idx]
        df_zinc_eval = df_zinc.iloc[idx].copy()
    else:
        Z_zinc_eval = Z_zinc
        df_zinc_eval = df_zinc.copy()

    # Build evaluation set: actives + zinc
    Z_eval = np.vstack([Z_act, Z_zinc_eval])
    labels = np.concatenate([np.ones(len(Z_act), dtype=int), np.zeros(len(Z_zinc_eval), dtype=int)])

    # 1-NN scoring against MF cloud (filtered by cutoff)
    scores, distances = nn_min_distance_scores(Z_mf_for_scoring, Z_eval)

    # Metrics
    ef1 = ef_at_k_percent(scores, labels, 1.0)
    ef5 = ef_at_k_percent(scores, labels, 5.0)
    ef10 = ef_at_k_percent(scores, labels, 10.0)
    roc = roc_auc(labels, scores)
    pr = pr_auc(labels, scores)

    # Spearman rho on actives only (pActivity vs score)
    if "pActivity" in df_act.columns:
        pact = pd.to_numeric(df_act["pActivity"], errors="coerce").to_numpy()
    elif "Standard Value (nM)" in df_act.columns:
        pact = to_pactivity_from_nM(df_act["Standard Value (nM)"]).to_numpy()
    else:
        pact = np.full(len(df_act), np.nan)
    rho, rho_p = spearman_rho(pact, scores[: len(df_act)])

    metrics = {
        "roc_auc": roc,
        "pr_auc": pr,
        "ef_1%": ef1,
        "ef_5%": ef5,
        "ef_10%": ef10,
        "spearman_rho": rho,
        "spearman_p": rho_p,
        "n_actives": int(len(df_act)),
        "n_zinc_eval": int(len(df_zinc_eval)),
        "n_mf_for_scoring": int(len(Z_mf_for_scoring)),
        "method": method,
        "dim": dim,
        "target": target,
        "affinity_cutoff_nM": affinity_cutoff_nM,
    }

    # Save metrics and ranked CSV
    (ws["metrics"] / "metrics.json").write_text(json.dumps(metrics, indent=2))

    ranked = pd.DataFrame({
        "score": scores,
        "distance": distances,
        "label": labels,
    })
    # Attach identifiers if available
    def safe_col(df: pd.DataFrame, name: str) -> Optional[pd.Series]:
        return df[name] if name in df.columns else None

    act_prefix = {"source": "actives"}
    zinc_prefix = {"source": "zinc"}
    df_eval_ids = pd.concat([
        pd.DataFrame({
            **act_prefix,
            "Compound ChEMBL ID": safe_col(df_act, "Compound ChEMBL ID"),
            "canonical_smiles": safe_col(df_act, "canonical_smiles") or safe_col(df_act, "SMILES"),
        }),
        pd.DataFrame({
            **zinc_prefix,
            "Compound ChEMBL ID": safe_col(df_zinc_eval, "Compound ChEMBL ID"),
            "canonical_smiles": safe_col(df_zinc_eval, "canonical_smiles") or safe_col(df_zinc_eval, "SMILES"),
        })
    ], ignore_index=True)
    ranked = pd.concat([df_eval_ids, ranked], axis=1)
    ranked.sort_values("score", ascending=False, inplace=True)
    ranked.to_csv(ws["artifacts"] / "ranked_scores.csv", index=False)

    # Log summary
    summary = {
        "molfuse_version": __version__,
        "invariants": {
            "scaler_fit": "MF+ZINC only",
            "umap_seed": None,
            "scoring": "exact_1NN in embedded space; score=-distance",
            "affinity_cutoff_application": "MF cloud only (for scoring)",
            "metrics": ["EF@1%", "EF@5%", "EF@10%", "ROC-AUC", "PR-AUC", "Spearman rho"],
            "target_preserving_exclusion": True,
        },
        "config": cfg,
        "metrics_path": str(ws["metrics"] / "metrics.json"),
        "ranked_path": str(ws["artifacts"] / "ranked_scores.csv"),
    }
    (ws["logs"] / "phase1_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
