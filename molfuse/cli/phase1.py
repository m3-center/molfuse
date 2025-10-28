from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import gc

import numpy as np
import pandas as pd
import joblib

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
    vals = 9.0 - np.log10(s)
    # Ensure we return a pandas Series to satisfy type checkers
    return pd.Series(vals, index=series.index, name=getattr(series, "name", None))


def remove_overlap_by_smiles(df: pd.DataFrame, smiles_col: str, to_exclude: pd.Series) -> pd.DataFrame:
    if smiles_col in df.columns:
        return df[~df[smiles_col].isin(set(to_exclude.dropna().astype(str)))]
    return df


def get_smiles_col(df: pd.DataFrame) -> Optional[str]:
    if "canonical_smiles" in df.columns:
        return "canonical_smiles"
    if "SMILES" in df.columns:
        return "SMILES"
    return None

def dedup_by_smiles(df: pd.DataFrame, label: str, logger: logging.Logger) -> pd.DataFrame:
    smiles_col = get_smiles_col(df)
    if smiles_col is None:
        raise RuntimeError(f"{label}: cannot deduplicate by SMILES; no 'canonical_smiles' or 'SMILES' column present")
    before = len(df)
    # Keep first occurrence of each SMILES string
    deduped = df.drop_duplicates(subset=[smiles_col]).copy()
    after = len(deduped)
    logger.info(f"{label}: deduplicated by {smiles_col}: {before} -> {after} (removed {before-after})")
    return deduped


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    with cfg_path.open("r") as f:
        cfg = json.load(f)

    run_name = cfg.get("run_name", f"{cfg.get('target','target')}_{cfg.get('method','pca')}_{cfg.get('dim',2)}d")
    ws = make_run_dirs(Path(args.workspace), phase="phase1", run_name=run_name)

    # Logger setup with both file and console handlers
    log_path = ws["logs"] / "run.log"
    logger = logging.getLogger(f"molfuse.phase1.{run_name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    # File handler for detailed log
    fh = logging.FileHandler(log_path, mode="w")
    fmt = logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    
    # Console handler for SLURM stdout
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    
    logger.info("Phase 1 started")
    representation = cfg.get("representation", "features").lower()  # "features" | "fingerprints"

    # Required inputs
    mf_csv = Path(cfg["mf_features_csv"]) if "mf_features_csv" in cfg else None
    zinc_csv = Path(cfg["zinc_features_csv"]) if "zinc_features_csv" in cfg else None
    actives_csv = Path(cfg.get("actives_features_csv", "")) if "actives_features_csv" in cfg else None
    if not (mf_csv and zinc_csv and mf_csv.exists() and zinc_csv.exists()):
        err = {
            "error": "Missing required CSV inputs",
            "required_keys": ["mf_features_csv", "zinc_features_csv"],
            "cfg": cfg,
        }
        (ws["logs"] / "phase1_error.json").write_text(json.dumps(err, indent=2))
        raise FileNotFoundError("Provide valid mf_features_csv and zinc_features_csv in config.")

    target = cfg.get("target", "UNKNOWN")
    accession = extract_accession(target)
    method = cfg.get("method", "pca").lower()
    dim = int(cfg.get("dim", 2))
    umap_params = cfg.get("umap_params", {})
    affinity_cutoff_nM = float(cfg.get("affinity_cutoff_nM", 100000))
    on_empty_cutoff = str(cfg.get("on_empty_cutoff", "error")).lower()  # "error" | "fallback"
    sample_zinc = int(cfg.get("sample_zinc", 0))

    # Load
    df_mf_all = pd.read_csv(mf_csv, low_memory=False)
    df_zinc = pd.read_csv(zinc_csv, low_memory=False)
    logger.info(f"Loaded MF file: {mf_csv} (rows={len(df_mf_all)})")
    logger.info(f"Loaded ZINC file: {zinc_csv} (rows={len(df_zinc)})")

    # Split MF vs Actives from MF file if actives CSV not provided or missing
    df_act: pd.DataFrame
    df_mf = df_mf_all.copy()
    if accession is not None:
        col_acc = "accession" if "accession" in df_mf_all.columns else ("Target Accession" if "Target Accession" in df_mf_all.columns else None)
        if col_acc is not None:
            # derive actives from MF file when not provided
            if not (actives_csv and actives_csv.exists()):
                df_act = df_mf_all[df_mf_all[col_acc] == accession].copy()
                df_mf = df_mf_all[df_mf_all[col_acc] != accession].copy()
                logger.info(f"Derived actives from MF by accession {accession}: rows={len(df_act)}")
                logger.info(f"MF after target exclusion by accession: rows={len(df_mf)}")
            else:
                df_act = pd.read_csv(actives_csv, low_memory=False)
                df_mf = df_mf_all[df_mf_all[col_acc] != accession].copy()
                logger.info(f"Loaded actives file: {actives_csv} (rows={len(df_act)})")
                logger.info(f"MF after target exclusion by accession: rows={len(df_mf)}")
        else:
            # No accession column; require explicit actives CSV
            if not (actives_csv and actives_csv.exists()):
                err = {
                    "error": "Cannot derive actives: accession column not found in MF file and no actives_features_csv provided.",
                    "mf_columns": list(df_mf_all.columns),
                    "cfg": cfg,
                }
                (ws["logs"] / "phase1_error.json").write_text(json.dumps(err, indent=2))
                raise FileNotFoundError("Provide actives_features_csv when MF file lacks an accession column.")
            df_act = pd.read_csv(actives_csv, low_memory=False)
            logger.info(f"Loaded actives file (no accession in MF): {actives_csv} (rows={len(df_act)})")
    else:
        # No accession derivable from target string; require explicit actives CSV
        if not (actives_csv and actives_csv.exists()):
            err = {
                "error": "Cannot derive actives: target accession not parseable and no actives_features_csv provided.",
                "target": target,
                "cfg": cfg,
            }
            (ws["logs"] / "phase1_error.json").write_text(json.dumps(err, indent=2))
            raise FileNotFoundError("Provide actives_features_csv or use target with accession suffix (e.g., ..._P00519).")
        else:
            df_act = pd.read_csv(actives_csv, low_memory=False)
            logger.info(f"Loaded actives file (no accession in target): {actives_csv} (rows={len(df_act)})")

    # Overlap removal by SMILES (actives vs MF+ZINC)
    smiles_col_act = get_smiles_col(df_act)
    if smiles_col_act is None:
        raise RuntimeError("Actives: no SMILES column found (canonical_smiles or SMILES required)")
    act_smiles = df_act[smiles_col_act]
    mf_before_ov = len(df_mf)
    zinc_before_ov = len(df_zinc)
    df_mf = remove_overlap_by_smiles(df_mf, smiles_col_act, act_smiles)
    df_zinc = remove_overlap_by_smiles(df_zinc, smiles_col_act, act_smiles)
    logger.info(
        f"Removed overlaps with actives by {smiles_col_act}: MF {mf_before_ov}->{len(df_mf)}, ZINC {zinc_before_ov}->{len(df_zinc)}"
    )
    # Additionally, enforce zero overlap between ZINC and MF by SMILES
    smiles_mf_col = get_smiles_col(df_mf)
    if smiles_mf_col is None:
        raise RuntimeError("MF: no SMILES column found (canonical_smiles or SMILES required)")
    zinc_before_mfov = len(df_zinc)
    df_zinc = remove_overlap_by_smiles(df_zinc, smiles_mf_col, df_mf[smiles_mf_col])
    logger.info(f"Removed ZINC-MF overlaps by {smiles_mf_col}: ZINC {zinc_before_mfov}->{len(df_zinc)}")

    # Deduplication strictly by SMILES
    df_mf = dedup_by_smiles(df_mf, label="MF", logger=logger)
    df_act = dedup_by_smiles(df_act, label="Actives", logger=logger)

    # Prepare feature matrices depending on representation
    if representation == "features":
        # Feature columns (numeric-only, exclude known non-features)
        feat_cols_mf = select_feature_columns(df_mf)
        feat_cols_zinc = select_feature_columns(df_zinc)
        common_feats = [c for c in feat_cols_mf if c in feat_cols_zinc]
        if not common_feats:
            raise RuntimeError("No common numeric feature columns found between MF and ZINC.")
        logger.info(f"Common numeric columns (features): n={len(common_feats)}")

        # Coerce numerics and drop NaNs
        def coerce_and_drop(df: pd.DataFrame, name: str) -> int:
            before = len(df)
            for c in common_feats:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            df.dropna(subset=[c for c in common_feats if c in df.columns], inplace=True)
            after = len(df)
            logger.info(f"{name}: dropped rows with NaNs in features: {before}->{after} (removed {before-after})")
            return after

        coerce_and_drop(df_mf, "MF")
        coerce_and_drop(df_zinc, "ZINC")
        coerce_and_drop(df_act, "Actives")

        # Fit/prepare scaler for features
        logger.info("Starting StandardScaler fit on MF+ZINC (features)")
        scaler = fit_scaler_on_mf_zinc(df_mf, df_zinc, common_feats)
        logger.info("Fitted StandardScaler on MF+ZINC (features)")
        X_mf = scaler.transform(df_mf[common_feats].to_numpy(dtype=float))
        X_zinc = scaler.transform(df_zinc[common_feats].to_numpy(dtype=float))
        # Actives projected using same scaler
        df_act_feat = df_act[[c for c in common_feats if c in df_act.columns]].reindex(columns=common_feats)
        for c in df_act_feat.columns:
            df_act_feat[c] = pd.to_numeric(df_act_feat[c], errors="coerce")
        df_act_feat = df_act_feat.dropna(axis=0, how="any")
        df_act = df_act.loc[df_act_feat.index].copy()
        X_act = scaler.transform(df_act_feat.to_numpy(dtype=float))
        # Save scaler
        scaler_path = ws["artifacts"] / "scaler.joblib"
        joblib.dump(scaler, scaler_path, compress=5)
        logger.info(f"Saved scaler to {scaler_path}")

        # Step 1 memory free: drop heavy feature columns from DataFrames (keep metadata only)
        drop_mf = [c for c in common_feats if c in df_mf.columns]
        drop_zinc = [c for c in common_feats if c in df_zinc.columns]
        drop_act = [c for c in common_feats if c in df_act.columns]
        if drop_mf:
            df_mf.drop(columns=drop_mf, inplace=True, errors="ignore")
        if drop_zinc:
            df_zinc.drop(columns=drop_zinc, inplace=True, errors="ignore")
        if drop_act:
            df_act.drop(columns=drop_act, inplace=True, errors="ignore")
        logger.info(
            f"Freed feature columns from DataFrames (kept IDs/SMILES). Dropped: MF={len(drop_mf)}, ZINC={len(drop_zinc)}, Actives={len(drop_act)}"
        )
        gc.collect()
    elif representation == "fingerprints":
        # Parse fingerprint strings (comma-separated 0/1) into numeric arrays
        def find_fp_col(df: pd.DataFrame) -> Optional[str]:
            candidates = [
                "Fingerprint", "fingerprint", "ECFP4", "ecfp4", "FP", "fp"
            ]
            for c in candidates:
                if c in df.columns:
                    return c
            return None

        fp_col_mf = find_fp_col(df_mf)
        fp_col_zinc = find_fp_col(df_zinc)
        if not fp_col_mf or not fp_col_zinc or fp_col_mf != fp_col_zinc:
            err = {
                "error": "Fingerprint column not found or inconsistent between MF and ZINC",
                "mf_cols": list(df_mf.columns),
                "zinc_cols": list(df_zinc.columns),
            }
            (ws["logs"] / "phase1_error.json").write_text(json.dumps(err, indent=2))
            raise RuntimeError("Fingerprint column not found or inconsistent between MF and ZINC (expected e.g., 'Fingerprint').")
        fp_col = fp_col_mf
        logger.info(f"Using fingerprint column: {fp_col}")

        def parse_fp_series(series: pd.Series, label: str) -> Tuple[np.ndarray, pd.Index]:
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
                    # Sanitize string: remove spaces, trim brackets, drop trailing commas, keep only 0/1/,
                    ss = s.replace(" ", "").strip()
                    if ss.startswith("[") and ss.endswith("]"):
                        ss = ss[1:-1]
                    # Remove any characters not 0,1, or comma (robust to stray quotes or text)
                    ss = re.sub(r"[^01,]", "", ss)
                    ss = ss.strip(",")
                    if not ss:
                        continue
                    arr = np.fromstring(ss, sep=",", dtype=np.uint8)
                    if expected_len is None:
                        expected_len = int(arr.shape[0])
                    if arr.shape[0] != expected_len or expected_len == 0:
                        continue  # skip inconsistent length rows
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

    # Parse MF and ZINC
        X_mf, idx_mf = parse_fp_series(df_mf[fp_col], "MF")
        df_mf = df_mf.loc[idx_mf].copy()
        X_zinc, idx_zinc = parse_fp_series(df_zinc[fp_col], "ZINC")
        df_zinc = df_zinc.loc[idx_zinc].copy()

        # Parse Actives
        if fp_col in df_act.columns:
            X_act, idx_act = parse_fp_series(df_act[fp_col], "Actives")
            df_act = df_act.loc[idx_act].copy()
        else:
            # If actives were derived and somehow lack fp column, align to empty
            X_act = np.zeros((0, X_mf.shape[1]), dtype=np.float32)
            logger.info("Actives: fingerprint column not found; no actives will be evaluated after parsing")

        # Passthrough scaler marker for fingerprints (no scaling)
        scaler_path = ws["artifacts"] / "scaler.joblib"
        joblib.dump({"type": "passthrough"}, scaler_path, compress=5)
        logger.info(f"Saved passthrough scaler marker to {scaler_path}")

        # Step 1 memory free: drop fingerprint column from DataFrames (keep metadata only)
        if fp_col in df_mf.columns:
            df_mf.drop(columns=[fp_col], inplace=True, errors="ignore")
        if fp_col in df_zinc.columns:
            df_zinc.drop(columns=[fp_col], inplace=True, errors="ignore")
        if fp_col in df_act.columns:
            df_act.drop(columns=[fp_col], inplace=True, errors="ignore")
        logger.info("Freed fingerprint column from DataFrames (kept IDs/SMILES)")
        gc.collect()
    else:
        raise ValueError(f"Unsupported representation: {representation}")

    # DR fit on MF+ZINC, projection for actives
    X_train = np.vstack([X_mf, X_zinc])
    # Step 1 extension: free per-set matrices now that the concatenated training matrix exists
    try:
        del X_mf
        del X_zinc
    except Exception:
        pass
    gc.collect()
    logger.info("Freed X_mf and X_zinc after building X_train (concat)")
    if method == "pca":
        logger.info(f"Starting PCA fit: dim={dim}")
        model, Z_train = fit_pca(X_train, n_components=dim)
        Z_act = model.transform(X_act)
        logger.info(f"PCA fitted: dim={dim}")
        model_path = ws["artifacts"] / "pca_model.joblib"
        joblib.dump(model, model_path, compress=5)
        logger.info(f"Saved PCA model to {model_path}")
    elif method == "umap":
        logger.info(
            f"Starting UMAP fit: dim={dim}, n_neighbors={int(umap_params.get('n_neighbors', 50))}, min_dist={float(umap_params.get('min_dist', 0.01))}, metric={umap_params.get('metric', 'jaccard' if representation=='fingerprints' else 'euclidean')}"
        )
        model, Z_train = fit_umap(
            X_train,
            n_components=dim,
            n_neighbors=int(umap_params.get("n_neighbors", 50)),
            min_dist=float(umap_params.get("min_dist", 0.01)),
            metric=umap_params.get("metric", "jaccard" if representation == "fingerprints" else "euclidean"),
            random_state=umap_params.get("random_state", None),
        )
        Z_act = model.transform(X_act)
        logger.info(
            f"UMAP fitted: dim={dim}, n_neighbors={int(umap_params.get('n_neighbors', 50))}, min_dist={float(umap_params.get('min_dist', 0.01))}, metric={umap_params.get('metric', 'jaccard' if representation=='fingerprints' else 'euclidean')}, random_state=None"
        )
        model_path = ws["artifacts"] / "umap_model.joblib"
        joblib.dump(model, model_path, compress=5)
        logger.info(f"Saved UMAP model to {model_path}")
    else:
        raise ValueError(f"Unsupported method: {method}")

    # Step 2 memory free: drop the concatenated training matrix after DR fit
    try:
        del X_train
    except Exception:
        pass
    gc.collect()
    logger.info("Freed X_train (concat) from memory after DR fit")

    # Also drop raw actives matrix after projection
    try:
        del X_act
    except Exception:
        pass
    gc.collect()
    logger.info("Freed X_act (raw) after projection")

    # Split Z_train back to Z_mf, Z_zinc
    n_mf = len(df_mf)
    Z_mf = Z_train[:n_mf]
    Z_zinc = Z_train[n_mf:]

    # Apply affinity cutoff to MF for scoring only (if column exists)
    if "Standard Value (nM)" in df_mf.columns:
        mask_cut = pd.to_numeric(df_mf["Standard Value (nM)"], errors="coerce") <= affinity_cutoff_nM
        Z_mf_for_scoring = Z_mf[mask_cut.to_numpy(dtype=bool)]
        if Z_mf_for_scoring.size == 0:
            msg = {
                "error": "Empty MF after applying affinity cutoff",
                "affinity_cutoff_nM": affinity_cutoff_nM,
                "n_mf_total": int(len(df_mf)),
                "n_mf_passing": 0,
                "policy": on_empty_cutoff,
            }
            (ws["logs"] / "phase1_error.json").write_text(json.dumps(msg, indent=2))
            if on_empty_cutoff == "fallback":
                Z_mf_for_scoring = Z_mf
                logger.info("Cutoff yielded empty MF; using fallback to full MF for scoring")
            else:
                raise RuntimeError(
                    f"No MF compounds pass the cutoff {affinity_cutoff_nM} nM; set on_empty_cutoff='fallback' to override."
                )
    else:
        # No cutoff column; proceed without filtering
        Z_mf_for_scoring = Z_mf
    logger.info(f"MF for scoring: n={len(Z_mf_for_scoring)} (cutoff={affinity_cutoff_nM} nM)")

    # Optional ZINC sampling
    if sample_zinc and sample_zinc > 0 and sample_zinc < len(df_zinc):
        idx = np.random.default_rng().choice(len(df_zinc), size=sample_zinc, replace=False)
        Z_zinc_eval = Z_zinc[idx]
        df_zinc_eval = df_zinc.iloc[idx].copy()
        logger.info(f"Sampled ZINC: {len(df_zinc)} -> {len(df_zinc_eval)} for evaluation")
    else:
        Z_zinc_eval = Z_zinc
        df_zinc_eval = df_zinc.copy()
        logger.info(f"Using all ZINC for evaluation: n={len(df_zinc_eval)}")

    # Build evaluation set: actives + zinc
    Z_eval = np.vstack([np.asarray(Z_act), np.asarray(Z_zinc_eval)])
    labels = np.concatenate([np.ones(len(Z_act), dtype=int), np.zeros(len(Z_zinc_eval), dtype=int)])

    # 1-NN scoring against MF cloud (filtered by cutoff)
    scores, distances = nn_min_distance_scores(Z_mf_for_scoring, Z_eval)
    # Clean up negative zero for readability
    scores[np.isclose(scores, 0.0)] = 0.0
    distances[np.isclose(distances, 0.0)] = 0.0

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
    logger.info(f"Metrics saved: {ws['metrics'] / 'metrics.json'}")

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
    logger.info(f"Ranked scores saved: {ws['artifacts'] / 'ranked_scores.csv'} (rows={len(ranked)})")

    # Save similarity space coordinates for MF, ZINC, and Actives
    def save_embedding(df_src: pd.DataFrame, Z: np.ndarray, name: str) -> None:
        smiles_c = get_smiles_col(df_src)
        cols: Dict[str, pd.Series] = {}
        if smiles_c:
            cols[smiles_c] = df_src[smiles_c].reset_index(drop=True)
        if "Compound ChEMBL ID" in df_src.columns:
            cols["Compound ChEMBL ID"] = df_src["Compound ChEMBL ID"].reset_index(drop=True)
        emb = pd.DataFrame(Z, columns=[f"z{i}" for i in range(Z.shape[1])])
        out = pd.concat([pd.DataFrame(cols), emb], axis=1)
        out_path = ws["artifacts"] / f"embedding_{name}.csv"
        out.to_csv(out_path, index=False)
        logger.info(f"Saved embedding for {name}: {out_path} (rows={len(out)}, dim={Z.shape[1]})")

    save_embedding(df_mf.reset_index(drop=True), np.asarray(Z_mf), "mf")
    save_embedding(df_zinc.reset_index(drop=True), np.asarray(Z_zinc), "zinc")
    save_embedding(df_act.reset_index(drop=True), np.asarray(Z_act), "actives")

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
    logger.info("="*80)
    logger.info("PHASE 1 COMPLETED SUCCESSFULLY")
    logger.info("="*80)


if __name__ == "__main__":
    main()
