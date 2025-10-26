#!/usr/bin/env python3
"""
Independent PCA vs UMAP comparison for TyrosineProteinKinaseABL1 (P00519)

This is a self-contained script (no imports from the repository) that:
- Loads a Molecular Function (MF) features CSV covering the relevant MF category (e.g., Transferase)
- Loads a ZINC features CSV (precomputed descriptors)
- Identifies held-out target actives as MF entries where accession == P00519
- Excludes target actives from MF cloud and removes any MF/actives SMILES from ZINC
- Deduplicates MF by Compound ChEMBL ID (median affinity if present; features take first)
- Fits PCA (5D) and UMAP (5D, n_neighbors=5, min_dist=0.1) on MF cloud + ZINC (training set)
- Projects held-out actives
- Scores all compounds by negative distance to nearest MF cloud neighbor in embedded space
- Computes EF@1%, ROC-AUC, PR-AUC
- Writes detailed logs and outputs under an independent working directory

Requirements: pandas, numpy, scikit-learn, umap-learn
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc as sklearn_auc

try:
    import umap
except ImportError:
    umap = None

ID_COLS_CHEMBL = ["Compound ChEMBL ID", "SMILES", "accession", "Standard Value (nM)"]
# Additional non-feature columns to always exclude from molecular features
NON_FEATURE_EXCLUDE = [
    "ZINC_ID",
    "MOLECULE ID",
    "DataSource",
    "LABEL",
    "MANUFACTURER",
    "TRANCHE",
    "Target ChEMBL ID",
    "Target Name",
    "Activity Type",
    "target_chembl_id",
]


def setup_logger(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "independent_pca_umap.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)-8s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, mode='w'),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.info(f"Logging to {log_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Independent PCA vs UMAP comparison for TyrosineProteinKinaseABL1 (P00519).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--mf_features_csv', required=True, help='MF features CSV (must include accession, SMILES, Compound ChEMBL ID, and numeric feature columns).')
    p.add_argument('--zinc_features_csv', required=True, help='ZINC features CSV (must include SMILES and the SAME numeric feature columns).')
    p.add_argument('--target_uniprot', default='P00519', help='Target UniProt accession (TyrosineProteinKinaseABL1).')
    p.add_argument('--output_root', default='tests/independent_tyro_pca_umap/work', help='Root output directory for logs and results.')
    p.add_argument('--pca_dim', type=int, default=5, help='PCA embedding dimension.')
    p.add_argument('--umap_dim', type=int, default=5, help='UMAP embedding dimension.')
    p.add_argument('--umap_n_neighbors', type=int, default=5, help='UMAP n_neighbors.')
    p.add_argument('--umap_min_dist', type=float, default=0.1, help='UMAP min_dist.')
    p.add_argument('--umap_init', choices=['spectral', 'tswspectral', 'pca', 'random'], default='spectral', help='UMAP initialization method. Use random to avoid spectral embedding warnings.')
    p.add_argument('--random_seed', type=str, default='42', help='Random seed for reproducibility. Use "none" to disable seeding (enables UMAP parallelism).')
    p.add_argument('--sample_zinc', type=int, default=None, help='Optional: sample this many ZINC rows for speed.')
    p.add_argument('--feature_accept_ratio', type=float, default=0.95, help='Minimum fraction of non-missing entries in a column that must be numeric to accept it as a feature.')
    p.add_argument('--skip_dedup', action='store_true', help='Skip deduplication by Compound ChEMBL ID (for diagnostics).')
    return p.parse_args()


def load_csv_with_info(path: str) -> pd.DataFrame:
    t0 = time.perf_counter()
    df = pd.read_csv(path, low_memory=False)
    logging.info(f"Loaded CSV: {path} (shape={df.shape}) in {time.perf_counter()-t0:.2f}s")
    return df


def find_feature_columns(df: pd.DataFrame, accept_ratio: float = 0.95) -> List[str]:
    # Heuristic: numeric columns excluding common ID/affinity columns
    exclude = set(ID_COLS_CHEMBL + NON_FEATURE_EXCLUDE) & set(df.columns)
    accepted: List[str] = []
    dropped: List[Tuple[str, float, str]] = []  # (col, ratio, dtype)
    for c in df.columns:
        if c in exclude:
            continue
        ser = df[c]
        # Only consider rows that are not missing for this column
        notna = ser.notna()
        total = int(notna.sum())
        if total == 0:
            dropped.append((c, 0.0, str(ser.dtype)))
            continue
        coerced = pd.to_numeric(ser, errors='coerce')
        numeric_ok = (notna & coerced.notna()).sum()
        ratio = float(numeric_ok) / float(total) if total > 0 else 0.0
        if ratio >= accept_ratio:
            accepted.append(c)
        else:
            dropped.append((c, ratio, str(ser.dtype)))
    logging.info(f"Detected {len(accepted)} feature columns using accept_ratio={accept_ratio}")
    if dropped:
        # Show a concise summary of dropped columns (limit to first 20 for readability)
        preview = ", ".join([f"{name}({ratio:.2%},{dtype})" for name, ratio, dtype in dropped[:20]])
        more = "" if len(dropped) <= 20 else f" ... (+{len(dropped)-20} more)"
        logging.info(f"Dropped {len(dropped)} candidate columns (insufficient numeric fraction): {preview}{more}")
    return accepted


def deduplicate_mf(df_mf: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    before = len(df_mf)
    agg_map: Dict[str, str] = {}
    for c in feature_cols:
        agg_map[c] = 'first'  # features should be invariant per compound
    if 'Standard Value (nM)' in df_mf.columns:
        agg_map['Standard Value (nM)'] = 'median'
    grouped = df_mf.groupby('Compound ChEMBL ID', as_index=False).agg({**agg_map, 'SMILES': 'first', 'accession': 'first'})
    after = len(grouped)
    logging.info(f"Deduplicated MF by Compound ChEMBL ID: {before} -> {after}")
    return grouped


def deduplicate_preserving_target(df_mf: pd.DataFrame, feature_cols: List[str], target_uniprot: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Deduplicate while ensuring any compound (Compound ChEMBL ID) that relates to the target_uniprot
    is assigned to the held-out set (not MF cloud). For actives, prefer rows with the target accession
    when aggregating.
    """
    if 'Compound ChEMBL ID' not in df_mf.columns or 'accession' not in df_mf.columns:
        raise ValueError("MF CSV must include 'Compound ChEMBL ID' and 'accession'.")

    # Aggregations for features and common fields
    agg_features: Dict[str, str] = {c: 'first' for c in feature_cols}
    agg_common: Dict[str, str] = {}
    if 'SMILES' in df_mf.columns:
        agg_common['SMILES'] = 'first'
    if 'Standard Value (nM)' in df_mf.columns:
        agg_common['Standard Value (nM)'] = 'median'

    # Actives: restrict to target rows, then group
    df_target_rows = df_mf[df_mf['accession'] == target_uniprot].copy()
    actives = pd.DataFrame(columns=['Compound ChEMBL ID'] + list(agg_features.keys()) + list(agg_common.keys()) + ['accession'])
    if not df_target_rows.empty:
        actives = df_target_rows.groupby('Compound ChEMBL ID', as_index=False).agg({**agg_features, **agg_common})
        actives['accession'] = target_uniprot

    # MF cloud: use non-target rows excluding any Compound IDs already assigned to actives
    ids_actives = set(actives['Compound ChEMBL ID'].astype(str).tolist())
    df_non_target = df_mf[df_mf['accession'] != target_uniprot].copy()
    cloud = pd.DataFrame(columns=['Compound ChEMBL ID'] + list(agg_features.keys()) + list(agg_common.keys()) + ['accession'])
    if not df_non_target.empty:
        df_non_target['__compound_id_str__'] = df_non_target['Compound ChEMBL ID'].astype(str)
        df_non_target_excl = df_non_target[~df_non_target['__compound_id_str__'].isin(ids_actives)].drop(columns=['__compound_id_str__'])
        if not df_non_target_excl.empty and 'accession' in df_non_target_excl.columns:
            cloud = df_non_target_excl.groupby('Compound ChEMBL ID', as_index=False).agg({**agg_features, **agg_common, 'accession': 'first'})

    return cloud, actives


def exclude_target_and_overlap(df_mf: pd.DataFrame, df_zinc: pd.DataFrame, target_uniprot: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Split actives vs MF cloud
    if 'accession' not in df_mf.columns:
        raise ValueError("MF CSV must include 'accession' to identify target ligands")
    df_actives = df_mf[df_mf['accession'] == target_uniprot].copy()
    df_mf_cloud = df_mf[df_mf['accession'] != target_uniprot].copy()
    logging.info(f"Actives (held-out) rows: {len(df_actives)} | MF cloud rows (pre-overlap removal): {len(df_mf_cloud)}")

    # Remove overlap by SMILES to avoid leakage
    smiles_mf = set(df_mf_cloud['SMILES'].dropna().astype(str).unique())
    smiles_act = set(df_actives['SMILES'].dropna().astype(str).unique())
    smiles_exclude = smiles_mf.union(smiles_act)
    before_zinc = len(df_zinc)
    df_zinc_clean = df_zinc[~df_zinc['SMILES'].astype(str).isin(smiles_exclude)].copy()
    logging.info(f"ZINC overlap removal by SMILES: {before_zinc} -> {len(df_zinc_clean)}")
    return df_mf_cloud, df_zinc_clean, df_actives


def fit_transform_embeddings(X_train: np.ndarray, X_act: np.ndarray, method: str, pca_dim: int, umap_dim: int, n_neighbors: int, min_dist: float, seed: Optional[int], umap_init: str = 'spectral') -> Tuple[np.ndarray, np.ndarray, str]:
    if method == 'pca':
        scaler = StandardScaler()
        Xs_train = scaler.fit_transform(X_train)
        Xs_act = scaler.transform(X_act)
        pca = PCA(n_components=pca_dim, random_state=seed)
        Z_train = np.asarray(pca.fit_transform(Xs_train), dtype=np.float32)
        Z_act = np.asarray(pca.transform(Xs_act), dtype=np.float32)
        tag = f"PCA-{pca_dim}D"
        return Z_train, Z_act, tag
    elif method == 'umap':
        if umap is None:
            raise RuntimeError("umap-learn is not installed. Please install 'umap-learn'.")
        scaler = StandardScaler()
        Xs_train = scaler.fit_transform(X_train)
        Xs_act = scaler.transform(X_act)
        reducer = umap.UMAP(n_components=umap_dim, n_neighbors=n_neighbors, min_dist=min_dist, metric='euclidean', random_state=seed, init=umap_init)
        Z_train = np.asarray(reducer.fit_transform(Xs_train), dtype=np.float32)
        Z_act = np.asarray(reducer.transform(Xs_act), dtype=np.float32)
        tag = f"UMAP-nn{n_neighbors}-md{min_dist}-{umap_dim}D"
        return Z_train, Z_act, tag
    else:
        raise ValueError("Unknown method: expected 'pca' or 'umap'")


def score_by_nn(Z_query: np.ndarray, Z_ref: np.ndarray) -> np.ndarray:
    if Z_query.size == 0 or Z_ref.size == 0:
        return np.array([], dtype=np.float32)
    nn = NearestNeighbors(n_neighbors=1, algorithm='auto', metric='euclidean')
    nn.fit(Z_ref)
    dists, _ = nn.kneighbors(Z_query, return_distance=True)
    return (-dists.astype(np.float32).ravel())


def build_ranking_and_metrics(scores_act: np.ndarray, scores_zinc: np.ndarray) -> Tuple[pd.DataFrame, Dict[str, float]]:
    # Combine into a single ranking list
    y = np.array([1] * len(scores_act) + [0] * len(scores_zinc))
    scores = np.concatenate([scores_act, scores_zinc])
    df = pd.DataFrame({
        'TYPE': ['HELDOUT_ACTIVE'] * len(scores_act) + ['DECOY'] * len(scores_zinc),
        'score': scores
    })
    df = df.dropna(subset=['score']).sort_values(by='score', ascending=False).reset_index(drop=True)
    df['RANK'] = np.arange(1, len(df) + 1)

    # Metrics
    valid = ~np.isnan(scores)
    y_valid = y[valid]
    s_valid = scores[valid]
    metrics: Dict[str, float] = {}
    if len(np.unique(y_valid)) >= 2 and len(y_valid) >= 2:
        try:
            metrics['roc_auc'] = float(roc_auc_score(y_valid, s_valid))
            p, r, _ = precision_recall_curve(y_valid, s_valid)
            metrics['pr_auc'] = float(sklearn_auc(r, p))
        except Exception:
            metrics['roc_auc'] = np.nan
            metrics['pr_auc'] = np.nan
    else:
        metrics['roc_auc'] = np.nan
        metrics['pr_auc'] = np.nan

    def ef_at(pct: float) -> float:
        N = len(s_valid)
        if N == 0:
            return np.nan
        k = int(np.ceil(pct * N))
        if k == 0:
            return np.nan
        top_is_active = (df['TYPE'].iloc[:k] == 'HELDOUT_ACTIVE').sum()
        total_actives = (y_valid == 1).sum()
        if total_actives == 0:
            return 1.0
        ef_den = total_actives / N
        return float((top_is_active / k) / ef_den) if ef_den > 0 else np.nan

    for pct in [0.01, 0.05, 0.10]:
        metrics[f'ef_{int(pct*100)}%'] = ef_at(pct)

    return df, metrics


def main():
    args = parse_args()
    # Interpret random seed (allow disabling by passing "none")
    seed: Optional[int]
    if args.random_seed is None:
        seed = None
    elif isinstance(args.random_seed, int):
        seed = args.random_seed
    else:
        rs = str(args.random_seed).strip().lower()
        seed = None if rs in ("none", "null", "na", "") else int(rs)

    # Output structure
    out_root = os.path.abspath(args.output_root)
    run_dir = os.path.join(out_root, f"tyro_pca_vs_umap_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run_dir, exist_ok=True)
    setup_logger(run_dir)

    logging.info("=== Independent PCA vs UMAP Test (TyrosineProteinKinaseABL1 / P00519) ===")
    logging.info(f"MF features CSV: {args.mf_features_csv}")
    logging.info(f"ZINC features CSV: {args.zinc_features_csv}")
    logging.info(f"Target UniProt: {args.target_uniprot}")
    if seed is None:
        logging.info("Random seed: disabled (UMAP may use parallelism; results will be non-deterministic).")
    else:
        logging.info(f"Random seed: {seed} (reproducible; UMAP parallelism may be limited).")

    # Load data
    df_mf_raw = load_csv_with_info(args.mf_features_csv)
    df_zinc_raw = load_csv_with_info(args.zinc_features_csv)

    # Identify features present in both
    feat_mf = find_feature_columns(df_mf_raw, accept_ratio=args.feature_accept_ratio)
    feat_zinc = find_feature_columns(df_zinc_raw, accept_ratio=args.feature_accept_ratio)
    feature_cols = sorted(list(set(feat_mf).intersection(feat_zinc)))
    if not feature_cols:
        logging.error("No overlapping numeric feature columns between MF and ZINC.")
        sys.exit(2)
    only_mf = sorted(list(set(feat_mf) - set(feat_zinc)))
    only_zinc = sorted(list(set(feat_zinc) - set(feat_mf)))
    logging.info(f"Using {len(feature_cols)} common feature columns for DR.")
    if only_mf:
        logging.info(f"Numeric features present only in MF (dropped): {only_mf}")
    if only_zinc:
        logging.info(f"Numeric features present only in ZINC (dropped): {only_zinc}")

    # Deduplication and split preserving target membership
    if 'Compound ChEMBL ID' not in df_mf_raw.columns or 'accession' not in df_mf_raw.columns:
        logging.error("MF CSV must include 'Compound ChEMBL ID' and 'accession'.")
        sys.exit(2)
    n_act_pre_split = int((df_mf_raw['accession'] == args.target_uniprot).sum())
    if args.skip_dedup:
        df_actives = df_mf_raw[df_mf_raw['accession'] == args.target_uniprot].copy()
        df_mf_cloud = df_mf_raw[df_mf_raw['accession'] != args.target_uniprot].copy()
        logging.info("Skipping deduplication by request (--skip_dedup)")
    else:
        df_mf_cloud, df_actives = deduplicate_preserving_target(df_mf_raw, feature_cols, args.target_uniprot)
    n_act_post_split = int(df_actives.shape[0])
    logging.info(f"Held-out target rows (pre-split vs post-dedup held-out): {n_act_pre_split} -> {n_act_post_split}")

    # Ensure SMILES column exists in ZINC
    if 'SMILES' not in df_zinc_raw.columns:
        logging.error("ZINC CSV must include 'SMILES'.")
        sys.exit(2)

    # Optionally sample ZINC for speed
    if args.sample_zinc is not None and args.sample_zinc > 0 and len(df_zinc_raw) > args.sample_zinc:
        df_zinc_raw = df_zinc_raw.sample(n=args.sample_zinc, random_state=seed)
        logging.info(f"Sampled ZINC: {args.sample_zinc} rows")

    # Remove overlaps from ZINC using SMILES from MF cloud and actives
    n_act_pre_na = int(df_actives.shape[0])
    smiles_mf = set(df_mf_cloud['SMILES'].dropna().astype(str).unique()) if 'SMILES' in df_mf_cloud.columns else set()
    smiles_act = set(df_actives['SMILES'].dropna().astype(str).unique()) if 'SMILES' in df_actives.columns else set()
    smiles_exclude = smiles_mf.union(smiles_act)
    before_zinc = len(df_zinc_raw)
    df_zinc = df_zinc_raw[~df_zinc_raw['SMILES'].astype(str).isin(smiles_exclude)].copy()
    logging.info(f"ZINC overlap removal by SMILES: {before_zinc} -> {len(df_zinc)}")

    # Prepare matrices
    # Report how many rows are lost due to non-numeric entries across selected feature columns
    mf_before = int(df_mf_cloud.shape[0])
    zinc_before = int(df_zinc.shape[0])
    act_before = int(df_actives.shape[0])

    mf_numeric = df_mf_cloud[feature_cols].apply(pd.to_numeric, errors='coerce')
    zinc_numeric = df_zinc[feature_cols].apply(pd.to_numeric, errors='coerce')
    act_numeric = df_actives[feature_cols].apply(pd.to_numeric, errors='coerce')

    X_mf = mf_numeric.dropna().values.astype(np.float32)
    X_zinc = zinc_numeric.dropna().values.astype(np.float32)
    X_act = act_numeric.dropna().values.astype(np.float32)

    logging.info(f"MF cloud rows after dropping any NaNs in selected feature columns: {X_mf.shape[0]} (from {mf_before})")
    logging.info(f"ZINC rows after dropping any NaNs in selected feature columns: {X_zinc.shape[0]} (from {zinc_before})")
    logging.info(f"Actives rows after dropping any NaNs in selected feature columns: {X_act.shape[0]} (from {n_act_pre_na})")
    logging.info(f"Shapes | MF cloud: {X_mf.shape} | ZINC: {X_zinc.shape} | Actives: {X_act.shape}")

    # Training set = MF cloud + ZINC
    X_train = np.vstack([X_mf, X_zinc])

    # PCA
    t0 = time.perf_counter()
    Z_train_pca, Z_act_pca, tag_pca = fit_transform_embeddings(
        X_train, X_act, method='pca', pca_dim=args.pca_dim, umap_dim=args.umap_dim,
        n_neighbors=args.umap_n_neighbors, min_dist=args.umap_min_dist, seed=seed
    )
    Z_mf_pca = Z_train_pca[:X_mf.shape[0]]
    Z_zinc_pca = Z_train_pca[X_mf.shape[0]:]
    scores_act_pca = score_by_nn(Z_act_pca, Z_mf_pca)
    scores_zinc_pca = score_by_nn(Z_zinc_pca, Z_mf_pca)
    df_rank_pca, metrics_pca = build_ranking_and_metrics(scores_act_pca, scores_zinc_pca)
    pca_time = time.perf_counter() - t0
    logging.info(f"PCA completed in {pca_time:.2f}s | Metrics: {json.dumps(metrics_pca, indent=2)}")

    # UMAP
    t1 = time.perf_counter()
    Z_train_umap, Z_act_umap, tag_umap = fit_transform_embeddings(
        X_train, X_act, method='umap', pca_dim=args.pca_dim, umap_dim=args.umap_dim,
        n_neighbors=args.umap_n_neighbors, min_dist=args.umap_min_dist, seed=seed,
        umap_init=args.umap_init
    )
    Z_mf_umap = Z_train_umap[:X_mf.shape[0]]
    Z_zinc_umap = Z_train_umap[X_mf.shape[0]:]
    scores_act_umap = score_by_nn(Z_act_umap, Z_mf_umap)
    scores_zinc_umap = score_by_nn(Z_zinc_umap, Z_mf_umap)
    df_rank_umap, metrics_umap = build_ranking_and_metrics(scores_act_umap, scores_zinc_umap)
    umap_time = time.perf_counter() - t1
    logging.info(f"UMAP completed in {umap_time:.2f}s | Metrics: {json.dumps(metrics_umap, indent=2)}")

    # Save outputs
    results_dir = os.path.join(run_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    df_rank_pca.to_csv(os.path.join(results_dir, f"ranking_{tag_pca}.csv"), index=False)
    df_rank_umap.to_csv(os.path.join(results_dir, f"ranking_{tag_umap}.csv"), index=False)

    summary = {
        'target_uniprot': args.target_uniprot,
        'feature_cols': feature_cols,
        'counts': {
            'mf_cloud': int(X_mf.shape[0]),
            'zinc': int(X_zinc.shape[0]),
            'actives': int(X_act.shape[0]),
        },
        'pca': {
            'tag': tag_pca,
            'time_sec': round(pca_time, 3),
            'metrics': metrics_pca,
        },
        'umap': {
            'tag': tag_umap,
            'time_sec': round(umap_time, 3),
            'metrics': metrics_umap,
        }
    }
    with open(os.path.join(results_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    logging.info("Saved ranking CSVs and summary.json")

    # Console summary
    print("=== Independent PCA vs UMAP Summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
