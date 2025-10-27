#!/usr/bin/env python3
"""
Standalone evaluation of Mordred full 2D vs full 2D+3D descriptor sets against the current 40-feature subset.

Pipeline:
1) Load SMILES subsets from three sources: target ligands, MF cloud, ZINC decoys (parameterized sample sizes)
2) Compute descriptor sets:
   a) Current 40-feature subset (selected from Mordred 2D by column name)
   b) Full Mordred 2D descriptors (numeric-only)
   c) Full Mordred 2D+3D descriptors (ensure 3D coordinates via RDKit ETKDG)
3) Clean (numeric cast, NaN/inf imputation, zero-variance removal) and scale (StandardScaler) each set
4) Build UMAP embeddings (2D) with n_neighbors=1, min_dist=0.1 for each representation
5) Score molecules by distance to the target centroid (in feature space), compute EF@1%
6) Save summary metrics and plots (UMAP scatter colored by class, and distance histograms)

Notes:
- This script is independent of the rest of the codebase and directly reads datasets from the datasets/ directory by default.
- You can pass custom CSVs for any of the three sources.
- 3D generation uses RDKit: AddHs, ETKDGv3 embedding, UFF/MMFF optimization. Failures are skipped; intersection across sets is enforced for fair comparison.

Author: Experimental script for quick hypothesis testing.
"""

from __future__ import annotations
import argparse
import math
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import re

import numpy as np
import pandas as pd
from tqdm import tqdm

# RDKit imports
from rdkit import Chem
from rdkit.Chem import AllChem

# Mordred imports
from mordred import Calculator, descriptors

# ML/DR imports
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import pairwise_distances, roc_auc_score, average_precision_score
import umap

# Viz
import matplotlib.pyplot as plt
import seaborn as sns


CURRENT_FEATURE_NAMES_40 = [
    # Matches LAB_BOOK.md documentation (40 columns)
    "DipoleMoment", "ABC", "nAcid", "nBase", "nAromAtom", "nAtom", "nH", "nC", "nN", "nO", "nS", "nP", "nX",
    "nBonds", "nBondsO", "nBondsS", "nBondsD", "nBondsT", "nBondsA", "nBondsM", "nBondsKS", "nBondsKD",
    "EState_VSA7", "nHBAcc", "nHBDon", "Lipinski", "apol", "bpol", "nRing", "n3Ring", "n4Ring", "n5Ring",
    "n6Ring", "n7Ring", "n8Ring", "nRot", "Diameter", "TopoShapeIndex", "Vabc", "MW",
]

POSSIBLE_SMILES_COLS = [
    "smiles", "SMILES", "canonical_smiles", "canonicalSmiles", "mol_smiles", "can_smiles", "Smiles"
]


def infer_smiles_column(df: pd.DataFrame) -> Optional[str]:
    for c in POSSIBLE_SMILES_COLS:
        if c in df.columns:
            return c
    # heuristic fallback: first object dtype column
    for c in df.columns:
        if df[c].dtype == object:
            # try to detect SMILES-like strings
            sample = df[c].dropna().astype(str).head(50)
            if sample.apply(lambda s: any(ch in s for ch in ["C", "N", "O", "=", "#"]) and len(s) <= 300).mean() > 0.5:
                return c
    return None


def read_smiles_sample_from_csv(csv_path: Path, n: int, seed: int = 42, smiles_col: Optional[str] = None) -> pd.DataFrame:
    """Read up to n smiles from a potentially large CSV, using chunked sampling if necessary."""
    rng = np.random.default_rng(seed)
    rows: List[pd.Series] = []
    total = 0
    # Try reading tiny header to get column names first
    header_df = pd.read_csv(csv_path, nrows=0)
    col = smiles_col or infer_smiles_column(header_df)
    if col is None:
        # try reading a small chunk to infer
        tmp = pd.read_csv(csv_path, nrows=1000)
        col = smiles_col or infer_smiles_column(tmp)
        if col is None:
            raise ValueError(f"Could not infer SMILES column for {csv_path}")

    # Reservoir sampling over chunks to avoid loading entire file
    chunk_iter = pd.read_csv(csv_path, chunksize=50_000, usecols=lambda c: c == col or c.lower().startswith("id"))
    reservoir: List[pd.Series] = []
    k = n
    seen = 0
    for chunk in chunk_iter:
        valid = chunk[chunk[col].notna() & (chunk[col].astype(str).str.len() > 0)]
        for _, row in valid.iterrows():
            seen += 1
            if len(reservoir) < k:
                reservoir.append(row)
            else:
                j = rng.integers(0, seen)
                if j < k:
                    reservoir[j] = row
        total += len(valid)
        if seen >= k * 10 and len(reservoir) >= k:  # Early stop if plenty seen
            break
    if not reservoir:
        return pd.DataFrame(columns=[col])
    out = pd.DataFrame(reservoir)
    return out[[c for c in out.columns if c == col or c.lower().startswith("id")]].rename(columns={col: "smiles"}).drop_duplicates(subset=["smiles"]).head(n)


def load_kw_affinity_files(base_dir: Path) -> List[Path]:
    return sorted([p for p in base_dir.glob("*.csv") if p.is_file()])


def load_target_ligands(default_dir: Path, target_kw_path: Optional[Path], n: int, seed: int) -> pd.DataFrame:
    if target_kw_path is None:
        # pick a deterministic default (first KW file)
        kw_files = load_kw_affinity_files(default_dir)
        if not kw_files:
            raise FileNotFoundError(f"No KW files found in {default_dir}")
        target_kw_path = kw_files[0]
    df = pd.read_csv(target_kw_path)
    col = infer_smiles_column(df)
    if col is None:
        raise ValueError(f"Could not find SMILES column in target ligands file: {target_kw_path}")
    df = df[df[col].notna()].rename(columns={col: "smiles"})
    df = df.drop_duplicates(subset=["smiles"]).sample(n=min(n, len(df)), random_state=seed)
    df = df[["smiles"]].copy()
    df["source"] = "target"
    return df


def load_mf_cloud(default_dir: Path, exclude_kw: Optional[Path], n: int, seed: int) -> pd.DataFrame:
    kw_files = load_kw_affinity_files(default_dir)
    if exclude_kw is not None:
        kw_files = [p for p in kw_files if p.name != exclude_kw.name]
    rng = np.random.default_rng(seed)

    smiles_accum: List[str] = []
    for p in kw_files:
        try:
            df = pd.read_csv(p, usecols=None)
            col = infer_smiles_column(df)
            if col is None: 
                continue
            s = df[col].dropna().astype(str).tolist()
            smiles_accum.extend(s)
            if len(smiles_accum) >= n * 5:  # heuristic to stop early
                break
        except Exception:
            continue
    if not smiles_accum:
        return pd.DataFrame(columns=["smiles", "source"]) 
    smiles_unique = pd.Series(smiles_accum, dtype=str).dropna().drop_duplicates()
    if len(smiles_unique) > n:
        idx = rng.choice(len(smiles_unique), size=n, replace=False)
        smiles_unique = smiles_unique.iloc[idx]
    out = pd.DataFrame({"smiles": smiles_unique})
    out["source"] = "mf_cloud"
    return out


def smiles_to_rdkit_mol(s: str) -> Optional[Chem.Mol]:
    try:
        m = Chem.MolFromSmiles(s)
        if m is None:
            return None
        Chem.SanitizeMol(m)
        return m
    except Exception:
        return None


def embed_3d(mol: Chem.Mol, seed: int = 42, max_attempts: int = 3) -> Optional[Chem.Mol]:
    try:
        m = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        params.numThreads = 0
        for _ in range(max_attempts):
            if AllChem.EmbedMolecule(m, params) == 0:
                try:
                    # optimize geometry
                    try:
                        AllChem.MMFFOptimizeMolecule(m)
                    except Exception:
                        AllChem.UFFOptimizeMolecule(m)
                    return m
                except Exception:
                    continue
        return None
    except Exception:
        return None


def _load_cache(cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        try:
            df = pd.read_csv(cache_path)
            if "smiles" in df.columns:
                df = df.drop_duplicates(subset=["smiles"], keep="last")
            return df
        except Exception:
            pass
    return pd.DataFrame(columns=["smiles"]).astype({"smiles": str})


def _save_cache(cache_path: Path, df: pd.DataFrame) -> None:
    # ensure smiles column is first for readability
    cols = ["smiles"] + [c for c in df.columns if c != "smiles"]
    tmp = df[cols].copy()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp.to_csv(cache_path, index=False)


def compute_mordred(
    df: pd.DataFrame,
    use_3d: bool,
    seed: int = 42,
    cache_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, List[bool], Dict[str, object]]:
    """Compute Mordred descriptors. Returns (descriptor_df, success_mask) aligned with input df rows.
    use_3d=False: 2D descriptors only (ignore_3D=True)
    use_3d=True:  2D+3D descriptors (ignore_3D=False) and requires successful 3D embedding
    """
    # Setup cache
    cache_dir = cache_dir or (Path(__file__).resolve().parent / "cache")
    cache_path = cache_dir / ("mordred_3d_cache.csv.gz" if use_3d else "mordred_2d_cache.csv.gz")
    cache_df = _load_cache(cache_path)

    # Determine which SMILES need calculation
    smiles_list = df["smiles"].astype(str).tolist()
    cached_subset = cache_df[cache_df["smiles"].isin(smiles_list)] if not cache_df.empty else pd.DataFrame(columns=["smiles"]).astype({"smiles": str})
    cached_smiles = set(cached_subset["smiles"].tolist()) if not cached_subset.empty else set()
    missing_smiles = [s for s in smiles_list if s not in cached_smiles]

    new_desc_df = pd.DataFrame()
    if missing_smiles:
        calc = Calculator(descriptors, ignore_3D=not use_3d)
        mols: List[Optional[Chem.Mol]] = []
        success_flags: List[bool] = []
        fail_reasons: Dict[str, List[str]] = {"rdkit_parse": [], "embed_3d": [], "mordred_calc_error": []}
        for s in tqdm(missing_smiles, desc=f"RDKit parse + {'3D' if use_3d else '2D'} prep (uncached)", leave=False):
            m = smiles_to_rdkit_mol(s)
            if m is None:
                mols.append(None)
                success_flags.append(False)
                fail_reasons["rdkit_parse"].append(s)
                continue
            if use_3d:
                m3d = embed_3d(m, seed=seed)
                if m3d is None:
                    mols.append(None)
                    success_flags.append(False)
                    fail_reasons["embed_3d"].append(s)
                    continue
                mols.append(m3d)
                success_flags.append(True)
            else:
                mols.append(m)
                success_flags.append(True)

        valid_idx = [i for i, ok in enumerate(success_flags) if ok]
        valid_mols = [mols[i] for i in valid_idx]
        valid_smiles = [missing_smiles[i] for i in valid_idx]

        if valid_mols:
            try:
                mordred_df = calc.pandas(valid_mols)
            except Exception:
                rows = []
                for m, smi in tqdm(list(zip(valid_mols, valid_smiles)), desc="Mordred per-mol (fallback)", leave=False):
                    try:
                        rows.append(calc(m))
                    except Exception:
                        rows.append({})
                        fail_reasons["mordred_calc_error"].append(smi)
                mordred_df = pd.DataFrame(rows)

            # numeric-only
            for c in mordred_df.columns:
                mordred_df[c] = pd.to_numeric(mordred_df[c], errors="coerce")
            mordred_df = mordred_df.select_dtypes(include=[np.number])
            mordred_df.insert(0, "smiles", valid_smiles)
            # de-duplicate by smiles to avoid one-to-many merge later
            new_desc_df = mordred_df.drop_duplicates(subset=["smiles"], keep="last")

        # Merge with cache (outer union of columns)
        if not new_desc_df.empty:
            updated_cache = (new_desc_df.copy() if cache_df.empty else pd.concat([cache_df, new_desc_df], axis=0, ignore_index=True))
            # enforce unique smiles keep last (new wins)
            updated_cache = updated_cache.drop_duplicates(subset=["smiles"], keep="last")
            _save_cache(cache_path, updated_cache)
            cache_df = updated_cache
        # persist failure reasons for missing SMILES subset (temporary store in stats_ext)
        stats_ext = fail_reasons
    else:
        stats_ext = {"rdkit_parse": [], "embed_3d": [], "mordred_calc_error": []}

    # Build aligned matrix for input SMILES
    if cache_df.empty:
        # nothing computed
        stats = {"failed_smiles": list(df["smiles"].astype(str).tolist()), "feature_nan_counts": {}, "failure_reasons": stats_ext}
        return pd.DataFrame(), [False] * len(df), stats

    # Ensure numeric only and consistent dtypes
    numeric_cols = [c for c in cache_df.columns if c != "smiles"]
    # Ensure cache has unique smiles to prevent one-to-many merge
    cache_df = cache_df.drop_duplicates(subset=["smiles"], keep="last")
    # Selected numeric subset
    X_cache = cache_df[["smiles"] + numeric_cols].copy()

    # Align to input order
    aligned = df[["smiles"]].merge(X_cache, on="smiles", how="left")
    # success mask: row has at least one non-NaN among descriptor columns
    desc_vals = aligned.drop(columns=["smiles"]).values
    success_mask = list(~np.isnan(desc_vals).all(axis=1))
    # stats
    failed_smiles = aligned.loc[~pd.Series(success_mask), "smiles"].astype(str).tolist()
    # per-feature failure counts (NaNs per column across all rows)
    feature_nan_counts = aligned.drop(columns=["smiles"]).isna().sum().to_dict()
    # add reasons for failures (only directly known for parse/embed/calc errors among the uncached set)
    stats = {"failed_smiles": failed_smiles, "feature_nan_counts": feature_nan_counts, "failure_reasons": stats_ext}
    # Return DataFrame without smiles, only descriptors
    result_df = aligned.drop(columns=["smiles"]) 
    return result_df, success_mask, stats


def clean_scale_features(X: pd.DataFrame, train_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[str]]:
    # Drop columns entirely NaN or zero-variance after imputation
    # 1) Impute per-column median
    imp = SimpleImputer(strategy="median")
    if train_mask is not None:
        X_imp = imp.fit_transform(X.iloc[train_mask])
        X_imp_full = imp.transform(X)
    else:
        X_imp_full = imp.fit_transform(X)
        X_imp = X_imp_full
    # 2) Remove zero-variance columns
    col_mask = (np.nan_to_num(np.std(X_imp, axis=0)) > 1e-12)
    X_imp_full = X_imp_full[:, col_mask]
    kept_cols = [c for c, k in zip(list(X.columns), col_mask) if k]
    # 3) Scale
    scaler = StandardScaler()
    if train_mask is not None:
        X_scaled_train = scaler.fit_transform(X_imp)
        X_scaled_full = scaler.transform(X_imp_full)
    else:
        X_scaled_full = scaler.fit_transform(X_imp_full)
    return X_scaled_full, kept_cols


def _drop_rare_element_families(df: pd.DataFrame) -> pd.DataFrame:
    # Drop E-state MAX/MIN descriptors for rare elements seldom present in drug-like chemistry
    rare_elems = ["Li", "Be", "Si", "Ge", "As", "Se", "Sn", "Pb"]
    patterns = []
    for el in rare_elems:
        # matches like MAXsLi, MAXssSiH2, MINssssSn, etc.
        patterns.append(re.compile(rf"^(MAX|MIN).*(?:{el})"))
    keep_cols = []
    for c in df.columns:
        if c == "smiles":
            keep_cols.append(c)
            continue
        if any(p.search(c) for p in patterns):
            continue
        keep_cols.append(c)
    return df[keep_cols]


def compute_feature_prevalence_by_set(X: pd.DataFrame, sources: np.ndarray) -> pd.DataFrame:
    # X does not include smiles; sources aligns with rows in X
    mask_T = (sources == "target")
    mask_MF = (sources == "mf_cloud")
    mask_Z = (sources == "zinc")
    prev_T = (~X[mask_T].isna()).mean(axis=0) if mask_T.any() else pd.Series(0.0, index=X.columns)
    prev_MF = (~X[mask_MF].isna()).mean(axis=0) if mask_MF.any() else pd.Series(0.0, index=X.columns)
    prev_Z = (~X[mask_Z].isna()).mean(axis=0) if mask_Z.any() else pd.Series(0.0, index=X.columns)
    cov = pd.DataFrame({"descriptor": X.columns, "prev_T": prev_T.values, "prev_MF": prev_MF.values, "prev_Z": prev_Z.values})
    return cov


def apply_feature_selection(X: pd.DataFrame, sources: np.ndarray, pf_target: float, pf_mf: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # returns filtered X and coverage DataFrame with 'selected' column
    cov = compute_feature_prevalence_by_set(X, sources)
    sel = (cov["prev_T"] >= pf_target) | (cov["prev_MF"] >= pf_mf)
    cov["selected"] = sel.values
    keep_cols = cov.loc[cov["selected"], "descriptor"].tolist()
    if not keep_cols:
        # fallback: keep top-N by prev_T then prev_MF
        tmp = cov.sort_values(["prev_T", "prev_MF"], ascending=False).head(min(50, len(cov)))
        keep_cols = tmp["descriptor"].tolist()
        cov.loc[:, "selected"] = cov["descriptor"].isin(keep_cols)
    return X[keep_cols], cov


def compute_row_completeness(X: pd.DataFrame) -> np.ndarray:
    # fraction of non-NaN per row
    if X.shape[1] == 0:
        return np.zeros(X.shape[0], dtype=float)
    return (~X.isna()).mean(axis=1).values


def apply_row_selection_by_set(X: pd.DataFrame, sources: np.ndarray, pr_target_guard: float, pr_mf: float, pr_z: float) -> np.ndarray:
    comp = compute_row_completeness(X)
    keep = np.ones(X.shape[0], dtype=bool)
    # Targets: only drop if below guard
    keep[(sources == "target") & (comp < pr_target_guard)] = False
    # MF: moderate
    keep[(sources == "mf_cloud") & (comp < pr_mf)] = False
    # ZINC: harsh
    keep[(sources == "zinc") & (comp < pr_z)] = False
    return keep


def sweep_coverage_thresholds(X: pd.DataFrame, sources: np.ndarray, pf_T_grid: List[float], pf_MF_grid: List[float], pr_T_guard: float, pr_MF_grid: List[float], pr_Z_grid: List[float], weights: Tuple[float, float, float, float]) -> pd.DataFrame:
    rows = []
    total_cols = X.shape[1]
    nT = (sources == "target").sum(); nMF = (sources == "mf_cloud").sum(); nZ = (sources == "zinc").sum()
    for pfT in pf_T_grid:
        for pfMF in pf_MF_grid:
            Xf, _ = apply_feature_selection(X, sources, pfT, pfMF)
            for prMF in pr_MF_grid:
                for prZ in pr_Z_grid:
                    keep_mask = apply_row_selection_by_set(Xf, sources, pr_T_guard, prMF, prZ)
                    frac_T = ((sources == "target") & keep_mask).sum() / max(1, nT)
                    frac_MF = ((sources == "mf_cloud") & keep_mask).sum() / max(1, nMF)
                    frac_Z = ((sources == "zinc") & keep_mask).sum() / max(1, nZ)
                    cols_frac = Xf.shape[1] / max(1, total_cols)
                    wT, wMF, wZ, wC = weights
                    coverage_score = wT*frac_T + wMF*frac_MF + wZ*frac_Z + wC*cols_frac
                    rows.append({
                        "pf_target": pfT, "pf_mf": pfMF, "pr_target_guard": pr_T_guard, "pr_mf": prMF, "pr_zinc": prZ,
                        "rows_frac_target": frac_T, "rows_frac_mf": frac_MF, "rows_frac_zinc": frac_Z,
                        "cols_frac": cols_frac, "cols_kept": Xf.shape[1], "coverage_score": coverage_score
                    })
    return pd.DataFrame(rows)


def compute_umap_2d(X: np.ndarray, seed: int = 42, n_neighbors: int = 2, min_dist: float = 0.1) -> np.ndarray:
    # UMAP requires n_neighbors > 1; if 1 is requested, bump to 2 and note.
    eff_nn = max(2, int(n_neighbors))
    if eff_nn != n_neighbors:
        print(f"[note] UMAP requires n_neighbors>1; requested {n_neighbors} → using {eff_nn}.")
    # No seed for UMAP as per request (random_state=None)
    reducer = umap.UMAP(n_neighbors=eff_nn, min_dist=min_dist, n_components=2, metric="euclidean", random_state=None)
    return reducer.fit_transform(X)


def procrustes_align(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Align target embedding to source via classical Procrustes analysis.
    Uses optimal orthogonal rotation and optimal scaling s = trace(S) / ||Yc||^2,
    where S are singular values of Yc^T Xc.
    Returns target_aligned in the coordinate frame of source.
    """
    if source.shape != target.shape:
        raise ValueError("Embeddings must have same shape for alignment")
    # center
    Xc = source - source.mean(axis=0, keepdims=True)
    Yc = target - target.mean(axis=0, keepdims=True)
    denom = float((Yc ** 2).sum())
    if denom == 0.0:
        return target.copy()
    # orthogonal Procrustes
    U, S, Vt = np.linalg.svd(Yc.T @ Xc, full_matrices=False)
    R = U @ Vt
    s = float(S.sum()) / denom
    aligned = s * (Yc @ R) + source.mean(axis=0, keepdims=True)
    return aligned


def ef_at_percent(scores: np.ndarray, labels_active: np.ndarray, percent: float = 1.0) -> float:
    """Compute EF@percent given scores (higher is better) and boolean active labels.
    EF@p% = (TPx/Nx) / (A/N) where Nx=ceil(p% of N).
    """
    N = len(scores)
    if N == 0:
        return float("nan")
    A = int(labels_active.sum())
    if A == 0:
        return float("nan")
    Nx = max(1, math.ceil((percent / 100.0) * N))
    order = np.argsort(-scores)  # descending
    top_idx = order[:Nx]
    TPx = int(labels_active[top_idx].sum())
    return (TPx / Nx) / (A / N)


def centroid_similarity_scores(X: np.ndarray, labels_active: np.ndarray) -> np.ndarray:
    """Score each point by negative Euclidean distance to the active-class centroid."""
    if labels_active.sum() == 0:
        return np.zeros(X.shape[0], dtype=float)
    centroid = X[labels_active].mean(axis=0, keepdims=True)
    dists = np.linalg.norm(X - centroid, axis=1)
    return -dists


def run_eval(args: argparse.Namespace) -> None:
    base_dir = Path(args.base_dir).resolve()
    datasets_dir = base_dir / "datasets"

    # 1) Load inputs (three sources)
    # Target ligands
    target_path = Path(args.target_ligands_csv).resolve() if args.target_ligands_csv else None
    target_df = load_target_ligands(
        default_dir=datasets_dir / "molecular_function_affinity_data",
        target_kw_path=target_path,
        n=args.n_target,
        seed=args.seed,
    )
    # MF cloud
    mf_cloud_df = load_mf_cloud(
        default_dir=datasets_dir / "molecular_function_affinity_data",
        exclude_kw=target_path,
        n=args.n_mf,
        seed=args.seed,
    )
    # ZINC decoys
    zinc_path = Path(args.zinc_csv).resolve() if args.zinc_csv else (datasets_dir / "zinc_data.csv")
    zinc_df = read_smiles_sample_from_csv(zinc_path, n=args.n_zinc, seed=args.seed)
    if not zinc_df.empty:
        zinc_df["source"] = "zinc"
    zinc_df = zinc_df[["smiles", "source"]]

    # Concatenate and deduplicate across sources with priority: target > mf_cloud > zinc
    all_df = pd.concat([target_df, mf_cloud_df, zinc_df], ignore_index=True)
    all_df = all_df.dropna(subset=["smiles"]).copy()
    all_df["smiles"] = all_df["smiles"].astype(str)
    priority = {"target": 0, "mf_cloud": 1, "zinc": 2}
    all_df["_prio"] = all_df["source"].map(priority).fillna(3).astype(int)
    all_df = all_df.sort_values(["smiles", "_prio"]).drop_duplicates(subset=["smiles"], keep="first").drop(columns=["_prio"]).reset_index(drop=True)
    data_df = all_df

    # Compute labels
    labels = data_df["source"].values
    is_active = (labels == "target")

    # 2) Compute descriptor sets
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2D only
    mordred2d_df, mask2d, stats2d = compute_mordred(data_df, use_3d=False, seed=args.seed, cache_dir=Path(args.cache_dir) if args.cache_dir else None)
    # 2D + 3D
    mordred3d_df, mask3d, stats3d = compute_mordred(data_df, use_3d=True, seed=args.seed, cache_dir=Path(args.cache_dir) if args.cache_dir else None)

    # Define valid intersection for fair comparison (must have both 2D and 3D descriptors)
    valid_mask = (
        np.array(mask2d, dtype=bool)
        & np.array(mask3d, dtype=bool)
    )
    valid_idx = np.where(valid_mask)[0]

    if len(valid_idx) < 10:
        print("Too few molecules with successful 3D descriptors. Consider reducing n_* or checking SMILES quality.")

    # Slice data and labels to intersection
    use_df = data_df.iloc[valid_idx].reset_index(drop=True)
    use_labels = labels[valid_idx]
    use_is_active = is_active[valid_idx]

    mordred2d_use = mordred2d_df.iloc[valid_idx].reset_index(drop=True)
    mordred3d_use = mordred3d_df.iloc[valid_idx].reset_index(drop=True)

    # Optional chemistry-aware pruning of rare-element E-state descriptor families
    mordred2d_use = _drop_rare_element_families(mordred2d_use)
    mordred3d_use = _drop_rare_element_families(mordred3d_use)

    # a) current 40-feature subset (select from Mordred 2D by name)
    current_cols = [c for c in CURRENT_FEATURE_NAMES_40 if c in mordred2d_use.columns]
    X_curr = mordred2d_use[current_cols].copy()
    # b) full 2D
    X_2d = mordred2d_use.copy()
    # c) full 2D+3D (union columns)
    X_2d3d = mordred3d_use.copy()

    # 2b) Feature selection based on prevalence priorities (protect targets, moderate MF)
    sources = use_labels
    X_2d_sel, cov2d = apply_feature_selection(X_2d, sources, pf_target=args.pf_target, pf_mf=args.pf_mf)
    X_2d3d_sel, cov2d3d = apply_feature_selection(X_2d3d, sources, pf_target=args.pf_target, pf_mf=args.pf_mf)
    cov2d.to_csv(out_dir / "feature_coverage_2d.csv", index=False)
    cov2d3d.to_csv(out_dir / "feature_coverage_2d3d.csv", index=False)

    # 2c) Row selection by set-specific completeness thresholds
    keep2d = apply_row_selection_by_set(X_2d_sel, sources, pr_target_guard=args.pr_target_guard, pr_mf=args.pr_mf, pr_z=args.pr_zinc)
    keep2d3d = apply_row_selection_by_set(X_2d3d_sel, sources, pr_target_guard=args.pr_target_guard, pr_mf=args.pr_mf, pr_z=args.pr_zinc)
    keep_final = keep2d & keep2d3d

    # Emit row completeness report
    comp2d = compute_row_completeness(X_2d_sel)
    comp2d3d = compute_row_completeness(X_2d3d_sel)
    row_rep = pd.DataFrame({
        "smiles": use_df["smiles"].values,
        "source": sources,
        "comp_2d": comp2d,
        "comp_2d3d": comp2d3d,
        "keep_2d": keep2d,
        "keep_2d3d": keep2d3d,
        "keep_final": keep_final,
    })
    row_rep.to_csv(out_dir / "row_completeness.csv", index=False)

    # Apply final mask to ensure fair comparison across representations
    use_df = use_df.iloc[keep_final].reset_index(drop=True)
    use_labels = use_labels[keep_final]
    use_is_active = use_is_active[keep_final]
    X_curr = X_curr.iloc[keep_final].reset_index(drop=True)
    X_2d_sel = X_2d_sel.iloc[keep_final].reset_index(drop=True)
    X_2d3d_sel = X_2d3d_sel.iloc[keep_final].reset_index(drop=True)

    # Cleaning + scaling
    X_curr_sc, kept_curr = clean_scale_features(X_curr)
    X_2d_sc, kept_2d = clean_scale_features(X_2d_sel)
    X_2d3d_sc, kept_2d3d = clean_scale_features(X_2d3d_sel)

    # Diagnostics: quantify 3D-only contribution after selection/cleaning
    cols_2d_all = set(mordred2d_use.columns.tolist())
    cols_2d3d_all = set(mordred3d_use.columns.tolist())
    cols_3d_only_all = cols_2d3d_all - cols_2d_all
    # Selected sets (pre-clean) and kept sets (post-clean)
    sel_2d_cols = set(X_2d_sel.columns.tolist())
    sel_2d3d_cols = set(X_2d3d_sel.columns.tolist())
    kept_2d_set = set(kept_2d)
    kept_2d3d_set = set(kept_2d3d)
    n_3d_only_selected = len([c for c in sel_2d3d_cols if c in cols_3d_only_all])
    n_3d_only_kept = len([c for c in kept_2d3d_set if c in cols_3d_only_all])

    # 3) UMAP embeddings
    emb_curr = compute_umap_2d(X_curr_sc, seed=args.seed, n_neighbors=args.umap_n_neighbors, min_dist=args.umap_min_dist)
    emb_2d = compute_umap_2d(X_2d_sc, seed=args.seed, n_neighbors=args.umap_n_neighbors, min_dist=args.umap_min_dist)
    emb_2d3d = compute_umap_2d(X_2d3d_sc, seed=args.seed, n_neighbors=args.umap_n_neighbors, min_dist=args.umap_min_dist)

    # 4) Scoring and EF@1% (use feature-space centroid similarity, not UMAP)
    scores_curr = centroid_similarity_scores(X_curr_sc, use_is_active)
    scores_2d = centroid_similarity_scores(X_2d_sc, use_is_active)
    scores_2d3d = centroid_similarity_scores(X_2d3d_sc, use_is_active)

    ef1_curr = ef_at_percent(scores_curr, use_is_active, percent=1.0)
    ef1_2d = ef_at_percent(scores_2d, use_is_active, percent=1.0)
    ef1_2d3d = ef_at_percent(scores_2d3d, use_is_active, percent=1.0)

    # ROC-AUC and PR-AUC (Average Precision)
    def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
        try:
            roc = roc_auc_score(y_true, y_score)
        except Exception:
            roc = float('nan')
        try:
            pr = average_precision_score(y_true, y_score)
        except Exception:
            pr = float('nan')
        return float(roc), float(pr)

    roc_curr, pr_curr = safe_auc(use_is_active.astype(int), scores_curr)
    roc_2d, pr_2d = safe_auc(use_is_active.astype(int), scores_2d)
    roc_2d3d, pr_2d3d = safe_auc(use_is_active.astype(int), scores_2d3d)

    # Additional diagnostics: ranking similarity and top-1% overlap
    N_eval = int(len(use_df))
    Nx = max(1, math.ceil(0.01 * N_eval))
    order_2d = np.argsort(-scores_2d)
    order_2d3d = np.argsort(-scores_2d3d)
    top2d = set(order_2d[:Nx].tolist())
    top2d3d = set(order_2d3d[:Nx].tolist())
    top1_overlap_frac = (len(top2d & top2d3d) / float(Nx)) if Nx > 0 else float("nan")
    # Spearman via rank correlation of scores (avoid extra deps)
    r2d = pd.Series(scores_2d).rank(method="average")
    r2d3d = pd.Series(scores_2d3d).rank(method="average")
    with np.errstate(invalid='ignore'):
        spearman_r = float(np.corrcoef(r2d, r2d3d)[0, 1]) if N_eval > 1 else float('nan')

    summary = {
        "n_total": int(len(use_df)),
        "n_target": int((use_labels == "target").sum()),
        "n_mf_cloud": int((use_labels == "mf_cloud").sum()),
        "n_zinc": int((use_labels == "zinc").sum()),
        "ef1_current_40": float(ef1_curr),
        "ef1_full_2d": float(ef1_2d),
        "ef1_full_2d3d": float(ef1_2d3d),
    "roc_auc_current_40": float(roc_curr),
    "roc_auc_full_2d": float(roc_2d),
    "roc_auc_full_2d3d": float(roc_2d3d),
    "pr_auc_current_40": float(pr_curr),
    "pr_auc_full_2d": float(pr_2d),
    "pr_auc_full_2d3d": float(pr_2d3d),
        "curr_dim": int(X_curr_sc.shape[1]),
        "full2d_dim": int(X_2d_sc.shape[1]),
        "full2d3d_dim": int(X_2d3d_sc.shape[1]),
        "thresholds": {
            "pf_target": args.pf_target,
            "pf_mf": args.pf_mf,
            "pr_target_guard": args.pr_target_guard,
            "pr_mf": args.pr_mf,
            "pr_zinc": args.pr_zinc,
        },
        "kept": {
            "rows_kept": int(len(use_df)),
            "cols_kept_2d": int(X_2d_sc.shape[1]),
            "cols_kept_2d3d": int(X_2d3d_sc.shape[1]),
            "cols_kept_3d_only": int(n_3d_only_kept),
            "cols_selected_3d_only": int(n_3d_only_selected),
        },
        "diagnostics": {
            "top1_overlap_frac": float(top1_overlap_frac),
            "spearman_r_scores": float(spearman_r),
        },
    }
    pd.Series(summary).to_json(out_dir / "summary.json", indent=2)
    # Persist kept column lists for transparency
    with open(out_dir / "kept_columns_2d.txt", "w") as f:
        f.write("\n".join(sorted(list(kept_2d_set))))
    with open(out_dir / "kept_columns_2d3d.txt", "w") as f:
        f.write("\n".join(sorted(list(kept_2d3d_set))))
    with open(out_dir / "kept_columns_2d3d_3donly.txt", "w") as f:
        f.write("\n".join(sorted([c for c in kept_2d3d_set if c in cols_3d_only_all])))
    if (summary["full2d_dim"] != summary["full2d3d_dim"]) and (summary["diagnostics"]["top1_overlap_frac"] == 1.0):
        print("[note] Full 2D and 2D+3D produced identical top-1% ranking despite different dims; 3D-only kept columns:", n_3d_only_kept)
    # Save failure tracking
    import json
    with open(out_dir / "descriptor_failures_2d.json", "w") as f:
        json.dump({
            "failed_smiles": stats2d["failed_smiles"],
            "feature_nan_counts": stats2d["feature_nan_counts"],
            "n_failed": len(stats2d["failed_smiles"]),
            "failure_reasons": stats2d.get("failure_reasons", {}),
        }, f, indent=2)
    with open(out_dir / "descriptor_failures_3d.json", "w") as f:
        json.dump({
            "failed_smiles": stats3d["failed_smiles"],
            "feature_nan_counts": stats3d["feature_nan_counts"],
            "n_failed": len(stats3d["failed_smiles"]),
            "failure_reasons": stats3d.get("failure_reasons", {}),
        }, f, indent=2)
    # Write CSVs of failure reasons (per-SMILES)
    def write_failure_reasons_csv(stats: Dict[str, object], path: Path):
        reasons = stats.get("failure_reasons", {}) or {}
        rows = []
        for reason, smiles_list in reasons.items():
            for s in smiles_list:
                rows.append({"smiles": s, "reason": reason})
        pd.DataFrame(rows, columns=["smiles", "reason"]).to_csv(path, index=False)

    write_failure_reasons_csv(stats2d, out_dir / "failure_reasons_2d.csv")
    write_failure_reasons_csv(stats3d, out_dir / "failure_reasons_3d.csv")

    # Write combined descriptor NaN counts CSV across 2D and 3D
    nan2d = pd.Series(stats2d.get("feature_nan_counts", {}), name="nan_count_2d")
    nan3d = pd.Series(stats3d.get("feature_nan_counts", {}), name="nan_count_3d")
    nan_df = pd.concat([nan2d, nan3d], axis=1).fillna(0).astype(int)
    nan_df.index.name = "descriptor"
    nan_df["total_rows_2d"] = int(len(mordred2d_df))
    nan_df["total_rows_3d"] = int(len(mordred3d_df))
    nan_df.reset_index().to_csv(out_dir / "descriptor_nan_counts.csv", index=False)
    print("EF@1%:", summary)

    # 5) Visualizations
    def plot_umap(emb: np.ndarray, labels: np.ndarray, title: str, ax: plt.Axes, xlim: Tuple[float, float], ylim: Tuple[float, float]):
        sns.scatterplot(ax=ax, x=emb[:, 0], y=emb[:, 1], hue=labels, s=10, alpha=0.7, palette={"target": "red", "mf_cloud": "blue", "zinc": "gray"}, legend=False)
        ax.set_title(title)
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    # Compute shared x/y limits across all embeddings with small margins
    all_emb = np.vstack([emb_curr, emb_2d, emb_2d3d])
    x_min, x_max = all_emb[:, 0].min(), all_emb[:, 0].max()
    y_min, y_max = all_emb[:, 1].min(), all_emb[:, 1].max()
    dx = (x_max - x_min) * 0.05 if x_max > x_min else 1.0
    dy = (y_max - y_min) * 0.05 if y_max > y_min else 1.0
    xlim = (x_min - dx, x_max + dx)
    ylim = (y_min - dy, y_max + dy)

    # Combined UMAP figure with shared axes
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    plot_umap(emb_curr, use_labels, f"UMAP (Current 40)", axes[0], xlim, ylim)
    plot_umap(emb_2d, use_labels, f"UMAP (Full 2D)", axes[1], xlim, ylim)
    plot_umap(emb_2d3d, use_labels, f"UMAP (Full 2D+3D)", axes[2], xlim, ylim)
    # One legend outside
    handles, labels_legend = axes[2].get_legend_handles_labels()
    fig.legend(handles, ["target", "mf_cloud", "zinc"], loc="upper right", frameon=False)
    plt.tight_layout()
    plt.savefig(out_dir / "umap_all.png", dpi=200)
    plt.close(fig)

    # Distance histograms to active centroid
    # Combined distance plots with shared axes across panels
    dists_curr = -scores_curr
    dists_2d = -scores_2d
    dists_2d3d = -scores_2d3d
    # Compute shared x-limits across all dists
    x_min_d = float(np.nanmin(np.concatenate([dists_curr, dists_2d, dists_2d3d])))
    x_max_d = float(np.nanmax(np.concatenate([dists_curr, dists_2d, dists_2d3d])))
    dx_d = (x_max_d - x_min_d) * 0.05 if x_max_d > x_min_d else 1.0
    xlim_d = (x_min_d - dx_d, x_max_d + dx_d)

    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)
    def panel(ax: plt.Axes, dists: np.ndarray, title: str):
        for grp, color in [("target", "red"), ("mf_cloud", "blue"), ("zinc", "gray")]:
            mask = (use_labels == grp)
            if mask.sum() == 0:
                continue
            sns.kdeplot(dists[mask], label=grp, color=color, fill=False, common_norm=False, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Distance to active centroid (lower is better)")
        ax.set_ylabel("Density")
        ax.set_xlim(*xlim_d)

    panel(axes2[0], dists_curr, "Current 40")
    panel(axes2[1], dists_2d, "Full 2D")
    panel(axes2[2], dists_2d3d, "Full 2D+3D")
    # One legend outside
    handles2, labels2 = axes2[2].get_legend_handles_labels()
    fig2.legend(handles2, ["target", "mf_cloud", "zinc"], loc="upper right", frameon=False)
    plt.tight_layout()
    plt.savefig(out_dir / "dist_hist_all.png", dpi=200)
    plt.close(fig2)

    # Movement plots: align and draw vectors between embeddings
    def movement_plot(Z_from: np.ndarray, Z_to: np.ndarray, labels_arr: np.ndarray, title: str, path: Path):
        # Align target embedding to source
        Z_to_aligned = procrustes_align(Z_from, Z_to)
        # Limits across both
        all_pts = np.vstack([Z_from, Z_to_aligned])
        x_min, x_max = all_pts[:, 0].min(), all_pts[:, 0].max()
        y_min, y_max = all_pts[:, 1].min(), all_pts[:, 1].max()
        dx = (x_max - x_min) * 0.05 if x_max > x_min else 1.0
        dy = (y_max - y_min) * 0.05 if y_max > y_min else 1.0
        xlim2 = (x_min - dx, x_max + dx)
        ylim2 = (y_min - dy, y_max + dy)

        fig, ax = plt.subplots(1, 1, figsize=(6.5, 6))
        # Plot start points
        palette = {"target": "red", "mf_cloud": "blue", "zinc": "gray"}
        for grp in ["target", "mf_cloud", "zinc"]:
            m = (labels_arr == grp)
            if m.sum() == 0:
                continue
            ax.scatter(Z_from[m, 0], Z_from[m, 1], s=8, alpha=0.6, color=palette[grp], label=f"{grp} (start)")
        # Draw movement vectors (thin, semi-transparent)
        for i in range(Z_from.shape[0]):
            ax.plot([Z_from[i, 0], Z_to_aligned[i, 0]], [Z_from[i, 1], Z_to_aligned[i, 1]], color="black", alpha=0.05, linewidth=0.5)
        ax.set_title(title)
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.set_xlim(*xlim2)
        ax.set_ylim(*ylim2)
        ax.legend(frameon=False, loc="best")
        plt.tight_layout()
        plt.savefig(path, dpi=200)
        plt.close(fig)

    movement_plot(emb_curr, emb_2d, use_labels, "Movement: Current40 → Full 2D (aligned)", out_dir / "move_curr_to_full2d.png")
    movement_plot(emb_2d, emb_2d3d, use_labels, "Movement: Full 2D → Full 2D+3D (aligned)", out_dir / "move_full2d_to_full2d3d.png")

    # Save a CSV of the working set and scores
    out_df = use_df.copy()
    out_df["score_current40"] = scores_curr
    out_df["score_full2d"] = scores_2d
    out_df["score_full2d3d"] = scores_2d3d
    out_df.to_csv(out_dir / "scored_molecules.csv", index=False)

    # Optional: parameter sweep for coverage Pareto
    if args.enable_sweep:
        def parse_grid(s: str, default: List[float]) -> List[float]:
            if not s:
                return default
            try:
                vals = [float(x.strip()) for x in s.split(",") if x.strip()]
                return vals if vals else default
            except Exception:
                return default

        pfT_grid = parse_grid(args.pf_target_grid, [0.9, 0.95, 0.98])
        pfMF_grid = parse_grid(args.pf_mf_grid, [0.5, 0.7, 0.9])
        prMF_grid = parse_grid(args.pr_mf_grid, [0.5, 0.7, 0.9])
        prZ_grid = parse_grid(args.pr_zinc_grid, [0.6, 0.8, 0.95])
        w = (args.w_target, args.w_mf, args.w_zinc, args.w_cols)
        # Sweep on 2D representation (more descriptors, more informative for coverage)
        cov_grid = sweep_coverage_thresholds(X_2d, sources, pfT_grid, pfMF_grid, args.pr_target_guard, prMF_grid, prZ_grid, w)
        cov_grid.to_csv(out_dir / "coverage_grid.csv", index=False)
        # Simple Pareto-like scatter: cols_frac vs rows_frac_target colored by rows_frac_mf, size by coverage_score
        plt.figure(figsize=(7, 5))
        sc = plt.scatter(cov_grid["cols_frac"], cov_grid["rows_frac_target"], c=cov_grid["rows_frac_mf"], s=50 + 150*cov_grid["coverage_score"], cmap="viridis", alpha=0.8)
        plt.xlabel("Fraction of columns kept (2D)")
        plt.ylabel("Fraction of targets kept")
        cbar = plt.colorbar(sc)
        cbar.set_label("Fraction of MF kept")
        plt.title("Coverage sweep (2D): Pareto overview")
        plt.tight_layout()
        plt.savefig(out_dir / "coverage_pareto.png", dpi=200)
        plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare current feature set vs Mordred full 2D and 2D+3D features.")
    p.add_argument("--base_dir", default=str(Path(__file__).resolve().parents[2]), help="Repo base directory (for default dataset paths)")
    p.add_argument("--target_ligands_csv", default=None, help="Path to target ligands CSV (SMILES column inferred). Defaults to first KW file under datasets/molecular_function_affinity_data.")
    p.add_argument("--zinc_csv", default=None, help="Path to ZINC SMILES CSV (default datasets/zinc_data.csv).")
    p.add_argument("--n_target", type=int, default=300, help="Number of target ligands to sample")
    p.add_argument("--n_mf", type=int, default=600, help="Number of MF cloud molecules to sample")
    p.add_argument("--n_zinc", type=int, default=600, help="Number of ZINC decoys to sample")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--umap_n_neighbors", type=int, default=10, help="UMAP n_neighbors (will be clamped to >=2 by UMAP)")
    p.add_argument("--umap_min_dist", type=float, default=0.1, help="UMAP min_dist")
    p.add_argument("--cache_dir", default="tests/mordred_full_feature_eval/cache", help="Directory to cache Mordred descriptors by SMILES")
    p.add_argument("--output_dir", default="tests/mordred_full_feature_eval/output", help="Output directory for results")
    # Coverage-aware selection parameters
    p.add_argument("--pf_target", type=float, default=0.95, help="Per-feature prevalence threshold in target set (retain if >=)")
    p.add_argument("--pf_mf", type=float, default=0.7, help="Per-feature prevalence threshold in MF set (retain if >=)")
    p.add_argument("--pr_target_guard", type=float, default=0.2, help="Per-row completeness guard for targets (drop if <)")
    p.add_argument("--pr_mf", type=float, default=0.6, help="Per-row completeness threshold for MF (drop if <)")
    p.add_argument("--pr_zinc", type=float, default=0.8, help="Per-row completeness threshold for ZINC (drop if <)")
    # Sweep controls
    p.add_argument("--enable_sweep", action="store_true", help="Run coverage threshold sweep and emit coverage_grid.csv and coverage_pareto.png")
    p.add_argument("--pf_target_grid", type=str, default="0.9,0.95,0.98")
    p.add_argument("--pf_mf_grid", type=str, default="0.5,0.7,0.9")
    p.add_argument("--pr_mf_grid", type=str, default="0.5,0.7,0.9")
    p.add_argument("--pr_zinc_grid", type=str, default="0.6,0.8,0.95")
    p.add_argument("--w_target", type=float, default=0.5, help="Weight for target row retention in sweep objective")
    p.add_argument("--w_mf", type=float, default=0.2, help="Weight for MF row retention in sweep objective")
    p.add_argument("--w_zinc", type=float, default=0.1, help="Weight for ZINC row retention in sweep objective")
    p.add_argument("--w_cols", type=float, default=0.2, help="Weight for column retention in sweep objective")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_eval(args)
