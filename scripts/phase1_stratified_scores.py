#!/usr/bin/env python3
"""
Phase 1 Potency-Stratified Scores (v4 molfuse)

- For each completed Phase 1 run, compute EF@1/5/10 for potency tiers (High, Medium, Weak)
- Uses ranked_scores.csv for rankings and dataset CSVs for affinity values
- Works even if only some runs have finished

Tiers (nM):
- High:   0.1 <= nM <= 100
- Medium: 100 <  nM <= 1000
- Weak:   1000 < nM <= 100000

Usage:
  python scripts/phase1_stratified_scores.py \
      --workspace_dir experiment_workspace_v4 \
      --phase phase1 \
      --output_dir reporting/phase1_stratified

Notes:
- Reads actives from actives_features_csv if provided in logs/phase1_summary.json.
  Otherwise derives actives by filtering mf_features_csv by accession parsed from target.
- Joins by Compound ChEMBL ID when available, else by SMILES/canonical_smiles.
- Self-contained; does not import archived modules.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

import numpy as np
import pandas as pd

# Plotting (headless safe)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import seaborn as sns  # type: ignore
    _HAVE_SNS = True
except Exception:
    sns = None  # type: ignore
    _HAVE_SNS = False


def _setup_logger(out_dir: Path) -> logging.Logger:
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


def _read_json(p: Path) -> Optional[dict]:
    try:
        with p.open("r") as f:
            return json.load(f)
    except Exception:
        return None


def _extract_accession(target: str) -> Optional[str]:
    m = re.match(r".*_([A-Z0-9]+)$", target)
    return m.group(1) if m else None


def _read_actives_df(summary_cfg: dict, target: str, logger: logging.Logger) -> Optional[pd.DataFrame]:
    # Prefer explicit actives_features_csv
    actives_path = summary_cfg.get("actives_features_csv")
    if actives_path:
        p = Path(actives_path)
        if p.exists():
            try:
                desired = {"Compound ChEMBL ID", "canonical_smiles", "SMILES", "Standard Value (nM)", "accession"}
                df = pd.read_csv(
                    p,
                    usecols=lambda c: c in desired,
                    low_memory=False,
                )
                logger.info(f"Loaded actives CSV: {p} (rows={len(df)})")
                return df
            except Exception as e:
                logger.info(f"Failed reading actives CSV {p}: {e}")
    # Derive from mf_features_csv by accession
    mf_path = summary_cfg.get("mf_features_csv")
    if mf_path:
        acc = _extract_accession(target)
        if not acc:
            return None
        p = Path(mf_path)
        if p.exists():
            try:
                desired = {"Compound ChEMBL ID", "canonical_smiles", "SMILES", "Standard Value (nM)", "accession"}
                df_all = pd.read_csv(
                    p,
                    usecols=lambda c: c in desired,
                    low_memory=False,
                )
                if "accession" in df_all.columns:
                    df = df_all[df_all["accession"] == acc].copy()
                    logger.info(f"Derived actives from MF by accession={acc}: rows={len(df)}")
                    return df
            except Exception as e:
                logger.info(f"Failed reading/deriving actives from MF {p}: {e}")
    return None


def _assign_potency_tier(nm: float) -> Optional[str]:
    try:
        v = float(nm)
    except Exception:
        return None
    if v < 0.1 or v > 100000:
        return None
    if v <= 100.0:
        return "High"
    if v <= 1000.0:
        return "Medium"
    return "Weak"


def _normalize_colnames(df: pd.DataFrame) -> Dict[str, str]:
    """Return a mapping of normalized lower-case names to original names."""
    return {c.lower().strip(): c for c in df.columns}


def _pick_first_present(mapping: Dict[str, str], candidates: List[str]) -> Optional[str]:
    for k in candidates:
        if k in mapping:
            return mapping[k]
    return None


def _join_affinity_to_actives(ranked: pd.DataFrame, actives_df: pd.DataFrame) -> pd.DataFrame:
    """Attach potency tiers to the actives rows in `ranked` by robustly joining to `actives_df`.

    Join strategy: prefer a ChEMBL compound ID; fallback to canonical SMILES (case-insensitive); last resort leaves tier NaN.
    """
    ra = ranked[ranked["source"] == "actives"].copy()
    if ra.empty:
        return ranked.assign(potency_tier=pd.Series(dtype=object))

    # Normalize column name lookups
    rmap = _normalize_colnames(ra)
    amap = _normalize_colnames(actives_df)

    # Candidate ID fields (various spellings seen across CSVs)
    id_candidates = [
        "compound chembl id", "compound_chembl_id", "molecule chembl id", "molecule_chembl_id",
    ]
    r_id_col = _pick_first_present(rmap, id_candidates)
    a_id_col = _pick_first_present(amap, id_candidates)

    if r_id_col and a_id_col:
        # ID-based join
        aff = actives_df[[a_id_col] + [c for c in ["Standard Value (nM)"] if c in actives_df.columns]].copy()
        # Normalize ID text for join
        ra["__join_id__"] = ra[r_id_col].astype(str).str.upper().str.strip()
        aff["__join_id__"] = aff[a_id_col].astype(str).str.upper().str.strip()
        aff = aff.dropna(subset=["__join_id__"]).drop_duplicates(subset=["__join_id__"])  # one potency per compound (assumed median in prep)
        merged = ra.merge(aff[["__join_id__", "Standard Value (nM)"]], on="__join_id__", how="left")
        merged.drop(columns=["__join_id__"], inplace=True, errors="ignore")
    else:
        # SMILES-based join
        r_smiles_col = _pick_first_present(rmap, ["canonical_smiles", "smiles"])  # ranked
        a_smiles_col = _pick_first_present(amap, ["canonical_smiles", "smiles"])  # actives
        if not r_smiles_col or not a_smiles_col:
            return ra.assign(potency_tier=pd.Series(dtype=object))
        ra["__join_smiles__"] = ra[r_smiles_col].astype(str).str.strip()
        act = actives_df.copy()
        act["__join_smiles__"] = act[a_smiles_col].astype(str).str.strip()
        aff = act[[c for c in ["__join_smiles__", "Standard Value (nM)"] if c in act.columns]].copy()
        aff = aff.dropna(subset=["__join_smiles__"]).drop_duplicates(subset=["__join_smiles__"])  # unique
        merged = ra.merge(aff, on="__join_smiles__", how="left")
        merged.drop(columns=["__join_smiles__"], inplace=True, errors="ignore")

    # Assign potency tiers (may be NaN if potency unavailable)
    merged["potency_tier"] = merged["Standard Value (nM)"].apply(_assign_potency_tier)
    # Stitch back with ZINC rows
    rz = ranked[ranked["source"] != "actives"].copy()
    out = pd.concat([merged, rz], ignore_index=True)
    # Preserve original sorting order; caller will re-sort anyway
    return out


def _ef_at_percent(ranked: pd.DataFrame, tier: str, pct: float) -> Optional[float]:
    """Compute EF@p% for a potency tier.

    EF definition used here:
      EF@p% = (hits_top_k_tier / N_tier) / (k / N_total) = hits_top_k_tier * N_total / (k * N_tier)
    where k = ceil(p * N_total) and N_total counts all rows (actives + zinc) in ranked_scores.
    This equals hits / (p * N_tier) when k == p*N_total; we keep the exact form to avoid rounding bias.
    """
    if ranked.empty:
        return None
    N_total = int(len(ranked))
    k = max(1, int(np.ceil(pct * N_total)))
    topk = ranked.head(k)
    ra = ranked[(ranked["source"] == "actives") & (ranked["potency_tier"].notna())]
    N_tier = int((ra["potency_tier"] == tier).sum())
    if N_tier == 0:
        return None
    hits = int(((topk["source"] == "actives") & (topk["potency_tier"] == tier)).sum())
    ef = (hits * N_total) / (k * N_tier)
    return float(ef)


def _overall_ef_at_percent(ranked: pd.DataFrame, pct: float) -> Optional[float]:
    if ranked.empty:
        return None
    N_total = int(len(ranked))
    k = max(1, int(np.ceil(pct * N_total)))
    topk = ranked.head(k)
    N_actives = int((ranked["source"] == "actives").sum())
    if N_actives == 0:
        return None
    hits = int((topk["source"] == "actives").sum())
    ef = (hits * N_total) / (k * N_actives)
    return float(ef)


def _per_run_stratified(ranked_path: Path, summary_cfg: dict, target: str, out_dir: Path, logger: logging.Logger) -> Optional[dict]:
    if not ranked_path.exists():
        return None
    ranked = pd.read_csv(ranked_path, low_memory=False)
    # Ensure correct order (top ranked first): prefer score desc, else distance asc
    if "score" in ranked.columns:
        ranked = ranked.sort_values(by="score", ascending=False, kind="mergesort").reset_index(drop=True)
    elif "distance" in ranked.columns:
        ranked = ranked.sort_values(by="distance", ascending=True, kind="mergesort").reset_index(drop=True)
    actives_df = _read_actives_df(summary_cfg, target, logger)
    if actives_df is None or actives_df.empty:
        logger.info("No actives dataframe available; skipping stratified computation for this run")
        return None
    # Join potency tiers to actives rows
    ranked = _join_affinity_to_actives(ranked, actives_df)
    # Debug diagnostics: counts and compositions
    N_total = int(len(ranked))
    k = max(1, int(np.ceil(0.01 * N_total)))
    N_actives = int((ranked["source"] == "actives").sum())
    topk = ranked.head(k)
    comp_all = ranked[ranked["source"] == "actives"]["potency_tier"].value_counts(dropna=True).to_dict()
    comp_top = topk[topk["source"] == "actives"]["potency_tier"].value_counts(dropna=True).to_dict()
    overall_ef1 = _overall_ef_at_percent(ranked, 0.01)
    logger.info(f"EF@1% overall: {overall_ef1:.3f} | N_total={N_total}, k={k}, N_actives={N_actives}")
    logger.info(f"Actives tier counts (all): {comp_all}")
    logger.info(f"Actives tier counts (top1%): {comp_top}")
    # Extract config/meta for plotting later
    method = None
    dim = None
    rep = None
    nn = None
    md = None
    try:
        # Try to read from summary config
        method = str(summary_cfg.get("method")) if "method" in summary_cfg else None
        dim_val = summary_cfg.get("dim") if "dim" in summary_cfg else None
        try:
            dim = int(dim_val) if dim_val is not None and str(dim_val).strip() != "" else None
        except Exception:
            dim = None
        if isinstance(summary_cfg.get("representation"), str):
            rep = str(summary_cfg.get("representation")).lower()
        umap_params = summary_cfg.get("umap_params", {}) or {}
        nn = umap_params.get("n_neighbors")
        md = umap_params.get("min_dist")
    except Exception:
        pass

    # Compute EF@1/5/10 for each tier
    out: Dict[str, object] = {"run_ranked": str(ranked_path), "method": method, "dim": dim, "representation": rep,
                              "umap_n_neighbors": nn, "umap_min_dist": md}
    for tier in ["High", "Medium", "Weak"]:
        out[f"EF1_{tier}"] = _ef_at_percent(ranked, tier, 0.01)
        out[f"EF5_{tier}"] = _ef_at_percent(ranked, tier, 0.05)
        out[f"EF10_{tier}"] = _ef_at_percent(ranked, tier, 0.10)
    out["EF1_overall"] = overall_ef1
    # Save per-run CSV
    per_run_csv = out_dir / f"{ranked_path.parent.parent.name}_stratified.csv"
    cols_keep = [c for c in ["source", "score", "distance", "potency_tier", "Compound ChEMBL ID", "canonical_smiles", "SMILES"] if c in ranked.columns]
    ranked[cols_keep].to_csv(per_run_csv, index=False)
    logger.info(f"Saved per-run stratified table: {per_run_csv}")
    return out


def _plot_stratified_ef1_all_umap_vs_pca(df: pd.DataFrame, out_dir: Path, logger: logging.Logger) -> None:
    """Plot PCA vs all UMAP hyperparameters (separate) with potency tiers on x-axis.
    Creates one figure per representation; columns = dimensions; hue = method+params.
    """
    dfx = df.copy()
    dfx["representation"] = dfx["representation"].fillna("features")
    dfx["method"] = dfx["method"].fillna("unknown").str.lower()
    # Long dataframe for EF@1%
    recs: List[Dict] = []
    for _, r in dfx.iterrows():
        for tier in ("High","Medium","Weak"):
            val = r.get(f"EF1_{tier}")
            if pd.notna(val):
                label = "PCA" if r.get("method") == "pca" else f"UMAP(nn={r.get('umap_n_neighbors')}, md={r.get('umap_min_dist')})"
                recs.append({
                    "representation": r.get("representation"),
                    "dim": r.get("dim"),
                    "group": label,
                    "tier": tier,
                    "value": float(val) if val is not None else np.nan,
                })
    if not recs:
        logger.info("No EF@1% records available for plotting (all-UMAP vs PCA)")
        return
    lf = pd.DataFrame(recs)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    for rep in sorted(lf["representation"].unique()):
        sub_rep = lf[lf["representation"] == rep]
        dims = sorted([int(d) for d in sub_rep["dim"].dropna().unique()])
        ncols = max(1, len(dims))
        fig, axes = plt.subplots(1, ncols, figsize=(5.2*ncols, 4.2), sharey=True)
        if ncols == 1:
            axes = [axes]  # type: ignore
        for ax, d in zip(axes, dims):
            s = sub_rep[sub_rep["dim"] == d]
            if s.empty:
                ax.set_visible(False)
                continue
            if _HAVE_SNS and sns is not None:
                sns.barplot(data=s, x="tier", y="value", hue="group", ax=ax, errorbar=("sd"), capsize=0.1)
            else:
                groups = sorted(s["group"].unique())
                tiers = ["High","Medium","Weak"]
                x = np.arange(len(tiers))
                width = 0.8/max(1,len(groups))
                for i,g in enumerate(groups):
                    vals = [s[(s["tier"]==t) & (s["group"]==g)]["value"].mean() for t in tiers]
                    stds = [s[(s["tier"]==t) & (s["group"]==g)]["value"].std() for t in tiers]
                    ax.bar(x + (i-(len(groups)-1)/2)*width, vals, yerr=stds, width=width, capsize=3, label=g)
                ax.set_xticks(x)
                ax.set_xticklabels(tiers)
            ax.set_title(f"dim={d}")
            ax.set_xlabel("Potency tier")
            ax.set_ylabel("EF@1%")
        handles, labels = axes[-1].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02,0.5))
        fig.suptitle(f"PCA vs all UMAP (by hyperparams) — {rep}")
        fig.tight_layout(rect=(0,0,0.92,0.95))
        fig.savefig(plots_dir / f"strat_ef1_all_umap_vs_pca_{rep}.png")
        fig.savefig(plots_dir / f"strat_ef1_all_umap_vs_pca_{rep}.pdf")
        plt.close(fig)


def _plot_stratified_ef1_best_umap_vs_pca(df: pd.DataFrame, out_dir: Path, logger: logging.Logger) -> None:
    """Plot PCA vs best UMAP per tier with potency tiers on x-axis.
    Best selected per (representation, dim, tier) by mean EF@1% across runs.
    """
    dfx = df.copy()
    dfx["representation"] = dfx["representation"].fillna("features")
    dfx["method"] = dfx["method"].fillna("unknown").str.lower()

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for rep in sorted(dfx["representation"].unique()):
        sub_rep = dfx[dfx["representation"] == rep]
        dims = sorted([int(d) for d in sub_rep["dim"].dropna().unique()])
        ncols = max(1, len(dims))
        fig, axes = plt.subplots(1, ncols, figsize=(5.2*ncols, 4.2), sharey=True)
        if ncols == 1:
            axes = [axes]  # type: ignore
        for ax, d in zip(axes, dims):
            sdim = sub_rep[sub_rep["dim"] == d]
            # Build table of candidates
            records = []
            for tier in ("High","Medium","Weak"):
                # PCA mean
                pca_vals = sdim[sdim["method"]=="pca"][f"EF1_{tier}"]
                pca_mean = float(pca_vals.mean()) if len(pca_vals)>0 else np.nan
                # Best UMAP per tier
                umap = sdim[sdim["method"]=="umap"].copy()
                if umap.empty:
                    continue
                # group by hyperparams
                g = umap.groupby(["umap_n_neighbors","umap_min_dist"], dropna=False)[f"EF1_{tier}"]
                umap_best = g.mean().sort_values(ascending=False).head(1)
                if len(umap_best) == 0:
                    continue
                (best_nn, best_md), best_val = umap_best.index[0], float(umap_best.iloc[0])
                records.append({
                    "tier": tier,
                    "PCA": pca_mean,
                    f"UMAP(nn={best_nn}, md={best_md})": best_val,
                })
            if not records:
                ax.set_visible(False)
                continue
            wide = pd.DataFrame(records)
            long = wide.melt(id_vars=["tier"], var_name="group", value_name="value")
            if _HAVE_SNS and sns is not None:
                sns.barplot(data=long, x="tier", y="value", hue="group", ax=ax, errorbar=None)
            else:
                groups = [c for c in wide.columns if c != "tier"]
                tiers = ["High","Medium","Weak"]
                x = np.arange(len(tiers))
                width = 0.8/max(1,len(groups))
                for i,gname in enumerate(groups):
                    vals = []
                    for t in tiers:
                        srow = wide[wide["tier"]==t][gname]
                        if srow.empty:
                            vals.append(np.nan)
                        else:
                            try:
                                vals.append(float(srow.iloc[0]))
                            except Exception:
                                vals.append(np.nan)
                    ax.bar(x + (i-(len(groups)-1)/2)*width, vals, width=width, label=gname)
                ax.set_xticks(x)
                ax.set_xticklabels(tiers)
            ax.set_title(f"dim={d}")
            ax.set_xlabel("Potency tier")
            ax.set_ylabel("EF@1%")
        handles, labels = axes[-1].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02,0.5))
        fig.suptitle(f"PCA vs best UMAP — {rep}")
        fig.tight_layout(rect=(0,0,0.92,0.95))
        fig.savefig(plots_dir / f"strat_ef1_best_umap_vs_pca_{rep}.png")
        fig.savefig(plots_dir / f"strat_ef1_best_umap_vs_pca_{rep}.pdf")
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Phase 1 Potency-Stratified Scores (v4 molfuse)")
    ap.add_argument("--workspace_dir", type=str, required=True, help="Path to experiment_workspace_v4")
    ap.add_argument("--phase", type=str, default="phase1", help="Phase subdirectory (default: phase1)")
    ap.add_argument("--output_dir", type=str, default="reporting/phase1_stratified", help="Output directory for stratified summaries")
    ap.add_argument("--metrics", type=str, default="ef1", help="Comma-separated list of metrics to plot (ef1,ef5,ef10). Default: ef1 only")
    args = ap.parse_args()

    workspace_dir = Path(args.workspace_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = _setup_logger(out_dir)

    phase_dir = workspace_dir / args.phase
    if not phase_dir.exists():
        logger.info(f"Phase directory not found: {phase_dir}")
        return

    rows: List[Dict] = []
    for run_dir in sorted([p for p in phase_dir.iterdir() if p.is_dir()]):
        ranked_path = run_dir / "artifacts" / "ranked_scores.csv"
        summary = _read_json(run_dir / "logs" / "phase1_summary.json")
        metrics = _read_json(run_dir / "metrics" / "metrics.json")
        if metrics is None:
            continue
        target = str(metrics.get("target", "UNKNOWN"))
        cfg = summary.get("config", {}) if (summary and isinstance(summary.get("config"), dict)) else {}
        logger.info(f"RUN: {run_dir.name}")
        res = _per_run_stratified(ranked_path, cfg, target, out_dir, logger)
        if res is None:
            continue
        res.update({
            "run_name": run_dir.name,
            "target": target,
        })
        rows.append(res)

    if not rows:
        logger.info("No stratified results produced (no eligible runs)")
        return

    df = pd.DataFrame(rows)
    out_csv = out_dir / "phase1_stratified_summary.csv"
    df.to_csv(out_csv, index=False)
    logger.info(f"Saved stratified summary: {out_csv} (rows={len(df)})")

    # Plot required comparisons for EF@1% by default
    logger.info("START: stratified EF@1% comparison plots")
    _plot_stratified_ef1_all_umap_vs_pca(df, out_dir, logger)
    _plot_stratified_ef1_best_umap_vs_pca(df, out_dir, logger)

    # Optional simple tier bars for selected metrics
    do_metrics = [m.strip().lower() for m in str(args.metrics).split(",") if m.strip()]
    if not do_metrics:
        do_metrics = ["ef1"]
    logger.info(f"Tiered bar plots for metrics: {do_metrics}")
    try:
        # Build long-form dataframe: columns [tier, metric, value]
        recs: List[Dict] = []
        for _, r in df.iterrows():
            for tier in ("High", "Medium", "Weak"):
                pairs = [("EF@1%", f"EF1_{tier}")]
                if "ef5" in do_metrics:
                    pairs.append(("EF@5%", f"EF5_{tier}"))
                if "ef10" in do_metrics:
                    pairs.append(("EF@10%", f"EF10_{tier}"))
                for m_name, col in pairs:
                    val = r.get(col, np.nan)
                    try:
                        val_f = float(val) if val is not None else np.nan
                    except Exception:
                        val_f = np.nan
                    recs.append({"tier": tier, "metric": m_name, "value": val_f})
        dfl = pd.DataFrame(recs)
        # Plot one figure per selected metric
        plots_dir = out_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        metrics_to_plot = ["EF@1%"]
        if "ef5" in do_metrics:
            metrics_to_plot.append("EF@5%")
        if "ef10" in do_metrics:
            metrics_to_plot.append("EF@10%")
        for metric in metrics_to_plot:
            sub = dfl[dfl["metric"] == metric].copy()
            if sub.empty:
                continue
            # Aggregate
            agg = sub.groupby("tier", as_index=False)["value"].agg(["mean", "std"]).reset_index()
            fig, ax = plt.subplots(figsize=(6, 4))
            if _HAVE_SNS and sns is not None:
                sns.barplot(data=sub, x="tier", y="value", ax=ax, errorbar=("sd"), capsize=0.1)
            else:
                tiers = ["High", "Medium", "Weak"]
                means = [sub[sub["tier"] == t]["value"].mean() for t in tiers]
                stds = [sub[sub["tier"] == t]["value"].std() for t in tiers]
                x = np.arange(len(tiers))
                ax.bar(x, means, yerr=stds, capsize=4, width=0.6, color=["#4daf4a", "#377eb8", "#984ea3"])  # green/blue/purple
                ax.set_xticks(x)
                ax.set_xticklabels(tiers)
            ax.set_title(f"Tiered {metric}")
            ax.set_ylabel(metric)
            ax.set_xlabel("Potency tier")
            fig.tight_layout()
            fig.savefig(plots_dir / f"tier_bars_{metric.replace('@','at').replace('%','pct').replace('/','_')}.png", dpi=300)
            fig.savefig(plots_dir / f"tier_bars_{metric.replace('@','at').replace('%','pct').replace('/','_')}.pdf")
            plt.close(fig)
        logger.info("Finished tiered bar plots")
    except Exception as e:
        logger.info(f"Tiered bar plots failed: {e}")


if __name__ == "__main__":
    main()
