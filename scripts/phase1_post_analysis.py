#!/usr/bin/env python3
"""
Phase 1 Post Analysis (v4 molfuse)

- Scans a workspace for Phase 1 runs (works with partially completed experiments)
- Aggregates metrics and lightweight config metadata
- Produces per-run and grouped summaries
- Extracts best configs per method/representation/dimension
- Generates key plots (PNG + PDF) without relying on archived scripts

Usage:
  python scripts/phase1_post_analysis.py \
      --workspace_dir experiment_workspace_v4 \
      --phase phase1 \
    --output_dir reporting/phase1_post_analysis \
        --metrics ef1 \
        --distance_xranges 0-0.5

Notes:
- Robust to missing files; skips runs lacking metrics.json
- Tries to read logs/phase1_summary.json for config; falls back to inference
- Avoids importing archived analysis modules; fully self-contained
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

# Matplotlib is standard; seaborn is optional (fallback to plain matplotlib if missing)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Publication-quality style for all figures
matplotlib.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.titleweight": "bold",
    "legend.fontsize": 9,
    "legend.framealpha": 0.9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "axes.linewidth": 1.2,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "lines.linewidth": 2.0,
    "patch.linewidth": 0.5,
})

from typing import Any
try:
    import seaborn as sns  # type: ignore
    _HAVE_SNS = True
except Exception:
    sns = None  # type: ignore[assignment]
    _HAVE_SNS = False


def _read_json(path: Path) -> Optional[dict]:
    try:
        with path.open("r") as f:
            return json.load(f)
    except Exception:
        return None


def _infer_representation(run_dir: Path) -> str:
    # Prefer logs/phase1_summary.json config.representation
    summary = _read_json(run_dir / "logs" / "phase1_summary.json")
    if summary and isinstance(summary.get("config"), dict):
        rep = str(summary["config"].get("representation", "features")).lower()
        if rep in ("features", "fingerprints"):
            return rep
    # Fallback: inspect scaler.joblib
    try:
        import joblib
        scaler = joblib.load(run_dir / "artifacts" / "scaler.joblib")
        if isinstance(scaler, dict) and scaler.get("type") == "passthrough":
            return "fingerprints"
        return "features"
    except Exception:
        return "features"


def _extract_umap_params(run_dir: Path, method: str) -> Tuple[Optional[int], Optional[float], Optional[str]]:
    if method != "umap":
        return None, None, None
    # Try summary config first
    summary = _read_json(run_dir / "logs" / "phase1_summary.json")
    if summary and isinstance(summary.get("config"), dict):
        cfg = summary["config"]
        params = cfg.get("umap_params", {}) or {}
        nn = params.get("n_neighbors")
        md = params.get("min_dist")
        metric = params.get("metric")
        try:
            nn = int(nn) if nn is not None else None
        except Exception:
            nn = None
        try:
            md = float(md) if md is not None else None
        except Exception:
            md = None
        if isinstance(metric, str):
            return nn, md, metric
        return nn, md, None
    return None, None, None


def _extract_method_dim_target(metrics: dict) -> Tuple[str, int, str, float]:
    method = str(metrics.get("method", "pca")).lower()
    dim = int(metrics.get("dim", 2))
    target = str(metrics.get("target", "UNKNOWN"))
    cutoff = float(metrics.get("affinity_cutoff_nM", 100000))
    return method, dim, target, cutoff


def scan_runs(workspace_dir: Path, phase: str = "phase1") -> pd.DataFrame:
    rows: List[Dict] = []
    phase_dir = workspace_dir / phase
    if not phase_dir.exists():
        raise FileNotFoundError(f"Phase directory not found: {phase_dir}")

    for run_dir in sorted([p for p in phase_dir.iterdir() if p.is_dir()]):
        metrics_path = run_dir / "metrics" / "metrics.json"
        if not metrics_path.exists():
            continue  # run incomplete; skip
        metrics = _read_json(metrics_path)
        if not metrics:
            continue
        method, dim, target, cutoff = _extract_method_dim_target(metrics)
        rep = _infer_representation(run_dir)
        nn, md, metric = _extract_umap_params(run_dir, method)

        # Attempt to get replicate index from config or run_name
        replicate: Optional[int] = None
        summary = _read_json(run_dir / "logs" / "phase1_summary.json")
        if summary and isinstance(summary.get("config"), dict):
            replicate = summary["config"].get("replicate")
            try:
                replicate = int(replicate) if replicate is not None else None
            except Exception:
                replicate = None
        if replicate is None:
            # Fallback: parse from run_name suffix like _repN
            m = None
            try:
                import re
                m = re.search(r"_rep(\d+)$", run_dir.name)
            except Exception:
                m = None
            if m:
                try:
                    replicate = int(m.group(1))
                except Exception:
                    replicate = None

        row = {
            "run_name": run_dir.name,
            "representation": rep,
            "method": method,
            "dim": dim,
            "target": target,
            "affinity_cutoff_nM": cutoff,
            "umap_n_neighbors": nn,
            "umap_min_dist": md,
            "umap_metric": metric,
            "replicate": replicate,
            # metrics
            "ef_1%": metrics.get("ef_1%", np.nan),
            "ef_5%": metrics.get("ef_5%", np.nan),
            "ef_10%": metrics.get("ef_10%", np.nan),
            "roc_auc": metrics.get("roc_auc", np.nan),
            "pr_auc": metrics.get("pr_auc", np.nan),
            "spearman_rho": metrics.get("spearman_rho", np.nan),
            "spearman_p": metrics.get("spearman_p", np.nan),
            "n_actives": metrics.get("n_actives", np.nan),
            "n_zinc_eval": metrics.get("n_zinc_eval", np.nan),
            "n_mf_for_scoring": metrics.get("n_mf_for_scoring", np.nan),
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=[
            "run_name","representation","method","dim","target","affinity_cutoff_nM",
            "umap_n_neighbors","umap_min_dist","umap_metric","replicate",
            "ef_1%","ef_5%","ef_10%","roc_auc","pr_auc","spearman_rho","spearman_p",
            "n_actives","n_zinc_eval","n_mf_for_scoring"
        ])

    return pd.DataFrame(rows)


def group_and_best(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    # Group by knobs that define a config; treat replicate as a replicate
    group_keys = ["representation", "method", "dim", "umap_n_neighbors", "umap_min_dist", "umap_metric"]
    # Important: include rows with NaNs in UMAP-only keys (so PCA isn't dropped)
    g = (
        df.groupby(group_keys, dropna=False)
        .agg(
            ef1_mean=("ef_1%", "mean"), ef1_std=("ef_1%", "std"),
            ef5_mean=("ef_5%", "mean"), ef10_mean=("ef_10%", "mean"),
            roc_mean=("roc_auc", "mean"), pr_mean=("pr_auc", "mean"),
            n_runs=("run_name", "count")
        )
        .reset_index()
    )

    # Best configs per representation x method x dim (by ef1_mean)
    best: Dict[str, dict] = {}
    for (rep, method, dim), sub in g.groupby(["representation", "method", "dim"], dropna=False):
        if len(sub) == 0:
            continue
        top = sub.sort_values("ef1_mean", ascending=False).iloc[0]
        key = f"{rep}__{method}__dim{int(dim)}"
        best[key] = {
            "representation": rep,
            "method": method,
            "dim": int(dim),
            "umap_n_neighbors": None if method != "umap" else (None if pd.isna(top["umap_n_neighbors"]) else int(top["umap_n_neighbors"])),
            "umap_min_dist": None if method != "umap" else (None if pd.isna(top["umap_min_dist"]) else float(top["umap_min_dist"])),
            "umap_metric": None if method != "umap" else (None if pd.isna(top["umap_metric"]) else str(top["umap_metric"]))
        }
    return g, best


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _bar_colors() -> Dict[str, str]:
    # Stable mapping across figures
    return {
        "features": "#1f77b4",  # blue
        "fingerprints": "#ff7f0e",  # orange
    }


def _compute_global_ylim(df_g: pd.DataFrame, mean_col: str, std_col: str) -> Tuple[float, float]:
    vals = df_g[mean_col].astype(float)
    errs = df_g[std_col].fillna(0).astype(float) if std_col in df_g.columns else 0
    top = (vals + errs).max()
    bottom = max(0.0, float((vals - (errs if isinstance(errs, pd.Series) else 0)).min()))
    # Add headroom
    head = 0.05 * top if np.isfinite(top) else 1.0
    return bottom, top + head


def plot_bars_combined(
    df_g: pd.DataFrame,
    out_dir: Path,
    metrics: List[str],
    sharey: bool,
    logger: logging.Logger,
) -> List[Path]:
    """Create combined bar figures per metric across dimensions with shared y-axis.

    metrics items must be in {"ef1","ef5","ef10"}.
    """
    _ensure_dir(out_dir)
    saved: List[Path] = []
    dims = sorted(df_g["dim"].dropna().unique().astype(int).tolist())
    methods = sorted(df_g["method"].dropna().unique().tolist())
    reps = sorted(df_g["representation"].dropna().unique().tolist())
    colors = _bar_colors()

    for metric in metrics:
        mean_col = {"ef1": "ef1_mean", "ef5": "ef5_mean", "ef10": "ef10_mean"}[metric]
        std_col = "ef1_std" if metric == "ef1" else None  # std only computed for ef1 in current grouping
        # Derive std for other metrics if present; otherwise zero
        if std_col is None:
            std_col = "tmp_std_zero"
            df_g = df_g.copy()
            df_g[std_col] = 0.0

        # Augment with a synthetic "UMAP (best)" category so bars can include best-UMAP alongside PCA and UMAP(avg)
        df_aug = df_g.copy()
        if "umap" in methods:
            best_rows: List[pd.DataFrame] = []
            for r in reps:
                sub_umap = df_g[(df_g["method"] == "umap") & (df_g["representation"] == r)]
                if sub_umap.empty:
                    continue
                # best per dim within this rep
                best_per_dim = sub_umap.sort_values(mean_col, ascending=False).groupby("dim", as_index=False).head(1)
                if not best_per_dim.empty:
                    tmp = best_per_dim.copy()
                    tmp["method"] = "umap_best"
                    best_rows.append(tmp)
            if best_rows:
                df_aug = pd.concat([df_aug, pd.concat(best_rows, ignore_index=True)], ignore_index=True)
                if "umap_best" not in methods:
                    methods = [m for m in methods if m != "umap_best"] + ["umap_best"]

        if sharey:
            y0, y1 = _compute_global_ylim(df_g, mean_col, std_col)
            logger.info(f"Bars[{metric}] global ylim: ({y0:.3f}, {y1:.3f})")
        else:
            y0 = y1 = None  # type: ignore

        ncols = max(1, len(dims))
        fig, axes = plt.subplots(1, ncols, figsize=(4.5*ncols, 4), sharey=sharey)
        if ncols == 1:
            axes = [axes]  # type: ignore

        for ax, d in zip(axes, dims):
            sub = df_aug[df_aug["dim"] == d]
            x = np.arange(len(methods), dtype=float)
            total_width = 0.8
            bw = total_width / max(1, len(reps))
            offset0 = - (len(reps)-1) / 2 * bw

            for i, r in enumerate(reps):
                y = []
                yerr = []
                for m in methods:
                    row = sub[(sub["method"]==m) & (sub["representation"]==r)]
                    if row.empty:
                        y.append(np.nan)
                        yerr.append(0.0)
                    else:
                        y.append(float(row.iloc[0][mean_col]))
                        yerr.append(float(row.iloc[0][std_col]) if std_col in row.columns else 0.0)
                xpos = x + offset0 + i*bw
                # Use slightly different shade for best-UMAP by overlaying hatch on the UMAP-best bars
                bar_colors = [colors.get(r, None) for _ in methods]
                bars = ax.bar(xpos, y, width=bw, label=r, color=bar_colors, yerr=yerr, capsize=3)
                # Apply hatching to highlight UMAP-best category
                for b, m in zip(bars, methods):
                    if m == "umap_best":
                        b.set_hatch("//")
                # Numeric labels
                for b, val in zip(bars, y):
                    if np.isfinite(val):
                        y_upper = (y1 if sharey and y1 is not None else max(1.0, b.get_height()))
                        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02*float(y_upper),
                                f"{val:.1f}", ha="center", va="bottom", fontsize=8)

            ax.set_xticks(x)
            def _xlabel(m: str) -> str:
                if m == "pca":
                    return "PCA"
                if m == "umap":
                    return "UMAP (avg)"
                if m == "umap_best":
                    return "UMAP (best)"
                return m
            ax.set_xticklabels([_xlabel(m) for m in methods])
            ax.set_xlabel("Method")
            if sharey:
                ax.set_ylim(y0, y1)
            ax.set_title(f"dim={d}")

        # Label like EF@1%, EF@5%, EF@10%
        label = {
            "ef1": "EF@1%",
            "ef5": "EF@5%",
            "ef10": "EF@10%",
        }.get(metric, metric)
        axes[0].set_ylabel(label)
        # Legend outside with proper spacing
        handles, labels_leg = axes[-1].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels_leg, title="Representation", loc="center left", 
                      bbox_to_anchor=(1.0, 0.5), frameon=True, fontsize=10)
        fig.suptitle(f"{label} (mean ± sd)", fontsize=14, fontweight='bold')
        fig.tight_layout(rect=(0, 0, 0.88, 0.96))
        p_png = out_dir / f"bars_{metric}_combined.png"
        p_pdf = out_dir / f"bars_{metric}_combined.pdf"
        fig.savefig(p_png, bbox_inches='tight', dpi=300)
        fig.savefig(p_pdf, bbox_inches='tight')
        plt.close(fig)
        saved.extend([p_png, p_pdf])

    return saved


def plot_umap_heatmaps(df_g: pd.DataFrame, out_dir: Path) -> None:
    _ensure_dir(out_dir)
    sub = df_g[df_g["method"] == "umap"].copy()
    if sub.empty:
        return
    reps = sorted(sub["representation"].dropna().unique().tolist())
    dims = sorted(sub["dim"].dropna().unique().astype(int).tolist())
    
    # Fixed layout: 2 rows (features, fingerprints) x N columns (dimensions)
    nrows = 2
    ncols = len(dims)
    
    # Shared color scale across all panels
    vmin = float(sub["ef1_mean"].min()) if np.isfinite(sub["ef1_mean"].min()) else None
    vmax = float(sub["ef1_mean"].max()) if np.isfinite(sub["ef1_mean"].max()) else None
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4.5*nrows), squeeze=False)
    mappable = None
    
    # Row 0: features, Row 1: fingerprints
    rep_order = ["features", "fingerprints"]
    
    for i, rep in enumerate(rep_order):
        if rep not in reps:
            # Hide entire row if representation not present
            for j in range(ncols):
                axes[i][j].set_visible(False)
            continue
            
        for j, d in enumerate(dims):
            ax = axes[i][j]
            ss = sub[(sub["representation"] == rep) & (sub["dim"] == d)]
            if ss.empty:
                ax.set_visible(False)
                continue
            try:
                pivot = ss.pivot_table(index="umap_min_dist", columns="umap_n_neighbors", values="ef1_mean")
            except Exception:
                ax.set_visible(False)
                continue
            # If only a single hyperparameter cell exists (1x1), it's not informative; skip that panel
            try:
                if hasattr(pivot, "shape") and tuple(pivot.shape) == (1, 1):
                    ax.set_visible(False)
                    continue
            except Exception:
                pass
            if _HAVE_SNS and sns is not None:
                hm = sns.heatmap(pivot, annot=True, fmt=".1f", cmap="viridis", 
                                vmin=vmin, vmax=vmax, cbar=False, ax=ax, 
                                annot_kws={"fontsize": 8})
                mappable = hm.collections[0]
            else:
                im = ax.imshow(pivot.values, aspect="auto", cmap="viridis", origin="upper", vmin=vmin, vmax=vmax)
                ax.set_yticks(range(len(pivot.index)))
                ax.set_yticklabels([str(x) for x in pivot.index], fontsize=9)
                ax.set_xticks(range(len(pivot.columns)))
                ax.set_xticklabels([str(x) for x in pivot.columns], fontsize=9)
                mappable = im
            
            # Titles only on top row
            if i == 0:
                ax.set_title(f"dim={d}", fontsize=11, fontweight='bold')
            
            # Y-label (representation name and axis label) only on leftmost column
            if j == 0:
                ax.set_ylabel(f"{rep}\numap_min_dist", fontsize=10, fontweight='bold')
            else:
                ax.set_ylabel("")
                
            ax.set_xlabel("n_neighbors", fontsize=9)

    # Shared colorbar on the right
    if mappable is not None:
        cbar_ax = fig.add_axes((0.92, 0.15, 0.02, 0.7))
        cbar = fig.colorbar(mappable, cax=cbar_ax)
        cbar.set_label("EF@1% (mean)", fontsize=11, fontweight='bold')
        cbar.ax.tick_params(labelsize=9)
    
    fig.suptitle("UMAP Hyperparameter Sensitivity: EF@1%", fontsize=14, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 0.91, 0.96))
    fig.savefig(out_dir / "umap_heatmap_grid.png", dpi=300, bbox_inches='tight')
    fig.savefig(out_dir / "umap_heatmap_grid.pdf", bbox_inches='tight')
    plt.close(fig)


def plot_seed_variability(df: pd.DataFrame, out_dir: Path) -> None:
    # Unified figure: rows=methods, cols=dims; hue=representation
    _ensure_dir(out_dir)
    methods = sorted(df["method"].dropna().unique().tolist())
    dims = sorted(df["dim"].dropna().unique().astype(int).tolist())
    nrows, ncols = max(1, len(methods)), max(1, len(dims))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6*ncols, 3.8*nrows), sharey=True)
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])  # type: ignore
    elif nrows == 1:
        axes = np.array([axes])  # shape (1, ncols)  # type: ignore
    elif ncols == 1:
        axes = np.array([[ax] for ax in axes])  # shape (nrows,1)  # type: ignore

    for i, m in enumerate(methods):
        for j, d in enumerate(dims):
            ax = axes[i][j]
            sub = df[(df["method"] == m) & (df["dim"] == d)]
            if sub.empty:
                ax.set_visible(False)
                continue
            if _HAVE_SNS and sns is not None:
                sns.violinplot(data=sub, x="representation", y="ef_1%", ax=ax, 
                              inner="quartile", palette="Set2", linewidth=1.5)
            else:
                groups = sorted(sub["representation"].dropna().unique().tolist())
                data = [sub[sub["representation"]==g]["ef_1%"].dropna().to_numpy() for g in groups]
                bp = ax.boxplot(data, labels=groups, patch_artist=True)
                # Style boxplots
                for patch in bp['boxes']:
                    patch.set_facecolor('#8fbce6')
                    patch.set_alpha(0.7)
            if i == 0:
                ax.set_title(f"dim={d}", fontsize=11, fontweight='bold')
            if j == 0:
                ax.set_ylabel(f"{m.upper()}\nEF@1%", fontsize=10, fontweight='bold')
            else:
                ax.set_ylabel("")
            ax.set_xlabel("Representation", fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle("Seed Variability — EF@1%", fontsize=14, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_dir / "seed_variability_grid.png", dpi=300, bbox_inches='tight')
    fig.savefig(out_dir / "seed_variability_grid.pdf", bbox_inches='tight')
    plt.close(fig)


def _setup_logger(out_dir: Path) -> logging.Logger:
    logger = logging.getLogger("phase1_post_analysis")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(out_dir / "analysis.log", mode="w")
    sh = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _collect_distance_samples_by_label(
    workspace_dir: Path,
    phase: str,
    df_runs: pd.DataFrame,
    df_grouped: pd.DataFrame,
    logger: logging.Logger,
    sources: Optional[List[str]] = None,
    per_run_cap: int = 100000,
) -> pd.DataFrame:
    """Collect distance samples labelled by comparison group.

    Groups:
      - PCA (best across dims for its available representation)
      - UMAP (best) features
      - UMAP (best) fingerprints
      - UMAP (avg) features [same dim as its best]
      - UMAP (avg) fingerprints [same dim as its best]
    """
    labels: List[Tuple[str, pd.DataFrame]] = []

    if not sources:
        sources = ["zinc"]

    def _is_close(a: Optional[float], b: Optional[float]) -> bool:
        if a is None or b is None or pd.isna(a) or pd.isna(b):
            return False
        return bool(np.isclose(float(a), float(b), rtol=1e-6, atol=1e-6))

    # Pick best PCA overall
    pca_rows = df_grouped[df_grouped["method"] == "pca"].copy()
    if not pca_rows.empty:
        pca_best = pca_rows.sort_values("ef1_mean", ascending=False).iloc[0]
        pca_rep = str(pca_best["representation"])  # usually "features"
        pca_dim = int(pca_best["dim"])  # type: ignore
        sel_pca = df_runs[(df_runs["method"] == "pca") & (df_runs["representation"] == pca_rep) & (df_runs["dim"] == pca_dim)]
        labels.append(("PCA", sel_pca))

    # For each representation, pick best UMAP and define avg set on same dim
    for rep in ["features", "fingerprints"]:
        umap_rep = df_grouped[(df_grouped["method"] == "umap") & (df_grouped["representation"] == rep)].copy()
        if umap_rep.empty:
            continue
        umap_best = umap_rep.sort_values("ef1_mean", ascending=False).iloc[0]
        dim_best = int(umap_best["dim"])  # type: ignore
        nn_best = umap_best.get("umap_n_neighbors")
        md_best = umap_best.get("umap_min_dist")
        metric_best = umap_best.get("umap_metric")

        # best
        runs_best = df_runs[
            (df_runs["method"] == "umap") & (df_runs["representation"] == rep) & (df_runs["dim"] == dim_best)
        ]
        if not runs_best.empty:
            # filter by matching HPs
            mask = runs_best.apply(
                lambda r: (
                    (r.get("umap_metric") == metric_best) and
                    (r.get("umap_n_neighbors") == nn_best) and
                    _is_close(r.get("umap_min_dist"), md_best)
                ), axis=1
            )
            runs_best = runs_best[mask]
        labels.append((f"UMAP best {rep}", runs_best))

        # avg on same dim
        runs_avg = df_runs[(df_runs["method"] == "umap") & (df_runs["representation"] == rep) & (df_runs["dim"] == dim_best)]
        labels.append((f"UMAP avg {rep}", runs_avg))

    # Read distances and tag by label (optionally for multiple sources)
    phase_dir = workspace_dir / phase
    out_rows: List[pd.DataFrame] = []
    for label, sel in labels:
        if sel is None or sel.empty:
            logger.info(f"No runs matched for label '{label}', skipping")
            continue
        dists: List[pd.Series] = []
        for run_name in sel["run_name"].tolist():
            ranked_path = phase_dir / run_name / "artifacts" / "ranked_scores.csv"
            if not ranked_path.exists():
                continue
            try:
                df = pd.read_csv(ranked_path, usecols=["source", "distance"])  # minimal columns
                # Concatenate requested sources for this run
                dd_all: List[pd.Series] = []
                for src in sources:
                    dd_src = df[df["source"] == src]["distance"].dropna()
                    if len(dd_src) > per_run_cap:
                        dd_src = dd_src.sample(n=per_run_cap, random_state=42)
                    # Tag source in index for later merge
                    dd_src.index = pd.Index([src] * len(dd_src), name="source")
                    dd_all.append(dd_src)
                if dd_all:
                    dists.append(pd.concat(dd_all))
            except Exception:
                continue
        if not dists:
            logger.info(f"No distances read for label '{label}'")
            continue
        # Stack preserves the 'source' from index (if present)
        ser = pd.concat(dists)
        # If source index exists, move to column; otherwise mark as 'zinc' (legacy)
        if ser.index.name == "source":
            df_lab = ser.reset_index()
            df_lab.columns = ["source", "distance"]
        else:
            df_lab = pd.DataFrame({"source": "zinc", "distance": ser.values})
        df_lab.insert(0, "label", label)
        out_rows.append(df_lab)
    if not out_rows:
        return pd.DataFrame(columns=["label", "source", "distance"])
    return pd.concat(out_rows, ignore_index=True)


def _plot_distance_hist_cdf_by_label(
    df_dist: pd.DataFrame,
    out_dir: Path,
    logger: logging.Logger,
    x_range: Tuple[float, float] = (0.0, 0.5),
) -> None:
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    if df_dist.empty:
        logger.info("No labelled distance samples found; skipping distance plots")
        return

    xmin, xmax = x_range
    labels = list(dict.fromkeys(df_dist["label"].tolist()))  # preserve order
    color_map = {
        "PCA": "#2ca02c",  # green
        "UMAP best features": "#1f77b4",  # blue
        "UMAP avg features": "#8fbce6",   # light blue
        "UMAP best fingerprints": "#ff7f0e",  # orange
        "UMAP avg fingerprints": "#ffbb78",   # light orange
    }
    
    # IMPROVED CDF: Overlaid ZINC and ACTIVES with distinct line styles
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    for lab in labels:
        color = color_map.get(lab, None)
        for source, ls, alpha, lw in [("zinc", "--", 0.7, 1.8), ("actives", "-", 1.0, 2.5)]:
            dd = df_dist[(df_dist["label"] == lab) & (df_dist["source"] == source)]["distance"].dropna().to_numpy(dtype=float)
            if dd.size == 0:
                continue
            dd_sorted = np.sort(dd)
            y = np.linspace(0, 1, len(dd_sorted), endpoint=True)
            # Construct label with method and source
            plot_label = f"{lab} ({source.upper()})" if source == "actives" else None
            ax.plot(dd_sorted, y, label=plot_label, color=color, linestyle=ls, 
                   linewidth=lw, alpha=alpha)
    
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("min-distance to MF cloud", fontsize=12)
    ax.set_ylabel("Cumulative Distribution Function", fontsize=12)
    ax.set_title("Distance CDF: Method Comparison (Solid=ACTIVES, Dashed=ZINC)", 
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Create custom legend with both method colors and line styles
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=color_map.get(lab), linewidth=2.5, linestyle='-', label=lab)
        for lab in labels if lab in color_map
    ]
    legend_elements.extend([
        Line2D([0], [0], color='black', linewidth=2, linestyle='-', label='ACTIVES'),
        Line2D([0], [0], color='black', linewidth=2, linestyle='--', label='ZINC'),
    ])
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9, frameon=True, ncol=1)
    
    fig.tight_layout()
    fig.savefig(plots_dir / "distance_cdf_compare_methods.png", dpi=300, bbox_inches='tight')
    fig.savefig(plots_dir / "distance_cdf_compare_methods.pdf", bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved improved CDF plot with overlaid ZINC/ACTIVES comparison")


def _plot_umap_histograms_split(
    df_dist: pd.DataFrame,
    out_dir: Path,
    logger: logging.Logger,
    x_range: Tuple[float, float] = (0.0, 0.5),
) -> List[Path]:
    """Create improved histograms with log-scale y-axis for better visibility.
    
    Two visualizations:
    1. Grid layout (5 methods) with log-scale y-axis
    2. Stacked layout per method with linear and log scales side-by-side
    """
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    xmin, xmax = x_range
    
    method_labels = [
        "PCA",
        "UMAP best features",
        "UMAP avg features", 
        "UMAP best fingerprints",
        "UMAP avg fingerprints",
    ]
    
    # ===== VISUALIZATION 1: Grid with Log Scale =====
    fig, axes = plt.subplots(2, 5, figsize=(20, 7), sharex=True)
    
    for idx, label in enumerate(method_labels):
        ax_lin = axes[0, idx]  # Linear scale (top row)
        ax_log = axes[1, idx]  # Log scale (bottom row)
        sub = df_dist[df_dist["label"] == label]
        
        if sub.empty:
            logger.info(f"No distance samples for '{label}', skipping")
            ax_lin.set_visible(False)
            ax_log.set_visible(False)
            continue
        
        # Plot ZINC and ACTIVES on both linear and log scales
        for src, color, alpha in [("zinc", "#5A7FC0", 0.7), ("actives", "#E85D2D", 0.8)]:
            dd = sub[sub["source"] == src]["distance"].dropna().to_numpy(dtype=float)
            if dd.size == 0:
                continue
            mask = (dd >= xmin) & (dd <= xmax)
            dd_filtered = dd[mask]
            
            # Linear scale
            ax_lin.hist(dd_filtered, bins=50, range=(xmin, xmax), density=True, 
                       alpha=alpha, label=src.upper(), color=color, edgecolor='white', linewidth=0.3)
            
            # Log scale
            ax_log.hist(dd_filtered, bins=50, range=(xmin, xmax), density=True, 
                       alpha=alpha, label=src.upper(), color=color, edgecolor='white', linewidth=0.3)
        
        # Configure linear axis (top)
        ax_lin.set_xlim(xmin, xmax)
        ax_lin.set_title(label, fontsize=10, fontweight='bold')
        ax_lin.grid(True, alpha=0.25, axis='y')
        if idx == 0:
            ax_lin.set_ylabel("Density (linear)", fontsize=10)
        if idx == 4:
            ax_lin.legend(loc="upper right", fontsize=8, frameon=True)
        
        # Configure log axis (bottom)
        ax_log.set_xlim(xmin, xmax)
        ax_log.set_yscale('log')
        ax_log.set_xlabel("min-distance to MF", fontsize=9)
        ax_log.grid(True, alpha=0.25, which='both')
        if idx == 0:
            ax_log.set_ylabel("Density (log scale)", fontsize=10)
    
    fig.suptitle(f"Distance Distributions: Linear vs Log Scale [{xmin:g}, {xmax:g}]", 
                fontsize=14, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    
    p_png_grid = plots_dir / "distance_hist_grid_log_scale.png"
    p_pdf_grid = plots_dir / "distance_hist_grid_log_scale.pdf"
    fig.savefig(p_png_grid, dpi=300, bbox_inches='tight')
    fig.savefig(p_pdf_grid, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved log-scale histogram grid: {p_png_grid}")
    
    # ===== VISUALIZATION 2: KDE (Kernel Density) overlay for smoother comparison =====
    fig2, axes2 = plt.subplots(1, 5, figsize=(20, 4), sharey=False)
    
    for idx, label in enumerate(method_labels):
        ax = axes2[idx]
        sub = df_dist[df_dist["label"] == label]
        
        if sub.empty:
            ax.set_visible(False)
            continue
        
        # Use KDE for smoother visualization
        for src, color, ls, lw in [("zinc", "#5A7FC0", "--", 2), ("actives", "#E85D2D", "-", 2.5)]:
            dd = sub[sub["source"] == src]["distance"].dropna().to_numpy(dtype=float)
            if dd.size < 10:  # Need sufficient samples for KDE
                continue
            mask = (dd >= xmin) & (dd <= xmax)
            dd_filtered = dd[mask]
            
            if len(dd_filtered) > 10:
                # Use scipy's KDE for better control
                kde = gaussian_kde(dd_filtered, bw_method='scott')
                x_eval = np.linspace(xmin, xmax, 300)
                density = kde(x_eval)
                ax.plot(x_eval, density, label=src.upper(), color=color, 
                       linestyle=ls, linewidth=lw, alpha=0.9)
        
        ax.set_xlim(xmin, xmax)
        ax.set_xlabel("min-distance to MF", fontsize=9)
        ax.set_title(label, fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.25, axis='y')
        
        if idx == 0:
            ax.set_ylabel("Density (KDE)", fontsize=10)
        if idx == 4:
            ax.legend(loc="upper right", fontsize=9, frameon=True)
    
    fig2.suptitle(f"Distance Distributions: Kernel Density Estimates [{xmin:g}, {xmax:g}]", 
                 fontsize=14, fontweight='bold')
    fig2.tight_layout(rect=(0, 0, 1, 0.94))
    
    p_png_kde = plots_dir / "distance_hist_grid_kde.png"
    p_pdf_kde = plots_dir / "distance_hist_grid_kde.pdf"
    fig2.savefig(p_png_kde, dpi=300, bbox_inches='tight')
    fig2.savefig(p_pdf_kde, bbox_inches='tight')
    plt.close(fig2)
    logger.info(f"Saved KDE histogram: {p_png_kde}")
    
    return [p_png_grid, p_pdf_grid, p_png_kde, p_pdf_kde]


def _plot_2d_scatter_embeddings(
    workspace_dir: Path,
    phase: str,
    df_runs: pd.DataFrame,
    df_grouped: pd.DataFrame,
    out_dir: Path,
    logger: logging.Logger,
    max_points_per_group: Optional[int] = 5000,
) -> List[Path]:
    """Create 2D scatter plots for selected method/representation combinations.
    
    Plots four panels:
    1. PCA features (2D)
    2. PCA fingerprints (2D)
    3. UMAP best features (2D)
    4. UMAP best fingerprints (2D)
    
    Each panel shows three groups: MF cloud, ZINC, ACTIVES
    """
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Define the four configurations to plot
    configs_to_plot = []
    
    # 1. PCA features (best)
    pca_feat = df_grouped[(df_grouped["method"] == "pca") & 
                          (df_grouped["representation"] == "features") & 
                          (df_grouped["dim"] == 2)].copy()
    if not pca_feat.empty:
        best_pca_feat = pca_feat.sort_values("ef1_mean", ascending=False).iloc[0]
        configs_to_plot.append({
            "label": "PCA features",
            "method": "pca",
            "representation": "features",
            "dim": 2,
            "umap_params": None
        })
    
    # 2. PCA fingerprints (best)
    pca_fing = df_grouped[(df_grouped["method"] == "pca") & 
                          (df_grouped["representation"] == "fingerprints") & 
                          (df_grouped["dim"] == 2)].copy()
    if not pca_fing.empty:
        best_pca_fing = pca_fing.sort_values("ef1_mean", ascending=False).iloc[0]
        configs_to_plot.append({
            "label": "PCA fingerprints",
            "method": "pca",
            "representation": "fingerprints",
            "dim": 2,
            "umap_params": None
        })
    
    # 3. UMAP best features
    umap_feat = df_grouped[(df_grouped["method"] == "umap") & 
                           (df_grouped["representation"] == "features") & 
                           (df_grouped["dim"] == 2)].copy()
    if not umap_feat.empty:
        best_umap_feat = umap_feat.sort_values("ef1_mean", ascending=False).iloc[0]
        configs_to_plot.append({
            "label": "UMAP best features",
            "method": "umap",
            "representation": "features",
            "dim": 2,
            "umap_params": {
                "n_neighbors": best_umap_feat.get("umap_n_neighbors"),
                "min_dist": best_umap_feat.get("umap_min_dist"),
                "metric": best_umap_feat.get("umap_metric")
            }
        })
    
    # 4. UMAP best fingerprints
    umap_fing = df_grouped[(df_grouped["method"] == "umap") & 
                           (df_grouped["representation"] == "fingerprints") & 
                           (df_grouped["dim"] == 2)].copy()
    if not umap_fing.empty:
        best_umap_fing = umap_fing.sort_values("ef1_mean", ascending=False).iloc[0]
        configs_to_plot.append({
            "label": "UMAP best fingerprints",
            "method": "umap",
            "representation": "fingerprints",
            "dim": 2,
            "umap_params": {
                "n_neighbors": best_umap_fing.get("umap_n_neighbors"),
                "min_dist": best_umap_fing.get("umap_min_dist"),
                "metric": best_umap_fing.get("umap_metric")
            }
        })
    
    if not configs_to_plot:
        logger.warning("No 2D configurations found for scatter plotting")
        return []
    
    # Create figure with 2x2 or 1x4 layout depending on what's available
    n_configs = len(configs_to_plot)
    if n_configs == 4:
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
    elif n_configs == 3:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    elif n_configs == 2:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(8, 7))
        axes = [axes]
    
    phase_dir = workspace_dir / phase
    
    for idx, config in enumerate(configs_to_plot):
        ax = axes[idx]
        
        # Find matching run directory
        matching_runs = df_runs[
            (df_runs["method"] == config["method"]) &
            (df_runs["representation"] == config["representation"]) &
            (df_runs["dim"] == config["dim"])
        ]
        
        if config["umap_params"]:
            # Filter by UMAP hyperparameters
            nn_target = config["umap_params"]["n_neighbors"]
            md_target = config["umap_params"]["min_dist"]
            metric_target = config["umap_params"]["metric"]
            matching_runs = matching_runs[
                (matching_runs["umap_n_neighbors"] == nn_target) &
                (np.isclose(matching_runs["umap_min_dist"].fillna(-1), 
                           float(md_target) if not pd.isna(md_target) else -1, atol=1e-6)) &
                (matching_runs["umap_metric"] == metric_target)
            ]
        
        if matching_runs.empty:
            logger.warning(f"No matching run found for {config['label']}")
            ax.set_visible(False)
            continue
        
        # Take first matching run
        run_row = matching_runs.iloc[0]
        run_dir = phase_dir / run_row["run_name"]
        artifacts_dir = run_dir / "artifacts"
        
        # Load embeddings
        emb_mf_path = artifacts_dir / "embedding_mf.csv"
        emb_zinc_path = artifacts_dir / "embedding_zinc.csv"
        emb_actives_path = artifacts_dir / "embedding_actives.csv"
        
        if not all([p.exists() for p in [emb_mf_path, emb_zinc_path, emb_actives_path]]):
            logger.warning(f"Missing embedding files for {config['label']} in {run_dir.name}")
            ax.set_visible(False)
            continue
        
        # Load data
        try:
            df_mf = pd.read_csv(emb_mf_path)
            df_zinc = pd.read_csv(emb_zinc_path)
            df_actives = pd.read_csv(emb_actives_path)
        except Exception as e:
            logger.warning(f"Failed to load embeddings for {config['label']}: {e}")
            ax.set_visible(False)
            continue
        
        # Subsample if necessary
        if max_points_per_group:
            if len(df_mf) > max_points_per_group:
                df_mf = df_mf.sample(n=max_points_per_group, random_state=42)
            if len(df_zinc) > max_points_per_group:
                df_zinc = df_zinc.sample(n=max_points_per_group, random_state=42)
            if len(df_actives) > max_points_per_group:
                df_actives = df_actives.sample(n=max_points_per_group, random_state=42)
        
        # Extract 2D coordinates (z0, z1)
        mf_x, mf_y = np.asarray(df_mf["z0"]), np.asarray(df_mf["z1"])
        zinc_x, zinc_y = np.asarray(df_zinc["z0"]), np.asarray(df_zinc["z1"])
        actives_x, actives_y = np.asarray(df_actives["z0"]), np.asarray(df_actives["z1"])
        
        # Plot with distinct styling for each group
        # Priority: ACTIVES > MF cloud > ZINC (zorder controls layering)
        
        # ZINC: bottom layer, small blue points
        ax.scatter(zinc_x, zinc_y, c="#5A7FC0", s=6, alpha=0.4, label=f"ZINC (n={len(df_zinc):,})", 
                  rasterized=True, edgecolors='none', zorder=1)
        
        # MF cloud: middle layer, medium gray points
        ax.scatter(mf_x, mf_y, c="#888888", s=10, alpha=0.5, label=f"MF cloud (n={len(df_mf):,})", 
                  rasterized=True, edgecolors='none', zorder=2)
        
        # ACTIVES: top layer, larger red points (most prominent)
        ax.scatter(actives_x, actives_y, c="#E85D2D", s=40, alpha=0.9, label=f"ACTIVES (n={len(df_actives):,})", 
                  edgecolors='white', linewidths=0.5, zorder=3)
        
        # Formatting
        ax.set_xlabel("Dimension 1 (z0)", fontsize=10)
        ax.set_ylabel("Dimension 2 (z1)", fontsize=10)
        ax.set_title(config["label"], fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.2)
        ax.legend(loc="best", fontsize=8, framealpha=0.9, markerscale=1.5)
        
        # Add EF@1% and hyperparameters as text annotation
        ef1_val = run_row.get("ef_1%", np.nan)
        annotation_text = []
        if not pd.isna(ef1_val):
            annotation_text.append(f"EF@1% = {ef1_val:.1f}")
        
        # Add hyperparameters for UMAP
        if config["umap_params"]:
            nn = config["umap_params"]["n_neighbors"]
            md = config["umap_params"]["min_dist"]
            metric = config["umap_params"]["metric"]
            if nn is not None and md is not None and metric is not None:
                annotation_text.append(f"n_neighbors={nn}, min_dist={md:.3f}")
                annotation_text.append(f"metric={metric}")
        
        if annotation_text:
            ax.text(0.02, 0.98, "\n".join(annotation_text), 
                   transform=ax.transAxes, fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    
    fig.suptitle("2D Embeddings: Molecular Group Separation", fontsize=14, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    
    p_png = plots_dir / "scatter_2d_embeddings.png"
    p_pdf = plots_dir / "scatter_2d_embeddings.pdf"
    fig.savefig(p_png, dpi=300, bbox_inches='tight')
    fig.savefig(p_pdf, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved 2D scatter plots: {p_png}")
    
    return [p_png, p_pdf]


def _plot_2d_density_embeddings(
    workspace_dir: Path,
    phase: str,
    df_runs: pd.DataFrame,
    df_grouped: pd.DataFrame,
    out_dir: Path,
    logger: logging.Logger,
) -> List[Path]:
    """Create density-based visualizations for full point clouds.
    
    Three visualization strategies per config:
    1. Hexbin density plots (efficient for large datasets)
    2. Contour density overlays
    3. Small multiples showing each group separately
    """
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Define the four configurations to plot (same as scatter)
    configs_to_plot = []
    
    # 1. PCA features
    pca_feat = df_grouped[(df_grouped["method"] == "pca") & 
                          (df_grouped["representation"] == "features") & 
                          (df_grouped["dim"] == 2)].copy()
    if not pca_feat.empty:
        configs_to_plot.append({
            "label": "PCA features",
            "method": "pca",
            "representation": "features",
            "dim": 2,
            "umap_params": None
        })
    
    # 2. PCA fingerprints
    pca_fing = df_grouped[(df_grouped["method"] == "pca") & 
                          (df_grouped["representation"] == "fingerprints") & 
                          (df_grouped["dim"] == 2)].copy()
    if not pca_fing.empty:
        configs_to_plot.append({
            "label": "PCA fingerprints",
            "method": "pca",
            "representation": "fingerprints",
            "dim": 2,
            "umap_params": None
        })
    
    # 3. UMAP best features
    umap_feat = df_grouped[(df_grouped["method"] == "umap") & 
                           (df_grouped["representation"] == "features") & 
                           (df_grouped["dim"] == 2)].copy()
    if not umap_feat.empty:
        best_umap_feat = umap_feat.sort_values("ef1_mean", ascending=False).iloc[0]
        configs_to_plot.append({
            "label": "UMAP best features",
            "method": "umap",
            "representation": "features",
            "dim": 2,
            "umap_params": {
                "n_neighbors": best_umap_feat.get("umap_n_neighbors"),
                "min_dist": best_umap_feat.get("umap_min_dist"),
                "metric": best_umap_feat.get("umap_metric")
            }
        })
    
    # 4. UMAP best fingerprints
    umap_fing = df_grouped[(df_grouped["method"] == "umap") & 
                           (df_grouped["representation"] == "fingerprints") & 
                           (df_grouped["dim"] == 2)].copy()
    if not umap_fing.empty:
        best_umap_fing = umap_fing.sort_values("ef1_mean", ascending=False).iloc[0]
        configs_to_plot.append({
            "label": "UMAP best fingerprints",
            "method": "umap",
            "representation": "fingerprints",
            "dim": 2,
            "umap_params": {
                "n_neighbors": best_umap_fing.get("umap_n_neighbors"),
                "min_dist": best_umap_fing.get("umap_min_dist"),
                "metric": best_umap_fing.get("umap_metric")
            }
        })
    
    if not configs_to_plot:
        logger.warning("No 2D configurations found for density plotting")
        return []
    
    saved_files = []
    phase_dir = workspace_dir / phase
    
    # ===== VISUALIZATION 1: Hexbin Density (efficient, full data) =====
    n_configs = len(configs_to_plot)
    if n_configs == 4:
        fig_hex, axes_hex = plt.subplots(2, 2, figsize=(16, 14))
        axes_hex = axes_hex.flatten()
    else:
        fig_hex, axes_hex = plt.subplots(1, n_configs, figsize=(6*n_configs, 6))
        if n_configs == 1:
            axes_hex = [axes_hex]
    
    for idx, config in enumerate(configs_to_plot):
        ax = axes_hex[idx]
        
        # Find matching run
        matching_runs = df_runs[
            (df_runs["method"] == config["method"]) &
            (df_runs["representation"] == config["representation"]) &
            (df_runs["dim"] == config["dim"])
        ]
        
        if config["umap_params"]:
            nn_target = config["umap_params"]["n_neighbors"]
            md_target = config["umap_params"]["min_dist"]
            metric_target = config["umap_params"]["metric"]
            matching_runs = matching_runs[
                (matching_runs["umap_n_neighbors"] == nn_target) &
                (np.isclose(matching_runs["umap_min_dist"].fillna(-1), 
                           float(md_target) if not pd.isna(md_target) else -1, atol=1e-6)) &
                (matching_runs["umap_metric"] == metric_target)
            ]
        
        if matching_runs.empty:
            ax.set_visible(False)
            continue
        
        run_row = matching_runs.iloc[0]
        run_dir = phase_dir / run_row["run_name"]
        artifacts_dir = run_dir / "artifacts"
        
        # Load ALL data (no subsampling)
        try:
            df_mf = pd.read_csv(artifacts_dir / "embedding_mf.csv")
            df_zinc = pd.read_csv(artifacts_dir / "embedding_zinc.csv")
            df_actives = pd.read_csv(artifacts_dir / "embedding_actives.csv")
        except Exception as e:
            logger.warning(f"Failed to load embeddings for {config['label']}: {e}")
            ax.set_visible(False)
            continue
        
        # Combine all for hexbin (to get shared extent)
        all_x = np.concatenate([df_mf["z0"], df_zinc["z0"], df_actives["z0"]])
        all_y = np.concatenate([df_mf["z1"], df_zinc["z1"], df_actives["z1"]])
        
        # Create hexbin for ALL molecules (background density)
        hb = ax.hexbin(all_x, all_y, gridsize=50, cmap='Greys', alpha=0.6, 
                      mincnt=1, edgecolors='none', linewidths=0.2)
        
        # Overlay ACTIVES as scatter (highest priority)
        ax.scatter(df_actives["z0"], df_actives["z1"], c='#E85D2D', s=30, alpha=0.9,
                  edgecolors='white', linewidths=0.5, label=f'ACTIVES (n={len(df_actives):,})', zorder=3)
        
        ax.set_xlabel("Dimension 1 (z0)", fontsize=10)
        ax.set_ylabel("Dimension 2 (z1)", fontsize=10)
        ax.set_title(config["label"], fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        
        # Add EF and hyperparams
        ef1_val = run_row.get("ef_1%", np.nan)
        annotation_text = []
        if not pd.isna(ef1_val):
            annotation_text.append(f"EF@1% = {ef1_val:.1f}")
        if config["umap_params"]:
            nn = config["umap_params"]["n_neighbors"]
            md = config["umap_params"]["min_dist"]
            metric = config["umap_params"]["metric"]
            if nn is not None and md is not None:
                annotation_text.append(f"n={nn}, d={md:.3f}, {metric}")
        if annotation_text:
            ax.text(0.02, 0.98, "\n".join(annotation_text), 
                   transform=ax.transAxes, fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    
    fig_hex.suptitle("2D Embeddings: Hexbin Density (Full Data)", fontsize=14, fontweight='bold')
    fig_hex.tight_layout(rect=(0, 0, 1, 0.96))
    
    p_hex_png = plots_dir / "scatter_2d_hexbin_density.png"
    p_hex_pdf = plots_dir / "scatter_2d_hexbin_density.pdf"
    fig_hex.savefig(p_hex_png, dpi=300, bbox_inches='tight')
    fig_hex.savefig(p_hex_pdf, bbox_inches='tight')
    plt.close(fig_hex)
    logger.info(f"Saved hexbin density plot: {p_hex_png}")
    saved_files.extend([p_hex_png, p_hex_pdf])
    
    # ===== VISUALIZATION 2: Contour Overlays (separate groups) =====
    for config in configs_to_plot:
        # Find matching run
        matching_runs = df_runs[
            (df_runs["method"] == config["method"]) &
            (df_runs["representation"] == config["representation"]) &
            (df_runs["dim"] == config["dim"])
        ]
        
        if config["umap_params"]:
            nn_target = config["umap_params"]["n_neighbors"]
            md_target = config["umap_params"]["min_dist"]
            metric_target = config["umap_params"]["metric"]
            matching_runs = matching_runs[
                (matching_runs["umap_n_neighbors"] == nn_target) &
                (np.isclose(matching_runs["umap_min_dist"].fillna(-1), 
                           float(md_target) if not pd.isna(md_target) else -1, atol=1e-6)) &
                (matching_runs["umap_metric"] == metric_target)
            ]
        
        if matching_runs.empty:
            continue
        
        run_row = matching_runs.iloc[0]
        run_dir = phase_dir / run_row["run_name"]
        artifacts_dir = run_dir / "artifacts"
        
        try:
            df_mf = pd.read_csv(artifacts_dir / "embedding_mf.csv")
            df_zinc = pd.read_csv(artifacts_dir / "embedding_zinc.csv")
            df_actives = pd.read_csv(artifacts_dir / "embedding_actives.csv")
        except Exception:
            continue
        
        # Create 1x3 figure showing each group separately + combined
        fig_cont, axes_cont = plt.subplots(1, 4, figsize=(20, 5))
        
        groups = [
            ("MF cloud", df_mf, "#888888", 0),
            ("ZINC", df_zinc, "#5A7FC0", 1),
            ("ACTIVES", df_actives, "#E85D2D", 2),
        ]
        
        # Individual group plots
        for ax_idx, (group_name, df_group, color, _) in enumerate(groups):
            ax = axes_cont[ax_idx]
            if len(df_group) < 100:  # Too few points for KDE
                ax.scatter(df_group["z0"], df_group["z1"], c=color, s=5, alpha=0.6)
            else:
                # KDE contour
                x = np.asarray(df_group["z0"])
                y = np.asarray(df_group["z1"])
                # Subsample for KDE if too large
                if len(x) > 10000:
                    indices = np.random.choice(len(x), 10000, replace=False)
                    x, y = x[indices], y[indices]
                
                try:
                    from scipy.stats import gaussian_kde
                    xy = np.vstack([x, y])
                    kde = gaussian_kde(xy, bw_method='scott')
                    
                    # Create grid
                    x_min, x_max = x.min(), x.max()
                    y_min, y_max = y.min(), y.max()
                    xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
                    positions = np.vstack([xx.ravel(), yy.ravel()])
                    density = kde(positions).reshape(xx.shape)
                    
                    ax.contourf(xx, yy, density, levels=10, cmap='Greys', alpha=0.6)
                    ax.contour(xx, yy, density, levels=5, colors=color, linewidths=1.5, alpha=0.8)
                except Exception:
                    # Fallback to scatter
                    ax.scatter(x, y, c=color, s=5, alpha=0.6)
            
            ax.set_title(f"{group_name} (n={len(df_group):,})", fontsize=10)
            ax.set_xlabel("z0", fontsize=9)
            ax.set_ylabel("z1", fontsize=9)
            ax.grid(True, alpha=0.2)
        
        # Combined overlay plot
        ax_combined = axes_cont[3]
        # Layer order: ZINC -> MF -> ACTIVES
        ax_combined.scatter(df_zinc["z0"], df_zinc["z1"], c="#5A7FC0", s=3, alpha=0.3, 
                           label=f"ZINC", rasterized=True, zorder=1)
        ax_combined.scatter(df_mf["z0"], df_mf["z1"], c="#888888", s=5, alpha=0.4, 
                           label=f"MF cloud", rasterized=True, zorder=2)
        ax_combined.scatter(df_actives["z0"], df_actives["z1"], c="#E85D2D", s=25, alpha=0.9,
                           edgecolors='white', linewidths=0.5, label=f"ACTIVES", zorder=3)
        ax_combined.set_title("Combined", fontsize=10, fontweight='bold')
        ax_combined.set_xlabel("z0", fontsize=9)
        ax_combined.set_ylabel("z1", fontsize=9)
        ax_combined.legend(loc='best', fontsize=8)
        ax_combined.grid(True, alpha=0.2)
        
        # Add EF and params to combined plot
        ef1_val = run_row.get("ef_1%", np.nan)
        annotation_text = []
        if not pd.isna(ef1_val):
            annotation_text.append(f"EF@1% = {ef1_val:.1f}")
        if config["umap_params"]:
            nn = config["umap_params"]["n_neighbors"]
            md = config["umap_params"]["min_dist"]
            metric = config["umap_params"]["metric"]
            if nn is not None and md is not None:
                annotation_text.append(f"n={nn}, d={md:.3f}, {metric}")
        if annotation_text:
            ax_combined.text(0.02, 0.98, "\n".join(annotation_text), 
                           transform=ax_combined.transAxes, fontsize=8, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
        
        fig_cont.suptitle(f"{config['label']} - Group Separation", fontsize=13, fontweight='bold')
        fig_cont.tight_layout(rect=(0, 0, 1, 0.94))
        
        # Save with config-specific name
        safe_label = config['label'].replace(' ', '_').lower()
        p_cont_png = plots_dir / f"scatter_2d_groups_{safe_label}.png"
        p_cont_pdf = plots_dir / f"scatter_2d_groups_{safe_label}.pdf"
        fig_cont.savefig(p_cont_png, dpi=300, bbox_inches='tight')
        fig_cont.savefig(p_cont_pdf, bbox_inches='tight')
        plt.close(fig_cont)
        logger.info(f"Saved group separation plot: {p_cont_png}")
        saved_files.extend([p_cont_png, p_cont_pdf])
    
    return saved_files


def main():
    ap = argparse.ArgumentParser(description="Phase 1 Post Analysis (v4 molfuse)")
    ap.add_argument("--workspace_dir", type=str, required=True, help="Path to experiment_workspace_v4")
    ap.add_argument("--phase", type=str, default="phase1", help="Phase subdirectory (default: phase1)")
    ap.add_argument("--output_dir", type=str, default="reporting/phase1_post_analysis", help="Output directory for summaries and plots")
    ap.add_argument("--metrics", type=str, default="ef1", help="Comma-separated metrics to plot as bars (default: ef1 only)")
    ap.add_argument("--no-sharey", dest="sharey", action="store_false", help="Disable shared y-axis for combined bar plots")
    ap.add_argument("--distance_xranges", type=str, default="0-0.5", help="X range for distance plots, e.g., 0-0.5")
    args = ap.parse_args()

    workspace_dir = Path(args.workspace_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = _setup_logger(out_dir)

    logger.info("START: scan_runs")
    # 1) Scan runs
    df_runs = scan_runs(workspace_dir, phase=args.phase)
    if df_runs.empty:
        logger.info("No completed runs found (no metrics.json). Nothing to analyze.")
        return
    logger.info(f"Found completed runs: n={len(df_runs)}")

    logger.info("START: save per-run summary")
    # 2) Save per-run summary
    runs_csv = out_dir / "phase1_summary_runs.csv"
    df_runs.to_csv(runs_csv, index=False)
    logger.info(f"Saved per-run summary: {runs_csv} (rows={len(df_runs)})")

    logger.info("START: group and best-configs")
    # 3) Group and select best configs
    df_grouped, best_map = group_and_best(df_runs)
    grouped_csv = out_dir / "phase1_summary_grouped.csv"
    df_grouped.to_csv(grouped_csv, index=False)
    logger.info(f"Saved grouped summary: {grouped_csv} (rows={len(df_grouped)})")

    best_json = out_dir / "phase1_best_configs.json"
    best_json.write_text(json.dumps(best_map, indent=2))
    logger.info(f"Saved best configs: {best_json} (n={len(best_map)})")

    logger.info("START: plots (bars, heatmaps, seed variability)")
    # 4) Plots
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    # Parse metrics
    metrics = [m.strip().lower() for m in str(args.metrics).split(",") if m.strip()]
    metrics = [m for m in metrics if m in ("ef1",)]
    if not metrics:
        metrics = ["ef1"]
    manifest: Dict[str, List[str]] = {}

    saved_bars = plot_bars_combined(df_grouped, plot_dir, metrics, getattr(args, "sharey", True), logger)
    manifest["bars"] = [str(p) for p in saved_bars]
    plot_umap_heatmaps(df_grouped, plot_dir)
    manifest.setdefault("heatmaps", []).extend([str(plot_dir/"umap_heatmap_grid.png"), str(plot_dir/"umap_heatmap_grid.pdf")])
    plot_seed_variability(df_runs, plot_dir)
    manifest.setdefault("seed_variability", []).extend([str(plot_dir/"seed_variability_grid.png"), str(plot_dir/"seed_variability_grid.pdf")])
    logger.info("Finished plots (bars/heatmaps/seed variability)")

    # 4b) Distance distributions (PCA vs UMAP variants)
    logger.info("START: distance distributions (method comparison)")
    # Parse a single x-range like "0-0.5"
    try:
        parts = str(args.distance_xranges).split("-")
        xmin, xmax = float(parts[0]), float(parts[1])
    except Exception:
        xmin, xmax = 0.0, 0.5
    df_dist_labeled = _collect_distance_samples_by_label(workspace_dir, args.phase, df_runs, df_grouped, logger, sources=["zinc", "actives"]) 
    _plot_distance_hist_cdf_by_label(df_dist_labeled, out_dir, logger, x_range=(xmin, xmax))
    # Grid histogram showing all methods
    dedicated = _plot_umap_histograms_split(df_dist_labeled, out_dir, logger, x_range=(xmin, xmax))
    manifest.setdefault("distances", []).extend([
        str(out_dir/"plots"/"distance_cdf_compare_methods.png"),
        str(out_dir/"plots"/"distance_cdf_compare_methods.pdf"),
    ] + [str(p) for p in dedicated])

    # 4c) 2D scatter plots for molecular group visualization
    logger.info("START: 2D scatter embeddings")
    scatter_files = _plot_2d_scatter_embeddings(workspace_dir, args.phase, df_runs, df_grouped, out_dir, logger)
    if scatter_files:
        manifest.setdefault("scatter_2d", []).extend([str(p) for p in scatter_files])
    logger.info("Finished 2D scatter plots")
    
    # 4d) 2D density-based visualizations (full data, alternative views)
    logger.info("START: 2D density embeddings (full data)")
    density_files = _plot_2d_density_embeddings(workspace_dir, args.phase, df_runs, df_grouped, out_dir, logger)
    if density_files:
        manifest.setdefault("density_2d", []).extend([str(p) for p in density_files])
    logger.info("Finished 2D density plots")

    # Write manifest
    (out_dir/"plots_manifest.json").write_text(json.dumps(manifest, indent=2))

    logger.info("START: write report")
    # 5) Lightweight report
    report_txt = out_dir / "report.txt"
    with report_txt.open("w") as f:
        f.write("Phase 1 Post Analysis (v4 molfuse)\n")
        f.write(f"Workspace: {workspace_dir}\n")
        f.write(f"Runs analyzed: {len(df_runs)}\n")
        f.write(f"Groups: {len(df_grouped)}\n")
        f.write(f"Best configs: {len(best_map)}\n")
        f.write("\nTop-5 (by EF@1% mean):\n")
        top5 = df_grouped.sort_values("ef1_mean", ascending=False).head(5)
        f.write(top5.to_string(index=False))
        f.write("\n")
        f.write("\nAll sorted (by EF@1% mean):\n")
        topall = df_grouped.sort_values("ef1_mean", ascending=False)
        f.write(topall.to_string(index=False))
        f.write("\n")
    logger.info(f"Saved report: {report_txt}")
    logger.info("DONE")


if __name__ == "__main__":
    main()
