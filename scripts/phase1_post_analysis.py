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
    --metrics ef1,ef5,ef10 \
    --distance_xranges 0-1,0-5

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

# Matplotlib is standard; seaborn is optional (fallback to plain matplotlib if missing)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Consistent style for all figures
matplotlib.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.2,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
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
    g = (
        df.groupby(group_keys)
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
            sub = df_g[df_g["dim"] == d]
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
                bars = ax.bar(xpos, y, width=bw, label=r, color=colors.get(r, None), yerr=yerr, capsize=3)
                # Numeric labels
                for b, val in zip(bars, y):
                    if np.isfinite(val):
                        y_upper = (y1 if sharey and y1 is not None else max(1.0, b.get_height()))
                        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02*float(y_upper),
                                f"{val:.1f}", ha="center", va="bottom", fontsize=8)

            ax.set_xticks(x)
            ax.set_xticklabels([m.upper() if m in ("pca","umap") else m for m in methods])
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
        # Legend outside
        handles, labels = axes[-1].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, title="Representation", loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.suptitle(f"{label} (mean ± sd)")
        fig.tight_layout(rect=(0,0,0.85,0.95))
        p_png = out_dir / f"bars_{metric}_combined.png"
        p_pdf = out_dir / f"bars_{metric}_combined.pdf"
        fig.savefig(p_png)
        fig.savefig(p_pdf)
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
    nrows, ncols = max(1, len(reps)), max(1, len(dims))
    # Shared color scale across all panels
    vmin = float(sub["ef1_mean"].min()) if np.isfinite(sub["ef1_mean"].min()) else None
    vmax = float(sub["ef1_mean"].max()) if np.isfinite(sub["ef1_mean"].max()) else None
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8*ncols, 4.2*nrows), squeeze=False)
    mappable = None
    for i, rep in enumerate(reps):
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
            if _HAVE_SNS and sns is not None:
                hm = sns.heatmap(pivot, annot=False, cmap="viridis", vmin=vmin, vmax=vmax, cbar=False, ax=ax)
                mappable = hm.collections[0]
            else:
                im = ax.imshow(pivot.values, aspect="auto", cmap="viridis", origin="upper", vmin=vmin, vmax=vmax)
                ax.set_yticks(range(len(pivot.index)))
                ax.set_yticklabels([str(x) for x in pivot.index])
                ax.set_xticks(range(len(pivot.columns)))
                ax.set_xticklabels([str(x) for x in pivot.columns])
                mappable = im
            if i == 0:
                ax.set_title(f"dim={d}")
            if j == 0:
                ax.set_ylabel(f"{rep}\nmin_dist")
            ax.set_xlabel("n_neighbors")

    # Shared colorbar
    if mappable is not None:
        cbar = fig.colorbar(mappable, ax=axes, location="right", shrink=0.9)
        cbar.set_label("EF@1% (mean)")
    fig.suptitle("UMAP EF@1% heatmaps (shared scale)")
    fig.tight_layout(rect=(0,0,0.92,0.95))
    fig.savefig(out_dir / "umap_heatmap_grid.png")
    fig.savefig(out_dir / "umap_heatmap_grid.pdf")
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
                sns.violinplot(data=sub, x="representation", y="ef_1%", ax=ax, inner="quartile")
            else:
                groups = sorted(sub["representation"].dropna().unique().tolist())
                data = [sub[sub["representation"]==g]["ef_1%"].dropna().to_numpy() for g in groups]
                ax.boxplot(data, labels=groups)
            if i == 0:
                ax.set_title(f"dim={d}")
            if j == 0:
                ax.set_ylabel(f"{m.upper()}\nEF@1%")
            ax.set_xlabel("Representation")

    fig.suptitle("Seed variability — EF@1%")
    fig.tight_layout(rect=(0,0,0.98,0.95))
    fig.savefig(out_dir / "seed_variability_grid.png")
    fig.savefig(out_dir / "seed_variability_grid.pdf")
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


def _collect_distance_samples(workspace_dir: Path, phase: str, max_actives: int = 50000, max_zinc: int = 100000) -> pd.DataFrame:
    """Scan ranked_scores.csv across runs and collect distance samples for actives and zinc."""
    phase_dir = workspace_dir / phase
    rows: List[pd.DataFrame] = []
    for run_dir in sorted([p for p in phase_dir.iterdir() if p.is_dir()]):
        ranked_path = run_dir / "artifacts" / "ranked_scores.csv"
        if not ranked_path.exists():
            continue
        try:
            df = pd.read_csv(ranked_path, usecols=["source", "distance"])  # minimal columns
            # Split and sample
            df_act = df[df["source"] == "actives"]
            df_zinc = df[df["source"] == "zinc"]
            if len(df_act) > max_actives:
                df_act = df_act.sample(n=max_actives, random_state=42)
            if len(df_zinc) > max_zinc:
                df_zinc = df_zinc.sample(n=max_zinc, random_state=42)
            rows.append(pd.concat([df_act, df_zinc], ignore_index=True))
        except Exception:
            continue
    if not rows:
        return pd.DataFrame(columns=["source", "distance"])
    return pd.concat(rows, ignore_index=True)


def _plot_distance_distributions(df_dist: pd.DataFrame, out_dir: Path, logger: logging.Logger, x_ranges: Optional[List[Tuple[float,float]]] = None) -> None:
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    if df_dist.empty:
        logger.info("No distance samples found; skipping distance distribution plots")
        return
    # Ranges setup
    if not x_ranges:
        x_ranges = [(0.0, 1.0), (0.0, 5.0)]
    logger.info(f"Distance hist x-ranges: {x_ranges}")

    # Histogram figure with columns per range
    ncols = len(x_ranges)
    fig, axes = plt.subplots(1, ncols, figsize=(5*ncols, 4), sharey=True)
    if ncols == 1:
        axes = [axes]  # type: ignore
    for ax, (xmin, xmax) in zip(axes, x_ranges):
        for src, color in [("actives", "tab:green"), ("zinc", "tab:blue")]:
            dd = df_dist[df_dist["source"] == src]["distance"].dropna().to_numpy(dtype=float)
            if dd.size == 0:
                continue
            mask = (dd >= xmin) & (dd <= xmax)
            ax.hist(dd[mask], bins=80, range=(xmin, xmax), density=True, alpha=0.5, label=src, color=color)
        ax.set_xlim(xmin, xmax)
        ax.set_xlabel("min-distance to MF cloud")
        ax.set_title(f"Histogram [{xmin:g},{xmax:g}]")
    axes[0].set_ylabel("density")
    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout(rect=(0,0,0.92,1))
    fig.savefig(plots_dir / "distance_hist_ranges.png")
    fig.savefig(plots_dir / "distance_hist_ranges.pdf")
    plt.close(fig)

    # CDF with xlim to the largest upper bound
    xmax = max([xr[1] for xr in x_ranges]) if x_ranges else None
    fig, ax = plt.subplots(figsize=(7, 4))
    for src, color in [("actives", "tab:green"), ("zinc", "tab:blue")]:
        dd = df_dist[df_dist["source"] == src]["distance"].dropna().to_numpy(dtype=float)
        if dd.size == 0:
            continue
        dd_sorted = np.sort(dd)
        y = np.linspace(0, 1, len(dd_sorted), endpoint=True)
        ax.plot(dd_sorted, y, label=src, color=color)
        # Percentiles
        for p in (50, 90):
            q = float(np.percentile(dd_sorted, p))
            ax.axvline(q, color=color, alpha=0.2, linestyle=":")
    if xmax is not None:
        ax.set_xlim(0, xmax)
    ax.set_title("Distance CDF (actives vs zinc)")
    ax.set_xlabel("min-distance to MF cloud")
    ax.set_ylabel("CDF")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "distance_cdf.png")
    fig.savefig(plots_dir / "distance_cdf.pdf")
    plt.close(fig)
    logger.info("Saved distance hist/CDF plots with fixed ranges")


def main():
    ap = argparse.ArgumentParser(description="Phase 1 Post Analysis (v4 molfuse)")
    ap.add_argument("--workspace_dir", type=str, required=True, help="Path to experiment_workspace_v4")
    ap.add_argument("--phase", type=str, default="phase1", help="Phase subdirectory (default: phase1)")
    ap.add_argument("--output_dir", type=str, default="reporting/phase1_post_analysis", help="Output directory for summaries and plots")
    ap.add_argument("--metrics", type=str, default="ef1,ef5,ef10", help="Comma-separated metrics to plot as bars (ef1,ef5,ef10)")
    ap.add_argument("--no-sharey", dest="sharey", action="store_false", help="Disable shared y-axis for combined bar plots")
    ap.add_argument("--distance_xranges", type=str, default="0-1,0-5", help="Comma-separated x ranges for distance hist, e.g., 0-1,0-5")
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
    metrics = [m for m in metrics if m in ("ef1","ef5","ef10")]
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

    # 4b) Distance distributions (actives vs zinc)
    logger.info("START: distance distributions")
    df_dist = _collect_distance_samples(workspace_dir, phase=args.phase)
    # Parse ranges like "0-1,0-5"
    xranges: List[Tuple[float,float]] = []
    try:
        for token in str(args.distance_xranges).split(","):
            token = token.strip()
            if not token:
                continue
            parts = token.split("-")
            if len(parts) == 2:
                xranges.append((float(parts[0]), float(parts[1])))
    except Exception:
        xranges = [(0.0, 1.0), (0.0, 5.0)]
    _plot_distance_distributions(df_dist, out_dir, logger, x_ranges=xranges)
    manifest.setdefault("distances", []).extend([
        str(out_dir/"plots"/"distance_hist_ranges.png"),
        str(out_dir/"plots"/"distance_hist_ranges.pdf"),
        str(out_dir/"plots"/"distance_cdf.png"),
        str(out_dir/"plots"/"distance_cdf.pdf"),
    ])

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
    logger.info(f"Saved report: {report_txt}")
    logger.info("DONE")


if __name__ == "__main__":
    main()
