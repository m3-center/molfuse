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
      --output_dir reporting/phase1_post_analysis

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


def plot_bars(df_g: pd.DataFrame, out_dir: Path) -> None:
    # EF@1% by method/representation per dimension
    _ensure_dir(out_dir)
    dims = sorted(df_g["dim"].dropna().unique().tolist())
    for d in dims:
        sub = df_g[df_g["dim"] == d]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        if _HAVE_SNS and sns is not None:
            sns.barplot(
                data=sub,
                x="method", y="ef1_mean", hue="representation",
                ax=ax, capsize=0.1, errwidth=1.2,
                errorbar=("sd"),
            )
        else:
            # Simple grouped bars by representation; no fancy error bars
            reps = sorted(sub["representation"].unique().tolist())
            methods = sorted(sub["method"].unique().tolist())
            width = 0.35
            x = np.arange(len(methods))
            for i, r in enumerate(reps):
                y = [sub[(sub["method"]==m) & (sub["representation"]==r)]["ef1_mean"].mean() for m in methods]
                ax.bar(x + i*width, y, width=width, label=r)
            ax.set_xticks(x + width * (len(reps)-1)/2)
            ax.set_xticklabels(methods)
        ax.set_title(f"EF@1% (mean ± sd) — dim={d}")
        ax.set_ylabel("EF@1%")
        ax.set_xlabel("Method")
        ax.legend(title="Representation")
        fig.tight_layout()
        fig.savefig(out_dir / f"bars_ef1_dim{d}.png", dpi=300)
        fig.savefig(out_dir / f"bars_ef1_dim{d}.pdf")
        plt.close(fig)


def plot_umap_heatmaps(df_g: pd.DataFrame, out_dir: Path) -> None:
    _ensure_dir(out_dir)
    sub = df_g[df_g["method"] == "umap"].copy()
    if sub.empty:
        return
    for rep in sorted(sub["representation"].unique().tolist()):
        for d in sorted(sub["dim"].unique().tolist()):
            ss = sub[(sub["representation"] == rep) & (sub["dim"] == d)]
            if ss.empty:
                continue
            # pivot: rows=min_dist, cols=n_neighbors
            try:
                pivot = ss.pivot_table(index="umap_min_dist", columns="umap_n_neighbors", values="ef1_mean")
            except Exception:
                continue
            fig, ax = plt.subplots(figsize=(7, 5))
            if _HAVE_SNS and sns is not None:
                sns.heatmap(pivot, annot=False, cmap="viridis", ax=ax)
            else:
                im = ax.imshow(pivot.values, aspect="auto", cmap="viridis", origin="upper")
                ax.set_yticks(range(len(pivot.index)))
                ax.set_yticklabels([str(x) for x in pivot.index])
                ax.set_xticks(range(len(pivot.columns)))
                ax.set_xticklabels([str(x) for x in pivot.columns])
                fig.colorbar(im, ax=ax)
            ax.set_title(f"UMAP EF@1% heatmap — {rep}, dim={d}")
            ax.set_xlabel("n_neighbors")
            ax.set_ylabel("min_dist")
            fig.tight_layout()
            fig.savefig(out_dir / f"umap_heatmap_{rep}_dim{d}.png", dpi=300)
            fig.savefig(out_dir / f"umap_heatmap_{rep}_dim{d}.pdf")
            plt.close(fig)


def plot_seed_variability(df: pd.DataFrame, out_dir: Path) -> None:
    # Violin/box for EF@1% per method and dimension
    _ensure_dir(out_dir)
    for d in sorted(df["dim"].dropna().unique().tolist()):
        sub = df[df["dim"] == d]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        if _HAVE_SNS and sns is not None:
            sns.violinplot(data=sub, x="method", y="ef_1%", hue="representation", split=True, ax=ax)
        else:
            # Fallback: box-like scatter
            for i, m in enumerate(sorted(sub["method"].unique().tolist())):
                for j, r in enumerate(sorted(sub["representation"].unique().tolist())):
                    y = sub[(sub["method"]==m) & (sub["representation"]==r)]["ef_1%"].values
                    x = np.full_like(y, i + j*0.3, dtype=float)
                    ax.scatter(x, y, s=10, alpha=0.5, label=f"{m}-{r}" if i==0 else None)
            ax.legend()
        ax.set_title(f"Seed variability — EF@1% by method (dim={d})")
        ax.set_ylabel("EF@1%")
        ax.set_xlabel("Method")
        fig.tight_layout()
        fig.savefig(out_dir / f"seed_variability_dim{d}.png", dpi=300)
        fig.savefig(out_dir / f"seed_variability_dim{d}.pdf")
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


def _plot_distance_distributions(df_dist: pd.DataFrame, out_dir: Path, logger: logging.Logger) -> None:
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    if df_dist.empty:
        logger.info("No distance samples found; skipping distance distribution plots")
        return
    # Histogram
    fig, ax = plt.subplots(figsize=(7, 4))
    for src, color in [("actives", "tab:green"), ("zinc", "tab:blue")]:
        dd = df_dist[df_dist["source"] == src]["distance"].dropna().to_numpy(dtype=float)
        if dd.size == 0:
            continue
        ax.hist(dd, bins=100, density=True, alpha=0.5, label=src, color=color)
    ax.set_title("Distance distributions (actives vs zinc)")
    ax.set_xlabel("min-distance to MF cloud")
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "distance_hist.png", dpi=300)
    fig.savefig(plots_dir / "distance_hist.pdf")
    plt.close(fig)

    # CDF
    fig, ax = plt.subplots(figsize=(7, 4))
    for src, color in [("actives", "tab:green"), ("zinc", "tab:blue")]:
        dd = df_dist[df_dist["source"] == src]["distance"].dropna().to_numpy(dtype=float)
        if dd.size == 0:
            continue
        dd_sorted = np.sort(dd)
        y = np.linspace(0, 1, len(dd_sorted), endpoint=True)
        ax.plot(dd_sorted, y, label=src, color=color)
    ax.set_title("Distance CDF (actives vs zinc)")
    ax.set_xlabel("min-distance to MF cloud")
    ax.set_ylabel("CDF")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "distance_cdf.png", dpi=300)
    fig.savefig(plots_dir / "distance_cdf.pdf")
    plt.close(fig)
    logger.info("Saved distance hist/CDF plots")


def main():
    ap = argparse.ArgumentParser(description="Phase 1 Post Analysis (v4 molfuse)")
    ap.add_argument("--workspace_dir", type=str, required=True, help="Path to experiment_workspace_v4")
    ap.add_argument("--phase", type=str, default="phase1", help="Phase subdirectory (default: phase1)")
    ap.add_argument("--output_dir", type=str, default="reporting/phase1_post_analysis", help="Output directory for summaries and plots")
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
    plot_bars(df_grouped, plot_dir)
    plot_umap_heatmaps(df_grouped, plot_dir)
    plot_seed_variability(df_runs, plot_dir)
    logger.info("Finished plots (bars/heatmaps/seed variability)")

    # 4b) Distance distributions (actives vs zinc)
    logger.info("START: distance distributions")
    df_dist = _collect_distance_samples(workspace_dir, phase=args.phase)
    _plot_distance_distributions(df_dist, out_dir, logger)

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
