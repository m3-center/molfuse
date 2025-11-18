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
      --metrics ef1

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
from scipy.spatial.distance import cdist

# Add molfuse metrics
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from molfuse.metrics.metrics import bedroc, ief

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


def _compute_bedroc_ief_from_ranked_scores(ranked_scores_path: Path, alpha_vals: List[float]) -> Dict[str, float]:
    """
    Retrospectively compute BEDROC and IEF from ranked_scores.csv.
    
    Args:
        ranked_scores_path: Path to artifacts/ranked_scores.csv
        alpha_vals: List of alpha values to compute (e.g., [20.0, 160.9])
    
    Returns:
        Dict with keys like 'bedroc_20', 'bedroc_160', 'ief_20', 'ief_160'
    """
    if not ranked_scores_path.exists():
        return {}
    
    try:
        df_scores = pd.read_csv(ranked_scores_path)
        if 'score' not in df_scores.columns or 'label' not in df_scores.columns:
            return {}
        
        labels = df_scores['label'].to_numpy()
        scores = df_scores['score'].to_numpy()
        
        result = {}
        for alpha in alpha_vals:
            bedroc_val = bedroc(labels, scores, alpha=alpha)
            ief_val = ief(labels, scores, alpha=alpha)
            
            # Format alpha for key (remove .0 if integer)
            alpha_key = f"{int(alpha)}" if alpha == int(alpha) else f"{alpha:.1f}".replace('.', '_')
            result[f'bedroc_{alpha_key}'] = bedroc_val
            result[f'ief_{alpha_key}'] = ief_val
        
        return result
    except Exception:
        return {}


def scan_runs(workspace_dir: Path, phase: str = "phase1", alpha_vals: Optional[List[float]] = None) -> pd.DataFrame:
    if alpha_vals is None:
        alpha_vals = [20.0, 160.9]  # Default: standard and aggressive early recognition
    
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

        # Compute BEDROC/IEF retrospectively from ranked_scores.csv
        ranked_scores_path = run_dir / "artifacts" / "ranked_scores.csv"
        bedroc_ief_metrics = _compute_bedroc_ief_from_ranked_scores(ranked_scores_path, alpha_vals)

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
        # Add BEDROC/IEF metrics
        row.update(bedroc_ief_metrics)
        
        rows.append(row)

    if not rows:
        base_cols = [
            "run_name","representation","method","dim","target","affinity_cutoff_nM",
            "umap_n_neighbors","umap_min_dist","umap_metric","replicate",
            "ef_1%","ef_5%","ef_10%","roc_auc","pr_auc","spearman_rho","spearman_p",
            "n_actives","n_zinc_eval","n_mf_for_scoring"
        ]
        # Add BEDROC/IEF columns
        for alpha in alpha_vals:
            alpha_key = f"{int(alpha)}" if alpha == int(alpha) else f"{alpha:.1f}".replace('.', '_')
            base_cols.extend([f'bedroc_{alpha_key}', f'ief_{alpha_key}'])
        return pd.DataFrame(columns=base_cols)

    return pd.DataFrame(rows)


def group_and_best(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    # Group by knobs that define a config; treat replicate as a replicate
    group_keys = ["representation", "method", "dim", "umap_n_neighbors", "umap_min_dist", "umap_metric"]
    
    # Build aggregation dict dynamically to include all BEDROC/IEF columns
    agg_dict = {
        "ef_1%": ["mean", "std"],
        "ef_5%": ["mean", "std"],
        "ef_10%": ["mean", "std"],
        "roc_auc": ["mean", "std"],
        "pr_auc": ["mean", "std"],
        "run_name": "count",
    }
    
    # Add BEDROC/IEF columns if they exist
    for col in df.columns:
        if col.startswith("bedroc_") or col.startswith("ief_"):
            agg_dict[col] = ["mean", "std"]
    
    # Important: include rows with NaNs in UMAP-only keys (so PCA isn't dropped)
    g = df.groupby(group_keys, dropna=False).agg(agg_dict).reset_index()
    
    # Flatten MultiIndex columns
    g.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in g.columns]
    
    # Rename for backwards compat
    rename_map = {
        "ef_1%_mean": "ef1_mean",
        "ef_1%_std": "ef1_std",
        "ef_5%_mean": "ef5_mean",
        "ef_10%_mean": "ef10_mean",
        "roc_auc_mean": "roc_mean",
        "pr_auc_mean": "pr_mean",
        "run_name_count": "n_runs",
    }
    g.rename(columns=rename_map, inplace=True)

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
                        # For UMAP (avg): average across all hyperparameter configurations at this dim
                        # For UMAP (best) and PCA: use single best config
                        if m == "umap" and len(row) > 1:
                            # Average across all hyperparameter configs (config-level mean)
                            y_val = float(row[mean_col].mean())
                            # Error bar represents variability across configs (not replicates)
                            yerr_val = float(row[mean_col].std()) if len(row) > 1 else 0.0
                        else:
                            # Single config: use its mean and std across replicates
                            y_val = float(row.iloc[0][mean_col])
                            yerr_val = float(row.iloc[0][std_col]) if std_col in row.columns else 0.0
                        y.append(y_val)
                        yerr.append(yerr_val)
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


def plot_bedroc_ief_bars(
    df_g: pd.DataFrame,
    out_dir: Path,
    alpha_vals: List[float],
    sharey: bool,
    logger: logging.Logger,
) -> List[Path]:
    """
    Create bar plots for BEDROC and IEF metrics across dimensions.
    
    Similar to plot_bars_combined but for BEDROC/IEF metrics.
    """
    _ensure_dir(out_dir)
    saved: List[Path] = []
    
    dims = sorted(df_g["dim"].dropna().unique().astype(int).tolist())
    methods = sorted(df_g["method"].dropna().unique().tolist())
    reps = sorted(df_g["representation"].dropna().unique().tolist())
    colors = _bar_colors()
    
    # Plot for each alpha and each metric type (BEDROC/IEF)
    for alpha in alpha_vals:
        alpha_key = f"{int(alpha)}" if alpha == int(alpha) else f"{alpha:.1f}".replace('.', '_')
        
        for metric_type in ["bedroc", "ief"]:
            mean_col = f"{metric_type}_{alpha_key}_mean"
            std_col = f"{metric_type}_{alpha_key}_std"
            
            # Check if columns exist
            if mean_col not in df_g.columns:
                continue
            
            if sharey:
                y0, y1 = _compute_global_ylim(df_g, mean_col, std_col)
                logger.info(f"Bars[{metric_type}(α={alpha})] global ylim: ({y0:.3f}, {y1:.3f})")
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
                        if row.empty or mean_col not in row.columns:
                            y.append(np.nan)
                            yerr.append(0.0)
                        else:
                            y_val = float(row.iloc[0][mean_col])
                            yerr_val = float(row.iloc[0][std_col]) if std_col in row.columns else 0.0
                            y.append(y_val)
                            yerr.append(yerr_val)
                    
                    xpos = x + offset0 + i*bw
                    bar_colors_list = [colors.get(r, None) for _ in methods]
                    bars = ax.bar(xpos, y, width=bw, label=r, color=bar_colors_list, yerr=yerr, capsize=3)
                    
                    # Numeric labels
                    for b, val in zip(bars, y):
                        if np.isfinite(val):
                            y_upper = (y1 if sharey and y1 is not None else max(1.0, b.get_height()))
                            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02*float(y_upper),
                                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)
                
                ax.set_xticks(x)
                ax.set_xticklabels([m.upper() if m == "pca" else m.capitalize() for m in methods])
                ax.set_xlabel("Method")
                if sharey:
                    ax.set_ylim(y0, y1)
                ax.set_title(f"dim={d}")
            
            # Y-axis label
            metric_label = f"{metric_type.upper()}(α={alpha})" if metric_type == "bedroc" else f"IEF(α={alpha})"
            axes[0].set_ylabel(metric_label)
            
            # Legend
            handles, labels_leg = axes[-1].get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels_leg, title="Representation", loc="center left", 
                          bbox_to_anchor=(1.0, 0.5), frameon=True, fontsize=10)
            
            fig.suptitle(f"{metric_label} (mean ± sd)", fontsize=14, fontweight='bold')
            fig.tight_layout(rect=(0, 0, 0.88, 0.96))
            
            p_png = out_dir / f"bars_{metric_type}_alpha{alpha_key}_combined.png"
            p_pdf = out_dir / f"bars_{metric_type}_alpha{alpha_key}_combined.pdf"
            fig.savefig(p_png, bbox_inches='tight', dpi=300)
            fig.savefig(p_pdf, bbox_inches='tight')
            plt.close(fig)
            saved.extend([p_png, p_pdf])
            logger.info(f"Saved {metric_label} bars: {p_png}")
    
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


def plot_4metric_comparison(
    df_grouped: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger,
    sharey: bool = False
) -> List[Path]:
    """
    Create 2×2 multi-metric panel comparing 4 methods on core metrics.
    
    Metrics (primary → secondary):
    - EF@1% (early enrichment, primary)
    - BEDROC α=20 (robust early enrichment, primary)
    - ROC-AUC (overall discrimination, secondary)
    - PR-AUC (precision-recall, secondary)
    
    Returns:
        List of saved file paths
    """
    logger.info("Generating 4-metric comparison panel...")
    
    _ensure_dir(output_dir)
    
    # Define methods and colors
    methods = [
        ("pca", "features"),
        ("pca", "fingerprints"),
        ("umap", "features"),
        ("umap", "fingerprints")
    ]
    
    colors = {
        ("pca", "features"): "#2E86AB",
        ("pca", "fingerprints"): "#A23B72",
        ("umap", "features"): "#F18F01",
        ("umap", "fingerprints"): "#06A77D"
    }
    
    labels = {
        ("pca", "features"): "PCA/Features",
        ("pca", "fingerprints"): "PCA/Fingerprints",
        ("umap", "features"): "UMAP/Features",
        ("umap", "fingerprints"): "UMAP/Fingerprints"
    }
    
    # Metrics configuration: (column_mean, column_std, ylabel, title)
    metric_configs = [
        ("ef1_mean", "ef1_std", "EF@1%", "Early Enrichment (EF@1%)"),
        ("bedroc_20_mean", "bedroc_20_std", "BEDROC (α=20)", "Robust Early Enrichment (BEDROC α=20)"),
        ("roc_mean", "roc_std", "ROC-AUC", "Overall Discrimination (ROC-AUC)"),
        ("pr_mean", "pr_std", "PR-AUC", "Precision-Recall (PR-AUC)")
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (mean_col, std_col, ylabel, title) in enumerate(metric_configs):
        ax = axes[idx]
        
        # Extract data for each method
        x_pos = np.arange(len(methods))
        means = []
        stds = []
        
        for method, representation in methods:
            subset = df_grouped[
                (df_grouped["method"] == method) &
                (df_grouped["representation"] == representation)
            ]
            
            if not subset.empty and mean_col in subset.columns:
                mean_val = float(subset[mean_col].iloc[0])
                std_val = float(subset[std_col].iloc[0]) if std_col in subset.columns else 0.0
            else:
                mean_val = 0.0
                std_val = 0.0
            
            means.append(mean_val)
            stds.append(std_val)
        
        # Plot bars
        bars = ax.bar(
            x_pos,
            means,
            yerr=stds,
            capsize=5,
            color=[colors[m] for m in methods],
            edgecolor='black',
            linewidth=1.5,
            alpha=0.8,
            error_kw={'linewidth': 2, 'ecolor': 'black'}
        )
        
        # Formatting
        ax.set_xticks(x_pos)
        ax.set_xticklabels([labels[m] for m in methods], rotation=45, ha='right', fontsize=10)
        ax.set_ylabel(ylabel, fontweight='bold', fontsize=11)
        ax.set_title(title, fontweight='bold', fontsize=12)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for i, (bar, mean_val) in enumerate(zip(bars, means)):
            if mean_val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + stds[i],
                    f'{mean_val:.2f}',
                    ha='center',
                    va='bottom',
                    fontsize=9,
                    fontweight='bold'
                )
        
        if sharey and idx > 0:
            # Share y-axis limits with first subplot
            axes[0].get_shared_y_axes().join(axes[0], ax)
    
    plt.tight_layout()
    
    # Save outputs
    output_png = output_dir / "phase1_4metric_comparison.png"
    output_pdf = output_dir / "phase1_4metric_comparison.pdf"
    fig.savefig(output_png, dpi=300, bbox_inches='tight')
    fig.savefig(output_pdf, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Saved 4-metric comparison: {output_png.name}")
    return [output_png, output_pdf]


def plot_4metric_stratified_tiers(
    df_stratified: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger
) -> List[Path]:
    """
    Create tier-stratified 4-metric comparison (High/Medium/Weak potency tiers).
    
    Generates 4 separate plots (one per metric), each showing 3 tiers.
    
    Args:
        df_stratified: DataFrame from phase1_stratified_scores.py (stratified_grouped.csv)
        output_dir: Output directory
        logger: Logger instance
    
    Returns:
        List of saved file paths
    """
    logger.info("Generating tier-stratified 4-metric comparisons...")
    
    _ensure_dir(output_dir)
    
    # Check required columns
    required_cols = ["ef_1%_high_mean", "ef_1%_medium_mean", "ef_1%_weak_mean",
                     "bedroc_20_high_mean", "bedroc_20_medium_mean", "bedroc_20_weak_mean",
                     "roc_high_mean", "roc_medium_mean", "roc_weak_mean",
                     "pr_high_mean", "pr_medium_mean", "pr_weak_mean"]
    
    missing_cols = [c for c in required_cols if c not in df_stratified.columns]
    if missing_cols:
        logger.warning(f"Missing columns for tier stratification: {missing_cols}")
        logger.warning("Skipping tier-stratified 4-metric plots")
        return []
    
    methods = [
        ("pca", "features"),
        ("pca", "fingerprints"),
        ("umap", "features"),
        ("umap", "fingerprints")
    ]
    
    colors = {
        "High": "#06A77D",
        "Medium": "#F18F01",
        "Weak": "#A23B72"
    }
    
    labels = {
        ("pca", "features"): "PCA/Feat",
        ("pca", "fingerprints"): "PCA/FP",
        ("umap", "features"): "UMAP/Feat",
        ("umap", "fingerprints"): "UMAP/FP"
    }
    
    # Metrics: (metric_base, ylabel, title)
    metrics_config = [
        ("ef_1%", "EF@1%", "Tier-Stratified Early Enrichment (EF@1%)"),
        ("bedroc_20", "BEDROC (α=20)", "Tier-Stratified BEDROC (α=20)"),
        ("roc", "ROC-AUC", "Tier-Stratified ROC-AUC"),
        ("pr", "PR-AUC", "Tier-Stratified PR-AUC")
    ]
    
    saved_files = []
    
    for metric_base, ylabel, title in metrics_config:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x_pos = np.arange(len(methods))
        width = 0.25
        
        # Extract data for each tier
        for tier_idx, tier in enumerate(["High", "Medium", "Weak"]):
            tier_col = f"{metric_base}_{tier.lower()}_mean"
            tier_std_col = f"{metric_base}_{tier.lower()}_std"
            
            means = []
            stds = []
            
            for method, representation in methods:
                subset = df_stratified[
                    (df_stratified["method"] == method) &
                    (df_stratified["representation"] == representation)
                ]
                
                if not subset.empty and tier_col in subset.columns:
                    mean_val = float(subset[tier_col].iloc[0])
                    std_val = float(subset[tier_std_col].iloc[0]) if tier_std_col in subset.columns else 0.0
                else:
                    mean_val = 0.0
                    std_val = 0.0
                
                means.append(mean_val)
                stds.append(std_val)
            
            # Plot grouped bars
            offset = (tier_idx - 1) * width
            ax.bar(
                x_pos + offset,
                means,
                width,
                yerr=stds,
                label=f"{tier} (≤{'100' if tier == 'High' else '1K' if tier == 'Medium' else '100K'} nM)",
                color=colors[tier],
                edgecolor='black',
                linewidth=1.2,
                alpha=0.8,
                capsize=4,
                error_kw={'linewidth': 1.5, 'ecolor': 'black'}
            )
        
        # Formatting
        ax.set_xticks(x_pos)
        ax.set_xticklabels([labels[m] for m in methods], fontsize=11)
        ax.set_ylabel(ylabel, fontweight='bold', fontsize=12)
        ax.set_title(title, fontweight='bold', fontsize=13)
        ax.legend(loc='best', fontsize=10, framealpha=0.95)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        # Save
        metric_name = metric_base.replace("_", "").replace("%", "pct")
        output_png = output_dir / f"phase1_tierstratified_{metric_name}.png"
        output_pdf = output_dir / f"phase1_tierstratified_{metric_name}.pdf"
        fig.savefig(output_png, dpi=300, bbox_inches='tight')
        fig.savefig(output_pdf, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Saved tier-stratified {metric_base}: {output_png.name}")
        saved_files.extend([output_png, output_pdf])
    
    return saved_files


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
        
        # Store in config for later use in axis range computation
        config["_data"] = (df_mf, df_zinc, df_actives)
    
    # Plot with independent axis ranges per subplot (better visibility)
    # No global axis range computation needed
    for idx, config in enumerate(configs_to_plot):
        ax = axes[idx]
        
        if "_data" not in config:
            continue
        
        df_mf, df_zinc, df_actives = config["_data"]
        run_row = df_runs[
            (df_runs["method"] == config["method"]) &
            (df_runs["representation"] == config["representation"]) &
            (df_runs["dim"] == config["dim"])
        ].iloc[0]
        
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
        
        # Compute independent axis ranges for this subplot (better visibility)
        x_min = min(mf_x.min(), zinc_x.min(), actives_x.min())
        x_max = max(mf_x.max(), zinc_x.max(), actives_x.max())
        y_min = min(mf_y.min(), zinc_y.min(), actives_y.min())
        y_max = max(mf_y.max(), zinc_y.max(), actives_y.max())
        
        # Add 5% padding
        x_range = x_max - x_min
        y_range = y_max - y_min
        ax.set_xlim(x_min - 0.05 * x_range, x_max + 0.05 * x_range)
        ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)
        
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
    
    # ===== Contour Overlays (separate groups) =====
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
        
        # Create 1x4 figure showing each group separately + combined
        fig_cont, axes_cont = plt.subplots(1, 4, figsize=(20, 5))
        
        # Compute global axis ranges for this configuration
        x_min = min(df_mf["z0"].min(), df_zinc["z0"].min(), df_actives["z0"].min())
        x_max = max(df_mf["z0"].max(), df_zinc["z0"].max(), df_actives["z0"].max())
        y_min = min(df_mf["z1"].min(), df_zinc["z1"].min(), df_actives["z1"].min())
        y_max = max(df_mf["z1"].max(), df_zinc["z1"].max(), df_actives["z1"].max())
        
        # Add 5% padding
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_min -= 0.05 * x_range
        x_max += 0.05 * x_range
        y_min -= 0.05 * y_range
        y_max += 0.05 * y_range
        
        groups = [
            ("MF cloud", df_mf, "#888888", 0),
            ("ACTIVES", df_actives, "#E85D2D", 1),
            ("ZINC", df_zinc, "#5A7FC0", 2),
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
                    
                    # Create grid using global ranges
                    xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
                    positions = np.vstack([xx.ravel(), yy.ravel()])
                    density = kde(positions).reshape(xx.shape)
                    
                    ax.contourf(xx, yy, density, levels=10, cmap='Greys', alpha=0.6)
                    ax.contour(xx, yy, density, levels=5, colors=color, linewidths=1.5, alpha=0.8)
                except Exception:
                    # Fallback to scatter
                    ax.scatter(x, y, c=color, s=5, alpha=0.6)
            
            # Set shared axis ranges
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            
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
        
        # Set shared axis ranges
        ax_combined.set_xlim(x_min, x_max)
        ax_combined.set_ylim(y_min, y_max)
        
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


def _plot_molecular_similarity_chains(
    workspace_dir: Path,
    phase: str,
    df_runs: pd.DataFrame,
    df_grouped: pd.DataFrame,
    out_dir: Path,
    logger: logging.Logger,
) -> List[Path]:
    """Visualize molecular similarity chains for closest/furthest ACTIVES.
    
    For each similarity space method:
    1. Find 3 closest and 3 furthest ACTIVES to MF cloud
    2. For each ACTIVE, find nearest MF molecule
    3. For that MF molecule, find nearest ZINC molecule
    4. Visualize the triplets (ACTIVE -> MF -> ZINC) as 2D molecular structures
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
    except ImportError:
        logger.error("RDKit not available - skipping molecular structure visualization")
        return []
    
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Define the four configurations to analyze
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
        logger.warning("No 2D configurations found for molecular chain visualization")
        return []
    
    saved_files = []
    phase_dir = workspace_dir / phase
    
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
            logger.warning(f"No matching run for {config['label']}")
            continue
        
        run_row = matching_runs.iloc[0]
        run_dir = phase_dir / run_row["run_name"]
        artifacts_dir = run_dir / "artifacts"
        
        # Load embeddings with SMILES
        try:
            df_mf = pd.read_csv(artifacts_dir / "embedding_mf.csv")
            df_zinc = pd.read_csv(artifacts_dir / "embedding_zinc.csv")
            df_actives = pd.read_csv(artifacts_dir / "embedding_actives.csv")
        except Exception as e:
            logger.warning(f"Failed to load embeddings for {config['label']}: {e}")
            continue
        
        # Verify SMILES column exists
        if "SMILES" not in df_actives.columns:
            logger.warning(f"No SMILES column in ACTIVES data for {config['label']}")
            continue
        
        # Compute distances from each ACTIVE to MF cloud
        actives_coords = df_actives[["z0", "z1"]].values
        mf_coords = df_mf[["z0", "z1"]].values
        
        # For each ACTIVE, find min distance to any MF molecule
        distances_to_mf = cdist(actives_coords, mf_coords, metric='euclidean')
        min_distances = distances_to_mf.min(axis=1)
        
        # Find 3 closest and 3 furthest ACTIVES
        closest_indices = np.argsort(min_distances)[:3]
        furthest_indices = np.argsort(min_distances)[-3:]
        
        selected_indices = list(closest_indices) + list(furthest_indices)
        
        # For each selected ACTIVE, find the chain: ACTIVE -> nearest MF -> nearest ZINC
        triplets = []
        for idx in selected_indices:
            active_smiles = df_actives.iloc[idx]["SMILES"]
            active_coord = actives_coords[idx]
            active_dist = min_distances[idx]
            
            # Find nearest MF
            nearest_mf_idx = distances_to_mf[idx].argmin()
            mf_smiles = df_mf.iloc[nearest_mf_idx]["SMILES"]
            mf_coord = mf_coords[nearest_mf_idx]
            
            # Find nearest ZINC to that MF
            zinc_coords = df_zinc[["z0", "z1"]].values
            distances_mf_to_zinc = cdist([mf_coord], zinc_coords, metric='euclidean')[0]
            nearest_zinc_idx = distances_mf_to_zinc.argmin()
            zinc_smiles = df_zinc.iloc[nearest_zinc_idx]["SMILES"]
            
            triplets.append({
                "active_smiles": active_smiles,
                "mf_smiles": mf_smiles,
                "zinc_smiles": zinc_smiles,
                "distance_to_mf": active_dist,
                "is_close": idx in closest_indices
            })
        
        # Create figure: 6 rows x 3 columns (ACTIVE, MF, ZINC)
        fig = plt.figure(figsize=(15, 24))
        gs = fig.add_gridspec(6, 3, hspace=0.4, wspace=0.3)
        
        for row_idx, triplet in enumerate(triplets):
            # Determine if this is a "close" or "far" case
            case_type = "CLOSE" if triplet["is_close"] else "FAR"
            dist_val = triplet["distance_to_mf"]
            
            # Draw ACTIVE
            ax_active = fig.add_subplot(gs[row_idx, 0])
            mol_active = Chem.MolFromSmiles(triplet["active_smiles"])
            if mol_active:
                img_active = Draw.MolToImage(mol_active, size=(400, 400))
                ax_active.imshow(img_active)
            ax_active.axis('off')
            smiles_short = triplet["active_smiles"][:50] + "..." if len(triplet["active_smiles"]) > 50 else triplet["active_smiles"]
            ax_active.set_title(f"ACTIVE ({case_type})\nd={dist_val:.3f}\n{smiles_short}", 
                               fontsize=8, fontweight='bold')
            
            # Draw MF
            ax_mf = fig.add_subplot(gs[row_idx, 1])
            mol_mf = Chem.MolFromSmiles(triplet["mf_smiles"])
            if mol_mf:
                img_mf = Draw.MolToImage(mol_mf, size=(400, 400))
                ax_mf.imshow(img_mf)
            ax_mf.axis('off')
            smiles_short = triplet["mf_smiles"][:50] + "..." if len(triplet["mf_smiles"]) > 50 else triplet["mf_smiles"]
            ax_mf.set_title(f"MF (nearest)\n{smiles_short}", fontsize=8)
            
            # Draw ZINC
            ax_zinc = fig.add_subplot(gs[row_idx, 2])
            mol_zinc = Chem.MolFromSmiles(triplet["zinc_smiles"])
            if mol_zinc:
                img_zinc = Draw.MolToImage(mol_zinc, size=(400, 400))
                ax_zinc.imshow(img_zinc)
            ax_zinc.axis('off')
            smiles_short = triplet["zinc_smiles"][:50] + "..." if len(triplet["zinc_smiles"]) > 50 else triplet["zinc_smiles"]
            ax_zinc.set_title(f"ZINC (nearest to MF)\n{smiles_short}", fontsize=8)
        
        # Add overall title
        fig.suptitle(f"{config['label']} - Molecular Similarity Chains\nTop 3 rows: Closest ACTIVES | Bottom 3 rows: Furthest ACTIVES", 
                    fontsize=14, fontweight='bold')
        
        # Save
        safe_label = config['label'].replace(' ', '_').lower()
        p_png = plots_dir / f"molecular_chains_{safe_label}.png"
        p_pdf = plots_dir / f"molecular_chains_{safe_label}.pdf"
        fig.savefig(p_png, dpi=300, bbox_inches='tight')
        fig.savefig(p_pdf, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Saved molecular chain visualization: {p_png}")
        saved_files.extend([p_png, p_pdf])
    
    return saved_files


def main():
    ap = argparse.ArgumentParser(description="Phase 1 Post Analysis (v4 molfuse)")
    ap.add_argument("--workspace_dir", type=str, required=True, help="Path to experiment_workspace_v4")
    ap.add_argument("--phase", type=str, default="phase1", help="Phase subdirectory (default: phase1)")
    ap.add_argument("--output_dir", type=str, default="reporting/phase1_post_analysis", help="Output directory for summaries and plots")
    ap.add_argument("--metrics", type=str, default="ef1", help="Comma-separated metrics to plot as bars (default: ef1 only)")
    ap.add_argument("--no-sharey", dest="sharey", action="store_false", help="Disable shared y-axis for combined bar plots")
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
    
    # BEDROC/IEF bar plots
    alpha_vals = [20.0, 160.9]  # Standard and aggressive early recognition
    saved_bedroc_ief = plot_bedroc_ief_bars(df_grouped, plot_dir, alpha_vals, getattr(args, "sharey", True), logger)
    manifest["bedroc_ief_bars"] = [str(p) for p in saved_bedroc_ief]
    
    # NEW: 4-metric comparison panel (EF@1%, BEDROC-20, ROC-AUC, PR-AUC)
    saved_4metric = plot_4metric_comparison(df_grouped, plot_dir, logger, sharey=getattr(args, "sharey", False))
    manifest["4metric_comparison"] = [str(p) for p in saved_4metric]
    
    # NEW: Tier-stratified 4-metric plots (if stratified data available)
    stratified_grouped_csv = out_dir / "stratified_grouped.csv"
    if stratified_grouped_csv.exists():
        logger.info(f"Found stratified data: {stratified_grouped_csv}")
        df_stratified = pd.read_csv(stratified_grouped_csv)
        saved_tier_4metric = plot_4metric_stratified_tiers(df_stratified, plot_dir, logger)
        manifest["4metric_tierstratified"] = [str(p) for p in saved_tier_4metric]
    else:
        logger.warning(f"Stratified data not found: {stratified_grouped_csv}")
        logger.warning("Skipping tier-stratified 4-metric plots. Run phase1_stratified_scores.py first.")
    
    plot_umap_heatmaps(df_grouped, plot_dir)
    manifest.setdefault("heatmaps", []).extend([str(plot_dir/"umap_heatmap_grid.png"), str(plot_dir/"umap_heatmap_grid.pdf")])
    logger.info("Finished plots (bars/heatmaps)")

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
    
    # 4e) Molecular similarity chain visualization
    logger.info("START: Molecular similarity chains (ACTIVE -> MF -> ZINC)")
    chain_files = _plot_molecular_similarity_chains(workspace_dir, args.phase, df_runs, df_grouped, out_dir, logger)
    if chain_files:
        manifest.setdefault("molecular_chains", []).extend([str(p) for p in chain_files])
    logger.info("Finished molecular chain visualizations")

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
