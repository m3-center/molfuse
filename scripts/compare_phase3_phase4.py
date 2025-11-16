#!/usr/bin/env python3
"""
Compare Phase 3 (MF Ablation on P00519) vs Phase 4 (Cross-Target Generalization)

Creates overlay plots showing:
1. Phase 3: MF size vs EF@1% for P00519 (within-target subsampling)
2. Phase 4: MF size vs EF@1% across 8 protein function categories

This validates whether the Phase 3 pattern (larger MF → higher EF) 
generalizes across different protein function categories.Usage:
    python scripts/compare_phase3_phase4.py \
        --phase3_aggregated reporting/phase3_post_analysis/phase3_summary_aggregated.csv \
        --phase3_stratified reporting/phase3_post_analysis/phase3_summary_stratified.csv \
        --phase4_aggregated reporting/phase4_post_analysis/phase4_summary_aggregated.csv \
        --phase4_stratified reporting/phase4_post_analysis/phase4_summary_stratified.csv \
        --output_dir reporting/phase3_phase4_comparison
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Publication-quality style
matplotlib.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "axes.titleweight": "bold",
    "legend.fontsize": 10,
    "font.family": "sans-serif",
})


def plot_phase3_phase4_overlay_overall(
    df_phase3: pd.DataFrame,
    df_phase4: pd.DataFrame,
    output_dir: Path
) -> None:
    """
    Overlay Phase 3 (P00519 MF ablation) and Phase 4 (cross-target) results.
    
    Focus on UMAP/features, Overall EF@1%.
    """
    print("\nGenerating Phase 3 vs Phase 4 overlay (Overall EF@1%)...")
    
    # Filter for UMAP/features
    p3_umap_feat = df_phase3[
        (df_phase3["method"] == "umap") & 
        (df_phase3["representation"] == "features")
    ].copy()
    
    p4_umap_feat = df_phase4[
        (df_phase4["method"] == "umap") & 
        (df_phase4["representation"] == "features")
    ].copy()
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Phase 3: MF ablation curve (line with error bars)
    # Note: Phase 3 uses mf_size_target_numeric (not mf_size_mean)
    p3_sorted = p3_umap_feat.sort_values("mf_size_target_numeric")
    
    # Phase 3 already aggregated, just use directly
    p3_x = np.array(p3_sorted["mf_size_target_numeric"].values, dtype=float)
    p3_y_mean = np.array(p3_sorted["ef_1%_mean"].values, dtype=float)
    p3_y_sem = np.array(p3_sorted["ef_1%_sem"].values, dtype=float)
    
    # Remove NaN
    valid_mask = ~np.isnan(p3_x) & ~np.isnan(p3_y_mean)
    p3_x = p3_x[valid_mask]
    p3_y_mean = p3_y_mean[valid_mask]
    p3_y_sem = p3_y_sem[valid_mask]
    
    # Plot Phase 3 curve
    ax.plot(p3_x, p3_y_mean, 'o-', color='#2E86AB', linewidth=2.5, markersize=8,
            label='Phase 3: P00519 (MF ablation)', zorder=3)
    ax.fill_between(p3_x, p3_y_mean - p3_y_sem, p3_y_mean + p3_y_sem,
                     color='#2E86AB', alpha=0.2, zorder=2)
    
    # Phase 4: Cross-target scatter
    p4_x = p4_umap_feat["natural_mf_size"].values
    p4_y = p4_umap_feat["ef_1%_mean"].values
    
    # Remove NaN
    valid_mask = ~np.isnan(p4_x) & ~np.isnan(p4_y)
    p4_x_valid = p4_x[valid_mask]
    p4_y_valid = p4_y[valid_mask]
    
    # Plot Phase 4 points with labels
    scatter = ax.scatter(p4_x_valid, p4_y_valid, s=150, color='#A23B72', alpha=0.7,
                        edgecolors='black', linewidth=1.5, label='Phase 4: 8 protein functions',
                        zorder=4)
    
    # Add target labels
    for i, (x, y, target) in enumerate(zip(p4_x_valid, p4_y_valid, p4_agg.loc[valid_mask, "target"])):
        # Clean up target name for display
        label = target.replace('_', ' ')
        ax.annotate(label, (x, y), fontsize=8, ha='left', va='bottom',
                   xytext=(5, 5), textcoords='offset points', alpha=0.8)
    
    # Add Phase 4 trendline
    if len(p4_x_valid) >= 3:
        rho, p_value = stats.spearmanr(p4_x_valid, p4_y_valid)
        z = np.polyfit(np.log10(p4_x_valid), p4_y_valid, 1)
        p = np.poly1d(z)
        
        # Extend trendline across overlapping range
        x_min = max(p3_x.min(), p4_x_valid.min())
        x_max = min(p3_x.max(), p4_x_valid.max())
        x_fit = np.logspace(np.log10(x_min), np.log10(x_max), 100)
        y_fit = p(np.log10(x_fit))
        
        ax.plot(x_fit, y_fit, '--', color='#A23B72', alpha=0.8, linewidth=2,
                label=f'Phase 4 trend (ρ={rho:.3f}, p={p_value:.3f})', zorder=3)
    
    ax.set_xscale("log")
    ax.set_xlabel("Natural MF Cloud Size (compounds)", fontweight="bold", fontsize=13)
    ax.set_ylabel("Overall EF@1% (Mean ± SEM)", fontweight="bold", fontsize=13)
    ax.set_title("Phase 3 vs Phase 4: MF Cloud Size Effect on Enrichment", 
                 fontweight="bold", fontsize=15)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=11, framealpha=0.95)
    
    plt.tight_layout()
    
    output_path_png = output_dir / "phase3_phase4_overlay_overall_ef1.png"
    output_path_pdf = output_dir / "phase3_phase4_overlay_overall_ef1.pdf"
    fig.savefig(output_path_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_path_pdf, bbox_inches="tight")
    plt.close(fig)
    
    print(f"Saved: {output_path_png.name}")


def plot_phase3_phase4_overlay_high_potency(
    df_phase3_strat: pd.DataFrame,
    df_phase4_strat: pd.DataFrame,
    output_dir: Path
) -> None:
    """
    Overlay Phase 3 and Phase 4 high-potency EF@1% curves.
    """
    print("\nGenerating Phase 3 vs Phase 4 overlay (High-Potency EF@1%)...")
    
    # Filter for UMAP/features
    p3_umap_feat = df_phase3_strat[
        (df_phase3_strat["method"] == "umap") & 
        (df_phase3_strat["representation"] == "features")
    ].copy()
    
    p4_umap_feat = df_phase4_strat[
        (df_phase4_strat["method"] == "umap") & 
        (df_phase4_strat["representation"] == "features")
    ].copy()
    
    # Aggregate Phase 4 by target
    p4_agg = p4_umap_feat.groupby(["target", "natural_mf_size"], dropna=False).agg({
        "ef_1%_high": "mean"
    }).reset_index()
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Phase 3: High-potency curve
    # Phase 3 stratified has individual runs, need to aggregate by mf_size_target
    # Convert mf_size_target (categorical string) to numeric first
    p3_umap_feat["mf_size_numeric"] = pd.to_numeric(p3_umap_feat["mf_size_target"], errors="coerce")
    
    p3_by_size = p3_umap_feat.groupby("mf_size_numeric", dropna=False).agg({
        "ef_1%_high": ["mean", "sem"]
    }).reset_index()
    p3_by_size.columns = ["mf_size_numeric", "ef_1%_high_mean", "ef_1%_high_sem"]
    p3_by_size = p3_by_size.sort_values("mf_size_numeric")
    
    # Remove NaN and convert to numpy arrays
    p3_valid = p3_by_size.dropna(subset=["mf_size_numeric", "ef_1%_high_mean"])
    
    p3_x = np.array(p3_valid["mf_size_numeric"].values, dtype=float)
    p3_y_mean = np.array(p3_valid["ef_1%_high_mean"].values, dtype=float)
    p3_y_sem = np.array(p3_valid["ef_1%_high_sem"].values, dtype=float)
    
    # Plot Phase 3 curve
    ax.plot(p3_x, p3_y_mean, 'o-', color='#2E86AB', linewidth=2.5, markersize=8,
            label='Phase 3: P00519 high-potency (≤100 nM)', zorder=3)
    ax.fill_between(p3_x, p3_y_mean - p3_y_sem, p3_y_mean + p3_y_sem,
                     color='#2E86AB', alpha=0.2, zorder=2)
    
    # Phase 4: Cross-target scatter
    p4_x = np.array(p4_agg["natural_mf_size"].values, dtype=float)
    p4_y = np.array(p4_agg["ef_1%_high"].values, dtype=float)
    
    # Remove NaN
    valid_mask = ~np.isnan(p4_x) & ~np.isnan(p4_y)
    p4_x_valid = p4_x[valid_mask]
    p4_y_valid = p4_y[valid_mask]
    
    # Plot Phase 4 points with labels
    scatter = ax.scatter(p4_x_valid, p4_y_valid, s=150, color='#F18F01', alpha=0.7,
                        edgecolors='black', linewidth=1.5, label='Phase 4: 8 protein functions (high-potency)',
                        zorder=4)
    
    # Add target labels
    for i, (x, y, target) in enumerate(zip(p4_x_valid, p4_y_valid, p4_agg.loc[valid_mask, "target"])):
        # Clean up target name for display
        label = target.replace('_', ' ')
        ax.annotate(label, (x, y), fontsize=8, ha='left', va='bottom',
                   xytext=(5, 5), textcoords='offset points', alpha=0.8)
    
    # Add Phase 4 trendline
    if len(p4_x_valid) >= 3:
        rho, p_value = stats.spearmanr(p4_x_valid, p4_y_valid)
        z = np.polyfit(np.log10(p4_x_valid), p4_y_valid, 1)
        p = np.poly1d(z)
        
        x_min = max(p3_x.min(), p4_x_valid.min())
        x_max = min(p3_x.max(), p4_x_valid.max())
        x_fit = np.logspace(np.log10(x_min), np.log10(x_max), 100)
        y_fit = p(np.log10(x_fit))
        
        ax.plot(x_fit, y_fit, '--', color='#F18F01', alpha=0.8, linewidth=2,
                label=f'Phase 4 trend (ρ={rho:.3f}, p={p_value:.3f})', zorder=3)
    
    ax.set_xscale("log")
    ax.set_xlabel("Natural MF Cloud Size (compounds)", fontweight="bold", fontsize=13)
    ax.set_ylabel("High-Potency EF@1% (≤100 nM, Mean ± SEM)", fontweight="bold", fontsize=13)
    ax.set_title("Phase 3 vs Phase 4: High-Potency Enrichment Across MF Sizes", 
                 fontweight="bold", fontsize=15)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=11, framealpha=0.95)
    
    plt.tight_layout()
    
    output_path_png = output_dir / "phase3_phase4_overlay_high_potency_ef1.png"
    output_path_pdf = output_dir / "phase3_phase4_overlay_high_potency_ef1.pdf"
    fig.savefig(output_path_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_path_pdf, bbox_inches="tight")
    plt.close(fig)
    
    print(f"Saved: {output_path_png.name}")


def main():
    parser = argparse.ArgumentParser(description="Compare Phase 3 vs Phase 4 MF size effects")
    parser.add_argument("--phase3_aggregated", type=str, required=True,
                       help="Phase 3 aggregated summary CSV")
    parser.add_argument("--phase3_stratified", type=str, required=True,
                       help="Phase 3 stratified summary CSV")
    parser.add_argument("--phase4_aggregated", type=str, required=True,
                       help="Phase 4 aggregated summary CSV")
    parser.add_argument("--phase4_stratified", type=str, required=True,
                       help="Phase 4 stratified summary CSV")
    parser.add_argument("--output_dir", type=str, default="reporting/phase3_phase4_comparison",
                       help="Output directory for comparison plots")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("PHASE 3 vs PHASE 4 COMPARISON")
    print("="*80)
    print(f"Phase 3 aggregated: {args.phase3_aggregated}")
    print(f"Phase 3 stratified: {args.phase3_stratified}")
    print(f"Phase 4 aggregated: {args.phase4_aggregated}")
    print(f"Phase 4 stratified: {args.phase4_stratified}")
    print(f"Output: {output_dir}")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    df_p3 = pd.read_csv(args.phase3_aggregated)
    df_p3_strat = pd.read_csv(args.phase3_stratified)
    df_p4 = pd.read_csv(args.phase4_aggregated)
    df_p4_strat = pd.read_csv(args.phase4_stratified)
    
    print(f"  Phase 3 aggregated: {len(df_p3)} rows")
    print(f"  Phase 3 stratified: {len(df_p3_strat)} rows")
    print(f"  Phase 4 aggregated: {len(df_p4)} rows")
    print(f"  Phase 4 stratified: {len(df_p4_strat)} rows")
    
    # Generate overlay plots
    plot_phase3_phase4_overlay_overall(df_p3, df_p4, output_dir)
    plot_phase3_phase4_overlay_high_potency(df_p3_strat, df_p4_strat, output_dir)
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print(f"Results saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
