#!/usr/bin/env python3
"""
Diagnose Phase 3 vs Phase 4 Discrepancy for P00519/Transferase

Investigates why Phase 3 and Phase 4 show different EF@1% values for the same
target (P00519) with the same MF cloud (Transferase).

Checks:
1. Active set sizes and compositions
2. ZINC decoy set sizes and compositions
3. MF cloud sizes and overlap
4. Embedding statistics (mean, std, distributions)
5. Score distributions
6. Random seed effects

Usage:
    python scripts/diagnose_phase3_phase4_discrepancy.py \
        --phase3_workspace experiment_workspace_v4/phase3/mf_ablation \
        --phase4_workspace experiment_workspace_v4/phase4/cross_target \
        --output_dir reporting/phase3_phase4_diagnosis
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def load_phase3_run(workspace_dir: Path, mf_size: str = "100000") -> Dict:
    """
    Load Phase 3 run data for P00519 at specified MF size.
    
    Returns dict with actives, zinc, mf, embeddings, scores, config
    """
    # Find UMAP/features run with specified MF size
    run_pattern = f"umap_features_dim2_mf{mf_size}_rep1"
    run_dir = workspace_dir / run_pattern
    
    if not run_dir.exists():
        raise FileNotFoundError(f"Phase 3 run not found: {run_dir}")
    
    artifacts_dir = run_dir / "artifacts"
    logs_dir = run_dir / "logs"
    
    data = {
        "run_name": run_pattern,
        "phase": "Phase 3",
        "mf_size_target": mf_size,
    }
    
    # Load config
    config_path = run_dir / "config.json"
    if config_path.exists():
        with config_path.open("r") as f:
            data["config"] = json.load(f)
    
    # Load summary
    summary_path = logs_dir / "phase3_summary.json"
    if summary_path.exists():
        with summary_path.open("r") as f:
            data["summary"] = json.load(f)
    
    # Load embeddings
    for name in ["actives", "zinc", "mf"]:
        emb_path = artifacts_dir / f"embedding_{name}.csv"
        if emb_path.exists():
            data[f"embedding_{name}"] = pd.read_csv(emb_path)
    
    # Load ranked scores
    ranked_path = artifacts_dir / "ranked_scores.csv"
    if ranked_path.exists():
        data["ranked_scores"] = pd.read_csv(ranked_path)
    
    return data


def load_phase4_run(workspace_dir: Path, target: str = "Transferase", replicate: int = 1) -> Dict:
    """
    Load Phase 4 run data for specified target.
    
    Returns dict with actives, zinc, mf, embeddings, scores, config
    """
    run_name = f"umap_features_{target}_rep{replicate}"
    run_dir = workspace_dir / run_name
    
    if not run_dir.exists():
        raise FileNotFoundError(f"Phase 4 run not found: {run_dir}")
    
    artifacts_dir = run_dir / "artifacts"
    logs_dir = run_dir / "logs"
    
    data = {
        "run_name": run_name,
        "phase": "Phase 4",
        "target_short": target,
    }
    
    # Load config
    config_path = run_dir / "config.json"
    if config_path.exists():
        with config_path.open("r") as f:
            data["config"] = json.load(f)
    
    # Load summary
    summary_path = logs_dir / "phase4_summary.json"
    if summary_path.exists():
        with summary_path.open("r") as f:
            data["summary"] = json.load(f)
    
    # Load embeddings
    for name in ["actives", "zinc", "mf"]:
        emb_path = artifacts_dir / f"embedding_{name}.csv"
        if emb_path.exists():
            data[f"embedding_{name}"] = pd.read_csv(emb_path)
    
    # Load ranked scores
    ranked_path = artifacts_dir / "ranked_scores.csv"
    if ranked_path.exists():
        data["ranked_scores"] = pd.read_csv(ranked_path)
    
    return data


def compare_dataset_sizes(p3_data: Dict, p4_data: Dict) -> pd.DataFrame:
    """Compare dataset sizes between Phase 3 and Phase 4."""
    
    results = []
    
    for dataset_name in ["actives", "zinc", "mf"]:
        p3_key = f"embedding_{dataset_name}"
        p4_key = f"embedding_{dataset_name}"
        
        p3_size = len(p3_data[p3_key]) if p3_key in p3_data else 0
        p4_size = len(p4_data[p4_key]) if p4_key in p4_data else 0
        
        results.append({
            "dataset": dataset_name,
            "phase3_size": p3_size,
            "phase4_size": p4_size,
            "difference": p4_size - p3_size,
            "pct_difference": 100 * (p4_size - p3_size) / p3_size if p3_size > 0 else np.nan
        })
    
    return pd.DataFrame(results)


def compare_compound_overlap(p3_data: Dict, p4_data: Dict) -> Dict[str, Dict]:
    """
    Compare compound sets between Phase 3 and Phase 4.
    
    Returns dict with overlap statistics for actives, zinc, and mf.
    """
    results = {}
    
    for dataset_name in ["actives", "zinc", "mf"]:
        p3_key = f"embedding_{dataset_name}"
        p4_key = f"embedding_{dataset_name}"
        
        if p3_key not in p3_data or p4_key not in p4_data:
            continue
        
        # Get compound IDs (SMILES or Compound ChEMBL ID)
        id_col = "Compound ChEMBL ID" if "Compound ChEMBL ID" in p3_data[p3_key].columns else "SMILES"
        
        p3_ids = set(p3_data[p3_key][id_col].values)
        p4_ids = set(p4_data[p4_key][id_col].values)
        
        overlap = p3_ids & p4_ids
        p3_only = p3_ids - p4_ids
        p4_only = p4_ids - p3_ids
        
        results[dataset_name] = {
            "p3_unique": len(p3_ids),
            "p4_unique": len(p4_ids),
            "overlap": len(overlap),
            "p3_only": len(p3_only),
            "p4_only": len(p4_only),
            "jaccard_similarity": len(overlap) / len(p3_ids | p4_ids) if len(p3_ids | p4_ids) > 0 else 0,
        }
    
    return results


def compare_embedding_statistics(p3_data: Dict, p4_data: Dict) -> pd.DataFrame:
    """
    Compare embedding space statistics.
    
    Checks mean, std, min, max, distribution for each dataset.
    """
    results = []
    
    for dataset_name in ["actives", "zinc", "mf"]:
        p3_key = f"embedding_{dataset_name}"
        p4_key = f"embedding_{dataset_name}"
        
        if p3_key not in p3_data or p4_key not in p4_data:
            continue
        
        # Extract embedding dimensions
        p3_df = p3_data[p3_key]
        p4_df = p4_data[p4_key]
        
        dim_cols = [c for c in p3_df.columns if c.startswith("dim_")]
        
        p3_emb = p3_df[dim_cols].values
        p4_emb = p4_df[dim_cols].values
        
        # Compute statistics
        results.append({
            "dataset": dataset_name,
            "phase": "Phase 3",
            "n_points": len(p3_emb),
            "n_dims": len(dim_cols),
            "mean_dim0": p3_emb[:, 0].mean() if len(dim_cols) > 0 else np.nan,
            "std_dim0": p3_emb[:, 0].std() if len(dim_cols) > 0 else np.nan,
            "mean_dim1": p3_emb[:, 1].mean() if len(dim_cols) > 1 else np.nan,
            "std_dim1": p3_emb[:, 1].std() if len(dim_cols) > 1 else np.nan,
            "global_mean": p3_emb.mean(),
            "global_std": p3_emb.std(),
            "min_value": p3_emb.min(),
            "max_value": p3_emb.max(),
        })
        
        results.append({
            "dataset": dataset_name,
            "phase": "Phase 4",
            "n_points": len(p4_emb),
            "n_dims": len(dim_cols),
            "mean_dim0": p4_emb[:, 0].mean() if len(dim_cols) > 0 else np.nan,
            "std_dim0": p4_emb[:, 0].std() if len(dim_cols) > 0 else np.nan,
            "mean_dim1": p4_emb[:, 1].mean() if len(dim_cols) > 1 else np.nan,
            "std_dim1": p4_emb[:, 1].std() if len(dim_cols) > 1 else np.nan,
            "global_mean": p4_emb.mean(),
            "global_std": p4_emb.std(),
            "min_value": p4_emb.min(),
            "max_value": p4_emb.max(),
        })
    
    return pd.DataFrame(results)


def compare_score_distributions(p3_data: Dict, p4_data: Dict) -> pd.DataFrame:
    """
    Compare score distributions for actives.
    
    Returns percentiles and statistical tests.
    """
    if "ranked_scores" not in p3_data or "ranked_scores" not in p4_data:
        return pd.DataFrame()
    
    p3_scores = p3_data["ranked_scores"]
    p4_scores = p4_data["ranked_scores"]
    
    # Get active scores
    p3_active_scores = p3_scores[p3_scores["label"] == 1]["score"].values
    p4_active_scores = p4_scores[p4_scores["label"] == 1]["score"].values
    
    # Compute percentiles
    percentiles = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    
    results = []
    for p in percentiles:
        results.append({
            "percentile": p,
            "phase3_score": np.percentile(p3_active_scores, p),
            "phase4_score": np.percentile(p4_active_scores, p),
        })
    
    df_percentiles = pd.DataFrame(results)
    
    # Statistical test: Kolmogorov-Smirnov
    ks_stat, ks_pval = stats.ks_2samp(p3_active_scores, p4_active_scores)
    
    # Mann-Whitney U test
    u_stat, u_pval = stats.mannwhitneyu(p3_active_scores, p4_active_scores, alternative="two-sided")
    
    print(f"\n  Kolmogorov-Smirnov test: D={ks_stat:.4f}, p={ks_pval:.4e}")
    print(f"  Mann-Whitney U test: U={u_stat:.0f}, p={u_pval:.4e}")
    
    if ks_pval < 0.05:
        print(f"  ⚠️  Score distributions are significantly different (p < 0.05)")
    else:
        print(f"  ✓ Score distributions are not significantly different (p >= 0.05)")
    
    return df_percentiles


def compare_config_parameters(p3_data: Dict, p4_data: Dict) -> pd.DataFrame:
    """
    Compare configuration parameters that might affect results.
    """
    if "config" not in p3_data or "config" not in p4_data:
        return pd.DataFrame()
    
    p3_cfg = p3_data["config"]
    p4_cfg = p4_data["config"]
    
    # Parameters to compare
    params = [
        "target",
        "affinity_cutoff_nM",
        "random_seed",
        "method",
        "representation",
        "dim",
        "mf_features_csv",
        "zinc_features_csv",
    ]
    
    results = []
    for param in params:
        p3_val = p3_cfg.get(param, "N/A")
        p4_val = p4_cfg.get(param, "N/A")
        
        results.append({
            "parameter": param,
            "phase3": str(p3_val),
            "phase4": str(p4_val),
            "match": p3_val == p4_val,
        })
    
    # Check UMAP params
    p3_umap = p3_cfg.get("umap_params", {})
    p4_umap = p4_cfg.get("umap_params", {})
    
    for umap_param in ["n_neighbors", "min_dist", "metric"]:
        p3_val = p3_umap.get(umap_param, "N/A")
        p4_val = p4_umap.get(umap_param, "N/A")
        
        results.append({
            "parameter": f"umap_{umap_param}",
            "phase3": str(p3_val),
            "phase4": str(p4_val),
            "match": p3_val == p4_val,
        })
    
    return pd.DataFrame(results)


def compare_summary_metrics(p3_data: Dict, p4_data: Dict) -> pd.DataFrame:
    """
    Compare final summary metrics.
    """
    if "summary" not in p3_data or "summary" not in p4_data:
        return pd.DataFrame()
    
    p3_sum = p3_data["summary"]
    p4_sum = p4_data["summary"]
    
    metrics = [
        "n_actives",
        "n_zinc",
        "mf_size_natural",
        "ef_1%",
        "ef_5%",
        "ef_10%",
        "roc_auc",
        "pr_auc",
        "spearman_rho",
    ]
    
    results = []
    for metric in metrics:
        p3_val = p3_sum.get(metric, np.nan)
        p4_val = p4_sum.get(metric, np.nan)
        
        diff = p4_val - p3_val if not (np.isnan(p3_val) or np.isnan(p4_val)) else np.nan
        pct_diff = 100 * diff / p3_val if p3_val != 0 and not np.isnan(diff) else np.nan
        
        results.append({
            "metric": metric,
            "phase3": p3_val,
            "phase4": p4_val,
            "difference": diff,
            "pct_difference": pct_diff,
        })
    
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Diagnose Phase 3 vs Phase 4 discrepancy")
    parser.add_argument("--phase3_workspace", type=str, required=True,
                       help="Phase 3 workspace directory (e.g., experiment_workspace_v4/phase3/mf_ablation)")
    parser.add_argument("--phase4_workspace", type=str, required=True,
                       help="Phase 4 workspace directory (e.g., experiment_workspace_v4/phase4/cross_target)")
    parser.add_argument("--phase3_mf_size", type=str, default="100000",
                       help="Phase 3 MF size to compare (default: 100000)")
    parser.add_argument("--phase4_target", type=str, default="Transferase",
                       help="Phase 4 target to compare (default: Transferase)")
    parser.add_argument("--phase4_replicate", type=int, default=1,
                       help="Phase 4 replicate number (default: 1)")
    parser.add_argument("--output_dir", type=str, default="reporting/phase3_phase4_diagnosis",
                       help="Output directory for diagnosis results")
    args = parser.parse_args()
    
    p3_workspace = Path(args.phase3_workspace)
    p4_workspace = Path(args.phase4_workspace)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("PHASE 3 vs PHASE 4 DISCREPANCY DIAGNOSIS")
    print("="*80)
    print(f"Phase 3 workspace: {p3_workspace}")
    print(f"Phase 4 workspace: {p4_workspace}")
    print(f"Phase 3 MF size: {args.phase3_mf_size}")
    print(f"Phase 4 target: {args.phase4_target} (replicate {args.phase4_replicate})")
    print(f"Output directory: {output_dir}")
    print("="*80)
    
    # Load data
    print("\n[1/7] Loading Phase 3 data...")
    p3_data = load_phase3_run(p3_workspace, args.phase3_mf_size)
    print(f"  ✓ Loaded: {p3_data['run_name']}")
    
    print("\n[2/7] Loading Phase 4 data...")
    p4_data = load_phase4_run(p4_workspace, args.phase4_target, args.phase4_replicate)
    print(f"  ✓ Loaded: {p4_data['run_name']}")
    
    # Compare dataset sizes
    print("\n[3/7] Comparing dataset sizes...")
    df_sizes = compare_dataset_sizes(p3_data, p4_data)
    print(df_sizes.to_string(index=False))
    df_sizes.to_csv(output_dir / "dataset_sizes.csv", index=False)
    
    # Compare compound overlap
    print("\n[4/7] Comparing compound overlap...")
    overlap_stats = compare_compound_overlap(p3_data, p4_data)
    for dataset, stats in overlap_stats.items():
        print(f"\n  {dataset.upper()}:")
        print(f"    Phase 3 unique: {stats['p3_unique']}")
        print(f"    Phase 4 unique: {stats['p4_unique']}")
        print(f"    Overlap: {stats['overlap']} ({stats['jaccard_similarity']:.2%} Jaccard similarity)")
        print(f"    Phase 3 only: {stats['p3_only']}")
        print(f"    Phase 4 only: {stats['p4_only']}")
    
    df_overlap = pd.DataFrame(overlap_stats).T
    df_overlap.to_csv(output_dir / "compound_overlap.csv")
    
    # Compare embedding statistics
    print("\n[5/7] Comparing embedding statistics...")
    df_embeddings = compare_embedding_statistics(p3_data, p4_data)
    print(df_embeddings.to_string(index=False))
    df_embeddings.to_csv(output_dir / "embedding_statistics.csv", index=False)
    
    # Compare score distributions
    print("\n[6/7] Comparing score distributions for actives...")
    df_scores = compare_score_distributions(p3_data, p4_data)
    if not df_scores.empty:
        print(df_scores.to_string(index=False))
        df_scores.to_csv(output_dir / "score_percentiles.csv", index=False)
    
    # Compare configs
    print("\n[7/7] Comparing configuration parameters...")
    df_configs = compare_config_parameters(p3_data, p4_data)
    if not df_configs.empty:
        print(df_configs.to_string(index=False))
        df_configs.to_csv(output_dir / "config_comparison.csv", index=False)
        
        mismatches = df_configs[~df_configs["match"]]
        if len(mismatches) > 0:
            print(f"\n  ⚠️  Found {len(mismatches)} parameter mismatches:")
            for _, row in mismatches.iterrows():
                print(f"    - {row['parameter']}: P3={row['phase3']}, P4={row['phase4']}")
    
    # Compare summary metrics
    print("\n[SUMMARY] Final metric comparison...")
    df_summary = compare_summary_metrics(p3_data, p4_data)
    if not df_summary.empty:
        print(df_summary.to_string(index=False))
        df_summary.to_csv(output_dir / "summary_metrics.csv", index=False)
        
        # Highlight EF@1% difference
        ef1_row = df_summary[df_summary["metric"] == "ef_1%"]
        if not ef1_row.empty:
            ef1_p3 = ef1_row["phase3"].values[0]
            ef1_p4 = ef1_row["phase4"].values[0]
            ef1_diff = ef1_row["pct_difference"].values[0]
            print(f"\n  EF@1% Discrepancy:")
            print(f"    Phase 3: {ef1_p3:.2f}%")
            print(f"    Phase 4: {ef1_p4:.2f}%")
            print(f"    Difference: {ef1_diff:+.2f}%")
    
    print("\n" + "="*80)
    print("DIAGNOSIS COMPLETE")
    print(f"Results saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
