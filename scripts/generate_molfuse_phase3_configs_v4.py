#!/usr/bin/env python3
"""
Generate Phase 3 Configuration Grid: MF Cloud Ablation Study

Generates configs for 4 methods × 6 MF sizes × 5 replicates = 120 runs

Strategy:
1. Read Phase 1 best hyperparameters from phase1_summary_grouped.csv
2. Read Phase 2 optimal cutoffs from phase2_best_cutoffs.json
3. Generate configs for MF sizes: [10, 100, 1000, 10000, 100000, "full"]
4. Each config includes: method, representation, dim, UMAP params, MF size, cutoff, replicate, seed

Output: configs/molfuse_phase3_grid/<method>_<repr>_dim<dim>_mf<size>_rep<N>.json

Usage:
    # Default: uses reporting/phase1_post_analysis/phase1_summary_grouped.csv
    #          and reporting/phase2_post_analysis/phase2_best_cutoffs.json
    python scripts/generate_molfuse_phase3_configs_v4.py \
        --output_dir configs/molfuse_phase3_grid
    
    # Custom paths:
    python scripts/generate_molfuse_phase3_configs_v4.py \
        --phase1_grouped reporting/phase1_post_analysis/phase1_summary_grouped.csv \
        --phase2_best_cutoffs reporting/phase2_post_analysis/phase2_best_cutoffs.json \
        --output_dir configs/molfuse_phase3_grid
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional


# ============================================================================
# Config Generation
# ============================================================================

def load_phase1_best(json_path: Optional[Path], phase1_grouped_csv: Optional[Path] = None) -> Dict[str, dict]:
    """
    Load Phase 1 best hyperparameters (method, representation, dim, umap params).
    
    Strategy:
    1. If json_path provided and exists: load directly
    2. Else if phase1_grouped_csv provided: extract best config per method×representation
    3. Else: FAIL (no assumptions)
    
    Args:
        json_path: Path to phase1_best_configs.json (keys: "features__pca__dim20", etc.)
        phase1_grouped_csv: Path to phase1_summary_grouped.csv (fallback)
    
    Returns:
        Dict mapping model_key (e.g., "pca_features") to config with dim, n_neighbors, min_dist
    """
    import pandas as pd
    
    # Strategy 1: Direct load from JSON
    if json_path and json_path.exists():
        with json_path.open("r") as f:
            phase1_raw = json.load(f)
        
        # Parse to identify best dimension per method×representation
        # Keys are like "features__pca__dim20"
        # We need to find best EF@1% per method×representation, but JSON only has hyperparams
        # Solution: Read the grouped CSV to get EF@1% values
        if not phase1_grouped_csv or not phase1_grouped_csv.exists():
            raise FileNotFoundError(
                f"phase1_best_configs.json exists but phase1_summary_grouped.csv is required to identify best dimensions.\n"
                f"Expected: {phase1_grouped_csv}"
            )
    
    # Strategy 2: Extract from grouped CSV
    if not phase1_grouped_csv or not phase1_grouped_csv.exists():
        raise FileNotFoundError(
            f"Phase 1 results not found. Provide either:\n"
            f"  --phase1_best (phase1_best_configs.json)\n"
            f"  --phase1_grouped (phase1_summary_grouped.csv)"
        )
    
    # Load grouped results and identify best per method×representation
    df = pd.read_csv(phase1_grouped_csv)
    best_configs = {}
    
    for (method, rep), grp in df.groupby(['method', 'representation']):
        # Find best by EF@1%
        best_row = grp.sort_values('ef1_mean', ascending=False).iloc[0]
        dim = int(best_row['dim'])
        
        # Build model_key matching Phase 2 format: "{method}_{representation}"
        model_key = f"{method}_{rep}"
        
        # Extract hyperparameters
        config = {
            "method": method,
            "representation": rep,
            "dim": dim,
        }
        
        # UMAP-specific params
        if method == "umap":
            config["umap_params"] = {
                "n_neighbors": int(best_row['umap_n_neighbors']) if pd.notna(best_row['umap_n_neighbors']) else 15,
                "min_dist": float(best_row['umap_min_dist']) if pd.notna(best_row['umap_min_dist']) else 0.1,
                "metric": str(best_row['umap_metric']) if pd.notna(best_row['umap_metric']) else "euclidean"
            }
        else:
            config["umap_params"] = None
        
        best_configs[model_key] = config
    
    return best_configs


def load_phase2_cutoffs(json_path: Optional[Path]) -> Dict[str, dict]:
    """
    Load Phase 2 best affinity cutoffs per model_key.
    
    Args:
        json_path: Path to phase2_best_cutoffs.json (optional)
    
    Returns:
        Dict mapping model_key to {cutoff_nM, ef1, ef5, ...}
        Returns empty dict if file not found (uses hardcoded 100 nM in generate_phase3_configs)
    """
    if not json_path or not json_path.exists():
        print("  WARNING: Phase 2 cutoffs not found, using hardcoded 100 nM cutoff")
        return {}
    
    with json_path.open("r") as f:
        return json.load(f)


def generate_phase3_configs(
    phase1_best: Dict,
    phase2_cutoffs: Dict,
    output_dir: Path,
    target: str = "TyrosineProteinKinaseABL1_P00519",
    mf_sizes: List | None = None,
    replicates: List[int] | None = None,
    data_dir: Path = Path("output_recalculated_full_datasets/datasets_2d_all"),
) -> List[Path]:
    """
    Generate Phase 3 configuration grid.
    
    Args:
        phase1_best: Dict mapping model_key → best hyperparameters
        phase2_cutoffs: Dict mapping method → optimal cutoff (nM)
        output_dir: Output directory for configs
        target: Target protein
        mf_sizes: List of MF cloud sizes [10, 100, 1000, 10000, 100000, "full"]
        replicates: List of replicate numbers [1, 2, 3, 4, 5]
    
    Returns:
        List of created config file paths
    """
    if mf_sizes is None:
        mf_sizes = [10, 100, 1000, 10000, 100000, "full"]
    
    if replicates is None:
        replicates = [1, 2, 3, 4, 5]
    
    output_dir.mkdir(parents=True, exist_ok=True)
    created_configs = []
    
    # Base paths (2D Mordred features + ECFP4 fingerprints)
    base_paths = {
        "mf_features_csv": str(data_dir / "KW-0808_Transferase_affinity_extracted_features.csv"),
        "zinc_features_csv": str(data_dir / "zinc" / "zinc_acquirable_extracted_features.csv"),
        "mf_fingerprints_csv": str(data_dir / "KW-0808_Transferase_affinity_extracted_fingerprints_ECFP4.csv"),
        "zinc_fingerprints_csv": str(data_dir / "zinc" / "zinc_acquirable_extracted_fingerprints_ECFP4.csv"),
    }
    
    # Generate configs for each method × MF size × replicate
    for model_key, best_config in phase1_best.items():
        method = best_config["method"]
        representation = best_config["representation"]
        dim = best_config["dim"]
        umap_params = best_config.get("umap_params")
        
        # Use optimal cutoff from Phase 2: 100 nM (best performance across all methods)
        # Note: This is used for SCORING only, not for filtering the training MF cloud
        cutoff_nM = 100
        
        for mf_size in mf_sizes:
            for replicate in replicates:
                # Generate run name (include replicate)
                mf_size_str = str(mf_size) if mf_size != "full" else "full"
                run_name = f"{method}_{representation}_dim{dim}_mf{mf_size_str}_rep{replicate}"
                
                # Random seed based on replicate number
                random_seed = replicate * 42  # Seeds: 42, 84, 126, 168, 210
                
                config = {
                    "run_name": run_name,
                    "phase3_run_name": "mf_ablation",
                    "target": target,
                    "method": method,
                    "representation": representation,
                    "dim": dim,
                    "mf_size": mf_size,
                    "affinity_cutoff_nM": cutoff_nM,
                    "replicate": replicate,
                    "random_seed": random_seed,
                    "umap_params": umap_params,
                    **base_paths,
                }
                
                # Save config
                config_path = output_dir / f"{run_name}.json"
                with config_path.open("w") as f:
                    json.dump(config, f, indent=2)
                
                created_configs.append(config_path)
                print(f"✓ Generated: {config_path.name}")
    
    return created_configs


def main():
    parser = argparse.ArgumentParser(description="Generate Phase 3 configuration grid")
    parser.add_argument("--phase1_best", type=str, default=None,
                       help="Path to Phase 1 best configs JSON (optional; will use phase1_grouped if not provided)")
    parser.add_argument("--phase1_grouped", type=str, default="reporting/phase1_post_analysis/phase1_summary_grouped.csv",
                       help="Path to Phase 1 grouped summary CSV (for extracting best hyperparameters)")
    parser.add_argument("--phase2_best_cutoffs", type=str, default="reporting/phase2_post_analysis/phase2_best_cutoffs.json",
                       help="Path to Phase 2 best cutoffs JSON (optional, uses 100 nM if not found)")
    parser.add_argument("--output-dir", type=str, default="configs/molfuse_phase3_grid",
                       help="Output directory for configs")
    parser.add_argument("--target", type=str, default="TyrosineProteinKinaseABL1_P00519",
                       help="Target protein")
    parser.add_argument("--mf_sizes", type=str, default="10,100,1000,10000,100000,full",
                       help="Comma-separated MF sizes")
    parser.add_argument("--replicates", type=str, default="1,2,3,4,5",
                       help="Comma-separated replicate numbers")
    parser.add_argument("--data-dir", type=str, default="output_recalculated_full_datasets/datasets_2d_all",
                       help="Root directory containing MF and ZINC CSV files (default: %(default)s).")
    args = parser.parse_args()
    
    # Parse inputs
    phase1_json_path = Path(args.phase1_best) if args.phase1_best else None
    phase1_grouped_path = Path(args.phase1_grouped)
    phase2_path = Path(args.phase2_best_cutoffs)
    output_dir = Path(args.output_dir)
    data_dir = Path(args.data_dir)
    
    # Parse MF sizes
    mf_sizes = []
    for s in args.mf_sizes.split(","):
        s = s.strip()
        if s.lower() == "full":
            mf_sizes.append("full")
        else:
            mf_sizes.append(int(s))
    
    # Parse replicates
    replicates = [int(r.strip()) for r in args.replicates.split(",")]
    
    # Load Phase 1 best and Phase 2 cutoffs
    phase1_best = load_phase1_best(phase1_json_path, phase1_grouped_path)
    phase2_cutoffs = load_phase2_cutoffs(phase2_path)
    
    print("="*80)
    print("PHASE 3 CONFIG GENERATOR: MF Cloud Ablation")
    print("="*80)
    print(f"Target: {args.target}")
    print(f"MF sizes: {mf_sizes}")
    print(f"Replicates: {replicates}")
    print(f"Output: {output_dir}")
    print()
    
    print("Phase 1 Best Configurations:")
    for model_key, cfg in phase1_best.items():
        umap_str = ""
        if cfg.get("umap_params"):
            umap_str = f" (n_neighbors={cfg['umap_params']['n_neighbors']}, min_dist={cfg['umap_params']['min_dist']})"
        print(f"  {model_key}: {cfg['method']}/{cfg['representation']} (dim={cfg['dim']}){umap_str}")
    print()
    
    print("Affinity Cutoff (for scoring only):")
    print("  Using 100 nM for all methods (optimal from Phase 2)")
    print("  Note: Cutoff applied to MF cloud for SCORING only, NOT for training")
    print()
    
    # Generate configs
    created = generate_phase3_configs(
        phase1_best,
        phase2_cutoffs,
        output_dir,
        target=args.target,
        mf_sizes=mf_sizes,
        replicates=replicates,
        data_dir=data_dir,
    )
    
    print("="*80)
    print(f"✓ Generated {len(created)} Phase 3 configs")
    print(f"  Models: {len(phase1_best)} methods")
    print(f"  MF sizes: {len(mf_sizes)} sizes")
    print(f"  Replicates: {len(replicates)} replicates")
    print(f"  Total runs: {len(created)}")
    print("="*80)
    print()
    print("Next steps:")
    print(f"  1. Review configs in {output_dir}")
    print(f"  2. Submit to HPC: bash hpc/submit_molfuse_phase3.sh {output_dir} experiment_workspace_v4")


if __name__ == "__main__":
    main()
