#!/usr/bin/env python3
"""
Generate Phase 3 Configuration Grid: MF Cloud Ablation Study

Generates configs for 4 methods × 6 MF sizes = 24 runs

Strategy:
1. Read Phase 1 best hyperparameters (from Phase 1 results or manual specification)
2. Read Phase 2 optimal cutoffs (method-specific: PCA cutoff, UMAP cutoff)
3. Generate configs for MF sizes: [10, 100, 1000, 10000, 100000, "full"]
4. Each config includes: method, representation, dim, UMAP params, MF size, cutoff, seed

Output: configs/molfuse_phase3_grid/<method>_<repr>_mf<size>.json

Usage:
    python scripts/generate_molfuse_phase3_configs_v4.py \
        --phase1_best configs/phase1_best_configs.json \
        --phase2_best_cutoffs reporting/phase2_post_analysis/phase2_best_cutoffs.json \
        --output_dir configs/molfuse_phase3_grid
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


# ============================================================================
# Default Best Configurations (if files not provided)
# ============================================================================

DEFAULT_PHASE1_BEST = {
    "pca_features": {
        "method": "pca",
        "representation": "features",
        "dim": 20,
        "umap_params": {"n_neighbors": 50, "min_dist": 0.01},  # Unused for PCA
    },
    "pca_fingerprints": {
        "method": "pca",
        "representation": "fingerprints",
        "dim": 20,
        "umap_params": {"n_neighbors": 50, "min_dist": 0.01},
    },
    "umap_features": {
        "method": "umap",
        "representation": "features",
        "dim": 20,
        "umap_params": {"n_neighbors": 50, "min_dist": 0.01},
    },
    "umap_fingerprints": {
        "method": "umap",
        "representation": "fingerprints",
        "dim": 20,
        "umap_params": {"n_neighbors": 50, "min_dist": 0.01},
    },
}

DEFAULT_PHASE2_CUTOFFS = {
    "pca": 100000,      # nM (permissive default)
    "umap": 100000,     # nM (permissive default)
}


# ============================================================================
# Config Generation
# ============================================================================

def load_phase1_best(path: Path | None) -> Dict:
    """Load Phase 1 best configurations from JSON."""
    if path is None or not path.exists():
        print(f"WARNING: Phase 1 best configs not found, using defaults")
        return DEFAULT_PHASE1_BEST
    
    with path.open("r") as f:
        return json.load(f)


def load_phase2_cutoffs(path: Path | None) -> Dict:
    """Load Phase 2 best cutoffs from JSON."""
    if path is None or not path.exists():
        print(f"WARNING: Phase 2 best cutoffs not found, using defaults")
        return DEFAULT_PHASE2_CUTOFFS
    
    with path.open("r") as f:
        data = json.load(f)
    
    # Convert from model_key → cutoff to method → cutoff
    cutoffs = {}
    for model_key, info in data.items():
        if "pca" in model_key.lower():
            cutoffs["pca"] = info["cutoff_nM"]
        elif "umap" in model_key.lower():
            cutoffs["umap"] = info["cutoff_nM"]
    
    return cutoffs if cutoffs else DEFAULT_PHASE2_CUTOFFS


def generate_phase3_configs(
    phase1_best: Dict,
    phase2_cutoffs: Dict,
    output_dir: Path,
    target: str = "TyrosineProteinKinaseABL1_P00519",
    mf_sizes: List = None,
    random_seed: int = 42,
) -> List[Path]:
    """
    Generate Phase 3 configuration grid.
    
    Args:
        phase1_best: Dict mapping model_key → best hyperparameters
        phase2_cutoffs: Dict mapping method → optimal cutoff (nM)
        output_dir: Output directory for configs
        target: Target protein
        mf_sizes: List of MF cloud sizes [10, 100, 1000, 10000, 100000, "full"]
        random_seed: Random seed for MF subsampling
    
    Returns:
        List of created config file paths
    """
    if mf_sizes is None:
        mf_sizes = [10, 100, 1000, 10000, 100000, "full"]
    
    output_dir.mkdir(parents=True, exist_ok=True)
    created_configs = []
    
    # Base paths (2D Mordred features)
    base_paths = {
        "mf_features_csv": "output_recalculated_full_datasets/datasets_2d_all/KW-0808_Transferase_affinity_extracted_features.csv",
        "zinc_features_csv": "output_recalculated_full_datasets/datasets_2d_all/zinc/zinc_acquirable_extracted_features.csv",
        "mf_fingerprints_csv": "datasets/molecular_function_features_fingerprints/KW-0808_Transferase_affinity_fingerprints.csv",
        "zinc_fingerprints_csv": "datasets/molecular_function_features_fingerprints/zinc_acquirable_fingerprints.csv",
    }
    
    # Generate configs for each method × MF size
    for model_key, best_config in phase1_best.items():
        method = best_config["method"]
        representation = best_config["representation"]
        dim = best_config["dim"]
        umap_params = best_config.get("umap_params", {"n_neighbors": 50, "min_dist": 0.01})
        
        # Get method-specific cutoff
        cutoff_nM = phase2_cutoffs.get(method, 100000)
        
        for mf_size in mf_sizes:
            # Generate run name
            mf_size_str = str(mf_size) if mf_size != "full" else "full"
            run_name = f"{method}_{representation}_dim{dim}_mf{mf_size_str}"
            
            config = {
                "run_name": run_name,
                "phase3_run_name": "mf_ablation",
                "target": target,
                "method": method,
                "representation": representation,
                "dim": dim,
                "mf_size": mf_size,
                "affinity_cutoff_nM": cutoff_nM,
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
                       help="Path to Phase 1 best configs JSON (optional)")
    parser.add_argument("--phase2_best_cutoffs", type=str, default=None,
                       help="Path to Phase 2 best cutoffs JSON (optional)")
    parser.add_argument("--output_dir", type=str, default="configs/molfuse_phase3_grid",
                       help="Output directory for configs")
    parser.add_argument("--target", type=str, default="TyrosineProteinKinaseABL1_P00519",
                       help="Target protein")
    parser.add_argument("--mf_sizes", type=str, default="10,100,1000,10000,100000,full",
                       help="Comma-separated MF sizes")
    parser.add_argument("--random_seed", type=int, default=42,
                       help="Random seed for MF subsampling")
    args = parser.parse_args()
    
    # Parse inputs
    phase1_path = Path(args.phase1_best) if args.phase1_best else None
    phase2_path = Path(args.phase2_best_cutoffs) if args.phase2_best_cutoffs else None
    output_dir = Path(args.output_dir)
    
    # Parse MF sizes
    mf_sizes = []
    for s in args.mf_sizes.split(","):
        s = s.strip()
        if s.lower() == "full":
            mf_sizes.append("full")
        else:
            mf_sizes.append(int(s))
    
    # Load Phase 1 best and Phase 2 cutoffs
    phase1_best = load_phase1_best(phase1_path)
    phase2_cutoffs = load_phase2_cutoffs(phase2_path)
    
    print("="*80)
    print("PHASE 3 CONFIG GENERATOR: MF Cloud Ablation")
    print("="*80)
    print(f"Target: {args.target}")
    print(f"MF sizes: {mf_sizes}")
    print(f"Random seed: {args.random_seed}")
    print(f"Output: {output_dir}")
    print()
    
    print("Phase 1 Best Configurations:")
    for model_key, cfg in phase1_best.items():
        print(f"  {model_key}: {cfg['method']}/{cfg['representation']} (dim={cfg['dim']})")
    print()
    
    print("Phase 2 Optimal Cutoffs:")
    for method, cutoff in phase2_cutoffs.items():
        print(f"  {method}: {cutoff} nM")
    print()
    
    # Generate configs
    created = generate_phase3_configs(
        phase1_best,
        phase2_cutoffs,
        output_dir,
        target=args.target,
        mf_sizes=mf_sizes,
        random_seed=args.random_seed,
    )
    
    print("="*80)
    print(f"✓ Generated {len(created)} Phase 3 configs")
    print(f"  Models: {len(phase1_best)} methods")
    print(f"  MF sizes: {len(mf_sizes)} sizes")
    print(f"  Total runs: {len(created)}")
    print("="*80)
    print()
    print("Next steps:")
    print(f"  1. Review configs in {output_dir}")
    print(f"  2. Submit to HPC: bash hpc/submit_molfuse_phase3.sh {output_dir} experiment_workspace_v4")


if __name__ == "__main__":
    main()
