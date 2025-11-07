#!/usr/bin/env python3
"""
Generate configs for UMAP initialization method comparison test.

Creates 5 configurations:
1. PCA 20D (baseline)
2. UMAP 20D init='spectral' (default)
3. UMAP 20D init='random'
4. UMAP 20D init='pca'
5. UMAP 20D init='tswspectral' (spectral embedding of fuzzy simplicial set)

Fixed parameters:
- Target: ABL1/P00519
- Dataset: Full 2D Mordred features
- ZINC: Full dataset
- UMAP: n_neighbors=50, min_dist=0.01, metric='euclidean'
"""

from __future__ import annotations

import json
from pathlib import Path

# Base configuration
BASE = {
    "target": "TyrosineProteinKinaseABL1_P00519",
    "mf_features_csv": "output_recalculated_full_datasets/datasets_2d_all/KW-0808_Transferase_affinity_extracted_features.csv",
    "zinc_features_csv": "output_recalculated_full_datasets/datasets_2d_all/zinc/zinc_acquirable_extracted_features.csv",
    "affinity_cutoff_nM": 100000,
    "dim": 20,
    "representation": "features",
}

# UMAP fixed parameters
UMAP_PARAMS = {
    "n_neighbors": 50,
    "min_dist": 0.01,
    "metric": "euclidean",
}

# Output directory
OUT_DIR = Path("tests/umap_init_comparison/configs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Generate PCA baseline
pca_config = {
    **BASE,
    "run_name": "PCA_20d_baseline",
    "method": "pca",
    "umap_params": UMAP_PARAMS,  # Keep for consistency
    "notes": "PCA 20D baseline for UMAP init comparison",
}
out_path = OUT_DIR / "pca_20d_baseline.json"
out_path.write_text(json.dumps(pca_config, indent=2))
print(f"✓ Generated: {out_path}")

# Generate UMAP configs with different init methods
init_methods = [
    ("spectral", "UMAP default spectral initialization"),
    ("random", "UMAP random initialization"),
    ("pca", "UMAP with PCA initialization"),
    ("tswspectral", "UMAP with t-SNE spectral initialization (fuzzy simplicial set)"),
]

for init, description in init_methods:
    config = {
        **BASE,
        "run_name": f"UMAP_20d_init_{init}",
        "method": "umap",
        "init": init,
        "umap_params": UMAP_PARAMS,
        "notes": description,
    }
    out_path = OUT_DIR / f"umap_20d_init_{init}.json"
    out_path.write_text(json.dumps(config, indent=2))
    print(f"✓ Generated: {out_path}")

print(f"\n✓ All configs generated in {OUT_DIR}")
print(f"  Total: 5 configurations (1 PCA + 4 UMAP inits)")
