#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from itertools import product

# Phase 1 config generator for molfuse v4 - Full 2D Mordred features
# Based on feature_comparison_v2.py results showing full 2D features yield highest EF@1%
# Writes configs to configs/molfuse_phase1_grid/

_parser = argparse.ArgumentParser(
    description="Generate Phase 1 config grid (hyperparameter sweep, ABL1 baseline)."
)
_parser.add_argument(
    "--data-dir",
    default="output_recalculated_full_datasets/datasets_2d_all",
    help=(
        "Root directory containing MF, ZINC, and actives CSV files. "
        "Expected layout: <data-dir>/KW-*/..., <data-dir>/zinc/..., <data-dir>/chembl/... "
        "(default: %(default)s)"
    ),
)
_parser.add_argument(
    "--output-dir",
    default="configs/molfuse_phase1_grid",
    help="Directory to write generated config files (default: %(default)s).",
)
_args = _parser.parse_args()
_DATA_DIR = Path(_args.data_dir)

BASE = {
    "target": "TyrosineProteinKinaseABL1_P00519",
    # Full 2D Mordred descriptors (1613 raw features)
    "mf_features_csv": str(_DATA_DIR / "KW-0808_Transferase_affinity_extracted_features.csv"),
    "zinc_features_csv": str(_DATA_DIR / "zinc" / "zinc_acquirable_extracted_features.csv"),
    "actives_features_csv": str(_DATA_DIR / "chembl" / "ABL1_P00519_actives_extracted_features.csv"),
    "sample_zinc": -1,
    "affinity_cutoff_nM": 100000,
    "on_empty_cutoff": "error",
}

# Grid (features)
PCA_DIMS = [2, 5, 10, 20]
UMAP_DIMS = [2, 5, 10, 20]
UMAP_NN = [5, 10, 50, 100, 500]
UMAP_MIN_DIST = [0.0, 0.01, 0.1]
UMAP_METRIC = "euclidean"


OUT_DIR = Path(_args.output_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Replicates
REPLICATES = [1, 2, 3, 4, 5]

# Generate PCA configs (features) with replicates
for d in PCA_DIMS:
    for r in REPLICATES:
        cfg = {
            **BASE,
            "run_name": f"ABL1_PCA_features_{d}d_rep{r}",
            "method": "pca",
            "dim": d,
            "umap_params": {"n_neighbors": 50, "min_dist": 0.01, "metric": UMAP_METRIC, "random_state": None},
            "notes": f"v4.0 generated PCA config (full 2D Mordred features, 1613 raw), replicate {r}",
            "representation": "features",
            "replicate": r,
        }
        out = OUT_DIR / f"ABL1_PCA_features_{d}d_rep{r}.json"
        out.write_text(json.dumps(cfg, indent=2))

# Generate UMAP configs (features) with replicates
for d, nn, md in product(UMAP_DIMS, UMAP_NN, UMAP_MIN_DIST):
    for r in REPLICATES:
        cfg = {
            **BASE,
            "run_name": f"ABL1_UMAP_features_{d}d_nn{nn}_md{str(md).replace('.', 'p')}_rep{r}",
            "method": "umap",
            "dim": d,
            "umap_params": {"n_neighbors": nn, "min_dist": md, "metric": UMAP_METRIC, "random_state": None},
            "notes": f"v4.0 generated UMAP config (full 2D Mordred features/Euclidean, 1613 raw), replicate {r}",
            "representation": "features",
            "replicate": r,
        }
        out = OUT_DIR / f"ABL1_UMAP_features_{d}d_nn{nn}_md{str(md).replace('.', 'p')}_rep{r}.json"
        out.write_text(json.dumps(cfg, indent=2))

# Fingerprint grid
FP_PCA_DIMS = [2, 5, 10, 20] 
FP_UMAP_DIMS = [2, 5, 10, 20]
FP_UMAP_NN = [5, 10, 50, 100, 500]
FP_UMAP_MIN_DIST = [0.0, 0.01, 0.1]

BASE_FP = {
    **BASE,
    # ECFP4 fingerprint CSVs
    "mf_features_csv": str(_DATA_DIR / "KW-0808_Transferase_affinity_extracted_fingerprints_ECFP4.csv"),
    "zinc_features_csv": str(_DATA_DIR / "zinc" / "zinc_acquirable_extracted_fingerprints_ECFP4.csv"),
    "actives_features_csv": str(_DATA_DIR / "chembl" / "ABL1_P00519_actives_extracted_fingerprints_ECFP4.csv"),
}

# Generate PCA configs (fingerprints) with replicates
for d in FP_PCA_DIMS:
    for r in REPLICATES:
        cfg = {
            **BASE_FP,
            "run_name": f"ABL1_PCA_fingerprints_{d}d_rep{r}",
            "method": "pca",
            "dim": d,
            "umap_params": {"n_neighbors": 50, "min_dist": 0.01, "metric": "euclidean", "random_state": None},
            "notes": f"v4.0 generated PCA config (fingerprints), replicate {r}",
            "representation": "fingerprints",
            "replicate": r,
        }
        out = OUT_DIR / f"ABL1_PCA_fingerprints_{d}d_rep{r}.json"
        out.write_text(json.dumps(cfg, indent=2))

# Generate UMAP configs (fingerprints, jaccard) with replicates
for d, nn, md in product(FP_UMAP_DIMS, FP_UMAP_NN, FP_UMAP_MIN_DIST):
    for r in REPLICATES:
        cfg = {
            **BASE_FP,
            "run_name": f"ABL1_UMAP_fingerprints_{d}d_nn{nn}_md{str(md).replace('.', 'p')}_rep{r}",
            "method": "umap",
            "dim": d,
            "umap_params": {"n_neighbors": nn, "min_dist": md, "metric": "jaccard", "random_state": None},
            "notes": f"v4.0 generated UMAP config (fingerprints/jaccard), replicate {r}",
            "representation": "fingerprints",
            "replicate": r,
        }
        out = OUT_DIR / f"ABL1_UMAP_fingerprints_{d}d_nn{nn}_md{str(md).replace('.', 'p')}_rep{r}.json"
        out.write_text(json.dumps(cfg, indent=2))

print(f"Wrote configs to {OUT_DIR.resolve()}")
