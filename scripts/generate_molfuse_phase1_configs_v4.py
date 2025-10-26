#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from itertools import product

# Simple Phase 1 config generator for molfuse v4
# Writes configs to configs/molfuse_phase1_grid/

BASE = {
    "target": "TyrosineProteinKinaseABL1_P00519",
    "mf_features_csv": "datasets/molecular_function_features_fingerprints/KW-0808_Transferase_affinity_extracted_features.csv",
    "zinc_features_csv": "datasets/molecular_function_features_fingerprints/zinc/zinc_acquirable_extracted_features.csv",
    "actives_features_csv": "datasets/molecular_function_features_fingerprints/chembl/ABL1_P00519_actives_extracted_features.csv",
    "sample_zinc": -1,
    "affinity_cutoff_nM": 100000,
    "on_empty_cutoff": "error",
}

# Grid (features)
PCA_DIMS = [2, 5, 10]
UMAP_DIMS = [2, 5, 10]
UMAP_NN = [5, 10, 50, 100, 500]
UMAP_MIN_DIST = [0.0, 0.01, 0.05, 0.1]
UMAP_METRIC = "euclidean"

OUT_DIR = Path("configs/molfuse_phase1_grid")
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
            "notes": f"v4.0 generated PCA config (features), replicate {r}",
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
            "notes": f"v4.0 generated UMAP config (features/Euclidean), replicate {r}",
            "representation": "features",
            "replicate": r,
        }
        out = OUT_DIR / f"ABL1_UMAP_features_{d}d_nn{nn}_md{str(md).replace('.', 'p')}_rep{r}.json"
        out.write_text(json.dumps(cfg, indent=2))

# Fingerprint grid
FP_PCA_DIMS = [2, 5, 10]
FP_UMAP_DIMS = [2, 5, 10]
FP_UMAP_NN = [5, 10, 50, 100, 500]
FP_UMAP_MIN_DIST = [0.0, 0.01, 0.05, 0.1]

BASE_FP = {
    **BASE,
    # Point to fingerprint CSVs (adjust path pattern if needed)
    "mf_features_csv": BASE["mf_features_csv"].replace("extracted_features.csv", "extracted_fingerprints_ECFP4.csv"),
    "zinc_features_csv": BASE["zinc_features_csv"].replace("extracted_features.csv", "extracted_fingerprints_ECFP4.csv"),
    "actives_features_csv": BASE["actives_features_csv"].replace("extracted_features.csv", "extracted_fingerprints_ECFP4.csv"),
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
