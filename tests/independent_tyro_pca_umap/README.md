Independent PCA vs UMAP test (TyrosineProteinKinaseABL1 / P00519)

This folder contains a self-contained script that compares PCA (5D) vs UMAP (5D, n_neighbors=5, min_dist=0.1) for enrichment on held-out actives of TyrosineProteinKinaseABL1 (P00519). It does not import any code from the rest of the repository.

Inputs
- MF features CSV (covering the relevant molecular function category, e.g., Transferase). Must include:
  - Columns: Compound ChEMBL ID, SMILES, accession, and numeric feature columns (39 RDKit descriptors or similar)
- ZINC features CSV with the same numeric feature columns and SMILES

What the script does
1) Deduplicates the MF file by Compound ChEMBL ID (features=first, affinity median if present)
2) Splits held-out actives (accession==P00519) from MF cloud (others)
3) Removes MF+actives SMILES from ZINC to avoid overlap
4) Fits StandardScaler + PCA(5D) on MF+ZINC; projects actives
5) Fits StandardScaler + UMAP(5D, nn=5, md=0.1) on MF+ZINC; projects actives
6) Scores all compounds by -distance to nearest MF neighbor in embedded space (exact 1-NN)
7) Computes EF@1%, ROC-AUC, PR-AUC, and saves ranking CSVs + summary.json

Run (example)

```bash
# Activate your environment first (example):
# mamba activate ummbas_screening

python tests/independent_tyro_pca_umap/compare_pca_umap_tyro.py \
  --mf_features_csv /ABS/PATH/TO/KW-0808_Transferase_affinity_extracted_features.csv \
  --zinc_features_csv /ABS/PATH/TO/precalculated_zinc_features.csv \
  --output_root tests/independent_tyro_pca_umap/work \
  --sample_zinc 200000
```

Outputs
- work/tyro_pca_vs_umap_YYYYMMDD_HHMMSS/
  - independent_pca_umap.log (detailed steps and counts)
  - results/
    - ranking_PCA-5D.csv
    - ranking_UMAP-nn5-md0.1-5D.csv
    - summary.json (counts, timings, EF@1%, ROC/PR AUC)

Notes
- UMAP requires `umap-learn` installed.
- If your ZINC CSV is very large, use `--sample_zinc` to run a faster sanity check first.
- The script identifies numeric feature columns present in BOTH CSVs automatically and uses their intersection to ensure compatibility.

Advanced options
- `--skip_dedup`: Skip deduplication by Compound ChEMBL ID. Useful for diagnostics if you want to compare counts pre/post dedup.
- `--umap_init {spectral,random}`: Choose UMAP initialization. `random` avoids spectral embedding and its associated warnings on very large or disconnected graphs; `spectral` is the default for determinism.

Diagnostics in logs
- The log will print numeric features found only in MF or only in ZINC (these are dropped), the common intersection used, and held-out actives counts pre/post dedup and after NaN filtering.
