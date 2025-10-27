Mordred Full Feature Evaluation

This standalone script compares:
- Current 40-feature subset (documented in LAB_BOOK.md) selected from Mordred 2D
- Full Mordred 2D feature set
- Full Mordred 2D + 3D feature set

It builds UMAP embeddings (n_neighbors=10, min_dist=0.1; auto-bumped to 2 if required) and evaluates EF@1% using centroid-distance scoring in feature space.

Usage

1) Create a fresh environment and install dependencies:
   - python -m venv .venv && source .venv/bin/activate
   - pip install -r requirements_mordred_test.txt

2) Run a quick test with defaults (uses datasets/):
   - python mordred_full_feature_compare.py --output_dir tests/mordred_full_feature_eval/output_small --n_target 100 --n_mf 200 --n_zinc 200

3) Use a specific target MF file (example):
   - python mordred_full_feature_compare.py --target_ligands_csv datasets/molecular_function_affinity_data/KW-0808_Transferase_affinity.csv --n_target 300 --n_mf 600 --n_zinc 600 --output_dir tests/mordred_full_feature_eval/output_transferase

Coverage-aware selection (new)

- Feature prevalence thresholds: keep a descriptor if prevalence ≥ pf_target in targets OR ≥ pf_mf in MF.
   - Defaults: pf_target=0.95 (protect targets), pf_mf=0.7 (moderate MF)
- Row completeness thresholds: drop rows per set (targets protected, ZINC harshest):
   - pr_target_guard=0.2, pr_mf=0.6, pr_zinc=0.8
- Apply a unified row mask (2D ∩ 2D+3D) across all three representations for fair EF/UMAP comparisons.

Sweep (optional)

- Enable `--enable_sweep` to grid over thresholds, saving `coverage_grid.csv` and a `coverage_pareto.png` summarizing trade-offs.
- Weights for the coverage objective can be tuned with `--w_target`, `--w_mf`, `--w_zinc`, `--w_cols`.

Outputs

- summary.json: EF@1% for each feature set, counts, effective dimensions, thresholds, and kept stats
- umap_all.png: combined UMAP scatter (Current 40, Full 2D, Full 2D+3D) with shared axes
- dist_hist_all.png: combined density plots of distances to active centroid for all three representations
- scored_molecules.csv: SMILES, source, and scores for all molecules used (intersection of successful 2D and 3D)
- feature_coverage_2d.csv, feature_coverage_2d3d.csv: per-descriptor prevalence by set and selection flag
- row_completeness.csv: per-SMILES completeness in 2D/2D+3D and keep flags (2D, 2D+3D, final)
- coverage_grid.csv, coverage_pareto.png (optional): threshold sweep report and Pareto overview

Notes

 - 3D: The script generates 3D conformers (ETKDGv3) with MMFF/UFF optimization. Molecules failing 3D are skipped; 2D∩3D-success is used to keep comparisons fair.
- Cleaning: Non-numeric Mordred columns are dropped; remaining NaNs are median-imputed; zero-variance columns are removed; StandardScaler is applied.
- EF@1% scoring: We compute distances to the active-class centroid in feature space and rank molecules by closeness.
 - Caching: Descriptor calculation is cached by SMILES in `tests/mordred_full_feature_eval/cache/` (separate files for 2D and 3D). Re-running on overlapping SMILES will reuse cached descriptors and only compute for new ones. Use `--cache_dir` to change the location.
 - Deduplication priority: When the same SMILES appears in multiple sources, we keep a single copy with priority Target > MF cloud > ZINC (mirrors main experiment policy).
 - Failure tracking: `failure_reasons_2d.csv`, `failure_reasons_3d.csv`, and `descriptor_nan_counts.csv` capture per-SMILES/per-descriptor diagnostics.

Key flags (selection/sweep)

- --pf_target, --pf_mf: Feature prevalence thresholds (defaults 0.95, 0.7)
- --pr_target_guard, --pr_mf, --pr_zinc: Row completeness thresholds per set (defaults 0.2, 0.6, 0.8)
- --enable_sweep and grids: --pf_target_grid, --pf_mf_grid, --pr_mf_grid, --pr_zinc_grid
- Sweep weights: --w_target, --w_mf, --w_zinc, --w_cols
