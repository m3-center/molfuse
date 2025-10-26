# molfuse v4.0 (Scaffold)

This is the v4.0 refactor scaffold to align the HPC pipelines with the independently validated test harness.

Core invariants:
- StandardScaler fits on MF+ZINC only; actives are projected using this scaler (no leakage)
- UMAP runs without a fixed seed (random_state=None) to enable parallelism
- Exact 1-NN scoring backend in embedded space; score = -min_distance
- Affinity cutoff applies to MF cloud for scoring only; actives are not filtered
- Spearman’s rho(pActivity vs score) is reported for actives
- Target-preserving exclusion enforced

## Try it (scaffold)

- Phase 1 (logs invariants and config into a structured workspace):
  python -m molfuse.cli.phase1 --config configs/molfuse_phase1_example.json --workspace ./experiment_workspace_v4

- Phase 2 (logs cutoff config):
  python -m molfuse.cli.phase2 --config configs/molfuse_phase1_example.json --workspace ./experiment_workspace_v4

Note: These CLIs are scaffolds; end-to-end execution will be implemented next.