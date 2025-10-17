# UMMBAS v3.0 - HPC Scripts

This directory contains SLURM job scripts for running UMMBAS v3.0 experiments on an HPC cluster.

## Files

### Active v3.0 Scripts

| File | Purpose |
|------|---------|
| `ummbas_v3_cpu.sh` | SLURM job script for running a single experiment (called by submit scripts) |
| `submit_v3_phase1.sh` | Submit all Phase 1 jobs (260 runs: dimensionality sweep) |
| `submit_v3_phase2.sh` | Submit all Phase 2 jobs (60 runs: MF cloud ablation) |
| `submit_v3_phase3.sh` | Submit all Phase 3 jobs (80 runs: cross-protein generalization) |
| `submit_v3_phase4.sh` | Submit all Phase 4 jobs (40 runs: affinity cutoff analysis) |

### Archive

| Directory | Contents |
|-----------|----------|
| `archive_v2/` | Old v2.0 HPC scripts (for reference only) |

## Quick Start

### Prerequisites (One-Time Setup)

1. **Setup workspace directories** (creates symlinks to `/work` partition):
   ```bash
   bash setup_hpc_workspaces.sh
   ```
   
   This creates:
   - `/work/ahagg2s/ummbas_results/experiment_workspace_v3_phase1/` → `experiment_workspace_v3_phase1`
   - `/work/ahagg2s/ummbas_results/experiment_workspace_v3_phase2/` → `experiment_workspace_v3_phase2`
   - `/work/ahagg2s/ummbas_results/experiment_workspace_v3_phase3/` → `experiment_workspace_v3_phase3`
   - `/work/ahagg2s/ummbas_results/experiment_workspace_v3_phase4/` → `experiment_workspace_v3_phase4`

2. **Update conda environment path** in `ummbas_v3_cpu.sh` (line 55):
   ```bash
   source /home/YOUR_USERNAME/miniforge3/bin/activate ummbas-screening
   ```

### Before Each Phase

3. **Generate configurations** for the phase you want to run:
   ```bash
   # For Phase 1
   python generate_phase1_configs.py
   ```

### Submit Jobs

```bash
# Phase 1: Hyperparameter Sweep (260 runs)
bash hpc/submit_v3_phase1.sh

# After Phase 1 completes, extract best configs:
python extract_phase1_best_configs.py

# Phase 2: MF Ablation (60 runs)
python generate_phase2_configs.py
bash hpc/submit_v3_phase2.sh

# Phase 3: Generalization (80 runs)
python generate_phase3_configs.py
bash hpc/submit_v3_phase3.sh

# Phase 4: Cutoff Analysis (40 runs)
python generate_phase4_configs.py
bash hpc/submit_v3_phase4.sh
```

### Monitor Progress

```bash
# Check SLURM queue
squeue -u $USER

# Check experiment progress
python scripts/check_hyperparam_status.py --workspace experiment_workspace_v3_phase1

# View logs
tail -f slurm_logs/UMMBAS_v3_*.out
```

## Resource Configuration

The SLURM script `ummbas_v3_cpu.sh` is configured for:

- **Partition:** `hpc` (CPU nodes)
- **CPUs:** 64 cores
- **Memory:** 350 GB
- **Time:** 24 hours
- **Excluded node:** wr43 (if problematic)

Adjust these in `ummbas_v3_cpu.sh` based on your cluster's resources.

## Seeds

All experiments run with 5 random seeds: **42, 43, 44, 45, 46**

Modify the `RANDOM_SEEDS` array in submission scripts if needed.

## Troubleshooting

### No configs found
```bash
# Generate configs first
python generate_phase1_configs.py  # or phase2, phase3, phase4
```

### Jobs fail immediately
- Check logs in `slurm_logs/`
- Verify conda environment path in `ummbas_v3_cpu.sh`
- Ensure `main_orchestrator.py` exists in project root

### Missing best configs (Phase 2-4)
```bash
# Extract best configs from Phase 1
python extract_phase1_best_configs.py \
    --workspace experiment_workspace_v3_phase1 \
    --output phase1_best_configs.json
```

## Expected Timeline

| Phase | Jobs | Time (32 cores) | Time (64 cores) |
|-------|------|-----------------|-----------------|
| Phase 1 | 260 | ~6-10 days | ~3-5 days |
| Phase 2 | 60 | ~1-2 days | ~0.5-1 day |
| Phase 3 | 80 | ~2-3 days | ~1-1.5 days |
| Phase 4 | 40 | ~1 day | ~0.5 day |
| **Total** | **440** | **~10-16 days** | **~5-8 days** |

Times depend on cluster load and specific hardware.
