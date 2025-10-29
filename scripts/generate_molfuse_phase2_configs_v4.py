#!/usr/bin/env python3
"""
Generate Phase 2 config files for affinity cutoff sensitivity analysis.

Outputs a single config JSON per target pointing to Phase 1 workspace.
Phase 2 CLI will auto-select best Phase 1 models.
"""
from __future__ import annotations

import json
from pathlib import Path


def generate_phase2_configs(
    output_dir: Path = Path("configs/molfuse_phase2_grid"),
    phase1_workspace: str = "experiment_workspace_v4",
    cutoff_list: list = None,
) -> None:
    """
    Generate Phase 2 config files.

    Args:
        output_dir: Directory to save configs
        phase1_workspace: Path to Phase 1 workspace (relative or absolute)
        cutoff_list: List of affinity cutoffs in nM (default: [100, 1000, 10000, 100000])
    """
    if cutoff_list is None:
        cutoff_list = [100, 1000, 10000, 100000]

    output_dir.mkdir(parents=True, exist_ok=True)

    # Phase 2 config (one per target; can expand to multiple targets later)
    config = {
        "run_name_prefix": "cutoff_sweep",
        "phase1_workspace": phase1_workspace,
        "phase1_phase_dir": "phase1",
        "affinity_cutoff_nM_list": cutoff_list,
        "notes": "Phase 2: Affinity Cutoff Sensitivity (Re-scoring Only). Reuses Phase 1 best models.",
    }

    config_path = output_dir / "phase2_cutoff_sweep.json"
    with config_path.open("w") as f:
        json.dump(config, f, indent=2)

    print(f"Generated Phase 2 config: {config_path}")
    print(f"  Cutoffs: {cutoff_list} nM")
    print(f"  Phase 1 workspace: {phase1_workspace}")
    print("\nUsage:")
    print(f"  python -m molfuse.cli.phase2 --config {config_path} --workspace {phase1_workspace}")


if __name__ == "__main__":
    generate_phase2_configs()
