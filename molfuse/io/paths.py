from __future__ import annotations

from pathlib import Path
from typing import Dict


def make_run_dirs(base_dir: Path, phase: str, run_name: str) -> Dict[str, Path]:
    """
    Create a minimal structured workspace for a run.

    Returns dict with keys: base, logs, artifacts, metrics.
    """
    run_base = base_dir / phase / run_name
    logs = run_base / "logs"
    artifacts = run_base / "artifacts"
    metrics = run_base / "metrics"
    for p in (run_base, logs, artifacts, metrics):
        p.mkdir(parents=True, exist_ok=True)
    return {"base": run_base, "logs": logs, "artifacts": artifacts, "metrics": metrics}
