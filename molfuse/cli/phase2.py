from __future__ import annotations

import argparse
import json
from pathlib import Path

from molfuse import __version__
from molfuse.io.paths import make_run_dirs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="molfuse v4.0 Phase 2 runner (scaffold)")
    p.add_argument("--config", type=str, required=True, help="Path to config JSON")
    p.add_argument("--workspace", type=str, required=True, help="Base workspace directory")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    with cfg_path.open("r") as f:
        cfg = json.load(f)

    run_name = cfg.get("run_name", f"{cfg.get('target','target')}_{cfg.get('method','pca')}_{cfg.get('cutoff_nM','NA')}nM")
    ws = make_run_dirs(Path(args.workspace), phase="phase2", run_name=run_name)

    summary = {
        "molfuse_version": __version__,
        "phase": "cutoff_sensitivity",
        "config": cfg,
    }
    (ws["logs"] / "phase2_scaffold_summary.json").write_text(json.dumps(summary, indent=2))

    # Future: Reuse Phase 1 models/simspaces; rerun ranking at different cutoffs.


if __name__ == "__main__":
    main()
