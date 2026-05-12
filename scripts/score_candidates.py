"""
CLI scoring script — headless / HPC equivalent of the MolFuSE GUI scorer.

Usage
-----
python scripts/score_candidates.py \
    --workspace experiment_workspace_v4 \
    --run-name <run_name> \
    --candidates candidates.parquet \
    --output scored_candidates.csv

The candidates file must contain a SMILES column.  If it already contains the
Mordred 2D descriptor columns expected by the model they are used directly and
descriptor calculation is skipped (fast path for pre-computed feature files).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Ensure the repo root is on sys.path so `molfuse` is importable even without
# `pip install -e .` (useful on HPC clusters that mount the repo directly).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from molfuse.gui.components.model_loader import ModelLoader
from molfuse.gui.components.projector import CandidateProjector
from molfuse.gui.components.descriptor_calc import DescriptorCalculator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("score_candidates")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Score candidate molecules against a trained MolFuSE model."
    )
    p.add_argument(
        "--workspace", required=True,
        help="Path to the experiment workspace directory.",
    )
    p.add_argument(
        "--list", action="store_true", dest="list_runs",
        help="List all valid run names in the workspace and exit.",
    )
    p.add_argument(
        "--run-name", default=None, dest="run_name",
        help="Name of the trained model run (subdirectory under phase1/ or phase4/).",
    )
    p.add_argument(
        "--candidates", default=None,
        help="CSV or Parquet file with a SMILES column.",
    )
    p.add_argument(
        "--output", default="scored_candidates.csv",
        help="Output CSV path (default: scored_candidates.csv).",
    )
    p.add_argument(
        "--phase", default=None, choices=["phase1", "phase4"],
        help="Which phase directory to scan (default: try both, pick first match).",
    )
    return p.parse_args()


def load_candidates(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main() -> int:
    args = parse_args()

    workspace = Path(args.workspace).expanduser().resolve()
    if not workspace.exists():
        logger.error(f"Workspace not found: {workspace}")
        return 1

    # Scan workspace
    phases = [args.phase] if args.phase else ["phase1", "phase4"]
    loader = ModelLoader(workspace)
    models = loader.scan_models(phases=phases)

    # --list: print available runs and exit
    if args.list_runs:
        if not models:
            print(f"No valid models found in {workspace} under {phases}")
            return 1
        valid = [m for m in models if m["valid"]]
        invalid = [m for m in models if not m["valid"]]
        col_w = max(len(m["run_name"]) for m in models) + 2
        header = f"{'RUN NAME':<{col_w}}  {'KW CATEGORY':<25}  {'DR METHOD':<8}  {'EF@1%':>6}  STATUS"
        print(header)
        print("-" * len(header))
        for m in sorted(valid, key=lambda x: x["run_name"]):
            ef = m.get("ef_1pct")
            ef_str = f"{ef:.2f}" if ef is not None else "  n/a"
            print(f"{m['run_name']:<{col_w}}  {m.get('kw_category', ''):<25}  {m.get('dr_method', ''):<8}  {ef_str:>6}")
        if invalid:
            print(f"\n{len(invalid)} incomplete run(s) skipped (missing artifacts).")
        return 0

    # Validate required args for scoring
    if not args.run_name:
        logger.error("--run-name is required for scoring. Use --list to see available runs.")
        return 1
    if not args.candidates:
        logger.error("--candidates is required for scoring.")
        return 1

    if not models:
        logger.error(f"No models found in {workspace} under {phases}")
        return 1

    # Find the requested run
    match = next((m for m in models if m["run_name"] == args.run_name), None)
    if match is None:
        available = [m["run_name"] for m in models]
        logger.error(
            f"Run '{args.run_name}' not found. Available runs:\n  "
            + "\n  ".join(available)
        )
        return 1

    if not match["valid"]:
        logger.error(f"Run '{args.run_name}' exists but is missing required artifacts.")
        return 1

    size_mb = match.get("dr_model_size_mb")
    if size_mb:
        logger.info(f"DR model size on disk: {size_mb} MB")
        if size_mb > 10_000:
            logger.warning(
                f"DR model is {size_mb:.0f} MB — ensure you have ≥ 48 GB of RAM available."
            )

    # Load model artifacts
    logger.info(f"Loading artifacts for run '{args.run_name}' …")
    artifacts = loader.load_model_artifacts(args.run_name)
    if artifacts is None:
        logger.error("Failed to load model artifacts.")
        return 1
    if artifacts.get("dr_model") == "OOM":
        size_gb = artifacts.get("dr_model_size_gb", "?")
        logger.error(
            f"Out of memory loading the DR model ({size_gb:.1f} GB required). "
            "Run this on a node with ≥ 48 GB RAM, or use the PCA version of this model."
        )
        return 1

    # Load candidates
    candidates_path = Path(args.candidates).expanduser().resolve()
    if not candidates_path.exists():
        logger.error(f"Candidates file not found: {candidates_path}")
        return 1

    logger.info(f"Loading candidates from {candidates_path} …")
    df_candidates = load_candidates(candidates_path)

    if "SMILES" not in df_candidates.columns:
        logger.error("Candidates file must contain a 'SMILES' column.")
        return 1

    logger.info(f"Loaded {len(df_candidates)} candidates.")

    # Check for pre-computed features
    feature_names = loader.get_feature_names_from_scaler(artifacts["scaler"])
    has_precomputed = (
        feature_names is not None
        and all(f in df_candidates.columns for f in feature_names)
    )

    if has_precomputed:
        logger.info("Pre-computed feature columns detected — skipping descriptor calculation.")
    else:
        logger.info("Computing Mordred 2D descriptors …")
        smiles_list = df_candidates["SMILES"].astype(str).tolist()
        calc = DescriptorCalculator(feature_names=feature_names)
        df_candidates = calc.compute_for_smiles_list(smiles_list, include_smiles=True)
        n_failed = df_candidates[calc.feature_names].isna().all(axis=1).sum()
        if n_failed:
            logger.warning(f"{n_failed} molecules failed descriptor computation and will be dropped.")

    # Project and score
    logger.info("Projecting and scoring candidates …")
    projector = CandidateProjector(artifacts)
    df_scored = projector.project_and_score(df_candidates, smiles_col="SMILES")

    if df_scored.empty:
        logger.error("No candidates remained after projection.")
        return 1

    # Write output
    output_path = Path(args.output).expanduser().resolve()
    df_scored.to_csv(output_path, index=False)
    logger.info(
        f"Scored {len(df_scored)} candidates → {output_path}\n"
        f"  Best score:  {df_scored['score'].iloc[0]:.4f}  (rank 1: {df_scored['SMILES'].iloc[0]})\n"
        f"  Worst score: {df_scored['score'].iloc[-1]:.4f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
