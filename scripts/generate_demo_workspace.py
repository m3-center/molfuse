"""
Generate a tiny demo workspace so the GUI can be tested without downloading
the full Zenodo data archive.

Creates:
  demo_workspace/phase1/demo_kinase_UMAP_2d/
      artifacts/  scaler.joblib  umap_model.joblib  embedding_mf.csv
                  embedding_zinc.csv  embedding_actives.csv
      metrics/    metrics.json
      logs/       phase1_summary.json

  demo_candidates.csv   (5 drug-like SMILES + pre-computed Mordred features)

Usage:
  python scripts/generate_demo_workspace.py [--workspace demo_workspace]

Then launch the GUI:
  python -m molfuse.gui --workspace demo_workspace
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from molfuse.io.paths import make_run_dirs
from molfuse.gui.components.descriptor_calc import DescriptorCalculator

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("generate_demo_workspace")

RNG = np.random.default_rng(42)

# Known kinase inhibitors as the "MF cloud"
MF_SMILES = [
    "CC1=NC=C(C(=O)NC2=CC(=CC=C2)C(F)(F)F)C=C1",
    "C1=CC2=C(C=C1)N=CN=C2NC3=CC=C(C=C3)OCC4=CC=CC=C4",
    "CC(C)(C)OC(=O)N1CCN(CC1)C2=NC3=CC=CC=C3N2",
    "COC1=CC2=C(C=C1OC)NC(=O)C2=CC3=CC=CC=C3",
    "CC1=C2C=C(C=CC2=NC(=C1)C3=CC=NC=C3)NC(=O)C4=CC=C(C=C4)CN5CCN(CC5)C",
    "O=C(Nc1ccc(OCc2ccccn2)cc1)c1cnc2ccccc2c1",
    "Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C",
    "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",
    "Nc1nc2ccc(F)cc2c(=O)n1Cc1ccc(F)cc1",
    "O=C(c1cccc(NC(=O)c2ccc(CN3CCNCC3)cc2)c1)Nc1ccc(F)cc1",
    "CC(=O)Nc1ccc(cc1)O",   # Paracetamol — distant from kinase cloud
    "c1ccc2ccccc2c1",        # Naphthalene — very distant
]

# ZINC-like decoys
ZINC_SMILES = [
    "CCCCCCC", "CC(=O)OCC", "CCOC(=O)CC(=O)OCC", "C1CCCCC1",
    "CCCC(=O)O", "CC(C)CO", "OCCO", "CCOCC",
    "CC(N)=O", "CCCCN", "CCC(O)CC", "CCOC(C)=O",
]

# Actives (held-out)
ACTIVE_SMILES = [
    "CC1=NC=C(C(=O)NC2=CC(=CC=C2)C(F)(F)F)C=C1",
    "O=C(Nc1ccc(OCc2ccccn2)cc1)c1cnc2ccccc2c1",
]

# Demo scoring candidates
CANDIDATE_SMILES = [
    "Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C",  # Imatinib — ranks high
    "CC(=O)Nc1ccc(cc1)O",                                            # Paracetamol — ranks low
    "c1ccc2ccccc2c1",                                                 # Naphthalene — ranks low
    "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",
    "Nc1nc2ccc(F)cc2c(=O)n1Cc1ccc(F)cc1",
]


def compute_descriptors(smiles_list: list[str], calc: DescriptorCalculator) -> pd.DataFrame:
    """Compute real Mordred descriptors; drop molecules that fail entirely."""
    df = calc.compute_for_smiles_list(smiles_list, include_smiles=True)
    feat_cols = calc.feature_names
    n_before = len(df)
    df = df.dropna(subset=feat_cols, how="all").reset_index(drop=True)
    if len(df) < n_before:
        logger.warning(f"{n_before - len(df)} molecules dropped (all-NaN descriptors)")
    # Fill column NaNs with column mean; fall back to 0 for fully-NaN columns
    for col in feat_cols:
        col_mean = df[col].mean()
        df[col] = df[col].fillna(0.0 if np.isnan(col_mean) else col_mean)
    return df


def make_embedding(X_scaled: np.ndarray, smiles_list: list[str],
                   dr_model, accessions: list[str] | None = None) -> pd.DataFrame:
    emb = dr_model.transform(X_scaled)
    df = pd.DataFrame(emb, columns=[f"z{i}" for i in range(emb.shape[1])])
    df.insert(0, "SMILES", smiles_list)
    df["accession"] = accessions if accessions else "DEMO"
    df["activity_type"] = "IC50"
    df["activity_value_nM"] = RNG.uniform(1, 1000, size=len(smiles_list))
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default="demo_workspace",
                        help="Workspace directory to create (default: demo_workspace)")
    args = parser.parse_args()

    workspace = Path(args.workspace).expanduser().resolve()
    run_name = "demo_kinase_UMAP_2d"
    dirs = make_run_dirs(workspace, "phase1", run_name)

    logger.info(f"Creating demo workspace at {workspace}")

    # Compute real Mordred descriptors — same calculator the GUI uses at runtime
    logger.info("Computing Mordred descriptors for demo molecules …")
    calc = DescriptorCalculator()
    feat_cols = calc.feature_names

    df_mf     = compute_descriptors(MF_SMILES,        calc)
    df_zinc   = compute_descriptors(ZINC_SMILES,      calc)
    df_act    = compute_descriptors(ACTIVE_SMILES,    calc)
    df_cands  = compute_descriptors(CANDIDATE_SMILES, calc)

    logger.info(f"Feature set: {len(feat_cols)} Mordred descriptors")

    # Drop zero-variance columns (constant features confuse scaler and UMAP)
    X_combined = np.vstack([df_mf[feat_cols].values, df_zinc[feat_cols].values])
    var = X_combined.var(axis=0)
    keep = [c for c, v in zip(feat_cols, var) if v > 0]
    if len(keep) < len(feat_cols):
        logger.info(f"Dropped {len(feat_cols) - len(keep)} zero-variance features; using {len(keep)}")
        feat_cols = keep

    # Fit scaler on MF + ZINC (same convention as full pipeline)
    X_fit = np.vstack([df_mf[feat_cols].values, df_zinc[feat_cols].values])
    scaler = StandardScaler()
    scaler.fit(X_fit)
    # Store feature names so ModelLoader.get_feature_names_from_scaler() works
    scaler.feature_names_in_ = np.array(feat_cols)

    X_mf_scaled   = scaler.transform(df_mf[feat_cols])
    X_zinc_scaled = scaler.transform(df_zinc[feat_cols])
    X_act_scaled  = scaler.transform(df_act[feat_cols])

    # Fit a tiny UMAP
    logger.info("Fitting UMAP (n=2, n_neighbors=4) on demo data …")
    try:
        from umap import UMAP
        umap_model = UMAP(n_components=2, n_neighbors=4, min_dist=0.1,
                          random_state=42, n_epochs=50, verbose=False)
        umap_model.fit(X_mf_scaled)
    except Exception as e:
        logger.error(f"UMAP fitting failed: {e}")
        return 1

    # Build and save embeddings
    df_emb_mf   = make_embedding(X_mf_scaled,   df_mf["SMILES"].tolist(),   umap_model,
                                  accessions=["P00519"] * len(df_mf))
    df_emb_zinc = make_embedding(X_zinc_scaled,  df_zinc["SMILES"].tolist(), umap_model)
    df_emb_act  = make_embedding(X_act_scaled,   df_act["SMILES"].tolist(),  umap_model,
                                  accessions=["P00519"] * len(df_act))

    joblib.dump(scaler,     dirs["artifacts"] / "scaler.joblib")
    joblib.dump(umap_model, dirs["artifacts"] / "umap_model.joblib")
    df_emb_mf.to_csv(  dirs["artifacts"] / "embedding_mf.csv",      index=False)
    df_emb_zinc.to_csv(dirs["artifacts"] / "embedding_zinc.csv",     index=False)
    df_emb_act.to_csv( dirs["artifacts"] / "embedding_actives.csv",  index=False)
    logger.info(f"Artifacts written to {dirs['artifacts']}")

    # metrics.json
    metrics = {
        "ef_at_1_percent": 12.5,
        "roc_auc": 0.84,
        "pr_auc": 0.71,
        "bedroc_alpha20": 0.62,
        "spearman_rho": 0.55,
        "n_actives": len(df_act),
        "n_mf": len(df_mf),
        "n_zinc": len(df_zinc),
    }
    with open(dirs["metrics"] / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # summary JSON (read by ModelLoader)
    summary = {
        "run_name": run_name,
        "target_accession": "P00519",
        "method": "umap",
        "representation": "features",
        "n_components": 2,
        "mf_features_csv": "KW-0808_Transferase_affinity_extracted_features.parquet",
        "molfuse_version": "4.0.0-demo",
    }
    with open(dirs["logs"] / "phase1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary and metrics written.")

    # demo_candidates.csv — include pre-computed Mordred features so the GUI
    # fast-path activates (skips descriptor recalculation on upload)
    cands_path = Path.cwd() / "demo_candidates.csv"
    df_cands.to_csv(cands_path, index=False)
    logger.info(f"Demo candidates written to {cands_path} ({len(feat_cols)} Mordred feature columns included)")

    print(f"\nDemo workspace ready.\n")
    print(f"Launch the GUI:\n  python -m molfuse.gui --workspace {args.workspace}")
    print(f"\nOr score via CLI:")
    print(f"  python scripts/score_candidates.py --workspace {args.workspace} --list")
    print(f"  python scripts/score_candidates.py --workspace {args.workspace} \\")
    print(f"      --run-name {run_name} --candidates demo_candidates.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())

