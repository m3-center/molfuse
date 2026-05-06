"""
DEPRECATED — do not use.

This script required local raw dataset files and is no longer maintained.

To try the GUI:
  Option A: Download the pre-trained model workspace from Zenodo and point
            the GUI at it:
              python -m molfuse.gui --workspace /path/to/downloaded/workspace

  Option B: Train a model on your own data using the GUI training tab, or
            via the CLI:
              python -m molfuse.cli.phase1 --config <config.json> \\
                  --workspace ./my_workspace

See README.md for full instructions.
"""

import sys
print(__doc__)
sys.exit(0)

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from molfuse.data.prep import select_feature_columns, fit_scaler_on_mf_zinc
from molfuse.dr.pca import fit_pca
from molfuse.dr.umap_ import fit_umap


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Generate demo workspace."""
    
    # Paths
    base_dir = Path(__file__).parent.parent
    demo_dir = base_dir / "demo"
    demo_workspace = demo_dir / "workspace"
    demo_data = demo_dir / "data"
    
    # Create directories
    demo_dir.mkdir(exist_ok=True)
    demo_workspace.mkdir(exist_ok=True)
    demo_data.mkdir(exist_ok=True)
    
    # Source data
    source_features = base_dir / "datasets" / "molecular_function_features_fingerprints" / "KW-0049_Antioxidant_affinity_extracted_features.csv"
    
    if not source_features.exists():
        logger.error(f"Source data not found: {source_features}")
        return
    
    logger.info(f"Loading source data from {source_features}")
    df_full = pd.read_csv(source_features, low_memory=False)
    logger.info(f"Loaded {len(df_full)} rows")
    
    # Target accession for actives
    target_accession = "P00441"
    
    # Separate MF (excluding target) and actives (target only)
    df_mf = df_full[df_full['accession'] != target_accession].copy()
    df_actives = df_full[df_full['accession'] == target_accession].copy()
    
    logger.info(f"MF cloud: {len(df_mf)} molecules")
    logger.info(f"Actives: {len(df_actives)} molecules")
    
    # Subsample for demo
    N_MF = min(1000, len(df_mf))
    N_ACTIVES = min(50, len(df_actives))
    N_ZINC = 10000
    
    df_mf_sample = df_mf.sample(n=N_MF, random_state=42).copy()
    df_actives_sample = df_actives.sample(n=N_ACTIVES, random_state=42).copy()
    
    # Create synthetic ZINC (random feature vectors for demo)
    logger.info("Creating synthetic ZINC data...")
    feature_cols = select_feature_columns(df_mf)
    
    # Use mean/std from MF to generate synthetic ZINC
    mf_features = df_mf_sample[feature_cols]
    feature_means = mf_features.mean()
    feature_stds = mf_features.std()
    
    np.random.seed(42)
    zinc_features = np.random.normal(
        loc=feature_means.values,
        scale=feature_stds.values * 1.5,  # Slightly wider distribution
        size=(N_ZINC, len(feature_cols))
    )
    
    df_zinc = pd.DataFrame(zinc_features, columns=feature_cols)
    df_zinc['SMILES'] = [f"ZINC_DEMO_{i}" for i in range(N_ZINC)]
    df_zinc['Compound ChEMBL ID'] = df_zinc['SMILES']
    
    logger.info(f"Created {len(df_zinc)} synthetic ZINC molecules")
    
    # Combine MF + ZINC for training
    df_train = pd.concat([df_mf_sample[feature_cols], df_zinc[feature_cols]], ignore_index=True)
    
    # Fit scaler
    logger.info("Fitting scaler...")
    scaler = StandardScaler()
    
    # Create proper DataFrame for fitting (ensures feature_names_in_ is set)
    df_train_features = df_train[feature_cols].copy()
    X_train = df_train_features.values
    X_train_scaled = scaler.fit_transform(df_train_features)
    
    # Split back
    X_mf_scaled = X_train_scaled[:N_MF]
    X_zinc_scaled = X_train_scaled[N_MF:]
    
    # Transform actives
    df_actives_features = df_actives_sample[feature_cols].copy()
    X_actives_scaled = scaler.transform(df_actives_features)
    
    # Fit UMAP model
    logger.info("Fitting UMAP (2D)...")
    umap_model, Z_train_umap = fit_umap(
        X_train_scaled,
        n_components=2,
        n_neighbors=10,
        min_dist=0.01,
        metric='euclidean'
    )
    
    # Split back the training embeddings
    Z_mf_umap = Z_train_umap[:N_MF]
    Z_zinc_umap = Z_train_umap[N_MF:]
    
    # Transform actives
    Z_actives_umap = umap_model.transform(X_actives_scaled)
    
    # Create Phase 1 run directory
    run_name = "demo_antioxidant_umap_2d"
    run_dir = demo_workspace / "phase1" / run_name
    artifacts_dir = run_dir / "artifacts"
    logs_dir = run_dir / "logs"
    metrics_dir = run_dir / "metrics"
    
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(exist_ok=True)
    metrics_dir.mkdir(exist_ok=True)
    
    # Save models
    logger.info(f"Saving artifacts to {artifacts_dir}")
    joblib.dump(scaler, artifacts_dir / "scaler.joblib")
    joblib.dump(umap_model, artifacts_dir / "umap_model.joblib")
    
    # Save embeddings
    df_embed_mf = df_mf_sample[['SMILES', 'Compound ChEMBL ID']].copy()
    df_embed_mf['z0'] = Z_mf_umap[:, 0]
    df_embed_mf['z1'] = Z_mf_umap[:, 1]
    if 'Activity Type' in df_mf_sample.columns:
        df_embed_mf['Activity Type'] = df_mf_sample['Activity Type'].values
    if 'Standard Value (nM)' in df_mf_sample.columns:
        df_embed_mf['Standard Value (nM)'] = df_mf_sample['Standard Value (nM)'].values
    if 'accession' in df_mf_sample.columns:
        df_embed_mf['accession'] = df_mf_sample['accession'].values
    
    df_embed_zinc = df_zinc[['SMILES', 'Compound ChEMBL ID']].copy()
    df_embed_zinc['z0'] = Z_zinc_umap[:, 0]
    df_embed_zinc['z1'] = Z_zinc_umap[:, 1]
    
    df_embed_actives = df_actives_sample[['SMILES', 'Compound ChEMBL ID']].copy()
    df_embed_actives['z0'] = Z_actives_umap[:, 0]
    df_embed_actives['z1'] = Z_actives_umap[:, 1]
    if 'Activity Type' in df_actives_sample.columns:
        df_embed_actives['Activity Type'] = df_actives_sample['Activity Type'].values
    if 'Standard Value (nM)' in df_actives_sample.columns:
        df_embed_actives['Standard Value (nM)'] = df_actives_sample['Standard Value (nM)'].values
    if 'accession' in df_actives_sample.columns:
        df_embed_actives['accession'] = df_actives_sample['accession'].values
    
    df_embed_mf.to_csv(artifacts_dir / "embedding_mf.csv", index=False)
    df_embed_zinc.to_csv(artifacts_dir / "embedding_zinc.csv", index=False)
    df_embed_actives.to_csv(artifacts_dir / "embedding_actives.csv", index=False)
    
    # Create run summary
    summary = {
        "run_name": run_name,
        "target_accession": target_accession,
        "method": "umap",
        "representation": "features",
        "n_components": 2,
        "n_neighbors": 15,
        "min_dist": 0.1,
        "metric": "euclidean",
        "mf_features_csv": str(source_features),
        "feature_columns": feature_cols,  # Save feature list
        "n_mf": N_MF,
        "n_zinc": N_ZINC,
        "n_actives": N_ACTIVES
    }
    
    with open(logs_dir / "phase1_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Create dummy metrics
    metrics = {
        "ef_at_1_percent": 5.0,
        "roc_auc": 0.75,
        "pr_auc": 0.65
    }
    
    with open(metrics_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Create example candidates CSV WITH FEATURES
    logger.info("Creating example candidates CSV with pre-computed features...")
    
    # Sample a few real molecules from actives as examples
    df_candidates_sample = df_actives.sample(n=min(10, len(df_actives)), random_state=42)
    
    # Include SMILES + all feature columns (so GUI doesn't need to compute descriptors)
    candidate_cols = ['SMILES', 'Compound ChEMBL ID'] + feature_cols
    df_candidates = df_candidates_sample[candidate_cols].copy()
    
    df_candidates.to_csv(demo_data / "example_candidates_with_features.csv", index=False)
    
    # Also create a SMILES-only version for testing descriptor computation
    df_candidates_smiles_only = df_candidates_sample[['SMILES']].copy()
    df_candidates_smiles_only.to_csv(demo_data / "example_candidates.csv", index=False)
    
    logger.info("=" * 60)
    logger.info("Demo case generation complete!")
    logger.info("=" * 60)
    logger.info(f"Workspace: {demo_workspace}")
    logger.info(f"Example candidates (with features): {demo_data / 'example_candidates_with_features.csv'}")
    logger.info(f"Example candidates (SMILES only): {demo_data / 'example_candidates.csv'}")
    logger.info("")
    logger.info("To launch GUI with demo:")
    logger.info(f"  python -m molfuse.gui.app --workspace {demo_workspace}")
    logger.info("")
    logger.info("Then:")
    logger.info("  1. Select KW category: KW-0049_Antioxidant")
    logger.info("  2. Select model: demo_antioxidant_pca_2d")
    logger.info(f"  3. Upload {demo_data / 'example_candidates_with_features.csv'}")
    logger.info("     (Pre-computed features; GUI will skip descriptor calculation)")
    logger.info("  4. Click 'Score Candidates'")
    logger.info("")
    logger.info("Note: This demo uses Mordred descriptors (from source data).")
    logger.info("      The GUI's RDKit descriptor calculator is for different feature sets.")
    logger.info("      For production, ensure candidate features match model's training features.")


if __name__ == "__main__":
    main()
