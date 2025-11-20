"""Model loader: Scans workspace and loads MolFuSE model artifacts."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Scans workspace for trained MolFuSE models and provides loading utilities.
    
    Validates model artifacts (scaler, DR model, embeddings, metrics) and
    extracts metadata for model selection.
    """
    
    def __init__(self, workspace_dir: Path):
        """
        Initialize model loader.
        
        Args:
            workspace_dir: Path to experiment workspace (contains phase1/, phase2/, etc.)
        """
        self.workspace_dir = Path(workspace_dir)
        self.available_models: List[Dict] = []
        
    def scan_models(self, phases: List[str] = ["phase1"]) -> List[Dict]:
        """
        Scan workspace for available models.
        
        Args:
            phases: List of phase directories to scan (e.g., ["phase1", "phase4"])
            
        Returns:
            List of model metadata dicts with keys:
            - run_name: str
            - phase: str
            - target_accession: str (from config or inferred)
            - molecular_function: str (KW category)
            - method: "pca" or "umap"
            - representation: "features" or "fingerprints"
            - n_components: int
            - ef_at_1: float (EF@1% from metrics)
            - artifacts_dir: Path
            - valid: bool (all required artifacts present)
        """
        models = []
        
        for phase in phases:
            phase_dir = self.workspace_dir / phase
            if not phase_dir.exists():
                logger.warning(f"Phase directory not found: {phase_dir}")
                continue
                
            # Iterate through run directories
            for run_dir in phase_dir.iterdir():
                if not run_dir.is_dir():
                    continue
                    
                run_name = run_dir.name
                artifacts_dir = run_dir / "artifacts"
                logs_dir = run_dir / "logs"
                metrics_path = run_dir / "metrics" / "metrics.json"
                summary_path = logs_dir / f"{phase}_summary.json"
                
                # Skip if missing critical directories
                if not artifacts_dir.exists():
                    continue
                
                # Load metadata from summary JSON
                metadata = {
                    "run_name": run_name,
                    "phase": phase,
                    "target_accession": "Unknown",
                    "molecular_function": "Unknown",
                    "method": "unknown",
                    "representation": "unknown",
                    "n_components": 0,
                    "ef_at_1": float('nan'),
                    "artifacts_dir": artifacts_dir,
                    "valid": False
                }
                
                # Try to load summary JSON
                if summary_path.exists():
                    try:
                        with open(summary_path, 'r') as f:
                            summary = json.load(f)
                            metadata["target_accession"] = summary.get("target_accession", "Unknown")
                            metadata["method"] = summary.get("method", "unknown")
                            metadata["representation"] = summary.get("representation", "unknown")
                            metadata["n_components"] = summary.get("n_components", 0)
                    except Exception as e:
                        logger.warning(f"Failed to load summary for {run_name}: {e}")
                
                # Infer molecular function from MF features CSV path if available
                if summary_path.exists():
                    try:
                        with open(summary_path, 'r') as f:
                            summary = json.load(f)
                            mf_csv = summary.get("mf_features_csv", "")
                            if "KW-" in mf_csv:
                                # Extract KW-XXXX_Name pattern
                                parts = mf_csv.split("KW-")
                                if len(parts) > 1:
                                    kw_part = parts[1].split("_affinity")[0]
                                    metadata["molecular_function"] = f"KW-{kw_part}"
                    except Exception:
                        pass
                
                # Load metrics
                if metrics_path.exists():
                    try:
                        with open(metrics_path, 'r') as f:
                            metrics = json.load(f)
                            metadata["ef_at_1"] = metrics.get("ef_at_1_percent", float('nan'))
                    except Exception as e:
                        logger.warning(f"Failed to load metrics for {run_name}: {e}")
                
                # Validate artifacts
                required_artifacts = [
                    artifacts_dir / "scaler.joblib",
                    artifacts_dir / "embedding_mf.csv"
                ]
                
                # Check for DR model (PCA or UMAP)
                if (artifacts_dir / "pca_model.joblib").exists():
                    required_artifacts.append(artifacts_dir / "pca_model.joblib")
                elif (artifacts_dir / "umap_model.joblib").exists():
                    required_artifacts.append(artifacts_dir / "umap_model.joblib")
                
                metadata["valid"] = all(f.exists() for f in required_artifacts)
                
                models.append(metadata)
        
        self.available_models = models
        logger.info(f"Found {len(models)} models ({sum(m['valid'] for m in models)} valid)")
        return models
    
    def get_models_by_kw(self, kw_category: str) -> List[Dict]:
        """
        Filter models by molecular function (KW category).
        
        Args:
            kw_category: Molecular function keyword (e.g., "KW-0505_Motor_protein")
            
        Returns:
            List of model metadata dicts matching the KW category
        """
        return [m for m in self.available_models if kw_category in m["molecular_function"]]
    
    def get_kw_categories(self) -> List[str]:
        """Get list of unique KW categories from available models."""
        kw_cats = set(m["molecular_function"] for m in self.available_models if m["molecular_function"] != "Unknown")
        return sorted(list(kw_cats))
    
    def load_model_artifacts(self, run_name: str) -> Optional[Dict]:
        """
        Load all artifacts for a specific model run.
        
        Args:
            run_name: Name of the run to load
            
        Returns:
            Dict with keys:
            - scaler: StandardScaler
            - dr_model: PCA or UMAP model
            - embedding_mf: DataFrame (MF embedding with metadata)
            - embedding_zinc: DataFrame (ZINC embedding, if available)
            - embedding_actives: DataFrame (actives embedding, if available)
            - metrics: Dict (metrics.json contents)
            - config: Dict (reconstruction from summary.json)
            Or None if model not found or invalid
        """
        # Find model by run_name
        model_meta = None
        for m in self.available_models:
            if m["run_name"] == run_name:
                model_meta = m
                break
        
        if not model_meta or not model_meta["valid"]:
            logger.error(f"Model {run_name} not found or invalid")
            return None
        
        artifacts_dir = model_meta["artifacts_dir"]
        run_dir = artifacts_dir.parent
        
        artifacts = {
            "metadata": model_meta,
            "scaler": None,
            "dr_model": None,
            "embedding_mf": None,
            "embedding_zinc": None,
            "embedding_actives": None,
            "metrics": {},
            "config": {}
        }
        
        # Load scaler
        try:
            artifacts["scaler"] = joblib.load(artifacts_dir / "scaler.joblib")
            logger.info(f"Loaded scaler for {run_name}")
        except Exception as e:
            logger.error(f"Failed to load scaler: {e}")
            return None
        
        # Load DR model
        dr_model_path = None
        if (artifacts_dir / "pca_model.joblib").exists():
            dr_model_path = artifacts_dir / "pca_model.joblib"
        elif (artifacts_dir / "umap_model.joblib").exists():
            dr_model_path = artifacts_dir / "umap_model.joblib"
        
        if dr_model_path:
            try:
                artifacts["dr_model"] = joblib.load(dr_model_path)
                logger.info(f"Loaded DR model from {dr_model_path.name}")
            except Exception as e:
                logger.error(f"Failed to load DR model: {e}")
                return None
        else:
            logger.error("No DR model found")
            return None
        
        # Load embeddings
        for embed_name in ["embedding_mf", "embedding_zinc", "embedding_actives"]:
            embed_path = artifacts_dir / f"{embed_name}.csv"
            if embed_path.exists():
                try:
                    df = pd.read_csv(embed_path)
                    artifacts[embed_name] = df
                    logger.info(f"Loaded {embed_name}: {len(df)} rows")
                except Exception as e:
                    logger.warning(f"Failed to load {embed_name}: {e}")
        
        # Load metrics
        metrics_path = run_dir / "metrics" / "metrics.json"
        if metrics_path.exists():
            try:
                with open(metrics_path, 'r') as f:
                    artifacts["metrics"] = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load metrics: {e}")
        
        # Load config reconstruction from summary
        summary_path = run_dir / "logs" / f"{model_meta['phase']}_summary.json"
        if summary_path.exists():
            try:
                with open(summary_path, 'r') as f:
                    artifacts["config"] = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load config: {e}")
        
        return artifacts
    
    def get_feature_names_from_scaler(self, scaler) -> Optional[List[str]]:
        """
        Extract feature names from scaler if available.
        
        Args:
            scaler: StandardScaler instance
            
        Returns:
            List of feature names or None if not available
        """
        if hasattr(scaler, 'feature_names_in_'):
            return list(scaler.feature_names_in_)
        return None
