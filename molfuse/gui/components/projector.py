"""Candidate projector and scorer: Projects candidates through trained model and scores via 1-NN."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


logger = logging.getLogger(__name__)


class CandidateProjector:
    """
    Projects candidate molecules through trained MolFuSE model and scores them.
    
    Workflow:
    1. Align candidate features with model's expected feature set
    2. Scale features using loaded StandardScaler
    3. Transform through DR model (PCA/UMAP)
    4. Score via 1-NN distance to MF cloud in embedded space
    5. Find nearest MF neighbor with metadata
    """
    
    def __init__(self, model_artifacts: Dict):
        """
        Initialize projector with loaded model artifacts.
        
        Args:
            model_artifacts: Dict from ModelLoader.load_model_artifacts()
                Must contain: scaler, dr_model, embedding_mf
        """
        self.scaler = model_artifacts["scaler"]
        self.dr_model = model_artifacts["dr_model"]
        self.embedding_mf = model_artifacts["embedding_mf"]
        self.metadata = model_artifacts["metadata"]
        
        # Extract feature names from scaler
        self.feature_names = None
        if hasattr(self.scaler, 'feature_names_in_'):
            self.feature_names = list(self.scaler.feature_names_in_)
            logger.info(f"Scaler expects {len(self.feature_names)} features")
        else:
            logger.warning("Scaler has no feature_names_in_ attribute; feature alignment may fail")
        
        # Extract embedding coordinate columns
        self.coord_cols = [col for col in self.embedding_mf.columns if col.startswith('z')]
        if not self.coord_cols:
            raise ValueError("No embedding coordinate columns (z0, z1, ...) found in embedding_mf")
        
        logger.info(f"Embedding dimensionality: {len(self.coord_cols)}")
        
        # Prepare MF cloud coordinates for 1-NN
        self.mf_coords = self.embedding_mf[self.coord_cols].values
        logger.info(f"MF cloud size: {len(self.mf_coords)} molecules")
    
    def align_features(self, df_candidates: pd.DataFrame) -> pd.DataFrame:
        """
        Align candidate features with model's expected feature set.
        
        Args:
            df_candidates: DataFrame with descriptor columns
            
        Returns:
            DataFrame with only expected features, in correct order
            Missing features filled with NaN
        """
        if self.feature_names is None:
            logger.warning("No feature names available from scaler; returning candidates as-is")
            # Return all numeric columns except metadata
            metadata_cols = ['SMILES', 'Compound ChEMBL ID', 'accession', 'Activity Type', 'Standard Value (nM)']
            numeric_cols = [col for col in df_candidates.columns 
                          if col not in metadata_cols and pd.api.types.is_numeric_dtype(df_candidates[col])]
            return df_candidates[numeric_cols]
        
        # Check which expected features are present
        missing_features = [f for f in self.feature_names if f not in df_candidates.columns]
        extra_features = [f for f in df_candidates.columns if f not in self.feature_names and f not in ['SMILES', 'Compound ChEMBL ID']]
        
        if missing_features:
            logger.warning(f"Candidates missing {len(missing_features)} expected features (will be filled with NaN)")
            logger.debug(f"Missing features: {missing_features[:10]}...")
        
        if extra_features:
            logger.info(f"Dropping {len(extra_features)} extra features not used by model")
        
        # Create aligned DataFrame
        df_aligned = pd.DataFrame()
        for feat in self.feature_names:
            if feat in df_candidates.columns:
                df_aligned[feat] = df_candidates[feat]
            else:
                df_aligned[feat] = np.nan
        
        return df_aligned
    
    def project_candidates(
        self,
        df_candidates: pd.DataFrame,
        smiles_col: str = "SMILES"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Project candidates through trained model.
        
        Args:
            df_candidates: DataFrame with descriptor columns (and optional SMILES/metadata)
            smiles_col: Name of SMILES column (for preservation in output)
            
        Returns:
            Tuple of (df_embedding, df_metadata):
            - df_embedding: DataFrame with columns [SMILES] + [z0, z1, ...]
            - df_metadata: Original metadata columns (SMILES, ChEMBL ID, etc.)
        """
        # Preserve metadata columns
        metadata_cols = ["SMILES", "Compound ChEMBL ID"]
        df_metadata = df_candidates[[col for col in metadata_cols if col in df_candidates.columns]].copy()
        
        # Align features
        df_features = self.align_features(df_candidates)
        
        # Drop rows with all NaN features (failed descriptor computation)
        n_before = len(df_features)
        valid_mask = ~df_features.isna().all(axis=1)
        df_features = df_features[valid_mask]
        df_metadata = df_metadata[valid_mask]
        n_after = len(df_features)
        
        if n_after < n_before:
            logger.warning(f"Dropped {n_before - n_after} candidates with all NaN features")
        
        if len(df_features) == 0:
            logger.error("No valid candidates remaining after feature alignment")
            return pd.DataFrame(), df_metadata
        
        # Check for remaining NaNs
        n_with_nan = df_features.isna().any(axis=1).sum()
        if n_with_nan > 0:
            logger.warning(
                f"{n_with_nan} candidates have partial NaN features "
                "(will be imputed by scaler if it includes imputation, otherwise may cause errors)"
            )
        
        # Scale features
        try:
            X_scaled = self.scaler.transform(df_features.values)
            logger.info(f"Scaled {len(X_scaled)} candidate feature vectors")
        except Exception as e:
            logger.error(f"Failed to scale features: {e}")
            raise
        
        # Project through DR model
        try:
            Z_candidates = self.dr_model.transform(X_scaled)
            logger.info(f"Projected candidates to {Z_candidates.shape[1]}D embedding")
        except Exception as e:
            logger.error(f"Failed to project through DR model: {e}")
            raise
        
        # Create embedding DataFrame
        coord_cols = [f"z{i}" for i in range(Z_candidates.shape[1])]
        df_embedding = pd.DataFrame(Z_candidates, columns=coord_cols)
        
        # Add SMILES if available
        if smiles_col in df_metadata.columns:
            df_embedding.insert(0, "SMILES", df_metadata[smiles_col].values)
        
        return df_embedding, df_metadata
    
    def score_candidates(
        self,
        df_embedding: pd.DataFrame,
        df_metadata: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Score candidates via 1-NN distance to MF cloud.
        
        Args:
            df_embedding: DataFrame from project_candidates (with z0, z1, ... columns)
            df_metadata: Metadata DataFrame from project_candidates
            
        Returns:
            DataFrame with columns:
            - Original metadata (SMILES, etc.)
            - score: -distance (higher is better)
            - distance: min distance to MF cloud
            - rank: rank by score (1 = best)
            - nearest_mf_smiles: SMILES of nearest MF neighbor
            - nearest_mf_distance: distance to nearest neighbor
            - nearest_mf_activity_type: Activity type of nearest neighbor (if available)
            - nearest_mf_activity_value_nM: Activity value of nearest neighbor (if available)
            - nearest_mf_accession: Target accession of nearest neighbor (if available)
        """
        # Extract coordinates
        coord_cols = [col for col in df_embedding.columns if col.startswith('z')]
        Z_candidates = df_embedding[coord_cols].values
        
        if len(Z_candidates) == 0:
            logger.error("No candidate coordinates to score")
            return pd.DataFrame()
        
        # Fit 1-NN on MF cloud
        try:
            nn = NearestNeighbors(n_neighbors=1, metric='euclidean', algorithm='auto', n_jobs=-1)
            nn.fit(self.mf_coords)
            logger.info(f"Fitted 1-NN on MF cloud ({len(self.mf_coords)} molecules)")
        except Exception as e:
            logger.error(f"Failed to fit 1-NN: {e}")
            raise
        
        # Query nearest neighbors
        try:
            distances, indices = nn.kneighbors(Z_candidates, return_distance=True)
            distances = distances.reshape(-1)
            indices = indices.reshape(-1)
            scores = -distances
            logger.info(f"Scored {len(scores)} candidates")
        except Exception as e:
            logger.error(f"Failed to score candidates: {e}")
            raise
        
        # Build results DataFrame
        df_results = df_metadata.copy()
        df_results['score'] = scores
        df_results['distance'] = distances
        
        # Add embedding coordinates
        for col in coord_cols:
            df_results[col] = df_embedding[col].values
        
        # Add nearest neighbor info
        nearest_mf_smiles = []
        nearest_mf_activity_type = []
        nearest_mf_activity_value = []
        nearest_mf_accession = []
        
        for idx in indices:
            mf_row = self.embedding_mf.iloc[idx]
            nearest_mf_smiles.append(mf_row.get('SMILES', 'N/A'))
            nearest_mf_activity_type.append(mf_row.get('Activity Type', 'N/A'))
            nearest_mf_activity_value.append(mf_row.get('Standard Value (nM)', np.nan))
            nearest_mf_accession.append(mf_row.get('accession', 'N/A'))
        
        df_results['nearest_mf_smiles'] = nearest_mf_smiles
        df_results['nearest_mf_activity_type'] = nearest_mf_activity_type
        df_results['nearest_mf_activity_value_nM'] = nearest_mf_activity_value
        df_results['nearest_mf_accession'] = nearest_mf_accession
        
        # Rank by score
        df_results = df_results.sort_values('score', ascending=False).reset_index(drop=True)
        df_results['rank'] = np.arange(1, len(df_results) + 1)
        
        logger.info(
            f"Ranked {len(df_results)} candidates. "
            f"Best score: {df_results['score'].iloc[0]:.4f}, "
            f"Worst score: {df_results['score'].iloc[-1]:.4f}"
        )
        
        return df_results
    
    def project_and_score(
        self,
        df_candidates: pd.DataFrame,
        smiles_col: str = "SMILES"
    ) -> pd.DataFrame:
        """
        Convenience method: project and score in one call.
        
        Args:
            df_candidates: DataFrame with descriptor columns
            smiles_col: Name of SMILES column
            
        Returns:
            DataFrame with scored and ranked candidates
        """
        df_embedding, df_metadata = self.project_candidates(df_candidates, smiles_col)
        
        if len(df_embedding) == 0:
            logger.error("No candidates to score after projection")
            return pd.DataFrame()
        
        df_results = self.score_candidates(df_embedding, df_metadata)
        return df_results
