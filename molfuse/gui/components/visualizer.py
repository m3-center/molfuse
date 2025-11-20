"""Visualizer: Creates plots and molecule renderings for GUI."""

from __future__ import annotations

import base64
from io import BytesIO
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from rdkit import Chem
from rdkit.Chem import Draw
from sklearn.decomposition import PCA


logger = logging.getLogger(__name__)


class Visualizer:
    """
    Creates visualizations for MolFuSE GUI:
    - 2D scatter plots of embedding space
    - Score distribution histograms
    - ROC/PR curves
    - Molecule structure images
    """
    
    @staticmethod
    def mol_to_image_base64(smiles: str, size: Tuple[int, int] = (200, 200)) -> Optional[str]:
        """
        Convert SMILES to base64-encoded PNG image.
        
        Args:
            smiles: SMILES string
            size: Image size (width, height)
            
        Returns:
            Base64-encoded PNG data URI or None if rendering fails
        """
        if not smiles or pd.isna(smiles):
            return None
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        try:
            Chem.rdDepictor.Compute2DCoords(mol)
            img = Draw.MolToImage(mol, size=size)
            buffer = BytesIO()
            img.save(buffer, format="PNG")
            encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return f"data:image/png;base64,{encoded}"
        except Exception as e:
            logger.warning(f"Failed to render SMILES '{smiles}': {e}")
            return None
    
    @staticmethod
    def project_to_2d(
        embedding: np.ndarray,
        method: str = "pca"
    ) -> np.ndarray:
        """
        Project high-dimensional embedding to 2D for visualization.
        
        Args:
            embedding: Array of shape (n_samples, n_dims)
            method: "pca" or "first2" (just use first 2 dimensions)
            
        Returns:
            Array of shape (n_samples, 2)
        """
        if embedding.shape[1] == 2:
            return embedding
        
        if method == "first2":
            return embedding[:, :2]
        
        elif method == "pca":
            pca = PCA(n_components=2, random_state=42)
            return pca.fit_transform(embedding)
        
        else:
            raise ValueError(f"Unknown projection method: {method}")
    
    def create_embedding_scatter(
        self,
        df_mf: pd.DataFrame,
        df_zinc: Optional[pd.DataFrame] = None,
        df_actives: Optional[pd.DataFrame] = None,
        df_candidates: Optional[pd.DataFrame] = None,
        coord_cols: Optional[List[str]] = None,
        projection_method: str = "first2",
        title: str = "Molecular Function Similarity Space"
    ) -> go.Figure:
        """
        Create 2D scatter plot of embedding space.
        
        Args:
            df_mf: MF embedding DataFrame
            df_zinc: Optional ZINC embedding DataFrame
            df_actives: Optional actives embedding DataFrame
            df_candidates: Optional candidates embedding DataFrame (scored)
            coord_cols: List of coordinate column names (e.g., ['z0', 'z1', ...])
            projection_method: "first2" (use z0, z1) or "pca" (project from higher dims)
            title: Plot title
            
        Returns:
            Plotly Figure object
        """
        if coord_cols is None:
            coord_cols = [col for col in df_mf.columns if col.startswith('z')]
        
        if len(coord_cols) < 2:
            raise ValueError(f"Need at least 2 coordinate columns, got {len(coord_cols)}")
        
        # Extract coordinates and project to 2D if needed
        def get_2d_coords(df):
            if df is None or len(df) == 0:
                return None
            coords = df[coord_cols].values
            if coords.shape[1] > 2:
                coords = self.project_to_2d(coords, method=projection_method)
            return coords
        
        mf_coords = get_2d_coords(df_mf)
        zinc_coords = get_2d_coords(df_zinc)
        actives_coords = get_2d_coords(df_actives)
        candidates_coords = get_2d_coords(df_candidates)
        
        # Create figure
        fig = go.Figure()
        
        # Plot MF cloud (blue, small, semi-transparent)
        if mf_coords is not None:
            fig.add_trace(go.Scattergl(
                x=mf_coords[:, 0],
                y=mf_coords[:, 1],
                mode='markers',
                marker=dict(size=4, color='blue', opacity=0.3),
                name='MF Cloud',
                hoverinfo='skip'
            ))
        
        # Plot ZINC (purple, small, semi-transparent)
        if zinc_coords is not None:
            fig.add_trace(go.Scattergl(
                x=zinc_coords[:, 0],
                y=zinc_coords[:, 1],
                mode='markers',
                marker=dict(size=4, color='purple', opacity=0.2),
                name='ZINC',
                hoverinfo='skip'
            ))
        
        # Plot actives (red, medium, opaque)
        if actives_coords is not None:
            hover_text = []
            for _, row in df_actives.iterrows():
                smiles = row.get('SMILES', 'N/A')
                activity_val = row.get('Standard Value (nM)', np.nan)
                if pd.notna(activity_val):
                    hover_text.append(f"Active<br>SMILES: {smiles}<br>Activity: {activity_val:.1f} nM")
                else:
                    hover_text.append(f"Active<br>SMILES: {smiles}")
            
            fig.add_trace(go.Scattergl(
                x=actives_coords[:, 0],
                y=actives_coords[:, 1],
                mode='markers',
                marker=dict(size=8, color='red', opacity=0.8),
                name='Actives',
                text=hover_text,
                hoverinfo='text'
            ))
        
        # Plot candidates (green, large, opaque, with scores)
        if candidates_coords is not None and 'score' in df_candidates.columns:
            hover_text = []
            for _, row in df_candidates.iterrows():
                smiles = row.get('SMILES', 'N/A')
                score = row.get('score', np.nan)
                rank = row.get('rank', 'N/A')
                distance = row.get('distance', np.nan)
                hover_text.append(
                    f"Candidate (Rank {rank})<br>"
                    f"SMILES: {smiles}<br>"
                    f"Score: {score:.4f}<br>"
                    f"Distance to MF: {distance:.4f}"
                )
            
            fig.add_trace(go.Scattergl(
                x=candidates_coords[:, 0],
                y=candidates_coords[:, 1],
                mode='markers',
                marker=dict(
                    size=12,
                    color=df_candidates['score'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Score"),
                    line=dict(color='black', width=1)
                ),
                name='Candidates',
                text=hover_text,
                hoverinfo='text'
            ))
        
        # Layout
        projection_note = "(2D PCA projection)" if projection_method == "pca" and len(coord_cols) > 2 else ""
        fig.update_layout(
            title=f"{title} {projection_note}",
            xaxis_title="z0" if projection_method == "first2" else "PC1",
            yaxis_title="z1" if projection_method == "first2" else "PC2",
            template='plotly_white',
            hovermode='closest',
            legend=dict(x=1.02, y=1, xanchor='left', yanchor='top'),
            width=900,
            height=700,
            xaxis=dict(scaleanchor="y", scaleratio=1)
        )
        
        return fig
    
    @staticmethod
    def create_score_distribution(
        df_candidates: pd.DataFrame,
        df_actives: Optional[pd.DataFrame] = None,
        df_zinc: Optional[pd.DataFrame] = None,
        title: str = "Score Distribution"
    ) -> go.Figure:
        """
        Create histogram of score distributions.
        
        Args:
            df_candidates: Scored candidates DataFrame
            df_actives: Optional scored actives DataFrame
            df_zinc: Optional scored ZINC DataFrame
            title: Plot title
            
        Returns:
            Plotly Figure object
        """
        fig = go.Figure()
        
        # Candidates
        if 'score' in df_candidates.columns:
            fig.add_trace(go.Histogram(
                x=df_candidates['score'],
                name='Candidates',
                opacity=0.7,
                marker_color='green',
                nbinsx=50
            ))
        
        # Actives
        if df_actives is not None and 'score' in df_actives.columns:
            fig.add_trace(go.Histogram(
                x=df_actives['score'],
                name='Actives',
                opacity=0.7,
                marker_color='red',
                nbinsx=50
            ))
        
        # ZINC
        if df_zinc is not None and 'score' in df_zinc.columns:
            fig.add_trace(go.Histogram(
                x=df_zinc['score'],
                name='ZINC',
                opacity=0.5,
                marker_color='purple',
                nbinsx=50
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Score (-distance to MF)",
            yaxis_title="Count",
            template='plotly_white',
            barmode='overlay',
            legend=dict(x=1.02, y=1, xanchor='left', yanchor='top'),
            width=900,
            height=400
        )
        
        return fig
    
    @staticmethod
    def create_roc_pr_curves(
        scores: np.ndarray,
        labels: np.ndarray,
        title_prefix: str = ""
    ) -> go.Figure:
        """
        Create ROC and PR curves side-by-side.
        
        Args:
            scores: Prediction scores (higher = more likely active)
            labels: Binary labels (1 = active, 0 = decoy)
            title_prefix: Prefix for plot title
            
        Returns:
            Plotly Figure with 2 subplots
        """
        from sklearn.metrics import roc_curve, precision_recall_curve, auc
        
        # ROC curve
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc_val = auc(fpr, tpr)
        
        # PR curve
        precision, recall, _ = precision_recall_curve(labels, scores)
        pr_auc_val = auc(recall, precision)
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                f"ROC Curve (AUC = {roc_auc_val:.3f})",
                f"PR Curve (AUC = {pr_auc_val:.3f})"
            )
        )
        
        # ROC
        fig.add_trace(
            go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random',
                      line=dict(color='gray', dash='dash')),
            row=1, col=1
        )
        
        # PR
        fig.add_trace(
            go.Scatter(x=recall, y=precision, mode='lines', name='PR', line=dict(color='green')),
            row=1, col=2
        )
        
        # Layout
        fig.update_xaxes(title_text="False Positive Rate", row=1, col=1)
        fig.update_yaxes(title_text="True Positive Rate", row=1, col=1)
        fig.update_xaxes(title_text="Recall", row=1, col=2)
        fig.update_yaxes(title_text="Precision", row=1, col=2)
        
        fig.update_layout(
            title_text=f"{title_prefix}ROC and PR Curves",
            template='plotly_white',
            showlegend=False,
            width=1200,
            height=500
        )
        
        return fig
