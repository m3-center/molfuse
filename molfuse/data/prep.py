from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


NON_FEATURE_COLUMNS = {
    "Compound ChEMBL ID",
    "canonical_smiles",
    "SMILES",
    "pActivity",
    "Standard Value (nM)",
    "Target Accession",
    "accession",
}

# Metadata columns for selective dtype specification during CSV loading
METADATA_COLUMNS = {
    'Compound ChEMBL ID',
    'SMILES',
    'Target ChEMBL ID',
    'Target Name',
    'Activity Type',
    'target_chembl_id',
    'accession',
}


def select_feature_columns(df: pd.DataFrame, min_numeric_fraction: float = 0.95) -> List[str]:
    """
    Select feature columns by:
    - Excluding known non-feature columns
    - Including columns with numeric dtype
    - Including object/mixed-type columns if they become numeric after coercion for
      at least `min_numeric_fraction` of their non-null entries.

    This makes selection robust to pandas mixed-type inference in CSVs.
    
    NOTE: When using optimized CSV loading (selective dtype + defensive conversion),
    all feature columns should already be numeric by the time this is called.
    The coercion logic is kept for backward compatibility with unoptimized loading.
    """
    candidates = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    numeric: List[str] = []
    for c in candidates:
        s = df[c]
        # Fast path: already numeric dtype (expected when using optimized loading)
        if pd.api.types.is_numeric_dtype(s):
            numeric.append(c)
            continue
        # Robust path: attempt numeric coercion and accept if mostly numeric
        coerced = pd.to_numeric(s, errors="coerce")
        nonnull = int(s.notna().sum())
        if nonnull == 0:
            continue
        numeric_count = int(coerced.notna().sum())
        if numeric_count / nonnull >= min_numeric_fraction:
            numeric.append(c)
    return numeric


def remove_zero_variance_features(df: pd.DataFrame, feature_cols: List[str], variance_threshold: float = 1e-12) -> List[str]:
    """
    Remove features with zero or near-zero variance.
    
    Args:
        df: DataFrame containing features
        feature_cols: List of feature column names to check
        variance_threshold: Minimum variance threshold (features below this are removed)
        
    Returns:
        List of feature column names with non-zero variance
    """
    X = df[feature_cols]
    variances = X.var(axis=0)
    nonzero_cols = variances[variances > variance_threshold].index.tolist()
    return nonzero_cols


def fit_scaler_on_mf_zinc(
    df_mf: pd.DataFrame, 
    df_zinc: pd.DataFrame, 
    feature_cols: List[str]
) -> Tuple[SimpleImputer, StandardScaler]:
    """
    Fit imputer and scaler on MF+ZINC only, per invariant to avoid leakage.
    
    Pipeline:
    1. Stack MF+ZINC features
    2. Fit SimpleImputer (median strategy) for NaN handling
    3. Fit StandardScaler on imputed data
    
    Args:
        df_mf: MF DataFrame
        df_zinc: ZINC DataFrame
        feature_cols: List of feature column names
        
    Returns:
        (imputer, scaler): Tuple of fitted imputer and scaler
    """
    # Stack MF+ZINC
    X_mf = df_mf[feature_cols].to_numpy(dtype=float, copy=False)
    X_zinc = df_zinc[feature_cols].to_numpy(dtype=float, copy=False)
    X_train = np.vstack([X_mf, X_zinc])
    
    # Fit imputer (median strategy for missing values)
    imputer = SimpleImputer(strategy='median', copy=True)
    X_imputed = imputer.fit_transform(X_train)
    
    # Fit scaler on imputed data
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    scaler.fit(X_imputed)
    
    return imputer, scaler
