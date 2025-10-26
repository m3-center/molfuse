from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
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


def select_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Select feature columns from a DataFrame by excluding known non-feature columns
    and keeping numeric columns only.
    """
    candidates = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    numeric = [c for c in candidates if pd.api.types.is_numeric_dtype(df[c])]
    return numeric


def fit_scaler_on_mf_zinc(df_mf: pd.DataFrame, df_zinc: pd.DataFrame, feature_cols: List[str]) -> StandardScaler:
    """
    Fit StandardScaler on MF+ZINC only, per invariant to avoid leakage.
    """
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    X_mf = df_mf[feature_cols].to_numpy(dtype=float, copy=False)
    X_zinc = df_zinc[feature_cols].to_numpy(dtype=float, copy=False)
    X_train = np.vstack([X_mf, X_zinc])
    scaler.fit(X_train)
    return scaler
