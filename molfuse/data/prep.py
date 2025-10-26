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


def select_feature_columns(df: pd.DataFrame, min_numeric_fraction: float = 0.95) -> List[str]:
    """
    Select feature columns by:
    - Excluding known non-feature columns
    - Including columns with numeric dtype
    - Including object/mixed-type columns if they become numeric after coercion for
      at least `min_numeric_fraction` of their non-null entries.

    This makes selection robust to pandas mixed-type inference in CSVs.
    """
    candidates = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    numeric: List[str] = []
    for c in candidates:
        s = df[c]
        # Fast path: already numeric dtype
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
