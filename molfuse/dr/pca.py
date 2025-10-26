from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.decomposition import PCA


def fit_pca(X_train: np.ndarray, n_components: int) -> Tuple[PCA, np.ndarray]:
    """
    Fit PCA on training data and return the fitted model and transformed training embedding.

    Inputs:
    - X_train: float32/float64 array (n_samples, n_features), scaled feature matrix for MF+ZINC
    - n_components: embedding dimension

    Outputs:
    - pca: fitted sklearn PCA object
    - Z_train: transformed training embedding (n_samples, n_components)
    """
    pca = PCA(n_components=n_components, svd_solver="auto", random_state=None)
    Z_train = pca.fit_transform(X_train)
    return pca, Z_train
