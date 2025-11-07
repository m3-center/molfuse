from __future__ import annotations

from typing import Tuple, Optional

import numpy as np

try:
    import umap
except ImportError as e:
    raise ImportError("umap-learn is required for UMAP functionality. Install with `pip install umap-learn`.\n" + str(e))


def fit_umap(
    X_train: np.ndarray,
    n_components: int,
    n_neighbors: int,
    min_dist: float,
    metric: str = "euclidean",
    random_state: Optional[int] = None,
    n_jobs: int = -1,
    init: str = "spectral",
) -> Tuple["umap.UMAP", np.ndarray]:
    """
    Fit UMAP on training data (MF+ZINC) and return the fitted model and transformed training embedding.

    - random_state=None enables parallel UMAP (non-deterministic but faster on HPC).
    - n_jobs is passed to numba/umap via environment; here retained for interface completeness.
    """
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,  # None → parallel by default
        init=init,
        verbose=False,
    )
    Z_train = reducer.fit_transform(X_train)
    return reducer, Z_train
