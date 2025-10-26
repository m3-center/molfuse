from __future__ import annotations

from typing import Tuple, Optional

import numpy as np
from sklearn.neighbors import NearestNeighbors


def nn_min_distance_scores(
    Z_train: np.ndarray,
    Z_query: np.ndarray,
    metric: str = "euclidean",
    algorithm: str = "auto",
    n_jobs: int = -1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute exact 1-NN distances from each query to the training set in embedded space.

    Returns tuple of (scores, distances):
    - scores: Negative of min distances (higher is better)
    - distances: Raw min distances (for diagnostics)
    """
    nn = NearestNeighbors(n_neighbors=1, metric=metric, algorithm=algorithm, n_jobs=n_jobs)
    nn.fit(Z_train)
    distances, _ = nn.kneighbors(Z_query, return_distance=True)
    distances = distances.reshape(-1)
    scores = -distances
    return scores, distances
