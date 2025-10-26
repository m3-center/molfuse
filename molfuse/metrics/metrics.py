from __future__ import annotations

import numpy as np
from typing import Tuple
from scipy.stats import spearmanr
from typing import Any
from sklearn.metrics import roc_auc_score, average_precision_score


def ef_at_k_percent(scores: np.ndarray, labels: np.ndarray, k_percent: float) -> float:
    """
    Compute Enrichment Factor at top k%.

    - scores: higher is better
    - labels: binary array (1=active, 0=decoy)
    - k_percent: float in (0,100]
    """
    assert 0 < k_percent <= 100
    n = len(scores)
    k = max(1, int(np.ceil(n * (k_percent / 100.0))))
    order = np.argsort(-scores)
    top_labels = labels[order][:k]
    hit_rate_top = top_labels.mean()
    hit_rate_all = labels.mean() if labels.mean() > 0 else 1e-12
    return float(hit_rate_top / hit_rate_all)


def spearman_rho(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Return Spearman's rho and p-value."""
    res: Any = spearmanr(x, y, nan_policy="omit")
    if hasattr(res, "statistic"):
        rho_val = res.statistic
        p_val = res.pvalue
    else:
        rho_val, p_val = res  # type: ignore[misc]
    rho_f: float = float(rho_val)
    p_f: float = float(p_val)
    return rho_f, p_f


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """ROC-AUC with robust handling for degenerate cases."""
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def pr_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """PR-AUC via average precision (AP); commonly used proxy for PR-AUC."""
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(average_precision_score(labels, scores))
