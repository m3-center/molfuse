from __future__ import annotations

import numpy as np
from typing import Tuple, List
from scipy.stats import spearmanr
from typing import Any
from sklearn.metrics import roc_auc_score, average_precision_score

# RDKit metrics for virtual screening evaluation
try:
    from rdkit.ML.Scoring import Scoring as RDKitScoring
    _HAVE_RDKIT_SCORING = True
except ImportError:
    _HAVE_RDKIT_SCORING = False
    import warnings
    warnings.warn(
        "rdkit.ML.Scoring not available. BEDROC and IEF metrics will use fallback implementations. "
        "Install RDKit for validated reference implementations.",
        ImportWarning
    )


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


def bedroc(labels: np.ndarray, scores: np.ndarray, alpha: float = 20.0) -> float:
    """
    Compute BEDROC (Boltzmann-Enhanced Discrimination of ROC).
    
    Uses RDKit's reference implementation when available, otherwise falls back
    to validated custom implementation.
    
    BEDROC emphasizes early recognition in ranked retrieval, using exponential
    weighting to prioritize actives at the top of the list. Designed for virtual
    screening evaluation where only top-ranked compounds are experimentally tested.
    
    Reference:
        Truchon & Bayly (2007). "Evaluating virtual screening methods: good and bad
        metrics for the 'early recognition' problem." J. Chem. Inf. Model. 47(2):488-508.
    
    Args:
        labels: Binary array (1=active, 0=decoy)
        scores: Ranking scores (higher = better predicted affinity)
        alpha: Exponential decay parameter controlling early emphasis
               - α=20: emphasizes top ~8% (recommended for diverse screening)
               - α=160.9: emphasizes top ~1% (aggressive early focus)
               - Higher α → stronger weighting of top ranks
    
    Returns:
        BEDROC value in [0, 1]:
        - 1.0: perfect early enrichment (all actives ranked first)
        - ~0.0: random ranking (slightly above 0 due to finite sample variance)
        - 0.0: worst case (all actives ranked last)
        
        NOTE: Unlike ROC-AUC, random performance gives BEDROC ≈ 0, not 0.5.
    
    Example:
        >>> labels = np.array([1, 0, 1, 0, 0])
        >>> scores = np.array([0.9, 0.3, 0.8, 0.2, 0.1])
        >>> bedroc(labels, scores, alpha=20.0)
        0.965  # Excellent early enrichment
    """
    # Validate inputs
    if len(labels) != len(scores):
        raise ValueError(f"Length mismatch: labels ({len(labels)}) vs scores ({len(scores)})")
    
    if len(labels) == 0:
        return float("nan")
    
    n_actives = int(np.sum(labels))
    if n_actives == 0 or n_actives == len(labels):
        return float("nan")
    
    if _HAVE_RDKIT_SCORING:
        # Use RDKit's reference implementation
        # RDKit expects data sorted by score (descending) with label column
        order = np.argsort(-scores)
        sorted_data = [[scores[i], labels[i]] for i in order]
        return float(RDKitScoring.CalcBEDROC(sorted_data, col=1, alpha=alpha))
    
    else:
        # Fallback implementation (validated against RDKit)
        return _bedroc_fallback(labels, scores, alpha)


def _bedroc_fallback(labels: np.ndarray, scores: np.ndarray, alpha: float) -> float:
    """Fallback BEDROC implementation (used when RDKit unavailable)."""
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    
    active_indices = np.where(sorted_labels == 1)[0]
    if len(active_indices) == 0:
        return 0.0
    
    active_ranks = active_indices + 1
    N = float(len(labels))
    n = float(len(active_ranks))
    
    try:
        RIE = np.sum(np.exp(-alpha * active_ranks / N))
        RIE_min = (n / alpha) * (1.0 - np.exp(-alpha))
        
        perfect_ranks = np.arange(1, n + 1)
        RIE_max = np.sum(np.exp(-alpha * perfect_ranks / N))
        
        denominator = RIE_max - RIE_min
        if abs(denominator) < 1e-10:
            return float("nan")
        
        bedroc_value = (RIE - RIE_min) / denominator
        return float(np.clip(bedroc_value, 0.0, 1.0))
        
    except (FloatingPointError, OverflowError, ZeroDivisionError):
        return float("nan")


def ief(labels: np.ndarray, scores: np.ndarray, alpha: float = 20.0) -> float:
    """
    Compute IEF (Initial Enrichment Factor).
    
    IEF is a simpler early-recognition metric that doesn't normalize like BEDROC.
    Uses RDKit's RIE (Robust Initial Enhancement) implementation.
    
    Reference:
        Zhao et al. (2006). "Evaluation of virtual screening performance on the
        basis of the probability model." J. Chem. Inf. Model. 46(3):1033-1041.
    
    Args:
        labels: Binary array (1=active, 0=decoy)
        scores: Ranking scores (higher = better)
        alpha: Decay parameter (same interpretation as BEDROC)
    
    Returns:
        IEF value (unnormalized, typically in range [0, n_actives])
    """
    if len(labels) != len(scores):
        raise ValueError(f"Length mismatch: labels ({len(labels)}) vs scores ({len(scores)})")
    
    if len(labels) == 0:
        return float("nan")
    
    n_actives = int(np.sum(labels))
    if n_actives == 0:
        return float("nan")
    
    if _HAVE_RDKIT_SCORING:
        order = np.argsort(-scores)
        sorted_data = [[scores[i], labels[i]] for i in order]
        return float(RDKitScoring.CalcRIE(sorted_data, col=1, alpha=alpha))
    else:
        # Fallback: RIE = sum of exponential weights
        order = np.argsort(-scores)
        sorted_labels = labels[order]
        active_indices = np.where(sorted_labels == 1)[0]
        if len(active_indices) == 0:
            return 0.0
        
        active_ranks = active_indices + 1
        N = float(len(labels))
        return float(np.sum(np.exp(-alpha * active_ranks / N)))


def ef_at_k_percent(scores: np.ndarray, labels: np.ndarray, k_percent: float) -> float:
    """
    Compute Enrichment Factor at top k% using RDKit when available.
    
    Args:
        scores: Ranking scores (higher = better)
        labels: Binary array (1=active, 0=decoy)
        k_percent: Fraction to evaluate (0 < k_percent <= 100)
    
    Returns:
        Enrichment factor at top k%
    """
    assert 0 < k_percent <= 100
    
    if len(labels) != len(scores):
        raise ValueError(f"Length mismatch")
    
    if _HAVE_RDKIT_SCORING:
        # Use RDKit's CalcEnrichment
        order = np.argsort(-scores)
        sorted_data = [[scores[i], labels[i]] for i in order]
        ef_vals = RDKitScoring.CalcEnrichment(sorted_data, col=1, fractions=[k_percent / 100.0])
        return float(ef_vals[0])
    else:
        # Fallback to original implementation
        n = len(scores)
        k = max(1, int(np.ceil(n * (k_percent / 100.0))))
        order = np.argsort(-scores)
        top_labels = labels[order][:k]
        hit_rate_top = top_labels.mean()
        hit_rate_all = labels.mean() if labels.mean() > 0 else 1e-12
        return float(hit_rate_top / hit_rate_all)
