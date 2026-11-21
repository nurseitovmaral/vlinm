from __future__ import annotations

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, roc_auc_score, roc_curve


def clustering_metrics(reference: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    if reference.ndim != 1 or predicted.shape != reference.shape or len(reference) == 0:
        raise ValueError("labels must be nonempty vectors of equal length")
    return {"ari": float(adjusted_rand_score(reference, predicted)), "nmi": float(normalized_mutual_info_score(reference, predicted))}


def binary_metrics(reference: np.ndarray, score: np.ndarray, threshold: float | None = None) -> dict[str, float]:
    labels = np.asarray(reference)
    values = np.asarray(score)
    if labels.ndim != 1 or values.shape != labels.shape or set(np.unique(labels)) != {0, 1} or not np.isfinite(values).all():
        raise ValueError("binary labels and finite scores are required")
    fpr, tpr, thresholds = roc_curve(labels, values)
    cutoff = float(thresholds[np.argmax(tpr - fpr)]) if threshold is None else float(threshold)
    predicted = values >= cutoff
    tp = int(np.sum(predicted & (labels == 1)))
    fp = int(np.sum(predicted & (labels == 0)))
    fn = int(np.sum(~predicted & (labels == 1)))
    tn = int(np.sum(~predicted & (labels == 0)))
    return {"auroc": float(roc_auc_score(labels, values)), "threshold": cutoff, "sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp), "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}
