"""Model evaluation: classification metrics, calibration, threshold selection."""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)


def classification_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    out: dict[str, float] = {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "brier": float(brier_score_loss(y_true, scores)),
        "base_rate": float(y_true.mean()),
    }
    thr = best_f1_threshold(y_true, scores)
    out["best_f1_threshold"] = float(thr)
    preds = (scores >= thr).astype(int)
    tp = int(((preds == 1) & (y_true == 1)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    out.update({
        "precision_at_best_f1": precision,
        "recall_at_best_f1": recall,
        "f1_at_best_f1": 2 * precision * recall / max(precision + recall, 1e-9),
    })
    r80 = recall_target_threshold(y_true, scores, target_recall=0.80)
    if r80 is not None:
        p_at = precision_at_threshold(y_true, scores, r80)
        out["recall80_threshold"] = float(r80)
        out["precision_at_recall80"] = float(p_at)
    return out


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores)
    y_sorted = np.asarray(y_true)[order]
    s_sorted = np.asarray(scores)[order]
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)
    total_pos = tps[-1]
    precisions = tps / np.maximum(tps + fps, 1)
    recalls = tps / max(total_pos, 1)
    f1s = 2 * precisions * recalls / np.maximum(precisions + recalls, 1e-9)
    return float(s_sorted[int(np.argmax(f1s))])


def precision_at_threshold(y_true: np.ndarray, scores: np.ndarray,
                           threshold: float) -> float:
    preds = (np.asarray(scores) >= threshold).astype(int)
    y_true = np.asarray(y_true).astype(int)
    tp = int(((preds == 1) & (y_true == 1)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    return tp / max(tp + fp, 1)


def recall_target_threshold(y_true: np.ndarray, scores: np.ndarray,
                            target_recall: float = 0.80) -> float | None:
    """Smallest threshold achieving at least `target_recall` (None if impossible)."""
    y_true = np.asarray(y_true).astype(int)
    total_pos = max(int(y_true.sum()), 1)
    order = np.argsort(-scores)
    y_sorted = y_true[order]
    s_sorted = np.asarray(scores, dtype=float)[order]
    recalls = np.cumsum(y_sorted) / total_pos
    hits = np.flatnonzero(recalls >= target_recall)
    if len(hits) == 0:
        return None
    return float(s_sorted[hits[0]])


def calibration_curve_data(y_true: np.ndarray, scores: np.ndarray,
                           bins: int = 10) -> list[dict[str, float]]:
    y_true = np.asarray(y_true).astype(float)
    scores = np.asarray(scores, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for i in range(bins):
        mask = (scores >= edges[i]) & (scores < edges[i + 1] + (i == bins - 1) * 1e-9)
        if mask.sum() >= 10:
            out.append({
                "bin_lower": round(float(edges[i]), 3),
                "mean_predicted": round(float(scores[mask].mean()), 4),
                "observed_rate": round(float(y_true[mask].mean()), 4),
                "count": int(mask.sum()),
            })
    return out


def fit_calibrator(val_scores: np.ndarray, val_y: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(np.asarray(val_scores, dtype=float), np.asarray(val_y, dtype=float))
    return iso


def expected_calibration_error(y_true: np.ndarray, scores: np.ndarray,
                               bins: int = 10) -> float:
    curve = calibration_curve_data(y_true, scores, bins)
    total = sum(c["count"] for c in curve)
    if total == 0:
        return 0.0
    return float(sum(abs(c["mean_predicted"] - c["observed_rate"]) * c["count"]
                     for c in curve) / total)
