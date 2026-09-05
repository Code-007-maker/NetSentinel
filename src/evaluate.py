"""
Evaluation metrics, validation-only threshold selection, and probability audits.

Threshold selection MUST receive validation arrays only.  This module has no
access to test labels by design — `select_decision_threshold` has no test-set
parameters.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Inclusive grid: 0.10, 0.15, ..., 0.90
DEFAULT_THRESHOLDS = np.round(np.arange(0.10, 0.90 + 1e-9, 0.05), 2)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None,
) -> Dict[str, float]:
    """Computes comprehensive evaluation metrics from aligned arrays."""
    metrics: Dict[str, float] = {}
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()
    if y_true.size == 0:
        return {
            "tn": 0.0, "fp": 0.0, "fn": 0.0, "tp": 0.0,
            "fpr": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "roc_auc": 0.0, "pr_auc": 0.0,
        }

    tn, fp, fn, tp = _confusion_counts(y_true, y_pred)
    metrics["tn"] = float(tn)
    metrics["fp"] = float(fp)
    metrics["fn"] = float(fn)
    metrics["tp"] = float(tp)
    metrics["fpr"] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))

    if y_prob is not None and len(np.unique(y_true)) > 1:
        y_prob = np.asarray(y_prob).astype(float).ravel()
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = 0.0
        metrics["pr_auc"] = 0.0

    return metrics


def _confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return int(tn), int(fp), int(fn), int(tp)


def select_decision_threshold(
    y_true_val: np.ndarray,
    y_prob_val: np.ndarray,
    thresholds: Optional[Sequence[float]] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Select a decision threshold using ONLY validation labels and scores.

    Criterion (Youden's J):
      J = TPR - FPR = recall - fpr
      Pick the threshold that maximises J on the validation set.
      Ties are broken by choosing the *higher* threshold (more conservative FPR).

    This is appropriate when ROC ranking is informative (high AUC) but a
    hardcoded 0.5 threshold classifies everything as attack (FPR=1).

    Parameters
    ----------
    y_true_val, y_prob_val:
        Validation labels and predicted probabilities.  Test arrays must not
        be passed here.
    """
    y_true_val = np.asarray(y_true_val).astype(int).ravel()
    y_prob_val = np.asarray(y_prob_val).astype(float).ravel()
    if y_true_val.shape != y_prob_val.shape:
        raise ValueError("Validation labels and probabilities must be aligned.")

    grid = np.asarray(DEFAULT_THRESHOLDS if thresholds is None else thresholds, dtype=float)

    if len(y_true_val) == 0 or len(np.unique(y_true_val)) < 2:
        fallback = 0.5
        pred = (y_prob_val >= fallback).astype(int)
        metrics = evaluate_predictions(y_true_val, pred, y_prob_val)
        metrics["threshold"] = fallback
        metrics["youden_j"] = metrics["recall"] - metrics["fpr"]
        metrics["selection_criterion"] = "fallback_0.5_single_class_val"
        return fallback, metrics

    best_t = float(grid[0])
    best_j = -np.inf
    best_metrics: Dict[str, float] = {}

    for t in grid:
        pred = (y_prob_val >= t).astype(int)
        m = evaluate_predictions(y_true_val, pred, y_prob_val)
        j = m["recall"] - m["fpr"]
        m["youden_j"] = float(j)
        # Max J; tie-break toward higher threshold (lower FPR).
        if j > best_j + 1e-12 or (abs(j - best_j) <= 1e-12 and t > best_t):
            best_j = j
            best_t = float(t)
            best_metrics = m

    best_metrics["threshold"] = best_t
    best_metrics["selection_criterion"] = "youden_j_validation_only"
    return best_t, best_metrics


def probability_audit(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """Distribution of attack probabilities by true class, plus confusion metrics."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob).astype(float).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    att = y_prob[y_true == 1]
    ben = y_prob[y_true == 0]
    metrics = evaluate_predictions(y_true, y_pred, y_prob)

    def _stats(arr: np.ndarray) -> Dict[str, float]:
        if arr.size == 0:
            return {"count": 0.0, "min": float("nan"), "mean": float("nan"),
                    "median": float("nan"), "max": float("nan")}
        return {
            "count": float(arr.size),
            "min": float(np.min(arr)),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "max": float(np.max(arr)),
        }

    audit = {
        "threshold": float(threshold),
        "predicted_attack": int(y_pred.sum()),
        "predicted_benign": int((y_pred == 0).sum()),
        "attack": _stats(att),
        "benign": _stats(ben),
    }
    audit.update(metrics)
    return audit


def format_probability_audit(audit: Dict, title: str = "TEST PROBABILITY AUDIT") -> str:
    att = audit["attack"]
    ben = audit["benign"]

    def _fmt(v):
        if isinstance(v, float) and np.isnan(v):
            return "n/a"
        if isinstance(v, float):
            return f"{v:.6f}"
        return str(v)

    lines = [
        title,
        "attack samples:",
        f"  count: {_fmt(att['count'])}",
        f"  min: {_fmt(att['min'])}",
        f"  mean: {_fmt(att['mean'])}",
        f"  median: {_fmt(att['median'])}",
        f"  max: {_fmt(att['max'])}",
        "",
        "benign samples:",
        f"  count: {_fmt(ben['count'])}",
        f"  min: {_fmt(ben['min'])}",
        f"  mean: {_fmt(ben['mean'])}",
        f"  median: {_fmt(ben['median'])}",
        f"  max: {_fmt(ben['max'])}",
        "",
        f"threshold: {audit['threshold']:.2f}",
        f"predicted attack: {audit['predicted_attack']}",
        f"predicted benign: {audit['predicted_benign']}",
        f"TP: {int(audit.get('tp', 0))}",
        f"TN: {int(audit.get('tn', 0))}",
        f"FP: {int(audit.get('fp', 0))}",
        f"FN: {int(audit.get('fn', 0))}",
        f"FPR: {audit.get('fpr', 0):.4f}",
        f"precision: {audit.get('precision', 0):.4f}",
        f"recall: {audit.get('recall', 0):.4f}",
        f"F1: {audit.get('f1', 0):.4f}",
    ]
    return "\n".join(lines)


def future_attack_at_horizon(future_attacks: List[int], k: int) -> Optional[int]:
    """
    Return the t+K label from a future-label list produced as
    attack_labels[t+1 : t+k+1]  (length K, last element is y_{t+K}).
    """
    if not future_attacks:
        return None
    if len(future_attacks) < k:
        return None
    return int(future_attacks[k - 1])
