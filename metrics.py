from __future__ import annotations

import numpy as np

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)

def interval_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    covered = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(covered) * 100.0)

def interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    return float(np.mean(upper - lower))

def pinball_loss(y_true: np.ndarray, y_pred_q: np.ndarray, q: float) -> float:
    """Mean pinball loss for a single quantile q."""
    y_true = np.asarray(y_true)
    y_pred_q = np.asarray(y_pred_q)
    diff = y_true - y_pred_q
    return float(np.mean(np.maximum(q * diff, (q - 1.0) * diff)))
