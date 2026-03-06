from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt

def plot_interval_case_study(
    dt: np.ndarray,
    y_true: np.ndarray,
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
    start_ts: str,
    days: int = 7,
    out_path: Optional[str] = None,
    title: str = "Interval case study (P10–P90)",
) -> None:
    import pandas as pd
    dt = pd.to_datetime(dt)
    start = pd.Timestamp(start_ts)
    end = start + pd.Timedelta(days=days)
    mask = (dt >= start) & (dt < end)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        raise ValueError("No timestamps in the requested window.")
    xs = dt[mask]
    yt = y_true[mask]
    lo = q10[mask]
    mid = q50[mask]
    hi = q90[mask]

    plt.figure(figsize=(10, 3))
    plt.plot(xs, yt, label="Actual")
    plt.plot(xs, mid, label="P50 forecast")
    plt.fill_between(xs, lo, hi, alpha=0.25, label="P10–P90")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Bikes/hour")
    plt.legend(loc="upper left", ncol=3, frameon=False)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=200)
    plt.close()

def plot_hourly_mae(
    hours: np.ndarray,
    model_to_abs_err: Dict[str, np.ndarray],
    out_path: Optional[str] = None,
    title: str = "Hourly MAE by model",
) -> None:
    # hours: (n_samples,) int 0..23 aligned with errors
    plt.figure(figsize=(6.5, 3.2))
    for name, abs_err in model_to_abs_err.items():
        mae_by_hr = []
        for h in range(24):
            m = abs_err[hours == h]
            mae_by_hr.append(float(np.mean(m)) if len(m) else np.nan)
        plt.plot(range(24), mae_by_hr, label=name)
    plt.title(title)
    plt.xlabel("Hour of day")
    plt.ylabel("MAE (bikes/hour)")
    plt.xticks(range(0,24,3))
    plt.legend(frameon=False, ncol=2, fontsize=8)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=200)
    plt.close()

def plot_feature_importance(
    names: List[str],
    weights: np.ndarray,
    top_k: int = 12,
    out_path: Optional[str] = None,
    title: str = "TFT feature importance (mean gate weight)",
) -> None:
    weights = np.asarray(weights)
    idx = np.argsort(weights)[::-1][:top_k]
    top_names = [names[i] for i in idx]
    top_w = weights[idx]

    plt.figure(figsize=(6.8, 3.2))
    plt.bar(range(len(top_w)), top_w)
    plt.xticks(range(len(top_w)), top_names, rotation=45, ha="right", fontsize=8)
    plt.ylabel("Mean weight")
    plt.title(title)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=200)
    plt.close()
