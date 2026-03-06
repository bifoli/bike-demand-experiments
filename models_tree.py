from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

@dataclass
class QuantileForecast:
    q10: np.ndarray
    q50: np.ndarray
    q90: np.ndarray

class RFQuantile:
    """Quantile Random Forest via per-tree distribution."""
    def __init__(self, n_estimators: int = 200, max_depth: int = 18, random_state: int = 42, n_jobs: int = -1):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self.trees_: List = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RFQuantile":
        self.model.fit(X, y)
        self.trees_ = list(self.model.estimators_)
        return self

    def predict(self, X: np.ndarray, quantiles=(0.1, 0.5, 0.9)) -> QuantileForecast:
        # predictions per tree: shape (n_trees, n_samples)
        preds = np.stack([t.predict(X) for t in self.trees_], axis=0)
        q10 = np.quantile(preds, quantiles[0], axis=0)
        q50 = np.quantile(preds, quantiles[1], axis=0)
        q90 = np.quantile(preds, quantiles[2], axis=0)
        return QuantileForecast(q10=q10, q50=q50, q90=q90)

class XGBQuantile:
    """Separate XGBoost models for q in {0.1,0.5,0.9}."""
    def __init__(self, params: Dict, random_state: int = 42):
        self.params = params
        self.random_state = random_state
        self.models: Dict[float, xgb.XGBRegressor] = {}

    def fit(self, X: np.ndarray, y: np.ndarray, quantiles=(0.1,0.5,0.9)) -> "XGBQuantile":
        for q in quantiles:
            m = xgb.XGBRegressor(
                objective="reg:quantileerror",
                quantile_alpha=q,
                random_state=self.random_state,
                **self.params,
            )
            m.fit(X, y)
            self.models[q] = m
        return self

    def predict(self, X: np.ndarray) -> QuantileForecast:
        q10 = self.models[0.1].predict(X)
        q50 = self.models[0.5].predict(X)
        q90 = self.models[0.9].predict(X)
        return QuantileForecast(q10=q10, q50=q50, q90=q90)
