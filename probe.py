"""
probe.py — Hallucination probe classifier (L1-Feature Selection edition).
"""

from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

class HallucinationProbe:
    def __init__(self) -> None:
        self._scaler = StandardScaler()
        self._model: LogisticRegression | None = None
        self._threshold: float = 0.5

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        X_scaled = self._scaler.fit_transform(X)
        
        self._model = LogisticRegression(
            penalty='l1',
            C=0.05,
            max_iter=3000,
            class_weight='balanced',
            random_state=42,
            solver='liblinear'
        )
        self._model.fit(X_scaled, y)
        
        active = np.count_nonzero(self._model.coef_)
        print(f"  [Probe] Lasso selection: {active} active features.")
        return self

    def fit_hyperparameters(self, X_val: np.ndarray, y_val: np.ndarray) -> "HallucinationProbe":
        probs = self.predict_proba(X_val)[:, 1]
        best_threshold = 0.5
        best_f1 = -1.0
        
        # Looking for a threshold that maximizes F1 on validation
        for t in np.linspace(0.1, 0.9, 101):
            y_pred_t = (probs >= t).astype(int)
            score = f1_score(y_val, y_pred_t, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_threshold = float(t)
        
        self._threshold = best_threshold
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self._scaler.transform(X)
        return self._model.predict_proba(X_scaled)