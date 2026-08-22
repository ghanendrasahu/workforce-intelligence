"""SHAP-based explainability: global importance, local attributions, PDP.

Local attributions are converted from log-odds SHAP values into *risk points*
(percentage-point deltas against the population baseline) so that business
users can read them directly.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap
from scipy.special import expit


class AttritionExplainer:
    """Wraps a fitted binary classifier with TreeExplainer semantics.

    Uses ``feature_perturbation="tree_path_dependent"`` (no background data)
    because LightGBM models trained with native categorical splits reject
    interventional SHAP with a background sample.
    """

    def __init__(self, model: object, feature_columns: list[str],
                 background: pd.DataFrame | None = None,
                 max_background: int = 500) -> None:
        del background, max_background  # kept for API compatibility
        self.model = model
        self.feature_columns = list(feature_columns)
        self.explainer = shap.TreeExplainer(model)
        self.base_margin = float(np.ravel(self.explainer.expected_value)[0])

    # ------------------------------------------------------------------
    def global_importance(self, X: pd.DataFrame, top_k: int = 15) -> list[dict[str, float]]:
        sv = self.explainer.shap_values(X[self.feature_columns], check_additivity=False)
        arr = np.asarray(sv[-1] if isinstance(sv, list) else sv, dtype=float)
        if arr.ndim == 3:  # (classes, rows, features)
            arr = arr[-1]
        mean_abs = np.abs(arr).mean(axis=0)
        order = np.argsort(-mean_abs)[:top_k]
        return [
            {"feature": self.feature_columns[i],
             "mean_abs_shap": round(float(mean_abs[i]), 5),
             "risk_points": round(self._logodds_to_points(float(mean_abs[i])), 2)}
            for i in order
        ]

    def partial_dependence(self, X: pd.DataFrame, features: list[str],
                           grid_resolution: int = 12) -> dict[str, dict[str, list[float]]]:
        """Model-agnostic marginal-effect curves (handles any column dtype)."""
        model: Any = self.model
        out: dict[str, dict[str, list[float]]] = {}
        sample = X[self.feature_columns].sample(min(1500, len(X)), random_state=0)
        for feat in features:
            col = sample[feat]
            if str(col.dtype).lower() in ("category", "object"):
                grid: list[Any] = sorted(col.astype(str).unique().tolist())
            else:
                lo, hi = float(col.min()), float(col.max())
                grid = np.linspace(lo, hi, grid_resolution).tolist()
            preds = []
            for g in grid:
                modified = sample.copy()
                modified[feat] = g
                proba = np.asarray(model.predict_proba(modified))[:, 1]
                preds.append(round(float(np.mean(proba)), 4))
            grid_out: list[Any] = [_json_safe(g) for g in grid]
            out[feat] = {"grid": [float(g) if isinstance(g, float) else g
                                  for g in grid_out],
                         "avg_prediction": [float(p) for p in preds]}
        return out

    # ------------------------------------------------------------------
    def local_explanation(self, row: pd.Series, top_k: int = 8) -> dict[str, object]:
        x = pd.DataFrame([row[self.feature_columns]])
        x_enc = self._encode_categories(x)
        sv = self.explainer.shap_values(x_enc, check_additivity=False)
        vals = self._single_row_shap(sv)
        contribs = sorted(
            zip(self.feature_columns, vals, strict=False), key=lambda t: -abs(t[1])
        )[:top_k]

        baseline_risk = float(expit(self.base_margin))
        margin = self.base_margin
        steps: list[dict[str, object]] = []
        for feat, shap_val in contribs:
            prev = float(expit(margin))
            margin += float(shap_val)
            steps.append({
                "feature": feat,
                "value": _json_safe(row[feat]),
                "shap_log_odds": round(float(shap_val), 4),
                "risk_points": round((float(expit(margin)) - prev) * 100, 2),
                "direction": "increases" if shap_val > 0 else "decreases",
            })

        final_risk = float(expit(self.base_margin + float(np.sum(vals))))
        return {
            "baseline_risk": round(baseline_risk, 4),
            "risk": round(final_risk, 4),
            "contributions": steps,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _single_row_shap(sv: object) -> np.ndarray:
        """Normalize SHAP output variants to a 1-D per-feature array."""
        arr = np.asarray(sv[-1] if isinstance(sv, list) else sv, dtype=float)
        return np.atleast_2d(arr)[0]

    def _encode_categories(self, X: pd.DataFrame) -> pd.DataFrame:
        """LightGBM sklearn API wants category dtype preserved; nothing to do."""
        Xc = X.copy()
        for col in self.feature_columns:
            if str(Xc[col].dtype) == "object":
                Xc[col] = Xc[col].astype("category")
        return Xc

    @staticmethod
    def _logodds_to_points(log_odds: float, base: float | None = None) -> float:
        base = base if base is not None else 0.0
        return (expit(base + log_odds) - expit(base)) * 100


def _json_safe(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return round(float(value), 4)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_json(data: object, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return p
