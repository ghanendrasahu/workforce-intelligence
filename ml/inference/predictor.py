"""Model registry: loads trained artifacts once, serves risk/explanations.

Single entry point for the FastAPI app, the AI agent tools and the dashboard.
Analytics run against the engineered snapshot panel held in memory.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from core.config import Settings, get_settings
from data.pipelines.build_dataset import FEATURE_COLUMNS, engineer_features
from ml.explainability.explainer import AttritionExplainer
from ml.explainability.whatif import CostConfig, MarketLookup, WhatIfEngine


class ModelRegistry:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        root_data = Path(self.settings.data_dir)
        root_models = Path(self.settings.models_dir)

        self.meta = json.loads((root_models.parent / "meta.json").read_text())
        self.model = joblib.load(root_models / "attrition_model.joblib")
        self.calibrator = joblib.load(root_models / "calibrator.joblib")
        self.anomaly_model: Any = None
        anomaly_path = root_models / "salary_anomaly.joblib"
        if anomaly_path.exists():
            self.anomaly_model = joblib.load(anomaly_path)

        thresholds = self.meta.get("thresholds", {})
        self.medium_cut = float(thresholds.get("medium_cutoff", 0.30))
        self.high_cut = float(thresholds.get("high_cutoff", 0.60))

        self.snapshots = pd.read_parquet(root_data / "snapshots.parquet")
        self.employees = pd.read_parquet(root_data / "employees.parquet")
        self.reviews = pd.read_parquet(root_data / "reviews.parquet")

        engineered, ot_mean, ot_std = engineer_features(
            self.snapshots,
            self.meta["overtime_mean"],
            self.meta["overtime_std"],
        )
        self.df = engineered
        self.cat_dtypes = {c: engineered[c].dtype
                           for c in self.meta["categorical_features"]}
        self.feature_columns = list(self.meta["feature_columns"])
        self._latest = engineered.sort_values("month_index").groupby("employee_id",
                                                                     sort=False).tail(1)
        self._latest = self._latest.reset_index(drop=True)

        market_table = pd.read_parquet(root_data / "market_salary.parquet")
        bounds = {k: tuple(v) for k, v in self.meta["numeric_bounds"].items()}
        costs = CostConfig(
            replacement_cost_multiplier=self.settings.replacement_cost_multiplier,
            training_cost_per_employee=self.settings.training_cost_per_employee,
            coaching_cost_per_employee=self.settings.coaching_cost_per_employee,
        )
        self.whatif = WhatIfEngine(
            MarketLookup(table=market_table), ot_mean, ot_std, bounds, costs,
        )

        bg = self.df.sample(min(1500, len(self.df)), random_state=0)
        self.explainer = AttritionExplainer(self.model, list(FEATURE_COLUMNS), bg)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def _conform(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Align feature frame to the exact training schema/dtypes."""
        missing = [c for c in FEATURE_COLUMNS if c not in rows.columns]
        if missing:
            raise ValueError(f"missing features: {missing}")
        out = rows[FEATURE_COLUMNS].copy()
        for col, dt in self.cat_dtypes.items():
            if out[col].dtype != dt:
                out[col] = out[col].astype(dt)
        return out

    def raw_scores(self, rows: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model.predict_proba(self._conform(rows)))[:, 1]

    def calibrated_scores(self, rows: pd.DataFrame) -> np.ndarray:
        raw = self.raw_scores(rows)
        return np.asarray(self.calibrator.predict(raw), dtype=float)

    def band(self, risk: float) -> str:
        if risk >= self.high_cut:
            return "high"
        if risk >= self.medium_cut:
            return "medium"
        return "low"

    def latest_row(self, employee_id: str) -> pd.Series | None:
        rows = self._latest[self._latest["employee_id"] == employee_id]
        if not len(rows):
            return None
        return rows.iloc[0]

    def employee_ids(self, limit: int = 50) -> list[str]:
        return self._latest["employee_id"].head(limit).tolist()

    def local_drivers(self, row: pd.Series, top_k: int = 6) -> dict[str, Any]:
        x = self._conform(pd.DataFrame([row]))
        return self.explainer.local_explanation(x.iloc[0], top_k=top_k)  # type: ignore[no-any-return]

    def search_employees(self, query: str = "", limit: int = 20) -> pd.DataFrame:
        q = query.strip().lower()
        cols = ["employee_id", "department", "location", "level"]
        if "role" in self._latest.columns:
            cols.append("role")
        frame = self._latest[cols]
        if q:
            mask = (
                frame["employee_id"].str.lower().str.contains(q, regex=False)
                | frame["department"].str.lower().str.contains(q, regex=False)
                | frame["location"].str.lower().str.contains(q, regex=False)
            )
            frame = frame[mask]
        return frame.head(limit)

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------
    def attrition_trend(self) -> pd.DataFrame:
        g = self.snapshots.groupby("month_index")["attrition_event"]
        rate = g.mean() * 12
        n_active = g.count()
        dates = (self.snapshots.groupby("month_index")["date"].first())
        out = pd.DataFrame({
            "month_index": rate.index,
            "date": pd.to_datetime(dates.values),
            "annualized_rate": np.round(rate.to_numpy() * 100, 2),
            "n_active": n_active.values,
        })
        return out

    def department_metrics(self) -> pd.DataFrame:
        last_m = int(self.snapshots["month_index"].max())
        window = self.snapshots[self.snapshots["month_index"] > last_m - 12]
        latest = self._latest.copy()
        latest["risk"] = self.calibrated_scores(latest)
        events = window.groupby("department")["attrition_event"].sum()
        headcount = latest.groupby("department").size()
        high_risk = latest.assign(
            hr=(latest["risk"] >= self.high_cut).astype(int),
        ).groupby("department")["hr"].sum()
        metrics = pd.DataFrame({
            "headcount": headcount,
            "attrition_events_12m": events,
            "avg_comp_ratio": latest.groupby("department")["comp_ratio"].mean().round(3),
            "avg_engagement": latest.groupby("department")["engagement_score"]
                .mean().round(1),
            "avg_overtime": latest.groupby("department")["overtime_hours"].mean().round(1),
            "avg_risk": latest.groupby("department")["risk"].mean().round(4),
            "high_risk_employees": high_risk,
        })
        metrics["annualized_attrition_pct"] = (
            metrics["attrition_events_12m"] / headcount * 100).round(2)
        return metrics.sort_values("annualized_attrition_pct", ascending=False)

    def high_risk_employees(self, top_k: int = 25) -> pd.DataFrame:
        latest = self._latest.copy()
        latest["risk"] = self.calibrated_scores(latest)
        cols = ["employee_id", "department", "location", "level", "salary",
                "comp_ratio", "engagement_score", "overtime_hours", "risk"]
        ranked = latest.sort_values("risk", ascending=False)
        return ranked[cols].head(top_k)

    def global_drivers(self) -> list[dict[str, Any]]:
        path = Path(self.settings.metrics_dir) / "global_shap.json"
        if path.exists():
            return json.loads(path.read_text())
        sample = self.df.sample(min(2000, len(self.df)), random_state=0)
        return self.explainer.global_importance(sample)

    def cohort_analysis(self, department: str | None = None,
                        location: str | None = None,
                        level_min: int = 1, level_max: int = 5) -> dict[str, object]:
        sub = self._latest[
            self._latest["department"].eq(department or self._latest["department"])
            & self._latest["location"].eq(location or self._latest["location"])
            & self._latest["level"].between(level_min, level_max)
        ]
        if not len(sub):
            return {"n_employees": 0}
        risks = self.calibrated_scores(sub.sample(min(4000, len(sub)), random_state=0))
        drivers = {}
        for feat in ["comp_ratio", "engagement_score", "manager_sentiment",
                     "months_since_promotion", "overtime_hours"]:
            drivers[feat] = round(float(sub[feat].astype(float).mean()), 3)
        return {
            "n_employees": int(len(sub)),
            "avg_risk": round(float(risks.mean()), 4),
            "p90_risk": round(float(np.quantile(risks, 0.9)), 4),
            "high_risk_share": round(float((risks >= self.high_cut).mean()), 4),
            "driver_means": drivers,
        }

    def reviews_sample(self, employee_id: str, limit: int = 5) -> pd.DataFrame:
        rev = self.reviews[self.reviews["employee_id"] == employee_id]
        return rev.tail(limit)


@lru_cache(maxsize=1)
def get_registry() -> ModelRegistry:
    return ModelRegistry()
