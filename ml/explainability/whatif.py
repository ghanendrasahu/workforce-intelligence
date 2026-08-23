"""What-if counterfactual simulator + business scenario costing.

Interventions mutate a feature row respecting feature relationships (a raise
changes comp_ratio against the market table; a promotion resets tenure-in-level,
bumps level and salary), then the frozen model re-scores. This is model-based
counterfactual reasoning - validated against the generator's ground truth in
tests and documented in docs/ml.md.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

NUMERIC_FEATURES_FOR_BOUNDS = [
    "age", "log_tenure_months", "level", "comp_ratio", "comp_peer_gap",
    "performance_rating", "engagement_score", "manager_sentiment",
    "overtime_hours", "workload_z", "overtime_peer_gap",
    "promo_pressure", "promos_last_12m",
]

_LEVEL_EXPECTED = {1: 24, 2: 30, 3: 36, 4: 48, 5: 60}


@dataclass
class InterventionParams:
    salary_raise_pct: float = 0.0          # e.g. 10 -> +10% salary
    workload_reduction_pct: float = 0.0    # e.g. 20 -> -20% overtime hours
    promotion: bool = False                # accelerate promotion now
    engagement_boost_points: float = 0.0   # training/wellness programmes
    manager_sentiment_boost: float = 0.0   # coaching intervention (0..1)

    def is_active(self) -> bool:
        return self != InterventionParams()

    def to_dict(self) -> dict[str, float | bool]:
        return {k: v for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> InterventionParams:
        d = data or {}
        allowed = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in allowed})


@dataclass
class CostConfig:
    replacement_cost_multiplier: float = 0.75
    training_cost_per_employee: float = 50_000.0
    coaching_cost_per_employee: float = 15_000.0


@dataclass
class MarketLookup:
    """Callable market-salary resolver built from the market table."""

    table: pd.DataFrame
    field_names: tuple[str, str, str, str] = ("department", "level", "location", "year")

    def __call__(self, department: object, level: int, location: object,
                 year: int | None = None) -> float:
        sub = self.table
        if year is not None:
            exact = sub[sub["year"] == year]
            sub = exact if len(exact) else sub[sub["year"] == sub["year"].max()]
        m = sub[(sub["department"] == department) & (sub["level"] == int(level))
                & (sub["location"] == location)]
        return float(m["market_salary"].iloc[-1]) if len(m) else float("nan")


@dataclass
class WhatIfResult:
    risk_before: float
    risk_after: float
    reduction_pp: float
    confidence: float
    interventions: dict[str, float | bool]
    annual_cost: float = 0.0
    annual_cost_avoided: float = 0.0


class WhatIfEngine:
    def __init__(self, market: MarketLookup, overtime_mean: float, overtime_std: float,
                 numeric_bounds: dict[str, tuple[float, float]],
                 costs: CostConfig | None = None) -> None:
        self.market = market
        self.overtime_mean = overtime_mean
        self.overtime_std = max(overtime_std, 1e-6)
        self.bounds = numeric_bounds
        self.costs = costs or CostConfig()

    # ------------------------------------------------------------------
    def apply(self, row: pd.Series, params: InterventionParams) -> pd.Series:
        r = row.copy()
        year = pd.Timestamp(row["date"]).year if "date" in row else None

        if params.salary_raise_pct:
            r["salary"] = float(r["salary"]) * (1 + params.salary_raise_pct / 100)
        if params.promotion:
            r["level"] = min(int(r["level"]) + 1, 5)
            r["salary"] = float(r["salary"]) * 1.12
            r["months_since_promotion"] = 0
            r["promo_pressure"] = 0.0
            r["promos_last_12m"] = float(r.get("promos_last_12m", 0)) + 1

        if params.salary_raise_pct or params.promotion:
            mkt = self.market(r["department"], int(r["level"]), r["location"], year)
            new_ratio = float(r["salary"]) / mkt
            if not np.isnan(new_ratio):
                # peer-relative feature shifts by the individual delta
                if "comp_peer_gap" in r.index:
                    r["comp_peer_gap"] = float(r["comp_peer_gap"]) + (
                        new_ratio - float(r["comp_ratio"]))
                r["comp_ratio"] = new_ratio

        if params.workload_reduction_pct:
            old_ot = float(r["overtime_hours"])
            r["overtime_hours"] = max(0.0, old_ot * (1 - params.workload_reduction_pct / 100))
            r["workload_z"] = (float(r["overtime_hours"]) - self.overtime_mean) \
                / self.overtime_std
            if "overtime_peer_gap" in r.index:
                r["overtime_peer_gap"] = float(r["overtime_peer_gap"]) + (
                    float(r["overtime_hours"]) - old_ot)

        if params.engagement_boost_points:
            r["engagement_score"] = min(100.0, float(r["engagement_score"])
                                        + params.engagement_boost_points)
        if params.manager_sentiment_boost:
            r["manager_sentiment"] = min(1.0, float(r["manager_sentiment"])
                                         + params.manager_sentiment_boost)

        r["salary_log"] = np.log(max(float(r["salary"]), 1.0))
        return r

    # ------------------------------------------------------------------
    def confidence(self, row: pd.Series) -> float:
        """Share of numeric features still inside training p1-p99 envelope."""
        inside, total = 0, 0
        for feat in NUMERIC_FEATURES_FOR_BOUNDS:
            if feat not in row.index or feat not in self.bounds:
                continue
            lo, hi = self.bounds[feat]
            total += 1
            if lo <= float(row[feat]) <= hi:
                inside += 1
        return round(inside / total, 3) if total else 0.5

    def simulate(self, row: pd.Series, params: InterventionParams,
                 score_fn) -> WhatIfResult:
        before = float(score_fn(pd.DataFrame([row]))[0])
        after_row = self.apply(row, params)
        after = float(score_fn(pd.DataFrame([after_row]))[0])
        result = WhatIfResult(
            risk_before=round(before, 4),
            risk_after=round(after, 4),
            reduction_pp=round((before - after) * 100, 2),
            confidence=self.confidence(after_row),
            interventions=params.to_dict(),
        )
        result.annual_cost = self._unit_cost(row, params)
        result.annual_cost_avoided = (
            max(0.0, before - after)
            * float(row["salary"])
            * self.costs.replacement_cost_multiplier
        )
        return result

    # ------------------------------------------------------------------
    def cohort_scenario(self, rows: pd.DataFrame, params: InterventionParams,
                        score_batch_fn) -> dict[str, object]:
        if not len(rows):
            return {"n_employees": 0}
        before = np.asarray(score_batch_fn(rows), dtype=float)
        applied = pd.DataFrame([self.apply(r, params) for _, r in rows.iterrows()])
        after = np.asarray(score_batch_fn(applied), dtype=float)

        deltas = np.clip(before - after, 0, None)
        salaries = rows["salary"].to_numpy(dtype=float)
        avoided = float((deltas * salaries * self.costs.replacement_cost_multiplier).sum())
        unit_costs = np.array([
            self._unit_cost(r, params) for _, r in rows.iterrows()
        ])
        cost = float(unit_costs.sum())

        return {
            "n_employees": int(len(rows)),
            "avg_risk_before": round(float(before.mean()), 4),
            "avg_risk_after": round(float(after.mean()), 4),
            "expected_attrition_before": f"{before.mean() * 100:.1f}%",
            "expected_attrition_after": f"{after.mean() * 100:.1f}%",
            "employees_moved_out_of_high_risk": int(((before >= 0.6) & (after < 0.6)).sum()),
            "annual_cost_of_intervention": round(cost, 0),
            "annual_cost_avoided": round(avoided, 0),
            "roi_multiple": round(avoided / cost, 2) if cost > 0 else None,
            "interventions": params.to_dict(),
        }

    # ------------------------------------------------------------------
    def _unit_cost(self, row: pd.Series, params: InterventionParams) -> float:
        salary = float(row["salary"])
        cost = 0.0
        if params.salary_raise_pct:
            cost += salary * params.salary_raise_pct / 100
        if params.promotion:
            cost += salary * 0.12
        if params.engagement_boost_points:
            cost += self.costs.training_cost_per_employee
        if params.manager_sentiment_boost:
            cost += self.costs.coaching_cost_per_employee
        return cost
