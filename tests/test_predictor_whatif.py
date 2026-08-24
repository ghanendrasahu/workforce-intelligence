"""Predictor + what-if engine behaviour on the trained tiny stack."""
from __future__ import annotations

import pandas as pd
from conftest import TinyStack

from ml.explainability.whatif import InterventionParams


def test_risk_scoring_bounds(tiny_stack: TinyStack) -> None:
    reg = tiny_stack["registry"]
    rows = reg._latest.head(50)
    risks = reg.calibrated_scores(rows)
    assert ((risks >= 0) & (risks <= 1)).all()


def test_salary_raise_reduces_underpaid_cohort_risk(
        tiny_stack: TinyStack) -> None:
    """Causal direction check: paying underpaid staff more must lower risk."""
    reg = tiny_stack["registry"]
    underpaid = reg._latest[reg._latest["comp_ratio"] < 0.85].head(200)
    assert len(underpaid) >= 20
    before = reg.raw_scores(underpaid).mean()
    raised = pd.DataFrame([reg.whatif.apply(r, InterventionParams(salary_raise_pct=15))
                           for _, r in underpaid.iterrows()])
    after = reg.raw_scores(raised).mean()
    assert after < before, f"raise should reduce risk: {before:.4f} -> {after:.4f}"


def test_local_drivers_sum_to_risk(tiny_stack: TinyStack) -> None:
    reg = tiny_stack["registry"]
    row = reg.latest_row(reg.employee_ids(5)[-1])
    local = reg.local_drivers(row, top_k=10)
    baseline_pp = local["baseline_risk"] * 100  # type: ignore[operator]
    total = baseline_pp + sum(c["risk_points"] for c in local["contributions"])  # type: ignore[union-attr]
    final_pp = local["risk"] * 100  # type: ignore[operator]
    assert abs(total - final_pp) < 2.0, f"{total:.2f} vs {final_pp:.2f}"


def test_confidence_within_envelope(tiny_stack: TinyStack) -> None:
    reg = tiny_stack["registry"]
    row = reg.latest_row(reg.employee_ids(3)[0])
    res = reg.whatif.simulate(row, InterventionParams(salary_raise_pct=5),
                              lambda x: reg.raw_scores(x))
    assert 0.0 <= res.confidence <= 1.0
    assert res.reduction_pp != 0 or res.risk_before == res.risk_after


def test_anomaly_model_flags_extreme_salary(tiny_stack: TinyStack) -> None:
    reg = tiny_stack["registry"]
    assert reg.anomaly_model is not None
    row = reg._latest.iloc[[0]].copy()
    row["comp_ratio"] = 3.0
    feats = row[["salary_log", "comp_ratio", "level",
                 "log_tenure_months"]].astype(float)
    score_normal = float(reg.anomaly_model.decision_function(
        reg._latest.iloc[[0]][["salary_log", "comp_ratio", "level",
                               "log_tenure_months"]].astype(float))[0])
    score_extreme = float(reg.anomaly_model.decision_function(feats)[0])
    assert score_extreme < score_normal
