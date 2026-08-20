"""Generator invariants: determinism, schema, causal ground truth."""
from __future__ import annotations

import pandas as pd

from data.ingestion.synthetic_generator import GeneratorParams, SyntheticWorkforceGenerator


def make() -> dict[str, pd.DataFrame | dict]:
    return SyntheticWorkforceGenerator(
        GeneratorParams(n_employees=300, n_months=20, seed=3,
                        target_annual_attrition=0.2)
    ).generate()


def test_deterministic_same_seed() -> None:
    a = make()
    b = make()
    pd.testing.assert_frame_equal(a["snapshots"], b["snapshots"])  # type: ignore[arg-type]
    assert a["ground_truth"] == b["ground_truth"]  # type: ignore[index]


def test_schema_and_integrity() -> None:
    out = make()
    snap: pd.DataFrame = out["snapshots"]  # type: ignore[assignment]
    employees: pd.DataFrame = out["employees"]  # type: ignore[assignment]

    for col in ("employee_id", "month_index", "salary", "comp_ratio",
                "attrition_event", "exit_month", "attrited_within_6m"):
        assert col in snap.columns
    assert snap["employee_id"].isin(employees["employee_id"]).all()
    assert not snap.duplicated(["employee_id", "month_index"]).any()

    # no snapshot rows after exit
    active = snap[snap["exit_month"].notna()]
    assert (active["month_index"] <= active["exit_month"]).all()

    # events align with exit month
    events = snap[snap["attrition_event"] == 1]
    assert (events["month_index"] == events["exit_month"]).all()


def test_horizon_label_consistency() -> None:
    out = make()
    snap: pd.DataFrame = out["snapshots"]  # type: ignore[assignment]
    labelled = snap[snap["attrited_within_6m"].notna()]
    positives = labelled[labelled["attrited_within_6m"] == 1]
    merged = positives.merge(
        snap[["employee_id", "month_index"]],
        left_on=["employee_id", "exit_month"],
        right_on=["employee_id", "month_index"],
        suffixes=("", "_exit"),
    )
    # every positive row's exit must fall inside its 6-month window
    assert ((merged["exit_month"] > merged["month_index"])
            & (merged["exit_month"] <= merged["month_index"] + 6)).all()


def test_treatment_reduces_realized_hazard() -> None:
    """Diff-in-diff sanity: treated employees' hazard falls vs comparable controls."""
    out = SyntheticWorkforceGenerator(
        GeneratorParams(n_employees=1200, n_months=24, seed=11,
                        target_annual_attrition=0.2, treatment_start_month=10,
                        treatment_share=0.5)
    ).generate()
    snap: pd.DataFrame = out["snapshots"]  # type: ignore[assignment]

    pre = snap[(snap["month_index"] >= 7) & (snap["month_index"] < 10)]
    eligible_ids = set(pre.loc[pre["comp_ratio"] < 0.95, "employee_id"])
    treated_ids = set(snap.loc[snap["treatment_flag"] == 1, "employee_id"])
    control_ids = eligible_ids - treated_ids
    assert len(treated_ids) > 50 and len(control_ids) > 50

    post = snap[(snap["month_index"] >= 12) & (snap["month_index"] <= 17)]

    def delta(ids: set[str]) -> float:
        d_pre = pre[pre["employee_id"].isin(ids)]["attrition_hazard_true"].mean()
        d_post = post[post["employee_id"].isin(ids)]["attrition_hazard_true"].mean()
        return float(d_post - d_pre)

    assert delta(treated_ids) < delta(control_ids), \
        f"treated {delta(treated_ids):.5f} should drop more than control {delta(control_ids):.5f}"


def test_hiring_keeps_panel_alive() -> None:
    out = make()
    gt: dict = out["ground_truth"]  # type: ignore[assignment]
    assert gt["n_hired"] > 0
    snap: pd.DataFrame = out["snapshots"]  # type: ignore[assignment]
    # late-snapshot cohorts must contain brand-new hires (tenure near zero),
    # proving intake keeps the panel demographically alive
    late = snap[snap["month_index"] == snap["month_index"].max()]
    assert int(late["tenure_months"].min()) <= 4
