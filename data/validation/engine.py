"""Lightweight data-quality engine: completeness, validity, integrity, drift.

Emits a machine-readable report (dict/JSON) with pass/fail per check and a
global status. Deliberately dependency-free (no Great Expectations) so it can
run inside CI and the API without extra services.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def population_stability_index(expected: pd.Series, actual: pd.Series,
                               bins: int = 10) -> float:
    """PSI between two numeric distributions; <0.1 stable, >0.25 major shift."""
    edges = np.unique(np.quantile(expected.dropna(), np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    e_counts = np.histogram(expected.dropna(), bins=edges)[0] / max(len(expected), 1)
    a_counts = np.histogram(actual.dropna(), bins=edges)[0] / max(len(actual), 1)
    eps = 1e-6
    e_pct = np.clip(e_counts, eps, None)
    a_pct = np.clip(a_counts, eps, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


class DataQualityEngine:
    """Validates the workforce panel and produces a structured report."""

    def __init__(self) -> None:
        self.issues: list[dict[str, str]] = []

    def _flag(self, check: str, severity: str, detail: str) -> None:
        self.issues.append({"check": check, "severity": severity, "detail": detail})

    def run(self, snapshots: pd.DataFrame, employees: pd.DataFrame,
            reviews: pd.DataFrame) -> dict:
        self.issues = []

        # Completeness ---------------------------------------------------
        null_pct = snapshots.isna().mean()
        for col, pct in null_pct[null_pct > 0.30].items():
            if col not in ("exit_month", "attrited_within_6m"):
                self._flag("completeness", "high",
                           f"{col}: {pct:.1%} nulls exceed 30% threshold")

        # Uniqueness -----------------------------------------------------
        dupes = int(snapshots.duplicated(["employee_id", "month_index"]).sum())
        if dupes:
            self._flag("uniqueness", "high", f"{dupes} duplicate employee-month rows")

        id_dupes = int(employees["employee_id"].duplicated().sum())
        if id_dupes:
            self._flag("uniqueness", "high", f"{id_dupes} duplicate employee_ids")

        # Domain validity --------------------------------------------------
        bad_salary = int((snapshots["salary"] <= 0).sum())
        if bad_salary:
            self._flag("validity", "high", f"{bad_salary} rows with salary <= 0")
        bad_rating = int((~snapshots["performance_rating"].between(1, 5)).fillna(False).sum())
        if bad_rating:
            self._flag("validity", "medium",
                       f"{bad_rating} performance ratings outside [1,5]")
        bad_sent = int((~snapshots["manager_sentiment"].between(-1, 1)).fillna(False).sum())
        if bad_sent:
            self._flag("validity", "medium",
                       f"{bad_sent} manager_sentiment values outside [-1,1]")
        bad_comp = int((~snapshots["comp_ratio"].between(0.3, 3.0)).fillna(False).sum())
        if bad_comp:
            self._flag("validity", "medium",
                       f"{bad_comp} comp_ratio values outside [0.3,3.0]")

        # Referential integrity ---------------------------------------------
        orphan = int((~snapshots["employee_id"].isin(employees["employee_id"])).sum())
        if orphan:
            self._flag("integrity", "high", f"{orphan} snapshot rows with unknown employee")
        orphan_rev = int((~reviews["employee_id"].isin(employees["employee_id"])).sum())
        if orphan_rev:
            self._flag("integrity", "medium",
                       f"{orphan_rev} review rows with unknown employee")

        # Temporal consistency ------------------------------------------------
        after_exit = snapshots[snapshots["exit_month"].notna()
                               & (snapshots["month_index"] > snapshots["exit_month"])]
        if len(after_exit):
            self._flag("temporal", "high",
                       f"{len(after_exit)} snapshot rows exist after exit_month")
        event_no_exit = snapshots[(snapshots["attrition_event"] == 1)
                                  & snapshots["exit_month"].isna()]
        if len(event_no_exit):
            self._flag("temporal", "high",
                       f"{len(event_no_exit)} attrition events lack an exit_month")
        mismatch = snapshots[
            snapshots["attrited_within_6m"].notna()
            & (snapshots["month_index"] >= snapshots["exit_month"].fillna(1e9))
            & (snapshots["attrition_event"] != 1)
        ]
        if len(mismatch):
            self._flag("temporal", "medium",
                       f"{len(mismatch)} post-exit label anomalies")

        # Drift: first vs last third of the timeline -------------------------
        mi = snapshots["month_index"]
        cut_lo, cut_hi = mi.max() // 3, 2 * mi.max() // 3
        drift_cols = ["comp_ratio", "overtime_hours", "engagement_score", "salary"]
        drift: dict[str, float] = {}
        for col in drift_cols:
            early = snapshots[mi <= cut_lo][col].astype(float)
            late = snapshots[mi >= cut_hi][col].astype(float)
            psi = population_stability_index(early, late)
            drift[col] = round(psi, 4)
            if psi > 0.25:
                self._flag("drift", "medium", f"{col}: PSI={psi:.2f} indicates major drift")

        passed = not any(i["severity"] == "high" for i in self.issues)
        return {
            "status": "pass" if passed else "fail",
            "n_checks_triggered": len(self.issues),
            "n_rows_snapshots": len(snapshots),
            "n_employees": len(employees),
            "n_reviews": len(reviews),
            "null_pct_max_excl_labels": round(
                float(null_pct.drop(["exit_month", "attrited_within_6m"]).max()), 4),
            "drift_psi": drift,
            "issues": self.issues,
        }

    @staticmethod
    def save_report(report: dict, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return p


def main(data_dir: str | Path = "artifacts/data") -> dict:
    from pathlib import Path as _P

    d = _P(data_dir)
    engine = DataQualityEngine()
    report = engine.run(
        pd.read_parquet(d / "snapshots.parquet"),
        pd.read_parquet(d / "employees.parquet"),
        pd.read_parquet(d / "reviews.parquet"),
    )
    out = DataQualityEngine.save_report(report, d / "quality_report.json")
    print(f"Quality status: {report['status']} ({report['n_checks_triggered']} issues) "
          f"-> {out}")
    return report


if __name__ == "__main__":
    main()
