"""Synthetic enterprise workforce simulator with explicit causal ground truth.

Simulates a monthly panel of employees over N months:

* static attributes (demographics, department, level, location, education)
* time-varying state (salary vs market, performance, engagement, manager
  sentiment, workload, promotion history)
* a *known structural attrition hazard* (logistic in the drivers) so that
  downstream SHAP/causal analyses can be validated against ground truth
* an A/B-style compensation pilot (randomised treatment) enabling causal
  effect estimation
* quarterly review text whose topics derive from the same latent drivers,
  giving the NLP layer verifiable labels

All randomness flows through one seeded generator => deterministic output.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit

# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------
DEPARTMENTS: dict[str, dict[str, float]] = {
    "Engineering": {"share": 0.32, "market_factor": 1.15, "overtime_base": 8.0},
    "Sales": {"share": 0.22, "market_factor": 1.00, "overtime_base": 6.0},
    "Support": {"share": 0.14, "market_factor": 0.85, "overtime_base": 7.0},
    "Operations": {"share": 0.12, "market_factor": 0.90, "overtime_base": 5.0},
    "Marketing": {"share": 0.10, "market_factor": 0.95, "overtime_base": 5.0},
    "Finance": {"share": 0.10, "market_factor": 0.98, "overtime_base": 5.0},
}

LOCATIONS: dict[str, float] = {
    "Bengaluru": 1.10,
    "Mumbai": 1.15,
    "Gurugram": 1.12,
    "Hyderabad": 1.05,
    "Pune": 1.00,
}

LEVEL_BASE_SALARY = {1: 800_000.0, 2: 1_400_000.0, 3: 2_200_000.0, 4: 3_200_000.0,
                     5: 4_500_000.0}
LEVEL_EXPECTED_MONTHS = np.array([0, 24, 30, 36, 48, 60])

ROLE_TITLES: dict[str, list[str]] = {
    "Engineering": ["Software Engineer", "Senior Software Engineer", "Lead Engineer",
                    "Engineering Manager", "Principal Engineer"],
    "Sales": ["Sales Executive", "Senior Sales Executive", "Regional Sales Lead",
              "Sales Manager", "Director of Sales"],
    "Support": ["Support Associate", "Senior Support Associate", "Support Lead",
                "Support Manager", "Head of Support"],
    "Operations": ["Operations Analyst", "Senior Operations Analyst", "Operations Lead",
                   "Operations Manager", "Head of Operations"],
    "Marketing": ["Marketing Associate", "Marketing Specialist", "Marketing Lead",
                  "Marketing Manager", "Head of Marketing"],
    "Finance": ["Financial Analyst", "Senior Financial Analyst", "Finance Lead",
                "Finance Manager", "Head of Finance"],
}

REVIEW_TOPICS = ["compensation", "workload", "career_progression", "management",
                 "recognition", "work_life_balance"]

NEGATIVE_TEMPLATES: dict[str, list[str]] = {
    "compensation": [
        "My pay has fallen behind the market and other companies offer much more for "
        "similar roles.",
        "Compensation adjustments have not kept up with my responsibilities this year.",
        "Salary is the main reason I have started looking at offers elsewhere.",
    ],
    "workload": [
        "We have been working weekends for three months and the workload is unsustainable.",
        "Constant firefighting means long evenings every week with no recovery time.",
        "The team is chronically understaffed and the pressure keeps building.",
    ],
    "career_progression": [
        "There is no clear promotion path and I have been at this level far longer than "
        "expected.",
        "Promotions seem frozen despite consistently strong delivery.",
        "I cannot see a growth path for myself here anymore.",
    ],
    "management": [
        "My manager gives little feedback and does not advocate for the team.",
        "Leadership ignores concerns raised in one-on-ones.",
        "I do not feel supported by my direct manager.",
    ],
    "recognition": [
        "Good work goes unnoticed here.",
        "There is no recognition for the extra effort we put in.",
    ],
    "work_life_balance": [
        "I cannot remember my last proper weekend off.",
        "Work-life balance is nonexistent right now.",
    ],
}

POSITIVE_TEMPLATES: dict[str, list[str]] = {
    "compensation": ["Compensation is competitive and I feel paid fairly."],
    "workload": ["Workload feels manageable lately."],
    "career_progression": ["I see a clear growth path ahead of me."],
    "management": ["My manager actively supports my growth."],
    "recognition": ["Effort is recognised and rewarded here."],
    "work_life_balance": ["I can maintain a healthy balance right now."],
}


@dataclass
class GeneratorParams:
    n_employees: int = 8_000
    n_months: int = 36
    start_date: str = "2023-01-01"
    seed: int = 42
    target_annual_attrition: float = 0.16
    treatment_start_month: int = 18
    treatment_share: float = 0.25
    review_share_per_quarter: float = 0.85


@dataclass
class GroundTruth:
    """True structural coefficients of the simulated hazard."""

    intercept: float = 0.0
    comp_gap: float = 2.4
    promo_pressure: float = 1.1
    workload_z: float = 0.45
    manager_negative: float = 1.5
    low_engagement: float = 0.8
    loyalty: float = -1.2


def role_for(department: str, level: int) -> str:
    return ROLE_TITLES[department][level - 1]


class SyntheticWorkforceGenerator:
    """Deterministic enterprise workforce simulation.

    Risk is driven mostly by *observable* state (compensation vs market,
    promotion pressure, workload, manager sentiment, engagement) with a small
    latent random effect - matching enterprise settings where attrition models
    achieve strong discrimination.
    """

    ANNUAL_HIRE_RATE = 0.13
    RANDOM_EFFECT_STD = 0.30

    def __init__(self, params: GeneratorParams | None = None) -> None:
        self.p = params or GeneratorParams()
        self.rng = np.random.default_rng(self.p.seed)

    # ------------------------------------------------------------------
    # Static employee attributes
    # ------------------------------------------------------------------
    def _static_frame(self, n_rows: int, id_start: int = 1) -> pd.DataFrame:
        n = n_rows
        rng = self.rng
        depts = list(DEPARTMENTS)
        probs = np.array([DEPARTMENTS[d]["share"] for d in depts])
        department = rng.choice(depts, size=n, p=probs / probs.sum())
        location = rng.choice(list(LOCATIONS), size=n)
        tenure_years = rng.uniform(0, 8, size=n)
        level = np.clip(
            1 + np.floor(tenure_years / 2.2 + rng.normal(0, 0.7, size=n)), 1, 5
        ).astype(int)
        start = pd.Timestamp(self.p.start_date)
        hire_date = start - pd.to_timedelta(tenure_years * 365.25, unit="D")
        return pd.DataFrame(
            {
                "employee_id": [f"EMP{i:05d}" for i in range(id_start, id_start + n)],
                "age": rng.integers(22, 56, size=n),
                "gender": rng.choice(["M", "F", "Other"], size=n, p=[0.58, 0.40, 0.02]),
                "education": rng.choice(["Bachelor", "Master", "PhD"], size=n,
                                        p=[0.55, 0.33, 0.12]),
                "department": department,
                "location": location,
                "role": [role_for(d, int(lv)) for d, lv in zip(department, level, strict=False)],
                "level": level,
                "hire_date": hire_date,
            }
        )

    def _market_table(self) -> pd.DataFrame:
        years = range(2023, 2023 + (self.p.n_months // 12) + 2)
        rows = [
            {
                "year": year,
                "department": dept,
                "level": level,
                "location": loc,
                "market_salary": (
                    LEVEL_BASE_SALARY[level]
                    * DEPARTMENTS[dept]["market_factor"]
                    * LOCATIONS[loc]
                    * 1.06 ** (year - 2023)
                ),
            }
            for year in years
            for dept in DEPARTMENTS
            for level in LEVEL_BASE_SALARY
            for loc in LOCATIONS
        ]
        return pd.DataFrame(rows)

    @staticmethod
    def _lookup_market(market: pd.DataFrame, year: int, dept: np.ndarray,
                       level: np.ndarray, loc: np.ndarray) -> np.ndarray:
        sub = market[market["year"] == year].set_index(["department", "level", "location"])
        idx = pd.MultiIndex.from_arrays([dept, level, loc])
        return sub["market_salary"].reindex(idx).to_numpy()

    # ------------------------------------------------------------------
    # Monthly panel simulation
    # ------------------------------------------------------------------
    def _calibrate_intercept(self, parts: np.ndarray, monthly_target: float) -> None:
        """Set the intercept so E[hazard] ~= monthly_target at t=0 covariates."""
        mean_lp = float(np.mean(parts))
        self.intercept = float(np.log(monthly_target / (1 - monthly_target)) - mean_lp)

    def generate(self) -> dict[str, Any]:
        """Two-pass simulation: analytic intercept, then empirical correction.

        Comp gaps and promotion pressure accumulate over the timeline, so a
        single t=0 calibration underestimates lifetime attrition. Pass 1
        measures the realized rate; pass 2 shifts the intercept so the full-
        horizon rate matches the target.
        """
        p = self.p
        monthly_target = 1 - (1 - p.target_annual_attrition) ** (1 / 12)

        self._intercept = 0.0
        first = self._simulate()
        realized = float(first["snapshots"]["attrition_event"].mean())
        correction = float(np.log(monthly_target / (1 - monthly_target))
                           - np.log(realized / (1 - realized)))
        self._intercept += correction

        result = self._simulate()
        gt = result["ground_truth"]
        gt["intercept"] = self.intercept
        gt["intercept_correction_pass2"] = round(correction, 4)
        return result

    @property
    def intercept(self) -> float:
        return self._intercept

    @intercept.setter
    def intercept(self, value: float) -> None:
        self._intercept = value

    def _simulate(self) -> dict[str, pd.DataFrame | dict]:
        p = self.p
        self.rng = np.random.default_rng(self.p.seed)  # fresh => deterministic per pass
        rng = self.rng

        # Base workforce plus a reserve pool that gets "hired" each January,
        # keeping tenure/headcount distributions stationary across the timeline.
        n_base = p.n_employees
        n_reserve = int(n_base * self.ANNUAL_HIRE_RATE * (p.n_months // 12 + 1))
        employees = pd.concat(
            [self._static_frame(n_base), self._static_frame(n_reserve, n_base + 1)],
            ignore_index=True,
        )
        market = self._market_table()

        n = len(employees)
        dept = employees["department"].to_numpy()
        loc = employees["location"].to_numpy()
        dept_ot = np.vectorize(lambda d: DEPARTMENTS[d]["overtime_base"])(dept)

        ability = rng.normal(0, 1, n)
        ambition = rng.normal(0, 1, n)
        loyalty = np.clip(rng.normal(0, 1, n), -2.5, 2.5)
        random_effect = rng.normal(0, self.RANDOM_EFFECT_STD, n)

        level = employees["level"].to_numpy().astype(int)
        salary = np.array([
            LEVEL_BASE_SALARY[int(lv)] * DEPARTMENTS[d]["market_factor"] * LOCATIONS[lc]
            for lv, d, lc in zip(level, dept, loc, strict=False)
        ])
        salary *= np.exp(rng.normal(-0.08, 0.12, n))
        expected = LEVEL_EXPECTED_MONTHS[level]
        months_since_promo = (rng.uniform(0.25, 1.0, n) * expected).clip(min=2)
        promo_count = (level - 1).astype(float)
        perf = np.clip(3.0 + 0.6 * ability + rng.normal(0, 0.35, n), 1.0, 5.0)
        mgr_sentiment = np.clip(rng.uniform(-0.6, 0.8, n) + rng.normal(0, 0.15, n), -1, 1)
        overtime = np.clip(dept_ot + rng.normal(0, 2, n), 0, 30)

        activated = np.zeros(n, dtype=bool)
        activated[:n_base] = True
        alive = activated.copy()
        exit_month = np.full(n, -1, dtype=int)
        treatment_flag = np.zeros(n, dtype=int)

        snap_cols: dict[str, list[np.ndarray]] = {c: [] for c in [
            "employee_id", "month_index", "date", "age", "tenure_months", "department",
            "location", "education", "gender", "level", "months_since_promotion",
            "promotion_count", "salary", "comp_ratio", "performance_rating",
            "engagement_score", "manager_sentiment", "overtime_hours",
            "attrition_hazard_true", "attrition_event", "treatment_flag"]}
        reviews_rows: list[dict[str, object]] = []
        n_hired = 0

        emp_ids = employees["employee_id"].to_numpy()
        edu = employees["education"].to_numpy()
        gender = employees["gender"].to_numpy()
        age_hire = rng.integers(22, 50, size=n).astype(float)
        hires = np.full(n, np.datetime64(p.start_date, "D"), dtype="datetime64[D]")
        start_ts = pd.Timestamp(p.start_date)
        monthly_target = 1 - (1 - p.target_annual_attrition) ** (1 / 12)
        engagement = np.full(n, 70.0)

        # Analytic intercept calibration (pass 1 only; pass 2 keeps correction)
        if not hasattr(self, "_intercept") or self._intercept == 0.0:
            market0 = self._lookup_market(market, start_ts.year,
                                          dept[:n_base], level[:n_base], loc[:n_base])
            parts0 = (
                GroundTruth.comp_gap * np.clip(1 - salary[:n_base] / market0, 0, 0.5)
                + GroundTruth.promo_pressure
                * np.clip((months_since_promo[:n_base] - expected[:n_base])
                          / expected[:n_base], 0, 1.5)
                + GroundTruth.workload_z * (overtime[:n_base] - overtime.mean())
                / max(overtime.std(), 1e-6)
                + GroundTruth.manager_negative
                * np.clip(-mgr_sentiment[:n_base], 0, 1)
                + GroundTruth.loyalty * loyalty[:n_base] / 2
            )
            self._calibrate_intercept(parts0, monthly_target)

        gt_coefs = {
            "intercept": self.intercept,
            "comp_gap": GroundTruth.comp_gap,
            "promo_pressure": GroundTruth.promo_pressure,
            "workload_z": GroundTruth.workload_z,
            "manager_negative": GroundTruth.manager_negative,
            "low_engagement": GroundTruth.low_engagement,
            "loyalty": GroundTruth.loyalty,
        }

        # ------------------------------------------------------------------
        for m in range(p.n_months):
            date = start_ts + pd.DateOffset(months=m)
            year = int(date.year)
            ids_alive = np.flatnonzero(alive)

            expected = LEVEL_EXPECTED_MONTHS[level]
            market_now = self._lookup_market(market, year, dept, level, loc)
            comp_ratio = salary / market_now
            comp_gap = np.clip(1 - comp_ratio, 0, 0.5)
            promo_pressure = np.clip((months_since_promo - expected) / expected, 0, 1.5)
            workload_z = (overtime - overtime.mean()) / max(overtime.std(), 1e-6)
            mgr_neg = np.clip(-mgr_sentiment, 0, 1)

            if m % 3 == 0:
                # NOTE: deliberately independent of comp_gap so that
                # compensation risk is not mediated through engagement -
                # keeps SHAP attributions actionable per driver.
                engagement = np.clip(
                    74 + 10 * loyalty - 7 * np.clip(workload_z, -2, 3)
                    + 4 * mgr_sentiment + rng.normal(0, 3, n),
                    0, 100,
                )

            lp = (
                self.intercept
                + gt_coefs["comp_gap"] * comp_gap
                + gt_coefs["promo_pressure"] * promo_pressure
                + gt_coefs["workload_z"] * workload_z
                + gt_coefs["manager_negative"] * mgr_neg
                + gt_coefs["low_engagement"] * np.clip((65 - engagement) / 15, 0, 2)
                + gt_coefs["loyalty"] * loyalty / 2
                + random_effect
            )
            hazard = expit(lp)
            events = (rng.random(n) < hazard) & alive
            newly_gone = np.flatnonzero(events)
            exit_month[newly_gone] = m

            # ---- snapshot (pre-event state; label looks forward) ----------
            take = ids_alive
            snap_cols["employee_id"].append(emp_ids[take])
            snap_cols["month_index"].append(np.full(len(take), m))
            snap_cols["date"].append(np.full(len(take), np.datetime64(date.date())))
            snap_cols["age"].append(
                age_hire[take]
                + np.asarray(np.datetime64(date.date()) - hires[take],
                             dtype="timedelta64[D]").astype(float) / 365.25
            )
            snap_cols["tenure_months"].append(
                np.asarray(np.datetime64(date.date()) - hires[take],
                           dtype="timedelta64[D]").astype(float) // 30
            )
            snap_cols["department"].append(dept[take])
            snap_cols["location"].append(loc[take])
            snap_cols["education"].append(edu[take])
            snap_cols["gender"].append(gender[take])
            snap_cols["level"].append(level[take])
            snap_cols["months_since_promotion"].append(months_since_promo[take])
            snap_cols["promotion_count"].append(promo_count[take])
            snap_cols["salary"].append(salary[take])
            snap_cols["comp_ratio"].append(comp_ratio[take])
            snap_cols["performance_rating"].append(perf[take])
            snap_cols["engagement_score"].append(engagement[take])
            snap_cols["manager_sentiment"].append(mgr_sentiment[take])
            snap_cols["overtime_hours"].append(overtime[take])
            snap_cols["attrition_hazard_true"].append(hazard[take])
            snap_cols["attrition_event"].append(events[take].astype(int))
            snap_cols["treatment_flag"].append(treatment_flag[take])

            if m % 3 == 0:
                reviews_rows.extend(
                    self._make_reviews(date, emp_ids, comp_gap, workload_z,
                                       promo_pressure, mgr_neg, loyalty, alive)
                )

            # ---- transitions for survivors --------------------------------
            surv = np.flatnonzero(alive & ~events)

            if m > 0 and m % 12 == 0:  # annual raise cycle
                merit = 0.01 * (perf >= 4.0) - 0.005 * (perf <= 2.0)
                salary[surv] *= 1 + 0.04 + rng.random(len(surv)) * 0.02 + merit[surv]

            if m > 0 and m % 12 == 4:  # annual hiring intake
                k = int(round(self.ANNUAL_HIRE_RATE * max(alive.sum(), 1)))
                pool = np.flatnonzero(~activated)[:k]
                if len(pool):
                    activated[pool] = True
                    alive[pool] = True
                    hires[pool] = np.datetime64(date.date())
                    lv_new = rng.choice([1, 2], len(pool), p=[0.7, 0.3])
                    level[pool] = lv_new
                    mkt_hire = self._lookup_market(market, year, dept[pool],
                                                   lv_new.astype(int), loc[pool])
                    salary[pool] = mkt_hire * np.exp(rng.normal(-0.03, 0.07, len(pool)))
                    LEVEL_EXPECTED_MONTHS[lv_new]
                    months_since_promo[pool] = rng.integers(1, 6, len(pool))
                    promo_count[pool] = lv_new - 1
                    perf[pool] = np.clip(3.0 + 0.5 * ability[pool]
                                         + rng.normal(0, 0.3, len(pool)), 1.0, 5.0)
                    engagement[pool] = np.clip(
                        74 + 8 * loyalty[pool] + rng.normal(0, 5, len(pool)), 0, 100)
                    mgr_sentiment[pool] = rng.uniform(-0.4, 0.7, len(pool))
                    overtime[pool] = np.clip(dept_ot[pool] + rng.normal(0, 2, len(pool)),
                                             0, 30)
                    n_hired = n_hired + len(pool)

            if m == p.treatment_start_month:  # randomised compensation pilot
                pool = surv[comp_ratio[surv] < 0.95]
                treated = pool[rng.random(len(pool)) < p.treatment_share]
                salary[treated] *= 1.06
                treatment_flag[treated] = 1

            elig = surv[months_since_promo[surv] >= 12]
            if len(elig):
                # promote-or-lose dynamics: high backlog => high promotion odds,
                # keeping promo_pressure near steady state across the timeline
                pl_logit = (
                    -2.2
                    + 0.45 * (perf[elig] - 3)
                    + 0.35 * ambition[elig]
                    + 2.2 * np.clip((months_since_promo[elig] - expected[elig])
                                    / expected[elig], 0, 2)
                )
                promoted = elig[rng.random(len(elig)) < expit(pl_logit) * 0.05]
                level[promoted] = np.minimum(level[promoted] + 1, 5)
                salary[promoted] *= rng.uniform(1.08, 1.15, len(promoted))
                months_since_promo[promoted] = 0
                promo_count[promoted] += 1

            months_since_promo[surv] += 1

            if m % 3 == 2:  # quarterly performance refresh
                fresh = np.clip(3.0 + 0.6 * ability - 0.15 * np.clip(workload_z - 1, 0, None)
                                + rng.normal(0, 0.35, n), 1.0, 5.0)
                perf[surv] = fresh[surv]

            crunch = 4.0 if m % 24 in (10, 11, 22, 23) else 0.0
            seasonal = 3 * np.sin(2 * np.pi * (m % 12) / 12)
            overtime = np.where(
                alive, dept_ot + seasonal + crunch + rng.normal(0, 2, n), overtime
            ).clip(0, 30)
            mgr_sentiment = np.where(
                alive, np.clip(mgr_sentiment + rng.normal(0, 0.05, n), -1, 1), mgr_sentiment
            )

            alive &= ~events

        snapshots = pd.DataFrame({k: np.concatenate(v) for k, v in snap_cols.items()})
        snapshots = snapshots.sort_values(["employee_id", "month_index"]).reset_index(drop=True)
        snapshots = self._attach_exit_and_label(snapshots, emp_ids, exit_month)

        reviews = pd.DataFrame(reviews_rows)
        n_leavers = int((exit_month >= 0).sum())
        realized_annual = float(snapshots.groupby("month_index")["attrition_event"]
                                .mean().mean()) * 12

        ground_truth = {
            "hazard_coefficients": gt_coefs,
            "random_effect_std": self.RANDOM_EFFECT_STD,
            "n_employees_base": n_base,
            "n_employees_total": n,
            "n_hired": int(n_hired),
            "final_headcount": int(alive.sum()),
            "n_snapshots": len(snapshots),
            "realized_annual_attrition": round(realized_annual, 4),
            "n_leavers": n_leavers,
            "treatment_assigned": int(treatment_flag.sum()),
            "treatment_start_month": p.treatment_start_month,
            "seed": p.seed,
        }
        return {
            "employees": employees,
            "snapshots": snapshots,
            "reviews": reviews,
            "market_salary": market,
            "ground_truth": ground_truth,
        }

    # ------------------------------------------------------------------
    def _make_reviews(self, date: pd.Timestamp, emp_ids: np.ndarray,
                      comp_gap: np.ndarray, workload_z: np.ndarray,
                      promo_pressure: np.ndarray, mgr_neg: np.ndarray,
                      loyalty: np.ndarray, alive: np.ndarray) -> list[dict[str, object]]:
        rng = self.rng
        idx = np.flatnonzero(alive)
        idx = idx[rng.random(len(idx)) < self.p.review_share_per_quarter]
        issues = np.column_stack([
            comp_gap[idx] * 3.0,
            np.clip(workload_z[idx], 0, None) * 0.8,
            promo_pressure[idx],
            mgr_neg[idx] * 0.9,
            np.full(len(idx), 0.1),
            np.clip(0.4 - loyalty[idx] * 0.3, 0, None),
        ])
        order = np.argsort(-issues, axis=1)
        rows: list[dict[str, object]] = []
        for k, i in enumerate(idx):
            primary = REVIEW_TOPICS[order[k, 0]]
            secondary = REVIEW_TOPICS[order[k, 1]]
            intensity = float(np.clip(issues[k, order[k, 0]], 0, 3))
            sentiment = float(np.clip(0.4 - 0.55 * intensity + 0.15 * loyalty[i]
                                      + rng.normal(0, 0.15), -1, 1))
            label = ("negative" if sentiment < -0.15
                     else "positive" if sentiment > 0.25 else "neutral")
            bank = (NEGATIVE_TEMPLATES[primary] if label == "negative"
                    else POSITIVE_TEMPLATES[primary])
            text = bank[int(rng.integers(len(bank)))]
            if label == "negative" and rng.random() < 0.4:
                sbank = NEGATIVE_TEMPLATES[secondary]
                text += " " + sbank[int(rng.integers(len(sbank)))]
            rows.append({
                "employee_id": emp_ids[i],
                "date": np.datetime64(date.date()),
                "topic_primary": primary,
                "topic_secondary": secondary,
                "sentiment_score": round(sentiment, 3),
                "sentiment_label": label,
                "text": text,
            })
        return rows

    @staticmethod
    def _attach_exit_and_label(snapshots: pd.DataFrame, emp_ids: np.ndarray,
                               exit_month: np.ndarray, horizon: int = 6) -> pd.DataFrame:
        exit_map = pd.Series(np.where(exit_month >= 0, exit_month, np.nan), index=emp_ids)
        em = snapshots["employee_id"].map(exit_map).to_numpy(dtype=float)
        mi = snapshots["month_index"].to_numpy(dtype=float)
        last_m = float(mi.max())
        snapshots["exit_month"] = em
        snapshots["attrited_within_6m"] = np.where(
            (~np.isnan(em)) & (em > mi) & (em <= mi + horizon), 1.0,
            np.where(np.isnan(em) & (mi + horizon > last_m), np.nan, 0.0),
        )
        return snapshots


def main(out_dir: str | Path = "artifacts/data") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = SyntheticWorkforceGenerator().generate()
    result["employees"].to_parquet(out / "employees.parquet", index=False)
    result["snapshots"].to_parquet(out / "snapshots.parquet", index=False)
    result["reviews"].to_parquet(out / "reviews.parquet", index=False)
    result["market_salary"].to_parquet(out / "market_salary.parquet", index=False)
    (out / "ground_truth.json").write_text(
        json.dumps(result["ground_truth"], indent=2, default=str), encoding="utf-8"
    )
    gt = result["ground_truth"]
    print(f"Generated {gt['n_snapshots']} snapshots for base cohort of "
          f"{gt['n_employees_base']} (+{gt['n_hired']} hired, final headcount "
          f"{gt['final_headcount']}); {gt['n_leavers']} leavers, realized annual "
          f"attrition {gt['realized_annual_attrition']:.1%} -> {out}")


if __name__ == "__main__":
    main()
