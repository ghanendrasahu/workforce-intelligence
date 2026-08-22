"""Feature engineering and dataset assembly for training/serving.

Defines the single source of truth for feature columns so that training,
inference and the what-if simulator all operate on identical inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_LEVEL_EXPECTED_SERIES = pd.Series(
    {1: 24, 2: 30, 3: 36, 4: 48, 5: 60}, dtype="float64", name="expected"
)

TARGET = "attrited_within_6m"
TIME_COL = "month_index"
ID_COL = "employee_id"

CATEGORICAL_FEATURES = ["department", "location", "education", "gender"]
NUMERIC_FEATURES = [
    "age",
    "log_tenure_months",
    "level",
    "comp_ratio",
    "comp_peer_gap",
    "performance_rating",
    "engagement_score",
    "manager_sentiment",
    "overtime_hours",
    "workload_z",
    "overtime_peer_gap",
    "promo_pressure",
    "promos_last_12m",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Rolling-origin temporal split, capped at fully-observable label months.
# With an H-month horizon, snapshots beyond (last_month - H) would contain
# only leaver rows (survivor labels unobservable) -> composition bias. Windows
# are derived from the timeline length in derive_splits(); keeping training
# adjacent to the test window minimizes covariate drift.
HORIZON_M = 6


def derive_splits(last_month: int, horizon: int,
                  min_start: int = 6) -> dict[str, tuple[int, int]]:
    """Temporal windows: last 3 observable months = test, 3 before = val."""
    cutoff = last_month - horizon
    if cutoff < min_start + 6:
        raise ValueError(f"timeline too short for horizon {horizon} "
                         f"(last_month={last_month})")
    return {
        "train": (min_start, cutoff - 6),
        "val": (cutoff - 5, cutoff - 3),
        "test": (cutoff - 2, cutoff),
    }

PROMO_TARGET = "promoted_within_12m"
PROMO_HORIZON = 12
HORIZON_M = 6


@dataclass
class DatasetBundle:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    feature_columns: list[str]
    categorical_features: list[str]
    overtime_mean: float
    overtime_std: float


def load_frames(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    d = Path(data_dir)
    return {
        "snapshots": pd.read_parquet(d / "snapshots.parquet"),
        "employees": pd.read_parquet(d / "employees.parquet"),
        "reviews": pd.read_parquet(d / "reviews.parquet"),
        "market_salary": pd.read_parquet(d / "market_salary.parquet"),
    }


def engineer_features(snapshots: pd.DataFrame,
                      overtime_mean: float | None = None,
                      overtime_std: float | None = None) -> tuple[pd.DataFrame, float, float]:
    df = snapshots.copy()
    df["salary_log"] = np.log(df["salary"].clip(lower=1))
    if overtime_mean is None or overtime_std is None:
        overtime_mean = float(df["overtime_hours"].mean())
        overtime_std = float(df["overtime_hours"].std() or 1.0)
    df["workload_z"] = (df["overtime_hours"] - overtime_mean) / max(overtime_std, 1e-6)

    # Bounded / stationary versions of trending counters: raw tenure and
    # months-since-promotion grow without limit and break temporal transfer.
    df["log_tenure_months"] = np.log1p(df["tenure_months"].clip(lower=0))
    expected = df["level"].map(_LEVEL_EXPECTED_SERIES).astype(float)
    df["promo_pressure"] = ((df["months_since_promotion"] - expected)
                            / expected).clip(0, 2)
    prev = df.groupby(ID_COL)["promotion_count"].shift(12)
    df["promos_last_12m"] = (df["promotion_count"] - prev.fillna(df["promotion_count"]))\
        .clip(lower=0)

    # Peer-relative context: separates "underpaid vs company" from market-wide
    # drift so late-timeline cross-sections stay informative.
    df["comp_peer_gap"] = df["comp_ratio"] - df.groupby(TIME_COL)["comp_ratio"].transform(
        "mean")
    df["overtime_peer_gap"] = (
        df["overtime_hours"]
        - df.groupby(TIME_COL)["overtime_hours"].transform("mean"))
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("category")
    return df, float(overtime_mean), float(overtime_std)


def build_attrition_bundle(data_dir: str | Path) -> tuple[DatasetBundle, pd.DataFrame]:
    """Returns split bundle + the engineered full frame (for analytics/serving)."""
    frames = load_frames(data_dir)
    df, ot_mean, ot_std = engineer_features(frames["snapshots"])

    # Drop months whose 6m-forward labels are unobservable for survivors.
    cutoff = int(df[TIME_COL].max()) - HORIZON_M
    labelled = df[(df[TARGET].notna()) & (df[TIME_COL] <= cutoff)].copy()
    labelled[TARGET] = labelled[TARGET].astype(int)

    windows = derive_splits(int(df[TIME_COL].max()), HORIZON_M)

    def _slice(lo: int, hi: int) -> tuple[pd.DataFrame, pd.Series]:
        part = labelled[(labelled[TIME_COL] >= lo) & (labelled[TIME_COL] <= hi)]
        return part[FEATURE_COLUMNS], part[TARGET]

    X_tr, y_tr = _slice(*windows["train"])
    X_va, y_va = _slice(*windows["val"])
    X_te, y_te = _slice(*windows["test"])

    bundle = DatasetBundle(
        X_train=X_tr, y_train=y_tr, X_val=X_va, y_val=y_va, X_test=X_te, y_test=y_te,
        feature_columns=FEATURE_COLUMNS, categorical_features=CATEGORICAL_FEATURES,
        overtime_mean=ot_mean, overtime_std=ot_std,
    )
    return bundle, df


def build_promotion_table(df: pd.DataFrame,
                          horizon: int = PROMO_HORIZON) -> pd.DataFrame:
    """promoted_within_12m per snapshot; NaN where the horizon is unobservable.

    Requires df sorted by (employee_id, month_index) - true for generator output.
    """
    emp = df[ID_COL].to_numpy()
    months = df[TIME_COL].to_numpy(dtype=float)
    counts = df["promotion_count"].to_numpy(dtype=float)
    exit_m = df["exit_month"].to_numpy(dtype=float)
    n = len(df)

    new_group = np.r_[True, emp[1:] != emp[:-1]]
    # promo_event[i] == promotion occurred during transitions right after month i
    promo_event = np.r_[False, (counts[1:] > counts[:-1])] & ~new_group

    group_start = np.flatnonzero(new_group)
    group_of_row = np.cumsum(new_group) - 1
    group_last = np.r_[group_start[1:] - 1, n - 1]
    rows = np.arange(n)
    end_idx = np.minimum(rows + horizon, group_last[group_of_row])

    cum = np.cumsum(promo_event)
    window_events = cum[end_idx] - cum[rows]
    last_m = float(months.max())

    labels = (window_events > 0).astype(float)
    unobservable = ((~np.isnan(exit_m)) & (exit_m <= months + horizon)) | (
        np.isnan(exit_m) & (months + horizon > last_m))
    labels[unobservable] = np.nan

    out = df[[ID_COL, TIME_COL, "exit_month", "promotion_count"]].copy()
    out[PROMO_TARGET] = labels
    return out


def main(data_dir: str | Path = "artifacts/data") -> None:
    """CLI sanity pass: build bundle and print split summary."""
    bundle, _ = build_attrition_bundle(data_dir)
    print(f"train={len(bundle.X_train)} val={len(bundle.X_val)} test={len(bundle.X_test)}")
    print(f"features={bundle.feature_columns}")
    print(f"positive rate train={bundle.y_train.mean():.3f} "
          f"test={bundle.y_test.mean():.3f}")


if __name__ == "__main__":
    main()
