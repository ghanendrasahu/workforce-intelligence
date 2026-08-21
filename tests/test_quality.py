"""Data-quality engine detection tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
from conftest import TinyStack

from data.validation.engine import DataQualityEngine, population_stability_index


def test_clean_data_passes(tiny_stack: TinyStack) -> None:
    engine = DataQualityEngine()
    # rebuild frames from the fixture's data dir
    data_dir = tiny_stack["data_dir"]
    snapshots = pd.read_parquet(f"{data_dir}/snapshots.parquet")
    employees = pd.read_parquet(f"{data_dir}/employees.parquet")
    reviews = pd.read_parquet(f"{data_dir}/reviews.parquet")
    rep = engine.run(snapshots, employees, reviews)
    assert rep["status"] == "pass"
    assert rep["n_checks_triggered"] == 0


def test_injected_anomalies_detected(tiny_stack: TinyStack) -> None:
    data_dir = tiny_stack["data_dir"]
    snapshots = pd.read_parquet(f"{data_dir}/snapshots.parquet").copy()
    employees = pd.read_parquet(f"{data_dir}/employees.parquet")
    reviews = pd.read_parquet(f"{data_dir}/reviews.parquet")

    snapshots.loc[snapshots.index[:5], "salary"] = -100
    snapshots.loc[snapshots.index[5:9], "performance_rating"] = 9.5
    dup = snapshots.iloc[[0]].copy()

    engine = DataQualityEngine()
    rep = engine.run(pd.concat([snapshots, dup]), employees, reviews)
    checks = {i["check"] for i in rep["issues"]}
    assert "validity" in checks
    assert "uniqueness" in checks
    assert rep["status"] == "fail"


def test_psi_detects_shift() -> None:
    rng = np.random.default_rng(0)
    expected = pd.Series(rng.normal(0, 1, 5000))
    same = pd.Series(rng.normal(0, 1, 5000))
    shifted = pd.Series(rng.normal(3, 1, 5000))
    assert population_stability_index(expected, same) < 0.1
    assert population_stability_index(expected, shifted) > 0.25
