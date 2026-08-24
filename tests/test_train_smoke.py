"""Training smoke test: artifacts produced, metrics sane, calibration works."""
from __future__ import annotations

import json
from pathlib import Path

from conftest import TinyStack


def test_training_produces_artifacts(tiny_stack: TinyStack) -> None:
    artifacts: Path = tiny_stack["artifacts"]
    meta = json.loads((artifacts / "meta.json").read_text())
    metrics = json.loads((artifacts / "metrics" / "metrics.json").read_text())

    for f in ("attrition_model.joblib", "calibrator.joblib",
              "salary_anomaly.joblib"):
        assert (artifacts / "models" / f).exists(), f

    assert meta["best_model_name"] in ("lightgbm", "xgboost")
    assert len(meta["feature_columns"]) > 8
    assert "high_cutoff" in meta["thresholds"]

    models = metrics["attrition"]["metrics"]
    best = meta["best_model_name"]
    test_auc = models[best]["test"]["roc_auc"]
    # tiny dataset => loose but meaningful floor
    assert test_auc > 0.55, f"AUC too low: {test_auc}"
    for name, m in models.items():
        assert 0.5 <= m["val"]["roc_auc"] <= 1.0, name
    ceiling = metrics["bayes_ceiling_proxy"]
    assert 0.0 <= ceiling["test_oracle_hazard_auc"] <= 1.0
    calib = metrics["attrition"]["calibration"]
    assert calib["ece_after"] <= calib["ece_before"] + 0.05


def test_global_importance_saved(tiny_stack: TinyStack) -> None:
    artifacts: Path = tiny_stack["artifacts"]
    drivers = json.loads((artifacts / "metrics" / "global_shap.json").read_text())
    assert len(drivers) >= 5
    assert all("feature" in d and "mean_abs_shap" in d for d in drivers)
