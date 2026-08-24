"""Shared fixtures: a tiny end-to-end stack generated once per session."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

import pandas as pd
import pytest

from core.config import Settings

if TYPE_CHECKING:
    from ml.inference.predictor import ModelRegistry


class TinyStack(TypedDict):
    root: Path
    data_dir: Path
    artifacts: Path
    settings: Settings
    registry: ModelRegistry
    snapshots: pd.DataFrame
    generator_result: dict[str, Any]


@pytest.fixture(scope="session")
def tiny_stack(tmp_path_factory: pytest.TempPathFactory) -> TinyStack:
    """Synthetic mini-dataset -> quality check -> trained artifacts -> registry."""
    root = tmp_path_factory.mktemp("stack")
    data_dir = root / "data"
    artifacts = root / "artifacts"

    from data.ingestion.synthetic_generator import GeneratorParams, SyntheticWorkforceGenerator
    from data.pipelines.build_dataset import load_frames
    from data.validation.engine import DataQualityEngine

    params = GeneratorParams(n_employees=450, n_months=26, seed=7,
                             target_annual_attrition=0.25,
                             treatment_start_month=10, treatment_share=0.35)
    result = SyntheticWorkforceGenerator(params).generate()
    data_dir.mkdir(parents=True, exist_ok=True)
    result["employees"].to_parquet(data_dir / "employees.parquet", index=False)
    result["snapshots"].to_parquet(data_dir / "snapshots.parquet", index=False)
    result["reviews"].to_parquet(data_dir / "reviews.parquet", index=False)
    result["market_salary"].to_parquet(data_dir / "market_salary.parquet", index=False)
    DataQualityEngine.save_report(result["ground_truth"],
                                  data_dir / "ground_truth.json")

    frames = load_frames(data_dir)
    report = DataQualityEngine().run(frames["snapshots"], frames["employees"],
                                     frames["reviews"])
    DataQualityEngine.save_report(report, data_dir / "quality_report.json")

    from ml.training.train import main as train_main

    train_main(str(data_dir), str(artifacts))

    settings = Settings(
        data_dir=data_dir,
        models_dir=artifacts / "models",
        metrics_dir=artifacts / "metrics",
        _env_file=None,  # type: ignore[call-arg]
    )
    from ml.inference.predictor import ModelRegistry

    registry = ModelRegistry(settings)
    return {
        "root": root,
        "data_dir": data_dir,
        "artifacts": artifacts,
        "settings": settings,
        "registry": registry,
        "snapshots": frames["snapshots"],
        "generator_result": result,
    }


@pytest.fixture()
def api_client(tiny_stack: TinyStack, monkeypatch: pytest.MonkeyPatch):
    settings: Settings = tiny_stack["settings"]
    monkeypatch.setenv("DATA_DIR", str(settings.data_dir))
    monkeypatch.setenv("MODELS_DIR", str(settings.models_dir))
    monkeypatch.setenv("METRICS_DIR", str(settings.metrics_dir))

    from core.config import get_settings

    get_settings.cache_clear()
    from apps.api.app import main as app_main

    app_main._state.clear()
    from fastapi.testclient import TestClient

    with TestClient(app_main.app) as client:
        yield client
    get_settings.cache_clear()
