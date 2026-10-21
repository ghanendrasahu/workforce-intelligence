"""FastAPI application exposing the workforce intelligence engine."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from ai.agent.analyst import ask_analyst
from apps.api.app.schemas import (
    AgentAskRequest,
    AgentAskResponse,
    BatchRiskRequest,
    DriverContribution,
    EmployeeRisk,
    HealthResponse,
    ScenarioRequest,
    WhatIfRequest,
)
from core.config import get_settings
from core.logging import configure_logging
from ml.inference.predictor import ModelRegistry, get_registry

_state: dict[str, object] = {}

logger = logging.getLogger("apps.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(json_logs=settings.log_json, level=settings.log_level)
    logger.info(
        "starting %s v%s (log_json=%s)",
        settings.app_name, settings.version, settings.log_json,
    )
    try:
        registry = get_registry()
        _state["registry"] = registry
        _state["model"] = registry.meta.get("best_model_name")
        _state["fingerprint"] = registry.meta.get("data_fingerprint")
        logger.info(
            "model registry loaded: model=%s data_fingerprint=%s",
            _state["model"], _state["fingerprint"],
        )
    except FileNotFoundError as exc:
        _state["registry"] = None
        _state["error"] = (
            f"Artifacts not found under {settings.data_dir} / {settings.models_dir}. "
            f"Run `python -m data.ingestion.synthetic_generator` and "
            f"`python -m ml.training.train` first. ({exc})"
        )
        logger.error("startup degraded: %s", _state["error"])
    yield
    logger.info("shutdown")


app = FastAPI(
    title=get_settings().app_name,
    version=get_settings().version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def reg() -> ModelRegistry:
    if _state.get("registry") is None:
        raise HTTPException(status_code=503, detail=str(_state.get("error")))
    return _state["registry"]  # type: ignore[return-value]


# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if _state.get("registry") is not None else "degraded",
        app=get_settings().app_name,
        version=get_settings().version,
        model_name=_state.get("model"),  # type: ignore[arg-type]
        data_fingerprint=_state.get("fingerprint"),  # type: ignore[arg-type]
    )


@app.get("/api/v1/employees/search")
def search_employees(q: str = "", limit: int = 20) -> list[dict]:
    frame = reg().search_employees(q, limit)
    return frame.to_dict(orient="records")


def _risk_payload(r: ModelRegistry, row: pd.Series) -> dict:
    raw = float(r.raw_scores(pd.DataFrame([row]))[0])
    cal = float(np.clip(r.calibrated_scores(pd.DataFrame([row]))[0], 0, 1))
    local = r.local_drivers(row, top_k=6)
    drivers = [DriverContribution(**c) for c in local["contributions"]]  # type: ignore[arg-type]
    emp = EmployeeRisk(
        employee_id=str(row["employee_id"]),
        department=str(row["department"]),
        location=str(row["location"]),
        level=int(row["level"]),
        salary=float(row["salary"]),
        risk=round(cal, 4),
        risk_raw=round(raw, 4),
        band=r.band(cal),
        top_drivers=drivers,
        confidence=r.whatif.confidence(row),
    )
    return emp.model_dump()


@app.get("/api/v1/employees/{employee_id}/risk", response_model=EmployeeRisk)
def employee_risk(employee_id: str) -> dict:
    r = reg()
    row = r.latest_row(employee_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Unknown employee {employee_id}")
    return _risk_payload(r, row)


@app.post("/api/v1/risk/batch", response_model=list[EmployeeRisk])
def batch_risk(payload: BatchRiskRequest) -> list[dict]:
    r = reg()
    out = []
    for eid in payload.employee_ids:
        row = r.latest_row(eid)
        if row is None:
            continue
        out.append(_risk_payload(r, row))
    return out


@app.get("/api/v1/insights/global-drivers")
def global_drivers() -> list[dict]:
    return reg().global_drivers()


@app.get("/api/v1/insights/partial-dependence")
def partial_dependence_data() -> dict:
    path = Path(get_settings().metrics_dir) / "pdp.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="PDP artifacts missing")
    import json

    return json.loads(path.read_text())  # type: ignore[no-any-return]


@app.get("/api/v1/analytics/trend")
def attrition_trend() -> list[dict]:
    frame = reg().attrition_trend()
    frame["date"] = frame["date"].astype(str)
    return frame.to_dict(orient="records")


@app.get("/api/v1/analytics/departments")
def department_metrics() -> list[dict]:
    return reg().department_metrics().reset_index().to_dict(orient="records")


@app.get("/api/v1/analytics/high-risk")
def high_risk(top_k: int = 25) -> list[dict]:
    return reg().high_risk_employees(top_k).to_dict(orient="records")


@app.get("/api/v1/analytics/cohort")
def cohort(department: str | None = None, location: str | None = None,
           level_min: int = 1, level_max: int = 5) -> dict:
    return reg().cohort_analysis(department, location, level_min, level_max)


@app.get("/api/v1/employees/{employee_id}/reviews")
def employee_reviews(employee_id: str, limit: int = 5) -> list[dict]:
    rev = reg().reviews_sample(employee_id, limit)
    rev["date"] = rev["date"].astype(str)
    return rev.to_dict(orient="records")


# ---------------------------------------------------------------------------
@app.post("/api/v1/whatif/simulate")
def whatif_simulate(payload: WhatIfRequest) -> dict:
    r = reg()
    row = r.latest_row(payload.employee_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Unknown employee {payload.employee_id}")
    result = r.whatif.simulate(row, payload.interventions,
                               lambda rows: r.raw_scores(rows))
    return {
        "employee_id": payload.employee_id,
        **result.__dict__,
        "band_before": r.band(result.risk_before),
        "band_after": r.band(result.risk_after),
    }


@app.post("/api/v1/whatif/scenario")
def whatif_scenario(payload: ScenarioRequest) -> dict:
    r = reg()
    sub = r._latest[
        r._latest["department"].eq(payload.department or r._latest["department"])
        & r._latest["location"].eq(payload.location or r._latest["location"])
        & r._latest["level"].between(payload.level_min, payload.level_max)
    ]
    if payload.min_risk is not None:
        risks = pd.Series(r.calibrated_scores(sub), index=sub.index)
        sub = sub[risks >= payload.min_risk]
    sub = sub.head(payload.limit)
    scenario = r.whatif.cohort_scenario(sub, payload.interventions,
                                        lambda rows: r.raw_scores(rows))
    scenario["filter"] = {
        "department": payload.department, "location": payload.location,
        "level_min": payload.level_min, "level_max": payload.level_max,
        "min_risk": payload.min_risk,
    }
    return scenario


# ---------------------------------------------------------------------------
@app.post("/api/v1/agent/ask", response_model=AgentAskResponse)
def agent_ask(payload: AgentAskRequest) -> AgentAskResponse:
    answer, tools_used = ask_analyst(payload.question, reg())
    return AgentAskResponse(answer=answer, tools_used=tools_used,
                            configured=True)
