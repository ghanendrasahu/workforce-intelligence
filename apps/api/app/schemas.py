"""Pydantic request/response schemas for the API."""
from __future__ import annotations

from pydantic import BaseModel, Field

from ml.explainability.whatif import InterventionParams


class HealthResponse(BaseModel):
    status: str
    app: str
    version: str
    model_name: str | None = None
    data_fingerprint: str | None = None


class DriverContribution(BaseModel):
    feature: str
    value: object
    shap_log_odds: float = 0.0
    risk_points: float = 0.0
    direction: str = "increases"


class EmployeeRisk(BaseModel):
    employee_id: str
    department: str
    location: str
    level: int
    salary: float
    risk: float
    risk_raw: float
    band: str
    top_drivers: list[DriverContribution] = []
    confidence: float = Field(1.0, description="feature-envelope confidence 0-1")


class BatchRiskRequest(BaseModel):
    employee_ids: list[str] = Field(min_length=1, max_length=500)


class WhatIfRequest(BaseModel):
    employee_id: str
    interventions: InterventionParams


class ScenarioRequest(BaseModel):
    """Cohort-level what-if scenario with business costing."""

    department: str | None = None
    location: str | None = None
    level_min: int = 1
    level_max: int = 5
    min_risk: float | None = Field(None, ge=0.0, le=1.0,
                                   description="only employees with calibrated risk >= this")
    limit: int = Field(2000, ge=1, le=20_000)
    interventions: InterventionParams


class AgentAskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


class AgentAskResponse(BaseModel):
    answer: str
    tools_used: list[str] = []
    configured: bool = True
