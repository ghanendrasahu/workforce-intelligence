"""Grounded analytics tools callable by the AI analyst.

Every tool returns compact JSON-serialisable dicts computed from the real
model/panel data. The LLM never invents numbers - it can only cite what these
functions return.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionToolParam

    from ml.inference.predictor import ModelRegistry

TOOLS: dict[str, dict[str, Any]] = {}


def tool(name: str, description: str, parameters: dict[str, Any]) -> Callable:
    def decorator(fn: Callable) -> Callable:
        TOOLS[name] = {
            "fn": fn,
            "schema": {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                },
            },
        }
        return fn
    return decorator


def register_all(registry: ModelRegistry) -> None:
    """(Re)bind tools against the loaded registry."""

    @tool(
        "query_attrition",
        "Annualized attrition rate and leaver counts for an optional department "
        "over the last 12 months of panel data.",
        {"type": "object", "properties": {
            "department": {"type": "string",
                           "enum": sorted(registry.snapshots["department"]
                                          .unique().tolist())},
        }},
    )
    def query_attrition(department: str | None = None) -> dict[str, Any]:
        snap = registry.snapshots
        last_m = int(snap["month_index"].max())
        win = snap[snap["month_index"] > last_m - 12]
        if department:
            win = win[win["department"] == department]
        events = int(win["attrition_event"].sum())
        latest = registry._latest
        scoped = latest[latest["department"] == department] if department else latest
        headcount = int(len(scoped))
        rate = events / max(headcount, 1) * 100
        company_rate = float(snap[snap["month_index"] > last_m - 12]["attrition_event"]
                             .sum()) / max(len(latest), 1) * 100
        return {
            "scope": department or "all departments",
            "annualized_attrition_pct": round(rate, 2),
            "company_wide_pct": round(company_rate, 2),
            "leavers_12m": events,
            "current_headcount": headcount,
        }

    @tool(
        "get_department_metrics",
        "Per-department table: headcount, attrition, compensation ratio, engagement, "
        "overtime and mean predicted risk.",
        {"type": "object", "properties": {}},
    )
    def get_department_metrics() -> dict[str, Any]:
        frame = registry.department_metrics()
        out = []
        for dept, row in frame.iterrows():
            out.append({
                "department": dept,
                "headcount": int(row["headcount"]),
                "attrition_events_12m": int(row["attrition_events_12m"]),
                "annualized_attrition_pct": float(row["annualized_attrition_pct"]),
                "avg_comp_ratio": float(row["avg_comp_ratio"]),
                "avg_engagement": float(row["avg_engagement"]),
                "avg_overtime_hours": float(row["avg_overtime"]),
                "avg_predicted_risk": float(row["avg_risk"]),
            })
        return {"departments": out}

    @tool(
        "get_feature_importance",
        "Global SHAP feature importance for the attrition model.",
        {"type": "object", "properties": {
            "top_k": {"type": "integer", "minimum": 3, "maximum": 15},
        }, "required": []},
    )
    def get_feature_importance(top_k: int = 8) -> dict[str, Any]:
        drivers = registry.global_drivers()[:top_k]
        return {"model": registry.meta["best_model_name"], "drivers": drivers}

    @tool(
        "run_cohort_analysis",
        "Risk profile of a cohort (department/location/level band): average risk, "
        "p90 risk, high-risk share and mean driver levels.",
        {"type": "object", "properties": {
            "department": {"type": "string"},
            "location": {"type": "string"},
            "level_min": {"type": "integer", "minimum": 1, "maximum": 5},
            "level_max": {"type": "integer", "minimum": 1, "maximum": 5},
        }, "required": []},
    )
    def run_cohort_analysis(department: str | None = None,
                            location: str | None = None,
                            level_min: int = 1, level_max: int = 5) -> dict[str, Any]:
        return registry.cohort_analysis(department, location, level_min, level_max)

    @tool(
        "run_what_if",
        "Simulate interventions (salary raise %, workload reduction %, promotion, "
        "training, manager coaching) for one employee or a whole department; returns "
        "predicted risk change and business cost/ROI.",
        {"type": "object", "properties": {
            "employee_id": {"type": "string"},
            "department": {"type": "string"},
            "salary_raise_pct": {"type": "number", "minimum": 0, "maximum": 50},
            "workload_reduction_pct": {"type": "number", "minimum": 0, "maximum": 100},
            "promotion": {"type": "boolean"},
            "engagement_boost_points": {"type": "number", "minimum": 0, "maximum": 30},
            "manager_sentiment_boost": {"type": "number", "minimum": 0, "maximum": 1},
        }, "required": []},
    )
    def run_what_if(employee_id: str | None = None, department: str | None = None,
                    salary_raise_pct: float = 0.0,
                    workload_reduction_pct: float = 0.0, promotion: bool = False,
                    engagement_boost_points: float = 0.0,
                    manager_sentiment_boost: float = 0.0) -> dict[str, Any]:
        from ml.explainability.whatif import InterventionParams

        params = InterventionParams(
            salary_raise_pct=salary_raise_pct,
            workload_reduction_pct=workload_reduction_pct,
            promotion=promotion,
            engagement_boost_points=engagement_boost_points,
            manager_sentiment_boost=manager_sentiment_boost,
        )
        if employee_id:
            row = registry.latest_row(employee_id)
            if row is None:
                return {"error": f"unknown employee_id {employee_id}"}
            res = registry.whatif.simulate(row, params,
                                           lambda rows: registry.raw_scores(rows))
            return res.__dict__
        sub = registry._latest
        if department:
            sub = sub[sub["department"] == department]
        scenario = registry.whatif.cohort_scenario(
            sub.head(5000), params, lambda rows: registry.raw_scores(rows))
        return scenario

    # keep references alive
    globals().update({"_bound": [query_attrition, get_department_metrics,
                                 get_feature_importance, run_cohort_analysis,
                                 run_what_if]})


def schemas() -> list[ChatCompletionToolParam]:
    return [t["schema"] for t in TOOLS.values()]


def execute(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if name not in TOOLS:
        return {"error": f"unknown tool {name}"}
    try:
        result = TOOLS[name]["fn"](**arguments)
    except Exception as exc:  # noqa: BLE001 - surfaced to the LLM as tool error
        return {"error": f"{type(exc).__name__}: {exc}"}
    return result  # type: ignore[no-any-return]
