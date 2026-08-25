"""Agent tools execute against the registry without any LLM involvement."""
from __future__ import annotations

from conftest import TinyStack

from ai.tools import analytics_tools


def test_tools_registered_and_executable(tiny_stack: TinyStack) -> None:
    registry = tiny_stack["registry"]
    analytics_tools.register_all(registry)

    schemas = analytics_tools.schemas()
    names = {s["function"]["name"] for s in schemas}
    expected = {"query_attrition", "get_department_metrics",
                "get_feature_importance", "run_cohort_analysis", "run_what_if"}
    assert expected <= names

    out = analytics_tools.execute("query_attrition", {})
    assert out["current_headcount"] > 0
    assert 0 < out["company_wide_pct"] < 100

    out = analytics_tools.execute("get_feature_importance", {"top_k": 5})
    assert len(out["drivers"]) == 5

    out = analytics_tools.execute("run_cohort_analysis", {"level_min": 1,
                                                          "level_max": 2})
    assert out["n_employees"] >= 0

    out = analytics_tools.execute("run_what_if",
                                  {"salary_raise_pct": 10, "department": None})
    assert "avg_risk_before" in out or "risk_before" in out or "error" not in out

    bad = analytics_tools.execute("nonexistent_tool", {})
    assert "error" in bad


def test_tool_output_json_serializable(tiny_stack: TinyStack) -> None:
    import json

    registry = tiny_stack["registry"]
    analytics_tools.register_all(registry)
    for name in ("query_attrition", "get_department_metrics"):
        result = analytics_tools.execute(name, {})
        json.dumps(result)  # must not raise
