"""API contract tests via TestClient (artifacts loaded through lifespan)."""
from __future__ import annotations


def test_health_ok(api_client) -> None:
    resp = api_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_name"] in ("lightgbm", "xgboost")


def test_employee_risk_endpoint(api_client) -> None:
    search = api_client.get("/api/v1/employees/search", params={"q": "", "limit": 1})
    eid = search.json()[0]["employee_id"]
    resp = api_client.get(f"/api/v1/employees/{eid}/risk")
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["risk"] <= 1.0
    assert body["band"] in ("low", "medium", "high")
    assert len(body["top_drivers"]) >= 3
    assert all("risk_points" in d for d in body["top_drivers"])


def test_unknown_employee_404(api_client) -> None:
    assert api_client.get("/api/v1/employees/NOPE999/risk").status_code == 404


def test_analytics_endpoints(api_client) -> None:
    trend = api_client.get("/api/v1/analytics/trend")
    assert trend.status_code == 200
    rows = trend.json()
    assert {"month_index", "annualized_rate", "n_active"} <= set(rows[0])

    depts = api_client.get("/api/v1/analytics/departments")
    assert depts.status_code == 200
    assert len(depts.json()) >= 3

    drivers = api_client.get("/api/v1/insights/global-drivers")
    assert drivers.status_code == 200
    assert len(drivers.json()) >= 5

    cohort = api_client.get("/api/v1/analytics/cohort",
                            params={"department": depts.json()[0]["department"]})
    assert cohort.status_code == 200
    assert cohort.json()["n_employees"] > 0


def test_batch_risk(api_client) -> None:
    ids = [r["employee_id"] for r in api_client.get(
        "/api/v1/employees/search", params={"limit": 4}).json()]
    resp = api_client.post("/api/v1/risk/batch", json={"employee_ids": ids})
    assert resp.status_code == 200
    assert len(resp.json()) == len(ids)


def test_whatif_simulate(api_client) -> None:
    eid = api_client.get("/api/v1/employees/search",
                         params={"limit": 1}).json()[0]["employee_id"]
    payload = {"employee_id": eid,
               "interventions": {"salary_raise_pct": 10, "promotion": False}}
    resp = api_client.post("/api/v1/whatif/simulate", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert 0 <= body["risk_before"] <= 1
    assert 0 <= body["risk_after"] <= 1
    assert "annual_cost" in body


def test_whatif_scenario(api_client) -> None:
    payload = {
        "department": None,
        "min_risk": 0.2,
        "limit": 300,
        "interventions": {"salary_raise_pct": 10},
    }
    resp = api_client.post("/api/v1/whatif/scenario", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_employees"] > 0
    assert "roi_multiple" in body and "annual_cost_avoided" in body


def test_agent_not_configured_returns_503(api_client, monkeypatch) -> None:
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    from core.config import get_settings

    get_settings.cache_clear()
    try:
        resp = api_client.post("/api/v1/agent/ask",
                               json={"question": "why is attrition up?"})
        assert resp.status_code == 503
    finally:
        get_settings.cache_clear()


def test_reviews_endpoint(api_client) -> None:
    eid = api_client.get("/api/v1/employees/search",
                         params={"limit": 5}).json()[0]["employee_id"]
    resp = api_client.get(f"/api/v1/employees/{eid}/reviews")
    assert resp.status_code == 200
