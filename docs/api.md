# API reference

Base URL (local): `http://127.0.0.1:8000` · interactive docs at `/docs`.

## Health

| Method | Path | Notes |
|--------|------|-------|
| GET | `/health` | status, model name, data fingerprint |

## Risk

| Method | Path | Notes |
|--------|------|-------|
| GET | `/api/v1/employees/search?q=&limit=` | id/department/location search |
| GET | `/api/v1/employees/{id}/risk` | calibrated risk, band, top SHAP drivers, confidence |
| POST | `/api/v1/risk/batch` | `{"employee_ids": [...]}` → list of risk payloads |
| GET | `/api/v1/employees/{id}/reviews` | generated review text + sentiment labels |

## What-if

| Method | Path | Body / params |
|--------|------|---------------|
| POST | `/api/v1/whatif/simulate` | `{"employee_id", "interventions": {salary_raise_pct, workload_reduction_pct, promotion, engagement_boost_points, manager_sentiment_boost}}` |
| POST | `/api/v1/whatif/scenario` | cohort filter (`department`, `min_risk`, `limit`) + interventions → avg risk before/after, cost avoided, ROI |

## Analytics

| Method | Path | Returns |
|--------|------|---------|
| GET | `/api/v1/analytics/trend` | monthly annualized attrition % + active headcount |
| GET | `/api/v1/analytics/departments` | per-dept scorecard |
| GET | `/api/v1/analytics/high-risk?top_k=` | highest-risk employees |
| GET | `/api/v1/analytics/cohort?department=&location=&level_min=&level_max=` | cohort risk profile + driver means |
| GET | `/api/v1/insights/global-drivers` | global SHAP importance |
| GET | `/api/v1/insights/partial-dependence` | marginal-effect curves |

## AI analyst

| Method | Path | Notes |
|--------|------|-------|
| POST | `/api/v1/agent/ask` | `{"question"}` → grounded answer + tools used; 503 if Azure OpenAI unconfigured |

Tools available to the agent: `query_attrition`, `get_department_metrics`,
`get_feature_importance`, `run_cohort_analysis`, `run_what_if`.

## Examples

```bash
curl http://127.0.0.1:8000/api/v1/employees/EMP00042/risk

curl -X POST http://127.0.0.1:8000/api/v1/whatif/scenario \
  -H "Content-Type: application/json" \
  -d '{"department": "Engineering", "min_risk": 0.3, "limit": 1000,
       "interventions": {"salary_raise_pct": 10, "workload_reduction_pct": 20}}'
```
