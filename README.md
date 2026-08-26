# topG - Workforce Intelligence & Decision Engine

An end-to-end AI decision platform that predicts employee attrition, explains *why*,
simulates *what happens if we act*, and prices the intervention - not just a
probability score.

> Enterprise AI decision intelligence; the workforce domain is the first implementation.
> The same architecture applies to customer churn, credit risk, fraud or supply-chain risk.
>
> **Data is synthetic by design** - generated from a known causal structure so every
> claim (driver attribution, intervention direction, discrimination ceiling) is testable.

## The difference

Most portfolio projects stop at `employee data → XGBoost → 82% attrition probability`.
This system answers what leadership actually asks:

```
Risk: 32%   (calibrated)
Primary drivers (risk points):
  engagement score      +8.1pp
  comp ratio            +5.9pp     <- below market benchmark
  promo pressure        +4.2pp
Intervention: 10% raise + 20% workload cut
  risk: 39.8% -> 37.4%  across high-risk cohort
  cost avoided: Rs 18.4L/yr    ROI: 0.25x (targeted cohorts > 1x)
Confidence: 92%
```

Plus an AI analyst that answers questions like *"why is attrition rising in
Engineering?"* by calling real analytics tools - never inventing numbers.

## Measured results (full run, seed 42)

| Metric | Value |
|---|---|
| Served model | LightGBM (isotonic-calibrated) |
| Test ROC-AUC / PR-AUC | 0.670 / 0.159 (val 0.721 / 0.240) |
| Discrimination ceiling (true hazard ranking) | **0.689** - models reach ~97% of the Bayes proxy limit |
| Calibration ECE (raw → calibrated) | 0.038 → **0.018** |
| Cox PH survival baseline (concordance) | 0.772 |
| Promotion model (12m, test ROC-AUC) | ≈ 0.62 |
| Salary anomaly detector | IsolationForest, contamination 0.02 |
| Data quality engine | pass (completeness/validity/integrity/drift PSI) |

The ceiling number matters: because the generator's data-generating process is known,
we can measure the theoretical best any observable-feature model could do on this data.
Remaining error is irreducible outcome noise, documented in `docs/ml.md`.

## Architecture

See [docs/architecture.md](docs/architecture.md). Monorepo layout:

```
data/          synthetic generator · quality engine · feature pipeline
ml/            training · inference registry · SHAP explainability · what-if engine
ai/            grounded analytics tools + Azure OpenAI tool-calling agent
apps/api       FastAPI service
apps/dashboard Streamlit executive UI (talks to the API only)
infrastructure/docker  compose: api · dashboard · postgres · redis · mlflow
tests/         26 tests: determinism, causal direction, contracts, API, tools
```

## Quickstart

```bash
python -m venv .venv
.\.venv\Scripts\activate                 # Windows (source .venv/bin/activate on unix)
pip install -e ".[dev]"
copy .env.example .env                   # fill Azure OpenAI vars for the AI analyst

# 1. generate synthetic enterprise data (~260k snapshots, deterministic)
python -m data.ingestion.synthetic_generator

# 2. run the data-quality gate
python -m data.validation.engine

# 3. train + calibrate + explain all models
python -m ml.training.train

# 4a. serve the API
uvicorn apps.api.app.main:app --reload

# 4b. in another terminal, launch the dashboard
streamlit run apps/dashboard/app.py
```

Or with Docker (mounts host `artifacts/`):

```bash
docker compose -f infrastructure/docker/docker-compose.yml up --build
```

## What you can do with it

- **Overview page** - attrition KPIs with trend sparklines, department scorecard,
  global SHAP drivers, highest-risk employees
- **Employee risk** - search anyone, see calibrated risk + per-driver risk points +
  their review-signal history
- **What-if simulator** - salary/workload/promotion/training/coaching sliders on one
  employee or a filtered cohort, with annual cost, cost-avoided and ROI
- **AI analyst** - chat interface over `query_attrition`, `get_department_metrics`,
  `get_feature_importance`, `run_cohort_analysis`, `run_what_if` (Azure OpenAI;
  returns HTTP 503 if unconfigured - it will not answer ungrounded)

## Engineering quality

- `ruff check .` clean · `mypy core data ml ai apps` clean
- `pytest` - 26 tests covering generator determinism, diff-in-diff treatment effect,
  horizon-label integrity, quality-gate detection, training smoke, driver-sum
  consistency, what-if causal direction, API contracts and agent tools
- GitHub Actions CI runs lint + typecheck + tests on every push/PR
- No secrets in code: `.env.example` documents every variable

## Documentation

- [Architecture](docs/architecture.md) · [Model card](docs/ml.md) ·
  [API reference](docs/api.md) · [Decision log](docs/decisions.md)

## Roadmap

PostgreSQL/pgvector persistence · review-text NLP pipeline (sentiment/topic models
with generator-provided ground-truth labels) · uplift modelling on the embedded
compensation pilot (T-learner/DML) · MLflow model registry + drift-triggered retraining ·
Prometheus/Grafana observability · AWS Terraform deployment.
