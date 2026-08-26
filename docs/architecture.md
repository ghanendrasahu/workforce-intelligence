# Architecture

## System overview

```
                    ┌─────────────────────────────┐
                    │  Synthetic workforce engine │
                    │  (causal ground truth)      │
                    └──────────────┬──────────────┘
                                   ↓ parquet
                    ┌─────────────────────────────┐
                    │   Data quality engine       │
                    │  completeness · validity ·  │
                    │  integrity · PSI drift      │
                    └──────────────┬──────────────┘
                                   ↓
              ┌────────────────────┴───────────────────┐
              ↓                                        ↓
   ┌─────────────────────┐                 ┌─────────────────────┐
   │   Training pipeline │                 │   Serving registry  │
   │  LR/RF/LGBM/XGB     │── artifacts ──▶│  ModelRegistry      │
   │  isotonic calibrator│                 │  risk·drivers·whatif│
   │  SHAP · Cox · promo │                 └──────────┬──────────┘
   └─────────────────────┘                            ↓
                                          ┌─────────────────────┐
                                          │   FastAPI           │
                                          │ /risk /whatif       │
                                          │ /analytics /agent   │
                                          └──────────┬──────────┘
                              ┌──────────────────────┼──────────────────┐
                              ↓                      ↓                  ↓
                   ┌──────────────────┐   ┌──────────────────┐  ┌─────────────┐
                   │ Streamlit        │   │ Azure OpenAI     │  │ PostgreSQL  │
                   │ executive UI     │   │ tool-calling     │  │ pgvector    │
                   └──────────────────┘   │ analyst          │  │ (roadmap)   │
                                          └──────────────────┘  └─────────────┘
```

## Components

| Path | Responsibility |
|------|----------------|
| `data/ingestion/synthetic_generator.py` | Monthly-panel enterprise simulator with a known structural hazard, randomised compensation pilot, hiring intake and generated review text |
| `data/validation/engine.py` | Dependency-free data-quality checks + PSI drift report (`quality_report.json`) |
| `data/pipelines/build_dataset.py` | Feature contract (single source of truth), rolling-origin temporal splits, promotion labels |
| `ml/training/` | Multi-model training, calibration, threshold policy, auxiliary models, MLflow hook |
| `ml/explainability/explainer.py` | SHAP global/local attribution, SHAP→"risk points" conversion, PDP curves |
| `ml/explainability/whatif.py` | Intervention application respecting feature relationships + cohort business case (cost/avoided/ROI) |
| `ml/inference/predictor.py` | `ModelRegistry`: loads artifacts once, serves scoring/explanations/analytics |
| `apps/api` | FastAPI service (CORS, versioned routes, pydantic schemas) |
| `apps/dashboard` | Streamlit executive UI talking to the API only |
| `ai/tools` + `ai/agent` | Grounded analytics tools + Azure OpenAI tool-calling loop |

## Key design decisions

1. **API-only dashboard** - the Streamlit app holds no model code; every number comes from the FastAPI service, matching how a production front end would integrate.
2. **Feature contract in one module** - training and serving import the same `FEATURE_COLUMNS`; the registry conforms incoming frames to training dtypes before scoring.
3. **Rolling-origin temporal validation** - splits are derived from timeline length and capped at fully-observable label months to avoid survivor-composition bias.
4. **Grounded agent** - the LLM orchestrates tools; it never fabricates metrics. Unconfigured credentials yield HTTP 503 rather than an ungrounded answer.

## Production hardening path (roadmap)

- PostgreSQL + pgvector for snapshot storage and review-embedding retrieval (RAG over free-text corpora)
- Redis response caching for hot analytics queries
- MLflow registry with staged promotion + GitHub Actions retrain on data drift (PSI > 0.25 triggers)
- Prometheus/Grafana exporters for API p95 latency and PSI gauges
- AWS ECS/Fargate deployment behind an ALB; Terraform modules under `infrastructure/`
