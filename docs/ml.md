# Model card - attrition risk engine

## Data

Synthetic enterprise panel (labelled synthetic by construction):

- 8,000 base employees + ~13% annual hiring intake, 36 monthly snapshots (~260k rows)
- Static: age, gender, education, department, location, role, level
- Time-varying: salary vs market table, performance, engagement, manager sentiment,
  overtime/workload, promotion history, treatment flag (compensation pilot)
- Quarterly free-text reviews generated from the same latent drivers (NLP-ready labels)

**True hazard** (known ground truth):
`logit(h) = b0 + 2.4·comp_gap + 1.1·promo_pressure + 0.45·workload_z +
1.3·manager_negative + 0.8·low_engagement − 1.2·loyalty/2 + ε_i`,
with per-employee random effect σ=0.30. Realized annualized attrition ≈ 19%.

## Feature contract

14 numeric features + 4 categorical, engineered to be *stationary* so that
temporal transfer holds:

- bounded counters: `promo_pressure = clip((months_since_promotion − expected)/expected, 0, 2)`,
  `promos_last_12m`, `log_tenure_months`
- peer-relative context: `comp_peer_gap`, `overtime_peer_gap` (vs month cross-section)

Raw `salary_log` is deliberately excluded: it is nearly employee-unique and lets trees
memorize identity instead of learning the causal structure (measured -0.06 AUC when included).

## Validation protocol

Strictly temporal, rolling-origin:

| Window | Months (of 36) |
|--------|----------------|
| train  | 6 – 23         |
| val    | 24 – 26        |
| test   | 27 – 29        |

Snapshots beyond `last_month − 6` are dropped: their labels are unobservable for
survivors, which would bias composition toward leavers.

## Results (full run, seed 42)

| Model | val ROC-AUC | test ROC-AUC | test PR-AUC |
|-------|-------------|--------------|-------------|
| Logistic regression | 0.701 | 0.677 | 0.176 |
| Random forest       | 0.720 | 0.674 | 0.163 |
| **LightGBM (served)** | **0.721** | **0.670** | **0.159** |
| XGBoost             | 0.712 | 0.676 | 0.163 |

**Discrimination ceiling:** ranking employees by the *true simulated hazard* scores
ROC-AUC **0.689** on the same test window. All four models land within ~1 point of
this Bayes-proxy limit - the remaining error is irreducible Bernoulli noise, not
model capacity. This is measurable because the data generator is known.

**Calibration:** isotonic on validation. Expected Calibration Error on test:
0.038 → **0.018**. High-risk cutoff chosen as the smallest threshold with
validation precision ≥ 0.65.

**Auxiliary models**

- Salary anomaly detection: IsolationForest on salary/market residuals (contamination 0.02)
- Survival baseline: Cox PH, concordance **0.772**
- Promotion within 12m: LightGBM, test ROC-AUC ≈ 0.62

## Explainability semantics

SHAP TreeExplainer (tree_path_dependent; required by native categorical splits).

Local attributions are converted from log-odds to **risk points**: walking features
in descending |SHAP| order, each step's contribution is the change in sigmoid
probability it produces. Contributions sum to (final risk − population baseline),
so a manager sees exactly "engagement −8.2pp, comp ratio +5.9pp …".

## What-if semantics & limitations

Interventions mutate the feature row respecting relationships (raise → comp_ratio
and peer-gap shift against the market table; promotion → level+salary+pressure
reset), then the frozen model re-scores. This is **model-based counterfactual
reasoning**, not identified causal estimation. Two safeguards:

1. Directional correctness against the simulator's ground truth is asserted in tests
   (`test_salary_raise_reduces_underpaid_cohort_risk`, diff-in-diff treatment test).
2. A randomised compensation pilot exists in the data precisely so that future work
   can estimate uplift (e.g., T-learner/DML) instead of relying on model perturbation.
   Cohort ROI numbers should be read as directional estimates.

Confidence heuristic: share of numeric features still inside the training p1-p99
envelope after intervention.
