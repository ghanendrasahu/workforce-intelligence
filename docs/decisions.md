# Decision log (ADRs)

## ADR-001 - Synthetic-first dataset with causal ground truth
**Context.** The classic IBM HR dataset is a single static snapshot with no causal
structure, no salary history and no text - impossible to validate what-if reasoning.
**Decision.** Generate an enterprise panel from a known structural hazard with a
randomised compensation pilot. Clearly labelled synthetic everywhere.
**Consequence.** Every downstream claim (SHAP recovery, intervention direction,
discrimination ceiling) is testable. Trade-off: no real-world domain noise.

## ADR-002 - Rolling-origin temporal splits derived from timeline length
**Decision.** Test = last 3 fully-observable months; validation = 3 before that;
training = everything earlier. Windows are computed from the data, not hardcoded.
**Consequence.** No leakage from the future, no survivor-composition bias in late
months, and the pipeline runs at any dataset scale (tests use 26 months).

## ADR-003 - Stationary/bounded features over raw counters
**Context.** Raw tenure/months-since-promotion trend without bound; trees trained on
early windows fail to transfer (measured ROC-AUC drop 0.72 → 0.62 on later months).
**Decision.** Replace raw counters with bounded pressure ratios and peer-relative gaps;
exclude near-unique identifiers like `salary_log` from the feature set.
**Consequence.** Val≈test generalization; slight loss of raw-information richness.

## ADR-004 - Served model must be TreeExplainer-native
**Decision.** Winner selection among gradient-boosting models by validation PR-AUC;
other models remain as reported benchmarks only.
**Consequence.** Per-request SHAP attributions stay millisecond-fast; no pipeline
wrappers in the serving path.

## ADR-005 - Model-based counterfactuals now, identified causal effects later
**Decision.** The what-if engine perturbs features respecting relationships and
re-scores with the frozen model; a randomised compensation pilot is embedded in the
generator so uplift modelling (T-learner/DML) can replace it.
**Consequence.** Working simulator today; honest documentation that cohort ROI is a
directional estimate, plus a built-in path to proper causal inference.

## ADR-006 - Grounded LLM agent over tool-calling, not RAG-only
**Decision.** The analyst answers exclusively via analytics tools whose outputs are
JSON computed from real artifacts; pgvector-based document retrieval is deferred to
the production roadmap.
**Consequence.** Zero fabricated numbers today; free-text corpus retrieval slots in
later without changing the agent loop.

## ADR-007 - Dependency-light data quality engine
**Decision.** Hand-rolled validators + PSI instead of Great Expectations for the slice.
**Consequence.** Runs in CI and inside the API with zero extra services; swap-in later
if rule management complexity grows.
