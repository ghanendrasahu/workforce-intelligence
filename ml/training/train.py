"""Training pipeline: multi-model comparison, calibration, explainability.

Trains attrition models (LogReg / RandomForest / LightGBM / XGBoost) on a
strictly temporal split, selects the winner by validation PR-AUC, fits an
isotonic calibrator, extracts SHAP explanations, and trains auxiliary models
(salary anomalies via IsolationForest, survival baseline via Cox PH,
promotion probability via LightGBM).

Artifacts land under artifacts/{models,metrics} plus meta.json describing the
feature contract consumed by the API.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier

from data.pipelines.build_dataset import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    PROMO_HORIZON,
    PROMO_TARGET,
    build_attrition_bundle,
    build_promotion_table,
    derive_splits,
)
from ml.explainability.explainer import AttritionExplainer, save_json
from ml.explainability.whatif import NUMERIC_FEATURES_FOR_BOUNDS
from ml.training.evaluate import (
    best_f1_threshold,
    calibration_curve_data,
    classification_metrics,
    expected_calibration_error,
    fit_calibrator,
)

SEED = 42


def _make_models() -> dict[str, Any]:
    pre = ColumnTransformer(
        [("num", StandardScaler(), NUMERIC_FEATURES),
         ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES)],
        verbose_feature_names_out=False,
    )
    pre_ord = ColumnTransformer(
        [("num", "passthrough", NUMERIC_FEATURES),
         ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
          CATEGORICAL_FEATURES)],
        verbose_feature_names_out=False,
    )
    return {
        "logistic_regression": Pipeline([("pre", pre),
                                         ("clf", LogisticRegression(
                                             max_iter=2000, class_weight="balanced",
                                             random_state=SEED))]),
        "random_forest": Pipeline([("pre", pre_ord),
                                   ("clf", RandomForestClassifier(
                                       n_estimators=400, min_samples_leaf=60,
                                       max_features=0.6,
                                       n_jobs=-1, random_state=SEED))]),
        "lightgbm": LGBMClassifier(
            n_estimators=600, learning_rate=0.05, num_leaves=31,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
            min_child_samples=150, reg_lambda=5.0, random_state=SEED,
            verbosity=-1),
        "xgboost": XGBClassifier(
            n_estimators=600, learning_rate=0.05, max_depth=5, subsample=0.8,
            colsample_bytree=0.7, min_child_weight=50, reg_lambda=5.0,
            tree_method="hist", enable_categorical=True,
            random_state=SEED, eval_metric="logloss",
            early_stopping_rounds=50),
    }


def _fit_model(name: str, model: Any, X_tr: pd.DataFrame, y_tr: np.ndarray,
               X_va: pd.DataFrame, y_va: np.ndarray) -> None:
    """Fit with early stopping for gradient-boosting models."""
    if name == "lightgbm":
        from lightgbm import early_stopping

        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                  callbacks=[early_stopping(50, verbose=False)])
    elif name == "xgboost":
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    else:
        model.fit(X_tr, y_tr)


def _prob(model: object, X: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(X)  # type: ignore[attr-defined]
    return np.asarray(proba)[:, 1]


def _data_fingerprint(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def train_attrition(bundle, out_models: Path, out_metrics: Path) -> dict:
    X_tr, y_tr = bundle.X_train, bundle.y_train.to_numpy()
    X_va, y_va = bundle.X_val, bundle.y_val.to_numpy()
    X_te, y_te = bundle.X_test, bundle.y_test.to_numpy()

    metrics: dict[str, dict] = {}
    fitted: dict[str, Any] = {}
    val_scores: dict[str, np.ndarray] = {}
    test_scores: dict[str, np.ndarray] = {}

    for name, model in _make_models().items():
        t0 = time.time()
        _fit_model(name, model, X_tr, y_tr, X_va, y_va)
        val_scores[name] = _prob(model, X_va)
        test_scores[name] = _prob(model, X_te)
        metrics[name] = {
            "val": classification_metrics(y_va, val_scores[name]),
            "test": classification_metrics(y_te, test_scores[name]),
            "fit_seconds": round(time.time() - t0, 2),
        }
        fitted[name] = model
        print(f"  {name:<22} val ROC-AUC={metrics[name]['val']['roc_auc']:.4f} "
              f"PR-AUC={metrics[name]['val']['pr_auc']:.4f}")

    # The served model must be TreeExplainer-native (SHAP local attributions
    # at request time); other models remain as benchmarked baselines.
    preferred: list[str] = [n for n in ("lightgbm", "xgboost") if n in metrics]
    best_name = max(preferred or list(metrics.keys()),
                    key=lambda k: metrics[k]["val"]["pr_auc"])
    best = fitted[best_name]

    # Calibration ---------------------------------------------------------
    iso = fit_calibrator(val_scores[best_name], y_va)
    cal_test = iso.predict(test_scores[best_name])
    ece_raw = expected_calibration_error(y_te, test_scores[best_name])
    ece_cal = expected_calibration_error(y_te, np.asarray(cal_test))
    cal_curve = calibration_curve_data(y_te, np.asarray(cal_test))

    # Thresholds (business-facing, chosen on validation only) --------------
    thr_best = best_f1_threshold(y_va, val_scores[best_name])
    hi_cut = thr_best
    order = np.argsort(-val_scores[best_name])
    y_sorted = y_va[order]
    s_sorted = val_scores[best_name][order]
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)
    precisions = tps / np.maximum(tps + fps, 1)
    hits = np.flatnonzero(precisions >= 0.65)
    if len(hits):
        hi_cut = float(min(s_sorted[hits]))

    thresholds = {
        "medium_cutoff": 0.30,
        "high_cutoff": round(hi_cut, 4),
        "selection_rule": "smallest threshold with validation precision >= 0.65",
    }

    # Explainability --------------------------------------------------------
    bg_sample = X_tr.sample(min(2000, len(X_tr)), random_state=SEED)
    explainer = AttritionExplainer(best, FEATURE_COLUMNS, bg_sample)
    shap_sample = X_va.sample(min(3000, len(X_va)), random_state=SEED)
    global_imp = explainer.global_importance(shap_sample)
    numeric_top: list[str] = [str(g["feature"]) for g in global_imp
                              if g["feature"] not in CATEGORICAL_FEATURES][:6]
    pdp = explainer.partial_dependence(X_va, numeric_top)

    return {
        "metrics": metrics,
        "best_model_name": best_name,
        "calibration": {
            "method": "isotonic",
            "ece_before": round(ece_raw, 4),
            "ece_after": round(ece_cal, 4),
            "test_calibration_curve": cal_curve,
            "test_metrics_calibrated": classification_metrics(y_te, np.asarray(cal_test)),
        },
        "thresholds": thresholds,
        "global_importance": global_imp,
        "pdp": pdp,
        "_fitted": fitted,
        "_best": best,
        "_calibrator": iso,
    }


def train_salary_anomaly(df_engineered: pd.DataFrame, out_models: Path) -> object:
    feats = ["salary_log", "comp_ratio", "level", "log_tenure_months"]
    iso = IsolationForest(n_estimators=200, contamination=0.02, random_state=SEED)
    iso.fit(df_engineered[feats].astype(float))
    return iso


def train_cox(labelled_frame: pd.DataFrame) -> dict | None:
    """Discrete-time Cox PH on snapshot rows (survival baseline)."""
    try:
        from lifelines import CoxPHFitter
    except ImportError:
        warnings.warn("lifelines unavailable - skipping Cox PH", stacklevel=1)
        return None
    cols = ["tenure_months", "attrition_event", "comp_ratio", "workload_z",
            "engagement_score", "manager_sentiment", "months_since_promotion",
            "level", "performance_rating"]
    sub = labelled_frame[cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) > 40_000:
        sub = sub.sample(40_000, random_state=SEED)
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(sub, duration_col="tenure_months", event_col="attrition_event")
    return {
        "concordance": round(float(cph.concordance_index_), 4),
        "coefficients": {k: round(float(v), 4) for k, v in
                         cph.params_.items()},
    }


def train_promotion(df_engineered: pd.DataFrame, out_metrics: Path) -> dict | None:
    try:
        windows = derive_splits(int(df_engineered["month_index"].max()), PROMO_HORIZON)
    except ValueError:
        return None  # timeline too short for a 12m observable horizon
    promo = build_promotion_table(df_engineered)
    key = ["employee_id", "month_index"]
    merged = df_engineered.merge(promo[key + [PROMO_TARGET]], on=key, how="left")
    labelled = merged[merged[PROMO_TARGET].notna()].copy()
    labelled[PROMO_TARGET] = labelled[PROMO_TARGET].astype(int)

    def _slice(lo: int, hi: int) -> tuple[pd.DataFrame, pd.Series]:
        part = labelled[(labelled["month_index"] >= lo) & (labelled["month_index"] <= hi)]
        return part[FEATURE_COLUMNS], part[PROMO_TARGET]

    X_tr, y_tr = _slice(*windows["train"])
    X_va, y_va = _slice(*windows["val"])
    X_te, y_te = _slice(*windows["test"])
    if any(y.nunique() < 2 or len(y) < 500 for y in (y_tr, y_va, y_te)):
        return None

    model = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                           subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                           verbosity=-1)
    model.fit(X_tr, y_tr)
    return {
        "lightgbm": {
            "val": classification_metrics(y_va.to_numpy(), _prob(model, X_va)),
            "test": classification_metrics(y_te.to_numpy(), _prob(model, X_te)),
        }
    }


def main(data_dir: str = "artifacts/data", artifacts_dir: str = "artifacts") -> None:
    root = Path(artifacts_dir)
    out_models = root / "models"
    out_metrics = root / "metrics"
    out_models.mkdir(parents=True, exist_ok=True)
    out_metrics.mkdir(parents=True, exist_ok=True)

    print("[1/5] Loading dataset ...")
    bundle, df_full = build_attrition_bundle(data_dir)
    labelled = df_full[df_full["attrited_within_6m"].notna()]
    print(f"      train={len(bundle.X_train)} val={len(bundle.X_val)} "
          f"test={len(bundle.X_test)}")

    print("[2/5] Training attrition models ...")
    res = train_attrition(bundle, out_models, out_metrics)
    best_name: str = res["best_model_name"]

    print(f"[3/5] Saving models (winner={best_name}) ...")
    joblib.dump(res["_best"], out_models / "attrition_model.joblib")
    joblib.dump(res["_fitted"]["lightgbm"], out_models / "attrition_lightgbm.joblib")
    joblib.dump(res["_calibrator"], out_models / "calibrator.joblib")

    print("[4/5] Auxiliary models ...")
    attr_windows = derive_splits(int(df_full["month_index"].max()), 6)
    train_rows = df_full[df_full["month_index"].between(*attr_windows["train"])]
    anomaly = train_salary_anomaly(train_rows, out_models)
    joblib.dump(anomaly, out_models / "salary_anomaly.joblib")
    cox = train_cox(labelled)
    promo = train_promotion(df_full, out_metrics)

    numeric_bounds = {}
    tr_num = bundle.X_train[NUMERIC_FEATURES_FOR_BOUNDS]
    for feat in NUMERIC_FEATURES_FOR_BOUNDS:
        q1, q99 = tr_num[feat].quantile([0.01, 0.99])
        numeric_bounds[feat] = [round(float(q1), 4), round(float(q99), 4)]

    # Discrimination ceiling: rank employees by the *true* simulated hazard.
    # Because the generator's data-generating process is known, this proxies
    # the Bayes-optimal ROC-AUC any model could reach on this data.
    from sklearn.metrics import roc_auc_score

    test_rows = labelled[labelled["month_index"].between(
        *derive_splits(int(df_full["month_index"].max()), 6)["test"])]
    oracle_auc = float(roc_auc_score(test_rows["attrited_within_6m"],
                                     test_rows["attrition_hazard_true"]))

    meta = {
        "best_model_name": best_name,
        "feature_columns": FEATURE_COLUMNS,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "overtime_mean": bundle.overtime_mean,
        "overtime_std": bundle.overtime_std,
        "numeric_bounds": numeric_bounds,
        "thresholds": res["thresholds"],
        "data_fingerprint": _data_fingerprint(Path(data_dir) / "snapshots.parquet"),
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (root / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    report = {
        "attrition": {k: v for k, v in res.items() if not k.startswith("_")},
        "salary_anomaly": {"model": "IsolationForest", "contamination": 0.02},
        "cox_proportional_hazards": cox,
        "promotion": promo,
        "bayes_ceiling_proxy": {
            "test_oracle_hazard_auc": round(oracle_auc, 4),
            "note": "AUC of the TRUE simulated hazard vs realized outcomes on the "
                    "test window - theoretical limit for any observable-feature model.",
        },
    }
    (out_metrics / "metrics.json").write_text(json.dumps(report, indent=2),
                                              encoding="utf-8")
    save_json(res["global_importance"], out_metrics / "global_shap.json")
    save_json(res["pdp"], out_metrics / "pdp.json")

    print("[5/5] Optional MLflow tracking ...")
    _mlflow_log(root, res, meta)

    test_m = res["metrics"][best_name]["test"]
    print("\n=== Summary ===")
    print(f"best model       : {best_name}")
    print(f"test ROC-AUC     : {test_m['roc_auc']:.4f}")
    print(f"test PR-AUC      : {test_m['pr_auc']:.4f}")
    print(f"ECE raw->calib   : {res['calibration']['ece_before']} -> "
          f"{res['calibration']['ece_after']}")
    print(f"artifacts ->     : {root.resolve()}")


def _mlflow_log(root: Path, res: dict, meta: dict) -> None:
    import os

    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        print("      MLFLOW_TRACKING_URI not set - skipping")
        return
    try:
        import mlflow
        test_m = res["metrics"][res["best_model_name"]]["test"]
        mlflow.set_tracking_uri(uri)
        with mlflow.start_run(run_name=f"attrition-{time.strftime('%Y%m%d-%H%M')}"):
            mlflow.log_param("best_model", res["best_model_name"])
            mlflow.log_param("features", len(meta["feature_columns"]))
            mlflow.log_metric("test_roc_auc", test_m["roc_auc"])
            mlflow.log_metric("test_pr_auc", test_m["pr_auc"])
            mlflow.log_artifact(str(root / "meta.json"))
            mlflow.log_artifacts(str(root / "metrics"))
        print(f"      logged run to {uri}")
    except Exception as exc:  # noqa: BLE001 - tracking must never break training
        print(f"      MLflow logging failed (non-fatal): {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train workforce attrition models")
    parser.add_argument("--data-dir", default="artifacts/data")
    parser.add_argument("--artifacts-dir", default="artifacts")
    args = parser.parse_args()
    main(args.data_dir, args.artifacts_dir)
