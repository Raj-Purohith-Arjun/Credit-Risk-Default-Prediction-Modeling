import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from xgboost import XGBClassifier
from src.config import (
    MODELS_DIR,
    METRICS_PATH,
    RANDOM_STATE,
    CV_FOLDS,
)
from src.data.preprocess import build_preprocessor

SCORING = ["accuracy", "roc_auc", "precision", "recall", "f1"]


def _get_estimators() -> dict:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1
        ),
        "xgboost": XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            verbosity=0,
        ),
    }


def build_pipeline(estimator) -> Pipeline:
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", estimator),
    ])


def train_all(X: pd.DataFrame, y: pd.Series) -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = {}

    for name, estimator in _get_estimators().items():
        pipe = build_pipeline(estimator)
        cv_scores = cross_validate(pipe, X, y, cv=cv, scoring=SCORING, n_jobs=-1)
        metrics = {
            metric: float(np.mean(cv_scores[f"test_{metric}"]))
            for metric in SCORING
        }
        pipe.fit(X, y)
        path = MODELS_DIR / f"{name}.joblib"
        joblib.dump(pipe, path)
        results[name] = {"metrics": metrics, "path": str(path)}
        print(f"{name}: AUC={metrics['roc_auc']:.4f}  F1={metrics['f1']:.4f}")

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    return results


def load_model(name: str) -> Pipeline:
    return joblib.load(MODELS_DIR / f"{name}.joblib")
