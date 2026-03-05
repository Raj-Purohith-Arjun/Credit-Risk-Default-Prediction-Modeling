import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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


def _get_estimators(scale_pos_weight: float = 1.0) -> dict:
    """Get estimator configurations with optimized hyperparameters.
    
    Args:
        scale_pos_weight: Weight for positive class to handle imbalance (neg/pos ratio)
    """
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000, 
            random_state=RANDOM_STATE, 
            class_weight="balanced",
            solver="lbfgs",
            C=0.5,  # Slightly more regularization
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=RANDOM_STATE, 
            class_weight="balanced_subsample",
            n_jobs=-1,
            max_features="sqrt",
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            eval_metric="auc",
            verbosity=0,
            use_label_encoder=False,
        ),
    }


def build_pipeline(estimator) -> Pipeline:
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", estimator),
    ])


def train_all(X: pd.DataFrame, y: pd.Series) -> dict:
    """Train all models with cross-validation.
    
    Uses optimized hyperparameters and handles class imbalance.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = {}

    # Calculate class imbalance ratio for XGBoost scale_pos_weight
    neg_count = (y == 0).sum()
    pos_count = (y == 1).sum()
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    
    print(f"Class distribution: {neg_count:,} negative, {pos_count:,} positive (ratio: {scale_pos_weight:.2f})")

    for name, estimator in _get_estimators(scale_pos_weight).items():
        print(f"Training {name}...")
        pipe = build_pipeline(estimator)
        cv_scores = cross_validate(pipe, X, y, cv=cv, scoring=SCORING, n_jobs=-1)
        metrics = {
            metric: float(np.mean(cv_scores[f"test_{metric}"]))
            for metric in SCORING
        }
        std_metrics = {
            f"{metric}_std": float(np.std(cv_scores[f"test_{metric}"]))
            for metric in SCORING
        }
        pipe.fit(X, y)
        path = MODELS_DIR / f"{name}.joblib"
        joblib.dump(pipe, path)
        results[name] = {"metrics": {**metrics, **std_metrics}, "path": str(path)}
        print(f"  {name}: AUC={metrics['roc_auc']:.4f} (±{std_metrics['roc_auc_std']:.4f})  F1={metrics['f1']:.4f} (±{std_metrics['f1_std']:.4f})")

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    return results


def load_model(name: str) -> Pipeline:
    return joblib.load(MODELS_DIR / f"{name}.joblib")
