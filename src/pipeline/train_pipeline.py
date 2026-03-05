"""End-to-end training pipeline."""
import sys
from pathlib import Path

# allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE, TEST_SIZE
from src.data.data_loader import load_raw
from src.data.preprocess import preprocess
from src.features.feature_engineering import engineer_features
from src.models.train import train_all, load_model
from src.models.evaluate import (
    evaluate_model,
    plot_roc_curves,
    plot_confusion_matrix,
    compare_models,
)
from src.explain.shap_analysis import run_shap


def run():
    print("Loading data...")
    df = load_raw()

    print("Preprocessing...")
    X, y = preprocess(df)

    print("Feature engineering...")
    X = engineer_features(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    print("Training models...")
    results = train_all(X_train, y_train)

    # load fitted pipelines for evaluation
    models = {name: load_model(name) for name in results}

    print("Evaluating models...")
    for name, pipe in models.items():
        metrics = evaluate_model(name, pipe, X_test, y_test)
        results[name]["test_metrics"] = metrics
        plot_confusion_matrix(name, pipe, X_test, y_test)

    compare_models(results)
    plot_roc_curves(models, X_test, y_test)

    print("Running SHAP analysis...")
    sample = X_test.sample(min(200, len(X_test)), random_state=RANDOM_STATE)
    for name, pipe in models.items():
        try:
            run_shap(name, pipe, sample)
        except Exception as e:
            print(f"SHAP skipped for {name}: {e}")

    print("Pipeline complete.")


if __name__ == "__main__":
    run()
