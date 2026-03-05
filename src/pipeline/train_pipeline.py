"""End-to-end training pipeline."""
import sys
from pathlib import Path

# allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE, TEST_SIZE, TARGET
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
from src.visualization.eda_plots import run_eda_visualizations
from src.visualization.model_plots import run_model_visualizations


def run():
    print("=" * 60)
    print("CREDIT RISK DEFAULT PREDICTION PIPELINE")
    print("=" * 60)
    
    print("\n[1/7] Loading data...")
    df = load_raw()
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

    print("\n[2/7] Preprocessing...")
    X, y = preprocess(df)
    print(f"  Features: {X.shape[1]}, Samples: {len(X):,}")
    print(f"  Target distribution: {y.value_counts().to_dict()}")

    print("\n[3/7] Feature engineering...")
    X = engineer_features(X)
    print(f"  Total features after engineering: {X.shape[1]}")
    
    # Create combined dataframe for EDA visualization
    data_viz = X.copy()
    data_viz[TARGET] = y
    
    print("\n[4/7] Generating EDA visualizations...")
    eda_plots = run_eda_visualizations(data_viz)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"\n  Train set: {len(X_train):,} samples")
    print(f"  Test set: {len(X_test):,} samples")

    print("\n[5/7] Training models...")
    results = train_all(X_train, y_train)

    # load fitted pipelines for evaluation
    models = {name: load_model(name) for name in results}

    print("\n[6/7] Evaluating models...")
    for name, pipe in models.items():
        metrics = evaluate_model(name, pipe, X_test, y_test)
        results[name]["test_metrics"] = metrics
        plot_confusion_matrix(name, pipe, X_test, y_test)

    compare_models(results)
    plot_roc_curves(models, X_test, y_test)
    
    print("\n  Generating model performance visualizations...")
    model_plots = run_model_visualizations(models, results, X_test, y_test)

    print("\n[7/7] Running SHAP analysis...")
    sample = X_test.sample(min(200, len(X_test)), random_state=RANDOM_STATE)
    for name, pipe in models.items():
        try:
            run_shap(name, pipe, sample)
        except Exception as e:
            print(f"  SHAP skipped for {name}: {e}")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print("\nOutputs generated:")
    print("  - models/              : Trained model pipelines (.joblib)")
    print("  - reports/metrics.json : Cross-validation and test metrics")
    print("  - reports/eda_plots/   : Exploratory data analysis plots")
    print("  - reports/model_plots/ : Model performance visualizations")
    print("  - reports/shap_plots/  : SHAP feature importance plots")


if __name__ == "__main__":
    run()
