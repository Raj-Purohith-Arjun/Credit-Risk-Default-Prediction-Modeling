import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)
from src.config import REPORTS_DIR
from src.models.predict import predict, predict_proba


def evaluate_model(name: str, pipe, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = predict(pipe, X_test)
    y_proba = predict_proba(pipe, X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    return {
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "accuracy": report["accuracy"],
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
        "f1": report["1"]["f1-score"],
    }


def plot_roc_curves(models: dict, X_test: pd.DataFrame, y_test: pd.Series):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, pipe in models.items():
        y_proba = predict_proba(pipe, X_test)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC Curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "roc_curves.png", dpi=150)
    plt.close(fig)


def plot_confusion_matrix(name: str, pipe, X_test: pd.DataFrame, y_test: pd.Series):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    y_pred = predict(pipe, X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f"Confusion Matrix — {name}")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / f"confusion_matrix_{name}.png", dpi=150)
    plt.close(fig)


def compare_models(results: dict) -> pd.DataFrame:
    rows = []
    for name, info in results.items():
        row = {"model": name}
        row.update(info["metrics"])
        rows.append(row)
    df = pd.DataFrame(rows).set_index("model")
    print("\nModel Comparison:")
    print(df.to_string())
    return df
