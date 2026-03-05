import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from src.config import SHAP_PLOTS_DIR


def _get_feature_names(pipe: Pipeline) -> list[str]:
    pre = pipe.named_steps["preprocessor"]
    try:
        return list(pre.get_feature_names_out())
    except Exception:
        n = pipe.named_steps["preprocessor"].transform(
            pd.DataFrame(columns=pre.feature_names_in_)
            if hasattr(pre, "feature_names_in_") else pd.DataFrame()
        ).shape[1]
        return [f"f{i}" for i in range(n)]


def _get_explainer(classifier, X_transformed: np.ndarray):
    is_tree = hasattr(classifier, "estimators_") or hasattr(classifier, "get_booster")
    if is_tree:
        return shap.TreeExplainer(classifier)
    return shap.LinearExplainer(classifier, X_transformed)


def _transform_X(pipe: Pipeline, X: pd.DataFrame) -> np.ndarray:
    return pipe.named_steps["preprocessor"].transform(X)


def run_shap(name: str, pipe: Pipeline, X_sample: pd.DataFrame):
    SHAP_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    classifier = pipe.named_steps["classifier"]
    X_transformed = _transform_X(pipe, X_sample)

    explainer = _get_explainer(classifier, X_transformed)
    shap_values = explainer.shap_values(X_transformed)

    # handle binary tree output
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values = shap_values[1]

    pre = pipe.named_steps["preprocessor"]
    try:
        feat_names = list(pre.get_feature_names_out())
    except Exception:
        feat_names = [f"f{i}" for i in range(X_transformed.shape[1])]

    # global summary plot
    shap.summary_plot(shap_values, X_transformed, feature_names=feat_names, show=False)
    plt.tight_layout()
    plt.savefig(SHAP_PLOTS_DIR / f"{name}_shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    # bar plot
    shap.summary_plot(shap_values, X_transformed, feature_names=feat_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(SHAP_PLOTS_DIR / f"{name}_shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    print(f"SHAP plots saved for {name}")
