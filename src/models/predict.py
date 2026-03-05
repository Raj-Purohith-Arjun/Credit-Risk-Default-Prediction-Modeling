import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def predict_proba(pipe: Pipeline, X: pd.DataFrame) -> np.ndarray:
    return pipe.predict_proba(X)[:, 1]


def predict(pipe: Pipeline, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
    return (predict_proba(pipe, X) >= threshold).astype(int)
