import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from src.config import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET,
    DATA_PROCESSED,
)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # keep relevant columns if raw LendingClub data
    keep = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
    existing = [c for c in keep if c in df.columns]
    df = df[existing]
    # parse emp_length to numeric if present as string
    if "emp_length" in df.columns and "emp_length_years" not in df.columns:
        df["emp_length_years"] = (
            df["emp_length"]
            .str.extract(r"(\d+)")
            .astype(float)
        )
        df.drop(columns=["emp_length"], inplace=True)
    df.dropna(subset=[TARGET], inplace=True)
    return df


def build_preprocessor() -> ColumnTransformer:
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", num_pipeline, NUMERIC_FEATURES),
        ("cat", cat_pipeline, CATEGORICAL_FEATURES),
    ], remainder="drop")


def preprocess(df: pd.DataFrame, save: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    df = _clean(df)
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)
    if save:
        DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        df.to_parquet(DATA_PROCESSED / "features.parquet", index=False)
    return X, y
