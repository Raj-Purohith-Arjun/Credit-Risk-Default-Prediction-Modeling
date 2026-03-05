import pandas as pd
import numpy as np


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["loan_to_income"] = df["loan_amnt"] / (df["annual_inc"] + 1)
    df["revol_util_norm"] = df["revol_util"] / 100
    df["delinq_rate"] = df["delinq_2yrs"] / (df["total_acc"] + 1)
    df["credit_age_proxy"] = df["total_acc"] - df["open_acc"]
    df["pub_rec_flag"] = (df["pub_rec"] > 0).astype(int)
    df["bankruptcy_flag"] = (df["pub_rec_bankruptcies"] > 0).astype(int)
    return df


def add_fico_band(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    bins = [0, 579, 669, 739, 799, 900]
    labels = ["poor", "fair", "good", "very_good", "exceptional"]
    df["fico_band"] = pd.cut(
        df["fico_range_low"], bins=bins, labels=labels, right=True
    ).astype(str)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_ratio_features(df)
    df = add_fico_band(df)
    return df


ENGINEERED_NUMERIC = [
    "loan_to_income",
    "revol_util_norm",
    "delinq_rate",
    "credit_age_proxy",
    "pub_rec_flag",
    "bankruptcy_flag",
]

ENGINEERED_CATEGORICAL = ["fico_band"]
