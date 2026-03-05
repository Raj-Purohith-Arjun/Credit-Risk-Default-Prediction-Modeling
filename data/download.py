"""Download LendingClub dataset or generate synthetic fallback."""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent / "raw"


def _download_kaggle(output_path: Path) -> bool:
    try:
        import kaggle  # noqa: F401
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            "wordsforthewise/lending-club",
            path=str(output_path),
            unzip=True,
        )
        return True
    except Exception:
        return False


def _generate_synthetic(n: int = 50_000) -> pd.DataFrame:
    rng = np.random.default_rng(42)

    loan_amnt = rng.uniform(1_000, 40_000, n)
    annual_inc = rng.lognormal(10.8, 0.6, n)
    fico = rng.integers(620, 850, n)
    dti = rng.uniform(0, 40, n)
    emp_length = rng.integers(0, 11, n).astype(float)
    delinq = rng.integers(0, 5, n)
    open_acc = rng.integers(2, 30, n)
    pub_rec = rng.integers(0, 3, n)
    revol_bal = rng.uniform(0, 30_000, n)
    revol_util = rng.uniform(0, 100, n)
    total_acc = open_acc + rng.integers(0, 20, n)
    mort_acc = rng.integers(0, 5, n)
    pub_rec_bankruptcies = rng.integers(0, 2, n)

    home_ownership = rng.choice(["RENT", "MORTGAGE", "OWN", "OTHER"], n, p=[0.47, 0.44, 0.08, 0.01])
    purpose = rng.choice(
        ["debt_consolidation", "credit_card", "home_improvement", "other", "major_purchase"],
        n, p=[0.50, 0.22, 0.10, 0.12, 0.06],
    )
    application_type = rng.choice(["Individual", "Joint App"], n, p=[0.88, 0.12])

    # logistic default probability
    log_odds = (
        -2.0
        + 0.05 * dti
        - 0.006 * (fico - 680)
        + 0.4 * delinq
        + 0.3 * pub_rec
        - 0.000008 * annual_inc
        + 0.00003 * loan_amnt
        + 0.01 * revol_util
    )
    prob = 1 / (1 + np.exp(-log_odds))
    default = rng.binomial(1, prob, n)

    return pd.DataFrame({
        "loan_amnt": loan_amnt.round(2),
        "annual_inc": annual_inc.round(2),
        "dti": dti.round(2),
        "delinq_2yrs": delinq,
        "fico_range_low": fico,
        "emp_length_years": emp_length,
        "open_acc": open_acc,
        "pub_rec": pub_rec,
        "revol_bal": revol_bal.round(2),
        "revol_util": revol_util.round(2),
        "total_acc": total_acc,
        "mort_acc": mort_acc,
        "pub_rec_bankruptcies": pub_rec_bankruptcies,
        "home_ownership": home_ownership,
        "purpose": purpose,
        "application_type": application_type,
        "default": default,
    })


def download(output_dir: Path = RAW_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "lending_club.csv"

    if out_path.exists():
        return out_path

    print("Attempting Kaggle download...")
    if _download_kaggle(output_dir):
        print("Kaggle download succeeded.")
        return out_path

    print("Kaggle unavailable — generating synthetic dataset.")
    df = _generate_synthetic()
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df):,} rows to {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50_000)
    args = parser.parse_args()
    download()
