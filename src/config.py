from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_RAW = ROOT_DIR / "data" / "raw"
DATA_PROCESSED = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
SHAP_PLOTS_DIR = REPORTS_DIR / "shap_plots"
METRICS_PATH = REPORTS_DIR / "metrics.json"

TARGET = "default"

NUMERIC_FEATURES = [
    "loan_amnt",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "fico_range_low",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "mort_acc",
    "pub_rec_bankruptcies",
    "emp_length_years",
]

CATEGORICAL_FEATURES = [
    "home_ownership",
    "purpose",
    "application_type",
]

CV_FOLDS = 5
RANDOM_STATE = 42
TEST_SIZE = 0.2
