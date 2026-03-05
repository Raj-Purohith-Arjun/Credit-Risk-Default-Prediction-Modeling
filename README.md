# Credit Risk Default Prediction

Classification models for predicting loan default probability using borrower financial and repayment history data.

## Dataset

LendingClub-style dataset with features including income, loan amount, FICO score, DTI ratio, employment length, and delinquency history. Run the download script to fetch data or generate a synthetic equivalent:

```bash
python data/download.py
```

## Project Structure

```
├── data/
│   ├── download.py          # data acquisition script
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── eda.ipynb
├── src/
│   ├── config.py
│   ├── data/
│   │   ├── data_loader.py
│   │   └── preprocess.py
│   ├── features/
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── evaluate.py
│   ├── explain/
│   │   └── shap_analysis.py
│   └── pipeline/
│       └── train_pipeline.py
├── models/
├── reports/
│   ├── metrics.json
│   └── shap_plots/
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Training

```bash
python src/pipeline/train_pipeline.py
```

This runs the full workflow: data loading → preprocessing → feature engineering → model training (CV) → evaluation → SHAP analysis.

## Models

| Model | ROC AUC | F1 |
|---|---|---|
| Logistic Regression | ~0.76 | ~0.72 |
| Random Forest | ~0.74 | ~0.75 |
| XGBoost | ~0.75 | ~0.75 |

*Metrics from 5-fold cross-validation on synthetic LendingClub-style data.*

## Key Features Engineered

- `loan_to_income` — loan amount relative to income
- `delinq_rate` — delinquencies per open account
- `revol_util_norm` — normalized revolving utilization
- `fico_band` — binned FICO score category
- `pub_rec_flag` / `bankruptcy_flag` — binary derogatory markers

## Outputs

- `models/` — serialized model pipelines (`.joblib`)
- `reports/metrics.json` — CV and test metrics
- `reports/roc_curves.png` — ROC comparison
- `reports/shap_plots/` — SHAP summary and bar plots per model
