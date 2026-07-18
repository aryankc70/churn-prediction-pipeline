# End-to-End Churn Prediction Pipeline

Production-style ML pipeline: data ingestion → feature engineering → XGBoost training (tracked with MLflow) → containerized FastAPI serving → CI/CD with GitHub Actions.

## Architecture
```
data/raw → src/data.py → src/features.py → src/train.py (MLflow) → models/model.json → src/api.py (FastAPI, Docker)
```

## Dataset
[Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` into `data/raw/`.

## Quickstart
```bash
pip install -r requirements.txt
python src/data.py          # clean + split
python src/features.py      # build feature matrix
python src/train.py         # train + log to MLflow
uvicorn src.api:app --reload  # serve predictions
```

## Docker
```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

## Results
| Model | ROC-AUC | F1 |
|---|---|---|
| Logistic Regression (baseline) | _fill in_ | _fill in_ |
| XGBoost (tuned) | _fill in_ | _fill in_ |

Run `mlflow ui` to browse experiments.

## Tests & CI
`pytest tests/` runs locally; GitHub Actions runs lint + tests on every push.
