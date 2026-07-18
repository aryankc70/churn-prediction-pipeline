"""Train baseline + XGBoost, log everything to MLflow, save best model."""
import json
from pathlib import Path

import mlflow
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from features import build_features

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data/processed"
MODELS = ROOT / "models"


def load():
    train = build_features(pd.read_csv(PROCESSED / "train.csv"))
    test = build_features(pd.read_csv(PROCESSED / "test.csv"))
    test = test.reindex(columns=train.columns, fill_value=0)
    X_tr, y_tr = train.drop(columns=["Churn"]), train["Churn"]
    X_te, y_te = test.drop(columns=["Churn"]), test["Churn"]
    return X_tr, y_tr, X_te, y_te


def main():
    mlflow.set_experiment("churn-prediction")
    X_tr, y_tr, X_te, y_te = load()

    with mlflow.start_run(run_name="logreg-baseline"):
        scaler = StandardScaler().fit(X_tr)
        lr = LogisticRegression(max_iter=2000).fit(scaler.transform(X_tr), y_tr)
        proba = lr.predict_proba(scaler.transform(X_te))[:, 1]
        mlflow.log_metric("roc_auc", roc_auc_score(y_te, proba))
        mlflow.log_metric("f1", f1_score(y_te, proba > 0.5))

    with mlflow.start_run(run_name="xgboost"):
        params = dict(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.8,
            scale_pos_weight=float((y_tr == 0).sum()) / float((y_tr == 1).sum()),
            eval_metric="auc",
        )
        mlflow.log_params(params)
        model = xgb.XGBClassifier(**params).fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)[:, 1]
        auc, f1 = roc_auc_score(y_te, proba), f1_score(y_te, proba > 0.5)
        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("f1", f1)
        print(f"XGBoost  ROC-AUC={auc:.4f}  F1={f1:.4f}")

        MODELS.mkdir(exist_ok=True)
        model.save_model(MODELS / "model.json")
        (MODELS / "columns.json").write_text(json.dumps(list(X_tr.columns)))


if __name__ == "__main__":
    main()
