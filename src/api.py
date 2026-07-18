"""FastAPI service that scores a single customer for churn risk."""
import json
from pathlib import Path

import pandas as pd
import xgboost as xgb
from fastapi import FastAPI
from pydantic import BaseModel, Field

from .features import align_columns, build_features

app = FastAPI(title="Churn Prediction API")
_model = xgb.XGBClassifier()
_model.load_model(Path("models/model.json"))
_columns = [c for c in json.loads(Path("models/columns.json").read_text()) if c != "Churn"]


class Customer(BaseModel):
    gender: str = "Female"
    SeniorCitizen: int = 0
    Partner: str = "No"
    Dependents: str = "No"
    tenure: int = Field(ge=0, default=1)
    PhoneService: str = "Yes"
    MultipleLines: str = "No"
    InternetService: str = "Fiber optic"
    OnlineSecurity: str = "No"
    OnlineBackup: str = "No"
    DeviceProtection: str = "No"
    TechSupport: str = "No"
    StreamingTV: str = "No"
    StreamingMovies: str = "No"
    Contract: str = "Month-to-month"
    PaperlessBilling: str = "Yes"
    PaymentMethod: str = "Electronic check"
    MonthlyCharges: float = 70.0
    TotalCharges: float = 70.0


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(customer: Customer):
    df = build_features(pd.DataFrame([customer.model_dump()]))
    df = align_columns(df, _columns)
    proba = float(_model.predict_proba(df)[0, 1])
    return {"churn_probability": round(proba, 4), "high_risk": proba > 0.5}
