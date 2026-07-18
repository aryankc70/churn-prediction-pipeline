import sys
import pandas as pd
sys.path.insert(0, "src")
from features import build_features, align_columns


def test_build_features_creates_domain_columns():
    df = pd.DataFrame([{
        "gender": "Male", "Partner": "Yes", "Dependents": "No",
        "PhoneService": "Yes", "PaperlessBilling": "No",
        "tenure": 12, "TotalCharges": 600.0, "MonthlyCharges": 50.0,
        "MultipleLines": "No", "InternetService": "DSL",
        "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "No", "StreamingMovies": "No",
        "Contract": "One year", "PaymentMethod": "Mailed check",
        "SeniorCitizen": 0,
    }])
    out = build_features(df)
    assert "tenure_bucket" in out.columns
    assert "charges_per_month_of_tenure" in out.columns


def test_align_columns_fills_missing():
    df = pd.DataFrame([{"a": 1}])
    out = align_columns(df, ["a", "b"])
    assert list(out.columns) == ["a", "b"]
    assert out["b"].iloc[0] == 0
