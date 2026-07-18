"""Feature engineering shared by training and serving."""
import pandas as pd

BINARY_MAP = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
CATEGORICAL = [
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
        if col in df:
            df[col] = df[col].map(BINARY_MAP).fillna(0).astype(int)
    df["tenure_bucket"] = pd.cut(df["tenure"], [-1, 6, 24, 48, 100], labels=False)
    df["charges_per_month_of_tenure"] = df["TotalCharges"] / df["tenure"].clip(lower=1)
    df = pd.get_dummies(df, columns=[c for c in CATEGORICAL if c in df], drop_first=True)
    return df


def align_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Make an inference frame match training columns exactly."""
    for c in columns:
        if c not in df:
            df[c] = 0
    return df[columns]
