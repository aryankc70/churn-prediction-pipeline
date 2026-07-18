"""Load raw Telco churn CSV, clean it, and write train/test splits."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

RAW = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
OUT = Path("data/processed")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # TotalCharges has blank strings for brand-new customers
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    return df.drop(columns=["customerID"])


def main():
    df = clean(pd.read_csv(RAW))
    train, test = train_test_split(df, test_size=0.2, stratify=df["Churn"], random_state=42)
    OUT.mkdir(parents=True, exist_ok=True)
    train.to_csv(OUT / "train.csv", index=False)
    test.to_csv(OUT / "test.csv", index=False)
    print(f"train={len(train)} test={len(test)} churn_rate={df['Churn'].mean():.3f}")


if __name__ == "__main__":
    main()
