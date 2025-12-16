from __future__ import annotations
import pandas as pd

FEATURE_COLS = ["gdp_growth", "co2_per_capita", "co2"]
TARGET_COL = "decoupled"


def load_panel(path: str = "data/processed/oecd_panel_2010_2023.csv") -> pd.DataFrame:
    df = pd.read_csv(path).copy()

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)

    return df


def chronological_split(df: pd.DataFrame):
    train = df[df["year"].between(2010, 2021)].copy()
    val = df[df["year"] == 2022].copy()
    test = df[df["year"] == 2023].copy()

    def _xy(d: pd.DataFrame):
        X = d[FEATURE_COLS].copy()
        y = d[TARGET_COL].copy()
        return X, y, d

    X_train, y_train, df_train = _xy(train)
    X_val, y_val, df_val = _xy(val)
    X_test, y_test, df_test = _xy(test)

    return X_train, X_val, X_test, y_train, y_val, y_test, df_train, df_val, df_test
