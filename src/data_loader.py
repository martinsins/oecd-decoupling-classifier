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

def make_forecast_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a one-year-ahead target:
    decoupled_next at year t equals decoupled at year t+1, within each country.
    """
    out = df.sort_values(["iso3", "year"]).copy()
    out["decoupled_next"] = out.groupby("iso3")[TARGET_COL].shift(-1)
    # drop last year per country (no t+1 label available)
    out = out.dropna(subset=["decoupled_next"]).copy()
    out["decoupled_next"] = out["decoupled_next"].astype(int)
    return out


def chronological_split_forecast(df: pd.DataFrame):
    """
    Forecasting split based on the FEATURE year.
    Feature years:
      train: 2010–2020  -> targets 2011–2021
      val  : 2021       -> targets 2022
      test : 2022       -> targets 2023
    """
    df_fc = make_forecast_target(df)

    train = df_fc[df_fc["year"].between(2010, 2020)].copy()
    val   = df_fc[df_fc["year"] == 2021].copy()
    test  = df_fc[df_fc["year"] == 2022].copy()

    def _xy(d: pd.DataFrame):
        X = d[FEATURE_COLS].copy()
        y = d["decoupled_next"].copy()
        return X, y, d

    X_train, y_train, df_train = _xy(train)
    X_val, y_val, df_val = _xy(val)
    X_test, y_test, df_test = _xy(test)

    return X_train, X_val, X_test, y_train, y_val, y_test, df_train, df_val, df_test
