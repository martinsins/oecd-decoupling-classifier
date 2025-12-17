from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier   # ← THIS LINE

def make_logistic_model(random_state: int = 42) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=2000,
                random_state=random_state,
                class_weight="balanced"   # <-- add this
            )),
        ]
    )

def make_tree_model(random_state: int = 42) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(max_depth=3, random_state=random_state)


def naive_carry_forward_baseline(
    df_test: pd.DataFrame,
    df_full: pd.DataFrame,
    target_col: str = "decoupled",
    majority_class: int = 0,
) -> np.ndarray:
    """
    Predict each country-year as last year's label for that country.
    Requires df_test and df_full to contain: iso3, year, target_col.
    """
    prev = df_full[["iso3", "year", target_col]].copy()
    prev["year"] = prev["year"] + 1
    prev = prev.rename(columns={target_col: "prev_label"})

    out = df_test[["iso3", "year"]].merge(prev, on=["iso3", "year"], how="left")
    out["prev_label"] = out["prev_label"].fillna(majority_class).astype(int)
    return out["prev_label"].to_numpy()

def majority_class_baseline(
    y_train: pd.Series,
) -> int:
    """
    Return the majority class label in the training labels.
    """
    # value_counts() returns counts by class label; idxmax() gives the most frequent label
    return int(y_train.value_counts().idxmax())

def forecast_carry_forward_baseline(
    df_test: pd.DataFrame,
    df_full: pd.DataFrame,
    target_col: str = "decoupled",
    majority_class: int = 0,
) -> np.ndarray:
    """
    Forecast baseline: predict decoupled_{t+1} using observed decoupled_{t}.
    For each test row at feature year t, return label at year t for that country.
    """
    cur = df_full[["iso3", "year", target_col]].copy()
    cur = cur.rename(columns={target_col: "cur_label"})

    out = df_test[["iso3", "year"]].merge(cur, on=["iso3", "year"], how="left")
    out["cur_label"] = out["cur_label"].fillna(majority_class).astype(int)
    return out["cur_label"].to_numpy()

