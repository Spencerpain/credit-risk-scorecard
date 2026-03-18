"""
preprocessing.py
Data loading, cleaning, and train/test splitting for the Give Me Some Credit dataset.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


TARGET = "SeriousDlqin2yrs"

FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.columns = df.columns.str.strip()
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # MonthlyIncome: ~19% missing — impute with median, add missingness flag
    df["MonthlyIncome_missing"] = df["MonthlyIncome"].isna().astype(int)
    df["MonthlyIncome"] = df["MonthlyIncome"].fillna(df["MonthlyIncome"].median())
    # NumberOfDependents: ~2.5% missing — impute with median
    df["NumberOfDependents"] = df["NumberOfDependents"].fillna(
        df["NumberOfDependents"].median()
    )
    return df


def cap_outliers(
    df: pd.DataFrame,
    cols: list,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> pd.DataFrame:
    """Winsorise extreme values to [lower_q, upper_q] percentile range."""
    df = df.copy()
    for col in cols:
        lo = df[col].quantile(lower_q)
        hi = df[col].quantile(upper_q)
        df[col] = df[col].clip(lo, hi)
    return df


def prepare_data(path: str, test_size: float = 0.3, random_state: int = 42):
    """Full pipeline: load → clean → split. Returns X_train, X_test, y_train, y_test."""
    df = load_data(path)
    df = handle_missing_values(df)

    continuous_cols = [
        "RevolvingUtilizationOfUnsecuredLines",
        "DebtRatio",
        "MonthlyIncome",
    ]
    df = cap_outliers(df, continuous_cols)

    # Drop rows where target is missing
    df = df.dropna(subset=[TARGET])

    all_features = FEATURES + ["MonthlyIncome_missing"]
    X = df[all_features]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test
