"""
binning.py
WoE (Weight of Evidence) binning and IV (Information Value) calculation
using the optbinning library.

IV interpretation:
  < 0.02  — Useless
  0.02–0.1 — Weak
  0.1–0.3  — Medium
  0.3–0.5  — Strong
  > 0.5   — Suspicious (likely data leakage)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from optbinning import BinningProcess


def fit_woe_bins(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    min_iv: float = 0.02,
) -> tuple:
    """
    Fit WoE bins on training data.

    Returns:
        binner: fitted BinningProcess object
        iv_table: DataFrame with IV per feature, sorted descending
        selected_features: list of features with IV >= min_iv
    """
    feature_names = X_train.columns.tolist()

    binner = BinningProcess(
        variable_names=feature_names,
        max_n_bins=10,
        min_bin_size=0.05,
    )
    binner.fit(X_train.values, y_train.values)

    # Build IV summary table
    rows = []
    for feat in feature_names:
        try:
            ob = binner.get_binned_variable(feat)
            iv = ob.iv
        except Exception:
            iv = 0.0
        rows.append({"feature": feat, "iv": iv})

    iv_table = (
        pd.DataFrame(rows)
        .sort_values("iv", ascending=False)
        .reset_index(drop=True)
    )
    iv_table["strength"] = iv_table["iv"].apply(_iv_label)

    selected_features = iv_table.loc[iv_table["iv"] >= min_iv, "feature"].tolist()

    return binner, iv_table, selected_features


def transform_woe(binner: BinningProcess, X: pd.DataFrame) -> pd.DataFrame:
    """Apply fitted WoE bins to a DataFrame. Returns WoE-transformed DataFrame."""
    X_woe = binner.transform(X.values, metric="woe")
    return pd.DataFrame(X_woe, columns=X.columns, index=X.index)


def plot_woe_chart(binner: BinningProcess, feature: str, ax=None):
    """Bar chart of WoE values per bin for a single feature."""
    ob = binner.get_binned_variable(feature)
    df = ob.binning_table.build()
    # Drop totals rows
    df = df[~df.index.isin(["Special", "Missing", "Totals"])].copy()
    df = df[df["Count"] > 0]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    colors = ["#d73027" if w < 0 else "#1a9641" for w in df["WoE"]]
    ax.bar(range(len(df)), df["WoE"], color=colors, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df.index, rotation=45, ha="right", fontsize=8)
    ax.set_title(f"WoE by Bin — {feature}")
    ax.set_ylabel("WoE")
    plt.tight_layout()
    return ax


def _iv_label(iv: float) -> str:
    if iv < 0.02:
        return "Useless"
    elif iv < 0.1:
        return "Weak"
    elif iv < 0.3:
        return "Medium"
    elif iv < 0.5:
        return "Strong"
    else:
        return "Suspicious"
