"""
evaluation.py
Model evaluation metrics: AUC-ROC, KS statistic, Gini coefficient.
Shared across logistic regression scorecard and XGBoost.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve


def compute_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return roc_auc_score(y_true, y_prob)


def compute_ks(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    KS Statistic: maximum separation between cumulative distributions
    of predicted probabilities for goods (0) and bads (1).
    Range: 0 to 1. Higher is better. > 0.4 is considered strong.
    """
    df = pd.DataFrame({"prob": y_prob, "target": y_true}).sort_values("prob")
    n_bad = (y_true == 1).sum()
    n_good = (y_true == 0).sum()
    df["cum_bad"] = (df["target"] == 1).cumsum() / n_bad
    df["cum_good"] = (df["target"] == 0).cumsum() / n_good
    ks = (df["cum_bad"] - df["cum_good"]).abs().max()
    return float(ks)


def compute_gini(auc: float) -> float:
    """Gini = 2 * AUC - 1. Range: 0 to 1. Equivalent to Somers' D."""
    return 2 * auc - 1


def evaluation_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
) -> dict:
    auc = compute_auc(y_true, y_prob)
    ks = compute_ks(y_true, y_prob)
    gini = compute_gini(auc)
    return {
        "model": model_name,
        "AUC-ROC": round(auc, 4),
        "KS Statistic": round(ks, 4),
        "Gini Coefficient": round(gini, 4),
    }


def compare_models(reports: list) -> pd.DataFrame:
    """Takes a list of evaluation_report dicts, returns styled comparison DataFrame."""
    return pd.DataFrame(reports).set_index("model")


def plot_roc_curves(models_dict: dict, y_true: np.ndarray, ax=None):
    """
    models_dict: {"Model Name": y_prob_array, ...}
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for (name, y_prob), color in zip(models_dict.items(), colors):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = compute_auc(y_true, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color, lw=2)

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right")
    plt.tight_layout()
    return ax


def plot_ks_chart(y_true: np.ndarray, y_prob: np.ndarray, model_name: str, ax=None):
    """Cumulative distributions of bads and goods — visualises the KS gap."""
    df = pd.DataFrame({"prob": y_prob, "target": y_true}).sort_values("prob")
    n_bad = (y_true == 1).sum()
    n_good = (y_true == 0).sum()
    df["cum_bad"] = (df["target"] == 1).cumsum() / n_bad
    df["cum_good"] = (df["target"] == 0).cumsum() / n_good

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(df["prob"], df["cum_bad"], label="Bads (Defaults)", color="#d73027", lw=2)
    ax.plot(df["prob"], df["cum_good"], label="Goods", color="#1a9641", lw=2)

    # Mark KS point
    ks_idx = (df["cum_bad"] - df["cum_good"]).abs().idxmax()
    ks_x = df.loc[ks_idx, "prob"]
    ks_y1 = df.loc[ks_idx, "cum_bad"]
    ks_y2 = df.loc[ks_idx, "cum_good"]
    ax.vlines(ks_x, ks_y2, ks_y1, colors="navy", linestyles="dashed", lw=1.5)
    ax.annotate(
        f"KS={abs(ks_y1-ks_y2):.3f}",
        xy=(ks_x, (ks_y1 + ks_y2) / 2),
        xytext=(ks_x + 0.05, (ks_y1 + ks_y2) / 2),
        fontsize=9,
        color="navy",
    )
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Cumulative %")
    ax.set_title(f"KS Chart — {model_name}")
    ax.legend()
    plt.tight_layout()
    return ax
