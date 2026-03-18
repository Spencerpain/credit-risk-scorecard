"""
xgboost_model.py
XGBoost model for credit default prediction.
Trained on raw (pre-WoE) features for a fair comparison with the scorecard.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier


DEFAULT_PARAMS = {
    "max_depth": 4,
    "learning_rate": 0.05,
    "n_estimators": 300,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "auc",
    "random_state": 42,
    "use_label_encoder": False,
}


def fit_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict = None,
) -> XGBClassifier:
    """
    Fit XGBoost with early stopping.
    scale_pos_weight handles class imbalance automatically.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    p["scale_pos_weight"] = neg / pos  # penalise majority class

    model = XGBClassifier(**p, early_stopping_rounds=20)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


def predict_proba_xgb(model: XGBClassifier, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def plot_feature_importance(
    model: XGBClassifier,
    feature_names: list,
    top_n: int = 15,
    ax=None,
):
    """Horizontal bar chart of top-N features by gain importance."""
    importance = model.get_booster().get_score(importance_type="gain")
    imp_df = (
        pd.DataFrame(importance.items(), columns=["feature", "gain"])
        .sort_values("gain", ascending=True)
        .tail(top_n)
    )
    # Map f0, f1... back to actual names if needed
    imp_df["feature"] = imp_df["feature"].apply(
        lambda x: feature_names[int(x[1:])] if x.startswith("f") and x[1:].isdigit() else x
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    ax.barh(imp_df["feature"], imp_df["gain"], color="#2166ac")
    ax.set_xlabel("Gain")
    ax.set_title("XGBoost Feature Importance (Gain)")
    plt.tight_layout()
    return ax


def plot_shap_summary(model: XGBClassifier, X_sample: pd.DataFrame):
    """SHAP beeswarm summary plot for model explainability."""
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    shap.summary_plot(shap_values, X_sample, show=True)
