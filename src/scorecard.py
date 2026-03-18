"""
scorecard.py
Logistic regression scorecard with standard PDO scaling.

Scorecard scaling formula:
  Factor = PDO / ln(2)
  Offset = Base_Score - Factor * ln(Base_Odds)
  Score  = Offset + Factor * (intercept + sum(coef_i * WoE_i))

Each feature-bin contributes:
  Points_i = -(coef_i * WoE_ij + intercept/n_features) * Factor
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from optbinning import BinningProcess


def fit_scorecard(
    X_woe_train: pd.DataFrame,
    y_train: pd.Series,
    C: float = 1.0,
    random_state: int = 42,
) -> LogisticRegression:
    """Fit logistic regression on WoE-transformed features."""
    model = LogisticRegression(
        C=C,
        class_weight="balanced",
        max_iter=1000,
        random_state=random_state,
        solver="lbfgs",
    )
    model.fit(X_woe_train, y_train)
    return model


def to_points_table(
    model: LogisticRegression,
    binner: BinningProcess,
    feature_names: list,
    pdo: int = 20,
    base_score: int = 600,
    base_odds: float = 19.0,
) -> pd.DataFrame:
    """
    Convert logistic regression coefficients to a scorecard points table.

    PDO (Points to Double the Odds): score shift that doubles the bad/good odds.
    Base score: score corresponding to base_odds.
    Base odds: good-to-bad odds at the base score (e.g. 19 means 19 goods per 1 bad).
    """
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)

    n_features = len(feature_names)
    intercept = model.intercept_[0]
    coefs = model.coef_[0]

    rows = []
    for i, feat in enumerate(feature_names):
        try:
            ob = binner.get_binned_variable(feat)
            bt = ob.binning_table.build()
            bt = bt[~bt.index.isin(["Special", "Missing", "Totals"])].copy()
            bt = bt[bt["Count"] > 0]

            for bin_label, row in bt.iterrows():
                woe = row["WoE"]
                # Distribute intercept evenly across features
                points = -(coefs[i] * woe + intercept / n_features) * factor
                rows.append({
                    "feature": feat,
                    "bin": bin_label,
                    "woe": round(woe, 4),
                    "coefficient": round(coefs[i], 4),
                    "points": round(points),
                })
        except Exception:
            continue

    return pd.DataFrame(rows)


def predict_score(
    model: LogisticRegression,
    binner: BinningProcess,
    X_raw: pd.DataFrame,
    feature_names: list,
    pdo: int = 20,
    base_score: int = 600,
    base_odds: float = 19.0,
) -> np.ndarray:
    """
    Predict integer scorecard scores for raw (pre-WoE) input data.
    Higher score = lower risk.
    """
    from src.binning import transform_woe

    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)

    X_woe = transform_woe(binner, X_raw[feature_names])
    log_odds = model.intercept_[0] + X_woe.values @ model.coef_[0]
    scores = offset + factor * log_odds
    return np.round(scores).astype(int)


def predict_proba_scorecard(
    model: LogisticRegression,
    X_woe: pd.DataFrame,
) -> np.ndarray:
    """Return probability of default (class=1) for WoE-transformed input."""
    return model.predict_proba(X_woe)[:, 1]
