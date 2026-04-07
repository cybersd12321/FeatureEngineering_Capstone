"""
src/pipeline.py
===============
Scikit-learn Pipeline builder for the StaySmart Hotels
Cancellation Risk Prediction capstone.

Usage (in the notebook):
    from src.pipeline import build_pipeline, NUMERIC_COLS, CATEGORICAL_COLS
    pipe = build_pipeline(LogisticRegression(max_iter=1000))
    pipe.fit(X_train, y_train)
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler, OneHotEncoder, PowerTransformer
)

# ── Feature column lists ──────────────────────────────────────────────────────
# These match the notebook's df_clean (after dropping leaky / irrelevant cols)

NUMERIC_COLS = [
    "leadtime",
    "arrivaldateweeknumber",
    "arrivaldatedayofmonth",
    "staysinweekendnights",
    "staysinweeknights",
    "adults",
    "children",
    "babies",
    "isrepeatedguest",
    "previouscancellations",
    "previousbookingsnotcanceled",
    "bookingchanges",
    "daysinwaitinglist",
    "adr",
    "requiredcarparkingspaces",
    "totalofspecialrequests",
]

CATEGORICAL_COLS = [
    "hotel",
    "arrivaldatemonth",
    "meal",
    "country",
    "marketsegment",
    "distributionchannel",
    "reservedroomtype",
    "assignedroomtype",
    "deposittype",
    "customertype",
]


# ── Sub-pipelines ─────────────────────────────────────────────────────────────

def _numeric_pipeline(use_power_transform: bool = True) -> Pipeline:
    """
    Numeric preprocessing:
      1. SimpleImputer  – median strategy (robust to outliers & skew)
      2. PowerTransformer (Yeo-Johnson) + StandardScaler
         OR plain StandardScaler when use_power_transform=False

    Why Yeo-Johnson?
      Several hotel-booking features (leadtime, previouscancellations, adr)
      are heavily right-skewed. Yeo-Johnson maps them closer to Gaussian,
      which directly benefits linear models and improves distance-based ones.
      It handles zero and negative values, unlike Box-Cox.
    """
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if use_power_transform:
        steps.append(("power", PowerTransformer(method="yeo-johnson",
                                                standardize=False)))
    steps.append(("scaler", StandardScaler()))
    return Pipeline(steps)


def _categorical_pipeline() -> Pipeline:
    """
    Categorical preprocessing:
      1. SimpleImputer  – most_frequent strategy
      2. OneHotEncoder  – drop='first' to avoid multicollinearity;
                          handle_unknown='ignore' for unseen categories
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe",     OneHotEncoder(drop="first",
                                  handle_unknown="ignore",
                                  sparse_output=False)),
    ])


# ── Master builder ────────────────────────────────────────────────────────────

def build_pipeline(
    classifier,
    numeric_cols=None,
    categorical_cols=None,
    use_power_transform: bool = True,
) -> Pipeline:
    """
    Build a full sklearn Pipeline:
        ColumnTransformer (preprocess) → classifier

    Parameters
    ----------
    classifier        : any sklearn-compatible estimator
    numeric_cols      : list of numeric feature names (default: NUMERIC_COLS)
    categorical_cols  : list of categorical feature names (default: CATEGORICAL_COLS)
    use_power_transform : apply Yeo-Johnson before scaling (default: True)

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    if numeric_cols is None:
        numeric_cols = NUMERIC_COLS
    if categorical_cols is None:
        categorical_cols = CATEGORICAL_COLS

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", _numeric_pipeline(use_power_transform), numeric_cols),
            ("cat", _categorical_pipeline(),                categorical_cols),
        ],
        remainder="drop",       # drop any columns not listed above
        n_jobs=-1,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   classifier),
    ])


# ── Convenience: evaluate pipeline with cross-validation ─────────────────────

def cv_evaluate_pipeline(
    name: str,
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    scoring: str = "roc_auc",
) -> dict:
    """
    Run StratifiedKFold cross-validation and return mean ± std.

    Parameters
    ----------
    name     : human-readable model name
    pipeline : the built Pipeline object
    X, y     : full feature matrix and target
    cv       : number of folds (default 5)
    scoring  : sklearn scoring metric (default 'roc_auc')

    Returns
    -------
    dict with keys: Model, Mean_ROC_AUC, Std_ROC_AUC
    """
    from sklearn.model_selection import cross_val_score, StratifiedKFold

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=skf, scoring=scoring, n_jobs=-1)
    return {
        "Model":        name,
        f"Mean_{scoring.upper().replace('-','_')}": round(scores.mean(), 4),
        f"Std_{scoring.upper().replace('-','_')}":  round(scores.std(),  4),
    }
