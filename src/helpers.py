"""
src/helpers.py
==============
Reusable utility functions for the StaySmart Hotels — Feature Engineering Capstone.

Import in the notebook:
    from src.helpers import evaluate_model, safe_agg_feature, plot_roc_curve, bar_compare
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, roc_curve
)


def evaluate_model(name, model, X_tr, X_te, y_tr, y_te, fit=True):
    """
    Fit (optional) and evaluate a binary classifier.

    Parameters
    ----------
    name         : str   — label shown in comparison tables
    model        : sklearn estimator with fit / predict / predict_proba
    X_tr, y_tr   : training features & target
    X_te, y_te   : test features & target
    fit          : bool  — if False, assumes model is already fitted

    Returns
    -------
    dict with keys: Model, Accuracy, ROC-AUC, F1, _prob, _pred
    """
    if fit:
        model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    prob = model.predict_proba(X_te)[:, 1]
    return {
        "Model":    name,
        "Accuracy": round(accuracy_score(y_te, pred), 4),
        "ROC-AUC":  round(roc_auc_score(y_te,  prob), 4),
        "F1":       round(f1_score(y_te,        pred), 4),
        "_prob":    prob,
        "_pred":    pred,
    }


def safe_agg_feature(df_train_index, df_full, group_col, target_col, new_col):
    """
    Leakage-safe group-mean aggregation.

    Computes mean(target_col) per group using TRAINING ROWS ONLY,
    then maps the result onto the full dataframe.

    Why this prevents leakage
    -------------------------
    If we used the full dataset, test-row targets would leak into the feature
    through the group mean. Training-only aggregation mimics production,
    where future cancellation rates are unknown at prediction time.

    Parameters
    ----------
    df_train_index : Index of training rows within df_full
    df_full        : full DataFrame (modified in-place)
    group_col      : column to group by     (e.g. 'country')
    target_col     : column to aggregate    (e.g. 'is_canceled')
    new_col        : name of new feature    (e.g. 'country_cancel_rate')

    Returns
    -------
    (df_full, agg_map)  — agg_map is the {group: mean} Series
    """
    train_subset = df_full.loc[df_train_index]
    agg_map = train_subset.groupby(group_col)[target_col].mean()
    df_full[new_col] = df_full[group_col].map(agg_map)
    return df_full, agg_map


def plot_roc_curve(ax, results_list, title="ROC Curve"):
    """
    Overlay ROC curves for a list of evaluate_model result dicts.

    Each dict must contain: _prob, _y_te, ROC-AUC, Model.
    """
    for r in results_list:
        fpr, tpr, _ = roc_curve(r["_y_te"], r["_prob"])
        ax.plot(fpr, tpr, label=f"{r['Model']} (AUC={r['ROC-AUC']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(fontsize=8)


def bar_compare(df_results, metric_cols=("Accuracy", "ROC-AUC", "F1"),
                title="Model Comparison", ax=None):
    """
    Grouped bar chart comparing multiple models on Accuracy / ROC-AUC / F1.

    Parameters
    ----------
    df_results  : DataFrame with a 'Model' column + metric columns
    metric_cols : tuple of metric names to plot
    title       : chart title
    ax          : matplotlib Axes to draw on (new figure created if None)

    Returns
    -------
    ax
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(df_results))
    w = 0.25
    colors = ["steelblue", "coral", "seagreen"]
    for i, m in enumerate(metric_cols):
        ax.bar(x + i * w, df_results[m], w, label=m,
               color=colors[i], alpha=0.85)
    ax.set_xticks(x + w)
    ax.set_xticklabels(df_results["Model"], rotation=15,
                       ha="right", fontsize=8)
    ax.set_ylim(0.5, 1.02)
    ax.set_title(title)
    ax.legend()
    return ax
