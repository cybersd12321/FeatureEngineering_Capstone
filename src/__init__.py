"""
src package — reusable utilities for the StaySmart Hotels
Feature Engineering Capstone.

Quick imports:
    from src.helpers  import evaluate_model, safe_agg_feature
    from src.pipeline import build_pipeline, NUMERIC_COLS, CATEGORICAL_COLS
"""
from .helpers  import evaluate_model, safe_agg_feature, plot_roc_curve, bar_compare
from .pipeline import build_pipeline, NUMERIC_COLS, CATEGORICAL_COLS, cv_evaluate_pipeline

__all__ = [
    # helpers
    "evaluate_model",
    "safe_agg_feature",
    "plot_roc_curve",
    "bar_compare",
    # pipeline
    "build_pipeline",
    "NUMERIC_COLS",
    "CATEGORICAL_COLS",
    "cv_evaluate_pipeline",
]
