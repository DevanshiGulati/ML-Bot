"""
Model Trainer
-------------
Loops over the models chosen by the Decision Agent,
builds a pipeline for each, runs RandomizedSearchCV,
and returns a leaderboard + the best fitted pipelines.
"""

import time
import numpy as np
import pandas as pd

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, KFold
from sklearn.metrics         import make_scorer, accuracy_score, f1_score, r2_score
from sklearn.metrics         import mean_squared_error, mean_absolute_error, roc_auc_score

from core.model_factory   import get_model
from core.pipeline_builder import build_pipeline, infer_column_types
from core.param_grids      import get_param_grid, get_n_iter


# ── Scorer map ────────────────────────────────────────────────────────────────
def _get_scorer(metric: str):
    scorers = {
        "accuracy": make_scorer(accuracy_score),
        "f1":       make_scorer(f1_score, average="weighted", zero_division=0),
        "roc_auc":  make_scorer(roc_auc_score, needs_proba=True,
                                 multi_class="ovr", average="weighted"),
        "r2":       make_scorer(r2_score),
        "rmse":     make_scorer(mean_squared_error, greater_is_better=False,
                                 squared=False),
        "mae":      make_scorer(mean_absolute_error, greater_is_better=False),
    }
    return scorers.get(metric, make_scorer(accuracy_score))


def _get_cv(problem_type: str, n_splits: int = 5, y=None):
    """Return StratifiedKFold for clf, KFold for regression."""
    if problem_type == "classification":
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    return KFold(n_splits=n_splits, shuffle=True, random_state=42)


def train_all_models(
    df:             pd.DataFrame,
    target_col:     str,
    decision:       dict,
    progress_cb=None,           # optional callback(model_name, status_str)
) -> tuple[dict, dict]:
    """
    Train all models chosen by the Decision Agent.

    Returns
    -------
    leaderboard    : {model_name: best_cv_score}
    fitted_pipelines : {model_name: fitted_sklearn_Pipeline}
    """
    problem_type   = decision["problem_type"]
    models_to_try  = decision["models_to_try"]
    metric         = decision["metric"]
    intensity      = decision["tuning_intensity"]

    X = df.drop(columns=[target_col])
    y = df[target_col]

    numeric_cols, categorical_cols = infer_column_types(df, target_col)

    scorer  = _get_scorer(metric)
    cv      = _get_cv(problem_type, n_splits=5, y=y)
    n_iter  = get_n_iter(intensity)

    leaderboard:       dict = {}
    fitted_pipelines:  dict = {}

    for model_name in models_to_try:
        t0 = time.time()

        if progress_cb:
            progress_cb(model_name, "training…")

        try:
            model    = get_model(model_name, problem_type)
            pipeline = build_pipeline(model, numeric_cols, categorical_cols)
            grid     = get_param_grid(model_name, intensity)

            if grid:
                search = RandomizedSearchCV(
                    estimator=pipeline,
                    param_distributions=grid,
                    n_iter=min(n_iter, _count_combinations(grid)),
                    scoring=scorer,
                    cv=cv,
                    refit=True,
                    random_state=42,
                    n_jobs=-1,
                    error_score="raise",
                )
                search.fit(X, y)
                best_pipeline = search.best_estimator_
                best_score    = search.best_score_
            else:
                # No grid (e.g. LinearRegression) → just cross-val fit
                from sklearn.model_selection import cross_val_score
                scores        = cross_val_score(pipeline, X, y, cv=cv, scoring=scorer)
                pipeline.fit(X, y)
                best_pipeline = pipeline
                best_score    = float(np.mean(scores))

            # Normalise RMSE/MAE (negative scorers) to positive for leaderboard display
            if metric in ("rmse", "mae"):
                best_score = abs(best_score)

            elapsed = time.time() - t0
            leaderboard[model_name]      = round(best_score, 5)
            fitted_pipelines[model_name] = best_pipeline

            if progress_cb:
                progress_cb(model_name, f"done — {metric}={best_score:.4f} ({elapsed:.1f}s)")

            print(f"[Trainer] {model_name}: {metric}={best_score:.4f} ({elapsed:.1f}s)")

        except Exception as exc:
            print(f"[Trainer] {model_name} failed: {exc}")
            if progress_cb:
                progress_cb(model_name, f"failed: {exc}")
            leaderboard[model_name] = -1.0   # mark as failed

    return leaderboard, fitted_pipelines


def _count_combinations(grid: dict) -> int:
    """Count total combinations in a param grid (product of all list lengths)."""
    total = 1
    for v in grid.values():
        total *= len(v)
    return total
