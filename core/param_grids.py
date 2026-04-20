"""
Param Grids
-----------
Hyperparameter search spaces for RandomizedSearchCV.
Each model has three intensity tiers: light / medium / deep.
"""

import numpy as np

# ── n_iter per intensity ─────────────────────────────────────────────────────
N_ITER = {
    "light":  10,
    "medium": 30,
    "deep":   60,
}

# ── Per-model grids ───────────────────────────────────────────────────────────
# Prefix all param names with "model__" so they work inside an sklearn Pipeline.

_GRIDS = {

    "LogisticRegression": {
        "light": {
            "model__C":        [0.01, 0.1, 1, 10],
            "model__solver":   ["lbfgs", "liblinear"],
        },
        "medium": {
            "model__C":        np.logspace(-3, 3, 12).tolist(),
            "model__solver":   ["lbfgs", "liblinear", "saga"],
            "model__penalty":  ["l2", "l1"],
        },
        "deep": {
            "model__C":        np.logspace(-4, 4, 20).tolist(),
            "model__solver":   ["lbfgs", "liblinear", "saga"],
            "model__penalty":  ["l2", "l1", "elasticnet"],
            "model__l1_ratio": np.linspace(0, 1, 10).tolist(),
        },
    },

    "DecisionTree": {
        "light": {
            "model__max_depth":        [3, 5, 10, None],
            "model__min_samples_split": [2, 5, 10],
        },
        "medium": {
            "model__max_depth":         [3, 5, 10, 15, None],
            "model__min_samples_split": [2, 5, 10, 20],
            "model__min_samples_leaf":  [1, 2, 4],
            "model__criterion":         ["gini", "entropy"],
        },
        "deep": {
            "model__max_depth":         [3, 5, 10, 15, 20, None],
            "model__min_samples_split": [2, 5, 10, 20, 50],
            "model__min_samples_leaf":  [1, 2, 4, 8],
            "model__criterion":         ["gini", "entropy"],
            "model__max_features":      ["sqrt", "log2", None],
        },
    },

    "RandomForest": {
        "light": {
            "model__n_estimators": [50, 100],
            "model__max_depth":    [5, 10, None],
        },
        "medium": {
            "model__n_estimators":      [50, 100, 200],
            "model__max_depth":         [5, 10, 15, None],
            "model__min_samples_split": [2, 5, 10],
            "model__max_features":      ["sqrt", "log2"],
        },
        "deep": {
            "model__n_estimators":      [100, 200, 400],
            "model__max_depth":         [5, 10, 15, 20, None],
            "model__min_samples_split": [2, 5, 10, 20],
            "model__min_samples_leaf":  [1, 2, 4],
            "model__max_features":      ["sqrt", "log2", 0.3, 0.5],
            "model__bootstrap":         [True, False],
        },
    },

    "XGB": {
        "light": {
            "model__n_estimators": [100, 200],
            "model__max_depth":    [3, 5, 7],
            "model__learning_rate": [0.05, 0.1, 0.2],
        },
        "medium": {
            "model__n_estimators":  [100, 200, 300],
            "model__max_depth":     [3, 5, 7, 9],
            "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "model__subsample":     [0.7, 0.8, 1.0],
            "model__colsample_bytree": [0.7, 0.8, 1.0],
        },
        "deep": {
            "model__n_estimators":     [100, 200, 400, 600],
            "model__max_depth":        [3, 5, 7, 9, 11],
            "model__learning_rate":    [0.005, 0.01, 0.05, 0.1, 0.2],
            "model__subsample":        [0.6, 0.7, 0.8, 0.9, 1.0],
            "model__colsample_bytree": [0.5, 0.6, 0.7, 0.8, 1.0],
            "model__reg_alpha":        [0, 0.01, 0.1, 1],
            "model__reg_lambda":       [0.1, 1, 5, 10],
            "model__gamma":            [0, 0.1, 0.5, 1],
        },
    },

    "LightGBM": {
        "light": {
            "model__n_estimators":  [100, 200],
            "model__max_depth":     [3, 5, 7],
            "model__learning_rate": [0.05, 0.1, 0.2],
        },
        "medium": {
            "model__n_estimators":  [100, 200, 300],
            "model__max_depth":     [3, 5, 7, -1],
            "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "model__num_leaves":    [31, 63, 127],
            "model__subsample":     [0.7, 0.8, 1.0],
        },
        "deep": {
            "model__n_estimators":   [100, 200, 400, 600],
            "model__max_depth":      [3, 5, 7, 9, -1],
            "model__learning_rate":  [0.005, 0.01, 0.05, 0.1, 0.2],
            "model__num_leaves":     [31, 63, 127, 255],
            "model__subsample":      [0.6, 0.7, 0.8, 0.9, 1.0],
            "model__colsample_bytree": [0.5, 0.7, 0.8, 1.0],
            "model__reg_alpha":      [0, 0.01, 0.1, 1],
            "model__reg_lambda":     [0, 0.01, 0.1, 1],
        },
    },

    "LinearRegression": {
        "light":  {"model__fit_intercept": [True, False]},
        "medium": {"model__fit_intercept": [True, False]},
        "deep":   {"model__fit_intercept": [True, False]},
    },
}


def get_param_grid(model_name: str, intensity: str) -> dict:
    """Return param grid for a model at a given tuning intensity."""
    intensity = intensity.lower()
    if intensity not in N_ITER:
        intensity = "medium"

    grid = _GRIDS.get(model_name, {}).get(intensity, {})
    return grid


def get_n_iter(intensity: str) -> int:
    return N_ITER.get(intensity.lower(), 30)
