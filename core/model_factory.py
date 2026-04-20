"""
Model Factory
-------------
Returns the correct untrained model instance given a name + problem type.
All models are from a strict whitelist — no arbitrary instantiation.
"""

from sklearn.linear_model   import LogisticRegression, LinearRegression
from sklearn.tree           import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble       import RandomForestClassifier, RandomForestRegressor
from xgboost                import XGBClassifier, XGBRegressor
from lightgbm               import LGBMClassifier, LGBMRegressor

# ── Whitelist ────────────────────────────────────────────────────────────────
_CLF_MAP = {
    "LogisticRegression": LogisticRegression,
    "DecisionTree":       DecisionTreeClassifier,
    "RandomForest":       RandomForestClassifier,
    "XGB":                XGBClassifier,
    "LightGBM":           LGBMClassifier,
}

_REG_MAP = {
    "LinearRegression": LinearRegression,
    "DecisionTree":     DecisionTreeRegressor,
    "RandomForest":     RandomForestRegressor,
    "XGB":              XGBRegressor,
    "LightGBM":         LGBMRegressor,
}

# ── Safe default kwargs ──────────────────────────────────────────────────────
_DEFAULT_KWARGS = {
    "LogisticRegression": {"max_iter": 1000, "random_state": 42},
    "DecisionTree":       {"random_state": 42},
    "RandomForest":       {"random_state": 42, "n_jobs": -1},
    "XGB":                {
        "random_state": 42,
        "verbosity": 0,
        "eval_metric": "logloss",   # silences XGB warnings
        "n_jobs": -1,
    },
    "LightGBM":           {"random_state": 42, "verbose": -1, "n_jobs": -1},
    "LinearRegression":   {"n_jobs": -1},
}


def get_model(model_name: str, problem_type: str):
    """
    Returns an untrained model instance.

    Parameters
    ----------
    model_name   : one of the whitelisted names
    problem_type : "classification" or "regression"

    Raises
    ------
    ValueError if model_name is not in the whitelist for the given problem type.
    """
    problem_type = problem_type.lower()
    if problem_type not in ("classification", "regression"):
        raise ValueError(f"Unknown problem_type: '{problem_type}'")

    model_map = _CLF_MAP if problem_type == "classification" else _REG_MAP

    if model_name not in model_map:
        available = list(model_map.keys())
        raise ValueError(
            f"Model '{model_name}' not available for {problem_type}. "
            f"Available: {available}"
        )

    cls    = model_map[model_name]
    kwargs = _DEFAULT_KWARGS.get(model_name, {}).copy()

    # XGB regression doesn't use eval_metric="logloss"
    if model_name == "XGB" and problem_type == "regression":
        kwargs.pop("eval_metric", None)

    return cls(**kwargs)


def list_models(problem_type: str) -> list[str]:
    """Return all whitelisted model names for a problem type."""
    if problem_type == "classification":
        return list(_CLF_MAP.keys())
    return list(_REG_MAP.keys())
