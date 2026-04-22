"""
Model Factory
-------------
Returns untrained model instances from a strict whitelist.
No arbitrary code execution — only whitelisted model classes.
"""

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree         import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble     import RandomForestClassifier, RandomForestRegressor
from xgboost              import XGBClassifier, XGBRegressor
from lightgbm             import LGBMClassifier, LGBMRegressor

_CLF = {
    "LogisticRegression": (LogisticRegression, {"max_iter": 1000, "random_state": 42}),
    "DecisionTree":       (DecisionTreeClassifier, {"random_state": 42}),
    "RandomForest":       (RandomForestClassifier, {"random_state": 42, "n_jobs": -1}),
    "XGB":                (XGBClassifier,  {"random_state": 42, "verbosity": 0,
                                            "eval_metric": "logloss", "n_jobs": -1}),
    "LightGBM":           (LGBMClassifier, {"random_state": 42, "verbose": -1, "n_jobs": -1}),
}

_REG = {
    "LinearRegression": (LinearRegression, {"n_jobs": -1}),
    "DecisionTree":     (DecisionTreeRegressor, {"random_state": 42}),
    "RandomForest":     (RandomForestRegressor, {"random_state": 42, "n_jobs": -1}),
    "XGB":              (XGBRegressor,  {"random_state": 42, "verbosity": 0, "n_jobs": -1}),
    "LightGBM":         (LGBMRegressor, {"random_state": 42, "verbose": -1, "n_jobs": -1}),
}


def get_model(model_name: str, problem_type: str):
    pool = _CLF if problem_type == "classification" else _REG
    if model_name not in pool:
        raise ValueError(f"Model '{model_name}' not in whitelist for {problem_type}. "
                         f"Available: {list(pool)}")
    cls, kwargs = pool[model_name]
    return cls(**kwargs)


def list_models(problem_type: str) -> list[str]:
    return list((_CLF if problem_type == "classification" else _REG).keys())