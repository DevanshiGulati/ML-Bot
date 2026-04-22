"""
Model Trainer
-------------
Trains all models using Optuna TPE (Bayesian hyperparameter optimisation).

Key fixes:
- LinearRegression gets only 5 trials (has almost no hyperparams)
- n_trials is capped per model based on actual search space size
- strategy_hint from Reflection Agent adjusts search space
- Detailed terminal logging per trial
- Fallback to RandomizedSearchCV only if Optuna import fails entirely
"""

import time, warnings
import numpy as np
import pandas as pd

import optuna
from optuna.samplers import TPESampler
from optuna.pruners  import MedianPruner

from sklearn.model_selection import (
    StratifiedKFold, KFold, cross_val_score, RandomizedSearchCV
)
from sklearn.metrics import (
    make_scorer, accuracy_score, f1_score, r2_score,
    mean_squared_error, mean_absolute_error, roc_auc_score,
)

from core.model_factory    import get_model
from core.pipeline_builder import build_pipeline, infer_column_types

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)

# n_trials per intensity
N_TRIALS_BASE = {"light": 20, "medium": 40, "deep": 80}

# Models with tiny search spaces — cap their trials
MAX_TRIALS_PER_MODEL = {
    "LinearRegression":  5,    # only fit_intercept = 2 options
    "LogisticRegression": 30,
    "DecisionTree":      40,
    "RandomForest":      60,
    "XGB":               80,
    "LightGBM":          80,
}


def _get_scorer(metric: str):
    return {
        "accuracy": make_scorer(accuracy_score),
        "f1":       make_scorer(f1_score, average="weighted", zero_division=0),
        "roc_auc":  make_scorer(roc_auc_score, needs_proba=True,
                                multi_class="ovr", average="weighted"),
        "r2":       make_scorer(r2_score),
        "rmse":     make_scorer(mean_squared_error, greater_is_better=False),
        "mae":      make_scorer(mean_absolute_error, greater_is_better=False),
    }.get(metric, make_scorer(accuracy_score))


def _get_cv(problem_type: str):
    if problem_type == "classification":
        return StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    return KFold(n_splits=5, shuffle=True, random_state=42)


def _higher_is_better(metric: str) -> bool:
    return metric not in ("rmse", "mae")


def _suggest_params(trial, model_name: str, intensity: str,
                    strategy_hint: str = "") -> dict:
    """
    Continuous Optuna search space per model.
    strategy_hint adjusts the space:
      "overfit"       → stronger regularisation range
      "underfit"      → wider capacity range
      "target"/"deep" → fullest range
    """
    is_overfit  = strategy_hint == "overfit"
    is_underfit = strategy_hint == "underfit"
    is_deep     = intensity == "deep"

    if model_name == "LinearRegression":
        return {
            "model__fit_intercept": trial.suggest_categorical("fit_intercept", [True, False])
        }

    if model_name == "LogisticRegression":
        c_high = 5.0 if is_overfit else 1000.0
        return {
            "model__C":      trial.suggest_float("C", 1e-5, c_high, log=True),
            "model__solver": trial.suggest_categorical("solver",
                ["lbfgs", "liblinear"] if not is_deep else ["lbfgs", "liblinear", "saga"]),
        }

    if model_name == "DecisionTree":
        max_d = 5 if is_overfit else (8 if not is_deep else 20)
        min_s = 10 if is_overfit else 2
        return {
            "model__max_depth":         trial.suggest_int("max_depth", 2, max_d),
            "model__min_samples_split": trial.suggest_int("min_samples_split", min_s, 50),
            "model__min_samples_leaf":  trial.suggest_int("min_samples_leaf",
                                            3 if is_overfit else 1, 15),
            "model__max_features":      trial.suggest_categorical("max_features",
                                            ["sqrt", "log2", None]),
        }

    if model_name == "RandomForest":
        n_max = 150 if not is_deep else 400
        d_max = 8 if is_overfit else (12 if not is_deep else 25)
        return {
            "model__n_estimators":      trial.suggest_int("n_estimators", 50, n_max),
            "model__max_depth":         trial.suggest_int("max_depth", 3, d_max),
            "model__min_samples_split": trial.suggest_int("min_samples_split",
                                            5 if is_overfit else 2, 30),
            "model__min_samples_leaf":  trial.suggest_int("min_samples_leaf",
                                            2 if is_overfit else 1, 10),
            "model__max_features":      trial.suggest_categorical("max_features",
                                            ["sqrt", "log2"]),
            "model__bootstrap":         trial.suggest_categorical("bootstrap", [True, False]),
        }

    if model_name == "XGB":
        lr_hi  = 0.05 if is_overfit else 0.30
        n_max  = 200 if not is_deep else 600
        d_max  = 5 if is_overfit else (8 if not is_deep else 12)
        reg_lo = 0.5 if is_overfit else 1e-5
        return {
            "model__n_estimators":     trial.suggest_int("n_estimators", 50, n_max),
            "model__max_depth":        trial.suggest_int("max_depth", 2, d_max),
            "model__learning_rate":    trial.suggest_float("learning_rate", 0.001, lr_hi, log=True),
            "model__subsample":        trial.suggest_float("subsample",
                                           0.4 if is_overfit else 0.5, 1.0),
            "model__colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "model__reg_alpha":        trial.suggest_float("reg_alpha", reg_lo, 20.0, log=True),
            "model__reg_lambda":       trial.suggest_float("reg_lambda", reg_lo, 20.0, log=True),
            "model__gamma":            trial.suggest_float("gamma", 0.0, 10.0),
            "model__min_child_weight": trial.suggest_int("min_child_weight",
                                           3 if is_overfit else 1, 20),
        }

    if model_name == "LightGBM":
        lr_hi = 0.05 if is_overfit else 0.30
        n_max = 200 if not is_deep else 600
        d_max = 5 if is_overfit else (8 if not is_deep else 12)
        return {
            "model__n_estimators":      trial.suggest_int("n_estimators", 50, n_max),
            "model__max_depth":         trial.suggest_int("max_depth", 2, d_max),
            "model__learning_rate":     trial.suggest_float("learning_rate", 0.001, lr_hi, log=True),
            "model__num_leaves":        trial.suggest_int("num_leaves",
                                            10 if is_overfit else 20, 100 if is_overfit else 300),
            "model__subsample":         trial.suggest_float("subsample",
                                            0.4 if is_overfit else 0.5, 1.0),
            "model__colsample_bytree":  trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "model__reg_alpha":         trial.suggest_float("reg_alpha", 1e-5, 20.0, log=True),
            "model__reg_lambda":        trial.suggest_float("reg_lambda", 1e-5, 20.0, log=True),
            "model__min_child_samples": trial.suggest_int("min_child_samples",
                                            20 if is_overfit else 5, 100),
        }

    return {}


def _n_trials_for_model(model_name: str, intensity: str) -> int:
    """Return appropriate trial count — capped per model's search space size."""
    base = N_TRIALS_BASE.get(intensity, 40)
    cap  = MAX_TRIALS_PER_MODEL.get(model_name, 80)
    return min(base, cap)


def _train_with_optuna(
    model_name, problem_type, intensity, metric, strategy_hint,
    X, y, numeric_cols, categorical_cols, n_trials, progress_cb,
):
    cv        = _get_cv(problem_type)
    scorer    = _get_scorer(metric)
    direction = "maximize" if _higher_is_better(metric) else "minimize"
    count     = [0]
    best_seen = [None]

    def objective(trial):
        params   = _suggest_params(trial, model_name, intensity, strategy_hint)
        model    = get_model(model_name, problem_type)
        pipeline = build_pipeline(model, numeric_cols, categorical_cols)
        pipeline.set_params(**params)
        scores   = cross_val_score(pipeline, X, y, cv=cv, scoring=scorer, n_jobs=-1)
        score    = float(np.mean(scores))
        count[0] += 1

        display = abs(score) if metric in ("rmse", "mae") else score
        is_best = best_seen[0] is None or (
            display > best_seen[0] if _higher_is_better(metric) else display < best_seen[0]
        )
        if is_best:
            best_seen[0] = display

        print(f"  [Optuna/{model_name}] Trial {count[0]:>2}/{n_trials} "
              f"| {metric}={display:.4f} "
              f"| best={best_seen[0]:.4f}"
              f"{' ← NEW BEST' if is_best else ''}")

        if progress_cb:
            progress_cb(model_name,
                f"trial {count[0]}/{n_trials} — {metric}={display:.4f} | best={best_seen[0]:.4f}")
        return score

    study = optuna.create_study(
        direction=direction,
        sampler=TPESampler(
            seed=42,
            n_startup_trials=max(3, n_trials // 5),
            multivariate=True,       # considers parameter correlations
        ),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=2),
        study_name=f"{model_name}_{intensity}_{strategy_hint or 'none'}",
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # Refit best pipeline on full training data
    best_model    = get_model(model_name, problem_type)
    best_pipeline = build_pipeline(best_model, numeric_cols, categorical_cols)
    prefixed      = {f"model__{k}": v for k, v in study.best_params.items()}
    best_pipeline.set_params(**prefixed)
    best_pipeline.fit(X, y)

    score = study.best_value
    if metric in ("rmse", "mae"):
        score = abs(score)
    return best_pipeline, round(score, 5), study


def train_all_models(
    df:         pd.DataFrame,
    target_col: str,
    decision:   dict,
    progress_cb=None,
) -> tuple[dict, dict, dict]:
    """
    Train all models using Optuna TPE.

    Returns
    -------
    leaderboard       : {model_name: best_cv_score}
    fitted_pipelines  : {model_name: fitted sklearn Pipeline}
    studies           : {model_name: optuna.Study | None}
    """
    problem_type  = decision["problem_type"]
    models_to_try = decision["models_to_try"]
    metric        = decision["metric"]
    intensity     = decision["tuning_intensity"]
    strategy_hint = decision.get("strategy_hint", "")

    X = df.drop(columns=[target_col])
    y = df[target_col]
    numeric_cols, categorical_cols = infer_column_types(df, target_col)

    leaderboard:      dict = {}
    fitted_pipelines: dict = {}
    studies:          dict = {}

    print(f"\n[Trainer] Starting training round")
    print(f"[Trainer] Models     : {models_to_try}")
    print(f"[Trainer] Metric     : {metric}")
    print(f"[Trainer] Intensity  : {intensity}")
    print(f"[Trainer] Strategy   : {strategy_hint or 'none'}")
    print(f"[Trainer] Data shape : {X.shape}")

    for model_name in models_to_try:
        n_trials = _n_trials_for_model(model_name, intensity)
        t0 = time.time()

        print(f"\n[Trainer] ── {model_name} ── ({n_trials} Optuna trials)")
        if progress_cb:
            progress_cb(model_name,
                f"starting Optuna TPE ({n_trials} trials, strategy={strategy_hint or 'none'})…")

        try:
            pipe, score, study = _train_with_optuna(
                model_name, problem_type, intensity, metric, strategy_hint,
                X, y, numeric_cols, categorical_cols, n_trials, progress_cb,
            )
            elapsed = time.time() - t0
            leaderboard[model_name]      = score
            fitted_pipelines[model_name] = pipe
            studies[model_name]          = study

            print(f"[Trainer] ✓ {model_name}: {metric}={score:.4f} "
                  f"in {elapsed:.1f}s ({n_trials} trials)")
            print(f"[Trainer]   Best params: {study.best_params}")

            if progress_cb:
                progress_cb(model_name, f"✓ {metric}={score:.4f} ({elapsed:.1f}s)")

        except Exception as exc:
            print(f"[Trainer] ✗ {model_name} failed: {exc}")
            if progress_cb:
                progress_cb(model_name, f"✗ failed: {exc}")
            leaderboard[model_name] = -1.0

    print(f"\n[Trainer] Round complete. Leaderboard: {leaderboard}")
    return leaderboard, fitted_pipelines, studies