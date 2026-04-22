"""
Model Validator
---------------
Runs 7 quality checks on every trained pipeline.

FIX: Generalisation gap is now abs(cv_score - test_score).
     Negative gaps (test > cv) are GOOD, not bad — they just
     mean the model generalises slightly better than CV estimated.
     We only flag when the gap is LARGE AND POSITIVE (cv >> test).

The 7 checks:
  1. Target score        — user's required minimum metric value
  2. No overfitting      — train score significantly above CV score
  3. No underfitting     — CV score above a meaningful floor
  4. Good generalisation — CV score NOT much larger than test score
  5. Stable CV folds     — std deviation across 5 folds acceptable
  6. Beats dummy         — beats DummyClassifier / DummyRegressor
  7. No data leakage     — train score not suspiciously perfect
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, KFold, train_test_split
)
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import (
    make_scorer, accuracy_score, f1_score, r2_score,
    mean_squared_error, mean_absolute_error, roc_auc_score,
)

MIN_ACCEPTABLE = {
    "accuracy": 0.60, "f1": 0.55, "roc_auc": 0.60,
    "r2": 0.30, "rmse": None, "mae": None,
}

OVERFIT_GAP_THRESHOLD    = 0.10   # train_score - cv_score
GENERALISE_GAP_THRESHOLD = 0.08   # only flagged when cv_score >> test_score
MAX_CV_STD               = 0.06
LEAKAGE_THRESHOLD        = 0.999


@dataclass
class CriterionResult:
    name:    str
    passed:  bool
    value:   float | None
    message: str


@dataclass
class ValidationReport:
    model_name:      str
    metric:          str
    cv_score:        float
    cv_std:          float
    train_score:     float
    test_score:      float
    baseline_score:  float
    overfit_gap:     float = 0.0
    generalise_gap:  float = 0.0    # signed: positive = cv > test (bad), negative = test > cv (fine)
    criteria:        list[CriterionResult] = field(default_factory=list)
    passed_all:      bool  = False
    target_met:      bool  = False
    needs_retune:    bool  = False
    summary:         str   = ""

    def as_dict(self) -> dict:
        return {
            "model_name":     self.model_name,
            "metric":         self.metric,
            "cv_score":       round(self.cv_score, 4),
            "cv_std":         round(self.cv_std, 4),
            "train_score":    round(self.train_score, 4),
            "test_score":     round(self.test_score, 4),
            "baseline_score": round(self.baseline_score, 4),
            "overfit_gap":    round(self.overfit_gap, 4),
            "generalise_gap": round(self.generalise_gap, 4),
            "passed_all":     self.passed_all,
            "target_met":     self.target_met,
            "needs_retune":   self.needs_retune,
            "summary":        self.summary,
            "criteria": [
                {"name": c.name, "passed": c.passed,
                 "value": round(c.value, 4) if c.value is not None else None,
                 "message": c.message}
                for c in self.criteria
            ],
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


def _higher_is_better(metric: str) -> bool:
    return metric not in ("rmse", "mae")


def _get_cv(problem_type: str):
    if problem_type == "classification":
        return StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    return KFold(n_splits=5, shuffle=True, random_state=42)


def validate_model(
    pipeline,
    df:           pd.DataFrame,
    target_col:   str,
    problem_type: str,
    metric:       str,
    model_name:   str,
    user_criteria: dict | None = None,
) -> ValidationReport:
    uc              = user_criteria or {}
    min_score       = uc.get("min_score")
    max_overfit_gap = uc.get("max_overfit_gap", OVERFIT_GAP_THRESHOLD)
    max_gen_gap     = uc.get("max_gen_gap",     GENERALISE_GAP_THRESHOLD)
    max_cv_std      = uc.get("max_cv_std",      MAX_CV_STD)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42,
        stratify=(y if problem_type == "classification" else None),
    )

    scorer = _get_scorer(metric)
    cv     = _get_cv(problem_type)
    hib    = _higher_is_better(metric)

    # ── CV scores (5-fold) ────────────────────────────────────────────────
    cv_raw    = cross_val_score(pipeline, X_train, y_train,
                                cv=cv, scoring=scorer, n_jobs=-1)
    cv_scores = np.abs(cv_raw) if not hib else cv_raw
    cv_mean   = float(np.mean(cv_scores))
    cv_std    = float(np.std(cv_scores))

    # ── Train score ───────────────────────────────────────────────────────
    pipeline.fit(X_train, y_train)
    train_score = abs(float(scorer(pipeline, X_train, y_train)))

    # ── Test score (held-out) ─────────────────────────────────────────────
    test_score = abs(float(scorer(pipeline, X_test, y_test)))

    # ── Baseline ──────────────────────────────────────────────────────────
    dummy = (DummyClassifier(strategy="most_frequent", random_state=42)
             if problem_type == "classification"
             else DummyRegressor(strategy="mean"))
    dummy.fit(X_train, y_train)
    baseline = abs(float(scorer(dummy, X_test, y_test)))

    # ── Gaps ──────────────────────────────────────────────────────────────
    # Overfitting: always train_score - cv_mean (positive = train >> cv)
    if hib:
        overfit_gap    = train_score - cv_mean
        # Generalise gap: cv_mean - test_score
        # Positive = cv was optimistic vs test (concerning)
        # Negative = test >= cv (the model generalises well or even better, totally fine)
        generalise_gap = cv_mean - test_score
    else:
        # For RMSE/MAE lower is better. "overfit" = train error much lower than cv error
        overfit_gap    = cv_mean - train_score
        generalise_gap = test_score - cv_mean   # positive = test worse than cv (concerning)

    # ── 7 Criteria ────────────────────────────────────────────────────────
    criteria: list[CriterionResult] = []

    # 1. Target score
    if min_score is not None:
        passed = cv_mean >= min_score if hib else cv_mean <= min_score
        criteria.append(CriterionResult(
            "Target score", passed, cv_mean,
            f"CV {metric}={cv_mean:.4f} {'≥' if hib else '≤'} target {min_score}"
            if passed else
            f"CV {metric}={cv_mean:.4f} does NOT meet target {min_score}",
        ))

    # 2. No overfitting (only flag when gap is meaningfully large)
    of_ok = overfit_gap <= max_overfit_gap
    criteria.append(CriterionResult(
        "No overfitting", of_ok, overfit_gap,
        (f"Train-CV gap={overfit_gap:.4f} acceptable (≤{max_overfit_gap})"
         if of_ok else
         f"Overfitting: train-CV gap={overfit_gap:.4f} > {max_overfit_gap}"),
    ))

    # 3. No underfitting
    min_acc = MIN_ACCEPTABLE.get(metric)
    if min_acc is not None:
        uf_ok = cv_mean >= min_acc
        criteria.append(CriterionResult(
            "No underfitting", uf_ok, cv_mean,
            (f"CV {metric}={cv_mean:.4f} above floor ({min_acc})"
             if uf_ok else
             f"Underfitting: CV {metric}={cv_mean:.4f} below floor ({min_acc})"),
        ))

    # 4. Good generalisation
    # Only fail when generalise_gap is POSITIVE and large (cv >> test).
    # Negative gap (test > cv) is fine — don't penalise it.
    gen_ok = generalise_gap <= max_gen_gap
    criteria.append(CriterionResult(
        "Good generalisation", gen_ok, generalise_gap,
        (f"CV-test gap={generalise_gap:.4f} OK (≤{max_gen_gap})"
         + (" [test>cv: model generalises well]" if generalise_gap < 0 else "")
         if gen_ok else
         f"Poor generalisation: CV score much higher than test score (gap={generalise_gap:.4f})"),
    ))

    # 5. Stable CV
    std_ok = cv_std <= max_cv_std
    criteria.append(CriterionResult(
        "Stable CV folds", std_ok, cv_std,
        (f"CV std={cv_std:.4f} stable (≤{max_cv_std})"
         if std_ok else
         f"High variance across folds: std={cv_std:.4f} > {max_cv_std}"),
    ))

    # 6. Beats baseline
    if hib:
        base_ok = cv_mean > baseline + 0.02
        margin  = cv_mean - baseline
    else:
        base_ok = cv_mean < baseline - 0.02
        margin  = baseline - cv_mean
    criteria.append(CriterionResult(
        "Beats dummy baseline", base_ok, margin,
        (f"Beats dummy by {margin:.4f}" if base_ok
         else f"Critical: model barely beats or loses to a dummy predictor"),
    ))

    # 7. No leakage
    leak_ok = train_score < LEAKAGE_THRESHOLD
    criteria.append(CriterionResult(
        "No data leakage", leak_ok, train_score,
        ("Train score realistic" if leak_ok
         else f"CRITICAL: train={train_score:.4f} suspiciously perfect — check for leakage"),
    ))

    # ── Verdict ───────────────────────────────────────────────────────────
    passed_all   = all(c.passed for c in criteria)
    target_met   = next((c.passed for c in criteria if c.name == "Target score"), True)
    needs_retune = not passed_all and leak_ok

    failed = [c.name for c in criteria if not c.passed]

    if passed_all:
        summary = f"All criteria passed. {model_name} is well-fitted and generalises well."
    elif not base_ok:
        summary = f"Critical: {model_name} does not beat a dummy predictor."
    elif not of_ok:
        summary = (f"{model_name} is overfitting (train-CV gap={overfit_gap:.3f}). "
                   "Reflection agent will try stronger regularisation.")
    elif min_acc and not next((c.passed for c in criteria if c.name == "No underfitting"), True):
        summary = (f"{model_name} is underfitting (CV {metric}={cv_mean:.3f}). "
                   "Reflection agent will try more complex models.")
    elif not gen_ok:
        summary = (f"{model_name} generalises poorly (CV-test gap={generalise_gap:.3f}). "
                   "Reflection agent will reduce complexity.")
    elif not target_met:
        summary = (f"{model_name} CV {metric}={cv_mean:.4f} misses target {min_score}. "
                   "Reflection agent will escalate tuning intensity.")
    else:
        summary = f"{model_name} partial pass. Failed: {', '.join(failed)}."

    print(f"[Validator] {model_name}: cv={cv_mean:.4f}±{cv_std:.4f} "
          f"train={train_score:.4f} test={test_score:.4f} "
          f"overfit={overfit_gap:.4f} gen={generalise_gap:.4f} "
          f"passed={passed_all}")

    return ValidationReport(
        model_name=model_name, metric=metric,
        cv_score=cv_mean, cv_std=cv_std,
        train_score=train_score, test_score=test_score, baseline_score=baseline,
        overfit_gap=overfit_gap, generalise_gap=generalise_gap,
        criteria=criteria, passed_all=passed_all,
        target_met=target_met, needs_retune=needs_retune, summary=summary,
    )


def validate_all_models(
    fitted_pipelines: dict,
    df:               pd.DataFrame,
    target_col:       str,
    decision:         dict,
    user_criteria:    dict | None = None,
) -> dict[str, ValidationReport]:
    reports = {}
    for name, pipeline in fitted_pipelines.items():
        if pipeline is None:
            continue
        try:
            reports[name] = validate_model(
                pipeline, df, target_col,
                decision["problem_type"], decision["metric"],
                name, user_criteria,
            )
        except Exception as exc:
            print(f"[Validator] {name} failed: {exc}")
    return reports