"""
Reflection Agent
----------------
Truly agentic: reviews ALL training results, reasons about what went
wrong, and decides the next move — approve or retune with adapted strategy.

Agentic properties:
  - Perceives: full leaderboard + 7-criterion ValidationReport per model
  - Reasons:   LLM diagnoses issues and forms a strategy
  - Acts:      approve (stop) or retune (continue with new strategy)
  - Reflects:  compares this round to ideal outcomes
  - Adapts:    changes model pool, intensity, and search space bias

Max retune rounds: 3 (prevents infinite loop).
"""

import json
from agents.llm_client import call_llm

MAX_RETUNE_ROUNDS = 3

MIN_THRESHOLDS = {
    "accuracy": 0.55, "f1": 0.50, "roc_auc": 0.55,
    "r2": 0.20, "rmse": None, "mae": None,
}

INTENSITY_LADDER = ["light", "medium", "deep"]


def _build_prompt(leaderboard, decision, validation_reports, retune_round) -> str:
    sorted_lb = dict(sorted(leaderboard.items(), key=lambda x: x[1], reverse=True))

    val_summary = {}
    for name, report in validation_reports.items():
        rd = report.as_dict()
        val_summary[name] = {
            "cv_score":        round(rd["cv_score"], 4),
            "train_score":     round(rd["train_score"], 4),
            "test_score":      round(rd["test_score"], 4),
            "overfit_gap":     round(rd["overfit_gap"], 4),
            "generalise_gap":  round(rd["generalise_gap"], 4),
            "cv_std":          round(rd["cv_std"], 4),
            "passed_all":      rd["passed_all"],
            "target_met":      rd["target_met"],
            "failed":          [c["name"] for c in rd["criteria"] if not c["passed"]],
        }

    any_passed = any(v["passed_all"] for v in val_summary.values())
    best       = max(leaderboard, key=lambda k: leaderboard[k])
    remaining  = MAX_RETUNE_ROUNDS - retune_round

    return f"""<s>[INST]
You are a senior ML engineer reviewing AutoML training results. Diagnose issues and decide the next step.

LEADERBOARD (sorted by score):
{json.dumps(sorted_lb, indent=2)}

VALIDATION DIAGNOSTICS PER MODEL:
{json.dumps(val_summary, indent=2)}

TRAINING CONFIG USED:
{json.dumps({k: v for k, v in decision.items() if k not in ['strategy_hint', 'llm_used']}, indent=2)}

STATUS:
- Round: {retune_round + 1} of maximum {MAX_RETUNE_ROUNDS + 1}
- Remaining retune rounds: {remaining}
- Any model passed all criteria: {any_passed}
- Best model by score: {best} ({leaderboard[best]:.4f})

DECISION GUIDE:
- overfitting = overfit_gap > 0.10 (train >> cv)
- underfitting = cv_score < {MIN_THRESHOLDS.get(decision.get('metric','accuracy'), 0.55)}
- poor_generalisation = generalise_gap > 0.08 (cv >> test)  
- target_not_met = target_met=false for all models
- unstable_cv = cv_std > 0.06

RULES:
1. If any model passed_all=true → retune=false, pick that model
2. If no model passed AND remaining > 0 → retune=true
3. If remaining = 0 → ALWAYS retune=false
4. issues: list ALL problems found

RESPOND WITH ONLY THIS JSON:
{{
  "final_model": "{best}",
  "retune": false,
  "issues": ["none"],
  "reasoning": "Concise explanation of your decision"
}}
[/INST]"""


def _rule_based_reflection(
    leaderboard: dict,
    validation_reports: dict,
    decision: dict,
    retune_round: int,
) -> dict:
    """Full rule-based reflection when LLM unavailable."""
    print(f"[ReflectionAgent] ⚠ LLM unavailable — rule-based reflection")

    metric     = decision.get("metric", "accuracy")
    threshold  = MIN_THRESHOLDS.get(metric)
    any_passed = any(r.passed_all for r in validation_reports.values())

    # Diagnose all issues across all models
    issues = set()
    for name, report in validation_reports.items():
        rd = report.as_dict()
        if rd["overfit_gap"] > 0.10:
            issues.add("overfitting")
        if threshold and rd["cv_score"] < threshold:
            issues.add("underfitting")
        if rd["generalise_gap"] > 0.08:
            issues.add("poor_generalisation")
        if not rd["target_met"]:
            issues.add("target_not_met")
        if rd["cv_std"] > 0.06:
            issues.add("unstable_cv")

    issues = list(issues) or ["none"]

    # Pick best passing model, or best by score
    passing = [n for n, r in validation_reports.items() if r.passed_all]
    final   = (max(passing, key=lambda n: leaderboard.get(n, 0))
               if passing else max(leaderboard, key=lambda k: leaderboard[k]))

    should_retune = (
        not any_passed
        and retune_round < MAX_RETUNE_ROUNDS
        and "none" not in issues
    )

    reasoning = (
        f"{final} selected. "
        + (f"Retuning — issues: {', '.join(issues)}." if should_retune
           else f"Approved — {'all criteria met' if any_passed else 'max rounds reached or no actionable issues'}.")
    )

    return {
        "final_model": final,
        "retune":      should_retune,
        "issues":      issues,
        "reasoning":   reasoning,
        "llm_used":    False,
    }


def _validate(reflection: dict, leaderboard: dict, retune_round: int) -> dict:
    if reflection.get("final_model") not in leaderboard:
        reflection["final_model"] = max(leaderboard, key=lambda k: leaderboard[k])

    retune = reflection.get("retune", False)
    if not isinstance(retune, bool):
        retune = str(retune).lower() == "true"
    if retune_round >= MAX_RETUNE_ROUNDS:
        retune = False
    reflection["retune"] = retune

    if not isinstance(reflection.get("issues"), list):
        reflection["issues"] = ["none"]

    if not reflection.get("reasoning"):
        reflection["reasoning"] = "Selected by score."

    reflection["llm_used"] = True
    return reflection


def _adapt_strategy(decision: dict, issues: list, retune_round: int) -> dict:
    """
    Build adapted decision for next round based on diagnosed issues.
    This is where the agent shows true adaptability.
    """
    new      = decision.copy()
    new["models_to_try"] = list(decision.get("models_to_try", []))
    cur_idx  = INTENSITY_LADDER.index(new.get("tuning_intensity", "medium"))
    pt       = decision.get("problem_type", "classification")

    print(f"\n[ReflectionAgent] Adapting strategy for round {retune_round + 2}:")
    print(f"  Detected issues : {issues}")
    print(f"  Current models  : {new['models_to_try']}")
    print(f"  Current intensity: {new['tuning_intensity']}")

    if "overfitting" in issues:
        # Remove most complex, push regularisation harder
        to_remove = next(
            (m for m in ["LightGBM", "XGB"] if m in new["models_to_try"]
             and len(new["models_to_try"]) > 2), None
        )
        if to_remove:
            new["models_to_try"].remove(to_remove)
            print(f"  Removed '{to_remove}' — too complex for this data")
        new["tuning_intensity"] = INTENSITY_LADDER[max(0, cur_idx - 1)]
        new["strategy_hint"] = "overfit"

    elif "underfitting" in issues:
        # Add powerful models, escalate search
        for m in (["XGB", "LightGBM"] if pt == "classification" else ["XGB", "LightGBM"]):
            if m not in new["models_to_try"]:
                new["models_to_try"].append(m)
        new["tuning_intensity"] = INTENSITY_LADDER[min(2, cur_idx + 1)]
        new["strategy_hint"] = "underfit"

    elif "poor_generalisation" in issues:
        # Simplify — remove most complex model
        to_remove = next(
            (m for m in ["XGB", "LightGBM"] if m in new["models_to_try"]
             and len(new["models_to_try"]) > 2), None
        )
        if to_remove:
            new["models_to_try"].remove(to_remove)
        new["tuning_intensity"] = INTENSITY_LADDER[max(0, cur_idx - 1)]
        new["strategy_hint"] = "generalisation"

    elif "target_not_met" in issues:
        # Escalate search budget
        new["tuning_intensity"] = INTENSITY_LADDER[min(2, cur_idx + 1)]
        new["strategy_hint"] = "target"

    elif "unstable_cv" in issues:
        # Use simpler models that are more stable
        new["models_to_try"] = (
            ["LogisticRegression", "RandomForest", "XGB"]
            if pt == "classification"
            else ["LinearRegression", "RandomForest", "XGB"]
        )
        new["strategy_hint"] = "stability"

    else:
        new["tuning_intensity"] = INTENSITY_LADDER[min(2, cur_idx + 1)]
        new["strategy_hint"] = "escalate"

    # Ensure at least 2 models
    if len(new["models_to_try"]) < 2:
        new["models_to_try"] = decision["models_to_try"]

    print(f"  Next models     : {new['models_to_try']}")
    print(f"  Next intensity  : {new['tuning_intensity']}")
    print(f"  Strategy hint   : {new['strategy_hint']}\n")
    return new


def _build_warnings(leaderboard, decision, validation_reports) -> list[str]:
    warnings = []
    metric    = decision.get("metric", "accuracy")
    threshold = MIN_THRESHOLDS.get(metric)
    best      = max(leaderboard.values(), default=0)

    if threshold and best < threshold:
        warnings.append(
            f"Best {metric} ({best:.3f}) is below minimum ({threshold}). "
            f"Consider: more data, feature engineering, or a different problem framing."
        )
    if best > 0.999:
        warnings.append(
            "Score >0.999 is suspiciously perfect. "
            "Check that the target column isn't leaking into features."
        )
    for name, report in validation_reports.items():
        rd = report.as_dict()
        if rd["overfit_gap"] > 0.10:
            warnings.append(
                f"{name}: Overfitting detected "
                f"(train={rd['train_score']:.3f}, cv={rd['cv_score']:.3f}, "
                f"gap={rd['overfit_gap']:.3f})."
            )
        if rd["generalise_gap"] > 0.08:
            warnings.append(
                f"{name}: Poor generalisation "
                f"(cv={rd['cv_score']:.3f}, test={rd['test_score']:.3f}, "
                f"gap={rd['generalise_gap']:.3f})."
            )
    return warnings


def run_reflection_agent(
    leaderboard:        dict,
    decision:           dict,
    validation_reports: dict,
    retune_round:       int = 0,
) -> tuple[dict, list[str], dict]:
    """
    Main entry — truly agentic review and strategy adaptation.

    Returns
    -------
    reflection    : {final_model, retune, issues, reasoning, llm_used}
    warnings      : list of human-readable warning strings
    next_decision : adapted strategy for next round (if retune=True)
    """
    print(f"\n{'─'*64}")
    print(f"[ReflectionAgent] Round {retune_round + 1}/{MAX_RETUNE_ROUNDS + 1} review")
    print(f"  Leaderboard : {leaderboard}")
    print(f"  Fully passed: {[n for n, r in validation_reports.items() if r.passed_all]}")

    prompt = _build_prompt(leaderboard, decision, validation_reports, retune_round)
    raw    = call_llm(prompt, max_new_tokens=250, agent_name="ReflectionAgent")

    if raw is not None:
        print(f"[ReflectionAgent] ✓ LLM responded — validating")
        reflection = _validate(raw, leaderboard, retune_round)
    else:
        reflection = _rule_based_reflection(
            leaderboard, validation_reports, decision, retune_round
        )

    warnings      = _build_warnings(leaderboard, decision, validation_reports)
    next_decision = decision.copy()

    if reflection["retune"]:
        issues        = reflection.get("issues", ["escalate"])
        next_decision = _adapt_strategy(decision, issues, retune_round)
        reflection["strategy_applied"] = next_decision.get("strategy_hint", "escalate")

    print(f"\n[ReflectionAgent] DECISION (llm={reflection.get('llm_used', False)}):")
    print(f"  final_model : {reflection['final_model']}")
    print(f"  retune      : {reflection['retune']}")
    print(f"  issues      : {reflection.get('issues', [])}")
    print(f"  reasoning   : {reflection['reasoning']}")
    print(f"{'─'*64}\n")

    return reflection, warnings, next_decision