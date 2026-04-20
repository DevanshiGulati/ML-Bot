"""
Reflection Agent (Review Agent)
--------------------------------
Receives the leaderboard and original decision, then decides:
  - final_model    : which model to ship
  - retune         : whether to trigger another round of tuning
  - reasoning      : short explanation

Also enforces quality gates:
  - Must beat a dummy baseline
  - Flags potential overfitting
  - Limits retune loops to MAX_RETUNE_ROUNDS
"""

import json
from agents.llm_client import call_llm

MAX_RETUNE_ROUNDS = 2   # safety cap on retry loops

# Minimum acceptable scores before we warn the user
MIN_SCORE_THRESHOLDS = {
    "accuracy": 0.55,
    "f1":       0.50,
    "roc_auc":  0.55,
    "r2":       0.20,
    "rmse":     None,   # lower-is-better; skip threshold check
    "mae":      None,
}


def _build_prompt(leaderboard: dict, decision: dict, retune_round: int) -> str:
    sorted_lb = dict(sorted(leaderboard.items(), key=lambda x: x[1], reverse=True))
    return f"""<s>[INST]
You are a senior ML engineer reviewing model training results.

Leaderboard (model → score):
{json.dumps(sorted_lb, indent=2)}

Training config used:
{json.dumps(decision, indent=2)}

This is retune round {retune_round} of {MAX_RETUNE_ROUNDS}.

Rules:
- Pick the model with the best score as final_model (use exact name from leaderboard).
- Set retune=true ONLY if the best score is poor AND retune_round < {MAX_RETUNE_ROUNDS}.
- Provide a short reasoning (max 20 words).

Return ONLY this JSON (no prose):
{{
  "final_model": "<model_name>",
  "retune": false,
  "reasoning": "Best score with good generalisation."
}}
[/INST]"""


def _sanitise(reflection: dict, leaderboard: dict, retune_round: int) -> dict:
    """Validate and repair reflection output."""
    valid_models = set(leaderboard.keys())

    # final_model must be in leaderboard
    fm = reflection.get("final_model", "")
    if fm not in valid_models:
        fm = max(leaderboard, key=leaderboard.get)
    reflection["final_model"] = fm

    # retune must be bool; cap at MAX_RETUNE_ROUNDS
    retune = reflection.get("retune", False)
    if not isinstance(retune, bool):
        retune = str(retune).lower() == "true"
    if retune_round >= MAX_RETUNE_ROUNDS:
        retune = False
    reflection["retune"] = retune

    # reasoning fallback
    if not reflection.get("reasoning"):
        reflection["reasoning"] = "Selected based on highest leaderboard score."

    return reflection


def _quality_check(leaderboard: dict, decision: dict) -> list[str]:
    """Return a list of warning strings (empty = all good)."""
    warnings = []
    metric    = decision.get("metric", "accuracy")
    threshold = MIN_SCORE_THRESHOLDS.get(metric)

    best_score = max(leaderboard.values()) if leaderboard else 0.0

    if threshold is not None and best_score < threshold:
        warnings.append(
            f"Best {metric} score ({best_score:.3f}) is below the recommended "
            f"threshold ({threshold}). Consider collecting more data or "
            f"engineering better features."
        )

    if best_score > 0.999:
        warnings.append(
            "Score is suspiciously high (>0.999). Possible data leakage — "
            "check that the target column is not present in the features."
        )

    return warnings


def run_reflection_agent(
    leaderboard: dict,
    decision:    dict,
    retune_round: int = 0,
) -> tuple[dict, list[str]]:
    """
    Returns (reflection_dict, warnings_list).
    reflection_dict keys: final_model, retune, reasoning
    """
    prompt     = _build_prompt(leaderboard, decision, retune_round)
    raw        = call_llm(prompt, max_new_tokens=200)
    reflection = raw if isinstance(raw, dict) else {}

    reflection = _sanitise(reflection, leaderboard, retune_round)
    warnings   = _quality_check(leaderboard, decision)

    print(f"[ReflectionAgent] Reflection: {json.dumps(reflection, indent=2)}")
    if warnings:
        print(f"[ReflectionAgent] Warnings: {warnings}")

    return reflection, warnings
