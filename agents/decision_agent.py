"""
Decision Agent
--------------
Receives dataset metadata and uses the LLM to decide:
  - problem_type  : classification | regression
  - models_to_try : list of model names
  - metric        : evaluation metric
  - tuning_intensity : light | medium | deep
"""

import json
from agents.llm_client import call_llm

# ── Whitelists ──────────────────────────────────────────────────────────────
VALID_PROBLEM_TYPES   = {"classification", "regression"}
VALID_MODELS_CLF      = {"LogisticRegression", "DecisionTree", "RandomForest", "XGB", "LightGBM"}
VALID_MODELS_REG      = {"LinearRegression",  "DecisionTree", "RandomForest", "XGB", "LightGBM"}
VALID_METRICS_CLF     = {"accuracy", "f1", "roc_auc"}
VALID_METRICS_REG     = {"r2", "rmse", "mae"}
VALID_INTENSITIES     = {"light", "medium", "deep"}


def _build_prompt(metadata: dict) -> str:
    return f"""<s>[INST]
You are an expert ML engineer. Analyse the dataset metadata below and return ONLY a valid JSON object — no explanation, no markdown.

Dataset metadata:
{json.dumps(metadata, indent=2)}

Rules:
- problem_type: "classification" if target has ≤20 unique values and is object/int, else "regression".
- models_to_try: choose 3-4 appropriate models from the allowed list.
  - Classification allowed: LogisticRegression, DecisionTree, RandomForest, XGB, LightGBM
  - Regression allowed: LinearRegression, DecisionTree, RandomForest, XGB, LightGBM
- metric: accuracy/f1/roc_auc for classification, r2/rmse/mae for regression.
- tuning_intensity: "light" if rows<500, "medium" if rows<5000, "deep" otherwise.

Return ONLY this JSON (no prose):
{{
  "problem_type": "classification",
  "models_to_try": ["LogisticRegression", "RandomForest", "XGB"],
  "metric": "accuracy",
  "tuning_intensity": "medium"
}}
[/INST]"""


def _sanitise(decision: dict, metadata: dict) -> dict:
    """Validate and repair LLM output. Fill in safe defaults if needed."""
    # problem_type
    pt = decision.get("problem_type", "").lower()
    if pt not in VALID_PROBLEM_TYPES:
        n_unique = metadata.get("target_unique_values", 2)
        pt = "classification" if n_unique <= 20 else "regression"
    decision["problem_type"] = pt

    # models_to_try
    valid_pool = VALID_MODELS_CLF if pt == "classification" else VALID_MODELS_REG
    raw_models = decision.get("models_to_try", [])
    models = [m for m in raw_models if m in valid_pool]
    if not models:
        models = (["LogisticRegression", "RandomForest", "XGB"]
                  if pt == "classification"
                  else ["LinearRegression", "RandomForest", "XGB"])
    decision["models_to_try"] = models

    # metric
    valid_metrics = VALID_METRICS_CLF if pt == "classification" else VALID_METRICS_REG
    metric = decision.get("metric", "")
    if metric not in valid_metrics:
        metric = "accuracy" if pt == "classification" else "r2"
    decision["metric"] = metric

    # tuning_intensity
    intensity = decision.get("tuning_intensity", "")
    if intensity not in VALID_INTENSITIES:
        rows = metadata.get("num_rows", 1000)
        intensity = "light" if rows < 500 else ("medium" if rows < 5000 else "deep")
    decision["tuning_intensity"] = intensity

    return decision


def run_decision_agent(metadata: dict) -> dict:
    """
    Main entry point.
    Tries the LLM first; falls back to rule-based defaults if unavailable.
    """
    prompt   = _build_prompt(metadata)
    raw      = call_llm(prompt, max_new_tokens=256)
    decision = raw if isinstance(raw, dict) else {}

    decision = _sanitise(decision, metadata)

    print(f"[DecisionAgent] Decision: {json.dumps(decision, indent=2)}")
    return decision
