"""
Decision Agent
--------------
Truly agentic: receives raw dataset metadata, reasons about it,
and produces a structured training plan.

Agentic properties:
  - Perceives: dataset metadata (rows, types, imbalance, correlations)
  - Reasons:   LLM analyses characteristics and selects strategy
  - Acts:      returns concrete plan (models, metric, intensity)
  - Adapts:    adjusts based on strategy_hint from previous round

Uses LLM when available, validated rule-based fallback otherwise.
Both paths produce identical output schema.
"""

import json
from agents.llm_client import call_llm

VALID_PROBLEM_TYPES = {"classification", "regression"}
VALID_MODELS_CLF    = {"LogisticRegression", "DecisionTree", "RandomForest", "XGB", "LightGBM"}
VALID_MODELS_REG    = {"LinearRegression",   "DecisionTree", "RandomForest", "XGB", "LightGBM"}
VALID_METRICS_CLF   = {"accuracy", "f1", "roc_auc"}
VALID_METRICS_REG   = {"r2", "rmse", "mae"}
VALID_INTENSITIES   = {"light", "medium", "deep"}


def _build_prompt(metadata: dict, strategy_hint: str = "") -> str:
    """
    Rich prompt with full dataset context.
    Includes strategy_hint if this is a retune round.
    """
    hint_section = ""
    if strategy_hint:
        hint_section = f"""
IMPORTANT — PREVIOUS ROUND CONTEXT:
The reflection agent identified these issues: {strategy_hint}
Adjust your model selection and intensity accordingly.
- If "overfit": prefer simpler models, stronger regularisation
- If "underfit": prefer complex models, higher intensity
- If "target_not_met": increase tuning intensity
- If "poor_generalisation": reduce model complexity
"""

    return f"""<s>[INST]
You are an expert ML engineer for an AutoML system. Your job is to analyse a dataset and decide the optimal training strategy.

DATASET ANALYSIS:
{json.dumps(metadata, indent=2)}
{hint_section}
DECISION RULES:
1. problem_type:
   - "classification" if target_unique_values <= 20 OR target_dtype = "object"
   - "regression" otherwise

2. models_to_try (pick exactly 3-4):
   Classification pool: LogisticRegression, DecisionTree, RandomForest, XGB, LightGBM
   Regression pool: LinearRegression, DecisionTree, RandomForest, XGB, LightGBM
   - Always include 1 baseline (LogisticRegression or LinearRegression)
   - Always include XGB or LightGBM (best for tabular data)
   - For small datasets (< 500 rows): avoid LightGBM

3. metric:
   Classification: "f1" if class_imbalance=true, else "accuracy"
   Regression: "r2" for general use, "rmse" if high variance in target

4. tuning_intensity:
   "light"  = num_rows < 500
   "medium" = num_rows 500–5000
   "deep"   = num_rows > 5000

RESPOND WITH ONLY THIS JSON (no text before or after):
{{
  "problem_type": "classification",
  "models_to_try": ["LogisticRegression", "RandomForest", "XGB", "LightGBM"],
  "metric": "accuracy",
  "tuning_intensity": "medium",
  "reasoning": "Dataset has 891 rows with binary target, tree ensembles optimal for tabular data"
}}
[/INST]"""


def _rule_based_decision(metadata: dict, strategy_hint: str = "") -> dict:
    """Complete rule-based fallback — same decision logic as LLM prompt."""
    n_unique   = metadata.get("target_unique_values", 2)
    dtype      = str(metadata.get("target_dtype", ""))
    rows       = metadata.get("num_rows", 1000)
    imbalanced = metadata.get("class_imbalance", False)
    n_cat      = metadata.get("num_categorical_features", 0)

    pt = "classification" if (n_unique <= 20 or "object" in dtype) else "regression"

    if pt == "classification":
        if rows < 500:
            models = ["LogisticRegression", "DecisionTree", "RandomForest", "XGB"]
        else:
            models = ["LogisticRegression", "RandomForest", "XGB", "LightGBM"]
        metric = "f1" if imbalanced else "accuracy"
    else:
        if rows < 500:
            models = ["LinearRegression", "DecisionTree", "RandomForest", "XGB"]
        else:
            models = ["LinearRegression", "RandomForest", "XGB", "LightGBM"]
        metric = "r2"

    intensity = "light" if rows < 500 else ("medium" if rows < 5000 else "deep")

    # Apply strategy hints from reflection agent
    if strategy_hint == "overfit":
        models = [m for m in models if m not in ["XGB", "LightGBM"]] or models[:2]
        intensity = {"deep": "medium", "medium": "light"}.get(intensity, intensity)
    elif strategy_hint == "underfit":
        if "XGB" not in models:
            models.append("XGB")
        if "LightGBM" not in models:
            models.append("LightGBM")
        intensity = {"light": "medium", "medium": "deep"}.get(intensity, intensity)
    elif strategy_hint == "target_not_met":
        intensity = {"light": "medium", "medium": "deep"}.get(intensity, intensity)

    return {
        "problem_type":    pt,
        "models_to_try":   models[:4],
        "metric":          metric,
        "tuning_intensity": intensity,
        "reasoning":       f"Rule-based: {pt}, {rows} rows, imbalanced={imbalanced}",
        "llm_used":        False,
    }


def _validate(decision: dict, metadata: dict) -> dict:
    """Validate and repair LLM output."""
    fixes = []

    # problem_type
    pt = str(decision.get("problem_type", "")).lower()
    if pt not in VALID_PROBLEM_TYPES:
        n_unique = metadata.get("target_unique_values", 2)
        pt = "classification" if n_unique <= 20 else "regression"
        fixes.append(f"problem_type → {pt}")
    decision["problem_type"] = pt

    # models_to_try
    pool   = VALID_MODELS_CLF if pt == "classification" else VALID_MODELS_REG
    models = [m for m in decision.get("models_to_try", []) if m in pool]
    if len(models) < 2:
        models = (["LogisticRegression", "RandomForest", "XGB", "LightGBM"]
                  if pt == "classification"
                  else ["LinearRegression", "RandomForest", "XGB", "LightGBM"])
        fixes.append("models_to_try → defaults")
    decision["models_to_try"] = models[:4]

    # metric
    valid_m = VALID_METRICS_CLF if pt == "classification" else VALID_METRICS_REG
    if decision.get("metric") not in valid_m:
        decision["metric"] = "accuracy" if pt == "classification" else "r2"
        fixes.append(f"metric → {decision['metric']}")

    # tuning_intensity
    if decision.get("tuning_intensity") not in VALID_INTENSITIES:
        rows = metadata.get("num_rows", 1000)
        decision["tuning_intensity"] = "light" if rows < 500 else ("medium" if rows < 5000 else "deep")
        fixes.append(f"tuning_intensity → {decision['tuning_intensity']}")

    if fixes:
        print(f"[DecisionAgent] Fixed LLM output: {fixes}")
    else:
        print(f"[DecisionAgent] LLM output valid — no fixes needed ✓")

    decision["llm_used"] = True
    return decision


def run_decision_agent(metadata: dict, strategy_hint: str = "") -> dict:
    """
    Main entry — truly agentic decision making.
    Uses LLM reasoning when available, rule-based when not.
    """
    print(f"\n{'─'*64}")
    print(f"[DecisionAgent] Starting analysis")
    print(f"  rows={metadata.get('num_rows')}  "
          f"features={metadata.get('num_features')}  "
          f"target_unique={metadata.get('target_unique_values')}  "
          f"imbalanced={metadata.get('class_imbalance')}  "
          f"missing={metadata.get('has_missing_values')}")
    if strategy_hint:
        print(f"  strategy_hint from reflection: {strategy_hint}")

    prompt = _build_prompt(metadata, strategy_hint)
    raw    = call_llm(prompt, max_new_tokens=350, agent_name="DecisionAgent")

    if raw is not None:
        print(f"[DecisionAgent] ✓ LLM decision received — validating...")
        decision = _validate(raw, metadata)
    else:
        print(f"[DecisionAgent] ⚠ LLM unavailable — rule-based fallback")
        decision = _rule_based_decision(metadata, strategy_hint)

    if strategy_hint:
        decision["strategy_hint"] = strategy_hint

    print(f"\n[DecisionAgent] DECISION (llm={decision.get('llm_used', False)}):")
    print(f"  problem_type    : {decision['problem_type']}")
    print(f"  models_to_try   : {decision['models_to_try']}")
    print(f"  metric          : {decision['metric']}")
    print(f"  tuning_intensity: {decision['tuning_intensity']}")
    print(f"  reasoning       : {decision.get('reasoning', 'N/A')}")
    print(f"{'─'*64}\n")

    return decision