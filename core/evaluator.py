"""
Evaluator
---------
Given the best fitted pipeline, generates:
  - Full metric report (accuracy, F1, ROC-AUC for clf; R², RMSE, MAE for reg)
  - Confusion matrix (classification)
  - Feature importances / SHAP values where available
  - Returns everything as a structured dict for the UI to render
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")               # headless backend for Streamlit
import matplotlib.pyplot as plt
import io, base64

from sklearn.model_selection  import train_test_split
from sklearn.metrics          import (
    accuracy_score, f1_score, roc_auc_score, classification_report,
    confusion_matrix, r2_score, mean_squared_error, mean_absolute_error,
)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def _fig_to_base64(fig) -> str:
    """Convert a matplotlib figure to a base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_b64


def evaluate_pipeline(
    pipeline,
    df:           pd.DataFrame,
    target_col:   str,
    problem_type: str,
    test_size:    float = 0.20,
) -> dict:
    """
    Evaluate the best pipeline on a held-out test split.

    Returns a dict:
      metrics          : {metric_name: value}
      confusion_matrix : 2-D list (clf only)
      clf_report       : str  (clf only)
      feature_importance_plot : base64 PNG or None
      shap_plot               : base64 PNG or None
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42,
        stratify=(y if problem_type == "classification" else None),
    )

    # Re-fit on train split to get a proper held-out evaluation
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    result: dict = {
        "metrics":                   {},
        "confusion_matrix":          None,
        "clf_report":                None,
        "feature_importance_plot":   None,
        "shap_plot":                 None,
    }

    # ── Metrics ───────────────────────────────────────────────────────────────
    if problem_type == "classification":
        result["metrics"]["accuracy"] = round(accuracy_score(y_test, y_pred), 4)
        result["metrics"]["f1_weighted"] = round(
            f1_score(y_test, y_pred, average="weighted", zero_division=0), 4
        )
        try:
            y_prob = pipeline.predict_proba(X_test)
            if y_prob.shape[1] == 2:
                result["metrics"]["roc_auc"] = round(
                    roc_auc_score(y_test, y_prob[:, 1]), 4
                )
            else:
                result["metrics"]["roc_auc"] = round(
                    roc_auc_score(y_test, y_prob, multi_class="ovr",
                                  average="weighted"), 4
                )
        except Exception:
            pass

        result["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
        result["clf_report"] = classification_report(
            y_test, y_pred, zero_division=0
        )

    else:  # regression
        result["metrics"]["r2"]   = round(r2_score(y_test, y_pred), 4)
        result["metrics"]["rmse"] = round(
            np.sqrt(mean_squared_error(y_test, y_pred)), 4
        )
        result["metrics"]["mae"]  = round(mean_absolute_error(y_test, y_pred), 4)

    # ── Feature importance (tree-based models) ───────────────────────────────
    try:
        inner_model = pipeline.named_steps["model"]
        preprocessor = pipeline.named_steps["preprocessor"]

        if hasattr(inner_model, "feature_importances_"):
            # Get feature names from preprocessor
            feature_names = preprocessor.get_feature_names_out()
            importances   = inner_model.feature_importances_

            top_n = min(20, len(feature_names))
            idx   = np.argsort(importances)[::-1][:top_n]

            fig, ax = plt.subplots(figsize=(8, 0.4 * top_n + 1))
            ax.barh(
                [feature_names[i] for i in idx[::-1]],
                importances[idx[::-1]],
                color="#5b6ef5",
            )
            ax.set_xlabel("Importance")
            ax.set_title("Feature importances (top 20)")
            fig.tight_layout()
            result["feature_importance_plot"] = _fig_to_base64(fig)

    except Exception as exc:
        print(f"[Evaluator] Feature importance failed: {exc}")

    # ── SHAP values (tree models; optional) ──────────────────────────────────
    if SHAP_AVAILABLE:
        try:
            inner_model  = pipeline.named_steps["model"]
            preprocessor = pipeline.named_steps["preprocessor"]
            X_test_proc  = preprocessor.transform(X_test)

            # Use TreeExplainer for tree models, LinearExplainer for linear
            if hasattr(inner_model, "feature_importances_"):
                explainer = shap.TreeExplainer(inner_model)
            elif hasattr(inner_model, "coef_"):
                explainer = shap.LinearExplainer(inner_model, X_test_proc)
            else:
                explainer = None

            if explainer is not None:
                shap_vals = explainer.shap_values(X_test_proc[:100])  # cap at 100 rows
                if isinstance(shap_vals, list):
                    shap_vals = shap_vals[1]  # binary clf: take positive class

                fig, ax = plt.subplots(figsize=(8, 4))
                feature_names = preprocessor.get_feature_names_out()
                shap.summary_plot(
                    shap_vals,
                    X_test_proc[:100],
                    feature_names=feature_names,
                    plot_type="bar",
                    show=False,
                    max_display=15,
                )
                result["shap_plot"] = _fig_to_base64(plt.gcf())
                plt.close("all")
        except Exception as exc:
            print(f"[Evaluator] SHAP failed: {exc}")

    return result
