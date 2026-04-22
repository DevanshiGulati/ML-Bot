"""
Evaluator
---------
Evaluates the best pipeline on a held-out 20% test split.
Produces:
  - Full metric report
  - Confusion matrix (classification)
  - Classification report
  - Feature importance plot
  - SHAP summary plot
"""

import io, base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, classification_report,
    confusion_matrix, r2_score, mean_squared_error, mean_absolute_error,
)

try:
    import shap
    SHAP_OK = True
except ImportError:
    SHAP_OK = False


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64


def evaluate_pipeline(
    pipeline,
    df:           pd.DataFrame,
    target_col:   str,
    problem_type: str,
    test_size:    float = 0.20,
) -> dict:
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42,
        stratify=(y if problem_type == "classification" else None),
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    result = {
        "metrics": {},
        "confusion_matrix":        None,
        "clf_report":              None,
        "feature_importance_plot": None,
        "shap_plot":               None,
    }

    # Metrics
    if problem_type == "classification":
        result["metrics"]["accuracy"]    = round(accuracy_score(y_test, y_pred), 4)
        result["metrics"]["f1_weighted"] = round(
            f1_score(y_test, y_pred, average="weighted", zero_division=0), 4)
        try:
            prob = pipeline.predict_proba(X_test)
            auc  = (roc_auc_score(y_test, prob[:, 1]) if prob.shape[1] == 2
                    else roc_auc_score(y_test, prob, multi_class="ovr", average="weighted"))
            result["metrics"]["roc_auc"] = round(auc, 4)
        except Exception:
            pass
        result["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
        result["clf_report"]       = classification_report(y_test, y_pred, zero_division=0)
    else:
        result["metrics"]["r2"]   = round(r2_score(y_test, y_pred), 4)
        result["metrics"]["rmse"] = round(np.sqrt(mean_squared_error(y_test, y_pred)), 4)
        result["metrics"]["mae"]  = round(mean_absolute_error(y_test, y_pred), 4)

    # Feature importances
    try:
        inner = pipeline.named_steps["model"]
        prep  = pipeline.named_steps["preprocessor"]
        if hasattr(inner, "feature_importances_"):
            names = prep.get_feature_names_out()
            imps  = inner.feature_importances_
            top   = min(20, len(names))
            idx   = np.argsort(imps)[::-1][:top]
            fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * top + 1)))
            ax.barh([names[i] for i in idx[::-1]], imps[idx[::-1]], color="#5b6ef5")
            ax.set_xlabel("Importance")
            ax.set_title("Feature importances (top 20)")
            result["feature_importance_plot"] = _fig_to_b64(fig)
    except Exception as e:
        print(f"[Evaluator] Feature importance: {e}")

    # SHAP
    if SHAP_OK:
        try:
            inner    = pipeline.named_steps["model"]
            prep     = pipeline.named_steps["preprocessor"]
            X_proc   = prep.transform(X_test)
            names    = prep.get_feature_names_out()
            if hasattr(inner, "feature_importances_"):
                explainer = shap.TreeExplainer(inner)
            elif hasattr(inner, "coef_"):
                explainer = shap.LinearExplainer(inner, X_proc)
            else:
                explainer = None
            if explainer is not None:
                sv = explainer.shap_values(X_proc[:100])
                if isinstance(sv, list):
                    sv = sv[1]
                shap.summary_plot(sv, X_proc[:100], feature_names=names,
                                  plot_type="bar", show=False, max_display=15)
                result["shap_plot"] = _fig_to_b64(plt.gcf())
                plt.close("all")
        except Exception as e:
            print(f"[Evaluator] SHAP: {e}")

    return result