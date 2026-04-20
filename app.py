"""
app.py  —  Agentic AutoML Platform
====================================
Streamlit UI that orchestrates the full pipeline:

  Upload → Extract metadata → Decision Agent → Train → Reflect → Evaluate
       → Download artifacts   +   In-platform inference
"""

import os
import io
import json
import joblib
import zipfile
import tempfile
import base64

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from dotenv import load_dotenv

def _safe_load_dotenv():
    """Load .env with UTF-8 encoding fallback for Windows BOM/UTF-16 issues."""
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.exists(env_path):
        return
    for encoding in ("utf-8-sig", "utf-8", "utf-16", "latin-1"):
        try:
            load_dotenv(dotenv_path=env_path, encoding=encoding, override=False)
            return
        except (UnicodeDecodeError, Exception):
            continue

_safe_load_dotenv()

# ── Internal modules ─────────────────────────────────────────────────────────
from core.metadata_extractor  import extract_metadata
from core.model_trainer        import train_all_models
from core.evaluator            import evaluate_pipeline
from agents.decision_agent     import run_decision_agent
from agents.reflection_agent   import run_reflection_agent

# ── Constants ────────────────────────────────────────────────────────────────
ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
MODEL_PATH    = os.path.join(ARTIFACTS_DIR, "best_pipeline.pkl")
META_PATH     = os.path.join(ARTIFACTS_DIR, "session_meta.json")

MAX_RETUNE_ROUNDS = 2


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _b64_img(b64: str):
    """Decode base64 PNG and return as BytesIO for st.image."""
    return io.BytesIO(base64.b64decode(b64))


def _make_zip(pipeline, metadata: dict, eval_result: dict) -> bytes:
    """Bundle pipeline.pkl + meta.json + report.txt into a ZIP."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # pipeline
        pkl_buf = io.BytesIO()
        joblib.dump(pipeline, pkl_buf)
        zf.writestr("model/best_pipeline.pkl", pkl_buf.getvalue())

        # metadata + decision
        zf.writestr("meta/session_meta.json", json.dumps(metadata, indent=2))

        # text report
        lines = ["AutoML Report", "=" * 40, ""]
        lines.append(f"Problem type : {metadata.get('problem_type', '')}")
        lines.append(f"Best model   : {metadata.get('final_model', '')}")
        lines.append(f"Metric       : {metadata.get('metric', '')}")
        lines.append("")
        lines.append("Leaderboard")
        lines.append("-" * 30)
        for m, s in metadata.get("leaderboard", {}).items():
            lines.append(f"  {m:<22} {s:.5f}")
        lines.append("")
        lines.append("Evaluation metrics (held-out test set)")
        lines.append("-" * 30)
        for k, v in eval_result.get("metrics", {}).items():
            lines.append(f"  {k:<22} {v}")
        if eval_result.get("clf_report"):
            lines.append("")
            lines.append("Classification Report")
            lines.append(eval_result["clf_report"])
        lines.append("")
        lines.append("Usage (Python)")
        lines.append("-" * 30)
        lines.append("import joblib, pandas as pd")
        lines.append("pipeline = joblib.load('model/best_pipeline.pkl')")
        lines.append("df_new   = pd.read_csv('your_new_data.csv')")
        lines.append("preds    = pipeline.predict(df_new)   # raw DataFrame, no pre-processing needed")

        zf.writestr("report/report.txt", "\n".join(lines))

    buf.seek(0)
    return buf.read()


def _progress_callback(model_name: str, status: str):
    """Update Streamlit status text during training."""
    if "training_status" in st.session_state:
        st.session_state["training_status"][model_name] = status


# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Agentic AutoML",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .stTabs [data-baseweb="tab"] { font-size: 15px; }
    .metric-card {
        background: #f8f9fb;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        border: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────────────────────

for key, default in {
    "df":               None,
    "target_col":       None,
    "metadata":         None,
    "decision":         None,
    "leaderboard":      None,
    "fitted_pipelines": None,
    "reflection":       None,
    "warnings":         [],
    "eval_result":      None,
    "best_pipeline":    None,
    "session_meta":     {},
    "training_status":  {},
    "retune_round":     0,
    "pipeline_done":    False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🤖 Agentic AutoML")
    st.markdown("Upload your dataset. The AI agents handle everything else.")
    st.divider()

    uploaded_file = st.file_uploader(
        "Upload training dataset (CSV / Excel)",
        type=["csv", "xlsx", "xls"],
    )

    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.session_state["df"] = df
            st.success(f"Loaded: {df.shape[0]} rows × {df.shape[1]} cols")
        except Exception as exc:
            st.error(f"Failed to read file: {exc}")

    if st.session_state["df"] is not None:
        df = st.session_state["df"]
        target_col = st.selectbox(
            "Select target (prediction) column",
            options=df.columns.tolist(),
            index=len(df.columns) - 1,
        )
        st.session_state["target_col"] = target_col

        st.divider()
        run_btn = st.button("🚀 Run AutoML", type="primary", use_container_width=True)
    else:
        run_btn = False

    st.divider()
    st.caption("Built with scikit-learn · Optuna · HuggingFace LLM")


# ─────────────────────────────────────────────────────────────────────────────
# Main tabs
# ─────────────────────────────────────────────────────────────────────────────

tab_data, tab_decision, tab_train, tab_evaluate, tab_download, tab_predict = st.tabs([
    "📊 Data", "🧠 Decision Agent", "🏋️ Training", "📈 Evaluation",
    "📦 Download", "🔮 Predict",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Data Explorer
# ══════════════════════════════════════════════════════════════════════════════

with tab_data:
    if st.session_state["df"] is None:
        st.info("Upload a dataset from the sidebar to get started.")
    else:
        df  = st.session_state["df"]
        tgt = st.session_state["target_col"]

        st.subheader("Dataset preview")
        st.dataframe(df.head(50), use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows",     df.shape[0])
        c2.metric("Columns",  df.shape[1])
        c3.metric("Missing",  int(df.isnull().sum().sum()))
        c4.metric("Target",   tgt)

        st.subheader("Column types & missing values")
        dtype_df = pd.DataFrame({
            "dtype":   df.dtypes.astype(str),
            "missing": df.isnull().sum(),
            "missing %": (df.isnull().sum() / len(df) * 100).round(2),
            "unique":  df.nunique(),
        })
        st.dataframe(dtype_df, use_container_width=True)

        st.subheader("Target distribution")
        fig, ax = plt.subplots(figsize=(6, 3))
        if df[tgt].nunique() <= 20:
            df[tgt].value_counts().plot(kind="bar", ax=ax, color="#5b6ef5")
        else:
            df[tgt].plot(kind="hist", bins=30, ax=ax, color="#5b6ef5")
        ax.set_title(f"Distribution of '{tgt}'")
        st.pyplot(fig, use_container_width=False)


# ══════════════════════════════════════════════════════════════════════════════
# RUN PIPELINE (triggered by sidebar button)
# ══════════════════════════════════════════════════════════════════════════════

if run_btn:
    df  = st.session_state["df"]
    tgt = st.session_state["target_col"]

    if df is None or tgt is None:
        st.sidebar.error("Please upload a dataset and select a target column.")
    else:
        # Reset previous run
        st.session_state.update({
            "metadata": None, "decision": None, "leaderboard": None,
            "fitted_pipelines": None, "reflection": None, "warnings": [],
            "eval_result": None, "best_pipeline": None,
            "training_status": {}, "retune_round": 0, "pipeline_done": False,
        })

        # ── Step 1: Extract metadata ──────────────────────────────────────────
        with st.spinner("Extracting dataset metadata…"):
            try:
                metadata = extract_metadata(df, tgt)
                st.session_state["metadata"] = metadata
            except Exception as exc:
                st.error(f"Metadata extraction failed: {exc}")
                st.stop()

        # ── Step 2: Decision Agent ────────────────────────────────────────────
        with st.spinner("Decision Agent is analysing your data…"):
            decision = run_decision_agent(metadata)
            st.session_state["decision"] = decision

        # ── Step 3: Training loop (with optional retune) ───────────────────
        retune_round   = 0
        leaderboard    = {}
        fitted_pipelines = {}

        while retune_round <= MAX_RETUNE_ROUNDS:
            with st.spinner(
                f"Training models (round {retune_round + 1})… "
                f"[{', '.join(decision['models_to_try'])}]"
            ):
                st.session_state["training_status"] = {}
                lb, fps = train_all_models(
                    df, tgt, decision,
                    progress_cb=_progress_callback,
                )
                leaderboard      = lb
                fitted_pipelines = fps

            # ── Step 4: Reflection Agent ──────────────────────────────────
            with st.spinner("Reflection Agent is reviewing results…"):
                reflection, warnings = run_reflection_agent(
                    leaderboard, decision, retune_round=retune_round
                )

            if not reflection.get("retune", False):
                break

            retune_round += 1
            # Allow the agent to nudge intensity up
            if decision["tuning_intensity"] == "light":
                decision["tuning_intensity"] = "medium"
            elif decision["tuning_intensity"] == "medium":
                decision["tuning_intensity"] = "deep"

        st.session_state.update({
            "leaderboard":      leaderboard,
            "fitted_pipelines": fitted_pipelines,
            "reflection":       reflection,
            "warnings":         warnings,
            "retune_round":     retune_round,
        })

        # ── Step 5: Select best pipeline and evaluate ─────────────────────
        final_model_name = reflection["final_model"]
        best_pipeline    = fitted_pipelines.get(final_model_name)

        if best_pipeline is None:
            # Fallback: pick highest scoring model
            final_model_name = max(leaderboard, key=lambda k: leaderboard[k])
            best_pipeline    = fitted_pipelines[final_model_name]

        with st.spinner("Evaluating best model on held-out test set…"):
            eval_result = evaluate_pipeline(
                best_pipeline, df, tgt, decision["problem_type"]
            )

        # ── Step 6: Save artifacts ────────────────────────────────────────
        joblib.dump(best_pipeline, MODEL_PATH)

        session_meta = {
            **metadata,
            "problem_type":  decision["problem_type"],
            "metric":        decision["metric"],
            "models_tried":  decision["models_to_try"],
            "leaderboard":   leaderboard,
            "final_model":   final_model_name,
            "reflection":    reflection,
            "warnings":      warnings,
        }
        with open(META_PATH, "w") as f:
            json.dump(session_meta, f, indent=2, default=str)

        st.session_state.update({
            "best_pipeline":  best_pipeline,
            "eval_result":    eval_result,
            "session_meta":   session_meta,
            "pipeline_done":  True,
        })

        st.success("✅ AutoML pipeline complete! Check the tabs above.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Decision Agent
# ══════════════════════════════════════════════════════════════════════════════

with tab_decision:
    if st.session_state["decision"] is None:
        st.info("Run the AutoML pipeline to see agent decisions.")
    else:
        decision = st.session_state["decision"]
        meta     = st.session_state["metadata"]

        st.subheader("🧠 Decision Agent output")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Problem type",     decision["problem_type"].capitalize())
        c2.metric("Metric",           decision["metric"])
        c3.metric("Tuning intensity", decision["tuning_intensity"].capitalize())
        c4.metric("Models to try",    len(decision["models_to_try"]))

        st.markdown("**Models selected by agent:**")
        st.write(decision["models_to_try"])

        st.subheader("📋 Dataset metadata used by agent")
        st.json(meta, expanded=False)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Training & Leaderboard
# ══════════════════════════════════════════════════════════════════════════════

with tab_train:
    if st.session_state["leaderboard"] is None:
        st.info("Run the AutoML pipeline to see training results.")
    else:
        lb         = st.session_state["leaderboard"]
        reflection = st.session_state["reflection"]
        warnings   = st.session_state["warnings"]
        decision   = st.session_state["decision"]
        metric     = decision["metric"]

        st.subheader("🏆 Model leaderboard")

        lb_df = pd.DataFrame(
            [{"Model": k, metric: v, "Rank": i + 1}
             for i, (k, v) in enumerate(
                 sorted(lb.items(), key=lambda x: x[1], reverse=True)
             )]
        )
        lb_df["Best"] = lb_df["Model"] == reflection["final_model"]
        st.dataframe(
            lb_df.style.apply(
                lambda row: ["background-color: #e8f5e9" if row["Best"] else "" for _ in row],
                axis=1,
            ),
            use_container_width=True,
            hide_index=True,
        )

        # Bar chart
        fig, ax = plt.subplots(figsize=(7, 3))
        colors = [
            "#2e7d32" if m == reflection["final_model"] else "#5b6ef5"
            for m in lb_df["Model"]
        ]
        ax.barh(lb_df["Model"], lb_df[metric], color=colors)
        ax.set_xlabel(metric)
        ax.set_title("Leaderboard")
        st.pyplot(fig, use_container_width=False)

        st.subheader("🤔 Reflection Agent decision")
        rc1, rc2 = st.columns(2)
        rc1.success(f"**Final model:** {reflection['final_model']}")
        rc2.info(f"**Reasoning:** {reflection['reasoning']}")

        if st.session_state["retune_round"] > 0:
            st.caption(f"Retune rounds performed: {st.session_state['retune_round']}")

        if warnings:
            st.subheader("⚠️ Quality warnings")
            for w in warnings:
                st.warning(w)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Evaluation
# ══════════════════════════════════════════════════════════════════════════════

with tab_evaluate:
    if st.session_state["eval_result"] is None:
        st.info("Run the AutoML pipeline to see evaluation results.")
    else:
        er       = st.session_state["eval_result"]
        decision = st.session_state["decision"]
        sm       = st.session_state["session_meta"]

        st.subheader(f"📈 Evaluation — {sm.get('final_model', '')} "
                     f"({decision['problem_type']})")

        # Metric cards
        cols = st.columns(len(er["metrics"]))
        for col, (k, v) in zip(cols, er["metrics"].items()):
            col.metric(k, v)

        # Confusion matrix
        if er["confusion_matrix"] is not None:
            st.subheader("Confusion matrix")
            cm  = np.array(er["confusion_matrix"])
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig, use_container_width=False)

        if er["clf_report"]:
            with st.expander("Full classification report"):
                st.code(er["clf_report"])

        # Feature importances
        if er["feature_importance_plot"]:
            st.subheader("Feature importances")
            st.image(_b64_img(er["feature_importance_plot"]))

        # SHAP
        if er["shap_plot"]:
            st.subheader("SHAP summary")
            st.image(_b64_img(er["shap_plot"]))


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Download
# ══════════════════════════════════════════════════════════════════════════════

with tab_download:
    if not st.session_state["pipeline_done"]:
        st.info("Complete the AutoML pipeline to download artifacts.")
    else:
        sm = st.session_state["session_meta"]
        er = st.session_state["eval_result"]

        st.subheader("📦 Download your trained model")
        st.markdown(
            "The download includes:\n"
            "- `model/best_pipeline.pkl` — full sklearn pipeline (preprocessor + model)\n"
            "- `meta/session_meta.json` — dataset stats, agent decisions, leaderboard\n"
            "- `report/report.txt` — metrics report + usage instructions\n\n"
            "**Load and use in Python:**\n"
            "```python\n"
            "import joblib, pandas as pd\n"
            "pipeline = joblib.load('best_pipeline.pkl')\n"
            "df_new   = pd.read_csv('new_data.csv')  # raw, unprocessed\n"
            "preds    = pipeline.predict(df_new)\n"
            "```"
        )

        zip_bytes = _make_zip(
            st.session_state["best_pipeline"], sm, er
        )
        st.download_button(
            label="⬇️ Download model package (.zip)",
            data=zip_bytes,
            file_name=f"automl_{sm.get('final_model', 'model')}.zip",
            mime="application/zip",
            type="primary",
        )

        st.divider()
        st.subheader("Session summary")
        c1, c2, c3 = st.columns(3)
        c1.metric("Final model",  sm.get("final_model", ""))
        c2.metric("Problem type", sm.get("problem_type", "").capitalize())
        c3.metric("Metric",       sm.get("metric", ""))


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — In-Platform Prediction
# ══════════════════════════════════════════════════════════════════════════════

with tab_predict:
    if not st.session_state["pipeline_done"]:
        st.info("Complete the AutoML pipeline first, then return here to predict.")
    else:
        pipeline = st.session_state["best_pipeline"]
        decision = st.session_state["decision"]
        df_orig  = st.session_state["df"]
        tgt      = st.session_state["target_col"]
        feature_cols = [c for c in df_orig.columns if c != tgt]

        st.subheader("🔮 Make predictions on new data")

        pred_method = st.radio(
            "Input method",
            ["Upload CSV", "Manual input"],
            horizontal=True,
        )

        if pred_method == "Upload CSV":
            pred_file = st.file_uploader(
                "Upload new data (CSV) — must have the same feature columns, no target needed",
                type=["csv"],
                key="predict_upload",
            )
            if pred_file:
                try:
                    df_new = pd.read_csv(pred_file)

                    # Drop target if accidentally included
                    if tgt in df_new.columns:
                        df_new = df_new.drop(columns=[tgt])

                    # Warn about missing columns
                    missing_cols = [c for c in feature_cols if c not in df_new.columns]
                    if missing_cols:
                        st.warning(f"Missing columns in uploaded file: {missing_cols}")

                    preds = pipeline.predict(df_new)
                    df_out = df_new.copy()
                    df_out["prediction"] = preds

                    if decision["problem_type"] == "classification":
                        try:
                            proba = pipeline.predict_proba(df_new)
                            max_proba = proba.max(axis=1).round(4)
                            df_out["confidence"] = max_proba
                        except Exception:
                            pass

                    st.success(f"Predictions complete for {len(df_new)} rows.")
                    st.dataframe(df_out, use_container_width=True)

                    csv_bytes = df_out.to_csv(index=False).encode()
                    st.download_button(
                        "⬇️ Download predictions CSV",
                        data=csv_bytes,
                        file_name="predictions.csv",
                        mime="text/csv",
                    )
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")

        else:  # Manual input
            st.markdown("Fill in values for each feature:")
            input_vals = {}
            cols = st.columns(min(3, len(feature_cols)))
            for i, col_name in enumerate(feature_cols):
                sample_val = df_orig[col_name].dropna().iloc[0] if len(df_orig[col_name].dropna()) else ""
                with cols[i % len(cols)]:
                    if df_orig[col_name].dtype == object:
                        options = df_orig[col_name].dropna().unique().tolist()
                        input_vals[col_name] = st.selectbox(col_name, options, key=f"inp_{col_name}")
                    elif df_orig[col_name].dtype in [np.float64, np.float32]:
                        input_vals[col_name] = st.number_input(
                            col_name, value=float(sample_val), key=f"inp_{col_name}"
                        )
                    else:
                        input_vals[col_name] = st.number_input(
                            col_name, value=int(sample_val) if sample_val != "" else 0,
                            step=1, key=f"inp_{col_name}"
                        )

            if st.button("Predict", type="primary"):
                try:
                    df_single = pd.DataFrame([input_vals])
                    pred      = pipeline.predict(df_single)[0]
                    st.success(f"**Prediction: `{pred}`**")

                    if decision["problem_type"] == "classification":
                        try:
                            proba     = pipeline.predict_proba(df_single)[0]
                            classes   = pipeline.classes_
                            proba_df  = pd.DataFrame({"class": classes, "probability": proba})
                            st.bar_chart(proba_df.set_index("class"))
                        except Exception:
                            pass
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")