"""
Agentic AutoML Platform — app.py
=================================
Full pipeline:
  Upload → Metadata → Decision Agent → Train → Validate → Reflect
       ↑_____________________________ retune (max 3 rounds) ___|
       → Evaluate → Download + In-platform Inference
"""

import os, io, json, base64, zipfile, warnings
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

# ── Load .env (Windows UTF-16 safe) ──────────────────────────────────────────
def _safe_load_dotenv():
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.exists(env_path):
        return
    for enc in ("utf-8-sig", "utf-8", "utf-16", "latin-1"):
        try:
            load_dotenv(dotenv_path=env_path, encoding=enc, override=False)
            return
        except Exception:
            continue

_safe_load_dotenv()

# ── Internal imports ──────────────────────────────────────────────────────────
from core.metadata_extractor import extract_metadata
from core.model_trainer      import train_all_models
from core.model_validator    import validate_all_models
from core.evaluator          import evaluate_pipeline
from agents.decision_agent   import run_decision_agent
from agents.reflection_agent import run_reflection_agent

# ── Constants ─────────────────────────────────────────────────────────────────
ARTIFACTS_DIR    = "artifacts"
MAX_RETUNE       = 3
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
MODEL_PATH       = os.path.join(ARTIFACTS_DIR, "best_pipeline.pkl")

# ── Helpers ───────────────────────────────────────────────────────────────────

def _b64img(b64: str):
    return io.BytesIO(base64.b64decode(b64))


def _make_zip(pipeline, session_meta: dict, eval_result: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        pkl_buf = io.BytesIO()
        joblib.dump(pipeline, pkl_buf)
        zf.writestr("model/best_pipeline.pkl", pkl_buf.getvalue())
        zf.writestr("meta/session_meta.json", json.dumps(session_meta, indent=2, default=str))

        lines = [
            "AutoML Report", "=" * 50, "",
            f"Problem type : {session_meta.get('problem_type','')}",
            f"Best model   : {session_meta.get('final_model','')}",
            f"Metric       : {session_meta.get('metric','')}",
            f"Retune rounds: {session_meta.get('retune_rounds', 0)}",
            "", "Leaderboard", "-" * 30,
        ]
        for m, s in session_meta.get("leaderboard", {}).items():
            lines.append(f"  {m:<22} {s:.5f}")
        lines += ["", "Evaluation metrics (held-out 20% test)", "-" * 30]
        for k, v in eval_result.get("metrics", {}).items():
            lines.append(f"  {k:<22} {v}")
        if eval_result.get("clf_report"):
            lines += ["", "Classification Report", eval_result["clf_report"]]
        lines += [
            "", "Usage", "-" * 30,
            "import joblib, pandas as pd",
            "pipeline = joblib.load('model/best_pipeline.pkl')",
            "df_new   = pd.read_csv('new_data.csv')",
            "preds    = pipeline.predict(df_new)  # no preprocessing needed",
        ]
        zf.writestr("report/report.txt", "\n".join(lines))
    buf.seek(0)
    return buf.read()


def _progress_cb(model_name: str, status: str):
    st.session_state.setdefault("training_status", {})[model_name] = status

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agentic AutoML",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container{padding-top:3.2rem}
.stTabs [data-baseweb="tab"]{font-size:14px}
</style>
""", unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────────────────────────
_defaults = {
    "df": None, "target_col": None, "metadata": None,
    "decision": None, "leaderboard": None,
    "fitted_pipelines": None, "studies": {},
    "validation_reports": {}, "reflection": None,
    "warnings": [], "eval_result": None,
    "best_pipeline": None, "session_meta": {},
    "training_status": {}, "retune_round": 0,
    "pipeline_done": False, "run_log": [],
    "user_criteria": {},
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🤖 Agentic AutoML")
    st.caption("Upload data. Agents handle everything else.")
    st.divider()

    uploaded = st.file_uploader("Training dataset (CSV / Excel)",
                                type=["csv", "xlsx", "xls"])
    if uploaded:
        try:
            df = (pd.read_csv(uploaded) if uploaded.name.endswith(".csv")
                  else pd.read_excel(uploaded))
            st.session_state["df"] = df
            st.success(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} cols")
        except Exception as e:
            st.error(f"Read failed: {e}")

    if st.session_state["df"] is not None:
        df  = st.session_state["df"]
        tgt = st.selectbox("Target column", df.columns.tolist(),
                           index=len(df.columns) - 1)
        st.session_state["target_col"] = tgt

        st.divider()
        st.markdown("**Quality criteria** *(optional)*")

        min_score = st.number_input(
            "Required score (0 = auto)", 0.0, 1.0, 0.0, 0.01,
            help="e.g. 0.90 means the agent will keep retuning until it hits 90%"
        )
        max_overfit = st.slider(
            "Max overfit gap (train − CV)", 0.01, 0.30, 0.10, 0.01,
            help="Larger gap = model memorised training data"
        )
        max_gen = st.slider(
            "Max generalisation gap (CV − test)", 0.01, 0.30, 0.08, 0.01,
            help="Larger gap = poor performance on unseen data"
        )
        st.session_state["user_criteria"] = {
            "min_score":       min_score if min_score > 0 else None,
            "max_overfit_gap": max_overfit,
            "max_gen_gap":     max_gen,
            "max_cv_std":      0.06,
        }

        st.divider()
        run_btn = st.button("🚀 Run AutoML", type="primary", use_container_width=True)
    else:
        run_btn = False

    st.divider()
    # LLM status indicator
    from agents.llm_client import HF_API_KEY, MODEL_CHAIN, get_current_model
    if HF_API_KEY:
        st.success(f"HF key: {HF_API_KEY[:8]}...{HF_API_KEY[-4:]}")
        active = get_current_model()
        st.caption(f"Active model: {active}")
        with st.expander("Model fallback chain"):
            for i, m in enumerate(MODEL_CHAIN):
                st.caption(f"{'→' if i==0 else '  '} {m}")
    else:
        st.warning("No API key — rule-based agents")
        with st.expander("Setup LLM (optional)"):
            st.markdown(
                "1. Get key: https://huggingface.co/settings/tokens\n"
                "2. Scope: **read**\n"
                "3. Add to `.env`:\n"
                "```\nHUGGINGFACE_API_KEY=hf_xxx\n```\n"
                "4. Restart Streamlit"
            )
    st.caption("sklearn | Optuna TPE | HuggingFace LLM")

# ── Tabs ──────────────────────────────────────────────────────────────────────
(tab_data, tab_decision, tab_train,
 tab_validate, tab_evaluate,
 tab_download, tab_predict) = st.tabs([
    "📊 Data", "🧠 Decision Agent", "🏋️ Training",
    "✅ Validation", "📈 Evaluation",
    "📦 Download", "🔮 Predict",
])

# ═════════════════════════════════════════════════════════════════════════════
# DATA TAB
# ═════════════════════════════════════════════════════════════════════════════
with tab_data:
    if st.session_state["df"] is None:
        st.info("Upload a dataset from the sidebar to get started.")
    else:
        df  = st.session_state["df"]
        tgt = st.session_state["target_col"]

        st.subheader("Dataset preview")
        st.dataframe(df.head(50), use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows",    f"{df.shape[0]:,}")
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing", int(df.isnull().sum().sum()))
        c4.metric("Target",  tgt)

        st.subheader("Column info")
        info_df = pd.DataFrame({
            "dtype":     df.dtypes.astype(str),
            "missing":   df.isnull().sum(),
            "missing %": (df.isnull().sum() / len(df) * 100).round(2),
            "unique":    df.nunique(),
        })
        st.dataframe(info_df, use_container_width=True)

        st.subheader(f"Target distribution — '{tgt}'")
        fig, ax = plt.subplots(figsize=(6, 3))
        if df[tgt].nunique() <= 20:
            df[tgt].value_counts().plot(kind="bar", ax=ax, color="#5b6ef5")
        else:
            df[tgt].plot(kind="hist", bins=30, ax=ax, color="#5b6ef5")
        ax.set_title(f"'{tgt}'")
        st.pyplot(fig, use_container_width=False)

# ═════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE (triggered by Run button)
# ═════════════════════════════════════════════════════════════════════════════
if run_btn:
    df  = st.session_state["df"]
    tgt = st.session_state["target_col"]
    uc  = st.session_state["user_criteria"]

    if df is None or tgt is None:
        st.sidebar.error("Upload a dataset and select a target column.")
    else:
        # Reset previous run
        for k in ["metadata","decision","leaderboard","fitted_pipelines","studies",
                  "validation_reports","reflection","warnings","eval_result",
                  "best_pipeline","training_status","retune_round","pipeline_done","run_log"]:
            st.session_state[k] = {} if k in ("training_status","studies",
                                               "validation_reports") else (
                [] if k in ("warnings","run_log") else None if k not in
                ("retune_round","pipeline_done") else
                (0 if k == "retune_round" else False))

        run_log: list[dict] = []

        # ── Step 1: Metadata ──────────────────────────────────────────────
        with st.spinner("Extracting dataset metadata…"):
            metadata = extract_metadata(df, tgt)
            st.session_state["metadata"] = metadata

        # ── Step 2: Decision Agent ────────────────────────────────────────
        with st.spinner("🧠 Decision Agent analysing dataset and planning pipeline…"):
            decision = run_decision_agent(metadata)
            st.session_state["decision"] = decision
            st.session_state["initial_decision"] = decision.copy()
            if decision.get("llm_used"):
                st.toast("✅ Decision Agent: LLM call successful", icon="🧠")
            else:
                st.toast("⚠️ Decision Agent: using rule-based fallback (check API key)", icon="⚠️")

        # ── Agentic retune loop ────────────────────────────────────────────
        retune_round     = 0
        leaderboard      = {}
        fitted_pipelines = {}
        studies          = {}
        reflection       = {}
        warnings_list    = []

        while retune_round <= MAX_RETUNE:

            strategy = decision.get("strategy_hint", "")
            round_label = (f"Round {retune_round + 1}"
                           + (f" [strategy: {strategy}]" if strategy else ""))

            with st.spinner(
                f"🏋️ Training ({round_label}) — "
                f"models: {decision['models_to_try']} | "
                f"intensity: {decision['tuning_intensity']}…"
            ):
                st.session_state["training_status"] = {}
                lb, fps, sts = train_all_models(df, tgt, decision, _progress_cb)
                leaderboard.update(lb)
                fitted_pipelines.update(fps)
                studies.update(sts)

            with st.spinner("✅ Validating models against quality criteria…"):
                val_reports = validate_all_models(fps, df, tgt, decision, uc)
                st.session_state["validation_reports"].update(val_reports)

            # Show what criteria each model passed/failed
            pass_summary = {n: r.passed_all for n, r in val_reports.items()}
            print(f"\n[App] Validation summary round {retune_round+1}: {pass_summary}")

            with st.spinner(f"🤔 Reflection Agent reviewing round {retune_round+1} results…"):
                reflection, warnings_list, next_decision = run_reflection_agent(
                    lb, decision, val_reports, retune_round,
                )
                if reflection.get("llm_used"):
                    st.toast(f"✅ Reflection Agent round {retune_round+1}: LLM call successful", icon="🤔")
                else:
                    st.toast(f"⚠️ Reflection Agent round {retune_round+1}: rule-based fallback", icon="⚠️")

            # Log this round
            run_log.append({
                "round":       retune_round + 1,
                "models":      list(lb.keys()),
                "scores":      {k: round(v, 4) for k, v in lb.items()},
                "retune":      reflection.get("retune", False),
                "reasoning":   reflection.get("reasoning", ""),
                "strategy":    next_decision.get("strategy_hint", "none"),
                "issues":      reflection.get("issues", []),
                "passed":      pass_summary,
                "intensity":   decision["tuning_intensity"],
            })

            print(f"\n[App] Round {retune_round+1} complete — "
                  f"retune={reflection.get('retune')} "
                  f"final_model={reflection.get('final_model')}")

            if not reflection.get("retune", False):
                break

            retune_round += 1
            decision = next_decision   # adapted strategy for next round
            # Re-run decision agent with strategy hint for true agentic behaviour
            strategy = decision.get("strategy_hint", "")
            print(f"\n[App] >>> Starting retune round {retune_round+1} "
                  f"with strategy: {strategy}\n")
            with st.spinner(f"🧠 Decision Agent re-planning with strategy: {strategy}…"):
                decision = run_decision_agent(
                    st.session_state["metadata"],
                    strategy_hint=strategy,
                )
                decision["strategy_hint"] = strategy

        # ── Final model selection ──────────────────────────────────────────
        best_name = reflection.get("final_model", "")
        if best_name not in fitted_pipelines:
            best_name = max(leaderboard, key=lambda k: leaderboard[k])
        best_pipeline = fitted_pipelines[best_name]

        # ── Evaluate best model ────────────────────────────────────────────
        with st.spinner(f"Evaluating {best_name} on held-out test set…"):
            eval_result = evaluate_pipeline(
                best_pipeline, df, tgt, decision["problem_type"]
            )

        # ── Save artifacts ─────────────────────────────────────────────────
        joblib.dump(best_pipeline, MODEL_PATH)

        session_meta = {
            **metadata,
            "problem_type":  decision["problem_type"],
            "metric":        decision["metric"],
            "models_tried":  list(fitted_pipelines.keys()),
            "leaderboard":   leaderboard,
            "final_model":   best_name,
            "reflection":    reflection,
            "warnings":      warnings_list,
            "retune_rounds": retune_round,
            "run_log":       run_log,
        }

        st.session_state.update({
            "leaderboard":       leaderboard,
            "fitted_pipelines":  fitted_pipelines,
            "studies":           studies,
            "reflection":        reflection,
            "warnings":          warnings_list,
            "eval_result":       eval_result,
            "best_pipeline":     best_pipeline,
            "session_meta":      session_meta,
            "retune_round":      retune_round,
            "pipeline_done":     True,
            "run_log":           run_log,
        })

        st.success(
            f"✅ Done! Best model: **{best_name}** "
            f"(rounds: {retune_round + 1}/{MAX_RETUNE + 1})"
        )

# ═════════════════════════════════════════════════════════════════════════════
# DECISION AGENT TAB
# ═════════════════════════════════════════════════════════════════════════════
with tab_decision:
    if st.session_state["decision"] is None:
        st.info("Run the AutoML pipeline first.")
    else:
        d  = st.session_state["decision"]
        m  = st.session_state["metadata"]
        rl = st.session_state.get("run_log", [])

        st.subheader("🧠 Decision Agent output")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Problem type",     d["problem_type"].capitalize())
        c2.metric("Metric",           d["metric"])
        c3.metric("Tuning intensity", d["tuning_intensity"].capitalize())
        c4.metric("Models selected",  len(d["models_to_try"]))
        from agents.llm_client import get_current_model
        llm_badge = f"🤖 LLM ({get_current_model()})" if d.get("llm_used") else "📐 Rule-based (LLM unavailable or failed)"
        col_badge1, col_badge2 = st.columns([2, 3])
        col_badge1.info(llm_badge)
        col_badge2.info(f"Models: {', '.join(d['models_to_try'])}")
        if d.get("reasoning"):
            st.caption(f"Reasoning: {d.get('reasoning', '')}")

        if rl:
            st.subheader("Agentic retune log")
            for entry in rl:
                status = "✅" if not entry["retune"] else "🔄"
                with st.expander(
                    f"{status} Round {entry['round']} — "
                    f"{entry['reasoning']}", expanded=True
                ):
                    sc1, sc2 = st.columns(2)
                    sc1.write(f"**Models:** {', '.join(entry['models'])}")
                    sc2.write(f"**Strategy:** {entry['strategy']}")
                    sc3, sc4 = st.columns(2)
                    sc3.write(f"**Scores:** {entry['scores']}")
                    sc4.write(f"**Retune?** {entry['retune']}")
                    sc5, _ = st.columns(2)
                    sc5.write(f"**Passed:** {entry['passed']}")

        st.subheader("Metadata used by agent")
        st.json(m, expanded=False)

# ═════════════════════════════════════════════════════════════════════════════
# TRAINING TAB
# ═════════════════════════════════════════════════════════════════════════════
with tab_train:
    if st.session_state["leaderboard"] is None:
        st.info("Run the AutoML pipeline first.")
    else:
        lb       = st.session_state["leaderboard"]
        refl     = st.session_state["reflection"]
        decision = st.session_state["decision"]
        metric   = decision["metric"]
        studies  = st.session_state.get("studies", {})

        st.subheader("🏆 Leaderboard")
        lb_rows = [
            {"Model": k, metric: v, "Best": k == refl.get("final_model")}
            for k, v in sorted(lb.items(), key=lambda x: x[1], reverse=True)
        ]
        lb_df = pd.DataFrame(lb_rows)

        def _highlight(row):
            return ["background-color:#e8f5e9" if row["Best"] else "" for _ in row]

        st.dataframe(
            lb_df.style.apply(_highlight, axis=1),
            use_container_width=True, hide_index=True,
        )

        fig, ax = plt.subplots(figsize=(7, 3))
        colors = ["#2e7d32" if r["Best"] else "#5b6ef5" for r in lb_rows]
        ax.barh([r["Model"] for r in lb_rows[::-1]],
                [r[metric] for r in lb_rows[::-1]], color=colors[::-1])
        ax.set_xlabel(metric)
        ax.set_title("Leaderboard")
        st.pyplot(fig, use_container_width=False)

        st.subheader("Reflection Agent decision")
        rc1, rc2, rc3 = st.columns(3)
        rc1.success(f"**Final model:** {refl.get('final_model','')}")
        rc2.info(f"**Reasoning:** {refl.get('reasoning','')}")
        rc3.metric("Retune rounds completed", st.session_state.get("retune_round", 0))

        if st.session_state.get("warnings"):
            st.subheader("⚠️ Warnings")
            for w in st.session_state["warnings"]:
                st.warning(w)

        # Optuna visualisation
        valid_studies = {k: v for k, v in studies.items() if v is not None}
        if valid_studies:
            st.subheader("Optuna HPO details")
            sel = st.selectbox("Study for model:", list(valid_studies.keys()))
            study = valid_studies[sel]
            oc1, oc2, oc3 = st.columns(3)
            oc1.metric("Best score",   f"{study.best_value:.5f}")
            oc2.metric("Trials run",   len(study.trials))
            oc3.metric("Best trial #", study.best_trial.number)
            with st.expander("Best hyperparameters"):
                st.json(study.best_params)
            scores = [t.value for t in study.trials if t.value is not None]
            if scores:
                fig, ax = plt.subplots(figsize=(7, 3))
                ax.plot(scores, alpha=0.4, color="#888780", linewidth=1, label="trial")
                best_so_far = [max(scores[:i+1]) for i in range(len(scores))]
                ax.plot(best_so_far, color="#1D9E75", linewidth=2, label="best so far")
                ax.set_xlabel("Trial"); ax.set_ylabel("Score")
                ax.set_title(f"Optuna optimisation history — {sel}")
                ax.legend()
                st.pyplot(fig, use_container_width=False)

# ═════════════════════════════════════════════════════════════════════════════
# VALIDATION TAB
# ═════════════════════════════════════════════════════════════════════════════
with tab_validate:
    vr = st.session_state.get("validation_reports", {})
    if not vr:
        st.info("Run the AutoML pipeline first.")
    else:
        decision = st.session_state["decision"]
        metric   = decision["metric"]
        uc       = st.session_state.get("user_criteria", {})

        st.subheader("✅ Model validation report")
        if uc.get("min_score"):
            st.caption(
                f"Criteria — Required {metric} ≥ {uc['min_score']} · "
                f"Max overfit gap ≤ {uc['max_overfit_gap']} · "
                f"Max generalisation gap ≤ {uc['max_gen_gap']}"
            )

        for model_name, report in vr.items():
            rd = report.as_dict()
            icon = "✅" if rd["passed_all"] else "⚠️"
            with st.expander(
                f"{icon} {model_name} — CV {metric}={rd['cv_score']:.4f} ± {rd['cv_std']:.4f}",
                expanded=True,
            ):
                v1, v2, v3, v4 = st.columns(4)
                v1.metric("Train score",   f"{rd['train_score']:.4f}")
                v2.metric("CV score",      f"{rd['cv_score']:.4f} ± {rd['cv_std']:.4f}")
                v3.metric("Test score",    f"{rd['test_score']:.4f}")
                v4.metric("Dummy baseline",f"{rd['baseline_score']:.4f}")

                g1, g2 = st.columns(2)
                of_icon = "🔴" if rd["overfit_gap"] > uc.get("max_overfit_gap", 0.10) else "🟢"
                # Negative generalise_gap means test >= cv — that is actually GOOD
                gen_gap = rd["generalise_gap"]
                gn_icon = "🟢" if gen_gap <= uc.get("max_gen_gap", 0.08) else "🔴"
                gen_label = f"{gen_gap:.4f}" + (" ✓ test≥cv" if gen_gap < 0 else "")
                g1.metric(f"{of_icon} Overfit gap (train−CV)",   f"{rd['overfit_gap']:.4f}")
                g2.metric(f"{gn_icon} Generalise gap (CV−test)", gen_label,
                          help="Negative = test score ≥ CV score (good generalisation). "
                               "Only flag when CV >> test.")

                st.markdown("**Criteria breakdown**")
                for cr in rd["criteria"]:
                    tick = "✅" if cr["passed"] else "❌"
                    val  = f" ({cr['value']:.4f})" if cr["value"] is not None else ""
                    st.markdown(f"{tick} **{cr['name']}**{val} — {cr['message']}")

                if rd["passed_all"]:
                    st.success(rd["summary"])
                elif not rd["target_met"] or rd["overfit_gap"] > 0.15:
                    st.error(rd["summary"])
                else:
                    st.warning(rd["summary"])

# ═════════════════════════════════════════════════════════════════════════════
# EVALUATION TAB
# ═════════════════════════════════════════════════════════════════════════════
with tab_evaluate:
    if st.session_state["eval_result"] is None:
        st.info("Run the AutoML pipeline first.")
    else:
        er  = st.session_state["eval_result"]
        sm  = st.session_state["session_meta"]
        dec = st.session_state["decision"]

        st.subheader(
            f"📈 Evaluation — {sm.get('final_model','')} ({dec['problem_type']})"
        )

        cols = st.columns(len(er["metrics"]))
        for col, (k, v) in zip(cols, er["metrics"].items()):
            col.metric(k, v)

        if er["confusion_matrix"] is not None:
            st.subheader("Confusion matrix")
            cm = np.array(er["confusion_matrix"])
            fig, ax = plt.subplots(figsize=(min(8, cm.shape[0] + 2), min(6, cm.shape[0] + 1)))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            st.pyplot(fig, use_container_width=False)

        if er["clf_report"]:
            with st.expander("Full classification report"):
                st.code(er["clf_report"])

        if er["feature_importance_plot"]:
            st.subheader("Feature importances")
            st.image(_b64img(er["feature_importance_plot"]))

        if er["shap_plot"]:
            st.subheader("SHAP summary")
            st.image(_b64img(er["shap_plot"]))

# ═════════════════════════════════════════════════════════════════════════════
# DOWNLOAD TAB
# ═════════════════════════════════════════════════════════════════════════════
with tab_download:
    if not st.session_state["pipeline_done"]:
        st.info("Complete the AutoML pipeline first.")
    else:
        sm = st.session_state["session_meta"]
        er = st.session_state["eval_result"]

        st.subheader("📦 Download trained model package")
        st.markdown(
            "**Contents of the ZIP:**\n"
            "- `model/best_pipeline.pkl` — full sklearn pipeline (preprocessor + model)\n"
            "- `meta/session_meta.json` — all agent decisions, leaderboard, run log\n"
            "- `report/report.txt` — metrics, scores, usage instructions\n\n"
            "**Load in Python:**\n"
            "```python\n"
            "import joblib, pandas as pd\n"
            "pipeline = joblib.load('best_pipeline.pkl')\n"
            "preds = pipeline.predict(pd.read_csv('new_data.csv'))\n"
            "```"
        )

        zip_bytes = _make_zip(st.session_state["best_pipeline"], sm, er)
        st.download_button(
            "⬇️ Download model package (.zip)",
            data=zip_bytes,
            file_name=f"automl_{sm.get('final_model','model')}.zip",
            mime="application/zip",
            type="primary",
        )

        st.divider()
        dc1, dc2, dc3, dc4 = st.columns(4)
        dc1.metric("Final model",    sm.get("final_model", ""))
        dc2.metric("Problem type",   sm.get("problem_type", "").capitalize())
        dc3.metric("Metric",         sm.get("metric", ""))
        dc4.metric("Retune rounds",  sm.get("retune_rounds", 0))

# ═════════════════════════════════════════════════════════════════════════════
# PREDICT TAB
# ═════════════════════════════════════════════════════════════════════════════
with tab_predict:
    if not st.session_state["pipeline_done"]:
        st.info("Complete the AutoML pipeline first.")
    else:
        pipeline  = st.session_state["best_pipeline"]
        decision  = st.session_state["decision"]
        df_orig   = st.session_state["df"]
        tgt       = st.session_state["target_col"]
        feat_cols = [c for c in df_orig.columns if c != tgt]

        st.subheader("🔮 Make predictions on new data")
        method = st.radio("Input method", ["Upload CSV", "Manual input"], horizontal=True)

        if method == "Upload CSV":
            pred_file = st.file_uploader(
                "Upload new data CSV (same feature columns, no target needed)",
                type=["csv"], key="pred_upload",
            )
            if pred_file:
                try:
                    df_new = pd.read_csv(pred_file)
                    if tgt in df_new.columns:
                        df_new = df_new.drop(columns=[tgt])
                    missing = [c for c in feat_cols if c not in df_new.columns]
                    if missing:
                        st.warning(f"Missing columns: {missing}")
                    preds    = pipeline.predict(df_new)
                    df_out   = df_new.copy()
                    df_out["prediction"] = preds
                    if decision["problem_type"] == "classification":
                        try:
                            proba = pipeline.predict_proba(df_new)
                            df_out["confidence"] = proba.max(axis=1).round(4)
                        except Exception:
                            pass
                    st.success(f"Predictions complete — {len(df_new):,} rows")
                    st.dataframe(df_out, use_container_width=True)
                    st.download_button(
                        "⬇️ Download predictions CSV",
                        df_out.to_csv(index=False).encode(),
                        "predictions.csv", "text/csv",
                    )
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")

        else:  # Manual input
            st.markdown("Enter values for each feature:")
            input_vals = {}
            cols = st.columns(min(3, len(feat_cols)))
            for i, col_name in enumerate(feat_cols):
                sample = df_orig[col_name].dropna()
                sample_val = sample.iloc[0] if len(sample) else 0
                with cols[i % len(cols)]:
                    if df_orig[col_name].dtype == object:
                        opts = df_orig[col_name].dropna().unique().tolist()
                        input_vals[col_name] = st.selectbox(col_name, opts, key=f"i_{col_name}")
                    elif df_orig[col_name].dtype in [np.float64, np.float32]:
                        input_vals[col_name] = st.number_input(
                            col_name, value=float(sample_val), key=f"i_{col_name}")
                    else:
                        input_vals[col_name] = st.number_input(
                            col_name, value=int(sample_val) if sample_val != "" else 0,
                            step=1, key=f"i_{col_name}")

            if st.button("Predict", type="primary"):
                try:
                    df_single = pd.DataFrame([input_vals])
                    pred      = pipeline.predict(df_single)[0]
                    st.success(f"**Prediction: `{pred}`**")
                    if decision["problem_type"] == "classification":
                        try:
                            proba   = pipeline.predict_proba(df_single)[0]
                            classes = pipeline.classes_
                            fig2, ax2 = plt.subplots(figsize=(6, 3))
                            ax2.barh([str(c) for c in classes], proba, color="#5b6ef5")
                            ax2.set_xlabel("Probability")
                            ax2.set_title("Prediction confidence")
                            st.pyplot(fig2, use_container_width=False)
                        except Exception:
                            pass
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")
