from pathlib import Path
from typing import Dict, List, Tuple
import json

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Milestone 2: LLM narrative layer (requires llm_narrative.py + OPENAI_API_KEY)
from llm_narrative import build_prompt, generate_narrative

# -----------------------------
# Config
# -----------------------------
FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
TARGET = "Outcome"

# Features where 0 in the dataset typically means "missing"
ZERO_MEANS_MISSING = {"Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"}

# v1: modifiable subset for what-if
MODIFIABLE = ["Glucose", "BMI", "BloodPressure"]

DATASET_PATH = Path("data/pima_diabetes.csv")
DISCLAIMER_PATH = Path("assets/disclaimer.md")

st.set_page_config(page_title="Diabetes Risk Triage Prototype", layout="wide")


# -----------------------------
# Utilities
# -----------------------------
def load_disclaimer() -> str:
    if DISCLAIMER_PATH.exists():
        return DISCLAIMER_PATH.read_text(encoding="utf-8")
    return "**Educational research prototype — not medical advice.**"


def load_dataset_from_repo() -> pd.DataFrame:
    """
    Load dataset from repo at data/pima_diabetes.csv.
    Expected:
      - Either headers include FEATURES + Outcome
      - Or no header with exactly 9 columns in the classic order
    """
    if not DATASET_PATH.exists():
        st.error(
            "Missing dataset file: data/pima_diabetes.csv\n\n"
            "Add the Pima dataset to your repo at that path, then redeploy."
        )
        st.stop()

    df = pd.read_csv(DATASET_PATH)

    expected = FEATURES + [TARGET]
    if set(expected).issubset(set(df.columns)):
        df = df[expected].copy()
        return df

    # If no headers but 9 columns
    if df.shape[1] == 9:
        df.columns = expected
        return df

    st.error(
        "Dataset format unexpected.\n\n"
        "Expected either:\n"
        "- Columns named: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, "
        "DiabetesPedigreeFunction, Age, Outcome\n"
        "- OR a headerless CSV with exactly 9 columns in that order."
    )
    st.stop()


def replace_zeros_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    for col in ZERO_MEANS_MISSING:
        if col in df2.columns:
            df2[col] = df2[col].replace(0, np.nan)
    return df2


def make_pipeline() -> Pipeline:
    """
    Pipeline:
      - Median imputation
      - Standard scaling
      - Logistic regression
    Note: zeros-as-missing is handled BEFORE the pipeline.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")

    return Pipeline(steps=[("preprocess", preprocessor), ("clf", clf)])


def predict_risk(pipe: Pipeline, x_row: pd.DataFrame) -> float:
    return float(pipe.predict_proba(x_row)[0, 1])


def compute_logit_contributions(
    pipe: Pipeline, x_row: pd.DataFrame
) -> Tuple[pd.DataFrame, float]:
    """
    Faithful local explanation for standardized logistic regression:

    logit = intercept + sum_i coef_i * x_i_scaled
    contribution_i = coef_i * x_i_scaled
    """
    pre = pipe.named_steps["preprocess"]
    clf = pipe.named_steps["clf"]

    x_scaled = np.asarray(pre.transform(x_row)).reshape(-1)
    coefs = clf.coef_.reshape(-1)
    intercept = float(clf.intercept_.reshape(-1)[0])

    records = []
    for feat, raw_val, scaled_val, coef in zip(
        FEATURES,
        x_row.iloc[0].values,
        x_scaled,
        coefs,
    ):
        contrib = float(coef * scaled_val)
        if contrib > 0:
            direction = "↑ increases risk"
        elif contrib < 0:
            direction = "↓ decreases risk"
        else:
            direction = "— neutral"

        records.append(
            {
                "Feature": feat,
                "Value": raw_val,
                "Scaled (z)": float(scaled_val),
                "Contribution (log-odds)": contrib,
                "Effect": direction,
            }
        )

    dfc = pd.DataFrame(records)
    dfc["Abs(contribution)"] = dfc["Contribution (log-odds)"].abs()
    dfc = dfc.sort_values("Abs(contribution)", ascending=False).drop(
        columns=["Abs(contribution)"]
    )
    return dfc, intercept


def format_imputation_warnings(x_input: Dict[str, float]) -> List[str]:
    warnings = []
    for col in ZERO_MEANS_MISSING:
        if col in x_input and (x_input[col] == 0):
            warnings.append(f"{col} missing → imputed (median)")
    return warnings


def top_drivers(contrib_df: pd.DataFrame, k: int = 3) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return top k positive and top k negative contributors."""
    up = contrib_df[contrib_df["Contribution (log-odds)"] > 0].sort_values(
        "Contribution (log-odds)", ascending=False
    ).head(k)
    down = contrib_df[contrib_df["Contribution (log-odds)"] < 0].sort_values(
        "Contribution (log-odds)", ascending=True
    ).head(k)
    return up, down


# -----------------------------
# Cached data/model
# -----------------------------
@st.cache_data(show_spinner=True)
def get_data() -> pd.DataFrame:
    df = load_dataset_from_repo()
    df = df[FEATURES + [TARGET]].copy()
    df = replace_zeros_with_nan(df)
    return df


@st.cache_resource(show_spinner=True)
def train_and_eval(df: pd.DataFrame):
    """
    Train on train split and evaluate on test split (holdout) to avoid in-sample optimism.
    Returns:
      - trained pipeline
      - metrics dict
    """
    X = df[FEATURES]
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipe = make_pipeline()
    pipe.fit(X_train, y_train)

    proba_test = pipe.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, proba_test))

    # confusion matrix at default threshold 0.5 (reported for reference)
    pred_test = (proba_test >= 0.5).astype(int)
    cm = confusion_matrix(y_test, pred_test)

    metrics = {
        "auc_test": auc,
        "cm_test": cm.tolist(),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "pos_rate": float(y.mean()),
    }
    return pipe, metrics


# -----------------------------
# UI
# -----------------------------
st.title("Diabetes Risk Triage Prototype (Tabular)")
st.markdown(load_disclaimer())

with st.expander("How it works (short)"):
    st.markdown(
        """
- Trains a lightweight Logistic Regression model on a public diabetes dataset.
- Treats zeros in some fields as missing and imputes using the dataset median.
- Produces a risk probability and a *faithful* feature-contribution explanation.
- Provides a what-if panel for modifiable inputs (glucose, BMI, blood pressure).
- (Optional) Generates an LLM narrative explanation that is strictly non-diagnostic.
"""
    )

df = get_data()
pipe, metrics = train_and_eval(df)

# Sidebar inputs + threshold
st.sidebar.header("Input (baseline)")
baseline: Dict[str, float] = {}

baseline["Pregnancies"] = st.sidebar.number_input(
    "Pregnancies", min_value=0, max_value=20, value=2, step=1
)
baseline["Glucose"] = st.sidebar.number_input(
    "Glucose (mg/dL)", min_value=0, max_value=300, value=120, step=1
)
baseline["BloodPressure"] = st.sidebar.number_input(
    "Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70, step=1
)
baseline["SkinThickness"] = st.sidebar.number_input(
    "Skin Thickness (mm)", min_value=0, max_value=100, value=20, step=1
)
baseline["Insulin"] = st.sidebar.number_input(
    "Insulin (mu U/mL)", min_value=0, max_value=900, value=80, step=1
)
baseline["BMI"] = st.sidebar.number_input(
    "BMI (kg/m²)", min_value=0.0, max_value=80.0, value=30.0, step=0.1
)
baseline["DiabetesPedigreeFunction"] = st.sidebar.number_input(
    "Diabetes Pedigree (unitless)", min_value=0.0, max_value=5.0, value=0.5, step=0.01
)
baseline["Age"] = st.sidebar.number_input(
    "Age (years)", min_value=1, max_value=120, value=33, step=1
)

st.sidebar.divider()
threshold = st.sidebar.slider("Decision threshold", 0.05, 0.95, 0.50, 0.01)
st.sidebar.caption("This threshold is for demo triage logic only.")

# Milestone 2: LLM toggle
st.sidebar.divider()
use_llm = st.sidebar.checkbox("Generate narrative explanation (LLM)", value=False)
st.sidebar.caption("Explanatory only. No diagnosis or medical advice.")

baseline_warnings = format_imputation_warnings(baseline)

# Prepare baseline row
x_base = pd.DataFrame([baseline], columns=FEATURES).copy()
x_base = replace_zeros_with_nan(x_base)

risk_base = predict_risk(pipe, x_base)
contrib_base, _ = compute_logit_contributions(pipe, x_base)
up3_base, down3_base = top_drivers(contrib_base, k=3)

flag_base = "Flagged (≥ threshold)" if risk_base >= threshold else "Not flagged (< threshold)"

tabs = st.tabs(["Baseline", "What-if", "Narrative", "Model & Limitations"])

# -----------------------------
# Baseline Tab
# -----------------------------
with tabs[0]:
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.subheader("Risk estimate (baseline)")
        st.metric("Predicted probability", f"{risk_base:.2f}")
        st.write(f"**Status @ threshold {threshold:.2f}:** {flag_base}")

        if baseline_warnings:
            st.warning("Missing values detected:\n\n- " + "\n- ".join(baseline_warnings))

        st.caption("Note: This probability is a model output on a public dataset; it is not medical advice.")

        st.subheader("Key drivers (baseline)")
        cA, cB = st.columns(2)
        with cA:
            st.markdown("**↑ Increasing risk**")
            if len(up3_base) == 0:
                st.write("- (none)")
            else:
                for _, r in up3_base.iterrows():
                    st.write(f"- {r['Feature']} ({r['Value']})")
        with cB:
            st.markdown("**↓ Decreasing risk**")
            if len(down3_base) == 0:
                st.write("- (none)")
            else:
                for _, r in down3_base.iterrows():
                    st.write(f"- {r['Feature']} ({r['Value']})")

    with right:
        st.subheader("Evidence (feature contributions)")
        st.caption(
            "Interpretation note: contributions are computed on standardized inputs (z-scores). "
            "A negative contribution means this value is lower than the dataset baseline in a way that reduces the model’s predicted risk."
        )
        with st.expander("Show full evidence table", expanded=True):
            st.dataframe(contrib_base, use_container_width=True, hide_index=True)

# -----------------------------
# What-if Tab
# -----------------------------
with tabs[1]:
    st.subheader("What-if simulator (modifiable inputs)")
    st.write("Adjust modifiable inputs and compare the new prediction to baseline.")

    w_col1, w_col2 = st.columns([1, 1], gap="large")

    with w_col1:
        w_glucose = st.slider(
            "Glucose (mg/dL)",
            min_value=0,
            max_value=300,
            value=int(baseline["Glucose"]),
            step=1,
        )
        w_bmi = st.slider(
            "BMI (kg/m²)",
            min_value=0.0,
            max_value=80.0,
            value=float(baseline["BMI"]),
            step=0.1,
        )
        w_bp = st.slider(
            "Blood Pressure (mm Hg)",
            min_value=0,
            max_value=200,
            value=int(baseline["BloodPressure"]),
            step=1,
        )

    whatif = baseline.copy()
    whatif["Glucose"] = w_glucose
    whatif["BMI"] = w_bmi
    whatif["BloodPressure"] = w_bp

    whatif_warnings = format_imputation_warnings(whatif)

    x_whatif = pd.DataFrame([whatif], columns=FEATURES).copy()
    x_whatif = replace_zeros_with_nan(x_whatif)

    risk_new = predict_risk(pipe, x_whatif)
    delta = float(risk_new - risk_base)
    flag_new = "Flagged (≥ threshold)" if risk_new >= threshold else "Not flagged (< threshold)"

    contrib_new, _ = compute_logit_contributions(pipe, x_whatif)
    up3_new, down3_new = top_drivers(contrib_new, k=3)

    with w_col2:
        st.subheader("Result (what-if)")
        st.metric("New probability", f"{risk_new:.2f}", delta=f"{delta:+.2f}")
        st.write(f"**Status @ threshold {threshold:.2f}:** {flag_new}")

        if whatif_warnings:
            st.warning("Missing values detected in what-if inputs:\n\n- " + "\n- ".join(whatif_warnings))

        st.subheader("Key drivers (what-if)")
        cA, cB = st.columns(2)
        with cA:
            st.markdown("**↑ Increasing risk**")
            if len(up3_new) == 0:
                st.write("- (none)")
            else:
                for _, r in up3_new.iterrows():
                    st.write(f"- {r['Feature']} ({r['Value']})")
        with cB:
            st.markdown("**↓ Decreasing risk**")
            if len(down3_new) == 0:
                st.write("- (none)")
            else:
                for _, r in down3_new.iterrows():
                    st.write(f"- {r['Feature']} ({r['Value']})")

    st.subheader("Evidence (what-if)")
    st.caption(
        "Interpretation note: contributions are computed on standardized inputs (z-scores). "
        "Compare baseline vs what-if to see which factors changed the most."
    )
    with st.expander("Show full what-if evidence table", expanded=True):
        st.dataframe(contrib_new, use_container_width=True, hide_index=True)

    # Download report
    report = {
        "disclaimer": "Educational research prototype — not medical advice.",
        "threshold": float(threshold),
        "baseline": {
            "inputs": {k: float(v) for k, v in baseline.items()},
            "risk_probability": float(risk_base),
            "status": flag_base,
            "missing_imputed": baseline_warnings,
            "top_drivers_up": [
                {"feature": r["Feature"], "value": float(r["Value"]), "contribution_logodds": float(r["Contribution (log-odds)"])}
                for _, r in up3_base.iterrows()
            ],
            "top_drivers_down": [
                {"feature": r["Feature"], "value": float(r["Value"]), "contribution_logodds": float(r["Contribution (log-odds)"])}
                for _, r in down3_base.iterrows()
            ],
        },
        "what_if": {
            "inputs": {k: float(v) for k, v in whatif.items()},
            "risk_probability": float(risk_new),
            "delta": float(delta),
            "status": flag_new,
            "missing_imputed": whatif_warnings,
            "top_drivers_up": [
                {"feature": r["Feature"], "value": float(r["Value"]), "contribution_logodds": float(r["Contribution (log-odds)"])}
                for _, r in up3_new.iterrows()
            ],
            "top_drivers_down": [
                {"feature": r["Feature"], "value": float(r["Value"]), "contribution_logodds": float(r["Contribution (log-odds)"])}
                for _, r in down3_new.iterrows()
            ],
        },
        "model_eval_holdout": metrics,
    }

    st.download_button(
        "Download JSON report",
        data=json.dumps(report, indent=2),
        file_name="diabetes_triage_report.json",
        mime="application/json",
    )

# -----------------------------
# Narrative Tab (Milestone 2)
# -----------------------------
with tabs[2]:
    st.subheader("Narrative explanation (LLM-assisted)")
    st.caption("This section explains the model output using only the evidence above. No medical advice is provided.")

    if not use_llm:
        st.info("Enable the LLM narrative from the sidebar to generate an explanation.")
    else:
        # Prefer what-if results if user has visited the What-if tab; otherwise fall back to baseline
        has_whatif = "risk_new" in locals()

        chosen_risk = risk_new if has_whatif else risk_base
        chosen_flag = flag_new if has_whatif else flag_base
        chosen_warnings = whatif_warnings if has_whatif else baseline_warnings

        # Use baseline drivers for now; if you want, switch to up3_new/down3_new when has_whatif
        chosen_up = up3_new if has_whatif else up3_base
        chosen_down = down3_new if has_whatif else down3_base

        prompt = build_prompt(
            risk=chosen_risk,
            threshold=threshold,
            flag=chosen_flag,
            top_up=[f"{r['Feature']} ({r['Value']})" for _, r in chosen_up.iterrows()],
            top_down=[f"{r['Feature']} ({r['Value']})" for _, r in chosen_down.iterrows()],
            imputation_notes=chosen_warnings,
        )

        with st.spinner("Generating explanation..."):
            try:
                narrative = generate_narrative(prompt)

                # Simple post-guardrail (optional but recommended)
                forbidden = ["diagnose", "treat", "should", "recommend", "medication", "doctor", "therapy"]
                if any(w in narrative.lower() for w in forbidden):
                    st.warning("Generated text exceeded safety bounds. Showing a neutral limitation statement instead.")
                    st.markdown(
                        "This is an educational model output based on limited features from a public dataset. "
                        "It cannot be used for medical decisions."
                    )
                else:
                    st.markdown(narrative)

                with st.expander("Show LLM prompt (for transparency)"):
                    st.code(prompt)

            except Exception:
                st.warning(
                    "Narrative generation is unavailable (missing API key or network issue). "
                    "Core model outputs are unaffected."
                )
                st.markdown(
                    "This is an educational model output based on limited features from a public dataset. "
                    "It cannot be used for medical decisions."
                )

# -----------------------------
# Model & Limitations Tab
# -----------------------------
with tabs[3]:
    st.subheader("Model performance (holdout test split)")
    st.write(f"- Train size: **{metrics['n_train']}**")
    st.write(f"- Test size: **{metrics['n_test']}**")
    st.write(f"- Positive rate in dataset: **{metrics['pos_rate']:.3f}**")
    st.write(f"- ROC-AUC (test): **{metrics['auc_test']:.3f}**")

    cm = np.array(metrics["cm_test"])
    cm_df = pd.DataFrame(cm, index=["True 0", "True 1"], columns=["Pred 0", "Pred 1"])
    st.write("Confusion matrix on test split at threshold 0.5 (reference):")
    st.dataframe(cm_df, use_container_width=True)

    st.subheader("Limitations (v1 + narrative layer)")
    st.markdown(
        """
- Uses a simple baseline Logistic Regression model trained on a public dataset.
- Some fields use median imputation when missing (zeros treated as missing for certain features).
- Holdout metrics are provided for transparency, but performance will vary by population and data quality.
- The LLM narrative is constrained to explanation and questions only; it is not permitted to provide medical advice.
- This tool is for educational demonstration only; it does not provide medical advice.
"""
    )

st.caption("© Research prototype demo. Add dataset citation and a model card in the repository for completeness.")
