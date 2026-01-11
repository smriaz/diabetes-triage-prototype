import io
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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

    # Read CSV
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
                "Scaled": float(scaled_val),
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
def train_model(df: pd.DataFrame) -> Pipeline:
    pipe = make_pipeline()
    X = df[FEATURES]
    y = df[TARGET].astype(int)
    pipe.fit(X, y)
    return pipe


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
"""
    )

df = get_data()
pipe = train_model(df)

# Sidebar inputs
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

baseline_warnings = format_imputation_warnings(baseline)

# Prepare baseline row
x_base = pd.DataFrame([baseline], columns=FEATURES).copy()
x_base = replace_zeros_with_nan(x_base)

risk_base = predict_risk(pipe, x_base)
contrib_base, _ = compute_logit_contributions(pipe, x_base)

# Layout
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("Risk estimate (baseline)")
    st.metric("Predicted probability", f"{risk_base:.2f}")

    if baseline_warnings:
        st.warning("Missing values detected:\n\n- " + "\n- ".join(baseline_warnings))

    st.caption("Note: This probability is a model output on a public dataset; it is not medical advice.")

    st.subheader("Evidence (feature contributions)")
    st.dataframe(contrib_base, use_container_width=True, hide_index=True)

with col2:
    st.subheader("What-if simulator (modifiable inputs)")
    st.write("Adjust modifiable inputs and compare the new prediction to baseline.")

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
    delta = risk_new - risk_base

    st.metric("New probability", f"{risk_new:.2f}", delta=f"{delta:+.2f}")

    if whatif_warnings:
        st.warning("Missing values detected in what-if inputs:\n\n- " + "\n- ".join(whatif_warnings))

    st.subheader("Evidence (what-if)")
    contrib_new, _ = compute_logit_contributions(pipe, x_whatif)
    st.dataframe(contrib_new, use_container_width=True, hide_index=True)

st.divider()
st.subheader("Limitations (v1)")
st.markdown(
    """
- Uses a simple baseline model trained on a public dataset.
- Some fields use median imputation when missing (zeros treated as missing for certain features).
- This tool is for educational demonstration only.
"""
)
