# Diabetes Risk Triage Prototype (Tabular)

**Educational research prototype — not medical advice.**

This repository contains a Streamlit-based research prototype that demonstrates how
a lightweight machine-learning model can be combined with explainability and
counterfactual “what-if” analysis for **risk triage on tabular data**.

The application is designed for **education, research, and system design exploration**.
It does **not** provide medical advice, diagnosis, or treatment.

---

## 🔍 What this project demonstrates

- End-to-end ML pipeline using a public dataset
- Transparent, model-faithful explanations (feature contributions)
- Interactive *what-if* simulation on modifiable inputs
- Responsible framing of ML outputs (uncertainty, limitations)
- Reproducible deployment using Streamlit Community Cloud

This project focuses on **how models are integrated and explained**, not on achieving
state-of-the-art clinical performance.

---

## 🧠 Dataset

- **Name:** Pima Indians Diabetes Dataset  
- **Type:** Tabular, binary classification  
- **Target:** Diabetes onset (`Outcome`)  
- **Source:** Publicly available research dataset (widely used in ML education)

### Input features
- Pregnancies
- Glucose (mg/dL)
- Blood Pressure (mm Hg)
- Skin Thickness (mm)
- Insulin (μU/mL)
- BMI (kg/m²)
- Diabetes Pedigree Function
- Age (years)

**Note:** In this dataset, zero values for some features represent missing data.
These are treated as missing and imputed using the median of the training data.

---

## ⚙️ Model

- **Algorithm:** Logistic Regression
- **Preprocessing:**
  - Median imputation for missing values
  - Standardization (z-score scaling)
- **Training:** Stratified train/test split (80/20)
- **Evaluation metric:** ROC-AUC on holdout test set

The model is intentionally simple to prioritize **interpretability and reproducibility**.

---

## 🧾 Explanation method

For a given input, the app computes **local feature contributions** using the model’s
actual parameters:

\[
\text{contribution}_i = \beta_i \times x_i^{(scaled)}
\]

Where:
- \(\beta_i\) is the learned coefficient
- \(x_i^{(scaled)}\) is the standardized feature value

This produces **faithful, model-derived explanations** rather than post-hoc approximations.

**Important interpretation note:**  
Contributions are relative to the dataset baseline (due to standardization).  
A feature may appear to “decrease risk” if its value is *lower than the dataset mean*.

---

## 🔁 What-if simulation

The app allows interactive modification of **modifiable inputs only**:
- Glucose
- BMI
- Blood Pressure

All other inputs (e.g., age, pregnancies) are held constant.

Each change:
- Re-runs the model
- Updates the risk probability
- Updates feature contributions
- Shows the delta relative to baseline

This enables **counterfactual reasoning** without implying real-world intervention advice.

---

## 🚦 Decision threshold (demo only)

Users can adjust a probability threshold to see how a simple triage flag would change.

> This threshold is included for educational purposes only and has no clinical meaning.

---

## 📊 Model performance (holdout)

Performance metrics are computed on a held-out test split for transparency:
- ROC-AUC
- Confusion matrix (reference threshold = 0.5)

These metrics are shown to illustrate evaluation practice, **not clinical validity**.

---

## 🖥️ Running the app

### Hosted
The app is designed to run directly on **Streamlit Community Cloud** from this repository.

### Files required
```text
data/pima_diabetes.csv
assets/disclaimer.md
app.py
requirements.txt
