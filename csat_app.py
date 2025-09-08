# csat_app.py — Hugging Face CSV + Streamlit Cloud Compatible

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import shap
import matplotlib.pyplot as plt

# -----------------------------
# Configs
# -----------------------------
CSV_URL       = "https://huggingface.co/datasets/sharmila122125/flipkart-csat-files/resolve/main/step5_sentiment_features.csv"
MODEL_PATH    = "models/best_model_XGBoost.pkl"
SCALER_PATH   = "models/scaler.pkl"
FEATURES_PATH = "models/features_used.pkl"

# -----------------------------
# Load Artifacts
# -----------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    features = joblib.load(FEATURES_PATH)
    return model, scaler, features

model, scaler, features = load_artifacts()

# -----------------------------
# UI Setup
# -----------------------------
st.set_page_config("CSAT Prediction", layout="wide")
st.markdown("##  Flipkart Customer Satisfaction — Hugging Face CSV Prediction")
st.markdown(f"""
Model: ` {os.path.basename(MODEL_PATH)}`  
CSV Source: Hugging Face  
""")

threshold = st.slider(" Threshold (Satisfaction ≥ this value is 1)", 0.0, 1.0, 0.5, 0.01)

# -----------------------------
# Load CSV from Hugging Face
# -----------------------------
try:
    hf_url = "https://huggingface.co/datasets/sharmila122125/customer-support-csat/resolve/main/step5_sentiment_features.csv"
    df = pd.read_csv(hf_url)

    st.success(f" File Loaded from Hugging Face: {df.shape[0]} rows × {df.shape[1]} columns")

    st.subheader(" Preview of Input Data")
    st.dataframe(df.head(15), use_container_width=True)

    # -----------------------------
    # Feature Selection & Scaling
    # -----------------------------
    X = df.copy()
    missing = [f for f in features if f not in X.columns]
    if missing:
        st.error(f" Missing features in file: {missing}")
        st.stop()

    X = X[features]
    if scaler:
        X = scaler.transform(X)

    # -----------------------------
    # Predict
    # -----------------------------
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)

    df["Satisfaction_Prob"]       = probs.round(5)
    df["Predicted_Label"]         = preds
    df["Satisfaction_Prediction"] = df["Predicted_Label"].map({0: "Unsatisfied", 1: "Satisfied"})

    # -----------------------------
    # Summary Stats
    # -----------------------------
    st.subheader(" Prediction Summary")
    satisfied_count   = int((preds == 1).sum())
    unsatisfied_count = int((preds == 0).sum())

    col1, col2 = st.columns(2)
    col1.success(f" Satisfied: {satisfied_count}")
    col2.error(f" Unsatisfied: {unsatisfied_count}")

    # -----------------------------
    # Display Results
    # -----------------------------
    st.subheader(" Prediction Results")
    styled_df = df[["Satisfaction_Prob", "Predicted_Label", "Satisfaction_Prediction"]].copy()
    styled_df["Satisfaction_Prediction"] = styled_df["Satisfaction_Prediction"].map({
        "Satisfied": f'<span style="color:green;font-weight:bold;">Satisfied</span>',
        "Unsatisfied": f'<span style="color:red;font-weight:bold;">Unsatisfied</span>'
    })

    st.write(styled_df.to_html(escape=False, index=False), unsafe_allow_html=True)

    # -----------------------------
    # Download Button
    # -----------------------------
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"csat_predictions_{now}.csv"
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(" Download Full Predictions as CSV", data=csv_bytes, file_name=filename, mime="text/csv")

except Exception as e:
    st.error(f" Error loading data: {e}")

# -----------------------------
# SHAP Explainability Section
# -----------------------------
with st.expander(" Show SHAP Explainability (First 500 rows)", expanded=False):
    st.info("Explaining predictions using SHAP (may take a few seconds)...")

    try:
        X_shap = X[:500]
        explainer = shap.Explainer(model, feature_names=features)
        shap_values = explainer(X_shap)

        st.subheader(" SHAP Summary Plot")
        fig = plt.figure()
        shap.plots.beeswarm(shap_values, show=False)
        st.pyplot(fig)

    except Exception as e:
        st.error(f" SHAP Error: {e}")
