# app_advanced.py
import streamlit as st
import pandas as pd
import numpy as np
import io
import os
from src.utils import load_model, ensure_features, sample_input_csv, plot_confusion, compute_metrics

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide", initial_sidebar_state="expanded")

# --- Sidebar ---
with st.sidebar:
    st.image("assets/CB LOGO 2 png.png", width=140)   # logo path (assets folder)
    st.markdown("**Powered by College Brains**")
    st.markdown("**Author:** Abhiraj Kumar Singh")
    st.write("---")
    st.markdown("**Model file:**")
    st.write("`best_model_gradient_boosting.pkl`")
    st.write("---")
    threshold = st.slider("Probability threshold for fraud", 0.0, 1.0, 0.5, 0.01)
    show_metrics = st.checkbox("Show evaluation metrics (requires 'Class' column in CSV)", value=True)
    st.write("---")
    if st.button("Create sample_input.csv"):
        path = sample_input_csv([
            "Time","V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11",
            "V12","V13","V14","V15","V16","V17","V18","V19","V20","V21","V22",
            "V23","V24","V25","V26","V27","V28","Amount"
        ])
        st.success(f"Sample CSV created: {path}")

st.title("Credit Card Fraud Detection")
st.markdown("Use the controls to predict single transactions or batch CSV files. Set threshold to adjust fraud sensitivity.")

# --- load model (safe)
try:
    model = load_model()
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Model load error: {e}")
    st.stop()

# features list (same as your model expects)
features = [
"Time","V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11",
"V12","V13","V14","V15","V16","V17","V18","V19","V20","V21","V22",
"V23","V24","V25","V26","V27","V28","Amount"
]

# ---------- Single input ----------
st.header("Single Transaction Prediction")
with st.form("single_input"):
    user_input = st.text_input("Enter comma-separated values for features (Time,V1,...,Amount)")
    submitted = st.form_submit_button("Predict single")
if submitted:
    if not user_input.strip():
        st.error("Please enter values.")
    else:
        try:
            vals = [float(x.strip()) for x in user_input.split(",")]
            if len(vals) != len(features):
                st.error(f"Need exactly {len(features)} values.")
            else:
                df_single = pd.DataFrame([vals], columns=features)
                probs = model.predict_proba(df_single)[:,1] if hasattr(model, "predict_proba") else model.decision_function(df_single)
                prob = float(probs[0])
                pred = 1 if prob >= threshold else 0
                st.metric("Fraud probability", f"{prob:.3f}")
                if pred == 1:
                    st.error("⚠️ Fraudulent Transaction Detected")
                else:
                    st.success("✅ Normal Transaction")
        except Exception as e:
            st.error("Error parsing input: " + str(e))

st.write("---")

# ---------- Batch CSV ----------
st.header("Batch Prediction from CSV")
uploaded = st.file_uploader("Upload CSV with required features (or use sample_input.csv)", type=["csv"])

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        if not ensure_features(df, features):
            st.error(f"The CSV must contain all features: {features}")
        else:
            # predict probs if possible
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(df[features])[:,1]
            else:
                # fallback to predict
                probs = np.asarray(model.predict(df[features]), dtype=float)
            preds = (probs >= threshold).astype(int)
            df_out = df.copy()
            df_out["Fraud_Prob"] = probs
            df_out["Prediction"] = ["Fraudulent" if p==1 else "Normal" for p in preds]
            st.success("Predictions ready")
            st.dataframe(df_out.head(200))
            # download
            csv_buf = io.StringIO()
            df_out.to_csv(csv_buf, index=False)
            st.download_button("Download predictions.csv", csv_buf.getvalue(), file_name="predictions.csv")
            # show metrics if true labels exist and checkbox
            if show_metrics and "Class" in df.columns:
                y_true = df["Class"].astype(int).values
                y_pred = preds
                buf = plot_confusion(y_true, y_pred)
                st.image(buf)
                metrics = compute_metrics(y_true, y_pred, y_prob=probs)
                st.json({"roc_auc": metrics["roc_auc"], "classification_report": metrics["report"]})
    except Exception as e:
        st.error("Error processing file: " + str(e))

# ---------- Footer ----------
st.write("---")
st.markdown("**Notes:** Model was trained using SMOTE oversampling and XGBoost (Gradient Boosting).")
st.markdown("**Contact:** Abhiraj Kumar Singh — Powered by College Brains")
