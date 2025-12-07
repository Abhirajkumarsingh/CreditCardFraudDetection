import streamlit as st
import pandas as pd
import joblib
import io

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------- CUSTOM CSS ----------------------
st.markdown("""
    <style>
        .title {
            font-size: 40px;
            font-weight: 700;
            color: #FF0080;
            text-align: center;
        }
        .subtitle {
            font-size: 18px;
            text-align: center;
            color: #bbbbbb;
        }
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #111;
            padding: 10px;
            color: #FF0080;
            text-align: center;
            font-size: 16px;
        }
        .box {
            background-color: #1a1a1a;
            padding: 18px;
            border-radius: 10px;
            border: 1px solid #FF0080;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------- HEADER ----------------------
st.markdown("<div class='title'>💳 Credit Card Fraud Detection System</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Machine Learning-based Fraud Classification Model</div>", unsafe_allow_html=True)

st.write("")

# ---------------------- SIDEBAR ----------------------
st.sidebar.header("📌 Instructions")
st.sidebar.info("""
**How to Use the App:**

1️⃣ *Single Transaction Prediction*  
→ Enter values comma-separated  
→ App predicts Fraud / Normal

2️⃣ *Batch Prediction*  
→ Upload CSV with same columns  
→ Download results with prediction

---

**Developer:** *Abhiraj Kumar Singh*  
**Powered By:** *College Brains*
""")

# ---------------------- LOAD MODEL ----------------------
try:
    load = joblib.load("best_model_gradient_boosting.pkl")
    model = load['model']
except:
    st.error("❌ Model file not found! Make sure best_model_gradient_boosting.pkl exists.")
    st.stop()

# Feature list
features = [
"Time","V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11",
"V12","V13","V14","V15","V16","V17","V18","V19","V20","V21","V22",
"V23","V24","V25","V26","V27","V28","Amount"
]

# ---------------------- SINGLE INPUT PREDICTION ----------------------
st.markdown("### 🔍 Single Transaction Prediction")

with st.container():
    st.markdown("<div class='box'>", unsafe_allow_html=True)
    
    with st.form("single_input_form"):
        user_input_str = st.text_input(
            "Enter all feature values (comma-separated):",
            placeholder="Example: 0, -1.23, 2.45, ..., 120.55"
        )
        submitted = st.form_submit_button("🔮 Predict")

    if submitted:
        try:
            user_input_list = [float(x.strip()) for x in user_input_str.split(",")]

            if len(user_input_list) != len(features):
                st.error(f"⚠️ Please enter exactly {len(features)} values.")
            else:
                input_df = pd.DataFrame([user_input_list], columns=features)
                pred = model.predict(input_df)[0]

                if pred == 1:
                    st.error("🚨 **Fraudulent Transaction Detected**")
                else:
                    st.success("🟢 **Normal Transaction**")

        except:
            st.error("❌ Invalid input format! Please enter only numeric values.")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------- BATCH PREDICTION ----------------------
st.markdown("### 📂 Batch Prediction (Upload CSV)")

st.markdown("<div class='box'>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload a CSV file:", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    missing = set(features) - set(df.columns)

    if missing:
        st.error(f"❌ CSV is missing required columns: {missing}")
    else:
        predictions = model.predict(df[features])
        df["Prediction"] = ["⚠️ Fraud" if p == 1 else "✔ Normal" for p in predictions]

        st.success("✅ Prediction completed successfully!")
        st.dataframe(df)

        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)

        st.download_button(
            label="⬇ Download Results CSV",
            data=csv_buffer.getvalue(),
            file_name="fraud_predictions.csv",
            mime="text/csv"
        )

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------- FOOTER ----------------------
st.markdown("""
    <div class='footer'>
        🚀 Powered by <b>College Brains</b> | Developed by <b>Abhiraj Kumar Singh</b>
    </div>
""", unsafe_allow_html=True)
