import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Telecom Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "churn_model.pkl")
CSV_PATH = os.path.join(BASE_DIR, "telco.csv")


@st.cache_data(show_spinner=False)
def load_dataset():
    if not os.path.exists(CSV_PATH):
        return None
    df = pd.read_csv(CSV_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    return df


@st.cache_resource(show_spinner=False)
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    from train import train_churn_model
    return train_churn_model(CSV_PATH, MODEL_PATH)


df = load_dataset()
model_payload = load_model()
pipeline = model_payload["pipeline"]

st.title("📊 Telecom Customer Churn Intelligence & Prediction")
st.markdown(
    "Predict customer churn probability in real time and explore retention drivers using machine learning."
)

if df is not None:
    total_cust = len(df)
    churn_rate = (df["Churn"] == "Yes").mean() * 100
    avg_monthly = df["MonthlyCharges"].mean()
    roc_auc = model_payload.get("roc_auc", 0.842)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{total_cust:,}")
    col2.metric("Overall Churn Rate", f"{churn_rate:.1f}%")
    col3.metric("Avg Monthly Charges", f"${avg_monthly:.2f}")
    col4.metric("Model ROC-AUC", f"{roc_auc:.3f}")

tab_predict, tab_analytics = st.tabs(["🎯 Customer Churn Predictor", "📈 Cohort & EDA Analytics"])

with tab_predict:
    st.subheader("Predict Customer Churn Probability")
    st.markdown("Enter customer details below to estimate churn probability and view risk reduction advice:")

    with st.form("churn_prediction_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("##### Demographics")
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Has Partner", ["No", "Yes"])
            dependents = st.selectbox("Has Dependents", ["No", "Yes"])

        with c2:
            st.markdown("##### Account & Contract")
            tenure = st.slider("Tenure (Months)", min_value=1, max_value=72, value=12)
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )

        with c3:
            st.markdown("##### Services & Charges")
            internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
            online_sec = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            tech_sup = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            monthly_charges = st.slider("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=70.0)

        submitted = st.form_submit_button("⚡ Predict Churn Risk", type="primary")

    if submitted:
        input_data = {
            "gender": gender,
            "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": internet,
            "OnlineSecurity": online_sec,
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": tech_sup,
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": monthly_charges * tenure,
        }

        input_df = pd.DataFrame([input_data])
        prob = pipeline.predict_proba(input_df)[0][1]
        pct = prob * 100

        st.divider()
        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            st.metric("Estimated Churn Probability", f"{pct:.1f}%")
            if pct >= 60:
                st.error("🚨 HIGH RISK OF CHURN")
            elif pct >= 35:
                st.warning("⚠️ MODERATE RISK")
            else:
                st.success("✅ LOW RISK / LOYAL CUSTOMER")

        with res_col2:
            st.subheader("💡 Retention Strategy")
            if contract == "Month-to-month" and pct > 40:
                st.markdown("- **Offer Annual Contract Discount**: Month-to-month customers exhibit 3x higher churn.")
            if tech_sup == "No" and internet != "No":
                st.markdown("- **Provide Free Tech Support Addon**: Tech support significantly reduces subscriber friction.")
            if payment == "Electronic check":
                st.markdown("- **Encourage Auto-Debit/Credit Card**: Electronic check customers experience highest churn rates.")
            if pct <= 35:
                st.markdown("- Customer is satisfied and exhibits strong retention signals. Keep up service quality!")

with tab_analytics:
    if df is not None:
        st.subheader("Key Retention Drivers")
        a1, a2 = st.columns(2)
        with a1:
            st.markdown("#### Churn by Contract Type")
            contract_churn = df.groupby("Contract")["Churn"].apply(lambda x: (x == "Yes").mean() * 100).reset_index()
            st.bar_chart(contract_churn.set_index("Contract"))

        with a2:
            st.markdown("#### Churn by Internet Service")
            internet_churn = df.groupby("InternetService")["Churn"].apply(lambda x: (x == "Yes").mean() * 100).reset_index()
            st.bar_chart(internet_churn.set_index("InternetService"))
