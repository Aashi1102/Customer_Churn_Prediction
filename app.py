import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ── Load artifacts ────────────────────────────────────────────────────────────
model        = joblib.load("churn_model.pkl")
scaler       = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
    layout="wide"
)

st.title("📉 Customer Churn Predictor")
st.write("Fill in the customer details below to predict whether they are likely to churn.")
st.divider()

# ── Input form ────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Demographics")
    gender          = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen  = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner         = st.selectbox("Has Partner", ["No", "Yes"])
    dependents      = st.selectbox("Has Dependents", ["No", "Yes"])

with col2:
    st.subheader("📦 Services")
    phone_service   = st.selectbox("Phone Service", ["No", "Yes"])
    multiple_lines  = st.selectbox("Multiple Lines", ["No", "Yes"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["No", "Yes"])
    online_backup   = st.selectbox("Online Backup", ["No", "Yes"])
    device_protection = st.selectbox("Device Protection", ["No", "Yes"])
    tech_support    = st.selectbox("Tech Support", ["No", "Yes"])
    streaming_tv    = st.selectbox("Streaming TV", ["No", "Yes"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"])

with col3:
    st.subheader("💳 Account & Charges")
    contract        = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    payment_method  = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
    tenure          = st.slider("Tenure (Months)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0, step=0.5)
    total_charges   = st.number_input("Total Charges ($)", 0.0, 10000.0,
                                       float(tenure * monthly_charges), step=10.0)

st.divider()

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("🔍 Predict Churn", use_container_width=True, type="primary"):

    # Helper
    yn = lambda v: 1 if v == "Yes" else 0

    # Engineered features
    charges_per_tenure = monthly_charges / (tenure + 1)
    is_new_customer    = 1 if tenure < 6 else 0
    total_services     = sum([
        yn(phone_service), yn(online_security), yn(online_backup),
        yn(device_protection), yn(tech_support), yn(streaming_tv), yn(streaming_movies)
    ])
    monthly_to_total   = monthly_charges / (total_charges + 1)

    # Raw input dict
    raw = {
        "gender":           1 if gender == "Female" else 0,
        "SeniorCitizen":    yn(senior_citizen),
        "Partner":          yn(partner),
        "Dependents":       yn(dependents),
        "tenure":           tenure,
        "PhoneService":     yn(phone_service),
        "MultipleLines":    yn(multiple_lines),
        "OnlineSecurity":   yn(online_security),
        "OnlineBackup":     yn(online_backup),
        "DeviceProtection": yn(device_protection),
        "TechSupport":      yn(tech_support),
        "StreamingTV":      yn(streaming_tv),
        "StreamingMovies":  yn(streaming_movies),
        "PaperlessBilling": yn(paperless_billing),
        "MonthlyCharges":   monthly_charges,
        "TotalCharges":     total_charges,
        # one-hot: InternetService
        "InternetService_DSL":          1 if internet_service == "DSL" else 0,
        "InternetService_Fiber optic":  1 if internet_service == "Fiber optic" else 0,
        "InternetService_No":           1 if internet_service == "No" else 0,
        # one-hot: Contract
        "Contract_Month-to-month":      1 if contract == "Month-to-month" else 0,
        "Contract_One year":            1 if contract == "One year" else 0,
        "Contract_Two year":            1 if contract == "Two year" else 0,
        # one-hot: PaymentMethod
        "PaymentMethod_Bank transfer (automatic)": 1 if payment_method == "Bank transfer (automatic)" else 0,
        "PaymentMethod_Credit card (automatic)":   1 if payment_method == "Credit card (automatic)" else 0,
        "PaymentMethod_Electronic check":          1 if payment_method == "Electronic check" else 0,
        "PaymentMethod_Mailed check":              1 if payment_method == "Mailed check" else 0,
        # engineered
        "charges_per_tenure":  charges_per_tenure,
        "is_new_customer":     is_new_customer,
        "total_services":      total_services,
        "monthly_to_total":    monthly_to_total,
    }

    # Build dataframe in correct column order
    df_input = pd.DataFrame([raw])[feature_names]

    # Scale numeric cols
    scale_cols = ["tenure", "MonthlyCharges", "TotalCharges", "charges_per_tenure", "monthly_to_total"]
    df_input[scale_cols] = scaler.transform(df_input[scale_cols])

    # Predict
    probability = model.predict_proba(df_input)[0][1]
    prediction  = int(probability >= 0.30)   # same threshold from tuning

    # ── Result ────────────────────────────────────────────────────────────────
    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        if prediction == 1:
            st.error("### ⚠️ Likely to Churn")
        else:
            st.success("### ✅ Likely to Stay")

        st.metric("Churn Probability", f"{probability:.1%}")

    with res_col2:
        st.subheader("📋 Risk Factors")
        risks = []
        if tenure < 6:
            risks.append("🔴 New customer (tenure < 6 months)")
        if contract == "Month-to-month":
            risks.append("🔴 Month-to-month contract (highest churn risk)")
        if internet_service == "Fiber optic":
            risks.append("🟡 Fiber optic internet (higher churn correlation)")
        if payment_method == "Electronic check":
            risks.append("🟡 Electronic check payment method")
        if monthly_charges > 70:
            risks.append(f"🟡 High monthly charges (${monthly_charges:.0f})")
        if total_services <= 1:
            risks.append("🟡 Few add-on services subscribed")
        if online_security == "No" and internet_service != "No":
            risks.append("🟢 No online security add-on")

        if risks:
            for r in risks:
                st.write(r)
        else:
            st.write("✅ No significant risk factors detected.")

    # Probability bar
    st.divider()
    st.write("**Churn Probability Gauge**")
    bar_color = "🟥" if probability > 0.6 else ("🟧" if probability > 0.35 else "🟩")
    filled = int(probability * 20)
    st.write(bar_color * filled + "⬜" * (20 - filled) + f"  {probability:.1%}")
