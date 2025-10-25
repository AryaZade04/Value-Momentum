import streamlit as st
import pandas as pd
import joblib

# ==============================
# Load model and scaler
# ==============================
model = joblib.load("renewal_model.pkl")
scaler = joblib.load("scaler.pkl")

# ==============================
# Define feature names (same order as training)
# ==============================
features = [
    'Distribution_channel', 'Seniority', 'Policies_in_force', 'Max_policies',
    'witnesses', 'police_report_available', 'Payment_history', 'Interaction_score',
    'Age_at_contract', 'Driving_experience', 'Contract_duration', 'Policy_momentum'
]

# ==============================
# Streamlit UI
# ==============================
st.set_page_config(page_title="Insurance Renewal Predictor", page_icon="📈", layout="wide")

st.title("📊 Motor Insurance Renewal Prediction Dashboard")
st.markdown("### Predict if a customer will renew their motor insurance policy.")

# Two-column layout
col1, col2 = st.columns(2)

with col1:
    Distribution_channel = st.number_input("Distribution Channel", min_value=0, max_value=5, step=1)
    Seniority = st.number_input("Seniority (Years)", min_value=0, max_value=50, step=1)
    Policies_in_force = st.number_input("Policies in Force", min_value=0, max_value=10, step=1)
    Max_policies = st.number_input("Max Policies Allowed", min_value=0, max_value=10, step=1)
    witnesses = st.number_input("Number of Witnesses", min_value=0, max_value=10, step=1)
    police_report_available = st.selectbox("Police Report Available", options=['NO', 'YES'])

with col2:
    Payment_history = st.selectbox("Payment History", options=['On-time', 'Delayed'])
    Interaction_score = st.number_input("Interaction Score", min_value=0, max_value=10, step=1)
    Age_at_contract = st.number_input("Age at Contract", min_value=18, max_value=100, step=1)
    Driving_experience = st.number_input("Driving Experience (Years)", min_value=0, max_value=80, step=1)
    Contract_duration = st.number_input("Contract Duration (Days)", min_value=0, max_value=1000, step=1)
    Policy_momentum = st.slider("Policy Momentum (0–1)", 0.0, 1.0, 0.5, step=0.01)

# ==============================
# Prediction Logic
# ==============================
if st.button("🔍 Predict Renewal"):
    # Convert categorical text to numeric (same encoding as training)
    police_report_available_num = 1 if police_report_available == 'YES' else 0
    Payment_history_num = 0 if Payment_history == 'On-time' else 1

    # Prepare input as DataFrame
    data = pd.DataFrame([[
        Distribution_channel, Seniority, Policies_in_force, Max_policies,
        witnesses, police_report_available_num, Payment_history_num, Interaction_score,
        Age_at_contract, Driving_experience, Contract_duration, Policy_momentum
    ]], columns=features)

    # Scale and predict
    scaled = scaler.transform(data)
    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0][1]

    # Show result
    st.markdown("---")
    if pred == 1:
        st.success(f"✅ This customer is **likely to RENEW** (Confidence: {prob:.2%})")
    else:
        st.error(f"❌ This customer is **unlikely to renew** (Confidence: {prob:.2%})")
