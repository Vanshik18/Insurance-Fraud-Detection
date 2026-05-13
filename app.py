import streamlit as st
import pickle
import numpy as np

# Load trained model
model = pickle.load(
    open('fraud_model_small.pkl', 'rb')
)

# Page configuration
st.set_page_config(
    page_title="Insurance Fraud Detection",
    page_icon="🚨",
    layout="centered"
)

# Title
st.title("🚨 Insurance Fraud Detection System")

st.write(
    "Predict the probability of fraudulent insurance claims using machine learning."
)

st.markdown("---")

# =========================
# INPUTS
# =========================

fault = st.selectbox(
    "Who Was At Fault?",
    ["Policy Holder", "Third Party"]
)

base_policy = st.selectbox(
    "Base Policy Type",
    ["Liability", "Collision", "All Perils"]
)

deductible = st.number_input(
    "Deductible Amount",
    0,
    100000,
    1000
)

police_report = st.selectbox(
    "Police Report Filed?",
    ["Yes", "No"]
)

accident_area = st.selectbox(
    "Accident Area",
    ["Urban", "Rural"]
)

past_claims = st.number_input(
    "Past Number of Claims",
    0,
    25,
    0
)

# =========================
# ENCODING
# =========================

fault_value = 1 if fault == "Policy Holder" else 0

policy_map = {
    "Liability": 0,
    "Collision": 1,
    "All Perils": 2
}

base_policy_value = policy_map[base_policy]

police_value = 1 if police_report == "Yes" else 0

area_value = 1 if accident_area == "Urban" else 0

# =========================
# PREDICTION
# =========================

if st.button("Predict Fraud Risk"):

    # Input array
    input_data = np.array([
        [
            fault_value,
            base_policy_value,
            deductible,
            police_value,
            area_value,
            past_claims
        ]
    ])

    # Raw model probability
    raw_probability = model.predict_proba(
        input_data
    )[0][1]

    # Calibrated probability
    probability = min(
        (raw_probability ** 0.5) * 1.8,
        1.0
    )

    # Custom fraud threshold
    if probability >= 0.40:
        prediction_label = "Fraudulent Claim"

    else:
        prediction_label = "Genuine Claim"

    st.markdown("---")

    # Show probability
    st.subheader(
        f"Fraud Probability: {probability*100:.2f}%"
    )

    # =========================
    # RISK LEVELS
    # =========================

    if probability < 0.30:
        st.success("✅ Low Fraud Risk")

    elif probability < 0.70:
        st.warning("⚠ Medium Fraud Risk")

    else:
        st.error("🚨 High Fraud Risk")

    # =========================
    # FINAL PREDICTION
    # =========================

    if prediction_label == "Fraudulent Claim":
        st.error("🚨 Fraudulent Claim Detected")

    else:
        st.success("✅ Genuine Claim")