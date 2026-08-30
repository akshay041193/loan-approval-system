import streamlit as st
import pandas as pd
import joblib
import numpy as np

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Loan Approval System",
    page_icon="💰",
    layout="centered"
)

st.title("💰 Loan Approval Prediction System")
st.markdown("Enter the applicant details below to check loan eligibility.")

# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("loan_model.pkl")

model = load_model()

# -----------------------------
# Input Form (no defaults)
# -----------------------------
with st.form("loan_form"):
    st.subheader("Personal Information")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox(
            "Gender",
            options=["Male", "Female"],
            index=None,
            placeholder="Select gender"
        )
        marital_status = st.selectbox(
            "Marital Status",
            options=["Married", "Single"],
            index=None,
            placeholder="Select marital status"
        )
        dependents = st.number_input(
            "Number of Dependents",
            min_value=0,
            max_value=10,
            value=None,
            placeholder="Enter number of dependents"
        )
        education = st.selectbox(
            "Education Level",
            options=["Graduate", "Not Graduate"],
            index=None,
            placeholder="Select education level"
        )
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=70,
            value=None,
            placeholder="Enter age"
        )

    with col2:
        employment_status = st.selectbox(
            "Employment Status",
            options=["Salaried", "Self-employed"],
            index=None,
            placeholder="Select employment status"
        )
        employer_category = st.selectbox(
            "Employer Category",
            options=["Private", "Government", "MNC", "Unemployed"],
            index=None,
            placeholder="Select employer category"
        )
        property_area = st.selectbox(
            "Property Area",
            options=["Urban", "Semiurban", "Rural"],
            index=None,
            placeholder="Select property area"
        )
        loan_purpose = st.selectbox(
            "Loan Purpose",
            options=["Personal", "Car", "Business", "Home"],
            index=None,
            placeholder="Select loan purpose"
        )

    st.subheader("Financial Information")

    col3, col4 = st.columns(2)

    with col3:
        applicant_income = st.number_input(
            "Applicant Income ($)",
            min_value=0,
            value=None,
            placeholder="Enter applicant income"
        )
        coapplicant_income = st.number_input(
            "Coapplicant Income ($)",
            min_value=0,
            value=None,
            placeholder="Enter coapplicant income"
        )
        credit_score = st.number_input(
            "Credit Score",
            min_value=300,
            max_value=850,
            value=None,
            placeholder="Enter credit score"
        )
        existing_loans = st.number_input(
            "Existing Loans",
            min_value=0,
            max_value=10,
            value=None,
            placeholder="Enter number of existing loans"
        )

    with col4:
        dti_ratio = st.number_input(
            "DTI Ratio",
            min_value=0.0,
            max_value=1.0,
            value=None,
            step=0.01,
            placeholder="Enter DTI ratio (e.g. 0.30)"
        )
        savings = st.number_input(
            "Savings ($)",
            min_value=0,
            value=None,
            placeholder="Enter savings"
        )
        collateral_value = st.number_input(
            "Collateral Value ($)",
            min_value=0,
            value=None,
            placeholder="Enter collateral value"
        )
        loan_amount = st.number_input(
            "Loan Amount ($)",
            min_value=1000,
            value=None,
            placeholder="Enter loan amount"
        )
        loan_term = st.selectbox(
            "Loan Term (months)",
            options=[12, 24, 36, 48, 60, 72, 84],
            index=None,
            placeholder="Select loan term"
        )

    submitted = st.form_submit_button("Predict Loan Approval")

# -----------------------------
# Prediction
# -----------------------------
if submitted:

    # 1. Build raw row (NO Applicant_ID)
    raw = pd.DataFrame({
        "Applicant_Income": [applicant_income],
        "Coapplicant_Income": [coapplicant_income],
        "Employment_Status": [employment_status],
        "Age": [age],
        "Marital_Status": [marital_status],
        "Dependents": [dependents],
        "Credit_Score": [credit_score],
        "Existing_Loans": [existing_loans],
        "DTI_Ratio": [dti_ratio],
        "Savings": [savings],
        "Collateral_Value": [collateral_value],
        "Loan_Amount": [loan_amount],
        "Loan_Term": [loan_term],
        "Loan_Purpose": [loan_purpose],
        "Property_Area": [property_area],
        "Education_Level": [education],
        "Gender": [gender],
        "Employer_Category": [employer_category],
    })

    # 2. Label-encode Education_Level (same as notebook)
    raw["Education_Level"] = raw["Education_Level"].map({
        "Graduate": 0,
        "Not Graduate": 1
    }).fillna(0)

    # 3. One-hot encode categorical columns
    cat_features = [
        "Employment_Status", "Marital_Status", "Loan_Purpose",
        "Property_Area", "Gender", "Employer_Category"
    ]
    raw_encoded = pd.get_dummies(raw, columns=cat_features, drop_first=True)

    # 4. Align columns to what the model expects
    expected_cols = list(model.feature_names_in_)
    for col in expected_cols:
        if col not in raw_encoded.columns:
            raw_encoded[col] = 0
    raw_encoded = raw_encoded[expected_cols]

    # 5. Predict
    prediction = model.predict(raw_encoded)[0]
    probability = model.predict_proba(raw_encoded)[0]

    st.markdown("---")
    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("✅ **Loan Approved**")
        st.metric("Confidence", f"{probability[1]*100:.1f}%")
    else:
        st.error("❌ **Loan Rejected**")
        st.metric("Confidence", f"{probability[0]*100:.1f}%")

    st.progress(float(probability[1]), text=f"Approval Probability: {probability[1]*100:.1f}%")