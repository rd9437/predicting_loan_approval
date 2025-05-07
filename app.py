import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import streamlit as st
import os

st.set_page_config(page_title="Loan Approval Prediction", layout="wide")

# ----------------- Model Training -----------------

@st.cache_resource
def train_or_load_model():
    # Check if model already exists
    if not os.path.exists("loan_model.pkl"):
        print("Training a new model...")
        # Load dataset
        dataset = pd.read_csv("loan_approval_dataset.csv")
        dataset.columns = dataset.columns.str.strip()
        
        # Drop unnecessary column and clean data
        dataset.drop(columns="loan_id", inplace=True)
        dataset['education'] = dataset['education'].str.strip()
        dataset['education'] = dataset['education'].map({'Not Graduate': 0, 'Graduate': 1})
        
        # Label encoding
        la = LabelEncoder()
        dataset["self_employed"] = la.fit_transform(dataset["self_employed"].astype(str))
        dataset["loan_status"] = la.fit_transform(dataset["loan_status"].astype(str))

        # Feature and target columns
        X = dataset.drop(columns=["loan_status"])
        y = dataset["loan_status"]

        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize and train model
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        # Save the trained model
        joblib.dump(model, "loan_model.pkl")

        # Model Evaluation
        y_pred = model.predict(X_test)
        print("Model Trained.")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))
    else:
        print("Loading existing model...")

    # Return the model
    return joblib.load("loan_model.pkl")

# ----------------- Streamlit App -----------------

model = train_or_load_model()

st.markdown("""
    <style>
    .main-title {
        font-size: 36px;
        font-weight: bold;
        color: #2e6c80;
        text-align: center;
        margin-top: 20px;
    }
    .sub-title {
        font-size: 18px;
        text-align: center;
        color: #666;
    }
    .stButton>button {
        background-color: #2e6c80;
        color: white;
        font-size: 18px;
        border-radius: 5px;
        height: 50px;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #1e4c61;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🏦 Loan Approval Prediction App</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Fill in the details below to see if your loan is likely to be approved.</div>', unsafe_allow_html=True)

with st.form("loan_input_form"):
    education = st.selectbox("Education", ["Graduate", "Not Graduate"], index=0)
    self_employed = st.selectbox("Self Employed", ["Yes", "No"], index=0)
    cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900)
    income_annum = st.number_input("Annual Income", min_value=0)
    loan_amount = st.number_input("Loan Amount", min_value=0)
    loan_term = st.number_input("Loan Term (in days)", min_value=0)
    residential_assets_value = st.number_input("Residential Assets Value", min_value=0)
    commercial_assets_value = st.number_input("Commercial Assets Value", min_value=0)
    luxury_assets_value = st.number_input("Luxury Assets Value", min_value=0)
    bank_asset_value = st.number_input("Bank Asset Value", min_value=0)
    no_of_dependents = st.number_input("Number of Dependents", min_value=0, step=1)

    submit_button = st.form_submit_button("Predict Loan Approval")

if submit_button:
    # Validate the inputs (basic checks)
    if loan_amount > income_annum:
        st.warning("Loan amount should not exceed annual income.")
    else:
        # Prepare the data for prediction
        input_data = pd.DataFrame({
            "education": [1 if education == "Graduate" else 0],
            "self_employed": [1 if self_employed == "Yes" else 0],
            "cibil_score": [cibil_score],
            "income_annum": [income_annum],
            "loan_amount": [loan_amount],
            "loan_term": [loan_term],
            "residential_assets_value": [residential_assets_value],
            "commercial_assets_value": [commercial_assets_value],
            "luxury_assets_value": [luxury_assets_value],
            "bank_asset_value": [bank_asset_value],
            "no_of_dependents": [no_of_dependents]
        })

        # Check column names match exactly with the training data
        expected_columns = [
            "no_of_dependents",
            "education",
            "self_employed",
            "income_annum",
            "loan_amount",
            "loan_term",
            "cibil_score",
            "residential_assets_value",
            "commercial_assets_value",
            "luxury_assets_value",
            "bank_asset_value",
        ]

        # Ensure input data matches expected column order
        input_data = input_data[expected_columns]

        # Predict with the trained model
        prediction = model.predict(input_data)
        result = "✅ Approved" if prediction[0] == 1 else "❌ Rejected"

        # Display the result
        st.markdown(f"<h3 style='text-align: center; color: #2e6c80;'>Loan Status: {result}</h3>", unsafe_allow_html=True)
