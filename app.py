import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ----------------------------------
# Page Configuration
# ----------------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="centered"
)

# ----------------------------------
# Load Model and Scaler
# ----------------------------------
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

# ----------------------------------
# App Title
# ----------------------------------
st.title("💳 Credit Card Fraud Detection")
st.write("This app predicts whether a transaction is **Fraudulent** or **Legitimate**.")

# ----------------------------------
# Sidebar Inputs
# ----------------------------------
st.sidebar.header("Transaction Details")

V1 = st.sidebar.number_input("V1", value=0.0)
V2 = st.sidebar.number_input("V2", value=0.0)
V3 = st.sidebar.number_input("V3", value=0.0)
V4 = st.sidebar.number_input("V4", value=0.0)
V5 = st.sidebar.number_input("V5", value=0.0)
Amount = st.sidebar.number_input("Transaction Amount", value=0.0)

# ----------------------------------
# Create Full Feature Set (V1–V28 + Amount)
# ----------------------------------
FEATURES = [
    'V1','V2','V3','V4','V5','V6','V7','V8','V9',
    'V10','V11','V12','V13','V14','V15','V16','V17','V18','V19',
    'V20','V21','V22','V23','V24','V25','V26','V27','V28','Amount'
]

# Initialize all features with 0
input_data = dict.fromkeys(FEATURES, 0.0)

# Assign user-entered values
input_data['V1'] = V1
input_data['V2'] = V2
input_data['V3'] = V3
input_data['V4'] = V4
input_data['V5'] = V5
input_data['Amount'] = Amount

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# ----------------------------------
# Display Input
# ----------------------------------
st.subheader("Entered Transaction Data")
st.dataframe(input_df)

# ----------------------------------
# Prediction
# ----------------------------------
if st.button("Predict"):

    scaled_data = scaler.transform(input_df)
    prediction = model.predict(scaled_data)
    probability = model.predict_proba(scaled_data)

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("🚨 Fraudulent Transaction Detected!")
    else:
        st.success("✅ Legitimate Transaction")

    st.write("Fraud Probability:")
    st.write(f"{probability[0][1] * 100:.2f}%")

