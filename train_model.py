import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Load trained model
# -----------------------------
model = joblib.load("model.pkl")

# -----------------------------
# Load dataset to get feature names
# -----------------------------
data = pd.read_csv("train.csv")

# Drop columns same as training
data = data.drop(columns=['timestamp', 'turbine_id'], errors='ignore')

# Separate features
X = data.drop("Target", axis=1)

feature_names = X.columns

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Wind Turbine Power Output Prediction")

st.write("Enter turbine parameters below:")

# -----------------------------
# User Inputs
# -----------------------------
user_inputs = []

for feature in feature_names:
    value = st.number_input(f"Enter {feature}", value=0.0)
    user_inputs.append(value)

# Convert to numpy array
input_data = np.array(user_inputs).reshape(1, -1)

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("Predict Power Output"):

    prediction = model.predict(input_data)

    st.success(f"Predicted Power Output: {prediction[0]}")