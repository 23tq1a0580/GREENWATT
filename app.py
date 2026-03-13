import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# -----------------------------
# Load trained model safely
# -----------------------------
model_path = "model.pkl"

if not os.path.exists(model_path):
    st.error("Model file not found. Please make sure model.pkl is in the project folder.")
    st.stop()

model = joblib.load(model_path)

# -----------------------------
# Load dataset to get feature names
# -----------------------------
data = pd.read_csv("train.csv")

# Drop same columns used during training
data = data.drop(columns=['timestamp', 'turbine_id'], errors='ignore')

# Separate features
X = data.drop("Target", axis=1)
feature_names = X.columns

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🌬 Wind Turbine Power Output Prediction")

st.write("Enter turbine parameters below:")

# -----------------------------
# User Inputs
# -----------------------------
input_data = {}

for feature in feature_names:
    input_data[feature] = st.number_input(f"Enter {feature}", value=0.0)

# Convert to dataframe
input_df = pd.DataFrame([input_data])

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("Predict Power Output"):

    prediction = model.predict(input_df)

    st.success(f"⚡ Predicted Power Output: {prediction[0]:.2f}")

    prediction = model.predict(input_data)


    st.success(f"Predicted Power Output: {prediction[0]}")
