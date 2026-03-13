import streamlit as st
import numpy as np
import joblib

# -----------------------------
# Load trained model
# -----------------------------
model = joblib.load("model.pkl")

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🌬 Wind Turbine Power Output Prediction")

st.write("Enter turbine sensor values:")

# -----------------------------
# User Inputs
# -----------------------------
wind_speed = st.number_input("Wind Speed", value=0.0)
wind_direction = st.number_input("Wind Direction", value=0.0)
temperature = st.number_input("Temperature", value=0.0)
pressure = st.number_input("Pressure", value=0.0)
humidity = st.number_input("Humidity", value=0.0)

# Convert inputs to array
input_data = np.array([[wind_speed, wind_direction, temperature, pressure, humidity]])

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("Predict Power Output"):
    
    prediction = model.predict(input_data)

    st.success(f"⚡ Predicted Power Output: {prediction[0]}")

