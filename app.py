import streamlit as st
import joblib
import numpy as np

model = joblib.load("model.pkl")

st.title("Wind Turbine Power Output Prediction")

st.write("Enter turbine parameters")

wind_speed = st.number_input("Wind Speed")
generator_speed = st.number_input("Generator Speed")
temperature = st.number_input("Temperature")
air_pressure = st.number_input("Air Pressure")
humidity = st.number_input("Humidity")

input_data = np.array([
    wind_speed,
    generator_speed,
    temperature,
    air_pressure,
    humidity
]).reshape(1,-1)

if st.button("Predict"):

    prediction = model.predict(input_data)

    st.success(f"Predicted Power Output: {prediction[0]}")