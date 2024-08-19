import streamlit as st
from backend import prepare_input, make_prediction, original_data

# 1. Collect User Inputs in main.py
city_name = st.selectbox("Select your city", original_data['city'].unique())

# Use radio buttons instead of dropdowns
smoker = st.radio("Are you a smoker?", ["YES", "NO"])
copd = st.radio("Do you have COPD?", ["YES", "NO"])
obesity = st.radio("Are you obese?", ["YES", "NO"])
depression = st.radio("Do you have depression?", ["YES", "NO"])

# 2. Prepare Input Data and Make Prediction
input_data = prepare_input(city_name, obesity, smoker, copd, depression)
prediction = make_prediction(input_data, smoker, copd, obesity, depression)

# 3. Display Prediction Result
st.write(f"Predicted life expectancy: {prediction:.2f} years")