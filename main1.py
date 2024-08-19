import streamlit as st
import pandas as pd
from backend import load_city_data, load_models, preprocess_input, make_prediction

# Load data and models
city_data = load_city_data()
base_models, meta_model, scaler = load_models()

st.title("Life Expectancy Predictor")

# City selection
city = st.selectbox("Select a city", city_data['City'].tolist())

# Health metrics inputs
obesity = st.radio("Obesity (BMI > 30)", ["No", "Yes"])
smoking = st.radio("Smoking", ["No", "Yes"])
copd = st.radio("COPD", ["No", "Yes"])
depression = st.radio("Depression", ["No", "Yes"])

if st.button("Predict Life Expectancy"):
    # Get city metrics
    city_metrics = city_data[city_data['City'] == city].iloc[0]
    
    # Combine city metrics with health inputs
    user_inputs = {
        'obesity': obesity,
        'smoking': smoking,
        'copd': copd,
        'depression': depression
    }
    
    # Preprocess input
    preprocessed_input = preprocess_input(city_metrics, user_inputs, scaler)
    
    # Make prediction
    prediction = make_prediction(preprocessed_input, (base_models, meta_model))
    
    st.subheader(f"Predicted Life Expectancy: {prediction:.2f} years")

# Display city metrics for transparency
st.write("City Metrics:")
st.write(city_metrics[['population_city', 'greenspacearea_km2', 'AQI']])