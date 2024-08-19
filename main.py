import streamlit as st
import numpy as np
import pickle
import joblib

# Load the scaler
scaler = joblib.load('scaler.pkl')

# Load the stacked model
with open('stacked_model.pkl', 'rb') as file:
    model_data = pickle.load(file)

# Extract base models and meta-model
base_models = model_data['base_models']
meta_model = model_data['meta_model']

def main():
    # Example user inputs
    city_population = st.selectbox("City Population", ["Small (1,000 - 50,000)", "Medium (50,001 - 300,000)", "Large (>300,000)"])
    green_space = st.selectbox("Green Space Availability", ["Low (<20 km²)", "Medium (20-100 km²)", "High (>100 km²)"])
    aqi = st.selectbox("Air Quality Index (AQI)", ["Excellent (1-50)", "Good (51-100)", "Moderate (101-150)", "Poor (151-200)", "Very Poor (>200)"])
    obesity = st.radio("Obesity (BMI > 30)", ["Yes", "No"])
    smoking = st.radio("Smoking", ["Yes", "No"])
    copd = st.radio("COPD", ["Yes", "No"])
    depression = st.radio("Depression", ["Yes", "No"])

    # Map inputs to numerical values based on your dataset's structure
    input_data = np.array([
        [convert_population(city_population), convert_green_space(green_space), convert_aqi(aqi), 
         convert_boolean(obesity), convert_boolean(smoking), convert_boolean(copd), convert_boolean(depression)]
    ])

    # Apply the loaded scaler to the input data
    input_data_scaled = scaler.transform(input_data)

    # Predict using the stacked model
    base_predictions = np.zeros((input_data_scaled.shape[0], len(base_models)))

    for i, (name, model) in enumerate(base_models):
        base_predictions[:, i] = model.predict(input_data_scaled)

    final_prediction = meta_model.predict(base_predictions)

    # Display the result
    st.write(f"Predicted Life Expectancy: {final_prediction[0]:.2f} years")

# The conversion functions stay the same as before
def convert_population(population):
    if population == "Small (1,000 - 50,000)":
        return 1
    elif population == "Medium (50,001 - 300,000)":
        return 2
    else:
        return 3

def convert_green_space(green_space):
    if green_space == "Low (<20 km²)":
        return 1
    elif green_space == "Medium (20-100 km²)":
        return 2
    else:
        return 3

def convert_aqi(aqi):
    if aqi == "Excellent (1-50)":
        return 1
    elif aqi == "Good (51-100)":
        return 2
    elif aqi == "Moderate (101-150)":
        return 3
    elif aqi == "Poor (151-200)":
        return 4
    else:
        return 5

def convert_boolean(value):
    return 1 if value == "Yes" else 0

if __name__ == "__main__":
    main()