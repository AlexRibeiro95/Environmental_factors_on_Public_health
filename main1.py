import streamlit as st
from backend import prepare_input, make_prediction, original_data

st.title("Life Expectancy Prediction")

st.write("Select your city and enter your health information:")

# Extract unique city names from the dataset
cities = original_data['city'].unique()

# User inputs
city_name = st.selectbox("Select your city:", cities)
obesity = st.radio("Are you obese?", ("YES", "NO"))
smoker = st.radio("Are you a smoker?", ("YES", "NO"))
copd = st.radio("Do you have COPD?", ("YES", "NO"))
depression = st.radio("Do you have depression?", ("YES", "NO"))

# When the user clicks the button
if st.button("Predict Life Expectancy"):
    try:
        # Prepare the input data
        input_data = prepare_input(city_name, obesity, smoker, copd, depression)
        
        # Make the prediction
        prediction = make_prediction(input_data)
        
        st.write(f"Predicted Life Expectancy: {prediction:.2f} years")
    except IndexError:
        st.error("City data not found. Please make sure the city is correctly selected.")