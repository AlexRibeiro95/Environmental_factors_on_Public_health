import streamlit as st
from backend import prepare_input, make_prediction, original_data
import requests
import os
import openai
from dotenv import load_dotenv

# Load environment variables from .env file

load_dotenv()

# Load your API key securely

api_key = os.getenv("GPT_API_KEY")
openai.api_key = api_key

def get_suggestions_from_openai(smoker, copd, obesity, depression, max_tokens=200):
    prompt = f"Based on the user's health data, generate health suggestions: smoker: {smoker}, copd: {copd}, obesity: {obesity}, depression: {depression}."
    
    try:
        # Initial request with a buffer for completing sentences
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides health suggestions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=max_tokens  # Provide extra tokens as a buffer
        )
        
        # Parse the response
        suggestions = response['choices'][0]['message']['content'].strip()
        
        # Check if the suggestions end with a complete sentence
        if not suggestions.endswith('.'):
            # Follow-up request to complete the sentence
            follow_up_prompt = "Please complete the previous suggestions."
            follow_up_response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": follow_up_prompt}
                ],
                temperature=0.7,
                max_tokens=300  # Smaller token count for the follow-up
            )
            follow_up_suggestions = follow_up_response['choices'][0]['message']['content'].strip()
            suggestions += " " + follow_up_suggestions
        
        return suggestions
    
    except openai.error.OpenAIError as e:
        # Handle any errors from the API
        st.error(f"Error fetching suggestions: {e}")
        return "Failed to retrieve suggestions. Please try again later."

# Set the page layout to wide
st.set_page_config(layout="wide")

# Apply custom CSS for background color
st.markdown(
    """
    <style>
    /* Set the background color for the entire page */
    .stApp {
        background-color: #7BAA89; /* Light grey color, replace with your desired color */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Custom CSS to change the sidebar background color to match the logo's background
st.markdown(
    """
    <style>
    /* Change the background color of the sidebar */
    [data-testid="stSidebar"] {
        background-color: #657781; /* The color code you provided */
    }
    /* Optional: Change the text color in the sidebar */
    [data-testid="stSidebar"] .css-1d391kg {
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Display the logo in the sidebar
st.sidebar.image("/Users/alexandreribeiro/Documents/GitHub/final_project/visualizations/logo.png", use_column_width=True)

# Continue with the rest of your sidebar content
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Project Overview", "Data Collection", "Machine Learning", "Calculator"])

# Page 1: Project Overview
if page == "Project Overview":
    st.title("Project Overview")
    st.write("""
    Welcome to the Life Expectancy Prediction Project. This tool aims to predict life expectancy based on various health metrics such as smoking status, obesity, and more. 
    Below you'll find different sections that explain how the project works, the data collected, and the machine learning models used.
    """)

# Page 2: Data Collection
elif page == "Data Collection":
    st.title("Data Collection")
    st.write("""
    The data used in this project is sourced from various public health databases and city-specific information. 
    The dataset includes information on air quality, green space area, population size, and health metrics such as obesity and smoking rates.
    """)

# Page 3: Machine Learning
elif page == "Machine Learning":
    st.title("Machine Learning")
    st.write("""
    The life expectancy prediction is based on a stacked machine learning model combining Random Forest, XGBoost, and Support Vector Regressor. 
    Each model is trained on the features from the dataset, and the final prediction is made using a meta-model.
    """)

# Page 4: Calculator
elif page == "Calculator":
    st.title("Life Expectancy Calculator")

    # Introduction to the Calculator
    st.write("""
    Welcome to the Life Expectancy Calculator. This tool helps estimate your life expectancy based on various health factors such as smoking status, obesity, COPD, and depression.
    Enter your details below to receive an estimate along with tailored health suggestions.
    """)
    # Disclaimer
    st.write("_Disclaimer: This tool is for informational purposes only and does not constitute medical advice. Always consult with a healthcare professional for personalized medical advice._")
    
    # Collect user inputs for health conditions
    city_name = st.selectbox("Select your city", original_data['city'].unique())
    smoker = st.radio("Are you a smoker?", ["YES", "NO"])
    copd = st.radio("Do you have COPD?", ["YES", "NO"])
    obesity = st.radio("Are you obese?", ["YES", "NO"])
    depression = st.radio("Do you have depression?", ["YES", "NO"])
    
    # Button to calculate life expectancy
    if st.button("Calculate Life Expectancy"):
        # Prepare input and make predictions
        input_data = prepare_input(city_name, obesity, smoker, copd, depression)
        prediction = make_prediction(input_data, smoker, copd, obesity, depression)
        
        # Display the prediction
        st.write(f"Predicted life expectancy: {prediction:.2f} years")
        
        # Fetch and display suggestions based on the prediction
        suggestions = get_suggestions_from_openai(smoker, copd, obesity, depression)
        st.write("### Health Suggestions")
        st.write(suggestions)
    
    